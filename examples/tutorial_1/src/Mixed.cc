#include "Mixed.h"

namespace Applications
{
  using namespace dealii;

  template <int dim>
  Mixed<dim>::Mixed()
      : ParameterAcceptor("Mixed"),
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        triangulation_1(mpi_communicator),
        triangulation_2(mpi_communicator),
        dof_handler_1(triangulation_1),
        dof_handler_2(triangulation_2),
        // Define the stable Taylor-Hood P2-P1 element pair for simplices
        fe(FE_SimplexP<dim>(2), dim, FE_SimplexP<dim>(1), 1),
        // Use MappingFE, matching the velocity element degree for geometry representation
        // mapping(fe.degree),
        mapping(MappingFE<dim>(FE_SimplexP<dim>(1))),
        forcing_term_1(dim),
        forcing_term_2(dim)
  {

    add_parameter("number of refinements source", n_refinements_1);
    add_parameter("grid generator arguments source", grid_generator_arguments_1);
    add_parameter("grid generator function source", grid_generator_function_1);
    add_parameter("forcing term expression source", forcing_term_expression_1);
    add_parameter("function constants source", function_constants_1);

    add_parameter("number of refinements target", n_refinements_2);
    add_parameter("grid generator arguments target", grid_generator_arguments_2);
    add_parameter("grid generator function target", grid_generator_function_2);
    add_parameter("forcing term expression target", forcing_term_expression_2);
    add_parameter("function constants target", function_constants_2);

    // Create output directory structure
    if (!std::filesystem::exists("output"))
    {
      std::filesystem::create_directory("output");
    }
    if (!std::filesystem::exists("output/data_mesh"))
    {
      std::filesystem::create_directory("output/data_mesh");
    }
    if (!std::filesystem::exists("output/density_field"))
    {
      std::filesystem::create_directory("output/density_field");
    }
  }

  template <int dim>
  void Mixed<dim>::setup_system(
      std::string &name,
      parallel::fullydistributed::Triangulation<dim> &tria,
      DoFHandler<dim> &dof_handler,
      IndexSet &locally_owned_dofs,
      IndexSet &locally_relevant_dofs,
      std::vector<IndexSet> &block_owned_dofs,
      std::vector<IndexSet> &block_relevant_dofs,
      AffineConstraints<double> &constraints,
      std::string &grid_generator_function,
      std::string &grid_generator_arguments,
      unsigned int n_refinements,
      VectorType &solution,
      VectorType &locally_relevant_solution,
      VectorType &rhs,
      MatrixType &mat,
      MatrixType &preconditioner_matrix)
  {
    // Step 1: Create serial triangulations for the process
    Triangulation<dim> temp_tria_1;
    Triangulation<dim> temp_tria_2;
    
    // Generate initial hypercube mesh
    GridGenerator::generate_from_name_and_arguments(
        temp_tria_1,
        grid_generator_function,
        grid_generator_arguments);

    // Refine the temporary triangulation
    temp_tria_1.refine_global(n_refinements);

    // Convert to simplices using a fresh triangulation
    GridGenerator::convert_hypercube_to_simplex_mesh(temp_tria_1, temp_tria_2);

    // Set all manifold ids to the same value (flat manifold)
    for (const auto &cell : temp_tria_2.active_cell_iterators())
    {
      cell->set_all_manifold_ids(numbers::flat_manifold_id);
    }

    // Step 2: Create the description for the fully distributed triangulation
    auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(
        temp_tria_2, mpi_communicator);

    // Step 3: Create the actual fully distributed triangulation
    tria.create_triangulation(construction_data);

    pcout << "Converted mesh to simplices." << std::endl;

    // Distribute DoFs using the simplex-based FE system
    dof_handler.distribute_dofs(fe);

    // Renumber DoFs component-wise (velocity components first, then pressure)
    std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0); // block 0 for velocity
    stokes_sub_blocks[dim] = 1;                             // block 1 for pressure
    DoFRenumbering::component_wise(dof_handler, stokes_sub_blocks);

    // Count DoFs per block
    auto dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, stokes_sub_blocks);
    const unsigned int n_u = dofs_per_block[0]; // Number of velocity DoFs
    const unsigned int n_p = dofs_per_block[1]; // Number of pressure DoFs

    pcout << "Number of active cells: "
          << tria.n_active_cells()
          << " (simplices)" << std::endl
          << "Total number of cells: " << tria.n_cells()
          << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << " (" << n_u << "[u] + " << n_p << "[p])" << std::endl;

    // Save mesh in VTK format (supports simplices)
    std::string vtk_filename = "output/data_mesh/" + name + ".vtk";
    std::ofstream vtk_out(vtk_filename);
    GridOut grid_out;
    grid_out.write_vtk(tria, vtk_out);
    pcout << "Saved " << vtk_filename << std::endl;

    // Setup IndexSets for parallel computations
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);

    block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    // Define sparsity pattern coupling for the main system matrix (Stokes/Darcy)
    // (u,u) block: yes
    // (u,p) block: yes
    // (p,u) block: yes
    // (p,p) block: no (standard formulation)
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (c == dim && d == dim) // (p, p) block
          coupling[c][d] = DoFTools::none;
        else // All other blocks couple velocity and pressure
          coupling[c][d] = DoFTools::always;

    // Setup constraints (e.g., hanging nodes)
    constraints.reinit(locally_relevant_dofs);
    // Note: Dirichlet boundary conditions are not applied here, assumed handled elsewhere or zero implicitly.
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    // Gather owned DoFs information for sparsity pattern distribution
    auto locally_owned_dofs_per_processor =
        Utilities::MPI::all_gather(mpi_communicator,
                                   locally_owned_dofs);

    // Create sparsity pattern for the system matrix
    const std::vector<types::global_dof_index> block_sizes = {n_u, n_p};
    BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);
    DoFTools::make_sparsity_pattern(
        dof_handler, coupling, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(
        dsp,
        locally_owned_dofs_per_processor,
        mpi_communicator,
        locally_relevant_dofs);

    // Initialize matrices and vectors.
    solution.reinit(block_owned_dofs, mpi_communicator);
    locally_relevant_solution.reinit(
        block_owned_dofs, block_relevant_dofs, mpi_communicator);
    rhs.reinit(block_owned_dofs, mpi_communicator);
    mat.reinit(block_owned_dofs, dsp, mpi_communicator);

    // Create sparsity pattern for the preconditioner matrix (only pressure block)
    {
      preconditioner_matrix.clear();

      // Only the (p, p) block is needed for the pressure mass matrix used in preconditioning
      Table<2, DoFTools::Coupling> coupling_p(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == dim && d == dim) // Only (p, p) block
            coupling_p[c][d] = DoFTools::always;
          else
            coupling_p[c][d] = DoFTools::none;

      BlockDynamicSparsityPattern dsp_p(block_sizes, block_sizes);

      DoFTools::make_sparsity_pattern(
          dof_handler, coupling_p, dsp_p, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
          dsp_p,
          locally_owned_dofs_per_processor,
          mpi_communicator,
          locally_relevant_dofs);
      preconditioner_matrix.reinit(
          block_owned_dofs, dsp_p, mpi_communicator);
    }
  }

  template <int dim>
  void Mixed<dim>::assemble_system(
      DoFHandler<dim> &dof_handler,
      AffineConstraints<double> &constraints,
      MatrixType &mat,
      MatrixType &pmat, // Preconditioner matrix (pressure mass matrix)
      VectorType &rhs,
      FunctionParser<dim> &forcing_term)
  {
    // Determine quadrature degree sufficient for simplex elements
    // P2 velocity (degree 2), P1 pressure (degree 1)
    // Integrands involve products like P2*P2 (degree 4) or div(P2)*P1 (degree 1*1=1) or P1*P1 (degree 2)
    // Need quadrature exact for degree 4 (vel mass term). Rule degree 2k -> exactness 2k+1.
    // QGaussSimplex(3) -> exact degree 3. QGaussSimplex(4) -> exact degree 5.
    // Let's use degree 4.
    const unsigned int quadrature_degree = 4; // fe.degree * 2 should be safe for mass matrix

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    const auto cell_worker = [&forcing_term, this](const Iterator &cell,
                                                   ScratchData<dim> &scratch_data,
                                                   CopyData &copy_data)
    {
      scratch_data.fe_values.reinit(cell);
      const unsigned int n_dofs =
          scratch_data.fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      const FEValues<dim> &fe_values = scratch_data.fe_values;

      const auto &q_points = fe_values.get_quadrature_points();
      const unsigned int n_q_points = q_points.size(); // Use actual number of quad points

      const FEValuesExtractors::Vector velocities(0);   // Velocity is component 0 to dim-1
      const FEValuesExtractors::Scalar pressure(dim); // Pressure is component dim

      std::vector<double> rhs_values(n_q_points);
      forcing_term.value_list(q_points, rhs_values);

      // Permeability/Viscosity term (assuming constant 1 for simplicity)
      double perm = 1.0;

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double JxW_q = fe_values.JxW(q); // Precompute JxW

        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          // Get values/divergences for basis function 'i' at point 'q'
          const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
          const double div_phi_i_u = fe_values[velocities].divergence(i, q);
          const double phi_i_p = fe_values[pressure].value(i, q);

          for (unsigned int j = 0; j < n_dofs; ++j)
          {
            // Get values/divergences for basis function 'j' at point 'q'
            const Tensor<1, dim> phi_j_u = fe_values[velocities].value(j, q);
            const double div_phi_j_u = fe_values[velocities].divergence(j, q);
            const double phi_j_p = fe_values[pressure].value(j, q);

            // Assemble system matrix contributions (Darcy/Stokes)
            copy_data.cell_matrix(i, j) +=
                (phi_i_u * perm * phi_j_u // (u, v) term
                 - phi_i_p * div_phi_j_u  // -(p, div v) term
                 - div_phi_i_u * phi_j_p) // -(q, div u) term
                * JxW_q;

            // Assemble preconditioner matrix contribution (pressure mass matrix)
            copy_data.cell_pmatrix(i, j) +=
                (phi_i_p * phi_j_p) // (p, q) term
                * JxW_q;
          }

          // Assemble RHS contribution (forcing term * pressure test function)
          // Assuming forcing term f relates to pressure equation: div(u) = f -> -(q, f) term on RHS
          copy_data.cell_rhs(i) += -phi_i_p * rhs_values[q] * JxW_q;
        }
      }
    };

    // Copier lambda: distributes local cell contributions to global matrices/vectors
    auto copier = [&](const CopyData &c)
    {
      constraints.distribute_local_to_global(
          c.cell_matrix, c.local_dof_indices, mat);

      constraints.distribute_local_to_global(
          c.cell_pmatrix, c.local_dof_indices, pmat);

      constraints.distribute_local_to_global(
          c.cell_rhs,
          c.local_dof_indices,
          rhs);
    };

    // Initialize ScratchData and CopyData
    ScratchData<dim> scratch_data(mapping, fe, quadrature_degree);
    CopyData copy_data;

    // Perform mesh loop using MeshWorker
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells); // Remove boundary face assembly flag

    // Finalize parallel assembly
    mat.compress(VectorOperation::add);
    pmat.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void Mixed<dim>::solve_system(
      MatrixType &mat,
      VectorType &rhs,
      VectorType &solution,
      VectorType &locally_relevant_solution,
      MatrixType &preconditioner_matrix, // This holds the pressure mass matrix
      AffineConstraints<double> &constraints)
  {
    // --- Preconditioner Setup ---
    // Preconditioner for the (0,0) block (velocity) - AMG for the velocity mass matrix term A=(u,u)/perm
    LA::MPI::PreconditionAMG prec_A;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
      // A should be symmetric positive definite if it's just the mass matrix term
      data.symmetric_operator = true;
      prec_A.initialize(mat.block(0, 0), data);
    }

    // Preconditioner for the Schur complement approximation S ~ M_p (pressure mass matrix)
    // Use AMG for the pressure mass matrix M_p stored in preconditioner_matrix.block(1,1)
    LA::MPI::PreconditionAMG prec_S;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
      data.symmetric_operator = true; // Mass matrix is SPD
      prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }

    // --- Create Linear Operators ---
    // Operator for the (0,0) block
    const auto A_op = linear_operator<LA::MPI::Vector>(mat.block(0, 0));
    // Preconditioned operator for (0,0) block
    const auto P_A_op = linear_operator(A_op, prec_A);

    // Operator for the pressure mass matrix (approximation to Schur complement)
    const auto S_approx_op = linear_operator<LA::MPI::Vector>(preconditioner_matrix.block(1, 1));
    // Preconditioned operator for S_approx
    const auto P_S_op = linear_operator(S_approx_op, prec_S);

    // Inner solver for applying the inverse of the pressure mass matrix preconditioner
    ReductionControl inner_solver_control(100, 1e-8 * rhs.block(1).l2_norm(), 1e-2); // Relative tolerance for inner solve
    SolverCG<LA::MPI::Vector> inner_solver(inner_solver_control);
    // Inverse operator for S_approx using the preconditioned CG
    const auto P_invS_op = inverse_operator(S_approx_op, inner_solver, P_S_op);

    // Block diagonal preconditioner P = diag(P_A, P_S) where P_A is AMG for A, P_S is AMG for M_p
    // This is often used directly with FGMRES for the block system.
    const auto P_diag = block_diagonal_operator<2, LA::MPI::BlockVector>(
        std::array<LinearOperator<typename LA::MPI::BlockVector::BlockType>, 2>{
            {P_A_op, P_S_op}});

    // Note: A more sophisticated block Schur complement preconditioner could be built,
    // involving P_invS_op, but using the block diagonal P_diag is simpler and often effective.

    // --- Solver Setup ---
    SolverControl solver_control(mat.m(), 1e-10 * rhs.l2_norm()); // Outer solver tolerance
    SolverFGMRES<LA::MPI::BlockVector> solver(solver_control);

    // --- Solve ---
    pcout << "   Solving linear system..." << std::endl;
    constraints.set_zero(solution); // Apply constraints (e.g., zero for hanging nodes before solve)
    solver.solve(mat, solution, rhs, P_diag); // Solve using FGMRES with the block diagonal preconditioner
    pcout << "   Solved in " << solver_control.last_step() << " FGMRES iterations." << std::endl;

    // Apply constraints to the solution (distribute hanging node values)
    constraints.distribute(solution);

    // Update the locally relevant solution vector which includes ghost values
    locally_relevant_solution = solution;
  }

  template <int dim>
  void Mixed<dim>::output_results(
      std::string &field_name,
      parallel::fullydistributed::Triangulation<dim> &triangulation,
      DoFHandler<dim> &dof_handler,
      VectorType &locally_relevant_solution) const
  {
    // Define names and interpretation for output variables
    std::vector<std::string> solution_names;
    for (int i=0; i<dim; ++i)
        solution_names.push_back("velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation;
    for (int i=0; i<dim; ++i)
        interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             interpretation);

    // Add subdomain information for parallel visualization
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    // Build patches using the correct mapping (MappingFE)
    data_out.build_patches(mapping, mapping.get_degree(),
                         DataOut<dim>::curved_inner_cells);

    // Write VTU files (suitable for simplices) to density_field directory
    std::string output_dir = "output/density_field/";
    data_out.write_vtu_with_pvtu_record(
        output_dir, field_name, 0, mpi_communicator);
    pcout << "Saved results to " << output_dir << field_name << ".pvtu" << std::endl;

    // Write HDF5 if needed (optional)
    DataOutBase::DataOutFilterFlags flags(true, true); // Output vertices and cells
    DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);

    std::string h5_filename = output_dir + "/" + field_name + ".h5";
    // data_out.write_hdf5_parallel(data_filter, // Uncomment if HDF5 output is desired
    //                             h5_filename,
    //                             mpi_communicator);
  }

  template <int dim>
  void Mixed<dim>::run_source()
  {
    pcout << "--- Setting up SOURCE system ---" << std::endl;
    std::string name = "source";
    setup_system(
        name,
        triangulation_1,
        dof_handler_1,
        locally_owned_dofs_1,
        locally_relevant_dofs_1,
        block_owned_dofs_1,
        block_relevant_dofs_1,
        constraints_1,
        grid_generator_function_1,
        grid_generator_arguments_1,
        n_refinements_1,
        solution_1,
        locally_relevant_solution_1,
        rhs_1,
        mat_1,
        preconditioner_matrix_1);

    pcout << "--- Assembling SOURCE system ---" << std::endl;
    forcing_term_1.initialize(
        FunctionParser<dim>::default_variable_names(), // Use default "x,y,z"
        forcing_term_expression_1,
        function_constants_1);
    assemble_system(
        dof_handler_1,
        constraints_1,
        mat_1,
        preconditioner_matrix_1,
        rhs_1,
        forcing_term_1);

    pcout << "--- Solving SOURCE system ---" << std::endl;
    solve_system(
        mat_1,
        rhs_1,
        solution_1,
        locally_relevant_solution_1,
        preconditioner_matrix_1,
        constraints_1);

    pcout << "--- Outputting SOURCE results ---" << std::endl;
    output_results(
        name,
        triangulation_1,
        dof_handler_1,
        locally_relevant_solution_1);
  }

  template <int dim>
  void Mixed<dim>::run_target()
  {
    pcout << "--- Setting up TARGET system ---" << std::endl;
    std::string name = "target";
    setup_system(
        name,
        triangulation_2,
        dof_handler_2,
        locally_owned_dofs_2,
        locally_relevant_dofs_2,
        block_owned_dofs_2,
        block_relevant_dofs_2,
        constraints_2,
        grid_generator_function_2,
        grid_generator_arguments_2,
        n_refinements_2,
        solution_2,
        locally_relevant_solution_2,
        rhs_2,
        mat_2,
        preconditioner_matrix_2);

    pcout << "--- Assembling TARGET system ---" << std::endl;
    forcing_term_2.initialize(
        FunctionParser<dim>::default_variable_names(), // Use default "x,y,z"
        forcing_term_expression_2,
        function_constants_2);
    assemble_system(
        dof_handler_2,
        constraints_2,
        mat_2,
        preconditioner_matrix_2,
        rhs_2,
        forcing_term_2);

    pcout << "--- Solving TARGET system ---" << std::endl;
    solve_system(
        mat_2,
        rhs_2,
        solution_2,
        locally_relevant_solution_2,
        preconditioner_matrix_2,
        constraints_2);

    pcout << "--- Outputting TARGET results ---" << std::endl;
    output_results(
        name,
        triangulation_2,
        dof_handler_2,
        locally_relevant_solution_2);
  }

  template <int dim>
  void Mixed<dim>::setup_density_fields()
  {
    pcout << "--- Setting up Density Fields ---" << std::endl;
    source_density_field = std::make_unique<PressureDensityField<dim>>(
        std::ref(dof_handler_1), std::ref(locally_relevant_solution_1), std::ref(mpi_communicator));
    target_density_field = std::make_unique<PressureDensityField<dim>>(
        std::ref(dof_handler_2), std::ref(locally_relevant_solution_2), std::ref(mpi_communicator));

    // Extract and normalize the pressure fields
    source_density_field->extract_pressure();
    source_density_field->normalize();
    source_density_field->output_density("output/density_field/source_density.vtk");

    target_density_field->extract_pressure();
    target_density_field->normalize();
    target_density_field->output_density("output/density_field/target_density.vtk");

    pcout << "Density fields have been extracted, normalized, and saved." << std::endl;
  }

  template <int dim>
  void Mixed<dim>::compute_optimal_transport()
  {
    // This will be implemented once we integrate with SemiDiscreteOT
    pcout << "--- Computing Optimal Transport (Placeholder) ---" << std::endl;
    // TODO: Integrate with SemiDiscreteOT library/code
    // Need to pass source_density_field->get_density() and target_density_field->get_density()
    // along with associated mesh/DoF information to the OT solver.
  }

  template <int dim>
  void Mixed<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "Using " << dealii::MultithreadInfo::n_threads() << " threads per rank." << std::endl;
    pcout << "Using Simplex Elements (P2-P1 Taylor-Hood)." << std::endl;


    run_source();
    run_target();

    // Add density field processing and optimal transport steps
    setup_density_fields();
    compute_optimal_transport();

    pcout << "--- Run Finished ---" << std::endl;
  }

// Explicit instantiation for 3D
template class Mixed<3>;

// Add 2D if needed
// template class Mixed<2>;

} // namespace Applications

int main(int argc, char *argv[])
{
  try
  {
    using namespace Applications;

    // Initialize MPI and Utilities
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1); // 1 thread per process for MPI communication

    // Set deal.II logging depth
    deallog.depth_console(0); // Reduce verbosity, 0 or 1 is usually good

    // Create the main application object (using 3D)
    Mixed<3> pde_solver;

    // Read parameters from file "parameters.prm"
    // Ensure this file exists and is configured correctly.
    try {
        ParameterAcceptor::initialize("parameters.prm");
    } catch (const std::exception &exc) {
        std::cerr << "Error initializing parameters from 'parameters.prm': " << exc.what() << std::endl;
        std::cerr << "Please ensure 'parameters.prm' exists and is readable." << std::endl;
        return 1;
    }

    // Run the simulation
    pde_solver.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
