#include "Mixed.h"

namespace Applications
{
  using namespace dealii;

  template <int dim>
  Mixed<dim>::Mixed()
      : ParameterAcceptor("Mixed"), 
        mpi_communicator(MPI_COMM_WORLD), 
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        triangulation_1(
            mpi_communicator,
            typename Triangulation<dim>::MeshSmoothing(
                Triangulation<dim>::smoothing_on_refinement |
                Triangulation<dim>::smoothing_on_coarsening)),
        triangulation_2(
            mpi_communicator,
            typename Triangulation<dim>::MeshSmoothing(
                Triangulation<dim>::smoothing_on_refinement |
                Triangulation<dim>::smoothing_on_coarsening)),
        dof_handler_1(triangulation_1), 
        dof_handler_2(triangulation_2), 
        fe(FE_Q<dim>(1), 3, FE_DGQ<dim>(0), 1), 
        mapping(2), 
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
      parallel::distributed::Triangulation<dim> &tria,
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
    GridGenerator::generate_from_name_and_arguments(
        tria,
        grid_generator_function,
        grid_generator_arguments);
    tria.refine_global(n_refinements);
    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
    stokes_sub_blocks[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, stokes_sub_blocks);

    auto dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, stokes_sub_blocks);

    const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];

    pcout << "Number of active cells: "
          << tria.n_active_cells()
          << std::endl
          << "Total number of cells: " << tria.n_cells()
          << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << " (" << n_u << '+' << n_p << ')' << std::endl;

    // Save mesh in MSH format
    GridOutFlags::Msh msh_flags(true, true);
    std::string msh_filename = "output/data_mesh/" + name + ".msh";
    std::ofstream msh_out(msh_filename);
    GridOut grid_out;
    grid_out.set_flags(msh_flags);
    grid_out.write_msh(tria, msh_out);
    pcout << "Saved " << msh_filename << std::endl;

    // Save mesh in VTK format
    std::string vtk_filename = "output/data_mesh/" + name + ".vtk";
    std::ofstream vtk_out(vtk_filename);
    grid_out.write_vtk(tria, vtk_out);
    pcout << "Saved " << vtk_filename << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);

    block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (c == dim && d == dim)
          coupling[c][d] = DoFTools::none;
        else if (c == dim || d == dim || c == d)
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    constraints.reinit(locally_relevant_dofs);
    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    auto locally_owned_dofs_per_processor =
        Utilities::MPI::all_gather(mpi_communicator,
                                   locally_owned_dofs);

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

    {
      preconditioner_matrix.clear();

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == dim && d == dim)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);

      DoFTools::make_sparsity_pattern(
          dof_handler, coupling, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
          dsp,
          locally_owned_dofs_per_processor,
          mpi_communicator,
          locally_relevant_dofs);
      preconditioner_matrix.reinit(
          block_owned_dofs, dsp, mpi_communicator);
    }
  }

  template <int dim>
  void Mixed<dim>::assemble_system(
      DoFHandler<dim> &dof_handler,
      AffineConstraints<double> &constraints,
      MatrixType &mat,
      MatrixType &pmat,
      VectorType &rhs,
      FunctionParser<dim> &forcing_term)
  {
    const unsigned int n_gauss_points = std::max(
                                            static_cast<unsigned int>(std::ceil(1. * (mapping.get_degree() + 1) / 2)),
                                            dof_handler.get_fe().degree + 1) +
                                        1;

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
      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      std::vector<double> rhs_values(q_points.size());
      forcing_term.value_list(q_points, rhs_values);

      for (unsigned int q = 0; q < q_points.size(); ++q)
      {
        double perm = 1;
        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
          const double div_phi_i_u = fe_values[velocities].divergence(i, q);
          const double phi_i_p = fe_values[pressure].value(i, q);

          for (unsigned int j = 0; j < n_dofs; ++j)
          {
            const Tensor<1, dim> phi_j_u =
                fe_values[velocities].value(j, q);
            const double div_phi_j_u =
                fe_values[velocities].divergence(j, q);
            const double phi_j_p = fe_values[pressure].value(j, q);

            copy_data.cell_matrix(i, j) +=
                (phi_i_u * perm * phi_j_u //
                 - phi_i_p * div_phi_j_u  //
                 - div_phi_i_u * phi_j_p) //
                * fe_values.JxW(q);

            copy_data.cell_pmatrix(i, j) +=
                (phi_i_p * phi_j_p) //
                * fe_values.JxW(q);
          }

          copy_data.cell_rhs(i) += -phi_i_p * rhs_values[q] * fe_values.JxW(q);
        }
      }
    };

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

    ScratchData<dim> scratch_data(mapping, fe, n_gauss_points);
    CopyData copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);

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
      MatrixType &preconditioner_matrix,
      AffineConstraints<double> &constraints)
  {
    LA::MPI::PreconditionAMG prec_A;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
      data.symmetric_operator = true;
      prec_A.initialize(mat.block(0, 0), data);
    }

    LA::MPI::PreconditionAMG prec_S;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
      data.symmetric_operator = true;
      prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }

    const auto A = linear_operator<LA::MPI::Vector>(
        mat.block(0, 0));
    const auto amgA = linear_operator(A, prec_A);

    const auto S =
        linear_operator<LA::MPI::Vector>(preconditioner_matrix.block(1, 1));
    const auto amgS = linear_operator(S, prec_S);

    ReductionControl inner_solver_control(
        100, 1e-8 * rhs.l2_norm(), 1.e-2);
    SolverCG<LA::MPI::Vector> cg(inner_solver_control);

    const auto invS = inverse_operator(S, cg, amgS);

    const auto P = block_diagonal_operator<2, LA::MPI::BlockVector>(
        std::array<LinearOperator<typename LA::MPI::BlockVector::BlockType>, 2>{
            {amgA, amgS}});

    SolverControl solver_control(mat.m(), 1e-10 * rhs.l2_norm());
    SolverFGMRES<LA::MPI::BlockVector> solver(solver_control);

    constraints.set_zero(solution);
    solver.solve(mat, solution, rhs, P);
    pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;

    constraints.distribute(solution);
    locally_relevant_solution = solution;
  }

  template <int dim>
  void Mixed<dim>::output_results(
      std::string &field_name,
      parallel::distributed::Triangulation<dim> &triangulation,
      DoFHandler<dim> &dof_handler,
      VectorType &locally_relevant_solution) const
  {
    std::vector<std::string> solution_names(dim, "u");
    solution_names.emplace_back("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(dim,
                       DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, mapping.get_degree(),
                         DataOut<dim>::curved_inner_cells);

    // Write VTU files to density_field directory
    std::string output_dir = "output/density_field/";
    data_out.write_vtu_with_pvtu_record(
        output_dir, field_name, 0, mpi_communicator);

    DataOutBase::DataOutFilterFlags flags(true, true);
    DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);
    
    std::string h5_filename = output_dir + "/" + field_name + ".h5";
    data_out.write_hdf5_parallel(data_filter, 
                                h5_filename,
                                mpi_communicator);
  }

  template <int dim>
  void Mixed<dim>::run_source()
  {
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
    forcing_term_1.initialize(
        "x, y, z", forcing_term_expression_1, function_constants_1);
    assemble_system(
        dof_handler_1,
        constraints_1,
        mat_1,
        preconditioner_matrix_1,
        rhs_1,
        forcing_term_1);
    solve_system(
        mat_1,
        rhs_1,
        solution_1,
        locally_relevant_solution_1,
        preconditioner_matrix_1,
        constraints_1);
    output_results(
        name,
        triangulation_1,
        dof_handler_1,
        locally_relevant_solution_1);
  }

  template <int dim>
  void Mixed<dim>::run_target()
  {
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
    forcing_term_2.initialize(
        "x, y, z", forcing_term_expression_2, function_constants_2);
    assemble_system(
        dof_handler_2,
        constraints_2,
        mat_2,
        preconditioner_matrix_2,
        rhs_2,
        forcing_term_2);
    solve_system(
        mat_2,
        rhs_2,
        solution_2,
        locally_relevant_solution_2,
        preconditioner_matrix_2,
        constraints_2);
    output_results(
        name,
        triangulation_2,
        dof_handler_2,
        locally_relevant_solution_2);
  }

  template <int dim>
  void Mixed<dim>::setup_density_fields()
  {
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

    pcout << "Density fields have been extracted and normalized." << std::endl;
  }

  template <int dim>
  void Mixed<dim>::compute_optimal_transport()
  {
    // This will be implemented once we integrate with SemiDiscreteOT
    pcout << "Computing optimal transport between density fields..." << std::endl;
  }

  template <int dim>
  void Mixed<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads() << std::endl;

    run_source();
    run_target();
    
    // Add density field processing and optimal transport
    setup_density_fields();
    compute_optimal_transport();
  }
} // namespace Applications

int main(int argc, char *argv[])
{
  try
  {
    using namespace Applications;

    deallog.depth_console(1);
    Utilities::MPI::MPI_InitFinalize
        mpi_initialization(argc, argv, 1);

    Mixed<3> pde_solver;
    ParameterAcceptor::initialize("parameters.prm");
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