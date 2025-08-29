#include "PotentialDensity.h"

namespace Applications
{
  using namespace dealii;

  PotentialDensity::PotentialDensity(const MPI_Comm &comm)
      : SemiDiscreteOT<3, 3>(comm),
        ParameterAcceptor("/Tutorial 2/PotentialDensity"),
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        tria_1(
            mpi_communicator,
            typename Triangulation<3>::MeshSmoothing(
                Triangulation<3>::smoothing_on_refinement |
                Triangulation<3>::smoothing_on_coarsening)),
        tria_2(
          mpi_communicator,
          typename Triangulation<3>::MeshSmoothing(
              Triangulation<3>::smoothing_on_refinement |
              Triangulation<3>::smoothing_on_coarsening)),
        dof_handler_1(tria_1),
        dof_handler_2(tria_2),
        fe(1),
        mapping(1)
  {
    add_parameter("number of refinements", n_refinements);
    add_parameter("number of eigenfunctions", n_evecs);
    add_parameter("number of conditioned densities", n_conditioned_densities);

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

  void PotentialDensity::setup_system()
  {
    std::string name_1 = "torus";

    const double centerline_radius = 2.0;
    const double inner_radius = 0.5;
    GridGenerator::torus(tria_1, centerline_radius, inner_radius);
    tria_1.refine_global(n_refinements);

    dof_handler_1.distribute_dofs(fe);

    pcout << "Number of active cells: "
          << tria_1.n_active_cells()
          << std::endl
          << "Total number of cells: " << tria_1.n_cells()
          << std::endl
          << "Number of degrees of freedom: " << dof_handler_1.n_dofs() <<  std::endl;

    // Save mesh in MSH format
    GridOutFlags::Msh msh_flags(true, true);
    std::string msh_filename = "output/data_mesh/" + name_1 + ".msh";
    std::ofstream msh_out(msh_filename);
    GridOut grid_out;
    grid_out.set_flags(msh_flags);
    grid_out.write_msh(tria_1, msh_out);
    pcout << "Saved " << msh_filename << std::endl;

    // Save mesh in VTK format
    std::string vtk_filename = "output/data_mesh/" + name_1 + ".vtk";
    std::ofstream vtk_out(vtk_filename);
    grid_out.write_vtk(tria_1, vtk_out);
    pcout << "Saved " << vtk_filename << std::endl;

    locally_owned_dofs = dof_handler_1.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler_1);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler_1, constraints);
    constraints.close();

    auto locally_owned_dofs_per_processor =
        Utilities::MPI::all_gather(
          mpi_communicator, locally_owned_dofs);

    // Initialize matrices and vectors.
    solution.reinit(locally_owned_dofs, mpi_communicator);
    locally_relevant_solution.reinit(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_flux_sparsity_pattern(dof_handler_1,
                                          dsp,
                                          constraints,
                                          false);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                            locally_owned_dofs,
                                            mpi_communicator,
                                            locally_relevant_dofs);

    stiffness_matrix.reinit(
      locally_owned_dofs,
      locally_owned_dofs,
      dsp,
      mpi_communicator);

    eigenfunctions.resize(n_evecs);
    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
      eigenfunctions[i].reinit(locally_owned_dofs, mpi_communicator);
    eigenvalues.resize(eigenfunctions.size());

    std::string name_2 = "simplex";
    std::vector<Point<3>> vertices = {
      Point<3>(2.5, -2, 0),
      Point<3>(-2.5*0.5, -2, 2.5*std::sqrt(3)/2),
      Point<3>(0, 2, 0),
      Point<3>(-2.5*0.5, -2, -2.5*std::sqrt(3)/2)
    };
    GridGenerator::simplex(tria_2, vertices);
    tria_2.refine_global(n_refinements);
    dof_handler_2.distribute_dofs(fe);

    // Save mesh in MSH format
    std::string msh_filename_2 = "output/data_mesh/" + name_2 + ".msh";
    std::ofstream msh_out_2(msh_filename_2);
    GridOut grid_out_2;
    grid_out_2.set_flags(msh_flags);
    grid_out_2.write_msh(tria_2, msh_out_2);
    pcout << "Saved " << msh_filename_2 << std::endl;

    // Save mesh in VTK format
    std::string vtk_filename_2 = "output/data_mesh/" + name_2 + ".vtk";
    std::ofstream vtk_out_2(vtk_filename_2);
    grid_out_2.write_vtk(tria_2, vtk_out_2);
    pcout << "Saved " << vtk_filename_2 << std::endl;
  }

  void PotentialDensity::assemble_system()
  {
    using Iterator = typename DoFHandler<3>::active_cell_iterator;

    auto cell_worker = [&](const Iterator &  cell,
                           ScratchData &scratch_data,
                           CopyData &        copy_data) {
      copy_data.cell_matrix = 0;

      FEValues<3> &fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      cell->get_dof_indices(copy_data.local_dof_indices);

      const unsigned int dofs_per_cell =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();

      for (unsigned int qpoint = 0; qpoint < fe_values.n_quadrature_points;
           ++qpoint)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // const Tensor<3> &hessian_i =
              //   fe_values.shape_hessian(i, qpoint);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // const Tensor<3> &hessian_j =
                  //   fe_values.shape_hessian(j, qpoint);

                    copy_data.cell_matrix(i, j) +=
                    scalar_product(
                      scratch_data.fe_values.shape_grad(i, qpoint),
                      scratch_data.fe_values.shape_grad(j, qpoint)) *
                    fe_values.JxW(qpoint);
                }
            }
        }
    };

    auto face_worker = [&](const Iterator &    cell,
                           const unsigned int &f,
                           const unsigned int &sf,
                           const Iterator &    ncell,
                           const unsigned int &nf,
                           const unsigned int &nsf,
                           ScratchData &  scratch_data,
                           CopyData &          copy_data) {};

    auto boundary_worker = [&](const Iterator &    cell,
                               const unsigned int &face_no,
                               ScratchData &  scratch_data,
                               CopyData &          copy_data) {};

    auto copier = [&](const CopyData &copy_data) {
      constraints.distribute_local_to_global(copy_data.cell_matrix,
                                             copy_data.local_dof_indices,
                                             stiffness_matrix);

      for (auto &cdf : copy_data.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 stiffness_matrix);
        }
    };

    const unsigned int n_gauss_points = std::max(
      static_cast<unsigned int>(std::ceil(1. * (mapping.get_degree() + 1) / 2)),
      dof_handler_1.get_fe().degree + 1) + 1;

    ScratchData   scratch_data(mapping,
                                  fe,
                                  n_gauss_points,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values | update_normal_vectors);
    CopyData copy_data(dof_handler_1.get_fe().n_dofs_per_cell());
    MeshWorker::mesh_loop(
      dof_handler_1.begin_active(),
      dof_handler_1.end(),
      cell_worker,
      copier,
      scratch_data,
      copy_data,
      MeshWorker::assemble_own_cells |
          MeshWorker::assemble_boundary_faces |
          MeshWorker::assemble_own_interior_faces_once |
          MeshWorker::assemble_ghost_faces_once,
      boundary_worker,
      face_worker);

    stiffness_matrix.compress(VectorOperation::add);
  }

  unsigned int PotentialDensity::solve()
  {
    pcout << "   Is symmetric? " << stiffness_matrix.is_symmetric(1e-9) << std::endl;

    SolverControl solver_control(dof_handler_1.n_dofs(), 1e-8);
    SLEPcWrappers::SolverKrylovSchur eigensolver(
      solver_control, mpi_communicator);

    eigensolver.set_which_eigenpairs(EPS_SMALLEST_REAL);
    eigensolver.set_problem_type(EPS_HEP);

    eigensolver.solve(stiffness_matrix,
                      eigenvalues,
                      eigenfunctions,
                      eigenfunctions.size());

    pcout << "Eigen problem solved\n";
    return solver_control.last_step();
  }

  void PotentialDensity::output_results() const
  {
    std::string field_name = "scalar_field";
    std::vector<std::string> solution_names(1, "scalar_field");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(1, DataComponentInterpretation::component_is_scalar);

    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler_1);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<3>::type_dof_data,
                             interpretation);
    Vector<float> subdomain(tria_1.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_1.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, mapping.get_degree(),
                         DataOut<3>::curved_inner_cells);

    // Write VTU files to density_field directory
    data_out.write_vtu_with_pvtu_record(
        output_dir, field_name, 0, mpi_communicator);

    DataOutBase::DataOutFilterFlags flags(true, true);
    DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);

    std::string h5_filename = output_dir + "/" + field_name + ".h5";
    data_out.write_hdf5_parallel(data_filter,
                                h5_filename,
                                mpi_communicator);

    std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
      data_filter, h5_filename, 0.0, mpi_communicator)});
    std::string xdmf_filename = output_dir + "/" + field_name + ".xdmf";
    data_out.write_xdmf_file(
      xdmf_entries,
      xdmf_filename,
      mpi_communicator);
  }

  void
  PotentialDensity::output_eigenfunctions() const
  {
    DataOut<3> data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(1, DataComponentInterpretation::component_is_scalar);

    data_out.attach_dof_handler(dof_handler_1);

    Vector<float> subdomain(tria_1.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_1.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
    {
      pcout << "   Writing eigenfunctions " << i << std::endl;
      data_out.add_data_vector(eigenfunctions[i],
                                std::string("eigenfunction_") +
                                  Utilities::int_to_string(i),
                                  DataOut<3>::type_dof_data, interpretation);
    }

    data_out.build_patches(mapping, mapping.get_degree() + 1,
                           DataOut<3>::curved_inner_cells);

    const std::string filename = "eigenvectors";
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, 0, mpi_communicator);
  }

  void
  PotentialDensity::output_normalized_source(LinearAlgebra::distributed::Vector<double, MemorySpace::Host> &source) const
  {
    DataOut<3> data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(1, DataComponentInterpretation::component_is_scalar);

    data_out.attach_dof_handler(dof_handler_1);

    Vector<float> subdomain(tria_1.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria_1.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(source,
                              std::string("source"),
                                DataOut<3>::type_dof_data, interpretation);

    data_out.build_patches(mapping, 1, DataOut<3>::no_curved_cells);

    const std::string filename = "source_density";
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, 0, mpi_communicator);
  }

  void PotentialDensity::output_conditioned_densities(
    std::vector<LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> &conditioned_densities) const
{
    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler_1);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(1, DataComponentInterpretation::component_is_scalar);

    Vector<float> subdomain(tria_1.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = tria_1.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");


    pcout << "   Writing " << conditioned_densities.size()
          << " conditioned densities to output file..." << std::endl;

    for (unsigned int i = 0; i < conditioned_densities.size(); ++i)
    {
      data_out.add_data_vector(
          conditioned_densities[i],
          std::string("conditioned_density_") + Utilities::int_to_string(i),
          DataOut<3>::type_dof_data,
          interpretation);
    }

    // --- Build patches and write output files ---
    data_out.build_patches(mapping, fe.degree);

    const std::string filename_base = "conditioned_densities";
    const std::string output_path = "output/density_field/";
    data_out.write_vtu_with_pvtu_record(output_path,
                                         filename_base,
                                         0,
                                         mpi_communicator);
    pcout << "   Saved conditioned densities to " << output_path << filename_base << "_0.pvtu" << std::endl;
}


void PotentialDensity::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads() << std::endl;

    this->param_manager.print_parameters();

    // get continuous source density field as eigenfunction of the Laplacian operator
    setup_system();
    assemble_system();

    const unsigned int n_iterations = solve();
    pcout << "   Solver converged in " << n_iterations << " iterations."
          << std::endl;
    output_eigenfunctions();

    // manually set up the source density
    pcout << "   Set source density as last eigenfunction." << std::endl;
    this->source_density.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> tmp_source_density;
    tmp_source_density.reinit(locally_owned_dofs, mpi_communicator);
    for (auto idx : tmp_source_density.locally_owned_elements())
    {
        tmp_source_density[idx] = std::exp(eigenfunctions[n_evecs-1][idx]);
    }
    tmp_source_density.compress(dealii::VectorOperation::insert);
    this->source_density = tmp_source_density;

    // normalize source
    auto quadrature = Utils::create_quadrature_for_mesh<3>(tria_1, solver_params.quadrature_order);
    // Calculate L1 norm
    double local_l1_norm = 0.0;
    FEValues<3> fe_values(mapping, fe, *quadrature,
                           update_values | update_JxW_values);
    std::vector<double> density_values(quadrature->size());

    for (const auto &cell : dof_handler_1.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(this->source_density, density_values);

        for (unsigned int q = 0; q < quadrature->size(); ++q) {
            local_l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
        }
    }

    double global_l1_norm = Utilities::MPI::sum(local_l1_norm, mpi_communicator);
    pcout << "Density L1 norm before normalization: " << global_l1_norm << std::endl;

    tmp_source_density /= global_l1_norm;
    this->source_density = tmp_source_density;
    this->sot_solver->setup_source(
      dof_handler_1,
      mapping,
      fe,
      this->source_density,
      this->solver_params.quadrature_order
    );
    output_normalized_source(this->source_density);

    std::map<types::global_dof_index, Point<3>> support_points;
    DoFTools::map_dofs_to_support_points(
      mapping, dof_handler_2, support_points);

    auto all_support_points = Utilities::MPI::all_gather(
      mpi_communicator, support_points);

    std::map<types::global_dof_index, Point<3>> global_support_points;
    for (const auto& proc_support_points : all_support_points) {
      for (const auto& [dof_idx, point] : proc_support_points) {
        global_support_points[dof_idx] = point;
      }
    }

    this->target_points.clear();
    this->target_points.reserve(global_support_points.size());
    for (const auto& [dof_idx, point] : global_support_points) {
      this->target_points.emplace_back(point);
    }
    pcout << "Number of support points: " << this->target_points.size() << std::endl;

    this->target_density.reinit(this->target_points.size());
    this->target_density = 1.0/this->target_points.size();
    this->sot_solver->setup_target(
      this->target_points, this->target_density);

    // Save target points as PLY file
    {
      std::string ply_filename = "output/data_mesh/target_points.ply";
      std::ofstream ply_out;
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      ply_out.open(ply_filename);

      // Header
      ply_out << "ply\n";
      ply_out << "format ascii 1.0\n";
      ply_out << "element vertex " << this->target_points.size() << "\n";
      ply_out << "property float x\n";
      ply_out << "property float y\n";
      ply_out << "property float z\n";
      ply_out << "property float density\n";  // Add density property
      ply_out << "end_header\n";

      // Data
      for (unsigned int i = 0; i < this->target_points.size(); ++i) {
        const auto& point = this->target_points[i];
        ply_out << point[0] << " " << point[1] << " " << point[2] << " "
           << this->target_density[i] << "\n";  // Include density value
      }

      ply_out.close();
      pcout << "Saved target points with density to " << ply_filename << std::endl;
      }
    }

    // set distanace
    this->sot_solver->set_distance_function("euclidean");

    // run regularized semi-discrete OT
    Vector<double> potential;
    potential.reinit(this->target_points.size());
    SotParameterManager::SolverParameters& solver_config = this->solver_params;
    this->sot_solver->solve(potential, solver_config);
    this->save_results(potential, "potentials");

    std::vector<unsigned int> potential_indices;
    unsigned int N = this->target_points.size() / n_conditioned_densities;
    for (unsigned int i = 0; i < n_conditioned_densities; ++i)
    {
      pcout << "   Adding potential index " << i * N << std::endl;
      potential_indices.push_back(i * N);
    }

    std::vector<LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> conditioned_densities;

    this->sot_solver->get_potential_conditioned_density(
      dof_handler_1, mapping,
      potential, potential_indices, conditioned_densities);

    output_conditioned_densities(
      conditioned_densities);

  }
}

using namespace dealii;

int main(int argc, char *argv[])
{
  try
  {
    using namespace Applications;

    deallog.depth_console(2);

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;
    const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(mpi_communicator);

    // Create conditional output stream
    ConditionalOStream pcout(std::cout, this_mpi_process == 0);

    PotentialDensity conditioned_density_test(mpi_communicator);

    // Use command line argument if provided, otherwise use default
    std::string param_file = (argc > 1) ? argv[1] : "parameters.prm";

    pcout << "Using parameter file: " << param_file << std::endl;
    ParameterAcceptor::initialize(param_file);

    conditioned_density_test.run();
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
