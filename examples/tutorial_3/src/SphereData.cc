#include "SphereData.h"

namespace Applications
{
  using namespace dealii;

  SphereData::SphereData(const MPI_Comm &comm)
      : Lloyd<2, 3>(comm),
        ParameterAcceptor("/Tutorial 3/SphereData"), 
        mpi_communicator(MPI_COMM_WORLD), 
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        triangulation(
            mpi_communicator,
            typename Triangulation<2, 3>::MeshSmoothing(
                Triangulation<2, 3>::smoothing_on_refinement |
                Triangulation<2, 3>::smoothing_on_coarsening)),
        dof_handler(triangulation), 
        fe(1), 
        mapping(1)
  {
    add_parameter("number of refinements", n_refinements);
    add_parameter("number of eigenfunctions", n_evecs);

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

  void SphereData::setup_system()
  {
    std::string name = "sphere";

    GridGenerator::hyper_sphere(triangulation);
    triangulation.refine_global(n_refinements);

    dof_handler.distribute_dofs(fe);

    pcout << "Number of active cells: "
          << triangulation.n_active_cells()
          << std::endl
          << "Total number of cells: " << triangulation.n_cells()
          << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs() <<  std::endl;

    // Save mesh in MSH format
    GridOutFlags::Msh msh_flags(true, true);
    std::string msh_filename = "output/data_mesh/" + name + ".msh";
    std::ofstream msh_out(msh_filename);
    GridOut grid_out;
    grid_out.set_flags(msh_flags);
    grid_out.write_msh(triangulation, msh_out);
    pcout << "Saved " << msh_filename << std::endl;

    // Save mesh in VTK format
    std::string vtk_filename = "output/data_mesh/" + name + ".vtk";
    std::ofstream vtk_out(vtk_filename);
    grid_out.write_vtk(triangulation, vtk_out);
    pcout << "Saved " << vtk_filename << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
    
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    auto locally_owned_dofs_per_processor =
        Utilities::MPI::all_gather(
          mpi_communicator, locally_owned_dofs);

    // Initialize matrices and vectors.
    solution.reinit(locally_owned_dofs, mpi_communicator);
    locally_relevant_solution.reinit(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                          dsp,
                                          constraints,
                                          false);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                            locally_owned_dofs,
                                            mpi_communicator,
                                            locally_relevant_dofs);

    // sparsity_pattern.copy_from(dsp);
    // sparsity_pattern.print(out);

    stiffness_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);
    
    eigenfunctions.resize(n_evecs);
    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
      eigenfunctions[i].reinit(locally_owned_dofs, mpi_communicator);
    eigenvalues.resize(eigenfunctions.size());
  }

  void SphereData::assemble_system()
  {
    using Iterator = typename DoFHandler<2, 3>::active_cell_iterator;

    auto cell_worker = [&](const Iterator &  cell,
                           ScratchData &scratch_data,
                           CopyData &        copy_data) {
      copy_data.cell_matrix = 0;

      FEValues<2, 3> &fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      cell->get_dof_indices(copy_data.local_dof_indices);

      const unsigned int dofs_per_cell =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();

      for (unsigned int qpoint = 0; qpoint < fe_values.n_quadrature_points;
           ++qpoint)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // const Tensor<2, 3> &hessian_i =
              //   fe_values.shape_hessian(i, qpoint);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // const Tensor<2, 3> &hessian_j =
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
      dof_handler.get_fe().degree + 1) + 1;

    ScratchData   scratch_data(mapping,
                                  fe,
                                  n_gauss_points,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values | update_normal_vectors);
    CopyData copy_data(dof_handler.get_fe().n_dofs_per_cell());
    MeshWorker::mesh_loop(
      dof_handler.begin_active(),
      dof_handler.end(),
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

  unsigned int SphereData::solve()
  {
    pcout << "   Is symmetric? " << stiffness_matrix.is_symmetric(1e-9) << std::endl;

    SolverControl solver_control(dof_handler.n_dofs(), 1e-8);
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

  void SphereData::output_results() const
  {
    std::string field_name = "scalar_field";
    std::vector<std::string> solution_names(1, "scalar_field");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(1, DataComponentInterpretation::component_is_scalar);

    DataOut<2, 3> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<2, 3>::type_dof_data,
                             interpretation);
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, mapping.get_degree(),
                         DataOut<2, 3>::curved_inner_cells);

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
  SphereData::output_eigenfunctions() const
  {
    DataOut<2, 3> data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(1, DataComponentInterpretation::component_is_scalar);

    data_out.attach_dof_handler(dof_handler);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
    {
      pcout << "   Writing eigenfunctions " << i << std::endl;
      data_out.add_data_vector(eigenfunctions[i],
                                std::string("eigenfunction_") +
                                  Utilities::int_to_string(i),
                                  DataOut<2, 3>::type_dof_data, interpretation);
    }

    data_out.build_patches(mapping, mapping.get_degree() + 1,
                           DataOut<2, 3>::curved_inner_cells);

    const std::string filename = "eigenvectors";
    std::ofstream output(filename);
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, 0, mpi_communicator);
  }

  void
  SphereData::output_normalized_source(LinearAlgebra::distributed::Vector<double> &source) const
  {
    DataOut<2, 3> data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(1, DataComponentInterpretation::component_is_scalar);

    data_out.attach_dof_handler(dof_handler);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(source,
                              std::string("source"),
                                DataOut<2, 3>::type_dof_data, interpretation);

    data_out.build_patches(mapping, mapping.get_degree() + 1,
                           DataOut<2, 3>::curved_inner_cells);

    const std::string filename = "source_density";
    std::ofstream output(filename);
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, 0, mpi_communicator);
  }

  void SphereData::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads() << std::endl;
    
    this->param_manager_lloyd.print_parameters();

    // get continuous source density field as eigenfunction of the Laplacian operator
    setup_system();
    assemble_system();
    const unsigned int n_iterations = solve();
    pcout << "   Solver converged in " << n_iterations << " iterations."
          << std::endl;
    output_eigenfunctions();

    // manually set up the source density
    this->source_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    for (auto idx : this->source_density.locally_owned_elements())
    {
        this->source_density[idx] = std::exp(eigenfunctions[n_evecs-1][idx]);
    }
    this->source_density.compress(dealii::VectorOperation::insert);
    this->sot_solver->setup_source(
      dof_handler,
      mapping,
      fe,
      this->source_density,
      this->solver_params.quadrature_order
    );
    this->source_density.update_ghost_values();
    
    // normalize source
    auto quadrature = Utils::create_quadrature_for_mesh<2, 3>(triangulation, solver_params.quadrature_order);
    // Calculate L1 norm
    double local_l1_norm = 0.0;
    FEValues<2, 3> fe_values(mapping, fe, *quadrature,
                           update_values | update_JxW_values);
    std::vector<double> density_values(quadrature->size());

    for (const auto &cell : dof_handler.active_cell_iterators()) {
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
    
    this->source_density /= global_l1_norm;
    this->source_density.update_ghost_values();
    output_normalized_source(this->source_density);

    // manually set up the target density
    this->target_points.clear();
    for (unsigned int i = 0; i < n_evecs-1; ++i)
    {
      Point<3> point;

      // sample random target points on the sphere
      for (unsigned int d = 0; d < 3; ++d)
        point[d] = 0.57;//2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0;

      if (i==1)
      {
        point[0] *=-1;
        point[2] *=-1;
      } else if (i==2)
      {
        point[0] *= -1;
        point[1] *= -1;
      } else if (i==3)
      {
        point[1] *= -1;
        point[2] *= -1;
      }

      double norm = point.norm();
      for (unsigned int d = 0; d < 3; ++d)
        point[d] /= norm;

      this->target_points.push_back(point);
    }

    this->target_density.reinit(this->target_points.size());
    this->target_density = 1.0/this->target_points.size();
    this->sot_solver->setup_target(
      this->target_points, this->target_density);

    // set distanace
    this->sot_solver->set_distance_function("spherical");

    // run the LLoyd algorithm
    const double absolute_threshold = 1e-6;
    const unsigned int max_iterations = 100;
    this->run_lloyd(absolute_threshold, max_iterations);
  }
} // namespace Applications

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
    
    SphereData sphere_data_lloyd(mpi_communicator);
    
    // Use command line argument if provided, otherwise use default
    std::string param_file = (argc > 1) ? argv[1] : "parameters.prm";
    
    pcout << "Using parameter file: " << param_file << std::endl;
    ParameterAcceptor::initialize(param_file);
    
    sphere_data_lloyd.run();
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