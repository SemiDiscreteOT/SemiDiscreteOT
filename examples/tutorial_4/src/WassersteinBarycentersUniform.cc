#include "WassersteinBarycentersUniform.h"

namespace Applications
{
  using namespace dealii;

  WassersteinBarycentersUniform::WassersteinBarycentersUniform(
    const unsigned int n_measures,
    const std::vector<double> weights,
    const MPI_Comm &comm)
      : WassersteinBarycenters<2, 3>(n_measures, weights, comm),
        ParameterAcceptor("/Tutorial 3/WassersteinBarycentersUniform"), 
        mpi_communicator(MPI_COMM_WORLD), 
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        tria_1(
            mpi_communicator,
            typename Triangulation<2, 3>::MeshSmoothing(
                Triangulation<2, 3>::smoothing_on_refinement |
                Triangulation<2, 3>::smoothing_on_coarsening)),
        tria_2(
          mpi_communicator,
          typename Triangulation<2, 3>::MeshSmoothing(
              Triangulation<2, 3>::smoothing_on_refinement |
              Triangulation<2, 3>::smoothing_on_coarsening)),
        volume(
          mpi_communicator,
          typename Triangulation<3, 3>::MeshSmoothing(
              Triangulation<3, 3>::smoothing_on_refinement |
              Triangulation<3, 3>::smoothing_on_coarsening)),
        dof_handler_1(tria_1), 
        dof_handler_2(tria_2),
        fe(1), 
        mapping(1)
  {
    add_parameter("number of refinements", n_refinements);

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

  void WassersteinBarycentersUniform::setup_system()
  {
    std::string name_1 = "sphere";
    std::string name_2 = "cylinder";


    GridGenerator::hyper_sphere(tria_1);
    tria_1.refine_global(n_refinements);

    dof_handler_1.distribute_dofs(fe);

    pcout << "Tria 1\n"
          << "  Number of active cells: "
          << tria_1.n_active_cells()
          << std::endl
          << "  Total number of cells: " << tria_1.n_cells()
          << std::endl
          << "  Number of degrees of freedom: " << dof_handler_1.n_dofs() <<  std::endl;

    const double length = 0.5;
    const double width = 0.5;
    const double height = 4.0;
    std::vector<unsigned int> repetitions = {1, 1, 4};
    GridGenerator::subdivided_hyper_rectangle(
      volume,
      repetitions,
      Point<3>(-length/2, -width/2, -height/2),
      Point<3>(length/2, width/2, height/2),
      true);
    GridGenerator::extract_boundary_mesh(volume, tria_2);
    tria_2.refine_global(n_refinements);
    dof_handler_2.reinit(tria_2);
    dof_handler_2.distribute_dofs(fe);

    pcout << "Tria 2\n"
          << "  Number of active cells: "
          << tria_2.n_active_cells()
          << std::endl
          << "  Total number of cells: " << tria_2.n_cells()
          << std::endl
          << "  Number of degrees of freedom: " << dof_handler_2.n_dofs() <<  std::endl;
    pcout << std::endl;
    
    DataOut<2, 3> data_out_1;
    data_out_1.attach_dof_handler(dof_handler_1);
    Vector<float> subdomain_1(tria_1.n_active_cells());
    for (unsigned int i = 0; i < subdomain_1.size(); ++i)
      subdomain_1(i) = tria_1.locally_owned_subdomain();
    data_out_1.add_data_vector(subdomain_1, "subdomain");
    data_out_1.build_patches(mapping, mapping.get_degree(),
                         DataOut<2, 3>::curved_inner_cells);
    std::string output = "output/data_mesh/";
    data_out_1.write_vtu_with_pvtu_record(
    output, name_1, 0, mpi_communicator);

    DataOut<2, 3> data_out_2;
    data_out_2.attach_dof_handler(dof_handler_2);
    Vector<float> subdomain_2(tria_2.n_active_cells());
    for (unsigned int i = 0; i < subdomain_2.size(); ++i)
      subdomain_2(i) = tria_2.locally_owned_subdomain();
    data_out_2.add_data_vector(subdomain_2, "subdomain");
    data_out_2.build_patches(mapping, mapping.get_degree(),
                         DataOut<2, 3>::curved_inner_cells);
    data_out_2.write_vtu_with_pvtu_record(
    output, name_2, 0, mpi_communicator);

    // init dofs
    locally_owned_dofs_1 = dof_handler_1.locally_owned_dofs();
    locally_relevant_dofs_1 =
        DoFTools::extract_locally_relevant_dofs(dof_handler_1);
    locally_owned_dofs_2 = dof_handler_2.locally_owned_dofs();
    locally_relevant_dofs_2 =
        DoFTools::extract_locally_relevant_dofs(dof_handler_2);
  }

  void WassersteinBarycentersUniform::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads() << std::endl;
    
    this->param_manager_wasserstein_barycenters.print_parameters();

    // get continuous source density field as eigenfunction of the Laplacian operator
    setup_system();

    // manually set up the first source density
    {
      this->sot_solvers[0]->source_density.reinit(
        locally_owned_dofs_1, locally_relevant_dofs_1, mpi_communicator);
      for (auto idx : this->sot_solvers[0]->source_density.locally_owned_elements())
        this->sot_solvers[0]->source_density[idx] = 1;
      this->sot_solvers[0]->source_density.compress(dealii::VectorOperation::insert);
      this->sot_solvers[0]->sot_solver->setup_source(
        dof_handler_1,
        mapping,
        fe,
        this->sot_solvers[0]->source_density,
        this->sot_solvers[0]->solver_params.quadrature_order
      );
      this->sot_solvers[0]->source_density.update_ghost_values();

      // normalize source
      auto quadrature_1 = Utils::create_quadrature_for_mesh<2, 3>(
        tria_1, this->sot_solvers[0]->solver_params.quadrature_order);

      // Calculate L1 norm
      double local_l1_norm = 0.0;
      FEValues<2, 3> fe_values_1(
        mapping, fe, *quadrature_1,
        update_values | update_JxW_values);
      std::vector<double> density_values(quadrature_1->size());

      for (const auto &cell : dof_handler_1.active_cell_iterators()) {
          if (!cell->is_locally_owned())
              continue;

          fe_values_1.reinit(cell);
          fe_values_1.get_function_values(
            this->sot_solvers[0]->source_density, density_values);

          for (unsigned int q = 0; q < quadrature_1->size(); ++q) {
              local_l1_norm += std::abs(density_values[q]) * fe_values_1.JxW(q);
          }
      }
      double global_l1_norm = Utilities::MPI::sum(
        local_l1_norm, mpi_communicator);
      pcout << "Density L1 norm before normalization tria_1: " << global_l1_norm << std::endl;

      this->sot_solvers[0]->source_density /= global_l1_norm;
      this->sot_solvers[0]->source_density.update_ghost_values();

      pcout << "  source measure dofs: " << this->sot_solvers[0]->source_density.size() << "; target measure: " << this->sot_solvers[0]->target_density.size() << std::endl;
    }

    // manually set up the second source density
    {
      this->sot_solvers[1]->source_density.reinit(
        locally_owned_dofs_2, locally_relevant_dofs_2, mpi_communicator);
      for (auto idx : this->sot_solvers[1]->source_density.locally_owned_elements())
        this->sot_solvers[1]->source_density[idx] = 1;
      this->sot_solvers[1]->source_density.compress(dealii::VectorOperation::insert);
      this->sot_solvers[1]->sot_solver->setup_source(
        dof_handler_2,
        mapping,
        fe,
        this->sot_solvers[1]->source_density,
        this->sot_solvers[1]->solver_params.quadrature_order
      );
      this->sot_solvers[1]->source_density.update_ghost_values();
  
      // normalize source
      auto quadrature_2 = Utils::create_quadrature_for_mesh<2, 3>(
        tria_2, this->sot_solvers[1]->solver_params.quadrature_order);
  
      // Calculate L1 norm
      double local_l1_norm = 0.0;
      FEValues<2, 3> fe_values_2(
        mapping, fe, *quadrature_2,
        update_values | update_JxW_values);
      std::vector<double> density_values(quadrature_2->size());
  
      for (const auto &cell : dof_handler_2.active_cell_iterators()) {
          if (!cell->is_locally_owned())
              continue;
  
          fe_values_2.reinit(cell);
          fe_values_2.get_function_values(
            this->sot_solvers[1]->source_density, density_values);
  
          for (unsigned int q = 0; q < quadrature_2->size(); ++q) {
              local_l1_norm += std::abs(density_values[q]) * fe_values_2.JxW(q);
          }
      }
      double global_l1_norm = Utilities::MPI::sum(
        local_l1_norm, mpi_communicator);
      pcout << "Density L1 norm before normalization tria_2: " << global_l1_norm << std::endl;

      this->sot_solvers[1]->source_density /= global_l1_norm;
      this->sot_solvers[1]->source_density.update_ghost_values();

      pcout << "  source measure dofs: " << this->sot_solvers[1]->source_density.size() << "; target measure: " << this->sot_solvers[1]->target_density.size() << std::endl;
    }

    // get support points from first tria
    std::vector<Point<3>> support_points;
    DoFTools::map_dofs_to_support_points(
      mapping, dof_handler_1, support_points);
    unsigned int n_support_points = support_points.size();
    std::vector<double> loc_sp(n_support_points * 3);
    for (auto idx: locally_owned_dofs_1)
      for (unsigned int d = 0; d < 3; ++d)
        loc_sp[idx * 3 + d] = support_points[idx][d];
    std::vector<double> global_sp(n_support_points * 3);
    MPI_Allreduce(
      loc_sp.data(), global_sp.data(),
      n_support_points * 3, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    support_points.clear();
    for (unsigned int i = 0; i < n_support_points; ++i) {
      support_points.emplace_back(
        global_sp[i * 3 + 0], global_sp[i * 3 + 1], global_sp[i * 3 + 2]);
    }
    pcout << "Number of support points: " << support_points.size() << std::endl;

    // Use the support points as target points
    for (unsigned int i = 0; i < n_measures; ++i) {
      this->sot_solvers[i]->target_points.clear();
      for (const auto& point : support_points) {
        this->sot_solvers[i]->target_points.push_back(point);
      }
      this->sot_solvers[i]->target_density.reinit(
        this->sot_solvers[i]->target_points.size());
      this->sot_solvers[i]->target_density = 1.0/this->sot_solvers[i]->target_points.size();

      this->sot_solvers[i]->sot_solver->setup_target(
        this->sot_solvers[i]->target_points,
        this->sot_solvers[i]->target_density);
      this->sot_solvers[i]->sot_solver->set_distance_function("euclidean");
    }

    // run the LLoyd algorithm
    const double absolute_threshold = 1e-8;
    const unsigned int max_iterations = 100;
    const double alpha = 10000; // step
    this->run_wasserstein_barycenters(
      absolute_threshold, absolute_threshold, max_iterations, alpha);
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
    
    unsigned int n_measures = 2;
    std::vector<double> weights(n_measures, 1.0 / n_measures);
    WassersteinBarycentersUniform wasserstein_barycenters_uniform(
      n_measures, weights, mpi_communicator);
    
    // Use command line argument if provided, otherwise use default
    std::string param_file = (argc > 1) ? argv[1] : "parameters.prm";
    
    pcout << "Using parameter file: " << param_file << std::endl;
    ParameterAcceptor::initialize(param_file);
    
    wasserstein_barycenters_uniform.run();
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