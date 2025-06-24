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
        comm(MPI_COMM_WORLD), 
        pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0)),
        tria_1(comm),
        tria_2(comm),
        dof_handler_1(tria_1), 
        dof_handler_2(tria_2),
        fe(1), 
        mapping(fe)
  {
    add_parameter("number of refinements", n_refinements);
    add_parameter("first mesh filename", filename_mesh_1);
    add_parameter("second mesh filename", filename_mesh_2);
    add_parameter("learning rate", alpha);
    add_parameter("number of iterations", max_iterations);
    add_parameter("absolute threshold", absolute_threshold);

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
    std::string name_1 = "mesh_1";
    std::string name_2 = "mesh_2";

    auto partition_mesh = [](::dealii::Triangulation<2, 3> &tria,
      const MPI_Comm               &mpi_comm,
      const unsigned int /*group_size*/) {
        GridTools::partition_triangulation(
        Utilities::MPI::n_mpi_processes(mpi_comm), tria);
      };

    auto generate_mesh_lambda_1 =
      [this](::dealii::Triangulation<2, 3> &tria) {
        GridIn<2, 3> gridin;
        gridin.attach_triangulation(tria);
        std::ifstream file(filename_mesh_1);
        gridin.read_msh(file);
    };

    auto construction_data_1 = TriangulationDescription::Utilities::
    create_description_from_triangulation_in_groups<2, 3>(
      generate_mesh_lambda_1,
      partition_mesh,
      comm,
      Utilities::MPI::n_mpi_processes(comm));

    tria_1.create_triangulation(construction_data_1);

    dof_handler_1.distribute_dofs(fe);

    pcout << "Tria 1\n"
          << "  Number of active cells: "
          << tria_1.n_cells()
          << std::endl
          << "  Total number of cells: " << tria_1.n_global_active_cells()
          << std::endl
          << "  Number of degrees of freedom: " << dof_handler_1.n_dofs() <<  std::endl;
      
    auto generate_mesh_lambda_2 =
      [this](::dealii::Triangulation<2, 3> &tria) {
        GridIn<2, 3> gridin;
        gridin.attach_triangulation(tria);
        std::ifstream file(filename_mesh_2);
        gridin.read_msh(file);
    };

    auto construction_data_2 = TriangulationDescription::Utilities::
    create_description_from_triangulation_in_groups<2, 3>(
      generate_mesh_lambda_2,
      partition_mesh,
      comm,
      Utilities::MPI::n_mpi_processes(comm));

    tria_2.create_triangulation(construction_data_2);
    
    dof_handler_2.reinit(tria_2);
    dof_handler_2.distribute_dofs(fe);

    pcout << "Tria 2\n"
          << "  Number of active cells: "
          << tria_2.n_cells()
          << std::endl
          << "  Total number of cells: " << tria_2.n_global_active_cells()
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
    output, name_1, 0, comm);

    DataOut<2, 3> data_out_2;
    data_out_2.attach_dof_handler(dof_handler_2);
    Vector<float> subdomain_2(tria_2.n_active_cells());
    for (unsigned int i = 0; i < subdomain_2.size(); ++i)
      subdomain_2(i) = tria_2.locally_owned_subdomain();
    data_out_2.add_data_vector(subdomain_2, "subdomain");
    data_out_2.build_patches(mapping, mapping.get_degree(),
                         DataOut<2, 3>::curved_inner_cells);
    data_out_2.write_vtu_with_pvtu_record(
    output, name_2, 0, comm);

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
          << Utilities::MPI::n_mpi_processes(comm)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads() << std::endl;
    
    this->param_manager_wasserstein_barycenters.print_parameters();

    // get continuous source density field as eigenfunction of the Laplacian operator
    setup_system();

    // manually set up the first source density
    {
      this->sot_solvers[0]->source_density.reinit(
        locally_owned_dofs_1, locally_relevant_dofs_1, comm);
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
      pcout << "Init source 1\n";

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
        local_l1_norm, comm);
      pcout << "Density L1 norm before normalization tria_1: " << global_l1_norm << std::endl;

      this->sot_solvers[0]->source_density /= global_l1_norm;
      this->sot_solvers[0]->source_density.update_ghost_values();

      pcout << "  source measure dofs: " << this->sot_solvers[0]->source_density.size()  << std::endl;
    }

    // manually set up the second source density
    {
      this->sot_solvers[1]->source_density.reinit(
        locally_owned_dofs_2, locally_relevant_dofs_2, comm);
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
      pcout << "Init source 2\n";
  
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
        local_l1_norm, comm);
      pcout << "Density L1 norm before normalization tria_2: " << global_l1_norm << std::endl;

      this->sot_solvers[1]->source_density /= global_l1_norm;
      this->sot_solvers[1]->source_density.update_ghost_values();

      pcout << "  source measure dofs: " << this->sot_solvers[1]->source_density.size() << std::endl;
    }

    // get support points from first tria    
    unsigned int local_n_cells{0};
    for (const auto &cell : tria_1.active_cell_iterators()) {
      if (cell->is_locally_owned())
        local_n_cells +=1;
    }

    std::vector<unsigned int> n_cells_per_rank(
      Utilities::MPI::n_mpi_processes(comm));
    MPI_Allgather(&local_n_cells, 1, MPI_UNSIGNED, 
      n_cells_per_rank.data(), 1, MPI_UNSIGNED, comm);
      
    // Calculate the accumulated number of cells
    std::vector<unsigned int> left(Utilities::MPI::n_mpi_processes(comm));
    std::vector<unsigned int> right(Utilities::MPI::n_mpi_processes(comm));
    left[0] = 0;
    right[0] = n_cells_per_rank[0];
    for (unsigned int i = 1; i < Utilities::MPI::n_mpi_processes(comm); ++i) {
      left[i] = left[i-1] + n_cells_per_rank[i-1]*3;
      right[i] = right[i-1] + n_cells_per_rank[i]*3;
    }
        
    const unsigned int n_global_cells = right[Utilities::MPI::n_mpi_processes(comm)-1]/3;
    pcout << "Total number of global cells: " << n_global_cells << std::endl;
    std::vector<double> loc_sp(n_global_cells * 3, 0);
    unsigned int shift = left[Utilities::MPI::this_mpi_process(comm)];
    unsigned int cell_index = 0;

    for (const auto &cell : tria_1.active_cell_iterators())
    {
      if (cell->is_locally_owned()) {
        for (unsigned int d = 0; d < 3; ++d)
          loc_sp[shift + cell_index * 3 + d] = cell->center()[d];
        cell_index+=1;
      }
    }

    std::vector<double> global_sp(n_global_cells * 3, 0);
    MPI_Allreduce(
      loc_sp.data(), global_sp.data(),
      n_global_cells * 3, MPI_DOUBLE, MPI_SUM, comm);

    std::vector<Point<3>> support_points;
    unsigned int factor = 1;
    for (unsigned int i = 0; i < int(n_global_cells/factor); ++i) {
      support_points.emplace_back(
        global_sp[i*3*factor + 0], global_sp[i*3*factor + 1], global_sp[i*3*factor + 2]);
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
    MPI_Comm comm = MPI_COMM_WORLD;
    const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(comm);

    // Create conditional output stream
    ConditionalOStream pcout(std::cout, this_mpi_process == 0);
    
    unsigned int n_measures = 2;
    std::vector<double> weights(n_measures, 1.0 / n_measures);
    WassersteinBarycentersUniform wasserstein_barycenters_uniform(
      n_measures, weights, comm);
    
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