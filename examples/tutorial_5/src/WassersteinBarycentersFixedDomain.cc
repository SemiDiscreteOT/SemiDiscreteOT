#include "WassersteinBarycentersFixedDomain.h"

namespace Applications
{
  using namespace dealii;

  class SphericalGaussian : public Function<3>
  {
  public:
    SphericalGaussian(const Point<3> &center,
                      const double sigma)
      : Function<3>(), center(center), sigma(sigma)
    {
      // Ensure center is normalized
      Assert(std::abs(center.norm() - 1.0) < 1e-10,
            ExcMessage("Center must be on the unit sphere."));
    }

    virtual double value(const Point<3> &p,
                        const unsigned int = 0) const override
    {
      const double norm_p = p.norm();
      if (norm_p == 0.0)
        return 0.0;

      // Project to the sphere if needed
      Point<3> x = p / norm_p;

      // Compute geodesic distance via arccos(dot product)
      const double dot = center * x;
      const double theta = std::acos(std::min(1.0, std::max(-1.0, dot)));

      return std::exp(- (theta * theta) / (2.0 * sigma * sigma));
    }

  private:
    const Point<3> center;
    const double sigma;
  };

  WassersteinBarycentersFixedDomain::WassersteinBarycentersFixedDomain(
    const unsigned int n_measures,
    const std::vector<double> weights,
    const MPI_Comm &comm)
      : WassersteinBarycenters<2, 3>(
          n_measures, weights, comm, UpdateMode::TargetMeasureOnly),
        ParameterAcceptor("/Tutorial 3/WassersteinBarycentersFixedDomain"), 
        mpi_communicator(MPI_COMM_WORLD), 
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        tria(
            mpi_communicator,
            typename Triangulation<2, 3>::MeshSmoothing(
                Triangulation<2, 3>::smoothing_on_refinement |
                Triangulation<2, 3>::smoothing_on_coarsening)),
        dof_handler(tria), 
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

  void WassersteinBarycentersFixedDomain::setup_system()
  {
    std::string name = "sphere";

    GridGenerator::hyper_sphere(tria);
    tria.refine_global(n_refinements);
    dof_handler.distribute_dofs(fe);

    pcout << "Tria\n"
          << "  Number of active cells: "
          << tria.n_active_cells()
          << std::endl
          << "  Total number of cells: " << tria.n_cells()
          << std::endl
          << "  Number of degrees of freedom: " << dof_handler.n_dofs() <<  std::endl;
    
    DataOut<2, 3> data_out;
    data_out.attach_dof_handler(dof_handler);
    Vector<float> subdomain(tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(
      mapping, mapping.get_degree(),
      DataOut<2, 3>::curved_inner_cells);
    std::string output = "output/data_mesh/";
    data_out.write_vtu_with_pvtu_record(
    output, name, 0, mpi_communicator);

    // init dofs
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
  }

  void WassersteinBarycentersFixedDomain::output_results() const
  {
    std::string field_name = "source_field";
    std::vector<std::string> solution_names_1(1, "scalar_field_1");
    std::vector<std::string> solution_names_2(1, "scalar_field_2");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(1, DataComponentInterpretation::component_is_scalar);

    DataOut<2, 3> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(
      source_1,
      solution_names_1,
      DataOut<2, 3>::type_dof_data,
      interpretation);
    data_out.add_data_vector(
      source_2,
      solution_names_2,
      DataOut<2, 3>::type_dof_data,
      interpretation);

    Vector<float> subdomain(tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, mapping.get_degree(),
                         DataOut<2, 3>::curved_inner_cells);

    // Write VTU files to density_field directory
    data_out.write_vtu_with_pvtu_record(
        output_dir, field_name, 0, mpi_communicator);
  }

  void WassersteinBarycentersFixedDomain::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads() << std::endl;
    
    this->param_manager_wasserstein_barycenters.print_parameters();

    setup_system();
    
    // set sources
    std::map<types::global_dof_index, Point<3>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    // source_1
    Point<3> center_on_sphere_1(1.0, 0.0, 0.0);
    double sigma_1 = 0.3;
    SphericalGaussian gaussian_1(center_on_sphere_1, sigma_1);
    source_1.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    for (auto idx : source_1.locally_owned_elements())
      source_1[idx] = gaussian_1.value(support_points[idx]);
    source_1.compress(dealii::VectorOperation::insert);

    // source_2
    Point<3> center_on_sphere_2(-1.0, 0.0, 0.0);
    double sigma_2 = 0.3;
    SphericalGaussian gaussian_2(center_on_sphere_2, sigma_2);
    source_2.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator); 
    for (auto idx : source_2.locally_owned_elements())
      source_2[idx] = gaussian_2.value(support_points[idx]);
    source_2.compress(dealii::VectorOperation::insert);

    output_results();

    // manually set up the first source density
    {
      this->sot_solvers[0]->source_density.reinit(
        locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
      for (auto idx : this->sot_solvers[0]->source_density.locally_owned_elements())
        this->sot_solvers[0]->source_density[idx] = gaussian_1.value(support_points[idx]);
      this->sot_solvers[0]->source_density.compress(dealii::VectorOperation::insert);
      this->sot_solvers[0]->sot_solver->setup_source(
        dof_handler,
        mapping,
        fe,
        this->sot_solvers[0]->source_density,
        this->sot_solvers[0]->solver_params.quadrature_order
      );
      this->sot_solvers[0]->source_density.update_ghost_values();

      // normalize source
      auto quadrature = Utils::create_quadrature_for_mesh<2, 3>(
        tria, this->sot_solvers[0]->solver_params.quadrature_order);

      // Calculate L1 norm
      double local_l1_norm = 0.0;
      FEValues<2, 3> fe_values(
        mapping, fe, *quadrature,
        update_values | update_JxW_values);
      std::vector<double> density_values(quadrature->size());

      for (const auto &cell : dof_handler.active_cell_iterators()) {
          if (!cell->is_locally_owned())
              continue;

          fe_values.reinit(cell);
          fe_values.get_function_values(
            this->sot_solvers[0]->source_density, density_values);

          for (unsigned int q = 0; q < quadrature->size(); ++q) {
              local_l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
          }
      }
      double global_l1_norm = Utilities::MPI::sum(
        local_l1_norm, mpi_communicator);
      pcout << "Density L1 norm before normalization tria: " << global_l1_norm << std::endl;

      this->sot_solvers[0]->source_density /= global_l1_norm;
      this->sot_solvers[0]->source_density.update_ghost_values();

      pcout << "  source measure dofs: " << this->sot_solvers[0]->source_density.size() << "; target measure: " << this->sot_solvers[0]->target_density.size() << std::endl;
    }

    // manually set up the second source density
    {
      this->sot_solvers[1]->source_density.reinit(
        locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
      for (auto idx : this->sot_solvers[1]->source_density.locally_owned_elements())
        this->sot_solvers[1]->source_density[idx] = gaussian_2.value(support_points[idx]);
      this->sot_solvers[1]->source_density.compress(dealii::VectorOperation::insert);
      this->sot_solvers[1]->sot_solver->setup_source(
        dof_handler,
        mapping,
        fe,
        this->sot_solvers[1]->source_density,
        this->sot_solvers[1]->solver_params.quadrature_order
      );
      this->sot_solvers[1]->source_density.update_ghost_values();
  
      // normalize source
      auto quadrature = Utils::create_quadrature_for_mesh<2, 3>(
        tria, this->sot_solvers[1]->solver_params.quadrature_order);
  
      // Calculate L1 norm
      double local_l1_norm = 0.0;
      FEValues<2, 3> fe_values(
        mapping, fe, *quadrature,
        update_values | update_JxW_values);
      std::vector<double> density_values(quadrature->size());
  
      for (const auto &cell : dof_handler.active_cell_iterators()) {
          if (!cell->is_locally_owned())
              continue;
  
          fe_values.reinit(cell);
          fe_values.get_function_values(
            this->sot_solvers[1]->source_density, density_values);
  
          for (unsigned int q = 0; q < quadrature->size(); ++q) {
              local_l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
          }
      }
      double global_l1_norm = Utilities::MPI::sum(
        local_l1_norm, mpi_communicator);
      pcout << "Density L1 norm before normalization tria: " << global_l1_norm << std::endl;

      this->sot_solvers[1]->source_density /= global_l1_norm;
      this->sot_solvers[1]->source_density.update_ghost_values();

      pcout << "  source measure dofs: " << this->sot_solvers[1]->source_density.size() << "; target measure: " << this->sot_solvers[1]->target_density.size() << std::endl;
    }

    // get support points from first tria
    std::vector<Point<3>> sp_v;
    DoFTools::map_dofs_to_support_points(
      mapping, dof_handler, sp_v);
    unsigned int n_support_points = sp_v.size();
    std::vector<double> loc_sp(n_support_points * 3);
    for (auto idx: locally_owned_dofs)
      for (unsigned int d = 0; d < 3; ++d)
        loc_sp[idx * 3 + d] = sp_v[idx][d];
    std::vector<double> global_sp(n_support_points * 3);
    MPI_Allreduce(
      loc_sp.data(), global_sp.data(),
      n_support_points * 3, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    sp_v.clear();
    for (unsigned int i = 0; i < n_support_points; ++i) {
      sp_v.emplace_back(
        global_sp[i * 3 + 0], global_sp[i * 3 + 1], global_sp[i * 3 + 2]);
    }
    pcout << "Number of support points: " << sp_v.size() << std::endl;

    // Use the support points as target points
    std::vector<double> tmp_target_density(sp_v.size(), 1.0);
    for (auto idx : locally_owned_dofs)
      tmp_target_density[idx] = this->sot_solvers[0]->source_density[idx];
    std::vector<double> global_tmp_target_density(
      tmp_target_density.size(), 0.0);
    MPI_Allreduce(
      tmp_target_density.data(), 
      global_tmp_target_density.data(),
      tmp_target_density.size(), 
      MPI_DOUBLE, 
      MPI_SUM, 
      mpi_communicator);

    // Normalize the target density to have total mass = 1
    double total_mass = std::accumulate(
      global_tmp_target_density.begin(), global_tmp_target_density.end(), 0.0);
    if (total_mass > 0.0) {
      for (auto& val : global_tmp_target_density)
        val /= total_mass;
    }
    
    for (unsigned int i = 0; i < n_measures; ++i) {
      this->sot_solvers[i]->target_points.clear();
      for (const auto& point : sp_v)
        this->sot_solvers[i]->target_points.push_back(point);
      
      this->sot_solvers[i]->target_density.reinit(
        this->sot_solvers[i]->target_points.size());

      for (unsigned int j = 0; j < global_tmp_target_density.size(); ++j)
        this->sot_solvers[i]->target_density[j] = global_tmp_target_density[j];

      this->sot_solvers[i]->sot_solver->setup_target(
        this->sot_solvers[i]->target_points,
        this->sot_solvers[i]->target_density);
      this->sot_solvers[i]->sot_solver->set_distance_function("spherical");
    }

    // run the LLoyd algorithm
    const double absolute_threshold = 1e-8;
    const unsigned int max_iterations = 100;
    const double alpha = 1.0; // step size for the update of the barycenter
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
    WassersteinBarycentersFixedDomain wasserstein_barycenters_uniform(
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