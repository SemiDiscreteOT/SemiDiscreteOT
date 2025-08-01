#include "KokkosSOT.h"

namespace Applications
{
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;

  void print_used_cuda_memory() {
  size_t free_mem = 0;
  size_t total_mem = 0;
  cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

  if (err != cudaSuccess) {
      std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << "\n";
      return;
  }

  size_t used_mem = total_mem - free_mem;

  std::cout << "CUDA Memory Usage:\n";
  std::cout << "  Used:  " << used_mem / (1024.0 * 1024.0) << " MB\n";
  std::cout << "  Free:  " << free_mem / (1024.0 * 1024.0) << " MB\n";
  std::cout << "  Total: " << total_mem / (1024.0 * 1024.0) << " MB\n";
}

  using namespace dealii;

  KokkosSOT::KokkosSOT(const MPI_Comm &comm)
      : SemiDiscreteOT<3, 3>(comm),
        ParameterAcceptor("/Tutorial 3/KokkosSOT"), 
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
    add_parameter("task", task);
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

  void KokkosSOT::setup_system()
  {
    std::string name_1 = "cube";

    // const double centerline_radius = 2.0;
    // const double inner_radius = 0.5;
    // GridGenerator::torus(tria_1, centerline_radius, inner_radius);

    const double left = -1.0;
    const double right = 1.0;
    GridGenerator::hyper_cube(tria_1, left, right);
    tria_1.refine_global(n_refinements+1);

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

    std::string name_2 = "ball";
    // std::vector<Point<3>> vertices = {
    //   Point<3>(2.5, -2, 0),
    //   Point<3>(-2.5*0.5, -2, 2.5*std::sqrt(3)/2),
    //   Point<3>(0, 2, 0),
    //   Point<3>(-2.5*0.5, -2, -2.5*std::sqrt(3)/2)
    // };
    // GridGenerator::simplex(tria_2, vertices);
    const double radius = 1.0;
    GridGenerator::hyper_ball(tria_2, Point<3>(0, 0, 0), radius);
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

  void KokkosSOT::assemble_system()
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

  unsigned int KokkosSOT::solve()
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

  void KokkosSOT::output_results() const
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
  KokkosSOT::output_eigenfunctions() const
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
    std::ofstream output(filename);
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, 0, mpi_communicator);
  }

  void
  KokkosSOT::output_normalized_source(LinearAlgebra::distributed::Vector<double, MemorySpace::Host> &source) const
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

    data_out.build_patches(mapping, mapping.get_degree() + 1,
                           DataOut<3>::curved_inner_cells);

    const std::string filename = "source_density";
    std::ofstream output(filename);
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, 0, mpi_communicator);
  }

  void KokkosSOT::output_conditioned_densities(
    std::vector<LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> &conditioned_densities,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> &target_indices,
     std::vector<unsigned int> &potential_indices) const
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

    pcout << "   Writing "<< conditioned_densities.size() << " conditioned densities " << std::endl;
    for (unsigned int i = 0; i < conditioned_densities.size(); ++i)
    {
      data_out.add_data_vector(
          conditioned_densities[i],
          std::string("conditioned_density_") + Utilities::int_to_string(i),
          DataOut<3>::type_dof_data,
          interpretation);
    }
    data_out.add_data_vector(
      target_indices,
      std::string("target_indices"),
      DataOut<3>::type_dof_data,
      interpretation);

    data_out.build_patches(
      mapping, mapping.get_degree() + 1,
      DataOut<3>::curved_inner_cells);

    const std::string filename = "conditioned_densities";
    std::ofstream output(filename);
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, 0, mpi_communicator);
  }

  void KokkosSOT::kokkos_init(
    Kokkos::DualView<double*[3], memory_space>y,
    Kokkos::DualView<double*, memory_space>nu,
    Kokkos::DualView<double*[3], memory_space>x,
    Kokkos::DualView<double*, memory_space>mu)
  {
    auto quadrature = Utils::create_quadrature_for_mesh<3>(tria_1, solver_params.quadrature_order);
    FEValues<3> fe_values(mapping, fe, *quadrature,
                           update_values | update_JxW_values | update_quadrature_points);
    std::vector<double> density_values(quadrature->size());
    
    for (unsigned int i = 0; i < y.extent(0); ++i){
      nu.h_view(i) = this->target_density[i];
      for (unsigned int d = 0; d < 3; ++d)
        y.h_view(i, d) = this->target_points[i][d];
    };
  
    y.modify_host();
    y.sync_device();
    nu.modify_host();
    nu.sync_device();
    std::cout << "Device pointer: " << y.d_view.data() << std::endl;
    std::cout << "Host pointer:   " << y.h_view.data() << std::endl;

    unsigned int source_index = 0;
    for (const auto &cell : dof_handler_1.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      fe_values.get_function_values(this->source_density, density_values);
      const std::vector<Point<3>>& q_points = fe_values.get_quadrature_points();

      for (unsigned int q = 0; q < q_points.size(); ++q)
      {
        for (unsigned int d = 0; d < 3; ++d)
          x.h_view(source_index, d) = q_points[q][d];
        mu.h_view(source_index) = density_values[q] * fe_values.JxW(q);
        ++source_index;
      }
    }

    x.modify_host();
    x.sync_device();
    mu.modify_host();
    mu.sync_device();
    pcout << "End Kokkos view\n";
  }

  double KokkosSOT::evaluate_functional_sot(
    const Kokkos::View<double*, memory_space> phi,
    Kokkos::View<double*, memory_space> grad,
    const Kokkos::DualView<double*[3], memory_space> y,
    const Kokkos::DualView<double*, memory_space> nu,
    const Kokkos::DualView<double*[3], memory_space> x,
    const Kokkos::DualView<double*, memory_space> mu)
  {
    Kokkos::View<int*, memory_space> argmax("argmax", x.extent(0));
    Kokkos::View<double*, memory_space> maxs("maxs", x.extent(0));

    Kokkos::parallel_for("ComputeArgmax", x.extent(0), KOKKOS_LAMBDA(int i) {
      double max_kernel = -1e20;
      int max_j = -1;
      for (int j = 0; j < y.extent(0); ++j) {
        double dist = 0.0;
        for (int d = 0; d < 3; ++d) {
          double tmp = x.d_view(i, d) - y.d_view(j, d); // euclidean distance
          dist += tmp * tmp;
        }
        double current_max = phi(j) - 0.5 * dist;
        if (current_max > max_kernel) {
          max_kernel = current_max;
          max_j = j;
        }
      }
      argmax(i) = max_j;
      maxs(i) = max_kernel;
    });

    // TODO: implement with ScatterView ?
    Kokkos::parallel_for("Scattersums", x.extent(0), KOKKOS_LAMBDA(int i) {
        int j = argmax(i);
        Kokkos::atomic_add(&grad(j), mu.d_view(i));
    });

    Kokkos::parallel_for("SubtractNu", y.extent(0), KOKKOS_LAMBDA(int i) {
      grad(i) -= nu.d_view(i);
    });
    
    double functional_value = 0.0;
    Kokkos::parallel_reduce("ComputeFunctionalValue", x.extent(0), KOKKOS_LAMBDA(int i, double &f_sum) {
      f_sum += maxs(i) * mu.d_view(i);
    }, functional_value);

    double dot_phi_nu = 0.0;
    Kokkos::parallel_reduce("ComputeDotPhiNu", y.extent(0), KOKKOS_LAMBDA(int j, double &dot_sum) {
      dot_sum += phi(j) * nu.d_view(j);
    }, dot_phi_nu);

    pcout << "Functional value: " << functional_value << " " << dot_phi_nu << std::endl;
    double grad_l2_norm = 0.0;
    Kokkos::parallel_reduce("GradL2Norm", grad.extent(0), KOKKOS_LAMBDA(int i, double &sum) {
      sum += grad(i) * grad(i);
    }, grad_l2_norm);
    grad_l2_norm = std::sqrt(grad_l2_norm);
    pcout << "Grad L2 norm: " << grad_l2_norm << std::endl;

    return functional_value - dot_phi_nu;
  }

  void KokkosSOT::kokkos_semidiscrete_ot(Vector<double> &potential)
  {
    const int n_target = this->target_points.size();
    Kokkos::View<double*, memory_space> phi("phi", n_target);
    Kokkos::DualView<double*[3], memory_space> y("y", n_target, 3);
    Kokkos::DualView<double*, memory_space> nu("nu", n_target);

    auto quadrature = Utils::create_quadrature_for_mesh<3>(tria_1, solver_params.quadrature_order);
    const int n_source = tria_1.n_active_cells() * quadrature->size();
    Kokkos::DualView<double*[3], memory_space> x("x", n_source, 3);
    Kokkos::DualView<double*, memory_space> mu("mu", n_source);

    auto phi_host = Kokkos::create_mirror_view(phi);
    for (int i = 0; i < n_target; ++i)
      phi_host(i) = potential[i];
    Kokkos::deep_copy(phi, phi_host);

    kokkos_init(y, nu, x, mu);
    
    pcout << "Running on Kokkos execution space: " 
        << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    SotParameterManager::SolverParameters& solver_config = this->solver_params;
    SolverKokkosBFGS::BFGSControl control(solver_config.max_iterations, solver_config.tolerance);
    SolverKokkosBFGS::AdditionalData additional_data;
    pcout << "x: " << x.extent(0) << ", y: " << y.extent(0) << std::endl;
    
    // pcout << "Using BFGS solver with max iterations: "
    //       << control.max_iter << ", tolerance: " << control.tolerance << std::endl;

    // SolverKokkosBFGS solver(control, additional_data);
    // solver.solve(
    //   [this, y, nu, x, mu](const Kokkos::View<double*, memory_space> phi_, Kokkos::View<double*, memory_space> grad_) {
    //     return this->evaluate_functional_sot(
    //       phi_, grad_, y, nu, x, mu);
    //   },
    //   phi
    // );


    // Simple gradient descent loop
    const int max_iters = 100;
    const double tol = 1e-6;
    const double step_size = 1e-3;
    Kokkos::View<double*, memory_space> grad("grad", n_target);

    for (int iter = 0; iter < max_iters; ++iter) {
      double fval = this->evaluate_functional_sot(phi, grad, y, nu, x, mu);

      // Compute grad L2 norm
      double grad_l2_norm = 0.0;
      Kokkos::parallel_reduce("GradL2Norm", grad.extent(0), KOKKOS_LAMBDA(int i, double &sum) {
      sum += grad(i) * grad(i);
      }, grad_l2_norm);
      grad_l2_norm = std::sqrt(grad_l2_norm);

      pcout << "GD iter " << iter << ", fval: " << fval << ", grad norm: " << grad_l2_norm << std::endl;

      if (grad_l2_norm < tol)
      break;

      // Gradient descent update
      Kokkos::parallel_for("GDUpdate", grad.extent(0), KOKKOS_LAMBDA(int i) {
      phi(i) -= step_size * grad(i);
      });
    }

    Kokkos::deep_copy(phi_host, phi);
    for (int i = 0; i < n_target; ++i)
      potential[i] = phi_host(i);
  }

  // TODO: use kokkos nested parallel_for ?
  double KokkosSOT::evaluate_functional_rsot(
    double epsilon,
    const Kokkos::View<double*, memory_space> phi,
    Kokkos::View<double*, memory_space> grad,
    const Kokkos::DualView<double*[3], memory_space> y,
    const Kokkos::DualView<double*, memory_space> nu,
    const Kokkos::DualView<double*[3], memory_space> x,
    const Kokkos::DualView<double*, memory_space> mu)
  {
    Kokkos::View<double*, memory_space> sums("sums", x.extent(0));
    Kokkos::View<double*, memory_space> max_exp("Maxexp", x.extent(0));

    Kokkos::parallel_for("sumsEval", team_policy(x.extent(0), Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team) {
      int i = team.league_rank();
      double max_exp_ = -1e20;
      
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, y.extent(0)),
        [&](int j, double& max_val) {
          double dist = 0.0;
          for (int d = 0; d < 3; ++d) {
            double tmp = x.d_view(i, d) - y.d_view(j, d);
            dist += tmp * tmp;
          }
          double exp=(phi(j)-0.5*dist)/epsilon;
          if (exp > max_val)
            max_val = exp;
        },
        Kokkos::Max<double>(max_exp_)  // Kokkos reducer
      );
      
      max_exp(i) = max_exp_;
    
      double sum_quad_i = 0.0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, y.extent(0)),
        [&](int j, double& inner_sum) {
          double dist = 0.0;
          for (int d = 0; d < 3; ++d) {
            double tmp = x.d_view(i, d) - y.d_view(j, d);
            dist += tmp * tmp;
          }
          double exp=(phi(j)-0.5*dist)/epsilon;
          inner_sum += nu.d_view(j)*Kokkos::exp(exp - max_exp_);
        },
      sum_quad_i);

      sums(i) = sum_quad_i;
    });

    Kokkos::parallel_for("GradAssemble", team_policy(y.extent(0), Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team) {
      int j = team.league_rank();
      double grad_j = 0.0;

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, x.extent(0)),
        [&](int i, double& inner_sum) {
          double dist = 0.0;
          for (int d = 0; d < 3; ++d) {
            double tmp = x.d_view(i, d) - y.d_view(j, d);
            dist += tmp * tmp;
          }
          double exp=(phi(j)-0.5*dist)/epsilon;
          inner_sum += mu.d_view(i)*Kokkos::exp(exp-max_exp(i))/sums(i);
        },
      grad_j);

      grad(j) = nu.d_view(j)*(grad_j-1.0);
    });
    
    double functional_value = 0.0;
    Kokkos::parallel_reduce("ComputeFunctionalValue", x.extent(0), KOKKOS_LAMBDA(int i, double &f_sum) {
      f_sum += epsilon * mu.d_view(i) * (max_exp(i)+Kokkos::log(sums(i)));
    }, functional_value);

    double dot_phi_nu = 0.0;
    Kokkos::parallel_reduce("ComputeDotPhiNu", y.extent(0), KOKKOS_LAMBDA(int j, double &dot_sum) {
      dot_sum += phi(j) * nu.d_view(j);
    }, dot_phi_nu);
    
    return functional_value - dot_phi_nu;
  }

  void KokkosSOT::kokkos_regularized_semidiscrete_ot(Vector<double> &potential)
  {
    print_used_cuda_memory();
    const int n_target = this->target_points.size();
    Kokkos::View<double*, memory_space> phi("phi", n_target);
    Kokkos::DualView<double*[3], memory_space> y("y", n_target, 3);
    Kokkos::DualView<double*, memory_space> nu("nu", n_target);

    pcout << "Memory space: "
      << typeid(typename decltype(phi)::memory_space).name() << std::endl;
    pcout << "Running on Kokkos execution space: " 
        << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    auto quadrature = Utils::create_quadrature_for_mesh<3>(tria_1, solver_params.quadrature_order);
    pcout << "Quadrature size: " << quadrature->size() << std::endl;
    const int n_source = tria_1.n_active_cells() * quadrature->size();
    Kokkos::DualView<double*[3], memory_space> x("x", n_source, 3);
    Kokkos::DualView<double*, memory_space> mu("mu", n_source);
    print_used_cuda_memory();
    
    auto phi_host = Kokkos::create_mirror_view(phi);
    for (int i = 0; i < n_target; ++i)
      phi_host(i) = potential[i];
    Kokkos::deep_copy(phi, phi_host);

    kokkos_init(y, nu, x, mu);

    SotParameterManager::SolverParameters& solver_config = this->solver_params;
    double epsilon = solver_config.epsilon;
    this->sot_solver->current_epsilon = epsilon;
    pcout << "Epsilon: " << epsilon << std::endl;

    SolverKokkosBFGS::BFGSControl control(solver_config.max_iterations, solver_config.tolerance);
    SolverKokkosBFGS::AdditionalData additional_data;
    pcout << "Using BFGS solver with max iterations: "
          << control.max_iter << ", tolerance: " << control.tolerance << std::endl;
    pcout << "x: " << x.extent(0) << ", y: " << y.extent(0) << std::endl;

    SolverKokkosBFGS solver(control, additional_data);
    solver.solve(
      [this, y, nu, x, mu, epsilon](const Kokkos::View<double*, memory_space> phi_, Kokkos::View<double*, memory_space> grad_) {
        return this->evaluate_functional_rsot(
          epsilon, phi_, grad_, y, nu, x, mu);
      },
      phi
    );

    // Copy phi back to potential
    Kokkos::deep_copy(phi_host, phi);
    for (int i = 0; i < n_target; ++i)
      potential[i] = phi_host(i);
  }

  void KokkosSOT::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads() << std::endl;
    
    this->param_manager.print_parameters();
    setup_system();

    bool non_uniform_source_density = false;
    if (non_uniform_source_density)
    {
      // get continuous source density field as eigenfunction of the Laplacian operator
      assemble_system();
  
      const unsigned int n_iterations = solve();
      pcout << "   Solver converged in " << n_iterations << " iterations."
            << std::endl;
      output_eigenfunctions();
      pcout << "   Set source density as last eigenfunction." << std::endl;
    }

    // manually set up the source density
    this->source_density.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> tmp_source_density;
    tmp_source_density.reinit(locally_owned_dofs, mpi_communicator);
    if (non_uniform_source_density)
    {
      // use the last eigenfunction as source density
      for (auto idx : tmp_source_density.locally_owned_elements())
        tmp_source_density[idx] = eigenfunctions[n_evecs-1][idx];
    }
    else
    {
      // use a constant source density
      for (auto idx : tmp_source_density.locally_owned_elements())
        tmp_source_density[idx] = 1.0;
    }
    tmp_source_density.compress(dealii::VectorOperation::insert);
    this->source_density = tmp_source_density;
    
    // normalize source
    auto quadrature = Utils::create_quadrature_for_mesh<3>(tria_1, solver_params.quadrature_order);
    double local_l1_norm = 0.0;
    FEValues<3> fe_values(mapping, fe, *quadrature,
                           update_values | update_JxW_values | update_quadrature_points);
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

    // manually set up the target density
    std::map<types::global_dof_index, Point<3>> support_points;
    DoFTools::map_dofs_to_support_points(
      mapping, dof_handler_2, support_points);

    unsigned int n_support_points = dof_handler_2.n_dofs();
    std::vector<double> loc_sp(n_support_points * 3);
    auto locally_owned_dofs_2 = dof_handler_2.locally_owned_dofs();
    for (auto idx: locally_owned_dofs_2)
      for (unsigned int d = 0; d < 3; ++d)
        loc_sp[idx * 3 + d] = support_points[idx][d];

    std::vector<double> global_sp(n_support_points * 3);
    MPI_Allreduce(
      loc_sp.data(), global_sp.data(),
      n_support_points * 3, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        
    this->target_points.clear();
    this->target_points.resize(n_support_points);
    for (unsigned int i = 0; i < n_support_points; ++i) {
      this->target_points[i][0] = global_sp[i * 3 + 0];
      this->target_points[i][1] = global_sp[i * 3 + 1];
      this->target_points[i][2] = global_sp[i * 3 + 2];
    }
    
    this->target_density.reinit(this->target_points.size());
    this->target_density = 1.0/this->target_points.size();
    this->sot_solver->setup_target(
      this->target_points, this->target_density);

    pcout << "Dof of source: " << this->source_density.size() << std::endl;
    pcout << "Number of targets: " << this->target_points.size() << std::endl;

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

    Vector<double> potential;
    bool threshold; // whether to threshold the potential: not supported with kokkos implementation
    bool save_conditioned_densities; // only supported for regularized sot algorithm

    if (task == "kokkossot")
    {
      potential.reinit(this->target_points.size());
      kokkos_semidiscrete_ot(potential);
      this->save_results(potential, "potentials_kokkossot");
      threshold = false;
      save_conditioned_densities = false;
    }
    else if (task == "kokkosrsot")
    {
      potential.reinit(this->target_points.size());
      kokkos_regularized_semidiscrete_ot(potential);
      this->save_results(potential, "potentials_kokkosrsot");
      threshold = false;
      save_conditioned_densities = true;
    }
    else if (task == "rsot")
    {
      potential.reinit(this->target_points.size());
      SotParameterManager::SolverParameters& solver_config = this->solver_params;
      this->sot_solver->solve(potential, solver_config);
      this->save_results(potential, "potentials_rsot");
      threshold = true;
      save_conditioned_densities = true;
    }
    else
    {
      pcout << "Unknown task: " << task << std::endl;
      return;
    }

    if (save_conditioned_densities)
    {
      std::vector<unsigned int> potential_indices;
      unsigned int N = this->target_points.size()/n_conditioned_densities;
      for (unsigned int i = 0; i < n_conditioned_densities; ++i)
        potential_indices.push_back(i*N);

      std::vector<LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> conditioned_densities;
      LinearAlgebra::distributed::Vector<double, MemorySpace::Host> target_indices;
  
      this->sot_solver->get_potential_conditioned_density(
        dof_handler_1, mapping,
        potential, potential_indices, conditioned_densities, target_indices, threshold);
  
      output_conditioned_densities(
        conditioned_densities,
        target_indices,
        potential_indices);
    }
  }
} // namespace Applications

using namespace dealii;

int main(int argc, char *argv[])
{
  try
  {
    InitFinalize mpi_initialization(argc, argv,          InitializeLibrary::MPI | InitializeLibrary::SLEPc | InitializeLibrary::PETSc| InitializeLibrary::Kokkos |
        InitializeLibrary::Zoltan | InitializeLibrary::P4EST);

    // Kokkos::initialize(argc, argv);

    // printf("Hello World on Kokkos execution space %s\n",
    //         typeid(Kokkos::DefaultExecutionSpace).name ());

    // int num_threads = Kokkos::DefaultExecutionSpace().concurrency();
    // printf("Number of available threads: %d\n", num_threads);
        
    using namespace Applications;

    deallog.depth_console(2);

    MPI_Comm mpi_communicator = MPI_COMM_WORLD;
    const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(mpi_communicator);

    // Create conditional output stream
    ConditionalOStream pcout(std::cout, this_mpi_process == 0);
    
    KokkosSOT conditioned_density_test(mpi_communicator);
    
    // Use command line argument if provided, otherwise use default
    std::string param_file = "parameters.prm";
    
    pcout << "Using parameter file: " << param_file << std::endl;
    ParameterAcceptor::initialize(param_file);
    
    conditioned_density_test.run();

    // Ensure all Kokkos resources are deallocated before finalizing
    // pcout << "Finalizing Kokkos..." << std::endl;
    // Kokkos::DefaultExecutionSpace().fence();
    // Kokkos::finalize();
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