#include "SemiDiscreteOT/core/SemiDiscreteOT.h"
namespace fs = std::filesystem;

using namespace dealii;

template <int dim, int spacedim>
SemiDiscreteOT<dim, spacedim>::SemiDiscreteOT(const MPI_Comm &comm)
    : mpi_communicator(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , param_manager(comm)
    , source_params(param_manager.source_params)
    , target_params(param_manager.target_params)
    , solver_params(param_manager.solver_params)
    , multilevel_params(param_manager.multilevel_params)
    , power_diagram_params(param_manager.power_diagram_params)
    , transport_map_params(param_manager.transport_map_params)
    , selected_task(param_manager.selected_task)
    , io_coding(param_manager.io_coding)
    , source_mesh(comm)
    , target_mesh()
    , dof_handler_source(source_mesh)
    , dof_handler_target(target_mesh)
    , mesh_manager(std::make_unique<MeshManager<dim, spacedim>>(comm))
    , sot_solver(std::make_unique<SotSolver<dim, spacedim>>(comm))
{
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::mesh_generation()
{
    mesh_manager->generate_mesh(source_mesh,
                                source_params.grid_generator_function,
                                source_params.grid_generator_arguments,
                                source_params.n_refinements,
                                source_params.use_tetrahedral_mesh);

    mesh_manager->generate_mesh(target_mesh,
                                target_params.grid_generator_function,
                                target_params.grid_generator_arguments,
                                target_params.n_refinements,
                                target_params.use_tetrahedral_mesh);

    mesh_manager->save_meshes(source_mesh, target_mesh);
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::load_meshes()
{
    mesh_manager->load_source_mesh(source_mesh);
    mesh_manager->load_target_mesh(target_mesh);

    // Print mesh statistics
    const unsigned int n_global_cells =
        Utilities::MPI::sum(source_mesh.n_locally_owned_active_cells(), mpi_communicator);
    const unsigned int n_global_vertices =
        Utilities::MPI::sum(source_mesh.n_vertices(), mpi_communicator);

    pcout << "Source mesh: " << n_global_cells << " cells, "
          << n_global_vertices << " vertices" << std::endl;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        pcout << "Target mesh: " << target_mesh.n_active_cells() << " cells, "
              << target_mesh.n_vertices() << " vertices" << std::endl;
    }
}

template <int dim, int spacedim>
std::vector<std::pair<std::string, std::string>> SemiDiscreteOT<dim, spacedim>::get_target_hierarchy_files() const
{
    return Utils::get_target_hierarchy_files(multilevel_params.target_hierarchy_dir);
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::load_target_points_at_level(
    const std::string& points_file,
    const std::string& density_file)
{
    pcout << "Loading target points from: " << points_file << std::endl;
    pcout << "Loading target densities from: " << density_file << std::endl;

    std::vector<Point<spacedim>> local_target_points;
    std::vector<double> local_densities;
    bool load_success = true;

    // Only rank 0 reads files
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        // Read points
        if (!Utils::read_vector(local_target_points, points_file, io_coding))
        {
            pcout << Color::red << Color::bold << "Error: Cannot read points file: " << points_file << Color::reset << std::endl;
            load_success = false;
        }

        // Read potentials
        if (load_success && !Utils::read_vector(local_densities, density_file, io_coding))
        {
            pcout << Color::red << Color::bold << "Error: Cannot read densities file: " << density_file << Color::reset << std::endl;
            load_success = false;
        }
    }

    // Broadcast success status
    load_success = Utilities::MPI::broadcast(mpi_communicator, load_success, 0);
    if (!load_success)
    {
        throw std::runtime_error("Failed to load target points or densities");
    }

    // Broadcast sizes
    unsigned int n_points = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        n_points = local_target_points.size();
    }
    n_points = Utilities::MPI::broadcast(mpi_communicator, n_points, 0);

    // Resize containers on non-root ranks
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    {
        local_target_points.resize(n_points);
        local_densities.resize(n_points);
    }

    // Broadcast data
    for (unsigned int i = 0; i < n_points; ++i)
    {
        local_target_points[i] = Utilities::MPI::broadcast(mpi_communicator, local_target_points[i], 0);
        local_densities[i] = Utilities::MPI::broadcast(mpi_communicator, local_densities[i], 0);
    }

    // Update class members
    target_points = std::move(local_target_points);
    target_density.reinit(n_points);
    for (unsigned int i = 0; i < n_points; ++i)
    {
        target_density[i] = local_densities[i];
    }

    pcout << Color::green << "Successfully loaded " << n_points << " target points at this level" << Color::reset << std::endl;
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::load_hierarchy_data(const std::string& hierarchy_dir, int specific_level) {
    has_hierarchy_data_ = Utils::load_hierarchy_data<dim, spacedim>(hierarchy_dir, child_indices_, specific_level, mpi_communicator, pcout);
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::normalize_density(LinearAlgebra::distributed::Vector<double>& density)
{
    auto quadrature = Utils::create_quadrature_for_mesh<dim, spacedim>(source_mesh, solver_params.quadrature_order);
    // Calculate L1 norm
    double local_l1_norm = 0.0;
    FEValues<dim, spacedim> fe_values(*mapping, *fe_system, *quadrature,
                           update_values | update_JxW_values);
    std::vector<double> density_values(quadrature->size());

    for (const auto &cell : dof_handler_source.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(density, density_values);

        for (unsigned int q = 0; q < quadrature->size(); ++q)
        {
            local_l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
        }
    }

    const double global_l1_norm = Utilities::MPI::sum(local_l1_norm, mpi_communicator);
    pcout << "Density L1 norm before normalization: " << global_l1_norm << std::endl;

    density /= global_l1_norm;
    density.update_ghost_values();
}


template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::setup_source_finite_elements(const bool is_multilevel)
{
    // Create finite element and mapping for the mesh.
    auto [fe, map] = Utils::create_fe_and_mapping_for_mesh<dim, spacedim>(source_mesh);
    fe_system = std::move(fe);
    mapping = std::move(map);

    // Distribute DoFs.
    dof_handler_source.distribute_dofs(*fe_system);
    IndexSet locally_owned_dofs = dof_handler_source.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler_source, locally_relevant_dofs);

    // Initialize source density.
    source_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    // Handle custom density if enabled.
    if (source_params.use_custom_density)
    {
        if (source_params.density_file_format == "vtk")
        {
            if constexpr (dim==spacedim)
            {
                if (is_multilevel)
                {
                    pcout << Color::green << "Interpolating source density from VTK to source mesh" << Color::reset << std::endl;
    
                    // For multilevel, use the stored VTKHandler if available
                    if (source_vtk_handler)
                    {
                        pcout << "Using stored VTKHandler for source density interpolation" << std::endl;
                        VectorTools::interpolate(dof_handler_source, *source_vtk_handler, source_density);
                    }
                    else
                    {
                        // Fallback to non-conforming nearest neighbor interpolation if VTKHandler not available
                        pcout << "VTKHandler not available, using non-conforming nearest neighbor interpolation" << std::endl;
                        Utils::interpolate_non_conforming_nearest(vtk_dof_handler_source,
                                                                  vtk_field_source,
                                                                  dof_handler_source,
                                                                  source_density);
                    }
    
                    pcout << "Source density interpolated from VTK to source mesh" << std::endl;
                }
                else
                {
                    pcout << "Using custom source density from file: " << source_params.density_file_path << std::endl;
                    bool density_loaded = false;
    
                    try
                    {
                        pcout << "Loading VTK file using VTKHandler: " << source_params.density_file_path << std::endl;
    
                        // Create VTKHandler instance and store it as a member variable
                        source_vtk_handler = std::make_unique<VTKHandler<dim>>(source_params.density_file_path);
    
                        // Setup the field for interpolation using the configured field name
                        source_vtk_handler->setup_field(source_params.density_field_name, VTKHandler<dim>::DataLocation::PointData, 0);
    
                        pcout << "Source density loaded from VTK file" << std::endl;
                        pcout << Color::green << "Interpolating source density from VTK to source mesh" << Color::reset << std::endl;
    
                        // Interpolate the field to the source mesh
                        VectorTools::interpolate(dof_handler_source, *source_vtk_handler, source_density);
    
                        pcout << "Successfully interpolated VTK field to source mesh" << std::endl;
                        density_loaded = true;
                    }
                    catch (const std::exception &e)
                    {
                        pcout << Color::red << "Error loading VTK file: " << e.what() << Color::reset << std::endl;
                        density_loaded = false;
                    }
    
                    if (!density_loaded)
                    {
                        pcout << Color::red << "Failed to load custom density, using uniform density" << Color::reset << std::endl;
                        source_density = 1.0;
                    }
                }

            } else {
                pcout << Color::red << "Unsupported dim!=spacedim" << Color::reset << std::endl;
                throw std::runtime_error("Unsupported dimension for source mesh");
            }
        }
        else
        {
            pcout << Color::red << "Unsupported density file format, using uniform density" << Color::reset << std::endl;
            source_density = 1.0;
        }
    }
    else
    {
        pcout << Color::green << "Using uniform source density" << Color::reset << std::endl;
        source_density = 1.0;
    }

    if (source_params.use_custom_density)
    {
        if (source_params.density_file_format == "vtk")
        {
            if constexpr (dim == spacedim)
            {
                if (is_multilevel)
                {
                    pcout << Color::green << "Interpolating source density from VTK to source mesh" << Color::reset << std::endl;
                    // For multilevel, use non-conforming nearest neighbor interpolation.
                    Utils::interpolate_non_conforming_nearest<dim, spacedim>(vtk_dof_handler_source,
                        vtk_field_source,
                        dof_handler_source,
                        source_density);                
                    pcout << "Source density interpolated from VTK to source mesh" << std::endl;
                } else
                {
                    pcout << "Using custom source density from file: " << source_params.density_file_path << std::endl;
                    bool density_loaded = false;

                    try {
                        pcout << "Loading VTK file using VTKHandler: " << source_params.density_file_path << std::endl;
                        
                        // Create VTKHandler instance and store it as a member variable
                        source_vtk_handler = std::make_unique<VTKHandler<dim, spacedim>>(source_params.density_file_path);
                        
                        // Setup the field for interpolation using the configured field name
                        source_vtk_handler->setup_field(
                            source_params.density_field_name, VTKHandler<dim, spacedim>::DataLocation::PointData, 0);
                        
                        pcout << "Source density loaded from VTK file" << std::endl;
                        pcout << Color::green << "Interpolating source density from VTK to source mesh" << Color::reset << std::endl;
                        
                        // Interpolate the field to the source mesh
                        VectorTools::interpolate(dof_handler_source, *source_vtk_handler, source_density);
                        
                        pcout << "Successfully interpolated VTK field to source mesh" << std::endl;
                        density_loaded = true;
                    } catch (const std::exception& e) {
                        pcout << Color::red << "Error loading VTK file: " << e.what() << Color::reset << std::endl;
                        density_loaded = false;
                    }

                    if (!density_loaded)
                    {
                        pcout << Color::red << "Failed to load custom density, using uniform density" << Color::reset << std::endl;
                        source_density = 1.0;
                    }
                }
            } else
            {
                pcout << Color::red << "Unsupported dim!=spacedim" << Color::reset << std::endl;
                throw std::runtime_error("Unsupported dimension for source mesh");
            }
        }
        else
        {
            pcout << Color::red << "Unsupported density file format, using uniform density" << Color::reset << std::endl;
            source_density = 1.0;
        }
    }
    else
    {
        pcout << Color::green << "Using uniform source density" << Color::reset << std::endl;
        source_density = 1.0;
    }

    source_density.update_ghost_values();
    normalize_density(source_density);

    unsigned int n_locally_owned = 0;
    for (const auto &cell : dof_handler_source.active_cell_iterators())
    {
        if (cell->is_locally_owned())
            ++n_locally_owned;
    }
    const unsigned int n_total_owned = Utilities::MPI::sum(n_locally_owned, mpi_communicator);
    pcout << "Total cells: " << source_mesh.n_active_cells()
          << ", Locally owned on proc " << this_mpi_process
          << ": " << n_locally_owned
          << ", Sum of owned: " << n_total_owned << std::endl;
    pcout << "Source mesh finite elements initialized with " << dof_handler_source.n_dofs() << " DoFs" << std::endl;
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::setup_target_finite_elements()
{
    // Load or compute target points (shared across all processes)
    const std::string directory = "output/data_points";
    bool points_loaded = false;
    target_points.clear(); // Clear on all processes

    // Only root process reads/computes points
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        auto [fe, map] = Utils::create_fe_and_mapping_for_mesh<dim, spacedim>(target_mesh);
        fe_system_target = std::move(fe);
        mapping_target = std::move(map);
        dof_handler_target.distribute_dofs(*fe_system_target);

        // Load or compute target points
        if (Utils::read_vector(target_points, directory + "/target_points", io_coding))
        {
            points_loaded = true;
            pcout << "Target points loaded from file" << std::endl;
        } else {
            std::map<types::global_dof_index, Point<spacedim>> support_points_target;
            DoFTools::map_dofs_to_support_points(*mapping_target, dof_handler_target, support_points_target);
            for (const auto &point_pair : support_points_target)
            {
                target_points.push_back(point_pair.second);
            }
            Utils::write_vector(target_points, directory + "/target_points", io_coding);
            points_loaded = true;
            pcout << "Target points computed and saved to file" << std::endl;
        }

        if (target_params.use_custom_density)
        {
            pcout << "Using custom target density from file: " << target_params.density_file_path << std::endl;
            bool density_loaded = false;

            if (target_params.density_file_format == "vtk")
            {
                if constexpr (dim==spacedim)
                {
                    // Use the new VTKHandler class to load and interpolate the field
                    try
                    {
                        pcout << "Loading VTK file using VTKHandler: " << target_params.density_file_path << std::endl;
    
                        // Create VTKHandler instance
                        VTKHandler<dim> vtk_handler(target_params.density_file_path);
    
                        // Setup the field for interpolation using the configured field name
                        vtk_handler.setup_field(target_params.density_field_name, VTKHandler<dim>::DataLocation::PointData, 0);
    
                        pcout << "Target density loaded from VTK file" << std::endl;
                        target_density.reinit(dof_handler_target.n_dofs());
                        pcout << "Target density size: " << target_density.size() << std::endl;
    
                        pcout << Color::green << "Interpolating target density from VTK to target mesh" << Color::reset << std::endl;
    
                        // Interpolate the field to the target mesh
                        VectorTools::interpolate(dof_handler_target, vtk_handler, target_density);
    
                        pcout << "Successfully interpolated VTK field to target mesh" << std::endl;
                        pcout << "L1 norm of interpolated field: " << target_density.l1_norm() << std::endl;
                        target_density /= target_density.l1_norm();
    
                        density_loaded = true;
                    }
                    catch (const std::exception &e)
                    {
                        pcout << Color::red << "Error loading VTK file: " << e.what() << Color::reset << std::endl;
                        density_loaded = false;
                    }
                }
                else {
                    pcout << Color::red << "Unsupported dim!=spacedim" << Color::reset << std::endl;
                    throw std::runtime_error("Unsupported dimension for target mesh");
                }
            }
            else
            {
                std::vector<double> density_values;
                // Try reading as plain text file
                density_loaded = Utils::read_vector(density_values, target_params.density_file_path);
                // normalize target density
                target_density = Vector<double>(density_values.size());
                for (unsigned int i = 0; i < density_values.size(); ++i)
                {
                    target_density[i] = density_values[i];
                }
                target_density /= target_density.l1_norm();
            }
            if (!density_loaded)
            {
                pcout << Color::red << "Failed to load custom density, using uniform density" << Color::reset << std::endl;
                target_density.reinit(target_points.size());
                target_density = 1.0 / target_points.size();
            }
        }
        else
        {
            pcout << Color::green << "Using uniform target density" << Color::reset << std::endl;
            target_density.reinit(target_points.size());
            target_density = 1.0 / target_points.size();
        }
    }

    // Broadcast success flag
    points_loaded = Utilities::MPI::broadcast(mpi_communicator, points_loaded, 0);
    if (!points_loaded)
    {
        throw std::runtime_error("Failed to load or compute target points");
    }

    // Broadcast target points size first
    unsigned int n_points = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        n_points = target_points.size();
    }
    n_points = Utilities::MPI::broadcast(mpi_communicator, n_points, 0);

    // Resize target points and density on non-root processes
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    {
        target_points.resize(n_points);
        target_density.reinit(n_points);
    }

    target_points = Utilities::MPI::broadcast(mpi_communicator, target_points, 0);
    target_density = Utilities::MPI::broadcast(mpi_communicator, target_density, 0);

    pcout << "Setup complete with " << target_points.size() << " target points" << std::endl;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        // Create output directory if it doesn't exist
        std::string output_dir = "output/data_density";
        fs::create_directories(output_dir);

        // Save density to file
        std::string density_file = output_dir + "/target_density";
        std::vector<double> output_density_values(target_density.begin(), target_density.end());

        Utils::write_vector(output_density_values, density_file, io_coding);
        pcout << "Target density saved to " << density_file << std::endl;
    }
}


template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::setup_finite_elements()
{
    setup_source_finite_elements();
    setup_target_finite_elements();
}


template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::setup_target_points()
{
    mesh_manager->load_target_mesh(target_mesh);
    setup_target_finite_elements();
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::assign_potentials_by_hierarchy(
    Vector<double>& potentials, int coarse_level, int fine_level, const Vector<double>& prev_potentials) {

    if (!has_hierarchy_data_ || coarse_level < 0 || fine_level < 0)
    {
        std::cerr << "Invalid hierarchy levels for potential assignment" << std::endl;
        return;
    }

    // Direct assignment if same level
    if (coarse_level == fine_level)
    {
        potentials = prev_potentials;
        return;
    }

    // Initialize potentials for current level
    potentials.reinit(target_points.size());

    if (multilevel_params.use_softmax_potential_transfer)
    {
        pcout << "Applying softmax-based potential assignment from level " << coarse_level
              << " to level " << fine_level << std::endl;
        pcout << "Source points: " << prev_potentials.size()
              << ", Target points: " << target_points.size() << std::endl;

        // Create SoftmaxRefinement instance
        SoftmaxRefinement<dim, spacedim> softmax_refiner(
            mpi_communicator,
            dof_handler_source,
            *mapping,
            *fe_system,
            source_density,
            solver_params.quadrature_order,
            current_distance_threshold);

        // Add timer for softmax refinement
        Timer softmax_timer;
        softmax_timer.start();

        // Apply softmax refinement
        potentials = softmax_refiner.compute_refinement(
            target_points,         // target_points_fine
            target_density,        // target_density_fine
            target_points_coarse,  // target_points_coarse
            target_density_coarse, // target_density_coarse
            prev_potentials,       // potentials_coarse
            solver_params.epsilon,
            fine_level,
            child_indices_);

        softmax_timer.stop();
        pcout << "Softmax-based potential assignment completed in " << softmax_timer.wall_time() << " seconds." << std::endl;
    }
    else
    {
        pcout << "Applying direct potential assignment from level " << coarse_level
              << " to level " << fine_level << std::endl;
        pcout << "Source points: " << prev_potentials.size()
              << ", Target points: " << target_points.size() << std::endl;

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
// Going from coarse to fine: Each child gets its parent's potential
#pragma omp parallel for
            for (size_t j = 0; j < prev_potentials.size(); ++j)
            {
                const auto &children = child_indices_[fine_level][j];
                for (size_t child : children)
                {
                    potentials[child] = prev_potentials[j];
                }
            }
        }

        // Broadcast the potentials to all processes
        Utilities::MPI::broadcast(mpi_communicator, potentials, 0);
    }
}

// run single sot optimization with epsilon scaling
template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::run_sot()
{
    Timer timer;
    timer.start();

    setup_finite_elements();

    pcout << Color::yellow << Color::bold << "Starting SOT optimization with " << target_points.size()
          << " target points and " << source_density.size() << " source points" << Color::reset << std::endl;

    // Configure solver parameters
    SotParameterManager::SolverParameters &solver_config = solver_params;

    // Set up source measure
    sot_solver->setup_source(dof_handler_source,
                             *mapping,
                             *fe_system,
                             source_density,
                             solver_config.quadrature_order);

    // Set up target measure
    sot_solver->setup_target(target_points, target_density);

    Vector<double> potential(target_points.size());

    try
    {
        if (solver_config.use_epsilon_scaling && epsilon_scaling_handler)
        {
            pcout << "Using epsilon scaling with EpsilonScalingHandler:" << std::endl
                  << "  Initial epsilon: " << solver_config.epsilon << std::endl
                  << "  Scaling factor: " << solver_config.epsilon_scaling_factor << std::endl
                  << "  Number of steps: " << solver_config.epsilon_scaling_steps << std::endl;
            // Compute epsilon distribution for a single level
            std::vector<std::vector<double>> epsilon_distribution =
                epsilon_scaling_handler->compute_epsilon_distribution(1);

            if (!epsilon_distribution.empty() && !epsilon_distribution[0].empty())
            {
                const auto &epsilon_sequence = epsilon_distribution[0];

                // Run optimization for each epsilon value
                for (size_t i = 0; i < epsilon_sequence.size(); ++i)
                {
                    pcout << "\nEpsilon scaling step " << i + 1 << "/" << epsilon_sequence.size()
                          << " (λ = " << epsilon_sequence[i] << ")" << std::endl;

                    solver_config.epsilon = epsilon_sequence[i];

                    try
                    {
                        sot_solver->solve(potential, solver_config);

                        // Save intermediate results
                        if (i < epsilon_sequence.size() - 1)
                        {
                            std::string eps_suffix = "_eps" + std::to_string(i + 1);
                            save_results(potential, "potential" + eps_suffix);
                        }
                    }
                    catch (const SolverControl::NoConvergence &exc)
                    {
                        if (exc.last_step >= solver_params.max_iterations)
                        {
                            pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << i + 1
                                  << " (epsilon=" << epsilon_sequence[i] << "): Max iterations reached"
                                  << Color::reset << std::endl;
                        }
                    }
                }
            }
        }
        else
        {
            // Run single optimization with original epsilon
            try
            {
                sot_solver->solve(potential, solver_config);
            }
            catch (const SolverControl::NoConvergence &exc)
            {
                pcout << Color::red << Color::bold << "Warning: Optimization did not converge." << Color::reset << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        pcout << Color::red << Color::bold << "An exception occurred during SOT solve: " << e.what() << Color::reset << std::endl;
    }

    // Save final results
    save_results(potential, "potentials");

    timer.stop();
    const double total_time = timer.wall_time();
    pcout << "\n"
          << Color::green << Color::bold << "SOT optimization completed in " << total_time << " seconds" << Color::reset << std::endl;
          
    // Save convergence info
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::string eps_dir = "output/epsilon_" + Utils::to_scientific_string(solver_config.epsilon);
        fs::create_directories(eps_dir);
        std::ofstream conv_info(eps_dir + "/convergence_info.txt");
        conv_info << "Regularization parameter (λ): " << solver_config.epsilon << "\n";
        conv_info << "Number of iterations: " << sot_solver->get_last_iteration_count() << "\n";
        conv_info << "Final function value: " << sot_solver->get_last_functional_value() << "\n";
        conv_info << "Last threshold value: " << sot_solver->get_last_distance_threshold() << "\n";
        conv_info << "Total execution time: " << total_time << " seconds\n";
        conv_info << "Convergence achieved: " << sot_solver->get_convergence_status() << "\n";
    }
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::run_target_multilevel(
    const std::string& source_mesh_file,
    Vector<double>* output_potentials,
    bool save_results_to_files)
{
    Timer global_timer;
    global_timer.start();

    // Reset total softmax refinement time
    double total_softmax_time = 0.0;

    pcout << Color::yellow << Color::bold << "Starting target point cloud multilevel SOT computation..." << Color::reset << std::endl;

    // Load source mesh based on input parameters
    if (source_mesh_file.empty())
    {
        mesh_manager->load_source_mesh(source_mesh);
        setup_source_finite_elements();
    }
    else
    {
        pcout << "Source mesh loaded from file: " << source_mesh_file << std::endl;
        mesh_manager->load_mesh_at_level(source_mesh, dof_handler_source, source_mesh_file);
        setup_source_finite_elements(true);
        pcout << "Source mesh loaded from file: " << source_mesh_file << std::endl;
    }

    // Get target point cloud hierarchy files (sorted from coarsest to finest)
    unsigned int num_levels = 0;
    std::vector<std::pair<std::string, std::string>> hierarchy_files;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        try
        {
            hierarchy_files = Utils::get_target_hierarchy_files(multilevel_params.target_hierarchy_dir);
        }
        catch (const std::exception &e)
        {
            pcout << Color::red << Color::bold << "Error: " << e.what() << Color::reset << std::endl;
            pcout << "Please run prepare_target_multilevel first." << std::endl;
            return;
        }

        if (hierarchy_files.empty())
        {
            pcout << "No target point cloud hierarchy found. Please run prepare_target_multilevel first." << std::endl;
            return;
        }
        num_levels = hierarchy_files.size();
    }

    num_levels = Utilities::MPI::broadcast(mpi_communicator, num_levels, 0);

    // Initialize hierarchy data structure but don't load data yet
    if (!has_hierarchy_data_)
    {
        pcout << "Initializing hierarchy data structure..." << std::endl;
        load_hierarchy_data(multilevel_params.target_hierarchy_dir, -1);
    }

    if (!has_hierarchy_data_)
    {
        pcout << "Failed to initialize hierarchy data. Cannot proceed with multilevel computation." << std::endl;
        return;
    }

    // Store original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.epsilon;

    // Create output directory
    std::string eps_dir = "output/epsilon_" + Utils::to_scientific_string(original_regularization);
    std::string target_multilevel_dir = eps_dir + "/target_multilevel";
    fs::create_directories(target_multilevel_dir);

    // Create/Open the summary log file on rank 0
    std::ofstream summary_log;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        summary_log.open(target_multilevel_dir + "/summary_log.txt", std::ios::app);
        // Write header if file is new or empty
        if (summary_log.tellp() == 0) {
            summary_log << "Target Level | Solver Iterations | Time (s) | Last Threshold\n";
            summary_log << "------------------------------------------------------------\n";
        }
    }

    // Setup epsilon scaling if enabled
    std::vector<std::vector<double>> epsilon_distribution;
    if (solver_params.use_epsilon_scaling && epsilon_scaling_handler)
    {
        pcout << "Computing epsilon distribution for target multilevel optimization..." << std::endl;
        epsilon_distribution = epsilon_scaling_handler->compute_epsilon_distribution(
            num_levels);
        epsilon_scaling_handler->print_epsilon_distribution();
    }

    // Vector to store current potentials solution
    Vector<double> level_potentials;

    // Configure solver parameters for this level
    SotParameterManager::SolverParameters &solver_config = solver_params;

    // Set up source measure (this remains constant across levels)
    sot_solver->setup_source(dof_handler_source,
                             *mapping,
                             *fe_system,
                             source_density,
                             solver_config.quadrature_order);

    // Process each level of the hierarchy (from coarsest to finest)
    for (size_t level = 0; level < num_levels; ++level)
    {
        int level_number = 0;
        std::string level_output_dir;
        std::string points_file;
        std::string potentials_file;
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            const auto &hierarchy_file = hierarchy_files[level];
            points_file = hierarchy_file.first;
            potentials_file = hierarchy_file.second;

            // Extract level number from filename
            std::string level_num = points_file.substr(
                points_file.find("level_") + 6,
                points_file.find("_points") - points_file.find("level_") - 6);
            level_number = std::stoi(level_num);

            // Create output directory for this level
            level_output_dir = target_multilevel_dir + "/target_level_" + level_num;
            fs::create_directories(level_output_dir);
        }
        level_number = Utilities::MPI::broadcast(mpi_communicator, level_number, 0);

        pcout << "\n"
              << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;
        pcout << Color::magenta << Color::bold << "Processing target point cloud level " << level_number << Color::reset << std::endl;
        pcout << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;

        // Load hierarchy data for this level only
        load_hierarchy_data(multilevel_params.target_hierarchy_dir, level_number);

        // Load target points for this level
        if (level > 0)
        {
            target_points_coarse = target_points;
            target_density_coarse = target_density;
        }
        load_target_points_at_level(points_file, potentials_file);
        pcout << "Target points loaded for level " << level_number << std::endl;
        pcout << "Target points size: " << target_points.size() << std::endl;

        // Set up target measure for this level
        sot_solver->setup_target(target_points, target_density);

        // Initialize potentials for this level
        Vector<double> current_level_potentials(target_points.size());
        if (level > 0)
        {
            // Use hierarchy-based potential transfer from previous level
            Timer transfer_timer;
            transfer_timer.start();
            assign_potentials_by_hierarchy(current_level_potentials, level_number + 1, level_number, level_potentials);
            transfer_timer.stop();

            // Update total softmax time if using softmax transfer
            if (multilevel_params.use_softmax_potential_transfer)
            {
                total_softmax_time += transfer_timer.wall_time();
            }
        }

        Timer level_timer;
        level_timer.start();

        // Apply epsilon scaling for this level if enabled
        if (solver_params.use_epsilon_scaling && epsilon_scaling_handler && !epsilon_distribution.empty())
        {
            const auto &level_epsilons = epsilon_scaling_handler->get_epsilon_values_for_level(level);

            if (!level_epsilons.empty())
            {
                pcout << "Using " << level_epsilons.size() << " epsilon values for level " << level_number << std::endl;

                // Process each epsilon value for this level
                for (size_t eps_idx = 0; eps_idx < level_epsilons.size(); ++eps_idx)
                {
                    double current_epsilon = level_epsilons[eps_idx];
                    pcout << "  Epsilon scaling step " << eps_idx + 1 << "/" << level_epsilons.size()
                          << " (λ = " << current_epsilon << ")" << std::endl;

                    // Update regularization parameter
                    solver_config.epsilon = current_epsilon;

                    try
                    {
                        // Run optimization with current epsilon
                        sot_solver->solve(current_level_potentials, solver_config);

                        // Save intermediate results if this is not the last epsilon for this level
                        if (eps_idx < level_epsilons.size() - 1)
                        {
                            std::string eps_suffix = "_eps" + std::to_string(eps_idx + 1);
                            save_results(current_level_potentials, level_output_dir + "/potentials" + eps_suffix, false);
                        }
                    }
                    catch (const SolverControl::NoConvergence &exc)
                    {
                        if (exc.last_step >= solver_params.max_iterations)
                        {
                            pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << eps_idx + 1
                                  << " (epsilon=" << current_epsilon << "): Max iterations reached"
                                  << Color::reset << std::endl;
                        }
                        pcout << Color::red << Color::bold << "  Warning: Optimization did not converge for epsilon "
                              << current_epsilon << " at target level " << level_number << Color::reset << std::endl;
                        // Continue with next epsilon value
                    }
                }
            }
            else
            {
                // If no epsilon values for this level, use the smallest epsilon from the sequence
                solver_config.epsilon = original_regularization;

                try
                {
                    // Run optimization with default epsilon
                    sot_solver->solve(current_level_potentials, solver_config);
                }
                catch (const SolverControl::NoConvergence &exc)
                {
                    if (exc.last_step >= solver_params.max_iterations)
                    {
                        pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << level_number
                              << " (epsilon=" << original_regularization << "): Max iterations reached"
                              << Color::reset << std::endl;
                    }
                    pcout << Color::red << Color::bold << "Warning: Optimization did not converge for level " << level_number << Color::reset << std::endl;
                    pcout << "  Iterations: " << exc.last_step << std::endl;
                }
            }
        }
        else
        {
            // No epsilon scaling, just run the optimization once
            try
            {
                // Run optimization for this level
                sot_solver->solve(current_level_potentials, solver_config);
            }
            catch (SolverControl::NoConvergence &exc)
            {
                if (exc.last_step >= solver_params.max_iterations)
                {
                    pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << level_number
                          << " (epsilon=" << original_regularization << "): Max iterations reached"
                          << Color::reset << std::endl;
                }
                pcout << Color::red << Color::bold << "Warning: Optimization did not converge for level " << level_number << Color::reset << std::endl;
                pcout << "  Iterations: " << exc.last_step << std::endl;
                if (level == 0)
                    return; // If coarsest level fails, abort
                // Otherwise continue to next level with current potentials
            }
        }

        level_timer.stop();
        const double level_time = level_timer.wall_time();
        const unsigned int last_iterations = sot_solver->get_last_iteration_count();
        current_distance_threshold = sot_solver->get_last_distance_threshold();
        
        // Log summary information to file (only rank 0)
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && summary_log.is_open()) {
            summary_log << std::setw(13) << level_number << " | "
                      << std::setw(17) << last_iterations << " | "
                      << std::setw(8) << std::fixed << std::setprecision(4) << level_time << " | "
                      << std::setw(14) << std::scientific << std::setprecision(6) << current_distance_threshold << "\n";
        }

        // Save results for this level
        save_results(current_level_potentials, level_output_dir + "/potentials", false);
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::ofstream conv_info(level_output_dir + "/convergence_info.txt");
            conv_info << "Regularization parameter (λ): " << solver_params.epsilon << "\n";
            conv_info << "Number of iterations: " << sot_solver->get_last_iteration_count() << "\n";
            conv_info << "Final function value: " << sot_solver->get_last_functional_value() << "\n";
            conv_info << "Convergence achieved: " << sot_solver->get_convergence_status() << "\n";
            conv_info << "Level: " << level_number << "\n";
            pcout << "Level " << level_number << " results saved to " << level_output_dir << "/potentials" << std::endl;
        }

        // Store potentials for next level
        level_potentials = current_level_potentials;
    }

    // If output_potentials is provided, copy the final potentials
    if (output_potentials != nullptr)
    {
        *output_potentials = level_potentials;
    }

    // Save final potentials in the top-level directory
    save_results(level_potentials, target_multilevel_dir + "/potentials", false);

    // Close the log file on rank 0
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && summary_log.is_open()) {
        summary_log.close();
    }

    // Restore original parameters
    solver_params.tolerance = original_tolerance;
    solver_params.max_iterations = original_max_iterations;
    solver_params.epsilon = original_regularization;

    global_timer.stop();
    const double total_time = global_timer.wall_time();
    
    // Save global convergence info
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::ofstream conv_info(target_multilevel_dir + "/convergence_info.txt");
        conv_info << "Regularization parameter (λ): " << original_regularization << "\n";
        conv_info << "Final number of iterations: " << sot_solver->get_last_iteration_count() << "\n";
        conv_info << "Final function value: " << sot_solver->get_last_functional_value() << "\n";
        conv_info << "Last threshold value: " << current_distance_threshold << "\n";
        conv_info << "Total execution time: " << total_time << " seconds\n";
        conv_info << "Convergence achieved: " << sot_solver->get_convergence_status() << "\n";
        conv_info << "Number of levels: " << num_levels << "\n";
    }
    
    pcout << "\n"
          << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;
    pcout << Color::magenta << Color::bold << "Total multilevel target computation time: " << total_time << " seconds" << Color::reset << std::endl;

    // Report total softmax time if applicable
    if (multilevel_params.use_softmax_potential_transfer && total_softmax_time > 0.0)
    {
        pcout << Color::magenta << Color::bold << "Total time spent on softmax potential transfers: " << total_softmax_time
              << " seconds (" << (total_softmax_time / total_time * 100.0) << "%)" << Color::reset << std::endl;
    }

    pcout << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::run_target_multilevel_for_source_level(
    const std::string& source_mesh_file, Vector<double>& potentials)
{
    run_target_multilevel(source_mesh_file, &potentials);
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::run_source_multilevel()
{
    Timer global_timer;
    global_timer.start();

    pcout << Color::yellow << Color::bold << "Starting source multilevel SOT computation..." << Color::reset << std::endl;

    // Retrieve source mesh hierarchy files.
    std::vector<std::string> source_mesh_files = mesh_manager->get_mesh_hierarchy_files(multilevel_params.source_hierarchy_dir);
    if (source_mesh_files.empty())
    {
        pcout << "No source mesh hierarchy found. Please run prepare_source_multilevel first." << std::endl;
        return;
    }
    unsigned int num_levels = source_mesh_files.size();

    // Backup original solver parameters.
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.epsilon;

    // Create output directory structure.
    std::string eps_dir = "output/epsilon_" + Utils::to_scientific_string(original_regularization);
    std::string source_multilevel_dir = eps_dir + "/source_multilevel";
    fs::create_directories(source_multilevel_dir);

    // Create/Open the summary log file on rank 0
    std::ofstream summary_log;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        summary_log.open(source_multilevel_dir + "/summary_log.txt", std::ios::app);
        // Write header if file is new or empty
        if (summary_log.tellp() == 0) {
            summary_log << "Source Level | Solver Iterations | Time (s) | Last Threshold\n";
            summary_log << "------------------------------------------------------------\n";
        }
    }

    // Setup epsilon scaling if enabled.
    std::vector<std::vector<double>> epsilon_distribution;
    if (solver_params.use_epsilon_scaling && epsilon_scaling_handler)
    {
        pcout << "Computing epsilon distribution for source multilevel optimization..." << std::endl;
        epsilon_distribution = epsilon_scaling_handler->compute_epsilon_distribution(num_levels);
        epsilon_scaling_handler->print_epsilon_distribution();
    }

    // Lambda to encapsulate the epsilon scaling and solver call.
    auto process_epsilon_scaling_for_source_multilevel =
        [this, &original_regularization, &epsilon_distribution](
            Vector<double> &potentials,
            const unsigned int level_idx, // 0 to num_levels-1
            const std::string &level_output_dir) {
        if (solver_params.use_epsilon_scaling && epsilon_scaling_handler && !epsilon_distribution.empty())
        {
            const auto &level_epsilons = epsilon_scaling_handler->get_epsilon_values_for_level(level_idx);
            if (!level_epsilons.empty())
            {
                pcout << "Using " << level_epsilons.size() << " epsilon values for source level " << level_idx << std::endl;
                for (size_t eps_idx = 0; eps_idx < level_epsilons.size(); ++eps_idx)
                {
                    double current_epsilon = level_epsilons[eps_idx];
                    pcout << "  Epsilon scaling step " << eps_idx + 1 << "/" << level_epsilons.size()
                          << " (λ = " << current_epsilon << ")" << std::endl;
                    solver_params.epsilon = current_epsilon;

                    try
                    {
                        sot_solver->solve(potentials, solver_params);
                        if (eps_idx < level_epsilons.size() - 1)
                        {
                            std::string eps_suffix = "_eps" + std::to_string(eps_idx + 1);
                            save_results(potentials, level_output_dir + "/potentials" + eps_suffix, false);
                        }
                    }
                    catch (const SolverControl::NoConvergence &exc)
                    {
                        if (exc.last_step >= solver_params.max_iterations)
                        {
                            pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << eps_idx + 1
                                  << " (epsilon=" << current_epsilon << "): Max iterations reached" << Color::reset << std::endl;
                        }
                        pcout << Color::red << Color::bold << "  Warning: Optimization did not converge for epsilon "
                              << current_epsilon << " at source level " << level_idx << Color::reset << std::endl;
                        // Continue with the next epsilon value.
                    }
                }
                return; // Epsilon scaling applied
            }
        }
        // If no epsilon scaling is applied or no epsilon values exist for this level.
        solver_params.epsilon = original_regularization;
        sot_solver->solve(potentials, solver_params);
    };

    Vector<double> final_potentials;
    Vector<double> previous_source_potentials;

    mesh_manager->load_source_mesh(source_mesh);                                
    setup_source_finite_elements();     
    setup_target_points(); 

    for (size_t source_level_idx = 0; source_level_idx < source_mesh_files.size(); ++source_level_idx)
    {
        const unsigned int current_level_display_name = num_levels - source_level_idx -1 ;

        pcout << Color::cyan << Color::bold
              << "============================================" << Color::reset << std::endl;
        pcout << Color::cyan << Color::bold << "Processing source mesh level " << current_level_display_name
              << " (mesh: " << source_mesh_files[source_level_idx] << ")" << Color::reset << std::endl;
        pcout << Color::cyan << Color::bold
              << "============================================" << Color::reset << std::endl;

        std::string source_level_dir = source_multilevel_dir + "/source_level_" + std::to_string(current_level_display_name);
        fs::create_directories(source_level_dir);

        Vector<double> level_potentials;
        Timer level_timer;
        level_timer.start();

        mesh_manager->load_mesh_at_level(source_mesh, dof_handler_source, source_mesh_files[source_level_idx]);
        setup_source_finite_elements(true); // true for is_multilevel

        if (source_level_idx == 0)
        {
            level_potentials.reinit(target_points.size());
            level_potentials = 0.0; // Initialize potentials for the coarsest level
        }
        else
        {
            level_potentials.reinit(previous_source_potentials.size());
            level_potentials = previous_source_potentials;
        }
        pcout << "  Max iterations: " << solver_params.max_iterations << std::endl;


        sot_solver->setup_source(dof_handler_source, *mapping, *fe_system, source_density, solver_params.quadrature_order);
        sot_solver->setup_target(target_points, target_density); 

        process_epsilon_scaling_for_source_multilevel(level_potentials, source_level_idx, source_level_dir);

        level_timer.stop();
        const double level_time = level_timer.wall_time();
        const unsigned int last_iterations = sot_solver->get_last_iteration_count();
        const double last_threshold = sot_solver->get_last_distance_threshold();

        // Log summary information to file (only rank 0)
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && summary_log.is_open()) {
            summary_log << std::setw(13) << current_level_display_name << " | "
                      << std::setw(17) << last_iterations << " | "
                      << std::setw(8) << std::fixed << std::setprecision(4) << level_time << " | "
                      << std::setw(14) << std::scientific << std::setprecision(6) << last_threshold << "\n";
        }
        
        save_results(level_potentials, source_level_dir + "/potentials", false);

        pcout << Color::blue << Color::bold << "Source level " << current_level_display_name << " summary:" << Color::reset << std::endl;
        pcout << Color::blue << "  Status: Completed" << Color::reset << std::endl;
        pcout << Color::blue << "  Time taken: " << level_timer.wall_time() << " seconds" << Color::reset << std::endl;
        pcout << Color::blue << "  Final number of iterations: " << sot_solver->get_last_iteration_count() << Color::reset << std::endl;
        pcout << Color::blue << "  Final function value: " << sot_solver->get_last_functional_value() << Color::reset << std::endl;
        pcout << Color::blue << "  Results saved in: " << source_level_dir << Color::reset << std::endl;

        previous_source_potentials = level_potentials;
        if (source_level_idx == source_mesh_files.size() - 1)
        {
            final_potentials = level_potentials;
        }
    }

    // Save final results from the finest level
    save_results(final_potentials, source_multilevel_dir + "/potentials", false);

    // Close the log file on rank 0
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && summary_log.is_open()) {
        summary_log.close();
    }

    solver_params.max_iterations = original_max_iterations;
    solver_params.tolerance = original_tolerance;
    solver_params.epsilon = original_regularization;

    global_timer.stop();
    const double total_time = global_timer.wall_time();
    
    // Save global convergence info
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::ofstream conv_info(source_multilevel_dir + "/convergence_info.txt");
        conv_info << "Regularization parameter (λ): " << original_regularization << "\n";
        conv_info << "Final number of iterations: " << sot_solver->get_last_iteration_count() << "\n";
        conv_info << "Final function value: " << sot_solver->get_last_functional_value() << "\n";
        conv_info << "Last threshold value: " << sot_solver->get_last_distance_threshold() << "\n";
        conv_info << "Total execution time: " << total_time << " seconds\n";
        conv_info << "Convergence achieved: " << sot_solver->get_convergence_status() << "\n";
        conv_info << "Number of levels: " << num_levels << "\n";
    }
    
    pcout << "\n"
          << Color::green << Color::bold
          << "============================================" << Color::reset << std::endl;
    pcout << Color::green << Color::bold << "Source multilevel SOT computation completed!" << Color::reset << std::endl;
    pcout << Color::green << Color::bold << "Total computation time: " << total_time << " seconds" << Color::reset << std::endl;
    pcout << Color::green << "Final results saved in: " << source_multilevel_dir << Color::reset << std::endl;
    pcout << Color::green << Color::bold
          << "============================================" << Color::reset << std::endl;
}

// TODO check if there are bug with dim, spacedim, custom distance
template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::run_combined_multilevel()
{
    Timer global_timer;
    global_timer.start();

    pcout << Color::yellow << Color::bold << "Starting combined multilevel SOT computation..." << Color::reset << std::endl;

    // Get source mesh hierarchy files
    std::vector<std::string> source_mesh_files = mesh_manager->get_mesh_hierarchy_files(multilevel_params.source_hierarchy_dir);
    unsigned int n_s_levels = Utilities::MPI::broadcast(mpi_communicator, source_mesh_files.size(), 0);

    // Get target point cloud hierarchy files
    std::vector<std::pair<std::string, std::string>> target_hierarchy_files;
    unsigned int n_t_levels = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        target_hierarchy_files = Utils::get_target_hierarchy_files(multilevel_params.target_hierarchy_dir);
        n_t_levels = target_hierarchy_files.size();
    }
    n_t_levels = Utilities::MPI::broadcast(mpi_communicator, n_t_levels, 0);

    // Calculate total combined levels
    const unsigned int n_c_levels = std::max(n_s_levels, n_t_levels);

    // Backup original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.epsilon;

    // Create output directory
    std::string eps_dir = "output/epsilon_" + Utils::to_scientific_string(original_regularization);
    std::string combined_multilevel_dir = eps_dir + "/combined_multilevel";
    fs::create_directories(combined_multilevel_dir);

    // Create/Open the summary log file on rank 0
    std::ofstream summary_log;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        summary_log.open(combined_multilevel_dir + "/summary_log.txt", std::ios::app);
        // Write header if file is new or empty
        if (summary_log.tellp() == 0) {
            summary_log << "Combined Level | Solver Iterations | Time (s) | Last Threshold\n";
            summary_log << "------------------------------------------------------------\n";
        }
    }

    // Setup epsilon scaling if enabled
    if (solver_params.use_epsilon_scaling && epsilon_scaling_handler) {
        pcout << "Computing epsilon distribution for combined multilevel optimization..." << std::endl;
        epsilon_scaling_handler->compute_epsilon_distribution(n_c_levels);
        epsilon_scaling_handler->print_epsilon_distribution();
    }

    // Initialize state variables
    Vector<double> current_potentials;
    int prev_actual_target_level_num = -1;
    unsigned int prev_source_idx = -1;  // Track previous source index
    double total_softmax_time = 0.0;

    // loading mesh and setup FEs, needed for first iteration in order to load interpolate 
    mesh_manager->load_source_mesh(source_mesh);                                
    setup_source_finite_elements();     

    // Main Loop (Combined Levels)
    for (unsigned int combined_iter = 0; combined_iter < n_c_levels; ++combined_iter) {
        Timer level_timer;
        level_timer.start();

        pcout << "\n" << Color::cyan << Color::bold
              << "============================================" << Color::reset << std::endl;
        pcout << Color::cyan << Color::bold << "Processing combined level " << combined_iter
              << " of " << n_c_levels - 1 << Color::reset << std::endl;

        // Determine Source and Target indices
        unsigned int current_source_idx = 0;
        unsigned int current_target_idx = 0;

        if (n_s_levels >= n_t_levels) {
            current_source_idx = combined_iter;
            current_target_idx = (combined_iter >= (n_s_levels - n_t_levels)) ?
                                 (combined_iter - (n_s_levels - n_t_levels)) : 0;
        } else {
            current_target_idx = combined_iter;
            current_source_idx = (combined_iter >= (n_t_levels - n_s_levels)) ?
                                 (combined_iter - (n_t_levels - n_s_levels)) : 0;
        }

        // Get current files
        const std::string current_source_mesh_file = source_mesh_files[current_source_idx];
        std::string current_target_points_file;
        std::string current_target_density_file;

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            current_target_points_file = target_hierarchy_files[current_target_idx].first;
            current_target_density_file = target_hierarchy_files[current_target_idx].second;
        }

        // Broadcast filenames
        current_target_points_file = Utilities::MPI::broadcast(mpi_communicator, current_target_points_file, 0);
        current_target_density_file = Utilities::MPI::broadcast(mpi_communicator, current_target_density_file, 0);

        // Extract target level number
        int current_actual_target_level_num = -1;
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            size_t level_pos = current_target_points_file.find("level_");
            size_t level_end = current_target_points_file.find("_points");
            if (level_pos != std::string::npos && level_end != std::string::npos) {
                current_actual_target_level_num = std::stoi(current_target_points_file.substr(
                    level_pos + 6, level_end - (level_pos + 6)));
            }
        }
        current_actual_target_level_num = Utilities::MPI::broadcast(mpi_communicator, current_actual_target_level_num, 0);

        // Create level directory
        std::string level_dir = combined_multilevel_dir + "/combined_level_" + std::to_string(combined_iter);
        fs::create_directories(level_dir);

        // Only load source mesh and setup FEs if source level changed
        if (current_source_idx != prev_source_idx) {
            mesh_manager->load_mesh_at_level(source_mesh, dof_handler_source, current_source_mesh_file);
            setup_source_finite_elements(true);
            prev_source_idx = current_source_idx;
        }

        // Save previous target data if not first iteration
        if (combined_iter > 0) {
            target_points_coarse = target_points;
            target_density_coarse = target_density;
        }

        // Load target points
        load_target_points_at_level(current_target_points_file, current_target_density_file);

        // Initialize potentials
        Vector<double> potentials_for_this_level(target_points.size());

        potentials_for_this_level = 0.0;  // Default initialization
        if (combined_iter > 0) {
            bool target_level_changed = (current_actual_target_level_num != prev_actual_target_level_num);
            bool hierarchical_transfer_possible = target_level_changed &&
                                                (prev_actual_target_level_num == current_actual_target_level_num + 1);

            if (hierarchical_transfer_possible) {
                load_hierarchy_data(multilevel_params.target_hierarchy_dir, current_actual_target_level_num);
                Timer transfer_timer;
                transfer_timer.start();

                assign_potentials_by_hierarchy(potentials_for_this_level,
                                            prev_actual_target_level_num,
                                            current_actual_target_level_num,
                                            current_potentials);

                transfer_timer.stop();
                if (multilevel_params.use_softmax_potential_transfer) {
                    total_softmax_time += transfer_timer.wall_time();
                }
            } else if (current_potentials.size() == potentials_for_this_level.size()) {
                potentials_for_this_level = current_potentials;
            }
        }

        // Setup solver
        sot_solver->setup_source(dof_handler_source, *mapping, *fe_system, source_density, solver_params.quadrature_order);
        sot_solver->setup_target(target_points, target_density);

        // Solve with epsilon scaling
        bool solve_successful = true;
        if (solver_params.use_epsilon_scaling && epsilon_scaling_handler) {
            const auto& level_epsilons = epsilon_scaling_handler->get_epsilon_values_for_level(combined_iter);
            if (!level_epsilons.empty()) {
                for (size_t eps_idx = 0; eps_idx < level_epsilons.size(); ++eps_idx) {
                    solver_params.epsilon = level_epsilons[eps_idx];
                    try {
                        sot_solver->solve(potentials_for_this_level, solver_params);
                        if (eps_idx < level_epsilons.size() - 1) {
                            save_results(potentials_for_this_level, level_dir + "/potentials_eps" + std::to_string(eps_idx + 1), false);
                        }
                    } catch (const SolverControl::NoConvergence& exc) {
                        solve_successful = false;
                    }
                }
            } else {
                solver_params.epsilon = original_regularization;
                try {
                    sot_solver->solve(potentials_for_this_level, solver_params);
                } catch (const SolverControl::NoConvergence& exc) {
                    solve_successful = false;
                }
            }
        } else {
            solver_params.epsilon = original_regularization;
            try {
                sot_solver->solve(potentials_for_this_level, solver_params);
            } catch (const SolverControl::NoConvergence& exc) {
                solve_successful = false;
            }
        }

        if (!solve_successful && combined_iter == 0) {
            pcout << Color::red << Color::bold << "Initial iteration failed. Aborting." << Color::reset << std::endl;
            break;
        }

        // Save results
        save_results(potentials_for_this_level, level_dir + "/potentials", false);

        // Update state for next iteration
        current_potentials = potentials_for_this_level;
        prev_actual_target_level_num = current_actual_target_level_num;
        current_distance_threshold = sot_solver->get_last_distance_threshold();

        level_timer.stop();
        const double level_time = level_timer.wall_time();
        const unsigned int last_iterations = sot_solver->get_last_iteration_count();

        // Log summary information to file (only rank 0)
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && summary_log.is_open()) {
            summary_log << std::setw(13) << combined_iter << " | "
                        << std::setw(17) << last_iterations << " | "
                        << std::setw(8) << std::fixed << std::setprecision(4) << level_time << " | "
                        << std::setw(14) << std::scientific << std::setprecision(6) << current_distance_threshold << "\n";
        }
    }
    
    // Save final potentials in the top-level directory
    if (current_potentials.size() > 0) {
        save_results(current_potentials, combined_multilevel_dir + "/potentials", false);
    }

    // Close the log file on rank 0
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && summary_log.is_open()) {
        summary_log.close();
    }

    // Restore original parameters
    solver_params.max_iterations = original_max_iterations;
    solver_params.tolerance = original_tolerance;
    solver_params.epsilon = original_regularization;

    global_timer.stop();
    const double total_time = global_timer.wall_time();
    
    // Save global convergence info
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::ofstream conv_info(combined_multilevel_dir + "/convergence_info.txt");
        conv_info << "Regularization parameter (λ): " << original_regularization << "\n";
        conv_info << "Final number of iterations: " << sot_solver->get_last_iteration_count() << "\n";
        conv_info << "Final function value: " << sot_solver->get_last_functional_value() << "\n";
        conv_info << "Last threshold value: " << current_distance_threshold << "\n";
        conv_info << "Total execution time: " << total_time << " seconds\n";
        conv_info << "Convergence achieved: " << sot_solver->get_convergence_status() << "\n";
        conv_info << "Number of levels: " << n_c_levels << "\n";
    }
    
    pcout << "\n" << Color::green << Color::bold
          << "============================================" << Color::reset << std::endl;
    pcout << Color::green << Color::bold << "Combined multilevel computation completed!" << Color::reset << std::endl;
    pcout << Color::green << Color::bold << "Total computation time: " << total_time << " seconds" << Color::reset << std::endl;
    if (multilevel_params.use_softmax_potential_transfer && total_softmax_time > 0.0) {
        pcout << Color::green << Color::bold << "Total softmax transfer time: " << total_softmax_time
              << " seconds (" << (total_softmax_time / total_time * 100.0) << "%)" << Color::reset << std::endl;
    }
    pcout << Color::green << Color::bold << "============================================" << Color::reset << std::endl;
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::run_multilevel()
{
    pcout << Color::yellow << Color::bold << "Starting multilevel SOT computation (dispatcher)..." << Color::reset << std::endl;

    const bool source_ml_enabled = multilevel_params.source_enabled;
    const bool target_ml_enabled = multilevel_params.target_enabled;

    if (source_ml_enabled && !target_ml_enabled)
    {
        pcout << "Executing Source-Only Multilevel SOT." << std::endl;
        run_source_multilevel();
    }
    else if (!source_ml_enabled && target_ml_enabled)
    {
        pcout << "Executing Target-Only Multilevel SOT." << std::endl;
        run_target_multilevel("", nullptr);
    }
    else if (source_ml_enabled && target_ml_enabled)
    {
        pcout << "Executing Combined Source and Target Multilevel SOT." << std::endl;
        run_combined_multilevel();
    }
    else
    {
        pcout << Color::red << Color::bold
              << "Error: No multilevel strategy enabled (neither source nor target). "
              << "Please enable at least one in parameters.prm and select the 'multilevel_sot' task."
              << Color::reset << std::endl;
    }
}

template <int dim, int spacedim>
template <int d, int s>
typename std::enable_if<d == 3 && s == 3>::type SemiDiscreteOT<dim, spacedim>::run_exact_sot()
{
    pcout << "Running exact SOT computation..." << std::endl;

    ExactSot exact_solver;

    // Set source mesh and target points
    if (!exact_solver.set_source_mesh("output/data_mesh/source.msh"))
    {
        pcout << "Failed to load source mesh for exact SOT" << std::endl;
        return;
    }

    // Load target points from the same location as used in setup_finite_elements
    const std::string directory = "output/data_points";
    if (!exact_solver.set_target_points(directory + "/target_points", io_coding))
    {
        pcout << "Failed to load target points for exact SOT" << std::endl;
        return;
    }

    // Set solver parameters
    exact_solver.set_parameters(
        solver_params.max_iterations,
        solver_params.tolerance);

    // Create output directory
    std::string output_dir = "output/exact_sot";
    fs::create_directories(output_dir);

    // Run the solver
    if (!exact_solver.run())
    {
        pcout << "Exact SOT computation failed" << std::endl;
        return;
    }

    // Save results
    if (!exact_solver.save_results(
            output_dir + "/potentials",
            output_dir + "/points"))
    {
        pcout << "Failed to save exact SOT results" << std::endl;
        return;
    }

    pcout << Color::green << Color::bold << "Exact SOT computation completed successfully" << Color::reset << std::endl;
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::prepare_source_multilevel()
{
    pcout << "Preparing multilevel mesh hierarchy..." << std::endl;

    // Create MeshHierarchyManager instance
    MeshHierarchy::MeshHierarchyManager hierarchy_manager(
        multilevel_params.source_min_vertices,
        multilevel_params.source_max_vertices);

    // Ensure source mesh exists
    const std::string source_mesh_file = "output/data_mesh/source.msh";
    if (!std::filesystem::exists(source_mesh_file))
    {
        pcout << "Source mesh file not found. Please run mesh_generation first." << std::endl;
        return;
    }

    try
    {
        // Create output directory if it doesn't exist
        const std::string hierarchy_dir = multilevel_params.source_hierarchy_dir;
        fs::create_directories(hierarchy_dir);

        // Generate hierarchy
        int num_levels = hierarchy_manager.generateHierarchyFromFile(
            source_mesh_file,
            hierarchy_dir);

        pcout << Color::green << Color::bold << "Successfully generated mesh hierarchy with " << num_levels << " levels" << Color::reset << std::endl;
        pcout << Color::green << "Mesh hierarchy saved in: " << hierarchy_dir << Color::reset << std::endl;
        pcout << Color::green << "Level 1 (finest) vertices: " << multilevel_params.source_max_vertices << Color::reset << std::endl;
        pcout << Color::green << "Coarsest level vertices: ~" << multilevel_params.source_min_vertices << Color::reset << std::endl;
    }
    catch (const std::exception &e)
    {
        pcout << Color::red << Color::bold << "Error: " << e.what() << Color::reset << std::endl;
    }
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::prepare_target_multilevel()
{

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        pcout << "Preparing target point cloud hierarchy..." << std::endl;
        // First ensure we have target points and density
        if (target_points.empty())
        {
            setup_target_finite_elements();
        }

        if (target_points.empty())
        {
            pcout << Color::red << Color::bold << "Error: No target points available" << Color::reset << std::endl;
            return;
        }

        // Create output directory
        fs::create_directories(multilevel_params.target_hierarchy_dir);
        if (multilevel_params.use_python_clustering)
        {
            // Only rank 0 executes the Python script
            pcout << "Using Python script for clustering: " << multilevel_params.python_script_name << std::endl;

            // Construct the Python command
            std::string python_cmd = "python3 ";
            if (multilevel_params.python_script_name.find('/') == std::string::npos)
            {
                if (fs::exists("python_scripts/" + multilevel_params.python_script_name))
                {
                    python_cmd += "python_scripts/";
                }
            }
            python_cmd += multilevel_params.python_script_name;

            pcout << Color::green << Color::bold << "Executing: " << python_cmd << Color::reset << std::endl;
            int result = std::system(python_cmd.c_str());

            if (result != 0)
            {
                pcout << Color::red << Color::bold << "Error: Python script execution failed with code "
                      << result << Color::reset << std::endl;
                return;
            }
            pcout << Color::green << "Python script executed successfully" << Color::reset << std::endl;
        }
        else
        {
            // Use built-in C++ implementation
            PointCloudHierarchy::PointCloudHierarchyManager hierarchy_manager(
                multilevel_params.target_min_points,
                multilevel_params.target_max_points);

            try
            {

                pcout << "Generating hierarchy with " << target_points.size() << " points..." << std::endl;

                // Convert dealii vector to std::vector
                std::vector<double> target_densities(target_points.size());
                for (size_t i = 0; i < target_points.size(); ++i)
                {
                    target_densities[i] = target_density[i];
                }

                int num_levels = hierarchy_manager.generateHierarchy<spacedim>(
                    target_points,
                    target_densities,
                    multilevel_params.target_hierarchy_dir);

                pcout << Color::green << Color::bold << "Successfully generated " << num_levels
                      << " levels of point cloud hierarchy" << Color::reset << std::endl;
            }
            catch (const std::exception &e)
            {
                pcout << Color::red << Color::bold << "Error: " << e.what() << Color::reset << std::endl;
                return;
            }
        }
        // Load the hierarchy data for use in computations
        load_hierarchy_data(multilevel_params.target_hierarchy_dir);
    }
    MPI_Barrier(mpi_communicator);
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::save_results(const Vector<double> &potential, const std::string &filename, bool add_epsilon_prefix)
{
    // Only rank 0 should create directories and write files
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::string full_path;
        if (add_epsilon_prefix)
        {
            // Create epsilon-specific directory
            std::string eps_dir = "output/epsilon_" + Utils::to_scientific_string(solver_params.epsilon);
            fs::create_directories(eps_dir);
            full_path = eps_dir + "/" + filename;
        }
        else
        {
            // Use the filename as is
            full_path = filename;
        }

        // Save potential
        std::vector<double> potential_vec(potential.begin(), potential.end());
        Utils::write_vector(potential_vec, full_path, io_coding);
    }
    // Make sure all processes wait for rank 0 to finish writing
    MPI_Barrier(mpi_communicator);
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::compute_transport_map()
{
    load_meshes();
    setup_finite_elements();

    // Let user select which folder(s) to use
    std::vector<std::string> selected_folders;
    try
    {
        selected_folders = Utils::select_folder();
    }
    catch (const std::exception &e)
    {
        pcout << "Error: " << e.what() << std::endl;
        return;
    }

    // Create transport map approximator
    OptimalTransportPlanSpace::OptimalTransportPlan<spacedim> transport_plan;

    // Get source points and density (serial version)
    std::vector<Point<spacedim>> source_points;
    Vector<double> source_density;

    const std::string directory = "output/data_points";
    bool points_loaded = false;
    if (Utils::read_vector(source_points, directory + "/source_points", io_coding))
    {
        points_loaded = true;
        std::cout << "Source points loaded from file" << std::endl;
    }

    if (!points_loaded)
    {
        std::map<types::global_dof_index, Point<spacedim>> support_points_source;
        DoFTools::map_dofs_to_support_points(*mapping, dof_handler_source, support_points_source);
        source_points.clear();
        for (const auto &point_pair : support_points_source)
        {
            source_points.push_back(point_pair.second);
        }

        // Save the computed points
        Utils::write_vector(source_points, directory + "/source_points", io_coding);
        std::cout << "Source points computed and saved to file" << std::endl;
    }

    source_density.reinit(source_points.size());
    source_density = 1.0 / source_points.size();

    // Convert densities to vectors
    std::vector<double> source_density_vec(source_density.begin(), source_density.end());
    std::vector<double> target_density_vec(target_density.begin(), target_density.end());

    std::cout << "Setup complete with " << source_points.size() << " source points and "
              << target_points.size() << " target points" << std::endl;

    // Process each selected folder
    for (const auto &selected_folder : selected_folders)
    {
        pcout << "\nProcessing folder: " << selected_folder << std::endl;

        // Read potentials from selected folder's results
        std::vector<double> potentials_vec;
        bool success = Utils::read_vector(potentials_vec, "output/" + selected_folder + "/potentials", io_coding);
        if (!success)
        {
            pcout << "Failed to read potentials from output/" + selected_folder + "/potentials" << std::endl;
            continue; // Skip to next folder
        }

        if (potentials_vec.size() != target_points.size())
        {
            pcout << Color::red << Color::bold << "Error: Mismatch between potentials size (" << potentials_vec.size()
                  << ") and target points size (" << target_points.size() << ")" << Color::reset << std::endl;
            continue; // Skip to next folder
        }

        // Extract regularization parameter from folder name
        std::string eps_str = selected_folder.substr(selected_folder.find("epsilon_") + 8);
        double epsilon = std::stod(eps_str);

        // Create output directory
        const std::string output_dir = "output/" + selected_folder + "/transport_map";
        fs::create_directories(output_dir);

        // Convert potentials to dealii::Vector format
        Vector<double> potentials_dealii(potentials_vec.size());
        std::copy(potentials_vec.begin(), potentials_vec.end(), potentials_dealii.begin());

        // Set up the transport plan
        transport_plan.set_source_measure(source_points, source_density_vec);
        transport_plan.set_target_measure(target_points, target_density_vec);
        transport_plan.set_potential(potentials_dealii, epsilon);
        transport_plan.set_truncation_radius(transport_map_params.truncation_radius);

        // Try different strategies and save results
        for (const auto &strategy : transport_plan.get_available_strategies())
        {
            pcout << "Computing transport map using " << strategy << " strategy..." << std::endl;

            transport_plan.set_strategy(strategy);
            transport_plan.compute_map();
            transport_plan.save_map(output_dir + "/" + strategy);

            pcout << "Results saved in " << output_dir + "/" + strategy << std::endl;
        }
    }

    if (selected_folders.size() > 1)
    {
        pcout << "\nCompleted transport map computation for all selected folders." << std::endl;
    }
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::compute_power_diagram()
{
    load_meshes();
    setup_finite_elements();

    // Let user select which folder(s) to use
    std::vector<std::string> selected_folders;
    try
    {
        selected_folders = Utils::select_folder();
    }
    catch (const std::exception &e)
    {
        pcout << "Error: " << e.what() << std::endl;
        return;
    }

    // Process each selected folder
    for (const auto &selected_folder : selected_folders)
    {
        pcout << "\nProcessing folder: " << selected_folder << std::endl;

        // Read potentials from selected folder's results
        std::vector<double> potentials_vec;
        bool success = Utils::read_vector(potentials_vec, "output/" + selected_folder + "/potentials", io_coding);
        if (!success)
        {
            pcout << "Failed to read potentials from output/" << selected_folder << "/potentials" << std::endl;
            continue; // Skip to next folder
        }

        if (potentials_vec.size() != target_points.size())
        {
            pcout << Color::red << Color::bold << "Error: Mismatch between potentials size (" << potentials_vec.size()
                  << ") and target points size (" << target_points.size() << ")" << Color::reset << std::endl;
            continue; // Skip to next folder
        }

        // Convert to dealii::Vector
        Vector<double> potentials(potentials_vec.size());
        std::copy(potentials_vec.begin(), potentials_vec.end(), potentials.begin());

        // Create output directory
        const std::string output_dir = "output/" + selected_folder + "/power_diagram_" + power_diagram_params.implementation;
        fs::create_directories(output_dir);

        // Create power diagram using factory function based on parameter choice
        std::unique_ptr<PowerDiagramSpace::PowerDiagramBase<dim, spacedim>> power_diagram;

        try
        {
            if (power_diagram_params.implementation == "geogram")
            {
                if constexpr (dim == 3 and spacedim==dim)
                {
                    power_diagram = PowerDiagramSpace::create_power_diagram<dim, spacedim>(
                        "geogram",
                        nullptr,
                        "output/data_mesh/source.msh");
                    pcout << "Using Geogram implementation for power diagram" << std::endl;
                }
                else
                {
                    pcout << "Geogram implementation is only available for 3D problems" << std::endl;
                    pcout << "Falling back to Deal.II implementation" << std::endl;
                    power_diagram = PowerDiagramSpace::create_power_diagram<dim, spacedim>(
                        "dealii",
                        &source_mesh);
                }
            }
            else
            {
                power_diagram = PowerDiagramSpace::create_power_diagram<dim, spacedim>(
                    "dealii",
                    &source_mesh);
                pcout << "Using Deal.II implementation for power diagram" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            pcout << Color::red << Color::bold << "Failed to initialize " << power_diagram_params.implementation
                  << " implementation: " << e.what() << Color::reset << std::endl;
            if (power_diagram_params.implementation == "geogram")
            {
                pcout << "Falling back to Deal.II implementation" << std::endl;
                power_diagram = PowerDiagramSpace::create_power_diagram<dim, spacedim>(
                    "dealii",
                    &source_mesh);
            }
            else
            {
                throw;
            }
        }

        // Set generators and compute power diagram
        power_diagram->set_generators(target_points, potentials);
        power_diagram->compute_power_diagram();
        power_diagram->compute_cell_centroids();

        // Save results
        power_diagram->save_centroids_to_file(output_dir + "/centroids");
        power_diagram->output_vtu(output_dir + "/power_diagram");

        pcout << "Power diagram computation completed for " << selected_folder << std::endl;
        pcout << "Results saved in " << output_dir << std::endl;
    }

    if (selected_folders.size() > 1)
    {
        pcout << "\nCompleted power diagram computation for all selected folders." << std::endl;
    }
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::save_discrete_measures()
{
    load_meshes();
    setup_finite_elements();

    // Create output directory
    const std::string directory = "output/discrete_measures";
    fs::create_directories(directory);

    // Get quadrature points and weights
    auto quadrature = Utils::create_quadrature_for_mesh<dim, spacedim>(source_mesh, solver_params.quadrature_order);

    FEValues<dim, spacedim> fe_values(*mapping, *fe_system, *quadrature,
                           update_values | update_quadrature_points | update_JxW_values);

    // Count total number of quadrature points
    const unsigned int n_q_points = quadrature->size();
    const unsigned int total_q_points = source_mesh.n_active_cells() * n_q_points;

    // Prepare vectors for quadrature data
    std::vector<Point<spacedim>> quad_points;
    std::vector<double> quad_weights;
    std::vector<double> density_values;
    quad_points.reserve(total_q_points);
    quad_weights.reserve(total_q_points);
    density_values.reserve(total_q_points);

    // Get density values at DoF points
    std::vector<double> local_density_values(n_q_points);

    // Loop over cells to collect quadrature data
    for (const auto &cell : dof_handler_source.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(source_density, local_density_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            quad_points.push_back(fe_values.quadrature_point(q));
            quad_weights.push_back(fe_values.JxW(q));
            density_values.push_back(local_density_values[q]);
        }
    }

    // Save quadrature data
    Utils::write_vector(quad_points, directory + "/quadrature_points", io_coding);
    Utils::write_vector(quad_weights, directory + "/quadrature_weights", io_coding);
    Utils::write_vector(density_values, directory + "/density_values", io_coding);

    // Save target measure data
    Utils::write_vector(target_points, directory + "/target_points", io_coding);
    std::vector<double> target_density_vec(target_density.begin(), target_density.end());
    Utils::write_vector(target_density_vec, directory + "/target_density", io_coding);

    // Save metadata
    std::ofstream meta(directory + "/metadata.txt");
    meta << "Dimension: " << dim << "\n"
         << "Space dimension: " << spacedim << "\n"
         << "Number of quadrature points per cell: " << n_q_points << "\n"
         << "Total number of quadrature points: " << total_q_points << "\n"
         << "Number of target points: " << target_points.size() << "\n"
         << "Quadrature order: " << solver_params.quadrature_order << "\n"
         << "Using tetrahedral mesh: " << source_params.use_tetrahedral_mesh << "\n";
    meta.close();

    pcout << "Discrete measures data saved in " << directory << std::endl;
    pcout << "Total quadrature points: " << total_q_points << std::endl;
    pcout << "Number of target points: " << target_points.size() << std::endl;
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::save_interpolated_fields()
{
    pcout << Color::yellow << Color::bold << "Starting field interpolation visualization..." << Color::reset << std::endl;

    std::string output_dir = "output/interpolated_fields";
    fs::create_directories(output_dir);

    std::string source_field_name = source_params.density_field_name;
    std::string target_field_name = target_params.density_field_name;

    pcout << "Using source field name: " << source_field_name << std::endl;
    pcout << "Using target field name: " << target_field_name << std::endl;

    pcout << "\n"
          << Color::cyan << Color::bold << "Processing base source mesh" << Color::reset << std::endl;

    mesh_manager->load_source_mesh(source_mesh);
    setup_source_finite_elements(false); // false = not multilevel

    std::string base_source_dir = output_dir + "/base_source";
    fs::create_directories(base_source_dir);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        DataOut<dim, spacedim> data_out;
        data_out.attach_dof_handler(dof_handler_source);
        data_out.add_data_vector(source_density, source_field_name);
        data_out.build_patches();

        std::string vtk_filename = base_source_dir + "/" + source_field_name + ".vtk";
        std::ofstream output(vtk_filename);
        data_out.write_vtk(output);

        pcout << "Saved interpolated field for base source mesh to: " << vtk_filename << std::endl;
    }

    pcout << "\n"
          << Color::cyan << Color::bold << "Processing base target mesh" << Color::reset << std::endl;

    mesh_manager->load_target_mesh(target_mesh);
    setup_target_finite_elements();

    std::string base_target_dir = output_dir + "/base_target";
    fs::create_directories(base_target_dir);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        DataOut<dim, spacedim> data_out;
        data_out.attach_dof_handler(dof_handler_target);
        data_out.add_data_vector(target_density, target_field_name);
        data_out.build_patches();

        std::string vtk_filename = base_target_dir + "/" + target_field_name + ".vtk";
        std::ofstream output(vtk_filename);
        data_out.write_vtk(output);

        pcout << "Saved interpolated field for base target mesh to: " << vtk_filename << std::endl;
    }

    if (multilevel_params.source_enabled)
    {
        pcout << "\n"
              << Color::cyan << Color::bold << "Processing multilevel source meshes" << Color::reset << std::endl;

        std::vector<std::string> source_mesh_files;
        unsigned int num_levels = 0;

        source_mesh_files = mesh_manager->get_mesh_hierarchy_files(multilevel_params.source_hierarchy_dir);
        if (source_mesh_files.empty())
        {
            pcout << Color::red << Color::bold << "No source mesh hierarchy found. Please run prepare_source_multilevel first." << Color::reset << std::endl;
        }
        else
        {
            num_levels = source_mesh_files.size();
            pcout << "Found " << num_levels << " levels in the source mesh hierarchy" << std::endl;

            for (size_t level = 0; level < source_mesh_files.size(); ++level)
            {
                const unsigned int level_name = num_levels - level - 1;
                pcout << "\n"
                      << Color::cyan << Color::bold
                      << "Processing source mesh level " << level_name
                      << " (mesh: " << source_mesh_files[level] << ")" << Color::reset << std::endl;

                std::string level_dir = output_dir + "/source_level_" + std::to_string(level_name);
                fs::create_directories(level_dir);

                mesh_manager->load_mesh_at_level(source_mesh, dof_handler_source, source_mesh_files[level]);

                setup_source_finite_elements(true);

                if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                {
                    DataOut<dim, spacedim> data_out;
                    data_out.attach_dof_handler(dof_handler_source);
                    data_out.add_data_vector(source_density, source_field_name);
                    data_out.build_patches();

                    std::string vtk_filename = level_dir + "/" + source_field_name + ".vtk";
                    std::ofstream output(vtk_filename);
                    data_out.write_vtk(output);

                    pcout << "Saved interpolated field for source level " << level_name
                          << " to: " << vtk_filename << std::endl;
                }
            }
        }
    }

    pcout << "\n"
          << Color::green << Color::bold
          << "Field interpolation visualization completed!" << Color::reset << std::endl;
    pcout << "Results saved in: " << output_dir << std::endl;
}

template <int dim, int spacedim>
void SemiDiscreteOT<dim, spacedim>::run()
{
    param_manager.print_parameters();

    if (solver_params.use_epsilon_scaling)
    {
        epsilon_scaling_handler = std::make_unique<EpsilonScalingHandler>(
            mpi_communicator,
            solver_params.epsilon,
            solver_params.epsilon_scaling_factor,
            solver_params.epsilon_scaling_steps);
    }

    if (selected_task == "mesh_generation")
    {
        mesh_generation();
    }
    else if (selected_task == "load_meshes")
    {
        load_meshes();
    }
    else if (selected_task == "sot")
    {
        load_meshes();
        run_sot();
    }
    else if (selected_task == "prepare_source_multilevel")
    {
        prepare_source_multilevel();
    }
    else if (selected_task == "prepare_target_multilevel")
    {
        mesh_manager->load_target_mesh(target_mesh);
        prepare_target_multilevel();
    }
    else if (selected_task == "prepare_multilevel")
    {
        if (multilevel_params.source_enabled)
            prepare_source_multilevel();
        if (multilevel_params.target_enabled)
        {
            mesh_manager->load_target_mesh(target_mesh);
            prepare_target_multilevel();
        }
    }
    else if (selected_task == "target_multilevel")
    {
        run_target_multilevel();
    }
    else if (selected_task == "source_multilevel")
    {
        run_source_multilevel();
    }
    else if (selected_task == "multilevel")
    {
        run_multilevel();
    }
    else if (selected_task == "exact_sot")
    {
        if constexpr (dim == 3 && spacedim==3) {
            load_meshes();
            setup_finite_elements();
            run_exact_sot();
        }
        else
        {
            pcout << "Exact SOT is only available for 3D problems" << std::endl;
        }
    }
    else if (selected_task == "power_diagram")
    {
        compute_power_diagram();
    }
    else if (selected_task == "map")
    {
        compute_transport_map();
    }
    else if (selected_task == "save_discrete_measures")
    {
        save_discrete_measures();
    }
    else if (selected_task == "save_interpolated_fields")
    {
        save_interpolated_fields();
    }
    else
    {
        pcout << "No valid task selected" << std::endl;
    }
}


// Explicit template instantiation
template class SemiDiscreteOT<2>;
template class SemiDiscreteOT<3>;
template class SemiDiscreteOT<2, 3>;
