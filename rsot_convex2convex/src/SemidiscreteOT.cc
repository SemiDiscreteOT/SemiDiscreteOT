#include "SemidiscreteOT.h"
#include "PowerDiagram.h"
#include "utils.h"
#include "ExactSot.h"
#include "OptimalTransportPlan.h"
#include "PointCloudHierarchy.h"
#include "SoftmaxRefinement.h"
#include <deal.II/base/timer.h>
#include <filesystem>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/vector_operations_internal.h>
#include <deal.II/grid/grid_tools.h>
#include "SotSolver.h"
#include "ColorDefinitions.h"
namespace fs = std::filesystem;


template <int dim>
SemidiscreteOT<dim>::SemidiscreteOT(const MPI_Comm &comm)
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
    , mesh_manager(std::make_unique<MeshManager<dim>>(comm))
    , sot_solver(std::make_unique<SotSolver<dim>>(comm))
{
    // Initialize with default hexahedral elements
    fe_system = std::make_unique<FE_Q<dim>>(1);
    mapping = std::make_unique<MappingQ1<dim>>();
    
}

template <int dim>
void SemidiscreteOT<dim>::mesh_generation()
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


template <int dim>
void SemidiscreteOT<dim>::load_meshes()
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

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        pcout << "Target mesh: " << target_mesh.n_active_cells() << " cells, " 
              << target_mesh.n_vertices() << " vertices" << std::endl;
    }
}

template <int dim>
std::vector<std::pair<std::string, std::string>> SemidiscreteOT<dim>::get_target_hierarchy_files() const
{
    return Utils::get_target_hierarchy_files(multilevel_params.target_hierarchy_dir);
}

template <int dim>
void SemidiscreteOT<dim>::load_target_points_at_level(
    const std::string& points_file, 
    const std::string& weights_file)
{
    pcout << "Loading target points from: " << points_file << std::endl;
    pcout << "Loading target weights from: " << weights_file << std::endl;
    
    std::vector<Point<dim>> local_target_points;
    std::vector<double> local_weights;
    bool load_success = true;
    
    // Only rank 0 reads files
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        // Read points
        if (!Utils::read_vector(local_target_points, points_file, io_coding)) {
            pcout << Color::red << Color::bold << "Error: Cannot read points file: " << points_file << Color::reset << std::endl;
            load_success = false;
        }
        
        // Read weights
        if (load_success && !Utils::read_vector(local_weights, weights_file, io_coding)) {
            pcout << Color::red << Color::bold << "Error: Cannot read weights file: " << weights_file << Color::reset << std::endl;
            load_success = false;
        }
    }
    
    // Broadcast success status
    load_success = Utilities::MPI::broadcast(mpi_communicator, load_success, 0);
    if (!load_success) {
        throw std::runtime_error("Failed to load target points or weights");
    }
    
    // Broadcast sizes
    unsigned int n_points = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        n_points = local_target_points.size();
    }
    n_points = Utilities::MPI::broadcast(mpi_communicator, n_points, 0);
    
    // Resize containers on non-root ranks
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
        local_target_points.resize(n_points);
        local_weights.resize(n_points);
    }
    
    // Broadcast data
    for (unsigned int i = 0; i < n_points; ++i) {
        local_target_points[i] = Utilities::MPI::broadcast(mpi_communicator, local_target_points[i], 0);
        local_weights[i] = Utilities::MPI::broadcast(mpi_communicator, local_weights[i], 0);
    }
    
    // Update class members
    target_points = std::move(local_target_points);
    target_density.reinit(n_points);
    for (unsigned int i = 0; i < n_points; ++i) {
        target_density[i] = local_weights[i];
    }
    
    pcout << Color::green << "Successfully loaded " << n_points << " target points at this level" << Color::reset << std::endl;
}

template <int dim>
void SemidiscreteOT<dim>::load_hierarchy_data(const std::string& hierarchy_dir, int specific_level) {
    has_hierarchy_data_ = Utils::load_hierarchy_data<dim>(hierarchy_dir, child_indices_, specific_level, mpi_communicator, pcout);
}

template <int dim>
void SemidiscreteOT<dim>::setup_source_finite_elements()
{
    // Check if we're using tetrahedral meshes
    bool use_simplex = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh);

    if (use_simplex) {
        fe_system = std::make_unique<FE_SimplexP<dim>>(1);
        mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    } else {
        fe_system = std::make_unique<FE_Q<dim>>(1);
        mapping = std::make_unique<MappingQ1<dim>>();
    }

    dof_handler_source.distribute_dofs(*fe_system);

    IndexSet locally_owned_dofs = dof_handler_source.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler_source, locally_relevant_dofs);

    source_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    source_density = 1.0;
    source_density.update_ghost_values();  // Ensure ghost values are updated

    // Create appropriate quadrature
    std::unique_ptr<Quadrature<dim>> quadrature;
    if (use_simplex) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(solver_params.quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(solver_params.quadrature_order);
    }

    // Debug info about parallel distribution
    unsigned int n_locally_owned = 0;
    for (const auto &cell : dof_handler_source.active_cell_iterators()) {
        if (cell->is_locally_owned())
            ++n_locally_owned;
    }
    const unsigned int n_total_owned =
        Utilities::MPI::sum(n_locally_owned, mpi_communicator);

    pcout << "Total cells: " << source_mesh.n_active_cells()
          << ", Locally owned on proc " << this_mpi_process
          << ": " << n_locally_owned
          << ", Sum of owned: " << n_total_owned << std::endl;

    // Verify normalization
    double local_l1_norm = 0.0;
    unsigned int local_cell_count = 0;
    FEValues<dim> fe_values(*mapping, *fe_system, *quadrature,
                           update_values | update_JxW_values);
    std::vector<double> density_values(quadrature->size());

    for (const auto &cell : dof_handler_source.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        local_cell_count++;
        fe_values.reinit(cell);
        fe_values.get_function_values(source_density, density_values);

        for (unsigned int q = 0; q < quadrature->size(); ++q)
        {
            local_l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
        }
    }

    // Debug info about local contributions
    std::cout << "Process " << this_mpi_process
          << " processed " << local_cell_count << " cells"
          << " with local L1 norm: " << local_l1_norm << std::endl;

    const double global_l1_norm =
        Utilities::MPI::sum(local_l1_norm, mpi_communicator);
    pcout << "Source density L1 norm: " << global_l1_norm << std::endl;
    source_density /= global_l1_norm; // Normalize to mass 1
}

template <int dim>
void SemidiscreteOT<dim>::setup_target_finite_elements()
{
    // Load or compute target points (shared across all processes)
    const std::string directory = "output/data_points";
    bool points_loaded = false;
    target_points.clear();  // Clear on all processes

    // Only root process reads/computes points
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        if (Utils::read_vector(target_points, directory + "/target_points", io_coding)) {
            points_loaded = true;
            pcout << "Target points loaded from file" << std::endl;
        } else {
            dof_handler_target.distribute_dofs(*fe_system);
            std::map<types::global_dof_index, Point<dim>> support_points_target;
            DoFTools::map_dofs_to_support_points(*mapping, dof_handler_target, support_points_target);
            for (const auto &point_pair : support_points_target) {
                target_points.push_back(point_pair.second);
            }
            Utils::write_vector(target_points, directory + "/target_points", io_coding);
            points_loaded = true;
            pcout << "Target points computed and saved to file" << std::endl;
        }
    }

    // Broadcast success flag
    points_loaded = Utilities::MPI::broadcast(mpi_communicator, points_loaded, 0);
    if (!points_loaded) {
        throw std::runtime_error("Failed to load or compute target points");
    }

    // Broadcast target points size first
    unsigned int n_points = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        n_points = target_points.size();
    }
    n_points = Utilities::MPI::broadcast(mpi_communicator, n_points, 0);

    // Resize target_points on non-root processes
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
        target_points.resize(n_points);
    }

    // Broadcast the actual points
    for (unsigned int i = 0; i < n_points; ++i) {
        target_points[i] = Utilities::MPI::broadcast(mpi_communicator, target_points[i], 0);
    }

    // Initialize target density (shared across all processes)
    target_density.reinit(target_points.size());
    target_density = 1.0 / target_points.size();
    pcout << "Setup complete with " << target_points.size() << " target points" << std::endl;
}


template <int dim>
void SemidiscreteOT<dim>::setup_finite_elements()
{
    // Check if we're using tetrahedral meshes
    bool use_simplex = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh);

    if (use_simplex) {
        fe_system = std::make_unique<FE_SimplexP<dim>>(1);
        mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    } else {
        fe_system = std::make_unique<FE_Q<dim>>(1);
        mapping = std::make_unique<MappingQ1<dim>>();
    }

    setup_source_finite_elements();
    setup_target_finite_elements();
}

template <int dim>
void SemidiscreteOT<dim>::setup_multilevel_finite_elements()
{
    // Check if we're using tetrahedral meshes
    bool use_simplex = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh);
    // Only distribute DoFs for source mesh
    dof_handler_source.distribute_dofs(*fe_system);

    // Initialize source density
    IndexSet locally_owned_dofs = dof_handler_source.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler_source, locally_relevant_dofs);

    source_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    source_density = 1.0;
    source_density.update_ghost_values();

    // Normalize source density
    double local_l1_norm = 0.0;
    std::unique_ptr<Quadrature<dim>> quadrature;
    if (use_simplex) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(solver_params.quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(solver_params.quadrature_order);
    }

    FEValues<dim> fe_values(*mapping, *fe_system, *quadrature,
                           update_values | update_JxW_values);
    std::vector<double> local_density_values(quadrature->size());

    for (const auto &cell : dof_handler_source.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(source_density, local_density_values);

        for (unsigned int q = 0; q < quadrature->size(); ++q) {
            local_l1_norm += std::abs(local_density_values[q]) * fe_values.JxW(q);
        }
    }

    const double global_l1_norm = Utilities::MPI::sum(local_l1_norm, mpi_communicator);
    source_density /= global_l1_norm;
    source_density.update_ghost_values();

    pcout << "Source mesh finite elements initialized with " 
          << dof_handler_source.n_dofs() << " DoFs" << std::endl;
}

template <int dim>
void SemidiscreteOT<dim>::setup_target_points()
{
    mesh_manager->load_target_mesh(target_mesh);
    setup_target_finite_elements();
}

template <int dim>
void SemidiscreteOT<dim>::assign_weights_by_hierarchy(
    Vector<double>& weights, int coarse_level, int fine_level, const Vector<double>& prev_weights) {
    
    if (!has_hierarchy_data_ || coarse_level < 0 || fine_level < 0) {
        std::cerr << "Invalid hierarchy levels for weight assignment" << std::endl;
        return;
    }
    
    // Direct assignment if same level
    if (coarse_level == fine_level) {
        weights = prev_weights;
        return;
    }

    // Initialize weights for current level
    weights.reinit(target_points.size());

    if (multilevel_params.use_softmax_weight_transfer) {
        pcout << "Applying softmax-based weight assignment from level " << coarse_level
              << " to level " << fine_level << std::endl;
        pcout << "Source points: " << prev_weights.size() 
              << ", Target points: " << target_points.size() << std::endl;

        // Create SoftmaxRefinement instance
        SoftmaxRefinement<dim> softmax_refiner(
            mpi_communicator,
            dof_handler_source,
            *mapping,
            *fe_system,
            source_density,
            solver_params.quadrature_order,
            current_distance_threshold);

        // Apply softmax refinement
        weights = softmax_refiner.compute_refinement(
            target_points,           // target_points_fine
            target_density,         // target_density_fine
            target_points_coarse,   // target_points_coarse
            target_density_coarse,  // target_density_coarse
            prev_weights,          // weights_coarse
            solver_params.regularization_param,
            fine_level,
            child_indices_);
        
        pcout << "Softmax-based weight assignment completed." << std::endl;
    }
    else {
        pcout << "Applying direct weight assignment from level " << coarse_level
              << " to level " << fine_level << std::endl;
        pcout << "Source points: " << prev_weights.size() 
              << ", Target points: " << target_points.size() << std::endl;

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            // Going from coarse to fine: Each child gets its parent's weight
            #pragma omp parallel for
            for (size_t j = 0; j < prev_weights.size(); ++j) {
                const auto& children = child_indices_[fine_level][j];
                for (size_t child : children) {
                    weights[child] = prev_weights[j];
                }
            }
        }

        // Broadcast the weights to all processes
        Utilities::MPI::broadcast(mpi_communicator, weights, 0);
    }
}


// run single sot optimization with epsilon scaling
template <int dim>
void SemidiscreteOT<dim>::run_sot()
{
    Timer timer;
    timer.start();

    setup_finite_elements();

    pcout << Color::yellow << Color::bold << "Starting SOT optimization with " << target_points.size()
          << " target points and " << source_density.size() << " source points" << Color::reset << std::endl;

    // Configure solver parameters
    ParameterManager::SolverParameters& solver_config = solver_params;

    // Set up source measure
    sot_solver->setup_source(dof_handler_source,
                           *mapping,
                           *fe_system,
                           source_density,
                           solver_config.quadrature_order);

    // Set up target measure
    sot_solver->setup_target(target_points, target_density);

    Vector<double> weights(target_points.size());

    if (solver_config.use_epsilon_scaling && epsilon_scaling_handler) {
        pcout << "Using epsilon scaling with EpsilonScalingHandler:" << std::endl
              << "  Initial epsilon: " << solver_config.regularization_param << std::endl
              << "  Scaling factor: " << solver_config.epsilon_scaling_factor << std::endl
                  << "  Number of steps: " << solver_config.epsilon_scaling_steps << std::endl;
        // Compute epsilon distribution for a single level
        std::vector<std::vector<double>> epsilon_distribution = 
            epsilon_scaling_handler->compute_epsilon_distribution(1, true, false);
        
        if (!epsilon_distribution.empty() && !epsilon_distribution[0].empty()) {
            const auto& epsilon_sequence = epsilon_distribution[0];
            
            // Run optimization for each epsilon value
            for (size_t i = 0; i < epsilon_sequence.size(); ++i) {
                pcout << "\nEpsilon scaling step " << i + 1 << "/" << epsilon_sequence.size()
                      << " (λ = " << epsilon_sequence[i] << ")" << std::endl;
                
                solver_config.regularization_param = epsilon_sequence[i];
                
                try {
                    sot_solver->solve(weights, solver_config);
                    
                    // Save intermediate results
                    if (i < epsilon_sequence.size() - 1) {
                        std::string eps_suffix = "_eps" + std::to_string(i + 1);
                        save_results(weights, "weights" + eps_suffix);
                    }
                    
                } catch (const SolverControl::NoConvergence& exc) {
                    if (exc.last_step >= solver_params.max_iterations) {
                        pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << i + 1
                              << " (epsilon=" << epsilon_sequence[i] << "): Max iterations reached"
                              << Color::reset << std::endl;
                    }
                }
            }
        } 
    } else {
        // Run single optimization with original epsilon
        try {
            sot_solver->solve(weights, solver_config);
        } catch (const SolverControl::NoConvergence& exc) {
            pcout << Color::red << Color::bold << "Warning: Optimization did not converge." << Color::reset << std::endl;
        }
    }

    // Save final results
    save_results(weights, "weights");

    timer.stop();
    pcout << "\n" << Color::green << Color::bold << "SOT optimization completed in " << timer.wall_time() << " seconds" << Color::reset << std::endl;
}

template <int dim>
void SemidiscreteOT<dim>::run_target_multilevel(
    const std::string& source_mesh_file,
    Vector<double>* output_weights,
    bool save_results_to_files)
{
    Timer global_timer;
    global_timer.start();
    
    if (save_results_to_files) {
        pcout << Color::yellow << Color::bold << "Starting target point cloud multilevel SOT computation..." << Color::reset << std::endl;
    } else {
        pcout << "Running target multilevel optimization for source mesh: " << source_mesh_file << std::endl;
    }
    
    // Load source mesh based on input parameters
    if (source_mesh_file.empty()) {
        mesh_manager->load_source_mesh(source_mesh);
    } else {
        mesh_manager->load_mesh_at_level(source_mesh, dof_handler_source, source_mesh_file);
    }
    setup_source_finite_elements();
    
    // Get target point cloud hierarchy files (sorted from coarsest to finest)
    unsigned int num_levels = 0;
    std::vector<std::pair<std::string, std::string>> hierarchy_files;
    if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        try {
            hierarchy_files = Utils::get_target_hierarchy_files(multilevel_params.target_hierarchy_dir);
        } catch (const std::exception& e) {
            pcout << Color::red << Color::bold << "Error: " << e.what() << Color::reset << std::endl;
            pcout << "Please run prepare_target_multilevel first." << std::endl;
            return;
        }
    
        if (hierarchy_files.empty()) {
            pcout << "No target point cloud hierarchy found. Please run prepare_target_multilevel first." << std::endl;
            return;
        }
        num_levels = hierarchy_files.size();
    }

    num_levels = Utilities::MPI::broadcast(mpi_communicator, num_levels, 0);
    
    // Initialize hierarchy data structure but don't load data yet
    if (!has_hierarchy_data_) {
        pcout << "Initializing hierarchy data structure..." << std::endl;
        load_hierarchy_data(multilevel_params.target_hierarchy_dir, -1);
    }
    
    if (!has_hierarchy_data_) {
        pcout << "Failed to initialize hierarchy data. Cannot proceed with multilevel computation." << std::endl;
        return;
    }
    
    // Store original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.regularization_param;
    
    // Setup epsilon scaling if enabled
    std::vector<std::vector<double>> epsilon_distribution;
    if (solver_params.use_epsilon_scaling && epsilon_scaling_handler) {
        pcout << "Computing epsilon distribution for target multilevel optimization..." << std::endl;
        epsilon_distribution = epsilon_scaling_handler->compute_epsilon_distribution(
            num_levels, true, false);
        epsilon_scaling_handler->print_epsilon_distribution();
    }
    
    // Vector to store current weights solution
    Vector<double> level_weights;
    
    // Configure solver parameters for this level
    ParameterManager::SolverParameters& solver_config = solver_params;
    
    // Set up source measure (this remains constant across levels)
    sot_solver->setup_source(dof_handler_source,
                           *mapping,
                           *fe_system,
                           source_density,
                           solver_config.quadrature_order);
    
    // Process each level of the hierarchy (from coarsest to finest)
    for (size_t level = 0; level < num_levels; ++level) {
        int level_number = 0;
        std::string level_output_dir;
        std::string points_file;
        std::string weights_file;
        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            const auto& hierarchy_file = hierarchy_files[level];
            points_file = hierarchy_file.first;
            weights_file = hierarchy_file.second;
        
            // Extract level number from filename
            std::string level_num = points_file.substr(
                points_file.find("level_") + 6, 
                points_file.find("_points") - points_file.find("level_") - 6
            );
            level_number = std::stoi(level_num);
        
            // Create output directory for this level (if saving results)
            if (save_results_to_files) {
                level_output_dir = multilevel_params.output_prefix + "/level_" + level_num;
                fs::create_directories(level_output_dir);
            }
        }
        level_number = Utilities::MPI::broadcast(mpi_communicator, level_number, 0);
        
        pcout << "\n" << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;
        pcout << Color::magenta << Color::bold << "Processing target point cloud level " << level_number << Color::reset << std::endl;
        pcout << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;

        // Load hierarchy data for this level only
        load_hierarchy_data(multilevel_params.target_hierarchy_dir, level_number);
        
        // Load target points for this level
        if (level > 0) {
            target_points_coarse = target_points;
            target_density_coarse = target_density;
        }
        load_target_points_at_level(points_file, weights_file);
        pcout << "Target points loaded for level " << level_number << std::endl;
        pcout << "Target points size: " << target_points.size() << std::endl;
        
        // Set up target measure for this level
        sot_solver->setup_target(target_points, target_density);
        
        // Initialize weights for this level
        Vector<double> current_level_weights(target_points.size());
        if (level > 0) {
            // Use hierarchy-based weight transfer from previous level
            assign_weights_by_hierarchy(current_level_weights, level_number+1, level_number, level_weights);
        }

        // Apply epsilon scaling for this level if enabled
        if (solver_params.use_epsilon_scaling && epsilon_scaling_handler && !epsilon_distribution.empty()) {
            const auto& level_epsilons = epsilon_scaling_handler->get_epsilon_values_for_level(level);
            
            if (!level_epsilons.empty()) {
                pcout << "Using " << level_epsilons.size() << " epsilon values for level " << level_number << std::endl;
                
                // Process each epsilon value for this level
                for (size_t eps_idx = 0; eps_idx < level_epsilons.size(); ++eps_idx) {
                    double current_epsilon = level_epsilons[eps_idx];
                    pcout << "  Epsilon scaling step " << eps_idx + 1 << "/" << level_epsilons.size()
                          << " (λ = " << current_epsilon << ")" << std::endl;
                    
                    // Update regularization parameter
                    solver_config.regularization_param = current_epsilon;
                    
                    Timer level_timer;
                    level_timer.start();
                    
                    try {
                        // Run optimization with current epsilon
                        sot_solver->solve(current_level_weights, solver_config);
                        
                        level_timer.stop();
                        pcout << "  Completed in " << level_timer.wall_time() << " seconds" << std::endl;
                        
                        // Save intermediate results if this is not the last epsilon for this level
                        if (save_results_to_files && eps_idx < level_epsilons.size() - 1) {
                            std::string eps_suffix = "_eps" + std::to_string(eps_idx + 1);
                            save_results(current_level_weights, level_output_dir + "/weights" + eps_suffix);
                        }
                    } catch (const SolverControl::NoConvergence& exc) {
                        if (exc.last_step >= solver_params.max_iterations) {
                            pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << eps_idx + 1
                                  << " (epsilon=" << current_epsilon << "): Max iterations reached"
                                  << Color::reset << std::endl;
                        }
                        pcout << Color::red << Color::bold << "  Warning: Optimization did not converge for epsilon " 
                              << current_epsilon << " at source level " << level_number << Color::reset << std::endl;
                        // Continue with next epsilon value
                    }
                }
            } else {
                // If no epsilon values for this level, use the smallest epsilon from the sequence
                solver_config.regularization_param = original_regularization;
                
                Timer level_timer;
                level_timer.start();
                
                try {
                    // Run optimization with default epsilon
                    sot_solver->solve(current_level_weights, solver_config);
                    
                    level_timer.stop();
                    pcout << "  Completed in " << level_timer.wall_time() << " seconds" << std::endl;
                } catch (const SolverControl::NoConvergence& exc) {
                    if (exc.last_step >= solver_params.max_iterations) {
                        pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << level_number
                              << " (epsilon=" << original_regularization << "): Max iterations reached"
                              << Color::reset << std::endl;
                    }
                    pcout << Color::red << Color::bold << "Warning: Optimization did not converge for level " << level_number << Color::reset << std::endl;
                    pcout << "  Iterations: " << exc.last_step << std::endl;
                }
            }
        } else {
            // No epsilon scaling, just run the optimization once
            Timer level_timer;
            level_timer.start();
            
            try {
                // Run optimization for this level
                sot_solver->solve(current_level_weights, solver_config);
                
                level_timer.stop();
                pcout << "  Completed in " << level_timer.wall_time() << " seconds" << std::endl;
            } catch (SolverControl::NoConvergence& exc) {
                if (exc.last_step >= solver_params.max_iterations) {
                    pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << level_number
                          << " (epsilon=" << original_regularization << "): Max iterations reached"
                          << Color::reset << std::endl;
                }
                pcout << Color::red << Color::bold << "Warning: Optimization did not converge for level " << level_number << Color::reset << std::endl;
                pcout << "  Iterations: " << exc.last_step << std::endl;
                if (level == 0) return;  // If coarsest level fails, abort
                // Otherwise continue to next level with current weights
            }
        }
            
        // Save results for this level (if requested)
        if(save_results_to_files) {
            save_results(current_level_weights, level_output_dir + "/weights");
            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
                std::ofstream conv_info(level_output_dir + "/convergence_info.txt");
                conv_info << "Regularization parameter (λ): " << solver_params.regularization_param << "\n";
                conv_info << "Number of iterations: " << sot_solver->get_last_iteration_count() << "\n";
                conv_info << "Final function value: " << sot_solver->get_last_functional_value() << "\n";
                conv_info << "Convergence achieved: " << sot_solver->get_convergence_status() << "\n";
                conv_info << "Level: " << level_number << "\n";
                pcout << "Level " << level_number << " results saved to " << level_output_dir << "/weights" << std::endl;
            }
        } else {
            pcout << "Level " << level_number << " completed" << std::endl;
        }
        
        // Store weights for next level
        level_weights = current_level_weights;
        current_distance_threshold = sot_solver->get_last_distance_threshold();
    }
    
    // If output_weights is provided, copy the final weights
    if (output_weights != nullptr) {
        *output_weights = level_weights;
    }
    
    // Restore original parameters
    solver_params.tolerance = original_tolerance;
    solver_params.max_iterations = original_max_iterations;
    solver_params.regularization_param = original_regularization;
    
    if (save_results_to_files) {
        global_timer.stop();
        pcout << "\n" << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;
        pcout << Color::magenta << Color::bold << "Total multilevel target computation time: " << global_timer.wall_time() << " seconds" << Color::reset << std::endl;
        pcout << Color::magenta << Color::bold << "----------------------------------------" << Color::reset << std::endl;
    }
}

template <int dim>
void SemidiscreteOT<dim>::run_target_multilevel_for_source_level(
    const std::string& source_mesh_file, Vector<double>& weights)
{
    run_target_multilevel(source_mesh_file, &weights, false);
}

template <int dim>
void SemidiscreteOT<dim>::run_multilevel_sot()
{
    Timer global_timer;
    global_timer.start();
    
    pcout << Color::yellow << Color::bold << "Starting multilevel SOT computation..." << Color::reset << std::endl;
    
    // Check if either source or target multilevel is enabled
    if (!multilevel_params.source_enabled && !multilevel_params.target_enabled) {
        pcout << Color::red << Color::bold << "Error: Neither source nor target multilevel is enabled. Please enable at least one in parameters.prm" << Color::reset << std::endl;
        return;
    }

    // If source multilevel is enabled, get mesh hierarchy files
    std::vector<std::string> source_mesh_files;
    if (multilevel_params.source_enabled) {
        source_mesh_files = mesh_manager->get_mesh_hierarchy_files();
        if (source_mesh_files.empty()) {
            pcout << "No source mesh hierarchy found. Please run prepare_source_multilevel first." << std::endl;
            return;
        }
    } else {
        // If only target multilevel is enabled, we need to load the base source mesh
        mesh_manager->load_source_mesh(source_mesh);
        setup_source_finite_elements();
    }
    
    // Store original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.regularization_param;
    
    // Create output directory structure
    std::string eps_dir = "output/epsilon_" + std::to_string(original_regularization);
    std::string multilevel_dir = "multilevel";
    fs::create_directories(eps_dir + "/" + multilevel_dir);

    // Initialize finite element system
    bool use_simplex = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh);
    if (use_simplex) {
        fe_system = std::make_unique<FE_SimplexP<dim>>(1);
        mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    } else {
        fe_system = std::make_unique<FE_Q<dim>>(1);
        mapping = std::make_unique<MappingQ1<dim>>();
    }

    // Setup epsilon scaling if enabled
    std::vector<std::vector<double>> epsilon_distribution;
    if (solver_params.use_epsilon_scaling && epsilon_scaling_handler) {
        pcout << "Computing epsilon distribution for multilevel optimization..." << std::endl;
        
        unsigned int num_levels = 0;
        if (multilevel_params.target_enabled) {
            // Get number of target levels
            std::vector<std::pair<std::string, std::string>> target_files = 
                Utils::get_target_hierarchy_files(multilevel_params.target_hierarchy_dir);
            num_levels = target_files.size();
        } else if (multilevel_params.source_enabled) {
            // Get number of source levels
            num_levels = source_mesh_files.size();
        }
        
        epsilon_distribution = epsilon_scaling_handler->compute_epsilon_distribution(
            num_levels, multilevel_params.target_enabled, multilevel_params.source_enabled);
        epsilon_scaling_handler->print_epsilon_distribution();
    }

    // Vector to store final weights
    Vector<double> final_weights;

    // Process source mesh levels
    if (multilevel_params.source_enabled) {
        pcout << "\n" << Color::yellow << Color::bold << "Processing source mesh hierarchy..." << Color::reset << std::endl;
        
        // Vector to store weights between source mesh levels
        Vector<double> previous_source_weights;
        
        for (size_t source_level = 0; source_level < source_mesh_files.size(); ++source_level) {
            pcout << "\n" << Color::cyan << Color::bold << "============================================" << Color::reset << std::endl;
            pcout << Color::cyan << Color::bold << "Processing source mesh level " << source_level 
                  << " (mesh: " << source_mesh_files[source_level] << ")" << Color::reset << std::endl;
            pcout << Color::cyan << Color::bold << "============================================" << Color::reset << std::endl;

            // Create directory for this source level
            std::string source_level_dir = eps_dir + "/" + multilevel_dir + "/source_level_" + std::to_string(source_level);
            fs::create_directories(source_level_dir);

            Vector<double> source_level_weights;
            if (source_level == 0) {
                // For the first source level, run target multilevel if enabled
                if (multilevel_params.target_enabled) {
                    run_target_multilevel_for_source_level(source_mesh_files[source_level], source_level_weights);
                } else {
                    // If target multilevel is not enabled, just run regular SOT
                    mesh_manager->load_mesh_at_level(source_mesh, dof_handler_source, source_mesh_files[source_level]);
                    setup_multilevel_finite_elements();
                    setup_target_points();
                    
                    source_level_weights.reinit(target_points.size());
                    sot_solver->setup_source(dof_handler_source, *mapping, *fe_system,
                                          source_density, solver_params.quadrature_order);
                    sot_solver->setup_target(target_points, target_density);
                    
                    // Apply epsilon scaling for this source level if enabled
                    if (solver_params.use_epsilon_scaling && epsilon_scaling_handler && !epsilon_distribution.empty()) {
                        const auto& level_epsilons = epsilon_scaling_handler->get_epsilon_values_for_level(source_level);
                        
                        if (!level_epsilons.empty()) {
                            pcout << "Using " << level_epsilons.size() << " epsilon values for source level " 
                                  << source_level << std::endl;
                            
                            // Process each epsilon value for this level
                            for (size_t eps_idx = 0; eps_idx < level_epsilons.size(); ++eps_idx) {
                                double current_epsilon = level_epsilons[eps_idx];
                                pcout << "  Epsilon scaling step " << eps_idx + 1 << "/" << level_epsilons.size()
                                      << " (λ = " << current_epsilon << ")" << std::endl;
                                
                                // Update regularization parameter
                                solver_params.regularization_param = current_epsilon;
                                
                                try {
                                    // Run optimization with current epsilon
                                    sot_solver->solve(source_level_weights, solver_params);
                                    
                                    // Save intermediate results if this is not the last epsilon for this level
                                    if (eps_idx < level_epsilons.size() - 1) {
                                        std::string eps_suffix = "_eps" + std::to_string(eps_idx + 1);
                                        save_results(source_level_weights, source_level_dir + "/weights" + eps_suffix);
                                    }
                                } catch (const SolverControl::NoConvergence& exc) {
                                    if (exc.last_step >= solver_params.max_iterations) {
                                        pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << eps_idx + 1
                                              << " (epsilon=" << current_epsilon << "): Max iterations reached"
                                              << Color::reset << std::endl;
                                    }
                                    pcout << Color::red << Color::bold << "  Warning: Optimization did not converge for epsilon " 
                                          << current_epsilon << " at source level " << source_level << Color::reset << std::endl;
                                    // Continue with next epsilon value
                                }
                            }
                        } else {
                            // If no epsilon values for this level, use the default epsilon
                            solver_params.regularization_param = original_regularization;
                            sot_solver->solve(source_level_weights, solver_params);
                        }
                    } else {
                        // No epsilon scaling, just run the optimization once
                        sot_solver->solve(source_level_weights, solver_params);
                    }
                }
                previous_source_weights = source_level_weights;
            } else {
                // Load the mesh for this level
                mesh_manager->load_mesh_at_level(source_mesh, dof_handler_source, source_mesh_files[source_level]);
                setup_multilevel_finite_elements();
                
                // Adjust solver parameters based on level
                double num_levels = static_cast<double>(source_mesh_files.size());
                double tolerance_exponent = static_cast<double>(source_level) - num_levels + 1.0;
                solver_params.tolerance = original_tolerance * std::pow(2.0, tolerance_exponent);
                
                pcout << "\nSource level " << source_level << " solver parameters:" << std::endl;
                pcout << "  Level: " << source_level << " of " << num_levels << std::endl;
                pcout << "  Tolerance: " << solver_params.tolerance << std::endl;
                pcout << "  Max iterations: " << solver_params.max_iterations << std::endl;
                
                // Initialize weights from previous level
                Vector<double> level_weights(target_points.size());
                level_weights = previous_source_weights;
                
                try {
                    Timer level_timer;
                    level_timer.start();
                    
                    // Set up source measure for SotSolver
                    sot_solver->setup_source(dof_handler_source,
                                           *mapping,
                                           *fe_system,
                                           source_density,
                                           solver_params.quadrature_order);
                    
                    // Set up target measure for SotSolver
                    sot_solver->setup_target(target_points, target_density);
                    
                    // Apply epsilon scaling for this source level if enabled
                    if (solver_params.use_epsilon_scaling && epsilon_scaling_handler && !epsilon_distribution.empty()) {
                        const auto& level_epsilons = epsilon_scaling_handler->get_epsilon_values_for_level(source_level);
                        
                        if (!level_epsilons.empty()) {
                            pcout << "Using " << level_epsilons.size() << " epsilon values for source level " 
                                  << source_level << std::endl;
                            
                            // Process each epsilon value for this level
                            for (size_t eps_idx = 0; eps_idx < level_epsilons.size(); ++eps_idx) {
                                double current_epsilon = level_epsilons[eps_idx];
                                pcout << "  Epsilon scaling step " << eps_idx + 1 << "/" << level_epsilons.size()
                                      << " (λ = " << current_epsilon << ")" << std::endl;
                                
                                // Update regularization parameter
                                solver_params.regularization_param = current_epsilon;
                                
                                try {
                                    // Run optimization with current epsilon
                                    sot_solver->solve(level_weights, solver_params);
                                    
                                    // Save intermediate results if this is not the last epsilon for this level
                                    if (eps_idx < level_epsilons.size() - 1) {
                                        std::string eps_suffix = "_eps" + std::to_string(eps_idx + 1);
                                        save_results(level_weights, source_level_dir + "/weights" + eps_suffix);
                                    }
                                } catch (const SolverControl::NoConvergence& exc) {
                                    if (exc.last_step >= solver_params.max_iterations) {
                                        pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << eps_idx + 1
                                              << " (epsilon=" << current_epsilon << "): Max iterations reached"
                                              << Color::reset << std::endl;
                                    }
                                    pcout << Color::red << Color::bold << "  Warning: Optimization did not converge for epsilon " 
                                          << current_epsilon << " at source level " << source_level << Color::reset << std::endl;
                                    // Continue with next epsilon value
                                }
                            }
                        } else {
                            // If no epsilon values for this level, use the default epsilon
                            solver_params.regularization_param = original_regularization;
                            sot_solver->solve(level_weights, solver_params);
                        }
                    } else {
                        // No epsilon scaling, just run the optimization once
                        sot_solver->solve(level_weights, solver_params);
                    }
                    
                    level_timer.stop();
                    
                    // Save results for this level
                    save_results(level_weights, source_level_dir + "/weights");
                    
                    // Store current solution for next level and as final result
                    previous_source_weights = level_weights;
                    if (source_level == source_mesh_files.size() - 1) {
                        final_weights = level_weights;
                    }
                    
                    pcout << "\n" << Color::blue << Color::bold << "Source level " << source_level << " summary:" << Color::reset << std::endl;
                    pcout << Color::blue << "  Status: Completed successfully" << Color::reset << std::endl;
                    pcout << Color::blue << "  Time taken: " << level_timer.wall_time() << " seconds" << Color::reset << std::endl;
                    pcout << Color::blue << "  Final number of iterations: " << sot_solver->get_last_iteration_count() << Color::reset << std::endl;
                    pcout << Color::blue << "  Final function value: " << sot_solver->get_last_functional_value() << Color::reset << std::endl;
                    pcout << Color::blue << "  Results saved in: " << source_level_dir << Color::reset << std::endl;
                    
                } catch (const std::exception& e) {
                    pcout << Color::red << Color::bold << "Error at source level " << source_level << ": " << e.what() << Color::reset << std::endl;
                    if (source_level == 0) return;  // If coarsest level fails, abort
                    // Otherwise continue to next level with previous weights
                }
            }
        }
    } else if (multilevel_params.target_enabled) {
        // If only target multilevel is enabled, run it directly
        run_target_multilevel("", &final_weights, true);
    }

    // Save final results
    save_results(final_weights, multilevel_dir + "/weights");

    // Restore original parameters
    solver_params.max_iterations = original_max_iterations;
    solver_params.tolerance = original_tolerance;
    solver_params.regularization_param = original_regularization;

    global_timer.stop();
    pcout << "\n" << Color::green << Color::bold << "============================================" << Color::reset << std::endl;
    pcout << Color::green << Color::bold << "Multilevel computation completed!" << Color::reset << std::endl;
    pcout << Color::green << Color::bold << "Total computation time: " << global_timer.wall_time() << " seconds" << Color::reset << std::endl;
    pcout << Color::green << "Final results saved in: " << eps_dir << "/" << multilevel_dir << Color::reset << std::endl;
    pcout << Color::green << Color::bold << "============================================" << Color::reset << std::endl;
}

template <int dim>
template <int d>
typename std::enable_if<d == 3>::type SemidiscreteOT<dim>::run_exact_sot()
{
    pcout << "Running exact SOT computation..." << std::endl;

    ExactSot exact_solver;

    // Set source mesh and target points
    if (!exact_solver.set_source_mesh("output/data_mesh/source.msh")) {
        pcout << "Failed to load source mesh for exact SOT" << std::endl;
        return;
    }

    // Load target points from the same location as used in setup_finite_elements
    const std::string directory = "output/data_points";
    if (!exact_solver.set_target_points(directory + "/target_points", io_coding)) {
        pcout << "Failed to load target points for exact SOT" << std::endl;
        return;
    }

    // Set solver parameters
    exact_solver.set_parameters(
        solver_params.max_iterations,
        solver_params.tolerance
    );

    // Create output directory
    std::string output_dir = "output/exact_sot";
    fs::create_directories(output_dir);

    // Run the solver
    if (!exact_solver.run()) {
        pcout << "Exact SOT computation failed" << std::endl;
        return;
    }

    // Save results
    if (!exact_solver.save_results(
            output_dir + "/weights",
            output_dir + "/points")) {
        pcout << "Failed to save exact SOT results" << std::endl;
        return;
    }

    pcout << Color::green << Color::bold << "Exact SOT computation completed successfully" << Color::reset << std::endl;
}

template <int dim>
void SemidiscreteOT<dim>::prepare_source_multilevel()
{
    pcout << "Preparing multilevel mesh hierarchy..." << std::endl;

    // Create MeshHierarchyManager instance
    MeshHierarchy::MeshHierarchyManager hierarchy_manager(
        multilevel_params.source_min_vertices,
        multilevel_params.source_max_vertices
    );

    // Ensure source mesh exists
    const std::string source_mesh_file = "output/data_mesh/source.msh";
    if (!std::filesystem::exists(source_mesh_file)) {
        pcout << "Source mesh file not found. Please run mesh_generation first." << std::endl;
        return;
    }

    try {
        // Create output directory if it doesn't exist
        const std::string hierarchy_dir = "output/data_mesh/multilevel";
        fs::create_directories(hierarchy_dir);

        // Generate hierarchy
        int num_levels = hierarchy_manager.generateHierarchyFromFile(
            source_mesh_file,
            hierarchy_dir
        );

        pcout << Color::green << Color::bold << "Successfully generated mesh hierarchy with " << num_levels << " levels" << Color::reset << std::endl;
        pcout << Color::green << "Mesh hierarchy saved in: " << hierarchy_dir << Color::reset << std::endl;
        pcout << Color::green << "Level 1 (finest) vertices: " << multilevel_params.source_max_vertices << Color::reset << std::endl;
        pcout << Color::green << "Coarsest level vertices: ~" << multilevel_params.source_min_vertices << Color::reset << std::endl;

    } catch (const std::exception& e) {
        pcout << Color::red << Color::bold << "Error: " << e.what() << Color::reset << std::endl;
    }
}

template <int dim>
void SemidiscreteOT<dim>::prepare_target_multilevel()
{
    pcout << "Preparing target point cloud hierarchy..." << std::endl;

    // Make sure target points are properly set up
    if (target_points.empty()) {
        pcout << "Target points not loaded. Attempting to load from file or from mesh..." << std::endl;
        
        // Try to load target points from file
        std::string target_points_file = "output/data_points/target_points.txt";
        if (fs::exists(target_points_file)) {
            pcout << "Loading target points from file: " << target_points_file << std::endl;
            
            std::ifstream in(target_points_file);
            if (!in.good()) {
                pcout << "Failed to open target points file." << std::endl;
                return;
            }
            
            target_points.clear();
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream iss(line);
                Point<dim> p;
                for (unsigned int d = 0; d < dim; ++d) {
                    if (!(iss >> p[d])) {
                        pcout << "Error parsing point coordinates." << std::endl;
                        return;
                    }
                }
                target_points.push_back(p);
            }
            
            pcout << "Loaded " << target_points.size() << " target points from file." << std::endl;
        } else if (!target_mesh.n_active_cells()) {
            mesh_manager->load_target_mesh(target_mesh);
        }
        
        // If target points still empty but we have a mesh, extract points from mesh
        if (target_points.empty() && target_mesh.n_active_cells() > 0) {
            pcout << "Extracting target points from mesh..." << std::endl;
            
            dof_handler_target.distribute_dofs(*fe_system);
            std::map<types::global_dof_index, Point<dim>> support_points_target;
            DoFTools::map_dofs_to_support_points(*mapping, dof_handler_target, support_points_target);
            
            for (const auto &point_pair : support_points_target) {
                target_points.push_back(point_pair.second);
            }
            
            // Write points to file for future use
            std::string output_dir = "output/data_points";
            fs::create_directories(output_dir);
            Utils::write_vector(target_points, output_dir + "/target_points", io_coding);
            
            pcout << "Extracted " << target_points.size() << " target points from mesh." << std::endl;
        }
    }
    
    // Check if we have target points now
    if (target_points.empty()) {
        pcout << "Failed to obtain target points. Cannot generate hierarchy." << std::endl;
        return;
    }
    
    // Create target density if needed
    if (target_density.size() != target_points.size()) {
        target_density.reinit(target_points.size());
        const double uniform_weight = 1.0 / target_points.size();
        for (size_t i = 0; i < target_points.size(); ++i) {
            target_density[i] = uniform_weight;
        }
    }

    // Check if we should use Python script for clustering
    if (multilevel_params.use_python_clustering) {
        // Only rank 0 should execute the Python script
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            pcout << "Using Python script for clustering: " << multilevel_params.python_script_name << std::endl;
            
            // Create output directory if it doesn't exist
            fs::create_directories(multilevel_params.target_hierarchy_dir);
            
            // Construct the command to run the Python script
            std::string python_cmd = "python3 ";
            
            // Check if the script is in the python_scripts directory or a full path
            if (multilevel_params.python_script_name.find('/') == std::string::npos) {
                // First try in python_scripts subdirectory
                if (fs::exists("python_scripts/" + multilevel_params.python_script_name)) {
                    python_cmd += "python_scripts/";
                }
                // If not found, assume it's in the run directory directly
            }
            
            python_cmd += multilevel_params.python_script_name;
            
            // Execute the Python script
            pcout << Color::green << Color::bold << "Executing: " << python_cmd << Color::reset << std::endl;
            int result = std::system(python_cmd.c_str());
            
            if (result != 0) {
                pcout << Color::red << Color::bold << "Error: Python script execution failed with code " 
                      << result << Color::reset << std::endl;
                return;
            }
            
            pcout << Color::green << "Python script executed successfully" << Color::reset << std::endl;
        }
        
        // Make sure all processes wait for rank 0 to finish the Python script
        MPI_Barrier(mpi_communicator);
        
        // Load the hierarchy data for use in computations
        load_hierarchy_data(multilevel_params.target_hierarchy_dir);
        pcout << "Loaded hierarchy data for direct parent-child weight assignment." << std::endl;
        return;
    }

    // If not using Python script, use the built-in C++ implementation
    // Create PointCloudHierarchyManager instance
    PointCloudHierarchy::PointCloudHierarchyManager hierarchy_manager(
        multilevel_params.target_min_points,
        multilevel_params.target_max_points
    );

    try {
        // Create output directory if it doesn't exist
        fs::create_directories(multilevel_params.target_hierarchy_dir);

        // Generate hierarchy
        std::vector<double> target_weights(target_points.size());
        for (size_t i = 0; i < target_points.size(); ++i) {
            target_weights[i] = target_density[i];
        }

        pcout << "Generating hierarchy with " << target_points.size() << " points..." << std::endl;
        
        int num_levels = hierarchy_manager.generateHierarchy<dim>(
            target_points,
            target_weights,
            multilevel_params.target_hierarchy_dir
        );
        
        pcout << Color::green << Color::bold << "Successfully generated " << num_levels << " levels of point cloud hierarchy." << Color::reset << std::endl;
        
        // Load the hierarchy data for use in computations
        load_hierarchy_data(multilevel_params.target_hierarchy_dir);
        pcout << "Loaded hierarchy data for direct parent-child weight assignment." << std::endl;
        
    } catch (const std::exception& e) {
        pcout << Color::red << Color::bold << "Error: " << e.what() << Color::reset << std::endl;
    }
}


template <int dim>
void SemidiscreteOT<dim>::save_results(const Vector<double> &weights,
                                     const std::string &filename)
{
    // Only rank 0 should create directories and write files
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        // Create epsilon-specific directory
        std::string eps_dir = "output/epsilon_" + std::to_string(solver_params.regularization_param);
        fs::create_directories(eps_dir);

        // Save weights
        std::vector<double> weights_vec(weights.begin(), weights.end());
        Utils::write_vector(weights_vec, eps_dir + "/" + filename, io_coding);
    }
    // Make sure all processes wait for rank 0 to finish writing
    MPI_Barrier(mpi_communicator);
}

template <int dim>
void SemidiscreteOT<dim>::compute_transport_map()
{
    load_meshes();
    setup_finite_elements();

    // Let user select which folder(s) to use
    std::vector<std::string> selected_folders;
    try {
        selected_folders = Utils::select_folder();
    } catch (const std::exception& e) {
        pcout << "Error: " << e.what() << std::endl;
        return;
    }

    // Create transport map approximator
    OptimalTransportPlanSpace::OptimalTransportPlan<dim> transport_plan;

    // Get source points and density (serial version)
    std::vector<Point<dim>> source_points;
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
        std::map<types::global_dof_index, Point<dim>> support_points_source;
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
    for (const auto& selected_folder : selected_folders) {
        pcout << "\nProcessing folder: " << selected_folder << std::endl;

        // Read weights from selected folder's results
        std::vector<double> weights_vec;
        bool success = Utils::read_vector(weights_vec, "output/" + selected_folder + "/weights", io_coding);
        if (!success) {
            pcout << "Failed to read weights from output/" + selected_folder + "/weights" << std::endl;
            continue; // Skip to next folder
        }

        if (weights_vec.size() != target_points.size()) {
            pcout << Color::red << Color::bold << "Error: Mismatch between weights size (" << weights_vec.size()
                  << ") and target points size (" << target_points.size() << ")" << Color::reset << std::endl;
            continue; // Skip to next folder
        }

        // Extract regularization parameter from folder name
        std::string eps_str = selected_folder.substr(selected_folder.find("epsilon_") + 8);
        double regularization_param = std::stod(eps_str);

        // Create output directory
        const std::string output_dir = "output/" + selected_folder + "/transport_map";
        fs::create_directories(output_dir);

        // Convert weights to dealii::Vector format
        Vector<double> weights_dealii(weights_vec.size());
        std::copy(weights_vec.begin(), weights_vec.end(), weights_dealii.begin());

        // Set up the transport plan
        transport_plan.set_source_measure(source_points, source_density_vec);
        transport_plan.set_target_measure(target_points, target_density_vec);
        transport_plan.set_potential(weights_dealii, regularization_param);

        // Try different strategies and save results
        for (const auto& strategy : transport_plan.get_available_strategies()) {
            pcout << "Computing transport map using " << strategy << " strategy..." << std::endl;
            
            transport_plan.set_strategy(strategy);
            transport_plan.compute_map();
            transport_plan.save_map(output_dir + "/" + strategy);
            
            pcout << "Results saved in " << output_dir + "/" + strategy << std::endl;
        }
    }

    if (selected_folders.size() > 1) {
        pcout << "\nCompleted transport map computation for all selected folders." << std::endl;
    }
}

template <int dim>
void SemidiscreteOT<dim>::compute_power_diagram()
{
    load_meshes();
    setup_finite_elements();

    // Let user select which folder(s) to use
    std::vector<std::string> selected_folders;
    try {
        selected_folders = Utils::select_folder();
    } catch (const std::exception& e) {
        pcout << "Error: " << e.what() << std::endl;
        return;
    }

    // Process each selected folder
    for (const auto& selected_folder : selected_folders) {
        pcout << "\nProcessing folder: " << selected_folder << std::endl;

        // Read weights from selected folder's results
        std::vector<double> weights_vec;
        bool success = Utils::read_vector(weights_vec, "output/" + selected_folder + "/weights", io_coding);
        if (!success) {
            pcout << "Failed to read weights from output/" << selected_folder << "/weights" << std::endl;
            continue; // Skip to next folder
        }

        if (weights_vec.size() != target_points.size()) {
            pcout << Color::red << Color::bold << "Error: Mismatch between weights size (" << weights_vec.size()
                  << ") and target points size (" << target_points.size() << ")" << Color::reset << std::endl;
            continue; // Skip to next folder
        }

        // Convert to dealii::Vector
        Vector<double> weights(weights_vec.size());
        std::copy(weights_vec.begin(), weights_vec.end(), weights.begin());

        // Create output directory
        const std::string output_dir = "output/" + selected_folder + "/power_diagram_"+power_diagram_params.implementation;
        fs::create_directories(output_dir);

        // Create power diagram using factory function based on parameter choice
        std::unique_ptr<PowerDiagramSpace::PowerDiagramBase<dim>> power_diagram;

        try {
            if (power_diagram_params.implementation == "geogram") {
                if constexpr (dim == 3) {
                    power_diagram = PowerDiagramSpace::create_power_diagram<dim>(
                        "geogram",
                        nullptr,
                        "output/data_mesh/source.msh"
                    );
                    pcout << "Using Geogram implementation for power diagram" << std::endl;
                } else {
                    pcout << "Geogram implementation is only available for 3D problems" << std::endl;
                    pcout << "Falling back to Deal.II implementation" << std::endl;
                    power_diagram = PowerDiagramSpace::create_power_diagram<dim>(
                        "dealii",
                        &source_mesh
                    );
                }
            } else {
                power_diagram = PowerDiagramSpace::create_power_diagram<dim>(
                    "dealii",
                    &source_mesh
                );
                pcout << "Using Deal.II implementation for power diagram" << std::endl;
            }
        } catch (const std::exception& e) {
            pcout << Color::red << Color::bold << "Failed to initialize " << power_diagram_params.implementation
                  << " implementation: " << e.what() << Color::reset << std::endl;
            if (power_diagram_params.implementation == "geogram") {
                pcout << "Falling back to Deal.II implementation" << std::endl;
                power_diagram = PowerDiagramSpace::create_power_diagram<dim>(
                    "dealii",
                    &source_mesh
                );
            } else {
                throw;
            }
        }

        // Set generators and compute power diagram
        power_diagram->set_generators(target_points, weights);
        power_diagram->compute_power_diagram();
        power_diagram->compute_cell_centroids();

        // Save results
        power_diagram->save_centroids_to_file(output_dir + "/centroids");
        power_diagram->output_vtu(output_dir + "/power_diagram");

        pcout << "Power diagram computation completed for " << selected_folder << std::endl;
        pcout << "Results saved in " << output_dir << std::endl;
    }

    if (selected_folders.size() > 1) {
        pcout << "\nCompleted power diagram computation for all selected folders." << std::endl;
    }
}

template <int dim>
void SemidiscreteOT<dim>::save_discrete_measures()
{
    load_meshes();
    setup_finite_elements();

    // Create output directory
    const std::string directory = "output/discrete_measures";
    fs::create_directories(directory);

    // Get quadrature points and weights
    std::unique_ptr<Quadrature<dim>> quadrature;
    if (source_params.use_tetrahedral_mesh) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(solver_params.quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(solver_params.quadrature_order);
    }

    FEValues<dim> fe_values(*mapping, *fe_system, *quadrature,
                           update_values | update_quadrature_points | update_JxW_values);

    // Count total number of quadrature points
    const unsigned int n_q_points = quadrature->size();
    const unsigned int total_q_points = source_mesh.n_active_cells() * n_q_points;

    // Prepare vectors for quadrature data
    std::vector<Point<dim>> quad_points;
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



template <int dim>
void SemidiscreteOT<dim>::run()
{
    param_manager.print_parameters();

    if (solver_params.use_epsilon_scaling) {
        epsilon_scaling_handler = std::make_unique<EpsilonScalingHandler>(
            mpi_communicator,
            solver_params.regularization_param,
            solver_params.epsilon_scaling_factor,
            solver_params.epsilon_scaling_steps
        );
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
        load_meshes();
        prepare_target_multilevel();
    }
    else if (selected_task == "prepare_multilevel")
    {
        if (multilevel_params.source_enabled)
            prepare_source_multilevel();
        if (multilevel_params.target_enabled)
        {
            load_meshes();
            prepare_target_multilevel();
        }
    }
    else if (selected_task == "source_multilevel_sot")
    {
        multilevel_params.source_enabled = true;
        multilevel_params.target_enabled = false;
        run_multilevel_sot();
    }
    else if (selected_task == "target_multilevel_sot")
    {
        run_target_multilevel();
    }
    else if (selected_task == "multilevel_sot")
    {
        run_multilevel_sot();
    }
    else if (selected_task == "exact_sot")
    {
        if constexpr (dim == 3) {
            load_meshes();
            setup_finite_elements();
            run_exact_sot();
        } else {
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
    else
    {
        pcout << "No valid task selected" << std::endl;
    }
}

template class SemidiscreteOT<2>;
template class SemidiscreteOT<3>;