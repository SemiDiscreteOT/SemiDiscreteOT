#include "ParameterManager.h"
#include <iomanip>  

ParameterManager::ParameterManager(const MPI_Comm &comm)
    : ParameterAcceptor("ParameterManager")
    , source_params(source_params_storage)
    , target_params(target_params_storage)
    , solver_params(solver_params_storage)
    , multilevel_params(multilevel_params_storage)
    , power_diagram_params(power_diagram_params_storage)
    , transport_map_params(transport_map_params_storage)
    , selected_task(selected_task_storage)
    , io_coding(io_coding_storage)
    , mpi_communicator(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
{
    add_parameter("selected_task", selected_task);
    add_parameter("io_coding", io_coding,
                 "File format for I/O operations (txt/bin)");

    enter_subsection("mesh_generation");
    {
        enter_subsection("source");
        {
            add_parameter("number of refinements", source_params.n_refinements);
            add_parameter("grid generator function", source_params.grid_generator_function);
            add_parameter("grid generator arguments", source_params.grid_generator_arguments);
            add_parameter("use tetrahedral mesh", source_params.use_tetrahedral_mesh,
                         "Whether to convert the mesh to tetrahedral cells (only for 3D)");
        }
        leave_subsection();

        enter_subsection("target");
        {
            add_parameter("number of refinements", target_params.n_refinements);
            add_parameter("grid generator function", target_params.grid_generator_function);
            add_parameter("grid generator arguments", target_params.grid_generator_arguments);
            add_parameter("use tetrahedral mesh", target_params.use_tetrahedral_mesh,
                         "Whether to convert the mesh to tetrahedral cells (only for 3D)");
        }
        leave_subsection();
    }
    leave_subsection();

    enter_subsection("rsot_solver");
    {
        add_parameter("max_iterations", solver_params.max_iterations,
                     "Maximum number of iterations for the optimization solver");
        add_parameter("tolerance", solver_params.tolerance,
                     "Convergence tolerance for the optimization solver");
        add_parameter("regularization_parameter", solver_params.regularization_param,
                     "Entropy regularization parameter (lambda)");
        add_parameter("epsilon", solver_params.epsilon,
                     "Truncation criterion for the kernel evaluation");
        add_parameter("tau", solver_params.tau,
                     "Truncation error tolerance for integral radius bound");
        add_parameter("verbose_output", solver_params.verbose_output,
                     "Enable detailed solver output");
        add_parameter("solver_type", solver_params.solver_type,
                     "Type of optimization solver (BFGS)");
        add_parameter("quadrature_order", solver_params.quadrature_order,
                     "Order of quadrature formula for numerical integration");
        add_parameter("number_of_threads", solver_params.n_threads,
                     "Number of threads to use for parallel SOT");
        add_parameter("use_epsilon_scaling", solver_params.use_epsilon_scaling,
                     "Enable epsilon scaling strategy");
        add_parameter("epsilon_scaling_factor", solver_params.epsilon_scaling_factor,
                     "Factor by which to reduce epsilon in each scaling step");
        add_parameter("epsilon_scaling_steps", solver_params.epsilon_scaling_steps,
                     "Number of epsilon scaling steps");
        add_parameter("use_caching", solver_params.use_caching,
                     "Enable distance threshold caching");
        add_parameter("max_cache_size_mb", solver_params.max_cache_size_mb,
                     "Maximum cache size in MB/per processor");
    }
    leave_subsection();

    enter_subsection("multilevel_parameters");
    {
        // Source mesh hierarchy parameters
        add_parameter("source_enabled", multilevel_params.source_enabled,
                     "Whether to use source multilevel approach");
        add_parameter("source_min_vertices", multilevel_params.source_min_vertices,
                     "Minimum number of vertices for the coarsest source level");
        add_parameter("source_max_vertices", multilevel_params.source_max_vertices,
                     "Maximum number of vertices for finest source level");
        add_parameter("source_hierarchy_dir", multilevel_params.source_hierarchy_dir,
                     "Directory to store the source mesh hierarchy");
        
        // Target point cloud hierarchy parameters
        add_parameter("target_enabled", multilevel_params.target_enabled,
                     "Whether to use target multilevel approach");
        add_parameter("target_min_points", multilevel_params.target_min_points,
                     "Minimum number of points for the coarsest target level");
        add_parameter("target_max_points", multilevel_params.target_max_points,
                     "Maximum number of points for the finest target level");
        add_parameter("target_hierarchy_dir", multilevel_params.target_hierarchy_dir,
                     "Directory to store target point cloud hierarchy");
        add_parameter("use_softmax_weight_transfer", multilevel_params.use_softmax_weight_transfer,
                     "Whether to use softmax-based weight transfer between target levels");
        
        // Common parameters
        add_parameter("output_prefix", multilevel_params.output_prefix,
                     "Directory prefix for multilevel SOT results");
    }
    leave_subsection();


    enter_subsection("power_diagram_parameters");
    {
        add_parameter("implementation", power_diagram_params.implementation,
                     "Implementation to use for power diagram computation (dealii/geogram)");
    }
    leave_subsection();

    enter_subsection("transport_map_parameters");
    {
        add_parameter("n_neighbors", transport_map_params.n_neighbors,
                     "Number of neighbors for local methods");
        add_parameter("kernel_width", transport_map_params.kernel_width,
                     "Kernel width for smooth approximations");
        add_parameter("interpolation_type", transport_map_params.interpolation_type,
                     "Type of interpolation");
    }
    leave_subsection();
}


void ParameterManager::print_logo() const
{
    std::string logo = R"(
 ▗▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▄▄▄ ▗▄▄▄▖ ▗▄▄▖ ▗▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖ ▗▄▖▗▄▄▄▖
▐▌   ▐▌   ▐▛▚▞▜▌  █  ▐▌  █  █  ▐▌   ▐▌   ▐▌ ▐▌▐▌     █  ▐▌   ▐▌ ▐▌ █  
 ▝▀▚▖▐▛▀▀▘▐▌  ▐▌  █  ▐▌  █  █   ▝▀▚▖▐▌   ▐▛▀▚▖▐▛▀▀▘  █  ▐▛▀▀▘▐▌ ▐▌ █  
▗▄▄▞▘▐▙▄▄▖▐▌  ▐▌▗▄█▄▖▐▙▄▄▀▗▄█▄▖▗▄▄▞▘▝▚▄▄▖▐▌ ▐▌▐▙▄▄▖  █  ▐▙▄▄▖▝▚▄▞▘ █  
    )";

    // Print the logo with cyan color
    pcout << CYAN << logo << RESET << std::endl;

    // Add a simple morphing visualization
    pcout << std::endl;
    pcout << GREEN << "                      ∎∎∎∎∎∎∎      " << BLUE << "          ◆         " << RESET << std::endl;
    pcout << GREEN << "                   ∎∎∎∎∎∎∎∎∎∎∎     " << MAGENTA << "     ↗     " << BLUE << "    ◆   ◆     " << RESET << std::endl;
    pcout << GREEN << "                 ∎∎∎∎∎∎∎∎∎∎∎∎∎    " << MAGENTA << "    ↗      " << BLUE << "   ◆     ◆    " << RESET << std::endl;
    pcout << GREEN << "               ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎    " << MAGENTA << "   ↗       " << BLUE << "  ◆  ◆◆  ◆    " << RESET << std::endl;
    pcout << GREEN << "              ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎    " << MAGENTA << "   →       " << BLUE << "  ◆ ◆  ◆ ◆    " << RESET << std::endl;
    pcout << GREEN << "               ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎    " << MAGENTA << "   ↘       " << BLUE << "  ◆  ◆◆  ◆    " << RESET << std::endl;
    pcout << GREEN << "                 ∎∎∎∎∎∎∎∎∎∎∎∎∎    " << MAGENTA << "    ↘      " << BLUE << "   ◆     ◆    " << RESET << std::endl;
    pcout << GREEN << "                   ∎∎∎∎∎∎∎∎∎∎∎     " << MAGENTA << "     ↘     " << BLUE << "    ◆   ◆     " << RESET << std::endl;
    pcout << GREEN << "                      ∎∎∎∎∎∎∎      " << BLUE << "          ◆         " << RESET << std::endl;
    pcout << std::endl;

    pcout << MAGENTA << "                           Semidiscrete Optimal Transport                           " << RESET << std::endl;
    pcout << std::endl;
}


void ParameterManager::print_parameters() const
{
    print_logo();
    print_task_information();
    
    // Print a visually appealing header for the selected task
    pcout << YELLOW << "\nCONFIGURATION FOR: " << BOLD << selected_task << RESET << std::endl;
    pcout << std::string(80, '-') << std::endl;
    
    pcout << "I/O Format: " << BOLD << io_coding << RESET << "\n" << std::endl;

    // Print relevant parameters based on selected task
    if (selected_task == "generate_mesh" || selected_task == "sot" || 
        selected_task == "load_meshes" || selected_task == "multilevel_sot" ||
        selected_task == "exact_sot")
    {
        print_section_header("MESH PARAMETERS");
        print_mesh_parameters();
    }

    if (selected_task == "sot" || selected_task == "multilevel_sot" || 
        selected_task == "exact_sot")
    {
        print_section_header("SOLVER PARAMETERS");
        print_solver_parameters();
    }

    if (selected_task == "multilevel_sot" || selected_task == "prepare_source_multilevel" || 
        selected_task == "prepare_target_multilevel")
    {
        print_section_header("MULTILEVEL PARAMETERS");
        print_multilevel_parameters();
    }

    if (selected_task == "power_diagram")
    {
        print_section_header("POWER DIAGRAM PARAMETERS");
        print_power_diagram_parameters();
    }

    if (selected_task == "map")
    {
        print_section_header("TRANSPORT MAP PARAMETERS");
        print_transport_map_parameters();
    }

    pcout << std::endl;
}

void ParameterManager::print_section_header(const std::string& section_name) const
{
    pcout << MAGENTA << section_name << RESET << std::endl;
    pcout << std::string(section_name.length(), '-') << std::endl;
}

void ParameterManager::print_task_information() const
{
    pcout << YELLOW << "\n=== AVAILABLE TASKS ===" << RESET << std::endl;
    pcout << std::string(80, '-') << std::endl;
    
    // Define task descriptions
    struct TaskInfo {
        std::string name;
        std::string description;
    };
    
    std::vector<TaskInfo> tasks = {
        {"generate_mesh", "Generate source and target meshes based on specified parameters"},
        {"load_meshes", "Load pre-existing source and target meshes"},
        {"sot", "Run standard semidiscrete optimal transport"},
        {"exact_sot", "Run exact semidiscrete optimal transport (no regularization)"},
        {"power_diagram", "Compute power diagram for given weights"},
        {"map", "Compute transport map between source and target"},
        {"prepare_source_multilevel", "Prepare source mesh hierarchy for multilevel approach"},
        {"prepare_target_multilevel", "Prepare target point cloud hierarchy for multilevel approach"},
        {"multilevel_sot", "Run multilevel semidiscrete optimal transport"}
    };
    
    // Print each task with proper formatting
    for (const auto& task : tasks) {
        // Highlight the currently selected task
        if (task.name == selected_task) {
            pcout << CYAN << BOLD;
        }
        
        // Print task name and description
        pcout << std::setw(30) << std::left << task.name 
              << task.description << RESET << std::endl;
    }
    
    pcout << std::string(80, '-') << std::endl;
    pcout << std::endl;
}

void ParameterManager::print_mesh_parameters() const
{
    // Source mesh parameters
    pcout << CYAN << "  Source Mesh:" << RESET << std::endl;
    pcout << "    Grid Generator Function: " << BOLD << source_params.grid_generator_function << RESET << std::endl;
    pcout << "    Grid Generator Arguments: " << BOLD << source_params.grid_generator_arguments << RESET << std::endl;
    pcout << "    Number of Refinements: " << BOLD << source_params.n_refinements << RESET << std::endl;
    pcout << "    Use Tetrahedral Mesh: " << BOLD << (source_params.use_tetrahedral_mesh ? "yes" : "no") << RESET << std::endl;
    pcout << std::endl;

    // Target mesh parameters
    pcout << CYAN << "  Target Mesh:" << RESET << std::endl;
    pcout << "    Grid Generator Function: " << BOLD << target_params.grid_generator_function << RESET << std::endl;
    pcout << "    Grid Generator Arguments: " << BOLD << target_params.grid_generator_arguments << RESET << std::endl;
    pcout << "    Number of Refinements: " << BOLD << target_params.n_refinements << RESET << std::endl;
    pcout << "    Use Tetrahedral Mesh: " << BOLD << (target_params.use_tetrahedral_mesh ? "yes" : "no") << RESET << std::endl;
    pcout << std::endl;
}

void ParameterManager::print_solver_parameters() const
{
    pcout << CYAN << "  Solver Configuration:" << RESET << std::endl;
    pcout << "    Solver Type: " << BOLD << solver_params.solver_type << RESET << std::endl;
    pcout << "    Maximum Iterations: " << BOLD << solver_params.max_iterations << RESET << std::endl;
    pcout << "    Tolerance: " << BOLD << solver_params.tolerance << RESET << std::endl;
    pcout << std::endl;
    
    pcout << CYAN << "  Regularization Settings:" << RESET << std::endl;
    pcout << "    Regularization Parameter: " << BOLD << solver_params.regularization_param << RESET << std::endl;
    pcout << "    Epsilon: " << BOLD << solver_params.epsilon << RESET << std::endl;
    pcout << "    Tau: " << BOLD << solver_params.tau << RESET << std::endl;
    
    if (solver_params.use_epsilon_scaling) {
        pcout << "    Epsilon Scaling: " << BOLD << "enabled" << RESET << std::endl;
        pcout << "    Epsilon Scaling Factor: " << BOLD << solver_params.epsilon_scaling_factor << RESET << std::endl;
        pcout << "    Epsilon Scaling Steps: " << BOLD << solver_params.epsilon_scaling_steps << RESET << std::endl;
    } else {
        pcout << "    Epsilon Scaling: " << BOLD << "disabled" << RESET << std::endl;
    }
    pcout << std::endl;
    
    pcout << CYAN << "  Computation Settings:" << RESET << std::endl;
    pcout << "    Quadrature Order: " << BOLD << solver_params.quadrature_order << RESET << std::endl;
    pcout << "    Number of Points: " << BOLD << solver_params.nb_points << RESET << std::endl;
    pcout << "    Number of Threads: " << BOLD << (solver_params.n_threads == 0 ? "auto" : std::to_string(solver_params.n_threads)) << RESET << std::endl;
    pcout << "    Caching: " << BOLD << (solver_params.use_caching ? "enabled" : "disabled") << RESET << std::endl;
    if (solver_params.use_caching) {
        pcout << "    Max Cache Size: " << BOLD << (solver_params.max_cache_size_mb == 0 ? "unlimited" : 
                                                  std::to_string(solver_params.max_cache_size_mb) + " MB") << RESET << std::endl;
    }
    pcout << std::endl;
}

void ParameterManager::print_multilevel_parameters() const
{
    // Source multilevel parameters
    pcout << CYAN << "  Source Multilevel:" << RESET << std::endl;
    pcout << "    Enabled: " << BOLD << (multilevel_params.source_enabled ? "yes" : "no") << RESET << std::endl;
    if (multilevel_params.source_enabled) {
        pcout << "    Minimum Vertices: " << BOLD << multilevel_params.source_min_vertices << RESET << std::endl;
        pcout << "    Maximum Vertices: " << BOLD << multilevel_params.source_max_vertices << RESET << std::endl;
        pcout << "    Hierarchy Directory: " << BOLD << multilevel_params.source_hierarchy_dir << RESET << std::endl;
    }
    pcout << std::endl;
    
    // Target multilevel parameters
    pcout << CYAN << "  Target Multilevel:" << RESET << std::endl;
    pcout << "    Enabled: " << BOLD << (multilevel_params.target_enabled ? "yes" : "no") << RESET << std::endl;
    if (multilevel_params.target_enabled) {
        pcout << "    Minimum Points: " << BOLD << multilevel_params.target_min_points << RESET << std::endl;
        pcout << "    Maximum Points: " << BOLD << multilevel_params.target_max_points << RESET << std::endl;
        pcout << "    Hierarchy Directory: " << BOLD << multilevel_params.target_hierarchy_dir << RESET << std::endl;
        pcout << "    Softmax Weight Transfer: " << BOLD << (multilevel_params.use_softmax_weight_transfer ? "enabled" : "disabled") << RESET << std::endl;
    }
    pcout << std::endl;
    
    // Common parameters
    pcout << CYAN << "  Output Settings:" << RESET << std::endl;
    pcout << "    Output Prefix: " << BOLD << multilevel_params.output_prefix << RESET << std::endl;
    pcout << std::endl;
}


void ParameterManager::print_power_diagram_parameters() const
{
    pcout << CYAN << "  Power Diagram Computation:" << RESET << std::endl;
    pcout << "    Implementation: " << BOLD << power_diagram_params.implementation << RESET << std::endl;
    pcout << std::endl;
}

void ParameterManager::print_transport_map_parameters() const
{
    pcout << CYAN << "  Transport Map Settings:" << RESET << std::endl;
    pcout << "    Number of Neighbors: " << BOLD << transport_map_params.n_neighbors << RESET << std::endl;
    pcout << "    Kernel Width: " << BOLD << transport_map_params.kernel_width << RESET << std::endl;
    pcout << "    Interpolation Type: " << BOLD << transport_map_params.interpolation_type << RESET << std::endl;
    pcout << std::endl;
} 