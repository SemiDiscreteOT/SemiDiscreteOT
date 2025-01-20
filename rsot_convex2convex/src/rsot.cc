#include "rsot.h"
#include "PowerDiagram.h"
#include "utils.h"
#include "ExactSot.h"
#include <deal.II/base/timer.h>
#include <filesystem>
namespace fs = std::filesystem;

template <int dim>
Convex2Convex<dim>::Convex2Convex()
    : ParameterAcceptor("Convex2Convex"), 
      source_mesh(), 
      target_mesh(), 
      dof_handler_source(source_mesh), 
      dof_handler_target(target_mesh)
{
    // Initialize with default hexahedral elements
    fe_system = std::make_unique<FE_Q<dim>>(1);
    mapping = std::make_unique<MappingQ1<dim>>();

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
        add_parameter("max_iterations",
                     solver_params.max_iterations,
                     "Maximum number of iterations for the optimization solver");

        add_parameter("tolerance",
                     solver_params.tolerance,
                     "Convergence tolerance for the optimization solver");

        add_parameter("regularization_parameter",
                     solver_params.regularization_param,
                     "Entropy regularization parameter (lambda)");

        add_parameter("epsilon",
                     solver_params.epsilon,
                     "Truncation criterion for the kernel evaluation (smaller values include more points)");

        add_parameter("verbose_output",
                     solver_params.verbose_output,
                     "Enable detailed solver output");

        add_parameter("debug",
                     solver_params.debug,
                     "Enable debug output for target point statistics");

        add_parameter("solver_type",
                     solver_params.solver_type,
                     "Type of optimization solver (BFGS)");

        add_parameter("quadrature_order",
                     solver_params.quadrature_order,
                     "Order of quadrature formula for numerical integration");

        add_parameter("number_of_threads",
                     solver_params.number_of_threads,
                     "Number of threads to use (0 means use all available cores)");
    }
    leave_subsection();

    enter_subsection("power_diagram_parameters");
    {
        add_parameter("implementation",
                     power_diagram_params.implementation,
                     "Implementation to use for power diagram computation (dealii/geogram)");
    }
    leave_subsection();
}

template <int dim>
void Convex2Convex<dim>::print_parameters()
{
    std::cout << "Selected Task: " << selected_task << std::endl;
    std::cout << "I/O Coding: " << io_coding << std::endl;

    std::cout << "Source Mesh Parameters:" << std::endl;
    std::cout << "  Grid Generator Function: " << source_params.grid_generator_function << std::endl;
    std::cout << "  Grid Generator Arguments: " << source_params.grid_generator_arguments << std::endl;
    std::cout << "  Number of Refinements: " << source_params.n_refinements << std::endl;

    std::cout << "Target Mesh Parameters:" << std::endl;
    std::cout << "  Grid Generator Function: " << target_params.grid_generator_function << std::endl;
    std::cout << "  Grid Generator Arguments: " << target_params.grid_generator_arguments << std::endl;
    std::cout << "  Number of Refinements: " << target_params.n_refinements << std::endl;

    std::cout << "RSOT Solver Parameters:" << std::endl;
    std::cout << "  Max Iterations: " << solver_params.max_iterations << std::endl;
    std::cout << "  Tolerance: " << solver_params.tolerance << std::endl;
    std::cout << "  Regularization Parameter (λ): " << solver_params.regularization_param << std::endl;
    std::cout << "  Verbose Output: " << (solver_params.verbose_output ? "Yes" : "No") << std::endl;
    std::cout << "  Solver Type: " << solver_params.solver_type << std::endl;
    std::cout << "  Quadrature Order: " << solver_params.quadrature_order << std::endl;
}

template <int dim>
void Convex2Convex<dim>::generate_mesh(Triangulation<dim> &tria,
                                       const std::string &grid_generator_function,
                                       const std::string &grid_generator_arguments,
                                       const unsigned int n_refinements,
                                       const bool use_tetrahedral_mesh)
{
    GridGenerator::generate_from_name_and_arguments(
        tria,
        grid_generator_function,
        grid_generator_arguments);

    if (use_tetrahedral_mesh && dim == 3) {
        GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria);
    }

    tria.refine_global(n_refinements);
}

template <int dim>
void Convex2Convex<dim>::save_meshes()
{
    const std::string directory = "output/data_mesh";

    // Use Utils::write_mesh for both meshes
    Utils::write_mesh(source_mesh,
                     directory + "/source",
                     std::vector<std::string>{"vtk", "msh"});

    Utils::write_mesh(target_mesh,
                     directory + "/target",
                     std::vector<std::string>{"vtk", "msh"});

    std::cout << "Meshes saved in VTK and MSH formats" << std::endl;
}

template <int dim>
void Convex2Convex<dim>::mesh_generation()
{
    generate_mesh(source_mesh,
                 source_params.grid_generator_function,
                 source_params.grid_generator_arguments,
                 source_params.n_refinements,
                 source_params.use_tetrahedral_mesh);

    generate_mesh(target_mesh,
                 target_params.grid_generator_function,
                 target_params.grid_generator_arguments,
                 target_params.n_refinements,
                 target_params.use_tetrahedral_mesh);

    save_meshes();
}

template <int dim>
void Convex2Convex<dim>::load_meshes()
{
    const std::string directory = "output/data_mesh";

    // Try loading source mesh
    GridIn<dim> grid_in_source;
    grid_in_source.attach_triangulation(source_mesh);
    bool source_loaded = false;

    // First try VTK
    std::ifstream in_vtk_source(directory + "/source.vtk");
    if (in_vtk_source.good()) {
        try {
            grid_in_source.read_vtk(in_vtk_source);
            source_loaded = true;
            std::cout << "Source mesh loaded from VTK format" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed to load source mesh from VTK: " << e.what() << std::endl;
        }
    }

    // If VTK failed, try MSH
    if (!source_loaded) {
        std::ifstream in_msh_source(directory + "/source.msh");
        if (in_msh_source.good()) {
            try {
                grid_in_source.read_msh(in_msh_source);
                source_loaded = true;
                std::cout << "Source mesh loaded from MSH format" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load source mesh from MSH: " << e.what() << std::endl;
            }
        }
    }

    if (!source_loaded) {
        throw std::runtime_error("Failed to load source mesh from either VTK or MSH format");
    }

    // Try loading target mesh
    GridIn<dim> grid_in_target;
    grid_in_target.attach_triangulation(target_mesh);
    bool target_loaded = false;

    // First try VTK
    std::ifstream in_vtk_target(directory + "/target.vtk");
    if (in_vtk_target.good()) {
        try {
            grid_in_target.read_vtk(in_vtk_target);
            target_loaded = true;
            std::cout << "Target mesh loaded from VTK format" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed to load target mesh from VTK: " << e.what() << std::endl;
        }
    }

    // If VTK failed, try MSH
    if (!target_loaded) {
        std::ifstream in_msh_target(directory + "/target.msh");
        if (in_msh_target.good()) {
            try {
                grid_in_target.read_msh(in_msh_target);
                target_loaded = true;
                std::cout << "Target mesh loaded from MSH format" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load target mesh from MSH: " << e.what() << std::endl;
            }
        }
    }

    if (!target_loaded) {
        throw std::runtime_error("Failed to load target mesh from either VTK or MSH format");
    }

    std::cout << "Source mesh: " << source_mesh.n_active_cells() << " cells, " << source_mesh.n_vertices() << " vertices" << std::endl;
    std::cout << "Target mesh: " << target_mesh.n_active_cells() << " cells, " << target_mesh.n_vertices() << " vertices" << std::endl;
}

// TODO: controllare da qua in poi
template <int dim>
void Convex2Convex<dim>::setup_finite_elements()
{
    // Check if we're using tetrahedral meshes
    bool use_simplex = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh);

    if (use_simplex) {
        // For simplex meshes, use FE_SimplexP and appropriate mapping
        fe_system = std::make_unique<FE_SimplexP<dim>>(1);
        mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    } else {
        // For hexahedral meshes, use FE_Q and MappingQ1
        fe_system = std::make_unique<FE_Q<dim>>(1);
        mapping = std::make_unique<MappingQ1<dim>>();
    }

    dof_handler_target.distribute_dofs(*fe_system);
    dof_handler_source.distribute_dofs(*fe_system);

    source_density.reinit(dof_handler_source.n_dofs());
    source_density = 1.0;

    // Compute actual L1 norm using appropriate quadrature
    std::unique_ptr<Quadrature<dim>> quadrature;
    if (use_simplex) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(solver_params.quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(solver_params.quadrature_order);
    }

    FEValues<dim> fe_values(*mapping, *fe_system, *quadrature,
                           update_values | update_JxW_values);

    std::vector<double> density_values(quadrature->size());
    double l1_norm = 0.0;

    for (const auto &cell : dof_handler_source.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(source_density, density_values);

        for (unsigned int q = 0; q < quadrature->size(); ++q)
        {
            l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
        }
    }

    std::cout << "Source density L1 norm: " << l1_norm << std::endl;
    source_density /= l1_norm; // Normalize to mass 1

    const std::string directory = "output/data_points";
    bool points_loaded = false;

    // Try to load target points from file first
    if (Utils::read_vector(target_points, directory + "/target_points", io_coding))
    {
        target_weights.reinit(target_points.size());
        target_weights = 1.0 / target_points.size();
        points_loaded = true;
        std::cout << "Target points loaded from file" << std::endl;
    }

    // If loading failed, compute and save them
    if (!points_loaded)
    {
        std::map<types::global_dof_index, Point<dim>> support_points_target;
        DoFTools::map_dofs_to_support_points(*mapping, dof_handler_target, support_points_target);
        target_points.clear();
        for (const auto &point_pair : support_points_target)
        {
            target_points.push_back(point_pair.second);
        }
        target_weights.reinit(support_points_target.size());
        target_weights = 1.0 / support_points_target.size();

        // Save the computed points
        Utils::write_vector(target_points, directory + "/target_points", io_coding);
        std::cout << "Target points computed and saved to file" << std::endl;
    }

    // Similar approach for source points
    points_loaded = false;
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

    std::cout << "Setup complete with " << source_points.size() << " source points and "
              << target_points.size() << " target points" << std::endl;

    // Initialize RTree with target points and their indices
    std::vector<IndexedPoint> indexed_points;
    indexed_points.reserve(target_points.size());
    for (std::size_t i = 0; i < target_points.size(); ++i) {
        indexed_points.emplace_back(target_points[i], i);
    }
    target_points_rtree = pack_rtree(indexed_points);

    std::cout << "RTree initialized for target points" << std::endl;
    std::cout << n_levels(target_points_rtree) << std::endl;
}

template <int dim>
void Convex2Convex<dim>::local_assemble_sot(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &scratch_data,
    CopyData &copy_data)
{
    scratch_data.fe_values.reinit(cell);
    const std::vector<Point<dim>> &q_points = scratch_data.fe_values.get_quadrature_points();
    scratch_data.fe_values.get_function_values(source_density, scratch_data.density_values);

    copy_data.functional_value = 0.0;
    copy_data.gradient_values = 0;

    // Get cell bounding box and extend it by the current distance threshold
    Point<dim> min_point = cell->vertex(0);
    Point<dim> max_point = min_point;
    
    // Find bounding box of the cell
    for (unsigned int v = 1; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        const Point<dim>& vertex = cell->vertex(v);
        for (unsigned int d = 0; d < dim; ++d) {
            min_point[d] = std::min(min_point[d], vertex[d]);
            max_point[d] = std::max(max_point[d], vertex[d]);
        }
    }
    
    // Extend bounding box by the distance threshold
    for (unsigned int d = 0; d < dim; ++d) {
        min_point[d] -= current_distance_threshold;
        max_point[d] += current_distance_threshold;
    }
    
    // Find target points within the extended bounding box
    std::vector<std::size_t> cell_target_indices;
    BoundingBox<dim> extended_box(std::make_pair(min_point, max_point));
    cell_target_indices = find_target_points_in_box(extended_box);

    // Debug tracking - only track total target points if debug is enabled
    if (solver_params.debug) {
        total_target_points += cell_target_indices.size();
    }

    // Process each quadrature point using the precomputed target indices
    for (unsigned int q = 0; q < q_points.size(); ++q)
    {
        const Point<dim> &x = q_points[q];
        double sum_exp = 0.0;
        std::vector<double> exp_terms;
        exp_terms.resize(cell_target_indices.size());

        // Compute exp terms for the precomputed target points
        for (size_t i = 0; i < cell_target_indices.size(); ++i)
        {
            const size_t idx = cell_target_indices[i];
            const double dist2 = (x - target_points[idx]).norm_square();
            
            // Only include points within the actual distance threshold
            if (dist2 <= current_distance_threshold * current_distance_threshold) {
                exp_terms[i] = target_weights[idx] * std::exp(((*current_weights)[idx] - 0.5 * dist2) / current_lambda);
                sum_exp += exp_terms[i];
            } else {
                exp_terms[i] = 0.0;
            }
        }

        copy_data.functional_value += scratch_data.density_values[q] * current_lambda * std::log(sum_exp) * scratch_data.fe_values.JxW(q);

        // Update gradient only for points within threshold
        for (size_t i = 0; i < cell_target_indices.size(); ++i)
        {
            if (exp_terms[i] > 0.0) {  // Only process points that were within threshold
                const size_t idx = cell_target_indices[i];
                copy_data.gradient_values[idx] += scratch_data.density_values[q] * (exp_terms[i] / sum_exp) * scratch_data.fe_values.JxW(q);
            }
        }
    }
}

template <int dim>
void Convex2Convex<dim>::copy_local_to_global(const CopyData &copy_data)
{
    std::lock_guard<std::mutex> lock(assembly_mutex);
    global_functional += copy_data.functional_value;
    global_gradient += copy_data.gradient_values;
}

template <int dim>
double Convex2Convex<dim>::evaluate_sot_functional(const Vector<double> &weights, Vector<double> &gradient)
{
    // Store current weights and lambda for parallel access
    current_weights = &weights;
    current_lambda = solver_params.regularization_param;
    
    // Compute the distance threshold for this iteration
    compute_distance_threshold();
    
    // Reset global values
    global_functional = 0.0;
    global_gradient = 0;
    global_gradient.reinit(target_points.size());

    // Debug tracking variables
    if (solver_params.debug) {
        total_target_points = 0;
    }

    // Check if we're using tetrahedral meshes
    bool use_simplex = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh);

    // Use appropriate quadrature
    std::unique_ptr<Quadrature<dim>> quadrature;
    if (use_simplex) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(solver_params.quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(solver_params.quadrature_order);
    }

    // Create lambda function for WorkStream
    auto worker = [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                        ScratchData &scratch_data,
                        CopyData &copy_data) {
        this->local_assemble_sot(cell, scratch_data, copy_data);
    };

    auto copier = [this](const CopyData &copy_data) {
        this->copy_local_to_global(copy_data);
    };

    // Create scratch and copy data objects
    ScratchData scratch_data(*fe_system, *mapping, *quadrature);
    CopyData copy_data(target_points.size());

    // Run parallel assembly
    WorkStream::run(dof_handler_source.begin_active(),
                   dof_handler_source.end(),
                   worker,
                   copier,
                   scratch_data,
                   copy_data);

    // Add linear term
    for (unsigned int i = 0; i < target_points.size(); ++i)
    {
        global_functional -= weights[i] * target_weights[i];
        global_gradient[i] -= target_weights[i];
    }

    // Debug output - calculate total quadrature points only once
    if (solver_params.debug) {
        const unsigned int points_per_cell = use_simplex ? 
            QGaussSimplex<dim>(solver_params.quadrature_order).size() :
            QGauss<dim>(solver_params.quadrature_order).size();
        const unsigned int total_quad_points = dof_handler_source.get_triangulation().n_active_cells() * points_per_cell;
        double avg_targets = static_cast<double>(total_target_points) / total_quad_points;
        std::cout << "Debug: Average target points per quadrature point: " << avg_targets 
                  << " (Total targets: " << total_target_points 
                  << ", Total quad points: " << total_quad_points << ")" << std::endl;
    }

    // Copy results to output gradient
    gradient = global_gradient;

    return global_functional;
}

template <int dim>
void Convex2Convex<dim>::run_sot()
{
    // Set number of threads based on parameter
    unsigned int n_threads = solver_params.number_of_threads;
    if (n_threads == 0) {
        n_threads = MultithreadInfo::n_cores();
    }
    MultithreadInfo::set_thread_limit(n_threads);
    std::cout << "Running parallel SOT with " << n_threads << " threads" << std::endl;

    Timer timer;
    timer.start();

    setup_finite_elements();

    std::cout << "Starting SOT optimization with " << target_points.size()
              << " target points..." << std::endl;

    Vector<double> weights(target_points.size());
    Vector<double> gradient(target_points.size());

    // Define VerboseSolverControl class
    class VerboseSolverControl : public SolverControl
    {
    public:
        VerboseSolverControl(unsigned int n, double tol)
            : SolverControl(n, tol) {}

        virtual State check(unsigned int step, double value) override
        {
            std::cout << "Iteration " << step
                      << " - Function value: " << value
                      << " - Relative residual: " << value / initial_value() << std::endl;
            return SolverControl::check(step, value);
        }
    };

    // Create solver control and store it in the class member
    solver_control = std::make_unique<VerboseSolverControl>(
        solver_params.max_iterations,
        solver_params.tolerance
    );

    if (!solver_params.verbose_output)
    {
        solver_control->log_history(false);
        solver_control->log_result(false);
    }

    SolverBFGS<Vector<double>> solver(*solver_control);

    try
    {
        std::cout << "Using regularization parameter λ = "
                  << solver_params.regularization_param << std::endl;
        solver.solve(
            [&](const Vector<double> &w, Vector<double> &grad) {
                return evaluate_sot_functional(w, grad);
            },
            weights
        );

        timer.stop();
        std::cout << "\nOptimization completed successfully!" << std::endl;
        std::cout << "Total solver time: " << timer.wall_time() << " seconds" << std::endl;
        std::cout << "Final number of iterations: " << solver_control->last_step() << std::endl;
        std::cout << "Final function value: " << solver_control->last_value() << std::endl;

        save_results(weights, "weights");
        std::cout << "Results saved to weights" << std::endl;
    }
    catch (SolverControl::NoConvergence &exc)
    {
        timer.stop();
        // Save results before reporting the error
        save_results(weights, "weights");
        std::cout << "\nMaximum iterations reached without convergence." << std::endl;
        std::cout << "Total solver time: " << timer.wall_time() << " seconds" << std::endl;
        std::cout << "Final number of iterations: " << solver_control->last_step() << std::endl;
        std::cout << "Final function value: " << solver_control->last_value() << std::endl;
        std::cout << "Intermediate results saved to weights" << std::endl;

        // Re-throw the exception
        throw;
    }
    catch (std::exception &exc)
    {
        timer.stop();
        std::cout << "Total solver time: " << timer.wall_time() << " seconds" << std::endl;
        std::cerr << "Error in SOT computation: " << exc.what() << std::endl;
    }
}

template <int dim>
void Convex2Convex<dim>::save_results(const Vector<double> &weights,
                                     const std::string &filename)
{
    // Create epsilon-specific directory
    std::string eps_dir = "output/epsilon_" + std::to_string(solver_params.regularization_param);
    fs::create_directories(eps_dir);

    // Save weights
    std::vector<double> weights_vec(weights.begin(), weights.end());
    Utils::write_vector(weights_vec, eps_dir + "/" + filename, io_coding);

    // Save convergence info if solver_control exists
    if (solver_control) {
        std::ofstream conv_info(eps_dir + "/convergence_info.txt");
        conv_info << "Regularization parameter (λ): " << solver_params.regularization_param << "\n";
        conv_info << "Number of iterations: " << solver_control->last_step() << "\n";
        conv_info << "Final function value: " << solver_control->last_value() << "\n";
        conv_info << "Convergence achieved: " << (solver_control->last_check() == SolverControl::success) << "\n";
    }
}

template <int dim>
void Convex2Convex<dim>::compute_power_diagram()
{
    load_meshes();
    setup_finite_elements();

    // Let user select which folder(s) to use
    std::vector<std::string> selected_folders;
    try {
        selected_folders = Utils::select_folder();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }

    // Process each selected folder
    for (const auto& selected_folder : selected_folders) {
        std::cout << "\nProcessing folder: " << selected_folder << std::endl;

        // Read weights from selected folder's results
        std::vector<double> weights_vec;
        bool success = Utils::read_vector(weights_vec, "output/" + selected_folder + "/weights", io_coding);
        if (!success) {
            std::cerr << "Failed to read weights from output/" << selected_folder << "/weights" << std::endl;
            continue; // Skip to next folder
        }

        if (weights_vec.size() != target_points.size()) {
            std::cerr << "Error: Mismatch between weights size (" << weights_vec.size()
                      << ") and target points size (" << target_points.size() << ")" << std::endl;
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
                    std::cout << "Using Geogram implementation for power diagram" << std::endl;
                } else {
                    std::cerr << "Geogram implementation is only available for 3D problems" << std::endl;
                    std::cout << "Falling back to Deal.II implementation" << std::endl;
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
                std::cout << "Using Deal.II implementation for power diagram" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize " << power_diagram_params.implementation 
                      << " implementation: " << e.what() << std::endl;
            if (power_diagram_params.implementation == "geogram") {
                std::cout << "Falling back to Deal.II implementation" << std::endl;
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

        std::cout << "Power diagram computation completed for " << selected_folder << std::endl;
        std::cout << "Results saved in " << output_dir << std::endl;
    }

    if (selected_folders.size() > 1) {
        std::cout << "\nCompleted power diagram computation for all selected folders." << std::endl;
    }
}

template <int dim>
template <int d>
typename std::enable_if<d == 3>::type Convex2Convex<dim>::run_exact_sot()
{
    std::cout << "Running exact SOT computation..." << std::endl;

    ExactSot exact_solver;

    // Set source mesh and target points
    if (!exact_solver.set_source_mesh("output/data_mesh/source.msh")) {
        std::cerr << "Failed to load source mesh for exact SOT" << std::endl;
        return;
    }

    // Load target points from the same location as used in setup_finite_elements
    const std::string directory = "output/data_points";
    if (!exact_solver.set_target_points(directory + "/target_points", io_coding)) {
        std::cerr << "Failed to load target points for exact SOT" << std::endl;
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
        std::cerr << "Exact SOT computation failed" << std::endl;
        return;
    }

    // Save results
    if (!exact_solver.save_results(
            output_dir + "/weights",
            output_dir + "/points")) {
        std::cerr << "Failed to save exact SOT results" << std::endl;
        return;
    }

    std::cout << "Exact SOT computation completed successfully" << std::endl;
}

template <int dim>
void Convex2Convex<dim>::save_discrete_measures()
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
    std::vector<double> target_weights_vec(target_weights.begin(), target_weights.end());
    Utils::write_vector(target_weights_vec, directory + "/target_weights", io_coding);

    // Save metadata
    std::ofstream meta(directory + "/metadata.txt");
    meta << "Dimension: " << dim << "\n"
         << "Number of quadrature points per cell: " << n_q_points << "\n"
         << "Total number of quadrature points: " << total_q_points << "\n"
         << "Number of target points: " << target_points.size() << "\n"
         << "Quadrature order: " << solver_params.quadrature_order << "\n"
         << "Using tetrahedral mesh: " << source_params.use_tetrahedral_mesh << "\n";
    meta.close();

    std::cout << "Discrete measures data saved in " << directory << std::endl;
    std::cout << "Total quadrature points: " << total_q_points << std::endl;
    std::cout << "Number of target points: " << target_points.size() << std::endl;
}

template <int dim>
void Convex2Convex<dim>::compute_distance_threshold() const
{
    // Compute the maximum weight (ψⱼ) from current_weights if available
    double max_weight = 0.0;
    if (current_weights != nullptr) {
        max_weight = *std::max_element(current_weights->begin(), current_weights->end());
    }
    
    // Find the minimum target weight (νⱼ)
    double min_target_weight = *std::min_element(target_weights.begin(), target_weights.end());
    
    // Compute the actual distance threshold based on the formula
    // |x-yⱼ|² ≥ -2λlog(ε/νⱼ) + 2ψⱼ
    double lambda = solver_params.regularization_param;
    double epsilon = solver_params.epsilon;
    
    // Using the most conservative case:
    // - maximum weight (ψⱼ) for positive contribution
    // - minimum target weight (νⱼ) for the log term
    double squared_threshold = -2.0 * lambda * std::log(epsilon/min_target_weight) + 2.0 * max_weight;
    current_distance_threshold = std::sqrt(std::max(0.0, squared_threshold));
}

template <int dim>
std::vector<std::size_t> Convex2Convex<dim>::find_nearest_target_points(
    const Point<dim>& query_point) const
{
    namespace bgi = boost::geometry::index;
    std::vector<std::size_t> indices;
    
    // Query all points within the precomputed threshold
    for (const auto& indexed_point : target_points_rtree | 
         bgi::adaptors::queried(bgi::satisfies([&](const IndexedPoint& p) {
             return (p.first - query_point).norm() <= current_distance_threshold;
         })))
    {
        indices.push_back(indexed_point.second);
    }
    
    return indices;
}

template <int dim>
std::vector<std::size_t> Convex2Convex<dim>::find_target_points_in_box(
    const BoundingBox<dim>& box) const
{
    namespace bgi = boost::geometry::index;
    std::vector<std::size_t> indices;
    
    // Query points that intersect with the box
    for (const auto& indexed_point : target_points_rtree | 
         bgi::adaptors::queried(bgi::intersects(box)))
    {
        indices.push_back(indexed_point.second);
    }
    
    return indices;
}

template <int dim>
void Convex2Convex<dim>::run()
{
    print_parameters();

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
    else if (selected_task == "exact_sot")
    {
        if constexpr (dim == 3) {
            load_meshes();
            setup_finite_elements();
            run_exact_sot();
        } else {
            std::cerr << "Exact SOT is only available for 3D problems" << std::endl;
        }
    }
    else if (selected_task == "power_diagram")
    {
        compute_power_diagram();
    }
    else if (selected_task == "save_discrete_measures")
    {
        save_discrete_measures();
    }
    else
    {
        std::cout << "No valid task selected" << std::endl;
    }
}

template class Convex2Convex<2>;
template class Convex2Convex<3>;
