#include "rsot.h"
#include "PowerDiagram.h"
#include "utils.h"
#include "ExactSot.h"
#include "OptimalTransportPlan.h"
#include "PointCloudHierarchy.h"
#include <deal.II/base/timer.h>
#include <filesystem>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/vector_operations_internal.h>
#include <deal.II/grid/grid_tools.h>
namespace fs = std::filesystem;

// Define VerboseSolverControl class
class VerboseSolverControl : public SolverControl
{
public:
    VerboseSolverControl(unsigned int n, double tol, ConditionalOStream& pcout_)
        : SolverControl(n, tol), pcout(pcout_) {}

    virtual State check(unsigned int step, double value) override
    {
        pcout << "Iteration " << step
                  << " - Function value: " << value
                  << " - Relative residual: " << value / initial_value() << std::endl;
        return SolverControl::check(step, value);
    }
private:
    ConditionalOStream& pcout;
};

template <int dim>
Convex2Convex<dim>::Convex2Convex(const MPI_Comm &comm)
    : ParameterAcceptor("Convex2Convex"),
      mpi_communicator(comm),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(comm)),
      this_mpi_process(Utilities::MPI::this_mpi_process(comm)),
      pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
      source_mesh(comm),  // fullydistributed triangulation only takes MPI comm
      target_mesh(),  // Regular triangulation without MPI
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
    
    // Add target multilevel parameters
    enter_subsection("target_multilevel_parameters");
    {
        add_parameter("min_points", target_multilevel_params.min_points,
                     "Minimum number of points for the coarsest level");
        add_parameter("max_points", target_multilevel_params.max_points,
                     "Maximum number of points for the finest level");
        add_parameter("hierarchy_output_dir", target_multilevel_params.hierarchy_output_dir,
                     "Directory to store target point cloud hierarchy");
        add_parameter("output_prefix", target_multilevel_params.output_prefix,
                     "Prefix for target multilevel outputs");
        add_parameter("enabled", target_multilevel_params.enabled,
                     "Whether to use target multilevel approach");
        add_parameter("use_softmax_weight_transfer", target_multilevel_params.use_softmax_weight_transfer,
                     "Whether to use softmax-based weight transfer");
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

        add_parameter("tau",
                     solver_params.tau,
                     "Truncation error tolerance for integral radius bound");

        add_parameter("verbose_output",
                     solver_params.verbose_output,
                     "Enable detailed solver output");

        add_parameter("solver_type",
                     solver_params.solver_type,
                     "Type of optimization solver (BFGS)");

        add_parameter("quadrature_order",
                     solver_params.quadrature_order,
                     "Order of quadrature formula for numerical integration");

        add_parameter("number_of_threads",
                     solver_params.number_of_threads,
                     "Number of threads to use for parallel SOT");

        add_parameter("use_epsilon_scaling",
                     solver_params.use_epsilon_scaling,
                     "Enable epsilon scaling strategy");

        add_parameter("epsilon_scaling_factor",
                     solver_params.epsilon_scaling_factor,
                     "Factor by which to reduce epsilon in each scaling step");

        add_parameter("epsilon_scaling_steps",
                     solver_params.epsilon_scaling_steps,
                     "Number of epsilon scaling steps");

        add_parameter("use_caching",
                     solver_params.use_caching,
                     "Enable distance threshold caching");
    }
    leave_subsection();

    enter_subsection("power_diagram_parameters");
    {
        add_parameter("implementation",
                     power_diagram_params.implementation,
                     "Implementation to use for power diagram computation (dealii/geogram)");
    }
    leave_subsection();

    enter_subsection("transport_map_parameters");
    {
        add_parameter("n_neighbors",
                     transport_map_params.n_neighbors,
                     "Number of neighbors for local methods");
        add_parameter("kernel_width",
                     transport_map_params.kernel_width,
                     "Kernel width for smooth approximations");
        add_parameter("interpolation_type",
                     transport_map_params.interpolation_type,
                     "Type of interpolation");
    }
    leave_subsection();

    enter_subsection("multilevel_parameters");
    {
        add_parameter("min_vertices",
                     multilevel_params.min_vertices,
                     "Minimum number of vertices for the coarsest level");
        
        add_parameter("max_vertices",
                     multilevel_params.max_vertices,
                     "Maximum number of vertices for level 1 mesh");
        
        add_parameter("hierarchy_output_dir",
                     multilevel_params.hierarchy_output_dir,
                     "Directory to store the mesh hierarchy");

        add_parameter("output_prefix",
                     multilevel_params.output_prefix,
                     "Directory prefix for multilevel SOT results");
    }
    leave_subsection();
}

template <int dim>
void Convex2Convex<dim>::print_parameters()
{
    pcout << "Selected Task: " << selected_task << std::endl;
    pcout << "I/O Coding: " << io_coding << std::endl;

    pcout << "Source Mesh Parameters:" << std::endl;
    pcout << "  Grid Generator Function: " << source_params.grid_generator_function << std::endl;
    pcout << "  Grid Generator Arguments: " << source_params.grid_generator_arguments << std::endl;
    pcout << "  Number of Refinements: " << source_params.n_refinements << std::endl;

    pcout << "Target Mesh Parameters:" << std::endl;
    pcout << "  Grid Generator Function: " << target_params.grid_generator_function << std::endl;
    pcout << "  Grid Generator Arguments: " << target_params.grid_generator_arguments << std::endl;
    pcout << "  Number of Refinements: " << target_params.n_refinements << std::endl;

    pcout << "RSOT Solver Parameters:" << std::endl;
    pcout << "  Max Iterations: " << solver_params.max_iterations << std::endl;
    pcout << "  Tolerance: " << solver_params.tolerance << std::endl;
    pcout << "  Regularization Parameter (λ): " << solver_params.regularization_param << std::endl;
    pcout << "  Verbose Output: " << (solver_params.verbose_output ? "Yes" : "No") << std::endl;
    pcout << "  Solver Type: " << solver_params.solver_type << std::endl;
    pcout << "  Quadrature Order: " << solver_params.quadrature_order << std::endl;
    pcout << "  Number of Threads: " << solver_params.number_of_threads << std::endl;
}

template <int dim>
template <typename TriangulationType>
void Convex2Convex<dim>::generate_mesh(TriangulationType &tria,
                                     const std::string &grid_generator_function,
                                     const std::string &grid_generator_arguments,
                                     const unsigned int n_refinements,
                                     const bool use_tetrahedral_mesh)
{
    if constexpr (std::is_same_v<TriangulationType, parallel::fullydistributed::Triangulation<dim>>) {
        // For fullydistributed triangulation, first create a serial triangulation
        Triangulation<dim> serial_tria;
        GridGenerator::generate_from_name_and_arguments(
            serial_tria,
            grid_generator_function,
            grid_generator_arguments);

        if (use_tetrahedral_mesh && dim == 3) {
            GridGenerator::convert_hypercube_to_simplex_mesh(serial_tria, serial_tria);
        }

        serial_tria.refine_global(n_refinements);

        // Set up the partitioner to use z-order curve
        tria.set_partitioner([](Triangulation<dim> &tria_to_partition, const unsigned int n_partitions) {
            GridTools::partition_triangulation_zorder(n_partitions, tria_to_partition);
        }, TriangulationDescription::Settings::construct_multigrid_hierarchy);

        // Create the construction data
        auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(
            serial_tria, mpi_communicator,
            TriangulationDescription::Settings::construct_multigrid_hierarchy);

        // Actually create the distributed triangulation
        tria.create_triangulation(construction_data);
    } else {
        // For regular triangulation, use the original code
        GridGenerator::generate_from_name_and_arguments(
            tria,
            grid_generator_function,
            grid_generator_arguments);

        if (use_tetrahedral_mesh && dim == 3) {
            GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria);
        }

        tria.refine_global(n_refinements);
    }
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

    pcout << "Meshes saved in VTK and MSH formats" << std::endl;
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
void Convex2Convex<dim>::load_mesh_source()
{
    const std::string directory = "output/data_mesh";

    // First load source mesh into a serial triangulation
    Triangulation<dim> serial_source;
    GridIn<dim> grid_in_source;
    grid_in_source.attach_triangulation(serial_source);
    bool source_loaded = false;
    // First try VTK
    std::ifstream in_vtk_source(directory + "/source.vtk");
    if (in_vtk_source.good()) {
        try {
            grid_in_source.read_vtk(in_vtk_source);
            source_loaded = true;
            pcout << "Source mesh loaded from VTK format" << std::endl;
        } catch (const std::exception& e) {
            pcout << "Failed to load source mesh from VTK format: " << e.what() << std::endl;
        }
    }

    // If VTK failed, try MSH
    if (!source_loaded) {
        std::ifstream in_msh_source(directory + "/source.msh");
        if (in_msh_source.good()) {
            try {
                grid_in_source.read_msh(in_msh_source);
                source_loaded = true;
                pcout << "Source mesh loaded from MSH format" << std::endl;
            } catch (const std::exception& e) {
                pcout << "Failed to load source mesh from MSH format: " << e.what() << std::endl;
            }
        }
    }

    if (!source_loaded) {
        throw std::runtime_error("Failed to load source mesh from either VTK or MSH format");
    }

    // Partition the serial source mesh using z-order curve
    GridTools::partition_triangulation_zorder(n_mpi_processes, serial_source);

    // Convert serial source mesh to fullydistributed without multigrid hierarchy
    auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(
        serial_source, mpi_communicator,
        TriangulationDescription::Settings::default_setting);
    source_mesh.create_triangulation(construction_data);
}

template <int dim>
void Convex2Convex<dim>::load_mesh_target()
{
    // Only rank 0 loads the target mesh
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
        return;
    }

    const std::string directory = "output/data_mesh";
    
    // Load target mesh (stays serial)
    GridIn<dim> grid_in_target;
    grid_in_target.attach_triangulation(target_mesh);
    bool target_loaded = false;

    // Try VTK for target
    std::ifstream in_vtk_target(directory + "/target.vtk");
    if (in_vtk_target.good()) {
        try {
            grid_in_target.read_vtk(in_vtk_target);
            target_loaded = true;
            pcout << "Target mesh loaded from VTK format" << std::endl;
        } catch (const std::exception& e) {
            pcout << "Failed to load target mesh from VTK format: " << e.what() << std::endl;
        }
    }

    // If VTK failed, try MSH for target
    if (!target_loaded) {
        std::ifstream in_msh_target(directory + "/target.msh");
        if (in_msh_target.good()) {
            try {
                grid_in_target.read_msh(in_msh_target);
                target_loaded = true;
                pcout << "Target mesh loaded from MSH format" << std::endl;
            } catch (const std::exception& e) {
                pcout << "Failed to load target mesh from MSH format: " << e.what() << std::endl;
            }
        }
    }

    if (!target_loaded) {
        throw std::runtime_error("Failed to load target mesh from either VTK or MSH format");
    }
}

template <int dim>
void Convex2Convex<dim>::load_meshes()
{
    load_mesh_source();
    load_mesh_target();

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
void Convex2Convex<dim>::setup_source_finite_elements()
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
void Convex2Convex<dim>::setup_target_finite_elements()
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
    // Initialize RTree with target points and their indices
    std::vector<IndexedPoint> indexed_points;
    indexed_points.reserve(target_points.size());
    for (std::size_t i = 0; i < target_points.size(); ++i) {
        indexed_points.emplace_back(target_points[i], i);
    }
    target_points_rtree = RTree(indexed_points.begin(), indexed_points.end());

    pcout << "RTree initialized for target points" << std::endl;
    pcout << n_levels(target_points_rtree) << std::endl;


}


template <int dim>
void Convex2Convex<dim>::setup_finite_elements()
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
void Convex2Convex<dim>::setup_target_points()
{
    load_mesh_target();
    setup_target_finite_elements();
}


template <int dim>
void Convex2Convex<dim>::local_assemble_sot(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &scratch_data,
    CopyData &copy_data)
{
    if (!cell->is_locally_owned())
        return;

    scratch_data.fe_values.reinit(cell);
    const std::vector<Point<dim>> &q_points = scratch_data.fe_values.get_quadrature_points();
    scratch_data.fe_values.get_function_values(source_density, scratch_data.density_values);

    copy_data.functional_value = 0.0;
    copy_data.gradient_values = 0;
    copy_data.C_integral = 0.0;  // Reset C integral contribution

    const unsigned int n_q_points = q_points.size();
    const double lambda_inv = 1.0 / current_lambda;

    if (solver_params.use_caching && is_caching_active) {
        // Caching path
        CellCache& cell_cache = [&]() -> CellCache& {
            std::lock_guard<std::mutex> lock(cache_mutex);
            return cell_caches[cell->id().to_string()];
        }();

        std::vector<std::size_t> cell_target_indices;
        bool need_computation = true;

        if (cell_cache.is_valid) {
            cell_target_indices = cell_cache.target_indices;
            need_computation = false;
        } else {
           cell_target_indices = find_nearest_target_points(cell->center());
            
            if (cell_target_indices.empty()) return;
            
            cell_cache.target_indices = cell_target_indices;
            cell_cache.precomputed_exp_terms.resize(n_q_points * cell_target_indices.size());
        }

        const unsigned int n_target_points = cell_target_indices.size();
        // Preload target data for vectorization
        std::vector<Point<dim>> target_positions(n_target_points);
        std::vector<double> target_densities(n_target_points);
        std::vector<double> weight_values(n_target_points);
        
        for (size_t i = 0; i < n_target_points; ++i) {
            const size_t idx = cell_target_indices[i];
            target_positions[i] = target_points[idx];
            target_densities[i] = target_density[idx];
            weight_values[i] = (*current_weights)[idx];
        }

        // Process quadrature points with caching
        for (unsigned int q = 0; q < n_q_points; ++q) {
            const Point<dim> &x = q_points[q];
            const double density_value = scratch_data.density_values[q];
            const double JxW = scratch_data.fe_values.JxW(q);

            double total_sum_exp = 0.0;
            std::vector<double> active_exp_terms(n_target_points, 0.0);

            const unsigned int base_idx = q * n_target_points;
            if (need_computation) {
                #pragma omp simd reduction(+:total_sum_exp)
                for (size_t i = 0; i < n_target_points; ++i) {
                    const double local_dist2 = (x - target_positions[i]).norm_square();
                    const double precomputed_term = target_densities[i] * 
                        std::exp(-0.5 * local_dist2 * lambda_inv);
                    cell_cache.precomputed_exp_terms[base_idx + i] = precomputed_term;
                    active_exp_terms[i] = precomputed_term * std::exp(weight_values[i] * lambda_inv);
                    total_sum_exp += active_exp_terms[i];
                }
            } else {
                #pragma omp simd reduction(+:total_sum_exp)
                for (size_t i = 0; i < n_target_points; ++i) {
                    const double cached_term = cell_cache.precomputed_exp_terms[base_idx + i];
                    active_exp_terms[i] = cached_term * std::exp(weight_values[i] * lambda_inv);
                    total_sum_exp += active_exp_terms[i];
                }
            }

            if (total_sum_exp <= 0.0) continue;

            copy_data.functional_value += density_value * current_lambda * 
                std::log(total_sum_exp) * JxW;

            // Add contribution to C integral
            copy_data.C_integral += density_value * JxW / total_sum_exp;

            const double scale = density_value * JxW / total_sum_exp;
            #pragma omp simd
            for (size_t i = 0; i < n_target_points; ++i) {
                if (active_exp_terms[i] > 0.0) {
                    copy_data.gradient_values[cell_target_indices[i]] += 
                        scale * active_exp_terms[i];
                }
            }
        }

        if (need_computation) {
            cell_cache.is_valid = true;
        }

    } else {
        // Direct computation path without caching overhead
        std::vector<std::size_t> cell_target_indices = find_nearest_target_points(cell->center());
        const unsigned int n_target_points = cell_target_indices.size();
        std::vector<Point<dim>> target_positions(n_target_points);
        std::vector<double> target_densities(n_target_points);
        std::vector<double> weight_values(n_target_points);
        
        for (size_t i = 0; i < n_target_points; ++i) {
            const size_t idx = cell_target_indices[i];
            target_positions[i] = target_points[idx];
            target_densities[i] = target_density[idx];
            weight_values[i] = (*current_weights)[idx];
        }
        for (unsigned int q = 0; q < n_q_points; ++q) {
            const Point<dim> &x = q_points[q];
            const double density_value = scratch_data.density_values[q];
            const double JxW = scratch_data.fe_values.JxW(q);
            
            if (cell_target_indices.empty()) continue;

            double total_sum_exp = 0.0;
            std::vector<double> exp_terms(n_target_points);

            #pragma omp simd reduction(+:total_sum_exp)
            for (size_t i = 0; i < n_target_points; ++i) {
                const double local_dist2 = (x - target_positions[i]).norm_square();
                exp_terms[i] = target_densities[i] * 
                    std::exp((weight_values[i] - 0.5 * local_dist2) * lambda_inv);
                total_sum_exp += exp_terms[i];
            }

            if (total_sum_exp <= 0.0) continue;

            copy_data.functional_value += density_value * current_lambda * 
                std::log(total_sum_exp) * JxW;

            // Add contribution to C integral
            copy_data.C_integral += density_value * JxW / total_sum_exp;

            const double scale = density_value * JxW / total_sum_exp;
            #pragma omp simd
            for (size_t i = 0; i < n_target_points; ++i) {
                if (exp_terms[i] > 0.0) {
                    copy_data.gradient_values[cell_target_indices[i]] += scale * exp_terms[i];
                }
            }
        }
    }
}

template <int dim>
Vector<double> Convex2Convex<dim>::softmax_refinement(const Vector<double> &weights)
{
    weights_coarse = &weights;
    current_lambda = solver_params.regularization_param;
    target_points_fine = target_points;

    // Ensure target_points_fine is not empty
    if (target_points_fine.empty()) {
        pcout << "Error: No fine level points loaded for softmax refinement" << std::endl;
        return Vector<double>();
    }

    Vector<double> local_process_weights(target_points_fine.size());
    weights_fine.reinit(target_points_fine.size());

    // Use appropriate quadrature
    std::unique_ptr<Quadrature<dim>> quadrature;
    if (source_params.use_tetrahedral_mesh) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(solver_params.quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(solver_params.quadrature_order);
    }

    // Create scratch and copy data objects
    ScratchData scratch_data(*fe_system, *mapping, *quadrature);
    CopyData copy_data(target_points_fine.size());

    // Create filtered iterator for locally owned cells
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        begin_filtered(IteratorFilters::LocallyOwnedCell(),
                      dof_handler_source.begin_active()),
        end_filtered(IteratorFilters::LocallyOwnedCell(),
                    dof_handler_source.end());

    // Parallel assembly
    WorkStream::run(
        begin_filtered,
        end_filtered,
        [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
               ScratchData &scratch_data,
               CopyData &copy_data) {
            this->local_assemble_softmax_refinement(cell, scratch_data, copy_data);
        },
        [&local_process_weights](const CopyData &copy_data) {
            local_process_weights += copy_data.weight_values;
        },
        scratch_data,
        copy_data);

    // Sum up contributions across all MPI processes
    weights_fine = 0;
    Utilities::MPI::sum(local_process_weights, mpi_communicator, weights_fine);

    // Apply epsilon scaling to weights
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        for (unsigned int i = 0; i < target_points_fine.size(); ++i) {
            if (weights_fine[i] > 0.0) {
                weights_fine[i] = -solver_params.regularization_param * std::log(weights_fine[i]);
            }
        }
    }

    // Broadcast final weights to all processes
    weights_fine = Utilities::MPI::broadcast(mpi_communicator, weights_fine, 0);
    return weights_fine;
}

template <int dim>
void Convex2Convex<dim>::local_assemble_softmax_refinement(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &scratch_data,
    CopyData &copy_data)
{
    if (!cell->is_locally_owned())
        return;

    scratch_data.fe_values.reinit(cell);
    const std::vector<Point<dim>> &q_points = scratch_data.fe_values.get_quadrature_points();
    scratch_data.fe_values.get_function_values(source_density, scratch_data.density_values);

    copy_data.weight_values = 0;

    const unsigned int n_q_points = q_points.size();
    const double lambda_inv = 1.0 / current_lambda;
    const double threshold_sq = current_distance_threshold * current_distance_threshold;

    // Get relevant coarse target points for this cell
    std::vector<std::size_t> cell_target_indices_coarse = find_nearest_target_points(cell->center());
    
    if (cell_target_indices_coarse.empty()) return;

    const unsigned int n_target_points_coarse = cell_target_indices_coarse.size();
    std::vector<Point<dim>> target_positions_coarse(n_target_points_coarse);
    std::vector<double> target_densities_coarse(n_target_points_coarse);
    std::vector<double> weight_values_coarse(n_target_points_coarse);
    
    // Load coarse target point data
    for (size_t i = 0; i < n_target_points_coarse; ++i) {
        const size_t idx = cell_target_indices_coarse[i];
        target_positions_coarse[i] = target_points_coarse[idx];
        target_densities_coarse[i] = target_density_coarse[idx];
        weight_values_coarse[i] = (*weights_coarse)[idx];
    }

    // Get fine points that are children of the coarse points
    std::vector<std::size_t> cell_target_indices_fine;
    std::vector<Point<dim>> target_positions_fine;
    std::vector<size_t> fine_to_coarse_index;  // Maps fine point index to its parent coarse point index

    // Add bounds checking for child_indices_ access
    if (current_level < 0 || current_level >= static_cast<int>(child_indices_.size())) {
        std::cerr << "Error: Invalid level " << current_level << " for child_indices_ of size " << child_indices_.size() << std::endl;
        return;
    }

    for (size_t i = 0; i < n_target_points_coarse; ++i) {
        const size_t coarse_idx = cell_target_indices_coarse[i];
        if (coarse_idx >= child_indices_[current_level].size()) {
            std::cerr << "Error: Invalid coarse index " << coarse_idx << " for child_indices_[" << current_level << "] of size " << child_indices_[current_level].size() << std::endl;
            continue;
        }
        const auto& children = child_indices_[current_level][coarse_idx];
        
        for (const auto& child_idx : children) {
            if (child_idx >= target_points_fine.size()) {
                std::cerr << "Error: Invalid child index " << child_idx << " for target_points_fine of size " << target_points_fine.size() << std::endl;
                continue;
            }
            cell_target_indices_fine.push_back(child_idx);
            target_positions_fine.push_back(target_points_fine[child_idx]);
        }
    }

    const unsigned int n_target_points_fine = cell_target_indices_fine.size();
    if (n_target_points_fine == 0) {
        std::cerr << "Warning: No valid fine points found for coarse points at level " << current_level << std::endl;
        return;
    }

    // For each quadrature point
    for (unsigned int q = 0; q < n_q_points; ++q) {
        const Point<dim> &x = q_points[q];
        const double density_value = scratch_data.density_values[q];
        const double JxW = scratch_data.fe_values.JxW(q);
        
        // First compute normalization using coarse points
        double total_sum_exp = 0.0;
        std::vector<double> exp_terms_coarse(n_target_points_coarse);

        #pragma omp simd reduction(+:total_sum_exp)
        for (size_t i = 0; i < n_target_points_coarse; ++i) {
            const double local_dist2 = (x - target_positions_coarse[i]).norm_square();
            if (local_dist2 <= threshold_sq) {
                exp_terms_coarse[i] = target_densities_coarse[i] * 
                    std::exp((weight_values_coarse[i] - 0.5 * local_dist2) * lambda_inv);
                total_sum_exp += exp_terms_coarse[i];
            }
        }

        if (total_sum_exp <= 0.0) continue;

        // Now update weights for fine points using their parent's exp term for normalization
        // scale is basically the exp(f potential) computed using the old weights
        const double scale = density_value * JxW / total_sum_exp;
        
        #pragma omp simd
        for (size_t i = 0; i < n_target_points_fine; ++i) {
            const double local_dist2_fine = (x - target_positions_fine[i]).norm_square();
            if (local_dist2_fine <= threshold_sq) {
                const double exp_term_fine = std::exp((- 0.5 * local_dist2_fine) * lambda_inv);
                copy_data.weight_values[cell_target_indices_fine[i]] += scale * exp_term_fine;
            }
        }
    }
}

template <int dim>
double Convex2Convex<dim>::evaluate_sot_functional(const Vector<double> &weights, Vector<double> &gradient)
{
    compute_distance_threshold();
    // Store current weights and lambda for parallel access
    current_weights = &weights;
    current_lambda = solver_params.regularization_param;

    // Reset global values for this MPI process
    double local_process_functional = 0.0;
    double local_C_integral = 0.0;  // Local C integral accumulator
    Vector<double> local_process_gradient(target_points.size());

    // Use appropriate quadrature
    std::unique_ptr<Quadrature<dim>> quadrature;
    if (source_params.use_tetrahedral_mesh) {
        quadrature = std::make_unique<QGaussSimplex<dim>>(solver_params.quadrature_order);
    } else {
        quadrature = std::make_unique<QGauss<dim>>(solver_params.quadrature_order);
    }

    // Create scratch and copy data objects
    ScratchData scratch_data(*fe_system, *mapping, *quadrature);
    CopyData copy_data(target_points.size());

    // Create filtered iterator for locally owned cells
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        begin_filtered(IteratorFilters::LocallyOwnedCell(),
                      dof_handler_source.begin_active()),
        end_filtered(IteratorFilters::LocallyOwnedCell(),
                    dof_handler_source.end());

    // Run parallel assembly using WorkStream
    WorkStream::run(begin_filtered,
                   end_filtered,
                   [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                         ScratchData &scratch_data,
                         CopyData &copy_data) {
                       this->local_assemble_sot(cell, scratch_data, copy_data);
                   },
                   [&local_process_functional, &local_process_gradient, &local_C_integral]
                   (const CopyData &copy_data) {
                       local_process_functional += copy_data.functional_value;
                       local_process_gradient += copy_data.gradient_values;
                       local_C_integral += copy_data.C_integral;  // Accumulate C integral locally
                   },
                   scratch_data,
                   copy_data);

    // Sum up contributions across all MPI processes
    global_functional = Utilities::MPI::sum(local_process_functional, mpi_communicator);
    global_C_integral = Utilities::MPI::sum(local_C_integral, mpi_communicator);  // Sum up C integral
    // pcout << "global_C_integral: " << global_C_integral << std::endl;

    // Convert to regular Vector for gradient
    gradient = 0;  // Reset gradient
    Utilities::MPI::sum(local_process_gradient, mpi_communicator, gradient);

    // Add linear term (only on root process to avoid duplication)
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        for (unsigned int i = 0; i < target_points.size(); ++i) {
            global_functional -= weights[i] * target_density[i];
            gradient[i] -= target_density[i];
        }
    }

    // Broadcast final results to all processes
    global_functional = Utilities::MPI::broadcast(mpi_communicator, global_functional, 0);
    gradient = Utilities::MPI::broadcast(mpi_communicator, gradient, 0);
    global_C_integral = Utilities::MPI::broadcast(mpi_communicator, global_C_integral, 0);  // Broadcast C integral

    // compute_distance_threshold();
    

    return global_functional;
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
    double min_target_weight = *std::min_element(target_density.begin(), target_density.end());

    // Compute the actual distance threshold based on the formula
    // |x-yⱼ|² ≥ -2λlog(ε/νⱼ) + 2ψⱼ
    double lambda = solver_params.regularization_param;
    double epsilon = solver_params.epsilon;

    // Using the most conservative case:
    // - maximum weight (ψⱼ) for positive contribution
    // - minimum target weight (νⱼ) for the log term
    double squared_threshold = -2.0 * lambda * std::log(epsilon/min_target_weight) + 2.0 * max_weight;
    double computed_threshold = std::sqrt(std::max(0.0, squared_threshold));

    if (!solver_params.use_caching) {
        // If caching is disabled, just use the computed threshold directly
        current_distance_threshold = computed_threshold;
        is_caching_active = false;
        if (solver_params.verbose_output) {
            pcout << "Using threshold without caching: " << current_distance_threshold << std::endl;
        }
        return;
    }

    // If we have an active cache and the computed threshold is smaller,
    // we can keep using the effective threshold
    if (is_caching_active && computed_threshold <= effective_distance_threshold) {
        current_distance_threshold = effective_distance_threshold;
        if (solver_params.verbose_output) {
            pcout << "Using cached threshold: " << current_distance_threshold << std::endl;
        }
        return;
    }

    // Either cache is not active or computed threshold is larger than effective threshold
    // Update both thresholds and enable caching
    current_distance_threshold = computed_threshold;
    effective_distance_threshold = computed_threshold * 1.1;  // 10% increase
    is_caching_active = true;
    
    if (solver_params.verbose_output) {
        pcout << "Updated thresholds - Current: " << current_distance_threshold 
              << ", Effective: " << effective_distance_threshold << std::endl;
    }
}

// template <int dim>
// void Convex2Convex<dim>::compute_distance_threshold() const
// {
//     // Early exit if no weights available
//     if (current_weights == nullptr) {
//         current_distance_threshold = std::numeric_limits<double>::max();
//         is_caching_active = false;
//         return;
//     }

//     // Calculate M and m (bounds on dual potentials φ)
//     double M = -std::numeric_limits<double>::max();
//     double m = std::numeric_limits<double>::max();
//     for (size_t i = 0; i < current_weights->size(); ++i) {
//         M = std::max(M, (*current_weights)[i]);
//         m = std::min(m, (*current_weights)[i]);
//     }

//     // Get current functional value (F(φ))
//     const double F_phi = global_functional;  // This is set in evaluate_sot_functional

//     // Get C from the last assembly (already computed in parallel)
//     const double C = global_C_integral;

//     // Use the dedicated tau parameter for truncation error control
//     const double tau = solver_params.tau;

//     // Compute the tau integral radius bound
//     // pcout << "M: " << M << " m: " << m << " C: " << C << " F_phi: " << F_phi << std::endl;
//     double computed_threshold = compute_tau_integral_radius(tau, M, C, F_phi);

//     if (!solver_params.use_caching) {
//         // If caching is disabled, just use the computed threshold directly
//         current_distance_threshold = computed_threshold;
//         is_caching_active = false;
//         if (solver_params.verbose_output) {
//             pcout << "Using tau integral radius bound: " << current_distance_threshold << std::endl;
//         }
//         return;
//     }

//     // If we have an active cache and the computed threshold is smaller,
//     // we can keep using the effective threshold
//     if (is_caching_active && computed_threshold <= effective_distance_threshold) {
//         current_distance_threshold = effective_distance_threshold;
//         if (solver_params.verbose_output) {
//             pcout << "Using cached threshold: " << current_distance_threshold << std::endl;
//         }
//         return;
//     }

//     // Either cache is not active or computed threshold is larger than effective threshold
//     // Update both thresholds and enable caching
//     current_distance_threshold = computed_threshold;
//     effective_distance_threshold = computed_threshold * 1.1;  // 10% increase
//     is_caching_active = true;
    
//     if (solver_params.verbose_output) {
//         pcout << "Updated thresholds using tau integral bound - Current: " << current_distance_threshold 
//               << ", Effective: " << effective_distance_threshold << std::endl;
//     }
// }



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
double Convex2Convex<dim>::compute_tau_integral_radius(
    const double tau,
    const double M,
    const double C,
    const double F_phi) const
{
    // Implementation of the tau integral radius bound from the derivation
    // R_int = sqrt(2M + 2λ*ln(λ*C/(τ*|F(φ)|)))
    const double lambda = solver_params.regularization_param;
    const double argument = (lambda * C) / (tau * std::abs(F_phi));
    const double radius_squared = 2.0 * M + 2.0 * lambda * std::log(argument);
    return std::sqrt(radius_squared);
}

template <int dim>
double Convex2Convex<dim>::calculate_cache_size_mb() const {
    double total_size_bytes = 0.0;
    
    for (const auto& cache_entry : cell_caches) {
        total_size_bytes += cache_entry.second.target_indices.size() * sizeof(std::size_t);
        total_size_bytes += cache_entry.second.precomputed_exp_terms.size() * sizeof(double);
        total_size_bytes += sizeof(bool) + sizeof(CellCache);
        total_size_bytes += cache_entry.first.capacity() * sizeof(char);
    }
    
    // Convert to MB
    return total_size_bytes / (1024.0 * 1024.0);
}

template <int dim>
void Convex2Convex<dim>::run_sot()
{
    // Set number of threads based on parameter
    unsigned int n_threads = solver_params.number_of_threads;
    const unsigned int n_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
    current_distance_threshold = 1e-1;

    if (n_threads == 0) {
        // Divide available cores by number of MPI processes
        n_threads = std::max(1U, MultithreadInfo::n_cores() / n_mpi_processes);
    }
    MultithreadInfo::set_thread_limit(n_threads);

    // Get MPI information
    const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int total_threads = n_threads * n_mpi_processes;

    pcout << "Parallel Configuration:" << std::endl
          << "  MPI Processes: " << n_mpi_processes
          << " (Current rank: " << this_mpi_process << ")" << std::endl
          << "  Available cores: " << MultithreadInfo::n_cores() << std::endl
          << "  Threads per process: " << n_threads << std::endl
          << "  Total parallel units: " << total_threads << std::endl;

    Timer timer;
    timer.start();

    setup_finite_elements();

    pcout << "Starting SOT optimization with " << target_points.size()
              << " target points..." << std::endl;

    Vector<double> weights(target_points.size());
    Vector<double> gradient(target_points.size());

    // Store original regularization parameter
    const double original_lambda = solver_params.regularization_param;

    if (solver_params.use_epsilon_scaling) {
        pcout << "Using epsilon scaling strategy with:" << std::endl
              << "  Initial epsilon: " << original_lambda << std::endl
              << "  Scaling factor: " << solver_params.epsilon_scaling_factor << std::endl
              << "  Number of steps: " << solver_params.epsilon_scaling_steps << std::endl;
    }

    // Define VerboseSolverControl class
    class VerboseSolverControl : public SolverControl
    {
    public:
        VerboseSolverControl(unsigned int n, double tol, ConditionalOStream& pcout_)
            : SolverControl(n, tol), pcout(pcout_) {}

        virtual State check(unsigned int step, double value) override
        {
            pcout << "Iteration " << step
                      << " - Function value: " << value
                      << " - Relative residual: " << value / initial_value() << std::endl;
            return SolverControl::check(step, value);
        }
    private:
        ConditionalOStream& pcout;
    };

    // Create solver control and store it in the class member
    solver_control = std::make_unique<VerboseSolverControl>(
        solver_params.max_iterations,
        solver_params.tolerance,
        pcout
    );

    if (!solver_params.verbose_output)
    {
        solver_control->log_history(false);
        solver_control->log_result(false);
    }

    // Function to run a single optimization with current epsilon
    auto run_single_optimization = [&](double current_lambda) {
        // Create solver control for current optimization
        solver_control = std::make_unique<VerboseSolverControl>(
            solver_params.max_iterations,
            solver_params.tolerance,
            pcout
        );

        if (!solver_params.verbose_output) {
            solver_control->log_history(false);
            solver_control->log_result(false);
        }

        SolverBFGS<Vector<double>> solver(*solver_control);

        try {
            pcout << "Using regularization parameter λ = " << current_lambda << std::endl;
            solver_params.regularization_param = current_lambda;  // Set current lambda for functional evaluation

            solver.solve(
                [&](const Vector<double> &w, Vector<double> &grad) {
                    return evaluate_sot_functional(w, grad);
                },
                weights  // Use current weights as initial guess
            );

            pcout << "Optimization step completed:" << std::endl
                  << "  Number of iterations: " << solver_control->last_step() << std::endl
                  << "  Final function value: " << solver_control->last_value() << std::endl;
                  
            // Print cache memory usage at the end of optimization if caching is enabled
            if (solver_params.use_caching) {
                pcout << "  Final cache memory usage: " << calculate_cache_size_mb() << " MB" << std::endl;
                pcout << "  Number of cached cells: " << cell_caches.size() << std::endl;
            }

            reset_distance_threshold_cache();

            return true;
        } catch (SolverControl::NoConvergence &exc) {
            pcout << "Warning: Optimization did not converge for λ = " << current_lambda << std::endl;
            return false;
        }
    };

    bool success = false;

    if (solver_params.use_epsilon_scaling) {
        // Compute sequence of epsilon values
        std::vector<double> epsilon_sequence;
        epsilon_sequence.reserve(solver_params.epsilon_scaling_steps);
        
        for (unsigned int i = 0; i < solver_params.epsilon_scaling_steps; ++i) {
            double scale_factor = std::pow(solver_params.epsilon_scaling_factor, 
                                         solver_params.epsilon_scaling_steps - 1 - i);
            epsilon_sequence.push_back(original_lambda * scale_factor);
        }

        // Run optimization for each epsilon value
        for (size_t i = 0; i < epsilon_sequence.size(); ++i) {
            pcout << "\nEpsilon scaling step " << i + 1 << "/" << epsilon_sequence.size()
                  << " (λ = " << epsilon_sequence[i] << ")" << std::endl;

            success = run_single_optimization(epsilon_sequence[i]);

            // Save intermediate results
            save_results(weights, "weights_eps_" + std::to_string(epsilon_sequence[i]));

            if (!success && i < epsilon_sequence.size() - 1) {
                pcout << "Warning: Optimization failed at step " << i + 1 
                      << ", continuing with next epsilon value" << std::endl;
            }
        }
    } else {
        // Run single optimization with original epsilon
        success = run_single_optimization(original_lambda);
    }

    timer.stop();

    if (success) {
        pcout << "\nOptimization completed successfully!" << std::endl;
    } else {
        pcout << "\nOptimization completed with warnings." << std::endl;
    }

    pcout << "Total solver time: " << timer.wall_time() << " seconds" << std::endl;
    pcout << "Final number of iterations: " << solver_control->last_step() << std::endl;
    pcout << "Final function value: " << solver_control->last_value() << std::endl;

    // Print final memory usage statistics
    if (solver_params.use_caching) {
        pcout << "Final cache memory usage: " << calculate_cache_size_mb() << " MB" << std::endl;
        pcout << "Number of cached cells: " << cell_caches.size() << std::endl;
    }

    // Save final results
    save_results(weights, "weights");
    pcout << "Final results saved to weights" << std::endl;

    // Restore original regularization parameter
    solver_params.regularization_param = original_lambda;
}


template <int dim>
void Convex2Convex<dim>::prepare_multilevel()
{
    pcout << "Preparing multilevel mesh hierarchy..." << std::endl;

    // Create MeshHierarchyManager instance
    MeshHierarchy::MeshHierarchyManager hierarchy_manager(
        multilevel_params.min_vertices,
        multilevel_params.max_vertices
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

        pcout << "Successfully generated mesh hierarchy with " << num_levels << " levels" << std::endl;
        pcout << "Mesh hierarchy saved in: " << hierarchy_dir << std::endl;
        pcout << "Level 1 (finest) vertices: " << multilevel_params.max_vertices << std::endl;
        pcout << "Coarsest level vertices: ~" << multilevel_params.min_vertices << std::endl;

    } catch (const std::exception& e) {
        pcout << "Error generating mesh hierarchy: " << e.what() << std::endl;
    }
}

template <int dim>
std::vector<std::string> Convex2Convex<dim>::get_mesh_hierarchy_files() const
{
    std::vector<std::string> mesh_files;
    const std::string dir = "output/data_mesh/multilevel";
    
    // List all .msh files in the hierarchy directory
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".msh") {
            mesh_files.push_back(entry.path().string());
        }
    }
    
    // Sort in reverse order (coarsest to finest)
    std::sort(mesh_files.begin(), mesh_files.end(), std::greater<std::string>());
    return mesh_files;
}

template <int dim>
void Convex2Convex<dim>::load_mesh_at_level(const std::string& mesh_file)
{
    pcout << "Attempting to load mesh from: " << mesh_file << std::endl;
    
    // Check if file exists
    if (!std::filesystem::exists(mesh_file)) {
        throw std::runtime_error("Mesh file does not exist: " + mesh_file);
    }

    // Check if file is readable and non-empty
    std::ifstream input(mesh_file);
    if (!input.good()) {
        throw std::runtime_error("Cannot open mesh file: " + mesh_file);
    }
    
    input.seekg(0, std::ios::end);
    if (input.tellg() == 0) {
        throw std::runtime_error("Mesh file is empty: " + mesh_file);
    }
    input.seekg(0, std::ios::beg);

    try {
        // First load into a serial triangulation
        Triangulation<dim> serial_source;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(serial_source);
        
        grid_in.read_msh(input);
        
        // Verify the mesh was loaded properly
        if (serial_source.n_active_cells() == 0) {
            throw std::runtime_error("Loaded mesh contains no cells");
        }
        
        pcout << "Successfully loaded serial mesh with "
              << serial_source.n_active_cells() << " cells and "
              << serial_source.n_vertices() << " vertices" << std::endl;
        
        // Partition the serial mesh using z-order curve
        GridTools::partition_triangulation_zorder(n_mpi_processes, serial_source);
        
        // Convert to fullydistributed triangulation
        auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(
            serial_source, mpi_communicator,
            TriangulationDescription::Settings::default_setting);
        
        // Clear old DoFHandler first
        dof_handler_source.clear();
        // Then clear and recreate triangulation
        source_mesh.clear();
        source_mesh.create_triangulation(construction_data);
        
        // DoFHandler is automatically reinitialized since it's connected to source_mesh
        
        // Verify the distributed mesh
        const unsigned int n_global_active_cells = 
            Utilities::MPI::sum(source_mesh.n_locally_owned_active_cells(), mpi_communicator);
            
        if (n_global_active_cells == 0) {
            throw std::runtime_error("Distributed mesh contains no cells");
        }
        
        pcout << "Successfully created distributed mesh with "
              << n_global_active_cells << " total cells across "
              << n_mpi_processes << " processes" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load mesh from " + mesh_file + 
                               "\nError: " + e.what());
    }
}

template <int dim>
void Convex2Convex<dim>::setup_multilevel_finite_elements()
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
void Convex2Convex<dim>::run_multilevel_sot()
{
    pcout << "Starting multilevel SOT computation..." << std::endl;
    
    // Get mesh hierarchy files (sorted from coarsest to finest)
    std::vector<std::string> mesh_files = get_mesh_hierarchy_files();
    if (mesh_files.empty()) {
        pcout << "No mesh hierarchy found. Please run prepare_multilevel first." << std::endl;
        return;
    }
    
    // Store original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.regularization_param;
    
    // Create epsilon directory with multilevel subdirectory
    std::string eps_dir = "output/epsilon_" + std::to_string(original_regularization);
    std::string multilevel_dir = "multilevel";
    fs::create_directories(eps_dir+"/multilevel");

    // Check if we're using tetrahedral meshes
    bool use_simplex = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh);

    if (use_simplex) {
        fe_system = std::make_unique<FE_SimplexP<dim>>(1);
        mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    } else {
        fe_system = std::make_unique<FE_Q<dim>>(1);
        mapping = std::make_unique<MappingQ1<dim>>();
    }

    // Set up target points and RTree (only needs to be done once)
    setup_target_points();
    
    // Vector to store previous level's solution
    Vector<double> previous_weights;
    Vector<double> final_weights;
    
    // Process each level from coarsest to finest
    for (size_t level = 0; level < mesh_files.size(); ++level) {
        pcout << "\nProcessing level " << level << " (mesh: " << mesh_files[level] << ")" << std::endl;
        
        // Load the mesh for this level
        load_mesh_at_level(mesh_files[level]);
        setup_multilevel_finite_elements();  // Use specialized setup
        
        // Adjust solver parameters based on level
        solver_params.max_iterations = original_max_iterations;
        
        // Adjust tolerance based on level:
        // level 0 (coarsest): tolerance * 10^(num_levels-0)
        // level 1: tolerance * 10^(num_levels-1)
        // level 2 (finest): tolerance * 10^(num_levels-2)
        double num_levels = static_cast<double>(mesh_files.size());

        double tolerance_exponent = static_cast<double>(level) - num_levels + 1.0;
        pcout << "tolerance_exponent: " << tolerance_exponent << std::endl;
        solver_params.tolerance = original_tolerance * std::pow(2.0, tolerance_exponent);
        
        pcout << "\nLevel " << level << " solver parameters:" << std::endl;
        pcout << "  Level: " << level << " of " << num_levels << std::endl;
        pcout << "  Tolerance: " << solver_params.tolerance << std::endl;
        pcout << "  Max iterations: " << solver_params.max_iterations << std::endl;
        
        // Initialize weights - either from previous level or zero
        Vector<double> level_weights(target_points.size());
        if (level > 0) {
            level_weights = previous_weights;
            pcout << "Initialized weights from previous level solution" << std::endl;
        }
        
        // Set the current_weights pointer to point to our local weights vector
        current_weights = &level_weights;
        
        // Run SOT at current level
        try {
            Timer level_timer;
            level_timer.start();
            
            // Create a verbose solver control that displays iteration info
            class VerboseSolverControl : public SolverControl
            {
            public:
                VerboseSolverControl(unsigned int n, double tol, ConditionalOStream& pcout_)
                    : SolverControl(n, tol), pcout(pcout_) {}

                virtual State check(unsigned int step, double value) override
                {
                    pcout << "Iteration " << step
                          << " - Function value: " << value
                          << " - Relative residual: " << value / initial_value() << std::endl;
                    return SolverControl::check(step, value);
                }
            private:
                ConditionalOStream& pcout;
            };

            solver_control = std::make_unique<VerboseSolverControl>(
                solver_params.max_iterations,
                solver_params.tolerance,
                pcout
            );
            
            if (!solver_params.verbose_output) {
                solver_control->log_history(false);
                solver_control->log_result(false);
            }
            
            SolverBFGS<Vector<double>> solver(*solver_control);
            solver.solve(
                [&](const Vector<double> &w, Vector<double> &grad) {
                    return evaluate_sot_functional(w, grad);
                },
                level_weights
            );
            
            level_timer.stop();
            
            // Save results for this level in the multilevel directory
            std::string level_dir = multilevel_dir + "/level_" + std::to_string(level);
            save_results(level_weights, level_dir + "/weights");
            
            // Store current solution for next level and as final result
            previous_weights = level_weights;
            if (level == mesh_files.size() - 1) {
                final_weights = level_weights;
            }

            reset_distance_threshold_cache();
            
            pcout << "\nLevel " << level << " summary:" << std::endl;
            pcout << "  Status: Completed successfully" << std::endl;
            pcout << "  Time taken: " << level_timer.wall_time() << " seconds" << std::endl;
            pcout << "  Final number of iterations: " << solver_control->last_step() << std::endl;
            pcout << "  Final function value: " << solver_control->last_value() << std::endl;
            pcout << "  Results saved in: " << level_dir << std::endl;

            
        } catch (const std::exception& e) {
            pcout << "Error at level " << level << ": " << e.what() << std::endl;
            if (level == 0) return;  // If coarsest level fails, abort
            // Otherwise continue to next level with zero initialization
        }
    }
    
    // Save final results in the multilevel directory
    save_results(final_weights, multilevel_dir + "/weights");  // Save final results in multilevel directory
    
    pcout << "\nMultilevel SOT computation summary:" << std::endl;
    pcout << "  Total levels processed: " << mesh_files.size() << std::endl;
    pcout << "  Original regularization parameter (λ): " << original_regularization << std::endl;
    pcout << "  Original tolerance: " << original_tolerance << std::endl;
    pcout << "  All results (including final) saved in: " << eps_dir << "/" << multilevel_dir << std::endl;
    
    pcout << "\nMultilevel SOT computation completed" << std::endl;
}

template <int dim>
void Convex2Convex<dim>::save_results(const Vector<double> &weights,
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

        // Save convergence info if solver_control exists
        if (solver_control) {
            std::ofstream conv_info(eps_dir + "/convergence_info.txt");
            conv_info << "Regularization parameter (λ): " << solver_params.regularization_param << "\n";
            conv_info << "Number of iterations: " << solver_control->last_step() << "\n";
            conv_info << "Final function value: " << solver_control->last_value() << "\n";
            conv_info << "Convergence achieved: " << (solver_control->last_check() == SolverControl::success) << "\n";
        }
    }

    // Make sure all processes wait for rank 0 to finish writing
    MPI_Barrier(mpi_communicator);
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
            pcout << "Error: Mismatch between weights size (" << weights_vec.size()
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
            pcout << "Failed to initialize " << power_diagram_params.implementation
                      << " implementation: " << e.what() << std::endl;
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
template <int d>
typename std::enable_if<d == 3>::type Convex2Convex<dim>::run_exact_sot()
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

    pcout << "Exact SOT computation completed successfully" << std::endl;
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
void Convex2Convex<dim>::compute_transport_map()
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
            pcout << "Error: Mismatch between weights size (" << weights_vec.size()
                      << ") and target points size (" << target_points.size() << ")" << std::endl;
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
void Convex2Convex<dim>::reset_distance_threshold_cache() const
{
    is_caching_active = false;
    // current_distance_threshold = 0.0;
    // effective_distance_threshold = 0.0;
    cell_caches.clear();
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
    else if (selected_task == "prepare_multilevel")
    {
        prepare_multilevel();
    }
    else if (selected_task == "multilevel_sot")
    {
        run_multilevel_sot();
    }
    else if (selected_task == "prepare_target_multilevel")
    {
        load_meshes();
        prepare_target_multilevel();
    }
    else if (selected_task == "target_multilevel_sot")
    {
        run_target_multilevel_sot();
    }
    else if (selected_task == "combined_multilevel_sot")
    {
        run_combined_multilevel_sot();
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

template <int dim>
void Convex2Convex<dim>::prepare_target_multilevel()
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
            // Target mesh not loaded, try to load it
            load_mesh_target();
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

    // Create PointCloudHierarchyManager instance
    PointCloudHierarchy::PointCloudHierarchyManager hierarchy_manager(
        target_multilevel_params.min_points,
        target_multilevel_params.max_points
    );

    try {
        // Create output directory if it doesn't exist
        fs::create_directories(target_multilevel_params.hierarchy_output_dir);

        // Generate hierarchy
        std::vector<double> target_weights(target_points.size());
        for (size_t i = 0; i < target_points.size(); ++i) {
            target_weights[i] = target_density[i];
        }

        pcout << "Generating hierarchy with " << target_points.size() << " points..." << std::endl;
        
        int num_levels = hierarchy_manager.generateHierarchy<dim>(
            target_points,
            target_weights,
            target_multilevel_params.hierarchy_output_dir
        );
        
        pcout << "Successfully generated " << num_levels << " levels of point cloud hierarchy." << std::endl;
        
        // Load the hierarchy data for use in computations
        load_hierarchy_data(target_multilevel_params.hierarchy_output_dir);
        pcout << "Loaded hierarchy data for direct parent-child weight assignment." << std::endl;
        
    } catch (const std::exception& e) {
        pcout << "Error generating point cloud hierarchy: " << e.what() << std::endl;
    }
}

template <int dim>
std::vector<std::pair<std::string, std::string>> Convex2Convex<dim>::get_target_hierarchy_files() const
{
    std::vector<std::pair<std::string, std::string>> files;
    const std::string dir = target_multilevel_params.hierarchy_output_dir;
    
    // Check if directory exists
    if (!fs::exists(dir)) {
        throw std::runtime_error("Target point cloud hierarchy directory does not exist: " + dir);
    }
    
    // Collect all level point files
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().filename().string().find("level_") == 0 && 
            entry.path().filename().string().find("_points.txt") != std::string::npos) {
            std::string points_file = entry.path().string();
            points_file = points_file.substr(0, points_file.length() - 4);
            
            std::string level_num = points_file.substr(
                points_file.find("level_") + 6, 
                points_file.find("_points") - points_file.find("level_") - 6
            );
            std::string weights_file = dir + "/level_" + level_num + "_weights";
            
            if (fs::exists(weights_file + ".txt")) {
                files.push_back({points_file, weights_file});
            }
        }
    }
    
    // Sort in reverse order (coarsest to finest)
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        int level_a = std::stoi(a.first.substr(a.first.find("level_") + 6, 1));
        int level_b = std::stoi(b.first.substr(b.first.find("level_") + 6, 1));
        return level_a > level_b;
    });
    
    return files;
}

template <int dim>
void Convex2Convex<dim>::load_target_points_at_level(
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
            pcout << "Error: Cannot read points file: " << points_file << std::endl;
            load_success = false;
        }
        
        // Read weights
        if (load_success && !Utils::read_vector(local_weights, weights_file, io_coding)) {
            pcout << "Error: Cannot read weights file: " << weights_file << std::endl;
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
    
    pcout << "Successfully loaded " << n_points << " target points at this level" << std::endl;
}

template <int dim>
void Convex2Convex<dim>::setup_custom_target_points(
    const std::vector<Point<dim>>& custom_target_points,
    const std::vector<double>& custom_target_weights)
{
    // Replace existing target points with custom points
    target_points = custom_target_points;
    
    // Initialize target density with custom weights
    target_density.reinit(custom_target_points.size());
    
    // Copy weights or create uniform weights if empty
    if (!custom_target_weights.empty()) {
        AssertDimension(custom_target_weights.size(), custom_target_points.size());
        for (size_t i = 0; i < custom_target_points.size(); ++i) {
            target_density[i] = custom_target_weights[i];
        }
    } else {
        const double weight = 1.0 / custom_target_points.size();
        for (size_t i = 0; i < custom_target_points.size(); ++i) {
            target_density[i] = weight;
        }
    }
    
    // Reset the RTree for the new points
    setup_target_points();
}

template <int dim>
void Convex2Convex<dim>::assign_weights_by_hierarchy(
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

    if (target_multilevel_params.use_softmax_weight_transfer) {
        pcout << "Applying softmax-based weight assignment from level " << coarse_level
              << " to level " << fine_level << std::endl;
        pcout << "Source points: " << prev_weights.size() 
              << ", Target points: " << target_points.size() << std::endl;

        // Store current level for use in local_assemble_softmax_refinement
        current_level = fine_level;

        // Apply softmax refinement
        weights = softmax_refinement(prev_weights);
        
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
            for (size_t j = 0; j < target_points.size(); ++j) {
                // Get parent indices for this target point
                const auto& parents = child_indices_[fine_level][j];
                if (!parents.empty()) {
                    // Assign the weight of the first parent (in case of multiple parents)
                    weights[j] = prev_weights[parents[0]];
                }
            }
        }

        // Broadcast the weights to all processes
        Utilities::MPI::broadcast(mpi_communicator, weights, 0);
    }
}

template <int dim>
void Convex2Convex<dim>::run_target_multilevel_sot()
{
    Timer global_timer;
    global_timer.start();
    
    pcout << "Starting target point cloud multilevel SOT computation..." << std::endl;
    
    // Load source mesh
    load_mesh_source();
    setup_source_finite_elements();
    
    // Get target point cloud hierarchy files (sorted from coarsest to finest)
    unsigned int num_levels = 0;
    std::vector<std::pair<std::string, std::string>> hierarchy_files;
    if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        try {
            hierarchy_files = get_target_hierarchy_files();
        } catch (const std::exception& e) {
            pcout << "Error getting target hierarchy files: " << e.what() << std::endl;
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
        load_hierarchy_data(target_multilevel_params.hierarchy_output_dir, -1);
    }
    
    if (!has_hierarchy_data_) {
        pcout << "Failed to initialize hierarchy data. Cannot proceed with multilevel computation." << std::endl;
        return;
    }
    
    // Store original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.regularization_param;
    
    // Vector to store current weights solution
    Vector<double> level_weights;
    
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
        
            // Create output directory for this level
            level_output_dir = target_multilevel_params.output_prefix + "/level_" + level_num;
            fs::create_directories(level_output_dir);
        }
        level_number = Utilities::MPI::broadcast(mpi_communicator, level_number, 0);
        
        pcout << "\n----------------------------------------" << std::endl;
        pcout << "Processing target point cloud level " << level_number << std::endl;
        pcout << "----------------------------------------" << std::endl;

        // Load hierarchy data for this level only
        load_hierarchy_data(target_multilevel_params.hierarchy_output_dir, level_number);
        
        // Load target points for this level
        if (level > 0) {
            target_points_coarse = target_points;
            target_density_coarse = target_density;
        }
        load_target_points_at_level(points_file, weights_file);
        pcout << "Target points loaded for level " << level_number << std::endl;
        pcout << "Target points size: " << target_points.size() << std::endl;
        
        // If we have weights from previous level, use them as initial guess
        if (level > 0) {
            int prev_level_number = 0;
            if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
                // Get previous level number
                std::string prev_level_num = hierarchy_files[level-1].first.substr(
                    hierarchy_files[level-1].first.find("level_") + 6, 
                    hierarchy_files[level-1].first.find("_points") - hierarchy_files[level-1].first.find("level_") - 6
                );
                prev_level_number = std::stoi(prev_level_num);
            }
            prev_level_number = Utilities::MPI::broadcast(mpi_communicator, prev_level_number, 0);
            
            pcout << "Transferring weights from level " << prev_level_number << " to level " << level_number << std::endl;

            // Store the previous level weights for weight transfer
            Vector<double> prev_level_weights = level_weights;

            
            // Initialize weights for current level
            level_weights.reinit(target_points.size());
            
            // Use direct parent-child relationship for weight transfer
            assign_weights_by_hierarchy(level_weights, prev_level_number, level_number, prev_level_weights);
            
            pcout << "Weights transferred using parent-child relationships" << std::endl;
        }
        else {
            level_weights.reinit(target_points.size());
        }

        // Initialize RTree with target points and their indices
        std::vector<IndexedPoint> indexed_points;
        indexed_points.reserve(target_points.size());
        for (std::size_t i = 0; i < target_points.size(); ++i) {
            indexed_points.emplace_back(target_points[i], i);
        }
        target_points_rtree = RTree(indexed_points.begin(), indexed_points.end());

        pcout << "RTree initialized for target points" << std::endl;
        pcout << n_levels(target_points_rtree) << std::endl;
        
        // Set the current_weights pointer to point to our local weights vector
        current_weights = &level_weights;
        pcout << "Size of level_weights: " << level_weights.size() << std::endl;
        pcout << "Size of target_points: " << target_points.size() << std::endl;
        // I want to print the first 4 component of the weight vector
        pcout << "First 4 components of level_weights: " << level_weights[0] << " " << level_weights[1] << " " << level_weights[2] << " " << level_weights[3] << std::endl;
        

        // Create solver control and store it in the class member
        solver_control = std::make_unique<VerboseSolverControl>(
            solver_params.max_iterations,
            solver_params.tolerance,
            pcout
        );
        
        // Timer for this level
        Timer timer;
        timer.start();
        
        // Run optimization for this level
        pcout << "Running optimization for level " << level_number << std::endl;
        try {
            SolverBFGS<Vector<double>> solver(*solver_control);
            
            solver.solve(
                [this](const Vector<double>& weights, Vector<double>& gradient) {
                    return this->evaluate_sot_functional(weights, gradient);
                },
                level_weights
            );
            
            pcout << "Level " << level_number << " optimization completed:" << std::endl
                  << "  Number of iterations: " << solver_control->last_step() << std::endl
                  << "  Final function value: " << solver_control->last_value() << std::endl;
            
        } catch (SolverControl::NoConvergence &exc) {
            pcout << "Warning: Optimization did not converge for level " << level_number << std::endl;
        }
        
        timer.stop();
        pcout << "Level " << level_number << " solver time: " << timer.wall_time() << " seconds" << std::endl;
        reset_distance_threshold_cache();
        
        // Save results for this level
        save_results(level_weights, level_output_dir + "/weights");
        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            std::ofstream conv_info(level_output_dir + "/convergence_info.txt");
            conv_info << "Regularization parameter (λ): " << solver_params.regularization_param << "\n";
            conv_info << "Number of iterations: " << solver_control->last_step() << "\n";
            conv_info << "Final function value: " << solver_control->last_value() << "\n";
            conv_info << "Convergence achieved: " << (solver_control->last_check() == SolverControl::success) << "\n";
            conv_info << "Level: " << level_number << "\n";
            pcout << "Level " << level_number << " results saved to " << level_output_dir << "/weights" << std::endl;
        }
    }
    
    // Restore original parameters
    solver_params.tolerance = original_tolerance;
    solver_params.max_iterations = original_max_iterations;
    solver_params.regularization_param = original_regularization;
    
    // Final weights are in level_weights
    pcout << "\nTarget point cloud multilevel SOT computation completed!" << std::endl;
    pcout << "Final results correspond to level 0 (finest level)" << std::endl;
    
    global_timer.stop();
    pcout << "\n----------------------------------------" << std::endl;
    pcout << "Total multilevel target computation time: " << global_timer.wall_time() << " seconds" << std::endl;
    pcout << "----------------------------------------" << std::endl;
}


template <int dim>
void Convex2Convex<dim>::load_hierarchy_data(const std::string& hierarchy_dir, int specific_level) {
    // Only rank 0 checks directory and counts levels
    int num_levels = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        while (true) {
            std::string points_file = hierarchy_dir + "/level_" + std::to_string(num_levels) + "_points.txt";
            if (!fs::exists(points_file)) {
                break;
            }
            num_levels++;
        }
    }
    
    // Broadcast number of levels
    num_levels = Utilities::MPI::broadcast(mpi_communicator, num_levels, 0);
    
    if (num_levels == 0) {
        pcout << "No hierarchy data found in " << hierarchy_dir << std::endl;
        return;
    }

    // If loading all levels
    if (specific_level == -1) {
        child_indices_.resize(num_levels - 1);
        has_hierarchy_data_ = true;
        return;
    }

    // Load data only for the specific level
    if (specific_level >= num_levels - 1) {
        return; // No parent-child relationships for the last level
    }

    // Ensure child_indices_ has enough space
    if (static_cast<int>(child_indices_.size()) < num_levels - 1) {
        child_indices_.resize(num_levels - 1);
    }

    bool level_load_success = true;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::string parents_file = hierarchy_dir + "/level_" + std::to_string(specific_level+1) + "_children.txt";
        std::ifstream parents_in(parents_file);
        
        if (!parents_in) {
            level_load_success = false;
        } else {
            std::vector<std::vector<size_t>> level_indices;
            std::string line;
            
            while (std::getline(parents_in, line)) {
                std::istringstream iss(line);
                int num_parents;
                iss >> num_parents;
                
                std::vector<size_t> parents;
                parents.reserve(num_parents);
                
                size_t parent_idx;
                while (num_parents-- > 0 && iss >> parent_idx) {
                    parents.push_back(parent_idx);
                }
                
                level_indices.push_back(parents);
            }
            
            // Store the parent-child relationships for this level
            if (!level_indices.empty()) {
                child_indices_[specific_level] = level_indices;
            }
        }
    }

    // Broadcast success status
    level_load_success = Utilities::MPI::broadcast(mpi_communicator, level_load_success, 0);
    
    if (!level_load_success) {
        pcout << "Error loading hierarchy data at level " << specific_level << std::endl;
        return;
    }

    // Broadcast level data size
    unsigned int n_parents = 0;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        n_parents = child_indices_[specific_level].size();
    }
    n_parents = Utilities::MPI::broadcast(mpi_communicator, n_parents, 0);
    pcout << "Number of parents: " << n_parents << std::endl;
    
    // Broadcast each parent's children
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
        child_indices_[specific_level].clear();
        child_indices_[specific_level].reserve(n_parents);
    }

    for (unsigned int i = 0; i < n_parents; ++i) {
        unsigned int n_children = 0;
        std::vector<size_t> children;
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            children = child_indices_[specific_level][i];
            n_children = children.size();
        }
        
        // Broadcast number of children
        n_children = Utilities::MPI::broadcast(mpi_communicator, n_children, 0);
        
        // Resize and broadcast children indices
        if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
            children.resize(n_children);
        }

        Utilities::MPI::broadcast(mpi_communicator, children, 0);
        if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0) {
            child_indices_[specific_level].push_back(children);
        }
    }

    has_hierarchy_data_ = true;
}

template <int dim>
void Convex2Convex<dim>::run_combined_multilevel_sot()
{
    Timer global_timer;
    global_timer.start();
    
    pcout << "Starting combined source-target multilevel SOT computation..." << std::endl;
    
    // Get source mesh hierarchy files (sorted from coarsest to finest)
    std::vector<std::string> source_mesh_files = get_mesh_hierarchy_files();
    if (source_mesh_files.empty()) {
        pcout << "No source mesh hierarchy found. Please run prepare_multilevel first." << std::endl;
        return;
    }

    // Store original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    const double original_regularization = solver_params.regularization_param;
    
    // Create output directory
    std::string eps_dir = "output/epsilon_" + std::to_string(original_regularization);
    std::string combined_dir = "combined_multilevel";
    fs::create_directories(eps_dir + "/" + combined_dir);

    // Vector to store weights between source mesh levels
    Vector<double> previous_source_weights;
    Vector<double> final_weights;
    
    // Process each source mesh level from coarsest to finest
    for (size_t source_level = 0; source_level < source_mesh_files.size(); ++source_level) {
        pcout << "\n============================================" << std::endl;
        pcout << "Processing source mesh level " << source_level 
              << " (mesh: " << source_mesh_files[source_level] << ")" << std::endl;
        pcout << "============================================" << std::endl;

        // Create directory for this source level
        std::string source_level_dir = eps_dir + "/" + combined_dir + "/source_level_" + std::to_string(source_level);
        fs::create_directories(source_level_dir);

        // Run target multilevel optimization for this source mesh level
        Vector<double> source_level_weights;
        if (source_level == 0) {
            run_target_multilevel_for_source_level(source_mesh_files[source_level], source_level_weights);
            previous_source_weights = source_level_weights;
        }
        else {
            // Load the mesh for this level
            load_mesh_at_level(source_mesh_files[source_level]);
            setup_multilevel_finite_elements();  // Use specialized setup
            
        // Adjust solver parameters based on level
        solver_params.max_iterations = original_max_iterations;
        
        // Adjust tolerance based on level:
        // level 0 (coarsest): tolerance * 10^(num_levels-0)
        // level 1: tolerance * 10^(num_levels-1)
        // level 2 (finest): tolerance * 10^(num_levels-2)
        double num_levels = static_cast<double>(source_mesh_files.size());

            double tolerance_exponent = static_cast<double>(source_level) - num_levels + 1.0;
            pcout << "tolerance_exponent: " << tolerance_exponent << std::endl;
            solver_params.tolerance = original_tolerance * std::pow(2.0, tolerance_exponent);
            
            pcout << "\nSource level " << source_level << " solver parameters:" << std::endl;
            pcout << "  Level: " << source_level << " of " << num_levels << std::endl;
            pcout << "  Tolerance: " << solver_params.tolerance << std::endl;
            pcout << "  Max iterations: " << solver_params.max_iterations << std::endl;
            
        // Initialize weights - either from previous level or zero
        Vector<double> level_weights(target_points.size());
        if (source_level > 0) {
            level_weights = previous_source_weights;
            pcout << "Initialized weights from previous level solution" << std::endl;
        }
        
        // Set the current_weights pointer to point to our local weights vector
        current_weights = &level_weights;
        
        // Run SOT at current level
        try {
            Timer level_timer;
            level_timer.start();
            
            // Create a verbose solver control that displays iteration info
            class VerboseSolverControl : public SolverControl
            {
            public:
                VerboseSolverControl(unsigned int n, double tol, ConditionalOStream& pcout_)
                    : SolverControl(n, tol), pcout(pcout_) {}

                virtual State check(unsigned int step, double value) override
                {
                    pcout << "Iteration " << step
                          << " - Function value: " << value
                          << " - Relative residual: " << value / initial_value() << std::endl;
                    return SolverControl::check(step, value);
                }
            private:
                ConditionalOStream& pcout;
            };

            solver_control = std::make_unique<VerboseSolverControl>(
                solver_params.max_iterations,
                solver_params.tolerance,
                pcout
            );
            
            if (!solver_params.verbose_output) {
                solver_control->log_history(false);
                solver_control->log_result(false);
            }
            
            SolverBFGS<Vector<double>> solver(*solver_control);
            solver.solve(
                [&](const Vector<double> &w, Vector<double> &grad) {
                    return evaluate_sot_functional(w, grad);
                },
                level_weights
            );
            
            level_timer.stop();
            
            // Save results for this level in the multilevel directory
            save_results(level_weights, source_level_dir + "/weights");
            
            // Store current solution for next level and as final result
            previous_source_weights = level_weights;
            if (source_level == source_mesh_files.size() - 1) {
                final_weights = level_weights;
            }

            reset_distance_threshold_cache();
            
            pcout << "\nSource level " << source_level << " summary:" << std::endl;
            pcout << "  Status: Completed successfully" << std::endl;
            pcout << "  Time taken: " << level_timer.wall_time() << " seconds" << std::endl;
            pcout << "  Final number of iterations: " << solver_control->last_step() << std::endl;
            pcout << "  Final function value: " << solver_control->last_value() << std::endl;
            pcout << "  Results saved in: " << source_level_dir << std::endl;

            
        } catch (const std::exception& e) {
            pcout << "Error at source level " << source_level << ": " << e.what() << std::endl;
            if (source_level == 0) return;  // If coarsest level fails, abort
            // Otherwise continue to next level with zero initialization
        }
        }


        // Save results for this source level
        save_results(source_level_weights, combined_dir + "/source_level_" + std::to_string(source_level) + "/weights");

        // Store final weights if this is the finest level
        if (source_level == source_mesh_files.size() - 1) {
            final_weights = source_level_weights;
        }

        pcout << "Source level " << source_level << " completed" << std::endl;
    }

    // Save final results
    save_results(final_weights, combined_dir + "/weights");

    // Restore original parameters
    solver_params.max_iterations = original_max_iterations;
    solver_params.tolerance = original_tolerance;
    solver_params.regularization_param = original_regularization;

    global_timer.stop();
    pcout << "\n============================================" << std::endl;
    pcout << "Combined multilevel computation completed!" << std::endl;
    pcout << "Total computation time: " << global_timer.wall_time() << " seconds" << std::endl;
    pcout << "Final results saved in: " << eps_dir << "/" << combined_dir << std::endl;
    pcout << "============================================" << std::endl;
}

template <int dim>
void Convex2Convex<dim>::run_target_multilevel_for_source_level(
    const std::string& source_mesh_file, Vector<double>& weights)
{
    pcout << "Running target multilevel optimization for source mesh: " << source_mesh_file << std::endl;

    // Load the source mesh for this level
    load_mesh_at_level(source_mesh_file);
    setup_source_finite_elements();

    // Get target point cloud hierarchy files
    unsigned int num_levels = 0;
    std::vector<std::pair<std::string, std::string>> hierarchy_files;
    if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        try {
            hierarchy_files = get_target_hierarchy_files();
            if (hierarchy_files.empty()) {
                pcout << "No target point cloud hierarchy found. Please run prepare_target_multilevel first." << std::endl;
                return;
            }
            num_levels = hierarchy_files.size();
        } catch (const std::exception& e) {
            pcout << "Error getting target hierarchy files: " << e.what() << std::endl;
            return;
        }
    }
    num_levels = Utilities::MPI::broadcast(mpi_communicator, num_levels, 0);

    // Initialize hierarchy data if needed
    if (!has_hierarchy_data_) {
        pcout << "Initializing hierarchy data structure..." << std::endl;
        load_hierarchy_data(target_multilevel_params.hierarchy_output_dir, -1);
    }

    if (!has_hierarchy_data_) {
        pcout << "Failed to initialize hierarchy data. Cannot proceed." << std::endl;
        return;
    }

    // Store original solver parameters
    const unsigned int original_max_iterations = solver_params.max_iterations;
    const double original_tolerance = solver_params.tolerance;
    
    // Vector to store current weights solution
    Vector<double> level_weights;
    
    // Process each target level from coarsest to finest
    for (size_t level = 0; level < num_levels; ++level) {
        int level_number = 0;
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
        }
        level_number = Utilities::MPI::broadcast(mpi_communicator, level_number, 0);

        pcout << "\n----------------------------------------" << std::endl;
        pcout << "Processing target point cloud level " << level_number << std::endl;
        pcout << "----------------------------------------" << std::endl;

        // Load hierarchy data for this level
        load_hierarchy_data(target_multilevel_params.hierarchy_output_dir, level_number);
        
        // Load target points for this level
        if (level > 0) {
            target_points_coarse = target_points;
            target_density_coarse = target_density;
        }
        load_target_points_at_level(points_file, weights_file);
        

        // If we have weights from previous level, use them as initial guess
        if (level > 0) {
            Vector<double> prev_level_weights = level_weights;
            level_weights.reinit(target_points.size());
            assign_weights_by_hierarchy(level_weights, level_number+1, level_number, prev_level_weights);
        } else {
            level_weights.reinit(target_points.size());
        }

        // Initialize RTree with target points and their indices
        std::vector<IndexedPoint> indexed_points;
        indexed_points.reserve(target_points.size());
        for (std::size_t i = 0; i < target_points.size(); ++i) {
            indexed_points.emplace_back(target_points[i], i);
        }
        target_points_rtree = RTree(indexed_points.begin(), indexed_points.end());

        pcout << "RTree initialized for target points" << std::endl;
        pcout << n_levels(target_points_rtree) << std::endl;

        current_weights = &level_weights;

        // Set up solver control
        solver_control = std::make_unique<VerboseSolverControl>(
            solver_params.max_iterations,
            solver_params.tolerance,
            pcout
        );

        // Run optimization for this level
        Timer timer;
        timer.start();
        
        try {
            SolverBFGS<Vector<double>> solver(*solver_control);
            current_weights = &level_weights;
            
            solver.solve(
                [this](const Vector<double>& w, Vector<double>& grad) {
                    return this->evaluate_sot_functional(w, grad);
                },
                level_weights
            );
            
            timer.stop();
            pcout << "Level " << level_number << " completed in " << timer.wall_time() << " seconds" << std::endl;
            reset_distance_threshold_cache();
            
        } catch (SolverControl::NoConvergence &exc) {
            pcout << "Warning: Optimization did not converge for level " << level_number << std::endl;
            if (level == 0) return;  // If coarsest level fails, abort
        }
    }

    // Return the final weights from the finest target level
    weights = level_weights;

    // Restore original parameters
    solver_params.max_iterations = original_max_iterations;
    solver_params.tolerance = original_tolerance;
}


template class Convex2Convex<2>;
template class Convex2Convex<3>;