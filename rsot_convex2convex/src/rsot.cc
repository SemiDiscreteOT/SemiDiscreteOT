#include "rsot.h"
#include "PowerDiagram.h"
#include "utils.h"
#include "ExactSot.h"
#include "OptimalTransportPlan.h"
#include <deal.II/base/timer.h>
#include <filesystem>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/vector_operations_internal.h>
#include <deal.II/grid/grid_tools.h>
namespace fs = std::filesystem;

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

        add_parameter("solver_type",
                     solver_params.solver_type,
                     "Type of optimization solver (BFGS)");

        add_parameter("quadrature_order",
                     solver_params.quadrature_order,
                     "Order of quadrature formula for numerical integration");

        add_parameter("number_of_threads",
                     solver_params.number_of_threads,
                     "Number of threads to use for parallel SOT");
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

    using VectorizedDouble = VectorizedArray<double>;
    constexpr unsigned int vectorization_width = VectorizedArray<double>::size();

    scratch_data.fe_values.reinit(cell);
    const std::vector<Point<dim>> &q_points = scratch_data.fe_values.get_quadrature_points();
    scratch_data.fe_values.get_function_values(source_density, scratch_data.density_values);

    copy_data.functional_value = 0.0;
    copy_data.gradient_values = 0;

    const unsigned int n_q_points = q_points.size();

    // // Get cell bounding box and extend it by the current distance threshold
    // Point<dim> min_point = cell->vertex(0);
    // Point<dim> max_point = min_point;

    // // Find bounding box of the cell
    // const unsigned int vertices_per_cell = (source_params.use_tetrahedral_mesh || target_params.use_tetrahedral_mesh) ? 4 : GeometryInfo<dim>::vertices_per_cell;
    // for (unsigned int v = 1; v < vertices_per_cell; ++v) {
    //     const Point<dim>& vertex = cell->vertex(v);
    //     for (unsigned int d = 0; d < dim; ++d) {
    //         min_point[d] = std::min(min_point[d], vertex[d]);
    //         max_point[d] = std::max(max_point[d], vertex[d]);
    //     }
    // }

    // // Extend bounding box by the distance threshold
    // for (unsigned int d = 0; d < dim; ++d) {
    //     min_point[d] -= current_distance_threshold;
    //     max_point[d] += current_distance_threshold;
    // }

    // // Find target points within the extended bounding box
    // std::vector<std::size_t> cell_target_indices;
    // BoundingBox<dim> extended_box(std::make_pair(min_point, max_point));
    // cell_target_indices = find_target_points_in_box(extended_box);

    // In alternativa potrei lavorare solo sul centro
    // Get cell center
    Point<dim> cell_center = cell->center();
    // Find target points near the cell center
    std::vector<std::size_t> cell_target_indices = find_nearest_target_points(cell_center);

    // Process quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q) {
        const Point<dim> &x = q_points[q];
        double total_sum_exp = 0.0;
        std::vector<double> exp_terms(cell_target_indices.size(), 0.0);

        // Process points in SIMD chunks
        for (size_t base_idx = 0; base_idx < cell_target_indices.size(); base_idx += vectorization_width) {
            VectorizedDouble dist2 = VectorizedDouble();
            VectorizedDouble weight = VectorizedDouble();
            VectorizedDouble density = VectorizedDouble();

            // Load a chunk of points into SIMD vectors
            for (unsigned int v = 0; v < vectorization_width; ++v) {
                const size_t i = base_idx + v;
                if (i < cell_target_indices.size()) {
                    const size_t target_idx = cell_target_indices[i];

                    // Load weight and density directly
                    weight[v] = (*current_weights)[target_idx];
                    density[v] = target_density[target_idx];

                    // Compute squared distance
                    double local_dist2 = 0.0;
                    for (unsigned int d = 0; d < dim; ++d) {
                        double diff = target_points[target_idx][d] - x[d];
                        local_dist2 += diff * diff;
                    }
                    dist2[v] = local_dist2;
                } else {
                    // Pad with zeros for incomplete chunks
                    weight[v] = 0.0;
                    density[v] = 0.0;
                    dist2[v] = std::numeric_limits<double>::infinity();
                }
            }

            // Vectorized computation for the entire chunk
            VectorizedDouble exp_term = density *
                std::exp((weight - 0.5 * dist2) / current_lambda);

            // Store results for this chunk
            for (unsigned int v = 0; v < vectorization_width; ++v) {
                const size_t i = base_idx + v;
                if (i < cell_target_indices.size()) {
                    exp_terms[i] = exp_term[v];
                    total_sum_exp += exp_term[v];
                }
            }
        }

        // Accumulate functional value
        copy_data.functional_value += scratch_data.density_values[q] *
            current_lambda * std::log(total_sum_exp) * scratch_data.fe_values.JxW(q);

        // Accumulate gradient values
        for (size_t i = 0; i < cell_target_indices.size(); ++i) {
            const size_t target_idx = cell_target_indices[i];
            copy_data.gradient_values[target_idx] +=
                scratch_data.density_values[q] * (exp_terms[i] / total_sum_exp) * scratch_data.fe_values.JxW(q);
        }
    }
}

template <int dim>
double Convex2Convex<dim>::evaluate_sot_functional(const Vector<double> &weights, Vector<double> &gradient)
{
    // Store current weights and lambda for parallel access
    current_weights = &weights;
    current_lambda = solver_params.regularization_param;

    compute_distance_threshold();

    // Reset global values for this MPI process
    double local_process_functional = 0.0;
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
                   [&local_process_functional, &local_process_gradient, this]
                   (const CopyData &copy_data) {
                       std::lock_guard<std::mutex> lock(this->assembly_mutex);
                       local_process_functional += copy_data.functional_value;
                       local_process_gradient += copy_data.gradient_values;
                   },
                   scratch_data,
                   copy_data);

    // Sum up contributions across all MPI processes
    global_functional = Utilities::MPI::sum(local_process_functional, mpi_communicator);

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
    global_functional =
        Utilities::MPI::broadcast(mpi_communicator, global_functional, 0);
    gradient =
        Utilities::MPI::broadcast(mpi_communicator, gradient, 0);

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
void Convex2Convex<dim>::run_sot()
{
    // Set number of threads based on parameter
    unsigned int n_threads = solver_params.number_of_threads;
    const unsigned int n_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);

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

    SolverBFGS<Vector<double>> solver(*solver_control);

    try
    {
        pcout << "Using regularization parameter λ = "
                  << solver_params.regularization_param << std::endl;
        solver.solve(
            [&](const Vector<double> &w, Vector<double> &grad) {
                return evaluate_sot_functional(w, grad);
            },
            weights
        );

        timer.stop();
        pcout << "\nOptimization completed successfully!" << std::endl;
        pcout << "Total solver time: " << timer.wall_time() << " seconds" << std::endl;
        pcout << "Final number of iterations: " << solver_control->last_step() << std::endl;
        pcout << "Final function value: " << solver_control->last_value() << std::endl;

        save_results(weights, "weights");
        pcout << "Results saved to weights" << std::endl;
    }
    catch (SolverControl::NoConvergence &exc)
    {
        timer.stop();
        // Save results before reporting the error
        save_results(weights, "weights");
        pcout << "\nMaximum iterations reached without convergence." << std::endl;
        pcout << "Total solver time: " << timer.wall_time() << " seconds" << std::endl;
        pcout << "Final number of iterations: " << solver_control->last_step() << std::endl;
        pcout << "Final function value: " << solver_control->last_value() << std::endl;
        pcout << "Intermediate results saved to weights" << std::endl;

        // Re-throw the exception
        throw;
    }
    catch (std::exception &exc)
    {
        timer.stop();
        pcout << "Total solver time: " << timer.wall_time() << " seconds" << std::endl;
        pcout << "Error in SOT computation: " << exc.what() << std::endl;
    }
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
    std::vector<double> density_values(quadrature->size());

    for (const auto &cell : dof_handler_source.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(source_density, density_values);

        for (unsigned int q = 0; q < quadrature->size(); ++q) {
            local_l1_norm += std::abs(density_values[q]) * fe_values.JxW(q);
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
        Vector<double> current_weights(target_points.size());
        if (level > 0) {
            current_weights = previous_weights;
            pcout << "Initialized weights from previous level solution" << std::endl;
        }
        
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
                current_weights
            );
            
            level_timer.stop();
            
            // Save results for this level in the multilevel directory
            std::string level_dir = multilevel_dir + "/level_" + std::to_string(level);
            save_results(current_weights, level_dir + "/weights");
            
            // Store current solution for next level and as final result
            previous_weights = current_weights;
            if (level == mesh_files.size() - 1) {
                final_weights = current_weights;
            }
            
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
    std::vector<double> source_density_vec;
    {
        // Get support points
        std::map<types::global_dof_index, Point<dim>> support_points_source;
        DoFTools::map_dofs_to_support_points(*mapping, dof_handler_source, support_points_source);
        
        // Convert to vectors maintaining order
        source_points.reserve(support_points_source.size());
        source_density_vec.reserve(support_points_source.size());
        
        for (const auto& point_pair : support_points_source) {
            source_points.push_back(point_pair.second);
            source_density_vec.push_back(1.0); // Initialize with uniform density
        }
        
        // Normalize source density
        double total_mass = std::accumulate(source_density_vec.begin(), source_density_vec.end(), 0.0);
        for (auto& density : source_density_vec) {
            density /= total_mass;
        }
    }

    // Convert target density to standard vector
    std::vector<double> target_density_vec(target_density.begin(), target_density.end());

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

template class Convex2Convex<2>;
template class Convex2Convex<3>;