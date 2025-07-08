#include <SemiDiscreteOT/core/SemiDiscreteOT.h>
#include <SemiDiscreteOT/utils/utils.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <iomanip>

using namespace dealii;

// Parameter classes using ParameterAcceptor
class BarycenterParameters : public ParameterAcceptor
{
public:
    BarycenterParameters() : ParameterAcceptor("Barycenter")
    {
        add_parameter("max_iterations", max_iterations,
                     "Maximum number of barycenter iterations");
        add_parameter("convergence_tolerance", convergence_tolerance,
                     "Convergence tolerance for barycenter iterations");
        add_parameter("weight_1", weight_1,
                     "Weight for first source measure");
        add_parameter("random_initialization", random_initialization,
                     "Use random initialization for barycenter points");
        add_parameter("weight_2", weight_2,
                     "Weight for second source measure");
        add_parameter("n_barycenter_points", n_barycenter_points,
                     "Number of points in barycenter discretization");
        add_parameter("random_seed", random_seed,
                     "Random seed for barycenter initialization");
        add_parameter("output_frequency", output_frequency,
                     "Frequency of output (every N iterations)");
        add_parameter("initial_bounds_min", initial_bounds_min,
                     "Minimum bound for initial barycenter points");
        add_parameter("initial_bounds_max", initial_bounds_max,
                     "Maximum bound for initial barycenter points");
    }

    unsigned int max_iterations = 100;
    double convergence_tolerance = 1e-6;
    double weight_1 = 0.5;
    double weight_2 = 0.5;
    unsigned int n_barycenter_points = 100;
    unsigned int random_seed = 42;
    unsigned int output_frequency = 5;
    double initial_bounds_min = -0.5;
    double initial_bounds_max = 0.5;
    bool random_initialization = true;
};

class StepControllerParameters : public ParameterAcceptor
{
public:
    StepControllerParameters() : ParameterAcceptor("Step Controller")
    {
        add_parameter("initial_alpha", initial_alpha,
                     "Initial step size");
        add_parameter("min_alpha", min_alpha,
                     "Minimum step size");
        add_parameter("max_alpha", max_alpha,
                     "Maximum step size");
        add_parameter("decay_factor", decay_factor,
                     "Factor by which to decrease step size");
        add_parameter("growth_factor", growth_factor,
                     "Factor by which to increase step size");
    }

    double initial_alpha = 1.0;
    double min_alpha = 1e-4;
    double max_alpha = 1e4;
    double decay_factor = 0.8;
    double growth_factor = 1.1;
};

class OptimalTransportParameters : public ParameterAcceptor
{
public:
    OptimalTransportParameters() : ParameterAcceptor("Optimal Transport")
    {
        add_parameter("epsilon", epsilon,
                     "Regularization parameter");
        add_parameter("distance_threshold", distance_threshold,
                     "Distance threshold for computational efficiency");
        add_parameter("tau", tau,
                     "Numerical stability parameter");
        add_parameter("max_iterations", max_iterations,
                     "Maximum iterations for OT solver");
        add_parameter("tolerance", tolerance,
                     "Tolerance for OT solver");
        add_parameter("use_log_sum_exp_trick", use_log_sum_exp_trick,
                     "Use log-sum-exp trick for numerical stability");
        add_parameter("verbose_output", verbose_output,
                     "Enable verbose output for OT solver");
        add_parameter("distance_threshold_type", distance_threshold_type,
                     "Type of distance threshold");
        add_parameter("source_multilevel_enabled", source_multilevel_enabled,
                     "Enable multilevel for source");
        add_parameter("target_multilevel_enabled", target_multilevel_enabled,
                     "Enable multilevel for target");
        add_parameter("source_min_vertices", source_min_vertices,
                     "Minimum number of vertices for source multilevel");
        add_parameter("source_max_vertices", source_max_vertices,
                     "Maximum number of vertices for source multilevel");
    }

    double epsilon = 1e-2;
    double distance_threshold = 1.5;
    double tau = 1e-12;
    unsigned int max_iterations = 1000;
    double tolerance = 1e-3;
    bool use_log_sum_exp_trick = true;
    bool verbose_output = false;
    std::string distance_threshold_type = "pointwise";
    bool source_multilevel_enabled = false;
    bool target_multilevel_enabled = false;
    unsigned int source_min_vertices = 100;
    unsigned int source_max_vertices = 500;
};

class FileParameters : public ParameterAcceptor
{
public:
    FileParameters() : ParameterAcceptor("Files")
    {
        add_parameter("source1_filename", source1_filename,
                     "Filename for first source mesh");
        add_parameter("source2_filename", source2_filename,
                     "Filename for second source mesh");
        add_parameter("output_prefix", output_prefix,
                     "Prefix for output files");
        add_parameter("save_vtk", save_vtk,
                     "Save results in VTK format");
        add_parameter("save_txt", save_txt,
                     "Save results in text format");
    }

    std::string source1_filename = "source1.msh";
    std::string source2_filename = "source2.msh";
    std::string output_prefix = "barycenter";
    bool save_vtk = true;
    bool save_txt = true;
};

template <int dim, int spacedim>
struct BarycenterScratchData
{
    BarycenterScratchData(const FiniteElement<dim, spacedim> &fe,
                          const Mapping<dim, spacedim> &mapping,
                          const unsigned int quadrature_degree)
        : fe_values(mapping, fe, QGauss<dim>(quadrature_degree),
                    update_values | update_quadrature_points | update_JxW_values),
          density_values(fe_values.get_quadrature().size())
    {
    }

    BarycenterScratchData(const BarycenterScratchData<dim, spacedim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                    scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    scratch_data.fe_values.get_update_flags()),
          density_values(scratch_data.density_values.size())
    {
    }

    FEValues<dim, spacedim> fe_values;
    std::vector<double> density_values;
};

template <int spacedim>
struct BarycenterCopyData
{
    BarycenterCopyData(const unsigned int n_target_points, const unsigned int dim)
        : grad_support_points(n_target_points * dim)
    {
    }
    Vector<double> grad_support_points;
};

template <int dim, int spacedim>
void local_assemble_barycenter_gradient(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    BarycenterScratchData<dim, spacedim> &scratch,
    BarycenterCopyData<spacedim> &copy,
    const Vector<double> &source_density,
    const std::vector<Point<spacedim>> &target_points,
    const Vector<double> &target_weights,
    const Vector<double> &potentials,
    const double epsilon,
    const double distance_threshold)
{
    const unsigned int n_q_points = scratch.fe_values.n_quadrature_points;
    const unsigned int n_target_points = target_points.size();

    scratch.fe_values.reinit(cell);
    scratch.fe_values.get_function_values(source_density, scratch.density_values);

    copy.grad_support_points = 0.0;

    std::vector<unsigned int> nearby_targets;
    const auto &cell_center = cell->center();
    for (unsigned int j = 0; j < n_target_points; ++j)
    {
        if (cell_center.distance(target_points[j]) <= distance_threshold)
        {
            nearby_targets.push_back(j);
        }
    }

    if (nearby_targets.empty())
        return;

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
        const Point<spacedim> x = scratch.fe_values.quadrature_point(q);
        const double JxW = scratch.fe_values.JxW(q);
        const double rho_x = scratch.density_values[q];

        if (std::abs(rho_x) < 1e-12)
            continue;

        std::vector<double> weights(nearby_targets.size());
        double sum_weights = 0.0;
        double max_exponent = -std::numeric_limits<double>::infinity();

        for (unsigned int idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const unsigned int j = nearby_targets[idx];
            const double dist_sq = x.distance_square(target_points[j]);
            const double exponent = (potentials[j] - 0.5 * dist_sq) / epsilon;
            max_exponent = std::max(max_exponent, exponent);
        }

        for (unsigned int idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const unsigned int j = nearby_targets[idx];
            const double dist_sq = x.distance_square(target_points[j]);
            const double exponent = (potentials[j] - 0.5 * dist_sq) / epsilon;
            weights[idx] = target_weights(j) * std::exp(exponent - max_exponent);
            sum_weights += weights[idx];
        }

        if (sum_weights < 1e-12)
            continue;

        for (unsigned int idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const unsigned int j = nearby_targets[idx];
            const double weight = weights[idx] / sum_weights;
            const double scaled_weight = weight * rho_x * JxW;

            const Tensor<1, spacedim> diff = target_points[j] - x;
            for (unsigned int d = 0; d < spacedim; ++d)
            {
                copy.grad_support_points[j * spacedim + d] += diff[d] * scaled_weight;
            }
        }
    }
}

template <int spacedim>
void copy_local_to_global_barycenter(
    const BarycenterCopyData<spacedim> &copy,
    Vector<double> &global_grad_support_points)
{
    global_grad_support_points.add(1.0, copy.grad_support_points);
}

class AdaptiveStepController
{
public:
    AdaptiveStepController(const StepControllerParameters &params)
        : alpha(params.initial_alpha),
          min_alpha(params.min_alpha),
          max_alpha(params.max_alpha),
          decay_factor(params.decay_factor),
          growth_factor(params.growth_factor),
          previous_change(std::numeric_limits<double>::max())
    {
    }

    double get() const { return alpha; }

    void update(double current_change)
    {
        if (current_change < previous_change)
        {
            alpha = std::min(alpha * growth_factor, max_alpha);
        }
        else
        {
            alpha = std::max(alpha * decay_factor, min_alpha);
        }
        previous_change = current_change;
    }

private:
    double alpha;
    const double min_alpha, max_alpha;
    const double decay_factor, growth_factor;
    double previous_change;
};

void save_vtk_output(const std::vector<Point<3>> &points,
                     const Vector<double> &weights,
                     const std::string &filename)
{
    std::ofstream vtk_file(filename);
    if (!vtk_file.is_open()) return;

    const unsigned int n_points = points.size();

    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Barycenter Points\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";
    vtk_file << "POINTS " << n_points << " float\n";

    for (const auto &point : points) {
        vtk_file << point[0] << " " << point[1] << " " << point[2] << "\n";
    }

    vtk_file << "CELLS " << n_points << " " << 2 * n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1 " << i << "\n";
    }

    vtk_file << "CELL_TYPES " << n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1\n";  // VTK_VERTEX
    }

    vtk_file << "POINT_DATA " << n_points << "\n";
    vtk_file << "SCALARS weights float 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << weights[i] << "\n";
    }

    vtk_file.close();
}

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    pcout << "=== Wasserstein Barycenter Tutorial ===" << std::endl;

    // Create parameter objects
    BarycenterParameters barycenter_params;
    StepControllerParameters step_params;
    OptimalTransportParameters ot_params;
    FileParameters file_params;

    // Parse parameter file
    std::string parameter_filename = "parameters.prm";
    if (argc > 1) {
        parameter_filename = argv[1];
    }

    try {
        ParameterAcceptor::initialize(parameter_filename);
        pcout << "Successfully loaded parameters from: " << parameter_filename << std::endl;
    } catch (const std::exception &e) {
        pcout << "Warning: Could not load parameter file '" << parameter_filename << "'." << std::endl;
        pcout << "Creating default parameter file..." << std::endl;

        // Create default parameter file
        std::ofstream param_file(parameter_filename);
        if (param_file.is_open()) {
            ParameterAcceptor::prm.print_parameters(param_file, ParameterHandler::Text);
            param_file.close();
            pcout << "Default parameter file created: " << parameter_filename << std::endl;
            pcout << "Please edit the parameters and run again." << std::endl;
        }
        return 1;
    }

    // Validate parameters
    if (std::abs(barycenter_params.weight_1 + barycenter_params.weight_2 - 1.0) > 1e-12) {
        pcout << "Warning: Weights do not sum to 1.0. Normalizing..." << std::endl;
        double sum = barycenter_params.weight_1 + barycenter_params.weight_2;
        barycenter_params.weight_1 /= sum;
        barycenter_params.weight_2 /= sum;
    }

    const int dim = 3;
    const int spacedim = 3;

    // Configure OT solver
    auto configure_solver = [&ot_params](SotParameterManager &p, int solver_id) {
        p.solver_params.epsilon = ot_params.epsilon;
        p.solver_params.use_log_sum_exp_trick = ot_params.use_log_sum_exp_trick;
        p.solver_params.verbose_output = ot_params.verbose_output;
        p.solver_params.tau = ot_params.tau;
        p.solver_params.distance_threshold_type = ot_params.distance_threshold_type;
        p.solver_params.max_iterations = ot_params.max_iterations;
        p.solver_params.tolerance = ot_params.tolerance;
        p.multilevel_params.source_enabled = ot_params.source_multilevel_enabled;
        p.multilevel_params.target_enabled = ot_params.target_multilevel_enabled;
        p.multilevel_params.source_min_vertices = ot_params.source_min_vertices;
        p.multilevel_params.source_max_vertices = ot_params.source_max_vertices;
        p.multilevel_params.source_hierarchy_dir = "source_hierarchy_" + std::to_string(solver_id);
    };

    SemiDiscreteOT<dim, spacedim> sot_problem_1(mpi_comm);
    sot_problem_1.configure([&](SotParameterManager &p) { configure_solver(p, 1); });
    SemiDiscreteOT<dim, spacedim> sot_problem_2(mpi_comm);
    sot_problem_2.configure([&](SotParameterManager &p) { configure_solver(p, 2); });

    // Load source meshes
    Triangulation<dim, spacedim> source1_tria;
    {
        GridIn<dim, spacedim> grid_in;
        grid_in.attach_triangulation(source1_tria);
        std::ifstream input_file(file_params.source1_filename);
        if (!input_file.is_open()) {
            pcout << "Error: Could not open source1 file: " << file_params.source1_filename << std::endl;
            return 1;
        }
        grid_in.read_msh(input_file);

        const double volume = GridTools::volume(source1_tria);
        if (volume > 1e-12) {
            GridTools::scale(1.0 / std::cbrt(volume), source1_tria);
            pcout << "Rescaled source 1 to unit volume." << std::endl;
        }
    }
    FE_SimplexP<dim, spacedim> source1_fe(1);
    DoFHandler<dim, spacedim> source1_dof_handler(source1_tria);
    source1_dof_handler.distribute_dofs(source1_fe);
    Vector<double> source1_density(source1_dof_handler.n_dofs());
    source1_density = 1.0;
    sot_problem_1.setup_source_measure(source1_tria, source1_dof_handler, source1_density, "source_1");

    Triangulation<dim, spacedim> source2_tria;
    {
        GridIn<dim, spacedim> grid_in;
        grid_in.attach_triangulation(source2_tria);
        std::ifstream input_file(file_params.source2_filename);
        if (!input_file.is_open()) {
            pcout << "Error: Could not open source2 file: " << file_params.source2_filename << std::endl;
            return 1;
        }
        grid_in.read_msh(input_file);

        const double volume = GridTools::volume(source2_tria);
        if (volume > 1e-12) {
            GridTools::scale(1.0 / std::cbrt(volume), source2_tria);
            pcout << "Rescaled source 2 to unit volume." << std::endl;
        }
    }
    FE_SimplexP<dim, spacedim> source2_fe(1);
    DoFHandler<dim, spacedim> source2_dof_handler(source2_tria);
    source2_dof_handler.distribute_dofs(source2_fe);
    Vector<double> source2_density(source2_dof_handler.n_dofs());
    source2_density = 1.0;
    sot_problem_2.setup_source_measure(source2_tria, source2_dof_handler, source2_density, "source_2");

    sot_problem_1.prepare_multilevel_hierarchies();
    sot_problem_2.prepare_multilevel_hierarchies();

    // Initialize barycenter
    pcout << "Initializing barycenter..." << std::endl;
    std::vector<Point<spacedim>> barycenter_points;
    if (barycenter_params.random_initialization) {
        pcout << "Using random initialization within the bounding box of both geometries" << std::endl;

        // Compute bounding box of both geometries
        dealii::BoundingBox<spacedim> bbox1 = GridTools::compute_bounding_box(source1_tria);
        dealii::BoundingBox<spacedim> bbox2 = GridTools::compute_bounding_box(source2_tria);
    
        // Combine bounding boxes
        Point<spacedim> min_point, max_point;
        std::pair<Point<spacedim>, Point<spacedim>> bounds1 = bbox1.get_boundary_points();
        std::pair<Point<spacedim>, Point<spacedim>> bounds2 = bbox2.get_boundary_points();
    
        for (unsigned int d = 0; d < spacedim; ++d) {
            min_point[d] = std::min(bounds1.first[d], bounds2.first[d]);
            max_point[d] = std::max(bounds1.second[d], bounds2.second[d]);
        }
    
        // Initialize random number generator
        std::mt19937 rng(barycenter_params.random_seed);
        std::vector<std::uniform_real_distribution<double>> dist;
        for (unsigned int d = 0; d < spacedim; ++d) {
            dist.emplace_back(min_point[d], max_point[d]);
        }
    
        // Generate random points within the bounding box
        barycenter_points.resize(barycenter_params.n_barycenter_points);
        for (unsigned int i = 0; i < barycenter_params.n_barycenter_points; ++i) {
            for (unsigned int d = 0; d < spacedim; ++d) {
                barycenter_points[i][d] = dist[d](rng);
            }
        }
    }
    else {
        pcout << "Using support points from the second mesh for initialization" << std::endl;
        std::map<types::global_dof_index, Point<spacedim>> support_points;
        DoFTools::map_dofs_to_support_points(MappingFE<dim, spacedim>(source2_fe),
                                            source2_dof_handler,
                                            support_points);
        for (const auto& [index, point] : support_points) {
            barycenter_points.push_back(point);
        }
        barycenter_params.n_barycenter_points = barycenter_points.size();
    }

    Vector<double> barycenter_weights(barycenter_params.n_barycenter_points);
    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
        Utils::write_vector(barycenter_points, "barycenter_points.txt");
        
        // Save initial barycenter points in VTK format
        if (file_params.save_vtk) {
            save_vtk_output(barycenter_points, barycenter_weights,
                           file_params.output_prefix + "_initial.vtk");
            pcout << "Saved initial barycenter to " << file_params.output_prefix << "_initial.vtk" << std::endl;
        }
    }
    barycenter_weights = 1.0 / barycenter_params.n_barycenter_points;
    pcout << "Initialized barycenter with " << barycenter_params.n_barycenter_points << " random points." << std::endl;

    // Initialize step controller
    AdaptiveStepController step_controller(step_params);
    Vector<double> potentials_1(barycenter_params.n_barycenter_points);
    Vector<double> potentials_2(barycenter_params.n_barycenter_points);

    // Main barycenter iteration loop
    for (unsigned int iter = 0; iter < barycenter_params.max_iterations; ++iter)
    {
        pcout << "\n--- Barycenter Iteration " << iter + 1 << " ---" << std::endl;
        std::vector<Point<spacedim>> prev_points = barycenter_points;

        sot_problem_1.setup_target_measure(barycenter_points, barycenter_weights);
        sot_problem_2.setup_target_measure(barycenter_points, barycenter_weights);

        pcout << "  Solving OT problems..." << std::endl;
        potentials_1 = sot_problem_1.solve(potentials_1);
        potentials_2 = sot_problem_2.solve(potentials_2);

        Vector<double> grad_points_1(barycenter_params.n_barycenter_points * spacedim);
        Vector<double> grad_points_2(barycenter_params.n_barycenter_points * spacedim);

        // Compute gradients
        MappingFE<dim, spacedim> source1_mapping(source1_fe);
        WorkStream::run(
            source1_dof_handler.begin_active(), source1_dof_handler.end(),
            [&](const auto& cell, auto& scratch, auto& copy) {
                if (cell->is_locally_owned())
                    local_assemble_barycenter_gradient(cell, scratch, copy, source1_density,
                                                      barycenter_points, barycenter_weights,
                                                      potentials_1, ot_params.epsilon,
                                                      ot_params.distance_threshold);
            },
            [&](const auto& copy) { copy_local_to_global_barycenter(copy, grad_points_1); },
            BarycenterScratchData<dim, spacedim>(source1_fe, source1_mapping, 2),
            BarycenterCopyData<spacedim>(barycenter_params.n_barycenter_points, spacedim));

        MappingFE<dim, spacedim> source2_mapping(source2_fe);
        WorkStream::run(
            source2_dof_handler.begin_active(), source2_dof_handler.end(),
            [&](const auto& cell, auto& scratch, auto& copy) {
                if (cell->is_locally_owned())
                    local_assemble_barycenter_gradient(cell, scratch, copy, source2_density,
                                                      barycenter_points, barycenter_weights,
                                                      potentials_2, ot_params.epsilon,
                                                      ot_params.distance_threshold);
            },
            [&](const auto& copy) { copy_local_to_global_barycenter(copy, grad_points_2); },
            BarycenterScratchData<dim, spacedim>(source2_fe, source2_mapping, 2),
            BarycenterCopyData<spacedim>(barycenter_params.n_barycenter_points, spacedim));

        Utilities::MPI::sum(grad_points_1, mpi_comm, grad_points_1);
        Utilities::MPI::sum(grad_points_2, mpi_comm, grad_points_2);

        // Update barycenter points
        Vector<double> total_grad_points(barycenter_params.n_barycenter_points * spacedim);
        total_grad_points.add(barycenter_params.weight_1, grad_points_1,
                             barycenter_params.weight_2, grad_points_2);

        const double alpha = step_controller.get();
        for (unsigned int i = 0; i < barycenter_params.n_barycenter_points; ++i)
        {
            for (unsigned int d = 0; d < spacedim; ++d)
            {
                barycenter_points[i][d] -= alpha * total_grad_points[i * spacedim + d];
            }
        }

        // Check convergence
        double point_change = 0.0;
        for (unsigned int i = 0; i < barycenter_params.n_barycenter_points; ++i)
        {
            point_change += barycenter_points[i].distance_square(prev_points[i]);
        }
        point_change = std::sqrt(point_change / barycenter_params.n_barycenter_points);
        step_controller.update(point_change);

        pcout << "  RMS Point Change: " << std::scientific << std::setprecision(3) << point_change
              << ", Step Size: " << std::fixed << std::setprecision(4) << alpha << std::endl;

        if (point_change < barycenter_params.convergence_tolerance)
        {
            pcout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }

        // Output intermediate results
        if (Utilities::MPI::this_mpi_process(mpi_comm) == 0 &&
            (iter + 1) % barycenter_params.output_frequency == 0)
        {
            std::string iter_suffix = "_iter_" + std::to_string(iter + 1);

            if (file_params.save_txt) {
                Utils::write_vector(barycenter_points, file_params.output_prefix + iter_suffix);
            }

            if (file_params.save_vtk) {
                save_vtk_output(barycenter_points, barycenter_weights,
                               file_params.output_prefix + iter_suffix + ".vtk");
            }
        }
    }

    // Save final results
    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
        if (file_params.save_txt) {
            Utils::write_vector(barycenter_points, file_params.output_prefix + "_final");
        }

        if (file_params.save_vtk) {
            save_vtk_output(barycenter_points, barycenter_weights,
                           file_params.output_prefix + "_final.vtk");
        }

        pcout << "\nSaved final barycenter to " << file_params.output_prefix << "_final.*" << std::endl;
    }

    pcout << "=== Barycenter Tutorial Complete ===" << std::endl;

    return 0;
}
