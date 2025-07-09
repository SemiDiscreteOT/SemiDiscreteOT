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

// =================================================================================
// These parameter classes and helper functions are preserved from your original code.
// =================================================================================

// Parameter classes using ParameterAcceptor
class BarycenterParameters : public ParameterAcceptor
{
public:
    BarycenterParameters() : ParameterAcceptor("Barycenter")
    {
        add_parameter("volume_scaling", volume_scaling,
                     "Volume scaling");
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
    unsigned int n_barycenter_points = 1000;
    unsigned int random_seed = 42;
    unsigned int output_frequency = 5;
    bool random_initialization = true;
    double initial_bounds_min = -1.0;
    double initial_bounds_max = 1.0;
    bool volume_scaling = false;
};

class OptimalTransportParameters : public ParameterAcceptor
{
public:
    OptimalTransportParameters() : ParameterAcceptor("Optimal Transport")
    {
        add_parameter("epsilon", epsilon, "Regularization parameter");
        add_parameter("tau", tau, "Numerical stability parameter");
        add_parameter("distance_threshold", distance_threshold,
                     "Distance threshold for computational efficiency");
        add_parameter("max_iterations", max_iterations, "Maximum iterations for OT solver");
        add_parameter("tolerance", tolerance, "Tolerance for OT solver");
        add_parameter("use_log_sum_exp_trick", use_log_sum_exp_trick, "Use log-sum-exp trick for numerical stability");
        add_parameter("verbose_output", verbose_output, "Enable verbose output for OT solver");
        add_parameter("distance_threshold_type", distance_threshold_type, "Type of distance threshold");
        add_parameter("source_multilevel_enabled", source_multilevel_enabled, "Enable multilevel for source");
        add_parameter("target_multilevel_enabled", target_multilevel_enabled, "Enable multilevel for target");
        add_parameter("source_min_vertices", source_min_vertices, "Minimum number of vertices for source multilevel");
        add_parameter("source_max_vertices", source_max_vertices, "Maximum number of vertices for source multilevel");
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
        add_parameter("source1_filename", source1_filename, "Filename for first source mesh");
        add_parameter("source2_filename", source2_filename, "Filename for second source mesh");
        add_parameter("output_prefix", output_prefix, "Prefix for output files");
        add_parameter("save_vtk", save_vtk, "Save results in VTK format");
        add_parameter("save_txt", save_txt, "Save results in text format");
    }

    std::string source1_filename = "source1.msh";
    std::string source2_filename = "source2.msh";
    std::string output_prefix = "barycenter";
    bool save_vtk = true;
    bool save_txt = true;
};

void save_vtk_output(const std::vector<Point<3>> &points, const Vector<double> &weights, const std::string &filename)
{
    std::ofstream vtk_file(filename);
    if (!vtk_file.is_open()) return;
    const unsigned int n_points = points.size();
    vtk_file << "# vtk DataFile Version 3.0\nBarycenter Points\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    vtk_file << "POINTS " << n_points << " double\n";
    for (const auto &point : points) vtk_file << point[0] << " " << point[1] << " " << point[2] << "\n";
    vtk_file << "CELLS " << n_points << " " << 2 * n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) vtk_file << "1 " << i << "\n";
    vtk_file << "CELL_TYPES " << n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) vtk_file << "1\n";
    if (weights.size() == n_points) {
        vtk_file << "POINT_DATA " << n_points << "\nSCALARS weights double 1\nLOOKUP_TABLE default\n";
        for (unsigned int i = 0; i < n_points; ++i) vtk_file << weights[i] << "\n";
    }
    vtk_file.close();
}


int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    pcout << "=== Wasserstein Barycenter Tutorial (Lloyd's Algorithm) ===" << std::endl;

    BarycenterParameters barycenter_params;
    OptimalTransportParameters ot_params;
    FileParameters file_params;

    std::string prm_file = (argc > 1) ? argv[1] : "parameters.prm";
    ParameterAcceptor::initialize(prm_file);

    if (std::abs(barycenter_params.weight_1 + barycenter_params.weight_2 - 1.0) > 1e-12) {
        double sum = barycenter_params.weight_1 + barycenter_params.weight_2;
        barycenter_params.weight_1 /= sum;
        barycenter_params.weight_2 /= sum;
    }
    pcout << "Weight 1: " << barycenter_params.weight_1 << ", Weight 2: " << barycenter_params.weight_2 << std::endl;

    const int dim = 3, spacedim = 3;

    auto configure_solver = [&](SotParameterManager &p, int solver_id) {
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
        p.multilevel_params.source_hierarchy_dir = "output/barycenter_h/source" + std::to_string(solver_id);
    };

    auto problem1 = std::make_shared<SemiDiscreteOT<dim, spacedim>>(mpi_comm);
    problem1->configure([&](SotParameterManager &p) { configure_solver(p, 1); });
    problem1->get_solver()->set_distance_function("euclidean");

    auto problem2 = std::make_shared<SemiDiscreteOT<dim, spacedim>>(mpi_comm);
    problem2->configure([&](SotParameterManager &p) { configure_solver(p, 2); });
    problem2->get_solver()->set_distance_function("euclidean");

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

        if (barycenter_params.volume_scaling) {
            const double volume = GridTools::volume(source1_tria);
            if (volume > 1e-12) {
                GridTools::scale(1.0 / std::cbrt(volume), source1_tria);
                pcout << "Rescaled source 1 to unit volume." << std::endl;
            }
        }
    }
    FE_SimplexP<dim, spacedim> source1_fe(1);
    DoFHandler<dim, spacedim> source1_dof_handler(source1_tria);
    source1_dof_handler.distribute_dofs(source1_fe);
    Vector<double> source1_density(source1_dof_handler.n_dofs());
    source1_density = 1.0;
    problem1->setup_source_measure(source1_tria, source1_dof_handler, source1_density, "source_1");

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

        if (barycenter_params.volume_scaling) {
            const double volume = GridTools::volume(source2_tria);
            if (volume > 1e-12) {
                GridTools::scale(1.0 / std::cbrt(volume), source2_tria);
                pcout << "Rescaled source 2 to unit volume." << std::endl;
            }
        }
    }
    FE_SimplexP<dim, spacedim> source2_fe(1);
    DoFHandler<dim, spacedim> source2_dof_handler(source2_tria);
    source2_dof_handler.distribute_dofs(source2_fe);
    Vector<double> source2_density(source2_dof_handler.n_dofs());
    source2_density = 1.0;
    problem2->setup_source_measure(source2_tria, source2_dof_handler, source2_density, "source_2");


    problem1->prepare_multilevel_hierarchies();
    problem2->prepare_multilevel_hierarchies();

    std::vector<Point<spacedim>> barycenter_points(barycenter_params.n_barycenter_points);
    Vector<double> barycenter_weights(barycenter_params.n_barycenter_points);
    barycenter_weights = 1.0 / barycenter_params.n_barycenter_points;

    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
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
        } else {
            // Use support points of second mesh for initialization
            std::map<types::global_dof_index, Point<spacedim>> support_points;
            DoFTools::map_dofs_to_support_points(MappingFE<dim>(source2_fe), source2_dof_handler, support_points);
            barycenter_points.resize(support_points.size());
            unsigned int i = 0;
            for(const auto& pair : support_points) barycenter_points[i++] = pair.second;
            barycenter_weights.reinit(barycenter_points.size());
            barycenter_weights = 1.0 / barycenter_points.size();
        }
        if (file_params.save_vtk) save_vtk_output(barycenter_points, barycenter_weights, file_params.output_prefix + "_initial.vtk");
    }
    for (auto &p : barycenter_points) p = Utilities::MPI::broadcast(mpi_comm, p, 0);
    pcout << "Initialized barycenter with " << barycenter_points.size() << " points." << std::endl;

    Vector<double> potentials_1(barycenter_points.size()), potentials_2(barycenter_points.size());
    const double theta = 0.5;

    for (unsigned int iter = 0; iter < barycenter_params.max_iterations; ++iter) {
        pcout << "\n--- Barycenter Iteration " << iter + 1 << " ---" << std::endl;
        const std::vector<Point<spacedim>> prev_points = barycenter_points;

        // Step 1: Solve for potentials from current barycenter to each source
        problem1->setup_target_measure(barycenter_points, barycenter_weights);
        potentials_1 = problem1->solve(potentials_1);

        problem2->setup_target_measure(barycenter_points, barycenter_weights);
        potentials_2 = problem2->solve(potentials_2);

        std::vector<Point<spacedim>> centroids_1 (barycenter_points.size());
        problem1->get_solver()->evaluate_weighted_barycenters(potentials_1, centroids_1, problem1->get_solver_params());

        std::vector<Point<spacedim>> centroids_2 (barycenter_points.size());
        problem2->get_solver()->evaluate_weighted_barycenters(potentials_2, centroids_2, problem2->get_solver_params());

        // Step 3: Update barycenter points with the Lloyd's algorithm rule
        for (unsigned int j = 0; j < barycenter_points.size(); ++j) {
            Point<spacedim> ideal_position = barycenter_params.weight_1 * centroids_1[j] + barycenter_params.weight_2 * centroids_2[j];
            barycenter_points[j] = (1.0 - theta) * prev_points[j] + theta * ideal_position;
        }

        // Step 4: Check for convergence
        double point_change = 0.0;
        if (Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
            for (unsigned int i = 0; i < barycenter_points.size(); ++i) {
                point_change += barycenter_points[i].distance_square(prev_points[i]);
            }
            point_change = std::sqrt(point_change / barycenter_points.size());
        }
        point_change = Utilities::MPI::broadcast(mpi_comm, point_change, 0);

        pcout << "  RMS Point Change: " << std::scientific << std::setprecision(4) << point_change << std::endl;
        if (iter > 0 && point_change < barycenter_params.convergence_tolerance) {
            pcout << Color::green << "Converged after " << iter + 1 << " iterations." << Color::reset << std::endl;
            break;
        }

        if (Utilities::MPI::this_mpi_process(mpi_comm) == 0 && (iter + 1) % barycenter_params.output_frequency == 0) {
            std::string iter_suffix = "_iter_" + std::to_string(iter + 1);
            if (file_params.save_vtk) save_vtk_output(barycenter_points, barycenter_weights, file_params.output_prefix + iter_suffix + ".vtk");
            if (file_params.save_txt) Utils::write_vector(barycenter_points, file_params.output_prefix + iter_suffix);
        }
    }

    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
        if (file_params.save_vtk) save_vtk_output(barycenter_points, barycenter_weights, file_params.output_prefix + "_final.vtk");
        pcout << "\nSaved final barycenter to " << file_params.output_prefix << "_final.vtk" << std::endl;
    }

    pcout << "=== Barycenter Tutorial Complete ===" << std::endl;
    return 0;
}
