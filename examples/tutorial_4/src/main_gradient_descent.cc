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
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <iomanip>

using namespace dealii;

// Boost geometry types for rtree
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using BoostPoint = bg::model::point<double, 3, bg::cs::cartesian>;
using IndexedPoint = std::pair<BoostPoint, std::size_t>;

// Global variables for rtree and distance threshold
bgi::rtree<IndexedPoint, bgi::rstar<16>> target_rtree;
double current_distance_threshold = 1.5;


// Function to find nearest target points using rtree
template<int spacedim>
std::vector<std::size_t> find_nearest_target_points(
    const Point<spacedim>& query_point,
    const bgi::rtree<IndexedPoint, bgi::rstar<16>>& rtree,
    double distance_threshold)
{
    std::vector<std::size_t> indices;
    
    // Convert dealii Point to BoostPoint
    BoostPoint boost_query;
    if constexpr (spacedim >= 1) bg::set<0>(boost_query, query_point[0]);
    if constexpr (spacedim >= 2) bg::set<1>(boost_query, query_point[1]);
    if constexpr (spacedim >= 3) bg::set<2>(boost_query, query_point[2]);
    if constexpr (spacedim < 3) bg::set<2>(boost_query, 0.0);
    
    // Find points within distance threshold
    for (const auto& indexed_point : rtree |
        bgi::adaptors::queried(bgi::satisfies([&](const IndexedPoint& p) {
            Point<spacedim> dealii_point;
            if constexpr (spacedim >= 1) dealii_point[0] = bg::get<0>(p.first);
            if constexpr (spacedim >= 2) dealii_point[1] = bg::get<1>(p.first);
            if constexpr (spacedim >= 3) dealii_point[2] = bg::get<2>(p.first);
            return euclidean_distance<spacedim>(dealii_point, query_point) <= distance_threshold;
        })))
    {
        indices.push_back(indexed_point.second);
    }

    return indices;
}

// Function to initialize rtree with target points
template<int spacedim>
void initialize_target_rtree(const std::vector<Point<spacedim>>& target_points,
                            bgi::rtree<IndexedPoint, bgi::rstar<16>>& rtree)
{
    rtree.clear();
    std::vector<IndexedPoint> indexed_points;
    indexed_points.reserve(target_points.size());
    
    for (std::size_t i = 0; i < target_points.size(); ++i) {
        BoostPoint boost_point;
        if constexpr (spacedim >= 1) bg::set<0>(boost_point, target_points[i][0]);
        if constexpr (spacedim >= 2) bg::set<1>(boost_point, target_points[i][1]);
        if constexpr (spacedim >= 3) bg::set<2>(boost_point, target_points[i][2]);
        if constexpr (spacedim < 3) bg::set<2>(boost_point, 0.0);
        indexed_points.emplace_back(boost_point, i);
    }
    
    rtree = bgi::rtree<IndexedPoint, bgi::rstar<16>>(indexed_points);
}

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
        add_parameter("sampling_id", sampling_id,
                     "Source ID to sample from (0=both, 1=source1, 2=source2)");
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
    unsigned int sampling_id = 0;
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
        add_parameter("target_min_points", target_min_points,
                     "Minimum number of points for target multilevel");
        add_parameter("target_max_points", target_max_points,
                     "Maximum number of points for target multilevel");
        add_parameter("use_python_clustering", use_python_clustering,
                     "Whether to use Python scripts for clustering");
        add_parameter("python_script_name", python_script_name,
                     "Name of the Python script to use");
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
    unsigned int target_min_points = 100;
    unsigned int target_max_points = 1000;
    bool use_python_clustering = true;
    std::string python_script_name = "multilevel_clustering_faiss_cpu.py";
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

    scratch.fe_values.reinit(cell);
    scratch.fe_values.get_function_values(source_density, scratch.density_values);

    copy.grad_support_points = 0.0;

    std::vector<std::size_t> nearby_targets = find_nearest_target_points<spacedim>(
        cell->center(), target_rtree, distance_threshold);

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

        for (std::size_t idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const std::size_t j = nearby_targets[idx];
            const double dist_sq = x.distance_square(target_points[j]);
            const double exponent = (potentials[j] - 0.5 * dist_sq) / epsilon;
            max_exponent = std::max(max_exponent, exponent);
        }

        for (std::size_t idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const std::size_t j = nearby_targets[idx];
            const double dist_sq = x.distance_square(target_points[j]);
            const double exponent = (potentials[j] - 0.5 * dist_sq) / epsilon;
            weights[idx] = target_weights(j) * std::exp(exponent - max_exponent);
            sum_weights += weights[idx];
        }

        if (sum_weights < 1e-12)
            continue;

        for (std::size_t idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const std::size_t j = nearby_targets[idx];
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

bool load_vtk_points(const std::string &filename, std::vector<Point<3>> &points)
{
    std::ifstream vtk_file(filename);
    if (!vtk_file.is_open()) return false;

    points.clear();
    std::string line;

    // Skip until we find the POINTS line
    while (std::getline(vtk_file, line)) {
        if (line.find("POINTS") != std::string::npos) {
            std::istringstream iss(line);
            std::string points_keyword;
            unsigned int n_points;
            std::string data_type;

            iss >> points_keyword >> n_points >> data_type;
            points.reserve(n_points);

            // Read the points
            for (unsigned int i = 0; i < n_points; ++i) {
                double x, y, z;
                if (!(vtk_file >> x >> y >> z)) return false;
                points.emplace_back(x, y, z);
            }

            return true;
        }
    }

    return false;
}

template <int dim, int spacedim>
std::vector<Point<spacedim>> sample_points_from_geometry(const Triangulation<dim, spacedim> &triangulation,
                                                         unsigned int n_samples,
                                                         std::mt19937 &rng)
{
    std::vector<Point<spacedim>> sampled_points;
    sampled_points.reserve(n_samples);

    // Calculate volumes of all cells for weighted sampling
    std::vector<double> cell_volumes;
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator> cells;

    for (const auto &cell : triangulation.active_cell_iterators()) {
        cells.push_back(cell);
        cell_volumes.push_back(cell->measure());
    }

    // Create discrete distribution for cell selection based on volume
    std::discrete_distribution<> cell_dist(cell_volumes.begin(), cell_volumes.end());

    // Uniform distributions for barycentric coordinates
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    for (unsigned int i = 0; i < n_samples; ++i) {
        // Select a random cell weighted by volume
        int cell_idx = cell_dist(rng);
        const auto &cell = cells[cell_idx];

        // Generate random coordinates for quad elements
        std::vector<double> xi(dim);
        for (int j = 0; j < dim; ++j) {
            xi[j] = uniform_dist(rng);
        }

        // Convert reference coordinates to actual point using bilinear interpolation
        Point<spacedim> sampled_point;
        if (dim == 2) {
            // For 2D quads: bilinear interpolation
            sampled_point = (1.0 - xi[0]) * (1.0 - xi[1]) * cell->vertex(0) +
                           xi[0] * (1.0 - xi[1]) * cell->vertex(1) +
                           xi[0] * xi[1] * cell->vertex(2) +
                           (1.0 - xi[0]) * xi[1] * cell->vertex(3);
        } else if (dim == 3) {
            // For 3D quads (hexahedra): trilinear interpolation
            sampled_point = (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) * cell->vertex(0) +
                           xi[0] * (1.0 - xi[1]) * (1.0 - xi[2]) * cell->vertex(1) +
                           xi[0] * xi[1] * (1.0 - xi[2]) * cell->vertex(2) +
                           (1.0 - xi[0]) * xi[1] * (1.0 - xi[2]) * cell->vertex(3) +
                           (1.0 - xi[0]) * (1.0 - xi[1]) * xi[2] * cell->vertex(4) +
                           xi[0] * (1.0 - xi[1]) * xi[2] * cell->vertex(5) +
                           xi[0] * xi[1] * xi[2] * cell->vertex(6) +
                           (1.0 - xi[0]) * xi[1] * xi[2] * cell->vertex(7);
        }

        sampled_points.push_back(sampled_point);
    }

    return sampled_points;
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
    std::string prm_file = (argc > 1) ? argv[1] : "parameters_gradient.prm";
    ParameterAcceptor::initialize(prm_file);

    // Validate parameters
    if (std::abs(barycenter_params.weight_1 + barycenter_params.weight_2 - 1.0) > 1e-12) {
        pcout << "Warning: Weights do not sum to 1.0. Normalizing..." << std::endl;
        double sum = barycenter_params.weight_1 + barycenter_params.weight_2;
        barycenter_params.weight_1 /= sum;
        barycenter_params.weight_2 /= sum;
    }

    const int dim = 2;
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
        p.multilevel_params.target_min_points = ot_params.target_min_points;
        p.multilevel_params.target_max_points = ot_params.target_max_points;
        p.multilevel_params.source_hierarchy_dir = "output/barycenter_h/source" + std::to_string(solver_id);
        p.multilevel_params.use_python_clustering = ot_params.use_python_clustering;
        p.multilevel_params.python_script_name = ot_params.python_script_name;
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

        if (barycenter_params.volume_scaling) {
            const double volume = GridTools::volume(source1_tria);
            if (volume > 1e-12) {
                GridTools::scale(1.0 / std::cbrt(volume), source1_tria);
                pcout << "Rescaled source 1 to unit volume." << std::endl;
            }
        }
    }
    FE_Q<dim, spacedim> source1_fe(1);
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

        if (barycenter_params.volume_scaling) {
            const double volume = GridTools::volume(source2_tria);
            if (volume > 1e-12) {
                GridTools::scale(1.0 / std::cbrt(volume), source2_tria);
                pcout << "Rescaled source 2 to unit volume." << std::endl;
            }
        }
    }
    FE_Q<dim, spacedim> source2_fe(1);
    DoFHandler<dim, spacedim> source2_dof_handler(source2_tria);
    source2_dof_handler.distribute_dofs(source2_fe);
    Vector<double> source2_density(source2_dof_handler.n_dofs());
    source2_density = 1.0;
    sot_problem_2.setup_source_measure(source2_tria, source2_dof_handler, source2_density, "source_2");

    if (ot_params.source_multilevel_enabled) {
        sot_problem_1.prepare_source_multilevel();
        sot_problem_2.prepare_source_multilevel();
    }

    std::vector<Point<spacedim>> barycenter_points(barycenter_params.n_barycenter_points);
    Vector<double> barycenter_weights(barycenter_params.n_barycenter_points);
    barycenter_weights = 1.0 / barycenter_params.n_barycenter_points;

    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
        // Try to load from barycenter_initial.vtk first
        std::vector<Point<spacedim>> loaded_points;
        if (load_vtk_points("barycenter_initial.vtk", loaded_points) &&
            loaded_points.size() == barycenter_params.n_barycenter_points) {
            pcout << "Successfully loaded barycenter initialization from barycenter_initial.vtk with "
                  << loaded_points.size() << " points" << std::endl;
            barycenter_points = loaded_points;
        } else {
            if (loaded_points.size() > 0 && loaded_points.size() != barycenter_params.n_barycenter_points) {
                pcout << "Warning: barycenter_initial.vtk contains " << loaded_points.size()
                      << " points but expected " << barycenter_params.n_barycenter_points
                      << ". Using fallback initialization." << std::endl;
            }

            if (barycenter_params.random_initialization) {
                pcout << "Using geometry-based initialization: sampling from source geometries" << std::endl;

                // Initialize random number generator
                std::mt19937 rng(barycenter_params.random_seed);

                // Calculate number of points to sample from each geometry
                unsigned int n_from_source1 = static_cast<unsigned int>(barycenter_params.weight_1 * barycenter_params.n_barycenter_points);
                unsigned int n_from_source2 = barycenter_params.n_barycenter_points - n_from_source1;

                // Sample based on sampling_id
                if (barycenter_params.sampling_id == 1) {
                    // Sample all points from source 1
                    pcout << "Sampling all " << barycenter_params.n_barycenter_points << " points from source1 (sampling_id=1)" << std::endl;
                    barycenter_points = sample_points_from_geometry(source1_tria, barycenter_params.n_barycenter_points, rng);
                }
                else if (barycenter_params.sampling_id == 2) {
                    // Sample all points from source 2
                    pcout << "Sampling all " << barycenter_params.n_barycenter_points << " points from source2 (sampling_id=2)" << std::endl;
                    barycenter_points = sample_points_from_geometry(source2_tria, barycenter_params.n_barycenter_points, rng);
                }
                else {
                    // Default behavior: sample from both sources based on weights
                    pcout << "Sampling " << n_from_source1 << " points from source1 (weight: " << barycenter_params.weight_1 << ")" << std::endl;
                    pcout << "Sampling " << n_from_source2 << " points from source2 (weight: " << barycenter_params.weight_2 << ")" << std::endl;

                    // Sample points from both geometries
                    std::vector<Point<spacedim>> points_from_source1 = sample_points_from_geometry(source1_tria, n_from_source1, rng);
                    std::vector<Point<spacedim>> points_from_source2 = sample_points_from_geometry(source2_tria, n_from_source2, rng);

                    // Combine the sampled points
                    barycenter_points.clear();
                    barycenter_points.reserve(barycenter_params.n_barycenter_points);
                    barycenter_points.insert(barycenter_points.end(), points_from_source1.begin(), points_from_source1.end());
                    barycenter_points.insert(barycenter_points.end(), points_from_source2.begin(), points_from_source2.end());

                    // Shuffle the combined points to avoid any ordering bias
                    std::shuffle(barycenter_points.begin(), barycenter_points.end(), rng);
                }
            } else {
                // Use support points of second mesh for initialization
                std::map<types::global_dof_index, Point<spacedim>> support_points;
                DoFTools::map_dofs_to_support_points(MappingQ1<dim,spacedim>(), source2_dof_handler, support_points);
                barycenter_points.resize(support_points.size());
                unsigned int i = 0;
                for(const auto& pair : support_points) barycenter_points[i++] = pair.second;
                barycenter_weights.reinit(barycenter_points.size());
                barycenter_weights = 1.0 / barycenter_points.size();
            }
        }
        if (file_params.save_vtk) save_vtk_output(barycenter_points, barycenter_weights, file_params.output_prefix + "_initial.vtk");
    }
    for (auto &p : barycenter_points) p = Utilities::MPI::broadcast(mpi_comm, p, 0);
    pcout << "Initialized barycenter with " << barycenter_points.size() << " points." << std::endl;
    
    // Initialize rtree with barycenter points
    initialize_target_rtree<spacedim>(barycenter_points, target_rtree);
    current_distance_threshold = ot_params.distance_threshold;

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
        if (ot_params.target_multilevel_enabled) {
            sot_problem_1.prepare_target_multilevel();
        }
        sot_problem_2.setup_target_measure(barycenter_points, barycenter_weights);

        if (ot_params.target_multilevel_enabled) {
            potentials_1 = sot_problem_1.solve();
            potentials_2 = sot_problem_2.solve();
        }
        else {
            potentials_1 = sot_problem_1.solve(potentials_1);
            potentials_2 = sot_problem_2.solve(potentials_2);
        }

        Vector<double> grad_points_1(barycenter_params.n_barycenter_points * spacedim);
        Vector<double> grad_points_2(barycenter_params.n_barycenter_points * spacedim);

        // Compute gradients
        MappingQ1<dim, spacedim> source1_mapping;
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

        MappingQ1<dim, spacedim> source2_mapping;
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
        
        // Update rtree with new barycenter points
        initialize_target_rtree<spacedim>(barycenter_points, target_rtree);

        // Check convergence
        double point_change = 0.0;
        for (unsigned int i = 0; i < barycenter_params.n_barycenter_points; ++i)
        {
            point_change += barycenter_points[i].distance_square(prev_points[i]);
        }
        point_change = std::sqrt(point_change / barycenter_params.n_barycenter_points);
        step_controller.update(point_change);

        pcout << "  RMS Point Change: " << std::scientific << std::setprecision(3) << point_change
              << ", Step Size: " << std::scientific << std::setprecision(3) << alpha << std::endl;

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
