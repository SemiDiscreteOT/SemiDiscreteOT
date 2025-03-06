#ifndef RSOT_H
#define RSOT_H

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/optimization/solver_bfgs.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/numerics/rtree.h>
#include "MeshHierarchy.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <atomic>
#include <boost/geometry/index/rtree.hpp>


using namespace dealii;

template <int dim>
class Convex2Convex : public ParameterAcceptor {
public:
    Convex2Convex(const MPI_Comm &mpi_communicator);
    void run();
    void save_discrete_measures();
    
    // New overload of run_sot that allows specifying target points directly
    void run_sot(const std::vector<Point<dim>>& custom_target_points, 
                 const std::vector<double>& custom_target_weights);

private:
    // MPI-related members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Scratch data for parallel assembly
    struct ScratchData {
        ScratchData(const FiniteElement<dim> &fe,
                   const Mapping<dim> &mapping,
                   const Quadrature<dim> &quadrature)
            : fe_values(mapping, fe, quadrature,
                       update_values | update_quadrature_points | update_JxW_values),
              density_values(quadrature.size()) {}

        ScratchData(const ScratchData &scratch_data)
            : fe_values(scratch_data.fe_values.get_mapping(),
                       scratch_data.fe_values.get_fe(),
                       scratch_data.fe_values.get_quadrature(),
                       update_values | update_quadrature_points | update_JxW_values),
              density_values(scratch_data.density_values) {}

        FEValues<dim> fe_values;
        std::vector<double> density_values;
    };

    // Copy data for parallel assembly
    struct CopyData {
        double functional_value{0.0};
        Vector<double> gradient_values;
        double C_integral{0.0};  // Added: contribution to the C integral
        Vector<double> weight_values;  // Added: for softmax refinement

        CopyData(const unsigned int n_target_points)
            : gradient_values(n_target_points),
              weight_values(n_target_points) {}
    };

    // Helper functions for parallel assembly
    void local_assemble_sot(const typename DoFHandler<dim>::active_cell_iterator &cell,
                           ScratchData &scratch_data,
                           CopyData &copy_data);

    void copy_local_to_global(const CopyData &copy_data);

    // Mutex for thread-safe updates
    mutable std::mutex assembly_mutex;
    double global_functional{0.0};
    double global_C_integral{0.0};  // Added: global C integral value
    LinearAlgebra::distributed::Vector<double> global_gradient;
    const Vector<double>* current_weights{nullptr};
    double current_lambda{0.0};

    // Debug tracking variables
    std::atomic<size_t> total_target_points{0};

    void mesh_generation();
    void print_parameters();
    void load_mesh_source();
    void load_mesh_target();
    void load_meshes(); 
    void run_sot();

    // Multilevel parameters for source mesh
    struct MultilevelParameters {
        int min_vertices = 1000;
        int max_vertices = 10000;
        std::string hierarchy_output_dir = "output/multilevel/meshes";
        std::string output_prefix = "output/multilevel/sot";  // Where to save multilevel results
    } multilevel_params;
    
    // Multilevel parameters for target point cloud
    struct TargetMultilevelParameters {
        int min_points = 100;
        int max_points = 1000;
        std::string hierarchy_output_dir = "output/target_multilevel";
        std::string output_prefix = "output/target_multilevel/sot";  // Where to save multilevel results
        bool enabled = false;  // Whether to use target multilevel approach
        bool use_softmax_weight_transfer = true;  // Whether to use softmax-based weight transfer
    } target_multilevel_params;

    // Source mesh multilevel methods
    void prepare_multilevel();
    void run_multilevel_sot();
    void load_mesh_at_level(const std::string& mesh_file);
    std::vector<std::string> get_mesh_hierarchy_files() const;
    
    // Target point cloud multilevel methods
    void prepare_target_multilevel();
    void run_target_multilevel_sot();
    void load_target_points_at_level(const std::string& points_file, const std::string& weights_file);
    std::vector<std::pair<std::string, std::string>> get_target_hierarchy_files() const;

    // Combined multilevel method
    void run_combined_multilevel_sot();
    void run_target_multilevel_for_source_level(const std::string& source_mesh_file, Vector<double>& weights);

    template <int d = dim>
    typename std::enable_if<d == 3>::type run_exact_sot();  // Only available for dim=3
    void compute_power_diagram();
    void compute_transport_map();  // New function

    std::string selected_task;
    std::string io_coding = "txt";

    struct MeshParameters {
        unsigned int n_refinements = 0;
        std::string grid_generator_function;
        std::string grid_generator_arguments;
        bool use_tetrahedral_mesh = false;
    };

    MeshParameters source_params;
    MeshParameters target_params;

    parallel::fullydistributed::Triangulation<dim> source_mesh;
    Triangulation<dim> target_mesh;  // Target mesh stays serial
    DoFHandler<dim> dof_handler_source;
    DoFHandler<dim> dof_handler_target;
    std::unique_ptr<FiniteElement<dim>> fe_system;
    std::unique_ptr<Mapping<dim>> mapping;
    LinearAlgebra::distributed::Vector<double> source_density;
    std::vector<Point<dim>> target_points;
    std::vector<Point<dim>> source_points;
    Vector<double> target_density;
    std::unique_ptr<SolverControl> solver_control;
    struct SolverParameters {
        unsigned int max_iterations = 1000;
        double tolerance = 1e-8;
        double regularization_param = 1e-3;
        double epsilon = 1e-8;  // Parameter for truncation criterion
        double tau = 1e-8;  // Truncation error tolerance for integral radius bound
        bool verbose_output = true;
        std::string solver_type = "BFGS";
        unsigned int quadrature_order = 3;
        unsigned int nb_points = 1000;
        unsigned int number_of_threads = 0;  // 0 means use all available cores
        bool use_epsilon_scaling = false;  // Whether to use epsilon scaling strategy
        double epsilon_scaling_factor = 2.0;  // Factor by which to reduce epsilon in each step
        unsigned int epsilon_scaling_steps = 5;  // Number of epsilon scaling steps
        bool use_caching = false;  // Whether to enable caching of computations
    } solver_params;

    struct PowerDiagramParameters {
        std::string implementation = "dealii";  // Options: dealii/geogram
    } power_diagram_params;

    struct TransportMapParameters {
        unsigned int n_neighbors = 10;
        double kernel_width = 0.1;
        std::string interpolation_type = "linear";
    } transport_map_params;

    // Template version of generate_mesh to handle both types of triangulation
    template <typename TriangulationType>
    void generate_mesh(TriangulationType &tria,
                      const std::string &grid_generator_function,
                      const std::string &grid_generator_arguments,
                      const unsigned int n_refinements,
                      const bool use_tetrahedral_mesh);

    void save_meshes();
    void setup_source_finite_elements();
    void setup_target_finite_elements();
    void setup_finite_elements();
    void setup_target_points();  // New helper to set up target points and RTree once
    void setup_custom_target_points(const std::vector<Point<dim>>& custom_target_points,
                                   const std::vector<double>& custom_target_weights);
    void setup_multilevel_finite_elements();  
    double evaluate_sot_functional(const Vector<double>& weights, Vector<double>& gradient);
    void save_results(const Vector<double>& weights, const std::string& filename);
    
    // Map weights from one target point set to another (for multilevel transfer)
    // void interpolate_weights(const std::vector<Point<dim>>& source_points,
    //                         const Vector<double>& source_weights,
    //                         const std::vector<Point<dim>>& target_points,
    //                         Vector<double>& target_weights);

    // Cache for local assembly computations
    struct CellCache {
        // Weight-independent data (can be cached)
        std::vector<std::size_t> target_indices;
        std::vector<double> precomputed_exp_terms;  // density * exp(-0.5*dist2/lambda) for each q_point and target
        bool is_valid;

        CellCache() : is_valid(false) {}
        void invalidate() { is_valid = false; }
    };
    // mutable std::map<typename DoFHandler<dim>::active_cell_iterator, CellCache> cell_caches;
    mutable std::unordered_map<std::string, CellCache> cell_caches;
    mutable std::mutex cache_mutex;

    // RTree for spatial queries on target points
    using IndexedPoint = std::pair<Point<dim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;
    RTree target_points_rtree;

    // Caching-related members
    mutable double effective_distance_threshold{0.0};  // 10% increased threshold for caching
    mutable bool is_caching_active{false};  // Whether caching is currently active

    // Softmax refinement members
    std::vector<Point<dim>>& target_points_fine{target_points};  // Fine level target points
    Vector<double>& target_density_coarse{target_density};  // Coarse level densities
    Vector<double> weights_fine;  // Fine level weights
    std::vector<Point<dim>> target_points_coarse;  // Coarse level target points
    const Vector<double>* weights_coarse{nullptr};  // Pointer to coarse level weights
    int current_level{0};  // Current level in hierarchy

    // Softmax refinement methods
    Vector<double> softmax_refinement(const Vector<double>& weights);
    void local_assemble_softmax_refinement(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        ScratchData &scratch_data,
        CopyData &copy_data);

    // Weight transfer between hierarchy levels using softmax refinement
    void assign_weights_by_hierarchy(Vector<double>& weights, int coarse_level, int fine_level, const Vector<double>& prev_weights);

    // Computed distance threshold for current iteration
    mutable double current_distance_threshold{0.0};
    // Compute the distance threshold based on current weights and parameters
    void compute_distance_threshold() const;
    // Reset the caching state and thresholds
    void reset_distance_threshold_cache() const;
    // Helper function for nearest neighbor queries
    std::vector<std::size_t> find_nearest_target_points(const Point<dim>& query_point) const;
    // Helper function for range queries
    std::vector<std::size_t> find_target_points_in_box(const BoundingBox<dim>& box) const;

    // Compute the tau integral radius bound for truncation
    double compute_tau_integral_radius(const double tau, const double M, const double C, const double F_phi) const;
    
    // Calculate the total size of the cache in MB
    double calculate_cache_size_mb() const;

    // Solve optimization problem for a specific level
    // void solve_level(int level);
    
    // Parent-child relationship data structures
    std::vector<std::vector<std::vector<size_t>>> parent_indices_;
    std::vector<std::vector<std::vector<size_t>>> child_indices_;
    bool has_hierarchy_data_{false};
    
    // Find target points using parent-child relationships
    std::vector<std::size_t> find_target_points_by_hierarchy(const Point<dim>& query_point, int level) const;
    
    // Load hierarchy data from files
    void load_hierarchy_data(const std::string& hierarchy_dir, int specific_level = -1);
};

#endif
