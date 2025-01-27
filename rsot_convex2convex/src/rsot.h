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

        CopyData(const unsigned int n_target_points)
            : gradient_values(n_target_points) {}
    };

    // Helper functions for parallel assembly
    void local_assemble_sot(const typename DoFHandler<dim>::active_cell_iterator &cell,
                           ScratchData &scratch_data,
                           CopyData &copy_data);

    void copy_local_to_global(const CopyData &copy_data);

    // Mutex for thread-safe updates
    mutable std::mutex assembly_mutex;
    double global_functional{0.0};
    LinearAlgebra::distributed::Vector<double> global_gradient;
    const Vector<double>* current_weights{nullptr};
    double current_lambda{0.0};

    // Debug tracking variables
    std::atomic<size_t> total_target_points{0};

    void mesh_generation();
    void print_parameters();
    void load_meshes();
    void run_sot();
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

    // Vectorized data members
    std::vector<VectorizedArray<double>> target_density_vec;
    std::vector<VectorizedArray<double>> current_weights_vec;
    std::vector<std::array<VectorizedArray<double>, dim>> target_points_vec;

    struct SolverParameters {
        unsigned int max_iterations = 1000;
        double tolerance = 1e-8;
        double regularization_param = 1e-3;
        double epsilon = 1e-8;  // Parameter for truncation criterion
        bool verbose_output = true;
        std::string solver_type = "BFGS";
        unsigned int quadrature_order = 3;
        unsigned int nb_points = 1000;
        unsigned int number_of_threads = 0;  // 0 means use all available cores
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
    void setup_finite_elements();
    double evaluate_sot_functional(const Vector<double>& weights, Vector<double>& gradient);
    void save_results(const Vector<double>& weights, const std::string& filename);


    // RTree for spatial queries on target points
    using IndexedPoint = std::pair<Point<dim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;
    RTree target_points_rtree;

    // Computed distance threshold for current iteration
    mutable double current_distance_threshold{0.0};
    // Compute the distance threshold based on current weights and parameters
    void compute_distance_threshold() const;
    // Helper function for nearest neighbor queries
    std::vector<std::size_t> find_nearest_target_points(const Point<dim>& query_point) const;
    // Helper function for range queries
    std::vector<std::size_t> find_target_points_in_box(const BoundingBox<dim>& box) const;
};

#endif
