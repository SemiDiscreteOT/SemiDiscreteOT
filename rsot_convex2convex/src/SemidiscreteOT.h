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
#include "MeshManager.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <atomic>
#include <boost/geometry/index/rtree.hpp>

#include "ParameterManager.h"
#include "SotSolver.h"
#include "EpsilonScalingHandler.h"

using namespace dealii;

template <int dim>
class SemidiscreteOT {
public:
    SemidiscreteOT(const MPI_Comm &mpi_communicator);
    void run();
    void save_discrete_measures();
    void run_sot(const std::vector<Point<dim>>& custom_target_points, 
                 const std::vector<double>& custom_target_weights);

private:
    // MPI-related members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Parameter manager and references
    ParameterManager param_manager;
    ParameterManager::MeshParameters& source_params;
    ParameterManager::MeshParameters& target_params;
    ParameterManager::SolverParameters& solver_params;
    ParameterManager::MultilevelParameters& multilevel_params;
    ParameterManager::PowerDiagramParameters& power_diagram_params;
    ParameterManager::TransportMapParameters& transport_map_params;
    std::string& selected_task;
    std::string& io_coding;

    // Mesh and DoF handler members
    parallel::fullydistributed::Triangulation<dim> source_mesh;
    Triangulation<dim> target_mesh;
    DoFHandler<dim> dof_handler_source;
    DoFHandler<dim> dof_handler_target;

    // Mesh manager
    std::unique_ptr<MeshManager<dim>> mesh_manager;

    // Epsilon scaling handler
    std::unique_ptr<EpsilonScalingHandler> epsilon_scaling_handler;

    // Core functionality methods
    void mesh_generation();
    void print_parameters();
    void load_meshes();
    void run_sot();
    void compute_power_diagram();
    void compute_transport_map();

    // Multilevel methods
    void prepare_source_multilevel();
    void prepare_target_multilevel();
    void run_multilevel_sot();
    void run_target_multilevel(const std::string& source_mesh_file = "",
                             Vector<double>* output_weights = nullptr,
                             bool save_results_to_files = true);
    void run_target_multilevel_for_source_level(const std::string& source_mesh_file, 
                                              Vector<double>& weights);

    // Setup methods
    void setup_source_finite_elements();
    void setup_target_finite_elements();
    void setup_finite_elements();
    void setup_target_points();
    void setup_multilevel_finite_elements();
    void save_results(const Vector<double>& weights, const std::string& filename);

    // Exact SOT method (3D only)
    template <int d = dim>
    typename std::enable_if<d == 3>::type run_exact_sot();

    // Finite element and mapping members
    std::unique_ptr<FiniteElement<dim>> fe_system;
    std::unique_ptr<Mapping<dim>> mapping;
    LinearAlgebra::distributed::Vector<double> source_density;
    std::vector<Point<dim>> target_points;
    std::vector<Point<dim>> source_points;
    Vector<double> target_density;

    // Multilevel computation state
    std::vector<Point<dim>> target_points_coarse;  // Coarse level target points
    Vector<double> target_density_coarse;          // Coarse level densities
    mutable double current_distance_threshold{0.0}; // Current distance threshold for computations

    // Weight transfer between hierarchy levels
    void assign_weights_by_hierarchy(Vector<double>& weights, 
                                   int coarse_level, 
                                   int fine_level, 
                                   const Vector<double>& prev_weights);

    // Helper methods
    std::vector<std::pair<std::string, std::string>> get_target_hierarchy_files() const;
    std::vector<std::string> get_mesh_hierarchy_files() const;
    void load_target_points_at_level(const std::string& points_file, 
                                   const std::string& weights_file);

    // Solver member
    std::unique_ptr<SotSolver<dim>> sot_solver;
};

#endif
