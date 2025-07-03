#ifndef RSOT_H
#define RSOT_H

#include <filesystem>
#include <memory>
#include <mutex>
#include <atomic>
#include <boost/geometry/index/rtree.hpp>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
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
#include <deal.II/lac/vector_operations_internal.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/optimization/solver_bfgs.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/data_out_base.h> 
#include <deal.II/numerics/data_out.h>  
#include <deal.II/base/logstream.h>     

#include "SemiDiscreteOT/solvers/SotSolver.h"
#include "SemiDiscreteOT/solvers/EpsilonScalingHandler.h"
#include "SemiDiscreteOT/solvers/ExactSot.h"
#include "SemiDiscreteOT/solvers/SoftmaxRefinement.h"
#include "SemiDiscreteOT/core/MeshManager.h"
#include "SemiDiscreteOT/core/MeshHierarchy.h"
#include "SemiDiscreteOT/core/OptimalTransportPlan.h"
#include "SemiDiscreteOT/core/PointCloudHierarchy.h"
#include "SemiDiscreteOT/core/PowerDiagram.h"
#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/utils/VtkHandler.h"
#include "SemiDiscreteOT/utils/ColorDefinitions.h"
#include "SemiDiscreteOT/utils/utils.h"


using namespace dealii;

template <int dim, int spacedim = dim>
class SemiDiscreteOT {
public:
    SemiDiscreteOT(const MPI_Comm &mpi_communicator);
    void run();

    /**
     * @brief Configure the solver parameters programmatically.
     * @param config_func A lambda function that takes a reference to the
     *                    SotParameterManager and modifies its parameters.
     */
    void configure(std::function<void(SotParameterManager&)> config_func);


    /**
     * @brief Setup source measure from standard dealii objects (simplified API for tutorials)
     * @param tria A standard Triangulation
     * @param dh A DoFHandler on the provided triangulation
     * @param density A standard Vector containing the density values
     * 
     * This method internally handles the conversion to distributed objects when needed for parallel computation.
     */
    void setup_source_measure(
        Triangulation<dim, spacedim>& tria,
        const DoFHandler<dim, spacedim>& dh,
        const Vector<double>& density);

    /**
     * @brief Set up the target measure from a discrete set of points and weights.
     * @param points A vector of target points.
     * @param weights A vector of weights/densities for each target point.
     */
    void setup_target_measure(
        const std::vector<Point<spacedim>>& points,
        const Vector<double>& weights);

    /**
     * @brief Pre-computes the multilevel hierarchies for source and/or target.
     * This must be called after setting up the base measures and before calling solve()
     * if multilevel computation is desired.
     */
    void prepare_multilevel_hierarchies();


    /**
     * @brief Run the optimal transport computation based on the current configuration.
     *        This method handles single-level, multilevel, and epsilon scaling automatically.
     * @return A vector containing the computed optimal transport potentials for the target points.
     */
    Vector<double> solve();


    void save_discrete_measures();
    
    void set_distance_function(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& dist)
        {
            sot_solver->distance_function = dist;
        }
        
        ConditionalOStream pcout;
protected:
    // MPI-related members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    // Solver member
    std::unique_ptr<SotSolver<dim, spacedim>> sot_solver;

    // Parameter manager and references
    SotParameterManager param_manager;
    SotParameterManager::MeshParameters& source_params;
    SotParameterManager::MeshParameters& target_params;
    SotParameterManager::SolverParameters& solver_params;
    SotParameterManager::MultilevelParameters& multilevel_params;
    SotParameterManager::PowerDiagramParameters& power_diagram_params;
    SotParameterManager::TransportMapParameters& transport_map_params;
    std::string& selected_task;
    std::string& io_coding;

    std::unique_ptr<DoFHandler<dim, spacedim>> initial_fine_dof_handler;
    std::unique_ptr<Vector<double>> initial_fine_density;
    bool is_setup_programmatically_ = false; 


    // Mesh and DoF handler members
    parallel::fullydistributed::Triangulation<dim, spacedim> source_mesh;
    Triangulation<dim, spacedim> target_mesh;
    DoFHandler<dim, spacedim> dof_handler_source;
    DoFHandler<dim, spacedim> dof_handler_target;

    std::unique_ptr<VTKHandler<dim,spacedim>> source_vtk_handler;
    DoFHandler<dim,spacedim> vtk_dof_handler_source;
    Vector<double> vtk_field_source;
    Triangulation<dim,spacedim> vtk_tria_source;
    // Finite element and mapping members
    std::unique_ptr<FiniteElement<dim, spacedim>> fe_system;
    std::unique_ptr<Mapping<dim, spacedim>> mapping;
    std::unique_ptr<FiniteElement<dim, spacedim>> fe_system_target;
    std::unique_ptr<Mapping<dim, spacedim>> mapping_target;
    LinearAlgebra::distributed::Vector<double> source_density;
    Vector<double> target_density;
    std::vector<Point<spacedim>> target_points;
    std::vector<Point<spacedim>> source_points;

    // Mesh manager
    std::unique_ptr<MeshManager<dim, spacedim>> mesh_manager;
    
    // Epsilon scaling handler
    std::unique_ptr<EpsilonScalingHandler> epsilon_scaling_handler;
    
    void save_results(const Vector<double>& potentials, const std::string& filename, bool add_epsilon_prefix = true);
    
    // Density normalization helper
    void normalize_density(LinearAlgebra::distributed::Vector<double>& density);
private:

    // Core functionality methods
    void mesh_generation();
    void load_meshes();
    
    /**
     * @brief Run single-level SOT computation.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_sot();
    
    void compute_power_diagram();
    void compute_transport_map();


    // Multilevel methods
    void prepare_source_multilevel();
    void prepare_target_multilevel();
    
    /**
     * @brief Run multilevel SOT computation (dispatcher method).
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_multilevel();
    
    /**
     * @brief Run combined source and target multilevel SOT computation.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_combined_multilevel();
    
    /**
     * @brief Run source-only multilevel SOT computation.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_source_multilevel();
    
    /**
     * @brief Run target-only multilevel SOT computation.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_target_multilevel();

    // Setup methods
    void setup_source_finite_elements(bool is_multilevel = false);
    void setup_target_finite_elements();
    void setup_finite_elements();
    void setup_target_points();
    void setup_multilevel_finite_elements();

    // Exact SOT method (3D only)
    template <int d = dim, int s = spacedim>
    typename std::enable_if<d == 3 && s == 3>::type run_exact_sot();    

    // Hierarchy-related members
    std::vector<std::vector<std::vector<size_t>>> child_indices_;
    bool has_hierarchy_data_{false};
    void load_hierarchy_data(const std::string& hierarchy_dir, int specific_level = -1);

    // Multilevel computation state
    std::vector<Point<spacedim>> target_points_coarse;  // Coarse level target points
    Vector<double> target_density_coarse;          // Coarse level densities
    mutable double current_distance_threshold{0.0}; // Current distance threshold for computations

    // Potential transfer between hierarchy levels
    void assign_potentials_by_hierarchy(Vector<double>& potentials, 
                                   int coarse_level, 
                                   int fine_level, 
                                   const Vector<double>& prev_potentials);

    // Helper methods
    std::vector<std::pair<std::string, std::string>> get_target_hierarchy_files() const;
    std::vector<std::string> get_mesh_hierarchy_files() const;
    void load_target_points_at_level(const std::string& points_file, 
                                   const std::string& density_file);

    /**
     * Save interpolated fields for source and target meshes.
     * This is useful for debugging and visualization purposes.
     * Uses the field names from the parameter files.
     */
    void save_interpolated_fields();
};

#endif

