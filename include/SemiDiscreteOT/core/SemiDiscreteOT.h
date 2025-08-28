#ifndef RSOT_H
#define RSOT_H

#include <filesystem>
#include <memory>
#include <mutex>
#include <atomic>
#include <boost/geometry/strategies/disjoint.hpp>
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

/**
 * @brief Main class for the semi-discrete optimal transport solver.
 *
 * This class orchestrates the entire process of solving a semi-discrete
 * optimal transport problem. It manages the source and target measures,
 * the solver, and the various numerical strategies that can be employed.
 *
 * @tparam dim The dimension of the source mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim = dim>
class SemiDiscreteOT {
public:
    /**
     * @brief Constructor for the SemiDiscreteOT class.
     * @param mpi_communicator The MPI communicator.
     */
    SemiDiscreteOT(const MPI_Comm &mpi_communicator);
    /**
     * @brief Runs the solver with the current configuration.
     */
    void run();

    /**
     * @brief Configure the solver parameters programmatically.
     * @param config_func A lambda function that takes a reference to the
     *                    SotParameterManager and modifies its parameters.
     */
    void configure(std::function<void(SotParameterManager&)> config_func);


    /**
     * @brief Setup source mesh from standard deal.II objects (simplified API for tutorials). Shared mesh is loaded from file, then `setup_source_measure` is called to setup the source measure from a shared `Vector<double>` or a distributed `LinearAlgebra::distributed::Vector<double>`.
     * @param tria A standard Triangulation
     * @param name An optional name for the source mesh (used for saving and hierarchy generation)
     * 
     * This method internally handles the conversion to distributed objects when needed for parallel computation.
     */
    void setup_source_mesh(
        Triangulation<dim, spacedim>& tria,
        const std::string& name = "source");

    /**
     * @brief Sets up the source measure for the semi-discrete optimal transport problem.
     *
     * This function initializes the source measure using the provided degrees of freedom
     * handler and density vector. The source measure represents the distribution of mass
     * in the source domain. The vector `density` bust me initialized with this class' `dof_handler_source`.
     *
     * @tparam dim The spatial dimension of the problem.
     * @tparam spacedim The dimension of the embedding space.
     * 
     * @param dh The degrees of freedom handler that describes the finite element space
     *           for the source domain.
     * @param density A serial vector representing the density values of the source
     *                measure at the degrees of freedom.
     */
    void setup_source_measure(
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
     * @brief Pre-computes the multilevel hierarchy for the source.
     * @param source_level The level of the source hierarchy to prepare.
     */
    void prepare_source_multilevel();

    /**     
     * @brief Pre-computes the multilevel hierarchy for the target.
     * @param target_level The level of the target hierarchy to prepare.
     */
    void prepare_target_multilevel();

    /**
     * @brief Get a pointer to the solver object.
     * @return Pointer to the solver object.
     */
    SotSolver<dim, spacedim> *get_solver() { return sot_solver.get(); }

    /**
     * @brief Get a reference to the solver parameters.
     * @return Reference to the solver parameters.
     */
    const SotParameterManager::SolverParameters &get_solver_params() const { return solver_params; }

    /**
     * @brief Get the coarsest potential from the multilevel solve.
     * @return Reference to the coarsest potential vector.
     */
    const Vector<double> &get_coarsest_potential() const { return coarsest_potential; }
    


    /**
     * @brief Run the optimal transport computation based on the current configuration.
     *        This method handles single-level, multilevel, and epsilon scaling automatically.
     * @param initial_potential Optional initial potential values to start the optimization from.
     * @return A vector containing the computed optimal transport potentials for the target points.
     */
    Vector<double> solve(const Vector<double>& initial_potential = Vector<double>());

    /**
     * @brief Saves the discrete source and target measures to files.
     */
    void save_discrete_measures();
    
    /**
     * @brief Sets the distance function to be used by the solver.
     * @param dist The distance function.
     */
    void set_distance_function(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& dist)
        {
            sot_solver->distance_function = dist;
        }
        
        ConditionalOStream pcout; ///< A conditional output stream for parallel printing.

    LinearAlgebra::distributed::Vector<double> source_density; ///< The source density.
    Vector<double> target_density; ///< The target density.
    std::vector<Point<spacedim>> target_points; ///< The target points.
    std::vector<Point<spacedim>> source_points; ///< The source points.

    // Mesh and DoF handler members
    parallel::fullydistributed::Triangulation<dim, spacedim> source_mesh; ///< The source mesh.
    Triangulation<dim, spacedim> target_mesh; ///< The target mesh.
    
    DoFHandler<dim, spacedim> dof_handler_source; ///< The DoF handler for the source mesh.
    DoFHandler<dim, spacedim> dof_handler_target; ///< The DoF handler for the target mesh.
    IndexSet source_fine_loc_owned_dofs;
    IndexSet source_fine_loc_relevant_dofs;

    // Solver member
    std::unique_ptr<SotSolver<dim, spacedim>> sot_solver; ///< The semi-discrete optimal transport solver.
protected:
    // MPI-related members
    MPI_Comm mpi_communicator; ///< The MPI communicator.
    const unsigned int n_mpi_processes; ///< The number of MPI processes.
    const unsigned int this_mpi_process; ///< The rank of the current MPI process.

    // Parameter manager and references
    SotParameterManager param_manager; ///< The parameter manager.
    SotParameterManager::MeshParameters& source_params; ///< A reference to the source mesh parameters.
    SotParameterManager::MeshParameters& target_params; ///< A reference to the target mesh parameters.
    SotParameterManager::SolverParameters& solver_params; ///< A reference to the solver parameters.
    SotParameterManager::MultilevelParameters& multilevel_params; ///< A reference to the multilevel parameters.
    SotParameterManager::PowerDiagramParameters& power_diagram_params; ///< A reference to the power diagram parameters.
    SotParameterManager::TransportMapParameters& transport_map_params; ///< A reference to the transport map parameters.
    std::string& selected_task; ///< A reference to the selected task.
    std::string& io_coding; ///< A reference to the I/O coding.

    std::unique_ptr<Triangulation<dim, spacedim>> initial_fine_tria; ///< The initial fine triangulation.
    std::unique_ptr<DoFHandler<dim, spacedim>> initial_fine_dof_handler; ///< The initial fine DoF handler.
    std::unique_ptr<Vector<double>> initial_fine_density; ///< The initial fine density.
    bool is_setup_programmatically_ = false; ///< A flag to indicate if the setup is done programmatically.

    // Source mesh name for saving and hierarchy generation
    std::string source_mesh_name = "source"; ///< The name of the source mesh.

    std::unique_ptr<VTKHandler<dim,spacedim>> source_vtk_handler; ///< The VTK handler for the source mesh.
    DoFHandler<dim,spacedim> vtk_dof_handler_source; ///< The DoF handler for the source VTK mesh.
    Vector<double> vtk_field_source; ///< The source field from the VTK file.
    Triangulation<dim,spacedim> vtk_tria_source; ///< The triangulation from the source VTK file.
    // Finite element and mapping members
    std::unique_ptr<FiniteElement<dim, spacedim>> fe_system; ///< The finite element system.
    std::unique_ptr<Mapping<dim, spacedim>> mapping; ///< The mapping.
    std::unique_ptr<FiniteElement<dim, spacedim>> fe_system_target; ///< The target finite element system.
    std::unique_ptr<Mapping<dim, spacedim>> mapping_target; ///< The target mapping.

    // Mesh manager
    std::unique_ptr<MeshManager<dim, spacedim>> mesh_manager; ///< The mesh manager.
    
    // Epsilon scaling handler
    std::unique_ptr<EpsilonScalingHandler> epsilon_scaling_handler; ///< The epsilon scaling handler.
    
    /**
     * @brief Saves the results of the computation.
     * @param potentials The computed optimal transport potentials.
     * @param filename The name of the file to save the results to.
     * @param add_epsilon_prefix Whether to add an epsilon prefix to the filename.
     */
    void save_results(const Vector<double>& potentials, const std::string& filename, bool add_epsilon_prefix = true);
    
    /**
     * @brief Normalizes the density vector.
     * @param density The density vector to normalize.
     */
    void normalize_density(LinearAlgebra::distributed::Vector<double>& density);
private:

    // Core functionality methods
    /**
     * @brief Generates the source and target meshes.
     */
    void mesh_generation();
    /**
     * @brief Loads the source and target meshes from files.
     */
    void load_meshes();
    
    /**
     * @brief Run single-level SOT computation.
     * @param initial_potential Optional initial potential values to start the optimization from.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_sot(const Vector<double>& initial_potential = Vector<double>());
    
    /**
     * @brief Computes the power diagram of the target points.
     */
    void compute_power_diagram();
    /**
     * @brief Computes the transport map from the source to the target measure.
     */
    void compute_transport_map();
    /**
     * @brief Computes the conditional density of the source measure.
     */
    void compute_conditional_density();


    
    /**
     * @brief Run multilevel SOT computation (dispatcher method).
     * @param initial_potential Optional initial potential values to start the optimization from.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_multilevel(const Vector<double>& initial_potential = Vector<double>());
    
    /**
     * @brief Run combined source and target multilevel SOT computation.
     * @param initial_potential Optional initial potential values to start the optimization from.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_combined_multilevel(const Vector<double>& initial_potential = Vector<double>());
    
    /**
     * @brief Run source-only multilevel SOT computation.
     * @param initial_potential Optional initial potential values to start the optimization from.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_source_multilevel(const Vector<double>& initial_potential = Vector<double>());
    
    /**
     * @brief Run target-only multilevel SOT computation.
     * @param initial_potential Optional initial potential values to start the optimization from.
     * @return Vector containing the optimal transport potentials for the target points.
     */
    Vector<double> run_target_multilevel(const Vector<double>& initial_potential = Vector<double>());

    // Setup methods
    /**
     * @brief Sets up the finite elements for the source mesh.
     * @param is_multilevel Whether the setup is for a multilevel computation.
     */
    void setup_source_finite_elements(bool is_multilevel = false);
    /**
     * @brief Sets up the finite elements for the target mesh.
     */
    void setup_target_finite_elements();
    /**
     * @brief Sets up the finite elements for both the source and target meshes.
     */
    void setup_finite_elements();
    /**
     * @brief Sets up the target points.
     */
    void setup_target_points();
    /**
     * @brief Sets up the finite elements for a multilevel computation.
     */
    void setup_multilevel_finite_elements();

    // Exact SOT method (3D only)
    /**
     * @brief Runs the exact semi-discrete optimal transport solver.
     */
    template <int d = dim, int s = spacedim>
    typename std::enable_if<d == 3 && s == 3>::type run_exact_sot();    

    // Hierarchy-related members
    std::vector<std::vector<std::vector<size_t>>> child_indices_; ///< The child indices for the multilevel hierarchy.
    bool has_hierarchy_data_{false}; ///< A flag to indicate if the hierarchy data is loaded.
    /**
     * @brief Loads the hierarchy data from a directory.
     * @param hierarchy_dir The directory containing the hierarchy data.
     * @param specific_level The specific level to load.
     */
    void load_hierarchy_data(const std::string& hierarchy_dir, int specific_level = -1);

    // Multilevel computation state
    std::vector<Point<spacedim>> target_points_coarse;  ///< The coarse level target points.
    Vector<double> target_density_coarse;          ///< The coarse level target densities.
    mutable double current_distance_threshold{0.0}; ///< The current distance threshold for computations.
    Vector<double> coarsest_potential;             ///< The coarsest level potential for the multilevel solve.

    // Potential transfer between hierarchy levels
    /**
     * @brief Assigns potentials by hierarchy.
     * @param potentials The potentials to assign.
     * @param coarse_level The coarse level.
     * @param fine_level The fine level.
     * @param prev_potentials The previous potentials.
     */
    void assign_potentials_by_hierarchy(Vector<double>& potentials, 
                                   int coarse_level, 
                                   int fine_level, 
                                   const Vector<double>& prev_potentials);

    // Helper methods
    /**
     * @brief Gets the target hierarchy files.
     * @return A vector of pairs of strings, where each pair contains the path to the points file and the density file.
     */
    std::vector<std::pair<std::string, std::string>> get_target_hierarchy_files() const;
    /**
     * @brief Gets the mesh hierarchy files.
     * @return A vector of strings, where each string is the path to a mesh file.
     */
    std::vector<std::string> get_mesh_hierarchy_files() const;
    /**
     * @brief Loads the target points at a specific level.
     * @param points_file The path to the points file.
     * @param density_file The path to the density file.
     */
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

