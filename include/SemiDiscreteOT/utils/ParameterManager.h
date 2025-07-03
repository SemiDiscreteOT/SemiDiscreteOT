#ifndef PARAMETER_MANAGER_H
#define PARAMETER_MANAGER_H

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <string>

// ANSI color codes for output formatting
#ifndef RESET
#define RESET   "\033[0m"
#define CYAN    "\033[36m"
#define BLUE    "\033[34m"
#define GREEN   "\033[32m"
#define MAGENTA "\033[35m"
#define YELLOW  "\033[33m"
#define RED     "\033[31m"
#define BOLD    "\033[1m"
#endif

using namespace dealii;

/**
 * A class to manage all parameters for the RSOT solver.
 * This class handles parameter declaration, storage, and access for all components
 * of the solver, including mesh generation, solver settings, and multilevel methods.
 * It inherits from ParameterAcceptor to integrate with deal.II's parameter handling system.
 */
class SotParameterManager : public ParameterAcceptor
{
public:
    /**
     * Constructor.
     * @param comm MPI communicator for parallel execution
     */
    SotParameterManager(const MPI_Comm &comm);

    /**
     * Parameters for mesh generation.
     * Controls both source and target mesh generation settings.
     */
    struct MeshParameters {
        unsigned int n_refinements = 0;          ///< Number of global refinement steps
        std::string grid_generator_function;     ///< Name of the grid generator function
        std::string grid_generator_arguments;    ///< Arguments for the grid generator
        bool use_tetrahedral_mesh = false;      ///< Whether to use tetrahedral elements (3D only)
        bool use_custom_density = false;        ///< Whether to use custom density
        std::string density_file_path;          ///< Path to the density file
        std::string density_file_format = "vtk"; ///< Format of the density file (vtk/h5)
        std::string density_field_name = "normalized_density"; ///< Name of the field in the VTK file
    };

    /**
     * Parameters controlling the RSOT solver behavior.
     */
    struct SolverParameters {
        unsigned int max_iterations = 1000;      ///< Maximum number of solver iterations
        double tolerance = 1;                    ///< Convergence tolerance
        std::string solver_control_type = "l1norm"; ///< Type of solver control to use (l1norm/componentwise)
        double epsilon = 1e-3;      ///< Entropy regularization parameter
        double tau = 1e-8;                       ///< Integral radius bound tolerance
        std::string distance_threshold_type = "pointwise"; ///< Type of distance threshold bound (pointwise|integral|geometric)
        bool verbose_output = true;              ///< Enable detailed solver output
        std::string solver_type = "BFGS";        ///< Type of optimization solver
        unsigned int quadrature_order = 3;       ///< Order of quadrature formula
        unsigned int nb_points = 1000;           ///< Number of points for discretization
        unsigned int n_threads = 0;              ///< Number of threads (0 = auto)
        bool use_epsilon_scaling = false;        ///< Enable epsilon scaling strategy
        double epsilon_scaling_factor = 2.0;     ///< Factor for epsilon reduction
        unsigned int epsilon_scaling_steps = 5;  ///< Number of scaling steps
        bool use_log_sum_exp_trick = false;     ///< Enable log-sum-exp trick for numerical stability with small entropy
    };

    /**
     * Parameters for multilevel approach.
     * Controls both source mesh and target point cloud hierarchies.
     */
    struct MultilevelParameters {
        // Source mesh hierarchy parameters
        bool source_enabled = true;             ///< Whether to use source multilevel approach
        int source_min_vertices = 1000;         ///< Minimum vertices for coarsest source level
        int source_max_vertices = 10000;        ///< Maximum vertices for finest source level
        std::string source_hierarchy_dir = "output/data_multilevel/source_multilevel";  ///< Source hierarchy directory
        
        // Target point cloud hierarchy parameters
        bool target_enabled = false;            ///< Whether to use target multilevel approach
        int target_min_points = 100;            ///< Minimum points for coarsest target level
        int target_max_points = 1000;           ///< Maximum points for finest target level
        std::string target_hierarchy_dir = "output/data_multilevel/target_multilevel";  ///< Target hierarchy directory
        bool use_softmax_potential_transfer = true;///< Use softmax for potential transfer between target levels
        
        // Python clustering parameters
        bool use_python_clustering = false;     ///< Whether to use Python scripts for clustering
        std::string python_script_name = "multilevel_clustering_scipy.py"; ///< Name of the Python script to use
        
        // Common parameters
        std::string output_prefix = "output/multilevel/sot";  ///< Output directory prefix
    };

    /**
     * Parameters for power diagram computation.
     */
    struct PowerDiagramParameters {
        std::string implementation = "dealii";   ///< Implementation choice (dealii/geogram)
    };

    /**
     * Parameters for transport map computation.
     */
    struct TransportMapParameters {
        double truncation_radius = -1.0;        ///< Truncation radius for map approximation (-1 = disabled)
    };

    // Const access to parameters through getters
    const MeshParameters& get_source_params() const { return source_params; }
    const MeshParameters& get_target_params() const { return target_params; }
    const SolverParameters& get_solver_params() const { return solver_params; }
    const MultilevelParameters& get_multilevel_params() const { return multilevel_params; }
    const PowerDiagramParameters& get_power_diagram_params() const { return power_diagram_params; }
    const TransportMapParameters& get_transport_map_params() const { return transport_map_params; }

    // Direct access to parameters through references
    MeshParameters& source_params;
    MeshParameters& target_params;
    SolverParameters& solver_params;
    MultilevelParameters& multilevel_params;
    PowerDiagramParameters& power_diagram_params;
    TransportMapParameters& transport_map_params;
    std::string& selected_task;
    std::string& io_coding;

    // MPI-related getters
    const MPI_Comm& get_mpi_communicator() const { return mpi_communicator; }
    const ConditionalOStream& get_pcout() const { return pcout; }

    /**
     * Print all relevant parameters based on the selected task.
     * This provides a comprehensive view of the current parameter settings.
     */
    void print_logo() const;
    virtual void print_parameters() const;
protected:
    // MPI members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Helper methods for printing specific parameter groups
    void print_mesh_parameters() const;
    void print_solver_parameters() const;
    void print_multilevel_parameters() const;
    void print_power_diagram_parameters() const;
    void print_transport_map_parameters() const;
    void print_task_information() const;
    void print_section_header(const std::string& section_name) const;
    
private:

    // Storage for parameters
    std::string selected_task_storage;
    std::string io_coding_storage = "txt";
    MeshParameters source_params_storage;
    MeshParameters target_params_storage;
    SolverParameters solver_params_storage;
    MultilevelParameters multilevel_params_storage;
    PowerDiagramParameters power_diagram_params_storage;
    TransportMapParameters transport_map_params_storage;
};

/**
 * A class to manage all parameters for the RSOT solver.
 * This class handles parameter declaration, storage, and access for all components
 * of the solver, including mesh generation, solver settings, and multilevel methods.
 * It inherits from ParameterAcceptor to integrate with deal.II's parameter handling system.
 */
 class LloydParameterManager : public SotParameterManager
 {
 public:
    /**
    * Constructor.
    * @param comm MPI communicator for parallel execution
    */
    LloydParameterManager(const MPI_Comm &comm);

    /**
    * Parameters for Lloyd algorithm.
    */
    struct LloydParameters {
        unsigned int max_iterations = 1000;          ///< Number of max iterations
        double relative_tolerance = 1e-8;                     ///< Convergence tolerance
    };
    
    const LloydParameters& get_lloyd_params() const { return lloyd_params; }

    // Direct access to parameters through references
    LloydParameters& lloyd_params;
    
    /**
     * Print all relevant parameters based on the selected task.
     * This provides a comprehensive view of the current parameter settings.
     */
    virtual void print_parameters() const override;
 private:
    // Helper methods for printing specific parameter groups
    void print_lloyd_parameters() const;
    void print_task_information() const;
    
    // Storage for parameters
    LloydParameters lloyd_params_storage;
 };

#endif 