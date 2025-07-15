#ifndef SOT_SOLVER_H
#define SOT_SOLVER_H

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <memory>
#include <map>
#include <mutex>
#include <atomic>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/optimization/solver_bfgs.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/utils/ColorDefinitions.h"
#include "SemiDiscreteOT/solvers/Distance.h"

using namespace dealii;

/**
 * @brief A solver for semi-discrete optimal transport problems.
 *
 * This class implements a solver for the dual formulation of the regularized semi-discrete optimal transport problem.
 *
 * @tparam dim The dimension of the source mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim=dim>
class SotSolver {
public:

    // Type definitions for RTree
    using IndexedPoint = std::pair<Point<spacedim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;

    /**
     * @brief A struct to hold all the necessary information about the source measure.
     */
    struct SourceMeasure {
        bool initialized = false; ///< Flag to check if source measure is set up
        SmartPointer<const DoFHandler<dim, spacedim>> dof_handler; ///< Pointer to the DoF handler for the source mesh.
        SmartPointer<const Mapping<dim, spacedim>> mapping; ///< Pointer to the mapping for the source mesh.
        SmartPointer<const FiniteElement<dim, spacedim>> fe; ///< Pointer to the finite element for the source mesh.
        SmartPointer<const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> density; ///< Pointer to the density vector of the source measure.
        unsigned int quadrature_order; ///< The order of the quadrature rule to use for integration.
        
        SourceMeasure() = default;
        /**
         * @brief Constructor for the SourceMeasure struct.
         */
        SourceMeasure(const DoFHandler<dim, spacedim>& dof_handler_,
                     const Mapping<dim, spacedim>& mapping_,
                     const FiniteElement<dim, spacedim>& fe_,
                     const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>& density_,
                     const unsigned int quadrature_order_)
            : initialized(true)
            , dof_handler(&dof_handler_)
            , mapping(&mapping_)
            , fe(&fe_)
            , density(&density_)
            , quadrature_order(quadrature_order_) 
        {}
    };

    /**
     * @brief A struct to hold all the necessary information about the target measure.
     */
    struct TargetMeasure {
        bool initialized = false; ///< Flag to check if target measure is set up
        std::vector<Point<spacedim>> points; ///< The points of the discrete target measure.
        Vector<double> density; ///< The weights of the discrete target measure.
        RTree rtree; ///< An R-tree for fast spatial queries on the target points.
        
        TargetMeasure() = default;
        /**
         * @brief Constructor for the TargetMeasure struct.
         */
        TargetMeasure(const std::vector<Point<spacedim>>& points_,
                     const Vector<double>& density_)
            : initialized(true)
            , points(points_)
            , density(density_) 
        {
            AssertThrow(points.size() == density.size(),
                       ExcDimensionMismatch(points.size(), density.size()));
            initialize_rtree();
        }

        /**
         * @brief Initializes the R-tree with the target points.
         */
        void initialize_rtree() {
            std::vector<IndexedPoint> indexed_points;
            indexed_points.reserve(points.size());
            for (std::size_t i = 0; i < points.size(); ++i) {
                indexed_points.emplace_back(points[i], i);
            }
            rtree = RTree(indexed_points.begin(), indexed_points.end());
        }
    };

    /**
     * @brief A struct to hold scratch data for parallel assembly.
     */
    struct ScratchData {
        ScratchData(const FiniteElement<dim, spacedim>& fe,
                   const Mapping<dim, spacedim>& mapping,
                   const Quadrature<dim>& quadrature)
            : fe_values(mapping, fe, quadrature,
                       update_values | update_quadrature_points | update_JxW_values)
            , density_values(quadrature.size()) {}

        ScratchData(const ScratchData& other)
            : fe_values(other.fe_values.get_mapping(),
                       other.fe_values.get_fe(),
                       other.fe_values.get_quadrature(),
                       update_values | update_quadrature_points | update_JxW_values)
            , density_values(other.density_values) {}

        FEValues<dim, spacedim> fe_values; ///< FEValues object for the current cell.
        std::vector<double> density_values; ///< The density values at the quadrature points of the current cell.
    };

    /**
     * @brief A struct to hold copy data for parallel assembly.
     */
    struct CopyData {
        double functional_value{0.0}; ///< The value of the functional on the current cell.
        Vector<double> gradient_values;  ///< The local contribution to the gradient.
        Vector<double> potential_values;  ///< The potential values at the target points.
        double local_C_sum = 0.0; ///< The sum of the scale terms for this cell.

        Vector<double> barycenters_values; ///< The barycenter values for the current cell.

        CopyData(const unsigned int n_target_points)
            : gradient_values(n_target_points),
              potential_values(n_target_points),
              barycenters_values(spacedim*n_target_points)
        {
            gradient_values = 0;  // Initialize local gradient to zero
            barycenters_values = 0;  // Initialize local barycenters to zero
        }
    };

    /**
     * @brief Constructor for the SotSolver.
     * @param comm The MPI communicator.
     */
    SotSolver(const MPI_Comm& comm);

    /**
     * @brief Sets up the source measure for the solver.
     * @param dof_handler The DoF handler for the source mesh.
     * @param mapping The mapping for the source mesh.
     * @param fe The finite element for the source mesh.
     * @param source_density The density of the source measure.
     * @param quadrature_order The order of the quadrature rule to use for integration.
     */
    void setup_source(const DoFHandler<dim, spacedim>& dof_handler,
                     const Mapping<dim, spacedim>& mapping,
                     const FiniteElement<dim, spacedim>& fe,
                     const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>& source_density,
                     const unsigned int quadrature_order);

    /**
     * @brief Sets up the target measure for the solver.
     * @param target_points The points of the discrete target measure.
     * @param target_density The weights of the discrete target measure.
     */
    void setup_target(const std::vector<Point<spacedim>>& target_points,
                     const Vector<double>& target_density);

    /**
     * @brief Configures the solver with the given parameters.
     * @param params The solver parameters.
     */
    void configure(const SotParameterManager::SolverParameters& params);

    /**
     * @brief Solves the optimal transport problem.
     * @param potential The vector to store the computed optimal transport potential.
     * @param params The solver parameters.
     */
    void solve(Vector<double>& potential,
              const SotParameterManager::SolverParameters& params);

    /**
     * @brief Alternative solve interface if measures are not set up beforehand.
     * @param potential The vector to store the computed optimal transport potential.
     * @param source The source measure.
     * @param target The target measure.
     * @param params The solver parameters.
     */
    void solve(Vector<double>& potential,
              const SourceMeasure& source,
              const TargetMeasure& target,
              const SotParameterManager::SolverParameters& params);
    
    /**
     * @brief Evaluates the weighted barycenters of the power cells.
     * @param potentials The optimal transport potentials.
     * @param barycenters_out The vector to store the computed barycenters.
     * @param params The solver parameters.
     */
    void evaluate_weighted_barycenters(
        const Vector<double>& potentials,
        std::vector<Point<spacedim>>& barycenters_out,
        const SotParameterManager::SolverParameters& params);
    
    /**
     * @brief Returns the value of the functional at the last iteration.
     */
    double get_last_functional_value() const { return global_functional; }
    /**
     * @brief Returns the number of iterations of the last solve.
     */
    unsigned int get_last_iteration_count() const;
    /**
     * @brief Returns the convergence status of the last solve.
     */
    bool get_convergence_status() const;
    /**
     * @brief Returns the distance threshold used in the last solve.
     */
    double get_last_distance_threshold() const { return current_distance_threshold; }
    /**
     * @brief Returns the global C value.
     */
    double get_C_global() const { return C_global; }
    
    /**
     * @brief Sets the distance threshold for the solver.
     * @param threshold The distance threshold.
     */
    void set_distance_threshold(double threshold) { current_distance_threshold = threshold; }

    /**
     * @brief Computes the covering radius of the target measure with respect to the source domain
     * 
     * The covering radius R0 is defined as:
     * R0 = max_{x∈Ω} min_{1≤j≤N} ||x - y_j||
     * 
     * which represents the maximum distance any point in the source domain
     * needs to travel to reach the nearest target point.
     * 
     * @return The covering radius value
     */
    double compute_covering_radius() const;

    /**
     * @brief Computes the geometric radius bound for truncating quadrature rules
     * 
     * The geometric radius bound R_geom is defined as:
     * R_geom^2 ≥ R_0^2 + 2Γ(ψ) + 2ε ln(ε/(ν_min * τ * |J_ε(ψ)|))
     * 
     * where:
     * - R_0 is the covering radius
     * - Γ(ψ) = M-m is the potential range (max - min)
     * - ε is the regularization parameter
     * - τ is the tolerance parameter
     * - ν_min is the minimum target density
     * - J_ε(ψ) is the functional value
     * 
     * @param potentials Current potential values
     * @param epsilon Regularization parameter
     * @param tolerance Desired tolerance for truncation error
     * @return The geometric radius bound
     */
    double compute_geometric_radius_bound(
        const Vector<double>& potentials,
        const double epsilon,
        const double tolerance) const;
    
    /**
     * @brief Sets the distance function to be used by the solver.
     * @param distance_name The name of the distance function.
     */
    void set_distance_function(const std::string &distance_name);

    /**
     * @brief Computes the conditional density of the source measure given a potential.
     * @param dof_handler The DoF handler for the source mesh.
     * @param mapping The mapping for the source mesh.
     * @param potential The optimal transport potential.
     * @param potential_indices The indices of the potential to use.
     * @param conditioned_densities The vector to store the computed conditional densities.
     */
    void get_potential_conditioned_density(
        const DoFHandler<dim, spacedim> &dof_handler,
        const Mapping<dim, spacedim> &mapping,
        const Vector<double> &potential,
        const std::vector<unsigned int> &potential_indices,
        std::vector<LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> &conditioned_densities);

    // Distance function
    std::string distance_name; ///< The name of the distance function.
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function; ///< The distance function.
    std::function<Vector<double>(const Point<spacedim>&, const Point<spacedim>&)> distance_function_gradient; ///< The gradient of the distance function.
    std::function<Point<spacedim>(const Point<spacedim>&, const Vector<double>&)> distance_function_exponential_map; ///< The exponential map of the distance function.

    /**
     * @brief Evaluates the dual functional and its gradient.
     * @param potential The potential at which to evaluate the functional.
     * @param gradient_out The vector to store the computed gradient.
     * @return The value of the functional.
     */
    double evaluate_functional(const Vector<double>& potential,
                             Vector<double>& gradient_out);

    /**
     * @brief Computes the Hessian matrix of the dual functional.
     * @param potential The potential at which to compute the Hessian.
     * @param hessian_out The matrix to store the computed Hessian.
     */
    void compute_hessian(const Vector<double>& potential,
                        LAPACKFullMatrix<double>& hessian_out);

    // Source and target measures
    SourceMeasure source_measure; ///< The source measure.
    TargetMeasure target_measure; ///< The target measure.

private:

    /**
     * @brief A verbose solver control class that prints the progress of the solver.
     */
    class VerboseSolverControl : public SolverControl
    {
    public:
        VerboseSolverControl(unsigned int n, double tol, bool use_componentwise, ConditionalOStream& pcout_)
            : SolverControl(n, tol)
            , pcout(pcout_)
            , use_componentwise_check(use_componentwise)
            , gradient(nullptr)
            , target_measure(nullptr)
            , user_tolerance_for_componentwise(1.0) 
        {}

        void set_gradient(const Vector<double>& grad) {
            gradient = &grad;
        }

        void set_target_measure(const Vector<double>& target_density, double user_tolerance) {
            AssertThrow(use_componentwise_check, ExcMessage("Target measure only needed for component-wise check"));
            target_measure = &target_density;
            user_tolerance_for_componentwise = user_tolerance;
        }


        virtual State check(unsigned int step, double value) override
        {
            AssertThrow(gradient != nullptr,
                        ExcMessage("Gradient vector not set in VerboseSolverControl"));

            double check_value = 0.0;
            std::string check_description;
            std::string color;

            if (use_componentwise_check)
            {
                AssertThrow(target_measure != nullptr,
                            ExcMessage("Target measure not set for component-wise check"));
                AssertThrow(gradient->size() == target_measure->size(),
                            ExcDimensionMismatch(gradient->size(), target_measure->size()));

                double max_scaled_residual = -std::numeric_limits<double>::infinity();
                for (unsigned int j = 0; j < gradient->size(); ++j) {
                    double scaled_residual = std::abs((*gradient)[j]) - (*target_measure)[j] * user_tolerance_for_componentwise;
                    max_scaled_residual = std::max(max_scaled_residual, scaled_residual);
                }
                check_value = max_scaled_residual; 

                if (check_value < tolerance()) { 
                    color = Color::green;
                } else if (check_value < 0) { 
                     color = Color::yellow;
                } else { 
                    color = Color::red;
                }

                check_description = "Max Scaled Residual (max |g_i| - T_i*tol): ";
                pcout << "Iteration " << CYAN << step << RESET
                      << " - L-2 gradient norm: " << color << value << RESET // value is L2 norm from BFGS
                      << " - " << check_description << color << check_value << RESET << std::endl;

            }
            else // Use L1-norm check
            {
                check_value = gradient->l1_norm(); 

                double rel_residual = (step == 0 || initial_l1_norm == 0.0) ?
                                      check_value : check_value / initial_l1_norm;

                if (step == 0)
                    initial_l1_norm = check_value;

                if (check_value < tolerance()) { 
                    color = Color::green;  
                } else if (rel_residual < 0.5) {
                    color = Color::yellow;  
                } else {
                    color = Color::red; 
                }

                check_description = "L-1 gradient norm: ";
                 pcout << "Iteration " << CYAN << step << RESET
                      << " - L-2 gradient norm: " << color << value << RESET 
                      << " - " << check_description << color << check_value << RESET
                      << " - Relative L-1 residual: " << color << rel_residual << RESET << std::endl;
            }

            last_check_value = check_value; 
            return SolverControl::check(step, check_value);
        }

        double get_last_check_value() const { return last_check_value; }

    private:
        ConditionalOStream& pcout;
        bool use_componentwise_check;
        double initial_l1_norm = 1.0; 
        const Vector<double>* gradient;
        const Vector<double>* target_measure; 
        double user_tolerance_for_componentwise; 
        double last_check_value = 0.0;
    };

    // Local assembly methods        
    void local_assemble(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
        ScratchData& scratch,
        CopyData& copy,
        std::function<void(CopyData&,
                           const Point<spacedim>&,
                           const std::vector<std::size_t>&,
                           const std::vector<double>&,
                           const std::vector<double>&,
                           const double&,
                           const double&,
                           const double&,
                           const double&,
                           const double&)> function_call);
    
    // Distance threshold and caching methods
    void compute_distance_threshold() const;
    std::vector<std::size_t> find_nearest_target_points(const Point<spacedim>& query_point) const;
    double compute_integral_radius_bound(
        const Vector<double>& potentials,
        double epsilon,
        double tolerance,
        double C_value,
        double current_functional_val) const;

    // Validation methods
    bool validate_measures() const;

    // Barycenters computation methods
    void compute_weighted_barycenters_non_euclidean(
        const Vector<double>& potentials,
        std::vector<Vector<double>>& barycenters_gradients_out,
        std::vector<Point<spacedim>>& barycenters_out
    );
    void compute_weighted_barycenters_euclidean(
        const Vector<double>& potentials,
        std::vector<Point<spacedim>>& barycenters_out);

    // MPI and parallel related members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Solver state
    std::unique_ptr<SolverControl> solver_control;
    mutable double current_distance_threshold;
    mutable double effective_distance_threshold;
    const Vector<double>* current_potential;
    double current_epsilon;
    mutable double global_functional;
    Vector<double> gradient;  
    double covering_radius;           
    double min_target_density;       
    double C_global = 0.0; // Sum of all scale terms

    // Current solver parameters
    SotParameterManager::SolverParameters current_params;

    // weighted truncated barycenters evaluation

    // Barycenters evaluation data
    Vector<double> barycenters;
    Vector<double> barycenters_gradients;

    // Barycenters points and gradients
    std::vector<Point<spacedim>> barycenters_points;
    std::vector<Vector<double>> barycenters_grads;
};

#endif // SOT_SOLVER_H
