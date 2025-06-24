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

#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/utils/ColorDefinitions.h"
#include "SemiDiscreteOT/solvers/Distance.h"

using namespace dealii;

template <int dim, int spacedim=dim>
class SotSolver {
public:

    // Type definitions for RTree
    using IndexedPoint = std::pair<Point<spacedim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;

    // Source measure data structure
    struct SourceMeasure {
        bool initialized = false; // Flag to check if source measure is set up
        SmartPointer<const DoFHandler<dim, spacedim>> dof_handler;
        SmartPointer<const Mapping<dim, spacedim>> mapping;
        SmartPointer<const FiniteElement<dim, spacedim>> fe;
        SmartPointer<const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> density;
        unsigned int quadrature_order;
        
        SourceMeasure() = default;
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

    // Target measure data structure
    struct TargetMeasure {
        bool initialized = false; // Flag to check if target measure is set up
        std::vector<Point<spacedim>> points;
        Vector<double> density;
        RTree rtree;
        
        TargetMeasure() = default;
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

        void initialize_rtree() {
            std::vector<IndexedPoint> indexed_points;
            indexed_points.reserve(points.size());
            for (std::size_t i = 0; i < points.size(); ++i) {
                indexed_points.emplace_back(points[i], i);
            }
            rtree = RTree(indexed_points.begin(), indexed_points.end());
        }
    };

    // Per-cell scratch data for parallel assembly
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

        FEValues<dim, spacedim> fe_values;
        std::vector<double> density_values;
    };

    // Per-cell copy data for parallel assembly
    struct CopyData {
        double functional_value{0.0};
        Vector<double> gradient_values;  // Local gradient contribution
        Vector<double> potential_values;  // For softmax refinement
        double local_C_sum = 0.0; // Sum of scale terms for this cell

        CopyData(const unsigned int n_target_points)
            : gradient_values(n_target_points),
              potential_values(n_target_points) 
        {
            gradient_values = 0;  // Initialize local gradient to zero
        }
    };

    // Copy data barycenters evaluation
    struct CopyDataBarycenters {
        Vector<double> barycenters_values;  // Local barycenters contribution

        CopyDataBarycenters(const unsigned int n_target_points)
            : barycenters_values(spacedim*n_target_points)
        {
            barycenters_values = 0;  // Initialize local barycenters to zero
        }
    };

    struct CopyDataWassersteinBarycenters {
        Vector<double> grad_support_points;
        Vector<double> grad_measure;

        CopyDataWassersteinBarycenters(const unsigned int n_target_points)
            : grad_support_points(spacedim*n_target_points),
            grad_measure(n_target_points)
        {
            grad_support_points = 0;
            grad_measure = 0;
        }
    };

    // Constructor
    SotSolver(const MPI_Comm& comm);

    // Setup methods
    void setup_source(const DoFHandler<dim, spacedim>& dof_handler,
                     const Mapping<dim, spacedim>& mapping,
                     const FiniteElement<dim, spacedim>& fe,
                     const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>& source_density,
                     const unsigned int quadrature_order);

    void setup_target(const std::vector<Point<spacedim>>& target_points,
                     const Vector<double>& target_density);

    // Main solver interface
    void solve(Vector<double>& potential,
              const SotParameterManager::SolverParameters& params);

    // Alternative solve interface if measures not set up beforehand
    void solve(Vector<double>& potential,
              const SourceMeasure& source,
              const TargetMeasure& target,
              const SotParameterManager::SolverParameters& params);
    
    void evaluate_weighted_barycenters(
        const Vector<double>& potentials,
        std::vector<Point<spacedim>>& barycenters_out,
        const SotParameterManager::SolverParameters& params);
    
    // Getters for solver results
    double get_last_functional_value() const { return global_functional; }
    unsigned int get_last_iteration_count() const;
    bool get_convergence_status() const;
    double get_cache_size_mb() const;
    double get_last_distance_threshold() const { return current_distance_threshold; }
    bool get_cache_limit_reached() const { return cache_limit_reached; }
    double get_C_global() const { return C_global; }

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
    
    // distance function methods
    void set_distance_function(const std::string &distance_name);

    // Distance function
    std::string distance_name;
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function;
    std::function<Vector<double>(const Point<spacedim>&, const Point<spacedim>&)> distance_function_gradient;
    std::function<Point<spacedim>(const Point<spacedim>&, const Vector<double>&)> distance_function_exponential_map;

    // Core evaluation method
    double evaluate_functional(const Vector<double>& potential,
                             Vector<double>& gradient_out);

    // Compute Hessian matrix for Newton solver
    void compute_hessian(const Vector<double>& potential,
                        LAPACKFullMatrix<double>& hessian_out);
       
    // Compute gradients with respect to target support points and measure
    void compute_grad_target(
        std::vector<std::pair<Vector<double>, double>> &target_gradients_out,
        const Vector<double>& potentials,
        const std::vector<std::pair<Point<spacedim>, double>> &target,
        const SotParameterManager::SolverParameters& params
    );

    // Source and target measures
    SourceMeasure source_measure;
    TargetMeasure target_measure;
    
private:

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

    // Cache for local assembly computations
    struct CellCache {
        std::vector<std::size_t> target_indices;
        std::vector<double> precomputed_distance_terms;
        bool is_valid;

        CellCache() : is_valid(false) {}
    };

    // Local assembly methods
    void local_assemble(const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
    ScratchData& scratch,
    CopyData& copy);

    // Local assembly methods for barycenters
    void local_assemble_barycenters(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
        ScratchData& scratch,
        CopyDataBarycenters& copy);
    
    void local_assemble_grad_target(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
        ScratchData& scratch,
        CopyDataWassersteinBarycenters& copy,
        const std::vector<std::pair<Point<spacedim>, double>>& current_barycenters);
    
    // Distance threshold and caching methods
    void compute_distance_threshold() const;
    void reset_distance_threshold_cache() const;
    std::vector<std::size_t> find_nearest_target_points(const Point<spacedim>& query_point) const;
    double estimate_cache_entry_size_mb(const std::vector<std::size_t>& target_indices, 
                                      unsigned int n_q_points) const;
    double compute_integral_radius_bound(
        const Vector<double>& potentials,
        double lambda,
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
    void local_assemble_barycenters_non_euclidean(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
        ScratchData& scratch,
        CopyDataBarycenters& copy,
        std::vector<Point<spacedim>>& barycenters_out);

    void compute_weighted_barycenters_euclidean(
        const Vector<double>& potentials,
        std::vector<Point<spacedim>>& barycenters_out);
    void local_assemble_barycenters_euclidean(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
        ScratchData& scratch,
        CopyDataBarycenters& copy);

    // MPI and parallel related members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Solver state
    std::unique_ptr<SolverControl> solver_control;
    mutable double current_distance_threshold;
    mutable double effective_distance_threshold;
    mutable bool is_caching_active;
    const Vector<double>* current_potential;
    double current_lambda;
    mutable double global_functional;
    Vector<double> gradient;  
    double covering_radius;           
    double min_target_density;       
    double C_global = 0.0; // Sum of all scale terms

    mutable std::unordered_map<std::string, CellCache> cell_caches;
    mutable std::mutex cache_mutex;
    std::atomic<size_t> total_target_points{0};
    mutable std::atomic<double> current_cache_size_mb{0.0};   ///< Current cache size in MB
    mutable bool cache_limit_reached{false};                  ///< Flag indicating if cache limit was reached

    // Current solver parameters
    SotParameterManager::SolverParameters current_params;

    // Barycenters evaluation data
    Vector<double> barycenters;
    Vector<double> barycenters_gradients;

    // Barycenters points and gradients
    std::vector<Point<spacedim>> barycenters_points;
    std::vector<Vector<double>> barycenters_grads;

    // Data for Wasserstein barycenters
    Vector<double> tmp_grad_support_points;
    Vector<double> tmp_grad_measure;
};

#endif // SOT_SOLVER_H
