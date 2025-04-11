#ifndef SOT_SOLVER_H
#define SOT_SOLVER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>
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
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <memory>
#include <map>
#include <mutex>
#include <atomic>
#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/utils/ColorDefinitions.h"

using namespace dealii;

template <int dim>
class SotSolver {
public:
    // Type definitions for RTree
    using IndexedPoint = std::pair<Point<dim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;

    // Source measure data structure
    struct SourceMeasure {
        SmartPointer<const DoFHandler<dim>> dof_handler;
        SmartPointer<const Mapping<dim>> mapping;
        SmartPointer<const FiniteElement<dim>> fe;
        SmartPointer<const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>> density;
        unsigned int quadrature_order;
        
        SourceMeasure() = default;
        SourceMeasure(const DoFHandler<dim>& dof_handler_,
                     const Mapping<dim>& mapping_,
                     const FiniteElement<dim>& fe_,
                     const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>& density_,
                     const unsigned int quadrature_order_)
            : dof_handler(&dof_handler_)
            , mapping(&mapping_)
            , fe(&fe_)
            , density(&density_)
            , quadrature_order(quadrature_order_) 
        {}
    };

    // Target measure data structure
    struct TargetMeasure {
        std::vector<Point<dim>> points;
        Vector<double> density;
        RTree rtree;
        
        TargetMeasure() = default;
        TargetMeasure(const std::vector<Point<dim>>& points_,
                     const Vector<double>& density_)
            : points(points_)
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
        ScratchData(const FiniteElement<dim>& fe,
                   const Mapping<dim>& mapping,
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

        FEValues<dim> fe_values;
        std::vector<double> density_values;
    };

    // Per-cell copy data for parallel assembly
    struct CopyData {
        double functional_value{0.0};
        Vector<double> gradient_values;  // Local gradient contribution
        double C_integral{0.0};
        Vector<double> potential_values;  // For softmax refinement

        CopyData(const unsigned int n_target_points)
            : gradient_values(n_target_points),
              potential_values(n_target_points) 
        {
            gradient_values = 0;  // Initialize local gradient to zero
        }
    };

    // Constructor
    SotSolver(const MPI_Comm& comm);

    // Setup methods
    void setup_source(const DoFHandler<dim>& dof_handler,
                     const Mapping<dim>& mapping,
                     const FiniteElement<dim>& fe,
                     const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>& source_density,
                     const unsigned int quadrature_order);

    void setup_target(const std::vector<Point<dim>>& target_points,
                     const Vector<double>& target_density);

    // Main solver interface
    void solve(Vector<double>& potential,
              const ParameterManager::SolverParameters& params);

    // Alternative solve interface if measures not set up beforehand
    void solve(Vector<double>& potential,
              const SourceMeasure& source,
              const TargetMeasure& target,
              const ParameterManager::SolverParameters& params);

    // Getters for solver results
    double get_last_functional_value() const { return global_functional; }
    unsigned int get_last_iteration_count() const;
    bool get_convergence_status() const;
    double get_cache_size_mb() const;
    double get_last_distance_threshold() const { return current_distance_threshold; }
    bool get_cache_limit_reached() const { return cache_limit_reached; }

private:
    // Core evaluation method
    double evaluate_functional(const Vector<double>& potential,
                             Vector<double>& gradient_out);

    // Local assembly methods
    void local_assemble(const typename DoFHandler<dim>::active_cell_iterator& cell,
                       ScratchData& scratch,
                       CopyData& copy);

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

    // MPI and parallel related members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Source and target measures
    SourceMeasure source_measure;
    TargetMeasure target_measure;

    // Solver state
    std::unique_ptr<SolverControl> solver_control;
    mutable double current_distance_threshold;
    mutable double effective_distance_threshold;
    mutable bool is_caching_active;
    const Vector<double>* current_potential;
    double current_lambda;
    mutable double global_functional;
    mutable double global_C_integral;
    Vector<double> gradient;  

    // Cache for local assembly computations
    struct CellCache {
        std::vector<std::size_t> target_indices;
        std::vector<double> precomputed_exp_terms;
        bool is_valid;

        CellCache() : is_valid(false) {}
    };

    mutable std::unordered_map<std::string, CellCache> cell_caches;
    mutable std::mutex cache_mutex;
    std::atomic<size_t> total_target_points{0};
    mutable std::atomic<double> current_cache_size_mb{0.0};   ///< Current cache size in MB
    mutable bool cache_limit_reached{false};                  ///< Flag indicating if cache limit was reached

    // Current solver parameters
    ParameterManager::SolverParameters current_params;

    // Distance threshold and caching methods
    void compute_distance_threshold() const;
    void reset_distance_threshold_cache() const;
    std::vector<std::size_t> find_nearest_target_points(const Point<dim>& query_point) const;
    double estimate_cache_entry_size_mb(const std::vector<std::size_t>& target_indices, 
                                      unsigned int n_q_points) const;

    // Validation methods
    bool validate_measures() const;
};

#endif // SOT_SOLVER_H
