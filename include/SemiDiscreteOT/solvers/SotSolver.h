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
        // Static constant for zero tolerance check
        static constexpr double zero_tolerance = std::numeric_limits<double>::min();  // Smallest positive normalized value

        // The tolerance passed to constructor is for checking if residual is effectively zero
        VerboseSolverControl(unsigned int n, double zero_tol, ConditionalOStream& pcout_)
            : SolverControl(n, zero_tol)  // This tolerance is for checking against zero
            , pcout(pcout_)
            , gradient(nullptr)
            , target_measure(nullptr)
            , user_tolerance(1.0)  
        {}

        void set_gradient(const Vector<double>& grad) {
            gradient = &grad;
        }

        void set_target_measure(const Vector<double>& target_density) {
            target_measure = &target_density;
        }

        void set_user_tolerance(double tol) {
            user_tolerance = tol;
        }

        virtual State check(unsigned int step, double value) override
        {
            AssertThrow(gradient != nullptr, 
                        ExcMessage("Gradient vector not set in VerboseSolverControl"));
            AssertThrow(target_measure != nullptr,
                        ExcMessage("Target measure not set in VerboseSolverControl"));

            // Check if abs(grad_j) < target_measure_j * user_tolerance for every j
            double max_scaled_residual = -std::numeric_limits<double>::infinity();
            for (unsigned int j = 0; j < gradient->size(); ++j) {
                double scaled_residual = std::abs((*gradient)[j]) - (*target_measure)[j] * user_tolerance;
                max_scaled_residual = std::max(max_scaled_residual, scaled_residual);
            }
            
            double rel_residual = (step == 0 || initial_residual == 0.0) ? 
                                  max_scaled_residual : max_scaled_residual / initial_residual;
            
            if (step == 0)
                initial_residual = max_scaled_residual;
            
            std::string color;
            if (max_scaled_residual < tolerance()) {  // tolerance() is the zero-check tolerance from constructor
                color = Color::green;  // Residual effectively zero
            } else if (rel_residual < 0.5) {  // Fixed threshold for progress indication
                color = Color::yellow;   // Making progress
            } else {
                color = Color::red; // Far from convergence
            }
            
            pcout << "Iteration " << CYAN << step << RESET
                  << " - L-2 gradient norm: " << color << value << RESET
                  << " - Max scaled residual: " << color << max_scaled_residual << RESET
                  << " - Relative residual: " << color << rel_residual << RESET << std::endl;
                
            // Check if max_scaled_residual is effectively zero using the constructor's tolerance
            return SolverControl::check(step, max_scaled_residual);
        }
        
    private:
        ConditionalOStream& pcout;
        double initial_residual = 1.0;
        const Vector<double>* gradient;
        const Vector<double>* target_measure;
        double user_tolerance; 
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
