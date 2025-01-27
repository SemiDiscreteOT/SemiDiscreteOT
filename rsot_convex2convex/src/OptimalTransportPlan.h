#ifndef OPTIMAL_TRANSPORT_PLAN_H
#define OPTIMAL_TRANSPORT_PLAN_H

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/parameter_acceptor.h>

#include <memory>
#include <string>
#include <vector>
#include <map>

using namespace dealii;

namespace OptimalTransportPlanSpace {

// Forward declarations
template <int dim> class MapApproximationStrategy;

/**
 * Class for computing and managing optimal transport map approximations.
 * This class provides various strategies for approximating the optimal transport
 * map given source/target measures and optimal transport potentials.
 */
template <int dim>
class OptimalTransportPlan : public ParameterAcceptor {
public:
    /**
     * Constructor taking an optional strategy name.
     */
    OptimalTransportPlan(const std::string& strategy_name = "nearest_neighbor");

    /**
     * Set the source measure data.
     * @param points Vector of source points
     * @param density Vector of density values at source points
     */
    void set_source_measure(const std::vector<Point<dim>>& points,
                          const std::vector<double>& density);

    /**
     * Set the target measure data.
     * @param points Vector of target points
     * @param density Vector of density values at target points
     */
    void set_target_measure(const std::vector<Point<dim>>& points,
                          const std::vector<double>& density);

    /**
     * Set the optimal transport potential.
     * @param potential Vector of potential values at target points
     * @param regularization_param The regularization parameter used (if any)
     */
    void set_potential(const Vector<double>& potential,
                      const double regularization_param = 0.0);

    /**
     * Compute the optimal transport map approximation using the current strategy.
     */
    void compute_map();

    /**
     * Save the computed transport map to files.
     * @param output_dir Directory where to save the results
     */
    void save_map(const std::string& output_dir) const;

    /**
     * Change the approximation strategy.
     * @param strategy_name Name of the strategy to use
     */
    void set_strategy(const std::string& strategy_name);

    /**
     * Get available strategy names.
     */
    static std::vector<std::string> get_available_strategies();

private:
    // Parameters for the approximation
    struct Parameters {
        unsigned int n_samples = 1000;        // Number of samples for Monte Carlo methods
        unsigned int n_neighbors = 10;         // Number of neighbors for local methods
        double kernel_width = 0.1;            // Kernel width for smooth approximations
        std::string interpolation_type = "linear"; // Type of interpolation
    } params;

    // Data members
    std::vector<Point<dim>> source_points;
    std::vector<double> source_density;
    std::vector<Point<dim>> target_points;
    std::vector<double> target_density;
    Vector<double> transport_potential;
    double regularization_parameter;

    // Strategy pattern implementation
    std::unique_ptr<MapApproximationStrategy<dim>> strategy;

    // Factory method to create strategies
    static std::unique_ptr<MapApproximationStrategy<dim>> 
    create_strategy(const std::string& name);
};

/**
 * Abstract base class for map approximation strategies.
 */
template <int dim>
class MapApproximationStrategy {
public:
    virtual ~MapApproximationStrategy() = default;

    virtual void compute_map(const std::vector<Point<dim>>& source_points,
                           const std::vector<double>& source_density,
                           const std::vector<Point<dim>>& target_points,
                           const std::vector<double>& target_density,
                           const Vector<double>& potential,
                           const double regularization_param) = 0;

    virtual void save_results(const std::string& output_dir) const = 0;

protected:
    std::vector<Point<dim>> mapped_points;
    std::vector<double> transport_density;
};

/**
 * Nearest neighbor strategy for map approximation.
 */
template <int dim>
class NearestNeighborStrategy : public MapApproximationStrategy<dim> {
public:
    void compute_map(const std::vector<Point<dim>>& source_points,
                    const std::vector<double>& source_density,
                    const std::vector<Point<dim>>& target_points,
                    const std::vector<double>& target_density,
                    const Vector<double>& potential,
                    const double regularization_param) override;

    void save_results(const std::string& output_dir) const override;
};

/**
 * Barycentric interpolation strategy for map approximation.
 */
template <int dim>
class BarycentricStrategy : public MapApproximationStrategy<dim> {
public:
    void compute_map(const std::vector<Point<dim>>& source_points,
                    const std::vector<double>& source_density,
                    const std::vector<Point<dim>>& target_points,
                    const std::vector<double>& target_density,
                    const Vector<double>& potential,
                    const double regularization_param) override;

    void save_results(const std::string& output_dir) const override;
};

/**
 * Kernel-based strategy for smooth map approximation.
 */
template <int dim>
class KernelStrategy : public MapApproximationStrategy<dim> {
public:
    void compute_map(const std::vector<Point<dim>>& source_points,
                    const std::vector<double>& source_density,
                    const std::vector<Point<dim>>& target_points,
                    const std::vector<double>& target_density,
                    const Vector<double>& potential,
                    const double regularization_param) override;

    void save_results(const std::string& output_dir) const override;
};

} // namespace OptimalTransportPlanSpace

#endif // OPTIMAL_TRANSPORT_PLAN_H 