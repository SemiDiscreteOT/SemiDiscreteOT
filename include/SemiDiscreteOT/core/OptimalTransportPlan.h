#ifndef OPTIMAL_TRANSPORT_PLAN_H
#define OPTIMAL_TRANSPORT_PLAN_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include "SemiDiscreteOT/solvers/Distance.h"
#include "SemiDiscreteOT/utils/utils.h"


using namespace dealii;

namespace OptimalTransportPlanSpace {

// Forward declarations
template <int spacedim> class MapApproximationStrategy;

/**
 * @brief A class for computing and managing optimal transport map approximations.
 *
 * This class provides various strategies for approximating the optimal transport
 * map given source/target measures and optimal transport potentials.
 *
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int spacedim>
class OptimalTransportPlan : public ParameterAcceptor {
public:
    /**
     * @brief Constructor taking an optional strategy name.
     * @param strategy_name The name of the approximation strategy to use.
     */
    OptimalTransportPlan(const std::string& strategy_name = "modal");

    /**
     * @brief Set the distance function used to compute distances between points.
     * This function accepts a callable object (e.g., lambda) that takes two points
     * as input and returns a double representing their distance.
     *
     * @param distance_function Function to compute distance between two points.
     */
    void set_distance_function(const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& dist)
    {
        distance_function = dist;
    }

    /**
     * @brief Set the source measure data.
     * @param points Vector of source points
     * @param density Vector of density values at source points
     */
    void set_source_measure(const std::vector<Point<spacedim>>& points,
                          const std::vector<double>& density);

    /**
     * @brief Set the target measure data.
     * @param points Vector of target points
     * @param density Vector of density values at target points
     */
    void set_target_measure(const std::vector<Point<spacedim>>& points,
                          const std::vector<double>& density);

    /**
     * @brief Set the optimal transport potential.
     * @param potential Vector of potential values at target points
     * @param regularization_param The regularization parameter used (if any)
     */
    void set_potential(const Vector<double>& potential,
                      const double regularization_param = 0.0);

    /**
     * @brief Set the truncation radius for map computation.
     * Points outside this radius will not be considered in the map computation.
     * A negative value means no truncation (all points are considered).
     * @param radius The truncation radius
     */
    void set_truncation_radius(double radius);

    /**
     * @brief Compute the optimal transport map approximation using the current strategy.
     */
    void compute_map();

    /**
     * @brief Save the computed transport map to files.
     * @param output_dir Directory where to save the results
     */
    void save_map(const std::string& output_dir) const;

    /**
     * @brief Change the approximation strategy.
     * @param strategy_name Name of the strategy to use
     */
    void set_strategy(const std::string& strategy_name);

    /**
     * @brief Get available strategy names.
     * @return A vector of strings containing the names of the available strategies.
     */
    static std::vector<std::string> get_available_strategies();

private:
    // Data members
    std::vector<Point<spacedim>> source_points;
    std::vector<double> source_density;
    std::vector<Point<spacedim>> target_points;
    std::vector<double> target_density;
    Vector<double> transport_potential;
    double epsilon;
    double truncation_radius = -1.0;  // Negative means no truncation

    // Distance function
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function;

    // Strategy pattern implementation
    std::unique_ptr<MapApproximationStrategy<spacedim>> strategy;

    // Factory method to create strategies
    static std::unique_ptr<MapApproximationStrategy<spacedim>> 
    create_strategy(const std::string& name);
};

/**
 * @brief Abstract base class for map approximation strategies.
 *
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int spacedim>
class MapApproximationStrategy {
public:
    virtual ~MapApproximationStrategy() = default;

    /**
     * @brief Computes the transport map.
     * @param distance_function The distance function.
     * @param source_points The source points.
     * @param source_density The source density.
     * @param target_points The target points.
     * @param target_density The target density.
     * @param potential The optimal transport potential.
     * @param regularization_param The regularization parameter.
     * @param truncation_radius The truncation radius.
     */
    virtual void compute_map(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function,
        const std::vector<Point<spacedim>>& source_points,
        const std::vector<double>& source_density,
        const std::vector<Point<spacedim>>& target_points,
        const std::vector<double>& target_density,
        const Vector<double>& potential,
        const double regularization_param,
        const double truncation_radius) = 0;

    /**
     * @brief Saves the results to a file.
     * @param output_dir The directory to save the results to.
     */
    virtual void save_results(const std::string& output_dir) const = 0;

protected:
    std::vector<Point<spacedim>> source_points; ///< The source points.
    std::vector<Point<spacedim>> mapped_points; ///< The mapped points.
    std::vector<double> transport_density; ///< The transported density.
};

/**
 * @brief Modal strategy for map approximation.
 *
 * Maps each source point to the target point that maximizes:
 * score = potential[j] - 0.5*||x-y||^2 + regularization_param * log(target_density[j])
 *
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
 template <int spacedim>
 class ModalStrategy : public MapApproximationStrategy<spacedim> {
 public:
     /**
      * @brief Computes the transport map.
      * @param distance_function The distance function.
      * @param source_points The source points.
      * @param source_density The source density.
      * @param target_points The target points.
      * @param target_density The target density.
      * @param potential The optimal transport potential.
      * @param regularization_param The regularization parameter.
      * @param truncation_radius The truncation radius.
      */
     void compute_map(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function,
        const std::vector<Point<spacedim>>& source_points,
        const std::vector<double>& source_density,
        const std::vector<Point<spacedim>>& target_points,
        const std::vector<double>& target_density,
        const Vector<double>& potential,
        const double regularization_param,
        const double truncation_radius) override;
 
     /**
      * @brief Saves the results to a file.
      * @param output_dir The directory to save the results to.
      */
     void save_results(const std::string& output_dir) const override;
 };

/**
 * @brief Barycentric interpolation strategy for map approximation.
 *
 * Computes a weighted average of target points where weights are:
 * w_j = target_density[j] * exp((potential[j] - 0.5*||x-y||^2) / regularization_param)
 *
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int spacedim>
class BarycentricStrategy : public MapApproximationStrategy<spacedim> {
public:
    /**
     * @brief Computes the transport map.
     * @param distance_function The distance function.
     * @param source_points The source points.
     * @param source_density The source density.
     * @param target_points The target points.
     * @param target_density The target density.
     * @param potential The optimal transport potential.
     * @param regularization_param The regularization parameter.
     * @param truncation_radius The truncation radius.
     */
    void compute_map(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function,
        const std::vector<Point<spacedim>>& source_points,
        const std::vector<double>& source_density,
        const std::vector<Point<spacedim>>& target_points,
        const std::vector<double>& target_density,
        const Vector<double>& potential,
        const double regularization_param,
        const double truncation_radius) override;

    /**
     * @brief Saves the results to a file.
     * @param output_dir The directory to save the results to.
     */
    void save_results(const std::string& output_dir) const override;
};

} // namespace OptimalTransportPlanSpace

#endif // OPTIMAL_TRANSPORT_PLAN_H 