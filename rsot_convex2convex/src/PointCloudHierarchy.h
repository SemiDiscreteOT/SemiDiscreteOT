#ifndef POINT_CLOUD_HIERARCHY_H
#define POINT_CLOUD_HIERARCHY_H

#include <deal.II/base/point.h>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <random>
#include <omp.h>

using namespace dealii;

namespace PointCloudHierarchy {

/**
 * @brief Class to manage a hierarchy of point clouds with different resolutions
 * The hierarchy is organized with coarser levels having fewer points
 * selected randomly from the finest level.
 */
class PointCloudHierarchyManager {
public:
    /**
     * @brief Constructor
     * @param min_points Minimum number of points for the coarsest level
     * @param max_points Maximum number of points for level 1 point cloud
     */
    PointCloudHierarchyManager(int min_points = 100, int max_points = 1000);

    /**
     * @brief Generate hierarchy of point clouds from input points
     * @param input_points Vector of input points (finest level)
     * @param input_weights Vector of input weights (optional, uniform weights if empty)
     * @param output_dir Directory to save the point cloud hierarchy
     * @return Number of levels generated
     * @throws std::runtime_error if processing fails
     */
    template <int dim>
    int generateHierarchy(
        const std::vector<Point<dim>>& input_points,
        const std::vector<double>& input_weights,
        const std::string& output_dir);

    /**
     * @brief Set the maximum number of points for level 1
     */
    void setMaxPoints(int max_points);

    /**
     * @brief Set the minimum number of points for coarsest level
     */
    void setMinPoints(int min_points);

    /**
     * @brief Get the number of levels in the last generated hierarchy
     */
    int getNumLevels() const;

    /**
     * @brief Get the number of points at a specific level
     */
    int getPointCount(int level) const;

    /**
     * @brief Calculate number of points for a given level
     */
    int getPointsForLevel(int base_points, int level) const;

private:
    int min_points_;
    int max_points_;
    int num_levels_;
    std::vector<int> level_point_counts_;

    /**
     * @brief Ensure directory exists, create if it doesn't
     */
    void ensureDirectoryExists(const std::string& path) const;

    /**
     * @brief Performs random sampling of points from the finest level
     * @param points Input points from finest level
     * @param weights Input weights
     * @param num_samples Number of points to sample
     * @param seed Random seed for reproducibility
     * @return Tuple of:
     *         - sampled points
     *         - normalized weights for sampled points
     *         - indices of sampled points in the original array
     */
    template <int dim>
    std::tuple<std::vector<std::array<double, dim>>, std::vector<double>, std::vector<size_t>>
    randomSampling(
        const std::vector<std::array<double, dim>>& points,
        const std::vector<double>& weights,
        size_t num_samples,
        unsigned int seed = 42);
};

} // namespace PointCloudHierarchy

#endif // POINT_CLOUD_HIERARCHY_H
