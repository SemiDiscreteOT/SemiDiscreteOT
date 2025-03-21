#ifndef POINT_CLOUD_HIERARCHY_H
#define POINT_CLOUD_HIERARCHY_H

#include <deal.II/base/point.h>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <unordered_map>
#include "SemiDiscreteOT/utils/dkm/dkm.hpp"
#include "SemiDiscreteOT/utils/dkm/dkm_parallel.hpp"
#include "SemiDiscreteOT/utils/dkm/dkm_utils.hpp"

using namespace dealii;

namespace PointCloudHierarchy {

/**
 * @brief Class to manage a hierarchy of point clouds with different resolutions
 * The hierarchy is organized with coarser levels having fewer points (parents)
 * and finer levels having more points (children).
 */
class PointCloudHierarchyManager {
public:
    /**
     * @brief Constructor
     * @param min_points Minimum number of points for the coarsest level (parent level)
     * @param max_points Maximum number of points for level 1 point cloud
     */
    PointCloudHierarchyManager(int min_points = 100, int max_points = 1000);

    /**
     * @brief Generate hierarchy of point clouds from input points
     * @param input_points Vector of input points (finest level)
     * @param input_density Vector of input density values (optional, uniform density if empty)
     * @param output_dir Directory to save the point cloud hierarchy
     * @return Number of levels generated
     * @throws std::runtime_error if processing fails
     */
    template <int dim>
    int generateHierarchy(
        const std::vector<Point<dim>>& input_points,
        const std::vector<double>& input_density,
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

    /**
     * @brief Get the parent indices for points at level L-1
     * @param level The level of the parents (must be > 0)
     * @return For each point at level L-1, returns index of its parent at level L
     */
    const std::vector<std::vector<size_t>>& getParentIndices(int level) const;

    /**
     * @brief Get the child indices for points at level L
     * @param level The level of points whose children we want (must be < num_levels-1)
     * @return For each point at level L, returns indices of its children at level L-1
     */
    const std::vector<std::vector<size_t>>& getChildIndices(int level) const;

private:
    int min_points_;
    int max_points_;
    int num_levels_;
    std::vector<int> level_point_counts_;

    // parent_indices_[L] stores the relationships between level L and level L+1
    // For each point at level L, parent_indices_[L][point_idx] contains indices of its parents at level L+1
    std::vector<std::vector<std::vector<size_t>>> parent_indices_;

    // child_indices_[L] stores the relationships between level L+1 and level L
    // For each point at level L+1, child_indices_[L][point_idx] contains indices of its children at level L
    std::vector<std::vector<std::vector<size_t>>> child_indices_;

    /**
     * @brief Ensure directory exists, create if it doesn't
     */
    void ensureDirectoryExists(const std::string& path) const;

    /**
     * @brief Performs parallel k-means clustering on a set of points with parent-child tracking
     * @param points Input points from finer level
     * @param densities Input densities
     * @param k Number of clusters (points at coarser level)
     * @return Tuple of:
     *         - cluster centers (parent points)
     *         - aggregated densities for parents
     *         - assignments (mapping of each child to its parent cluster)
     */
    template <int dim>
    std::tuple<std::vector<std::array<double, dim>>, std::vector<double>, std::vector<int>>
    kmeansClustering(
        const std::vector<std::array<double, dim>>& points,
        const std::vector<double>& densities,
        size_t k);
};

} // namespace PointCloudHierarchy

#endif // POINT_CLOUD_HIERARCHY_H
