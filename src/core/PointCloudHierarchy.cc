#include "SemiDiscreteOT/core/PointCloudHierarchy.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <limits>
#include <chrono>

namespace fs = std::filesystem;

namespace PointCloudHierarchy {

PointCloudHierarchyManager::PointCloudHierarchyManager(int min_points, int max_points)
    : min_points_(min_points)
    , max_points_(max_points)
    , num_levels_(0)
{}

void PointCloudHierarchyManager::setMaxPoints(int max_points) {
    max_points_ = max_points;
}

void PointCloudHierarchyManager::setMinPoints(int min_points) {
    min_points_ = min_points;
}

int PointCloudHierarchyManager::getNumLevels() const {
    return num_levels_;
}

int PointCloudHierarchyManager::getPointCount(int level) const {
    if (level < 0 || level >= static_cast<int>(level_point_counts_.size())) {
        throw std::out_of_range("Level index out of range");
    }
    return level_point_counts_[level];
}

const std::vector<std::vector<size_t>>& PointCloudHierarchyManager::getParentIndices(int level) const {
    if (level <= 0 || level >= num_levels_) {
        throw std::out_of_range("Level index out of range for parent indices");
    }
    return parent_indices_[level-1];
}

const std::vector<std::vector<size_t>>& PointCloudHierarchyManager::getChildIndices(int level) const {
    if (level < 0 || level >= num_levels_-1) {
        throw std::out_of_range("Level index out of range for child indices");
    }
    return child_indices_[level];
}

void PointCloudHierarchyManager::ensureDirectoryExists(const std::string& path) const {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

int PointCloudHierarchyManager::getPointsForLevel(int base_points, int level) const {
    // Level 0 always uses original point cloud size
    if (level == 0) {
        return base_points;
    }
    // Level 1 starts with max_points (if smaller than base_points)
    if (level == 1) {
        return std::min(base_points, max_points_);
    }
    // For subsequent levels, use a reduction factor of 4 based on level 1's size
    int level1_points = std::min(base_points, max_points_);
    int points = static_cast<int>(level1_points / std::pow(4.0, level - 1));
    return std::max(points, min_points_);
}

template <int dim>
std::tuple<std::vector<std::array<double, dim>>, std::vector<double>, std::vector<int>> 
PointCloudHierarchyManager::kmeansClustering(
    const std::vector<std::array<double, dim>>& points,
    const std::vector<double>& weights,
    size_t k) {

    const size_t n_points = points.size();
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads" << std::endl;
    
    if (points.size() <= k) {
        // If fewer points than clusters, return original points
        std::vector<int> assignments(points.size());
        std::iota(assignments.begin(), assignments.end(), 0); // Each point is its own cluster
        return {points, weights, assignments};
    }
    
    // Set up clustering parameters
    dkm::clustering_parameters<double> params(k);

    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run parallel k-means clustering
    std::vector<std::array<double, dim>> centers;
    std::vector<uint32_t> cluster_assignments;
    std::tie(centers, cluster_assignments) = dkm::kmeans_lloyd_parallel(points, k);
    
    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    
    std::cout << "  K-means clustering took " << duration.count() << " seconds" << std::endl;
    
    // Convert cluster assignments to int
    std::vector<int> assignments(cluster_assignments.begin(), cluster_assignments.end());
    
    // Compute weights for each cluster
    std::vector<double> cluster_weights(k, 0.0);
    for (size_t i = 0; i < points.size(); ++i) {
        int cluster = assignments[i];
        double point_weight = weights.empty() ? 1.0 : weights[i];
        cluster_weights[cluster] += point_weight;
    }
    
    return {centers, cluster_weights, assignments};
}

template <int dim>
int PointCloudHierarchyManager::generateHierarchy(
    const std::vector<Point<dim>>& input_points,
    const std::vector<double>& input_weights,
    const std::string& output_dir) {
    
    // Validate inputs
    if (input_points.empty()) {
        throw std::runtime_error("Input point cloud is empty");
    }
    
    // Start timing the entire hierarchy generation
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Create uniform weights if none provided
    std::vector<double> weights;
    if (input_weights.empty()) {
        weights.resize(input_points.size(), 1.0 / input_points.size());
    } else {
        if (input_weights.size() != input_points.size()) {
            throw std::runtime_error("Weight vector size doesn't match point cloud size");
        }
        weights = input_weights;
    }
    
    // Create output directories
    ensureDirectoryExists(output_dir);
    
    // Calculate number of levels
    const int total_points = input_points.size();
    const int level1_points = std::min(total_points, max_points_);
    
    num_levels_ = std::min(5, static_cast<int>(std::log(level1_points / min_points_) / std::log(4.0)) + 2);
    num_levels_ = std::max(1, num_levels_); // At least one level
    
    std::cout << "Initial point cloud has " << total_points << " points" << std::endl;
    std::cout << "Level 1 will have maximum of " << max_points_ << " points" << std::endl;
    std::cout << "Creating " << num_levels_ << " levels of point clouds" << std::endl;
    
    // Reset level point counts and parent-child relationships
    level_point_counts_.clear();
    parent_indices_.clear();
    child_indices_.clear();
    
    // Resize parent and child indices containers
    parent_indices_.resize(num_levels_ - 1);
    child_indices_.resize(num_levels_ - 1);
    
    // Convert input points to std::array format
    std::vector<std::array<double, dim>> points_array(input_points.size());
    for (size_t i = 0; i < input_points.size(); ++i) {
        for (int d = 0; d < dim; ++d) {
            points_array[i][d] = input_points[i][d];
        }
    }
    
    // Store all point clouds for each level
    std::vector<std::vector<std::array<double, dim>>> level_points_vec(num_levels_);
    std::vector<std::vector<double>> level_weights_vec(num_levels_);
    
    // Level 0 is always the original point cloud
    level_points_vec[0] = points_array;
    level_weights_vec[0] = weights;
    level_point_counts_.push_back(input_points.size());
    
    std::cout << "Level 0: using original point cloud with " << level_points_vec[0].size() << " points" << std::endl;
    
    // Generate coarser levels
    for (int level = 1; level < num_levels_; ++level) {
        int points_for_level = getPointsForLevel(total_points, level);
        std::cout << "Level " << level << ": targeting " << points_for_level << " points" << std::endl;
        
        // Use k-means clustering to create coarser point cloud with parent-child tracking
        std::vector<std::array<double, dim>> coarse_points;
        std::vector<double> coarse_weights;
        std::vector<int> assignments;
        
        std::tie(coarse_points, coarse_weights, assignments) = kmeansClustering<dim>(
            level_points_vec[level-1], level_weights_vec[level-1], points_for_level);
        
        level_points_vec[level] = coarse_points;
        level_weights_vec[level] = coarse_weights;
        level_point_counts_.push_back(coarse_points.size());
        
        std::cout << "  Generated " << coarse_points.size() << " points after clustering" << std::endl;
        
        // Build parent-child relationships
        const size_t n_fine_points = level_points_vec[level-1].size();
        const size_t n_coarse_points = coarse_points.size();
        
        // Initialize parent indices for previous (finer) level
        parent_indices_[level-1].resize(n_fine_points);
        
        // Initialize child indices for this (coarser) level
        child_indices_[level-1].resize(n_coarse_points);
        
        // Populate parent-child relationships
        for (size_t i = 0; i < n_fine_points; ++i) {
            int coarse_point_idx = assignments[i];
            if (coarse_point_idx >= 0 && coarse_point_idx < static_cast<int>(n_coarse_points)) {
                // Add the coarse point as parent of this fine point
                parent_indices_[level-1][i].push_back(coarse_point_idx);
                
                // Add this fine point as child of the coarse point
                child_indices_[level-1][coarse_point_idx].push_back(i);
            }
        }
    }
    
    // Save the point clouds and parent-child relationships for each level
    for (int level = 0; level < num_levels_; ++level) {
        std::string points_file = output_dir + "/level_" + std::to_string(level) + "_points.txt";
        std::string weights_file = output_dir + "/level_" + std::to_string(level) + "_weights.txt";
        
        std::ofstream points_out(points_file);
        std::ofstream weights_out(weights_file);
        
        if (!points_out || !weights_out) {
            throw std::runtime_error("Failed to open output files for level " + std::to_string(level));
        }
        
        // Write points and weights
        for (size_t i = 0; i < level_points_vec[level].size(); ++i) {
            for (int d = 0; d < dim; ++d) {
                points_out << level_points_vec[level][i][d] << (d < dim - 1 ? " " : "");
            }
            points_out << std::endl;
            
            weights_out << level_weights_vec[level][i] << std::endl;
        }
        
        // Save parent-child relationships for non-boundary levels
        // Parents: For each point at level L, save its parent at level L+1
        if (level < num_levels_ - 1) {
            std::string parents_file = output_dir + "/level_" + std::to_string(level) + "_parents.txt";
            std::ofstream parents_out(parents_file);
            
            if (!parents_out) {
                throw std::runtime_error("Failed to open parents file for level " + std::to_string(level));
            }
            
            // Write parent indices for points at current level
            for (size_t i = 0; i < parent_indices_[level].size(); ++i) {
                parents_out << parent_indices_[level][i].size();
                for (const auto& parent_idx : parent_indices_[level][i]) {
                    parents_out << " " << parent_idx;
                }
                parents_out << std::endl;
            }
        }
        
        // Children: For each point at level L, save its children at level L-1
        if (level > 0) {
            std::string children_file = output_dir + "/level_" + std::to_string(level) + "_children.txt";
            std::ofstream children_out(children_file);
            
            if (!children_out) {
                throw std::runtime_error("Failed to open children file for level " + std::to_string(level));
            }
            
            // Write child indices for points at current level
            for (size_t i = 0; i < child_indices_[level-1].size(); ++i) {
                children_out << child_indices_[level-1][i].size();
                for (const auto& child_idx : child_indices_[level-1][i]) {
                    children_out << " " << child_idx;
                }
                children_out << std::endl;
            }
        }
        
        std::cout << "Saved level " << level << " point cloud and relationships to " << output_dir << std::endl;
    }
    
    // Stop timing the entire hierarchy generation
    auto stop_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::seconds>(stop_total - start_total);
    
    std::cout << "Total hierarchy generation took " << duration_total.count() << " seconds" << std::endl;
    
    return num_levels_;
}

// Explicit template instantiations for dimensions 2 and 3
template int PointCloudHierarchyManager::generateHierarchy<2>(
    const std::vector<Point<2>>&, const std::vector<double>&, const std::string&);
template int PointCloudHierarchyManager::generateHierarchy<3>(
    const std::vector<Point<3>>&, const std::vector<double>&, const std::string&);

// Explicit template instantiations
template std::tuple<std::vector<std::array<double, 2>>, std::vector<double>, std::vector<int>> 
PointCloudHierarchyManager::kmeansClustering<2>(
    const std::vector<std::array<double, 2>>&, const std::vector<double>&, size_t);
template std::tuple<std::vector<std::array<double, 3>>, std::vector<double>, std::vector<int>> 
PointCloudHierarchyManager::kmeansClustering<3>(
    const std::vector<std::array<double, 3>>&, const std::vector<double>&, size_t);

} // namespace PointCloudHierarchy