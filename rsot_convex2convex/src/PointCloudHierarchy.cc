#include "PointCloudHierarchy.h"
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

    // Calculate a reduction factor based on the number of levels and min_points
    // This ensures that each level has a different number of points
    double reduction_factor = std::pow((double)min_points_ / base_points, 1.0 / (num_levels_ - 1));

    // Calculate points for this level using the reduction factor
    int points = static_cast<int>(base_points * std::pow(reduction_factor, level));

    // Ensure we don't go below min_points
    return std::max(points, min_points_);
}

template <int dim>
std::tuple<std::vector<std::array<double, dim>>, std::vector<double>, std::vector<size_t>>
PointCloudHierarchyManager::randomSampling(
    const std::vector<std::array<double, dim>>& points,
    const std::vector<double>& weights,
    size_t num_samples,
    unsigned int seed) {

    const size_t n_points = points.size();

    // If we need more samples than available points, return all points
    if (n_points <= num_samples) {
        std::vector<size_t> indices(n_points);
        std::iota(indices.begin(), indices.end(), 0);
        return {points, weights, indices};
    }

    // Create a vector of indices and shuffle it
    std::vector<size_t> indices(n_points);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Take the first num_samples indices
    indices.resize(num_samples);

    // Extract the sampled points and weights
    std::vector<std::array<double, dim>> sampled_points(num_samples);
    std::vector<double> sampled_weights(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        sampled_points[i] = points[indices[i]];
        sampled_weights[i] = weights.empty() ? 1.0 : weights[indices[i]];
    }

    // Normalize the weights to sum to 1.0
    double weight_sum = std::accumulate(sampled_weights.begin(), sampled_weights.end(), 0.0);
    if (weight_sum > 0.0) {
        for (auto& w : sampled_weights) {
            w /= weight_sum;
        }
    } else {
        // If all weights are zero, use uniform weights
        for (auto& w : sampled_weights) {
            w = 1.0 / num_samples;
        }
    }

    return {sampled_points, sampled_weights, indices};
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

    // Calculate number of levels based on min_points and total_points
    const int total_points = input_points.size();

    // Adjust max_points if necessary to ensure level 1 is different from level 0
    if (total_points <= max_points_) {
        // If original point cloud is smaller than max_points, adjust max_points
        // to be a fraction of the original size to ensure level 1 is different
        max_points_ = static_cast<int>(total_points * 0.75);
        // Ensure max_points is at least min_points + 1
        max_points_ = std::max(max_points_, min_points_ + 1);
    }

    // Calculate number of levels
    // We want to ensure a smooth reduction from total_points to min_points
    double log_ratio = std::log((double)min_points_ / (double)max_points_);
    num_levels_ = log_ratio >= 0 ? 1 : static_cast<int>(std::ceil(std::log((double)min_points_ / (double)max_points_) / std::log(0.5))) + 2;
    num_levels_ = std::min(num_levels_, 5); // Cap at 5 levels
    num_levels_ = std::max(num_levels_, 2); // At least 2 levels

    std::cout << "Initial point cloud has " << total_points << " points" << std::endl;
    std::cout << "Using max_points = " << max_points_ << " for level 1" << std::endl;
    std::cout << "Creating " << num_levels_ << " levels of point clouds" << std::endl;

    // Reset level point counts
    level_point_counts_.clear();

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
    std::vector<std::vector<size_t>> level_indices_vec(num_levels_);

    // Level 0 is always the original point cloud
    level_points_vec[0] = points_array;
    level_weights_vec[0] = weights;
    level_point_counts_.push_back(input_points.size());

    std::cout << "Level 0: using original point cloud with " << level_points_vec[0].size() << " points" << std::endl;

    // Generate coarser levels by random sampling from the finest level
    for (int level = 1; level < num_levels_; ++level) {
        int points_for_level = getPointsForLevel(total_points, level);

        // Ensure each level has a different number of points
        if (level > 1 && points_for_level >= level_point_counts_[level-1]) {
            points_for_level = static_cast<int>(level_point_counts_[level-1] * 0.5);
            points_for_level = std::max(points_for_level, min_points_);
        }

        std::cout << "Level " << level << ": targeting " << points_for_level << " points" << std::endl;

        // Use random sampling to create coarser point cloud
        std::vector<std::array<double, dim>> sampled_points;
        std::vector<double> sampled_weights;
        std::vector<size_t> sampled_indices;

        // Use a different seed for each level for better randomness
        unsigned int seed = 42;

        std::tie(sampled_points, sampled_weights, sampled_indices) = randomSampling<dim>(
            points_array, weights, points_for_level, seed);

        level_points_vec[level] = sampled_points;
        level_weights_vec[level] = sampled_weights;
        level_indices_vec[level] = sampled_indices;
        level_point_counts_.push_back(sampled_points.size());

        std::cout << "  Generated " << sampled_points.size() << " points by random sampling" << std::endl;
    }

    // Save the point clouds for each level
    for (int level = 0; level < num_levels_; ++level) {
        std::string points_file = output_dir + "/level_" + std::to_string(level) + "_points.txt";
        std::string weights_file = output_dir + "/level_" + std::to_string(level) + "_weights.txt";
        std::string indices_file = output_dir + "/level_" + std::to_string(level) + "_indices.txt";

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

        // For levels > 0, save the indices of the original points that were sampled
        if (level > 0) {
            std::ofstream indices_out(indices_file);
            if (!indices_out) {
                throw std::runtime_error("Failed to open indices file for level " + std::to_string(level));
            }

            for (const auto& idx : level_indices_vec[level]) {
                indices_out << idx << std::endl;
            }
        }

        std::cout << "Saved level " << level << " point cloud to " << output_dir << std::endl;
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

// Explicit template instantiations for randomSampling
template std::tuple<std::vector<std::array<double, 2>>, std::vector<double>, std::vector<size_t>>
PointCloudHierarchyManager::randomSampling<2>(
    const std::vector<std::array<double, 2>>&, const std::vector<double>&, size_t, unsigned int);
template std::tuple<std::vector<std::array<double, 3>>, std::vector<double>, std::vector<size_t>>
PointCloudHierarchyManager::randomSampling<3>(
    const std::vector<std::array<double, 3>>&, const std::vector<double>&, size_t, unsigned int);

} // namespace PointCloudHierarchy
