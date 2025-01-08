#pragma once

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include "utils.h"
#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/basic/process.h>
#include <geogram/basic/progress.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/voronoi/CVT.h>
#include <exploragram/optimal_transport/optimal_transport_3d.h>
#include <exploragram/optimal_transport/sampling.h>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

// Forward declarations for Geogram types
namespace GEO {
    class Mesh;
    class OptimalTransportMap3d;
}

/**
 * @brief Class to handle exact semi-discrete optimal transport using Geogram
 * 
 * This class provides a clean interface to compute exact semi-discrete optimal 
 * transport between a source mesh and target points using the Geogram library.
 */
class ExactSot {
public:
    /**
     * @brief Constructor
     */
    ExactSot();

    /**
     * @brief Destructor
     */
    ~ExactSot();

    // Delete copy operations due to unique GEO::Mesh ownership
    ExactSot(const ExactSot&) = delete;
    ExactSot& operator=(const ExactSot&) = delete;

    /**
     * @brief Set source mesh from file
     * @param filename Path to the source mesh file
     * @return true if successful, false otherwise
     */
    bool set_source_mesh(const std::string& filename);

    /**
     * @brief Set target points from file
     * @param filename Path to the target points file
     * @return true if successful, false otherwise
     */
    bool set_target_points(const std::string& filename);

    /**
     * @brief Set parameters for the solver
     * @param max_iterations Maximum number of iterations
     * @param epsilon Convergence tolerance
     */
    void set_parameters(unsigned int max_iterations = 1000,
                       double epsilon = 0.01);

    /**
     * @brief Run the exact SOT computation
     * @return true if successful, false otherwise
     */
    bool run();

    /**
     * @brief Get computed weights
     * @return Vector of computed weights
     */
    std::vector<double> get_weights() const;

    /**
     * @brief Get target points
     * @return Vector of target points
     */
    std::vector<dealii::Point<3>> get_target_points() const;

    /**
     * @brief Save results to files
     * @param weights_file Path to save weights
     * @param points_file Path to save points
     * @param io_coding File format for I/O operations (txt/bin)
     * @return true if successful, false otherwise
     */
    bool save_results(const std::string& weights_file,
                     const std::string& points_file,
                     const std::string& io_coding = "txt") const;

private:
    // Helper function to load volume mesh
    bool load_volume_mesh(const std::string& filename, GEO::Mesh& mesh);

    // Member variables
    std::unique_ptr<GEO::Mesh> source_mesh;
    GEO::vector<double> weights;
    std::vector<dealii::Point<3>> target_points;
    
    // Parameters
    unsigned int max_iterations_;
    double epsilon_;
};
