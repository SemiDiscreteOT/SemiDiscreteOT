#ifndef MESH_HIERARCHY_H
#define MESH_HIERARCHY_H

#include <geogram/basic/common.h>
#include <geogram/mesh/mesh.h>
#include <string>

namespace MeshHierarchy {

/**
 * @brief Class to manage a hierarchy of meshes with different resolutions
 */
class MeshHierarchyManager {
public:
    /**
     * @brief Constructor
     * @param min_vertices Minimum number of vertices for the coarsest level
     * @param max_vertices Maximum number of vertices for level 1 mesh
     */
    MeshHierarchyManager(int min_vertices = 1000, int max_vertices = 10000);

    /**
     * @brief Generate hierarchy of meshes from input mesh file
     * @param input_mesh_file Path to the input mesh file
     * @param output_dir Directory to save the mesh hierarchy
     * @param fill_volume Whether to fill volume with tetrahedra (true for 3D, false for 2D surface meshes)
     * @return Number of levels generated
     * @throws std::runtime_error if mesh loading or processing fails
     */
    int generateHierarchyFromFile(const std::string& input_mesh_file, const std::string& output_dir, bool fill_volume = true);

    /**
     * @brief Set the maximum number of vertices for level 1
     */
    void setMaxVertices(int max_vertices);

    /**
     * @brief Set the minimum number of vertices for coarsest level
     */
    void setMinVertices(int min_vertices);

    /**
     * @brief Get the number of levels in the last generated hierarchy
     */
    int getNumLevels() const;

private:
    int min_vertices_;
    int max_vertices_;
    int num_levels_;
    bool is_initialized_;

    /**
     * @brief Initialize Geogram if not already initialized
     */
    void initializeGeogram();

    /**
     * @brief Load a volume mesh from file
     * @param filename Path to the mesh file
     * @param M Reference to the Geogram mesh object
     * @param fill_volume Whether to fill volume with tetrahedra (true for 3D, false for 2D surface meshes)
     */
    bool loadVolumeMesh(const std::string& filename, GEO::Mesh& M, bool fill_volume) const;

    /**
     * @brief Calculate number of points for a given level
     */
    int getPointsForLevel(int base_points, int level) const;

    /**
     * @brief Ensure directory exists, create if it doesn't
     */
    void ensureDirectoryExists(const std::string& path) const;
};

} // namespace MeshHierarchy

#endif // MESH_HIERARCHY_H 