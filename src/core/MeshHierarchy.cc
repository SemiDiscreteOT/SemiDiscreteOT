#include "SemiDiscreteOT/core/MeshHierarchy.h"
#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/mesh/mesh_remesh.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace MeshHierarchy {

MeshHierarchyManager::MeshHierarchyManager(int min_vertices, int max_vertices)
    : min_vertices_(min_vertices)
    , max_vertices_(max_vertices)
    , num_levels_(0)
    , is_initialized_(false)
{}

void MeshHierarchyManager::initializeGeogram() {
    if (!is_initialized_) {
        GEO::initialize();
        GEO::CmdLine::import_arg_group("algo");
        GEO::CmdLine::set_arg("algo:delaunay", "BPOW");
        is_initialized_ = true;
    }
}

bool MeshHierarchyManager::loadVolumeMesh(const std::string& filename, GEO::Mesh& M) const {
    std::cout << "Loading volume mesh..." << std::endl;
    GEO::MeshIOFlags flags;
    flags.set_element(GEO::MESH_CELLS);
    flags.set_element(GEO::MESH_FACETS);
    flags.set_attribute(GEO::MESH_CELL_REGION);

    if(!mesh_load(filename, M, flags)) {
        return false;
    }

    if(!M.facets.are_simplices()) {
        std::cout << "Triangulating facets..." << std::endl;
        M.facets.triangulate();
    }

    if(!M.cells.are_simplices()) {
        std::cout << "Tetrahedralizing cells..." << std::endl;
        if(!mesh_tetrahedralize(M, true, true)) {
            return false;
        }
    }

    if(M.cells.nb() == 0) {
        std::cout << "File " << filename << " does not contain a volume" << std::endl;
        std::cout << "Trying to tetrahedralize..." << std::endl;
        if(!mesh_tetrahedralize(M, true, true)) {
            return false;
        }
    }
    return true;
}

void MeshHierarchyManager::setMaxVertices(int max_vertices) {
    max_vertices_ = max_vertices;
}

void MeshHierarchyManager::setMinVertices(int min_vertices) {
    min_vertices_ = min_vertices;
}

int MeshHierarchyManager::getNumLevels() const {
    return num_levels_;
}

void MeshHierarchyManager::ensureDirectoryExists(const std::string& path) const {
    if (!GEO::FileSystem::is_directory(path)) {
        GEO::FileSystem::create_directory(path);
    }
}

int MeshHierarchyManager::getPointsForLevel(int base_points, int level) const {
    // Level 0 always uses original mesh size
    if (level == 0) {
        return base_points;
    }
    // Level 1 starts with max_points (if smaller than base_points)
    if (level == 1) {
        return std::min(base_points, max_vertices_);
    }
    // For subsequent levels, use a reduction factor of 4 based on level 1's size
    int level1_points = std::min(base_points, max_vertices_);
    int points = static_cast<int>(level1_points / std::pow(4.0, level - 1));
    return std::max(points, min_vertices_);
}

int MeshHierarchyManager::generateHierarchyFromFile(const std::string& input_mesh_file, const std::string& output_dir) {
    // Initialize Geogram if needed
    initializeGeogram();

    // Load the input mesh
    GEO::Mesh input_mesh;
    if (!loadVolumeMesh(input_mesh_file, input_mesh)) {
        throw std::runtime_error("Failed to load input mesh: " + input_mesh_file);
    }

    // Create output directories
    std::string::size_type pos = 0;
    while ((pos = output_dir.find('/', pos + 1)) != std::string::npos) {
        ensureDirectoryExists(output_dir.substr(0, pos));
    }
    ensureDirectoryExists(output_dir);

    // Calculate number of levels
    const int total_vertices = input_mesh.vertices.nb();
    
    int surface_vertices = 0;
    std::vector<bool> visited(input_mesh.vertices.nb(), false);
    for(GEO::index_t c: input_mesh.facet_corners) {
        visited[input_mesh.facet_corners.vertex(c)] = true;
    }
    for(GEO::index_t v: input_mesh.vertices) {
        if(visited[v]) {
            ++surface_vertices;
        }
    }

    const int level1_vertices = std::min(surface_vertices, max_vertices_);
    num_levels_ = std::min(5, static_cast<int>(std::log(level1_vertices/min_vertices_) / std::log(4.0)) + 2);

    std::cout << "Initial mesh has " << total_vertices << " total vertices (" << surface_vertices << " surface vertices)" << std::endl;
    std::cout << "Level 1 will have maximum of " << max_vertices_ << " vertices" << std::endl;
    std::cout << "Creating " << num_levels_ << " levels of meshes" << std::endl;

    // Generate and save meshes for each level
    for(int level = 0; level < num_levels_; ++level) {
        GEO::Mesh level_mesh;
        
        if(level == 0) {
            // Always use original mesh for level 0
            level_mesh.copy(input_mesh);
            std::cout << "Level 0: using original mesh with " << surface_vertices << " surface vertices" << std::endl;
        } else {
            // For other levels, generate coarser versions
            int points_for_level = getPointsForLevel(surface_vertices, level);
            std::cout << "Level " << level << ": targeting " << points_for_level << " surface points" << std::endl;
            
            GEO::remesh_smooth(input_mesh, level_mesh, points_for_level);
            
            // Apply high-quality tetrahedralization
            GEO::MeshTetrahedralizeParameters params;
            params.refine = true;
            params.refine_quality = 1.0;
            mesh_tetrahedralize(level_mesh, params);
        }

        // Save the mesh for this level
        std::string output_file = output_dir + "/level_" + std::to_string(level) + ".msh";
        mesh_save(level_mesh, output_file);
        std::cout << "Saved level " << level << " mesh to " << output_file << std::endl;
    }

    return num_levels_;
}

} // namespace MeshHierarchy 