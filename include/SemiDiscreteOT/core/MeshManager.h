#ifndef MESH_MANAGER_H
#define MESH_MANAGER_H

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/numerics/data_out.h>

#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

using namespace dealii;

template <int dim, int spacedim = dim>
class MeshManager {
public:
    /**
     * @brief Constructor for MeshManager
     * @param comm MPI communicator
     */
    MeshManager(const MPI_Comm& comm);

    /**
     * @brief Generate mesh using deal.II grid generator
     * @param tria Triangulation to generate
     * @param grid_generator_function Name of grid generator function
     * @param grid_generator_arguments Arguments for grid generator
     * @param n_refinements Number of global refinements
     * @param use_tetrahedral_mesh Whether to convert to tetrahedral mesh
     */
    template<typename TriangulationType>
    void generate_mesh(TriangulationType& tria,
                      const std::string& grid_generator_function,
                      const std::string& grid_generator_arguments,
                      const unsigned int n_refinements,
                      const bool use_tetrahedral_mesh);

    /**
     * @brief Load source mesh from file into distributed triangulation
     * @param source_mesh Distributed triangulation to load into
     */
    void load_source_mesh(parallel::fullydistributed::Triangulation<dim, spacedim>& source_mesh);

    /**
     * @brief Load target mesh from file into serial triangulation
     * @param target_mesh Serial triangulation to load into
     */
    void load_target_mesh(Triangulation<dim, spacedim>& target_mesh);

    /**
     * @brief Load mesh at specific refinement level
     * @param source_mesh Distributed triangulation to load into
     * @param dof_handler_source DoF handler for the mesh
     * @param mesh_file Path to mesh file
     */
    void load_mesh_at_level(parallel::fullydistributed::Triangulation<dim, spacedim>& source_mesh,
                           DoFHandler<dim, spacedim>& dof_handler_source,
                           const std::string& mesh_file);

    /**
     * @brief Save source and target meshes to files
     * @param source_mesh Distributed source triangulation
     * @param target_mesh Serial target triangulation
     */
    void save_meshes(const parallel::fullydistributed::Triangulation<dim, spacedim>& source_mesh,
                    const Triangulation<dim, spacedim>& target_mesh);

    /**
     * @brief Write mesh to file in specified formats with optional cell data
     * @param mesh Triangulation to write
     * @param filepath Base path for output files (without extension)
     * @param formats Vector of output formats ("vtk", "msh", "vtu")
     * @param cell_data Optional vector of cell data to include
     * @param data_name Name for the cell data field
     * @return true if write successful, false otherwise
     */
    template<typename TriangulationType>
    bool write_mesh(const TriangulationType& mesh,
                   const std::string& filepath,
                   const std::vector<std::string>& formats,
                   const std::vector<double>* cell_data = nullptr,
                   const std::string& data_name = "cell_data");

    /**
     * @brief Get sorted list of mesh hierarchy files
     * @param dir Directory containing hierarchy files
     * @return Vector of mesh file paths, sorted coarsest to finest
     */
    static std::vector<std::string> get_mesh_hierarchy_files(
        const std::string& dir = "output/data_multilevel/source_multilevel");

private:
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Constants
    const std::string mesh_directory = "output/data_mesh";
};

// Include template implementations
#include "MeshManager.templates.h"

#endif // MESH_MANAGER_H 