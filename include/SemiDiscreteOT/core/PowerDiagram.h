#ifndef POWER_DIAGRAM_H
#define POWER_DIAGRAM_H

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/numerics/data_out.h>
#include <vector>
#include <string>
#include <deal.II/lac/vector.h>
#include <memory>
#include <geogram/basic/smart_pointer.h>
#include <geogram/delaunay/delaunay.h>
#include <geogram/voronoi/RVD.h>

#include "SemiDiscreteOT/solvers/SotSolver.h"

// Forward declarations for Geogram types
namespace GEO {
    class Mesh;
    class MeshCellsAABB;
}

namespace PowerDiagramSpace {

using namespace dealii;

/**
 * @brief A base class for power diagrams.
 *
 * This class provides a common interface for different implementations of
 * power diagrams, such as those based on deal.II or Geogram.
 *
 * @tparam dim The dimension of the mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim=dim>
class PowerDiagramBase {
public:
    virtual ~PowerDiagramBase() = default;
    
    /**
     * @brief Sets the generator points and their potentials.
     * @param points The generator points.
     * @param potentials The potentials of the generator points.
     */
    virtual void set_generators(const std::vector<Point<spacedim>> &points,
                              const Vector<double> &potentials) = 0;
                       
    /**
     * @brief Computes the power diagram.
     */
    virtual void compute_power_diagram() = 0;
    
    /**
     * @brief Outputs the power diagram to a VTU file.
     * @param filename The name of the output file.
     */
    virtual void output_vtu(const std::string& filename) const = 0;
    
    /**
     * @brief Computes the centroids of the power cells.
     */
    virtual void compute_cell_centroids() = 0;
    /**
     * @brief Saves the centroids of the power cells to a file.
     * @param filename The name of the output file.
     */
    virtual void save_centroids_to_file(const std::string& filename) const = 0;
    /**
     * @brief Returns the centroids of the power cells.
     */
    virtual const std::vector<Point<spacedim>>& get_cell_centroids() const = 0;

protected:
    std::vector<Point<spacedim>> generator_points; ///< The generator points.
    std::vector<double> generator_potentials; ///< The potentials of the generator points.
    std::vector<Point<spacedim>> cell_centroids; ///< The centroids of the power cells.
};

/**
 * @brief A class for computing power diagrams using deal.II.
 *
 * @tparam dim The dimension of the mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim=dim>
class DealIIPowerDiagram : public PowerDiagramBase<dim, spacedim> {
public:
    /**
     * @brief Constructor for the DealIIPowerDiagram class.
     * @param source_mesh The source mesh.
     */
    DealIIPowerDiagram(const Triangulation<dim, spacedim> &source_mesh);
    
    /**
     * @brief Sets the generator points and their potentials.
     * @param points The generator points.
     * @param potentials The potentials of the generator points.
     */
    void set_generators(const std::vector<Point<spacedim>> &points,
                       const Vector<double> &potentials) override;

    /**
     * @brief Sets the distance function to be used.
     * @param dist The distance function.
     */
    void set_distance_function(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& dist)
    {
        distance_function = dist;
    }
                       
    /**
     * @brief Computes the power distance between a point and a generator.
     * @param point The point.
     * @param generator_idx The index of the generator.
     * @return The power distance.
     */
    double power_distance(const Point<spacedim> &point,
                         const unsigned int generator_idx) const;
                         
    /**
     * @brief Computes the power diagram.
     */
    void compute_power_diagram() override;
    
    /**
     * @brief Outputs the power diagram to a VTU file.
     * @param filename The name of the output file.
     */
    void output_vtu(const std::string& filename) const override;
    
    /**
     * @brief Returns the assignment of a cell to a generator.
     * @param cell_index The index of the cell.
     * @return The index of the generator.
     */
    unsigned int get_cell_assignment(const unsigned int cell_index) const;
    /**
     * @brief Returns the assignments of all cells to generators.
     */
    const std::vector<unsigned int>& get_cell_assignments() const;
    
    /**
     * @brief Computes the centroids of the power cells.
     */
    void compute_cell_centroids() override;
    /**
     * @brief Saves the centroids of the power cells to a file.
     * @param filename The name of the output file.
     */
    void save_centroids_to_file(const std::string& filename) const override;
    /**
     * @brief Returns the centroids of the power cells.
     */
    const std::vector<Point<spacedim>>& get_cell_centroids() const override;

private:
    const Triangulation<dim, spacedim>* source_triangulation; ///< The source triangulation.
    std::vector<unsigned int> cell_assignments; ///< The assignments of cells to generators.
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function; ///< The distance function.
};

// TODO enable if dim=3 ?
/**
 * @brief A class for computing power diagrams using Geogram.
 *
 * @tparam dim The dimension of the mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim = dim>
class GeogramPowerDiagram : public PowerDiagramBase<dim, spacedim> {
public:
    /**
     * @brief Constructor for the GeogramPowerDiagram class.
     * @param source_mesh_file The path to the source mesh file.
     */
    GeogramPowerDiagram(const std::string& source_mesh_file);
    ~GeogramPowerDiagram();
    
    /**
     * @brief Sets the generator points and their potentials.
     * @param points The generator points.
     * @param potentials The potentials of the generator points.
     */
    void set_generators(const std::vector<Point<spacedim>> &points,
                       const Vector<double> &potentials) override;
                       
    /**
     * @brief Computes the power diagram.
     */
    void compute_power_diagram() override;
    
    /**
     * @brief Outputs the power diagram to a VTU file.
     * @param filename The name of the output file.
     */
    void output_vtu(const std::string& filename) const override;
    
    /**
     * @brief Computes the centroids of the power cells.
     */
    void compute_cell_centroids() override;
    /**
     * @brief Saves the centroids of the power cells to a file.
     * @param filename The name of the output file.
     */
    void save_centroids_to_file(const std::string& filename) const override;
    /**
     * @brief Returns the centroids of the power cells.
     */
    const std::vector<Point<spacedim>>& get_cell_centroids() const override;

private:
    void init_power_diagram();
    bool load_volume_mesh(const std::string& filename, GEO::Mesh& mesh);
    
    std::unique_ptr<GEO::Mesh> source_mesh; ///< The source mesh.
    int dimension_voronoi; ///< The dimension of the Voronoi diagram.
    std::unique_ptr<GEO::Mesh> RVD_mesh; ///< The restricted Voronoi diagram mesh.
    GEO::Delaunay_var delaunay; ///< The Delaunay triangulation.
    GEO::RestrictedVoronoiDiagram_var RVD; ///< The restricted Voronoi diagram.
    std::vector<double> points_mesh_target; ///< The target points.
    std::vector<double> points_mesh_target_lifted; ///< The lifted target points.
    bool save_RVD = false; ///< A flag to indicate whether to save the restricted Voronoi diagram.
    bool save_morph = false; ///< A flag to indicate whether to save the morphed mesh.
};

/**
 * @brief A factory function to create a power diagram.
 *
 * @tparam dim The dimension of the mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 * @param implementation_type The type of implementation to use ("dealii" or "geogram").
 * @param dealii_mesh A pointer to a deal.II triangulation (if using the "dealii" implementation).
 * @param geogram_mesh_file The path to a Geogram mesh file (if using the "geogram" implementation).
 * @return A unique pointer to a PowerDiagramBase object.
 */
template <int dim, int spacedim>
std::unique_ptr<PowerDiagramBase<dim, spacedim>> create_power_diagram(
    const std::string& implementation_type,
    const Triangulation<dim, spacedim>* dealii_mesh = nullptr,
    const std::string& geogram_mesh_file = "") {
    
    if (implementation_type == "dealii" && dealii_mesh != nullptr) {
        return std::make_unique<DealIIPowerDiagram<dim, spacedim>>(*dealii_mesh);
    } else if (implementation_type == "geogram" && !geogram_mesh_file.empty()) {
        if constexpr (dim==3 && dim==spacedim)
            return std::make_unique<GeogramPowerDiagram<dim, spacedim>>(geogram_mesh_file);
        else
            throw std::runtime_error("Geogram power diagram is only available for dim==spacedim");
    }
    throw std::runtime_error("Invalid power diagram implementation type or missing required parameters");
}

}  // namespace PowerDiagramSpace

#endif