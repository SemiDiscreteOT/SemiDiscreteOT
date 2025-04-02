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

template <int dim, int spacedim=dim>
class PowerDiagramBase {
public:
    virtual ~PowerDiagramBase() = default;
    
    virtual void set_generators(const std::vector<Point<spacedim>> &points,
                              const Vector<double> &potentials) = 0;
                       
    virtual void compute_power_diagram() = 0;
    
    virtual void output_vtu(const std::string& filename) const = 0;
    
    virtual void compute_cell_centroids() = 0;
    virtual void save_centroids_to_file(const std::string& filename) const = 0;
    virtual const std::vector<Point<spacedim>>& get_cell_centroids() const = 0;

protected:
    std::vector<Point<spacedim>> generator_points;
    std::vector<double> generator_potentials;
    std::vector<Point<spacedim>> cell_centroids;
};

template <int dim, int spacedim=dim>
class DealIIPowerDiagram : public PowerDiagramBase<dim, spacedim> {
public:
    DealIIPowerDiagram(const Triangulation<dim, spacedim> &source_mesh);
    
    void set_generators(const std::vector<Point<spacedim>> &points,
                       const Vector<double> &potentials) override;

    void set_distance_function(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& dist)
    {
        distance_function = dist;
    }
                       
    double power_distance(const Point<spacedim> &point,
                         const unsigned int generator_idx) const;
                         
    void compute_power_diagram() override;
    
    void output_vtu(const std::string& filename) const override;
    
    unsigned int get_cell_assignment(const unsigned int cell_index) const;
    const std::vector<unsigned int>& get_cell_assignments() const;
    
    void compute_cell_centroids() override;
    void save_centroids_to_file(const std::string& filename) const override;
    const std::vector<Point<spacedim>>& get_cell_centroids() const override;

private:
    const Triangulation<dim, spacedim>* source_triangulation;
    std::vector<unsigned int> cell_assignments;
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function;
};

// TODO enable if dim=3 ?
template <int dim, int spacedim = dim>
class GeogramPowerDiagram : public PowerDiagramBase<dim, spacedim> {
public:
    GeogramPowerDiagram(const std::string& source_mesh_file);
    ~GeogramPowerDiagram();
    
    void set_generators(const std::vector<Point<spacedim>> &points,
                       const Vector<double> &potentials) override;
                       
    void compute_power_diagram() override;
    
    void output_vtu(const std::string& filename) const override;
    
    void compute_cell_centroids() override;
    void save_centroids_to_file(const std::string& filename) const override;
    const std::vector<Point<spacedim>>& get_cell_centroids() const override;

private:
    void init_power_diagram();
    bool load_volume_mesh(const std::string& filename, GEO::Mesh& mesh);
    
    std::unique_ptr<GEO::Mesh> source_mesh;
    int dimension_voronoi;
    std::unique_ptr<GEO::Mesh> RVD_mesh;
    GEO::Delaunay_var delaunay;
    GEO::RestrictedVoronoiDiagram_var RVD;
    std::vector<double> points_mesh_target;
    std::vector<double> points_mesh_target_lifted;
    bool save_RVD = false;
    bool save_morph = false;
};

// Factory function to create appropriate PowerDiagram implementation
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