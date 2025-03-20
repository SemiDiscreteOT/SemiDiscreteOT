#include "SemiDiscreteOT/core/PowerDiagram.h"
#include "SemiDiscreteOT/utils/utils.h"
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_AABB.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/delaunay/delaunay.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <filesystem>

namespace PowerDiagramSpace {

template <int dim>
GeogramPowerDiagram<dim>::GeogramPowerDiagram(const std::string& source_mesh_file)
    : source_mesh(std::make_unique<GEO::Mesh>()),
      dimension_voronoi(dim + 1),
      RVD_mesh(std::make_unique<GEO::Mesh>())
{
    GEO::initialize();
    using namespace GEO;

    CmdLine::import_arg_group("standard");
    CmdLine::import_arg_group("algo");
    CmdLine::import_arg_group("opt");
    CmdLine::declare_arg("nb_iter", 1000, "number of iterations for OTM");
 
    CmdLine::declare_arg_group(
        "RVD", "RVD output options", CmdLine::ARG_ADVANCED);
    CmdLine::declare_arg("RVD", false, "save restricted Voronoi diagram");
    CmdLine::declare_arg(
        "RVD_iter", false, "save restricted Voronoi diagram at each iteration");
    CmdLine::declare_arg(
        "RVD:borders_only", false, "save only border of RVD");
    CmdLine::declare_arg(
        "RVD:integration_simplices", true, "export RVD as integration simplices");

    CmdLine::declare_arg("multilevel", true, "use multilevel algorithm");
    CmdLine::declare_arg("BRIO", true,
                         "use BRIO reordering to compute the levels");
    CmdLine::declare_arg("ratio", 0.125, "ratio between levels");
    CmdLine::declare_arg("epsilon", 0.01, "relative measure error in a cell");
    CmdLine::declare_arg(
        "lock", true, "Lock lower levels when sampling shape");
    CmdLine::declare_arg(
        "fitting_degree", 2, "degree for interpolating weights");
    CmdLine::declare_arg(
        "project", true, "project sampling on border");
    CmdLine::declare_arg(
        "feature_sensitive", true, "attempt to recover hard edges");
    CmdLine::declare_arg(
        "singular", false, "compute and save singular surface");
    CmdLine::set_arg("algo:delaunay", "BPOW");
    CmdLine::declare_arg(
        "recenter", true, "recenter target onto source mesh");
    CmdLine::declare_arg(
        "rescale", true, "rescale target to match source volume");
    
    if (!load_volume_mesh(source_mesh_file, *source_mesh)) {
        throw std::runtime_error("Failed to load source mesh: " + source_mesh_file);
    }
    
    source_mesh->vertices.set_dimension(dimension_voronoi);
}

template <int dim>
GeogramPowerDiagram<dim>::~GeogramPowerDiagram() = default;

template <int dim>
void GeogramPowerDiagram<dim>::set_generators(
    const std::vector<Point<dim>> &points,
    const Vector<double> &potentials)
{
    this->generator_points = points;
    this->generator_potentials.resize(potentials.size());
    for (unsigned int i = 0; i < potentials.size(); ++i) {
        this->generator_potentials[i] = potentials[i];
    }
    
    // Convert points to Geogram format
    points_mesh_target.clear();
    points_mesh_target.reserve(points.size() * dim);
    for (const auto& point : points) {
        for (unsigned int d = 0; d < dim; ++d) {
            points_mesh_target.push_back(point[d]);
        }
    }
    
    init_power_diagram();
}

template <int dim>
void GeogramPowerDiagram<dim>::init_power_diagram()
{
    const size_t nb_points = this->generator_points.size();
    
    // Initialize lifted points (dimension + 1)
    points_mesh_target_lifted.resize(nb_points * dimension_voronoi);
    
    // Copy first dim coordinates
    for (size_t i = 0; i < nb_points; ++i) {
        for (int d = 0; d < dim; ++d) {
            points_mesh_target_lifted[i * dimension_voronoi + d] = 
                points_mesh_target[i * dim + d];
        }
    }
    
    // Compute lifting coordinate based on potentials
    double W = 0.0;
    for (size_t i = 0; i < nb_points; ++i) {
        W = std::max(W, this->generator_potentials[i]);
    }
    
    for (size_t i = 0; i < nb_points; ++i) {
        points_mesh_target_lifted[dimension_voronoi * i + dim] = 
            ::sqrt(W - this->generator_potentials[i]);
    }
    
    // Create Delaunay triangulation
    delaunay = GEO::Delaunay::create(GEO::coord_index_t(dimension_voronoi), "BPOW");
    
    // Create Restricted Voronoi Diagram
    RVD = GEO::RestrictedVoronoiDiagram::create(delaunay, source_mesh.get());
    
    RVD->set_volumetric(true);
    RVD->set_check_SR(true);
    RVD->create_threads();
}

template <int dim>
void GeogramPowerDiagram<dim>::compute_power_diagram()
{
    if (!delaunay || !RVD) {
        throw std::runtime_error("Power diagram not initialized. Call set_generators first.");
    }
    
    delaunay->set_vertices(
        this->generator_points.size(), 
        points_mesh_target_lifted.data()
    );
    
    RVD->compute_RVD(
        *RVD_mesh,
        0,
        false,  // cells_borders_only
        true    // integration_simplices
    );
    
    if (save_RVD) {
        GEO::mesh_save(*RVD_mesh, "RVD.meshb");
    }
}

template <int dim>
void GeogramPowerDiagram<dim>::compute_cell_centroids()
{
    if (!RVD_mesh) {
        throw std::runtime_error("Power diagram not computed. Call compute_power_diagram first.");
    }
    
    GEO::Attribute<GEO::index_t> tet_region(RVD_mesh->cells.attributes(), "region");
    
    const size_t nb_vertices = RVD->delaunay()->nb_vertices();
    std::vector<GEO::vec3> centroids(nb_vertices, GEO::vec3(0.0, 0.0, 0.0));
    std::vector<double> volumes(nb_vertices, 0.0);
    
    // Compute weighted centroids
    std::cout << "Computing centroids..." << std::endl;
    for (GEO::index_t t = 0; t < RVD_mesh->cells.nb(); ++t) {
        const GEO::index_t v = tet_region[t];
        const GEO::index_t v0 = RVD_mesh->cells.tet_vertex(t, 0);
        const GEO::index_t v1 = RVD_mesh->cells.tet_vertex(t, 1);
        const GEO::index_t v2 = RVD_mesh->cells.tet_vertex(t, 2);
        const GEO::index_t v3 = RVD_mesh->cells.tet_vertex(t, 3);
        
        GEO::vec3 p0(RVD_mesh->vertices.point_ptr(v0));
        GEO::vec3 p1(RVD_mesh->vertices.point_ptr(v1));
        GEO::vec3 p2(RVD_mesh->vertices.point_ptr(v2));
        GEO::vec3 p3(RVD_mesh->vertices.point_ptr(v3));
        
        const double volume = GEO::Geom::tetra_signed_volume(p0, p1, p2, p3);
        centroids[v] += (volume / 4.0) * (p0 + p1 + p2 + p3);
        volumes[v] += volume;
    }
    
    // Normalize centroids and convert to Deal.II format
    this->cell_centroids.clear();
    for (size_t v = 0; v < nb_vertices; ++v) {
        if (volumes[v] != 0.0) {
            const double s = 1.0 / ::fabs(volumes[v]);
            const GEO::vec3& c = centroids[v] * s;
            this->cell_centroids.push_back(Point<dim>(c.x, c.y, c.z));
        }
    }
}

template <int dim>
void GeogramPowerDiagram<dim>::output_vtu(const std::string& /*filename*/) const
{
    // For now, we'll just save the RVD mesh if save_RVD is true
    // TODO: Implement proper VTU output if needed
}

template <int dim>
void GeogramPowerDiagram<dim>::save_centroids_to_file(const std::string& filename) const
{
    Utils::write_vector(this->cell_centroids, filename, "txt");
}

template <int dim>
const std::vector<Point<dim>>& GeogramPowerDiagram<dim>::get_cell_centroids() const
{
    return this->cell_centroids;
}

template <int dim>
bool GeogramPowerDiagram<dim>::load_volume_mesh(const std::string& filename, GEO::Mesh& mesh)
{
    GEO::MeshIOFlags flags;
    flags.set_element(GEO::MESH_CELLS);
    flags.set_attribute(GEO::MESH_CELL_REGION);
    
    if (!GEO::mesh_load(filename, mesh, flags)) {
        return false;
    }
    
    if (!mesh.cells.are_simplices()) {
        std::cerr << "File " << filename 
                 << " should only have tetrahedra" << std::endl;
        return false;
    }
    
    if (mesh.cells.nb() == 0) {
        std::cout << "File " << filename 
                 << " does not contain a volume" << std::endl;
        std::cout << "Trying to tetrahedralize..." << std::endl;
        if (!GEO::mesh_tetrahedralize(mesh, true, false)) {
            return false;
        }
    }
    return true;
}

// Explicit instantiation
template class GeogramPowerDiagram<2>;
template class GeogramPowerDiagram<3>;

} 