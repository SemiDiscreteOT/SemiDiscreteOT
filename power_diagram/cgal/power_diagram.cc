#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/numerics/data_out.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>

using namespace dealii;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_3<K> RT;
typedef RT::Weighted_point Weighted_point;
typedef K::Point_3 CGALPoint;

int main() {
    // Read mesh using deal.II
    Triangulation<3> triangulation;
    GridIn<3> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file("data/target.vtk");
    grid_in.read_vtk(input_file);

    // Setup CGAL power diagram
    std::vector<Weighted_point> generators = {
        Weighted_point(CGALPoint(0,0,0), 0.1),
        Weighted_point(CGALPoint(1,1,1), 0.2)
    };
    RT rt;
    rt.insert(generators.begin(), generators.end());

    // Process cells
    Vector<double> cell_markers(triangulation.n_active_cells());
    unsigned int cell_index = 0;
    for (const auto &cell : triangulation.active_cell_iterators()) {
        Point<3> center = cell->center();
        CGALPoint cgal_center(center[0], center[1], center[2]);
        RT::Vertex_handle nearest = rt.nearest_power_vertex(cgal_center);
        cell_markers[cell_index++] = std::distance(generators.begin(),
            std::find(generators.begin(), generators.end(), nearest->point()));
    }

    // Output
    std::ofstream output_file("power_diagram_3d.vtu");
    DataOut<3> data_out;
    data_out.attach_triangulation(triangulation);
    data_out.add_data_vector(cell_markers, "power_region");
    data_out.build_patches();
    data_out.write_vtu(output_file);

    return 0;
}
