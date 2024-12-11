#include "PowerDiagram.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

int main() {
    using namespace dealii;
    
    Triangulation<3> sphere_mesh;
    GridGenerator::hyper_ball(sphere_mesh);
    sphere_mesh.refine_global(2);

    FE_Q<3> fe(1);
    DoFHandler<3> dof_handler(sphere_mesh);
    dof_handler.distribute_dofs(fe);
    std::map<types::global_dof_index, Point<3>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<3>(), dof_handler, support_points);
    
    std::vector<Point<3>> generators;
    for (const auto& point_pair : support_points) {
        generators.push_back(point_pair.second);
    }
    
    std::vector<double> weights(generators.size(), 1.0/generators.size());
    
    Triangulation<3> triangulation;
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(4);
    
    PowerDiagramSpace::PowerDiagram<3> power_diagram(triangulation);
    power_diagram.set_generators(generators, weights);
    power_diagram.compute_power_diagram();
    power_diagram.output_vtu("power_diagram_3d.vtu");
    
    return 0;
}