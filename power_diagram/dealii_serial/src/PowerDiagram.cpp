#include "PowerDiagram.h"
#include <fstream>

namespace PowerDiagramSpace {

template <int dim>
PowerDiagram<dim>::PowerDiagram(const Triangulation<dim> &source_mesh)
    : source_triangulation(&source_mesh)
{}

template <int dim>
void PowerDiagram<dim>::set_generators(const std::vector<Point<dim>> &points,
                                     const std::vector<double> &weights)
{
    Assert(points.size() == weights.size(),
           ExcDimensionMismatch(points.size(), weights.size()));
    
    generator_points = points;
    generator_weights = weights;
}

template <int dim>
double PowerDiagram<dim>::power_distance(const Point<dim> &point,
                                       const unsigned int generator_idx) const
{
    Assert(generator_idx < generator_points.size(),
           ExcIndexRange(generator_idx, 0, generator_points.size()));
    const double squared_distance = 
        point.distance_square(generator_points[generator_idx]);
    return squared_distance - generator_weights[generator_idx];
}

template <int dim>
void PowerDiagram<dim>::compute_power_diagram()
{
    cell_assignments.clear();
    cell_assignments.resize(source_triangulation->n_active_cells());
    
    for (const auto &cell : source_triangulation->active_cell_iterators())
    {
        const Point<dim> cell_center = cell->center();
        unsigned int closest_generator = 0;
        double min_power_distance = power_distance(cell_center, 0);
        
        for (unsigned int i = 1; i < generator_points.size(); ++i)
        {
            const double current_power_distance = 
                power_distance(cell_center, i);
            if (current_power_distance < min_power_distance)
            {
                min_power_distance = current_power_distance;
                closest_generator = i;
            }
        }
        cell_assignments[cell->active_cell_index()] = closest_generator;
    }
}

template <int dim>
void PowerDiagram<dim>::output_vtu(const std::string& filename) const
{
    DataOut<dim> data_out;
    data_out.attach_triangulation(*source_triangulation);
    
    Vector<double> cell_data(cell_assignments.begin(), cell_assignments.end());
    data_out.add_data_vector(cell_data, "power_region");
    
    data_out.build_patches();
    std::ofstream output_file(filename);
    data_out.write_vtu(output_file);
}

template <int dim>
unsigned int PowerDiagram<dim>::get_cell_assignment(const unsigned int cell_index) const
{
    Assert(cell_index < cell_assignments.size(),
           ExcIndexRange(cell_index, 0, cell_assignments.size()));
    return cell_assignments[cell_index];
}

template <int dim>
const std::vector<unsigned int>& PowerDiagram<dim>::get_cell_assignments() const
{
    return cell_assignments;
}

template class PowerDiagram<3>;

}