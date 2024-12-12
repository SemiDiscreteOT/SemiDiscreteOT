#include "PowerDiagram.h"
#include "utils.h"
#include <fstream>

namespace PowerDiagramSpace {

template <int dim>
PowerDiagram<dim>::PowerDiagram(const Triangulation<dim> &source_mesh)
    : source_triangulation(&source_mesh)
{}

template <int dim>
void PowerDiagram<dim>::set_generators(const std::vector<Point<dim>> &points,
                                     const Vector<double> &weights)
{
    Assert(points.size() == weights.size(),
           ExcDimensionMismatch(points.size(), weights.size()));
    
    generator_points = points;
    generator_weights.resize(weights.size());
    for (unsigned int i = 0; i < weights.size(); ++i)
        generator_weights[i] = weights[i];
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
    // Convert cell_assignments to vector<double> for compatibility
    std::vector<double> cell_data(cell_assignments.begin(), cell_assignments.end());
    
    // Use Utils::write_mesh with VTU format
    Utils::write_mesh(*source_triangulation, 
                     filename, 
                     std::vector<std::string>{"vtu"}, 
                     &cell_data,
                     "power_region");
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

template <int dim>
void PowerDiagram<dim>::compute_cell_centroids()
{
    Assert(!cell_assignments.empty(),
           ExcMessage("Power diagram must be computed before calculating cell centroids."));
    
    const unsigned int n_cells = generator_points.size();
    cell_centroids.clear();
    std::vector<Point<dim>> temp_centroids(n_cells, Point<dim>());
    std::vector<double> cell_volumes(n_cells, 0.0);
    unsigned int empty_cells = 0;
    
    // Compute weighted sum of mesh element centers for each power cell
    for (const auto &element : source_triangulation->active_cell_iterators())
    {
        const unsigned int cell_idx = cell_assignments[element->active_cell_index()];
        const double volume = element->measure();
        temp_centroids[cell_idx] += element->center() * volume;
        cell_volumes[cell_idx] += volume;
    }
    
    // Store only valid centroids
    for (unsigned int i = 0; i < n_cells; ++i)
    {
        if (cell_volumes[i] > 0.0)
        {
            cell_centroids.push_back(temp_centroids[i] / cell_volumes[i]);
        }
        else
        {
            empty_cells++;
        }
    }
    
    std::cout << "Found " << empty_cells << " empty power cells out of " 
              << n_cells << " generators." << std::endl;
}

template <int dim>
void PowerDiagram<dim>::save_centroids_to_file(const std::string& filename) const
{
    Assert(!cell_centroids.empty(),
           ExcMessage("Cell centroids have not been computed. Call compute_cell_centroids() first."));
    
    // Use Utils::write_vector for centroid output
    Utils::write_vector(cell_centroids, filename, "txt");
}

template <int dim>
const std::vector<Point<dim>>& PowerDiagram<dim>::get_cell_centroids() const
{
    Assert(!cell_centroids.empty(),
           ExcMessage("Cell centroids have not been computed. Call compute_cell_centroids() first."));
    return cell_centroids;
}

template class PowerDiagram<2>;
template class PowerDiagram<3>;

}