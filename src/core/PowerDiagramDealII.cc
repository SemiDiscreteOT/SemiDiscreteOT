#include "SemiDiscreteOT/core/PowerDiagram.h"
#include "SemiDiscreteOT/utils/utils.h"
#include <fstream>

namespace PowerDiagramSpace {

template <int dim>
DealIIPowerDiagram<dim>::DealIIPowerDiagram(const Triangulation<dim> &source_mesh)
    : source_triangulation(&source_mesh)
{}

template <int dim>
void DealIIPowerDiagram<dim>::set_generators(const std::vector<Point<dim>> &points,
                                          const Vector<double> &potentials)
{
    Assert(points.size() == potentials.size(),
           ExcDimensionMismatch(points.size(), potentials.size()));
    
    this->generator_points = points;
    this->generator_potentials.resize(potentials.size());
    for (unsigned int i = 0; i < potentials.size(); ++i)
        this->generator_potentials[i] = potentials[i];
}

template <int dim>
double DealIIPowerDiagram<dim>::power_distance(const Point<dim> &point,
                                            const unsigned int generator_idx) const
{
    Assert(generator_idx < this->generator_points.size(),
           ExcIndexRange(generator_idx, 0, this->generator_points.size()));
    const double squared_distance = 
        point.distance_square(this->generator_points[generator_idx]);
    return squared_distance - this->generator_potentials[generator_idx];
}

template <int dim>
void DealIIPowerDiagram<dim>::compute_power_diagram()
{
    cell_assignments.clear();
    cell_assignments.resize(source_triangulation->n_active_cells());
    
    for (const auto &cell : source_triangulation->active_cell_iterators())
    {
        const Point<dim> cell_center = cell->center();
        unsigned int closest_generator = 0;
        double min_power_distance = power_distance(cell_center, 0);
        
        for (unsigned int i = 1; i < this->generator_points.size(); ++i)
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
void DealIIPowerDiagram<dim>::output_vtu(const std::string& filename) const
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
unsigned int DealIIPowerDiagram<dim>::get_cell_assignment(const unsigned int cell_index) const
{
    Assert(cell_index < cell_assignments.size(),
           ExcIndexRange(cell_index, 0, cell_assignments.size()));
    return cell_assignments[cell_index];
}

template <int dim>
const std::vector<unsigned int>& DealIIPowerDiagram<dim>::get_cell_assignments() const
{
    return cell_assignments;
}

template <int dim>
void DealIIPowerDiagram<dim>::compute_cell_centroids()
{
    Assert(!cell_assignments.empty(),
           ExcMessage("Power diagram must be computed before calculating cell centroids."));
    
    const unsigned int n_cells = this->generator_points.size();
    this->cell_centroids.clear();
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
            this->cell_centroids.push_back(temp_centroids[i] / cell_volumes[i]);
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
void DealIIPowerDiagram<dim>::save_centroids_to_file(const std::string& filename) const
{
    Assert(!this->cell_centroids.empty(),
           ExcMessage("Cell centroids have not been computed. Call compute_cell_centroids() first."));
    
    // Use Utils::write_vector for centroid output
    Utils::write_vector(this->cell_centroids, filename, "txt");
}

template <int dim>
const std::vector<Point<dim>>& DealIIPowerDiagram<dim>::get_cell_centroids() const
{
    Assert(!this->cell_centroids.empty(),
           ExcMessage("Cell centroids have not been computed. Call compute_cell_centroids() first."));
    return this->cell_centroids;
}

// Explicit instantiation
template class DealIIPowerDiagram<2>;
template class DealIIPowerDiagram<3>;

} 