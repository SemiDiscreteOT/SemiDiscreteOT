#include "PowerDiagram.h"
#include <deal.II/base/mpi.h>
#include <fstream>

namespace PowerDiagramSpace {

template <int dim>
PowerDiagram<dim>::PowerDiagram(const parallel::distributed::Triangulation<dim> &source_mesh)
    : source_triangulation(&source_mesh)
    , mpi_communicator(source_mesh.get_communicator())
{}

template <int dim>
void PowerDiagram<dim>::set_generators(
    const std::vector<Point<dim>> &points,
    const std::vector<double> &weights)
{
    Assert(points.size() == weights.size(),
           ExcDimensionMismatch(points.size(), weights.size()));

    // Broadcast generator data from root process to all processes
    const unsigned int n_points = Utilities::MPI::max(points.size(), mpi_communicator);
    generator_points.resize(n_points);
    generator_weights.resize(n_points);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        generator_points = points;
        generator_weights = weights;
    }

    // Broadcast the data to all processes
    for (unsigned int i = 0; i < n_points; ++i)
    {
        std::vector<double> point_coords(dim);
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            for (unsigned int d = 0; d < dim; ++d)
                point_coords[d] = generator_points[i][d];
        }

        Utilities::MPI::broadcast(mpi_communicator, point_coords, 0);

        if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
        {
            for (unsigned int d = 0; d < dim; ++d)
                generator_points[i][d] = point_coords[d];
        }
    }

    Utilities::MPI::broadcast(mpi_communicator, generator_weights, 0);
}

template <int dim>
double PowerDiagram<dim>::power_distance(
    const Point<dim> &point,
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
    cell_assignments.resize(source_triangulation->n_locally_owned_active_cells());

    unsigned int local_cell_index = 0;
    for (const auto &cell :
         source_triangulation->active_cell_iterators() |
         IteratorFilters::LocallyOwnedCell())
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

        cell_assignments[local_cell_index++] = closest_generator;
    }
}

template <int dim>
void PowerDiagram<dim>::output_vtu(const std::string& filename) const
{
    DataOut<dim> data_out;
    data_out.attach_triangulation(*source_triangulation);

    Vector<double> cell_data(source_triangulation->n_active_cells());

    unsigned int local_cell_index = 0;
    for (const auto &cell :
         source_triangulation->active_cell_iterators() |
         IteratorFilters::LocallyOwnedCell())
    {
        cell_data[cell->active_cell_index()] = cell_assignments[local_cell_index++];
    }

    data_out.add_data_vector(cell_data, "power_region");
    data_out.build_patches();

    const std::string proc_filename =
        filename + "." +
        Utilities::int_to_string(source_triangulation->locally_owned_subdomain(), 4) +
        ".vtu";

    std::ofstream output_file(proc_filename.c_str());
    data_out.write_vtu(output_file);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
            filenames.push_back(filename + "." +
                              Utilities::int_to_string(i, 4) + ".vtu");

        std::ofstream master_output((filename + ".pvtu").c_str());
        data_out.write_pvtu_record(master_output, filenames);
    }
}

template <int dim>
unsigned int PowerDiagram<dim>::get_cell_assignment(
    const unsigned int cell_index) const
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

// Explicit instantiation
template class PowerDiagram<3>;

}
