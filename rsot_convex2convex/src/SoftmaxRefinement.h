#ifndef SOFTMAX_REFINEMENT_H
#define SOFTMAX_REFINEMENT_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/multithread_info.h>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <memory>
#include <vector>
#include <type_traits>

using namespace dealii;

template <int dim>
class SoftmaxRefinement {
public:
    // Scratch and copy data structures for parallel assembly
    struct ScratchData {
        ScratchData(const FiniteElement<dim> &fe,
                   const Mapping<dim> &mapping,
                   const Quadrature<dim> &quadrature)
            : fe_values(mapping, fe, quadrature,
                       update_values | update_quadrature_points | update_JxW_values),
              density_values(quadrature.size()) {}

        ScratchData(const ScratchData &scratch_data)
            : fe_values(scratch_data.fe_values.get_mapping(),
                       scratch_data.fe_values.get_fe(),
                       scratch_data.fe_values.get_quadrature(),
                       update_values | update_quadrature_points | update_JxW_values),
              density_values(scratch_data.density_values) {}

        FEValues<dim> fe_values;
        std::vector<double> density_values;
    };

    struct CopyData {
        Vector<double> weight_values;

        CopyData(const unsigned int n_target_points)
            : weight_values(n_target_points) {}
    };

    // Constructor
    SoftmaxRefinement(MPI_Comm mpi_comm,
                      const DoFHandler<dim>& dof_handler,
                      const Mapping<dim>& mapping,
                      const FiniteElement<dim>& fe,
                      const LinearAlgebra::distributed::Vector<double>& source_density,
                      unsigned int quadrature_order,
                      double distance_threshold);

    // Main computation method
    Vector<double> compute_refinement(
        const std::vector<Point<dim>>& target_points_fine,
        const Vector<double>& target_density_fine,
        const std::vector<Point<dim>>& target_points_coarse,
        const Vector<double>& target_density_coarse,
        const Vector<double>& weights_coarse,
        double regularization_param);

private:
    // MPI members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Source mesh and FE data
    const DoFHandler<dim>& dof_handler;
    const Mapping<dim>& mapping;
    const FiniteElement<dim>& fe;
    const LinearAlgebra::distributed::Vector<double>& source_density;
    const unsigned int quadrature_order;

    // Spatial search structures
    using IndexedPoint = std::pair<Point<dim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;
    RTree coarse_points_rtree;
    RTree fine_points_rtree;

    // Current computation state
    double current_lambda{0.0};
    const double current_distance_threshold;
    const std::vector<Point<dim>>* current_target_points_fine{nullptr};
    const Vector<double>* current_target_density_fine{nullptr};
    const std::vector<Point<dim>>* current_target_points_coarse{nullptr};
    const Vector<double>* current_target_density_coarse{nullptr};
    const Vector<double>* current_weights_coarse{nullptr};

    // Helper methods
    void setup_rtrees();
    std::vector<std::size_t> find_nearest_coarse_points(const Point<dim>& query_point) const;
    std::vector<std::size_t> find_nearest_fine_points(const Point<dim>& query_point) const;
    
    // Local assembly method
    void local_assemble(const typename DoFHandler<dim>::active_cell_iterator &cell,
                       ScratchData &scratch_data,
                       CopyData &copy_data);

    // Helper method to check if we're using simplex elements
    bool is_simplex_element() const
    {
        try {
            const auto* simplex_fe = dynamic_cast<const FE_SimplexP<dim>*>(&fe);
            return (simplex_fe != nullptr);
        }
        catch (...) {
            return false;
        }
    }
};

#endif // SOFTMAX_REFINEMENT_H 