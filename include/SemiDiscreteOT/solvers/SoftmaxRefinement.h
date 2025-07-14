#ifndef SOFTMAX_REFINEMENT_H
#define SOFTMAX_REFINEMENT_H

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <memory>
#include <vector>
#include <type_traits>

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

#include "SemiDiscreteOT/solvers/SotSolver.h"

using namespace dealii;

template <int dim, int spacedim = dim>
class SoftmaxRefinement {
public:
    void set_distance_function(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& dist)
    {
        distance_function = dist;
    }

    // Scratch and copy data structures for parallel assembly
    struct ScratchData {
        ScratchData(const FiniteElement<dim, spacedim> &fe,
                   const Mapping<dim, spacedim> &mapping,
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

        FEValues<dim, spacedim> fe_values;
        std::vector<double> density_values;
    };

    struct CopyData {
        Vector<double> potential_values;

        CopyData(const unsigned int n_target_points)
            : potential_values(n_target_points) {}
    };

    // Constructor
    SoftmaxRefinement(MPI_Comm mpi_comm,
                      const DoFHandler<dim, spacedim>& dof_handler,
                      const Mapping<dim, spacedim>& mapping,
                      const FiniteElement<dim, spacedim>& fe,
                      const LinearAlgebra::distributed::Vector<double>& source_density,
                      unsigned int quadrature_order,
                      double distance_threshold,
                      bool use_log_sum_exp_trick = true);

    // Main computation method
    Vector<double> compute_refinement(
        const std::vector<Point<spacedim>>& target_points_fine,
        const Vector<double>& target_density_fine,
        const std::vector<Point<spacedim>>& target_points_coarse,
        const Vector<double>& target_density_coarse,
        const Vector<double>& potential_coarse,
        double regularization_param,
        int current_level,
        const std::vector<std::vector<std::vector<size_t>>>& child_indices);

private:
    // MPI members
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;
    
    // Distance function
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function;

    // Source mesh and FE data
    const DoFHandler<dim, spacedim>& dof_handler;
    const Mapping<dim, spacedim>& mapping;
    const FiniteElement<dim, spacedim>& fe;
    const LinearAlgebra::distributed::Vector<double>& source_density;
    const unsigned int quadrature_order;

    // Spatial search structure
    using IndexedPoint = std::pair<Point<spacedim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;
    RTree target_points_rtree;

    // Current computation state
    double current_lambda{0.0};
    const double current_distance_threshold;
    const bool use_log_sum_exp_trick;
    const std::vector<Point<spacedim>>* current_target_points_fine{nullptr};
    const Vector<double>* current_target_density_fine{nullptr};
    const std::vector<Point<spacedim>>* current_target_points_coarse{nullptr};
    const Vector<double>* current_target_density_coarse{nullptr};
    const Vector<double>* current_potential_coarse{nullptr};
    const std::vector<std::vector<std::vector<size_t>>>* current_child_indices{nullptr};
    int current_level{0};

    // Helper methods
    void setup_rtree();
    std::vector<std::size_t> find_nearest_target_points(const Point<spacedim>& query_point) const;
    
    // Local assembly method
    void local_assemble(const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
                       ScratchData &scratch_data,
                       CopyData &copy_data);

    // Helper method to check if we're using simplex elements
    bool is_simplex_element() const
    {
        try {
            const auto* simplex_fe = dynamic_cast<const FE_SimplexP<dim, spacedim>*>(&fe);
            return (simplex_fe != nullptr);
        }
        catch (...) {
            return false;
        }
    }
};

#endif // SOFTMAX_REFINEMENT_H 