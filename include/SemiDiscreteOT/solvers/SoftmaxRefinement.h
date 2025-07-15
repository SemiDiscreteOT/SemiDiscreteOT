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

/**
 * @brief A class for refining the optimal transport potential using a softmax operation.
 *
 * This class implements a method for refining the optimal transport potential
 * from a coarser level to a finer level in a multilevel scheme. It uses a
 * softmax-like operation to distribute the potential from the coarse points
 * to their children in the finer level.
 *
 * @tparam dim The dimension of the source mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim = dim>
class SoftmaxRefinement {
public:
    /**
     * @brief Sets the distance function to be used.
     * @param dist The distance function.
     */
    void set_distance_function(
        const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& dist)
    {
        distance_function = dist;
    }

    /**
     * @brief A struct to hold scratch data for parallel assembly.
     */
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

        FEValues<dim, spacedim> fe_values; ///< FEValues object for the current cell.
        std::vector<double> density_values; ///< The density values at the quadrature points of the current cell.
    };

    /**
     * @brief A struct to hold copy data for parallel assembly.
     */
    struct CopyData {
        Vector<double> potential_values; ///< The potential values at the target points.

        CopyData(const unsigned int n_target_points)
            : potential_values(n_target_points) {}
    };

    /**
     * @brief Constructor for the SoftmaxRefinement class.
     * @param mpi_comm The MPI communicator.
     * @param dof_handler The DoF handler for the source mesh.
     * @param mapping The mapping for the source mesh.
     * @param fe The finite element for the source mesh.
     * @param source_density The density of the source measure.
     * @param quadrature_order The order of the quadrature rule to use for integration.
     * @param distance_threshold The distance threshold for the R-tree search.
     * @param use_log_sum_exp_trick Whether to use the log-sum-exp trick for numerical stability.
     */
    SoftmaxRefinement(MPI_Comm mpi_comm,
                      const DoFHandler<dim, spacedim>& dof_handler,
                      const Mapping<dim, spacedim>& mapping,
                      const FiniteElement<dim, spacedim>& fe,
                      const LinearAlgebra::distributed::Vector<double>& source_density,
                      unsigned int quadrature_order,
                      double distance_threshold,
                      bool use_log_sum_exp_trick = true);

    /**
     * @brief Computes the refined potential.
     * @param target_points_fine The points of the fine target measure.
     * @param target_density_fine The weights of the fine target measure.
     * @param target_points_coarse The points of the coarse target measure.
     * @param target_density_coarse The weights of the coarse target measure.
     * @param potential_coarse The potential at the coarse level.
     * @param regularization_param The regularization parameter.
     * @param current_level The current level in the multilevel hierarchy.
     * @param child_indices The indices of the children of each coarse point.
     * @return The refined potential.
     */
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
    MPI_Comm mpi_communicator; ///< The MPI communicator.
    const unsigned int n_mpi_processes; ///< The number of MPI processes.
    const unsigned int this_mpi_process; ///< The rank of the current MPI process.
    ConditionalOStream pcout; ///< A conditional output stream for parallel printing.
    
    // Distance function
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function; ///< The distance function.

    // Source mesh and FE data
    const DoFHandler<dim, spacedim>& dof_handler; ///< The DoF handler for the source mesh.
    const Mapping<dim, spacedim>& mapping; ///< The mapping for the source mesh.
    const FiniteElement<dim, spacedim>& fe; ///< The finite element for the source mesh.
    const LinearAlgebra::distributed::Vector<double>& source_density; ///< The density of the source measure.
    const unsigned int quadrature_order; ///< The order of the quadrature rule to use for integration.

    // Spatial search structure
    using IndexedPoint = std::pair<Point<spacedim>, std::size_t>;
    using RTreeParams = boost::geometry::index::rstar<8>;
    using RTree = boost::geometry::index::rtree<IndexedPoint, RTreeParams>;
    RTree target_points_rtree; ///< An R-tree for fast spatial queries on the target points.

    // Current computation state
    double current_lambda{0.0}; ///< The current regularization parameter.
    const double current_distance_threshold; ///< The current distance threshold.
    const bool use_log_sum_exp_trick; ///< Whether to use the log-sum-exp trick.
    const std::vector<Point<spacedim>>* current_target_points_fine{nullptr}; ///< The current fine target points.
    const Vector<double>* current_target_density_fine{nullptr}; ///< The current fine target density.
    const std::vector<Point<spacedim>>* current_target_points_coarse{nullptr}; ///< The current coarse target points.
    const Vector<double>* current_target_density_coarse{nullptr}; ///< The current coarse target density.
    const Vector<double>* current_potential_coarse{nullptr}; ///< The current coarse potential.
    const std::vector<std::vector<std::vector<size_t>>>* current_child_indices{nullptr}; ///< The current child indices.
    int current_level{0}; ///< The current level.

    // Helper methods
    /**
     * @brief Sets up the R-tree.
     */
    void setup_rtree();
    /**
     * @brief Finds the nearest target points to a query point.
     * @param query_point The query point.
     * @return A vector of indices of the nearest target points.
     */
    std::vector<std::size_t> find_nearest_target_points(const Point<spacedim>& query_point) const;
    
    /**
     * @brief Assembles the local contributions to the refined potential.
     * @param cell The current cell.
     * @param scratch_data The scratch data.
     * @param copy_data The copy data.
     */
    void local_assemble(const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
                       ScratchData &scratch_data,
                       CopyData &copy_data);

    /**
     * @brief Checks if the finite element is a simplex element.
     */
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