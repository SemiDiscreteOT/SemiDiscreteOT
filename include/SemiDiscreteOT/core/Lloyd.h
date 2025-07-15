#ifndef LLOYD_H
#define LLOYD_H

#include "SemiDiscreteOT/core/SemiDiscreteOT.h"
#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/solvers/SotSolver.h"

using namespace dealii;

/**
 * @brief A class for performing Lloyd's algorithm for semi-discrete optimal transport.
 *
 * This class implements Lloyd's algorithm, which is an iterative method for
 * finding the optimal locations of the target points in a semi-discrete
 * optimal transport problem. It alternates between solving the dual problem
 * for the optimal transport potential and moving the target points to the
 * centroids of their power cells.
 *
 * @tparam dim The dimension of the source mesh.
 * @tparam spacedim The dimension of the space the mesh is embedded in.
 */
template <int dim, int spacedim = dim>
class Lloyd : public SemiDiscreteOT<dim, spacedim>
{
public:
    /**
     * @brief Constructor for the Lloyd class.
     * @param mpi_communicator The MPI communicator.
     */
    Lloyd(const MPI_Comm &mpi_communicator);
    /**
     * @brief Runs the Lloyd's algorithm.
     */
    void run();
    /**
     * @brief Runs the Lloyd's algorithm with the given parameters.
     * @param absolute_threshold The absolute threshold for the step norm.
     * @param max_iterations The maximum number of iterations.
     */
    void run_lloyd(
        const double absolute_threshold=1e-8,
        const unsigned int max_iterations=100);
    /**
     * @brief Runs a single iteration of the semi-discrete optimal transport solver.
     * @param n_iter The current iteration number.
     */
    void run_sot_iteration(const unsigned int n_iter);
    /**
     * @brief Runs a single iteration of the centroid computation.
     * @param n_iter The current iteration number.
     */
    void run_centroid_iteration(const unsigned int n_iter);
    /**
     * @brief Computes the L2 norm of the step.
     * @param barycenters_next The barycenters of the next iteration.
     * @param barycenters_prev The barycenters of the previous iteration.
     * @param l2_norm The L2 norm of the step.
     */
    void compute_step_norm(
        const std::vector<Point<spacedim>>& barycenters_next,
        const std::vector<Point<spacedim>>& barycenters_prev,
        double &l2_norm);
    
    LloydParameterManager param_manager_lloyd; ///< The parameter manager for the Lloyd's algorithm.
    std::vector<Point<spacedim>> barycenters; ///< The barycenters of the power cells.
    std::vector<Point<spacedim>> barycenters_prev; ///< The barycenters of the power cells in the previous iteration.
    Vector<double> potential; ///< The optimal transport potential.
};

#endif

