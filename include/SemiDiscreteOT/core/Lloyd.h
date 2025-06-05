#ifndef LLOYD_H
#define LLOYD_H

#include "SemiDiscreteOT/core/SemiDiscreteOT.h"
#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/solvers/SotSolver.h"

using namespace dealii;

template <int dim, int spacedim = dim>
class Lloyd : public SemiDiscreteOT<dim, spacedim>
{
public:
    Lloyd(const MPI_Comm &mpi_communicator);
    void run();
    void run_lloyd(
        const double absolute_threshold=1e-8,
        const unsigned int max_iterations=100);
    void run_sot_iteration(const unsigned int n_iter);
    void run_centroid_iteration(const unsigned int n_iter);
    void compute_step_norm(
        const std::vector<Point<spacedim>>& barycenters_next,
        const std::vector<Point<spacedim>>& barycenters_prev,
        double &l2_norm);
    
    LloydParameterManager param_manager_lloyd;
    std::vector<Point<spacedim>> barycenters;
    std::vector<Point<spacedim>> barycenters_prev;
    Vector<double> potential;
};

#endif

