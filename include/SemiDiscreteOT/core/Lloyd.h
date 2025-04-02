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
    void run_lloyd();
    void run_sot_iteration(const unsigned int n_iter);
    void run_centroid_iteration();
    
    LloydParameterManager param_manager_lloyd;
private:
    Vector<double> potential;
};

#endif

