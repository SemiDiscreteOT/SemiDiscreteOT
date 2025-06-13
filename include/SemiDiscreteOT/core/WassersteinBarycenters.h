#ifndef WASSERSTEINBARYCENTERS_H
#define WASSERSTEINBARYCENTERS_H

#include "SemiDiscreteOT/core/SemiDiscreteOT.h"
#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/solvers/SotSolver.h"

using namespace dealii;

enum class UpdateMode {
    TargetSupportPointsOnly,
    TargetMeasureOnly,
    Both
};

template <int dim, int spacedim = dim>
class WassersteinBarycenters
{
public:
    WassersteinBarycenters(
        const unsigned int n_measures,
        const std::vector<double> weights,
        const MPI_Comm &comm,
        UpdateMode update_flag = UpdateMode::TargetSupportPointsOnly);

    void run();
    void run_wasserstein_barycenters(
        const double absolute_threshold_measure=1e-8,
        const double absolute_threshold_support_points=1e-8,
        const unsigned int max_iterations=100,
        const double alpha=1.0);
    void run_sot_iterations(const unsigned int n_iter);
    void run_optimize_barycenters(
        const unsigned int n_iter,
        const double alpha);
    void compute_step_norm(
        const std::vector<std::pair<Point<spacedim>, double>>& barycenters_next,
        const std::vector<std::pair<Point<spacedim>, double>>& barycenters_prev,
        double &l2_norm_step_measure,
        double &l2_norm_step_support_points);

    UpdateMode update_mode = UpdateMode::Both;

    ConditionalOStream pcout;
    const unsigned int n_measures;
    const std::vector<double> weights;

    std::vector<std::unique_ptr<SemiDiscreteOT<dim, spacedim>>> sot_solvers;
    WassersteinBarycentersParameterManager param_manager_wasserstein_barycenters;

    std::vector<Vector<double>> potentials;
    std::vector<std::pair<Point<spacedim>, double>> barycenters;
    std::vector<std::pair<Point<spacedim>, double>> barycenters_prev;
    std::vector<std::pair<Vector<double>, double>> barycenters_gradients;
};

#endif

