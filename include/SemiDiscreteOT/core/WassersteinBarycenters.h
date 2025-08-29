#ifndef WASSERSTEINBARYCENTERS_H
#define WASSERSTEINBARYCENTERS_H

#include "SemiDiscreteOT/core/SemiDiscreteOT.h"
#include "SemiDiscreteOT/utils/ParameterManager.h"
#include "SemiDiscreteOT/solvers/SotSolver.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

enum class UpdateMode {
    TargetSupportOnly,
    TargetMeasureOnly
};
#include "SemiDiscreteOT/core/wasserstein_solver_bfgs.h"

using namespace dealii;

// Boost geometry types for rtree
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using BoostPoint = bg::model::point<double, 3, bg::cs::cartesian>;
using IndexedPoint = std::pair<BoostPoint, std::size_t>;

// Parameter classes using ParameterAcceptor
class BarycenterParameters : public ParameterAcceptor
{
public:
    BarycenterParameters() : ParameterAcceptor("Barycenter")
    {
        add_parameter("volume_scaling", volume_scaling,
                     "Volume scaling");
        add_parameter("max_iterations", max_iterations,
                     "Maximum number of barycenter iterations");
        add_parameter("convergence_tolerance", convergence_tolerance,
                     "Convergence tolerance for barycenter iterations");
        add_parameter("weights", weights,
                     "Weights of Wasserstein barycenters");
        add_parameter("output_frequency", output_frequency,
                     "Frequency of output (every N iterations)");
        add_parameter("silence_output", silence_output,
                     "Silence SotSolver output");
    }

    unsigned int max_iterations = 100;
    double convergence_tolerance = 1e-6;
    std::vector<double> weights = {0.5, 0.5};
    bool volume_scaling = false;
    unsigned int output_frequency = 5;
    bool silence_output = true;
};

class OptimalTransportParameters : public ParameterAcceptor
{
public:
    OptimalTransportParameters() : ParameterAcceptor("Optimal Transport")
    {
        add_parameter("epsilon", epsilon,
                     "Regularization parameter");
        add_parameter("distance_threshold", distance_threshold,
                     "Distance threshold for computational efficiency");
        add_parameter("tau", tau,
                     "Numerical stability parameter");
        add_parameter("max_iterations", max_iterations,
                     "Maximum iterations for OT solver");
        add_parameter("quadrature_order", quadrature_order,
                     "Quadrature order for integration");
        add_parameter("tolerance", tolerance,
                     "Tolerance for OT solver");
        add_parameter("use_log_sum_exp_trick", use_log_sum_exp_trick,
                     "Use log-sum-exp trick for numerical stability");
        add_parameter("verbose_output", verbose_output,
                     "Enable verbose output for OT solver");
        add_parameter("timer_output", timer_output,
                     "Enable verbose timer output for OT solver");
        add_parameter("distance_threshold_type", distance_threshold_type,
                     "Type of distance threshold");
        add_parameter("source_multilevel_enabled", source_multilevel_enabled,
                     "Enable multilevel for source");
        add_parameter("target_multilevel_enabled", target_multilevel_enabled,
                     "Enable multilevel for target");

        add_parameter("source_min_vertices", source_min_vertices,
                     "Minimum number of vertices for source multilevel");
        add_parameter("source_max_vertices", source_max_vertices,
                     "Maximum number of vertices for source multilevel");
        add_parameter("target_min_points", target_min_points,
                     "Minimum number of points for target multilevel");
        add_parameter("target_max_points", target_max_points,
                     "Maximum number of points for target multilevel");

        add_parameter("use_python_clustering", use_python_clustering,
                     "Whether to use Python scripts for clustering");
        add_parameter("python_script_name", python_script_name,
                     "Name of the Python script to use");
    }

    double epsilon = 1e-2;
    double distance_threshold = 1.5;
    double tau = 1e-12;
    unsigned int max_iterations = 1000;
    unsigned int quadrature_order = 3;
    double tolerance = 1e-3;
    bool use_log_sum_exp_trick = true;
    bool verbose_output = false;
    bool timer_output = false;
    std::string distance_threshold_type = "pointwise";
    bool source_multilevel_enabled = false;
    bool target_multilevel_enabled = false;
    unsigned int source_min_vertices = 100;
    unsigned int source_max_vertices = 500;
    unsigned int target_min_points = 100;
    unsigned int target_max_points = 1000;
    bool use_python_clustering = true;
    std::string python_script_name = "multilevel_clustering_faiss_cpu.py";
};

class FileParameters : public ParameterAcceptor
{
public:
    FileParameters() : ParameterAcceptor("Files")
    {
        add_parameter("source_filenames", source_filenames,
                     "Filenames for source meshes");
        add_parameter("output_prefix", output_prefix,
                     "Prefix for output files");
        add_parameter("save_vtk", save_vtk,
                     "Save results in VTK format");
        add_parameter("save_txt", save_txt,
                     "Save results in text format");
    }

    std::vector<std::string> source_filenames = {"source.msh", "source.msh"};
    std::string output_prefix = "barycenter";
    bool save_vtk = true;
    bool save_txt = true;
};

template <int dim, int spacedim>
struct BarycenterScratchData
{
    BarycenterScratchData(const FiniteElement<dim, spacedim> &fe,
                          const Mapping<dim, spacedim> &mapping,
                          const unsigned int quadrature_degree)
        : fe_values(mapping, fe, QGauss<dim>(quadrature_degree),
                    update_values | update_quadrature_points | update_JxW_values),
          density_values(fe_values.get_quadrature().size())
    {
    }

    BarycenterScratchData(const BarycenterScratchData<dim, spacedim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                    scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    scratch_data.fe_values.get_update_flags()),
          density_values(scratch_data.density_values.size())
    {
    }

    FEValues<dim, spacedim> fe_values;
    std::vector<double> density_values;
};

template <int spacedim>
struct BarycenterCopyData
{
    BarycenterCopyData(const unsigned int grad_size)
    {
        grads.reinit(grad_size);
    }

    Vector<double> grads;
    double value = 0.0; ///< The value of the functional.
};

template <int dim, int spacedim = dim, UpdateMode update_flag = UpdateMode::TargetMeasureOnly>
class WassersteinBarycenters
{
public:
    // Parameter objects
    BarycenterParameters barycenter_params;
    OptimalTransportParameters ot_params;
    FileParameters file_params;

    // Distance function for Barycenters
    std::string distance_name; ///< The name of the distance function.
    std::function<double(const Point<spacedim>&, const Point<spacedim>&)> distance_function; ///< The distance function.
    std::function<Vector<double>(const Point<spacedim>&, const Point<spacedim>&)> distance_function_gradient; ///< The gradient of the distance function.
    std::function<Point<spacedim>(const Point<spacedim>&, const Vector<double>&)> distance_function_exponential_map; ///< The exponential map of the distance function.

    WassersteinBarycenters(const MPI_Comm &comm);

    void run_wasserstein_barycenters();
    void configure();

    void local_assemble_barycenter_gradient(
        const SotSolver<dim, spacedim> &sot_pb,
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
        BarycenterScratchData<dim, spacedim> &scratch,
        BarycenterCopyData<spacedim> &copy,
        const LinearAlgebra::distributed::Vector<double, MemorySpace::Host> &source_density,
        const std::vector<Point<spacedim>> &target_points,
        const Vector<double> &target_weights,
        const Vector<double> &potentials,
        const double epsilon,
        const double distance_threshold);

    void set_distance_function(const std::string &distance_name);

    void init_sol(
        Vector<double> &w,
        const std::vector<Point<spacedim>> &bpoints,
        const Vector<double> &bweights);

    void update_barycenter(
        const Vector<double> &w,
        std::vector<Point<spacedim>> &bpoints,
        Vector<double> &bweights,
        const bool reinit_tree=true);

    double evaluate_functional(
        const Vector<double>& w, Vector<double>& grad);

    void save_vtk_output(
        const std::vector<Point<spacedim>> &points,
    const Vector<double> &weights,
    const std::string &filename) const;

    void save_vtk_output(
        const std::vector<Point<spacedim>> &points,
    const Vector<double> &weights,
    const Vector<double> &grad,
    const std::string &filename) const;

    ConditionalOStream pcout;
    MPI_Comm mpi_comm;
    unsigned int n_measures;
    std::vector<double> weights;

    std::vector<std::unique_ptr<SemiDiscreteOT<dim, spacedim>>> sot_problems;
    std::vector<Vector<double>> potentials;

    unsigned int grad_size;
    std::vector<Point<spacedim>> barycenter_points;
    Vector<double> barycenter_weights;

    Vector<double> sol;
    Vector<double> sol_grad;
    double functional_value;

    bgi::rtree<IndexedPoint, bgi::rstar<16>> target_rtree;

    /**
     * @brief A verbose solver control class that prints the progress of the solver.
     */
    class VerboseSolverControl : public SolverControl
    {
    public:
        VerboseSolverControl(
            unsigned int n, double tol, ConditionalOStream& pcout_)
            : SolverControl(n, tol)
            , pcout(pcout_)
            , bweights(nullptr)
            , bsupport(nullptr)
        {}

        void set_barycenters(
            const Vector<double> &grad,
            const std::vector<Point<spacedim>> &barycenter_points,
            const Vector<double> &barycenter_weights,
            const double &functional_value) {
            gradient = &grad;
            bsupport = &barycenter_points;
            bweights = &barycenter_weights;
            fvalue = &functional_value;

            bweights_prev.reinit(barycenter_weights.size());
            bsupport_prev.resize(barycenter_points.size(), Point<spacedim>());
        }

        virtual State check(unsigned int step, double value) override
        {
            AssertThrow(bweights != nullptr,
                        ExcMessage("bweights pointer not set in VerboseSolverControl"));
            AssertThrow(bsupport != nullptr,
                        ExcMessage("bsupport pointer not set in VerboseSolverControl"));

            double check_value = 0.0;
            std::string check_description;
            std::string color;

            if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
            {
                for (unsigned int i = 0; i < bweights->size(); ++i){
                    check_value +=
                        std::pow((*bweights)[i] - (bweights_prev)[i], 2);
                    bweights_prev[i] = (*bweights)[i];
                }
                check_value = std::sqrt(check_value);
            }
            else if constexpr (update_flag == UpdateMode::TargetSupportOnly)
            {
                check_value = 0.0;
                for (unsigned int i = 0; i < bsupport->size(); ++i){
                    check_value += (bsupport->at(i) - bsupport_prev[i]).norm();
                    bsupport_prev[i] = (*bsupport)[i];
                }
            }

            // double rel_residual = (step == 0 || prev_l2_norm < 1e-8) ?
            //                         check_value : check_value / prev_l2_norm;

            l2_grad = gradient->l2_norm();
            double rel_residual_grad = (step == 0 || prev_l2_grad  < 1e-8) ?
                                    l2_grad : l2_grad / prev_l2_grad ;
            prev_l2_grad = l2_grad;
                                    

            if (l2_grad < tolerance()) { 
                color = Color::green;  
            } else if (l2_grad < tolerance()*10) {
                color = Color::yellow;  
            } else {
                color = Color::red; 
            }
            
            check_description = "L-2 gradient norm: ";
            pcout << "Wasserstein Barycenter Iteration " << CYAN << step << RESET
                << " - Functional value: " << color << (*fvalue) << RESET
                << " - L-2 norm grad " << color << l2_grad << RESET
                << " - rel L-2 norm grad " << color << rel_residual_grad << RESET
                << " - Current L-2 update: " << color << check_value << RESET 
                << " - Previous L-2 update: " << color << prev_l2_norm << RESET << std::endl;

            last_check_value = check_value; 
            prev_l2_norm = check_value;

            if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
            {
                return SolverControl::check(step, l2_grad);
            } else if constexpr (update_flag == UpdateMode::TargetSupportOnly)
            {
                return SolverControl::check(step, rel_residual_grad);
            }
        }

        double get_last_check_value() const { return last_check_value; }

    private:
        ConditionalOStream& pcout;
        double prev_l2_norm = 1.0; 
        double prev_l2_grad = 1.0; ///< The previous L2 norm of the gradient.
        double last_check_value = 0.0;
        double l2_grad = 0.0; ///< The L2 norm of the gradient.

        const double* fvalue = nullptr; ///< Pointer to the functional value.
        const Vector<double>* gradient;
        const Vector<double>* bweights;
        const std::vector<Point<spacedim>>* bsupport;
        std::vector<Point<spacedim>> bsupport_prev;
        Vector<double> bweights_prev;
    };

    std::unique_ptr<VerboseSolverControl> solver_control;
};

#endif

