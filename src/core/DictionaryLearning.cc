#include "SemiDiscreteOT/core/DictionaryLearning.h"

namespace fs = std::filesystem;
using namespace dealii;

template<int spacedim>
void initialize_target_rtree(const std::vector<Point<spacedim>>& target_points,
                            bgi::rtree<IndexedPoint, bgi::rstar<16>>& rtree)
{
    rtree.clear();
    std::vector<IndexedPoint> indexed_points;
    indexed_points.reserve(target_points.size());
    
    for (std::size_t i = 0; i < target_points.size(); ++i) {
        BoostPoint boost_point;
        if constexpr (spacedim >= 1) bg::set<0>(boost_point, target_points[i][0]);
        if constexpr (spacedim >= 2) bg::set<1>(boost_point, target_points[i][1]);
        if constexpr (spacedim >= 3) bg::set<2>(boost_point, target_points[i][2]);
        if constexpr (spacedim < 3) bg::set<2>(boost_point, 0.0);
        indexed_points.emplace_back(boost_point, i);
    }
    
    rtree = bgi::rtree<IndexedPoint, bgi::rstar<16>>(indexed_points);
}

// Function to find nearest target points using rtree
template<int spacedim>
std::vector<std::size_t> find_nearest_target_points(
    const std::function<double(const Point<spacedim>&, const Point<spacedim>&)>& distance_function,
    const Point<spacedim>& query_point,
    const bgi::rtree<IndexedPoint, bgi::rstar<16>>& rtree,
    double distance_threshold)
{
    std::vector<std::size_t> indices;
    
    // Convert dealii Point to BoostPoint
    BoostPoint boost_query;
    if constexpr (spacedim >= 1) bg::set<0>(boost_query, query_point[0]);
    if constexpr (spacedim >= 2) bg::set<1>(boost_query, query_point[1]);
    if constexpr (spacedim >= 3) bg::set<2>(boost_query, query_point[2]);
    if constexpr (spacedim < 3) bg::set<2>(boost_query, 0.0);
    
    // Find points within distance threshold
    for (const auto& indexed_point : rtree |
        bgi::adaptors::queried(bgi::satisfies([&](const IndexedPoint& p) {
            Point<spacedim> dealii_point;
            if constexpr (spacedim >= 1) dealii_point[0] = bg::get<0>(p.first);
            if constexpr (spacedim >= 2) dealii_point[1] = bg::get<1>(p.first);
            if constexpr (spacedim >= 3) dealii_point[2] = bg::get<2>(p.first);
            return distance_function(dealii_point, query_point) <= distance_threshold;
        })))
    {
        indices.push_back(indexed_point.second);
    }

    return indices;
}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::local_assemble_barycenter_gradient(
    const SotSolver<dim, spacedim> &sot_solver,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    BarycenterScratchData<dim, spacedim> &scratch,
    BarycenterCopyData<spacedim> &copy,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::Host> &source_density,
    const std::vector<Point<spacedim>> &target_points,
    const Vector<double> &target_weights,
    const Vector<double> &potentials,
    const double epsilon,
    const double distance_threshold)
{
    const unsigned int n_q_points = scratch.fe_values.n_quadrature_points;

    scratch.fe_values.reinit(cell);
    scratch.fe_values.get_function_values(source_density, scratch.density_values);

    copy.grads = 0.0;

    std::vector<std::size_t> nearby_targets = find_nearest_target_points<spacedim>(
        sot_solver.distance_function, cell->center(), target_rtree, distance_threshold);

    if (nearby_targets.empty())
        return;

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
        const Point<spacedim> x = scratch.fe_values.quadrature_point(q);
        const double JxW = scratch.fe_values.JxW(q);
        const double rho_x = scratch.density_values[q];

        if (std::abs(rho_x) < 1e-12)
            continue;

        std::vector<double> weights(nearby_targets.size());
        double sum_weights = 0.0;
        double max_exponent = -std::numeric_limits<double>::infinity();

        for (std::size_t idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const std::size_t j = nearby_targets[idx];
            const double dist_sq = x.distance_square(target_points[j]);
            const double exponent = (potentials[j] - 0.5 * dist_sq) / epsilon;
            max_exponent = std::max(max_exponent, exponent);
        }

        for (std::size_t idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const std::size_t j = nearby_targets[idx];
            const double dist_sq = x.distance_square(target_points[j]);
            const double exponent = (potentials[j] - 0.5 * dist_sq) / epsilon;
            weights[idx] = target_weights(j) * std::exp(exponent - max_exponent);
            sum_weights += weights[idx];
        }

        if (sum_weights < 1e-12)
            continue;

        for (std::size_t idx = 0; idx < nearby_targets.size(); ++idx)
        {
            const std::size_t j = nearby_targets[idx];      
            const double weight = weights[idx] / sum_weights;
            const double scaled_weight = weight * rho_x * JxW;

            if constexpr (update_flag == UpdateMode::TargetSupportOnly)
            {
                const Vector<double> v = sot_solver.distance_function_gradient(target_points[j], x);
                for (unsigned int d = 0; d < spacedim; ++d)
                {
                    copy.grads[j * spacedim + d] += v[d] * scaled_weight;
                }
            } else if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
            {
                copy.grads[j] += scaled_weight;
            }
            copy.value += epsilon * std::log(sum_weights) * rho_x * JxW;
        }
    }
}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::set_distance_function(
    const std::string &distance_name_)
{
    distance_name = distance_name_;
    if (distance_name == "euclidean") {
        pcout << "Using Euclidean distance function." << std::endl;
        distance_function = [](const Point<spacedim> x, const Point<spacedim> y) { return euclidean_distance<spacedim>(x, y); };
        distance_function_gradient = [](const Point<spacedim> x, const Point<spacedim> y) { return euclidean_distance_gradient<spacedim>(x, y); };
        distance_function_exponential_map = [](const Point<spacedim> x, const Vector<double> v) { return euclidean_distance_exp_map<spacedim>(x, v); };

    } else if (distance_name == "spherical") {
        pcout << "Using Spherical distance function." << std::endl;
        distance_function = [](const Point<spacedim> x, const Point<spacedim> y) { return spherical_distance<spacedim>(x, y); };
        distance_function_gradient = [](const Point<spacedim> x, const Point<spacedim> y) { return spherical_distance_gradient<spacedim>(x, y); };
        distance_function_exponential_map = [](const Point<spacedim> x, const Vector<double> v) { return spherical_distance_exp_map<spacedim>(x, v); };

    } else {
        throw std::invalid_argument("Unknown distance function: " + distance_name);
    }
}

template <int dim, int spacedim, UpdateMode update_flag>
DictionaryLearning<dim, spacedim, update_flag>::DictionaryLearning(const MPI_Comm &comm)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
      mpi_comm(comm){}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::configure()
{
    // Configure OT solver
    n_measures = barycenter_params.weights.size();
    // Check that we have at least 2 measures for meaningful barycenters
    Assert(n_measures >= 2,
        ExcMessage("Wasserstein Barycenters require at least 2 measures, but only " + 
                std::to_string(n_measures) + " provided."));
    
    weights = barycenter_params.weights;

    // Check that weights vector has correct size
    Assert(weights.size() == n_measures,
        ExcMessage("Number of weights (" + std::to_string(weights.size()) + 
               ") must match number of measures (" + std::to_string(n_measures) + ")"));

    // Check that weights sum to 1
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (std::abs(sum_weights - 1.0) >= 1e-10) {
        pcout << "Warning: Sum of weights is not 1.0, but " << sum_weights << ". Normalizing..." << std::endl;
        for (auto &weight : weights) {
            weight /= sum_weights;
        }
    }
    
    auto configure_solver = [this](SotParameterManager &p, int solver_id) {
      p.solver_params.epsilon = ot_params.epsilon;
      p.solver_params.use_log_sum_exp_trick = ot_params.use_log_sum_exp_trick;
      p.solver_params.verbose_output = ot_params.verbose_output;
      p.solver_params.tau = ot_params.tau;
      p.solver_params.distance_threshold_type = ot_params.distance_threshold_type;
      p.solver_params.max_iterations = ot_params.max_iterations;
      p.solver_params.tolerance = ot_params.tolerance;

      p.multilevel_params.source_enabled = ot_params.source_multilevel_enabled;
      p.multilevel_params.target_enabled = ot_params.target_multilevel_enabled;

      p.multilevel_params.source_min_vertices = ot_params.source_min_vertices;
      p.multilevel_params.source_max_vertices = ot_params.source_max_vertices;
      p.multilevel_params.target_min_points = ot_params.target_min_points;
      p.multilevel_params.target_max_points = ot_params.target_max_points;
    
      p.multilevel_params.source_hierarchy_dir = "output/barycenter_h/source" + std::to_string(solver_id);
      p.multilevel_params.use_python_clustering = ot_params.use_python_clustering;
      p.multilevel_params.python_script_name = ot_params.python_script_name;
    };
    
    sot_problems.resize(n_measures);
    for (unsigned int i = 0; i < n_measures; ++i)
    {
      sot_problems[i] = std::make_unique<SemiDiscreteOT<dim, spacedim>>(mpi_comm);
      sot_problems[i]->configure([&](SotParameterManager &p) { configure_solver(p, i); });
    }
}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::run_wasserstein_barycenters()
{  
    Assert(barycenter_points_.size() > 0,
        ExcMessage("Barycenter points vector must not be empty."));
    
    if constexpr (update_flag == UpdateMode::TargetSupportOnly)
        grad_size = barycenter_points.size() * spacedim;
    else if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
        grad_size = barycenter_points.size();

    // Initialize potentials vector with n_measures elements
    potentials.resize(n_measures);
    for (unsigned int i = 0; i < n_measures; ++i) {
        potentials[i].reinit(barycenter_points.size());
    }

    initialize_target_rtree<spacedim>(barycenter_points, target_rtree);

    for (unsigned int i = 0; i < n_measures; ++i) {
        sot_problems[i]->setup_target_measure(
            barycenter_points, barycenter_weights);

        if (ot_params.target_multilevel_enabled) {
            sot_problems[i]->prepare_target_multilevel();
        }
    }
    
    if (ot_params.source_multilevel_enabled) {
      for (unsigned int i = 0; i < n_measures; ++i)
        sot_problems[i]->prepare_source_multilevel();
    }

    // Source and target measures must be set
    for (unsigned int i = 0; i < n_measures; ++i) {
        Assert(sot_problems[i]->sot_solver->source_measure.initialized,
            ExcMessage("Source measure must be set before running SOT iteration"));
        Assert(sot_problems[i]->sot_solver->target_measure.initialized,
            ExcMessage("Target points must be set before running SOT iteration"));
    }

    for (unsigned int i = 1; i < n_measures; ++i) {
        Assert(sot_problems[i]->target_points.size() == sot_problems[0]->target_points.size();,
            ExcMessage("All measures must have the same number of target points. "
                        "Measure 0 has " + std::to_string(sot_problems[0]->target_points.size();) + 
                        " points, but measure " + std::to_string(i) + 
                        " has " + std::to_string(sot_problems[i]->target_points.size()) + " points."));
    }

    pcout << Color::yellow << Color::bold << "\nStarting Wassertein Barycenters algorithm with " << n_measures << " measures:\n" << std::endl;
    for (unsigned int i = 0; i < n_measures; ++i)
        pcout << "  source measure " << i + 1 << " weight: "<< weights[i] << "; dofs: " << sot_problems[i]->source_density.size() << "; target measure: " << sot_problems[i]->target_density.size() << std::endl;

    pcout << std::endl;
    if (ot_params.target_multilevel_enabled)
        pcout << "Target multi-level enabled\n";
    if (ot_params.source_multilevel_enabled)
        pcout << "Source multi-level enabled\n";
    pcout << Color::reset << std::endl;
    
    sol.reinit(grad_size);
    init_sol(sol, barycenter_points, barycenter_weights);

    sol_grad.reinit(grad_size);

    Timer timer;
    timer.start();

    try {
        solver_control = std::make_unique<VerboseSolverControl>(
            barycenter_params.max_iterations,
            barycenter_params.convergence_tolerance,
            pcout);
        solver_control->set_barycenters(
            sol_grad, barycenter_points, barycenter_weights, functional_value);
        WassersteinSolverBFGS<spacedim, Vector<double>, update_flag> solver(
            *solver_control, distance_function_exponential_map, distance_name);
        solver.solve(
            [this](const Vector<double>& w, Vector<double>& grad) {
                return this->evaluate_functional(w, grad);
            },
            sol
        );

    } catch (SolverControl::NoConvergence& exc) {
        pcout << "Warning: Optimization did not converge" << std::endl
              << "  Iterations: " << exc.last_step << std::endl
              << "  Residual: " << exc.last_residual << std::endl;
    } catch (const dealii::ExceptionBase& exc) {
        if (std::string(exc.what()).find("Could not find the initial bracket") != std::string::npos) {
            pcout << "BFGS ended: " << exc.what() << std::endl;
        } else if (std::string(exc.what()).find("Could not complete the sectioning phase") != std::string::npos) {
            pcout << "Warning: Line minimization did not converge: " << exc.what() << std::endl;
        } else {
            throw;
        }
    }

    // Assert that the L1 norm of barycenter_weights is 1
    double l1_norm = barycenter_weights.l1_norm();
    AssertThrow(std::abs(l1_norm - 1.0) < 1e-10,
        ExcMessage("L1 norm of barycenter weights is not 1.0, but " + std::to_string(l1_norm)));

    pcout << Color::green << Color::bold << "Wasserstein Barycenters optimization completed:" << std::endl
        << "  Time taken: " << timer.wall_time() << " seconds" << std::endl
        << "  Iterations: " << solver_control->last_step() << std::endl
        << "  Final update step: " << solver_control->last_value() << Color::reset << std::endl;
}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::init_sol(
    Vector<double> &w,
    const std::vector<Point<spacedim>> &bpoints,
    const Vector<double> &bweights)
{
    if constexpr (update_flag == UpdateMode::TargetSupportOnly)
    {
        for (unsigned int i = 0; i < bpoints.size(); ++i)
            for (unsigned int d = 0; d < spacedim; ++d)
                w[i * spacedim + d] = bpoints[i][d];
    }
    else if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
    {
        for (unsigned int i = 0; i < bpoints.size(); ++i)
            w[i] = std::log(bweights[i]);
    }
}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::update_barycenter(
    const Vector<double> &w,
    std::vector<Point<spacedim>> &bpoints,
    Vector<double> &bweights,
    const bool reinit_tree)
{
    if constexpr (update_flag == UpdateMode::TargetSupportOnly)
    {
        for (unsigned int i = 0; i < bpoints.size(); ++i)
            for (unsigned int d = 0; d < spacedim; ++d)
            {
                bpoints[i][d] = w[i * spacedim + d];
            }
        if (reinit_tree){
            initialize_target_rtree<spacedim>(bpoints, target_rtree);}
    }
    else if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
    {
        double sum_weights = 0.0;
        for (unsigned int i = 0; i < bpoints.size(); ++i)
        {
            bweights[i] = std::exp(w[i]);
            sum_weights += bweights[i];
        }
        if (std::abs(sum_weights)> 1e-10)
        {
            bweights /= sum_weights;
        }
    }
}

template <int dim, int spacedim, UpdateMode update_flag>
double DictionaryLearning<dim, spacedim, update_flag>::evaluate_functional(
    const Vector<double>& w, Vector<double>& grad)
{
    // barycenter_points, barycenter_weights
    if constexpr (update_flag == UpdateMode::TargetSupportOnly)
    {
        update_barycenter(
            w, barycenter_points, barycenter_weights, true);
    } else if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
    {
        update_barycenter(
            w, barycenter_points, barycenter_weights, false);
    }
 
    std::string step = std::to_string(
        (solver_control->last_step() >= 0 && solver_control->last_step() <= barycenter_params.max_iterations * 10) 
        ? solver_control->last_step() 
        : 0);

    {
        if (barycenter_params.silence_output)
        {
            struct SilenceStdStreams {
                std::streambuf* cout_old;
                struct NullBuf : std::streambuf {
                    int overflow(int c) override { return traits_type::not_eof(c); }
                } nullbuf;
                SilenceStdStreams() {
                    cout_old = std::cout.rdbuf();
                    std::cout.rdbuf(&nullbuf);
                }
                ~SilenceStdStreams() {
                    std::cout.rdbuf(cout_old);
                }
            } silence;
        }

        for (unsigned int i = 0; i < n_measures; ++i) {
            sot_problems[i]->setup_target_measure(
                barycenter_points, barycenter_weights);

            if (ot_params.target_multilevel_enabled) {
                sot_problems[i]->prepare_target_multilevel();
            }
        }

        if (ot_params.target_multilevel_enabled) {
            for (unsigned int i = 0; i < n_measures; ++i)
                potentials[i] = sot_problems[i]->solve();
        }
        else {
            for (unsigned int i = 0; i < n_measures; ++i)
                potentials[i] = sot_problems[i]->solve(potentials[i]);
        }
    }

    std::vector<Vector<double>> grad_update(n_measures);
    std::vector<Vector<double>> grad_update_local(n_measures);
    for (unsigned int i = 0; i < n_measures; ++i)
    {
        grad_update[i].reinit(grad_size);
        grad_update_local[i].reinit(grad_size);
    }
    std::vector<double> values(n_measures, 0.0);
    std::vector<double> values_local(n_measures, 0.0);
    
    for (unsigned int i = 0; i < n_measures; ++i)
    {
        // Create filtered iterators for locally owned cells
        FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
            begin_filtered(IteratorFilters::LocallyOwnedCell(),
                          sot_problems[i]->sot_solver->source_measure.dof_handler->begin_active()),
            end_filtered(IteratorFilters::LocallyOwnedCell(),
                        sot_problems[i]->sot_solver->source_measure.dof_handler->end());

        WorkStream::run(
            begin_filtered,
            end_filtered,
            [&](const auto& cell, auto& scratch, auto& copy) {
                if (cell->is_locally_owned())
                {
                    local_assemble_barycenter_gradient(
                        *sot_problems[i]->sot_solver,
                        cell,
                        scratch,
                        copy,
                        *sot_problems[i]->sot_solver->source_measure.density,
                        barycenter_points,
                        barycenter_weights,
                        potentials[i],
                        ot_params.epsilon,
                        ot_params.distance_threshold);
                }
            },
            [&](const auto& copy) {
                grad_update_local[i].add(1.0, copy.grads);
                values_local[i] += copy.value;
            },
            BarycenterScratchData<dim, spacedim>(
                *sot_problems[i]->sot_solver->source_measure.fe,
                *sot_problems[i]->sot_solver->source_measure.mapping,
                sot_problems[i]->sot_solver->source_measure.quadrature_order),
            BarycenterCopyData<spacedim>(grad_size));
        
        Utilities::MPI::sum(
            grad_update_local[i], mpi_comm, grad_update[i]);
        values[i] = Utilities::MPI::sum(values_local[i], mpi_comm);
    }

    // Compute weighted sum of gradients
    functional_value = 0.0;
    grad.reinit(grad_size);
    for (unsigned int i = 0; i < n_measures; ++i)
    {
        grad.add(-weights[i], grad_update[i]);
        grad.add(weights[i], potentials[i]);

        double prod = potentials[i] * barycenter_weights;
        functional_value += weights[i] * (prod-values[i]);
    }

    if constexpr (update_flag == UpdateMode::TargetMeasureOnly)
    {
        double dot = grad * barycenter_weights;
        for (unsigned int i = 0; i < grad.size(); ++i)
            grad[i] = barycenter_weights[i] * (grad[i] - dot);
    }

    sol_grad = grad;

    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
        save_vtk_output(barycenter_points, barycenter_weights, sol_grad, "barycenter_" + step + ".vtk");
    }

    return functional_value;
}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::save_vtk_output(
    const std::vector<Point<spacedim>> &points,
    const Vector<double> &weights,
    const std::string &filename) const
{
    std::ofstream vtk_file(filename);
    if (!vtk_file.is_open()) return;

    const unsigned int n_points = points.size();

    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Barycenter Points\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";
    vtk_file << "POINTS " << n_points << " float\n";

    for (const auto &point : points) {
        for (unsigned int d = 0; d < spacedim; ++d) {
            vtk_file << point[d] << " ";
        }
        vtk_file << "\n";
    }

    vtk_file << "CELLS " << n_points << " " << 2 * n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1 " << i << "\n";
    }

    vtk_file << "CELL_TYPES " << n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1\n";  // VTK_VERTEX
    }

    vtk_file << "POINT_DATA " << n_points << "\n";
    vtk_file << "SCALARS weights float 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << weights[i] << "\n";
    }

    // if (update_flag == UpdateMode::TargetMeasureOnly)
    //     vtk_file << "VECTORS sol_grad float\n";
    //     for (unsigned int i = 0; i < n_points; ++i) {
    //         vtk_file << sol_grad[i] << " ";
    //         vtk_file << "\n";
    //     }

    vtk_file.close();
}

template <int dim, int spacedim, UpdateMode update_flag>
void DictionaryLearning<dim, spacedim, update_flag>::save_vtk_output(
    const std::vector<Point<spacedim>> &points,
    const Vector<double> &weights,
    const Vector<double> &grad,
    const std::string &filename) const
{
    std::ofstream vtk_file(filename);
    if (!vtk_file.is_open()) return;

    const unsigned int n_points = points.size();

    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Barycenter Points\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n";
    vtk_file << "POINTS " << n_points << " float\n";

    for (const auto &point : points) {
        for (unsigned int d = 0; d < spacedim; ++d) {
            vtk_file << point[d] << " ";
        }
        vtk_file << "\n";
    }

    vtk_file << "CELLS " << n_points << " " << 2 * n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1 " << i << "\n";
    }

    vtk_file << "CELL_TYPES " << n_points << "\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << "1\n";  // VTK_VERTEX
    }

    vtk_file << "POINT_DATA " << n_points << "\n";

    vtk_file << "SCALARS weights float 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (unsigned int i = 0; i < n_points; ++i) {
        vtk_file << weights[i] << "\n";
    }

    if (update_flag == UpdateMode::TargetMeasureOnly)
    {
        vtk_file << "SCALARS grad float 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for (unsigned int i = 0; i < n_points; ++i) {
            vtk_file << grad[i] << "\n";
        }
    }

    vtk_file.close();
}

template class DictionaryLearning<2, 2, UpdateMode::TargetSupportOnly>;
template class DictionaryLearning<3, 3, UpdateMode::TargetSupportOnly>;
template class DictionaryLearning<2, 3, UpdateMode::TargetSupportOnly>;

template class DictionaryLearning<2, 2, UpdateMode::TargetMeasureOnly>;
template class DictionaryLearning<3, 3, UpdateMode::TargetMeasureOnly>;
template class DictionaryLearning<2, 3, UpdateMode::TargetMeasureOnly>;