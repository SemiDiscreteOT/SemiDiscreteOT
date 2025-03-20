#include "SemiDiscreteOT/solvers/SotSolver.h"
#include <deal.II/optimization/solver_bfgs.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>




template <int dim>
SotSolver<dim>::SotSolver(const MPI_Comm& comm)
    : mpi_communicator(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, this_mpi_process == 0)
    , current_distance_threshold(1e-1)
    , effective_distance_threshold(1e-1)
    , is_caching_active(false)
    , current_potential(nullptr)
    , current_lambda(1.0)
    , global_functional(0.0)
    , global_C_integral(0.0)
    , current_cache_size_mb(0.0)
    , cache_limit_reached(false)
{}

template <int dim>
void SotSolver<dim>::setup_source(
    const DoFHandler<dim>& dof_handler,
    const Mapping<dim>& mapping,
    const FiniteElement<dim>& fe,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>& source_density,
    const unsigned int quadrature_order)
{
    source_measure = SourceMeasure(dof_handler, mapping, fe, source_density, quadrature_order);
}

template <int dim>
void SotSolver<dim>::setup_target(
    const std::vector<Point<dim>>& target_points,
    const Vector<double>& target_density)
{
    target_measure = TargetMeasure(target_points, target_density);
}

template <int dim>
bool SotSolver<dim>::validate_measures() const
{
    if (!source_measure.dof_handler) {
        pcout << "Error: Source measure not set up" << std::endl;
        return false;
    }
    if (target_measure.points.empty()) {
        pcout << "Error: Target measure not set up" << std::endl;
        return false;
    }
    if (target_measure.density.size() != target_measure.points.size()) {
        pcout << "Error: Target density size mismatch" << std::endl;
        return false;
    }
    return true;
}

template <int dim>
void SotSolver<dim>::solve(
    Vector<double>& potentials,
    const SourceMeasure& source,
    const TargetMeasure& target,
    const ParameterManager::SolverParameters& params)
{
    // Set up measures
    source_measure = source;
    target_measure = target;

    // Call main solve method
    solve(potentials, params);
}

template <int dim>
void SotSolver<dim>::solve(
    Vector<double>& potentials,
    const ParameterManager::SolverParameters& params)
{
    if (!validate_measures()) {
        throw std::runtime_error("Invalid measures configuration");
    }

    // Store parameters
    current_params = params;
    current_lambda = params.regularization_param;

    // Set up parallel environment
    unsigned int n_threads = params.n_threads;
    if (n_threads == 0) {
        n_threads = std::max(1U, MultithreadInfo::n_cores() / n_mpi_processes);
    }
    MultithreadInfo::set_thread_limit(n_threads);

    pcout << "Parallel Configuration:" << std::endl
          << "  MPI Processes: " << n_mpi_processes
          << " (Current rank: " << this_mpi_process << ")" << std::endl
          << "  Threads per process: " << n_threads << std::endl
          << "  Total parallel units: " << n_threads * n_mpi_processes << std::endl;

    // Initialize potentials if needed
    if (potentials.size() != target_measure.points.size()) {
        potentials.reinit(target_measure.points.size());
    }

    // Initialize gradient member variable
    gradient.reinit(potentials.size());

    // Create solver control with verbose output
    solver_control = std::make_unique<VerboseSolverControl>(
        params.max_iterations,
        params.tolerance,
        pcout
    );

    if (!params.verbose_output) {
        solver_control->log_history(false);
        solver_control->log_result(false);
    }

    try {
        Timer timer;
        timer.start();

        // Create and run BFGS solver
        SolverBFGS<Vector<double>> solver(*solver_control);
        current_potential = &potentials;

        solver.solve(
            [this](const Vector<double>& w, Vector<double>& grad) {
                return this->evaluate_functional(w, grad);
            },
            potentials
        );

        timer.stop();

        pcout << Color::green << Color::bold << "Optimization completed:" << std::endl
              << "  Time taken: " << timer.wall_time() << " seconds" << std::endl
              << "  Iterations: " << solver_control->last_step() << std::endl
              << "  Final function value: " << solver_control->last_value() << Color::reset << std::endl;

        if (params.use_caching) {
            pcout << "  Final cache size: " << get_cache_size_mb() << " MB" << std::endl
                  << "  Cached cells: " << cell_caches.size() << std::endl;
        }

    } catch (SolverControl::NoConvergence& exc) {
        pcout << "Warning: Optimization did not converge" << std::endl
              << "  Iterations: " << exc.last_step << std::endl
              << "  Residual: " << exc.last_residual << std::endl;
        throw;
    }

    // Reset solver state
    reset_distance_threshold_cache();
    current_potential = nullptr;
}

template <int dim>
double SotSolver<dim>::evaluate_functional(
    const Vector<double>& potentials,
    Vector<double>& gradient_out)
{
    // Store current potentials for use in local assembly
    current_potential = &potentials;
    current_lambda = current_params.regularization_param;

    // Reset global accumulators
    global_functional = 0.0;
    global_C_integral = 0.0;
    gradient = 0;  // Reset class member gradient
    double local_process_functional = 0.0;
    double local_process_C_integral = 0.0;  // Local C integral accumulator
    Vector<double> local_process_gradient(target_measure.points.size());

    // Update distance threshold for target point search
    compute_distance_threshold();
    if (current_params.verbose_output) {
        pcout << "Using distance threshold: " << current_distance_threshold
              << " (Effective: " << effective_distance_threshold << ")" << std::endl;
    }

    try {
        // Determine if we're using simplex elements
        bool use_simplex = (dynamic_cast<const FE_SimplexP<dim>*>(&*source_measure.fe) != nullptr);

        // Create appropriate quadrature rule
        std::unique_ptr<Quadrature<dim>> quadrature;
        if (use_simplex) {
            quadrature = std::make_unique<QGaussSimplex<dim>>(source_measure.quadrature_order);
        } else {
            quadrature = std::make_unique<QGauss<dim>>(source_measure.quadrature_order);
        }

        // Create scratch and copy data
        ScratchData scratch_data(*source_measure.fe,
                               *source_measure.mapping,
                               *quadrature);
        CopyData copy_data(target_measure.points.size());

        // Create filtered iterators for locally owned cells
        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
            begin_filtered(IteratorFilters::LocallyOwnedCell(),
                          source_measure.dof_handler->begin_active()),
            end_filtered(IteratorFilters::LocallyOwnedCell(),
                        source_measure.dof_handler->end());

        // Parallel assembly using WorkStream
        WorkStream::run(
            begin_filtered,
            end_filtered,
            [this](const typename DoFHandler<dim>::active_cell_iterator& cell,
                   ScratchData& scratch,
                   CopyData& copy) {
                this->local_assemble(cell, scratch, copy);
            },
            [this, &local_process_functional, &local_process_C_integral, &local_process_gradient](const CopyData& copy) {
                std::lock_guard<std::mutex> lock(cache_mutex);
                local_process_functional += copy.functional_value;
                local_process_C_integral += copy.C_integral;
                local_process_gradient += copy.gradient_values;
            },
            scratch_data,
            copy_data);

        // Synchronize across MPI processes
        global_functional = Utilities::MPI::sum(local_process_functional, mpi_communicator);
        global_C_integral = Utilities::MPI::sum(local_process_C_integral, mpi_communicator);
        gradient = 0;  // Reset gradient
        Utilities::MPI::sum(local_process_gradient, mpi_communicator, gradient);

        // Add linear term (only on root process to avoid duplication)
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
            for (unsigned int i = 0; i < target_measure.points.size(); ++i) {
                global_functional -= potentials[i] * target_measure.density[i];
                gradient[i] -= target_measure.density[i];
            }
        }

        // Broadcast final results to all processes
        global_functional = Utilities::MPI::broadcast(mpi_communicator, global_functional, 0);
        gradient = Utilities::MPI::broadcast(mpi_communicator, gradient, 0);

        // Copy result to output gradient
        gradient_out = gradient;

        if (current_params.verbose_output) {
            pcout << "Functional evaluation completed:" << std::endl
                  << "  Function value: " << global_functional << std::endl
                  << "  C integral: " << global_C_integral << std::endl;
            if (current_params.use_caching) {
                pcout << "  Cache size: " << get_cache_size_mb() << " MB" << std::endl
                      << "  Cached cells: " << cell_caches.size() << std::endl;
                if (cache_limit_reached) {
                    pcout << "  \033[1;31mCache limit reached: No new entries being added\033[0m" << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        pcout << "Error in functional evaluation: " << e.what() << std::endl;
        throw;
    }

    return global_functional;
}

template <int dim>
void SotSolver<dim>::local_assemble(
    const typename DoFHandler<dim>::active_cell_iterator& cell,
    ScratchData& scratch,
    CopyData& copy)
{
    if (!cell->is_locally_owned())
        return;

    scratch.fe_values.reinit(cell);
    const std::vector<Point<dim>>& q_points = scratch.fe_values.get_quadrature_points();
    scratch.fe_values.get_function_values(*source_measure.density, scratch.density_values);

    copy.functional_value = 0.0;
    copy.gradient_values = 0;
    copy.C_integral = 0.0;

    const unsigned int n_q_points = q_points.size();
    const double lambda_inv = 1.0 / current_lambda;

    if (current_params.use_caching && is_caching_active) {
        // Caching path
        std::string cell_id = cell->id().to_string();
        CellCache* cell_cache_ptr = nullptr;
        std::vector<std::size_t> cell_target_indices;
        bool need_computation = true;
        bool cache_entry_exists = false;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cell_caches.find(cell_id);
            cache_entry_exists = (it != cell_caches.end());

            if (cache_entry_exists) {
                cell_cache_ptr = &it->second;
                cell_target_indices = cell_cache_ptr->target_indices;
                need_computation = !cell_cache_ptr->is_valid;
            }
        }

        if (!cache_entry_exists) {
            // If cache limit is already reached, skip directly to direct computation
            if (cache_limit_reached) {
                goto direct_computation;
            }

            // Need to find target points and check if cache size would be exceeded
            cell_target_indices = find_nearest_target_points(cell->center());
            if (cell_target_indices.empty()) return;

            double entry_size_mb = estimate_cache_entry_size_mb(cell_target_indices, n_q_points);
            double projected_size = current_cache_size_mb.load() + entry_size_mb;

            if (current_params.max_cache_size_mb > 0 && projected_size > current_params.max_cache_size_mb) {
                // Cache limit would be exceeded, don't add new entry but continue with direct computation
                {
                    std::lock_guard<std::mutex> lock(cache_mutex);
                    if (!cache_limit_reached) {
                        cache_limit_reached = true;
                        pcout << "\n\033[1;31mWARNING: Cache size limit (" << current_params.max_cache_size_mb
                              << " MB) reached. Using existing cache entries but not adding new ones.\033[0m" << std::endl;
                    }
                }
                // Continue with direct computation for this cell
                goto direct_computation;
            } else {
                // Create new cache entry
                std::lock_guard<std::mutex> lock(cache_mutex);
                CellCache& cell_cache = cell_caches[cell_id];
                cell_cache_ptr = &cell_cache;
                cell_cache.target_indices = cell_target_indices;
                cell_cache.precomputed_exp_terms.resize(n_q_points * cell_target_indices.size());
                current_cache_size_mb.store(current_cache_size_mb.load() + entry_size_mb);
            }
        }

        // Proceed with cached or newly created entry
        const unsigned int n_target_points = cell_target_indices.size();

        // Preload target data for vectorization
        std::vector<Point<dim>> target_positions(n_target_points);
        std::vector<double> target_densities(n_target_points);
        std::vector<double> potential_values(n_target_points);

        for (size_t i = 0; i < n_target_points; ++i) {
            const size_t idx = cell_target_indices[i];
            target_positions[i] = target_measure.points[idx];
            target_densities[i] = target_measure.density[idx];
            potential_values[i] = (*current_potential)[idx];
        }

        // Process quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q) {
            const Point<dim>& x = q_points[q];
            const double density_value = scratch.density_values[q];
            const double JxW = scratch.fe_values.JxW(q);

            double total_sum_exp = 0.0;
            std::vector<double> active_exp_terms(n_target_points, 0.0);

            const unsigned int base_idx = q * n_target_points;
            if (need_computation) {
                #pragma omp simd reduction(+:total_sum_exp)
                for (size_t i = 0; i < n_target_points; ++i) {
                    if (cell_target_indices[i] >= target_measure.points.size()) {
                        continue;
                    }
                    const double local_dist2 = (x - target_positions[i]).norm_square();
                    const double precomputed_term = target_densities[i] *
                        std::exp(-0.5 * local_dist2 * lambda_inv);
                    cell_cache_ptr->precomputed_exp_terms[base_idx + i] = precomputed_term;
                    active_exp_terms[i] = precomputed_term * std::exp(potential_values[i] * lambda_inv);
                    total_sum_exp += active_exp_terms[i];
                }
            } else {
                #pragma omp simd reduction(+:total_sum_exp)
                for (size_t i = 0; i < n_target_points; ++i) {
                    if (cell_target_indices[i] >= target_measure.points.size()) {
                        continue;
                    }
                    const double cached_term = cell_cache_ptr->precomputed_exp_terms[base_idx + i];
                    active_exp_terms[i] = cached_term * std::exp(potential_values[i] * lambda_inv);
                    total_sum_exp += active_exp_terms[i];
                }
            }

            if (total_sum_exp <= 0.0) continue;

            copy.functional_value += density_value * current_lambda *
                std::log(total_sum_exp) * JxW;
            copy.C_integral += density_value * JxW / total_sum_exp;

            const double scale = density_value * JxW / total_sum_exp;
            #pragma omp simd
            for (size_t i = 0; i < n_target_points; ++i) {
                if (cell_target_indices[i] >= target_measure.points.size()) {
                    continue;
                }
                if (active_exp_terms[i] > 0.0) {
                    copy.gradient_values[cell_target_indices[i]] += scale * active_exp_terms[i];
                }
            }
        }

        if (need_computation && cell_cache_ptr) {
            cell_cache_ptr->is_valid = true;
        }

        return; // Skip the direct computation path
    }

direct_computation:
    // Direct computation path
    std::vector<std::size_t> cell_target_indices = find_nearest_target_points(cell->center());
    if (cell_target_indices.empty()) return;

    const unsigned int n_target_points = cell_target_indices.size();

    std::vector<Point<dim>> target_positions(n_target_points);
    std::vector<double> target_densities(n_target_points);
    std::vector<double> potential_values(n_target_points);

    for (size_t i = 0; i < n_target_points; ++i) {
        const size_t idx = cell_target_indices[i];
        target_positions[i] = target_measure.points[idx];
        target_densities[i] = target_measure.density[idx];
        potential_values[i] = (*current_potential)[idx];
    }

    for (unsigned int q = 0; q < n_q_points; ++q) {
        const Point<dim>& x = q_points[q];
        const double density_value = scratch.density_values[q];
        const double JxW = scratch.fe_values.JxW(q);

        double total_sum_exp = 0.0;
        std::vector<double> exp_terms(n_target_points);

        #pragma omp simd reduction(+:total_sum_exp)
        for (size_t i = 0; i < n_target_points; ++i) {
            const double local_dist2 = (x - target_positions[i]).norm_square();
            exp_terms[i] = target_densities[i] *
                std::exp((potential_values[i] - 0.5 * local_dist2) * lambda_inv);
            total_sum_exp += exp_terms[i];
        }

        if (total_sum_exp <= 0.0) continue;

        copy.functional_value += density_value * current_lambda *
            std::log(total_sum_exp) * JxW;
        copy.C_integral += density_value * JxW / total_sum_exp;

        const double scale = density_value * JxW / total_sum_exp;
        #pragma omp simd
        for (size_t i = 0; i < n_target_points; ++i) {
            if (exp_terms[i] > 0.0) {
                copy.gradient_values[cell_target_indices[i]] += scale * exp_terms[i];
            }
        }
    }
}

template <int dim>
void SotSolver<dim>::compute_distance_threshold() const
{
    if (current_potential == nullptr) {
        current_distance_threshold = std::numeric_limits<double>::max();
        is_caching_active = false;
        return;
    }

    double max_potential = *std::max_element(current_potential->begin(), current_potential->end());
    double min_target_density = *std::min_element(target_measure.density.begin(), target_measure.density.end());

    double squared_threshold = -2.0 * current_lambda *
        std::log(current_params.epsilon/min_target_density) + 2.0 * max_potential;
    double computed_threshold = std::sqrt(std::max(0.0, squared_threshold));

    if (!current_params.use_caching) {
        current_distance_threshold = computed_threshold;
        is_caching_active = false;
        return;
    }

    // Even if cache limit is reached, we still want to use existing cache entries
    if (is_caching_active && computed_threshold <= effective_distance_threshold) {
        current_distance_threshold = effective_distance_threshold;
        return;
    }

    current_distance_threshold = computed_threshold;
    effective_distance_threshold = computed_threshold * 1.1;
    is_caching_active = true;
}

template <int dim>
std::vector<std::size_t> SotSolver<dim>::find_nearest_target_points(
    const Point<dim>& query_point) const
{
    namespace bgi = boost::geometry::index;
    std::vector<std::size_t> indices;

    for (const auto& indexed_point : target_measure.rtree |
         bgi::adaptors::queried(bgi::satisfies([&](const IndexedPoint& p) {
             return (p.first - query_point).norm() <= current_distance_threshold;
         })))
    {
        indices.push_back(indexed_point.second);
    }

    return indices;
}

template <int dim>
void SotSolver<dim>::reset_distance_threshold_cache() const
{
    // Reset all cache-related state
    is_caching_active = false;
    cell_caches.clear();
    current_cache_size_mb.store(0.0);
    cache_limit_reached = false;
}

template <int dim>
double SotSolver<dim>::get_cache_size_mb() const
{
    return current_cache_size_mb.load();
}

template <int dim>
double SotSolver<dim>::estimate_cache_entry_size_mb(const std::vector<std::size_t>& target_indices,
                                                  unsigned int n_q_points) const
{
    // Size of the indices vector
    double indices_size = target_indices.size() * sizeof(std::size_t);

    // Size of the precomputed terms (n_q_points * target_indices.size() doubles)
    double precomputed_terms_size = n_q_points * target_indices.size() * sizeof(double);

    // Small overhead for the CellCache struct itself
    double struct_overhead = sizeof(CellCache);

    // Total size in MB
    return (indices_size + precomputed_terms_size + struct_overhead) / (1024.0 * 1024.0);
}

template <int dim>
unsigned int SotSolver<dim>::get_last_iteration_count() const
{
    return solver_control ? solver_control->last_step() : 0;
}

template <int dim>
bool SotSolver<dim>::get_convergence_status() const
{
    return solver_control && solver_control->last_check() == SolverControl::success;
}

// Explicit instantiation
template class SotSolver<2>;
template class SotSolver<3>;

