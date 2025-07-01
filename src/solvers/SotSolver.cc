#include "SemiDiscreteOT/solvers/SotSolver.h"

template <int dim, int spacedim>
SotSolver<dim, spacedim>::SotSolver(const MPI_Comm& comm)
    : mpi_communicator(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, this_mpi_process == 0)
    , current_distance_threshold(1e-1)
    , effective_distance_threshold(1e-1)
    , current_potential(nullptr)
    , current_epsilon(1.0)
    , global_functional(0.0)
    , gradient()
    , covering_radius(0.0)
    , min_target_density(0.0)
    , C_global(0.0)
{
    distance_function = [](const Point<spacedim> x, const Point<spacedim> y) { return euclidean_distance<spacedim>(x, y); };
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::set_distance_function(
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

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::setup_source(
    const DoFHandler<dim, spacedim>& dof_handler,
    const Mapping<dim, spacedim>& mapping,
    const FiniteElement<dim, spacedim>& fe,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::Host>& source_density,
    const unsigned int quadrature_order)
{
    source_measure = SourceMeasure(dof_handler, mapping, fe, source_density, quadrature_order);
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::setup_target(
    const std::vector<Point<spacedim>>& target_points,
    const Vector<double>& target_density)
{
    target_measure = TargetMeasure(target_points, target_density);
}

template <int dim, int spacedim>
bool SotSolver<dim, spacedim>::validate_measures() const
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

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::solve(
    Vector<double>& potentials,
    const SourceMeasure& source,
    const TargetMeasure& target,
    const SotParameterManager::SolverParameters& params)
{
    // Set up measures
    source_measure = source;
    target_measure = target;

    // Call main solve method
    solve(potentials, params);
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::solve(
    Vector<double>& potentials,
    const SotParameterManager::SolverParameters& params)
{
    if (!validate_measures()) {
        throw std::runtime_error("Invalid measures configuration");
    }
    
    // Store parameters
    current_params = params;
    current_epsilon = params.epsilon;

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

    // Compute and cache the covering radius and minimum target density
    covering_radius = compute_covering_radius() * 1.1;
    min_target_density = *std::min_element(target_measure.density.begin(), target_measure.density.end());
    
    pcout << "Covering radius (R0): " << covering_radius << std::endl
          << "  (Maximum distance from any source cell center to the nearest target point)" << std::endl;

    // Print log-sum-exp status if using small entropy
    if (params.epsilon < 1e-2) {
        pcout << "Small entropy detected (λ = " << params.epsilon << ")" << std::endl;
        pcout << "  Log-Sum-Exp trick: " << (params.use_log_sum_exp_trick ? "enabled" : "disabled") << std::endl;
        if (!params.use_log_sum_exp_trick && params.epsilon < 1e-4) {
            pcout << "  \033[1;33mWARNING: Using very small entropy without Log-Sum-Exp trick may cause numerical instability\033[0m" << std::endl;
        }
    }

    // Initialize potentials if needed
    if (potentials.size() != target_measure.points.size()) {
        potentials.reinit(target_measure.points.size());
    }

    // Initialize gradient member variable
    gradient.reinit(potentials.size());

    // Configure Solver Control based on selected tolerance type
    bool use_componentwise = (params.solver_control_type == "componentwise");
    double solver_ctrl_tolerance;
    if (use_componentwise) {
        solver_ctrl_tolerance = std::numeric_limits<double>::min();
    } else {
        solver_ctrl_tolerance = params.tolerance;
    }

    solver_control = std::make_unique<VerboseSolverControl>(
        params.max_iterations,
        solver_ctrl_tolerance,
        use_componentwise,
        pcout
    );

    if (!params.verbose_output) {
        solver_control->log_history(false);
        solver_control->log_result(false);
    }

    auto* verbose_control = dynamic_cast<VerboseSolverControl*>(solver_control.get());
    AssertThrow(verbose_control != nullptr, ExcInternalError());

    verbose_control->set_gradient(gradient);

    if (use_componentwise) {
        verbose_control->set_target_measure(target_measure.density, params.tolerance);
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

    } catch (SolverControl::NoConvergence& exc) {
        pcout << "Warning: Optimization did not converge" << std::endl
              << "  Iterations: " << exc.last_step << std::endl
              << "  Residual: " << exc.last_residual << std::endl;
        throw;
    }

    // Reset solver state
    current_potential = nullptr;
}

template <int dim, int spacedim>
double SotSolver<dim, spacedim>::evaluate_functional(
    const Vector<double>& potentials,
    Vector<double>& gradient_out)
{
    // Store current potentials for use in local assembly
    current_potential = &potentials;
    current_epsilon = current_params.epsilon;

    // Update distance threshold for target point search
    compute_distance_threshold();

    // Reset global accumulators
    global_functional = 0.0;
    gradient = 0;  // Reset class member gradient
    C_global = 0.0; // Reset C_global
    double local_process_functional = 0.0;
    Vector<double> local_process_gradient(target_measure.points.size());
    double local_process_C_sum = 0.0; // Accumulator for C_sum on this MPI process

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
        FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
            begin_filtered(IteratorFilters::LocallyOwnedCell(),
                          source_measure.dof_handler->begin_active()),
            end_filtered(IteratorFilters::LocallyOwnedCell(),
                        source_measure.dof_handler->end());

        auto function_call = [this](
            CopyData& copy,
            const Point<spacedim> &x,
            const std::vector<std::size_t> &cell_target_indices,
            const std::vector<double> &exp_terms,
            const std::vector<double> &target_densities,
            const double &density_value,
            const double &JxW,
            const double &total_sum_exp,
            const double &max_exponent,
            const double &current_epsilon)
        {

            // Calculate functional value based on whether log-sum-exp is used
            if (current_params.use_log_sum_exp_trick) {
                copy.functional_value += density_value * current_epsilon * 
                    (max_exponent + std::log(total_sum_exp)) * JxW;
            } else {
                copy.functional_value += density_value * current_epsilon *
                    std::log(total_sum_exp) * JxW;
            }
            const double scale = density_value * JxW / total_sum_exp;
            copy.local_C_sum += scale; // Add scale to local_C_sum for this cell q-point
            #pragma omp simd
            for (size_t i = 0; i < cell_target_indices.size(); ++i) {
                if (exp_terms[i] > 0.0) {
                    copy.gradient_values[cell_target_indices[i]] += scale * exp_terms[i];
                }
            }
        };

        // Parallel assembly using WorkStream
        WorkStream::run(
            begin_filtered,
            end_filtered,
            [this, &function_call](const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
                   ScratchData& scratch,
                   CopyData& copy) {
                this->local_assemble(cell, scratch, copy, function_call);
            },
            [this, &local_process_functional, &local_process_gradient, &local_process_C_sum](const CopyData& copy) {
                local_process_functional += copy.functional_value;
                local_process_gradient += copy.gradient_values;
                local_process_C_sum += copy.local_C_sum; // Accumulate local_C_sum
            },
            scratch_data,
            copy_data);

        // Synchronize across MPI processes
        global_functional = Utilities::MPI::sum(local_process_functional, mpi_communicator);
        gradient = 0;  // Reset gradient
        Utilities::MPI::sum(local_process_gradient, mpi_communicator, gradient);
        C_global = Utilities::MPI::sum(local_process_C_sum, mpi_communicator); // Sum C_global across processes

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
            pcout << "Functional evaluation completed:" << std::endl;
            pcout << "  Function value: " << global_functional << std::endl;
            pcout << "  C_global: " << C_global << std::endl;
            
            // Calculate and print the geometric radius bound for comparison
            if (current_potential != nullptr && current_potential->size() == target_measure.points.size()) {
                double geom_radius_bound = compute_geometric_radius_bound(*current_potential, current_epsilon, current_params.tau);
                
                // Calculate traditional pointwise bound for comparison
                double max_pot = *std::max_element(current_potential->begin(), current_potential->end());
                double min_tgt_density = min_target_density > 0.0 ? 
                                      min_target_density : 
                                      *std::min_element(target_measure.density.begin(), target_measure.density.end());
                double sq_threshold = -2.0 * current_epsilon * std::log(current_params.tau/min_tgt_density) + 2.0 * max_pot;
                double pointwise_bound = std::sqrt(std::max(0.0, sq_threshold));

                // Calculate the new integral radius bound
                double integral_radius_bound = compute_integral_radius_bound(
                    *current_potential, 
                    current_epsilon, 
                    current_params.tau, 
                    C_global,           
                    global_functional   
                );
                
                pcout << "  Current distance threshold: " << current_distance_threshold 
                      << " (using: " << current_params.distance_threshold_type << ")" << std::endl
                      << "  Pointwise bound (eps_machine=" << current_params.tau << "): " << pointwise_bound << std::endl
                      << "  Integral bound (C_global=" << C_global << ", τ=" << current_params.tau << "): " << integral_radius_bound << std::endl
                      << "  Geometric bound (τ=" << current_params.tau << "): " << geom_radius_bound
                      << " (ratio to pointwise: " << (pointwise_bound > 1e-9 ? geom_radius_bound/pointwise_bound : 0.0) << ")" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        pcout << "Error in functional evaluation: " << e.what() << std::endl;
        throw;
    }

    return global_functional;
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::compute_distance_threshold() const
{
    if (current_potential == nullptr) {
        current_distance_threshold = std::numeric_limits<double>::max();
        effective_distance_threshold = std::numeric_limits<double>::max();
        return;
    }

    // Choose distance threshold calculation method based on parameter
    double computed_threshold = 0.0;
    std::string used_method_for_log;
    
    if (current_params.distance_threshold_type == "integral") {
            computed_threshold = compute_integral_radius_bound(
            *current_potential, 
            current_epsilon, 
            current_params.tau, 
            C_global, 
            global_functional
        );
        computed_threshold = std::max(computed_threshold, covering_radius);
        used_method_for_log = "integral (C_global based)";
    } else if (current_params.distance_threshold_type == "geometric") {
        // Use geometric radius bound (integral approach)
        computed_threshold = compute_geometric_radius_bound(*current_potential, current_epsilon, current_params.tau);
        used_method_for_log = "geometric (covering radius based)";
    } else { // Default to pointwise, or if type is explicitly "pointwise"
        double max_potential = *std::max_element(current_potential->begin(), current_potential->end());
        double current_min_target_density = min_target_density > 0.0 ? 
                                          min_target_density : 
                                          *std::min_element(target_measure.density.begin(), target_measure.density.end());

        if (current_min_target_density <= 0 || current_params.tau <=0) {
             computed_threshold = std::numeric_limits<double>::max();   
        } else {
            double squared_threshold = -2.0 * current_epsilon *
                std::log(current_params.tau/current_min_target_density) + 2.0 * max_potential;
            computed_threshold = std::sqrt(std::max(0.0, squared_threshold));
        }
        used_method_for_log = "pointwise (tau based)";
    }

    if (current_params.verbose_output) {
        pcout << "Computed distance threshold using " << used_method_for_log << ": " << computed_threshold << std::endl;
    }
    
    double new_proposed_effective_threshold = computed_threshold * 1.1;

    current_distance_threshold = computed_threshold;
}

template <int dim, int spacedim>
double SotSolver<dim, spacedim>::compute_covering_radius() const
{
    namespace bgi = boost::geometry::index;
    
    if (!validate_measures()) {
        throw std::runtime_error("Invalid measures configuration for computing covering radius");
    }
    
    double max_min_distance = 0.0;
    
    // Iterate through all locally owned cells in source domain
    FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
        begin_filtered(IteratorFilters::LocallyOwnedCell(),
                      source_measure.dof_handler->begin_active()),
        end_filtered(IteratorFilters::LocallyOwnedCell(),
                    source_measure.dof_handler->end());
                    
    std::vector<double> local_min_distances;
    
    // For each cell, find the minimum distance to any target point
    for (auto cell = begin_filtered; cell != end_filtered; ++cell) {
        const Point<spacedim>& cell_center = cell->center();
        
        // Use rtree to find the nearest target point - fixed to match container type
        std::vector<IndexedPoint> nearest_results;
        target_measure.rtree.query(bgi::nearest(cell_center, 1), std::back_inserter(nearest_results));
        
        if (!nearest_results.empty()) {
            const Point<spacedim>& nearest_point = nearest_results[0].first;
            const double distance = (nearest_point - cell_center).norm();
            local_min_distances.push_back(distance);
        }
    }
    
    // Find maximum of all minimum distances
    if (!local_min_distances.empty()) {
        max_min_distance = *std::max_element(local_min_distances.begin(), local_min_distances.end());
    }
    
    // Synchronize across MPI processes
    return Utilities::MPI::max(max_min_distance, mpi_communicator);
}

template <int dim, int spacedim>
double SotSolver<dim, spacedim>::compute_integral_radius_bound(
    const Vector<double>& potentials,
    double epsilon,      // Regularization parameter (epsilon in formula)
    double tolerance,   // Tolerance for relative error (tau in formula)
    double C_value,     // Accumulated C_global (C in formula)
    double current_functional_val) const // J_epsilon(psi)
{
    /**
     * Computes the integral radius bound R_int based on the formula:
     * R_int^2 >= 2*M + 2*epsilon*log( (epsilon*C_value) / (tolerance*|J_epsilon(psi)|) )
     * where M is the max potential value.
     */

    double max_potential = *std::max_element(potentials.begin(), potentials.end());

    double abs_functional_val = std::abs(current_functional_val);
    const double safety_value = 1e-10; // Consistent with geometric_radius_bound
    if (abs_functional_val < safety_value) {
        abs_functional_val = safety_value;
    }

    double log_numerator = epsilon * C_value;
    double log_denominator = tolerance * abs_functional_val;

    // log_numerator must be positive (epsilon > 0, C_value > 0 ensured by checks)
    // log_denominator must be positive (tolerance > 0, abs_functional_val >= safety_value > 0 ensured by checks)
    double log_argument = log_numerator / log_denominator;
    
    double radius_squared = 2.0 * max_potential + 2.0 * epsilon * std::log(log_argument);

    return std::sqrt(std::max(0.0, radius_squared));
}

template <int dim, int spacedim>
double SotSolver<dim, spacedim>::compute_geometric_radius_bound(
    const Vector<double>& potentials,
    const double epsilon,
    const double tolerance) const
{
    if (!validate_measures() || potentials.size() != target_measure.points.size()) {
        throw std::runtime_error("Invalid configuration for computing geometric radius bound");
    }
    
    // Use cached covering radius value instead of computing it again
    // (if it's not already computed, use the method)
    double r0 = covering_radius > 0.0 ? covering_radius : compute_covering_radius();
    
    // Compute potential range Γ(ψ) = M - m
    double min_potential = *std::min_element(potentials.begin(), potentials.end());
    double max_potential = *std::max_element(potentials.begin(), potentials.end());
    double potential_range = max_potential - min_potential;
    
    // Use cached minimum target density
    double min_density = min_target_density > 0.0 ? 
                        min_target_density : 
                        *std::min_element(target_measure.density.begin(), target_measure.density.end());
    
    // Use current functional value or safety value if close to zero
    double functional_value = std::abs(global_functional);
    const double safety_value = 1e-10;
    if (functional_value < safety_value) {
        functional_value = safety_value;
    }
    
    // Calculate the radius bound R_geom according to the formula:
    // R_geom^2 ≥ R_0^2 + 2Γ(ψ) + 2ε ln(ε/(ν_min * τ * |J_ε(ψ)|))
    double log_term = std::log(epsilon / (min_density * tolerance * functional_value));
    double radius_squared = r0 * r0 + 2.0 * potential_range + 2.0 * epsilon * log_term;
    
    // Ensure radius is not less than covering radius
    if (radius_squared < r0 * r0) {
        radius_squared = r0 * r0;
    }
    
    return std::sqrt(radius_squared);
}

template <int dim, int spacedim>
std::vector<std::size_t> SotSolver<dim, spacedim>::find_nearest_target_points(
    const Point<spacedim>& query_point) const
{
    namespace bgi = boost::geometry::index;
    std::vector<std::size_t> indices;
    
    // Finds indices of target points in `target_measure.rtree` that are within the current distance threshold from the query point through filter provided by `bgi`.
    for (const auto& indexed_point : target_measure.rtree |
        bgi::adaptors::queried(bgi::satisfies([&](const IndexedPoint& p) {
            return distance_function(p.first, query_point) <= current_distance_threshold;
        })))
    {
        indices.push_back(indexed_point.second);
    }

    return indices;
}

template <int dim, int spacedim>
unsigned int SotSolver<dim, spacedim>::get_last_iteration_count() const
{
    return solver_control ? solver_control->last_step() : 0;
}

template <int dim, int spacedim>
bool SotSolver<dim, spacedim>::get_convergence_status() const
{
    return solver_control && solver_control->last_check() == SolverControl::success;
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::evaluate_weighted_barycenters(
    const Vector<double>& potentials,
    std::vector<Point<spacedim>>& barycenters_out,
    const SotParameterManager::SolverParameters& params)
{
    if (!validate_measures()) {
        throw std::runtime_error("Invalid measures configuration");
    }

    // Barycenter evaluation parameters
    double solver_tolerance = params.tolerance;
    unsigned int max_iterations = params.max_iterations;

    bool use_componentwise = (params.solver_control_type == "componentwise");

    try
    {
        unsigned int n_iter = 0;

        Timer timer;
        timer.start();
        current_potential = &potentials;

        if (distance_name == "euclidean") {
            compute_weighted_barycenters_euclidean(
                *current_potential, barycenters_out);
                
        } else if (distance_name == "spherical")
        {
            for (n_iter = 0; n_iter < max_iterations; ++n_iter)
            {
                // evaluate barycenters_gradients
                // Initialize barycenters_grads with the correct size
                barycenters_grads = std::vector<Vector<double>>(target_measure.points.size(), Vector<double>(spacedim));
                
                // Compute weighted barycenters using non-Euclidean distance
                compute_weighted_barycenters_non_euclidean(
                    *current_potential, barycenters_grads, barycenters_out);
                
                // evaluated inside `compute_weighted_barycenters_non_euclidean`
                double l2_norm = barycenters_gradients.l2_norm();

                if (l2_norm < solver_tolerance) {
                    pcout << "Iteration " << CYAN << n_iter + 1 << RESET
                      << " - L-2 gradient norm: " << Color::green << l2_norm << " < " << solver_tolerance << RESET << std::endl;
                    break;
                } else {

                    pcout << "Iteration " << CYAN << n_iter + 1 << RESET
                      << " - L-2 gradient norm: " << Color::yellow << l2_norm << " > " << solver_tolerance << RESET << std::endl;

                    for (unsigned int i=0; i<target_measure.points.size();++i)
                    {
                        barycenters_out[i] = distance_function_exponential_map(barycenters_out[i], barycenters_grads[i]);
                    }
                }
            } 
        }

        timer.stop();
        
        pcout << Color::green << Color::bold << "Optimization completed:" << std::endl
        << "  Time taken: " << timer.wall_time() << " seconds" << std::endl
        << "  Iterations: " << n_iter+1 << std::endl
        << "  Distance type: " << distance_name << std::endl << Color::reset;
        
    } catch (SolverControl::NoConvergence& exc) {
        pcout << "Warning: Barycenters evaluation did not converge" << std::endl
        << "  Iterations: " << exc.last_step << std::endl
        << "  Residual: " << exc.last_residual << std::endl;
        throw;
    }

    // Reset solver state
    current_potential = nullptr;
}

template <int dim, int spacedim>
void SotSolver<dim,spacedim>::compute_weighted_barycenters_non_euclidean(
    const Vector<double>& potentials,
    std::vector<Vector<double>>& barycenters_gradients_out,
    std::vector<Point<spacedim>>& barycenters_out
)
{
    // Store current potentials for use in local assembly
    current_potential = &potentials;
    current_epsilon = current_params.epsilon;

    barycenters_gradients.reinit(spacedim*target_measure.points.size());
    Vector<double> local_process_barycenters(spacedim*target_measure.points.size());

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
        FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
            begin_filtered(IteratorFilters::LocallyOwnedCell(),
                          source_measure.dof_handler->begin_active()),
            end_filtered(IteratorFilters::LocallyOwnedCell(),
                        source_measure.dof_handler->end());
        
        // Function call
        auto function_call = [this, &barycenters_out](
            CopyData& copy,
            const Point<spacedim> &x,
            const std::vector<std::size_t> &cell_target_indices,
            const std::vector<double> &exp_terms,
            const std::vector<double> &target_densities,
            const double &density_value,
            const double &JxW,
            const double &total_sum_exp,
            const double &max_exponent,
            const double &current_epsilon)
        {
            const double scale = density_value * JxW / total_sum_exp;
    
            #pragma omp simd
            for (size_t i = 0; i < exp_terms.size(); ++i) { 
                if (exp_terms[i] > 0.0) {
                    auto v = distance_function_gradient(barycenters_out[cell_target_indices[i]], x);
                    for (unsigned int d = 0; d < spacedim; ++d)
                        copy.barycenters_values[spacedim*cell_target_indices[i] + d] += scale * (exp_terms[i]) * v[d];
                }
            }
        };

        // Parallel assembly using WorkStream
        WorkStream::run(
            begin_filtered,
            end_filtered,
            [this, &function_call](const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
                   ScratchData& scratch,
                   CopyData& copy) {
                this->local_assemble(
                    cell, scratch, copy, function_call);
            },
            [this, &local_process_barycenters](const CopyData& copy) {
                local_process_barycenters += copy.barycenters_values;
            },
            scratch_data,
            copy_data);

        // Synchronize across MPI processes
        Utilities::MPI::sum(local_process_barycenters, mpi_communicator, barycenters_gradients);

        // Copy result to output barycenters_gradients TODO why do I need this?
        // Resize output vector and fill with barycenters_gradients data
        barycenters_gradients_out.resize(target_measure.points.size());
        for (unsigned int i = 0; i < target_measure.points.size(); ++i) {
            for (unsigned int d = 0; d < spacedim; ++d) {
                // - for gradient descent
                barycenters_gradients_out[i][d] = -barycenters_gradients[spacedim * i + d];
            }
        }

    } catch (const std::exception& e) {
        pcout << "Error in functional evaluation: " << e.what() << std::endl;
        throw;
    }
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::local_assemble(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
    ScratchData& scratch,
    CopyData& copy,
    std::function<void(CopyData&,
                       const Point<spacedim>&,
                       const std::vector<std::size_t>&,
                       const std::vector<double>&,
                       const std::vector<double>&,
                       const double&,
                       const double&,
                       const double&,
                       const double&,
                       const double&)> function_call)
{
    if (!cell->is_locally_owned())
        return;

    scratch.fe_values.reinit(cell);
    const std::vector<Point<spacedim>>& q_points = scratch.fe_values.get_quadrature_points();
    scratch.fe_values.get_function_values(*source_measure.density, scratch.density_values);

    copy.barycenters_values = 0;
    copy.functional_value = 0.0;
    copy.gradient_values = 0;
    copy.local_C_sum = 0.0; // Accumulator for C_sum on this cell

    const unsigned int n_q_points = q_points.size();
    const double epsilon_inv = 1.0 / current_epsilon;
    const bool use_log_sum_exp = current_params.use_log_sum_exp_trick;

    std::vector<std::size_t> cell_target_indices = find_nearest_target_points(cell->center());
    if (cell_target_indices.empty()) return;

    const unsigned int n_target_points = cell_target_indices.size();

    std::vector<Point<spacedim>> target_positions(n_target_points);
    std::vector<double> target_densities(n_target_points);
    std::vector<double> potential_values(n_target_points);

    for (size_t i = 0; i < n_target_points; ++i) {
        const size_t idx = cell_target_indices[i];
        target_positions[i] = target_measure.points[idx];
        target_densities[i] = target_measure.density[idx];
        potential_values[i] = (*current_potential)[idx];
    }

    for (unsigned int q = 0; q < n_q_points; ++q) {
        const Point<spacedim>& x = q_points[q];
        const double density_value = scratch.density_values[q];
        const double JxW = scratch.fe_values.JxW(q);

        double total_sum_exp = 0.0;
        double max_exponent = -std::numeric_limits<double>::max();
        std::vector<double> exp_terms(n_target_points);

        if (use_log_sum_exp) {
            // First pass: find maximum exponent
            #pragma omp simd reduction(max:max_exponent)
            for (size_t i = 0; i < n_target_points; ++i) {
                const double local_dist2 = std::pow(distance_function(x, target_positions[i]), 2);
                const double exponent = (potential_values[i] - 0.5 * local_dist2) * epsilon_inv;
                max_exponent = std::max(max_exponent, exponent);
            }
            
            // Second pass: compute shifted exponentials
            #pragma omp simd reduction(+:total_sum_exp)
            for (size_t i = 0; i < n_target_points; ++i) {
                const double local_dist2 = std::pow(distance_function(x, target_positions[i]), 2);
                const double shifted_exp = std::exp((potential_values[i] - 0.5 * local_dist2) * epsilon_inv - max_exponent);
                exp_terms[i] = target_densities[i] * shifted_exp;
                total_sum_exp += exp_terms[i];
            }
        } else {
            // Original computation method
            #pragma omp simd reduction(+:total_sum_exp)
            for (size_t i = 0; i < n_target_points; ++i) {
                const double local_dist2 = std::pow(distance_function(x, target_positions[i]), 2);
                exp_terms[i] = target_densities[i] *
                    std::exp((potential_values[i] - 0.5 * local_dist2) * epsilon_inv);
                total_sum_exp += exp_terms[i];
            }
        }

        if (total_sum_exp <= 0.0) continue;
        
        function_call(
            copy,
            x,
            cell_target_indices,
            exp_terms,
            target_densities,
            density_value,
            JxW,
            total_sum_exp,
            max_exponent,
            current_epsilon
        );
    }
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::compute_weighted_barycenters_euclidean(
    const Vector<double>& potentials,
    std::vector<Point<spacedim>>& barycenters_out)
{
    // Store current potentials for use in local assembly
    current_potential = &potentials;
    current_epsilon = current_params.epsilon;

    barycenters.reinit(spacedim*target_measure.points.size());
    Vector<double> local_process_barycenters(spacedim*target_measure.points.size());

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
        FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
            begin_filtered(IteratorFilters::LocallyOwnedCell(),
                          source_measure.dof_handler->begin_active()),
            end_filtered(IteratorFilters::LocallyOwnedCell(),
                        source_measure.dof_handler->end());

        // Function call
        auto function_call = [this](
            CopyData& copy,
            const Point<spacedim> &x,
            const std::vector<std::size_t> &cell_target_indices,
            const std::vector<double> &exp_terms,
            const std::vector<double> &target_densities,
            const double &density_value,
            const double &JxW,
            const double &total_sum_exp,
            const double &max_exponent,
            const double &current_epsilon)
        {
            const double scale = density_value * JxW / total_sum_exp;
            
            #pragma omp simd
            for (size_t i = 0; i < exp_terms.size(); ++i) {
                if (exp_terms[i] > 0.0) {
                    for (unsigned int d = 0; d < spacedim; ++d) {
                        copy.barycenters_values[spacedim*cell_target_indices[i] + d] += scale * (exp_terms[i]) * x[d];
                    }
                }
            }
        };

        // Parallel assembly using WorkStream
        WorkStream::run(
            begin_filtered,
            end_filtered,
            [this, &function_call](const typename DoFHandler<dim, spacedim>::active_cell_iterator& cell,
                   ScratchData& scratch,
                   CopyData& copy) {
                this->local_assemble(cell, scratch, copy, function_call);
            },
            [this, &local_process_barycenters](const CopyData& copy) {
                local_process_barycenters += copy.barycenters_values;
            },
            scratch_data,
            copy_data);

        // Synchronize across MPI processes
        Utilities::MPI::sum(local_process_barycenters, mpi_communicator, barycenters);

        // Copy result to output barycenters TODO why do I need this?
        // Resize output vector and fill with barycenters data
        barycenters_out.resize(target_measure.points.size());
        for (unsigned int i = 0; i < target_measure.points.size(); ++i) {
            for (unsigned int d = 0; d < spacedim; ++d) {
                barycenters_out[i][d] = barycenters[spacedim * i + d];
            }
        }

    } catch (const std::exception& e) {
        pcout << "Error in functional evaluation: " << e.what() << std::endl;
        throw;
    }
}

template <int dim, int spacedim>
void SotSolver<dim, spacedim>::get_potential_conditioned_density(
    const DoFHandler<dim, spacedim> &dof_handler,
    const Mapping<dim, spacedim> &mapping,
    const Vector<double> &potential,
    const std::vector<unsigned int> &potential_indices,
    std::vector<Vector<double>> &conditioned_densities,
    Vector<double> &number_of_non_thresholded_targets)
{
    pcout << "Computing conditioned densities for specified potential indices\n";

    conditioned_densities.resize(potential_indices.size());
    
    unsigned int n_active_cells = dof_handler.get_triangulation().n_active_cells();
    number_of_non_thresholded_targets.reinit(
        n_active_cells);
    for (unsigned int idensity = 0; idensity < conditioned_densities.size(); ++idensity)
        conditioned_densities[idensity].reinit(n_active_cells);
        
    double epsilon_inv = 1.0 / current_epsilon;
    std::set<std::size_t> all_found_targets;
    
    try {
        pcout << "Init conditional densities..." << std::endl;
        bool use_simplex = (dynamic_cast<const FE_SimplexP<dim>*>(&*source_measure.fe) != nullptr);

        std::unique_ptr<Quadrature<dim>> quadrature;
            if (use_simplex) {
                quadrature = std::make_unique<QGaussSimplex<dim>>(source_measure.quadrature_order);
            } else {
                quadrature = std::make_unique<QGauss<dim>>(source_measure.quadrature_order);
            }
        FEValues<dim, spacedim> fe_values(
            *source_measure.mapping,
            *source_measure.fe,
            *quadrature,
            update_values | update_quadrature_points | update_JxW_values);

        unsigned int local_cell_index = 0;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
            {
                ++local_cell_index;
                continue;
            }

            fe_values.reinit(cell);
            const std::vector<Point<spacedim>>& q_points = fe_values.get_quadrature_points();
            const unsigned int n_q_points = q_points.size();
            std::vector<double> density_values((*quadrature).size());
            fe_values.get_function_values(*source_measure.density, density_values);

            std::vector<std::size_t> cell_target_indices = find_nearest_target_points(cell->center());
            if (cell_target_indices.empty()) return;

            // Add these indices to our accumulated set
            all_found_targets.insert(cell_target_indices.begin(), cell_target_indices.end());

            number_of_non_thresholded_targets[local_cell_index] = cell_target_indices.size();

            const unsigned int n_target_points = cell_target_indices.size();
            
            std::vector<Point<spacedim>> target_positions(n_target_points);
            std::vector<double> target_densities(n_target_points);
            std::vector<double> potential_values(n_target_points);

            for (size_t i = 0; i < n_target_points; ++i) {
                const size_t idx = cell_target_indices[i];
                target_positions[i] = target_measure.points[idx];
                target_densities[i] = target_measure.density[idx];
                potential_values[i] = potential[idx];
            }
            
            for (unsigned int q = 0; q < n_q_points; ++q) {
                const Point<spacedim>& x = q_points[q];
                const double density_value = density_values[q];
                const double JxW = fe_values.JxW(q);
        
                double total_sum_exp = 0.0;
                double max_exponent = -std::numeric_limits<double>::max();
                std::vector<double> exp_terms(n_target_points);
        
                if (current_params.use_log_sum_exp_trick) {
                    // First pass: find maximum exponent
                    #pragma omp simd reduction(max:max_exponent)
                    for (size_t i = 0; i < n_target_points; ++i) {
                        const double local_dist2 = std::pow(distance_function(x, target_positions[i]), 2);
                        const double exponent = (potential_values[i] - 0.5 * local_dist2) * epsilon_inv;
                        max_exponent = std::max(max_exponent, exponent);
                    }
                    // Second pass: compute shifted exponentials
                    #pragma omp simd reduction(+:total_sum_exp)
                    for (size_t i = 0; i < n_target_points; ++i) {
                        const double local_dist2 = std::pow(distance_function(x, target_positions[i]), 2);
                        const double shifted_exp = std::exp((potential_values[i] - 0.5 * local_dist2) * epsilon_inv - max_exponent);
                        exp_terms[i] = shifted_exp;
                        total_sum_exp += target_densities[i] * exp_terms[i];
                    }
                } else {
                    // Original computation method
                    #pragma omp simd reduction(+:total_sum_exp)
                    for (size_t i = 0; i < n_target_points; ++i) {
                        const double local_dist2 = std::pow(distance_function(x, target_positions[i]), 2);
                        exp_terms[i] = 
                            std::exp((potential_values[i] - 0.5 * local_dist2) * epsilon_inv);
                        total_sum_exp += target_densities[i] * exp_terms[i];
                    }
                }
        
                if (total_sum_exp <= 0.0)
                {
                    ++local_cell_index;
                    continue;
                }

                for (unsigned int idensity = 0; idensity < potential_indices.size(); ++idensity)
                {
                    conditioned_densities[idensity][local_cell_index] += JxW * density_value * (exp_terms[potential_indices[idensity]]/total_sum_exp);
                }
            }
            ++local_cell_index;
        }
            
        pcout << "Traversed targets: " << all_found_targets.size() << " targets over a total of " << potential.size() << std::endl;
        
        number_of_non_thresholded_targets.compress(VectorOperation::insert);
        for (unsigned int idensity = 0; idensity < conditioned_densities.size(); ++idensity)
        {
            conditioned_densities[idensity].compress(VectorOperation::add);
        }
    } catch (const std::exception& e) {
        pcout << "Error in conditional densities initialization: " << e.what() << std::endl;
        throw;
    }
}

// Explicit instantiation
template class SotSolver<2>;
template class SotSolver<3>;
template class SotSolver<2, 3>;

