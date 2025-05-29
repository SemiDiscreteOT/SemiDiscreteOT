#include "SemiDiscreteOT/core/Lloyd.h"

namespace fs = std::filesystem;

using namespace dealii;

template <int dim, int spacedim>
Lloyd<dim, spacedim>::Lloyd(const MPI_Comm &comm)
    : SemiDiscreteOT<dim, spacedim>(comm)
    , param_manager_lloyd(comm)
{}

template <int dim, int spacedim>
void Lloyd<dim, spacedim>::run_lloyd(
    const double absolute_threshold,
    const unsigned int max_iterations)
{
    Timer timer;
    timer.start();

    this->pcout << Color::yellow << Color::bold << "Starting Lloyd algorithm with " << this->target_points.size()
          << " target points and " << this->source_density.size() << " source points" << Color::reset << std::endl;

    double l2_norm = 0.0;
    
    for (unsigned int n_iter = 0; n_iter < max_iterations; ++n_iter)
    {
        run_sot_iteration(n_iter);
        run_centroid_iteration(n_iter);
        compute_step_norm(barycenters, this->target_points, l2_norm);
        if (l2_norm < absolute_threshold) {
            this->pcout << Color::green << Color::bold << "Lloyd algorithm converged in "
                        << n_iter + 1 << " iterations with L2 norm: " << l2_norm 
                        << " < threshold: " << absolute_threshold << Color::reset << std::endl;
            break;
        } else {
            this->pcout << Color::green << Color::bold << "Lloyd: "<< "Lloyd algorithm did not converge in "
                        << n_iter + 1 << " iterations with L2 norm: " << l2_norm 
                        << " > threshold: " << absolute_threshold << Color::reset << std::endl;
            this->sot_solver->setup_target(
                barycenters, this->target_density);
        }
    }
}

template <int dim, int spacedim>
void Lloyd<dim, spacedim>::compute_step_norm(
    const std::vector<Point<spacedim>>& barycenters_next,
    const std::vector<Point<spacedim>>& barycenters_prev,
    double &l2_norm)
{
    Assert(barycenters_next.size() == barycenters_prev.size(),
        ExcDimensionMismatch(barycenters_next.size(), barycenters_prev.size()));

    l2_norm = 0.0;
    // Sum up squared distances between corresponding points
    for (size_t i = 0; i < barycenters_next.size(); ++i) {
        l2_norm += barycenters_next[i].distance_square(barycenters_prev[i]);
    }

    // Compute root mean squared distance
    l2_norm = std::sqrt(l2_norm / barycenters_next.size());

    // Print the displacement norm
    this->pcout << "  L2 norm: " << l2_norm << std::endl;
}

template <int dim, int spacedim>
void Lloyd<dim, spacedim>::run_centroid_iteration(
    const unsigned int n_iter)
{
    Timer timer;
    timer.start();

    // Configure solver parameters
    SotParameterManager::SolverParameters& solver_config = this->solver_params;

    if (solver_config.use_epsilon_scaling && this->epsilon_scaling_handler) {
        this->pcout << "Using epsilon scaling with EpsilonScalingHandler:" << std::endl
              << "  Initial epsilon: " << solver_config.regularization_param << std::endl
              << "  Scaling factor: " << solver_config.epsilon_scaling_factor << std::endl
              << "  Number of steps: " << solver_config.epsilon_scaling_steps << std::endl;
        // Compute epsilon distribution for a single level
        std::vector<std::vector<double>> epsilon_distribution =
            this->epsilon_scaling_handler->compute_epsilon_distribution(1);

        if (!epsilon_distribution.empty() && !epsilon_distribution[0].empty()) {
            const auto& epsilon_sequence = epsilon_distribution[0];

            // Run optimization for each epsilon value
            for (size_t i = 0; i < epsilon_sequence.size(); ++i) {
                this->pcout << "\nEpsilon scaling step " << i + 1 << "/" << epsilon_sequence.size()
                      << " (λ = " << epsilon_sequence[i] << ")" << std::endl;

                solver_config.regularization_param = epsilon_sequence[i];

                try {
                    this->sot_solver->compute_weighted_barycenters(potential, barycenters);

                } catch (const SolverControl::NoConvergence& exc) {
                    if (exc.last_step >= this->solver_params.max_iterations) {
                        this->pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << i + 1
                              << " (epsilon=" << epsilon_sequence[i] << "): Max iterations reached"
                              << Color::reset << std::endl;
                    }
                }
            }
        }
    } else {
        // Run single optimization with original epsilon
        try {
            this->sot_solver->compute_weighted_barycenters(potential, barycenters);
        } catch (const SolverControl::NoConvergence& exc) {
            this->pcout << Color::red << Color::bold << "Warning: Optimization did not converge." << Color::reset << std::endl;
        }
    }

    // Save barycenters to text file
    // Only process with rank 0 should write the file
    int rank;
    MPI_Comm_rank(this->mpi_communicator, &rank);
    if (rank == 0) {
        std::string filename = "barycenters_" + std::to_string(n_iter) + ".ply";
        std::ofstream out(filename);
        if (out.is_open()) {
            // Write PLY header
            out << "ply\n";
            out << "format ascii 1.0\n";
            out << "element vertex " << barycenters.size() << "\n";
            for (unsigned int d = 0; d < spacedim; ++d) {
                out << "property float " << (d == 0 ? 'x' : (d == 1 ? 'y' : 'z')) << "\n";
            }
            out << "end_header\n";
            
            // Write vertex data
            for (const auto& point : barycenters) {
                for (unsigned int d = 0; d < spacedim; ++d) {
                    out << point[d] << " ";
                }
                out << "\n";
            }
            out.close();
            this->pcout << "  Barycenters saved to " << filename << std::endl;
        } else {
            this->pcout << Color::red << "  Failed to save barycenters to " << filename << Color::reset << std::endl;
        }
    }

    timer.stop();
    this->pcout << Color::green << Color::bold << "Lloyd: "<< n_iter << " Barycenter iteration completed in " << timer.wall_time() << " seconds" << Color::reset << std::endl;
}

// run single sot optimization with epsilon scaling
template <int dim, int spacedim>
void Lloyd<dim, spacedim>::run_sot_iteration(
    const unsigned int n_iter)
{
    Timer timer;
    timer.start();

    // Configure solver parameters
    SotParameterManager::SolverParameters& solver_config = this->solver_params;

    // // Set up source measure
    // this->sot_solver->setup_source(this->dof_handler_source,
    //                        *this->mapping,
    //                        *this->fe_system,
    //                        this->source_density,
    //                        solver_config.quadrature_order);

    // // Set up target measure
    // this->sot_solver->setup_target(this->target_points, this->target_density);

    // if (n_iter==0)
    potential.reinit(this->target_points.size());

    if (solver_config.use_epsilon_scaling && this->epsilon_scaling_handler) {
        this->pcout << "Using epsilon scaling with EpsilonScalingHandler:" << std::endl
              << "  Initial epsilon: " << solver_config.regularization_param << std::endl
              << "  Scaling factor: " << solver_config.epsilon_scaling_factor << std::endl
              << "  Number of steps: " << solver_config.epsilon_scaling_steps << std::endl;
        // Compute epsilon distribution for a single level
        std::vector<std::vector<double>> epsilon_distribution =
            this->epsilon_scaling_handler->compute_epsilon_distribution(1);

        if (!epsilon_distribution.empty() && !epsilon_distribution[0].empty()) {
            const auto& epsilon_sequence = epsilon_distribution[0];

            // Run optimization for each epsilon value
            for (size_t i = 0; i < epsilon_sequence.size(); ++i) {
                this->pcout << "\nEpsilon scaling step " << i + 1 << "/" << epsilon_sequence.size()
                      << " (λ = " << epsilon_sequence[i] << ")" << std::endl;

                solver_config.regularization_param = epsilon_sequence[i];

                try {
                    this->sot_solver->solve(potential, solver_config);

                } catch (const SolverControl::NoConvergence& exc) {
                    if (exc.last_step >= this->solver_params.max_iterations) {
                        this->pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << i + 1
                              << " (epsilon=" << epsilon_sequence[i] << "): Max iterations reached"
                              << Color::reset << std::endl;
                    }
                }
            }
        }
    } else {
        // Run single optimization with original epsilon
        try {
            this->sot_solver->solve(potential, solver_config);
        } catch (const SolverControl::NoConvergence& exc) {
            this->pcout << Color::red << Color::bold << "Warning: Optimization did not converge." << Color::reset << std::endl;
        }
    }

    // Save final results
    this->save_results(potential, "potentials_"+std::to_string(n_iter));

    timer.stop();
    this->pcout << "\n" << Color::green << Color::bold << "Lloyd: "<< n_iter << " SOT iteration completed in " << timer.wall_time() << " seconds" << Color::reset << std::endl;
}

template <int dim, int spacedim>
void Lloyd<dim, spacedim>::run()
{
    param_manager_lloyd.print_parameters();

    if (this->solver_params.use_epsilon_scaling) {
        this->epsilon_scaling_handler = std::make_unique<EpsilonScalingHandler>(
            this->mpi_communicator,
            this->solver_params.regularization_param,
            this->solver_params.epsilon_scaling_factor,
            this->solver_params.epsilon_scaling_steps
        );
    }

    if (this->selected_task == "lloyd")
    {
        run_lloyd();
    }
    else
    {
        this->pcout << "No valid task selected" << std::endl;
    }
}

template class Lloyd<2>;
template class Lloyd<3>;
template class Lloyd<2, 3>;