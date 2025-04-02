#include "SemiDiscreteOT/core/Lloyd.h"

namespace fs = std::filesystem;

using namespace dealii;

template <int dim, int spacedim>
Lloyd<dim, spacedim>::Lloyd(const MPI_Comm &comm)
    : SemiDiscreteOT<dim, spacedim>(comm)
    , param_manager_lloyd(comm)
{}

template <int dim, int spacedim>
void Lloyd<dim, spacedim>::run_lloyd()
{
    Timer timer;
    timer.start();

    this->pcout << Color::yellow << Color::bold << "Starting Lloyd algorithm with " << this->target_points.size()
          << " target points and " << this->source_density.size() << " source points" << Color::reset << std::endl;

    unsigned int n_iter = 0;
    run_sot_iteration(n_iter);
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
    this->save_results(potential, "potentials");

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