#include "SemiDiscreteOT/core/WassersteinBarycenters.h"

namespace fs = std::filesystem;

using namespace dealii;

template <int dim, int spacedim>
WassersteinBarycenters<dim, spacedim>::WassersteinBarycenters(
    const unsigned int n_measures,
    const std::vector<double> weights,
    const MPI_Comm &comm,
    UpdateMode update_flag)
    : n_measures(n_measures)
    , weights(weights)
    , update_mode(update_flag)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , param_manager_wasserstein_barycenters(comm)
{
    // Check that we have at least 2 measures for meaningful barycenters
    Assert(n_measures >= 2,
        ExcMessage("Wasserstein Barycenters require at least 2 measures, but only " + 
                std::to_string(n_measures) + " provided."));

    // Check that weights vector has correct size
    Assert(weights.size() == n_measures,
        ExcMessage("Number of weights (" + std::to_string(weights.size()) + 
               ") must match number of measures (" + std::to_string(n_measures) + ")"));

    // Check that weights sum to 1
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    Assert(std::abs(sum_weights - 1.0) < 1e-10,
        ExcMessage("Sum of weights must be 1, but is " + std::to_string(sum_weights)));

    // Check that update_flag is a valid UpdateMode
    Assert(update_flag == UpdateMode::TargetSupportPointsOnly ||
        update_flag == UpdateMode::TargetMeasureOnly ||
        update_flag == UpdateMode::Both,
        ExcMessage("Invalid update mode provided. Must be one of: TargetSupportPointsOnly, "
               "TargetMeasureOnly, or Both"));

    // Initialize sot_solvers vector with n_measures elements
    sot_solvers.resize(n_measures);
    for (unsigned int i = 0; i < n_measures; ++i) {
        sot_solvers[i] = std::make_unique<SemiDiscreteOT<dim, spacedim>>(comm);
    }
    potentials.resize(n_measures);
}

template <int dim, int spacedim>
void WassersteinBarycenters<dim, spacedim>::run_wasserstein_barycenters(
    const double absolute_threshold_measure,
    const double absolute_threshold_support_points,
    const unsigned int max_iterations,
    const double alpha)
{    
    Timer timer;
    timer.start();

    pcout << Color::yellow << Color::bold << "Starting Wassertein Barycenters algorithm with " << n_measures << " measures:\n" << std::endl;
    for (unsigned int i = 0; i < n_measures; ++i)
        pcout << "  source measure " << i + 1 << " dofs: " << sot_solvers[i]->source_density.size() << "; target measure: " << sot_solvers[i]->target_density.size() << std::endl;
    pcout << Color::reset << std::endl;

    double l2_norm_step_measure = 0.0;
    double l2_norm_step_support_points = 0.0;

    // Check that all solvers have the same number of target points
    if (n_measures > 1) {
        const size_t first_target_size = sot_solvers[0]->target_points.size();
        for (unsigned int i = 1; i < n_measures; ++i) {
            Assert(sot_solvers[i]->target_points.size() == first_target_size,
                   ExcMessage("All measures must have the same number of target points. "
                              "Measure 0 has " + std::to_string(first_target_size) + 
                              " points, but measure " + std::to_string(i) + 
                              " has " + std::to_string(sot_solvers[i]->target_points.size()) + " points."));
        }
    }
    
    // Init barycenters
    barycenters.resize(sot_solvers[0]->target_points.size());
    for (size_t i = 0; i < barycenters.size(); ++i) {
        barycenters[i] = {
            sot_solvers[0]->target_points[i], sot_solvers[0]->target_density[i]};
    }
    barycenters_prev.resize(sot_solvers[0]->target_points.size());
    for (size_t i = 0; i < barycenters_prev.size(); ++i) {
        barycenters_prev[i] = {
            sot_solvers[0]->target_points[i], sot_solvers[0]->target_density[i]};
    }

    // Be sure that all sot_solvers have the same target density
    for (unsigned int i = 1; i < n_measures; ++i) {
        sot_solvers[i]->target_density = sot_solvers[0]->target_density;
        sot_solvers[i]->target_points = sot_solvers[0]->target_points;
        sot_solvers[i]->sot_solver->setup_target(
            sot_solvers[i]->target_points,
            sot_solvers[i]->target_density
        );
    }

    int rank;
    MPI_Comm_rank(this->sot_solvers[0]->mpi_communicator, &rank);
    if (rank == 0) {
        std::string filename = "wbarycenters_" + std::to_string(0) + ".ply";
        std::ofstream out(filename);
        if (out.is_open()) {
            // Write PLY header
            out << "ply\n";
            out << "format ascii 1.0\n";
            out << "element vertex " << barycenters.size() << "\n";
            for (unsigned int d = 0; d < spacedim; ++d) {
                out << "property float " << (d == 0 ? 'x' : (d == 1 ? 'y' : 'z')) << "\n";
            }
            // Add density property
            out << "property float density\n";
            out << "end_header\n";
            
            // Write vertex data
            for (unsigned int point = 0; point < barycenters.size(); ++point) {
                // Write position coordinates
                for (unsigned int d = 0; d < spacedim; ++d) {
                    out << barycenters[point].first[d] << " ";
                }
                // Write density value
                out << sot_solvers[0]->target_density[point] << "\n";
            }
            out.close();
            this->pcout << "Wasserstein barycenters saved to " << filename << std::endl << std::endl;
        } else {
            this->pcout << Color::red << "  Failed to save barycenters to " << filename << Color::reset << std::endl;
        }
    }

    // Alternate sot iterations with wasserstein barycenters updates
    for (unsigned int n_iter = 0; n_iter < max_iterations; ++n_iter)
    {
        run_sot_iterations(n_iter);
        run_optimize_barycenters(n_iter, alpha);

        compute_step_norm(
            barycenters,
            barycenters_prev,
            l2_norm_step_measure,
            l2_norm_step_support_points);
        
        // Update barycenters for next iteration
        barycenters_prev = barycenters;

        if (l2_norm_step_measure < absolute_threshold_measure && l2_norm_step_support_points < absolute_threshold_support_points) {
            pcout << Color::green << Color::bold << "WassersteinBarycenters algorithm converged in "
                        << n_iter + 1 << " iterations with:\n" <<
                        "   L2 norm of measure update step: " << l2_norm_step_measure 
                        << " < threshold: " << absolute_threshold_measure << std::endl <<
                        "   L2 norm of support points update step: " << l2_norm_step_support_points 
                        << " < threshold: " << absolute_threshold_support_points << 
                        Color::reset << std::endl;
            break;
        } else {
            pcout << Color::red << Color::bold << "WassersteinBarycenters algorithm converged in "
                        << n_iter + 1 << " iterations with:\n" <<
                        "   L2 norm of measure update step: " << l2_norm_step_measure 
                        << " < threshold: " << absolute_threshold_measure << std::endl <<
                        "   L2 norm of support points update step: " << l2_norm_step_support_points 
                        << " < threshold: " << absolute_threshold_support_points << 
                        Color::reset << std::endl;

            for (unsigned int i = 1; i < n_measures; ++i) {
                if (update_mode == UpdateMode::TargetSupportPointsOnly ||
                    update_mode == UpdateMode::Both) {
                    for (unsigned int j = 0; j < barycenters.size(); ++j) {
                        sot_solvers[i]->target_points[j] = barycenters[j].first;
                    }
                } else if (update_mode == UpdateMode::TargetMeasureOnly ||
                    update_mode == UpdateMode::Both) {
                    for (unsigned int j = 0; j < barycenters.size(); ++j) {
                        sot_solvers[i]->target_density[j] = barycenters[j].second;
                    }
                }

                sot_solvers[i]->sot_solver->setup_target(
                    sot_solvers[i]->target_points,
                    sot_solvers[i]->target_density
                );
            }
        }
        pcout << std::endl;
    }
}

template <int dim, int spacedim>
void WassersteinBarycenters<dim, spacedim>::compute_step_norm(
    const std::vector<std::pair<Point<spacedim>, double>>& barycenters_next,
    const std::vector<std::pair<Point<spacedim>, double>>& barycenters_prev,
    double &l2_norm_step_measure,
    double &l2_norm_step_support_points)
{
    Assert(barycenters_next.size() == barycenters_prev.size(),
        ExcDimensionMismatch(barycenters_next.size(), barycenters_prev.size()));

    l2_norm_step_measure = 0.0;
    // Sum up squared distances between corresponding points
    for (size_t i = 0; i < barycenters_next.size(); ++i) {
        l2_norm_step_measure += barycenters_next[i].first.distance_square(barycenters_prev[i].first);
        l2_norm_step_support_points += std::pow(barycenters_next[i].second-(barycenters_prev[i].second), 2);
    }

    // Compute root mean squared distance
    l2_norm_step_measure = std::sqrt(l2_norm_step_measure / barycenters_next.size());
    l2_norm_step_support_points = std::sqrt(l2_norm_step_support_points / barycenters_next.size());
}

template <int dim, int spacedim>
void WassersteinBarycenters<dim, spacedim>::run_optimize_barycenters(
    const unsigned int n_iter,
    const double alpha)
{
    pcout << std::endl << Color::yellow << Color::bold << "Starting Wasserstein barycenter update iteration " << n_iter+1 << " with " << barycenters.size()
            << " barycenters" << Color::reset << std::endl;

    Timer timer;
    timer.start();

    for (unsigned int i = 0; i < n_measures; ++i) {
        // Configure solver parameters
        SotParameterManager::SolverParameters& solver_config = sot_solvers[i]->solver_params;
    
        try
        {
    
            if (solver_config.use_epsilon_scaling &&
                sot_solvers[i]->epsilon_scaling_handler) {
                pcout << "Using epsilon scaling with EpsilonScalingHandler:" << std::endl
                    << "  Initial epsilon: " << solver_config.regularization_param << std::endl
                    << "  Scaling factor: " << solver_config.epsilon_scaling_factor << std::endl
                    << "  Number of steps: " << solver_config.epsilon_scaling_steps << std::endl;
                // Compute epsilon distribution for a single level
                std::vector<std::vector<double>> epsilon_distribution =
                    sot_solvers[i]->epsilon_scaling_handler->compute_epsilon_distribution(1);
    
                if (!epsilon_distribution.empty() && !epsilon_distribution[0].empty()) {
                    const auto& epsilon_sequence = epsilon_distribution[0];
    
                    // Run optimization for each epsilon value
                    for (size_t i = 0; i < epsilon_sequence.size(); ++i) {
                        pcout << "\nEpsilon scaling step " << i + 1 << "/" << epsilon_sequence.size()
                            << " (λ = " << epsilon_sequence[i] << ")" << std::endl;
    
                        solver_config.regularization_param = epsilon_sequence[i];
    
                        try {
                            sot_solvers[i]->sot_solver->compute_grad_target(
                                barycenters_gradients,
                                potentials[i],
                                barycenters_prev,
                                solver_config);
    
                        } catch (const SolverControl::NoConvergence& exc) {
                            if (exc.last_step >= sot_solvers[i]->solver_params.max_iterations) {
                                pcout << Color::red << Color::bold << "  Warning: Barycenter evaluation failed at step " << i + 1
                                    << " (epsilon=" << epsilon_sequence[i] << "): Max iterations reached"
                                    << Color::reset << std::endl;
                            }
                        }
                    }
                }
            } else {
                // Run single Barycenter evaluation with original epsilon
                try {
                    sot_solvers[i]->sot_solver->compute_grad_target(
                        barycenters_gradients,
                        potentials[i],
                        barycenters_prev,
                        solver_config);

                } catch (const SolverControl::NoConvergence& exc) {
                    pcout << Color::red << Color::bold << "Warning: Barycenter evaluation did not converge." << Color::reset << std::endl;
                }
            }
        } catch (const std::exception &e){
            pcout << Color::red << Color::bold << "An exception occurred during barycenter evaluation: " << e.what() << Color::reset << std::endl;
        }
        
        if (update_mode == UpdateMode::TargetSupportPointsOnly)
        {
            for (unsigned int k=0; k<barycenters.size(); ++k)
            {
                for (unsigned int j = 0; j < spacedim; ++j)
                {
                    barycenters_gradients[k].first*=alpha*weights[i];
                    barycenters[k].first = sot_solvers[i]->sot_solver->distance_function_exponential_map(
                        barycenters[k].first, barycenters_gradients[k].first
                    );
                    // barycenters[k].first[j] += alpha*weights[i]*barycenters_gradients[k].first[j];                    
                }
            }          
        }
        else if (update_mode == UpdateMode::TargetMeasureOnly)
        {
            double total_mass = 0.0;
            for (unsigned int k=0; k<barycenters.size(); ++k)
            {
                barycenters[k].second = barycenters[k].second*(std::exp(alpha*barycenters_gradients[k].second*barycenters[k].second));
                total_mass += std::abs(barycenters[k].second);
                // barycenters[k].second -= weights[i]*barycenters_gradients[k].second;
            }
            if (total_mass > 0.0)
            {
                for (unsigned int k=0; k<barycenters.size(); ++k)
                {
                    barycenters[k].second /= total_mass;
                }
            }
            

        }
        else if (update_mode == UpdateMode::Both)
        {
            double total_mass = 0.0;
            for (unsigned int k=0; k<barycenters.size(); ++k)
            {
                for (unsigned int j = 0; j < spacedim; ++j)
                {
                    barycenters_gradients[k].first*=alpha*weights[i];
                    barycenters[k].first = sot_solvers[i]->sot_solver->distance_function_exponential_map(
                        barycenters[k].first, barycenters_gradients[k].first
                    );
                    // barycenters[k].first[j] += alpha*weights[i]*barycenters_gradients[k].first[j];
                }
                barycenters[k].second -= weights[i]*barycenters_gradients[k].second;
                total_mass += std::abs(barycenters[k].second);
            }    
            
            if (total_mass > 0.0)
            {
                for (unsigned int k=0; k<barycenters.size(); ++k)
                {
                    barycenters[k].second /= total_mass;
                }
            }
        }
            
    }

    int rank;
    MPI_Comm_rank(this->sot_solvers[0]->mpi_communicator, &rank);
    if (rank == 0) {
        std::string filename = "wbarycenters_" + std::to_string(n_iter+1) + ".ply";
        std::ofstream out(filename);
        if (out.is_open()) {
            // Write PLY header
            out << "ply\n";
            out << "format ascii 1.0\n";
            out << "element vertex " << barycenters.size() << "\n";
            for (unsigned int d = 0; d < spacedim; ++d) {
                out << "property float " << (d == 0 ? 'x' : (d == 1 ? 'y' : 'z')) << "\n";
            }
            // Add gradient properties
            for (unsigned int d = 0; d < spacedim; ++d) {
                out << "property float gradient_" << (d == 0 ? 'x' : (d == 1 ? 'y' : 'z')) << "\n";
            }
            // Add density and gradient values
            out << "property float density\n";
            out << "property float gradient_density\n";
            out << "end_header\n";
            
            // Write vertex data
            for (unsigned int point = 0; point < barycenters.size(); ++point) {
                // Write position coordinates
                for (unsigned int d = 0; d < spacedim; ++d) {
                    out << barycenters[point].first[d] << " ";
                }
                // Write gradient coordinates
                for (unsigned int d = 0; d < spacedim; ++d) {
                    out << barycenters_gradients[point].first[d] << " ";
                }
                // Write density and gradient values
                out << barycenters[point].second << " ";
                out << barycenters_gradients[point].second << "\n";
            }
            out.close();
            this->pcout << "Wasserstein barycenters and gradients saved to " << filename << std::endl << std::endl;
        } else {
            this->pcout << Color::red << "  Failed to save barycenters to " << filename << Color::reset << std::endl;
        }
    }

    timer.stop();
    pcout << Color::green << Color::bold << "WassersteinBarycenters: "<< n_iter+1 << " Barycenter iteration completed in " << timer.wall_time() << " seconds" << Color::reset << std::endl << std::endl;
}

// run single sot optimization with epsilon scaling
template <int dim, int spacedim>
void WassersteinBarycenters<dim, spacedim>::run_sot_iterations(
    const unsigned int n_iter)
{
    Timer timer;
    timer.start();

    for (unsigned int i = 0; i < n_measures; ++i) {
        pcout << Color::yellow << Color::bold << "Starting SOT iteration " << n_iter+1 << " with " << sot_solvers[i]->target_points.size()
              << " target points and " << sot_solvers[i]->source_density.size() << " source dofs" << Color::reset << std::endl;
    
        // Source and target measures must be set
        Assert(sot_solvers[i]->source_measure.initialized,
            ExcMessage("Source measure must be set before running SOT iteration"));
        Assert(sot_solvers[i]->target_measure.initialized,
            ExcMessage("Target points must be set before running SOT iteration"));
    
    
        // Configure solver parameters
        SotParameterManager::SolverParameters& solver_config = sot_solvers[i]->solver_params;
    
        potentials[i].reinit(sot_solvers[i]->target_points.size());
        try
        {
            if (solver_config.use_epsilon_scaling && sot_solvers[i]->epsilon_scaling_handler)
            {
                pcout << "Using epsilon scaling with EpsilonScalingHandler:" << std::endl
                      << "  Initial epsilon: " << solver_config.regularization_param << std::endl
                      << "  Scaling factor: " << solver_config.epsilon_scaling_factor << std::endl
                      << "  Number of steps: " << solver_config.epsilon_scaling_steps << std::endl;
                // Compute epsilon distribution for a single level
                std::vector<std::vector<double>> epsilon_distribution =
                sot_solvers[i]->epsilon_scaling_handler->compute_epsilon_distribution(1);
    
                if (!epsilon_distribution.empty() && !epsilon_distribution[0].empty())
                {
                    const auto &epsilon_sequence = epsilon_distribution[0];
    
                    // Run optimization for each epsilon value
                    for (size_t i = 0; i < epsilon_sequence.size(); ++i)
                    {
                        pcout << "\nEpsilon scaling step " << i + 1 << "/" << epsilon_sequence.size()
                              << " (λ = " << epsilon_sequence[i] << ")" << std::endl;
    
                        solver_config.regularization_param = epsilon_sequence[i];
    
                        try
                        {
                            sot_solvers[i]->sot_solver->solve(potentials[i], solver_config);
    
                            // Save intermediate results
                            if (i < epsilon_sequence.size() - 1)
                            {
                                std::string eps_suffix = "_eps" + std::to_string(i + 1);
                                sot_solvers[i]->save_results(potentials[i], "potential" + eps_suffix);
                            }
                        }
                        catch (const SolverControl::NoConvergence &exc)
                        {
                            if (exc.last_step >= sot_solvers[i]->solver_params.max_iterations)
                            {
                                pcout << Color::red << Color::bold << "  Warning: Optimization failed at step " << i + 1
                                      << " (epsilon=" << epsilon_sequence[i] << "): Max iterations reached"
                                      << Color::reset << std::endl;
                            }
                        }
                    }
                }
            }
            else
            {
                // Run single optimization with original epsilon
                try
                {
                    sot_solvers[i]->sot_solver->solve(potentials[i], solver_config);
                }
                catch (const SolverControl::NoConvergence &exc)
                {
                    pcout << Color::red << Color::bold << "Warning: Optimization did not converge." << Color::reset << std::endl;
                }
            }
        }
        catch (const std::exception &e)
        {
            pcout << Color::red << Color::bold << "An exception occurred during SOT solve: " << e.what() << Color::reset << std::endl;
        }
        
        timer.stop();
        const double total_time = timer.wall_time();
              
        // Save convergence info
        if (Utilities::MPI::this_mpi_process(sot_solvers[i]->mpi_communicator) == 0) {
            std::string eps_dir = "output/epsilon_" + Utils::to_scientific_string(solver_config.regularization_param);
            fs::create_directories(eps_dir);
            std::ofstream conv_info(eps_dir + "/convergence_info.txt");
            conv_info << "Regularization parameter (λ): " << solver_config.regularization_param << "\n";
            conv_info << "Number of iterations: " << sot_solvers[i]->sot_solver->get_last_iteration_count() << "\n";
            conv_info << "Final function value: " << sot_solvers[i]->sot_solver->get_last_functional_value() << "\n";
            conv_info << "Last threshold value: " << sot_solvers[i]->sot_solver->get_last_distance_threshold() << "\n";
            conv_info << "Total execution time: " << total_time << " seconds\n";
            conv_info << "Convergence achieved: " << sot_solvers[i]->sot_solver->get_convergence_status() << "\n";
        }
    }
}

template <int dim, int spacedim>
void WassersteinBarycenters<dim, spacedim>::run()
{
    param_manager_wasserstein_barycenters.print_parameters();

    for (unsigned int i = 0; i < n_measures; ++i) {
        if (sot_solvers[i]->solver_params.use_epsilon_scaling) {
            sot_solvers[i]->epsilon_scaling_handler = std::make_unique<EpsilonScalingHandler>(
                sot_solvers[i]->mpi_communicator,
                sot_solvers[i]->solver_params.regularization_param,
                sot_solvers[i]->solver_params.epsilon_scaling_factor,
                sot_solvers[i]->solver_params.epsilon_scaling_steps
            );
        }
    }

    if (sot_solvers[0]->selected_task == "wbarycenters")
    {
        run_wasserstein_barycenters();
    }
    else
    {
        pcout << "No valid task selected" << std::endl;
    }
}

template class WassersteinBarycenters<2>;
template class WassersteinBarycenters<3>;
template class WassersteinBarycenters<2, 3>;