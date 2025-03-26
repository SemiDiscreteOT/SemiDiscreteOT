#include "SemiDiscreteOT/solvers/EpsilonScalingHandler.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

EpsilonScalingHandler::EpsilonScalingHandler(
    const MPI_Comm& comm,
    double initial_epsilon,
    double scaling_factor,
    unsigned int num_steps)
    : mpi_communicator(comm)
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, this_mpi_process == 0)
    , initial_epsilon(initial_epsilon)
    , scaling_factor(scaling_factor)
    , num_steps(num_steps)
{
    // Validate parameters
    if (scaling_factor <= 1.0) {
        pcout << "Warning: Epsilon scaling factor should be > 1.0. Using default value of 2.0." << std::endl;
        this->scaling_factor = 2.0;
    }
    
    if (num_steps == 0) {
        pcout << "Warning: Number of epsilon scaling steps cannot be 0. Using default value of 1." << std::endl;
        this->num_steps = 1;
    }
}

std::vector<double> EpsilonScalingHandler::generate_epsilon_sequence() const
{
    std::vector<double> epsilon_sequence;
    epsilon_sequence.reserve(num_steps);
    
    // Generate sequence from largest to smallest epsilon
    for (unsigned int i = 0; i < num_steps; ++i) {
        double scale_factor = std::pow(scaling_factor, num_steps - 1 - i);
        epsilon_sequence.push_back(initial_epsilon * scale_factor);
    }
    
    return epsilon_sequence;
}

std::vector<std::vector<double>> EpsilonScalingHandler::distribute_epsilon_values(
    const std::vector<double>& epsilon_sequence,
    unsigned int num_levels)
{
    std::vector<std::vector<double>> distribution(num_levels);
    
    if (num_levels == 0) {
        pcout << "Warning: No levels provided for epsilon distribution." << std::endl;
        return distribution;
    }
    
    // If we have more epsilon values than levels, we need to assign multiple
    // epsilon values to some levels (starting from the coarsest)
    if (epsilon_sequence.size() <= num_levels) {
        // Simple case: we have fewer or equal epsilon values than levels
        // Assign one epsilon value per level, starting from the coarsest
        for (unsigned int level = 0; level < num_levels; ++level) {
            if (level < epsilon_sequence.size()) {
                distribution[level].push_back(epsilon_sequence[level]);
            } else {
                // For levels without a dedicated epsilon, use the smallest epsilon
                distribution[level].push_back(epsilon_sequence.back());
            }
        }
    } else {
        // Complex case: we have more epsilon values than levels
        // Distribute them evenly, with more values assigned to coarser levels
        unsigned int remaining_epsilons = epsilon_sequence.size();
        unsigned int remaining_levels = num_levels;
        
        unsigned int epsilon_index = 0;
        for (unsigned int level = 0; level < num_levels; ++level) {
            // Calculate how many epsilon values to assign to this level
            unsigned int epsilons_for_level = (remaining_epsilons + remaining_levels - 1) / remaining_levels;
            
            // Add epsilon values for this level
            for (unsigned int i = 0; i < epsilons_for_level && epsilon_index < epsilon_sequence.size(); ++i) {
                distribution[level].push_back(epsilon_sequence[epsilon_index++]);
            }
            
            remaining_epsilons -= epsilons_for_level;
            remaining_levels--;
        }
    }
    
    return distribution;
}

std::vector<std::vector<double>> EpsilonScalingHandler::compute_epsilon_distribution(
    unsigned int num_levels)
{
    // Generate the sequence of epsilon values
    std::vector<double> epsilon_sequence = generate_epsilon_sequence();
    // Distribute epsilon values across levels
    epsilon_distribution = distribute_epsilon_values(epsilon_sequence, num_levels);
    return epsilon_distribution;
}

const std::vector<double>& EpsilonScalingHandler::get_epsilon_values_for_level(
    unsigned int level_index) const
{
    if (level_index >= epsilon_distribution.size()) {
        // Return empty vector for invalid level index
        static const std::vector<double> empty_vector;
        pcout << "Warning: Requested epsilon values for invalid level index: " 
              << level_index << std::endl;
        return empty_vector;
    }
    
    return epsilon_distribution[level_index];
}

void EpsilonScalingHandler::print_epsilon_distribution() const
{
    if (this_mpi_process != 0) return;
    
    if (epsilon_distribution.empty()) {
        pcout << "Epsilon distribution has not been computed yet." << std::endl;
        return;
    }
    
    pcout << "\n----------------------------------------" << std::endl;
    pcout << "Epsilon Scaling Distribution:" << std::endl;
    pcout << "----------------------------------------" << std::endl;
    pcout << "Initial epsilon: " << initial_epsilon << std::endl;
    pcout << "Scaling factor: " << scaling_factor << std::endl;
    pcout << "Number of steps: " << num_steps << std::endl;
    pcout << "Number of levels: " << epsilon_distribution.size() << std::endl;
    pcout << "----------------------------------------" << std::endl;
    auto num_levels = epsilon_distribution.size()-1;
    
    for (unsigned int level = 0; level < epsilon_distribution.size(); ++level) {
        pcout << "Level " << num_levels - level << " (";
        if (level == 0) {
            pcout << "coarsest";
        } else if (level == epsilon_distribution.size() - 1) {
            pcout << "finest";
        } else {
            pcout << "intermediate";
        }
        pcout << "): ";
        
        const auto& level_epsilons = epsilon_distribution[level];
        pcout << level_epsilons.size() << " epsilon value(s): ";
        
        for (unsigned int i = 0; i < level_epsilons.size(); ++i) {
            pcout << std::scientific << std::setprecision(4) << level_epsilons[i];
            if (i < level_epsilons.size() - 1) {
                pcout << ", ";
            }
        }
        pcout << std::endl;
    }
    pcout << "----------------------------------------" << std::endl;
    pcout << std::defaultfloat;
} 