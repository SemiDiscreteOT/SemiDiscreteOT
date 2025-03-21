#ifndef EPSILON_SCALING_HANDLER_H
#define EPSILON_SCALING_HANDLER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

using namespace dealii;

/**
 * @brief Handler for epsilon scaling in multilevel optimization.
 * 
 * This class manages the distribution of epsilon values across different levels
 * of a multilevel optimization process, either for target point clouds or source meshes.
 * It supports distributing epsilon values across levels according to user-defined strategies.
 */
class EpsilonScalingHandler {
public:
    /**
     * @brief Constructor.
     * 
     * @param comm MPI communicator
     * @param initial_epsilon Initial epsilon value
     * @param scaling_factor Factor by which epsilon is scaled between steps
     * @param num_steps Total number of epsilon scaling steps
     */
    EpsilonScalingHandler(const MPI_Comm& comm,
                         double initial_epsilon,
                         double scaling_factor,
                         unsigned int num_steps);

    /**
     * @brief Compute epsilon distribution for multilevel optimization.
     * 
     * @param num_levels Number of levels in the hierarchy
     * @param target_enabled Whether target multilevel is enabled
     * @param source_enabled Whether source multilevel is enabled
     * @return Vector of vectors containing epsilon values for each level
     */
    std::vector<std::vector<double>> compute_epsilon_distribution(
        unsigned int num_levels,
        bool target_enabled,
        bool source_enabled);

    /**
     * @brief Get epsilon values for a specific level.
     * 
     * @param level_index Index of the level (0 = coarsest)
     * @return Vector of epsilon values for the specified level
     */
    const std::vector<double>& get_epsilon_values_for_level(unsigned int level_index) const;

    /**
     * @brief Print the epsilon distribution.
     */
    void print_epsilon_distribution() const;

private:
    MPI_Comm mpi_communicator;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    double initial_epsilon;
    double scaling_factor;
    unsigned int num_steps;

    std::vector<std::vector<double>> epsilon_distribution;

    /**
     * @brief Generate the sequence of epsilon values.
     * 
     * @return Vector of epsilon values from largest to smallest
     */
    std::vector<double> generate_epsilon_sequence() const;

    /**
     * @brief Distribute epsilon values across levels.
     * 
     * @param epsilon_sequence Sequence of epsilon values
     * @param num_levels Number of levels
     * @param use_target Whether to use target multilevel (if false, use source)
     * @return Vector of vectors containing epsilon values for each level
     */
    std::vector<std::vector<double>> distribute_epsilon_values(
        const std::vector<double>& epsilon_sequence,
        unsigned int num_levels,
        bool use_target);
};

#endif // EPSILON_SCALING_HANDLER_H 