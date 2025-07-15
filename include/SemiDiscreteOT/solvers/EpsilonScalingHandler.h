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
 */
/**
 * @brief Handler for epsilon scaling in multilevel optimization.
 *
 * This class manages the distribution of epsilon values across different levels
 * of a multilevel optimization process, either for target point clouds or source meshes.
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
     * @return Vector of vectors containing epsilon values for each level
     */
    std::vector<std::vector<double>> compute_epsilon_distribution(
        unsigned int num_levels);

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
    MPI_Comm mpi_communicator; ///< The MPI communicator.
    const unsigned int this_mpi_process; ///< The rank of the current MPI process.
    ConditionalOStream pcout; ///< A conditional output stream for parallel printing.

    double initial_epsilon; ///< The initial epsilon value.
    double scaling_factor; ///< The scaling factor for epsilon.
    unsigned int num_steps; ///< The total number of epsilon scaling steps.

    std::vector<std::vector<double>> epsilon_distribution; ///< The computed epsilon distribution.

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
     * @return Vector of vectors containing epsilon values for each level
     */
    std::vector<std::vector<double>> distribute_epsilon_values(
        const std::vector<double>& epsilon_sequence,
        unsigned int num_levels);
};

#endif // EPSILON_SCALING_HANDLER_H 