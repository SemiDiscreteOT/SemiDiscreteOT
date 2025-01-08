#!/bin/bash

# Directory structure
mkdir -p params output

# Base parameter file
cat > params/base_params.prm << EOL
subsection Convex2Convex

  set selected_task = sot  # Options: mesh_generation, load_meshes, sot, power_diagram
  set io_coding = txt

  subsection mesh_generation
    subsection source
      set number of refinements = 4
      set grid generator function = hyper_cube
      set grid generator arguments = -1 : 1 : false
    end

    subsection target
      set number of refinements = 3
      set grid generator function = hyper_ball
      set grid generator arguments = 0, 0, 0 : 1 : true
    end
  end

  subsection rsot_solver
    set max_iterations = 10000
    set tolerance = 1e-7
    set regularization_parameter = 1e-3
    set verbose_output = true
    set solver_type = BFGS
    set quadrature_order = 3
  end

end
EOL

# Array of epsilon values to test
epsilon_values=(
    1e1
    1e0
    1e-1
    1e-2
    1e-3
    1e-4
)

# Function to run a single simulation
run_simulation() {
    local eps=$1
    echo "----------------------------------------"
    echo "Starting simulation with epsilon = $eps"
    echo "----------------------------------------"

    # Create parameter file for this epsilon
    local param_file="params/eps_${eps}.prm"
    cp params/base_params.prm "$param_file"

    # Update regularization parameter
    sed -i "s/set regularization_parameter = .*/set regularization_parameter = ${eps}/" "$param_file"

    # Run the simulation
    if ./rsot.exe "$param_file"; then
        echo "✓ Simulation completed successfully for epsilon = $eps"
    else
        echo "✗ Error in simulation for epsilon = $eps"
        return 1
    fi
}

# Run all simulations
echo "Starting batch simulations..."
echo "======================================"

failed_sims=()
for eps in "${epsilon_values[@]}"; do
    if ! run_simulation "$eps"; then
        failed_sims+=("$eps")
    fi
done

# Generate summary report
echo "Generating summary report..."
{
    echo "# Batch Simulation Results"
    echo "## Configuration"
    echo "- Date: $(date)"
    echo "- Total simulations: ${#epsilon_values[@]}"
    echo "- Failed simulations: ${#failed_sims[@]}"
    echo
    echo "## Results"
    echo "| Epsilon | Iterations | Final Value | Converged |"
    echo "|---------|------------|-------------|-----------|"

    for eps in "${epsilon_values[@]}"; do
        info_file="output/epsilon_${eps}/convergence_info.txt"
        if [ -f "$info_file" ]; then
            iterations=$(grep "Number of iterations" "$info_file" | cut -d: -f2 | tr -d ' ')
            final_value=$(grep "Final function value" "$info_file" | cut -d: -f2 | tr -d ' ')
            converged=$(grep "Convergence achieved" "$info_file" | cut -d: -f2 | tr -d ' ')
            echo "| $eps | $iterations | $final_value | $converged |"
        else
            echo "| $eps | Failed | N/A | No |"
        fi
    done

    if [ ${#failed_sims[@]} -gt 0 ]; then
        echo
        echo "## Failed Simulations"
        echo "The following epsilon values failed:"
        for eps in "${failed_sims[@]}"; do
            echo "- $eps"
        done
    fi
} > output/batch_summary.md

echo "======================================"
echo "Batch simulations completed!"
echo "Summary report saved to output/batch_summary.md"
if [ ${#failed_sims[@]} -gt 0 ]; then
    echo "Warning: ${#failed_sims[@]} simulations failed"
    exit 1
fi
