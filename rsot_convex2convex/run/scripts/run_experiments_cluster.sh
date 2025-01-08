#!/bin/bash

# Create a parameter file containing epsilon values
cat > epsilon_params.csv << EOL
epsilon
1e1
1e0
1e-1
1e-2
1e-3
1e-4
EOL

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

# Skip the header line and read epsilon values
tail -n +2 "epsilon_params.csv" | while IFS=',' read -r epsilon
do
    # Create a temporary job script with expanded variables
    temp_script=$(mktemp)
    cat << 'EOF' | sed "s/EPSILON/$epsilon/g" > "$temp_script"
#!/bin/bash
#SBATCH --job-name=RSOT_eps_EPSILON
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=your.email@domain.com
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=regular1,regular2,long1,long2
#SBATCH --output=output/rsot_eps_EPSILON_%j.out
#SBATCH --error=output/rsot_eps_EPSILON_%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $SLURM_SUBMIT_DIR

# Create parameter file for this epsilon
param_file="params/eps_EPSILON.prm"
cp params/base_params.prm "$param_file"

# Update regularization parameter
sed -i "s/set regularization_parameter = .*/set regularization_parameter = EPSILON/" "$param_file"

# Run the simulation
./rsot.exe "$param_file"

# Check if simulation was successful
if [ $? -eq 0 ]; then
    echo "✓ Simulation completed successfully for epsilon = EPSILON"
else
    echo "✗ Error in simulation for epsilon = EPSILON"
    exit 1
fi

# Generate entry for summary report
info_file="output/epsilon_EPSILON/convergence_info.txt"
if [ -f "$info_file" ]; then
    iterations=$(grep "Number of iterations" "$info_file" | cut -d: -f2 | tr -d ' ')
    final_value=$(grep "Final function value" "$info_file" | cut -d: -f2 | tr -d ' ')
    converged=$(grep "Convergence achieved" "$info_file" | cut -d: -f2 | tr -d ' ')
    echo "EPSILON,$iterations,$final_value,$converged" >> output/results_summary.csv
else
    echo "EPSILON,Failed,N/A,No" >> output/results_summary.csv
fi
EOF

    # Submit the temporary job script
    sbatch "$temp_script"

    # Remove the temporary script
    rm "$temp_script"

    echo "Submitted job for epsilon=$epsilon"
done

# Create a script to generate the final summary report
cat > generate_summary.sh << 'EOL'
#!/bin/bash

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
sleep 10  # Give time for jobs to be registered
squeue -u $USER | grep "RSOT_eps" > /dev/null
while [ $? -eq 0 ]; do
    sleep 60
    squeue -u $USER | grep "RSOT_eps" > /dev/null
done

# Generate summary report
{
    echo "# Batch Simulation Results"
    echo "## Configuration"
    echo "- Date: $(date)"
    echo "- Total simulations: $(wc -l < output/results_summary.csv)"
    echo
    echo "## Results"
    echo "| Epsilon | Iterations | Final Value | Converged |"
    echo "|---------|------------|-------------|-----------|"
    
    while IFS=',' read -r eps iter val conv; do
        echo "| $eps | $iter | $val | $conv |"
    done < output/results_summary.csv
} > output/batch_summary.md

echo "Summary report generated at output/batch_summary.md"
EOL

chmod +x generate_summary.sh

echo "All jobs submitted. Run ./generate_summary.sh after all jobs complete to generate the final report."
