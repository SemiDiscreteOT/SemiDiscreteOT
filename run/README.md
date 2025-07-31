# SemiDiscreteOT - Runtime Directory

<div align="center">
  <img src="../docs/images/logo_SemiDiscreteOT.png" alt="SemiDiscreteOT Logo" width="400"/>
</div>

This directory contains the main executable and runtime files for the SemiDiscreteOT library.

## Overview

The **SemiDiscreteOT** library provides efficient computation of semi-discrete optimal transport maps using numerical optimization techniques. The library supports both standard optimization and multilevel approaches for improved performance and scalability.

## Contents

- **`sot`** - Main executable for semi-discrete optimal transport computations
- **`parameters.prm`** - Configuration file with all solver parameters and settings
- **`python_scripts/`** - Ready-to-use Python clustering scripts for target multilevel preparation

## Quick Start

### 1. Basic Usage

Run the executable with default parameters:
```bash
./sot
```

Or specify a custom parameter file:
```bash
./sot custom_parameters.prm
```

### 2. Parallel Execution

The library supports both multithreading and multiprocessing:

**MPI parallel execution:**
```bash
mpirun -np 4 ./sot parameters.prm
```

**Combined MPI + OpenMP:**
```bash
export OMP_NUM_THREADS=2
mpirun -np 4 ./sot parameters.prm
```

### 3. Workflow Overview

The SemiDiscreteOT workflow typically follows these steps:

#### Step 1: Mesh Generation
Generate source and target meshes based on your problem configuration:

```bash
# Edit parameters.prm to set:
# set selected_task = mesh_generation
./sot
```

This creates meshes in `output/data_mesh/` directory.

#### Step 2: Standard SOT Computation
Solve the semi-discrete optimal transport problem:

```bash
# Edit parameters.prm to set:
# set selected_task = sot
./sot
```

Results are saved in `output/epsilon_<value>/` directory.

#### Step 3: Multilevel SOT (Advanced)
For large-scale problems, use the multilevel approach:

**Prepare multilevel hierarchies:**
```bash
# Prepare both hierarchies automatically (recommended):
# set selected_task = prepare_multilevel

# Or prepare individually:
# set selected_task = prepare_source_multilevel
# set selected_task = prepare_target_multilevel
./sot
```

**Run multilevel optimization:**
```bash
# set selected_task = multilevel
./sot
```

## Available Tasks

Configure the `selected_task` parameter in `parameters.prm`:

| Task | Description |
|------|-------------|
| `mesh_generation` | Generate source and target computational meshes |
| `load_meshes` | Load existing meshes from files |
| `sot` | Standard semi-discrete optimal transport computation |
| `multilevel` | Multilevel optimization for large-scale problems |
| `prepare_multilevel` | Prepare both source and target hierarchies (based on enabled parameters) |
| `prepare_source_multilevel` | Generate mesh hierarchy for source measure |
| `prepare_target_multilevel` | Generate point cloud hierarchy for target measure |
| `exact_sot` | Exact computation using analytical methods (3D only) |
| `power_diagram` | Compute and visualize power diagrams |
| `map` | Generate transport maps and visualizations |

## Configuration Guide

### Basic Parameters

Edit `parameters.prm` to configure your computation:

```ini
subsection SotParameterManager
  set selected_task = sot           # Choose your task
  set io_coding = txt               # File format: txt/binary
  
  subsection mesh_generation
    subsection source
      set number of refinements = 6
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
    set epsilon = 1e-2              # Regularization parameter
    set tau = 1e-1                  # Truncation error tolerance for integral radius bound
    set tolerance = 1e-2
  end
end
```

### Multilevel Configuration

For large-scale problems, enable multilevel optimization using the `multilevel` task:

```ini
set selected_task = multilevel

subsection multilevel_parameters
  # Source mesh hierarchy
  set source_enabled = true        # Enable source multilevel approach
  set source_min_vertices = 100
  set source_max_vertices = 10000
  
  # Target point cloud hierarchy  
  set target_enabled = true        # Enable target multilevel approach
  set target_min_points = 100
  set target_max_points = 50000
  
  # Advanced options
  set use_softmax_potential_transfer = true
  set use_python_clustering = true
  set python_script_name = multilevel_clustering_faiss_gpu.py  # Available scripts
end
```

**Available Python clustering scripts:**
- `multilevel_clustering_faiss_cpu.py` - CPU-based FAISS clustering (good for moderate datasets)
- `multilevel_clustering_faiss_gpu.py` - GPU-accelerated FAISS clustering (fastest for large datasets)  
- `multilevel_clustering_scipy.py` - SciPy-based clustering (fallback option)

### Custom Density Fields

You can use custom density fields from VTK files:

```ini
subsection source
  set use custom density = true
  set density file path = output/density_field/source_density.vtu
  set density field name = p
  set density file format = vtk
end

subsection target
  set use custom density = true  
  set density file path = output/density_field/target_density.vtu
  set density field name = p
  set density file format = vtk
end
```

## Grid Generator Functions

Available mesh generation functions:

### Source/Target Domains
- **`hyper_cube`** - Arguments: `xmin : xmax : colorize`
- **`hyper_ball`** - Arguments: `center_x, center_y, center_z : radius : colorize`
- **`hyper_sphere`** - Arguments: `center_x, center_y, center_z : radius`
- **`subdivided_hyper_cube`** - Arguments: `repetitions_x, repetitions_y, repetitions_z : xmin : xmax : colorize`

### Example Configurations
```ini
# Unit cube [-1,1]³
set grid generator function = hyper_cube
set grid generator arguments = -1 : 1 : false

# Unit ball centered at origin
set grid generator function = hyper_ball  
set grid generator arguments = 0, 0, 0 : 1 : true

# Subdivided cube with 4×4×4 cells
set grid generator function = subdivided_hyper_cube
set grid generator arguments = 4, 4, 4 : -1 : 1 : false
```

## Output Structure

The library generates organized output in the following structure:

```
output/
├── data_mesh/              # Generated meshes
│   ├── source.msh
│   ├── source.vtk
│   ├── target.msh
│   └── target.vtk
├── data_points/            # Target point coordinates
├── data_density/           # Density field data
├── epsilon_1.00e-02/       # Results for ε=0.01
│   ├── potentials          # Computed potentials
│   ├── convergence_info.txt
│   └── source_multilevel/  # Multilevel results
└── data_multilevel/        # Hierarchy data
    ├── source_multilevel/
    └── target_multilevel/
```

## Performance Tips

### Standard Computation
- Start with moderate mesh refinement (4-6 levels)
- Use `epsilon = 1e-2` for initial tests
- Enable `verbose_output = true` for monitoring

### Multilevel Optimization
- Recommended for problems with >10K points
- Set appropriate hierarchy bounds:
  - Source: `min_vertices = 100`, `max_vertices = 10000`
  - Target: `min_points = 100`, `max_points = 50000`
- Enable epsilon scaling for better convergence

### Solver Configuration
```ini
subsection rsot_solver
  set max_iterations = 10000
  set epsilon = 1e-2                  # Regularization parameter
  set tau = 1e-1                      # Truncation error tolerance for integral radius bound
  set tolerance = 1e-2                # Convergence tolerance
  set use_epsilon_scaling = true      # Gradual regularization reduction
  set epsilon_scaling_factor = 10     # Reduction factor
  set epsilon_scaling_steps = 3       # Number of scaling steps
  set use_log_sum_exp_trick = false   # Enable for numerical stability with small epsilon
  set number_of_threads = 0           # Use all available cores
end
```

**Key Parameters:**
- **`epsilon`** - Regularization parameter for the entropic optimal transport problem
- **`tau`** - Truncation error tolerance that controls the radius bound for distance computations
- **`tolerance`** - Convergence tolerance for the optimization algorithm
- **`distance_threshold_type`** - Type of radius bound computation (see Distance Threshold Types below)
- **`use_epsilon_scaling`** - Enables gradual reduction of epsilon for better convergence
- **`use_log_sum_exp_trick`** - Essential for numerical stability with small regularization parameters

### Numerical Stability for Small Epsilon Values

For high-precision computations with small regularization parameters (relative to the average transportation cost):

```ini
subsection rsot_solver
  set epsilon = 1e-5                  # Small regularization parameter
  set use_log_sum_exp_trick = true    # REQUIRED for small epsilon values
  set use_epsilon_scaling = true      # Recommended for small epsilon
  set epsilon_scaling_factor = 10
  set epsilon_scaling_steps = 4       # More steps for gradual approach
end
```

**Important:** When using epsilon values that are small relative to the average transportation cost (typically around 1e-3 of the average cost), always enable `use_log_sum_exp_trick = true` to prevent numerical instability in the softmax computations.

## Distance Threshold Types

The library supports three different methods for computing distance thresholds, controlled by the `distance_threshold_type` parameter. For theoretical details, see Section 4.1 of our paper: [Efficient Numerical Strategies for Entropy-Regularized Semi-Discrete Optimal Transport](https://arxiv.org/abs/TEMP-LINK) (ArXiv link coming soon).

### Available Types:

- **`pointwise`** (Default)
- **`integral`** 
- **`geometric`**

### Configuration:

```ini
subsection rsot_solver
  set distance_threshold_type = integral  # Options: pointwise/integral/geometric
  set tau = 1e-1                         # Truncation error tolerance parameter
end
```

---

For detailed API documentation and advanced usage, refer to the main project documentation.