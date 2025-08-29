# Tutorial 0: Basic API Usage

<div align="center">
  <img src="../../docs/images/logo_SemiDiscreteOT.png" alt="SemiDiscreteOT Logo" width="300"/>
</div>

This tutorial demonstrates the basic usage of the **SemiDiscreteOT** C++ API for solving semi-discrete optimal transport problems programmatically, as an alternative to the binary-based workflow shown in the [`run/`](../../run/) directory.

## Overview

Tutorial 0 showcases how to:
- Set up source and target measures programmatically
- Configure solver parameters through the API
- Solve optimal transport problems using C++ code
- Handle mesh loading and density field setup
- Use the multilevel optimization features

This approach provides more flexibility and control compared to the parameter file-based binary execution, making it ideal for integration into larger C++ applications or custom workflows.

## Quick Start

### Build and Run

```bash
# From the tutorial_0 directory
mkdir -p build && cd build
cmake ..
make -j4

# Run the tutorial
./tutorial_0
```

### Expected Output

```
========================================================
      SemiDiscreteOT: Tutorial 1      
========================================================

--- User: Preparing source data ---
Mesh loaded from file: source.msh
Triangulation created with X cells
Density vector computed with Y DoFs

--- Setting up SemiDiscreteOT problem ---

--- Tutorial completed successfully ---
Computed Z optimal transport potentials
```

## Code Structure

The tutorial demonstrates the complete API workflow:

### 1. Source Measure Setup

```cpp
// Load or generate source mesh
Triangulation<dim,spacedim> tria;
GridIn<dim,spacedim> grid_in;
grid_in.attach_triangulation(tria);

// Try to load from file, fallback to generated mesh
std::ifstream input_file("source.msh");
if (!input_file.is_open()) {
    GridGenerator::hyper_cube(tria, 0, 1);
    tria.refine_global(2);
} else {
    grid_in.read_msh(input_file);
}

// Setup finite element space
FE_SimplexP<dim,spacedim> fe(1);
DoFHandler<dim,spacedim> dof_handler(tria);
dof_handler.distribute_dofs(fe);

// Initialize density field
Vector<double> density_vector(dof_handler.n_dofs());
density_vector = 1.0;  // Uniform density
```

### 2. Target Measure Setup

```cpp
// Load target points from file or use defaults
std::vector<Point<spacedim>> target_points;
if (!Utils::read_vector(target_points, "target_points.txt")) {
    // Fallback to default points
    target_points = {
        Point<spacedim>(0.1, 0.1, 0.8),
        Point<spacedim>(0.8, 0.1, -0.1),
        Point<spacedim>(0.1, 0.8, -0.8)
    };
}

// Set uniform target weights
Vector<double> target_weights(target_points.size());
target_weights = 1.0;
```

### 3. Solver Configuration

```cpp
SemiDiscreteOT<dim,spacedim> ot_problem(mpi_communicator);

// Configure using lambda function
ot_problem.configure([&](SotParameterManager &params) {
    // Multilevel settings
    params.multilevel_params.source_enabled = true;
    params.multilevel_params.target_enabled = false;
    params.multilevel_params.source_min_vertices = 100;
    params.multilevel_params.source_max_vertices = 10000;
    
    // Solver parameters
    params.solver_params.epsilon = 0.01;              // Regularization
    params.solver_params.verbose_output = false;      // Disable verbose mode
    params.solver_params.tau = 1e-12;                 // Truncation tolerance
    params.solver_params.tolerance = 1e-2;            // Convergence tolerance
    params.solver_params.distance_threshold_type = "pointwise";
    params.solver_params.use_log_sum_exp_trick = false;
});
```

### 4. Problem Setup and Solution

```cpp
// Setup the optimal transport problem
ot_problem.setup_source_mesh(tria);
ot_problem.setup_source_measure(density_vector)
ot_problem.setup_target_measure(target_points, target_weights);
ot_problem.prepare_multilevel_hierarchies();

// Solve and get results
Vector<double> potentials = ot_problem.solve();
```

## Input Files

### Source Mesh (`run/source.msh`)
- 3D mesh file in Gmsh format (.msh)
- Defines the source domain geometry
- Tutorial falls back to a unit cube if file is missing

### Target Points (`run/target_points.txt`)
- ASCII file with 3D point coordinates
- One point per line: `x y z`
- Defines the discrete target measure support
- Tutorial uses default points if file is missing

## Configuration Options

### Solver Parameters

```cpp
params.solver_params.epsilon = 0.01;                    // Regularization parameter
params.solver_params.tau = 1e-12;                       // Truncation tolerance  
params.solver_params.tolerance = 1e-2;                  // Convergence tolerance
params.solver_params.max_iterations = 10000;            // Maximum iterations
params.solver_params.use_log_sum_exp_trick = false;     // Numerical stability
params.solver_params.distance_threshold_type = "pointwise"; // Distance computation
```

### Multilevel Parameters

```cpp
// Source multilevel hierarchy
params.multilevel_params.source_enabled = true;         // Enable source multilevel
params.multilevel_params.source_min_vertices = 100;     // Minimum mesh size
params.multilevel_params.source_max_vertices = 10000;   // Maximum mesh size

// Target multilevel hierarchy  
params.multilevel_params.target_enabled = false;        // Enable target multilevel
params.multilevel_params.target_min_points = 100;       // Minimum point count
params.multilevel_params.target_max_points = 50000;     // Maximum point count
```

## Extensions and Customization

### Custom Density Functions

```cpp
// Example: Gaussian density field
for (const auto& [dof_index, point] : support_points) {
    const Point<spacedim> &p = point;
    double dist_sq = p.square();
    density_vector[dof_index] = std::exp(-dist_sq / (2.0 * sigma * sigma));
}
```

### Custom Target Distributions

```cpp
// Example: Random target points
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);

std::vector<Point<spacedim>> target_points;
for (int i = 0; i < n_target_points; ++i) {
    target_points.emplace_back(dis(gen), dis(gen), dis(gen));
}
```