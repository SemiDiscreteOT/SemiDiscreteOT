# Tutorial 1: Semi-Discrete Optimal Transport with Darcy Equation PDE Solver

This tutorial demonstrates the integration of the SemiDiscreteOT library with a finite element PDE solver for the Darcy equation. The example showcases how to compute optimal transport between density fields derived from pressure solutions of PDEs on different computational domains.

## Overview

The tutorial solves the mixed formulation of Darcy's equation on two different computational domains (source and target), extracts and normalizes the pressure fields to create density distributions, and then computes the optimal transport between these distributions using the SemiDiscreteOT library.

### Mathematical Problem

We solve the mixed formulation of Darcy's equation:
```
u + ∇p = 0     (Darcy's law)
∇·u = f        (Mass conservation)
```

where:
- `u` is the velocity field (vector)
- `p` is the pressure field (scalar)
- `f` is the source/sink term

The finite element discretization uses:
- `FE_Q<dim>(1)` elements for velocity components (3 components in 3D)
- `FE_DGQ<dim>(0)` elements for pressure (1 component)

## Files Structure

```
tutorial_1/
├── src/
│   ├── Mixed.h                    # Main PDE solver class declaration
│   ├── Mixed.cc                   # PDE solver implementation
│   ├── PressureDensityField.h     # Density field extraction class
│   └── PressureDensityField.cc    # Density field implementation
├── run/
│   ├── parameters.prm             # Configuration parameters
│   └── tutorial_1                 # Compiled executable
├── CMakeLists.txt                 # Build configuration
└── README.md                      # This file
```

## Configuration Parameters

The `parameters.prm` file contains the following configurable parameters:

### Source Domain Configuration
- `number of refinements source`: Number of mesh refinement levels (default: 4)
- `grid generator function source`: Mesh type (default: "hyper_cube")
- `grid generator arguments source`: Mesh generation arguments (default: "-1 : 1 : false")
- `forcing term expression source`: Source term function (default: exponential function)
- `function constants source`: Named constants for the forcing term (default: r:1)

### Target Domain Configuration
- `number of refinements target`: Number of mesh refinement levels (default: 3)
- `grid generator function target`: Mesh type (default: "hyper_ball")
- `grid generator arguments target`: Mesh generation arguments (default: "0.0, 0.0, 0.0: 1 : true")
- `forcing term expression target`: Source term function (default: exponential function)
- `function constants target`: Named constants for the forcing term (default: r:1)

### Solver Parameters
- `max iterations outer`: Maximum outer solver iterations (default: 200000)
- `max iterations inner`: Maximum inner solver iterations (default: 100)
- `tolerance outer`: Outer solver tolerance (default: 1e-8)
- `tolerance inner`: Inner solver tolerance (default: 1e-10)

## Key Classes

### Mixed<dim>
The main PDE solver class that:
- Sets up finite element spaces and meshes for source and target domains
- Assembles the mixed formulation system matrices
- Solves the linear systems using block preconditioned FGMRES with AMG
- Outputs solution fields in VTU and HDF5 formats
- Integrates with SemiDiscreteOT for optimal transport computation

### PressureDensityField<dim>
A utility class that:
- Extracts pressure components from mixed FE solutions
- Normalizes pressure fields to create valid density distributions
- Ensures positivity of density values
- Outputs density fields for visualization

## Output Files

The simulation generates several output files in the `output/` directory:

### Mesh Files (`output/data_mesh/`)
- `source.msh`, `source.vtk`: Source domain mesh in MSH and VTK formats
- `target.msh`, `target.vtk`: Target domain mesh in MSH and VTK formats

### Solution Fields (`output/density_field/`)
- `source.vtu`, `source.pvtu`: Source domain velocity and pressure fields
- `target.vtu`, `target.pvtu`: Target domain velocity and pressure fields
- `source.h5`, `target.h5`: HDF5 format solution files
- `source_density.vtk`: Normalized source density field
- `target_density.vtk`: Normalized target density field

## Algorithm Workflow

1. **PDE Solution Phase**:
   - Generate computational meshes for source and target domains
   - Set up finite element spaces and degrees of freedom
   - Assemble system matrices for the mixed Darcy formulation
   - Solve linear systems using block-preconditioned iterative solvers
   - Output velocity and pressure fields

2. **Density Extraction Phase**:
   - Extract pressure components from mixed FE solutions
   - Normalize pressure fields to unit integral (creating probability measures)
   - Ensure positivity by shifting negative values
   - Output normalized density fields for visualization

3. **Optimal Transport Phase**:
   - Convert target density field to discrete point measures
   - Set up source measure using continuous density on FE mesh
   - Configure SemiDiscreteOT solver parameters
   - Solve the semi-discrete optimal transport problem
   - Compute optimal transport potentials

## API Integration Example

The tutorial demonstrates how to use the SemiDiscreteOT API:

```cpp
// Create optimal transport problem
SemiDiscreteOT<dim, dim> ot_problem(mpi_communicator);

// Configure solver parameters
ot_problem.configure([&](SotParameterManager &params) {
    params.solver_params.max_iterations = 10000;
    params.solver_params.epsilon = 1e-2;
    params.solver_params.tau = 1e-5;
    params.solver_params.tolerance = 1e-2;
    params.solver_params.quadrature_order = 3;
    params.solver_params.verbose_output = true;
});

// Setup measures
ot_problem.setup_source_measure(triangulation, dof_handler, density);
ot_problem.setup_target_measure(target_points, target_weights);

// Solve
Vector<double> potentials = ot_problem.solve();
```