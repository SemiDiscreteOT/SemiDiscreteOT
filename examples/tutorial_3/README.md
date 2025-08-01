# Tutorial 3: Blue Noise Sampling via Optimal Quantization

This tutorial demonstrates **blue noise sampling** on a Riemannian manifold (the unit sphere) using the `SemiDiscreteOT` library. It showcases optimal quantization through the Lloyd algorithm with geodesic distance costs, highlighting the framework's ability to handle non-Euclidean problems.

## Overview

Blue noise sampling generates spatially well-distributed point sets that appear uniform yet random, avoiding clumps and voids. This tutorial implements optimal quantization of a single continuous measure μ, which can be viewed as a Wasserstein barycenter problem with a single source (K=1).

### Key Features

- **Riemannian Manifold**: Sampling on the unit sphere $S^2$
- **Geodesic Distance Cost**: Using squared geodesic distance $c(x,y) = d_g(x,y)^2$
- **Non-uniform Density**: Source density defined by the 5th eigenfunction of the Laplace-Beltrami operator
- **Lloyd Algorithm**: Iterative optimization of point locations via Riemannian barycenters

## Mathematical Formulation

The optimization problem seeks to find optimal locations $Y = \{y_1, \ldots, y_N\}$ for a discrete measure $\nu(Y) = \frac{1}{N}\sum_{k=1}^N \delta_{y_k}$ that best represents the continuous source measure μ:

$$\min_{Y \subset \mathcal{Y}^N} F(Y), \quad \text{where} \quad F(Y) = \mathcal{W}_{\varepsilon,c}(\mu, \nu(Y))$$

The first-order optimality condition using the Envelope Theorem is:

$$\nabla_{y_k} F(Y^{\ast}) = \int_{\Omega} \nabla_{y_k} c(x, y_k^{\ast}) \,d\pi_k^{\ast}(Y^{\ast})(x) = \mathbf{0}$$

This motivates the Lloyd algorithm, which alternates between:
1. **Partitioning**: Solving the RSOT problem for fixed locations
2. **Updating**: Computing Riemannian barycenters for each site:

$$y_k^{(t+1)} = \arg\min_{y \in \mathcal{Y}} \int_{\Omega} c(x, y) \,d\pi_k^{(t)}(x)$$

## Implementation Details

### Source Density
The source measure is defined by the 5th eigenfunction of the Laplace-Beltrami operator on the sphere, creating regions of varying density. The eigenfunction computation uses:
- **Finite Element Discretization**: P1 elements on sphere surface
- **Mesh Refinement**: Adaptive refinement (6 levels by default)
- **Eigenvalue Solver**: SLEPc for computing eigenfunctions

### Lloyd Algorithm on Manifolds
For the squared geodesic distance cost, the Riemannian barycenter is computed via manifold-based gradient descent:

$$\nabla_{y_j} \int_{\Omega} d^2(y_j, x)\mu_j(x|\boldsymbol{\psi})\,dx = \int_{\Omega} -2\log_{y_j}(x)\mu_j(x|\boldsymbol{\psi})\,dx$$

Points evolve along the manifold: 

$$y_j^{(t+1)} = \exp_{y_j^{(t)}}(-\alpha \delta y_j)$$

## Parameter Configuration

The `parameters.prm` file controls the simulation:

```prm
subsection Tutorial 3
  subsection SphereData
    set number of refinements = 6        # Mesh refinement level
    set number of eigenfunctions = 5     # Number of eigenfunctions to compute
  end

  subsection SotParameterManager
    set selected_task = lloyd            # Run Lloyd algorithm
    
    subsection lloyd parameters
      set max iterations = 100           # Maximum Lloyd iterations
      set relative tolerance = 0         # Convergence tolerance
    end

    subsection rsot_solver
      set epsilon = 1e-2                 # Regularization parameter
      set tolerance = 1e-3               # RSOT solver tolerance
      set max_iterations = 10000         # Maximum RSOT iterations
    end
  end
end
```

## Expected Results

The algorithm produces well-distributed blue noise patterns that:
- **Respect Density**: Points concentrate in high-density regions (defined by eigenfunction)
- **Maintain Distribution**: Avoid clustering while ensuring good coverage
- **Adapt to Curvature**: Account for the sphere's intrinsic geometry
- **Converge Iteratively**: Demonstrate Lloyd algorithm convergence

### Output Files

1. **Eigenfunctions**: `output/density_field/eigenvectors_*.vtu` - Visualization of computed eigenfunctions
2. **Source Density**: `output/density_field/source_density.vtk` - Non-uniform density field
3. **Final Points**: Optimized point locations after Lloyd convergence
4. **Convergence Data**: Lloyd iteration progress and energy evolution

## References

This tutorial implements concepts from Sections 6.4 of the [accompanying paper](https://arxiv.org/abs/2507.23602).