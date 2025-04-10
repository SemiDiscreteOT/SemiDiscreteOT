# SemiDiscreteOT

<div align="center">
  <img src="docs/images/logo_SemiDiscreteOT.png" alt="SemiDiscreteOT Logo" width="400"/>
</div>

## Overview

SemiDiscreteOT is a high-performance C++ library for solving regularized semi-discrete optimal transport problems. It implements efficient numerical strategies for computing optimal transport plans between continuous and discrete probability measures, with a focus on large-scale applications.

## Publication

**EFFICIENT NUMERICAL STRATEGIES FOR REGULARIZED SEMI-DISCRETE OPTIMAL TRANSPORT**  
*Moaad Khamlich, Francesco Romor, and Gianluigi Rozza*

*Abstract:* Semi-discrete optimal transport (SOT), which deals with mapping a continuous probability measure to a discrete one, is a fundamental problem with wide-ranging applications. Entropic regularization is often employed to ensure smoother transport plans and enable efficient computation via convex duality. However, solving the regularized semi-discrete optimal transport (RSOT) dual problem presents a significant computational challenge, particularly when the discrete measure involves a large number of support points, as the objective function evaluation involves dense interactions. This paper presents a cohesive set of numerical strategies designed to efficiently tackle the RSOT dual problem. We focus on accelerating the evaluation of the dual objective and its gradient by combining distance-based truncation of interactions with fast spatial query structures (R-trees). We further enhance performance through an optional computational cache that exploits temporal coherence during optimization. To handle large-scale problems, we incorporate multilevel acceleration techniques based on hierarchies of both the continuous source mesh and the discrete target measure, including a discussion on potential transfer schemes between levels. Finally, we integrate a strategy for scheduling the regularization parameter. These methods, when combined, significantly reduce the computational burden associated with RSOT, making it practical for large-scale applications.

## Features

- Efficient computation of regularized semi-discrete optimal transport
- Distance-based truncation of interactions
- Fast spatial query structures using R-trees
- Computational cache for temporal coherence
- Multilevel acceleration techniques
- Adaptive regularization parameter scheduling
- Integration with Deal.II for finite element computations
- Support for both structured and unstructured meshes
- Python bindings for easy integration

## Dependencies

- Deal.II 9.5.0 or later
- Geogram
- VTK (optional, for visualization)
- CMake 3.13.0 or later
- C++17 compatible compiler

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SemiDiscreteOT.git
cd SemiDiscreteOT
```

2. Create a build directory and configure with CMake:
```bash
mkdir build && cd build
cmake ..
```

3. Build the library:
```bash
make
```

## Usage

Basic example of using SemiDiscreteOT:

```cpp
#include <SemiDiscreteOT/SemiDiscreteOT.h>

// Initialize the solver
SotSolver solver;

// Set up source and target measures
// ...

// Solve the optimal transport problem
solver.solve();
```

For more detailed examples, please refer to the `examples` directory.

## Documentation

Detailed documentation is available in the `docs` directory. You can build the documentation using:

```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Moaad Khamlich (PhD student at SISSA)
- Francesco Romor (Postdoc at Weierstrass Institute, Berlin)