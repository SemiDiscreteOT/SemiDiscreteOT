# SemiDiscreteOT

<div align="center">
  <img src="docs/images/logo_SemiDiscreteOT.png" alt="SemiDiscreteOT Logo" width="400"/>
</div>

## Overview

SemiDiscreteOT is a high-performance C++ library for solving regularized semi-discrete optimal transport problems. It provides efficient numerical strategies for computing optimal transport plans between continuous and discrete probability measures, with a focus on large-scale applications.

The methods implemented in this library are detailed in the publication: **"Efficient Numerical Strategies for Regularized Semi-Discrete Optimal Transport"** by Moaad Khamlich, Francesco Romor, and Gianluigi Rozza.

## Features

*   **Efficient Solvers:** Fast and scalable algorithms for regularized semi-discrete optimal transport.
*   **Performance Optimizations:** Includes distance-based truncation based on R-tree spatial queries.
*   **Multilevel Acceleration:** Supports multilevel techniques for both source and target measures to handle large-scale problems.
*   **Flexible:** Integrates with Deal.II for finite element computations and supports both structured and unstructured meshes.

## Dependencies

*   Deal.II (9.5.0 or later)
*   Geogram
*   VTK 
*   CMake (3.13.0 or later)
*   A C++17 compatible compiler

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/SemiDiscreteOT.git
    cd SemiDiscreteOT
    ```

2.  **Build the library:**
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

3.  **Run an example:**
    Check out the `examples` directory for detailed use cases.

## Documentation

Detailed documentation is available in the `docs` directory. To build it, run:

```bash
cd docs
make html
```

## How to Cite

If you use SemiDiscreteOT in your research, please cite the following publication:

```bibtex
@article{khamlich2024efficient,
  title={Efficient Numerical Strategies for Regularized Semi-Discrete Optimal Transport},
  author={Khamlich, Moaad and Romor, Francesco and Rozza, Gianluigi},
  journal={arXiv preprint},
  year={2024},
  note={Available at: \url{https://github.com/yourusername/SemiDiscreteOT}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Moaad Khamlich (PhD student at SISSA)
- Francesco Romor (Postdoc at Weierstrass Institute, Berlin)