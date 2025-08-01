# SemiDiscreteOT

<div align="center">
  <img src="docs/images/logo_SemiDiscreteOT.png" alt="SemiDiscreteOT Logo" width="400"/>
</div>

<div align="center">

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/SemiDiscreteOT/SemiDiscreteOT)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://semidiscreteot.github.io/SemiDiscreteOT/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.23602-b31b1b.svg)](https://arxiv.org/abs/2507.23602)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/SemiDiscreteOT/SemiDiscreteOT)

</div>

## Overview

SemiDiscreteOT is a high-performance C++ library for solving regularized semi-discrete optimal transport problems. It provides efficient numerical strategies for computing optimal transport plans between continuous and discrete probability measures, with a focus on large-scale applications.

The methods implemented in this library are detailed in the publication: [**"Efficient Numerical Strategies for Entropy-Regularized Semi-Discrete Optimal Transport"**](https://arxiv.org/abs/2507.23602) by Moaad Khamlich, Francesco Romor, and Gianluigi Rozza (2025).

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

### Option 1: Using Docker

Docker images for linux/amd64 are available on [Docker Hub](https://hub.docker.com/r/moaadkhamlich/semidiscreteot). These images are based on [Deal.II](https://www.dealii.org/) v9.5.0 and include a pre-configured [Geogram](https://github.com/BrunoLevy/geogram) installation, providing an isolated environment with a compiled version of the SemiDiscreteOT library. To pull the image, run the following command:

```bash
docker pull moaadkhamlich/semidiscreteot:latest
```

Once the image is downloaded, you can start the container and mount your working directory by executing:

```bash
docker run -ti --rm -v "${PWD}:/workspace" moaadkhamlich/semidiscreteot:latest
```

### Option 2: Building from Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SemiDiscreteOT/SemiDiscreteOT.git
    cd SemiDiscreteOT
    ```

2.  **Build the library:**
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

3.  **Run an example:**
    Check out the [`examples`](https://github.com/SemiDiscreteOT/SemiDiscreteOT/tree/master/examples) directory for detailed use cases.

## Documentation

### API Documentation

**Online Documentation:** [https://semidiscreteot.github.io/SemiDiscreteOT/](https://semidiscreteot.github.io/SemiDiscreteOT/) (Auto-generated with Doxygen)

### Build Documentation Locally

```bash
doxygen Doxyfile
xdg-open docs/doxygen/html/index.html
```

## How to Cite

If you use SemiDiscreteOT in your research, please cite the following publication:

```bibtex
@misc{khamlich2025efficientnumericalstrategiesentropyregularized,
      title={Efficient Numerical Strategies for Entropy-Regularized Semi-Discrete Optimal Transport}, 
      author={Moaad Khamlich and Francesco Romor and Gianluigi Rozza},
      year={2025},
      eprint={2507.23602},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2507.23602}, 
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0) - see the [LICENSE](LICENSE) file for details. For more information about LGPL-3.0, visit the [GNU website](https://www.gnu.org/licenses/lgpl-3.0.html).

## Authors

- [Moaad Khamlich](https://scholar.google.com/citations?user=0ONdrqkAAAAJ&hl=it) (PhD student at [SISSA](https://www.sissa.it/))
- [Francesco Romor](https://scholar.google.com/citations?user=gcTE3TgAAAAJ&hl=en) (Postdoc at [Weierstrass Institute](https://www.wias-berlin.de/), Berlin)
