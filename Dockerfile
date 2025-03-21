FROM dealii/dealii:v9.5.0-focal

# Install Geogram dependencies
USER root
RUN apt-get update && apt-get install -y \
    libglfw3-dev \
    libglu1-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set Geogram installation directory
ENV GEOGRAM_DIR=/usr/local/geogram/build
ENV LD_LIBRARY_PATH="${GEOGRAM_DIR}/lib"

# Install Geogram
WORKDIR /usr/local
RUN git clone --recursive https://github.com/BrunoLevy/geogram.git \
    && cd geogram/src/lib \
    && git clone https://github.com/BrunoLevy/exploragram.git \
    && cd ../.. \
    && mkdir -p build/Linux64-gcc-dynamic-Release \
    && cd build/Linux64-gcc-dynamic-Release \
    && cmake ../.. \
        -DCMAKE_INSTALL_PREFIX=. \
        -DGEOGRAM_WITH_GRAPHICS=OFF \
        -DGEOGRAM_WITH_LUA=OFF \
        -DGEOGRAM_WITH_HLBFGS=OFF \
        -DGEOGRAM_WITH_TETGEN=OFF \
        -DGEOGRAM_WITH_TRIANGLE=OFF \
        -DGEOGRAM_WITH_EXPLORAGRAM=ON \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && ln -s ${GEOGRAM_DIR}/lib/cmake/geogram /usr/local/lib/cmake/geogram \
    && rm -rf /usr/local/geogram/.git

WORKDIR /workspace
