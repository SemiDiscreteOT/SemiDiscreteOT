cmake_minimum_required(VERSION 3.3.0)

# Set target name and source files
SET(TARGET "power_diagram.exe")
SET(TARGET_SRC
    power_diagram.cc
)

# Deal.II configuration
SET(DEAL_II_DIR "/scratch/mkhamlic/spack/opt/spack/linux-rocky8-x86_64_v4/gcc-12.2.0/dealii-9.5.0-nbal44n4aoskliqi4yri5icwmcgumyxt")
FIND_PACKAGE(deal.II 9.5.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n"
    "Set -DDEAL_II_DIR=/path/to/deal.II or environment variable DEAL_II_DIR"
  )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/run)

# Set CGAL directory before autopilot
SET(CGAL_DIR "/scratch/mkhamlic/spack/opt/spack/linux-rocky8-x86_64_v4/gcc-12.2.0/cgal-5.6-4luxksujzes47kdvjzl2jecqmxh2hfve/lib64/cmake/CGAL")
find_package(CGAL REQUIRED COMPONENTS Core)
include(${CGAL_USE_FILE})
DEAL_II_INVOKE_AUTOPILOT()
target_link_libraries(${TARGET} CGAL::CGAL CGAL_Core)

# Add OpenMPI after everything else
#execute_process(
    #COMMAND spack location -i openmpi@4.1.6%gcc@12.2.0
    #OUTPUT_VARIABLE OPENMPI_ROOT
    #OUTPUT_STRIP_TRAILING_WHITESPACE
#)
#link_directories(${OPENMPI_ROOT}/lib)
