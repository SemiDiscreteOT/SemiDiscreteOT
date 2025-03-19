CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)

# Deal.II setup first
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

# Initialize Deal.II first, then set project
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(SemiDiscreteOT VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/run)

# Geogram setup
SET(GEOGRAM_DIR "/scratch/mkhamlic/lib/geogram")
INCLUDE_DIRECTORIES(${GEOGRAM_DIR}/src/lib)
SET(CMAKE_MODULE_PATH ${GEOGRAM_DIR}/cmake)
SET(GEOGRAM_LIBS
    ${GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libgeogram.so
    ${GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libexploragram.so
)


# Library source files
SET(LIB_SRC
  src/core/SemiDiscreteOT.cc
  src/core/PowerDiagramDealII.cc
  src/core/PowerDiagramGeogram.cc
  src/core/MeshHierarchy.cc
  src/core/OptimalTransportPlan.cc
  src/core/PointCloudHierarchy.cc
  src/solvers/SotSolver.cc
  src/solvers/EpsilonScalingHandler.cc
  src/solvers/ExactSot.cc
  src/solvers/SoftmaxRefinement.cc
  src/utils/ParameterManager.cc
)

# Create the library
ADD_LIBRARY(SemiDiscreteOT SHARED ${LIB_SRC})

# Add include directories for the library
TARGET_INCLUDE_DIRECTORIES(SemiDiscreteOT
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${GEOGRAM_DIR}/src/lib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Setup Deal.II for the library
DEAL_II_SETUP_TARGET(SemiDiscreteOT)

# Add Geogram libraries
TARGET_LINK_LIBRARIES(SemiDiscreteOT ${GEOGRAM_LIBS})

# Add compile definitions for both implementations
TARGET_COMPILE_DEFINITIONS(SemiDiscreteOT
    PRIVATE
        POWER_DIAGRAM_WITH_DEALII
        POWER_DIAGRAM_WITH_GEOGRAM
)

# Create the executable
ADD_EXECUTABLE(computeSot.exe src/computeSot.cc)
TARGET_LINK_LIBRARIES(computeSot.exe SemiDiscreteOT)
DEAL_II_SETUP_TARGET(computeSot.exe)

# Copy Python scripts to run/python_scripts directory
ADD_CUSTOM_COMMAND(
    TARGET computeSot.exe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying Python scripts to run/python_scripts directory"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_scripts
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/python_scripts ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_scripts
)

# Installation rules
INSTALL(TARGETS SemiDiscreteOT
    EXPORT SemiDiscreteOT-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
    COMPONENT Runtime
)


# Export targets
INSTALL(EXPORT SemiDiscreteOT-targets
    FILE SemiDiscreteOTTargets.cmake
    NAMESPACE SemiDiscreteOT::
    DESTINATION lib/cmake/SemiDiscreteOT
)

# Generate and install package config files
INCLUDE(CMakePackageConfigHelpers)
WRITE_BASIC_PACKAGE_VERSION_FILE(
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
 )

CONFIGURE_PACKAGE_CONFIG_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SemiDiscreteOTConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfig.cmake"
    INSTALL_DESTINATION lib/cmake/SemiDiscreteOT
)

INSTALL(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfigVersion.cmake"
    DESTINATION lib/cmake/SemiDiscreteOT
)

# Add examples subdirectory if building examples
OPTION(BUILD_EXAMPLES "Build the examples" ON)
IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()
