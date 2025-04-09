CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)

SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_C_COMPILER gcc)

PROJECT(SemiDiscreteOT VERSION 1.0.0 LANGUAGES CXX C)



# Check Deal.II
MESSAGE("Deal.II installation path: $ENV{DEAL_II_DIR}")

# Check Geogram
MESSAGE("Geogram installation path: $ENV{GEOGRAM_DIR}")

# Set default build type to Release if not specified
IF(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

# Add compiler flags for Release build
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

SET(DEAL_II_DIR "/scratch/mkhamlic/spack/opt/spack/linux-rocky8-x86_64_v4/gcc-12.2.0/dealii-9.5.0-nbal44n4aoskliqi4yri5icwmcgumyxt")

# Deal.II setup first
FIND_PACKAGE(deal.II 9.5.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ ${DEAL_II_DIR}
)

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n"
    "Set -DDEAL_II_DIR=/path/to/deal.II or environment variable DEAL_II_DIR"
  )
ENDIF()

FIND_PACKAGE(VTK QUIET)

# Add custom release target
ADD_CUSTOM_TARGET(release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Switch CMAKE_BUILD_TYPE to Release and rebuild"
)

# Initialize Deal.II first, then set project
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/run)

# Geogram setup
INCLUDE_DIRECTORIES($ENV{GEOGRAM_DIR}/src/lib)
SET(CMAKE_MODULE_PATH $ENV{GEOGRAM_DIR}/cmake)
SET(GEOGRAM_LIBS
    $ENV{GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libgeogram.so
    $ENV{GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libexploragram.so
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
  src/utils/VtkHandler.cc
)

# Create the library
ADD_LIBRARY(SemiDiscreteOT SHARED ${LIB_SRC})

# Add include directories for the library
TARGET_INCLUDE_DIRECTORIES(SemiDiscreteOT
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $ENV{GEOGRAM_DIR}/src/lib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Setup Deal.II for the library
DEAL_II_SETUP_TARGET(SemiDiscreteOT)

# Add Geogram libraries
TARGET_LINK_LIBRARIES(SemiDiscreteOT ${GEOGRAM_LIBS} ${VTK_LIBRARIES})

# Add compile definitions for both implementations
TARGET_COMPILE_DEFINITIONS(SemiDiscreteOT
    PRIVATE
        POWER_DIAGRAM_WITH_DEALII
        POWER_DIAGRAM_WITH_GEOGRAM
)

# Create the executable
ADD_EXECUTABLE(sot src/ComputeSot.cc)
TARGET_LINK_LIBRARIES(sot SemiDiscreteOT)
DEAL_II_SETUP_TARGET(sot)

# Copy Python scripts to run/python_scripts directory
ADD_CUSTOM_COMMAND(
    TARGET sot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying Python scripts to run/python_scripts directory"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_scripts
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/python_scripts ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_scripts
)

# Installation rules
INSTALL(TARGETS SemiDiscreteOT
    EXPORT SemiDiscreteOTTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install header files
INSTALL(DIRECTORY include/ DESTINATION include)

# Export targets
EXPORT(EXPORT SemiDiscreteOTTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTTargets.cmake"
    NAMESPACE SemiDiscreteOT::
)

# Install the export
INSTALL(EXPORT SemiDiscreteOTTargets
    FILE SemiDiscreteOTTargets.cmake
    NAMESPACE SemiDiscreteOT::
    DESTINATION lib/cmake/SemiDiscreteOT
)

# Generate the config file
INCLUDE(CMakePackageConfigHelpers)
CONFIGURE_PACKAGE_CONFIG_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SemiDiscreteOTConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfig.cmake"
    INSTALL_DESTINATION lib/cmake/SemiDiscreteOT
)

WRITE_BASIC_PACKAGE_VERSION_FILE(
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the config files
INSTALL(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SemiDiscreteOTConfigVersion.cmake"
    DESTINATION lib/cmake/SemiDiscreteOT
)

# Add examples subdirectory if building examples
OPTION(BUILD_EXAMPLES "Build the examples" OFF)
IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()


