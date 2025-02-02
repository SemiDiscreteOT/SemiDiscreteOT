SET(TARGET "rsot.exe")
SET(TARGET_SRC
  src/main.cc
  src/rsot.cc
  src/PowerDiagramDealII.cc
  src/PowerDiagramGeogram.cc
  src/ExactSot.cc
  src/OptimalTransportPlan.cc
  src/MeshHierarchy.cc
)

SET(GEOGRAM_DIR "/Users/moaadkhamlich/scratch/libs/geogram")

# Geogram setup
INCLUDE_DIRECTORIES(${GEOGRAM_DIR}/src/lib)
SET(CMAKE_MODULE_PATH ${GEOGRAM_DIR}/cmake)

# Set library extension based on platform
IF(APPLE)
    SET(LIB_EXTENSION "dylib")
ELSE()
    SET(LIB_EXTENSION "so")
ENDIF()

SET(GEOGRAM_LIBS
    ${GEOGRAM_DIR}/build/Darwin-clang-dynamic-Release/lib/libgeogram.${LIB_EXTENSION}
    ${GEOGRAM_DIR}/build/Darwin-clang-dynamic-Release/lib/libexploragram.${LIB_EXTENSION}
)

# Deal.II setup
SET(DEAL_II_DIR "/scratch/mkhamlic/spack/opt/spack/linux-rocky8-x86_64_v4/gcc-12.2.0/dealii-9.5.0-nbal44n4aoskliqi4yri5icwmcgumyxt")
CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)
FIND_PACKAGE(deal.II 9.5.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n"
    "Set -DDEAL_II_DIR=/path/to/deal.II or environment variable DEAL_II_DIR"
  )
ENDIF()

# Initialize Deal.II
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/run)

# Setup Deal.II target
DEAL_II_INVOKE_AUTOPILOT()

# Link Geogram libraries
TARGET_LINK_LIBRARIES(${TARGET} ${GEOGRAM_LIBS})

# Add compile definitions for both implementations
TARGET_COMPILE_DEFINITIONS(${TARGET} PRIVATE
    POWER_DIAGRAM_WITH_DEALII
    POWER_DIAGRAM_WITH_GEOGRAM
)
