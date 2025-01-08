SET(TARGET "rsot.exe")
SET(TARGET_SRC
  src/main.cc
  src/rsot.cc
  src/PowerDiagram.cc
  src/ExactSot.cc
)

SET(GEOGRAM_DIR "/scratch/mkhamlic/lib/geogram")

INCLUDE_DIRECTORIES(${GEOGRAM_DIR}/src/lib)
SET(CMAKE_MODULE_PATH ${GEOGRAM_DIR}/cmake)
SET(GEOGRAM_DIR ${GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libgeogram.so ${GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libexploragram.so)

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
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/run)
DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES( ${TARGET} ${GEOGRAM_DIR} )
