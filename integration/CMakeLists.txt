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
PROJECT("integration")

# First executable
SET(TARGET1 "integrate.exe")
ADD_EXECUTABLE(${TARGET1} integrate.cc)
DEAL_II_SETUP_TARGET(${TARGET1})

# Second executable
SET(TARGET2 "integrate_marked.exe")
ADD_EXECUTABLE(${TARGET2} integrate_marked.cc)
DEAL_II_SETUP_TARGET(${TARGET2})