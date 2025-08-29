# TODO: fix CMakeList

SET(TARGET "KokkosSOT")
SET(TARGET_SRC src/${TARGET}.cc)
SET(TARGET_INC src/)

SET(CMAKE_BUILD_TYPE "Release")

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)
project(KokkosSOT)

if(NOT DEFINED ENV{KOKKOS_DIR})
  message(FATAL_ERROR "Environment variable KOKKOS_DIR is not set. Please set it to the Kokkos installation directory.")
endif()

if(NOT DEFINED ENV{DEAL_II_DIR})
  message(FATAL_ERROR "Environment variable DEAL_II_DIR is not set. Please set it to the deal.II installation directory.")
endif()

if(NOT DEFINED ENV{GEOGRAM_DIR})
  message(FATAL_ERROR "Environment variable GEOGRAM_DIR is not set. Please set it to the Geogram installation directory.")
endif()

if(NOT DEFINED ENV{SOT_DIR})
  message(FATAL_ERROR "Environment variable SOT_DIR is not set. Please set it to the SemiDiscreteOT installation directory.")
endif()
# Init dealii
find_package(deal.II 9.5.0 REQUIRED
  HINTS  $ENV{DEAL_II_DIR}
  )

# Set environment variables for OpenMP
if(NOT DEFINED ENV{OMP_PROC_BIND})
  message(WARNING "OMP_PROC_BIND environment variable not set. Setting it to 'spread' for best performance.")
  set(ENV{OMP_PROC_BIND} "spread")
endif()

if(NOT DEFINED ENV{OMP_PLACES})
  message(WARNING "OMP_PLACES environment variable not set. Setting it to 'threads' for best performance.")
  set(ENV{OMP_PLACES} "threads")
endif()

deal_ii_initialize_cached_variables()
message(STATUS "DEAL_II_DIR: $ENV{DEAL_II_DIR}")

# Init Kokkos
# Tell CMake where to find Kokkos

# Find Kokkos
find_package(Kokkos REQUIRED HINTS  $ENV{KOKKOS_DIR})
set(KOKKOS_INCLUDE_DIRS $ENV{KOKKOS_DIR}/include)
message(STATUS "KOKKOS_ENABLE_CUDA_UVM: ${KOKKOS_ENABLE_CUDA_UVM}")

if(Kokkos_FOUND)
  set(CMAKE_CXX_COMPILER "$ENV{KOKKOS_DIR}/bin/nvcc_wrapper")
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
  message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
  message(STATUS "CMAKE_CXX_FLAGS DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
  message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
  message(STATUS "CMAKE_LINKER_FLAGS: ${CMAKE_LINKER_FLAGS}")
  message(STATUS "CMAKE_LINKER_FLAGS_DEBUG: ${CMAKE_LINKER_FLAGS_DEBUG}")
  message(STATUS "CMAKE_LINKER_FLAGS_RELEASE: ${CMAKE_LINKER_FLAGS_RELEASE}")
  message(STATUS "CMAKE_DEFINITIONS: ${CMAKE_DEFINITIONS}")
  message(STATUS "CMAKE_DEFINITIONS_DEBUG: ${CMAKE_DEFINITIONS_DEBUG}")
  message(STATUS "CMAKE_DEFINITIONS_RELEASE: ${CMAKE_DEFINITIONS_RELEASE}")
endif()

# Setup Geogram
set(GEOGRAM_LIBS
    $ENV{GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libgeogram.so
    $ENV{GEOGRAM_DIR}/build/Linux64-gcc-dynamic-Release/lib/libexploragram.so
)

# Find SemiDiscreteOT
find_package(SemiDiscreteOT REQUIRED HINTS  $ENV{SOT_DIR})

add_executable(${TARGET} ${TARGET_SRC})
include_directories(${TARGET_INC})

# The variables introduced with setup need to be fixed
deal_ii_setup_target(${TARGET})

message(STATUS "DEAL_II_CXX_COMPILER: ${DEAL_II_CXX_COMPILER}")
message(STATUS "DEAL_II_CXX_FLAGS: ${DEAL_II_CXX_FLAGS}")
message(STATUS "DEAL_II_CXX_FLAGS_DEBUG: ${DEAL_II_CXX_FLAGS_DEBUG}")
message(STATUS "DEAL_II_CXX_FLAGS_RELEASE: ${DEAL_II_CXX_FLAGS_RELEASE}")
message(STATUS "DEAL_II_LINKER_FLAGS: ${DEAL_II_LINKER_FLAGS}")
message(STATUS "DEAL_II_LINKER_FLAGS_DEBUG: ${DEAL_II_LINKER_FLAGS_DEBUG}")
message(STATUS "DEAL_II_LINKER_FLAGS_RELEASE: ${DEAL_II_LINKER_FLAGS_RELEASE}")
message(STATUS "DEAL_II_DEFINITIONS: ${DEAL_II_DEFINITIONS}")
message(STATUS "DEAL_II_DEFINITIONS_DEBUG: ${DEAL_II_DEFINITIONS_DEBUG}")
message(STATUS "DEAL_II_DEFINITIONS_RELEASE: ${DEAL_II_DEFINITIONS_RELEASE}")
message(STATUS "DEAL_II_WARNING_FLAGS: ${DEAL_II_WARNING_FLAGS}")

set(DEAL_II_CXX_COMPILER ${CMAKE_CXX_COMPILER})
# set(DEAL_II_CXX_FLAGS ${CMAKE_CXX_FLAGS})
# set(DEAL_II_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
# set(DEAL_II_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
# set(DEAL_II_LINKER_FLAGS ${CMAKE_LINKER_FLAGS})
# set(DEAL_II_LINKER_FLAGS_DEBUG ${CMAKE_LINKER_FLAGS_DEBUG})
# set(DEAL_II_LINKER_FLAGS_RELEASE ${CMAKE_LINKER_FLAGS_RELEASE})
# set(DEAL_II_DEFINITIONS ${CMAKE_DEFINITIONS})
# set(DEAL_II_DEFINITIONS_DEBUG ${CMAKE_DEFINITIONS_DEBUG})
# set(DEAL_II_DEFINITIONS_RELEASE ${CMAKE_DEFINITIONS_RELEASE})

# remove dealii flags and remove -Wall -Wa--compress-debug-sections
set_target_properties(${TARGET} PROPERTIES COMPILE_OPTIONS "")

set(MEMORY_SPACE_CUDA ON)
if(MEMORY_SPACE_CUDA)
  target_compile_definitions(${TARGET} PRIVATE MEMORY_SPACE_CUDA=1)
  message(STATUS "Running on Kokkos::CudaSpace")
else()
  target_compile_definitions(${TARGET} PRIVATE MEMORY_SPACE_CUDA=0)
  message(STATUS "Running on Kokkos::OpenMP")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-pedantic -Wextra -Wmissing-braces -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wsuggest-override -Wswitch -Wsynth -Wwrite-strings -Wno-placement-new -Wno-deprecated-declarations -Wno-literal-suffix -Wno-psabi -Wno-unused-local-typedefs -fopenmp -fopenmp-simd -pthread -O3 -funroll-loops -funroll-all-loops -Wno-unused-local-typedefs -Wno-unused-parameter --extended-lambda>)
else()
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-pedantic -Wextra -Wmissing-braces -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wsuggest-override -Wswitch -Wsynth -Wwrite-strings -Wno-placement-new -Wno-deprecated-declarations -Wno-literal-suffix -Wno-psabi -Wno-unused-local-typedefs -fopenmp -fopenmp-simd -pthread -O0 -ggdb --extended-lambda>)
endif()

target_link_libraries(${TARGET}
    $ENV{KOKKOS_DIR}/lib64/libkokkoscore.a
    $ENV{KOKKOS_DIR}/lib64/libkokkossimd.a
    $ENV{KOKKOS_DIR}/lib64/libkokkoscontainers.a
    ${GEOGRAM_LIBS}
    $ENV{SOT_DIR}/lib/libSemiDiscreteOT.so
)

target_include_directories(${TARGET} PUBLIC
    ${KOKKOS_INCLUDE_DIRS}
    $ENV{SOT_DIR}/include
    $ENV{GEOGRAM_DIR}/src/lib
)

set_target_properties(${TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
)