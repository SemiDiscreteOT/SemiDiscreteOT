# TODO: fix CMakeList

SET(TARGET "KokkosSOT")
SET(TARGET_SRC src/${TARGET}.cc)
SET(TARGET_INC src/)

SET(CMAKE_BUILD_TYPE "Debug")

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)
project(KokkosSOT)

# Init dealii
find_package(deal.II 9.5.0 REQUIRED
  HINTS  $ENV{DEAL_II_DIR}
  )
deal_ii_initialize_cached_variables()
message(STATUS "DEAL_II_DIR: $ENV{DEAL_II_DIR}")

# Init Kokkos
# Tell CMake where to find Kokkos
set(KOKKOS_DIR "/Home/flow/romor/local/gpu_kokkos-4.2.0")
set(CMAKE_PREFIX_PATH "/Home/flow/romor/local/gpu_kokkos-4.2.0/lib64/cmake/Kokkos" "/Home/flow/romor/local/sot/lib/cmake/SemiDiscreteOT")
set(Kokkos_INCLUDE_DIRS /Home/flow/romor/local/gpu_kokkos-4.2.0/include)

# Find Kokkos
find_package(Kokkos REQUIRED)
if(Kokkos_FOUND)
  set(CMAKE_CXX_COMPILER "${KOKKOS_DIR}/bin/nvcc_wrapper")
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
set(SOT_DIR "/Home/flow/romor/local/sot")
find_package(SemiDiscreteOT REQUIRED)

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

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-pedantic -Wextra -Wmissing-braces -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wsuggest-override -Wswitch -Wsynth -Wwrite-strings -Wno-placement-new -Wno-deprecated-declarations -Wno-literal-suffix -Wno-psabi -Wno-unused-local-typedefs -fopenmp-simd -pthread -O3 -funroll-loops -funroll-all-loops -Wno-unused-local-typedefs -Wno-unused-parameter --extended-lambda>)
else()
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-pedantic -Wextra -Wmissing-braces -Woverloaded-virtual -Wpointer-arith -Wsign-compare -Wsuggest-override -Wswitch -Wsynth -Wwrite-strings -Wno-placement-new -Wno-deprecated-declarations -Wno-literal-suffix -Wno-psabi -Wno-unused-local-typedefs -fopenmp-simd -pthread -O0 -ggdb --extended-lambda>)
endif()

target_link_libraries(${TARGET}
    ${KOKKOS_DIR}/lib64/libkokkoscore.a
    ${KOKKOS_DIR}/lib64/libkokkossimd.a
    ${KOKKOS_DIR}/lib64/libkokkoscontainers.a
    ${GEOGRAM_LIBS}
    ${SOT_DIR}/lib/libSemiDiscreteOT.so
)

target_include_directories(${TARGET} PUBLIC
    ${Kokkos_INCLUDE_DIRS}
    ${SOT_DIR}/include
    $ENV{GEOGRAM_DIR}/src/lib
)

message(STATUS "Kokkos include directories: ${Kokkos_INCLUDE_DIRS}")

set_target_properties(${TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
)