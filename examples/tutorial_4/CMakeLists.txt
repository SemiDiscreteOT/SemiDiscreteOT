##
#  CMake script for Tutorial 4: Wasserstein Barycenters
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)

# Deal.II setup first
FIND_PACKAGE(deal.II 9.5.0 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set project name
PROJECT(Tutorial4 VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SemiDiscreteOT (for standalone builds outside the main project)
IF(NOT TARGET SemiDiscreteOT)
    FIND_PACKAGE(SemiDiscreteOT REQUIRED)
ENDIF()

# Find VTK (optional: comes through deal.II or SemiDiscreteOT)
FIND_PACKAGE(VTK QUIET)

# Create executables
ADD_EXECUTABLE(tutorial_4_gradient_descent src/main_gradient_descent.cc)
ADD_EXECUTABLE(tutorial_4_lloyd src/main_lloyd.cc)

# Add include directories
TARGET_INCLUDE_DIRECTORIES(tutorial_4_gradient_descent PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
TARGET_INCLUDE_DIRECTORIES(tutorial_4_lloyd PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link with SemiDiscreteOT and Deal.II
IF(TARGET SemiDiscreteOT::SemiDiscreteOT)
    TARGET_LINK_LIBRARIES(tutorial_4_gradient_descent SemiDiscreteOT::SemiDiscreteOT)
    TARGET_LINK_LIBRARIES(tutorial_4_lloyd SemiDiscreteOT::SemiDiscreteOT)
ELSE()
    TARGET_LINK_LIBRARIES(tutorial_4_gradient_descent SemiDiscreteOT)
    TARGET_LINK_LIBRARIES(tutorial_4_lloyd SemiDiscreteOT)
ENDIF()
DEAL_II_SETUP_TARGET(tutorial_4_gradient_descent)
DEAL_II_SETUP_TARGET(tutorial_4_lloyd)

# Set output directory
SET_TARGET_PROPERTIES(tutorial_4_gradient_descent tutorial_4_lloyd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
)
