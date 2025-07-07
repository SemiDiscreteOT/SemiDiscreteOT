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

SET(SemiDiscreteOT_DIR ../../build)
FIND_PACKAGE(VTK REQUIRED)
FIND_PACKAGE(SemiDiscreteOT REQUIRED)

# Create executable from main.cc
ADD_EXECUTABLE(tutorial_4 main.cc)

# Add include directories
TARGET_INCLUDE_DIRECTORIES(tutorial_4 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link with SemiDiscreteOT and Deal.II
TARGET_LINK_LIBRARIES(tutorial_4 SemiDiscreteOT::SemiDiscreteOT)
DEAL_II_SETUP_TARGET(tutorial_4)

# Set output directory
SET_TARGET_PROPERTIES(tutorial_4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
