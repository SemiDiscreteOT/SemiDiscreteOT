##
#  CMake script
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)

# Deal.II setup first
FIND_PACKAGE(deal.II 9.5.0 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set project name
PROJECT(Tutorial1 VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(SemiDiscreteOT_DIR ../../build)
#FIND_PACKAGE(SemiDiscreteOT REQUIRED)
#FIND_PACKAGE(VTK REQUIRED)

# Collect source files
FILE(GLOB_RECURSE TUTORIAL_SOURCES "src/*.cc")

# Create executable
ADD_EXECUTABLE(tutorial_2 ${TUTORIAL_SOURCES})

# Add include directories
TARGET_INCLUDE_DIRECTORIES(tutorial_2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link with RSOT and Deal.II
#TARGET_LINK_LIBRARIES(tutorial_2 SemiDiscreteOT::SemiDiscreteOT)
TARGET_LINK_LIBRARIES(tutorial_2 SemiDiscreteOT)
DEAL_II_SETUP_TARGET(tutorial_2)

# Set output directory
SET_TARGET_PROPERTIES(tutorial_2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
)

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
