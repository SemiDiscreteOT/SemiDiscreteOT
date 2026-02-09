##
#  CMake script
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)


# Deal.II setup first
FIND_PACKAGE(deal.II 9.5.0 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set project name
PROJECT(Tutorial0 VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SemiDiscreteOT (for standalone builds outside the main project)
IF(NOT TARGET SemiDiscreteOT)
    FIND_PACKAGE(SemiDiscreteOT REQUIRED)
ENDIF()

# Find VTK (optional: comes through deal.II or SemiDiscreteOT)
FIND_PACKAGE(VTK QUIET)

# Collect source files
FILE(GLOB_RECURSE TUTORIAL_SOURCES "src/*.cc")

# Create executable
ADD_EXECUTABLE(tutorial_0 ${TUTORIAL_SOURCES})

# Add include directories
TARGET_INCLUDE_DIRECTORIES(tutorial_0 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link with SemiDiscreteOT and Deal.II
IF(TARGET SemiDiscreteOT::SemiDiscreteOT)
    TARGET_LINK_LIBRARIES(tutorial_0 SemiDiscreteOT::SemiDiscreteOT)
ELSE()
    TARGET_LINK_LIBRARIES(tutorial_0 SemiDiscreteOT)
ENDIF()
DEAL_II_SETUP_TARGET(tutorial_0)

# Set output directory
SET_TARGET_PROPERTIES(tutorial_0 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
)
