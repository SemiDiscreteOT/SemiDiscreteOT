##
#  CMake script
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)

# Deal.II setup first
FIND_PACKAGE(deal.II 9.5.0 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set project name
PROJECT(Tutorial3 VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SemiDiscreteOT (for standalone builds outside the main project)
IF(NOT TARGET SemiDiscreteOT)
    FIND_PACKAGE(SemiDiscreteOT REQUIRED)
ENDIF()

# Collect source files
FILE(GLOB_RECURSE TUTORIAL_SOURCES "src/*.cc")

# Create executable
ADD_EXECUTABLE(tutorial_3 ${TUTORIAL_SOURCES})

# Add include directories
TARGET_INCLUDE_DIRECTORIES(tutorial_3 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link with SemiDiscreteOT and Deal.II
IF(TARGET SemiDiscreteOT::SemiDiscreteOT)
    TARGET_LINK_LIBRARIES(tutorial_3 SemiDiscreteOT::SemiDiscreteOT)
ELSE()
    TARGET_LINK_LIBRARIES(tutorial_3 SemiDiscreteOT)
ENDIF()
DEAL_II_SETUP_TARGET(tutorial_3)

# Set output directory
SET_TARGET_PROPERTIES(tutorial_3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
)
