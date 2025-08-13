# Examples CMakeLists.txt
CMAKE_MINIMUM_REQUIRED(VERSION 3.13.0)

# Deal.II setup first
FIND_PACKAGE(deal.II 9.5.0 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Add all example subdirectories
ADD_SUBDIRECTORY(tutorial_0)

# Add all example subdirectories
ADD_SUBDIRECTORY(tutorial_1)

# Add all example subdirectories
ADD_SUBDIRECTORY(tutorial_2)

# Add all example subdirectories
ADD_SUBDIRECTORY(tutorial_3)

# Add all example subdirectories
ADD_SUBDIRECTORY(tutorial_4)

# Add all example subdirectories
ADD_SUBDIRECTORY(tutorial_5)

# Common settings for all examples
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
