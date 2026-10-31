# Helios standard CMakeLists.txt file version 1.10
cmake_minimum_required(VERSION 3.15)
project(helios)

#-------- USER INPUTS ---------#

#provide the path to Helios base directory, either as an absolut path or a path relative to the location of this file
set( BASE_DIRECTORY "../.." )

#define the name of the executable to be created
set( EXECUTABLE_NAME "robotics_apple_harvest" )

#provide name of source file(s) (separate multiple file names with semicolon)
set( SOURCE_FILES "main.cpp;harvest_env.cpp" )

#specify which plug-ins to use (separate plug-in names with semicolon)
set( PLUGINS "plantarchitecture;physicsengine;robotics;visualizer" )

# PyTorch brings its own OpenMP runtime, and two in one process can give wrong results. The environment has its own threads, so it does not need OpenMP.
set( ENABLE_OPENMP OFF CACHE BOOL "" FORCE )

# The plug-ins are static libraries that end up in the shared library below
set( CMAKE_POSITION_INDEPENDENT_CODE ON )

#-------- DO NOT MODIFY ---------#
include( "${BASE_DIRECTORY}/core/CMake_project.cmake" )

# The environment as a shared library, for harvest_env.py
add_library( apple_harvest_env SHARED harvest_env.cpp )
target_link_libraries( apple_harvest_env PRIVATE helios plantarchitecture physicsengine robotics visualizer )
