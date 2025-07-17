# Helios standard CMakeLists.txt file version 1.9
cmake_minimum_required(VERSION 3.15)
project(helios)

set( BASE_DIRECTORY "../.." )
set( EXECUTABLE_NAME "radiation_homogeneous_canopy" )
set( SOURCE_FILES "main.cpp" )
set( PLUGINS "radiation" )

include( "${BASE_DIRECTORY}/core/CMake_project.cmake" )

