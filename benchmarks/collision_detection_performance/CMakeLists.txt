# Helios standard CMakeLists.txt file version 1.10
cmake_minimum_required(VERSION 3.15)
project(helios)

set( BASE_DIRECTORY "../.." )
set( EXECUTABLE_NAME "collision_detection_performance" )
set( SOURCE_FILES "main.cpp" )
set( PLUGINS "collisiondetection" )

include( "${BASE_DIRECTORY}/core/CMake_project.cmake" )