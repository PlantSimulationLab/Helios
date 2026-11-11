# Helios standard CMakeLists.txt file version 1.10
cmake_minimum_required(VERSION 3.15)
project(helios)

#-------- USER INPUTS ---------#

#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file.
#run_regression.py passes -DHELIOS_DIR=<tree> to build against another generation of the plug-in.
if( DEFINED HELIOS_DIR )
    set( BASE_DIRECTORY "${HELIOS_DIR}" )
else()
    set( BASE_DIRECTORY "../../../.." )
endif()

#define the name of the executable to be created
set( EXECUTABLE_NAME "soil_regression" )

#provide name of source file(s) (separate multiple file names with semicolon)
set( SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/regression.cpp" )

#specify which plug-ins to use (separate plug-in names with semicolon)
set( PLUGINS "soiltransport" )

#-------- DO NOT MODIFY ---------#
include( "${BASE_DIRECTORY}/core/CMake_project.cmake" )
