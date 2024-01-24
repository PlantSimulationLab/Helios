# Helios standard CMakeLists.txt file version 1.8
if ( WIN32 )
    cmake_minimum_required(VERSION 3.15)
else()
    cmake_minimum_required(VERSION 2.5)
endif()
project(helios)

#-------- USER INPUTS ---------#

#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file
set( BASE_DIRECTORY "../.." )
	
#define the name of the executable to be created
set( EXECUTABLE_NAME "leafoptics_selftest" )

#provide name of source file(s) (separate multiple file names with semicolon)
set( SOURCE_FILES "main.cpp" )

#specify which plug-ins to use (separate plug-in names with semicolon)
set( PLUGINS "leafoptics" )

#-------- DO NOT MODIFY ---------#
include( "${BASE_DIRECTORY}/core/CMake_project.txt" )