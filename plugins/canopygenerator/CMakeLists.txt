cmake_minimum_required(VERSION 2.0)

project(helios)  

include_directories(include)
include_directories(../../core/include)

add_library( canopygenerator STATIC "src/CanopyGenerator.cpp;src/grapevine.cpp;src/whitespruce.cpp;src/tomato.cpp" )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY textures DESTINATION ${CMAKE_BINARY_DIR}/plugins/canopygenerator/ )
