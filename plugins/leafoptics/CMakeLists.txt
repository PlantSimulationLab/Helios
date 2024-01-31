cmake_minimum_required(VERSION 2.0)

project(helios)  

include_directories(include)
include_directories(../../core/include)

add_library( leafoptics STATIC "src/LeafOptics.cpp" )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/leafoptics/ )

