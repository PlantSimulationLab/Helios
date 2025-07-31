cmake_minimum_required(VERSION 2.0)
project(helios)

include_directories(include)
include_directories(../../core/include)
include_directories(/usr/include/eigen3)

add_library( irrigation STATIC "src/IrrigationModel.cpp" )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )