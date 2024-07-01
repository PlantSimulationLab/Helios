cmake_minimum_required(VERSION 3.15)

project(helios)  

include_directories(include)
include_directories(../../core/include)

add_library( plantarchitecture STATIC "src/PlantArchitecture.cpp;src/PlantLibrary.cpp;src/Assets.cpp;src/InputOutput.cpp;src/selfTest.cpp")

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY assets DESTINATION ${CMAKE_BINARY_DIR}/plugins/plantarchitecture/ )
