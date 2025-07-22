cmake_minimum_required(VERSION 3.15)

project(helios)  


if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library( syntheticannotation STATIC "src/SyntheticAnnotation.cpp" "src/selfTest.cpp" )

target_include_directories(syntheticannotation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_dependencies( syntheticannotation visualizer )
target_link_libraries( syntheticannotation PUBLIC visualizer )