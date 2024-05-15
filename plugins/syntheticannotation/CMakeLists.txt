cmake_minimum_required(VERSION 2.0)

project(helios)  

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../core/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/lib/glm")

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library( syntheticannotation STATIC "${CMAKE_CURRENT_SOURCE_DIR}/src/SyntheticAnnotation.cpp" )

target_link_libraries( syntheticannotation visualizer )
add_dependencies( syntheticannotation visualizer )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/lib/s_hull_pro;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )
