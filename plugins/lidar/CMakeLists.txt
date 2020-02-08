cmake_minimum_required(VERSION 2.0)

project(helios)  

include_directories(include)
include_directories(../../core/include)
include_directories(../visualizer/include)
include_directories(../visualizer/lib/glm)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

find_package(CUDA REQUIRED)

LIST( APPEND CUDA_NVCC_FLAGS "-g --use_fast_math -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50" )

#if( ${UNIX} AND NOT ${APPLE} )
    LIST( APPEND CUDA_NVCC_FLAGS "-std=c++11" )
    #endif()

LIST( APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__" )

SET( CUDA_PROPAGATE_HOST_FLAGS OFF )

CUDA_ADD_LIBRARY( lidar STATIC "src/LiDAR.cu" "src/LiDAR.cpp" "src/fileIO.cpp" "../../core/src/pugixml.cpp" "lib/s_hull_pro/s_hull_pro.cpp" )

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("lib/s_hull_pro")
target_link_libraries( lidar ${CUDA_LIBRARIES} )

#add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/ "plugins/visualizer" )
target_link_libraries( lidar visualizer )
add_dependencies( lidar visualizer )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/lib/s_hull_pro;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )

