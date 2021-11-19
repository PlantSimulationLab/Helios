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

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math" )

if( CMAKE_BUILD_TYPE STREQUAL Debug )
     set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g" )
endif()

if( NOT WIN32 )
     set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
endif()

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__" )

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
file( COPY data DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )

