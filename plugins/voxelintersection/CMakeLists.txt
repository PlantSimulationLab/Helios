cmake_minimum_required(VERSION 2.0)

project(helios)  

include_directories(include)
include_directories(../../core/include)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

find_package(CUDA REQUIRED)

if( OPTIX_VERSION_LEGACY )
     set( GPU_COMPUTE "compute_35" )
else()
     set( GPU_COMPUTE "compute_50" )
endif()

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math --gpu-architecture=${GPU_COMPUTE} -Wno-deprecated-gpu-targets" )

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
     set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -O0" )
else()
     set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3" )
endif()

if( NOT WIN32 )
     set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
endif()

SET( CUDA_PROPAGATE_HOST_FLAGS OFF )

CUDA_ADD_LIBRARY( voxelintersection STATIC "src/VoxelIntersection.cu" "src/VoxelIntersection.cpp" )

include_directories("${CUDA_INCLUDE_DIRS}")
target_link_libraries( voxelintersection ${CUDA_LIBRARIES} )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

