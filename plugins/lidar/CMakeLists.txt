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

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__" )

SET( CUDA_PROPAGATE_HOST_FLAGS OFF )

CUDA_ADD_LIBRARY( lidar STATIC "src/LiDAR.cu" "src/LiDAR.cpp" "src/fileIO.cpp" "src/selfTest.cpp" "../../core/src/pugixml.cpp" "lib/s_hull_pro/s_hull_pro.cpp" )

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("lib/s_hull_pro")
target_link_libraries( lidar ${CUDA_LIBRARIES} )

#add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/ "plugins/visualizer" )
target_link_libraries( lidar visualizer )
add_dependencies( lidar visualizer )

add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip" )
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip/include" )

#set( WITH_LASZIP FALSE )
#set( WITH_UTILITIES FALSE )
#set( WITH_TESTS FALSE )
#set( WITH_GEOTIFF FALSE )
#if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
#     set( LASZIP_LIBRARY "${CMAKE_BINARY_DIR}/plugins/lidar/lib/laszip/bin/Debug/liblaszip.so" )
#else()
#     set( LASZIP_LIBRARY "${CMAKE_BINARY_DIR}/plugins/lidar/lib/laszip/bin/Release/liblaszip.so" )
#endif()
#set( LASZIP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip/include/laszip" )
#add_subdirectory( "lib/libLAS" )
#include_directories( "lib/libLAS/include/liblas" )
#if( WIN32 )
#     set( LIBLAS_LIB_NAME "liblas" )
#else()
#     set( LIBLAS_LIB_NAME "las" )
#endif()
#target_link_libraries( lidar "${LIBLAS_LIB_NAME}" )
#add_dependencies( "${LIBLAS_LIB_NAME}" laszip )
#add_dependencies( lidar "${LIBLAS_LIB_NAME}" )


set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/lib/s_hull_pro;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )
file( COPY data DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )

