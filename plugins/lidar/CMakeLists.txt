cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

# Set CUDA properties
if( OPTIX_VERSION_LEGACY )
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
else()
     include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
endif()

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Wno-deprecated-gpu-targets")

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
else()
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(lidar STATIC "src/LiDAR.cu" "src/LiDAR.cpp" "src/fileIO.cpp" "tests/selfTest.cpp" "../../core/lib/pugixml/pugixml.cpp" "lib/s_hull_pro/s_hull_pro.cpp")

# Set CUDA properties for the target
set_target_properties(lidar PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON )

target_include_directories(lidar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(lidar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib/s_hull_pro)

target_link_libraries(lidar PUBLIC visualizer CUDA::cudart_static)
add_dependencies(lidar visualizer)

#add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip" )
#include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip/include" )

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

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )
file( COPY data DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )

add_executable(lidar_tests "tests/TestMain.cpp")
target_link_libraries(lidar_tests PRIVATE lidar)
add_test(NAME lidar_tests COMMAND lidar_tests)