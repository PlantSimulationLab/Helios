cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# Enable CUDA as a language (modern CMake approach)
enable_language(CUDA)

# Set CUDA properties
if( OPTIX_VERSION_LEGACY )
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
else()
     include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Wno-deprecated-gpu-targets")

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
else()
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(aeriallidar STATIC "src/AerialLiDAR.cu" "src/AerialLiDAR.cpp" "src/fileIO.cpp" "src/selfTest.cpp" "../../core/lib/pugixml/pugixml.cpp")

# Set CUDA properties for the target
set_target_properties(aeriallidar PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(aeriallidar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Find CUDA toolkit for linking
find_package(CUDAToolkit REQUIRED)

target_link_libraries(aeriallidar PUBLIC visualizer CUDA::cudart_static)
add_dependencies(aeriallidar visualizer)

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/aeriallidar/ )