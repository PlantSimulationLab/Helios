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

add_library(voxelintersection STATIC "src/VoxelIntersection.cu" "src/VoxelIntersection.cpp" "tests/selfTest.cpp")

# Set CUDA properties for the target
set_target_properties(voxelintersection PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(voxelintersection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(voxelintersection PUBLIC CUDA::cudart_static)

add_executable(voxelintersection_tests "tests/TestMain.cpp")
target_link_libraries(voxelintersection_tests PRIVATE voxelintersection)
add_test(NAME voxelintersection_tests COMMAND voxelintersection_tests)