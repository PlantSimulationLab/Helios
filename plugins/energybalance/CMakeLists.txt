cmake_minimum_required(VERSION 3.15)
project(helios)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

# Set CUDA architectures before enabling CUDA language
if( OPTIX_VERSION_LEGACY )
     set(CMAKE_CUDA_ARCHITECTURES "35")
else()
     include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
endif()

# Enable CUDA as a language (modern CMake approach)
enable_language(CUDA)

# Set additional CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Wno-deprecated-gpu-targets")

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
else()
     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(energybalance STATIC "src/EnergyBalanceModel.cpp" "src/EnergyBalanceModel.cu" "tests/selfTest.cpp")

# Set CUDA properties for the target
set_target_properties(energybalance PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON )

target_link_libraries(energybalance PUBLIC CUDA::cudart_static)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(energybalance PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(energybalance_tests "tests/TestMain.cpp")
target_link_libraries(energybalance_tests PRIVATE energybalance)
add_test(NAME energybalance_tests COMMAND energybalance_tests)