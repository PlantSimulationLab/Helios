cmake_minimum_required(VERSION 3.15)

project(helios)  


if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

find_package(CUDA REQUIRED)

if( OPTIX_VERSION_LEGACY )
     set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
else()
     include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
endif()

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math -Wno-deprecated-gpu-targets" )

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
     set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -O0" )
else()
     set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3" )
endif()

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++${CMAKE_CXX_STANDARD}" )

SET( CUDA_PROPAGATE_HOST_FLAGS OFF )

CUDA_ADD_LIBRARY( aeriallidar STATIC "src/AerialLiDAR.cu" "src/AerialLiDAR.cpp" "src/fileIO.cpp" "src/selfTest.cpp" "../../core/src/pugixml.cpp" )

target_include_directories(aeriallidar
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/lib/glm
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries( aeriallidar
    PRIVATE
        helios
        ${CUDA_LIBRARIES}
        visualizer
)

#add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/ "plugins/visualizer" )
add_dependencies( aeriallidar visualizer )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/aeriallidar/ )

