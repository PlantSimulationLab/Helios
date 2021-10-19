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

LIST( APPEND CUDA_NVCC_FLAGS "-g --use_fast_math" )

LIST( APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__" )

SET( CUDA_PROPAGATE_HOST_FLAGS OFF )

CUDA_ADD_LIBRARY( aeriallidar STATIC "src/AerialLiDAR.cu" "src/AerialLiDAR.cpp" "src/fileIO.cpp" "../../core/src/pugixml.cpp" )

include_directories("${CUDA_INCLUDE_DIRS}")
target_link_libraries( aeriallidar ${CUDA_LIBRARIES} )

#add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/ "plugins/visualizer" )
target_link_libraries( aeriallidar visualizer )
add_dependencies( aeriallidar visualizer )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/aeriallidar/ )

