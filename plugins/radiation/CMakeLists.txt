cmake_minimum_required(VERSION 2.0)

project(helios)

include_directories(include)
include_directories(../../core/include)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(radiation STATIC "src/RadiationModel.cpp")

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	target_link_libraries( radiation ${CUDA_LIBRARIES} )
endif(CUDA_FOUND)

if(UNIX AND NOT APPLE)
        #set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-3.8.0/)
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-4.0.1/)
        #set(OPTIX_PATH /usr/local/OptiX-3.6.0/)
	include_directories(${OPTIX_PATH}include)
	add_subdirectory(${OPTIX_PATH} "plugins/radiation")	
	target_link_libraries( radiation ${OPTIX_PATH}lib64/liboptix.so )
endif(UNIX AND NOT APPLE)
if(APPLE)
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/mac64-4.0.2/)
	include_directories(${OPTIX_PATH}include)
	add_subdirectory(${OPTIX_PATH})
	target_link_libraries( radiation ${OPTIX_PATH}lib64/liboptix.dylib )
endif(APPLE)
if(WIN32)
	message( FATAL_ERROR "ERROR: Windows platforms are not supported." )
endif(WIN32)	
if(NOT DEFINED OPTIX_PATH)
       message( FATAL_ERROR "ERROR: Could not determine operating system for unknown reason." )
endif(NOT DEFINED OPTIX_PATH)

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${OPTIX_PATH}include;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --use_fast_math -gencode arch=compute_30,code=sm_30)

set( CUDA_SOURCES "src/primitiveIntersection.cu;src/rayGeneration.cu;src/rayHit.cu;src/rayHit_MCRT.cu" )
cuda_compile_ptx(  cuda_ptx_files ${CUDA_SOURCES} )
add_custom_target(ptx ALL DEPENDS ${cuda_ptx_files}  ${CUDA_SOURCES} ${OPTIX_LIBRARIES} SOURCES ${CUDA_SOURCES} )

#for cmake version 3.7 or greater, cuda_compile_ptx puts an _1_ in the ptx filename.  Rename the file to stay consitent.
if( CMAKE_VERSION VERSION_GREATER 3.6 )
add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_primitiveIntersection.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx )
add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayGeneration.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx )
add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayHit.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx )
add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayHit_MCRT.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayHit_MCRT.cu.ptx )
endif()

file( COPY include/disk.png DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )

