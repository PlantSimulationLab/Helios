cmake_minimum_required(VERSION 3.15)

project(helios)


if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(radiation STATIC "src/RadiationModel.cpp" "src/CameraCalibration.cpp" "tests/selfTest.cpp")

target_include_directories(radiation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	target_include_directories(radiation PUBLIC "${CUDA_INCLUDE_DIRS}" )
	target_link_libraries( radiation PUBLIC ${CUDA_LIBRARIES} )
endif(CUDA_FOUND)

if(UNIX AND NOT APPLE)
	if( OPTIX_VERSION_LEGACY )
		set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-5.1.0/)
		message("Using legacy OptiX version 5.1")
	else()
		set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-6.5.0/)
		message("Using OptiX version 6.5")
	endif()
	target_include_directories(radiation PUBLIC "${OPTIX_PATH}include" )
	target_link_libraries( radiation PUBLIC ${OPTIX_PATH}lib64/liboptix.so )
endif(UNIX AND NOT APPLE)
if(WIN32)
	if( OPTIX_VERSION_LEGACY )
		set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-5.1.1/)
		set(OPTIX_LIB "optix.51.lib")
		set(OPTIX_DLL "optix.51.dll")
		message("Using legacy OptiX version 5.1")
	else()
		set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-6.5.0/)
		set(OPTIX_LIB "optix.6.5.0.lib")
		set(OPTIX_DLL "optix.6.5.0.dll")
		message("Using OptiX version 6.5")
	endif()
	target_include_directories(radiation PUBLIC "${OPTIX_PATH}include" )
	target_link_libraries( radiation PUBLIC "${OPTIX_PATH}lib64/${OPTIX_LIB}" )
	add_custom_command( TARGET radiation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}lib64/${OPTIX_LIB}" "${CMAKE_BINARY_DIR}/." )
	add_custom_command( TARGET radiation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}bin64/${OPTIX_DLL}" "${CMAKE_BINARY_DIR}/." )

endif(WIN32)	
if(NOT DEFINED OPTIX_PATH)
       message( FATAL_ERROR "ERROR: Could not determine operating system for unknown reason." )
endif(NOT DEFINED OPTIX_PATH)

if( OPTIX_VERSION_LEGACY )
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
else()
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
endif()

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math -Wno-deprecated-gpu-targets" )

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
	set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -O0" )
	set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ON )
else()
	set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3" )
	set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 ON )
endif()

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++${CMAKE_CXX_STANDARD}" )

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${OPTIX_PATH}include")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${CUDA_INCLUDE_DIRS}")

set( CUDA_SOURCES
		src/primitiveIntersection.cu
		src/rayGeneration.cu
		src/rayHit.cu
)

cuda_compile_ptx(  cuda_ptx_files ${CUDA_SOURCES} )
add_custom_target(ptx DEPENDS ${cuda_ptx_files} ${CUDA_SOURCES} ${OPTIX_LIBRARIES} SOURCES ${CUDA_SOURCES} )
add_dependencies( radiation ptx )

#for cmake version 3.7 or greater, cuda_compile_ptx puts an _1_ in the ptx filename.  Rename the file to stay consistent.
if( CMAKE_VERSION VERSION_GREATER 3.6 )
add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_primitiveIntersection.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx DEPENDS ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_primitiveIntersection.cu.ptx )
add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayGeneration.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx DEPENDS ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayGeneration.cu.ptx )
add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayHit.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx DEPENDS ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayHit.cu.ptx )
endif()

# Ensure main project depends on PTX target
#add_dependencies(radiation ptx)

file( COPY include/disk.png DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/)
file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )
file( COPY camera_light_models DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )

add_executable(radiation_tests "tests/TestMain.cpp")
target_link_libraries(radiation_tests PRIVATE radiation)
add_test(NAME radiation_tests COMMAND radiation_tests)