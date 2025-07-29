cmake_minimum_required(VERSION 3.15)

project(helios)


if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(radiation STATIC "src/RadiationModel.cpp" "src/CameraCalibration.cpp" "tests/selfTest.cpp")

target_include_directories(radiation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

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

target_link_libraries(radiation PUBLIC CUDA::cudart_static)

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

# Set additional CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Wno-deprecated-gpu-targets")

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
	set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ON )
else()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
	set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 ON )
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${OPTIX_PATH}include")

set( CUDA_SOURCES
		src/primitiveIntersection.cu
		src/rayGeneration.cu
		src/rayHit.cu
)

# Modern PTX compilation using custom targets (avoids OBJECT library limitations)
set(PTX_FILES "")
foreach(CUDA_SOURCE ${CUDA_SOURCES})
	get_filename_component(CUDA_SOURCE_NAME ${CUDA_SOURCE} NAME_WE)
	set(PTX_FILE "${CMAKE_BINARY_DIR}/plugins/radiation/${CUDA_SOURCE_NAME}.ptx")
	list(APPEND PTX_FILES ${PTX_FILE})
	
	# Create custom command to compile each CUDA source to PTX
	add_custom_command(
		OUTPUT ${PTX_FILE}
		COMMAND ${CMAKE_CUDA_COMPILER}
			${CMAKE_CUDA_FLAGS}
			-I"${CMAKE_CURRENT_SOURCE_DIR}/include"
			-I"${OPTIX_PATH}include"
			--ptx
			-o ${PTX_FILE}
			${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}
		COMMENT "Compiling ${CUDA_SOURCE} to PTX"
		VERBATIM
	)
endforeach()

# Create custom target for all PTX files
add_custom_target(radiation_ptx ALL DEPENDS ${PTX_FILES})

# Create legacy-named PTX files for compatibility (the application expects these names)
add_custom_command(TARGET radiation_ptx POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different 
		${CMAKE_BINARY_DIR}/plugins/radiation/primitiveIntersection.ptx
		${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx
	COMMAND ${CMAKE_COMMAND} -E copy_if_different 
		${CMAKE_BINARY_DIR}/plugins/radiation/rayGeneration.ptx
		${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx  
	COMMAND ${CMAKE_COMMAND} -E copy_if_different 
		${CMAKE_BINARY_DIR}/plugins/radiation/rayHit.ptx
		${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx
	COMMENT "Creating legacy-named PTX files for compatibility"
)

# Ensure main project depends on PTX target
add_dependencies(radiation radiation_ptx)

file( COPY include/disk.png DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/)
file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )
file( COPY camera_light_models DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )

add_executable(radiation_tests "tests/TestMain.cpp")
target_link_libraries(radiation_tests PRIVATE radiation)
add_test(NAME radiation_tests COMMAND radiation_tests)