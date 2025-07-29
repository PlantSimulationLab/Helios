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
	if(EXISTS "${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
		include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
	else()
		set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86")
	endif()
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

# Windows-specific defines to prevent doctest compilation errors
if(WIN32 AND MSVC)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
	target_compile_definitions(radiation PRIVATE 
		NOMINMAX  # Prevent Windows.h min/max macros
		_USE_MATH_DEFINES  # Enable M_PI etc
	)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${OPTIX_PATH}include")

set( CUDA_SOURCES
		src/primitiveIntersection.cu
		src/rayGeneration.cu
		src/rayHit.cu
)

# Modern PTX compilation using CMake CUDA language support
add_library(radiation_cuda OBJECT ${CUDA_SOURCES})

# Set CUDA compilation to PTX instead of object files
set_property(TARGET radiation_cuda PROPERTY CUDA_PTX_COMPILATION ON)

# Set CUDA properties
set_target_properties(radiation_cuda PROPERTIES
	CUDA_SEPARABLE_COMPILATION OFF
	POSITION_INDEPENDENT_CODE ON
)

# Include directories for CUDA compilation
target_include_directories(radiation_cuda PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPTIX_PATH}include
)

# Get the list of PTX files that will be generated
set(PTX_FILES "")
foreach(CUDA_SOURCE ${CUDA_SOURCES})
	get_filename_component(CUDA_SOURCE_NAME ${CUDA_SOURCE} NAME_WE)
	
	# CMake generates PTX files in a specific pattern
	set(PTX_FILE "${CMAKE_BINARY_DIR}/plugins/radiation/CMakeFiles/radiation_cuda.dir/src/${CUDA_SOURCE_NAME}.ptx")
	list(APPEND PTX_FILES ${PTX_FILE})
endforeach()

# Create custom target to copy PTX files to expected locations
add_custom_target(ptx ALL
	DEPENDS radiation_cuda
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/plugins/radiation/
)

# Copy PTX files with both legacy and new naming conventions
set(CUDA_SOURCE_INDEX 0)
foreach(CUDA_SOURCE ${CUDA_SOURCES})
	get_filename_component(CUDA_SOURCE_NAME ${CUDA_SOURCE} NAME_WE)
	list(GET PTX_FILES ${CUDA_SOURCE_INDEX} PTX_FILE)
	
	# Copy to new-style name
	add_custom_command(TARGET ptx POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${PTX_FILE}
			${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_${CUDA_SOURCE_NAME}.cu.ptx
		COMMENT "Copying ${CUDA_SOURCE_NAME}.ptx to new-style name"
	)
	
	# Copy to legacy name
	add_custom_command(TARGET ptx POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${PTX_FILE}
			${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_${CUDA_SOURCE_NAME}.cu.ptx
		COMMENT "Copying ${CUDA_SOURCE_NAME}.ptx to legacy name"
	)
	
	math(EXPR CUDA_SOURCE_INDEX "${CUDA_SOURCE_INDEX} + 1")
endforeach()

# Ensure main project depends on PTX target
add_dependencies(radiation ptx)

file( COPY include/disk.png DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/)
file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )
file( COPY camera_light_models DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )

add_executable(radiation_tests "tests/TestMain.cpp")
target_link_libraries(radiation_tests PRIVATE radiation)
add_test(NAME radiation_tests COMMAND radiation_tests)