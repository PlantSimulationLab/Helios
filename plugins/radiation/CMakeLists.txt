cmake_minimum_required(VERSION 3.15)

project(helios)


if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# ========== Core radiation plugin sources (backend-agnostic) ==========
set(RADIATION_CORE_SOURCES
    "src/RadiationModel.cpp"
    "src/RadiationCamera.cpp"
    "src/CameraCalibration.cpp"
    "src/LensFlare.cpp"
    "src/RayTracingTypes.cpp"
    "src/backends/BackendFactory.cpp"
    "tests/selfTest.cpp"
)

# ========== Detect Available Backends ==========

# Define FORCE_VULKAN_BACKEND as a cached variable so -D flag works properly
# This must be done BEFORE checking the variable value
if(NOT DEFINED FORCE_VULKAN_BACKEND)
    set(FORCE_VULKAN_BACKEND OFF CACHE BOOL "Force Vulkan backend even if CUDA is available (for testing)")
endif()

message(STATUS "[Radiation] FORCE_VULKAN_BACKEND = ${FORCE_VULKAN_BACKEND}")

# Try to find CUDA/OptiX (optional)
find_package(CUDAToolkit QUIET)

set(HAVE_OPTIX FALSE)
set(OPTIX_BACKEND_SOURCES "")

if(CUDAToolkit_FOUND AND NOT FORCE_VULKAN_BACKEND)
	message(STATUS "CUDA found - OptiX backend will be available")
	set(HAVE_OPTIX TRUE)
	set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

	# Set CUDA architectures before enabling CUDA language
	if( OPTIX_VERSION_LEGACY )
		set(CMAKE_CUDA_ARCHITECTURES "35")
	else()
		include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
	endif()

	# Enable CUDA as a language (modern CMake approach)
	enable_language(CUDA)

	# Add OptiX backend source
	list(APPEND OPTIX_BACKEND_SOURCES "src/backends/OptiX6Backend.cpp")
elseif(FORCE_VULKAN_BACKEND)
	message(STATUS "FORCE_VULKAN_BACKEND=ON - OptiX backend disabled for testing")
else()
	message(STATUS "CUDA not found - OptiX backend will NOT be available")
endif()

# Try to find Vulkan (optional, or force it for testing)
if(FORCE_VULKAN_BACKEND)
	find_package(Vulkan REQUIRED)
else()
	find_package(Vulkan QUIET)
endif()

set(HAVE_VULKAN FALSE)
set(VULKAN_BACKEND_SOURCES "")

if(Vulkan_FOUND)
	message(STATUS "Vulkan found - Vulkan compute backend will be available")
	set(HAVE_VULKAN TRUE)

	# Add Vulkan backend sources
	list(APPEND VULKAN_BACKEND_SOURCES
		"src/backends/VulkanComputeBackend.cpp"
		"src/backends/VulkanDevice.cpp"
		"src/backends/BVHBuilder.cpp"
	)
else()
	message(STATUS "Vulkan not found - Vulkan compute backend will NOT be available")
endif()

# Ensure at least one backend is available
if(NOT HAVE_OPTIX AND NOT HAVE_VULKAN)
	message(FATAL_ERROR
		"ERROR: No ray tracing backend available. "
		"Install CUDA Toolkit (for OptiX backend) or Vulkan SDK (for Vulkan compute backend). "
		"At least one backend is required to build the radiation plugin."
	)
endif()

# ========== Create radiation library with available backends ==========
add_library(radiation STATIC
	${RADIATION_CORE_SOURCES}
	${OPTIX_BACKEND_SOURCES}
	${VULKAN_BACKEND_SOURCES}
)

target_include_directories(radiation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/lib/json" )
target_include_directories(radiation PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backends")

# Add compile definitions for available backends
if(HAVE_OPTIX)
	target_compile_definitions(radiation PUBLIC HELIOS_HAVE_OPTIX)
endif()
if(HAVE_VULKAN)
	target_compile_definitions(radiation PUBLIC HELIOS_HAVE_VULKAN)
	target_include_directories(radiation PRIVATE
		"${Vulkan_INCLUDE_DIRS}"
		"${CMAKE_CURRENT_SOURCE_DIR}/lib/vma"
	)
	target_link_libraries(radiation PUBLIC ${Vulkan_LIBRARIES})
endif()

# ========== OptiX Backend Configuration (if available) ==========
if(HAVE_OPTIX)

# Windows-specific: Override CMake's automatic CUDA flag injection
if(WIN32 AND MSVC)
	# Clear all CMake-generated CUDA flags that contain problematic MSVC options
	set(CMAKE_CUDA_FLAGS_DEBUG "")
	set(CMAKE_CUDA_FLAGS_RELEASE "")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "")
	set(CMAKE_CUDA_FLAGS_MINSIZEREL "")
	
	# Override the CUDA compile rule to prevent MSVC flag injection
	set(CMAKE_CUDA_COMPILE_OBJECT 
		"<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE> -o <OBJECT>")
endif()

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

# Set CUDA flags after enabling language to avoid conflicts with CMake defaults
# Clear any CMake-generated optimization flags to prevent conflicts
set(CMAKE_CUDA_FLAGS_DEBUG_INIT "")
set(CMAKE_CUDA_FLAGS_RELEASE_INIT "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT "")
set(CMAKE_CUDA_FLAGS_MINSIZEREL_INIT "")

if(WIN32)
	# On Windows, use minimal flags and override any CMake defaults
	set(CMAKE_CUDA_FLAGS "--use_fast_math")
	if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ON )
	else()
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 ON )
	endif()
	
	# Force override CMake's build-type specific flags after they're set
	set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_MINSIZEREL "${CMAKE_CUDA_FLAGS}")
	
	# Windows-specific defines
	target_compile_definitions(radiation PRIVATE 
		NOMINMAX
		_USE_MATH_DEFINES
		_MWAITXINTRIN_H_INCLUDED
		__STRICT_ANSI__
	)
else()
	# Non-Windows platforms: set base flags without optimization
	set(CMAKE_CUDA_FLAGS "--use_fast_math -Wno-deprecated-gpu-targets")
	
	if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ON )
	else()
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 ON )
	endif()
	
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${OPTIX_PATH}include")
endif()

set( CUDA_SOURCES
		src/primitiveIntersection.cu
		src/rayGeneration.cu
		src/rayHit.cu
)

# OptiX-specific PTX compilation using direct nvcc calls
# OptiX requires special compilation with OptiX headers and runtime support

# Prepare OptiX-specific compilation flags  
set(OPTIX_NVCC_FLAGS
	--ptx
	--use_fast_math
	-I${CMAKE_CURRENT_SOURCE_DIR}/include
	-I${OPTIX_PATH}include
)

if(WIN32)
	list(APPEND OPTIX_NVCC_FLAGS -DNOMINMAX -D_USE_MATH_DEFINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ )
	# Force UTF-8 encoding for MSVC preprocessor to handle Unicode characters in source files
	list(APPEND OPTIX_NVCC_FLAGS -Xcompiler /utf-8)
	if(CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
		list(APPEND OPTIX_NVCC_FLAGS -g -O0)
	else()
		# Use -O3 for PTX compilation on Windows Release builds
		list(APPEND OPTIX_NVCC_FLAGS -O3)
	endif()
else()
	list(APPEND OPTIX_NVCC_FLAGS -Wno-deprecated-gpu-targets)
	if(CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
		list(APPEND OPTIX_NVCC_FLAGS -g -O0)
	else()
		# Use -O3 for PTX compilation on non-Windows Release builds
		list(APPEND OPTIX_NVCC_FLAGS -O3)
	endif()
	list(APPEND OPTIX_NVCC_FLAGS -std=c++${CMAKE_CXX_STANDARD})
endif()

# Generate PTX files using custom commands
set(PTX_FILES "")
foreach(CUDA_SOURCE ${CUDA_SOURCES})
	get_filename_component(CUDA_SOURCE_NAME ${CUDA_SOURCE} NAME_WE)
	get_filename_component(CUDA_SOURCE_DIR ${CUDA_SOURCE} DIRECTORY)
	
	# Output PTX file path
	set(PTX_FILE "${CMAKE_BINARY_DIR}/plugins/radiation/${CUDA_SOURCE_NAME}.ptx")
	list(APPEND PTX_FILES ${PTX_FILE})
	
	# Source file absolute path
	set(CUDA_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}")
	
	# Custom command to compile CUDA source to PTX with OptiX support
	add_custom_command(
		OUTPUT ${PTX_FILE}
		COMMAND ${CMAKE_CUDA_COMPILER} ${OPTIX_NVCC_FLAGS} -o ${PTX_FILE} ${CUDA_SOURCE_PATH}
		DEPENDS ${CUDA_SOURCE_PATH}
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Compiling OptiX PTX file ${CUDA_SOURCE_NAME}.ptx"
		VERBATIM
	)
endforeach()

# Create custom target to build all PTX files and copy them to expected locations
add_custom_target(ptx ALL
	DEPENDS ${PTX_FILES}
)

# Ensure output directory exists
add_custom_command(TARGET ptx PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/plugins/radiation/
)

# Copy PTX files with both legacy and new naming conventions for compatibility
set(CUDA_SOURCE_INDEX 0)
foreach(CUDA_SOURCE ${CUDA_SOURCES})
	get_filename_component(CUDA_SOURCE_NAME ${CUDA_SOURCE} NAME_WE)
	list(GET PTX_FILES ${CUDA_SOURCE_INDEX} PTX_FILE)
	
	# Copy to new-style name (expected by newer versions)
	add_custom_command(TARGET ptx POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${PTX_FILE}
			${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_${CUDA_SOURCE_NAME}.cu.ptx
	)
	
	# Copy to legacy name (expected by older versions)
	add_custom_command(TARGET ptx POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${PTX_FILE}
			${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_${CUDA_SOURCE_NAME}.cu.ptx
	)
	
	math(EXPR CUDA_SOURCE_INDEX "${CUDA_SOURCE_INDEX} + 1")
endforeach()

# Ensure main project depends on PTX target
add_dependencies(radiation ptx)

endif(HAVE_OPTIX)
# ========== End OptiX Backend Configuration ==========

# ========== Vulkan Backend Configuration (if available) ==========
if(HAVE_VULKAN)

# Find glslangValidator for shader compilation
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

if(NOT GLSLANG_VALIDATOR)
	message(FATAL_ERROR "glslangValidator not found. Install Vulkan SDK or glslang-tools package.")
endif()

message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR}")

# Common shader includes (shaders depend on these) - use absolute paths for proper dependency tracking
set(SHADER_COMMON_INCLUDES
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/random.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/transforms.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/buffer_indexing.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/bvh_traversal.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/intersections.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/material.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/texture_mask.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/sky_distribution.glsl
)

# Shader source files
set(SHADER_SOURCES
	shaders/direct_raygen.comp
	shaders/diffuse_raygen.comp
	shaders/camera_raygen.comp
	shaders/pixel_label_raygen.comp
)

# Generate SPIR-V files using custom commands
set(SPIRV_FILES "")
foreach(SHADER_SOURCE ${SHADER_SOURCES})
	get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)

	# Output SPIR-V file path
	set(SPIRV_FILE "${CMAKE_BINARY_DIR}/plugins/radiation/${SHADER_NAME}.spv")
	list(APPEND SPIRV_FILES ${SPIRV_FILE})

	# Source file absolute path
	set(SHADER_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}")

	# Custom command to compile GLSL to SPIR-V
	# Depends on both the shader source and common includes
	add_custom_command(
		OUTPUT ${SPIRV_FILE}
		COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1 -I${CMAKE_CURRENT_SOURCE_DIR}/shaders -o ${SPIRV_FILE} ${SHADER_SOURCE_PATH}
		DEPENDS ${SHADER_SOURCE_PATH} ${SHADER_COMMON_INCLUDES}
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Compiling Vulkan shader ${SHADER_NAME}.comp to SPIR-V"
		VERBATIM
	)
endforeach()

# Create custom target to build all SPIR-V files
add_custom_target(vulkan_shaders ALL
	DEPENDS ${SPIRV_FILES}
)

# Ensure output directory exists
add_custom_command(TARGET vulkan_shaders PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/plugins/radiation/
)

# Ensure main project depends on shader target
add_dependencies(radiation vulkan_shaders)

endif(HAVE_VULKAN)
# ========== End Vulkan Backend Configuration ==========

file( COPY include/disk.png DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/)
file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )
file( COPY camera_light_models DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )
file( COPY camera_library DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )

if(BUILD_TESTS)
    add_executable(radiation_tests "tests/TestMain.cpp")
    target_link_libraries(radiation_tests PRIVATE radiation)
    add_test(NAME radiation_tests COMMAND radiation_tests)
endif()