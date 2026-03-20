cmake_minimum_required(VERSION 3.22.1)

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

# Verify that the CUDA language can actually be enabled (e.g., VS integration is installed)
if(CUDAToolkit_FOUND AND NOT FORCE_VULKAN_BACKEND)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA toolkit found but CUDA language could not be enabled (missing Visual Studio integration?) - falling back to Vulkan-only build")
        set(CUDAToolkit_FOUND FALSE)
    endif()
endif()

set(HAVE_OPTIX FALSE)
set(HAVE_OPTIX8 FALSE)
set(OPTIX_BACKEND_SOURCES "")

if(CUDAToolkit_FOUND AND NOT FORCE_VULKAN_BACKEND)
	set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

	# Set CUDA architectures before enabling CUDA language
	include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")

	# Enable CUDA as a language (modern CMake approach)
	enable_language(CUDA)

	# ---- Driver version detection ----
	# Detect driver major version to select OptiX 6.5 vs OptiX 8.1
	set(NVIDIA_DRIVER_MAJOR 0)
	if(NOT SKIP_DRIVER_VERSION_CHECK)
		find_program(NVIDIA_SMI_EXE nvidia-smi)
		if(NVIDIA_SMI_EXE)
			execute_process(
				COMMAND ${NVIDIA_SMI_EXE} --query-gpu=driver_version --format=csv,noheader,nounits
				OUTPUT_VARIABLE _DRIVER_VER
				OUTPUT_STRIP_TRAILING_WHITESPACE
				ERROR_QUIET
				RESULT_VARIABLE _DRIVER_RESULT
			)
			if(_DRIVER_RESULT EQUAL 0 AND _DRIVER_VER)
				string(REGEX MATCH "[^\n]+" _DRIVER_VER "${_DRIVER_VER}")
				string(STRIP "${_DRIVER_VER}" _DRIVER_VER)
				string(REGEX MATCH "^([0-9]+)" NVIDIA_DRIVER_MAJOR "${_DRIVER_VER}")
				message(STATUS "[Radiation] Detected NVIDIA driver version: ${_DRIVER_VER}")
			endif()
		endif()
	endif()

	# ---- CUDA Toolkit vs Driver compatibility check ----
	# The CUDA toolkit version must not exceed the driver's supported CUDA version,
	# otherwise compiled device code (PTX/OptiXIR) will contain NVVM IR too new for
	# the driver's JIT compiler, causing OPTIX_ERROR_INVALID_INPUT at runtime.
	if(NOT SKIP_DRIVER_VERSION_CHECK AND NVIDIA_SMI_EXE)
		execute_process(
			COMMAND ${NVIDIA_SMI_EXE} --query-gpu=driver_version --format=csv,noheader,nounits
			OUTPUT_VARIABLE _DRIVER_CUDA_CHECK
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET
			RESULT_VARIABLE _CUDA_CHECK_RESULT
		)
		if(_CUDA_CHECK_RESULT EQUAL 0)
			# Query the maximum CUDA version supported by the driver
			execute_process(
				COMMAND ${NVIDIA_SMI_EXE}
				OUTPUT_VARIABLE _SMI_FULL_OUTPUT
				OUTPUT_STRIP_TRAILING_WHITESPACE
				ERROR_QUIET
			)
			string(REGEX MATCH "CUDA Version: ([0-9]+)\\.([0-9]+)" _CUDA_VER_MATCH "${_SMI_FULL_OUTPUT}")
			if(_CUDA_VER_MATCH)
				set(_DRIVER_CUDA_MAJOR "${CMAKE_MATCH_1}")
				set(_DRIVER_CUDA_MINOR "${CMAKE_MATCH_2}")
				message(STATUS "[Radiation] Driver supports up to CUDA ${_DRIVER_CUDA_MAJOR}.${_DRIVER_CUDA_MINOR}")

				# Compare with installed CUDA toolkit version
				if(CUDAToolkit_VERSION_MAJOR AND CUDAToolkit_VERSION_MINOR)
					if(CUDAToolkit_VERSION_MAJOR GREATER _DRIVER_CUDA_MAJOR OR
					   (CUDAToolkit_VERSION_MAJOR EQUAL _DRIVER_CUDA_MAJOR AND CUDAToolkit_VERSION_MINOR GREATER _DRIVER_CUDA_MINOR))
						message(FATAL_ERROR
							"\n"
							"======================================================================\n"
							"CUDA Toolkit / Driver version mismatch!\n"
							"\n"
							"  Installed CUDA Toolkit:  ${CUDAToolkit_VERSION}\n"
							"  Driver supports up to:   CUDA ${_DRIVER_CUDA_MAJOR}.${_DRIVER_CUDA_MINOR}\n"
							"  NVIDIA driver version:   ${_DRIVER_VER}\n"
							"\n"
							"The CUDA ${CUDAToolkit_VERSION} compiler generates device code (NVVM IR)\n"
							"that is too new for this driver's JIT compiler. This will cause\n"
							"OPTIX_ERROR_INVALID_INPUT at runtime.\n"
							"\n"
							"Fix options:\n"
							"  1. Update your NVIDIA driver to one that supports CUDA ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}+\n"
							"  2. Install CUDA Toolkit ${_DRIVER_CUDA_MAJOR}.${_DRIVER_CUDA_MINOR} or older\n"
							"  3. Use -DFORCE_VULKAN_BACKEND=ON to bypass OptiX entirely\n"
							"  4. Use -DSKIP_DRIVER_VERSION_CHECK=ON to bypass this check\n"
							"======================================================================\n"
						)
					endif()
				endif()
			endif()
		endif()
	endif()

	# ---- Select OptiX version based on driver ----
	# OptiX 8 requires driver >= 560 (OptiX 8.0 minimum). OptiX 6.5 unsupported on >= 590.
	if(NOT OPTIX_VERSION_LEGACY AND (NVIDIA_DRIVER_MAJOR GREATER_EQUAL 560 OR NVIDIA_DRIVER_MAJOR EQUAL 0))
		# Try to find OptiX 8 SDK headers
		# Search order: user-supplied OPTIX8_INCLUDE_DIR → bundled in-tree path
		set(_OPTIX8_BUNDLED_INCLUDE "")
		if(UNIX AND NOT APPLE)
			set(_OPTIX8_BUNDLED_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-8.1.0/include")
		elseif(WIN32)
			set(_OPTIX8_BUNDLED_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-8.1.0/include")
		endif()

		if(DEFINED OPTIX8_INCLUDE_DIR AND EXISTS "${OPTIX8_INCLUDE_DIR}/optix.h")
			set(_OPTIX8_INCLUDE "${OPTIX8_INCLUDE_DIR}")
			message(STATUS "[Radiation] Using user-supplied OptiX 8 SDK: ${_OPTIX8_INCLUDE}")
		elseif(EXISTS "${_OPTIX8_BUNDLED_INCLUDE}/optix.h")
			set(_OPTIX8_INCLUDE "${_OPTIX8_BUNDLED_INCLUDE}")
			message(STATUS "[Radiation] Using bundled OptiX 8 SDK: ${_OPTIX8_INCLUDE}")
		else()
			set(_OPTIX8_INCLUDE "")
			message(STATUS "[Radiation] OptiX 8 SDK not found (driver >= 560 detected). "
				"To enable OptiX 8 backend, either:\n"
				"  - Place the OptiX 8.1 SDK headers in ${_OPTIX8_BUNDLED_INCLUDE}\n"
				"  - Or set -DOPTIX8_INCLUDE_DIR=/path/to/OptiX8/include\n"
				"Falling back to OptiX 6.5 if available.")
		endif()

		if(_OPTIX8_INCLUDE)
			set(HAVE_OPTIX8 TRUE)
			message(STATUS "[Radiation] OptiX 8.1 backend enabled (driver ${_DRIVER_VER})")
			list(APPEND OPTIX_BACKEND_SOURCES "src/backends/OptiX8Backend.cpp")
		endif()
	endif()

	# Fall back to OptiX 6.5 if OptiX 8 not available (driver < 560 or SDK not found)
	if(NOT HAVE_OPTIX8)
		message(STATUS "[Radiation] Using OptiX 6.5 backend")
		set(HAVE_OPTIX TRUE)
		list(APPEND OPTIX_BACKEND_SOURCES "src/backends/OptiX6Backend.cpp")
	endif()

elseif(FORCE_VULKAN_BACKEND)
	message(STATUS "FORCE_VULKAN_BACKEND=ON - OptiX backend disabled for testing")
else()
	message(STATUS "CUDA not found - OptiX backend will NOT be available")
endif()

# Try to find Vulkan headers and libraries
# Search order: bundled headers → system Vulkan SDK
set(_VULKAN_BUNDLED_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/lib/vulkan/include")

set(HAVE_VULKAN FALSE)
set(VULKAN_BACKEND_SOURCES "")

if(EXISTS "${_VULKAN_BUNDLED_INCLUDE}/vulkan/vulkan.h")
	# Use bundled Vulkan headers - always enable backend when headers are available
	set(Vulkan_INCLUDE_DIRS "${_VULKAN_BUNDLED_INCLUDE}")
	set(HAVE_VULKAN TRUE)
	message(STATUS "[Radiation] Using bundled Vulkan headers: ${_VULKAN_BUNDLED_INCLUDE}")

	# Try to find the Vulkan loader library (needed for linking/runtime)
	# Do NOT call find_package(Vulkan) here — it would overwrite our bundled Vulkan_INCLUDE_DIRS
	if(APPLE)
		find_library(Vulkan_LIBRARIES NAMES vulkan vulkan-1 MoltenVK PATHS
			"$ENV{VULKAN_SDK}/lib"
			"/opt/homebrew/lib"
			"/usr/local/lib"
		)
	else()
		find_library(Vulkan_LIBRARIES NAMES vulkan vulkan-1 PATHS
			"$ENV{VULKAN_SDK}/Lib"
			"$ENV{VULKAN_SDK}/lib"
			"/usr/lib"
			"/usr/lib/x86_64-linux-gnu"
			"/usr/lib64"
			"/usr/local/lib"
		)
	endif()

	# Platform-specific fallbacks when find_library fails
	if(WIN32 AND NOT Vulkan_LIBRARIES)
		# On Windows, vulkan-1.dll is installed by GPU drivers in System32
		# but the import library (vulkan-1.lib) is only in the Vulkan SDK.
		# Generate an import library from a bundled .def file.
		set(_VULKAN_DEF "${CMAKE_CURRENT_SOURCE_DIR}/lib/vulkan/vulkan-1.def")
		set(_VULKAN_IMPLIB "${CMAKE_BINARY_DIR}/lib/vulkan-1.lib")
		if(EXISTS "${_VULKAN_DEF}" AND MSVC)
			file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
			# Use CMAKE_AR (lib.exe) from the MSVC toolchain
			execute_process(
				COMMAND "${CMAKE_AR}" /DEF:${_VULKAN_DEF} /OUT:${_VULKAN_IMPLIB} /MACHINE:X64
				WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
				OUTPUT_QUIET
				ERROR_QUIET
				RESULT_VARIABLE _LIB_RESULT
			)
			if(_LIB_RESULT EQUAL 0)
				set(Vulkan_LIBRARIES "${_VULKAN_IMPLIB}")
				message(STATUS "[Radiation] Generated Vulkan import library from bundled .def file")
			else()
				message(WARNING "[Radiation] Failed to generate Vulkan import library (CMAKE_AR=${CMAKE_AR}). Install Vulkan SDK.")
				set(Vulkan_LIBRARIES "vulkan-1")
			endif()
		else()
			set(Vulkan_LIBRARIES "vulkan-1")
			message(STATUS "[Radiation] Using system Vulkan loader (installed by GPU drivers)")
		endif()
	elseif(NOT WIN32 AND NOT Vulkan_LIBRARIES)
		# On Linux/macOS, fall back to default library name so the linker can resolve it
		if(APPLE)
			set(Vulkan_LIBRARIES "vulkan")
			message(STATUS "[Radiation] Vulkan loader not found. Install with: brew install vulkan-loader molten-vk")
		else()
			set(Vulkan_LIBRARIES "vulkan")
			message(STATUS "[Radiation] Vulkan loader not found. Install with: sudo apt-get install libvulkan-dev")
		endif()
	elseif(Vulkan_LIBRARIES)
		message(STATUS "[Radiation] Vulkan loader library found: ${Vulkan_LIBRARIES}")
	endif()
else()
	# Bundled headers not found, try system Vulkan SDK
	if(FORCE_VULKAN_BACKEND)
		find_package(Vulkan REQUIRED)
	else()
		find_package(Vulkan QUIET)
	endif()

	if(Vulkan_FOUND)
		message(STATUS "[Radiation] Using system Vulkan SDK")
		set(HAVE_VULKAN TRUE)
	else()
		message(STATUS "[Radiation] Vulkan not found - Vulkan compute backend will NOT be available")
	endif()
endif()

if(HAVE_VULKAN)
	# Add Vulkan backend sources
	list(APPEND VULKAN_BACKEND_SOURCES
		"src/backends/VulkanComputeBackend.cpp"
		"src/backends/VulkanDevice.cpp"
		"src/backends/BVHBuilder.cpp"
	)
endif()

# Ensure at least one backend is available
if(NOT HAVE_OPTIX AND NOT HAVE_OPTIX8 AND NOT HAVE_VULKAN)
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
if(HAVE_OPTIX8)
	target_compile_definitions(radiation PUBLIC HELIOS_HAVE_OPTIX8)
endif()
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

# ========== OptiX 8.1 Backend Configuration ==========
if(HAVE_OPTIX8)

# OptiX 8 is header-only (driver provides liboptix.so at runtime via optixInit())
target_include_directories(radiation PUBLIC "${_OPTIX8_INCLUDE}")
target_include_directories(radiation PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/src/optix8"
)
target_link_libraries(radiation PUBLIC CUDA::cudart_static)

# CUDA flags for host-side compilation of OptiX8Backend.cpp
set(CMAKE_CUDA_FLAGS_DEBUG_INIT "")
set(CMAKE_CUDA_FLAGS_RELEASE_INIT "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT "")
set(CMAKE_CUDA_FLAGS_MINSIZEREL_INIT "")

if(WIN32)
	set(CMAKE_CUDA_FLAGS "--use_fast_math")
	target_compile_definitions(radiation PRIVATE NOMINMAX _USE_MATH_DEFINES _MWAITXINTRIN_H_INCLUDED __STRICT_ANSI__)
else()
	set(CMAKE_CUDA_FLAGS "--use_fast_math -Wno-deprecated-gpu-targets")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
endif()

# ---- Compile OptiX 8 device code to OptiXIR (preferred) or PTX ----
set(OPTIX8_DEVICE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/optix8/OptiX8DeviceCode.cu")
set(OPTIX8_IR_OUTPUT "${CMAKE_BINARY_DIR}/plugins/radiation/OptiX8DeviceCode.optixir")

set(OPTIX8_NVCC_FLAGS
	--optix-ir
	--use_fast_math
	-diag-suppress 20044
	-I${CMAKE_CURRENT_SOURCE_DIR}/src/optix8
	-I${CMAKE_CURRENT_SOURCE_DIR}/include
	-I${_OPTIX8_INCLUDE}
)

if(WIN32)
	list(APPEND OPTIX8_NVCC_FLAGS -DNOMINMAX -D_USE_MATH_DEFINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -Xcompiler /utf-8)
else()
	list(APPEND OPTIX8_NVCC_FLAGS -Wno-deprecated-gpu-targets -std=c++${CMAKE_CXX_STANDARD})
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
	list(APPEND OPTIX8_NVCC_FLAGS -g -O0)
else()
	list(APPEND OPTIX8_NVCC_FLAGS -O3)
endif()

add_custom_command(
	OUTPUT ${OPTIX8_IR_OUTPUT}
	COMMAND ${CMAKE_CUDA_COMPILER} ${OPTIX8_NVCC_FLAGS} -o ${OPTIX8_IR_OUTPUT} ${OPTIX8_DEVICE_SOURCE}
	DEPENDS ${OPTIX8_DEVICE_SOURCE} "${CMAKE_CURRENT_SOURCE_DIR}/src/optix8/OptiX8LaunchParams.h"
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Compiling OptiX 8.1 device code to OptiXIR"
	VERBATIM
)

add_custom_target(optix8_ir ALL DEPENDS ${OPTIX8_IR_OUTPUT})
add_custom_command(TARGET optix8_ir PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/plugins/radiation/
)
add_dependencies(radiation optix8_ir)

endif(HAVE_OPTIX8)
# ========== End OptiX 8.1 Backend Configuration ==========

# ========== OptiX 6.5 Backend Configuration (legacy drivers) ==========
if(HAVE_OPTIX)

# ---- NVIDIA driver version compatibility check ----
# NVIDIA drivers 590.x+ removed support for the OptiX 6.5 API.
if(NOT SKIP_DRIVER_VERSION_CHECK AND NVIDIA_DRIVER_MAJOR AND NVIDIA_DRIVER_MAJOR GREATER_EQUAL 590)
	message(FATAL_ERROR
		"\n"
		"======================================================================\n"
		"NVIDIA driver version ${_DRIVER_VER} is not compatible with OptiX 6.5.\n"
		"\n"
		"NVIDIA drivers 590.x and later removed support for the OptiX 6.5 API\n"
		"used by Helios. This is a known issue affecting both Linux and Windows.\n"
		"\n"
		"Workarounds:\n"
		"  1. Downgrade to an NVIDIA driver < 590 (e.g., 550.x or 535.x LTS).\n"
		"  2. Install the OptiX 8.1 SDK in lib/OptiX/linux64-8.1.0/ to use the\n"
		"     OptiX 8 backend (compatible with drivers >= 560).\n"
		"  3. If you have the Vulkan SDK installed, use -DFORCE_VULKAN_BACKEND=ON.\n"
		"\n"
		"To bypass this check (e.g., cross-compiling), set:\n"
		"  -DSKIP_DRIVER_VERSION_CHECK=ON\n"
		"======================================================================\n"
	)
endif()

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
	set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-6.5.0/)
	message(STATUS "[Radiation] Using OptiX version 6.5")
	target_include_directories(radiation PUBLIC "${OPTIX_PATH}include" )
	target_link_libraries( radiation PUBLIC ${OPTIX_PATH}lib64/liboptix.so )
endif(UNIX AND NOT APPLE)
if(WIN32)
	set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-6.5.0/)
	set(OPTIX_LIB "optix.6.5.0.lib")
	set(OPTIX_DLL "optix.6.5.0.dll")
	message(STATUS "[Radiation] Using OptiX version 6.5")
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
# ========== End OptiX 6.5 Backend Configuration ==========

# ========== Vulkan Backend Configuration (if available) ==========
if(HAVE_VULKAN)

# ---- Build glslang from bundled source or find system installation ----
set(_GLSLANG_BUNDLED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/glslang-16.2.0/glslang-16.2.0")

if(EXISTS "${_GLSLANG_BUNDLED_DIR}/CMakeLists.txt")
	message(STATUS "[Radiation] Building glslang from bundled source")

	# Configure glslang build options: only build what we need
	set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
	set(ENABLE_SPIRV ON CACHE BOOL "" FORCE)
	set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
	set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
	set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
	set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
	set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

	add_subdirectory("${_GLSLANG_BUNDLED_DIR}" "${CMAKE_BINARY_DIR}/glslang-build" EXCLUDE_FROM_ALL)

	set(GLSLANG_VALIDATOR $<TARGET_FILE:glslang-standalone>)
	message(STATUS "[Radiation] glslangValidator will be built from source")
else()
	# Fallback: find system-installed glslangValidator
	find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
	if(NOT GLSLANG_VALIDATOR)
		message(FATAL_ERROR "glslangValidator not found. Install Vulkan SDK or glslang-tools package.")
	endif()
	message(STATUS "[Radiation] Found system glslangValidator: ${GLSLANG_VALIDATOR}")
endif()

# Common shader includes (shaders depend on these) - use absolute paths for proper dependency tracking
set(SHADER_COMMON_INCLUDES
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/random.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/transforms.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/buffer_indexing.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/bvh_traversal.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/periodic_boundary.glsl
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/common/intersections.glsl
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

# If using bundled glslang, shaders must wait for glslang-standalone to be built
if(TARGET glslang-standalone)
	add_dependencies(vulkan_shaders glslang-standalone)
endif()

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