cmake_minimum_required(VERSION 3.15)

project(helios)

# ============================================================================
# DEPRECATION WARNING
# ============================================================================
message(WARNING "
********************************************************************************
*** DEPRECATED PLUGIN: VoxelIntersection                                    ***
********************************************************************************
The VoxelIntersection plugin has been DEPRECATED and merged into the
CollisionDetection plugin. All functionality is now available in CollisionDetection.

PLEASE UPDATE YOUR CODE:
  - Replace: set(PLUGINS \"voxelintersection\")
  - With:    set(PLUGINS \"collisiondetection\")

  - Replace: #include \"VoxelIntersection.h\"
  - With:    #include \"CollisionDetection.h\"

  - Replace: VoxelIntersection class
  - With:    CollisionDetection class

All method names remain the same. See CollisionDetection plugin documentation
for details: https://baileylab.ucdavis.edu/software/helios/doc/

This plugin will be REMOVED in a future release.
********************************************************************************
")

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

# Windows-specific: Override CMake's automatic CUDA flag injection
if(WIN32 AND MSVC)
	# Clear all CMake-generated CUDA flags that contain problematic MSVC options
	set(CMAKE_CUDA_FLAGS_DEBUG "")
	set(CMAKE_CUDA_FLAGS_RELEASE "")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "")
	set(CMAKE_CUDA_FLAGS_MINSIZEREL "")
	
	# Override the CUDA compile rule to prevent MSVC flag injection
	set(CMAKE_CUDA_COMPILE_OBJECT "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE> -o <OBJECT>")
endif()

# Set CUDA flags after enabling language to override defaults
if(WIN32)
	# On Windows, use minimal flags and override any CMake defaults
	set(CMAKE_CUDA_FLAGS "--use_fast_math")
	
	# Force override CMake's build-type specific flags after they're set
	set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_MINSIZEREL "${CMAKE_CUDA_FLAGS}")
else()
	# Non-Windows platforms use full flags
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Wno-deprecated-gpu-targets")
	
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
endif()

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
else()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

add_library(voxelintersection STATIC "src/VoxelIntersection.cu" "src/VoxelIntersection.cpp" "tests/selfTest.cpp")

# Set CUDA properties for the target
set_target_properties(voxelintersection PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON POSITION_INDEPENDENT_CODE ON)

# Windows-specific: Clear any inherited global compile options that cause nvcc issues
if(WIN32 AND MSVC)
	# Remove the global /utf-8 flag and other MSVC options from CUDA compilation
	set_property(TARGET voxelintersection PROPERTY COMPILE_OPTIONS "")
	set_property(TARGET voxelintersection PROPERTY COMPILE_DEFINITIONS "")
	
	# Windows-specific defines
	target_compile_definitions(voxelintersection PRIVATE NOMINMAX _USE_MATH_DEFINES _MWAITXINTRIN_H_INCLUDED __STRICT_ANSI__ )
endif()

target_include_directories(voxelintersection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(voxelintersection PUBLIC CUDA::cudart_static)

if(BUILD_TESTS)
    add_executable(voxelintersection_tests "tests/TestMain.cpp")
    target_link_libraries(voxelintersection_tests PRIVATE voxelintersection)
    add_test(NAME voxelintersection_tests COMMAND voxelintersection_tests)
endif()