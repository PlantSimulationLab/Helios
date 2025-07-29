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
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
else()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

add_library(energybalance STATIC "src/EnergyBalanceModel.cpp" "src/EnergyBalanceModel.cu" "tests/selfTest.cpp")

# Set CUDA properties for the target
set_target_properties(energybalance PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON )

# Windows-specific: Clear any inherited global compile options that cause nvcc issues
if(WIN32 AND MSVC)
	# Remove the global /utf-8 flag and other MSVC options from CUDA compilation
	set_property(TARGET energybalance PROPERTY COMPILE_OPTIONS "")
	set_property(TARGET energybalance PROPERTY COMPILE_DEFINITIONS "")
	
	# Windows-specific defines
	target_compile_definitions(energybalance PRIVATE NOMINMAX _USE_MATH_DEFINES _MWAITXINTRIN_H_INCLUDED __STRICT_ANSI__ )
endif()

target_link_libraries(energybalance PUBLIC CUDA::cudart_static)

target_include_directories(energybalance PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(energybalance_tests "tests/TestMain.cpp")
target_link_libraries(energybalance_tests PRIVATE energybalance)
add_test(NAME energybalance_tests COMMAND energybalance_tests)