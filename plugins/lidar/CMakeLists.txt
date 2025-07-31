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

add_library(lidar STATIC "src/LiDAR.cu" "src/LiDAR.cpp" "src/fileIO.cpp" "src/QSM.cpp" "tests/selfTest.cpp" "../../core/lib/pugixml/pugixml.cpp" "lib/s_hull_pro/s_hull_pro.cpp")

# Set CUDA properties for the target
set_target_properties(lidar PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON )

# Windows-specific: Clear any inherited global compile options that cause nvcc issues
if(WIN32 AND MSVC)
	# Remove the global /utf-8 flag and other MSVC options from CUDA compilation
	set_property(TARGET lidar PROPERTY COMPILE_OPTIONS "")
	set_property(TARGET lidar PROPERTY COMPILE_DEFINITIONS "")
	
	# Windows-specific defines
	target_compile_definitions(lidar PRIVATE NOMINMAX _USE_MATH_DEFINES _MWAITXINTRIN_H_INCLUDED __STRICT_ANSI__ )
endif()

target_include_directories(lidar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(lidar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib/s_hull_pro)

target_link_libraries(lidar PUBLIC visualizer CUDA::cudart_static)
add_dependencies(lidar visualizer)

#add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip" )
#include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip/include" )

#set( WITH_LASZIP FALSE )
#set( WITH_UTILITIES FALSE )
#set( WITH_TESTS FALSE )
#set( WITH_GEOTIFF FALSE )
#if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
#     set( LASZIP_LIBRARY "${CMAKE_BINARY_DIR}/plugins/lidar/lib/laszip/bin/Debug/liblaszip.so" )
#else()
#     set( LASZIP_LIBRARY "${CMAKE_BINARY_DIR}/plugins/lidar/lib/laszip/bin/Release/liblaszip.so" )
#endif()
#set( LASZIP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip/include/laszip" )
#add_subdirectory( "lib/libLAS" )
#include_directories( "lib/libLAS/include/liblas" )
#if( WIN32 )
#     set( LIBLAS_LIB_NAME "liblas" )
#else()
#     set( LIBLAS_LIB_NAME "las" )
#endif()
#target_link_libraries( lidar "${LIBLAS_LIB_NAME}" )
#add_dependencies( "${LIBLAS_LIB_NAME}" laszip )
#add_dependencies( lidar "${LIBLAS_LIB_NAME}" )

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )
file( COPY data DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )

add_executable(lidar_tests "tests/TestMain.cpp")
target_link_libraries(lidar_tests PRIVATE lidar)
add_test(NAME lidar_tests COMMAND lidar_tests)