cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# Try to find CUDA toolkit
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found - building with GPU acceleration")
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
    	
    	# Add position independent code flag for Linux (needed for static libraries)
    	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    	
    	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
    endif()
    
    if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
    	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
    else()
    	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()
    
    # Build with CUDA support
    add_library(collisiondetection STATIC "src/CollisionDetection.cu" "src/CollisionDetection.cpp" "src/CollisionDetection_RayTracing.cpp" "src/CollisionDetection_Slicing.cpp" "tests/selfTest.cpp")
    
    # Set CUDA properties for the target
    set_target_properties(collisiondetection PROPERTIES CUDA_RUNTIME_LIBRARY Static CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Windows-specific: Clear any inherited global compile options that cause nvcc issues
    if(WIN32 AND MSVC)
    	# Remove the global /utf-8 flag and other MSVC options from CUDA compilation
    	set_property(TARGET collisiondetection PROPERTY COMPILE_OPTIONS "")
    	set_property(TARGET collisiondetection PROPERTY COMPILE_DEFINITIONS "")
    	
    	# Windows-specific defines
    	target_compile_definitions(collisiondetection PRIVATE NOMINMAX _USE_MATH_DEFINES _MWAITXINTRIN_H_INCLUDED __STRICT_ANSI__ )
    endif()
    
    target_include_directories(collisiondetection PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    
    target_link_libraries(collisiondetection PUBLIC CUDA::cudart_static)

    # Define preprocessor macro to indicate CUDA support
    target_compile_definitions(collisiondetection PRIVATE HELIOS_CUDA_AVAILABLE)

else()
    message(STATUS "CUDA not found - building CPU-only version")
    
    # Build CPU-only version (exclude .cu file)
    add_library(collisiondetection STATIC "src/CollisionDetection.cpp" "src/CollisionDetection_RayTracing.cpp" "src/CollisionDetection_Slicing.cpp" "tests/selfTest.cpp")
    
    target_include_directories(collisiondetection PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    
endif()

if(BUILD_TESTS)
    add_executable(collisiondetection_tests "tests/TestMain.cpp")
    target_link_libraries(collisiondetection_tests PRIVATE collisiondetection)
    add_test(NAME collisiondetection_tests COMMAND collisiondetection_tests)
endif()

# Add benchmarks if enabled
if(BUILD_BENCHMARKS)
    # Only add benchmarks when we're in the main Helios project context
    if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/collision_detection_performance")
        add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks/collision_detection_performance collision_detection_benchmark)
    endif()
endif()