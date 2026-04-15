cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(parameteroptimization STATIC "src/ParameterOptimization.cpp" "tests/selfTest.cpp")

target_include_directories(parameteroptimization PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

# ---- NLopt (for L-BFGS, BOBYQA, SLSQP gradient-based/local optimization) ----
# Strategy: build from bundled source, fall back to system-installed NLopt

set(_NLOPT_BUNDLED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/nlopt-2.10.1")

if(EXISTS "${_NLOPT_BUNDLED_DIR}/CMakeLists.txt")
    message(STATUS "[ParameterOptimization] Building NLopt from bundled source")

    # Configure NLopt build: static library only, no extras
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(NLOPT_CXX ON CACHE BOOL "" FORCE)
    set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
    set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
    set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
    set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
    set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)
    set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)

    add_subdirectory("${_NLOPT_BUNDLED_DIR}" "${CMAKE_BINARY_DIR}/nlopt-build" EXCLUDE_FROM_ALL)

    target_compile_definitions(parameteroptimization PUBLIC HELIOS_HAVE_NLOPT)
    target_link_libraries(parameteroptimization PUBLIC nlopt)
    # NLopt generates nlopt.hpp in the build directory
    target_include_directories(parameteroptimization PUBLIC
        "${CMAKE_BINARY_DIR}/nlopt-build/src/api"
        "${_NLOPT_BUNDLED_DIR}/src/api"
    )
else()
    # Fallback: try to find system-installed NLopt
    find_package(NLopt QUIET)

    if(NOT NLopt_FOUND)
        if(APPLE)
            find_library(NLopt_LIBRARY NAMES nlopt PATHS "/opt/homebrew/lib" "/usr/local/lib")
            find_path(NLopt_INCLUDE_DIR NAMES nlopt.hpp PATHS "/opt/homebrew/include" "/usr/local/include")
        else()
            find_library(NLopt_LIBRARY NAMES nlopt PATHS "/usr/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/usr/local/lib")
            find_path(NLopt_INCLUDE_DIR NAMES nlopt.hpp PATHS "/usr/include" "/usr/local/include")
        endif()
        if(NLopt_LIBRARY AND NLopt_INCLUDE_DIR)
            set(NLopt_FOUND TRUE)
        endif()
    endif()

    if(NLopt_FOUND)
        message(STATUS "[ParameterOptimization] Using system NLopt")
        target_compile_definitions(parameteroptimization PUBLIC HELIOS_HAVE_NLOPT)
        if(TARGET NLopt::nlopt)
            target_link_libraries(parameteroptimization PUBLIC NLopt::nlopt)
        else()
            target_include_directories(parameteroptimization PUBLIC "${NLopt_INCLUDE_DIR}")
            target_link_libraries(parameteroptimization PUBLIC "${NLopt_LIBRARY}")
        endif()
    else()
        message(STATUS "[ParameterOptimization] NLopt not found — L-BFGS, BOBYQA, SLSQP disabled (GA/BO/CMA-ES/Adam still available)")
    endif()
endif()

# ---- Tests ----

if(BUILD_TESTS)
    add_executable( parameteroptimization_tests "tests/TestMain.cpp" )
    target_link_libraries( parameteroptimization_tests PRIVATE parameteroptimization )
    add_test( NAME parameteroptimization_tests COMMAND parameteroptimization_tests )
endif()
