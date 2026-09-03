cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(parameteroptimization STATIC "src/ParameterOptimization.cpp" "tests/selfTest.cpp")

target_include_directories(parameteroptimization PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

# ---- NLopt (for L-BFGS, BOBYQA, SLSQP gradient-based/local optimization) ----
# Strategy: build from bundled source, fall back to system-installed NLopt

# NLopt is MIT-licensed except for src/algs/luksan/, which is LGPL-2.1 and whose
# COPYRIGHT additionally carries an ACM notice restricting redistribution. Per NLopt's
# own COPYING, omitting that directory makes the compiled library entirely MIT.
#
# L-BFGS (NLOPT_LD_LBFGS) is implemented entirely by the Luksan code, so turning this OFF
# removes L-BFGS; ParameterOptimization then reports that explicitly rather than failing
# obscurely. BOBYQA (MIT) and SLSQP (BSD-3) are unaffected, as are the four algorithms
# that need no NLopt at all (GA, Bayesian Optimization, CMA-ES, Adam).
#
# Defaults to ON so existing builds are unchanged.
option(HELIOS_NLOPT_LUKSAN "Enable NLopt's LGPL Luksan solvers (provides L-BFGS). Set OFF for an MIT-licensed NLopt build without L-BFGS." ON)

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
    set(NLOPT_LUKSAN ${HELIOS_NLOPT_LUKSAN} CACHE BOOL "" FORCE)

    add_subdirectory("${_NLOPT_BUNDLED_DIR}" "${CMAKE_BINARY_DIR}/nlopt-build" EXCLUDE_FROM_ALL)

    target_compile_definitions(parameteroptimization PUBLIC HELIOS_HAVE_NLOPT)

    # The bundled build is authoritative: we set NLOPT_LUKSAN ourselves, so L-BFGS
    # availability is known with certainty rather than assumed.
    if(HELIOS_NLOPT_LUKSAN)
        target_compile_definitions(parameteroptimization PUBLIC HELIOS_HAVE_LBFGS)
    else()
        message(STATUS "[ParameterOptimization] Luksan solvers disabled — NLopt build is MIT-licensed; L-BFGS unavailable (use Adam or BOBYQA)")
    endif()

    # Linking the nlopt target propagates its own INTERFACE include directories,
    # which already cover both the generated nlopt.hpp and the checked-in nlopt.h.
    target_link_libraries(parameteroptimization PUBLIC nlopt)
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

        # Whether a system NLopt was built with the Luksan solvers cannot be detected:
        # NLOPT_LUKSAN is PRIVATE to NLopt's own library target, nlopt.h declares
        # NLOPT_LD_LBFGS unconditionally regardless of what was compiled, nlopt_config.h
        # is not installed, and NLoptConfig.cmake does not record it.
        #
        # We therefore assume it is present — every mainstream distribution (Homebrew,
        # Debian, Fedora, conda-forge, vcpkg) uses NLopt's upstream default of ON — and
        # let the user assert otherwise. If the assumption is wrong, requesting L-BFGS
        # now fails loudly rather than silently returning an unoptimized result.
        set(HELIOS_SYSTEM_NLOPT_HAS_LUKSAN ON CACHE BOOL
            "System NLopt was built with the Luksan solvers (required for L-BFGS). Set OFF if yours was built with -DNLOPT_LUKSAN=OFF.")

        if(HELIOS_NLOPT_LUKSAN AND HELIOS_SYSTEM_NLOPT_HAS_LUKSAN)
            target_compile_definitions(parameteroptimization PUBLIC HELIOS_HAVE_LBFGS)
        else()
            message(STATUS "[ParameterOptimization] L-BFGS disabled (Luksan solvers unavailable) — use Adam or BOBYQA")
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
