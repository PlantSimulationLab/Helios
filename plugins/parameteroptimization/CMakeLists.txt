cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(parameteroptimization STATIC "src/ParameterOptimization.cpp" "tests/selfTest.cpp")

target_include_directories(parameteroptimization PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

if(BUILD_TESTS)
    add_executable( parameteroptimization_tests "tests/TestMain.cpp" )
    target_link_libraries( parameteroptimization_tests PRIVATE parameteroptimization )
    add_test( NAME parameteroptimization_tests COMMAND parameteroptimization_tests )
endif()