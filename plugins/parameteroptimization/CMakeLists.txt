cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(parameteroptimization STATIC "src/ParameterOptimization.cpp" "tests/selfTest.cpp")

target_include_directories(parameteroptimization PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

add_executable( optimization_tests "tests/TestMain.cpp" )
target_link_libraries( optimization_tests PRIVATE parameteroptimization )
add_test( NAME optimization_tests COMMAND optimization_tests )