cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(planthydraulics STATIC "src/PlantHydraulicsModel.cpp" "tests/selfTest.cpp")

target_include_directories(planthydraulics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

add_executable( planthydraulics_tests "tests/TestMain.cpp" )
target_link_libraries( planthydraulics_tests PRIVATE planthydraulics )
add_test( NAME planthydraulics_tests COMMAND planthydraulics_tests )