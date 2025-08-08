cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( solarposition STATIC "src/SolarPosition.cpp" "tests/selfTest.cpp" )

target_include_directories(solarposition PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

if(BUILD_TESTS)
    add_executable( solarposition_tests "tests/TestMain.cpp" )
    target_link_libraries( solarposition_tests PRIVATE solarposition )
    add_test( NAME solarposition_tests COMMAND solarposition_tests )
endif()