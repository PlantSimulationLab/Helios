cmake_minimum_required(VERSION 3.15)

project(helios)

add_library( syntheticannotation STATIC "src/SyntheticAnnotation.cpp" "tests/selfTest.cpp" )

target_include_directories(syntheticannotation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_dependencies( syntheticannotation visualizer )
target_link_libraries( syntheticannotation PUBLIC visualizer )

if(BUILD_TESTS)
    add_executable( syntheticannotation_tests "tests/TestMain.cpp" )
    target_link_libraries( syntheticannotation_tests PRIVATE syntheticannotation )
    add_test( NAME syntheticannotation_tests COMMAND syntheticannotation_tests )
endif()