cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( leafoptics STATIC "src/LeafOptics.cpp" "tests/selfTest.cpp" )

target_include_directories(leafoptics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/leafoptics/ )

if(BUILD_TESTS)
    add_executable(leafoptics_tests "tests/TestMain.cpp")
    target_link_libraries(leafoptics_tests PRIVATE leafoptics)
    add_test(NAME leafoptics_tests COMMAND leafoptics_tests)
endif()

