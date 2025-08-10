cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( photosynthesis STATIC "src/PhotosynthesisModel.cpp" "tests/selfTest.cpp" )

target_include_directories(photosynthesis PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

if(BUILD_TESTS)
    add_executable(photosynthesis_tests "tests/TestMain.cpp")
    target_link_libraries(photosynthesis_tests PRIVATE photosynthesis)
    add_test(NAME photosynthesis_tests COMMAND photosynthesis_tests)
endif()