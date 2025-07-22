cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( boundarylayerconductance STATIC "src/BoundaryLayerConductanceModel.cpp" "tests/selfTest.cpp" )

target_include_directories(boundarylayerconductance PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

add_executable( boundarylayerconductance_tests "tests/TestMain.cpp" )
target_link_libraries( boundarylayerconductance_tests PRIVATE boundarylayerconductance )
add_test( NAME boundarylayerconductance_tests COMMAND boundarylayerconductance_tests )