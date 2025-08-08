cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( stomatalconductance STATIC "src/StomatalConductanceModel.cpp" "tests/selfTest.cpp" )

target_include_directories(stomatalconductance PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

add_executable(stomatalconductance_tests "tests/TestMain.cpp")
target_link_libraries(stomatalconductance_tests PRIVATE stomatalconductance)
add_test(NAME stomatalconductance_tests COMMAND stomatalconductance_tests)