cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(weberpenntree STATIC "src/WeberPennTree.cpp" "tests/selfTest.cpp" )

target_include_directories(weberpenntree PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

file( COPY leaves/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/leaves/ )
file( COPY wood/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/wood/ )
file( COPY xml/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/xml/ )

add_executable(weberpenntree_tests "tests/TestMain.cpp")
target_link_libraries(weberpenntree_tests PRIVATE weberpenntree)
add_test(NAME weberpenntree_tests COMMAND weberpenntree_tests)