cmake_minimum_required(VERSION 2.5)
project(helios)

include_directories(include)
include_directories(../../core/include)

add_library(weberpenntree STATIC "src/WeberPennTree.cpp" )

file( COPY leaves/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/leaves/ )
file( COPY wood/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/wood/ )
file( COPY xml/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/xml/ )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

