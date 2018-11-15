cmake_minimum_required(VERSION 2.5)
project(helios)

SET(CMAKE_CXX_COMPILER_ID "GNU")
if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7 )
    SET(CMAKE_CXX_FLAGS "-g -std=c++0x")
else()
    SET(CMAKE_CXX_FLAGS "-g -std=c++11")	
endif()

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )
include_directories(include)
include_directories(../../core/include)

add_library(weberpenntree STATIC "src/WeberPennTree.cpp" )

file( COPY leaves/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/leaves/ )
file( COPY xml/ DESTINATION ${CMAKE_BINARY_DIR}/plugins/weberpenntree/xml/ )

