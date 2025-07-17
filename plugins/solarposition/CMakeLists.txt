cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( solarposition STATIC "src/SolarPosition.cpp" "src/selfTest.cpp" )

target_include_directories(solarposition PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )