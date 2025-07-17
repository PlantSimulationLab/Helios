cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(planthydraulics STATIC "src/PlantHydraulicsModel.cpp" "src/selfTest.cpp")

target_include_directories(planthydraulics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )