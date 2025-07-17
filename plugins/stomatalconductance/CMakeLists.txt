cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( stomatalconductance STATIC "src/StomatalConductanceModel.cpp" )

target_include_directories(stomatalconductance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )