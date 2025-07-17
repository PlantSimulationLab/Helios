cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( boundarylayerconductance STATIC "src/BoundaryLayerConductanceModel.cpp" )

target_include_directories(boundarylayerconductance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )