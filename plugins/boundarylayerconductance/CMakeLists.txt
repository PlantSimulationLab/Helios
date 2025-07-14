cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( boundarylayerconductance STATIC "src/BoundaryLayerConductanceModel.cpp" )

target_include_directories(boundarylayerconductance
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
)

target_link_libraries(boundarylayerconductance PRIVATE helios)

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

