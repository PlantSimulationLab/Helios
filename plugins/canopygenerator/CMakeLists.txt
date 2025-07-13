cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( canopygenerator STATIC "src/CanopyGenerator.cpp;src/grapevine.cpp;src/whitespruce.cpp;src/tomato.cpp;src/strawberry.cpp;src/walnut.cpp;src/sorghum.cpp;src/bean.cpp")

target_include_directories(canopygenerator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
)

target_link_libraries(canopygenerator PRIVATE helios)

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file( COPY textures DESTINATION ${CMAKE_BINARY_DIR}/plugins/canopygenerator/ )
