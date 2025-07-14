cmake_minimum_required(VERSION 3.15)

project(helios)


add_library(planthydraulics STATIC "src/PlantHydraulicsModel.cpp" "src/selfTest.cpp")

target_include_directories(planthydraulics
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
)

target_link_libraries(planthydraulics PRIVATE helios)

set(PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE)