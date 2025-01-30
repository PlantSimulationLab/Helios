# Helios standard CMakeLists.txt file version 1.9
cmake_minimum_required(VERSION 3.15)
project(helios)

#-------- USER INPUTS ---------#

#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file
set( BASE_DIRECTORY "../.." )

#define the name of the executable to be created
set( EXECUTABLE_NAME "CanopyTransportXML" )

#provide name of source file(s) (separate multiple file names with semicolon)
#set( SOURCE_FILES "main.cpp;InitializeSimulation.cpp;BuildGeometry.cpp;InitializeRadiation.cpp;InitializeEnergyBalance.cpp" )

set(SOURCE_FILES
        main.cpp
        InitializeSimulation.cpp
        BuildGeometry.cpp
        InitializeRadiation.cpp
        InitializeEnergyBalance.cpp
        #    tinyfiledialogs.c
        ${BASE_DIRECTORY}/imgui/imgui.cpp
        ${BASE_DIRECTORY}/imgui/imgui_draw.cpp
        ${BASE_DIRECTORY}/imgui/imgui_widgets.cpp
        ${BASE_DIRECTORY}/imgui/imgui_demo.cpp
        ${BASE_DIRECTORY}/imgui/imgui_tables.cpp  # Table features
        ${BASE_DIRECTORY}/imgui/backends/imgui_impl_glfw.cpp
        ${BASE_DIRECTORY}/imgui/backends/imgui_impl_opengl3.cpp
        ${BASE_DIRECTORY}/imgui/misc/cpp/imgui_stdlib.cpp
)

set(IMGUI_DIR "${BASE_DIRECTORY}/imgui")
set(IMGUI_BACKEND_DIR "${IMGUI_DIR}/backends")

set( PLUGINS "visualizer;plantarchitecture;radiation;solarposition;energybalance;boundarylayerconductance;stomatalconductance;photosynthesis" )


#-------- DO NOT MODIFY ---------#
include( "${BASE_DIRECTORY}/core/CMake_project.txt" )
include_directories(${BASE_DIRECTORY}/imgui)
include_directories(${BASE_DIRECTORY}/imgui/backends)

target_include_directories(${EXECUTABLE_NAME} PRIVATE ${IMGUI_DIR} ${IMGUI_BACKEND_DIR})
