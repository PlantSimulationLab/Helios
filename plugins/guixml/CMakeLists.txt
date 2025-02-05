# Helios standard CMakeLists.txt file version 1.9
cmake_minimum_required(VERSION 3.15)

project(helios)

include_directories(include)
include_directories(../../core/include)
#TODO: make everything below conditional
include_directories(../boundarylayerconductance/include)
include_directories(../energybalance/include)
include_directories(../plantarchitecture/include)

include_directories(../radiation/include)
#include_directories("${OPTIX_PATH}include")
include_directories(../solarposition/include)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/lib/glm")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(guixml STATIC "src/BuildGeometry.cpp;src/guixml.cpp;src/InitializeEnergyBalance.cpp;src/InitializeRadiation.cpp;src/InitializeSimulation.cpp;")

target_include_directories(guixml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/boundarylayerconductance/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/energybalance/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/plantarchitecture/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/radiation/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/solarposition/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/visualizer/include
)

#RADIATION
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries( guixml ${CUDA_LIBRARIES} )
endif(CUDA_FOUND)
if(UNIX AND NOT APPLE)
    if( OPTIX_VERSION_LEGACY )
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../radiation/lib/OptiX/linux64-5.1.0/)
        message("Using legacy OptiX version 5.1")
    else()
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../radiation/lib/OptiX/linux64-6.5.0/)
        message("Using OptiX version 6.5")
    endif()
    include_directories(${OPTIX_PATH}include)
    add_subdirectory(${OPTIX_PATH} "plugins/radiation")
    target_link_libraries( guixml ${OPTIX_PATH}lib64/liboptix.so )
endif(UNIX AND NOT APPLE)
if(APPLE)
    set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../radiation/lib/OptiX/mac64-4.0.2/)
    include_directories(${OPTIX_PATH}include)
    add_subdirectory(${OPTIX_PATH})
    target_link_libraries( guixml ${OPTIX_PATH}lib64/liboptix.dylib )
endif(APPLE)
if(WIN32)
    if( OPTIX_VERSION_LEGACY )
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../radiation/lib/OptiX/windows64-5.1.1/)
        set(OPTIX_LIB "optix.51.lib")
        set(OPTIX_DLL "optix.51.dll")
        message("Using legacy OptiX version 5.1")
    else()
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../radiation/lib/OptiX/windows64-6.5.0/)
        set(OPTIX_LIB "optix.6.5.0.lib")
        set(OPTIX_DLL "optix.6.5.0.dll")
        message("Using OptiX version 6.5")
    endif()
    include_directories(${OPTIX_PATH}include)
    target_link_libraries( guixml "${OPTIX_PATH}lib64/${OPTIX_LIB}" )
    add_custom_command( TARGET guixml POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}lib64/${OPTIX_LIB}" "${CMAKE_BINARY_DIR}/." )
    add_custom_command( TARGET guixml POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}bin64/${OPTIX_DLL}" "${CMAKE_BINARY_DIR}/." )

endif(WIN32)
if(NOT DEFINED OPTIX_PATH)
    message( FATAL_ERROR "ERROR: Could not determine operating system for unknown reason." )
endif(NOT DEFINED OPTIX_PATH)
message(STATUS "OptiX include directory: ${OPTIX_PATH}include")

# RADIATION END

target_link_libraries( guixml energybalance )
target_link_libraries( guixml plantarchitecture )
target_link_libraries( guixml radiation )
target_link_libraries( guixml solarposition )
target_link_libraries( guixml visualizer )
#imgui
include_directories(lib/imgui)
include_directories(lib/imgui/backends)
include_directories(lib/imgui/misc/cpp)

add_library(imgui STATIC "lib/imgui/imgui.cpp;lib/imgui/imgui_draw.cpp;lib/imgui/imgui_widgets.cpp;lib/imgui/imgui_demo.cpp;lib/imgui/imgui_tables.cpp;lib/imgui/backends/imgui_impl_glfw.cpp;lib/imgui/backends/imgui_impl_opengl3.cpp;lib/imgui/misc/cpp/imgui_stdlib.cpp;")

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS};${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui" PARENT_SCOPE )
