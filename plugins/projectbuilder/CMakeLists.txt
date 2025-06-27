# Helios standard CMakeLists.txt file version 1.9
cmake_minimum_required(VERSION 3.15)

project(helios)

set(PLUGIN_INCLUDE_DIRS "")
set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/ProjectBuilder.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/InitializeSimulation.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/selfTest.cpp")

# Conditional Plugins
file(GLOB PLUGIN_DIRS LIST_DIRECTORIES true ${CMAKE_SOURCE_DIR}/plugins/*)

if("boundarylayerconductance" IN_LIST PLUGINS)
    add_definitions(-DENABLE_BOUNDARYLAYERCONDUCTANCEMODEL)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../boundarylayerconductance/include")
    list(APPEND PLUGIN_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/boundarylayerconductance/include
    )
endif()


if("energybalance" IN_LIST PLUGINS)
    add_definitions(-DENABLE_ENERGYBALANCEMODEL)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../energybalance/include")
    list(APPEND PLUGIN_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/energybalance/include)
endif()

if("energybalance" IN_LIST PLUGINS AND "boundarylayerconductance" IN_LIST PLUGINS)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/InitializeEnergyBalance")
    list(APPEND SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/InitializeEnergyBalance.cpp"
    )
endif()

if("plantarchitecture" IN_LIST PLUGINS)
    add_definitions(-DENABLE_PLANT_ARCHITECTURE)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../plantarchitecture/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/BuildGeometry")
    list(APPEND PLUGIN_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/plantarchitecture/include
    )
    list(APPEND SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/BuildGeometry.cpp"
    )
endif()

if("canopygenerator" IN_LIST PLUGINS)
    add_definitions(-DENABLE_CANOPY_GENERATOR)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../canopygenerator/include")
    list(APPEND PLUGIN_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/canopygenerator/include)
endif()

if("radiation" IN_LIST PLUGINS)
    add_definitions(-DENABLE_RADIATION_MODEL)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../radiation/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/InitializeRadiation")
    list(APPEND PLUGIN_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/radiation/include
    )
    list(APPEND SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/InitializeRadiation.cpp"
    )
endif()

if("solarposition" IN_LIST PLUGINS OR "radiation" IN_LIST PLUGINS) # solarposition needs to be loaded if radiation is loaded
    if (NOT "solarposition" IN_LIST PLUGINS)
        set( TEMP_PLUGIN "solarposition")
        message("-- loading plug-in ${TEMP_PLUGIN}")
        if( ${TEMP_PLUGIN} STREQUAL ${EXECUTABLE_NAME} )
            message( FATAL_ERROR "The executable name cannot be the same as a plugin name. Please rename your executable." )
        endif()
        add_subdirectory( "${BASE_DIRECTORY}/plugins/${TEMP_PLUGIN}" "${PROJECT_BINARY_DIR}/plugins/${TEMP_PLUGIN}" )
        target_link_libraries( ${EXECUTABLE_NAME} ${TEMP_PLUGIN} )
        target_link_libraries( ${TEMP_PLUGIN} helios )
    endif()
    add_definitions(-DENABLE_SOLARPOSITION)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../solarposition/include")
    list(APPEND PLUGIN_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/solarposition/include
    )
endif()

if("visualizer" IN_LIST PLUGINS)
    add_definitions(-DENABLE_HELIOS_VISUALIZER)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/include")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../visualizer/lib/glm")
    list(APPEND PLUGIN_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/visualizer/include
    )
    list(APPEND SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/imgui.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/imgui_draw.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/imgui_widgets.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/imgui_demo.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/imgui_tables.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/backends/imgui_impl_glfw.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/backends/imgui_impl_opengl3.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/misc/cpp/imgui_stdlib.cpp"
    )
    if(UNIX AND NOT APPLE)
        #nfd
        list(APPEND SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/lib/nativefiledialog/src/nfd_common.c"
            # uncomment for GTK
#            "${CMAKE_CURRENT_SOURCE_DIR}/lib/nativefiledialog/src/nfd_gtk.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/lib/nativefiledialog/src/nfd_zenity.c"
        )
    endif(UNIX AND NOT APPLE)
    if(APPLE)
        #nfd
        list(APPEND SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/lib/nativefiledialog/src/nfd_cocoa.m"
            "${CMAKE_CURRENT_SOURCE_DIR}/lib/nativefiledialog/src/nfd_common.c"
        )
    endif(APPLE)
endif()

include_directories(include)
include_directories(../../core/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(projectbuilder STATIC ${SOURCES})

# this target includes all plugin include dirs unconditionally
#target_include_directories(projectbuilder PUBLIC
#    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/boundarylayerconductance/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/energybalance/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/plantarchitecture/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/radiation/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/solarposition/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/visualizer/include
#)

target_include_directories(projectbuilder PUBLIC ${PLUGIN_INCLUDE_DIRS})

#target_compile_definitions(projectbuilder PUBLIC BOUNDARYLAYERCONDUCTANCEMODEL)
#target_compile_definitions(projectbuilder PUBLIC BOUNDARYLAYERCONDUCTANCEMODEL)

# CUDA + OPTIX required for energybalance and radiation
if("radiation" IN_LIST PLUGINS OR "energybalanace" IN_LIST PLUGINS)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        target_link_libraries( projectbuilder ${CUDA_LIBRARIES} )
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
        target_link_libraries( projectbuilder ${OPTIX_PATH}lib64/liboptix.so )
    endif(UNIX AND NOT APPLE)
    if(APPLE)
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../radiation/lib/OptiX/mac64-4.0.2/)
        include_directories(${OPTIX_PATH}include)
        target_link_libraries( projectbuilder ${OPTIX_PATH}lib64/liboptix.dylib )
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
        target_link_libraries( projectbuilder "${OPTIX_PATH}lib64/${OPTIX_LIB}" )
        add_custom_command( TARGET projectbuilder POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}lib64/${OPTIX_LIB}" "${CMAKE_BINARY_DIR}/." )
        add_custom_command( TARGET projectbuilder POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}bin64/${OPTIX_DLL}" "${CMAKE_BINARY_DIR}/." )

    endif(WIN32)
    if(NOT DEFINED OPTIX_PATH)
        message( FATAL_ERROR "ERROR: Could not determine operating system for unknown reason." )
    endif(NOT DEFINED OPTIX_PATH)
    message(STATUS "OptiX include directory: ${OPTIX_PATH}include")
endif()
# CUDA + OPTIX END

if("boundarylayerconductance" IN_LIST PLUGINS)
    target_link_libraries( projectbuilder boundarylayerconductance )
endif()

if("energybalance" IN_LIST PLUGINS)
    target_link_libraries( projectbuilder energybalance )
endif()

if("plantarchitecture" IN_LIST PLUGINS)
    target_link_libraries( projectbuilder plantarchitecture )
endif()

if("radiation" IN_LIST PLUGINS)
    target_link_libraries( projectbuilder radiation )
endif()

if("solarposition" IN_LIST PLUGINS OR "radiation" IN_LIST PLUGINS)
    target_link_libraries( projectbuilder solarposition )
endif()

if("visualizer" IN_LIST PLUGINS)
    target_link_libraries( projectbuilder visualizer )
    #imgui
    include_directories(lib/imgui)
    include_directories(lib/imgui/backends)
    include_directories(lib/imgui/misc/cpp)
    target_include_directories( projectbuilder PUBLIC lib/imgui )
    target_include_directories( projectbuilder PUBLIC lib/imgui/backends )
    target_include_directories( projectbuilder PUBLIC lib/imgui/misc/cpp )
endif()

target_link_libraries( projectbuilder helios )

add_dependencies( projectbuilder helios )

if("visualizer" IN_LIST PLUGINS)
    set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS};${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui;${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/backends;${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui/misc/cpp" PARENT_SCOPE )
    if(UNIX AND NOT APPLE)
        #nfd
        include_directories(lib/nativefiledialog/src)
        include_directories(lib/nativefiledialog/src/include)
        target_include_directories( projectbuilder PUBLIC lib/nativefiledialog/src )
        target_include_directories( projectbuilder PUBLIC lib/nativefiledialog/src/include )
        target_link_libraries(projectbuilder dl)
        # uncomment for GTK
#        find_package(PkgConfig REQUIRED)
#        pkg_check_modules(GTK REQUIRED gtkmm-3.0)
#        include_directories(${GTK_INCLUDE_DIRS})
#        link_directories(${GTK_INCLUDE_DIRS})
#        add_definitions(${GTK_CFLAGS_OTHER})
#        target_include_directories( projectbuilder PUBLIC ${GTK_INCLUDE_DIRS} )
#        target_link_directories( projectbuilder PUBLIC ${GTK_LIBRARY_DIRS} )
#        target_link_libraries( projectbuilder ${GTK_LIBRARIES} )
    endif(UNIX AND NOT APPLE)
    if(APPLE)
        find_library(COCOA_LIBRARY Cocoa REQUIRED)
        if(NOT COCOA_LIBRARY)
            message(FATAL_ERROR "Cocoa library not found.")
        endif()
        target_link_libraries(projectbuilder ${COCOA_LIBRARY})
        add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:-x objective-c++>"
            "$<$<COMPILE_LANGUAGE:C>:-x objective-c++>"
        )
        # list(APPEND CMAKE_C_FLAGS  "-x objective-c++")
        # list(APPEND CMAKE_CXX_FLAGS "-x objective-c++")
        #nfd
        include_directories(lib/nativefiledialog/src)
        include_directories(lib/nativefiledialog/src/include)
        target_include_directories( projectbuilder PUBLIC lib/nativefiledialog/src )
        target_include_directories( projectbuilder PUBLIC lib/nativefiledialog/src/include )
    endif(APPLE)
else()
    set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS};" PARENT_SCOPE )
endif()

file( COPY inputs DESTINATION ${CMAKE_BINARY_DIR}/plugins/projectbuilder/ )
