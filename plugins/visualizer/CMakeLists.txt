cmake_minimum_required(VERSION 3.15)

project(helios)

include_directories(include)
include_directories(../../core/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(visualizer STATIC src/Visualizer.cpp)

#freetype
include_directories("lib/freetype-2.7/include")
add_subdirectory("lib/freetype-2.7")
target_link_libraries( visualizer freetype )

#glew
include_directories("lib/glew-2.2.0/include")
add_subdirectory("lib/glew-2.2.0/build/cmake")
target_link_libraries( visualizer glew_s )
#target_link_libraries( visualizer GLEW_static )

#jpeg (built with the Helios core)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../core/lib/libjpeg-9a")
include_directories("${CMAKE_BINARY_DIR}/lib/libjpeg-9a")
target_link_libraries( visualizer jpeg )
add_dependencies( visualizer jpeg )

#libpng
include_directories("../../core/lib/libpng")
target_link_libraries( visualizer png_static )
add_dependencies( png_static zlibstatic )

#glfw
include_directories("lib/glfw-3.3.2/include")
add_subdirectory("lib/glfw-3.3.2")
target_link_libraries( visualizer glfw )

#glm
include_directories("lib/glm")

#openGL
find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)
	include_directories(${OPENGL_INCLUDE_DIRS})
	target_link_libraries( visualizer ${OPENGL_LIBRARIES} )
endif(OPENGL_FOUND)

#X11      
find_package(X11 REQUIRED)
if(X11_FOUND)
	include_directories(${X11_INCLUDE_DIRS})
	target_link_libraries( visualizer ${X11_LIBRARIES} )
endif(X11_FOUND)

target_link_libraries( visualizer helios )

add_dependencies( visualizer helios )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS};${CMAKE_CURRENT_SOURCE_DIR}/lib/glm" PARENT_SCOPE )

file( COPY shaders DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )
file( COPY textures DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )
file( COPY fonts DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )

