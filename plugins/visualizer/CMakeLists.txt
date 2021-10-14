cmake_minimum_required(VERSION 2.5)

project(helios)

include_directories(include)
include_directories(../../core/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(visualizer STATIC src/Visualizer.cpp)

target_link_libraries( visualizer helios )

#freetype
include_directories("lib/freetype-2.7/include")
add_subdirectory("lib/freetype-2.7")
target_link_libraries( visualizer freetype )

#glew
include_directories("lib/glew-2.2.0/include")
add_subdirectory("lib/glew-2.2.0/build/cmake")
target_link_libraries( visualizer glew_s )
#target_link_libraries( visualizer GLEW_static )

#jpeg
include_directories("lib/libjpeg-9a")
include_directories("${CMAKE_BINARY_DIR}/plugins/visualizer/lib/libjpeg-9a")
add_subdirectory("lib/libjpeg-9a")
target_link_libraries( visualizer jpeg )

#libpng
include_directories("../../core/lib/libpng")
target_link_libraries( visualizer png_static )
add_dependencies( png_static zlibstatic )

#glfw
#include_directories("lib/glfw-3.2.1/include")
#add_subdirectory("lib/glfw-3.2.1")
include_directories("lib/glfw-3.3.2/include")
add_subdirectory("lib/glfw-3.3.2")
target_link_libraries( visualizer glfw )
add_custom_command( TARGET visualizer POST_BUILD_COMMAND CMAKE_COMMAND -E copy "${CMAKE_BINARY_DIR}/lib/glfw3.dll" "${CMAKE_BINARY_DIR}/" )

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

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS};${CMAKE_CURRENT_SOURCE_DIR}/lib/glm" PARENT_SCOPE )

file( COPY shaders DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )
file( COPY textures DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )
file( COPY fonts DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )

