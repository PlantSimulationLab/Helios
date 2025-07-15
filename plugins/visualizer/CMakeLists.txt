cmake_minimum_required(VERSION 3.15)

project(helios)

include_directories(include)
include_directories(../../core/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

set( VISUALIZER_SOURCES
	"src/Visualizer.cpp"
	"src/GeometryHandler.cpp"
	"src/selfTest.cpp"
)

add_library(visualizer STATIC ${VISUALIZER_SOURCES})

#External libraries
set(SAVED_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

#freetype
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/freetype-2.7/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/freetype-2.7")
target_link_libraries( visualizer freetype )

#glew
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/glew-2.2.0/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/glew-2.2.0/build/cmake")
target_link_libraries( visualizer glew_s )
#target_link_libraries( visualizer GLEW_static )

#jpeg (built with the Helios core)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../core/lib/libjpeg-9a")
include_directories("${CMAKE_BINARY_DIR}/lib/libjpeg-9a")
target_link_libraries( visualizer jpeg )
add_dependencies( visualizer jpeg )

#libpng
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../core/lib/libpng")
target_link_libraries( visualizer png_static )
add_dependencies( png_static zlibstatic )

#glfw
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/glfw-3.3.2/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/glfw-3.3.2")
target_link_libraries( visualizer glfw )
target_compile_options(glfw
		PRIVATE
		# for clang, AppleClang & gcc
		$<$<AND:$<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>,$<COMPILE_LANGUAGE:OBJC>>:-Wno-deprecated-declarations>
		$<$<AND:$<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>,$<COMPILE_LANGUAGE:C>>:-Wno-deprecated-declarations>
		# for MSVC
		$<$<C_COMPILER_ID:MSVC>:/wd4996>
)


#glm
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/glm")

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

set(CMAKE_MESSAGE_LOG_LEVEL ${SAVED_LOG_LEVEL})

target_link_libraries( visualizer helios )

add_dependencies( visualizer helios )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS};${CMAKE_CURRENT_SOURCE_DIR}/lib/glm" PARENT_SCOPE )

file( COPY fonts DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )

file(GLOB_RECURSE SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag")
file(GLOB_RECURSE TEXTURE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/textures/*")

add_custom_command(
		OUTPUT "${CMAKE_BINARY_DIR}/plugins/visualizer/shader_assets.stamp"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/visualizer/shaders"
		COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders"
		"${CMAKE_BINARY_DIR}/plugins/visualizer/shaders"
		COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/plugins/visualizer/shader_assets.stamp"
		DEPENDS ${SHADER_FILES}
)
add_custom_target(copy_shaders ALL DEPENDS "${CMAKE_BINARY_DIR}/plugins/visualizer/shader_assets.stamp")

add_custom_command(
		OUTPUT "${CMAKE_BINARY_DIR}/plugins/visualizer/texture_assets.stamp"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/visualizer/textures"
		COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/textures"
		"${CMAKE_BINARY_DIR}/plugins/visualizer/textures"
		COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/plugins/visualizer/texture_assets.stamp"
		DEPENDS ${TEXTURE_FILES}
)
add_custom_target(copy_textures ALL DEPENDS "${CMAKE_BINARY_DIR}/plugins/visualizer/texture_assets.stamp")

add_dependencies(visualizer copy_shaders copy_textures)

add_executable( visualizer_tests "src/selfTest.cpp" )
target_link_libraries( visualizer_tests PRIVATE visualizer )
add_test( NAME visualizer_tests COMMAND visualizer_tests )