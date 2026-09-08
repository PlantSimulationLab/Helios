cmake_minimum_required(VERSION 3.15)

project(helios)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

set( VISUALIZER_SOURCES
	#"src/Visualizer.cpp"
	"src/VisualizerCore.cpp"
	"src/VisualizerGeometry.cpp"
	"src/VisualizerRendering.cpp"
	"src/VisualizerContext.cpp"
	"src/GeometryHandler.cpp"
	"tests/selfTest.cpp"
)

add_library(visualizer STATIC ${VISUALIZER_SOURCES})

target_include_directories(visualizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/lib/glm" )

#External libraries
set(SAVED_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

#freetype
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/freetype-2.7/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/freetype-2.7")
target_link_libraries( visualizer PRIVATE freetype )

#glew
# PUBLIC rather than directory-scoped: Visualizer.h includes <GL/glew.h> for the OpenGL 1.5+
# types it declares (GLsizeiptr), and that header is consumed by the lidar, aeriallidar,
# syntheticannotation and projectbuilder plugins and by the samples, none of which are in this
# directory's scope. The library itself stays PRIVATE -- only the declarations propagate.
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/glew-2.2.0/include")
target_include_directories( visualizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lib/glew-2.2.0/include" )
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/glew-2.2.0/build/cmake")
target_link_libraries( visualizer PRIVATE glew_s )
# Disable OpenGL deprecation warnings for GLEW
target_compile_options(glew_s
		PRIVATE
		# for clang, AppleClang & gcc
		$<$<AND:$<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>,$<COMPILE_LANGUAGE:C>>:-Wno-deprecated-declarations>
		$<$<AND:$<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>,$<COMPILE_LANGUAGE:C>>:-DGL_SILENCE_DEPRECATION>
		# for MSVC
		$<$<C_COMPILER_ID:MSVC>:/wd4996>
)

#glfw
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/glfw-3.3.2/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/glfw-3.3.2")
target_link_libraries( visualizer PUBLIC glfw )
target_compile_options(glfw
		PRIVATE
		# for clang, AppleClang & gcc
		$<$<AND:$<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>,$<COMPILE_LANGUAGE:OBJC>>:-Wno-deprecated-declarations>
		$<$<AND:$<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>,$<COMPILE_LANGUAGE:C>>:-Wno-deprecated-declarations>
		# for MSVC
		$<$<C_COMPILER_ID:MSVC>:/wd4996>
)

#openGL
find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)
	include_directories(${OPENGL_INCLUDE_DIRS})
	target_link_libraries( visualizer PRIVATE ${OPENGL_LIBRARIES} )
	# Silence OpenGL deprecation warnings on macOS
	if(APPLE)
		target_compile_definitions(visualizer PRIVATE GL_SILENCE_DEPRECATION)
	endif()
endif(OPENGL_FOUND)

#EGL -- optional, Linux only. Provides a headless OpenGL context bound directly to the GPU, which
# is the only way to rasterize on a batch compute node: GLFW's headless mode is a hidden window and
# still requires X11 or Wayland. When EGL is absent the visualizer builds and behaves exactly as
# before, and headless rendering keeps using the GLFW path.
if(UNIX AND NOT APPLE)
	find_path(EGL_INCLUDE_DIR EGL/egl.h)
	find_library(EGL_LIBRARY NAMES EGL)
	if(EGL_INCLUDE_DIR AND EGL_LIBRARY)
		target_include_directories( visualizer PRIVATE ${EGL_INCLUDE_DIR} )
		target_link_libraries( visualizer PRIVATE ${EGL_LIBRARY} )
		target_compile_definitions( visualizer PRIVATE HELIOS_VISUALIZER_EGL )
	endif()
endif()

#X11      
find_package(X11 REQUIRED)
if(X11_FOUND)
	include_directories(${X11_INCLUDE_DIRS})
	target_link_libraries( visualizer PRIVATE ${X11_LIBRARIES} )
endif(X11_FOUND)

set(CMAKE_MESSAGE_LOG_LEVEL ${SAVED_LOG_LEVEL})

add_dependencies( visualizer helios )

file( COPY fonts DESTINATION ${CMAKE_BINARY_DIR}/plugins/visualizer/ )

file(GLOB_RECURSE SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.geom")
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
add_custom_target(copy_shaders DEPENDS "${CMAKE_BINARY_DIR}/plugins/visualizer/shader_assets.stamp")
set_target_properties(copy_shaders PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_custom_command(
		OUTPUT "${CMAKE_BINARY_DIR}/plugins/visualizer/texture_assets.stamp"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/visualizer/textures"
		COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/textures"
		"${CMAKE_BINARY_DIR}/plugins/visualizer/textures"
		COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/plugins/visualizer/texture_assets.stamp"
		DEPENDS ${TEXTURE_FILES}
)
add_custom_target(copy_textures DEPENDS "${CMAKE_BINARY_DIR}/plugins/visualizer/texture_assets.stamp")
set_target_properties(copy_textures PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_dependencies(visualizer copy_shaders copy_textures)

if(BUILD_TESTS)
    add_executable( visualizer_tests "tests/TestMain.cpp" )
    target_link_libraries( visualizer_tests PRIVATE visualizer )
    add_test( NAME visualizer_tests COMMAND visualizer_tests )
endif()