cmake_minimum_required(VERSION 3.15)

project(helios)  

include_directories(include)
include_directories(../../core/include)

add_library( plantarchitecture STATIC "src/PlantArchitecture.cpp;src/PlantLibrary.cpp;src/Assets.cpp;src/InputOutput.cpp;src/CarbohydrateModel.cpp;src/Hungarian.cpp;src/selfTest.cpp")

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file(GLOB_RECURSE OBJ_ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/obj/*")
add_custom_target(copy_obj_assets ALL COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/obj" COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OBJ_ASSET_FILES} "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/obj/" DEPENDS ${OBJ_ASSET_FILES})
add_dependencies(plantarchitecture copy_obj_assets)
file(GLOB_RECURSE TEXTURE_ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/*")
add_custom_target(copy_texture_assets ALL COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/textures" COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEXTURE_ASSET_FILES} "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/textures/" DEPENDS ${TEXTURE_ASSET_FILES})
add_dependencies(plantarchitecture copy_texture_assets)