cmake_minimum_required(VERSION 3.15)

project(helios)  

include_directories(include)
include_directories(../../core/include)

add_library( plantarchitecture STATIC "src/PlantArchitecture.cpp;src/PlantLibrary.cpp;src/Assets.cpp;src/InputOutput.cpp;src/CarbohydrateModel.cpp;src/Hungarian.cpp;src/selfTest.cpp")

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

file(GLOB_RECURSE OBJ_ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/obj/*")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/assets/obj"  DESTINATION "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets")
file(GLOB_RECURSE TEXTURE_ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/*")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures" DESTINATION "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets")