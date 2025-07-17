cmake_minimum_required(VERSION 3.15)

project(helios)

add_library( plantarchitecture STATIC "src/PlantArchitecture.cpp;src/PlantLibrary.cpp;src/Assets.cpp;src/InputOutput.cpp;src/CarbohydrateModel.cpp;src/Hungarian.cpp;src/selfTest.cpp")

target_include_directories(plantarchitecture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

file(GLOB_RECURSE OBJ_ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/obj/*")
add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/obj_assets.stamp"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/obj"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/obj"
        "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/obj"
        COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/obj_assets.stamp"
        DEPENDS ${OBJ_ASSET_FILES}
)
add_custom_target(copy_obj_assets DEPENDS "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/obj_assets.stamp")
add_dependencies(plantarchitecture copy_obj_assets)
file(GLOB_RECURSE TEXTURE_ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures/*")
add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/texture_assets.stamp"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/textures"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/textures"
        "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/assets/textures"
        COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/texture_assets.stamp"
        DEPENDS ${TEXTURE_ASSET_FILES}
        COMMENT "Copy texture assets"
)
add_custom_target(copy_texture_assets DEPENDS "${CMAKE_BINARY_DIR}/plugins/plantarchitecture/texture_assets.stamp")
add_dependencies(plantarchitecture copy_texture_assets)