cmake_minimum_required(VERSION 3.15)

project(helios)

# Automatically include collision detection plugin if not already in PLUGINS list
if(NOT "collisiondetection" IN_LIST PLUGINS AND NOT TARGET collisiondetection)
    message(STATUS "[Plant Architecture] Automatically loading collision detection dependency")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../collisiondetection" "${CMAKE_BINARY_DIR}/plugins/collisiondetection_auto")
    target_link_libraries(collisiondetection PUBLIC helios)
endif()

add_library( plantarchitecture STATIC "src/PlantArchitecture.cpp"
        "src/PlantLibrary.cpp"
        "src/Assets.cpp"
        "src/InputOutput.cpp"
        "src/CarbohydrateModel.cpp"
        "src/Hungarian.cpp"
        "tests/selfTest.cpp"
)

target_include_directories(plantarchitecture PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

target_link_libraries(plantarchitecture PUBLIC collisiondetection)
add_dependencies(plantarchitecture collisiondetection)

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

if(BUILD_TESTS)
    add_executable(plantarchitecture_tests "tests/TestMain.cpp")
    target_link_libraries(plantarchitecture_tests PRIVATE plantarchitecture)
    add_test(NAME plantarchitecture_tests COMMAND plantarchitecture_tests)
endif()