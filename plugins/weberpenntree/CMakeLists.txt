cmake_minimum_required(VERSION 3.15)

project(helios)

add_library(weberpenntree STATIC "src/WeberPennTree.cpp" "tests/selfTest.cpp" )

target_include_directories(weberpenntree PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

file(GLOB_RECURSE LEAF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/leaves/*")
file(GLOB_RECURSE WOOD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/wood/*")
file(GLOB_RECURSE XML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/xml/*")

add_custom_command(
	OUTPUT "${CMAKE_BINARY_DIR}/plugins/weberpenntree/asset_files.stamp"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/weberpenntree/leaves"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CMAKE_CURRENT_SOURCE_DIR}/leaves"
	"${CMAKE_BINARY_DIR}/plugins/weberpenntree/leaves"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/weberpenntree/wood"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CMAKE_CURRENT_SOURCE_DIR}/wood"
	"${CMAKE_BINARY_DIR}/plugins/weberpenntree/wood"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/plugins/weberpenntree/xml"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CMAKE_CURRENT_SOURCE_DIR}/xml"
	"${CMAKE_BINARY_DIR}/plugins/weberpenntree/xml"
	COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/plugins/weberpenntree/asset_files.stamp"
	DEPENDS ${LEAF_FILES} ${WOOD_FILES} ${XML_FILES}
)
add_custom_target(copy_weberpenntree_assets DEPENDS "${CMAKE_BINARY_DIR}/plugins/weberpenntree/asset_files.stamp")
set_target_properties(copy_weberpenntree_assets PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_dependencies(weberpenntree copy_weberpenntree_assets)

if(BUILD_TESTS)
    add_executable(weberpenntree_tests "tests/TestMain.cpp")
    target_link_libraries(weberpenntree_tests PRIVATE weberpenntree)
    add_test(NAME weberpenntree_tests COMMAND weberpenntree_tests)
endif()