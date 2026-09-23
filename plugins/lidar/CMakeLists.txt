cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# Automatically include collision detection plugin if not already in PLUGINS list
if(NOT "collisiondetection" IN_LIST PLUGINS AND NOT TARGET collisiondetection)
    message(STATUS "[LiDAR] Automatically loading collision detection dependency")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../collisiondetection" "${CMAKE_BINARY_DIR}/plugins/collisiondetection_auto")
    target_link_libraries(collisiondetection PUBLIC helios)
endif()

# Automatically include visualizer plugin if not already in PLUGINS list
if(NOT "visualizer" IN_LIST PLUGINS AND NOT TARGET visualizer)
    message(STATUS "[LiDAR] Automatically loading visualizer dependency")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../visualizer" "${CMAKE_BINARY_DIR}/plugins/visualizer_auto")
    target_link_libraries(visualizer PUBLIC helios)
endif()

add_library(lidar STATIC "src/LiDAR.cpp" "src/fileIO.cpp" "tests/selfTest.cpp" "../../core/lib/pugixml/pugixml.cpp")

target_include_directories(lidar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(lidar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/CDT/include)

target_link_libraries(lidar PUBLIC visualizer collisiondetection)
add_dependencies(lidar visualizer collisiondetection)

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )
file( COPY data DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )

if(BUILD_TESTS)
    add_executable(lidar_tests "tests/TestMain.cpp")
    target_link_libraries(lidar_tests PRIVATE lidar)
    add_test(NAME lidar_tests COMMAND lidar_tests)
endif()