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

add_library(lidar STATIC "src/LiDAR.cpp" "src/fileIO.cpp" "tests/selfTest.cpp" "../../core/lib/pugixml/pugixml.cpp" "lib/s_hull_pro/s_hull_pro.cpp")

target_include_directories(lidar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(lidar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/s_hull_pro)

target_link_libraries(lidar PUBLIC visualizer collisiondetection)
add_dependencies(lidar visualizer collisiondetection)

#add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip" )
#include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip/include" )

#set( WITH_LASZIP FALSE )
#set( WITH_UTILITIES FALSE )
#set( WITH_TESTS FALSE )
#set( WITH_GEOTIFF FALSE )
#if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
#     set( LASZIP_LIBRARY "${CMAKE_BINARY_DIR}/plugins/lidar/lib/laszip/bin/Debug/liblaszip.so" )
#else()
#     set( LASZIP_LIBRARY "${CMAKE_BINARY_DIR}/plugins/lidar/lib/laszip/bin/Release/liblaszip.so" )
#endif()
#set( LASZIP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/laszip/include/laszip" )
#add_subdirectory( "lib/libLAS" )
#include_directories( "lib/libLAS/include/liblas" )
#if( WIN32 )
#     set( LIBLAS_LIB_NAME "liblas" )
#else()
#     set( LIBLAS_LIB_NAME "las" )
#endif()
#target_link_libraries( lidar "${LIBLAS_LIB_NAME}" )
#add_dependencies( "${LIBLAS_LIB_NAME}" laszip )
#add_dependencies( lidar "${LIBLAS_LIB_NAME}" )

file( COPY xml DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )
file( COPY data DESTINATION ${CMAKE_BINARY_DIR}/plugins/lidar/ )

if(BUILD_TESTS)
    add_executable(lidar_tests "tests/TestMain.cpp")
    target_link_libraries(lidar_tests PRIVATE lidar)
    add_test(NAME lidar_tests COMMAND lidar_tests)
endif()