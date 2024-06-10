cmake_minimum_required(VERSION 3.15)

project(helios)

include_directories(include)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

add_library( helios STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/Context.cpp;${CMAKE_CURRENT_SOURCE_DIR}/src/Context_fileIO.cpp;${CMAKE_CURRENT_SOURCE_DIR}/src/Context_data.cpp;${CMAKE_CURRENT_SOURCE_DIR}/src/global.cpp;${CMAKE_CURRENT_SOURCE_DIR}/src/selfTest.cpp;${CMAKE_CURRENT_SOURCE_DIR}/src/pugixml.cpp )

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib")
add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib" "${CMAKE_BINARY_DIR}/lib/zlib" )
target_link_libraries( zlibstatic )

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/libpng")
add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/lib/libpng" "${CMAKE_BINARY_DIR}/lib/libpng" )
add_dependencies( png_static zlibstatic )

target_link_libraries( helios png_static )

include_directories("${CMAKE_BINARY_DIR}/lib/libjpeg-9a;${CMAKE_CURRENT_SOURCE_DIR}/lib/libjpeg-9a")
add_subdirectory("lib/libjpeg-9a" "${CMAKE_BINARY_DIR}/lib/libjpeg-9a" )
target_link_libraries( helios jpeg )

file( COPY "${CMAKE_CURRENT_SOURCE_DIR}/lib/images" DESTINATION "${CMAKE_BINARY_DIR}/lib/" )
file( COPY "${CMAKE_CURRENT_SOURCE_DIR}/lib/models" DESTINATION "${CMAKE_BINARY_DIR}/lib/" )
file( COPY "${CMAKE_CURRENT_SOURCE_DIR}/lib/testdata" DESTINATION "${CMAKE_BINARY_DIR}/lib/" )

set( PLUGIN_INCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include; PARENT_SCOPE )
