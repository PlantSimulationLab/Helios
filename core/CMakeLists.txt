cmake_minimum_required(VERSION 2.4)

project(helios)

SET(CMAKE_CXX_COMPILER_ID "GNU")
if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7 )
    SET(CMAKE_CXX_FLAGS "-g -std=c++0x")
else()
    SET(CMAKE_CXX_FLAGS "-g -std=c++11")	
endif()
SET(CMAKE_C_FLAGS "-g" )

SET( PLUGIN_INCLUDE_PATHS "include" )

include_directories(include)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

file( MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

add_library( helios STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/Context.cpp;${CMAKE_CURRENT_SOURCE_DIR}/src/global.cpp;${CMAKE_CURRENT_SOURCE_DIR}/src/pugixml.cpp )

include_directories("lib/zlib")
add_subdirectory("lib/zlib" "${CMAKE_BINARY_DIR}/lib" )
target_link_libraries( zlibstatic )

include_directories("lib/libpng")
add_subdirectory("lib/libpng" "${CMAKE_BINARY_DIR}/lib" )
add_dependencies( png_static zlibstatic )
target_link_libraries( helios png_static )

file( COPY lib/images DESTINATION ${CMAKE_BINARY_DIR}/lib/ )

set( PLUGIN_INCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS} PARENT_SCOPE )


