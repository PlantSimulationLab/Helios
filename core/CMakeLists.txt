cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# Defining Helios Context source files to be built
set(HELIOS_SOURCES
        src/Context.cpp
        src/Context_fileIO.cpp
        src/Context_data.cpp
        src/global.cpp
        src/selfTest.cpp
        src/pugixml.cpp
)
add_library(helios STATIC ${HELIOS_SOURCES})

# Defining Helios Context include files
target_include_directories(helios
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# External libraries
set(SAVED_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/zlib"     "${CMAKE_BINARY_DIR}/lib/zlib")

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/lib/libpng")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/libpng"   "${CMAKE_BINARY_DIR}/lib/libpng")
add_dependencies( png_static zlibstatic )

include_directories("${CMAKE_BINARY_DIR}/lib/libjpeg-9a" "${CMAKE_CURRENT_SOURCE_DIR}/lib/libjpeg-9a")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/libjpeg-9a" "${CMAKE_BINARY_DIR}/lib/libjpeg-9a")

target_link_libraries( helios PRIVATE png_static  jpeg ) #note that zlib is already linked by libpng

set(CMAKE_MESSAGE_LOG_LEVEL ${SAVED_LOG_LEVEL})

file( COPY "${CMAKE_CURRENT_SOURCE_DIR}/lib/images" DESTINATION "${CMAKE_BINARY_DIR}/lib/" )
file( COPY "${CMAKE_CURRENT_SOURCE_DIR}/lib/models" DESTINATION "${CMAKE_BINARY_DIR}/lib/" )
file( COPY "${CMAKE_CURRENT_SOURCE_DIR}/lib/testdata" DESTINATION "${CMAKE_BINARY_DIR}/lib/" )

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;" PARENT_SCOPE )
