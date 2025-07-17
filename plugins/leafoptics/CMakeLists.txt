cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( leafoptics STATIC "src/LeafOptics.cpp" )

target_include_directories(leafoptics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )

file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/leafoptics/ )

