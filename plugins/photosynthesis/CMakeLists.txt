cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( photosynthesis STATIC "src/PhotosynthesisModel.cpp" )

target_include_directories(photosynthesis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )