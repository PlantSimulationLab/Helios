cmake_minimum_required(VERSION 3.0)
project(helios)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_library(dummy ${SOURCES})


