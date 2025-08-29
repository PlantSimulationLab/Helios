cmake_minimum_required(VERSION 3.10)
project(helios)

include_directories(include)
include_directories(../../core/include)
#include_directories(/usr/include/eigen3)

add_library(irrigation STATIC
        src/IrrigationModel.cpp
        tests/selfTest.cpp
)

target_include_directories(irrigation PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Test executable include all test sources
add_executable(irrigation_tests
        tests/TestMain.cpp
)

target_link_libraries(irrigation_tests PRIVATE irrigation)

# run from the source tests directory
#enable_testing()
#add_test(NAME irrigation_tests
#        COMMAND irrigation_tests
 #       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
#)


#add_library( irrigation STATIC "src/IrrigationModel.cpp"
 #       tests/selfTest.cpp
 #       tests/TestMain.cpp)

#target_include_directories(irrigation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

#add_executable( irrigation_tests "tests/TestMain.cpp" )
#target_link_libraries( irrigation_tests PRIVATE irrigation )
#add_test( NAME irrigation_tests COMMAND irrigation_tests )

#set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

