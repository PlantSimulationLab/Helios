cmake_minimum_required(VERSION 3.15)
project(helios)

add_library(irrigation STATIC
        src/IrrigationModel.cpp
)

target_include_directories(irrigation PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/lib/pugixml
)

# for global.h using pugi:: types while pugixml include is
target_compile_options(irrigation PRIVATE -include pugixml.hpp)

add_executable(irrigation_tests
        tests/TestMain.cpp
        tests/selfTest.cpp
)

target_include_directories(irrigation_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/lib/doctest
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/lib/pugixml
)

target_link_libraries(irrigation_tests PRIVATE irrigation)
add_test(NAME irrigation_tests COMMAND irrigation_tests)



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

