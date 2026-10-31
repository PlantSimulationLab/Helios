cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT "physicsengine" IN_LIST PLUGINS AND NOT TARGET physicsengine)
    message(STATUS "[Robotics] Automatically loading physics engine dependency")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../physicsengine" "${CMAKE_BINARY_DIR}/plugins/physicsengine_auto")
    target_link_libraries(physicsengine PUBLIC helios)
endif()

add_library(robotics STATIC "src/Robot.cpp" "tests/selfTest.cpp")

target_include_directories(robotics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(robotics PUBLIC helios physicsengine)

if(BUILD_TESTS)
    # The physics engine loads the plant adapter whenever tests are built
    target_compile_definitions(robotics PRIVATE HELIOS_ROBOTICS_PLANT_TESTS)
    add_executable(robotics_tests "tests/TestMain.cpp")
    target_link_libraries(robotics_tests PRIVATE robotics)
    add_test(NAME robotics_tests COMMAND robotics_tests)
endif()
