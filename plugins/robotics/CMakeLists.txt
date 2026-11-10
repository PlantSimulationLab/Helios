cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT "physicsengine" IN_LIST PLUGINS AND NOT TARGET physicsengine)
    message(STATUS "[Robotics] Automatically loading physics engine dependency")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../physicsengine" "${CMAKE_BINARY_DIR}/plugins/physicsengine_auto")
    target_link_libraries(physicsengine PUBLIC helios)
endif()

add_library(robotics STATIC "src/Robot.cpp" "src/ToolVelocityAction.cpp" "src/VacuumCup.cpp" "src/ProtectiveStop.cpp" "src/ControlLoop.cpp" "src/ParallelRobotSimulations.cpp" "src/JointActions.cpp" "tests/selfTest.cpp")

target_include_directories(robotics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(robotics PUBLIC helios physicsengine)

# The camera draws with the Visualizer, so it is only built when the visualizer plug-in is in use, and the rest of the plug-in needs no OpenGL
if("visualizer" IN_LIST PLUGINS)
    target_sources(robotics PRIVATE "src/RobotCamera.cpp")
    target_link_libraries(robotics PUBLIC visualizer)
    target_compile_definitions(robotics PUBLIC HELIOS_ROBOTICS_CAMERA)
endif()

if(BUILD_TESTS)
    # The physics engine loads the plant adapter whenever tests are built
    target_compile_definitions(robotics PRIVATE HELIOS_ROBOTICS_PLANT_TESTS)
    add_executable(robotics_tests "tests/TestMain.cpp")
    target_link_libraries(robotics_tests PRIVATE robotics)
    add_test(NAME robotics_tests COMMAND robotics_tests)
endif()
