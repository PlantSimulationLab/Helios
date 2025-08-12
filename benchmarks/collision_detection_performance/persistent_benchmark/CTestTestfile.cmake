# CMake generated Testfile for 
# Source directory: /home/bnbailey/CLionProjects/Helios/benchmarks/collision_detection_performance
# Build directory: /home/bnbailey/CLionProjects/Helios/benchmarks/collision_detection_performance/persistent_benchmark
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(Test0 "/home/bnbailey/CLionProjects/Helios/benchmarks/collision_detection_performance/persistent_benchmark/collision_detection_performance" "0")
set_tests_properties(Test0 PROPERTIES  _BACKTRACE_TRIPLES "/home/bnbailey/CLionProjects/Helios/core/CMake_project.cmake;199;add_test;/home/bnbailey/CLionProjects/Helios/core/CMake_project.cmake;0;;/home/bnbailey/CLionProjects/Helios/benchmarks/collision_detection_performance/CMakeLists.txt;10;include;/home/bnbailey/CLionProjects/Helios/benchmarks/collision_detection_performance/CMakeLists.txt;0;")
subdirs("lib")
subdirs("plugins/collisiondetection")
