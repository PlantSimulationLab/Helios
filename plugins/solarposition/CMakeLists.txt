cmake_minimum_required(VERSION 3.15)

project(helios)  

add_library( solarposition STATIC
    "src/SolarPosition.cpp"
    "src/PragueSkyModel.cpp"
    "src/PragueSkyModelInterface.cpp"
    "tests/selfTest.cpp" )

target_include_directories(solarposition PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )

if(BUILD_TESTS)
    add_executable( solarposition_tests "tests/TestMain.cpp" )
    target_link_libraries( solarposition_tests PRIVATE solarposition )
    add_test( NAME solarposition_tests COMMAND solarposition_tests )
endif()

# Copy SSolar-GOA spectral data files to build directory
file( COPY assets/ssolar_goa DESTINATION ${CMAKE_BINARY_DIR}/plugins/solarposition/ )

# Copy Prague Sky Model data files to build directory
file( COPY lib/prague_sky_model DESTINATION ${CMAKE_BINARY_DIR}/plugins/solarposition/lib/ )