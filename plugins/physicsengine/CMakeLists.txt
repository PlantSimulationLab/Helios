cmake_minimum_required(VERSION 3.15)

project(helios)

# MuJoCo version this plug-in is written against. Its C API changes between minor releases, so it is pinned.
set(HELIOS_MUJOCO_VERSION "3.16.0")

option(HELIOS_MUJOCO_FETCH "Download and build MuJoCo from source when an installed MuJoCo ${HELIOS_MUJOCO_VERSION} is not found" ON)

# The engine itself knows nothing about plants. The plant adapter is compiled when the plant architecture plug-in is
# in use, and the self-test needs it to exercise the adapter.
set(HELIOS_PHYSICS_ENGINE_PLANTS OFF)
if("plantarchitecture" IN_LIST PLUGINS OR TARGET plantarchitecture OR BUILD_TESTS)
    set(HELIOS_PHYSICS_ENGINE_PLANTS ON)
endif()
if(HELIOS_PHYSICS_ENGINE_PLANTS AND NOT "plantarchitecture" IN_LIST PLUGINS AND NOT TARGET plantarchitecture)
    message(STATUS "[PhysicsEngine] Automatically loading plant architecture dependency for the self-test")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../plantarchitecture" "${CMAKE_BINARY_DIR}/plugins/plantarchitecture_auto")
    target_link_libraries(plantarchitecture PUBLIC helios)
endif()

# Use an installed MuJoCo when one is given (-Dmujoco_DIR=<prefix>/lib/cmake/mujoco, or CMAKE_PREFIX_PATH)
find_package(mujoco ${HELIOS_MUJOCO_VERSION} EXACT CONFIG QUIET)

if(mujoco_FOUND)
    message(STATUS "[PhysicsEngine] Using installed MuJoCo ${mujoco_VERSION} from ${mujoco_DIR}")
elseif(HELIOS_MUJOCO_FETCH)
    message(STATUS "[PhysicsEngine] Installed MuJoCo ${HELIOS_MUJOCO_VERSION} not found - downloading and building it from source")
    include(FetchContent)
    set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
    set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MUJOCO_TEST_PYTHON_UTIL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            mujoco
            URL "https://github.com/google-deepmind/mujoco/archive/refs/tags/${HELIOS_MUJOCO_VERSION}.tar.gz"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(mujoco)
else()
    message(FATAL_ERROR "[PhysicsEngine] MuJoCo ${HELIOS_MUJOCO_VERSION} was not found. Either install it and pass -Dmujoco_DIR=<install prefix>/lib/cmake/mujoco, "
            "or configure with -DHELIOS_MUJOCO_FETCH=ON to download and build it automatically.")
endif()

find_package(Threads REQUIRED)

add_library(physicsengine STATIC "src/PhysicsEngine.cpp" "src/ContactSensor.cpp" "tests/selfTest.cpp")

target_include_directories(physicsengine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# MuJoCo is public: code built on the engine uses the MuJoCo API directly on PhysicsEngine::getMuJoCoModel() and getMuJoCoData()
target_link_libraries(physicsengine PUBLIC helios mujoco::mujoco PRIVATE Threads::Threads)

if(HELIOS_PHYSICS_ENGINE_PLANTS)
    target_sources(physicsengine PRIVATE "src/PlantPhysics.cpp")
    target_link_libraries(physicsengine PUBLIC plantarchitecture)
    target_compile_definitions(physicsengine PRIVATE HELIOS_PHYSICS_ENGINE_PLANTS)
endif()

if(BUILD_TESTS)
    add_executable(physicsengine_tests "tests/TestMain.cpp")
    target_link_libraries(physicsengine_tests PRIVATE physicsengine)
    add_test(NAME physicsengine_tests COMMAND physicsengine_tests)
endif()
