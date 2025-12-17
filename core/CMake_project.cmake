option(ENABLE_OPENMP "Enable building with OpenMP" ON)
option(BUILD_TESTS "Build test executables" OFF)
option(BUILD_BENCHMARKS "Build performance benchmark executables" OFF)


# Set CMake policies to avoid warnings on newer CMake versions
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
if(POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If a build type is not specified, explicitly set it to Debug.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

message(STATUS "[Helios] Build type: ${CMAKE_BUILD_TYPE}")

# -- automatically force a cmake re-configure if the code version was updated --#
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message( STATUS "[Helios] Detected Git commit hash: ${GIT_COMMIT_HASH}" )
    
    # Get current branch name
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE CURRENT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE BRANCH_RESULT
    )
    
    # Only check for updates if we're on master/main branch
    if(BRANCH_RESULT EQUAL 0 AND (CURRENT_BRANCH STREQUAL "master" OR CURRENT_BRANCH STREQUAL "main"))
        execute_process(
            COMMAND ${GIT_EXECUTABLE} fetch --dry-run
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_QUIET
            ERROR_QUIET
            RESULT_VARIABLE FETCH_RESULT
        )
        
        if(FETCH_RESULT EQUAL 0)
            # Check if local branch is behind remote
            execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-list HEAD..origin/${CURRENT_BRANCH} --count
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE COMMITS_BEHIND
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE REV_LIST_RESULT
            )
            
            if(REV_LIST_RESULT EQUAL 0 AND COMMITS_BEHIND AND NOT COMMITS_BEHIND STREQUAL "0")
                message(WARNING "[Helios] Your local ${CURRENT_BRANCH} branch is ${COMMITS_BEHIND} commit(s) behind the remote ${CURRENT_BRANCH} branch. Consider updating with 'git pull origin ${CURRENT_BRANCH}'.")
            endif()
        endif()
    endif()
else()
    set(GIT_COMMIT_HASH "unknown")
endif()

# --- write it to a cache-version file that CMake will include
if(NOT DEFINED HELIOS_PREVIOUS_COMMIT)
    set(HELIOS_PREVIOUS_COMMIT "" CACHE STRING "Last configured Helios Git commit")
endif()
message( STATUS "[Helios] Last configured Helios Git commit hash: ${HELIOS_PREVIOUS_COMMIT}" )

if(NOT HELIOS_PREVIOUS_COMMIT STREQUAL GIT_COMMIT_HASH)
  message(STATUS "[Helios] Git commit version change detected, automatically re-configuring...")
  
  # Clean CUDA object files to prevent linking issues with stale objects
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    message(STATUS "[Helios] Cleaning CUDA object files due to version change...")
    file(GLOB_RECURSE CUDA_OBJECTS 
      "${CMAKE_BINARY_DIR}/**/*.cu.o"
      "${CMAKE_BINARY_DIR}/**/*.cudafe*"
      "${CMAKE_BINARY_DIR}/**/cmake_device_link.o"
    )
    if(CUDA_OBJECTS)
      file(REMOVE ${CUDA_OBJECTS})
      message(STATUS "[Helios] Removed ${CMAKE_CURRENT_LIST_SIZE} CUDA object files")
    endif()
  endif()
  
  # update cache for next time
  set(HELIOS_PREVIOUS_COMMIT "${GIT_COMMIT_HASH}" CACHE STRING "Last configured Helios Git commit" FORCE)
endif()

# ——— push the current hash into every compile command ———
add_compile_definitions(HELIOS_GIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
configure_file( "${BASE_DIRECTORY}/core/lib/detect_GPU_compute.cmake" "${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake" COPYONLY )

if ( WIN32 )
    string(REGEX REPLACE "/MD*" "/MT" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "/MD*" "/MT" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    
    # Ensure Release builds have optimization flags for C++ code
    set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG /W1")
    set(CMAKE_C_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG /W1")
    
    cmake_policy(SET CMP0091 NEW)
    set( CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}" )
        set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/lib" )
        set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/lib" )
    endforeach(OUTPUTCONFIG)
endif()
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
# Enable parallel compilation on Windows (use all available cores)
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
cmake_policy(SET CMP0079 NEW)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()
set( EXECUTABLE_NAME ${EXECUTABLE_NAME} )
set( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" CACHE STRING "" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" CACHE STRING "" )
add_executable( ${EXECUTABLE_NAME} ${SOURCE_FILES} )

# Handle CMake 4.0+ compatibility with third-party libraries (like libpng)
# that may have minimum version requirements older than 3.5
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()


add_subdirectory( "${BASE_DIRECTORY}/core" "lib" )
target_link_libraries( ${EXECUTABLE_NAME} PUBLIC helios)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE stdc++fs)
endif()
if(APPLE) #get rid of annoying duplicate library warning on Mac
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_warn_duplicate_libraries")
endif()

# -- linking plug-ins --#
LIST(LENGTH PLUGINS PLUGIN_COUNT)
message( STATUS "[Helios] Loading ${PLUGIN_COUNT} plug-ins")
foreach(PLUGIN ${PLUGINS})
    message( STATUS "[Helios] Loading plug-in ${PLUGIN}")
    if( ${PLUGIN} STREQUAL ${EXECUTABLE_NAME} )
        message( FATAL_ERROR "[Helios] The executable name cannot be the same as a plugin name. Please rename your executable." )
    endif()
    add_subdirectory( "${BASE_DIRECTORY}/plugins/${PLUGIN}" "${PROJECT_BINARY_DIR}/plugins/${PLUGIN}" )
    target_link_libraries( ${EXECUTABLE_NAME} PUBLIC ${PLUGIN} )
    target_link_libraries( ${PLUGIN} PUBLIC helios )
    if( NOT APPLE )
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
            target_link_libraries(${PLUGIN} PRIVATE stdc++fs)
        endif()
    endif()
endforeach(PLUGIN)

target_compile_definitions(helios PUBLIC $<$<CONFIG:Debug>:HELIOS_DEBUG>  $<$<CONFIG:RelWithDebInfo>:HELIOS_DEBUG> )

if( ENABLE_OPENMP )
    # First try standard OpenMP detection
    find_package(OpenMP QUIET)
    
    # If OpenMP not found and we're on macOS with Apple Clang, try automatic libomp configuration
    if(NOT OpenMP_CXX_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(STATUS "[Helios] Configuring OpenMP for Apple Clang on macOS...")
        
        # Find Homebrew
        find_program(BREW NAMES brew)
        if(BREW)
            # Check if libomp is installed
            execute_process(COMMAND ${BREW} ls libomp 
                           RESULT_VARIABLE BREW_RESULT_CODE 
                           OUTPUT_QUIET ERROR_QUIET)
            
            if(NOT BREW_RESULT_CODE)
                # Get libomp prefix
                execute_process(COMMAND ${BREW} --prefix libomp 
                               OUTPUT_VARIABLE LIBOMP_PREFIX 
                               OUTPUT_STRIP_TRAILING_WHITESPACE)
                
                # Configure OpenMP for Apple Clang + libomp
                set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
                set(OpenMP_CXX_INCLUDE_DIRS "${LIBOMP_PREFIX}/include")
                
                # Add include directory globally for compatibility
                include_directories("${LIBOMP_PREFIX}/include")
                
                # Create OpenMP target if it doesn't exist
                if(NOT TARGET OpenMP::OpenMP_CXX)
                    add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
                    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                        IMPORTED_LOCATION "${OpenMP_omp_LIBRARY}"
                        INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
                        INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_CXX_INCLUDE_DIRS}"
                        INTERFACE_LINK_LIBRARIES "${OpenMP_omp_LIBRARY}")
                endif()
                
                set(OpenMP_FOUND TRUE)
                set(OpenMP_CXX_FOUND TRUE)
                
                message(STATUS "[Helios] Automatically configured OpenMP using Homebrew libomp from ${LIBOMP_PREFIX}")
            else()
                message(WARNING "[Helios] OpenMP requires libomp on macOS. Install with: brew install libomp")
            endif()
        else()
            message(WARNING "[Helios] OpenMP on macOS requires Homebrew. Install from https://brew.sh then run: brew install libomp")
        endif()
    endif()
    
    # Final OpenMP configuration
    if (OpenMP_CXX_FOUND)
        message( STATUS "[Helios] Enabling OpenMP support" )
        target_link_libraries(helios PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(helios PUBLIC USE_OPENMP)
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            message(WARNING "[Helios] You are using Apple Clang compiler, which does not support OpenMP. The program will compile without OpenMP support.")
        else()
            message(WARNING "[Helios] OpenMP not found! The program will compile without OpenMP support.")
        endif()
    endif()
endif()

enable_testing()
add_test(NAME Test0 COMMAND ${EXECUTABLE_NAME} 0)