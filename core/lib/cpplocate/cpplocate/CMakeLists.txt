# Minimal CMakeLists.txt for cpplocate integration into Helios

# Target name
set(target cpplocate)

# Source files
set(headers
    include/cpplocate/cpplocate.h
)

set(sources
    source/cpplocate.cpp
    ../liblocate/source/liblocate.c
    ../liblocate/source/utils.c
)

# Build library
add_library(${target} STATIC
    ${sources}
    ${headers}
)

# Include directories
target_include_directories(${target}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../liblocate/include
)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(${target} PRIVATE shlwapi)
elseif(APPLE)
    target_link_libraries(${target} PRIVATE "-framework CoreFoundation")
endif()

# Platform-specific compile definitions
if(WIN32)
    set(SYSTEM_DEFINE "SYSTEM_WINDOWS")
elseif(APPLE)
    set(SYSTEM_DEFINE "SYSTEM_DARWIN")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(SYSTEM_DEFINE "SYSTEM_LINUX")
elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
    set(SYSTEM_DEFINE "SYSTEM_FREEBSD")
elseif(CMAKE_SYSTEM_NAME STREQUAL "SunOS")
    set(SYSTEM_DEFINE "SYSTEM_SOLARIS")
endif()

# Compile definitions
target_compile_definitions(${target}
    PUBLIC
    LIBLOCATE_EXPORTS
    liblocate_EXPORTS
    ${SYSTEM_DEFINE}
)

# Set properties
set_target_properties(${target}
    PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)