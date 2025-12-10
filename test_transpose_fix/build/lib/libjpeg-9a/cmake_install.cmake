# Install script for directory: /home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Library" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/test_transpose_fix/build/lib/libjpeg.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Header" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/jerror.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Header" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/jmorecfg.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Header" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/jpeglib.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Header" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/test_transpose_fix/build/lib/libjpeg-9a/jconfig.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/README")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/install.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/usage.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/wizard.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/example.c")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/libjpeg.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/structure.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/coderules.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/filelist.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Documentation" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/libjpeg/doc" TYPE FILE FILES "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libjpeg-9a/change.log")
endif()

