# genout.cmake.in
# Generate .out from .c with awk (generic), based upon the automake logic.

# Copyright (C) 2016 Glenn Randers-Pehrson
# Written by Roger Leigh, 2016

# This code is released under the libpng license.
# For conditions of distribution and use, see the disclaimer
# and license in png.h

# Variables substituted from CMakeLists.txt
set(SRCDIR "/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libpng")
set(BINDIR "/home/bnbailey/CLionProjects/Helios_gpu_migration/temp_3915399_1765388951/build/lib/libpng")

set(AWK "/usr/bin/gawk")
set(CMAKE_C_COMPILER "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/bin/gcc")
set(CMAKE_C_FLAGS )
set(INCDIR "/home/bnbailey/CLionProjects/Helios_gpu_migration/temp_3915399_1765388951/build/lib/libpng")
set(PNG_PREFIX "")
set(PNGLIB_MAJOR "1")
set(PNGLIB_MINOR "6")
set(PNGLIB_VERSION "1.6.26")
set(ZLIBINCDIR "/home/bnbailey/CLionProjects/Helios_gpu_migration/temp_3915399_1765388951/build/lib/zlib;/home/bnbailey/CLionProjects/Helios_gpu_migration/core/lib/libpng/../zlib")

set(PLATFORM_C_FLAGS)
if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "")
  set(CMAKE_OSX_SYSROOT "")
  if(CMAKE_OSX_ARCHITECTURES)
    set(PLATFORM_C_FLAGS ${PLATFORM_C_FLAGS} -arch ${CMAKE_OSX_ARCHITECTURES})
  endif()
  if(CMAKE_OSX_SYSROOT)
    set(PLATFORM_C_FLAGS ${PLATFORM_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT})
  endif()
endif()

get_filename_component(INPUTEXT "${INPUT}" EXT)
get_filename_component(OUTPUTEXT "${OUTPUT}" EXT)
get_filename_component(INPUTBASE "${INPUT}" NAME_WE)
get_filename_component(OUTPUTBASE "${OUTPUT}" NAME_WE)
get_filename_component(INPUTDIR "${INPUT}" PATH)
get_filename_component(OUTPUTDIR "${OUTPUT}" PATH)

if ("${INPUTEXT}" STREQUAL ".c" AND "${OUTPUTEXT}" STREQUAL ".out")
  get_filename_component(GENDIR "${OUTPUT}" PATH)
  file(MAKE_DIRECTORY "${GENDIR}")

  file(REMOVE "${OUTPUT}.tf1" "${OUTPUT}.tf2")

  set(INCLUDES "-I${INCDIR}")
  if(ZLIBINCDIR)
    foreach(dir ${ZLIBINCDIR})
      list(APPEND INCLUDES "-I${dir}")
    endforeach()
  endif()

  if(PNG_PREFIX)
    set(PNG_PREFIX_DEF "-DPNG_PREFIX=${PNG_PREFIX}")
  endif()

  execute_process(COMMAND "${CMAKE_C_COMPILER}" "-E"
                          ${CMAKE_C_FLAGS}
                          ${PLATFORM_C_FLAGS}
                          "-I${SRCDIR}"
                          "-I${BINDIR}"
                          ${INCLUDES}
                          "-DPNGLIB_LIBNAME=PNG${PNGLIB_MAJOR}${PNGLIB_MINOR}_0"
                          "-DPNGLIB_VERSION=${PNGLIB_VERSION}"
                          "-DSYMBOL_PREFIX=${SYMBOL_PREFIX}"
                          "-DPNG_NO_USE_READ_MACROS"
                          "-DPNG_BUILDING_SYMBOL_TABLE"
                          ${PNG_PREFIX_DEF}
                          "${INPUT}"
                  OUTPUT_FILE "${OUTPUT}.tf1"
                  WORKING_DIRECTORY "${BINDIR}"
                  RESULT_VARIABLE CPP_FAIL)
  if(CPP_FAIL)
    message(FATAL_ERROR "Failed to generate ${OUTPUT}.tf1")
  endif()

  execute_process(COMMAND "${AWK}" -f "${SRCDIR}/scripts/dfn.awk"
                          "out=${OUTPUT}.tf2" "${OUTPUT}.tf1"
                  WORKING_DIRECTORY "${BINDIR}"
                  RESULT_VARIABLE AWK_FAIL)
  if(AWK_FAIL)
    message(FATAL_ERROR "Failed to generate ${OUTPUT}.tf2")
  endif()

  file(REMOVE "${OUTPUT}.tf1")
  file(RENAME "${OUTPUT}.tf2" "${OUTPUT}")
else()
  message(FATAL_ERROR "Unsupported conversion: ${INPUTEXT} to ${OUTPUTEXT}")
endif()
