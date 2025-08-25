#pragma once

// Simple API export header for liblocate integration into Helios

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIBLOCATE_API
#  ifdef _WIN32
#    ifdef LIBLOCATE_EXPORTS
#      define LIBLOCATE_API __declspec(dllexport)
#    else
#      define LIBLOCATE_API __declspec(dllimport)
#    endif
#  else
#    define LIBLOCATE_API
#  endif
#endif

#ifdef __cplusplus
}
#endif