#pragma once

// Simple API export header for cpplocate integration into Helios

#ifndef CPPLOCATE_API
#  ifdef _WIN32
#    ifdef LIBLOCATE_EXPORTS
#      define CPPLOCATE_API __declspec(dllexport)
#    else
#      define CPPLOCATE_API __declspec(dllimport)
#    endif
#  else
#    define CPPLOCATE_API
#  endif
#endif