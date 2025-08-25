
#pragma once


#include <liblocate/liblocate_api.h>


#ifdef __cplusplus
extern "C"
{
#endif


/**
*  @brief
*    Get path to the current executable
*
*  @param[out] path
*    Path to executable (including filename)
*  @param[out] pathLength
*    Number of characters of path without null byte
*
*  @remark
*    The caller takes memory ownership over *path.
*/
LIBLOCATE_API void getExecutablePath(char ** path, unsigned int * pathLength);

/**
*  @brief
*    Get path to the current application bundle
*
*  @param[out] path
*    Path to bundle (including filename)
*  @param[out] pathLength
*    Number of characters of path without null byte
*
*  @remark
*    If the current executable is part of a macOS application bundle,
*    this function returns the part to the bundle. Otherwise, an
*    empty string is returned.
*
*  @remark
*    The caller takes memory ownership over *path.
*/
LIBLOCATE_API void getBundlePath(char ** path, unsigned int * pathLength);

/**
*  @brief
*    Get path to the current module
*
*  @param[out] path
*    Path to module (directory in which the executable is located)
*  @param[out] pathLength
*    Number of characters of path without null byte
*
*  @remark
*    The caller takes memory ownership over *path.
*/
LIBLOCATE_API void getModulePath(char ** path, unsigned int * pathLength);

/**
*  @brief
*    Get path to dynamic library
*
*  @param[in] symbol
*    A symbol from the library, e.g., a function or variable pointer
*  @param[out] path
*    Path to library (including filename)
*  @param[out] pathLength
*    Length of path
*
*  @remark
*    If symbol is null pointer, an empty string is returned.
*
*  @remark
*    The caller takes memory ownership over *path.
*/
LIBLOCATE_API void getLibraryPath(void * symbol, char ** path, unsigned int * pathLength);

/**
*  @brief
*    Locate path to a file or directory
*
*  @param[out] path
*    Path to file or directory
*  @param[out] pathLength
*    Length of path
*  @param[in] relPath
*    Relative path to a file or directory (e.g., 'data/logo.png')
*  @param[in] relPathLength
*    Length of relPath
*  @param[in] systemDir
*    Subdirectory for system installs (e.g., 'share/myappname')
*  @param[in] systemDirLength
*    Length of systemDir
*  @param[in] symbol
*    A symbol from the library, e.g., a function or variable pointer
*
*  @remark
*    This function tries to locate the named file or directory based
*    on the location of the current executable or library. If the
*    file or directory could be found, the base path from which the
*    relative path can be resolved is returned. Otherwise, an empty
*    string is returned.
*
*  @remark
*    The caller takes memory ownership over *path.
*/
LIBLOCATE_API void locatePath(char ** path, unsigned int * pathLength, const char * relPath, unsigned int relPathLength,
    const char * systemDir, unsigned int systemDirLength, void * symbol);

/**
*  @brief
*    Get platform specific path separator
*
*  @param[out] sep
*    Path separator (e.g., '`/`' or '`\`')
*/
LIBLOCATE_API void pathSeparator(char * sep);

/**
*  @brief
*    Get platform specific shared library prefix (e.g., 'lib' on UNIX systems, '' on Windows)
*
*  @param[out] prefix
*    Library prefix
*  @param[out] prefixLength
*    Length of prefix
*
*  @remark
*    The caller takes memory ownership over *prefix.
*/
LIBLOCATE_API void libPrefix(char ** prefix, unsigned int * prefixLength);

/**
*  @brief
*    Get main platform specific shared library extension (e.g., 'dll', 'dylib', or 'so')
*
*  @param[out] extension
*    Library extension
*  @param[out] extensionLength
*    Length of extension
*
*  @remark
*    The caller takes memory ownership over *extension.
*/
LIBLOCATE_API void libExtension(char ** extension, unsigned int * extensionLength);

/**
*  @brief
*    Get platform specific shared library extensions (e.g., ['dll'], ['so'], or ['so', 'dylib'])
*
*  @param[out] extensions
*    Library extensions
*  @param[out] extensionLengths
*    Length of extensions
*  @param[out] extensionCount
*    Number of extensions (the length of both arrays)
*
*  @remark
*    The caller takes memory ownership over *extensions and every string pointer within as well as *extensionLengths.
*/
LIBLOCATE_API void libExtensions(char *** extensions, unsigned int ** extensionLengths, unsigned int * extensionCount);

/**
*  @brief
*    Get home directory of the current user
*
*  @param[out] dir
*    Home directory
*  @param[out] dirLength
*    Length of directory
*
*  @remark
*    The caller takes memory ownership over *dir.
*/
LIBLOCATE_API void homeDir(char ** dir, unsigned int * dirLength);

/**
*  @brief
*    Get profile directory of the current user
*
*  @param[out] dir
*    Profile directory
*  @param[out] dirLength
*    Length of directory
*
*  @remark
*    The caller takes memory ownership over *dir.
*/
LIBLOCATE_API void profileDir(char ** dir, unsigned int * dirLength);

/**
*  @brief
*    Get document directory of the current user
*
*  @param[out] dir
*    Document directory
*  @param[out] dirLength
*    Length of directory
*
*  @remark
*    The caller takes memory ownership over *dir.
*/
LIBLOCATE_API void documentDir(char ** dir, unsigned int * dirLength);

/**
*  @brief
*    Get roaming directory for the named application
*
*  @param[out] dir
*    Roaming directory
*  @param[out] dirLength
*    Length of directory
*  @param[in] application
*    Application name
*  @param[in] applicationLength
*    Length of application name
*
*  @remark
*    The caller takes memory ownership over *dir.
*/
LIBLOCATE_API void roamingDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength);

/**
*  @brief
*    Get local directory for the named application
*
*  @param[out] dir
*    Local directory
*  @param[out] dirLength
*    Length of directory
*  @param[in] application
*    Application name
*  @param[in] applicationLength
*    Length of application name
*
*  @remark
*    The caller takes memory ownership over *dir.
*/
LIBLOCATE_API void localDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength);

/**
*  @brief
*    Get config directory for the named application
*
*  @param[out] dir
*    Config directory
*  @param[out] dirLength
*    Length of directory
*  @param[in] application
*    Application name
*  @param[in] applicationLength
*    Length of application name
*
*  @remark
*    The caller takes memory ownership over *dir.
*/
LIBLOCATE_API void configDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength);

/**
*  @brief
*    Get temporary directory for the named application
*
*  @param[out] dir
*    Temporary directory
*  @param[out] dirLength
*    Length of directory
*  @param[in] application
*    Application name
*  @param[in] applicationLength
*    Length of application name
*
*  @remark
*    The caller takes memory ownership over *dir.
*/
LIBLOCATE_API void tempDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength);


#ifdef __cplusplus
} // extern "C"
#endif
