
#pragma once


#include <string>
#include <vector>

#include <cpplocate/cpplocate_api.h>


namespace cpplocate
{


/**
*  @brief
*    Get path to the current executable
*
*  @return
*    Path to executable (including filename)
*
*  @remark
*    It is assumed the executable path is static throughout the process.
*/
CPPLOCATE_API std::string getExecutablePath();

/**
*  @brief
*    Get path to the current application bundle
*
*  @return
*    Path to bundle (including filename)
*
*  @remark
*    If the current executable is part of a macOS application bundle,
*    this function returns the part to the bundle. Otherwise, an
*    empty string is returned.
*
*  @remark
*    It is assumed the bundle path is static throughout the process.
*/
CPPLOCATE_API std::string getBundlePath();

/**
*  @brief
*    Get path to the current module
*
*  @return
*    Path to module (directory in which the executable is located)
*
*  @remark
*    It is assumed the executable name is static throughout the process.
*/
CPPLOCATE_API std::string getModulePath();

/**
*  @brief
*    Get path to dynamic library
*
*  @param[in] symbol
*    A symbol from the library, e.g., a function or variable pointer
*
*  @return
*    Path to library (including filename)
*
*  @remark
*    If symbol is nullptr, an empty string is returned.
*/
CPPLOCATE_API std::string getLibraryPath(void * symbol);

/**
*  @brief
*    Locate path to a file or directory
*
*  @param[in] relPath
*    Relative path to a file or directory (e.g., 'data/logo.png')
*  @param[in] systemDir
*    Subdirectory for system installs (e.g., 'share/myappname')
*  @param[in] symbol
*    A symbol from the library, e.g., a function or variable pointer
*
*  @return
*    Path to file or directory
*
*  @remark
*    This function tries to locate the named file or directory based
*    on the location of the current executable or library. If the
*    file or directory could be found, the base path from which the
*    relative path can be resolved is returned. Otherwise, an empty
*    string is returned.
*/
CPPLOCATE_API std::string locatePath(const std::string & relPath, const std::string & systemDir, void * symbol);


/**
*  @brief
*    Get platform specific path separator
*
*  @return
*    Path separator (e.g., '`/`' or '`\`')
*/
CPPLOCATE_API std::string pathSeparator();

/**
*  @brief
*    Get platform specific shared library prefix
*
*  @return
*    Library prefix (e.g., 'lib' on UNIX systems, '' on Windows)
*/
CPPLOCATE_API std::string libPrefix();

/**
*  @brief
*    Get main platform specific shared library extension
*
*  @return
*    Library extension (e.g., 'dll', 'dylib', or 'so')
*/
CPPLOCATE_API std::string libExtension();

/**
*  @brief
*    Get the list platform specific shared library extensions
*
*  @return
*    List of library extensions (e.g., ['dll'], ['so'], or ['dylib', 'so'])
*/
CPPLOCATE_API std::vector<std::string> libExtensions();


/**
*  @brief
*    Get home directory of the current user
*
*  @return
*    Home directory
*/
CPPLOCATE_API std::string homeDir();

/**
*  @brief
*    Get profile directory of the current user
*
*  @return
*    Profile directory
*/
CPPLOCATE_API std::string profileDir();

/**
*  @brief
*    Get document directory of the current user
*
*  @return
*    Document directory
*/
CPPLOCATE_API std::string documentDir();


/**
*  @brief
*    Get config directory for the named application
*
*  @param[in] application
*    Application name
*
*  @return
*    Config directory
*/
CPPLOCATE_API std::string configDir(const std::string & application);

/**
*  @brief
*    Get roaming directory for the named application
*
*  @param[in] application
*    Application name
*
*  @return
*    Roaming directory
*/
CPPLOCATE_API std::string roamingDir(const std::string & application);

/**
*  @brief
*    Get local directory for the named application
*
*  @param[in] application
*    Application name
*
*  @return
*    Local directory
*/
CPPLOCATE_API std::string localDir(const std::string & application);

/**
*  @brief
*    Get temporary directory for the named application
*
*  @param[in] application
*    Application name
*
*  @return
*    Temporary directory
*/
CPPLOCATE_API std::string tempDir(const std::string & application);


} // namespace cpplocate
