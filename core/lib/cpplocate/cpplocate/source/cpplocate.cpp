
#include <cpplocate/cpplocate.h>

#include <stdlib.h>

#include <liblocate/liblocate.h>


namespace
{


/**
*  @brief
*    Convert c-string to std::string.
*
*  @param[in] path
*    The character array
*  @param[in] length
*    The length of the character array (i.e., the number of characters)
*
*  @return
*    The contents of path as string
*
*  @remark
*    The path-pointer is freed.
*/
std::string obtainStringFromLibLocate(char * path, unsigned int length)
{
    if (path == nullptr || length == 0)
    {
        return std::string();
    }

    auto result = std::string(path, length);

    // liblocate transfer ownership of pointer behind path variable
    free(path);

    return result;
}


} // namespace


namespace cpplocate
{


std::string getExecutablePath()
{
    char * path = nullptr;
    unsigned int length = 0;

    ::getExecutablePath(&path, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(path, length);
}

std::string getBundlePath()
{
    char * path = nullptr;
    unsigned int length = 0;

    ::getBundlePath(&path, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(path, length);
}

std::string getModulePath()
{
    char * path = nullptr;
    unsigned int length = 0;

    ::getModulePath(&path, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(path, length);
}

std::string getLibraryPath(void * symbol)
{
    char * path = nullptr;
    unsigned int length = 0;

    ::getLibraryPath(symbol, &path, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(path, length);
}

std::string locatePath(const std::string & relPath, const std::string & systemDir, void * symbol)
{
    char * path = nullptr;
    unsigned int length = 0;

    ::locatePath(&path, &length, relPath.c_str(), (unsigned int)relPath.size(), systemDir.c_str(), (unsigned int)systemDir.size(), symbol);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(path, length);
}

std::string pathSeparator()
{
    char sep;

    ::pathSeparator(&sep);

    return std::string(1, sep);
}

std::string libPrefix()
{
    char * prefix = nullptr;
    unsigned int length = 0;

    ::libPrefix(&prefix, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(prefix, length);
}

std::string libExtension()
{
    char * extension = nullptr;
    unsigned int length = 0;

    ::libExtension(&extension, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(extension, length);
}

std::vector<std::string> libExtensions()
{
    char ** extensions = nullptr;
    unsigned int * lengths = nullptr;
    unsigned int count = 0;

    ::libExtensions(&extensions, &lengths, &count);

    // Convert c-array of c-strings and handle memory ownership
    auto result = std::vector<std::string>(count);

    for (auto i = 0u; i < count; ++i)
    {
        // Convert to string and free memory from liblocate
        result[i] = obtainStringFromLibLocate(extensions[i], lengths[i]);
    }

    if (count > 0)
    {
        free(extensions);
        free(lengths);
    }

    return result;
}

std::string homeDir()
{
    char * dir = nullptr;
    unsigned int length = 0;

    ::homeDir(&dir, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(dir, length);
}

std::string profileDir()
{
    char * dir = nullptr;
    unsigned int length = 0;

    ::profileDir(&dir, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(dir, length);
}

std::string documentDir()
{
    char * dir = nullptr;
    unsigned int length = 0;

    ::documentDir(&dir, &length);

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(dir, length);
}

std::string roamingDir(const std::string & application)
{
    char * dir = nullptr;
    unsigned int length = 0;

    ::roamingDir(&dir, &length, application.c_str(), (unsigned int)application.size());

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(dir, length);
}

std::string localDir(const std::string & application)
{
    char * dir = nullptr;
    unsigned int length = 0;

    ::localDir(&dir, &length, application.c_str(), (unsigned int)application.size());

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(dir, length);
}

std::string configDir(const std::string & application)
{
    char * dir = nullptr;
    unsigned int length = 0;

    ::configDir(&dir, &length, application.c_str(), (unsigned int)application.size());

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(dir, length);
}

std::string tempDir(const std::string & application)
{
    char * dir = nullptr;
    unsigned int length = 0;

    ::tempDir(&dir, &length, application.c_str(), (unsigned int)application.size());

    // Convert to string and free memory from liblocate
    return obtainStringFromLibLocate(dir, length);
}


} // namespace cpplocate
