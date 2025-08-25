
#include "utils.h"

#include <stdlib.h>
#include <string.h>

#ifdef SYSTEM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #include <Windows.h>
#else
    #include <sys/stat.h>
#endif


#define windowsPathDelim '\\'
#define windowsPathsDelim ';'

#define unixPathDelim '/'
#define unixPathsDelim ':'

#ifdef SYSTEM_WINDOWS
    #define pathDelim '\\'
    #define pathsDelim ';'
#else
    #define pathDelim '/'
    #define pathsDelim ':'
#endif

#define macOSBundlePath "/Contents/MacOS"
#define macOSBundlePathLength 15


unsigned char checkStringParameter(const char * path, unsigned int * pathLength)
{
    if (path == 0x0 && pathLength != 0x0)
    {
        *pathLength = 0;
    }

    return path != 0x0;
}

unsigned char checkStringOutParameter(char ** path, unsigned int * pathLength)
{
    if (path == 0x0 && pathLength != 0x0)
    {
        *pathLength = 0;
    }

    return path != 0x0;
}

unsigned char checkStringVectorOutParameter(char *** paths, unsigned int ** lengths, unsigned int * count)
{
    if (paths == 0x0 || count == 0x0 || lengths == 0x0)
    {
        if (paths != 0x0)
        {
            *paths = 0x0;
        }

        if (lengths != 0x0)
        {
            *lengths = 0x0;
        }

        if (count != 0x0)
        {
            *count = 0;
        }
    }

    return paths != 0x0;
}

void invalidateStringOutParameter(char ** path, unsigned int * pathLength)
{
    *path = 0x0;
    if (pathLength != 0x0)
    {
        *pathLength = 0;
    }
}

void copyToStringOutParameter(const char * source, unsigned int length, char ** target, unsigned int * targetLength)
{
    *target = (char *)malloc(sizeof(char) * (length + 1));
    memcpy(*target, source, length);
    (*target)[length] = 0;
    if (targetLength != 0x0)
    {
        *targetLength = length;
    }
}

void unifyPathDelimiters(char * path, unsigned int pathLength)
{
    if (path == 0x0 || pathLength == 0)
    {
        return;
    }

    const char * end = path + pathLength;

    for (char * iter = path; iter < end; ++iter)
    {
        if (*iter == windowsPathDelim)
        {
            *iter = unixPathDelim;
        }
    }
}

void getDirectoryPart(const char * fullpath, unsigned int length, unsigned int * newLength)
{
    if (newLength == 0x0)
    {
        return;
    }

    if (fullpath == 0x0 || length == 0)
    {
        *newLength = 0;
        return;
    }

    const char * iter = fullpath + length - 1;
    while (iter > fullpath && *iter != unixPathDelim && *iter != windowsPathDelim)
    {
        --iter;
    }

    *newLength = iter > fullpath ? (unsigned int)(iter - fullpath) : length;
}

void getBundlePart(const char * fullpath, unsigned int length, unsigned int * newLength)
{
    if (newLength == 0x0)
    {
        return;
    }

    if (fullpath == 0x0 || length == 0)
    {
        *newLength = 0;

        return;
    }

    // check for /Contents/MacOS
    const char * potentialBundleStart = fullpath + length - macOSBundlePathLength;

    if (strncmp(potentialBundleStart, macOSBundlePath, macOSBundlePathLength) != 0)
    {
        // No bundle
        *newLength = 0;

        return;
    }

    *newLength = length - macOSBundlePathLength;
}

void getSystemBasePath(const char * path, unsigned int pathLength, unsigned int * subLength)
{
    static const char * systemPaths[] = {
        "/usr/bin/",
        "/usr/lib/",
        "/usr/lib32/",
        "/usr/lib64/",
        "/usr/local/bin/",
        "/usr/local/lib/",
        "/usr/local/lib32/",
        "/usr/local/lib64/",
    };
    static unsigned char systemPathLengths[] = { 9, 11, 15, 17 };
    static unsigned char systemPathPrefixLength[] = { 5, 11 };

    if (!checkStringParameter(path, subLength))
    {
        return;
    }

    for (int i = 0; i < 8; ++i)
    {
        const char * systemPath = systemPaths[i];
        unsigned char systemPathLength = systemPathLengths[i >> 1];

        const char * iter = path + pathLength - 1;
        const char * resetSearchIter = systemPath + systemPathLength - 1;
        const char * searchIter = resetSearchIter;
        while (searchIter >= systemPath && iter >= path)
        {
            searchIter = *iter == *searchIter ? searchIter - 1 : resetSearchIter;
            --iter;
        }

        if (searchIter < systemPath) // sub-systemPath-string found
        {
            *subLength = (unsigned int)(iter - path) + systemPathPrefixLength[i >> 2] + 1;
            return;
        }
    }

    *subLength = 0;
}

void getEnv(const char * name, unsigned int nameLength, char ** value, unsigned int * valueLength)
{
    (void)nameLength;

    if (name == 0x0 || value == 0x0)
    {
        if (valueLength != 0x0)
        {
            *valueLength = 0;
        }

        return;
    }

    const char * systemValue = getenv(name);

    if (systemValue == 0x0)
    {
        invalidateStringOutParameter(value, valueLength);
        return;
    }

    unsigned int systemValueLength = (unsigned int)strlen(systemValue);

    if (systemValueLength == 0)
    {
        invalidateStringOutParameter(value, valueLength);
        return;
    }

    copyToStringOutParameter(systemValue, systemValueLength, value, valueLength);
}

unsigned char fileExists(const char * path, unsigned int pathLength)
{
    (void)pathLength;

    if (path == 0)
    {
        return 0;
    }

#ifdef SYSTEM_WINDOWS

    WIN32_FILE_ATTRIBUTE_DATA fileInfo;
    return (GetFileAttributesExA(path, GetFileExInfoStandard, &fileInfo) != 0);

#else

    struct stat fileInfo;
    return (stat(path, &fileInfo) == 0);

#endif
}
