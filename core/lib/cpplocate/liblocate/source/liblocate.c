
#include <liblocate/liblocate.h>

#include <string.h>
#include <stdlib.h>

#if defined(SYSTEM_LINUX)
    #include <unistd.h>
    #include <limits.h>
    #include <pwd.h>
    #include <linux/limits.h>
    #define _GNU_SOURCE
    #define __USE_GNU
    #include <dlfcn.h>
#elif defined(SYSTEM_WINDOWS)
    #include <Windows.h>
#elif defined(SYSTEM_SOLARIS)
    #include <limits.h>
    #include <dlfcn.h>
    #include <unistd.h>
    #include <pwd.h>
#elif defined(SYSTEM_DARWIN)
    #include <mach-o/dyld.h>
    #include <dlfcn.h>
    #include <sys/syslimits.h>
    #include <unistd.h>
    #include <pwd.h>
#elif defined(SYSTEM_FREEBSD)
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <dlfcn.h>
    #include <unistd.h>
    #include <pwd.h>
#endif

#include "utils.h"


void getExecutablePath(char ** path, unsigned int * pathLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(path, pathLength))
    {
        return;
    }

#if defined SYSTEM_LINUX

    // Preallocate PATH_MAX (e.g., 4096) characters and hope the executable path isn't longer (including null byte)
    char exePath[PATH_MAX];

    // Return written bytes, indicating if memory was sufficient
    int len = readlink("/proc/self/exe", exePath, PATH_MAX);

    if (len <= 0 || len == PATH_MAX) // memory not sufficient or general error occured
    {
        invalidateStringOutParameter(path, pathLength);
        return;
    }

    // Copy contents to caller, create caller ownership
    copyToStringOutParameter(exePath, len, path, pathLength);

#elif defined SYSTEM_WINDOWS

    // Preallocate MAX_PATH (e.g., 4095) characters and hope the executable path isn't longer (including null byte)
    char exePath[MAX_PATH];

    // Return written bytes, indicating if memory was sufficient
    unsigned int len = GetModuleFileNameA(GetModuleHandleA(0x0), exePath, MAX_PATH);
    if (len == 0) // memory not sufficient or general error occured
    {
        invalidateStringOutParameter(path, pathLength);
        return;
    }

    // Copy contents to caller, create caller ownership
    copyToStringOutParameter(exePath, len, path, pathLength);

#elif defined SYSTEM_SOLARIS

    // Preallocate PATH_MAX (e.g., 4096) characters and hope the executable path isn't longer (including null byte)
    char exePath[PATH_MAX];

    // Convert executable path to canonical path, return null pointer on error
    if (realpath(getexecname(), exePath) == 0x0)
    {
        invalidateStringOutParameter(path, pathLength);
        return;
    }

    // Copy contents to caller, create caller ownership
    unsigned int len = strlen(exePath);
    copyToStringOutParameter(exePath, len, path, pathLength);

#elif defined SYSTEM_DARWIN

    // Preallocate PATH_MAX (e.g., 4096) characters and hope the executable path isn't longer (including null byte)
    char exePath[PATH_MAX];

    unsigned int len = (unsigned int)PATH_MAX;

    // Obtain executable path to canonical path, return zero on success
    if (_NSGetExecutablePath(exePath, &len) == 0)
    {
        // Convert executable path to canonical path, return null pointer on error
        char * realPath = realpath(exePath, 0x0);

        if (realPath == 0x0)
        {
            invalidateStringOutParameter(path, pathLength);
            return;
        }

        // Copy contents to caller, create caller ownership
        unsigned int len = strlen(realPath);
        copyToStringOutParameter(realPath, len, path, pathLength);

        free(realPath);
    }
    else // len is initialized with the required number of bytes (including zero byte)
    {
        char * intermediatePath = (char *)malloc(sizeof(char) * len);

        // Convert executable path to canonical path, return null pointer on error
        if (_NSGetExecutablePath(intermediatePath, &len) != 0)
        {
            free(intermediatePath);
            invalidateStringOutParameter(path, pathLength);
            return;
        }

        char * realPath = realpath(intermediatePath, 0x0);

        free(intermediatePath);

        // Check if conversion to canonical path succeeded
        if (realPath == 0x0)
        {
            invalidateStringOutParameter(path, pathLength);
            return;
        }

        // Copy contents to caller, create caller ownership
        unsigned int len = strlen(realPath);
        copyToStringOutParameter(realPath, len, path, pathLength);

        free(realPath);
    }

#elif defined SYSTEM_FREEBSD

    // Preallocate characters and hope the executable path isn't longer (including null byte)
    char exePath[2048];

    unsigned int len = 2048;

    int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };

    // Obtain executable path by syscall
    if (sysctl(mib, 4, exePath, &len, 0x0, 0) != 0)
    {
        invalidateStringOutParameter(path, pathLength);
        return;
    }

    // Copy contents to caller, create caller ownership
    copyToStringOutParameter(exePath, len, path, pathLength);

#else

    // If no OS could be detected ... degrade gracefully
    invalidateStringOutParameter(path, pathLength);

#endif
}

void getBundlePath(char ** path, unsigned int * pathLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(path, pathLength))
    {
        return;
    }

    // Get directory where the executable is located
    char * executablePath = 0x0;
    unsigned int executablePathLength = 0;
    getExecutablePath(&executablePath, &executablePathLength);

    // Extract directory part from executable path (without trailing slash)
    unsigned int executablePathDirectoryLength = 0;
    getDirectoryPart(executablePath, executablePathLength, &executablePathDirectoryLength);

    // Convert to canonical path
    unifyPathDelimiters(executablePath, executablePathDirectoryLength);

    // check for /Contents/MacOS
    unsigned int bundlePathLength = 0;
    getBundlePart(executablePath, executablePathDirectoryLength, &bundlePathLength);

    if (bundlePathLength == 0) // No bundle
    {
        free(executablePath);
        invalidateStringOutParameter(path, pathLength);
        return;
    }

    // Copy contents to caller, create caller ownership
    copyToStringOutParameter(executablePath, bundlePathLength, path, pathLength);

    free(executablePath);
}

void getModulePath(char ** path, unsigned int * pathLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(path, pathLength))
    {
        return;
    }

    char * executablePath = 0x0;
    unsigned int executablePathLength = 0;
    getExecutablePath(&executablePath, &executablePathLength);
    unsigned int executablePathDirectoryLength = 0;

    getDirectoryPart(executablePath, executablePathLength, &executablePathDirectoryLength);

    copyToStringOutParameter(executablePath, executablePathDirectoryLength, path, pathLength);

    free(executablePath);
}

void getLibraryPath(void * symbol, char ** path, unsigned int * pathLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(path, pathLength))
    {
        return;
    }

    if (!symbol)
    {
        invalidateStringOutParameter(path, pathLength);
        return;
    }

#if defined SYSTEM_WINDOWS

    char systemPath[MAX_PATH];
    systemPath[0] = '\0';

    HMODULE module;

    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)symbol,
            &module))
    {
        GetModuleFileNameA(module, systemPath, MAX_PATH);
    }

    unsigned int len = (unsigned int)strnlen(systemPath, MAX_PATH);

    copyToStringOutParameter(systemPath, len, path, pathLength);

#else

    Dl_info dlInfo;
    dladdr(symbol, &dlInfo);

    if (!dlInfo.dli_fname)
    {
        invalidateStringOutParameter(path, pathLength);
        return;
    }

    unsigned int len = strlen(dlInfo.dli_fname);
    copyToStringOutParameter(dlInfo.dli_fname, len, path, pathLength);

#endif

    // Return path with system path delimiters
    // unifyPathDelimiters(*path, *pathLength);
}

void locatePath(char ** path, unsigned int * pathLength, const char * relPath, unsigned int relPathLength,
    const char * systemDir, unsigned int systemDirLength, void * symbol)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(path, pathLength))
    {
        return;
    }

    // Obtain executable path
    char * executablePath = 0x0;
    unsigned int executablePathLength = 0;
    getExecutablePath(&executablePath, &executablePathLength);
    unsigned int executablePathDirectoryLength = 0;

    // Obtain bundle path (in case of macOS)
    char * bundlePath = 0x0;
    unsigned int bundlePathLength = 0;
    getBundlePath(&bundlePath, &bundlePathLength);

    // Obtain library path
    char * libraryPath = 0x0;
    unsigned int libraryPathLength = 0;
    getLibraryPath(symbol, &libraryPath, &libraryPathLength);
    unsigned int libraryPathDirectoryLength = 0;

    // Extract directory parts of executable and library paths
    getDirectoryPart(libraryPath, libraryPathLength, &libraryPathDirectoryLength);
    getDirectoryPart(executablePath, executablePathLength, &executablePathDirectoryLength);

    // Compute the size of the maximal possible path to circumvent reallocation of output
    unsigned int maxLength = executablePathDirectoryLength;
    maxLength = maxLength > libraryPathDirectoryLength ? maxLength : libraryPathDirectoryLength;
    maxLength = maxLength > bundlePathLength + 19 ? maxLength : bundlePathLength + 19; // for "/Contents/Resources"
    maxLength += relPathLength + 7 + 2; // for the extra upward path checks, the extra path delimiter and null byte suffix

    const char * dirs[] = { libraryPath, executablePath, bundlePath };
    const unsigned int lengths[] = { libraryPathDirectoryLength, executablePathDirectoryLength, bundlePathLength };

    char * subdir = (char *)malloc(sizeof(char) * maxLength);
    unsigned int subdirLength = 0;
    unsigned int resultdirLength = 0;

    // Check libraryPath, executablePath, and bundlePath as base directories
    for (unsigned char i = 0; i < 3; ++i)
    {
        // Obtain current base directory and associated length
        const char * dir = dirs[i];
        const unsigned int length = lengths[i];

        // Early out for missing base directory
        if (length <= 0)
        {
            continue;
        }

        // Initialize directory to test with base directory
        memcpy(subdir, dir, length);

        // Check <basedir>/<relpath>, <basedir>/../<relpath>, and <basedir>/../../<relpath>
        for (unsigned char j = 0; j < 3; ++j)
        {
            unsigned char relDirLength = j * 3 + 1;

            subdirLength = length;
            // Copy either '/', '/../', or '/../../', depending on current iteration
            memcpy(subdir+subdirLength, "/../../", relDirLength);
            subdirLength += relDirLength;
            memcpy(subdir+subdirLength, relPath, relPathLength);
            resultdirLength = subdirLength;
            subdirLength += relPathLength;

            // End subdirectory with null byte for system functions
            subdir[subdirLength] = 0;

            if (fileExists(subdir, subdirLength)) // successfully found directory
            {
                copyToStringOutParameter(subdir, resultdirLength, path, pathLength);

                goto out;
            }
        }

        if (systemDirLength <= 0)
            continue;

        // Check if it is a system path
        // length + 1 points to the '/' of the directory part of the path

        unsigned int systemBasePathLength;
        getSystemBasePath(dir, length + 1, &systemBasePathLength);

        if (systemBasePathLength > 0)
        {
            subdirLength = systemBasePathLength;
            memcpy(subdir+subdirLength, "/", 1);
            subdirLength += 1;
            memcpy(subdir+subdirLength, systemDir, systemDirLength);
            subdirLength += systemDirLength;
            memcpy(subdir+subdirLength, "/", 1);
            subdirLength += 1;
            memcpy(subdir+subdirLength, relPath, relPathLength);
            resultdirLength = subdirLength;
            subdirLength += relPathLength;

            // End subdirectory with null byte for system functions
            subdir[subdirLength] = 0;

            if (fileExists(subdir, subdirLength)) // successfully found directory
            {
                copyToStringOutParameter(subdir, resultdirLength, path, pathLength);

                goto out;
            }
        }
    }

    // Check app bundle resources
    {
        // Obtain current base directory and associated length
        const char * dir = dirs[2];
        const unsigned int length = lengths[2];

        // Initialize directory to test with base directory
        memcpy(subdir, dir, length);

        subdirLength = length;
        memcpy(subdir+subdirLength, "/Contents/Resources/", 20);
        subdirLength += 20;
        memcpy(subdir+subdirLength, relPath, relPathLength);
        resultdirLength = subdirLength;
        subdirLength += relPathLength;

        // End subdirectory with null byte for system functions
        subdir[subdirLength] = 0;

        if (fileExists(subdir, subdirLength)) // successfully found directory
        {
            copyToStringOutParameter(subdir, resultdirLength, path, pathLength);

            goto out;
        }
    }

    // Could not find path
    invalidateStringOutParameter(path, pathLength);

out:
    // Free temporary memory
    free(libraryPath);
    free(executablePath);
    free(bundlePath);
    free(subdir);
}

void pathSeparator(char * sep)
{
    if (sep != 0x0)
    {
#ifdef SYSTEM_WINDOWS
        *sep = '\\';
#else
        *sep = '/';
#endif
    }
}

void libPrefix(char ** prefix, unsigned int * prefixLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(prefix, prefixLength))
    {
        return;
    }

#if defined SYSTEM_WINDOWS || defined SYSTEM_DARWIN
    copyToStringOutParameter("", 0, prefix, prefixLength);
#else
    copyToStringOutParameter("lib", 3, prefix, prefixLength);
#endif
}

void libExtension(char ** extension, unsigned int * extensionLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(extension, extensionLength))
    {
        return;
    }

#if defined SYSTEM_WINDOWS
    copyToStringOutParameter("dll", 3, extension, extensionLength);
#elif defined SYSTEM_DARWIN
    copyToStringOutParameter("dylib", 5, extension, extensionLength);
#else
    copyToStringOutParameter("so", 2, extension, extensionLength);
#endif
}

void libExtensions(char *** extensions, unsigned int ** extensionLengths, unsigned int * extensionCount)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringVectorOutParameter(extensions, extensionLengths, extensionCount))
    {
        return;
    }

#if defined SYSTEM_WINDOWS
    *extensionCount = 1;
    *extensions = (char **)malloc(sizeof(char *) * *extensionCount);
    *extensionLengths = (unsigned int *)malloc(sizeof(unsigned int) * *extensionCount);
    copyToStringOutParameter("dll", 3, *extensions + 0, *extensionLengths + 0);
#elif defined SYSTEM_DARWIN
    *extensionCount = 2;
    *extensions = (char **)malloc(sizeof(char *) * *extensionCount);
    *extensionLengths = (unsigned int *)malloc(sizeof(unsigned int) * *extensionCount);
    copyToStringOutParameter("dylib", 5, *extensions + 0, *extensionLengths + 0);
    copyToStringOutParameter("so", 2, *extensions + 1, *extensionLengths + 1);
#else
    *extensionCount = 1;
    *extensions = (char **)malloc(sizeof(char *) * *extensionCount);
    *extensionLengths = (unsigned int *)malloc(sizeof(unsigned int) * *extensionCount);
    copyToStringOutParameter("so", 2, *extensions + 0, *extensionLengths + 0);
#endif
}

void homeDir(char ** dir, unsigned int * dirLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(dir, dirLength))
    {
        return;
    }

    #ifdef SYSTEM_WINDOWS

        char * homeDrive, * homePath;
        unsigned int homeDriveLen, homePathLen;
        getEnv("HOMEDRIVE", 9, &homeDrive, &homeDriveLen);
        getEnv("HOMEPATH", 8, &homePath, &homePathLen);

        unsigned int homeLen = homeDriveLen + homePathLen;
        char * home = (char *)malloc(sizeof(char) * homeLen);
        memcpy(home, homeDrive, homeDriveLen);
        memcpy(home + homeDriveLen, homePath, homePathLen);

        copyToStringOutParameter(home, homeLen, dir, dirLength);

        free(homePath);
        free(homeDrive);

    #else // every other UNIX, including Linux and macOS

        char * home;
        unsigned int homeLen;

        // First, test
        getEnv("HOME", 4, &home, &homeLen);

        if (home != 0x0 && homeLen > 0) {
            copyToStringOutParameter(home, homeLen, dir, dirLength);

            free(home);

            return;
        }

        free(home);

        // Fallback using UNIX passwd structure for the current user
        struct passwd* pwd = getpwuid(getuid());

        if (pwd != 0x0)
        {
            copyToStringOutParameter(pwd->pw_dir, strlen(pwd->pw_dir), dir, dirLength);

            return;
        }

        // No home directory was found
        invalidateStringOutParameter(dir, dirLength);

    #endif

}

void profileDir(char ** dir, unsigned int * dirLength)
{
    homeDir(dir, dirLength);
}

void documentDir(char ** dir, unsigned int * dirLength)
{
    homeDir(dir, dirLength);
}

void configDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength)
{
    // Early exit when invalid out-parameters are passed
    if (!checkStringOutParameter(dir, dirLength))
    {
        return;
    }

    char * configPath;
    unsigned int configPathLen;

    #if defined SYSTEM_WINDOWS
        char * appData;
        unsigned int appDataLen;
        getEnv("APPDATA", 7, &appData, &appDataLen);

        configPathLen = appDataLen + 1 + applicationLength;
        configPath = (char *)malloc(sizeof(char) * configPathLen);
        memcpy(configPath, appData, appDataLen);
        configPath[appDataLen] = '\\';

        free(appData);
    #elif defined SYSTEM_DARWIN
        char * home;
        unsigned int homeLen;
        getEnv("HOME", 4, &home, &homeLen);

        configPathLen = homeLen + 21 + applicationLength;
        configPath = (char *)malloc(sizeof(char) * configPathLen);
        memcpy(configPath, home, homeLen);
        memcpy(configPath + homeLen, "/Library/Preferences/", 21);

        free(home);
    #else
        char * home;
        unsigned int homeLen;
        getEnv("HOME", 4, &home, &homeLen);

        configPathLen = homeLen + 9 + applicationLength;
        configPath = (char *)malloc(sizeof(char) * configPathLen);
        memcpy(configPath, home, homeLen);
        memcpy(configPath + homeLen, "/.config/", 9);

        free(home);
    #endif

    unsigned int configPrefixLen = configPathLen - applicationLength;
    memcpy(configPath + configPrefixLen, application, applicationLength);
    copyToStringOutParameter(configPath, configPathLen, dir, dirLength);

    free(configPath);
}

void roamingDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength)
{
    configDir(dir, dirLength, application, applicationLength);
}

void localDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength)
{
    configDir(dir, dirLength, application, applicationLength);
}

void tempDir(char ** dir, unsigned int * dirLength, const char * application, unsigned int applicationLength)
{
    configDir(dir, dirLength, application, applicationLength);
}
