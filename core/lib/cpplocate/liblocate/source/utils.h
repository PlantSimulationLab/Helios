
#pragma once


#ifdef __cplusplus
extern "C"
{
#endif


unsigned char checkStringParameter(const char * path, unsigned int * pathLength);
unsigned char checkStringOutParameter(char ** path, unsigned int * pathLength);
unsigned char checkStringVectorOutParameter(char *** paths, unsigned int ** lengths, unsigned int * count);
void invalidateStringOutParameter(char ** path, unsigned int * pathLength);
void copyToStringOutParameter(const char * source, unsigned int length, char ** target, unsigned int * targetLength);

/**
*  @brief
*    Convert path into unified form (replace '\' with '/')
*
*  @param[inout] path
*    Path
*  @param[in] pathLength
*    The length of path
*/
void unifyPathDelimiters(char * path, unsigned int pathLength);

/**
*  @brief
*    Cut away filename portion of a path, get path to directory
*
*  @param[in] fullpath
*    Path (e.g., '/path/to/file.txt')
*  @param[in] length
*    Length of fullpath (excluding null byte)
*  @param[out] newLength
*    The end of the substring representing the directory path (e.g., '/path/to')
*/
void getDirectoryPart(const char * fullpath, unsigned int length, unsigned int * newLength);

/**
*  @brief
*    Cut away filename and bundle portion of a path, get path to bundle root
*
*  @param[in] fullpath
*    Path (e.g., '/path/to/file.txt')
*  @param[in] length
*    Length of fullpath (excluding null byte)
*  @param[out] newLength
*    The end of the substring representing the directory path (e.g., '/path/to')
*/
void getBundlePart(const char * fullpath, unsigned int length, unsigned int * newLength);

/**
*  @brief
*    Get system base path for path to library or executable
*
*  @param[in] path
*    Path to library or executable (e.g., '/usr/bin/myapp')
*  @param[in] pathLength
*    The length of the path
*  @param[out] systemPathLength
*    The last position of the system part of the path (e.g., '/usr')
*
*  @remarks
*    This function returns the base path if the given file
*    is a system path. Otherwise, it returns an empty string.
*
*    Examples:
*      '/usr/bin/myapp' -> '/usr'
*      '/usr/local/bin/myapp' -> '/usr/local'
*      '/usr/lib/mylib.so' -> '/usr'
*      '/usr/lib64/mylib.so' -> '/usr'
*      '/usr/local/lib64/mylib.so' -> '/usr/local'
*      '/crosscompile/armv4/usr/lib/mylib.so.2' -> '/crosscompile/armv4/usr'
*/
void getSystemBasePath(const char * path, unsigned int pathLength, unsigned int * systemPathLength);

/**
*  @brief
*    Get value of environment variable
*
*  @param[in] name
*    Name of environment variable
*  @param[in] nameLength
*    The length of name
*  @param[out] value
*    The value of the referenced environment variable
*  @param[out] valueLength
*    The length of value
*
*  @return
*    Value of the environment variable
*
*  The caller takes memory ownership over *value.
*/
void getEnv(const char * name, unsigned int nameLength, char ** value, unsigned int * valueLength);

/**
*  @brief
*    Check if file or directory exists
*
*  @param[in] path
*    Path to file or directory
*  @param[in] pathLength
*    Length of path
*
*  @return
*    'true' if it exists, else 'false'
*/
unsigned char fileExists(const char * path, unsigned int pathLength);


#ifdef __cplusplus
}
#endif
