
#include <cstdlib>

#include <gmock/gmock.h>

#include "../../liblocate/source/utils.h"


class utils_test : public testing::Test
{
public:
    utils_test()
    {
    }
};

TEST_F(utils_test, unifiedPath_EmptyPath)
{
    unifyPathDelimiters(nullptr, 0);

    SUCCEED();
}

TEST_F(utils_test, unifiedPath_NoPath)
{
    const char * source = "A-string";
    const char * expected = "A-string";

    const unsigned int length = strlen(source);
    char * actual = reinterpret_cast<char*>(malloc(sizeof(char) * length+1));
    memcpy(actual, source, length+1);

    unifyPathDelimiters(actual, length);

    EXPECT_STREQ(expected, actual);
    EXPECT_EQ(0, actual[length]);
    EXPECT_EQ(length, strlen(actual));

    free(actual);
}

TEST_F(utils_test, unifiedPath_UnixPath)
{
    const char * source = "/usr/include/c++/v1/tuple";
    const char * expected = "/usr/include/c++/v1/tuple";

    const unsigned int length = strlen(source);
    char * actual = reinterpret_cast<char*>(malloc(sizeof(char) * length+1));
    memcpy(actual, source, length+1);

    unifyPathDelimiters(actual, length);

    EXPECT_STREQ(expected, actual);
    EXPECT_EQ(0, actual[length]);
    EXPECT_EQ(length, strlen(actual));

    free(actual);
}

TEST_F(utils_test, unifiedPath_WindowsPath)
{
    const char * source = "C:\\dev\\include\\c++\\v1\\tuple";
    const char * expected = "C:/dev/include/c++/v1/tuple";

    const unsigned int length = strlen(source);
    char * actual = reinterpret_cast<char*>(malloc(sizeof(char) * length+1));
    memcpy(actual, source, length+1);

    unifyPathDelimiters(actual, length);

    EXPECT_STREQ(expected, actual);
    EXPECT_EQ(0, actual[length]);
    EXPECT_EQ(length, strlen(actual));

    free(actual);
}

TEST_F(utils_test, getDirectoryPath_EmptyPath)
{
    unsigned int newLength = 10;

    getDirectoryPart(nullptr, 0, &newLength);

    EXPECT_EQ(0, newLength); // ""
}

TEST_F(utils_test, getDirectoryPath_NoPath)
{
    const char * source = "A-string";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getDirectoryPart(source, length, &newLength);

    EXPECT_EQ(8, newLength); // "A-string"
}

TEST_F(utils_test, getDirectoryPath_UnixPath)
{
    const char * source = "/usr/include/c++/v1/tuple";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getDirectoryPart(source, length, &newLength);

    EXPECT_EQ(19, newLength); // "/usr/include/c++/v1"
}

TEST_F(utils_test, getDirectoryPath_WindowsPath)
{
    const char * source = "C:\\dev\\include\\c++\\v1\\tuple";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getDirectoryPart(source, length, &newLength);

    EXPECT_EQ(21, newLength); // "C:\\dev\\include\\c++\\v1"
}

TEST_F(utils_test, getBundlePart_EmptyPath)
{
    unsigned int newLength = 10;

    getBundlePart(nullptr, 0, &newLength);

    EXPECT_EQ(0, newLength); // ""
}

TEST_F(utils_test, getBundlePart_NoPath)
{
    const char * source = "/usr/local/lib/test";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getBundlePart(source, length, &newLength);

    EXPECT_EQ(0, newLength); // ""
}

TEST_F(utils_test, getBundlePart_BundlePath)
{
    const char * source = "/home/user/liblocate/test.app/Contents/MacOS";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getBundlePart(source, length, &newLength);

    EXPECT_EQ(29, newLength); // "/home/user/liblocate/test.app"
}

TEST_F(utils_test, getSystemBasePath_EmptyPath)
{
    getSystemBasePath(nullptr, 0, nullptr);

    SUCCEED();
}

TEST_F(utils_test, getSystemBasePath_NoPath)
{
    const char * source = "A-string";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getSystemBasePath(source, length, &newLength);

    EXPECT_EQ(0, newLength); // ""
}

TEST_F(utils_test, getSystemBasePath_SystemPath)
{
    const char * source = "/usr/local/lib/cpplocate";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getSystemBasePath(source, length, &newLength);

    EXPECT_EQ(11, newLength); // "/usr/local/"
}

TEST_F(utils_test, getSystemBasePath_ChrootPath)
{
    const char * source = "/home/user/dev/deploy/usr/local/lib/cpplocate";
    const unsigned int length = strlen(source);
    unsigned int newLength = 0;

    getSystemBasePath(source, length, &newLength);

    EXPECT_EQ(32, newLength); // "/home/user/dev/deploy/usr/local/"
}

TEST_F(utils_test, getEnv_NoName)
{
    char * value = nullptr;
    unsigned int valueLength = 0;

    getEnv(nullptr, 0, &value, &valueLength);

    EXPECT_EQ(0, valueLength);
    EXPECT_EQ(0x0, value);
}

TEST_F(utils_test, getEnv_NoEnv)
{
    const char * key = "_NO_ENV_";
    const unsigned int keyLength = strlen(key);

    char * value = nullptr;
    unsigned int valueLength = 0;

    getEnv(key, keyLength, &value, &valueLength);

    EXPECT_EQ(0, valueLength);
    EXPECT_EQ(0x0, value);
}

TEST_F(utils_test, getEnv_Value)
{
#ifdef SYSTEM_WINDOWS
    const char * key = "OS";
    const unsigned int keyLength = strlen(key);

    char * value = nullptr;
    unsigned int valueLength = 0;

    const char * expected = "Windows_NT";
    const unsigned int expectedLength = strlen(expected);

    getEnv(key, keyLength, &value, &valueLength);

    EXPECT_EQ(expectedLength, valueLength);
    EXPECT_EQ(0, strcmp(value, expected));

    free(value);
#else
    const char * key = "CPPLOCATE_DISPLAY";
    const unsigned int keyLength = strlen(key);

    char * value = nullptr;
    unsigned int valueLength = 0;

    // Ensure a set environment variable through the C API
    // This variable simulates the DISPLAY environment variable
    setenv(key, ":0", true);

    // Retrieve environment variable through liblocate API
    getEnv(key, keyLength, &value, &valueLength);

    // Content is in format ':[display number]'
    ASSERT_LT(1, valueLength);
    EXPECT_EQ(':', value[0]);
    EXPECT_EQ(0, value[valueLength]);
    EXPECT_EQ(valueLength, strlen(value));

    free(value);
#endif
}

TEST_F(utils_test, fileExists_NoPath)
{
    const auto result = fileExists(nullptr, 0);

    ASSERT_FALSE(result);
}

TEST_F(utils_test, fileExists_EmptyPath)
{
    const char * path = "";
    const auto result = fileExists(path, strlen(path));

    ASSERT_FALSE(result);
}
