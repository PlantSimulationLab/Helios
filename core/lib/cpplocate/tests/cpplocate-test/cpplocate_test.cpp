
#include <gmock/gmock.h>

#include <cpplocate/cpplocate.h>


class cpplocate_test : public testing::Test
{
public:
    cpplocate_test()
    {
    }
};

TEST_F(cpplocate_test, getExecutablePath_Return)
{
    const auto result = cpplocate::getExecutablePath();

    EXPECT_LT(0, result.size());
    EXPECT_NE(nullptr, result.c_str());
}

TEST_F(cpplocate_test, getBundlePath_Return)
{
#ifdef SYSTEM_DARWIN
    const auto result = cpplocate::getBundlePath();

    EXPECT_LT(0, result.size());
    EXPECT_NE(nullptr, result.c_str());
#else
    SUCCEED();
#endif
}

TEST_F(cpplocate_test, getModulePath_Return)
{
    const auto result = cpplocate::getModulePath();

    EXPECT_LT(0, result.size());
    EXPECT_NE(nullptr, result.c_str());
}

TEST_F(cpplocate_test, getLibraryPath_Return)
{
    const auto result = cpplocate::getLibraryPath(reinterpret_cast<void*>(cpplocate::getExecutablePath));

    EXPECT_LT(0, result.size());
    EXPECT_NE(nullptr, result.c_str());
}

TEST_F(cpplocate_test, locatePath_Return)
{
    const auto relPath = std::string("source/version.h.in");
    const auto systemPath = std::string("share/liblocate");

    const auto result = cpplocate::locatePath(relPath, systemPath, reinterpret_cast<void*>(cpplocate::getExecutablePath));

    EXPECT_LT(0, result.size());
    EXPECT_NE(nullptr, result.c_str());
}

TEST_F(cpplocate_test, pathSeperator)
{
    #ifdef WIN32
        ASSERT_EQ("\\", cpplocate::pathSeparator());
    #elif __APPLE__
        ASSERT_EQ("/", cpplocate::pathSeparator());
    #else
        ASSERT_EQ("/", cpplocate::pathSeparator());
    #endif
}

TEST_F(cpplocate_test, libPrefix)
{
    #ifdef WIN32
        ASSERT_EQ("", cpplocate::libPrefix());
    #elif __APPLE__
        ASSERT_EQ("", cpplocate::libPrefix());
    #else
        ASSERT_EQ("lib", cpplocate::libPrefix());
    #endif
}

TEST_F(cpplocate_test, libExtension)
{
    #ifdef WIN32
        ASSERT_EQ("dll", cpplocate::libExtension());
    #elif __APPLE__
        ASSERT_EQ("dylib", cpplocate::libExtension());
    #else
        ASSERT_EQ("so", cpplocate::libExtension());
    #endif
}

TEST_F(cpplocate_test, libExtensions)
{
    const auto extensions = cpplocate::libExtensions();
    #ifdef WIN32
        ASSERT_EQ(1, extensions.size());
        ASSERT_EQ("dll", extensions[0]);
    #elif __APPLE__
        ASSERT_EQ(2, extensions.size());
        ASSERT_EQ("dylib", extensions[0]);
    #else
        ASSERT_EQ(1, extensions.size());
        ASSERT_EQ("so", extensions[0]);
    #endif
}

TEST_F(cpplocate_test, homeDirectory)
{
    const auto dir = cpplocate::homeDir();
    ASSERT_LT(0, dir.size());
}
