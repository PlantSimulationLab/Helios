#include "WeberPennTree.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

using namespace helios;

DOCTEST_TEST_CASE("WeberPennTree Constructor") {
    Context context;
    DOCTEST_CHECK_NOTHROW(WeberPennTree weberpenntree(&context));
}

DOCTEST_TEST_CASE("WeberPennTree Build Default Library Trees") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    float spacing = 5;
    std::vector<std::string> trees = {"Almond", "Apple", "Avocado", "Lemon", "Olive", "Orange", "Peach", "Pistachio", "Walnut"};

    for (int i = 0; i < trees.size(); i++) {
        Context context;
        WeberPennTree weberpenntree(&context);

        DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(1));
        DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(3, 3)));

        vec3 origin(i * spacing, 0, 0);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree(trees.at(i).c_str(), origin, 0.75f));

        // Verify that primitives were created
        std::vector<uint> all_UUIDs = context.getAllUUIDs();
        DOCTEST_CHECK(!all_UUIDs.empty());
    }

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("WeberPennTree Individual Tree Types") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    DOCTEST_SUBCASE("Almond Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Almond", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Apple Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Avocado Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Avocado", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Lemon Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Lemon", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Olive Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Olive", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Orange Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Orange", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Peach Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Peach", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Pistachio Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Pistachio", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    DOCTEST_SUBCASE("Walnut Tree") {
        Context context;
        WeberPennTree weberpenntree(&context);
        DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Walnut", make_vec3(0, 0, 0)));
        DOCTEST_CHECK(!context.getAllUUIDs().empty());
    }

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("WeberPennTree Recursion Level") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    WeberPennTree weberpenntree(&context);

    // Test setting different recursion levels
    DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(0));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(1));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setBranchRecursionLevel(2));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("WeberPennTree Leaf Subdivisions") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    WeberPennTree weberpenntree(&context);

    // Test setting different leaf subdivisions
    DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(1, 1)));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(3, 3)));
    DOCTEST_CHECK_NOTHROW(weberpenntree.setLeafSubdivisions(make_int2(5, 5)));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("WeberPennTree Scaled Trees") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    WeberPennTree weberpenntree(&context);

    // Test building trees with different scales
    DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(0, 0, 0), 0.5f));
    DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(5, 0, 0), 1.0f));
    DOCTEST_CHECK_NOTHROW(weberpenntree.buildTree("Apple", make_vec3(10, 0, 0), 1.5f));

    DOCTEST_CHECK(!context.getAllUUIDs().empty());

    std::cout.rdbuf(old);
}

int WeberPennTree::selfTest() {
    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}
