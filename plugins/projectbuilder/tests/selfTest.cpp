#include "ProjectBuilder.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

using namespace helios;

DOCTEST_TEST_CASE("ProjectBuilder - Constructor") {
    DOCTEST_CHECK_NOTHROW(ProjectBuilder projectbuilder);
}

DOCTEST_TEST_CASE("ProjectBuilder - buildFromXML") {
    ProjectBuilder projectbuilder;
    DOCTEST_CHECK_NOTHROW(projectbuilder.buildFromXML());
}

int ProjectBuilder::selfTest() {
    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}