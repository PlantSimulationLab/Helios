#include "ProjectBuilder.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

DOCTEST_TEST_CASE("ProjectBuilder - Constructor") {
    DOCTEST_CHECK_NOTHROW(ProjectBuilder projectbuilder);
}

DOCTEST_TEST_CASE("ProjectBuilder - buildFromXML") {
    ProjectBuilder projectbuilder;
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_NOTHROW(projectbuilder.buildFromXML());
}

int ProjectBuilder::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
