#include "ProjectBuilder.h"

#ifdef ENABLE_RADIATION_MODEL
#include "RadiationModel.h"
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

/// Skip the current test case if radiation requires a GPU backend that is not available.
#ifdef ENABLE_RADIATION_MODEL
#define SKIP_IF_NO_GPU() \
    do { \
        if (!RadiationModel::isGPUBackendAvailable()) { \
            DOCTEST_MESSAGE("SKIPPED: No GPU backend available (radiation plugin requires GPU)"); \
            return; \
        } \
    } while (0)
#else
#define SKIP_IF_NO_GPU() ((void)0)
#endif

DOCTEST_TEST_CASE("ProjectBuilder - Constructor") {
    DOCTEST_CHECK_NOTHROW(ProjectBuilder projectbuilder);
}

DOCTEST_TEST_CASE("ProjectBuilder - buildFromXML") {
    SKIP_IF_NO_GPU();
    ProjectBuilder projectbuilder;
    DOCTEST_CHECK_NOTHROW(projectbuilder.buildFromXML());
}

int ProjectBuilder::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
