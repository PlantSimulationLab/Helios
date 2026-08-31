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
            helios::requireGPUOrFail("A projectbuilder test needs the radiation plugin's GPU backend and would have skipped."); \
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

DOCTEST_TEST_CASE("ProjectBuilder - updateGround replaces the ground built from XML") {
    SKIP_IF_NO_GPU();
    ProjectBuilder projectbuilder;
    DOCTEST_CHECK_NOTHROW(projectbuilder.buildFromXML());
    DOCTEST_REQUIRE(projectbuilder.context != nullptr);

    // Coverage for updateGround(), which had none, plus the object ID that BuildGeometry() now hands back.
    //
    // This is deliberately not claimed as a regression test for the uninitialized ProjectBuilder::ground_objID that
    // updateGround() used to read: that read has no observable effect in the normal path, because updateGround()
    // deletes the ground primitives first and emptying an object deletes it, so the ground object goes away whether
    // or not the ID is correct. The only behavior the stale read changed was the case where the indeterminate value
    // happened to name an unrelated live object, which was then deleted -- not something a test can provoke
    // deterministically. The fix is the in-class initializer plus handing the real ID back.
    // BuildGeometry() is only compiled in when the plant architecture plug-in is enabled, and it is what builds the
    // initial ground and hands its object ID back. Note the ID is not expected to still name a live object here:
    // buildFromXML() finishes by calling updateGround(), which replaces that ground with a freshly built one.
#ifdef ENABLE_PLANT_ARCHITECTURE
    uint published_ground_objID = 0;
    DOCTEST_REQUIRE(projectbuilder.context->doesGlobalDataExist("ground_objID"));
    projectbuilder.context->getGlobalData("ground_objID", published_ground_objID);
    DOCTEST_CHECK(published_ground_objID != 0);
#endif

    size_t objects_after_build = projectbuilder.context->getAllObjectIDs().size();
    DOCTEST_CHECK_NOTHROW(projectbuilder.updateGround());
    size_t objects_after_update = projectbuilder.context->getAllObjectIDs().size();
    DOCTEST_CHECK(objects_after_update == objects_after_build);
}

int ProjectBuilder::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
