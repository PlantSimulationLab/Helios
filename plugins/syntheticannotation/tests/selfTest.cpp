#include "SyntheticAnnotation.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

TEST_CASE("SyntheticAnnotation initialization") {
    Context context_test;
    // Test that construction doesn't throw and object can be created
    DOCTEST_CHECK_NOTHROW(SyntheticAnnotation test_instance(&context_test));
}

TEST_CASE("SyntheticAnnotation setBackgroundColor") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that setBackgroundColor doesn't throw
    DOCTEST_CHECK_NOTHROW(test_instance.setBackgroundColor(helios::make_RGBcolor(0.5f, 0.5f, 0.5f)));
    DOCTEST_CHECK_NOTHROW(test_instance.setBackgroundColor(helios::make_RGBcolor(0.9f, 0.9f, 0.9f)));
}

TEST_CASE("SyntheticAnnotation setWindowSize") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that setWindowSize doesn't throw with valid values
    DOCTEST_CHECK_NOTHROW(test_instance.setWindowSize(1280, 720));
    DOCTEST_CHECK_NOTHROW(test_instance.setWindowSize(1920, 1080));
}

TEST_CASE("SyntheticAnnotation setCameraPosition") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    helios::vec3 cam_pos(1, 1, 1);
    helios::vec3 cam_lookat(0, 0, 0);

    // Test that setCameraPosition doesn't throw
    DOCTEST_CHECK_NOTHROW(test_instance.setCameraPosition(cam_pos, cam_lookat));

    // Test vector version
    std::vector<helios::vec3> positions = {cam_pos};
    std::vector<helios::vec3> lookats = {cam_lookat};
    DOCTEST_CHECK_NOTHROW(test_instance.setCameraPosition(positions, lookats));
}

TEST_CASE("SyntheticAnnotation object detection enable/disable") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that enable/disable methods don't throw
    DOCTEST_CHECK_NOTHROW(test_instance.enableObjectDetection());
    DOCTEST_CHECK_NOTHROW(test_instance.disableObjectDetection());
}

TEST_CASE("SyntheticAnnotation semantic segmentation enable/disable") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that enable/disable methods don't throw
    DOCTEST_CHECK_NOTHROW(test_instance.enableSemanticSegmentation());
    DOCTEST_CHECK_NOTHROW(test_instance.disableSemanticSegmentation());
}

TEST_CASE("SyntheticAnnotation instance segmentation enable/disable") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that enable/disable methods don't throw
    DOCTEST_CHECK_NOTHROW(test_instance.enableInstanceSegmentation());
    DOCTEST_CHECK_NOTHROW(test_instance.disableInstanceSegmentation());
}

TEST_CASE("SyntheticAnnotation labelPrimitives single UUID") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    uint sampleUUID = 12345;
    // Test that labelPrimitives doesn't throw with valid inputs
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives(sampleUUID, "test_label"));
}

TEST_CASE("SyntheticAnnotation labelPrimitives const char label") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that labelPrimitives doesn't throw with valid label
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives("test_label"));
}

TEST_CASE("SyntheticAnnotation labelPrimitives vector") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    std::vector<uint> uuids = {123, 456, 789};
    // Test that labelPrimitives doesn't throw with vector input
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives(uuids, "vector_test_label"));
}

TEST_CASE("SyntheticAnnotation labelPrimitives vector groups") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    std::vector<std::vector<uint>> uuid_groups = {{123, 456}, {789, 1011}};
    // Test that labelPrimitives doesn't throw with nested vector input
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives(uuid_groups, "group_test_label"));
}

TEST_CASE("SyntheticAnnotation labelUnlabeledPrimitives") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that labelUnlabeledPrimitives doesn't throw
    DOCTEST_CHECK_NOTHROW(test_instance.labelUnlabeledPrimitives("unlabeled_test"));
}

TEST_CASE("SyntheticAnnotation addSkyDome") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that addSkyDome doesn't throw (though the file might not exist)
    DOCTEST_CHECK_NOTHROW(test_instance.addSkyDome("plugins/visualizer/textures/SkyDome_clouds.jpg"));
}

int SyntheticAnnotation::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
