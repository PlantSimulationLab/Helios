#include "CanopyGenerator.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

using namespace helios;

DOCTEST_TEST_CASE("CanopyGenerator Homogeneous Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_0(&context_test);
    canopygenerator_0.disableMessages();
    
    HomogeneousCanopyParameters params_0;
    DOCTEST_CHECK_NOTHROW(canopygenerator_0.buildCanopy(params_0));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Spherical Crowns Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_1(&context_test);
    canopygenerator_1.disableMessages();
    
    SphericalCrownsCanopyParameters params_1;
    DOCTEST_CHECK_NOTHROW(canopygenerator_1.buildCanopy(params_1));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator VSP Grapevine Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_2(&context_test);
    canopygenerator_2.disableMessages();
    
    VSPGrapevineParameters params_2;
    params_2.grape_radius = 0;
    DOCTEST_CHECK_NOTHROW(canopygenerator_2.buildCanopy(params_2));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Split Trellis Grapevine Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_3(&context_test);
    canopygenerator_3.disableMessages();
    
    SplitGrapevineParameters params_3;
    params_3.grape_radius = 0;
    DOCTEST_CHECK_NOTHROW(canopygenerator_3.buildCanopy(params_3));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Unilateral Trellis Grapevine Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_4(&context_test);
    canopygenerator_4.disableMessages();
    
    UnilateralGrapevineParameters params_4;
    params_4.grape_radius = 0;
    DOCTEST_CHECK_NOTHROW(canopygenerator_4.buildCanopy(params_4));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Goblet Trellis Grapevine Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_5(&context_test);
    canopygenerator_5.disableMessages();
    
    GobletGrapevineParameters params_5;
    params_5.grape_radius = 0;
    DOCTEST_CHECK_NOTHROW(canopygenerator_5.buildCanopy(params_5));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Strawberry Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_7(&context_test);
    canopygenerator_7.disableMessages();
    
    StrawberryParameters params_7;
    DOCTEST_CHECK_NOTHROW(canopygenerator_7.buildCanopy(params_7));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Walnut Tree Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_8(&context_test);
    canopygenerator_8.disableMessages();
    
    WalnutCanopyParameters params_8;
    params_8.fruit_radius = 0.f;
    DOCTEST_CHECK_NOTHROW(canopygenerator_8.buildCanopy(params_8));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Sorghum Plant Canopy Generation") {
    Context context_test;
    CanopyGenerator canopygenerator_9(&context_test);
    canopygenerator_9.disableMessages();
    
    SorghumCanopyParameters params_9;
    DOCTEST_CHECK_NOTHROW(canopygenerator_9.buildCanopy(params_9));
    
    // Verify canopy was created
    std::vector<uint> all_UUIDs = context_test.getAllUUIDs();
    DOCTEST_CHECK(!all_UUIDs.empty());
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

DOCTEST_TEST_CASE("CanopyGenerator Primitive Deletion Test") {
    Context context_test;
    CanopyGenerator canopygenerator_6(&context_test);
    canopygenerator_6.disableMessages();
    
    HomogeneousCanopyParameters params_6;
    DOCTEST_CHECK_NOTHROW(canopygenerator_6.buildCanopy(params_6));
    
    std::vector<uint> UUIDs_leaves = flatten(canopygenerator_6.getLeafUUIDs(0));
    DOCTEST_CHECK(UUIDs_leaves.size() > 23);
    
    // Delete some primitives
    DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUIDs_leaves.at(0)));
    DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUIDs_leaves.at(11)));
    DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUIDs_leaves.at(23)));
    DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUIDs_leaves.back()));
    
    // Update UUIDs after deletion
    UUIDs_leaves = flatten(canopygenerator_6.getLeafUUIDs(0));
    
    // Verify all remaining UUIDs exist
    for (uint UUID: UUIDs_leaves) {
        DOCTEST_CHECK(context_test.doesPrimitiveExist(UUID));
    }
    
    std::vector<uint> UUIDs_all = canopygenerator_6.getAllUUIDs(0);
    
    // Verify all remaining primitives exist
    for (uint p: UUIDs_all) {
        DOCTEST_CHECK(context_test.doesPrimitiveExist(p));
    }
    
    context_test.deletePrimitive(context_test.getAllUUIDs());
}

int CanopyGenerator::selfTest() {
    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}