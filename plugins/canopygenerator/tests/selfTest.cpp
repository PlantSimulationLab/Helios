#include "CanopyGenerator.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

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

DOCTEST_TEST_CASE("CanopyGenerator White Spruce canopy_configuration selects the plant layout") {
    // Regression test. buildCanopy(WhiteSpruceCanopyParameters) tested canopy_configuration with != in both branches, so
    // "uniform" produced the random layout and "random" (the default) produced a regular grid.
    const int2 plant_count = make_int2(2, 2);
    const vec2 plant_spacing = make_vec2(10, 10);

    // Largest horizontal distance of any trunk from the center of its grid cell
    auto maxTrunkOffsetFromGrid = [&](const std::string &configuration) {
        Context context_test;
        CanopyGenerator canopygenerator(&context_test);
        canopygenerator.disableMessages();

        WhiteSpruceCanopyParameters params;
        params.canopy_configuration = configuration;
        params.plant_count = plant_count;
        params.plant_spacing = plant_spacing;
        canopygenerator.buildCanopy(params);

        DOCTEST_REQUIRE(canopygenerator.getPlantCount() == uint(plant_count.x * plant_count.y));

        float max_offset = 0.f;
        uint plant_ID = 0;
        for (int j = 0; j < plant_count.y; j++) {
            for (int i = 0; i < plant_count.x; i++) {
                vec2 xbounds, ybounds, zbounds;
                context_test.getDomainBoundingBox(canopygenerator.getTrunkUUIDs(plant_ID), xbounds, ybounds, zbounds);
                const vec2 trunk_center(0.5f * (xbounds.x + xbounds.y), 0.5f * (ybounds.x + ybounds.y));
                const vec2 cell_center((float(i) + 0.5f - 0.5f * float(plant_count.x)) * plant_spacing.x, (float(j) + 0.5f - 0.5f * float(plant_count.y)) * plant_spacing.y);
                max_offset = std::max(max_offset, (trunk_center - cell_center).magnitude());
                plant_ID++;
            }
        }
        return max_offset;
    };

    // Uniform: every trunk stands at the center of its cell
    DOCTEST_CHECK(maxTrunkOffsetFromGrid("uniform") < 0.01f);
    // Random: trunks are scattered over cells several meters wide, so it is essentially impossible for all of them to be central
    DOCTEST_CHECK(maxTrunkOffsetFromGrid("random") > 0.1f);
}

int CanopyGenerator::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

DOCTEST_TEST_CASE("CanopyGenerator keeps the plants of earlier canopies when another canopy is built") {
    Context context_test;
    CanopyGenerator canopygenerator(&context_test);
    canopygenerator.disableMessages();

    VSPGrapevineParameters vines;
    vines.plant_count = make_int2(2, 1);
    canopygenerator.buildCanopy(vines);
    DOCTEST_REQUIRE(canopygenerator.getPlantCount() == 2);
    const auto vine_leaves = canopygenerator.getLeafUUIDs(1);
    const auto vine_trunk = canopygenerator.getTrunkUUIDs(1);
    DOCTEST_REQUIRE(!vine_leaves.empty());
    DOCTEST_REQUIRE(!vine_trunk.empty());

    SphericalCrownsCanopyParameters spheres;
    spheres.plant_count = make_int2(1, 1);
    spheres.canopy_origin = make_vec3(20, 0, 0);
    canopygenerator.buildCanopy(spheres);
    DOCTEST_CHECK(canopygenerator.getPlantCount() == 3);
    DOCTEST_CHECK(canopygenerator.getLeafUUIDs(1) == vine_leaves);
    const auto sphere_leaves = canopygenerator.getLeafUUIDs(2);
    DOCTEST_CHECK(!sphere_leaves.empty());

    ConicalCrownsCanopyParameters cones;
    cones.plant_count = make_int2(1, 1);
    cones.canopy_origin = make_vec3(40, 0, 0);
    canopygenerator.buildCanopy(cones);
    DOCTEST_CHECK(canopygenerator.getPlantCount() == 4);
    DOCTEST_CHECK(canopygenerator.getLeafUUIDs(2) == sphere_leaves);
    const auto cone_leaves = canopygenerator.getLeafUUIDs(3);
    DOCTEST_CHECK(!cone_leaves.empty());

    HomogeneousCanopyParameters homogeneous;
    homogeneous.canopy_origin = make_vec3(60, 0, 0);
    canopygenerator.buildCanopy(homogeneous);
    DOCTEST_CHECK(canopygenerator.getPlantCount() == 5);
    DOCTEST_CHECK(canopygenerator.getLeafUUIDs(1) == vine_leaves);
    DOCTEST_CHECK(canopygenerator.getLeafUUIDs(3) == cone_leaves);
    DOCTEST_CHECK(!canopygenerator.getLeafUUIDs(4).empty());

    // The vines' trunks are still found under the vines' plant IDs, and the canopies without trunks have none
    DOCTEST_CHECK(canopygenerator.getTrunkUUIDs(1) == vine_trunk);
    DOCTEST_CHECK(canopygenerator.getTrunkUUIDs(4).empty());

    // A plant with no trunk built before one that has a trunk does not shift the trunk's plant ID
    Context context_tomato;
    CanopyGenerator tomato_then_vine(&context_tomato);
    tomato_then_vine.disableMessages();
    TomatoParameters tomatoes;
    tomatoes.plant_count = make_int2(1, 1);
    tomato_then_vine.buildCanopy(tomatoes);
    VSPGrapevineParameters one_vine;
    one_vine.plant_count = make_int2(1, 1);
    one_vine.canopy_origin = make_vec3(20, 0, 0);
    tomato_then_vine.buildCanopy(one_vine);
    DOCTEST_REQUIRE(tomato_then_vine.getPlantCount() == 2);
    DOCTEST_CHECK(tomato_then_vine.getTrunkUUIDs(0).empty());
    DOCTEST_CHECK(!tomato_then_vine.getTrunkUUIDs(1).empty());
}

DOCTEST_TEST_CASE("CanopyGenerator loadXML reads every <canopygenerator> block") {
    // The loop over <canopygenerator> blocks advanced with next_sibling("CanopyGenerator"), which XML matches case-sensitively,
    // so only the first block of a file was read and the canopies in any later block were silently left out.
    const std::filesystem::path directory = std::filesystem::temp_directory_path();
    const std::string block = "  <canopygenerator>\n    <HomogeneousCanopyParameters>\n      <leaf_area_index>0.5</leaf_area_index>\n    </HomogeneousCanopyParameters>\n  </canopygenerator>\n";
    const std::filesystem::path one_block_file = directory / "helios_canopygenerator_one_block.xml";
    const std::filesystem::path two_block_file = directory / "helios_canopygenerator_two_blocks.xml";
    {
        std::ofstream one(one_block_file);
        one << "<helios>\n" << block << "</helios>\n";
        std::ofstream two(two_block_file);
        two << "<helios>\n" << block << block << "</helios>\n";
    }

    size_t one_block_primitives = 0;
    size_t two_block_primitives = 0;
    {
        Context context;
        CanopyGenerator canopygenerator(&context);
        canopygenerator.disableMessages();
        capture_cout cout_buffer;
        canopygenerator.loadXML(one_block_file.string().c_str(), true);
        one_block_primitives = context.getPrimitiveCount();
    }
    {
        Context context;
        CanopyGenerator canopygenerator(&context);
        canopygenerator.disableMessages();
        capture_cout cout_buffer;
        canopygenerator.loadXML(two_block_file.string().c_str(), true);
        two_block_primitives = context.getPrimitiveCount();
    }
    std::filesystem::remove(one_block_file);
    std::filesystem::remove(two_block_file);

    DOCTEST_REQUIRE(one_block_primitives > 0);
    DOCTEST_CHECK(two_block_primitives == 2 * one_block_primitives);
}
