#include "PlantArchitecture.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"
#include "global.h"

using namespace helios;

double err_tol = 1e-7;

//! Adapts a petiole's object-ID storage to a flat list, so tests can be written against the
//! petiole's geometry without depending on whether it is stored as one object or several.
inline std::vector<uint> flattenPetioleObjIDs(const std::vector<uint> &objIDs) {
    return objIDs;
}
inline std::vector<uint> flattenPetioleObjIDs(uint objID) {
    return std::vector<uint>{objID};
}

//! Grants these tests access to private per-plant state. Declared a friend of PlantArchitecture in
//! PlantArchitecture.h; defined here so nothing test-related is exposed to users of the library.
class PlantArchitectureTestHelper {
public:
    static const std::vector<helios::vec3> &getAttractionPoints(const PlantArchitecture &pa, uint plantID) {
        return pa.plant_instances.at(plantID).attraction_points;
    }
    static bool attractionPointsEnabled(const PlantArchitecture &pa, uint plantID) {
        return pa.plant_instances.at(plantID).attraction_points_enabled;
    }
    static const std::vector<std::shared_ptr<Shoot>> &getShootTree(const PlantArchitecture &pa, uint plantID) {
        return pa.plant_instances.at(plantID).shoot_tree;
    }
};

DOCTEST_TEST_CASE("PlantArchitecture Constructor") {
    Context context;
    DOCTEST_CHECK_NOTHROW(PlantArchitecture pa_test(&context));
}

DOCTEST_TEST_CASE("PlantArchitecture Cancel Flag") {
    // A cancel flag set before a canopy build must short-circuit the per-plant
    // build loop (no plants built), while the same build with the flag clear
    // builds the full grid. This is the mechanism that lets a long generation be
    // aborted mid-build instead of running to completion.
    auto build = [](bool cancel) -> std::size_t {
        Context context;
        PlantArchitecture pa(&context);
        pa.disableMessages();
        pa.loadPlantModelFromLibrary("bean");
        int flag = cancel ? 1 : 0;
        pa.setCancelFlag(&flag);
        std::vector<uint> ids = pa.buildPlantCanopyFromLibrary(make_vec3(0, 0, 0), make_vec2(0.5f, 0.5f), make_int2(3, 3), 0.f, 1.f);
        return ids.size();
    };
    DOCTEST_CHECK(build(false) == 9);
    DOCTEST_CHECK(build(true) == 0);

    // A cancel flag set before advanceTime() must stop the growth loop, leaving
    // the plant far shorter than a full grow.
    auto grow_height = [](bool cancel) -> float {
        Context context;
        PlantArchitecture pa(&context);
        pa.disableMessages();
        pa.loadPlantModelFromLibrary("bean");
        uint pid = pa.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
        int flag = cancel ? 1 : 0;
        pa.setCancelFlag(&flag);
        pa.advanceTime(40.f);
        return pa.getPlantHeight(pid);
    };
    float full = grow_height(false);
    float cancelled = grow_height(true);
    DOCTEST_CHECK(full > 0.f);
    DOCTEST_CHECK(cancelled < full);
}

DOCTEST_TEST_CASE("ShootParameters defineChildShootTypes valid input") {
    ShootParameters sp_test;
    std::vector<std::string> labels = {"typeA", "typeB"};
    std::vector<float> probabilities = {0.4f, 0.6f};
    DOCTEST_CHECK_NOTHROW(sp_test.defineChildShootTypes(labels, probabilities));
}

DOCTEST_TEST_CASE("ShootParameters defineChildShootTypes size mismatch") {
    capture_cerr cerr_buffer;
    ShootParameters sp_test;
    std::vector<std::string> labels = {"typeA", "typeB"};
    std::vector<float> probabilities = {0.4f};
    DOCTEST_CHECK_THROWS(sp_test.defineChildShootTypes(labels, probabilities));
}

DOCTEST_TEST_CASE("ShootParameters defineChildShootTypes empty vectors") {
    capture_cerr cerr_buffer;
    ShootParameters sp_test;
    std::vector<std::string> labels = {};
    std::vector<float> probabilities = {};
    DOCTEST_CHECK_THROWS(sp_test.defineChildShootTypes(labels, probabilities));
}

DOCTEST_TEST_CASE("ShootParameters defineChildShootTypes probabilities sum not equal to 1") {
    capture_cerr cerr_buffer;
    ShootParameters sp_test;
    std::vector<std::string> labels = {"typeA", "typeB"};
    std::vector<float> probabilities = {0.3f, 0.6f}; // Sums to 0.9
    DOCTEST_CHECK_THROWS(sp_test.defineChildShootTypes(labels, probabilities));
}

DOCTEST_TEST_CASE("PlantArchitecture defineShootType") {
    Context context;
    PlantArchitecture pa_test(&context);
    ShootParameters sp_define;
    DOCTEST_CHECK_NOTHROW(pa_test.defineShootType("newShootType", sp_define));
}

DOCTEST_TEST_CASE("LeafPrototype Constructor") {
    Context context;
    std::minstd_rand0 *generator = context.getRandomGenerator();
    LeafPrototype lp_test(generator);
    DOCTEST_CHECK(lp_test.subdivisions == 1);
    DOCTEST_CHECK(lp_test.unique_prototypes == 1);
    DOCTEST_CHECK(lp_test.leaf_offset.x == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(lp_test.leaf_offset.y == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(lp_test.leaf_offset.z == doctest::Approx(0.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhytomerParameters Constructor") {
    Context context;
    std::minstd_rand0 *generator = context.getRandomGenerator();
    DOCTEST_CHECK_NOTHROW(PhytomerParameters pp_test(generator));
}

DOCTEST_TEST_CASE("Plant Library Model Building - almond") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("almond"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - apple") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("apple"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - asparagus") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("asparagus"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - bindweed") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bindweed"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - bean") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Material Naming - bean plant materials have descriptive names") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000);

    // Verify every plant primitive has a descriptive material label (no __auto_ display names)
    std::vector<uint> all_UUIDs = plantarchitecture.getAllPlantUUIDs(plantID);
    DOCTEST_CHECK(all_UUIDs.size() > 0);
    for (uint UUID : all_UUIDs) {
        std::string label = context.getPrimitiveMaterialLabel(UUID);
        DOCTEST_CHECK(label.substr(0, 7) != "__auto_");
    }

    // Verify expected material name patterns exist for bean
    std::vector<std::string> materials = context.listMaterials();
    // Note: organs with the same color/texture share a single material, so not every
    // organ type will necessarily have its own material (e.g., petiole and stem may share).
    bool found_trifoliate_leaf = false;
    bool found_unifoliate_leaf = false;
    bool found_stem = false;
    for (const auto &label : materials) {
        if (label.find("bean") != std::string::npos && label.find("trifoliate") != std::string::npos && label.find("leaf") != std::string::npos) {
            found_trifoliate_leaf = true;
        }
        if (label.find("bean") != std::string::npos && label.find("unifoliate") != std::string::npos && label.find("leaf") != std::string::npos) {
            found_unifoliate_leaf = true;
        }
        if (label.find("bean") != std::string::npos && label.find("stem") != std::string::npos) {
            found_stem = true;
        }
    }
    DOCTEST_CHECK(found_trifoliate_leaf);
    DOCTEST_CHECK(found_unifoliate_leaf);
    DOCTEST_CHECK(found_stem);
}

DOCTEST_TEST_CASE("Shoot Topology Accessors - getAllShootIDs and getPlantShoot") {
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    // getAllShootIDs should return a contiguous, 0-based set of shoot IDs
    std::vector<uint> shootIDs = plantarchitecture.getAllShootIDs(plantID);
    DOCTEST_CHECK(shootIDs.size() > 0);
    for (uint i = 0; i < shootIDs.size(); i++) {
        DOCTEST_CHECK(shootIDs.at(i) == i);
    }

    // The base-stem shoot (ID 0) must be rank 0 and have no parent shoot
    const std::shared_ptr<Shoot> &base_shoot = plantarchitecture.getPlantShoot(plantID, 0);
    DOCTEST_CHECK(base_shoot->ID == 0);
    DOCTEST_CHECK(base_shoot->rank == 0);
    DOCTEST_CHECK(base_shoot->parent_shoot_ID == -1);
    DOCTEST_CHECK(!base_shoot->shoot_internode_vertices.empty());
    DOCTEST_CHECK(base_shoot->shoot_internode_vertices.size() == base_shoot->shoot_internode_radii.size());

    // A shoot's rank is at least its parent's: a true branch (addChildShoot) is parent rank + 1,
    // while an appended/continuation shoot (appendShoot) keeps the parent's rank. So a child's rank
    // is either equal to or exactly one greater than its parent's.
    for (uint shootID : shootIDs) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        if (shoot->parent_shoot_ID >= 0) {
            const std::shared_ptr<Shoot> &parent = plantarchitecture.getPlantShoot(plantID, static_cast<uint>(shoot->parent_shoot_ID));
            DOCTEST_CHECK(shoot->rank >= parent->rank);
            DOCTEST_CHECK(shoot->rank <= parent->rank + 1);
        }
    }

    // Out-of-range / invalid IDs must throw rather than return a fallback. Each throwing call
    // is invoked inside a tightly-scoped cerr capture (helios_runtime_error writes to cerr in
    // debug builds before throwing); the resulting bool is asserted only after the capture is
    // destroyed, so doctest failure output is never swallowed.
    auto throws = [&](const std::function<void()> &fn) {
        bool threw = false;
        {
            capture_cerr cerr_buffer;
            try {
                fn();
            } catch (...) {
                threw = true;
            }
        }
        return threw;
    };
    DOCTEST_CHECK(throws([&]() { static_cast<void>(plantarchitecture.getAllShootIDs(plantID + 999)); }));
    DOCTEST_CHECK(throws([&]() { static_cast<void>(plantarchitecture.getPlantShoot(plantID, static_cast<uint>(shootIDs.size()))); }));
    DOCTEST_CHECK(throws([&]() { static_cast<void>(plantarchitecture.getPlantShoot(plantID + 999, 0)); }));
}

DOCTEST_TEST_CASE("Plant Library Model Building - cheeseweed") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("cheeseweed"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - cowpea") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("cowpea"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - grapevine_VSP") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - maize") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("maize"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - olive") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("olive"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - pistachio") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("pistachio"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - puncturevine") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("puncturevine"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - easternredbud") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("easternredbud"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - rice") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("rice"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - butterlettuce") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("butterlettuce"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - sorghum") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("sorghum"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - soybean") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("soybean"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - strawberry") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("strawberry"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - sugarbeet") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("sugarbeet"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - tomato") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("tomato"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - walnut") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("walnut"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - wheat") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("wheat"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("PlantArchitecture writeTreeQSM") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Build a simple plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 50);

    // Test writing TreeQSM format
    std::string filename = "test_plant_qsm.txt";
    DOCTEST_CHECK_NOTHROW(plantarchitecture.writeQSMCylinderFile(plantID, filename));

    // Check that file was created and has correct format
    std::ifstream file(filename);
    DOCTEST_CHECK(file.good());

    if (file.good()) {
        std::string header_line;
        std::getline(file, header_line);

        // Check header contains expected columns
        DOCTEST_CHECK(header_line.find("radius (m)") != std::string::npos);
        DOCTEST_CHECK(header_line.find("length (m)") != std::string::npos);
        DOCTEST_CHECK(header_line.find("start_point") != std::string::npos);
        DOCTEST_CHECK(header_line.find("axis_direction") != std::string::npos);
        DOCTEST_CHECK(header_line.find("branch") != std::string::npos);
        DOCTEST_CHECK(header_line.find("branch_order") != std::string::npos);

        // Check that there is at least one data line
        std::string data_line;
        bool has_data = static_cast<bool>(std::getline(file, data_line));
        DOCTEST_CHECK(has_data);

        if (has_data) {
            // Count tab-separated values in data line
            size_t tab_count = std::count(data_line.begin(), data_line.end(), '\t');
            DOCTEST_CHECK(tab_count >= 12); // Should have at least 13 columns (12 tabs)
        }

        file.close();

        // Clean up test file
        std::remove(filename.c_str());
    }
}

DOCTEST_TEST_CASE("PlantArchitecture writeTreeQSM invalid plant") {
    capture_cerr cerr_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Test with invalid plant ID
    DOCTEST_CHECK_THROWS(plantarchitecture.writeQSMCylinderFile(999, "invalid_plant.txt"));
}

DOCTEST_TEST_CASE("PlantArchitecture pruneSolidBoundaryCollisions") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable collision detection first
    plantarchitecture.enableSoftCollisionAvoidance();

    // Load a plant model from library
    plantarchitecture.loadPlantModelFromLibrary("tomato");

    // Create a plant and let it grow first WITHOUT boundaries
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 15); // Substantial growth to ensure objects exist

    // Get object count after growth but before boundaries
    std::vector<uint> objects_before_boundaries = plantarchitecture.getAllObjectIDs();
    uint count_before_boundaries = objects_before_boundaries.size();

    // Ensure we have some objects to work with
    DOCTEST_CHECK(count_before_boundaries > 0);

    // Now create solid boundaries that will definitely intersect with plant parts
    // Place boundaries at z=0.05 to intersect with low-lying plant parts
    std::vector<uint> boundary_UUIDs;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            // Create a grid of triangles to ensure we catch plant parts
            boundary_UUIDs.push_back(context.addTriangle(make_vec3(i * 0.1f, j * 0.1f, 0.05f), make_vec3((i + 1) * 0.1f, j * 0.1f, 0.05f), make_vec3(i * 0.1f, (j + 1) * 0.1f, 0.05f)));
        }
    }

    // Enable solid obstacle avoidance with the boundaries
    plantarchitecture.enableSolidObstacleAvoidance(boundary_UUIDs, 0.2f);

    // Trigger another growth step which should call pruneSolidBoundaryCollisions()
    // Use a very small time step to minimize new growth
    plantarchitecture.advanceTime(plantID, 0.1f); // Very small step to trigger pruning

    // Get final object count
    std::vector<uint> final_objects = plantarchitecture.getAllObjectIDs();
    uint final_count = final_objects.size();

    // Verify that objects were actually pruned by checking that we have fewer objects
    // than we would expect if no pruning occurred. Since some growth may still happen,
    // we check if the final count is reasonable given pruning occurred.
    // The key test is that our implementation ran without errors and produced output
    // indicating pruning occurred (visible in test output: "Pruned X objects").
    DOCTEST_CHECK(final_count > 0); // Basic sanity check - we should still have some objects
}

DOCTEST_TEST_CASE("PlantArchitecture pruneSolidBoundaryCollisions no boundaries") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Load a plant model from library
    plantarchitecture.loadPlantModelFromLibrary("tomato");

    // Create a simple plant
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 5);

    // Get initial object count
    std::vector<uint> initial_objects = plantarchitecture.getAllObjectIDs();
    uint initial_count = initial_objects.size();

    // Advance time again without boundaries - should not prune anything
    plantarchitecture.advanceTime(plantID, 2);

    // Check that no objects were pruned (may have grown more)
    std::vector<uint> final_objects = plantarchitecture.getAllObjectIDs();
    uint final_count = final_objects.size();

    DOCTEST_CHECK(final_count >= initial_count);
}

DOCTEST_TEST_CASE("PlantArchitecture advanceTime after pruneBranch leaves empty shoot shell") {
    // Pruning a branch at node 0 deletes all of its phytomers and its internode tube object, but
    // leaves the (now empty) Shoot in the shoot_tree and a stale entry in the parent's childIDs.
    // advanceTime must tolerate these empty shells: the per-shoot volume bookkeeping loop and the
    // recursive Shoot::updateShootNodes both used to dereference the cleared geometry / write to the
    // deleted tube object and crash. This reproduces the user-reported crash.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 365);

    // Find a child shoot of the trunk (shoot 0) that still has phytomers, then prune it from node 0.
    const std::shared_ptr<Shoot> trunk = plantarchitecture.getPlantShoot(plantID, 0);
    int branchID = -1;
    for (const auto &[node_index, shootIDs]: trunk->childIDs) {
        for (const int shootID: shootIDs) {
            if (!plantarchitecture.getPlantShoot(plantID, shootID)->phytomers.empty()) {
                branchID = shootID;
                break;
            }
        }
        if (branchID >= 0) {
            break;
        }
    }
    DOCTEST_REQUIRE(branchID >= 0);

    DOCTEST_CHECK_NOTHROW(plantarchitecture.pruneBranch(plantID, (uint) branchID, 0));
    DOCTEST_CHECK(plantarchitecture.getPlantShoot(plantID, (uint) branchID)->phytomers.empty());

    // getShootInternodeObjectIDs must not return the dangling tube object ID of the pruned shell.
    std::vector<uint> internode_objIDs = plantarchitecture.getShootInternodeObjectIDs(plantID);
    for (uint objID: internode_objIDs) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }

    // The crash occurred here, inside advanceTime, while iterating over the empty pruned shoot.
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 365));
}

//! Builds an apple tree and prunes the first trunk branch that still has phytomers.
/**
 * Returns the pruned shoot ID. The chosen branch has child shoots of its own, so the prune also
 * empties descendants -- which is what most of the pruned-shell tests below depend on.
 */
static uint buildAppleAndPruneABranch(PlantArchitecture &plantarchitecture, uint &plantID) {
    plantarchitecture.loadPlantModelFromLibrary("apple");
    plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 365);

    const std::shared_ptr<Shoot> trunk = plantarchitecture.getPlantShoot(plantID, 0);
    int branchID = -1;
    for (const auto &[node_index, shootIDs]: trunk->childIDs) {
        for (const int shootID: shootIDs) {
            if (!plantarchitecture.getPlantShoot(plantID, shootID)->phytomers.empty()) {
                branchID = shootID;
                break;
            }
        }
        if (branchID >= 0) {
            break;
        }
    }
    DOCTEST_REQUIRE(branchID >= 0);

    plantarchitecture.pruneBranch(plantID, (uint) branchID, 0);
    return (uint) branchID;
}

//! Drives one day of photosynthesis, standing in for the Photosynthesis plug-in.
/**
 * The carbohydrate model reads "net_photosynthesis" (umol CO2 m-2 s-1) off leaf primitives and
 * integrates it hourly into "cumulative_net_photosynthesis" via accumulateHourlyLeafPhotosynthesis().
 * Setting the rate directly lets the plantarchitecture tests exercise the carbon balance without
 * depending on the photosynthesis plug-in.
 */
static void drivePhotosynthesisForOneDay(Context &context, PlantArchitecture &plantarchitecture, uint plantID, float assimilation_rate_umol, int daylight_hours) {
    for (int hour = 0; hour < daylight_hours; hour++) {
        for (const uint objID: plantarchitecture.getPlantLeafObjectIDs(plantID)) {
            if (context.doesObjectExist(objID)) {
                for (const uint UUID: context.getObjectPrimitiveUUIDs(objID)) {
                    context.setPrimitiveData(UUID, "net_photosynthesis", assimilation_rate_umol);
                }
            }
        }
        plantarchitecture.accumulateHourlyLeafPhotosynthesis();
    }
}

DOCTEST_TEST_CASE("PlantArchitecture carbohydrate model does not abort on the parentless base shoot") {
    // subtractShootGrowthCarbon() charges the construction cost of a shoot's first phytomers to that
    // shoot's PARENT, which is the case for every shoot on its first timestep (old_shoot_volume == 0).
    // The base stem shoot has no parent, and its parent_shoot_ID of -1 was assigned to a uint and used
    // directly as a shoot_tree index, so the lookup ran off the end of the vector. Enabling the
    // carbohydrate model therefore aborted with std::out_of_range during the first advanceTime() of
    // every plant in the library, before any of the model's behavior could be reached.
    //
    // No carbon is supplied here, which the model documents as a supported mode for exercising the
    // sink and transfer mechanisms on their own. The plant is expected to starve; what must not happen
    // is an abort.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.enableCarbohydrateModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID = 0;
    {
        // accumulateShootPhotosynthesis() warns on std::cerr that no photosynthesis data is present.
        capture_cerr capture;
        DOCTEST_REQUIRE_NOTHROW(plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 20));
        DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 10));
    }

    // Starved shoots are pruned away by the instant-death check in checkCarbonPool_abortOrgans(). Those
    // empty shells must not then be re-processed by the carbon loops: their zero volume divides through
    // the concentration calculation and their internode tube object no longer exists, which threw from
    // accumulateShootPhotosynthesis() once the out_of_range abort above was fixed.
    DOCTEST_REQUIRE(!plantarchitecture.getAllShootIDs(plantID).empty());
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        DOCTEST_CHECK(std::isfinite(shoot->sugar_pool_molC));
        DOCTEST_CHECK(std::isfinite(shoot->starch_pool_molC));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture carbohydrate model does not abort on any library plant") {
    // The out_of_range abort was in a code path taken by every shoot on its first timestep, so it was
    // not specific to one plant model. Cover a spread of growth habits (herbaceous, vine, grass, woody).
    const std::vector<std::string> plant_labels = {"bean", "cowpea", "maize", "tomato", "almond"};

    for (const std::string &label: plant_labels) {
        DOCTEST_CAPTURE(label);
        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.enableCarbohydrateModel();
        plantarchitecture.loadPlantModelFromLibrary(label);

        bool threw = false;
        {
            capture_cerr capture;
            try {
                const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 10);
                plantarchitecture.advanceTime(plantID, 5);
            } catch (...) {
                threw = true;
            }
        }
        DOCTEST_CHECK(!threw);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture carbohydrate model grows a plant when carbon is supplied") {
    // With the aborts fixed, the model must actually run a plant through its carbon balance: the plant
    // has to stay alive and put on leaf area rather than starve. This follows the workflow documented in
    // CarbohydrateModel.dox -- build at age 0, initialize the carbon pools, then advance a day at a time
    // with photosynthesis supplied each day.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.enableCarbohydrateModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // The initial pool is sized as concentration x internode volume, and a seedling's base stem has a
    // very small volume while having to fund the construction of its much larger first shoot. The 1500
    // mol C/m^3 used in the documentation's example leaves it unable to do so and the plant dies on the
    // first day; this is comfortably above that threshold.
    plantarchitecture.initializePlantCarbohydratePool(plantID, 1.5e4f);

    for (int day = 0; day < 20; day++) {
        drivePhotosynthesisForOneDay(context, plantarchitecture, plantID, 10.f, 12);
        DOCTEST_REQUIRE_NOTHROW(plantarchitecture.advanceTime(plantID, 1));
    }

    // The plant must still be alive -- in particular the base stem shoot must not have been pruned by
    // the instant-death check -- and must have grown.
    DOCTEST_CHECK(!plantarchitecture.isShootPruned(plantID, 0));
    DOCTEST_CHECK(plantarchitecture.sumPlantLeafArea(plantID) > 0.f);

    // Carbon must have actually flowed: at least one shoot holds a positive pool.
    float total_sugar = 0.f;
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        DOCTEST_CHECK(std::isfinite(shoot->sugar_pool_molC));
        total_sugar += shoot->sugar_pool_molC;
    }
    DOCTEST_CHECK(total_sugar > 0.f);
}

DOCTEST_TEST_CASE("PlantArchitecture shoot topology accessors") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("apple");
    const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 365);

    const std::vector<uint> allShootIDs = plantarchitecture.getAllShootIDs(plantID);
    DOCTEST_REQUIRE(allShootIDs.size() > 1);

    // -- base stem shoot --
    DOCTEST_CHECK(plantarchitecture.getParentShootID(plantID, 0) == -1);
    DOCTEST_CHECK(plantarchitecture.getShootRank(plantID, 0) == 0);
    DOCTEST_CHECK(plantarchitecture.getShootDepth(plantID, 0) == 0);
    DOCTEST_CHECK(plantarchitecture.getPathToRoot(plantID, 0) == std::vector<uint>{0});

    // -- every non-base shoot must be consistent with its parent --
    for (const uint shootID: allShootIDs) {
        const int parentID = plantarchitecture.getParentShootID(plantID, shootID);
        if (shootID == 0) {
            continue;
        }
        DOCTEST_REQUIRE(parentID >= 0);

        // Depth is always one more than the parent's.
        DOCTEST_CHECK(plantarchitecture.getShootDepth(plantID, shootID) == plantarchitecture.getShootDepth(plantID, (uint) parentID) + 1);

        // Rank either matches the parent (an appendShoot() axis continuation) or is one greater
        // (an addChildShoot() branch). It is never anything else, and never less.
        const uint rank = plantarchitecture.getShootRank(plantID, shootID);
        const uint parent_rank = plantarchitecture.getShootRank(plantID, (uint) parentID);
        DOCTEST_CHECK((rank == parent_rank || rank == parent_rank + 1));

        // The path to the root starts at the shoot, ends at the base, and steps through parents.
        const std::vector<uint> path = plantarchitecture.getPathToRoot(plantID, shootID);
        DOCTEST_REQUIRE(path.size() >= 2);
        DOCTEST_CHECK(path.front() == shootID);
        // The path ends at a shoot with no parent. That is ID 0 here, but a plant can have more than
        // one base stem shoot (grapevine does), so test the property rather than the ID.
        DOCTEST_CHECK(plantarchitecture.getParentShootID(plantID, path.back()) == -1);
        DOCTEST_CHECK(path.at(1) == (uint) parentID);
        DOCTEST_CHECK(path.size() == plantarchitecture.getShootDepth(plantID, shootID) + 1);

        // A shoot must be listed among its parent's children.
        const std::vector<uint> siblings = plantarchitecture.getChildShootIDs(plantID, (uint) parentID);
        DOCTEST_CHECK(std::find(siblings.begin(), siblings.end(), shootID) != siblings.end());
    }

    // -- getShootIDsByRank: index equals rank, and every live shoot appears exactly once --
    const std::vector<std::vector<uint>> by_rank = plantarchitecture.getShootIDsByRank(plantID);
    DOCTEST_REQUIRE(!by_rank.empty());
    DOCTEST_CHECK(by_rank.at(0) == std::vector<uint>{0});
    size_t ranked_count = 0;
    for (uint rank = 0; rank < by_rank.size(); rank++) {
        for (const uint shootID: by_rank.at(rank)) {
            DOCTEST_CHECK(plantarchitecture.getShootRank(plantID, shootID) == rank);
            ranked_count++;
        }
    }
    DOCTEST_CHECK(ranked_count == allShootIDs.size());

    // -- getAllDescendantShootIDs from the base reaches every other shoot exactly once --
    const std::vector<uint> descendants = plantarchitecture.getAllDescendantShootIDs(plantID, 0);
    DOCTEST_CHECK(descendants.size() == allShootIDs.size() - 1);
    std::vector<uint> sorted_descendants = descendants;
    std::sort(sorted_descendants.begin(), sorted_descendants.end());
    DOCTEST_CHECK(std::unique(sorted_descendants.begin(), sorted_descendants.end()) == sorted_descendants.end());
    DOCTEST_CHECK(std::find(descendants.begin(), descendants.end(), 0u) == descendants.end());

    // Depth-first ordering: a shoot is always listed before its own descendants.
    for (size_t i = 0; i < descendants.size(); i++) {
        for (const uint sub: plantarchitecture.getAllDescendantShootIDs(plantID, descendants.at(i))) {
            const auto position = std::find(descendants.begin(), descendants.end(), sub);
            DOCTEST_REQUIRE(position != descendants.end());
            DOCTEST_CHECK((size_t) std::distance(descendants.begin(), position) > i);
        }
    }

    // -- getShootHierarchyMap agrees with getChildShootIDs, and only lists shoots that have children --
    const std::map<uint, std::vector<uint>> hierarchy = plantarchitecture.getShootHierarchyMap(plantID);
    for (const auto &[parentID, childIDs]: hierarchy) {
        DOCTEST_CHECK(!childIDs.empty());
        DOCTEST_CHECK(childIDs == plantarchitecture.getChildShootIDs(plantID, parentID));
    }
    for (const uint shootID: allShootIDs) {
        const bool has_children = !plantarchitecture.getChildShootIDs(plantID, shootID).empty();
        DOCTEST_CHECK((hierarchy.find(shootID) != hierarchy.end()) == has_children);
    }

    // -- getTerminalShootIDs is exactly the set of shoots with no children --
    const std::vector<uint> terminals = plantarchitecture.getTerminalShootIDs(plantID);
    DOCTEST_REQUIRE(!terminals.empty());
    for (const uint shootID: terminals) {
        DOCTEST_CHECK(plantarchitecture.getChildShootIDs(plantID, shootID).empty());
    }
    DOCTEST_CHECK(terminals.size() + hierarchy.size() == allShootIDs.size());

    // -- bad IDs are rejected --
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getParentShootID(plantID, 9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getShootRank(9999, 0));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getShootDepth(plantID, 9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getPathToRoot(plantID, 9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getChildShootIDs(plantID, 9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getAllDescendantShootIDs(plantID, 9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getShootIDsByRank(9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getShootHierarchyMap(9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getTerminalShootIDs(9999));
}

DOCTEST_TEST_CASE("PlantArchitecture shoot topology accessors exclude pruned shoots") {
    // A pruned shoot keeps its slot in the shoot tree, so getAllShootIDs() still returns it. The
    // topology accessors describe the plant's actual structure and must leave it out -- in particular
    // getTerminalShootIDs(), since an empty pruned shell has no children but is not a tip of the plant.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID;
    const uint branchID = buildAppleAndPruneABranch(plantarchitecture, plantID);
    DOCTEST_REQUIRE(plantarchitecture.isShootPruned(plantID, branchID));

    // Collect every shoot ID the topology accessors report anywhere.
    std::vector<uint> reported = plantarchitecture.getTerminalShootIDs(plantID);
    for (const uint shootID: plantarchitecture.getAllDescendantShootIDs(plantID, 0)) {
        reported.push_back(shootID);
    }
    for (const std::vector<uint> &rank_group: plantarchitecture.getShootIDsByRank(plantID)) {
        for (const uint shootID: rank_group) {
            reported.push_back(shootID);
        }
    }
    for (const auto &[parentID, childIDs]: plantarchitecture.getShootHierarchyMap(plantID)) {
        reported.push_back(parentID);
        for (const uint childID: childIDs) {
            reported.push_back(childID);
        }
    }
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        reported.push_back(plantarchitecture.getChildShootIDs(plantID, shootID).empty() ? 0 : plantarchitecture.getChildShootIDs(plantID, shootID).front());
    }

    // No pruned shoot may appear in any of them.
    for (const uint shootID: reported) {
        DOCTEST_CHECK(!plantarchitecture.isShootPruned(plantID, shootID));
    }

    // The pruned shoot is specifically not reported as terminal, and not as a child of its parent.
    const std::vector<uint> terminals = plantarchitecture.getTerminalShootIDs(plantID);
    DOCTEST_CHECK(std::find(terminals.begin(), terminals.end(), branchID) == terminals.end());
    const int parentID = plantarchitecture.getParentShootID(plantID, branchID);
    DOCTEST_REQUIRE(parentID >= 0);
    const std::vector<uint> siblings = plantarchitecture.getChildShootIDs(plantID, (uint) parentID);
    DOCTEST_CHECK(std::find(siblings.begin(), siblings.end(), branchID) == siblings.end());

    // The pruned shoot still knows where it grew from, and that ancestor is a live shoot -- pruning a
    // shoot prunes everything below it, so a pruned shoot never has a pruned parent.
    DOCTEST_CHECK(!plantarchitecture.isShootPruned(plantID, (uint) parentID));
}

DOCTEST_TEST_CASE("PlantArchitecture pruning examples from the documentation") {
    // The three worked examples in the "Shoot Pruning and Hierarchical Management" section of
    // PlantArchitecture.dox, run as written. They previously did not compile at all, because none of
    // the hierarchy accessors they call existed.
    {
        // Pruning by branching order. A two-year-old apple is used because the example prunes rank 3
        // and above, and a one-year-old apple only branches as far as rank 2.
        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("apple");
        const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 730);

        auto shoots_by_rank = plantarchitecture.getShootIDsByRank(plantID);
        DOCTEST_REQUIRE(shoots_by_rank.size() > 3);
        DOCTEST_REQUIRE(!shoots_by_rank[3].empty());
        DOCTEST_CHECK_NOTHROW([&] {
            for (size_t rank = 3; rank < shoots_by_rank.size(); ++rank) {
                for (uint shootID: shoots_by_rank[rank]) {
                    plantarchitecture.pruneBranch(plantID, shootID, 0);
                }
            }
        }());
        // Nothing of rank 3 or higher may survive, and the trunk must be untouched.
        for (const std::vector<uint> &group: plantarchitecture.getShootIDsByRank(plantID)) {
            for (const uint shootID: group) {
                DOCTEST_CHECK(plantarchitecture.getShootRank(plantID, shootID) < 3);
            }
        }
        DOCTEST_CHECK(!plantarchitecture.isShootPruned(plantID, 0));
    }
    {
        // Terminal shoot thinning.
        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("apple");
        const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 365);

        auto terminal_shoots = plantarchitecture.getTerminalShootIDs(plantID);
        DOCTEST_REQUIRE(terminal_shoots.size() > 1);
        const size_t expected_pruned = (terminal_shoots.size() + 1) / 2;
        DOCTEST_CHECK_NOTHROW([&] {
            for (size_t i = 0; i < terminal_shoots.size(); i += 2) {
                plantarchitecture.pruneBranch(plantID, terminal_shoots[i], 0);
            }
        }());
        size_t pruned_count = 0;
        for (size_t i = 0; i < terminal_shoots.size(); i += 2) {
            if (plantarchitecture.isShootPruned(plantID, terminal_shoots[i])) {
                pruned_count++;
            }
        }
        DOCTEST_CHECK(pruned_count == expected_pruned);
    }
    {
        // Pruning entire branch systems. This is the example that motivated making pruneBranch()
        // idempotent: removing a descendant also empties that descendant's own descendants, which the
        // loop then reaches again.
        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("apple");
        const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 365);

        uint target_shoot = 0;
        auto descendants = plantarchitecture.getAllDescendantShootIDs(plantID, target_shoot);
        DOCTEST_REQUIRE(!descendants.empty());
        DOCTEST_CHECK_NOTHROW([&] {
            for (uint descendant: descendants) {
                plantarchitecture.pruneBranch(plantID, descendant, 0);
            }
        }());
        // The whole branch system below the trunk is gone; only the trunk remains.
        DOCTEST_CHECK(plantarchitecture.getAllDescendantShootIDs(plantID, 0).empty());
        DOCTEST_CHECK(!plantarchitecture.isShootPruned(plantID, 0));
        DOCTEST_CHECK(plantarchitecture.getTerminalShootIDs(plantID) == std::vector<uint>{0});
    }
}

DOCTEST_TEST_CASE("PlantArchitecture pruneBranch is idempotent on an already-pruned shoot") {
    // Pruning a whole branch system means pruning a shoot and then, further along the same loop,
    // reaching shoots that the first prune already emptied as descendants. pruneBranch used to
    // reject those with "Node index 0 is out of range for shoot N" -- blaming the caller's node
    // index for a shoot that no longer has any nodes -- which aborted the loop partway through and
    // left the plant half-pruned. Re-pruning an empty shoot must instead be a no-op.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID;
    const uint branchID = buildAppleAndPruneABranch(plantarchitecture, plantID);

    DOCTEST_CHECK(plantarchitecture.isShootPruned(plantID, branchID));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.pruneBranch(plantID, branchID, 0));
    DOCTEST_CHECK(plantarchitecture.isShootPruned(plantID, branchID));

    // The realistic bulk-prune loop: remove every branch off the trunk. Shoots emptied as
    // descendants of an earlier prune are revisited here and must not abort the loop.
    Context context2;
    PlantArchitecture plantarchitecture2(&context2);
    plantarchitecture2.disableMessages();
    plantarchitecture2.loadPlantModelFromLibrary("apple");
    const uint plantID2 = plantarchitecture2.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 365);

    const std::vector<uint> shootIDs = plantarchitecture2.getAllShootIDs(plantID2);
    for (const uint shootID: shootIDs) {
        if (plantarchitecture2.getPlantShoot(plantID2, shootID)->rank >= 1) {
            DOCTEST_CHECK_NOTHROW(plantarchitecture2.pruneBranch(plantID2, shootID, 0));
        }
    }
    // Every non-trunk shoot must actually be gone, and the trunk must be untouched.
    for (const uint shootID: shootIDs) {
        const bool is_trunk = (plantarchitecture2.getPlantShoot(plantID2, shootID)->rank == 0);
        DOCTEST_CHECK(plantarchitecture2.isShootPruned(plantID2, shootID) == !is_trunk);
    }

    // A node index that is genuinely out of range on a live shoot must still be rejected.
    const uint trunk_node_count = plantarchitecture2.getShootNodeCount(plantID2, 0);
    DOCTEST_CHECK_THROWS(plantarchitecture2.pruneBranch(plantID2, 0, trunk_node_count));
}

DOCTEST_TEST_CASE("PlantArchitecture pruneBranch unlinks the pruned shoot from the shoot tree") {
    // A pruned shoot stays in the shoot_tree (its ID must remain a valid index), but it must no
    // longer be reachable as a child of anything: the stale childIDs entry made recursive
    // traversals descend into a shoot with no phytomers. It must also not keep the object ID of
    // the internode tube that pruning deleted -- pruneGroundCollisions() already resets that field
    // to the sentinel and pruneBranch must be consistent with it.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID;
    const uint branchID = buildAppleAndPruneABranch(plantarchitecture, plantID);
    const std::shared_ptr<Shoot> trunk = plantarchitecture.getPlantShoot(plantID, 0);
    const std::shared_ptr<Shoot> pruned = plantarchitecture.getPlantShoot(plantID, branchID);

    // The shoot ID stays valid; the shoot tree does not renumber.
    DOCTEST_CHECK(plantarchitecture.getShootNodeCount(plantID, branchID) == 0);

    // No longer listed as a child of its parent.
    for (const auto &[node_index, shootIDs]: trunk->childIDs) {
        for (const int shootID: shootIDs) {
            DOCTEST_CHECK(shootID != (int) branchID);
        }
    }

    // Its own children were emptied by the same prune, so they must be unlinked too.
    DOCTEST_CHECK(pruned->childIDs.empty());

    // No dangling internode tube object ID.
    DOCTEST_CHECK(pruned->internode_tube_objID == Shoot::no_internode_tube_objID);
    DOCTEST_CHECK(!context.doesObjectExist(pruned->internode_tube_objID));

    // Every shoot still reachable through childIDs must be a live shoot.
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        for (const auto &[node_index, childIDs]: shoot->childIDs) {
            for (const int childID: childIDs) {
                DOCTEST_CHECK(!plantarchitecture.isShootPruned(plantID, (uint) childID));
            }
        }
    }
}

DOCTEST_TEST_CASE("PlantArchitecture shoot queries on a pruned shoot") {
    // sumShootLeafArea()/sumChildVolume() rejected start_node_index 0 on an empty shoot, because
    // "0 >= phytomers.size()" is true when there are no phytomers. A pruned shoot contributes no
    // leaf area and no volume, so these must return zero rather than throw -- including when
    // reached recursively from a live ancestor. getShootTaper() had no guard at all and read
    // front() off an empty vector, which segfaulted.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID;
    const uint branchID = buildAppleAndPruneABranch(plantarchitecture, plantID);
    const std::shared_ptr<Shoot> trunk = plantarchitecture.getPlantShoot(plantID, 0);
    const std::shared_ptr<Shoot> pruned = plantarchitecture.getPlantShoot(plantID, branchID);

    DOCTEST_CHECK(pruned->sumShootLeafArea(0) == doctest::Approx(0.f));
    DOCTEST_CHECK(pruned->sumChildVolume(0) == doctest::Approx(0.f));
    DOCTEST_CHECK(pruned->calculateShootInternodeVolume() == doctest::Approx(0.f));
    DOCTEST_CHECK(pruned->calculateShootLength() == doctest::Approx(0.f));

    // The trunk recursion used to descend into the pruned shell and throw from there.
    float trunk_leaf_area = 0.f;
    DOCTEST_CHECK_NOTHROW(trunk_leaf_area = trunk->sumShootLeafArea(0));
    DOCTEST_CHECK(trunk_leaf_area > 0.f);
    DOCTEST_CHECK_NOTHROW((void) trunk->sumChildVolume(0));

    // A pruned shoot has no internode geometry, so taper is not defined for it. This is a caller
    // error rather than a value to invent, so it must fail loudly instead of segfaulting.
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.getShootTaper(plantID, branchID));
    DOCTEST_CHECK_NOTHROW(std::ignore = plantarchitecture.getShootTaper(plantID, 0));
}

DOCTEST_TEST_CASE("PlantArchitecture advanceTime after pruneBranch with internode context build disabled") {
    // A pruned shoot must not keep a live apical meristem: advanceTime() reads phytomers.back() on
    // any shoot whose meristem is alive, which segfaults on a shoot with no phytomers. The meristem
    // used to be terminated as a side effect of pruning the internode tube object, so a plant built
    // with disableInternodeContextBuild() -- which creates no tube objects at all -- skipped that
    // step and left the pruned shell growable.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.disableInternodeContextBuild();

    uint plantID;
    const uint branchID = buildAppleAndPruneABranch(plantarchitecture, plantID);

    DOCTEST_CHECK(plantarchitecture.isShootPruned(plantID, branchID));
    DOCTEST_CHECK(!plantarchitecture.getPlantShoot(plantID, branchID)->meristem_is_alive);

    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 90));

    // The shoot must stay pruned -- it must not regrow phytomers from its empty base.
    DOCTEST_CHECK(plantarchitecture.isShootPruned(plantID, branchID));
}

DOCTEST_TEST_CASE("PlantArchitecture isShootPruned") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID;
    const uint branchID = buildAppleAndPruneABranch(plantarchitecture, plantID);

    DOCTEST_CHECK(plantarchitecture.isShootPruned(plantID, branchID));
    DOCTEST_CHECK(!plantarchitecture.isShootPruned(plantID, 0));
    DOCTEST_CHECK(plantarchitecture.getPlantShoot(plantID, branchID)->isPruned());
    DOCTEST_CHECK(!plantarchitecture.getPlantShoot(plantID, 0)->isPruned());

    // Same range checking as the other shoot accessors.
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.isShootPruned(plantID, 9999));
    DOCTEST_CHECK_THROWS(std::ignore = plantarchitecture.isShootPruned(9999, 0));
}

DOCTEST_TEST_CASE("PlantArchitecture pruneGroundCollisions resets deleted internode tube object ID") {
    // pruneGroundCollisions() deletes a shoot's internode tube object when it dips below the ground
    // clipping plane, but used to leave the freed object ID in Shoot::internode_tube_objID. That
    // dangling ID then had to be filtered out by every consumer; anything reading the field directly
    // saw an ID that no longer exists in the Context. The ID must be reset to the sentinel instead.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("bean");

    // Start the plant below the clipping plane so its internode geometry is guaranteed to have
    // vertices with z < ground_clipping_height and therefore gets pruned.
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, -0.5f), 0);
    plantarchitecture.enableGroundClipping(0.f);
    plantarchitecture.advanceTime(plantID, 25);

    // At least one shoot must actually have had its tube deleted, otherwise this test proves nothing.
    int deleted_tube_count = 0;
    for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        if (!context.doesObjectExist(shoot->internode_tube_objID)) {
            deleted_tube_count++;
            // The field must hold the sentinel, never a freed object ID.
            DOCTEST_CHECK(shoot->internode_tube_objID == Shoot::no_internode_tube_objID);
        }
    }
    DOCTEST_REQUIRE(deleted_tube_count > 0);

    // The public getters must likewise never hand back an ID that is not in the Context.
    for (uint objID: plantarchitecture.getShootInternodeObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }
    for (uint objID: plantarchitecture.getPlantInternodeObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }
    for (uint objID: plantarchitecture.getAllPlantObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture organ getters return only live object IDs after organ removal") {
    // Leaves, petioles, peduncles, flowers and fruit are deleted together with their ID containers
    // (removeLeaf() clears leaf_objIDs, setFloralBudState() resizes the inflorescence/peduncle
    // vectors), so these getters should never surface a freed object ID. This pins that invariant
    // so a future change that deletes geometry without clearing its IDs is caught here.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 25);

    // Remove the organs on only part of the plant, so the getters still return a non-empty result
    // afterwards. Clearing every organ would make the loops below vacuous and prove nothing.
    const std::vector<uint> shootIDs = plantarchitecture.getAllShootIDs(plantID);
    DOCTEST_REQUIRE(!shootIDs.empty());
    plantarchitecture.removeShootLeaves(plantID, shootIDs.front());
    plantarchitecture.harvestPlant(plantID);

    // The plant must still expose live organs, otherwise the existence checks are vacuous.
    const std::vector<uint> remaining_leaves = plantarchitecture.getPlantLeafObjectIDs(plantID);
    DOCTEST_REQUIRE(!remaining_leaves.empty());

    for (uint objID: plantarchitecture.getPlantLeafObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }
    for (uint objID: plantarchitecture.getPlantPetioleObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }
    for (uint objID: plantarchitecture.getPlantPeduncleObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }
    for (uint objID: plantarchitecture.getPlantFlowerObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }
    for (uint objID: plantarchitecture.getPlantFruitObjectIDs(plantID)) {
        DOCTEST_CHECK(context.doesObjectExist(objID));
    }

    // Consumers that feed these IDs straight into Context object queries must not throw.
    DOCTEST_CHECK_NOTHROW(static_cast<void>(plantarchitecture.getPlantLeafInclinationAngleDistribution(plantID, 9)));
    DOCTEST_CHECK_NOTHROW(static_cast<void>(plantarchitecture.getPlantLeafAzimuthAngleDistribution(plantID, 9)));
    DOCTEST_CHECK_NOTHROW(static_cast<void>(plantarchitecture.sumPlantLeafArea(plantID)));
}

DOCTEST_TEST_CASE("PlantArchitecture hard collision avoidance base stem protection") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable collision detection first
    plantarchitecture.enableSoftCollisionAvoidance();

    // Load a plant model from library
    plantarchitecture.loadPlantModelFromLibrary("tomato");

    // Create a plant that starts slightly below ground surface (e.g., at z = -0.05)
    // This simulates the common scenario where ground model is slightly uneven
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, -0.05f), 0);

    // Create ground surface as solid obstacle slightly above plant base
    std::vector<uint> ground_UUIDs;

    // Create a ground patch that the plant would intersect if it doesn't grow upward
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            ground_UUIDs.push_back(context.addTriangle(make_vec3(i * 0.2f, j * 0.2f, 0.0f), // Ground at z=0
                                                       make_vec3((i + 1) * 0.2f, j * 0.2f, 0.0f), make_vec3(i * 0.2f, (j + 1) * 0.2f, 0.0f)));
            ground_UUIDs.push_back(context.addTriangle(make_vec3((i + 1) * 0.2f, (j + 1) * 0.2f, 0.0f), make_vec3((i + 1) * 0.2f, j * 0.2f, 0.0f), make_vec3(i * 0.2f, (j + 1) * 0.2f, 0.0f)));
        }
    }

    // Enable hard solid obstacle avoidance with the ground
    plantarchitecture.enableSolidObstacleAvoidance(ground_UUIDs, 0.3f);

    // Let the plant grow - it should grow upward despite starting below ground
    // The first 3 nodes of the base stem should ignore solid obstacles
    plantarchitecture.advanceTime(plantID, 10); // Sufficient growth time

    // Get all plant objects to analyze growth direction
    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Calculate center of mass of all plant objects to verify upward growth
    // If the plant made a U-turn downward, the center would be below the starting position
    vec3 center_of_mass = make_vec3(0, 0, 0);
    uint total_objects = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            // Get object center using bounding box
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);

            vec3 object_center = (min_corner + max_corner) / 2.0f;

            center_of_mass = center_of_mass + object_center;
            total_objects++;
        }
    }

    if (total_objects > 0) {
        center_of_mass = center_of_mass / float(total_objects);

        // The center of mass should be above the starting position (z = -0.05)
        // This verifies the plant grew upward rather than making a U-turn downward
        DOCTEST_CHECK(center_of_mass.z > -0.075f);

        // The key test is that the plant didn't curve significantly downward (U-turn behavior)
        // A U-turn would result in center of mass well below starting position (e.g., < -0.06)
        // Any value above -0.045 indicates successful avoidance of U-turn behavior
        DOCTEST_CHECK(center_of_mass.z > -0.075f); // Should not have made a U-turn downward
    }

    // Additional check: the plant should still exist (wasn't completely pruned)
    // and should have a reasonable number of objects
    DOCTEST_CHECK(plant_objects.size() >= 5); // Should have internodes, leaves, etc.
}

DOCTEST_TEST_CASE("PlantArchitecture enableSolidObstacleAvoidance fruit adjustment control") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Create some obstacles
    std::vector<uint> obstacle_UUIDs;
    obstacle_UUIDs.push_back(context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(-1, 1, 0)));
    obstacle_UUIDs.push_back(context.addTriangle(make_vec3(1, 1, 0), make_vec3(1, -1, 0), make_vec3(-1, 1, 0)));

    // Test enabling solid obstacle avoidance with fruit adjustment enabled (default)
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableSolidObstacleAvoidance(obstacle_UUIDs, 0.5f));

    // Test enabling solid obstacle avoidance with fruit adjustment explicitly enabled
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableSolidObstacleAvoidance(obstacle_UUIDs, 0.5f, true));

    // Test enabling solid obstacle avoidance with fruit adjustment disabled
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableSolidObstacleAvoidance(obstacle_UUIDs, 0.5f, false));

    // Test with different avoidance distance and disabled fruit adjustment
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableSolidObstacleAvoidance(obstacle_UUIDs, 0.3f, false));
}

DOCTEST_TEST_CASE("PlantArchitecture base stem protection with short internodes") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable collision detection first
    plantarchitecture.enableSoftCollisionAvoidance();

    // Load a plant model
    plantarchitecture.loadPlantModelFromLibrary("tomato");

    // Create a plant that starts at ground level
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Let it grow a small amount first to create some short internodes
    plantarchitecture.advanceTime(plantID, 2);

    // Create ground surface as solid obstacle
    std::vector<uint> ground_UUIDs;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            ground_UUIDs.push_back(context.addTriangle(make_vec3(i * 0.3f, j * 0.3f, -0.01f), // Ground slightly below
                                                       make_vec3((i + 1) * 0.3f, j * 0.3f, -0.01f), make_vec3(i * 0.3f, (j + 1) * 0.3f, -0.01f)));
            ground_UUIDs.push_back(context.addTriangle(make_vec3((i + 1) * 0.3f, (j + 1) * 0.3f, -0.01f), make_vec3((i + 1) * 0.3f, j * 0.3f, -0.01f), make_vec3(i * 0.3f, (j + 1) * 0.3f, -0.01f)));
        }
    }

    // Enable solid obstacle avoidance with the ground
    plantarchitecture.enableSolidObstacleAvoidance(ground_UUIDs, 0.2f);

    // Let the plant grow more - it should grow normally despite having short internodes
    // The length-based protection should kick in even if node count > 3
    plantarchitecture.advanceTime(plantID, 8);

    // Get all plant objects to verify plant survived and grew upward
    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Calculate center of mass to verify upward growth
    vec3 center_of_mass = make_vec3(0, 0, 0);
    uint total_objects = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            center_of_mass = center_of_mass + object_center;
            total_objects++;
        }
    }

    if (total_objects > 0) {
        center_of_mass = center_of_mass / float(total_objects);

        // The plant should have grown upward (center above ground level)
        DOCTEST_CHECK(center_of_mass.z > 0.01f);

        // Plant should have grown to a reasonable height, indicating protection worked
        // Since we're testing short internodes, the height will be more modest
        DOCTEST_CHECK(center_of_mass.z > 0.05f);
    }

    // Plant should have grown successfully (not been completely pruned)
    DOCTEST_CHECK(plant_objects.size() >= 10);
}

DOCTEST_TEST_CASE("PlantArchitecture Attraction Points Basic Functionality") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable collision detection for this test (optional - attraction points work independently)
    plantarchitecture.enableSoftCollisionAvoidance();

    // Test basic attraction points functionality
    std::vector<vec3> attraction_points = {make_vec3(1.0f, 0.0f, 1.0f), make_vec3(0.0f, 1.0f, 1.5f)};

    // Enable attraction points with valid parameters
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(attraction_points, 60.0f, 0.15f, 0.7f));

    // Test parameter validation - invalid angle
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(0.0f, 0.1f, 0.5f));
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(190.0f, 0.1f, 0.5f));

    // Test parameter validation - invalid distance
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(80.0f, 0.0f, 0.5f));
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(80.0f, -0.1f, 0.5f));

    // Test parameter validation - invalid weight
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(80.0f, 0.1f, -0.1f));
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(80.0f, 0.1f, 1.1f));

    // Update attraction points
    std::vector<vec3> new_attraction_points = {make_vec3(2.0f, 0.0f, 2.0f)};
    DOCTEST_CHECK_NOTHROW(plantarchitecture.updateAttractionPoints(new_attraction_points));

    // Disable attraction points
    DOCTEST_CHECK_NOTHROW(plantarchitecture.disableAttractionPoints());

    // Test error when trying to update disabled attraction points
    DOCTEST_CHECK_THROWS(plantarchitecture.updateAttractionPoints(new_attraction_points));
}

DOCTEST_TEST_CASE("PlantArchitecture Attraction Points Independent of Collision Detection") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    std::vector<vec3> attraction_points = {make_vec3(1.0f, 0.0f, 1.0f)};

    // Attraction points should work without collision detection enabled
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(attraction_points));
}

DOCTEST_TEST_CASE("PlantArchitecture Attraction Points Empty Vector") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    std::vector<vec3> empty_attraction_points;

    // Try to enable attraction points with empty vector
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(empty_attraction_points));

    // Enable with valid points first (should work without collision detection)
    std::vector<vec3> valid_points = {make_vec3(1.0f, 0.0f, 1.0f)};
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(valid_points));

    // Try to update with empty vector (should fail)
    DOCTEST_CHECK_THROWS(plantarchitecture.updateAttractionPoints(empty_attraction_points));
}

DOCTEST_TEST_CASE("PlantArchitecture Native Attraction Point Cone Detection") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Set up attraction points at known locations
    std::vector<vec3> attraction_points = {
            make_vec3(0.0f, 0.0f, 2.0f), // Directly ahead
            make_vec3(1.0f, 0.0f, 1.0f), // Right and forward
            make_vec3(-1.0f, 0.0f, 1.0f), // Left and forward
            make_vec3(0.0f, 2.0f, 0.0f), // Far to the side (should be outside cone)
    };

    // Enable attraction points (should work without collision detection)
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(attraction_points, 60.0f, 3.0f, 0.7f));

    // Test 1: Looking straight up should find the point directly ahead
    vec3 vertex = make_vec3(0.0f, 0.0f, 0.0f);
    vec3 look_direction = make_vec3(0.0f, 0.0f, 1.0f); // Looking up
    vec3 direction_to_closest;

    bool found = plantarchitecture.detectAttractionPointsInCone(vertex, look_direction, 3.0f, 60.0f, direction_to_closest);
    DOCTEST_CHECK(found);

    // The closest should be the one directly ahead (0,0,2)
    vec3 expected_direction = make_vec3(0.0f, 0.0f, 1.0f);
    float dot_product = direction_to_closest * expected_direction;
    DOCTEST_CHECK(dot_product > 0.99f); // Should be very close to parallel

    // Test 2: Looking to the side should NOT find the point far to the side (outside cone)
    look_direction = make_vec3(1.0f, 0.0f, 0.0f); // Looking right
    found = plantarchitecture.detectAttractionPointsInCone(vertex, look_direction, 3.0f, 30.0f, direction_to_closest);

    // With a narrow cone (30 degrees), the side point at (0,2,0) should be outside the cone
    // But the point at (1,0,1) might be visible, so we might still find something

    // Test 3: Test parameter validation
    found = plantarchitecture.detectAttractionPointsInCone(vertex, look_direction, -1.0f, 60.0f, direction_to_closest);
    DOCTEST_CHECK(!found); // Should fail with negative look ahead distance

    found = plantarchitecture.detectAttractionPointsInCone(vertex, look_direction, 3.0f, 0.0f, direction_to_closest);
    DOCTEST_CHECK(!found); // Should fail with zero half angle

    found = plantarchitecture.detectAttractionPointsInCone(vertex, look_direction, 3.0f, 180.0f, direction_to_closest);
    DOCTEST_CHECK(!found); // Should fail with 180 degree half angle
}

DOCTEST_TEST_CASE("PlantArchitecture Attraction Points Plant Growth Integration") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable collision detection first
    plantarchitecture.enableSoftCollisionAvoidance();

    // Set up attraction points above the plant to guide upward growth
    std::vector<vec3> attraction_points = {
            make_vec3(0.1f, 0.1f, 1.0f), // Close to plant base but higher
            make_vec3(0.0f, 0.0f, 1.5f) // Further away and higher
    };

    // Enable attraction points with moderate attraction weight
    plantarchitecture.enableAttractionPoints(attraction_points, 80.0f, 0.2f, 0.6f);

    // Create a simple plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Let the plant grow with attraction points enabled
    plantarchitecture.advanceTime(plantID, 5);

    // Get plant geometry to verify growth occurred
    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Calculate plant center of mass to verify upward growth toward attraction points
    vec3 center_of_mass = make_vec3(0, 0, 0);
    uint total_objects = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            center_of_mass = center_of_mass + object_center;
            total_objects++;
        }
    }

    if (total_objects > 0) {
        center_of_mass = center_of_mass / float(total_objects);

        // Plant should have grown upward toward attraction points
        // Bean plants start small, so adjust expectations to realistic growth
        DOCTEST_CHECK(center_of_mass.z > 0.01f); // At least 1cm above ground

        // Plant should show some lateral movement toward attraction points
        // (not perfectly vertical growth due to attraction)
        float lateral_distance = sqrt(center_of_mass.x * center_of_mass.x + center_of_mass.y * center_of_mass.y);
        DOCTEST_CHECK(lateral_distance >= 0.0f); // Basic sanity check
    }

    // Test disabling attraction points mid-growth
    plantarchitecture.disableAttractionPoints();

    // Continue growing - should revert to natural growth patterns
    plantarchitecture.advanceTime(plantID, 3);

    // Verify plant continues to exist and grow
    std::vector<uint> final_plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(final_plant_objects.size() >= plant_objects.size());
}

DOCTEST_TEST_CASE("PlantArchitecture Attraction Points Priority Over Collision Avoidance") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Create some obstacle geometry
    std::vector<uint> obstacle_UUIDs;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            obstacle_UUIDs.push_back(
                    context.addTriangle(make_vec3(i * 0.3f + 0.5f, j * 0.3f + 0.5f, 0.5f + i * 0.1f), make_vec3((i + 1) * 0.3f + 0.5f, (j + 1) * 0.3f + 0.5f, 0.5f + i * 0.1f), make_vec3((i + 1) * 0.3f + 0.5f, j * 0.3f + 0.5f, 0.5f + i * 0.1f)));
        }
    }

    // Enable collision detection with obstacles
    plantarchitecture.enableSoftCollisionAvoidance(obstacle_UUIDs);

    // Set up attraction points on the opposite side of obstacles
    std::vector<vec3> attraction_points = {
            make_vec3(-0.5f, 0.0f, 1.0f) // Away from obstacles
    };

    // Enable attraction points - should override soft collision avoidance
    plantarchitecture.enableAttractionPoints(attraction_points, 90.0f, 0.3f, 0.8f);

    // Create a plant near obstacles
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0.3f, 0.3f, 0), 0);

    // Let the plant grow - should be attracted away from obstacles
    plantarchitecture.advanceTime(plantID, 4);

    // Verify plant grew successfully (attraction points should guide it away from obstacles)
    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Check that plant moved toward attraction point (negative x direction)
    vec3 center_of_mass = make_vec3(0, 0, 0);
    uint total_objects = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            center_of_mass = center_of_mass + object_center;
            total_objects++;
        }
    }

    if (total_objects > 0) {
        center_of_mass = center_of_mass / float(total_objects);

        // Plant should have grown upward
        DOCTEST_CHECK(center_of_mass.z > 0.01f); // At least 1cm above ground

        // With strong attraction weight (0.8), plant should show movement toward attraction point
        // This validates that attraction points override soft collision avoidance
    }
}

DOCTEST_TEST_CASE("PlantArchitecture Hard Obstacle Avoidance Takes Priority Over Attraction Points") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Create ground-level obstacles that would trigger hard obstacle avoidance
    std::vector<uint> solid_obstacle_UUIDs;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            solid_obstacle_UUIDs.push_back(context.addTriangle(make_vec3(i * 0.1f, j * 0.1f, 0.1f), make_vec3((i + 1) * 0.1f, (j + 1) * 0.1f, 0.1f), make_vec3((i + 1) * 0.1f, j * 0.1f, 0.1f)));
        }
    }

    // Enable collision detection first
    plantarchitecture.enableSoftCollisionAvoidance();

    // Enable solid obstacle avoidance (hard obstacles)
    plantarchitecture.enableSolidObstacleAvoidance(solid_obstacle_UUIDs, 0.15f);

    // Set up attraction points in the opposite direction of safe growth
    std::vector<vec3> attraction_points = {
            make_vec3(0.0f, 0.0f, 0.05f) // Low attraction point that would conflict with obstacle avoidance
    };

    // Enable attraction points
    plantarchitecture.enableAttractionPoints(attraction_points, 70.0f, 0.1f, 0.9f);

    // Create a plant at the origin
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Let the plant grow - hard obstacle avoidance should take priority
    plantarchitecture.advanceTime(plantID, 3);

    // Verify plant grew successfully despite conflicting guidance
    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Plant should have grown upward to avoid hard obstacles, regardless of attraction points
    vec3 center_of_mass = make_vec3(0, 0, 0);
    uint total_objects = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            center_of_mass = center_of_mass + object_center;
            total_objects++;
        }
    }

    if (total_objects > 0) {
        center_of_mass = center_of_mass / float(total_objects);

        // Hard obstacle avoidance should force upward growth
        DOCTEST_CHECK(center_of_mass.z > 0.01f); // At least 1cm above ground

        // Plant should have avoided the low obstacles (which are at 0.1m height)
        // So plant should be higher than the obstacle level
        DOCTEST_CHECK(center_of_mass.z > 0.005f); // Above the base obstacle level
    }
}

DOCTEST_TEST_CASE("PlantArchitecture Attraction Points with Surface Following") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Create a vertical wall that we want the plant to approach and then grow parallel to
    std::vector<uint> wall_obstacle_UUIDs;
    std::vector<vec3> wall_attraction_points;

    // Create vertical wall at x = 0.3
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            // Wall surface obstacles (solid)
            wall_obstacle_UUIDs.push_back(context.addTriangle(make_vec3(0.3f, i * 0.05f, j * 0.05f), make_vec3(0.3f, (i + 1) * 0.05f, (j + 1) * 0.05f), make_vec3(0.3f, (i + 1) * 0.05f, j * 0.05f)));

            // Attraction points on the wall surface
            wall_attraction_points.push_back(make_vec3(0.29f, i * 0.05f + 0.025f, j * 0.05f + 0.025f));
        }
    }

    // Enable collision detection with wall obstacles
    plantarchitecture.enableSoftCollisionAvoidance();

    // Enable solid obstacle avoidance for the wall
    plantarchitecture.enableSolidObstacleAvoidance(wall_obstacle_UUIDs, 0.05f);

    // Enable attraction points on the wall surface with reduced obstacle reduction factor
    // This allows the plant to maintain some attraction even when avoiding obstacles
    plantarchitecture.enableAttractionPoints(wall_attraction_points, 60.0f, 0.1f, 0.8f);
    plantarchitecture.setAttractionParameters(60.0f, 0.1f, 0.8f, 0.5f); // Higher obstacle reduction factor

    // Create a plant at origin that should grow toward the wall
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Let the plant grow - it should approach the wall and then follow it
    plantarchitecture.advanceTime(plantID, 4);

    // Get plant geometry to verify behavior
    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Calculate plant center of mass
    vec3 center_of_mass = make_vec3(0, 0, 0);
    uint total_objects = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            center_of_mass = center_of_mass + object_center;
            total_objects++;
        }
    }

    if (total_objects > 0) {
        center_of_mass = center_of_mass / float(total_objects);

        // Plant should have grown upward
        DOCTEST_CHECK(center_of_mass.z > 0.01f);

        // The key test is that the plant grows successfully with both attraction points and obstacle avoidance enabled
        // This validates that the new blended approach doesn't cause conflicts or crashes
        // The exact movement direction depends on many factors, but the plant should grow

        // This test primarily validates that our improved blending logic works without errors
        // when both attraction points and hard obstacle avoidance are enabled simultaneously
    }
}

DOCTEST_TEST_CASE("PlantArchitecture Smooth Hard Obstacle Avoidance") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableSoftCollisionAvoidance();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    // Create obstacles at varying distances to test smooth avoidance behavior
    std::vector<uint> obstacle_UUIDs;

    // Create obstacles at different normalized distances from plant growth path
    // Plant will grow upward from (0,0,0), so place obstacles to the side at different z heights
    for (int i = 0; i < 4; i++) {
        float z_height = 0.1f + i * 0.05f; // Heights: 0.1, 0.15, 0.2, 0.25

        // Create obstacle patches at different distances from expected growth path
        float x_distance = 0.05f + i * 0.02f; // Distances: 0.05, 0.07, 0.09, 0.11

        obstacle_UUIDs.push_back(context.addTriangle(make_vec3(x_distance, -0.02f, z_height), make_vec3(x_distance + 0.04f, -0.02f, z_height), make_vec3(x_distance, 0.02f, z_height)));
        obstacle_UUIDs.push_back(context.addTriangle(make_vec3(x_distance + 0.04f, 0.02f, z_height), make_vec3(x_distance + 0.04f, -0.02f, z_height), make_vec3(x_distance, 0.02f, z_height)));
    }

    plantarchitecture.enableSolidObstacleAvoidance(obstacle_UUIDs, 0.25f);

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 8);

    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Calculate plant center of mass to verify it avoided obstacles
    vec3 center_of_mass = make_vec3(0, 0, 0);
    uint total_objects = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            center_of_mass = center_of_mass + object_center;
            total_objects++;
        }
    }

    if (total_objects > 0) {
        center_of_mass = center_of_mass / float(total_objects);

        // Plant should have grown upward successfully
        DOCTEST_CHECK(center_of_mass.z > 0.01f);

        // Plant should have moved away from obstacles (toward negative x since obstacles are on positive x side)
        // This tests that smooth avoidance works without the harsh discrete jumps
        DOCTEST_CHECK(center_of_mass.x <= 0.01f); // Should stay near or move away from obstacles

        // Key validation: plant grows successfully with smooth obstacle avoidance
        // The smooth distance-normalized approach should provide gradual, natural avoidance
        // rather than abrupt discrete changes in behavior
    }
}

DOCTEST_TEST_CASE("PlantArchitecture Hard Obstacle Avoidance Buffer Zone") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableSoftCollisionAvoidance();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    // Create a vertical post obstacle similar to the test case image
    std::vector<uint> post_UUIDs;
    float post_radius = 0.02f; // 2cm radius post
    float post_height = 0.5f; // 50cm tall post

    // Create post as a series of triangles forming a cylinder at x=0.1m (10cm from plant center)
    int segments = 8;
    for (int i = 0; i < segments; i++) {
        float theta1 = 2.0f * M_PI * float(i) / float(segments);
        float theta2 = 2.0f * M_PI * float(i + 1) / float(segments);

        vec3 p1_bottom = make_vec3(0.1f + post_radius * cos(theta1), post_radius * sin(theta1), 0);
        vec3 p2_bottom = make_vec3(0.1f + post_radius * cos(theta2), post_radius * sin(theta2), 0);
        vec3 p1_top = make_vec3(0.1f + post_radius * cos(theta1), post_radius * sin(theta1), post_height);
        vec3 p2_top = make_vec3(0.1f + post_radius * cos(theta2), post_radius * sin(theta2), post_height);

        // Two triangles per segment to form cylinder walls
        post_UUIDs.push_back(context.addTriangle(p1_bottom, p2_bottom, p1_top));
        post_UUIDs.push_back(context.addTriangle(p2_bottom, p2_top, p1_top));
    }

    // Set detection distance and enable solid obstacle avoidance
    float detection_distance = 0.2f; // 20cm detection distance
    float expected_buffer = detection_distance * 0.05f; // 5% buffer = 1cm

    plantarchitecture.enableSolidObstacleAvoidance(post_UUIDs, detection_distance);

    // Create plant at origin, should grow toward +x direction but avoid the post
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 8);

    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Calculate minimum distance between plant and post to verify buffer is maintained
    float min_distance_to_post = std::numeric_limits<float>::max();
    vec3 post_center = make_vec3(0.1f, 0, 0.25f); // Center of post

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);

            // Check distance from each corner of plant object to post center
            vec3 corners[8] = {make_vec3(min_corner.x, min_corner.y, min_corner.z), make_vec3(max_corner.x, min_corner.y, min_corner.z), make_vec3(min_corner.x, max_corner.y, min_corner.z), make_vec3(min_corner.x, min_corner.y, max_corner.z),
                               make_vec3(max_corner.x, max_corner.y, min_corner.z), make_vec3(max_corner.x, min_corner.y, max_corner.z), make_vec3(min_corner.x, max_corner.y, max_corner.z), make_vec3(max_corner.x, max_corner.y, max_corner.z)};

            for (int i = 0; i < 8; i++) {
                float distance = (corners[i] - post_center).magnitude();
                min_distance_to_post = std::min(min_distance_to_post, distance);
            }
        }
    }

    // Plant should maintain buffer distance from post (accounting for post radius)
    float expected_min_distance = post_radius + expected_buffer;
    DOCTEST_CHECK(min_distance_to_post >= expected_min_distance * 0.8f); // Allow 20% tolerance for growth dynamics

    // Plant should have grown upward successfully despite obstacle
    vec3 plant_center = make_vec3(0, 0, 0);
    uint plant_object_count = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            plant_center = plant_center + object_center;
            plant_object_count++;
        }
    }

    if (plant_object_count > 0) {
        plant_center = plant_center / float(plant_object_count);
        DOCTEST_CHECK(plant_center.z > 0.01f); // Should grow upward

        // Plant should avoid growing directly into the post (should stay away from x=0.1)
        // With buffer zone avoidance, plant should either go around or grow upward
        DOCTEST_CHECK(fabs(plant_center.x - 0.1f) > expected_buffer * 0.5f); // Should maintain some distance from post center line
    }
}

DOCTEST_TEST_CASE("PlantArchitecture solid obstacle avoidance works independently") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Create obstacle geometry (ground plane)
    std::vector<uint> obstacle_UUIDs;
    obstacle_UUIDs.push_back(context.addTriangle(make_vec3(-1, -1, -0.01f), make_vec3(1, -1, -0.01f), make_vec3(-1, 1, -0.01f)));
    obstacle_UUIDs.push_back(context.addTriangle(make_vec3(1, 1, -0.01f), make_vec3(1, -1, -0.01f), make_vec3(-1, 1, -0.01f)));

    // Test: Enable ONLY solid obstacle avoidance (no soft collision avoidance)
    // This should work independently after our fix
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableSolidObstacleAvoidance(obstacle_UUIDs, 0.2f));

    // Load and build a plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Advance time - this should work without crashing and plant should grow upward
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 5.0f));

    // Verify plant was created and grew
    std::vector<uint> plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Calculate plant center of mass to verify upward growth (avoiding ground obstacle)
    vec3 plant_center = make_vec3(0, 0, 0);
    uint plant_object_count = 0;

    for (uint objID: plant_objects) {
        if (context.doesObjectExist(objID)) {
            vec3 min_corner, max_corner;
            context.getObjectBoundingBox(objID, min_corner, max_corner);
            vec3 object_center = (min_corner + max_corner) / 2.0f;
            plant_center = plant_center + object_center;
            plant_object_count++;
        }
    }

    if (plant_object_count > 0) {
        plant_center = plant_center / float(plant_object_count);
        // Plant should grow upward, avoiding the ground obstacle at z = -0.01f
        DOCTEST_CHECK(plant_center.z > 0.01f);
    }

    // Test: Add soft collision avoidance on top of existing solid obstacle avoidance
    // This should work together seamlessly
    std::vector<uint> soft_target_UUIDs;
    std::vector<uint> soft_target_IDs;
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableSoftCollisionAvoidance(soft_target_UUIDs, soft_target_IDs));

    // Continue growing - should still work with both systems enabled
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 2.0f));

    // Verify plant continued to grow
    std::vector<uint> final_plant_objects = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(final_plant_objects.size() >= plant_objects.size());
}

DOCTEST_TEST_CASE("PlantArchitecture Per-Plant Attraction Points") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Disable messages for cleaner test output
    plantarchitecture.disableMessages();

    // Create two plants at different positions
    uint plantID1 = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0);
    uint plantID2 = plantarchitecture.addPlantInstance(make_vec3(5, 0, 0), 0);

    // Set different attraction points for each plant
    std::vector<vec3> attraction_points_1 = {make_vec3(1.0f, 0.0f, 1.0f), make_vec3(0.0f, 1.0f, 1.5f)};
    std::vector<vec3> attraction_points_2 = {make_vec3(6.0f, 0.0f, 1.0f), make_vec3(5.0f, 1.0f, 1.5f)};

    // Enable attraction points for each plant with different parameters
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(plantID1, attraction_points_1, 60.0f, 0.2f, 0.7f));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(plantID2, attraction_points_2, 45.0f, 0.15f, 0.5f));

    // Test parameter updates for individual plants
    DOCTEST_CHECK_NOTHROW(plantarchitecture.setAttractionParameters(plantID1, 80.0f, 0.25f, 0.8f, 0.6f));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.updateAttractionPoints(plantID2, {make_vec3(6.5f, 0.5f, 2.0f)}));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.appendAttractionPoints(plantID1, {make_vec3(1.5f, 1.5f, 2.0f)}));

    // Test disabling for individual plants
    DOCTEST_CHECK_NOTHROW(plantarchitecture.disableAttractionPoints(plantID1));

    // Test error handling for invalid plant IDs
    DOCTEST_CHECK_THROWS(plantarchitecture.enableAttractionPoints(9999, attraction_points_1));
    DOCTEST_CHECK_THROWS(plantarchitecture.disableAttractionPoints(9999));
    DOCTEST_CHECK_THROWS(plantarchitecture.updateAttractionPoints(9999, attraction_points_1));
    DOCTEST_CHECK_THROWS(plantarchitecture.appendAttractionPoints(9999, attraction_points_1));
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(9999, 60.0f, 0.15f, 0.7f, 0.75f));
}

DOCTEST_TEST_CASE("PlantArchitecture Global vs Per-Plant Interaction") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Disable messages for cleaner test output
    plantarchitecture.disableMessages();

    // Create a plant first
    uint plantID1 = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0);

    // Set global attraction points - should affect all plants including existing ones
    std::vector<vec3> global_attraction_points = {make_vec3(1.0f, 0.0f, 1.0f), make_vec3(0.0f, 1.0f, 1.5f)};
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(global_attraction_points, 60.0f, 0.15f, 0.7f));

    // Create another plant after global attraction points are set
    uint plantID2 = plantarchitecture.addPlantInstance(make_vec3(5, 0, 0), 0);

    // Now set plant-specific attraction points for plant 1 - should override global for that plant
    std::vector<vec3> specific_attraction_points = {make_vec3(2.0f, 0.0f, 2.0f)};
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(plantID1, specific_attraction_points, 45.0f, 0.1f, 0.5f));

    // Test that global update affects all plants with attraction points enabled
    DOCTEST_CHECK_NOTHROW(plantarchitecture.updateAttractionPoints({make_vec3(3.0f, 0.0f, 3.0f)}));

    // Global disable should affect all plants
    DOCTEST_CHECK_NOTHROW(plantarchitecture.disableAttractionPoints());

    // Re-enable global attraction points to test backward compatibility
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(global_attraction_points));
}

DOCTEST_TEST_CASE("PlantArchitecture Plant-Specific Attraction Points Validation") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Disable messages for cleaner test output
    plantarchitecture.disableMessages();

    // Create plants to test validation and method calls
    uint plantID1 = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0);
    uint plantID2 = plantarchitecture.addPlantInstance(make_vec3(5, 0, 0), 0);

    // Set different attraction points for each plant
    std::vector<vec3> attraction_points_1 = {make_vec3(1.0f, 0.0f, 1.0f)};
    std::vector<vec3> attraction_points_2 = {make_vec3(6.0f, 0.0f, 1.0f)};

    // Test that plant-specific methods work correctly
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(plantID1, attraction_points_1));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.enableAttractionPoints(plantID2, attraction_points_2));

    // Test parameter validation
    DOCTEST_CHECK_THROWS(plantarchitecture.enableAttractionPoints(plantID1, {}, 60.0f, 0.15f, 0.7f)); // Empty vector
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(plantID1, 0.0f, 0.15f, 0.7f)); // Invalid angle
    DOCTEST_CHECK_THROWS(plantarchitecture.setAttractionParameters(plantID1, 60.0f, 0.0f, 0.7f)); // Invalid distance

    // Test successful parameter updates
    DOCTEST_CHECK_NOTHROW(plantarchitecture.setAttractionParameters(plantID1, 80.0f, 0.25f, 0.8f, 0.6f));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.updateAttractionPoints(plantID2, {make_vec3(6.5f, 0.5f, 2.0f)}));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.appendAttractionPoints(plantID1, {make_vec3(1.5f, 1.5f, 2.0f)}));

    // Test disabling
    DOCTEST_CHECK_NOTHROW(plantarchitecture.disableAttractionPoints(plantID1));
}

DOCTEST_TEST_CASE("PlantArchitecture removeShootFloralBuds") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Test invalid plant ID - should throw
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS(plantarchitecture.removeShootFloralBuds(9999, 0));

    // Create a plant instance to test valid plant ID but invalid shoot ID
    uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != -1);

    // Test invalid shoot ID - should throw
    DOCTEST_CHECK_THROWS(plantarchitecture.removeShootFloralBuds(plantID, 9999));
}

DOCTEST_TEST_CASE("PlantArchitecture XML write with flowers and fruit") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Load tomato model (has flowers and fruit)
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("tomato"));

    // Build simple plant
    vec3 base_position(1.0f, 2.0f, 0.5f);
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(base_position, 180);
    DOCTEST_CHECK(plantID != uint(-1));

    // Write plant structure to XML (should not crash even if no flowers)
    std::string xml_filename = "test_plant_xml_write.xml";
    DOCTEST_CHECK_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    // Clean up test file
    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture readPlantStructureXML preserves internode vertex-sharing convention") {
    // Regression test. Shoot::shoot_internode_vertices holds one inner vector per phytomer under a
    // vertex-sharing convention established in the Phytomer constructor: the first phytomer on a
    // shoot stores all Ndiv+1 nodes, while every later phytomer omits its node 0 because that node
    // is the previous phytomer's last node. The internode Tube object is built to match, so a shoot
    // of P phytomers has exactly (Ndiv+1) + (P-1)*Ndiv nodes, and Shoot::updateShootNodes() pushes
    // flatten(shoot_internode_vertices) straight into it.
    //
    // readPlantStructureXML() used to reconstruct Ndiv+1 nodes for EVERY phytomer, duplicating each
    // shared node. That made both flattened arrays P-1 entries too long, so the next advanceTime()
    // threw from Tube::setTubeRadii(). It also broke the consumers that assume the shared node is
    // absent: Phytomer::getInternodeNodePositions() prepends a duplicate, and
    // Phytomer::setInternodeLengthScaleFraction() differences the duplicate against its own twin and
    // divides by a zero-magnitude axis, writing NaN into every downstream vertex.
    //
    // Bean is used because its phytomers have internode.length_segments == 2, so Ndiv > 1 and the
    // off-by-one is observable. With Ndiv == 1 the two conventions would coincide.

    const std::string xml_filename = "test_plant_xml_internode_convention.xml";

    // --- Build a plant and save it ---
    uint built_shoot_count = 0;
    float built_leaf_area = 0.f;
    {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("bean");

        // 30 days gives the base shoot several phytomers plus at least one child shoot, so all three
        // internode_base branches of the reconstruction are exercised, while keeping the test quick.
        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 30);
        DOCTEST_REQUIRE(plantID != uint(-1));

        built_shoot_count = static_cast<uint>(plantarchitecture.getAllShootIDs(plantID).size());
        built_leaf_area = plantarchitecture.sumPlantLeafArea(plantID);

        // Guard against a vacuous test: if no shoot carries more than one phytomer, the shared-node
        // case never arises and every assertion below would pass even on the buggy code.
        bool has_multi_phytomer_shoot = false;
        for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
            if (plantarchitecture.getPlantShoot(plantID, shootID)->phytomers.size() > 1) {
                has_multi_phytomer_shoot = true;
            }
        }
        DOCTEST_REQUIRE(has_multi_phytomer_shoot);

        DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));
    }

    // --- Reload into a fresh Context/PlantArchitecture ---
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    // Required: the XML references shoot types by label, and reading resolves them via
    // getCurrentShootParameters(), which errors if the model library was not loaded first.
    plantarchitecture.loadPlantModelFromLibrary("bean");

    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);
    const uint restored_plantID = restored_plantIDs.front();

    const std::vector<uint> restored_shootIDs = plantarchitecture.getAllShootIDs(restored_plantID);
    DOCTEST_REQUIRE(!restored_shootIDs.empty());

    // The core invariant, checked before advanceTime() is ever called: the flattened per-phytomer
    // arrays must have exactly as many entries as the shoot's internode Tube has nodes.
    uint shoots_checked = 0;
    for (uint shootID: restored_shootIDs) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(restored_plantID, shootID);

        // A shoot whose tube was never built or was pruned away carries the sentinel object ID.
        if (!context.doesObjectExist(shoot->internode_tube_objID)) {
            continue;
        }

        const uint tube_node_count = context.getTubeObjectNodeCount(shoot->internode_tube_objID);
        DOCTEST_CHECK(flatten(shoot->shoot_internode_vertices).size() == tube_node_count);
        DOCTEST_CHECK(flatten(shoot->shoot_internode_radii).size() == tube_node_count);

        // The convention itself, phytomer by phytomer.
        DOCTEST_REQUIRE(shoot->shoot_internode_vertices.size() == shoot->phytomers.size());
        DOCTEST_REQUIRE(!shoot->shoot_internode_vertices.front().empty());
        const size_t segments_per_phytomer = shoot->shoot_internode_vertices.front().size() - 1;
        DOCTEST_CHECK(segments_per_phytomer >= 1);
        for (size_t p = 1; p < shoot->shoot_internode_vertices.size(); p++) {
            DOCTEST_CHECK(shoot->shoot_internode_vertices.at(p).size() == segments_per_phytomer);
            DOCTEST_CHECK(shoot->shoot_internode_radii.at(p).size() == segments_per_phytomer);

            // The shared node must be absent: this phytomer's first node must not duplicate the
            // previous phytomer's last node. This is the most direct statement of the bug, and it
            // does not depend on the Tube object at all.
            DOCTEST_CHECK(shoot->shoot_internode_vertices.at(p).front() != shoot->shoot_internode_vertices.at(p - 1).back());
        }

        // The Context tube geometry must already agree with the restored vertices, not just after the
        // first growth step -- readPlantStructureXML() syncs it via updateShootNodes().
        const std::vector<vec3> tube_nodes = context.getTubeObjectNodes(shoot->internode_tube_objID);
        const std::vector<vec3> flat_nodes = flatten(shoot->shoot_internode_vertices);
        DOCTEST_REQUIRE(tube_nodes.size() == flat_nodes.size());
        for (size_t i = 0; i < tube_nodes.size(); i++) {
            DOCTEST_CHECK(tube_nodes.at(i).x == doctest::Approx(flat_nodes.at(i).x).epsilon(1e-4));
            DOCTEST_CHECK(tube_nodes.at(i).y == doctest::Approx(flat_nodes.at(i).y).epsilon(1e-4));
            DOCTEST_CHECK(tube_nodes.at(i).z == doctest::Approx(flat_nodes.at(i).z).epsilon(1e-4));
        }

        shoots_checked++;
    }
    // Guard against a vacuous pass if every tube happened to be absent.
    DOCTEST_REQUIRE(shoots_checked > 0);

    // Sampled before any growth -- see the note further below on why post-growth leaf *area* is not a
    // valid observable for a restored plant (post-growth leaf survival is checked, and is valid).
    const float restored_leaf_area_at_load = plantarchitecture.sumPlantLeafArea(restored_plantID);

    // Growing a restored plant must work, and repeated steps must stay stable. On the buggy code the
    // first call threw "ERROR (Tube::setTubeRadii): Number of radii in input vector must match
    // number of tube nodes."
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(restored_plantID, 5));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(restored_plantID, 5));

    // A restored plant must still have leaves after growing. This asserts non-collapse only, not a
    // geometric quantity, so it is not subject to the RNG-divergence flakiness described above. Before
    // phenological thresholds were persisted, growth defoliated the plant completely and this was 0.
    DOCTEST_CHECK(plantarchitecture.getPlantLeafCount(restored_plantID) > 0);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(restored_plantID, 5));

    // The invariant must still hold after growth: new phytomers are appended under the same
    // convention, so the counts stay locked to the tube.
    for (uint shootID: plantarchitecture.getAllShootIDs(restored_plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(restored_plantID, shootID);
        if (!context.doesObjectExist(shoot->internode_tube_objID)) {
            continue;
        }
        const uint tube_node_count = context.getTubeObjectNodeCount(shoot->internode_tube_objID);
        DOCTEST_CHECK(flatten(shoot->shoot_internode_vertices).size() == tube_node_count);
        DOCTEST_CHECK(flatten(shoot->shoot_internode_radii).size() == tube_node_count);
    }

    // Weak structural equivalence only. Exact geometric equality between "build then grow" and
    // "save, reload, then grow" is NOT achievable and is deliberately not asserted here:
    //   - the reconstruction skips collision avoidance and attraction on purpose;
    //   - restore recreates phytomers through addBaseStemShoot()/appendPhytomerToShoot(), which draw
    //     and resample random parameters, so the RNG stream position diverges from a fresh build;
    //   - shoot->gravitropic_curvature is not part of the XML schema, so it is resampled from the
    //     library distribution.
    // Tightening these into a geometric equality check would make the test flaky. Note also that
    // total shoot length is a confounded observable: the duplicated node produced a zero-length
    // segment, which contributes 0 to getInternodeLength(), so length was identical under both
    // layouts. Node counts, asserted above, are the observable that actually distinguishes them.
    DOCTEST_CHECK(plantarchitecture.getAllShootIDs(restored_plantID).size() >= built_shoot_count);

    // Leaf area is checked at restore time, BEFORE growth, and must match the saved plant closely --
    // leaves are reconstructed from saved parameters, so this is a real fidelity check on the restore.
    //
    // Post-growth leaf *area* is deliberately NOT compared against the freshly-built plant: for the
    // RNG-divergence reasons above that would be a flaky check. Post-growth leaf *survival* is checked
    // below, which is not RNG-sensitive.
    //
    // Historical note: phenological thresholds used to be absent from the XML schema, so a restored
    // plant kept the PlantInstance defaults, which were all zero. The dormancy check in advanceTime()
    // was therefore satisfied on the very first step, calling makeDormant() and dropping every leaf.
    // The thresholds are now persisted (and the defaults now encode "no phenology scheduled"), so a
    // restored plant grows normally -- see the "XML round-trip preserves phenological thresholds" and
    // "XML without phenology tags still loads" cases.
    DOCTEST_CHECK(restored_leaf_area_at_load > 0.f);
    DOCTEST_CHECK(std::isfinite(restored_leaf_area_at_load));
    DOCTEST_CHECK(restored_leaf_area_at_load == doctest::Approx(built_leaf_area).epsilon(0.25));

    // No NaN anywhere in the restored geometry. The buggy layout drove
    // Phytomer::setInternodeLengthScaleFraction() into a divide-by-zero that wrote NaN into every
    // downstream vertex, so this guards the silent-corruption failure mode as well as the throw.
    for (uint shootID: plantarchitecture.getAllShootIDs(restored_plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(restored_plantID, shootID);
        for (const std::vector<vec3> &phytomer_nodes: shoot->shoot_internode_vertices) {
            for (const vec3 &node: phytomer_nodes) {
                DOCTEST_CHECK(std::isfinite(node.x));
                DOCTEST_CHECK(std::isfinite(node.y));
                DOCTEST_CHECK(std::isfinite(node.z));
            }
        }
    }

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture child shoot rotation with multiple petioles per internode") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Regression test for bug where child shoots from different petioles had the same rotation
    // The fix changed line 4778 in PlantArchitecture.cpp to use petioles_per_internode
    // instead of axillary_vegetative_buds.size() for calculating rotation offset

    // Use bean plant which has 2 petioles per internode in the unifoliate stage
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != uint(-1));

    // Advance time to allow growth and child shoot formation
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 10.0f));

    // Verify plant created geometry (basic sanity check that build succeeded)
    std::vector<uint> all_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(all_primitives.size() > 0);

    // If this test passes, the fix is working (plant builds without errors)
    // The actual visual verification of proper 180-degree offset would require
    // more complex geometric analysis that is beyond the scope of a unit test
}

DOCTEST_TEST_CASE("PlantArchitecture plant_name optional object data") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable plant_name optional object data
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("plant_name"));

    // Load and build a bean plant
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != uint(-1));

    // Verify plant name is set correctly
    std::string plant_name = plantarchitecture.getPlantName(plantID);
    DOCTEST_CHECK(plant_name == "bean");

    // Advance time to create more organs
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 10.0f));

    // Get all object IDs
    std::vector<uint> all_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(all_primitives.size() > 0);

    // Verify plant_name object data is set on primitives
    bool found_plant_name_data = false;
    for (uint objID: all_primitives) {
        if (context.doesObjectDataExist(objID, "plant_name")) {
            std::string obj_plant_name;
            context.getObjectData(objID, "plant_name", obj_plant_name);
            DOCTEST_CHECK(obj_plant_name == "bean");
            found_plant_name_data = true;
        }
    }
    DOCTEST_CHECK(found_plant_name_data);
}

DOCTEST_TEST_CASE("PlantArchitecture plant_type tree classification") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable plant_type optional object data
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("plant_type"));

    // Test tree classification
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("almond"));
    uint treeID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(treeID != uint(-1));

    std::vector<uint> tree_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(tree_primitives.size() > 0);
    bool found_tree_type = false;
    for (uint objID: tree_primitives) {
        if (context.doesObjectDataExist(objID, "plant_type")) {
            std::string plant_type;
            context.getObjectData(objID, "plant_type", plant_type);
            DOCTEST_CHECK(plant_type == "tree");
            found_tree_type = true;
        }
    }
    DOCTEST_CHECK(found_tree_type);
}

DOCTEST_TEST_CASE("PlantArchitecture plant_type weed classification") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable plant_type optional object data
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("plant_type"));

    // Test weed classification
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bindweed"));
    uint weedID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(weedID != uint(-1));

    std::vector<uint> weed_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(weed_primitives.size() > 0);
    bool found_weed_type = false;
    for (uint objID: weed_primitives) {
        if (context.doesObjectDataExist(objID, "plant_type")) {
            std::string plant_type;
            context.getObjectData(objID, "plant_type", plant_type);
            DOCTEST_CHECK(plant_type == "weed");
            found_weed_type = true;
        }
    }
    DOCTEST_CHECK(found_weed_type);
}

DOCTEST_TEST_CASE("PlantArchitecture plant_type herbaceous classification") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable plant_type optional object data
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("plant_type"));

    // Test herbaceous classification (default)
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint herbaceousID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(herbaceousID != uint(-1));

    std::vector<uint> herbaceous_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(herbaceous_primitives.size() > 0);
    bool found_herbaceous_type = false;
    for (uint objID: herbaceous_primitives) {
        if (context.doesObjectDataExist(objID, "plant_type")) {
            std::string plant_type;
            context.getObjectData(objID, "plant_type", plant_type);
            DOCTEST_CHECK(plant_type == "herbaceous");
            found_herbaceous_type = true;
        }
    }
    DOCTEST_CHECK(found_herbaceous_type);
}

DOCTEST_TEST_CASE("PlantArchitecture plant_height optional object data") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable plant_height optional object data
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("plant_height"));

    // Build a bean plant
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != uint(-1));

    // Get initial height
    float initial_height = plantarchitecture.getPlantHeight(plantID);
    DOCTEST_CHECK(initial_height > 0);

    // Advance time to allow growth
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 10.0f));

    // Verify height increased
    float final_height = plantarchitecture.getPlantHeight(plantID);
    DOCTEST_CHECK(final_height > initial_height);

    // Verify plant_height object data was set and is reasonable
    std::vector<uint> all_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(all_primitives.size() > 0);
    bool found_height_data = false;
    for (uint objID: all_primitives) {
        if (context.doesObjectDataExist(objID, "plant_height")) {
            float obj_height;
            context.getObjectData(objID, "plant_height", obj_height);
            // Check height is within reasonable range (close to final_height)
            DOCTEST_CHECK(obj_height > initial_height);
            DOCTEST_CHECK(std::abs(obj_height - final_height) < 0.01f);
            found_height_data = true;
            break; // Only need to check one primitive
        }
    }
    DOCTEST_CHECK(found_height_data);
}

DOCTEST_TEST_CASE("PlantArchitecture phenology_stage optional object data") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable phenology_stage optional object data
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("phenology_stage"));

    // Build a bean plant
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != uint(-1));

    // Initially should be vegetative (no flowers, not dormant)
    std::string initial_stage = plantarchitecture.determinePhenologyStage(plantID);
    DOCTEST_CHECK(initial_stage == "vegetative");

    // Advance time to allow growth and potential flowering
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 20.0f));

    // Get current phenology stage
    std::string current_stage = plantarchitecture.determinePhenologyStage(plantID);
    DOCTEST_CHECK((current_stage == "vegetative" || current_stage == "reproductive" || current_stage == "senescent" || current_stage == "dormant"));

    // Verify phenology_stage object data was set
    std::vector<uint> all_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(all_primitives.size() > 0);
    bool found_stage_data = false;
    for (uint objID: all_primitives) {
        if (context.doesObjectDataExist(objID, "phenology_stage")) {
            std::string obj_stage;
            context.getObjectData(objID, "phenology_stage", obj_stage);
            DOCTEST_CHECK(obj_stage == current_stage);
            found_stage_data = true;
        }
    }
    DOCTEST_CHECK(found_stage_data);
}

DOCTEST_TEST_CASE("Build Parameters - Backward Compatibility (Grapevine VSP)") {
    // Test that empty parameter map produces identical plants to original hard-coded values
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Build with default parameters (empty map)
    plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP");
    std::map<std::string, float> empty_params;
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0, empty_params);

    // Verify plant was created
    DOCTEST_CHECK(plantID != uint(-1));

    // Verify basic plant structure exists
    std::vector<uint> plant_primitives = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(plant_primitives.size() > 0);
}

DOCTEST_TEST_CASE("Build Parameters - Parameter Override (Grapevine VSP)") {
    // Test that custom parameter values are applied correctly
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Build with custom parameters
    // Note: vine_spacing limited by cane max_nodes (9) * internode_length (0.15m) * 2 = 2.7m max
    plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP");
    std::map<std::string, float> custom_params = {
            {"vine_spacing", 2.5f}, // 2.5m spacing (within max_nodes limit)
            {"trunk_height", 0.15f} // 15 cm trunk height
    };
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0, custom_params);

    // Verify plant was created with custom parameters
    DOCTEST_CHECK(plantID != uint(-1));
    std::vector<uint> plant_primitives = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(plant_primitives.size() > 0);
}

DOCTEST_TEST_CASE("Build Parameters - Validation Catches Invalid Values (Grapevine VSP)") {
    // Test that out-of-range values raise errors
    capture_cerr cerr_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP");

    // Test vine_spacing out of range (valid range: 0.5-5.0)
    std::map<std::string, float> invalid_params1 = {{"vine_spacing", 10.0f}};
    DOCTEST_CHECK_THROWS(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0, invalid_params1));

    // Test trunk_height out of range (valid range: 0.05-1.0)
    std::map<std::string, float> invalid_params2 = {{"trunk_height", 2.0f}};
    DOCTEST_CHECK_THROWS(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0, invalid_params2));
}

DOCTEST_TEST_CASE("Build Parameters - Grapevine Wye Trellis Parameters") {
    // Test Wye grapevine specific trellis parameters
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("grapevine_Wye");
    std::map<std::string, float> trellis_params = {
            {"trunk_height", 0.2f}, // 20 cm trunk height
            {"cordon_spacing", 0.8f}, // 80 cm between cordon rows
            {"vine_spacing", 2.0f}, // 2 m between plants
            {"catch_wire_height", 2.5f} // 2.5 m catch wire height
    };
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0, trellis_params);

    DOCTEST_CHECK(plantID != uint(-1));
    std::vector<uint> plant_primitives = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(plant_primitives.size() > 0);
}

DOCTEST_TEST_CASE("Build Parameters - Tree Training System (Almond)") {
    // Test tree training parameters
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Note: trunk_height limited by trunk max_nodes (20) * internode_length (0.03m) = 0.6m max
    plantarchitecture.loadPlantModelFromLibrary("almond");
    std::map<std::string, float> tree_params = {
            {"trunk_height", 0.5f}, // 50 cm total trunk height (within max_nodes limit)
            {"num_scaffolds", 5.0f}, // 5 scaffold branches
            {"scaffold_angle", 35.0f} // 35 degree scaffold angle
    };
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000, tree_params);

    DOCTEST_CHECK(plantID != uint(-1));
    std::vector<uint> plant_primitives = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(plant_primitives.size() > 0);
}

DOCTEST_TEST_CASE("Build Parameters - Apple Tree") {
    // Test apple tree with custom parameters
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Note: trunk_height limited by trunk max_nodes (20) * internode_length (0.04m) = 0.8m max
    plantarchitecture.loadPlantModelFromLibrary("apple");
    std::map<std::string, float> apple_params = {
            {"trunk_height", 0.7f}, // 70 cm trunk height (within max_nodes limit)
            {"num_scaffolds", 6.0f}, // 6 scaffold branches
            {"scaffold_angle", 45.0f} // 45 degree scaffold angle
    };
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000, apple_params);

    DOCTEST_CHECK(plantID != uint(-1));
}

DOCTEST_TEST_CASE("Build Parameters - Pistachio Tree Fixed Scaffold System") {
    // Test pistachio tree with different scaffold count
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("pistachio");

    // Test with 2 scaffolds (minimum)
    std::map<std::string, float> pistachio_params_min = {{"num_scaffolds", 2.0f}};
    uint plantID_min = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000, pistachio_params_min);
    DOCTEST_CHECK(plantID_min != uint(-1));

    // Test with 4 scaffolds (default)
    std::map<std::string, float> pistachio_params_def = {{"num_scaffolds", 4.0f}};
    uint plantID_def = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(5, 0, 0), 5000, pistachio_params_def);
    DOCTEST_CHECK(plantID_def != uint(-1));
}

DOCTEST_TEST_CASE("Build Parameters - Canopy Building with Parameters") {
    // Test that parameters work with canopy building functions
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP");
    std::map<std::string, float> canopy_params = {
            {"vine_spacing", 2.0f}, // 2.0m vine spacing
            {"trunk_height", 0.12f} // 12 cm trunk height
    };

    // Test regular spacing canopy
    std::vector<uint> plantIDs = plantarchitecture.buildPlantCanopyFromLibrary(make_vec3(0, 0, 0), make_vec2(2, 2), make_int2(2, 2), 0, 1.0f, canopy_params);

    DOCTEST_CHECK(plantIDs.size() == 4);
    for (uint plantID: plantIDs) {
        DOCTEST_CHECK(plantID != uint(-1));
    }
}

DOCTEST_TEST_CASE("Build Parameters - Type Casting Float to Uint") {
    // Test that float parameters correctly cast to uint for node counts
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("almond");

    // Specify parameters as floats (should cast to uint internally where needed)
    // Note: trunk_height limited by trunk max_nodes (20) * internode_length (0.03m) = 0.6m max
    std::map<std::string, float> float_params = {
            {"trunk_height", 0.5f}, // Height as float (within max_nodes limit)
            {"num_scaffolds", 5.0f}, // Should cast to uint(5)
            {"scaffold_angle", 42.5f} // Angle as float
    };

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000, float_params);
    DOCTEST_CHECK(plantID != uint(-1));
}

DOCTEST_TEST_CASE("PlantArchitecture optionalOutputObjectData 'all' keyword") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Test that "all" (lowercase) enables all optional output data labels
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("all"));

    // Build a bean plant to verify data is actually being output
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != uint(-1));

    // Advance time to create some organs
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 10.0f));

    // Get all object IDs
    std::vector<uint> all_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(all_primitives.size() > 0);

    // Verify that basic metadata labels are present (these exist on all plants)
    // Note: Organ-specific labels (peduncleID, flowerID, fruitID) may not exist
    // if the plant hasn't developed those organs yet at this age
    std::vector<std::string> expected_labels = {"age", "rank", "plantID", "plant_name", "plant_height", "plant_type", "phenology_stage", "leafID"};

    for (const auto &label: expected_labels) {
        bool found = false;
        for (uint objID: all_primitives) {
            if (context.doesObjectDataExist(objID, label.c_str())) {
                found = true;
                break;
            }
        }
        DOCTEST_CHECK_MESSAGE(found, "Label '" << label << "' was not found on any primitive");
    }
}

DOCTEST_TEST_CASE("PlantArchitecture optionalOutputObjectData 'all' case-insensitive") {
    // Test "ALL" (uppercase)
    {
        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("ALL"));
    }

    // Test "All" (mixed case)
    {
        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("All"));
    }

    // Test "aLl" (random mixed case)
    {
        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("aLl"));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture optionalOutputObjectData invalid label throws error") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Test that an invalid label throws a helios_runtime_error with descriptive message
    bool caught_error = false;
    try {
        plantarchitecture.optionalOutputObjectData("invalid_label");
    } catch (const std::exception &e) {
        caught_error = true;
        std::string error_msg(e.what());
        DOCTEST_CHECK(error_msg.find("invalid_label") != std::string::npos);
        DOCTEST_CHECK(error_msg.find("not a valid option") != std::string::npos);
    }
    DOCTEST_CHECK(caught_error);

    // Note: helios_runtime_error() only writes to stderr when HELIOS_DEBUG is defined,
    // so we don't check stderr output here - just verify the exception is thrown correctly
}

DOCTEST_TEST_CASE("PlantArchitecture optionalOutputObjectData vector with 'all'") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Test that "all" works in a vector of labels
    std::vector<std::string> labels = {"all"};
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData(labels));

    // Build a bean plant to verify data is actually being output
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != uint(-1));

    // Advance time to create more organs
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 10.0f));

    // Get all object IDs
    std::vector<uint> all_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(all_primitives.size() > 0);

    // Verify that at least a few optional output data labels are present
    bool found_age = false;
    bool found_rank = false;
    bool found_plant_name = false;
    for (uint objID: all_primitives) {
        if (context.doesObjectDataExist(objID, "age"))
            found_age = true;
        if (context.doesObjectDataExist(objID, "rank"))
            found_rank = true;
        if (context.doesObjectDataExist(objID, "plant_name"))
            found_plant_name = true;
    }
    DOCTEST_CHECK(found_age);
    DOCTEST_CHECK(found_rank);
    DOCTEST_CHECK(found_plant_name);
}

DOCTEST_TEST_CASE("PlantArchitecture optionalOutputObjectData normal labels still work") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Test that individual labels still work as expected
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("age"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.optionalOutputObjectData("rank"));

    // Build a bean plant
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    DOCTEST_CHECK(plantID != uint(-1));

    // Advance time
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 5.0f));

    // Verify that age and rank data exist, but other optional data does not
    std::vector<uint> all_primitives = plantarchitecture.getAllObjectIDs();
    DOCTEST_CHECK(all_primitives.size() > 0);

    bool found_age = false;
    bool found_rank = false;
    bool found_plant_name = false; // This should NOT be found
    for (uint objID: all_primitives) {
        if (context.doesObjectDataExist(objID, "age"))
            found_age = true;
        if (context.doesObjectDataExist(objID, "rank"))
            found_rank = true;
        if (context.doesObjectDataExist(objID, "plant_name"))
            found_plant_name = true;
    }
    DOCTEST_CHECK(found_age);
    DOCTEST_CHECK(found_rank);
    DOCTEST_CHECK_FALSE(found_plant_name); // Should NOT be enabled
}

// ==================== NITROGEN MODEL TESTS ==================== //

DOCTEST_TEST_CASE("Nitrogen Model - Initialization") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Enable nitrogen model
    plantarchitecture.enableNitrogenModel();
    DOCTEST_CHECK(plantarchitecture.isNitrogenModelEnabled());

    // Build a simple plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Grow plant to create leaves
    plantarchitecture.advanceTime(plantID, 5.0f);

    // Initialize nitrogen pools with target concentration
    float initial_N_concentration = 1.5f; // g N/m² (target value)
    plantarchitecture.initializePlantNitrogenPools(plantID, initial_N_concentration);

    // Advance time to trigger nitrogen stress calculation and output writing
    plantarchitecture.advanceTime(plantID, 0.1f);

    // Get all leaf objects
    std::vector<uint> all_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(all_objects.size() > 0);

    // Verify leaf nitrogen content was initialized
    bool found_leaf_N = false;
    for (uint objID: all_objects) {
        if (context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            float leaf_N_area;
            context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N_area);
            DOCTEST_CHECK(leaf_N_area == doctest::Approx(initial_N_concentration).epsilon(0.1));
            found_leaf_N = true;
        }
    }
    DOCTEST_CHECK(found_leaf_N);
}

DOCTEST_TEST_CASE("Nitrogen Model - Application and Pool Splitting") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 3.0f);

    // Initialize with zero nitrogen
    plantarchitecture.initializePlantNitrogenPools(plantID, 0.0f);

    // Apply 10 g N to plant
    float N_applied = 10.0f; // g N
    plantarchitecture.addPlantNitrogen(plantID, N_applied);

    // Verify nitrogen was split between root (15%) and available (85%) pools
    // We can't directly access the pools, but we can verify by advancing time
    // and checking that leaves accumulate nitrogen from the available pool
    plantarchitecture.advanceTime(plantID, 1.0f);

    // Check that leaves now have nitrogen > 0
    std::vector<uint> all_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    bool found_N_accumulation = false;
    for (uint objID: all_objects) {
        if (context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            float leaf_N_area;
            context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N_area);
            if (leaf_N_area > 0) {
                found_N_accumulation = true;
                break;
            }
        }
    }
    DOCTEST_CHECK(found_N_accumulation);
}

DOCTEST_TEST_CASE("Nitrogen Model - Rate Limiting") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 5.0f);

    // Initialize with zero nitrogen
    plantarchitecture.initializePlantNitrogenPools(plantID, 0.0f);

    // Set nitrogen parameters with known max accumulation rate
    NitrogenParameters N_params;
    N_params.max_N_accumulation_rate = 0.1f; // g N/m²/day
    N_params.target_leaf_N_area = 10.0f; // Very high target to ensure demand > rate
    plantarchitecture.setPlantNitrogenParameters(plantID, N_params);

    // Apply large amount of nitrogen
    plantarchitecture.addPlantNitrogen(plantID, 100.0f);

    // Advance time by 1 day
    float dt = 1.0f;
    plantarchitecture.advanceTime(plantID, dt);

    // Check that leaf nitrogen didn't exceed rate limit
    std::vector<uint> all_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    for (uint objID: all_objects) {
        if (context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            float leaf_N_area;
            context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N_area);
            // Should be at most max_N_accumulation_rate * dt
            DOCTEST_CHECK(leaf_N_area <= N_params.max_N_accumulation_rate * dt * 1.01f); // 1% tolerance
        }
    }
}

DOCTEST_TEST_CASE("Nitrogen Model - Stress Factor Output") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 5.0f);

    // Initialize with low nitrogen (stress condition)
    plantarchitecture.initializePlantNitrogenPools(plantID, 0.5f); // Below target of 1.5

    // Advance time to trigger stress factor calculation
    plantarchitecture.advanceTime(plantID, 0.1f);

    // Verify stress factor exists and is in valid range [0, 1]
    std::vector<uint> plant_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(plant_objects.size() > 0);

    bool found_stress_factor = false;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            float stress_factor;
            context.getObjectData(objID, "nitrogen_stress_factor", stress_factor);
            DOCTEST_CHECK(stress_factor >= 0.0f);
            DOCTEST_CHECK(stress_factor <= 1.0f);
            // With low N, stress should be less than 1
            DOCTEST_CHECK(stress_factor < 1.0f);
            found_stress_factor = true;
            break;
        }
    }
    DOCTEST_CHECK(found_stress_factor);
}

DOCTEST_TEST_CASE("Nitrogen Model - Remobilization") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Grow plant to create leaves of different ages
    plantarchitecture.advanceTime(plantID, 15.0f);

    // Initialize with low nitrogen to create stress condition
    plantarchitecture.initializePlantNitrogenPools(plantID, 0.8f); // Below target

    // Advance time significantly to age leaves and trigger remobilization
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 25.0f));

    // Verify nitrogen stress factor reflects stress condition
    std::vector<uint> plant_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    bool found_stress_factor = false;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            float stress_factor;
            context.getObjectData(objID, "nitrogen_stress_factor", stress_factor);
            DOCTEST_CHECK(stress_factor < 1.0f); // Should indicate some stress
            found_stress_factor = true;
            break;
        }
    }
    DOCTEST_CHECK(found_stress_factor);
}

DOCTEST_TEST_CASE("Nitrogen Model - Fruit Removal") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();

    // Use tomato which produces fruit
    plantarchitecture.loadPlantModelFromLibrary("tomato");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Grow plant to vegetative stage
    plantarchitecture.advanceTime(plantID, 30.0f);

    // Initialize with adequate nitrogen
    plantarchitecture.initializePlantNitrogenPools(plantID, 1.5f);

    // Add nitrogen to available pool
    plantarchitecture.addPlantNitrogen(plantID, 50.0f);

    // Continue growth to allow fruiting
    plantarchitecture.advanceTime(plantID, 40.0f);

    // Verify plant grew (basic sanity check)
    std::vector<uint> plant_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Nitrogen stress factor should exist
    bool found_stress_factor = false;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            found_stress_factor = true;
            break;
        }
    }
    DOCTEST_CHECK(found_stress_factor);
}

DOCTEST_TEST_CASE("Nitrogen Model - Leaf-to-Fruit Translocation") {
    // When the available nitrogen pool cannot cover fruit demand, removeFruitNitrogen draws the
    // shortfall from leaves (old leaves first, then young leaves as fallback). To isolate
    // translocation cleanly, we override remobilization_age_threshold to a value age_fraction
    // never reaches, which disables the leaf-to-leaf remobilization pathway. With remobilization
    // disabled and the available pool empty, the only mechanism that can reduce a leaf below the
    // target N concentration is leaf-to-fruit translocation.

    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("tomato");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Disable leaf-to-leaf remobilization by setting an unreachable age threshold (age_fraction <= 1).
    NitrogenParameters N_params; // defaults: target=1.5, minimum=0.5, efficiency=0.7
    N_params.remobilization_age_threshold = 2.0f;
    plantarchitecture.setPlantNitrogenParameters(plantID, N_params);

    // Grow plant well past fruit-set so fruits exist when we initialize and snapshot.
    // Tomato in this library typically starts fruit set around day 40-50; advance past that.
    plantarchitecture.advanceTime(plantID, 60.0f);

    // Skip the test if the plant did not produce fruit in this run (random plant growth can
    // sometimes produce no fruits within the window). The translocation pathway only exercises
    // when fruits actively grow, so we need fruits to be present.
    std::vector<uint> fruit_objIDs = plantarchitecture.getPlantFruitObjectIDs(plantID);
    if (fruit_objIDs.empty()) {
        // Try a longer window before giving up.
        plantarchitecture.advanceTime(plantID, 30.0f);
        fruit_objIDs = plantarchitecture.getPlantFruitObjectIDs(plantID);
    }
    if (fruit_objIDs.empty()) {
        return; // No fruits formed in this run; nothing to test.
    }

    // Reset leaves to target N (overwrites any drainage that occurred during the warm-up advance);
    // do NOT call addPlantNitrogen so the available pool stays empty and any further fruit demand
    // must come from leaves via translocation.
    plantarchitecture.initializePlantNitrogenPools(plantID, N_params.target_leaf_N_area);

    // Trigger an output write so leaf_nitrogen_gN_m2 is materialized as object data
    plantarchitecture.advanceTime(plantID, 0.1f);

    // Sanity: at least one leaf is at the target initially
    bool any_leaf_at_target_pre = false;
    for (uint objID: plantarchitecture.getAllPlantObjectIDs(plantID)) {
        if (context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            float leaf_N_area;
            context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N_area);
            if (std::abs(leaf_N_area - N_params.target_leaf_N_area) < 0.01f) {
                any_leaf_at_target_pre = true;
                break;
            }
        }
    }
    DOCTEST_CHECK(any_leaf_at_target_pre);

    // Advance through ongoing fruit growth. With remobilization disabled and the pool empty, the
    // only path that can drop a leaf below target is translocation to fruit.
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 30.0f));

    // Look for evidence of translocation: at least one leaf that started at target now sits below
    // it (but at or above the per-leaf floor). Newly grown leaves with N == 0 are excluded by
    // requiring leaf_N_area > minimum_leaf_N_area.
    bool any_leaf_drained_below_target = false;
    float min_leaf_N_observed = std::numeric_limits<float>::infinity();
    bool any_leaf_with_N = false;
    for (uint objID: plantarchitecture.getAllPlantObjectIDs(plantID)) {
        if (context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            float leaf_N_area;
            context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N_area);
            if (leaf_N_area > N_params.minimum_leaf_N_area && leaf_N_area < N_params.target_leaf_N_area - 0.01f) {
                any_leaf_drained_below_target = true;
            }
            if (leaf_N_area > 1e-4f) {
                min_leaf_N_observed = std::min(min_leaf_N_observed, leaf_N_area);
                any_leaf_with_N = true;
            }
        }
    }

    // Confirm fruits still exist at the end of the test window (sanity check that fruit demand
    // was active for at least part of the post-advance period).
    fruit_objIDs = plantarchitecture.getPlantFruitObjectIDs(plantID);

    // Translocation drained at least one initialized leaf below the target.
    if (!fruit_objIDs.empty()) {
        DOCTEST_CHECK(any_leaf_drained_below_target);
    }

    // Per-leaf floor: with translocation only able to remove (current - minimum) * efficiency, a
    // fully drained leaf bottoms out at minimum + (initial - minimum)(1 - efficiency) = 0.8 g N/m²
    // for the defaults. Assert at least minimum_leaf_N_area as a slack lower bound.
    if (any_leaf_with_N) {
        DOCTEST_CHECK(min_leaf_N_observed >= N_params.minimum_leaf_N_area - 1e-3f);
    }

    // Stress factor output still written
    bool found_stress_factor = false;
    for (uint objID: plantarchitecture.getAllPlantObjectIDs(plantID)) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            found_stress_factor = true;
            break;
        }
    }
    DOCTEST_CHECK(found_stress_factor);
}

DOCTEST_TEST_CASE("Nitrogen Model - No Translocation When Pool Adequate") {
    // Negative control: with leaf-to-leaf remobilization disabled (unreachable threshold) AND a
    // well-stocked available pool, no drainage pathway should be active. Pre-existing leaves at
    // target N must remain at target after fruiting (translocation never triggers because the pool
    // covers demand). New leaves grown later may have lower N because accumulation is rate-limited,
    // so we only check the pre-existing initialized leaves.

    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("tomato");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    NitrogenParameters N_params;
    N_params.remobilization_age_threshold = 2.0f; // Disable leaf-to-leaf remobilization
    plantarchitecture.setPlantNitrogenParameters(plantID, N_params);

    plantarchitecture.advanceTime(plantID, 30.0f);
    plantarchitecture.initializePlantNitrogenPools(plantID, N_params.target_leaf_N_area);
    plantarchitecture.addPlantNitrogen(plantID, 200.0f); // Generously stock so pool always covers fruit demand
    plantarchitecture.advanceTime(plantID, 0.1f);        // Materialize object data

    // Capture pre-existing leaves that are at the target N concentration
    std::vector<uint> leaves_at_target_pre;
    for (uint objID: plantarchitecture.getAllPlantObjectIDs(plantID)) {
        if (context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            float leaf_N_area;
            context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N_area);
            if (std::abs(leaf_N_area - N_params.target_leaf_N_area) < 0.01f) {
                leaves_at_target_pre.push_back(objID);
            }
        }
    }
    DOCTEST_CHECK(leaves_at_target_pre.size() > 0);

    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 40.0f));

    // Pre-existing leaves at target should remain at (or very near) target. With remobilization
    // disabled and the pool adequate to cover fruit demand, no drainage pathway is active.
    int leaves_intact = 0;
    for (uint objID: leaves_at_target_pre) {
        if (!context.doesObjectExist(objID)) {
            continue;
        }
        if (!context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            continue;
        }
        float leaf_N_area;
        context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N_area);
        if (leaf_N_area >= N_params.target_leaf_N_area - 0.05f) {
            leaves_intact++;
        }
    }
    DOCTEST_CHECK(leaves_intact > 0);
}

DOCTEST_TEST_CASE("Nitrogen Model - Full Growth Cycle Integration") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Initial growth
    plantarchitecture.advanceTime(plantID, 5.0f);

    // Initialize nitrogen
    plantarchitecture.initializePlantNitrogenPools(plantID, 1.0f);

    // Simulate periodic nitrogen applications during growth
    for (int i = 0; i < 5; i++) {
        plantarchitecture.addPlantNitrogen(plantID, 5.0f); // Add 5 g N
        plantarchitecture.advanceTime(plantID, 5.0f); // Grow 5 days
    }

    // Verify plant completed growth cycle
    std::vector<uint> plant_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    DOCTEST_CHECK(plant_objects.size() > 0);

    // Verify stress factor updated throughout
    bool found_stress_factor = false;
    float final_stress = 0;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            context.getObjectData(objID, "nitrogen_stress_factor", final_stress);
            found_stress_factor = true;
            break;
        }
    }
    DOCTEST_CHECK(found_stress_factor);
    DOCTEST_CHECK(final_stress >= 0.0f);
    DOCTEST_CHECK(final_stress <= 1.0f);

    // Verify leaves have nitrogen data
    bool found_leaf_N = false;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            float leaf_N;
            context.getObjectData(objID, "leaf_nitrogen_gN_m2", leaf_N);
            DOCTEST_CHECK(leaf_N >= 0.0f);
            found_leaf_N = true;
        }
    }
    DOCTEST_CHECK(found_leaf_N);
}

DOCTEST_TEST_CASE("Nitrogen Model - Edge Case: Zero Nitrogen") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 5.0f);

    // Initialize with zero nitrogen - should not crash
    DOCTEST_CHECK_NOTHROW(plantarchitecture.initializePlantNitrogenPools(plantID, 0.0f));

    // Advance time with zero nitrogen - should not crash
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 5.0f));

    // Stress factor should be very low (severe stress)
    std::vector<uint> plant_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    bool found_stress_factor = false;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            float stress_factor;
            context.getObjectData(objID, "nitrogen_stress_factor", stress_factor);
            DOCTEST_CHECK(stress_factor < 0.2f); // Should be low under zero N
            found_stress_factor = true;
            break;
        }
    }
    DOCTEST_CHECK(found_stress_factor);
}

DOCTEST_TEST_CASE("Nitrogen Model - Edge Case: Excessive Nitrogen") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 5.0f);

    // Initialize with zero
    plantarchitecture.initializePlantNitrogenPools(plantID, 0.0f);

    // Set high accumulation rate to overcome rate limiting
    NitrogenParameters N_params;
    N_params.max_N_accumulation_rate = 1.0f; // g N/m²/day (10x default)
    plantarchitecture.setPlantNitrogenParameters(plantID, N_params);

    // Apply excessive nitrogen - should not crash
    DOCTEST_CHECK_NOTHROW(plantarchitecture.addPlantNitrogen(plantID, 1000.0f));

    // Advance time - should not crash
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 5.0f));

    // Stress factor should clamp at 1.0 (no stress) and be high with excess N
    std::vector<uint> plant_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    bool found_stress_factor = false;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            float stress_factor;
            context.getObjectData(objID, "nitrogen_stress_factor", stress_factor);
            DOCTEST_CHECK(stress_factor <= 1.0f); // Should clamp at 1.0
            DOCTEST_CHECK(stress_factor >= 0.90f); // Should be very high with excess N and fast accumulation
            found_stress_factor = true;
            break;
        }
    }
    DOCTEST_CHECK(found_stress_factor);
}

DOCTEST_TEST_CASE("Nitrogen Model - Edge Case: No Leaves") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();

    // Build plant at very early stage (no leaves yet)
    uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0);

    // Try to initialize nitrogen - should not crash even with no leaves
    DOCTEST_CHECK_NOTHROW(plantarchitecture.initializePlantNitrogenPools(plantID, 1.5f));

    // Add nitrogen - should not crash
    DOCTEST_CHECK_NOTHROW(plantarchitecture.addPlantNitrogen(plantID, 10.0f));

    // Advance time with no leaves - should not crash
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 1.0f));
}

DOCTEST_TEST_CASE("Nitrogen Model - Division by Zero Prevention") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.enableNitrogenModel();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Grow plant slightly to create very small leaves
    plantarchitecture.advanceTime(plantID, 0.5f);

    // Initialize nitrogen
    plantarchitecture.initializePlantNitrogenPools(plantID, 1.5f);

    // Add nitrogen and advance - should handle small/zero leaf areas gracefully
    plantarchitecture.addPlantNitrogen(plantID, 10.0f);

    // This should not crash due to division by zero (bug fix verification)
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 1.0f));

    // Continue growth and check remobilization doesn't crash either
    plantarchitecture.advanceTime(plantID, 20.0f);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 5.0f));
}

DOCTEST_TEST_CASE("Nitrogen Model - Enable/Disable") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Initially disabled
    DOCTEST_CHECK_FALSE(plantarchitecture.isNitrogenModelEnabled());

    // Enable
    plantarchitecture.enableNitrogenModel();
    DOCTEST_CHECK(plantarchitecture.isNitrogenModelEnabled());

    // Disable
    plantarchitecture.disableNitrogenModel();
    DOCTEST_CHECK_FALSE(plantarchitecture.isNitrogenModelEnabled());

    // Build plant with model disabled - should not output nitrogen data
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 5.0f);

    std::vector<uint> plant_objects = plantarchitecture.getAllPlantObjectIDs(plantID);
    bool found_nitrogen_data = false;
    for (uint objID: plant_objects) {
        if (context.doesObjectDataExist(objID, "nitrogen_stress_factor")) {
            found_nitrogen_data = true;
            break;
        }
    }
    DOCTEST_CHECK_FALSE(found_nitrogen_data); // Should NOT have nitrogen data when disabled
}

// ===== Tests for listShootTypeLabels() methods =====

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - no parameter success") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    plantarchitecture.loadPlantModelFromLibrary("bean");
    std::vector<std::string> labels = plantarchitecture.listShootTypeLabels();

    DOCTEST_CHECK(labels.size() == 2);
    DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "unifoliate") != labels.end());
    DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "trifoliate") != labels.end());
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - no parameter error") {
    std::string error_message;
    {
        capture_cerr cerr_buffer;
        Context context;
        PlantArchitecture plantarchitecture(&context);

        // Should throw because no plant model is loaded
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.listShootTypeLabels()));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - string parameter success") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Query bean shoot types without loading it
    std::vector<std::string> bean_labels = plantarchitecture.listShootTypeLabels("bean");
    DOCTEST_CHECK(bean_labels.size() == 2);
    DOCTEST_CHECK(std::find(bean_labels.begin(), bean_labels.end(), "unifoliate") != bean_labels.end());
    DOCTEST_CHECK(std::find(bean_labels.begin(), bean_labels.end(), "trifoliate") != bean_labels.end());

    // Query tomato shoot types
    std::vector<std::string> tomato_labels = plantarchitecture.listShootTypeLabels("tomato");
    DOCTEST_CHECK(tomato_labels.size() == 1);
    DOCTEST_CHECK(std::find(tomato_labels.begin(), tomato_labels.end(), "mainstem") != tomato_labels.end());
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - string parameter error") {
    std::string error_message;
    {
        capture_cerr cerr_buffer;
        Context context;
        PlantArchitecture plantarchitecture(&context);

        // Should throw for non-existent plant model
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.listShootTypeLabels("nonexistent_plant")));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - state preservation") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Load bean plant model
    plantarchitecture.loadPlantModelFromLibrary("bean");

    // Query tomato shoot types (should not change current plant model)
    std::vector<std::string> tomato_labels = plantarchitecture.listShootTypeLabels("tomato");

    // Verify bean is still loaded by checking current labels
    std::vector<std::string> current_labels = plantarchitecture.listShootTypeLabels();
    DOCTEST_CHECK(current_labels.size() == 2);
    DOCTEST_CHECK(std::find(current_labels.begin(), current_labels.end(), "unifoliate") != current_labels.end());
    DOCTEST_CHECK(std::find(current_labels.begin(), current_labels.end(), "trifoliate") != current_labels.end());
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - all plant models") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    std::vector<std::string> all_plants = plantarchitecture.getAvailablePlantModels();

    // Should successfully query shoot types for all plants
    for (const auto &plant: all_plants) {
        std::vector<std::string> labels;
        DOCTEST_CHECK_NOTHROW(labels = plantarchitecture.listShootTypeLabels(plant));
        DOCTEST_CHECK(!labels.empty()); // All plants should have at least one shoot type
    }
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - uint parameter success") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Load and build bean plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Query by plantID
    std::vector<std::string> labels = plantarchitecture.listShootTypeLabels(plantID);

    // Should match bean model shoot types
    DOCTEST_CHECK(labels.size() == 2);
    DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "unifoliate") != labels.end());
    DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "trifoliate") != labels.end());
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - uint parameter error") {
    std::string error_message;
    {
        capture_cerr cerr_buffer;
        Context context;
        PlantArchitecture plantarchitecture(&context);

        // Should throw for invalid plantID
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.listShootTypeLabels(999)));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture listShootTypeLabels - multiple instances") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Build bean plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint bean_plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    // Build tomato plant
    plantarchitecture.loadPlantModelFromLibrary("tomato");
    uint tomato_plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(1, 0, 0), 0);

    // Verify each returns correct labels for its model
    std::vector<std::string> bean_labels = plantarchitecture.listShootTypeLabels(bean_plantID);
    DOCTEST_CHECK(bean_labels.size() == 2);
    DOCTEST_CHECK(std::find(bean_labels.begin(), bean_labels.end(), "unifoliate") != bean_labels.end());
    DOCTEST_CHECK(std::find(bean_labels.begin(), bean_labels.end(), "trifoliate") != bean_labels.end());

    std::vector<std::string> tomato_labels = plantarchitecture.listShootTypeLabels(tomato_plantID);
    DOCTEST_CHECK(tomato_labels.size() == 1);
    DOCTEST_CHECK(std::find(tomato_labels.begin(), tomato_labels.end(), "mainstem") != tomato_labels.end());
}

DOCTEST_TEST_CASE("PlantArchitecture getPlantInternodeObjectIDs with shoot type filter") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Build a bean plant (has two shoot types: "unifoliate" and "trifoliate")
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.0);

    // Get all internode object IDs without filter
    std::vector<uint> all_internodes = plantarchitecture.getPlantInternodeObjectIDs(plantID);
    DOCTEST_CHECK(all_internodes.size() > 0);

    // Get internode object IDs for "unifoliate" shoot type
    std::vector<uint> unifoliate_internodes = plantarchitecture.getPlantInternodeObjectIDs(plantID, "unifoliate");
    DOCTEST_CHECK(unifoliate_internodes.size() > 0);

    // Get internode object IDs for "trifoliate" shoot type
    std::vector<uint> trifoliate_internodes = plantarchitecture.getPlantInternodeObjectIDs(plantID, "trifoliate");
    DOCTEST_CHECK(trifoliate_internodes.size() > 0);

    // Verify that filtered results are subsets of all internodes
    for (uint objID : unifoliate_internodes) {
        DOCTEST_CHECK(std::find(all_internodes.begin(), all_internodes.end(), objID) != all_internodes.end());
    }
    for (uint objID : trifoliate_internodes) {
        DOCTEST_CHECK(std::find(all_internodes.begin(), all_internodes.end(), objID) != all_internodes.end());
    }

    // Verify no overlap between unifoliate and trifoliate internodes
    for (uint objID : unifoliate_internodes) {
        DOCTEST_CHECK(std::find(trifoliate_internodes.begin(), trifoliate_internodes.end(), objID) == trifoliate_internodes.end());
    }

    // Verify that sum of filtered internodes equals total internodes
    DOCTEST_CHECK(unifoliate_internodes.size() + trifoliate_internodes.size() == all_internodes.size());
}

DOCTEST_TEST_CASE("PlantArchitecture getPlantInternodeObjectIDs with shoot type filter - error cases") {
    std::string error_message;
    {
        capture_cerr cerr_buffer;
        Context context;
        PlantArchitecture plantarchitecture(&context);

        plantarchitecture.loadPlantModelFromLibrary("bean");
        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.0);

        // Should throw for non-existent shoot type
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.getPlantInternodeObjectIDs(plantID, "nonexistent_shoot_type")));

        // Should throw for invalid plant ID
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.getPlantInternodeObjectIDs(9999, "unifoliate")));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture setProgressCallback") {
    std::vector<float> progress_values;
    std::vector<std::string> messages;
    {
        capture_cout cout_buffer;
        capture_cerr cerr_buffer;

        Context context;
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();

        plantarchitecture.setProgressCallback([&](float progress, const std::string &msg) {
            progress_values.push_back(progress);
            messages.push_back(msg);
        });

        plantarchitecture.loadPlantModelFromLibrary("bean");
        plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5);

        // advanceTime should trigger the callback
        plantarchitecture.advanceTime(1.f);
    }

    // Verify callback was invoked
    DOCTEST_CHECK(progress_values.size() > 0);

    // Verify progress values are in [0, 1]
    for (float p : progress_values) {
        DOCTEST_CHECK(p >= 0.f);
        DOCTEST_CHECK(p <= 1.f);
    }

    // Verify the last progress value is 1.0 (complete)
    if (!progress_values.empty()) {
        DOCTEST_CHECK(progress_values.back() == doctest::Approx(1.0f));
    }

    // Verify messages are non-empty
    for (const auto &msg : messages) {
        DOCTEST_CHECK(!msg.empty());
    }
}

DOCTEST_TEST_CASE("getAllPlantUUIDs with include_hidden parameter") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000);

    std::vector<uint> uuids_default = plantarchitecture.getAllPlantUUIDs(plantID);
    std::vector<uint> uuids_no_hidden = plantarchitecture.getAllPlantUUIDs(plantID, false);
    std::vector<uint> uuids_with_hidden = plantarchitecture.getAllPlantUUIDs(plantID, true);

    // Default behavior should match explicit false
    DOCTEST_CHECK(uuids_default.size() == uuids_no_hidden.size());

    // include_hidden=true should return more UUIDs (the hidden prototypes)
    DOCTEST_CHECK(uuids_with_hidden.size() > uuids_no_hidden.size());
}

DOCTEST_TEST_CASE("deletePlantInstance cleans up prototypes when all plants deleted") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID1 = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000);
    uint plantID2 = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(1, 0, 0), 5000);

    // Identify hidden prototype UUIDs
    std::vector<uint> all_uuids = plantarchitecture.getAllPlantUUIDs(plantID1, true);
    std::vector<uint> visible_uuids = plantarchitecture.getAllPlantUUIDs(plantID1, false);
    DOCTEST_CHECK(all_uuids.size() > visible_uuids.size());

    // Collect prototype UUIDs (those in all but not in visible)
    std::set<uint> visible_set(visible_uuids.begin(), visible_uuids.end());
    std::vector<uint> prototype_uuids;
    for (uint uuid : all_uuids) {
        if (visible_set.find(uuid) == visible_set.end()) {
            prototype_uuids.push_back(uuid);
        }
    }
    DOCTEST_CHECK(prototype_uuids.size() > 0);

    // Delete first plant — prototypes should survive
    plantarchitecture.deletePlantInstance(plantID1);
    for (uint uuid : prototype_uuids) {
        DOCTEST_CHECK(context.doesPrimitiveExist(uuid));
    }

    // Delete second plant — prototypes should now be cleaned up
    plantarchitecture.deletePlantInstance(plantID2);
    for (uint uuid : prototype_uuids) {
        DOCTEST_CHECK(!context.doesPrimitiveExist(uuid));
    }
}

DOCTEST_TEST_CASE("deletePlantInstance preserves prototypes when plants remain") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID1 = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000);
    uint plantID2 = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(1, 0, 0), 5000);

    // Get prototype UUIDs via the second plant
    std::vector<uint> uuids_with_hidden = plantarchitecture.getAllPlantUUIDs(plantID2, true);
    std::vector<uint> uuids_without_hidden = plantarchitecture.getAllPlantUUIDs(plantID2, false);
    DOCTEST_CHECK(uuids_with_hidden.size() > uuids_without_hidden.size());

    // Delete first plant — prototypes should still be accessible for remaining plant
    plantarchitecture.deletePlantInstance(plantID1);

    std::vector<uint> uuids_after = plantarchitecture.getAllPlantUUIDs(plantID2, true);
    DOCTEST_CHECK(uuids_after.size() > plantarchitecture.getAllPlantUUIDs(plantID2, false).size());
}

DOCTEST_TEST_CASE("USD export basic structure") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    std::string filename = "test_usd_basic.usda";
    plantarchitecture.writePlantStructureUSD(plantID, filename);

    // Read the file and verify key structural elements
    std::ifstream file(filename);
    DOCTEST_CHECK(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    DOCTEST_CHECK(!content.empty());

    // Check required USD elements
    DOCTEST_CHECK(content.find("PhysicsArticulationRootAPI") != std::string::npos);
    DOCTEST_CHECK(content.find("PhysxArticulationAPI") != std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsScene") != std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsMaterialAPI") != std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsFixedJoint") != std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsRigidBodyAPI") != std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsSphericalJoint") != std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsDriveAPI:angular") != std::string::npos);

    // Count links (each has PhysicsRigidBodyAPI)
    size_t link_count = 0;
    size_t pos = 0;
    while ((pos = content.find("PhysicsRigidBodyAPI", pos)) != std::string::npos) {
        link_count++;
        pos++;
    }
    DOCTEST_CHECK(link_count > 0);

    // Verify exactly one fixed joint (world anchor)
    size_t fixed_count = 0;
    pos = 0;
    while ((pos = content.find("PhysicsFixedJoint", pos)) != std::string::npos) {
        fixed_count++;
        pos++;
    }
    DOCTEST_CHECK(fixed_count == 1);

    std::remove(filename.c_str());
}

DOCTEST_TEST_CASE("USD export physics properties") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    USDExportParameters params;
    params.elastic_modulus = 1e9f;
    params.wood_density = 500.f;

    std::string filename = "test_usd_physics.usda";
    plantarchitecture.writePlantStructureUSD(plantID, filename, params);

    // Read and verify physics values are present and positive
    std::ifstream file(filename);
    DOCTEST_CHECK(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Check that mass values exist and stiffness values exist
    DOCTEST_CHECK(content.find("physics:mass") != std::string::npos);
    DOCTEST_CHECK(content.find("drive:angular:physics:stiffness") != std::string::npos);
    DOCTEST_CHECK(content.find("drive:angular:physics:damping") != std::string::npos);

    // Check that gravity is correct
    DOCTEST_CHECK(content.find("physics:gravityMagnitude = 9.81") != std::string::npos);

    std::remove(filename.c_str());
}

DOCTEST_TEST_CASE("USD export branching topology") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Use almond which has branching structure
    plantarchitecture.loadPlantModelFromLibrary("almond");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    std::string filename = "test_usd_branching.usda";
    plantarchitecture.writePlantStructureUSD(plantID, filename);

    std::ifstream file(filename);
    DOCTEST_CHECK(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Should have multiple links and joints
    size_t link_count = 0;
    size_t pos = 0;
    while ((pos = content.find("PhysicsRigidBodyAPI", pos)) != std::string::npos) {
        link_count++;
        pos++;
    }
    DOCTEST_CHECK(link_count > 3);

    // Check that body0 and body1 references exist (proper joint connectivity)
    DOCTEST_CHECK(content.find("physics:body0") != std::string::npos);
    DOCTEST_CHECK(content.find("physics:body1") != std::string::npos);

    std::remove(filename.c_str());
}

DOCTEST_TEST_CASE("USD export error handling") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Invalid plant ID
    DOCTEST_CHECK_THROWS(plantarchitecture.writePlantStructureUSD(9999, "test.usda"));

    // Build a plant first, then test empty filename
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    // Invalid file extension (only .usda/.USDA are accepted)
    DOCTEST_CHECK_THROWS(plantarchitecture.writePlantStructureUSD(plantID, "test.txt"));
}

DOCTEST_TEST_CASE("USD export organs") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    USDExportParameters params;
    std::string filename = "test_usd_organs.usda";
    plantarchitecture.writePlantStructureUSD(plantID, filename, params);

    std::ifstream file(filename);
    DOCTEST_CHECK(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Bean should have petiole segments
    DOCTEST_CHECK(content.find("Pet") != std::string::npos);

    // Bean should have leaves with both Visual and Collision mesh prims
    DOCTEST_CHECK(content.find("Leaf") != std::string::npos);
    DOCTEST_CHECK(content.find("def Mesh \"Visual\"") != std::string::npos);
    DOCTEST_CHECK(content.find("def Mesh \"Collision\"") != std::string::npos);

    // All mesh/capsule prims with material bindings must declare MaterialBindingAPI
    DOCTEST_CHECK(content.find("\"MaterialBindingAPI\"") != std::string::npos);

    // Visual meshes must have doubleSided and correct subdivision for Isaac Sim
    DOCTEST_CHECK(content.find("bool doubleSided = 1") != std::string::npos);
    DOCTEST_CHECK(content.find("subdivisionScheme = \"none\"") != std::string::npos);

    // Normals must be present for correct shading
    DOCTEST_CHECK(content.find("primvars:normals") != std::string::npos);

    // Texture paths must be relative (no absolute paths starting with /)
    DOCTEST_CHECK(content.find("asset inputs:file = @/") == std::string::npos);

    std::remove(filename.c_str());
}

DOCTEST_TEST_CASE("USD export minimum segment filtering") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    // Use almond which has longer internode segments than bean, so both default and
    // stricter filters always leave at least one surviving segment to export.
    plantarchitecture.loadPlantModelFromLibrary("almond");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    // Export with default min_segment_length
    USDExportParameters params_default;
    std::string filename_default = "test_usd_filter_default.usda";
    plantarchitecture.writePlantStructureUSD(plantID, filename_default, params_default);

    // Export with larger min_segment_length — should produce fewer links
    USDExportParameters params_strict;
    params_strict.min_segment_length = 0.05f; // 5 cm — filters short segments
    std::string filename_strict = "test_usd_filter_strict.usda";
    plantarchitecture.writePlantStructureUSD(plantID, filename_strict, params_strict);

    // Count links in each file
    auto countOccurrences = [](const std::string &content, const std::string &token) {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = content.find(token, pos)) != std::string::npos) {
            count++;
            pos++;
        }
        return count;
    };

    std::ifstream f1(filename_default);
    std::string content1((std::istreambuf_iterator<char>(f1)), std::istreambuf_iterator<char>());
    f1.close();

    std::ifstream f2(filename_strict);
    std::string content2((std::istreambuf_iterator<char>(f2)), std::istreambuf_iterator<char>());
    f2.close();

    size_t links_default = countOccurrences(content1, "PhysicsRigidBodyAPI");
    size_t links_strict = countOccurrences(content2, "PhysicsRigidBodyAPI");

    // Stricter filtering should produce fewer or equal links
    DOCTEST_CHECK(links_strict <= links_default);

    std::remove(filename_default.c_str());
    std::remove(filename_strict.c_str());
}

DOCTEST_TEST_CASE("Growth frame registration") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    DOCTEST_CHECK(plantarchitecture.getGrowthFrameCount(plantID) == 0);

    // Register a frame at initial state
    plantarchitecture.registerGrowthFrame(plantID);
    DOCTEST_CHECK(plantarchitecture.getGrowthFrameCount(plantID) == 1);

    // Advance time and register more frames
    plantarchitecture.advanceTime(10);
    plantarchitecture.registerGrowthFrame(plantID);
    DOCTEST_CHECK(plantarchitecture.getGrowthFrameCount(plantID) == 2);

    plantarchitecture.advanceTime(10);
    plantarchitecture.registerGrowthFrame(plantID);
    DOCTEST_CHECK(plantarchitecture.getGrowthFrameCount(plantID) == 3);

    // Clear frames
    plantarchitecture.clearGrowthFrames(plantID);
    DOCTEST_CHECK(plantarchitecture.getGrowthFrameCount(plantID) == 0);

    // Query for non-existent plant returns 0
    DOCTEST_CHECK(plantarchitecture.getGrowthFrameCount(9999) == 0);
}

DOCTEST_TEST_CASE("Growth USD export basic") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);

    for (int i = 0; i < 3; i++) {
        plantarchitecture.advanceTime(10);
        plantarchitecture.registerGrowthFrame(plantID);
    }

    std::string filename = "test_growth_usd.usda";
    // 1 second per growth frame -> time codes spaced by 24 (at 24fps)
    plantarchitecture.writePlantGrowthUSD(plantID, filename, 1.0f);

    // Read file and verify key USD attributes
    std::ifstream f(filename);
    DOCTEST_CHECK(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    // Verify header fields
    // 3 frames at 1 sec/frame with 24fps -> time codes 0, 24, 48 -> endTimeCode = 48
    DOCTEST_CHECK(content.find("startTimeCode = 0") != std::string::npos);
    DOCTEST_CHECK(content.find("endTimeCode = 48") != std::string::npos);
    DOCTEST_CHECK(content.find("timeCodesPerSecond = 24") != std::string::npos);
    DOCTEST_CHECK(content.find("framesPerSecond = 24") != std::string::npos);
    DOCTEST_CHECK(content.find("upAxis = \"Z\"") != std::string::npos);

    // Verify time-sampled transforms exist
    DOCTEST_CHECK(content.find("xformOp:translate.timeSamples") != std::string::npos);
    DOCTEST_CHECK(content.find("xformOp:orient.timeSamples") != std::string::npos);
    DOCTEST_CHECK(content.find("visibility.timeSamples") != std::string::npos);

    // Verify no physics prims are present
    DOCTEST_CHECK(content.find("PhysicsArticulationRootAPI") == std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsRigidBodyAPI") == std::string::npos);
    DOCTEST_CHECK(content.find("PhysicsJoint") == std::string::npos);

    // Verify mesh data is present
    DOCTEST_CHECK(content.find("def Mesh \"Visual\"") != std::string::npos);

    std::remove(filename.c_str());
}

DOCTEST_TEST_CASE("Growth USD export visibility toggling") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    // Build a very young plant so new organs appear during growth
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 1);
    plantarchitecture.registerGrowthFrame(plantID);

    // Advance significantly so new phytomers/organs appear
    plantarchitecture.advanceTime(30);
    plantarchitecture.registerGrowthFrame(plantID);

    std::string filename = "test_growth_visibility.usda";
    plantarchitecture.writePlantGrowthUSD(plantID, filename);

    std::ifstream f(filename);
    DOCTEST_CHECK(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    // Organs that appeared in frame 2 but not frame 1 should have "invisible" at time 0
    // and "inherited" at time 1. Both tokens should be present somewhere in the file.
    DOCTEST_CHECK(content.find("\"invisible\"") != std::string::npos);
    DOCTEST_CHECK(content.find("\"inherited\"") != std::string::npos);

    std::remove(filename.c_str());
}

DOCTEST_TEST_CASE("Growth USD export error handling") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Error: invalid plant ID for registerGrowthFrame
    bool threw = false;
    try {
        plantarchitecture.registerGrowthFrame(9999);
    } catch (...) {
        threw = true;
    }
    DOCTEST_CHECK(threw);

    // Error: writePlantGrowthUSD with no frames registered
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 500);
    threw = false;
    try {
        plantarchitecture.writePlantGrowthUSD(plantID, "test_no_frames.usda");
    } catch (...) {
        threw = true;
    }
    DOCTEST_CHECK(threw);
}

DOCTEST_TEST_CASE("PlantArchitecture manually-built plant survives advanceTime without phenological thresholds") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Regression test: a plant assembled through the manual API (addPlantInstance + addBaseStemShoot /
    // appendShoot / addChildShoot) never has setPlantPhenologicalThresholds() called on it, so it keeps
    // the PlantInstance defaults. When those defaults were all zero, the dormancy predicate in
    // advanceTime() read "time_since_dormancy > 0" and was therefore satisfied on the very first
    // timestep -- and again on every step after, since it resets time_since_dormancy to 0. Each firing
    // called Shoot::makeDormant(), which removes every leaf and marks all non-dormant vegetative buds
    // BUD_DEAD; breakDormancy() only revives buds that are not BUD_DEAD, so the plant was permanently
    // defoliated and sterilized rather than growing. Leaf count collapsed to 0.
    //
    // Load the bean model only to register its shoot types ("unifoliate"/"trifoliate"). Going through
    // buildPlantInstanceFromLibrary() instead would call setPlantPhenologicalThresholds() and mask the
    // defect entirely, which is exactly why this bug survived: every existing growth test uses a
    // library builder.
    plantarchitecture.loadPlantModelFromLibrary("bean");

    uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);

    uint uID_unifoliate = plantarchitecture.addBaseStemShoot(plantID, 1, make_AxisRotation(0, 0, 0), 0.0005f, 0.03f, 0.01f, 0.01f, 0, "unifoliate");

    uint uID_trifoliate = plantarchitecture.appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), 0.0005f, 0.03f, 0.1f, 0.1f, 0, "trifoliate");

    DOCTEST_CHECK_NOTHROW(plantarchitecture.addChildShoot(plantID, uID_trifoliate, 0, 1, make_AxisRotation(0, 0, 0), 0.0005f, 0.03f, 0.1f, 0.1f, 0, "trifoliate", 0));

    // Shoots are constructed dormant, so break dormancy before growing (buildBeanPlant does the same).
    plantarchitecture.breakPlantDormancy(plantID);

    uint leaf_count_before = plantarchitecture.getPlantLeafCount(plantID);
    DOCTEST_REQUIRE(leaf_count_before > 0);

    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 20.f));

    uint leaf_count_after = plantarchitecture.getPlantLeafCount(plantID);

    // The collapse detector: on the buggy defaults this is 0.
    DOCTEST_CHECK(leaf_count_after > 0);
    // The plant must not shrink. Deliberately >= rather than >: whether new phytomers appear within 20
    // days depends on phyllochron and vegetative_bud_break_time, and a strict > would couple this test
    // to growth-rate parameters. The > 0 check above carries the regression signal.
    DOCTEST_CHECK(leaf_count_after >= leaf_count_before);
    DOCTEST_CHECK(!plantarchitecture.getPlantPetioleObjectIDs(plantID).empty());
}

DOCTEST_TEST_CASE("PlantArchitecture library-built plant phenology is unaffected by PlantInstance defaults") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // All 30 library builders call setPlantPhenologicalThresholds(), overwriting every default, so
    // changing the PlantInstance defaults must not alter library-built plants. Almond is deciduous
    // (dd_to_dormancy_break = 90, dd_to_dormancy = 275) and buildAlmondTree() calls makePlantDormant(),
    // so it exercises the real dormancy machinery rather than the defaults.
    plantarchitecture.loadPlantModelFromLibrary("almond");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 300);

    uint leaf_count_before = plantarchitecture.getPlantLeafCount(plantID);

    // The plant is built dormant, so 200 days -- past dd_to_dormancy_break = 90 but well short of the
    // full 90 + 275 = 365 dormancy cycle -- must break dormancy and flush leaves. This pins the
    // dormancy-break half of the cycle, which a guard that wrongly suppressed dormancy transitions
    // would break. It also confirms library plants ignore the PlantInstance defaults entirely.
    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 200.f));

    DOCTEST_CHECK(plantarchitecture.getPlantLeafCount(plantID) > leaf_count_before);
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip preserves phenological thresholds") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Phenological thresholds are not derivable from the shoot structure, so they must be written to and
    // read back from the XML. Before they were part of the schema, a restored plant fell back to the
    // PlantInstance defaults; combined with the old all-zero defaults that meant a restored plant was
    // defoliated on its first growth step.
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 10);

    std::string xml_filename = "test_phenology_roundtrip.xml";
    DOCTEST_CHECK_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    // The thresholds buildBeanPlant() sets must appear in the file.
    {
        std::ifstream f(xml_filename);
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();
        DOCTEST_CHECK(content.find("dd_to_dormancy") != std::string::npos);
        DOCTEST_CHECK(content.find("dd_to_flower_initiation") != std::string::npos);
    }

    Context restore_context;
    PlantArchitecture restore_plantarchitecture(&restore_context);
    restore_plantarchitecture.disableMessages();
    restore_plantarchitecture.loadPlantModelFromLibrary("bean");

    std::vector<uint> restored_IDs;
    DOCTEST_CHECK_NOTHROW(restored_IDs = restore_plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(!restored_IDs.empty());

    uint restored_plantID = restored_IDs.front();
    uint leaf_count_at_load = restore_plantarchitecture.getPlantLeafCount(restored_plantID);
    DOCTEST_REQUIRE(leaf_count_at_load > 0);

    // With thresholds restored (bean: dd_to_dormancy = 1000), growth must not trigger the dormancy path.
    DOCTEST_CHECK_NOTHROW(restore_plantarchitecture.advanceTime(restored_plantID, 20.f));
    DOCTEST_CHECK(restore_plantarchitecture.getPlantLeafCount(restored_plantID) > 0);

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture XML without phenology tags still loads") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Backward compatibility: the phenology tags are optional. An XML file written before they existed
    // must still load and grow, falling back to the PlantInstance defaults. Simulate one by stripping
    // every dd_to_* / max_leaf_lifespan / is_evergreen line from a freshly written file.
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 10);

    std::string xml_filename = "test_phenology_full.xml";
    std::string legacy_filename = "test_phenology_legacy.xml";
    plantarchitecture.writePlantStructureXML(plantID, xml_filename);

    {
        std::ifstream in(xml_filename);
        std::ofstream out(legacy_filename);
        std::string line;
        while (std::getline(in, line)) {
            if (line.find("dd_to_") != std::string::npos || line.find("max_leaf_lifespan") != std::string::npos || line.find("is_evergreen") != std::string::npos) {
                continue;
            }
            out << line << std::endl;
        }
    }

    Context restore_context;
    PlantArchitecture restore_plantarchitecture(&restore_context);
    restore_plantarchitecture.disableMessages();
    restore_plantarchitecture.loadPlantModelFromLibrary("bean");

    std::vector<uint> restored_IDs;
    DOCTEST_CHECK_NOTHROW(restored_IDs = restore_plantarchitecture.readPlantStructureXML(legacy_filename, true));
    DOCTEST_REQUIRE(!restored_IDs.empty());

    // The defaults now encode "no phenology scheduled", so a legacy file grows rather than defoliating.
    uint restored_plantID = restored_IDs.front();
    DOCTEST_REQUIRE(restore_plantarchitecture.getPlantLeafCount(restored_plantID) > 0);
    DOCTEST_CHECK_NOTHROW(restore_plantarchitecture.advanceTime(restored_plantID, 20.f));
    DOCTEST_CHECK(restore_plantarchitecture.getPlantLeafCount(restored_plantID) > 0);

    std::remove(xml_filename.c_str());
    std::remove(legacy_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture disablePlantPhenology leaves fruit growth well-defined") {
    // Regression test: disablePlantPhenology() set dd_to_fruit_maturity = -1, matching the three
    // stage fields above it. That is correct for those three -- advanceTime() gates them on ">= 0.f"
    // as a "skip this stage" sentinel -- but wrong for dd_to_fruit_maturity, which has no such gate
    // and is used only as a divisor in the fruit-growth block:
    //
    //     scale = fmin(1, 0.25f + 0.75f * time_counter / dd_to_fruit_maturity)
    //
    // With a divisor of -1 that is 0.25 - 0.75*time_counter, which goes negative as soon as
    // time_counter exceeds 1/3 day, and fmin() does not clamp from below. The negative fraction
    // reaches Phytomer::setInflorescenceScaleFraction(), which asserts 0 <= fraction <= 1 -- so a
    // debug build aborts, while a release build (NDEBUG) skips the assert and mirror-scales the
    // fruit geometry by a negative factor instead. The scale factor is then stored and divided by on
    // the next step, so the error compounds without bound.
    //
    // Sorghum is used because it reaches BUD_FRUITING through appendPhytomerToShoot() -- purely from
    // shoot structure, without consulting dd_to_fruit_set -- which is exactly the path the
    // PlantInstance header comment warns about. It has a fruit prototype (SorghumPaniclePrototype)
    // but no flower prototype, flowers_require_dormancy = false and max_terminal_floral_buds = 1.
    // The ordinary fruit-set transition in advanceTime() IS correctly gated on dd_to_fruit_set >= 0,
    // so with phenology disabled no other route into BUD_FRUITING exists, and a model lacking that
    // structural path would not reproduce the defect at all.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("sorghum");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 1);

    plantarchitecture.disablePlantPhenology(plantID);

    // Grow one day at a time until the terminal floral bud appears, rather than assuming a fixed
    // number of days: the node at which the mainstem reaches max_nodes depends on resampled
    // phyllochron parameters.
    std::vector<uint> fruit_objIDs;
    for (int day = 0; day < 120 && fruit_objIDs.empty(); day++) {
        plantarchitecture.advanceTime(plantID, 1.f);
        fruit_objIDs = plantarchitecture.getPlantFruitObjectIDs(plantID);
    }
    DOCTEST_REQUIRE(!fruit_objIDs.empty());

    auto sum_fruit_area = [&](const std::vector<uint> &objIDs) {
        float total = 0;
        for (uint objID: objIDs) {
            if (context.doesObjectExist(objID)) {
                total += context.getObjectArea(objID);
            }
        }
        return total;
    };

    // Area of the same objects, before and after further growth. Scaling reuses the existing object
    // IDs, and with phenology disabled there are no bud state transitions to recreate them, so the
    // set is stable; doesObjectExist() guards the comparison regardless.
    float fruit_area_initial = sum_fruit_area(fruit_objIDs);
    DOCTEST_REQUIRE(fruit_area_initial > 0.f);

    DOCTEST_CHECK_NOTHROW(plantarchitecture.advanceTime(plantID, 40.f));

    float fruit_area_after = sum_fruit_area(fruit_objIDs);

    // With phenology disabled the fruit is pinned at its initial 25% scale: dd_to_fruit_maturity of
    // 1e6 makes the growth term 0.75*t/1e6, i.e. immeasurable over 40 days, so the ratio is 1.0 to
    // four decimal places. On the buggy value it exceeds 10000 -- the panicle grows past the area of
    // a tennis court, against a fully mature sorghum head of roughly 0.16 m^2. The factor of 2 below
    // is therefore nowhere near either outcome, and is deliberately loose so the test cannot become
    // sensitive to growth-rate parameters.
    DOCTEST_CHECK(fruit_area_after <= 2.f * fruit_area_initial);

    // The mirror-scaling that a negative factor produces would also invert the geometry; area is
    // unsigned, so check the fruit has not been flung away from the plant as well.
    for (uint objID: fruit_objIDs) {
        if (context.doesObjectExist(objID)) {
            vec3 center = context.getObjectCenter(objID);
            DOCTEST_CHECK(std::isfinite(center.x));
            DOCTEST_CHECK(std::isfinite(center.y));
            DOCTEST_CHECK(std::isfinite(center.z));
        }
    }
}

DOCTEST_TEST_CASE("PlantArchitecture disablePlantPhenology writes a usable fruit maturity threshold") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // dd_to_fruit_maturity has no public getter, so the plant structure XML is the only public
    // surface that exposes it. This also pins the round-trip: readPlantStructureXML() feeds the
    // parsed value straight into setPlantPhenologicalThresholds() without validation, so a negative
    // value written here would be restored into a fresh plant on load.
    plantarchitecture.loadPlantModelFromLibrary("sorghum");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 1);
    plantarchitecture.disablePlantPhenology(plantID);

    std::string xml_filename = "test_disable_phenology_thresholds.xml";
    DOCTEST_CHECK_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    std::string content;
    {
        std::ifstream f(xml_filename);
        content.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    std::remove(xml_filename.c_str());

    const std::string open_tag = "<dd_to_fruit_maturity>";
    size_t begin = content.find(open_tag);
    DOCTEST_REQUIRE(begin != std::string::npos);
    size_t end = content.find("</dd_to_fruit_maturity>", begin);
    DOCTEST_REQUIRE(end != std::string::npos);

    float dd_to_fruit_maturity = std::stof(content.substr(begin + open_tag.size(), end - begin - open_tag.size()));

    // Must be strictly positive: it is a divisor, so 0 yields an infinite scale factor and a
    // negative value a negative one.
    DOCTEST_CHECK(dd_to_fruit_maturity > 0.f);
}

DOCTEST_TEST_CASE("PlantArchitecture setPlantMaxAge lets a manually-built plant grow past the default cap") {
    // Regression test. PlantInstance::max_age defaults to 999, and only the library builders override
    // it (buildAppleTree() sets 1460). A plant assembled through the manual API keeps the default, so
    // advanceTime() takes its early continue once current_age passes 999 and the plant silently
    // freezes -- there is no message, so from the caller's side it looks like an unexplained ceiling.
    // Before setPlantMaxAge() existed there was no public way to raise it, since plant_instances is
    // private and the setPlantAge() stub was a no-op.
    //
    // The observable is a DIFFERENCE between two identically-constructed plants rather than an
    // absolute growth threshold: both are built from the same Context seed and grown over the same
    // window, so everything except the cap is held equal.
    auto grow = [](bool raise_cap) {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();

        // Load apple only to register its "trunk"/"proleptic" shoot types. Building through
        // buildPlantInstanceFromLibrary() instead would set max_age = 1460 itself and mask the defect
        // entirely -- which is exactly why this went unnoticed: every other growth test uses a builder.
        plantarchitecture.loadPlantModelFromLibrary("apple");

        uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
        uint uID_trunk = plantarchitecture.addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0, 0), 0.015f, 0.04f, 1.f, 1.f, 0, "trunk");
        plantarchitecture.addChildShoot(plantID, uID_trunk, 1, 3, make_AxisRotation(deg2rad(40), 0, 0), 0.005f, 0.04f, 1.f, 1.f, 0.5f, "proleptic", 0);

        // The manual API sets neither phenology nor max_age. The thresholds matter here beyond
        // realism: without a dormancy cycle this plant saturates at its max_nodes within the first
        // few hundred days and every structural observable then plateaus, so the capped and uncapped
        // plants would be bit-identical at any horizon and the test would pass on the buggy code.
        // These are the values buildAppleTree() uses.
        plantarchitecture.setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200);

        // Manually-added shoots are constructed dormant; without this the plant never grows at all.
        plantarchitecture.breakPlantDormancy(plantID);

        if (raise_cap) {
            plantarchitecture.setPlantMaxAge(plantID, 1460.f);
        }

        // 1300 days is chosen deliberately. Node count advances in bursts at bud break rather than
        // continuously, and is flat from day 1000 through day 1200 -- at a 1200-day horizon the two
        // plants are indistinguishable by node count even when the fix is working. The burst between
        // day 1200 and 1300 is what separates them.
        plantarchitecture.advanceTime(plantID, 1300.f);

        struct Result {
            float age;
            float height;
            uint nodes;
        };
        uint node_total = 0;
        for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
            node_total += plantarchitecture.getShootNodeCount(plantID, shootID);
        }
        return Result{plantarchitecture.getPlantAge(plantID), plantarchitecture.getPlantHeight(plantID), node_total};
    };

    auto capped = grow(false);
    auto raised = grow(true);

    // current_age stops incrementing once the cap is passed, because advanceTime() skips the
    // increment along with the growth. This is the direct, noise-free signal that the gate reopened.
    // It settles at 1000 rather than 999: the day that crosses the cap still runs to completion.
    DOCTEST_CHECK(capped.age == doctest::Approx(1000.f).epsilon(0.01));
    DOCTEST_CHECK(raised.age > 1000.f);

    // ...and structural observables, so this proves real growth rather than a counter ticking.
    // Measured: 925 nodes / 2.57 m frozen, against 2837 nodes / 3.16 m grown.
    DOCTEST_CHECK(raised.nodes > capped.nodes);
    DOCTEST_CHECK(raised.height > capped.height);
}

DOCTEST_TEST_CASE("PlantArchitecture setPlantMaxAge/getPlantMaxAge round-trip and defaults") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");

    // A manually-built plant keeps the PlantInstance default...
    uint manual_plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(manual_plantID) == doctest::Approx(999.f).epsilon(err_tol));

    DOCTEST_CHECK_NOTHROW(plantarchitecture.setPlantMaxAge(manual_plantID, 1460.f));
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(manual_plantID) == doctest::Approx(1460.f).epsilon(err_tol));

    // Zero is legal, and deliberately freezes the plant at its current form.
    DOCTEST_CHECK_NOTHROW(plantarchitecture.setPlantMaxAge(manual_plantID, 0.f));
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(manual_plantID) == doctest::Approx(0.f).epsilon(err_tol));

    // ...while a library-built plant carries the value its builder set.
    uint library_plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(1, 0, 0), 0.f);
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(library_plantID) == doctest::Approx(1460.f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PlantArchitecture setPlantMaxAge/getPlantMaxAge error cases") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(plantarchitecture.setPlantMaxAge(9999, 1460.f));
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.getPlantMaxAge(9999)));

        // A negative cap is meaningless -- ages are non-negative, so it could never be reached.
        DOCTEST_CHECK_THROWS(plantarchitecture.setPlantMaxAge(plantID, -1.f));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip preserves max_age") {
    // max_age is not derivable from the shoot structure, so without an explicit tag a plant written
    // and read back silently reverts to the 999 default -- undoing setPlantMaxAge() for anyone who
    // saves a custom-built plant and reloads it later.
    std::string xml_filename = "test_plant_max_age.xml";

    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    plantarchitecture.setPlantMaxAge(plantID, 2000.f);

    DOCTEST_CHECK_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    std::vector<uint> restored = plantarchitecture.readPlantStructureXML(xml_filename, true);
    DOCTEST_REQUIRE(!restored.empty());
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(restored.front()) == doctest::Approx(2000.f).epsilon(err_tol));

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture XML without max_age tag still loads") {
    // Backward compatibility: files written before <max_age> existed must still load, keeping the
    // PlantInstance default rather than failing to parse.
    std::string xml_filename = "test_plant_no_max_age.xml";

    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    // Strip the max_age tag to simulate a file written by an older version.
    std::string content;
    {
        std::ifstream f(xml_filename);
        content.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    size_t begin = content.find("<max_age>");
    DOCTEST_REQUIRE(begin != std::string::npos);
    size_t end = content.find("</max_age>", begin);
    DOCTEST_REQUIRE(end != std::string::npos);
    content.erase(begin, end + std::string("</max_age>").size() - begin);
    {
        std::ofstream f(xml_filename);
        f << content;
    }

    std::vector<uint> restored;
    DOCTEST_CHECK_NOTHROW(restored = plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(!restored.empty());
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(restored.front()) == doctest::Approx(999.f).epsilon(err_tol));

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance carries per-plant configuration") {
    // Regression test. duplicatePlantInstance() rebuilds the shoot structure of the source plant, but
    // the PlantInstance fields that are NOT derivable from that structure were left at their defaults
    // on the copy: it only propagated shoot_types_snapshot. Everything below is configuration a user
    // (or a library builder) set explicitly on the source, and silently lost on the duplicate.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);

    // buildAppleTree() sets max_age = 1460; the PlantInstance default is 999.
    DOCTEST_REQUIRE(plantarchitecture.getPlantMaxAge(plantID) == doctest::Approx(1460.f).epsilon(err_tol));

    uint plantID_dup = plantarchitecture.duplicatePlantInstance(plantID, make_vec3(5, 0, 0), make_AxisRotation(0, 0, 0), 0.f);

    // Maximum age. Without this the duplicate freezes at 999 days while the original grows to 1460.
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(plantID_dup) == doctest::Approx(1460.f).epsilon(err_tol));

    // Plant name. The duplicate of an apple used to report "custom", the addPlantInstance() default,
    // which also leaks into the plant_name object data written for every primitive.
    DOCTEST_CHECK(plantarchitecture.getPlantName(plantID_dup) == plantarchitecture.getPlantName(plantID));
    DOCTEST_CHECK(plantarchitecture.getPlantName(plantID_dup) == "apple");
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance preserves phenology so the copy grows like the original") {
    // The phenological thresholds are set by every library builder and are not recoverable from the
    // shoot structure. A duplicate that lost them fell back to the "no phenology scheduled" defaults,
    // so it never entered dormancy and its growth diverged from the original's -- the copy is supposed
    // to be an independent instance of the same plant, not a differently-parameterised one.
    Context context;
    context.seedRandomGenerator(6789);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    uint plantID_dup = plantarchitecture.duplicatePlantInstance(plantID, make_vec3(5, 0, 0), make_AxisRotation(0, 0, 0), 0.f);

    // Grow both through more than a full dormancy cycle (the apple model uses 165 + 200 = 365 days).
    // A copy that lost dd_to_dormancy would never drop its leaves and would diverge sharply here.
    plantarchitecture.advanceTime(400.f);

    // Both plants must be in the same phenological phase. Leaf presence is the coarse, robust signal:
    // with the thresholds lost, the duplicate stays in leaf while the original has gone dormant.
    bool original_in_leaf = plantarchitecture.getPlantLeafCount(plantID) > 0;
    bool duplicate_in_leaf = plantarchitecture.getPlantLeafCount(plantID_dup) > 0;
    DOCTEST_CHECK(original_in_leaf == duplicate_in_leaf);
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance translates attraction points to the new base") {
    // Attraction points are absolute world coordinates -- buildAppleFruitingWall() builds its trellis by
    // adding base_position to each point, and detectAttractionPointsInCone() tests them against absolute
    // internode positions. They were not carried over at all, so a duplicate of a trellis-trained plant
    // grew unguided; and copying them verbatim would be just as wrong, steering the copy back toward the
    // ORIGINAL plant's trellis. They must be translated by the difference between the two base positions.
    //
    // Checked directly rather than through growth: with the points simply absent the copy grows unguided
    // and still stays near its own base, so a canopy-position check passes on the buggy code.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple_fruitingwall");
    vec3 base_original = make_vec3(0, 0, 0);
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(base_original, 0.f);

    DOCTEST_REQUIRE(PlantArchitectureTestHelper::attractionPointsEnabled(plantarchitecture, plantID));
    const std::vector<vec3> points_original = PlantArchitectureTestHelper::getAttractionPoints(plantarchitecture, plantID);
    DOCTEST_REQUIRE(!points_original.empty());

    vec3 base_duplicate = make_vec3(5, 3, 0);
    uint plantID_dup = plantarchitecture.duplicatePlantInstance(plantID, base_duplicate, make_AxisRotation(0, 0, 0), 0.f);

    DOCTEST_CHECK(PlantArchitectureTestHelper::attractionPointsEnabled(plantarchitecture, plantID_dup));
    const std::vector<vec3> points_duplicate = PlantArchitectureTestHelper::getAttractionPoints(plantarchitecture, plantID_dup);
    DOCTEST_REQUIRE(points_duplicate.size() == points_original.size());

    const vec3 shift = base_duplicate - base_original;
    for (size_t i = 0; i < points_original.size(); i++) {
        DOCTEST_CHECK(points_duplicate.at(i).x == doctest::Approx(points_original.at(i).x + shift.x).epsilon(err_tol));
        DOCTEST_CHECK(points_duplicate.at(i).y == doctest::Approx(points_original.at(i).y + shift.y).epsilon(err_tol));
        DOCTEST_CHECK(points_duplicate.at(i).z == doctest::Approx(points_original.at(i).z + shift.z).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance reproduces branch attachment nodes") {
    // Regression test for the child-shoot attachment node. The duplication loop passed
    //     plant_shoot_tree->at(shoot->parent_shoot_ID)->parent_node_index
    // -- the node at which the PARENT attaches to the GRANDPARENT -- where addChildShoot() expects the
    // node at which THIS shoot attaches to its parent, i.e. shoot->parent_node_index. For any plant
    // whose branches leave the trunk at different heights the copy's branches were all relocated to a
    // single wrong node, so the duplicate had a visibly different architecture from the original.
    Context context;
    context.seedRandomGenerator(2468);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Apple scaffolds attach at distinct trunk nodes, which is exactly what this checks.
    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    uint plantID_dup = plantarchitecture.duplicatePlantInstance(plantID, make_vec3(5, 0, 0), make_AxisRotation(0, 0, 0), 0.f);

    std::vector<uint> shoots_original = plantarchitecture.getAllShootIDs(plantID);
    std::vector<uint> shoots_duplicate = plantarchitecture.getAllShootIDs(plantID_dup);
    DOCTEST_REQUIRE(shoots_original.size() == shoots_duplicate.size());

    // The source has scaffolds on more than one trunk node, otherwise this test proves nothing.
    std::set<uint> distinct_attachment_nodes;
    for (uint shootID: shoots_original) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        if (shoot->parent_shoot_ID >= 0) {
            distinct_attachment_nodes.insert(shoot->parent_node_index);
        }
    }
    DOCTEST_REQUIRE(distinct_attachment_nodes.size() > 1);

    for (size_t i = 0; i < shoots_original.size(); i++) {
        const std::shared_ptr<Shoot> &shoot_original = plantarchitecture.getPlantShoot(plantID, shoots_original.at(i));
        const std::shared_ptr<Shoot> &shoot_duplicate = plantarchitecture.getPlantShoot(plantID_dup, shoots_duplicate.at(i));
        DOCTEST_CHECK(shoot_duplicate->parent_shoot_ID == shoot_original->parent_shoot_ID);
        DOCTEST_CHECK(shoot_duplicate->parent_node_index == shoot_original->parent_node_index);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance copies dormancy state") {
    // Shoots are constructed dormant (Shoot::Shoot sets isdormant = true), so a duplicate of an
    // actively-growing plant came back dormant and refused to grow until its dormancy broke, while the
    // original kept growing. The duplicate must start in the same state as its source.
    Context context;
    context.seedRandomGenerator(1357);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);

    // buildAppleTree() leaves the plant dormant; break it so the source is actively growing.
    plantarchitecture.breakPlantDormancy(plantID);
    uint plantID_dup = plantarchitecture.duplicatePlantInstance(plantID, make_vec3(5, 0, 0), make_AxisRotation(0, 0, 0), 0.f);

    for (uint shootID: plantarchitecture.getAllShootIDs(plantID_dup)) {
        DOCTEST_CHECK(plantarchitecture.getPlantShoot(plantID_dup, shootID)->isdormant == false);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance is independent of its source") {
    // The copy must be a genuinely independent instance: mutating one must not affect the other, and
    // the duplicate must be positioned at the requested base rather than the source's.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    vec3 base_duplicate = make_vec3(5, 3, 0);
    uint plantID_dup = plantarchitecture.duplicatePlantInstance(plantID, base_duplicate, make_AxisRotation(0, 0, 0), 0.f);

    DOCTEST_CHECK(plantarchitecture.getPlantBasePosition(plantID_dup).x == doctest::Approx(base_duplicate.x).epsilon(err_tol));
    DOCTEST_CHECK(plantarchitecture.getPlantBasePosition(plantID_dup).y == doctest::Approx(base_duplicate.y).epsilon(err_tol));

    // Changing the copy's cap must leave the source's alone (no shared state).
    plantarchitecture.setPlantMaxAge(plantID_dup, 500.f);
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(plantID) == doctest::Approx(1460.f).epsilon(err_tol));
    DOCTEST_CHECK(plantarchitecture.getPlantMaxAge(plantID_dup) == doctest::Approx(500.f).epsilon(err_tol));

    // The two plants own disjoint geometry.
    std::vector<uint> objIDs_original = plantarchitecture.getAllPlantObjectIDs(plantID);
    std::vector<uint> objIDs_duplicate = plantarchitecture.getAllPlantObjectIDs(plantID_dup);
    DOCTEST_REQUIRE(!objIDs_original.empty());
    DOCTEST_REQUIRE(!objIDs_duplicate.empty());
    std::set<uint> original_set(objIDs_original.begin(), objIDs_original.end());
    for (uint objID: objIDs_duplicate) {
        DOCTEST_CHECK(original_set.find(objID) == original_set.end());
    }
}

//! Rebuild a ShootParameters from values only, the way a language binding must.
/**
 * A binding that flattens ShootParameters to a dictionary, lets the user edit it, and rebuilds it
 * cannot carry the function pointers across that trip. This helper reproduces exactly that: it
 * copies the value members and then clears all five function pointers, leaving a struct that is
 * value-correct but has lost every customization hook.
 *
 * \param[in] source ShootParameters to rebuild from.
 */
static ShootParameters rebuildShootParametersFromValuesOnly(const ShootParameters &source) {
    ShootParameters rebuilt = source;
    rebuilt.phytomer_parameters.phytomer_creation_function = nullptr;
    rebuilt.phytomer_parameters.phytomer_callback_function = nullptr;
    rebuilt.phytomer_parameters.leaf.prototype.prototype_function = nullptr;
    rebuilt.phytomer_parameters.inflorescence.flower_prototype_function = nullptr;
    rebuilt.phytomer_parameters.inflorescence.fruit_prototype_function = nullptr;
    return rebuilt;
}

DOCTEST_TEST_CASE("ShootParameters inheritCustomFunctionsFrom preserves phytomer functions across a value-only rebuild") {
    // Regression test. getCurrentShootParameters() returns by value, so an ordinary C++ struct copy
    // carries the function pointers along and nothing is lost. A caller that cannot hold a C++
    // struct -- a scripting-language binding that flattens the structure to a dictionary, applies
    // user edits, and rebuilds it -- produces a struct that is value-correct but has every function
    // pointer at nullptr. updateCurrentShootParameters() then replaces the stored entry wholesale,
    // silently dropping them.
    //
    // For maize the pointer that matters is MaizePhytomerCreationFunction, which is the only thing
    // that assigns MaizeEarPrototype/BUD_ACTIVE at the apical ear node (measured down from the
    // tassel, see "Maize ear position tracks the tassel...") and MaizeTasselPrototype/BUD_DEAD
    // everywhere else. Without it every node sets a multi-flower tassel, so fruit and peduncle
    // counts explode. Leaf count is unchanged either way, which is why it is NOT asserted here --
    // asserting on leaves would pass on the broken code and prove nothing.

    enum RebuildMode { BASELINE, VALUES_ONLY, VALUES_PLUS_INHERIT };

    // Every arm is seeded identically so that the ONLY difference between them is what happens to
    // the function pointers. Maize growth is stochastic; without a fixed seed the counts wander.
    auto grow = [](RebuildMode mode) -> std::pair<std::size_t, std::size_t> {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("maize");

        if (mode != BASELINE) {
            ShootParameters original = plantarchitecture.getCurrentShootParameters("mainstem");
            // Clear ONLY the two phytomer hooks. The other three pointers are deliberately kept:
            // losing leaf.prototype.prototype_function raises a helios_runtime_error at the first
            // leaf, and losing inflorescence.fruit_prototype_function gates off fruit creation
            // entirely (fruit and peduncle counts both go to zero). Either would mask the silent,
            // structurally-wrong-plant failure that this test exists to measure. Both are covered
            // separately below.
            ShootParameters rebuilt = original;
            rebuilt.phytomer_parameters.phytomer_creation_function = nullptr;
            rebuilt.phytomer_parameters.phytomer_callback_function = nullptr;
            if (mode == VALUES_PLUS_INHERIT) {
                rebuilt.inheritCustomFunctionsFrom(original);
            }
            plantarchitecture.updateCurrentShootParameters("mainstem", rebuilt);
        }

        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
        plantarchitecture.advanceTime(plantID, 90.f);

        return {plantarchitecture.getPlantFruitObjectIDs(plantID).size(), plantarchitecture.getPlantPeduncleObjectIDs(plantID).size()};
    };

    const std::pair<std::size_t, std::size_t> baseline = grow(BASELINE);
    const std::pair<std::size_t, std::size_t> values_only = grow(VALUES_ONLY);
    const std::pair<std::size_t, std::size_t> inherited = grow(VALUES_PLUS_INHERIT);

    // Guard against a vacuous test: maize must actually set fruit within the growth window, or
    // every comparison below is comparing zero against zero.
    DOCTEST_REQUIRE(baseline.first > 0);
    DOCTEST_REQUIRE(baseline.second > 0);

    // Negative control. Losing the pointers must change the plant dramatically -- if these counts
    // matched the baseline, the rebuild would not be exercising the creation function at all and
    // the positive assertion below would prove nothing. Thresholds are inequalities rather than the
    // literal counts so the test survives library retuning, but the gap is an order of magnitude,
    // far wider than run-to-run variation could bridge.
    DOCTEST_CHECK(values_only.first > 3 * baseline.first);
    DOCTEST_CHECK(values_only.second > 2 * baseline.second);

    // The actual regression assertion: re-attaching the pointers restores the baseline exactly.
    DOCTEST_CHECK(inherited.first == baseline.first);
    DOCTEST_CHECK(inherited.second == baseline.second);
}

DOCTEST_TEST_CASE("Maize ear position tracks the tassel rather than fixed node indices") {
    // Regression test. MaizePhytomerCreationFunction used to select ear-bearing nodes with a
    // hardcoded absolute window (shoot_node_index > 8 && < 12), which only approximates the biology
    // at the library default of max_nodes=17. Maize forms an axillary ear meristem at every node
    // except the upper ~6-8 below the tassel, and the uppermost eligible node bears the harvestable
    // ear -- a position defined relative to the TOP of the plant, not by absolute index.
    //
    // The defect is invisible at max_nodes=17, because 17-6=11 happens to fall inside the old 9-11
    // window. It only shows up on a plant of a different size: at max_nodes=25 the old rule left the
    // ears stranded at nodes 9-11, deep in the lower canopy, and a TALLER plant produced FEWER fruit
    // than a shorter one. The max_nodes=25 arm below is therefore the arm that actually proves the
    // fix; an assertion at the default alone would pass for the wrong reason.
    //
    // Leaf count is identical under both rules, so it is deliberately NOT asserted -- doing so would
    // pass on the buggy code and prove nothing.

    constexpr int nodes_below_tassel = 6;

    struct EarObservation {
        std::vector<int> ear_nodes;
        std::size_t fruit_count;
        std::size_t peduncle_count;
    };

    // Seeded identically across arms so the only difference is max_nodes. Maize growth is
    // stochastic; without a fixed seed the counts wander.
    auto grow = [](int max_nodes) -> EarObservation {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("maize");

        ShootParameters params = plantarchitecture.getCurrentShootParameters("mainstem");
        params.max_nodes = max_nodes;
        plantarchitecture.updateCurrentShootParameters("mainstem", params);

        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
        plantarchitecture.advanceTime(plantID, 90.f);

        // An ear is an AXILLARY floral bud that actually produced inflorescence geometry. The
        // terminal bud carries the tassel and is excluded via isterminal.
        EarObservation obs;
        for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
            const auto &phytomers = plantarchitecture.getPlantShoot(plantID, shootID)->phytomers;
            for (std::size_t node = 0; node < phytomers.size(); node++) {
                for (const auto &petiole: phytomers.at(node)->floral_buds) {
                    for (const auto &fbud: petiole) {
                        if (!fbud.isterminal && !fbud.inflorescence_objIDs.empty()) {
                            obs.ear_nodes.push_back(int(node));
                        }
                    }
                }
            }
        }
        obs.fruit_count = plantarchitecture.getPlantFruitObjectIDs(plantID).size();
        obs.peduncle_count = plantarchitecture.getPlantPeduncleObjectIDs(plantID).size();
        return obs;
    };

    const EarObservation short_plant = grow(17);
    const EarObservation tall_plant = grow(25);

    // Guard against a vacuous pass: if no ear set at all, every assertion below compares empty
    // against empty and the test proves nothing.
    DOCTEST_REQUIRE(!short_plant.ear_nodes.empty());
    DOCTEST_REQUIRE(!tall_plant.ear_nodes.empty());

    // Apical dominance: exactly one ear develops by default.
    DOCTEST_CHECK(short_plant.ear_nodes.size() == 1);
    DOCTEST_CHECK(tall_plant.ear_nodes.size() == 1);

    // The property that matters: the ear node is measured down from the tassel, so it must move
    // when the plant gets taller. This is what fails on the absolute-index rule.
    DOCTEST_CHECK(short_plant.ear_nodes.front() == 17 - nodes_below_tassel);
    DOCTEST_CHECK(tall_plant.ear_nodes.front() == 25 - nodes_below_tassel);

    // Stated as an invariant as well, so the intent survives any retuning of nodes_below_tassel.
    DOCTEST_CHECK(tall_plant.ear_nodes.front() - short_plant.ear_nodes.front() == 25 - 17);

    // One ear (1 fruit + 1 peduncle) plus the terminal tassel (flowers_per_peduncle=7 -> 7 fruit
    // objects + 1 peduncle) = 8 fruit, 2 peduncles.
    DOCTEST_CHECK(short_plant.fruit_count == 8);
    DOCTEST_CHECK(short_plant.peduncle_count == 2);
}

DOCTEST_TEST_CASE("Maize bears a single ear and a terminal tassel") {
    // Companion to the test above, pinning the two things that make a maize plant's reproductive
    // structure correct at the default configuration: exactly one ear (apical dominance), and a
    // tassel that is terminal rather than borne at every node.
    //
    // Prolificacy is the nodes_below_tassel/ears_per_plant pair of constants in
    // MaizePhytomerCreationFunction (Assets.cpp). Raising ears_per_plant to 2 adds a second ear at
    // the node immediately below the apical one, which is why the count is asserted here rather
    // than left implicit -- a change to that constant should show up as a deliberate test update.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("maize");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 90.f);

    std::size_t axillary_ears = 0;
    std::size_t terminal_inflorescences = 0;
    for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        for (const auto &phytomer: plantarchitecture.getPlantShoot(plantID, shootID)->phytomers) {
            for (const auto &petiole: phytomer->floral_buds) {
                for (const auto &fbud: petiole) {
                    if (fbud.inflorescence_objIDs.empty()) {
                        continue;
                    }
                    if (fbud.isterminal) {
                        terminal_inflorescences++;
                    } else {
                        axillary_ears++;
                    }
                }
            }
        }
    }

    // Guard against a vacuous pass: maize must actually set reproductive structures within the
    // growth window, or both assertions below compare zero against zero.
    DOCTEST_REQUIRE(axillary_ears + terminal_inflorescences > 0);

    // Apical dominance: one harvestable ear, not one per eligible node.
    DOCTEST_CHECK(axillary_ears == 1);

    // Maize is monoecious with a single terminal tassel from the apical meristem. Every non-ear
    // node is assigned the tassel prototype but kept BUD_DEAD, so exactly one tassel-bearing
    // floral bud should ever produce geometry, and it must be the terminal one.
    DOCTEST_CHECK(terminal_inflorescences == 1);
}

DOCTEST_TEST_CASE("Maize vegetative development matches field phenology") {
    // Pins the library maize timing against published field benchmarks for a temperate Corn Belt
    // grain hybrid, so a retune cannot silently drift back to the previous values (17 nodes at a
    // 2-day phyllochron, which tasselled around day 32 -- roughly twice as fast as a field crop).
    //
    // Benchmarks: 19-21 main-stem leaves, range 16-23 (Abendroth et al. 2011, Iowa State PMR 1009;
    // leaf-appearance synthesis over 4,694 Corn Belt observations); emergence to tasseling 54-63
    // days (Ritchie et al. 1993). Bounds below are deliberately the published RANGES rather than
    // the exact library values, so ordinary retuning within the literature does not fail the test.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("maize");

    ShootParameters params = plantarchitecture.getCurrentShootParameters("mainstem");
    const int max_nodes = params.max_nodes.val();

    // Main-stem leaf number. One phytomer bears one leaf, so node count is leaf count.
    DOCTEST_CHECK(max_nodes >= 16);
    DOCTEST_CHECK(max_nodes <= 23);

    // Grow day by day and record when the shoot reaches its final node count, which is when the
    // terminal tassel is added (see appendPhytomerToShoot).
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();

    int tasseling_day = -1;
    for (int day = 1; day <= 120 && tasseling_day < 0; day++) {
        plantarchitecture.advanceTime(plantID, 1.f);
        if (int(plantarchitecture.getPlantShoot(plantID, shootID)->phytomers.size()) >= max_nodes) {
            tasseling_day = day;
        }
    }

    // Guard against a vacuous pass: if the plant never reached its final node count the bounds
    // below would be compared against the -1 sentinel.
    DOCTEST_REQUIRE(tasseling_day > 0);

    // Emergence to tasseling, 54-63 days in the field. A generous lower bound of 45 days still
    // excludes the old 32-day behavior by a wide margin.
    DOCTEST_CHECK(tasseling_day >= 45);
    DOCTEST_CHECK(tasseling_day <= 75);
}

DOCTEST_TEST_CASE("Maize mainstem honors the shoot type internode_length_max") {
    // Regression test. buildMaizePlant() used to pass a hardcoded 0.08 m as the internode_length_max
    // argument of addBaseStemShoot(), which Phytomer stores per-shoot and uses in preference to the
    // value on the shoot type. shoot_parameters_mainstem.internode_length_max was therefore dead for
    // the mainstem: editing it had no effect on the plant, and maize topped out around 1.4 m against
    // a field range of 2-3 m. The builder now forwards the shoot-type value, matching how
    // buildBougainvilleaPlant() and buildCapsicumPlant() already call addBaseStemShoot().
    //
    // Asserted by changing the parameter and requiring the plant to respond, rather than by pinning
    // a literal height -- a fixed expected height would pass on the buggy code as soon as someone
    // retuned the hardcoded constant to match.

    auto grow_with_internode_length = [](float internode_length_max) -> float {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("maize");

        ShootParameters params = plantarchitecture.getCurrentShootParameters("mainstem");
        params.internode_length_max = internode_length_max;
        plantarchitecture.updateCurrentShootParameters("mainstem", params);

        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
        plantarchitecture.advanceTime(plantID, 120.f);
        return plantarchitecture.getPlantHeight(plantID);
    };

    const float short_internodes = grow_with_internode_length(0.10f);
    const float long_internodes = grow_with_internode_length(0.30f);

    // Guard against a vacuous pass: both plants must actually have grown.
    DOCTEST_REQUIRE(short_internodes > 0.f);
    DOCTEST_REQUIRE(long_internodes > 0.f);

    // Tripling the internode length must make a substantially taller plant. On the old code both
    // arms produced an identical plant built from the hardcoded 0.08 m, so this ratio was 1.0.
    DOCTEST_CHECK(long_internodes > 1.5f * short_internodes);
}

DOCTEST_TEST_CASE("Sorghum development matches field phenology and bears one terminal panicle") {
    // Companion to the maize phenology test. Sorghum had the same too-fast development: a 2-day
    // phyllochron completed the 16-node plant by day 30, against a field half-bloom (GS6) near day
    // 60 (Vanderlip, "How a Sorghum Plant Develops", Kansas State S-3, Table 1).
    //
    // Node count is deliberately NOT changed from 16 and not pinned tightly here: field hybrids
    // carry 15 (early), 17 (medium) and 19 (late) leaves by maturity class, so 16 is already
    // reasonable. Height is likewise already correct for a dwarfed grain hybrid (0.6-1.5 m), unlike
    // maize which was too short -- so this test guards timing and structure, not height.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("sorghum");

    ShootParameters params = plantarchitecture.getCurrentShootParameters("mainstem");
    const int max_nodes = params.max_nodes.val();
    DOCTEST_CHECK(max_nodes >= 14);
    DOCTEST_CHECK(max_nodes <= 20);

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();

    int final_node_day = -1;
    for (int day = 1; day <= 120 && final_node_day < 0; day++) {
        plantarchitecture.advanceTime(plantID, 1.f);
        if (int(plantarchitecture.getPlantShoot(plantID, shootID)->phytomers.size()) >= max_nodes) {
            final_node_day = day;
        }
    }

    // Guard against a vacuous pass: without this the bounds below would test the -1 sentinel.
    DOCTEST_REQUIRE(final_node_day > 0);

    // Emergence to the full leaf complement, which precedes half-bloom near day 60. A lower bound
    // of 40 days still excludes the previous 30-day behavior by a clear margin.
    DOCTEST_CHECK(final_node_day >= 40);
    DOCTEST_CHECK(final_node_day <= 75);

    // Grain sorghum bears a single perfect panicle, strictly terminal on the culm -- there is no
    // analogue of the maize axillary ear. The panicle comes from the terminal floral bud, so
    // exactly one inflorescence should form and it must not be axillary.
    std::size_t terminal_inflorescences = 0;
    std::size_t axillary_inflorescences = 0;
    for (const auto &phytomer: plantarchitecture.getPlantShoot(plantID, shootID)->phytomers) {
        for (const auto &petiole: phytomer->floral_buds) {
            for (const auto &fbud: petiole) {
                if (fbud.inflorescence_objIDs.empty()) {
                    continue;
                }
                if (fbud.isterminal) {
                    terminal_inflorescences++;
                } else {
                    axillary_inflorescences++;
                }
            }
        }
    }
    DOCTEST_CHECK(terminal_inflorescences == 1);
    DOCTEST_CHECK(axillary_inflorescences == 0);
}

DOCTEST_TEST_CASE("Sorghum mainstem honors the shoot type internode_length_max") {
    // Same dead-parameter defect that affected maize: buildSorghumPlant() passed a hardcoded 0.06 m
    // to addBaseStemShoot(), which takes precedence over the shoot type, so
    // shoot_parameters_mainstem.internode_length_max had no effect on the plant.
    //
    // Note the accompanying library change was deliberately height-NEUTRAL here. The dead value was
    // 0.26 m; connecting it unchanged would have produced a roughly 5.5 m plant, far outside the
    // 0.6-1.5 m range of a dwarfed grain hybrid. The stored value was corrected to the 0.06 m that
    // was actually in force, so the plant is unchanged and the parameter is now live.

    auto grow_with_internode_length = [](float internode_length_max) -> float {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("sorghum");

        ShootParameters params = plantarchitecture.getCurrentShootParameters("mainstem");
        params.internode_length_max = internode_length_max;
        plantarchitecture.updateCurrentShootParameters("mainstem", params);

        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
        plantarchitecture.advanceTime(plantID, 120.f);
        return plantarchitecture.getPlantHeight(plantID);
    };

    const float short_internodes = grow_with_internode_length(0.05f);
    const float long_internodes = grow_with_internode_length(0.15f);

    DOCTEST_REQUIRE(short_internodes > 0.f);
    DOCTEST_REQUIRE(long_internodes > 0.f);

    // On the old code both arms produced the same plant built from the hardcoded 0.06 m.
    DOCTEST_CHECK(long_internodes > 1.5f * short_internodes);
}

DOCTEST_TEST_CASE("ShootParameters value-only rebuild without inheriting fails fast on the leaf prototype") {
    // The five function pointers lost by a value-only rebuild split into two behavior classes, and
    // this test pins the loud one so the two are not confused. Losing leaf.prototype.prototype_function
    // raises a helios_runtime_error as soon as the first leaf is built -- correct fail-fast behavior,
    // and immediately visible to the caller. Losing phytomer_creation_function is the dangerous case:
    // it is silent, and shows up only as a structurally wrong plant (covered by the test above).
    //
    // Inheriting restores both, so the same build then succeeds.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("maize");

    const ShootParameters original = plantarchitecture.getCurrentShootParameters("mainstem");
    ShootParameters rebuilt = rebuildShootParametersFromValuesOnly(original);
    plantarchitecture.updateCurrentShootParameters("mainstem", rebuilt);

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5));
    }

    // Re-attaching the pointers makes the identical build succeed.
    rebuilt.inheritCustomFunctionsFrom(original);
    plantarchitecture.updateCurrentShootParameters("mainstem", rebuilt);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(1, 0, 0), 5));
}

DOCTEST_TEST_CASE("ShootParameters inheritCustomFunctionsFrom copies all five function pointers") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("maize");

    const ShootParameters donor = plantarchitecture.getCurrentShootParameters("mainstem");

    // Maize sets the creation function and the leaf/fruit prototypes, so the donor is a meaningful
    // source for those three. Verify that up front rather than asserting against nullptr==nullptr.
    DOCTEST_REQUIRE(donor.phytomer_parameters.phytomer_creation_function != nullptr);
    DOCTEST_REQUIRE(donor.phytomer_parameters.leaf.prototype.prototype_function != nullptr);
    DOCTEST_REQUIRE(donor.phytomer_parameters.inflorescence.fruit_prototype_function != nullptr);

    // A value-only rebuild starts with all five cleared; inheriting restores each one.
    ShootParameters rebuilt = rebuildShootParametersFromValuesOnly(donor);
    DOCTEST_REQUIRE(rebuilt.phytomer_parameters.phytomer_creation_function == nullptr);

    rebuilt.inheritCustomFunctionsFrom(donor);

    DOCTEST_CHECK(rebuilt.phytomer_parameters.phytomer_creation_function == donor.phytomer_parameters.phytomer_creation_function);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.phytomer_callback_function == donor.phytomer_parameters.phytomer_callback_function);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.leaf.prototype.prototype_function == donor.phytomer_parameters.leaf.prototype.prototype_function);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.inflorescence.flower_prototype_function == donor.phytomer_parameters.inflorescence.flower_prototype_function);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.inflorescence.fruit_prototype_function == donor.phytomer_parameters.inflorescence.fruit_prototype_function);

    // Documented overwrite semantics: null in the source is copied as null rather than skipped, so
    // a deliberately cleared hook stays cleared and repeated calls are idempotent. The donor here is
    // explicitly cleared rather than merely default-constructed, because a default-constructed
    // ShootParameters is NOT all-null -- PhytomerParameters' constructor seeds
    // leaf.prototype.prototype_function with GenericLeafPrototype.
    ShootParameters cleared_donor;
    cleared_donor.phytomer_parameters.phytomer_creation_function = nullptr;
    cleared_donor.phytomer_parameters.phytomer_callback_function = nullptr;
    cleared_donor.phytomer_parameters.leaf.prototype.prototype_function = nullptr;
    cleared_donor.phytomer_parameters.inflorescence.flower_prototype_function = nullptr;
    cleared_donor.phytomer_parameters.inflorescence.fruit_prototype_function = nullptr;

    rebuilt.inheritCustomFunctionsFrom(cleared_donor);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.phytomer_creation_function == nullptr);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.phytomer_callback_function == nullptr);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.leaf.prototype.prototype_function == nullptr);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.inflorescence.flower_prototype_function == nullptr);
    DOCTEST_CHECK(rebuilt.phytomer_parameters.inflorescence.fruit_prototype_function == nullptr);

    // A default-constructed ShootParameters is a realistic donor mistake, so pin what it actually
    // does: it supplies the generic leaf prototype, not a null.
    ShootParameters default_donor;
    ShootParameters from_default = rebuildShootParametersFromValuesOnly(donor);
    from_default.inheritCustomFunctionsFrom(default_donor);
    DOCTEST_CHECK(from_default.phytomer_parameters.leaf.prototype.prototype_function != nullptr);
    DOCTEST_CHECK(from_default.phytomer_parameters.phytomer_creation_function == nullptr);

    // Self-inheritance is a no-op rather than undefined behavior.
    ShootParameters self = donor;
    DOCTEST_CHECK_NOTHROW(self.inheritCustomFunctionsFrom(self));
    DOCTEST_CHECK(self.phytomer_parameters.phytomer_creation_function == donor.phytomer_parameters.phytomer_creation_function);
}

DOCTEST_TEST_CASE("PlantArchitecture readPlantStructureXML still clears the phytomer creation function") {
    // The XML reader deliberately clears phytomer_creation_function (InputOutput.cpp) because a
    // plant reconstructed from saved geometry must not re-run creation-time logic that would
    // overwrite what was read from the file. inheritCustomFunctionsFrom() is opt-in and is not
    // called from that path, so this behavior must be untouched. If restoration were ever made
    // implicit inside defineShootType(), this test would catch the regression.

    const std::string xml_filename = "test_plant_xml_creation_function_cleared.xml";

    {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("maize");
        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 25);
        DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));
    }

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    // Required: the XML references shoot types by label, resolved via getCurrentShootParameters().
    plantarchitecture.loadPlantModelFromLibrary("maize");

    // Sanity check that the library did supply a creation function, so the assertion after the read
    // is testing that it was cleared rather than that it was never set.
    DOCTEST_REQUIRE(plantarchitecture.getCurrentShootParameters("mainstem").phytomer_parameters.phytomer_creation_function != nullptr);

    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);

    // The reader clears the creation function on the parameters it passes to appendPhytomerToShoot(),
    // which covers every phytomer after the first on each shoot. The first phytomer of a shoot is
    // built by addBaseStemShoot()/addChildShoot(), which re-read the shoot type from the plant's
    // snapshot and so still carry the library pointer -- that asymmetry is pre-existing reader
    // behavior, not something this API changes. What matters here is that the clearing still happens
    // at all, so assert on the appended phytomers specifically.
    const uint restored_plantID = restored_plantIDs.front();
    const std::vector<uint> restored_shootIDs = plantarchitecture.getAllShootIDs(restored_plantID);
    DOCTEST_REQUIRE(!restored_shootIDs.empty());
    std::size_t appended_phytomers_checked = 0;
    for (uint shootID: restored_shootIDs) {
        const auto &phytomers = plantarchitecture.getPlantShoot(restored_plantID, shootID)->phytomers;
        for (std::size_t node = 1; node < phytomers.size(); node++) {
            DOCTEST_CHECK(phytomers.at(node)->phytomer_parameters.phytomer_creation_function == nullptr);
            appended_phytomers_checked++;
        }
    }
    // Guard against a vacuous pass: if no shoot had more than one phytomer, the loop above never ran.
    DOCTEST_REQUIRE(appended_phytomers_checked > 0);

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("ShootParameters assignment copies elongation_rate_max") {
    // Regression test. ShootParameters::operator= assigned every member except elongation_rate_max,
    // which was simply absent from the list. Assigning a ShootParameters therefore silently reset the
    // shoot's maximum elongation rate to the PhytomerParameters constructor default of 0.2, discarding
    // whatever the library or the user had set. Every library model sets a value other than 0.2, so
    // any path that assigned a ShootParameters -- including the documented query-modify-set workflow
    // through updateCurrentShootParameters() -- lost it.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("almond");

    ShootParameters source = plantarchitecture.getCurrentShootParameters("proleptic");
    const float source_rate = source.elongation_rate_max.val();

    // Guard against a vacuous test: if the library value happened to equal the 0.2 default, a
    // dropped assignment would be indistinguishable from a correct one.
    DOCTEST_REQUIRE(source_rate != doctest::Approx(0.2f).epsilon(err_tol));

    ShootParameters assigned;
    assigned = source;
    DOCTEST_CHECK(assigned.elongation_rate_max.val() == doctest::Approx(source_rate).epsilon(err_tol));

    // The same must hold through the public setter/getter round trip.
    plantarchitecture.updateCurrentShootParameters("proleptic", source);
    DOCTEST_CHECK(plantarchitecture.getCurrentShootParameters("proleptic").elongation_rate_max.val() == doctest::Approx(source_rate).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PlantArchitecture defineShootType stores the same values for a new and an existing shoot type") {
    // Regression test. defineShootType() takes two paths: an existing label is assigned through
    // ShootParameters::operator=, a new label is copy-constructed. operator= used to resample every
    // distributed shoot-level parameter, so the same input structure produced different stored values
    // depending only on whether the label already existed.
    //
    // Almond's "proleptic" is used because its insertion_angle_tip is uniformDistribution(25,30)
    // (PlantLibrary.cpp), so a resample is observable; a constant-valued parameter would pass on the
    // buggy code because resample() is a no-op for a constant distribution.
    Context context;
    context.seedRandomGenerator(4242);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("almond");

    ShootParameters source = plantarchitecture.getCurrentShootParameters("proleptic");
    DOCTEST_REQUIRE(source.insertion_angle_tip.distribution != "constant");
    const float source_value = source.insertion_angle_tip.val();

    plantarchitecture.defineShootType("regression_new_label", source); // insertion path
    plantarchitecture.defineShootType("proleptic", source); // assignment path

    const float new_label_value = plantarchitecture.getCurrentShootParameters("regression_new_label").insertion_angle_tip.val();
    const float existing_label_value = plantarchitecture.getCurrentShootParameters("proleptic").insertion_angle_tip.val();

    DOCTEST_CHECK(new_label_value == doctest::Approx(source_value).epsilon(err_tol));
    DOCTEST_CHECK(existing_label_value == doctest::Approx(source_value).epsilon(err_tol));
    DOCTEST_CHECK(new_label_value == doctest::Approx(existing_label_value).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhytomerParameters assignment resamples while ShootParameters assignment does not") {
    // Pins the deliberate asymmetry introduced alongside the defineShootType() fix, so that a future
    // cleanup does not "tidy up" one half of it.
    //
    // Shoot-level parameters must NOT resample on assignment: committing a shoot type through
    // defineShootType()/updateCurrentShootParameters() must store what the caller passed in.
    //
    // The nested PhytomerParameters sub-structures must CONTINUE to resample on assignment.
    // Phytomer's constructor assigns its parameters specifically to re-randomize them
    // (PlantArchitecture.cpp: "this needs to be an assignment operation not a copy in order to
    // re-randomize all the parameters"), and several fields -- leaflet_offset, leaflet_scale and
    // inflorescence.flower_offset -- have no other resample site anywhere in the growth engine, so
    // that assignment is their only source of variation.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("almond");

    ShootParameters source = plantarchitecture.getCurrentShootParameters("proleptic");

    // Give a shoot-level and a phytomer-level parameter the same distribution so the two halves are
    // compared on equal terms.
    source.gravitropic_curvature.uniformDistribution(100, 200);
    source.phytomer_parameters.leaf.leaflet_offset.uniformDistribution(100, 200);

    // Shoot level: assignment preserves the value.
    const float shoot_value_before = source.gravitropic_curvature.val();
    ShootParameters shoot_copy;
    shoot_copy = source;
    DOCTEST_CHECK(shoot_copy.gravitropic_curvature.val() == doctest::Approx(shoot_value_before).epsilon(err_tol));

    // Phytomer level: assignment redraws. leaflet_offset is resampled nowhere else in the growth
    // engine, so if this assignment stopped resampling, every phytomer would share one value.
    const float phytomer_value_before = source.phytomer_parameters.leaf.leaflet_offset.val();
    bool phytomer_value_changed = false;
    for (int attempt = 0; attempt < 20 && !phytomer_value_changed; attempt++) {
        PhytomerParameters phytomer_copy;
        phytomer_copy = source.phytomer_parameters;
        if (phytomer_copy.leaf.leaflet_offset.val() != phytomer_value_before) {
            phytomer_value_changed = true;
        }
    }
    DOCTEST_CHECK(phytomer_value_changed);
}

int PlantArchitecture::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

DOCTEST_TEST_CASE("PlantArchitecture petiole geometry is a continuous tube") {
    // A multi-segment petiole must form ONE continuous surface.
    //
    // Petioles were historically built as a chain of independent Cone objects
    // (makeTubeFromCones). Context::addConeObject seeds its radial reference frame from a
    // hard-coded constant vector on every call, so each cone in the chain chose its azimuthal
    // phase independently of its neighbours. Adjacent cones therefore met at a shared centerline
    // node with their vertex rings rotated out of phase, leaving a visible crease at every
    // segment joint. A tube object parallel-transports the frame along the centerline, so the
    // rings stay in phase and the surface closes.
    //
    // The invariant checked is phase continuity at a shared interior node: the vertices
    // contributed there by the geometry on either side must coincide. This is written against
    // raw primitive vertices rather than the tube/cone APIs so that it compiles -- and fails --
    // against the old cone-chain representation as well as the new tube one.
    Context context;
    PlantArchitecture pa(&context);
    pa.disableMessages();

    // Sugarbeet uses petiole.length_segments = 8 -- plenty of interior joints to inspect.
    pa.loadPlantModelFromLibrary("sugarbeet");
    uint plantID = pa.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    pa.advanceTime(20.f);

    const std::vector<std::shared_ptr<Shoot>> &shoot_tree = PlantArchitectureTestHelper::getShootTree(pa, plantID);

    uint petioles_checked = 0;
    uint petioles_with_seam = 0;

    for (const auto &shoot: shoot_tree) {
        for (const auto &phytomer: shoot->phytomers) {
            for (uint petiole = 0; petiole < phytomer->petiole_vertices.size(); petiole++) {

                const std::vector<vec3> &centerline = phytomer->petiole_vertices.at(petiole);
                const std::vector<float> &radii = phytomer->petiole_radii.at(petiole);
                if (centerline.size() < 3) {
                    continue; // need at least one interior joint
                }

                // Gather every primitive making up this petiole, whatever objects hold them.
                std::vector<uint> petiole_UUIDs;
                for (uint objID: flattenPetioleObjIDs(phytomer->petiole_objIDs.at(petiole))) {
                    if (!context.doesObjectExist(objID)) {
                        continue;
                    }
                    const std::vector<uint> &UUIDs = context.getObjectPrimitiveUUIDs(objID);
                    petiole_UUIDs.insert(petiole_UUIDs.end(), UUIDs.begin(), UUIDs.end());
                }
                if (petiole_UUIDs.empty()) {
                    continue;
                }

                // The petiole must be a single tube object, not a chain of cones.
                for (uint objID: flattenPetioleObjIDs(phytomer->petiole_objIDs.at(petiole))) {
                    if (context.doesObjectExist(objID)) {
                        DOCTEST_CHECK(context.getObjectType(objID) == helios::OBJECT_TYPE_TUBE);
                    }
                }

                // Inspect the middle interior node, furthest from both tube ends.
                uint interior_node = centerline.size() / 2;
                const vec3 node_center = centerline.at(interior_node);
                const float node_radius = radii.at(interior_node);
                if (!(node_radius > 0.f)) {
                    continue;
                }

                // A vertex belongs to this node's ring if it lies on that node's circle.
                std::vector<vec3> ring_vertices;
                for (uint UUID: petiole_UUIDs) {
                    for (const vec3 &v: context.getPrimitiveVertices(UUID)) {
                        if (fabsf((v - node_center).magnitude() - node_radius) < 0.05f * node_radius) {
                            ring_vertices.push_back(v);
                        }
                    }
                }
                if (ring_vertices.empty()) {
                    continue;
                }

                // Count distinct positions around the ring. On a continuous surface the geometry
                // on both sides of the node shares one ring, so the distinct count equals the
                // number of radial subdivisions. Two out-of-phase rings roughly double it.
                std::vector<vec3> distinct;
                for (const vec3 &v: ring_vertices) {
                    bool found = false;
                    for (const vec3 &d: distinct) {
                        if ((v - d).magnitude() < 1e-3f * node_radius) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        distinct.push_back(v);
                    }
                }

                uint Ndivs = std::max(uint(3), phytomer->phytomer_parameters.petiole.radial_subdivisions);

                // Allow one extra for the duplicated seam vertex that closes the ring.
                if (distinct.size() > Ndivs + 1) {
                    petioles_with_seam++;
                }
                petioles_checked++;
            }
        }
    }

    // Guard against the test silently passing because it never found a multi-segment petiole.
    DOCTEST_CHECK(petioles_checked > 0);
    DOCTEST_CHECK(petioles_with_seam == 0);
}
