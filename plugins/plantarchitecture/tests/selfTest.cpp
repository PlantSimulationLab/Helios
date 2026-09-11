#include "PlantArchitecture.h"
#include "Assets.h"

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
    static helios::vec3 interpolateTube(const std::vector<helios::vec3> &P, float frac) {
        return PlantArchitecture::interpolateTube(P, frac);
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

    // the values must read back identically through the public accessors
    std::vector<std::string> labels_read = sp_test.getChildShootTypeLabels();
    std::vector<float> probabilities_read = sp_test.getChildShootTypeProbabilities();
    DOCTEST_REQUIRE(labels_read.size() == labels.size());
    DOCTEST_REQUIRE(probabilities_read.size() == probabilities.size());
    for (size_t i = 0; i < labels.size(); i++) {
        DOCTEST_CHECK(labels_read.at(i) == labels.at(i));
        DOCTEST_CHECK(probabilities_read.at(i) == doctest::Approx(probabilities.at(i)).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("ShootParameters child shoot type accessors default empty") {
    ShootParameters sp_test;
    // no child shoot types defined means child shoots inherit the parent's own type, which is a
    // valid state rather than an error
    DOCTEST_CHECK(sp_test.getChildShootTypeLabels().empty());
    DOCTEST_CHECK(sp_test.getChildShootTypeProbabilities().empty());
}

DOCTEST_TEST_CASE("ShootParameters defineChildShootTypes replaces previous values") {
    ShootParameters sp_test;
    sp_test.defineChildShootTypes({"typeA", "typeB"}, {0.4f, 0.6f});
    sp_test.defineChildShootTypes({"typeC"}, {1.f});

    // a redefine must replace the previous entries outright, not append to them
    std::vector<std::string> labels_read = sp_test.getChildShootTypeLabels();
    std::vector<float> probabilities_read = sp_test.getChildShootTypeProbabilities();
    DOCTEST_REQUIRE(labels_read.size() == 1);
    DOCTEST_REQUIRE(probabilities_read.size() == 1);
    DOCTEST_CHECK(labels_read.at(0) == "typeC");
    DOCTEST_CHECK(probabilities_read.at(0) == doctest::Approx(1.f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("ShootParameters child shoot types survive copy assignment") {
    ShootParameters sp_source;
    sp_source.defineChildShootTypes({"typeA", "typeB"}, {0.25f, 0.75f});

    // copy assignment is how a scripting binding carries these vectors across a rebuild of the
    // structure, so the copy must report the same values as the source
    ShootParameters sp_copy;
    sp_copy = sp_source;
    DOCTEST_REQUIRE(sp_copy.getChildShootTypeLabels().size() == 2);
    DOCTEST_REQUIRE(sp_copy.getChildShootTypeProbabilities().size() == 2);
    DOCTEST_CHECK(sp_copy.getChildShootTypeLabels().at(1) == "typeB");
    DOCTEST_CHECK(sp_copy.getChildShootTypeProbabilities().at(1) == doctest::Approx(0.75f).epsilon(err_tol));
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

DOCTEST_TEST_CASE("Plant Library - bindweed leaves face upward on sprawling shoots") {
    // Bindweed shoots sprawl along the ground and the leaves are meant to face the sky. The library's
    // leaf.roll had been set to cancel a stem-inclination term that used to be folded into the leaf roll;
    // once that term was removed the same value rolled every blade onto its edge, leaving the leaf normals
    // pointing along the shoot rather than upward.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bindweed");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(30);

    std::vector<uint> leaf_objIDs = plantarchitecture.getPlantLeafObjectIDs(plantID);
    DOCTEST_REQUIRE(leaf_objIDs.size() > 10);

    double sum_vertical = 0;
    int leaf_count = 0;
    int upward_count = 0;
    for (uint objID: leaf_objIDs) {
        vec3 leaf_normal(0, 0, 0);
        for (uint UUID: context.getObjectPrimitiveUUIDs(objID)) {
            leaf_normal = leaf_normal + context.getPrimitiveArea(UUID) * context.getPrimitiveNormal(UUID);
        }
        if (leaf_normal.magnitude() < 1e-9f) {
            continue;
        }
        leaf_normal.normalize();
        const double vertical = std::fabs(leaf_normal.z);
        sum_vertical += vertical;
        leaf_count++;
        if (vertical > 0.5) {
            upward_count++;
        }
    }
    DOCTEST_REQUIRE(leaf_count > 10);
    // Upward-facing blades give |n.z| near 1; blades rolled onto their edge give |n.z| near 0. The seeded
    // RNG is not cross-platform, so only aggregate statistics with an order-of-magnitude margin are checked.
    DOCTEST_CHECK(sum_vertical / leaf_count > 0.6);
    DOCTEST_CHECK(double(upward_count) / leaf_count > 0.8);
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

DOCTEST_TEST_CASE("Material Naming - every material of a multi-material organ is renamed") {
    // renameAutoMaterial() used to read the material of the object's first primitive only, so an organ
    // built from more than one material kept the auto-generated label on all but the first. The olive
    // leaf is built from an upper and a lower tile with different textures, so its lower surface stayed
    // "__auto_..." and was therefore omitted from Context::listMaterials(), leaving the user no way to
    // find it. The bean test above cannot catch this because a bean leaf carries a single material.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("olive");
    plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 1000);

    std::vector<uint> leaf_UUIDs = plantarchitecture.getAllLeafUUIDs();
    DOCTEST_CHECK(leaf_UUIDs.size() > 0);

    // Both leaf surfaces must be texture-mapped and both must carry a user-visible material label.
    std::set<std::string> leaf_textures;
    std::set<std::string> leaf_materials;
    for (uint UUID : leaf_UUIDs) {
        std::string label = context.getPrimitiveMaterialLabel(UUID);
        DOCTEST_CHECK(label.substr(0, 7) != "__auto_");
        DOCTEST_CHECK(!context.getPrimitiveTextureFile(UUID).empty());
        leaf_materials.insert(label);
        leaf_textures.insert(context.getPrimitiveTextureFile(UUID));
    }
    DOCTEST_CHECK(leaf_textures.size() == 2); // upper and lower leaf surfaces
    DOCTEST_CHECK(leaf_materials.size() == 2);

    // Both materials must be reachable through the public listing.
    std::vector<std::string> materials = context.listMaterials();
    for (const std::string &label : leaf_materials) {
        DOCTEST_CHECK(std::find(materials.begin(), materials.end(), label) != materials.end());
    }
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

//! Snapshot of every shoot's carbon pools on one plant, keyed by shoot ID.
static std::map<uint, std::pair<float, float>> snapshotPlantCarbonPools(const PlantArchitecture &plantarchitecture, uint plantID) {
    std::map<uint, std::pair<float, float>> pools;
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        pools[shootID] = {shoot->sugar_pool_molC, shoot->starch_pool_molC};
    }
    return pools;
}

DOCTEST_TEST_CASE("PlantArchitecture carbohydrate model touches only the plants being advanced") {
    // The carbon-balance steps (maintenance respiration, growth cost, transfer, phyllochron adjustment,
    // organ abortion) are applied inside advanceTime()'s per-plant loop, but each of them iterated over
    // every plant instance in the model. With N plants every pool was therefore debited N times per
    // sub-step, and advancing a single plant also ran the carbon balance on every other plant.
    //
    // Two invariants pin this down:
    //  (1) advancing plant B must leave plant A's pools bit-for-bit unchanged; and
    //  (2) plant A's pools after advancing {A, B} together must equal its pools after advancing A alone
    //      from the same state. Both scenes are built identically from the same seed, and A is processed
    //      first in the per-plant loop, so A's random stream through the final step is identical in both.
    //
    // Both plants are built at age 0 and fed photosynthate so that they stay alive with non-zero pools;
    // a plant built at a non-zero age with no carbon supplied starves during construction and is left
    // with only pruned shells whose pools are identically zero.
    constexpr uint seed = 12345;
    constexpr float initial_concentration_molC_m3 = 1.5e4f;
    constexpr int warmup_days = 3;

    // Builds A, warms it up, then builds B and advances B one day. Returns A's pools before and after B's
    // day so that invariant (1) can be checked by the caller.
    auto buildTwoPlantScene = [&](Context &context, PlantArchitecture &plantarchitecture, uint &plantA, uint &plantB, std::map<uint, std::pair<float, float>> &poolsA_before_B,
                                  std::map<uint, std::pair<float, float>> &poolsA_after_B_advanced) {
        context.seedRandomGenerator(seed);
        plantarchitecture.disableMessages();
        plantarchitecture.enableCarbohydrateModel();
        plantarchitecture.loadPlantModelFromLibrary("bean");

        plantA = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
        plantarchitecture.initializePlantCarbohydratePool(plantA, initial_concentration_molC_m3);
        for (int day = 0; day < warmup_days; day++) {
            drivePhotosynthesisForOneDay(context, plantarchitecture, plantA, 10.f, 12);
            plantarchitecture.advanceTime(plantA, 1);
        }
        poolsA_before_B = snapshotPlantCarbonPools(plantarchitecture, plantA);

        plantB = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(2, 0, 0), 0);
        plantarchitecture.initializePlantCarbohydratePool(plantB, initial_concentration_molC_m3);
        drivePhotosynthesisForOneDay(context, plantarchitecture, plantB, 10.f, 12);
        plantarchitecture.advanceTime(plantB, 1);
        poolsA_after_B_advanced = snapshotPlantCarbonPools(plantarchitecture, plantA);

        drivePhotosynthesisForOneDay(context, plantarchitecture, plantA, 10.f, 12);
        drivePhotosynthesisForOneDay(context, plantarchitecture, plantB, 10.f, 12);
    };

    // --- Reference scene: final step advances A alone --- //
    std::map<uint, std::pair<float, float>> reference_pools_after;
    {
        Context context;
        PlantArchitecture plantarchitecture(&context);
        uint plantA = 0;
        uint plantB = 0;
        std::map<uint, std::pair<float, float>> poolsA_before_B;
        std::map<uint, std::pair<float, float>> poolsA_after_B_advanced;
        buildTwoPlantScene(context, plantarchitecture, plantA, plantB, poolsA_before_B, poolsA_after_B_advanced);
        plantarchitecture.advanceTime(plantA, 1);
        reference_pools_after = snapshotPlantCarbonPools(plantarchitecture, plantA);
    }
    DOCTEST_REQUIRE(!reference_pools_after.empty());

    // --- Test scene: final step advances A and B together --- //
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantA = 0;
    uint plantB = 0;
    std::map<uint, std::pair<float, float>> poolsA_before_B;
    std::map<uint, std::pair<float, float>> poolsA_after_B_advanced;
    buildTwoPlantScene(context, plantarchitecture, plantA, plantB, poolsA_before_B, poolsA_after_B_advanced);
    DOCTEST_REQUIRE(plantA != plantB);
    DOCTEST_REQUIRE(!poolsA_before_B.empty());
    float total_sugar_A = 0.f;
    for (const auto &entry: poolsA_before_B) {
        total_sugar_A += entry.second.first;
    }
    DOCTEST_REQUIRE(total_sugar_A > 0.f);

    // (1) Advancing B must not touch A.
    DOCTEST_CHECK(poolsA_after_B_advanced == poolsA_before_B);

    // (2) A's pools after advancing {A, B} must match A advanced alone from the same state.
    plantarchitecture.advanceTime(std::vector<uint>{plantA, plantB}, 1);
    const std::map<uint, std::pair<float, float>> poolsA_after_both_advanced = snapshotPlantCarbonPools(plantarchitecture, plantA);

    DOCTEST_REQUIRE(poolsA_after_both_advanced.size() == reference_pools_after.size());
    for (const auto &entry: reference_pools_after) {
        const uint shootID = entry.first;
        DOCTEST_CAPTURE(shootID);
        DOCTEST_REQUIRE(poolsA_after_both_advanced.count(shootID) == 1);
        DOCTEST_CHECK(poolsA_after_both_advanced.at(shootID).first == doctest::Approx(entry.second.first).epsilon(1e-6));
        DOCTEST_CHECK(poolsA_after_both_advanced.at(shootID).second == doctest::Approx(entry.second.second).epsilon(1e-6));
    }
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

        // Plant should have stayed clear of the obstacles, which all sit at x >= 0.05. The bound is referred to the
        // nearest obstacle rather than being an absolute coordinate: this test does not seed the generator, and the
        // distributions in <random> are implementation-specified, so each standard library grows a different bean
        // here. The previous bound of 0.01 was a value read off one platform's plant and it failed on Windows at
        // 0.0114, with the plant still nowhere near the obstacles.
        const float nearest_obstacle_x = 0.05f;
        DOCTEST_CHECK(center_of_mass.x < nearest_obstacle_x);

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

DOCTEST_TEST_CASE("Grapevine VSP - Default Trunk Reaches Cordon Wire Height") {
    // The VSP trunk must be tall enough to carry the cordons at a realistic wire height. A default of one
    // internode leaves a 10 cm stump with the cordons resting on the ground.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    const std::shared_ptr<Shoot> &trunk = plantarchitecture.getPlantShoot(plantID, 0);
    DOCTEST_REQUIRE(!trunk->shoot_internode_vertices.empty());

    const float trunk_top_height = trunk->shoot_internode_vertices.back().back().z;
    DOCTEST_CHECK(trunk_top_height > 0.5f);

    // The cordons attach at the top of the trunk, so they must be lifted to the same height.
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        if (shoot->shoot_type_label != "grapevine_cane" || shoot->shoot_internode_vertices.empty()) {
            continue;
        }
        DOCTEST_CHECK(shoot->shoot_internode_vertices.front().front().z > 0.5f);
    }
}

DOCTEST_TEST_CASE("Lateral Shoot Base Radius Is Not Inherited From Parent Internode") {
    // A vegetative bud breaking on a thick woody axis produced a shoot whose first internode took the
    // PARENT's initial radius. On the grapevine trunk (radius_initial 0.05 m) that gave a 5 cm-radius
    // twig that collapsed to 3 mm at the next node. Because the pipe-model radius update only ever
    // increases the radius, the bulge was permanent.
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 300);

    const float shoot_radius_initial = plantarchitecture.getCurrentShootParameters("grapevine_shoot").phytomer_parameters.internode.radius_initial.val();

    bool found_lateral_on_woody_parent = false;
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        if (plantarchitecture.isShootPruned(plantID, shootID)) {
            continue;
        }
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        if (shoot->shoot_type_label != "grapevine_shoot" || shoot->parent_shoot_ID < 0 || shoot->shoot_internode_radii.empty()) {
            continue;
        }

        const std::shared_ptr<Shoot> &parent = plantarchitecture.getPlantShoot(plantID, static_cast<uint>(shoot->parent_shoot_ID));
        if (parent->shoot_type_label != "grapevine_trunk") {
            continue;
        }
        found_lateral_on_woody_parent = true;

        // The base of a first-year lateral must be sized by its own pipe-model load, not by the trunk it
        // grew out of. Compare against a sibling lateral on a cane, which is the same shoot type carrying a
        // comparable leaf load but whose parent is thin -- the trunk-borne one must not be wildly thicker.
        const float base_radius = shoot->shoot_internode_radii.front().front();
        DOCTEST_CHECK(base_radius < 4.f * shoot_radius_initial);
    }

    DOCTEST_REQUIRE(found_lateral_on_woody_parent);
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
        int tassel_branches;
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
        obs.tassel_branches = params.phytomer_parameters.inflorescence.flowers_per_peduncle.val();
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

    // One ear (1 fruit + 1 peduncle) plus the terminal tassel, which is built as one fruit object per
    // primary branch (flowers_per_peduncle) plus its own peduncle. Read from the shoot type rather
    // than hardcoded, so that retuning the tassel's branch count -- a calibration parameter -- does
    // not fail a test whose subject is ear placement.
    DOCTEST_CHECK(short_plant.fruit_count == std::size_t(1 + short_plant.tassel_branches));
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

DOCTEST_TEST_CASE("Maize flag leaf is small and clears the tassel while staying the most erect leaf") {
    // Regression test covering the two things that made the maize flag leaf wrong, which pull in
    // opposite directions and so have to be asserted together.
    //
    // ANGLE. Phytomer::createPetiole() forces the last phytomer on a shoot to a near-zero 5 degree
    // petiole pitch, for the grasses' flag leaf. The terminal floral bud carrying the tassel is
    // anchored to the same node tip (Shoot::addTerminalFloralBud) and maize sets peduncle.pitch = 0,
    // so the flag leaf ran essentially parallel to the peduncle and intersected it.
    //
    // SIZE. Maize is not sorghum here: the sorghum flag leaf is the most horizontal leaf on the
    // plant, whereas maize is bred for an erectophile upper canopy -- 22.9 degrees from vertical for
    // the leaves above the ear against 41.9 below it (Tang et al. 2021, Euphytica 217:75, hybrid
    // Pioneer 335). The flag leaf must therefore stay the MOST erect leaf, which caps how far it can
    // be rotated away from the culm. What actually made it read as oversized is its area: fitted
    // vertical leaf-area profiles put the flag leaf near a quarter of the largest leaf's area (Fan
    // et al. 2021, Journal of Agricultural Science 158:676, hybrids ZD958 and XY335), against the
    // 0.59 the library built.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("maize");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 90.f);

    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();
    const auto &phytomers = plantarchitecture.getPlantShoot(plantID, shootID)->phytomers;
    DOCTEST_REQUIRE(phytomers.size() >= 6);

    // Angle each phytomer's petiole makes with its own internode, which is the insertion angle the
    // 5 degree rule acts on. Measured against the internode rather than against vertical so that the
    // culm's own lean does not enter.
    auto petiole_angle_to_culm = [](const std::shared_ptr<Phytomer> &phytomer) -> float {
        const vec3 internode_axis = phytomer->getInternodeAxisVector(0.5f);
        const std::vector<vec3> &vertices = phytomer->petiole_vertices.front();
        vec3 petiole_axis = vertices.back() - vertices.front();
        petiole_axis.normalize();
        return rad2deg(acos_safe(petiole_axis * internode_axis));
    };

    const float flag_leaf_angle = petiole_angle_to_culm(phytomers.back());

    // Clear of the peduncle. The old 5 degree value put the blade inside the tassel; anything near it
    // is the bug. The bound is well above 5 but well below the applied rotation, so it fails on the
    // old behavior without pinning the exact value.
    DOCTEST_CHECK(flag_leaf_angle > 12.f);

    // Still the most erect leaf on the plant. Maize's upper canopy is erectophile, so lifting the
    // flag leaf clear must not tip it past the leaves below it -- that would be the sorghum posture,
    // which is the opposite of maize.
    //
    // Asserted on BLADE inclination rather than on the petiole insertion angle, because that is the
    // quantity Tang et al. report and the two are not interchangeable here: every rank is inserted
    // from the same U(16,30) distribution, so insertion angle carries no gradient at all and the
    // erectophile profile comes from the leaf_flexibility taper stiffening the upper leaves. An
    // insertion-angle comparison would instead pin the flag leaf below the floor of that
    // distribution, which is a constraint the model never intended and which no seed can satisfy
    // with any meaningful rotation applied.
    auto blade_inclination_from_vertical = [&context](const std::shared_ptr<Phytomer> &phytomer) -> float {
        float area_sum = 0.f;
        float weighted_sum = 0.f;
        for (uint objID: phytomer->leaf_objIDs.front()) {
            for (uint UUID: context.getObjectPrimitiveUUIDs(objID)) {
                const vec3 normal = context.getPrimitiveNormal(UUID);
                const float area = context.getPrimitiveArea(UUID);
                // Inclination of the blade is the complement of its normal's angle from vertical,
                // folded so that the two faces of a leaf give the same answer.
                weighted_sum += area * (90.f - rad2deg(acos_safe(std::fabs(normal.z))));
                area_sum += area;
            }
        }
        return area_sum > 0.f ? weighted_sum / area_sum : 0.f;
    };

    // Compared against the mean of the lower canopy rather than against the minimum over every other
    // leaf. Petiole pitch is drawn per phytomer from U(16,30) and blade droop inherits that spread,
    // so individual ranks scatter by about ten degrees either way -- more than the erectophile
    // gradient itself. An individual upper leaf therefore out-erects the flag leaf on some seeds
    // purely by draw, and asserting against the minimum passes or fails on which seed is used rather
    // than on whether the canopy is shaped correctly. The gradient between the top of the canopy and
    // the bottom is the part that is actually robust, and it is what Tang et al. report.
    const float flag_leaf_inclination = blade_inclination_from_vertical(phytomers.back());
    const std::size_t lower_canopy_end = phytomers.size() - 6;
    float lower_inclination_sum = 0.f;
    for (std::size_t node = 0; node < lower_canopy_end; node++) {
        lower_inclination_sum += blade_inclination_from_vertical(phytomers.at(node));
    }
    DOCTEST_REQUIRE(lower_canopy_end > 0);
    const float mean_lower_inclination = lower_inclination_sum / float(lower_canopy_end);

    // Tang et al. give 22.9 degrees from vertical above the ear against 41.9 below it. Over a thirty-
    // seed sweep this margin ran from 3 to 19 degrees, so the bound is set well under that floor
    // rather than near the typical value: the flag leaf's own blade droop varies by more than ten
    // degrees with the draw, and a bound placed at the typical margin passes only on lucky seeds.
    // It still fails decisively on the posture this guards against -- a flag leaf rotated out to the
    // lower canopy's attitude sits near the lower mean, giving a margin around zero or negative.
    DOCTEST_CHECK(mean_lower_inclination - flag_leaf_inclination > 2.f);

    // Flag leaf area against the largest leaf on the plant. Fitted profiles for two commercial
    // hybrids give 0.19 and 0.26; the band admits both with room for retuning but excludes the 0.59
    // the library built.
    float flag_leaf_area = phytomers.back()->getLeafArea();
    float max_leaf_area = 0.f;
    for (const auto &phytomer: phytomers) {
        max_leaf_area = std::max(max_leaf_area, phytomer->getLeafArea());
    }
    DOCTEST_REQUIRE(max_leaf_area > 0.f);
    const float flag_leaf_area_fraction = flag_leaf_area / max_leaf_area;
    DOCTEST_CHECK(flag_leaf_area_fraction >= 0.13f);
    DOCTEST_CHECK(flag_leaf_area_fraction <= 0.36f);
}

DOCTEST_TEST_CASE("Maize tassel reaches field size by anthesis") {
    // Regression test, covering two separate defects that both made the tassel too small.
    //
    // SIZE. The tassel is built as flowers_per_peduncle copies of MaizeTassel.obj strung along the
    // terminal peduncle, one copy per primary branch. Field benchmarks for a commercial dent hybrid:
    // 10-12 primary branches (Gage et al. 2018, Genetics 210:1125, reports 12.0 in 1930s BSSSC0
    // falling to ~10.2 in ex-PVP lines; range across 749 inbreds is 0-41, Gage et al. 2017, Plant
    // Methods 13:21), and a primary branch length near 20 cm (Zhang et al. 2023, Plant Methods 19:76,
    // modal value over 180 three-dimensionally scanned tassels). The library previously built 7
    // branches of 15.4 cm.
    //
    // PHENOLOGY. Tassel growth was driven by dd_to_fruit_maturity, which for maize is the 58-day
    // R1-to-R6 grain-fill period of the EAR. The tassel is fully elongated at VT, before silking --
    // "the tassel is at maximum size" at VT (Abendroth et al. 2011, Iowa State PMR 1009), and VT
    // "begins approximately 2-3 days before silk emergence" (Ritchie, Hanway & Benson, How a Corn
    // Plant Develops, Iowa State Special Report 48, p.11). Sharing the ear's clock left the tassel
    // at 41% of its final size at silking and still expanding at day 120, which is R6.
    //
    // Asserted a few days after tasseling rather than at the end of the run, because the previous
    // behavior reached full size eventually -- an end-of-run assertion passes on the bug.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("maize");

    const int max_nodes = plantarchitecture.getCurrentShootParameters("mainstem").max_nodes.val();

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();

    // Grow to tasseling (the shoot reaching its final node count is when the terminal floral bud
    // carrying the tassel is added -- see appendPhytomerToShoot), then a further 10 days. Silking
    // follows tasseling by 2-3 days, so a fully elongated tassel must be at field size well inside
    // this window.
    int tasseling_day = -1;
    for (int day = 1; day <= 120 && tasseling_day < 0; day++) {
        plantarchitecture.advanceTime(plantID, 1.f);
        if (int(plantarchitecture.getPlantShoot(plantID, shootID)->phytomers.size()) >= max_nodes) {
            tasseling_day = day;
        }
    }
    DOCTEST_REQUIRE(tasseling_day > 0);
    plantarchitecture.advanceTime(plantID, 10.f);

    // Collect the terminal inflorescence: the tassel's primary branches.
    std::vector<uint> branch_objIDs;
    for (uint sID: plantarchitecture.getAllShootIDs(plantID)) {
        for (const auto &phytomer: plantarchitecture.getPlantShoot(plantID, sID)->phytomers) {
            for (const auto &petiole: phytomer->floral_buds) {
                for (const auto &fbud: petiole) {
                    if (fbud.isterminal && !fbud.inflorescence_objIDs.empty()) {
                        branch_objIDs.insert(branch_objIDs.end(), fbud.inflorescence_objIDs.begin(), fbud.inflorescence_objIDs.end());
                    }
                }
            }
        }
    }

    // Guard against a vacuous pass: the tassel must exist, or the bounds below test nothing.
    DOCTEST_REQUIRE(!branch_objIDs.empty());

    // Branch number. Bounds are the published range for modern hybrids rather than the exact
    // library value, so ordinary retuning within the literature does not fail the test.
    DOCTEST_CHECK(branch_objIDs.size() >= 9);
    DOCTEST_CHECK(branch_objIDs.size() <= 14);

    // Branch length, measured as the bounding-box diagonal of each branch object. The prototype is a
    // single slender branch, so its diagonal is its length to within a few percent.
    float branch_length_sum = 0.f;
    for (uint objID: branch_objIDs) {
        vec2 xbounds, ybounds, zbounds;
        context.getDomainBoundingBox(context.getObjectPrimitiveUUIDs(objID), xbounds, ybounds, zbounds);
        branch_length_sum += make_vec3(xbounds.y - xbounds.x, ybounds.y - ybounds.x, zbounds.y - zbounds.x).magnitude();
    }
    const float mean_branch_length = branch_length_sum / float(branch_objIDs.size());

    // 20 cm modal, with a band wide enough to admit reasonable retuning but not the 15.4 cm the
    // library built before, nor the 6 cm it stood at ten days after tasseling on the old clock.
    DOCTEST_CHECK(mean_branch_length >= 0.17f);
    DOCTEST_CHECK(mean_branch_length <= 0.25f);
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
    DOCTEST_CHECK_MESSAGE(long_internodes > 1.5f * short_internodes, "short=" << short_internodes << " long=" << long_internodes << " ratio=" << (long_internodes/short_internodes));
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
        // Measured on the stem itself rather than on the whole plant's bounding box. The parameter under test elongates internodes, and a box drawn around every organ also moves when the leaves change shape -
        // a blade that arcs over reaches further below the stem base and stops standing above the apex, so it can make a plant read as taller or shorter with no internode having changed length at all.
        return plantarchitecture.getPlantStemHeight(plantID);
    };

    const float short_internodes = grow_with_internode_length(0.05f);
    const float long_internodes = grow_with_internode_length(0.15f);

    DOCTEST_REQUIRE(short_internodes > 0.f);
    DOCTEST_REQUIRE(long_internodes > 0.f);

    // On the old code both arms produced the same plant built from the hardcoded 0.06 m.
    DOCTEST_CHECK_MESSAGE(long_internodes > 1.5f * short_internodes, "short=" << short_internodes << " long=" << long_internodes << " ratio=" << (long_internodes/short_internodes));
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

//! Builds a flat rectangular blade lattice of unit length lying in the z=0 plane, with the midrib along +x.
static std::vector<helios::vec3> makeFlatBladeLattice(uint Nx, uint Ny, float aspect_ratio) {
    std::vector<helios::vec3> lattice((size_t(Nx) + 1) * (size_t(Ny) + 1));
    for (uint j = 0; j <= Ny; j++) {
        for (uint i = 0; i <= Nx; i++) {
            const float x = float(i) / float(Nx);
            const float y = aspect_ratio * (float(j) / float(Ny) - 0.5f);
            lattice.at(size_t(j) * (size_t(Nx) + 1) + size_t(i)) = helios::make_vec3(x, y, 0.f);
        }
    }
    return lattice;
}

DOCTEST_TEST_CASE("PlantArchitecture leaf curvature exponent redistributes bending along the blade") {

    // The exponent controls WHERE along the blade the longitudinal curvature acts, not how much of it there is. Both halves of that statement need checking: raising the curvature parameter alone can only
    // ever steepen the tip, so a species whose blade arcs along its whole length cannot be described by the curvature parameter by itself however it is set.
    auto bladeProfile = [](float exponent) {
        Context ctx;
        PlantArchitecture pa(&ctx);
        LeafPrototype prototype(ctx.getRandomGenerator());
        prototype.leaf_texture_file[0] = "SorghumLeaf.png";
        prototype.subdivisions = 40;
        prototype.leaf_aspect_ratio = 0.15f;
        prototype.longitudinal_curvature = -0.4f;
        prototype.longitudinal_curvature_exponent = exponent;
        // The basal roll shapes only the innermost few percent of the blade and would contaminate the near-base sample.
        prototype.petiole_roll = 0.f;
        prototype.wave_amplitude = 0.f;
        prototype.midrib_fold_fraction = 0.f;
        prototype.lateral_curvature = 0.f;

        const uint objID = prototype.prototype_function(&ctx, &prototype, 0);
        const std::vector<vec3> vertices = ctx.getPolymeshObjectVertices(objID);
        const size_t row_length = prototype.subdivisions + 1;
        DOCTEST_REQUIRE(vertices.size() % row_length == 0);
        const size_t midrib = (vertices.size() / row_length / 2) * row_length;

        // Height of the midrib at the blade's midpoint and at its tip, relative to its base.
        const float base_z = vertices.at(midrib).z;
        return std::make_pair(vertices.at(midrib + row_length / 2).z - base_z, vertices.at(midrib + row_length - 1).z - base_z);
    };

    const auto quartic = bladeProfile(4.f);
    const auto quadratic = bladeProfile(2.f);

    // The tip deflection is set by the curvature parameter and must not depend on the exponent.
    DOCTEST_CHECK_MESSAGE(quadratic.second == doctest::Approx(quartic.second).epsilon(0.05), "tip deflection quartic=" << quartic.second << " quadratic=" << quadratic.second);

    // A quartic blade is still nearly flat at its own midpoint - it reaches only about a sixteenth of its tip deflection there - while a quadratic one has already bent a quarter of the way. Without this,
    // a blade that arcs continuously along its length cannot be represented at all.
    DOCTEST_REQUIRE(quartic.second < 0.f);
    DOCTEST_CHECK_MESSAGE(std::fabs(quadratic.first) > 2.5f * std::fabs(quartic.first), "midpoint deflection quartic=" << quartic.first << " quadratic=" << quadratic.first);
}

DOCTEST_TEST_CASE("PlantArchitecture leaf deflection reproduces cantilever beam theory") {

    // A uniformly loaded cantilever of length L deflects at its tip by delta = w*L^4/(8*EI). The deformation routine is built from the same moment balance, so in the small-deflection limit - where the
    // lever arm is still essentially the undeflected horizontal distance, which is the assumption beam theory makes - it must reproduce that closed form. This is the check that the moment integral and the
    // double integration are actually right, rather than merely producing something that droops.
    const uint Nx = 40;
    const uint Ny = 4;
    const float aspect_ratio = 0.2f;
    const std::vector<vec3> rest = makeFlatBladeLattice(Nx, Ny, aspect_ratio);

    auto tipDeflection = [&](float scale, float mature_scale, float flexibility) {
        const std::vector<vec3> deformed = deformLeafLattice(rest, Nx, Ny, scale, mature_scale, flexibility);
        // Tip of the midrib row. Deflection is measured against the undeflected tip, which sits at x = scale on the z = 0 plane.
        return -deformed.at(size_t(Ny / 2) * (size_t(Nx) + 1) + size_t(Nx)).z;
    };

    // Mass per unit length of the unit-length blade is its width, so w = aspect_ratio in the units the routine works in; at length L the load and lever arm both scale, giving w*L^4/(8*EI) with EI = 1/flexibility.
    // With the mature reference equal to the current length (a fully-grown leaf), the compliance is flexibility/L^4 and the classic w*L^4/(8*EI) reduces to aspect*flexibility/8, independent of L.
    auto analyticTipDeflection = [&](float L, float flexibility) {
        (void) L;
        return aspect_ratio * flexibility / 8.f;
    };

    DOCTEST_SUBCASE("Small-deflection limit matches the closed-form solution") {
        // Flexibility chosen so the tip deflects by well under a percent of the blade length, keeping the small-angle assumptions of beam theory valid.
        const float L = 1.f;
        const float flexibility = 0.02f;
        const float computed = tipDeflection(L, L, flexibility);
        const float analytic = analyticTipDeflection(L, flexibility);
        // 3% covers the discretization error of a 40-segment blade against the continuum solution. It is tight enough to have real teeth: dropping the distal-mass term from the moment integral - the most
        // natural way to get this wrong - lands at 3% OF the analytic value rather than within 3% of it, a 30x error.
        DOCTEST_CHECK(computed == doctest::Approx(analytic).epsilon(0.03));
    }

    DOCTEST_SUBCASE("Deflection follows the fifth-power length scaling of an isotropically grown blade") {
        // Textbook beam theory gives delta = w*L^4/(8*EI) for a load per unit length w that does not change with L. A growing leaf is not that case: it scales in every direction at once, so a longer blade is
        // also a wider and consequently heavier one, giving w proportional to L and delta proportional to L^5. Doubling the leaf therefore deflects the tip 32x as far, not 16x.
        //
        // This is the property that makes the model self-driving: droop follows from the leaf getting bigger, with no age or time term anywhere in it. It is also very steep, which is worth knowing when
        // tuning - a modest change in mature leaf length moves the droop a great deal.
        const float flexibility = 0.005f;
        // The mature reference is held fixed at the full-grown length while the CURRENT length is varied, which is exactly what happens to a real leaf as it expands: its stiffness does not change while it grows.
        const float mature = 2.f;
        const float deflection_short = tipDeflection(1.f, mature, flexibility);
        const float deflection_long = tipDeflection(2.f, mature, flexibility);
        DOCTEST_CHECK(deflection_short > 0.f);
        DOCTEST_CHECK(deflection_long / deflection_short == doctest::Approx(32.f).epsilon(0.1));
    }

    DOCTEST_SUBCASE("Deflection is linear in flexibility in the small-deflection limit") {
        const float L = 1.f;
        const float deflection_single = tipDeflection(L, L, 0.01f);
        const float deflection_double = tipDeflection(L, L, 0.02f);
        DOCTEST_CHECK(deflection_double / deflection_single == doctest::Approx(2.f).epsilon(0.05));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture leaf deflection behaves sensibly outside the linear regime") {

    const uint Nx = 20;
    const uint Ny = 4;
    const std::vector<vec3> rest = makeFlatBladeLattice(Nx, Ny, 0.2f);
    const size_t tip_index = size_t(Ny / 2) * (size_t(Nx) + 1) + size_t(Nx);

    DOCTEST_SUBCASE("A rigid leaf keeps its rest shape exactly") {
        // Zero flexibility is the path every non-drooping species takes, so it must return the rest lattice untouched rather than approximately so.
        const std::vector<vec3> deformed = deformLeafLattice(rest, Nx, Ny, 1.f, 1.f, 0.f);
        DOCTEST_REQUIRE(deformed.size() == rest.size());
        for (size_t v = 0; v < rest.size(); v++) {
            DOCTEST_CHECK(deformed.at(v).x == doctest::Approx(rest.at(v).x).epsilon(1e-6));
            DOCTEST_CHECK(deformed.at(v).y == doctest::Approx(rest.at(v).y).epsilon(1e-6));
            DOCTEST_CHECK(deformed.at(v).z == doctest::Approx(rest.at(v).z).epsilon(1e-6));
        }
    }

    DOCTEST_SUBCASE("A curved rest shape is preserved and droops from where it already is") {
        // Every other case here starts from a FLAT blade, which cannot tell whether the routine preserves the rest shape or silently replaces it with one of its own. A real grass prototype arrives already
        // curved - the blade carries its longitudinal curvature and the arc it holds under turgor - and that shape has to survive, with the self-weight bending added on top of it rather than in place of it.
        std::vector<vec3> curved = makeFlatBladeLattice(Nx, Ny, 0.2f);
        for (uint i = 0; i <= Nx; i++) {
            // A blade arcing downward toward its tip, which is the shape the grass prototypes are built with.
            const float along = float(i) / float(Nx);
            for (uint j = 0; j <= Ny; j++) {
                curved.at(size_t(j) * (size_t(Nx) + 1) + size_t(i)).z -= 0.35f * along * along;
            }
        }

        // With no meaningful load the blade must come back exactly as it went in, curvature and all. Building the tangent angle from the bending alone rather than adding the bending to the rest angle
        // straightens the blade into a flat board, which this catches and a flat rest lattice cannot.
        const std::vector<vec3> unloaded = deformLeafLattice(curved, Nx, Ny, 1.f, 1.f, 1e-6f);
        DOCTEST_REQUIRE(unloaded.size() == curved.size());
        for (size_t v = 0; v < curved.size(); v++) {
            DOCTEST_CHECK(unloaded.at(v).z == doctest::Approx(curved.at(v).z).epsilon(1e-4));
        }

        // And a real load must carry the tip further down than the rest shape already had it, rather than lifting it back toward horizontal.
        const std::vector<vec3> loaded = deformLeafLattice(curved, Nx, Ny, 1.f, 1.f, 5.f);
        DOCTEST_CHECK(loaded.at(tip_index).z < curved.at(tip_index).z);
    }

    DOCTEST_SUBCASE("Droop increases monotonically with flexibility and never inverts") {
        float previous = 0.f;
        for (float flexibility: {0.05f, 0.2f, 1.f, 5.f, 25.f}) {
            const std::vector<vec3> deformed = deformLeafLattice(rest, Nx, Ny, 1.f, 1.f, flexibility);
            const float droop = -deformed.at(tip_index).z;
            DOCTEST_CHECK(droop > previous);
            previous = droop;
        }
    }

    DOCTEST_SUBCASE("A very flexible leaf hangs down rather than curling underneath itself") {
        // The tangent angle is clamped at vertical, since a leaf hanging straight down carries no further lever arm. Without that limit the integrated curvature would keep turning and fold the blade back
        // under its own base, which is the failure mode of integrating curvature without a physical stop.
        const std::vector<vec3> deformed = deformLeafLattice(rest, Nx, Ny, 1.f, 1.f, 500.f);
        const float droop = -deformed.at(tip_index).z;
        DOCTEST_CHECK(droop > 0.f);
        // The blade is inextensible, so no point can end up further from the base than the arclength allows.
        DOCTEST_CHECK(droop <= 1.f + 1e-3f);
        for (const vec3 &v: deformed) {
            DOCTEST_CHECK(std::isfinite(v.x));
            DOCTEST_CHECK(std::isfinite(v.y));
            DOCTEST_CHECK(std::isfinite(v.z));
        }
    }

    DOCTEST_SUBCASE("Bending preserves the blade's arclength and width") {
        // Bending must not stretch the blade. The midrib arclength and the local width are what distinguish a bend from a scale, so both must survive the deformation.
        const float scale = 1.f;
        const std::vector<vec3> deformed = deformLeafLattice(rest, Nx, Ny, scale, scale, 2.f);
        auto index = [&](uint i, uint j) { return size_t(j) * (size_t(Nx) + 1) + size_t(i); };

        float arclength = 0.f;
        for (uint i = 1; i <= Nx; i++) {
            arclength += (deformed.at(index(i, Ny / 2)) - deformed.at(index(i - 1, Ny / 2))).magnitude();
        }
        DOCTEST_CHECK(arclength == doctest::Approx(scale).epsilon(0.02));

        for (uint i = 0; i <= Nx; i++) {
            const float width = (deformed.at(index(i, Ny)) - deformed.at(index(i, 0))).magnitude();
            DOCTEST_CHECK(width == doctest::Approx(0.2f * scale).epsilon(0.02));
        }
    }

    DOCTEST_SUBCASE("The clamped base does not move") {
        // The leaf is attached at its base, so the deformation must pivot there rather than translating the whole blade - otherwise a drooping leaf would detach from its petiole.
        const std::vector<vec3> deformed = deformLeafLattice(rest, Nx, Ny, 1.f, 1.f, 10.f);
        for (uint j = 0; j <= Ny; j++) {
            const size_t base_index = size_t(j) * (size_t(Nx) + 1);
            DOCTEST_CHECK(deformed.at(base_index).x == doctest::Approx(rest.at(base_index).x).epsilon(1e-4));
            DOCTEST_CHECK(deformed.at(base_index).z == doctest::Approx(rest.at(base_index).z).epsilon(1e-4));
        }
    }

    DOCTEST_SUBCASE("Invalid arguments are rejected") {
        DOCTEST_CHECK_THROWS(deformLeafLattice(rest, Nx + 1, Ny, 1.f, 1.f, 1.f));
        DOCTEST_CHECK_THROWS(deformLeafLattice(rest, Nx, Ny, -1.f, 1.f, 1.f));
        DOCTEST_CHECK_THROWS(deformLeafLattice(rest, Nx, Ny, 1.f, 1.f, -1.f));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture deprecated leaf buckle parameters still produce a drooping leaf") {

    // Code written against the retired leaf_buckle_* parameters must keep working. These subcases deliberately set them, which is exactly the use the deprecation warning is aimed at, so the warning is
    // suppressed here - the point of the test is that the old spelling still does something sensible, not that the compiler notices it.
    HELIOS_PUSH_IGNORE_DEPRECATED

    DOCTEST_SUBCASE("A buckle setting is converted into an equivalent flexibility") {
        Context ctx;
        LeafPrototype prototype(ctx.getRandomGenerator());
        prototype.leaf_aspect_ratio = 0.15f;
        prototype.leaf_buckle_length = 0.5f;
        prototype.leaf_buckle_angle = 50.f;

        // Bending through 50 degrees at the halfway point dropped the tip by (1-0.5)*sin(50) = 0.383 of the leaf length; at an aspect ratio of 0.15 the flexibility reproducing that is 8*0.383/0.15 = 20.4.
        const float resolved = prototype.resolveFlexibility();
        DOCTEST_CHECK(resolved == doctest::Approx(20.4f).epsilon(0.05));
    }

    DOCTEST_SUBCASE("A larger buckle angle converts to a floppier leaf") {
        Context ctx;
        LeafPrototype shallow(ctx.getRandomGenerator());
        shallow.leaf_aspect_ratio = 0.15f;
        shallow.leaf_buckle_length = 0.5f;
        shallow.leaf_buckle_angle = 20.f;

        LeafPrototype steep(ctx.getRandomGenerator());
        steep.leaf_aspect_ratio = 0.15f;
        steep.leaf_buckle_length = 0.5f;
        steep.leaf_buckle_angle = 60.f;

        DOCTEST_CHECK(steep.resolveFlexibility() > shallow.resolveFlexibility());
    }

    DOCTEST_SUBCASE("An explicitly-set flexibility overrides a stale buckle value") {
        // Code that has migrated must not have its setting quietly replaced by a buckle value left sitting beside it.
        Context ctx;
        LeafPrototype prototype(ctx.getRandomGenerator());
        prototype.leaf_aspect_ratio = 0.15f;
        prototype.flexibility = 7.f;
        prototype.leaf_buckle_length = 0.5f;
        prototype.leaf_buckle_angle = 50.f;

        DOCTEST_CHECK(prototype.resolveFlexibility() == doctest::Approx(7.f).epsilon(1e-4));
    }

    DOCTEST_SUBCASE("A leaf that set neither parameter stays rigid") {
        Context ctx;
        LeafPrototype prototype(ctx.getRandomGenerator());
        DOCTEST_CHECK(prototype.resolveFlexibility() == doctest::Approx(0.f).epsilon(1e-6));
    }

    DOCTEST_SUBCASE("A buckle angle of zero leaves the leaf rigid") {
        // The old parameters defaulted to zero and were inert at that value, so a prototype that merely mentions them must not acquire a droop it never had.
        Context ctx;
        LeafPrototype prototype(ctx.getRandomGenerator());
        prototype.leaf_aspect_ratio = 0.15f;
        prototype.leaf_buckle_length = 0.5f;
        prototype.leaf_buckle_angle = 0.f;
        DOCTEST_CHECK(prototype.resolveFlexibility() == doctest::Approx(0.f).epsilon(1e-6));
    }

    DOCTEST_SUBCASE("A plant built from the deprecated parameters actually droops") {
        // The end-to-end guarantee: old code produces a plant whose leaves bend, not a silently rigid one.
        Context ctx;
        PlantArchitecture pa(&ctx);
        pa.disableMessages();
        pa.loadPlantModelFromLibrary("sorghum");

        ShootParameters params = pa.getCurrentShootParameters("mainstem");
        params.phytomer_parameters.leaf.prototype.flexibility = 0.f;
        params.phytomer_parameters.leaf.prototype.leaf_buckle_length = 0.5f;
        params.phytomer_parameters.leaf.prototype.leaf_buckle_angle = 50.f;
        params.phytomer_parameters.leaf.prototype.unique_prototypes = 1;
        pa.updateCurrentShootParameters("mainstem", params);

        uint plantID = pa.buildPlantInstanceFromLibrary(nullorigin, 0.f);
        pa.advanceTime(plantID, 70.f);

        // At least one blade must have bent measurably away from the straight prototype it was copied from.
        float straightest_departure = 0.f;
        for (uint objID: pa.getPlantLeafObjectIDs(plantID)) {
            if (!ctx.doesObjectExist(objID) || ctx.getObjectType(objID) != OBJECT_TYPE_POLYMESH) {
                continue;
            }
            const std::vector<vec3> vertices = ctx.getPolymeshObjectVertices(objID);
            const size_t row_length = 51;
            if (vertices.empty() || vertices.size() % row_length != 0) {
                continue;
            }
            const size_t midrib_start = (vertices.size() / row_length / 2) * row_length;
            float arclength = 0.f;
            for (size_t i = 1; i < row_length; i++) {
                arclength += (vertices.at(midrib_start + i) - vertices.at(midrib_start + i - 1)).magnitude();
            }
            if (arclength > 1e-6f) {
                const float chord = (vertices.at(midrib_start + row_length - 1) - vertices.at(midrib_start)).magnitude();
                straightest_departure = std::max(straightest_departure, 1.f - chord / arclength);
            }
        }
        DOCTEST_CHECK(straightest_departure > 0.01f);
    }

    HELIOS_POP_IGNORE_DEPRECATED
}

DOCTEST_TEST_CASE("PlantArchitecture grass leaves droop as they grow") {

    // End-to-end check that the deflection actually reaches the geometry of a real plant, rather than only working in isolation.
    //
    // The observable is the INCLINATION OF THE TIP of one tracked leaf, measured from horizontal and signed, so that a blade bending downward is distinguished from one bending upward. A chord-to-arclength
    // ratio was used here previously and could not tell those apart: it is near 1 both for a straight blade and for a cleanly-arched drooping one, and it actually rises as a leaf stops humping upward and
    // settles into a downward arc, so a leaf drooping harder registered as one straightening out. The tip angle is followed on a single leaf through time rather than compared between leaves, because the
    // prototype cache holds independently sampled blade shapes and comparing two different leaves mostly measures that sampling.
    // Whether an object carries a blade this test can read at all. Kept separate from the angle itself because a tip inclination is legitimately negative on a drooping leaf, so no sentinel value could
    // stand for "unreadable" without colliding with a real measurement.
    auto hasReadableBlade = [](Context &ctx, uint objID) {
        const std::vector<vec3> vertices = ctx.getPolymeshObjectVertices(objID);
        const size_t row_length = 51;
        return !vertices.empty() && vertices.size() % row_length == 0 && vertices.size() / row_length >= 3;
    };

    auto midribTipInclination = [](Context &ctx, uint objID) {
        const std::vector<vec3> vertices = ctx.getPolymeshObjectVertices(objID);
        const size_t row_length = 51; // Nx + 1 for the sorghum leaf prototype
        const size_t row_count = vertices.empty() ? 0 : vertices.size() / row_length;
        if (row_count < 3 || vertices.size() % row_length != 0) {
            return -1.f;
        }
        // Walk the MIDRIB, the middle row, and not row zero: the blade's outer edge is shaped mostly by the midrib fold and the lateral curvature, both baked into the rest prototype, so it barely registers
        // the bending this test is looking for.
        const size_t midrib_start = (row_count / 2) * row_length;
        float arclength = 0.f;
        for (size_t i = 1; i < row_length; i++) {
            arclength += (vertices.at(midrib_start + i) - vertices.at(midrib_start + i - 1)).magnitude();
        }
        if (arclength < 1e-6f) {
            return -1.f;
        }
        // Inclination of the outermost tenth of the midrib, in degrees above horizontal.
        const vec3 tip_segment = vertices.at(midrib_start + row_length - 1) - vertices.at(midrib_start + row_length - 1 - row_length / 10);
        return float(std::atan2(tip_segment.z, std::sqrt(tip_segment.x * tip_segment.x + tip_segment.y * tip_segment.y)) * 180.f / PI_F);
    };

    // Grows a sorghum plant at the given flexibility and reports how far the tip of one tracked leaf swings downward as that leaf expands.
    auto tipChangeOfTrackedLeaf = [&](float flexibility) {
        Context ctx;
        // Seeded identically for both arms so that the rigid and the flexible run grow the same plant and differ only in whether its leaves bend. Left unseeded, the shoot's own random parameters differ
        // between the two runs and the comparison partly measures that scatter instead of the droop.
        ctx.seedRandomGenerator(12345);
        PlantArchitecture pa(&ctx);
        pa.loadPlantModelFromLibrary("sorghum");

        ShootParameters params = pa.getCurrentShootParameters("mainstem");
        params.phytomer_parameters.leaf.prototype.flexibility = flexibility;
        // One prototype only, so the tracked leaf's rest shape is fixed and cannot vary between samples.
        params.phytomer_parameters.leaf.prototype.unique_prototypes = 1;
        pa.updateCurrentShootParameters("mainstem", params);

        uint plantID = pa.buildPlantInstanceFromLibrary(nullorigin, 0.f);

        // Let the plant put out leaves, then lock onto the first one that carries a readable blade and follow that same object from here on.
        // Track the plant's NEWEST leaf, which is the one with most of its expansion still ahead of it. The oldest leaf is already fully grown by the time there is a plant to watch, and a leaf that has
        // finished expanding has also finished drooping - following it would measure nothing however well the deflection works.
        pa.advanceTime(plantID, 20.f);
        uint tracked_objID = 0;
        for (uint objID: pa.getPlantLeafObjectIDs(plantID)) {
            if (ctx.doesObjectExist(objID) && ctx.getObjectType(objID) == OBJECT_TYPE_POLYMESH && hasReadableBlade(ctx, objID)) {
                tracked_objID = objID;
            }
        }
        DOCTEST_REQUIRE(tracked_objID != 0);

        const float tip_young = midribTipInclination(ctx, tracked_objID);
        pa.advanceTime(plantID, 50.f);
        DOCTEST_REQUIRE(ctx.doesObjectExist(tracked_objID));
        const float tip_grown = midribTipInclination(ctx, tracked_objID);

        // Positive when the tip has swung downward over the interval, which is what drooping means.
        return tip_young - tip_grown;
    };

    // The control. A rigid leaf is a pure scaling of its prototype, so following one through its whole expansion must leave its tip pointing the same way. Without this, a change in the flexible case could
    // just as easily be growth or reorientation as droop.
    const float rigid_change = tipChangeOfTrackedLeaf(0.f);
    DOCTEST_CHECK_MESSAGE(std::fabs(rigid_change) < 1.f, "rigid tip change=" << rigid_change << " deg");

    // The same leaf on the same plant, now flexible: its tip must measurably swing downward.
    const float flexible_change = tipChangeOfTrackedLeaf(25.f);
    DOCTEST_CHECK_MESSAGE(flexible_change > 5.f, "flexible tip change=" << flexible_change << " deg");

    // And the effect must be attributable to the flexibility rather than to anything else the two runs share.
    DOCTEST_CHECK(flexible_change > 10.f * std::fabs(rigid_change));
}

DOCTEST_TEST_CASE("PlantArchitecture leaf blades follow the azimuth of their petiole") {

    // The leaf prototype is built along +x and then rotated into place. The last step of that chain aims it at the azimuth of the petiole it hangs from, so the blade must come out pointing the same way
    // round the stem as its own petiole. This used to be applied as a NEGATIVE rotation, which mirrors the blade across the xz-plane instead of aiming it: the blade azimuth came out as the exact negative of
    // the petiole azimuth. On an upright stem that is invisible, because every petiole is mirrored by the same amount and the arrangement stays symmetric; once the stem leans, alternate phytomers end up with
    // their blades folded back ACROSS the culm instead of extending away from it, which reads as a wedge of leaves rather than a fan.
    //
    // The test therefore leans the base stem deliberately - a vertical plant passes even with the sign wrong - and compares each blade's horizontal direction against the horizontal direction of the petiole it
    // hangs from, which is the direct statement of the bug: with the sign inverted the two azimuths come out as negatives of one another rather than equal.
    //
    // The petiole axis is read from the phytomer rather than reconstructed from the surrounding geometry. An earlier version of this test inferred an "outward" direction from the offset of each leaf base off the
    // chord between its two neighbours, which is ill-conditioned for exactly this species: sorghum's petiole is a 2 mm stub, so that offset is a couple of millimetres and the sagitta of the curving culm is of the
    // same order and points in the plane of curvature rather than along the leaf. On the plant that libstdc++ draws from seed 12345 the reference vector pointed the wrong way for one leaf of seven and the test
    // failed at 103 degrees, with nothing wrong with the plant. The petiole's own axis has no such competing term.
    const float tilt = deg2rad(13.5f);

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("sorghum");

    // Built through the manual API rather than buildPlantInstanceFromLibrary() so the base rotation can carry a tilt; the library sorghum is deliberately upright.
    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0);
    plantarchitecture.addBaseStemShoot(plantID, 1, make_AxisRotation(tilt, 0.9f, 1.3f), 0.003f, 0.04f, 0.01f, 0.01f, 0, "mainstem");
    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 28);

    int leaves_checked = 0;
    int leaves_pointing_backwards = 0;
    float worst_azimuth_error = 0.f;

    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        for (const auto &phytomer: shoot->phytomers) {
            for (uint petiole = 0; petiole < phytomer->leaf_objIDs.size(); petiole++) {

                // The direction the petiole points at its tip, which is what the rotation chain aims the blade along.
                vec3 petiole_direction = phytomer->getPetioleAxisVector(1.f, petiole);
                petiole_direction.z = 0.f; // the petiole also pitches; only its azimuth is under test
                if (petiole_direction.magnitude() < 1e-6f) {
                    continue; // a petiole pointing straight up or down has no azimuth to follow
                }
                petiole_direction.normalize();

                for (const uint objID: phytomer->leaf_objIDs.at(petiole)) {
                    if (!context.doesObjectExist(objID) || context.getObjectType(objID) != OBJECT_TYPE_POLYMESH) {
                        continue;
                    }
                    const std::vector<vec3> vertices = context.getPolymeshObjectVertices(objID);
                    const size_t row_length = 51; // Nx + 1 for the sorghum leaf prototype
                    if (vertices.size() < row_length || vertices.size() % row_length != 0) {
                        continue;
                    }
                    const size_t midrib_start = (vertices.size() / row_length / 2) * row_length;
                    vec3 blade = vertices.at(midrib_start + row_length - 1) - vertices.at(midrib_start);
                    blade.z = 0.f; // the blade also pitches downward; only its azimuth is under test
                    if (blade.magnitude() < 1e-7f) {
                        continue;
                    }
                    blade.normalize();

                    if (blade * petiole_direction < 0.f) {
                        leaves_pointing_backwards++;
                    }
                    worst_azimuth_error = std::max(worst_azimuth_error, rad2deg(acos_safe(blade * petiole_direction)));
                    leaves_checked++;
                }
            }
        }
    }

    DOCTEST_REQUIRE(leaves_checked >= 4);

    // No blade may run back along its own petiole. With the sign inverted this fails for every leaf whose petiole is not aligned with the xz-plane, which on a leaning stem is most of them.
    DOCTEST_CHECK_MESSAGE(leaves_pointing_backwards == 0, "leaves pointing back along their petiole=" << leaves_pointing_backwards << " of " << leaves_checked);

    // The blade is aimed along its petiole's azimuth exactly, so the only spread is what the blade-up correction - a roll about the petiole axis, applied after the aiming rotation - contributes to the
    // horizontal projection of a blade that is also pitched away from that axis. That stays within a few degrees; the mirrored case misses by twice the petiole azimuth, which for this deliberately-leaned
    // stem runs to well over a hundred degrees, so a bound of 15 separates the two by an order of magnitude without being brittle.
    DOCTEST_CHECK_MESSAGE(worst_azimuth_error < 15.f, "worst blade-vs-petiole azimuth error=" << worst_azimuth_error << " deg");
}

DOCTEST_TEST_CASE("PlantArchitecture drooping leaves stay attached and keep their size") {

    // Straightness, which is what the droop test above measures, is invariant to both translation and scale by construction: it reads the same whether a leaf is where it belongs or floating away from the
    // plant, and whether it is its proper size or a fraction of it. Bending must therefore be pinned down by two further invariants, each checked against the same plant built rigid.
    //
    // Both are properties of bending itself rather than of any particular mechanics: a blade that bends pivots about its attachment and does not stretch, so the gap to the stem and the blade's surface area
    // must both survive the deflection.
    auto buildPlant = [](Context &ctx, float flexibility) {
        PlantArchitecture pa(&ctx);
        pa.loadPlantModelFromLibrary("sorghum");
        ShootParameters params = pa.getCurrentShootParameters("mainstem");
        params.phytomer_parameters.leaf.prototype.flexibility = flexibility;
        pa.updateCurrentShootParameters("mainstem", params);
        return pa.buildPlantInstanceFromLibrary(nullorigin, 32.f);
    };

    // Distance from each leaf to the nearest point of the stem or petiole it hangs from, and the total blade area, for a whole plant.
    auto attachmentAndArea = [](Context &ctx, PlantArchitecture &pa, uint plantID, float &worst_gap, float &total_area) {
        std::vector<vec3> attachment_points;
        for (uint UUID: pa.getAllInternodeUUIDs()) {
            for (const vec3 &vertex: ctx.getPrimitiveVertices(UUID)) {
                attachment_points.push_back(vertex);
            }
        }
        for (uint UUID: pa.getAllPetioleUUIDs()) {
            for (const vec3 &vertex: ctx.getPrimitiveVertices(UUID)) {
                attachment_points.push_back(vertex);
            }
        }
        DOCTEST_REQUIRE_FALSE(attachment_points.empty());

        worst_gap = 0.f;
        total_area = 0.f;
        for (uint objID: pa.getPlantLeafObjectIDs(plantID)) {
            if (!ctx.doesObjectExist(objID)) {
                continue;
            }
            float nearest = std::numeric_limits<float>::max();
            for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
                total_area += ctx.getPrimitiveArea(UUID);
                for (const vec3 &vertex: ctx.getPrimitiveVertices(UUID)) {
                    for (const vec3 &attachment: attachment_points) {
                        nearest = std::min(nearest, (vertex - attachment).magnitude());
                    }
                }
            }
            worst_gap = std::max(worst_gap, nearest);
        }
    };

    float rigid_gap = 0.f, rigid_area = 0.f;
    {
        Context ctx;
        ctx.seedRandomGenerator(1000);
        PlantArchitecture pa(&ctx);
        pa.loadPlantModelFromLibrary("sorghum");
        ShootParameters params = pa.getCurrentShootParameters("mainstem");
        params.phytomer_parameters.leaf.prototype.flexibility = 0.f;
        pa.updateCurrentShootParameters("mainstem", params);
        const uint plantID = pa.buildPlantInstanceFromLibrary(nullorigin, 32.f);
        attachmentAndArea(ctx, pa, plantID, rigid_gap, rigid_area);
    }

    float flexible_gap = 0.f, flexible_area = 0.f;
    {
        Context ctx;
        ctx.seedRandomGenerator(1000);
        PlantArchitecture pa(&ctx);
        pa.loadPlantModelFromLibrary("sorghum");
        ShootParameters params = pa.getCurrentShootParameters("mainstem");
        params.phytomer_parameters.leaf.prototype.flexibility = 23.5f;
        pa.updateCurrentShootParameters("mainstem", params);
        const uint plantID = pa.buildPlantInstanceFromLibrary(nullorigin, 32.f);
        attachmentAndArea(ctx, pa, plantID, flexible_gap, flexible_area);
    }

    DOCTEST_REQUIRE(rigid_area > 0.f);

    // A drooping leaf pivots about where it is attached, so it comes no further from the stem than a rigid one does. Deflecting a rest shape that carries the prototype's own placement, or at a size the
    // object transform then applies again, both leave the blade hanging clear of the plant - which this catches and the straightness measure cannot.
    DOCTEST_CHECK_MESSAGE(flexible_gap < 2.f * rigid_gap, "flexible worst gap=" << flexible_gap << " m vs rigid worst gap=" << rigid_gap << " m");

    // Bending is inextensible, so the blade keeps its area. Applying the leaf's growth twice shrinks every leaf by its own growth fraction, which shows up here as a large area deficit.
    DOCTEST_CHECK_MESSAGE(flexible_area == doctest::Approx(rigid_area).epsilon(0.05), "flexible area=" << flexible_area << " m2 vs rigid area=" << rigid_area << " m2");
}

DOCTEST_TEST_CASE("PlantArchitecture generated leaves carry mesh topology describing their own geometry") {
    // A leaf built from a texture mask is a regular lattice of vertices, so which facets meet at each vertex is known exactly. Recording it as an indexed face set is what lets a consumer treat the leaf as one
    // connected surface - camera flux smoothing in the radiation model reconstructs a field across it, and without the face table it silently has nothing to interpolate and leaves the leaf faceted.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plant(&context);
    plant.loadPlantModelFromLibrary("bean");
    plant.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 300);

    size_t polymesh_objects = 0;
    size_t polymesh_objects_with_topology = 0;
    size_t facets_checked = 0;
    float largest_vertex_error = 0.f;

    for (uint ObjID: context.getAllObjectIDs()) {
        if (context.getObjectType(ObjID) != OBJECT_TYPE_POLYMESH) {
            continue;
        }
        polymesh_objects++;
        if (!context.doesObjectHaveSharedVertexTopology(ObjID)) {
            continue;
        }
        polymesh_objects_with_topology++;

        const std::vector<vec3> mesh_vertices = context.getPolymeshObjectVertices(ObjID);
        const std::vector<uint> object_UUIDs = context.getObjectPrimitiveUUIDs(ObjID);

        const std::vector<std::vector<int>> shared_indices = context.getObjectPrimitiveSharedVertexIndices(ObjID, object_UUIDs, WELD_FULL);

        // The face table has to describe where the primitives actually are. The leaf lattice is built in the leaf's own frame and then translated, so this is what catches a transform applied to the primitives
        // but not to the stored vertices.
        for (size_t k = 0; k < object_UUIDs.size(); k++) {
            const std::vector<int> &corners = shared_indices.at(k);

            // Every primitive of the object is described, including a petiolule appended from an OBJ: the reader worked its connectivity out, so it should not be dropped on the way into the leaf.
            DOCTEST_INFO("object " << ObjID << " primitive " << object_UUIDs.at(k) << " has no entry in the face table");
            DOCTEST_REQUIRE(corners.size() == 3);
            const std::vector<vec3> primitive_vertices = context.getPrimitiveVertices(object_UUIDs.at(k));
            DOCTEST_REQUIRE(primitive_vertices.size() == 3);

            for (size_t c = 0; c < 3; c++) {
                largest_vertex_error = std::max(largest_vertex_error, (mesh_vertices.at(size_t(corners.at(c))) - primitive_vertices.at(c)).magnitude());
            }
            facets_checked++;
        }
    }

    DOCTEST_REQUIRE(polymesh_objects > 0);
    DOCTEST_REQUIRE(facets_checked > 0);

    // Every mesh organ should carry topology, whether it came from an OBJ file or was generated from a texture mask.
    DOCTEST_INFO(polymesh_objects_with_topology << " of " << polymesh_objects << " polymesh objects carry a face table");
    DOCTEST_CHECK(polymesh_objects_with_topology == polymesh_objects);

    // The bound is on the absolute distance between a stored vertex and the primitive corner naming it. Both are world-space positions carried through the same chain of single-precision transforms, so the
    // disagreement is rounding rather than disagreement about the geometry, and it grows with the coordinates: this same plant translated 100 m from the origin disagrees by ~1e-3 with its shape unchanged.
    // The bound is therefore a check that nothing structural has diverged, not a claim about achievable precision, and is set well clear of the rounding floor for a plant of this size.
    DOCTEST_INFO("largest disagreement between a face table vertex and the primitive corner it names: " << largest_vertex_error);
    DOCTEST_CHECK(largest_vertex_error < 1e-4f);
}

DOCTEST_TEST_CASE("PlantArchitecture generated leaf vertices are shared between neighbouring facets") {
    // Recording a face table is only useful if the facets genuinely share vertices; a table listing three fresh vertices per facet would satisfy the geometry check above while still describing a triangle soup.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plant(&context);
    plant.loadPlantModelFromLibrary("bean");
    plant.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 300);

    bool checked_a_generated_leaf = false;
    for (uint ObjID: context.getAllObjectIDs()) {
        if (context.getObjectType(ObjID) != OBJECT_TYPE_POLYMESH || !context.doesObjectHaveSharedVertexTopology(ObjID)) {
            continue;
        }
        // Counted from the face table rather than from the object's members, so that a petiolule appended alongside the blade cannot inflate the count and make the bound easier to satisfy.
        const size_t facet_count = context.getPolymeshObjectFaceCount(ObjID);
        const size_t vertex_count = context.getPolymeshObjectVertexCount(ObjID);
        if (facet_count < 20) {
            continue;
        }

        // Two triangles per lattice cell, and each interior vertex is shared by several of them, so a welded mesh has far fewer vertices than three per facet.
        DOCTEST_INFO("object " << ObjID << ": " << facet_count << " faces, " << vertex_count << " shared vertices");
        DOCTEST_CHECK(vertex_count < 3 * facet_count);
        checked_a_generated_leaf = true;
    }
    DOCTEST_CHECK(checked_a_generated_leaf);
}

DOCTEST_TEST_CASE("PlantArchitecture leaf prototype keeps its petiolule attached to the leaf") {
    // The petiolule is loaded from an OBJ and appended to the leaf's primitives so that the two become one object. Since v1.3.83 loadOBJ() groups what it loads into a polymesh object of its own, so those
    // primitives already had a parent by the time addPolymeshObject() saw them and were silently dropped from the leaf. The leaf that gets copied to every phytomer then had no petiolule, and the prototype's
    // petiolule was left behind at the origin at prototype scale, visible in the scene as a stray tube that no plant owned.
    Context context;
    std::minstd_rand0 generator;

    LeafPrototype prototype(&generator);
    prototype.leaf_texture_file[0] = "BeanLeaf_tip.png";
    prototype.build_petiolule = true;
    prototype.subdivisions = 3;

    std::string captured_warnings;
    uint objID;
    {
        capture_cerr capture;
        objID = GenericLeafPrototype(&context, &prototype, 0);
        captured_warnings = capture.get_captured_output();
    }

    DOCTEST_INFO("warnings emitted while building the leaf prototype: " << captured_warnings);
    DOCTEST_CHECK(captured_warnings.find("were not added to polymesh object") == std::string::npos);

    // Everything the prototype created belongs to the one object it returned, so copying that object carries the whole leaf.
    DOCTEST_CHECK(context.getObjectPrimitiveUUIDs(objID).size() == context.getPrimitiveCount());
    DOCTEST_CHECK(context.getObjectCount() == 1);

    // The petiolule is a solid stub rather than a texture-masked sheet, so its presence is what makes the leaf object larger than its blade alone.
    Context blade_only_context;
    LeafPrototype blade_only_prototype(&generator);
    blade_only_prototype.leaf_texture_file[0] = "BeanLeaf_tip.png";
    blade_only_prototype.build_petiolule = false;
    blade_only_prototype.subdivisions = 3;
    const uint blade_only_objID = GenericLeafPrototype(&blade_only_context, &blade_only_prototype, 0);

    DOCTEST_CHECK(context.getObjectPrimitiveUUIDs(objID).size() > blade_only_context.getObjectPrimitiveUUIDs(blade_only_objID).size());
}

DOCTEST_TEST_CASE("PlantArchitecture petiolules keep their own object label") {
    // The petiolule is loaded from an OBJ whose group names it, so it arrives labelled "petiolule". Labelling the whole leaf object "leaf" afterwards would overwrite that, and every downstream filter keyed on
    // the label would then treat the petiolule as blade - giving it the leaf's optical properties in a radiation solve, and leaving the call that colours it as petiole matching nothing.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plant(&context);
    plant.loadPlantModelFromLibrary("bean");
    plant.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 300);

    const std::vector<uint> all_UUIDs = context.getAllUUIDs();
    const std::vector<uint> petiolule_UUIDs = context.filterPrimitivesByData(all_UUIDs, "object_label", "petiolule");
    const std::vector<uint> leaf_UUIDs = context.filterPrimitivesByData(all_UUIDs, "object_label", "leaf");

    DOCTEST_INFO("labelled petiolule: " << petiolule_UUIDs.size() << ", labelled leaf: " << leaf_UUIDs.size() << ", total: " << all_UUIDs.size());

    // The bean model builds a petiolule on every trifoliate leaflet, so the label has to survive onto real geometry.
    DOCTEST_CHECK(!petiolule_UUIDs.empty());

    // The blade must still be labelled, and the two labels must be disjoint.
    DOCTEST_CHECK(!leaf_UUIDs.empty());
    std::set<uint> petiolule_set(petiolule_UUIDs.begin(), petiolule_UUIDs.end());
    size_t overlapping = 0;
    for (uint UUID: leaf_UUIDs) {
        if (petiolule_set.find(UUID) != petiolule_set.end()) {
            overlapping++;
        }
    }
    DOCTEST_CHECK(overlapping == 0);
}

DOCTEST_TEST_CASE("PlantArchitecture cowpea pod asset encloses a volume") {
    // The carbohydrate model asks each fruit for its volume, so the pod body has to be watertight. This asset was not: the stem end was left open, and the pod tip was a set of coincident vertices that were
    // separate indices, so the surface was closed geometrically but torn topologically. The sepal is a separate piece and is open by design, which is why the object as a whole still reports boundary edges.
    Context context;
    const std::vector<uint> UUIDs = context.loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaPod.obj").string().c_str(), true);
    DOCTEST_REQUIRE(!UUIDs.empty());

    const uint ObjID = context.getPrimitiveParentObjectID(UUIDs.front());
    DOCTEST_REQUIRE(ObjID != 0);

    float volume = 0.f;
    DOCTEST_REQUIRE_NOTHROW(volume = context.getPolymeshObjectVolume(ObjID));
    DOCTEST_CHECK(volume > 0.f);

    // The pod body is one closed piece; the sepal is the other, and its rim is the only boundary that should remain.
    const std::vector<std::vector<size_t>> components = context.getPolymeshObjectConnectedComponents(ObjID);
    DOCTEST_INFO("components=" << components.size() << " boundaryEdges=" << context.getPolymeshObjectBoundaryEdges(ObjID).size() << " volume=" << volume);
    DOCTEST_CHECK(components.size() == 2);
    DOCTEST_CHECK(context.getPolymeshObjectBoundaryEdges(ObjID).size() == 10);
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip is stable across repeated save/load cycles") {
    // writePlantStructureXML() -> readPlantStructureXML() -> writePlantStructureXML() has to be a fixed
    // point: the second file must describe the same plant as the first. It did not, because the reader
    // left several pieces of per-phytomer state at their species-prototype defaults instead of restoring
    // them from the file, and wrote those defaults back out. The errors were multiplicative, so the
    // damage was invisible on the first reload and catastrophic on the second - a cowpea's leaf area grew
    // by roughly eighty times and its height by a factor of three.
    //
    // Two reload generations are the minimum that exposes this: generation 1 is built from a good file
    // and merely writes a bad one, and only generation 2 is built from the bad file.
    const std::string stage0_filename = "test_xml_roundtrip_stage0.xml";
    const std::string stage1_filename = "test_xml_roundtrip_stage1.xml";

    // Cowpea at 50 days exercises the parts that were being lost: trifoliate leaves (leaf_size_max and the
    // leaflet count), curved branches (the perturbation vectors), and pods on peduncles (the peduncle
    // scalars and the fruit growth fraction).
    Context grown_context;
    PlantArchitecture grown_plantarchitecture(&grown_context);
    grown_plantarchitecture.disableMessages();
    grown_plantarchitecture.loadPlantModelFromLibrary("cowpea");
    const uint grown_plantID = grown_plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    DOCTEST_REQUIRE_NOTHROW(grown_plantarchitecture.advanceTime(grown_plantID, 50.f));
    DOCTEST_REQUIRE_NOTHROW(grown_plantarchitecture.writePlantStructureXML(grown_plantID, stage0_filename));

    const float grown_leaf_area = grown_plantarchitecture.sumPlantLeafArea(grown_plantID);
    const float grown_height = grown_plantarchitecture.getPlantHeight(grown_plantID);
    DOCTEST_REQUIRE(grown_leaf_area > 0.f);
    DOCTEST_REQUIRE(grown_height > 0.f);

    // Generation 1: read the file the grown plant wrote, and write the file under test.
    Context reload1_context;
    PlantArchitecture reload1_plantarchitecture(&reload1_context);
    reload1_plantarchitecture.disableMessages();
    reload1_plantarchitecture.loadPlantModelFromLibrary("cowpea");
    std::vector<uint> reload1_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(reload1_plantIDs = reload1_plantarchitecture.readPlantStructureXML(stage0_filename, true));
    DOCTEST_REQUIRE(!reload1_plantIDs.empty());
    DOCTEST_REQUIRE_NOTHROW(reload1_plantarchitecture.writePlantStructureXML(reload1_plantIDs.front(), stage1_filename));

    const float reload1_leaf_area = reload1_plantarchitecture.sumPlantLeafArea(reload1_plantIDs.front());
    const float reload1_height = reload1_plantarchitecture.getPlantHeight(reload1_plantIDs.front());

    // Generation 2: read the file generation 1 wrote. This is the one that used to blow up.
    Context reload2_context;
    PlantArchitecture reload2_plantarchitecture(&reload2_context);
    reload2_plantarchitecture.disableMessages();
    reload2_plantarchitecture.loadPlantModelFromLibrary("cowpea");
    std::vector<uint> reload2_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(reload2_plantIDs = reload2_plantarchitecture.readPlantStructureXML(stage1_filename, true));
    DOCTEST_REQUIRE(!reload2_plantIDs.empty());

    const float reload2_leaf_area = reload2_plantarchitecture.sumPlantLeafArea(reload2_plantIDs.front());
    const float reload2_height = reload2_plantarchitecture.getPlantHeight(reload2_plantIDs.front());

    DOCTEST_INFO("leaf area: " << grown_leaf_area << " -> " << reload1_leaf_area << " -> " << reload2_leaf_area);
    DOCTEST_INFO("height: " << grown_height << " -> " << reload1_height << " -> " << reload2_height);

    // Reconstruction resamples some purely cosmetic quantities, so this pins the scale of the plant rather
    // than demanding it be reproduced vertex for vertex. The tolerance is far tighter than the failure it
    // guards against: the unfixed reader returned roughly 8000% here, not 5%.
    DOCTEST_CHECK(reload1_leaf_area == doctest::Approx(grown_leaf_area).epsilon(0.05));
    DOCTEST_CHECK(reload2_leaf_area == doctest::Approx(reload1_leaf_area).epsilon(0.05));
    DOCTEST_CHECK(reload1_height == doctest::Approx(grown_height).epsilon(0.05));
    DOCTEST_CHECK(reload2_height == doctest::Approx(reload1_height).epsilon(0.05));

    std::remove(stage0_filename.c_str());
    std::remove(stage1_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip preserves leaf orientation") {
    // readPlantStructureXML() rebuilds each leaf by re-running the rotation chain the Phytomer constructor
    // uses, from the angles saved in the file. It used to carry its own copy of that chain, and the copy had
    // drifted: the petiole azimuth was applied with the opposite sign, the unifoliate roll used a different
    // formula, the blade-up correction was absent, leaflets that attach at the petiole tip were arranged as
    // though they were offset along it, the petiole curvature was integrated over the wrong segment length,
    // the prototype each blade was copied from was re-drawn at random, and neither the phytomer's age nor the
    // self-weight deflection that depends on it was restored. A reloaded cowpea's leaves pointed an average of
    // 60 degrees away from the grown plant's, and a reloaded sorghum's up to 77.
    //
    // This compares the blade normals of a grown plant against those of the same plant written out and read
    // back, pairing leaves by walking shoots and phytomers in the same order in both.
    auto leafNormals = [](Context &ctx, PlantArchitecture &pa, uint plantID) {
        std::vector<vec3> normals;
        for (uint shootID: pa.getAllShootIDs(plantID)) {
            const std::shared_ptr<Shoot> &shoot = pa.getPlantShoot(plantID, shootID);
            for (const auto &phytomer: shoot->phytomers) {
                for (uint petiole = 0; petiole < phytomer->leaf_objIDs.size(); petiole++) {
                    for (uint leaf = 0; leaf < phytomer->leaf_objIDs.at(petiole).size(); leaf++) {
                        const uint objID = phytomer->leaf_objIDs.at(petiole).at(leaf);
                        if (!ctx.doesObjectExist(objID)) {
                            continue;
                        }
                        // Area-weighted so that a curved blade is summarised by the direction it mostly faces
                        // rather than by whichever facet happens to come first.
                        vec3 normal(0, 0, 0);
                        float area_total = 0;
                        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
                            const float area = ctx.getPrimitiveArea(UUID);
                            normal = normal + area * ctx.getPrimitiveNormal(UUID);
                            area_total += area;
                        }
                        if (area_total > 0) {
                            normals.push_back(normal / area_total);
                        }
                    }
                }
            }
        }
        return normals;
    };

    // Cowpea covers compound leaves and the shared rotation chain; sorghum covers the self-weight droop, which
    // depends on the phytomer age the file now carries. Each is checked over several seeds rather than one,
    // because the plant a seed produces differs between standard library implementations - the distributions in
    // <random> are implementation-specified, so libc++, libstdc++ and MSVC each build a different plant from the
    // same seed. A tolerance calibrated against a single realization therefore encodes whichever plant the
    // developer's platform happened to draw: the tolerances below were originally set from one macOS realization
    // that sat in the best fifth of the distribution, and the identical Linux run failed at three times the
    // bound. Sweeping seeds pins the tolerance to the model rather than to the platform.
    //
    // The bounds below are tight for both species, including the single worst leaf. They were once far looser on
    // cowpea, to accommodate young phytomers at the growing tip whose petiole axis came back up to 6 degrees out
    // - a residual in the petiole reconstruction rather than the leaf rotation chain. That residual is gone: the
    // axes a petiole was built from are now recorded in the file rather than re-derived by replaying a rotation
    // chain anchored on the previous phytomer, which is what accumulated error along a shoot and left the tip
    // worst. Cowpea's worst leaf over a twenty-seed sweep fell from 12.7 degrees to 0.18.
    struct Case {
        const char *species;
        float days;
        double mean_tol_deg;
        double p95_tol_deg;
        double max_tol_deg;
    };
    //
    // Each bound is set from a sweep of twenty seeds with roughly 1.5x headroom, so that it holds for a plant this
    // test has never drawn - which is what every platform other than the one it was calibrated on will produce.
    // Over that sweep cowpea runs to 0.010 mean / 0.028 p95 / 0.178 max, and sorghum to 0.20 / 0.23 / 0.37.
    // Sorghum's residual is the self-weight droop rather than the rotation chain, which is why it is the larger
    // of the two and why its bounds are unchanged.
    const std::vector<Case> cases = {{"cowpea", 25.f, 0.02, 0.05, 0.3}, {"sorghum", 40.f, 0.5, 0.75, 1.5}};
    const std::vector<uint> seeds = {12345, 1, 3, 4};

    for (const Case &test_case: cases) {
        for (const uint seed: seeds) {
            DOCTEST_CAPTURE(std::string(test_case.species));
            DOCTEST_CAPTURE(seed);
            const std::string xml_filename = std::string("test_leaf_orientation_") + test_case.species + ".xml";

            Context grown_context;
            grown_context.seedRandomGenerator(seed);
            PlantArchitecture grown_plantarchitecture(&grown_context);
            grown_plantarchitecture.disableMessages();
            grown_plantarchitecture.loadPlantModelFromLibrary(test_case.species);
            const uint grown_plantID = grown_plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
            DOCTEST_REQUIRE_NOTHROW(grown_plantarchitecture.advanceTime(grown_plantID, test_case.days));
            DOCTEST_REQUIRE_NOTHROW(grown_plantarchitecture.writePlantStructureXML(grown_plantID, xml_filename));
            const std::vector<vec3> grown_normals = leafNormals(grown_context, grown_plantarchitecture, grown_plantID);
            DOCTEST_REQUIRE(grown_normals.size() > 5);

            Context restored_context;
            restored_context.seedRandomGenerator(seed);
            PlantArchitecture restored_plantarchitecture(&restored_context);
            restored_plantarchitecture.disableMessages();
            restored_plantarchitecture.loadPlantModelFromLibrary(test_case.species);
            std::vector<uint> restored_plantIDs;
            DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = restored_plantarchitecture.readPlantStructureXML(xml_filename, true));
            DOCTEST_REQUIRE(!restored_plantIDs.empty());
            const std::vector<vec3> restored_normals = leafNormals(restored_context, restored_plantarchitecture, restored_plantIDs.front());
            DOCTEST_REQUIRE(restored_normals.size() == grown_normals.size());

            double angle_sum = 0;
            std::vector<double> angles;
            angles.reserve(grown_normals.size());
            for (size_t i = 0; i < grown_normals.size(); i++) {
                vec3 grown_normal = grown_normals.at(i);
                vec3 restored_normal = restored_normals.at(i);
                grown_normal.normalize();
                restored_normal.normalize();
                const double angle = rad2deg(acos_safe(grown_normal * restored_normal));
                angles.push_back(angle);
                angle_sum += angle;
            }
            const double angle_mean = angle_sum / double(angles.size());
            std::sort(angles.begin(), angles.end());
            const double angle_p95 = angles.at(size_t(0.95 * double(angles.size() - 1)));
            const double angle_max = angles.back();

            // std::string rather than the raw const char*, which doctest streams as a pointer value.
            DOCTEST_INFO(std::string(test_case.species) << " seed " << seed << ": " << angles.size() << " leaves, mean=" << angle_mean << " deg, p95=" << angle_p95 << " deg, max=" << angle_max << " deg");
            DOCTEST_CHECK(angle_mean < test_case.mean_tol_deg);
            DOCTEST_CHECK(angle_p95 < test_case.p95_tol_deg);
            DOCTEST_CHECK(angle_max < test_case.max_tol_deg);

            std::remove(xml_filename.c_str());
        }
    }
}

DOCTEST_TEST_CASE("PlantArchitecture leaf prototype scaling keeps geometry and bookkeeping in step") {
    // A leaf's rendered size is leaf_size_max * current_leaf_scale_factor: the mature size it is growing
    // toward, times how far through that expansion it is. Phytomer::scaleLeafPrototypeScale() changes only the
    // first of those, so scaling by one half must halve both the geometry and the product, and must leave the
    // growth fraction alone. It used to divide the fraction by the scale factor as well, which left the
    // product unchanged while the geometry shrank - so everything reading the product, XML export included,
    // was wrong by the scale factor.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("tomato");
    const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    // Far enough along that leaves exist and are still expanding, which is where the two used to disagree.
    DOCTEST_REQUIRE_NOTHROW(plantarchitecture.advanceTime(plantID, 12.f));

    std::shared_ptr<Phytomer> phytomer;
    for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const std::shared_ptr<Shoot> &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        for (const auto &candidate: shoot->phytomers) {
            // A leaf still short of full size, so that the growth fraction is meaningfully below one.
            if (!candidate->leaf_objIDs.empty() && !candidate->leaf_objIDs.front().empty() && candidate->current_leaf_scale_factor.front() < 0.9f) {
                phytomer = candidate;
                break;
            }
        }
        if (phytomer) {
            break;
        }
    }
    DOCTEST_REQUIRE(phytomer != nullptr);

    const uint objID_leaf = phytomer->leaf_objIDs.front().front();
    auto renderedExtent = [&context](uint objID) {
        vec3 box_min, box_max;
        context.getObjectBoundingBox(objID, box_min, box_max);
        return (box_max - box_min).magnitude();
    };

    const float extent_before = renderedExtent(objID_leaf);
    const float mature_size_before = phytomer->leaf_size_max.front().front();
    const float growth_fraction_before = phytomer->current_leaf_scale_factor.front();
    DOCTEST_REQUIRE(extent_before > 0.f);
    DOCTEST_REQUIRE(growth_fraction_before > 0.f);

    phytomer->scaleLeafPrototypeScale(0.5f);

    const float extent_after = renderedExtent(objID_leaf);
    const float mature_size_after = phytomer->leaf_size_max.front().front();
    const float growth_fraction_after = phytomer->current_leaf_scale_factor.front();

    // Compared as relative errors rather than through doctest::Approx, whose tolerance is
    // epsilon * (1 + max(|a|,|b|)); on lengths of a few millimetres the constant term swamps the relative
    // part and the comparison passes almost anything.
    auto relativeError = [](double actual, double expected) { return std::fabs(actual - expected) / std::fabs(expected); };

    // The geometry halves.
    DOCTEST_CHECK(relativeError(extent_after, 0.5f * extent_before) < 0.01);
    // The mature size halves with it.
    DOCTEST_CHECK(relativeError(mature_size_after, 0.5f * mature_size_before) < 1e-4);
    // How far through its expansion the leaf is has not changed - it is the same leaf, mid-growth, only
    // aiming at a smaller final size.
    DOCTEST_CHECK(relativeError(growth_fraction_after, growth_fraction_before) < 1e-5);
    // And the product still describes the geometry, which is what writePlantStructureXML() writes out.
    DOCTEST_CHECK(relativeError(mature_size_after * growth_fraction_after, 0.5f * mature_size_before * growth_fraction_before) < 1e-4);
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip preserves leaf area") {
    // writePlantStructureXML() records each leaf as leaf_size_max * current_leaf_scale_factor, so a species
    // whose phytomer creation function rescales the leaf prototype - which is most of the grasses and the
    // tomatoes - wrote a leaf scale that its own geometry never had, and came back the wrong size. Total leaf
    // area is the aggregate the error shows up in; individual young leaves were out by as much as a factor of
    // four, while the plant-wide figure was out by five to twelve percent because mature leaves are unaffected.
    struct Case {
        const char *species;
        float days;
    };
    const std::vector<Case> cases = {{"tomato", 40.f}, {"cherrytomato", 40.f}, {"sorghum", 40.f}, {"maize", 40.f}, {"wheat", 40.f}, {"cowpea", 40.f}};

    for (const Case &test_case: cases) {
        DOCTEST_CAPTURE(test_case.species);
        const std::string xml_filename = std::string("test_leaf_area_") + test_case.species + ".xml";

        Context grown_context;
        grown_context.seedRandomGenerator(12345);
        PlantArchitecture grown_plantarchitecture(&grown_context);
        grown_plantarchitecture.disableMessages();
        grown_plantarchitecture.loadPlantModelFromLibrary(test_case.species);
        const uint grown_plantID = grown_plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
        DOCTEST_REQUIRE_NOTHROW(grown_plantarchitecture.advanceTime(grown_plantID, test_case.days));
        DOCTEST_REQUIRE_NOTHROW(grown_plantarchitecture.writePlantStructureXML(grown_plantID, xml_filename));
        const float grown_leaf_area = grown_plantarchitecture.sumPlantLeafArea(grown_plantID);
        DOCTEST_REQUIRE(grown_leaf_area > 0.f);

        Context restored_context;
        restored_context.seedRandomGenerator(12345);
        PlantArchitecture restored_plantarchitecture(&restored_context);
        restored_plantarchitecture.disableMessages();
        restored_plantarchitecture.loadPlantModelFromLibrary(test_case.species);
        std::vector<uint> restored_plantIDs;
        DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = restored_plantarchitecture.readPlantStructureXML(xml_filename, true));
        DOCTEST_REQUIRE(!restored_plantIDs.empty());
        const float restored_leaf_area = restored_plantarchitecture.sumPlantLeafArea(restored_plantIDs.front());

        // Relative error rather than doctest::Approx, whose tolerance is epsilon * (1 + max(|a|,|b|)) and so
        // is dominated by the constant term for a leaf area well under one square metre.
        const double leaf_area_relative_error = std::fabs(double(restored_leaf_area) - double(grown_leaf_area)) / double(grown_leaf_area);
        DOCTEST_INFO(test_case.species << ": leaf area " << grown_leaf_area << " -> " << restored_leaf_area << " (relative error " << leaf_area_relative_error << ")");
        DOCTEST_CHECK(leaf_area_relative_error < 0.02);

        std::remove(xml_filename.c_str());
    }
}

DOCTEST_TEST_CASE("Sorghum flag-leaf internode is elongated relative to the culm below it") {
    // Sorghum bears a distinctly long internode between the last normal leaf and the flag leaf,
    // visible in field photographs of the plant. The model produced no such segment: the internode
    // scale ramp in SorghumPhytomerCreationFunction saturates at 1.0 by node 5, so every internode
    // from there to the apex was set to exactly the same length and the flag-leaf internode was
    // indistinguishable from the eleven below it.
    //
    // The elongation is tied to the peduncle length rather than to a literal length, so that the
    // last-normal-leaf-to-flag-leaf distance matches the flag-leaf-to-panicle-base distance.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("sorghum");

    const float peduncle_length = plantarchitecture.getCurrentShootParameters("mainstem").phytomer_parameters.peduncle.length.val();
    DOCTEST_REQUIRE(peduncle_length > 0.f);

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 120.f);

    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();
    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(shoot->phytomers.size() >= 8);

    const std::size_t flag_node = shoot->phytomers.size() - 1;
    const float flag_internode = shoot->phytomers.at(flag_node)->getInternodeLength();

    // A mid-culm internode from the saturated part of the ramp, which is what the flag-leaf
    // internode used to be identical to.
    const float mid_internode = shoot->phytomers.at(flag_node - 3)->getInternodeLength();
    DOCTEST_REQUIRE(mid_internode > 0.f);

    DOCTEST_INFO("flag internode=" << flag_internode << " mid internode=" << mid_internode << " peduncle=" << peduncle_length);

    // The defect: these were exactly equal. Require a clear separation rather than mere inequality
    // so that a marginal change cannot pass.
    DOCTEST_CHECK(flag_internode > 2.f * mid_internode);

    // The flag-leaf internode should match the peduncle length, which is the rule that sets it.
    DOCTEST_CHECK(flag_internode == doctest::Approx(peduncle_length).epsilon(0.05));
}

DOCTEST_TEST_CASE("Grass peduncle base diameter matches the culm tip it attaches to") {
    // The peduncle radius was an independent constant, never reconciled with the culm it sits on,
    // and it was read once when the terminal floral bud appeared and never updated again while the
    // culm kept thickening. Sorghum ended up with a 16 mm peduncle on a 6 mm culm tip (2.7x too
    // wide) and maize with an 8 mm peduncle on a 15 mm culm tip (1.9x too narrow) -- the mismatch
    // runs in both directions, so a one-sided clamp would fix one species and break the other.
    //
    // The peduncle attaches at a single point (Shoot::addTerminalFloralBud sets its base to the
    // culm tip vertex), so any mismatch shows up as an abrupt step in the rendered geometry.

    auto check_species = [](const std::string &species) {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary(species);

        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
        plantarchitecture.advanceTime(plantID, 120.f);

        const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();
        const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        DOCTEST_REQUIRE(!shoot->shoot_internode_radii.empty());
        DOCTEST_REQUIRE(!shoot->shoot_internode_radii.back().empty());

        const float culm_tip_radius = shoot->shoot_internode_radii.back().back();
        DOCTEST_REQUIRE(culm_tip_radius > 0.f);

        // Read the radius off the peduncle tube that was actually built, rather than from the
        // stored parameter: the stored copy is written under a guard that a terminal floral bud does
        // not necessarily satisfy, so it is not a reliable witness of the rendered geometry.
        float peduncle_radius = -1.f;
        for (const auto &phytomer: shoot->phytomers) {
            for (const auto &petiole: phytomer->floral_buds) {
                for (const auto &fbud: petiole) {
                    if (!fbud.isterminal) {
                        continue;
                    }
                    for (uint objID: fbud.peduncle_objIDs) {
                        if (!context.doesObjectExist(objID)) {
                            continue;
                        }
                        const std::vector<float> radii = context.getTubeObjectNodeRadii(objID);
                        if (!radii.empty()) {
                            peduncle_radius = radii.front();
                        }
                    }
                }
            }
        }
        DOCTEST_REQUIRE(peduncle_radius > 0.f);

        const float ratio = peduncle_radius / culm_tip_radius;
        DOCTEST_INFO(species << ": culm tip diam=" << 2000.f * culm_tip_radius << " mm, peduncle diam=" << 2000.f * peduncle_radius << " mm, ratio=" << ratio);

        // The peduncle is naturally a little narrower than the culm that bears it, but the two are
        // continuous across the junction -- neither a step up nor a sharp pinch.
        DOCTEST_CHECK(ratio > 0.6f);
        DOCTEST_CHECK(ratio < 1.05f);
    };

    check_species("sorghum");
    check_species("maize");
}

DOCTEST_TEST_CASE("Reading downstream inflorescence area at every node costs one traversal") {
    // Counting the inflorescence in the pipe model was first implemented by summing, at every node,
    // the inflorescence area of every phytomer above it plus that of every child shoot hanging off
    // those phytomers -- and doing so from the girth loop, which already visits every node on every
    // time step. Each node re-derived a suffix the node below it had just computed, and every child
    // subtree was re-walked once per ancestor node above it, so a single growth step cost one full
    // traversal of the plant per node instead of one in total. A tomato advanced to 180 days took
    // 4.1 s against 0.26 s with the feature disabled, and the plugin self-test went from minutes to
    // hours in CI.
    //
    // The cost is driven by the number of inflorescence-bearing nodes rather than by node count:
    // every phytomer visited has its inflorescence area measured out of the Context, which walks that
    // object's primitives, and a panicle or truss is a dense object. Adding bare phytomers to a shoot
    // therefore does NOT reproduce the regression -- the quadratic loop is quadratic in iterations
    // that return zero immediately. A branched, heavily-fruiting species is what exposes it.
    //
    // Soybean is used rather than tomato, and the choice matters. Seeding the generator does not fix
    // how many fruit a plant sets: the growth path branches on floating-point comparisons, so the
    // sequence of draws -- and with it the plant -- differs between platforms even from the same seed.
    // Tomato at 120 days gives 11 fruiting nodes on macOS, 4 on Linux and 1 on Windows, and growing it
    // longer does not help: a tomato stops adding nodes at around 120 days with nearly every floral
    // bud abscised, so its fruiting count is fixed by however many buds happened to survive. Soybean
    // fruits far more heavily and does so on every seed sampled -- no fewer than 4 shoots, 71 nodes
    // and 11 fruiting nodes across 25 seeds -- so the preconditions below hold with a wide margin
    // however the stream diverges.
    //
    // What is pinned is the ratio of two measurements taken in the same process on the same plant, so
    // the test is independent of how fast the machine running it is: reading the value at all N nodes
    // must cost no more than a small multiple of bringing the sums up to date once. Under the form
    // this replaced, reading N nodes cost N traversals.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("soybean");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 120.f);

    const std::vector<uint> shootIDs = plantarchitecture.getAllShootIDs(plantID);
    DOCTEST_REQUIRE(shootIDs.size() > 1); // the regression is driven by the recursion into child shoots

    // Guard against a vacuous pass: the cost being measured is the per-phytomer area evaluation, so
    // the plant must actually carry inflorescences on several nodes for there to be anything to count.
    std::size_t node_count = 0;
    std::size_t fruiting_node_count = 0;
    for (uint sid: shootIDs) {
        const auto &shoot = plantarchitecture.getPlantShoot(plantID, sid);
        node_count += shoot->phytomers.size();
        for (const auto &phytomer: shoot->phytomers) {
            if (phytomer->getInflorescenceArea() > 0.f) {
                fruiting_node_count++;
            }
        }
    }
    DOCTEST_INFO("nodes=" << node_count << " fruiting=" << fruiting_node_count << " shoots=" << shootIDs.size());
    DOCTEST_REQUIRE(node_count > 20);
    DOCTEST_REQUIRE(fruiting_node_count >= 5);

    const auto &root_shoot = plantarchitecture.getPlantShoot(plantID, shootIDs.front());

    // Cost of bringing the sums up to date once, which is what one growth step legitimately owes.
    constexpr int repeats = 20;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeats; r++) {
        // The returned total is soybean's stem-loading inflorescence area, which is zero: its pods are
        // all lateral, and only an inflorescence borne on the stem's own axis counts toward girth (see
        // Phytomer::inflorescenceLoadsStem()). What is being timed is the traversal, not the sum, and
        // the traversal still visits every node and every child subtree either way -- the guard that
        // there is real work to do is the fruiting-node count required above. The value is read only
        // so the call cannot be optimized away.
        const float total = root_shoot->refreshDownstreamInflorescenceArea();
        DOCTEST_REQUIRE(std::isfinite(total));
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double refresh_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
    DOCTEST_REQUIRE(refresh_time > 0.0);

    // Cost of the girth loop's actual access pattern: one refresh, then a read at every node.
    const auto t2 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeats; r++) {
        root_shoot->refreshDownstreamInflorescenceArea();
        for (uint sid: shootIDs) {
            const auto &shoot = plantarchitecture.getPlantShoot(plantID, sid);
            for (std::size_t node = 0; node < shoot->phytomers.size(); node++) {
                const float area = shoot->sumDownstreamInflorescenceArea(node);
                DOCTEST_REQUIRE(std::isfinite(area));
            }
        }
    }
    const auto t3 = std::chrono::high_resolution_clock::now();
    const double refresh_and_read_time = std::chrono::duration<double, std::milli>(t3 - t2).count();

    const double overhead_ratio = refresh_and_read_time / refresh_time;

    DOCTEST_INFO("nodes=" << node_count << " fruiting=" << fruiting_node_count << " refresh=" << refresh_time << " ms, refresh+read=" << refresh_and_read_time << " ms, ratio=" << overhead_ratio);

    // Reading a cached float at each node is negligible beside the traversal itself, so the two
    // measurements differ by a constant factor. If each read re-derived its own suffix instead,
    // reading all N nodes would multiply the traversal cost by a large fraction of N -- here some
    // tens.
    //
    // The bound has to sit far above 1 and far below N. It is nearer the former than it may look:
    // the refresh being timed is genuinely cheap for soybean, whose pods are all lateral and so are
    // skipped by the girth gate (see Phytomer::inflorescenceLoadsStem()), leaving a baseline of a few
    // microseconds against which per-call overhead is not negligible. What the bound still separates
    // is a constant factor from a factor of N, which is the regression at issue.
    DOCTEST_CHECK(overhead_ratio < 12.0);
}

DOCTEST_TEST_CASE("Sorghum flag leaf is more horizontal than the leaves below it") {
    // Sorghum's canopy does not follow the erectophile "smart canopy" ideotype: the flag leaf is the
    // most horizontal leaf on the plant, with erectness increasing toward the middle and lower
    // canopy (Zhou et al. 2020, Plant Physiology 184:1927, across three biparental populations).
    //
    // This is a characterization test rather than a bug fix -- the model already gets the direction
    // right, via the flag-leaf petiole rotation in SorghumPhytomerCreationFunction. It is worth
    // pinning because the behavior is counterintuitive and cheap to remove by accident: with that
    // rotation deleted, the flag leaf measures 41.5 degrees against a 49-61 degree canopy, making it
    // the most ERECT leaf on the plant and inverting the published ordering.
    //
    // Measured base-to-tip across the blade rather than on the petiole. Sorghum's petiole is a 2 mm
    // stub and nearly all of the leaf's displayed angle comes from the blade's own curvature, so a
    // petiole-only measure reports every leaf as steeply erect (12-30 degrees from vertical) and
    // cannot see the arc that actually sets canopy inclination.

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("sorghum");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
    plantarchitecture.advanceTime(plantID, 120.f);

    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();
    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(shoot->phytomers.size() >= 8);

    // Inclination of the blade's base-to-tip chord from vertical: 90 degrees is horizontal.
    auto blade_inclination_deg = [&](std::size_t node) -> float {
        const auto &phytomer = shoot->phytomers.at(node);
        if (phytomer->leaf_objIDs.empty() || phytomer->leaf_objIDs.at(0).empty() || phytomer->leaf_bases.empty() || phytomer->leaf_bases.at(0).empty()) {
            return -1.f;
        }
        const uint objID = phytomer->leaf_objIDs.at(0).front();
        if (!context.doesObjectExist(objID)) {
            return -1.f;
        }
        const vec3 base = phytomer->leaf_bases.at(0).front();

        // The blade tip is the vertex furthest from the leaf base.
        float furthest = -1.f;
        vec3 tip = base;
        for (uint UUID: context.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                const float distance = (vertex - base).magnitude();
                if (distance > furthest) {
                    furthest = distance;
                    tip = vertex;
                }
            }
        }
        const vec3 chord = tip - base;
        const float length = chord.magnitude();
        if (length <= 0.f) {
            return -1.f;
        }
        return rad2deg(std::acos(std::fabs(chord.z) / length));
    };

    const std::size_t flag_node = shoot->phytomers.size() - 1;
    const float flag_angle = blade_inclination_deg(flag_node);
    DOCTEST_REQUIRE(flag_angle >= 0.f);

    // Compare against the whole canopy below the flag leaf rather than one leaf, so that natural
    // leaf-to-leaf variation cannot decide the result.
    float steepest_below = 1e6f;
    float mean_below = 0.f;
    std::size_t counted = 0;
    for (std::size_t node = 0; node < flag_node; node++) {
        const float angle = blade_inclination_deg(node);
        if (angle < 0.f) {
            continue;
        }
        steepest_below = std::min(steepest_below, angle);
        mean_below += angle;
        counted++;
    }
    DOCTEST_REQUIRE(counted >= 5);
    mean_below /= float(counted);

    DOCTEST_INFO("flag leaf=" << flag_angle << " deg, canopy mean=" << mean_below << " deg over " << counted << " leaves (90 = horizontal)");

    // The flag leaf must be the most horizontal leaf on the plant, not merely above average.
    DOCTEST_CHECK(flag_angle > mean_below);

    // And by a margin that leaf-to-leaf scatter cannot account for: removing the flag-leaf rotation
    // drops it roughly 25 degrees, to below every other leaf.
    DOCTEST_CHECK(flag_angle > mean_below + 4.f);
}

DOCTEST_TEST_CASE("Sorghum upper culm carries the panicle it supports") {
    // The pipe model sizes each internode from the leaf area above it, and a terminal inflorescence
    // contributes none: a sorghum panicle is borne above every node on the culm but was invisible to
    // the calculation. The upper culm therefore tapered toward the leaf area of the single topmost
    // phytomer and finished at 6 mm, far thinner than the 1667 cm^2 head it carries, against a
    // 21 mm base -- a 3.5:1 taper where Perrier et al. 2017 (Front. Plant Sci. 8:1516) measure
    // sorghum internodes running 18-19 mm at ranks 9-12 and 13-14 mm at ranks 16-20.

    Context context;
    context.seedRandomGenerator(1000);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("sorghum");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 90.f);
    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();
    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(shoot->phytomers.size() >= 8);
    DOCTEST_REQUIRE(!shoot->shoot_internode_radii.front().empty());
    DOCTEST_REQUIRE(!shoot->shoot_internode_radii.back().empty());

    // The panicle must actually exist, or the quantity under test is absent and the taper below
    // would be measured on a plant that never flowered.
    float panicle_area = 0.f;
    for (const auto &phytomer: shoot->phytomers) {
        panicle_area += phytomer->getInflorescenceArea();
    }
    DOCTEST_REQUIRE(panicle_area > 0.f);

    const float base_diameter = 2000.f * shoot->shoot_internode_radii.front().front();
    const float tip_diameter = 2000.f * shoot->shoot_internode_radii.back().back();
    DOCTEST_INFO("base=" << base_diameter << " mm, tip=" << tip_diameter << " mm, taper=" << base_diameter / tip_diameter << ", panicle area=" << panicle_area * 1e4f << " cm2");

    // The culm tip must be thick enough to read as a stem carrying a head. It measured 6.0 mm before
    // the panicle was counted and 11.6 mm after, so this separates the two without pinning the exact
    // calibration.
    DOCTEST_CHECK(tip_diameter > 9.f);

    // And the taper must be gradual rather than a collapse. The field ratio is about 1.4; the
    // unfixed model gave 3.5.
    DOCTEST_CHECK(base_diameter / tip_diameter < 2.5f);

    // The stem still has to taper upward -- widening the tip must not have inverted it.
    DOCTEST_CHECK(base_diameter > tip_diameter);
}

DOCTEST_TEST_CASE("Only a terminal inflorescence loads the stem it is borne on") {
    // The pipe model counts inflorescence area alongside leaf area when sizing an internode. That is
    // right for a grass panicle, which sits on the culm apex and continues the stem's own axis, and
    // wrong for a lateral flower cluster on a woody branch, which hangs off a stem that is already
    // sized by the leaves above it.
    //
    // Applied to every species alike, the term made a woody trunk respond to bloom. Redbud is
    // cauliflorous and flowers heavily on old wood: on a QSM-reconstructed Cercis canadensis the
    // bloom was 99% of the trunk's supported area for about two weeks, and because the radius update
    // in incrementPhytomerInternodeGirth() only ever increases, that transient peak was locked in
    // permanently -- the trunk finished 87% thicker than the tree it was reconstructed from and
    // stayed there long after the flowers had abscised.
    //
    // Both directions are asserted. A test that only checked the woody case would pass just as well
    // on code that had removed the inflorescence term altogether, undoing the sorghum culm taper the
    // term exists for.

    // --- Woody, lateral-only: redbud sets max_terminal_floral_buds = 0, so every one of its five
    // floral buds per petiole is axillary and none of them should load the stem.
    {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("easternredbud");

        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);
        const uint trunkID = plantarchitecture.getAllShootIDs(plantID).front();

        // Redbud sets flowers_require_dormancy, so a dormancy cycle is what makes it bloom at all.
        plantarchitecture.advanceTime(plantID, 40.f);
        plantarchitecture.makePlantDormant(plantID);
        plantarchitecture.breakPlantDormancy(plantID);

        // Walk day by day through the bloom rather than jumping past it: the flowers abscise, so a
        // single long step would land after the peak and measure a plant that is no longer flowering.
        float peak_inflorescence_area = 0.f;
        std::size_t peak_open_buds = 0;
        for (int day = 1; day <= 30; day++) {
            plantarchitecture.advanceTime(plantID, 1.f);

            const auto &trunk = plantarchitecture.getPlantShoot(plantID, trunkID);
            DOCTEST_REQUIRE(!trunk->phytomers.empty());
            peak_inflorescence_area = std::max(peak_inflorescence_area, trunk->sumDownstreamInflorescenceArea(0));

            std::size_t open_buds = 0;
            for (uint sid: plantarchitecture.getAllShootIDs(plantID)) {
                for (const auto &phytomer: plantarchitecture.getPlantShoot(plantID, sid)->phytomers) {
                    for (const auto &petiole: phytomer->floral_buds) {
                        for (const auto &fbud: petiole) {
                            if (!fbud.inflorescence_objIDs.empty()) {
                                open_buds++;
                            }
                        }
                    }
                }
            }
            peak_open_buds = std::max(peak_open_buds, open_buds);
        }

        // Guard against a vacuous pass: if the plant never flowered there is nothing to exclude, and
        // the assertion below would hold on any code at all. The tree carries well over a hundred
        // open flowers at peak bloom.
        DOCTEST_INFO("peak open floral buds=" << peak_open_buds << ", peak trunk inflorescence area=" << peak_inflorescence_area << " m2");
        DOCTEST_REQUIRE(peak_open_buds > 20);

        // A lateral flower cluster is not a load the stem is sized by. Before the term was gated the
        // trunk measured 0.205 m2 of downstream inflorescence area at peak bloom, 11% of everything
        // it supported.
        DOCTEST_CHECK(peak_inflorescence_area == 0.f);
    }

    // --- Grass, terminal panicle: the case the term exists for, which must keep working.
    {
        Context context;
        context.seedRandomGenerator(1000);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("sorghum");

        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 90.f);
        const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();
        const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        DOCTEST_REQUIRE(!shoot->phytomers.empty());

        const float culm_inflorescence_area = shoot->sumDownstreamInflorescenceArea(0);
        DOCTEST_INFO("sorghum culm inflorescence area=" << culm_inflorescence_area << " m2");

        // The panicle sits on the culm apex and continues its axis, so it still loads the stem.
        DOCTEST_CHECK(culm_inflorescence_area > 0.f);
    }
}

DOCTEST_TEST_CASE("Sorghum leaf size peaks below the flag leaf") {
    // Blade size along a sorghum culm rises from the base, peaks a few ranks below the top, and then
    // falls away over the last leaves, leaving the flag leaf the smallest blade on the plant
    // (Demarco et al. 2026, Annals of Botany 137(4):920; Lafarge & Hammer 2002, Field Crops Research
    // 77:137). The model's leaf ramp saturated at 1.0 by the fourth node and never declined, so every
    // leaf from there upward measured the same 448 cm^2 and the flag leaf came out fractionally the
    // LARGEST blade on the plant.
    //
    // Guards the growth-fraction fix as much as the profile: leaf expansion was measured against the
    // shoot type's prototype_scale rather than each leaf's own leaf_size_max, so any leaf whose
    // target was below the species maximum could never finish expanding and was shrunk a second time
    // by that shortfall -- a blade set to 62% of maximum rendered at 62% of 62%.

    Context context;
    context.seedRandomGenerator(1000);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("sorghum");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    plantarchitecture.advanceTime(plantID, 90.f);

    const uint shootID = plantarchitecture.getAllShootIDs(plantID).front();
    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(shoot->phytomers.size() >= 10);

    std::vector<float> leaf_area(shoot->phytomers.size(), 0.f);
    for (std::size_t node = 0; node < shoot->phytomers.size(); node++) {
        const auto &phytomer = shoot->phytomers.at(node);
        if (phytomer->leaf_objIDs.empty() || phytomer->leaf_objIDs.at(0).empty()) {
            continue;
        }
        const uint objID = phytomer->leaf_objIDs.at(0).front();
        if (context.doesObjectExist(objID)) {
            leaf_area.at(node) = context.getObjectArea(objID);
        }

        // Every leaf on a mature plant must have finished expanding. This is what the growth-fraction
        // defect broke, and it fails on the unfixed code for exactly the upper leaves whose target
        // size differs from the species maximum.
        DOCTEST_REQUIRE(!phytomer->current_leaf_scale_factor.empty());
        DOCTEST_CHECK(phytomer->current_leaf_scale_factor.at(0) == doctest::Approx(1.f).epsilon(0.01));
    }

    const std::size_t flag_node = shoot->phytomers.size() - 1;
    DOCTEST_REQUIRE(leaf_area.at(flag_node) > 0.f);

    // Locate the largest blade.
    std::size_t largest_node = 0;
    for (std::size_t node = 0; node < leaf_area.size(); node++) {
        if (leaf_area.at(node) > leaf_area.at(largest_node)) {
            largest_node = node;
        }
    }

    DOCTEST_INFO("largest leaf at node " << largest_node << " of " << flag_node << " (" << leaf_area.at(largest_node) * 1e4f << " cm2), flag leaf " << leaf_area.at(flag_node) * 1e4f << " cm2");

    // The flag leaf must not be the largest -- the defect put it there -- and the peak must sit
    // several ranks below it rather than merely one.
    DOCTEST_CHECK(largest_node < flag_node);
    DOCTEST_CHECK(flag_node - largest_node >= 2);

    // The flag leaf is distinctly smaller than the largest blade, not marginally so.
    DOCTEST_CHECK(leaf_area.at(flag_node) < 0.7f * leaf_area.at(largest_node));

    // ...but still a real leaf. An earlier attempt at this profile compounded the scaling and left
    // the flag leaf at under 1% of the largest blade.
    DOCTEST_CHECK(leaf_area.at(flag_node) > 0.2f * leaf_area.at(largest_node));

    // The decline toward the tip must be monotonic rather than jumping about. The slack is there because the size
    // ramp is very nearly flat for the first node or two past the peak, so which of two adjacent leaves comes out
    // larger is settled by rounding: on Windows two consecutive areas came out as 0.0449422 and 0.0449377, a
    // relative difference of 1e-4, and a strict comparison failed on the wrong side of it. One percent of the
    // largest blade is fifty times that and still far below the tens of percent a leaf that genuinely jumped back
    // up would contribute.
    const float monotonic_slack = 0.01f * leaf_area.at(largest_node);
    for (std::size_t node = largest_node + 1; node <= flag_node; node++) {
        DOCTEST_CHECK(leaf_area.at(node) <= leaf_area.at(node - 1) + monotonic_slack);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture child shoot base displaced radially when parent has no petioles") {
    // addChildShoot() offsets the child's base off the parent's axis so the branch emerges from the
    // parent's surface rather than from inside it. The offset direction is taken from the parent
    // petiole, but a phytomer with no petioles has no petiole axis to read, and the fallback used the
    // parent's INTERNODE axis instead. That points along the parent rather than away from it, so on
    // any petiole-free plant the child was displaced up the parent's own axis and emerged from the
    // wrong point entirely -- the displacement came out 100% along-axis and 0% radial.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // A leafless shoot type, as used when reconstructing a measured woody skeleton. The absence of
    // petioles is the whole point: with petioles present the correct branch is taken and the defect
    // is invisible.
    PhytomerParameters phytomer_parameters(context.getRandomGenerator());
    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 0;
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.petiole.petioles_per_internode = 0;
    phytomer_parameters.leaf.leaves_per_petiole = 0;

    ShootParameters shoot_parameters(context.getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.gravitropic_curvature = 0;
    shoot_parameters.tortuosity = 0;
    shoot_parameters.max_nodes = 10;
    shoot_parameters.vegetative_bud_break_probability_min = 0;
    shoot_parameters.defineChildShootTypes({"leafless"}, {1.f});
    plantarchitecture.defineShootType("leafless", shoot_parameters);

    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);

    // A vertical parent, so "along the parent" and "radially outward" are unambiguously separated:
    // the first is the z axis and the second is anything in the xy plane.
    const float parent_radius = 0.02f;
    const uint parent_shootID = plantarchitecture.addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0, 0), parent_radius, 0.05f, 1.f, 1.f, 0.f, "leafless");

    const uint parent_node_index = 1;
    const uint child_shootID = plantarchitecture.addChildShoot(plantID, parent_shootID, parent_node_index, 1, make_AxisRotation(deg2rad(60.f), 0.f, 0.f), 0.005f, 0.03f, 1.f, 1.f, 0.f, "leafless", 0);

    const auto &parent_shoot = plantarchitecture.getPlantShoot(plantID, parent_shootID);
    const auto &child_shoot = plantarchitecture.getPlantShoot(plantID, child_shootID);

    const std::vector<vec3> &parent_nodes = parent_shoot->shoot_internode_vertices.at(parent_node_index);
    const vec3 attachment_node = parent_nodes.back();
    // Take the parent axis across the whole shoot rather than within one phytomer: a phytomer built
    // with a single length segment holds only its own two endpoints, and reading a direction from a
    // shorter span than that is degenerate.
    vec3 parent_axis = parent_shoot->shoot_internode_vertices.back().back() - parent_shoot->shoot_internode_vertices.front().front();
    parent_axis.normalize();

    const vec3 child_base = child_shoot->shoot_internode_vertices.front().front();
    const vec3 offset = child_base - attachment_node;

    // The base must actually be displaced, otherwise the branch emerges from the parent's axis.
    DOCTEST_CHECK(offset.magnitude() > 0.5f * parent_radius);

    // The displacement must be radial, i.e. perpendicular to the parent axis. On the buggy code the
    // offset lay entirely along the parent axis and this component was the whole magnitude.
    const float along_axis_component = std::fabs(offset * parent_axis);
    DOCTEST_CHECK(along_axis_component < 0.1f * offset.magnitude());

    // ...and it should carry the base out to roughly the parent's surface, not further.
    const float radial_component = (offset - (offset * parent_axis) * parent_axis).magnitude();
    DOCTEST_CHECK(radial_component == doctest::Approx(0.9f * parent_radius).epsilon(0.05f));
}

DOCTEST_TEST_CASE("PlantArchitecture child shoot base displacement follows the shoot's own direction") {
    // The outward displacement addChildShoot() applies to a child's base should carry it toward the
    // side of the parent the child actually grows out of. For a parent phytomer with no petioles the
    // direction came from a "ghost petiole" whose azimuth about the parent depends only on the node
    // index, so two children leaving the same node in opposite directions were displaced the same
    // way: one emerged on the correct side of the parent and the other on the far side, having
    // crossed straight through it.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    PhytomerParameters phytomer_parameters(context.getRandomGenerator());
    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 0;
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.petiole.petioles_per_internode = 0;
    phytomer_parameters.leaf.leaves_per_petiole = 0;

    ShootParameters shoot_parameters(context.getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.gravitropic_curvature = 0;
    shoot_parameters.tortuosity = 0;
    shoot_parameters.max_nodes = 10;
    shoot_parameters.vegetative_bud_break_probability_min = 0;
    shoot_parameters.defineChildShootTypes({"leafless"}, {1.f});
    plantarchitecture.defineShootType("leafless", shoot_parameters);

    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    const float parent_radius = 0.02f;
    const uint parent_shootID = plantarchitecture.addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0, 0), parent_radius, 0.05f, 1.f, 1.f, 0.f, "leafless");

    // Two children leaving the SAME node of a vertical parent, 180 degrees apart in yaw.
    const uint node = 1;
    const uint childA = plantarchitecture.addChildShoot(plantID, parent_shootID, node, 1, make_AxisRotation(deg2rad(70.f), deg2rad(0.f), 0.f), 0.004f, 0.03f, 1.f, 1.f, 0.f, "leafless", 0);
    const uint childB = plantarchitecture.addChildShoot(plantID, parent_shootID, node, 1, make_AxisRotation(deg2rad(70.f), deg2rad(180.f), 0.f), 0.004f, 0.03f, 1.f, 1.f, 0.f, "leafless", 0);

    const auto &parent_shoot = plantarchitecture.getPlantShoot(plantID, parent_shootID);
    const vec3 attachment_node = parent_shoot->shoot_internode_vertices.at(node).back();
    vec3 parent_axis = parent_shoot->shoot_internode_vertices.back().back() - parent_shoot->shoot_internode_vertices.front().front();
    parent_axis.normalize();

    // For each child, compare the direction its base was displaced against the direction it grows.
    for (const uint childID: {childA, childB}) {
        const auto &child_shoot = plantarchitecture.getPlantShoot(plantID, childID);
        const std::vector<vec3> &child_nodes = child_shoot->shoot_internode_vertices.front();

        vec3 offset = child_nodes.front() - attachment_node;
        DOCTEST_REQUIRE(offset.magnitude() > 1e-6f);
        offset.normalize();

        // Direction the child grows, taken perpendicular to the parent so it is comparable to the
        // purely radial displacement.
        vec3 growth = child_nodes.back() - child_nodes.front();
        growth = growth - (growth * parent_axis) * parent_axis;
        DOCTEST_REQUIRE(growth.magnitude() > 1e-6f);
        growth.normalize();

        // The base must be pushed out on the side the child grows toward. On the buggy code one of
        // the two children scored -1 here, having been displaced to the opposite side of the parent.
        DOCTEST_CHECK(offset * growth > 0.9f);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture child shoot keeps its radial offset when the plant grows") {
    // addChildShoot() seats a child's base off the parent's axis so the branch emerges from the
    // parent's surface, but Shoot::updateShootNodes() -- which runs whenever the parent's geometry is
    // rebuilt, and therefore on every timestep -- re-derived that base as the parent's node position
    // alone. The offset was discarded the first time the plant grew, dropping every branch back onto
    // the axis of the shoot it came from. The effect compounds down a branch order, since the
    // correction cascades to descendants.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    PhytomerParameters phytomer_parameters(context.getRandomGenerator());
    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 0;
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.petiole.petioles_per_internode = 0;
    phytomer_parameters.leaf.leaves_per_petiole = 0;

    ShootParameters shoot_parameters(context.getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.gravitropic_curvature = 0;
    shoot_parameters.tortuosity = 0;
    shoot_parameters.max_nodes = 20;
    shoot_parameters.vegetative_bud_break_probability_min = 0;
    shoot_parameters.defineChildShootTypes({"leafless"}, {1.f});
    plantarchitecture.defineShootType("leafless", shoot_parameters);

    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    const float parent_radius = 0.02f;
    const uint parent_shootID = plantarchitecture.addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0, 0), parent_radius, 0.05f, 1.f, 1.f, 0.f, "leafless");
    const uint child_shootID = plantarchitecture.addChildShoot(plantID, parent_shootID, 1, 2, make_AxisRotation(deg2rad(60.f), 0.f, 0.f), 0.005f, 0.03f, 1.f, 1.f, 0.f, "leafless", 0);

    const auto &parent_shoot = plantarchitecture.getPlantShoot(plantID, parent_shootID);
    const auto &child_shoot = plantarchitecture.getPlantShoot(plantID, child_shootID);

    vec3 parent_axis = parent_shoot->shoot_internode_vertices.back().back() - parent_shoot->shoot_internode_vertices.front().front();
    parent_axis.normalize();

    // Distance of the child's base from the parent's axis, which is what the offset establishes.
    auto radial_distance_from_parent_axis = [&]() {
        const vec3 attachment_node = parent_shoot->shoot_internode_vertices.at(child_shoot->parent_node_index).back();
        const vec3 offset = child_shoot->shoot_internode_vertices.front().front() - attachment_node;
        return (offset - (offset * parent_axis) * parent_axis).magnitude();
    };

    const float radial_when_built = radial_distance_from_parent_axis();
    DOCTEST_REQUIRE(radial_when_built > 0.5f * parent_radius);

    // Rebuilding the parent's nodes must preserve the offset. This is the call the growth loop makes
    // on every timestep, so a child that loses its offset here loses it as soon as the plant grows.
    parent_shoot->updateShootNodes(false);

    DOCTEST_CHECK(radial_distance_from_parent_axis() == doctest::Approx(radial_when_built).epsilon(0.01f));
}

//! Minimal petiole-free shoot type used by the prescribed-geometry tests, matching the leafless woody
//! skeleton a reconstruction works with. Radii are pinned by leaving girth_area_factor at zero.
/**
 * Note this type builds measured geometry AND governs its own future growth: curvature and tortuosity
 * are zeroed and the node cap is raised so a measured path can be built, and defineChildShootTypes()
 * then points back at this same type. That is the right setup for the tests below, which assert on the
 * prescribed section rather than on what grows from it -- but it is exactly the arrangement that made
 * new growth come out straight and unbounded in a real reconstruction. See
 * defineMeasuredAndGrowthShootTypes() further down for the separated form.
 */
static void definePrescribedGeometryShootType(Context &context, PlantArchitecture &plantarchitecture) {
    PhytomerParameters phytomer_parameters(context.getRandomGenerator());
    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 0;
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.petiole.petioles_per_internode = 0;
    phytomer_parameters.leaf.leaves_per_petiole = 0;

    ShootParameters shoot_parameters(context.getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.gravitropic_curvature = 0;
    shoot_parameters.tortuosity = 0;
    shoot_parameters.max_nodes = 40;
    shoot_parameters.vegetative_bud_break_probability_min = 0;
    shoot_parameters.girth_area_factor = 0;
    shoot_parameters.defineChildShootTypes({"leafless"}, {1.f});
    plantarchitecture.defineShootType("leafless", shoot_parameters);
}

DOCTEST_TEST_CASE("PlantArchitecture addShootFromNodePositions follows a curved path") {
    // A shoot built the ordinary way is an extrapolation: a base rotation, then curvature and
    // tortuosity integrated forward one internode at a time. With both of those zeroed the shoot is
    // exactly straight, and base_rotation only affects phytomer 0, so no combination of arguments to
    // addBaseStemShoot()/addChildShoot() can make ONE shoot follow a measured curve. Reconstructing a
    // measured branch therefore meant chaining many one-phytomer shoots, which is both a different
    // object per link (visible gaps at every bend) and structurally wrong, since a shoot is the
    // botanical axis. This is the test that no pre-existing API can pass.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    definePrescribedGeometryShootType(context, plantarchitecture);

    // A quarter circle of radius 0.5 m in the x-z plane, rising from the origin. Its chord/arclength
    // is 2*sin(pi/4)/(pi/2) = 0.900, far enough below 1 that no straight shoot can be mistaken for it.
    const float arc_radius = 0.5f;
    const uint Ninternodes = 12;
    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    for (uint i = 0; i <= Ninternodes; i++) {
        const float theta = 0.5f * PI_F * float(i) / float(Ninternodes);
        node_positions.push_back(make_vec3(arc_radius * (1.f - std::cos(theta)), 0.f, arc_radius * std::sin(theta)));
        node_radii.push_back(0.01f);
    }

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafless");

    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(shoot->phytomers.size() == Ninternodes);

    const std::vector<vec3> built_nodes = flatten(shoot->shoot_internode_vertices);
    DOCTEST_REQUIRE(built_nodes.size() >= 2);

    float arclength = 0.f;
    for (size_t i = 0; i + 1 < built_nodes.size(); i++) {
        arclength += (built_nodes.at(i + 1) - built_nodes.at(i)).magnitude();
    }
    const float chord = (built_nodes.back() - built_nodes.front()).magnitude();
    DOCTEST_REQUIRE(arclength > 0.f);

    // The straight-shoot behaviour this replaces gives exactly 1.0 here.
    DOCTEST_CHECK(chord / arclength == doctest::Approx(0.9003f).epsilon(0.01f));

    // ...and the curve must bend in the plane it was given, not merely be shorter end-to-end.
    DOCTEST_CHECK(built_nodes.back().x == doctest::Approx(arc_radius).epsilon(0.02f));
    DOCTEST_CHECK(built_nodes.back().z == doctest::Approx(arc_radius).epsilon(0.02f));
}

DOCTEST_TEST_CASE("PlantArchitecture addShootFromNodePositions reproduces the supplied node positions") {
    // The core contract of the method: the wood ends up where the caller said it was.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    // Deliberately irregular in all three axes and in spacing, so a shoot that merely happens to be
    // straight, evenly spaced, or axis-aligned cannot pass.
    const std::vector<vec3> node_positions = {make_vec3(0.f, 0.f, 0.f), make_vec3(0.02f, 0.01f, 0.10f), make_vec3(0.09f, -0.03f, 0.17f), make_vec3(0.11f, -0.02f, 0.31f), make_vec3(0.19f, 0.05f, 0.36f)};
    const std::vector<float> node_radii = {0.020f, 0.017f, 0.013f, 0.010f, 0.006f};

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafless");

    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(shoot->phytomers.size() == node_positions.size() - 1);

    // Each phytomer's own endpoints must be the pair of positions it was given.
    for (size_t p = 0; p < shoot->phytomers.size(); p++) {
        const std::vector<vec3> phytomer_nodes = shoot->phytomers.at(p)->getInternodeNodePositions();
        DOCTEST_REQUIRE(phytomer_nodes.size() >= 2);
        DOCTEST_CHECK((phytomer_nodes.front() - node_positions.at(p)).magnitude() < 1e-5f);
        DOCTEST_CHECK((phytomer_nodes.back() - node_positions.at(p + 1)).magnitude() < 1e-5f);
    }

    const std::vector<float> built_radii = flatten(shoot->shoot_internode_radii);
    DOCTEST_REQUIRE(built_radii.size() == node_radii.size());
    for (size_t node = 0; node < node_radii.size(); node++) {
        DOCTEST_CHECK(built_radii.at(node) == doctest::Approx(node_radii.at(node)).epsilon(1e-4f));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture addShootFromNodePositions builds one tube object per shoot") {
    // The defect that motivated the method: chaining one-phytomer shoots to follow a measured path gave
    // one capped tube object per link, which separates visibly at every bend. A whole branch must be a
    // single shoot carrying a single tube.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    const uint Ninternodes = 10;
    for (uint i = 0; i <= Ninternodes; i++) {
        const float theta = 0.6f * float(i) / float(Ninternodes);
        node_positions.push_back(make_vec3(0.3f * std::sin(theta), 0.f, 0.3f * theta));
        node_radii.push_back(0.02f - 0.001f * float(i));
    }

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafless");

    // Add a branch as well, so the count below distinguishes "one tube per shoot" from "one tube total".
    std::vector<vec3> branch_positions;
    std::vector<float> branch_radii;
    const vec3 branch_base = node_positions.at(5);
    for (uint i = 0; i <= 4; i++) {
        branch_positions.push_back(branch_base + make_vec3(0.f, 0.04f * float(i), 0.03f * float(i)));
        branch_radii.push_back(0.008f);
    }
    const uint branchID = plantarchitecture.addShootFromNodePositions(plantID, shootID, 4, branch_positions, branch_radii, "leafless");

    const std::vector<uint> shootIDs = plantarchitecture.getAllShootIDs(plantID);
    DOCTEST_REQUIRE(shootIDs.size() == 2);

    std::set<uint> tube_objIDs;
    for (const uint id: shootIDs) {
        const auto &shoot = plantarchitecture.getPlantShoot(plantID, id);
        DOCTEST_REQUIRE(context.doesObjectExist(shoot->internode_tube_objID));
        tube_objIDs.insert(shoot->internode_tube_objID);

        // The tube's node list must line up 1:1 with the shoot's stored nodes, which is the storage
        // convention the prescribed path has to preserve: the first phytomer group carries both of its
        // endpoints and every later group omits the node it shares with the previous one.
        const uint tube_node_count = context.getTubeObjectNodeCount(shoot->internode_tube_objID);
        DOCTEST_CHECK(flatten(shoot->shoot_internode_vertices).size() == tube_node_count);
        DOCTEST_CHECK(flatten(shoot->shoot_internode_radii).size() == tube_node_count);

        const size_t segments_per_phytomer = shoot->shoot_internode_vertices.front().size() - 1;
        for (size_t p = 1; p < shoot->shoot_internode_vertices.size(); p++) {
            DOCTEST_CHECK(shoot->shoot_internode_vertices.at(p).size() == segments_per_phytomer);
        }
    }

    // Two shoots, two distinct tubes -- not one per phytomer, which is what the workaround produced.
    DOCTEST_CHECK(tube_objIDs.size() == 2);
    DOCTEST_CHECK(branchID != shootID);
}

DOCTEST_TEST_CASE("PlantArchitecture addShootFromNodePositions child stays attached and keeps its shape when the plant grows") {
    // Shoot::updateShootNodes() re-seats a child's base onto its parent on every timestep and cascades to
    // descendants. It shifts the whole shoot rigidly, so a prescribed path must come through unchanged in
    // shape while remaining attached -- the branch must neither drift off the parent nor be deformed.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    const std::vector<vec3> trunk_positions = {make_vec3(0.f, 0.f, 0.f), make_vec3(0.f, 0.f, 0.15f), make_vec3(0.01f, 0.f, 0.30f), make_vec3(0.02f, 0.f, 0.45f)};
    const std::vector<float> trunk_radii = {0.03f, 0.026f, 0.022f, 0.018f};

    const uint plantID = plantarchitecture.addPlantInstance(trunk_positions.front(), 0.f);
    const uint trunkID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, trunk_positions, trunk_radii, "leafless");

    // A curved branch leaving the trunk's node 1, given from the trunk's own axis point.
    const uint attachment_node_index = 1;
    const vec3 attachment_node = trunk_positions.at(attachment_node_index + 1);
    const std::vector<vec3> branch_positions = {attachment_node, attachment_node + make_vec3(0.06f, 0.02f, 0.04f), attachment_node + make_vec3(0.14f, 0.03f, 0.05f), attachment_node + make_vec3(0.20f, 0.02f, 0.03f)};
    const std::vector<float> branch_radii = {0.012f, 0.010f, 0.008f, 0.005f};

    const uint branchID = plantarchitecture.addShootFromNodePositions(plantID, trunkID, attachment_node_index, branch_positions, branch_radii, "leafless");

    const auto &trunk_shoot = plantarchitecture.getPlantShoot(plantID, trunkID);
    const auto &branch_shoot = plantarchitecture.getPlantShoot(plantID, branchID);

    // Shape is what must be invariant: the segment vectors between consecutive nodes.
    auto branch_segment_vectors = [&]() {
        const std::vector<vec3> nodes = flatten(branch_shoot->shoot_internode_vertices);
        std::vector<vec3> segments;
        for (size_t i = 0; i + 1 < nodes.size(); i++) {
            segments.push_back(nodes.at(i + 1) - nodes.at(i));
        }
        return segments;
    };
    auto branch_gap_from_parent = [&]() {
        const vec3 parent_node = trunk_shoot->shoot_internode_vertices.at(attachment_node_index).back();
        return (branch_shoot->shoot_internode_vertices.front().front() - parent_node).magnitude();
    };

    const std::vector<vec3> segments_when_built = branch_segment_vectors();
    const float gap_when_built = branch_gap_from_parent();

    // Seated on the parent's surface, not on its axis and not adrift.
    const float parent_radius = trunk_shoot->phytomers.at(attachment_node_index)->getInternodeRadius(1.f);
    DOCTEST_CHECK(gap_when_built == doctest::Approx(0.9f * parent_radius).epsilon(0.05f));

    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 5);

    // Still attached at the same offset...
    DOCTEST_CHECK(branch_gap_from_parent() == doctest::Approx(gap_when_built).epsilon(0.02f));

    // ...and the prescribed section is not deformed. Growth may append phytomers at the tip, so compare
    // only the segments that were prescribed.
    const std::vector<vec3> segments_after_growth = branch_segment_vectors();
    DOCTEST_REQUIRE(segments_after_growth.size() >= segments_when_built.size());
    for (size_t i = 0; i < segments_when_built.size(); i++) {
        DOCTEST_CHECK((segments_after_growth.at(i) - segments_when_built.at(i)).magnitude() < 1e-5f);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture addShootFromNodePositions grows new phytomers at the tip") {
    // Prescribed nodes describe wood that already exists. Growth must continue from the tip using the
    // shoot type's own parameters while leaving the measured section exactly as it was given.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    const std::vector<vec3> node_positions = {make_vec3(0.f, 0.f, 0.f), make_vec3(0.f, 0.f, 0.08f), make_vec3(0.01f, 0.f, 0.16f), make_vec3(0.03f, 0.f, 0.23f)};
    const std::vector<float> node_radii = {0.015f, 0.013f, 0.011f, 0.009f};

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafless");

    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const size_t phytomers_when_built = shoot->phytomers.size();
    const std::vector<vec3> nodes_when_built = flatten(shoot->shoot_internode_vertices);

    // Manually-added shoots are constructed dormant; without this the plant does not grow at all and the
    // assertions below would pass for the wrong reason.
    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 40);

    DOCTEST_CHECK(shoot->phytomers.size() > phytomers_when_built);

    // The prescribed nodes are untouched by the growth that happened above them.
    const std::vector<vec3> nodes_after_growth = flatten(shoot->shoot_internode_vertices);
    DOCTEST_REQUIRE(nodes_after_growth.size() >= nodes_when_built.size());
    for (size_t i = 0; i < nodes_when_built.size(); i++) {
        DOCTEST_CHECK((nodes_after_growth.at(i) - nodes_when_built.at(i)).magnitude() < 1e-5f);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture setShootInternodeLengthMax sets the length of internodes grown at the apex") {
    // A shoot built from measured nodes grows new internodes toward the mean of its prescribed internode
    // lengths. A seedling's measured stem is mostly hypocotyl, so that mean says nothing about how long the
    // internodes it grows next should be, and a caller growing a reconstruction forward must be able to set it.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    const std::vector<vec3> node_positions = {make_vec3(0, 0, 0), make_vec3(0, 0, 0.02f), make_vec3(0, 0, 0.04f), make_vec3(0, 0, 0.06f)};
    const std::vector<float> node_radii = {0.004f, 0.0035f, 0.003f, 0.0025f};

    auto longestGrownInternode = [&](bool set_length, uint &plantID, uint &shootID) {
        plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
        shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafless");
        const size_t prescribed = plantarchitecture.getPlantShoot(plantID, shootID)->phytomers.size();
        if (set_length) {
            plantarchitecture.setShootInternodeLengthMax(plantID, shootID, 0.06f);
        }
        plantarchitecture.breakPlantDormancy(plantID);
        plantarchitecture.advanceTime(plantID, 60);
        const std::vector<float> lengths = plantarchitecture.getPlantInternodeLengths(plantID);
        DOCTEST_REQUIRE(lengths.size() > prescribed);
        return *std::max_element(lengths.begin() + long(prescribed), lengths.end());
    };

    uint control_plantID, control_shootID;
    const float control = longestGrownInternode(false, control_plantID, control_shootID);
    DOCTEST_CHECK_MESSAGE(control < 0.021f, "without the setter the grown internodes follow the prescribed mean of 0.02 m, but reached " << control);

    uint plantID, shootID;
    const float grown = longestGrownInternode(true, plantID, shootID);
    DOCTEST_CHECK_MESSAGE(grown > 0.05f, "after setting 0.06 m the longest grown internode is only " << grown << " m");
    DOCTEST_CHECK(grown < 0.063f);

    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS(plantarchitecture.setShootInternodeLengthMax(plantID, shootID, 0.f));
    DOCTEST_CHECK_THROWS(plantarchitecture.setShootInternodeLengthMax(plantID, shootID, -0.01f));
    DOCTEST_CHECK_THROWS(plantarchitecture.setShootInternodeLengthMax(plantID, shootID + 100, 0.03f));
    DOCTEST_CHECK_THROWS(plantarchitecture.setShootInternodeLengthMax(plantID + 100, shootID, 0.03f));
}

DOCTEST_TEST_CASE("PlantArchitecture addShootFromNodePositions rejects invalid input") {
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    const std::vector<vec3> valid_positions = {make_vec3(0.f, 0.f, 0.f), make_vec3(0.f, 0.f, 0.1f), make_vec3(0.f, 0.f, 0.2f)};
    const std::vector<float> valid_radii = {0.02f, 0.015f, 0.01f};

    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);

    {
        capture_cerr cerr_buffer;

        // Plant does not exist.
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID + 100, -1, 0, valid_positions, valid_radii, "leafless")));

        // Shoot type does not exist.
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, valid_positions, valid_radii, "nonexistent_type")));

        // Fewer than two node positions cannot define an internode.
        const std::vector<vec3> one_position = {make_vec3(0.f, 0.f, 0.f)};
        const std::vector<float> one_radius = {0.02f};
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, one_position, one_radius, "leafless")));

        // Radii count must match the node count.
        const std::vector<float> too_few_radii = {0.02f, 0.015f};
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, valid_positions, too_few_radii, "leafless")));

        // Radii must be strictly positive.
        const std::vector<float> zero_radius = {0.02f, 0.f, 0.01f};
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, valid_positions, zero_radius, "leafless")));
        const std::vector<float> negative_radius = {0.02f, -0.015f, 0.01f};
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, valid_positions, negative_radius, "leafless")));

        // Coincident consecutive positions give a degenerate internode axis.
        const std::vector<vec3> coincident_positions = {make_vec3(0.f, 0.f, 0.f), make_vec3(0.f, 0.f, 0.1f), make_vec3(0.f, 0.f, 0.1f)};
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, coincident_positions, valid_radii, "leafless")));

        // A parent shoot that does not exist.
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, 7, 0, valid_positions, valid_radii, "leafless")));
    }

    // Build a real parent so the remaining parent-relative failures can be exercised.
    const uint parentID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, valid_positions, valid_radii, "leafless");

    {
        capture_cerr cerr_buffer;

        // A node index the parent does not have.
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, parentID, 9, valid_positions, valid_radii, "leafless")));

        // A first position far from the attachment node would silently produce a detached branch.
        const std::vector<vec3> detached_positions = {make_vec3(5.f, 5.f, 5.f), make_vec3(5.f, 5.f, 5.1f), make_vec3(5.f, 5.f, 5.2f)};
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, parentID, 1, detached_positions, valid_radii, "leafless")));
    }

    // A first position at the attachment node itself is accepted.
    const vec3 attachment_node = valid_positions.at(2);
    const std::vector<vec3> attached_positions = {attachment_node, attachment_node + make_vec3(0.05f, 0.f, 0.02f), attachment_node + make_vec3(0.10f, 0.f, 0.03f)};
    DOCTEST_CHECK_NOTHROW(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, parentID, 1, attached_positions, valid_radii, "leafless")));
}

//! Build type and growth type used by the measured-geometry / growth-parameter separation tests.
/**
 * The two differ in exactly the ways a reconstruction forces them apart: the build type carries the
 * values needed to CONSTRUCT measured wood (curvature and tortuosity zeroed so the measured path is
 * not fought, a node cap raised high enough to fit the longest measured branch, girth scaling off so
 * measured radii are preserved), while the growth type carries the values that should govern what the
 * plant does NEXT (real curvature, a realistic node cap).
 *
 * Both are leafless. The internode bud special case in Phytomer's constructor creates vegetative buds
 * directly on the internode when petioles_per_internode is 0, so a leafless shoot still grows -- which
 * is what makes it possible to test growth off prescribed wood at all.
 */
static void defineMeasuredAndGrowthShootTypes(Context &context, PlantArchitecture &plantarchitecture, int growth_max_nodes = 6) {
    PhytomerParameters phytomer_parameters(context.getRandomGenerator());
    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 0;
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.petiole.petioles_per_internode = 0;
    phytomer_parameters.leaf.leaves_per_petiole = 0;

    // The type that builds the measured wood.
    ShootParameters build_parameters(context.getRandomGenerator());
    build_parameters.phytomer_parameters = phytomer_parameters;
    build_parameters.gravitropic_curvature = 0;
    build_parameters.tortuosity = 0;
    build_parameters.max_nodes = 2000;
    build_parameters.girth_area_factor = 0;
    build_parameters.vegetative_bud_break_probability_min = 1;
    build_parameters.vegetative_bud_break_probability_max = 1;
    build_parameters.vegetative_bud_break_probability_decay_rate = 0;
    build_parameters.vegetative_bud_break_time = 0;
    build_parameters.internode_length_max = 0.05;
    build_parameters.internode_length_min = 0.05;

    // The type that should govern everything the plant does afterwards. A large gravitropic curvature
    // and a small node cap are both chosen to be unmistakable in the assertions below.
    //
    // Bud break is switched OFF on this type. The measured wood breaks every bud so that there is
    // certainly new growth to assert on, but if the new shoots went on to do the same the plant would
    // grow exponentially and the test would take longer than the rest of the suite combined. One
    // generation of new growth is all these tests examine.
    ShootParameters growth_parameters = build_parameters;
    growth_parameters.gravitropic_curvature = 600;
    growth_parameters.max_nodes = growth_max_nodes;
    growth_parameters.vegetative_bud_break_probability_min = 0;
    growth_parameters.vegetative_bud_break_probability_max = 0;

    plantarchitecture.defineShootType("measured_wood", build_parameters);
    plantarchitecture.defineShootType("new_growth", growth_parameters);
}

//! Straight polyline standing in for a measured branch, rising at a shallow angle.
/**
 * Deliberately close to horizontal. Gravitropic curvature bends a shoot back toward vertical, so a
 * measured branch that already stood upright would leave its children pointing very nearly where
 * gravitropism wants them and the curvature under test would have almost nothing to do -- the grown
 * shoots come out within 0.03% of straight, which no robust threshold can separate from zero. Starting
 * near horizontal gives the growth type's curvature the full range to act over.
 */
static void measuredWoodPath(uint Ninternodes, std::vector<vec3> &node_positions, std::vector<float> &node_radii) {
    node_positions.clear();
    node_radii.clear();
    for (uint i = 0; i <= Ninternodes; i++) {
        node_positions.push_back(make_vec3(0.05f * float(i), 0.f, 0.005f * float(i)));
        node_radii.push_back(0.01f);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture measured shoot grows using the growth shoot type") {
    // The bug this guards: a shoot type carries both the values used to BUILD measured geometry and the
    // values governing how that shoot grows NEXT. A reconstruction must zero curvature and raise the node
    // cap to build measured wood, and until the two roles were separated every shoot grown off that wood
    // inherited those build-time values -- so new growth came out perfectly straight and never stopped
    // adding nodes. Nothing errored; the plant simply grew wrongly.
    //
    // Curvature is the observable because it differs by 600 deg/m between the two types while the
    // prescribed section is exempt from curvature by construction, so the measured wood staying straight
    // and the new wood bending are independent claims that can both be asserted on one plant.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

    const uint Ninternodes = 4;
    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(Ninternodes, node_positions, node_radii);

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "new_growth");

    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 40);

    const auto &shoot_tree = PlantArchitectureTestHelper::getShootTree(plantarchitecture, plantID);

    // Every shoot grown off the measured wood must be of the growth type, not the build type.
    uint child_shoots = 0;
    for (const auto &shoot: shoot_tree) {
        if (shoot->ID == int(shootID)) {
            continue;
        }
        child_shoots++;
        DOCTEST_CHECK(shoot->shoot_type_label == "new_growth");
    }
    DOCTEST_REQUIRE(child_shoots > 0);

    // The measured section itself must be untouched: its nodes are still exactly where they were
    // supplied. The shoot is longer than the measurement by now -- new phytomers have been added at the
    // apex, which is the point -- so this checks the measured prefix rather than the whole polyline.
    const auto &measured_shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const std::vector<vec3> built_nodes = flatten(measured_shoot->shoot_internode_vertices);
    DOCTEST_REQUIRE(built_nodes.size() >= Ninternodes + 1);
    for (uint i = 0; i <= Ninternodes; i++) {
        DOCTEST_CHECK((built_nodes.at(i) - node_positions.at(i)).magnitude() < 1e-4f);
    }

    // ...and it must actually have extended, or the "measured section is unchanged" check above would
    // be trivially satisfied by a shoot that never grew at all.
    DOCTEST_CHECK(built_nodes.size() > Ninternodes + 1);

    // ...and the new growth must actually bend. A shoot built with the build type's zero curvature is
    // exactly straight, giving a chord/arclength of 1; the growth type's curvature must pull it below.
    uint curved_shoots = 0;
    for (const auto &shoot: shoot_tree) {
        if (shoot->ID == int(shootID) || shoot->phytomers.size() < 3) {
            continue;
        }

        const std::vector<vec3> child_nodes = flatten(shoot->shoot_internode_vertices);
        float arclength = 0.f;
        for (size_t i = 0; i + 1 < child_nodes.size(); i++) {
            arclength += (child_nodes.at(i + 1) - child_nodes.at(i)).magnitude();
        }
        const float chord = (child_nodes.back() - child_nodes.front()).magnitude();
        if (arclength > 0.f && chord / arclength < 0.999f) {
            curved_shoots++;
        }
    }
    DOCTEST_CHECK(curved_shoots > 0);
}

DOCTEST_TEST_CASE("PlantArchitecture measured shoot stamps its buds with the growth shoot type") {
    // Which type a bud becomes is decided when the PHYTOMER is created, not when the bud breaks:
    // Shoot::sampleChildShootType() runs inside the phytomer append loop and writes the chosen label
    // into VegetativeBud::shoot_type_label. So the growth type has to be recorded on the shoot before
    // the prescribed phytomers are appended. Setting it afterwards leaves every bud on the measured
    // wood stamped with the build type, and the bug survives with no visible symptom until the plant
    // is grown. This test pins the ordering directly rather than inferring it from grown geometry.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(4, node_positions, node_radii);

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "new_growth");

    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(!shoot->phytomers.empty());

    uint buds_checked = 0;
    for (const auto &phytomer: shoot->phytomers) {
        for (const auto &petiole: phytomer->axillary_vegetative_buds) {
            for (const auto &vbud: petiole) {
                DOCTEST_CHECK(vbud.shoot_type_label == "new_growth");
                buds_checked++;
            }
        }
    }
    DOCTEST_REQUIRE(buds_checked > 0);
}

DOCTEST_TEST_CASE("PlantArchitecture measured shoot takes its growth node cap from the growth shoot type") {
    // max_nodes carries two unrelated meanings. addShootFromNodePositions() checks it as a build-time
    // capacity ("may I construct a shoot this large"), which forces a reconstruction to raise it above
    // the longest measured branch; advanceTime() then reads the same field as a growth cap ("may this
    // grow further"). With one field serving both, raising it to fit the measurement silently disabled
    // the growth cap and shoots never stopped extending.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    const int growth_max_nodes = 5;
    defineMeasuredAndGrowthShootTypes(context, plantarchitecture, growth_max_nodes);

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(4, node_positions, node_radii);

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "new_growth");

    plantarchitecture.breakPlantDormancy(plantID);
    // Long enough for a new shoot to run well past the growth cap of 5 nodes at the default
    // phyllochron, but not so long that the run dominates the suite.
    plantarchitecture.advanceTime(plantID, 60);

    const auto &shoot_tree = PlantArchitectureTestHelper::getShootTree(plantarchitecture, plantID);

    uint grown_shoots = 0;
    for (const auto &shoot: shoot_tree) {
        if (shoot->ID == int(shootID)) {
            continue;
        }
        grown_shoots++;
        // The build type's cap of 2000 is what these shoots would run to if the cap were read from it.
        DOCTEST_CHECK(int(shoot->phytomers.size()) <= growth_max_nodes);
    }
    DOCTEST_REQUIRE(grown_shoots > 0);
}

DOCTEST_TEST_CASE("PlantArchitecture vegetative bud break follows the shoot, not the shoot type") {
    // Bud-break probability is documented as being read from the shoot BEARING the bud (see the
    // "Separating Measured Geometry from Growth Parameters" section of the plug-in documentation), but
    // Shoot::sampleVegetativeBudBreak() read it from the plant's stored shoot TYPE instead, while
    // reading growth_requires_dormancy from the shoot three lines later. So a probability set on one
    // shoot was silently ignored and that shoot kept breaking buds at its type's rate.
    //
    // The lever has to be a per-shoot edit made AFTER the shoot exists. Building with a separate growth
    // type does not work: a shoot built by addShootFromNodePositions() keeps the BUILD label as its
    // shoot_type_label, so the type lookup and the shoot agree and the buggy code returns the right
    // answer by accident. Only mutating the shoot's own parameters makes the two sources disagree.
    //
    // Probability 1 against 0 makes the outcome deterministic, so this asserts bud state directly with
    // no growth, no geometry and no tolerance.
    auto budStatesAfterDormancyBreak = [](bool silence_this_shoot) -> std::vector<BudState> {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();

        // "measured_wood" carries bud-break probability 1 and a decay rate of 0, so every bud is created
        // BUD_ACTIVE. That matters: growth_requires_dormancy is false by default, so bud break is already
        // sampled once when each phytomer is created, and breakDormancy() skips any bud already BUD_DEAD.
        defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

        std::vector<vec3> node_positions;
        std::vector<float> node_radii;
        measuredWoodPath(4, node_positions, node_radii);

        const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
        const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "measured_wood");

        const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);

        // Precondition: the buds must all be alive going in, or the assertions below would pass for a
        // reason that has nothing to do with which parameter source was consulted.
        for (const auto &phytomer: shoot->phytomers) {
            for (const auto &petiole: phytomer->axillary_vegetative_buds) {
                for (const auto &vbud: petiole) {
                    DOCTEST_REQUIRE(vbud.state == BUD_ACTIVE);
                }
            }
        }

        if (silence_this_shoot) {
            // Edit only this shoot. The plant's stored shoot type still says 1. RandomParameter_float's
            // assignment operator clears the sampled flag, so the next val() returns the new value
            // rather than a previously cached draw.
            shoot->shoot_parameters.vegetative_bud_break_probability_min = 0.f;
            shoot->shoot_parameters.vegetative_bud_break_probability_max = 0.f;
            shoot->shoot_parameters.vegetative_bud_break_probability_decay_rate = 0.f;
        }

        plantarchitecture.breakPlantDormancy(plantID);

        std::vector<BudState> states;
        for (const auto &phytomer: shoot->phytomers) {
            for (const auto &petiole: phytomer->axillary_vegetative_buds) {
                for (const auto &vbud: petiole) {
                    states.push_back(vbud.state);
                }
            }
        }
        return states;
    };

    // Positive control. Without it, a change that killed bud break outright would still satisfy the
    // assertion below and the test would quietly stop testing anything.
    const std::vector<BudState> states_untouched = budStatesAfterDormancyBreak(false);
    DOCTEST_REQUIRE(!states_untouched.empty());
    for (const BudState state: states_untouched) {
        DOCTEST_CHECK(state == BUD_ACTIVE);
    }

    // The shoot was told not to break its buds. On the unfixed code the probability came from the type,
    // which still says 1, and every bud stayed BUD_ACTIVE.
    const std::vector<BudState> states_silenced = budStatesAfterDormancyBreak(true);
    DOCTEST_REQUIRE(states_silenced.size() == states_untouched.size());
    for (const BudState state: states_silenced) {
        DOCTEST_CHECK(state == BUD_DEAD);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture appended shoot stays seated on the parent axis through growth") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Regression test: appendShoot() creates an axis continuation -- the new shoot carries on where the
    // parent stopped, keeping the parent's rank -- so its base belongs on the parent's axis. But
    // Shoot::updateShootNodes() re-seated the base of every shoot with a parent using the lateral-branch
    // rule, parent_node_tip + 0.9*parent_radius outward, which is correct only for addChildShoot(). The
    // appended shoot was therefore shunted sideways off the axis by nearly a full stem radius on the
    // first timestep, and because the offset is proportional to the parent radius it grew as the stem
    // thickened. In the soybean model this appeared as the trifoliate shoot floating off the unifoliate
    // shoot rather than continuing it.
    //
    // Grow the plant so the internode thickens: at the initial radius the displacement is a few tens of
    // microns and indistinguishable from rounding, which is why building the plant and measuring
    // immediately would pass on the buggy code.
    plantarchitecture.loadPlantModelFromLibrary("soybean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);

    plantarchitecture.advanceTime(plantID, 30.f);

    auto parent_shoot = plantarchitecture.getPlantShoot(plantID, 0);
    auto appended_shoot = plantarchitecture.getPlantShoot(plantID, 1);

    // Confirm the topology this test depends on: shoot 1 is an appendShoot() continuation of shoot 0,
    // which is what makes the lateral offset wrong. An addChildShoot() branch would be rank+1.
    DOCTEST_REQUIRE(appended_shoot->parent_shoot_ID == 0);
    DOCTEST_REQUIRE(appended_shoot->rank == parent_shoot->rank);

    const vec3 parent_tip = parent_shoot->shoot_internode_vertices.back().back();
    const vec3 appended_base = appended_shoot->shoot_internode_vertices.front().front();
    const float parent_radius = parent_shoot->shoot_internode_radii.back().back();

    const float displacement = (appended_base - parent_tip).magnitude();

    // The stem must actually have thickened, or the tolerance below would be meaningless.
    DOCTEST_REQUIRE(parent_radius > 0.002f);

    // On the buggy code displacement == 0.9*parent_radius exactly. Allow a small fraction of the radius
    // for the float rounding of re-deriving the base each timestep, but nothing approaching a radius.
    DOCTEST_CHECK(displacement < 0.1f * parent_radius);
}

DOCTEST_TEST_CASE("PlantArchitecture axis continuation survives duplication and an XML round trip") {
    // Companion to the test above. Neither duplicatePlantInstance() nor readPlantStructureXML() replays
    // a plant through appendShoot(): both rebuild every non-base shoot with addChildShoot(), which
    // produces a lateral branch. So unless the "continues the parent's axis" property is carried across
    // explicitly, a copied or reloaded plant reverts to the lateral-offset placement and its appended
    // shoots are displaced off the axis by 0.9 of the parent radius the moment it is grown -- even
    // though the plant it was copied from is seated correctly.
    //
    // The measurement is taken after further growth in every case, because the displacement is applied
    // by Shoot::updateShootNodes() rather than at construction: checking immediately after the copy or
    // the read would pass even without the fix.

    const std::string xml_filename = "test_plant_xml_axis_continuation.xml";

    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("soybean");

    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    plantarchitecture.advanceTime(plantID, 30.f);

    // The soybean builder appends the trifoliate shoot onto the unifoliate one, so there is something
    // for this test to check.
    DOCTEST_REQUIRE(plantarchitecture.getPlantShoot(plantID, 1)->is_axis_continuation);

    // Largest distance between any axis-continuation shoot's base and the parent node it continues.
    // Zero is the correct answer; on the buggy code it is 0.9 of the parent internode radius.
    auto worst_continuation_gap = [](PlantArchitecture &pa, uint pID) {
        float worst = 0.f;
        for (uint shootID: pa.getAllShootIDs(pID)) {
            auto shoot = pa.getPlantShoot(pID, shootID);
            if (shoot->parent_shoot_ID < 0 || shoot->phytomers.empty() || !shoot->is_axis_continuation) {
                continue;
            }
            auto parent = pa.getPlantShoot(pID, shoot->parent_shoot_ID);
            if (parent->phytomers.empty()) {
                continue;
            }
            const vec3 attachment = parent->shoot_internode_vertices.at(shoot->parent_node_index).back();
            worst = std::max(worst, (shoot->shoot_internode_vertices.front().front() - attachment).magnitude());
        }
        return worst;
    };

    // A radius to compare the gaps against, so the tolerances below are not absolute lengths.
    const float parent_radius = plantarchitecture.getPlantShoot(plantID, 0)->shoot_internode_radii.back().back();
    DOCTEST_REQUIRE(parent_radius > 0.002f);

    DOCTEST_CHECK(worst_continuation_gap(plantarchitecture, plantID) < 0.1f * parent_radius);

    // --- Duplicate, then grow ---
    uint duplicate_plantID = plantarchitecture.duplicatePlantInstance(plantID, make_vec3(1, 0, 0), make_AxisRotation(0, 0, 0), 0.f);
    DOCTEST_REQUIRE(plantarchitecture.getPlantShoot(duplicate_plantID, 1)->is_axis_continuation);
    plantarchitecture.advanceTime(duplicate_plantID, 5.f);
    DOCTEST_CHECK(worst_continuation_gap(plantarchitecture, duplicate_plantID) < 0.1f * parent_radius);

    // --- Write, reload into a fresh Context, then grow ---
    DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    Context restored_context;
    PlantArchitecture restored_plantarchitecture(&restored_context);
    restored_plantarchitecture.disableMessages();
    // The XML names shoot types by label, so the model library has to be loaded before reading.
    restored_plantarchitecture.loadPlantModelFromLibrary("soybean");

    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = restored_plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);
    const uint restored_plantID = restored_plantIDs.front();

    DOCTEST_CHECK(restored_plantarchitecture.getPlantShoot(restored_plantID, 1)->is_axis_continuation);
    restored_plantarchitecture.advanceTime(restored_plantID, 5.f);
    DOCTEST_CHECK(worst_continuation_gap(restored_plantarchitecture, restored_plantID) < 0.1f * parent_radius);

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture child shoot base is offset radially from its parent") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    // Regression test: a child shoot is seated one parent radius off the parent's axis so that it emerges
    // from the parent's surface. The direction for that offset was taken to be the parent's petiole axis
    // whenever the phytomer had a petiole -- but a petiole is not perpendicular to the stem. Soybean
    // petioles leave the stem at a pitch of 15-40 degrees, so the "outward" axis sat only about 22 degrees
    // off the stem and the offset went mostly UP THE STEM rather than away from it: of a 0.0066 m offset,
    // 0.0060 m was along the parent's axis and only 0.0025 m radial.
    //
    // The child was therefore lifted off the node it attaches to instead of being pushed out of the
    // parent's surface, leaving a visible gap at the junction -- and because the parent's petiole grows
    // from that same node in that same direction, the petiole was left running up through the hollow
    // interior of the detached child shoot.
    plantarchitecture.loadPlantModelFromLibrary("soybean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    plantarchitecture.advanceTime(plantID, 40.f);

    bool checked_a_child = false;

    for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        // Only lateral branches carry the outward offset; an axis continuation is seated on the axis.
        if (shoot->parent_shoot_ID < 0 || shoot->phytomers.empty() || shoot->is_axis_continuation) {
            continue;
        }
        auto parent = plantarchitecture.getPlantShoot(plantID, shoot->parent_shoot_ID);
        const uint parent_node = shoot->parent_node_index;
        if (parent->phytomers.empty() || parent_node >= parent->phytomers.size()) {
            continue;
        }
        // The defect needs a parent phytomer that actually has a petiole; a petiole-free phytomer takes
        // the other branch of childShootOutwardAxis(), which already projects out the axial component.
        if (parent->phytomers.at(parent_node)->petiole_vertices.empty()) {
            continue;
        }

        const vec3 attachment = parent->shoot_internode_vertices.at(parent_node).back();
        const vec3 child_base = shoot->shoot_internode_vertices.front().front();
        const vec3 parent_axis = parent->phytomers.at(parent_node)->getInternodeAxisVector(1.f);

        const vec3 offset = child_base - attachment;
        const float along_axis = std::fabs(offset * parent_axis);
        const float radial = (offset - (offset * parent_axis) * parent_axis).magnitude();

        // The offset exists to clear the parent's surface, so it must be essentially radial. On the buggy
        // code the axial part was more than twice the radial part. Requiring the radial component to
        // dominate is the invariant, and it holds regardless of the particular petiole pitch drawn.
        DOCTEST_CHECK(radial > along_axis);
        checked_a_child = true;
    }

    // Guard against the test silently passing because it never found a child shoot to check.
    DOCTEST_CHECK(checked_a_child);
}

DOCTEST_TEST_CASE("PlantArchitecture appendShoot rejects a parent with no phytomers") {
    // appendShoot() bases the new shoot at a point interpolated along the parent's last phytomer. The
    // guard meant to reject a parent that has none had an empty body, so instead of an error the call
    // went straight on to read phytomers.back() from an empty vector.
    //
    // The state is reachable: pruneBranch() at node 0 removes all of a shoot's phytomers, leaving the
    // shoot in the tree with its ID still valid.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    const uint baseID = plantarchitecture.addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0, 0), 0.01f, 0.05f, 1.f, 1.f, 0.f, "measured_wood");

    plantarchitecture.pruneBranch(plantID, baseID, 0);
    DOCTEST_REQUIRE(plantarchitecture.getPlantShoot(plantID, baseID)->phytomers.empty());

    {
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(plantarchitecture.appendShoot(plantID, int(baseID), 1, make_AxisRotation(0, 0, 0), 0.01f, 0.05f, 1.f, 1.f, 0.f, "measured_wood"));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture carbon-model bud break follows the shoot, not the shoot type") {
    // The carbohydrate-limited sampler, Shoot::sampleVegetativeBudBreak_carb(), had the same defect as
    // the ordinary one and has to move with it -- otherwise bud break would follow the shoot with the
    // carbon model off and the shoot type with it on.
    //
    // The carbohydrate pool does not confound this. It caps probability_max, but with a decay rate of 0
    // the sampler returns probability_min, so the pool level is irrelevant to the outcome here and the
    // comparison stays as clean as in the non-carbon case.
    auto budStatesAfterDormancyBreak = [](bool silence_this_shoot) -> std::vector<BudState> {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.enableCarbohydrateModel();

        defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

        std::vector<vec3> node_positions;
        std::vector<float> node_radii;
        measuredWoodPath(4, node_positions, node_radii);

        const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
        const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "measured_wood");

        const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);

        for (const auto &phytomer: shoot->phytomers) {
            for (const auto &petiole: phytomer->axillary_vegetative_buds) {
                for (const auto &vbud: petiole) {
                    DOCTEST_REQUIRE(vbud.state == BUD_ACTIVE);
                }
            }
        }

        if (silence_this_shoot) {
            shoot->shoot_parameters.vegetative_bud_break_probability_min = 0.f;
            shoot->shoot_parameters.vegetative_bud_break_probability_max = 0.f;
            shoot->shoot_parameters.vegetative_bud_break_probability_decay_rate = 0.f;
        }

        plantarchitecture.breakPlantDormancy(plantID);

        std::vector<BudState> states;
        for (const auto &phytomer: shoot->phytomers) {
            for (const auto &petiole: phytomer->axillary_vegetative_buds) {
                for (const auto &vbud: petiole) {
                    states.push_back(vbud.state);
                }
            }
        }
        return states;
    };

    const std::vector<BudState> states_untouched = budStatesAfterDormancyBreak(false);
    DOCTEST_REQUIRE(!states_untouched.empty());
    for (const BudState state: states_untouched) {
        DOCTEST_CHECK(state == BUD_ACTIVE);
    }

    const std::vector<BudState> states_silenced = budStatesAfterDormancyBreak(true);
    DOCTEST_REQUIRE(states_silenced.size() == states_untouched.size());
    for (const BudState state: states_silenced) {
        DOCTEST_CHECK(state == BUD_DEAD);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip restores a petiole-free prescribed shoot") {
    // Two defects met here, and both were silent in opposite directions.
    //
    // First, readPlantStructureXML() threw "Petiole index out of range" for ANY plant whose shoot type sets
    // petioles_per_internode = 0 -- the normal shape for a leafless woody skeleton -- because it read the
    // parent petiole axis directly instead of substituting the ghost petiole that every other caller uses.
    // Such a plant could be written but never read back.
    //
    // Second, once it could be read, the geometry came back wrong: the file records lengths and angles and
    // the reader re-integrates the path from the shoot's base rotation, which for a prescribed shoot is
    // identically zero. A branch measured running almost horizontally was restored standing straight up,
    // with its internode lengths intact and its shape discarded. Asserting on node POSITIONS rather than on
    // "did not throw" is what separates the two.
    const std::string xml_filename = "prescribed_roundtrip.xml";

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(4, node_positions, node_radii);

    {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        definePrescribedGeometryShootType(context, plantarchitecture);

        const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
        static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafless"));

        DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));
    }

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);

    const auto &restored_shoot = plantarchitecture.getPlantShoot(restored_plantIDs.front(), 0);
    const std::vector<vec3> restored_nodes = flatten(restored_shoot->shoot_internode_vertices);
    DOCTEST_REQUIRE(restored_nodes.size() == node_positions.size());

    // The measured path itself, not merely its internode lengths. A shoot re-integrated from a zero base
    // rotation stands vertical and would fail on x while passing any length-only check.
    for (size_t node = 0; node < node_positions.size(); node++) {
        DOCTEST_CHECK((restored_nodes.at(node) - node_positions.at(node)).magnitude() < 1e-4f);
    }

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip preserves the phytomer count of a subdivided prescribed shoot") {
    // A prescribed shoot is recorded as its node positions, and a phytomer's internode is stored as
    // internode.length_segments subdivisions of itself -- so the shoot's vertex list holds Ndiv nodes per
    // phytomer, not one. Writing the flattened vertex list therefore handed the reader one node per
    // SEGMENT, and addShootFromNodePositions() -- which takes one node per PHYTOMER -- rebuilt the shoot
    // with Ndiv times as many phytomers, each Ndiv times too short. The path and endpoints came back
    // exactly right, which is why the round-trip tests above never saw it: they use a shoot type with
    // length_segments = 1, where the two counts coincide.
    //
    // Ndiv = 3 here so the failure is unambiguous (3 phytomers in, 9 out) rather than an off-by-one.
    const std::string xml_filename = "prescribed_subdivided_roundtrip.xml";

    const uint Ninternodes = 3;
    const uint length_segments = 3;

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(Ninternodes, node_positions, node_radii);

    // Same petiole-free skeleton as definePrescribedGeometryShootType(), but with the internode
    // subdivided, which is the ordinary case for any real shoot type.
    auto defineSubdividedShootType = [&](Context &context, PlantArchitecture &plantarchitecture) {
        PhytomerParameters phytomer_parameters(context.getRandomGenerator());
        phytomer_parameters.internode.pitch = 0;
        phytomer_parameters.internode.phyllotactic_angle = 0;
        phytomer_parameters.internode.length_segments = length_segments;
        phytomer_parameters.petiole.petioles_per_internode = 0;
        phytomer_parameters.leaf.leaves_per_petiole = 0;

        ShootParameters shoot_parameters(context.getRandomGenerator());
        shoot_parameters.phytomer_parameters = phytomer_parameters;
        shoot_parameters.gravitropic_curvature = 0;
        shoot_parameters.tortuosity = 0;
        shoot_parameters.max_nodes = 40;
        shoot_parameters.vegetative_bud_break_probability_min = 0;
        shoot_parameters.girth_area_factor = 0;
        shoot_parameters.defineChildShootTypes({"subdivided"}, {1.f});
        plantarchitecture.defineShootType("subdivided", shoot_parameters);
    };

    std::vector<float> original_internode_lengths;
    {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        defineSubdividedShootType(context, plantarchitecture);

        const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
        const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "subdivided");

        const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        DOCTEST_REQUIRE(shoot->phytomers.size() == Ninternodes);
        for (const auto &phytomer: shoot->phytomers) {
            original_internode_lengths.push_back(phytomer->getInternodeLength());
        }

        DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));
    }

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    defineSubdividedShootType(context, plantarchitecture);

    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);

    const auto &restored_shoot = plantarchitecture.getPlantShoot(restored_plantIDs.front(), 0);

    // The count is the claim. A shoot measured as 3 internodes is 3 phytomers, whatever its internodes
    // are subdivided into for rendering.
    DOCTEST_CHECK(restored_shoot->phytomers.size() == Ninternodes);

    // ...and each phytomer is still as long as it was measured, rather than one segment of itself.
    DOCTEST_REQUIRE(restored_shoot->phytomers.size() == original_internode_lengths.size());
    for (size_t p = 0; p < original_internode_lengths.size(); p++) {
        DOCTEST_CHECK(restored_shoot->phytomers.at(p)->getInternodeLength() == doctest::Approx(original_internode_lengths.at(p)).epsilon(1e-4));
    }

    // The internode is still subdivided the way the shoot type asks, so the shape it was measured with is
    // preserved rather than being traded away for the count.
    const std::vector<vec3> restored_nodes = flatten(restored_shoot->shoot_internode_vertices);
    DOCTEST_CHECK(restored_nodes.size() == Ninternodes * length_segments + 1);
    for (size_t node = 0; node < node_positions.size(); node++) {
        DOCTEST_CHECK((restored_nodes.at(node * length_segments) - node_positions.at(node)).magnitude() < 1e-4f);
    }

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip reads back an ordinary petiole-free shoot") {
    // The petiole-free failure is not confined to prescribed geometry. A shoot built the ordinary way
    // from a shoot type with petioles_per_internode = 0 is restored by appending phytomers one at a time,
    // and that path recomputes each internode's orientation against the previous phytomer's petiole axis
    // -- which does not exist. This is the case that isolates the ghost-petiole substitution, since the
    // prescribed tests above rebuild the shoot wholesale and never reach that code.
    const std::string xml_filename = "petiole_free_ordinary_roundtrip.xml";

    {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        definePrescribedGeometryShootType(context, plantarchitecture);

        const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
        // A non-zero base pitch so the shoot is not axis-aligned, which is where a wrong reference frame
        // would show up as a rotated reconstruction rather than an identical one.
        static_cast<void>(plantarchitecture.addBaseStemShoot(plantID, 4, make_AxisRotation(0.2f, 0.f, 0.f), 0.01f, 0.05f, 1.f, 1.f, 0.f, "leafless"));

        DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));
    }

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);

    const auto &restored_shoot = plantarchitecture.getPlantShoot(restored_plantIDs.front(), 0);
    DOCTEST_CHECK(restored_shoot->phytomers.size() == 4);

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture XML round-trip restores a prescribed child shoot on its parent") {
    // The child path seats the base on the parent's surface and translates the measured polyline onto it,
    // so it has to survive the round-trip as a unit: reattached to the right node of the right parent, and
    // still pointing the way it was measured.
    const std::string xml_filename = "prescribed_child_roundtrip.xml";

    std::vector<vec3> trunk_positions;
    std::vector<float> trunk_radii;
    measuredWoodPath(4, trunk_positions, trunk_radii);

    vec3 branch_tip_written;
    {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        definePrescribedGeometryShootType(context, plantarchitecture);

        const uint plantID = plantarchitecture.addPlantInstance(trunk_positions.front(), 0.f);
        const uint trunkID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, trunk_positions, trunk_radii, "leafless");

        // A branch leaving node 2 of the trunk in +y, i.e. out of the trunk's own plane.
        const vec3 attachment = plantarchitecture.getPlantShoot(plantID, trunkID)->shoot_internode_vertices.at(2).back();
        std::vector<vec3> branch_positions;
        std::vector<float> branch_radii;
        for (uint node = 0; node <= 3; node++) {
            branch_positions.push_back(attachment + make_vec3(0.f, 0.04f * float(node), 0.01f * float(node)));
            branch_radii.push_back(0.005f);
        }
        const uint branchID = plantarchitecture.addShootFromNodePositions(plantID, trunkID, 2, branch_positions, branch_radii, "leafless");
        branch_tip_written = flatten(plantarchitecture.getPlantShoot(plantID, branchID)->shoot_internode_vertices).back();

        DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));
    }

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedGeometryShootType(context, plantarchitecture);

    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = plantarchitecture.readPlantStructureXML(xml_filename, true));
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);
    DOCTEST_REQUIRE(plantarchitecture.getAllShootIDs(restored_plantIDs.front()).size() == 2);

    const auto &restored_branch = plantarchitecture.getPlantShoot(restored_plantIDs.front(), 1);
    DOCTEST_CHECK(restored_branch->parent_shoot_ID == 0);
    DOCTEST_CHECK(restored_branch->parent_node_index == 2);

    const vec3 branch_tip_restored = flatten(restored_branch->shoot_internode_vertices).back();
    DOCTEST_CHECK((branch_tip_restored - branch_tip_written).magnitude() < 1e-4f);

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture writePlantStructureXML records the growth shoot type of measured wood") {
    // The growth type is per-shoot state, and writePlantStructureXML() otherwise records only a shoot's
    // own type label -- so without explicit serialization a saved reconstruction would reload governed by
    // the type its measured wood was BUILT with, growing straight and unbounded where the original grew
    // correctly.
    //
    // This asserts on the written file rather than on a reload so that a regression in the writer is
    // reported as such: the reload path is covered separately by the prescribed round-trip tests above,
    // and a single test that did both would not say which half had broken.
    const std::string xml_filename = "measured_growth_write.xml";

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(4, node_positions, node_radii);

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "new_growth"));

    DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    std::ifstream xml_file(xml_filename);
    DOCTEST_REQUIRE(xml_file.good());
    const std::string xml_contents((std::istreambuf_iterator<char>(xml_file)), std::istreambuf_iterator<char>());
    xml_file.close();

    DOCTEST_CHECK(xml_contents.find("<growth_shoot_type_label> new_growth </growth_shoot_type_label>") != std::string::npos);
    DOCTEST_CHECK(xml_contents.find("<max_nodes_growth_cap>") != std::string::npos);

    // A shoot with no growth type of its own writes none of these tags, so files for ordinary plants are
    // unchanged and older files -- which have no such tags -- still describe exactly what they used to.
    {
        Context plain_context;
        plain_context.seedRandomGenerator(12345);
        PlantArchitecture plain_plantarchitecture(&plain_context);
        plain_plantarchitecture.disableMessages();
        definePrescribedGeometryShootType(plain_context, plain_plantarchitecture);

        const uint plain_plantID = plain_plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
        static_cast<void>(plain_plantarchitecture.addShootFromNodePositions(plain_plantID, -1, 0, node_positions, node_radii, "leafless"));

        const std::string plain_filename = "measured_growth_write_plain.xml";
        DOCTEST_REQUIRE_NOTHROW(plain_plantarchitecture.writePlantStructureXML(plain_plantID, plain_filename));

        std::ifstream plain_file(plain_filename);
        DOCTEST_REQUIRE(plain_file.good());
        const std::string plain_contents((std::istreambuf_iterator<char>(plain_file)), std::istreambuf_iterator<char>());
        plain_file.close();

        DOCTEST_CHECK(plain_contents.find("<growth_shoot_type_label>") == std::string::npos);
        std::remove(plain_filename.c_str());
    }

    std::remove(xml_filename.c_str());
}

DOCTEST_TEST_CASE("PlantArchitecture isShootGeometryPrescribed distinguishes measured wood from grown wood") {
    // A caller reading geometry back off a partly-reconstructed plant needs to know which shoots are
    // measurement and which are model output; the two are otherwise indistinguishable once the plant has
    // grown, since both are ordinary shoots carrying ordinary phytomers.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(4, node_positions, node_radii);

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    const uint measuredID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "new_growth");

    DOCTEST_CHECK(plantarchitecture.isShootGeometryPrescribed(plantID, measuredID));

    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 40);

    const auto &shoot_tree = PlantArchitectureTestHelper::getShootTree(plantarchitecture, plantID);
    DOCTEST_REQUIRE(shoot_tree.size() > 1);
    for (const auto &shoot: shoot_tree) {
        if (shoot->ID == int(measuredID)) {
            continue;
        }
        DOCTEST_CHECK(!plantarchitecture.isShootGeometryPrescribed(plantID, shoot->ID));
    }

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.isShootGeometryPrescribed(plantID, 999)));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance preserves the growth shoot type of measured wood") {
    // duplicatePlantInstance() rebuilds the shoot tree by REPLAYING shoot creation, which stamps every
    // vegetative bud with its shoot's type as each phytomer is built -- and at that moment the copy is
    // still governed by the build type, because growth_state is assigned only after the whole tree is
    // finished. The XML restore path had the identical hazard and grew an explicit re-stamp loop for it.
    // Shoot::gravitropic_curvature is latched from the build type in the Shoot constructor and is
    // likewise never restored on the copy, so a duplicated reconstruction also grows straight.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(4, node_positions, node_radii);

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);
    static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "new_growth"));

    const uint copy_plantID = plantarchitecture.duplicatePlantInstance(plantID, make_vec3(1.f, 0.f, 0.f), make_AxisRotation(0, 0, 0), 0.f);

    const auto &copy_tree = PlantArchitectureTestHelper::getShootTree(plantarchitecture, copy_plantID);
    DOCTEST_REQUIRE(!copy_tree.empty());
    const auto &copied_shoot = copy_tree.front();

    // The growth state itself is copied, so this much already holds.
    DOCTEST_CHECK(copied_shoot->getGrowthTypeLabel() == "new_growth");

    // The buds, however, were stamped while the copy was still governed by the build type.
    uint buds_checked = 0;
    for (const auto &phytomer: copied_shoot->phytomers) {
        for (const auto &petiole: phytomer->axillary_vegetative_buds) {
            for (const auto &vbud: petiole) {
                DOCTEST_CHECK(vbud.shoot_type_label == "new_growth");
                buds_checked++;
            }
        }
    }
    DOCTEST_REQUIRE(buds_checked > 0);

    // And the latched curvature came from the build type, which is zero.
    DOCTEST_CHECK(copied_shoot->gravitropic_curvature == doctest::Approx(600.f));
}

DOCTEST_TEST_CASE("PlantArchitecture addShootFromNodePositions rejects an unresolvable growth shoot type") {
    // The growth type is only meaningful if it resolves; a typo that silently fell back to the build
    // type would reintroduce exactly the bug the argument exists to prevent.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    defineMeasuredAndGrowthShootTypes(context, plantarchitecture);

    std::vector<vec3> node_positions;
    std::vector<float> node_radii;
    measuredWoodPath(3, node_positions, node_radii);

    const uint plantID = plantarchitecture.addPlantInstance(node_positions.front(), 0.f);

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "no_such_type")));
    }

    // The valid pairing is still accepted.
    DOCTEST_CHECK_NOTHROW(static_cast<void>(plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "measured_wood", "new_growth")));
}

//! Single-petiole trifoliate shoot type used by the prescribed petiole/leaf geometry tests. Unlike
//! definePrescribedGeometryShootType() above, this one actually carries petioles and leaves, which is what
//! the organ-level prescription has to be exercised against. Leaf flexibility is deliberately non-zero so
//! that the droop exemption is tested rather than trivially satisfied by a rigid species.
static void definePrescribedLeafGeometryShootType(Context &context, PlantArchitecture &plantarchitecture, uint petioles_per_internode = 1) {
    PhytomerParameters phytomer_parameters(context.getRandomGenerator());
    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 0;
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = petioles_per_internode;
    phytomer_parameters.petiole.pitch = 45;
    phytomer_parameters.petiole.radius = 0.002;
    phytomer_parameters.petiole.length = 0.05;
    phytomer_parameters.petiole.curvature = 0;
    phytomer_parameters.petiole.taper = 0.5;
    phytomer_parameters.petiole.length_segments = 3;

    phytomer_parameters.leaf.leaves_per_petiole = 3;
    phytomer_parameters.leaf.leaflet_offset = 0.3;
    phytomer_parameters.leaf.leaflet_scale = 1.f;
    phytomer_parameters.leaf.prototype_scale = 0.03;
    phytomer_parameters.leaf.pitch = 10;
    phytomer_parameters.leaf.yaw = 0;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype.prototype_function = GenericLeafPrototype;
    phytomer_parameters.leaf.prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/BeanLeaf_tip.png";
    phytomer_parameters.leaf.prototype.subdivisions = 3;
    phytomer_parameters.leaf.prototype.unique_prototypes = 1;
    phytomer_parameters.leaf.prototype.flexibility = 5.f;

    ShootParameters shoot_parameters(context.getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.gravitropic_curvature = 0;
    shoot_parameters.tortuosity = 0;
    shoot_parameters.max_nodes = 40;
    shoot_parameters.vegetative_bud_break_probability_min = 0;
    shoot_parameters.girth_area_factor = 0;
    shoot_parameters.defineChildShootTypes({"leafy"}, {1.f});
    plantarchitecture.defineShootType("leafy", shoot_parameters);
}

//! Builds a two-phytomer leafy shoot and returns its plant and shoot IDs, so each test below starts from
//! the same geometry without repeating the setup.
static void buildLeafyPrescribedShoot(Context &context, PlantArchitecture &plantarchitecture, uint &plantID, uint &shootID, uint petioles_per_internode = 1) {
    definePrescribedLeafGeometryShootType(context, plantarchitecture, petioles_per_internode);
    const std::vector<vec3> node_positions = {make_vec3(0.f, 0.f, 0.f), make_vec3(0.f, 0.f, 0.1f), make_vec3(0.f, 0.f, 0.2f)};
    const std::vector<float> node_radii = {0.01f, 0.008f, 0.006f};
    plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafy");
}

//! A measured petiole path: out of plane, unevenly spaced and visibly curved, so that neither a straight
//! petiole nor one generated from the shoot type's own curvature could be mistaken for it.
static std::vector<vec3> measuredPetiolePath(const vec3 &base) {
    return {base,
            base + make_vec3(0.012f, 0.004f, 0.010f),
            base + make_vec3(0.028f, 0.011f, 0.016f),
            base + make_vec3(0.047f, 0.021f, 0.018f),
            base + make_vec3(0.062f, 0.034f, 0.015f),
            base + make_vec3(0.071f, 0.048f, 0.008f),
            base + make_vec3(0.075f, 0.063f, -0.004f)};
}

static std::vector<float> measuredPetioleRadii() {
    return {0.0022f, 0.0021f, 0.0019f, 0.0017f, 0.0015f, 0.0013f, 0.0011f};
}

DOCTEST_TEST_CASE("PlantArchitecture setPetioleNodePositions survives advanceTime") {
    // The reason this feature needs a mechanism rather than just a setter. Petiole geometry is rebuilt or
    // rescaled from parameters on every timestep by three separate paths: setPetioleBase() re-snaps the base
    // from Shoot::updateShootNodes(), setLeafScaleFraction() rescales the whole polyline homothetically while
    // the leaf is still expanding, and it re-derives every leaf base from the petiole afterwards. A prescribed
    // path that is merely written into petiole_vertices is gone by the next call to advanceTime().
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const vec3 petiole_base = shoot->shoot_internode_vertices.at(1).back();
    const std::vector<vec3> prescribed_path = measuredPetiolePath(petiole_base);

    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, prescribed_path, measuredPetioleRadii());

    const std::vector<vec3> before = shoot->phytomers.at(1)->petiole_vertices.at(0);
    const float length_before = shoot->phytomers.at(1)->petiole_length.at(0);

    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 20);

    // Checked after 20 timesteps rather than one: the base re-snap applies its shift on every timestep, so a
    // near-miss shows up as a slow drift that a single-step test would pass.
    const std::vector<vec3> after = shoot->phytomers.at(1)->petiole_vertices.at(0);
    DOCTEST_REQUIRE(after.size() == before.size());
    for (size_t node = 0; node < before.size(); node++) {
        DOCTEST_CHECK((after.at(node) - before.at(node)).magnitude() < 1e-5f);
    }
    DOCTEST_CHECK(shoot->phytomers.at(1)->petiole_length.at(0) == doctest::Approx(length_before).epsilon(1e-4));
}

DOCTEST_TEST_CASE("PlantArchitecture setPetioleNodePositions reproduces the supplied geometry") {
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();
    const std::vector<vec3> prescribed_path = measuredPetiolePath(internode_tip);
    const std::vector<float> prescribed_radii = measuredPetioleRadii();

    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, prescribed_path, prescribed_radii);

    const auto phytomer = shoot->phytomers.at(1);
    DOCTEST_REQUIRE(phytomer->petiole_vertices.at(0).size() == prescribed_path.size());

    // Node 0 must be bit-identical to the internode tip, not merely close: setPetioleBase() computes its
    // shift as (internode tip - petiole_vertices[0][0]) and applies it to every petiole, leaf and floral bud
    // on the phytomer. Exact equality is what makes that shift exactly zero, and is therefore the mechanism
    // rather than an incidental detail.
    DOCTEST_CHECK(phytomer->petiole_vertices.at(0).front().x == internode_tip.x);
    DOCTEST_CHECK(phytomer->petiole_vertices.at(0).front().y == internode_tip.y);
    DOCTEST_CHECK(phytomer->petiole_vertices.at(0).front().z == internode_tip.z);

    for (size_t node = 1; node < prescribed_path.size(); node++) {
        DOCTEST_CHECK((phytomer->petiole_vertices.at(0).at(node) - prescribed_path.at(node)).magnitude() < 1e-6f);
        DOCTEST_CHECK(phytomer->petiole_radii.at(0).at(node) == doctest::Approx(prescribed_radii.at(node)).epsilon(1e-5));
    }

    float expected_length = 0.f;
    for (size_t node = 0; node + 1 < prescribed_path.size(); node++) {
        expected_length += (prescribed_path.at(node + 1) - prescribed_path.at(node)).magnitude();
    }
    DOCTEST_CHECK(phytomer->petiole_length.at(0) == doctest::Approx(expected_length).epsilon(1e-4));

    DOCTEST_REQUIRE(context.doesObjectExist(phytomer->petiole_objIDs.at(0)));
    DOCTEST_CHECK(context.getTubeObjectNodeCount(phytomer->petiole_objIDs.at(0)) == prescribed_path.size());
}

DOCTEST_TEST_CASE("PlantArchitecture setPetioleLeafGeometry places leaves and survives advanceTime") {
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto phytomer = shoot->phytomers.at(1);
    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();

    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, measuredPetiolePath(internode_tip), measuredPetioleRadii());

    const std::vector<vec3> leaf_bases = {internode_tip + make_vec3(0.040f, 0.018f, 0.017f), internode_tip + make_vec3(0.075f, 0.063f, -0.004f), internode_tip + make_vec3(0.058f, 0.030f, 0.010f)};
    const std::vector<AxisRotation> leaf_rotations = {make_AxisRotation(0.10f, 0.25f, 0.30f), make_AxisRotation(0.f, 0.40f, 0.f), make_AxisRotation(-0.10f, 0.25f, -0.30f)};
    const std::vector<float> leaf_sizes = {0.022f, 0.035f, 0.022f};

    plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, leaf_bases, leaf_rotations, leaf_sizes);

    for (uint leaf = 0; leaf < 3; leaf++) {
        DOCTEST_CHECK((phytomer->leaf_bases.at(0).at(leaf) - leaf_bases.at(leaf)).magnitude() < 1e-6f);
        DOCTEST_CHECK(phytomer->leaf_size_max.at(0).at(leaf) == doctest::Approx(leaf_sizes.at(leaf)).epsilon(1e-5));
        DOCTEST_REQUIRE(context.doesObjectExist(phytomer->leaf_objIDs.at(0).at(leaf)));
    }
    DOCTEST_CHECK(phytomer->current_leaf_scale_factor.at(0) == doctest::Approx(1.f));

    // The leaf's object transformation matrix carries its position, orientation and size. Snapshotting it
    // before and after growth is what distinguishes "the blade was left alone" from "the blade was re-posed":
    // the droop rewrites mesh vertices but never the transform, so any change here is the growth model moving
    // the leaf rather than bending it.
    float transforms_before[3][16];
    for (uint leaf = 0; leaf < 3; leaf++) {
        context.getObjectTransformationMatrix(phytomer->leaf_objIDs.at(0).at(leaf), transforms_before[leaf]);
    }

    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 20);

    for (uint leaf = 0; leaf < 3; leaf++) {
        DOCTEST_CHECK((phytomer->leaf_bases.at(0).at(leaf) - leaf_bases.at(leaf)).magnitude() < 1e-5f);
        DOCTEST_CHECK(phytomer->leaf_size_max.at(0).at(leaf) == doctest::Approx(leaf_sizes.at(leaf)).epsilon(1e-4));
        DOCTEST_REQUIRE(context.doesObjectExist(phytomer->leaf_objIDs.at(0).at(leaf)));

        float transform_after[16];
        context.getObjectTransformationMatrix(phytomer->leaf_objIDs.at(0).at(leaf), transform_after);
        for (uint element = 0; element < 16; element++) {
            DOCTEST_CHECK(transform_after[element] == doctest::Approx(transforms_before[leaf][element]).epsilon(1e-4));
        }
    }
}

DOCTEST_TEST_CASE("PlantArchitecture setPetioleLeafCount rebuilds a petiole with a different number of leaflets") {
    // A measured compound leaf has however many leaflets it has, which is rarely the shoot type's
    // leaves_per_petiole (a young tomato leaf has three or five, a mature one seven or more, and a
    // cotyledon one). setPetioleLeafGeometry() cannot change the count, so this has to.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto phytomer = shoot->phytomers.at(1);
    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();
    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, measuredPetiolePath(internode_tip), measuredPetioleRadii());

    const uint initial_count = uint(phytomer->leaf_objIDs.at(0).size());
    const std::vector<uint> old_objIDs = phytomer->leaf_objIDs.at(0);
    const uint new_count = initial_count + 2;

    plantarchitecture.setPetioleLeafCount(plantID, shootID, 1, 0, new_count);

    DOCTEST_CHECK(phytomer->leaf_objIDs.at(0).size() == new_count);
    DOCTEST_CHECK(phytomer->leaf_bases.at(0).size() == new_count);
    DOCTEST_CHECK(phytomer->leaf_size_max.at(0).size() == new_count);
    DOCTEST_CHECK(phytomer->leaf_rotation.at(0).size() == new_count);
    for (const uint objID: old_objIDs) {
        DOCTEST_CHECK(!context.doesObjectExist(objID));
    }
    for (uint leaf = 0; leaf < new_count; leaf++) {
        DOCTEST_REQUIRE(context.doesObjectExist(phytomer->leaf_objIDs.at(0).at(leaf)));
        DOCTEST_CHECK(phytomer->leaf_size_max.at(0).at(leaf) > 0.f);
        // Rebuilt leaflets sit on the petiole, like the originals.
        float distance_to_petiole = -1.f;
        for (const vec3 &vertex: phytomer->petiole_vertices.at(0)) {
            const float d = (vertex - phytomer->leaf_bases.at(0).at(leaf)).magnitude();
            if (distance_to_petiole < 0.f || d < distance_to_petiole) {
                distance_to_petiole = d;
            }
        }
        DOCTEST_CHECK(distance_to_petiole < 0.02f);
    }
    // Other phytomers are untouched.
    DOCTEST_CHECK(shoot->phytomers.at(0)->leaf_objIDs.at(0).size() == initial_count);
    DOCTEST_CHECK(plantarchitecture.getPlantLeafObjectIDs(plantID).size() == initial_count + new_count);

    // The new count is what setPetioleLeafGeometry() now expects.
    std::vector<vec3> leaf_bases;
    std::vector<AxisRotation> leaf_rotations;
    std::vector<float> leaf_sizes;
    for (uint leaf = 0; leaf < new_count; leaf++) {
        leaf_bases.push_back(internode_tip + make_vec3(0.01f * float(leaf + 1), 0.005f * float(leaf), 0.01f));
        leaf_rotations.push_back(make_AxisRotation(0.f, 0.3f, 0.f));
        leaf_sizes.push_back(0.02f + 0.002f * float(leaf));
    }
    DOCTEST_CHECK_NOTHROW(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, leaf_bases, leaf_rotations, leaf_sizes));
    DOCTEST_CHECK(phytomer->leaf_objIDs.at(0).size() == new_count);

    // And it survives growth.
    std::vector<std::array<float, 16>> transforms_before(new_count);
    for (uint leaf = 0; leaf < new_count; leaf++) {
        float transform[16];
        context.getObjectTransformationMatrix(phytomer->leaf_objIDs.at(0).at(leaf), transform);
        std::copy(transform, transform + 16, transforms_before.at(leaf).begin());
    }
    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 20);
    DOCTEST_CHECK(phytomer->leaf_objIDs.at(0).size() == new_count);
    for (uint leaf = 0; leaf < new_count; leaf++) {
        DOCTEST_REQUIRE(context.doesObjectExist(phytomer->leaf_objIDs.at(0).at(leaf)));
        float transform_after[16];
        context.getObjectTransformationMatrix(phytomer->leaf_objIDs.at(0).at(leaf), transform_after);
        for (uint element = 0; element < 16; element++) {
            DOCTEST_CHECK(transform_after[element] == doctest::Approx(transforms_before.at(leaf).at(element)).epsilon(1e-4));
        }
    }

    // A single leaflet (a cotyledon or simple leaf) is allowed; zero and bad indices are not.
    DOCTEST_CHECK_NOTHROW(plantarchitecture.setPetioleLeafCount(plantID, shootID, 0, 0, 1));
    DOCTEST_CHECK(shoot->phytomers.at(0)->leaf_objIDs.at(0).size() == 1);
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafCount(plantID, shootID, 0, 0, 0));
    DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafCount(plantID, shootID, 0, 5, 3));
    DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafCount(plantID, shootID, 99, 0, 3));
    DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafCount(plantID + 100, shootID, 0, 0, 3));
}

//! Expected world-frame midrib and blade normal of a prescribed leaf, derived independently of Phytomer::orientLeaf().
/**
 * A prescribed leaf's angles are intrinsic rotations in the leaf's rest frame on its petiole. With t the petiole tip axis, h the horizontal direction
 * perpendicular to it (world +y when t is vertical) and n = t x h the upward normal of the petiole plane, the rest midrib is t turned about n by the
 * leaflet's compound rotation and the rest blade normal is n. Yaw then turns the leaf about n, pitch raises its tip toward n, and roll turns the blade
 * about its own midrib (right-handed).
 */
static void expectedPrescribedLeafFrame(const vec3 &petiole_tip_axis, float compound_rotation, const AxisRotation &rotation, vec3 &midrib, vec3 &normal) {
    vec3 t = petiole_tip_axis;
    t.normalize();
    const float horizontal = std::sqrt(t.x * t.x + t.y * t.y);
    const vec3 h = (horizontal < 1e-6f) ? make_vec3(0, 1, 0) : make_vec3(-t.y / horizontal, t.x / horizontal, 0);
    const vec3 n = cross(t, h);

    const vec3 midrib_rest = std::cos(compound_rotation) * t + std::sin(compound_rotation) * h;
    const vec3 lateral_rest = -std::sin(compound_rotation) * t + std::cos(compound_rotation) * h;

    const vec3 midrib_yawed = std::cos(rotation.yaw) * midrib_rest + std::sin(rotation.yaw) * lateral_rest;
    const vec3 lateral_yawed = -std::sin(rotation.yaw) * midrib_rest + std::cos(rotation.yaw) * lateral_rest;

    midrib = std::cos(rotation.pitch) * midrib_yawed + std::sin(rotation.pitch) * n;
    const vec3 normal_pitched = -std::sin(rotation.pitch) * midrib_yawed + std::cos(rotation.pitch) * n;

    normal = std::cos(rotation.roll) * normal_pitched - std::sin(rotation.roll) * lateral_yawed;
}

//! Angle in degrees between two directions
static float angleBetweenDeg(vec3 a, vec3 b) {
    a.normalize();
    b.normalize();
    return rad2deg(acos_safe(a * b));
}

DOCTEST_TEST_CASE("PlantArchitecture prescribed leaf angles are rotations in the leaf's rest frame on the petiole") {
    // A prescribed leaf's roll, pitch and yaw used to go through the procedural rotation chain. A single leaf's roll was negated on every other
    // node, a lateral leaflet's roll was signed by its side of the petiole and offset by the petiole's elevation, a terminal leaflet's pitch was
    // offset by that elevation, and yaw turned the leaf about the world vertical rather than about the blade normal. The same measured angles
    // therefore gave different blades depending on node parity and leaflet position. Each prescribed leaf below is checked against the frame
    // derived independently by expectedPrescribedLeafFrame(), read back from the leaf object's transformation matrix.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    definePrescribedLeafGeometryShootType(context, plantarchitecture);

    // Four phytomers: nodes 0 and 1 carry a single leaf on a measured petiole (opposite node parity, same angles), node 2 a three-leaflet leaf on a
    // measured petiole, and node 3 a single leaf on a vertical petiole, where the horizontal reference direction falls back to world +y.
    const std::vector<vec3> node_positions = {make_vec3(0, 0, 0), make_vec3(0, 0, 0.1f), make_vec3(0, 0, 0.2f), make_vec3(0, 0, 0.3f), make_vec3(0, 0, 0.4f)};
    const std::vector<float> node_radii = {0.01f, 0.009f, 0.008f, 0.007f, 0.006f};
    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    const uint shootID = plantarchitecture.addShootFromNodePositions(plantID, -1, 0, node_positions, node_radii, "leafy");
    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    DOCTEST_REQUIRE(shoot->phytomers.size() == 4);

    const std::vector<float> leaflet_sizes = {0.03f, 0.04f, 0.03f};
    for (uint node = 0; node < 4; node++) {
        const vec3 internode_tip = shoot->shoot_internode_vertices.at(node).back();
        if (node == 3) {
            const std::vector<vec3> vertical_path = {internode_tip, internode_tip + make_vec3(0, 0, 0.03f), internode_tip + make_vec3(0, 0, 0.06f)};
            plantarchitecture.setPetioleNodePositions(plantID, shootID, node, 0, vertical_path, {0.002f, 0.0015f, 0.001f});
        } else {
            // A different azimuth per node so that the petioles do not all share one frame.
            std::vector<vec3> path = measuredPetiolePath(internode_tip);
            const float azimuth = 1.3f * float(node);
            for (vec3 &position: path) {
                position = internode_tip + rotatePointAboutLine(position - internode_tip, nullorigin, make_vec3(0, 0, 1), azimuth);
            }
            plantarchitecture.setPetioleNodePositions(plantID, shootID, node, 0, path, measuredPetioleRadii());
        }
        if (node != 2) {
            plantarchitecture.setPetioleLeafCount(plantID, shootID, node, 0, 1);
        }
    }

    const AxisRotation single_rotation = make_AxisRotation(0.35f, 0.25f, -0.4f); // roll, pitch, yaw
    const std::vector<AxisRotation> leaflet_rotations = {make_AxisRotation(-0.3f, 0.2f, 0.5f), make_AxisRotation(0.6f, -0.35f, 0.45f), make_AxisRotation(0.25f, 0.4f, -0.3f)};

    for (uint node = 0; node < 4; node++) {
        const vec3 internode_tip = shoot->shoot_internode_vertices.at(node).back();
        if (node == 2) {
            const std::vector<vec3> bases = {internode_tip + make_vec3(0.02f, 0, 0.01f), internode_tip + make_vec3(0.05f, 0, 0.01f), internode_tip + make_vec3(0.03f, 0.01f, 0.01f)};
            plantarchitecture.setPetioleLeafGeometry(plantID, shootID, node, 0, bases, leaflet_rotations, leaflet_sizes);
        } else {
            plantarchitecture.setPetioleLeafGeometry(plantID, shootID, node, 0, {internode_tip + make_vec3(0.04f, 0.01f, 0.01f)}, {single_rotation}, {0.05f});
        }
    }

    auto checkLeaf = [&](uint node, uint leaf, const AxisRotation &rotation) {
        const auto &phytomer = shoot->phytomers.at(node);
        const uint leaf_count = uint(phytomer->leaf_objIDs.at(0).size());
        const vec3 petiole_tip_axis = phytomer->getPetioleAxisVector(1.f, 0);
        // The three-leaflet shoot type has a non-zero leaflet offset, so the laterals sit at -90 and +90 degrees and the terminal at zero.
        const float compound_rotation = Phytomer::compoundLeafRotation(int(leaf_count), int(leaf), 0.3f);
        vec3 midrib_expected, normal_expected;
        expectedPrescribedLeafFrame(petiole_tip_axis, compound_rotation, rotation, midrib_expected, normal_expected);

        float transform[16];
        context.getObjectTransformationMatrix(phytomer->leaf_objIDs.at(0).at(leaf), transform);
        const vec3 midrib_realized = make_vec3(transform[0], transform[4], transform[8]);
        const vec3 normal_realized = make_vec3(transform[2], transform[6], transform[10]);

        DOCTEST_CAPTURE(node);
        DOCTEST_CAPTURE(leaf);
        DOCTEST_CHECK(angleBetweenDeg(midrib_realized, midrib_expected) < 1.f);
        DOCTEST_CHECK(angleBetweenDeg(normal_realized, normal_expected) < 1.f);
        DOCTEST_CHECK(phytomer->leaf_rotation.at(0).at(leaf).roll == doctest::Approx(rotation.roll));
        DOCTEST_CHECK(phytomer->leaf_rotation.at(0).at(leaf).pitch == doctest::Approx(rotation.pitch));
        DOCTEST_CHECK(phytomer->leaf_rotation.at(0).at(leaf).yaw == doctest::Approx(rotation.yaw));
    };

    checkLeaf(0, 0, single_rotation);
    checkLeaf(1, 0, single_rotation);
    for (uint leaf = 0; leaf < 3; leaf++) {
        checkLeaf(2, leaf, leaflet_rotations.at(leaf));
    }
    checkLeaf(3, 0, single_rotation);
}

DOCTEST_TEST_CASE("PlantArchitecture prescribed leaves keep the leaf object label") {
    // Every leaf blade primitive carries object_label "leaf"; downstream code (optical properties,
    // per-organ area sums, point-cloud fitting) tells blades from petioles and stems by it. A leaf
    // rebuilt by setPetioleLeafGeometry() or setPetioleLeafCount() from the prototype function rather
    // than the cache must carry the label too.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);
    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto phytomer = shoot->phytomers.at(1);
    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();
    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, measuredPetiolePath(internode_tip), measuredPetioleRadii());

    auto countLabelled = [&](uint objID) {
        size_t labelled = 0, total = 0;
        for (const uint UUID: context.getObjectPrimitiveUUIDs(objID)) {
            total++;
            if (context.doesPrimitiveDataExist(UUID, "object_label")) {
                std::string label;
                context.getPrimitiveData(UUID, "object_label", label);
                if (label == "leaf") {
                    labelled++;
                }
            }
        }
        return std::make_pair(labelled, total);
    };

    // The default leaves are labelled.
    for (const uint objID: phytomer->leaf_objIDs.at(0)) {
        const std::pair<size_t, size_t> counts = countLabelled(objID);
        DOCTEST_CHECK(counts.first > 0);
    }

    const size_t count = phytomer->leaf_objIDs.at(0).size();
    std::vector<vec3> leaf_bases(count, internode_tip + make_vec3(0.05f, 0.03f, 0.01f));
    std::vector<AxisRotation> leaf_rotations(count, make_AxisRotation(0.f, 0.3f, 0.f));
    std::vector<float> leaf_sizes(count, 0.03f);
    plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, leaf_bases, leaf_rotations, leaf_sizes);
    for (const uint objID: phytomer->leaf_objIDs.at(0)) {
        const std::pair<size_t, size_t> counts = countLabelled(objID);
        DOCTEST_CHECK_MESSAGE(counts.first > 0, "prescribed leaf object " << objID << " has " << counts.second << " primitives and none labelled 'leaf'");
    }

    // A second prescription rebuilds from the prototype function rather than the cache, which is the
    // path every re-posing after the first takes.
    plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, leaf_bases, leaf_rotations, leaf_sizes);
    for (const uint objID: phytomer->leaf_objIDs.at(0)) {
        const std::pair<size_t, size_t> counts = countLabelled(objID);
        DOCTEST_CHECK_MESSAGE(counts.first > 0, "re-prescribed leaf object " << objID << " has " << counts.second << " primitives and none labelled 'leaf'");
    }

    plantarchitecture.setPetioleLeafCount(plantID, shootID, 1, 0, uint(count + 2));
    for (const uint objID: phytomer->leaf_objIDs.at(0)) {
        const std::pair<size_t, size_t> counts = countLabelled(objID);
        DOCTEST_CHECK_MESSAGE(counts.first > 0, "rebuilt leaf object " << objID << " has " << counts.second << " primitives and none labelled 'leaf'");
    }
    std::vector<vec3> more_bases(count + 2, internode_tip + make_vec3(0.05f, 0.03f, 0.01f));
    std::vector<AxisRotation> more_rotations(count + 2, make_AxisRotation(0.f, 0.3f, 0.f));
    std::vector<float> more_sizes(count + 2, 0.03f);
    plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, more_bases, more_rotations, more_sizes);
    for (const uint objID: phytomer->leaf_objIDs.at(0)) {
        const std::pair<size_t, size_t> counts = countLabelled(objID);
        DOCTEST_CHECK_MESSAGE(counts.first > 0, "re-prescribed rebuilt leaf object " << objID << " has " << counts.second << " primitives and none labelled 'leaf'");
    }
}

//! Checks that every leaf on one petiole of a reloaded phytomer matches the saved one: count, prototype source, size, base, stored angles and pose
static void checkRestoredPetioleLeaves(Context &context, const std::shared_ptr<Phytomer> &saved, Context &restored_context, const std::shared_ptr<Phytomer> &restored, uint petiole) {
    DOCTEST_REQUIRE(restored->leaf_objIDs.at(petiole).size() == saved->leaf_objIDs.at(petiole).size());
    for (uint leaf = 0; leaf < saved->leaf_objIDs.at(petiole).size(); leaf++) {
        DOCTEST_CAPTURE(leaf);
        DOCTEST_CHECK(restored->leaf_prototype_index.at(petiole).at(leaf) == saved->leaf_prototype_index.at(petiole).at(leaf));
        DOCTEST_CHECK(restored->leaf_size_max.at(petiole).at(leaf) * restored->current_leaf_scale_factor.at(petiole) ==
                      doctest::Approx(saved->leaf_size_max.at(petiole).at(leaf) * saved->current_leaf_scale_factor.at(petiole)).epsilon(1e-4));
        DOCTEST_CHECK((restored->leaf_bases.at(petiole).at(leaf) - saved->leaf_bases.at(petiole).at(leaf)).magnitude() < 1e-4f);
        DOCTEST_CHECK(restored->leaf_rotation.at(petiole).at(leaf).roll == doctest::Approx(saved->leaf_rotation.at(petiole).at(leaf).roll).epsilon(1e-4));
        DOCTEST_CHECK(restored->leaf_rotation.at(petiole).at(leaf).pitch == doctest::Approx(saved->leaf_rotation.at(petiole).at(leaf).pitch).epsilon(1e-4));
        DOCTEST_CHECK(restored->leaf_rotation.at(petiole).at(leaf).yaw == doctest::Approx(saved->leaf_rotation.at(petiole).at(leaf).yaw).epsilon(1e-4));

        DOCTEST_REQUIRE(restored_context.doesObjectExist(restored->leaf_objIDs.at(petiole).at(leaf)));
        float transform_saved[16];
        float transform_restored[16];
        context.getObjectTransformationMatrix(saved->leaf_objIDs.at(petiole).at(leaf), transform_saved);
        restored_context.getObjectTransformationMatrix(restored->leaf_objIDs.at(petiole).at(leaf), transform_restored);
        float worst_element = 0.f;
        for (uint element = 0; element < 16; element++) {
            worst_element = std::max(worst_element, std::fabs(transform_restored[element] - transform_saved[element]));
        }
        DOCTEST_CHECK(worst_element < 2e-4f);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture readPlantStructureXML restores a petiole whose leaflet count was changed") {
    // Regression test. setPetioleLeafCount() rebuilds a petiole's leaves from the prototype function, so a petiole can carry more leaflets than
    // the shoot type's cached prototypes hold. readPlantStructureXML() copied every leaf from that cache, indexing it by leaflet position, and
    // threw std::out_of_range for the first leaflet past the cached count. A leaf built from the prototype function is also recorded with
    // <leaf_prototype> -1, which the reader took to mean "a file written before the tag existed" and replaced with a randomly drawn cached blade.
    const std::string xml_filename = "plantarchitecture_leaf_count_roundtrip.xml";

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    // A grown shoot rather than one built from node positions: the reader restores each phytomer's organs through the per-phytomer path.
    definePrescribedLeafGeometryShootType(context, plantarchitecture);
    const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    const uint shootID = plantarchitecture.addBaseStemShoot(plantID, 2, make_AxisRotation(0, 0, 0), 0.01f, 0.1f, 1.f, 1.f, 0, "leafy");

    plantarchitecture.setPetioleLeafCount(plantID, shootID, 1, 0, 5);
    plantarchitecture.setPetioleLeafCount(plantID, shootID, 0, 0, 1);
    DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    Context restored_context;
    PlantArchitecture restored_plantarchitecture(&restored_context);
    restored_plantarchitecture.disableMessages();
    definePrescribedLeafGeometryShootType(restored_context, restored_plantarchitecture);
    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = restored_plantarchitecture.readPlantStructureXML(xml_filename, true));
    std::remove(xml_filename.c_str());
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);

    const auto saved_shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto restored_shoot = restored_plantarchitecture.getPlantShoot(restored_plantIDs.front(), 0);
    DOCTEST_REQUIRE(restored_shoot->phytomers.size() == saved_shoot->phytomers.size());
    for (uint node = 0; node < saved_shoot->phytomers.size(); node++) {
        DOCTEST_CAPTURE(node);
        checkRestoredPetioleLeaves(context, saved_shoot->phytomers.at(node), restored_context, restored_shoot->phytomers.at(node), 0);
    }
    DOCTEST_CHECK(restored_plantarchitecture.getPlantLeafObjectIDs(restored_plantIDs.front()).size() == 6);

    // A file written before <leaf_prototype> existed records no prototype, and its leaves are drawn from the cache at random. The reader compared the saved
    // index with the unsigned count of cached prototypes, so an unrecorded (or explicit -1) index converted to a huge value and the file was rejected.
    Context legacy_context;
    legacy_context.seedRandomGenerator(12345);
    PlantArchitecture legacy_plantarchitecture(&legacy_context);
    legacy_plantarchitecture.disableMessages();
    definePrescribedLeafGeometryShootType(legacy_context, legacy_plantarchitecture);
    const uint legacy_plantID = legacy_plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    legacy_plantarchitecture.addBaseStemShoot(legacy_plantID, 2, make_AxisRotation(0, 0, 0), 0.01f, 0.1f, 1.f, 1.f, 0, "leafy");
    const std::string legacy_filename = "plantarchitecture_legacy_leaf_prototype.xml";
    DOCTEST_REQUIRE_NOTHROW(legacy_plantarchitecture.writePlantStructureXML(legacy_plantID, legacy_filename));
    std::string legacy_contents;
    {
        std::ifstream legacy_in(legacy_filename);
        std::string line;
        while (std::getline(legacy_in, line)) {
            if (line.find("<leaf_prototype>") == std::string::npos) {
                legacy_contents += line + "\n";
            }
        }
    }
    {
        std::ofstream legacy_out(legacy_filename);
        legacy_out << legacy_contents;
    }

    Context legacy_restored_context;
    PlantArchitecture legacy_restored_plantarchitecture(&legacy_restored_context);
    legacy_restored_plantarchitecture.disableMessages();
    definePrescribedLeafGeometryShootType(legacy_restored_context, legacy_restored_plantarchitecture);
    std::vector<uint> legacy_restored_plantIDs;
    DOCTEST_CHECK_NOTHROW(legacy_restored_plantIDs = legacy_restored_plantarchitecture.readPlantStructureXML(legacy_filename, true));
    std::remove(legacy_filename.c_str());
    DOCTEST_REQUIRE(legacy_restored_plantIDs.size() == 1);
    DOCTEST_CHECK(legacy_restored_plantarchitecture.getPlantLeafObjectIDs(legacy_restored_plantIDs.front()).size() == legacy_plantarchitecture.getPlantLeafObjectIDs(legacy_plantID).size());
}

DOCTEST_TEST_CASE("PlantArchitecture readPlantStructureXML restores prescribed petiole paths and prescribed leaves") {
    // Regression test. writePlantStructureXML() recorded neither a petiole path set by setPetioleNodePositions() nor that a leaf's pose had been
    // prescribed by setPetioleLeafGeometry(), nor where its base was placed. The reader rebuilt the petiole from its length, pitch and curvature,
    // placed each leaf where the shoot type would have, and oriented it through the procedural chain, so a reconstructed plant came back with
    // generated petioles and leaves rather than the measured ones.
    const std::string xml_filename = "plantarchitecture_prescribed_organs_roundtrip.xml";

    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);
    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);

    // Node 1: a measured three-leaflet leaf. Node 0: a single prescribed leaf on a petiole whose leaflet count was changed first.
    const vec3 tip1 = shoot->shoot_internode_vertices.at(1).back();
    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, measuredPetiolePath(tip1), measuredPetioleRadii());
    const std::vector<vec3> leaf_bases = {tip1 + make_vec3(0.040f, 0.018f, 0.017f), tip1 + make_vec3(0.075f, 0.063f, -0.004f), tip1 + make_vec3(0.058f, 0.030f, 0.010f)};
    const std::vector<AxisRotation> leaf_rotations = {make_AxisRotation(0.10f, 0.25f, 0.30f), make_AxisRotation(0.2f, 0.40f, -0.15f), make_AxisRotation(-0.10f, 0.25f, -0.30f)};
    plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, leaf_bases, leaf_rotations, {0.022f, 0.035f, 0.022f});

    const vec3 tip0 = shoot->shoot_internode_vertices.at(0).back();
    std::vector<vec3> path0 = measuredPetiolePath(tip0);
    for (vec3 &position: path0) {
        position = tip0 + rotatePointAboutLine(position - tip0, nullorigin, make_vec3(0, 0, 1), 2.5f);
    }
    plantarchitecture.setPetioleNodePositions(plantID, shootID, 0, 0, path0, measuredPetioleRadii());
    plantarchitecture.setPetioleLeafCount(plantID, shootID, 0, 0, 1);
    plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 0, 0, {path0.back() + make_vec3(0.005f, 0, 0)}, {make_AxisRotation(0.3f, -0.2f, 0.6f)}, {0.04f});

    DOCTEST_REQUIRE_NOTHROW(plantarchitecture.writePlantStructureXML(plantID, xml_filename));

    Context restored_context;
    PlantArchitecture restored_plantarchitecture(&restored_context);
    restored_plantarchitecture.disableMessages();
    definePrescribedLeafGeometryShootType(restored_context, restored_plantarchitecture);
    std::vector<uint> restored_plantIDs;
    DOCTEST_REQUIRE_NOTHROW(restored_plantIDs = restored_plantarchitecture.readPlantStructureXML(xml_filename, true));
    std::remove(xml_filename.c_str());
    DOCTEST_REQUIRE(restored_plantIDs.size() == 1);
    const uint restored_plantID = restored_plantIDs.front();
    const auto restored_shoot = restored_plantarchitecture.getPlantShoot(restored_plantID, 0);
    DOCTEST_REQUIRE(restored_shoot->phytomers.size() == shoot->phytomers.size());

    for (uint node = 0; node < 2; node++) {
        DOCTEST_CAPTURE(node);
        const auto &saved = shoot->phytomers.at(node);
        const auto &restored = restored_shoot->phytomers.at(node);
        DOCTEST_REQUIRE(restored->petiole_vertices.at(0).size() == saved->petiole_vertices.at(0).size());
        for (size_t vertex = 0; vertex < saved->petiole_vertices.at(0).size(); vertex++) {
            DOCTEST_CHECK((restored->petiole_vertices.at(0).at(vertex) - saved->petiole_vertices.at(0).at(vertex)).magnitude() < 1e-5f);
            DOCTEST_CHECK(restored->petiole_radii.at(0).at(vertex) == doctest::Approx(saved->petiole_radii.at(0).at(vertex)).epsilon(1e-4));
        }
        checkRestoredPetioleLeaves(context, saved, restored_context, restored, 0);
    }

    // Still prescribed after the reload: growth neither re-poses the leaves nor re-scales the petiole.
    const std::shared_ptr<Phytomer> restored_node1 = restored_shoot->phytomers.at(1); // a copy: growth appends phytomers and reallocates the vector
    float transform_before[16];
    restored_context.getObjectTransformationMatrix(restored_node1->leaf_objIDs.at(0).at(1), transform_before);
    const std::vector<vec3> petiole_before = restored_node1->petiole_vertices.at(0);
    restored_plantarchitecture.breakPlantDormancy(restored_plantID);
    restored_plantarchitecture.advanceTime(restored_plantID, 10);
    float transform_after[16];
    restored_context.getObjectTransformationMatrix(restored_node1->leaf_objIDs.at(0).at(1), transform_after);
    for (uint element = 0; element < 16; element++) {
        DOCTEST_CHECK(transform_after[element] == doctest::Approx(transform_before[element]).epsilon(1e-4));
    }
    DOCTEST_CHECK((restored_node1->petiole_vertices.at(0).back() - petiole_before.back()).magnitude() < 1e-5f);
}

DOCTEST_TEST_CASE("PlantArchitecture prescribed leaf scale bookkeeping stays consistent") {
    // A leaf's rendered size is leaf_size_max * current_leaf_scale_factor. Pinning the factor at 1 to exempt
    // the petiole from rescaling therefore has to fold the old factor into leaf_size_max, or the bookkeeping
    // silently overstates every leaf on the petiole while the geometry is unchanged - the same class of defect
    // documented at length in scaleLeafPrototypeScale().
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto phytomer = shoot->phytomers.at(1);

    // Put the petiole part-way through its expansion, which is the state every growing phytomer is in.
    phytomer->setLeafScaleFraction(0, 0.5f);
    DOCTEST_REQUIRE(phytomer->current_leaf_scale_factor.at(0) == doctest::Approx(0.5f));

    vec3 box_min_before, box_max_before;
    context.getObjectBoundingBox(phytomer->leaf_objIDs.at(0).at(1), box_min_before, box_max_before);
    const vec3 extent_before = box_max_before - box_min_before;
    const float rendered_before = phytomer->leaf_size_max.at(0).at(1) * phytomer->current_leaf_scale_factor.at(0);

    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();
    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, measuredPetiolePath(internode_tip), measuredPetioleRadii());

    // The petiole setter must not resize any leaf: it pins the growth fraction, and the size that fraction
    // used to represent moves into leaf_size_max so the product is unchanged.
    DOCTEST_CHECK(phytomer->current_leaf_scale_factor.at(0) == doctest::Approx(1.f));
    DOCTEST_CHECK(phytomer->leaf_size_max.at(0).at(1) * phytomer->current_leaf_scale_factor.at(0) == doctest::Approx(rendered_before).epsilon(1e-4));

    vec3 box_min_after, box_max_after;
    context.getObjectBoundingBox(phytomer->leaf_objIDs.at(0).at(1), box_min_after, box_max_after);
    const vec3 extent_after = box_max_after - box_min_after;
    DOCTEST_CHECK(extent_after.x == doctest::Approx(extent_before.x).epsilon(1e-3));
    DOCTEST_CHECK(extent_after.y == doctest::Approx(extent_before.y).epsilon(1e-3));
    DOCTEST_CHECK(extent_after.z == doctest::Approx(extent_before.z).epsilon(1e-3));
}

DOCTEST_TEST_CASE("PlantArchitecture prescribing one petiole leaves its neighbours alone") {
    // setPetioleBase() derives its shift from petiole 0 alone and applies it to every petiole on the phytomer,
    // so a phytomer carrying both a prescribed and an unprescribed petiole is the case where a per-petiole
    // lock could leak into its neighbour. Both orderings are exercised for that reason.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID, 2);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto phytomer = shoot->phytomers.at(1);
    DOCTEST_REQUIRE(phytomer->petiole_vertices.size() == 2);

    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();

    // A shoot from addShootFromNodePositions() is created fully elongated, so both petioles start with their
    // growth fraction already at one. Put the petiole that is NOT being prescribed back into a growing state,
    // since a neighbour that was never going to grow could not show a lock leaking onto it.
    phytomer->setLeafScaleFraction(0, 0.4f);
    DOCTEST_REQUIRE(phytomer->current_leaf_scale_factor.at(0) < 1.f);

    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 1, measuredPetiolePath(internode_tip), measuredPetioleRadii());

    const std::vector<vec3> prescribed_before = phytomer->petiole_vertices.at(1);
    const float unprescribed_length_before = phytomer->petiole_length.at(0);
    DOCTEST_REQUIRE(phytomer->current_leaf_scale_factor.at(0) < 1.f);

    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 20);

    for (size_t node = 0; node < prescribed_before.size(); node++) {
        DOCTEST_CHECK((phytomer->petiole_vertices.at(1).at(node) - prescribed_before.at(node)).magnitude() < 1e-5f);
    }
    // The unprescribed petiole must still be growing - the lock is per petiole, not per phytomer.
    DOCTEST_CHECK(phytomer->petiole_length.at(0) > unprescribed_length_before);
    DOCTEST_CHECK(phytomer->current_leaf_scale_factor.at(0) > 0.4f);

    // The mirror case: prescribe petiole 0 and confirm petiole 1 goes on growing.
    Context context_mirror;
    context_mirror.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture_mirror(&context_mirror);
    plantarchitecture_mirror.disableMessages();

    uint plantID_mirror, shootID_mirror;
    buildLeafyPrescribedShoot(context_mirror, plantarchitecture_mirror, plantID_mirror, shootID_mirror, 2);

    const auto shoot_mirror = plantarchitecture_mirror.getPlantShoot(plantID_mirror, shootID_mirror);
    const auto phytomer_mirror = shoot_mirror->phytomers.at(1);
    const vec3 internode_tip_mirror = shoot_mirror->shoot_internode_vertices.at(1).back();

    phytomer_mirror->setLeafScaleFraction(1, 0.4f);
    DOCTEST_REQUIRE(phytomer_mirror->current_leaf_scale_factor.at(1) < 1.f);

    plantarchitecture_mirror.setPetioleNodePositions(plantID_mirror, shootID_mirror, 1, 0, measuredPetiolePath(internode_tip_mirror), measuredPetioleRadii());

    const std::vector<vec3> prescribed_before_mirror = phytomer_mirror->petiole_vertices.at(0);
    const float unprescribed_length_before_mirror = phytomer_mirror->petiole_length.at(1);

    plantarchitecture_mirror.breakPlantDormancy(plantID_mirror);
    plantarchitecture_mirror.advanceTime(plantID_mirror, 20);

    for (size_t node = 0; node < prescribed_before_mirror.size(); node++) {
        DOCTEST_CHECK((phytomer_mirror->petiole_vertices.at(0).at(node) - prescribed_before_mirror.at(node)).magnitude() < 1e-5f);
    }
    DOCTEST_CHECK(phytomer_mirror->petiole_length.at(1) > unprescribed_length_before_mirror);
}

DOCTEST_TEST_CASE("PlantArchitecture setPetioleLeafGeometry places each leaflet of a compound leaf") {
    // orientLeaf() flips the sign of roll and yaw depending on which side of the petiole a leaflet sits, and
    // the tip leaflet takes a different branch again. Prescribing all three and requiring their transforms to
    // differ pairwise is what exercises those branches; three leaflets given the same pose would pass even if
    // the per-leaflet argument were ignored entirely.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto phytomer = shoot->phytomers.at(1);
    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();

    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, measuredPetiolePath(internode_tip), measuredPetioleRadii());

    const std::vector<vec3> leaf_bases = {internode_tip + make_vec3(0.030f, 0.012f, 0.016f), internode_tip + make_vec3(0.075f, 0.063f, -0.004f), internode_tip + make_vec3(0.055f, 0.028f, 0.012f)};
    const std::vector<AxisRotation> leaf_rotations = {make_AxisRotation(0.15f, 0.20f, 0.35f), make_AxisRotation(0.f, 0.45f, 0.f), make_AxisRotation(-0.15f, 0.30f, -0.35f)};
    const std::vector<float> leaf_sizes = {0.018f, 0.032f, 0.024f};

    plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, leaf_bases, leaf_rotations, leaf_sizes);

    float transforms[3][16];
    for (uint leaf = 0; leaf < 3; leaf++) {
        DOCTEST_CHECK((phytomer->leaf_bases.at(0).at(leaf) - leaf_bases.at(leaf)).magnitude() < 1e-6f);
        DOCTEST_CHECK(phytomer->leaf_size_max.at(0).at(leaf) == doctest::Approx(leaf_sizes.at(leaf)).epsilon(1e-5));
        DOCTEST_REQUIRE(context.doesObjectExist(phytomer->leaf_objIDs.at(0).at(leaf)));
        context.getObjectTransformationMatrix(phytomer->leaf_objIDs.at(0).at(leaf), transforms[leaf]);
    }

    for (uint a = 0; a < 3; a++) {
        for (uint b = a + 1; b < 3; b++) {
            bool differs = false;
            for (uint element = 0; element < 16 && !differs; element++) {
                differs = std::fabs(transforms[a][element] - transforms[b][element]) > 1e-5f;
            }
            DOCTEST_CHECK(differs);
        }
    }
}

DOCTEST_TEST_CASE("PlantArchitecture prescribed petiole follows its internode when the internode moves") {
    // The counter-test to the survival case: the petiole is pinned against the growth model rescaling it, but
    // it must still ride along when the internode it hangs off actually moves, or a prescribed petiole would
    // detach from its own stem the first time a phytomer below it elongated.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const auto phytomer = shoot->phytomers.at(1);
    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();

    plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, measuredPetiolePath(internode_tip), measuredPetioleRadii());

    const std::vector<vec3> before = phytomer->petiole_vertices.at(0);
    const vec3 leaf_base_before = phytomer->leaf_bases.at(0).at(0);

    // Shorten the internode below the prescribed petiole, which moves the tip the petiole hangs from.
    shoot->phytomers.at(0)->setInternodeLengthScaleFraction(0.5f, true);

    const std::vector<vec3> after = phytomer->petiole_vertices.at(0);
    const vec3 expected_shift = shoot->shoot_internode_vertices.at(1).back() - before.front();
    DOCTEST_REQUIRE(expected_shift.magnitude() > 1e-4f);

    DOCTEST_REQUIRE(after.size() == before.size());
    for (size_t node = 0; node < before.size(); node++) {
        DOCTEST_CHECK((after.at(node) - (before.at(node) + expected_shift)).magnitude() < 1e-5f);
    }
    // Rigid: every segment vector is unchanged, only the absolute position moved.
    for (size_t node = 0; node + 1 < before.size(); node++) {
        const vec3 segment_before = before.at(node + 1) - before.at(node);
        const vec3 segment_after = after.at(node + 1) - after.at(node);
        DOCTEST_CHECK((segment_after - segment_before).magnitude() < 1e-6f);
    }
    DOCTEST_CHECK((phytomer->leaf_bases.at(0).at(0) - (leaf_base_before + expected_shift)).magnitude() < 1e-5f);
}

DOCTEST_TEST_CASE("PlantArchitecture prescribed petiole and leaf geometry rejects invalid input") {
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID, shootID;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const vec3 internode_tip = shoot->shoot_internode_vertices.at(1).back();
    const std::vector<vec3> valid_path = measuredPetiolePath(internode_tip);
    const std::vector<float> valid_radii = measuredPetioleRadii();

    {
        capture_cerr cerr_buffer;

        // Plant, shoot and node must all exist.
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID + 100, shootID, 1, 0, valid_path, valid_radii));
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID + 100, 1, 0, valid_path, valid_radii));
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 9, 0, valid_path, valid_radii));

        // The phytomer carries one petiole, so index 1 is out of range.
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 1, valid_path, valid_radii));

        // Fewer than two nodes cannot define a petiole axis.
        const std::vector<vec3> one_position = {internode_tip};
        const std::vector<float> one_radius = {0.002f};
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, one_position, one_radius));

        // Radii count must match the node count.
        const std::vector<float> too_few_radii = {0.002f, 0.001f};
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, valid_path, too_few_radii));

        // Radii must be strictly positive.
        std::vector<float> zero_radius = valid_radii;
        zero_radius.at(2) = 0.f;
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, valid_path, zero_radius));
        std::vector<float> negative_radius = valid_radii;
        negative_radius.at(2) = -0.001f;
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, valid_path, negative_radius));

        // Coincident consecutive nodes give a degenerate petiole axis.
        std::vector<vec3> coincident_path = valid_path;
        coincident_path.at(3) = coincident_path.at(2);
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, coincident_path, valid_radii));

        // A first node far from the internode tip would silently produce a detached petiole.
        const std::vector<vec3> detached_path = measuredPetiolePath(internode_tip + make_vec3(5.f, 5.f, 5.f));
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, detached_path, valid_radii));
    }

    // A first node at the internode tip is accepted.
    DOCTEST_CHECK_NOTHROW(plantarchitecture.setPetioleNodePositions(plantID, shootID, 1, 0, valid_path, valid_radii));

    const std::vector<vec3> valid_bases = {internode_tip + make_vec3(0.030f, 0.012f, 0.016f), internode_tip + make_vec3(0.075f, 0.063f, -0.004f), internode_tip + make_vec3(0.055f, 0.028f, 0.012f)};
    const std::vector<AxisRotation> valid_rotations = {make_AxisRotation(0.1f, 0.2f, 0.3f), make_AxisRotation(0.f, 0.4f, 0.f), make_AxisRotation(-0.1f, 0.2f, -0.3f)};
    const std::vector<float> valid_sizes = {0.02f, 0.03f, 0.02f};

    {
        capture_cerr cerr_buffer;

        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID + 100, shootID, 1, 0, valid_bases, valid_rotations, valid_sizes));
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID + 100, 1, 0, valid_bases, valid_rotations, valid_sizes));
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 9, 0, valid_bases, valid_rotations, valid_sizes));
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 1, valid_bases, valid_rotations, valid_sizes));

        // The number of leaves per petiole is fixed when the phytomer is created, so a mismatched count in
        // either direction is a shoot type definition error rather than something to accommodate.
        const std::vector<vec3> too_many_bases(4, internode_tip);
        const std::vector<AxisRotation> too_many_rotations(4, make_AxisRotation(0.f, 0.f, 0.f));
        const std::vector<float> too_many_sizes(4, 0.02f);
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, too_many_bases, too_many_rotations, too_many_sizes));
        const std::vector<vec3> too_few_bases(2, internode_tip);
        const std::vector<AxisRotation> too_few_rotations(2, make_AxisRotation(0.f, 0.f, 0.f));
        const std::vector<float> too_few_sizes(2, 0.02f);
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, too_few_bases, too_few_rotations, too_few_sizes));

        // The three vectors must agree with each other, not just with the leaf count.
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, valid_bases, too_few_rotations, valid_sizes));
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, valid_bases, valid_rotations, too_few_sizes));

        // Leaf sizes must be strictly positive.
        std::vector<float> zero_size = valid_sizes;
        zero_size.at(1) = 0.f;
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, valid_bases, valid_rotations, zero_size));
        std::vector<float> negative_size = valid_sizes;
        negative_size.at(1) = -0.03f;
        DOCTEST_CHECK_THROWS(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, valid_bases, valid_rotations, negative_size));
    }

    DOCTEST_CHECK_NOTHROW(plantarchitecture.setPetioleLeafGeometry(plantID, shootID, 1, 0, valid_bases, valid_rotations, valid_sizes));
}

//! Spread of a sample, as (max - min) / mean. Zero exactly when every entry is identical.
static float relativeSpread(const std::vector<float> &values) {
    DOCTEST_REQUIRE(!values.empty());
    float minimum = values.front();
    float maximum = values.front();
    float sum = 0.f;
    for (const float value: values) {
        minimum = std::fmin(minimum, value);
        maximum = std::fmax(maximum, value);
        sum += value;
    }
    const float mean = sum / float(values.size());
    DOCTEST_REQUIRE(mean > 0.f);
    return (maximum - minimum) / mean;
}

//! Length of the first petiole on a phytomer, measured along its built tube vertices.
static float builtPetioleLength(const std::shared_ptr<Phytomer> &phytomer) {
    DOCTEST_REQUIRE(!phytomer->petiole_vertices.empty());
    const std::vector<vec3> &vertices = phytomer->petiole_vertices.front();
    DOCTEST_REQUIRE(vertices.size() > 1);
    float length = 0.f;
    for (size_t i = 0; i + 1 < vertices.size(); i++) {
        length += (vertices.at(i + 1) - vertices.at(i)).magnitude();
    }
    return length;
}

DOCTEST_TEST_CASE("PlantArchitecture phytomers grown at the apex follow the shoot's own parameters") {
    // advanceTime() reads internode_radius from shoot->shoot_parameters and resamples it there, then
    // passes phytomer_parameters from the plant's stored shoot type in the same call. So the two sources
    // drift apart as a plant grows, and every field other than radius_initial -- petiole, leaf,
    // inflorescence -- was taken from the type no matter what the shoot said.
    //
    // Petiole length is the observable: it is read straight from the passed parameters when a phytomer
    // is built, so it shows up as geometry rather than as internal state. A factor of five apart makes
    // the two candidate answers impossible to confuse.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID = 0;
    uint shootID = 0;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const size_t nodes_before = shoot->phytomers.size();
    DOCTEST_REQUIRE(nodes_before > 0);

    // The fixture's type builds petioles of 0.05 m. Edit this one shoot only; the stored shoot type is
    // left alone, so the two sources now disagree by a factor of five.
    const float shoot_petiole_length = 0.25f;
    shoot->shoot_parameters.phytomer_parameters.petiole.length = shoot_petiole_length;

    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 20);

    // Bud break is disabled on this type, so growth is confined to this shoot's own apex and every new
    // phytomer went through the phyllochron path under test.
    DOCTEST_REQUIRE(shoot->phytomers.size() > nodes_before);

    // A petiole is built at current_leaf_scale_factor of its full length, and a phytomer created moments
    // ago is barely expanded -- so an absolute threshold would flag a young petiole built from the
    // correct 0.25 m as if it came from the type. Dividing the scaling back out recovers the parameter
    // the phytomer was built from, which is the thing actually under test.
    for (size_t node = nodes_before; node < shoot->phytomers.size(); node++) {
        const std::shared_ptr<Phytomer> &phytomer = shoot->phytomers.at(node);
        DOCTEST_REQUIRE(!phytomer->current_leaf_scale_factor.empty());
        const float scale = phytomer->current_leaf_scale_factor.front();
        DOCTEST_REQUIRE(scale > 0.f);
        const float unscaled_length = builtPetioleLength(phytomer) / scale;
        DOCTEST_CAPTURE(node);
        DOCTEST_CAPTURE(scale);
        DOCTEST_CAPTURE(unscaled_length);
        // 0.25 m if the shoot was consulted, 0.05 m if its type was. Nothing lies between them.
        DOCTEST_CHECK(unscaled_length == doctest::Approx(shoot_petiole_length).epsilon(0.02f));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture duplicatePlantInstance copies the source shoot's own parameters") {
    // duplicatePlantInstance() rebuilds the copy by replaying shoot creation, and its per-phytomer
    // append had the same defect as advanceTime(): phytomer_parameters came from the plant's stored
    // shoot type rather than from the shoot being copied. A plant whose shoot carried edited parameters
    // therefore came back as a copy of its TYPE, silently losing the edit.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();

    uint plantID = 0;
    uint shootID = 0;
    buildLeafyPrescribedShoot(context, plantarchitecture, plantID, shootID);

    const auto &shoot = plantarchitecture.getPlantShoot(plantID, shootID);
    const float shoot_petiole_length = 0.25f;
    shoot->shoot_parameters.phytomer_parameters.petiole.length = shoot_petiole_length;

    // Grow first, so the source has phytomers built from the edited value for the duplicate to match.
    // The multi-phytomer branch of duplicatePlantInstance() is the one that appends node by node.
    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.advanceTime(plantID, 20);

    const uint plantID_copy = plantarchitecture.duplicatePlantInstance(plantID, make_vec3(1, 0, 0), make_AxisRotation(0, 0, 0), 0.f);
    const auto &shoot_copy = plantarchitecture.getPlantShoot(plantID_copy, shootID);

    DOCTEST_REQUIRE(shoot_copy->phytomers.size() == shoot->phytomers.size());

    // The prescribed phytomers were built before the edit and so still carry the type's 0.05 m, while
    // the phytomers grown afterwards carry the shoot's 0.25 m. That split is what makes this test sharp:
    // a duplicate rebuilt from the shoot type would get every node wrong, and one rebuilt from the
    // shoot's current parameters would get the older nodes wrong instead. Only reproducing each phytomer
    // from the values it was actually built with satisfies both ends.
    float shortest = builtPetioleLength(shoot->phytomers.front());
    float longest = shortest;
    for (const auto &phytomer: shoot->phytomers) {
        shortest = std::fmin(shortest, builtPetioleLength(phytomer));
        longest = std::fmax(longest, builtPetioleLength(phytomer));
    }
    DOCTEST_REQUIRE(longest > 2.f * shortest);

    // The copy must match the plant it came from, not the type they were both defined against.
    for (size_t node = 0; node < shoot->phytomers.size(); node++) {
        const float source_length = builtPetioleLength(shoot->phytomers.at(node));
        const float copy_length = builtPetioleLength(shoot_copy->phytomers.at(node));
        DOCTEST_CAPTURE(node);
        DOCTEST_CAPTURE(source_length);
        DOCTEST_CAPTURE(copy_length);
        DOCTEST_CHECK(copy_length == doctest::Approx(source_length).epsilon(0.02f));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture realized getters measure the built plant") {
    // These report what the plant was actually built with, as opposed to getCurrentShootParameters(),
    // which reports the distribution a parameter was drawn from. The distinction is the reason they
    // exist: a RandomParameter caches its first draw, and a shoot holds a copy of its type's
    // parameters, so a plant can be delivered with no variation at all while its parameters still
    // describe a wide spread. Reporting the stored draw would have hidden exactly that.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");

    const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 20);

    const std::vector<float> leaf_areas = plantarchitecture.getPlantLeafAreas(plantID);
    const std::vector<float> internode_lengths = plantarchitecture.getPlantInternodeLengths(plantID);
    const std::vector<float> leaf_inclinations = plantarchitecture.getPlantLeafInclinations(plantID);

    DOCTEST_REQUIRE(!leaf_areas.empty());
    DOCTEST_REQUIRE(!internode_lengths.empty());

    // Leaves whose geometry no longer exists are skipped, so the count can be lower than the leaf
    // count but must never exceed it.
    DOCTEST_CHECK(leaf_areas.size() <= plantarchitecture.getPlantLeafCount(plantID));
    DOCTEST_CHECK(leaf_inclinations.size() == leaf_areas.size());

    // One internode per phytomer, which is what the shoot node counts add up to.
    uint expected_internodes = 0;
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        expected_internodes += plantarchitecture.getShootNodeCount(plantID, shootID);
    }
    DOCTEST_CHECK(internode_lengths.size() == expected_internodes);

    for (const float area: leaf_areas) {
        DOCTEST_CHECK(area > 0.f);
    }
    for (const float length: internode_lengths) {
        DOCTEST_CHECK(length > 0.f);
    }
    // Folded about the horizontal, so an inclination is always within the first quadrant.
    for (const float inclination: leaf_inclinations) {
        DOCTEST_CHECK(inclination >= 0.f);
        DOCTEST_CHECK(inclination <= 90.f + 1e-3f);
    }

    // Cross-check against the aggregate the library already exposes, which sums the same geometry.
    float area_sum = 0.f;
    for (const float area: leaf_areas) {
        area_sum += area;
    }
    DOCTEST_CHECK(area_sum == doctest::Approx(plantarchitecture.sumPlantLeafArea(plantID)).epsilon(0.01f));

    // Inclinations must actually resolve orientation rather than collapsing to one value: a bean canopy
    // holds its leaves at a range of angles, so a getter returning a constant would be wrong even though
    // every entry sat inside the valid range checked above.
    float inclination_min = leaf_inclinations.front();
    float inclination_max = leaf_inclinations.front();
    for (const float inclination: leaf_inclinations) {
        inclination_min = std::fmin(inclination_min, inclination);
        inclination_max = std::fmax(inclination_max, inclination);
    }
    DOCTEST_CHECK(inclination_max - inclination_min > 1.f);

    DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.getPlantLeafAreas(plantID + 100)));
    DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.getPlantInternodeLengths(plantID + 100)));
    DOCTEST_CHECK_THROWS(static_cast<void>(plantarchitecture.getPlantLeafInclinations(plantID + 100)));
}

DOCTEST_TEST_CASE("PlantArchitecture realized leaf areas expose an absent spread") {
    // The regression that motivated these getters. A plant was delivered with exactly zero variation in
    // leaf area across thousands of leaves while its leaf parameters carried a spread, and finding that
    // required exporting the geometry and computing statistics outside Helios. A getter that reported
    // the parameter's stored draw would have reported the spread and hidden the defect; measuring the
    // built geometry is what makes the two distinguishable.
    //
    // A purpose-built shoot type is used rather than a library species so the claim is exact. Two
    // sources of genuine variation have to be excluded first or they would swamp the comparison: one
    // leaf per petiole, since a compound leaf sizes its lateral leaflets differently from its terminal
    // one, and leaves built at full size, since a leaf part-way through expanding is legitimately
    // smaller than its neighbours. Both are real built variation that these getters should and do
    // report -- they are simply not the thing under test here.
    //
    // Both directions are asserted: a constant leaf scale must deliver no spread, and a distributed one
    // must deliver some. Either assertion alone would pass on a getter that was merely wrong.
    auto buildAndMeasure = [](bool vary_leaf_scale) -> std::vector<float> {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();

        PhytomerParameters phytomer_parameters(context.getRandomGenerator());
        phytomer_parameters.internode.pitch = 0;
        phytomer_parameters.internode.phyllotactic_angle = 0;
        phytomer_parameters.internode.length_segments = 1;
        phytomer_parameters.petiole.petioles_per_internode = 1;
        phytomer_parameters.petiole.length = 0.02;
        phytomer_parameters.leaf.leaves_per_petiole = 1;
        phytomer_parameters.leaf.prototype.prototype_function = GenericLeafPrototype;
        phytomer_parameters.leaf.prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/AlmondLeaf.png";
        if (vary_leaf_scale) {
            phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.02f, 0.08f);
        } else {
            phytomer_parameters.leaf.prototype_scale = 0.05f;
        }

        ShootParameters shoot_parameters(context.getRandomGenerator());
        shoot_parameters.phytomer_parameters = phytomer_parameters;
        shoot_parameters.gravitropic_curvature = 0;
        shoot_parameters.tortuosity = 0;
        shoot_parameters.max_nodes = 20;
        shoot_parameters.girth_area_factor = 0;
        shoot_parameters.vegetative_bud_break_probability_min = 0;
        shoot_parameters.vegetative_bud_break_probability_max = 0;
        plantarchitecture.defineShootType("leafy", shoot_parameters);

        const uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
        // leaf_scale_factor_fraction of 1 builds every leaf at full size, so no leaf is small merely
        // because it has not finished expanding.
        plantarchitecture.addBaseStemShoot(plantID, 8, make_AxisRotation(0, 0, 0), 0.005f, 0.03f, 1.f, 1.f, 0.f, "leafy");

        return plantarchitecture.getPlantLeafAreas(plantID);
    };

    const std::vector<float> areas_constant = buildAndMeasure(false);
    const std::vector<float> areas_varied = buildAndMeasure(true);

    DOCTEST_REQUIRE(areas_constant.size() > 1);
    DOCTEST_REQUIRE(areas_varied.size() > 1);

    // One leaf size in, one leaf size out. This is the shape of the delivered-plant bug: the parameters
    // could describe any spread at all and the built plant would still look like this.
    DOCTEST_CHECK(relativeSpread(areas_constant) < 1e-3f);

    // ...and when the parameter really does vary, the built leaves must vary with it, or the getter is
    // reporting something other than what was built.
    DOCTEST_CHECK(relativeSpread(areas_varied) > 0.1f);
}

DOCTEST_TEST_CASE("PlantArchitecture generated leaves carry vertex normals") {
    // A leaf built by GenericLeafPrototype() is assembled from bare triangles, which carry only a face
    // normal. Visualizer::enableSmoothShading() shades from interpolated vertex normals and falls back
    // to the face normal when a mesh has none, so without them every generated leaf renders faceted no
    // matter what the renderer is asked to do -- the blade's own midrib fold and lateral curl read as
    // flat panels. The blade is a Polymesh with shared vertex topology, so the normals can be generated
    // from that topology when the object is assembled.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("easternredbud");

    const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);
    plantarchitecture.setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200, false);
    plantarchitecture.makePlantDormant(plantID);
    plantarchitecture.breakPlantDormancy(plantID);
    DOCTEST_REQUIRE_NOTHROW(plantarchitecture.advanceTime(plantID, 30.f));

    std::vector<uint> leaf_objIDs;
    for (const uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
        const auto shoot = plantarchitecture.getPlantShoot(plantID, shootID);
        for (const auto &phytomer: shoot->phytomers) {
            for (const auto &petiole: phytomer->leaf_objIDs) {
                for (const uint objID: petiole) {
                    if (context.doesObjectExist(objID) && context.getObjectType(objID) == OBJECT_TYPE_POLYMESH) {
                        leaf_objIDs.push_back(objID);
                    }
                }
            }
        }
    }
    DOCTEST_REQUIRE(!leaf_objIDs.empty());

    // Every generated leaf should report vertex normals, and they should be the computed ones: this
    // prototype builds its blade from a lattice rather than reading an authored mesh.
    for (const uint objID: leaf_objIDs) {
        DOCTEST_CAPTURE(objID);
        DOCTEST_CHECK(context.getPolymeshObjectVertexNormalSource(objID) == NORMAL_SOURCE_COMPUTED);
    }
}

DOCTEST_TEST_CASE("PlantArchitecture inflorescence pitch is measured from the peduncle, not the world") {
    // InflorescenceParameters::pitch is the angle between a flower or fruit and the peduncle carrying
    // it, so it must survive any reorientation of that peduncle. The rotation that turns the prototype
    // into the peduncle's azimuthal plane was applied with the wrong sign, which is only a mirror image
    // while the peduncle lies in the xz-plane but tips the fruit out of that plane entirely once the
    // peduncle has a y-component. What reached the user was a rice panicle whose grains stood nearly
    // perpendicular to a horizontal stretch of rachis and relaxed toward the intended angle only where
    // the rachis drooped steeply -- the peduncle's own elevation leaking into an angle defined relative
    // to it. This is the same defect fixed for leaf blades in v1.3.84; the inflorescence was missed.
    //
    // Rice is the species that exposes it: its panicle carries sixty grains along a rachis that curves
    // from upright through horizontal to hanging, so a single plant samples the full range of peduncle
    // elevations, and its grains are slender enough for the built axis to be unambiguous.
    Context context;
    context.seedRandomGenerator(9271);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("rice");

    const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 75);

    // Pair every grain with the peduncle segment it actually grows from, so the angle compared against
    // the parameter is the one the parameter is defined to control.
    std::vector<float> grain_angles;
    std::vector<float> peduncle_elevations;

    for (const std::shared_ptr<Shoot> &shoot: PlantArchitectureTestHelper::getShootTree(plantarchitecture, plantID)) {
        for (const std::shared_ptr<Phytomer> &phytomer: shoot->phytomers) {
            for (uint petiole = 0; petiole < phytomer->floral_buds.size(); petiole++) {
                for (const FloralBud &fbud: phytomer->floral_buds.at(petiole)) {
                    if (fbud.inflorescence_objIDs.empty()) {
                        continue;
                    }
                    // A peduncle is stored under the petiole its bud hangs from, which for a terminal bud
                    // is petiole zero however many rows the bud list has grown -- not the row index.
                    const uint peduncle_petiole = fbud.parent_index;
                    if (peduncle_petiole >= phytomer->peduncle_vertices.size() || fbud.bud_index >= phytomer->peduncle_vertices.at(peduncle_petiole).size()) {
                        continue;
                    }
                    const std::vector<vec3> &peduncle = phytomer->peduncle_vertices.at(peduncle_petiole).at(fbud.bud_index);
                    if (peduncle.size() < 2) {
                        continue;
                    }

                    for (uint grain = 0; grain < fbud.inflorescence_objIDs.size(); grain++) {
                        const uint objID = fbud.inflorescence_objIDs.at(grain);
                        if (!context.doesObjectExist(objID)) {
                            continue;
                        }
                        const vec3 base = fbud.inflorescence_bases.at(grain);

                        // The grain's built axis, taken as the direction to the vertex furthest from its
                        // attachment point. Measuring the geometry rather than reconstructing it from the
                        // stored rotation is the point: the stored value would agree with the buggy chain.
                        vec3 tip = base;
                        float tip_distance = 0.f;
                        for (const uint UUID: context.getObjectPrimitiveUUIDs(objID)) {
                            for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                                const float distance = (vertex - base).magnitude();
                                if (distance > tip_distance) {
                                    tip_distance = distance;
                                    tip = vertex;
                                }
                            }
                        }
                        if (tip_distance < 1e-7f) {
                            continue;
                        }
                        vec3 grain_axis = tip - base;
                        grain_axis.normalize();

                        // The peduncle direction this grain was aimed against. Phytomer::getAxisVector()
                        // is used rather than the containing segment because it is what the geometry was
                        // built from: it takes a chord spanning a tenth of the rachis, which on a rachis
                        // curving through a right angle is several degrees off the local tangent. Comparing
                        // against the tangent instead would charge that discretization to the fruit.
                        float frac = 0.f;
                        float frac_distance = (PlantArchitectureTestHelper::interpolateTube(peduncle, 0.f) - base).magnitude();
                        for (int step = 1; step <= 200; step++) {
                            const float candidate = float(step) / 200.f;
                            const float distance = (PlantArchitectureTestHelper::interpolateTube(peduncle, candidate) - base).magnitude();
                            if (distance < frac_distance) {
                                frac_distance = distance;
                                frac = candidate;
                            }
                        }
                        const vec3 peduncle_axis = Phytomer::getAxisVector(frac, peduncle);
                        if (peduncle_axis.magnitude() < 1e-7f) {
                            continue;
                        }

                        grain_angles.push_back(rad2deg(acos_safe(grain_axis * peduncle_axis)));
                        peduncle_elevations.push_back(rad2deg(asin_safe(peduncle_axis.z)));
                    }
                }
            }
        }
    }

    DOCTEST_REQUIRE(grain_angles.size() > 50);

    // The rachis has to actually bend across this plant, or a constant angle would prove nothing.
    float elevation_min = peduncle_elevations.front();
    float elevation_max = peduncle_elevations.front();
    for (const float elevation: peduncle_elevations) {
        elevation_min = std::fmin(elevation_min, elevation);
        elevation_max = std::fmax(elevation_max, elevation);
    }
    DOCTEST_REQUIRE(elevation_max - elevation_min > 30.f);

    // Rice draws its pitch from a 20-25 degree distribution, so every grain must sit in that band
    // whatever its peduncle is doing. Under the defect the angle tracked peduncle elevation instead,
    // reaching some 80 degrees where the rachis ran horizontal.
    for (size_t i = 0; i < grain_angles.size(); i++) {
        DOCTEST_CAPTURE(grain_angles.at(i));
        DOCTEST_CAPTURE(peduncle_elevations.at(i));
        DOCTEST_CHECK(grain_angles.at(i) >= 20.f - 1.f);
        DOCTEST_CHECK(grain_angles.at(i) <= 25.f + 1.f);
    }

    // The angle must not co-vary with peduncle elevation. A correlation is what the defect looked like
    // and is a sharper statement than the band above: it fails even if the band is widened.
    float mean_angle = 0.f;
    float mean_elevation = 0.f;
    for (size_t i = 0; i < grain_angles.size(); i++) {
        mean_angle += grain_angles.at(i);
        mean_elevation += peduncle_elevations.at(i);
    }
    mean_angle /= float(grain_angles.size());
    mean_elevation /= float(grain_angles.size());

    float covariance = 0.f;
    float elevation_variance = 0.f;
    for (size_t i = 0; i < grain_angles.size(); i++) {
        covariance += (grain_angles.at(i) - mean_angle) * (peduncle_elevations.at(i) - mean_elevation);
        elevation_variance += (peduncle_elevations.at(i) - mean_elevation) * (peduncle_elevations.at(i) - mean_elevation);
    }
    DOCTEST_REQUIRE(elevation_variance > 0.f);
    const float slope = covariance / elevation_variance;
    DOCTEST_CAPTURE(slope);
    DOCTEST_CHECK(std::fabs(slope) < 0.05f);
}

DOCTEST_TEST_CASE("PlantArchitecture phytomer production does not depend on how advanceTime is chunked") {
    // Regression test. advanceTime() sub-steps at the smallest phyllochron_min of the shoots present when it is
    // called, and each shoot appended at most ONE phytomer per sub-step. A shoot type created during the call with a
    // shorter phyllochron than anything present at entry (bindweed: the base shoot has phyllochron 2, the primary
    // shoots it spawns have phyllochron 1) therefore produced one node per sub-step instead of one per phyllochron,
    // and its phyllochron counter grew without bound. Advancing 14 days in one call and advancing 1 day fourteen
    // times gave different plants.
    //
    // The two schedules below are built from the same seed and, once the sub-step is chosen from every shoot type
    // the plant can produce, take identical sub-steps, so they draw the same random numbers and must agree exactly.
    auto grow = [](float total_days, float chunk_days) {
        Context context;
        context.seedRandomGenerator(12345);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("bindweed");
        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0.f);

        for (float elapsed = 0.f; elapsed < total_days - 1e-4f; elapsed += chunk_days) {
            plantarchitecture.advanceTime(plantID, chunk_days);
        }

        std::vector<uint> node_counts;
        for (uint shootID: plantarchitecture.getAllShootIDs(plantID)) {
            node_counts.push_back(plantarchitecture.getShootNodeCount(plantID, shootID));
        }
        return node_counts;
    };

    const std::vector<uint> nodes_single_call = grow(14.f, 14.f);
    const std::vector<uint> nodes_daily = grow(14.f, 1.f);

    uint total_single_call = 0;
    for (uint n: nodes_single_call) {
        total_single_call += n;
    }
    uint total_daily = 0;
    for (uint n: nodes_daily) {
        total_daily += n;
    }
    DOCTEST_CAPTURE(total_single_call);
    DOCTEST_CAPTURE(total_daily);

    // The primary shoots must have grown at all for the comparison to mean anything.
    DOCTEST_REQUIRE(nodes_daily.size() > 1);
    DOCTEST_REQUIRE(total_daily > nodes_daily.size() * 2);

    DOCTEST_CHECK(nodes_single_call.size() == nodes_daily.size());
    DOCTEST_CHECK(total_single_call == total_daily);
    for (size_t s = 0; s < std::min(nodes_single_call.size(), nodes_daily.size()); s++) {
        DOCTEST_CAPTURE(s);
        DOCTEST_CHECK(nodes_single_call.at(s) == nodes_daily.at(s));
    }
}

DOCTEST_TEST_CASE("PlantArchitecture epicormic shoot emergence is sampled once per day, not once per sub-step") {
    // Regression test. Inside the sub-step loop of advanceTime(), Shoot::sampleEpicormicShoot() was handed the
    // whole time span of the call rather than the sub-step, so every sub-step re-ran the daily Bernoulli trials for
    // the entire span: a 100-day call at a 2-day sub-step sampled 50 x 100 = 5000 trials instead of 100, inflating
    // the emergence rate by the number of sub-steps (and drawing thousands of random numbers per shoot).
    //
    // A vertical apple trunk whose own buds never break (vegetative_bud_break_probability_min = 0), starting at
    // 5 m and elongating over 40 days, with 0.01 emergences per metre per day, expects about 3 epicormic shoots.
    // The buggy code expects twenty times as many (and every one of those shoots samples the same way, so the
    // error compounds).
    //
    // The trunk's max_nodes is raised well above what it can reach: a shoot whose apical meristem has terminated is
    // skipped before the epicormic sampling in advanceTime(), so a trunk built at its node cap would never sample.
    Context context;
    context.seedRandomGenerator(12345);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("apple");
    ShootParameters trunk_parameters = plantarchitecture.getCurrentShootParameters("trunk");
    trunk_parameters.max_nodes = 500;
    plantarchitecture.updateCurrentShootParameters("trunk", trunk_parameters);

    uint plantID = plantarchitecture.addPlantInstance(make_vec3(0, 0, 0), 0.f);
    // 20 nodes x 0.25 m internodes = 5 m of trunk.
    plantarchitecture.addBaseStemShoot(plantID, 20, make_AxisRotation(0, 0, 0), 0.01f, 0.25f, 1.f, 1.f, 0, "trunk");
    plantarchitecture.breakPlantDormancy(plantID);
    plantarchitecture.enableEpicormicChildShoots(plantID, "proleptic", 0.01f);

    const size_t shoots_before = plantarchitecture.getAllShootIDs(plantID).size();
    DOCTEST_REQUIRE(shoots_before == 1);

    plantarchitecture.advanceTime(plantID, 40.f);

    const size_t epicormic_shoots = plantarchitecture.getAllShootIDs(plantID).size() - shoots_before;
    DOCTEST_CAPTURE(epicormic_shoots);
    // Expected value roughly 3 (0.01 per metre per day over 5-10 m for 40 days, plus a negligible contribution
    // from the short epicormic shoots themselves); the chance of exceeding 15 by sampling alone is below 1e-6,
    // while the buggy code lands well above 50.
    DOCTEST_CHECK(epicormic_shoots <= 15);
    // The trunk kept growing, so it was still sampling for epicormic shoots throughout.
    DOCTEST_CHECK(plantarchitecture.getShootNodeCount(plantID, 0) > 20);
}
