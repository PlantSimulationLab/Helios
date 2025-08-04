#include "PlantArchitecture.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

double err_tol = 1e-7;

DOCTEST_TEST_CASE("PlantArchitecture Constructor") {
    Context context;
    DOCTEST_CHECK_NOTHROW(PlantArchitecture pa_test(&context));
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
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("almond"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - apple") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("apple"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - asparagus") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("asparagus"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - bindweed") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bindweed"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - bean") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - cheeseweed") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("cheeseweed"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - cowpea") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("cowpea"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - grapevine_VSP") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - maize") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("maize"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - olive") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("olive"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - pistachio") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("pistachio"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - puncturevine") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("puncturevine"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - easternredbud") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("easternredbud"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - rice") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("rice"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - butterlettuce") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("butterlettuce"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - sorghum") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("sorghum"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - soybean") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("soybean"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - strawberry") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("strawberry"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - sugarbeet") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("sugarbeet"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - tomato") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("tomato"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - walnut") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("walnut"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("Plant Library Model Building - wheat") {
    capture_cout cout_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("wheat"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));
}

DOCTEST_TEST_CASE("PlantArchitecture writeTreeQSM") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

    // Build a simple plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5);

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

    // Test with invalid plant ID
    DOCTEST_CHECK_THROWS(plantarchitecture.writeQSMCylinderFile(999, "invalid_plant.txt"));
}

// ========== Hierarchical Shoot Query Method Tests ==========

DOCTEST_TEST_CASE("PlantArchitecture getShootIDsByRank basic functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    // Build a very young bean plant
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5);
    
    // Test getShootIDsByRank
    auto shoots_by_rank = plantarchitecture.getShootIDsByRank(plantID);
    
    // Should have at least rank 0 (base shoots)
    DOCTEST_CHECK(shoots_by_rank.size() >= 1);
    DOCTEST_CHECK(!shoots_by_rank[0].empty()); // Should have base shoots
    
    // Verify ranks are sequential
    for (size_t rank = 0; rank < shoots_by_rank.size(); ++rank) {
        for (uint shootID : shoots_by_rank[rank]) {
            uint actual_rank = plantarchitecture.getShootRank(plantID, shootID);
            DOCTEST_CHECK(actual_rank == rank);
        }
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getShootIDsByRank with tree plant") {
    capture_cout cout_buffer;

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    // Build a young almond tree (just after initial training)
    plantarchitecture.loadPlantModelFromLibrary("almond");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 200);
    
    auto shoots_by_rank = plantarchitecture.getShootIDsByRank(plantID);
    
    // Should have at least base shoots
    DOCTEST_CHECK(shoots_by_rank.size() >= 1);
    DOCTEST_CHECK(!shoots_by_rank[0].empty());

}

DOCTEST_TEST_CASE("PlantArchitecture getShootHierarchyMap functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 10);
    
    auto hierarchy_map = plantarchitecture.getShootHierarchyMap(plantID);
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    // Verify parent-child relationships are consistent
    for (const auto& parent_children : hierarchy_map) {
        uint parent_id = parent_children.first;
        const auto& children = parent_children.second;
        
        // Parent should exist
        bool parent_exists = std::find(all_shoots.begin(), all_shoots.end(), parent_id) != all_shoots.end();
        DOCTEST_CHECK(parent_exists);
        
        // Each child should report this parent
        for (uint child_id : children) {
            int reported_parent = plantarchitecture.getParentShootID(plantID, child_id);
            DOCTEST_CHECK(reported_parent == static_cast<int>(parent_id));
        }
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getAllDescendantShootIDs functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 15);
    
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    if (all_shoots.size() > 1) {
        // Test with first shoot (should be base shoot)
        uint base_shoot = all_shoots[0];
        auto descendants = plantarchitecture.getAllDescendantShootIDs(plantID, base_shoot);
        
        // All descendants should have higher or equal rank than parent
        uint parent_rank = plantarchitecture.getShootRank(plantID, base_shoot);
        for (uint descendant : descendants) {
            uint descendant_rank = plantarchitecture.getShootRank(plantID, descendant);
            DOCTEST_CHECK(descendant_rank > parent_rank);
        }
        
        // No descendant should be the parent itself
        bool parent_in_descendants = std::find(descendants.begin(), descendants.end(), base_shoot) != descendants.end();
        DOCTEST_CHECK(!parent_in_descendants);
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getChildShootIDs functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 12);
    
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    for (uint shootID : all_shoots) {
        auto children = plantarchitecture.getChildShootIDs(plantID, shootID);
        
        // Each child should report this shoot as parent
        for (uint child : children) {
            int parent = plantarchitecture.getParentShootID(plantID, child);
            DOCTEST_CHECK(parent == static_cast<int>(shootID));
        }
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getParentShootID functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 8);
    
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    auto shoots_by_rank = plantarchitecture.getShootIDsByRank(plantID);
    
    // Base shoots (rank 0) should have parent ID of -1
    for (uint base_shoot : shoots_by_rank[0]) {
        int parent = plantarchitecture.getParentShootID(plantID, base_shoot);
        DOCTEST_CHECK(parent == -1);
    }
    
    // Higher rank shoots should have valid parents
    for (size_t rank = 1; rank < shoots_by_rank.size(); ++rank) {
        for (uint shootID : shoots_by_rank[rank]) {
            int parent = plantarchitecture.getParentShootID(plantID, shootID);
            DOCTEST_CHECK(parent >= 0);
            
            // Parent should exist in all_shoots
            bool parent_exists = std::find(all_shoots.begin(), all_shoots.end(), static_cast<uint>(parent)) != all_shoots.end();
            DOCTEST_CHECK(parent_exists);
        }
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getShootRank functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 7);
    
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    // Test that ranks are consistent with hierarchy
    for (uint shootID : all_shoots) {
        uint rank = plantarchitecture.getShootRank(plantID, shootID);
        int parent = plantarchitecture.getParentShootID(plantID, shootID);
        
        if (parent == -1) {
            // Base shoots should have rank 0
            DOCTEST_CHECK(rank == 0);
        } else {
            // Child rank should be parent rank + 1
            uint parent_rank = plantarchitecture.getShootRank(plantID, static_cast<uint>(parent));
            DOCTEST_CHECK(rank == parent_rank + 1);
        }
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getAllShootIDs functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 6);
    
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    // Should have at least one shoot
    DOCTEST_CHECK(!all_shoots.empty());
    
    // All IDs should be valid and sequential starting from 0
    for (size_t i = 0; i < all_shoots.size(); ++i) {
        DOCTEST_CHECK(all_shoots[i] == i);
    }
    
    // Test that all shoots can be queried
    for (uint shootID : all_shoots) {
        uint rank = 0;
        int parent_id = 0;
        DOCTEST_CHECK_NOTHROW(rank = plantarchitecture.getShootRank(plantID, shootID));
        DOCTEST_CHECK_NOTHROW(parent_id = plantarchitecture.getParentShootID(plantID, shootID));
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getTerminalShootIDs functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 9);
    
    auto terminal_shoots = plantarchitecture.getTerminalShootIDs(plantID);
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    // Should have at least some terminal shoots
    DOCTEST_CHECK(!terminal_shoots.empty());
    
    // All terminal shoots should have no children
    for (uint terminal_shoot : terminal_shoots) {
        auto children = plantarchitecture.getChildShootIDs(plantID, terminal_shoot);
        DOCTEST_CHECK(children.empty());
        
        // Should be in all_shoots
        bool found = std::find(all_shoots.begin(), all_shoots.end(), terminal_shoot) != all_shoots.end();
        DOCTEST_CHECK(found);
    }
    
    // Non-terminal shoots should have children
    for (uint shootID : all_shoots) {
        auto children = plantarchitecture.getChildShootIDs(plantID, shootID);
        bool is_terminal = std::find(terminal_shoots.begin(), terminal_shoots.end(), shootID) != terminal_shoots.end();
        
        if (children.empty()) {
            DOCTEST_CHECK(is_terminal);
        } else {
            DOCTEST_CHECK(!is_terminal);
        }
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getShootDepth functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000);
    
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    // Test that depth equals rank (in this implementation)
    for (uint shootID : all_shoots) {
        uint depth = plantarchitecture.getShootDepth(plantID, shootID);
        uint rank = plantarchitecture.getShootRank(plantID, shootID);
        DOCTEST_CHECK(depth == rank);
    }

}

DOCTEST_TEST_CASE("PlantArchitecture getPathToRoot functionality") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 11);
    
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    
    for (uint shootID : all_shoots) {
        auto path = plantarchitecture.getPathToRoot(plantID, shootID);
        
        // Path should not be empty
        DOCTEST_CHECK(!path.empty());
        
        // First element should be the shoot itself
        DOCTEST_CHECK(path[0] == shootID);
        
        // Last element should be a root shoot (parent == -1)
        uint root_shoot = path.back();
        int root_parent = plantarchitecture.getParentShootID(plantID, root_shoot);
        DOCTEST_CHECK(root_parent == -1);
        
        // Each consecutive pair should be child-parent relationship
        for (size_t i = 0; i < path.size() - 1; ++i) {
            uint child = path[i];
            uint expected_parent = path[i + 1];
            int actual_parent = plantarchitecture.getParentShootID(plantID, child);
            DOCTEST_CHECK(actual_parent == static_cast<int>(expected_parent));
        }
        
        // Path length should equal rank + 1
        uint rank = plantarchitecture.getShootRank(plantID, shootID);
        DOCTEST_CHECK(path.size() == rank + 1);
    }

}

// ========== Error Handling Tests ==========

DOCTEST_TEST_CASE("PlantArchitecture hierarchical methods invalid plant ID") {
    capture_cerr cerr_buffer;
    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    uint invalid_plant_id = 999;
    
    // All methods should throw for invalid plant ID
    std::vector<std::vector<uint>> rank_result;
    std::map<uint, std::vector<uint>> hierarchy_result;
    std::vector<uint> descendants_result;
    std::vector<uint> children_result;
    int parent_result;
    uint rank_val;
    std::vector<uint> all_shoots_result;
    std::vector<uint> terminal_result;
    uint depth_result;
    std::vector<uint> path_result;
    
    DOCTEST_CHECK_THROWS(rank_result = plantarchitecture.getShootIDsByRank(invalid_plant_id));
    DOCTEST_CHECK_THROWS(hierarchy_result = plantarchitecture.getShootHierarchyMap(invalid_plant_id));
    DOCTEST_CHECK_THROWS(descendants_result = plantarchitecture.getAllDescendantShootIDs(invalid_plant_id, 0));
    DOCTEST_CHECK_THROWS(children_result = plantarchitecture.getChildShootIDs(invalid_plant_id, 0));
    DOCTEST_CHECK_THROWS(parent_result = plantarchitecture.getParentShootID(invalid_plant_id, 0));
    DOCTEST_CHECK_THROWS(rank_val = plantarchitecture.getShootRank(invalid_plant_id, 0));
    DOCTEST_CHECK_THROWS(all_shoots_result = plantarchitecture.getAllShootIDs(invalid_plant_id));
    DOCTEST_CHECK_THROWS(terminal_result = plantarchitecture.getTerminalShootIDs(invalid_plant_id));
    DOCTEST_CHECK_THROWS(depth_result = plantarchitecture.getShootDepth(invalid_plant_id, 0));
    DOCTEST_CHECK_THROWS(path_result = plantarchitecture.getPathToRoot(invalid_plant_id, 0));
}

DOCTEST_TEST_CASE("PlantArchitecture hierarchical methods invalid shoot ID") {
    capture_cout cout_buffer;

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 3);
    
    uint invalid_shoot_id = 999;
    
    // Methods that take shoot ID should throw for invalid shoot ID
    std::vector<uint> descendants_result2;
    std::vector<uint> children_result2;
    int parent_result2;
    uint rank_val2;
    uint depth_result2;
    std::vector<uint> path_result2;
    
    DOCTEST_CHECK_THROWS(descendants_result2 = plantarchitecture.getAllDescendantShootIDs(plantID, invalid_shoot_id));
    DOCTEST_CHECK_THROWS(children_result2 = plantarchitecture.getChildShootIDs(plantID, invalid_shoot_id));
    DOCTEST_CHECK_THROWS(parent_result2 = plantarchitecture.getParentShootID(plantID, invalid_shoot_id));
    DOCTEST_CHECK_THROWS(rank_val2 = plantarchitecture.getShootRank(plantID, invalid_shoot_id));
    DOCTEST_CHECK_THROWS(depth_result2 = plantarchitecture.getShootDepth(plantID, invalid_shoot_id));
    DOCTEST_CHECK_THROWS(path_result2 = plantarchitecture.getPathToRoot(plantID, invalid_shoot_id));
}

// ========== Edge Case Tests ==========

DOCTEST_TEST_CASE("PlantArchitecture hierarchical methods single shoot plant") {

    Context context;
    PlantArchitecture plantarchitecture(&context);
    
    plantarchitecture.loadPlantModelFromLibrary("maize");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 1); // Very young plant
    
    auto shoots_by_rank = plantarchitecture.getShootIDsByRank(plantID);
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    auto terminal_shoots = plantarchitecture.getTerminalShootIDs(plantID);
    
    // Should have exactly one shoot at rank 0
    DOCTEST_CHECK(shoots_by_rank.size() == 1);
    DOCTEST_CHECK(shoots_by_rank[0].size() == 1);
    DOCTEST_CHECK(all_shoots.size() == 1);
    DOCTEST_CHECK(terminal_shoots.size() == 1);
    
    uint shoot_id = all_shoots[0];
    
    // Single shoot should be root and terminal
    DOCTEST_CHECK(plantarchitecture.getParentShootID(plantID, shoot_id) == -1);
    DOCTEST_CHECK(plantarchitecture.getShootRank(plantID, shoot_id) == 0);
    DOCTEST_CHECK(plantarchitecture.getChildShootIDs(plantID, shoot_id).empty());
    DOCTEST_CHECK(plantarchitecture.getAllDescendantShootIDs(plantID, shoot_id).empty());
    
    auto path = plantarchitecture.getPathToRoot(plantID, shoot_id);
    DOCTEST_CHECK(path.size() == 1);
    DOCTEST_CHECK(path[0] == shoot_id);

}

DOCTEST_TEST_CASE("PlantArchitecture hierarchical methods consistency check") {

    Context context;
    PlantArchitecture plantarchitecture(&context);

    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 13);

    auto shoots_by_rank = plantarchitecture.getShootIDsByRank(plantID);
    auto hierarchy_map = plantarchitecture.getShootHierarchyMap(plantID);
    auto all_shoots = plantarchitecture.getAllShootIDs(plantID);
    auto terminal_shoots = plantarchitecture.getTerminalShootIDs(plantID);

    // Count shoots in shoots_by_rank should equal total shoots
    size_t total_in_ranks = 0;
    for (const auto& rank_shoots : shoots_by_rank) {
        total_in_ranks += rank_shoots.size();
    }
    DOCTEST_CHECK(total_in_ranks == all_shoots.size());

    // All shoots should appear exactly once in shoots_by_rank
    std::set<uint> shoots_in_ranks;
    for (const auto& rank_shoots : shoots_by_rank) {
        for (uint shootID : rank_shoots) {
            DOCTEST_CHECK(shoots_in_ranks.find(shootID) == shoots_in_ranks.end()); // Should not be duplicate
            shoots_in_ranks.insert(shootID);
        }
    }
    DOCTEST_CHECK(shoots_in_ranks.size() == all_shoots.size());

    // Terminal shoots + non-terminal shoots should equal all shoots
    std::set<uint> non_terminal_shoots;
    for (const auto& parent_children : hierarchy_map) {
        if (!parent_children.second.empty()) {
            non_terminal_shoots.insert(parent_children.first);
        }
    }
    DOCTEST_CHECK(terminal_shoots.size() + non_terminal_shoots.size() == all_shoots.size());

}

int PlantArchitecture::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
