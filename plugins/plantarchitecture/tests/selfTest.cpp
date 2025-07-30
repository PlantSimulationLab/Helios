#include "PlantArchitecture.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

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
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("almond"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - apple") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("apple"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - asparagus") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("asparagus"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - bindweed") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bindweed"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - bean") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("bean"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - cheeseweed") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("cheeseweed"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - cowpea") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("cowpea"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - grapevine_VSP") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - maize") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("maize"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - olive") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("olive"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - pistachio") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("pistachio"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - puncturevine") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("puncturevine"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - easternredbud") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("easternredbud"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - rice") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("rice"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - butterlettuce") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("butterlettuce"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - sorghum") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("sorghum"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - soybean") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("soybean"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - strawberry") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("strawberry"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - sugarbeet") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("sugarbeet"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - tomato") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("tomato"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - walnut") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("walnut"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("Plant Library Model Building - wheat") {
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    Context context;
    PlantArchitecture plantarchitecture(&context);
    DOCTEST_CHECK_NOTHROW(plantarchitecture.loadPlantModelFromLibrary("wheat"));
    DOCTEST_CHECK_NOTHROW(plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 5000));

    std::cout.rdbuf(old);
}

DOCTEST_TEST_CASE("PlantArchitecture writeTreeQSM") {
    Context context;
    PlantArchitecture plantarchitecture(&context);

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

    // Test with invalid plant ID
    DOCTEST_CHECK_THROWS(plantarchitecture.writeQSMCylinderFile(999, "invalid_plant.txt"));
}

int PlantArchitecture::selfTest() {
    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}
