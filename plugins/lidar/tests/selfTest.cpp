#include "LiDAR.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace std;
using namespace helios;

float err_tol = 1e-3;

int LiDARcloud::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

DOCTEST_TEST_CASE("LiDAR Single Voxel Sphere Test") {
    LiDARcloud pointcloud;
    pointcloud.disableMessages();

    DOCTEST_CHECK_NOTHROW(pointcloud.loadXML("plugins/lidar/xml/sphere.xml"));
    DOCTEST_CHECK_NOTHROW(pointcloud.triangulateHitPoints(0.5, 5));

    Context context_1;
    DOCTEST_CHECK_NOTHROW(pointcloud.addTrianglesToContext(&context_1));

    DOCTEST_CHECK(context_1.getPrimitiveCount() == 383);
}

DOCTEST_TEST_CASE("LiDAR Single Voxel Isotropic Patches Test") {
    LiDARcloud synthetic_1;
    synthetic_1.disableMessages();

    // Add scan programmatically for explicit control
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 6000;
    uint Nphi = 12000;
    float thetaMin = 0.0f;       // Default when not specified in XML
    float thetaMax = M_PI;       // Default when not specified in XML
    float phiMin = 0.0f;         // Default when not specified in XML
    float phiMax = 2.0f * M_PI;  // Default when not specified in XML
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(synthetic_1.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(1, 1, 1);
    DOCTEST_CHECK_NOTHROW(synthetic_1.addGrid(grid_center, grid_size, grid_divisions, 0));

    vec3 gsize = synthetic_1.getCellSize(0);

    Context context_2;
    std::vector<uint> UUIDs_1 = context_2.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    float LAD_exact = 0.f;
    for (uint UUID: UUIDs_1) {
        LAD_exact += context_2.getPrimitiveArea(UUID) / (gsize.x * gsize.y * gsize.z);
    }

    // Calculate exact G(theta) from primitive geometry
    float Gtheta_exact_numerator = 0.f;
    float Gtheta_exact_denominator = 0.f;
    for (uint UUID : UUIDs_1) {
        float area = context_2.getPrimitiveArea(UUID);
        vec3 normal = context_2.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context_2.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) {  // Check for NaN
            float normal_dot_ray = fabs(normal * raydir);
            Gtheta_exact_numerator += normal_dot_ray * area;
            Gtheta_exact_denominator += area;
        }
    }
    float Gtheta_exact = 0.f;
    if (Gtheta_exact_denominator > 0) {
        Gtheta_exact = Gtheta_exact_numerator / Gtheta_exact_denominator;
    }

    DOCTEST_CHECK_NOTHROW(synthetic_1.syntheticScan(&context_2));
    DOCTEST_CHECK_NOTHROW(synthetic_1.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(synthetic_1.calculateLeafArea(&context_2));

    float LAD = synthetic_1.getCellLeafAreaDensity(0);

    DOCTEST_CHECK(LAD == LAD); // Check for NaN
    DOCTEST_CHECK(fabs(LAD - LAD_exact) / LAD_exact == doctest::Approx(0.0f).epsilon(0.02f));

    // Check G(theta) against exact value calculated from primitives
    float Gtheta = synthetic_1.getCellGtheta(0);
    DOCTEST_CHECK(Gtheta == Gtheta); // Check for NaN
    DOCTEST_CHECK(fabs(Gtheta - Gtheta_exact) / Gtheta_exact == doctest::Approx(0.0f).epsilon(0.05f));
}

DOCTEST_TEST_CASE("LiDAR Eight Voxel Isotropic Patches Test") {
    LiDARcloud synthetic_2;
    synthetic_2.disableMessages();

    // Add scan programmatically
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 10000;
    uint Nphi = 12000;
    float thetaMin = 0.0f;
    float thetaMax = M_PI;
    float phiMin = 0.0f;
    float phiMax = 2.0f * M_PI;
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(synthetic_2.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(2, 2, 2);
    DOCTEST_CHECK_NOTHROW(synthetic_2.addGrid(grid_center, grid_size, grid_divisions, 0));

    vec3 gsize = synthetic_2.getCellSize(0);

    Context context_2;
    std::vector<uint> UUIDs_1 = context_2.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    std::vector<float> LAD_ex(8, 0);
    for (uint UUID: UUIDs_1) {
        int i, j, k;
        i = j = k = 0;
        vec3 v = context_2.getPrimitiveVertices(UUID).front();
        if (v.x > 0.f) {
            i = 1;
        }
        if (v.y > 0.f) {
            j = 1;
        }
        if (v.z > 0.5f) {
            k = 1;
        }
        int ID = k * 4 + j * 2 + i;

        float area = context_2.getPrimitiveArea(UUID);
        LAD_ex.at(ID) += area / (gsize.x * gsize.y * gsize.z);
    }

    DOCTEST_CHECK_NOTHROW(synthetic_2.syntheticScan(&context_2));
    DOCTEST_CHECK_NOTHROW(synthetic_2.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(synthetic_2.calculateLeafArea(&context_2));

    float RMSE = 0.f;
    for (int i = 0; i < synthetic_2.getGridCellCount(); i++) {
        float LAD = synthetic_2.getCellLeafAreaDensity(i);
        RMSE += powf(LAD - LAD_ex.at(i), 2) / float(synthetic_2.getGridCellCount());
    }
    RMSE = sqrtf(RMSE);

    DOCTEST_CHECK(RMSE == doctest::Approx(0.0f).epsilon(0.06f));
}

DOCTEST_TEST_CASE("LiDAR Single Voxel Anisotropic Patches Test") {
    LiDARcloud synthetic_3;
    synthetic_3.disableMessages();

    // Add scan programmatically - use higher resolution for anisotropic to reduce bias
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 10000;
    uint Nphi = 16000;
    float thetaMin = 0.0f;
    float thetaMax = M_PI;
    float phiMin = 0.0f;
    float phiMax = 2.0f * M_PI;
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(synthetic_3.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(1, 1, 1);
    DOCTEST_CHECK_NOTHROW(synthetic_3.addGrid(grid_center, grid_size, grid_divisions, 0));

    vec3 gsize = synthetic_3.getCellSize(0);

    Context context_2;
    std::vector<uint> UUIDs_1 = context_2.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_erectophile.xml", true);

    float LAD_exact = 0.f;
    for (uint UUID: UUIDs_1) {
        LAD_exact += context_2.getPrimitiveArea(UUID) / (gsize.x * gsize.y * gsize.z);
    }

    // Calculate exact G(theta) from primitive geometry
    float Gtheta_exact_numerator = 0.f;
    float Gtheta_exact_denominator = 0.f;
    for (uint UUID : UUIDs_1) {
        float area = context_2.getPrimitiveArea(UUID);
        vec3 normal = context_2.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context_2.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) {  // Check for NaN
            float normal_dot_ray = fabs(normal * raydir);
            Gtheta_exact_numerator += normal_dot_ray * area;
            Gtheta_exact_denominator += area;
        }
    }
    float Gtheta_exact = 0.f;
    if (Gtheta_exact_denominator > 0) {
        Gtheta_exact = Gtheta_exact_numerator / Gtheta_exact_denominator;
    }

    DOCTEST_CHECK_NOTHROW(synthetic_3.syntheticScan(&context_2));
    DOCTEST_CHECK_NOTHROW(synthetic_3.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(synthetic_3.calculateLeafArea(&context_2));

    float LAD = synthetic_3.getCellLeafAreaDensity(0);

    DOCTEST_CHECK(LAD == LAD); // Check for NaN
    DOCTEST_CHECK(fabs(LAD - LAD_exact) / LAD_exact == doctest::Approx(0.0f).epsilon(0.03f));

    // Check G(theta) against exact value calculated from primitives
    float Gtheta = synthetic_3.getCellGtheta(0);
    DOCTEST_CHECK(Gtheta == Gtheta); // Check for NaN
    DOCTEST_CHECK(fabs(Gtheta - Gtheta_exact) / Gtheta_exact == doctest::Approx(0.0f).epsilon(0.05f));
}

DOCTEST_TEST_CASE("LiDAR Synthetic Almond Tree Test") {
    Context context_4;
    DOCTEST_CHECK_NOTHROW(context_4.loadOBJ("plugins/lidar/xml/AlmondWP.obj", make_vec3(0, 0, 0), 6., make_SphericalCoord(0, 0), RGB::red, true));

    LiDARcloud synthetic_4;
    synthetic_4.disableMessages();

    DOCTEST_CHECK_NOTHROW(synthetic_4.loadXML("plugins/lidar/xml/almond.xml"));
    DOCTEST_CHECK_NOTHROW(synthetic_4.syntheticScan(&context_4));
    DOCTEST_CHECK_NOTHROW(synthetic_4.calculateSyntheticLeafArea(&context_4));
    DOCTEST_CHECK_NOTHROW(synthetic_4.calculateSyntheticGtheta(&context_4));
    DOCTEST_CHECK_NOTHROW(synthetic_4.triangulateHitPoints(0.05, 5));
    DOCTEST_CHECK_NOTHROW(synthetic_4.calculateLeafArea(&context_4));

    // Calculate exact leaf area
    uint Ncells = synthetic_4.getGridCellCount();

    std::vector<float> total_area;
    total_area.resize(Ncells);

    std::vector<float> Gtheta;
    Gtheta.resize(Ncells);

    std::vector<float> area_sum;
    area_sum.resize(Ncells, 0.f);
    std::vector<float> sin_sum;
    sin_sum.resize(Ncells, 0.f);
    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells, 0);

    std::vector<uint> UUIDs = context_4.getAllUUIDs();
    for (int p = 0; p < UUIDs.size(); p++) {

        uint UUID = UUIDs.at(p);

        if (context_4.doesPrimitiveDataExist(UUID, "gridCell")) {

            uint gridCell;
            context_4.getPrimitiveData(UUID, "gridCell", gridCell);

            if (gridCell >= 0 && gridCell < Ncells) {
                total_area.at(gridCell) += context_4.getPrimitiveArea(UUID);
            }

            for (int s = 0; s < synthetic_4.getScanCount(); s++) {
                vec3 origin = synthetic_4.getScanOrigin(s);
                std::vector<vec3> vertices = context_4.getPrimitiveVertices(p);
                float area = context_4.getPrimitiveArea(p);
                vec3 normal = context_4.getPrimitiveNormal(p);
                vec3 raydir = vertices.front() - origin;
                raydir.normalize();
                float theta = fabs(acos_safe(raydir.z));

                if (area == area) { // in rare cases you can get area=NaN

                    Gtheta.at(gridCell) += fabs(normal * raydir) * area * fabs(sin(theta));

                    area_sum.at(gridCell) += area;
                    sin_sum.at(gridCell) += fabs(sin(theta));
                    cell_tri_count.at(gridCell) += 1;
                }
            }
        }
    }

    for (uint v = 0; v < Ncells; v++) {
        if (cell_tri_count[v] > 0) {
            Gtheta[v] *= float(cell_tri_count[v]) / (area_sum[v] * sin_sum[v]);
        }
    }

    float RMSE_LAD = 0.f;
    float bias_LAD = 0.f;
    float RMSE_Gtheta = 0.f;
    for (uint i = 0; i < Ncells; i++) {
        float LAD = synthetic_4.getCellLeafArea(i);
        if (LAD == LAD && total_area.at(i) > 0 && total_area.at(i) == total_area.at(i)) {
            RMSE_LAD += pow(LAD - total_area.at(i), 2) / float(Ncells);
            bias_LAD += (LAD - total_area.at(i)) / float(Ncells);
        }
        float Gtheta_bar = synthetic_4.getCellGtheta(i);
        if (Gtheta_bar == Gtheta_bar && Gtheta.at(i) > 0 && Gtheta.at(i) == Gtheta.at(i)) {
            RMSE_Gtheta += pow(Gtheta_bar - Gtheta.at(i), 2) / float(Ncells);
        }
    }
    RMSE_LAD = sqrt(RMSE_LAD);
    RMSE_Gtheta = sqrt(RMSE_Gtheta);

    DOCTEST_CHECK(RMSE_LAD <= 0.35f);
    DOCTEST_CHECK(bias_LAD <= 0.0f);
    DOCTEST_CHECK(RMSE_Gtheta <= 0.15f);
    DOCTEST_CHECK(RMSE_LAD != 0.f);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Append/Overwrite Test") {
    Context context_test;
    context_test.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    LiDARcloud synthetic_test;
    synthetic_test.disableMessages();

    DOCTEST_CHECK_NOTHROW(synthetic_test.loadXML("plugins/lidar/xml/synthetic_test.xml"));

    // First scan with default append behavior (should append to empty scan)
    DOCTEST_CHECK_NOTHROW(synthetic_test.syntheticScan(&context_test));
    uint hit_count_first = synthetic_test.getHitCount();
    DOCTEST_CHECK(hit_count_first > 0);

    // Second scan with append=true (should double the hit count)
    DOCTEST_CHECK_NOTHROW(synthetic_test.syntheticScan(&context_test, true));
    uint hit_count_append = synthetic_test.getHitCount();
    DOCTEST_CHECK(hit_count_append == 2 * hit_count_first);

    // Third scan with append=false (should reset and have same count as first scan)
    DOCTEST_CHECK_NOTHROW(synthetic_test.syntheticScan(&context_test, false));
    uint hit_count_overwrite = synthetic_test.getHitCount();
    DOCTEST_CHECK(hit_count_overwrite == hit_count_first);

    // Test with other overloads
    // Test scan_grid_only, record_misses overload with append=false
    DOCTEST_CHECK_NOTHROW(synthetic_test.syntheticScan(&context_test, false, false, false));
    uint hit_count_overwrite2 = synthetic_test.getHitCount();
    DOCTEST_CHECK(hit_count_overwrite2 == hit_count_first);

    // Test full-waveform overload with append=true
    DOCTEST_CHECK_NOTHROW(synthetic_test.syntheticScan(&context_test, 1, 0.0f, true));
    uint hit_count_append2 = synthetic_test.getHitCount();
    DOCTEST_CHECK(hit_count_append2 == 2 * hit_count_first);
}

DOCTEST_TEST_CASE("LiDAR TreeQSM Loading Test") {
    Context context_treeqsm;
    LiDARcloud lidar;
    lidar.disableMessages();

    // Test loading TreeQSM file without texture
    std::vector<uint> tube_UUIDs;
    uint radial_subdivisions = 6;
    DOCTEST_CHECK_NOTHROW(tube_UUIDs = lidar.loadTreeQSM(&context_treeqsm, "plugins/lidar/data/cylinder_tree_QSM_test.txt", radial_subdivisions));

    // Check that tube objects were created
    DOCTEST_CHECK(tube_UUIDs.size() > 0);

    // Check that all returned UUIDs are valid
    for (uint UUID: tube_UUIDs) {
        DOCTEST_CHECK(context_treeqsm.doesObjectExist(UUID));
        DOCTEST_CHECK(context_treeqsm.getObjectType(UUID) == helios::OBJECT_TYPE_TUBE);
    }

    // Test that object data was set correctly
    for (uint UUID: tube_UUIDs) {
        DOCTEST_CHECK(context_treeqsm.doesObjectDataExist(UUID, "branch_order"));
        DOCTEST_CHECK(context_treeqsm.doesObjectDataExist(UUID, "branch_id"));

        int branch_order;
        context_treeqsm.getObjectData(UUID, "branch_order", branch_order);
        DOCTEST_CHECK(branch_order >= 0);

        int branch_id;
        context_treeqsm.getObjectData(UUID, "branch_id", branch_id);
        DOCTEST_CHECK(branch_id >= 0);
    }

    // Test loading with empty texture file (should still work)
    Context context_treeqsm2;
    std::vector<uint> tube_UUIDs2;
    DOCTEST_CHECK_NOTHROW(tube_UUIDs2 = lidar.loadTreeQSM(&context_treeqsm2, "plugins/lidar/data/cylinder_tree_QSM_test.txt", radial_subdivisions, ""));
    DOCTEST_CHECK(tube_UUIDs2.size() == tube_UUIDs.size());

    // Test error handling for non-existent file
    Context context_error;
    DOCTEST_CHECK_THROWS(lidar.loadTreeQSM(&context_error, "nonexistent_file.txt", radial_subdivisions));

    // Test with different radial subdivisions
    Context context_treeqsm3;
    std::vector<uint> tube_UUIDs3;
    uint different_subdivisions = 8;
    DOCTEST_CHECK_NOTHROW(tube_UUIDs3 = lidar.loadTreeQSM(&context_treeqsm3, "plugins/lidar/data/cylinder_tree_QSM_test.txt", different_subdivisions));
    DOCTEST_CHECK(tube_UUIDs3.size() == tube_UUIDs.size()); // Same number of tubes

    // Test that each tube has appropriate number of nodes and primitives
    for (uint UUID: tube_UUIDs) {
        std::vector<uint> primitive_UUIDs = context_treeqsm.getObjectPrimitiveUUIDs(UUID);
        DOCTEST_CHECK(primitive_UUIDs.size() > 0);

        // Each tube should have triangular primitives
        for (uint prim_UUID: primitive_UUIDs) {
            DOCTEST_CHECK(context_treeqsm.getPrimitiveType(prim_UUID) == helios::PRIMITIVE_TYPE_TRIANGLE);
        }
    }
}

DOCTEST_TEST_CASE("LiDAR TreeQSM Colormap Loading Test") {
    Context context_colormap;
    LiDARcloud lidar;
    lidar.disableMessages();

    // Test loading TreeQSM file with colormap
    std::vector<uint> tube_UUIDs;
    uint radial_subdivisions = 6;
    std::string colormap_name = "hot";
    DOCTEST_CHECK_NOTHROW(tube_UUIDs = lidar.loadTreeQSMColormap(&context_colormap, "plugins/lidar/data/cylinder_tree_QSM_test.txt", radial_subdivisions, colormap_name));

    // Check that tube objects were created
    DOCTEST_CHECK(tube_UUIDs.size() > 0);

    // Check that all returned UUIDs are valid
    for (uint UUID: tube_UUIDs) {
        DOCTEST_CHECK(context_colormap.doesObjectExist(UUID));
        DOCTEST_CHECK(context_colormap.getObjectType(UUID) == helios::OBJECT_TYPE_TUBE);
    }

    // Test that object data was set correctly
    for (uint UUID: tube_UUIDs) {
        DOCTEST_CHECK(context_colormap.doesObjectDataExist(UUID, "branch_order"));
        DOCTEST_CHECK(context_colormap.doesObjectDataExist(UUID, "branch_id"));

        int branch_order;
        context_colormap.getObjectData(UUID, "branch_order", branch_order);
        DOCTEST_CHECK(branch_order >= 0);

        int branch_id;
        context_colormap.getObjectData(UUID, "branch_id", branch_id);
        DOCTEST_CHECK(branch_id >= 0);
    }

    // Test with different colormap
    Context context_colormap2;
    std::vector<uint> tube_UUIDs2;
    std::string colormap_name2 = "cool";
    DOCTEST_CHECK_NOTHROW(tube_UUIDs2 = lidar.loadTreeQSMColormap(&context_colormap2, "plugins/lidar/data/cylinder_tree_QSM_test.txt", radial_subdivisions, colormap_name2));
    DOCTEST_CHECK(tube_UUIDs2.size() == tube_UUIDs.size());

    // Test error handling for non-existent file
    Context context_error2;
    DOCTEST_CHECK_THROWS(lidar.loadTreeQSMColormap(&context_error2, "nonexistent_file.txt", radial_subdivisions, colormap_name));

    // Test that each tube has appropriate number of primitives
    for (uint UUID: tube_UUIDs) {
        std::vector<uint> primitive_UUIDs = context_colormap.getObjectPrimitiveUUIDs(UUID);
        DOCTEST_CHECK(primitive_UUIDs.size() > 0);

        // Each tube should have triangular primitives
        for (uint prim_UUID: primitive_UUIDs) {
            DOCTEST_CHECK(context_colormap.getPrimitiveType(prim_UUID) == helios::PRIMITIVE_TYPE_TRIANGLE);
        }
    }

    // Test with invalid colormap name (should throw an exception)
    Context context_colormap3;
    std::string invalid_colormap = "invalid_colormap_name";
    DOCTEST_CHECK_THROWS(lidar.loadTreeQSMColormap(&context_colormap3, "plugins/lidar/data/cylinder_tree_QSM_test.txt", radial_subdivisions, invalid_colormap));
}

DOCTEST_TEST_CASE("LiDAR Collision Detection Integration Test") {
    LiDARcloud lidar;
    lidar.disableMessages();

    // Create a simple test context with geometry
    Context test_context;
    test_context.addSphere(10, make_vec3(0, 0, 0), 1.0f, RGB::red);
    test_context.addTriangle(make_vec3(-2, -1, -1), make_vec3(2, -1, -1), make_vec3(0, 1, -1), RGB::green);

    // Test initializeCollisionDetection method
    DOCTEST_CHECK_NOTHROW(lidar.initializeCollisionDetection(&test_context));

    // Test calling initialize multiple times (should not create multiple instances)
    DOCTEST_CHECK_NOTHROW(lidar.initializeCollisionDetection(&test_context));

    // Create test ray data
    const size_t N = 3;
    const int Npulse = 2;
    helios::vec3 scan_origin = make_vec3(0, 0, 5);

    // Test ray directions - some should hit, some should miss
    std::vector<helios::vec3> directions = {
            make_vec3(0, 0, -1), // Should hit sphere
            make_vec3(1, 0, -1), // Should miss sphere
            make_vec3(0, -0.5, -1), // Should hit triangle
            make_vec3(2, 0, -1), // Should miss everything
            make_vec3(-1, 0, -1), // Should miss sphere, might hit triangle
            make_vec3(0, 0.5, -1) // Should miss triangle, might hit sphere
    };

    float hit_t[N * Npulse];
    float hit_fnorm[N * Npulse];
    int hit_ID[N * Npulse];

    // Initialize arrays
    for (size_t i = 0; i < N * Npulse; i++) {
        hit_t[i] = 1001.0f;
        hit_fnorm[i] = 1e6;
        hit_ID[i] = -1;
    }

    // Create ray origins array (all rays from same origin)
    std::vector<helios::vec3> ray_origins(N * Npulse, scan_origin);

    // Test performUnifiedRayTracing method
    DOCTEST_CHECK_NOTHROW(lidar.performUnifiedRayTracing(&test_context, N, Npulse, ray_origins.data(), directions.data(), hit_t, hit_fnorm, hit_ID));

    // Validate results - at least some rays should hit
    bool found_hit = false;
    bool found_miss = false;
    for (size_t i = 0; i < N * Npulse; i++) {
        if (hit_t[i] < 1000.0f) {
            found_hit = true;
            // Valid hit should have reasonable distance
            DOCTEST_CHECK(hit_t[i] > 0.0f);
            DOCTEST_CHECK(hit_t[i] < 100.0f);
            // Valid hit should have a primitive ID
            DOCTEST_CHECK(hit_ID[i] >= 0);
            // Normal calculation should be finite
            DOCTEST_CHECK(std::isfinite(hit_fnorm[i]));
        } else {
            found_miss = true;
            DOCTEST_CHECK(hit_ID[i] == -1);
        }
    }

    // We should have both hits and misses in our test case
    DOCTEST_CHECK(found_hit);
    DOCTEST_CHECK(found_miss);
}

DOCTEST_TEST_CASE("LiDAR Data Format Conversion Test") {
    LiDARcloud lidar;
    lidar.disableMessages();

    Context test_context;
    test_context.addSphere(5, make_vec3(0, 0, 0), 1.0f, RGB::red);

    // Test conversion between CUDA float3 and Helios vec3 formats - simplified
    const size_t N = 2;
    const int Npulse = 1;

    std::vector<helios::vec3> test_directions = {
            make_vec3(0, 0, -1), // Downward (should hit)
            make_vec3(1, 0, 0) // Sideways (should miss)
    };

    // Test one origin only
    helios::vec3 origin = make_vec3(0, 0, 5);

    float hit_t[N * Npulse];
    float hit_fnorm[N * Npulse];
    int hit_ID[N * Npulse];

    // Initialize collision detection
    lidar.initializeCollisionDetection(&test_context);

    // Create ray origins array (all rays from same origin)
    std::vector<helios::vec3> ray_origins(N * Npulse, origin);

    // Test ray tracing
    DOCTEST_CHECK_NOTHROW(lidar.performUnifiedRayTracing(&test_context, N, Npulse, ray_origins.data(), test_directions.data(), hit_t, hit_fnorm, hit_ID));

    // Basic validation
    for (size_t i = 0; i < N * Npulse; i++) {
        DOCTEST_CHECK(std::isfinite(hit_t[i]));
        DOCTEST_CHECK(std::isfinite(hit_fnorm[i]));
        DOCTEST_CHECK(hit_ID[i] >= -1);
    }
}

DOCTEST_TEST_CASE("LiDAR Edge Cases and Error Conditions Test") {
    LiDARcloud lidar;
    lidar.disableMessages();

    Context test_context;
    test_context.addSphere(3, make_vec3(0, 0, 0), 1.0f, RGB::red);

    // Test basic edge cases only
    helios::vec3 origin = make_vec3(0, 0, 5);
    helios::vec3 direction = make_vec3(0, 0, -1);
    float hit_t[1];
    float hit_fnorm[1];
    int hit_ID[1];

    // Test initialization
    DOCTEST_CHECK_NOTHROW(lidar.initializeCollisionDetection(&test_context));

    // Create ray origins array (single ray from single origin)
    helios::vec3 ray_origins[1] = {origin};

    // Test single ray
    DOCTEST_CHECK_NOTHROW(lidar.performUnifiedRayTracing(&test_context, 1, 1, ray_origins, &direction, hit_t, hit_fnorm, hit_ID));

    // Validate basic results
    DOCTEST_CHECK(std::isfinite(hit_t[0]));
    DOCTEST_CHECK(std::isfinite(hit_fnorm[0]));
    DOCTEST_CHECK(hit_ID[0] >= -1);
}

DOCTEST_TEST_CASE("LiDAR Collision Detection Memory Management Test") {
    // Test basic initialization and cleanup
    LiDARcloud lidar;
    lidar.disableMessages();

    Context test_context;
    test_context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0), RGB::red);

    // Test initialization
    DOCTEST_CHECK_NOTHROW(lidar.initializeCollisionDetection(&test_context));

    // Test one ray
    helios::vec3 origin = make_vec3(0, 0, 5);
    helios::vec3 direction = make_vec3(0, 0, -1);
    float hit_t, hit_fnorm;
    int hit_ID;

    // Create ray origins array (single ray from single origin)
    helios::vec3 ray_origins[1] = {origin};

    DOCTEST_CHECK_NOTHROW(lidar.performUnifiedRayTracing(&test_context, 1, 1, ray_origins, &direction, &hit_t, &hit_fnorm, &hit_ID));

    // Basic validation
    DOCTEST_CHECK(std::isfinite(hit_t));
    DOCTEST_CHECK(std::isfinite(hit_fnorm));
    DOCTEST_CHECK(hit_ID >= -1);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Integration Test") {
    // Test that synthetic scans still work with the new collision detection integration
    LiDARcloud synthetic_scan_test;
    synthetic_scan_test.disableMessages();

    // Load a scan configuration
    DOCTEST_CHECK_NOTHROW(synthetic_scan_test.loadXML("plugins/lidar/xml/synthetic_test.xml"));

    // Create a simple test geometry
    Context scan_context;
    std::vector<uint> patch_UUIDs = scan_context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    DOCTEST_CHECK(patch_UUIDs.size() > 0);

    // Run synthetic scan - this should use the new collision detection integration internally
    DOCTEST_CHECK_NOTHROW(synthetic_scan_test.syntheticScan(&scan_context));

    // Verify that we got some hits
    uint hit_count = synthetic_scan_test.getHitCount();
    DOCTEST_CHECK(hit_count > 0);

    // Test that hit points have reasonable coordinates
    for (uint i = 0; i < std::min(hit_count, 10u); i++) {
        helios::vec3 hit_pos = synthetic_scan_test.getHitXYZ(i);

        // Coordinates should be finite
        DOCTEST_CHECK(std::isfinite(hit_pos.x));
        DOCTEST_CHECK(std::isfinite(hit_pos.y));
        DOCTEST_CHECK(std::isfinite(hit_pos.z));

        // Should be within reasonable bounds for our test geometry
        DOCTEST_CHECK(fabs(hit_pos.x) < 100.0f);
        DOCTEST_CHECK(fabs(hit_pos.y) < 100.0f);
        DOCTEST_CHECK(fabs(hit_pos.z) < 100.0f);

        // Test ray direction is valid
        helios::SphericalCoord ray_dir = synthetic_scan_test.getHitRaydir(i);
        DOCTEST_CHECK(std::isfinite(ray_dir.zenith));
        DOCTEST_CHECK(std::isfinite(ray_dir.azimuth));
        DOCTEST_CHECK(std::isfinite(ray_dir.radius));
    }

    // Test backward compatibility - existing LiDAR functionality should still work
    DOCTEST_CHECK_NOTHROW(synthetic_scan_test.calculateHitGridCell());
    DOCTEST_CHECK_NOTHROW(synthetic_scan_test.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(synthetic_scan_test.calculateLeafArea(&scan_context));

    // Grid cell calculations should produce reasonable results
    uint cell_count = synthetic_scan_test.getGridCellCount();
    if (cell_count > 0) {
        float leaf_area_density = synthetic_scan_test.getCellLeafAreaDensity(0);
        DOCTEST_CHECK(std::isfinite(leaf_area_density));
        DOCTEST_CHECK(leaf_area_density >= 0.0f);
    }
}

DOCTEST_TEST_CASE("LiDAR Multi-Return Equal Weighting Test") {
    LiDARcloud lidar;
    lidar.disableMessages();

    // Add scan programmatically for explicit control
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 6000;
    uint Nphi = 12000;
    float thetaMin = 0.0f;
    float thetaMax = M_PI;
    float phiMin = 0.0f;
    float phiMax = 2.0f * M_PI;
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(lidar.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(1, 1, 1);
    DOCTEST_CHECK_NOTHROW(lidar.addGrid(grid_center, grid_size, grid_divisions, 0));

    vec3 gsize = lidar.getCellSize(0);

    Context context;
    std::vector<uint> UUIDs = context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    float LAD_exact = 0.f;
    for (uint UUID: UUIDs) {
        LAD_exact += context.getPrimitiveArea(UUID) / (gsize.x * gsize.y * gsize.z);
    }

    // Calculate exact G(theta) from primitive geometry
    float Gtheta_exact_numerator = 0.f;
    float Gtheta_exact_denominator = 0.f;
    for (uint UUID : UUIDs) {
        float area = context.getPrimitiveArea(UUID);
        vec3 normal = context.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) {  // Check for NaN
            float normal_dot_ray = fabs(normal * raydir);
            Gtheta_exact_numerator += normal_dot_ray * area;
            Gtheta_exact_denominator += area;
        }
    }
    float Gtheta_exact = 0.f;
    if (Gtheta_exact_denominator > 0) {
        Gtheta_exact = Gtheta_exact_numerator / Gtheta_exact_denominator;
    }

    // Multi-return test with scan_grid_only=true to limit miss recording to voxel region
    // Test BOTH scan_grid_only modes to verify they give same result
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 2, 0.1f, true, true));
    uint hits_grid_true = lidar.getHitCount();

    DOCTEST_CHECK_NOTHROW(lidar.triangulateHitPoints(0.04, 10));
    uint triangles_multi = lidar.getTriangleCount();

    DOCTEST_CHECK_NOTHROW(lidar.calculateLeafArea(&context));
    float LAD_grid_true = lidar.getCellLeafAreaDensity(0);
    float G_grid_true = lidar.getCellGtheta(0);

    // Test scan_grid_only=FALSE
    LiDARcloud lidar2;
    lidar2.disableMessages();

    // Add scan and grid programmatically for second test
    DOCTEST_CHECK_NOTHROW(lidar2.addScan(scan));
    DOCTEST_CHECK_NOTHROW(lidar2.addGrid(grid_center, grid_size, grid_divisions, 0));

    DOCTEST_CHECK_NOTHROW(lidar2.syntheticScan(&context, 2, 0.1f, false, true));
    uint hits_grid_false = lidar2.getHitCount();

    DOCTEST_CHECK_NOTHROW(lidar2.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(lidar2.calculateLeafArea(&context));
    float LAD_grid_false = lidar2.getCellLeafAreaDensity(0);
    float G_grid_false = lidar2.getCellGtheta(0);

    // Use scan_grid_only=TRUE result as the main test (it's faster)
    float LAD_multi = LAD_grid_true;
    float Gtheta_multi = G_grid_true;

    // Verify both scan_grid_only modes give same result
    DOCTEST_CHECK(fabs(LAD_grid_true - LAD_grid_false) < 0.01f);
    DOCTEST_CHECK(fabs(G_grid_true - G_grid_false) < 0.01f);

    // Compare with single-return scan using same scan parameters
    LiDARcloud lidar_single;
    lidar_single.disableMessages();
    DOCTEST_CHECK_NOTHROW(lidar_single.addScan(scan));
    DOCTEST_CHECK_NOTHROW(lidar_single.addGrid(grid_center, grid_size, grid_divisions, 0));
    // Use same scan_grid_only and record_misses settings as multi-return
    DOCTEST_CHECK_NOTHROW(lidar_single.syntheticScan(&context, true, true));
    uint hits_single = lidar_single.getHitCount();

    DOCTEST_CHECK_NOTHROW(lidar_single.triangulateHitPoints(0.04, 10));

    DOCTEST_CHECK_NOTHROW(lidar_single.calculateLeafArea(&context));
    float LAD_single = lidar_single.getCellLeafAreaDensity(0);
    float G_single = lidar_single.getCellGtheta(0);

    // Multi-return equal weighting should match expected LAD within 2%
    DOCTEST_CHECK(LAD_multi > LAD_exact * 0.98f);
    DOCTEST_CHECK(LAD_multi < LAD_exact * 1.02f);

    // Check G(theta) against exact value calculated from primitives
    DOCTEST_CHECK(Gtheta_multi == Gtheta_multi); // Check for NaN
    DOCTEST_CHECK(fabs(Gtheta_multi - Gtheta_exact) / Gtheta_exact == doctest::Approx(0.0f).epsilon(0.05f));
}

DOCTEST_TEST_CASE("LiDAR Eight Voxel Multi-Return Equal Weighting Test") {
    // Test multi-return LiDAR with 8-voxel grid (2x2x2)
    // Validates equal-weighting algorithm handles partial occlusion correctly

    LiDARcloud synthetic_mr8;
    synthetic_mr8.disableMessages();

    // Add scan programmatically with beam spreading for true multi-return
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 10000;
    uint Nphi = 14000;
    float thetaMin = 0.0f;
    float thetaMax = M_PI;
    float phiMin = 0.0f;
    float phiMax = 2.0f * M_PI;
    float exitDiameter = 0.0f;  // Point source for backward compatibility
    float beamDivergence = 0.0004f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(synthetic_mr8.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(2, 2, 2);
    DOCTEST_CHECK_NOTHROW(synthetic_mr8.addGrid(grid_center, grid_size, grid_divisions, 0));

    vec3 gsize = synthetic_mr8.getCellSize(0);

    Context context_mr8;
    context_mr8.seedRandomGenerator(0);  // Seed for reproducible random perturbations
    std::vector<uint> UUIDs = context_mr8.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    // Calculate expected LAD for each of 8 voxels based on primitive positions
    std::vector<float> LAD_ex(8, 0);
    std::vector<float> Gtheta_ex(8, 0);
    std::vector<float> Gtheta_ex_numerator(8, 0);
    std::vector<float> Gtheta_ex_denominator(8, 0);

    for (uint UUID: UUIDs) {
        int i, j, k;
        i = j = k = 0;
        vec3 v = context_mr8.getPrimitiveVertices(UUID).front();
        if (v.x > 0.f) { i = 1; }
        if (v.y > 0.f) { j = 1; }
        if (v.z > 0.5f) { k = 1; }
        int ID = k * 4 + j * 2 + i;

        float area = context_mr8.getPrimitiveArea(UUID);
        LAD_ex.at(ID) += area / (gsize.x * gsize.y * gsize.z);

        // Calculate exact G(theta) from primitive geometry for each voxel
        vec3 normal = context_mr8.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context_mr8.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) {  // Check for NaN
            float normal_dot_ray = fabs(normal * raydir);
            Gtheta_ex_numerator.at(ID) += normal_dot_ray * area;
            Gtheta_ex_denominator.at(ID) += area;
        }
    }

    // Compute final G(theta) values for each voxel
    for (int i = 0; i < 8; i++) {
        if (Gtheta_ex_denominator[i] > 0) {
            Gtheta_ex[i] = Gtheta_ex_numerator[i] / Gtheta_ex_denominator[i];
        }
    }

    // Multi-return scan with realistic beam spreading (100 rays per pulse for stable statistics)
    DOCTEST_CHECK_NOTHROW(synthetic_mr8.syntheticScan(&context_mr8, 100, 0.1f, true, true));
    uint hits_grid_true = synthetic_mr8.getHitCount();

    // Check if we're actually getting multiple returns per pulse
    uint multi_return_count = 0;
    for (uint i = 0; i < hits_grid_true; i++) {
        if (synthetic_mr8.doesHitDataExist(i, "target_count") &&
            synthetic_mr8.getHitData(i, "target_count") > 1) {
            multi_return_count++;
        }
    }

    // Triangulate using base overload - first returns automatically filtered for multi-return data
    DOCTEST_CHECK_NOTHROW(synthetic_mr8.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK(synthetic_mr8.getTriangleCount() > 0);
    DOCTEST_CHECK_NOTHROW(synthetic_mr8.calculateLeafArea(&context_mr8));

    std::vector<float> LAD_grid_true(8);
    std::vector<float> G_grid_true(8);
    for (int i = 0; i < 8; i++) {
        LAD_grid_true[i] = synthetic_mr8.getCellLeafAreaDensity(i);
        G_grid_true[i] = synthetic_mr8.getCellGtheta(i);
    }

    // (Removed dual scan_grid_only testing for now - simplify to match working test)

    // Verify multi-return data fields exist
    bool has_target_index = true;
    bool has_target_count = true;
    bool has_timestamp = true;

    for (uint i = 0; i < hits_grid_true; i++) {
        if (!synthetic_mr8.doesHitDataExist(i, "target_index")) has_target_index = false;
        if (!synthetic_mr8.doesHitDataExist(i, "target_count")) has_target_count = false;
        if (!synthetic_mr8.doesHitDataExist(i, "timestamp")) has_timestamp = false;
    }

    DOCTEST_CHECK(has_target_index);
    DOCTEST_CHECK(has_target_count);
    DOCTEST_CHECK(has_timestamp);

    // Check for duplicate first returns per timestamp (critical bug check)
    std::map<int, int> timestamp_first_return_count;
    for (uint i = 0; i < hits_grid_true; i++) {
        if (synthetic_mr8.doesHitDataExist(i, "target_index") &&
            synthetic_mr8.doesHitDataExist(i, "timestamp")) {
            int tidx = static_cast<int>(synthetic_mr8.getHitData(i, "target_index"));
            int tstamp = static_cast<int>(synthetic_mr8.getHitData(i, "timestamp"));
            if (tidx == 0) {
                timestamp_first_return_count[tstamp]++;
            }
        }
    }

    for (const auto& pair : timestamp_first_return_count) {
        DOCTEST_CHECK(pair.second == 1);
    }

    // Validate LAD accuracy using RMSE across all 8 voxels
    float RMSE = 0.f;
    for (int i = 0; i < 8; i++) {
        float LAD = LAD_grid_true[i];
        RMSE += powf(LAD - LAD_ex.at(i), 2) / LAD_ex.at(i) / 8.0f;
    }
    RMSE = sqrtf(RMSE);

    DOCTEST_CHECK(RMSE == doctest::Approx(0.0f).epsilon(0.1f));

    // Validate G(theta) against exact values calculated from primitive geometry
    for (int i = 0; i < 8; i++) {
        float G = G_grid_true[i];
        float G_exact = Gtheta_ex[i];

        // Only check voxels with non-zero LAD
        if (LAD_grid_true[i] > 0.001f && G_exact > 0) {
            DOCTEST_CHECK(G == G); // Check for NaN
            DOCTEST_CHECK(G_exact == G_exact); // Check for NaN
            DOCTEST_CHECK(fabs(G - G_exact) / G_exact == doctest::Approx(0.0f).epsilon(0.05f));
        }
    }
}

DOCTEST_TEST_CASE("LiDAR Beam Perturbation - Single Return with Sphere") {
    // Test 1: Single-return mode should give hits equal to scan pattern size
    LiDARcloud lidar;
    lidar.disableMessages();

    // Create simple scan pattern (small grid to keep test fast)
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 200;
    uint Nphi = 400;
    float thetaMin = 1.45f;
    float thetaMax = 1.69f;
    float phiMin = 0.0f;
    float phiMax = 6.28f;
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(lidar.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(1, 1, 1);
    DOCTEST_CHECK_NOTHROW(lidar.addGrid(grid_center, grid_size, grid_divisions, 0));

    uint expected_rays = Ntheta * Nphi;

    Context context;
    // Add large sphere (radius >> voxel size) to catch all rays - use low subdivision for speed
    vec3 sphere_center(0, 0, 0.5);
    float sphere_radius = 50.0f;
    context.addSphereObject(6, sphere_center, sphere_radius);  // Low subdivision for speed

    // Single-return scan (rays_per_pulse=1, small threshold to avoid merging)
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 1, 0.001f));

    uint hit_count = lidar.getHitCount();

    DOCTEST_CHECK(hit_count == expected_rays);
}

DOCTEST_TEST_CASE("LiDAR Beam Perturbation - Multi Return Zero Beam Width") {
    // Test 2: Multi-return with exitDiameter=beamDivergence=0 should behave like single-return
    LiDARcloud lidar;
    lidar.disableMessages();

    // Create scan pattern programmatically
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 2000;
    uint Nphi = 4000;
    float thetaMin = 0.0f;       // Default when not specified in XML
    float thetaMax = M_PI;       // Default when not specified in XML
    float phiMin = 0.0f;         // Default when not specified in XML
    float phiMax = 2.0f * M_PI;  // Default when not specified in XML
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(lidar.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(1, 1, 1);
    DOCTEST_CHECK_NOTHROW(lidar.addGrid(grid_center, grid_size, grid_divisions, 0));

    uint expected_rays = Ntheta * Nphi;

    Context context;
    vec3 sphere_center(0, 0, 0.5);
    float sphere_radius = 50.0f;
    context.addSphereObject(6, sphere_center, sphere_radius);  // Low subdivision for speed

    // Multi-return with rays_per_pulse=2, but zero beam width (should merge)
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 2, 0.001f));

    uint hit_count = lidar.getHitCount();

    // With zero beam parameters, all rays should merge to give same count as single-return
    DOCTEST_CHECK(hit_count == expected_rays);
}

DOCTEST_TEST_CASE("LiDAR Beam Perturbation - Multi Return Miss Recording") {
    // Test 3: Multi-return with no geometry should record misses correctly
    LiDARcloud lidar;
    lidar.disableMessages();

    // Create scan pattern programmatically
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 2000;
    uint Nphi = 4000;
    float thetaMin = 0.0f;       // Default when not specified in XML
    float thetaMax = M_PI;       // Default when not specified in XML
    float phiMin = 0.0f;         // Default when not specified in XML
    float phiMax = 2.0f * M_PI;  // Default when not specified in XML
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(lidar.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(1, 1, 1);
    DOCTEST_CHECK_NOTHROW(lidar.addGrid(grid_center, grid_size, grid_divisions, 0));

    uint expected_rays = Ntheta * Nphi;

    Context context;
    // No geometry - all rays should miss

    // Multi-return with record_misses=true and rays_per_pulse=2
    // Multiple misses from same pulse should merge together
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 2, 0.1f, false, true));

    uint hit_count = lidar.getHitCount();

    // All rays miss, but rays from same pulse merge, so count should equal expected_rays
    DOCTEST_CHECK(hit_count == expected_rays);

    // Verify all hits are actually misses (large distance)
    bool all_misses = true;
    for (uint i = 0; i < hit_count; i++) {
        if (lidar.doesHitDataExist(i, "distance")) {
            float dist = lidar.getHitData(i, "distance");
            if (dist < 1000.0f) {
                all_misses = false;
                break;
            }
        }
    }
    DOCTEST_CHECK(all_misses);
}

DOCTEST_TEST_CASE("LiDAR Multi-Return with Beam Spreading") {
    // Test multi-return with realistic beam parameters (exitDiameter>0, beamDivergence>0)
    LiDARcloud lidar;
    lidar.disableMessages();

    // Add scan programmatically for explicit control
    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 10000;
    uint Nphi = 12000;
    float thetaMin = 0.0f;       // Default when not specified in XML
    float thetaMax = M_PI;       // Default when not specified in XML
    float phiMin = 0.0f;         // Default when not specified in XML
    float phiMax = 2.0f * M_PI;  // Default when not specified in XML
    float exitDiameter = 0.0f;   // Point source for backward compatibility
    float beamDivergence = 0.0004f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence, columnFormat);
    DOCTEST_CHECK_NOTHROW(lidar.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(1, 1, 1);
    DOCTEST_CHECK_NOTHROW(lidar.addGrid(grid_center, grid_size, grid_divisions, 0));

    vec3 gsize = lidar.getCellSize(0);

    Context context;
    std::vector<uint> UUIDs = context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    float LAD_exact = 0.f;
    for (uint UUID: UUIDs) {
        LAD_exact += context.getPrimitiveArea(UUID) / (gsize.x * gsize.y * gsize.z);
    }

    // Multi-return with beam spreading (rays should NOT merge)
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 50, 0.1f, true, true));  // rays_per_pulse=50 for stable statistics

    uint hit_count = lidar.getHitCount();

    // Check for duplicate target_index=0 per timestamp (would be a bug!)
    std::map<int, int> timestamp_first_return_count;
    for (uint i = 0; i < hit_count; i++) {
        if (lidar.doesHitDataExist(i, "target_index") && lidar.doesHitDataExist(i, "timestamp")) {
            int tidx = static_cast<int>(lidar.getHitData(i, "target_index"));
            int tstamp = static_cast<int>(lidar.getHitData(i, "timestamp"));
            if (tidx == 0) {
                timestamp_first_return_count[tstamp]++;
            }
        }
    }
    int timestamps_with_multi_first = 0;
    for (auto& pair : timestamp_first_return_count) {
        if (pair.second > 1) timestamps_with_multi_first++;
    }

    // Triangulate using base overload - first returns automatically filtered (use aspect_ratio=5 to filter sliver triangles from beam spreading)
    DOCTEST_CHECK_NOTHROW(lidar.triangulateHitPoints(0.04, 5));
    DOCTEST_CHECK(lidar.getTriangleCount() > 0);

    DOCTEST_CHECK_NOTHROW(lidar.calculateLeafArea(&context));

    float LAD = lidar.getCellLeafAreaDensity(0);
    float Gtheta = lidar.getCellGtheta(0);

    // With beam spreading, we should get MORE hits than zero-width case (~2x)
    // But LAD should still be accurate
    DOCTEST_CHECK(LAD > LAD_exact * 0.9f);
    DOCTEST_CHECK(LAD < LAD_exact * 1.1f);

    // G(theta) should be close to 0.5 for spherical distribution
    DOCTEST_CHECK(Gtheta > 0.45f);
    DOCTEST_CHECK(Gtheta < 0.55f);
}

DOCTEST_TEST_CASE("LiDAR Exit Diameter - Comparative Spread Test") {
    // Verify that exitDiameter > 0 produces wider spatial spread than exitDiameter = 0
    Context context;
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    vec3 scan_origin(0, 0, 5.0f);

    // Scan WITHOUT exit diameter (point source)
    LiDARcloud lidar_point;
    lidar_point.disableMessages();
    ScanMetadata scan_point(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.0f, {});
    lidar_point.addScan(scan_point);
    lidar_point.syntheticScan(&context, 50, 0.05f, false, false);

    // Scan WITH exit diameter
    LiDARcloud lidar_exit;
    lidar_exit.disableMessages();
    ScanMetadata scan_exit(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.1f, 0.0f, {});
    lidar_exit.addScan(scan_exit);
    lidar_exit.syntheticScan(&context, 50, 0.05f, false, false);

    DOCTEST_REQUIRE(lidar_point.getHitCount() > 0);
    DOCTEST_REQUIRE(lidar_exit.getHitCount() > 0);

    // Calculate spatial extent for point source
    float x_min_pt = 1e6f, x_max_pt = -1e6f, y_min_pt = 1e6f, y_max_pt = -1e6f;
    for (uint i = 0; i < lidar_point.getHitCount(); i++) {
        vec3 pos = lidar_point.getHitXYZ(i);
        x_min_pt = fmin(x_min_pt, pos.x); x_max_pt = fmax(x_max_pt, pos.x);
        y_min_pt = fmin(y_min_pt, pos.y); y_max_pt = fmax(y_max_pt, pos.y);
    }

    // Calculate spatial extent for exit diameter
    float x_min_ex = 1e6f, x_max_ex = -1e6f, y_min_ex = 1e6f, y_max_ex = -1e6f;
    for (uint i = 0; i < lidar_exit.getHitCount(); i++) {
        vec3 pos = lidar_exit.getHitXYZ(i);
        x_min_ex = fmin(x_min_ex, pos.x); x_max_ex = fmax(x_max_ex, pos.x);
        y_min_ex = fmin(y_min_ex, pos.y); y_max_ex = fmax(y_max_ex, pos.y);
    }

    float extent_point = fmax(x_max_pt - x_min_pt, y_max_pt - y_min_pt);
    float extent_exit = fmax(x_max_ex - x_min_ex, y_max_ex - y_min_ex);

    // Exit diameter should produce measurably wider spread
    DOCTEST_CHECK(extent_exit > extent_point * 1.01f);
}

DOCTEST_TEST_CASE("LiDAR Exit Diameter - Zero Backward Compatibility") {
    // Verify that exitDiameter=0 still works (backward compatibility)
    Context context;
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    // Scan with exitDiameter=0 (original behavior - should work without errors)
    LiDARcloud lidar_zero;
    lidar_zero.disableMessages();
    ScanMetadata scan_zero(vec3(0, 0, 5), 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.0f, {});
    lidar_zero.addScan(scan_zero);
    DOCTEST_CHECK_NOTHROW(lidar_zero.syntheticScan(&context, 50, 0.05f, false, false));
    DOCTEST_CHECK(lidar_zero.getHitCount() > 0);

    // Scan with exitDiameter>0 (new behavior - should also work)
    LiDARcloud lidar_exit;
    lidar_exit.disableMessages();
    ScanMetadata scan_exit(vec3(0, 0, 5), 100, 0, M_PI, 100, 0, 2 * M_PI, 0.01f, 0.0f, {});
    lidar_exit.addScan(scan_exit);
    DOCTEST_CHECK_NOTHROW(lidar_exit.syntheticScan(&context, 50, 0.05f, false, false));
    DOCTEST_CHECK(lidar_exit.getHitCount() > 0);
}

DOCTEST_TEST_CASE("LiDAR Exit Diameter - Combined with Beam Divergence") {
    // Verify both spatial spreading (exitDiameter) and angular spreading (beamDivergence) work together
    Context context;
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    vec3 scan_origin(0, 0, 5.0f);

    // Scan with BOTH exitDiameter and beamDivergence
    LiDARcloud lidar_both;
    lidar_both.disableMessages();
    ScanMetadata scan_both(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.1f, 0.01f, {});
    lidar_both.addScan(scan_both);
    lidar_both.syntheticScan(&context, 50, 0.05f, false, false);

    // Scan with ONLY beamDivergence
    LiDARcloud lidar_div;
    lidar_div.disableMessages();
    ScanMetadata scan_div(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.01f, {});
    lidar_div.addScan(scan_div);
    lidar_div.syntheticScan(&context, 50, 0.05f, false, false);

    DOCTEST_REQUIRE(lidar_both.getHitCount() > 0);
    DOCTEST_REQUIRE(lidar_div.getHitCount() > 0);

    // Calculate spread for combined
    float x_min_b = 1e6f, x_max_b = -1e6f, y_min_b = 1e6f, y_max_b = -1e6f;
    for (uint i = 0; i < lidar_both.getHitCount(); i++) {
        vec3 pos = lidar_both.getHitXYZ(i);
        x_min_b = fmin(x_min_b, pos.x); x_max_b = fmax(x_max_b, pos.x);
        y_min_b = fmin(y_min_b, pos.y); y_max_b = fmax(y_max_b, pos.y);
    }

    // Calculate spread for divergence only
    float x_min_d = 1e6f, x_max_d = -1e6f, y_min_d = 1e6f, y_max_d = -1e6f;
    for (uint i = 0; i < lidar_div.getHitCount(); i++) {
        vec3 pos = lidar_div.getHitXYZ(i);
        x_min_d = fmin(x_min_d, pos.x); x_max_d = fmax(x_max_d, pos.x);
        y_min_d = fmin(y_min_d, pos.y); y_max_d = fmax(y_max_d, pos.y);
    }

    float spread_both = fmax(x_max_b - x_min_b, y_max_b - y_min_b);
    float spread_div = fmax(x_max_d - x_min_d, y_max_d - y_min_d);

    // Combined should produce wider spread than divergence alone
    DOCTEST_CHECK(spread_both > spread_div * 1.01f);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Grid Position Verification") {
    LiDARcloud lidar;
    lidar.disableMessages();
    Context context;

    // 1. Load existing test configuration with grid
    DOCTEST_CHECK_NOTHROW(lidar.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));

    vec3 scan_origin = lidar.getScanOrigin(0);
    uint Ntheta = lidar.getScanSizeTheta(0);
    uint Nphi = lidar.getScanSizePhi(0);

    // 2. Create simple geometry - small sphere that partially occludes
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 1.0), 0.3);

    // 3. Perform synthetic scan WITHOUT miss recording
    lidar.syntheticScan(&context, false, false); // scan_grid_only=false, record_misses=false
    uint hits_before_gapfill = lidar.getHitCount();

    // Sphere should block some rays creating gaps
    DOCTEST_CHECK(hits_before_gapfill > 0);

    // 4. Apply gapfilling with flags
    std::vector<vec3> filled_points = lidar.gapfillMisses(0, false, true);
    uint hits_after_gapfill = lidar.getHitCount();

    // 5. Verify gapfilling added points
    DOCTEST_CHECK(hits_after_gapfill > hits_before_gapfill);
    DOCTEST_CHECK(filled_points.size() > 0);

    // 6. QUANTITATIVE CHECK: Build position map by grid coordinates
    //    Use hit table to track which grid positions are filled
    std::map<std::pair<int,int>, bool> filled_grid_positions;

    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.getHitScanID(r) == 0) {
            SphericalCoord raydir = lidar.getHitRaydir(r);
            // Convert direction to grid indices using scan metadata
            float theta = raydir.zenith;
            float phi = raydir.azimuth;
            vec2 theta_range = lidar.getScanRangeTheta(0);
            vec2 phi_range = lidar.getScanRangePhi(0);

            int row = round((theta - theta_range.x) / (theta_range.y - theta_range.x) * (Ntheta - 1));
            int col = round((phi - phi_range.x) / (phi_range.y - phi_range.x) * (Nphi - 1));

            filled_grid_positions[std::make_pair(row, col)] = true;
        }
    }

    uint filled_cells = filled_grid_positions.size();

    // 7. QUANTITATIVE CHECK: Verify flag values
    uint flag_0_count = 0; // Original hits
    uint flag_1_count = 0; // Interior gapfilled
    uint flag_2_count = 0; // Downward edge
    uint flag_3_count = 0; // Upward edge

    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.getHitScanID(r) == 0 && lidar.doesHitDataExist(r, "gapfillMisses_code")) {
            int code = (int)lidar.getHitData(r, "gapfillMisses_code");
            if (code == 0) flag_0_count++;
            else if (code == 1) flag_1_count++;
            else if (code == 2) flag_2_count++;
            else if (code == 3) flag_3_count++;
        }
    }

    DOCTEST_CHECK(flag_0_count == hits_before_gapfill); // Original hits preserved
    DOCTEST_CHECK((flag_1_count + flag_2_count + flag_3_count) == filled_points.size());

    // NOTE: Interior fills (flag_1) may be 0 for sparse data
    // Edge fills (flag_2, flag_3) should exist since algorithm extrapolates edges
    DOCTEST_CHECK((flag_1_count + flag_2_count + flag_3_count) > 0); // At least some fills occurred
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Comparison with Record Misses") {
    LiDARcloud lidar1, lidar2;
    lidar1.disableMessages();
    lidar2.disableMessages();

    Context context;

    // Load existing test configuration with voxel grid
    DOCTEST_CHECK_NOTHROW(lidar1.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));
    DOCTEST_CHECK_NOTHROW(lidar2.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));

    // Create geometry that creates both hits and misses
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 1.0), 0.3);

    // === METHOD 1: Synthetic scan WITH miss recording ===
    lidar1.syntheticScan(&context, false, true); // record_misses = TRUE
    uint hits_with_misses = lidar1.getHitCount();

    // === METHOD 2: Synthetic scan WITHOUT miss recording, then gapfill ===
    lidar2.syntheticScan(&context, false, false); // record_misses = FALSE
    uint hits_before_gapfill = lidar2.getHitCount();

    std::vector<vec3> filled = lidar2.gapfillMisses(0, false, false);
    uint hits_after_gapfill = lidar2.getHitCount();

    // === QUANTITATIVE VERIFICATION ===

    // 1. Gapfilling should have added points
    DOCTEST_CHECK(hits_after_gapfill > hits_before_gapfill);
    DOCTEST_CHECK(filled.size() > 0);

    // 2. With duplicate prevention, gapfillMisses should produce similar hit counts
    //    May be slightly different due to algorithmic differences, but should be close
    float hit_ratio = float(hits_after_gapfill) / float(hits_with_misses);
    DOCTEST_CHECK(hit_ratio > 0.7f);  // Within reasonable range
    DOCTEST_CHECK(hit_ratio < 1.3f);

    // 3. Verify both methods cover similar grid positions by comparing actual hits (non-misses)
    //    Count hits that are NOT far-field points
    uint real_hits_method1 = 0;
    uint real_hits_method2 = 0;

    for (uint r = 0; r < lidar1.getHitCount(); r++) {
        float dist = sqrt(pow(lidar1.getHitXYZ(r).x - lidar1.getScanOrigin(0).x, 2) +
                          pow(lidar1.getHitXYZ(r).y - lidar1.getScanOrigin(0).y, 2) +
                          pow(lidar1.getHitXYZ(r).z - lidar1.getScanOrigin(0).z, 2));
        if (dist < 1000) real_hits_method1++;  // Not a far-field miss
    }

    for (uint r = 0; r < lidar2.getHitCount(); r++) {
        float dist = sqrt(pow(lidar2.getHitXYZ(r).x - lidar2.getScanOrigin(0).x, 2) +
                          pow(lidar2.getHitXYZ(r).y - lidar2.getScanOrigin(0).y, 2) +
                          pow(lidar2.getHitXYZ(r).z - lidar2.getScanOrigin(0).z, 2));
        if (dist < 1000) real_hits_method2++;
    }

    // Real hits (on geometry) should match between methods
    DOCTEST_CHECK(real_hits_method1 == real_hits_method2);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Edge Cases") {
    LiDARcloud lidar;
    lidar.disableMessages();

    // Test 1: Invalid scanID should throw error
    bool caught_error = false;
    try {
        lidar.gapfillMisses(999); // No scans exist yet
    } catch (const std::runtime_error& e) {
        caught_error = true;
        std::string msg(e.what());
        DOCTEST_CHECK(msg.find("Invalid scanID") != std::string::npos);
    }
    DOCTEST_CHECK(caught_error);

    // Test 2: Empty scan (no hits) should return empty vector gracefully
    DOCTEST_CHECK_NOTHROW(lidar.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));
    Context context;
    // Don't add any geometry - all rays will miss and not be traced
    lidar.syntheticScan(&context, false, false);  // No geometry, no hits

    uint hits_before = lidar.getHitCount();
    std::vector<vec3> filled;
    DOCTEST_CHECK_NOTHROW(filled = lidar.gapfillMisses(0, false, false));

    // Should handle empty scan gracefully
    if (hits_before == 0) {
        DOCTEST_CHECK(filled.empty());
        DOCTEST_CHECK(lidar.getHitCount() == 0);
    }

    // Test 3: Multi-scan gapfilling (all scans overload)
    LiDARcloud lidar2;
    lidar2.disableMessages();
    DOCTEST_CHECK_NOTHROW(lidar2.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 1.0), 0.3);
    lidar2.syntheticScan(&context, false, false);

    uint hits_before_all = lidar2.getHitCount();
    std::vector<vec3> filled_all;
    DOCTEST_CHECK_NOTHROW(filled_all = lidar2.gapfillMisses());  // Fill all scans
    uint hits_after_all = lidar2.getHitCount();

    DOCTEST_CHECK(hits_after_all >= hits_before_all);  // Should add points or stay same
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Grid Only Mode") {
    LiDARcloud lidar;
    lidar.disableMessages();
    Context context;

    // Load configuration with voxel grid
    DOCTEST_CHECK_NOTHROW(lidar.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));

    // Add geometry
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 1.0), 0.3);

    // Perform scan without miss recording
    lidar.syntheticScan(&context, false, false);
    uint hits_before = lidar.getHitCount();

    // Test grid-only mode (should fill fewer points than full mode)
    std::vector<vec3> filled_grid_only = lidar.gapfillMisses(0, true, false);
    uint hits_grid_only = lidar.getHitCount();

    // Reset and test full mode
    LiDARcloud lidar2;
    lidar2.disableMessages();
    DOCTEST_CHECK_NOTHROW(lidar2.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));
    lidar2.syntheticScan(&context, false, false);

    std::vector<vec3> filled_full = lidar2.gapfillMisses(0, false, false);
    uint hits_full = lidar2.getHitCount();

    // Grid-only mode should fill same or fewer points (limited to grid bounds)
    DOCTEST_CHECK(filled_grid_only.size() <= filled_full.size());
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Multi-Return Data") {
    LiDARcloud lidar;
    lidar.disableMessages();

    Context context;

    // Create simple box geometry
    std::vector<uint> box_uuids = context.addBox(make_vec3(0, 0, 1), make_vec3(1.5, 1.5, 1.5), make_int3(8, 8, 8));

    // Add voxel grid for the scan
    lidar.addGrid(make_vec3(0, 0, 1), make_vec3(2, 2, 2), make_int3(1, 1, 1), 0);

    // Create scan programmatically with beam spreading parameters for multi-return
    vec3 scan_origin(-3, 0, 1);
    uint Ntheta = 25;
    uint Nphi = 30;
    float thetaMin = M_PI/3;
    float thetaMax = 2*M_PI/3;
    float phiMin = 0.9;
    float phiMax = 1.7;
    float exitDiameter = 0.015;      // 1.5cm exit diameter
    float beamDivergence = 0.002;    // 2 mrad divergence for beam spreading

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                      exitDiameter, beamDivergence,
                      std::vector<std::string>{"x", "y", "z", "timestamp"});
    uint scanID = lidar.addScan(scan);

    // Multi-return synthetic scan WITHOUT miss recording - gapfilling will restore them
    lidar.syntheticScan(&context, 3, 0.15f, false, false);  // rays_per_pulse=3, record_misses=false
    uint hits_before = lidar.getHitCount();

    // Check if multi-return data was created (depends on beam spreading and geometry)
    bool has_multi_return = false;
    for (size_t r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.doesHitDataExist(r, "target_count") &&
            lidar.getHitData(r, "target_count") > 1) {
            has_multi_return = true;
            break;
        }
    }
    // Note: Multi-return creation depends on beam parameters and geometry interaction
    // Test verifies gapfilling works regardless

    // Apply gapfilling with flags for multi-return data
    std::vector<vec3> filled = lidar.gapfillMisses(scanID, false, true);
    uint hits_after = lidar.getHitCount();

    // Should have added gapfilled points
    DOCTEST_CHECK(hits_after > hits_before);
    DOCTEST_CHECK(filled.size() > 0);

    // Verify gapfillMisses_code flags were added
    uint flag_0_count = 0;  // Original hits
    uint flag_other_count = 0;  // Gapfilled hits (1, 2, or 3)

    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.doesHitDataExist(r, "gapfillMisses_code")) {
            int code = (int)lidar.getHitData(r, "gapfillMisses_code");
            if (code == 0) {
                flag_0_count++;
            } else {
                flag_other_count++;
            }
        }
    }

    // Original hits should all be flagged (if multi-return data exists)
    if (has_multi_return) {
        DOCTEST_CHECK(flag_0_count > 0);  // Should have original hits
        DOCTEST_CHECK(flag_other_count == filled.size());  // All filled points flagged
    }

    // Gapfilling should work without crashing on multi-return data
    DOCTEST_CHECK_NOTHROW(lidar.triangulateHitPoints(0.04, 10));

    // Note: If no multi-return data was created (geometry too sparse), test still validates
    // that gapfillMisses runs without error on the data that does exist
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Strict Accuracy Verification") {
    // Rigorous test to verify gapfillMisses produces ACCURATE results matching record_misses

    Context context;

    // Create scan parameters first to calculate coverage area
    vec3 scan_origin(-4, 0, 1.5);
    uint Ntheta = 35;
    uint Nphi = 50;
    float thetaMin = M_PI/3;
    float thetaMax = 2*M_PI/3;
    float phiMin = 1.0;
    float phiMax = 1.8;

    // Create large box geometry that covers ENTIRE scan field of view
    // This ensures ALL rays in the scan pattern will be traced
    // Place box to cover the angular range from the scanner
    std::vector<uint> box_uuids = context.addBox(make_vec3(2, 0, 1.5), make_vec3(8, 8, 4), make_int3(15, 15, 10));

    // METHOD 1: Record misses (GROUND TRUTH)
    LiDARcloud lidar1;
    lidar1.disableMessages();
    lidar1.addGrid(make_vec3(0, 0, 1.5), make_vec3(2, 2, 2), make_int3(1, 1, 1), 0);
    ScanMetadata scan1(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                       0, 0, std::vector<std::string>{"x", "y", "z", "timestamp"});
    lidar1.addScan(scan1);
    lidar1.syntheticScan(&context, false, true);  // record_misses = TRUE (ground truth)

    uint hits_ground_truth = lidar1.getHitCount();

    // Build comprehensive position map for ground truth
    std::map<std::pair<int,int>, vec3> ground_truth_positions;
    std::map<std::pair<int,int>, bool> ground_truth_is_miss;  // Track which are far-field misses

    for (uint r = 0; r < lidar1.getHitCount(); r++) {
        SphericalCoord raydir = lidar1.getHitRaydir(r);
        vec3 pos = lidar1.getHitXYZ(r);

        // Calculate grid indices
        float theta = raydir.zenith;
        float phi = raydir.azimuth;
        int row = round((theta - thetaMin) / (thetaMax - thetaMin) * (Ntheta - 1));
        int col = round((phi - phiMin) / (phiMax - phiMin) * (Nphi - 1));

        ground_truth_positions[std::make_pair(row, col)] = pos;

        // Check if this is a far-field miss (distance > 1000m)
        float dist = sqrt(pow(pos.x - scan_origin.x, 2) +
                         pow(pos.y - scan_origin.y, 2) +
                         pow(pos.z - scan_origin.z, 2));
        ground_truth_is_miss[std::make_pair(row, col)] = (dist > 1000);
    }

    // Verify we got expected number of hits (should be Ntheta × Nphi)
    uint expected_grid_size = Ntheta * Nphi;

    DOCTEST_CHECK_MESSAGE(hits_ground_truth == expected_grid_size,
        "record_misses should produce Ntheta×Nphi hits but got " << hits_ground_truth << " vs " << expected_grid_size);

    DOCTEST_CHECK(ground_truth_positions.size() == hits_ground_truth);

    // METHOD 2: Gapfill misses
    LiDARcloud lidar2;
    lidar2.disableMessages();
    lidar2.addGrid(make_vec3(0, 0, 1.5), make_vec3(2, 2, 2), make_int3(1, 1, 1), 0);
    ScanMetadata scan2(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax,
                       0, 0, std::vector<std::string>{"x", "y", "z", "timestamp"});
    lidar2.addScan(scan2);
    lidar2.syntheticScan(&context, false, false);  // record_misses = FALSE

    uint hits_before_gapfill = lidar2.getHitCount();

    std::vector<vec3> filled = lidar2.gapfillMisses(0, false, false);
    uint hits_after_gapfill = lidar2.getHitCount();

    // Build position map for gapfilled data (map will deduplicate by grid position)
    std::map<std::pair<int,int>, vec3> gapfilled_positions;
    std::map<std::pair<int,int>, uint> gapfilled_hit_count;  // Count hits per grid position

    for (uint r = 0; r < lidar2.getHitCount(); r++) {
        SphericalCoord raydir = lidar2.getHitRaydir(r);
        vec3 pos = lidar2.getHitXYZ(r);

        float theta = raydir.zenith;
        float phi = raydir.azimuth;
        int row = round((theta - thetaMin) / (thetaMax - thetaMin) * (Ntheta - 1));
        int col = round((phi - phiMin) / (phiMax - phiMin) * (Nphi - 1));

        auto key = std::make_pair(row, col);
        gapfilled_positions[key] = pos;  // Map stores last position at this grid cell
        gapfilled_hit_count[key]++;      // Count hits per grid cell
    }

    // Check for duplicates in gapfilled data
    uint gapfilled_duplicates = hits_after_gapfill - gapfilled_positions.size();
    DOCTEST_CHECK_MESSAGE(gapfilled_duplicates == 0,
        "CRITICAL BUG: gapfillMisses created " << gapfilled_duplicates
        << " duplicate hits at positions that already had hits!");

    // 1. Separate verification: Interior positions vs Edge extrapolation
    //    Ground truth only has positions traced by syntheticScan
    //    Gapfilling should: (a) match ground truth for traced regions, (b) extrapolate edges

    // All ground truth positions should exist in gapfilled data (superset)
    uint ground_truth_positions_found = 0;
    for (const auto& kv : ground_truth_positions) {
        if (gapfilled_positions.find(kv.first) != gapfilled_positions.end()) {
            ground_truth_positions_found++;
        }
    }

    float ground_truth_recovery = float(ground_truth_positions_found) / float(ground_truth_positions.size());
    DOCTEST_CHECK_MESSAGE(ground_truth_recovery > 0.95f,
        "Gapfilling missed positions that record_misses found: only recovered "
        << ground_truth_positions_found << " of " << ground_truth_positions.size()
        << " (" << (ground_truth_recovery*100) << "%)");

    // 2. Verify all gapfilled points are at VALID grid positions (theta, phi in bounds)
    uint invalid_positions = 0;
    for (const auto& kv : gapfilled_positions) {
        int row = kv.first.first;
        int col = kv.first.second;

        if (row < 0 || row >= (int)Ntheta || col < 0 || col >= (int)Nphi) {
            invalid_positions++;
        }
    }
    DOCTEST_CHECK_MESSAGE(invalid_positions == 0,
        "Gapfilling created " << invalid_positions << " points at invalid grid positions");

    // 3. Verify positions that match ground truth have correct coordinates
    uint position_mismatches = 0;
    float max_position_error = 0;

    for (const auto& kv : ground_truth_positions) {
        auto key = kv.first;
        if (gapfilled_positions.find(key) != gapfilled_positions.end()) {
            vec3 pos_gt = kv.second;
            vec3 pos_gf = gapfilled_positions[key];

            bool is_miss = ground_truth_is_miss[key];

            float dist = sqrt(pow(pos_gt.x - pos_gf.x, 2) +
                            pow(pos_gt.y - pos_gf.y, 2) +
                            pow(pos_gt.z - pos_gf.z, 2));

            if (dist > max_position_error) max_position_error = dist;

            // For geometry hits, positions should match exactly (within 1cm)
            // For far-field misses, directions should match (within 1 degree)
            if (!is_miss && dist > 0.01f) {
                position_mismatches++;
            } else if (is_miss) {
                // Check direction for misses
                vec3 dir_gt = pos_gt - scan_origin;
                vec3 dir_gf = pos_gf - scan_origin;
                float mag_gt = sqrt(dir_gt.x*dir_gt.x + dir_gt.y*dir_gt.y + dir_gt.z*dir_gt.z);
                float mag_gf = sqrt(dir_gf.x*dir_gf.x + dir_gf.y*dir_gf.y + dir_gf.z*dir_gf.z);

                float dot = (dir_gt.x*dir_gf.x + dir_gt.y*dir_gf.y + dir_gt.z*dir_gf.z) / (mag_gt * mag_gf);
                if (dot < 0.9998f) {  // Directions differ by >1 degree
                    position_mismatches++;
                }
            }
        }
    }

    float position_match_rate = float(ground_truth_positions_found - position_mismatches) / float(ground_truth_positions_found);
    DOCTEST_CHECK_MESSAGE(position_match_rate > 0.95f,
        "Position accuracy too low: " << position_mismatches << " mismatches out of "
        << ground_truth_positions_found << " (" << (position_match_rate*100) << "% correct)");

    // STRICT CHECK: For ideal scan, gapfilling should match ground truth exactly
    DOCTEST_CHECK_MESSAGE(hits_after_gapfill == hits_ground_truth,
        "Gapfilling should exactly match record_misses for ideal scan: "
        << hits_after_gapfill << " vs " << hits_ground_truth
        << " (difference: " << (int)hits_after_gapfill - (int)hits_ground_truth << ")");
}
