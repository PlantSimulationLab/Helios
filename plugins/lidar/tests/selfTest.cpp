#include "LiDAR.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace std;
using namespace helios;

float err_tol = 1e-3;

int LiDARcloud::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

DOCTEST_TEST_CASE("LiDAR Single Voxel Sphere Test") {
    LiDARcloud pointcloud;
    pointcloud.disableMessages();

    DOCTEST_CHECK_NOTHROW(pointcloud.loadXML("plugins/lidar/xml/sphere.xml"));
    DOCTEST_CHECK_NOTHROW(pointcloud.triangulateHitPoints(0.5, 5));

    Context context_1;
    DOCTEST_CHECK_NOTHROW(pointcloud.addTrianglesToContext(&context_1));

    DOCTEST_CHECK(context_1.getPrimitiveCount() == 386);
}

DOCTEST_TEST_CASE("LiDAR Single Voxel Isotropic Patches Test") {
    LiDARcloud synthetic_1;
    synthetic_1.disableMessages();

    DOCTEST_CHECK_NOTHROW(synthetic_1.loadXML("plugins/lidar/xml/synthetic_test.xml"));

    vec3 gsize = synthetic_1.getCellSize(0);

    Context context_2;
    std::vector<uint> UUIDs_1 = context_2.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    float LAD_exact = 0.f;
    for (uint UUID: UUIDs_1) {
        LAD_exact += context_2.getPrimitiveArea(UUID) / (gsize.x * gsize.y * gsize.z);
    }

    DOCTEST_CHECK_NOTHROW(synthetic_1.syntheticScan(&context_2));
    DOCTEST_CHECK_NOTHROW(synthetic_1.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(synthetic_1.calculateLeafAreaGPU());

    float LAD = synthetic_1.getCellLeafAreaDensity(0);

    DOCTEST_CHECK(LAD == LAD); // Check for NaN
    DOCTEST_CHECK(fabs(LAD - LAD_exact) / LAD_exact == doctest::Approx(0.0f).epsilon(0.02f));
}

DOCTEST_TEST_CASE("LiDAR Eight Voxel Isotropic Patches Test") {
    LiDARcloud synthetic_2;
    synthetic_2.disableMessages();

    DOCTEST_CHECK_NOTHROW(synthetic_2.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));

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
    DOCTEST_CHECK_NOTHROW(synthetic_2.calculateLeafAreaGPU());

    float RMSE = 0.f;
    for (int i = 0; i < synthetic_2.getGridCellCount(); i++) {
        float LAD = synthetic_2.getCellLeafAreaDensity(i);
        RMSE += powf(LAD - LAD_ex.at(i), 2) / float(synthetic_2.getGridCellCount());
    }
    RMSE = sqrtf(RMSE);

    DOCTEST_CHECK(RMSE == doctest::Approx(0.0f).epsilon(0.05f));
}

DOCTEST_TEST_CASE("LiDAR Single Voxel Anisotropic Patches Test") {
    LiDARcloud synthetic_3;
    synthetic_3.disableMessages();

    DOCTEST_CHECK_NOTHROW(synthetic_3.loadXML("plugins/lidar/xml/synthetic_test.xml"));

    vec3 gsize = synthetic_3.getCellSize(0);

    Context context_2;
    std::vector<uint> UUIDs_1 = context_2.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    float LAD_exact = 0.f;
    for (uint UUID: UUIDs_1) {
        LAD_exact += context_2.getPrimitiveArea(UUID) / (gsize.x * gsize.y * gsize.z);
    }

    DOCTEST_CHECK_NOTHROW(synthetic_3.syntheticScan(&context_2));
    DOCTEST_CHECK_NOTHROW(synthetic_3.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(synthetic_3.calculateLeafAreaGPU());

    float LAD = synthetic_3.getCellLeafAreaDensity(0);

    DOCTEST_CHECK(LAD == LAD); // Check for NaN
    DOCTEST_CHECK(fabs(LAD - LAD_exact) / LAD_exact == doctest::Approx(0.0f).epsilon(0.03f));
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
    DOCTEST_CHECK_NOTHROW(synthetic_4.calculateLeafAreaGPU());

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

    DOCTEST_CHECK(RMSE_LAD <= 0.25f);
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
