#include "LiDAR.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

using namespace std;
using namespace helios;

float err_tol = 1e-3;

int LiDARcloud::selfTest() {
    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
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