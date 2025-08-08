#include "AerialLiDAR.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

int AerialLiDARcloud::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

DOCTEST_TEST_CASE("AerialLiDARcloud Dense Vegetation Test") {
    float err_tol = 0.1;

    //------- dense vegatation test to check "mean dr" method for calculating LAD -------//

    Context context_1;

    vec3 boxsize(10, 10, 10);

    int3 Nleaves(100, 100, 100);

    float L = 0.05;

    bool flag;
    float LAD_exact;

    for (int k = 0; k < Nleaves.z; k++) {
        for (int j = 0; j < Nleaves.y; j++) {
            for (int i = 0; i < Nleaves.x; i++) {

                vec3 x(context_1.randu() * boxsize.x, context_1.randu() * boxsize.y, context_1.randu() * boxsize.z);

                float theta = acos(1.f - context_1.randu());
                float phi = context_1.randu() * 2.f * M_PI;

                context_1.addPatch(x, make_vec2(L, L), make_SphericalCoord(theta, phi));
            }
        }
    }

    context_1.addPatch(make_vec3(0.5 * boxsize.x, 0.5 * boxsize.y, -0.001), make_vec2(boxsize.x, boxsize.y));

    LAD_exact = float(Nleaves.x * Nleaves.y * Nleaves.z) * L * L / (boxsize.x * boxsize.y * boxsize.z);

    flag = true;
    AerialLiDARcloud lidar_1;
    lidar_1.disableMessages();

    lidar_1.syntheticScan(&context_1, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    helios::vec3 center = lidar_1.getGridCenter();
    DOCTEST_CHECK(center.x == 5.0);
    DOCTEST_CHECK(center.y == 5.0);
    DOCTEST_CHECK(center.z == 5.0);

    helios::int3 resolution = lidar_1.getGridResolution();
    DOCTEST_CHECK(resolution.x == 2);
    DOCTEST_CHECK(resolution.y == 2);
    DOCTEST_CHECK(resolution.z == 2);

    int v = 6;
    helios::int3 ijk = lidar_1.gridindex2ijk(v);
    DOCTEST_CHECK(v == lidar_1.gridijk2index(ijk));

    lidar_1.calculateLeafAreaGPU(0.5, 10);

    for (int v = 4; v < 8; v++) {
        float LAD = lidar_1.getCellLeafAreaDensity(lidar_1.gridindex2ijk(v));
        DOCTEST_CHECK(fabs(LAD - LAD_exact) / LAD_exact <= err_tol);
    }
}

DOCTEST_TEST_CASE("AerialLiDARcloud Sparse Vegetation Test") {
    float err_tol = 0.1;

    //------- sparse vegatation test to check "mean P" method for calculating LAD -------//

    Context context_2;

    vec3 boxsize(10, 10, 10);
    int3 Nleaves = make_int3(25, 25, 25);
    float L = 0.05;

    for (int k = 0; k < Nleaves.z; k++) {
        for (int j = 0; j < Nleaves.y; j++) {
            for (int i = 0; i < Nleaves.x; i++) {

                vec3 x(context_2.randu() * boxsize.x, context_2.randu() * boxsize.y, context_2.randu() * boxsize.z);

                float theta = acos(1.f - context_2.randu());
                float phi = context_2.randu() * 2.f * M_PI;

                context_2.addPatch(x, make_vec2(L, L), make_SphericalCoord(theta, phi));
            }
        }
    }

    context_2.addPatch(make_vec3(0.5 * boxsize.x, 0.5 * boxsize.y, -0.001), make_vec2(boxsize.x, boxsize.y));

    float LAD_exact = float(Nleaves.x * Nleaves.y * Nleaves.z) * L * L / (boxsize.x * boxsize.y * boxsize.z);

    AerialLiDARcloud lidar_2;
    lidar_2.disableMessages();

    lidar_2.syntheticScan(&context_2, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    lidar_2.calculateLeafAreaGPU(0.5, 10);

    for (int v = 0; v < 8; v++) {
        float LAD = lidar_2.getCellLeafAreaDensity(lidar_2.gridindex2ijk(v));
        DOCTEST_CHECK(fabs(LAD - LAD_exact) / LAD_exact <= 1.5 * err_tol);
    }
}

DOCTEST_TEST_CASE("AerialLiDARcloud Ground and Canopy Height Test") {
    float err_tol = 0.1;

    //------- sparse vegatation ground and canopy height estimation -------//

    Context context_3;

    vec3 boxsize(10, 10, 10);
    int3 Nleaves = make_int3(25, 25, 35);
    float L = 0.05;

    for (int k = 0; k < Nleaves.z; k++) {
        for (int j = 0; j < Nleaves.y; j++) {
            for (int i = 0; i < Nleaves.x; i++) {

                vec3 x(context_3.randu() * boxsize.x, context_3.randu() * boxsize.y, context_3.randu() * boxsize.z);

                float theta = acos(1.f - context_3.randu());
                float phi = context_3.randu() * 2.f * M_PI;

                context_3.addPatch(x, make_vec2(L, L), make_SphericalCoord(theta, phi));
            }
        }
    }

    float zground = 0.2;

    context_3.addPatch(make_vec3(0.5 * boxsize.x, 0.5 * boxsize.y, zground), make_vec2(boxsize.x, boxsize.y));

    AerialLiDARcloud lidar_3;
    lidar_3.disableMessages();

    lidar_3.syntheticScan(&context_3, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    for (int r = 0; r < lidar_3.getHitCount(); r++) {
        lidar_3.setHitData(r, "target_index", 1);
        lidar_3.setHitData(r, "target_count", 1);
    }

    lidar_3.generateHeightModel(100, 0.5, 0.1, 0.5, 0.1);

    for (int v = 0; v < 8; v++) {
        int3 index = lidar_3.gridindex2ijk(v);
        float zg = lidar_3.getCellGroundHeight(make_int2(index.x, index.y));
        DOCTEST_CHECK(fabs(zg - zground) / fabs(zground) <= 1.5 * err_tol);
    }

    for (int r = 0; r < lidar_3.getHitCount(); r++) {
        vec3 xyz = lidar_3.getHitXYZ(r);
        if (fabs(xyz.z - zground) > 9) {
            lidar_3.setHitData(r, "target_index", 1);
        } else {
            lidar_3.setHitData(r, "target_index", 2);
        }
        lidar_3.setHitData(r, "target_count", 2);
    }

    lidar_3.generateHeightModel(400, 0.5, 0.1, 1.0, 0.2);

    for (int v = 0; v < 8; v++) {
        int3 index = lidar_3.gridindex2ijk(v);
        float zc = lidar_3.getCellVegetationHeight(make_int2(index.x, index.y));
        float zm = lidar_3.getCellMaximumHitHeight(make_int2(index.x, index.y));

        DOCTEST_CHECK(fabs(zc - (boxsize.z - 0.5)) / fabs(boxsize.z - 0.5) <= 1.5 * err_tol);
        DOCTEST_CHECK(fabs(zm - boxsize.z) / fabs(boxsize.z) <= err_tol);
    }
}

DOCTEST_TEST_CASE("AerialLiDARcloud enableMessages") {
    AerialLiDARcloud lidar;
    lidar.disableMessages(); // Disable first
    DOCTEST_CHECK_NOTHROW(lidar.enableMessages()); // Then enable again
}

DOCTEST_TEST_CASE("AerialLiDARcloud addHitPoint with scanID, hit_xyz, ray_origin") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata scan_meta(make_vec3(0, 0, 0), make_vec2(1, 1), 30.0f, 10.0f, 0.1f, 0.05f);
    lidar.addScan(scan_meta);

    vec3 hit = make_vec3(1, 2, 3);
    vec3 origin = make_vec3(0, 0, 0);
    DOCTEST_CHECK_NOTHROW(lidar.addHitPoint(0, hit, origin));
    DOCTEST_CHECK(lidar.getHitCount() == 1);
}

DOCTEST_TEST_CASE("AerialLiDARcloud addHitPoint with scanID, hit_xyz, direction") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata scan_meta(make_vec3(0, 0, 0), make_vec2(1, 1), 30.0f, 10.0f, 0.1f, 0.05f);
    lidar.addScan(scan_meta);

    vec3 hit = make_vec3(1, 1, 1);
    SphericalCoord dir = make_SphericalCoord(1.0f, 0.5f);
    DOCTEST_CHECK_NOTHROW(lidar.addHitPoint(0, hit, dir));
    DOCTEST_CHECK(lidar.getHitCount() == 1);
}

DOCTEST_TEST_CASE("AerialLiDARcloud addHitPoint with scanID, hit_xyz, direction, data") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata scan_meta(make_vec3(0, 0, 0), make_vec2(1, 1), 30.0f, 10.0f, 0.1f, 0.05f);
    lidar.addScan(scan_meta);

    vec3 hit = make_vec3(1, 1, 1);
    SphericalCoord dir = make_SphericalCoord(1.0f, 0.5f);
    std::map<std::string, float> data = {{"reflectance", 123.45f}};
    DOCTEST_CHECK_NOTHROW(lidar.addHitPoint(0, hit, dir, data));
    DOCTEST_CHECK(lidar.getHitCount() == 1);
}

DOCTEST_TEST_CASE("AerialLiDARcloud addHitPoint with scanID, hit_xyz, direction, color") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata scan_meta(make_vec3(0, 0, 0), make_vec2(1, 1), 30.0f, 10.0f, 0.1f, 0.05f);
    lidar.addScan(scan_meta);

    vec3 hit = make_vec3(1, 1, 1);
    SphericalCoord dir = make_SphericalCoord(1.0f, 0.5f);
    RGBcolor color = make_RGBcolor(0.2f, 0.8f, 0.6f);
    DOCTEST_CHECK_NOTHROW(lidar.addHitPoint(0, hit, dir, color));
    DOCTEST_CHECK(lidar.getHitCount() == 1);
}

DOCTEST_TEST_CASE("AerialLiDARcloud deleteHitPoint") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata scan_meta(make_vec3(0, 0, 0), make_vec2(1, 1), 10.0f, 1.0f, 0.1f, 0.01f);
    lidar.addScan(scan_meta);

    vec3 hit = make_vec3(1, 1, 1);
    vec3 origin = make_vec3(0, 0, 0);
    lidar.addHitPoint(0, hit, origin);

    DOCTEST_CHECK_NOTHROW(lidar.deleteHitPoint(0));
    DOCTEST_CHECK(lidar.getHitCount() == 0);
}

DOCTEST_TEST_CASE("AerialLiDARcloud getHitColor") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata scan_meta(make_vec3(0, 0, 0), make_vec2(1, 1), 10.0f, 1.0f, 0.1f, 0.01f);
    lidar.addScan(scan_meta);

    RGBcolor expected_color = make_RGBcolor(0.2f, 0.6f, 0.9f);
    SphericalCoord dir = make_SphericalCoord(1.0f, 1.0f);
    vec3 hit = make_vec3(1, 1, 1);
    lidar.addHitPoint(0, hit, dir, expected_color);

    RGBcolor actual_color = lidar.getHitColor(0);
    DOCTEST_CHECK(actual_color.r == expected_color.r);
    DOCTEST_CHECK(actual_color.g == expected_color.g);
    DOCTEST_CHECK(actual_color.b == expected_color.b);
}

DOCTEST_TEST_CASE("AerialLiDARcloud coordinateShift") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata scan_meta(make_vec3(0, 0, 0), make_vec2(1, 1), 10.0f, 1.0f, 0.1f, 0.01f);
    lidar.addScan(scan_meta);

    vec3 hit = make_vec3(1, 1, 1);
    vec3 origin = make_vec3(0, 0, 0);
    lidar.addHitPoint(0, hit, origin);

    vec3 shift = make_vec3(5, 5, 5);
    lidar.coordinateShift(shift);
    vec3 new_pos = lidar.getHitXYZ(0);

    DOCTEST_CHECK(fabs(new_pos.x - 6.0f) < 1e-4);
    DOCTEST_CHECK(fabs(new_pos.y - 6.0f) < 1e-4);
    DOCTEST_CHECK(fabs(new_pos.z - 6.0f) < 1e-4);
}

DOCTEST_TEST_CASE("AerialLiDARcloud getHitBoundingBox") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    AerialScanMetadata meta(make_vec3(0, 0, 0), make_vec2(1, 1), 30.0f, 1.0f, 0.1f, 0.01f);
    lidar.addScan(meta);
    lidar.addHitPoint(0, make_vec3(1, 2, 3), make_vec3(0, 0, 0));

    vec3 boxmin, boxmax;
    DOCTEST_CHECK_NOTHROW(lidar.getHitBoundingBox(boxmin, boxmax));
    DOCTEST_CHECK(boxmin.x <= boxmax.x);
    DOCTEST_CHECK(boxmin.y <= boxmax.y);
    DOCTEST_CHECK(boxmin.z <= boxmax.z);
}

DOCTEST_TEST_CASE("AerialLiDARcloud getGridExtent") {
    AerialLiDARcloud lidar;
    vec3 extent = lidar.getGridExtent();

    DOCTEST_CHECK(extent.x == 0);
    DOCTEST_CHECK(extent.y == 0);
    DOCTEST_CHECK(extent.z == 0);
}

DOCTEST_TEST_CASE("AerialLiDARcloud getCellLeafArea") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    int3 grid_idx = make_int3(0, 0, 0);
    lidar.setCellLeafArea(2.5f, grid_idx);

    float value = lidar.getCellLeafArea(grid_idx);
    DOCTEST_CHECK(fabs(value - 2.5f) < 1e-6);
}

DOCTEST_TEST_CASE("AerialLiDARcloud getCellTransmissionProbability") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    int3 grid_idx = make_int3(0, 0, 0);
    lidar.setCellTransmissionProbability(10, 4, grid_idx);

    int denom = -1, trans = -1;
    lidar.getCellTransmissionProbability(grid_idx, denom, trans);
    DOCTEST_CHECK(denom == 10);
    DOCTEST_CHECK(trans == 4);
}

DOCTEST_TEST_CASE("AerialLiDARcloud calculateCoverFraction") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)); // Just a flat patch for now
    lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    lidar.calculateHitGridCell(); // Required for grid info

    // Assign 'ground_flag' to all hits
    for (uint i = 0; i < lidar.getHitCount(); ++i) {
        lidar.setHitData(i, "ground_flag", 1.0f); // Treat all as ground hits
    }

    DOCTEST_CHECK_NOTHROW(lidar.calculateCoverFraction()); // Now won't error on missing ground_flag
}

DOCTEST_TEST_CASE("AerialLiDARcloud setCellCoverFraction") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    lidar.calculateHitGridCell(); // Good to ensure grid is ready
    DOCTEST_CHECK_NOTHROW(lidar.setCellCoverFraction(0.75f, make_int2(0, 0)));
}

DOCTEST_TEST_CASE("AerialLiDARcloud getCellCoverFraction") {
    AerialLiDARcloud lidar;
    lidar.disableMessages();

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

    lidar.calculateHitGridCell(); // Required for valid cell lookup
    lidar.setCellCoverFraction(0.85f, make_int2(0, 0));
    float val = lidar.getCellCoverFraction(make_int2(0, 0));
    DOCTEST_CHECK(fabs(val - 0.85f) < 1e-6);
}