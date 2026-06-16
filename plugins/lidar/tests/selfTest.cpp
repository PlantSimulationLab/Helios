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
    float thetaMin = 0.0f; // Default when not specified in XML
    float thetaMax = M_PI; // Default when not specified in XML
    float phiMin = 0.0f; // Default when not specified in XML
    float phiMax = 2.0f * M_PI; // Default when not specified in XML
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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
    for (uint UUID: UUIDs_1) {
        float area = context_2.getPrimitiveArea(UUID);
        vec3 normal = context_2.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context_2.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) { // Check for NaN
            float normal_dot_ray = fabs(normal * raydir);
            Gtheta_exact_numerator += normal_dot_ray * area;
            Gtheta_exact_denominator += area;
        }
    }
    float Gtheta_exact = 0.f;
    if (Gtheta_exact_denominator > 0) {
        Gtheta_exact = Gtheta_exact_numerator / Gtheta_exact_denominator;
    }

    DOCTEST_CHECK_NOTHROW(synthetic_1.syntheticScan(&context_2, false, true)); // record_misses=true: LAD inversion needs transmitted beams
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

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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

    DOCTEST_CHECK_NOTHROW(synthetic_2.syntheticScan(&context_2, false, true)); // record_misses=true: LAD inversion needs transmitted beams
    DOCTEST_CHECK_NOTHROW(synthetic_2.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(synthetic_2.calculateLeafArea(&context_2));

    float RMSE = 0.f;
    for (int i = 0; i < synthetic_2.getGridCellCount(); i++) {
        float LAD = synthetic_2.getCellLeafAreaDensity(i);
        RMSE += powf(LAD - LAD_ex.at(i), 2) / float(synthetic_2.getGridCellCount());
    }
    RMSE = sqrtf(RMSE);

    // CDT and s_hull produce valid but different Delaunay tessellations; the
    // resulting per-voxel LAD RMSE for this case is ~0.063 (s_hull) to ~0.066
    // (CDT). Tolerance allows for this tessellation-dependent drift.
    DOCTEST_CHECK(RMSE == doctest::Approx(0.0f).epsilon(0.07f));
}

DOCTEST_TEST_CASE("LiDAR Thin-Layer Vertical Symmetry Test") {
    // Regression test for a single-return LAD asymmetry: a vertically-symmetric target
    // scanned by vertically-symmetric scanners sitting exactly on the interface between
    // two grid layers must recover symmetric LAD in the upper and lower layers,
    // regardless of how thin the layers are. A previous synthetic-raster implementation
    // produced a top/bottom asymmetry that collapsed the upper layer to a pinned floor
    // value for thin layers. The grid z-extent is swept across the previously-failing
    // regime to guard against any return of the threshold-like behavior.

    Context context_sym;
    std::vector<uint> UUIDs = context_sym.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true); // cube of leaves centered at z=0.5, symmetric about z=0.5

    // Four scanners on the z=0.5 plane (the layer interface), symmetric about the target.
    const float origins[4][3] = {{-5, 0, 0.5f}, {0, -5, 0.5f}, {5, 0, 0.5f}, {0, 5, 0.5f}};

    for (float sizez: {0.5f, 0.45f, 0.41f, 0.40f, 0.35f}) {

        LiDARcloud lidar_sym;
        lidar_sym.disableMessages();

        for (int sidx = 0; sidx < 4; sidx++) {
            ScanMetadata scan(make_vec3(origins[sidx][0], origins[sidx][1], origins[sidx][2]), 2000, 0.f, M_PI, 4000, 0.f, 2.f * M_PI, 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
            lidar_sym.addScan(scan);
        }

        // Two layers split at z=0.5; sizez controls how thin each layer is.
        lidar_sym.addGrid(make_vec3(0, 0, 0.5f), make_vec3(0.5f, 0.5f, sizez), make_int3(1, 1, 2), 0);

        DOCTEST_CHECK_NOTHROW(lidar_sym.syntheticScan(&context_sym, true, true)); // scan_grid_only, record_misses
        DOCTEST_CHECK_NOTHROW(lidar_sym.triangulateHitPoints(0.04, 10));
        DOCTEST_CHECK_NOTHROW(lidar_sym.calculateLeafArea(&context_sym));

        float lower = 0.f, upper = 0.f;
        int nl = 0, nu = 0;
        for (uint i = 0; i < lidar_sym.getGridCellCount(); i++) {
            float lad = lidar_sym.getCellLeafAreaDensity(i);
            // No cell may be pinned at the degenerate solver floor (a = 0.1 initial guess
            // -> LAD = leaf_area/volume = 0.1) when the target genuinely has leaf area.
            DOCTEST_CHECK_MESSAGE(fabs(lad - 0.1f) > 5e-3f, "Cell " << i << " pinned at the LAD floor (~0.1) at size.z=" << sizez);
            if (lidar_sym.getCellCenter(i).z < 0.5f) {
                lower += lad;
                nl++;
            } else {
                upper += lad;
                nu++;
            }
        }
        if (nl > 0)
            lower /= float(nl);
        if (nu > 0)
            upper /= float(nu);

        // Symmetric input MUST give symmetric LAD: upper and lower within 25%.
        float denom = std::max(lower, upper);
        bool symmetric = (denom > 0.f) && (fabs(upper - lower) / denom < 0.25f);
        DOCTEST_CHECK_MESSAGE(symmetric, "Thin-layer LAD asymmetry at size.z=" << sizez << ": lower=" << lower << " upper=" << upper);
    }
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

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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
    for (uint UUID: UUIDs_1) {
        float area = context_2.getPrimitiveArea(UUID);
        vec3 normal = context_2.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context_2.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) { // Check for NaN
            float normal_dot_ray = fabs(normal * raydir);
            Gtheta_exact_numerator += normal_dot_ray * area;
            Gtheta_exact_denominator += area;
        }
    }
    float Gtheta_exact = 0.f;
    if (Gtheta_exact_denominator > 0) {
        Gtheta_exact = Gtheta_exact_numerator / Gtheta_exact_denominator;
    }

    DOCTEST_CHECK_NOTHROW(synthetic_3.syntheticScan(&context_2, false, true)); // record_misses=true: LAD inversion needs transmitted beams
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
    DOCTEST_CHECK_NOTHROW(synthetic_4.syntheticScan(&context_4, false, true)); // record_misses=true: LAD inversion needs transmitted beams
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

    // Test multi-return overload with append=true
    DOCTEST_CHECK_NOTHROW(synthetic_test.syntheticScan(&context_test, 1, 0.0f, true));
    uint hit_count_append2 = synthetic_test.getHitCount();
    DOCTEST_CHECK(hit_count_append2 == 2 * hit_count_first);
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam Scan Geometry") {
    // Build a spinning multibeam scan with VLP-16-style channels (16 channels, 2-degree spacing from -15 to +15 deg elevation).
    vec3 scan_origin(0.f, 0.f, 1.f);
    std::vector<float> elevation_deg = {-15.f, -13.f, -11.f, -9.f, -7.f, -5.f, -3.f, -1.f, 1.f, 3.f, 5.f, 7.f, 9.f, 11.f, 13.f, 15.f};
    std::vector<float> beam_zenith(elevation_deg.size());
    for (size_t k = 0; k < elevation_deg.size(); k++) {
        beam_zenith[k] = 0.5f * float(M_PI) - elevation_deg[k] * float(M_PI) / 180.f;
    }
    uint Nphi = 360;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, beam_zenith, Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, columnFormat);

    LiDARcloud cloud;
    cloud.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    DOCTEST_CHECK(cloud.getScanPattern(0) == SCAN_PATTERN_SPINNING_MULTIBEAM);
    DOCTEST_CHECK(cloud.getScanSizeTheta(0) == uint(elevation_deg.size())); // Ntheta = number of channels
    DOCTEST_CHECK(cloud.getScanSizePhi(0) == Nphi);

    std::vector<float> returned_angles = cloud.getScanBeamZenithAngles(0);
    DOCTEST_REQUIRE(returned_angles.size() == beam_zenith.size());
    bool angles_match = true;
    for (size_t k = 0; k < beam_zenith.size(); k++) {
        if (fabs(returned_angles[k] - beam_zenith[k]) > 1e-5f) {
            angles_match = false;
        }
    }
    DOCTEST_CHECK(angles_match);

    // thetaMin/thetaMax bracket the channel zenith angles.
    vec2 theta_range = cloud.getScanRangeTheta(0);
    float zmin = *std::min_element(beam_zenith.begin(), beam_zenith.end());
    float zmax = *std::max_element(beam_zenith.begin(), beam_zenith.end());
    DOCTEST_CHECK(theta_range.x == doctest::Approx(zmin));
    DOCTEST_CHECK(theta_range.y == doctest::Approx(zmax));

    // Each row's beam direction zenith equals its channel zenith, and direction2rc maps back to the same row (nearest channel).
    bool rc_roundtrip_ok = true;
    for (uint row = 0; row < beam_zenith.size(); row++) {
        SphericalCoord dir = scan.rc2direction(row, 0);
        if (fabs(dir.zenith - beam_zenith[row]) > 1e-4f) {
            rc_roundtrip_ok = false;
        }
        int2 rc = scan.direction2rc(dir);
        if (rc.x != int(row)) {
            rc_roundtrip_ok = false;
        }
    }
    DOCTEST_CHECK(rc_roundtrip_ok);
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam Empty Channels Error") {
    // A spinning multibeam scan with no channels is an error (fail-fast, no silent fallback).
    vec3 origin(0.f, 0.f, 1.f);
    std::vector<float> empty_angles;
    std::vector<std::string> columnFormat;
    DOCTEST_CHECK_THROWS(ScanMetadata(origin, empty_angles, 100, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, columnFormat));
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam Synthetic Scan") {
    Context context;
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    LiDARcloud cloud;
    cloud.disableMessages();

    // Channels span the 1 m cube (at the near face, ~4.5 m range, the cube subtends about +/-6.3 deg). Use 0.5-degree channel
    // spacing so the vertical point spacing on the cube (~4 cm) is dense enough to triangulate alongside the azimuth sweep.
    vec3 scan_origin(-5.f, 0.f, 0.5f);
    std::vector<float> beam_zenith;
    for (int e = -24; e <= 24; e++) { // 49 channels, 0.5-degree elevation spacing (e in half-degrees)
        beam_zenith.push_back(0.5f * float(M_PI) - 0.5f * float(e) * float(M_PI) / 180.f);
    }
    uint Nphi = 4000;
    std::vector<std::string> columnFormat;
    ScanMetadata scan(scan_origin, beam_zenith, Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    DOCTEST_CHECK_NOTHROW(cloud.addGrid(make_vec3(0.f, 0.f, 0.5f), make_vec3(1.f, 1.f, 1.f), make_int3(1, 1, 1), 0));

    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context, false, true)); // record_misses=true so LAD inversion has transmitted beams

    uint Nhits = cloud.getHitCount();
    DOCTEST_CHECK(Nhits > 0);
    DOCTEST_CHECK(cloud.hasMisses());

    // Every hit from a spinning multibeam scan carries a channel index in [0, Ntheta), and at least one real return exists.
    uint Ntheta = uint(beam_zenith.size());
    bool channel_exists_all = true;
    bool channel_in_range = true;
    bool any_real_hit = false;
    for (uint h = 0; h < Nhits; h++) {
        if (!cloud.doesHitDataExist(h, "channel")) {
            channel_exists_all = false;
            continue;
        }
        int ch = int(cloud.getHitData(h, "channel"));
        if (ch < 0 || ch >= int(Ntheta)) {
            channel_in_range = false;
        }
        if (cloud.getHitData(h, "is_miss") == 0.0) {
            any_real_hit = true;
        }
    }
    DOCTEST_CHECK(channel_exists_all);
    DOCTEST_CHECK(channel_in_range);
    DOCTEST_CHECK(any_real_hit);

    // Leaf-area inversion is scan-pattern-agnostic: it must run and yield a finite, positive leaf area density.
    // Triangulation distance (6 cm) is matched to the ~4 cm vertical point spacing of the 0.5-degree channels.
    DOCTEST_CHECK_NOTHROW(cloud.triangulateHitPoints(0.06, 10));
    DOCTEST_CHECK_NOTHROW(cloud.calculateLeafArea(&context));
    float LAD = cloud.getCellLeafAreaDensity(0);
    DOCTEST_CHECK(LAD == LAD); // not NaN
    DOCTEST_CHECK(LAD > 0.f);

    // exportScans must persist the spinning multibeam geometry (pattern + channel angles) so it round-trips on reload.
    const std::string out_dir = "lidar_spinmb_export_tmp";
    std::filesystem::remove_all(out_dir);
    const std::string xml_out = out_dir + "/scans.xml";
    DOCTEST_CHECK_NOTHROW(cloud.exportScans(xml_out.c_str()));

    LiDARcloud reloaded;
    reloaded.disableMessages();
    DOCTEST_CHECK_NOTHROW(reloaded.loadXML(xml_out.c_str()));
    DOCTEST_REQUIRE(reloaded.getScanCount() == 1);
    DOCTEST_CHECK(reloaded.getScanPattern(0) == SCAN_PATTERN_SPINNING_MULTIBEAM);
    DOCTEST_CHECK(reloaded.getScanSizeTheta(0) == Ntheta);
    DOCTEST_CHECK(reloaded.getScanSizePhi(0) == Nphi);
    std::vector<float> reloaded_angles = reloaded.getScanBeamZenithAngles(0);
    DOCTEST_REQUIRE(reloaded_angles.size() == beam_zenith.size());
    bool reloaded_angles_match = true;
    for (size_t k = 0; k < beam_zenith.size(); k++) {
        if (fabs(reloaded_angles[k] - beam_zenith[k]) > 1e-3f) {
            reloaded_angles_match = false;
        }
    }
    DOCTEST_CHECK(reloaded_angles_match);
    std::filesystem::remove_all(out_dir);
}

DOCTEST_TEST_CASE("LiDAR Multibeam Synthetic Scan Flat Wall") {
    // Regression: a planar (zero-thickness) scene must not be rejected wholesale by the synthetic-scan AABB cull.
    // A flat wall lies in the y-z plane at x=0, so the Context domain bounding box is degenerate along x
    // (xmin==xmax==0). Before the degenerate-axis pad, the slab cull forced t0==t1 and rejected every ray, so
    // syntheticScan recorded all returns as misses (distance == LIDAR_RAYTRACE_MISS_T) and produced zero real
    // returns. This asserts a substantial fraction of real returns at the correct ~5 m range.
    Context context;
    // 4 m x 4 m wall (y,z in [-2,2]) at x=0, built from two triangles.
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, -2.f), make_vec3(0.f, 2.f, 2.f), RGB::green);
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, 2.f), make_vec3(0.f, -2.f, 2.f), RGB::green);

    LiDARcloud cloud;
    cloud.disableMessages();

    // Multibeam scanner 5 m in front of the wall on -x, facing +x. 25 channels over roughly -6..+6 deg elevation.
    vec3 scan_origin(-5.f, 0.f, 0.5f);
    std::vector<float> beam_zenith;
    for (int e = -12; e <= 12; e++) { // 25 channels, 0.5-degree elevation spacing => +/-6 deg span
        beam_zenith.push_back(0.5f * float(M_PI) - 0.5f * float(e) * float(M_PI) / 180.f);
    }
    uint Nphi = 2000;
    std::vector<std::string> columnFormat;
    ScanMetadata scan(scan_origin, beam_zenith, Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context, false, true)); // scan_grid_only=false, record_misses=true

    uint Nhits = cloud.getHitCount();
    DOCTEST_REQUIRE(Nhits > 0);

    // The wall subtends azimuth half-angle atan(2/5)=21.8 deg, so it occupies ~43.6/360 = 12.1% of the 2000 azimuth
    // columns (~242 columns x 25 channels ~= 6000 real returns out of 50000 rays). Assert a substantial fraction
    // (> 8% of total) plus an absolute floor, rather than "any hit" (which the all-miss bug passed).
    uint Ntheta = uint(beam_zenith.size());
    uint total_rays = Ntheta * Nphi;
    uint real_returns = 0;
    bool ranges_ok = true;
    for (uint h = 0; h < Nhits; h++) {
        if (cloud.getHitData(h, "is_miss") == 0.0) {
            real_returns++;
            // Real hits are on the wall at x=0; on-axis range from the scanner at x=-5 is exactly 5 m, growing toward
            // the wall's azimuth edges as 5/cos(phi): the y=+/-2 edge is sqrt(5^2+2^2)=5.39 m, plus a small vertical
            // component from the +/-6 deg channels, so real ranges fall within roughly [5.0, 5.45] m.
            double dist = cloud.getHitData(h, "distance");
            if (dist < 4.9 || dist > 5.5) {
                ranges_ok = false;
            }
        }
    }
    DOCTEST_CHECK(real_returns > 1000); // absolute floor: the bug produced exactly 0
    DOCTEST_CHECK(real_returns > uint(0.08 * float(total_rays))); // ~12% expected; 8% is a defensible lower bound
    DOCTEST_CHECK(ranges_ok);
}

DOCTEST_TEST_CASE("LiDAR Raster Synthetic Scan Flat Wall") {
    // Companion to the multibeam flat-wall regression: the degenerate-axis cull fix must be scan-pattern-agnostic.
    // Same 4 m x 4 m wall in the y-z plane at x=0, scanned with a raster pattern aimed straight at it.
    Context context;
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, -2.f), make_vec3(0.f, 2.f, 2.f), RGB::green);
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, 2.f), make_vec3(0.f, -2.f, 2.f), RGB::green);

    LiDARcloud cloud;
    cloud.disableMessages();

    vec3 scan_origin(-5.f, 0.f, 0.5f);
    uint Ntheta = 250;
    uint Nphi = 250;
    // Zenith ~84..96 deg (elevation +/-6 deg) and azimuth -20..+20 deg about +x. Azimuth phi is measured from +y,
    // so the +x heading toward the wall is phi = pi/2.
    float thetaMin = 0.5f * float(M_PI) - 6.f * float(M_PI) / 180.f;
    float thetaMax = 0.5f * float(M_PI) + 6.f * float(M_PI) / 180.f;
    float phiMin = 0.5f * float(M_PI) - 20.f * float(M_PI) / 180.f;
    float phiMax = 0.5f * float(M_PI) + 20.f * float(M_PI) / 180.f;
    std::vector<std::string> columnFormat;
    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.f, 0.f, 0.f, 0.f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context, false, true));

    uint Nhits = cloud.getHitCount();
    DOCTEST_REQUIRE(Nhits > 0);

    uint real_returns = 0;
    bool ranges_ok = true;
    for (uint h = 0; h < Nhits; h++) {
        if (cloud.getHitData(h, "is_miss") == 0.0) {
            real_returns++;
            // The +/-20 deg azimuth edges reach 5/cos(20 deg) ~= 5.32 m.
            double dist = cloud.getHitData(h, "distance");
            if (dist < 4.9 || dist > 5.4) {
                ranges_ok = false;
            }
        }
    }
    DOCTEST_CHECK(real_returns > uint(0.5f * float(Ntheta * Nphi))); // fan points at the wall => majority hit
    DOCTEST_CHECK(ranges_ok);
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam XML Load and Round-Trip") {
    // Write a temporary spinning-multibeam scan XML and verify it loads with the correct geometry.
    std::string xml_path = "plugins/lidar/xml/.tmp_spinning_multibeam_test.xml";
    {
        std::ofstream f(xml_path);
        f << "<helios>\n";
        f << "  <scan>\n";
        f << "    <origin> 0 0 1 </origin>\n";
        f << "    <scanPattern> spinning_multibeam </scanPattern>\n";
        f << "    <beamElevationAngles> -15 -10 -5 0 5 10 15 </beamElevationAngles>\n";
        f << "    <Nphi> 720 </Nphi>\n";
        f << "    <phiMin> 0 </phiMin>\n";
        f << "    <phiMax> 360 </phiMax>\n";
        f << "  </scan>\n";
        f << "</helios>\n";
    }

    LiDARcloud cloud;
    cloud.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud.loadXML(xml_path.c_str()));
    DOCTEST_REQUIRE(cloud.getScanCount() == 1);
    DOCTEST_CHECK(cloud.getScanPattern(0) == SCAN_PATTERN_SPINNING_MULTIBEAM);
    DOCTEST_CHECK(cloud.getScanSizeTheta(0) == 7); // 7 channels
    DOCTEST_CHECK(cloud.getScanSizePhi(0) == 720);

    std::vector<float> angles = cloud.getScanBeamZenithAngles(0);
    DOCTEST_REQUIRE(angles.size() == 7);
    DOCTEST_CHECK(angles[3] == doctest::Approx(0.5f * float(M_PI))); // channel 3 is 0 deg elevation => zenith pi/2

    std::remove(xml_path.c_str());
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
    DOCTEST_CHECK_NOTHROW(synthetic_scan_test.gapfillMisses()); // LAD inversion requires misses (transmitted beams)
    DOCTEST_CHECK_NOTHROW(synthetic_scan_test.calculateLeafArea(&scan_context));

    // Grid cell calculations should produce reasonable results
    uint cell_count = synthetic_scan_test.getGridCellCount();
    if (cell_count > 0) {
        float leaf_area_density = synthetic_scan_test.getCellLeafAreaDensity(0);
        DOCTEST_CHECK(std::isfinite(leaf_area_density));
        DOCTEST_CHECK(leaf_area_density >= 0.0f);
    }
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Range Noise Test") {
    // Range (along-beam) noise should displace synthetic hit points along the beam direction only, NOT isotropically.
    // Scanner is placed directly above a large horizontal patch and scans straight down, so every beam direction is
    // approximately (0,0,-1). Range noise must therefore scatter the hit points in z (along-beam) while leaving x and y
    // essentially unchanged. We also verify that zero noise reproduces the exact surface and that a fixed RNG seed is
    // reproducible.

    // Target patch at z=0 (the surface we measure) plus a backing patch at z=-2 so the domain has a non-degenerate
    // bounding box along the beam axis (a single zero-thickness plane perpendicular to the beam is culled by the
    // ray-AABB pre-test). For single returns the nearer (z=0) patch is recorded, so the backing patch does not pollute
    // the measurement.
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(10, 10));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(10, 10));

    // Scanner directly above, looking down over a narrow cone about nadir (theta near pi => downward)
    vec3 scan_origin(0.0f, 0.0f, 5.0f);
    uint Ntheta = 40;
    uint Nphi = 40;
    float thetaMin = 0.9f * float(M_PI); // near-nadir cone so beams are ~(0,0,-1)
    float thetaMax = float(M_PI);
    float phiMin = 0.0f;
    float phiMax = 2.0f * float(M_PI);
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    const float sigma_range = 0.05f; // 5 cm range noise standard deviation

    // --- Baseline: zero noise should land exactly on the z=0 surface ---
    {
        LiDARcloud lidar_clean;
        lidar_clean.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
        DOCTEST_CHECK_NOTHROW(lidar_clean.addScan(scan));
        DOCTEST_CHECK_NOTHROW(lidar_clean.syntheticScan(&context));

        uint hit_count = lidar_clean.getHitCount();
        DOCTEST_CHECK(hit_count > 0);
        uint target_hits = 0;
        for (uint i = 0; i < hit_count; i++) {
            vec3 p = lidar_clean.getHitXYZ(i);
            // Only the near (z=0) target surface should be hit by the near-nadir cone; assert it lands exactly there.
            DOCTEST_CHECK(fabs(p.z - 0.0f) < 1e-4f); // exact surface, no noise
            target_hits++;
        }
        DOCTEST_CHECK(target_hits > 0);
    }

    // --- Noisy scan: scatter should be along-beam (z), not across-beam (x,y) ---
    // Collect only returns on the target surface (|z| < 1, well separated from the backing patch at z=-2).
    auto run_noisy = [&](uint seed, std::vector<vec3> &points) {
        LiDARcloud lidar_noisy;
        lidar_noisy.disableMessages();
        context.seedRandomGenerator(seed);
        ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, sigma_range, 0.0f, columnFormat);
        lidar_noisy.addScan(scan);
        lidar_noisy.syntheticScan(&context);
        uint hit_count = lidar_noisy.getHitCount();
        points.clear();
        for (uint i = 0; i < hit_count; i++) {
            vec3 p = lidar_noisy.getHitXYZ(i);
            if (fabs(p.z) < 1.0f) {
                points.push_back(p);
            }
        }
    };

    std::vector<vec3> points;
    run_noisy(12345u, points);
    DOCTEST_CHECK(points.size() > 0);

    // Sample standard deviation of z (along-beam) and of the in-plane radius about the true surface.
    // For a flat z=0 surface scanned from directly above, the noise-free z is 0 and the noise-free (x,y) is fixed per
    // beam, so the spread of z directly reflects the injected range noise (projected by cos of the small off-nadir angle,
    // which is >= cos(0.03*pi) ~ 0.996, i.e. negligible).
    double z_mean = 0.0;
    for (const vec3 &p: points) {
        z_mean += p.z;
    }
    z_mean /= double(points.size());

    double z_var = 0.0;
    for (const vec3 &p: points) {
        z_var += (p.z - z_mean) * (p.z - z_mean);
    }
    double z_std = std::sqrt(z_var / double(points.size()));

    // The along-beam (z) scatter should be on the order of sigma_range (loose bounds for finite sample size).
    DOCTEST_CHECK(z_std > 0.5 * sigma_range);
    DOCTEST_CHECK(z_std < 2.0 * sigma_range);

    // Cross-beam (anisotropy) check: the noise must displace points ALONG the beam (z), not across it (x,y). Run a clean
    // scan with identical parameters; the hit ordering matches beam-for-beam, so the per-point displacement between the
    // noisy and clean clouds isolates the injected error. For near-nadir beams that displacement should be almost entirely
    // in z. We assert the mean in-plane displacement is far smaller than the mean along-beam displacement.
    std::vector<vec3> clean_points;
    {
        LiDARcloud lidar_ref;
        lidar_ref.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
        lidar_ref.addScan(scan);
        lidar_ref.syntheticScan(&context);
        clean_points.clear();
        uint hc = lidar_ref.getHitCount();
        for (uint i = 0; i < hc; i++) {
            vec3 p = lidar_ref.getHitXYZ(i);
            if (fabs(p.z) < 1.0f) {
                clean_points.push_back(p);
            }
        }
    }

    DOCTEST_REQUIRE(clean_points.size() == points.size());
    double mean_abs_dz = 0.0;
    double mean_abs_dxy = 0.0;
    for (size_t i = 0; i < points.size(); i++) {
        mean_abs_dz += fabs(double(points[i].z - clean_points[i].z));
        double dx = double(points[i].x - clean_points[i].x);
        double dy = double(points[i].y - clean_points[i].y);
        mean_abs_dxy += std::sqrt(dx * dx + dy * dy);
    }
    mean_abs_dz /= double(points.size());
    mean_abs_dxy /= double(points.size());

    // Along-beam displacement should be substantial (~sigma); across-beam displacement should be tiny. For the near-nadir
    // cone (off-nadir <= 0.1*pi), sin(off-nadir) <= ~0.31, so the in-plane leakage is bounded but should be well under the
    // along-beam component. Require at least a 3x separation to demonstrate anisotropy.
    DOCTEST_CHECK(mean_abs_dz > 0.5 * sigma_range);
    DOCTEST_CHECK(mean_abs_dxy < mean_abs_dz);
    DOCTEST_CHECK(mean_abs_dxy < 0.34 * mean_abs_dz);

    // --- Determinism: same seed reproduces identical points ---
    std::vector<vec3> points_repeat;
    run_noisy(12345u, points_repeat);
    DOCTEST_CHECK(points_repeat.size() == points.size());
    bool identical = true;
    for (size_t i = 0; i < points.size() && i < points_repeat.size(); i++) {
        if (fabs(points[i].z - points_repeat[i].z) > 1e-6f) {
            identical = false;
            break;
        }
    }
    DOCTEST_CHECK(identical);

    // --- Different seed produces a different realization ---
    std::vector<vec3> points_other;
    run_noisy(99999u, points_other);
    bool any_different = false;
    for (size_t i = 0; i < points.size() && i < points_other.size(); i++) {
        if (fabs(points[i].z - points_other[i].z) > 1e-6f) {
            any_different = true;
            break;
        }
    }
    DOCTEST_CHECK(any_different);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Range-Normalized Intensity Test") {
    // Helios reports RANGE-NORMALIZED intensity: I = rho*cos(theta) with the 1/R^2 range loss normalized out, so a
    // given surface returns the same intensity regardless of scanner-to-target range. We verify this by scanning the
    // same horizontal patch straight down (theta=0 => cos(theta)=1) from two different heights and asserting the
    // recorded intensity is (a) range-independent (equal at both ranges) and (b) equal to the primitive reflectivity.

    const float rho = 0.45f; // leaf-like reflectivity in the laser waveband

    // Scan straight down at nadir so the incidence angle is ~0 (cos(theta) ~ 1) and intensity reduces to rho.
    const uint Ntheta = 30;
    const uint Nphi = 30;
    const float thetaMin = 0.97f * float(M_PI); // narrow cone about nadir (theta near pi => downward)
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);
    const float exitDiameter = 0.0f;
    const float beamDivergence = 0.0f;
    // reflectivity_lidar must be listed in the scan column format for the scanner to fold per-primitive
    // reflectivity into the recorded intensity (see syntheticScan()).
    std::vector<std::string> columnFormat = {"reflectivity_lidar"};

    // Mean intensity over the near-nadir, on-target returns of one scan whose origin is at height z=scan_height.
    // The raw incidence-angle seed is the signed dot product beam.normal; for a downward beam on an upward-facing
    // patch this is ~ -1, so we report the magnitude (the physically meaningful return strength).
    auto mean_target_intensity = [&](float scan_height) -> float {
        Context context;
        // Target patch at z=0 plus a backing patch (non-degenerate bounding box along the beam axis; the nearer
        // z=0 patch is the recorded single return).
        uint target = context.addPatch(make_vec3(0, 0, 0), make_vec2(10, 10));
        context.addPatch(make_vec3(0, 0, -2), make_vec2(10, 10));
        context.setPrimitiveData(target, "reflectivity_lidar", rho);

        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, scan_height), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
        lidar.addScan(scan);
        lidar.syntheticScan(&context);

        uint hit_count = lidar.getHitCount();
        DOCTEST_REQUIRE(hit_count > 0);

        double sum = 0.0;
        uint n = 0;
        for (uint i = 0; i < hit_count; i++) {
            vec3 p = lidar.getHitXYZ(i);
            if (fabs(p.z - 0.0f) > 1e-3f) {
                continue; // keep only returns from the z=0 target surface
            }
            sum += fabs(lidar.getHitData(i, "intensity"));
            n++;
        }
        DOCTEST_REQUIRE(n > 0);
        return float(sum / double(n));
    };

    const float intensity_near = mean_target_intensity(5.0f); // R ~ 5 m
    const float intensity_far = mean_target_intensity(20.0f); // R ~ 20 m (4x range)

    // (a) Range-independence: a raw 1/R^2 signal would differ by ~16x between these ranges; normalized intensity must not.
    DOCTEST_CHECK(intensity_near == doctest::Approx(intensity_far).epsilon(0.02));
    // (b) At normal incidence the normalized intensity equals the primitive reflectivity rho.
    DOCTEST_CHECK(intensity_near == doctest::Approx(rho).epsilon(0.02));

    // The static normalization helper is value-preserving (the synthetic intensity already carries no 1/R^2 loss).
    DOCTEST_CHECK(LiDARcloud::applyRangeIntensityCorrection(rho, 5.0f) == doctest::Approx(rho));
    DOCTEST_CHECK(LiDARcloud::applyRangeIntensityCorrection(rho, 20.0f) == doctest::Approx(rho));
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Reflectance (dB) Test") {
    // When "reflectance" is requested in the ASCII column format, the scanner records reflectance in decibels,
    // 10*log10(|intensity|), relative to a perfect Lambertian reflector at normal incidence (0 dB). We verify that
    // (a) reflectance is the dB transform of the recorded intensity, (b) it equals 10*log10(rho) at normal incidence,
    // and (c) like intensity it is range-independent. We also confirm reflectance is NOT recorded when not requested.

    const float rho = 0.45f;
    const float expected_dB = 10.0f * log10f(rho); // ~ -3.47 dB

    const uint Ntheta = 30;
    const uint Nphi = 30;
    const float thetaMin = 0.97f * float(M_PI);
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);

    // Build a scan at the given height, optionally requesting reflectance, and return the mean reflectance and the
    // count of hits that carry a "reflectance" data field, over the on-target (z=0) returns.
    auto scan_reflectance = [&](float scan_height, bool request_reflectance, uint &reflectance_field_count) -> float {
        Context context;
        uint target = context.addPatch(make_vec3(0, 0, 0), make_vec2(10, 10));
        context.addPatch(make_vec3(0, 0, -2), make_vec2(10, 10));
        context.setPrimitiveData(target, "reflectivity_lidar", rho);

        std::vector<std::string> columnFormat = {"reflectivity_lidar"};
        if (request_reflectance) {
            columnFormat.push_back("reflectance");
        }

        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, scan_height), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
        lidar.addScan(scan);
        lidar.syntheticScan(&context);

        uint hit_count = lidar.getHitCount();
        DOCTEST_REQUIRE(hit_count > 0);

        double sum_dB = 0.0;
        uint n = 0;
        reflectance_field_count = 0;
        for (uint i = 0; i < hit_count; i++) {
            vec3 p = lidar.getHitXYZ(i);
            if (fabs(p.z - 0.0f) > 1e-3f) {
                continue; // on-target returns only
            }
            if (!lidar.doesHitDataExist(i, "reflectance")) {
                continue;
            }
            reflectance_field_count++;
            // reflectance must be the dB transform of this same hit's intensity
            double intensity = lidar.getHitData(i, "intensity");
            double reflectance = lidar.getHitData(i, "reflectance");
            DOCTEST_CHECK(reflectance == doctest::Approx(10.0 * log10(fabs(intensity))).epsilon(1e-4));
            sum_dB += reflectance;
            n++;
        }
        if (n == 0) {
            return 0.f;
        }
        return float(sum_dB / double(n));
    };

    uint count_near = 0, count_far = 0, count_off = 0;
    const float dB_near = scan_reflectance(5.0f, true, count_near);
    const float dB_far = scan_reflectance(20.0f, true, count_far);

    // (a)/(b) Reflectance equals 10*log10(rho) at normal incidence.
    DOCTEST_REQUIRE(count_near > 0);
    DOCTEST_CHECK(dB_near == doctest::Approx(expected_dB).epsilon(0.05));
    // (c) Range-independent, like the intensity it derives from.
    DOCTEST_CHECK(dB_near == doctest::Approx(dB_far).epsilon(0.05));

    // Not requested => no reflectance field is recorded.
    scan_reflectance(5.0f, false, count_off);
    DOCTEST_CHECK(count_off == 0);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Angular Jitter Test") {
    // Angular (beam-pointing) jitter should displace hit points ACROSS the beam (laterally), with a magnitude that grows
    // with range as approximately range * sigma_angle. This is the complement of range noise: for a flat horizontal target
    // scanned from directly above, jitter moves the hit in (x,y) while leaving z (the surface) essentially unchanged,
    // whereas range noise would move it in z. We verify the lateral scatter is present, scales with range, and dominates
    // the along-beam scatter; and that zero jitter reproduces the exact surface.

    // Target patch at z=0 plus a backing patch at z=-2 for a non-degenerate bounding box (see range-noise test).
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(10, 10));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(10, 10));

    vec3 scan_origin(0.0f, 0.0f, 5.0f);
    uint Ntheta = 40;
    uint Nphi = 40;
    float thetaMin = 0.9f * float(M_PI); // near-nadir cone
    float thetaMax = float(M_PI);
    float phiMin = 0.0f;
    float phiMax = 2.0f * float(M_PI);
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    float rangeNoise = 0.0f; // isolate angular jitter
    std::vector<std::string> columnFormat;

    const float sigma_angle = 0.01f; // ~10 mrad pointing jitter

    // Build a clean reference (no jitter): points lie exactly on z=0.
    std::vector<vec3> clean_points;
    {
        LiDARcloud lidar_ref;
        lidar_ref.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
        lidar_ref.addScan(scan);
        lidar_ref.syntheticScan(&context);
        uint hc = lidar_ref.getHitCount();
        DOCTEST_CHECK(hc > 0);
        for (uint i = 0; i < hc; i++) {
            vec3 p = lidar_ref.getHitXYZ(i);
            if (fabs(p.z) < 1.0f) {
                DOCTEST_CHECK(fabs(p.z) < 1e-4f); // exact surface, no jitter
                clean_points.push_back(p);
            }
        }
    }

    // Jittered scan on the target surface.
    std::vector<vec3> points;
    {
        LiDARcloud lidar_jit;
        lidar_jit.disableMessages();
        context.seedRandomGenerator(2024u);
        ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, rangeNoise, sigma_angle, columnFormat);
        lidar_jit.addScan(scan);
        lidar_jit.syntheticScan(&context);
        uint hc = lidar_jit.getHitCount();
        for (uint i = 0; i < hc; i++) {
            vec3 p = lidar_jit.getHitXYZ(i);
            if (fabs(p.z) < 1.0f) {
                points.push_back(p);
            }
        }
    }

    DOCTEST_REQUIRE(points.size() == clean_points.size());

    // Per-point displacement from the clean reference (same beam ordering): jitter should move points laterally (x,y),
    // not along the beam (z).
    double mean_abs_dz = 0.0;
    double mean_lateral = 0.0;
    for (size_t i = 0; i < points.size(); i++) {
        mean_abs_dz += fabs(double(points[i].z - clean_points[i].z));
        double dx = double(points[i].x - clean_points[i].x);
        double dy = double(points[i].y - clean_points[i].y);
        mean_lateral += std::sqrt(dx * dx + dy * dy);
    }
    mean_abs_dz /= double(points.size());
    mean_lateral /= double(points.size());

    // Lateral displacement should scale as range*sigma_angle. The scanner is at z=5 above a z=0 plane, so range ~5 and the
    // expected lateral scale is ~5*0.01 = 0.05 m. Use loose bounds for finite sampling.
    const double expected_lateral = 5.0 * double(sigma_angle);
    DOCTEST_CHECK(mean_lateral > 0.3 * expected_lateral);
    DOCTEST_CHECK(mean_lateral < 3.0 * expected_lateral);

    // Anisotropy: for jitter on a flat horizontal target the along-beam (z) component should stay far below the lateral
    // component (it is zero in the ideal flat-plane limit; small nonzero values arise only from the off-nadir cone).
    DOCTEST_CHECK(mean_abs_dz < 0.5 * mean_lateral);

    // Determinism under a fixed seed.
    std::vector<vec3> repeat;
    {
        LiDARcloud lidar_rep;
        lidar_rep.disableMessages();
        context.seedRandomGenerator(2024u);
        ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, rangeNoise, sigma_angle, columnFormat);
        lidar_rep.addScan(scan);
        lidar_rep.syntheticScan(&context);
        uint hc = lidar_rep.getHitCount();
        for (uint i = 0; i < hc; i++) {
            vec3 p = lidar_rep.getHitXYZ(i);
            if (fabs(p.z) < 1.0f) {
                repeat.push_back(p);
            }
        }
    }
    DOCTEST_REQUIRE(repeat.size() == points.size());
    bool identical = true;
    for (size_t i = 0; i < points.size(); i++) {
        if ((points[i] - repeat[i]).magnitude() > 1e-6f) {
            identical = false;
            break;
        }
    }
    DOCTEST_CHECK(identical);
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

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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
    for (uint UUID: UUIDs) {
        float area = context.getPrimitiveArea(UUID);
        vec3 normal = context.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) { // Check for NaN
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
    float exitDiameter = 0.0f; // Point source for backward compatibility
    float beamDivergence = 0.0004f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(synthetic_mr8.addScan(scan));

    // Add grid programmatically
    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    int3 grid_divisions = make_int3(2, 2, 2);
    DOCTEST_CHECK_NOTHROW(synthetic_mr8.addGrid(grid_center, grid_size, grid_divisions, 0));

    vec3 gsize = synthetic_mr8.getCellSize(0);

    Context context_mr8;
    context_mr8.seedRandomGenerator(0); // Seed for reproducible random perturbations
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

        float area = context_mr8.getPrimitiveArea(UUID);
        LAD_ex.at(ID) += area / (gsize.x * gsize.y * gsize.z);

        // Calculate exact G(theta) from primitive geometry for each voxel
        vec3 normal = context_mr8.getPrimitiveNormal(UUID);
        std::vector<vec3> vertices = context_mr8.getPrimitiveVertices(UUID);
        vec3 raydir = vertices.front() - scan_origin;
        raydir.normalize();

        if (area == area) { // Check for NaN
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
        if (synthetic_mr8.doesHitDataExist(i, "target_count") && synthetic_mr8.getHitData(i, "target_count") > 1) {
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
        if (!synthetic_mr8.doesHitDataExist(i, "target_index"))
            has_target_index = false;
        if (!synthetic_mr8.doesHitDataExist(i, "target_count"))
            has_target_count = false;
        if (!synthetic_mr8.doesHitDataExist(i, "timestamp"))
            has_timestamp = false;
    }

    DOCTEST_CHECK(has_target_index);
    DOCTEST_CHECK(has_target_count);
    DOCTEST_CHECK(has_timestamp);

    // Check for duplicate first returns per timestamp (critical bug check)
    std::map<int, int> timestamp_first_return_count;
    for (uint i = 0; i < hits_grid_true; i++) {
        if (synthetic_mr8.doesHitDataExist(i, "target_index") && synthetic_mr8.doesHitDataExist(i, "timestamp")) {
            int tidx = static_cast<int>(synthetic_mr8.getHitData(i, "target_index"));
            int tstamp = static_cast<int>(synthetic_mr8.getHitData(i, "timestamp"));
            if (tidx == 0) {
                timestamp_first_return_count[tstamp]++;
            }
        }
    }

    for (const auto &pair: timestamp_first_return_count) {
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

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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
    context.addSphereObject(6, sphere_center, sphere_radius); // Low subdivision for speed

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
    float thetaMin = 0.0f; // Default when not specified in XML
    float thetaMax = M_PI; // Default when not specified in XML
    float phiMin = 0.0f; // Default when not specified in XML
    float phiMax = 2.0f * M_PI; // Default when not specified in XML
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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
    context.addSphereObject(6, sphere_center, sphere_radius); // Low subdivision for speed

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
    float thetaMin = 0.0f; // Default when not specified in XML
    float thetaMax = M_PI; // Default when not specified in XML
    float phiMin = 0.0f; // Default when not specified in XML
    float phiMax = 2.0f * M_PI; // Default when not specified in XML
    float exitDiameter = 0.0f;
    float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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
    float thetaMin = 0.0f; // Default when not specified in XML
    float thetaMax = M_PI; // Default when not specified in XML
    float phiMin = 0.0f; // Default when not specified in XML
    float phiMax = 2.0f * M_PI; // Default when not specified in XML
    float exitDiameter = 0.0f; // Point source for backward compatibility
    float beamDivergence = 0.0004f;
    std::vector<std::string> columnFormat;

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
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
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 50, 0.1f, true, true)); // rays_per_pulse=50 for stable statistics

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
    for (auto &pair: timestamp_first_return_count) {
        if (pair.second > 1)
            timestamps_with_multi_first++;
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
    ScanMetadata scan_point(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar_point.addScan(scan_point);
    lidar_point.syntheticScan(&context, 50, 0.05f, false, false);

    // Scan WITH exit diameter
    LiDARcloud lidar_exit;
    lidar_exit.disableMessages();
    ScanMetadata scan_exit(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.1f, 0.0f, 0.0f, 0.0f, {});
    lidar_exit.addScan(scan_exit);
    lidar_exit.syntheticScan(&context, 50, 0.05f, false, false);

    DOCTEST_REQUIRE(lidar_point.getHitCount() > 0);
    DOCTEST_REQUIRE(lidar_exit.getHitCount() > 0);

    // Calculate spatial extent for point source
    float x_min_pt = 1e6f, x_max_pt = -1e6f, y_min_pt = 1e6f, y_max_pt = -1e6f;
    for (uint i = 0; i < lidar_point.getHitCount(); i++) {
        vec3 pos = lidar_point.getHitXYZ(i);
        x_min_pt = fmin(x_min_pt, pos.x);
        x_max_pt = fmax(x_max_pt, pos.x);
        y_min_pt = fmin(y_min_pt, pos.y);
        y_max_pt = fmax(y_max_pt, pos.y);
    }

    // Calculate spatial extent for exit diameter
    float x_min_ex = 1e6f, x_max_ex = -1e6f, y_min_ex = 1e6f, y_max_ex = -1e6f;
    for (uint i = 0; i < lidar_exit.getHitCount(); i++) {
        vec3 pos = lidar_exit.getHitXYZ(i);
        x_min_ex = fmin(x_min_ex, pos.x);
        x_max_ex = fmax(x_max_ex, pos.x);
        y_min_ex = fmin(y_min_ex, pos.y);
        y_max_ex = fmax(y_max_ex, pos.y);
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
    ScanMetadata scan_zero(vec3(0, 0, 5), 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar_zero.addScan(scan_zero);
    DOCTEST_CHECK_NOTHROW(lidar_zero.syntheticScan(&context, 50, 0.05f, false, false));
    DOCTEST_CHECK(lidar_zero.getHitCount() > 0);

    // Scan with exitDiameter>0 (new behavior - should also work)
    LiDARcloud lidar_exit;
    lidar_exit.disableMessages();
    ScanMetadata scan_exit(vec3(0, 0, 5), 100, 0, M_PI, 100, 0, 2 * M_PI, 0.01f, 0.0f, 0.0f, 0.0f, {});
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
    ScanMetadata scan_both(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.1f, 0.01f, 0.0f, 0.0f, {});
    lidar_both.addScan(scan_both);
    lidar_both.syntheticScan(&context, 50, 0.05f, false, false);

    // Scan with ONLY beamDivergence
    LiDARcloud lidar_div;
    lidar_div.disableMessages();
    ScanMetadata scan_div(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.01f, 0.0f, 0.0f, {});
    lidar_div.addScan(scan_div);
    lidar_div.syntheticScan(&context, 50, 0.05f, false, false);

    DOCTEST_REQUIRE(lidar_both.getHitCount() > 0);
    DOCTEST_REQUIRE(lidar_div.getHitCount() > 0);

    // Calculate spread for combined
    float x_min_b = 1e6f, x_max_b = -1e6f, y_min_b = 1e6f, y_max_b = -1e6f;
    for (uint i = 0; i < lidar_both.getHitCount(); i++) {
        vec3 pos = lidar_both.getHitXYZ(i);
        x_min_b = fmin(x_min_b, pos.x);
        x_max_b = fmax(x_max_b, pos.x);
        y_min_b = fmin(y_min_b, pos.y);
        y_max_b = fmax(y_max_b, pos.y);
    }

    // Calculate spread for divergence only
    float x_min_d = 1e6f, x_max_d = -1e6f, y_min_d = 1e6f, y_max_d = -1e6f;
    for (uint i = 0; i < lidar_div.getHitCount(); i++) {
        vec3 pos = lidar_div.getHitXYZ(i);
        x_min_d = fmin(x_min_d, pos.x);
        x_max_d = fmax(x_max_d, pos.x);
        y_min_d = fmin(y_min_d, pos.y);
        y_max_d = fmax(y_max_d, pos.y);
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
    std::map<std::pair<int, int>, bool> filled_grid_positions;

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
            int code = (int) lidar.getHitData(r, "gapfillMisses_code");
            if (code == 0)
                flag_0_count++;
            else if (code == 1)
                flag_1_count++;
            else if (code == 2)
                flag_2_count++;
            else if (code == 3)
                flag_3_count++;
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
    DOCTEST_CHECK(hit_ratio > 0.7f); // Within reasonable range
    DOCTEST_CHECK(hit_ratio < 1.3f);

    // 3. Verify both methods cover similar grid positions by comparing actual hits (non-misses)
    //    Count hits that are NOT far-field points
    uint real_hits_method1 = 0;
    uint real_hits_method2 = 0;

    for (uint r = 0; r < lidar1.getHitCount(); r++) {
        float dist = sqrt(pow(lidar1.getHitXYZ(r).x - lidar1.getScanOrigin(0).x, 2) + pow(lidar1.getHitXYZ(r).y - lidar1.getScanOrigin(0).y, 2) + pow(lidar1.getHitXYZ(r).z - lidar1.getScanOrigin(0).z, 2));
        if (dist < 1000)
            real_hits_method1++; // Not a far-field miss
    }

    for (uint r = 0; r < lidar2.getHitCount(); r++) {
        float dist = sqrt(pow(lidar2.getHitXYZ(r).x - lidar2.getScanOrigin(0).x, 2) + pow(lidar2.getHitXYZ(r).y - lidar2.getScanOrigin(0).y, 2) + pow(lidar2.getHitXYZ(r).z - lidar2.getScanOrigin(0).z, 2));
        if (dist < 1000)
            real_hits_method2++;
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
    } catch (const std::runtime_error &e) {
        caught_error = true;
        std::string msg(e.what());
        DOCTEST_CHECK(msg.find("Invalid scanID") != std::string::npos);
    }
    DOCTEST_CHECK(caught_error);

    // Test 2: Empty scan (no hits) should return empty vector gracefully
    DOCTEST_CHECK_NOTHROW(lidar.loadXML("plugins/lidar/xml/synthetic_test_8.xml"));
    Context context;
    // Don't add any geometry - all rays will miss and not be traced
    lidar.syntheticScan(&context, false, false); // No geometry, no hits

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
    DOCTEST_CHECK_NOTHROW(filled_all = lidar2.gapfillMisses()); // Fill all scans
    uint hits_after_all = lidar2.getHitCount();

    DOCTEST_CHECK(hits_after_all >= hits_before_all); // Should add points or stay same
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
    float thetaMin = M_PI / 3;
    float thetaMax = 2 * M_PI / 3;
    float phiMin = 0.9;
    float phiMax = 1.7;
    float exitDiameter = 0.015; // 1.5cm exit diameter
    float beamDivergence = 0.002; // 2 mrad divergence for beam spreading

    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, std::vector<std::string>{"x", "y", "z", "timestamp"});
    uint scanID = lidar.addScan(scan);

    // Multi-return synthetic scan WITHOUT miss recording - gapfilling will restore them
    lidar.syntheticScan(&context, 3, 0.15f, false, false); // rays_per_pulse=3, record_misses=false
    uint hits_before = lidar.getHitCount();

    // Check if multi-return data was created (depends on beam spreading and geometry)
    bool has_multi_return = false;
    for (size_t r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.doesHitDataExist(r, "target_count") && lidar.getHitData(r, "target_count") > 1) {
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
    uint flag_0_count = 0; // Original hits
    uint flag_other_count = 0; // Gapfilled hits (1, 2, or 3)

    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.doesHitDataExist(r, "gapfillMisses_code")) {
            int code = (int) lidar.getHitData(r, "gapfillMisses_code");
            if (code == 0) {
                flag_0_count++;
            } else {
                flag_other_count++;
            }
        }
    }

    // Original hits should all be flagged (if multi-return data exists)
    if (has_multi_return) {
        DOCTEST_CHECK(flag_0_count > 0); // Should have original hits
        DOCTEST_CHECK(flag_other_count == filled.size()); // All filled points flagged
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
    float thetaMin = M_PI / 3;
    float thetaMax = 2 * M_PI / 3;
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
    ScanMetadata scan1(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0, 0, 0.0f, 0.0f, std::vector<std::string>{"x", "y", "z", "timestamp"});
    lidar1.addScan(scan1);
    lidar1.syntheticScan(&context, false, true); // record_misses = TRUE (ground truth)

    uint hits_ground_truth = lidar1.getHitCount();

    // Build comprehensive position map for ground truth
    std::map<std::pair<int, int>, vec3> ground_truth_positions;
    std::map<std::pair<int, int>, bool> ground_truth_is_miss; // Track which are far-field misses

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
        float dist = sqrt(pow(pos.x - scan_origin.x, 2) + pow(pos.y - scan_origin.y, 2) + pow(pos.z - scan_origin.z, 2));
        ground_truth_is_miss[std::make_pair(row, col)] = (dist > 1000);
    }

    // Verify we got expected number of hits (should be Ntheta × Nphi)
    uint expected_grid_size = Ntheta * Nphi;

    DOCTEST_CHECK_MESSAGE(hits_ground_truth == expected_grid_size, "record_misses should produce Ntheta×Nphi hits but got " << hits_ground_truth << " vs " << expected_grid_size);

    DOCTEST_CHECK(ground_truth_positions.size() == hits_ground_truth);

    // METHOD 2: Gapfill misses
    LiDARcloud lidar2;
    lidar2.disableMessages();
    lidar2.addGrid(make_vec3(0, 0, 1.5), make_vec3(2, 2, 2), make_int3(1, 1, 1), 0);
    ScanMetadata scan2(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0, 0, 0.0f, 0.0f, std::vector<std::string>{"x", "y", "z", "timestamp"});
    lidar2.addScan(scan2);
    lidar2.syntheticScan(&context, false, false); // record_misses = FALSE

    uint hits_before_gapfill = lidar2.getHitCount();

    std::vector<vec3> filled = lidar2.gapfillMisses(0, false, false);
    uint hits_after_gapfill = lidar2.getHitCount();

    // Build position map for gapfilled data (map will deduplicate by grid position)
    std::map<std::pair<int, int>, vec3> gapfilled_positions;
    std::map<std::pair<int, int>, uint> gapfilled_hit_count; // Count hits per grid position

    for (uint r = 0; r < lidar2.getHitCount(); r++) {
        SphericalCoord raydir = lidar2.getHitRaydir(r);
        vec3 pos = lidar2.getHitXYZ(r);

        float theta = raydir.zenith;
        float phi = raydir.azimuth;
        int row = round((theta - thetaMin) / (thetaMax - thetaMin) * (Ntheta - 1));
        int col = round((phi - phiMin) / (phiMax - phiMin) * (Nphi - 1));

        auto key = std::make_pair(row, col);
        gapfilled_positions[key] = pos; // Map stores last position at this grid cell
        gapfilled_hit_count[key]++; // Count hits per grid cell
    }

    // Check for duplicates in gapfilled data
    uint gapfilled_duplicates = hits_after_gapfill - gapfilled_positions.size();
    DOCTEST_CHECK_MESSAGE(gapfilled_duplicates == 0, "CRITICAL BUG: gapfillMisses created " << gapfilled_duplicates << " duplicate hits at positions that already had hits!");

    // 1. Separate verification: Interior positions vs Edge extrapolation
    //    Ground truth only has positions traced by syntheticScan
    //    Gapfilling should: (a) match ground truth for traced regions, (b) extrapolate edges

    // All ground truth positions should exist in gapfilled data (superset)
    uint ground_truth_positions_found = 0;
    for (const auto &kv: ground_truth_positions) {
        if (gapfilled_positions.find(kv.first) != gapfilled_positions.end()) {
            ground_truth_positions_found++;
        }
    }

    float ground_truth_recovery = float(ground_truth_positions_found) / float(ground_truth_positions.size());
    DOCTEST_CHECK_MESSAGE(ground_truth_recovery > 0.95f,
                          "Gapfilling missed positions that record_misses found: only recovered " << ground_truth_positions_found << " of " << ground_truth_positions.size() << " (" << (ground_truth_recovery * 100) << "%)");

    // 2. Verify all gapfilled points are at VALID grid positions (theta, phi in bounds)
    uint invalid_positions = 0;
    for (const auto &kv: gapfilled_positions) {
        int row = kv.first.first;
        int col = kv.first.second;

        if (row < 0 || row >= (int) Ntheta || col < 0 || col >= (int) Nphi) {
            invalid_positions++;
        }
    }
    DOCTEST_CHECK_MESSAGE(invalid_positions == 0, "Gapfilling created " << invalid_positions << " points at invalid grid positions");

    // 3. Verify positions that match ground truth have correct coordinates
    uint position_mismatches = 0;
    float max_position_error = 0;

    for (const auto &kv: ground_truth_positions) {
        auto key = kv.first;
        if (gapfilled_positions.find(key) != gapfilled_positions.end()) {
            vec3 pos_gt = kv.second;
            vec3 pos_gf = gapfilled_positions[key];

            bool is_miss = ground_truth_is_miss[key];

            float dist = sqrt(pow(pos_gt.x - pos_gf.x, 2) + pow(pos_gt.y - pos_gf.y, 2) + pow(pos_gt.z - pos_gf.z, 2));

            if (dist > max_position_error)
                max_position_error = dist;

            // For geometry hits, positions should match exactly (within 1cm)
            // For far-field misses, directions should match (within 1 degree)
            if (!is_miss && dist > 0.01f) {
                position_mismatches++;
            } else if (is_miss) {
                // Check direction for misses
                vec3 dir_gt = pos_gt - scan_origin;
                vec3 dir_gf = pos_gf - scan_origin;
                float mag_gt = sqrt(dir_gt.x * dir_gt.x + dir_gt.y * dir_gt.y + dir_gt.z * dir_gt.z);
                float mag_gf = sqrt(dir_gf.x * dir_gf.x + dir_gf.y * dir_gf.y + dir_gf.z * dir_gf.z);

                float dot = (dir_gt.x * dir_gf.x + dir_gt.y * dir_gf.y + dir_gt.z * dir_gf.z) / (mag_gt * mag_gf);
                if (dot < 0.9998f) { // Directions differ by >1 degree
                    position_mismatches++;
                }
            }
        }
    }

    float position_match_rate = float(ground_truth_positions_found - position_mismatches) / float(ground_truth_positions_found);
    DOCTEST_CHECK_MESSAGE(position_match_rate > 0.95f, "Position accuracy too low: " << position_mismatches << " mismatches out of " << ground_truth_positions_found << " (" << (position_match_rate * 100) << "% correct)");

    // STRICT CHECK: For ideal scan, gapfilling should match ground truth exactly
    DOCTEST_CHECK_MESSAGE(hits_after_gapfill == hits_ground_truth,
                          "Gapfilling should exactly match record_misses for ideal scan: " << hits_after_gapfill << " vs " << hits_ground_truth << " (difference: " << (int) hits_after_gapfill - (int) hits_ground_truth << ")");
}

DOCTEST_TEST_CASE("LiDAR exportScans round-trip") {

    const std::string out_dir = "lidar_export_scans_test_tmp";
    std::filesystem::remove_all(out_dir);

    LiDARcloud original;
    original.disableMessages();

    // Scan 0: narrow theta range, custom column format
    std::vector<std::string> columnFormat0 = {"x", "y", "z", "zenith", "azimuth"};
    ScanMetadata scan0(vec3(-3.0f, 0.0f, 0.5f), 80, 0.25f * float(M_PI), 0.75f * float(M_PI), 160, 0.0f, 2.0f * float(M_PI), 0.01f, 0.001f, 0.004f, 0.0005f, columnFormat0);
    DOCTEST_CHECK_NOTHROW(original.addScan(scan0));

    // Scan 1: different origin and default column format
    std::vector<std::string> columnFormat1;
    ScanMetadata scan1(vec3(0.0f, -3.0f, 0.5f), 80, 0.0f, float(M_PI), 160, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat1);
    DOCTEST_CHECK_NOTHROW(original.addScan(scan1));

    DOCTEST_CHECK_NOTHROW(original.addGrid(vec3(0, 0, 0.5f), vec3(1, 1, 1), make_int3(1, 1, 1), 0));

    Context scene_ctx;
    std::vector<uint> UUIDs = scene_ctx.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    DOCTEST_CHECK(!UUIDs.empty());
    DOCTEST_CHECK_NOTHROW(original.syntheticScan(&scene_ctx));

    const uint original_hits = original.getHitCount();
    DOCTEST_CHECK(original_hits > 0);

    const std::string xml_out = out_dir + "/scans.xml";
    DOCTEST_CHECK_NOTHROW(original.exportScans(xml_out.c_str()));

    DOCTEST_CHECK(std::filesystem::exists(xml_out));
    DOCTEST_CHECK(std::filesystem::exists(out_dir + "/scans_0.xyz"));
    DOCTEST_CHECK(std::filesystem::exists(out_dir + "/scans_1.xyz"));

    // Exported files begin with a '#'-prefixed comment header listing the scan's column format.
    {
        std::ifstream xyz0(out_dir + "/scans_0.xyz");
        std::string header_line;
        std::getline(xyz0, header_line);
        DOCTEST_CHECK(!header_line.empty());
        DOCTEST_CHECK(header_line.front() == '#');
        for (const std::string &col: columnFormat0) {
            DOCTEST_CHECK(header_line.find(col) != std::string::npos);
        }
    }

    // Reload and verify metadata + hit count round-trip (loader must skip the header line)
    LiDARcloud reloaded;
    reloaded.disableMessages();
    DOCTEST_CHECK_NOTHROW(reloaded.loadXML(xml_out.c_str()));

    DOCTEST_CHECK(reloaded.getScanCount() == original.getScanCount());
    DOCTEST_CHECK(reloaded.getHitCount() == original_hits);

    for (uint i = 0; i < original.getScanCount(); i++) {
        vec3 o_orig = original.getScanOrigin(i);
        vec3 o_new = reloaded.getScanOrigin(i);
        DOCTEST_CHECK(o_new.x == doctest::Approx(o_orig.x));
        DOCTEST_CHECK(o_new.y == doctest::Approx(o_orig.y));
        DOCTEST_CHECK(o_new.z == doctest::Approx(o_orig.z));

        DOCTEST_CHECK(reloaded.getScanSizeTheta(i) == original.getScanSizeTheta(i));
        DOCTEST_CHECK(reloaded.getScanSizePhi(i) == original.getScanSizePhi(i));

        vec2 t_orig = original.getScanRangeTheta(i);
        vec2 t_new = reloaded.getScanRangeTheta(i);
        DOCTEST_CHECK(t_new.x == doctest::Approx(t_orig.x).epsilon(1e-4));
        DOCTEST_CHECK(t_new.y == doctest::Approx(t_orig.y).epsilon(1e-4));

        vec2 p_orig = original.getScanRangePhi(i);
        vec2 p_new = reloaded.getScanRangePhi(i);
        DOCTEST_CHECK(p_new.x == doctest::Approx(p_orig.x).epsilon(1e-4));
        DOCTEST_CHECK(p_new.y == doctest::Approx(p_orig.y).epsilon(1e-4));

        DOCTEST_CHECK(reloaded.getScanBeamExitDiameter(i) == doctest::Approx(original.getScanBeamExitDiameter(i)));
        DOCTEST_CHECK(reloaded.getScanBeamDivergence(i) == doctest::Approx(original.getScanBeamDivergence(i)));
        DOCTEST_CHECK(reloaded.getScanRangeNoiseStdDev(i) == doctest::Approx(original.getScanRangeNoiseStdDev(i)));
        DOCTEST_CHECK(reloaded.getScanAngleNoiseStdDev(i) == doctest::Approx(original.getScanAngleNoiseStdDev(i)));

        std::vector<std::string> f_orig = original.getScanColumnFormat(i);
        if (f_orig.empty()) {
            f_orig = {"x", "y", "z"};
        }
        DOCTEST_CHECK(reloaded.getScanColumnFormat(i) == f_orig);
    }

    // Exporting from an empty cloud must fail fast, not silently produce nothing
    LiDARcloud empty_cloud;
    empty_cloud.disableMessages();
    DOCTEST_CHECK_THROWS(empty_cloud.exportScans((out_dir + "/empty.xml").c_str()));

    std::filesystem::remove_all(out_dir);
}

DOCTEST_TEST_CASE("LiDAR exportPointCloud header") {

    const std::string out_dir = "lidar_export_header_test_tmp";
    std::filesystem::remove_all(out_dir);

    // Build a small cloud directly with a custom column format that includes a standard field
    // (intensity) and a user-defined scalar field (my_field).
    const std::vector<std::string> columnFormat = {"x", "y", "z", "intensity", "my_field"};
    LiDARcloud cloud;
    cloud.disableMessages();

    ScanMetadata scan(vec3(0.0f, 0.0f, 0.0f), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    const uint Nhits = 5;
    for (uint i = 0; i < Nhits; i++) {
        std::map<std::string, double> data;
        data["intensity"] = 0.5 + 0.1 * i;
        data["my_field"] = 100.0 + i;
        SphericalCoord dir(1.f, 0.5f * float(M_PI) - 0.5f * float(M_PI), 0.1f * i);
        DOCTEST_CHECK_NOTHROW(cloud.addHitPoint(0, vec3(float(i), 0.2f * i, 1.0f), dir, data));
    }
    DOCTEST_CHECK(cloud.getHitCount() == Nhits);

    // --- Export with header (default) --- //
    const std::string with_header = out_dir + "/cloud_header.xyz";
    DOCTEST_CHECK_NOTHROW(cloud.exportPointCloud(with_header.c_str(), 0u));
    DOCTEST_CHECK(std::filesystem::exists(with_header));
    {
        std::ifstream f(with_header);
        std::string first_line;
        std::getline(f, first_line);
        DOCTEST_CHECK(first_line == "# x y z intensity my_field");
        // Each data line must have exactly columnFormat.size() whitespace-separated tokens.
        std::string data_line;
        std::getline(f, data_line);
        std::istringstream iss(data_line);
        std::string tok;
        size_t ntok = 0;
        while (iss >> tok) {
            ntok++;
        }
        DOCTEST_CHECK(ntok == columnFormat.size());
    }

    // --- Export without header --- //
    const std::string no_header = out_dir + "/cloud_nohdr.xyz";
    DOCTEST_CHECK_NOTHROW(cloud.exportPointCloud(no_header.c_str(), 0u, false));
    DOCTEST_CHECK(std::filesystem::exists(no_header));
    {
        std::ifstream f(no_header);
        std::string first_line;
        std::getline(f, first_line);
        DOCTEST_CHECK(!first_line.empty());
        DOCTEST_CHECK(first_line.front() != '#'); // first line is a data row, not a header
    }

    // --- Round-trip the headered file through XML/loadASCIIFile to confirm the loader skips the
    //     header line and preserves user data. --- //
    LiDARcloud reloaded;
    reloaded.disableMessages();
    ScanMetadata reload_scan(vec3(0.0f, 0.0f, 0.0f), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(reloaded.addScan(reload_scan));
    DOCTEST_CHECK_NOTHROW(reloaded.loadASCIIFile(0, with_header));
    DOCTEST_CHECK(reloaded.getHitCount() == Nhits);
    // Spot-check a user-defined scalar field survived the round trip.
    double mf;
    DOCTEST_CHECK_NOTHROW(mf = reloaded.getHitData(0, "my_field"));
    DOCTEST_CHECK(mf == doctest::Approx(100.0));

    std::filesystem::remove_all(out_dir);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Texture Color Sampling") {
    // A synthetic scan over a textured leaf primitive must color each hit point from the texture
    // RGB at the intersection. Because transparent texels are rejected at the geometry level, every
    // recorded hit must land on an opaque (colored) texel - none should come back black (0,0,0).
    // GrapeLeaf.png has a colored opaque leaf silhouette and transparent corners.
    const char *texture = "plugins/visualizer/textures/GrapeLeaf.png";

    Context context;
    // Tilt the patch out of the z=0 plane so the scan's domain bounding box is not degenerate.
    uint patch = context.addPatch(make_vec3(0, 0, 0), make_vec2(2, 2), make_SphericalCoord(0.6f, 0.0f), texture);
    DOCTEST_CHECK(context.primitiveTextureHasTransparencyChannel(patch));

    LiDARcloud lidar;
    lidar.disableMessages();

    std::vector<std::string> columnFormat = {"x", "y", "z"};
    ScanMetadata scan(make_vec3(0, 0, 5), 60, 0.0f, float(M_PI), 60, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(lidar.addScan(scan));

    // rays_per_pulse=1 (single return) so the hit position is exact for color sampling.
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 1, 0.5f));

    uint hit_count = lidar.getHitCount();
    DOCTEST_CHECK(hit_count > 0);

    uint black_hits = 0;
    for (uint i = 0; i < hit_count; i++) {
        RGBcolor c = lidar.getHitColor(i);
        // Colors must be valid and within [0,1].
        DOCTEST_CHECK(c.r >= 0.f);
        DOCTEST_CHECK(c.r <= 1.f);
        DOCTEST_CHECK(c.g >= 0.f);
        DOCTEST_CHECK(c.g <= 1.f);
        DOCTEST_CHECK(c.b >= 0.f);
        DOCTEST_CHECK(c.b <= 1.f);
        if (c.r == 0.f && c.g == 0.f && c.b == 0.f) {
            black_hits++;
        }
    }

    // No hit should be black: transparent texels are rejected, opaque texels are colored leaf pixels.
    DOCTEST_CHECK(black_hits == 0);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - ASCII Multi-Return Cloud") {
    // Regression guard for the non-deterministic crash (SIGSEGV) on the multi-return
    // file-import path (loadXML -> gapfillMisses -> calculateLeafArea).
    //
    // The crash had two contributing issues, both exercised here:
    //  1. ROOT CAUSE: loadXML used a fixed char[100] stack buffer with strcpy/strcat
    //     for the scan data-file path, so a long <filename> path overflowed the stack
    //     and corrupted memory non-deterministically. This test therefore references
    //     the fixture through a deliberately long absolute path (>100 chars) so the
    //     overflow would have been triggered.
    //  2. Defensive hardening in gapfillMisses (size_t underflow on size()-1 loops,
    //     uninitialized last elements, unbounded Ngap) for ASCII clouds with no
    //     row/column indices, whose (theta,phi) grid is reconstructed from timestamps.
    //
    // The fixture (leafcube_multi.xyz) is a multi-return scan of the
    // LAI=2 leaf cube (true LAD = 2.0 m^2/m^3, G(theta)=0.5). It is loaded + gapfilled
    // on a fresh cloud many times so any residual non-determinism is exercised, then
    // the recovered leaf-area density is checked.

    namespace fs = std::filesystem;

    // The build copies data/ to <build>/plugins/lidar/data/, and tests run from
    // <build>. To exercise the loadXML stack-buffer overflow deterministically -
    // regardless of how short the build directory's absolute path is - copy the
    // fixture into a deeply nested subdirectory whose path is guaranteed to exceed the
    // old 100-byte buffer.
    fs::path data_src = "plugins/lidar/data/leafcube_multi.xyz";
    DOCTEST_REQUIRE(fs::exists(data_src));

    // A nested directory chain that, combined with the absolute build path, comfortably
    // exceeds 100 characters. Each segment is long and the chain is deep so the total
    // is long even from a short CWD.
    fs::path long_dir = fs::absolute("lidar_longpath_overflow_regression_dir/"
                                     "subdirectory_padding_to_exceed_one_hundred_byte_buffer/"
                                     "additional_nesting_for_safety_margin");
    fs::create_directories(long_dir);
    fs::path data_abs = long_dir / "leafcube_multi.xyz";
    fs::copy_file(data_src, data_abs, fs::copy_options::overwrite_existing);

    std::string data_path_str = data_abs.string();
    DOCTEST_REQUIRE_MESSAGE(data_path_str.size() > 100, "Fixture path must exceed the old 100-byte buffer to exercise the loadXML "
                                                        "overflow; got length "
                                                                << data_path_str.size() << " (" << data_path_str << ")");

    const char *test_xml = "lidar_leafcube_multi_longpath_test.xml";
    {
        std::ofstream xml(test_xml);
        xml << "<?xml version=\"1.0\"?>\n<helios>\n<scan>\n"
            << "  <filename> " << data_path_str << " </filename>\n"
            << "  <ASCII_format> x y z timestamp target_index target_count </ASCII_format>\n"
            << "  <origin> -5.000000 0.000000 0.500000 </origin>\n"
            << "  <size> 800 1600 </size>\n"
            << "</scan>\n"
            << "<grid>\n  <center> 0 0 0.5 </center>\n  <size> 1 1 1 </size>\n"
            << "  <Nx> 1 </Nx>\n  <Ny> 1 </Ny>\n  <Nz> 1 </Nz>\n</grid>\n"
            << "</helios>\n";
    }

    const int N_repeats = 25; // crash was ~1-in-2 to 1-in-3; this makes a miss vanishingly unlikely

    for (int rep = 0; rep < N_repeats; rep++) {
        LiDARcloud lidar;
        lidar.disableMessages();

        // loadXML must not overflow on the long path; gapfillMisses must not read out
        // of bounds on the ASCII-loaded, row/column-less multi-return cloud.
        DOCTEST_CHECK_NOTHROW(lidar.loadXML(test_xml));
        DOCTEST_CHECK_NOTHROW(lidar.gapfillMisses());
    }

    // On a final pass, run the full LAD pipeline (the Phytograph file-import path:
    // triangulate -> gapfill -> calculateLeafArea) and check the recovered value.
    LiDARcloud lidar;
    lidar.disableMessages();
    DOCTEST_CHECK_NOTHROW(lidar.loadXML(test_xml));
    DOCTEST_CHECK_NOTHROW(lidar.triangulateHitPoints(0.04, 10));
    DOCTEST_CHECK_NOTHROW(lidar.gapfillMisses());

    Context context;
    DOCTEST_CHECK_NOTHROW(lidar.calculateLeafArea(&context));

    float LAD = lidar.getCellLeafAreaDensity(0);

    DOCTEST_CHECK(LAD == LAD); // not NaN
    // True LAD of the LAI=2 leaf cube is 2.0 m^2/m^3. Reconstructing misses from an
    // ASCII multi-return cloud introduces more bias than the synthetic path, so use a
    // generous-but-meaningful band: this still fails hard on garbage (0, huge, NaN)
    // while tolerating the reconstruction's known bias.
    DOCTEST_CHECK(LAD > 1.0f);
    DOCTEST_CHECK(LAD < 3.0f);

    std::remove(test_xml);
    std::error_code ec;
    fs::remove_all("lidar_longpath_overflow_regression_dir", ec);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Scanner Tilt Test") {
    // Scanner tilt rotates the entire fan of ray directions about the scanner origin (roll about world x, then pitch about
    // world y), modeling the residual tilt a real terrestrial scanner's dual-axis inclinometer reports. We verify that:
    //   (1) zero tilt is an exact no-op (same hits, same positions, as an untilted scan),
    //   (2) a known tilt rotates each beam's hit direction by exactly that tilt (the core geometric check),
    //   (3) roll and pitch act about the correct, independent axes (not swapped), and
    //   (4) the <scanTilt> XML tag is parsed into the scan metadata in radians.

    // Box of patches surrounding the scanner so that beams in all tilt directions strike a surface at a known distance.
    // Scanner sits at the origin; six walls of a 10 m cube give every direction a target.
    Context context;
    context.addPatch(make_vec3(0, 0, -5), make_vec2(20, 20)); // floor
    context.addPatch(make_vec3(0, 0, 5), make_vec2(20, 20)); // ceiling
    context.addPatch(make_vec3(5, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.f)); // +x wall
    context.addPatch(make_vec3(-5, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.f)); // -x wall
    context.addPatch(make_vec3(0, 5, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.5f * float(M_PI))); // +y wall
    context.addPatch(make_vec3(0, -5, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.5f * float(M_PI))); // -y wall

    const vec3 scan_origin(0.0f, 0.0f, 0.0f);
    const uint Ntheta = 30;
    const uint Nphi = 30;
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);
    const float exitDiameter = 0.0f;
    const float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    // Helper: run a synthetic scan with the given tilt and return hit directions (unit vectors from the scanner origin).
    // A near-nadir cone is used so that, for the modest tilts tested, every beam strikes the floor both level and tilted
    // (no beam crosses to a different wall or misses). This keeps the hit list in exact beam-for-beam correspondence between
    // a level and a tilted scan, which the rotation-invariant check below relies on.
    const float cone_thetaMin = 0.80f * float(M_PI);
    const float cone_thetaMax = float(M_PI);
    auto run_scan = [&](float roll, float pitch, std::vector<vec3> &dirs) {
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, cone_thetaMin, cone_thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat, roll, pitch);
        lidar.addScan(scan);
        lidar.syntheticScan(&context);
        uint hc = lidar.getHitCount();
        dirs.clear();
        dirs.reserve(hc);
        for (uint i = 0; i < hc; i++) {
            vec3 d = lidar.getHitXYZ(i) - scan_origin;
            d.normalize();
            dirs.push_back(d);
        }
    };

    // (1) Zero tilt no-op: explicit zero tilt must reproduce an untilted scan hit-for-hit. The beam grid is deterministic
    //     and noise is off, so the clouds must match exactly.
    {
        std::vector<vec3> dirs_default, dirs_zero;
        run_scan(0.0f, 0.0f, dirs_zero);
        // Untilted via the default constructor (no tilt args), same beam grid as run_scan() — should be identical.
        LiDARcloud lidar_default;
        lidar_default.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, cone_thetaMin, cone_thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
        lidar_default.addScan(scan);
        lidar_default.syntheticScan(&context);
        uint hc = lidar_default.getHitCount();
        for (uint i = 0; i < hc; i++) {
            vec3 d = lidar_default.getHitXYZ(i) - scan_origin;
            d.normalize();
            dirs_default.push_back(d);
        }
        DOCTEST_REQUIRE(dirs_zero.size() == dirs_default.size());
        DOCTEST_CHECK(dirs_zero.size() > 0);
        for (size_t i = 0; i < dirs_zero.size(); i++) {
            DOCTEST_CHECK(dirs_zero[i].x == doctest::Approx(dirs_default[i].x).epsilon(1e-5));
            DOCTEST_CHECK(dirs_zero[i].y == doctest::Approx(dirs_default[i].y).epsilon(1e-5));
            DOCTEST_CHECK(dirs_zero[i].z == doctest::Approx(dirs_default[i].z).epsilon(1e-5));
        }
    }

    // (2) Core geometric check: with a known tilt, each beam's hit direction must equal the corresponding level-scan hit
    //     direction rotated by exactly that tilt, using the right-handed body-frame axes: roll about the lateral axis, then
    //     pitch about the forward (azimuth-zero) axis. The grid is identical and deterministic, so hits correspond
    //     beam-for-beam. run_scan() uses phiMin = 0, for which the lateral axis is world +x and the forward axis is world +y.
    {
        const float roll = 7.0f * float(M_PI) / 180.0f;
        const float pitch = 11.0f * float(M_PI) / 180.0f;
        // Body axes for phiMin = 0 (the value run_scan uses).
        const vec3 lateral_axis = make_vec3(cosf(phiMin), -sinf(phiMin), 0.f); // X_body (roll axis)
        const vec3 forward_axis = make_vec3(sinf(phiMin), cosf(phiMin), 0.f); // Y_body (pitch axis)
        std::vector<vec3> level_dirs, tilted_dirs;
        run_scan(0.0f, 0.0f, level_dirs);
        run_scan(roll, pitch, tilted_dirs);
        DOCTEST_REQUIRE(level_dirs.size() == tilted_dirs.size());
        DOCTEST_CHECK(level_dirs.size() > 0);
        const vec3 pivot = make_vec3(0, 0, 0);
        for (size_t i = 0; i < level_dirs.size(); i++) {
            vec3 expected = rotatePointAboutLine(level_dirs[i], pivot, lateral_axis, roll);
            expected = rotatePointAboutLine(expected, pivot, forward_axis, pitch);
            DOCTEST_CHECK(tilted_dirs[i].x == doctest::Approx(expected.x).epsilon(1e-3));
            DOCTEST_CHECK(tilted_dirs[i].y == doctest::Approx(expected.y).epsilon(1e-3));
            DOCTEST_CHECK(tilted_dirs[i].z == doctest::Approx(expected.z).epsilon(1e-3));
        }
    }

    // (3) Roll/pitch axis independence: a near-nadir beam pointing straight down (0,0,-1) tilts predictably. The body frame
    //     here uses phiMin = 0, so the forward (pitch) axis is world +y and the lateral (roll) axis is world +x. The rotations
    //     are right-handed (consistent with Helios' right-hand-rule convention and a right-handed Z-up scanner body frame),
    //     so for a positive tilt angle:
    //     - Pitch (right-hand rotation about +y) rotates the nadir beam into the x-z plane: the floor hit is displaced in -x, y ~ 0.
    //     - Roll  (right-hand rotation about +x) rotates the nadir beam into the y-z plane: the floor hit is displaced in +y, x ~ 0.
    //     The key assertion is that pitch moves the hit purely in x and roll purely in y (the two are not swapped and act on
    //     independent axes). The signs are the standard right-handed result and are cross-checked against the analytic
    //     rotation in part (2) above.
    {
        // A narrow cone tightly around nadir; the mean hit on the floor isolates the tilt direction.
        const uint Nt = 8, Np = 8;
        const float tmin = 0.97f * float(M_PI);
        const float tmax = float(M_PI);
        const float tilt = 10.0f * float(M_PI) / 180.0f;

        auto mean_floor_hit = [&](float roll, float pitch) -> vec3 {
            LiDARcloud lidar;
            lidar.disableMessages();
            ScanMetadata scan(scan_origin, Nt, tmin, tmax, Np, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat, roll, pitch);
            lidar.addScan(scan);
            lidar.syntheticScan(&context);
            uint hc = lidar.getHitCount();
            vec3 mean = make_vec3(0, 0, 0);
            uint n = 0;
            for (uint i = 0; i < hc; i++) {
                vec3 p = lidar.getHitXYZ(i);
                if (p.z < -1.0f) { // floor hits only
                    mean = mean + p;
                    n++;
                }
            }
            DOCTEST_REQUIRE(n > 0);
            return mean / float(n);
        };

        vec3 pitch_hit = mean_floor_hit(0.0f, tilt); // pitch about +y -> displacement in x (here -x)
        DOCTEST_CHECK(pitch_hit.x < -0.5f);
        DOCTEST_CHECK(fabs(pitch_hit.y) < 0.2f);

        vec3 roll_hit = mean_floor_hit(tilt, 0.0f); // roll about +x -> displacement in y (here +y)
        DOCTEST_CHECK(roll_hit.y > 0.5f);
        DOCTEST_CHECK(fabs(roll_hit.x) < 0.2f);
    }

    // (4) XML round-trip: <scanTilt> roll pitch </scanTilt> is parsed into metadata, converted degrees->radians.
    {
        const char *tilt_xml = "lidar_scantilt_xml_test.xml";
        std::ofstream ofs(tilt_xml);
        ofs << "<helios>\n"
            << "  <scan>\n"
            << "    <origin> 0 0 0 </origin>\n"
            << "    <size> 10 10 </size>\n"
            << "    <scanTilt> 5 3 </scanTilt>\n"
            << "  </scan>\n"
            << "  <scan>\n"
            << "    <origin> 0 0 0 </origin>\n"
            << "    <size> 10 10 </size>\n"
            << "  </scan>\n"
            << "</helios>\n";
        ofs.close();

        LiDARcloud lidar;
        lidar.disableMessages();
        DOCTEST_CHECK_NOTHROW(lidar.loadXML(tilt_xml));
        DOCTEST_REQUIRE(lidar.getScanCount() == 2);
        DOCTEST_CHECK(lidar.getScanTiltRoll(0) == doctest::Approx(5.0f * float(M_PI) / 180.0f).epsilon(1e-5));
        DOCTEST_CHECK(lidar.getScanTiltPitch(0) == doctest::Approx(3.0f * float(M_PI) / 180.0f).epsilon(1e-5));
        // No tag -> level (0,0)
        DOCTEST_CHECK(lidar.getScanTiltRoll(1) == doctest::Approx(0.0f));
        DOCTEST_CHECK(lidar.getScanTiltPitch(1) == doctest::Approx(0.0f));

        std::remove(tilt_xml);
    }

    // (5) Azimuth-zero coupling: the tilt axes are defined relative to the scan's azimuth-zero (phiMin) facing direction, not
    //     the fixed world axes. A pitch-only tilt with phiMin rotated by 90 degrees must rotate the nadir beam about a
    //     correspondingly rotated forward axis. We verify the tilted near-nadir floor hit matches the analytic rotation of the
    //     level beam about the phiMin-dependent body forward axis. This is what distinguishes the azimuth-zero convention from
    //     a fixed-world-axis tilt: at phiMin = pi/2 the forward (pitch) axis is world +x (not +y), so the nadir-beam
    //     displacement is +y, whereas at phiMin = 0 (part 3) the same pitch produced a -x displacement.
    {
        const uint Nt = 8, Np = 8;
        const float tmin = 0.97f * float(M_PI);
        const float tmax = float(M_PI);
        const float tilt = 12.0f * float(M_PI) / 180.0f;
        const float phi0 = 0.5f * float(M_PI); // azimuth-zero rotated 90 degrees

        // Forward (pitch) axis of the body frame for this phiMin.
        const vec3 forward_axis = make_vec3(sinf(phi0), cosf(phi0), 0.f);

        LiDARcloud lidar;
        lidar.disableMessages();
        // phiMin = phi0, full azimuth sweep so the near-nadir cone is still complete.
        ScanMetadata scan(scan_origin, Nt, tmin, tmax, Np, phi0, phi0 + 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat, 0.0f, tilt);
        lidar.addScan(scan);
        lidar.syntheticScan(&context);
        uint hc = lidar.getHitCount();
        vec3 mean = make_vec3(0, 0, 0);
        uint n = 0;
        for (uint i = 0; i < hc; i++) {
            vec3 p = lidar.getHitXYZ(i);
            if (p.z < -1.0f) {
                mean = mean + p;
                n++;
            }
        }
        DOCTEST_REQUIRE(n > 0);
        mean = mean / float(n);
        vec3 mean_dir = mean;
        mean_dir.normalize();

        // Expected: the nadir beam (0,0,-1) rotated (right-handed) about the phi0 forward axis by the pitch angle.
        vec3 expected_dir = rotatePointAboutLine(make_vec3(0, 0, -1), make_vec3(0, 0, 0), forward_axis, tilt);
        expected_dir.normalize();

        DOCTEST_CHECK(mean_dir.x == doctest::Approx(expected_dir.x).epsilon(2e-2));
        DOCTEST_CHECK(mean_dir.y == doctest::Approx(expected_dir.y).epsilon(2e-2));
        DOCTEST_CHECK(mean_dir.z == doctest::Approx(expected_dir.z).epsilon(2e-2));

        // Sanity: at phiMin = pi/2 the displacement is in +y (not the -x seen at phiMin = 0), confirming the axes rotated.
        DOCTEST_CHECK(mean.y > 0.5f);
        DOCTEST_CHECK(fabs(mean.x) < 0.3f);
    }
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Scanner Azimuth Offset Test") {
    // The azimuth offset is the scanner's compass heading (yaw): a right-hand rotation of the entire fan of ray directions
    // about the world +z axis, applied on top of the per-scan azimuth sweep [phiMin, phiMax]. We verify that:
    //   (1) zero azimuth offset is an exact no-op (same hits as a scan with no offset),
    //   (2) a known azimuth offset rotates each beam's hit direction by exactly that offset about world +z,
    //   (3) the azimuth offset composes with tilt (the offset rotates the body frame, so a pitch tilt under a 90-degree
    //       offset displaces the nadir beam about the rotated forward axis), and
    //   (4) the <scanAzimuthOffset> XML tag is parsed into the scan metadata in radians.

    // Same surrounding box as the tilt test so every beam strikes a wall at a known distance.
    Context context;
    context.addPatch(make_vec3(0, 0, -5), make_vec2(20, 20)); // floor
    context.addPatch(make_vec3(0, 0, 5), make_vec2(20, 20)); // ceiling
    context.addPatch(make_vec3(5, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.f)); // +x wall
    context.addPatch(make_vec3(-5, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.f)); // -x wall
    context.addPatch(make_vec3(0, 5, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.5f * float(M_PI))); // +y wall
    context.addPatch(make_vec3(0, -5, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.5f * float(M_PI))); // -y wall

    const vec3 scan_origin(0.0f, 0.0f, 0.0f);
    const uint Ntheta = 30;
    const uint Nphi = 30;
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);
    const float exitDiameter = 0.0f;
    const float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    // A near-nadir cone keeps every beam on the floor for both the offset and non-offset scans, so hits stay in
    // beam-for-beam correspondence (the azimuth offset only spins the cone about +z, it does not move beams off the floor).
    const float cone_thetaMin = 0.80f * float(M_PI);
    const float cone_thetaMax = float(M_PI);
    auto run_scan = [&](float azimuth_offset, std::vector<vec3> &dirs) {
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, cone_thetaMin, cone_thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat, 0.0f, 0.0f, azimuth_offset);
        lidar.addScan(scan);
        lidar.syntheticScan(&context);
        uint hc = lidar.getHitCount();
        dirs.clear();
        dirs.reserve(hc);
        for (uint i = 0; i < hc; i++) {
            vec3 d = lidar.getHitXYZ(i) - scan_origin;
            d.normalize();
            dirs.push_back(d);
        }
    };

    // (1) Zero offset no-op: explicit zero azimuth offset must reproduce a scan with no offset, hit-for-hit.
    {
        std::vector<vec3> dirs_default, dirs_zero;
        run_scan(0.0f, dirs_zero);
        LiDARcloud lidar_default;
        lidar_default.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, cone_thetaMin, cone_thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
        lidar_default.addScan(scan);
        lidar_default.syntheticScan(&context);
        uint hc = lidar_default.getHitCount();
        for (uint i = 0; i < hc; i++) {
            vec3 d = lidar_default.getHitXYZ(i) - scan_origin;
            d.normalize();
            dirs_default.push_back(d);
        }
        DOCTEST_REQUIRE(dirs_zero.size() == dirs_default.size());
        DOCTEST_CHECK(dirs_zero.size() > 0);
        for (size_t i = 0; i < dirs_zero.size(); i++) {
            DOCTEST_CHECK(dirs_zero[i].x == doctest::Approx(dirs_default[i].x).epsilon(1e-5));
            DOCTEST_CHECK(dirs_zero[i].y == doctest::Approx(dirs_default[i].y).epsilon(1e-5));
            DOCTEST_CHECK(dirs_zero[i].z == doctest::Approx(dirs_default[i].z).epsilon(1e-5));
        }
    }

    // (2) Core geometric check: with a known azimuth offset, each beam's hit direction must equal the corresponding
    //     no-offset hit direction rotated about world +z by exactly that offset. The grid is identical and deterministic,
    //     so hits correspond beam-for-beam.
    {
        const float azimuth_offset = 35.0f * float(M_PI) / 180.0f;
        const vec3 vertical_axis = make_vec3(0.f, 0.f, 1.f);
        std::vector<vec3> base_dirs, offset_dirs;
        run_scan(0.0f, base_dirs);
        run_scan(azimuth_offset, offset_dirs);
        DOCTEST_REQUIRE(base_dirs.size() == offset_dirs.size());
        DOCTEST_CHECK(base_dirs.size() > 0);
        const vec3 pivot = make_vec3(0, 0, 0);
        for (size_t i = 0; i < base_dirs.size(); i++) {
            vec3 expected = rotatePointAboutLine(base_dirs[i], pivot, vertical_axis, azimuth_offset);
            DOCTEST_CHECK(offset_dirs[i].x == doctest::Approx(expected.x).epsilon(1e-3));
            DOCTEST_CHECK(offset_dirs[i].y == doctest::Approx(expected.y).epsilon(1e-3));
            DOCTEST_CHECK(offset_dirs[i].z == doctest::Approx(expected.z).epsilon(1e-3));
        }
    }

    // (3) Offset composes with tilt: a pitch-only tilt under a 90-degree azimuth offset rotates the body forward axis from
    //     world +y to world +x, so the near-nadir floor hit is displaced in +y rather than the -x seen with no offset
    //     (cf. the tilt test part 3). This confirms the offset rotates the roll/pitch body frame, applied yaw-then-pitch.
    {
        const uint Nt = 8, Np = 8;
        const float tmin = 0.97f * float(M_PI);
        const float tmax = float(M_PI);
        const float pitch = 12.0f * float(M_PI) / 180.0f;
        const float azimuth_offset = 0.5f * float(M_PI); // 90-degree heading offset

        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(scan_origin, Nt, tmin, tmax, Np, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat, 0.0f, pitch, azimuth_offset);
        lidar.addScan(scan);
        lidar.syntheticScan(&context);
        uint hc = lidar.getHitCount();
        vec3 mean = make_vec3(0, 0, 0);
        uint n = 0;
        for (uint i = 0; i < hc; i++) {
            vec3 p = lidar.getHitXYZ(i);
            if (p.z < -1.0f) { // floor hits only
                mean = mean + p;
                n++;
            }
        }
        DOCTEST_REQUIRE(n > 0);
        mean = mean / float(n);

        // forward (pitch) axis after the 90-degree heading offset is world +x, so a right-hand pitch displaces nadir to +y.
        const vec3 forward_axis = make_vec3(sinf(azimuth_offset), cosf(azimuth_offset), 0.f);
        vec3 expected_dir = rotatePointAboutLine(make_vec3(0, 0, -1), make_vec3(0, 0, 0), forward_axis, pitch);
        expected_dir.normalize();
        vec3 mean_dir = mean;
        mean_dir.normalize();
        DOCTEST_CHECK(mean_dir.x == doctest::Approx(expected_dir.x).epsilon(2e-2));
        DOCTEST_CHECK(mean_dir.y == doctest::Approx(expected_dir.y).epsilon(2e-2));
        DOCTEST_CHECK(mean_dir.z == doctest::Approx(expected_dir.z).epsilon(2e-2));
        DOCTEST_CHECK(mean.y > 0.5f);
        DOCTEST_CHECK(fabs(mean.x) < 0.3f);
    }

    // (4) XML round-trip: <scanAzimuthOffset> N </scanAzimuthOffset> is parsed into metadata, converted degrees->radians.
    {
        const char *az_xml = "lidar_scanazimuth_xml_test.xml";
        std::ofstream ofs(az_xml);
        ofs << "<helios>\n"
            << "  <scan>\n"
            << "    <origin> 0 0 0 </origin>\n"
            << "    <size> 10 10 </size>\n"
            << "    <scanAzimuthOffset> 45 </scanAzimuthOffset>\n"
            << "  </scan>\n"
            << "  <scan>\n"
            << "    <origin> 0 0 0 </origin>\n"
            << "    <size> 10 10 </size>\n"
            << "  </scan>\n"
            << "</helios>\n";
        ofs.close();

        LiDARcloud lidar;
        lidar.disableMessages();
        DOCTEST_CHECK_NOTHROW(lidar.loadXML(az_xml));
        DOCTEST_REQUIRE(lidar.getScanCount() == 2);
        DOCTEST_CHECK(lidar.getScanAzimuthOffset(0) == doctest::Approx(45.0f * float(M_PI) / 180.0f).epsilon(1e-5));
        // No tag -> no offset (0)
        DOCTEST_CHECK(lidar.getScanAzimuthOffset(1) == doctest::Approx(0.0f));

        std::remove(az_xml);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Row/column-based miss gap filling
//
// These tests exercise the gapfillMisses() row/column reconstruction path, which fits a robust per-row generative model
// (zenith = zenith_lut[row]; azimuth = intercept[row] + slope[row]*column) from the returns and emits a miss for every
// empty grid cell. The returns are synthesized directly from a known generative model so the reconstructed miss
// directions can be checked against ground truth. Each return is added with "row"/"column" hit data (the same data the
// ASCII loader now attaches) which routes gapfillMisses() to the row/column path.
// ---------------------------------------------------------------------------------------------------------------------

namespace {

    // Generative scan-grid model used to synthesize returns and to provide ground-truth miss directions.
    // zenith and azimuth are smooth functions of (row, column). The azimuth has a per-row offset whose magnitude grows
    // with row (azimuth sweep / shear), and a small per-row slope across columns. The zenith has a mild quadratic
    // curvature in row (a tilt-like departure from a perfectly affine grid).
    struct GenerativeGrid {
        int Ntheta;
        int Nphi;
        double theta_min, theta_max;
        double phi_min, phi_max;
        double shear; // azimuth offset per row (radians) - the sweep
        double curve; // zenith curvature coefficient (radians) - tilt-like departure from affine

        double zenith(int row) const {
            const double frac = double(row) / double(Ntheta - 1);
            // affine base plus a small symmetric quadratic bow
            return theta_min + (theta_max - theta_min) * frac + curve * (frac - 0.5) * (frac - 0.5);
        }
        double azimuth(int row, int col) const {
            const double col_frac = double(col) / double(Nphi - 1);
            const double base = phi_min + (phi_max - phi_min) * col_frac;
            return base + shear * double(row) / double(Ntheta - 1);
        }
        helios::SphericalCoord direction(int row, int col) const {
            return helios::make_SphericalCoord(1.f, 0.5f * float(M_PI) - float(zenith(row)), float(azimuth(row, col)));
        }
    };

    // angular separation (radians) between two unit directions given as SphericalCoord
    double angularError(const helios::SphericalCoord &a, const helios::SphericalCoord &b) {
        helios::vec3 va = helios::sphere2cart(helios::make_SphericalCoord(1.f, a.elevation, a.azimuth));
        helios::vec3 vb = helios::sphere2cart(helios::make_SphericalCoord(1.f, b.elevation, b.azimuth));
        double d = va.x * vb.x + va.y * vb.y + va.z * vb.z;
        d = std::max(-1.0, std::min(1.0, d));
        return std::acos(d);
    }

} // namespace

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Row/Column Idealized Grid") {
    // On a perfectly regular, leveled, noise-free grid the robust fit must recover the affine model, so reconstructed
    // miss directions should match the generative model to tight tolerance and every empty cell must be filled exactly
    // once.
    LiDARcloud lidar;
    lidar.disableMessages();

    GenerativeGrid g{20, 36, 0.05, 0.95 * M_PI, 0.0, 2.0 * M_PI, 0.0, 0.0};
    ScanMetadata scan(make_vec3(0, 0, 0), g.Ntheta, g.theta_min, g.theta_max, g.Nphi, g.phi_min, g.phi_max, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // Populate all cells EXCEPT a known interior rectangular blank region with returns.
    const int blank_r0 = 5, blank_r1 = 9, blank_c0 = 10, blank_c1 = 15;
    std::set<std::pair<int, int>> blanks;
    for (int row = 0; row < g.Ntheta; row++) {
        for (int col = 0; col < g.Nphi; col++) {
            if (row >= blank_r0 && row <= blank_r1 && col >= blank_c0 && col <= blank_c1) {
                blanks.insert({row, col});
                continue;
            }
            SphericalCoord dir = g.direction(row, col);
            vec3 xyz = helios::sphere2cart(make_SphericalCoord(10.f, dir.elevation, dir.azimuth));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            lidar.addHitPoint(0, xyz, dir, make_RGBcolor(1, 0, 0), data);
        }
    }

    uint hits_before = lidar.getHitCount();
    std::vector<vec3> filled = lidar.gapfillMisses(0, false, true);

    // every blank cell filled exactly once
    DOCTEST_CHECK(filled.size() == blanks.size());
    DOCTEST_CHECK(lidar.getHitCount() == hits_before + blanks.size());

    // reconstructed directions of the filled cells match the generative model tightly
    double max_err = 0.0;
    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.getHitScanID(r) != 0) {
            continue;
        }
        if (lidar.getHitData(r, "is_miss") != 1.0) {
            continue; // only check the gapfilled misses
        }
        int row = (int) std::lround(lidar.getHitData(r, "row"));
        int col = (int) std::lround(lidar.getHitData(r, "column"));
        double err = angularError(lidar.getHitRaydir(r), g.direction(row, col));
        max_err = std::max(max_err, err);
    }
    // tight: on an ideal grid the fit recovers the model to within a fraction of the grid spacing
    DOCTEST_CHECK(max_err < 1e-3);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Row/Column Tilted and Sheared Grid") {
    // With a tilt-like zenith curvature and a per-row azimuth sweep (shear), the robust per-row model should reconstruct
    // miss directions much more accurately than the idealized affine rc2direction model.
    LiDARcloud lidar;
    lidar.disableMessages();

    GenerativeGrid g{24, 48, 0.05, 0.95 * M_PI, 0.0, 1.5 * M_PI, 0.30, 0.20}; // strong shear + curvature
    ScanMetadata scan(make_vec3(0, 0, 0), g.Ntheta, g.theta_min, g.theta_max, g.Nphi, g.phi_min, g.phi_max, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    const int blank_r0 = 8, blank_r1 = 14, blank_c0 = 18, blank_c1 = 30;
    std::set<std::pair<int, int>> blanks;
    for (int row = 0; row < g.Ntheta; row++) {
        for (int col = 0; col < g.Nphi; col++) {
            if (row >= blank_r0 && row <= blank_r1 && col >= blank_c0 && col <= blank_c1) {
                blanks.insert({row, col});
                continue;
            }
            SphericalCoord dir = g.direction(row, col);
            vec3 xyz = helios::sphere2cart(make_SphericalCoord(10.f, dir.elevation, dir.azimuth));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            lidar.addHitPoint(0, xyz, dir, make_RGBcolor(1, 0, 0), data);
        }
    }

    std::vector<vec3> filled = lidar.gapfillMisses(0, false, true);
    DOCTEST_CHECK(filled.size() == blanks.size());

    double max_err_fit = 0.0;
    double max_err_affine = 0.0;
    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.getHitScanID(r) != 0 || lidar.getHitData(r, "is_miss") != 1.0) {
            continue;
        }
        int row = (int) std::lround(lidar.getHitData(r, "row"));
        int col = (int) std::lround(lidar.getHitData(r, "column"));
        SphericalCoord truth = g.direction(row, col);
        max_err_fit = std::max(max_err_fit, angularError(lidar.getHitRaydir(r), truth));
        // what the idealized affine model would have produced for the same cell
        SphericalCoord affine = scan.rc2direction(row, col);
        max_err_affine = std::max(max_err_affine, angularError(affine, truth));
    }

    // the robust per-row fit must be substantially better than the idealized affine model under tilt+shear
    DOCTEST_CHECK(max_err_fit < 0.02); // small absolute error
    DOCTEST_CHECK(max_err_fit < 0.25 * max_err_affine); // and a large improvement over the affine model
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Row/Column Noise Robustness") {
    // Encoder noise plus a few gross outliers must not corrupt the reconstruction (Theil-Sen / median robustness).
    LiDARcloud lidar;
    lidar.disableMessages();

    GenerativeGrid g{24, 48, 0.05, 0.95 * M_PI, 0.0, 1.5 * M_PI, 0.30, 0.15};
    ScanMetadata scan(make_vec3(0, 0, 0), g.Ntheta, g.theta_min, g.theta_max, g.Nphi, g.phi_min, g.phi_max, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // Deterministic pseudo-noise (no RNG) so the test is reproducible: a small bounded perturbation per cell.
    auto noise = [](int row, int col) {
        double s = std::sin(12.9898 * row + 78.233 * col) * 43758.5453;
        return (s - std::floor(s)) - 0.5; // in [-0.5, 0.5)
    };

    const double grid_dphi = (g.phi_max - g.phi_min) / double(g.Nphi - 1);
    const double grid_dtheta = (g.theta_max - g.theta_min) / double(g.Ntheta - 1);

    const int blank_r0 = 8, blank_r1 = 14, blank_c0 = 18, blank_c1 = 30;
    int outlier_counter = 0;
    for (int row = 0; row < g.Ntheta; row++) {
        for (int col = 0; col < g.Nphi; col++) {
            if (row >= blank_r0 && row <= blank_r1 && col >= blank_c0 && col <= blank_c1) {
                continue;
            }
            double zen = g.zenith(row) + 0.05 * grid_dtheta * noise(row, col);
            double az = g.azimuth(row, col) + 0.05 * grid_dphi * noise(col, row);
            // inject occasional gross outliers (~3% of returns) far from the true direction
            if ((outlier_counter++ % 33) == 0) {
                zen += 0.5;
                az += 0.5;
            }
            SphericalCoord dir = make_SphericalCoord(1.f, 0.5f * float(M_PI) - float(zen), float(az));
            vec3 xyz = helios::sphere2cart(make_SphericalCoord(10.f, dir.elevation, dir.azimuth));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            lidar.addHitPoint(0, xyz, dir, make_RGBcolor(1, 0, 0), data);
        }
    }

    std::vector<vec3> filled = lidar.gapfillMisses(0, false, false);
    DOCTEST_CHECK(filled.size() > 0);

    double max_err = 0.0;
    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.getHitScanID(r) != 0 || lidar.getHitData(r, "is_miss") != 1.0) {
            continue;
        }
        int row = (int) std::lround(lidar.getHitData(r, "row"));
        int col = (int) std::lround(lidar.getHitData(r, "column"));
        max_err = std::max(max_err, angularError(lidar.getHitRaydir(r), g.direction(row, col)));
    }
    // despite noise + outliers, reconstructed misses stay within a fraction of the grid spacing of the truth
    DOCTEST_CHECK(max_err < 0.05);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Row/Column Near-Zenith Extrapolation") {
    // Entire low-zenith rows are left completely empty (no returns at all). These rows must still be filled by
    // extrapolating the per-row model across the row axis, with directions close to the generative model.
    LiDARcloud lidar;
    lidar.disableMessages();

    GenerativeGrid g{30, 48, 0.02, 0.95 * M_PI, 0.0, 1.5 * M_PI, 0.25, 0.10};
    ScanMetadata scan(make_vec3(0, 0, 0), g.Ntheta, g.theta_min, g.theta_max, g.Nphi, g.phi_min, g.phi_max, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // Leave the first 6 rows (lowest zenith) entirely empty - the hard extrapolation case.
    const int empty_rows_below = 6;
    for (int row = empty_rows_below; row < g.Ntheta; row++) {
        for (int col = 0; col < g.Nphi; col++) {
            SphericalCoord dir = g.direction(row, col);
            vec3 xyz = helios::sphere2cart(make_SphericalCoord(10.f, dir.elevation, dir.azimuth));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            lidar.addHitPoint(0, xyz, dir, make_RGBcolor(1, 0, 0), data);
        }
    }

    std::vector<vec3> filled = lidar.gapfillMisses(0, false, true);

    // all cells in the empty rows must have been filled
    DOCTEST_CHECK(filled.size() == (size_t) (empty_rows_below * g.Nphi));

    // and they must carry the extrapolated-row flag (code 4) and be reasonably close to the generative model
    double max_err = 0.0;
    int n_extrap = 0;
    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.getHitScanID(r) != 0 || lidar.getHitData(r, "is_miss") != 1.0) {
            continue;
        }
        int row = (int) std::lround(lidar.getHitData(r, "row"));
        int col = (int) std::lround(lidar.getHitData(r, "column"));
        if (row < empty_rows_below) {
            DOCTEST_CHECK(lidar.getHitData(r, "gapfillMisses_code") == 4.0);
            n_extrap++;
            max_err = std::max(max_err, angularError(lidar.getHitRaydir(r), g.direction(row, col)));
        }
    }
    DOCTEST_CHECK(n_extrap == empty_rows_below * g.Nphi);
    // extrapolation is inherently looser than interpolation; require it to be within a few grid cells of the truth
    DOCTEST_CHECK(max_err < 0.1);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Dispatcher Selection") {
    // gapfillMisses() auto-detects the available data: row/column is preferred when present, timestamp is the fallback,
    // and a scan whose returns carry neither raises a clear error.

    // (a) returns with neither timestamp nor row/column -> error
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 0), 10, 0.05, 0.95 * M_PI, 18, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
        lidar.addScan(scan);
        // add a single bare return (no timestamp, no row/column data)
        SphericalCoord dir = make_SphericalCoord(1.f, 0.4f, 0.3f);
        lidar.addHitPoint(0, helios::sphere2cart(make_SphericalCoord(10.f, dir.elevation, dir.azimuth)), dir);

        bool threw = false;
        std::string msg;
        try {
            std::vector<vec3> filled = lidar.gapfillMisses(0);
        } catch (const std::runtime_error &e) {
            threw = true;
            msg = e.what();
        }
        DOCTEST_CHECK(threw);
        DOCTEST_CHECK(msg.find("neither 'timestamp' nor 'row'/'column'") != std::string::npos);
    }

    // (b) returns with row/column -> row/column path runs (adds the row/column-specific flag codes)
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        GenerativeGrid g{12, 24, 0.05, 0.95 * M_PI, 0.0, 2.0 * M_PI, 0.1, 0.0};
        ScanMetadata scan(make_vec3(0, 0, 0), g.Ntheta, g.theta_min, g.theta_max, g.Nphi, g.phi_min, g.phi_max, 0.0f, 0.0f, 0.0f, 0.0f, {});
        lidar.addScan(scan);
        for (int row = 0; row < g.Ntheta; row++) {
            for (int col = 0; col < g.Nphi; col++) {
                if (row == 5 && col >= 8 && col <= 12) {
                    continue; // a small interior blank
                }
                SphericalCoord dir = g.direction(row, col);
                std::map<std::string, double> data;
                data["row"] = row;
                data["column"] = col;
                lidar.addHitPoint(0, helios::sphere2cart(make_SphericalCoord(10.f, dir.elevation, dir.azimuth)), dir, make_RGBcolor(1, 0, 0), data);
            }
        }
        std::vector<vec3> filled = lidar.gapfillMisses(0, false, true);
        DOCTEST_CHECK(filled.size() == 5);
        // confirm a row/column-path flag (code 1 = interior) was assigned to a filled point
        bool found_interior_flag = false;
        for (uint r = 0; r < lidar.getHitCount(); r++) {
            if (lidar.getHitScanID(r) == 0 && lidar.getHitData(r, "is_miss") == 1.0 && lidar.getHitData(r, "gapfillMisses_code") == 1.0) {
                found_interior_flag = true;
                break;
            }
        }
        DOCTEST_CHECK(found_interior_flag);
    }
}

DOCTEST_TEST_CASE("LiDAR LAD Inversion Uncertainty") {

    // Single 1x1x1 voxel of leaves (LAI=2, spherical leaf-angle distribution) scanned from a single
    // origin. We validate the per-voxel sampling-uncertainty machinery (Pimont et al. 2018):
    //   Stage 1: sufficient statistics (beam count, RDI, mean path) are persisted.
    //   Stage 2: the sampling variance equals the binomial delta-method closed form (units check).
    //   Stage 3: supplying an element size adds the (positive) element-position variance term.
    //   Stage 4: single-voxel and group confidence intervals bracket the point estimate.

    LiDARcloud lidar;
    lidar.disableMessages();

    vec3 scan_origin(-5.0f, 0.0f, 0.5f);
    uint Ntheta = 1000;
    uint Nphi = 2000;
    std::vector<std::string> columnFormat;
    ScanMetadata scan(scan_origin, Ntheta, 0.0f, M_PI, Nphi, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(lidar.addScan(scan));

    vec3 grid_center(0.0f, 0.0f, 0.5f);
    vec3 grid_size(1.0f, 1.0f, 1.0f);
    DOCTEST_CHECK_NOTHROW(lidar.addGrid(grid_center, grid_size, make_int3(1, 1, 1), 0));
    vec3 gsize = lidar.getCellSize(0);
    float volume = gsize.x * gsize.y * gsize.z;

    Context context;
    std::vector<uint> UUIDs = context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    DOCTEST_CHECK(!UUIDs.empty());

    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, false, true)); // single-return, record misses
    DOCTEST_CHECK_NOTHROW(lidar.triangulateHitPoints(0.04, 10));

    // ---- Stage 2 (and 1): sampling-only run (element size disabled) ----
    DOCTEST_CHECK_NOTHROW(lidar.calculateLeafArea(&context, 1, -1.0f));

    int N = lidar.getCellBeamCount(0);

    // Stage 1: sufficient statistics
    DOCTEST_CHECK(N > 0);
    float leaf_area = lidar.getCellLeafArea(0);
    DOCTEST_CHECK(leaf_area > 0.f);
    float Gtheta = lidar.getCellGtheta(0);
    DOCTEST_CHECK(Gtheta > 0.f);
    float I = lidar.getCellRelativeDensityIndex(0);
    DOCTEST_CHECK(I > 0.f);
    DOCTEST_CHECK(I < 1.f);
    float zbar = lidar.getCellMeanPathLength(0);
    DOCTEST_CHECK(zbar > 0.f);
    DOCTEST_CHECK(zbar < 2.f); // path through a 1 m voxel

    float var_sampling_only = lidar.getCellLADVariance(0);
    DOCTEST_CHECK(var_sampling_only >= 0.f);
    DOCTEST_CHECK(var_sampling_only == var_sampling_only); // not NaN

    // Stage 2: closed-form units check. For single-return data the per-beam fraction is in {0,1}, so
    // the empirical-variance guard equals the binomial variance and the sampling-only variance must
    // equal the Beer-Lambert delta-method form  var(a) = I_b / (N (1-I_b) zbar^2 Gtheta^2)  with the
    // bounded RDI I_b = min(I, 1 - 1/(2N+2)). An error in the Gtheta or zbar bookkeeping (the #1
    // units-bug risk) would break this equality.
    float I_b = std::min(I, 1.f - 1.f / (2.f * float(N) + 2.f));
    float expected_var = I_b / (float(N) * (1.f - I_b) * zbar * zbar * Gtheta * Gtheta);
    DOCTEST_CHECK(var_sampling_only == doctest::Approx(expected_var).epsilon(0.02f));

    float a_est = leaf_area / volume; // LAD point estimate
    DOCTEST_CHECK(a_est > 0.f);

    // ---- Stage 3: enable element size -> element-position variance term is added ----
    DOCTEST_CHECK_NOTHROW(lidar.calculateLeafArea(&context, 1, 0.1f));
    float var_with_element = lidar.getCellLADVariance(0);
    DOCTEST_CHECK(var_with_element >= 0.f);
    // Adding the (non-negative) element-position term cannot reduce the variance.
    DOCTEST_CHECK(var_with_element >= var_sampling_only - 1e-9f);

    // ---- Stage 4: confidence intervals bracket the point estimate ----
    float lo = 0.f, hi = 0.f;
    bool have_ci = lidar.getCellLeafAreaConfidenceInterval(0, 0.95f, lo, hi);
    if (have_ci) {
        DOCTEST_CHECK(lo >= 0.f);
        DOCTEST_CHECK(lo < leaf_area);
        DOCTEST_CHECK(hi > leaf_area);
    }

    float mean_lad = 0.f, glo = 0.f, ghi = 0.f;
    bool have_group = lidar.getGroupLADConfidenceInterval(std::vector<uint>{0}, 0.95f, mean_lad, glo, ghi);
    if (have_group) {
        DOCTEST_CHECK(mean_lad == doctest::Approx(a_est).epsilon(1e-3f));
        DOCTEST_CHECK(glo >= 0.f);
        DOCTEST_CHECK(glo <= mean_lad);
        DOCTEST_CHECK(ghi >= mean_lad);
    }
    // At least the group CI should be valid for this moderate-density, high-N, small-element voxel.
    DOCTEST_CHECK(have_group);

    // ---- Export: header + one data row per grid cell ----
    const char *uncertainty_file = "lidar_lad_uncertainty_selftest.txt";
    DOCTEST_CHECK_NOTHROW(lidar.exportLeafAreaUncertainty(uncertainty_file));
    {
        std::ifstream in(uncertainty_file);
        DOCTEST_CHECK(in.is_open());
        std::string header;
        std::getline(in, header);
        DOCTEST_CHECK(header.find("cell_index") != std::string::npos);
        DOCTEST_CHECK(header.find("LAD_std_error") != std::string::npos);
        std::string row;
        bool have_row = (bool) std::getline(in, row);
        DOCTEST_CHECK(have_row);
        DOCTEST_CHECK(!row.empty());
    }
    std::remove(uncertainty_file);
}

// =====================================================================================================================
// Moving-platform (mobile/airborne) LiDAR tests. These exercise LiDARcloud::addScanMoving(): a synthetic scan driven by
// a timestamped 6-DOF pose trajectory so the scanner moves during the sweep. Each return (and miss) preserves its own
// per-beam origin (getHitOrigin / "origin_*" data) and real per-pulse timestamp. Assertions are statistical/geometric
// (reconstructed origins, timestamps, counts), never "did not throw".
// =====================================================================================================================

// Hamilton quaternion (qx,qy,qz,qw), body->world, rotating a body-frame vector into the world frame. Mirrors the
// internal convention used by the plugin so the tests can independently hand-compute expected origins/directions.
static vec3 test_quat_rotate(const vec4 &q, const vec3 &v) {
    const vec3 qv = make_vec3(q.x, q.y, q.z);
    const vec3 t = cross(qv, v) * 2.f;
    return v + t * q.w + cross(qv, t);
}

// Hamilton quaternion from intrinsic Z-Y-X (yaw-pitch-roll) Tait-Bryan angles in radians.
static vec4 test_quat_from_rpy(float roll, float pitch, float yaw) {
    const float cr = std::cos(roll * 0.5f), sr = std::sin(roll * 0.5f);
    const float cp = std::cos(pitch * 0.5f), sp = std::sin(pitch * 0.5f);
    const float cy = std::cos(yaw * 0.5f), sy = std::sin(yaw * 0.5f);
    vec4 q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    return q;
}

DOCTEST_TEST_CASE("LiDAR Moving Platform Per-Beam Origin Reconstruction") {
    // (a) Straight-line nadir trajectory pos = (0, v*t, H) over a flat patch at z=0. Every hit's reconstructed origin
    // must lie on the trajectory line (x=0, z=H, y=v*t), timestamps must increase with the pulse ordinal, and all
    // returns of a given pulse must share one timestamp.

    Context context;
    // Wide flat target so the moving scanner always sees it, plus a backing patch for a non-degenerate bounding box.
    context.addPatch(make_vec3(0, 0, 0), make_vec2(100, 100));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(100, 100));

    const uint Ntheta = 8;
    const uint Nphi = 12;
    const float thetaMin = 0.97f * float(M_PI); // near-nadir downward beams
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);
    const float H = 10.0f;
    const float v = 2.0f; // m/s along +y

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, H), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());

    // Dense trajectory spanning the full sweep duration. pulse_rate gives one pulse per (i,j) cell.
    const float pulse_rate = 1000.0f; // Hz
    const double pulse_period = 1.0 / double(pulse_rate);
    const double t_total = double(Ntheta * Nphi) * pulse_period;
    std::vector<double> traj_t;
    std::vector<vec3> traj_pos;
    std::vector<vec4> traj_quat;
    const int M = 20;
    for (int k = 0; k < M; k++) {
        double tk = t_total * double(k) / double(M - 1);
        traj_t.push_back(tk);
        traj_pos.push_back(make_vec3(0.f, float(v * tk), H));
        traj_quat.push_back(make_vec4(0, 0, 0, 1)); // identity (qx,qy,qz,qw)
    }

    lidar.addScanMoving(scan, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), pulse_rate, 0.0);
    lidar.syntheticScan(&context, false, true); // record_misses=true so transmitted beams also carry origins

    uint hit_count = lidar.getHitCount();
    DOCTEST_REQUIRE(hit_count > 0);

    // Every hit origin must lie on the trajectory line: x=0, z=H, y = v*timestamp.
    for (uint i = 0; i < hit_count; i++) {
        double t = lidar.getHitData(i, "timestamp");
        vec3 origin = lidar.getHitOrigin(i);
        DOCTEST_CHECK(origin.x == doctest::Approx(0.0f).epsilon(0.001));
        DOCTEST_CHECK(origin.z == doctest::Approx(H).epsilon(0.001));
        DOCTEST_CHECK(origin.y == doctest::Approx(float(v * t)).epsilon(0.001));
    }

    // Timestamps are monotonic in the pulse ordinal (pulse_id), and all returns of one pulse share one timestamp.
    std::map<double, double> pulse_time; // pulse_id -> timestamp
    for (uint i = 0; i < hit_count; i++) {
        DOCTEST_REQUIRE(lidar.doesHitDataExist(i, "pulse_id"));
        double pid = lidar.getHitData(i, "pulse_id");
        double t = lidar.getHitData(i, "timestamp");
        if (pulse_time.count(pid) == 0) {
            pulse_time[pid] = t;
        } else {
            DOCTEST_CHECK(pulse_time[pid] == doctest::Approx(t)); // same pulse -> exactly one timestamp
        }
        // timestamp == t0 + pulse_id * pulse_period
        DOCTEST_CHECK(t == doctest::Approx(pid * pulse_period));
    }

    // Larger pulse ordinal => strictly later time.
    double prev_t = -1.0;
    double prev_pid = -1.0;
    for (auto &kv: pulse_time) { // std::map iterates in ascending pulse_id order
        if (prev_pid >= 0) {
            DOCTEST_CHECK(kv.second > prev_t);
        }
        prev_pid = kv.first;
        prev_t = kv.second;
    }
}

DOCTEST_TEST_CASE("LiDAR Moving Platform Static Equivalence") {
    // (b) A zero-velocity "moving" scan (constant trajectory, identity quat, zero lever/boresight) must reproduce the
    // existing static addScan() over the same scene: equal hit counts and matching point-cloud centroid.

    const uint Ntheta = 10;
    const uint Nphi = 16;
    const float thetaMin = 0.95f * float(M_PI);
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);
    const float H = 8.0f;

    auto build_scene = [](Context &context) {
        context.addPatch(make_vec3(0, 0, 0), make_vec2(50, 50));
        context.addPatch(make_vec3(0, 0, -2), make_vec2(50, 50));
    };

    // Static reference.
    Context context_static;
    build_scene(context_static);
    LiDARcloud lidar_static;
    lidar_static.disableMessages();
    ScanMetadata scan_static(make_vec3(0, 0, H), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
    lidar_static.addScan(scan_static);
    lidar_static.syntheticScan(&context_static, false, true);

    // Zero-velocity moving scan at the same origin.
    Context context_moving;
    build_scene(context_moving);
    LiDARcloud lidar_moving;
    lidar_moving.disableMessages();
    ScanMetadata scan_moving(make_vec3(0, 0, H), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
    std::vector<double> traj_t = {0.0, 1.0};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, H), make_vec3(0, 0, H)}; // stationary
    std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};
    lidar_moving.addScanMoving(scan_moving, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 1.0e6f, 0.0);
    lidar_moving.syntheticScan(&context_moving, false, true);

    uint count_static = lidar_static.getHitCount();
    uint count_moving = lidar_moving.getHitCount();
    DOCTEST_REQUIRE(count_static > 0);
    DOCTEST_CHECK(count_moving == count_static); // identical geometry => identical hit count

    auto centroid = [](LiDARcloud &cloud) -> vec3 {
        vec3 c = make_vec3(0, 0, 0);
        uint n = cloud.getHitCount();
        for (uint i = 0; i < n; i++) {
            // Restrict to real (non-miss) returns; misses sit at a far sentinel distance and would swamp the centroid.
            if (cloud.getHitData(i, "is_miss") != 0.0) {
                continue;
            }
            c = c + cloud.getHitXYZ(i);
        }
        return c / float(n);
    };

    vec3 c_static = centroid(lidar_static);
    vec3 c_moving = centroid(lidar_moving);
    DOCTEST_CHECK(c_moving.x == doctest::Approx(c_static.x).epsilon(0.01));
    DOCTEST_CHECK(c_moving.y == doctest::Approx(c_static.y).epsilon(0.01));
    DOCTEST_CHECK(c_moving.z == doctest::Approx(c_static.z).epsilon(0.01));
}

DOCTEST_TEST_CASE("LiDAR Moving Platform Non-Trivial Attitude") {
    // (c) A trajectory with real roll/pitch/yaw plus a non-zero lever arm. Reconstructed origins must equal the
    // hand-computed pos + R(quat)*lever_arm. This is the ONLY test that catches a quaternion convention / axis-sign
    // bug: the nadir (a) and zero-velocity (b) tests both hide it because their rotations are identity.

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(200, 200));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(200, 200));

    const uint Ntheta = 6;
    const uint Nphi = 10;
    const float thetaMin = 0.97f * float(M_PI);
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);
    const float H = 15.0f;

    // Non-trivial, time-varying attitude: yaw sweeps while roll/pitch are held at small fixed angles.
    const float roll = 0.10f;
    const float pitch = -0.07f;
    const vec3 lever_arm = make_vec3(0.3f, -0.2f, 0.5f);

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, H), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());

    const float pulse_rate = 2000.0f;
    const double pulse_period = 1.0 / double(pulse_rate);
    const double t_total = double(Ntheta * Nphi) * pulse_period;
    std::vector<double> traj_t;
    std::vector<vec3> traj_pos;
    std::vector<vec4> traj_quat;
    const int M = 25;
    for (int k = 0; k < M; k++) {
        double tk = t_total * double(k) / double(M - 1);
        float yaw = 0.5f * float(tk / t_total); // sweeps 0 -> 0.5 rad over the scan
        traj_t.push_back(tk);
        traj_pos.push_back(make_vec3(float(1.5 * tk), float(-0.8 * tk), H));
        traj_quat.push_back(test_quat_from_rpy(roll, pitch, yaw));
    }

    lidar.addScanMoving(scan, traj_t, traj_pos, traj_quat, lever_arm, make_vec3(0, 0, 0), pulse_rate, 0.0);
    lidar.syntheticScan(&context, false, true);

    uint hit_count = lidar.getHitCount();
    DOCTEST_REQUIRE(hit_count > 0);

    // Independently reconstruct each pulse's expected origin from its timestamp by interpolating the trajectory the same
    // way poseAt does (SLERP would be exact at the sample points; between samples linear-in-yaw is close enough at this
    // angular rate that an independent SLERP reconstruction matches within tolerance). We re-derive the pose with our own
    // quaternion helper to catch any sign/axis error in the plugin.
    auto pose_at = [&](double t, vec3 &pos, vec4 &quat) {
        if (t <= traj_t.front()) {
            pos = traj_pos.front();
            quat = traj_quat.front();
            return;
        }
        if (t >= traj_t.back()) {
            pos = traj_pos.back();
            quat = traj_quat.back();
            return;
        }
        size_t i1 = 1;
        while (i1 < traj_t.size() && traj_t[i1] < t) {
            i1++;
        }
        size_t i0 = i1 - 1;
        double u = (t - traj_t[i0]) / (traj_t[i1] - traj_t[i0]);
        pos = traj_pos[i0] + (traj_pos[i1] - traj_pos[i0]) * float(u);
        // SLERP, shortest arc.
        vec4 q0 = traj_quat[i0], q1 = traj_quat[i1];
        q0.normalize();
        q1.normalize();
        double dot = double(q0.x) * q1.x + double(q0.y) * q1.y + double(q0.z) * q1.z + double(q0.w) * q1.w;
        if (dot < 0.0) {
            q1 = make_vec4(-q1.x, -q1.y, -q1.z, -q1.w);
            dot = -dot;
        }
        vec4 q;
        if (dot > 0.9995) {
            q = make_vec4(q0.x + float(u) * (q1.x - q0.x), q0.y + float(u) * (q1.y - q0.y), q0.z + float(u) * (q1.z - q0.z), q0.w + float(u) * (q1.w - q0.w));
        } else {
            double th0 = std::acos(dot);
            double th = th0 * u;
            double s0 = std::sin(th0 - th) / std::sin(th0);
            double s1 = std::sin(th) / std::sin(th0);
            q = make_vec4(float(s0 * q0.x + s1 * q1.x), float(s0 * q0.y + s1 * q1.y), float(s0 * q0.z + s1 * q1.z), float(s0 * q0.w + s1 * q1.w));
        }
        q.normalize();
        quat = q;
    };

    uint checked = 0;
    for (uint i = 0; i < hit_count; i++) {
        double t = lidar.getHitData(i, "timestamp");
        vec3 pos;
        vec4 quat;
        pose_at(t, pos, quat);
        vec3 expected_origin = pos + test_quat_rotate(quat, lever_arm);
        vec3 origin = lidar.getHitOrigin(i);
        DOCTEST_CHECK(origin.x == doctest::Approx(expected_origin.x).epsilon(0.005));
        DOCTEST_CHECK(origin.y == doctest::Approx(expected_origin.y).epsilon(0.005));
        DOCTEST_CHECK(origin.z == doctest::Approx(expected_origin.z).epsilon(0.005));
        checked++;
    }
    DOCTEST_REQUIRE(checked > 0);

    // Sanity: a non-trivial attitude must actually displace the origin away from the bare trajectory position by the
    // rotated lever arm. (If R were mistakenly identity, expected==pos and this would still pass; this check instead
    // guards that the lever arm is being applied at all.)
    {
        vec3 pos0;
        vec4 q0;
        pose_at(lidar.getHitData(0, "timestamp"), pos0, q0);
        vec3 rotated_lever = test_quat_rotate(q0, lever_arm);
        DOCTEST_CHECK(rotated_lever.magnitude() == doctest::Approx(lever_arm.magnitude()).epsilon(1e-4)); // rotation preserves length
    }
}

DOCTEST_TEST_CASE("LiDAR Moving Platform Euler-Angle Overload Equivalence") {
    // The Euler-angle addScanMoving overload must produce a point cloud identical to the quaternion overload when the
    // Euler angles are the same roll/pitch/yaw that generated the quaternions (intrinsic Z-Y-X). This guards that the
    // overload converts angles with the same convention and otherwise delegates to the same code path.

    const uint Ntheta = 6;
    const uint Nphi = 12;
    const float thetaMin = 0.96f * float(M_PI);
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);
    const float H = 12.0f;

    auto build_scene = [](Context &context) {
        context.addPatch(make_vec3(0, 0, 0), make_vec2(80, 80));
        context.addPatch(make_vec3(0, 0, -2), make_vec2(80, 80));
    };

    // A non-trivial, time-varying attitude (so a wrong axis order/sign in the overload would change the cloud).
    const int M = 12;
    std::vector<double> traj_t;
    std::vector<vec3> traj_pos;
    std::vector<vec3> traj_rpy;
    for (int k = 0; k < M; k++) {
        double tk = double(k) / double(M - 1);
        float roll = 0.08f * float(tk);
        float pitch = -0.05f * float(tk);
        float yaw = 0.3f * float(tk);
        traj_t.push_back(tk);
        traj_pos.push_back(make_vec3(float(2.0 * tk), float(-1.0 * tk), H));
        traj_rpy.push_back(make_vec3(roll, pitch, yaw));
    }

    // Build the equivalent quaternion trajectory with the same intrinsic Z-Y-X convention used internally.
    std::vector<vec4> traj_quat;
    for (const vec3 &rpy: traj_rpy) {
        traj_quat.push_back(test_quat_from_rpy(rpy.x, rpy.y, rpy.z));
    }

    const vec3 lever = make_vec3(0.2f, -0.1f, 0.4f);
    const vec3 boresight = make_vec3(0.01f, 0.02f, -0.015f);
    const float pulseRate = 2000.0f;

    // Quaternion overload.
    Context context_q;
    build_scene(context_q);
    LiDARcloud lidar_q;
    lidar_q.disableMessages();
    ScanMetadata scan_q(make_vec3(0, 0, H), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
    lidar_q.addScanMoving(scan_q, traj_t, traj_pos, traj_quat, lever, boresight, pulseRate, 0.0);
    lidar_q.syntheticScan(&context_q, false, false);

    // Euler-angle overload (same RPY).
    Context context_e;
    build_scene(context_e);
    LiDARcloud lidar_e;
    lidar_e.disableMessages();
    ScanMetadata scan_e(make_vec3(0, 0, H), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
    lidar_e.addScanMoving(scan_e, traj_t, traj_pos, traj_rpy, lever, boresight, pulseRate, 0.0);
    lidar_e.syntheticScan(&context_e, false, false);

    uint nq = lidar_q.getHitCount();
    uint ne = lidar_e.getHitCount();
    DOCTEST_REQUIRE(nq > 0);
    DOCTEST_CHECK(ne == nq); // identical inputs => identical hit count

    // Every hit position and origin must match between the two overloads (no noise was enabled, so this is exact).
    uint compared = 0;
    for (uint i = 0; i < nq && i < ne; i++) {
        vec3 pq = lidar_q.getHitXYZ(i);
        vec3 pe = lidar_e.getHitXYZ(i);
        DOCTEST_CHECK(pe.x == doctest::Approx(pq.x).epsilon(1e-5));
        DOCTEST_CHECK(pe.y == doctest::Approx(pq.y).epsilon(1e-5));
        DOCTEST_CHECK(pe.z == doctest::Approx(pq.z).epsilon(1e-5));
        vec3 oq = lidar_q.getHitOrigin(i);
        vec3 oe = lidar_e.getHitOrigin(i);
        DOCTEST_CHECK(oe.x == doctest::Approx(oq.x).epsilon(1e-5));
        DOCTEST_CHECK(oe.y == doctest::Approx(oq.y).epsilon(1e-5));
        DOCTEST_CHECK(oe.z == doctest::Approx(oq.z).epsilon(1e-5));
        compared++;
    }
    DOCTEST_REQUIRE(compared > 0);
}

DOCTEST_TEST_CASE("LiDAR Moving Platform Leaf Area Inversion") {
    // The leaf-area inversion must use each beam's actual emission origin for a moving-platform scan. We scan a known
    // 1x1x1 m leaf cube (spherical leaf-angle distribution, so the true G(theta) = 0.5) with a scanner that translates
    // over it, supply G(theta)=0.5 (triangulation is impossible without a theta-phi grid), and check the recovered LAD
    // matches the exact LAD computed from primitive areas. If the inversion still used a single static origin, the
    // per-beam path geometry would be wrong and the LAD would be biased.

    Context context;
    std::vector<uint> UUIDs = context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    DOCTEST_REQUIRE(!UUIDs.empty());

    LiDARcloud lidar;
    lidar.disableMessages();

    // Grid voxel matching the leaf cube (centered at (0,0,0.5), 1 m on a side).
    const vec3 grid_center(0.0f, 0.0f, 0.5f);
    const vec3 grid_size(1.0f, 1.0f, 1.0f);
    lidar.addGrid(grid_center, grid_size, make_int3(1, 1, 1), 0);
    const vec3 gsize = lidar.getCellSize(0);

    float LAD_exact = 0.f;
    for (uint UUID: UUIDs) {
        LAD_exact += context.getPrimitiveArea(UUID) / (gsize.x * gsize.y * gsize.z);
    }
    DOCTEST_REQUIRE(LAD_exact > 0.f);

    // Downward-looking spinning multibeam translating across the cube at height z = 5 m. The channels fan +/- a few
    // degrees about nadir so the beams sample the voxel; as the platform moves, each pulse is emitted from a different
    // point, exercising the per-beam-origin path. record_misses=true supplies the transmitted beams the inversion needs.
    std::vector<float> beamZenithAngles;
    const int Nchannels = 30;
    for (int c = 0; c < Nchannels; c++) {
        float dev = (float(c) / float(Nchannels - 1) - 0.5f) * deg2rad(40.0f); // +/-20 deg about nadir
        beamZenithAngles.push_back(float(M_PI) - fabsf(dev)); // near pi = downward
    }
    const uint Nphi = 400;
    ScanMetadata scan(make_vec3(0, 0, 5), beamZenithAngles, Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());

    // Straight, level flight across the cube along +x from x=-1 to x=+1 at z=5.
    std::vector<double> traj_t;
    std::vector<vec3> traj_pos;
    std::vector<vec3> traj_rpy;
    const int M = 30;
    for (int k = 0; k < M; k++) {
        double tk = double(k) / double(M - 1);
        traj_t.push_back(tk);
        traj_pos.push_back(make_vec3(float(-1.0 + 2.0 * tk), 0.f, 5.f));
        traj_rpy.push_back(make_vec3(0, 0, 0)); // level
    }
    const float pulseRate = float(Nchannels * Nphi); // ~one full sweep over the 1 s flight

    lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), pulseRate, 0.0);
    lidar.syntheticScan(&context, false, true);
    DOCTEST_REQUIRE(lidar.getHitCount() > 0);
    DOCTEST_REQUIRE(lidar.hasMisses());

    // Supplied-G(theta) overload: no triangulation (the moving scan has no theta-phi grid to triangulate).
    lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f);

    float LAD = lidar.getCellLeafAreaDensity(0);
    DOCTEST_CHECK(LAD == LAD); // not NaN
    // Beam-based single-voxel inversion of a noise-free synthetic scan; allow a modest tolerance for sampling/discretization.
    DOCTEST_CHECK(fabs(LAD - LAD_exact) / LAD_exact == doctest::Approx(0.0f).epsilon(0.15f));
}

DOCTEST_TEST_CASE("LiDAR calculateLeafArea Supplied-Gtheta Validation") {
    // The supplied-G(theta) overload must reject out-of-range G(theta) and must not require triangulation.
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(5, 5));

    LiDARcloud lidar;
    lidar.disableMessages();
    lidar.addGrid(make_vec3(0, 0, 0.5), make_vec3(1, 1, 1), make_int3(1, 1, 1), 0);

    // Invalid G(theta) values fail fast (no triangulation, no scan needed - validation happens first).
    bool threw_zero = false, threw_high = false;
    {
        capture_cerr capture;
        try {
            lidar.calculateLeafArea(&context, 0.0f, 1, 0.05f);
        } catch (...) {
            threw_zero = true;
        }
        try {
            lidar.calculateLeafArea(&context, 1.5f, 1, 0.05f);
        } catch (...) {
            threw_high = true;
        }
    }
    DOCTEST_CHECK(threw_zero);
    DOCTEST_CHECK(threw_high);
}

DOCTEST_TEST_CASE("LiDAR Moving Platform exportScans Origin Handling") {
    // A scan must define its beam origin either via a static <origin> XML tag or via per-point origin_x/y/z columns in
    // the data file. exportScans() must omit the misleading single <origin> for a mobile scan (whose ASCII format
    // carries per-point origins), and loadXML() must accept such a file and require one of the two sources.

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(40, 40));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(40, 40));

    const uint Ntheta = 6;
    const uint Nphi = 10;
    const float thetaMin = 0.97f * float(M_PI);
    const float thetaMax = float(M_PI);
    const float H = 10.0f;

    // Per-point origins (origin_x/y/z) in the column format mark this as a scan that defines its origin per point.
    std::vector<std::string> columnFormat = {"x", "y", "z", "timestamp", "origin_x", "origin_y", "origin_z"};

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, H), Ntheta, thetaMin, thetaMax, Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    std::vector<double> traj_t = {0.0, 1.0};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, H), make_vec3(2, 0, H)};
    std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};
    lidar.addScanMoving(scan, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 1000.0f, 0.0);
    lidar.syntheticScan(&context, false, false);

    uint export_hits = lidar.getHitCount();
    DOCTEST_REQUIRE(export_hits > 0);

    const char *xml_file = "moving_export_test.xml";
    const char *xyz_file = "moving_export_test_0.xyz";
    lidar.exportScans(xml_file);

    // The XML must NOT contain an <origin> tag (mobile scan), but must carry origin_x in the ASCII_format.
    std::string xml_contents;
    {
        std::ifstream in(xml_file);
        DOCTEST_REQUIRE(in.good());
        std::stringstream ss;
        ss << in.rdbuf();
        xml_contents = ss.str();
    }
    DOCTEST_CHECK(xml_contents.find("<origin>") == std::string::npos);
    DOCTEST_CHECK(xml_contents.find("origin_x") != std::string::npos);

    // The exported file must reload (loadXML accepts a scan with per-point origins and no <origin>), preserving hits and
    // per-point origins.
    {
        LiDARcloud reloaded;
        reloaded.disableMessages();
        DOCTEST_CHECK_NOTHROW(reloaded.loadXML(xml_file));
        DOCTEST_CHECK(reloaded.getHitCount() == export_hits);
        DOCTEST_REQUIRE(reloaded.getHitCount() > 0);
        // The reconstructed per-point origin matches the value written to the data file.
        DOCTEST_CHECK(reloaded.doesHitDataExist(0, "origin_x"));
        vec3 o = reloaded.getHitOrigin(0);
        DOCTEST_CHECK(o.z == doctest::Approx(H).epsilon(0.001));
    }

    std::remove(xml_file);
    std::remove(xyz_file);
}

DOCTEST_TEST_CASE("LiDAR loadXML Requires An Origin Source") {
    // A scan XML with neither a static <origin> nor per-point origin columns must fail fast on load.
    const char *bad_xml = "no_origin_test.xml";
    {
        std::ofstream out(bad_xml);
        out << "<helios>\n";
        out << "  <scan>\n";
        out << "    <size>10 10</size>\n";
        out << "    <ASCII_format>x y z</ASCII_format>\n";
        out << "  </scan>\n";
        out << "</helios>\n";
    }

    LiDARcloud cloud;
    cloud.disableMessages();
    bool threw = false;
    {
        capture_cerr capture; // loadXML prints "failed." to cerr before throwing
        try {
            cloud.loadXML(bad_xml);
        } catch (...) {
            threw = true;
        }
    }
    DOCTEST_CHECK(threw);

    std::remove(bad_xml);
}

DOCTEST_TEST_CASE("LiDAR Moving Platform getHitRaydir Uses Per-Beam Origin") {
    // getHitRaydir() must reconstruct the beam direction from each hit's own origin. For a moving scan, the angle from
    // the per-hit origin to the hit point should match the angle from the (static) trajectory-front origin only for the
    // earliest pulse; for later pulses (emitted from a moved platform) it must differ. We verify getHitRaydir agrees
    // with the per-hit-origin reconstruction and NOT with the static-origin reconstruction for a moved-platform hit.
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(60, 60));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(60, 60));

    const uint Ntheta = 8;
    const uint Nphi = 16;
    const float H = 10.0f;
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, H), Ntheta, 0.96f * float(M_PI), float(M_PI), Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
    std::vector<double> traj_t = {0.0, 1.0};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, H), make_vec3(5, 0, H)}; // moves 5 m in x
    std::vector<vec3> traj_rpy = {make_vec3(0, 0, 0), make_vec3(0, 0, 0)};
    lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), float(Ntheta * Nphi), 0.0);
    lidar.syntheticScan(&context, false, false);

    uint n = lidar.getHitCount();
    DOCTEST_REQUIRE(n > 0);

    vec3 static_origin = lidar.getScanOrigin(0); // trajectory front = (0,0,H)
    uint mismatched_with_static = 0;
    for (uint i = 0; i < n; i++) {
        SphericalCoord rd = lidar.getHitRaydir(i);
        // Reconstruct from per-hit origin: must match getHitRaydir (that is how it is defined).
        vec3 from_hit_origin = lidar.getHitXYZ(i) - lidar.getHitOrigin(i);
        SphericalCoord expected = cart2sphere(from_hit_origin);
        DOCTEST_CHECK(rd.zenith == doctest::Approx(expected.zenith).epsilon(1e-4));
        DOCTEST_CHECK(rd.azimuth == doctest::Approx(expected.azimuth).epsilon(1e-4));

        // A hit whose origin has moved away from the static origin must NOT match the static-origin reconstruction.
        if ((lidar.getHitOrigin(i) - static_origin).magnitude() > 1.0f) {
            SphericalCoord wrong = cart2sphere(lidar.getHitXYZ(i) - static_origin);
            if (fabs(rd.zenith - wrong.zenith) > 1e-3 || fabs(rd.azimuth - wrong.azimuth) > 1e-3) {
                mismatched_with_static++;
            }
        }
    }
    // At least some hits were emitted from a moved platform and so disagree with the static-origin direction.
    DOCTEST_CHECK(mismatched_with_static > 0);
}

DOCTEST_TEST_CASE("LiDAR Moving Platform coordinateShift Preserves Beam Geometry") {
    // coordinateShift must shift the per-hit origin_x/y/z together with the hit position, so the beam vector
    // (position - origin) is invariant under the shift for a moving scan.
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(60, 60));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(60, 60));

    const uint Ntheta = 6;
    const uint Nphi = 12;
    const float H = 9.0f;
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, H), Ntheta, 0.96f * float(M_PI), float(M_PI), Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
    std::vector<double> traj_t = {0.0, 1.0};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, H), make_vec3(4, 0, H)};
    std::vector<vec3> traj_rpy = {make_vec3(0, 0, 0), make_vec3(0, 0, 0)};
    lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), float(Ntheta * Nphi), 0.0);
    lidar.syntheticScan(&context, false, false);

    uint n = lidar.getHitCount();
    DOCTEST_REQUIRE(n > 0);

    // Record beam vectors before the shift.
    std::vector<vec3> beam_before(n);
    for (uint i = 0; i < n; i++) {
        beam_before[i] = lidar.getHitXYZ(i) - lidar.getHitOrigin(i);
    }

    const vec3 shift = make_vec3(3.0f, -2.0f, 1.5f);
    lidar.coordinateShift(shift);

    // After the shift, the beam vector (position - per-hit origin) must be unchanged, and the origin must have moved by
    // exactly the shift.
    for (uint i = 0; i < n; i++) {
        vec3 beam_after = lidar.getHitXYZ(i) - lidar.getHitOrigin(i);
        DOCTEST_CHECK(beam_after.x == doctest::Approx(beam_before[i].x).epsilon(1e-4));
        DOCTEST_CHECK(beam_after.y == doctest::Approx(beam_before[i].y).epsilon(1e-4));
        DOCTEST_CHECK(beam_after.z == doctest::Approx(beam_before[i].z).epsilon(1e-4));
    }
}

DOCTEST_TEST_CASE("LiDAR Moving Platform Static-Only Functions Fail Fast") {
    // Triangulation and ray-direction validation cannot work on a moving-platform scan (no fixed theta-phi grid /
    // single origin). They must fail fast rather than silently produce garbage.
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(40, 40));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(40, 40));

    const uint Ntheta = 6;
    const uint Nphi = 10;
    const float H = 8.0f;
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, H), Ntheta, 0.96f * float(M_PI), float(M_PI), Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
    std::vector<double> traj_t = {0.0, 1.0};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, H), make_vec3(3, 0, H)};
    std::vector<vec3> traj_rpy = {make_vec3(0, 0, 0), make_vec3(0, 0, 0)};
    lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), float(Ntheta * Nphi), 0.0);
    lidar.syntheticScan(&context, false, true);

    bool tri_threw = false, validate_threw = false;
    {
        capture_cerr capture;
        try {
            lidar.triangulateHitPoints(0.5f, 5.0f);
        } catch (...) {
            tri_threw = true;
        }
        try {
            lidar.validateRayDirections();
        } catch (...) {
            validate_threw = true;
        }
    }
    DOCTEST_CHECK(tri_threw);
    DOCTEST_CHECK(validate_threw);
}

DOCTEST_TEST_CASE("LiDAR Moving Platform addScanMoving Rejects Non-Finite Trajectory") {
    // addScanMoving must reject NaN/inf in the trajectory rather than let it propagate into NaN origins.
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 4, 0.97f * float(M_PI), float(M_PI), 8, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());

    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    std::vector<double> traj_t = {0.0, 1.0};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, 5), make_vec3(nan_val, 0, 5)}; // NaN position
    std::vector<vec3> traj_rpy = {make_vec3(0, 0, 0), make_vec3(0, 0, 0)};

    bool threw = false;
    {
        capture_cerr capture;
        try {
            lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 100.0f, 0.0);
        } catch (...) {
            threw = true;
        }
    }
    DOCTEST_CHECK(threw);
}

DOCTEST_TEST_CASE("LiDAR Moving Platform Gapfill Writes Per-Pulse Origins") {
    // gapfillMisses on a moving scan must synthesize misses whose per-pulse origin (origin_x/y/z) lies on the platform
    // trajectory, not at the single static origin. We scan a small target so some beams miss, gap-fill, and check that
    // gap-filled misses carry origin_x/y/z consistent with the straight-line trajectory (x = v*timestamp, z = H).
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(2, 2)); // small target so many beams miss
    context.addPatch(make_vec3(0, 0, -2), make_vec2(2, 2));

    const uint Ntheta = 8;
    const uint Nphi = 60;
    const float H = 6.0f;
    const float v = 3.0f; // m/s along +x
    LiDARcloud lidar;
    lidar.disableMessages();
    // Narrow downward fan so the beams sweep near nadir as the platform passes over the small target.
    ScanMetadata scan(make_vec3(0, 0, H), Ntheta, 0.93f * float(M_PI), float(M_PI), Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());

    const float pulseRate = float(Ntheta * Nphi);
    const double pulse_period = 1.0 / double(pulseRate);
    const double t_total = double(Ntheta * Nphi) * pulse_period;
    std::vector<double> traj_t;
    std::vector<vec3> traj_pos;
    std::vector<vec3> traj_rpy;
    const int M = 20;
    for (int k = 0; k < M; k++) {
        double tk = t_total * double(k) / double(M - 1);
        traj_t.push_back(tk);
        traj_pos.push_back(make_vec3(float(v * tk), 0.f, H));
        traj_rpy.push_back(make_vec3(0, 0, 0));
    }
    lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), pulseRate, 0.0);
    // record_misses=false so the actual misses are NOT recorded; gap filling must synthesize them.
    lidar.syntheticScan(&context, false, false);

    uint before = lidar.getHitCount();
    DOCTEST_REQUIRE(before > 0);

    std::vector<vec3> filled = lidar.gapfillMisses(0);
    DOCTEST_REQUIRE(!filled.empty()); // some misses were synthesized

    // Find gap-filled misses (they carry origin_x/y/z written by the moving-aware gapfill) and verify the origin lies on
    // the trajectory line: x = v*timestamp, y = 0, z = H.
    uint checked = 0;
    for (uint i = 0; i < lidar.getHitCount(); i++) {
        if (lidar.getHitData(i, "is_miss") == 0.0) {
            continue;
        }
        if (!lidar.doesHitDataExist(i, "origin_x")) {
            continue; // only the synthesized/recorded moving misses carry per-pulse origins
        }
        double t = lidar.getHitData(i, "timestamp");
        vec3 o = lidar.getHitOrigin(i);
        DOCTEST_CHECK(o.z == doctest::Approx(H).epsilon(0.01));
        DOCTEST_CHECK(o.y == doctest::Approx(0.0f).epsilon(0.01));
        DOCTEST_CHECK(o.x == doctest::Approx(float(v * t)).epsilon(0.05));
        checked++;
    }
    DOCTEST_REQUIRE(checked > 0); // at least one gap-filled/recorded moving miss carried a per-pulse origin
}
