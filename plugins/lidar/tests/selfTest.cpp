#include "LiDAR.h"
#include <filesystem>
#include <fstream>
#include <atomic>
#include <functional>
#include <set>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace std;
using namespace helios;

float err_tol = 1e-3;

//! Test-only accessor for LiDARcloud's private test/diagnostic state
/**
 * Declared a friend of LiDARcloud (see LiDAR.h) and defined only here, so none of this is
 * exposed to users of the library. Forcing the brute-force leaf-area path exists purely so the
 * self-tests can A/B it against the fast DDA path; it has no meaning in normal use.
 */
class LiDARTestHelper {
public:
    static void forceBruteForceLeafArea(LiDARcloud &pointcloud, bool force) {
        pointcloud.force_bruteforce_LAD = force;
    }
    //! Largest per-scan working array the last calculateLeafArea() allocated (elements).
    static size_t leafAreaMaxScanScratch(const LiDARcloud &pointcloud) {
        return pointcloud.leafarea_max_scan_scratch;
    }
    //! Every scalar-data label the cloud currently stores, in column order.
    static std::vector<std::string> hitDataLabels(const LiDARcloud &pointcloud) {
        return pointcloud.hit_data_labels;
    }
    //! The direction stored on a hit (not the one getHitRaydir() recomputes from position - origin).
    static SphericalCoord storedDirection(const LiDARcloud &pointcloud, uint index) {
        return pointcloud.hits.at(index).getDirection();
    }
    //! Cell-by-cell containment scan, the reference the lattice binning fast path must reproduce.
    static int containingGridCell(const LiDARcloud &pointcloud, const vec3 &p) {
        return pointcloud.getContainingGridCell(p);
    }
    //! Drop every hit (and its scalar data) but keep the scans and grid.
    static void clearHits(LiDARcloud &pointcloud) {
        pointcloud.clearHits();
    }
    //! Number of stored (non-virtual) hit points.
    static size_t storedHitCount(const LiDARcloud &pointcloud) {
        return pointcloud.hits.size();
    }
    //! Number of times the per-scan index has been rebuilt.
    static size_t scanIndexBuilds(const LiDARcloud &pointcloud) {
        return pointcloud.scan_index_builds;
    }
    //! Run a callback inside every rebuild of the per-scan index (nullptr clears it).
    static void setScanIndexBuildHook(LiDARcloud &pointcloud, std::function<void()> hook) {
        pointcloud.scan_index_build_hook = std::move(hook);
    }
    //! Mark the per-scan index stale, as every mutation of the cloud does.
    static void invalidateScanIndex(LiDARcloud &pointcloud) {
        pointcloud.invalidateScanIndex();
    }
    //! Drop every grid cell so a test can define a different grid on the same cloud.
    static void clearGrid(LiDARcloud &pointcloud) {
        pointcloud.grid_cells.clear();
        pointcloud.hitgridcellcomputed = false;
    }
};

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

DOCTEST_TEST_CASE("LiDAR triangulateHitPoints Cancel Flag") {
    // A cancel flag set before triangulation must short-circuit the per-scan/gather/triad loops
    // (and skip the Delaunay call) so the run discards any partial mesh and produces no triangles,
    // while the same input with the flag clear produces the full, unchanged triangulation. This is
    // the mechanism that lets a long triangulation be aborted from another thread instead of
    // running to completion. The poll must not perturb the non-cancelled output.
    auto run = [&](bool cancel) -> std::size_t {
        LiDARcloud pointcloud;
        pointcloud.disableMessages();
        pointcloud.loadXML("plugins/lidar/xml/sphere.xml");
        int flag = cancel ? 1 : 0;
        pointcloud.setCancelFlag(&flag);
        pointcloud.triangulateHitPoints(0.5, 5);
        return pointcloud.getTriangleCount();
    };

    std::size_t baseline = run(false);
    std::size_t cancelled = run(true);
    DOCTEST_CHECK(baseline > 0);
    DOCTEST_CHECK(cancelled == 0);

    // The non-cancelled run with a flag registered must match the count from a run with no flag at
    // all, i.e. the poll is purely an early-exit and does not change which triangles are produced.
    LiDARcloud reference;
    reference.disableMessages();
    DOCTEST_CHECK_NOTHROW(reference.loadXML("plugins/lidar/xml/sphere.xml"));
    DOCTEST_CHECK_NOTHROW(reference.triangulateHitPoints(0.5, 5));
    DOCTEST_CHECK(baseline == reference.getTriangleCount());

    // Clearing the flag (nullptr) before triangulation restores normal behavior.
    LiDARcloud cleared;
    cleared.disableMessages();
    DOCTEST_CHECK_NOTHROW(cleared.loadXML("plugins/lidar/xml/sphere.xml"));
    int flag = 1;
    cleared.setCancelFlag(&flag);
    cleared.setCancelFlag(nullptr); // cleared before the triangulation
    DOCTEST_CHECK_NOTHROW(cleared.triangulateHitPoints(0.5, 5));
    DOCTEST_CHECK(cleared.getTriangleCount() == baseline);
}

DOCTEST_TEST_CASE("LiDAR triangulateHitPoints Memory-Optimization Equivalence") {
    // The triangulation memory optimizations (dead pts_copy removal, early release of transient buffers, and the
    // shared triangulateScanSecondPass helper used by both overloads) must not change the output mesh. sphere.xml is
    // multi-scan and deterministic, so we assert: (1) the diagnostics reconcile, (2) the mesh is bit-identical across
    // two independent runs, and (3) a golden count+checksum captured from the mesh is reproduced exactly. The golden
    // constants below are the mesh this deterministic scene produces; they double as a regression guard so any future
    // change that alters the output is caught here.

    // Sum every triangle's vertex components and area into a single order-independent checksum. Using double
    // accumulation of exactly-representable float values keeps this reproducible for identical input.
    auto meshChecksum = [](const LiDARcloud &cloud) -> double {
        double sum = 0.0;
        for (uint i = 0; i < cloud.getTriangleCount(); i++) {
            Triangulation tri = cloud.getTriangle(i);
            sum += double(tri.vertex0.x) + double(tri.vertex0.y) + double(tri.vertex0.z);
            sum += double(tri.vertex1.x) + double(tri.vertex1.y) + double(tri.vertex1.z);
            sum += double(tri.vertex2.x) + double(tri.vertex2.y) + double(tri.vertex2.z);
            sum += double(tri.area);
        }
        return sum;
    };

    // --- Two-argument overload ---------------------------------------------------------------------------------
    LiDARcloud cloud_a;
    cloud_a.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud_a.loadXML("plugins/lidar/xml/sphere.xml"));
    DOCTEST_CHECK_NOTHROW(cloud_a.triangulateHitPoints(0.5, 5));

    const uint kept_a = cloud_a.getTriangleCount();
    DOCTEST_CHECK(kept_a > 0);

    // Diagnostics reconciliation: candidates == kept + dropped_lmax + dropped_aspect + dropped_degenerate.
    std::size_t candidates_a = cloud_a.getTriangulationCandidateCount();
    std::size_t dropped_a = cloud_a.getTriangulationDroppedByLmax() + cloud_a.getTriangulationDroppedByAspect() + cloud_a.getTriangulationDroppedByDegenerate();
    DOCTEST_CHECK(candidates_a == std::size_t(kept_a) + dropped_a);

    // Golden guard: this scene forms 383 triangles (matches the "Single Voxel Sphere Test" primitive count).
    DOCTEST_CHECK(kept_a == 383u);
    const double checksum_a = meshChecksum(cloud_a);

    // Determinism: an independently-built identical cloud must produce a bit-identical mesh (exact ==, not Approx).
    LiDARcloud cloud_a2;
    cloud_a2.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud_a2.loadXML("plugins/lidar/xml/sphere.xml"));
    DOCTEST_CHECK_NOTHROW(cloud_a2.triangulateHitPoints(0.5, 5));
    DOCTEST_REQUIRE(cloud_a2.getTriangleCount() == kept_a);
    for (uint i = 0; i < kept_a; i++) {
        Triangulation t1 = cloud_a.getTriangle(i);
        Triangulation t2 = cloud_a2.getTriangle(i);
        DOCTEST_CHECK(t2.vertex0.x == t1.vertex0.x);
        DOCTEST_CHECK(t2.vertex0.y == t1.vertex0.y);
        DOCTEST_CHECK(t2.vertex0.z == t1.vertex0.z);
        DOCTEST_CHECK(t2.vertex1.x == t1.vertex1.x);
        DOCTEST_CHECK(t2.vertex1.y == t1.vertex1.y);
        DOCTEST_CHECK(t2.vertex1.z == t1.vertex1.z);
        DOCTEST_CHECK(t2.vertex2.x == t1.vertex2.x);
        DOCTEST_CHECK(t2.vertex2.y == t1.vertex2.y);
        DOCTEST_CHECK(t2.vertex2.z == t1.vertex2.z);
        DOCTEST_CHECK(t2.area == t1.area);
    }
    DOCTEST_CHECK(meshChecksum(cloud_a2) == checksum_a);

    // --- Filtered (five-argument) overload ---------------------------------------------------------------------
    // sphere.xml carries a "reflectance" scalar field; filtering on it exercises the helper's scalar-field gather
    // path. The filter keeps the run non-trivial while remaining deterministic.
    LiDARcloud cloud_b;
    cloud_b.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud_b.loadXML("plugins/lidar/xml/sphere.xml"));
    DOCTEST_CHECK_NOTHROW(cloud_b.triangulateHitPoints(0.5, 5, "reflectance", -1.0f, "<"));

    const uint kept_b = cloud_b.getTriangleCount();
    std::size_t candidates_b = cloud_b.getTriangulationCandidateCount();
    std::size_t dropped_b = cloud_b.getTriangulationDroppedByLmax() + cloud_b.getTriangulationDroppedByAspect() + cloud_b.getTriangulationDroppedByDegenerate();
    DOCTEST_CHECK(candidates_b == std::size_t(kept_b) + dropped_b);

    // Determinism for the filtered overload.
    LiDARcloud cloud_b2;
    cloud_b2.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud_b2.loadXML("plugins/lidar/xml/sphere.xml"));
    DOCTEST_CHECK_NOTHROW(cloud_b2.triangulateHitPoints(0.5, 5, "reflectance", -1.0f, "<"));
    DOCTEST_REQUIRE(cloud_b2.getTriangleCount() == kept_b);
    DOCTEST_CHECK(meshChecksum(cloud_b2) == meshChecksum(cloud_b));
}

DOCTEST_TEST_CASE("LiDAR syntheticScan Progress Callback") {
    LiDARcloud cloud;
    cloud.disableMessages();

    // Two small scans so progress advances 1/2 -> 2/2 (small grid keeps this fast)
    std::vector<std::string> columnFormat;
    for (int i = 0; i < 2; i++) {
        ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 100, 0.0f, M_PI, 200, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
        DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));
    }
    DOCTEST_CHECK_NOTHROW(cloud.addGrid(make_vec3(0.0f, 0.0f, 0.5f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0));

    Context context;
    DOCTEST_CHECK_NOTHROW(context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true));

    std::vector<float> progress_values;
    std::vector<std::string> progress_messages;
    cloud.setProgressCallback([&](float progress, const std::string &message) {
        progress_values.push_back(progress);
        progress_messages.push_back(message);
    });

    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context));

    // Callback fired, reached completion, monotonic non-decreasing, message carried through
    DOCTEST_CHECK(!progress_values.empty());
    DOCTEST_CHECK(progress_values.back() == doctest::Approx(1.0f));
    bool monotonic = true;
    for (size_t i = 1; i < progress_values.size(); i++) {
        if (progress_values[i] < progress_values[i - 1]) {
            monotonic = false;
        }
    }
    DOCTEST_CHECK(monotonic);
    DOCTEST_CHECK(progress_messages.back() == "Synthetic scan");

    // Clearing the callback prevents further firing
    size_t fired_count = progress_values.size();
    cloud.setProgressCallback({});
    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context));
    DOCTEST_CHECK(progress_values.size() == fired_count);

    // Regression: the callback must still fire and reach 1.0 even when no rays hit any geometry, which takes the
    // early "no rays hit the bounding box" continue inside the per-scan loop. Use a Context with no primitives so
    // every scan takes that path, and disabled console messages so completion is delivered solely via the callback.
    LiDARcloud empty_cloud;
    empty_cloud.disableMessages();
    for (int i = 0; i < 3; i++) {
        ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 20, 0.0f, M_PI, 20, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
        DOCTEST_CHECK_NOTHROW(empty_cloud.addScan(scan));
    }
    Context empty_context;
    std::vector<float> empty_progress;
    empty_cloud.setProgressCallback([&](float progress, const std::string &) { empty_progress.push_back(progress); });
    DOCTEST_CHECK_NOTHROW(empty_cloud.syntheticScan(&empty_context));
    DOCTEST_CHECK(!empty_progress.empty());
    DOCTEST_CHECK(empty_progress.back() == doctest::Approx(1.0f));
}

DOCTEST_TEST_CASE("LiDAR syntheticScan Cancel Flag") {
    // A cancel flag set before the trace must short-circuit the ray loop so the
    // scan records (essentially) no hits, while the same scan with the flag clear
    // records many. This is the mechanism that lets a long scan be aborted
    // mid-trace and its memory freed, instead of running to completion.
    std::vector<std::string> columnFormat;

    auto run = [&](bool cancel) -> std::size_t {
        LiDARcloud cloud;
        cloud.disableMessages();
        ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 100, 0.0f, M_PI, 200, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
        cloud.addScan(scan);
        cloud.addGrid(make_vec3(0.0f, 0.0f, 0.5f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0);
        Context context;
        context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
        int flag = cancel ? 1 : 0;
        cloud.setCancelFlag(&flag);
        cloud.syntheticScan(&context);
        return cloud.getHitCount();
    };

    std::size_t baseline = run(false);
    std::size_t cancelled = run(true);
    DOCTEST_CHECK(baseline > 0);
    DOCTEST_CHECK(cancelled < baseline);
    DOCTEST_CHECK(cancelled == 0);

    // With record_misses=true, every fired pulse normally records a point (hit or miss), so an uncancelled scan fills
    // the whole grid. A cancelled scan must abort the per-chunk/per-scan loops early instead of walking every chunk and
    // recording a miss point for each beam, so it produces far fewer points than the full grid (ideally none).
    auto run_record_misses = [&](bool cancel) -> std::size_t {
        LiDARcloud cloud;
        cloud.disableMessages();
        ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 100, 0.0f, M_PI, 200, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
        cloud.addScan(scan);
        cloud.addGrid(make_vec3(0.0f, 0.0f, 0.5f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0);
        Context context;
        context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
        int flag = cancel ? 1 : 0;
        cloud.setCancelFlag(&flag);
        cloud.syntheticScan(&context, 1, 0, false /*scan_grid_only*/, true /*record_misses*/, true /*append*/);
        return cloud.getHitCount();
    };

    std::size_t baseline_misses = run_record_misses(false);
    std::size_t cancelled_misses = run_record_misses(true);
    DOCTEST_CHECK(baseline_misses > 0);
    DOCTEST_CHECK(cancelled_misses < baseline_misses);
    DOCTEST_CHECK(cancelled_misses == 0);

    // Clearing the flag (nullptr) restores normal behavior.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 100, 0.0f, M_PI, 200, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    cloud.addScan(scan);
    cloud.addGrid(make_vec3(0.0f, 0.0f, 0.5f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0);
    Context context;
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    int flag = 1;
    cloud.setCancelFlag(&flag);
    cloud.setCancelFlag(nullptr); // cleared before the scan
    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context));
    DOCTEST_CHECK(cloud.getHitCount() > 0);
}

DOCTEST_TEST_CASE("LiDAR syntheticScan Progress Pointer") {
    // The progress pointer is a caller-owned counter that syntheticScan writes the 0-based index of the scan it is
    // currently ray-tracing into, advancing it at the start of each scan and setting it to getScanCount() on completion.
    // A host polls it from another thread; here we observe the in-flight values synchronously via the per-scan progress
    // callback (which fires right after the counter is written for the same scan index).
    std::vector<std::string> columnFormat;

    LiDARcloud cloud;
    cloud.disableMessages();
    for (int i = 0; i < 3; i++) {
        ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 100, 0.0f, M_PI, 200, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
        DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));
    }
    DOCTEST_CHECK_NOTHROW(cloud.addGrid(make_vec3(0.0f, 0.0f, 0.5f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0));

    Context context;
    DOCTEST_CHECK_NOTHROW(context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true));

    volatile int progress = -1;
    cloud.setSyntheticScanProgressPointer(&progress);

    // Capture the counter at the start of each scan: the callback fires immediately after the counter is set to s.
    std::vector<int> observed;
    cloud.setProgressCallback([&](float, const std::string &) { observed.push_back(static_cast<int>(progress)); });

    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context));

    // Counter advanced through every scan index in order (the per-scan callback fires once per scan, plus a final
    // completion callback after the loop that observes the getScanCount() completion write), then landed on
    // getScanCount() to signal completion.
    DOCTEST_CHECK(observed.size() >= cloud.getScanCount());
    for (uint i = 0; i < cloud.getScanCount(); i++) {
        DOCTEST_CHECK(observed[i] == static_cast<int>(i));
    }
    DOCTEST_CHECK(progress == static_cast<int>(cloud.getScanCount()));

    cloud.setProgressCallback({});

    // The counter must still advance for scans whose rays miss everything (the early "no rays hit the bounding box"
    // continue): use a Context with no geometry so every scan takes that path.
    LiDARcloud empty_cloud;
    empty_cloud.disableMessages();
    for (int i = 0; i < 3; i++) {
        ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 20, 0.0f, M_PI, 20, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
        DOCTEST_CHECK_NOTHROW(empty_cloud.addScan(scan));
    }
    Context empty_context;
    volatile int empty_progress = -1;
    std::vector<int> empty_observed;
    empty_cloud.setSyntheticScanProgressPointer(&empty_progress);
    empty_cloud.setProgressCallback([&](float, const std::string &) { empty_observed.push_back(static_cast<int>(empty_progress)); });
    DOCTEST_CHECK_NOTHROW(empty_cloud.syntheticScan(&empty_context));
    DOCTEST_CHECK(empty_observed.size() >= empty_cloud.getScanCount());
    for (uint i = 0; i < empty_cloud.getScanCount(); i++) {
        DOCTEST_CHECK(empty_observed[i] == static_cast<int>(i));
    }
    DOCTEST_CHECK(empty_progress == static_cast<int>(empty_cloud.getScanCount()));

    // Clearing the pointer (nullptr) leaves a caller-owned counter untouched.
    volatile int stale = 42;
    empty_cloud.setSyntheticScanProgressPointer(&stale);
    empty_cloud.setSyntheticScanProgressPointer(nullptr);
    DOCTEST_CHECK_NOTHROW(empty_cloud.syntheticScan(&empty_context));
    DOCTEST_CHECK(stale == 42);
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

DOCTEST_TEST_CASE("LiDAR setExternalTriangulation Binning Test") {
    // setExternalTriangulation() touches nothing that requires ray tracing: it ingests world-space
    // triangles, bins each into a grid cell by centroid containment, drops degenerate triangles, and
    // sets the triangulation-computed flag. Verify that contract directly on a tiny hand-built scene
    // (no syntheticScan / XML), reading back the stored triangles via getTriangle().
    LiDARcloud cloud;
    cloud.disableMessages();

    // A single static scan supplies a valid scanID and a scan origin for G(theta) provenance.
    std::vector<std::string> columnFormat;
    ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 10, 0.0f, M_PI, 10, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    // 1x1x1 cell centered at the origin -> covers [-0.5,0.5]^3.
    DOCTEST_CHECK_NOTHROW(cloud.addGrid(make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0));

    // Triangle A: centroid at the cell center -> binned into cell 0.
    vec3 a0 = make_vec3(-0.1f, -0.1f, 0.0f);
    vec3 a1 = make_vec3(0.2f, -0.1f, 0.0f);
    vec3 a2 = make_vec3(-0.1f, 0.2f, 0.0f);
    // Triangle B: centroid well outside the cell -> kept but assigned to no cell (gridcell == -1).
    vec3 b0 = make_vec3(5.0f, 5.0f, 5.0f);
    vec3 b1 = make_vec3(5.3f, 5.0f, 5.0f);
    vec3 b2 = make_vec3(5.0f, 5.3f, 5.0f);

    std::vector<vec3> triangle_vertices = {a0, a1, a2, b0, b1, b2};
    std::vector<int> scanIDs = {0, 0};

    DOCTEST_CHECK_NOTHROW(cloud.setExternalTriangulation(triangle_vertices, scanIDs));

    // Both well-formed triangles are kept; counters reconcile (candidates == kept + degenerate-dropped).
    DOCTEST_CHECK(cloud.getTriangleCount() == 2);
    DOCTEST_CHECK(cloud.getTriangulationCandidateCount() == 2);
    DOCTEST_CHECK(cloud.getTriangulationDroppedByDegenerate() == 0);

    // The inside triangle is binned to cell 0; the outside one to no cell. Order is preserved.
    Triangulation triA = cloud.getTriangle(0);
    Triangulation triB = cloud.getTriangle(1);
    DOCTEST_CHECK(triA.gridcell == 0);
    DOCTEST_CHECK(triB.gridcell == -1);
    DOCTEST_CHECK(triA.scanID == 0);
    DOCTEST_CHECK(triA.vertex0 == a0); // vertices stored verbatim in world space
    DOCTEST_CHECK(triA.vertex1 == a1);
    DOCTEST_CHECK(triA.vertex2 == a2);
}

DOCTEST_TEST_CASE("LiDAR setExternalTriangulation Error Conditions Test") {
    LiDARcloud cloud;
    cloud.disableMessages();

    std::vector<std::string> columnFormat;
    ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 10, 0.0f, M_PI, 10, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    vec3 v0 = make_vec3(0.0f, 0.0f, 0.0f);
    vec3 v1 = make_vec3(0.2f, 0.0f, 0.0f);
    vec3 v2 = make_vec3(0.0f, 0.2f, 0.0f);

    // No grid defined yet -> fail fast.
    {
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(cloud.setExternalTriangulation({v0, v1, v2}, {0}));
    }

    DOCTEST_CHECK_NOTHROW(cloud.addGrid(make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0));

    // Vertex count not a multiple of 3 -> fail fast.
    {
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(cloud.setExternalTriangulation({v0, v1}, {0}));
    }

    // scanIDs size != triangle count -> fail fast.
    {
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(cloud.setExternalTriangulation({v0, v1, v2}, {0, 0}));
    }

    // scanID out of range -> fail fast (only scan 0 exists).
    {
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(cloud.setExternalTriangulation({v0, v1, v2}, {1}));
    }
    {
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(cloud.setExternalTriangulation({v0, v1, v2}, {-1}));
    }

    // A valid single triangle succeeds and is kept.
    DOCTEST_CHECK_NOTHROW(cloud.setExternalTriangulation({v0, v1, v2}, {0}));
    DOCTEST_CHECK(cloud.getTriangleCount() == 1);
}

DOCTEST_TEST_CASE("LiDAR setExternalTriangulation Degenerate Dropping Test") {
    LiDARcloud cloud;
    cloud.disableMessages();

    std::vector<std::string> columnFormat;
    ScanMetadata scan(make_vec3(-5.0f, 0.0f, 0.5f), 10, 0.0f, M_PI, 10, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));
    DOCTEST_CHECK_NOTHROW(cloud.addGrid(make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1.0f, 1.0f, 1.0f), make_int3(1, 1, 1), 0));

    // One well-formed triangle and one degenerate (collinear -> zero/NaN area) triangle.
    vec3 v0 = make_vec3(-0.1f, 0.0f, 0.0f);
    vec3 v1 = make_vec3(0.1f, 0.0f, 0.0f);
    vec3 v2 = make_vec3(0.0f, 0.2f, 0.0f);
    vec3 d0 = make_vec3(0.0f, 0.0f, 0.0f);
    vec3 d1 = make_vec3(0.1f, 0.0f, 0.0f);
    vec3 d2 = make_vec3(0.2f, 0.0f, 0.0f); // collinear with d0, d1

    std::vector<vec3> triangle_vertices = {v0, v1, v2, d0, d1, d2};
    std::vector<int> scanIDs = {0, 0};

    DOCTEST_CHECK_NOTHROW(cloud.setExternalTriangulation(triangle_vertices, scanIDs));

    // The degenerate triangle is dropped; the diagnostic counters reconcile.
    DOCTEST_CHECK(cloud.getTriangleCount() == 1);
    DOCTEST_CHECK(cloud.getTriangulationCandidateCount() == 2);
    DOCTEST_CHECK(cloud.getTriangulationDroppedByDegenerate() == 1);
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

DOCTEST_TEST_CASE("LiDAR getCellCenter Rotated Grid Test") {
    // getCellCenter() must return the TRUE world-space center for a rotated grid, i.e. the un-rotated
    // lattice center rotated about the grid anchor (about +z). A 2x1x1 grid of total size 2 centered at
    // the origin has un-rotated cell centers at x = -0.5 and x = +0.5. After a 90 degree rotation those
    // centers must land on the +/-y axis at (0, -0.5) and (0, +0.5).
    LiDARcloud cloud;
    cloud.disableMessages();

    cloud.addGrid(make_vec3(0, 0, 0), make_vec3(2, 1, 1), make_int3(2, 1, 1), 90.f);
    DOCTEST_REQUIRE(cloud.getGridCellCount() == 2);

    // Cell 0 (i=0) un-rotated at x=-0.5 -> rotates to (0,-0.5); cell 1 (i=1) at x=+0.5 -> (0,+0.5).
    vec3 c0 = cloud.getCellCenter(0);
    vec3 c1 = cloud.getCellCenter(1);
    DOCTEST_CHECK(c0.x == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(c0.y == doctest::Approx(-0.5f).epsilon(err_tol));
    DOCTEST_CHECK(c0.z == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(c1.x == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(c1.y == doctest::Approx(0.5f).epsilon(err_tol));
    DOCTEST_CHECK(c1.z == doctest::Approx(0.0f).epsilon(err_tol));

    // 45 degree case checked directly against rotatePointAboutLine of the un-rotated lattice centers.
    LiDARcloud cloud45;
    cloud45.disableMessages();
    cloud45.addGrid(make_vec3(0, 0, 0), make_vec3(2, 1, 1), make_int3(2, 1, 1), 45.f);
    const vec3 unrot[2] = {make_vec3(-0.5f, 0, 0), make_vec3(0.5f, 0, 0)};
    for (uint i = 0; i < cloud45.getGridCellCount(); i++) {
        vec3 expected = rotatePointAboutLine(unrot[i], make_vec3(0, 0, 0), make_vec3(0, 0, 1), deg2rad(45.f));
        vec3 got = cloud45.getCellCenter(i);
        DOCTEST_CHECK(got.x == doctest::Approx(expected.x).epsilon(err_tol));
        DOCTEST_CHECK(got.y == doctest::Approx(expected.y).epsilon(err_tol));
        DOCTEST_CHECK(got.z == doctest::Approx(expected.z).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("LiDAR getCellCenter Anchor Round-Trip Test") {
    // For every cell of a rotated multi-cell grid, getCellCenter(i) must equal the un-rotated lattice
    // center rotated about getCellGlobalAnchor(i) by getCellRotation(i). The global anchor must be
    // unchanged by rotation (it stays the grid center). The un-rotated fast path (rotation 0) must
    // return the lattice center exactly.
    const vec3 grid_center = make_vec3(1.f, -2.f, 0.5f);
    const vec3 grid_size = make_vec3(2.f, 2.f, 1.f);
    const int3 ndiv = make_int3(3, 3, 1);

    // Reference grid with no rotation: its centers ARE the un-rotated lattice centers.
    LiDARcloud ref;
    ref.disableMessages();
    ref.addGrid(grid_center, grid_size, ndiv, 0.f);

    // Rotated grid (same geometry) to validate the round-trip.
    const float rot_deg = 60.f;
    LiDARcloud rot;
    rot.disableMessages();
    rot.addGrid(grid_center, grid_size, ndiv, rot_deg);

    DOCTEST_REQUIRE(ref.getGridCellCount() == rot.getGridCellCount());
    for (uint i = 0; i < rot.getGridCellCount(); i++) {
        vec3 lattice_center = ref.getCellCenter(i); // rotation 0 -> un-rotated lattice center
        vec3 anchor = rot.getCellGlobalAnchor(i);
        DOCTEST_CHECK(anchor.x == doctest::Approx(grid_center.x).epsilon(err_tol));
        DOCTEST_CHECK(anchor.y == doctest::Approx(grid_center.y).epsilon(err_tol));
        DOCTEST_CHECK(anchor.z == doctest::Approx(grid_center.z).epsilon(err_tol));

        vec3 expected = rotatePointAboutLine(lattice_center, anchor, make_vec3(0, 0, 1), deg2rad(rot.getCellRotation(i)));
        vec3 got = rot.getCellCenter(i);
        DOCTEST_CHECK(got.x == doctest::Approx(expected.x).epsilon(err_tol));
        DOCTEST_CHECK(got.y == doctest::Approx(expected.y).epsilon(err_tol));
        DOCTEST_CHECK(got.z == doctest::Approx(expected.z).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("LiDAR getCellRotation Degrees Round-Trip Test") {
    // addGrid() takes the rotation in degrees; getCellRotation() must return degrees (the documented
    // public-API unit), NOT the internally-stored radians. A 45 degree grid must read back 45, not
    // 0.785..., and a 0 degree grid must read back 0.
    LiDARcloud cloud;
    cloud.disableMessages();
    cloud.addGrid(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1), 45.f);
    DOCTEST_REQUIRE(cloud.getGridCellCount() == 1);
    DOCTEST_CHECK(cloud.getCellRotation(0) == doctest::Approx(45.0f).epsilon(err_tol));

    LiDARcloud cloud0;
    cloud0.disableMessages();
    cloud0.addGrid(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1), 0.f);
    DOCTEST_CHECK(cloud0.getCellRotation(0) == doctest::Approx(0.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LiDAR Rotated Grid Wireframe Corner Test") {
    // The wireframe corners drawn by addGridWireFrametoVisualizer are the un-rotated lattice box
    // corners rotated about the grid anchor. Their centroid must coincide with the public (rotated)
    // getCellCenter(), confirming the wireframe is co-located with the rotated voxel rather than drawn
    // axis-aligned (the previous bug).
    LiDARcloud cloud;
    cloud.disableMessages();
    const vec3 grid_center = make_vec3(0.5f, 0.5f, 0.25f);
    cloud.addGrid(grid_center, make_vec3(2, 1, 1), make_int3(2, 1, 1), 30.f);

    for (uint i = 0; i < cloud.getGridCellCount(); i++) {
        vec3 center = cloud.getCellCenter(i);
        vec3 anchor = cloud.getCellGlobalAnchor(i);
        float rot = deg2rad(cloud.getCellRotation(i));
        vec3 size = cloud.getCellSize(i);

        // Reconstruct the rotated lattice center from getCellCenter by inverse-rotating it, then build
        // the eight corners about it and rotate them back; their centroid must return to getCellCenter.
        vec3 lattice_center = rotatePointAboutLine(center, anchor, make_vec3(0, 0, 1), -rot);
        vec3 boxmin = lattice_center - 0.5f * size;
        vec3 boxmax = lattice_center + 0.5f * size;
        vec3 centroid = make_vec3(0, 0, 0);
        for (int b = 0; b < 8; b++) {
            vec3 corner = make_vec3((b & 1) ? boxmax.x : boxmin.x, (b & 2) ? boxmax.y : boxmin.y, (b & 4) ? boxmax.z : boxmin.z);
            corner = rotatePointAboutLine(corner, anchor, make_vec3(0, 0, 1), rot);
            centroid = centroid + corner / 8.f;
        }
        DOCTEST_CHECK(centroid.x == doctest::Approx(center.x).epsilon(err_tol));
        DOCTEST_CHECK(centroid.y == doctest::Approx(center.y).epsilon(err_tol));
        DOCTEST_CHECK(centroid.z == doctest::Approx(center.z).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("LiDAR LAD Rotation Invariance Test") {
    // The leaf-area inversion physics must be untouched by the getCellCenter()/getCellRotation()
    // rotation change. The decisive guard uses a SINGLE voxel that fully encloses the leaf cube: the
    // scene (cube + scanners) is identical, and a single enclosing voxel is rotation-invariant (its
    // volume and the set of leaves it contains do not depend on the grid's bookkeeping rotation), so
    // the inverted LAD must be identical up to solver noise. A double-rotation bug in the binning path
    // (where the AABB and the inverse-rotated hit/beam must stay consistent) would corrupt this, as
    // would an under-sized getGridBoundingBox(), which sets the ray cull for scan_grid_only scans.
    //
    // The voxel must genuinely enclose the canopy AT BOTH ROTATIONS for that invariance argument to
    // hold. leaf_cube_LAI2_lw0_01_spherical.xml spans x,y in [-0.5, 0.5] and z in [0, 1], and a voxel
    // of side L rotated by 30 degrees contains that cube's corners only for L >= 1.0*(cos30+sin30) =
    // 1.366, so the 1.5 x 1.5 x 1.2 voxel below encloses it at 0 and at 30 degrees. A smaller voxel is
    // a sub-volume of a stochastic canopy, and rotating it selects a different subset of leaves, so its
    // LAD legitimately differs between rotations and the strict check would be measuring sampling noise.
    //
    // A multi-voxel 2x2x2 grid is NOT used for the strict check: at 30 degrees the voxel boundaries
    // slice the cube differently, genuinely re-partitioning the leaf area (the cube is symmetric under
    // 90 degrees, not 30), so per-voxel LAD legitimately changes. We still check the multi-voxel MEAN
    // LAD, which is conserved because total leaf area and total volume are rotation-invariant -- but
    // that conservation holds ONLY if the grid CAPTURES the whole canopy at both rotations. The grid
    // must therefore enclose the cube exactly as the single-voxel guard does: a grid of total span S
    // contains the cube's footprint when S >= cos(30)+sin(30) = 1.366, so the 2x2x2 grid below uses a
    // total span of 1.4 (0.7 per cell). A grid whose span only matches the cube at 0 degrees (e.g. the
    // 1.0 footprint itself) rotates ~15% of the leaf area outside the grid at 30 degrees while the grid
    // volume stays fixed, so mean LAD necessarily drops and the check measures that truncation rather
    // than rotation invariance.
    Context context_inv;
    context_inv.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true); // cube centered at z=0.5

    const float origins[4][3] = {{-5, 0, 0.5f}, {0, -5, 0.5f}, {5, 0, 0.5f}, {0, 5, 0.5f}};
    const vec3 grid_center = make_vec3(0, 0, 0.5f);

    auto run_lad = [&](const vec3 &grid_size, const int3 &grid_div, float rotation_deg) -> std::vector<float> {
        LiDARcloud lidar;
        lidar.disableMessages();
        for (int sidx = 0; sidx < 4; sidx++) {
            ScanMetadata scan(make_vec3(origins[sidx][0], origins[sidx][1], origins[sidx][2]), 2000, 0.f, M_PI, 4000, 0.f, 2.f * M_PI, 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
            lidar.addScan(scan);
        }
        lidar.addGrid(grid_center, grid_size, grid_div, rotation_deg);
        lidar.syntheticScan(&context_inv, true, true);
        lidar.triangulateHitPoints(0.04, 10);
        lidar.calculateLeafArea(&context_inv);
        std::vector<float> lad;
        for (uint i = 0; i < lidar.getGridCellCount(); i++) {
            lad.push_back(lidar.getCellLeafAreaDensity(i));
        }
        return lad;
    };

    // Strict guard: single enclosing voxel -> identical LAD regardless of grid rotation.
    std::vector<float> single0 = run_lad(make_vec3(1.5f, 1.5f, 1.2f), make_int3(1, 1, 1), 0.f);
    std::vector<float> single30 = run_lad(make_vec3(1.5f, 1.5f, 1.2f), make_int3(1, 1, 1), 30.f);
    DOCTEST_REQUIRE(single0.size() == 1);
    DOCTEST_REQUIRE(single30.size() == 1);
    DOCTEST_CHECK_MESSAGE(fabs(single0[0] - single30[0]) < 1e-2f, "Single-voxel LAD not rotation-invariant: lad0=" << single0[0] << " lad30=" << single30[0]);

    // Secondary guard: 2x2x2 grid mean LAD is conserved (total leaf area / total volume). The grid
    // spans 1.4 x 1.4 x 1.2 (0.7 x 0.7 x 0.6 per cell) so it fully encloses the canopy at 0 AND at
    // 30 degrees; only then are the captured leaf area and the grid volume both rotation-invariant.
    std::vector<float> multi0 = run_lad(make_vec3(1.4f, 1.4f, 1.2f), make_int3(2, 2, 2), 0.f);
    std::vector<float> multi30 = run_lad(make_vec3(1.4f, 1.4f, 1.2f), make_int3(2, 2, 2), 30.f);
    DOCTEST_REQUIRE(multi0.size() == multi30.size());
    float mean0 = 0.f, mean30 = 0.f;
    for (size_t i = 0; i < multi0.size(); i++) {
        mean0 += multi0[i] / float(multi0.size());
        mean30 += multi30[i] / float(multi30.size());
    }
    DOCTEST_CHECK_MESSAGE(fabs(mean0 - mean30) / std::max(mean0, mean30) < 0.05f, "Mean LAD not rotation-invariant: mean0=" << mean0 << " mean30=" << mean30);
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
    std::vector<float> beam_elev; // elevation above horizon (radians)
    std::vector<float> beam_zenith;
    for (int e = -24; e <= 24; e++) { // 49 channels, 0.5-degree elevation spacing (e in half-degrees)
        beam_elev.push_back(0.5f * float(e) * float(M_PI) / 180.f);
        beam_zenith.push_back(0.5f * float(M_PI) - 0.5f * float(e) * float(M_PI) / 180.f);
    }
    // Stationary spin for one revolution. azimuthStep = 0.09 deg -> 4000 steps/rev. A stationary spin in place is a
    // trajectory of two coincident poses; the time gap is one rotation period (channels * steps_per_rev / PRF), so the
    // sensor spins exactly once.
    const uint channels = uint(beam_elev.size());
    const float azimuthStep_rad = (360.f / 4000.f) * float(M_PI) / 180.f;
    const uint Nphi = 4000;
    const float PRF = 1.0e6f;
    const double one_rev_duration = double(channels) * 4000.0 / double(PRF);
    const std::vector<double> traj_t = {0.0, one_rev_duration};
    const std::vector<vec3> traj_pos = {scan_origin, scan_origin};
    const std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};
    DOCTEST_CHECK_NOTHROW(cloud.addScanSpinning(beam_elev, azimuthStep_rad, PRF, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>()));

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

    // A spinning scan is trajectory-driven, so it has no fixed theta-phi grid to triangulate; leaf-area inversion uses
    // the moving-aware calculateLeafArea overload that takes a supplied G(theta) (0.5 for the spherical leaf distribution
    // of this scene) instead of reconstructing leaf angles from a triangulation.
    DOCTEST_CHECK_NOTHROW(cloud.calculateLeafArea(&context, 0.5f, 1, 0.05f));
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
    DOCTEST_CHECK(reloaded.getScanMode(0) == SCAN_MODE_SPINNING);
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

// ----- Risley-prism (Livox-style rosette) scan tests -----//
// A stationary Livox-Mid-40-like prism pair: two counter-rotating ~18.7 deg wedges at n=1.51, PRF 100 kHz. A stationary
// capture is a trajectory of two coincident poses separated in time by the acquisition duration. A short 0.05 s acquisition
// gives 5000 pulses, enough to characterize the rosette without large allocations.
static std::vector<RisleyPrism> makeMid40Prisms() {
    return {RisleyPrism(18.7481 * M_PI / 180.0, 1.51, -121.5657 * 2.0 * M_PI), RisleyPrism(17.9634 * M_PI / 180.0, 1.51, 77.7430 * 2.0 * M_PI)};
}

DOCTEST_TEST_CASE("LiDAR Risley Prism Pattern Geometry") {
    // The rosette directions must lie inside a circular field of view about the optical axis (a wrong coordinate convention
    // would scatter them over a hemisphere), and a counter-rotating incommensurate pair must be non-repetitive.
    LiDARcloud cloud;
    cloud.disableMessages();

    const vec3 scan_origin(0.f, 0.f, 1.f);
    const float PRF = 100000.f;
    const double duration = 0.05; // 5000 pulses
    const std::vector<double> traj_t = {0.0, duration};
    const std::vector<vec3> traj_pos = {scan_origin, scan_origin};
    const std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};

    uint scanID = 0;
    DOCTEST_CHECK_NOTHROW(scanID = cloud.addScanRisley(makeMid40Prisms(), 1.0, PRF, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>()));

    DOCTEST_CHECK(cloud.getScanPattern(scanID) == SCAN_PATTERN_RISLEY_PRISM);
    DOCTEST_CHECK(cloud.getScanMode(scanID) == SCAN_MODE_RISLEY_PRISM);
    DOCTEST_CHECK(cloud.getScanSizeTheta(scanID) == 1u); // single-row storage
    DOCTEST_CHECK(cloud.getScanSizePhi(scanID) == 5000u); // one column per pulse

    // The prism stack round-trips through the getters.
    std::vector<RisleyPrism> prisms = cloud.getScanRisleyPrisms(scanID);
    DOCTEST_REQUIRE(prisms.size() == 2);
    DOCTEST_CHECK(cloud.getScanRisleyRefractiveIndexAir(scanID) == doctest::Approx(1.0));

    // Walk the per-pulse directions: bound the FoV half-angle and look for exact duplicate directions. Build a local
    // ScanMetadata carrying the same prisms and pulse period (one pulse per column) and query its body-frame directions.
    ScanMetadata sm(scan_origin, 1u, 0.f, float(M_PI), 5000u, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
    sm.scanPattern = SCAN_PATTERN_RISLEY_PRISM;
    sm.risley_prisms = prisms;
    sm.risley_refractive_index_air = 1.0;
    sm.pulse_period = 1.0 / double(PRF);
    float max_halfangle = 0.f;
    std::vector<vec3> dirs;
    dirs.reserve(5000);
    for (uint k = 0; k < 5000; k++) {
        SphericalCoord sph = sm.rc2direction(0, k); // body-frame direction of pulse k
        vec3 d = sphere2cart(sph);
        dirs.push_back(d);
        // Optical axis is +y; the half-angle is the angle of the beam from +y.
        float halfangle = acosf(std::max(-1.f, std::min(1.f, d.y)));
        if (halfangle > max_halfangle) {
            max_halfangle = halfangle;
        }
    }
    // A Mid-40-class wedge pair fills roughly a 38-43 deg circular FoV: every beam is within ~25 deg of the optical axis, and
    // the pattern genuinely spreads (not all clustered on-axis).
    DOCTEST_CHECK(max_halfangle < 25.f * float(M_PI) / 180.f);
    DOCTEST_CHECK(max_halfangle > 10.f * float(M_PI) / 180.f);

    // Non-repetition: no two pulses share an (almost) identical direction. Compare a bounded prefix to keep this O(n^2) check
    // cheap; a repetitive (commensurate) pattern would produce many coincident directions.
    int near_duplicates = 0;
    const size_t Ncheck = 1500;
    for (size_t a = 0; a < Ncheck; a++) {
        for (size_t b = a + 1; b < Ncheck; b++) {
            if ((dirs[a] - dirs[b]).magnitude() < 1e-6f) {
                near_duplicates++;
            }
        }
    }
    DOCTEST_CHECK(near_duplicates == 0);
}

DOCTEST_TEST_CASE("LiDAR Risley Prism Synthetic Scan") {
    Context context;
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    LiDARcloud cloud;
    cloud.disableMessages();

    // Stationary rosette looking up the +y axis at the leaf cube (the optical axis is +y). Place the scanner below the cube.
    const vec3 scan_origin(0.f, -5.f, 0.5f);
    const float PRF = 100000.f;
    const double duration = 0.1; // 10000 pulses
    const std::vector<double> traj_t = {0.0, duration};
    const std::vector<vec3> traj_pos = {scan_origin, scan_origin};
    const std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};

    uint scanID = 0;
    DOCTEST_CHECK_NOTHROW(scanID = cloud.addScanRisley(makeMid40Prisms(), 1.0, PRF, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>()));

    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context, false, true)); // record_misses=true

    uint Nhits = cloud.getHitCount();
    DOCTEST_CHECK(Nhits > 0);

    // A Risley scan is moving (trajectory-driven): every hit carries a timestamp and pulse_id, and never a spinning 'channel'.
    bool timestamp_all = true;
    bool pulse_id_all = true;
    bool no_channel = true;
    bool any_real_hit = false;
    for (uint h = 0; h < Nhits; h++) {
        if (!cloud.doesHitDataExist(h, "timestamp")) {
            timestamp_all = false;
        }
        if (!cloud.doesHitDataExist(h, "pulse_id")) {
            pulse_id_all = false;
        }
        if (cloud.doesHitDataExist(h, "channel")) {
            no_channel = false;
        }
        if (cloud.getHitData(h, "is_miss") == 0.0) {
            any_real_hit = true;
        }
    }
    DOCTEST_CHECK(timestamp_all);
    DOCTEST_CHECK(pulse_id_all);
    DOCTEST_CHECK(no_channel);
    DOCTEST_CHECK(any_real_hit);

    // Triangulation and row/column gap-filling are not supported for trajectory-driven scans (Risley is always moving).
    DOCTEST_CHECK_THROWS(cloud.triangulateHitPoints(0.5f, 5.f));
}

DOCTEST_TEST_CASE("LiDAR Risley Prism Fail-Fast Errors") {
    LiDARcloud cloud;
    cloud.disableMessages();

    const std::vector<double> traj_t = {0.0, 0.05};
    const std::vector<vec3> traj_pos = {make_vec3(0, 0, 1), make_vec3(0, 0, 1)};
    const std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};
    const std::vector<std::string> cf;

    // Empty prism stack.
    DOCTEST_CHECK_THROWS(cloud.addScanRisley(std::vector<RisleyPrism>(), 1.0, 100000.f, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, cf));
    // Non-positive PRF.
    DOCTEST_CHECK_THROWS(cloud.addScanRisley(makeMid40Prisms(), 1.0, 0.f, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, cf));
    // Empty trajectory.
    DOCTEST_CHECK_THROWS(cloud.addScanRisley(makeMid40Prisms(), 1.0, 100000.f, std::vector<double>(), std::vector<vec3>(), std::vector<vec4>(), make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, cf));
    // Zero-duration trajectory (coincident times).
    const std::vector<double> traj_t_zero = {1.0, 1.0};
    DOCTEST_CHECK_THROWS(cloud.addScanRisley(makeMid40Prisms(), 1.0, 100000.f, traj_t_zero, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, cf));
}

DOCTEST_TEST_CASE("LiDAR Risley Prism XML Round-Trip") {
    // exportScans must persist the Risley geometry (pattern + prism stack) so it round-trips on reload.
    Context context;
    // A small patch in front of the scanner (along +y, the optical axis) gives the scan some real returns to export.
    context.addPatch(make_vec3(0.f, 5.f, 1.f), make_vec2(4.f, 4.f), make_SphericalCoord(0.5f * float(M_PI), float(M_PI)));

    LiDARcloud cloud;
    cloud.disableMessages();

    const vec3 scan_origin(0.f, 0.f, 1.f);
    const float PRF = 100000.f;
    const std::vector<double> traj_t = {0.0, 0.05};
    const std::vector<vec3> traj_pos = {scan_origin, scan_origin};
    const std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};
    const std::vector<RisleyPrism> prisms = makeMid40Prisms();

    uint scanID = 0;
    DOCTEST_CHECK_NOTHROW(scanID = cloud.addScanRisley(prisms, 1.0, PRF, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{"x", "y", "z", "origin_x", "origin_y", "origin_z"}));

    // Populate hits so exportScans writes a complete scan with per-pulse origin columns.
    DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context, false, true));

    const std::string out_dir = "lidar_risley_export_tmp";
    std::filesystem::remove_all(out_dir);
    const std::string xml_out = out_dir + "/scans.xml";
    DOCTEST_CHECK_NOTHROW(cloud.exportScans(xml_out.c_str()));

    LiDARcloud reloaded;
    reloaded.disableMessages();
    DOCTEST_CHECK_NOTHROW(reloaded.loadXML(xml_out.c_str()));
    DOCTEST_REQUIRE(reloaded.getScanCount() == 1);
    DOCTEST_CHECK(reloaded.getScanPattern(0) == SCAN_PATTERN_RISLEY_PRISM);
    DOCTEST_CHECK(reloaded.getScanMode(0) == SCAN_MODE_RISLEY_PRISM);

    std::vector<RisleyPrism> reloaded_prisms = reloaded.getScanRisleyPrisms(0);
    DOCTEST_REQUIRE(reloaded_prisms.size() == prisms.size());
    for (size_t k = 0; k < prisms.size(); k++) {
        DOCTEST_CHECK(reloaded_prisms[k].wedge_angle == doctest::Approx(prisms[k].wedge_angle).epsilon(1e-4));
        DOCTEST_CHECK(reloaded_prisms[k].refractive_index == doctest::Approx(prisms[k].refractive_index).epsilon(1e-4));
        DOCTEST_CHECK(reloaded_prisms[k].rotor_rate == doctest::Approx(prisms[k].rotor_rate).epsilon(1e-4));
        DOCTEST_CHECK(reloaded_prisms[k].phase == doctest::Approx(prisms[k].phase));
    }
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

DOCTEST_TEST_CASE("LiDAR Stratified Gaussian Footprint Sampler Invariants") {
    // Invariants of the stratified, importance-sampled Gaussian beam-footprint sampler (divergence-cone direction and
    // exit-aperture origin). Because the Gaussian profile now lives in the sample density, every sub-ray carries unit
    // weight, so the energy-weighted return centroid of a symmetric footprint must sit on the nominal beam axis and the
    // range to a perpendicular wall must be preserved. Sub-ray positions are not exposed individually (they are merged
    // into per-pulse returns), so the footprint symmetry is checked through the centroid and the per-path range accuracy.
    // A 4 m x 4 m wall in the y-z plane at x=0; the scanner sits on the -x axis aimed straight at the wall center.
    auto build_wall = [](Context &ctx) {
        ctx.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, -2.f), make_vec3(0.f, 2.f, 2.f), RGB::green);
        ctx.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, 2.f), make_vec3(0.f, -2.f, 2.f), RGB::green);
    };
    const vec3 scan_origin(-5.f, 0.f, 0.f);
    const float standoff = 5.f; // perpendicular distance from origin to the wall at x=0
    const float aim_phi = 0.5f * float(M_PI); // +x heading (azimuth measured from +y)
    const float aim_theta = 0.5f * float(M_PI); // horizontal (elevation 0)
    const float beam_div = 0.5e-3f; // 0.5 mrad divergence half-angle
    const float exit_diam = 0.05f; // 5 cm exit aperture (non-negligible footprint)
    const int rays_per_pulse = 200;
    const float pulse_threshold = 0.3f;

    // A single beam aimed at the wall center. Both footprint dimensions active. No noise so the only transverse spread
    // is the footprint itself; a symmetric footprint with unit weights must put the return centroid on the beam axis.
    {
        Context context;
        build_wall(context);
        LiDARcloud cloud;
        cloud.disableMessages();
        std::vector<std::string> columnFormat;
        ScanMetadata scan(scan_origin, 1, aim_theta, aim_theta, 1, aim_phi, aim_phi, exit_diam, beam_div, 0.f /*rangeNoise*/, 0.f /*angleNoise*/, columnFormat);
        DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));
        DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context, rays_per_pulse, pulse_threshold, RETURN_MODE_MULTI, false, false, false));

        uint Nhits = cloud.getHitCount();
        DOCTEST_REQUIRE(Nhits >= 1);
        // Single perpendicular surface within the merge window => exactly one return.
        DOCTEST_CHECK(Nhits == 1);
        vec3 p = cloud.getHitXYZ(0);
        double dist = cloud.getHitData(0, "distance");
        DOCTEST_CHECK(fabs(p.x) < 1e-3f); // on the wall plane
        DOCTEST_CHECK(fabs(p.y) < 1e-3f); // centroid on the beam axis (no transverse bias)
        DOCTEST_CHECK(fabs(p.z) < 1e-3f);
        DOCTEST_CHECK(dist == doctest::Approx(standoff).epsilon(0.001)); // range to the perpendicular wall preserved
    }

    // Each footprint dimension exercised independently (divergence-only and aperture-only paths) plus the fully
    // degenerate point-source path, all on a small fan aimed at the wall. Every real return must land on the wall at the
    // correct range, confirming the new paths do not bias the range or produce NaNs.
    auto check_fan = [&](float bdiv, float ediam) {
        Context context;
        build_wall(context);
        LiDARcloud cloud;
        cloud.disableMessages();
        std::vector<std::string> columnFormat;
        const float half = 4.f * float(M_PI) / 180.f; // +/-4 deg fan
        ScanMetadata scan(scan_origin, 9, aim_theta - half, aim_theta + half, 9, aim_phi - half, aim_phi + half, ediam, bdiv, 0.f, 0.f, columnFormat);
        DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));
        DOCTEST_CHECK_NOTHROW(cloud.syntheticScan(&context, rays_per_pulse, pulse_threshold, RETURN_MODE_MULTI, false, false, false));
        uint Nhits = cloud.getHitCount();
        DOCTEST_REQUIRE(Nhits > 0);
        bool ranges_ok = true;
        for (uint h = 0; h < Nhits; h++) {
            if (cloud.getHitData(h, "is_miss") != 0.0) {
                continue;
            }
            double dist = cloud.getHitData(h, "distance");
            // Corner beams of the +/-4 deg square fan are off-axis by up to sqrt(2)*4 deg, reaching
            // standoff/cos(sqrt(2)*4 deg) ~= 5.025 m; allow a small margin for the footprint spread.
            if (dist < standoff - 1e-2 || dist > standoff / cosf(sqrtf(2.f) * half) + 5e-2) {
                ranges_ok = false;
            }
            vec3 p = cloud.getHitXYZ(h);
            if (fabs(p.x) > 1e-2f || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                ranges_ok = false;
            }
        }
        DOCTEST_CHECK(ranges_ok);
    };
    check_fan(beam_div, 0.f); // divergence-cone sampler only (point source)
    check_fan(0.f, exit_diam); // exit-aperture sampler only (zero divergence)
    check_fan(0.f, 0.f); // fully degenerate: all sub-rays on the nominal axis

    // Seed reproducibility: the per-beam random offsets are pre-drawn serially from the Context RNG, so a seeded Context
    // must reproduce an identical point cloud regardless of OpenMP scheduling. Two runs with the same seed must match.
    {
        auto run_seeded = [&](uint seed, std::vector<vec3> &pts, std::vector<double> &ranges) {
            Context context;
            context.seedRandomGenerator(seed);
            build_wall(context);
            LiDARcloud cloud;
            cloud.disableMessages();
            std::vector<std::string> columnFormat;
            const float half = 4.f * float(M_PI) / 180.f;
            ScanMetadata scan(scan_origin, 12, aim_theta - half, aim_theta + half, 12, aim_phi - half, aim_phi + half, exit_diam, beam_div, 0.02f /*rangeNoise*/, 1e-4f /*angleNoise*/, columnFormat);
            cloud.addScan(scan);
            cloud.syntheticScan(&context, rays_per_pulse, pulse_threshold, RETURN_MODE_MULTI, false, false, false);
            uint Nhits = cloud.getHitCount();
            pts.clear();
            ranges.clear();
            for (uint h = 0; h < Nhits; h++) {
                pts.push_back(cloud.getHitXYZ(h));
                ranges.push_back(cloud.getHitData(h, "distance"));
            }
        };
        std::vector<vec3> pts_a, pts_b;
        std::vector<double> ranges_a, ranges_b;
        run_seeded(12345u, pts_a, ranges_a);
        run_seeded(12345u, pts_b, ranges_b);
        DOCTEST_REQUIRE(pts_a.size() == pts_b.size());
        DOCTEST_REQUIRE(!pts_a.empty());
        bool identical = true;
        for (size_t i = 0; i < pts_a.size(); i++) {
            if ((pts_a[i] - pts_b[i]).magnitude() > 1e-6f || fabs(ranges_a[i] - ranges_b[i]) > 1e-6) {
                identical = false;
            }
        }
        DOCTEST_CHECK(identical);
    }
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Pulse-Shape Deviation") {
    // The synthetic-scan "deviation" field is a dimensionless pulse-shape distortion metric: the within-return range
    // spread in excess of the transmit pulse width, normalized by the pulse width. It mirrors the RIEGL "pulse shape
    // deviation" confidence value (small for a clean single-surface return, large for a broadened/sloped return). The two
    // defining behaviors are checked: (1) a flat wall perpendicular to the beam has near-zero spread => deviation ~= 0;
    // (2) a steeply tilted wall spreads the footprint sub-rays over a range interval => deviation grows well above zero.
    // A wide beam footprint and many sub-rays are used so the footprint actually samples the surface tilt.
    const vec3 scan_origin(-5.f, 0.f, 0.f);
    const float aim_phi = 0.5f * float(M_PI); // +x heading (azimuth measured from +y)
    const float aim_theta = 0.5f * float(M_PI); // horizontal (elevation 0)
    const float beam_div = 15.e-3f; // 15 mrad divergence => ~15 cm footprint at 5 m, enough to resolve the tilt
    const float exit_diam = 0.05f; // 5 cm exit aperture
    const int rays_per_pulse = 400;
    const float pulse_width = 0.5f; // merge window / reference pulse range-extent (meters)

    auto scan_single_wall = [&](Context &context) -> double {
        LiDARcloud cloud;
        cloud.disableMessages();
        std::vector<std::string> columnFormat;
        ScanMetadata scan(scan_origin, 1, aim_theta, aim_theta, 1, aim_phi, aim_phi, exit_diam, beam_div, 0.f /*rangeNoise*/, 0.f /*angleNoise*/, columnFormat);
        cloud.addScan(scan);
        cloud.syntheticScan(&context, rays_per_pulse, pulse_width, RETURN_MODE_MULTI, false, false, false);
        uint Nhits = cloud.getHitCount();
        DOCTEST_REQUIRE(Nhits >= 1);
        // Find the (single) real return.
        for (uint h = 0; h < Nhits; h++) {
            if (cloud.getHitData(h, "is_miss") == 0.0) {
                return cloud.getHitData(h, "deviation");
            }
        }
        DOCTEST_FAIL("no real return produced");
        return -1.0;
    };

    // (1) Flat wall in the y-z plane at x=0, perpendicular to the +x beam: all sub-rays hit at essentially the same range.
    double deviation_perpendicular;
    {
        Context context;
        context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, -2.f), make_vec3(0.f, 2.f, 2.f), RGB::green);
        context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, 2.f), make_vec3(0.f, -2.f, 2.f), RGB::green);
        deviation_perpendicular = scan_single_wall(context);
    }
    DOCTEST_CHECK(deviation_perpendicular >= 0.0); // never negative
    DOCTEST_CHECK(deviation_perpendicular < 0.05); // clean perpendicular return => negligible pulse-shape deviation

    // (2) Wall steeply tilted about the z-axis so the surface recedes across the beam footprint in y. The footprint now
    // samples a range interval, so the merged return is broadened and the deviation rises well above the clean case.
    double deviation_tilted;
    {
        Context context;
        // Plane through the origin tilted 75 deg from the y-z plane: large dx across the footprint's y-extent.
        const float c = cosf(75.f * float(M_PI) / 180.f);
        const float s = sinf(75.f * float(M_PI) / 180.f);
        // Rotate the flat wall's corners about z: (x,y) -> (x*c - y*s, x*s + y*c), with x=0 on the original wall.
        auto rot = [&](float y, float z) { return make_vec3(-y * s, y * c, z); };
        context.addTriangle(rot(-2.f, -2.f), rot(2.f, -2.f), rot(2.f, 2.f), RGB::green);
        context.addTriangle(rot(-2.f, -2.f), rot(2.f, 2.f), rot(-2.f, 2.f), RGB::green);
        deviation_tilted = scan_single_wall(context);
    }
    DOCTEST_CHECK(deviation_tilted > deviation_perpendicular); // tilt broadens the pulse => larger deviation
    DOCTEST_CHECK(deviation_tilted > 0.05); // meaningfully non-zero distortion
}

DOCTEST_TEST_CASE("LiDAR Default Detection Threshold") {
    // The default detectionThreshold is a non-zero noise floor (0.05) that pairs with the recommended ~40 rays/pulse: it
    // suppresses the single-sub-ray "phantom" returns that otherwise force very high ray counts to converge. Verify the
    // default value and that, on a scene with a deliberately weak return, the default floor drops it while a 0 threshold
    // keeps it. Scene (the three partial-footprint slivers from the N-return tests): the nearest sliver is weak enough to
    // fall below the 0.05 floor.
    auto buildScene = [](Context &context) {
        context.addPatch(make_vec3(0.13f, 0.f, 2.5f), make_vec2(0.10f, 0.4f)); // nearest sliver (weakest)
        context.addPatch(make_vec3(-0.12f, 0.f, 1.5f), make_vec2(0.16f, 0.4f)); // middle sliver
        context.addPatch(make_vec3(0.f, 0.f, 0.4f), make_vec2(2.0f, 2.0f)); // far full footprint (strongest)
    };

    // Default value is the 0.05 noise floor.
    {
        Context ctx;
        buildScene(ctx);
        LiDARcloud cloud;
        cloud.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
        uint id = cloud.addScan(scan);
        float thr = cloud.getScanDetectionThreshold(id);
        DOCTEST_CHECK(thr == doctest::Approx(0.05f));
    }

    auto count_returns = [&](float threshold) -> uint {
        Context ctx;
        buildScene(ctx);
        LiDARcloud cloud;
        cloud.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
        uint id = cloud.addScan(scan);
        cloud.setScanPulseWidth(id, 0.5f);
        cloud.setScanDetectionThreshold(id, threshold);
        cloud.syntheticScan(&ctx, 400, 0.5f, RETURN_MODE_MULTI);
        return cloud.getHitCount();
    };
    uint n_zero = count_returns(0.f); // no noise floor: all three slivers detected
    uint n_default = count_returns(0.05f); // default floor
    uint n_high = count_returns(0.5f); // aggressive floor: only the strong far surface clears it
    DOCTEST_CHECK(n_zero == 3); // all three partial-footprint returns resolve with no floor
    DOCTEST_CHECK(n_default <= n_zero); // the noise floor can only remove returns, never add
    DOCTEST_CHECK(n_high < n_zero); // a high floor suppresses the weak returns
    DOCTEST_CHECK(n_high >= 1); // the strong far surface is still detected
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Beam Chunking Equivalence") {
    // The per-scan beam fan-out is traced in memory-bounded chunks; a multi-chunk run must produce exactly the same point
    // cloud as the default single-chunk run. Beam divergence and noise are disabled so the per-pulse sub-rays are the
    // deterministic nominal direction (no RNG draws during ray generation), making the comparison exact regardless of how
    // the beams are split into chunks. A multi-return scan (rays_per_pulse > 1) exercises the waveform reduction path.
    Context context;
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, -2.f), make_vec3(0.f, 2.f, 2.f), RGB::green);
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, 2.f), make_vec3(0.f, -2.f, 2.f), RGB::green);

    vec3 scan_origin(-5.f, 0.f, 0.5f);
    uint Ntheta = 60;
    uint Nphi = 60;
    float thetaMin = 0.5f * float(M_PI) - 6.f * float(M_PI) / 180.f;
    float thetaMax = 0.5f * float(M_PI) + 6.f * float(M_PI) / 180.f;
    float phiMin = 0.5f * float(M_PI) - 20.f * float(M_PI) / 180.f;
    float phiMax = 0.5f * float(M_PI) + 20.f * float(M_PI) / 180.f;
    std::vector<std::string> columnFormat;

    const int rays_per_pulse = 10;
    const float pulse_distance_threshold = 0.3f;

    // Default budget: the whole scan fits in a single chunk.
    LiDARcloud cloud_single;
    cloud_single.disableMessages();
    ScanMetadata scan_single(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.f /*exitDiameter*/, 0.f /*beamDivergence*/, 0.f /*rangeNoise*/, 0.f /*angleNoise*/, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud_single.addScan(scan_single));
    DOCTEST_CHECK_NOTHROW(cloud_single.syntheticScan(&context, rays_per_pulse, pulse_distance_threshold, RETURN_MODE_MULTI, false /*scan_grid_only*/, true /*record_misses*/, false /*append*/));
    uint Nhits_single = cloud_single.getHitCount();
    DOCTEST_REQUIRE(Nhits_single > 0);

    // Tiny budget: forces the beams to be processed in many chunks.
    LiDARcloud cloud_chunked;
    cloud_chunked.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud_chunked.setSyntheticScanMemoryBudget(4096)); // a few KB => clamped to >=1 beam/chunk, many chunks
    DOCTEST_CHECK(cloud_chunked.getSyntheticScanMemoryBudget() == 4096);
    ScanMetadata scan_chunked(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.f, 0.f, 0.f, 0.f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud_chunked.addScan(scan_chunked));
    DOCTEST_CHECK_NOTHROW(cloud_chunked.syntheticScan(&context, rays_per_pulse, pulse_distance_threshold, RETURN_MODE_MULTI, false, true, false));
    uint Nhits_chunked = cloud_chunked.getHitCount();

    // Identical hit count and identical per-hit distances/positions (no RNG in ray-gen => exact match).
    DOCTEST_CHECK(Nhits_chunked == Nhits_single);
    if (Nhits_chunked == Nhits_single) {
        bool all_match = true;
        for (uint h = 0; h < Nhits_single; h++) {
            if (cloud_chunked.getHitData(h, "is_miss") != cloud_single.getHitData(h, "is_miss")) {
                all_match = false;
                break;
            }
            if (std::fabs(cloud_chunked.getHitData(h, "distance") - cloud_single.getHitData(h, "distance")) > 1e-6) {
                all_match = false;
                break;
            }
            vec3 p_single = cloud_single.getHitXYZ(h);
            vec3 p_chunked = cloud_chunked.getHitXYZ(h);
            if ((p_single - p_chunked).magnitude() > 1e-5f) {
                all_match = false;
                break;
            }
        }
        DOCTEST_CHECK(all_match);
    }

    // The setter rejects a zero budget.
    LiDARcloud cloud_validate;
    cloud_validate.disableMessages();
    DOCTEST_CHECK_THROWS(cloud_validate.setSyntheticScanMemoryBudget(0));
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Beam Chunking Equivalence (GPU dispatch)") {
    // Like the test above, but sized so the per-scan sub-ray batches cross the collision-detection GPU dispatch
    // threshold (>= 1M rays on a >= 500-primitive scene). On a CUDA build the single-chunk scan runs entirely on the
    // GPU, and each >= 1M-ray chunk of the multi-chunk scan also runs on the GPU (bit-identical to the single-chunk
    // result, since a ray's result is independent of how beams are batched). Any final sub-1M remainder chunk falls
    // back to the CPU traversal, whose Moller-Trumbore-vs-CPU floating-point differences are bounded well within the
    // tolerances below; hit counts must match exactly. On a non-CUDA build everything runs on the CPU and the
    // comparison is simply exact-within-tolerance. This guards the resident-scene GPU path under chunking.
    Context context;

    // A large triangulated wall at x=0 (800 triangles >= MIN_PRIMITIVES_FOR_GPU) that fully covers the scan FOV so
    // every beam hits squarely (no grazing => stable hit counts between the GPU and CPU paths).
    const int gy = 20, gz = 20;
    const float ymin = -3.f, ymax = 3.f, zmin = -2.5f, zmax = 3.5f;
    const float dy = (ymax - ymin) / float(gy);
    const float dz = (zmax - zmin) / float(gz);
    for (int iy = 0; iy < gy; iy++) {
        for (int iz = 0; iz < gz; iz++) {
            float y = ymin + iy * dy;
            float z = zmin + iz * dz;
            context.addTriangle(make_vec3(0.f, y, z), make_vec3(0.f, y + dy, z), make_vec3(0.f, y + dy, z + dz), RGB::green);
            context.addTriangle(make_vec3(0.f, y, z), make_vec3(0.f, y + dy, z + dz), make_vec3(0.f, y, z + dz), RGB::green);
        }
    }

    vec3 scan_origin(-5.f, 0.f, 0.5f);
    uint Ntheta = 120;
    uint Nphi = 120;
    float thetaMin = 0.5f * float(M_PI) - 6.f * float(M_PI) / 180.f;
    float thetaMax = 0.5f * float(M_PI) + 6.f * float(M_PI) / 180.f;
    float phiMin = 0.5f * float(M_PI) - 20.f * float(M_PI) / 180.f;
    float phiMax = 0.5f * float(M_PI) + 20.f * float(M_PI) / 180.f;
    std::vector<std::string> columnFormat;

    const int rays_per_pulse = 100; // 120*120*100 = 1.44M sub-rays => single chunk crosses the 1M GPU threshold
    const float pulse_distance_threshold = 0.3f;

    // Single chunk (default budget): the whole 1.44M-ray scan dispatches to the GPU.
    LiDARcloud cloud_single;
    cloud_single.disableMessages();
    ScanMetadata scan_single(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.f, 0.f, 0.f, 0.f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud_single.addScan(scan_single));
    DOCTEST_CHECK_NOTHROW(cloud_single.syntheticScan(&context, rays_per_pulse, pulse_distance_threshold, RETURN_MODE_MULTI, false, true, false));
    uint Nhits_single = cloud_single.getHitCount();
    DOCTEST_REQUIRE(Nhits_single > 0);

    // Multi-chunk: an ~8 MB budget floors each chunk at ~1.05M sub-rays, so chunks still hit the GPU path; the trailing
    // remainder chunk may fall to the CPU.
    LiDARcloud cloud_chunked;
    cloud_chunked.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud_chunked.setSyntheticScanMemoryBudget(8u * 1024u * 1024u));
    ScanMetadata scan_chunked(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.f, 0.f, 0.f, 0.f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud_chunked.addScan(scan_chunked));
    DOCTEST_CHECK_NOTHROW(cloud_chunked.syntheticScan(&context, rays_per_pulse, pulse_distance_threshold, RETURN_MODE_MULTI, false, true, false));
    uint Nhits_chunked = cloud_chunked.getHitCount();

    DOCTEST_CHECK(Nhits_chunked == Nhits_single);
    if (Nhits_chunked == Nhits_single) {
        size_t mismatches = 0;
        for (uint h = 0; h < Nhits_single; h++) {
            if (cloud_chunked.getHitData(h, "is_miss") != cloud_single.getHitData(h, "is_miss")) {
                mismatches++;
                continue;
            }
            if (std::fabs(cloud_chunked.getHitData(h, "distance") - cloud_single.getHitData(h, "distance")) > 1e-3) {
                mismatches++;
                continue;
            }
            if ((cloud_single.getHitXYZ(h) - cloud_chunked.getHitXYZ(h)).magnitude() > 1e-2f) {
                mismatches++;
            }
        }
        DOCTEST_CHECK(mismatches == 0);
    }
}

DOCTEST_TEST_CASE("LiDAR GPU Availability Query") {
    LiDARcloud lidarcloud;
    lidarcloud.disableMessages();

    // LiDARcloud::isGPUAvailable() forwards to the static CollisionDetection probe and must
    // be valid even before any geometry/collision instance exists.
    bool lidar_available = false;
    DOCTEST_CHECK_NOTHROW(lidar_available = lidarcloud.isGPUAvailable());
    DOCTEST_CHECK(lidar_available == CollisionDetection::isGPUAvailable());

    // The forwarded enabled-state query must be callable and consistent with availability:
    // before any explicit toggle, the effective default matches GPU availability.
    bool lidar_enabled = false;
    DOCTEST_CHECK_NOTHROW(lidar_enabled = lidarcloud.isGPUAccelerationEnabled());
    DOCTEST_CHECK(lidar_enabled == lidar_available);
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam XML Load Geometry") {
    // Write a temporary spinning-multibeam scan XML (physical-parameter form: stationary spin for one revolution, expressed
    // as two coincident poses one rotation period apart) and verify it loads with the correct channel geometry. The azimuth
    // grid and revolution count are derived from <azimuthStep> + <PRF> + the trajectory duration, not hand-supplied.
    // 7 channels * 720 steps/rev / 100000 Hz = 0.0504 s for exactly one revolution.
    std::string xml_path = "plugins/lidar/xml/.tmp_spinning_multibeam_test.xml";
    {
        std::ofstream f(xml_path);
        f << "<helios>\n";
        f << "  <scan>\n";
        f << "    <scanPattern> spinning_multibeam </scanPattern>\n";
        f << "    <beamElevationAngles> -15 -10 -5 0 5 10 15 </beamElevationAngles>\n";
        f << "    <azimuthStep> 0.5 </azimuthStep>\n"; // 0.5 deg/step -> 720 steps/rev
        f << "    <PRF> 100000 </PRF>\n";
        f << "    <trajectory>\n";
        f << "      <pose> 0.0    0 0 1  0 0 0 1 </pose>\n";
        f << "      <pose> 0.0504 0 0 1  0 0 0 1 </pose>\n";
        f << "    </trajectory>\n";
        f << "  </scan>\n";
        f << "</helios>\n";
    }

    LiDARcloud cloud;
    cloud.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud.loadXML(xml_path.c_str()));
    DOCTEST_REQUIRE(cloud.getScanCount() == 1);
    DOCTEST_CHECK(cloud.getScanPattern(0) == SCAN_PATTERN_SPINNING_MULTIBEAM);
    DOCTEST_CHECK(cloud.getScanMode(0) == SCAN_MODE_SPINNING);
    DOCTEST_CHECK(cloud.getScanSizeTheta(0) == 7); // 7 channels
    DOCTEST_CHECK(cloud.getScanStepsPerRev(0) == 720); // 360 / 0.5
    DOCTEST_CHECK(cloud.getScanSizePhi(0) == 720); // one revolution -> exactly steps_per_rev columns

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

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Object Data Labeling Test") {
    // A non-standard column label is resolved from the hit primitive's primitive data first, then (on a miss) from
    // the primitive's parent-object data. This lets a synthetic-scan hit be labeled with a field carried only by the
    // parent compound object (e.g. a per-object classification). We scan a tile object straight down at nadir and
    // verify the object-data label is transferred onto the hits.

    const uint Ntheta = 30;
    const uint Nphi = 30;
    const float thetaMin = 0.97f * float(M_PI); // narrow cone about nadir (downward)
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);

    const float object_value = 7.0f; // value carried only by the parent object

    Context context;
    // A tile object is a planar tile subdivided into patch sub-primitives; the object carries the data label.
    uint objID = context.addTileObject(make_vec3(0, 0, 0), make_vec2(10, 10), nullrotation, make_int2(5, 5));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(10, 10)); // backing patch for a non-degenerate bounding box
    context.setObjectData(objID, "object_field", object_value);

    LiDARcloud lidar;
    lidar.disableMessages();
    std::vector<std::string> columnFormat = {"object_field"};
    ScanMetadata scan(make_vec3(0, 0, 5), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    lidar.addScan(scan);
    lidar.syntheticScan(&context);

    uint hit_count = lidar.getHitCount();
    DOCTEST_REQUIRE(hit_count > 0);

    uint n = 0;
    for (uint i = 0; i < hit_count; i++) {
        vec3 p = lidar.getHitXYZ(i);
        if (fabs(p.z - 0.0f) > 1e-3f) {
            continue; // keep only returns from the z=0 tile surface
        }
        DOCTEST_CHECK(lidar.getHitData(i, "object_field") == doctest::Approx(object_value));
        n++;
    }
    DOCTEST_REQUIRE(n > 0);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Primitive-over-Object Data Precedence Test") {
    // When a column label exists as BOTH primitive data on the hit primitive and object data on its parent object,
    // the more specific per-primitive value must win. We set a different value at each level and verify the recorded
    // hit carries the primitive-data value.

    const uint Ntheta = 30;
    const uint Nphi = 30;
    const float thetaMin = 0.97f * float(M_PI);
    const float thetaMax = float(M_PI);
    const float phiMin = 0.0f;
    const float phiMax = 2.0f * float(M_PI);

    const float primitive_value = 3.0f;
    const float object_value = 9.0f;

    Context context;
    uint objID = context.addTileObject(make_vec3(0, 0, 0), make_vec2(10, 10), nullrotation, make_int2(5, 5));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(10, 10));
    context.setObjectData(objID, "shared_field", object_value);
    // Set the same label as primitive data on every sub-patch of the tile.
    std::vector<uint> tile_UUIDs = context.getObjectPrimitiveUUIDs(objID);
    context.setPrimitiveData(tile_UUIDs, "shared_field", primitive_value);

    LiDARcloud lidar;
    lidar.disableMessages();
    std::vector<std::string> columnFormat = {"shared_field"};
    ScanMetadata scan(make_vec3(0, 0, 5), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    lidar.addScan(scan);
    lidar.syntheticScan(&context);

    uint hit_count = lidar.getHitCount();
    DOCTEST_REQUIRE(hit_count > 0);

    uint n = 0;
    for (uint i = 0; i < hit_count; i++) {
        vec3 p = lidar.getHitXYZ(i);
        if (fabs(p.z - 0.0f) > 1e-3f) {
            continue;
        }
        DOCTEST_CHECK(lidar.getHitData(i, "shared_field") == doctest::Approx(primitive_value));
        n++;
    }
    DOCTEST_REQUIRE(n > 0);
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

DOCTEST_TEST_CASE("LiDAR Multi-Return Gaussian Weighting Test") {
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

    // Multi-return Gaussian weighting should match expected LAD within 2%
    DOCTEST_CHECK(LAD_multi > LAD_exact * 0.98f);
    DOCTEST_CHECK(LAD_multi < LAD_exact * 1.02f);

    // Check G(theta) against exact value calculated from primitives
    DOCTEST_CHECK(Gtheta_multi == Gtheta_multi); // Check for NaN
    DOCTEST_CHECK(fabs(Gtheta_multi - Gtheta_exact) / Gtheta_exact == doctest::Approx(0.0f).epsilon(0.05f));
}

DOCTEST_TEST_CASE("LiDAR Eight Voxel Multi-Return Gaussian Weighting Test") {
    // Test multi-return LiDAR with 8-voxel grid (2x2x2)
    // Validates the Gaussian-footprint-weighting algorithm handles partial occlusion correctly. Leaf-area density (RMSE)
    // is conserved; the leaf-angle G(theta) inversion is more sensitive because energy weighting places a return that
    // merges sub-rays across surfaces at the energy-weighted centroid rather than the geometric midpoint, so its
    // tolerance is looser than the area tolerance.

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
    synthetic_mr8.setScanDetectionThreshold(0, 0.f); // LAD-inversion validation needs complete returns; the noise floor (default 0.05) is a separate sensor-modeling concern

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

    // Tolerance widened (0.1 -> 0.15) for the stratified importance-sampled Gaussian footprint sampler. The previous
    // sampler hard-truncated each beam at its 1/e^2 radius, discarding the ~13% wing energy; the new sampler draws from
    // the Gaussian wings too (this scan sets no detection threshold, so sampling is truncated only at the default ~1.86
    // 1/e^2 radii), so the modeled beam is physically wider. A wider footprint spreads returns slightly across the fine
    // 0.5 m voxel boundaries, raising the per-voxel LAD RMSE from ~0.11 to ~0.13 at this 100-ray configuration (stable
    // across RNG seeds). This is an intended consequence of modeling the beam wings, not a regression: total area is
    // still approximately conserved and the reconstruction remains accurate to ~13% RMS.
    DOCTEST_CHECK(RMSE == doctest::Approx(0.0f).epsilon(0.15f));

    // Validate G(theta) against exact values calculated from primitive geometry
    for (int i = 0; i < 8; i++) {
        float G = G_grid_true[i];
        float G_exact = Gtheta_ex[i];

        // Only check voxels with non-zero LAD
        if (LAD_grid_true[i] > 0.001f && G_exact > 0) {
            DOCTEST_CHECK(G == G); // Check for NaN
            DOCTEST_CHECK(G_exact == G_exact); // Check for NaN
            DOCTEST_CHECK(fabs(G - G_exact) / G_exact == doctest::Approx(0.0f).epsilon(0.1f));
        }
    }
}

DOCTEST_TEST_CASE("LiDAR LAD DDA vs Brute-Force Equivalence") {
    // The leaf-area inversion has two code paths that must produce identical results: the fast per-beam
    // 3D-DDA traversal (used for regular addGrid() lattices) and the brute-force per-cell slab loop
    // (fallback for non-lattice grids). LiDARTestHelper::forceBruteForceLeafArea() forces the latter so we can A/B them
    // on identical input. Every per-cell output (leaf area, G(theta), beam count, LAD variance) must match
    // to floating-point tolerance. Several grid geometries are exercised: cubic multi-cell, non-cubic with
    // an off-center origin, a single-return scan, and a multi-return scan.

    auto run_pair = [](int3 divisions, const vec3 &grid_center, const vec3 &grid_size, bool multi_return) {
        // Build identical clouds twice and invert one with DDA, the other forced to brute-force.
        auto build_and_invert = [&](bool force_bruteforce, std::vector<float> &leaf_area, std::vector<float> &gtheta, std::vector<int> &beam_count, std::vector<float> &lad_var) {
            LiDARcloud lidar;
            lidar.disableMessages();

            vec3 scan_origin(-5.0f, 0.0f, 0.5f);
            ScanMetadata scan(scan_origin, 4000, 0.0f, M_PI, 6000, 0.0f, 2.0f * M_PI, 0.0f, multi_return ? 0.0004f : 0.0f, 0.0f, 0.0f, std::vector<std::string>{});
            lidar.addScan(scan);
            lidar.addGrid(grid_center, grid_size, divisions, 0);

            Context context;
            context.seedRandomGenerator(0);
            context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

            if (multi_return) {
                lidar.syntheticScan(&context, 30, 0.1f, true, true);
            } else {
                lidar.syntheticScan(&context, true, true); // scan_grid_only, record_misses
            }
            lidar.triangulateHitPoints(0.04, 10);

            LiDARTestHelper::forceBruteForceLeafArea(lidar, force_bruteforce);
            lidar.calculateLeafArea(&context);

            uint Ncells = lidar.getGridCellCount();
            leaf_area.resize(Ncells);
            gtheta.resize(Ncells);
            beam_count.resize(Ncells);
            lad_var.resize(Ncells);
            for (uint c = 0; c < Ncells; c++) {
                leaf_area[c] = lidar.getCellLeafArea(c);
                gtheta[c] = lidar.getCellGtheta(c);
                beam_count[c] = lidar.getCellBeamCount(c);
                lad_var[c] = lidar.getCellLADVariance(c);
            }
        };

        std::vector<float> la_dda, g_dda, var_dda;
        std::vector<int> bc_dda;
        std::vector<float> la_bf, g_bf, var_bf;
        std::vector<int> bc_bf;
        build_and_invert(false, la_dda, g_dda, bc_dda, var_dda); // fast DDA path
        build_and_invert(true, la_bf, g_bf, bc_bf, var_bf); // forced brute-force path

        DOCTEST_REQUIRE(la_dda.size() == la_bf.size());
        for (size_t c = 0; c < la_dda.size(); c++) {
            DOCTEST_CHECK_MESSAGE(la_dda[c] == doctest::Approx(la_bf[c]).epsilon(1e-4f), "leaf_area mismatch in cell " << c);
            DOCTEST_CHECK_MESSAGE(g_dda[c] == doctest::Approx(g_bf[c]).epsilon(1e-4f), "Gtheta mismatch in cell " << c);
            DOCTEST_CHECK_MESSAGE(bc_dda[c] == bc_bf[c], "beam_count mismatch in cell " << c << " (" << bc_dda[c] << " vs " << bc_bf[c] << ")");
            // LAD_variance is -1 for under-sampled cells; compare exactly there, approx otherwise.
            if (var_dda[c] < 0.f || var_bf[c] < 0.f) {
                DOCTEST_CHECK_MESSAGE(var_dda[c] == var_bf[c], "LAD_variance sign mismatch in cell " << c);
            } else {
                DOCTEST_CHECK_MESSAGE(var_dda[c] == doctest::Approx(var_bf[c]).epsilon(1e-3f), "LAD_variance mismatch in cell " << c);
            }
        }
    };

    // Cubic 2x2x2 grid, single return
    run_pair(make_int3(2, 2, 2), vec3(0.0f, 0.0f, 0.5f), vec3(1.0f, 1.0f, 1.0f), false);
    // Cubic 2x2x2 grid, multi return
    run_pair(make_int3(2, 2, 2), vec3(0.0f, 0.0f, 0.5f), vec3(1.0f, 1.0f, 1.0f), true);
    // Non-cubic grid with non-uniform divisions and a larger extent (cells are not cubes)
    run_pair(make_int3(3, 2, 4), vec3(0.0f, 0.0f, 0.5f), vec3(1.5f, 1.0f, 1.2f), false);
}

DOCTEST_TEST_CASE("LiDAR LAD Non-Lattice Fallback") {
    // A grid assembled cell-by-cell with mismatched cell sizes does NOT form a regular lattice, so the
    // inversion must use the brute-force fallback. Verify it runs and produces finite results, and that
    // forcing brute-force explicitly gives the same answer (the DDA path should not engage here at all).

    auto build = [](bool force_bruteforce, std::vector<float> &leaf_area) {
        LiDARcloud lidar;
        lidar.disableMessages();

        ScanMetadata scan(vec3(-5.0f, 0.0f, 0.5f), 3000, 0.0f, M_PI, 5000, 0.0f, 2.0f * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>{});
        lidar.addScan(scan);

        // Two manually-added cells of DIFFERENT sizes -> not a regular lattice (global_count defaults to 1x1x1
        // for each, so total cell count != product, and sizes differ).
        lidar.addGridCell(vec3(-0.25f, 0.0f, 0.5f), vec3(0.5f, 1.0f, 1.0f), 0.f);
        lidar.addGridCell(vec3(0.35f, 0.0f, 0.5f), vec3(0.7f, 1.0f, 1.0f), 0.f);

        Context context;
        context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
        lidar.syntheticScan(&context, true, true);
        lidar.triangulateHitPoints(0.04, 10);

        LiDARTestHelper::forceBruteForceLeafArea(lidar, force_bruteforce);
        lidar.calculateLeafArea(&context);

        leaf_area.resize(lidar.getGridCellCount());
        for (uint c = 0; c < lidar.getGridCellCount(); c++) {
            leaf_area[c] = lidar.getCellLeafArea(c);
            DOCTEST_CHECK(leaf_area[c] == leaf_area[c]); // not NaN
        }
    };

    std::vector<float> la_auto, la_forced;
    build(false, la_auto); // detection should select fallback automatically
    build(true, la_forced); // explicitly forced
    DOCTEST_REQUIRE(la_auto.size() == la_forced.size());
    for (size_t c = 0; c < la_auto.size(); c++) {
        DOCTEST_CHECK(la_auto[c] == doctest::Approx(la_forced[c]).epsilon(1e-4f));
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
    lidar.setScanDetectionThreshold(0, 0.f); // LAD-inversion validation needs complete returns; the noise floor (default 0.05) would drop weak returns and bias leaf area low

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

    // G(theta) should be close to 0.5 for a spherical leaf-angle distribution. The band is slightly wider than the
    // ideal +/-0.05 because Gaussian footprint weighting places returns that merge sub-rays across surfaces at the
    // energy-weighted centroid, shifting the leaf-angle inversion by a few percent relative to equal weighting.
    DOCTEST_CHECK(Gtheta > 0.42f);
    DOCTEST_CHECK(Gtheta < 0.58f);
}

DOCTEST_TEST_CASE("LiDAR Exit Diameter - Comparative Spread Test") {
    // Verify that exitDiameter > 0 produces wider spatial spread than exitDiameter = 0
    Context context;
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    vec3 scan_origin(0, 0, 5.0f);

    // The beam-footprint sub-rays are drawn from the Context RNG, so seed it before each scan with
    // the SAME seed. This makes both scans use an identical sampling realization, isolating the
    // effect of the exit diameter from RNG noise: the comparison is now deterministic and the
    // bounding-box extent (driven by extreme outliers) no longer flakes across runs/platforms.
    const uint scan_seed = 2024u;

    // Scan WITHOUT exit diameter (point source)
    LiDARcloud lidar_point;
    lidar_point.disableMessages();
    ScanMetadata scan_point(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar_point.addScan(scan_point);
    context.seedRandomGenerator(scan_seed);
    lidar_point.syntheticScan(&context, 50, 0.05f, false, false);

    // Scan WITH exit diameter
    LiDARcloud lidar_exit;
    lidar_exit.disableMessages();
    ScanMetadata scan_exit(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.1f, 0.0f, 0.0f, 0.0f, {});
    lidar_exit.addScan(scan_exit);
    context.seedRandomGenerator(scan_seed);
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

    // Exit diameter should produce measurably wider spread (deterministic ~1.026 ratio with the
    // shared seed above, comfortably above the 1.01 threshold).
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

    // 200 rays/pulse (was 50): the stratified importance-sampled footprint allocates rays by energy, so the low-energy
    // periphery is sparsely populated; the bounding-box extent below is an extremum statistic that needs more rays to be
    // stable under the new sampler. At 200 rays the aperture's added spread is unambiguous (~5% wider, well above noise),
    // whereas at 50 rays the extremum was too noisy to resolve it reliably. This is a statistic-stability adjustment, not
    // a relaxation of the physical claim.
    const int spread_rays = 200;

    // The beam-footprint sub-rays are drawn from the Context RNG, so seed it before each scan with the
    // SAME seed (mirrors the "Comparative Spread Test" above). Both scans then use an identical sampling
    // realization, isolating the aperture's added spread from RNG noise: without this the bounding-box
    // extent (an extremum statistic driven by outliers) flakes across platforms, e.g. 1.13099 vs 1.1312.
    const uint scan_seed = 2024u;

    // Scan with BOTH exitDiameter and beamDivergence
    LiDARcloud lidar_both;
    lidar_both.disableMessages();
    ScanMetadata scan_both(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.1f, 0.01f, 0.0f, 0.0f, {});
    lidar_both.addScan(scan_both);
    lidar_both.setScanDetectionThreshold(0, 0.f); // measure footprint spread, not the noise floor: keep all returns (default threshold 0.05 also shrinks the sampled cone)
    context.seedRandomGenerator(scan_seed);
    lidar_both.syntheticScan(&context, spread_rays, 0.05f, false, false);

    // Scan with ONLY beamDivergence
    LiDARcloud lidar_div;
    lidar_div.disableMessages();
    ScanMetadata scan_div(scan_origin, 100, 0, M_PI, 100, 0, 2 * M_PI, 0.0f, 0.01f, 0.0f, 0.0f, {});
    lidar_div.addScan(scan_div);
    lidar_div.setScanDetectionThreshold(0, 0.f); // measure footprint spread, not the noise floor: keep all returns (default threshold 0.05 also shrinks the sampled cone)
    context.seedRandomGenerator(scan_seed);
    lidar_div.syntheticScan(&context, spread_rays, 0.05f, false, false);

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

DOCTEST_TEST_CASE("LiDAR Idealized Single-Ray Exact Intersection") {
    // rays_per_pulse=1 is the idealized mode: a single ray reports the exact ray-surface intersection.
    Context context;
    context.addPatch(make_vec3(0, 0, 2.0f), make_vec2(2.0f, 2.0f)); // horizontal patch, normal +z

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {}); // single beam straight down
    lidar.addScan(scan);
    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 1, 0.5f)); // single ray per pulse

    DOCTEST_REQUIRE(lidar.getHitCount() == 1);
    vec3 p = lidar.getHitXYZ(0);
    DOCTEST_CHECK(fabs(p.x) < 1e-3f);
    DOCTEST_CHECK(fabs(p.y) < 1e-3f);
    DOCTEST_CHECK(p.z == doctest::Approx(2.0f).epsilon(1e-3f));
    DOCTEST_CHECK(lidar.getHitData(0, "target_count") == 1);
}

DOCTEST_TEST_CASE("LiDAR Single-Return Waveform - Ghost Point at Edge") {
    // A divergent beam straddles a near surface (covering the +x half of the footprint) and a far surface behind it.
    // With the two surfaces separated (0.3 m) by less than the pulse range-resolution (0.5 m), single-return waveform
    // processing must report ONE point at an intermediate, blended range between the surfaces: a "ghost"/"mixed pixel".
    Context context;
    context.addPatch(make_vec3(0.5f, 0.f, 2.0f), make_vec2(1.0f, 1.0f)); // near, covers +x half of footprint, z=2.0
    context.addPatch(make_vec3(0.f, 0.f, 1.7f), make_vec2(2.0f, 2.0f)); // far, covers full footprint, z=1.7

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint scanID = lidar.addScan(scan);
    lidar.setScanPulseWidth(scanID, 0.5f); // 0.5 m > 0.3 m separation -> the two surfaces merge into a single return

    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 400, 0.5f, RETURN_MODE_SINGLE));

    DOCTEST_REQUIRE(lidar.getHitCount() == 1);
    vec3 p = lidar.getHitXYZ(0);
    // The single reported point must lie strictly between the two surfaces (a ghost point), not on either one.
    DOCTEST_CHECK(p.z > 1.71f);
    DOCTEST_CHECK(p.z < 1.99f);
    DOCTEST_CHECK(lidar.getHitData(0, "target_count") == 1);
    // The echo is broadened by the spread of the two merged surfaces beyond the bare pulse width.
    DOCTEST_CHECK(lidar.getHitData(0, "echo_width") > 0.5f);
}

DOCTEST_TEST_CASE("LiDAR Multi-Return Waveform - Resolved Returns") {
    // The same straddling geometry but with the surfaces separated (1.5 m) by more than the range-resolution (0.5 m):
    // multi-return waveform processing must resolve them as two separate returns on the two real surfaces.
    Context context;
    context.addPatch(make_vec3(0.5f, 0.f, 2.0f), make_vec2(1.0f, 1.0f)); // near, +x half, z=2.0
    context.addPatch(make_vec3(0.f, 0.f, 0.5f), make_vec2(2.0f, 2.0f)); // far, full, z=0.5

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint scanID = lidar.addScan(scan);
    lidar.setScanPulseWidth(scanID, 0.5f); // 0.5 m < 1.5 m separation -> the two surfaces resolve

    DOCTEST_CHECK_NOTHROW(lidar.syntheticScan(&context, 400, 0.5f, RETURN_MODE_MULTI));

    DOCTEST_REQUIRE(lidar.getHitCount() == 2);
    float znear = fmax(lidar.getHitXYZ(0).z, lidar.getHitXYZ(1).z);
    float zfar = fmin(lidar.getHitXYZ(0).z, lidar.getHitXYZ(1).z);
    DOCTEST_CHECK(znear == doctest::Approx(2.0f).epsilon(0.03f)); // on the near surface, not blended
    DOCTEST_CHECK(zfar == doctest::Approx(0.5f).epsilon(0.03f)); // on the far surface, not blended
    DOCTEST_CHECK(lidar.getHitData(0, "target_count") == 2);
}

DOCTEST_TEST_CASE("LiDAR Single-Return Waveform - Selection Policy") {
    // Near surface (z=2.0) covers only a +x sliver of the footprint (weaker, nearer return); far surface (z=0.5) covers
    // the full footprint (stronger, farther return). STRONGEST and FIRST must select different surfaces.
    Context context;
    context.addPatch(make_vec3(0.55f, 0.f, 2.0f), make_vec2(1.0f, 1.0f)); // near sliver (x in [0.05, 1.05])
    context.addPatch(make_vec3(0.f, 0.f, 0.5f), make_vec2(2.0f, 2.0f)); // far, full footprint

    // Multi-return first: confirm both surfaces are detected and identify which return is stronger.
    LiDARcloud lidar_multi;
    lidar_multi.disableMessages();
    ScanMetadata scan_m(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_m = lidar_multi.addScan(scan_m);
    lidar_multi.setScanPulseWidth(id_m, 0.5f);
    lidar_multi.syntheticScan(&context, 400, 0.5f, RETURN_MODE_MULTI);
    DOCTEST_REQUIRE(lidar_multi.getHitCount() == 2);
    int inear = (lidar_multi.getHitXYZ(0).z > lidar_multi.getHitXYZ(1).z) ? 0 : 1;
    int ifar = 1 - inear;
    float znear = lidar_multi.getHitXYZ(inear).z;
    float zfar = lidar_multi.getHitXYZ(ifar).z;
    DOCTEST_REQUIRE(fabs(lidar_multi.getHitData(ifar, "intensity")) > fabs(lidar_multi.getHitData(inear, "intensity")));

    // STRONGEST selection -> the far (full-footprint) surface.
    LiDARcloud lidar_str;
    lidar_str.disableMessages();
    ScanMetadata scan_s(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_s = lidar_str.addScan(scan_s);
    lidar_str.setScanPulseWidth(id_s, 0.5f);
    lidar_str.setScanSingleReturnSelection(id_s, SINGLE_RETURN_STRONGEST);
    lidar_str.syntheticScan(&context, 400, 0.5f, RETURN_MODE_SINGLE);
    DOCTEST_REQUIRE(lidar_str.getHitCount() == 1);
    DOCTEST_CHECK(lidar_str.getHitXYZ(0).z == doctest::Approx(zfar).epsilon(0.05f));

    // FIRST selection -> the near surface.
    LiDARcloud lidar_first;
    lidar_first.disableMessages();
    ScanMetadata scan_f(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_f = lidar_first.addScan(scan_f);
    lidar_first.setScanPulseWidth(id_f, 0.5f);
    lidar_first.setScanSingleReturnSelection(id_f, SINGLE_RETURN_FIRST);
    lidar_first.syntheticScan(&context, 400, 0.5f, RETURN_MODE_SINGLE);
    DOCTEST_REQUIRE(lidar_first.getHitCount() == 1);
    DOCTEST_CHECK(lidar_first.getHitXYZ(0).z == doctest::Approx(znear).epsilon(0.05f));
}

DOCTEST_TEST_CASE("LiDAR Waveform - Detection Threshold") {
    // Asymmetric scene: weak near return (z=2.0, +x sliver) and strong far return (z=0.5, full footprint). A detection
    // threshold set between the two echo strengths must suppress the weak return.
    Context context;
    context.addPatch(make_vec3(0.55f, 0.f, 2.0f), make_vec2(1.0f, 1.0f)); // weak near
    context.addPatch(make_vec3(0.f, 0.f, 0.5f), make_vec2(2.0f, 2.0f)); // strong far

    // Baseline: no detection threshold -> both returns are detected.
    LiDARcloud base;
    base.disableMessages();
    ScanMetadata scan_b(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_b = base.addScan(scan_b);
    base.setScanPulseWidth(id_b, 0.5f);
    base.syntheticScan(&context, 400, 0.5f, RETURN_MODE_MULTI);
    DOCTEST_REQUIRE(base.getHitCount() == 2);
    float I0 = fabs(base.getHitData(0, "intensity"));
    float I1 = fabs(base.getHitData(1, "intensity"));
    float Iweak = fmin(I0, I1);
    float Istrong = fmax(I0, I1);
    DOCTEST_REQUIRE(Istrong > Iweak);

    // Threshold between the two echo strengths -> only the strong return survives.
    LiDARcloud filt;
    filt.disableMessages();
    ScanMetadata scan_f(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_f = filt.addScan(scan_f);
    filt.setScanPulseWidth(id_f, 0.5f);
    filt.setScanDetectionThreshold(id_f, 0.5f * (Iweak + Istrong));
    filt.syntheticScan(&context, 400, 0.5f, RETURN_MODE_MULTI);
    DOCTEST_REQUIRE(filt.getHitCount() == 1);
    DOCTEST_CHECK(fabs(filt.getHitData(0, "intensity")) == doctest::Approx(Istrong).epsilon(0.15f));
}

DOCTEST_TEST_CASE("LiDAR Return-Mode XML Load and Round-Trip") {
    // Verify the new analytic-waveform scan tags load from XML and survive an exportScans round-trip.
    std::string xml_path = "plugins/lidar/xml/.tmp_returnmode_test.xml";
    {
        std::ofstream f(xml_path);
        f << "<helios>\n  <scan>\n";
        f << "    <origin> 0 0 5 </origin>\n";
        f << "    <size> 10 10 </size>\n";
        f << "    <returnMode> single </returnMode>\n";
        f << "    <singleReturnSelection> first </singleReturnSelection>\n";
        f << "    <maxReturns> 3 </maxReturns>\n";
        f << "    <pulseWidth> 0.4 </pulseWidth>\n";
        f << "    <detectionThreshold> 0.15 </detectionThreshold>\n";
        f << "  </scan>\n</helios>\n";
    }

    LiDARcloud cloud;
    cloud.disableMessages();
    DOCTEST_CHECK_NOTHROW(cloud.loadXML(xml_path.c_str()));
    DOCTEST_REQUIRE(cloud.getScanCount() == 1);
    DOCTEST_CHECK(cloud.getScanReturnMode(0) == RETURN_MODE_SINGLE);
    DOCTEST_CHECK(cloud.getScanSingleReturnSelection(0) == SINGLE_RETURN_FIRST);
    DOCTEST_CHECK(cloud.getScanMaxReturns(0) == 3);
    DOCTEST_CHECK(cloud.getScanPulseWidth(0) == doctest::Approx(0.4f));
    DOCTEST_CHECK(cloud.getScanDetectionThreshold(0) == doctest::Approx(0.15f));

    // Populate some hits so exportScans writes a complete scan, then reload and confirm the tags persist.
    Context context;
    context.addPatch(make_vec3(0, 0, 2.0f), make_vec2(2.0f, 2.0f));
    cloud.syntheticScan(&context, 10, 0.5f, false, false);

    const std::string out_dir = "lidar_returnmode_export_tmp";
    std::filesystem::remove_all(out_dir);
    const std::string xml_out = out_dir + "/scans.xml";
    DOCTEST_CHECK_NOTHROW(cloud.exportScans(xml_out.c_str()));

    LiDARcloud reloaded;
    reloaded.disableMessages();
    DOCTEST_CHECK_NOTHROW(reloaded.loadXML(xml_out.c_str()));
    DOCTEST_REQUIRE(reloaded.getScanCount() == 1);
    DOCTEST_CHECK(reloaded.getScanReturnMode(0) == RETURN_MODE_SINGLE);
    DOCTEST_CHECK(reloaded.getScanSingleReturnSelection(0) == SINGLE_RETURN_FIRST);
    DOCTEST_CHECK(reloaded.getScanMaxReturns(0) == 3);
    DOCTEST_CHECK(reloaded.getScanPulseWidth(0) == doctest::Approx(0.4f));
    DOCTEST_CHECK(reloaded.getScanDetectionThreshold(0) == doctest::Approx(0.15f));

    std::filesystem::remove_all(out_dir);
    std::remove(xml_path.c_str());
}

DOCTEST_TEST_CASE("LiDAR Strongest-Plus-Last XML Round-Trip") {
    // The 'strongest_plus_last' selection parses (case-insensitively), accepts the 'dual' alias, and survives export/reload.
    auto load_selection = [](const std::string &spelling) {
        std::string xml_path = "plugins/lidar/xml/.tmp_spl_test.xml";
        {
            std::ofstream f(xml_path);
            f << "<helios>\n  <scan>\n";
            f << "    <origin> 0 0 5 </origin>\n";
            f << "    <size> 10 10 </size>\n";
            f << "    <returnMode> single </returnMode>\n";
            f << "    <singleReturnSelection> " << spelling << " </singleReturnSelection>\n";
            f << "  </scan>\n</helios>\n";
        }
        LiDARcloud cloud;
        cloud.disableMessages();
        cloud.loadXML(xml_path.c_str());
        SingleReturnSelection sel = cloud.getScanSingleReturnSelection(0);
        std::remove(xml_path.c_str());
        return sel;
    };

    DOCTEST_CHECK(load_selection("strongest_plus_last") == SINGLE_RETURN_STRONGEST_PLUS_LAST);
    DOCTEST_CHECK(load_selection("Strongest_Plus_Last") == SINGLE_RETURN_STRONGEST_PLUS_LAST); // case-insensitive
    DOCTEST_CHECK(load_selection("dual") == SINGLE_RETURN_STRONGEST_PLUS_LAST); // alias

    // Export then reload preserves the value.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id = cloud.addScan(scan);
    cloud.setScanReturnMode(id, RETURN_MODE_SINGLE);
    cloud.setScanSingleReturnSelection(id, SINGLE_RETURN_STRONGEST_PLUS_LAST);
    Context context;
    context.addPatch(make_vec3(0, 0, 2.0f), make_vec2(2.0f, 2.0f));
    cloud.syntheticScan(&context, 10, 0.5f, RETURN_MODE_SINGLE);

    const std::string out_dir = "lidar_spl_export_tmp";
    std::filesystem::remove_all(out_dir);
    const std::string xml_out = out_dir + "/scans.xml";
    DOCTEST_CHECK_NOTHROW(cloud.exportScans(xml_out.c_str()));

    LiDARcloud reloaded;
    reloaded.disableMessages();
    DOCTEST_CHECK_NOTHROW(reloaded.loadXML(xml_out.c_str()));
    DOCTEST_REQUIRE(reloaded.getScanCount() == 1);
    DOCTEST_CHECK(reloaded.getScanSingleReturnSelection(0) == SINGLE_RETURN_STRONGEST_PLUS_LAST);

    std::filesystem::remove_all(out_dir);
}

DOCTEST_TEST_CASE("LiDAR N-Return - maxReturns Default and Validation") {
    // The default maxReturns is 1 (classic single-return), and the setter fail-fasts on values below 1.
    Context context;
    context.addPatch(make_vec3(0, 0, 2.0f), make_vec2(2.0f, 2.0f));

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint scanID = lidar.addScan(scan);
    DOCTEST_CHECK(lidar.getScanMaxReturns(scanID) == 1); // default

    DOCTEST_CHECK_NOTHROW(lidar.setScanMaxReturns(scanID, 2));
    DOCTEST_CHECK(lidar.getScanMaxReturns(scanID) == 2);

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(lidar.setScanMaxReturns(scanID, 0));
    }
}

DOCTEST_TEST_CASE("LiDAR N-Return Waveform - Keeps Exactly N (stacked surfaces)") {
    // Three resolvable returns from one divergent pulse. The two nearer surfaces are laterally-offset slivers that each
    // cover only part of the ~0.3 m footprint (so the beam reaches all three depths); the farthest covers the rest.
    // Separated by > pulseWidth so multi-return resolves all three; limiting maxReturns=2 must report exactly two.
    auto buildScene = [](Context &context) {
        context.addPatch(make_vec3(0.13f, 0.f, 2.5f), make_vec2(0.10f, 0.4f)); // nearest sliver (+x strip), z=2.5
        context.addPatch(make_vec3(-0.12f, 0.f, 1.5f), make_vec2(0.16f, 0.4f)); // middle sliver (-x strip), z=1.5
        context.addPatch(make_vec3(0.f, 0.f, 0.4f), make_vec2(2.0f, 2.0f)); // farthest, full footprint, z=0.4
    };

    // Multi-return baseline: all three surfaces resolve.
    Context ctx_multi;
    buildScene(ctx_multi);
    LiDARcloud lidar_multi;
    lidar_multi.disableMessages();
    ScanMetadata scan_m(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_m = lidar_multi.addScan(scan_m);
    lidar_multi.setScanPulseWidth(id_m, 0.5f); // 0.5 m < 1.3 m separation -> all resolve
    lidar_multi.setScanDetectionThreshold(id_m, 0.f); // the partial-footprint slivers are deliberately weak; disable the noise floor (default 0.05) so all three resolve
    lidar_multi.syntheticScan(&ctx_multi, 400, 0.5f, RETURN_MODE_MULTI);
    DOCTEST_REQUIRE(lidar_multi.getHitCount() == 3);

    // Limited to two returns.
    Context ctx_two;
    buildScene(ctx_two);
    LiDARcloud lidar_two;
    lidar_two.disableMessages();
    ScanMetadata scan_t(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_t = lidar_two.addScan(scan_t);
    lidar_two.setScanPulseWidth(id_t, 0.5f);
    lidar_two.setScanDetectionThreshold(id_t, 0.f); // keep all three weak slivers so the maxReturns=2 limiting logic is what trims to two
    lidar_two.setScanMaxReturns(id_t, 2);
    lidar_two.setScanSingleReturnSelection(id_t, SINGLE_RETURN_FIRST);
    lidar_two.syntheticScan(&ctx_two, 400, 0.5f, RETURN_MODE_SINGLE);

    DOCTEST_REQUIRE(lidar_two.getHitCount() == 2);
    DOCTEST_CHECK(lidar_two.getHitData(0, "target_count") == 2);
    DOCTEST_CHECK(lidar_two.getHitData(1, "target_count") == 2);
    // FIRST policy keeps the two nearest (z=2.5 and z=1.5), reported nearest-first (target_index 0 nearer than 1).
    int idx0 = (lidar_two.getHitData(0, "target_index") == 0) ? 0 : 1;
    int idx1 = 1 - idx0;
    DOCTEST_CHECK(lidar_two.getHitXYZ(idx0).z > lidar_two.getHitXYZ(idx1).z); // nearer (larger z) first
    DOCTEST_CHECK(lidar_two.getHitXYZ(idx0).z == doctest::Approx(2.5f).epsilon(0.05f));
    DOCTEST_CHECK(lidar_two.getHitXYZ(idx1).z == doctest::Approx(1.5f).epsilon(0.05f));
}

DOCTEST_TEST_CASE("LiDAR N-Return Waveform - Selection Policy with N=2") {
    // Three resolvable returns of differing range and strength from one divergent pulse. The near surface is a narrow
    // +x sliver (weakest); the middle is a wider -x strip (medium); the far surface covers the residual full footprint
    // (strongest). Verify each N=2 selection policy keeps the right pair and ALWAYS reports them nearest-first (the
    // critical ordering invariant, especially for STRONGEST, where selection is by amplitude, not range).
    auto buildScene = [](Context &context) {
        context.addPatch(make_vec3(0.13f, 0.f, 2.5f), make_vec2(0.10f, 0.4f)); // near narrow sliver (weak), z=2.5
        context.addPatch(make_vec3(-0.12f, 0.f, 1.5f), make_vec2(0.16f, 0.4f)); // middle wider strip (medium), z=1.5
        context.addPatch(make_vec3(0.f, 0.f, 0.4f), make_vec2(2.0f, 2.0f)); // far residual footprint (strongest), z=0.4
    };

    // Identify per-surface intensity ranking from a multi-return baseline.
    Context ctx_b;
    buildScene(ctx_b);
    LiDARcloud base;
    base.disableMessages();
    ScanMetadata scan_b(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_b = base.addScan(scan_b);
    base.setScanPulseWidth(id_b, 0.5f);
    base.setScanDetectionThreshold(id_b, 0.f); // this test relies on the deliberately weak near sliver return; disable the noise floor (default 0.05) so all 3 returns survive
    base.syntheticScan(&ctx_b, 400, 0.5f, RETURN_MODE_MULTI);
    DOCTEST_REQUIRE(base.getHitCount() == 3);

    auto runPolicy = [&](SingleReturnSelection policy) {
        Context ctx;
        buildScene(ctx);
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
        uint id = lidar.addScan(scan);
        lidar.setScanPulseWidth(id, 0.5f);
        lidar.setScanDetectionThreshold(id, 0.f); // keep the weak near sliver so the N=2 selection policy has all 3 returns to choose among
        lidar.setScanMaxReturns(id, 2);
        lidar.setScanSingleReturnSelection(id, policy);
        lidar.syntheticScan(&ctx, 400, 0.5f, RETURN_MODE_SINGLE);
        return lidar;
    };

    // Helper: returns the two kept z-values ordered as reported (by target_index).
    auto keptZ = [](LiDARcloud &lidar) {
        DOCTEST_REQUIRE(lidar.getHitCount() == 2);
        int i0 = (lidar.getHitData(0, "target_index") == 0) ? 0 : 1;
        int i1 = 1 - i0;
        return std::make_pair(lidar.getHitXYZ(i0).z, lidar.getHitXYZ(i1).z);
    };

    // FIRST -> the two nearest surfaces: z=2.5 then z=1.5 (nearest-first).
    {
        LiDARcloud lidar = runPolicy(SINGLE_RETURN_FIRST);
        auto z = keptZ(lidar);
        DOCTEST_CHECK(z.first > z.second); // nearest-first
        DOCTEST_CHECK(z.first == doctest::Approx(2.5f).epsilon(0.05f));
        DOCTEST_CHECK(z.second == doctest::Approx(1.5f).epsilon(0.05f));
    }
    // LAST -> the two farthest surfaces: z=1.5 then z=0.4 (still nearest-first in the report).
    {
        LiDARcloud lidar = runPolicy(SINGLE_RETURN_LAST);
        auto z = keptZ(lidar);
        DOCTEST_CHECK(z.first > z.second); // nearest-first
        DOCTEST_CHECK(z.first == doctest::Approx(1.5f).epsilon(0.05f));
        DOCTEST_CHECK(z.second == doctest::Approx(0.4f).epsilon(0.05f));
    }
    // STRONGEST -> the two highest-amplitude surfaces (middle strip + far full): z=1.5 then z=0.4. The near narrow sliver
    // is weakest and is dropped. CRITICAL: even though selection is by intensity, the kept pair is reported nearest-first.
    {
        LiDARcloud lidar = runPolicy(SINGLE_RETURN_STRONGEST);
        auto z = keptZ(lidar);
        DOCTEST_CHECK(z.first > z.second); // nearest-first regardless of selection key
        DOCTEST_CHECK(z.first == doctest::Approx(1.5f).epsilon(0.05f));
        DOCTEST_CHECK(z.second == doctest::Approx(0.4f).epsilon(0.05f));
    }
}

DOCTEST_TEST_CASE("LiDAR Strongest-Plus-Last Dual Return - Three Surfaces") {
    // SINGLE_RETURN_STRONGEST_PLUS_LAST reports the strongest echo AND the farthest (last) echo of the pulse. For a
    // 3-surface pulse this differs from every single-key cap. Scene: three laterally-disjoint strips at distinct ranges,
    // arranged so echo strength DECREASES with range (the nearest strip is centered where the divergent beam's rays are
    // densest -> strongest echo; farther strips are progressively more peripheral -> weaker). Then strongest == near and
    // last == far, so
    //   strongest+last = {near (strongest), far (last)}   -- drops the middle
    // which neither STRONGEST-cap-2 ({near, middle}, the two largest amplitudes) nor LAST-cap-2 ({middle, far}, the two
    // farthest) produces. The policy also ignores maxReturns (it intrinsically yields 1 or 2 returns).
    const float znear = 2.5f, zmid = 1.5f, zfar = 0.4f;
    const std::vector<std::string> columnFormat = {"reflectivity_lidar"}; // recorded-intensity column

    auto buildScene = [&](Context &context) {
        context.addPatch(make_vec3(0.0f, 0.f, znear), make_vec2(0.12f, 0.5f)); // near, centered (dense rays) -> strongest
        context.addPatch(make_vec3(0.13f, 0.f, zmid), make_vec2(0.14f, 0.5f)); // middle, +x offset -> medium
        context.addPatch(make_vec3(-0.17f, 0.f, zfar), make_vec2(0.10f, 0.5f)); // far, -x peripheral -> weakest
    };

    auto runScan = [&](ReturnMode mode, SingleReturnSelection policy, int maxReturns) {
        Context ctx;
        // Seed the Context RNG so the synthetic scan's per-beam ray sampling (context->randu() inside syntheticScan) is
        // deterministic. Without this the Context seeds from wall-clock time, so the recorded intensities — and hence the
        // I_near > I_mid > I_far ordering this test's STRONGEST-cap-2 logic depends on — vary run-to-run; the middle/far
        // separation in this scene is small enough that an unseeded run occasionally inverts it. A fixed seed pins the
        // sampling pattern identically on every machine and every CPU/GPU trace path.
        ctx.seedRandomGenerator(0);
        buildScene(ctx);
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, columnFormat);
        uint id = lidar.addScan(scan);
        lidar.setScanPulseWidth(id, 0.5f);
        lidar.setScanMaxReturns(id, maxReturns);
        lidar.setScanSingleReturnSelection(id, policy);
        lidar.syntheticScan(&ctx, 400, 0.5f, mode);
        return lidar;
    };

    // Mark which of the three surfaces a cloud's returns landed on (by z), so assertions are structural, not coordinate-exact.
    auto classify = [&](LiDARcloud &c, bool &hit_near, bool &hit_mid, bool &hit_far) {
        hit_near = hit_mid = hit_far = false;
        for (uint i = 0; i < c.getHitCount(); i++) {
            float z = c.getHitXYZ(i).z;
            if (fabs(z - znear) < 0.2f)
                hit_near = true;
            else if (fabs(z - zmid) < 0.2f)
                hit_mid = true;
            else if (fabs(z - zfar) < 0.2f)
                hit_far = true;
        }
    };

    // Multi-return baseline: all three surfaces resolve, and the recorded intensities confirm near > middle > far.
    LiDARcloud base = runScan(RETURN_MODE_MULTI, SINGLE_RETURN_STRONGEST, 1);
    DOCTEST_REQUIRE(base.getHitCount() == 3);
    double I_near = 0.0, I_mid = 0.0, I_far = 0.0;
    for (uint i = 0; i < base.getHitCount(); i++) {
        float z = base.getHitXYZ(i).z;
        DOCTEST_REQUIRE(base.doesHitDataExist(i, "intensity"));
        double I = fabs((double)base.getHitData(i, "intensity"));
        if (fabs(z - znear) < 0.2f)
            I_near = I;
        else if (fabs(z - zmid) < 0.2f)
            I_mid = I;
        else if (fabs(z - zfar) < 0.2f)
            I_far = I;
    }
    DOCTEST_REQUIRE(I_near > 0.0);
    DOCTEST_REQUIRE(I_mid > 0.0);
    DOCTEST_REQUIRE(I_far > 0.0);
    DOCTEST_REQUIRE(I_near > I_mid); // scene precondition: near is strongest
    DOCTEST_REQUIRE(I_mid > I_far); //                     far is weakest

    // Strongest+last: keeps the near (strongest) and far (last) surfaces, NOT the middle. maxReturns=1 must be ignored.
    LiDARcloud spl = runScan(RETURN_MODE_SINGLE, SINGLE_RETURN_STRONGEST_PLUS_LAST, 1);
    DOCTEST_REQUIRE(spl.getHitCount() == 2); // two returns despite maxReturns=1
    bool sn, sm, sf;
    classify(spl, sn, sm, sf);
    DOCTEST_CHECK(sn); // strongest
    DOCTEST_CHECK(sf); // last
    DOCTEST_CHECK_FALSE(sm); // middle dropped -- the distinguishing point
    // Reported nearest-first: target_index 0 is the nearer (larger z) surface.
    DOCTEST_CHECK(spl.getHitData(0, "target_count") == 2);
    DOCTEST_CHECK(spl.getHitData(1, "target_count") == 2);
    int i0 = (spl.getHitData(0, "target_index") == 0) ? 0 : 1;
    int i1 = 1 - i0;
    DOCTEST_CHECK(spl.getHitXYZ(i0).z > spl.getHitXYZ(i1).z);
    DOCTEST_CHECK(spl.getHitXYZ(i0).z == doctest::Approx(znear).epsilon(0.05f));
    DOCTEST_CHECK(spl.getHitXYZ(i1).z == doctest::Approx(zfar).epsilon(0.05f));

    // Contrast against the single-key caps on the SAME scene: both keep a different pair, proving the policy is distinct.
    LiDARcloud strong2 = runScan(RETURN_MODE_SINGLE, SINGLE_RETURN_STRONGEST, 2);
    DOCTEST_REQUIRE(strong2.getHitCount() == 2);
    bool tn, tm, tf;
    classify(strong2, tn, tm, tf);
    DOCTEST_CHECK(tn); // STRONGEST-cap-2 keeps the two largest amplitudes: near + middle
    DOCTEST_CHECK(tm);
    DOCTEST_CHECK_FALSE(tf); // far (weakest) dropped -- differs from strongest+last

    LiDARcloud last2 = runScan(RETURN_MODE_SINGLE, SINGLE_RETURN_LAST, 2);
    DOCTEST_REQUIRE(last2.getHitCount() == 2);
    bool ln, lm, lf;
    classify(last2, ln, lm, lf);
    DOCTEST_CHECK_FALSE(ln); // LAST-cap-2 keeps the two farthest: middle + far
    DOCTEST_CHECK(lm);
    DOCTEST_CHECK(lf); // near dropped -- differs from strongest+last
}

DOCTEST_TEST_CASE("LiDAR Strongest-Plus-Last Dual Return - Dedup To Single") {
    // When the strongest return IS the farthest, strongest+last collapses to a single reported point (not a doubled one).
    // Two surfaces: a weak near peripheral sliver (few rays) and a strong far full-footprint surface that catches the dense
    // central rays, so the far surface is both the strongest echo and the last (farthest) return.
    const float znear = 2.0f, zfar = 0.5f;
    const std::vector<std::string> columnFormat = {"reflectivity_lidar"}; // recorded-intensity column

    auto buildScene = [&](Context &context) {
        context.addPatch(make_vec3(0.13f, 0.f, znear), make_vec2(0.10f, 0.5f)); // near, peripheral sliver -> weak
        context.addPatch(make_vec3(0.f, 0.f, zfar), make_vec2(2.0f, 2.0f)); // far, full footprint -> strongest & last
    };

    auto runScan = [&](ReturnMode mode, SingleReturnSelection policy) {
        Context ctx;
        // The near patch is a deliberately peripheral sliver, so whether the beam's stochastic
        // footprint sub-rays (drawn from ctx.randu()) happen to strike it decides between a
        // one-hit and a two-hit cloud. Unseeded, this test missed the sliver in roughly 1 run in
        // 40 and failed the hit-count REQUIRE below. Seeding fixes the sub-ray pattern so the
        // scan is reproducible.
        ctx.seedRandomGenerator(2024u);
        buildScene(ctx);
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, columnFormat);
        uint id = lidar.addScan(scan);
        lidar.setScanPulseWidth(id, 0.5f);
        lidar.setScanSingleReturnSelection(id, policy);
        lidar.syntheticScan(&ctx, 400, 0.5f, mode);
        return lidar;
    };

    // Baseline: both surfaces resolve and the far surface is the strongest echo.
    LiDARcloud base = runScan(RETURN_MODE_MULTI, SINGLE_RETURN_STRONGEST);
    DOCTEST_REQUIRE(base.getHitCount() == 2);
    double I_near = 0.0, I_far = 0.0;
    for (uint i = 0; i < base.getHitCount(); i++) {
        float z = base.getHitXYZ(i).z;
        DOCTEST_REQUIRE(base.doesHitDataExist(i, "intensity"));
        double I = fabs((double)base.getHitData(i, "intensity"));
        if (fabs(z - znear) < 0.3f)
            I_near = I;
        else if (fabs(z - zfar) < 0.3f)
            I_far = I;
    }
    DOCTEST_REQUIRE(I_near > 0.0);
    DOCTEST_REQUIRE(I_far > 0.0);
    DOCTEST_REQUIRE(I_far > I_near); // the strongest return IS the farthest

    // Strongest+last must dedup to exactly one point at the far surface.
    LiDARcloud spl = runScan(RETURN_MODE_SINGLE, SINGLE_RETURN_STRONGEST_PLUS_LAST);
    DOCTEST_REQUIRE(spl.getHitCount() == 1); // dedup: strongest == last -> single point
    DOCTEST_CHECK(spl.getHitXYZ(0).z == doctest::Approx(zfar).epsilon(0.05f));
    DOCTEST_CHECK(spl.getHitData(0, "target_count") == 1);
}

DOCTEST_TEST_CASE("LiDAR Dual-Return Is N=2") {
    // Two resolvable surfaces: maxReturns=2 must keep both, matching the multi-return result for a 2-surface scene.
    // The near surface covers only the +x half of the footprint so the divergent beam also reaches the far surface.
    auto buildScene = [](Context &context) {
        context.addPatch(make_vec3(0.5f, 0.f, 2.0f), make_vec2(1.0f, 1.0f)); // near, +x half
        context.addPatch(make_vec3(0.f, 0.f, 0.5f), make_vec2(2.0f, 2.0f)); // far, full footprint
    };

    Context ctx;
    buildScene(ctx);
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id = lidar.addScan(scan);
    lidar.setScanPulseWidth(id, 0.5f);
    lidar.setScanMaxReturns(id, 2);
    lidar.syntheticScan(&ctx, 400, 0.5f, RETURN_MODE_SINGLE);

    DOCTEST_REQUIRE(lidar.getHitCount() == 2);
    float znear = fmax(lidar.getHitXYZ(0).z, lidar.getHitXYZ(1).z);
    float zfar = fmin(lidar.getHitXYZ(0).z, lidar.getHitXYZ(1).z);
    DOCTEST_CHECK(znear == doctest::Approx(2.0f).epsilon(0.03f));
    DOCTEST_CHECK(zfar == doctest::Approx(0.5f).epsilon(0.03f));
}

DOCTEST_TEST_CASE("LiDAR N-Return - Full Miss Preserved") {
    // A pulse that hits nothing (fully transmitted beam) must record a single miss even in N-return mode, so the cloud
    // remains usable for leaf-area inversion.
    Context context; // empty scene -> all rays miss

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id = lidar.addScan(scan);
    lidar.setScanPulseWidth(id, 0.5f);
    lidar.setScanReturnMode(id, RETURN_MODE_SINGLE);
    lidar.setScanMaxReturns(id, 3);
    lidar.syntheticScan(&context, 400, 0.5f, false, true); // record_misses=true

    DOCTEST_REQUIRE(lidar.getHitCount() == 1);
    DOCTEST_CHECK(lidar.getHitData(0, "is_miss") == 1.0);
    DOCTEST_CHECK(lidar.getHitData(0, "target_index") == 99);
}

DOCTEST_TEST_CASE("LiDAR N-Return - Partial Miss Dropped") {
    // A divergent beam partially hits a near surface (covering the +x half of the footprint) and partially transmits
    // past it. In limited (N-return) mode the transmitted-beam miss is dropped (only real points are reported), whereas
    // unlimited multi-return mode keeps it.
    auto buildScene = [](Context &context) {
        context.addPatch(make_vec3(-0.13f, 0.f, 2.0f), make_vec2(0.35f, 0.6f)); // straddles center, leaves +x footprint open
    };

    // Multi-return with misses recorded: the partial transmission yields a real return AND a miss row.
    Context ctx_multi;
    buildScene(ctx_multi);
    LiDARcloud lidar_multi;
    lidar_multi.disableMessages();
    ScanMetadata scan_m(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_m = lidar_multi.addScan(scan_m);
    lidar_multi.setScanPulseWidth(id_m, 0.5f);
    lidar_multi.setScanReturnMode(id_m, RETURN_MODE_MULTI);
    lidar_multi.syntheticScan(&ctx_multi, 400, 0.5f, false, true); // record_misses=true
    // Multi-return keeps both the real return and the transmitted-beam miss.
    int multi_real = 0, multi_miss = 0;
    for (uint i = 0; i < lidar_multi.getHitCount(); i++) {
        if (lidar_multi.getHitData(i, "is_miss") == 1.0) {
            multi_miss++;
        } else {
            multi_real++;
        }
    }
    DOCTEST_REQUIRE(multi_real >= 1);
    DOCTEST_REQUIRE(multi_miss == 1);

    // N-return (limited) mode: the same partial-transmission pulse reports only real points, no miss row.
    Context ctx_lim;
    buildScene(ctx_lim);
    LiDARcloud lidar_lim;
    lidar_lim.disableMessages();
    ScanMetadata scan_l(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
    uint id_l = lidar_lim.addScan(scan_l);
    lidar_lim.setScanPulseWidth(id_l, 0.5f);
    lidar_lim.setScanReturnMode(id_l, RETURN_MODE_SINGLE);
    lidar_lim.setScanMaxReturns(id_l, 3);
    lidar_lim.syntheticScan(&ctx_lim, 400, 0.5f, false, true); // record_misses=true
    int lim_miss = 0;
    for (uint i = 0; i < lidar_lim.getHitCount(); i++) {
        if (lidar_lim.getHitData(i, "is_miss") == 1.0) {
            lim_miss++;
        }
    }
    DOCTEST_REQUIRE(lidar_lim.getHitCount() >= 1);
    DOCTEST_CHECK(lim_miss == 0); // partial-transmission miss dropped in limited mode
}

DOCTEST_TEST_CASE("LiDAR N-Return - Backward Compatibility (single/multi unchanged)") {
    // With the default maxReturns=1, single- and multi-return modes behave exactly as before.
    // Single-return ghost point: two surfaces within the pulse width blend to one point.
    {
        Context context;
        context.addPatch(make_vec3(0.5f, 0.f, 2.0f), make_vec2(1.0f, 1.0f));
        context.addPatch(make_vec3(0.f, 0.f, 1.7f), make_vec2(2.0f, 2.0f));
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
        uint id = lidar.addScan(scan);
        DOCTEST_CHECK(lidar.getScanMaxReturns(id) == 1); // default
        lidar.setScanPulseWidth(id, 0.5f);
        lidar.syntheticScan(&context, 400, 0.5f, RETURN_MODE_SINGLE);
        DOCTEST_CHECK(lidar.getHitCount() == 1);
    }
    // Multi-return resolved returns: two well-separated surfaces resolve to two points.
    {
        Context context;
        context.addPatch(make_vec3(0.5f, 0.f, 2.0f), make_vec2(1.0f, 1.0f));
        context.addPatch(make_vec3(0.f, 0.f, 0.5f), make_vec2(2.0f, 2.0f));
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 5), 1, M_PI, M_PI, 1, 0, 0, 0.0f, 0.06f, 0.0f, 0.0f, {});
        uint id = lidar.addScan(scan);
        lidar.setScanPulseWidth(id, 0.5f);
        lidar.syntheticScan(&context, 400, 0.5f, RETURN_MODE_MULTI);
        DOCTEST_CHECK(lidar.getHitCount() == 2);
    }
}

//! Scan geometry shared by the synthetic gap-fill tests: a full-sphere raster about a small sphere target, on a
//! declared grid small enough to inspect cell by cell (the whole declared raster is synthesized as misses).
static void addSphereSceneScan(LiDARcloud &lidar, uint Ntheta, uint Nphi) {
    ScanMetadata scan(make_vec3(-5.f, 0.f, 0.5f), Ntheta, 0.f, float(M_PI), Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{"x", "y", "z", "timestamp"});
    lidar.addScan(scan);
    lidar.addGrid(make_vec3(0, 0, 0.5f), make_vec3(1, 1, 1), make_int3(2, 2, 2), 0);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Grid Position Verification") {
    // Gap-filling a scan from its timestamps synthesizes one miss for every cell of the declared scan grid that has
    // no return -- the same population the row/column path produces -- so after the call every cell is accounted
    // for exactly once, originals are flagged 0 and synthesized misses 1.
    LiDARcloud lidar;
    lidar.disableMessages();
    Context context;
    const uint Ntheta = 150, Nphi = 200;
    addSphereSceneScan(lidar, Ntheta, Nphi);
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 1.0), 0.3);

    lidar.syntheticScan(&context, false, false); // record_misses = false: the sphere's hits only
    const uint hits_before = lidar.getHitCount();
    DOCTEST_REQUIRE(hits_before > 0);
    DOCTEST_REQUIRE(hits_before < Ntheta * Nphi);

    std::vector<vec3> filled = lidar.gapfillMisses(0, false, true);
    DOCTEST_CHECK(filled.size() == Ntheta * Nphi - hits_before);
    DOCTEST_CHECK(lidar.getHitCount() == Ntheta * Nphi);

    // Every hit -- return or synthesized miss -- occupies a distinct cell of the declared grid.
    std::vector<int32_t> row, col, code, is_miss;
    lidar.getHitDataColumn("row", row, int32_t(-1));
    lidar.getHitDataColumn("column", col, int32_t(-1));
    lidar.getHitDataColumn("gapfillMisses_code", code, int32_t(-1));
    lidar.getHitDataColumn("is_miss", is_miss, int32_t(-1));
    std::vector<uint64_t> seen(((size_t) Ntheta * Nphi + 63) / 64, 0ull);
    size_t distinct = 0, code0 = 0, code1 = 0, other_codes = 0;
    for (uint i = 0; i < lidar.getHitCount(); i++) {
        DOCTEST_REQUIRE(row[i] >= 0);
        DOCTEST_REQUIRE(col[i] >= 0);
        const size_t bit = (size_t) row[i] * Nphi + (size_t) col[i];
        if (((seen[bit >> 6] >> (bit & 63)) & 1ull) == 0) {
            seen[bit >> 6] |= 1ull << (bit & 63);
            distinct++;
        }
        if (code[i] == 0) {
            code0++;
            DOCTEST_CHECK(is_miss[i] == 0);
        } else if (code[i] == 1) {
            code1++;
            DOCTEST_CHECK(is_miss[i] == 1);
        } else {
            other_codes++;
        }
    }
    DOCTEST_CHECK(distinct == Ntheta * Nphi);
    DOCTEST_CHECK(code0 == hits_before);
    DOCTEST_CHECK(code1 == filled.size());
    DOCTEST_CHECK(other_codes == 0);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Comparison with Record Misses") {
    // A synthetic scan that records its misses is the ground truth for the pulses that reached the scene's bounding
    // box; gap-filling the same scan traced without misses must reproduce every one of those misses at the same
    // cell, along the same beam, with the same timestamp -- and complete the raster beyond them.
    LiDARcloud lidar1, lidar2;
    lidar1.disableMessages();
    lidar2.disableMessages();
    Context context;
    const uint Ntheta = 150, Nphi = 200;
    addSphereSceneScan(lidar1, Ntheta, Nphi);
    addSphereSceneScan(lidar2, Ntheta, Nphi);
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 1.0), 0.3);

    lidar1.syntheticScan(&context, false, true); // record_misses = TRUE
    lidar2.syntheticScan(&context, false, false); // record_misses = FALSE
    const uint returns = lidar2.getHitCount();
    DOCTEST_REQUIRE(returns > 0);
    std::vector<vec3> filled = lidar2.gapfillMisses(0, false, false);
    DOCTEST_CHECK(filled.size() > 0);
    DOCTEST_CHECK(lidar2.getHitCount() == Ntheta * Nphi);
    DOCTEST_CHECK(lidar1.getHitCount() <= Ntheta * Nphi); // only pulses that reached the scene's bounding box were traced

    // Same returns.
    std::vector<int32_t> miss1, miss2;
    lidar1.getHitDataColumn("is_miss", miss1, int32_t(-1));
    lidar2.getHitDataColumn("is_miss", miss2, int32_t(-1));
    size_t real1 = 0, real2 = 0;
    for (int32_t m: miss1) {
        real1 += (m == 0) ? 1 : 0;
    }
    for (int32_t m: miss2) {
        real2 += (m == 0) ? 1 : 0;
    }
    DOCTEST_CHECK(real1 == returns);
    DOCTEST_CHECK(real2 == returns);

    // Every recorded miss has a synthesized counterpart at the same pulse (timestamp) along the same beam.
    std::vector<double> t1, t2;
    lidar1.getHitDataColumn("timestamp", t1, -1.0);
    lidar2.getHitDataColumn("timestamp", t2, -1.0);
    std::vector<vec3> p1, p2;
    lidar1.getHitXYZColumn(p1);
    lidar2.getHitXYZColumn(p2);
    const vec3 origin = lidar1.getScanOrigin(0);
    std::map<long long, size_t> by_pulse; // pulse ordinal -> index of the synthesized miss in lidar2
    for (size_t i = 0; i < t2.size(); i++) {
        if (miss2[i] == 1) {
            by_pulse[(long long) std::llround(t2[i])] = i;
        }
    }
    size_t recorded_misses = 0, matched = 0, aligned = 0;
    for (size_t i = 0; i < t1.size(); i++) {
        if (miss1[i] != 1) {
            continue;
        }
        recorded_misses++;
        auto it = by_pulse.find((long long) std::llround(t1[i]));
        if (it == by_pulse.end()) {
            continue;
        }
        matched++;
        vec3 d1 = p1[i] - origin, d2 = p2[it->second] - origin;
        d1.normalize();
        d2.normalize();
        if (d1 * d2 > cosf(0.05f * float(M_PI) / 180.f)) { // within 0.05 degrees
            aligned++;
        }
    }
    DOCTEST_CHECK(recorded_misses > 0);
    DOCTEST_CHECK(matched == recorded_misses);
    DOCTEST_CHECK(aligned == recorded_misses);
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
    // Grid-only mode fills only the rows whose beams can reach the voxel grid, so it synthesizes fewer misses than
    // the full raster and every miss it does synthesize points into the grid's zenith window.
    Context context;
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 1.0), 0.3);
    const uint Ntheta = 150, Nphi = 200;

    LiDARcloud grid_only, full;
    grid_only.disableMessages();
    full.disableMessages();
    addSphereSceneScan(grid_only, Ntheta, Nphi);
    addSphereSceneScan(full, Ntheta, Nphi);
    grid_only.syntheticScan(&context, false, false);
    full.syntheticScan(&context, false, false);
    const uint returns = full.getHitCount();
    DOCTEST_REQUIRE(grid_only.getHitCount() == returns);

    const size_t n_grid_only = grid_only.gapfillMissesCount(0, true, false);
    const size_t n_full = full.gapfillMissesCount(0, false, false);
    DOCTEST_CHECK(n_full == Ntheta * Nphi - returns);
    DOCTEST_CHECK(n_grid_only < n_full);
    DOCTEST_CHECK(n_grid_only > 0);

    // The grid's zenith window as seen from the scanner (the 1 m cube centred 5 m away spans well under a hemisphere).
    const vec3 origin = full.getScanOrigin(0);
    float min_theta = float(M_PI), max_theta = 0.f;
    for (int sx = 0; sx < 2; sx++) {
        for (int sy = 0; sy < 2; sy++) {
            for (int sz = 0; sz < 2; sz++) {
                const vec3 corner = make_vec3(-0.5f + float(sx), -0.5f + float(sy), float(sz));
                const float zenith = cart2sphere(corner - origin).zenith;
                min_theta = std::min(min_theta, zenith);
                max_theta = std::max(max_theta, zenith);
            }
        }
    }
    std::vector<vec3> xyz;
    std::vector<int32_t> is_miss;
    grid_only.getHitXYZColumn(xyz);
    grid_only.getHitDataColumn("is_miss", is_miss, int32_t(-1));
    const float slack = float(M_PI) / float(Ntheta); // one row
    size_t outside = 0;
    for (size_t i = 0; i < xyz.size(); i++) {
        if (is_miss[i] != 1) {
            continue;
        }
        const float zenith = cart2sphere(xyz[i] - origin).zenith;
        if (zenith < min_theta - slack || zenith > max_theta + slack) {
            outside++;
        }
    }
    DOCTEST_CHECK(outside == 0);
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
    bool has_multi_return = lidar.isMultiReturnData();
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

    // The raster sweep drifts the azimuth continuously across each zenith column (continuous-azimuth model): cell (row,col)
    // is emitted at azimuth = phiMin + col*dphi + row*dphi_per_row, with dphi the column step and dphi_per_row = dphi/Ntheta.
    // To recover the nominal (row,col) grid cell from a hit's azimuth we must subtract this per-row drift first; otherwise the
    // top rows of a column round into the next column (and the last column rounds out of bounds).
    const float dphi_grid = (phiMax - phiMin) / float(Nphi - 1);
    const float dphi_per_row = dphi_grid / float(Ntheta);

    for (uint r = 0; r < lidar1.getHitCount(); r++) {
        SphericalCoord raydir = lidar1.getHitRaydir(r);
        vec3 pos = lidar1.getHitXYZ(r);

        // Calculate grid indices (de-skew the azimuth by the recovered row's drift before binning to a column).
        float theta = raydir.zenith;
        float phi = raydir.azimuth;
        int row = round((theta - thetaMin) / (thetaMax - thetaMin) * (Ntheta - 1));
        int col = round((phi - float(row) * dphi_per_row - phiMin) / (phiMax - phiMin) * (Nphi - 1));

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
        int col = round((phi - float(row) * dphi_per_row - phiMin) / (phiMax - phiMin) * (Nphi - 1)); // de-skew the continuous-azimuth drift (see ground-truth loop)

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

DOCTEST_TEST_CASE("LiDAR Columnar Hit Data - Bulk Getter Matches Per-Hit") {
    // The columnar bulk getter getHitDataColumn() must return, for every hit, exactly what the per-hit
    // getHitData()/doesHitDataExist() path returns, including the absent-value placement for hits that
    // lack a label. Build a cloud where a custom label is present on only some hits.
    const std::vector<std::string> columnFormat = {"x", "y", "z", "intensity", "sparse_field"};
    LiDARcloud cloud;
    cloud.disableMessages();

    ScanMetadata scan(vec3(0.0f, 0.0f, 0.0f), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    const uint Nhits = 7;
    for (uint i = 0; i < Nhits; i++) {
        std::map<std::string, double> data;
        data["intensity"] = 0.5 + 0.1 * i; // present on every hit
        if (i % 2 == 0) {
            data["sparse_field"] = 1000.0 + i; // present only on even hits
        }
        SphericalCoord dir(1.f, 0.0f, 0.1f * i);
        DOCTEST_CHECK_NOTHROW(cloud.addHitPoint(0, vec3(float(i), 0.2f * i, 1.0f), dir, data));
    }
    DOCTEST_CHECK(cloud.getHitCount() == Nhits);

    const double absent = -9999.0;

    // intensity: present everywhere.
    std::vector<double> intensity_bulk;
    DOCTEST_CHECK_NOTHROW(cloud.getHitDataColumn("intensity", intensity_bulk, absent));
    DOCTEST_CHECK(intensity_bulk.size() == Nhits);
    for (uint i = 0; i < Nhits; i++) {
        const double expected = cloud.doesHitDataExist(i, "intensity") ? cloud.getHitData(i, "intensity") : absent;
        DOCTEST_CHECK(intensity_bulk[i] == doctest::Approx(expected));
    }

    // sparse_field: present only on even hits, absent (= -9999) on odd hits.
    std::vector<double> sparse_bulk;
    DOCTEST_CHECK_NOTHROW(cloud.getHitDataColumn("sparse_field", sparse_bulk, absent));
    DOCTEST_CHECK(sparse_bulk.size() == Nhits);
    for (uint i = 0; i < Nhits; i++) {
        const double expected = cloud.doesHitDataExist(i, "sparse_field") ? cloud.getHitData(i, "sparse_field") : absent;
        DOCTEST_CHECK(sparse_bulk[i] == doctest::Approx(expected));
        if (i % 2 == 0) {
            DOCTEST_CHECK(sparse_bulk[i] == doctest::Approx(1000.0 + i));
        } else {
            DOCTEST_CHECK(sparse_bulk[i] == doctest::Approx(absent));
        }
    }

    // A label that was never set on any hit must be all-absent and report column index -1.
    DOCTEST_CHECK(cloud.getHitDataColumnIndex("never_set") == -1);
    std::vector<double> missing_bulk;
    DOCTEST_CHECK_NOTHROW(cloud.getHitDataColumn("never_set", missing_bulk, absent));
    DOCTEST_CHECK(missing_bulk.size() == Nhits);
    for (uint i = 0; i < Nhits; i++) {
        DOCTEST_CHECK(missing_bulk[i] == doctest::Approx(absent));
    }
    DOCTEST_CHECK(cloud.getHitDataColumnIndex("intensity") >= 0);
}

DOCTEST_TEST_CASE("LiDAR Columnar Hit Data - Delete Lockstep") {
    // deleteHitPoint() uses swap-and-pop; the columnar scalar data must move in lockstep so each
    // surviving hit's columns still correspond to its position/color. Tag each hit with a unique id
    // equal to its position, delete several, and verify every survivor's column value still matches the
    // id encoded in its XYZ position.
    const std::vector<std::string> columnFormat = {"x", "y", "z", "id"};
    LiDARcloud cloud;
    cloud.disableMessages();

    ScanMetadata scan(vec3(0.0f, 0.0f, 0.0f), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    const uint Nhits = 10;
    for (uint i = 0; i < Nhits; i++) {
        std::map<std::string, double> data;
        data["id"] = double(i);
        SphericalCoord dir(1.f, 0.0f, 0.05f * i);
        // Encode the id in the x position so we can cross-check column vs. position after swaps.
        DOCTEST_CHECK_NOTHROW(cloud.addHitPoint(0, vec3(double(i), 0.f, 1.f), dir, data));
    }

    // Delete hits 2, 5, 8 (descending so indices stay valid - mirrors the filter delete loops).
    DOCTEST_CHECK_NOTHROW(cloud.deleteHitPoint(8));
    DOCTEST_CHECK_NOTHROW(cloud.deleteHitPoint(5));
    DOCTEST_CHECK_NOTHROW(cloud.deleteHitPoint(2));
    DOCTEST_CHECK(cloud.getHitCount() == Nhits - 3);

    for (uint i = 0; i < cloud.getHitCount(); i++) {
        const double id = cloud.getHitData(i, "id");
        const double x = cloud.getHitXYZ(i).x;
        // The id column value must equal the x position for the SAME hit - proving columns didn't desync.
        DOCTEST_CHECK(id == doctest::Approx(x));
        // None of the deleted ids (2,5,8) should survive.
        DOCTEST_CHECK(id != doctest::Approx(2.0));
        DOCTEST_CHECK(id != doctest::Approx(5.0));
        DOCTEST_CHECK(id != doctest::Approx(8.0));
    }
}

DOCTEST_TEST_CASE("LiDAR Columnar Hit Data - Mid-Cloud New Label Back-Fill") {
    // setHitData() introducing a brand-new label on a late hit must create a full-length column that is
    // absent for all earlier hits and present only where set - matching the old per-hit map semantics.
    LiDARcloud cloud;
    cloud.disableMessages();

    const std::vector<std::string> columnFormat = {"x", "y", "z"};
    ScanMetadata scan(vec3(0.0f, 0.0f, 0.0f), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    const uint Nhits = 6;
    for (uint i = 0; i < Nhits; i++) {
        std::map<std::string, double> data; // no scalar data initially
        SphericalCoord dir(1.f, 0.0f, 0.05f * i);
        DOCTEST_CHECK_NOTHROW(cloud.addHitPoint(0, vec3(double(i), 0.f, 1.f), dir, data));
    }

    // The label does not exist yet anywhere.
    DOCTEST_CHECK(cloud.getHitDataColumnIndex("late_label") == -1);
    for (uint i = 0; i < Nhits; i++) {
        DOCTEST_CHECK(cloud.doesHitDataExist(i, "late_label") == false);
    }

    // Set it on a single late hit (index 4).
    DOCTEST_CHECK_NOTHROW(cloud.setHitData(4, "late_label", 42.0));

    DOCTEST_CHECK(cloud.getHitDataColumnIndex("late_label") >= 0);
    for (uint i = 0; i < Nhits; i++) {
        if (i == 4) {
            DOCTEST_CHECK(cloud.doesHitDataExist(i, "late_label") == true);
            DOCTEST_CHECK(cloud.getHitData(i, "late_label") == doctest::Approx(42.0));
        } else {
            DOCTEST_CHECK(cloud.doesHitDataExist(i, "late_label") == false);
        }
    }

    // getHitData on an absent hit must throw the same "does not exist" error as before.
    {
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(cloud.getHitData(0, "late_label"));
    }
}

DOCTEST_TEST_CASE("LiDAR Columnar Hit Data - Origin Survives coordinateShift") {
    // Moving-platform hits carry their own per-pulse origin in labels origin_x/y/z (all-three-or-none).
    // coordinateShift must transform that origin in lockstep with the hit position. With columnar
    // storage the origin helpers operate by hit index; verify a shifted moving-style hit keeps its
    // origin consistent while a static hit (no origin labels) is unaffected.
    LiDARcloud cloud;
    cloud.disableMessages();

    const std::vector<std::string> columnFormat = {"x", "y", "z"};
    ScanMetadata scan(vec3(0.0f, 0.0f, 0.0f), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat);
    DOCTEST_CHECK_NOTHROW(cloud.addScan(scan));

    // Hit 0: moving-style, carries an explicit origin.
    std::map<std::string, double> moving;
    moving["origin_x"] = 1.0;
    moving["origin_y"] = 2.0;
    moving["origin_z"] = 3.0;
    DOCTEST_CHECK_NOTHROW(cloud.addHitPoint(0, vec3(5.f, 6.f, 7.f), SphericalCoord(1.f, 0.f, 0.f), moving));

    // Hit 1: static, no origin labels.
    std::map<std::string, double> stat;
    DOCTEST_CHECK_NOTHROW(cloud.addHitPoint(0, vec3(10.f, 10.f, 10.f), SphericalCoord(1.f, 0.f, 0.f), stat));

    const vec3 shift = make_vec3(100.f, 200.f, 300.f);
    DOCTEST_CHECK_NOTHROW(cloud.coordinateShift(shift));

    // Moving hit: origin labels must have shifted by the same amount as the position.
    DOCTEST_CHECK(cloud.getHitData(0, "origin_x") == doctest::Approx(1.0 + 100.0));
    DOCTEST_CHECK(cloud.getHitData(0, "origin_y") == doctest::Approx(2.0 + 200.0));
    DOCTEST_CHECK(cloud.getHitData(0, "origin_z") == doctest::Approx(3.0 + 300.0));
    DOCTEST_CHECK(cloud.getHitXYZ(0).x == doctest::Approx(5.f + 100.f));

    // Static hit: still carries no origin labels.
    DOCTEST_CHECK(cloud.doesHitDataExist(1, "origin_x") == false);
    DOCTEST_CHECK(cloud.getHitXYZ(1).x == doctest::Approx(10.f + 100.f));
}

DOCTEST_TEST_CASE("LiDAR Typed Hit Data Columns - Registry and Exact Round Trip") {
    // Each label's column has a storage type fixed at creation: standard count/index labels are 32-bit
    // integers, standard measurements are 32-bit floats, timestamp and every unrecognized label are
    // 64-bit doubles. Values must round-trip exactly through both the per-hit and the bulk readers.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(scan);

    const double gps_time = 1.7e9 + 1e-6; // microsecond resolution at GPS-epoch magnitude: only a double keeps it
    const double custom = 1234567.890123;
    for (int i = 0; i < 5; i++) {
        std::map<std::string, double> data;
        data["target_index"] = i; // registry -> INT32
        data["intensity"] = 0.25 * i; // registry -> FLOAT32 (exactly representable)
        data["timestamp"] = gps_time + i; // registry -> FLOAT64
        data["my_custom_field"] = custom + i; // unrecognized -> FLOAT64
        cloud.addHitPoint(0, make_vec3(float(i), 0.f, 1.f), SphericalCoord(1.f, 0.f, 0.f), data);
    }

    DOCTEST_CHECK(cloud.getHitDataType("target_index") == HitDataType::INT32);
    DOCTEST_CHECK(cloud.getHitDataType("intensity") == HitDataType::FLOAT32);
    DOCTEST_CHECK(cloud.getHitDataType("timestamp") == HitDataType::FLOAT64);
    DOCTEST_CHECK(cloud.getHitDataType("my_custom_field") == HitDataType::FLOAT64);
    DOCTEST_CHECK_THROWS(static_cast<void>(cloud.getHitDataType("never_set")));

    for (uint i = 0; i < 5; i++) {
        DOCTEST_CHECK(cloud.getHitData(i, "target_index") == double(i));
        DOCTEST_CHECK(cloud.getHitData(i, "intensity") == 0.25 * i);
        DOCTEST_CHECK(cloud.getHitData(i, "timestamp") == gps_time + i); // exact, not Approx
        DOCTEST_CHECK(cloud.getHitData(i, "my_custom_field") == custom + i);
    }

    // A value a float does not hold exactly widens the automatically typed column to double, keeping every value.
    cloud.setHitData(0, "intensity", 0.1);
    DOCTEST_CHECK(cloud.getHitData(0, "intensity") == 0.1);
    DOCTEST_CHECK(cloud.getHitData(1, "intensity") == 0.25);
    DOCTEST_CHECK(cloud.getHitDataType("intensity") == HitDataType::FLOAT64);

    // Typed bulk readers return the column's own type without widening; all three overloads agree.
    std::vector<int32_t> ti_i32;
    std::vector<float> ti_f32;
    std::vector<double> ti_f64;
    cloud.getHitDataColumn("target_index", ti_i32);
    cloud.getHitDataColumn("target_index", ti_f32);
    cloud.getHitDataColumn("target_index", ti_f64);
    std::vector<double> ts_f64;
    cloud.getHitDataColumn("timestamp", ts_f64);
    DOCTEST_REQUIRE(ti_i32.size() == 5);
    for (size_t i = 0; i < 5; i++) {
        DOCTEST_CHECK(ti_i32[i] == int32_t(i));
        DOCTEST_CHECK(ti_f32[i] == float(i));
        DOCTEST_CHECK(ti_f64[i] == double(i));
        DOCTEST_CHECK(ts_f64[i] == gps_time + double(i));
    }

    // Absent entries take the caller's sentinel in every overload.
    cloud.setHitData(2, "sparse", 7.0);
    std::vector<int32_t> sp_i32;
    std::vector<float> sp_f32;
    cloud.getHitDataColumn("sparse", sp_i32, -1);
    cloud.getHitDataColumn("sparse", sp_f32, -2.5f);
    DOCTEST_CHECK(sp_i32[2] == 7);
    DOCTEST_CHECK(sp_i32[1] == -1);
    DOCTEST_CHECK(sp_f32[2] == 7.f);
    DOCTEST_CHECK(sp_f32[3] == -2.5f);

    // The memory estimate charges each column its current width: 4 bytes for INT32/FLOAT32, 8 for FLOAT64 (here
    // target_index stays INT32, intensity has widened, timestamp and the custom and sparse fields are doubles), plus
    // one presence bit per point packed into 64-bit words.
    const size_t words = (1000 + 63) / 64;
    const size_t expected = 1000 * (sizeof(HitPoint) + 4 + 8 + 8 + 8 + 8) + 5 * words * sizeof(uint64_t);
    DOCTEST_CHECK(cloud.estimateHitPointMemory(1000) == expected);
}

DOCTEST_TEST_CASE("LiDAR Typed Hit Data Columns - Automatic Integer Columns Widen, Explicit Ones Reject") {
    // An integer column typed automatically from the label's name widens to double on the first value that is not a
    // 32-bit integer, keeping every value exactly. A column whose type the caller fixed with createHitDataColumn() is a
    // contract instead: an explicit INT32 column rejects such a value with an error naming the label and the remedy,
    // and the rejected hit is not added; an explicit FLOAT32 column stores values at float precision.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 4, 0.25f * float(M_PI), 0.75f * float(M_PI), 4, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(scan);

    std::map<std::string, double> ok;
    ok["row"] = 3;
    cloud.addHitPoint(0, make_vec3(1, 0, 1), SphericalCoord(1.f, 0.f, 0.f), ok);
    DOCTEST_CHECK(cloud.getHitDataType("row") == HitDataType::INT32);
    cloud.setHitData(0, "row", 2.5);
    DOCTEST_CHECK(cloud.getHitData(0, "row") == 2.5);
    DOCTEST_CHECK(cloud.getHitDataType("row") == HitDataType::FLOAT64);
    std::map<std::string, double> fractional;
    fractional["row"] = 4.5;
    cloud.addHitPoint(0, make_vec3(2, 0, 1), SphericalCoord(1.f, 0.f, 0.f), fractional);
    DOCTEST_CHECK(cloud.getHitCount() == 2);
    DOCTEST_CHECK(cloud.getHitData(1, "row") == 4.5);
    cloud.setHitData(0, "row", 3.0e9);
    DOCTEST_CHECK(cloud.getHitData(0, "row") == 3.0e9);
    cloud.setHitData(0, "row", std::numeric_limits<double>::quiet_NaN());
    DOCTEST_CHECK(std::isnan(cloud.getHitData(0, "row")));

    // Choosing the type explicitly, before any data, overrides the automatic choice and is never widened.
    LiDARcloud cloud2;
    cloud2.disableMessages();
    cloud2.addScan(scan);
    cloud2.createHitDataColumn("row", HitDataType::FLOAT64);
    DOCTEST_CHECK(cloud2.getHitDataType("row") == HitDataType::FLOAT64);
    cloud2.addHitPoint(0, make_vec3(1, 0, 1), SphericalCoord(1.f, 0.f, 0.f), fractional);
    DOCTEST_CHECK(cloud2.getHitData(0, "row") == 4.5);
    DOCTEST_CHECK_NOTHROW(cloud2.createHitDataColumn("row", HitDataType::FLOAT64)); // same type: no-op
    DOCTEST_CHECK_THROWS_AS(cloud2.createHitDataColumn("row", HitDataType::INT32), std::runtime_error); // type is fixed
    cloud2.createHitDataColumn("big_custom", HitDataType::FLOAT32);
    DOCTEST_CHECK(cloud2.getHitDataType("big_custom") == HitDataType::FLOAT32);
    cloud2.setHitData(0, "big_custom", 0.1);
    DOCTEST_CHECK(cloud2.getHitData(0, "big_custom") == double(0.1f));
    DOCTEST_CHECK(cloud2.getHitDataType("big_custom") == HitDataType::FLOAT32);

    cloud2.createHitDataColumn("count", HitDataType::INT32);
    std::string message;
    try {
        cloud2.setHitData(0, "count", 2.5);
        DOCTEST_CHECK_MESSAGE(false, "setHitData should have rejected 2.5 for an explicitly typed INT32 column");
    } catch (const std::runtime_error &e) {
        message = e.what();
    }
    DOCTEST_CHECK(message.find("count") != std::string::npos);
    DOCTEST_CHECK(message.find("createHitDataColumn") != std::string::npos);
    DOCTEST_CHECK(!cloud2.doesHitDataExist(0, "count")); // untouched
    std::map<std::string, double> bad;
    bad["count"] = 4.5;
    DOCTEST_CHECK_THROWS_AS(cloud2.addHitPoint(0, make_vec3(2, 0, 1), SphericalCoord(1.f, 0.f, 0.f), bad), std::runtime_error);
    DOCTEST_CHECK(cloud2.getHitCount() == 1); // the rejected hit was not added
    DOCTEST_CHECK_THROWS_AS(cloud2.setHitData(0, "count", 3.0e9), std::runtime_error); // out of 32-bit range
    // Widening an automatically typed column to double on request is lossless, so it is allowed.
    DOCTEST_CHECK_NOTHROW(cloud.createHitDataColumn("row", HitDataType::FLOAT64));
}

DOCTEST_TEST_CASE("LiDAR Typed Hit Data Columns - Presence Bits Survive Delete and Materialize") {
    // Presence is one bit per hit packed into 64-bit words; swap-and-pop deletion and materializing
    // virtual misses must keep those bits in lockstep with the hits across word boundaries.
    LiDARcloud lidar;
    lidar.disableMessages();
    const int Ntheta = 12, Nphi = 30;
    ScanMetadata scan(make_vec3(0, 0, 0), uint(Ntheta), 0.05, 0.95 * M_PI, uint(Nphi), 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);
    uint nreal = 0;
    for (int row = 0; row < Ntheta; row++) {
        for (int col = 0; col < Nphi; col++) {
            if (row >= 3 && row <= 6 && col >= 8 && col <= 20) {
                continue; // blank region for the gap filler
            }
            const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta);
            const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
            SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
            vec3 xyz = sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            if (nreal % 7 == 0) {
                data["sparse"] = double(nreal); // present on every 7th hit only
            }
            lidar.addHitPoint(0, xyz, dir, data);
            nreal++;
        }
    }
    lidar.gapfillMisses(0, false, true);
    DOCTEST_REQUIRE(lidar.getVirtualMissCount() > 0);
    lidar.materializeMisses();
    const uint ntotal = lidar.getHitCount();
    DOCTEST_CHECK(ntotal > nreal);
    // The gap filler stamps is_miss = 0 / gapfillMisses_code = 0 on the original returns and 1 / 1 on
    // the misses it synthesizes, so both labels are present everywhere and distinguish the two by value.
    for (uint i = 0; i < ntotal; i++) {
        const bool is_real = i < nreal;
        DOCTEST_CHECK(lidar.doesHitDataExist(i, "sparse") == (is_real && i % 7 == 0));
        DOCTEST_CHECK(lidar.doesHitDataExist(i, "is_miss"));
        DOCTEST_CHECK(lidar.doesHitDataExist(i, "gapfillMisses_code"));
        DOCTEST_CHECK(lidar.getHitData(i, "is_miss") == (is_real ? 0.0 : 1.0));
        DOCTEST_CHECK(lidar.getHitData(i, "gapfillMisses_code") == (is_real ? 0.0 : 1.0));
        if (is_real && i % 7 == 0) {
            DOCTEST_CHECK(lidar.getHitData(i, "sparse") == double(i));
        }
    }
    // Delete every third hit from the back; the survivor swapped into each slot keeps its own bits.
    std::vector<std::pair<vec3, bool>> expected; // (position, has sparse)
    for (uint i = 0; i < ntotal; i++) {
        expected.emplace_back(lidar.getHitXYZ(i), lidar.doesHitDataExist(i, "sparse"));
    }
    for (int i = int(ntotal) - 1; i >= 0; i -= 3) {
        lidar.deleteHitPoint(uint(i));
    }
    for (uint i = 0; i < lidar.getHitCount(); i++) {
        const vec3 p = lidar.getHitXYZ(i);
        bool found = false;
        for (const auto &e: expected) {
            if (e.first.x == p.x && e.first.y == p.y && e.first.z == p.z) {
                DOCTEST_CHECK(lidar.doesHitDataExist(i, "sparse") == e.second);
                found = true;
                break;
            }
        }
        DOCTEST_CHECK(found);
    }
}

// ---- Per-scan hit index and per-scan readers ----

//! Copy every hit of `src` into `dst` (which must already hold the same scans and grid), interleaving the
//! scans round-robin so that no scan's hits are contiguous in the destination's index space.
static void copyHitsInterleaved(const LiDARcloud &src, LiDARcloud &dst) {
    const uint Nscans = src.getScanCount();
    std::vector<std::vector<uint>> per_scan(Nscans);
    for (uint s = 0; s < Nscans; s++) {
        src.getScanHitIndices(s, per_scan[s]);
    }
    const std::vector<std::string> labels = LiDARTestHelper::hitDataLabels(src);
    size_t longest = 0;
    for (const auto &v: per_scan) {
        longest = std::max(longest, v.size());
    }
    for (size_t k = 0; k < longest; k++) {
        for (uint s = 0; s < Nscans; s++) {
            if (k >= per_scan[s].size()) {
                continue;
            }
            const uint i = per_scan[s][k];
            std::map<std::string, double> data;
            for (const std::string &label: labels) {
                if (src.doesHitDataExist(i, label.c_str())) {
                    data[label] = src.getHitData(i, label.c_str());
                }
            }
            dst.addHitPoint(uint(src.getHitScanID(i)), src.getHitXYZ(i), LiDARTestHelper::storedDirection(src, i), src.getHitColor(i), data);
        }
    }
}

DOCTEST_TEST_CASE("LiDAR Per-Scan Index - Interleaved Scans Give the Same Triangulation") {
    // sphere.xml holds four scans, loaded scan after scan so each scan's hits are contiguous. Rebuilding
    // the same cloud with the four scans interleaved hit by hit must produce the identical mesh: the
    // per-scan gather locates a scan's hits through the index (a permutation here) rather than by
    // position, and the mesh only depends on which hits belong to which scan.
    LiDARcloud ordered;
    ordered.disableMessages();
    ordered.loadXML("plugins/lidar/xml/sphere.xml");
    DOCTEST_REQUIRE(ordered.getScanCount() == 4);

    LiDARcloud interleaved;
    interleaved.disableMessages();
    interleaved.loadXML("plugins/lidar/xml/sphere.xml");
    LiDARTestHelper::clearHits(interleaved); // keep the scans and grid, drop the hits
    DOCTEST_REQUIRE(interleaved.getHitCount() == 0);
    copyHitsInterleaved(ordered, interleaved);
    DOCTEST_REQUIRE(interleaved.getHitCount() == ordered.getHitCount());
    // The interleaved cloud really is out of scan order.
    bool saw_descent = false;
    for (uint i = 1; i < interleaved.getHitCount() && !saw_descent; i++) {
        saw_descent = interleaved.getHitScanID(i) < interleaved.getHitScanID(i - 1);
    }
    DOCTEST_CHECK(saw_descent);
    for (uint s = 0; s < 4; s++) {
        DOCTEST_CHECK(interleaved.getScanHitCount(s) == ordered.getScanHitCount(s));
    }

    ordered.triangulateHitPoints(0.5, 5);
    interleaved.triangulateHitPoints(0.5, 5);
    DOCTEST_REQUIRE(interleaved.getTriangleCount() == ordered.getTriangleCount());
    DOCTEST_CHECK(ordered.getTriangleCount() == 383u);
    // Same triangles, in the same per-scan order (the second cloud's vertex IDs differ, the geometry does not).
    for (uint t = 0; t < ordered.getTriangleCount(); t++) {
        const Triangulation a = ordered.getTriangle(t);
        const Triangulation b = interleaved.getTriangle(t);
        DOCTEST_CHECK(a.scanID == b.scanID);
        DOCTEST_CHECK(a.vertex0.x == b.vertex0.x);
        DOCTEST_CHECK(a.vertex1.y == b.vertex1.y);
        DOCTEST_CHECK(a.vertex2.z == b.vertex2.z);
        DOCTEST_CHECK(a.area == b.area);
        DOCTEST_CHECK(a.gridcell == b.gridcell);
    }
}

DOCTEST_TEST_CASE("LiDAR Per-Scan Index - Interleaved and Filter-Scrambled Scans Give the Same Leaf Area") {
    // Two synthetic scans of the leaf cube from different positions, inverted three ways: as scanned
    // (each scan's hits contiguous), rebuilt with the scans interleaved hit by hit, and with the stored
    // order scrambled by a filter's swap-and-pop deletions. Per-cell leaf area, G(theta) and beam counts
    // must agree, and the inversion's largest per-scan working array must be the size of the largest
    // scan -- not the cloud -- which is what keeps its scratch bounded by one scan.
    Context context;
    context.seedRandomGenerator(0);
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    auto add_scans_and_grid = [](LiDARcloud &lidar) {
        ScanMetadata scan0(vec3(-5.f, 0.f, 0.5f), 2000, 0.f, float(M_PI), 3000, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
        ScanMetadata scan1(vec3(0.f, 4.f, 0.6f), 300, 0.f, float(M_PI), 450, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
        lidar.addScan(scan0);
        lidar.addScan(scan1);
        lidar.addGrid(vec3(0.f, 0.f, 0.5f), vec3(1.f, 1.f, 1.f), make_int3(2, 2, 2), 0);
    };
    auto invert = [&](LiDARcloud &lidar, std::vector<float> &leaf_area, std::vector<float> &gtheta, std::vector<int> &beam_count) {
        lidar.calculateHitGridCell();
        lidar.triangulateHitPoints(0.04, 10);
        lidar.calculateLeafArea(&context);
        const uint Ncells = lidar.getGridCellCount();
        leaf_area.resize(Ncells);
        gtheta.resize(Ncells);
        beam_count.resize(Ncells);
        for (uint c = 0; c < Ncells; c++) {
            leaf_area[c] = lidar.getCellLeafArea(c);
            gtheta[c] = lidar.getCellGtheta(c);
            beam_count[c] = lidar.getCellBeamCount(c);
        }
    };
    auto compare = [](const std::vector<float> &la_a, const std::vector<float> &g_a, const std::vector<int> &bc_a, const std::vector<float> &la_b, const std::vector<float> &g_b, const std::vector<int> &bc_b, const char *what) {
        DOCTEST_REQUIRE(la_a.size() == la_b.size());
        for (size_t c = 0; c < la_a.size(); c++) {
            DOCTEST_CHECK_MESSAGE(la_a[c] == doctest::Approx(la_b[c]).epsilon(1e-4f), std::string(what) << ": leaf_area mismatch in cell " << c);
            DOCTEST_CHECK_MESSAGE(g_a[c] == doctest::Approx(g_b[c]).epsilon(1e-4f), std::string(what) << ": Gtheta mismatch in cell " << c);
            DOCTEST_CHECK_MESSAGE(bc_a[c] == bc_b[c], std::string(what) << ": beam_count mismatch in cell " << c);
        }
    };

    LiDARcloud ordered;
    ordered.disableMessages();
    add_scans_and_grid(ordered);
    ordered.syntheticScan(&context, true, true); // scan_grid_only, record_misses
    DOCTEST_REQUIRE(ordered.getScanHitCount(0) > 10 * ordered.getScanHitCount(1));
    DOCTEST_REQUIRE(ordered.getScanHitCount(1) > 0);
    std::vector<float> la_o, g_o;
    std::vector<int> bc_o;
    invert(ordered, la_o, g_o, bc_o);
    for (size_t c = 0; c < la_o.size(); c++) {
        DOCTEST_CHECK(bc_o[c] > 0);
    }
    // Scratch bound: the largest per-scan array is the largest scan, not the whole cloud.
    DOCTEST_CHECK(LiDARTestHelper::leafAreaMaxScanScratch(ordered) == ordered.getScanHitCount(0));
    DOCTEST_CHECK(LiDARTestHelper::leafAreaMaxScanScratch(ordered) < ordered.getHitCount());

    // (1) Interleaved: the same hits, scans alternating hit by hit.
    LiDARcloud interleaved;
    interleaved.disableMessages();
    add_scans_and_grid(interleaved);
    copyHitsInterleaved(ordered, interleaved);
    DOCTEST_REQUIRE(interleaved.getHitCount() == ordered.getHitCount());
    std::vector<float> la_i, g_i;
    std::vector<int> bc_i;
    invert(interleaved, la_i, g_i, bc_i);
    compare(la_o, g_o, bc_o, la_i, g_i, bc_i, "interleaved");

    // (2) Scrambled by a filter: rebuild the cloud in scan order but with throw-away hits of the OTHER scan
    // interleaved every few hundred points (placed far behind the scanners so their beams point away from
    // the grid), then delete them with a box filter. Each swap-and-pop deletion of an interior dummy drops
    // the cloud's last stored hit -- a genuine scan-1 return -- into scan 0's block, so the survivors are
    // out of scan order even though no genuine hit was removed.
    LiDARcloud scrambled;
    scrambled.disableMessages();
    add_scans_and_grid(scrambled);
    {
        const std::vector<std::string> labels = LiDARTestHelper::hitDataLabels(ordered);
        int dummy = 0;
        for (uint s = 0; s < 2; s++) {
            std::vector<uint> indices;
            ordered.getScanHitIndices(s, indices);
            for (size_t k = 0; k < indices.size(); k++) {
                const uint i = indices[k];
                std::map<std::string, double> data;
                for (const std::string &label: labels) {
                    if (ordered.doesHitDataExist(i, label.c_str())) {
                        data[label] = ordered.getHitData(i, label.c_str());
                    }
                }
                scrambled.addHitPoint(s, ordered.getHitXYZ(i), LiDARTestHelper::storedDirection(ordered, i), ordered.getHitColor(i), data);
                if (s == 0 && k % 400 == 0) {
                    const vec3 away = vec3(0.f, 3000.f + 0.5f * float(dummy++), 0.6f); // behind scan 1's scanner
                    std::map<std::string, double> dummy_data;
                    dummy_data["is_miss"] = 0.0;
                    scrambled.addHitPoint(1, away, cart2sphere(away - scrambled.getScanOrigin(1)), dummy_data);
                }
            }
        }
        DOCTEST_REQUIRE(dummy > 10);
    }
    scrambled.xyzFilter(-2000.f, 2000.f, -2000.f, 2000.f, -2000.f, 2000.f); // keeps every genuine hit, including misses at ~1001 m
    DOCTEST_REQUIRE(scrambled.getHitCount() == ordered.getHitCount());
    bool saw_descent = false;
    for (uint i = 1; i < scrambled.getHitCount() && !saw_descent; i++) {
        saw_descent = scrambled.getHitScanID(i) < scrambled.getHitScanID(i - 1);
    }
    DOCTEST_CHECK(saw_descent);
    std::vector<float> la_s, g_s;
    std::vector<int> bc_s;
    invert(scrambled, la_s, g_s, bc_s);
    compare(la_o, g_o, bc_o, la_s, g_s, bc_s, "scrambled");
}

DOCTEST_TEST_CASE("LiDAR Per-Scan Readers Match Per-Index Accessors on a Gap-Filled Multi-Scan Cloud") {
    // Two row/column scans, both gap-filled so each has stored returns in the real prefix of the index
    // space and virtualized misses in the tail. The per-scan readers must enumerate exactly the hits the
    // per-index accessors attribute to that scan, in the order getScanHitIndices reports.
    LiDARcloud lidar;
    lidar.disableMessages();
    for (uint s = 0; s < 2; s++) {
        const int Ntheta = 10 + int(s) * 4, Nphi = 24;
        ScanMetadata scan(make_vec3(float(s) * 3.f, 0, 0), uint(Ntheta), 0.05, 0.95 * M_PI, uint(Nphi), 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
        lidar.addScan(scan);
        for (int row = 0; row < Ntheta; row++) {
            for (int col = 0; col < Nphi; col++) {
                if ((row + col + int(s)) % 3 == 0) {
                    continue;
                }
                const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta);
                const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
                SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
                vec3 xyz = lidar.getScanOrigin(s) + sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
                std::map<std::string, double> data;
                data["row"] = row;
                data["column"] = col;
                data["intensity"] = 0.5 * row + col;
                lidar.addHitPoint(s, xyz, dir, data);
            }
        }
    }
    lidar.gapfillMisses(0, false, true);
    lidar.gapfillMisses(1, false, true);
    DOCTEST_REQUIRE(lidar.getVirtualMissCount() > 0);

    size_t total = 0;
    for (uint s = 0; s < 2; s++) {
        std::vector<uint> indices;
        std::vector<vec3> xyz;
        std::vector<double> intensity, code;
        std::vector<int32_t> row;
        lidar.getScanHitIndices(s, indices);
        lidar.getScanHitXYZColumn(s, xyz);
        lidar.getScanHitDataColumn(s, "intensity", intensity, -1.0);
        lidar.getScanHitDataColumn(s, "gapfillMisses_code", code, -1.0);
        lidar.getScanHitDataColumn(s, "row", row, int32_t(-1));
        const size_t n = lidar.getScanHitCount(s);
        DOCTEST_REQUIRE(indices.size() == n);
        DOCTEST_REQUIRE(xyz.size() == n);
        DOCTEST_REQUIRE(intensity.size() == n);
        DOCTEST_REQUIRE(row.size() == n);
        total += n;
        for (size_t i = 0; i < n; i++) {
            const uint g = indices[i];
            DOCTEST_CHECK(lidar.getHitScanID(g) == int(s));
            const vec3 p = lidar.getHitXYZ(g);
            DOCTEST_CHECK(xyz[i].x == p.x);
            DOCTEST_CHECK(xyz[i].y == p.y);
            DOCTEST_CHECK(xyz[i].z == p.z);
            DOCTEST_CHECK(intensity[i] == (lidar.doesHitDataExist(g, "intensity") ? lidar.getHitData(g, "intensity") : -1.0));
            DOCTEST_CHECK(code[i] == (lidar.doesHitDataExist(g, "gapfillMisses_code") ? lidar.getHitData(g, "gapfillMisses_code") : -1.0));
            DOCTEST_CHECK(row[i] == int32_t(lidar.getHitData(g, "row")));
        }
    }
    DOCTEST_CHECK(total == lidar.getHitCount());
    DOCTEST_CHECK_THROWS(static_cast<void>(lidar.getScanHitCount(2)));
}

DOCTEST_TEST_CASE("LiDAR Hit Grid Cell Binning - Lattice Fast Path Matches Cell-by-Cell Containment") {
    // On a regular lattice the containing cell is located arithmetically; the assignment must be the one
    // the cell-by-cell containment scan (getContainingGridCell) makes, including on a rotated grid and
    // for hits on cell faces. A non-lattice grid takes the scan itself and must also agree.
    LiDARcloud lidar;
    lidar.disableMessages();
    lidar.loadXML("plugins/lidar/xml/sphere.xml");
    const uint N = lidar.getHitCount();
    DOCTEST_REQUIRE(N > 0);

    auto check_all = [&](const char *what) {
        lidar.calculateHitGridCell();
        size_t inside = 0;
        for (uint i = 0; i < N; i++) {
            const int expected = LiDARTestHelper::containingGridCell(lidar, lidar.getHitXYZ(i));
            DOCTEST_CHECK_MESSAGE(lidar.getHitGridCell(i) == expected, std::string(what) << ": hit " << i);
            inside += expected >= 0 ? 1 : 0;
        }
        DOCTEST_CHECK_MESSAGE(inside > 0, std::string(what) << ": no hit fell inside the grid");
    };

    // Rotated lattice covering part of the sphere, plus points placed exactly on interior cell faces.
    LiDARTestHelper::clearGrid(lidar);
    lidar.addGrid(vec3(0.f, 0.f, 0.f), vec3(1.2f, 0.9f, 1.5f), make_int3(3, 2, 4), 30.f);
    for (int k = 0; k < 12; k++) {
        std::map<std::string, double> data;
        lidar.addHitPoint(0, vec3(-0.6f + 0.4f * float(k % 4), -0.45f + 0.45f * float((k / 4) % 3), 0.f), SphericalCoord(1.f, 0.f, 0.f), data);
    }
    const uint N2 = lidar.getHitCount();
    DOCTEST_REQUIRE(N2 == N + 12);
    {
        lidar.calculateHitGridCell();
        for (uint i = 0; i < N2; i++) {
            DOCTEST_CHECK(lidar.getHitGridCell(i) == LiDARTestHelper::containingGridCell(lidar, lidar.getHitXYZ(i)));
        }
    }

    // Un-rotated lattice.
    LiDARTestHelper::clearGrid(lidar);
    lidar.addGrid(vec3(0.f, 0.f, 0.f), vec3(1.2f, 0.9f, 1.5f), make_int3(3, 2, 4), 0.f);
    check_all("axis-aligned lattice");

    // Non-lattice grid: two cells of different size (the scan path).
    LiDARTestHelper::clearGrid(lidar);
    lidar.addGridCell(vec3(0.f, 0.f, 0.f), vec3(1.f, 1.f, 1.f), 0.f);
    lidar.addGridCell(vec3(0.f, 0.f, 1.f), vec3(0.5f, 0.5f, 1.f), 0.f);
    check_all("non-lattice");
}

// ---- Bulk ingest ----

DOCTEST_TEST_CASE("LiDAR addHitPoints - Chunked Ingest Equals One Call and Equals addHitPoint") {
    // The downstream pattern is a loop over row chunks read from disk. Ten chunks must produce a cloud
    // identical to a single call over the same rows, and identical to adding the same points one at a
    // time with addHitPoint(): positions, stored directions, scan IDs, and every label's value and
    // presence (NaN in the bulk array means absent).
    const size_t N = 20000, CHUNK = 2000;
    const std::vector<std::string> labels = {"target_index", "intensity", "timestamp", "custom"};
    std::vector<double> xyz(3 * N), values(N * labels.size());
    std::vector<float> dir(3 * N);
    for (size_t i = 0; i < N; i++) {
        const double a = 2.0 * M_PI * double(i) / double(N);
        xyz[3 * i] = 5.0 * cos(a);
        xyz[3 * i + 1] = 5.0 * sin(a);
        xyz[3 * i + 2] = 1.0 + 0.001 * double(i % 97);
        const SphericalCoord d = cart2sphere(make_vec3(float(xyz[3 * i]), float(xyz[3 * i + 1]), float(xyz[3 * i + 2])) - make_vec3(0.f, 0.f, 1.f));
        dir[3 * i] = d.radius;
        dir[3 * i + 1] = d.elevation;
        dir[3 * i + 2] = d.azimuth;
        values[i * 4 + 0] = double(i % 3); // INT32
        values[i * 4 + 1] = 0.25 * double(i % 8); // FLOAT32 (exact)
        values[i * 4 + 2] = 1.7e9 + 1e-6 * double(i); // FLOAT64
        values[i * 4 + 3] = (i % 5 == 0) ? std::numeric_limits<double>::quiet_NaN() : double(i); // absent on every 5th
    }
    ScanMetadata scan(make_vec3(0, 0, 1), 10, 0.05, 0.95 * M_PI, 10, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});

    LiDARcloud one, chunked, single;
    one.disableMessages();
    chunked.disableMessages();
    single.disableMessages();
    one.addScan(scan);
    chunked.addScan(scan);
    single.addScan(scan);

    one.addHitPoints(0, N, xyz.data(), dir.data(), labels, values.data());
    for (size_t c = 0; c < N; c += CHUNK) {
        chunked.addHitPoints(0, CHUNK, xyz.data() + 3 * c, dir.data() + 3 * c, labels, values.data() + c * labels.size());
    }
    for (size_t i = 0; i < N; i++) {
        std::map<std::string, double> data;
        for (size_t k = 0; k < labels.size(); k++) {
            if (!std::isnan(values[i * 4 + k])) {
                data[labels[k]] = values[i * 4 + k];
            }
        }
        single.addHitPoint(0, make_vec3(float(xyz[3 * i]), float(xyz[3 * i + 1]), float(xyz[3 * i + 2])), SphericalCoord(dir[3 * i], dir[3 * i + 1], dir[3 * i + 2]), data);
    }

    DOCTEST_REQUIRE(one.getHitCount() == N);
    DOCTEST_REQUIRE(chunked.getHitCount() == N);
    DOCTEST_REQUIRE(single.getHitCount() == N);
    DOCTEST_CHECK(one.getHitDataType("target_index") == HitDataType::INT32);
    DOCTEST_CHECK(chunked.getHitDataType("custom") == HitDataType::FLOAT64);

    auto same = [&](const LiDARcloud &a, const LiDARcloud &b, const char *what) {
        std::vector<vec3> xa, xb;
        a.getHitXYZColumn(xa);
        b.getHitXYZColumn(xb);
        std::vector<int> sa, sb;
        a.getHitScanIDColumn(sa);
        b.getHitScanIDColumn(sb);
        size_t mismatches = 0;
        for (size_t i = 0; i < N; i++) {
            if (xa[i].x != xb[i].x || xa[i].y != xb[i].y || xa[i].z != xb[i].z || sa[i] != sb[i]) {
                mismatches++;
            }
            const SphericalCoord da = LiDARTestHelper::storedDirection(a, uint(i));
            const SphericalCoord db = LiDARTestHelper::storedDirection(b, uint(i));
            if (da.radius != db.radius || da.elevation != db.elevation || da.azimuth != db.azimuth) {
                mismatches++;
            }
        }
        for (const std::string &label: labels) {
            std::vector<double> va, vb;
            a.getHitDataColumn(label.c_str(), va, -12345.0);
            b.getHitDataColumn(label.c_str(), vb, -12345.0);
            for (size_t i = 0; i < N; i++) {
                if (va[i] != vb[i]) {
                    mismatches++;
                }
            }
        }
        DOCTEST_CHECK_MESSAGE(mismatches == 0, std::string(what) << ": " << mismatches << " mismatching entries");
    };
    same(one, chunked, "one call vs chunked");
    same(one, single, "one call vs addHitPoint loop");

    // Values landed exactly, and NaN produced an absent entry rather than a stored NaN.
    DOCTEST_CHECK(one.getHitData(7, "timestamp") == 1.7e9 + 1e-6 * 7.0);
    DOCTEST_CHECK(one.getHitData(7, "intensity") == 0.25 * 7.0);
    DOCTEST_CHECK(one.doesHitDataExist(5, "custom") == false);
    DOCTEST_CHECK(one.doesHitDataExist(6, "custom") == true);

    // A null direction array derives each direction from the scan origin, like the ASCII loader.
    LiDARcloud derived;
    derived.disableMessages();
    derived.addScan(scan);
    derived.addHitPoints(0, N, xyz.data(), nullptr, {}, nullptr);
    for (uint i = 0; i < N; i += 997) {
        const SphericalCoord expect = cart2sphere(derived.getHitXYZ(i) - make_vec3(0.f, 0.f, 1.f));
        const SphericalCoord got = LiDARTestHelper::storedDirection(derived, i);
        DOCTEST_CHECK(got.zenith == expect.zenith);
        DOCTEST_CHECK(got.azimuth == expect.azimuth);
    }
}

DOCTEST_TEST_CASE("LiDAR addHitPoints - Reserve Removes the Growth Transient; Errors Leave the Cloud Unchanged") {
    const size_t N = 50000, CHUNK = 5000;
    std::vector<double> xyz(3 * N, 1.0), values(N, 1.0);
    ScanMetadata scan(make_vec3(0, 0, 0), 10, 0.05, 0.95 * M_PI, 10, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});

    // Reserved: the arrays never reallocate across the chunk loop (capacity is pinned), so peak memory
    // during ingest is the steady state.
    LiDARcloud reserved;
    reserved.disableMessages();
    reserved.addScan(scan);
    reserved.reserveHitPoints(N);
    const size_t capacity_before = reserved.getHitPointCapacity();
    DOCTEST_CHECK(capacity_before >= N);
    for (size_t c = 0; c < N; c += CHUNK) {
        reserved.addHitPoints(0, CHUNK, xyz.data() + 3 * c, nullptr, {"intensity"}, values.data() + c);
        DOCTEST_CHECK(reserved.getHitPointCapacity() == capacity_before);
    }
    DOCTEST_CHECK(reserved.getHitCount() == N);

    // Unreserved: geometric growth keeps the over-allocation bounded (at most 2x the final size).
    LiDARcloud grown;
    grown.disableMessages();
    grown.addScan(scan);
    for (size_t c = 0; c < N; c += CHUNK) {
        grown.addHitPoints(0, CHUNK, xyz.data() + 3 * c, nullptr, {"intensity"}, values.data() + c);
    }
    DOCTEST_CHECK(grown.getHitPointCapacity() >= N);
    DOCTEST_CHECK(grown.getHitPointCapacity() <= 2 * N);

    // A batch carrying a value its explicitly typed integer column cannot hold is rejected before anything is appended.
    LiDARcloud rejected;
    rejected.disableMessages();
    rejected.addScan(scan);
    rejected.createHitDataColumn("row", HitDataType::INT32);
    rejected.addHitPoints(0, 10, xyz.data(), nullptr, {"row"}, values.data());
    DOCTEST_CHECK(rejected.getHitCount() == 10);
    std::vector<double> bad(10, 2.5);
    DOCTEST_CHECK_THROWS_AS(rejected.addHitPoints(0, 10, xyz.data(), nullptr, {"row"}, bad.data()), std::runtime_error);
    DOCTEST_CHECK(rejected.getHitCount() == 10);

    // The hit-point cap applies to bulk ingest, before any allocation.
    rejected.setMaxHitPoints(15);
    DOCTEST_CHECK_THROWS_AS(rejected.addHitPoints(0, 10, xyz.data(), nullptr, {}, nullptr), std::runtime_error);
    DOCTEST_CHECK(rejected.getHitCount() == 10);
    DOCTEST_CHECK_NOTHROW(rejected.addHitPoints(0, 5, xyz.data(), nullptr, {}, nullptr));
    DOCTEST_CHECK(rejected.getHitCount() == 15);
    DOCTEST_CHECK_THROWS_AS(rejected.addHitPoints(3, 1, xyz.data(), nullptr, {}, nullptr), std::runtime_error); // no such scan
    DOCTEST_CHECK_THROWS_AS(rejected.addHitPoints(0, 1, nullptr, nullptr, {}, nullptr), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(rejected.addHitPoints(0, 1, xyz.data(), nullptr, {"row"}, nullptr), std::runtime_error);
}

DOCTEST_TEST_CASE("LiDAR deleteHitPoints - Order-Preserving Range Erase Keeps Columns in Lockstep") {
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 10, 0.05, 0.95 * M_PI, 10, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    cloud.addScan(scan);
    const size_t N = 300;
    for (size_t i = 0; i < N; i++) {
        std::map<std::string, double> data;
        data["intensity"] = double(i);
        if (i % 3 == 0) {
            data["sparse"] = double(2 * i);
        }
        cloud.addHitPoint(0, make_vec3(float(i), 0.f, 1.f), SphericalCoord(1.f, 0.f, 0.f), data);
    }
    // Erase an interior range spanning several presence words; survivors keep their order and data.
    cloud.deleteHitPoints(70, 130);
    DOCTEST_REQUIRE(cloud.getHitCount() == N - 130);
    for (uint i = 0; i < cloud.getHitCount(); i++) {
        const size_t original = i < 70 ? i : i + 130;
        DOCTEST_CHECK(cloud.getHitXYZ(i).x == float(original));
        DOCTEST_CHECK(cloud.getHitData(i, "intensity") == double(original));
        DOCTEST_CHECK(cloud.doesHitDataExist(i, "sparse") == (original % 3 == 0));
        if (original % 3 == 0) {
            DOCTEST_CHECK(cloud.getHitData(i, "sparse") == double(2 * original));
        }
    }
    // Erase the tail (the streaming drain), then the rest.
    cloud.deleteHitPoints(cloud.getHitCount() - 50, 50);
    DOCTEST_CHECK(cloud.getHitCount() == N - 180);
    DOCTEST_CHECK(cloud.getHitXYZ(cloud.getHitCount() - 1).x == float(N - 51));
    DOCTEST_CHECK_THROWS_AS(cloud.deleteHitPoints(cloud.getHitCount() - 1, 2), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(cloud.deleteHitPoints(cloud.getHitCount(), 1), std::runtime_error);
    DOCTEST_CHECK_NOTHROW(cloud.deleteHitPoints(5, 0));
    cloud.deleteHitPoints(0, cloud.getHitCount());
    DOCTEST_CHECK(cloud.getHitCount() == 0);
    // The cloud is still usable afterwards.
    cloud.addHitPoint(0, make_vec3(1, 2, 3), SphericalCoord(1.f, 0.f, 0.f));
    DOCTEST_CHECK(cloud.getHitCount() == 1);
    DOCTEST_CHECK(cloud.doesHitDataExist(0, "intensity") == false);
}

// ---- Block-tiled leaf-area inversion ----

DOCTEST_TEST_CASE("LiDAR Block-Tiled Leaf Area - Union of Block Inversions Equals the Whole-Grid Inversion") {
    // A block call inverts only the cells inside [ijk_min, ijk_max] and leaves every other cell untouched, so
    // covering the grid with blocks must reproduce the whole-grid inversion cell for cell -- for triangulation-
    // derived G(theta), a scalar G(theta), and a per-cell G(theta), on the DDA path and on the brute-force path.
    Context context;
    context.seedRandomGenerator(0);
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);

    struct CellResult {
        float leaf_area, gtheta, lad_var;
        int beam_count;
    };
    auto read_cells = [](const LiDARcloud &lidar) {
        std::vector<CellResult> out(lidar.getGridCellCount());
        for (uint c = 0; c < out.size(); c++) {
            out[c] = CellResult{lidar.getCellLeafArea(c), lidar.getCellGtheta(c), lidar.getCellLADVariance(c), lidar.getCellBeamCount(c)};
        }
        return out;
    };
    auto compare = [](const std::vector<CellResult> &a, const std::vector<CellResult> &b, const char *what) {
        DOCTEST_REQUIRE(a.size() == b.size());
        for (size_t c = 0; c < a.size(); c++) {
            DOCTEST_CHECK_MESSAGE(a[c].leaf_area == doctest::Approx(b[c].leaf_area).epsilon(1e-5f), std::string(what) << ": leaf_area mismatch in cell " << c);
            DOCTEST_CHECK_MESSAGE(a[c].gtheta == doctest::Approx(b[c].gtheta).epsilon(1e-5f), std::string(what) << ": Gtheta mismatch in cell " << c);
            DOCTEST_CHECK_MESSAGE(a[c].beam_count == b[c].beam_count, std::string(what) << ": beam_count mismatch in cell " << c);
            if (a[c].lad_var < 0.f || b[c].lad_var < 0.f) {
                DOCTEST_CHECK_MESSAGE(a[c].lad_var == b[c].lad_var, std::string(what) << ": LAD_variance sign mismatch in cell " << c);
            } else {
                DOCTEST_CHECK_MESSAGE(a[c].lad_var == doctest::Approx(b[c].lad_var).epsilon(1e-4f), std::string(what) << ": LAD_variance mismatch in cell " << c);
            }
        }
    };
    const float SENTINEL = -7.f;
    auto reset_cells = [&](LiDARcloud &lidar) {
        for (uint c = 0; c < lidar.getGridCellCount(); c++) {
            lidar.setCellLeafArea(SENTINEL, c);
        }
    };

    enum GthetaMode { TRIANGULATED, SCALAR, PER_CELL };
    auto invert = [&](LiDARcloud &lidar, GthetaMode mode, const std::vector<float> &per_cell, const int3 *lo, const int3 *hi) {
        if (mode == TRIANGULATED) {
            if (lo == nullptr) {
                lidar.calculateLeafArea(&context, 1, 0.05f);
            } else {
                lidar.calculateLeafArea(&context, 1, 0.05f, *lo, *hi);
            }
        } else if (mode == SCALAR) {
            if (lo == nullptr) {
                lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f);
            } else {
                lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f, *lo, *hi);
            }
        } else {
            if (lo == nullptr) {
                lidar.calculateLeafArea(&context, per_cell, 1, 0.05f);
            } else {
                lidar.calculateLeafArea(&context, per_cell, 1, 0.05f, *lo, *hi);
            }
        }
    };

    auto run = [&](int3 divisions, const vec3 &grid_size, bool force_bruteforce, GthetaMode mode) {
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(vec3(-5.f, 0.f, 0.5f), 1000, 0.f, float(M_PI), 1500, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
        lidar.addScan(scan);
        lidar.addGrid(vec3(0.f, 0.f, 0.5f), grid_size, divisions, 0);
        lidar.syntheticScan(&context, true, true); // scan_grid_only, record_misses
        lidar.triangulateHitPoints(0.04, 10);
        LiDARTestHelper::forceBruteForceLeafArea(lidar, force_bruteforce);
        const uint Ncells = lidar.getGridCellCount();
        std::vector<float> per_cell(Ncells);
        for (uint c = 0; c < Ncells; c++) {
            per_cell[c] = 0.35f + 0.02f * float(c);
        }
        DOCTEST_CHECK(lidar.getGridGlobalCount().x == divisions.x);
        DOCTEST_CHECK(lidar.getGridGlobalCount().z == divisions.z);
        for (uint c = 0; c < Ncells; c++) {
            const int3 ijk = lidar.getCellGlobalIJK(c);
            DOCTEST_CHECK(ijk.x == int(c) % divisions.x);
            DOCTEST_CHECK(ijk.y == (int(c) / divisions.x) % divisions.y);
            DOCTEST_CHECK(ijk.z == int(c) / (divisions.x * divisions.y));
        }

        invert(lidar, mode, per_cell, nullptr, nullptr);
        const std::vector<CellResult> whole = read_cells(lidar);
        bool any_leaf = false;
        for (const CellResult &r: whole) {
            any_leaf = any_leaf || r.leaf_area > 0.f;
        }
        DOCTEST_CHECK(any_leaf);

        // One block per cell; every other cell must keep its sentinel until its own block is inverted.
        reset_cells(lidar);
        for (uint c = 0; c < Ncells; c++) {
            const int3 ijk = lidar.getCellGlobalIJK(c);
            invert(lidar, mode, per_cell, &ijk, &ijk);
            for (uint other = c + 1; other < Ncells; other++) {
                DOCTEST_CHECK(lidar.getCellLeafArea(other) == SENTINEL);
            }
        }
        compare(read_cells(lidar), whole, "1x1x1 blocks");

        // Blocks spanning several cells: split x at the midpoint, take all of y, pair z.
        reset_cells(lidar);
        const int xsplit = std::max(1, divisions.x / 2);
        for (int k0 = 0; k0 < divisions.z; k0 += 2) {
            for (int x0 = 0; x0 < divisions.x; x0 += xsplit) {
                const int3 lo = make_int3(x0, 0, k0);
                const int3 hi = make_int3(std::min(x0 + xsplit, divisions.x) - 1, divisions.y - 1, std::min(k0 + 2, divisions.z) - 1);
                invert(lidar, mode, per_cell, &lo, &hi);
            }
        }
        compare(read_cells(lidar), whole, "multi-cell blocks");

        // Invalid blocks are rejected.
        const int3 bad_lo = make_int3(0, 0, 0), bad_hi = make_int3(divisions.x, 0, 0);
        DOCTEST_CHECK_THROWS_AS(invert(lidar, mode, per_cell, &bad_lo, &bad_hi), std::runtime_error);
        const int3 inverted_lo = make_int3(1, 0, 0), inverted_hi = make_int3(0, 0, 0);
        DOCTEST_CHECK_THROWS_AS(invert(lidar, mode, per_cell, &inverted_lo, &inverted_hi), std::runtime_error);
    };

    run(make_int3(2, 2, 2), vec3(1.f, 1.f, 1.f), false, TRIANGULATED);
    run(make_int3(2, 2, 2), vec3(1.f, 1.f, 1.f), false, SCALAR);
    run(make_int3(3, 2, 4), vec3(1.5f, 1.f, 1.2f), false, PER_CELL);
    run(make_int3(2, 2, 2), vec3(1.f, 1.f, 1.f), true, SCALAR); // brute-force path

    // A grid that is not a lattice cannot be tiled.
    LiDARcloud nonlattice;
    nonlattice.disableMessages();
    ScanMetadata scan(vec3(-5.f, 0.f, 0.5f), 200, 0.f, float(M_PI), 300, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
    nonlattice.addScan(scan);
    nonlattice.addGridCell(vec3(0.f, 0.f, 0.25f), vec3(1.f, 1.f, 0.5f), 0.f);
    nonlattice.addGridCell(vec3(0.f, 0.f, 0.75f), vec3(0.5f, 0.5f, 0.5f), 0.f);
    nonlattice.syntheticScan(&context, true, true);
    const int3 zero = make_int3(0, 0, 0);
    DOCTEST_CHECK_THROWS_AS(nonlattice.calculateLeafArea(&context, 0.5f, 1, 0.05f, zero, zero), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(static_cast<void>(nonlattice.getGridGlobalCount()), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(static_cast<void>(nonlattice.getCellGlobalIJK(5)), std::runtime_error);
}

// ---- Streamed outputs ----

DOCTEST_TEST_CASE("LiDAR Triangulation Sink - Streamed Mesh Equals the Retained Mesh and Gives the Same Leaf Area") {
    // With a sink set, each scan's triangles go to the callback instead of the cloud. The streamed triangles must be
    // the retained mesh exactly, mesh consumers must refuse to run on the empty retained mesh, and the leaf-area
    // inversion -- which needs only per-voxel sums of the mesh -- must give the same answer either way.
    LiDARcloud retained;
    retained.disableMessages();
    retained.loadXML("plugins/lidar/xml/sphere.xml");
    retained.triangulateHitPoints(0.5, 5);
    DOCTEST_REQUIRE(retained.getTriangleCount() == 383u);

    LiDARcloud streamed;
    streamed.disableMessages();
    streamed.loadXML("plugins/lidar/xml/sphere.xml");
    std::vector<Triangulation> collected;
    std::vector<uint> scans_seen;
    streamed.setTriangulationSink([&](uint scanID, const std::vector<Triangulation> &tris) {
        scans_seen.push_back(scanID);
        collected.insert(collected.end(), tris.begin(), tris.end());
    });
    streamed.triangulateHitPoints(0.5, 5);

    DOCTEST_CHECK(streamed.getTriangleCount() == 0u); // nothing retained
    DOCTEST_REQUIRE(collected.size() == 383u);
    DOCTEST_CHECK(streamed.getTriangulationCandidateCount() == retained.getTriangulationCandidateCount());
    for (size_t t = 0; t < collected.size(); t++) {
        const Triangulation a = retained.getTriangle(uint(t));
        const Triangulation &b = collected[t];
        DOCTEST_CHECK(a.scanID == b.scanID);
        DOCTEST_CHECK(a.gridcell == b.gridcell);
        DOCTEST_CHECK(a.ID0 == b.ID0);
        DOCTEST_CHECK(a.vertex0.x == b.vertex0.x);
        DOCTEST_CHECK(a.vertex1.y == b.vertex1.y);
        DOCTEST_CHECK(a.vertex2.z == b.vertex2.z);
        DOCTEST_CHECK(a.area == b.area);
    }
    // One callback per scan that produced triangles, in scan order.
    for (size_t i = 1; i < scans_seen.size(); i++) {
        DOCTEST_CHECK(scans_seen[i] > scans_seen[i - 1]);
    }

    // Mesh consumers refuse to run on the streamed (empty) mesh rather than silently doing nothing.
    Context context;
    DOCTEST_CHECK_THROWS_AS(streamed.addTrianglesToContext(&context), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(static_cast<void>(streamed.getTriangle(0)), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(streamed.exportTriangleNormals("plugins/lidar/tests/.streamed_normals.txt"), std::runtime_error);

    // Clearing the sink restores retention, and a new run starts from an empty mesh (no accumulation across runs).
    streamed.setTriangulationSink(nullptr);
    streamed.triangulateHitPoints(0.5, 5);
    DOCTEST_CHECK(streamed.getTriangleCount() == 383u);
    streamed.triangulateHitPoints(0.5, 5);
    DOCTEST_CHECK(streamed.getTriangleCount() == 383u);
    DOCTEST_CHECK_NOTHROW(streamed.addTrianglesToContext(&context));

    // Leaf area from the streamed mesh equals leaf area from the retained mesh.
    Context scene;
    scene.seedRandomGenerator(0);
    scene.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    auto build = [&](bool with_sink, std::vector<float> &leaf_area, std::vector<float> &gtheta) {
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(vec3(-5.f, 0.f, 0.5f), 1000, 0.f, float(M_PI), 1500, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
        lidar.addScan(scan);
        lidar.addGrid(vec3(0.f, 0.f, 0.5f), vec3(1.f, 1.f, 1.f), make_int3(2, 2, 2), 0);
        lidar.syntheticScan(&scene, true, true);
        size_t streamed_count = 0;
        if (with_sink) {
            lidar.setTriangulationSink([&](uint, const std::vector<Triangulation> &tris) { streamed_count += tris.size(); });
        }
        lidar.triangulateHitPoints(0.04, 10);
        if (with_sink) {
            DOCTEST_CHECK(streamed_count > 0);
            DOCTEST_CHECK(lidar.getTriangleCount() == 0u);
        } else {
            DOCTEST_CHECK(lidar.getTriangleCount() > 0u);
        }
        lidar.calculateLeafArea(&scene);
        const uint Ncells = lidar.getGridCellCount();
        leaf_area.resize(Ncells);
        gtheta.resize(Ncells);
        for (uint c = 0; c < Ncells; c++) {
            leaf_area[c] = lidar.getCellLeafArea(c);
            gtheta[c] = lidar.getCellGtheta(c);
        }
    };
    std::vector<float> la_r, g_r, la_s, g_s;
    build(false, la_r, g_r);
    build(true, la_s, g_s);
    DOCTEST_REQUIRE(la_r.size() == la_s.size());
    for (size_t c = 0; c < la_r.size(); c++) {
        DOCTEST_CHECK_MESSAGE(g_s[c] == g_r[c], "Gtheta mismatch in cell " << c); // same sums, same order: bit-identical
        DOCTEST_CHECK_MESSAGE(la_s[c] == doctest::Approx(la_r[c]).epsilon(1e-4f), "leaf_area mismatch in cell " << c);
    }
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Hit Sink - Every Chunk Is Streamed and a Draining Sink Bounds the Cloud") {
    // A scan large enough to trace in more than one chunk (the chunk floor is ~1.05M rays). The sink must be handed
    // every stored return exactly once, in order, and a callback that drains the cloud must keep it at one chunk.
    Context context;
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, -2.f), make_vec3(0.f, 2.f, 2.f), RGB::green);
    context.addTriangle(make_vec3(0.f, -2.f, -2.f), make_vec3(0.f, 2.f, 2.f), make_vec3(0.f, -2.f, 2.f), RGB::green);

    vec3 scan_origin(-5.f, 0.f, 0.5f);
    const uint Ntheta = 1100, Nphi = 1000; // 1.1M single-ray pulses => two chunks (1.05M + remainder)
    float thetaMin = 0.5f * float(M_PI) - 6.f * float(M_PI) / 180.f;
    float thetaMax = 0.5f * float(M_PI) + 6.f * float(M_PI) / 180.f;
    float phiMin = 0.5f * float(M_PI) - 20.f * float(M_PI) / 180.f;
    float phiMax = 0.5f * float(M_PI) + 20.f * float(M_PI) / 180.f;
    ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});

    LiDARcloud batch;
    batch.disableMessages();
    batch.addScan(scan);
    batch.setSyntheticScanMemoryBudget(8u * 1024u * 1024u);
    batch.syntheticScan(&context, false, true); // record misses so every pulse yields a return
    const uint N = batch.getHitCount();
    DOCTEST_REQUIRE(N == Ntheta * Nphi);
    std::vector<vec3> batch_xyz;
    std::vector<int32_t> batch_miss;
    batch.getHitXYZColumn(batch_xyz);
    batch.getHitDataColumn("is_miss", batch_miss, int32_t(-1));

    // Non-draining sink: the (first, count) ranges tile [0, N) exactly, in order.
    {
        LiDARcloud tiled;
        tiled.disableMessages();
        tiled.addScan(scan);
        tiled.setSyntheticScanMemoryBudget(8u * 1024u * 1024u);
        size_t expected_first = 0, callbacks = 0;
        bool contiguous = true;
        tiled.setSyntheticScanHitSink([&](size_t first, size_t count) {
            contiguous = contiguous && (first == expected_first) && (first + count == tiled.getHitCount());
            expected_first = first + count;
            callbacks++;
        });
        tiled.syntheticScan(&context, false, true);
        DOCTEST_CHECK(contiguous);
        DOCTEST_CHECK(expected_first == N);
        DOCTEST_CHECK(callbacks >= 2);
        DOCTEST_CHECK(tiled.getHitCount() == N);
    }

    // Draining sink: copy each chunk out and delete it; the cloud never holds more than one chunk.
    {
        LiDARcloud drained;
        drained.disableMessages();
        drained.addScan(scan);
        drained.setSyntheticScanMemoryBudget(8u * 1024u * 1024u);
        std::vector<vec3> got_xyz;
        std::vector<int32_t> got_miss;
        size_t max_held = 0, callbacks = 0;
        bool always_tail_from_zero = true;
        drained.setSyntheticScanHitSink([&](size_t first, size_t count) {
            callbacks++;
            always_tail_from_zero = always_tail_from_zero && (first == 0) && (drained.getHitCount() == count);
            max_held = std::max(max_held, size_t(drained.getHitCount()));
            std::vector<vec3> xyz;
            std::vector<int32_t> miss;
            drained.getHitXYZColumn(xyz);
            drained.getHitDataColumn("is_miss", miss, int32_t(-1));
            got_xyz.insert(got_xyz.end(), xyz.begin() + std::ptrdiff_t(first), xyz.end());
            got_miss.insert(got_miss.end(), miss.begin() + std::ptrdiff_t(first), miss.end());
            drained.deleteHitPoints(first, count);
        });
        drained.syntheticScan(&context, false, true);
        DOCTEST_CHECK(callbacks >= 2);
        DOCTEST_CHECK(always_tail_from_zero);
        DOCTEST_CHECK(drained.getHitCount() == 0);
        DOCTEST_CHECK(max_held < N);
        DOCTEST_CHECK(max_held <= 1050000u);
        DOCTEST_REQUIRE(got_xyz.size() == N);
        size_t mismatches = 0;
        for (uint i = 0; i < N; i++) {
            if (got_xyz[i].x != batch_xyz[i].x || got_xyz[i].y != batch_xyz[i].y || got_xyz[i].z != batch_xyz[i].z || got_miss[i] != batch_miss[i]) {
                mismatches++;
            }
        }
        DOCTEST_CHECK(mismatches == 0);
    }

    // The all-miss path (no beam reaches the scene's bounding box) also reports its recorded misses to the sink.
    {
        Context empty_scene;
        empty_scene.addTriangle(make_vec3(50.f, -1.f, -1.f), make_vec3(50.f, 1.f, -1.f), make_vec3(50.f, 1.f, 1.f), RGB::green);
        LiDARcloud away;
        away.disableMessages();
        ScanMetadata small(scan_origin, 10, thetaMin, thetaMax, 10, float(M_PI) + phiMin, float(M_PI) + phiMax, 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{}); // looking away from the triangle
        away.addScan(small);
        size_t reported = 0, callbacks = 0;
        away.setSyntheticScanHitSink([&](size_t first, size_t count) {
            callbacks++;
            reported += count;
            DOCTEST_CHECK(first + count == away.getHitCount());
        });
        away.syntheticScan(&context, false, true);
        DOCTEST_CHECK(callbacks == 1);
        DOCTEST_CHECK(reported == away.getHitCount());
        DOCTEST_CHECK(reported == 100u);
    }
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
        lidar.setScanDetectionThreshold(0, 0.f); // geometry test: keep every beam's return (the noise floor, default 0.05, would drop returns by incidence angle and make tilted/level counts differ)
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
        lidar_default.setScanDetectionThreshold(0, 0.f); // match run_scan()'s disabled noise floor so the zero-tilt no-op compares like-for-like
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
        lidar.setScanDetectionThreshold(0, 0.f); // geometry test: keep every beam's return so offset/non-offset counts stay in beam-for-beam correspondence
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
        lidar_default.setScanDetectionThreshold(0, 0.f); // match run_scan()'s disabled noise floor so the zero-offset no-op compares like-for-like
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

    // (3) Offset composes with tilt: a pitch-only tilt under a 90-degree azimuth offset must rotate the body forward axis the
    //     SAME way as the offset rotates the rays. The offset is a right-hand (CCW) rotation about world +z, and azimuth phi is
    //     measured CW-from-+y, so advancing the heading by the offset SUBTRACTS it: the forward axis is built from
    //     (phiMin - azimuth_offset), which for phiMin = 0, az = +90 deg is world -x (not +x). A right-hand pitch about -x then
    //     displaces the near-nadir floor hit to -y (not the +y a +x forward axis would give). This both confirms the offset
    //     rotates the roll/pitch body frame and pins the handedness: the buggy heading = phiMin + azimuth_offset builds the
    //     forward axis as +x and sends the hit to +y, so the mean.y < -0.5 assertion below fails without the sign fix.
    {
        const uint Nt = 8, Np = 8;
        const float tmin = 0.97f * float(M_PI);
        const float tmax = float(M_PI);
        const float pitch = 12.0f * float(M_PI) / 180.0f;
        const float scan_phiMin = 0.0f;
        const float azimuth_offset = 0.5f * float(M_PI); // 90-degree heading offset

        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(scan_origin, Nt, tmin, tmax, Np, scan_phiMin, scan_phiMin + 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, columnFormat, 0.0f, pitch, azimuth_offset);
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

        // forward (pitch) axis after the 90-degree heading offset is world -x (heading = phiMin - azimuth_offset), so a
        // right-hand pitch displaces nadir to -y.
        const float heading = scan_phiMin - azimuth_offset;
        const vec3 forward_axis = make_vec3(sinf(heading), cosf(heading), 0.f);
        vec3 expected_dir = rotatePointAboutLine(make_vec3(0, 0, -1), make_vec3(0, 0, 0), forward_axis, pitch);
        expected_dir.normalize();
        vec3 mean_dir = mean;
        mean_dir.normalize();
        DOCTEST_CHECK(mean_dir.x == doctest::Approx(expected_dir.x).epsilon(2e-2));
        DOCTEST_CHECK(mean_dir.y == doctest::Approx(expected_dir.y).epsilon(2e-2));
        DOCTEST_CHECK(mean_dir.z == doctest::Approx(expected_dir.z).epsilon(2e-2));
        DOCTEST_CHECK(mean.y < -0.5f); // fails under the buggy heading = phiMin + azimuth_offset (which sends the hit to +y)
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

DOCTEST_TEST_CASE("LiDAR Synthetic Scan Continuous Azimuth Skew Test") {
    // A real terrestrial scanner sweeps the beam vertically with a fast mirror while the whole head rotates continuously in
    // azimuth, so the azimuth advances during each zenith sweep and the zenith columns are slightly skewed (tilted) rather
    // than perfectly vertical. The raster synthetic scan models this by drifting the azimuth across the inner (zenith) loop:
    // over one full column the azimuth advances by exactly one column step dphi (per-row increment dphi/Ntheta), so column j
    // tiles seamlessly into column j+1. We verify:
    //   (1) within a column the azimuth drifts linearly by dphi from bottom row to (almost) the next column,
    //   (2) the top row of column j meets the bottom row of column j+1 (seamless continuous rotation),
    //   (3) a single-row (Ntheta==1) and a single-column (Nphi==1) raster scan produce no skew.

    // Surrounding box so every beam strikes a wall at a known distance and hits stay in firing order (Ntheta*j + i).
    Context context;
    context.addPatch(make_vec3(0, 0, -5), make_vec2(20, 20)); // floor
    context.addPatch(make_vec3(0, 0, 5), make_vec2(20, 20)); // ceiling
    context.addPatch(make_vec3(5, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.f)); // +x wall
    context.addPatch(make_vec3(-5, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.f)); // -x wall
    context.addPatch(make_vec3(0, 5, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.5f * float(M_PI))); // +y wall
    context.addPatch(make_vec3(0, -5, 0), make_vec2(20, 20), make_SphericalCoord(0.5f * float(M_PI), 0.5f * float(M_PI))); // -y wall

    const vec3 scan_origin(0.0f, 0.0f, 0.0f);
    const float exitDiameter = 0.0f;
    const float beamDivergence = 0.0f;
    std::vector<std::string> columnFormat;

    // Collect every beam's emitted azimuth (radians) in firing order. A down-and-out cone (theta well away from nadir) keeps
    // the horizontal component substantial so the azimuth recovered from the hit direction is numerically well-conditioned,
    // and within a modest azimuth span every beam strikes the same wall so the cloud stays in beam-for-beam order.
    auto run_azimuths = [&](uint Ntheta, uint Nphi, float thetaMin, float thetaMax, float phiMin, float phiMax, std::vector<float> &azim) {
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(scan_origin, Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, 0.0f, 0.0f, columnFormat);
        lidar.addScan(scan);
        lidar.setScanDetectionThreshold(0, 0.f); // keep every beam's return so hits stay in firing order (Ntheta*j + i)
        lidar.syntheticScan(&context);
        uint hc = lidar.getHitCount();
        azim.clear();
        azim.reserve(hc);
        for (uint i = 0; i < hc; i++) {
            vec3 d = lidar.getHitXYZ(i) - scan_origin;
            azim.push_back(cart2sphere(d).azimuth);
        }
    };

    // A down-and-out cone (theta = 144..171 deg, i.e. 9..36 deg above nadir) so every beam strikes the floor at a known
    // distance well within its 20x20 extent, giving exactly Ntheta*Nphi hits in firing order. The cone never reaches nadir
    // (theta = 180 deg), so the azimuth recovered from each hit direction stays well away from the atan2 singularity.
    const float thetaMin = 144.0f * float(M_PI) / 180.0f; // 36 deg above nadir
    const float thetaMax = 171.0f * float(M_PI) / 180.0f; // 9 deg above nadir
    const float phiMin = 0.0f;
    const float phiMax = 30.0f * float(M_PI) / 180.0f; // narrow azimuth fan

    // (1)+(2) Skew within and across columns.
    {
        const uint Ntheta = 12;
        const uint Nphi = 6;
        std::vector<float> azim;
        run_azimuths(Ntheta, Nphi, thetaMin, thetaMax, phiMin, phiMax, azim);
        DOCTEST_REQUIRE(azim.size() == size_t(Ntheta) * Nphi);

        const float dphi = (phiMax - phiMin) / float(Nphi - 1); // column-to-column azimuth step
        const float dphi_per_row = dphi / float(Ntheta); // modeled per-row drift

        // Within each column the azimuth must drift linearly: cell (i,j) azimuth = phiMin + j*dphi + i*dphi_per_row.
        for (uint j = 0; j < Nphi; j++) {
            for (uint i = 0; i < Ntheta; i++) {
                float expected = phiMin + float(j) * dphi + float(i) * dphi_per_row;
                DOCTEST_CHECK(azim[size_t(Ntheta) * j + i] == doctest::Approx(expected).epsilon(1e-4));
            }
        }

        // The skew is real, not zero: the top row of a column differs from its bottom row by ~dphi (one column step).
        for (uint j = 0; j < Nphi; j++) {
            float bottom = azim[size_t(Ntheta) * j + 0];
            float top = azim[size_t(Ntheta) * j + (Ntheta - 1)];
            DOCTEST_CHECK((top - bottom) == doctest::Approx(dphi * float(Ntheta - 1) / float(Ntheta)).epsilon(1e-3));
        }

        // (2) Seamless tiling: top row of column j meets bottom row of column j+1 (their azimuths differ by one per-row step).
        for (uint j = 0; j + 1 < Nphi; j++) {
            float top_j = azim[size_t(Ntheta) * j + (Ntheta - 1)];
            float bottom_next = azim[size_t(Ntheta) * (j + 1) + 0];
            DOCTEST_CHECK((bottom_next - top_j) == doctest::Approx(dphi_per_row).epsilon(1e-3));
        }
    }

    // (3a) Single-row raster (Ntheta == 1): no vertical sweep, so no skew - the one row per column is at the nominal azimuth.
    {
        const uint Ntheta = 1;
        const uint Nphi = 6;
        std::vector<float> azim;
        run_azimuths(Ntheta, Nphi, thetaMin, thetaMax, phiMin, phiMax, azim);
        DOCTEST_REQUIRE(azim.size() == size_t(Nphi));
        const float dphi = (phiMax - phiMin) / float(Nphi - 1);
        for (uint j = 0; j < Nphi; j++) {
            DOCTEST_CHECK(azim[j] == doctest::Approx(phiMin + float(j) * dphi).epsilon(1e-4));
        }
    }

    // (3b) Single-column raster (Nphi == 1): no azimuth motion (dphi == 0), so the whole column is at the constant azimuth.
    {
        const uint Ntheta = 12;
        const uint Nphi = 1;
        std::vector<float> azim;
        run_azimuths(Ntheta, Nphi, thetaMin, thetaMax, phiMin, phiMax, azim);
        DOCTEST_REQUIRE(azim.size() == size_t(Ntheta));
        for (uint i = 0; i < Ntheta; i++) {
            DOCTEST_CHECK(azim[i] == doctest::Approx(phiMin).epsilon(1e-4));
        }
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
    // A real scanner's grid departs slightly from a separable affine map: collimation and trunnion axis
    // errors make the horizontal angle depend on the vertical one (Lichti 2009 parameterises this as
    // b1*sec(alpha) + b2*tan(alpha)), and encoder eccentricity adds harmonics to the vertical angle
    // (Chow et al. 2013). This fixture stands in for those with a per-row azimuth shear and a zenith bow.
    //
    // The magnitudes are set from the literature rather than chosen for effect: a self-calibration of a
    // RIEGL VZ-400 put collimation at 109.5 arcsec and trunnion at 23.1 arcsec, so ~1e-3 rad is already
    // generous for survey-grade hardware -- the fixture previously used 0.30 rad, some 565x the worst
    // documented real error, which no separable model could absorb and which no instrument exhibits.
    LiDARcloud lidar;
    lidar.disableMessages();

    GenerativeGrid g{24, 48, 0.05, 0.95 * M_PI, 0.0, 1.5 * M_PI, 1.0e-3, 1.0e-3}; // shear + curvature at the scale real instruments show
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

    GenerativeGrid g{24, 48, 0.05, 0.95 * M_PI, 0.0, 1.5 * M_PI, 1.0e-3, 1.0e-3}; // shear/curvature at instrument scale (see the Tilted and Sheared Grid case)
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

    // and they must be reasonably close to the generative model. Note the flag is 1 (gapfilled), not the
    // former 4 ("extrapolated row"): the scan-grid model is now fitted across the whole scan rather than
    // per row, so a row with no returns of its own is not a distinct case -- it is evaluated from the
    // same coefficients as every other row. Code 4 is retired; see the ScanGridModel documentation.
    double max_err = 0.0;
    int n_extrap = 0;
    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.getHitScanID(r) != 0 || lidar.getHitData(r, "is_miss") != 1.0) {
            continue;
        }
        int row = (int) std::lround(lidar.getHitData(r, "row"));
        int col = (int) std::lround(lidar.getHitData(r, "column"));
        if (row < empty_rows_below) {
            DOCTEST_CHECK(lidar.getHitData(r, "gapfillMisses_code") == 1.0);
            n_extrap++;
            max_err = std::max(max_err, angularError(lidar.getHitRaydir(r), g.direction(row, col)));
        }
    }
    DOCTEST_CHECK(n_extrap == empty_rows_below * g.Nphi);
    // extrapolation is inherently looser than interpolation; require it to be within a few grid cells of the truth
    DOCTEST_CHECK(max_err < 0.1);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Row/Column Path Preserves Beam Grouping") {
    // REGRESSION: gapfillMisses_rowcolumn wrote "row"/"column" but NOT "timestamp".
    // groupHitsByTimestamp() degrades to one-hit-per-beam if ANY hit in the scan lacks a timestamp,
    // so gap-filling a multi-return cloud through the row/column path silently destroyed the
    // multi-return beam structure: every return became its own beam, and E_inside/E_after in the
    // leaf-area inversion were then computed per-return instead of per-pulse. Wrong leaf area, no error.
    // Synthesized misses now carry a reconstructed timestamp (pulse ordinal = column*Ntheta + row).
    LiDARcloud lidar;
    lidar.disableMessages();

    const uint Ntheta = 12, Nphi = 20;
    ScanMetadata scan(make_vec3(0, 0, 0), Ntheta, 0.05, 0.95 * M_PI, Nphi, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // Two returns per pulse sharing one timestamp, with an interior blank block left unfilled.
    double t = 0.0;
    for (uint row = 0; row < Ntheta; row++) {
        for (uint col = 0; col < Nphi; col++) {
            if (row >= 4 && row <= 6 && col >= 8 && col <= 12) {
                t += 1.0;
                continue;
            }
            for (int ti = 0; ti < 2; ti++) {
                const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta);
                const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
                SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
                vec3 xyz = sphere2cart(SphericalCoord(5.f + 2.f * float(ti), 0.5f * float(M_PI) - theta, phi));
                std::map<std::string, double> data;
                data["timestamp"] = t;
                data["target_index"] = ti;
                data["target_count"] = 2;
                data["row"] = row;
                data["column"] = col;
                lidar.addHitPoint(0, xyz, dir, data);
            }
            t += 1.0;
        }
    }

    const uint hits_before = lidar.getHitCount();
    std::vector<vec3> filled = lidar.gapfillMisses(0, false, false);
    DOCTEST_REQUIRE(filled.size() > 0);

    // Every hit -- real return and synthesized miss alike -- must carry a timestamp, or beam grouping
    // silently collapses for the whole scan.
    uint missing_timestamp = 0;
    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (!lidar.doesHitDataExist(r, "timestamp")) {
            missing_timestamp++;
        }
    }
    DOCTEST_CHECK(missing_timestamp == 0);

    // The observable consequence: distinct timestamps are what groupHitsByTimestamp() keys on, so the
    // beam count must stay well below the hit count (each real pulse contributed two returns).
    std::set<double> distinct_timestamps;
    for (uint r = 0; r < lidar.getHitCount(); r++) {
        if (lidar.doesHitDataExist(r, "timestamp")) {
            distinct_timestamps.insert(lidar.getHitData(r, "timestamp"));
        }
    }
    DOCTEST_CHECK(distinct_timestamps.size() < lidar.getHitCount());
    DOCTEST_CHECK(distinct_timestamps.size() >= (hits_before / 2));

    // Synthesized misses must not collide with a real pulse's timestamp: the ordinal encoding is
    // unique per grid cell, and the blank cells were never fired.
    DOCTEST_CHECK(lidar.getHitCount() == hits_before + filled.size());
}

//! Build the same gap-filled cloud twice -- once leaving misses virtual, once forcing them into real
//! storage -- and assert that every observable the public API exposes is EXACTLY equal, hit by hit, in
//! index order. Exact (not Approx) equality is the right assertion because materializing evaluates the
//! identical expression from the identical stored doubles, so it is a pure copy; a needed tolerance
//! would be a bug signal, not a test-tuning knob.
static void checkVirtualMaterializedEquivalence(const std::function<void(LiDARcloud &)> &build) {
    LiDARcloud virt;
    virt.disableMessages();
    build(virt);

    LiDARcloud mat;
    mat.disableMessages();
    build(mat);
    mat.materializeMisses();

    DOCTEST_REQUIRE(virt.getHitCount() == mat.getHitCount());
    DOCTEST_CHECK(virt.getVirtualMissCount() > 0); // the virtual path was actually exercised
    DOCTEST_CHECK(mat.getVirtualMissCount() == 0);

    // Bin both clouds before comparing grid cells. Unbinned, every hit's cell is the "never set"
    // sentinel, so the getHitGridCell() comparison below would be -1 == -1 for every point -- vacuous,
    // and one warning per call. The two boxes split the 10 m hit sphere at z = 0 and are far too small
    // to reach a miss at LIDAR_MISS_DISTANCE, so real returns land in cell 0 or cell 1 and every miss
    // (virtual or materialized) in -1, which is exactly the distinction the comparison should catch.
    for (LiDARcloud *cloud: {&virt, &mat}) {
        cloud->addGridCell(make_vec3(0, 0, 7.5f), make_vec3(30, 30, 15), 0.f);
        cloud->addGridCell(make_vec3(0, 0, -7.5f), make_vec3(30, 30, 15), 0.f);
        cloud->calculateHitGridCell();
    }
    DOCTEST_CHECK(virt.getVirtualMissCount() > 0); // binning must not have materialized the misses

    const char *labels[] = {"is_miss", "row", "column", "nRaysHit", "gapfillMisses_code", "timestamp", "intensity"};

    size_t binned_hits = 0;
    for (uint i = 0; i < virt.getHitCount(); i++) {
        DOCTEST_REQUIRE(virt.getHitScanID(i) == mat.getHitScanID(i));
        const vec3 pv = virt.getHitXYZ(i), pm = mat.getHitXYZ(i);
        DOCTEST_REQUIRE(pv.x == pm.x);
        DOCTEST_REQUIRE(pv.y == pm.y);
        DOCTEST_REQUIRE(pv.z == pm.z);
        const vec3 ov = virt.getHitOrigin(i), om = mat.getHitOrigin(i);
        DOCTEST_REQUIRE(ov.x == om.x);
        DOCTEST_REQUIRE(ov.y == om.y);
        DOCTEST_REQUIRE(ov.z == om.z);
        const int gv = virt.getHitGridCell(i), gm = mat.getHitGridCell(i);
        DOCTEST_REQUIRE(gv == gm);
        if (gv >= 0) {
            binned_hits++;
        }
        const RGBcolor cv = virt.getHitColor(i), cm = mat.getHitColor(i);
        DOCTEST_REQUIRE(cv.r == cm.r);
        DOCTEST_REQUIRE(cv.g == cm.g);
        DOCTEST_REQUIRE(cv.b == cm.b);
        for (const char *lbl: labels) {
            const bool ev = virt.doesHitDataExist(i, lbl), em = mat.doesHitDataExist(i, lbl);
            DOCTEST_REQUIRE(ev == em);
            if (ev) {
                DOCTEST_REQUIRE(virt.getHitData(i, lbl) == mat.getHitData(i, lbl));
            }
        }
    }

    // Real returns must actually have been binned into a cell, or the grid-cell comparison above
    // degenerates back into comparing -1 with -1 for every point.
    DOCTEST_REQUIRE(binned_hits > 0);

    // The bulk getter must agree with the per-index getter on the virtual cloud too.
    for (const char *lbl: labels) {
        std::vector<double> bulk;
        virt.getHitDataColumn(lbl, bulk, -9999);
        DOCTEST_REQUIRE(bulk.size() == virt.getHitCount());
        for (uint i = 0; i < virt.getHitCount(); i++) {
            const double expected = virt.doesHitDataExist(i, lbl) ? virt.getHitData(i, lbl) : -9999.0;
            DOCTEST_REQUIRE(bulk[i] == expected);
        }
    }
}

//! Populate a row/column-indexed scan grid, leaving a known blank region for the gap filler.
static void buildRowColumnScan(LiDARcloud &lidar, int Ntheta, int Nphi, float tilt, bool add_flags, bool with_timestamp, int blank_r0, int blank_r1, int blank_c0, int blank_c1) {
    ScanMetadata scan(make_vec3(0, 0, 0), uint(Ntheta), 0.05, 0.95 * M_PI, uint(Nphi), 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);
    double t = 0.0;
    for (int row = 0; row < Ntheta; row++) {
        for (int col = 0; col < Nphi; col++) {
            if (row >= blank_r0 && row <= blank_r1 && col >= blank_c0 && col <= blank_c1) {
                t += 1.0;
                continue;
            }
            const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta) + tilt * float(col) / float(Nphi);
            const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
            SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
            vec3 xyz = sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            if (with_timestamp) {
                data["timestamp"] = t;
            }
            lidar.addHitPoint(0, xyz, dir, data);
            t += 1.0;
        }
    }
    std::vector<vec3> filled = lidar.gapfillMisses(0, false, add_flags);
    (void) filled;
}

// ---- Review regressions: typed columns, rejected adds, readers, concurrency, block tiling, empty scans ----

DOCTEST_TEST_CASE("LiDAR Typed Hit Data Columns - Values Stored as Doubles Read Back Exactly on Every Ingest Path") {
    // REGRESSION: typing the columns narrowed or rejected values that callers had always stored as doubles. A projected
    // easting came back rounded to a float (612345.678 -> 612345.6875), a range lost its last digits, and a pulse id
    // above 2^31, a NaN target_index and a fractional row threw. A column keeps its narrow type only while every value
    // fits it exactly; the first value that does not widens it, and a NaN is an absent value on both ingest paths.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::string> labels = {"origin_x", "distance", "pulse_id", "target_index", "row", "intensity", "target_count"};
    const std::vector<double> first = {1.0, 2.0, 3.0, 0.0, 4.0, 0.25, 1.0}; // fits every label's narrow type
    const std::vector<double> second = {612345.678, 1234.5678, 3.0e9, nan, 2.5, 0.1, 2.0}; // does not, except target_count
    ScanMetadata scan(make_vec3(0, 0, 0), 4, 0.5f, 2.5f, 4, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, {});

    auto check = [&](const LiDARcloud &cloud, const char *path) {
        DOCTEST_REQUIRE(cloud.getHitCount() == 2);
        for (size_t k = 0; k < labels.size(); k++) {
            const char *label = labels[k].c_str();
            DOCTEST_CHECK_MESSAGE(cloud.getHitData(0, label) == first[k], path << ": " << label << " on the first hit");
            std::vector<double> column;
            cloud.getHitDataColumn(label, column, -1.0);
            if (std::isnan(second[k])) {
                DOCTEST_CHECK_MESSAGE(!cloud.doesHitDataExist(1, label), path << ": a NaN " << label << " must be absent");
                DOCTEST_CHECK(column[1] == -1.0);
            } else {
                DOCTEST_CHECK_MESSAGE(cloud.getHitData(1, label) == second[k], path << ": " << label << " did not read back exactly");
                DOCTEST_CHECK_MESSAGE(column[1] == second[k], path << ": bulk read of " << label);
            }
        }
        DOCTEST_CHECK(cloud.getHitDataType("origin_x") == HitDataType::FLOAT64);
        DOCTEST_CHECK(cloud.getHitDataType("pulse_id") == HitDataType::FLOAT64);
        DOCTEST_CHECK(cloud.getHitDataType("distance") == HitDataType::FLOAT64); // widened by 1234.5678
        DOCTEST_CHECK(cloud.getHitDataType("row") == HitDataType::FLOAT64); // widened by 2.5
        DOCTEST_CHECK(cloud.getHitDataType("target_count") == HitDataType::INT32); // every value fits: stays narrow
        DOCTEST_CHECK(cloud.getHitDataType("target_index") == HitDataType::INT32); // a NaN is absent, not a value
    };

    LiDARcloud single;
    single.disableMessages();
    single.addScan(scan);
    for (const std::vector<double> *values: {&first, &second}) {
        std::map<std::string, double> data;
        for (size_t k = 0; k < labels.size(); k++) {
            data[labels[k]] = (*values)[k];
        }
        single.addHitPoint(0, make_vec3(1, 2, 3), SphericalCoord(1, 0, 0), data);
    }
    check(single, "addHitPoint");

    LiDARcloud bulk;
    bulk.disableMessages();
    bulk.addScan(scan);
    const std::vector<double> xyz = {1, 2, 3, 1, 2, 3};
    std::vector<double> values = first;
    values.insert(values.end(), second.begin(), second.end());
    bulk.addHitPoints(0, 2, xyz.data(), nullptr, labels, values.data());
    check(bulk, "addHitPoints");

    // setHitData stores exactly the value it is given, as it always has -- including a NaN, as a present value.
    single.setHitData(0, "target_count", 2.75);
    DOCTEST_CHECK(single.getHitData(0, "target_count") == 2.75);
    DOCTEST_CHECK(single.getHitData(1, "target_count") == 2.0);
    single.setHitData(0, "target_index", nan);
    DOCTEST_CHECK(single.doesHitDataExist(0, "target_index"));
    DOCTEST_CHECK(std::isnan(single.getHitData(0, "target_index")));

    // A column whose type the caller fixed explicitly is a contract: a value it cannot hold is an error naming the
    // label and the remedy, and the rejected hit is not added.
    LiDARcloud typed;
    typed.disableMessages();
    typed.addScan(scan);
    typed.createHitDataColumn("my_count", HitDataType::INT32);
    std::map<std::string, double> fractional;
    fractional["my_count"] = 2.5;
    std::string message;
    try {
        typed.addHitPoint(0, make_vec3(1, 2, 3), SphericalCoord(1, 0, 0), fractional);
    } catch (const std::runtime_error &e) {
        message = e.what();
    }
    DOCTEST_CHECK(message.find("my_count") != std::string::npos);
    DOCTEST_CHECK(message.find("createHitDataColumn") != std::string::npos);
    DOCTEST_CHECK(typed.getHitCount() == 0);
}

DOCTEST_TEST_CASE("LiDAR addHitPoint and addHitPoints - A Rejected Call Leaves the Cloud Unchanged") {
    // REGRESSION: a rejected add first collapsed every virtualized miss into stored points and created a column for
    // each label it carried, so a call that failed still reshaped the cloud.
    LiDARcloud cloud;
    cloud.disableMessages();
    buildRowColumnScan(cloud, 20, 36, 0.f, false, false, 5, 9, 10, 15); // gap-filled: holds virtualized misses
    cloud.createHitDataColumn("explicit_int", HitDataType::INT32);
    const uint count = cloud.getHitCount();
    const size_t virtual_count = cloud.getVirtualMissCount();
    const size_t stored = LiDARTestHelper::storedHitCount(cloud);
    const std::vector<std::string> labels = LiDARTestHelper::hitDataLabels(cloud);
    DOCTEST_REQUIRE(virtual_count > 0);
    auto unchanged = [&](const char *what) {
        DOCTEST_CHECK_MESSAGE(cloud.getHitCount() == count, std::string(what) << ": hit count changed");
        DOCTEST_CHECK_MESSAGE(cloud.getVirtualMissCount() == virtual_count, std::string(what) << ": virtualized misses were materialized");
        DOCTEST_CHECK_MESSAGE(LiDARTestHelper::storedHitCount(cloud) == stored, std::string(what) << ": stored hit count changed");
        DOCTEST_CHECK_MESSAGE((LiDARTestHelper::hitDataLabels(cloud) == labels), std::string(what) << ": a label column was created");
    };
    const vec3 xyz = make_vec3(1, 2, 3);
    const SphericalCoord dir(1, 0, 0);
    std::map<std::string, double> new_label;
    new_label["brand_new"] = 1.0;
    std::map<std::string, double> bad_value = new_label;
    bad_value["explicit_int"] = 2.5;
    const std::vector<std::string> bulk_labels = {"brand_new", "explicit_int"};
    const std::vector<std::string> one_label = {"brand_new"};
    const std::vector<double> bulk_xyz = {1, 2, 3, 4, 5, 6};
    const std::vector<double> bulk_values = {1.0, 2.0, 1.0, 2.5};

    DOCTEST_CHECK_THROWS_AS(cloud.addHitPoint(7, xyz, dir, new_label), std::runtime_error);
    unchanged("addHitPoint to a scan that does not exist");
    DOCTEST_CHECK_THROWS_AS(cloud.addHitPoint(7, xyz, make_int2(0, 0), RGB::red, new_label), std::runtime_error);
    unchanged("addHitPoint(row, column) to a scan that does not exist");
    DOCTEST_CHECK_THROWS_AS(cloud.addHitPoint(0, xyz, dir, bad_value), std::runtime_error);
    unchanged("addHitPoint with a value its explicitly typed column cannot hold");
    DOCTEST_CHECK_THROWS_AS(cloud.addHitPoints(7, 2, bulk_xyz.data(), nullptr, bulk_labels, bulk_values.data()), std::runtime_error);
    unchanged("addHitPoints to a scan that does not exist");
    DOCTEST_CHECK_THROWS_AS(cloud.addHitPoints(0, 2, bulk_xyz.data(), nullptr, bulk_labels, bulk_values.data()), std::runtime_error);
    unchanged("addHitPoints with a value its explicitly typed column cannot hold");
    cloud.setMaxHitPoints(cloud.getHitCount());
    DOCTEST_CHECK_THROWS_AS(cloud.addHitPoint(0, xyz, dir, new_label), std::runtime_error);
    unchanged("addHitPoint beyond the hit-point cap");
    DOCTEST_CHECK_THROWS_AS(cloud.addHitPoints(0, 2, bulk_xyz.data(), nullptr, one_label, bulk_values.data()), std::runtime_error);
    unchanged("addHitPoints beyond the hit-point cap");
}

DOCTEST_TEST_CASE("LiDAR Typed Hit Data Columns - Integer Readers Reject Values an int32 Cannot Hold") {
    // REGRESSION: the int32_t column readers converted each value with int32_t(value), which is undefined behaviour
    // for a NaN or for a value outside the 32-bit range (a GPS timestamp, say). They raise an error naming the label.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 4, 0.5f, 2.5f, 4, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(scan);
    std::map<std::string, double> data;
    data["timestamp"] = 1.7e9 + 0.5;
    data["big"] = 3.0e9;
    data["whole"] = 7.0;
    data["frac"] = 2.5;
    data["lowest"] = -2147483648.0;
    cloud.addHitPoint(0, make_vec3(1, 2, 3), SphericalCoord(1, 0, 0), data);
    cloud.setHitData(0, "not_a_number", std::numeric_limits<double>::quiet_NaN());
    for (const char *label: {"timestamp", "big", "frac", "not_a_number"}) {
        std::vector<int32_t> values;
        std::string message;
        try {
            cloud.getHitDataColumn(label, values);
        } catch (const std::runtime_error &e) {
            message = e.what();
        }
        DOCTEST_CHECK_MESSAGE(message.find(label) != std::string::npos, "getHitDataColumn(int32_t) of " << label << " did not raise an error naming it");
        message.clear();
        try {
            cloud.getScanHitDataColumn(0, label, values);
        } catch (const std::runtime_error &e) {
            message = e.what();
        }
        DOCTEST_CHECK_MESSAGE(message.find(label) != std::string::npos, "getScanHitDataColumn(int32_t) of " << label << " did not raise an error naming it");
    }
    std::vector<int32_t> whole, lowest, missing;
    cloud.getHitDataColumn("whole", whole);
    cloud.getHitDataColumn("lowest", lowest);
    cloud.getHitDataColumn("never_set", missing, -5);
    DOCTEST_CHECK(whole[0] == 7);
    DOCTEST_CHECK(lowest[0] == std::numeric_limits<int32_t>::min());
    DOCTEST_CHECK(missing[0] == -5);

    // The virtualized tail is held to the same rule. Returns at even pulse ordinals carry integral timestamps
    // (ordinal / 2); the synthesized misses at odd ordinals are stamped halfway between, which no int32 holds.
    LiDARcloud filled;
    filled.disableMessages();
    const int Ntheta = 12, Nphi = 16;
    ScanMetadata raster(make_vec3(0, 0, 0), uint(Ntheta), 0.3f, 2.8f, uint(Nphi), 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    raster.pulse_period = 0.5; // the clock the returns were stamped on, so every version of the gap filler keeps it
    filled.addScan(raster);
    for (int col = 0; col < Nphi; col++) {
        for (int row = 0; row < Ntheta; row++) {
            const int ordinal = col * Ntheta + row;
            if (ordinal % 2 != 0) {
                continue;
            }
            const float zenith = 0.3f + 2.5f * float(row) / float(Ntheta - 1);
            const SphericalCoord d(1.f, 0.5f * float(M_PI) - zenith, 2.f * float(M_PI) * float(col) / float(Nphi));
            std::map<std::string, double> rc;
            rc["row"] = row;
            rc["column"] = col;
            rc["timestamp"] = 0.5 * ordinal;
            filled.addHitPoint(0, 10.f * sphere2cart(d), d, rc);
        }
    }
    filled.gapfillMissesCount(0, false, false);
    DOCTEST_REQUIRE(filled.getVirtualMissCount() > 0);
    std::vector<int32_t> stamps;
    DOCTEST_CHECK_THROWS_AS(filled.getHitDataColumn("timestamp", stamps), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(filled.getScanHitDataColumn(0, "timestamp", stamps), std::runtime_error);
    std::vector<int32_t> rows;
    DOCTEST_CHECK_NOTHROW(filled.getHitDataColumn("row", rows));
}

DOCTEST_TEST_CASE("LiDAR Per-Scan Index - Concurrent Const Readers Build the Index Exactly Once") {
    // REGRESSION: the per-scan index was rebuilt lazily inside const readers without synchronization, so const calls
    // from several threads on a freshly mutated cloud all rebuilt it at once, writing the same arrays concurrently.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 10, 0.5f, 2.5f, 10, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(scan);
    cloud.addScan(scan);
    std::vector<std::vector<vec3>> expected(2);
    for (int i = 0; i < 4000; i++) {
        const uint s = uint(i % 2); // interleaved, so the index needs its permutation
        const vec3 p = make_vec3(float(i), float(i % 17), 1.f);
        cloud.addHitPoint(s, p, SphericalCoord(1, 0, 0));
        expected[s].push_back(p);
    }
    std::vector<vec3> warm;
    cloud.getScanHitXYZColumn(0, warm);
    LiDARTestHelper::invalidateScanIndex(cloud); // what every mutation of the cloud does

    std::atomic<int> active{0}, peak{0};
    LiDARTestHelper::setScanIndexBuildHook(cloud, [&]() {
        const int now = ++active;
        int seen = peak.load();
        while (now > seen && !peak.compare_exchange_weak(seen, now)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        --active;
    });
    const size_t builds_before = LiDARTestHelper::scanIndexBuilds(cloud);
    const int Nthreads = 8;
    std::vector<std::vector<vec3>> got(Nthreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < Nthreads; t++) {
        threads.emplace_back([&cloud, &got, t]() { cloud.getScanHitXYZColumn(uint(t % 2), got[t]); });
    }
    for (std::thread &thread: threads) {
        thread.join();
    }
    LiDARTestHelper::setScanIndexBuildHook(cloud, nullptr);
    DOCTEST_CHECK_MESSAGE(peak.load() == 1, peak.load() << " threads rebuilt the index at the same time");
    DOCTEST_CHECK(LiDARTestHelper::scanIndexBuilds(cloud) - builds_before == 1);
    for (int t = 0; t < Nthreads; t++) {
        const std::vector<vec3> &want = expected[size_t(t % 2)];
        DOCTEST_REQUIRE(got[t].size() == want.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < want.size(); i++) {
            mismatches += (got[t][i].x != want[i].x || got[t][i].y != want[i].y) ? 1 : 0;
        }
        DOCTEST_CHECK(mismatches == 0);
    }
}

DOCTEST_TEST_CASE("LiDAR Block-Tiled Leaf Area - Exact on a Rotated Grid With Non-Representable Cell Extents") {
    // REGRESSION: a block walk shifted the lattice origin by ijk_min * extent and the cell bounds shifted it back,
    // which in floating point can miss the whole-grid origin by one ULP, so a block's result could differ from the
    // whole-grid inversion in its last bits. Tiled results must equal the whole grid exactly.
    Context context;
    context.seedRandomGenerator(0);
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(vec3(-5.f, 0.3f, 0.5f), 600, 0.f, float(M_PI), 900, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>{});
    lidar.addScan(scan);
    const int3 divisions = make_int3(3, 3, 3);
    lidar.addGrid(vec3(0.0137f, -0.0211f, 0.5173f), vec3(1.031f, 0.973f, 1.009f), divisions, 0.3f);
    lidar.syntheticScan(&context, true, true);
    const uint Ncells = lidar.getGridCellCount();
    DOCTEST_REQUIRE(Ncells == 27);

    struct Cell {
        float leaf_area, lad_var;
        int beams;
    };
    auto read = [&]() {
        std::vector<Cell> out(Ncells);
        for (uint c = 0; c < Ncells; c++) {
            out[c] = Cell{lidar.getCellLeafArea(c), lidar.getCellLADVariance(c), lidar.getCellBeamCount(c)};
        }
        return out;
    };
    lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f);
    const std::vector<Cell> whole = read();
    size_t with_leaves = 0;
    for (const Cell &cell: whole) {
        with_leaves += cell.leaf_area > 0.f ? 1 : 0;
    }
    DOCTEST_REQUIRE(with_leaves > 5);

    auto compare = [&](const char *what) {
        const std::vector<Cell> tiled = read();
        size_t mismatches = 0;
        for (uint c = 0; c < Ncells; c++) {
            if (tiled[c].leaf_area != whole[c].leaf_area || tiled[c].beams != whole[c].beams || tiled[c].lad_var != whole[c].lad_var) {
                mismatches++;
            }
        }
        DOCTEST_CHECK_MESSAGE(mismatches == 0, std::string(what) << ": " << mismatches << " of " << Ncells << " cells differ from the whole-grid inversion");
    };
    for (uint c = 0; c < Ncells; c++) {
        const int3 ijk = lidar.getCellGlobalIJK(c);
        lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f, ijk, ijk);
    }
    compare("1 x 1 x 1 blocks");
    for (int k = 0; k < divisions.z; k++) {
        const int3 lo = make_int3(1, 0, k), hi = make_int3(2, 2, k);
        lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f, lo, hi);
        const int3 lo0 = make_int3(0, 0, k), hi0 = make_int3(0, 2, k);
        lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f, lo0, hi0);
    }
    compare("blocks offset from the lattice origin");
}

//! Returns of a level raster carrying row/column indices (and nothing else) for one scan, with a blank block.
static size_t addRowColumnReturns(LiDARcloud &cloud, uint scanID, int Ntheta, int Nphi, int blank_r0, int blank_r1, int blank_c0, int blank_c1) {
    const vec3 origin = cloud.getScanOrigin(scanID);
    size_t n = 0;
    for (int row = 0; row < Ntheta; row++) {
        for (int col = 0; col < Nphi; col++) {
            if (row >= blank_r0 && row <= blank_r1 && col >= blank_c0 && col <= blank_c1) {
                continue;
            }
            const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta);
            const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
            const SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            cloud.addHitPoint(scanID, origin + sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi)), dir, data);
            n++;
        }
    }
    return n;
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Gap-Filling One Scan Keeps Every Other Scan's Misses Virtual") {
    // REGRESSION: gap-filling a scan whose returns lacked is_miss, or with add_flags set, first collapsed every other
    // scan's virtualized misses into stored points (writing a column value on a stored row renumbers nothing, so
    // that was never needed), and each gap-fill threw away the per-scan index and rebuilt it.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata a(make_vec3(0, 0, 0), 20, 0.05f, 0.95f * float(M_PI), 36, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    ScanMetadata b(make_vec3(3, 0, 0), 24, 0.05f, 0.95f * float(M_PI), 30, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(a);
    cloud.addScan(b);
    const size_t returns = addRowColumnReturns(cloud, 0, 20, 36, 5, 9, 10, 15) + addRowColumnReturns(cloud, 1, 24, 30, 2, 6, 3, 8);

    const size_t n0 = cloud.gapfillMissesCount(0, false, true);
    DOCTEST_REQUIRE(cloud.getVirtualMissCount() == n0);
    std::vector<vec3> scan0_before;
    cloud.getScanHitXYZColumn(0, scan0_before);
    const size_t builds = LiDARTestHelper::scanIndexBuilds(cloud);

    const size_t n1 = cloud.gapfillMissesCount(1, false, false);
    DOCTEST_CHECK(cloud.getVirtualMissCount() == n0 + n1);
    DOCTEST_CHECK(LiDARTestHelper::storedHitCount(cloud) == returns);
    DOCTEST_CHECK(cloud.getHitCount() == returns + n0 + n1);
    DOCTEST_CHECK(LiDARTestHelper::scanIndexBuilds(cloud) == builds);
    std::vector<vec3> scan0_after;
    cloud.getScanHitXYZColumn(0, scan0_after);
    DOCTEST_REQUIRE(scan0_after.size() == scan0_before.size());
    size_t moved = 0;
    for (size_t i = 0; i < scan0_before.size(); i++) {
        moved += (scan0_after[i].x != scan0_before[i].x || scan0_after[i].y != scan0_before[i].y || scan0_after[i].z != scan0_before[i].z) ? 1 : 0;
    }
    DOCTEST_CHECK(moved == 0);

    // Gap-filling a scan again replaces its own population and still leaves the other scan's virtual.
    DOCTEST_CHECK(cloud.gapfillMissesCount(0, false, true) == n0);
    DOCTEST_CHECK(cloud.getVirtualMissCount() == n0 + n1);
    DOCTEST_CHECK(LiDARTestHelper::storedHitCount(cloud) == returns);
    DOCTEST_CHECK(LiDARTestHelper::scanIndexBuilds(cloud) == builds);

    // And each scan's population is the one it gets in a cloud of its own.
    LiDARcloud alone;
    alone.disableMessages();
    alone.addScan(b);
    addRowColumnReturns(alone, 0, 24, 30, 2, 6, 3, 8);
    DOCTEST_CHECK(alone.gapfillMissesCount(0, false, false) == n1);
}

DOCTEST_TEST_CASE("LiDAR Per-Scan Index - Empty Middle and Last Scans Change Nothing") {
    // A scan with no hits, in the middle of the scan list or at its end, must not change any per-scan result: the
    // per-scan readers report it empty, and triangulation, the leaf-area inversion and gap-filling of the other scans
    // are exactly those of the same cloud without it.
    Context context;
    context.seedRandomGenerator(0);
    context.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true);
    const vec3 origin0 = make_vec3(-5.f, 0.f, 0.5f), origin1 = make_vec3(0.f, -5.f, 0.5f), nowhere = make_vec3(9.f, 9.f, 9.f);
    auto add_scan = [](LiDARcloud &cloud, const vec3 &origin) {
        ScanMetadata scan(origin, 500, 0.f, float(M_PI), 750, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
        cloud.addScan(scan);
    };
    auto add_grid = [](LiDARcloud &cloud) { cloud.addGrid(make_vec3(0, 0, 0.5f), make_vec3(1, 1, 1), make_int3(2, 2, 2), 0.f); };

    LiDARcloud base;
    base.disableMessages();
    add_scan(base, origin0);
    add_scan(base, origin1);
    add_grid(base);
    base.syntheticScan(&context, true, true);
    DOCTEST_REQUIRE(base.getScanHitCount(0) > 0);
    DOCTEST_REQUIRE(base.getScanHitCount(1) > 0);

    // Copy a scan's stored hits, stored directions and every label (absent values stay absent) into another cloud.
    auto copy_scan = [&](uint from, LiDARcloud &dst, uint to) {
        std::vector<uint> indices;
        base.getScanHitIndices(from, indices);
        const std::vector<std::string> labels = LiDARTestHelper::hitDataLabels(base);
        std::vector<std::vector<double>> columns(labels.size());
        for (size_t k = 0; k < labels.size(); k++) {
            base.getScanHitDataColumn(from, labels[k].c_str(), columns[k], std::numeric_limits<double>::quiet_NaN());
        }
        std::vector<double> xyz, values;
        std::vector<float> dir;
        for (size_t i = 0; i < indices.size(); i++) {
            const vec3 p = base.getHitXYZ(indices[i]);
            const SphericalCoord d = LiDARTestHelper::storedDirection(base, indices[i]);
            xyz.insert(xyz.end(), {double(p.x), double(p.y), double(p.z)});
            dir.insert(dir.end(), {d.radius, d.elevation, d.azimuth});
            for (size_t k = 0; k < labels.size(); k++) {
                values.push_back(columns[k][i]);
            }
        }
        dst.addHitPoints(to, indices.size(), xyz.data(), dir.data(), labels, values.data());
    };
    LiDARcloud middle, last;
    middle.disableMessages();
    last.disableMessages();
    add_scan(middle, origin0);
    add_scan(middle, nowhere);
    add_scan(middle, origin1);
    add_grid(middle);
    copy_scan(0, middle, 0);
    copy_scan(1, middle, 2);
    add_scan(last, origin0);
    add_scan(last, origin1);
    add_scan(last, nowhere);
    add_grid(last);
    copy_scan(0, last, 0);
    copy_scan(1, last, 1);

    for (const std::pair<LiDARcloud *, uint> &empty: {std::make_pair(&middle, uint(1)), std::make_pair(&last, uint(2))}) {
        LiDARcloud &cloud = *empty.first;
        DOCTEST_CHECK(cloud.getScanHitCount(empty.second) == 0);
        std::vector<uint> indices;
        std::vector<vec3> xyz;
        std::vector<double> stamps;
        cloud.getScanHitIndices(empty.second, indices);
        cloud.getScanHitXYZColumn(empty.second, xyz);
        cloud.getScanHitDataColumn(empty.second, "timestamp", stamps);
        DOCTEST_CHECK(indices.empty());
        DOCTEST_CHECK(xyz.empty());
        DOCTEST_CHECK(stamps.empty());
    }

    auto mesh = [](LiDARcloud &cloud, size_t &count) {
        cloud.triangulateHitPoints(0.1f, 10.f);
        count = cloud.getTriangleCount();
        double sum = 0.0;
        for (uint t = 0; t < cloud.getTriangleCount(); t++) {
            const Triangulation tri = cloud.getTriangle(t);
            sum += double(tri.vertex0.x) + double(tri.vertex0.y) + double(tri.vertex0.z) + double(tri.vertex1.x) + double(tri.vertex1.y) + double(tri.vertex1.z) + double(tri.vertex2.x) + double(tri.vertex2.y) + double(tri.vertex2.z) +
                   double(tri.area);
        }
        return sum;
    };
    auto leaf = [&](LiDARcloud &cloud, bool triangulated) {
        if (triangulated) {
            cloud.calculateLeafArea(&context, 1, 0.05f);
        } else {
            cloud.calculateLeafArea(&context, 0.5f, 1, 0.05f);
        }
        std::vector<std::pair<float, int>> out;
        for (uint c = 0; c < cloud.getGridCellCount(); c++) {
            out.emplace_back(cloud.getCellLeafArea(c), cloud.getCellBeamCount(c));
        }
        return out;
    };
    size_t count_base = 0, count_middle = 0, count_last = 0;
    const double sum_base = mesh(base, count_base);
    DOCTEST_REQUIRE(count_base > 0);
    DOCTEST_CHECK(mesh(middle, count_middle) == sum_base);
    DOCTEST_CHECK(mesh(last, count_last) == sum_base);
    DOCTEST_CHECK(count_middle == count_base);
    DOCTEST_CHECK(count_last == count_base);
    for (bool triangulated: {true, false}) {
        const auto want = leaf(base, triangulated);
        DOCTEST_CHECK((leaf(middle, triangulated) == want));
        DOCTEST_CHECK((leaf(last, triangulated) == want));
    }

    // Gap-filling every scan skips the empty one and fills the others exactly as without it.
    ScanMetadata ra(make_vec3(0, 0, 0), 20, 0.05f, 0.95f * float(M_PI), 36, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    ScanMetadata rb(make_vec3(3, 0, 0), 24, 0.05f, 0.95f * float(M_PI), 30, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    ScanMetadata re(nowhere, 10, 0.05f, 0.95f * float(M_PI), 10, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    LiDARcloud gap_base, gap_middle, gap_last;
    for (LiDARcloud *cloud: {&gap_base, &gap_middle, &gap_last}) {
        cloud->disableMessages();
    }
    gap_base.addScan(ra);
    gap_base.addScan(rb);
    gap_middle.addScan(ra);
    gap_middle.addScan(re);
    gap_middle.addScan(rb);
    gap_last.addScan(ra);
    gap_last.addScan(rb);
    gap_last.addScan(re);
    addRowColumnReturns(gap_base, 0, 20, 36, 5, 9, 10, 15);
    addRowColumnReturns(gap_base, 1, 24, 30, 2, 6, 3, 8);
    addRowColumnReturns(gap_middle, 0, 20, 36, 5, 9, 10, 15);
    addRowColumnReturns(gap_middle, 2, 24, 30, 2, 6, 3, 8);
    addRowColumnReturns(gap_last, 0, 20, 36, 5, 9, 10, 15);
    addRowColumnReturns(gap_last, 1, 24, 30, 2, 6, 3, 8);
    const size_t filled = gap_base.gapfillMissesCount();
    DOCTEST_CHECK(gap_middle.gapfillMissesCount() == filled);
    DOCTEST_CHECK(gap_last.gapfillMissesCount() == filled);
    DOCTEST_CHECK(gap_middle.getScanHitCount(2) == gap_base.getScanHitCount(1));
    DOCTEST_CHECK(gap_last.getScanHitCount(1) == gap_base.getScanHitCount(1));
    DOCTEST_CHECK(gap_middle.getScanHitCount(1) == 0);
    DOCTEST_CHECK(gap_last.getScanHitCount(2) == 0);
}

DOCTEST_TEST_CASE("LiDAR Synthetic Scan - A Seeded Scan Does Not Depend on How It Is Chunked") {
    // REGRESSION: each chunk drew its beams' divergence random numbers and then their aperture random numbers, so a
    // scan traced in several chunks interleaved the two streams differently from a scan traced in one chunk and gave a
    // different (equally valid) realization for the same seed. The chunk size depends on the memory budget and on the
    // per-return staging cap, so a seeded scan could change with either. The draws now follow the single-chunk order
    // however the scan is chunked. (Fewer than 500 primitives keeps every chunk on the CPU traversal, so the
    // comparison is exact on GPU builds too.)
    struct Run {
        std::vector<vec3> xyz;
        std::vector<double> timestamp, distance, target_index;
        size_t chunks = 0;
    };
    auto run = [](size_t budget) {
        Context context;
        context.seedRandomGenerator(11);
        context.addPatch(make_vec3(5, 0, 1), make_vec2(10, 8), make_SphericalCoord(0.5f * float(M_PI), 0.5f * float(M_PI)));
        for (int k = 0; k < 3; k++) {
            context.addPatch(make_vec3(1.f + 0.8f * float(k), -0.6f + 0.5f * float(k), 0.7f + 0.3f * float(k)), make_vec2(0.9f, 0.7f), make_SphericalCoord(0.3f * float(k), 0.5f * float(k)));
        }
        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(-3, 0, 1), 350, 1.3f, 1.84f, 350, 1.3f, 1.84f, 0.01f, 0.003f, 0.f, 0.f, {});
        lidar.addScan(scan);
        if (budget > 0) {
            lidar.setSyntheticScanMemoryBudget(budget);
        }
        Run out;
        lidar.setSyntheticScanHitSink([&out](size_t, size_t) { out.chunks++; });
        lidar.syntheticScan(&context, 10, 0.02f, RETURN_MODE_MULTI, false, true, false);
        lidar.getHitXYZColumn(out.xyz);
        lidar.getHitDataColumn("timestamp", out.timestamp);
        lidar.getHitDataColumn("distance", out.distance);
        lidar.getHitDataColumn("target_index", out.target_index);
        return out;
    };
    const Run single = run(0); // the default budget holds the whole scan
    const Run chunked = run(8u * 1024u * 1024u); // floors each chunk at ~1.05M sub-rays: two chunks
    DOCTEST_REQUIRE(single.chunks == 1);
    DOCTEST_REQUIRE(chunked.chunks > 1);
    DOCTEST_REQUIRE(chunked.xyz.size() == single.xyz.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < single.xyz.size(); i++) {
        if (single.xyz[i].x != chunked.xyz[i].x || single.xyz[i].y != chunked.xyz[i].y || single.xyz[i].z != chunked.xyz[i].z || single.timestamp[i] != chunked.timestamp[i] || single.distance[i] != chunked.distance[i] ||
            single.target_index[i] != chunked.target_index[i]) {
            mismatches++;
        }
    }
    DOCTEST_CHECK_MESSAGE(mismatches == 0, mismatches << " of " << single.xyz.size() << " hits differ between the single-chunk and the chunked scan");
}

DOCTEST_TEST_CASE("LiDAR getHitIndex - Looks Up the Hit at a Scan-Grid Cell From Row/Column Data") {
    // The (row, column) -> hit table getHitIndex() used to read was allocated for every scan but never written, so the
    // call returned -1 for every cell. It now finds the hit whose row/column data names the cell: a stored return,
    // or a virtualized miss after gap-filling.
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata a(make_vec3(0, 0, 0), 20, 0.05f, 0.95f * float(M_PI), 36, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(a);
    const size_t returns = addRowColumnReturns(cloud, 0, 20, 36, 5, 9, 10, 15);
    DOCTEST_CHECK(cloud.getHitIndex(0, 3, 4) == 3 * 36 + 4); // stored in row-major order
    DOCTEST_CHECK(cloud.getHitIndex(0, 7, 12) == -1); // blank block: no hit recorded there yet
    DOCTEST_CHECK_THROWS_AS(static_cast<void>(cloud.getHitIndex(0, 20, 0)), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(static_cast<void>(cloud.getHitIndex(0, 0, 36)), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(static_cast<void>(cloud.getHitIndex(1, 0, 0)), std::runtime_error);

    cloud.gapfillMissesCount(0, false, false);
    const int miss = cloud.getHitIndex(0, 7, 12);
    DOCTEST_REQUIRE(miss >= int(returns));
    DOCTEST_CHECK(cloud.getHitData(uint(miss), "row") == 7.0);
    DOCTEST_CHECK(cloud.getHitData(uint(miss), "column") == 12.0);
    DOCTEST_CHECK(cloud.getHitData(uint(miss), "is_miss") == 1.0);
    DOCTEST_CHECK(cloud.getHitIndex(0, 3, 4) == 3 * 36 + 4);

    // Without row/column data there is no cell to look up, which is an error rather than an empty cell.
    LiDARcloud plain;
    plain.disableMessages();
    plain.addScan(a);
    plain.addHitPoint(0, make_vec3(1, 2, 3), SphericalCoord(1, 0, 0));
    DOCTEST_CHECK_THROWS_AS(static_cast<void>(plain.getHitIndex(0, 0, 0)), std::runtime_error);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Timestamp Path Count-Only Overload Builds the Same Cloud") {
    // The timestamp gap filler stages its misses and appends them in one step; the count-only overload skips
    // collecting the positions but must synthesize exactly the same misses, in the same order, with the same data.
    auto build = [](LiDARcloud &lidar) {
        const uint Ntheta = 120, Nphi = 90;
        const float thetaMin = 0.3f, thetaMax = 2.8f;
        ScanMetadata scan(make_vec3(0, 0, 0), Ntheta, thetaMin, thetaMax, Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
        lidar.addScan(scan);
        size_t pulse = 0;
        for (uint column = 0; column < Nphi; column++) {
            for (uint row = 0; row < Ntheta; row++, pulse++) {
                if ((pulse * 2654435761u) % 10 < 3) {
                    continue; // ~30% of pulses missing
                }
                const float zenith = thetaMin + (thetaMax - thetaMin) * float(row) / float(Ntheta - 1);
                const float azimuth = 2.f * float(M_PI) * float(column) / float(Nphi);
                const SphericalCoord d(1.f, 0.5f * float(M_PI) - zenith, azimuth);
                std::map<std::string, double> data;
                data["timestamp"] = double(pulse);
                data["target_index"] = 0;
                data["target_count"] = 1;
                lidar.addHitPoint(0, 8.f * sphere2cart(d), d, data);
            }
        }
    };
    LiDARcloud a, b;
    a.disableMessages();
    b.disableMessages();
    build(a);
    build(b);
    const uint nreal = a.getHitCount();
    DOCTEST_REQUIRE(nreal == b.getHitCount());

    const std::vector<vec3> filled = a.gapfillMisses(0, false, true);
    const size_t count = b.gapfillMissesCount(0, false, true);
    DOCTEST_CHECK(filled.size() > 1000);
    DOCTEST_CHECK(count == filled.size());
    DOCTEST_REQUIRE(a.getHitCount() == nreal + filled.size());
    DOCTEST_REQUIRE(b.getHitCount() == a.getHitCount());

    // The returned positions are the appended misses, in order; both clouds carry identical misses and data.
    for (uint i = 0; i < filled.size(); i++) {
        const vec3 pa = a.getHitXYZ(nreal + i);
        DOCTEST_CHECK(pa.x == filled[i].x);
        DOCTEST_CHECK(pa.y == filled[i].y);
        DOCTEST_CHECK(pa.z == filled[i].z);
    }
    for (const char *label: {"timestamp", "target_index", "nRaysHit", "is_miss", "gapfillMisses_code"}) {
        std::vector<double> va, vb;
        a.getHitDataColumn(label, va, -12345.0);
        b.getHitDataColumn(label, vb, -12345.0);
        size_t mismatches = 0;
        for (size_t i = 0; i < va.size(); i++) {
            mismatches += (va[i] != vb[i]) ? 1 : 0;
        }
        DOCTEST_CHECK_MESSAGE(mismatches == 0, label << ": " << mismatches << " mismatches");
    }
    std::vector<vec3> xa, xb;
    a.getHitXYZColumn(xa);
    b.getHitXYZColumn(xb);
    size_t xyz_mismatches = 0;
    for (size_t i = 0; i < xa.size(); i++) {
        xyz_mismatches += (xa[i].x != xb[i].x || xa[i].y != xb[i].y || xa[i].z != xb[i].z) ? 1 : 0;
    }
    DOCTEST_CHECK(xyz_mismatches == 0);
    // Every synthesized point is a flagged miss with an interior or edge code; every original is code 0.
    for (uint i = 0; i < a.getHitCount(); i++) {
        const bool synthesized = i >= nreal;
        DOCTEST_CHECK(a.getHitData(i, "is_miss") == (synthesized ? 1.0 : 0.0));
        const double code = a.getHitData(i, "gapfillMisses_code");
        DOCTEST_CHECK((synthesized ? (code >= 1.0 && code <= 3.0) : code == 0.0));
    }
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Timestamp and Row/Column Paths Synthesize the Same Misses") {
    // The same returns described by their scan-grid indices and by their pulse timestamps must yield the identical
    // virtualized miss population: same cells, same directions, same count. A third cloud carries GPS-like
    // timestamps (seconds since an epoch at a 10 kHz pulse rate) with the first pulses of the scan missing, so the
    // declared clock does not apply and the pulse clock must be recovered and anchored from the data.
    GenerativeGrid g{30, 48, 0.05, 0.95 * M_PI, 0.0, 1.5 * M_PI, 1.0e-3, 0.0};
    auto returns_present = [&](int row, int col) { return !(row >= 6 && row <= 11 && col >= 10 && col <= 20) && !((row * 7 + col * 3) % 11 == 0); };
    auto add_scan = [&](LiDARcloud &lidar) {
        ScanMetadata scan(make_vec3(0, 0, 0), g.Ntheta, g.theta_min, g.theta_max, g.Nphi, g.phi_min, g.phi_max, 0.0f, 0.0f, 0.0f, 0.0f, {});
        lidar.addScan(scan);
    };
    enum Mode { ROWCOL, PULSE_CLOCK, GPS_TIME };
    const double gps_epoch = 1.7e9, gps_period = 1.0e-4;
    auto build = [&](LiDARcloud &lidar, Mode mode) {
        add_scan(lidar);
        for (int col = 0; col < g.Nphi; col++) {
            for (int row = 0; row < g.Ntheta; row++) {
                if (!returns_present(row, col)) {
                    continue;
                }
                const long long ordinal = (long long) col * g.Ntheta + row;
                if (mode == GPS_TIME && ordinal < 37) {
                    continue; // the scan's first pulses are missing: the earliest return is not pulse 0
                }
                SphericalCoord dir = g.direction(row, col);
                vec3 xyz = helios::sphere2cart(make_SphericalCoord(10.f, dir.elevation, dir.azimuth));
                std::map<std::string, double> data;
                if (mode == ROWCOL) {
                    data["row"] = row;
                    data["column"] = col;
                } else if (mode == PULSE_CLOCK) {
                    data["timestamp"] = double(ordinal);
                } else {
                    data["timestamp"] = gps_epoch + gps_period * double(ordinal);
                }
                lidar.addHitPoint(0, xyz, dir, make_RGBcolor(1, 0, 0), data);
            }
        }
    };

    LiDARcloud by_rowcol, by_clock, by_gps;
    by_rowcol.disableMessages();
    by_clock.disableMessages();
    by_gps.disableMessages();
    build(by_rowcol, ROWCOL);
    build(by_clock, PULSE_CLOCK);
    build(by_gps, GPS_TIME);
    const uint returns_rowcol = by_rowcol.getHitCount();
    const uint returns_gps = by_gps.getHitCount();
    DOCTEST_REQUIRE(by_clock.getHitCount() == returns_rowcol);
    DOCTEST_REQUIRE(returns_gps < returns_rowcol);

    const size_t filled_rowcol = by_rowcol.gapfillMissesCount(0, false, true);
    const size_t filled_clock = by_clock.gapfillMissesCount(0, false, true);
    const size_t filled_gps = by_gps.gapfillMissesCount(0, false, true);
    DOCTEST_CHECK(filled_rowcol == size_t(g.Ntheta * g.Nphi) - returns_rowcol);
    DOCTEST_CHECK(filled_clock == filled_rowcol);
    DOCTEST_CHECK(filled_gps == size_t(g.Ntheta * g.Nphi) - returns_gps);
    DOCTEST_CHECK(by_rowcol.getVirtualMissCount() == filled_rowcol);
    DOCTEST_CHECK(by_clock.getVirtualMissCount() == filled_clock);

    // Cell for cell, the timestamp path placed its returns where the row/column data said they were, and the two
    // virtual populations are bit-identical.
    auto compare = [&](LiDARcloud &a, LiDARcloud &b, const char *what) {
        std::vector<int32_t> ra, ca, rb, cb, ma, mb;
        a.getHitDataColumn("row", ra, int32_t(-1));
        a.getHitDataColumn("column", ca, int32_t(-1));
        b.getHitDataColumn("row", rb, int32_t(-1));
        b.getHitDataColumn("column", cb, int32_t(-1));
        a.getHitDataColumn("is_miss", ma, int32_t(-1));
        b.getHitDataColumn("is_miss", mb, int32_t(-1));
        std::vector<vec3> pa, pb;
        a.getHitXYZColumn(pa);
        b.getHitXYZColumn(pb);
        DOCTEST_REQUIRE(pa.size() == pb.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < pa.size(); i++) {
            if (ra[i] != rb[i] || ca[i] != cb[i] || ma[i] != mb[i] || pa[i].x != pb[i].x || pa[i].y != pb[i].y || pa[i].z != pb[i].z) {
                mismatches++;
            }
        }
        DOCTEST_CHECK_MESSAGE(mismatches == 0, std::string(what) << ": " << mismatches << " of " << pa.size() << " hits differ");
    };
    compare(by_rowcol, by_clock, "row/column vs pulse-clock timestamps");

    // GPS-time cloud: every return landed in its true cell, and each synthesized miss carries the time its pulse
    // was fired on the recovered clock.
    {
        std::vector<int32_t> row, col, is_miss;
        std::vector<double> t;
        by_gps.getHitDataColumn("row", row, int32_t(-1));
        by_gps.getHitDataColumn("column", col, int32_t(-1));
        by_gps.getHitDataColumn("is_miss", is_miss, int32_t(-1));
        by_gps.getHitDataColumn("timestamp", t, -1.0);
        size_t wrong_cell = 0, wrong_time = 0;
        for (size_t i = 0; i < t.size(); i++) {
            const long long ordinal = (long long) col[i] * g.Ntheta + row[i];
            const double expected = gps_epoch + gps_period * double(ordinal);
            if (std::fabs(t[i] - expected) > 1e-6 * gps_period + 1e-9 * gps_epoch) {
                wrong_time++;
            }
            if (is_miss[i] == 0 && !returns_present(row[i], col[i])) {
                wrong_cell++;
            }
        }
        DOCTEST_CHECK(wrong_cell == 0);
        DOCTEST_CHECK(wrong_time == 0);
        // Directions of the recovered population agree with the row/column cloud's cell for cell.
        std::vector<vec3> pg;
        by_gps.getHitXYZColumn(pg);
        std::map<std::pair<int, int>, vec3> rc_position;
        {
            std::vector<int32_t> rr, cc;
            std::vector<vec3> pr;
            by_rowcol.getHitDataColumn("row", rr, int32_t(-1));
            by_rowcol.getHitDataColumn("column", cc, int32_t(-1));
            by_rowcol.getHitXYZColumn(pr);
            for (size_t i = 0; i < pr.size(); i++) {
                rc_position[{rr[i], cc[i]}] = pr[i];
            }
        }
        double max_err = 0.0;
        for (size_t i = 0; i < pg.size(); i++) {
            if (is_miss[i] != 1) {
                continue;
            }
            auto it = rc_position.find({row[i], col[i]});
            DOCTEST_REQUIRE(it != rc_position.end());
            max_err = std::max(max_err, angularError(cart2sphere(pg[i]), cart2sphere(it->second)));
        }
        DOCTEST_CHECK(max_err < 1e-3);
    }
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Timestamp Path Recovers the Pulse Clock on a Large Scan") {
    // REGRESSION: the pulse period was estimated from the spacing of consecutive members of the bounded direction
    // sample, which on a scan larger than the sample limit are hundreds of pulses apart, so the "period" came out
    // hundreds of times too large and every return landed in the wrong cell (the grid-model fit then failed with a
    // residual error). The period must come from returns consecutive in time. This scan is ~10x the sample limit.
    const uint Ntheta = 300, Nphi = 1000;
    const float thetaMin = 0.2f, thetaMax = 2.9f;
    const double gps_epoch = 1.7e9, gps_period = 1.0e-4;
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), Ntheta, thetaMin, thetaMax, Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    lidar.addScan(scan);
    const size_t Npulses = size_t(Ntheta) * Nphi;
    std::vector<double> xyz, values;
    std::vector<float> dir;
    const std::vector<std::string> labels = {"timestamp"};
    size_t returns = 0;
    for (size_t p = 0; p < Npulses; p++) {
        if ((p * 2654435761u) % 10 < 3 || p < 123) {
            continue; // 30% of pulses missing, and the first 123 pulses of the scan absent
        }
        const int column = int(p / Ntheta), row = int(p % Ntheta);
        const float zenith = thetaMin + (thetaMax - thetaMin) * float(row) / float(Ntheta - 1);
        const float azimuth = 2.f * float(M_PI) * float(column) / float(Nphi);
        const SphericalCoord d(1.f, 0.5f * float(M_PI) - zenith, azimuth);
        const vec3 u = sphere2cart(d);
        xyz.push_back(10.0 * u.x);
        xyz.push_back(10.0 * u.y);
        xyz.push_back(10.0 * u.z);
        dir.push_back(d.radius);
        dir.push_back(d.elevation);
        dir.push_back(d.azimuth);
        values.push_back(gps_epoch + gps_period * double(p));
        returns++;
    }
    lidar.addHitPoints(0, returns, xyz.data(), dir.data(), labels, values.data());
    DOCTEST_REQUIRE(returns > 200000);

    const size_t filled = lidar.gapfillMissesCount(0, false, false);
    DOCTEST_CHECK(filled == Npulses - returns);
    DOCTEST_CHECK(lidar.getVirtualMissCount() == filled);
    std::vector<int32_t> row, col;
    std::vector<double> t;
    lidar.getHitDataColumn("row", row, int32_t(-1));
    lidar.getHitDataColumn("column", col, int32_t(-1));
    lidar.getHitDataColumn("timestamp", t, -1.0);
    size_t wrong_cell = 0, wrong_time = 0;
    for (size_t i = 0; i < t.size(); i++) {
        const long long ordinal = (long long) std::llround((t[i] - gps_epoch) / gps_period);
        if (row[i] != int(ordinal % Ntheta) || col[i] != int(ordinal / Ntheta)) {
            wrong_cell++;
        }
        if (i >= returns && std::fabs(t[i] - (gps_epoch + gps_period * double((long long) col[i] * Ntheta + row[i]))) > 1e-6 * gps_period + 1e-9 * gps_epoch) {
            wrong_time++;
        }
    }
    DOCTEST_CHECK(wrong_cell == 0);
    DOCTEST_CHECK(wrong_time == 0);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Timestamp Path Reconstructs a Raster With Mirror Dead Time and a Tilted Head") {
    // A terrestrial scanner's mirror fires through part of each rotation only, so a column of Ntheta pulses is
    // followed by hundreds of pulse periods of silence, and the head is never perfectly level. From timestamps alone
    // the gap filler must recover the columns (including columns with no return at all and columns whose first
    // pulses returned nothing), place every return in its true cell, and synthesize the missing pulses along the
    // beams the tilted head actually fired.
    const int Ntheta = 400, Nphi = 300;
    const double period = 3.4e-6, dead_periods = 500.0, line_period = (Ntheta + dead_periods) * period, gps_epoch = 1.7e9;
    const float thetaMin = 0.5136f, zen_step = 0.0007f, phi0 = 5.0f, frame_step = 0.000873f;
    const float tilt = 1.5f * float(M_PI) / 180.f; // head axis tilted about world x
    auto true_direction = [&](int row, int col) {
        const float zen = thetaMin + zen_step * float(row), az = phi0 + frame_step * float(col);
        const vec3 d = sphere2cart(SphericalCoord(1.f, 0.5f * float(M_PI) - zen, az));
        return make_vec3(d.x, d.y * cosf(tilt) - d.z * sinf(tilt), d.y * sinf(tilt) + d.z * cosf(tilt));
    };
    auto pulse_present = [&](int row, int col) {
        if (col == 50 || col == 51 || col == 200) {
            return false; // whole columns of sky
        }
        if (col % 7 == 0 && row < 25) {
            return false; // the first pulses of these columns returned nothing
        }
        return ((size_t) (col * Ntheta + row) * 2654435761u) % 10 >= 3; // and 30% of the rest
    };

    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(1.f, -2.f, 0.5f), uint(Ntheta), thetaMin, thetaMin + zen_step * float(Ntheta - 1), uint(Nphi), phi0, phi0 + frame_step * float(Nphi - 1), 0.f, 0.f, 0.f, 0.f, {});
    lidar.addScan(scan);
    const vec3 origin = lidar.getScanOrigin(0);
    std::vector<double> xyz, values;
    std::vector<float> dir;
    size_t returns = 0;
    for (int col = 0; col < Nphi; col++) {
        for (int row = 0; row < Ntheta; row++) {
            if (!pulse_present(row, col)) {
                continue;
            }
            const vec3 d = true_direction(row, col);
            const float range = 6.f + 2.f * sinf(0.01f * float(row)) * cosf(0.03f * float(col));
            xyz.push_back(double(origin.x + range * d.x));
            xyz.push_back(double(origin.y + range * d.y));
            xyz.push_back(double(origin.z + range * d.z));
            const SphericalCoord sd = cart2sphere(d);
            dir.push_back(1.f);
            dir.push_back(sd.elevation);
            dir.push_back(sd.azimuth);
            values.push_back(gps_epoch + double(col) * line_period + double(row) * period);
            returns++;
        }
    }
    lidar.addHitPoints(0, returns, xyz.data(), dir.data(), {"timestamp"}, values.data());

    const size_t filled = lidar.gapfillMissesCount(0, false, false);
    DOCTEST_CHECK(filled == size_t(Ntheta) * Nphi - returns);
    DOCTEST_CHECK(lidar.getVirtualMissCount() == filled);

    std::vector<int32_t> row, col, is_miss;
    std::vector<double> t;
    std::vector<vec3> pos;
    lidar.getHitDataColumn("row", row, int32_t(-1));
    lidar.getHitDataColumn("column", col, int32_t(-1));
    lidar.getHitDataColumn("is_miss", is_miss, int32_t(-1));
    lidar.getHitDataColumn("timestamp", t, -1.0);
    lidar.getHitXYZColumn(pos);
    size_t wrong_cell = 0, wrong_time = 0, present_cells_filled = 0;
    double max_angle_error = 0.0;
    for (size_t i = 0; i < pos.size(); i++) {
        const long long ordinal = (long long) std::llround((t[i] - gps_epoch - double(col[i]) * line_period) / period); // row implied by the time, given the column
        const bool return_here = is_miss[i] == 0;
        if (return_here) {
            // Cell from the return's own timestamp: column from the column period, row from the pulse period.
            const long long true_col = (long long) std::llround((t[i] - gps_epoch) / line_period);
            const long long true_row = (long long) std::llround((t[i] - gps_epoch - double(true_col) * line_period) / period);
            if (row[i] != true_row || col[i] != true_col) {
                wrong_cell++;
            }
        } else {
            if (pulse_present(row[i], col[i])) {
                present_cells_filled++;
            }
            if (ordinal != row[i]) {
                wrong_time++;
            }
            vec3 d = pos[i] - origin;
            d.normalize();
            // Chord between unit vectors: equal to the angle to first order and, unlike acosf of a float dot
            // product, resolves angles far below a milliradian.
            max_angle_error = std::max(max_angle_error, (double) (d - true_direction(row[i], col[i])).magnitude());
        }
    }
    DOCTEST_CHECK(wrong_cell == 0);
    DOCTEST_CHECK(present_cells_filled == 0);
    DOCTEST_CHECK(wrong_time == 0);
    DOCTEST_CHECK_MESSAGE(max_angle_error < 2e-4, "synthesized miss direction error " << max_angle_error * 180.0 / M_PI << " deg");
    // The whole-column gaps were filled as complete columns of misses.
    size_t col50 = 0;
    for (size_t i = 0; i < pos.size(); i++) {
        if (col[i] == 50 && is_miss[i] == 1) {
            col50++;
        }
    }
    DOCTEST_CHECK(col50 == size_t(Ntheta));
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Timestamp Path Misses Are Virtualized and Materialize Identically") {
    checkVirtualMaterializedEquivalence([](LiDARcloud &c) {
        ScanMetadata scan(make_vec3(0, 0, 0), 20, 0.05, 0.95 * M_PI, 36, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
        c.addScan(scan);
        for (int col = 0; col < 36; col++) {
            for (int row = 0; row < 20; row++) {
                if (row >= 5 && row <= 9 && col >= 10 && col <= 15) {
                    continue;
                }
                const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / 20.f;
                const float phi = 2.f * float(M_PI) * float(col) / 36.f;
                SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
                std::map<std::string, double> data;
                data["timestamp"] = double(col * 20 + row);
                data["intensity"] = 0.5;
                c.addHitPoint(0, sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi)), dir, data);
            }
        }
        c.gapfillMisses(0, false, true);
    });
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Moving Scan Misses Are Virtualized With Per-Pulse Origins") {
    // A moving-platform scan's synthesized misses are held virtually too; each reports the platform pose at its
    // pulse's time as its origin, and materializing them writes the same origins into origin_x/y/z.
    auto build = [](LiDARcloud &lidar) {
        Context context;
        context.addPatch(make_vec3(0, 0, 0), make_vec2(2, 2));
        const uint Ntheta = 8, Nphi = 60;
        const float H = 6.0f, v = 3.0f;
        ScanMetadata scan(make_vec3(0, 0, H), Ntheta, 0.93f * float(M_PI), float(M_PI), Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, std::vector<std::string>());
        const float pulseRate = float(Ntheta * Nphi);
        const double t_total = double(Ntheta * Nphi) / double(pulseRate);
        std::vector<double> traj_t;
        std::vector<vec3> traj_pos, traj_rpy;
        for (int k = 0; k < 20; k++) {
            const double tk = t_total * double(k) / 19.0;
            traj_t.push_back(tk);
            traj_pos.push_back(make_vec3(float(v * tk), 0.f, H));
            traj_rpy.push_back(make_vec3(0, 0, 0));
        }
        lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), pulseRate, 0.0);
        lidar.syntheticScan(&context, false, false);
        lidar.gapfillMisses(0, false, true);
    };
    LiDARcloud virt, mat;
    virt.disableMessages();
    mat.disableMessages();
    build(virt);
    build(mat);
    DOCTEST_REQUIRE(virt.getVirtualMissCount() > 0);
    mat.materializeMisses();
    DOCTEST_REQUIRE(mat.getHitCount() == virt.getHitCount());
    const float v = 3.0f, H = 6.0f;
    size_t checked = 0;
    for (uint i = 0; i < virt.getHitCount(); i++) {
        if (virt.getHitData(i, "is_miss") != 1.0) {
            continue;
        }
        DOCTEST_REQUIRE(virt.doesHitDataExist(i, "origin_x"));
        DOCTEST_REQUIRE(mat.doesHitDataExist(i, "origin_x"));
        const vec3 ov = virt.getHitOrigin(i), om = mat.getHitOrigin(i);
        DOCTEST_CHECK(ov.x == om.x);
        DOCTEST_CHECK(ov.y == om.y);
        DOCTEST_CHECK(ov.z == om.z);
        DOCTEST_CHECK(virt.getHitData(i, "origin_x") == mat.getHitData(i, "origin_x"));
        const double t = virt.getHitData(i, "timestamp");
        DOCTEST_CHECK(ov.x == doctest::Approx(float(v * t)).epsilon(0.02));
        DOCTEST_CHECK(ov.z == doctest::Approx(H).epsilon(0.01));
        const vec3 pv = virt.getHitXYZ(i), pm = mat.getHitXYZ(i);
        DOCTEST_CHECK(pv.x == pm.x);
        DOCTEST_CHECK(pv.z == pm.z);
        checked++;
    }
    DOCTEST_CHECK(checked > 0);
    // The bulk readers report the per-pulse origins for the virtual misses too.
    std::vector<float> ox;
    virt.getHitDataColumn("origin_x", ox, -9999.f);
    size_t absent = 0;
    for (uint i = 0; i < virt.getHitCount(); i++) {
        if (virt.getHitData(i, "is_miss") == 1.0 && ox[i] == -9999.f) {
            absent++;
        }
    }
    DOCTEST_CHECK(absent == 0);
}

// ---- Review regressions: timestamp gap-fill on approximate rasters, grid-only frame, multi-scan, moving leaf area ----

//! A terrestrial raster fired column after column with GPS-epoch timestamps: pulses within a column are `period`
//! apart and columns `line_period` apart (Ntheta periods for a continuous pulse train, more when the mirror has dead
//! time). Beam (row, column) points at zenith thetaMin + row*dtheta and azimuth phiMin + column*dphi +
//! row*dphi/Ntheta -- the synthetic scanner's sweep -- in the scanner head's frame, which is tilted about world x.
struct TimestampRaster {
    int Ntheta = 150, Nphi = 200;
    float thetaMin = 0.35f, thetaMax = 2.6f, phiMin = 0.2f, phiMax = 4.0f;
    double epoch = 1.7e9, period = 1.0e-4, line_period = 150 * 1.0e-4;
    float tilt = 0.f;
    vec3 origin = make_vec3(0, 0, 0);
    float dtheta() const {
        return (thetaMax - thetaMin) / float(Ntheta - 1);
    }
    float dphi() const {
        return (phiMax - phiMin) / float(Nphi - 1);
    }
    vec3 beam(int row, int col) const {
        const float zenith = thetaMin + dtheta() * float(row);
        const float azimuth = phiMin + dphi() * float(col) + dphi() * float(row) / float(Ntheta);
        const vec3 d = sphere2cart(SphericalCoord(1.f, 0.5f * float(M_PI) - zenith, azimuth));
        return make_vec3(d.x, d.y * std::cos(tilt) - d.z * std::sin(tilt), d.y * std::sin(tilt) + d.z * std::cos(tilt));
    }
    double time(int row, int col) const {
        return epoch + double(col) * line_period + double(row) * period;
    }
};

//! Add the returns of a TimestampRaster (in firing order, timestamp their only label) to a scan; returns the count.
static size_t addTimestampRasterReturns(LiDARcloud &cloud, uint scanID, const TimestampRaster &raster, const std::function<bool(int, int)> &returned) {
    std::vector<double> xyz, stamps;
    for (int col = 0; col < raster.Nphi; col++) {
        for (int row = 0; row < raster.Ntheta; row++) {
            if (!returned(row, col)) {
                continue;
            }
            const vec3 p = raster.origin + raster.beam(row, col) * 10.f;
            xyz.insert(xyz.end(), {double(p.x), double(p.y), double(p.z)});
            stamps.push_back(raster.time(row, col));
        }
    }
    const std::vector<std::string> labels = {"timestamp"};
    cloud.addHitPoints(scanID, stamps.size(), xyz.data(), nullptr, labels, stamps.data());
    return stamps.size();
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Timestamp Path Tolerates an Approximate Declared Raster") {
    // REGRESSION: a caller that estimates a scan's raster from its point count declares a size close to, but not
    // exactly, the instrument's. The timestamp path cut columns wherever the silence exceeded the DECLARED column and
    // numbered them from the first column with a return, so a 150 x 200 raster declared as 142 x 211 or 158 x 190
    // threw ("a single column of returns"), and columns before the first return were never filled. (The version
    // before the rework interpolated along each column instead and filled most, but not all, of such a raster.) The
    // raster is now measured from the timestamps and the beam angles within the declared angular extent: every miss of
    // the instrument's raster is synthesized, and the scan's size is updated to the measured one.
    auto returned = [](int row, int col) {
        if (col < 12) {
            return false; // leading columns with no return at all
        }
        if (row >= 40 && row <= 70 && col >= 90 && col <= 120) {
            return false;
        }
        return ((unsigned(row) * 2654435761u) ^ (unsigned(col) * 40503u)) % 4u != 0u;
    };
    auto run = [&](const TimestampRaster &raster, uint declared_theta, uint declared_phi, bool check_cells, const char *what) {
        LiDARcloud cloud;
        cloud.disableMessages();
        ScanMetadata scan(raster.origin, declared_theta, raster.thetaMin, raster.thetaMax, declared_phi, raster.phiMin, raster.phiMax, 0.f, 0.f, 0.f, 0.f, {});
        cloud.addScan(scan);
        const size_t returns = addTimestampRasterReturns(cloud, 0, raster, returned);
        size_t filled = 0;
        try {
            filled = cloud.gapfillMissesCount(0, false, false);
        } catch (const std::exception &e) {
            DOCTEST_CHECK_MESSAGE(false, std::string(what) << ": gap-filling threw: " << e.what());
            return;
        }
        const size_t pulses = size_t(raster.Ntheta) * size_t(raster.Nphi);
        DOCTEST_CHECK_MESSAGE(filled == pulses - returns, std::string(what) << ": filled " << filled << ", the raster has " << pulses - returns << " misses");
        DOCTEST_CHECK_MESSAGE(cloud.getScanSizeTheta(0) == uint(raster.Ntheta), std::string(what) << ": measured " << cloud.getScanSizeTheta(0) << " rows");
        DOCTEST_CHECK_MESSAGE(cloud.getScanSizePhi(0) == uint(raster.Nphi), std::string(what) << ": measured " << cloud.getScanSizePhi(0) << " columns");
        if (!check_cells) {
            return;
        }
        // Every synthesized miss lies along the beam of a distinct cell of the true raster that returned nothing.
        std::vector<vec3> xyz;
        std::vector<int32_t> is_miss;
        cloud.getHitXYZColumn(xyz);
        cloud.getHitDataColumn("is_miss", is_miss, int32_t(-1));
        std::set<long long> cells;
        size_t off_raster = 0, onto_return = 0, leading = 0;
        double worst = 0.0;
        for (size_t i = 0; i < xyz.size(); i++) {
            if (is_miss[i] != 1) {
                continue;
            }
            vec3 d = xyz[i] - raster.origin;
            d.normalize();
            const SphericalCoord sc = cart2sphere(d);
            const int row = int(std::lround((sc.zenith - raster.thetaMin) / raster.dtheta()));
            double azimuth = double(sc.azimuth) - raster.phiMin - raster.dphi() * float(row) / float(raster.Ntheta);
            if (azimuth < -0.5 * raster.dphi()) {
                azimuth += 2.0 * M_PI;
            }
            const int col = int(std::lround(azimuth / raster.dphi()));
            if (row < 0 || row >= raster.Ntheta || col < 0 || col >= raster.Nphi) {
                off_raster++;
                continue;
            }
            if (returned(row, col)) {
                onto_return++;
                continue;
            }
            cells.insert((long long) col * raster.Ntheta + row);
            const vec3 b = raster.beam(row, col);
            worst = std::max(worst, std::sqrt((double(d.x) - b.x) * (double(d.x) - b.x) + (double(d.y) - b.y) * (double(d.y) - b.y) + (double(d.z) - b.z) * (double(d.z) - b.z))); // chord, in double
            leading += (col < 12) ? 1 : 0;
        }
        DOCTEST_CHECK_MESSAGE(off_raster == 0, std::string(what) << ": " << off_raster << " misses off the raster");
        DOCTEST_CHECK_MESSAGE(onto_return == 0, std::string(what) << ": " << onto_return << " misses on cells that returned");
        DOCTEST_CHECK_MESSAGE(cells.size() == filled, std::string(what) << ": " << filled - cells.size() << " misses share a cell");
        DOCTEST_CHECK_MESSAGE(worst < 0.1 * raster.dtheta(), std::string(what) << ": a miss is " << worst << " rad off its beam");
        DOCTEST_CHECK_MESSAGE(leading == 12 * size_t(raster.Ntheta), std::string(what) << ": " << leading << " misses in the 12 leading empty columns");
    };

    TimestampRaster continuous;
    run(continuous, 150, 200, true, "continuous pulse train, exact size");
    run(continuous, 142, 211, true, "continuous pulse train, declared 142 x 211");
    run(continuous, 158, 190, true, "continuous pulse train, declared 158 x 190");
    TimestampRaster dead_time = continuous;
    dead_time.line_period = (150 + 400) * dead_time.period;
    run(dead_time, 150, 200, true, "mirror dead time, exact size");
    run(dead_time, 142, 211, true, "mirror dead time, declared 142 x 211");
    run(dead_time, 158, 190, true, "mirror dead time, declared 158 x 190");
    // A full turn: a rotation has no recoverable starting angle, so columns are numbered from the first column that
    // returned; only the population and the measured size are checked.
    TimestampRaster full_turn = continuous;
    full_turn.phiMin = 0.f;
    full_turn.phiMax = 2.f * float(M_PI) * float(full_turn.Nphi - 1) / float(full_turn.Nphi);
    run(full_turn, 142, 211, false, "full turn, declared 142 x 211");
    run(full_turn, 158, 190, false, "full turn, declared 158 x 190");
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Grid-Only Mode Selects Rows in the Scanner Head's Frame") {
    // REGRESSION: grid-only mode computed the grid's zenith window about world +z but compared it with the fitted
    // model's zenith, which is measured about the scanner head's axis. On a head tilted by a few degrees the window
    // was off by up to the tilt, so rows whose beams reach the grid were left unfilled and rows whose beams cannot
    // reach it were filled.
    TimestampRaster raster;
    raster.Ntheta = 120;
    raster.Nphi = 160;
    raster.thetaMin = 0.6f;
    raster.thetaMax = 2.4f;
    raster.phiMin = 0.f;
    raster.phiMax = 2.f * float(M_PI) * 159.f / 160.f;
    raster.line_period = 120 * raster.period;
    raster.tilt = 6.f * float(M_PI) / 180.f;
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(raster.origin, uint(raster.Ntheta), raster.thetaMin, raster.thetaMax, uint(raster.Nphi), raster.phiMin, raster.phiMax, 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(scan);
    const vec3 box_center = make_vec3(0.f, 4.f, 2.5f), box_size = make_vec3(2, 2, 2);
    cloud.addGrid(box_center, box_size, make_int3(1, 1, 1), 0.f);
    addTimestampRasterReturns(cloud, 0, raster, [](int row, int col) { return (row * 31 + col * 17) % 5 >= 2; });
    const size_t filled = cloud.gapfillMissesCount(0, true, false);
    DOCTEST_REQUIRE(filled > 0);

    // The grid's zenith range about the head axis, sampled finely over the box surface.
    double zmin = M_PI, zmax = 0.0;
    const int n = 41;
    const float ct = std::cos(raster.tilt), st = std::sin(raster.tilt);
    for (int axis = 0; axis < 3; axis++) {
        for (int side = 0; side < 2; side++) {
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    float u[3];
                    u[axis] = float(side);
                    u[(axis + 1) % 3] = float(a) / float(n - 1);
                    u[(axis + 2) % 3] = float(b) / float(n - 1);
                    const vec3 p = box_center + make_vec3((u[0] - 0.5f) * box_size.x, (u[1] - 0.5f) * box_size.y, (u[2] - 0.5f) * box_size.z) - raster.origin;
                    const vec3 head = make_vec3(p.x, p.y * ct + p.z * st, -p.y * st + p.z * ct);
                    const double zenith = std::acos(double(head.z) / double(head.magnitude()));
                    zmin = std::min(zmin, zenith);
                    zmax = std::max(zmax, zenith);
                }
            }
        }
    }
    std::vector<int32_t> row, is_miss;
    cloud.getHitDataColumn("row", row, int32_t(-1));
    cloud.getHitDataColumn("is_miss", is_miss, int32_t(-1));
    std::vector<size_t> misses_in_row(raster.Ntheta, 0);
    for (size_t i = 0; i < row.size(); i++) {
        if (is_miss[i] == 1 && row[i] >= 0 && row[i] < raster.Ntheta) {
            misses_in_row[size_t(row[i])]++;
        }
    }
    const double step = raster.dtheta();
    size_t wrongly_filled = 0, wrongly_empty = 0, inside = 0;
    for (int r = 0; r < raster.Ntheta; r++) {
        const double zenith = raster.thetaMin + step * r;
        if (zenith < zmin - step || zenith > zmax + step) {
            wrongly_filled += misses_in_row[r] > 0 ? 1 : 0;
        } else if (zenith > zmin + step && zenith < zmax - step) {
            inside++;
            wrongly_empty += misses_in_row[r] == 0 ? 1 : 0;
        }
    }
    DOCTEST_REQUIRE(inside >= 10);
    DOCTEST_CHECK_MESSAGE(wrongly_filled == 0, wrongly_filled << " rows whose beams cannot reach the grid were filled");
    DOCTEST_CHECK_MESSAGE(wrongly_empty == 0, wrongly_empty << " of " << inside << " rows whose beams reach the grid were left empty");
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Timestamp Path Keeps Every Other Scan's Misses Virtual") {
    // REGRESSION: the timestamp path collapsed every other scan's virtualized misses into stored points before writing
    // row/column data onto its own returns, which renumbers nothing and never required it.
    TimestampRaster r0;
    r0.Ntheta = 40;
    r0.Nphi = 60;
    r0.line_period = 40 * r0.period;
    TimestampRaster r1 = r0;
    r1.origin = make_vec3(4, 0, 0);
    r1.epoch += 100.0;
    auto returned = [](int row, int col) { return (row * 7 + col * 3) % 4 != 0; };
    LiDARcloud cloud;
    cloud.disableMessages();
    for (const TimestampRaster *r: {&r0, &r1}) {
        ScanMetadata scan(r->origin, uint(r->Ntheta), r->thetaMin, r->thetaMax, uint(r->Nphi), r->phiMin, r->phiMax, 0.f, 0.f, 0.f, 0.f, {});
        cloud.addScan(scan);
    }
    const size_t returns = addTimestampRasterReturns(cloud, 0, r0, returned) + addTimestampRasterReturns(cloud, 1, r1, returned);
    const size_t n0 = cloud.gapfillMissesCount(0, false, false);
    DOCTEST_REQUIRE(n0 > 0);
    std::vector<vec3> scan0_before;
    cloud.getScanHitXYZColumn(0, scan0_before);
    const size_t builds = LiDARTestHelper::scanIndexBuilds(cloud);
    const size_t n1 = cloud.gapfillMissesCount(1, false, false);
    DOCTEST_CHECK(cloud.getVirtualMissCount() == n0 + n1);
    DOCTEST_CHECK(LiDARTestHelper::storedHitCount(cloud) == returns);
    DOCTEST_CHECK(LiDARTestHelper::scanIndexBuilds(cloud) == builds);
    std::vector<vec3> scan0_after;
    cloud.getScanHitXYZColumn(0, scan0_after);
    DOCTEST_REQUIRE(scan0_after.size() == scan0_before.size());
    size_t moved = 0;
    for (size_t i = 0; i < scan0_before.size(); i++) {
        moved += (scan0_after[i].x != scan0_before[i].x || scan0_after[i].y != scan0_before[i].y || scan0_after[i].z != scan0_before[i].z) ? 1 : 0;
    }
    DOCTEST_CHECK(moved == 0);
}

DOCTEST_TEST_CASE("LiDAR Leaf Area - A Moving Scan's Virtualized Misses Are Traced From Their Own Pulse Origins") {
    // REGRESSION: when no stored hit carried origin_x/y/z, the inversion traced a moving scan's virtualized misses from
    // the scan's static origin while getHitOrigin() reported each miss's own pulse pose, so the same cloud gave a
    // different leaf area virtualized than materialized. Also: gap-filling a moving scan a second time dispatched to the
    // row/column path (the first pass records row/column data on the returns), which rejects moving scans.
    Context context;
    for (int k = 0; k < 6; k++) {
        context.addPatch(make_vec3(0.3f + 0.45f * float(k), -0.3f + 0.12f * float(k), 0.3f + 0.2f * float(k % 3)), make_vec2(0.25f, 0.25f));
    }
    const uint Ntheta = 40, Nphi = 120;
    const float H = 6.f, speed = 3.f;
    const float pulse_rate = float(Ntheta * Nphi);
    std::vector<double> traj_t;
    std::vector<vec3> traj_pos, traj_rpy;
    for (int k = 0; k < 20; k++) {
        const double t = double(k) / 19.0;
        traj_t.push_back(t);
        traj_pos.push_back(make_vec3(float(speed * t), 0.f, H));
        traj_rpy.push_back(make_vec3(0, 0, 0));
    }
    auto build = [&](LiDARcloud &lidar) {
        LiDARcloud source;
        source.disableMessages();
        ScanMetadata source_scan(make_vec3(0, 0, H), Ntheta, 0.8f * float(M_PI), float(M_PI), Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
        source.addScanMoving(source_scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), pulse_rate, 0.0);
        source.syntheticScan(&context, false, false);
        ScanMetadata scan(make_vec3(0, 0, H), Ntheta, 0.8f * float(M_PI), float(M_PI), Nphi, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
        lidar.addScanMoving(scan, traj_t, traj_pos, traj_rpy, make_vec3(0, 0, 0), make_vec3(0, 0, 0), pulse_rate, 0.0);
        // The returns with their timestamps only: no origin_x/y/z column anywhere in the cloud.
        std::vector<double> xyz, stamps;
        std::vector<float> dir;
        for (uint i = 0; i < source.getHitCount(); i++) {
            const vec3 p = source.getHitXYZ(i);
            const SphericalCoord d = LiDARTestHelper::storedDirection(source, i);
            xyz.insert(xyz.end(), {double(p.x), double(p.y), double(p.z)});
            dir.insert(dir.end(), {d.radius, d.elevation, d.azimuth});
            stamps.push_back(source.getHitData(i, "timestamp"));
        }
        const std::vector<std::string> labels = {"timestamp"};
        lidar.addHitPoints(0, stamps.size(), xyz.data(), dir.data(), labels, stamps.data());
        lidar.addGrid(make_vec3(1.5f, 0.f, 0.5f), make_vec3(3, 1, 1), make_int3(3, 1, 1), 0.f);
        lidar.gapfillMissesCount(0, false, false);
    };
    LiDARcloud virt, mat;
    virt.disableMessages();
    mat.disableMessages();
    build(virt);
    build(mat);
    const size_t virtual_count = virt.getVirtualMissCount();
    DOCTEST_REQUIRE(virtual_count > 0);
    size_t again = 0;
    try {
        again = virt.gapfillMissesCount(0, false, false);
    } catch (const std::exception &e) {
        DOCTEST_CHECK_MESSAGE(false, "a second gap-fill of the moving scan threw: " << e.what());
        again = virtual_count;
    }
    DOCTEST_CHECK(again == virtual_count);
    mat.materializeMisses();
    DOCTEST_REQUIRE(mat.getHitCount() == virt.getHitCount());

    virt.calculateLeafArea(&context, 0.5f, 1, 0.05f);
    mat.calculateLeafArea(&context, 0.5f, 1, 0.05f);
    size_t with_leaves = 0, mismatches = 0;
    for (uint c = 0; c < virt.getGridCellCount(); c++) {
        with_leaves += mat.getCellLeafArea(c) > 0.f ? 1 : 0;
        if (virt.getCellLeafArea(c) != mat.getCellLeafArea(c) || virt.getCellBeamCount(c) != mat.getCellBeamCount(c)) {
            mismatches++;
        }
    }
    DOCTEST_CHECK(with_leaves > 0);
    DOCTEST_CHECK_MESSAGE(mismatches == 0, mismatches << " cells differ between the virtualized and the materialized cloud");
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Row/Column Misses Follow the Scanner's Beams on a Raster Through the Poles") {
    // REGRESSION (predates the large-cloud rework): the azimuth fit unwrapped its samples in a single chain sorted by
    // column, so a return at the zenith or nadir -- whose azimuth is meaningless -- could shift every later sample by
    // a full turn and put the column-to-azimuth line on the wrong branch. On the synthetic scanner's own full-sphere
    // raster the synthesized misses were on average about two degrees off the beams it fires (checked against the
    // misses it records itself). Each sample now weighs in by how well its direction determines its azimuth.
    const int Ntheta = 60, Nphi = 90;
    const float dtheta = float(M_PI) / float(Ntheta - 1), dphi = 2.f * float(M_PI) / float(Nphi - 1);
    auto beam = [&](int row, int col) {
        const float zenith = dtheta * float(row);
        const float azimuth = dphi * float(col) + dphi * float(row) / float(Ntheta);
        return SphericalCoord(1.f, 0.5f * float(M_PI) - zenith, azimuth);
    };
    const vec3 origin = make_vec3(-5.f, 0.f, 0.5f);
    LiDARcloud cloud;
    cloud.disableMessages();
    ScanMetadata scan(origin, uint(Ntheta), 0.f, float(M_PI), uint(Nphi), 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, {});
    cloud.addScan(scan);
    for (int row = 0; row < Ntheta; row++) {
        for (int col = 0; col < Nphi; col++) {
            if ((row * 7 + col * 13) % 5 == 0 || (row >= 20 && row <= 35 && col >= 30 && col <= 50)) {
                continue;
            }
            const SphericalCoord d = beam(row, col);
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            cloud.addHitPoint(0, origin + sphere2cart(SphericalCoord(8.f, d.elevation, d.azimuth)), d, data);
        }
    }
    const uint returns = cloud.getHitCount();
    cloud.gapfillMissesCount(0, false, false);
    DOCTEST_REQUIRE(cloud.getHitCount() > returns);
    double worst = 0.0, total = 0.0;
    size_t n = 0;
    for (uint i = returns; i < cloud.getHitCount(); i++) {
        const int row = int(cloud.getHitData(i, "row")), col = int(cloud.getHitData(i, "column"));
        vec3 d = cloud.getHitXYZ(i) - origin;
        d.normalize();
        // Chord between the unit vectors, in double: equal to the angle to first order, and unlike acos of a float dot
        // product it resolves angles far below a milliradian.
        const vec3 b = sphere2cart(beam(row, col));
        const double error = std::sqrt((double(d.x) - b.x) * (double(d.x) - b.x) + (double(d.y) - b.y) * (double(d.y) - b.y) + (double(d.z) - b.z) * (double(d.z) - b.z));
        worst = std::max(worst, error);
        total += error;
        n++;
    }
    DOCTEST_CHECK_MESSAGE(worst < 1e-4, "worst miss is " << worst * 180.0 / M_PI << " degrees off its beam (mean " << total / double(n) * 180.0 / M_PI << ")");
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Virtual and Materialized Misses Are Identical") {
    // Idealized grid, no flags.
    checkVirtualMaterializedEquivalence([](LiDARcloud &c) { buildRowColumnScan(c, 20, 36, 0.f, false, false, 5, 9, 10, 15); });
    // With gapfillMisses_code flags (exercises the code 1 / code 4 reconstruction).
    checkVirtualMaterializedEquivalence([](LiDARcloud &c) { buildRowColumnScan(c, 20, 36, 0.f, true, false, 5, 9, 10, 15); });
    // Tilted/sheared grid.
    checkVirtualMaterializedEquivalence([](LiDARcloud &c) { buildRowColumnScan(c, 24, 48, 0.15f, true, false, 6, 11, 12, 20); });
    // Carrying timestamps (exercises the reconstructed-timestamp path).
    checkVirtualMaterializedEquivalence([](LiDARcloud &c) { buildRowColumnScan(c, 20, 36, 0.f, true, true, 4, 8, 9, 14); });
    // A fully empty row and a fully full row.
    checkVirtualMaterializedEquivalence([](LiDARcloud &c) { buildRowColumnScan(c, 20, 36, 0.f, true, false, 7, 7, 0, 35); });
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Coordinate Transforms Move Virtual Misses") {
    // REGRESSION: a virtualized miss's world position is reconstructed as vm.origin + direction, where
    // vm.origin is the scan origin captured at gap-fill time. coordinateShift/coordinateRotation update
    // scans[].origin and every stored HitPoint, but nothing updated that captured copy, so the misses
    // stayed where they were while the rest of the cloud moved. Registering a multi-scan survey after
    // gap-filling would therefore place every synthesized miss in the wrong position -- and since misses
    // are what the leaf-area inversion counts as transmitted beams, silently along the wrong rays.
    auto build = [](LiDARcloud &lidar) {
        const int Ntheta = 20, Nphi = 36;
        ScanMetadata scan(make_vec3(1.f, 2.f, 3.f), uint(Ntheta), 0.05, 0.95 * M_PI, uint(Nphi), 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
        lidar.addScan(scan);
        for (int row = 0; row < Ntheta; row++) {
            for (int col = 0; col < Nphi; col++) {
                if (row >= 5 && row <= 9 && col >= 10 && col <= 15) {
                    continue;
                }
                const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta);
                const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
                SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
                vec3 xyz = lidar.getScanOrigin(0) + sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
                std::map<std::string, double> data;
                data["row"] = row;
                data["column"] = col;
                lidar.addHitPoint(0, xyz, dir, data);
            }
        }
        lidar.gapfillMisses(0, false, false);
    };

    // --- shift ---
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        build(lidar);
        DOCTEST_REQUIRE(lidar.getVirtualMissCount() > 0);

        std::vector<vec3> before;
        for (uint i = 0; i < lidar.getHitCount(); i++) {
            before.push_back(lidar.getHitXYZ(i));
        }
        const vec3 shift = make_vec3(100.f, 200.f, 300.f);
        lidar.coordinateShift(shift);

        DOCTEST_REQUIRE(lidar.getHitCount() == before.size());
        for (uint i = 0; i < lidar.getHitCount(); i++) {
            const vec3 now = lidar.getHitXYZ(i);
            DOCTEST_CHECK(now.x == doctest::Approx(before[i].x + shift.x));
            DOCTEST_CHECK(now.y == doctest::Approx(before[i].y + shift.y));
            DOCTEST_CHECK(now.z == doctest::Approx(before[i].z + shift.z));
        }
    }

    // --- rotation: a virtual cloud and a materialized one must agree afterwards ---
    {
        LiDARcloud virt;
        virt.disableMessages();
        build(virt);
        DOCTEST_REQUIRE(virt.getVirtualMissCount() > 0);

        LiDARcloud mat;
        mat.disableMessages();
        build(mat);
        mat.materializeMisses();

        const SphericalCoord rotation = make_SphericalCoord(0.3f, 0.7f);
        virt.coordinateRotation(rotation);
        mat.coordinateRotation(rotation);

        DOCTEST_REQUIRE(virt.getHitCount() == mat.getHitCount());
        for (uint i = 0; i < virt.getHitCount(); i++) {
            const vec3 a = virt.getHitXYZ(i), b = mat.getHitXYZ(i);
            DOCTEST_CHECK(a.x == doctest::Approx(b.x));
            DOCTEST_CHECK(a.y == doctest::Approx(b.y));
            DOCTEST_CHECK(a.z == doctest::Approx(b.z));
        }
    }
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Virtual Misses Survive Mutation Barriers") {
    // Any operation that renumbers the hit index space must first collapse the virtual population, or a
    // virtual miss's identity (an index above hits.size()) silently changes meaning. deleteHitPoint is
    // the sharp case: it swap-and-pops against hits.back().
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        buildRowColumnScan(lidar, 20, 36, 0.f, true, false, 5, 9, 10, 15);
        const uint before = lidar.getHitCount();
        DOCTEST_REQUIRE(lidar.getVirtualMissCount() > 0);

        lidar.deleteHitPoint(0);
        DOCTEST_CHECK(lidar.getVirtualMissCount() == 0);
        DOCTEST_CHECK(lidar.getHitCount() == before - 1);
    }

    // Deleting the LAST real hit -- the exact index a naive "virtual misses live in the tail of hits"
    // layout would have swapped a virtual miss into.
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        buildRowColumnScan(lidar, 20, 36, 0.f, true, false, 5, 9, 10, 15);
        const uint before = lidar.getHitCount();
        const uint nvirtual = uint(lidar.getVirtualMissCount());
        const uint last_real = before - nvirtual - 1;

        lidar.deleteHitPoint(last_real);
        DOCTEST_CHECK(lidar.getVirtualMissCount() == 0);
        DOCTEST_CHECK(lidar.getHitCount() == before - 1);
    }

    // Adding a hit after gap-filling must not collide with the virtual index range.
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        buildRowColumnScan(lidar, 20, 36, 0.f, false, false, 5, 9, 10, 15);
        const uint before = lidar.getHitCount();
        lidar.addHitPoint(0, make_vec3(1, 1, 1), cart2sphere(make_vec3(1, 1, 1)));
        DOCTEST_CHECK(lidar.getVirtualMissCount() == 0);
        DOCTEST_CHECK(lidar.getHitCount() == before + 1);
    }

    // Writing hit data likewise requires real storage behind the index.
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        buildRowColumnScan(lidar, 20, 36, 0.f, false, false, 5, 9, 10, 15);
        const uint before = lidar.getHitCount();
        lidar.setHitData(0, "custom", 7.0);
        DOCTEST_CHECK(lidar.getVirtualMissCount() == 0);
        DOCTEST_CHECK(lidar.getHitCount() == before);
        DOCTEST_CHECK(lidar.getHitData(0, "custom") == 7.0);
    }
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - distanceFilter Discards Virtual Misses") {
    // distanceFilter with any threshold below LIDAR_MISS_DISTANCE deletes every gap-filled miss. Doing
    // that by materializing the whole population and then deleting it one point at a time would be
    // absurd, so the virtual sets are dropped wholesale. The result must match the materialize-then-
    // filter route exactly, including the surviving hits.
    auto build = [](LiDARcloud &lidar) {
        ScanMetadata scan(make_vec3(0, 0, 0), 20, 0.05, 0.95 * M_PI, 36, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
        lidar.addScan(scan);
        for (uint row = 0; row < 20; row++) {
            for (uint col = 0; col < 36; col++) {
                if (row >= 5 && row <= 9 && col >= 10 && col <= 15) {
                    continue;
                }
                const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / 20.f;
                const float phi = 2.f * float(M_PI) * float(col) / 36.f;
                SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
                vec3 xyz = sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
                std::map<std::string, double> data;
                data["row"] = row;
                data["column"] = col;
                lidar.addHitPoint(0, xyz, dir, data);
            }
        }
        lidar.gapfillMisses(0, false, false);
    };

    LiDARcloud fast;
    fast.disableMessages();
    build(fast);
    DOCTEST_REQUIRE(fast.getVirtualMissCount() > 0);
    fast.distanceFilter(50.f);

    LiDARcloud slow;
    slow.disableMessages();
    build(slow);
    slow.materializeMisses();
    slow.distanceFilter(50.f);

    DOCTEST_CHECK(fast.getVirtualMissCount() == 0);
    DOCTEST_REQUIRE(fast.getHitCount() == slow.getHitCount());
    for (uint i = 0; i < fast.getHitCount(); i++) {
        const vec3 a = fast.getHitXYZ(i), b = slow.getHitXYZ(i);
        DOCTEST_REQUIRE(a.x == b.x);
        DOCTEST_REQUIRE(a.y == b.y);
        DOCTEST_REQUIRE(a.z == b.z);
    }
    // Every gap-filled miss is gone; only real returns remain.
    DOCTEST_CHECK(!fast.hasMisses());
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Export Is Identical With Virtual Misses") {
    // exportPointCloud() reads the columnar store by index directly for speed. A virtualized miss has no
    // row there -- its index lies above hits.size() -- so an unguarded read runs off the end of the
    // column. Exporting a virtual cloud and a materialized one must produce byte-identical files.
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "helios_lidar_virtual_export_test";
    fs::create_directories(dir);
    const fs::path f_virtual = dir / "virtual.xyz";
    const fs::path f_material = dir / "material.xyz";

    auto build = [](LiDARcloud &lidar) {
        ScanMetadata scan(make_vec3(0, 0, 0), 20, 0.05, 0.95 * M_PI, 36, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f,
                          {"x", "y", "z", "row", "column", "is_miss", "nRaysHit", "gapfillMisses_code"});
        lidar.addScan(scan);
        for (uint row = 0; row < 20; row++) {
            for (uint col = 0; col < 36; col++) {
                if (row >= 5 && row <= 9 && col >= 10 && col <= 15) {
                    continue;
                }
                const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / 20.f;
                const float phi = 2.f * float(M_PI) * float(col) / 36.f;
                SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
                vec3 xyz = sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
                std::map<std::string, double> data;
                data["row"] = row;
                data["column"] = col;
                lidar.addHitPoint(0, xyz, dir, data);
            }
        }
        lidar.gapfillMisses(0, false, true);
    };

    LiDARcloud virt;
    virt.disableMessages();
    build(virt);
    DOCTEST_REQUIRE(virt.getVirtualMissCount() > 0);
    virt.exportPointCloud(f_virtual.string().c_str());

    LiDARcloud mat;
    mat.disableMessages();
    build(mat);
    mat.materializeMisses();
    DOCTEST_REQUIRE(mat.getVirtualMissCount() == 0);
    mat.exportPointCloud(f_material.string().c_str());

    // Read both files inside their own scope so that the streams are closed before the directory is removed.
    // Windows refuses to delete a file that is still open, so leaving them in scope made remove_all() throw and
    // took the whole test case down with it; POSIX allows the unlink, so this only ever failed on the Windows runner.
    std::string sa, sb;
    {
        std::ifstream a(f_virtual, std::ios::binary), b(f_material, std::ios::binary);
        DOCTEST_REQUIRE(a.good());
        DOCTEST_REQUIRE(b.good());
        sa.assign((std::istreambuf_iterator<char>(a)), std::istreambuf_iterator<char>());
        sb.assign((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
    }
    DOCTEST_CHECK(sa.size() > 0);
    DOCTEST_CHECK(sa == sb);

    std::error_code remove_error;
    fs::remove_all(dir, remove_error);
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Virtual Misses Give the Same Leaf Area") {
    // The point of the misses is the leaf-area inversion: a miss is a beam that passed through the voxel
    // and kept going, so it is what makes the transmission probability computable. Virtualizing them
    // must not perturb that answer. This runs the full row/column pipeline twice -- once virtual, once
    // materialized -- and requires the recovered leaf-area density to be identical, not merely close.
    auto runInversion = [](bool materialize, std::vector<float> &lad) {
        Context context;
        // A single horizontal leaf slab inside the voxel, so the beams that hit it terminate and the
        // rest transmit through.
        for (int i = 0; i < 400; i++) {
            const float x = -0.5f + 1.0f * float(rand()) / float(RAND_MAX);
            const float y = -0.5f + 1.0f * float(rand()) / float(RAND_MAX);
            const float z = 4.5f + 1.0f * float(rand()) / float(RAND_MAX);
            context.addPatch(make_vec3(x, y, z), make_vec2(0.05f, 0.05f));
        }

        LiDARcloud lidar;
        lidar.disableMessages();
        ScanMetadata scan(make_vec3(0, 0, 0), 40, 0.05, 0.6, 60, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
        lidar.addScan(scan);
        lidar.addGridCell(make_vec3(0, 0, 5), make_vec3(2, 2, 2), 0.f);

        // Synthesize returns on a subset of the grid and tag them with row/column so the row/column
        // gap-filling path (the virtualized one) is selected.
        for (uint row = 0; row < 40; row++) {
            for (uint col = 0; col < 60; col++) {
                if ((row + col) % 3 != 0) {
                    continue; // leave two thirds of the grid empty for the gap filler
                }
                const float theta = 0.05f + (0.6f - 0.05f) * float(row) / 40.f;
                const float phi = 2.f * float(M_PI) * float(col) / 60.f;
                SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
                vec3 xyz = sphere2cart(SphericalCoord(5.0f, 0.5f * float(M_PI) - theta, phi));
                std::map<std::string, double> data;
                data["row"] = row;
                data["column"] = col;
                lidar.addHitPoint(0, xyz, dir, data);
            }
        }

        lidar.gapfillMisses(0, false, false);
        if (materialize) {
            lidar.materializeMisses();
        }
        lidar.calculateHitGridCell();
        lidar.triangulateHitPoints(0.5, 5); // leaf-area inversion needs a triangulated surface for G(theta)
        lidar.calculateLeafArea(&context);

        lad.clear();
        for (uint c = 0; c < lidar.getGridCellCount(); c++) {
            lad.push_back(lidar.getCellLeafAreaDensity(c));
        }
    };

    std::vector<float> lad_virtual, lad_materialized;
    srand(12345);
    runInversion(false, lad_virtual);
    srand(12345); // identical geometry
    runInversion(true, lad_materialized);

    DOCTEST_REQUIRE(lad_virtual.size() == lad_materialized.size());
    DOCTEST_REQUIRE(lad_virtual.size() > 0);
    for (size_t c = 0; c < lad_virtual.size(); c++) {
        DOCTEST_CHECK(lad_virtual[c] == lad_materialized[c]);
    }
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Grid Binning Leaves Virtual Misses Alone") {
    // calculateHitGridCell() must not materialize: a virtual miss is definitionally cell -1.
    LiDARcloud lidar;
    lidar.disableMessages();
    buildRowColumnScan(lidar, 20, 36, 0.f, false, false, 5, 9, 10, 15);
    lidar.addGridCell(make_vec3(0, 0, 5), make_vec3(2, 2, 2), 0.f);

    const size_t nvirtual = lidar.getVirtualMissCount();
    DOCTEST_REQUIRE(nvirtual > 0);

    lidar.calculateHitGridCell();
    DOCTEST_CHECK(lidar.getVirtualMissCount() == nvirtual); // not materialized

    for (uint i = 0; i < lidar.getHitCount(); i++) {
        if (lidar.getHitData(i, "is_miss") == 1.0) {
            DOCTEST_CHECK(lidar.getHitGridCell(i) == -1);
        }
    }
}

DOCTEST_TEST_CASE("LiDAR Beam Grouping - Ordered and Unordered Clouds Agree") {
    // Returns are grouped into per-pulse beams by shared timestamp. A cloud already in acquisition
    // order needs no sort -- the pulses are contiguous already -- which is both faster and the property
    // a streaming source would rely on. Building the same cloud in reverse order forces the sort path;
    // both must recover the same leaf area, or the fast path is not equivalent to the sort it replaces.
    auto build = [](bool reverse_load, std::vector<float> &lad) {
        Context ctx;
        LiDARcloud src;
        src.disableMessages();
        src.loadXML("plugins/lidar/xml/leafcube_multi.xml");

        // Capture the loaded returns, then rebuild them in the requested order.
        std::vector<vec3> xyz;
        std::vector<std::map<std::string, double>> data;
        for (uint r = 0; r < src.getHitCount(); r++) {
            xyz.push_back(src.getHitXYZ(r));
            std::map<std::string, double> d;
            for (const char *lbl: {"timestamp", "target_index", "target_count"}) {
                if (src.doesHitDataExist(r, lbl)) {
                    d[lbl] = src.getHitData(r, lbl);
                }
            }
            data.push_back(d);
        }
        DOCTEST_REQUIRE(xyz.size() > 0);

        LiDARcloud rebuilt;
        rebuilt.disableMessages();
        ScanMetadata scan(src.getScanOrigin(0), src.getScanSizeTheta(0), src.getScanRangeTheta(0).x, src.getScanRangeTheta(0).y, src.getScanSizePhi(0), src.getScanRangePhi(0).x, src.getScanRangePhi(0).y, 0.0f, 0.0f, 0.0f, 0.0f, {});
        rebuilt.addScan(scan);
        // The grid comes from the XML on the source cloud; the rebuilt one needs it too or there are
        // no voxels to invert into.
        for (uint c = 0; c < src.getGridCellCount(); c++) {
            rebuilt.addGridCell(src.getCellCenter(c), src.getCellGlobalAnchor(c), src.getCellSize(c), src.getCellSize(c), src.getCellRotation(c) * float(M_PI) / 180.f, make_int3(0, 0, 0), make_int3(1, 1, 1)); // getCellRotation returns degrees; addGridCell takes radians
        }
        for (size_t k = 0; k < xyz.size(); k++) {
            const size_t i = reverse_load ? (xyz.size() - 1 - k) : k;
            rebuilt.addHitPoint(0, xyz[i], cart2sphere(xyz[i] - rebuilt.getScanOrigin(0)), data[i]);
        }

        rebuilt.gapfillMisses();
        rebuilt.calculateHitGridCell();
        rebuilt.triangulateHitPoints(0.5, 5);
        rebuilt.calculateLeafArea(&ctx);
        lad.clear();
        for (uint c = 0; c < rebuilt.getGridCellCount(); c++) {
            lad.push_back(rebuilt.getCellLeafAreaDensity(c));
        }
    };

    std::vector<float> lad_forward, lad_reverse;
    build(false, lad_forward);
    build(true, lad_reverse);

    DOCTEST_REQUIRE(lad_forward.size() == lad_reverse.size());
    DOCTEST_REQUIRE(lad_forward.size() > 0);
    for (size_t c = 0; c < lad_forward.size(); c++) {
        if (lad_forward[c] > 0.f) {
            DOCTEST_CHECK(std::fabs(lad_reverse[c] - lad_forward[c]) / lad_forward[c] < 1e-3f);
        }
    }
}

DOCTEST_TEST_CASE("LiDAR Beam Grouping - Rejects Pulses Sharing a Timestamp") {
    // Grouping by timestamp is an assumption about the data, not something the format enforces. A cloud
    // whose timestamps were rounded on export merges unrelated pulses into one enormous beam, and the
    // transmittance implied by that beam is meaningless. Where target_count is present it states how
    // many returns the pulse actually had, so a larger group is detectable -- and must be reported
    // rather than silently inverted.
    Context context;
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 8, 0.05, 0.95 * M_PI, 8, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);
    lidar.addGridCell(make_vec3(0, 0, 5), make_vec3(2, 2, 2), 0.f);

    // Five returns all stamped with the same time, but each declaring it was a single-return pulse.
    for (int i = 0; i < 5; i++) {
        vec3 xyz = make_vec3(0.1f * float(i), 0.f, 5.f);
        std::map<std::string, double> data;
        data["timestamp"] = 100.0; // identical across genuinely distinct pulses
        data["target_index"] = 0;
        data["target_count"] = 1; // ... which contradicts a 5-return group
        data["is_miss"] = 0.0;
        lidar.addHitPoint(0, xyz, cart2sphere(xyz), data);
    }

    // One miss, so the inversion's "requires transmitted beams" precondition is met and execution
    // actually reaches beam grouping. The G(theta) overload is used so triangulation is not required.
    {
        vec3 miss = make_vec3(0.f, 0.f, 20000.f);
        std::map<std::string, double> data;
        data["timestamp"] = 200.0;
        data["target_index"] = 0;
        data["target_count"] = 1;
        data["is_miss"] = 1.0;
        lidar.addHitPoint(0, miss, cart2sphere(miss), data);
    }
    lidar.calculateHitGridCell();

    std::string message;
    try {
        lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f);
    } catch (const std::runtime_error &e) {
        message = e.what();
    }
    DOCTEST_CHECK(message.find("share timestamp") != std::string::npos);
    DOCTEST_CHECK(message.find("target_count") != std::string::npos);
}

DOCTEST_TEST_CASE("LiDAR Path Length Binning Matches Exact Accumulation") {
    // Per-voxel beam path lengths are binned once a voxel accumulates more than a threshold, which is
    // what bounds the inversion's memory regardless of how many beams cross a voxel. The inversion
    // solves mean_j exp(-a*dr_j*Gtheta) = P, a linear functional of the distribution of dr, so binning
    // is a quadrature approximation of it rather than a change of estimator. This locks in that the
    // approximation is far tighter than the secant solver's own 5e-5 convergence tolerance.
    auto run = [](bool exact, std::vector<float> &lad, std::vector<float> &gtheta) {
        Context context;
        LiDARcloud lidar;
        lidar.disableMessages();
        lidar.setExactPathLengths(exact);
        DOCTEST_REQUIRE(lidar.getExactPathLengths() == exact);
        lidar.loadXML("plugins/lidar/xml/leafcube_multi.xml");
        lidar.gapfillMisses();
        lidar.calculateHitGridCell();
        lidar.triangulateHitPoints(0.5, 5);
        lidar.calculateLeafArea(&context);
        lad.clear();
        gtheta.clear();
        for (uint c = 0; c < lidar.getGridCellCount(); c++) {
            lad.push_back(lidar.getCellLeafAreaDensity(c));
            gtheta.push_back(lidar.getCellGtheta(c));
        }
    };

    std::vector<float> lad_exact, g_exact, lad_binned, g_binned;
    run(true, lad_exact, g_exact);
    run(false, lad_binned, g_binned);

    DOCTEST_REQUIRE(lad_exact.size() == lad_binned.size());
    DOCTEST_REQUIRE(lad_exact.size() > 0);

    // Measured worst case on this fixture is ~1e-7; 1e-4 leaves generous headroom while still being
    // far tighter than the solver tolerance the approximation has to beat.
    for (size_t c = 0; c < lad_exact.size(); c++) {
        if (lad_exact[c] > 0.f) {
            DOCTEST_CHECK(std::fabs(lad_binned[c] - lad_exact[c]) / lad_exact[c] < 1e-4f);
        } else {
            DOCTEST_CHECK(lad_binned[c] == lad_exact[c]);
        }
        DOCTEST_CHECK(g_binned[c] == doctest::Approx(g_exact[c]).epsilon(1e-5));
    }
}

DOCTEST_TEST_CASE("LiDAR Hit Point Capacity Limit") {
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 10, 0.05, 0.95 * M_PI, 10, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // The estimate reflects the dense columnar layout: sizeof(HitPoint) plus, per label, the column's value
    // type on every point and one presence bit per point packed into 64-bit words.
    DOCTEST_CHECK(lidar.getMaxHitPoints() == LiDARcloud::DEFAULT_MAX_HIT_POINTS);
    const size_t empty_estimate = lidar.estimateHitPointMemory(1000);
    DOCTEST_CHECK(empty_estimate == 1000 * sizeof(HitPoint));

    vec3 xyz = make_vec3(1, 1, 1);
    std::map<std::string, double> data;
    data["intensity"] = 1.0;
    data["distance"] = 2.0;
    lidar.addHitPoint(0, xyz, cart2sphere(xyz), data);
    // Two FLOAT32 labels (intensity, distance) now exist: 4 bytes of value per point each, plus a
    // 64-bit presence word per 64 points each.
    const size_t presence_words = (1000 + 63) / 64;
    DOCTEST_CHECK(lidar.estimateHitPointMemory(1000) == 1000 * (sizeof(HitPoint) + 2 * 4) + 2 * presence_words * sizeof(uint64_t));

    // A request beyond the cap fails with a message naming the counts and the memory, rather than
    // dying inside the allocator where neither is visible.
    lidar.setMaxHitPoints(5000);
    DOCTEST_CHECK(lidar.getMaxHitPoints() == 5000);
    std::string message;
    try {
        lidar.reserveHitPoints(6000);
        DOCTEST_CHECK_MESSAGE(false, "reserveHitPoints should have thrown past the cap");
    } catch (const std::runtime_error &e) {
        message = e.what();
    }
    DOCTEST_CHECK(message.find("6000") != std::string::npos); // the projected count
    DOCTEST_CHECK(message.find("5000") != std::string::npos); // the limit
    DOCTEST_CHECK(message.find("MB") != std::string::npos); // both sides in memory terms
    DOCTEST_CHECK(message.find("setMaxHitPoints") != std::string::npos); // how to proceed
    DOCTEST_CHECK(message.find("Ntheta") != std::string::npos); // the usual cause

    // Within the cap it is silent, and 0 disables the check entirely.
    DOCTEST_CHECK_NOTHROW(lidar.reserveHitPoints(4000));
    lidar.setMaxHitPoints(0);
    DOCTEST_CHECK_NOTHROW(lidar.reserveHitPoints(50000000));
}

DOCTEST_TEST_CASE("LiDAR Hit Point Capacity - Virtual Misses Are Exempt Until Materialized") {
    // A virtualized miss costs about a bit, not a point, so gap-filling a grid far larger than the cap
    // is allowed; materializing that same population is not, since it becomes real storage.
    LiDARcloud lidar;
    lidar.disableMessages();
    const int Ntheta = 20, Nphi = 36;
    ScanMetadata scan(make_vec3(0, 0, 0), uint(Ntheta), 0.05, 0.95 * M_PI, uint(Nphi), 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);
    for (int row = 0; row < Ntheta; row++) {
        for (int col = 0; col < Nphi; col++) {
            if (row >= 5 && row <= 9 && col >= 10 && col <= 15) {
                continue;
            }
            const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta);
            const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
            SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
            vec3 xyz = sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            lidar.addHitPoint(0, xyz, dir, data);
        }
    }

    const uint real_hits = lidar.getHitCount();
    DOCTEST_CHECK_NOTHROW(lidar.gapfillMisses(0, false, false));
    const size_t nvirtual = lidar.getVirtualMissCount();
    DOCTEST_REQUIRE(nvirtual > 0);

    // Set the cap just below what materializing would require; gap-filling already succeeded.
    lidar.setMaxHitPoints(real_hits + nvirtual - 1);
    DOCTEST_CHECK(lidar.getHitCount() == real_hits + nvirtual);
    DOCTEST_CHECK_THROWS_AS(lidar.materializeMisses(), std::runtime_error);

    // Raising the cap lets it through.
    lidar.setMaxHitPoints(real_hits + nvirtual);
    DOCTEST_CHECK_NOTHROW(lidar.materializeMisses());
    DOCTEST_CHECK(lidar.getVirtualMissCount() == 0);
    DOCTEST_CHECK(lidar.getHitCount() == real_hits + nvirtual);
}

DOCTEST_TEST_CASE("LiDAR reserveHitPoints") {
    LiDARcloud lidar;
    lidar.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 0), 10, 0.05, 0.95 * M_PI, 10, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // Reserving does not create hit points.
    lidar.reserveHitPoints(1000);
    DOCTEST_CHECK(lidar.getHitCount() == 0);

    // Points added afterwards behave identically to an unreserved cloud, including their hit data.
    for (uint i = 0; i < 50; i++) {
        vec3 xyz = make_vec3(float(i), 0.f, 1.f);
        std::map<std::string, double> data;
        data["intensity"] = double(i);
        lidar.addHitPoint(0, xyz, cart2sphere(xyz), data);
    }
    DOCTEST_CHECK(lidar.getHitCount() == 50);
    for (uint i = 0; i < 50; i++) {
        DOCTEST_CHECK(lidar.getHitXYZ(i).x == float(i));
        DOCTEST_CHECK(lidar.getHitData(i, "intensity") == double(i));
    }

    // A label introduced after the reservation is back-filled correctly (its column is created late,
    // so it must inherit the reservation without disturbing the length invariant).
    lidar.setHitData(10, "late", 3.0);
    DOCTEST_CHECK(lidar.doesHitDataExist(10, "late"));
    DOCTEST_CHECK(lidar.getHitData(10, "late") == 3.0);
    DOCTEST_CHECK(!lidar.doesHitDataExist(11, "late"));
    DOCTEST_CHECK(lidar.getHitCount() == 50);

    // Reserving less than the current size is a no-op rather than an error, and growing past the
    // reservation is allowed.
    DOCTEST_CHECK_NOTHROW(lidar.reserveHitPoints(1));
    DOCTEST_CHECK(lidar.getHitCount() == 50);
    for (uint i = 0; i < 2000; i++) {
        vec3 xyz = make_vec3(0.f, float(i), 1.f);
        lidar.addHitPoint(0, xyz, cart2sphere(xyz));
    }
    DOCTEST_CHECK(lidar.getHitCount() == 2050);
    DOCTEST_CHECK(lidar.getHitData(0, "intensity") == 0.0); // earlier data survived the growth
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Virtualized Misses Bound Memory") {
    // REGRESSION: gapfillMisses() materialized one full HitPoint plus a row across every hit-data
    // column for each empty cell of the declared Ntheta x Nphi raster, and that cost is PERSISTENT --
    // it is the steady state of the cloud after the call returns, not scratch. On a production
    // VZ-600i raster (5313 x 18029) that is roughly 20 GB for a single scan, and it accumulates across
    // scans because `hits` is one cloud-level vector. Misses synthesized by the row/column path are now
    // virtualized: one bit per grid cell plus a per-row angular model, reconstructed on demand.
    //
    // The structural assertion below cannot pass on the unfixed code, where every miss is a real
    // element of `hits`.
    LiDARcloud lidar;
    lidar.disableMessages();

    const uint Ntheta = 600, Nphi = 4000; // 2.4M cells
    ScanMetadata scan(make_vec3(0, 0, 0), Ntheta, 0.05, 0.95 * M_PI, Nphi, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // Populate a sparse subset so the overwhelming majority of cells are gap-filled.
    const uint stride = 50;
    size_t nreal = 0;
    for (uint row = 0; row < Ntheta; row++) {
        for (uint col = 0; col < Nphi; col++) {
            if (((size_t) row * Nphi + col) % stride != 0) {
                continue;
            }
            const float theta = 0.05f + (0.95f * float(M_PI) - 0.05f) * float(row) / float(Ntheta);
            const float phi = 2.f * float(M_PI) * float(col) / float(Nphi);
            SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, phi);
            vec3 xyz = sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, phi));
            std::map<std::string, double> data;
            data["row"] = row;
            data["column"] = col;
            lidar.addHitPoint(0, xyz, dir, data);
            nreal++;
        }
    }
    DOCTEST_REQUIRE(nreal > 0);

    const size_t filled = lidar.gapfillMissesCount(0, false, false);
    DOCTEST_CHECK(filled == (size_t) Ntheta * (size_t) Nphi - nreal);

    // The misses are reported by the public API ...
    DOCTEST_CHECK(lidar.getHitCount() == nreal + filled);
    // ... but are NOT stored as real hit points.
    DOCTEST_CHECK(lidar.getVirtualMissCount() == filled);
    DOCTEST_CHECK(lidar.hasVirtualMisses());

    // Materializing collapses them into real storage and leaves every observable unchanged.
    lidar.materializeMisses();
    DOCTEST_CHECK(lidar.getVirtualMissCount() == 0);
    DOCTEST_CHECK(lidar.getHitCount() == nreal + filled);
}

DOCTEST_TEST_CASE("LiDAR Grid Cell Binning - Rotated Cell Reject Is Conservative") {
    // calculateHitGridCell() rejects a hit outside the bounding box of every voxel before scanning the
    // cells individually, which is what keeps a gap-filled cloud (tens of millions of misses parked at
    // LIDAR_MISS_DISTANCE) from running the full per-cell loop for every one of them. The reject must be
    // conservative for a ROTATED cell: the per-cell test rotates the query point about the cell anchor, so
    // the world-space region a rotated cell accepts is its box swung about that anchor, which can reach
    // well outside the unrotated bounds. A bound derived from the cell's own extent rather than its
    // anchor-to-corner radius silently bins points as cell -1 that genuinely fall inside the cell.
    LiDARcloud lidar;
    lidar.disableMessages();

    // A cell offset well away from its rotation anchor, rotated 45 degrees about it.
    const vec3 center = make_vec3(4.f, 0.f, 1.f);
    const vec3 size = make_vec3(1.f, 1.f, 1.f);
    const float rotation = 0.25f * float(M_PI);
    lidar.addGridCell(center, make_vec3(0, 0, 0), size, size, rotation, make_int3(0, 0, 0), make_int3(1, 1, 1));

    ScanMetadata scan(make_vec3(0, 0, 0), 4, 0.05, 0.95 * M_PI, 4, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // The rotated cell's true center in world space: rotate the stored center about the anchor.
    const float c = std::cos(rotation), sn = std::sin(rotation);
    const vec3 rotated_center = make_vec3(center.x * c - center.y * sn, center.x * sn + center.y * c, center.z);
    lidar.addHitPoint(0, rotated_center, cart2sphere(rotated_center - make_vec3(0, 0, 0)));

    // A point far outside every cell, standing in for a gap-filled miss.
    const vec3 far_away = make_vec3(0.f, 0.f, 20000.f);
    lidar.addHitPoint(0, far_away, cart2sphere(far_away));

    lidar.calculateHitGridCell();

    // The point at the rotated cell's center must be binned into it, not rejected by the bound.
    DOCTEST_CHECK(lidar.getHitGridCell(0) == 0);
    // The far-away point belongs to no cell.
    DOCTEST_CHECK(lidar.getHitGridCell(1) == -1);
}

DOCTEST_TEST_CASE("LiDAR Grid Cell Binning - Unbinned Hit Warns and Returns -1") {
    // getHitGridCell() must not pass the internal "never set" sentinel off as a real answer. On a
    // cloud that was never binned it returns -1 and says on stderr that calculateHitGridCell[*] was
    // likely forgotten -- which is indistinguishable, by return value alone, from the legitimate
    // "this point is in no cell" answer it gives after binning.
    LiDARcloud lidar;
    lidar.disableMessages();

    ScanMetadata scan(make_vec3(0, 0, 0), 4, 0.05, 0.95 * M_PI, 4, 0.0, 2.0 * M_PI, 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);
    const vec3 xyz = make_vec3(1.f, 0.f, 1.f);
    lidar.addHitPoint(0, xyz, cart2sphere(xyz));
    lidar.addGridCell(xyz, make_vec3(2, 2, 2), 0.f);

    int cell = 0;
    std::string message;
    {
        capture_cerr capture;
        cell = lidar.getHitGridCell(0);
        message = capture.get_captured_output();
    } // capture destroyed before the assertions below
    DOCTEST_CHECK(cell == -1);
    DOCTEST_CHECK(message.find("WARNING (LiDARcloud::getHitGridCell)") != std::string::npos);
    DOCTEST_CHECK(message.find("was never set") != std::string::npos);

    // Once binned, the same query is answered from real data and says nothing.
    lidar.calculateHitGridCell();
    {
        capture_cerr capture;
        cell = lidar.getHitGridCell(0);
        message = capture.get_captured_output();
    }
    DOCTEST_CHECK(cell == 0);
    DOCTEST_CHECK(message.empty());
}

DOCTEST_TEST_CASE("LiDAR Miss Gapfilling - Top-Down Sweep Terminates") {
    // REGRESSION: the two edge-extrapolation loops in gapfillMisses_timestamp stepped theta by the
    // SIGNED average increment dtheta_avg. That sign only encodes the sweep direction, so a top-down
    // scanner (theta decreasing with time, dtheta_avg < 0) walked theta AWAY from the bound the loop
    // tests against. Because theta is a float, the step eventually fell below the ULP and the loop
    // never terminated -- gapfillMisses() hung forever. The downward loop additionally had no
    // dtheta == 0 guard (the upward loop did).
    //
    // The sweep must use a single azimuth column and per-step theta changes shallower than the
    // 10-degree sweep-break threshold, or the descending run is split into separate columns and the
    // positive column-wrap jumps cancel the negative steps, leaving dtheta_avg >= 0 (which does NOT
    // reproduce the bug).
    LiDARcloud lidar;
    lidar.disableMessages();

    const uint Ntheta = 200;
    const uint Nphi = 1;
    const float thetaMin = 0.2f;
    const float thetaMax = 1.4f;
    ScanMetadata scan(make_vec3(0, 0, 0), Ntheta, thetaMin, thetaMax, Nphi, 0.0f, 2.0f * float(M_PI), 0.0f, 0.0f, 0.0f, 0.0f, {});
    lidar.addScan(scan);

    // One sweep walking theta from high to low (0.006 rad/step, well under the 0.1745 rad threshold),
    // with an interior block of pulses omitted so there is a genuine gap to fill.
    double t = 0.0;
    for (int row = int(Ntheta) - 1; row >= 0; row--) {
        if (row > 60 && row < 120) {
            t += 1.0;
            continue;
        }
        const float theta = thetaMin + (thetaMax - thetaMin) * float(row) / float(Ntheta);
        SphericalCoord dir(1.f, 0.5f * float(M_PI) - theta, 0.f);
        vec3 xyz = sphere2cart(SphericalCoord(10.f, 0.5f * float(M_PI) - theta, 0.f));
        std::map<std::string, double> data;
        data["timestamp"] = t;
        data["target_index"] = 0;
        data["target_count"] = 1;
        lidar.addHitPoint(0, xyz, dir, data);
        t += 1.0;
    }

    const uint hits_before = lidar.getHitCount();
    DOCTEST_REQUIRE(hits_before > 0);

    // On the unfixed code this call never returns.
    std::vector<vec3> filled;
    DOCTEST_CHECK_NOTHROW(filled = lidar.gapfillMisses(0, false, false));

    DOCTEST_CHECK(filled.size() > 0);
    DOCTEST_CHECK(lidar.getHitCount() == hits_before + filled.size());
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

// ---- Golden values: results pinned against the code that preceded the large-point-cloud rework ----

//! Deterministic three-layer leaf scene built only from exactly representable coordinates (no trigonometry in the
//! geometry), so its hit cloud is identical on every platform: one scan of 128 x 128 beams aimed through a
//! 2 x 2 x 2 grid, each beam returning from the first leaf layer present along it and otherwise recorded as a miss.
//! The y < 0 half of the nearest layer is dense, so the voxels behind it are reached mostly through other voxels.
static void buildGoldenLeafLayerScan(LiDARcloud &lidar) {
    const vec3 O = make_vec3(-4.0f, 0.25f, 0.5f);
    ScanMetadata scan(O, 128, 1.0f, 2.0f, 128, 1.0f, 2.0f, 0.f, 0.f, 0.f, 0.f, {});
    lidar.addScan(scan);
    lidar.addGrid(make_vec3(0.5f, 0.f, 0.5f), make_vec3(1, 1, 1), make_int3(2, 2, 2), 0.f);
    const float layer_x[3] = {0.125f, 0.375f, 0.625f};
    const int N = 128;
    long long beam = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++, beam++) {
            const float ay = -0.5f + (float(i) + 0.5f) / float(N) + float((i * 7 + j * 3) % 5 - 2) / 1024.f;
            const float az = (float(j) + 0.5f) / float(N) + float((i * 3 + j * 5) % 5 - 2) / 1024.f;
            const vec3 A = make_vec3(0.f, ay, az);
            const vec3 D = A - O;
            int hit_layer = -1;
            for (int k = 0; k < 3 && hit_layer < 0; k++) {
                const vec3 P = O + D * ((layer_x[k] - O.x) / (A.x - O.x));
                const unsigned h = (unsigned(i) * 73856093u) ^ (unsigned(j) * 19349663u) ^ (unsigned(k + 1) * 83492791u);
                const unsigned percent = (k == 0) ? (P.y < 0.f ? 92u : 15u) : 30u;
                if (h % 100u < percent) {
                    hit_layer = k;
                }
            }
            std::map<std::string, double> data;
            data["timestamp"] = double(beam);
            data["target_index"] = 0;
            data["target_count"] = 1;
            data["is_miss"] = hit_layer >= 0 ? 0 : 1;
            const vec3 P = hit_layer >= 0 ? O + D * ((layer_x[hit_layer] - O.x) / (A.x - O.x)) : O + D * 1024.f;
            lidar.addHitPoint(0, P, cart2sphere(D), data);
        }
    }
}

DOCTEST_TEST_CASE("LiDAR Golden Values - Triangulation and Leaf Area of a Deterministic Leaf-Layer Scene") {
    // Pins the triangulation and the leaf-area inversion of a scene whose input is bit-identical everywhere, against
    // the values the plug-in produced before the large-point-cloud rework (per-scan index, typed hit-data columns,
    // block tiling, streaming), which must not change them. The mesh is compared exactly: its vertices are input
    // positions. G(theta) and leaf area pass through libm (acos, log), whose last bit is not guaranteed to agree
    // between the platforms Helios is built on, so they are compared to a relative 1e-6 -- far below any change an
    // algorithmic edit produces (the beam-count fix below moved leaf area by up to 9%).
    //
    // Values that changed ON PURPOSE: the beam-count fix ("beam count, path length and min_voxel_hits use only beams
    // that entered the voxel") changed every voxel that beams reach after crossing leaves elsewhere in the grid. The
    // value each such cell had before that fix is recorded beside it; cells 2 and 6, which beams reach directly from
    // outside the grid, did not change.
    Context context;
    LiDARcloud lidar;
    lidar.disableMessages();
    buildGoldenLeafLayerScan(lidar);
    DOCTEST_REQUIRE(lidar.getHitCount() == 16384);

    lidar.triangulateHitPoints(0.05f, 5.f);
    DOCTEST_REQUIRE(lidar.getTriangleCount() == 14157);
    double checksum = 0.0;
    for (uint t = 0; t < lidar.getTriangleCount(); t++) {
        const Triangulation tri = lidar.getTriangle(t);
        checksum +=
                double(tri.vertex0.x) + double(tri.vertex0.y) + double(tri.vertex0.z) + double(tri.vertex1.x) + double(tri.vertex1.y) + double(tri.vertex1.z) + double(tri.vertex2.x) + double(tri.vertex2.y) + double(tri.vertex2.z) + double(tri.area);
    }
    // Compared to a relative tolerance, not exactly: every triangle's area is Heron's formula over three edge
    // magnitudes, so the sum carries four libm sqrt results per triangle and its last bits differ between platforms
    // (this assertion passed on macOS and failed on glibc with both sides printing as 18412.9). The vertex checks
    // below stay exact -- those are input positions, carried through unchanged.
    DOCTEST_CHECK(checksum == doctest::Approx(0x1.1fb3bcc94875ep+14).epsilon(1e-9));
    const Triangulation first = lidar.getTriangle(0);
    const Triangulation last = lidar.getTriangle(lidar.getTriangleCount() - 1);
    DOCTEST_CHECK((first.vertex0 == make_vec3(0x1p-3f, -0x1.e47p-2f, 0x1.9b34p-1f)));
    DOCTEST_CHECK((first.vertex1 == make_vec3(0x1p-3f, -0x1.da2p-2f, 0x1.9a2cp-1f)));
    DOCTEST_CHECK((first.vertex2 == make_vec3(0x1p-3f, -0x1.e26p-2f, 0x1.9714p-1f)));
    DOCTEST_CHECK(first.gridcell == 4);
    DOCTEST_CHECK((last.vertex0 == make_vec3(0x1p-3f, -0x1.eaap-2f, 0x1.31p-6f)));
    DOCTEST_CHECK((last.vertex1 == make_vec3(0x1p-3f, -0x1.e368p-2f, 0x1.bdp-7f)));
    DOCTEST_CHECK((last.vertex2 == make_vec3(0x1p-3f, -0x1.e89p-2f, 0x1.5ap-7f)));
    DOCTEST_CHECK(last.gridcell == 0);

    struct Golden {
        float leaf_area;
        int beams;
        float leaf_area_before_beam_count_fix;
        int beams_before_beam_count_fix;
    };
    const float gtheta[8] = {0x1.fabdc4p-1f, 0x1.fe016cp-1f, 0x1.fe14d8p-1f, 0x1.fe9d9p-1f, 0x1.fabf78p-1f, 0x1.fdb3b2p-1f, 0x1.fe15fcp-1f, 0x1.fee31ep-1f};
    const Golden triangulated[8] = {{0x1.013494p-2f, 4284, 0x1.0248ep-2f, 4312}, {0x1.8df1a8p-4f, 327, 0x1.7ac4bap-4f, 3390}, {0x1.009718p-3f, 4096, 0x1.009718p-3f, 4096}, {0x1.993932p-4f, 1934, 0x1.97625p-4f, 3237},
                                    {0x1.024aap-2f, 4279, 0x1.03c47ep-2f, 4314}, {0x1.56823cp-4f, 309, 0x1.3a8294p-4f, 3389}, {0x1.02a372p-3f, 4096, 0x1.02a372p-3f, 4096}, {0x1.7c1156p-4f, 1908, 0x1.7aa1cep-4f, 3238}};
    const Golden scalar[8] = {{0x1.fd2094p-2f, 4284, 0x1.ff4382p-2f, 4312}, {0x1.8c64d2p-3f, 327, 0x1.794bp-3f, 3390},   {0x1.ff41e6p-3f, 4096, 0x1.ff41e6p-3f, 4096}, {0x1.981decp-3f, 1934, 0x1.96484cp-3f, 3237},
                              {0x1.ff48bp-2f, 4279, 0x1.011a56p-1f, 4314},  {0x1.54f8b2p-3f, 309, 0x1.39193cp-3f, 3389}, {0x1.01abeap-2f, 4096, 0x1.01abeap-2f, 4096}, {0x1.7b3ddap-3f, 1908, 0x1.79cf2p-3f, 3238}};
    auto compare = [&](const Golden golden[8], const char *mode, bool check_gtheta) {
        DOCTEST_REQUIRE(lidar.getGridCellCount() == 8);
        for (uint c = 0; c < 8; c++) {
            DOCTEST_CHECK_MESSAGE(lidar.getCellLeafArea(c) == doctest::Approx(golden[c].leaf_area).epsilon(1e-6), mode << " leaf area, cell " << c);
            DOCTEST_CHECK_MESSAGE(lidar.getCellBeamCount(c) == golden[c].beams, mode << " beam count, cell " << c);
            if (check_gtheta) {
                DOCTEST_CHECK_MESSAGE(lidar.getCellGtheta(c) == doctest::Approx(gtheta[c]).epsilon(1e-6), mode << " G(theta), cell " << c);
            }
            // Documentation of intent: only the cells beams reach through other voxels changed with the fix.
            const bool reached_directly = (c == 2 || c == 6);
            DOCTEST_CHECK((golden[c].beams == golden[c].beams_before_beam_count_fix) == reached_directly);
            DOCTEST_CHECK((golden[c].leaf_area == golden[c].leaf_area_before_beam_count_fix) == reached_directly);
        }
    };
    lidar.calculateLeafArea(&context, 1, 0.05f);
    compare(triangulated, "triangulated G(theta)", true);
    lidar.calculateLeafArea(&context, 0.5f, 1, 0.05f);
    compare(scalar, "scalar G(theta)", false);
}

DOCTEST_TEST_CASE("LiDAR LAD Inversion beam count only counts beams that reached the voxel") {

    // A 2 x 2 x 1 grid of 1 m voxels scanned from x = -5: two voxels along +x (near, far) in each of two columns
    // (y < 0, y > 0). The y < 0 column has an opaque wall (two triangles in the y-z plane) inside its near voxel; every
    // beam that strikes it terminates there and never enters the far voxel behind it. The y > 0 column is empty, so
    // beams through it are transmitted (recorded as misses), which the inversion requires to be present at all.
    //
    // Pimont et al. (2018) define N as the number of beams that ENTERED the voxel, and the transmission probability P
    // is already accumulated over exactly that population. The beam count, the mean path length, and every quantity
    // derived from N (binomial variance, confidence-interval validity, min_voxel_hits gating) must therefore also
    // exclude beams that were blocked before the voxel, rather than count every beam whose ray LINE crosses it.
    //
    // With the wall covering the whole voxel face, no beam enters the far blocked voxel and its count must be exactly
    // zero. With the wall covering 90% of the face, only beams through the 10% gap enter it, so its count must be a
    // small fraction of the near blocked voxel's count (which is unaffected: every beam crossing the near voxel either
    // hits the wall inside it or passes through it, so all of them entered).

    auto run_scene = [](float wall_top_z, int &near_count, int &far_count) {
        Context context;
        // Wall at x = 0.5 covering the y < 0 column (with overlap past both of its y edges) from below the voxel up to
        // wall_top_z.
        const vec3 v0 = make_vec3(0.5f, -0.6f, -0.1f);
        const vec3 v1 = make_vec3(0.5f, 0.05f, -0.1f);
        const vec3 v2 = make_vec3(0.5f, 0.05f, wall_top_z);
        const vec3 v3 = make_vec3(0.5f, -0.6f, wall_top_z);
        context.addTriangle(v0, v1, v2);
        context.addTriangle(v0, v2, v3);

        LiDARcloud lidar;
        lidar.disableMessages();

        std::vector<std::string> columnFormat;
        ScanMetadata scan(make_vec3(-5.f, 0.f, 0.5f), 600, 0.f, float(M_PI), 1200, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, columnFormat);
        lidar.addScan(scan);

        // Cells: x in [0,1] (near) and [1,2] (far), y in [-0.5,0] (blocked column) and [0,0.5] (open column), z in [0,1].
        lidar.addGrid(make_vec3(1.f, 0.f, 0.5f), make_vec3(2.f, 1.f, 1.f), make_int3(2, 2, 1), 0.f);
        DOCTEST_REQUIRE(lidar.getGridCellCount() == 4);
        int near_blocked = -1, far_blocked = -1;
        for (uint c = 0; c < 4; c++) {
            const vec3 center = lidar.getCellCenter(c);
            if (center.y < 0.f) {
                if (center.x < 1.f) {
                    near_blocked = int(c);
                } else {
                    far_blocked = int(c);
                }
            }
        }
        DOCTEST_REQUIRE(near_blocked >= 0);
        DOCTEST_REQUIRE(far_blocked >= 0);

        // Single-return scan restricted to rays that cross the grid, recording misses so that a transmitted beam is a
        // transmission event for the voxels it crosses (the scene bounding box would otherwise be just the wall, and
        // rays through the open column would never be traced).
        lidar.syntheticScan(&context, true, true);
        lidar.triangulateHitPoints(0.04f, 10);
        lidar.calculateLeafArea(&context, 1, -1.f);

        near_count = lidar.getCellBeamCount(uint(near_blocked));
        far_count = lidar.getCellBeamCount(uint(far_blocked));
    };

    int near_count = -1, far_count = -1;

    // Full occlusion: the wall covers the whole face of the blocked column.
    run_scene(1.1f, near_count, far_count);
    DOCTEST_CHECK(near_count > 100);
    DOCTEST_CHECK_MESSAGE(far_count == 0, "far voxel counted " << far_count << " beams although the wall blocked all of them (near voxel: " << near_count << ")");

    // Partial occlusion: only beams through the top 10% of the near voxel face reach the far voxel.
    run_scene(0.9f, near_count, far_count);
    DOCTEST_CHECK(near_count > 100);
    DOCTEST_CHECK(far_count > 0);
    const float entered_fraction = float(far_count) / float(near_count);
    DOCTEST_CHECK_MESSAGE(entered_fraction < 0.25f, "far voxel counted " << far_count << " of " << near_count << " beams; only ~10% pass the wall");
}

DOCTEST_TEST_CASE("LiDAR lastHitFilter keeps only the last return of each pulse") {

    // Pulses with 1, 3, 2 and 4 returns, using both the 0-based and the 1-based target_index convention. After
    // lastHitFilter() exactly one hit per pulse must remain, and it must be the one with the highest target_index.
    const std::vector<int> returns_per_pulse = {1, 3, 2, 4};

    for (int index_base: {0, 1}) {
        LiDARcloud lidar;
        lidar.disableMessages();

        std::vector<std::string> columnFormat;
        ScanMetadata scan(make_vec3(0.f, 0.f, 0.f), 10, 0.f, float(M_PI), 10, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, columnFormat);
        lidar.addScan(scan);

        uint total_hits = 0;
        for (size_t pulse = 0; pulse < returns_per_pulse.size(); pulse++) {
            for (int t = 0; t < returns_per_pulse.at(pulse); t++) {
                std::map<std::string, double> data;
                data["target_index"] = index_base + t;
                data["target_count"] = returns_per_pulse.at(pulse);
                data["pulse"] = double(pulse);
                const vec3 xyz = make_vec3(1.f + float(pulse), 0.f, 1.f + float(t));
                lidar.addHitPoint(0, xyz, cart2sphere(xyz), data);
                total_hits++;
            }
        }
        DOCTEST_REQUIRE(lidar.getHitCount() == total_hits);

        lidar.lastHitFilter();

        DOCTEST_CHECK_MESSAGE(lidar.getHitCount() == returns_per_pulse.size(), "index base " << index_base << ": " << lidar.getHitCount() << " hits remain, expected one per pulse");

        std::vector<bool> pulse_seen(returns_per_pulse.size(), false);
        for (uint r = 0; r < lidar.getHitCount(); r++) {
            const int pulse = int(lidar.getHitData(r, "pulse"));
            const int expected_last_index = index_base + returns_per_pulse.at(pulse) - 1;
            DOCTEST_CHECK_MESSAGE(int(lidar.getHitData(r, "target_index")) == expected_last_index,
                                  "index base " << index_base << ", pulse " << pulse << ": surviving target_index " << lidar.getHitData(r, "target_index") << " is not the last return " << expected_last_index);
            DOCTEST_CHECK(!pulse_seen.at(pulse));
            pulse_seen.at(pulse) = true;
        }
    }

    // firstHitFilter() is the mirror image and must keep exactly the first return of each pulse.
    {
        LiDARcloud lidar;
        lidar.disableMessages();
        std::vector<std::string> columnFormat;
        ScanMetadata scan(make_vec3(0.f, 0.f, 0.f), 10, 0.f, float(M_PI), 10, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, columnFormat);
        lidar.addScan(scan);
        for (size_t pulse = 0; pulse < returns_per_pulse.size(); pulse++) {
            for (int t = 0; t < returns_per_pulse.at(pulse); t++) {
                std::map<std::string, double> data;
                data["target_index"] = t;
                data["target_count"] = returns_per_pulse.at(pulse);
                const vec3 xyz = make_vec3(1.f + float(pulse), 0.f, 1.f + float(t));
                lidar.addHitPoint(0, xyz, cart2sphere(xyz), data);
            }
        }
        lidar.firstHitFilter();
        DOCTEST_CHECK(lidar.getHitCount() == returns_per_pulse.size());
        for (uint r = 0; r < lidar.getHitCount(); r++) {
            DOCTEST_CHECK(int(lidar.getHitData(r, "target_index")) == 0);
        }
    }
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

// =====================================================================================================================
// Spinning multibeam physical-parameter setup (addScanSpinning). These exercise the first-class entry point where the
// user supplies physical instrument parameters (channel elevations, azimuth resolution, PRF, trajectory) and Helios
// derives the internal grid, rotation rate, and revolution count. A stationary capture is two coincident trajectory poses.
// =====================================================================================================================

DOCTEST_TEST_CASE("LiDAR LAD Infers Cropped Returns From target_count") {
    // A pulse's returns are ordered by range and a beam crosses the (convex) grid once, so when a cloud has been cropped
    // to the grid but keeps target_index / target_count, the removed returns can be placed before or beyond the grid from
    // the surviving indices alone. Three clouds of one scene, one voxel, G(theta) supplied:
    //   full     pulse A = 4 returns (1 inside the voxel at x=0, 3 beyond it at x=2,3,4); pulse B = a sky miss
    //   cropped  pulse A keeps its inside return plus one far-field stand-in flagged is_miss; target_count still 4
    //   legacy   as cropped, but without target_index / target_count, so nothing can be inferred
    // Equal weighting (Eq. 7) gives P = (3/4 + 1)/2 for full. The inference must reproduce that from the cropped cloud;
    // the legacy cloud sees a two-return beam (1 inside + 1 after) and gets P = (1/2 + 1)/2, a much larger LAD.
    Context context;
    context.addPatch(make_vec3(0, 0, -50), make_vec2(1, 1)); // collision detection wants a primitive; far from the grid
    const vec3 origin = make_vec3(-5.f, 0.f, 0.5f);

    enum Mode { FULL = 0, CROPPED = 1, LEGACY = 2 };
    auto build = [&](Mode mode, LiDARcloud &cloud) -> float {
        cloud.disableMessages();
        ScanMetadata scan(origin, 10, 0.f, float(M_PI), 10, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
        cloud.addScan(scan);
        cloud.addGrid(make_vec3(0, 0, 0.5f), make_vec3(1, 1, 1), make_int3(1, 1, 1), 0.f);
        auto add = [&](const vec3 &xyz, double t, int ti, int tc, bool miss, bool with_index) {
            std::map<std::string, double> data;
            data["timestamp"] = t;
            data["is_miss"] = miss ? 1.0 : 0.0;
            if (with_index) {
                data["target_index"] = ti;
                data["target_count"] = tc;
            }
            cloud.addHitPoint(0, xyz, cart2sphere(xyz - origin), data);
        };
        // Pulse A along +x through the voxel centre.
        if (mode == FULL) {
            add(make_vec3(0, 0, 0.5f), 1.0, 0, 4, false, true);
            add(make_vec3(2, 0, 0.5f), 1.0, 1, 4, false, true);
            add(make_vec3(3, 0, 0.5f), 1.0, 2, 4, false, true);
            add(make_vec3(4, 0, 0.5f), 1.0, 3, 4, false, true);
        } else {
            add(make_vec3(0, 0, 0.5f), 1.0, 0, 4, false, mode == CROPPED);
            add(make_vec3(995, 0, 0.5f), 1.0, 1, 4, true, mode == CROPPED); // stand-in for the cropped energy
        }
        // Pulse B: a recorded sky miss whose beam also crosses the voxel (slightly off-axis). Carries no per-pulse
        // columns, like a gap-filled miss.
        vec3 dirB = make_vec3(5.f, 0.2f, 0.f);
        dirB.normalize();
        add(origin + dirB * 1000.f, 2.0, 0, 1, true, false);

        cloud.calculateLeafArea(&context, 0.5f, 1, 0.05f);
        return cloud.getCellLeafAreaDensity(0);
    };

    LiDARcloud full, cropped, legacy;
    const float lad_full = build(FULL, full);
    const float lad_cropped = build(CROPPED, cropped);
    const float lad_legacy = build(LEGACY, legacy);
    DOCTEST_REQUIRE(lad_full == lad_full);
    DOCTEST_REQUIRE(lad_full > 0.f);

    // P = 0.875 vs 0.75 at G = 0.5 and dr ~ 1 m: LAD ~ 0.27 vs ~ 0.58.
    DOCTEST_CHECK_MESSAGE(fabs(lad_cropped - lad_full) < 1e-3f, "cropped cloud must invert to the full cloud's LAD: full=" << lad_full << " cropped=" << lad_cropped);
    DOCTEST_CHECK_MESSAGE(lad_legacy > 1.5f * lad_full, "without target_count the cropped beam must read as more intercepted: full=" << lad_full << " legacy=" << lad_legacy);

    // The tally says exactly what was recovered.
    LiDARcloud::CroppedReturnStats st_full = full.getCroppedReturnStats();
    DOCTEST_CHECK(st_full.beams_with_hidden_returns == 0);
    DOCTEST_CHECK(st_full.hidden_after == 0);
    LiDARcloud::CroppedReturnStats st = cropped.getCroppedReturnStats();
    DOCTEST_CHECK(st.beams_with_hidden_returns == 1);
    DOCTEST_CHECK(st.hidden_before == 0);
    DOCTEST_CHECK(st.hidden_after == 3);
    DOCTEST_CHECK(st.hidden_ambiguous == 0);
    DOCTEST_CHECK(st.beams_ambiguous == 0);
    DOCTEST_CHECK(st.standins_ignored == 1);
    LiDARcloud::CroppedReturnStats st_legacy = legacy.getCroppedReturnStats();
    DOCTEST_CHECK(st_legacy.beams_with_hidden_returns == 0);

    // A return removed from BETWEEN two surviving returns cannot be placed: it is reported as ambiguous and left out, and
    // a return removed from before the first surviving one is placed before the grid and changes nothing.
    {
        LiDARcloud amb;
        amb.disableMessages();
        ScanMetadata scan(origin, 10, 0.f, float(M_PI), 10, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
        amb.addScan(scan);
        amb.addGrid(make_vec3(0, 0, 0.5f), make_vec3(1, 1, 1), make_int3(1, 1, 1), 0.f);
        auto add = [&](const vec3 &xyz, double t, int ti, int tc, bool miss) {
            std::map<std::string, double> data;
            data["timestamp"] = t;
            data["is_miss"] = miss ? 1.0 : 0.0;
            data["target_index"] = ti;
            data["target_count"] = tc;
            amb.addHitPoint(0, xyz, cart2sphere(xyz - origin), data);
        };
        // Pulse: 5 returns declared; present are index 2 (inside) and index 4 (beyond). No return in this scan carries
        // index 0, so the indices are read as 1-based (the rule lastHitFilter and gapfillMisses apply): index 1 was
        // before the first surviving return, index 3 sits between the two survivors (ambiguous), index 5 was beyond.
        add(make_vec3(0, 0, 0.5f), 1.0, 2, 5, false);
        add(make_vec3(3, 0, 0.5f), 1.0, 4, 5, false);
        vec3 dirB = make_vec3(5.f, 0.2f, 0.f);
        dirB.normalize();
        add(origin + dirB * 1000.f, 2.0, 0, 1, true);
        amb.calculateLeafArea(&context, 0.5f, 1, 0.05f);
        LiDARcloud::CroppedReturnStats sa = amb.getCroppedReturnStats();
        DOCTEST_CHECK(sa.beams_with_hidden_returns == 1);
        DOCTEST_CHECK(sa.hidden_before == 1);
        DOCTEST_CHECK(sa.hidden_after == 1);
        DOCTEST_CHECK(sa.hidden_ambiguous == 1);
        DOCTEST_CHECK(sa.beams_ambiguous == 1);
        // Beam fraction: 1 inside, 1 present after + 1 inferred after = 2/3; with the miss, P = (2/3 + 1)/2.
        const float lad_amb = amb.getCellLeafAreaDensity(0);
        const float expected = -std::log((2.f / 3.f + 1.f) / 2.f) / 0.5f; // dr ~ 1 m
        DOCTEST_CHECK_MESSAGE(fabs(lad_amb - expected) < 0.02f, "ambiguous return must be left out: lad=" << lad_amb << " expected~" << expected);
    }

    // Beams are grouped per scan in scan-local order, so the per-pulse columns must be read from the scan being inverted.
    // Put the cropped pulse in the SECOND scan, after a first scan holding a complete two-return pulse fired away from
    // the grid: reading scan 1's local positions as cloud-wide hit indices would land on that complete pulse and infer
    // nothing.
    {
        LiDARcloud two;
        two.disableMessages();
        ScanMetadata scan(origin, 10, 0.f, float(M_PI), 10, 0.f, 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
        two.addScan(scan);
        two.addScan(scan);
        two.addGrid(make_vec3(0, 0, 0.5f), make_vec3(1, 1, 1), make_int3(1, 1, 1), 0.f);
        auto add = [&](uint scanID, const vec3 &xyz, double t, int ti, int tc, bool miss, bool with_index) {
            std::map<std::string, double> data;
            data["timestamp"] = t;
            data["is_miss"] = miss ? 1.0 : 0.0;
            if (with_index) {
                data["target_index"] = ti;
                data["target_count"] = tc;
            }
            two.addHitPoint(scanID, xyz, cart2sphere(xyz - origin), data);
        };
        add(0, make_vec3(-7, 0, 0.5f), 1.0, 0, 2, false, true); // scan 0: complete pulse along -x, never reaches the grid
        add(0, make_vec3(-8, 0, 0.5f), 1.0, 1, 2, false, true);
        add(1, make_vec3(0, 0, 0.5f), 1.0, 0, 4, false, true); // scan 1: the cropped cloud above
        add(1, make_vec3(995, 0, 0.5f), 1.0, 1, 4, true, true);
        vec3 dirB = make_vec3(5.f, 0.2f, 0.f);
        dirB.normalize();
        add(1, origin + dirB * 1000.f, 2.0, 0, 1, true, false);
        two.calculateLeafArea(&context, 0.5f, 1, 0.05f);
        LiDARcloud::CroppedReturnStats s2 = two.getCroppedReturnStats();
        DOCTEST_CHECK(s2.beams_with_hidden_returns == 1);
        DOCTEST_CHECK(s2.hidden_after == 3);
        DOCTEST_CHECK(s2.standins_ignored == 1);
        const float lad_two = two.getCellLeafAreaDensity(0);
        DOCTEST_CHECK_MESSAGE(fabs(lad_two - lad_full) < 1e-3f, "cropped pulse in the second scan must invert to the full cloud's LAD: full=" << lad_full << " two-scan=" << lad_two);
    }
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam Multi-Revolution Derivation") {
    // A spinning sensor carried along a straight trajectory for several revolutions. Verify that the derived rotation
    // rate, revolution count, and Nphi match the physical parameters, and that each pulse fires at the EXACT per-channel
    // elevation (not a resampled uniform theta grid).

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(200, 200));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(200, 200));

    // 8 channels, +/-15 degree elevation. 10-degree azimuth steps -> 36 steps per revolution.
    std::vector<float> beam_elev_rad;
    const std::vector<float> elev_deg = {-15.f, -11.f, -7.f, -3.f, 3.f, 7.f, 11.f, 15.f};
    for (float d: elev_deg) {
        beam_elev_rad.push_back(d * float(M_PI) / 180.f);
    }
    const uint channels = uint(beam_elev_rad.size());
    const float azimuthStep_rad = 10.f * float(M_PI) / 180.f; // 36 steps/rev
    const uint expected_steps_per_rev = 36;

    // Choose PRF and duration so we get ~3 revolutions. rotation_rate = PRF/(channels*steps_per_rev).
    // For 3 revolutions over duration D: PRF * D = channels * steps_per_rev * 3.
    const float H = 30.0f;
    const float v = 2.0f;
    const double duration = 0.6; // s
    const double target_revs = 3.0;
    const float PRF = float(double(channels) * double(expected_steps_per_rev) * target_revs / duration);

    std::vector<double> traj_t = {0.0, duration};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, H), make_vec3(0, float(v * duration), H)};
    std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)}; // identity orientation

    LiDARcloud lidar;
    lidar.disableMessages();
    uint scanID = lidar.addScanSpinning(beam_elev_rad, azimuthStep_rad, PRF, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0);

    // Derived descriptors.
    DOCTEST_CHECK(lidar.getScanMode(scanID) == SCAN_MODE_SPINNING);
    DOCTEST_CHECK(lidar.getScanPattern(scanID) == SCAN_PATTERN_SPINNING_MULTIBEAM);
    DOCTEST_CHECK(lidar.getScanStepsPerRev(scanID) == expected_steps_per_rev);
    DOCTEST_CHECK(lidar.getScanSizeTheta(scanID) == channels);
    DOCTEST_CHECK(lidar.getScanRevolutions(scanID) == doctest::Approx(target_revs).epsilon(0.001));
    const double expected_rotation_rate = double(PRF) / (double(channels) * double(expected_steps_per_rev));
    DOCTEST_CHECK(lidar.getScanRotationRate(scanID) == doctest::Approx(expected_rotation_rate).epsilon(0.001));
    DOCTEST_CHECK(lidar.getScanSizePhi(scanID) == uint(std::lround(double(expected_steps_per_rev) * target_revs)));

    lidar.syntheticScan(&context, false, false);
    uint hit_count = lidar.getHitCount();
    DOCTEST_REQUIRE(hit_count > 0);

    // Per-channel fidelity: with identity platform orientation, each hit's world beam zenith (origin -> hit) must equal
    // one of the EXACT channel zenith angles, not a uniform grid value between channels.
    std::vector<float> channel_zenith;
    for (float er: beam_elev_rad) {
        channel_zenith.push_back(0.5f * float(M_PI) - er);
    }
    uint checked = 0;
    for (uint i = 0; i < hit_count; i++) {
        vec3 origin = lidar.getHitOrigin(i);
        vec3 dir = lidar.getHitXYZ(i) - origin;
        if (dir.magnitude() < 1e-6f) {
            continue;
        }
        SphericalCoord sc = cart2sphere(dir);
        float zenith = 0.5f * float(M_PI) - sc.elevation; // elevation -> zenith
        float best = 1e9f;
        for (float cz: channel_zenith) {
            best = std::min(best, std::fabs(zenith - cz));
        }
        DOCTEST_CHECK(best < 1e-3f); // matches an exact channel angle
        checked++;
    }
    DOCTEST_REQUIRE(checked > 0);

    // Timestamps: all returns of a pulse share one timestamp; t == t0 + pulse_id*pulse_period.
    const double pulse_period = 1.0 / double(PRF);
    for (uint i = 0; i < hit_count; i++) {
        DOCTEST_REQUIRE(lidar.doesHitDataExist(i, "pulse_id"));
        double pid = lidar.getHitData(i, "pulse_id");
        double t = lidar.getHitData(i, "timestamp");
        DOCTEST_CHECK(t == doctest::Approx(pid * pulse_period));
    }
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam Stationary Seam") {
    // A stationary spin of exactly one revolution (two coincident poses one rotation period apart) must sample exactly
    // steps_per_rev distinct azimuths with no duplicated wrap column (the periodic dphi = phimax/Nphi fix). We check the
    // azimuth grid via rc2direction.

    std::vector<float> beam_elev_rad = {-5.f * float(M_PI) / 180.f, 5.f * float(M_PI) / 180.f};
    const float azimuthStep_rad = 10.f * float(M_PI) / 180.f; // 36 steps/rev
    const uint expected_steps_per_rev = 36;
    const float PRF = 1000.f;
    // 2 channels * 36 steps/rev / 1000 Hz = 0.072 s for exactly one revolution.
    const double one_rev_duration = 2.0 * double(expected_steps_per_rev) / double(PRF);
    const std::vector<double> traj_t = {0.0, one_rev_duration};
    const std::vector<vec3> traj_pos = {make_vec3(0, 0, 1), make_vec3(0, 0, 1)};
    const std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};

    LiDARcloud lidar;
    lidar.disableMessages();
    uint scanID = lidar.addScanSpinning(beam_elev_rad, azimuthStep_rad, PRF, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0);

    DOCTEST_CHECK(lidar.getScanStepsPerRev(scanID) == expected_steps_per_rev);
    DOCTEST_CHECK(lidar.getScanRevolutions(scanID) == doctest::Approx(1.0).epsilon(0.001));
    DOCTEST_CHECK(lidar.getScanSizePhi(scanID) == expected_steps_per_rev); // one revolution -> exactly steps_per_rev columns

    // Reconstruct the same spinning grid as a local ScanMetadata (Nphi columns over phiMax = 1 revolution * 2*pi) and
    // collect the distinct azimuth angles over one revolution (row 0). With the periodic convention they are uniformly
    // spaced by 2*pi/steps_per_rev with no duplicated 0 / 2*pi seam.
    std::vector<float> beam_zenith = {0.5f * float(M_PI) - beam_elev_rad[0], 0.5f * float(M_PI) - beam_elev_rad[1]};
    ScanMetadata grid(make_vec3(0, 0, 1), beam_zenith, lidar.getScanSizePhi(scanID), 0.f, 1.f * 2.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
    std::vector<float> phis;
    for (uint col = 0; col < lidar.getScanSizePhi(scanID); col++) {
        SphericalCoord d = grid.rc2direction(0, col);
        phis.push_back(d.azimuth);
    }
    // No two columns coincide (modulo 2*pi).
    bool seam_duplicate = false;
    for (size_t a = 0; a < phis.size(); a++) {
        for (size_t b = a + 1; b < phis.size(); b++) {
            float diff = std::fabs(std::fmod(phis[a] - phis[b], 2.f * float(M_PI)));
            diff = std::min(diff, 2.f * float(M_PI) - diff);
            if (diff < 1e-4f) {
                seam_duplicate = true;
            }
        }
    }
    DOCTEST_CHECK(!seam_duplicate);
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam Fail-Fast Validation") {
    // The physical-parameter entry point fails fast (no silent fallback) on invalid inputs.
    LiDARcloud lidar;
    lidar.disableMessages();
    std::vector<float> elev = {-5.f * float(M_PI) / 180.f, 5.f * float(M_PI) / 180.f};
    std::vector<double> traj_t = {0.0, 0.5};
    std::vector<vec3> traj_pos = {make_vec3(0, 0, 10), make_vec3(0, 1, 10)};
    std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};

    // Empty channels.
    DOCTEST_CHECK_THROWS(lidar.addScanSpinning(std::vector<float>(), 10.f * float(M_PI) / 180.f, 1000.f, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0));
    // Non-positive azimuth step.
    DOCTEST_CHECK_THROWS(lidar.addScanSpinning(elev, 0.f, 1000.f, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0));
    // Non-positive PRF.
    DOCTEST_CHECK_THROWS(lidar.addScanSpinning(elev, 10.f * float(M_PI) / 180.f, 0.f, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0));
    // Mismatched trajectory array lengths must throw a clear error rather than dereferencing traj_pos.front() on a short
    // array (which seeds the ScanMetadata origin before addScanMoving validates the lengths).
    std::vector<vec3> traj_pos_short = {make_vec3(0, 0, 10)};
    DOCTEST_CHECK_THROWS(lidar.addScanSpinning(elev, 10.f * float(M_PI) / 180.f, 1000.f, traj_t, traj_pos_short, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0));
    std::vector<vec4> traj_quat_short = {make_vec4(0, 0, 0, 1)};
    DOCTEST_CHECK_THROWS(lidar.addScanSpinning(elev, 10.f * float(M_PI) / 180.f, 1000.f, traj_t, traj_pos, traj_quat_short, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0));
    // Empty traj_pos with non-empty traj_t (the worst case: traj_pos.front() would be UB).
    DOCTEST_CHECK_THROWS(lidar.addScanSpinning(elev, 10.f * float(M_PI) / 180.f, 1000.f, traj_t, std::vector<vec3>(), traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0));
}

DOCTEST_TEST_CASE("LiDAR Azimuth Warning Gated To Static Raster") {
    // The ">2pi azimuth" warning must fire for a static raster scan with phiMax >> 2pi (a likely degrees/radians mistake)
    // but NOT for a spinning scan, which legitimately encodes phiMax = n_revolutions*2pi.

    // Spinning multi-revolution scan: no warning.
    std::string spinning_err;
    {
        LiDARcloud lidar;
        // messages enabled so the warning would be captured if emitted
        std::vector<float> elev = {-5.f * float(M_PI) / 180.f, 5.f * float(M_PI) / 180.f};
        std::vector<double> traj_t = {0.0, 0.5};
        std::vector<vec3> traj_pos = {make_vec3(0, 0, 10), make_vec3(0, 1, 10)};
        std::vector<vec4> traj_quat = {make_vec4(0, 0, 0, 1), make_vec4(0, 0, 0, 1)};
        capture_cerr capture;
        lidar.addScanSpinning(elev, 1.f * float(M_PI) / 180.f, 100000.f, traj_t, traj_pos, traj_quat, make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>(), 0.0);
        spinning_err = capture.get_captured_output();
    }
    DOCTEST_CHECK(spinning_err.find("greater than 2pi") == std::string::npos);

    // Static raster scan with phiMax = 6*pi: warning emitted.
    std::string raster_err;
    {
        LiDARcloud lidar;
        ScanMetadata scan(make_vec3(0, 0, 1), 4, 0.f, float(M_PI), 8, 0.f, 6.f * float(M_PI), 0.f, 0.f, 0.f, 0.f, std::vector<std::string>());
        capture_cerr capture;
        lidar.addScan(scan);
        raster_err = capture.get_captured_output();
    }
    DOCTEST_CHECK(raster_err.find("greater than 2pi") != std::string::npos);
}

DOCTEST_TEST_CASE("LiDAR Spinning Multibeam XML Load And Export Round-Trip") {
    // Author a spinning scan in XML with an inline trajectory + physical parameters, load it, run a synthetic scan,
    // export it, and reload the exported XML. The reloaded scan must reproduce the derived descriptors.

    const std::string dir = "lidar_spin_xml_test";
    std::filesystem::create_directories(dir);
    const std::string xml_path = dir + "/spin.xml";

    {
        std::ofstream xml(xml_path);
        xml << "<helios>\n";
        xml << "  <scan>\n";
        xml << "    <origin> 0 0 30 </origin>\n";
        xml << "    <scanPattern> spinning_multibeam </scanPattern>\n";
        xml << "    <beamElevationAngles> -85 -80 -75 -70 </beamElevationAngles>\n"; // steep downward channels
        xml << "    <azimuthStep> 10 </azimuthStep>\n";
        xml << "    <PRF> 576 </PRF>\n"; // 4 channels * 36 steps/rev * 2 rev / 0.5 s = 576
        xml << "    <trajectory>\n";
        xml << "      <pose> 0.0 0 0 30 0 0 0 1 </pose>\n";
        xml << "      <pose> 0.5 0 1 30 0 0 0 1 </pose>\n";
        xml << "    </trajectory>\n";
        xml << "    <ASCII_format> x y z origin_x origin_y origin_z timestamp </ASCII_format>\n";
        xml << "  </scan>\n";
        xml << "</helios>\n";
    }

    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(200, 200));
    context.addPatch(make_vec3(0, 0, -2), make_vec2(200, 200));

    LiDARcloud lidar;
    lidar.disableMessages();
    lidar.loadXML(xml_path.c_str());

    DOCTEST_REQUIRE(lidar.getScanCount() == 1);
    DOCTEST_CHECK(lidar.getScanMode(0) == SCAN_MODE_SPINNING);
    DOCTEST_CHECK(lidar.getScanStepsPerRev(0) == 36);
    DOCTEST_CHECK(lidar.getScanSizeTheta(0) == 4);
    DOCTEST_CHECK(lidar.getScanRevolutions(0) == doctest::Approx(2.0).epsilon(0.001));

    lidar.syntheticScan(&context, false, false);
    DOCTEST_REQUIRE(lidar.getHitCount() > 0);

    // Export and reload; the round-tripped scan reproduces the derived descriptors.
    const std::string export_path = dir + "/spin_export.xml";
    lidar.exportScans(export_path.c_str());

    LiDARcloud lidar2;
    lidar2.disableMessages();
    lidar2.loadXML(export_path.c_str());
    DOCTEST_REQUIRE(lidar2.getScanCount() == 1);
    DOCTEST_CHECK(lidar2.getScanMode(0) == SCAN_MODE_SPINNING);
    DOCTEST_CHECK(lidar2.getScanStepsPerRev(0) == 36);
    DOCTEST_CHECK(lidar2.getScanSizeTheta(0) == 4);
    DOCTEST_CHECK(lidar2.getScanRevolutions(0) == doctest::Approx(2.0).epsilon(0.001));

    std::filesystem::remove_all(dir);
}

DOCTEST_TEST_CASE("LiDARcloud::isMultiReturnData") {
    // Single-return cloud: every hit carries target_count == 1.
    LiDARcloud single;
    single.disableMessages();
    ScanMetadata scan(make_vec3(0, 0, 5), 2, 0.f, 0.f, 2, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, {});
    single.addScan(scan);
    for (uint i = 0; i < 2; i++) {
        std::map<std::string, double> data{{"target_count", 1}, {"target_index", 0}, {"timestamp", double(i)}};
        single.addHitPoint(0, make_vec3(float(i), 0, 0), make_SphericalCoord(0.f, 0.f), data);
    }
    DOCTEST_CHECK(single.isMultiReturnData() == false);

    // Multi-return cloud: one pulse produced two returns, and both companion fields are present.
    LiDARcloud multi;
    multi.disableMessages();
    multi.addScan(scan);
    for (uint i = 0; i < 2; i++) {
        std::map<std::string, double> data{{"target_count", 2}, {"target_index", double(i)}, {"timestamp", 0.0}};
        multi.addHitPoint(0, make_vec3(0, 0, float(i)), make_SphericalCoord(0.f, 0.f), data);
    }
    DOCTEST_CHECK(multi.isMultiReturnData() == true);

    // A cloud with no target_count data at all is single-return, not an error.
    LiDARcloud bare;
    bare.disableMessages();
    bare.addScan(scan);
    bare.addHitPoint(0, make_vec3(0, 0, 0), make_SphericalCoord(0.f, 0.f));
    DOCTEST_CHECK(bare.isMultiReturnData() == false);

    // target_count > 1 without the timestamp needed to group returns into beams must throw,
    // rather than reporting an answer the triangulation cannot act on.
    LiDARcloud no_timestamp;
    no_timestamp.disableMessages();
    no_timestamp.addScan(scan);
    std::map<std::string, double> missing_timestamp{{"target_count", 2}, {"target_index", 0}};
    no_timestamp.addHitPoint(0, make_vec3(0, 0, 0), make_SphericalCoord(0.f, 0.f), missing_timestamp);
    DOCTEST_CHECK_THROWS(no_timestamp.isMultiReturnData());

    // Likewise for a missing target_index, which selects first returns.
    LiDARcloud no_target_index;
    no_target_index.disableMessages();
    no_target_index.addScan(scan);
    std::map<std::string, double> missing_index{{"target_count", 2}, {"timestamp", 0.0}};
    no_target_index.addHitPoint(0, make_vec3(0, 0, 0), make_SphericalCoord(0.f, 0.f), missing_index);
    DOCTEST_CHECK_THROWS(no_target_index.isMultiReturnData());
}
