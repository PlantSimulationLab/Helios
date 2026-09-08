#include "SyntheticAnnotation.h"
#include "json.hpp"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

TEST_CASE("SyntheticAnnotation initialization") {
    Context context_test;
    // Test that construction doesn't throw and object can be created
    DOCTEST_CHECK_NOTHROW(SyntheticAnnotation test_instance(&context_test));
}

TEST_CASE("SyntheticAnnotation setBackgroundColor") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that setBackgroundColor doesn't throw
    DOCTEST_CHECK_NOTHROW(test_instance.setBackgroundColor(helios::make_RGBcolor(0.5f, 0.5f, 0.5f)));
    DOCTEST_CHECK_NOTHROW(test_instance.setBackgroundColor(helios::make_RGBcolor(0.9f, 0.9f, 0.9f)));
}

TEST_CASE("SyntheticAnnotation setWindowSize") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that setWindowSize doesn't throw with valid values
    DOCTEST_CHECK_NOTHROW(test_instance.setWindowSize(1280, 720));
    DOCTEST_CHECK_NOTHROW(test_instance.setWindowSize(1920, 1080));
}

TEST_CASE("SyntheticAnnotation setCameraPosition") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    helios::vec3 cam_pos(1, 1, 1);
    helios::vec3 cam_lookat(0, 0, 0);

    // Test that setCameraPosition doesn't throw
    DOCTEST_CHECK_NOTHROW(test_instance.setCameraPosition(cam_pos, cam_lookat));

    // Test vector version
    std::vector<helios::vec3> positions = {cam_pos};
    std::vector<helios::vec3> lookats = {cam_lookat};
    DOCTEST_CHECK_NOTHROW(test_instance.setCameraPosition(positions, lookats));
}

TEST_CASE("SyntheticAnnotation object detection enable/disable") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that enable/disable methods don't throw
    DOCTEST_CHECK_NOTHROW(test_instance.enableObjectDetection());
    DOCTEST_CHECK_NOTHROW(test_instance.disableObjectDetection());
}

TEST_CASE("SyntheticAnnotation semantic segmentation enable/disable") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that enable/disable methods don't throw
    DOCTEST_CHECK_NOTHROW(test_instance.enableSemanticSegmentation());
    DOCTEST_CHECK_NOTHROW(test_instance.disableSemanticSegmentation());
}

TEST_CASE("SyntheticAnnotation instance segmentation enable/disable") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that enable/disable methods don't throw
    DOCTEST_CHECK_NOTHROW(test_instance.enableInstanceSegmentation());
    DOCTEST_CHECK_NOTHROW(test_instance.disableInstanceSegmentation());
}

TEST_CASE("SyntheticAnnotation labelPrimitives single UUID") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    uint sampleUUID = 12345;
    // Test that labelPrimitives doesn't throw with valid inputs
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives(sampleUUID, "test_label"));
}

TEST_CASE("SyntheticAnnotation labelPrimitives const char label") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that labelPrimitives doesn't throw with valid label
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives("test_label"));
}

TEST_CASE("SyntheticAnnotation labelPrimitives vector") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    std::vector<uint> uuids = {123, 456, 789};
    // Test that labelPrimitives doesn't throw with vector input
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives(uuids, "vector_test_label"));
}

TEST_CASE("SyntheticAnnotation labelPrimitives vector groups") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    std::vector<std::vector<uint>> uuid_groups = {{123, 456}, {789, 1011}};
    // Test that labelPrimitives doesn't throw with nested vector input
    DOCTEST_CHECK_NOTHROW(test_instance.labelPrimitives(uuid_groups, "group_test_label"));
}

TEST_CASE("SyntheticAnnotation labelUnlabeledPrimitives") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // Test that labelUnlabeledPrimitives doesn't throw
    DOCTEST_CHECK_NOTHROW(test_instance.labelUnlabeledPrimitives("unlabeled_test"));
}

TEST_CASE("SyntheticAnnotation addSkyDome") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    // An existing texture file is accepted. Tests run from the build directory, where the
    // visualizer's textures are copied, so this path resolves.
    if (std::filesystem::exists("plugins/visualizer/textures/SkyDome_clouds.jpg")) {
        DOCTEST_CHECK_NOTHROW(test_instance.addSkyDome("plugins/visualizer/textures/SkyDome_clouds.jpg"));
    }

    // A missing file is a hard error rather than being silently ignored: addSkyDome() used to be
    // an empty stub, so a mistyped path produced no diagnostic and no sky dome.
    DOCTEST_CHECK_THROWS(test_instance.addSkyDome("plugins/visualizer/textures/does_not_exist.jpg"));
}

TEST_CASE("SyntheticAnnotation labelUnlabeledPrimitives labels the remainder") {
    // Regression test: labelUnlabeledPrimitives() was an empty stub -- it contained a loop with
    // no body -- so it silently did nothing. Primitives the user expected to be labeled were
    // left out of every annotation.

    Context context_test;
    uint labeled = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    uint unlabeled_a = context_test.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));
    uint unlabeled_b = context_test.addPatch(make_vec3(4, 0, 0), make_vec2(1, 1));

    SyntheticAnnotation test_instance(&context_test);
    test_instance.labelPrimitives(std::vector<uint>{labeled}, "plant");
    test_instance.labelUnlabeledPrimitives("background");

    // Every primitive must now carry an object label...
    DOCTEST_CHECK(context_test.doesPrimitiveDataExist(labeled, "syntheticannotation_labelID"));
    DOCTEST_CHECK(context_test.doesPrimitiveDataExist(unlabeled_a, "syntheticannotation_labelID"));
    DOCTEST_CHECK(context_test.doesPrimitiveDataExist(unlabeled_b, "syntheticannotation_labelID"));

    // ...and the two formerly-unlabeled primitives must be distinct objects, not one group.
    uint id_a = 0, id_b = 0, id_labeled = 0;
    context_test.getPrimitiveData(unlabeled_a, "syntheticannotation_labelID", id_a);
    context_test.getPrimitiveData(unlabeled_b, "syntheticannotation_labelID", id_b);
    context_test.getPrimitiveData(labeled, "syntheticannotation_labelID", id_labeled);
    DOCTEST_CHECK(id_a != id_b);
    DOCTEST_CHECK(id_a != id_labeled);
    DOCTEST_CHECK(id_b != id_labeled);
}

TEST_CASE("SyntheticAnnotation setCameraFieldOfView") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    DOCTEST_CHECK_NOTHROW(test_instance.setCameraFieldOfView(44.5f));
    DOCTEST_CHECK_NOTHROW(test_instance.setCameraFieldOfView(179.f));

    // A field of view outside (0,180) has no perspective projection, so it is rejected here rather
    // than producing a silently degenerate projection matrix inside the Visualizer.
    DOCTEST_CHECK_THROWS(test_instance.setCameraFieldOfView(0.f));
    DOCTEST_CHECK_THROWS(test_instance.setCameraFieldOfView(-10.f));
    DOCTEST_CHECK_THROWS(test_instance.setCameraFieldOfView(180.f));
    DOCTEST_CHECK_THROWS(test_instance.setCameraFieldOfView(200.f));
}

TEST_CASE("SyntheticAnnotation RGB rendering enable/disable") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    DOCTEST_CHECK_NOTHROW(test_instance.disableRGBRendering());
    DOCTEST_CHECK_NOTHROW(test_instance.enableRGBRendering());
}

TEST_CASE("SyntheticAnnotation setMinimumLabelPixels") {
    Context context_test;
    SyntheticAnnotation test_instance(&context_test);

    DOCTEST_CHECK_NOTHROW(test_instance.setMinimumLabelPixels(3));
    DOCTEST_CHECK_NOTHROW(test_instance.setMinimumLabelPixels(500));

    // A negative threshold is meaningless and is rejected rather than silently accepted.
    DOCTEST_CHECK_THROWS(test_instance.setMinimumLabelPixels(-1));

    // A threshold below three is rejected too: an object covering one or two pixels has no
    // traceable outline, so it would pass this filter only to be dropped by the mask writer with
    // no diagnostic. Rejecting it here turns a silent loss of annotations into an explicit error.
    DOCTEST_CHECK_THROWS(test_instance.setMinimumLabelPixels(0));
    DOCTEST_CHECK_THROWS(test_instance.setMinimumLabelPixels(2));
}


// ---------------------------------------------------------------------------
// Regression tests for render() output correctness.
//
// render() rasterizes through a Visualizer, so these tests only run where an
// OpenGL context can be created. They exercise the annotation-writing code paths, which
// were previously untested: every other test in this file is a CHECK_NOTHROW on
// a setter.
// ---------------------------------------------------------------------------

//! Returns true if an OpenGL context usable by render() can be created on this machine
/**
 * render() rasterizes through a Visualizer, so the tests that call it cannot run where no OpenGL
 * context can be created at all. Unlike the visualizer plug-in's tests, these are not skipped by
 * name in CI (`--visbuildonly` skips only executables whose name contains "visualizer"), so this
 * guard is what keeps them from failing on a runner with no GL driver.
 *
 * The check is a probe rather than a platform assumption: it constructs a Visualizer exactly as
 * render() does and reports whether that succeeded. Inspecting DISPLAY/WAYLAND_DISPLAY cannot
 * answer the question on macOS or Windows, which never set either variable, and assuming a context
 * is always available there is what broke the macOS, Windows and Windows GPU runners in v1.3.82 --
 * all three create no OpenGL context at all. Probing also covers the case where a display is
 * advertised but the driver still cannot produce a context.
 *
 * The probe constructs the Visualizer with the same arguments render() uses. render() now draws
 * both of its passes headless, so the probe is headless too; probing a windowed Visualizer would
 * report "no display" on batch machines where render() in fact works.
 *
 * The result is computed once and cached: each probe builds and tears down a real GL context, and
 * repeating that for every test case is slow and needlessly re-enters GLFW init/terminate.
 */
static bool syntheticAnnotationDisplayAvailable() {
    static const bool context_available = []() {
        try {
            // Same arguments as the RGB pass in SyntheticAnnotation::render(), at minimum size:
            // the probe only needs the context to come up, not to draw anything.
            Visualizer probe(16, 16, 8, false, true);
            probe.disableMessages();
        } catch (const std::exception &e) {
            // Report why, so that a runner silently skipping every render test is distinguishable
            // from one that has no graphics at all.
            std::cerr << "NOTE: SyntheticAnnotation render tests skipped -- no OpenGL context: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "NOTE: SyntheticAnnotation render tests skipped -- no OpenGL context." << std::endl;
            return false;
        }
        return true;
    }();
    return context_available;
}

//! Build a scene with two well-separated patches so each labeled object is distinguishable.
/**
 * The patches are rotated to face the -y camera used by these tests. An unrotated patch is
 * horizontal, so it would be viewed edge-on and cover no pixels at all -- every mask would come
 * out empty and the tests would compare nothing.
 */
static std::vector<std::vector<uint>> buildTwoObjectScene(Context &context) {
    std::vector<std::vector<uint>> groups;
    groups.push_back({context.addPatch(make_vec3(-0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))});
    groups.push_back({context.addPatch(make_vec3(0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))});
    return groups;
}

TEST_CASE("SyntheticAnnotation instance segmentation writes one COCO annotation per object") {
    // Instance masks are written as a single COCO JSON per view, with one annotation per visible
    // object. The file must be readable by the visualizer's own COCO parser, which is what makes
    // it displayable with Visualizer::displayImageWithSegmentationMasks().

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups = buildTwoObjectScene(context);

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.disableObjectDetection();
    annotation.disableSemanticSegmentation();
    annotation.enableInstanceSegmentation();
    annotation.labelPrimitives(groups, "patch");

    std::string outdir = "./sa_test_instanceseg/";
    std::filesystem::remove_all(outdir);

    annotation.render(outdir.c_str());

    const std::string image_file = outdir + "view00000/RGB_rendering.jpeg";
    const std::string json_file = outdir + "view00000/instances.json";
    const bool json_exists = std::filesystem::exists(json_file);

    // Parse with the visualizer's reader, so the test fails if anything about the document would
    // stop it being displayed.
    std::vector<Visualizer::SegmentationMask> masks;
    if (json_exists) {
        masks = Visualizer::readSegmentationMaskFile(json_file, image_file);
    }

    std::filesystem::remove_all(outdir);

    DOCTEST_REQUIRE(json_exists);

    // The two patches are separate objects of the same label, so they must remain two annotations
    // rather than being merged into one region.
    DOCTEST_CHECK(masks.size() == 2);
    for (const Visualizer::SegmentationMask &mask: masks) {
        DOCTEST_CHECK(mask.class_name == "patch");
        DOCTEST_CHECK(mask.polygons.size() >= 1);
        DOCTEST_CHECK(mask.image_size.x == 200.f);
        DOCTEST_CHECK(mask.image_size.y == 200.f);
    }
}

TEST_CASE("SyntheticAnnotation XML enable/disable flags are not inverted") {
    // Regression test: the global-data flag parsing used std::string::compare() as
    // though it returned a boolean. compare() returns 0 on a match, so "enabled"
    // took the falsy branch and "disabled" took the truthy one -- inverting every
    // flag. Setting the flag to "disabled" here must actually disable the output.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups = buildTwoObjectScene(context);

    // Request object detection ON and the two segmentation outputs OFF via global data.
    context.setGlobalData("object_detection", std::string("enabled"));
    context.setGlobalData("semantic_segmentation", std::string("disabled"));
    context.setGlobalData("instance_segmentation", std::string("disabled"));

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.labelPrimitives(groups, "patch");

    std::string outdir = "./sa_test_xmlflags/";
    std::filesystem::remove_all(outdir);

    annotation.render(outdir.c_str());

    std::string viewdir = outdir + "view00000/";
    bool wrote_rectangular = false;
    bool wrote_semantic = false;
    bool wrote_instance = false;
    if (std::filesystem::exists(viewdir)) {
        for (const auto &entry: std::filesystem::directory_iterator(viewdir)) {
            const std::string name = entry.path().filename().string();
            if (name == "RGB_rendering.txt") {
                wrote_rectangular = true;
            } else if (name.rfind("semantic_segmentation", 0) == 0) {
                wrote_semantic = true;
            } else if (name == "instances.json") {
                wrote_instance = true;
            }
        }
    }

    std::filesystem::remove_all(outdir);

    DOCTEST_CHECK(wrote_rectangular); // "enabled" must enable
    DOCTEST_CHECK(!wrote_semantic); // "disabled" must disable
    DOCTEST_CHECK(!wrote_instance); // "disabled" must disable
}

TEST_CASE("SyntheticAnnotation semantic segmentation mask matches framebuffer dimensions") {
    // Regression test for the pixel indexing in the semantic-segmentation and
    // instance-segmentation writers. The index was computed as
    //     t = 3*((W-1)*j + i + j)
    // which simplifies to 3*(W*j + i) with j running from H down to 1. On the first
    // iteration that addresses 3*(W*H + W - 1), a full row past the end of a
    // 3*W*H-element buffer, and source row 0 was never read at all. The mask was
    // therefore shifted by one row with its top row sourced from out-of-bounds heap
    // memory. The index is now 3*((H-1-j)*W + i) over j in [0,H).
    //
    // This test pins the emitted mask to exactly framebufferH rows of framebufferW
    // columns. Note the framebuffer is larger than the requested window on a
    // high-DPI display, so the expected size is queried rather than assumed.
    //
    // The mask content is asserted separately by the tests that check label IDs and
    // orientation; this one is only concerned with the mask's dimensions.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups;
    groups.push_back({context.addPatch(make_vec3(0, 0, 0), make_vec2(1.0f, 1.0f), make_SphericalCoord(0.5f * M_PI, 0.f))});

    const uint window_size = 200;

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(window_size, window_size);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.disableObjectDetection();
    annotation.disableInstanceSegmentation();
    annotation.enableSemanticSegmentation();
    annotation.labelPrimitives(groups, "patch");

    std::string outdir = "./sa_test_semseg/";
    std::filesystem::remove_all(outdir);

    annotation.render(outdir.c_str());

    // Determine the framebuffer size the same way render() does. The ID pass renders headless
    // (see render()), which draws to an offscreen framebuffer at exactly the requested size
    // rather than to a window that a high-DPI display would scale up.
    uint framebufferW = 0, framebufferH = 0;
    {
        Visualizer vis(window_size, window_size, 0, false, true);
        vis.disableMessages();
        vis.getFramebufferSize(framebufferW, framebufferH);
        vis.closeWindow();
    }

    std::string maskfile = outdir + "view00000/semantic_segmentation.txt";
    size_t nrows = 0;
    std::vector<size_t> row_widths;
    if (std::filesystem::exists(maskfile)) {
        std::ifstream file(maskfile);
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            std::istringstream iss(line);
            size_t count = 0;
            int value;
            while (iss >> value) {
                count++;
            }
            row_widths.push_back(count);
            nrows++;
        }
    }

    std::filesystem::remove_all(outdir);

    DOCTEST_REQUIRE(nrows > 0);
    DOCTEST_REQUIRE(framebufferH > 0);

    // Exactly one output row per framebuffer row -- no row dropped, none duplicated.
    DOCTEST_CHECK(nrows == size_t(framebufferH));

    bool all_rows_correct_width = true;
    for (size_t r = 0; r < row_widths.size(); r++) {
        if (row_widths.at(r) != size_t(framebufferW)) {
            all_rows_correct_width = false;
        }
    }
    DOCTEST_CHECK(all_rows_correct_width);
}

TEST_CASE("SyntheticAnnotation pixel row indexing stays in bounds") {
    // Regression test for the pixel index used by the segmentation mask writers.
    //
    // The original expression was
    //     t = 3*((W-1)*j + i + j)   with j running from H down to 1,
    // which simplifies to 3*(W*j + i). At j==H that is 3*(W*H + i) -- up to a full
    // row past the end of the 3*W*H-element pixel buffer -- while source row 0 was
    // never visited. The corrected expression is 3*((H-1-j)*W + i) over j in [0,H).
    //
    // This mirrors the arithmetic directly (it is a private detail of render(), so
    // it cannot be called here) and asserts the two properties that matter: every
    // offset addresses a valid RGB triple, and every source row is visited exactly
    // once. It fails loudly against the old expression.

    const int W = 37; // deliberately non-square and not a power of two
    const int H = 23;
    const size_t buffer_size = size_t(3) * size_t(W) * size_t(H);

    std::vector<int> row_visit_count(H, 0);
    bool all_offsets_in_bounds = true;

    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            const size_t t = 3 * size_t((H - 1 - j) * W + i);

            // Must be able to read t, t+1 and t+2.
            if (t + 2 >= buffer_size) {
                all_offsets_in_bounds = false;
                continue;
            }
            if (i == 0) {
                row_visit_count.at(size_t((t / 3) / W))++;
            }
        }
    }

    DOCTEST_CHECK(all_offsets_in_bounds);

    // Every source row visited exactly once -- no row skipped, none read twice.
    bool every_row_visited_once = true;
    for (int r = 0; r < H; r++) {
        if (row_visit_count.at(size_t(r)) != 1) {
            every_row_visited_once = false;
        }
    }
    DOCTEST_CHECK(every_row_visited_once);
}

TEST_CASE("SyntheticAnnotation does not collide with existing object_label data") {
    // "object_label" is an established convention across Helios for a *string* naming the organ or
    // material a primitive belongs to: PlantArchitecture sets "leaf"/"petiole"/"peduncle",
    // ProjectBuilder filters on those names, and Context::loadOBJ records material group names
    // there. This plug-in used the same key for its own uint label ID, so annotating any such scene
    // aborted with a primitive-data type conflict -- and had the types agreed, the appearance guard
    // would have deleted the scene's own labels on the way out.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups = buildTwoObjectScene(context);
    for (const std::vector<uint> &group: groups) {
        context.setPrimitiveData(group, "object_label", "leaf");
    }

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.disableRGBRendering();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.enableObjectDetection();
    annotation.disableSemanticSegmentation();
    annotation.disableInstanceSegmentation();
    annotation.labelPrimitives(groups, "patch");

    const std::string outdir = "./sa_test_labelcollide/";
    std::filesystem::remove_all(outdir);
    DOCTEST_CHECK_NOTHROW(annotation.render(outdir.c_str()));

    // The scene's own labels survive the round trip, with their original type and value.
    for (const std::vector<uint> &group: groups) {
        for (uint UUID: group) {
            DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID, "object_label"));
            std::string organ;
            context.getPrimitiveData(UUID, "object_label", organ);
            DOCTEST_CHECK(organ == "leaf");
        }
    }

    std::filesystem::remove_all(outdir);
}

TEST_CASE("SyntheticAnnotation camera field of view scales the rendered geometry") {
    // The field of view is what makes a rasterized annotation pass geometrically comparable to a
    // camera specified elsewhere (e.g. a RadiationModel camera given by its HFOV). Without it the
    // projection was pinned at the Visualizer's 45 degree default, so bounding boxes came out at
    // the wrong scale no matter how the camera was placed.
    //
    // Halving the field of view must magnify the subject: the same patch at the same distance has
    // to occupy a larger fraction of the frame.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    auto box_area_fraction = [](float fov, const std::string &outdir) -> float {
        Context context;
        context.addPatch(make_vec3(0, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f));

        SyntheticAnnotation annotation(&context);
        annotation.disableMessages();
        annotation.disableRGBRendering();
        annotation.setWindowSize(200, 200);
        annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
        annotation.setCameraFieldOfView(fov);
        annotation.enableObjectDetection();
        annotation.disableSemanticSegmentation();
        annotation.disableInstanceSegmentation();
        annotation.labelPrimitives(context.getAllUUIDs(), "patch");

        std::filesystem::remove_all(outdir);
        annotation.render(outdir.c_str());

        // YOLO bounding boxes: class cx cy w h, normalized to the frame.
        std::ifstream file(outdir + "view00000/RGB_rendering.txt");
        int cls;
        float cx, cy, w, h;
        float area = 0.f;
        while (file >> cls >> cx >> cy >> w >> h) {
            area = std::max(area, w * h);
        }
        return area;
    };

    const float area_wide = box_area_fraction(60.f, "./sa_test_fov_wide/");
    const float area_narrow = box_area_fraction(30.f, "./sa_test_fov_narrow/");

    DOCTEST_CHECK(area_wide > 0.f);
    DOCTEST_CHECK(area_narrow > 0.f);

    // The patch subtends a fixed angle, so its width in the frame scales as tan(FOV/2) and the
    // area as its square. Halving 60 deg to 30 deg gives tan(30)/tan(15) = 2.155 in width, so
    // roughly 4.6x in area. The tolerance is wide because the box is quantized to 200 pixels.
    const float ratio = area_narrow / area_wide;
    DOCTEST_CHECK(ratio == doctest::Approx(4.64f).epsilon(0.15));

    std::filesystem::remove_all("./sa_test_fov_wide/");
    std::filesystem::remove_all("./sa_test_fov_narrow/");
}

TEST_CASE("SyntheticAnnotation RGB rendering can be skipped") {
    // The RGB pass rasterizes the whole scene a second time only to produce a preview image. It
    // dominates the cost of a large scene, so a caller that wants only annotations must be able to
    // turn it off -- and turning it off must not disturb the annotations themselves.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups = buildTwoObjectScene(context);

    auto run = [&](bool rgb_enabled, const std::string &outdir) {
        SyntheticAnnotation annotation(&context);
        annotation.disableMessages();
        if (rgb_enabled) {
            annotation.enableRGBRendering();
        } else {
            annotation.disableRGBRendering();
        }
        annotation.setWindowSize(200, 200);
        annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
        annotation.enableObjectDetection();
        annotation.disableSemanticSegmentation();
        annotation.disableInstanceSegmentation();
        annotation.labelPrimitives(groups, "patch");

        std::filesystem::remove_all(outdir);
        annotation.render(outdir.c_str());
    };

    run(true, "./sa_test_rgb_on/");
    run(false, "./sa_test_rgb_off/");

    DOCTEST_CHECK(std::filesystem::exists("./sa_test_rgb_on/view00000/RGB_rendering.jpeg"));
    DOCTEST_CHECK(!std::filesystem::exists("./sa_test_rgb_off/view00000/RGB_rendering.jpeg"));

    // The annotations are unaffected by whether the preview was written.
    auto read_labels = [](const std::string &path) {
        std::vector<std::string> lines;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        return lines;
    };
    const std::vector<std::string> labels_on = read_labels("./sa_test_rgb_on/view00000/RGB_rendering.txt");
    const std::vector<std::string> labels_off = read_labels("./sa_test_rgb_off/view00000/RGB_rendering.txt");
    DOCTEST_CHECK(labels_on.size() == 2);
    DOCTEST_CHECK(labels_on == labels_off);

    std::filesystem::remove_all("./sa_test_rgb_on/");
    std::filesystem::remove_all("./sa_test_rgb_off/");
}

TEST_CASE("SyntheticAnnotation label IDs survive the render round trip") {
    // Regression test for the ID encoding, which is the mechanism the whole plug-in rests on:
    // each label ID is encoded as an RGB color, rendered, and decoded back from the pixels.
    //
    // Two defects broke it. (1) The fragment shader multiplied vertex-interpolated colors by a
    // hardcoded 1.5 before the lighting branch, so an ID written as color byte 1 came back as 2
    // and any byte above 170 saturated to 255 -- making the encoding non-invertible. render()
    // now calls Visualizer::enableExactColorMode(). (2) The windowed pixel readback guesses
    // whether the front or back buffer holds the current frame from a nine-pixel content score;
    // against a white background the guess lands on the stale buffer and every mask came back as
    // a copy of the previously rendered object. render() now renders the ID pass headless.
    //
    // Two separated patches are labeled as distinct objects, so the combined pixel ID image must
    // contain exactly the two assigned IDs.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    // Patches are rotated to face the -y camera; an unrotated patch is horizontal and would be
    // viewed edge-on, covering no pixels at all.
    std::vector<std::vector<uint>> groups;
    groups.push_back({context.addPatch(make_vec3(-0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))});
    groups.push_back({context.addPatch(make_vec3(0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))});

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.disableObjectDetection();
    annotation.disableSemanticSegmentation();
    annotation.disableInstanceSegmentation();
    annotation.labelPrimitives(groups, "patch");

    std::string outdir = "./sa_test_idroundtrip/";
    std::filesystem::remove_all(outdir);

    annotation.render(outdir.c_str());

    // Count how many pixels carry each decoded ID.
    std::map<int, size_t> id_counts;
    std::string idfile = outdir + "view00000/pixelID_combined.txt";
    if (std::filesystem::exists(idfile)) {
        std::ifstream file(idfile);
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int value;
            while (iss >> value) {
                id_counts[value]++;
            }
        }
    }

    std::filesystem::remove_all(outdir);

    // labelPrimitives() assigns IDs starting at 1, so the two groups are IDs 1 and 2. Both must
    // appear, and with a substantial number of pixels rather than a stray anti-aliased few.
    DOCTEST_CHECK(id_counts[1] > 100);
    DOCTEST_CHECK(id_counts[2] > 100);

    // The two patches are the same size and equidistant from the camera, so their pixel counts
    // must match. Under the stale-buffer defect one object's mask was a copy of the other's.
    if (id_counts[1] > 0 && id_counts[2] > 0) {
        const double ratio = double(id_counts[1]) / double(id_counts[2]);
        DOCTEST_CHECK(ratio > 0.9);
        DOCTEST_CHECK(ratio < 1.1);
    }

    // No ID beyond the two that were assigned should appear (white background is 16777215).
    bool only_expected_ids = true;
    for (const auto &entry: id_counts) {
        if (entry.first != 0 && entry.first != 1 && entry.first != 2 && entry.first != 16777215 && entry.second > 100) {
            only_expected_ids = false;
        }
    }
    DOCTEST_CHECK(only_expected_ids);
}

TEST_CASE("SyntheticAnnotation semantic segmentation writes one multi-class mask") {
    // The semantic mask is a single multi-class image covering all labels, not one file per
    // label -- there is no per-label uniquifier in the filename, the mask is accumulated across
    // labels, and semantic_segmentation_ID_mapping.txt records the class index of each label
    // name. This pins that contract down: every label must appear in the one mask, with its
    // pixels where that label's geometry actually is.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    // One patch per label, on opposite sides of the frame so each class's pixels are
    // identifiable by position alone.
    std::vector<uint> left = {context.addPatch(make_vec3(-0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))};
    std::vector<uint> right = {context.addPatch(make_vec3(0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))};

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.disableObjectDetection();
    annotation.disableInstanceSegmentation();
    annotation.enableSemanticSegmentation();
    annotation.labelPrimitives(left, "leftpatch");
    annotation.labelPrimitives(right, "rightpatch");

    std::string outdir = "./sa_test_multiclass/";
    std::filesystem::remove_all(outdir);

    annotation.render(outdir.c_str());

    // Read the mask, recording for each class value how many pixels it covers and how many of
    // those are in the left half of the image.
    std::map<int, size_t> class_counts;
    std::map<int, size_t> class_in_left_half;
    std::string maskfile = outdir + "view00000/semantic_segmentation.txt";
    if (std::filesystem::exists(maskfile)) {
        std::ifstream file(maskfile);
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<int> row;
            int value;
            while (iss >> value) {
                row.push_back(value);
            }
            for (size_t column = 0; column < row.size(); column++) {
                class_counts[row.at(column)]++;
                if (column < row.size() / 2) {
                    class_in_left_half[row.at(column)]++;
                }
            }
        }
    }

    // The mapping file assigns a class index to each label name.
    size_t mapping_entries = 0;
    std::string mapfile = outdir + "view00000/semantic_segmentation_ID_mapping.txt";
    if (std::filesystem::exists(mapfile)) {
        std::ifstream file(mapfile);
        std::string line;
        std::getline(file, line); // header: "Element Label"
        while (std::getline(file, line)) {
            if (!line.empty()) {
                mapping_entries++;
            }
        }
    }

    std::filesystem::remove_all(outdir);

    // Both labels recorded in the mapping, and both present in the single mask.
    DOCTEST_CHECK(mapping_entries == 2);
    DOCTEST_CHECK(class_counts[1] > 100);
    DOCTEST_CHECK(class_counts[2] > 100);

    // The two patches are the same size and equidistant from the camera, so the classes cover
    // comparable areas.
    if (class_counts[1] > 0 && class_counts[2] > 0) {
        const double ratio = double(class_counts[1]) / double(class_counts[2]);
        DOCTEST_CHECK(ratio > 0.9);
        DOCTEST_CHECK(ratio < 1.1);
    }

    // Each class sits on its own side of the image: labelIDs is keyed by label name, so
    // "leftpatch" sorts before "rightpatch" and takes class index 1.
    DOCTEST_CHECK(class_in_left_half[1] == class_counts[1]);
    DOCTEST_CHECK(class_in_left_half[2] == 0);
}

TEST_CASE("SyntheticAnnotation render restores primitive appearance and clears label data") {
    // Regression test: render() recolors every primitive with its label's ID code and restored
    // the colors in a plain loop at the end of the function, so any error thrown partway through
    // left the caller's whole scene painted in ID codes. It also never removed the label-ID
    // primitive data it added, leaking internal state into the user's Context.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups = buildTwoObjectScene(context);
    const uint UUID = groups.at(0).at(0);

    const RGBcolor original_color = make_RGBcolor(0.25f, 0.5f, 0.75f);
    context.setPrimitiveColor(UUID, original_color);

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.disableObjectDetection();
    annotation.disableSemanticSegmentation();
    annotation.disableInstanceSegmentation();
    annotation.labelPrimitives(groups, "patch");

    std::string outdir = "./sa_test_restore/";
    std::filesystem::remove_all(outdir);
    annotation.render(outdir.c_str());
    std::filesystem::remove_all(outdir);

    // The user's color must survive the render.
    const RGBcolor after = context.getPrimitiveColor(UUID);
    DOCTEST_CHECK(after.r == doctest::Approx(original_color.r));
    DOCTEST_CHECK(after.g == doctest::Approx(original_color.g));
    DOCTEST_CHECK(after.b == doctest::Approx(original_color.b));

    // The internal label data must not be left behind in the user's Context.
    DOCTEST_CHECK(!context.doesPrimitiveDataExist(UUID, "syntheticannotation_labelID"));
}

TEST_CASE("SyntheticAnnotation render is repeatable") {
    // render() clears the "syntheticannotation_labelID" data it sets, so the coloring pass must re-establish it
    // rather than assume it survived. Otherwise a second render() would treat every primitive as
    // unlabeled, color them all white, and silently write blank annotations.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups = buildTwoObjectScene(context);

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.disableObjectDetection();
    annotation.disableSemanticSegmentation();
    annotation.disableInstanceSegmentation();
    annotation.labelPrimitives(groups, "patch");

    auto count_labeled_pixels = [](const std::string &dir) {
        std::map<int, size_t> counts;
        std::string idfile = dir + "view00000/pixelID_combined.txt";
        if (std::filesystem::exists(idfile)) {
            std::ifstream file(idfile);
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int value;
                while (iss >> value) {
                    counts[value]++;
                }
            }
        }
        return counts;
    };

    std::string dir1 = "./sa_test_repeat1/";
    std::string dir2 = "./sa_test_repeat2/";
    std::filesystem::remove_all(dir1);
    std::filesystem::remove_all(dir2);

    annotation.render(dir1.c_str());
    std::map<int, size_t> first = count_labeled_pixels(dir1);

    annotation.render(dir2.c_str());
    std::map<int, size_t> second = count_labeled_pixels(dir2);

    std::filesystem::remove_all(dir1);
    std::filesystem::remove_all(dir2);

    // The second render must produce the same IDs over the same areas as the first.
    DOCTEST_CHECK(first[1] > 100);
    DOCTEST_CHECK(first[2] > 100);
    DOCTEST_CHECK(second[1] == first[1]);
    DOCTEST_CHECK(second[2] == first[2]);
}

TEST_CASE("SyntheticAnnotation writes ID mapping and per-class bounding boxes") {
    // Regression test: ID_mapping.txt was opened and closed with its only write commented out, so
    // every run produced a zero-byte file and the integer IDs in the other outputs could not be
    // traced back to a label. Separately, the bounding-box annotations hardcoded the class index
    // to 0 for every object, so a multi-class dataset was unusable for training.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<uint> left = {context.addPatch(make_vec3(-0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))};
    std::vector<uint> right = {context.addPatch(make_vec3(0.7f, 0, 0), make_vec2(0.6f, 0.6f), make_SphericalCoord(0.5f * M_PI, 0.f))};

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.enableObjectDetection();
    annotation.disableSemanticSegmentation();
    annotation.disableInstanceSegmentation();
    annotation.labelPrimitives(left, "leftpatch");
    annotation.labelPrimitives(right, "rightpatch");

    std::string outdir = "./sa_test_mapping/";
    std::filesystem::remove_all(outdir);
    annotation.render(outdir.c_str());

    // ID_mapping.txt must actually contain one row per labeled object.
    size_t mapping_rows = 0;
    std::string mapfile = outdir + "ID_mapping.txt";
    if (std::filesystem::exists(mapfile)) {
        std::ifstream file(mapfile);
        std::string line;
        std::getline(file, line); // header
        while (std::getline(file, line)) {
            if (!line.empty()) {
                mapping_rows++;
            }
        }
    }

    // classes.txt lists the class names in class-ID order.
    std::vector<std::string> class_names;
    std::string classfile = outdir + "classes.txt";
    if (std::filesystem::exists(classfile)) {
        std::ifstream file(classfile);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                class_names.push_back(line);
            }
        }
    }

    // All of the view's boxes are in one file, so each label is distinguished by the class column
    // rather than by which file the box is in.
    std::vector<Visualizer::BoundingBox> boxes;
    const std::string bbox_file = outdir + "view00000/RGB_rendering.txt";
    if (std::filesystem::exists(bbox_file)) {
        boxes = Visualizer::readBoundingBoxFile(bbox_file);
    }

    // The class name file has to sit beside the annotation file, which is where the visualizer
    // looks for it when no class file is named explicitly.
    std::map<uint, std::string> sibling_class_names;
    const std::string sibling_classes = outdir + "view00000/classes.txt";
    const bool sibling_classes_exist = std::filesystem::exists(sibling_classes);
    if (sibling_classes_exist) {
        sibling_class_names = Visualizer::readBoundingBoxClassNames(sibling_classes);
    }

    std::filesystem::remove_all(outdir);

    DOCTEST_CHECK(mapping_rows == 2);
    DOCTEST_REQUIRE(class_names.size() == 2);
    DOCTEST_CHECK(class_names.at(0) == "leftpatch");
    DOCTEST_CHECK(class_names.at(1) == "rightpatch");

    // One box per labeled object, and the two labels must land on different class indices.
    DOCTEST_REQUIRE(boxes.size() == 2);
    DOCTEST_CHECK(boxes.at(0).class_ID != boxes.at(1).class_ID);

    // Every class ID present in the annotations must be named in the class file beside it, or the
    // visualizer's overlay throws when it cannot label a box.
    DOCTEST_REQUIRE(sibling_classes_exist);
    for (const Visualizer::BoundingBox &box: boxes) {
        DOCTEST_CHECK(sibling_class_names.find(box.class_ID) != sibling_class_names.end());
    }
}

TEST_CASE("SyntheticAnnotation annotation outputs share a top-down image orientation") {
    // Regression test: the annotation outputs disagreed about which way is up.
    //
    // getWindowPixelsRGB() returns the framebuffer bottom-up. The segmentation masks flip it to
    // top-down, but pixelID_combined.txt emitted raw buffer order and getGroupRectangularBBox()
    // measured its row extent without flipping. The result was that the YOLO y_center was
    // measured from the bottom of the image -- the format requires the top -- and
    // pixelID_combined.txt came out vertically mirrored relative to the masks and the RGB image.
    // writePixelID() had the same split internally: its bounding box was measured bottom-up while
    // the mask it wrote was emitted top-down, so the two did not describe the same rows.
    //
    // A single object is placed high in the scene, so every output must report it near the top.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    Context context;
    std::vector<std::vector<uint>> groups;
    groups.push_back({context.addPatch(make_vec3(0, 0, 0.8f), make_vec2(0.4f, 0.4f), make_SphericalCoord(0.5f * M_PI, 0.f))});

    SyntheticAnnotation annotation(&context);
    annotation.disableMessages();
    annotation.setWindowSize(200, 200);
    annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
    annotation.enableObjectDetection();
    annotation.enableSemanticSegmentation();
    annotation.enableInstanceSegmentation();
    annotation.labelPrimitives(groups, "high");

    std::string outdir = "./sa_test_orientation/";
    std::filesystem::remove_all(outdir);
    annotation.render(outdir.c_str());

    const std::string viewdir = outdir + "view00000/";

    // Index of the first row containing the given value, counting from the top of the file.
    auto first_row_with = [](const std::string &path, int wanted) {
        std::ifstream file(path);
        std::string line;
        int row = 0;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int value;
            while (iss >> value) {
                if (value == wanted) {
                    return row;
                }
            }
            row++;
        }
        return -1;
    };

    const int background = 16777215;

    // Semantic mask: first row that is not background.
    int semantic_first_row = -1;
    {
        std::ifstream file(viewdir + "semantic_segmentation.txt");
        std::string line;
        int row = 0;
        while (std::getline(file, line) && semantic_first_row < 0) {
            std::istringstream iss(line);
            int value;
            while (iss >> value) {
                if (value != background) {
                    semantic_first_row = row;
                    break;
                }
            }
            row++;
        }
    }

    const int pixelID_first_row = first_row_with(viewdir + "pixelID_combined.txt", 1);

    // YOLO box: class x_center y_center width height, all normalized.
    float y_center = -1.f;
    {
        std::ifstream file(viewdir + "RGB_rendering.txt");
        std::string line;
        if (std::getline(file, line)) {
            std::istringstream iss(line);
            int class_id;
            float x_center, width, height;
            iss >> class_id >> x_center >> y_center >> width >> height;
        }
    }

    // Instance mask header: xmin xmax ymin ymax.
    int instance_ymin = -1;
    {
        // The COCO bounding box is [x, y, width, height] with the origin at the top-left.
        const std::vector<Visualizer::SegmentationMask> masks = Visualizer::readSegmentationMaskFile(viewdir + "instances.json", viewdir + "RGB_rendering.jpeg");
        for (const Visualizer::SegmentationMask &mask: masks) {
            for (const std::vector<helios::vec2> &polygon: mask.polygons) {
                for (const helios::vec2 &point: polygon) {
                    if (instance_ymin < 0 || int(point.y) < instance_ymin) {
                        instance_ymin = int(point.y);
                    }
                }
            }
        }
    }

    std::filesystem::remove_all(outdir);

    DOCTEST_REQUIRE(semantic_first_row >= 0);
    DOCTEST_REQUIRE(pixelID_first_row >= 0);
    DOCTEST_REQUIRE(y_center >= 0.f);
    DOCTEST_REQUIRE(instance_ymin >= 0);

    // The object sits in the upper part of the frame, so its y_center must be in the top half.
    // Measured from the bottom instead, this comes out at roughly 0.82.
    DOCTEST_CHECK(y_center < 0.5f);

    // pixelID_combined.txt must agree with the semantic mask rather than being its mirror image.
    DOCTEST_CHECK(pixelID_first_row == semantic_first_row);

    // The instance mask's bounding box must describe the same rows as the other outputs.
    DOCTEST_CHECK(instance_ymin == semantic_first_row);
}

TEST_CASE("SyntheticAnnotation COCO masks describe the visible extent of an object") {
    // The instance masks describe each object as it actually appears in the image: the parts of it
    // hidden behind other objects are not included. This is the convention the COCO format uses,
    // and it is what makes these annotations comparable with the radiation plug-in's camera output.
    //
    // Previously the plug-in rendered each object on its own and wrote its full un-occluded extent,
    // which is a different thing entirely. The same object is annotated here twice, once with an
    // occluder in front of it and once without, and the occluded annotation must be smaller.

    if (!syntheticAnnotationDisplayAvailable()) {
        return;
    }

    auto annotated_area = [](bool with_occluder) {
        Context context;

        // The object of interest.
        std::vector<std::vector<uint>> target;
        target.push_back({context.addPatch(make_vec3(0, 0, 0), make_vec2(0.8f, 0.8f), make_SphericalCoord(0.5f * M_PI, 0.f))});

        SyntheticAnnotation annotation(&context);
        annotation.disableMessages();
        annotation.setWindowSize(200, 200);
        annotation.setCameraPosition(make_vec3(0, -3, 0), make_vec3(0, 0, 0));
        annotation.disableObjectDetection();
        annotation.disableSemanticSegmentation();
        annotation.enableInstanceSegmentation();
        annotation.labelPrimitives(target, "target");

        if (with_occluder) {
            // A second patch nearer the camera, covering the upper part of the target. It is
            // labeled too, so that it is drawn into the ID pass and therefore occludes.
            std::vector<std::vector<uint>> occluder;
            occluder.push_back({context.addPatch(make_vec3(0, -1.f, 0.3f), make_vec2(1.2f, 0.5f), make_SphericalCoord(0.5f * M_PI, 0.f))});
            annotation.labelPrimitives(occluder, "occluder");
        }

        const std::string outdir = with_occluder ? "./sa_test_occluded/" : "./sa_test_unoccluded/";
        std::filesystem::remove_all(outdir);
        annotation.render(outdir.c_str());

        int area = 0;
        const std::string json_file = outdir + "view00000/instances.json";
        if (std::filesystem::exists(json_file)) {
            std::ifstream file(json_file);
            nlohmann::json coco;
            file >> coco;
            for (const auto &ann: coco["annotations"]) {
                // Category 0 is "occluder" when present, so identify the target by name.
                int category_id = ann["category_id"];
                for (const auto &cat: coco["categories"]) {
                    if (cat["id"] == category_id && cat["name"] == "target") {
                        area += int(ann["area"]);
                    }
                }
            }
        }

        std::filesystem::remove_all(outdir);
        return area;
    };

    const int unoccluded_area = annotated_area(false);
    const int occluded_area = annotated_area(true);

    DOCTEST_REQUIRE(unoccluded_area > 0);
    DOCTEST_REQUIRE(occluded_area > 0);

    // Hiding part of the object must shrink its annotation. Under the previous un-occluded
    // behavior the two areas would be the same.
    DOCTEST_CHECK(occluded_area < unoccluded_area);
}

int SyntheticAnnotation::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
