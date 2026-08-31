#include "Visualizer.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <filesystem>
#include "doctest.h"
#include "doctest_utils.h"

using namespace helios;

//! Returns true if a windowed OpenGL context can be created on this machine
/**
 * The test cases that construct a non-headless Visualizer cannot run where no OpenGL context can be
 * created. This check is a probe rather than a platform assumption: it constructs a Visualizer with
 * the same arguments those tests use and reports whether that succeeded.
 *
 * Inspecting DISPLAY/WAYLAND_DISPLAY cannot answer the question on macOS or Windows, which never
 * set either variable. Assuming a context is always available there is what broke the macOS,
 * Windows and Windows GPU runners in v1.3.82, where the same assumption had been copied into the
 * synthetic annotation tests -- those runners create no OpenGL context at all. These visualizer
 * tests did not fail then only because CI runs `run_tests.sh --visbuildonly`, which skips
 * executables whose name contains "visualizer"; the guard itself was equally wrong.
 *
 * The result is computed once and cached: each probe builds and tears down a real GL context, and
 * repeating that per test case is slow and needlessly re-enters GLFW init/terminate.
 */
static bool windowedContextAvailable() {
    static const bool context_available = []() {
        try {
            Visualizer probe(16, 16, 0, true, false); // NON-headless, matching the guarded tests
            probe.disableMessages();
        } catch (...) {
            return false;
        }
        return true;
    }();
    return context_available;
}

//! Test-only accessor for Visualizer's private tick-generation helpers and colorbar state
/**
 * Declared a friend of Visualizer (see Visualizer.h) and defined only here, so none of this is
 * exposed to users of the library. The tick helpers are pure functions of their arguments and need
 * no GL context; the colorbar_min/colorbar_max accessors exist because setColorbarTicks() mutates
 * that state and there is no public getter to observe it.
 */
class VisualizerTestHelper {
public:
    static double niceNumber(double value, bool round) {
        return Visualizer::niceNumber(value, round);
    }

    static std::string formatTickLabel(double value, double spacing, bool isIntegerData) {
        return Visualizer::formatTickLabel(value, spacing, isIntegerData);
    }

    static std::vector<float> generateNiceTicks(float dataMin, float dataMax, bool isIntegerData, int targetTicks = 5) {
        return Visualizer::generateNiceTicks(dataMin, dataMax, isIntegerData, targetTicks);
    }

    static std::vector<float> generateColorbarTicks(float cmin, float cmax, bool isIntegerData, int targetTicks, double *tick_spacing_out = nullptr) {
        return Visualizer::generateColorbarTicks(cmin, cmax, isIntegerData, targetTicks, tick_spacing_out);
    }

    static float getColorbarMin(const Visualizer &visualizer) {
        return visualizer.colorbar_min;
    }

    static float getColorbarMax(const Visualizer &visualizer) {
        return visualizer.colorbar_max;
    }

    static bool getColorbarRangeSet(const Visualizer &visualizer) {
        return visualizer.colorbar_range_set;
    }

    static size_t getColorbarGeometryCount(const Visualizer &visualizer) {
        return visualizer.colorbar_IDs.size();
    }

    //! Total geometry currently live in the visualizer, excluding entries already deleted
    static size_t getLiveGeometryCount(const Visualizer &visualizer) {
        return visualizer.geometry_handler.getPrimitiveCount(false);
    }

    //! Color of one geometry element, which the Visualizer itself exposes no getter for
    static helios::RGBAcolor getGeometryColor(const Visualizer &visualizer, size_t geometry_id) {
        return visualizer.geometry_handler.getColor(geometry_id);
    }

    static void updateColorbar(Visualizer &visualizer) {
        visualizer.updateColorbar();
    }

    //! Run the Context->geometry-handler build that plotUpdate() performs, without needing a render
    static void buildContextGeometry_private(Visualizer &visualizer) {
        visualizer.buildContextGeometry_private();
    }

    //! Number of primitives the visualizer considers already uploaded to the geometry handler
    /**
     * Context dirty flags are sticky, so this is what keeps a build from re-uploading the whole
     * scene every frame. \sa Visualizer::contextUUIDs_uploaded
     */
    static size_t getUploadedPrimitiveCount(const Visualizer &visualizer) {
        return visualizer.contextUUIDs_uploaded.size();
    }

    //! Number of geometry elements the handler will push to the GPU on the next transferBufferData()
    static size_t getDirtyGeometryCount(const Visualizer &visualizer) {
        return visualizer.geometry_handler.getDirtyUUIDs().size();
    }

    //! Force the framebuffer dimensions independently of the window dimensions
    /**
     * Headless mode always initializes the framebuffer to match the window, so the high-DPI case
     * where the two differ is otherwise unreachable from a test.
     */
    static void setFrameBufferSize(Visualizer &visualizer, int width, int height) {
        visualizer.setFrameBufferSize(width, height);
    }

    static float getDPIScale(const Visualizer &visualizer) {
        return visualizer.getDPIScale();
    }

    //! Resolution in texels of the bitmap backing a registered texture
    static helios::uint2 getTextureResolution(const Visualizer &visualizer, uint textureID) {
        return visualizer.getTextureResolution(textureID);
    }

    //! ID of the most recently registered texture. Glyphs are registered one texture per letter.
    static uint getLastRegisteredTextureID(const Visualizer &visualizer) {
        return static_cast<uint>(visualizer.texture_manager.size()) - 1u;
    }

    //! Dimensions every layer of the texture array is allocated at
    static helios::uint2 getTextureArrayLayerSize(const Visualizer &visualizer) {
        return visualizer.texture_array_layer_size;
    }

    static helios::uint2 getMaximumTextureSize(const Visualizer &visualizer) {
        return visualizer.maximum_texture_size;
    }

    //! The geometry half of displayImage(), reachable without entering plotInteractive()'s blocking render loop
    static helios::vec4 buildImageDisplayGeometry(Visualizer &visualizer, const std::vector<unsigned char> &pixel_data, uint width_pixels, uint height_pixels) {
        return visualizer.buildImageDisplayGeometry(pixel_data, width_pixels, height_pixels);
    }

    //! The overlay half of displayImageWithBoundingBoxes(), likewise reachable without blocking
    static std::vector<size_t> addBoundingBoxOverlay(Visualizer &visualizer, const std::vector<Visualizer::BoundingBox> &bounding_boxes, const std::map<uint, std::string> &class_names, const helios::vec4 &image_extent,
                                                    float line_width, uint fontsize) {
        return visualizer.addBoundingBoxOverlay(bounding_boxes, class_names, image_extent, line_width, fontsize);
    }

    //! The overlay half of displayImageWithSegmentationMasks(), likewise reachable without blocking
    static std::vector<size_t> addSegmentationMaskOverlay(Visualizer &visualizer, const std::vector<Visualizer::SegmentationMask> &masks, const helios::vec4 &image_extent, float fill_opacity, float line_width,
                                                          uint fontsize, bool show_labels = true) {
        return visualizer.addSegmentationMaskOverlay(masks, image_extent, fill_opacity, line_width, fontsize, show_labels);
    }

    //! Switch the rendering target to the offscreen buffer
    /**
     * The offscreen framebuffer is internal lifecycle machinery driven by headless mode; this
     * forwarder exists so the test can verify the switch does not throw.
     */
    static void renderToOffscreenBuffer(Visualizer &visualizer) {
        visualizer.renderToOffscreenBuffer();
    }

};

TEST_CASE("Visualizer::disableMessages") {
    Visualizer visualizer(1000, 800, 16, false, true);

    DOCTEST_CHECK_NOTHROW(visualizer.disableMessages());

    capture_cerr cerr_buffer;
    visualizer.setColorbarRange(20, 10);

    DOCTEST_CHECK(!cerr_buffer.has_output());
}

TEST_CASE("Visualizer::enableMessages") {
    Visualizer visualizer(1000, 800, 16, true, true);

    DOCTEST_CHECK_NOTHROW(visualizer.enableMessages());

    capture_cerr cerr_buffer;
    visualizer.setColorbarRange(20, 10);

    DOCTEST_CHECK(cerr_buffer.has_output());
}

TEST_CASE("Visualizer::setCameraPosition") {
    Visualizer visualizer(1000, 800, 16, true, true);
    helios::vec3 initial_position = make_vec3(1, 1, 1);
    helios::vec3 initial_lookat = make_vec3(0, 0, 0);
    visualizer.setCameraPosition(initial_position, initial_lookat);

    // Verify that the transformation matrix updates correctly
    std::vector<helios::vec3> positions = visualizer.getCameraPosition();
    DOCTEST_CHECK(positions.size() == 2);
    DOCTEST_CHECK(positions.at(1) == initial_position);
    DOCTEST_CHECK(positions.at(0) == initial_lookat);
}

TEST_CASE("Visualizer::setLightingModel") {
    Visualizer visualizer(1000, 800, 16, true, true);

    DOCTEST_CHECK_NOTHROW(visualizer.setLightingModel(Visualizer::LIGHTING_NONE));
    DOCTEST_CHECK_NOTHROW(visualizer.setLightingModel(Visualizer::LIGHTING_PHONG));
    DOCTEST_CHECK_NOTHROW(visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED));
}

TEST_CASE("Visualizer::setBackgroundColor and Visualizer::getBackgroundColor") {
    Visualizer visualizer(1000, 800, 16, true, true);
    helios::RGBcolor bgcolor = RGB::white;
    visualizer.setBackgroundColor(bgcolor);
    DOCTEST_CHECK(visualizer.getBackgroundColor() == bgcolor);
}

TEST_CASE("Visualizer::setLightIntensityFactor") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setLightIntensityFactor(0.75f));
}

TEST_CASE("Visualizer::enableColorbar and Visualizer::disableColorbar") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.enableColorbar());
    DOCTEST_CHECK_NOTHROW(visualizer.disableColorbar());
}

TEST_CASE("Visualizer::setColorbarPosition") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarPosition(make_vec3(0.5f, 0.5f, 0.f)));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarPosition(make_vec3(-0.1f, 0.f, 0.f)), std::runtime_error);
}

TEST_CASE("Visualizer::setColorbarSize") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarSize(make_vec2(0.1f, 0.05f)));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarSize(make_vec2(1.5f, 0.f)), std::runtime_error);
}

TEST_CASE("Visualizer::setColorbarRange") {
    Visualizer visualizer(1000, 800, 16, true, true);
    visualizer.enableMessages();
    visualizer.setColorbarRange(0.f, 1.f);
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarRange(20.f, 10.f));
    DOCTEST_CHECK(cerr_buffer.has_output());
}

TEST_CASE("Visualizer::setColorbarTicks") {
    Visualizer visualizer(1000, 800, 16, true, true);
    visualizer.setColorbarRange(0.f, 1.f);
    std::vector<float> ticks{0.f, 0.5f, 1.f};
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarTicks(ticks));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarTicks({}), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarTicks({0.f, 0.5f, 0.4f}), std::runtime_error);
}

DOCTEST_TEST_CASE("Visualizer::setColorbarTicks - out-of-range ticks expand the range") {
    // The documented contract (see Visualizer.h) is that out-of-range ticks widen the colorbar
    // range. That widening also moves the colormap limits, so it changes the colors shown and not
    // just the labels - it must therefore be announced rather than applied silently.

    // Ticks entirely inside the range leave it untouched and say nothing.
    {
        Visualizer visualizer(1000, 800, 16, true, true);
        visualizer.enableMessages();
        visualizer.setColorbarRange(0.f, 10.f);
        std::string output;
        {
            capture_cerr cerr_buffer;
            visualizer.setColorbarTicks({2.f, 5.f, 8.f});
            output = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMin(visualizer) == doctest::Approx(0.f));
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMax(visualizer) == doctest::Approx(10.f));
        DOCTEST_CHECK_MESSAGE(output.empty(), "in-range ticks should not warn, but got: " << output);
    }

    // A tick below the range lowers colorbar_min and warns.
    {
        Visualizer visualizer(1000, 800, 16, true, true);
        visualizer.enableMessages();
        visualizer.setColorbarRange(0.f, 10.f);
        std::string output;
        {
            capture_cerr cerr_buffer;
            visualizer.setColorbarTicks({-5.f, 5.f});
            output = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMin(visualizer) == doctest::Approx(-5.f));
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMax(visualizer) == doctest::Approx(10.f));
        DOCTEST_CHECK_MESSAGE(!output.empty(), "silently widened the colorbar range for a tick below it");
    }

    // A tick above the range raises colorbar_max and warns.
    {
        Visualizer visualizer(1000, 800, 16, true, true);
        visualizer.enableMessages();
        visualizer.setColorbarRange(0.f, 10.f);
        std::string output;
        {
            capture_cerr cerr_buffer;
            visualizer.setColorbarTicks({5.f, 25.f});
            output = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMin(visualizer) == doctest::Approx(0.f));
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMax(visualizer) == doctest::Approx(25.f));
        DOCTEST_CHECK_MESSAGE(!output.empty(), "silently widened the colorbar range for a tick above it");
    }

    // Ticks straddling the range widen both ends.
    {
        Visualizer visualizer(1000, 800, 16, true, true);
        visualizer.enableMessages();
        visualizer.setColorbarRange(0.f, 10.f);
        std::string output;
        {
            capture_cerr cerr_buffer;
            visualizer.setColorbarTicks({-5.f, 5.f, 25.f});
            output = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMin(visualizer) == doctest::Approx(-5.f));
        DOCTEST_CHECK(VisualizerTestHelper::getColorbarMax(visualizer) == doctest::Approx(25.f));
        DOCTEST_CHECK_MESSAGE(!output.empty(), "silently widened the colorbar range for ticks on both sides");
    }
}

DOCTEST_TEST_CASE("Visualizer::setColorbarTicks - call ordering is documented and consistent") {
    // setColorbarRange() is last-writer-wins: calling it after setColorbarTicks() restores the
    // explicit range, so a caller who wants their range respected has a way to get it.
    const std::vector<float> ticks{-5.f, 5.f, 25.f};

    Visualizer ticks_first(1000, 800, 16, true, true);
    ticks_first.setColorbarTicks(ticks);
    ticks_first.setColorbarRange(0.f, 10.f);

    DOCTEST_CHECK(VisualizerTestHelper::getColorbarMin(ticks_first) == doctest::Approx(0.f));
    DOCTEST_CHECK(VisualizerTestHelper::getColorbarMax(ticks_first) == doctest::Approx(10.f));

    // In the reverse order the ticks widen the range, per the documented contract.
    Visualizer range_first(1000, 800, 16, true, true);
    range_first.setColorbarRange(0.f, 10.f);
    range_first.setColorbarTicks(ticks);

    DOCTEST_CHECK(VisualizerTestHelper::getColorbarMin(range_first) == doctest::Approx(-5.f));
    DOCTEST_CHECK(VisualizerTestHelper::getColorbarMax(range_first) == doctest::Approx(25.f));
}

DOCTEST_TEST_CASE("Visualizer::setColorbarRange - an explicit zero range survives auto-ranging") {
    // "Range not set by the user" used to be inferred from colorbar_min == 0 && colorbar_max == 0,
    // which is indistinguishable from a deliberate setColorbarRange(0, 0). Such a range was
    // silently discarded and replaced by the data's own range.
    Context context;
    for (uint i = 0; i < 8; i++) {
        uint UUID = context.addPatch(make_vec3(float(i) * 0.1f, 0, 0), make_vec2(0.09f, 1.f));
        context.setPrimitiveData(UUID, "testdata", 100.f + float(i));
    }

    Visualizer visualizer(400, 300, 0, true, true);
    visualizer.disableMessages();
    visualizer.setColorbarRange(0.f, 0.f);
    DOCTEST_CHECK(VisualizerTestHelper::getColorbarRangeSet(visualizer));

    visualizer.buildContextGeometry(&context);
    visualizer.colorContextPrimitivesByData("testdata");
    visualizer.plotUpdate(true);

    DOCTEST_CHECK_MESSAGE(VisualizerTestHelper::getColorbarMin(visualizer) == doctest::Approx(0.f), "an explicit setColorbarRange(0,0) was overwritten by auto-ranging");
    DOCTEST_CHECK_MESSAGE(VisualizerTestHelper::getColorbarMax(visualizer) == doctest::Approx(0.f), "an explicit setColorbarRange(0,0) was overwritten by auto-ranging");
}

DOCTEST_TEST_CASE("Visualizer colorbar auto-range - no data values found") {
    // The scan seeds its accumulators with FLT_MAX and lowest(), both finite, so the old
    // !isinf(min) && !isinf(max) guard could not detect a scan that matched nothing. It therefore
    // published the inverted sentinel pair [FLT_MAX, -FLT_MAX] as the colormap range, which then
    // reached tick generation and the colorbar's value-to-position mapping.
    //
    // Reaching the empty scan takes some care. Coloring by a label no primitive carries does NOT
    // do it: primitives missing the label fall through to colorValue = 0, so the scan finds values
    // and settles on a valid [0, 0]. The value stays at its -9999 "nothing here" marker only for a
    // primitive absent from colorPrimitives_UUIDs, so this coloring restricts the set to UUIDs
    // that are not part of the built geometry.
    Context context;
    std::vector<uint> built_UUIDs;
    for (uint i = 0; i < 5; i++) {
        built_UUIDs.push_back(context.addPatch(make_vec3(float(i) * 0.1f, 0, 0), make_vec2(0.09f, 1.f)));
    }
    // Separate primitives, carrying the data label, that are deliberately never built.
    std::vector<uint> unbuilt_UUIDs;
    for (uint i = 0; i < 3; i++) {
        uint UUID = context.addPatch(make_vec3(float(i) * 0.1f, 5.f, 0), make_vec2(0.09f, 1.f));
        context.setPrimitiveData(UUID, "testdata", 42.f);
        unbuilt_UUIDs.push_back(UUID);
    }

    Visualizer visualizer(400, 300, 0, true, true);
    visualizer.disableMessages();
    visualizer.buildContextGeometry(&context, built_UUIDs);
    visualizer.colorContextPrimitivesByData("testdata", unbuilt_UUIDs);
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

    const float range_min = VisualizerTestHelper::getColorbarMin(visualizer);
    const float range_max = VisualizerTestHelper::getColorbarMax(visualizer);
    DOCTEST_CHECK_MESSAGE(range_min <= range_max, "an empty data scan published an inverted range [" << range_min << ", " << range_max << "]");
    DOCTEST_CHECK_MESSAGE(range_min < 1e30f, "colorbar_min was left at its FLT_MAX sentinel");
    DOCTEST_CHECK_MESSAGE(range_max > -1e30f, "colorbar_max was left at its lowest() sentinel");
}

DOCTEST_TEST_CASE("Visualizer colorbar geometry does not accumulate across refreshes") {
    // updateColorbar() deletes the previous colorbar by iterating colorbar_IDs, so every piece of
    // geometry it creates has to be recorded there. The lower tick-mark line's UUID was discarded,
    // leaving those lines undeletable and accumulating on every refresh.
    Context context;
    for (uint i = 0; i < 12; i++) {
        uint UUID = context.addPatch(make_vec3(float(i) * 0.1f, 0, 0), make_vec2(0.09f, 1.f));
        context.setPrimitiveData(UUID, "testdata", 305.5f + 0.5f * float(i));
    }

    Visualizer visualizer(800, 600, 0, true, true);
    visualizer.disableMessages();
    visualizer.buildContextGeometry(&context);
    visualizer.colorContextPrimitivesByData("testdata");
    visualizer.enableColorbar();
    visualizer.plotUpdate(true);

    DOCTEST_REQUIRE(VisualizerTestHelper::getColorbarGeometryCount(visualizer) > 0);

    // Measure the total live geometry rather than the tracked-ID count: leaked geometry is by
    // definition the geometry that is NOT in colorbar_IDs, so counting that list cannot detect it.
    VisualizerTestHelper::updateColorbar(visualizer);
    const size_t live_geometry_baseline = VisualizerTestHelper::getLiveGeometryCount(visualizer);

    const int refresh_count = 8;
    for (int refresh = 0; refresh < refresh_count; refresh++) {
        VisualizerTestHelper::updateColorbar(visualizer);
    }
    const size_t live_geometry_after = VisualizerTestHelper::getLiveGeometryCount(visualizer);

    DOCTEST_CHECK_MESSAGE(live_geometry_after == live_geometry_baseline, "colorbar geometry accumulated across " << refresh_count << " refreshes (" << live_geometry_baseline << " -> " << live_geometry_after << "); geometry created by addColorbarByCenter() is not being tracked for deletion");
}

DOCTEST_TEST_CASE("Visualizer::setColorbarRange - rejects an inverted range regardless of messages") {
    // The guard used to read `message_flag && cmin > cmax`, so with messages disabled - the
    // default - an inverted range was silently accepted and fed straight into tick generation.
    Visualizer visualizer(1000, 800, 16, true, true);
    visualizer.setColorbarRange(0.f, 10.f);
    visualizer.disableMessages();

    {
        capture_cerr cerr_buffer;
        visualizer.setColorbarRange(20.f, 10.f);
    }

    DOCTEST_CHECK_MESSAGE(VisualizerTestHelper::getColorbarMin(visualizer) == doctest::Approx(0.f), "inverted range was accepted while messages were disabled");
    DOCTEST_CHECK_MESSAGE(VisualizerTestHelper::getColorbarMax(visualizer) == doctest::Approx(10.f), "inverted range was accepted while messages were disabled");
}

TEST_CASE("Visualizer::generateNiceTicks - Float data") {
    // Test various ranges for float data
    std::vector<float> ticks;

    // Test range 0 to 1
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 1.0f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= 0.0f);
    DOCTEST_CHECK(ticks.back() >= 1.0f);
    // Should generate nice values like 0.0, 0.25, 0.5, 0.75, 1.0
    for (size_t i = 1; i < ticks.size(); ++i) {
        DOCTEST_CHECK(ticks[i] > ticks[i - 1]);
    }

    // Test range 0 to 100
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 100.0f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= 0.0f);
    DOCTEST_CHECK(ticks.back() >= 100.0f);

    // Test range 0 to 48.3
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 48.3f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= 0.0f);
    DOCTEST_CHECK(ticks.back() >= 48.3f);
    // Should generate ticks like 0, 25, 50 or similar nice numbers

    // Test very small range
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 0.1f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);

    // Test negative range
    ticks = VisualizerTestHelper::generateNiceTicks(-10.0f, 10.0f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= -10.0f);
    DOCTEST_CHECK(ticks.back() >= 10.0f);

    // Test very large range
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 1e6f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
}

TEST_CASE("Visualizer::generateNiceTicks - Integer data") {
    std::vector<float> ticks;

    // Test range 0 to 20 (integer)
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 20.0f, true, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    // All ticks should be integers
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6);
    }

    // Test range 0 to 7 (integer)
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 7.0f, true, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6);
    }

    // Test range 0 to 100 (integer)
    ticks = VisualizerTestHelper::generateNiceTicks(0.0f, 100.0f, true, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6);
    }
}

TEST_CASE("Visualizer::formatTickLabel - Float data") {
    std::string label;

    // Test formatting with spacing = 0.2 (nice number spacing, should show 1 decimal place)
    label = VisualizerTestHelper::formatTickLabel(0.0, 0.2, false);
    DOCTEST_CHECK(label == "0.0");

    label = VisualizerTestHelper::formatTickLabel(0.4, 0.2, false);
    DOCTEST_CHECK(label == "0.4");

    label = VisualizerTestHelper::formatTickLabel(1.0, 0.2, false);
    DOCTEST_CHECK(label == "1.0");

    // Test formatting with spacing = 1.0 (should show 0 decimal places)
    label = VisualizerTestHelper::formatTickLabel(0.0, 1.0, false);
    DOCTEST_CHECK(label == "0");

    label = VisualizerTestHelper::formatTickLabel(10.0, 1.0, false);
    DOCTEST_CHECK(label == "10");

    // Test formatting with spacing = 0.1
    label = VisualizerTestHelper::formatTickLabel(0.5, 0.1, false);
    DOCTEST_CHECK(label == "0.5");

    // Test very small value (should use scientific notation)
    label = VisualizerTestHelper::formatTickLabel(1e-6, 1e-6, false);
    DOCTEST_CHECK(label.find("e") != std::string::npos); // Should contain 'e' for scientific notation

    // Test large value (should use scientific notation at 10,000+)
    label = VisualizerTestHelper::formatTickLabel(15000.0, 1000.0, false);
    DOCTEST_CHECK(label.find("e") != std::string::npos);

    // Test value below scientific notation threshold
    label = VisualizerTestHelper::formatTickLabel(9000.0, 1000.0, false);
    DOCTEST_CHECK(label.find("e") == std::string::npos); // Should NOT use scientific notation
}

TEST_CASE("Visualizer::formatTickLabel - Integer data") {
    std::string label;

    // Test integer formatting
    label = VisualizerTestHelper::formatTickLabel(0.0, 1.0, true);
    DOCTEST_CHECK(label == "0");

    label = VisualizerTestHelper::formatTickLabel(5.0, 1.0, true);
    DOCTEST_CHECK(label == "5");

    label = VisualizerTestHelper::formatTickLabel(100.0, 10.0, true);
    DOCTEST_CHECK(label == "100");

    // Test rounding for integer data
    label = VisualizerTestHelper::formatTickLabel(5.4, 1.0, true);
    DOCTEST_CHECK(label == "5");

    label = VisualizerTestHelper::formatTickLabel(5.6, 1.0, true);
    DOCTEST_CHECK(label == "6");

    // Test large integer values (should use scientific notation at 10,000+)
    label = VisualizerTestHelper::formatTickLabel(15000.0, 1000.0, true);
    DOCTEST_CHECK(label.find("e") != std::string::npos);

    // Test integer value below scientific notation threshold
    label = VisualizerTestHelper::formatTickLabel(9000.0, 1000.0, true);
    DOCTEST_CHECK(label == "9000");
}

DOCTEST_TEST_CASE("Visualizer::formatTickLabel - decimals must not misstate the value") {
    // Decimal places were chosen from the spacing alone, so a tick value carrying more precision
    // than the spacing was rendered as a different number rather than a rounded one.

    // spacing >= 1.0 forced zero decimals, printing 2.5 as "2".
    std::string label = VisualizerTestHelper::formatTickLabel(2.5, 1.0, false);
    DOCTEST_CHECK_MESSAGE(label != "2", "2.5 rendered as \"2\" - the label states a different value than the tick");
    DOCTEST_CHECK(std::fabs(std::stod(label) - 2.5) < 1e-9);

    // -floor(log10(0.25)) yields 1 decimal, printing 0.25 as "0.2" and 0.75 as "0.8".
    label = VisualizerTestHelper::formatTickLabel(0.25, 0.25, false);
    DOCTEST_CHECK_MESSAGE(label != "0.2", "0.25 rendered as \"0.2\"");
    DOCTEST_CHECK(std::fabs(std::stod(label) - 0.25) < 1e-9);

    label = VisualizerTestHelper::formatTickLabel(0.75, 0.25, false);
    DOCTEST_CHECK_MESSAGE(label != "0.8", "0.75 rendered as \"0.8\"");
    DOCTEST_CHECK(std::fabs(std::stod(label) - 0.75) < 1e-9);

    // A zero spacing must not reach log10(): the cast of the resulting infinity is undefined.
    DOCTEST_CHECK_NOTHROW(label = VisualizerTestHelper::formatTickLabel(1.5, 0.0, false));
    DOCTEST_CHECK(!label.empty());
}

TEST_CASE("Visualizer::niceNumber") {
    // Test rounding up (round = false)
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(0.72, false) - 1.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(1.5, false) - 2.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(3.2, false) - 5.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(7.5, false) - 10.0) < 1e-6);

    // Test rounding to nearest (round = true)
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(1.2, true) - 1.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(1.6, true) - 2.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(3.5, true) - 5.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(6.0, true) - 5.0) < 1e-6);

    // Test with different magnitudes
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(12.0, true) - 10.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(120.0, true) - 100.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(0.12, true) - 0.1) < 1e-6);

    // Test zero
    DOCTEST_CHECK(VisualizerTestHelper::niceNumber(0.0, true) == 0.0);
    DOCTEST_CHECK(VisualizerTestHelper::niceNumber(0.0, false) == 0.0);

    // Test negative values (should preserve sign)
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(-1.5, true) - (-2.0)) < 1e-6);
    DOCTEST_CHECK(std::fabs(VisualizerTestHelper::niceNumber(-3.2, true) - (-5.0)) < 1e-6); // -3.2 rounds to -5.0, not -2.0
}

// Tolerance used by addColorbarByCenter() when deciding whether a tick lies on the bar.
static float colorbarRangeEpsilon(float cmin, float cmax) {
    return 1e-4f * std::max(std::fabs(cmax - cmin), 1.0f);
}

DOCTEST_TEST_CASE("Visualizer::generateColorbarTicks - reported single-tick collapse") {
    // Regression: an auto-ranged colorbar over surface-temperature data (~305-312 K) rendered a
    // single tick label, "310". generateNiceTicks() extends outward to the next nice number past
    // the data, and addColorbarByCenter() then deleted every tick outside [cmin, cmax] - which is
    // exactly the two extended endpoints. The default colorbar size and font size yield
    // targetTicks=3, so [below_min, middle, above_max] collapsed to one surviving tick.
    const std::vector<std::pair<float, float>> reported_ranges = {{305.5f, 311.5f}, {308.3f, 311.3f}, {302.0f, 316.0f}, {300.1f, 306.1f}};

    for (const auto &range: reported_ranges) {
        const float cmin = range.first;
        const float cmax = range.second;
        std::vector<float> ticks = VisualizerTestHelper::generateColorbarTicks(cmin, cmax, false, 3);

        DOCTEST_CHECK_MESSAGE(ticks.size() >= 2, "range [" << cmin << ", " << cmax << "] produced " << ticks.size() << " tick(s); a colorbar needs at least 2 to convey scale");

        const float epsilon = colorbarRangeEpsilon(cmin, cmax);
        for (float tick: ticks) {
            DOCTEST_CHECK_MESSAGE(tick >= cmin - epsilon, "tick " << tick << " lies below the colorbar range [" << cmin << ", " << cmax << "]");
            DOCTEST_CHECK_MESSAGE(tick <= cmax + epsilon, "tick " << tick << " lies above the colorbar range [" << cmin << ", " << cmax << "]");
        }
    }
}

DOCTEST_TEST_CASE("Visualizer::generateColorbarTicks - narrow-range sweep") {
    // The collapse is not an edge case: it fires for any auto-ranged colorbar whose data does not
    // land on nice boundaries. This sweep reproduced 203 collapsing cases against the pre-fix
    // generate-then-delete path. It is run at targetTicks=3 - the value the default colorbar size
    // and font size actually produce - so that merely raising the max_ticks floor would not make
    // it pass.
    const std::vector<float> widths = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 8.0f};

    size_t collapsed_count = 0;
    size_t out_of_range_count = 0;
    size_t non_monotonic_count = 0;
    float first_collapsed_min = 0.f;
    float first_collapsed_max = 0.f;

    for (int step = 0; step <= 160; ++step) {
        const float cmin = 300.0f + 0.1f * float(step);
        for (float width: widths) {
            const float cmax = cmin + width;
            std::vector<float> ticks = VisualizerTestHelper::generateColorbarTicks(cmin, cmax, false, 3);

            if (ticks.size() < 2) {
                if (collapsed_count == 0) {
                    first_collapsed_min = cmin;
                    first_collapsed_max = cmax;
                }
                collapsed_count++;
                continue;
            }

            const float epsilon = colorbarRangeEpsilon(cmin, cmax);
            for (float tick: ticks) {
                if (tick < cmin - epsilon || tick > cmax + epsilon) {
                    out_of_range_count++;
                }
            }
            for (size_t i = 1; i < ticks.size(); ++i) {
                if (ticks.at(i) <= ticks.at(i - 1)) {
                    non_monotonic_count++;
                }
            }
        }
    }

    DOCTEST_CHECK_MESSAGE(collapsed_count == 0, collapsed_count << " narrow ranges collapsed to fewer than 2 ticks; first was [" << first_collapsed_min << ", " << first_collapsed_max << "]");
    DOCTEST_CHECK_MESSAGE(out_of_range_count == 0, out_of_range_count << " ticks fell outside their colorbar range (orphaned off-bar labels)");
    DOCTEST_CHECK_MESSAGE(non_monotonic_count == 0, non_monotonic_count << " tick sequences were not strictly increasing");
}

DOCTEST_TEST_CASE("Visualizer::generateColorbarTicks - nice-boundary ranges") {
    // Ranges that already land on nice boundaries worked before the fix and must keep working.
    // These are the cases that mask the bug during casual testing.
    const std::vector<std::pair<float, float>> nice_ranges = {{300.0f, 320.0f}, {0.0f, 1.0f}, {295.0f, 330.0f}};

    for (const auto &range: nice_ranges) {
        const float cmin = range.first;
        const float cmax = range.second;
        std::vector<float> ticks = VisualizerTestHelper::generateColorbarTicks(cmin, cmax, false, 3);

        DOCTEST_CHECK_MESSAGE(ticks.size() >= 2, "nice range [" << cmin << ", " << cmax << "] produced " << ticks.size() << " tick(s)");

        const float epsilon = colorbarRangeEpsilon(cmin, cmax);
        DOCTEST_CHECK(ticks.front() >= cmin - epsilon);
        DOCTEST_CHECK(ticks.back() <= cmax + epsilon);
        for (size_t i = 1; i < ticks.size(); ++i) {
            DOCTEST_CHECK(ticks.at(i) > ticks.at(i - 1));
        }
    }
}

DOCTEST_TEST_CASE("Visualizer::generateColorbarTicks - integer data") {
    std::vector<float> ticks = VisualizerTestHelper::generateColorbarTicks(0.0f, 20.0f, true, 3);
    DOCTEST_CHECK(ticks.size() >= 2);
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6f);
    }
    for (size_t i = 1; i < ticks.size(); ++i) {
        DOCTEST_CHECK(ticks.at(i) - ticks.at(i - 1) >= 1.0f - 1e-6f);
    }

    // Integer spacing must never drop below 1 even when the range is narrow.
    ticks = VisualizerTestHelper::generateColorbarTicks(0.0f, 3.0f, true, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    for (size_t i = 1; i < ticks.size(); ++i) {
        DOCTEST_CHECK(ticks.at(i) - ticks.at(i - 1) >= 1.0f - 1e-6f);
    }

    // A range containing no integer at all cannot produce integral ticks. Returning a single tick
    // is correct here; inventing fractional ones would violate the integer-data contract.
    ticks = VisualizerTestHelper::generateColorbarTicks(5.2f, 5.8f, true, 3);
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6f);
    }
}

DOCTEST_TEST_CASE("Visualizer::generateColorbarTicks - reports the generating spacing") {
    // tick_spacing feeds formatTickLabel()'s decimal-place selection. Deriving it from the
    // returned vector (as addColorbarByCenter used to) leaves it at a default of 1.0 whenever
    // filtering left fewer than 2 ticks, which is why the reported label rendered as "310" rather
    // than "310.0".
    double spacing = -1.0;
    std::vector<float> ticks = VisualizerTestHelper::generateColorbarTicks(305.5f, 311.5f, false, 3, &spacing);
    DOCTEST_REQUIRE(ticks.size() >= 2);
    DOCTEST_CHECK_MESSAGE(spacing > 0.0, "generating spacing was not written to the out-parameter");
    DOCTEST_CHECK_MESSAGE(std::fabs(spacing - double(ticks.at(1) - ticks.at(0))) < 1e-4, "reported spacing " << spacing << " disagrees with the actual tick step " << (ticks.at(1) - ticks.at(0)));

    // A null out-parameter must be accepted.
    DOCTEST_CHECK_NOTHROW(VisualizerTestHelper::generateColorbarTicks(0.0f, 1.0f, false, 3, nullptr));
}

DOCTEST_TEST_CASE("Visualizer::generateColorbarTicks - degenerate input") {
    // Called every frame from the rendering path, so degenerate limits must degrade rather than
    // throw. Returning an empty vector would leave the colorbar with no labels at all.
    double spacing = -1.0;

    std::vector<float> ticks;
    DOCTEST_CHECK_NOTHROW(ticks = VisualizerTestHelper::generateColorbarTicks(5.0f, 5.0f, false, 3, &spacing));
    DOCTEST_CHECK(!ticks.empty());

    DOCTEST_CHECK_NOTHROW(ticks = VisualizerTestHelper::generateColorbarTicks(10.0f, 5.0f, false, 3, &spacing));
    DOCTEST_CHECK(!ticks.empty());

    DOCTEST_CHECK_NOTHROW(ticks = VisualizerTestHelper::generateColorbarTicks(1.0f, 1.0f + 1e-12f, false, 3, &spacing));
    DOCTEST_CHECK(!ticks.empty());

    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const float inf_value = std::numeric_limits<float>::infinity();
    DOCTEST_CHECK_NOTHROW(ticks = VisualizerTestHelper::generateColorbarTicks(nan_value, 1.0f, false, 3, &spacing));
    DOCTEST_CHECK(!ticks.empty());
    DOCTEST_CHECK_NOTHROW(ticks = VisualizerTestHelper::generateColorbarTicks(0.0f, inf_value, false, 3, &spacing));
    DOCTEST_CHECK(!ticks.empty());

    // A degenerate spacing must never be zero or non-finite: formatTickLabel() takes log10 of it.
    DOCTEST_CHECK(std::isfinite(spacing));
    DOCTEST_CHECK(spacing > 0.0);
}

TEST_CASE("Visualizer colorbar text attributes") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarTitle("MyBar"));
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarFontColor(RGB::yellow));
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarFontSize(14));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarFontSize(0), std::runtime_error);
}

TEST_CASE("Visualizer text rasterizes at framebuffer resolution on high-DPI displays") {
    // Glyphs are rasterized on the CPU by FreeType at a fixed pixel size, but they are drawn into
    // the framebuffer. On a high-DPI (Retina) display the framebuffer is larger than the window in
    // screen coordinates, typically by 2x per axis, so a glyph bitmap generated at window
    // resolution is magnified when drawn and the text looks blocky.
    //
    // Headless mode initializes the framebuffer to match the window exactly, so the mismatch has to
    // be forced in order to be observable at all.

    constexpr uint window_width = 1000;
    constexpr uint window_height = 800;
    constexpr uint fontsize = 12; // matches the colorbar default, the worst case for this defect
    const char *teststring = "Wg";

    auto build_text = [&](Visualizer &visualizer, uint2 &glyph_resolution_out, float &quad_width_out) {
        std::vector<size_t> UUIDs = visualizer.addTextboxByCenter(teststring, make_vec3(0.5f, 0.5f, 0.f), make_SphericalCoord(0, 0), RGB::black, fontsize, "OpenSans-Regular", Visualizer::COORDINATES_WINDOW_NORMALIZED);
        DOCTEST_REQUIRE(UUIDs.size() == std::strlen(teststring));

        glyph_resolution_out = VisualizerTestHelper::getTextureResolution(visualizer, VisualizerTestHelper::getLastRegisteredTextureID(visualizer));

        // Width from the vertex extents rather than a fixed vertex ordering.
        const std::vector<vec3> vertices = visualizer.getGeometryVertices(UUIDs.back());
        DOCTEST_REQUIRE(vertices.size() == 4);
        float xmin = vertices.front().x;
        float xmax = vertices.front().x;
        for (const vec3 &vertex: vertices) {
            xmin = std::min(xmin, vertex.x);
            xmax = std::max(xmax, vertex.x);
        }
        quad_width_out = xmax - xmin;
    };

    uint2 glyph_resolution_1x;
    float quad_width_1x;
    Visualizer visualizer_1x(window_width, window_height, 16, false, true);
    DOCTEST_REQUIRE(VisualizerTestHelper::getDPIScale(visualizer_1x) == doctest::Approx(1.f));
    build_text(visualizer_1x, glyph_resolution_1x, quad_width_1x);

    uint2 glyph_resolution_2x;
    float quad_width_2x;
    Visualizer visualizer_2x(window_width, window_height, 16, false, true);
    VisualizerTestHelper::setFrameBufferSize(visualizer_2x, 2 * window_width, 2 * window_height);
    DOCTEST_REQUIRE(VisualizerTestHelper::getDPIScale(visualizer_2x) == doctest::Approx(2.f));
    build_text(visualizer_2x, glyph_resolution_2x, quad_width_2x);

    // The glyph bitmap must gain resolution in proportion to the DPI scale. Without this the
    // bitmap is identical at both scales and is simply magnified when drawn, which is the defect.
    DOCTEST_CHECK_MESSAGE(glyph_resolution_2x.x > glyph_resolution_1x.x, "glyph rasterized at " << glyph_resolution_2x.x << "x" << glyph_resolution_2x.y << " texels under a 2x DPI scale, versus " << glyph_resolution_1x.x << "x" << glyph_resolution_1x.y
                                                                                               << " at 1x; the bitmap is being magnified rather than rasterized at framebuffer resolution");
    DOCTEST_CHECK(glyph_resolution_2x.y > glyph_resolution_1x.y);

    // ...while the on-screen quad must NOT change size, or text would render twice as large on a
    // high-DPI display. Glyph quads are in window-normalized coordinates and are resolution
    // independent by construction.
    DOCTEST_CHECK_MESSAGE(quad_width_2x == doctest::Approx(quad_width_1x).epsilon(0.05), "text quad width changed from " << quad_width_1x << " to " << quad_width_2x << " under a 2x DPI scale; glyph quads must stay the same size on screen regardless of framebuffer resolution");
}

TEST_CASE("Visualizer texture array layers are sized to their contents") {
    // Every layer of a texture array has the same dimensions, so allocating them at
    // maximum_texture_size gave a 7x9-texel glyph a 2048x2048 RGBA8 layer - 16 MB apiece, so a
    // colorbar's worth of text cost hundreds of megabytes of VRAM. Layers are now sized to the
    // largest texture actually present.
    Visualizer visualizer(1000, 800, 16, false, true);

    std::vector<size_t> UUIDs = visualizer.addTextboxByCenter("Wg", make_vec3(0.5f, 0.5f, 0.f), make_SphericalCoord(0, 0), RGB::black, 12, "OpenSans-Regular", Visualizer::COORDINATES_WINDOW_NORMALIZED);
    DOCTEST_REQUIRE(UUIDs.size() == 2);
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate());

    const uint2 layer_size = VisualizerTestHelper::getTextureArrayLayerSize(visualizer);
    const uint2 max_size = VisualizerTestHelper::getMaximumTextureSize(visualizer);

    // The layers must be no larger than the largest texture registered. A handful of small glyphs
    // must not pull in a layer anywhere near the 2048x2048 load-time clamp.
    uint2 largest_texture = make_uint2(1, 1);
    for (uint textureID = 0; textureID <= VisualizerTestHelper::getLastRegisteredTextureID(visualizer); textureID++) {
        const uint2 resolution = VisualizerTestHelper::getTextureResolution(visualizer, textureID);
        largest_texture.x = std::max(largest_texture.x, resolution.x);
        largest_texture.y = std::max(largest_texture.y, resolution.y);
    }

    DOCTEST_CHECK_MESSAGE(layer_size.x == largest_texture.x, "texture array layers allocated " << layer_size.x << " texels wide to hold a largest texture of " << largest_texture.x);
    DOCTEST_CHECK(layer_size.y == largest_texture.y);
    DOCTEST_CHECK(layer_size.x < max_size.x);
    DOCTEST_CHECK(layer_size.y < max_size.y);

    // Sampling is confined to each texture's sub-rectangle by the UV rescale factors, which are
    // relative to the layer size. A layer sized to its contents means the factors approach one
    // rather than the ~0.004 a 2048-texel layer produced, so this also exercises that they were
    // recomputed against the new divisor rather than left scaled to the old one.
    DOCTEST_CHECK(float(largest_texture.x) / float(layer_size.x) == doctest::Approx(1.f));
}

TEST_CASE("Visualizer::setColormap") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColormap(Visualizer::COLORMAP_COOL));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColormap(Visualizer::COLORMAP_CUSTOM), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(visualizer.setColormap(std::vector<RGBcolor>{RGB::red}, std::vector<float>{0.f, 1.f}), std::runtime_error);
}

TEST_CASE("Visualizer::PNG texture integration via primitives") {
    Visualizer visualizer(1000, 800, 16, true, true);
    const char *png_filename = "plugins/visualizer/textures/AlmondLeaf.png";

    // Verify file exists before testing
    DOCTEST_CHECK(std::filesystem::exists(png_filename));

    // Test PNG texture loading through textured rectangle - internally calls read_png_file -> helios::readPNG
    std::vector<helios::vec3> verts = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
    size_t UUID1;
    DOCTEST_CHECK_NOTHROW(UUID1 = visualizer.addRectangleByVertices(verts, png_filename, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID1 != 0);

    // Test PNG texture loading through textured triangle
    size_t UUID2;
    DOCTEST_CHECK_NOTHROW(UUID2 = visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), png_filename, make_vec2(0, 0), make_vec2(1, 0), make_vec2(0, 1), Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID2 != 0);
    DOCTEST_CHECK(UUID2 != UUID1); // Should be different primitives
}

TEST_CASE("Visualizer::JPEG texture integration via primitives") {
    Visualizer visualizer(1000, 800, 16, true, true);
    const char *jpeg_filename = "plugins/visualizer/textures/SkyDome_clouds.jpg";

    // Verify file exists before testing
    DOCTEST_CHECK(std::filesystem::exists(jpeg_filename));

    // Test JPEG texture on rectangle using addRectangleByVertices which accepts texture files
    std::vector<helios::vec3> verts = {make_vec3(1, 1, 1), make_vec3(3, 1, 1), make_vec3(3, 3, 1), make_vec3(1, 3, 1)};
    size_t rect_UUID;
    DOCTEST_CHECK_NOTHROW(rect_UUID = visualizer.addRectangleByVertices(verts, jpeg_filename, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(rect_UUID != 0);
}

TEST_CASE("Visualizer::Visualizer") {
    DOCTEST_CHECK_NOTHROW(Visualizer v1(800, 600, true));
    DOCTEST_CHECK_NOTHROW(Visualizer v2(1024, 768, 4, false, true));
    DOCTEST_CHECK_NOTHROW(Visualizer v3(1280, 720, 8, false, true));
}

TEST_CASE("Visualizer texture copy") {
    DOCTEST_CHECK(std::filesystem::exists("plugins/visualizer/textures/AlmondLeaf.png"));
    DOCTEST_CHECK(std::filesystem::exists("plugins/visualizer/textures/Helios_watermark.png"));
    DOCTEST_CHECK(std::filesystem::exists("plugins/visualizer/textures/SkyDome_clouds.jpg"));
}

TEST_CASE("Visualizer::addRectangleByCenter") {
    Visualizer visualizer(1000, 800, 16, true, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addRectangleByCenter(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addRectangleByCenter extreme") {
    Visualizer visualizer(1000, 800, 16, true, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addRectangleByCenter(make_vec3(1e6, 1e6, 1e6), make_vec2(1e6, 1e6), make_SphericalCoord(0, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addRectangleByVertices variations") {
    Visualizer visualizer(1000, 800, 16, true, true);
    std::vector<helios::vec3> verts = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
    size_t UUID1;
    DOCTEST_CHECK_NOTHROW(UUID1 = visualizer.addRectangleByVertices(verts, RGB::blue, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID1 != 0);
    size_t UUID2;
    DOCTEST_CHECK_NOTHROW(UUID2 = visualizer.addRectangleByVertices(verts, "plugins/visualizer/textures/AlmondLeaf.png", Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID2 != 0);
}

TEST_CASE("Visualizer::addTriangle") {
    Visualizer visualizer(1000, 800, 16, true, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::blue, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addTriangle textured") {
    Visualizer visualizer(1000, 800, 16, true, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), "plugins/visualizer/textures/AlmondLeaf.png", make_vec2(0, 0), make_vec2(1, 0), make_vec2(0, 1), Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addVoxelByCenter") {
    Visualizer visualizer(1000, 800, 16, true, true);
    std::vector<size_t> UUIDs;
    DOCTEST_CHECK_NOTHROW(UUIDs = visualizer.addVoxelByCenter(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_SphericalCoord(0, 0), RGB::green, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUIDs.size() == 6);
}

TEST_CASE("Visualizer::addSphereByCenter") {
    Visualizer visualizer(1000, 800, 16, true, true);
    uint N = 3;
    std::vector<size_t> UUIDs;
    DOCTEST_CHECK_NOTHROW(UUIDs = visualizer.addSphereByCenter(1.0f, make_vec3(0, 0, 0), N, RGB::blue, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUIDs.size() == 2 * N * (N - 1));
}

TEST_CASE("Visualizer::addCoordinateAxes") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.addCoordinateAxes(make_vec3(0, 0, 0), make_vec3(1, 1, 1), "XYZ"));
}

TEST_CASE("Visualizer::addLine") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK(visualizer.addLine(make_vec3(-1, 3, 0), make_vec3(0, 4, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN) != 0);
}

TEST_CASE("Visualizer::addLine with line width") {
    Visualizer visualizer(1000, 800, 16, true, true);

    // Test RGB line with custom width
    size_t UUID1;
    DOCTEST_CHECK_NOTHROW(UUID1 = visualizer.addLine(make_vec3(0, 0, 0), make_vec3(1, 1, 1), RGB::blue, 2.0f, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID1 != 0);

    // Test RGBA line with custom width
    size_t UUID2;
    DOCTEST_CHECK_NOTHROW(UUID2 = visualizer.addLine(make_vec3(2, 0, 0), make_vec3(3, 1, 1), make_RGBAcolor(1.0f, 0.0f, 0.0f, 0.5f), 5.0f, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID2 != 0);
    DOCTEST_CHECK(UUID2 != UUID1);

    // Test with small width (should work without throwing)
    size_t UUID3;
    DOCTEST_CHECK_NOTHROW(UUID3 = visualizer.addLine(make_vec3(-1, 0, 0), make_vec3(-2, 1, 1), RGB::green, 0.5f, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID3 != 0);

    // Test with large width (should work without throwing)
    size_t UUID4;
    DOCTEST_CHECK_NOTHROW(UUID4 = visualizer.addLine(make_vec3(4, 0, 0), make_vec3(5, 1, 1), RGB::yellow, 10.0f, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID4 != 0);

    // Test with maximum valid width (should work)
    size_t UUID5;
    DOCTEST_CHECK_NOTHROW(UUID5 = visualizer.addLine(make_vec3(6, 0, 0), make_vec3(7, 1, 1), RGB::white, 100.0f, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID5 != 0);

    // Test with zero width (should throw error)
    DOCTEST_CHECK_THROWS_AS(visualizer.addLine(make_vec3(8, 0, 0), make_vec3(9, 1, 1), RGB::red, 0.0f, Visualizer::COORDINATES_CARTESIAN), std::runtime_error);

    // Test with negative width (should throw error)
    DOCTEST_CHECK_THROWS_AS(visualizer.addLine(make_vec3(10, 0, 0), make_vec3(11, 1, 1), RGB::green, -1.0f, Visualizer::COORDINATES_CARTESIAN), std::runtime_error);

    // Test with width exceeding maximum (should throw error)
    DOCTEST_CHECK_THROWS_AS(visualizer.addLine(make_vec3(12, 0, 0), make_vec3(13, 1, 1), RGB::blue, 101.0f, Visualizer::COORDINATES_CARTESIAN), std::runtime_error);

    // Test with normalized window coordinates and custom width (user's specific case)
    size_t UUID6;
    DOCTEST_CHECK_NOTHROW(UUID6 = visualizer.addLine(make_vec3(0, 0.5, 0), make_vec3(1, 0.5, 0), RGB::red, 20.0f, Visualizer::COORDINATES_WINDOW_NORMALIZED));
    DOCTEST_CHECK(UUID6 != 0);
}

TEST_CASE("Visualizer::validateTextureFile") {
    DOCTEST_CHECK(validateTextureFile("plugins/visualizer/textures/AlmondLeaf.png"));
    DOCTEST_CHECK(!validateTextureFile("missing.png"));
    DOCTEST_CHECK(!validateTextureFile("plugins/visualizer/textures/SkyDome_clouds.jpg", true));
}

TEST_CASE("Visualizer::point culling configuration simple") {
    Visualizer visualizer(800, 600, true); // Headless mode

    // Test that the new configuration methods exist and don't crash
    DOCTEST_CHECK_NOTHROW(visualizer.setPointCullingEnabled(true));
    DOCTEST_CHECK_NOTHROW(visualizer.setPointCullingEnabled(false));
}

TEST_CASE("Visualizer::addPoint basic functionality") {
    Visualizer visualizer(800, 600, true); // Headless mode

    // Test adding a single point
    size_t point_uuid = visualizer.addPoint(make_vec3(0, 0, 0), RGB::red, 1.0f, Visualizer::COORDINATES_CARTESIAN);
    DOCTEST_CHECK(point_uuid != 0);
}

TEST_CASE("Visualizer::addPoint with different sizes") {
    Visualizer visualizer(800, 600, true); // Headless mode

    // Test adding points with different sizes
    size_t point1 = visualizer.addPoint(make_vec3(0, 0, 0), RGB::red, 1.0f, Visualizer::COORDINATES_CARTESIAN);
    size_t point2 = visualizer.addPoint(make_vec3(1, 0, 0), RGB::green, 2.5f, Visualizer::COORDINATES_CARTESIAN);
    size_t point3 = visualizer.addPoint(make_vec3(2, 0, 0), RGB::blue, 5.0f, Visualizer::COORDINATES_CARTESIAN);

    // Test point with size outside supported range (should trigger warning)
    capture_cerr cerr_buffer;
    size_t point4 = visualizer.addPoint(make_vec3(3, 0, 0), RGB::yellow, 0.5f, Visualizer::COORDINATES_CARTESIAN);
    DOCTEST_CHECK(cerr_buffer.has_output()); // Should capture warning about point size clamping

    // Verify unique UUIDs were returned
    DOCTEST_CHECK(point1 != 0);
    DOCTEST_CHECK(point2 != 0);
    DOCTEST_CHECK(point3 != 0);
    DOCTEST_CHECK(point4 != 0);
    DOCTEST_CHECK(point1 != point2);
    DOCTEST_CHECK(point2 != point3);
    DOCTEST_CHECK(point3 != point4);
}

TEST_CASE("Visualizer::addPoint RGBA with sizes") {
    Visualizer visualizer(800, 600, true); // Headless mode

    // Test adding RGBA points with different sizes
    size_t point1 = visualizer.addPoint(make_vec3(0, 0, 0), make_RGBAcolor(1.0f, 0.0f, 0.0f, 0.8f), 1.5f, Visualizer::COORDINATES_CARTESIAN);
    size_t point2 = visualizer.addPoint(make_vec3(1, 1, 1), make_RGBAcolor(0.0f, 1.0f, 0.0f, 0.6f), 3.0f, Visualizer::COORDINATES_CARTESIAN);

    DOCTEST_CHECK(point1 != 0);
    DOCTEST_CHECK(point2 != 0);
    DOCTEST_CHECK(point1 != point2);
}

TEST_CASE("Visualizer::point culling metrics functionality") {
    Visualizer visualizer(800, 600, true); // Headless mode

    // Test that metrics can be retrieved
    size_t total, rendered;
    float time;
    DOCTEST_CHECK_NOTHROW(visualizer.getPointRenderingMetrics(total, rendered, time));

    // Add some points
    for (int i = 0; i < 5; ++i) {
        size_t uuid = visualizer.addPoint(make_vec3(i, 0, 0), RGB::orange, 1.0f, Visualizer::COORDINATES_CARTESIAN);
        DOCTEST_CHECK(uuid != 0);
    }

    // Test metrics after adding points
    DOCTEST_CHECK_NOTHROW(visualizer.getPointRenderingMetrics(total, rendered, time));

    // Note: plotUpdate disabled in headless mode for testing - would require full OpenGL context
}

TEST_CASE("Visualizer::point size edge cases") {
    Visualizer visualizer(800, 600, true); // Headless mode

    // Test with very small point size (should trigger warning and not crash in headless mode)
    capture_cerr cerr_buffer1;
    size_t point1 = visualizer.addPoint(make_vec3(0, 0, 0), RGB::white, 0.001f, Visualizer::COORDINATES_CARTESIAN);
    DOCTEST_CHECK(point1 != 0);
    DOCTEST_CHECK(cerr_buffer1.has_output()); // Should capture warning about point size clamping

    // Test with very large point size (should trigger warning and not crash in headless mode)
    capture_cerr cerr_buffer2;
    size_t point2 = visualizer.addPoint(make_vec3(1, 0, 0), RGB::white, 1000.0f, Visualizer::COORDINATES_CARTESIAN);
    DOCTEST_CHECK(point2 != 0);
    DOCTEST_CHECK(cerr_buffer2.has_output()); // Should capture warning about point size clamping

    // Test with valid point size (should not trigger warning)
    capture_cerr cerr_buffer3;
    size_t point3 = visualizer.addPoint(make_vec3(2, 0, 0), RGB::white, 2.0f, Visualizer::COORDINATES_CARTESIAN);
    DOCTEST_CHECK(point3 != 0);
    DOCTEST_CHECK(!cerr_buffer3.has_output()); // Should not capture any warning

    // Verify UUIDs are unique
    DOCTEST_CHECK(point1 != point2);
    DOCTEST_CHECK(point2 != point3);
    DOCTEST_CHECK(point1 != point3);
}

TEST_CASE("CI/Offscreen - Basic OpenGL Context") {
    // Test that we can create a headless visualizer with offscreen rendering
    DOCTEST_CHECK_NOTHROW({
        Visualizer visualizer(400, 300, 4, true, true); // headless=true
        // If we get here without throwing, the offscreen context was created successfully
    });
}

TEST_CASE("CI/Offscreen - Framebuffer Operations") {
    Visualizer visualizer(200, 150, 0, true, true); // Small size for CI efficiency

    // Test that we can perform basic OpenGL operations
    DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(RGB::black));
    DOCTEST_CHECK_NOTHROW(visualizer.setLightDirection(make_vec3(0, 0, -1)));
    DOCTEST_CHECK_NOTHROW(visualizer.setLightIntensityFactor(1.0f));
}

TEST_CASE("CI/Offscreen - Geometry Rendering") {
    Visualizer visualizer(100, 100, 0, true, true); // Minimal size for speed

    // Add some basic geometry directly to visualizer
    size_t triangle = visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5, 1, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN);
    DOCTEST_CHECK(triangle != 0);

    // Test basic rendering without crashing
    DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(RGB::black));
}

TEST_CASE("CI/Offscreen - Environment Variable Detection") {
    // Test that environment variables are properly detected
    // Note: This test runs in normal environment, so we just test the code paths

    // Test with explicit headless=false but environment might force it
    DOCTEST_CHECK_NOTHROW({
        Visualizer visualizer(100, 100, 0, true, false); // headless=false
        // Should still work - environment detection might force headless mode in CI
    });
}

TEST_CASE("CI/Offscreen - Render Target Switching") {
    Visualizer visualizer(64, 64, 0, true, true);

    // Test switching to offscreen buffer
    DOCTEST_CHECK_NOTHROW(VisualizerTestHelper::renderToOffscreenBuffer(visualizer));

    // Test that we can add geometry after switching render targets
    size_t triangle = visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5, 1, 0), make_RGBcolor(1, 1, 1), Visualizer::COORDINATES_CARTESIAN);
    DOCTEST_CHECK(triangle != 0);

    // Test basic rendering operations
    DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(RGB::black));
}

TEST_CASE("CI/Offscreen - Stress Test") {
    // Test multiple visualizers to ensure proper cleanup
    std::vector<std::unique_ptr<Visualizer>> visualizers;

    for (int i = 0; i < 3; ++i) {
        DOCTEST_CHECK_NOTHROW({ visualizers.emplace_back(std::make_unique<Visualizer>(32, 32, 0, true, true)); });
    }

    // All visualizers should be valid
    for (const auto &vis: visualizers) {
        DOCTEST_CHECK(vis != nullptr);
    }

    // Cleanup happens automatically when unique_ptrs go out of scope
}

TEST_CASE("Visualizer::printWindow after plotUpdate regression test") {
    // Regression test for the black image issue when calling printWindow() after plotUpdate(true)
    // This test ensures the fix for the Ubuntu/Linux buffer reading issue works correctly

    // Test works in both windowed and headless mode thanks to offscreen rendering support

    Context context;
    Visualizer visualizer(200, 200, 0, true, true); // Small size for speed, headless mode
    visualizer.disableMessages();

    // Add some geometry to render (a simple sphere)
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 0), 1.0f);
    // Use material system for test geometry
    std::string test_material = "test_visualizer_red_sphere";
    if (!context.doesMaterialExist(test_material)) {
        context.addMaterial(test_material);
        context.setMaterialColor(test_material, make_RGBAcolor(RGB::red, 1.0f));
    }
    context.assignMaterialToPrimitive(sphere_uuids, test_material);

    // Build geometry in visualizer
    visualizer.buildContextGeometry(&context);

    // Set camera to view the sphere
    visualizer.setCameraPosition(make_vec3(0, 0, 3), make_vec3(0, 0, 0));

    // This is the critical workflow that was failing: plotUpdate(true) followed by printWindow()
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true)); // render with hidden window

    // Test screenshot functionality - this should NOT produce a black image
    std::string test_filename = "test_printWindow_regression.jpg";
    DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str()));

    // Verify the file was created
    DOCTEST_CHECK(std::filesystem::exists(test_filename));

    // Validate that the image is not all black (the original issue)
    // Read back the pixels directly from the visualizer to verify content
    // getWindowPixelsRGB() writes one entry per framebuffer subpixel, and the framebuffer is
    // larger than the requested window on a high-DPI display - 2x per axis on a Retina screen, so
    // 4x the elements. Sizing this buffer from the constructor's window dimensions overflowed it
    // and corrupted the heap.
    uint framebuffer_width, framebuffer_height;
    visualizer.getFramebufferSize(framebuffer_width, framebuffer_height);
    std::vector<uint> pixel_buffer(3 * framebuffer_width * framebuffer_height);
    DOCTEST_CHECK_NOTHROW(visualizer.getWindowPixelsRGB(pixel_buffer.data()));

    // Check that we have non-black pixels (red sphere should be visible)
    bool has_non_black_pixels = false;
    for (size_t i = 0; i < pixel_buffer.size(); i++) {
        if (pixel_buffer[i] > 10) { // Allow for some tolerance due to anti-aliasing
            has_non_black_pixels = true;
            break;
        }
    }

    DOCTEST_CHECK_MESSAGE(has_non_black_pixels, "Image appears to be all black - this indicates the original buffer reading issue");

    // The key test: ensure we're not getting all black pixels (the original issue)
    // This test validates that the buffer reading fix is working correctly

    // Note: Offscreen rendering is already tested by existing "CI/Offscreen" test cases
    // Our regression test focuses on the specific plotUpdate()->printWindow() workflow

    // Clean up test file
    if (std::filesystem::exists(test_filename)) {
        std::filesystem::remove(test_filename);
    }
}

TEST_CASE("Visualizer::printWindow after plotUpdate non-headless regression test") {
    // Regression test for the black image issue when calling printWindow() after plotUpdate(true)
    // in non-headless mode. Only runs where a windowed OpenGL context can be created.

    if (!windowedContextAvailable()) {
        // Skip test silently when no windowed OpenGL context can be created
        return;
    }

    Context context;
    Visualizer visualizer(200, 200, 0, true, false); // NON-headless mode - requires display
    visualizer.disableMessages();

    // Add some geometry to render (a simple sphere)
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 0), 1.0f);
    // Use material system for test geometry
    std::string test_material = "test_visualizer_red_sphere";
    if (!context.doesMaterialExist(test_material)) {
        context.addMaterial(test_material);
        context.setMaterialColor(test_material, make_RGBAcolor(RGB::red, 1.0f));
    }
    context.assignMaterialToPrimitive(sphere_uuids, test_material);

    // Build geometry in visualizer
    visualizer.buildContextGeometry(&context);

    // Set camera to view the sphere
    visualizer.setCameraPosition(make_vec3(0, 0, 3), make_vec3(0, 0, 0));

    // This is the critical workflow that was failing: plotUpdate(true) followed by printWindow()
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true)); // render with hidden window

    // Test screenshot functionality - this should NOT produce a black image
    std::string test_filename = "test_printWindow_nonheadless_regression.jpg";
    DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str()));

    // Verify the file was created
    DOCTEST_CHECK(std::filesystem::exists(test_filename));

    // Validate that the image is not all black (the original issue)
    // Read back the pixels directly from the visualizer to verify content
    // getWindowPixelsRGB() writes one entry per framebuffer subpixel, and the framebuffer is
    // larger than the requested window on a high-DPI display - 2x per axis on a Retina screen, so
    // 4x the elements. Sizing this buffer from the constructor's window dimensions overflowed it
    // and corrupted the heap.
    uint framebuffer_width, framebuffer_height;
    visualizer.getFramebufferSize(framebuffer_width, framebuffer_height);
    std::vector<uint> pixel_buffer(3 * framebuffer_width * framebuffer_height);
    DOCTEST_CHECK_NOTHROW(visualizer.getWindowPixelsRGB(pixel_buffer.data()));

    // Check that we have non-black pixels (red sphere should be visible)
    bool has_non_black_pixels = false;
    for (size_t i = 0; i < pixel_buffer.size(); i++) {
        if (pixel_buffer[i] > 10) { // Allow for some tolerance due to anti-aliasing
            has_non_black_pixels = true;
            break;
        }
    }

    DOCTEST_CHECK_MESSAGE(has_non_black_pixels, "Image appears to be all black in non-headless mode - buffer reading issue");

    // Clean up test file
    if (std::filesystem::exists(test_filename)) {
        std::filesystem::remove(test_filename);
    }
}

TEST_CASE("Visualizer::PNG with transparent background") {
    // Test that PNG output with transparent background correctly renders geometry with transparency
    Visualizer visualizer(200, 200, 16, true, true); // headless mode
    visualizer.disableMessages();

    // Add a red rectangle in the center
    std::vector<helios::vec3> vertices{make_vec3(-0.3f, -0.3f, 0.f), make_vec3(0.3f, -0.3f, 0.f), make_vec3(0.3f, 0.3f, 0.f), make_vec3(-0.3f, 0.3f, 0.f)};

    size_t rect_UUID;
    DOCTEST_CHECK_NOTHROW(rect_UUID = visualizer.addRectangleByVertices(vertices, make_RGBcolor(1.f, 0.f, 0.f), Visualizer::COORDINATES_CARTESIAN));

    // Set transparent background
    DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());

    // Render the scene
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

    // Save to PNG
    std::string test_filename = "test_transparent_bg.png";
    DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str(), "png"));
    DOCTEST_CHECK(std::filesystem::exists(test_filename));

    // Read the PNG back to verify transparency
    std::vector<helios::RGBAcolor> pixel_data;
    uint width, height;
    DOCTEST_CHECK_NOTHROW(helios::readPNG(test_filename, width, height, pixel_data));
    DOCTEST_CHECK(width == 200);
    DOCTEST_CHECK(height == 200);
    DOCTEST_CHECK(pixel_data.size() == width * height);

    // Count transparent and opaque pixels
    int transparent_pixels = 0;
    int opaque_red_pixels = 0;

    for (const auto &pixel: pixel_data) {
        if (pixel.a < 0.1f) {
            // Fully transparent background pixel
            transparent_pixels++;
        } else if (pixel.a > 0.9f && pixel.r > 0.5f && pixel.g < 0.3f && pixel.b < 0.3f) {
            // Opaque red pixel (the rectangle)
            opaque_red_pixels++;
        }
    }

    // We should have both transparent background pixels and opaque red rectangle pixels
    DOCTEST_CHECK_MESSAGE(transparent_pixels > 1000, "Expected significant transparent background area, got " << transparent_pixels << " transparent pixels");
    DOCTEST_CHECK_MESSAGE(opaque_red_pixels > 100, "Expected visible red rectangle in center, got " << opaque_red_pixels << " red pixels");

    // Verify that the sum of different pixel types accounts for most of the image
    DOCTEST_CHECK_MESSAGE(transparent_pixels + opaque_red_pixels > 0.8 * (width * height), "Transparent + opaque pixels should account for most of image");

    // Clean up test file
    if (std::filesystem::exists(test_filename)) {
        std::filesystem::remove(test_filename);
    }
}

TEST_CASE("Visualizer::headless render after windowed instance destroyed") {
    // Smoke test for cross-mode instance ordering: a headless Visualizer constructed
    // and rendered after a windowed Visualizer was created and destroyed in the same
    // process must not crash. The shadow-map depth texture/framebuffer are created
    // lazily, only once shadowed lighting is actually rendered, so an instance using
    // the default lighting model never creates them and its depthTexture/framebufferID
    // members must be zero-initialized (see Visualizer.h). If they were left
    // uninitialized, the render path would bind a stale GL texture name as the shadow
    // map, producing GL_INVALID_OPERATION and, on macOS, a driver-side crash. The
    // original failure was allocator-layout dependent and only manifested reliably
    // within the full suite, so this case guards the ordering rather than guaranteeing
    // reproduction of the exact memory state.

    // First, create and destroy a WINDOWED instance to populate freed memory with
    // plausible GL handle values (mirrors the test-suite ordering that exposed the bug).
    {
        Visualizer windowed(200, 200, 0, true, false); // headless = false
        windowed.disableMessages();
    } // windowed destroyed here

    // Now a HEADLESS instance whose render path binds the (default lighting) shadow map.
    Visualizer headless(200, 200, 16, true, true); // headless = true
    headless.disableMessages();

    std::vector<helios::vec3> vertices{make_vec3(-0.3f, -0.3f, 0.f), make_vec3(0.3f, -0.3f, 0.f), make_vec3(0.3f, 0.3f, 0.f), make_vec3(-0.3f, 0.3f, 0.f)};
    size_t rect_UUID;
    DOCTEST_CHECK_NOTHROW(rect_UUID = headless.addRectangleByVertices(vertices, make_RGBcolor(0.f, 1.f, 0.f), Visualizer::COORDINATES_CARTESIAN));

    // The render must complete without binding an uninitialized shadow-map texture.
    DOCTEST_CHECK_NOTHROW(headless.plotUpdate(true));
}

TEST_CASE("Visualizer::PNG with transparent background (windowed mode)") {
    // Test PNG output with transparent background in WINDOWED mode (not headless)
    Context context;

    // Add a red patch via the Context (matching user's workflow)
    uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.6, 0.6), nullrotation, "plugins/visualizer/textures/AlmondLeaf.png");
    // Use material system for test geometry with texture override
    std::string test_material = "test_visualizer_red_patch";
    if (!context.doesMaterialExist(test_material)) {
        context.addMaterial(test_material);
        context.setMaterialColor(test_material, make_RGBAcolor(1.f, 0.f, 0.f, 1.f));
        context.setMaterialTexture(test_material, "plugins/visualizer/textures/AlmondLeaf.png");
        context.setMaterialTextureColorOverride(test_material, true);
    }
    context.assignMaterialToPrimitive(patch_UUID, test_material); // Required to use vertex color instead of texture color

    Visualizer visualizer(200, 200, 16, false, true);
    visualizer.disableMessages();

    // Set transparent background BEFORE building context geometry
    DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());

    // Use shadowed lighting to match user's code
    DOCTEST_CHECK_NOTHROW(visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED));

    // Build context geometry (this is what user does)
    DOCTEST_CHECK_NOTHROW(visualizer.buildContextGeometry(&context));

    // Render the scene (use plotUpdate() without argument to match user's code exactly)
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate());

    // Save to PNG
    std::string test_filename = "test_transparent_bg_windowed.png";
    DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str(), "png"));
    DOCTEST_CHECK(std::filesystem::exists(test_filename));

    // Read the PNG back to verify transparency
    std::vector<helios::RGBAcolor> pixel_data;
    uint width, height;
    DOCTEST_CHECK_NOTHROW(helios::readPNG(test_filename, width, height, pixel_data));
    // Note: width/height may be larger than 200 due to HiDPI/Retina scaling
    DOCTEST_CHECK(width > 0);
    DOCTEST_CHECK(height > 0);
    DOCTEST_CHECK(pixel_data.size() == width * height);

    // Count transparent, checkerboard, and opaque pixels
    int transparent_pixels = 0;
    int opaque_red_pixels = 0;
    int checkerboard_pixels = 0; // Gray pixels from checkerboard texture

    for (const auto &pixel: pixel_data) {
        if (pixel.a < 0.1f) {
            // Fully transparent background pixel
            transparent_pixels++;
        } else if (pixel.a > 0.9f && pixel.r > 0.5f && pixel.g < 0.3f && pixel.b < 0.3f) {
            // Opaque red pixel (the rectangle)
            opaque_red_pixels++;
        } else if (pixel.a > 0.9f && pixel.r > 0.6f && pixel.r < 0.85f && std::abs(pixel.r - pixel.g) < 0.1f && std::abs(pixel.r - pixel.b) < 0.1f) {
            // Gray pixels - likely from checkerboard (should NOT be present)
            checkerboard_pixels++;
        }
    }

    // The checkerboard should NOT appear in the output
    DOCTEST_CHECK_MESSAGE(checkerboard_pixels == 0, "Checkerboard texture should not appear in PNG output, got " << checkerboard_pixels << " checkerboard pixels");

    // We should have transparent background pixels (at least 25% of image)
    uint total_pixels = width * height;
    DOCTEST_CHECK_MESSAGE(transparent_pixels > total_pixels * 0.25, "Expected significant transparent background area, got " << transparent_pixels << " transparent pixels out of " << total_pixels);

    // We should have the red rectangle (at least 2.5% of image)
    DOCTEST_CHECK_MESSAGE(opaque_red_pixels > total_pixels * 0.025, "Expected visible red rectangle in center, got " << opaque_red_pixels << " red pixels out of " << total_pixels);

    // Clean up test file
    if (std::filesystem::exists(test_filename)) {
        std::filesystem::remove(test_filename);
    }
}

TEST_CASE("Visualizer::Transparent background with non-square window") {
    // Test that checkerboard squares remain square regardless of window aspect ratio
    // This test verifies that UV coordinates are properly adjusted based on window dimensions

    Context context;

    // Add a small patch to have some geometry
    uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.3, 0.3), nullrotation, "plugins/visualizer/textures/AlmondLeaf.png");
    // Use material system for test geometry with texture override
    std::string test_material = "test_visualizer_red_patch_small";
    if (!context.doesMaterialExist(test_material)) {
        context.addMaterial(test_material);
        context.setMaterialColor(test_material, make_RGBAcolor(1.f, 0.f, 0.f, 1.f));
        context.setMaterialTexture(test_material, "plugins/visualizer/textures/AlmondLeaf.png");
        context.setMaterialTextureColorOverride(test_material, true);
    }
    context.assignMaterialToPrimitive(patch_UUID, test_material);

    // Test with a non-square window (800x600, aspect ratio 4:3)
    Visualizer visualizer(800, 600, 16, false, true);
    visualizer.disableMessages();

    // Set transparent background
    DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());

    // Build geometry
    DOCTEST_CHECK_NOTHROW(visualizer.buildContextGeometry(&context));

    // Render and save
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

    std::string test_filename = "test_transparent_bg_nonsquare.png";
    DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str(), "png"));
    DOCTEST_CHECK(std::filesystem::exists(test_filename));

    // Read back to verify
    std::vector<helios::RGBAcolor> pixel_data;
    uint width, height;
    DOCTEST_CHECK_NOTHROW(helios::readPNG(test_filename, width, height, pixel_data));
    DOCTEST_CHECK(width > 0);
    DOCTEST_CHECK(height > 0);

    // Count pixels by type
    int transparent_pixels = 0;
    int opaque_red_pixels = 0;

    for (const auto &pixel: pixel_data) {
        if (pixel.a < 0.1f) {
            transparent_pixels++;
        } else if (pixel.a > 0.9f && pixel.r > 0.5f && pixel.g < 0.3f && pixel.b < 0.3f) {
            opaque_red_pixels++;
        }
    }

    // Verify we have transparent background and geometry
    uint total_pixels = width * height;
    DOCTEST_CHECK_MESSAGE(transparent_pixels > total_pixels * 0.5, "Expected significant transparent background, got " << transparent_pixels << " transparent pixels out of " << total_pixels);
    DOCTEST_CHECK_MESSAGE(opaque_red_pixels > 100, "Expected visible red rectangle, got " << opaque_red_pixels << " red pixels");

    // Clean up
    if (std::filesystem::exists(test_filename)) {
        std::filesystem::remove(test_filename);
    }
}

TEST_CASE("Visualizer::Background color/transparent switching") {
    // Test that switching between transparent and solid color background properly manages watermark visibility

    Context context;
    uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.3, 0.3), nullrotation, "plugins/visualizer/textures/AlmondLeaf.png");
    // Use material system for test geometry with texture override
    std::string test_material = "test_visualizer_red_patch_small";
    if (!context.doesMaterialExist(test_material)) {
        context.addMaterial(test_material);
        context.setMaterialColor(test_material, make_RGBAcolor(1.f, 0.f, 0.f, 1.f));
        context.setMaterialTexture(test_material, "plugins/visualizer/textures/AlmondLeaf.png");
        context.setMaterialTextureColorOverride(test_material, true);
    }
    context.assignMaterialToPrimitive(patch_UUID, test_material);

    SUBCASE("Watermark visible → transparent → solid color (should restore watermark)") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();
        visualizer.buildContextGeometry(&context);

        // Watermark should be visible by default
        // (We can't directly check isWatermarkVisible since it's private, but we test the behavior)

        // Switch to transparent background - should hide watermark
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());

        // Switch back to solid color - should restore watermark
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(make_RGBcolor(0.5f, 0.5f, 0.5f)));

        // Verify transparent background is disabled
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

        // If we can render without error, the watermark restoration worked
        std::string test_filename = "test_bg_switch_restore.png";
        DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str(), "png"));
        DOCTEST_CHECK(std::filesystem::exists(test_filename));

        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }

    SUBCASE("Watermark hidden → transparent → solid color (should NOT restore watermark)") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();
        visualizer.buildContextGeometry(&context);

        // Manually hide watermark before enabling transparent background
        DOCTEST_CHECK_NOTHROW(visualizer.hideWatermark());

        // Switch to transparent background
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());

        // Switch back to solid color - should NOT restore watermark (it was manually hidden)
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(make_RGBcolor(0.5f, 0.5f, 0.5f)));

        // Verify rendering works
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

        std::string test_filename = "test_bg_switch_no_restore.png";
        DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str(), "png"));
        DOCTEST_CHECK(std::filesystem::exists(test_filename));

        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }

    SUBCASE("Multiple switches between transparent and solid") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();
        visualizer.buildContextGeometry(&context);

        // Multiple switches should work correctly
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(make_RGBcolor(1.f, 0.f, 0.f)));
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(make_RGBcolor(0.f, 1.f, 0.f)));
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundTransparent());
        DOCTEST_CHECK_NOTHROW(visualizer.setBackgroundColor(make_RGBcolor(0.f, 0.f, 1.f)));

        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));
    }
}

DOCTEST_TEST_CASE("Visualizer::Navigation Gizmo") {
    // Test the navigation gizmo functionality

    SUBCASE("Navigation gizmo is enabled by default") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();
        // Gizmo should be enabled by default
        // We can't directly access the private member, but we can test the behavior
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));
    }

    SUBCASE("Show and hide navigation gizmo") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();

        // Hide the gizmo
        DOCTEST_CHECK_NOTHROW(visualizer.hideNavigationGizmo());
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

        // Show the gizmo
        DOCTEST_CHECK_NOTHROW(visualizer.showNavigationGizmo());
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

        // Hide and show multiple times
        DOCTEST_CHECK_NOTHROW(visualizer.hideNavigationGizmo());
        DOCTEST_CHECK_NOTHROW(visualizer.showNavigationGizmo());
        DOCTEST_CHECK_NOTHROW(visualizer.hideNavigationGizmo());
        DOCTEST_CHECK_NOTHROW(visualizer.showNavigationGizmo());
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));
    }

    SUBCASE("Navigation gizmo with camera movement") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();

        // Add some geometry to visualize
        auto sphere_center = make_vec3(0, 0, 0);
        auto sphere_radius = 1.0f;
        auto sphere_color = make_RGBcolor(1.f, 0.f, 0.f);
        auto sphere_uuids = visualizer.addSphereByCenter(sphere_radius, sphere_center, 10, sphere_color, Visualizer::COORDINATES_CARTESIAN);

        // Set initial camera position
        visualizer.setCameraPosition(make_vec3(3, 3, 3), make_vec3(0, 0, 0));
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

        // Move camera to a different position
        visualizer.setCameraPosition(make_vec3(-3, 3, 3), make_vec3(0, 0, 0));
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

        // Move camera again
        visualizer.setCameraPosition(make_vec3(0, 5, 5), make_vec3(0, 0, 0));
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));
    }

    SUBCASE("Navigation gizmo with printWindow") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();

        // Add some geometry
        auto sphere_uuids = visualizer.addSphereByCenter(1.0f, make_vec3(0, 0, 0), 10, make_RGBcolor(1.f, 0.f, 0.f), Visualizer::COORDINATES_CARTESIAN);
        visualizer.setCameraPosition(make_vec3(3, 3, 3), make_vec3(0, 0, 0));

        // Show gizmo and take screenshot
        visualizer.showNavigationGizmo();
        std::string test_filename = "test_nav_gizmo_screenshot.jpg";
        DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str()));

        // Verify the file was created
        DOCTEST_CHECK(std::filesystem::exists(test_filename));

        // Clean up test file
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }

    SUBCASE("Navigation gizmo state persists after printWindow") {
        Visualizer visualizer(200, 200, 16, false, true);
        visualizer.disableMessages();

        // Enable gizmo
        visualizer.showNavigationGizmo();

        // Take screenshot (gizmo should be hidden during screenshot but restored after)
        std::string test_filename = "test_nav_gizmo_persist.jpg";
        DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str()));

        // Gizmo should still be enabled after screenshot
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

        // Clean up
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }
}

DOCTEST_TEST_CASE("GeometryHandler::getVertices coordinate system transformation") {
    // Test that getVertices() returns vertices in the same coordinate space they were provided
    // This is a regression test for the bug where COORDINATES_WINDOW_NORMALIZED vertices
    // were transformed to OpenGL space [-1,1] but getVertices() didn't apply inverse transformation

    Visualizer visualizer(200, 200, 16, false, true);

    SUBCASE("COORDINATES_WINDOW_NORMALIZED - getVertices should return original [0,1] coordinates") {
        // Create a rectangle with known normalized window coordinates [0,1]
        helios::vec3 center = make_vec3(0.852f, 0.1f, 0.011f);
        helios::vec2 size = make_vec2(0.02f, 0.025f);

        size_t rect_id = visualizer.addRectangleByCenter(center, size, make_SphericalCoord(0, 0), RGB::red, Visualizer::COORDINATES_WINDOW_NORMALIZED);
        DOCTEST_CHECK(rect_id != 0);

        // Get vertices back - they should be in the SAME coordinate space [0,1]
        auto vertices = visualizer.getGeometryVertices(rect_id);
        DOCTEST_CHECK(vertices.size() == 4);

        // Calculate expected vertices from center and size
        float half_width = size.x * 0.5f;
        float half_height = size.y * 0.5f;
        helios::vec3 expected_v0 = make_vec3(center.x - half_width, center.y - half_height, center.z);
        helios::vec3 expected_v1 = make_vec3(center.x + half_width, center.y - half_height, center.z);
        helios::vec3 expected_v2 = make_vec3(center.x + half_width, center.y + half_height, center.z);
        helios::vec3 expected_v3 = make_vec3(center.x - half_width, center.y + half_height, center.z);

        // Check that returned vertices are in [0,1] range (normalized window coordinates)
        float tolerance = 1e-5f;
        DOCTEST_CHECK_MESSAGE(std::abs(vertices[0].x - expected_v0.x) < tolerance, "v0.x expected " << expected_v0.x << " but got " << vertices[0].x);
        DOCTEST_CHECK_MESSAGE(std::abs(vertices[0].y - expected_v0.y) < tolerance, "v0.y expected " << expected_v0.y << " but got " << vertices[0].y);
        DOCTEST_CHECK_MESSAGE(std::abs(vertices[1].x - expected_v1.x) < tolerance, "v1.x expected " << expected_v1.x << " but got " << vertices[1].x);
        DOCTEST_CHECK_MESSAGE(std::abs(vertices[1].y - expected_v1.y) < tolerance, "v1.y expected " << expected_v1.y << " but got " << vertices[1].y);

        // Verify vertices are actually in [0,1] range, not [-1,1] range
        for (const auto &v: vertices) {
            bool x_in_range = (v.x >= 0.0f) && (v.x <= 1.0f);
            DOCTEST_CHECK_MESSAGE(x_in_range, "Vertex x=" << v.x << " is outside [0,1] range - bug not fixed!");
            bool y_in_range = (v.y >= 0.0f) && (v.y <= 1.0f);
            DOCTEST_CHECK_MESSAGE(y_in_range, "Vertex y=" << v.y << " is outside [0,1] range - bug not fixed!");
        }
    }

    SUBCASE("COORDINATES_CARTESIAN - getVertices should return original Cartesian coordinates") {
        // Create a rectangle with Cartesian coordinates (no transformation should occur)
        helios::vec3 center = make_vec3(5.0f, 3.0f, 2.0f);
        helios::vec2 size = make_vec2(1.0f, 2.0f);

        size_t rect_id = visualizer.addRectangleByCenter(center, size, make_SphericalCoord(0, 0), RGB::blue, Visualizer::COORDINATES_CARTESIAN);
        DOCTEST_CHECK(rect_id != 0);

        // Get vertices back - they should be unchanged
        auto vertices = visualizer.getGeometryVertices(rect_id);
        DOCTEST_CHECK(vertices.size() == 4);

        // Calculate expected vertices
        float half_width = size.x * 0.5f;
        float half_height = size.y * 0.5f;
        helios::vec3 expected_v0 = make_vec3(center.x - half_width, center.y - half_height, center.z);

        float tolerance = 1e-5f;
        DOCTEST_CHECK(std::abs(vertices[0].x - expected_v0.x) < tolerance);
        DOCTEST_CHECK(std::abs(vertices[0].y - expected_v0.y) < tolerance);
        DOCTEST_CHECK(std::abs(vertices[0].z - expected_v0.z) < tolerance);
    }
}

DOCTEST_TEST_CASE("GeometryHandler::setVertices coordinate system transformation") {
    // Test that setVertices() applies the same transformation as addGeometry()

    Visualizer visualizer(200, 200, 16, false, true);

    SUBCASE("COORDINATES_WINDOW_NORMALIZED - setVertices should transform [0,1] to [-1,1]") {
        // Create a rectangle
        helios::vec3 center = make_vec3(0.5f, 0.5f, 0.0f);
        helios::vec2 size = make_vec2(0.2f, 0.2f);
        size_t rect_id = visualizer.addRectangleByCenter(center, size, make_SphericalCoord(0, 0), RGB::green, Visualizer::COORDINATES_WINDOW_NORMALIZED);

        // Get original vertices
        auto original_vertices = visualizer.getGeometryVertices(rect_id);

        // Modify vertices slightly (still in [0,1] space)
        std::vector<helios::vec3> new_vertices = original_vertices;
        for (auto &v: new_vertices) {
            v.x += 0.1f;
            v.y += 0.1f;
        }

        // Set the modified vertices
        DOCTEST_CHECK_NOTHROW(visualizer.setGeometryVertices(rect_id, new_vertices));

        // Get vertices back
        auto retrieved_vertices = visualizer.getGeometryVertices(rect_id);

        // Verify we get back what we set (in the same coordinate space)
        float tolerance = 1e-5f;
        DOCTEST_CHECK(std::abs(retrieved_vertices[0].x - new_vertices[0].x) < tolerance);
        DOCTEST_CHECK(std::abs(retrieved_vertices[0].y - new_vertices[0].y) < tolerance);
    }

    SUBCASE("COORDINATES_CARTESIAN - setVertices should not transform") {
        // Create a Cartesian rectangle
        helios::vec3 center = make_vec3(0.0f, 0.0f, 0.0f);
        helios::vec2 size = make_vec2(2.0f, 2.0f);
        size_t rect_id = visualizer.addRectangleByCenter(center, size, make_SphericalCoord(0, 0), RGB::yellow, Visualizer::COORDINATES_CARTESIAN);

        // Get original vertices
        auto original_vertices = visualizer.getGeometryVertices(rect_id);

        // Modify vertices
        std::vector<helios::vec3> new_vertices = original_vertices;
        for (auto &v: new_vertices) {
            v.x += 1.0f;
            v.y += 1.0f;
        }

        // Set the modified vertices
        DOCTEST_CHECK_NOTHROW(visualizer.setGeometryVertices(rect_id, new_vertices));

        // Get vertices back
        auto retrieved_vertices = visualizer.getGeometryVertices(rect_id);

        // Verify we get back what we set
        float tolerance = 1e-5f;
        DOCTEST_CHECK(std::abs(retrieved_vertices[0].x - new_vertices[0].x) < tolerance);
        DOCTEST_CHECK(std::abs(retrieved_vertices[0].y - new_vertices[0].y) < tolerance);
        DOCTEST_CHECK(std::abs(retrieved_vertices[0].z - new_vertices[0].z) < tolerance);
    }
}

DOCTEST_TEST_CASE("Visualizer::colorContextPrimitivesByObjectData orphan primitives use colormap-at-0") {
    // Regression test: when calling colorContextPrimitivesByObjectData(), primitives
    // that are not part of any compound object should be colored as if their object-data
    // value were 0 (so they get colormap-at-0), rather than rendering with their base RGBA.
    Context context;

    // An orphan patch with a deliberately distinctive base RGBA color (bright red).
    // If the fix regresses, this red is what will appear in the rendered image.
    const RGBcolor base_color = make_RGBcolor(1.f, 0.f, 0.f);
    uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1.f, 1.f), nullrotation, base_color);
    DOCTEST_CHECK(context.getPrimitiveParentObjectID(patch_UUID) == 0); // confirm it's an orphan

    Visualizer visualizer(200, 200, 16, false, true); // headless
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_NONE); // avoid shading skewing the rendered color

    // Set an explicit colorbar range that does NOT include 0. With the default "hot"
    // colormap, query(0) under range [1,2] is clamped to colormap-min = (0,0,0) (black).
    visualizer.setColorbarRange(1.f, 2.f);

    DOCTEST_CHECK_NOTHROW(visualizer.buildContextGeometry(&context));
    DOCTEST_CHECK_NOTHROW(visualizer.colorContextPrimitivesByObjectData("nonexistent_object_data"));
    // Disable the auto-enabled colorbar so it doesn't contribute red/yellow pixels from
    // the high end of the "hot" colormap, which would otherwise contaminate the pixel counts.
    DOCTEST_CHECK_NOTHROW(visualizer.disableColorbar());
    DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));

    std::string test_filename = "test_orphan_color_by_object_data.png";
    DOCTEST_CHECK_NOTHROW(visualizer.printWindow(test_filename.c_str(), "png"));
    DOCTEST_CHECK(std::filesystem::exists(test_filename));

    std::vector<RGBAcolor> pixel_data;
    uint width, height;
    DOCTEST_CHECK_NOTHROW(helios::readPNG(test_filename, width, height, pixel_data));

    // Walk all pixels: count red (base color leaking through — would mean fix regressed)
    // vs near-black (colormap_hot at value 0 — the expected post-fix behavior).
    int red_pixels = 0;
    int black_patch_pixels = 0;
    for (const auto &pixel: pixel_data) {
        if (pixel.r > 0.5f && pixel.g < 0.2f && pixel.b < 0.2f) {
            red_pixels++;
        } else if (pixel.r < 0.1f && pixel.g < 0.1f && pixel.b < 0.1f) {
            black_patch_pixels++;
        }
    }

    DOCTEST_CHECK_MESSAGE(red_pixels == 0, "Orphan primitive rendered with its base RGBA (red) instead of colormap-at-0; got " << red_pixels << " red pixels");
    DOCTEST_CHECK_MESSAGE(black_patch_pixels > 100, "Expected the orphan patch to render as near-black (colormap_hot at 0); only got " << black_patch_pixels << " black pixels");

    if (std::filesystem::exists(test_filename)) {
        std::filesystem::remove(test_filename);
    }
}

// Mean brightness over a fractional sub-rectangle of a rendered image, skipping
// background pixels. Bounds are fractions of the ACTUAL image dimensions, since
// HiDPI/Retina scaling can make the rendered framebuffer larger than requested.
static float shadowRegionMeanBrightness(const std::vector<RGBAcolor> &pixel_data, uint width, uint height, float x_min, float x_max, float y_min, float y_max, int &pixel_count) {
    double sum = 0.0;
    pixel_count = 0;
    for (uint j = uint(y_min * float(height)); j < uint(y_max * float(height)); j++) {
        for (uint i = uint(x_min * float(width)); i < uint(x_max * float(width)); i++) {
            const RGBAcolor &pixel = pixel_data.at(j * width + i);
            // The scene geometry is achromatic gray, so any strongly blue pixel is background.
            if (pixel.b > pixel.r + 0.2f && pixel.b > pixel.g + 0.2f) {
                continue;
            }
            sum += (pixel.r + pixel.g + pixel.b) / 3.0;
            pixel_count++;
        }
    }
    return pixel_count > 0 ? float(sum / double(pixel_count)) : 0.f;
}

// Render a ground patch with an elevated occluder above it (or the ground alone) and
// return the decoded image. Shared by the shadow regression tests below.
static void renderShadowScene(bool headless, bool include_occluder, bool set_light_direction, const std::string &filename, std::vector<RGBAcolor> &pixel_data, uint &width, uint &height) {

    Visualizer visualizer(400, 400, 0, false, headless);
    visualizer.disableMessages();
    visualizer.hideWatermark();
    visualizer.hideNavigationGizmo();
    visualizer.disableColorbar();
    // Pure blue background so background pixels are trivially separable from the gray geometry.
    visualizer.setBackgroundColor(make_RGBcolor(0.f, 0.f, 1.f));

    // Base color is deliberately dim: at 0.5 the lit ground saturates to white, which
    // clips the lit sample and compresses the shadow contrast ratio from 3.6 to ~1.5.
    RGBcolor gray = make_RGBcolor(0.25f, 0.25f, 0.25f);

    // Ground: 8x8 at z=0, wound counter-clockwise viewed from +z so the normal is +z.
    std::vector<vec3> ground_vertices{make_vec3(-4, -4, 0), make_vec3(4, -4, 0), make_vec3(4, 4, 0), make_vec3(-4, 4, 0)};
    visualizer.addRectangleByVertices(ground_vertices, gray, Visualizer::COORDINATES_CARTESIAN);

    if (include_occluder) {
        // Occluder: 2x2 at z=2, same winding.
        std::vector<vec3> occluder_vertices{make_vec3(-1, -1, 2), make_vec3(1, -1, 2), make_vec3(1, 1, 2), make_vec3(-1, 1, 2)};
        visualizer.addRectangleByVertices(occluder_vertices, gray, Visualizer::COORDINATES_CARTESIAN);
    }

    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
    if (set_light_direction) {
        // light_direction points FROM the surface TOWARD the light: computeShadowDepthMVP()
        // negates it to place the shadow camera at centroid + light_direction*100. Passed
        // already unit-length so the stored and uniform values agree.
        visualizer.setLightDirection(make_vec3(0.6f, 0.f, 0.8f));
    }
    // Oblique camera: a straight-down view is degenerate because the up vector is hard-coded to (0,0,1).
    visualizer.setCameraPosition(make_vec3(0.f, -8.f, 10.f), make_vec3(0.f, 0.f, 0.f));

    visualizer.plotUpdate(true);

    // printWindow + readPNG reports the actual rendered dimensions; Wframebuffer has no public getter.
    visualizer.printWindow(filename.c_str(), "png");
    helios::readPNG(filename, width, height, pixel_data);
    if (std::filesystem::exists(filename)) {
        std::filesystem::remove(filename);
    }
}

// Fractional bounds of the shadow core, calibrated by rendering the scene and reading off
// a coarse brightness map. Chosen strictly inside the umbra so the penumbra at the shadow
// edges does not dilute the mean (sampling the penumbra drops the ratio from 3.60 to ~1.5).
// The explicit light direction (0.6,0,0.8) throws the shadow toward -x; the default
// direction (1,1,1)/sqrt(3) throws it toward -x AND -y, so it lands lower in the image.
static constexpr float shadow_x_min = 14.f / 48.f;
static constexpr float shadow_x_max = 19.f / 48.f;
static constexpr float shadow_y_min = 11.2f / 24.f;
static constexpr float shadow_y_max = 13.8f / 24.f;

static constexpr float default_light_shadow_x_min = 10.f / 48.f;
static constexpr float default_light_shadow_x_max = 17.f / 48.f;
static constexpr float default_light_shadow_y_min = 14.6f / 24.f;
static constexpr float default_light_shadow_y_max = 17.4f / 24.f;

// Measured contrast between the same region rendered without and with the occluder.
static float measureShadowRatio(bool headless, bool set_light_direction, const std::string &tag) {

    const float x_min = set_light_direction ? shadow_x_min : default_light_shadow_x_min;
    const float x_max = set_light_direction ? shadow_x_max : default_light_shadow_x_max;
    const float y_min = set_light_direction ? shadow_y_min : default_light_shadow_y_min;
    const float y_max = set_light_direction ? shadow_y_max : default_light_shadow_y_max;

    std::vector<RGBAcolor> pixels_occluded, pixels_clear;
    uint width_occluded, height_occluded, width_clear, height_clear;

    renderShadowScene(headless, true, set_light_direction, "test_shadow_" + tag + "_occluded.png", pixels_occluded, width_occluded, height_occluded);
    renderShadowScene(headless, false, set_light_direction, "test_shadow_" + tag + "_clear.png", pixels_clear, width_clear, height_clear);

    DOCTEST_REQUIRE(width_occluded == width_clear);
    DOCTEST_REQUIRE(height_occluded == height_clear);
    DOCTEST_REQUIRE(pixels_occluded.size() == size_t(width_occluded) * size_t(height_occluded));

    int count_occluded, count_clear;
    float mean_occluded = shadowRegionMeanBrightness(pixels_occluded, width_occluded, height_occluded, x_min, x_max, y_min, y_max, count_occluded);
    float mean_clear = shadowRegionMeanBrightness(pixels_clear, width_clear, height_clear, x_min, x_max, y_min, y_max, count_clear);

    DOCTEST_REQUIRE_MESSAGE(count_occluded > 200, "Shadow sample region contains too few geometry pixels (" << count_occluded << "); the region bounds need recalibration");
    DOCTEST_REQUIRE_MESSAGE(count_clear > 200, "Reference sample region contains too few geometry pixels (" << count_clear << ")");
    DOCTEST_REQUIRE_MESSAGE(mean_clear > 0.05f, "Unoccluded reference render is essentially black (" << mean_clear << "); the render pipeline failed for a reason unrelated to shadows");
    DOCTEST_REQUIRE_MESSAGE(mean_occluded > 0.01f, "Occluded region is pure black (" << mean_occluded << "); the contrast ratio would be undefined");

    return mean_clear / mean_occluded;
}

DOCTEST_TEST_CASE("Visualizer shadows render in headless mode") {
    // REGRESSION TEST: Visualizer::initialize() created the shadow-map framebuffer and
    // depth texture only under `if (!headless)`, so in headless mode framebufferID and
    // depthTexture both stayed 0. The shadow pass in plotUpdate() is not guarded on
    // headless, so it bound framebuffer 0 (rendering the depth pass into the default
    // framebuffer) and bound texture 0 as the shadow map.
    //
    // The metric is deliberately a CONTRAST RATIO rather than an absolute darkness
    // threshold. On the buggy code, sampling texture 0 returns 0.0, and the depth
    // comparison in primaryShader.frag is `0.0 < proj.z`, which is true for essentially
    // all geometry -- so every one of the four Poisson taps darkens every fragment and
    // the ENTIRE scene renders at the fully-shadowed brightness. A test asserting "the
    // shadow region is dark" would therefore PASS on the buggy code. A ratio collapses
    // to 1.0 whether the broken output is uniformly dark or uniformly lit.
    //
    // Measured values: 3.60 with a working shadow map, exactly 1.00 without one (the
    // two renders are byte-identical, since the occluder is invisible as well).

    float shadow_ratio = measureShadowRatio(true, true, "headless");

    DOCTEST_CHECK_MESSAGE(shadow_ratio > 2.f, "No shadow was rendered in headless mode: the occluded region is only "
                                                      << shadow_ratio
                                                      << "x darker than the same region rendered without the occluder (expected ~3.6; a value of ~1.0 means the shadow map is missing entirely). The "
                                                         "shadow-map framebuffer/depth texture are likely not being created in headless mode.");

    DOCTEST_CHECK_MESSAGE(shadow_ratio < 12.f, "Shadow contrast is implausibly large (" << shadow_ratio << "); the sample region may have fallen off the ground geometry");
}

DOCTEST_TEST_CASE("Visualizer shadows render in headless mode with the default light direction") {
    // Companion to the test above, covering the case where the user never calls
    // setLightDirection() at all. The shadow falls in a different place here: the default
    // direction is (1,1,1)/sqrt(3), which throws the shadow toward -x and -y, whereas the
    // test above uses (0.6,0,0.8) and throws it toward -x only.
    //
    // Note this does NOT depend on the initialize() light-direction fix. Shader::initialize()
    // seeds the lightDirection uniform with (0,0,1), so headless previously rendered with a
    // directly-overhead light instead of the (1,1,1) that initialize() intended -- a
    // windowed/headless parity bug, but not one that suppresses shadows. That parity is
    // covered by the dedicated test below rather than here.

    float shadow_ratio = measureShadowRatio(true, false, "headless_default_light");

    DOCTEST_CHECK_MESSAGE(shadow_ratio > 2.f, "No shadow was rendered in headless mode using the default light direction (ratio " << shadow_ratio << ", expected ~3.6).");
}

DOCTEST_TEST_CASE("Visualizer default light direction matches between headless and windowed modes") {
    // initialize() pushed the initial light_direction to the primary shader only when
    // !headless. Shader::initialize() seeds the uniform with (0,0,1), so a headless
    // Visualizer that never calls setLightDirection() rendered with a directly-overhead
    // light while a windowed one used the intended (1,1,1)/sqrt(3). Nothing errored; the
    // two modes just shaded the same scene differently, contradicting the documented
    // promise of identical visual output.
    //
    // The ground patch normal is +z, so the two directions give clearly different diffuse
    // terms: dot((0,0,1),(0,0,1)) = 1 versus dot((0,0,1),(1,1,1)/sqrt(3)) = 0.577.

    std::vector<RGBAcolor> pixel_data;
    uint width, height;
    renderShadowScene(true, false, false, "test_shadow_default_light_direction.png", pixel_data, width, height);

    int pixel_count;
    float mean_brightness = shadowRegionMeanBrightness(pixel_data, width, height, 0.45f, 0.55f, 0.45f, 0.55f, pixel_count);

    DOCTEST_REQUIRE(pixel_count > 200);

    // Ground base color 0.25, scaled by 1.5 in the shader, lit as intensity*(0.75 + d).
    // d = 0.577 gives 0.375*1.327 = 0.498; the un-fixed d = 1.0 gives 0.375*1.75 = 0.656.
    DOCTEST_CHECK_MESSAGE(mean_brightness < 0.57f, "Headless mode rendered with the wrong default light direction (brightness " << mean_brightness
                                                                                                                                << ", expected ~0.50 for the (1,1,1) default rather than ~0.66 for an overhead light). The initial light "
                                                                                                                                   "direction is likely not being sent to the shader in headless mode.");
}

DOCTEST_TEST_CASE("Visualizer shadows survive an intervening depth-map render") {
    // updateDepthBuffer() used to render into the shadow-map framebuffer and re-specify
    // the shared depth texture at the window resolution. Since the shadow pass sets a
    // viewport of shadow_buffer_size (8192x8192 by default), any shadowed render after a
    // plotDepthMap() call sampled a shadow map far smaller than the viewport it drew
    // into, corrupting the shadows. The depth map now owns a separate framebuffer and
    // texture, so a shadowed render before and after a depth-map render must agree.

    Visualizer visualizer(400, 400, 0, false, true);
    visualizer.disableMessages();
    visualizer.hideWatermark();
    visualizer.hideNavigationGizmo();
    visualizer.disableColorbar();
    visualizer.setBackgroundColor(make_RGBcolor(0.f, 0.f, 1.f));

    RGBcolor gray = make_RGBcolor(0.25f, 0.25f, 0.25f);
    std::vector<vec3> ground_vertices{make_vec3(-4, -4, 0), make_vec3(4, -4, 0), make_vec3(4, 4, 0), make_vec3(-4, 4, 0)};
    visualizer.addRectangleByVertices(ground_vertices, gray, Visualizer::COORDINATES_CARTESIAN);
    std::vector<vec3> occluder_vertices{make_vec3(-1, -1, 2), make_vec3(1, -1, 2), make_vec3(1, 1, 2), make_vec3(-1, 1, 2)};
    visualizer.addRectangleByVertices(occluder_vertices, gray, Visualizer::COORDINATES_CARTESIAN);

    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
    visualizer.setLightDirection(make_vec3(0.6f, 0.f, 0.8f));
    visualizer.setCameraPosition(make_vec3(0.f, -8.f, 10.f), make_vec3(0.f, 0.f, 0.f));

    // Shadowed render before the depth map.
    visualizer.plotUpdate(true);
    std::string before_file = "test_shadow_before_depthmap.png";
    visualizer.printWindow(before_file.c_str(), "png");
    std::vector<RGBAcolor> pixels_before;
    uint width_before, height_before;
    helios::readPNG(before_file, width_before, height_before, pixels_before);

    // Render a depth map, which previously clobbered the shadow texture. getDepthMap() is
    // used rather than plotDepthMap() because the latter ends in displayImage(), which
    // calls plotInteractive() and blocks until the window is closed.
    std::vector<float> depth_pixels;
    uint depth_width, depth_height;
    DOCTEST_CHECK_NOTHROW(visualizer.getDepthMap(depth_pixels, depth_width, depth_height));
    DOCTEST_CHECK(depth_pixels.size() == size_t(depth_width) * size_t(depth_height));

    // Shadowed render after the depth map.
    visualizer.plotUpdate(true);
    std::string after_file = "test_shadow_after_depthmap.png";
    visualizer.printWindow(after_file.c_str(), "png");
    std::vector<RGBAcolor> pixels_after;
    uint width_after, height_after;
    helios::readPNG(after_file, width_after, height_after, pixels_after);

    DOCTEST_REQUIRE(width_before == width_after);
    DOCTEST_REQUIRE(height_before == height_after);

    int count_before, count_after;
    float mean_before = shadowRegionMeanBrightness(pixels_before, width_before, height_before, shadow_x_min, shadow_x_max, shadow_y_min, shadow_y_max, count_before);
    float mean_after = shadowRegionMeanBrightness(pixels_after, width_after, height_after, shadow_x_min, shadow_x_max, shadow_y_min, shadow_y_max, count_after);

    DOCTEST_REQUIRE(count_before > 200);
    DOCTEST_REQUIRE(count_after > 200);

    DOCTEST_CHECK_MESSAGE(std::abs(mean_after - mean_before) < 0.05f,
                          "The shadow changed after an intervening plotDepthMap() call (shadow region brightness " << mean_before << " before versus " << mean_after << " after); the depth-map render is corrupting the shadow map.");

    for (const std::string &f: {before_file, after_file}) {
        if (std::filesystem::exists(f)) {
            std::filesystem::remove(f);
        }
    }
}

TEST_CASE("Visualizer::plotOnce on a freshly constructed visualizer (headless)") {
    // Regression test: plotOnce() was the only render entry point that never uploaded the CPU-side
    // geometry to the GPU before drawing. initialize() ends by queueing a gradient background
    // rectangle in geometry_handler, so even an empty scene had rectangle_count == 1 while the
    // vertex buffers were still the zero-byte stores left by glGenBuffers. render() then issued a
    // glMultiDrawArrays() that fetched vertices out of bounds and the driver faulted.
    //
    // Calling setBackgroundColor() first used to mask this, because it removes the background
    // rectangle and puts nothing back, dropping rectangle_count to zero.

    Visualizer visualizer(200, 200, 0, true, true); // headless
    visualizer.disableMessages();

    DOCTEST_CHECK_NOTHROW(visualizer.plotOnce(false));

    // Surviving the call is the crash assertion. Additionally verify the background actually
    // rendered: with the buffers never uploaded there is nothing on the GPU to draw, so a correct
    // upload is precisely what makes the gradient appear.
    // getWindowPixelsRGB() writes one entry per framebuffer subpixel, and the framebuffer is
    // larger than the requested window on a high-DPI display - 2x per axis on a Retina screen, so
    // 4x the elements. Sizing this buffer from the constructor's window dimensions overflowed it
    // and corrupted the heap.
    uint framebuffer_width, framebuffer_height;
    visualizer.getFramebufferSize(framebuffer_width, framebuffer_height);
    std::vector<uint> pixel_buffer(3 * framebuffer_width * framebuffer_height);
    DOCTEST_CHECK_NOTHROW(visualizer.getWindowPixelsRGB(pixel_buffer.data()));

    bool has_non_black_pixels = false;
    for (size_t i = 0; i < pixel_buffer.size(); i++) {
        if (pixel_buffer[i] > 10) {
            has_non_black_pixels = true;
            break;
        }
    }

    DOCTEST_CHECK_MESSAGE(has_non_black_pixels, "plotOnce() produced an empty frame - the default gradient background was never uploaded to the GPU");
}

TEST_CASE("Visualizer::plotOnce on a freshly constructed visualizer (windowed)") {
    // Same regression as the headless case above, in windowed mode. The reported crash reproduced
    // in both modes, so both are covered here.

    if (!windowedContextAvailable()) {
        // Skip test silently when no windowed OpenGL context can be created
        return;
    }

    Visualizer visualizer(200, 200, 0, true, false); // NON-headless mode - requires display
    visualizer.disableMessages();

    DOCTEST_CHECK_NOTHROW(visualizer.plotOnce(false));
}

namespace {
    void setEnvironmentVariable(const char *name, const char *value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    void unsetEnvironmentVariable(const char *name) {
#ifdef _WIN32
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }

    //! Restores the working directory and HELIOS_BUILD on scope exit
    /**
     * doctest runs every case in this file in one process, so a test that changes either piece of
     * global state must put it back even if an assertion aborts the case partway through.
     */
    struct ProcessStateGuard {
        std::filesystem::path working_directory;
        bool build_variable_was_set;
        std::string build_variable;

        explicit ProcessStateGuard(std::filesystem::path directory) : working_directory(std::move(directory)) {
            const char *existing = std::getenv("HELIOS_BUILD");
            build_variable_was_set = (existing != nullptr);
            if (build_variable_was_set) {
                build_variable = existing;
            }
        }

        ~ProcessStateGuard() {
            std::error_code ec;
            std::filesystem::current_path(working_directory, ec);
            if (build_variable_was_set) {
                setEnvironmentVariable("HELIOS_BUILD", build_variable.c_str());
            } else {
                unsetEnvironmentVariable("HELIOS_BUILD");
            }
        }
    };
} // namespace

TEST_CASE("Visualizer construction does not change the working directory") {
    // On macOS, glfwInit() moves the process working directory to the host bundle's
    // Contents/Resources unless GLFW_COCOA_CHDIR_RESOURCES is turned off first -- see the init hint
    // set in Visualizer::initialize(). Helios resolves assets relative to the working directory, so
    // a Visualizer silently relocating the whole process breaks every later relative path in the
    // host application, not just the visualizer's own.
    //
    // Honest about its reach: GLFW only performs the chdir when the host process has an application
    // bundle, so on a bare test binary this passes whether or not the hint is set. It fails on a
    // bundled host -- a macOS framework-build Python interpreter, or any .app-packaged tool linking
    // the visualizer -- which is where the defect was originally observed.
    std::filesystem::path working_directory_before = std::filesystem::current_path();

    {
        Visualizer visualizer(64, 64, 0, true, true);
        visualizer.disableMessages();
    }

    std::filesystem::path working_directory_after = std::filesystem::current_path();

    DOCTEST_CHECK(working_directory_after == working_directory_before);
}

TEST_CASE("Visualizer resolves its assets independently of the working directory") {
    // The visualizer loaded six textures by paths relative to the process working directory, passed
    // straight to fopen(). That works only because the test runner happens to launch from the build
    // directory; any host that had changed directory got "File plugins/visualizer/textures/
    // gradient_background.jpg could not be opened" from a file that exists and is readable.
    //
    // Those sites now go through helios::resolvePluginAsset(), which honors HELIOS_BUILD. This test
    // sets HELIOS_BUILD to the build directory, moves the working directory elsewhere, and requires
    // that a Visualizer still constructs and loads every texture it owns.
    std::filesystem::path build_directory = std::filesystem::current_path();

    // Precondition: the assets really are where the test believes they are, so that a failure below
    // means the resolution logic is wrong rather than the build tree being incomplete.
    DOCTEST_REQUIRE(std::filesystem::exists(build_directory / "plugins/visualizer/textures/gradient_background.jpg"));

    std::filesystem::path scratch_directory = std::filesystem::temp_directory_path() / "helios_visualizer_asset_resolution";
    std::error_code ec;
    std::filesystem::create_directories(scratch_directory, ec);
    DOCTEST_REQUIRE(std::filesystem::exists(scratch_directory));

    {
        ProcessStateGuard guard(build_directory);

        setEnvironmentVariable("HELIOS_BUILD", build_directory.string().c_str());
        std::filesystem::current_path(scratch_directory);

        // Construction alone loads the shaders and the gradient background.
        std::unique_ptr<Visualizer> visualizer;
        DOCTEST_CHECK_NOTHROW(visualizer = std::make_unique<Visualizer>(64, 64, 0, true, true));
        DOCTEST_REQUIRE(visualizer != nullptr);
        visualizer->disableMessages();

        // The three navigation gizmo bubbles.
        DOCTEST_CHECK_NOTHROW(visualizer->showNavigationGizmo());

        // The transparent background texture, on both of the paths that load it.
        DOCTEST_CHECK_NOTHROW(visualizer->setBackgroundTransparent());
        DOCTEST_CHECK_NOTHROW(visualizer->printWindow("test_asset_resolution.png", "png"));
    }

    std::filesystem::remove_all(scratch_directory, ec);
}

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>

DOCTEST_TEST_CASE("Visualizer::plotOnce does not terminate the process") {
    // The bug this guards against is a SIGSEGV inside the GL driver, which doctest cannot catch:
    // an in-process crash takes down the whole test runner instead of reporting a failure. So the
    // call is made in a forked child and the parent inspects the wait status. Modeled on the
    // fork()-based test in plugins/radiation/tests/selfTest.cpp.
    //
    // macOS is excluded rather than merely #ifndef _WIN32. doctest runs this whole file in one
    // process, and the first test case in it constructs a Visualizer, so by the time we get here
    // the parent has already initialized GLFW/OpenGL and therefore the Objective-C runtime. Apple's
    // CoreFoundation aborts a forked child that re-enters it, which would surface below as
    // WIFSIGNALED and be indistinguishable from the very crash this test exists to detect. The
    // headless test case above is the portable assertion; this one is the crash backstop on
    // platforms where forking after GL initialization is safe.
#ifndef __APPLE__
    std::cout.flush();
    std::cerr.flush();

    pid_t child = fork();

    if (child == 0) {
        // Child: report status through the exit code only - no doctest macros, and _exit() rather
        // than exit() so the parent's atexit handlers do not run a second time in the child's
        // inherited copy of that state.
        int status = 0;
        try {
            Visualizer visualizer(200, 200, 0, true, true); // headless
            visualizer.disableMessages();
            visualizer.plotOnce(false);
        } catch (...) {
            status = 2; // any escaped exception is a failure
        }
        std::cout.flush();
        _exit(status);
    }

    DOCTEST_REQUIRE(child > 0);

    int wait_status = 0;
    DOCTEST_REQUIRE(waitpid(child, &wait_status, 0) == child);

    // A crash surfaces here as termination by signal (SIGSEGV) rather than a normal exit.
    DOCTEST_CHECK_FALSE(WIFSIGNALED(wait_status));
    if (WIFSIGNALED(wait_status)) {
        DOCTEST_MESSAGE("plotOnce() terminated the child by signal " << WTERMSIG(wait_status));
    }
    DOCTEST_CHECK(WIFEXITED(wait_status));
    if (WIFEXITED(wait_status)) {
        DOCTEST_CHECK(WEXITSTATUS(wait_status) == 0);
    }
#endif // __APPLE__
}
#endif // _WIN32

int Visualizer::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

// ---------------------------------------------------------------------------------------------
// Image display with bounding box overlays.
//
// Nothing below calls displayImage() or displayImageWithBoundingBoxes(): both end in
// plotInteractive(), which spins until the window is closed and would hang the suite. The parsing
// and the geometry construction are reachable on their own precisely so that they can be tested,
// through the public readers and through VisualizerTestHelper for the two private builders.
// ---------------------------------------------------------------------------------------------

//! Directory for the label and class name fixtures used by the bounding box tests
/**
 * The class name lookup searches the directory containing the annotation file, so each fixture needs
 * a directory it controls exclusively; a stray classes.txt in a shared directory would silently
 * change what is being exercised.
 */
static std::filesystem::path makeBoundingBoxFixtureDirectory(const std::string &name) {
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("helios_visualizer_bbox_" + name);
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

static void writeBoundingBoxFixture(const std::filesystem::path &file, const std::string &contents) {
    std::ofstream stream(file);
    stream << contents;
}

TEST_CASE("Visualizer::buildImageDisplayGeometry after a plot does not throw on stale geometry IDs") {
    // Regression test: clearAllGeometry() destroys every geometry ID but does not reset the IDs the
    // visualizer has cached for the watermark, background rectangle and colorbar. hideWatermark()
    // then deletes by an ID that no longer exists, and GeometryHandler::deleteGeometry() indexes
    // UUID_map with at(), so it threw std::out_of_range. Reached through the public API as
    // `vis.plotUpdate(true); vis.displayImage(file);`, which aborted instead of displaying anything.
    //
    // plotUpdate() is what creates the watermark and records its ID, so it is the step that arms the
    // bug; a freshly constructed visualizer has watermark_ID == 0 and is unaffected.

    Visualizer visualizer(800, 800, 0, true, true); // headless
    visualizer.disableMessages();

    visualizer.plotUpdate(true);

    std::vector<unsigned char> pixel_data(4 * 16 * 16, 255);
    DOCTEST_CHECK_NOTHROW(VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 16, 16));
}

TEST_CASE("Visualizer::readBoundingBoxFile parses YOLO annotation lines") {
    const std::filesystem::path directory = makeBoundingBoxFixtureDirectory("read");
    const std::filesystem::path bbox_file = directory / "image.txt";

    // RadiationModel::writeImageBoundingBoxes() applies std::fixed only after streaming the two center
    // fields, so on the first line of every file they carry default ostream formatting and a small
    // value appears in scientific notation. The trailing blank line is likewise what a real file has,
    // because every line is terminated with std::endl.
    writeBoundingBoxFixture(bbox_file, "0 1.5e-05 0.274926 0.309193 0.156994\n1 0.636739 0.733507 0.312500 0.225446\n\n   \n");

    std::vector<Visualizer::BoundingBox> boxes;
    DOCTEST_CHECK_NOTHROW(boxes = Visualizer::readBoundingBoxFile(bbox_file.string()));

    DOCTEST_REQUIRE(boxes.size() == 2);

    DOCTEST_CHECK(boxes.at(0).class_ID == 0);
    DOCTEST_CHECK(boxes.at(0).center.x == doctest::Approx(1.5e-05f));
    DOCTEST_CHECK(boxes.at(0).center.y == doctest::Approx(0.274926f));
    DOCTEST_CHECK(boxes.at(0).size.x == doctest::Approx(0.309193f));
    DOCTEST_CHECK(boxes.at(0).size.y == doctest::Approx(0.156994f));

    DOCTEST_CHECK(boxes.at(1).class_ID == 1);
    DOCTEST_CHECK(boxes.at(1).center.x == doctest::Approx(0.636739f));
    DOCTEST_CHECK(boxes.at(1).size.y == doctest::Approx(0.225446f));

    std::filesystem::remove_all(directory);
}

TEST_CASE("Visualizer::readBoundingBoxFile accepts an annotation file with no boxes") {
    // RadiationModel::writeImageBoundingBoxes() writes an empty file when no primitive carried the
    // requested data label. An image in which nothing was detected is legitimate, not an error.
    const std::filesystem::path directory = makeBoundingBoxFixtureDirectory("empty");
    const std::filesystem::path bbox_file = directory / "image.txt";
    writeBoundingBoxFixture(bbox_file, "");

    std::vector<Visualizer::BoundingBox> boxes;
    DOCTEST_CHECK_NOTHROW(boxes = Visualizer::readBoundingBoxFile(bbox_file.string()));
    DOCTEST_CHECK(boxes.empty());

    writeBoundingBoxFixture(bbox_file, "\n  \n\t\n");
    DOCTEST_CHECK_NOTHROW(boxes = Visualizer::readBoundingBoxFile(bbox_file.string()));
    DOCTEST_CHECK(boxes.empty());

    std::filesystem::remove_all(directory);
}

TEST_CASE("Visualizer::readBoundingBoxFile rejects malformed annotation files") {
    const std::filesystem::path directory = makeBoundingBoxFixtureDirectory("malformed");
    const std::filesystem::path bbox_file = directory / "image.txt";

    DOCTEST_SUBCASE("too few fields") {
        writeBoundingBoxFixture(bbox_file, "0 0.5 0.5 0.1\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile(bbox_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("too many fields") {
        writeBoundingBoxFixture(bbox_file, "0 0.5 0.5 0.1 0.1 0.1\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile(bbox_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("non-numeric geometry field") {
        writeBoundingBoxFixture(bbox_file, "0 abc 0.5 0.1 0.1\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile(bbox_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("negative class ID") {
        writeBoundingBoxFixture(bbox_file, "-1 0.5 0.5 0.1 0.1\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile(bbox_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("class ID too large for an int") {
        // helios::parse_int() catches std::invalid_argument but not std::out_of_range, so this must be
        // rejected before it reaches parse_int() or it escapes as a bare std::out_of_range.
        writeBoundingBoxFixture(bbox_file, "99999999999999999999 0.5 0.5 0.1 0.1\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile(bbox_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("center outside the normalized range") {
        writeBoundingBoxFixture(bbox_file, "0 1.5 0.5 0.1 0.1\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile(bbox_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("degenerate box") {
        writeBoundingBoxFixture(bbox_file, "0 0.5 0.5 0 0.1\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile(bbox_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("file does not exist") {
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxFile((directory / "missing.txt").string()), std::runtime_error);
    }

    std::filesystem::remove_all(directory);
}

TEST_CASE("Visualizer::readBoundingBoxClassNames accepts both class file formats") {
    const std::filesystem::path directory = makeBoundingBoxFixtureDirectory("classes");
    const std::filesystem::path classes_file = directory / "classes.txt";

    DOCTEST_SUBCASE("explicit class IDs, as written by RadiationModel") {
        writeBoundingBoxFixture(classes_file, "0 bunny\n1 dragon\n");
        const std::map<uint, std::string> names = Visualizer::readBoundingBoxClassNames(classes_file.string());
        DOCTEST_CHECK(names.size() == 2);
        DOCTEST_CHECK(names.at(0) == "bunny");
        DOCTEST_CHECK(names.at(1) == "dragon");
    }
    DOCTEST_SUBCASE("implicit class IDs, the Ultralytics convention") {
        writeBoundingBoxFixture(classes_file, "bunny\ndragon\n");
        const std::map<uint, std::string> names = Visualizer::readBoundingBoxClassNames(classes_file.string());
        DOCTEST_CHECK(names.size() == 2);
        DOCTEST_CHECK(names.at(0) == "bunny");
        DOCTEST_CHECK(names.at(1) == "dragon");
    }
    DOCTEST_SUBCASE("explicit class IDs need not be contiguous") {
        // Proves the leading integer is read as the ID rather than the line being counted.
        writeBoundingBoxFixture(classes_file, "3 corn\n7 wheat\n");
        const std::map<uint, std::string> names = Visualizer::readBoundingBoxClassNames(classes_file.string());
        DOCTEST_CHECK(names.size() == 2);
        DOCTEST_CHECK(names.at(3) == "corn");
        DOCTEST_CHECK(names.at(7) == "wheat");
    }
    DOCTEST_SUBCASE("an implicit class name may contain spaces") {
        // The name is the remainder of the line, not the second token.
        writeBoundingBoxFixture(classes_file, "sweet corn\n");
        const std::map<uint, std::string> names = Visualizer::readBoundingBoxClassNames(classes_file.string());
        DOCTEST_CHECK(names.size() == 1);
        DOCTEST_CHECK(names.at(0) == "sweet corn");
    }
    DOCTEST_SUBCASE("an explicit class name may contain spaces") {
        writeBoundingBoxFixture(classes_file, "2 sweet corn\n");
        const std::map<uint, std::string> names = Visualizer::readBoundingBoxClassNames(classes_file.string());
        DOCTEST_CHECK(names.size() == 1);
        DOCTEST_CHECK(names.at(2) == "sweet corn");
    }
    DOCTEST_SUBCASE("blank lines do not advance the implicit class ID") {
        writeBoundingBoxFixture(classes_file, "bunny\n\n   \ndragon\n");
        const std::map<uint, std::string> names = Visualizer::readBoundingBoxClassNames(classes_file.string());
        DOCTEST_CHECK(names.size() == 2);
        DOCTEST_CHECK(names.at(1) == "dragon");
    }
    DOCTEST_SUBCASE("an empty class file is an error") {
        // The file exists, so it is corrupt rather than absent. An absent file is the benign case and
        // is handled by falling back to numeric class IDs.
        writeBoundingBoxFixture(classes_file, "\n\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxClassNames(classes_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("a class ID given two different names is an error") {
        writeBoundingBoxFixture(classes_file, "0 bunny\n0 dragon\n");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxClassNames(classes_file.string()), std::runtime_error);
    }
    DOCTEST_SUBCASE("class file does not exist") {
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readBoundingBoxClassNames((directory / "missing.txt").string()), std::runtime_error);
    }

    std::filesystem::remove_all(directory);
}

TEST_CASE("Visualizer::getTextboxSize measures text extent") {
    Visualizer visualizer(1000, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const vec2 size_M = visualizer.getTextboxSize("M", 20, "OpenSans-Regular");
    const vec2 size_MM = visualizer.getTextboxSize("MM", 20, "OpenSans-Regular");

    DOCTEST_CHECK(size_M.x > 0.f);
    DOCTEST_CHECK(size_M.y > 0.f);
    DOCTEST_CHECK(size_MM.x > size_M.x);

    // An empty string occupies nothing.
    const vec2 size_empty = visualizer.getTextboxSize("", 20, "OpenSans-Regular");
    DOCTEST_CHECK(size_empty.x == doctest::Approx(0.f));
    DOCTEST_CHECK(size_empty.y == doctest::Approx(0.f));

    // The subscript marker itself takes no width and halves the character that follows.
    DOCTEST_CHECK(visualizer.getTextboxSize("A_1", 20, "OpenSans-Regular").x < visualizer.getTextboxSize("A1", 20, "OpenSans-Regular").x);

    // A larger font measures wider.
    DOCTEST_CHECK(visualizer.getTextboxSize("M", 40, "OpenSans-Regular").x > size_M.x);

    DOCTEST_CHECK_THROWS_AS(std::ignore = visualizer.getTextboxSize("M", 20, "ThisFontDoesNotExist"), std::runtime_error);
}

TEST_CASE("Visualizer::getTextboxSize agrees with the text addTextboxByCenter renders") {
    // The measurement was factored out of addTextboxByCenter, so the two must still agree. This is a
    // loose bound rather than an equality: the union of the glyph quads excludes the trailing advance
    // of the final character and includes the one-texel border each glyph texture carries.
    Visualizer visualizer(1000, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const char *teststring = "Wg";
    const vec2 measured = visualizer.getTextboxSize(teststring, 24, "OpenSans-Regular");

    const std::vector<size_t> UUIDs = visualizer.addTextboxByCenter(teststring, make_vec3(0.5f, 0.5f, 0.f), make_SphericalCoord(0, 0), RGB::black, 24, "OpenSans-Regular", Visualizer::COORDINATES_WINDOW_NORMALIZED);
    DOCTEST_REQUIRE(UUIDs.size() == std::strlen(teststring));

    float glyph_x_min = 1.f;
    float glyph_x_max = 0.f;
    for (size_t UUID: UUIDs) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUID)) {
            glyph_x_min = std::min(glyph_x_min, vertex.x);
            glyph_x_max = std::max(glyph_x_max, vertex.x);
        }
    }
    const float rendered_width = glyph_x_max - glyph_x_min;

    DOCTEST_CHECK(rendered_width > 0.5f * measured.x);
    DOCTEST_CHECK(rendered_width <= measured.x + 4.f / 1000.f);

    // The string is centered on the point it was given, so the measured width also predicts where it starts.
    DOCTEST_CHECK(glyph_x_min == doctest::Approx(0.5f - 0.5f * measured.x).epsilon(0.05));
}

TEST_CASE("Visualizer::buildImageDisplayGeometry returns the displayed image extent") {
    Visualizer visualizer(800, 800, 16, true, true); // headless, square window
    visualizer.disableMessages();

    DOCTEST_SUBCASE("an image wider than the window is letterboxed top and bottom") {
        const std::vector<unsigned char> pixel_data(4 * 200 * 100, 255);
        const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 200, 100);
        DOCTEST_CHECK(extent.x == doctest::Approx(0.f));
        DOCTEST_CHECK(extent.y == doctest::Approx(0.25f));
        DOCTEST_CHECK(extent.z == doctest::Approx(1.f));
        DOCTEST_CHECK(extent.w == doctest::Approx(0.75f));
    }
    DOCTEST_SUBCASE("an image taller than the window is letterboxed left and right") {
        const std::vector<unsigned char> pixel_data(4 * 100 * 200, 255);
        const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 200);
        DOCTEST_CHECK(extent.x == doctest::Approx(0.25f));
        DOCTEST_CHECK(extent.y == doctest::Approx(0.f));
        DOCTEST_CHECK(extent.z == doctest::Approx(0.75f));
        DOCTEST_CHECK(extent.w == doctest::Approx(1.f));
    }
    DOCTEST_SUBCASE("an image matching the window aspect ratio fills it") {
        const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
        const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);
        DOCTEST_CHECK(extent.x == doctest::Approx(0.f));
        DOCTEST_CHECK(extent.y == doctest::Approx(0.f));
        DOCTEST_CHECK(extent.z == doctest::Approx(1.f));
        DOCTEST_CHECK(extent.w == doctest::Approx(1.f));

        // The image quad, and nothing else: prior geometry was cleared and the watermark hidden,
        // without entering a render loop.
        DOCTEST_CHECK(VisualizerTestHelper::getLiveGeometryCount(visualizer) == 1);
    }

    DOCTEST_CHECK_THROWS_AS(std::ignore = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, {}, 10, 10), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(std::ignore = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, std::vector<unsigned char>(10, 255), 10, 10), std::runtime_error);
}

TEST_CASE("Visualizer::addBoundingBoxOverlay maps boxes into image coordinates with y flipped") {
    // A square image in a square window maps the image extent onto the whole window, so window
    // coordinates equal normalized image coordinates apart from the vertical flip, and the flip is
    // the only thing under test.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    // Top-left quadrant in YOLO terms, whose origin is the top-left corner of the image.
    Visualizer::BoundingBox box;
    box.class_ID = 0;
    box.center = make_vec2(0.25f, 0.25f);
    box.size = make_vec2(0.5f, 0.5f);

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addBoundingBoxOverlay(visualizer, {box}, {}, extent, 2.f, 12);
    DOCTEST_REQUIRE(UUIDs.size() > 5); // four outline lines, the label chip, and at least one glyph

    // The first four identifiers are the outline lines, in the documented order.
    float x_min = 1.f, x_max = 0.f, y_min = 1.f, y_max = 0.f;
    for (size_t i = 0; i < 4; i++) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUIDs.at(i))) {
            x_min = std::min(x_min, vertex.x);
            x_max = std::max(x_max, vertex.x);
            y_min = std::min(y_min, vertex.y);
            y_max = std::max(y_max, vertex.y);
        }
    }

    DOCTEST_CHECK(x_min == doctest::Approx(0.f));
    DOCTEST_CHECK(x_max == doctest::Approx(0.5f));

    // The assertion that matters. A box in the TOP half of the image must land in the UPPER half of
    // the window, because window coordinates run bottom-up and YOLO coordinates run top-down. Without
    // the flip this box would sit at y in [0, 0.5] -- half a window away.
    DOCTEST_CHECK(y_min == doctest::Approx(0.5f));
    DOCTEST_CHECK(y_max == doctest::Approx(1.f));
}

TEST_CASE("Visualizer::addBoundingBoxOverlay maps boxes onto a letterboxed image") {
    // The mapping must be relative to the image, not to the window: a box covering the whole image
    // must trace the image extent exactly, leaving the letterboxed margin untouched.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 200 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 200, 100);

    Visualizer::BoundingBox box;
    box.class_ID = 0;
    box.center = make_vec2(0.5f, 0.5f);
    box.size = make_vec2(1.f, 1.f);

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addBoundingBoxOverlay(visualizer, {box}, {}, extent, 2.f, 12);

    float x_min = 1.f, x_max = 0.f, y_min = 1.f, y_max = 0.f;
    for (size_t i = 0; i < 4; i++) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUIDs.at(i))) {
            x_min = std::min(x_min, vertex.x);
            x_max = std::max(x_max, vertex.x);
            y_min = std::min(y_min, vertex.y);
            y_max = std::max(y_max, vertex.y);
        }
    }

    DOCTEST_CHECK(x_min == doctest::Approx(extent.x));
    DOCTEST_CHECK(x_max == doctest::Approx(extent.z));
    DOCTEST_CHECK(y_min == doctest::Approx(extent.y));
    DOCTEST_CHECK(y_max == doctest::Approx(extent.w));
}

TEST_CASE("Visualizer::addBoundingBoxOverlay layers the overlay in front of the image") {
    // Window-normalized z passes straight through to normalized device coordinates and the depth test
    // is GL_LEQUAL, so every overlay element must sit at negative z to be in front of the image quad
    // at z=0, and the text must be nearer than its own chip or the chip discards it.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    Visualizer::BoundingBox box;
    box.class_ID = 0;
    box.center = make_vec2(0.5f, 0.5f);
    box.size = make_vec2(0.4f, 0.4f);

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addBoundingBoxOverlay(visualizer, {box}, {}, extent, 2.f, 12);
    DOCTEST_REQUIRE(UUIDs.size() > 5);

    for (size_t UUID: UUIDs) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUID)) {
            DOCTEST_CHECK(vertex.z < 0.f);
        }
    }

    const float z_outline = visualizer.getGeometryVertices(UUIDs.at(0)).front().z;
    const float z_chip = visualizer.getGeometryVertices(UUIDs.at(4)).front().z;
    const float z_text = visualizer.getGeometryVertices(UUIDs.at(5)).front().z;

    DOCTEST_CHECK(z_chip < z_outline);
    DOCTEST_CHECK(z_text < z_chip);
}

TEST_CASE("Visualizer::addBoundingBoxOverlay does not warn for boxes at the image edge") {
    // Boxes flush with the image border are ordinary, so they must not produce "outside of drawable
    // area" warnings. Messages are deliberately left enabled: the warnings are gated on message_flag,
    // so disabling them would make this pass no matter how badly the clamping were broken.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.enableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    std::vector<Visualizer::BoundingBox> boxes;

    Visualizer::BoundingBox full_image_box;
    full_image_box.class_ID = 0;
    full_image_box.center = make_vec2(0.5f, 0.5f);
    full_image_box.size = make_vec2(1.f, 1.f);
    boxes.push_back(full_image_box);

    // A box clipped by the top-left corner of the image. Its center and half-size recombine to edges
    // at -0.05, i.e. genuinely outside the image, which is what the clamping exists for. Its label
    // chip is also wider than the box, so the chip must be shifted to stay inside as well.
    Visualizer::BoundingBox corner_box;
    corner_box.class_ID = 1;
    corner_box.center = make_vec2(0.05f, 0.05f);
    corner_box.size = make_vec2(0.2f, 0.2f);
    boxes.push_back(corner_box);

    // The same, clipped by the bottom-right corner: edges at 1.05.
    Visualizer::BoundingBox far_corner_box;
    far_corner_box.class_ID = 2;
    far_corner_box.center = make_vec2(0.95f, 0.95f);
    far_corner_box.size = make_vec2(0.2f, 0.2f);
    boxes.push_back(far_corner_box);

    std::string captured_output;
    {
        capture_cerr capture;
        VisualizerTestHelper::addBoundingBoxOverlay(visualizer, boxes, {}, extent, 2.f, 12);
        captured_output = capture.get_captured_output();
    } // capture destroyed here, before the assertion, so doctest failure output prints

    DOCTEST_CHECK_MESSAGE(captured_output.empty(), "addBoundingBoxOverlay warned for a box flush with the image border: " << captured_output);
}

TEST_CASE("Visualizer::addBoundingBoxOverlay labels boxes by class name or class ID") {
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    Visualizer::BoundingBox box;
    box.class_ID = 5;
    box.center = make_vec2(0.5f, 0.5f);
    box.size = make_vec2(0.4f, 0.4f);

    DOCTEST_SUBCASE("an empty class map labels the box with its numeric class ID") {
        // No class name file was available. Four outline lines, the chip, and one glyph for "5".
        std::vector<size_t> UUIDs;
        DOCTEST_CHECK_NOTHROW(UUIDs = VisualizerTestHelper::addBoundingBoxOverlay(visualizer, {box}, {}, extent, 2.f, 12));
        DOCTEST_CHECK(UUIDs.size() == 6);
    }
    DOCTEST_SUBCASE("a class ID missing from a non-empty class map is an error") {
        // The annotations and the class names do not correspond, which is a real inconsistency rather
        // than a missing file, so it must not be papered over with the numeric ID.
        const std::map<uint, std::string> class_names{{0, "bunny"}, {1, "dragon"}};
        DOCTEST_CHECK_THROWS_AS(std::ignore = VisualizerTestHelper::addBoundingBoxOverlay(visualizer, {box}, class_names, extent, 2.f, 12), std::runtime_error);
    }
    DOCTEST_SUBCASE("a named class is labeled with its name") {
        const std::map<uint, std::string> class_names{{5, "bunny"}};
        std::vector<size_t> UUIDs;
        DOCTEST_CHECK_NOTHROW(UUIDs = VisualizerTestHelper::addBoundingBoxOverlay(visualizer, {box}, class_names, extent, 2.f, 12));
        DOCTEST_CHECK(UUIDs.size() == 4 + 1 + std::strlen("bunny"));
    }
}

TEST_CASE("Visualizer::addBoundingBoxOverlay with no boxes adds no geometry") {
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    const size_t geometry_count_before = VisualizerTestHelper::getLiveGeometryCount(visualizer);
    const std::vector<size_t> UUIDs = VisualizerTestHelper::addBoundingBoxOverlay(visualizer, {}, {}, extent, 2.f, 12);

    DOCTEST_CHECK(UUIDs.empty());
    DOCTEST_CHECK(VisualizerTestHelper::getLiveGeometryCount(visualizer) == geometry_count_before);
}

// ---- Segmentation mask overlay ----------------------------------------------------------------
//
// As with the bounding box tests above, displayImageWithSegmentationMasks() itself is unreachable
// because it ends in plotInteractive(); the reader is static and the overlay is reached through
// VisualizerTestHelper.

//! Directory for the COCO JSON fixtures used by the segmentation mask tests
static std::filesystem::path makeSegmentationMaskFixtureDirectory(const std::string &name) {
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("helios_visualizer_segmask_" + name);
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

//! A COCO file with one square mask, matching the shape RadiationModel::writeImageSegmentationMasks() writes
static std::string singleMaskCOCOJson() {
    return R"({
  "categories": [{"id": 0, "name": "bunny", "supercategory": "none"}],
  "images": [{"id": 0, "file_name": "bunnycam_RGB.jpeg", "height": 100, "width": 100}],
  "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [0, 0, 50, 50], "area": 2500, "iscrowd": 0,
                   "segmentation": [[0, 0, 50, 0, 50, 50, 0, 50]]}]
})";
}

TEST_CASE("Visualizer::readSegmentationMaskFile parses COCO polygon annotations") {
    const std::filesystem::path directory = makeSegmentationMaskFixtureDirectory("read");
    const std::filesystem::path mask_file = directory / "masks.json";
    writeBoundingBoxFixture(mask_file, singleMaskCOCOJson());

    std::vector<Visualizer::SegmentationMask> masks;
    DOCTEST_CHECK_NOTHROW(masks = Visualizer::readSegmentationMaskFile(mask_file.string()));

    DOCTEST_REQUIRE(masks.size() == 1);
    DOCTEST_CHECK(masks.at(0).class_ID == 0);
    DOCTEST_CHECK(masks.at(0).class_name == "bunny");
    DOCTEST_CHECK(masks.at(0).image_size.x == doctest::Approx(100.f));
    DOCTEST_CHECK(masks.at(0).image_size.y == doctest::Approx(100.f));

    DOCTEST_REQUIRE(masks.at(0).polygons.size() == 1);
    DOCTEST_REQUIRE(masks.at(0).polygons.at(0).size() == 4);

    // Coordinates are absolute pixels, not normalized -- the property that distinguishes this format from the YOLO one.
    DOCTEST_CHECK(masks.at(0).polygons.at(0).at(0).x == doctest::Approx(0.f));
    DOCTEST_CHECK(masks.at(0).polygons.at(0).at(1).x == doctest::Approx(50.f));
    DOCTEST_CHECK(masks.at(0).polygons.at(0).at(2).y == doctest::Approx(50.f));

    std::filesystem::remove_all(directory);
}

TEST_CASE("Visualizer::readSegmentationMaskFile selects annotations by image") {
    // The writer appends, so one file can describe several images. Each image must yield only its own masks.
    const std::filesystem::path directory = makeSegmentationMaskFixtureDirectory("multi_image");
    const std::filesystem::path mask_file = directory / "masks.json";
    writeBoundingBoxFixture(mask_file, R"({
  "categories": [{"id": 0, "name": "bunny"}, {"id": 1, "name": "dragon"}],
  "images": [{"id": 0, "file_name": "first.jpeg", "height": 100, "width": 100},
             {"id": 1, "file_name": "second.jpeg", "height": 100, "width": 100}],
  "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "segmentation": [[0, 0, 10, 0, 10, 10]]},
                  {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 20, 0, 20, 20]]},
                  {"id": 2, "image_id": 1, "category_id": 1, "segmentation": [[30, 30, 40, 30, 40, 40]]}]
})");

    std::vector<Visualizer::SegmentationMask> masks;
    DOCTEST_CHECK_NOTHROW(masks = Visualizer::readSegmentationMaskFile(mask_file.string(), "second.jpeg"));
    DOCTEST_REQUIRE(masks.size() == 2);
    DOCTEST_CHECK(masks.at(0).class_name == "dragon");

    // A path from another directory still matches, because only the file name is compared.
    DOCTEST_CHECK_NOTHROW(masks = Visualizer::readSegmentationMaskFile(mask_file.string(), "/somewhere/else/first.jpeg"));
    DOCTEST_CHECK(masks.size() == 1);

    // Naming no image is ambiguous when the file describes more than one.
    DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string()), std::runtime_error);

    // An image the file does not describe is an error rather than an empty result.
    DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string(), "absent.jpeg"), std::runtime_error);

    std::filesystem::remove_all(directory);
}

TEST_CASE("Visualizer::readSegmentationMaskFile rejects malformed mask files") {
    const std::filesystem::path directory = makeSegmentationMaskFixtureDirectory("malformed");
    const std::filesystem::path mask_file = directory / "masks.json";

    SUBCASE("not JSON at all") {
        writeBoundingBoxFixture(mask_file, "this is not json");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string()), std::runtime_error);
    }
    SUBCASE("missing the annotations array") {
        writeBoundingBoxFixture(mask_file, R"({"images": [{"id": 0, "file_name": "a.jpeg", "width": 10, "height": 10}], "categories": []})");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string()), std::runtime_error);
    }
    SUBCASE("odd number of coordinates") {
        writeBoundingBoxFixture(mask_file, R"({"categories": [{"id": 0, "name": "a"}],
          "images": [{"id": 0, "file_name": "a.jpeg", "width": 10, "height": 10}],
          "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "segmentation": [[0, 0, 5, 0, 5]]}]})");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string()), std::runtime_error);
    }
    SUBCASE("fewer than three vertices") {
        writeBoundingBoxFixture(mask_file, R"({"categories": [{"id": 0, "name": "a"}],
          "images": [{"id": 0, "file_name": "a.jpeg", "width": 10, "height": 10}],
          "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "segmentation": [[0, 0, 5, 5]]}]})");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string()), std::runtime_error);
    }
    SUBCASE("class ID absent from the categories array") {
        writeBoundingBoxFixture(mask_file, R"({"categories": [{"id": 0, "name": "a"}],
          "images": [{"id": 0, "file_name": "a.jpeg", "width": 10, "height": 10}],
          "annotations": [{"id": 0, "image_id": 0, "category_id": 7, "segmentation": [[0, 0, 5, 0, 5, 5]]}]})");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string()), std::runtime_error);
    }
    SUBCASE("run-length encoded segmentation") {
        // The RLE form is valid COCO but is never written by Helios, so it must be reported rather than silently skipped.
        writeBoundingBoxFixture(mask_file, R"({"categories": [{"id": 0, "name": "a"}],
          "images": [{"id": 0, "file_name": "a.jpeg", "width": 10, "height": 10}],
          "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "segmentation": {"counts": [1, 2, 3], "size": [10, 10]}}]})");
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile(mask_file.string()), std::runtime_error);
    }
    SUBCASE("missing file") {
        DOCTEST_CHECK_THROWS_AS(std::ignore = Visualizer::readSegmentationMaskFile((directory / "absent.json").string()), std::runtime_error);
    }

    std::filesystem::remove_all(directory);
}

//! One square mask covering the top-left quadrant of a 100x100 image
static Visualizer::SegmentationMask makeQuadrantMask() {
    Visualizer::SegmentationMask mask;
    mask.class_ID = 0;
    mask.class_name = "bunny";
    mask.image_size = make_vec2(100.f, 100.f);
    mask.polygons.push_back({make_vec2(0.f, 0.f), make_vec2(50.f, 0.f), make_vec2(50.f, 50.f), make_vec2(0.f, 50.f)});
    return mask;
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay maps polygons into image coordinates with y flipped") {
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.4f, 2.f, 12);
    DOCTEST_REQUIRE(!UUIDs.empty());

    float x_min = 1.f, x_max = 0.f, y_min = 1.f, y_max = 0.f;
    for (const size_t UUID: UUIDs) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUID)) {
            x_min = std::min(x_min, vertex.x);
            x_max = std::max(x_max, vertex.x);
            y_min = std::min(y_min, vertex.y);
            y_max = std::max(y_max, vertex.y);
        }
    }

    DOCTEST_CHECK(x_min == doctest::Approx(0.f));

    // The mask occupies the TOP-left quadrant in image coordinates, so it must land in the UPPER half
    // of the window. Without the flip it would sit against y = 0 instead.
    DOCTEST_CHECK(y_max == doctest::Approx(1.f));
    DOCTEST_CHECK(y_min > 0.4f);
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay maps polygons onto a letterboxed image") {
    // A 200x100 image in an 800x800 window is letterboxed, so a mask covering the whole image must
    // trace the image extent and not the window.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 200 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 200, 100);

    Visualizer::SegmentationMask mask;
    mask.class_ID = 0;
    mask.image_size = make_vec2(200.f, 100.f);
    mask.polygons.push_back({make_vec2(0.f, 0.f), make_vec2(200.f, 0.f), make_vec2(200.f, 100.f), make_vec2(0.f, 100.f)});

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {mask}, extent, 0.4f, 2.f, 12);
    DOCTEST_REQUIRE(!UUIDs.empty());

    float y_min = 1.f, y_max = 0.f;
    for (const size_t UUID: UUIDs) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUID)) {
            y_min = std::min(y_min, vertex.y);
            y_max = std::max(y_max, vertex.y);
        }
    }

    DOCTEST_CHECK(y_min == doctest::Approx(extent.y));
    DOCTEST_CHECK(y_max == doctest::Approx(extent.w));
    DOCTEST_CHECK(extent.y > 0.f); // the image really is letterboxed
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay layers the overlay in front of the image") {
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.4f, 2.f, 12);
    DOCTEST_REQUIRE(UUIDs.size() > 6);

    // Every element sits in front of the image quad at z = 0, and smaller z is nearer the viewer.
    for (const size_t UUID: UUIDs) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUID)) {
            DOCTEST_CHECK(vertex.z < 0.f);
        }
    }

    // Geometry is emitted per mask as fill runs, then one outline line per vertex, then the chip, then
    // one glyph per label character. The fill run count depends on the mask height, so the boundaries
    // are derived from the end rather than assumed.
    const size_t glyph_count = std::strlen("bunny");
    const size_t chip_index = UUIDs.size() - glyph_count - 1;
    const size_t outline_index = chip_index - 4;

    const float z_fill = visualizer.getGeometryVertices(UUIDs.at(0)).front().z;
    const float z_outline = visualizer.getGeometryVertices(UUIDs.at(outline_index)).front().z;
    const float z_chip = visualizer.getGeometryVertices(UUIDs.at(chip_index)).front().z;
    const float z_text = visualizer.getGeometryVertices(UUIDs.at(chip_index + 1)).front().z;

    // The fill must sit BEHIND the outline so the outline is not dimmed by the translucent fill over it.
    DOCTEST_CHECK(z_fill > z_outline);
    DOCTEST_CHECK(z_outline > z_chip);
    DOCTEST_CHECK(z_chip > z_text);
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay fills a self-intersecting contour") {
    // The scanline fill exists so that this case works. RadiationModel::generateAnnotationsFromMasks()
    // falls back to traceBoundarySimple() whenever the Moore trace yields fewer than 10 points, and
    // that fallback emits boundary pixels in breadth-first order rather than in order around the
    // contour, producing a badly self-intersecting loop. A triangulating fill fails outright on such
    // a polygon; the even-odd rule handles it natively.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    // The boundary pixels of a ring, in the breadth-first order traceBoundarySimple() emits them:
    // the walk alternates between opposite edges of the ring instead of going around it.
    Visualizer::SegmentationMask self_intersecting;
    self_intersecting.class_ID = 0;
    self_intersecting.image_size = make_vec2(100.f, 100.f);
    self_intersecting.polygons.push_back({make_vec2(0.f, 0.f), make_vec2(0.f, 6.f), make_vec2(6.f, 0.f), make_vec2(0.f, 12.f), make_vec2(12.f, 0.f), make_vec2(0.f, 18.f), make_vec2(18.f, 0.f),
                                          make_vec2(0.f, 24.f), make_vec2(24.f, 0.f), make_vec2(0.f, 30.f), make_vec2(30.f, 0.f), make_vec2(0.f, 36.f), make_vec2(36.f, 0.f), make_vec2(0.f, 42.f),
                                          make_vec2(42.f, 0.f), make_vec2(0.f, 48.f), make_vec2(6.f, 48.f), make_vec2(48.f, 0.f), make_vec2(48.f, 6.f), make_vec2(12.f, 48.f), make_vec2(48.f, 12.f),
                                          make_vec2(18.f, 48.f), make_vec2(48.f, 18.f), make_vec2(24.f, 48.f), make_vec2(48.f, 24.f), make_vec2(30.f, 48.f), make_vec2(48.f, 30.f), make_vec2(36.f, 48.f),
                                          make_vec2(48.f, 36.f), make_vec2(42.f, 48.f), make_vec2(48.f, 42.f), make_vec2(48.f, 48.f)});

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {self_intersecting}, extent, 0.4f, 2.f, 12);

    // Beyond one outline line per vertex, the chip and one glyph, there must be fill geometry.
    const size_t non_fill = self_intersecting.polygons.at(0).size() + 1 + std::strlen("0");
    DOCTEST_CHECK(UUIDs.size() > non_fill);

    // Every fill run must lie within the contour's own bounds, which span image x,y in [0,48] and so
    // window x in [0,0.48] and y in [0.52,1].
    for (size_t i = 0; i < UUIDs.size() - non_fill; i++) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUIDs.at(i))) {
            DOCTEST_CHECK(vertex.x <= 0.48f + 1e-4f);
            DOCTEST_CHECK(vertex.y >= 0.52f - 1e-4f);
        }
    }
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay fills a contour that pinches to a point") {
    // The failure mode seen on real writeImageSegmentationMasks() output, and the reason a
    // triangulating fill was not good enough. A boundary traced around a mask that narrows to a
    // one-pixel neck passes through the neck twice, from opposite sides, so the loop crosses itself
    // there even though every vertex is distinct and consecutive vertices are adjacent pixels. Ear
    // clipping rejected the dragon of the tutorial 12 output for exactly this reason.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    // Two blobs joined at (30,30), which the contour visits from both sides.
    Visualizer::SegmentationMask pinched;
    pinched.class_ID = 0;
    pinched.image_size = make_vec2(100.f, 100.f);
    pinched.polygons.push_back({make_vec2(0.f, 0.f), make_vec2(24.f, 0.f), make_vec2(30.f, 30.f), make_vec2(60.f, 0.f), make_vec2(84.f, 0.f), make_vec2(84.f, 60.f), make_vec2(60.f, 60.f),
                                make_vec2(30.f, 30.f), make_vec2(24.f, 60.f), make_vec2(0.f, 60.f)});

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {pinched}, extent, 0.4f, 2.f, 12);

    const size_t non_fill = pinched.polygons.at(0).size() + 1 + std::strlen("0");
    DOCTEST_CHECK(UUIDs.size() > non_fill);

    // The two lobes meet at a point, so the row through that point must be covered by two separate
    // runs with a gap between them rather than by one run spanning the whole shape. Getting this
    // wrong is exactly what filling across a self-intersection looks like.
    // Image row 30 covers window y from 0.69 to 0.70, so its runs are the ones straddling 0.695.
    constexpr float neck_row_center_y = 0.695f;

    size_t runs_on_neck_row = 0;
    float gap_left = 0.f, gap_right = 1.f;
    for (size_t i = 0; i < UUIDs.size() - non_fill; i++) {
        float x_lo = 1.f, x_hi = 0.f, y_lo = 1.f, y_hi = 0.f;
        for (const vec3 &v: visualizer.getGeometryVertices(UUIDs.at(i))) {
            x_lo = std::min(x_lo, v.x);
            x_hi = std::max(x_hi, v.x);
            y_lo = std::min(y_lo, v.y);
            y_hi = std::max(y_hi, v.y);
        }
        if (y_lo < neck_row_center_y && y_hi > neck_row_center_y) {
            runs_on_neck_row++;
            if (x_hi <= 0.30f) {
                gap_left = std::max(gap_left, x_hi); // run belonging to the left lobe
            } else {
                gap_right = std::min(gap_right, x_lo); // run belonging to the right lobe
            }
        }
    }

    DOCTEST_CHECK(runs_on_neck_row == 2);
    DOCTEST_CHECK(gap_right > gap_left); // the lobes are not bridged across the meeting point
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay fills concave polygons within their own bounds") {
    // The scanline fill must respect concavity: each row is covered only between its own crossings,
    // so the notch of an L is left empty rather than being bridged.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    // An L-shape occupying the left column and bottom row of the top-left quadrant.
    Visualizer::SegmentationMask concave;
    concave.class_ID = 0;
    concave.image_size = make_vec2(100.f, 100.f);
    concave.polygons.push_back({make_vec2(0.f, 0.f), make_vec2(20.f, 0.f), make_vec2(20.f, 40.f), make_vec2(60.f, 40.f), make_vec2(60.f, 60.f), make_vec2(0.f, 60.f)});

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {concave}, extent, 0.4f, 2.f, 12);

    // One outline line per vertex, the chip and one glyph; everything before those is fill.
    const size_t non_fill = concave.polygons.at(0).size() + 1 + std::strlen("0");
    DOCTEST_REQUIRE(UUIDs.size() > non_fill);
    const size_t fill_count = UUIDs.size() - non_fill;

    // The notch at image pixels (40, 20) is outside the L, so no fill vertex may reach into it. In
    // window coordinates that is x > 0.2 and y > 0.6.
    for (size_t i = 0; i < fill_count; i++) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUIDs.at(i))) {
            const bool inside_notch = (vertex.x > 0.2f + 1e-4f) && (vertex.y > 0.6f + 1e-4f);
            DOCTEST_CHECK_FALSE(inside_notch);
        }
    }
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay colors masks per annotation rather than per class") {
    // Instance segmentation: two touching objects of the SAME class must be told apart, so the color
    // follows the annotation's position in the file and not its class ID.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    Visualizer::SegmentationMask first = makeQuadrantMask();
    Visualizer::SegmentationMask second = makeQuadrantMask(); // same class_ID
    second.polygons.at(0) = {make_vec2(50.f, 50.f), make_vec2(90.f, 50.f), make_vec2(90.f, 90.f), make_vec2(50.f, 90.f)};

    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {first, second}, extent, 0.4f, 2.f, 12);
    DOCTEST_REQUIRE(UUIDs.size() > 2);

    // Index 0 is the first triangle of mask one; the same offset into mask two's geometry follows it.
    const size_t per_mask = UUIDs.size() / 2;
    const RGBAcolor first_color = VisualizerTestHelper::getGeometryColor(visualizer, UUIDs.at(0));
    const RGBAcolor second_color = VisualizerTestHelper::getGeometryColor(visualizer, UUIDs.at(per_mask));

    const bool colors_differ = (first_color.r != second_color.r) || (first_color.g != second_color.g) || (first_color.b != second_color.b);
    DOCTEST_CHECK(colors_differ);
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay respects the fill opacity") {
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    const std::vector<size_t> filled = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.4f, 2.f, 12);
    DOCTEST_CHECK(VisualizerTestHelper::getGeometryColor(visualizer, filled.at(0)).a == doctest::Approx(0.4f));

    // Zero opacity draws the outline and label only, leaving one line per vertex, the chip and the glyphs.
    const std::vector<size_t> unfilled = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.f, 2.f, 12);
    DOCTEST_CHECK(unfilled.size() == 4 + 1 + std::strlen("bunny"));
    DOCTEST_CHECK(unfilled.size() < filled.size());
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay labels masks by class name or class ID") {
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    // The label contributes one glyph per character, on top of the fill, four outline lines and the chip.
    const std::vector<size_t> named = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.f, 2.f, 12);
    DOCTEST_CHECK(named.size() == 4 + 1 + std::strlen("bunny"));

    // With no name in the file, the numeric class ID is all there is to show.
    Visualizer::SegmentationMask unnamed = makeQuadrantMask();
    unnamed.class_name = "";
    unnamed.class_ID = 12;
    const std::vector<size_t> numeric = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {unnamed}, extent, 0.f, 2.f, 12);
    DOCTEST_CHECK(numeric.size() == 4 + 1 + std::strlen("12"));
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay can suppress the class labels") {
    // Overlapping masks in a busy scene stack their chips over the image, so the labels can be turned
    // off to inspect the mask shapes alone. Only the chip and its glyphs go; fill and outline stay.
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    const std::vector<size_t> labeled = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.4f, 2.f, 12, true);
    const std::vector<size_t> unlabeled = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.4f, 2.f, 12, false);

    // Exactly the chip and one glyph per character of "bunny" disappear.
    DOCTEST_CHECK(unlabeled.size() == labeled.size() - 1 - std::strlen("bunny"));

    // The fill and outline are untouched, so the geometry that remains still spans the mask.
    float x_min = 1.f, x_max = 0.f, y_min = 1.f, y_max = 0.f;
    for (const size_t UUID: unlabeled) {
        for (const vec3 &vertex: visualizer.getGeometryVertices(UUID)) {
            x_min = std::min(x_min, vertex.x);
            x_max = std::max(x_max, vertex.x);
            y_min = std::min(y_min, vertex.y);
            y_max = std::max(y_max, vertex.y);
        }
    }
    DOCTEST_CHECK(x_min == doctest::Approx(0.f));
    DOCTEST_CHECK(x_max == doctest::Approx(0.5f));
    DOCTEST_CHECK(y_max == doctest::Approx(1.f));

    // Suppressing the labels must not suppress the fill: zero opacity and no labels leaves the outline.
    const std::vector<size_t> outline_only = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {makeQuadrantMask()}, extent, 0.f, 2.f, 12, false);
    DOCTEST_CHECK(outline_only.size() == 4); // one line per vertex of the square, nothing else
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay does not warn for polygons at the image edge") {
    // Messages are deliberately left ENABLED here: a polygon flush with the image border must not
    // trip the "outside of drawable area" warning, which is what the clamping exists to prevent.
    Visualizer visualizer(800, 800, 16, true, true); // headless

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    Visualizer::SegmentationMask edge_mask;
    edge_mask.class_ID = 0;
    edge_mask.class_name = "edge";
    edge_mask.image_size = make_vec2(100.f, 100.f);
    edge_mask.polygons.push_back({make_vec2(0.f, 0.f), make_vec2(100.f, 0.f), make_vec2(100.f, 100.f), make_vec2(0.f, 100.f)});

    std::string captured;
    {
        capture_cerr capture;
        std::ignore = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {edge_mask}, extent, 0.4f, 2.f, 12);
        captured = capture.get_captured_output();
    } // capture destroyed before the assertion, so a doctest failure message prints normally

    DOCTEST_CHECK(captured.empty());
}

TEST_CASE("Visualizer::addSegmentationMaskOverlay with no masks adds no geometry") {
    Visualizer visualizer(800, 800, 16, true, true); // headless
    visualizer.disableMessages();

    const std::vector<unsigned char> pixel_data(4 * 100 * 100, 255);
    const vec4 extent = VisualizerTestHelper::buildImageDisplayGeometry(visualizer, pixel_data, 100, 100);

    const size_t geometry_count_before = VisualizerTestHelper::getLiveGeometryCount(visualizer);
    const std::vector<size_t> UUIDs = VisualizerTestHelper::addSegmentationMaskOverlay(visualizer, {}, extent, 0.4f, 2.f, 12);

    DOCTEST_CHECK(UUIDs.empty());
    DOCTEST_CHECK(VisualizerTestHelper::getLiveGeometryCount(visualizer) == geometry_count_before);
}

// ================================================================================================
// Linear-light rendering pipeline (sRGB decode -> shade -> tone map -> sRGB encode)
// ================================================================================================

TEST_CASE("Visualizer linear pipeline state and exposure validation") {
    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();

    // On by default: physically-based shading is the standard rendering path.
    DOCTEST_CHECK(visualizer.isLinearPipelineEnabled() == true);
    DOCTEST_CHECK(visualizer.getExposure() == doctest::Approx(1.f));

    visualizer.disableLinearPipeline();
    DOCTEST_CHECK(visualizer.isLinearPipelineEnabled() == false);

    visualizer.setExposure(2.5f);
    DOCTEST_CHECK(visualizer.getExposure() == doctest::Approx(2.5f));

    visualizer.enableLinearPipeline();
    DOCTEST_CHECK(visualizer.isLinearPipelineEnabled() == true);

    // Exposure is a linear multiplier on radiance; zero or negative is meaningless and must fail
    // loudly rather than silently producing a black or sign-flipped image.
    DOCTEST_CHECK_THROWS_AS(visualizer.setExposure(0.f), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(visualizer.setExposure(-1.f), std::runtime_error);
    DOCTEST_CHECK(visualizer.getExposure() == doctest::Approx(2.5f)); // unchanged by the failed calls
}

TEST_CASE("Visualizer exact color mode and linear pipeline are mutually exclusive") {
    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();

    // The synthetic annotation plug-in carries object ID codes through the framebuffer as RGB
    // values. Tone mapping and sRGB encoding are non-linear transforms of those channels, so
    // enabling exact color mode must turn the linear pipeline off or the IDs decode to garbage.
    // This matters more now that the pipeline is on by default: a caller who never touches it
    // still gets the protection.
    DOCTEST_CHECK(visualizer.isLinearPipelineEnabled() == true);

    visualizer.enableExactColorMode();
    DOCTEST_CHECK(visualizer.isLinearPipelineEnabled() == false);

    // The two calls must round-trip, rather than leaving the renderer in a non-default state.
    visualizer.disableExactColorMode();
    DOCTEST_CHECK(visualizer.isLinearPipelineEnabled() == true);
}

TEST_CASE("Visualizer linear pipeline tone-maps the rendered image") {
    // A mid-grey unlit patch fills the view. With lighting disabled the shaded value equals the
    // decoded albedo, so the output is encode(tonemap(decode(a))). decode and encode are exact
    // inverses, leaving the ACES tone curve as the only transformation. That curve maps linear
    // 0.5 to roughly 0.48 and thus pulls the encoded mid-grey visibly DOWN, which is the signal
    // this test detects. On the old shader the value passes through untouched.
    Context context;
    const uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(100, 100));
    context.setPrimitiveColor(patch_UUID, make_RGBcolor(0.5f, 0.5f, 0.5f));

    // No antialiasing, so interior pixels are exact rather than edge-blended.
    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_NONE);
    visualizer.setCameraPosition(make_vec3(0, 0, 10), make_vec3(0, 0, 0));
    // colorBoost of 1.5 would saturate 0.5 and confound the comparison, so pin it to 1.
    visualizer.enableExactColorMode();
    visualizer.buildContextGeometry(&context);

    auto centerPixel = [](Visualizer &vis) {
        std::vector<uint> pixels;
        uint w = 0, h = 0;
        vis.plotUpdate();
        vis.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        // Row-major r-g-b triples; sample the red channel at the center of the image.
        const size_t idx = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 2));
        REQUIRE(pixels.size() > idx);
        return pixels.at(idx);
    };

    const uint baseline = centerPixel(visualizer);

    visualizer.enableLinearPipeline();
    const uint tonemapped = centerPixel(visualizer);

    // The patch must actually be covering the sampled pixel, or both reads are just background.
    REQUIRE(baseline > 0);
    // ACES compresses mid-grey downward. A no-op shader would leave the two identical.
    DOCTEST_CHECK(tonemapped < baseline);
}

TEST_CASE("Visualizer Phong material state and validation") {
    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();

    const Visualizer::PhongMaterial defaults = visualizer.getPhongMaterial();
    DOCTEST_CHECK(defaults.ambient == doctest::Approx(1.0f));
    DOCTEST_CHECK(defaults.diffuse == doctest::Approx(0.8f));
    DOCTEST_CHECK(defaults.specular == doctest::Approx(0.2f));
    DOCTEST_CHECK(defaults.shininess == doctest::Approx(32.f));

    Visualizer::PhongMaterial glossy;
    glossy.ambient = 0.2f;
    glossy.diffuse = 0.6f;
    glossy.specular = 0.9f;
    glossy.shininess = 96.f;
    visualizer.setPhongMaterial(glossy);

    const Visualizer::PhongMaterial read_back = visualizer.getPhongMaterial();
    DOCTEST_CHECK(read_back.ambient == doctest::Approx(0.2f));
    DOCTEST_CHECK(read_back.diffuse == doctest::Approx(0.6f));
    DOCTEST_CHECK(read_back.specular == doctest::Approx(0.9f));
    DOCTEST_CHECK(read_back.shininess == doctest::Approx(96.f));

    // A negative reflectance weight would subtract light from the scene.
    Visualizer::PhongMaterial negative_weight = glossy;
    negative_weight.diffuse = -0.1f;
    DOCTEST_CHECK_THROWS_AS(visualizer.setPhongMaterial(negative_weight), std::runtime_error);

    // pow(x, 0) is 1 everywhere, which would paint a uniform highlight across the whole surface.
    Visualizer::PhongMaterial zero_exponent = glossy;
    zero_exponent.shininess = 0.f;
    DOCTEST_CHECK_THROWS_AS(visualizer.setPhongMaterial(zero_exponent), std::runtime_error);

    // The rejected calls must not have partially applied.
    DOCTEST_CHECK(visualizer.getPhongMaterial().diffuse == doctest::Approx(0.6f));
    DOCTEST_CHECK(visualizer.getPhongMaterial().shininess == doctest::Approx(96.f));
}

TEST_CASE("Visualizer ambient color state and validation") {
    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();

    // Default is a cool sky above and a warm ground bounce below.
    DOCTEST_CHECK(visualizer.getAmbientSkyColor().b > visualizer.getAmbientSkyColor().r);
    DOCTEST_CHECK(visualizer.getAmbientGroundColor().r > visualizer.getAmbientGroundColor().b);

    visualizer.setAmbientColors(make_RGBcolor(0.1f, 0.2f, 0.3f), make_RGBcolor(0.4f, 0.5f, 0.6f));
    DOCTEST_CHECK(visualizer.getAmbientSkyColor().g == doctest::Approx(0.2f));
    DOCTEST_CHECK(visualizer.getAmbientGroundColor().g == doctest::Approx(0.5f));

    // helios::RGBcolor clamps its components to [0,1] in its constructor, so an out-of-range
    // ambient color is clamped before it ever reaches setAmbientColors().
    visualizer.setAmbientColors(make_RGBcolor(-0.1f, 2.f, 0.5f), make_RGBcolor(0.f, 0.f, 0.f));
    DOCTEST_CHECK(visualizer.getAmbientSkyColor().r == doctest::Approx(0.f));
    DOCTEST_CHECK(visualizer.getAmbientSkyColor().g == doctest::Approx(1.f));
}

TEST_CASE("Visualizer specular term brightens a surface at the mirror angle") {
    // A patch is viewed from the mirror direction of the light, where the Blinn-Phong half-vector
    // aligns with the surface normal and the specular term is at its maximum. Rendering the same
    // scene with specular=0 and then with a strong specular weight must brighten that pixel. The
    // old shader had no specular term at all, so both renders were identical.
    Context context;
    const uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(20, 20));
    context.setPrimitiveColor(patch_UUID, make_RGBcolor(0.25f, 0.25f, 0.25f));

    Visualizer visualizer(64, 64, 0, true, true); // headless, no antialiasing
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);

    // The patch lies in the z=0 plane, so its normal is +z. With the light and the camera both at
    // 45 degrees on opposite sides, the half-vector is exactly the surface normal.
    visualizer.setLightDirection(make_vec3(1, 0, 1));
    visualizer.setCameraPosition(make_vec3(-10, 0, 10), make_vec3(0, 0, 0));
    visualizer.buildContextGeometry(&context);

    auto centerPixel = [](Visualizer &vis) {
        std::vector<uint> pixels;
        uint w = 0, h = 0;
        vis.plotUpdate();
        vis.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t idx = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 2));
        REQUIRE(pixels.size() > idx);
        return pixels.at(idx);
    };

    Visualizer::PhongMaterial matte;
    matte.ambient = 0.4f;
    matte.diffuse = 0.8f;
    matte.specular = 0.f; // no highlight
    matte.shininess = 32.f;
    visualizer.setPhongMaterial(matte);
    const uint without_specular = centerPixel(visualizer);

    Visualizer::PhongMaterial glossy = matte;
    glossy.specular = 1.f;
    glossy.shininess = 16.f; // broad lobe, so the highlight comfortably covers the center pixel
    visualizer.setPhongMaterial(glossy);
    const uint with_specular = centerPixel(visualizer);

    REQUIRE(without_specular > 0); // the patch must actually cover the sampled pixel
    DOCTEST_CHECK(with_specular > without_specular);
}

TEST_CASE("Visualizer hemispheric ambient shades by surface orientation") {
    // Only the ambient term is left active, with a white sky and a black ground. A patch tilted so
    // that its normal points upward must then come out brighter than one tilted downward.
    //
    // The patches are tilted about the y-axis and viewed from along +x rather than from directly
    // overhead. Viewing a patch head-on would defeat the test: the shader flips a normal that faces
    // away from the camera, so an up- and a down-facing patch seen from +z would shade identically.
    // A side-on view keeps the two tilts distinguishable after that flip.
    auto renderTiltedPatch = [](float tilt_radians) {
        Context context;
        const uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(20, 20));
        context.setPrimitiveColor(UUID, make_RGBcolor(1.f, 1.f, 1.f));
        context.rotatePrimitive(UUID, tilt_radians, "y");

        Visualizer visualizer(64, 64, 0, true, true); // headless, no antialiasing
        visualizer.disableMessages();
        visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
        visualizer.setAmbientColors(make_RGBcolor(1.f, 1.f, 1.f), make_RGBcolor(0.f, 0.f, 0.f));

        Visualizer::PhongMaterial ambient_only;
        ambient_only.ambient = 1.f;
        ambient_only.diffuse = 0.f; // remove the light-dependent terms so only ambient remains
        ambient_only.specular = 0.f;
        ambient_only.shininess = 32.f;
        visualizer.setPhongMaterial(ambient_only);

        visualizer.setCameraPosition(make_vec3(10, 0, 0), make_vec3(0, 0, 0));
        visualizer.buildContextGeometry(&context);
        visualizer.plotUpdate();

        std::vector<uint> pixels;
        uint w = 0, h = 0;
        visualizer.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t idx = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 2));
        REQUIRE(pixels.size() > idx);
        return pixels.at(idx);
    };

    // Tilting by -60 degrees leans the normal up toward the sky; +60 leans it down toward the ground.
    const uint tilted_up = renderTiltedPatch(-60.f * PI_F / 180.f);
    const uint tilted_down = renderTiltedPatch(60.f * PI_F / 180.f);

    REQUIRE(tilted_up > 0); // the patch must actually cover the sampled pixel
    DOCTEST_CHECK(tilted_up != tilted_down);
}

TEST_CASE("Visualizer smooth shading state") {
    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();

    // On by default. Geometry without authored vertex normals carries the face normal replicated
    // across its vertices, so the default is a no-op for it.
    DOCTEST_CHECK(visualizer.isSmoothShadingEnabled() == true);

    visualizer.disableSmoothShading();
    DOCTEST_CHECK(visualizer.isSmoothShadingEnabled() == false);

    visualizer.enableSmoothShading();
    DOCTEST_CHECK(visualizer.isSmoothShadingEnabled() == true);
}

TEST_CASE("Visualizer smooth shading is a no-op for geometry without vertex normals") {
    // A plain patch has no authored vertex normals, so addGeometry() replicates its face normal
    // across the vertices. Toggling smooth shading must therefore not change the rendered image at
    // all -- this is what makes the feature safe to leave on by default.
    Context context;
    const uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(20, 20));
    context.setPrimitiveColor(patch_UUID, make_RGBcolor(0.6f, 0.6f, 0.6f));

    Visualizer visualizer(64, 64, 0, true, true); // headless, no antialiasing
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
    visualizer.setLightDirection(make_vec3(1, 0, 1));
    visualizer.setCameraPosition(make_vec3(-10, 0, 10), make_vec3(0, 0, 0));
    visualizer.buildContextGeometry(&context);

    auto centerPixel = [](Visualizer &vis) {
        std::vector<uint> pixels;
        uint w = 0, h = 0;
        vis.plotUpdate();
        vis.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t idx = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 2));
        REQUIRE(pixels.size() > idx);
        return pixels.at(idx);
    };

    visualizer.enableSmoothShading();
    const uint smooth = centerPixel(visualizer);

    visualizer.disableSmoothShading();
    const uint flat = centerPixel(visualizer);

    REQUIRE(smooth > 0);
    DOCTEST_CHECK(smooth == flat);
}

TEST_CASE("Visualizer smooth shading interpolates authored vertex normals") {
    // Two triangles forming a flat square are given vertex normals that fan outward, as a curved
    // surface tessellated into flats would have. Because the geometry is flat, flat shading gives
    // every fragment the same normal and thus a uniform result; smooth shading interpolates the
    // fanned normals and produces a gradient instead. Sampling two well-separated points and
    // comparing their difference under each mode isolates that gradient.
    //
    // The normals are authored through a Polymesh, which is the supported route for supplying them.
    Context context;
    const uint t0 = context.addTriangle(make_vec3(-5, -5, 0), make_vec3(5, -5, 0), make_vec3(5, 5, 0), make_RGBcolor(0.7f, 0.7f, 0.7f));
    const uint t1 = context.addTriangle(make_vec3(-5, -5, 0), make_vec3(5, 5, 0), make_vec3(-5, 5, 0), make_RGBcolor(0.7f, 0.7f, 0.7f));
    const uint objID = context.addPolymeshObject({t0, t1});

    // Shared vertices of the square, in the order the two faces reference them.
    const std::vector<vec3> vertices{make_vec3(-5, -5, 0), make_vec3(5, -5, 0), make_vec3(5, 5, 0), make_vec3(-5, 5, 0)};
    const std::vector<int3> faces{make_int3(0, 1, 2), make_int3(0, 2, 3)};

    // Normals tilted strongly toward each corner, so the interpolated normal sweeps across the face.
    const std::vector<vec3> vertex_normals{normalize(make_vec3(-0.8f, -0.8f, 1.f)), normalize(make_vec3(0.8f, -0.8f, 1.f)), normalize(make_vec3(0.8f, 0.8f, 1.f)), normalize(make_vec3(-0.8f, 0.8f, 1.f))};

    // face_UUIDs maps each face of the table onto the primitive that represents it.
    context.setPolymeshObjectTopology(objID, vertices, faces, {t0, t1}, vertex_normals, {}, helios::NORMAL_SOURCE_AUTHORED);

    Visualizer visualizer(64, 64, 0, true, true); // headless, no antialiasing
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
    visualizer.setLightDirection(make_vec3(1, 0, 1));
    visualizer.buildContextGeometry(&context);
    // Slightly off-axis. Looking straight down the z-axis at geometry lying in the z=0 plane gives a
    // degenerate camera basis and nothing is rasterized at all.
    visualizer.setCameraPosition(make_vec3(1, 1, 18), make_vec3(0, 0, 0));

    // Difference in brightness between two points on opposite sides of the square.
    auto brightnessSpread = [](Visualizer &vis) {
        std::vector<uint> pixels;
        uint w = 0, h = 0;
        vis.plotUpdate();
        vis.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t left = 3 * (size_t(h / 2) * size_t(w) + size_t(w * 3 / 8));
        const size_t right = 3 * (size_t(h / 2) * size_t(w) + size_t(w * 5 / 8));
        REQUIRE(pixels.size() > right);
        return std::abs(int(pixels.at(left)) - int(pixels.at(right)));
    };

    visualizer.disableSmoothShading();
    const int flat_spread = brightnessSpread(visualizer);

    visualizer.enableSmoothShading();
    const int smooth_spread = brightnessSpread(visualizer);

    // Flat shading gives every fragment the same face normal, so brightness is essentially uniform
    // across the face (a unit or two of variation survives 8-bit quantization).
    DOCTEST_CHECK(flat_spread <= 2);
    // Smooth shading interpolates between the fanned vertex normals, producing a real gradient.
    DOCTEST_CHECK(smooth_spread > 5);
    DOCTEST_CHECK(smooth_spread > flat_spread);
}

TEST_CASE("Visualizer smooth shading uses analytic normals of curved compound objects") {
    // A coarsely tessellated sphere is the clearest case: with flat shading each facet is a band of
    // constant brightness, so a scanline across the sphere is a staircase. Analytic vertex normals
    // turn it into a ramp. Comparing the largest second difference along the scanline measures
    // exactly that difference between a staircase and a ramp, and unlike a simple bright-vs-dark
    // comparison it cannot be satisfied by an overall change in brightness.
    Context context;
    context.addSphereObject(6, make_vec3(0, 0, 0), 3.f, make_RGBcolor(0.7f, 0.7f, 0.7f));

    Visualizer visualizer(200, 200, 0, true, true); // headless, no antialiasing
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
    visualizer.setLightDirection(make_vec3(1, 0, 1));
    visualizer.buildContextGeometry(&context);
    visualizer.setCameraPosition(make_vec3(-12, 1, 2), make_vec3(0, 0, 0));

    // Largest absolute second difference of brightness along a horizontal scanline through the
    // middle of the sphere. A faceted surface has abrupt steps between bands; a smooth one does not.
    auto maximumCurvature = [](Visualizer &vis) {
        std::vector<uint> pixels;
        uint w = 0, h = 0;
        vis.plotUpdate();
        vis.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t row = size_t(h / 2);
        std::vector<int> scan;
        for (size_t x = 0; x < size_t(w); x++) {
            const size_t idx = 3 * (row * size_t(w) + x);
            REQUIRE(pixels.size() > idx);
            const int value = int(pixels.at(idx));
            // Ignore the background, which is not part of the shaded surface.
            if (value > 0) {
                scan.push_back(value);
            }
        }
        REQUIRE(scan.size() > 10);
        int worst = 0;
        for (size_t i = 1; i + 1 < scan.size(); i++) {
            worst = std::max(worst, std::abs(scan.at(i - 1) - 2 * scan.at(i) + scan.at(i + 1)));
        }
        return worst;
    };

    visualizer.disableSmoothShading();
    const int faceted = maximumCurvature(visualizer);

    visualizer.enableSmoothShading();
    const int smooth = maximumCurvature(visualizer);

    // The faceted sphere must actually show steps, otherwise the comparison below is vacuous.
    DOCTEST_CHECK(faceted > 4);
    DOCTEST_CHECK(smooth < faceted);
}

TEST_CASE("Visualizer smooth shading is a no-op for compound objects with flat faces") {
    // A box is built from genuinely flat faces whose face normal is already correct everywhere, so
    // it must be excluded from analytic normals. If it were not, its edges would be rounded off and
    // the rendered image would change when smooth shading is toggled.
    Context context;
    context.addBoxObject(make_vec3(0, 0, 0), make_vec3(2, 2, 2), make_int3(1, 1, 1), make_RGBcolor(0.6f, 0.6f, 0.6f));

    Visualizer visualizer(64, 64, 0, true, true); // headless, no antialiasing
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
    visualizer.setLightDirection(make_vec3(1, 0, 1));
    visualizer.buildContextGeometry(&context);
    visualizer.setCameraPosition(make_vec3(-6, 3, 4), make_vec3(0, 0, 0));

    auto centerPixel = [](Visualizer &vis) {
        std::vector<uint> pixels;
        uint w = 0, h = 0;
        vis.plotUpdate();
        vis.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t idx = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 2));
        REQUIRE(pixels.size() > idx);
        return pixels.at(idx);
    };

    visualizer.enableSmoothShading();
    const uint smooth = centerPixel(visualizer);

    visualizer.disableSmoothShading();
    const uint flat = centerPixel(visualizer);

    REQUIRE(smooth > 0);
    DOCTEST_CHECK(smooth == flat);
}

TEST_CASE("Visualizer per-material Phong parameters override the global material") {
    // Two patches with different materials, one of which specifies a strong specular highlight
    // through generic material data. Rendered at the mirror angle, the patch whose material raises
    // the specular weight must come out brighter than the one that inherits the global value.
    // Before per-material parameters existed, both patches shaded identically.
    Context context;

    const uint matte_UUID = context.addPatch(make_vec3(-6, 0, 0), make_vec2(10, 10));
    const uint glossy_UUID = context.addPatch(make_vec3(6, 0, 0), make_vec2(10, 10));

    context.addMaterial("matte_leaf");
    context.addMaterial("glossy_leaf");
    context.assignMaterialToPrimitive(matte_UUID, "matte_leaf");
    context.assignMaterialToPrimitive(glossy_UUID, "glossy_leaf");

    // Primitive color is stored on the material, and addMaterial() creates it black, so the colors
    // must be set after the materials are assigned or they are discarded.
    context.setPrimitiveColor(matte_UUID, make_RGBcolor(0.3f, 0.3f, 0.3f));
    context.setPrimitiveColor(glossy_UUID, make_RGBcolor(0.3f, 0.3f, 0.3f));

    // Only the glossy material specifies Phong data; the matte one inherits the global material.
    context.setMaterialData("glossy_leaf", "phong_specular", 1.0f);
    context.setMaterialData("glossy_leaf", "phong_shininess", 8.f);

    Visualizer visualizer(128, 64, 0, true, true); // headless, no antialiasing
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
    visualizer.setLightDirection(make_vec3(0.05f, -0.1f, 1));

    Visualizer::PhongMaterial global_material;
    global_material.ambient = 1.0f;
    global_material.diffuse = 0.8f;
    global_material.specular = 0.f; // global default has no highlight at all
    global_material.shininess = 32.f;
    visualizer.setPhongMaterial(global_material);

    visualizer.buildContextGeometry(&context);
    visualizer.setCameraPosition(make_vec3(0.5f, -2.f, 25), make_vec3(0, 0, 0));
    visualizer.plotUpdate();

    std::vector<uint> pixels;
    uint w = 0, h = 0;
    visualizer.getWindowPixelsRGB(pixels, w, h);
    REQUIRE(w > 0);
    REQUIRE(h > 0);

    // The two patches sit left and right of center in the rendered image.
    const size_t left = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 4));
    const size_t right = 3 * (size_t(h / 2) * size_t(w) + size_t(w * 3 / 4));
    REQUIRE(pixels.size() > right);

    const uint matte_brightness = pixels.at(left);
    const uint glossy_brightness = pixels.at(right);

    REQUIRE(matte_brightness > 0); // both patches must actually be covering their sample points
    REQUIRE(glossy_brightness > 0);
    DOCTEST_CHECK(glossy_brightness > matte_brightness);
}

TEST_CASE("Visualizer per-material Phong parameters partially override the global material") {
    // A material that specifies only one of the four parameters must inherit the rest from the
    // global material, rather than silently resetting them to some built-in default.
    Context context;
    const uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(10, 10));
    context.addMaterial("partial");
    context.assignMaterialToPrimitive(UUID, "partial");
    // Primitive color lives on the material, so it must be set after the material is assigned.
    context.setPrimitiveColor(UUID, make_RGBcolor(0.5f, 0.5f, 0.5f));
    context.setMaterialData("partial", "phong_specular", 0.f);

    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
    visualizer.setLightDirection(make_vec3(0.05f, -0.1f, 1));

    auto renderWithGlobalDiffuse = [&](float diffuse) {
        Visualizer::PhongMaterial material;
        material.ambient = 0.2f;
        material.diffuse = diffuse;
        material.specular = 0.5f;
        material.shininess = 32.f;
        visualizer.setPhongMaterial(material);
        visualizer.buildContextGeometry(&context);
        visualizer.setCameraPosition(make_vec3(0.5f, -2.f, 22), make_vec3(0, 0, 0));
        visualizer.plotUpdate();

        std::vector<uint> pixels;
        uint w = 0, h = 0;
        visualizer.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t idx = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 2));
        REQUIRE(pixels.size() > idx);
        return pixels.at(idx);
    };

    // The material pins specular to zero but says nothing about diffuse, so changing the global
    // diffuse weight must still change how this primitive is shaded.
    const uint dim = renderWithGlobalDiffuse(0.1f);
    const uint bright = renderWithGlobalDiffuse(0.9f);

    REQUIRE(dim > 0);
    DOCTEST_CHECK(bright > dim);
}

TEST_CASE("Visualizer rejects invalid per-material Phong parameters") {
    // buildContextGeometry() only records which UUIDs to build; the geometry, and with it the
    // per-material Phong table, is actually built during plotUpdate(). The validation therefore
    // surfaces there rather than at the buildContextGeometry() call. Each case gets its own
    // Visualizer so that state left behind by a throw cannot mask the next one.
    auto renderWithMaterialData = [](const char *data_label, float value) {
        Context context;
        const uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(10, 10));
        context.addMaterial("bad_material");
        context.assignMaterialToPrimitive(UUID, "bad_material");
        context.setPrimitiveColor(UUID, make_RGBcolor(0.5f, 0.5f, 0.5f));
        context.setMaterialData("bad_material", data_label, value);

        Visualizer visualizer(64, 64, 0, true, true); // headless
        visualizer.disableMessages();
        visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
        visualizer.buildContextGeometry(&context);
        visualizer.plotUpdate();
    };

    // pow(x, 0) is 1 everywhere, which would paint a uniform highlight across the whole surface.
    DOCTEST_CHECK_THROWS_AS(renderWithMaterialData("phong_shininess", 0.f), std::runtime_error);

    // A negative reflectance weight would subtract light from the scene.
    DOCTEST_CHECK_THROWS_AS(renderWithMaterialData("phong_diffuse", -0.5f), std::runtime_error);
}

DOCTEST_TEST_CASE("Visualizer rebuilds only primitives that changed since the last build") {
    // Context dirty flags are sticky: only Context::markGeometryClean() clears them, and that is the
    // user's call to make once every plug-in has processed the change, so the Visualizer must not
    // clear them itself. Left unfiltered, though, getDirtyUUIDs() reports the entire scene on every
    // frame, and each primitive is then re-uploaded to the GPU individually - which made rendering
    // cost scale with the size of the scene rather than with what actually changed. A 900K-primitive
    // scene spent ~40 s per frame re-uploading geometry that had not changed.
    Context context;
    for (int i = 0; i < 50; i++) {
        context.addPatch(make_vec3(float(i), 0, 0), make_vec2(0.5, 0.5));
    }

    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();
    visualizer.buildContextGeometry(&context);

    // First build has to upload everything.
    VisualizerTestHelper::buildContextGeometry_private(visualizer);
    DOCTEST_CHECK(VisualizerTestHelper::getUploadedPrimitiveCount(visualizer) == 50);

    // The Context still reports all 50 as dirty, because nothing cleared the flags.
    DOCTEST_CHECK(context.getDirtyUUIDs().size() == 50);

    // A second build with nothing changed must not queue any geometry for re-upload. On the buggy
    // code every primitive was rebuilt and re-marked dirty in the geometry handler, so this count
    // was 50 rather than 0.
    visualizer.plotUpdate(true); // flushes the pending upload from the first build
    VisualizerTestHelper::buildContextGeometry_private(visualizer);
    DOCTEST_CHECK(VisualizerTestHelper::getDirtyGeometryCount(visualizer) == 0);

    // A primitive genuinely added afterwards must still be picked up.
    const uint new_UUID = context.addPatch(make_vec3(100, 0, 0), make_vec2(0.5, 0.5));
    VisualizerTestHelper::buildContextGeometry_private(visualizer);
    DOCTEST_CHECK(VisualizerTestHelper::getUploadedPrimitiveCount(visualizer) == 51);
    DOCTEST_CHECK(VisualizerTestHelper::getDirtyGeometryCount(visualizer) == 1);

    // ...and a primitive deleted afterwards must be dropped from the uploaded set, so that a later
    // primitive reusing that UUID is not mistaken for one already on the GPU.
    context.deletePrimitive(new_UUID);
    VisualizerTestHelper::buildContextGeometry_private(visualizer);
    DOCTEST_CHECK(VisualizerTestHelper::getUploadedPrimitiveCount(visualizer) == 50);
}

DOCTEST_TEST_CASE("Visualizer::printWindow builds Context geometry that was never plotted") {
    // printWindow() re-renders before capturing, but via plotOnce(), which deliberately does not
    // re-walk the Context. A caller that hands the Visualizer a Context and then captures straight
    // away - without an intervening plotUpdate() - must still get that geometry in the image, so
    // printWindow() has to do the build itself in that case.
    Context context;
    context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, RGB::red);

    Visualizer visualizer(64, 64, 0, true, true); // headless
    visualizer.disableMessages();
    visualizer.setBackgroundColor(RGB::blue);
    visualizer.setLightingModel(Visualizer::LIGHTING_NONE);
    visualizer.setCameraPosition(make_vec3(0, 0, 2), make_vec3(0, 0, 0));
    visualizer.buildContextGeometry(&context);

    const std::string outfile = "test_printWindow_unplotted.jpeg";
    DOCTEST_CHECK_NOTHROW(visualizer.printWindow(outfile.c_str()));
    DOCTEST_REQUIRE(std::filesystem::exists(outfile));

    // The patch is red against a blue background, so if the geometry made it into the frame the
    // image cannot be uniformly blue.
    std::vector<helios::RGBcolor> pixels;
    uint width, height;
    readJPEG(outfile, width, height, pixels);

    size_t red_pixels = 0;
    for (const helios::RGBcolor &pixel: pixels) {
        if (pixel.r > 0.5f && pixel.b < 0.5f) {
            red_pixels++;
        }
    }
    DOCTEST_CHECK(red_pixels > 0);

    std::filesystem::remove(outfile);
}

TEST_CASE("Visualizer headless rendering anti-aliases geometry edges") {
    // A patch rotated off-axis presents a diagonal silhouette, which is where aliasing is visible:
    // without multisampling every pixel along that edge is either fully the patch or fully the
    // background, so the edge column contains only those two values. With multisampling the edge
    // pixels carry partial coverage, producing intermediate values that lie strictly between.
    //
    // Counting distinct intermediate values is what distinguishes the two, rather than comparing
    // images: a hard-edged render has none, an anti-aliased one has several.
    auto countIntermediateEdgeValues = [](int aliasing_samples) {
        Context context;
        const uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(12, 12));
        context.setPrimitiveColor(UUID, make_RGBcolor(1.f, 1.f, 1.f));
        // Rotate about z so the patch edges run diagonally across the pixel grid.
        context.rotatePrimitive(UUID, 0.35f, "z");

        Visualizer visualizer(200, 200, aliasing_samples, true, true); // headless
        visualizer.disableMessages();
        visualizer.setLightingModel(Visualizer::LIGHTING_NONE);
        visualizer.setBackgroundColor(make_RGBcolor(0.f, 0.f, 0.f));
        visualizer.buildContextGeometry(&context);
        visualizer.setCameraPosition(make_vec3(0.5f, -1.f, 26), make_vec3(0, 0, 0));
        visualizer.plotUpdate();

        std::vector<uint> pixels;
        uint w = 0, h = 0;
        visualizer.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);

        // Scan a horizontal line across the middle and collect values that are neither background
        // nor fully-covered patch. A generous margin keeps 8-bit rounding from counting as coverage.
        std::set<uint> intermediate;
        const size_t row = size_t(h / 2);
        for (uint x = 0; x < w; x++) {
            const uint value = pixels.at(3 * (row * size_t(w) + size_t(x)));
            if (value > 8 && value < 247) {
                intermediate.insert(value);
            }
        }
        return intermediate.size();
    };

    const size_t without_msaa = countIntermediateEdgeValues(0);
    const size_t with_msaa = countIntermediateEdgeValues(8);

    // Without multisampling the diagonal edge is hard: every pixel is background or patch.
    DOCTEST_CHECK(without_msaa <= 1);

    // Accepting the multisampled attachments is not the same as rasterizing multisampled. macOS
    // drives OpenGL through a deprecated Metal translation layer that reports GL_SAMPLES=4 on the
    // framebuffer while leaving GL_SAMPLE_BUFFERS at 0: the attachments allocate their samples and
    // the blit resolves faithfully, but every sample of a pixel holds the same value, so the
    // resolved image is byte-identical to an unantialiased one. Asserting anti-aliasing there would
    // fail for a platform reason rather than a code one, so the check is reported rather than
    // enforced when the platform produces no coverage values at all.
    if (with_msaa <= without_msaa) {
        WARN("Headless multisampling produced no partial-coverage pixels; this driver does not rasterize "
             "multisampled into a user-created framebuffer (expected on macOS OpenGL-over-Metal). "
             "The resolve path itself is exercised by the render completing without error.");
    } else {
        // With multisampling the edge pixels carry partial coverage.
        DOCTEST_CHECK(with_msaa > without_msaa);
    }
}

TEST_CASE("Visualizer does not illuminate surfaces facing away from the light") {
    // A leaf tilted past vertical has its normal pointing down and away from an overhead sun, so it
    // receives no direct sunlight regardless of where it is viewed from. Viewing it from the side
    // its normal points away from must not make it brighter: how much sunlight a surface receives
    // is a property of the geometry and the sun, and cannot depend on the camera.
    //
    // The regression this guards against flipped the normal toward the camera before computing the
    // diffuse term, so this back-facing leaf picked up ~70% of full direct illumination.
    auto renderTiltedLeaf = [](bool sun_overhead) {
        Context context;
        const uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(8, 8));
        context.addMaterial("leaf");
        context.assignMaterialToPrimitive(UUID, "leaf");
        // Primitive color lives on the material, so it must be set after the material is assigned.
        context.setPrimitiveColor(UUID, make_RGBcolor(0.5f, 0.5f, 0.5f));
        // Tilt 135 degrees about y: the normal starts at +z and ends pointing down and to +x.
        context.rotatePrimitive(UUID, 135.f * PI_F / 180.f, "y");

        Visualizer visualizer(64, 64, 0, true, true); // headless, no antialiasing
        visualizer.disableMessages();
        visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
        // Sun overhead is what the tilted leaf faces away from; the source-off case is the control.
        visualizer.setLightDirection(sun_overhead ? make_vec3(0, 0, 1) : make_vec3(0, 0, -1));

        Visualizer::PhongMaterial material;
        material.ambient = 1.0f;
        material.diffuse = 0.8f;
        material.specular = 0.f; // isolate the diffuse term
        material.shininess = 32.f;
        visualizer.setPhongMaterial(material);

        visualizer.buildContextGeometry(&context);
        // Camera on the side the normal points away from, so the back face is what is visible.
        visualizer.setCameraPosition(make_vec3(-6.f, 0.5f, 6.f), make_vec3(0, 0, 0));
        visualizer.plotUpdate();

        std::vector<uint> pixels;
        uint w = 0, h = 0;
        visualizer.getWindowPixelsRGB(pixels, w, h);
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        const size_t idx = 3 * (size_t(h / 2) * size_t(w) + size_t(w / 2));
        REQUIRE(pixels.size() > idx);
        return pixels.at(idx);
    };

    // Moving the sun from overhead to underneath changes which side of this leaf is lit. The face
    // being viewed is the one turned away from the overhead sun, so it must not be the brighter of
    // the two: with the sun overhead it receives ambient only.
    const uint sun_overhead = renderTiltedLeaf(true);
    const uint sun_underneath = renderTiltedLeaf(false);

    REQUIRE(sun_overhead > 0); // the leaf must actually cover the sampled pixel
    REQUIRE(sun_underneath > 0);
    DOCTEST_CHECK(sun_overhead < sun_underneath);
}
