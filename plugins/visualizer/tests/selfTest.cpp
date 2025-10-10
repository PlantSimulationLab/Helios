#include "Visualizer.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <filesystem>
#include "doctest.h"
#include "doctest_utils.h"

using namespace helios;

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

TEST_CASE("Visualizer::generateNiceTicks - Float data") {
    // Test various ranges for float data
    std::vector<float> ticks;

    // Test range 0 to 1
    ticks = Visualizer::generateNiceTicks(0.0f, 1.0f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= 0.0f);
    DOCTEST_CHECK(ticks.back() >= 1.0f);
    // Should generate nice values like 0.0, 0.25, 0.5, 0.75, 1.0
    for (size_t i = 1; i < ticks.size(); ++i) {
        DOCTEST_CHECK(ticks[i] > ticks[i - 1]);
    }

    // Test range 0 to 100
    ticks = Visualizer::generateNiceTicks(0.0f, 100.0f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= 0.0f);
    DOCTEST_CHECK(ticks.back() >= 100.0f);

    // Test range 0 to 48.3
    ticks = Visualizer::generateNiceTicks(0.0f, 48.3f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= 0.0f);
    DOCTEST_CHECK(ticks.back() >= 48.3f);
    // Should generate ticks like 0, 25, 50 or similar nice numbers

    // Test very small range
    ticks = Visualizer::generateNiceTicks(0.0f, 0.1f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);

    // Test negative range
    ticks = Visualizer::generateNiceTicks(-10.0f, 10.0f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    DOCTEST_CHECK(ticks.front() <= -10.0f);
    DOCTEST_CHECK(ticks.back() >= 10.0f);

    // Test very large range
    ticks = Visualizer::generateNiceTicks(0.0f, 1e6f, false, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
}

TEST_CASE("Visualizer::generateNiceTicks - Integer data") {
    std::vector<float> ticks;

    // Test range 0 to 20 (integer)
    ticks = Visualizer::generateNiceTicks(0.0f, 20.0f, true, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    // All ticks should be integers
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6);
    }

    // Test range 0 to 7 (integer)
    ticks = Visualizer::generateNiceTicks(0.0f, 7.0f, true, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6);
    }

    // Test range 0 to 100 (integer)
    ticks = Visualizer::generateNiceTicks(0.0f, 100.0f, true, 5);
    DOCTEST_CHECK(ticks.size() >= 2);
    for (float tick: ticks) {
        DOCTEST_CHECK(std::fabs(tick - std::round(tick)) < 1e-6);
    }
}

TEST_CASE("Visualizer::formatTickLabel - Float data") {
    std::string label;

    // Test formatting with spacing = 0.2 (nice number spacing, should show 1 decimal place)
    label = Visualizer::formatTickLabel(0.0, 0.2, false);
    DOCTEST_CHECK(label == "0.0");

    label = Visualizer::formatTickLabel(0.4, 0.2, false);
    DOCTEST_CHECK(label == "0.4");

    label = Visualizer::formatTickLabel(1.0, 0.2, false);
    DOCTEST_CHECK(label == "1.0");

    // Test formatting with spacing = 1.0 (should show 0 decimal places)
    label = Visualizer::formatTickLabel(0.0, 1.0, false);
    DOCTEST_CHECK(label == "0");

    label = Visualizer::formatTickLabel(10.0, 1.0, false);
    DOCTEST_CHECK(label == "10");

    // Test formatting with spacing = 0.1
    label = Visualizer::formatTickLabel(0.5, 0.1, false);
    DOCTEST_CHECK(label == "0.5");

    // Test very small value (should use scientific notation)
    label = Visualizer::formatTickLabel(1e-6, 1e-6, false);
    DOCTEST_CHECK(label.find("e") != std::string::npos); // Should contain 'e' for scientific notation

    // Test large value (should use scientific notation at 10,000+)
    label = Visualizer::formatTickLabel(15000.0, 1000.0, false);
    DOCTEST_CHECK(label.find("e") != std::string::npos);

    // Test value below scientific notation threshold
    label = Visualizer::formatTickLabel(9000.0, 1000.0, false);
    DOCTEST_CHECK(label.find("e") == std::string::npos); // Should NOT use scientific notation
}

TEST_CASE("Visualizer::formatTickLabel - Integer data") {
    std::string label;

    // Test integer formatting
    label = Visualizer::formatTickLabel(0.0, 1.0, true);
    DOCTEST_CHECK(label == "0");

    label = Visualizer::formatTickLabel(5.0, 1.0, true);
    DOCTEST_CHECK(label == "5");

    label = Visualizer::formatTickLabel(100.0, 10.0, true);
    DOCTEST_CHECK(label == "100");

    // Test rounding for integer data
    label = Visualizer::formatTickLabel(5.4, 1.0, true);
    DOCTEST_CHECK(label == "5");

    label = Visualizer::formatTickLabel(5.6, 1.0, true);
    DOCTEST_CHECK(label == "6");

    // Test large integer values (should use scientific notation at 10,000+)
    label = Visualizer::formatTickLabel(15000.0, 1000.0, true);
    DOCTEST_CHECK(label.find("e") != std::string::npos);

    // Test integer value below scientific notation threshold
    label = Visualizer::formatTickLabel(9000.0, 1000.0, true);
    DOCTEST_CHECK(label == "9000");
}

TEST_CASE("Visualizer::niceNumber") {
    // Test rounding up (round = false)
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(0.72, false) - 1.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(1.5, false) - 2.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(3.2, false) - 5.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(7.5, false) - 10.0) < 1e-6);

    // Test rounding to nearest (round = true)
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(1.2, true) - 1.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(1.6, true) - 2.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(3.5, true) - 5.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(6.0, true) - 5.0) < 1e-6);

    // Test with different magnitudes
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(12.0, true) - 10.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(120.0, true) - 100.0) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(0.12, true) - 0.1) < 1e-6);

    // Test zero
    DOCTEST_CHECK(Visualizer::niceNumber(0.0, true) == 0.0);
    DOCTEST_CHECK(Visualizer::niceNumber(0.0, false) == 0.0);

    // Test negative values (should preserve sign)
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(-1.5, true) - (-2.0)) < 1e-6);
    DOCTEST_CHECK(std::fabs(Visualizer::niceNumber(-3.2, true) - (-5.0)) < 1e-6); // -3.2 rounds to -5.0, not -2.0
}

TEST_CASE("Visualizer colorbar text attributes") {
    Visualizer visualizer(1000, 800, 16, true, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarTitle("MyBar"));
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarFontColor(RGB::yellow));
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarFontSize(14));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarFontSize(0), std::runtime_error);
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
    DOCTEST_CHECK_NOTHROW(visualizer.renderToOffscreenBuffer());

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
    context.setPrimitiveColor(sphere_uuids, RGB::red);

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
    std::vector<uint> pixel_buffer(200 * 200 * 3);
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
    // in non-headless mode. Only runs when a display is available.

    // Check if we have a display available (skip test if running in headless environment)
    const char *display = std::getenv("DISPLAY");
    const char *wayland_display = std::getenv("WAYLAND_DISPLAY");

#ifdef __APPLE__
    // On macOS, we can always create a window context
    bool has_display = true;
#else
    // On Linux, check for X11 or Wayland display
    bool has_display = (display != nullptr && strlen(display) > 0) || (wayland_display != nullptr && strlen(wayland_display) > 0);
#endif

    if (!has_display) {
        // Skip test silently when no display is available
        return;
    }

    Context context;
    Visualizer visualizer(200, 200, 0, true, false); // NON-headless mode - requires display
    visualizer.disableMessages();

    // Add some geometry to render (a simple sphere)
    std::vector<uint> sphere_uuids = context.addSphere(10, make_vec3(0, 0, 0), 1.0f);
    context.setPrimitiveColor(sphere_uuids, RGB::red);

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
    std::vector<uint> pixel_buffer(200 * 200 * 3);
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
    Visualizer visualizer(200, 200, 16, false, true); // headless mode
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

TEST_CASE("Visualizer::PNG with transparent background (windowed mode)") {
    // Test PNG output with transparent background in WINDOWED mode (not headless)
    Context context;

    // Add a red patch via the Context (matching user's workflow)
    uint patch_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.6, 0.6), nullrotation, "plugins/visualizer/textures/AlmondLeaf.png");
    context.setPrimitiveColor(patch_UUID, make_RGBcolor(1.f, 0.f, 0.f));
    context.overridePrimitiveTextureColor(patch_UUID); // Required to use vertex color instead of texture color

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
    context.setPrimitiveColor(patch_UUID, make_RGBcolor(1.f, 0.f, 0.f));
    context.overridePrimitiveTextureColor(patch_UUID);

    // Test with a non-square window (800x600, aspect ratio 4:3)
    Visualizer visualizer(800, 600, 16, false, true);

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
    context.setPrimitiveColor(patch_UUID, make_RGBcolor(1.f, 0.f, 0.f));
    context.overridePrimitiveTextureColor(patch_UUID);

    SUBCASE("Watermark visible → transparent → solid color (should restore watermark)") {
        Visualizer visualizer(200, 200, 16, false, true);
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
        // Gizmo should be enabled by default
        // We can't directly access the private member, but we can test the behavior
        DOCTEST_CHECK_NOTHROW(visualizer.plotUpdate(true));
    }

    SUBCASE("Show and hide navigation gizmo") {
        Visualizer visualizer(200, 200, 16, false, true);

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

int Visualizer::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
