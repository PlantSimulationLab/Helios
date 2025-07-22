#include "Visualizer.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

using namespace helios;

TEST_CASE("Visualizer::disableMessages") {
    Visualizer visualizer(1000, true);

    DOCTEST_CHECK_NOTHROW(visualizer.disableMessages());

    capture_cerr cerr_buffer;
    visualizer.setColorbarRange(20, 10);

    DOCTEST_CHECK(!cerr_buffer.has_output());
}

TEST_CASE("Visualizer::enableMessages") {
    Visualizer visualizer(1000, true);

    DOCTEST_CHECK_NOTHROW(visualizer.enableMessages());

    capture_cerr cerr_buffer;
    visualizer.setColorbarRange(20, 10);

    DOCTEST_CHECK(cerr_buffer.has_output());
}

TEST_CASE("Visualizer::setCameraPosition") {
    Visualizer visualizer(1000, true);
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
    Visualizer visualizer(1000, true);

    DOCTEST_CHECK_NOTHROW(visualizer.setLightingModel(Visualizer::LIGHTING_NONE));
    DOCTEST_CHECK_NOTHROW(visualizer.setLightingModel(Visualizer::LIGHTING_PHONG));
    DOCTEST_CHECK_NOTHROW(visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED));
}

TEST_CASE("Visualizer::setBackgroundColor and Visualizer::getBackgroundColor") {
    Visualizer visualizer(1000, true);
    helios::RGBcolor bgcolor = RGB::white;
    visualizer.setBackgroundColor(bgcolor);
    DOCTEST_CHECK(visualizer.getBackgroundColor() == bgcolor);
}

TEST_CASE("Visualizer::setLightIntensityFactor") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setLightIntensityFactor(0.75f));
}

TEST_CASE("Visualizer::enableColorbar and Visualizer::disableColorbar") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK_NOTHROW(visualizer.enableColorbar());
    DOCTEST_CHECK_NOTHROW(visualizer.disableColorbar());
}

TEST_CASE("Visualizer::setColorbarPosition") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarPosition(make_vec3(0.5f, 0.5f, 0.f)));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarPosition(make_vec3(-0.1f, 0.f, 0.f)), std::runtime_error);
}

TEST_CASE("Visualizer::setColorbarSize") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarSize(make_vec2(0.1f, 0.05f)));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarSize(make_vec2(1.5f, 0.f)), std::runtime_error);
}

TEST_CASE("Visualizer::setColorbarRange") {
    Visualizer visualizer(1000, true);
    visualizer.enableMessages();
    visualizer.setColorbarRange(0.f, 1.f);
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarRange(20.f, 10.f));
    DOCTEST_CHECK(cerr_buffer.has_output());
}

TEST_CASE("Visualizer::setColorbarTicks") {
    Visualizer visualizer(1000, true);
    visualizer.setColorbarRange(0.f, 1.f);
    std::vector<float> ticks{0.f, 0.5f, 1.f};
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarTicks(ticks));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarTicks({}), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarTicks({0.f, 0.5f, 0.4f}), std::runtime_error);
}

TEST_CASE("Visualizer colorbar text attributes") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarTitle("MyBar"));
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarFontColor(RGB::yellow));
    DOCTEST_CHECK_NOTHROW(visualizer.setColorbarFontSize(14));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColorbarFontSize(0), std::runtime_error);
}

TEST_CASE("Visualizer::setColormap") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK_NOTHROW(visualizer.setColormap(Visualizer::COLORMAP_COOL));
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(visualizer.setColormap(Visualizer::COLORMAP_CUSTOM), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(visualizer.setColormap(std::vector<RGBcolor>{RGB::red}, std::vector<float>{0.f, 1.f}), std::runtime_error);
}

TEST_CASE("Visualizer::PNG texture integration via primitives") {
    Visualizer visualizer(1000, true);
    const char* png_filename = "plugins/visualizer/textures/AlmondLeaf.png";
    
    // Verify file exists before testing
    DOCTEST_CHECK(std::filesystem::exists(png_filename));
    
    // Test PNG texture loading through textured rectangle - internally calls read_png_file -> helios::readPNG
    std::vector<helios::vec3> verts = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
    size_t UUID1;
    DOCTEST_CHECK_NOTHROW(UUID1 = visualizer.addRectangleByVertices(verts, png_filename, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID1 != 0);
    
    // Test PNG texture loading through textured triangle
    size_t UUID2;
    DOCTEST_CHECK_NOTHROW(UUID2 = visualizer.addTriangle(
        make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), 
        png_filename, make_vec2(0, 0), make_vec2(1, 0), make_vec2(0, 1), 
        Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID2 != 0);
    DOCTEST_CHECK(UUID2 != UUID1); // Should be different primitives
}

TEST_CASE("Visualizer::JPEG texture integration via primitives") {
    Visualizer visualizer(1000, true);
    const char* jpeg_filename = "plugins/visualizer/textures/SkyDome_clouds.jpg";
    
    // Verify file exists before testing
    DOCTEST_CHECK(std::filesystem::exists(jpeg_filename));
    
    // Test JPEG texture loading through sky dome - internally calls read_JPEG_file -> helios::readJPEG
    uint N = 3;
    std::vector<size_t> UUIDs;
    DOCTEST_CHECK_NOTHROW(UUIDs = visualizer.addSkyDomeByCenter(5.0f, make_vec3(0, 0, 0), N, jpeg_filename));
    DOCTEST_CHECK(UUIDs.size() == (N - 1) * (2 * (N - 1) + 1));
    
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
    Visualizer visualizer(1000, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addRectangleByCenter(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addRectangleByCenter extreme") {
    Visualizer visualizer(1000, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addRectangleByCenter(make_vec3(1e6, 1e6, 1e6), make_vec2(1e6, 1e6), make_SphericalCoord(0, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addRectangleByVertices variations") {
    Visualizer visualizer(1000, true);
    std::vector<helios::vec3> verts = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
    size_t UUID1;
    DOCTEST_CHECK_NOTHROW(UUID1 = visualizer.addRectangleByVertices(verts, RGB::blue, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID1 != 0);
    size_t UUID2;
    DOCTEST_CHECK_NOTHROW(UUID2 = visualizer.addRectangleByVertices(verts, "plugins/visualizer/textures/AlmondLeaf.png", Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID2 != 0);
}

TEST_CASE("Visualizer::addTriangle") {
    Visualizer visualizer(1000, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::blue, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addTriangle textured") {
    Visualizer visualizer(1000, true);
    size_t UUID;
    DOCTEST_CHECK_NOTHROW(UUID = visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), "plugins/visualizer/textures/AlmondLeaf.png", make_vec2(0, 0), make_vec2(1, 0), make_vec2(0, 1), Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUID != 0);
}

TEST_CASE("Visualizer::addVoxelByCenter") {
    Visualizer visualizer(1000, true);
    std::vector<size_t> UUIDs;
    DOCTEST_CHECK_NOTHROW(UUIDs = visualizer.addVoxelByCenter(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_SphericalCoord(0, 0), RGB::green, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUIDs.size() == 6);
}

TEST_CASE("Visualizer::addSphereByCenter") {
    Visualizer visualizer(1000, true);
    uint N = 3;
    std::vector<size_t> UUIDs;
    DOCTEST_CHECK_NOTHROW(UUIDs = visualizer.addSphereByCenter(1.0f, make_vec3(0, 0, 0), N, RGB::blue, Visualizer::COORDINATES_CARTESIAN));
    DOCTEST_CHECK(UUIDs.size() == 2 * N * (N - 1));
}

TEST_CASE("Visualizer::addSkyDomeByCenter") {
    Visualizer visualizer(1000, true);
    uint N = 3;
    std::vector<size_t> UUIDs;
    DOCTEST_CHECK_NOTHROW(UUIDs = visualizer.addSkyDomeByCenter(5.0f, make_vec3(0, 0, 0), N, "plugins/visualizer/textures/SkyDome_clouds.jpg"));
    DOCTEST_CHECK(UUIDs.size() == (N - 1) * (2 * (N - 1) + 1));
}

TEST_CASE("Visualizer::addCoordinateAxes") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK_NOTHROW(visualizer.addCoordinateAxes(make_vec3(0, 0, 0), make_vec3(1, 1, 1), "XYZ"));
}

TEST_CASE("Visualizer::addLine") {
    Visualizer visualizer(1000, true);
    DOCTEST_CHECK(visualizer.addLine(make_vec3(-1, 3, 0), make_vec3(0, 4, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN) != 0);
}

TEST_CASE("Visualizer::validateTextureFile") {
    DOCTEST_CHECK(validateTextureFile("plugins/visualizer/textures/AlmondLeaf.png"));
    DOCTEST_CHECK(!validateTextureFile("missing.png"));
    DOCTEST_CHECK(!validateTextureFile("plugins/visualizer/textures/SkyDome_clouds.jpg", true));
}

int Visualizer::selfTest() {

    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}
