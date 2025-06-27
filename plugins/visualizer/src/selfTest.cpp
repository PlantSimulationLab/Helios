#include "Visualizer.h"

using namespace helios;

int Visualizer::selfTest() {
    int error_count = 0; // Track total errors

    if (message_flag) {
        std::cout << "Running visualizer self-test..." << std::flush;
    }

    Visualizer visualizer(1000);

    // ---- Test enableMessages() ---- //
    {
        visualizer.enableMessages();
        if (!visualizer.message_flag) {
            std::cerr << "failed enableMessages() test." << std::endl;
            error_count++;
        }
    }

    // ---- Test disableMessages() ---- //
    {
        visualizer.disableMessages();
        if (visualizer.message_flag) {
            std::cerr << "failed disableMessages() test." << std::endl;
            error_count++;
        }
    }

    // ---- Test openWindow() ---- //
    {
        try {
            visualizer.openWindow();
            if (visualizer.window == nullptr) {
                std::cerr << "failed openWindow() test - window is null." << std::endl;
                error_count++;
            }
        } catch (const std::exception &e) {
            std::cerr << "failed openWindow() test: " << e.what() << std::endl;
            error_count++;
        }
    }

    // ---- Test setCameraPosition() ---- //
    {
        helios::vec3 initial_position = make_vec3(1, 1, 1);
        helios::vec3 initial_lookat = make_vec3(0, 0, 0);
        visualizer.setCameraPosition(initial_position, initial_lookat);

        // Verify that the transformation matrix updates correctly
        if (visualizer.camera_eye_location != initial_position ||
            visualizer.camera_lookat_center != initial_lookat) {
            std::cerr << "failed setCameraPosition() test." << std::endl;
            error_count++;
        }
    }

    // ---- Test setLightingModel() ---- //
    {
        visualizer.setLightingModel(Visualizer::LIGHTING_NONE);
        visualizer.setLightingModel(Visualizer::LIGHTING_PHONG);
        visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

        // Edge case: Casting an integer to LightingModel to simulate an unknown value
        visualizer.setLightingModel(static_cast<Visualizer::LightingModel>(9999));
    }


    // ---- Test initialize() ---- //
    {
        try {
            visualizer.initialize(800, 600, 4, true);
            visualizer.initialize(1024, 768, 8, false);
            visualizer.initialize(1280, 720, 0, false); // Edge case: No aliasing
        } catch (const std::exception &e) {
            std::cerr << "failed initialize() test: " << e.what() << std::endl;
            error_count++;
        }
    }

    // ---- Test read_png_file() ---- //
    {
        std::vector<unsigned char> texture;
        uint height = 0, width = 0;
        std::string filename = "plugins/visualizer/textures/Almondleaf.png";  // Provide a valid PNG test file
        read_png_file(filename.c_str(), texture, height, width);

        if (texture.empty() || height == 0 || width == 0) {
            std::cerr << "failed read_png_file() test." << std::endl;
            error_count++;
        }
    }


    // ---- Test Writing JPEG File (Window) ---- //
    {
        visualizer.openWindow();
        if (visualizer.window != nullptr) {
            int result = write_JPEG_file("plugins/visualizer/textures/dirt.jpg", 800, 600, visualizer.window);
            if (result != 1) {
                std::cerr << "ERROR: write_JPEG_file(window) failed.\n";
                error_count++;
            }
        }
    }

    // ---- Test Writing JPEG File (Pixel Data) ---- //
    {
        uint width = 100, height = 100;
        std::vector<helios::RGBcolor> data(width * height, helios::RGBcolor(1.0, 0.0, 0.0));
        int result = write_JPEG_file("plugins/visualizer/textures/dirt.jpg", width, height, data);
        if (result != 1) {
            std::cerr << "ERROR: write_JPEG_file(data) failed.\n";
            error_count++;
        }
    }

    // ---- Test PNG File Reading (Error Handling) ---- //
    {
        std::vector<unsigned char> texture;
        uint height = 0, width = 0;
        std::string filename = "invalid.png";

        try {
            read_png_file(filename.c_str(), texture, height, width);
            std::cerr << "ERROR: read_png_file should have failed but didn't.\n";
            error_count++;
        } catch (...) {
            std::cerr << "Passed read_png_file() error handling test.\n";
        }
    }



    // ---- Test Visualizer Constructors ---- //
    {
        try {
            Visualizer v1(800, 600);
            Visualizer v2(1024, 768, 4);
            Visualizer v3(1280, 720, 8, false);
            std::cout << "Passed Visualizer constructors test.\n";
        } catch (...) {
            std::cerr << "ERROR: Visualizer constructors test failed.\n";
            error_count++;
        }
    }


    // ---- Test printWindow() ---- //
    {
        try {
            visualizer.printWindow();
            std::cout << "Passed printWindow() test.\n";
        } catch (...) {
            std::cerr << "ERROR: printWindow() test failed.\n";
            error_count++;
        }
    }

    // ---- Test setLightDirection() ---- //
    {
        try {
            visualizer.setLightDirection(make_vec3(1, 1, 1));
            std::cout << "Passed setLightDirection() test.\n";
        } catch (...) {
            std::cerr << "ERROR: setLightDirection() test failed.\n";
            error_count++;
        }
    }

    // ---- Test getDomainBoundingBox() ---- //
    {
        try {
            vec2 xbounds, ybounds, zbounds;
            visualizer.getDomainBoundingBox(xbounds, ybounds, zbounds);
            std::cout << "Passed getDomainBoundingBox() test.\n";
        } catch (...) {
            std::cerr << "ERROR: getDomainBoundingBox() test failed.\n";
            error_count++;
        }
    }

    // ---- Test getDomainBoundingRadius() ---- //
    {
        try {
            float radius = visualizer.getDomainBoundingRadius();
            std::cout << "Passed getDomainBoundingRadius() test.\n";
        } catch (...) {
            std::cerr << "ERROR: getDomainBoundingRadius() test failed.\n";
            error_count++;
        }
    }

    // ---- Test updatePerspectiveTransformation() ---- //
    {
        try {
            visualizer.updatePerspectiveTransformation(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
            std::cout << "Passed updatePerspectiveTransformation() test.\n";
        } catch (...) {
            std::cerr << "ERROR: updatePerspectiveTransformation() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addRectangleByCenter() ---- //
    {
        try {
            visualizer.addRectangleByCenter(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN);
            std::cout << "Passed addRectangleByCenter() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addRectangleByCenter() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addRectangleByCenter() with extreme values ---- //
    {
        try {
            visualizer.addRectangleByCenter(make_vec3(1e6, 1e6, 1e6), make_vec2(1e6, 1e6), make_SphericalCoord(0, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN);
            std::cout << "PASSED: addRectangleByCenter() executed successfully with extreme values." << std::endl;
        } catch (...) {
            std::cerr << "FAILED: addRectangleByCenter() threw an exception on extreme values." << std::endl;
            error_count++;
        }
    }

    // ---- Test addTriangle() ---- //
    {
        try {
            visualizer.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::blue, Visualizer::COORDINATES_CARTESIAN);
            std::cout << "Passed addTriangle() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addTriangle() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addVoxelByCenter() ---- //
    {
        try {
            visualizer.addVoxelByCenter(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_SphericalCoord(0, 0), RGB::green, Visualizer::COORDINATES_CARTESIAN);
            std::cout << "Passed addVoxelByCenter() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addVoxelByCenter() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addSphereByCenter() ---- //
    {
        try {
            visualizer.addSphereByCenter(1.0f, make_vec3(0, 0, 0), 10, RGB::blue, Visualizer::COORDINATES_CARTESIAN);
            std::cout << "Passed addSphereByCenter() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addSphereByCenter() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addSkyDomeByCenter() ---- //
    {
        try {
            visualizer.addSkyDomeByCenter(5.0f, make_vec3(0, 0, 0), 10, "plugins/visualizer/textures/SkyDome_clouds.jpg");
            std::cout << "Passed addSkyDomeByCenter() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addSkyDomeByCenter() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addColorbarByCenter() ---- //
    {
        try {
            std::vector<helios::RGBcolor> colors = { RGB::red, RGB::green, RGB::blue };
            std::vector<float> positions = { 0.0f, 0.5f, 1.0f };

            Colormap colormap;
            colormap.set(colors, positions, 100, 0.0f, 1.0f);  // Correct initialization

            // Adjusting position to ensure it stays within bounds
            helios::vec3 safe_position = make_vec3(0.5, 0.1, 0.1);
            helios::vec2 safe_size = make_vec2(0.1, 0.05);

            visualizer.addColorbarByCenter("Colorbar", safe_size, safe_position, RGB::black, colormap);
            std::cout << "Passed addColorbarByCenter() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addColorbarByCenter() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addCoordinateAxes() ---- //
    {
        try {
            visualizer.addCoordinateAxes(make_vec3(0, 0, 0), make_vec3(1, 1, 1), "XYZ");
            std::cout << "Passed addCoordinateAxes() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addCoordinateAxes() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addDiskByCenter() ---- //
    {
        try {
            visualizer.addDiskByCenter(make_vec3(0, 3, 0), make_vec2(sqrtf(2) / 2.f, sqrtf(2) / 2.f), make_SphericalCoord(0, 0), 50, RGB::blue, Visualizer::COORDINATES_CARTESIAN);
            std::cout << "Passed addDiskByCenter() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addDiskByCenter() test failed.\n";
            error_count++;
        }
    }

    // ---- Test addLine() ---- //
    {
        try {
            visualizer.addLine(make_vec3(-1, 3, 0), make_vec3(0, 4, 0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN);
            visualizer.addLine(make_vec3(0, 4, 0), make_vec3(1, 3, 0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN);
            visualizer.addLine(make_vec3(1, 3, 0), make_vec3(0, 2, 0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN);
            visualizer.addLine(make_vec3(0, 2, 0), make_vec3(-1, 3, 0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN);
            std::cout << "Passed addLine() test.\n";
        } catch (...) {
            std::cerr << "ERROR: addLine() test failed.\n";
            error_count++;
        }
    }





    // ---- Final Test Summary ---- //
    if (error_count > 0) {
        std::cout << "completed with " << error_count << " errors." << std::endl;
    } else {
        std::cerr << "passed all tests." << std::endl;
    }

    return error_count;
}
