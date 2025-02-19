#include "SyntheticAnnotation.h"
#include "Context.h"
#include <iostream>


using namespace helios;

int SyntheticAnnotation::selfTest()const {
    std::cout << "Running synthetic annotation self-test..." << std::flush;

    int error_count = 0; // Track errors

    // ---- Test RGB Encoding ---- //
    {
        SyntheticAnnotation test_instance(nullptr);  // Create a test instance

        int ID1 = 77830;
        int ID2 = test_instance.rgb2int(test_instance.int2rgb(ID1));
        if (ID1 != ID2) {
            std::cerr << "FAILED: RGB encoding scheme incorrect." << std::endl;
            error_count++;
        }
    }

    // ---- Test Initialization ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        // Ensure object initializes correctly
        if (!test_instance.context) {
            std::cerr << "FAILED: context is null in test instance." << std::endl;
            error_count++;
        }
    }

    // ---- Test setBackgroundColor ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        helios::RGBcolor originalColor = helios::make_RGBcolor(0.9, 0.9, 0.9);
        test_instance.setBackgroundColor(helios::make_RGBcolor(0.5, 0.5, 0.5));

        if (test_instance.background_color.r != 0.5 ||
            test_instance.background_color.g != 0.5 ||
            test_instance.background_color.b != 0.5) {
            std::cerr << "FAILED: setBackgroundColor()" << std::endl;
            error_count++;
        }

        // Restore original color
        test_instance.setBackgroundColor(originalColor);
    }

    // ---- Test setWindowSize ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        test_instance.setWindowSize(1280, 720);
        if (test_instance.window_width != 1280 || test_instance.window_height != 720) {
            std::cerr << "FAILED: setWindowSize()" << std::endl;
            error_count++;
        }
    }

    // ---- Test setCameraPosition ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        helios::vec3 cam_pos(1, 1, 1);
        helios::vec3 cam_lookat(0, 0, 0);
        test_instance.setCameraPosition(cam_pos, cam_lookat);

        if (test_instance.camera_position.empty() || test_instance.camera_lookat.empty()) {
            std::cerr << "FAILED: setCameraPosition()" << std::endl;
            error_count++;
        }
    }

    // ---- Test Enable/Disable Object Detection ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        test_instance.enableObjectDetection();
        if (!test_instance.objectdetection_enabled) {
            std::cerr << "FAILED: enableObjectDetection()" << std::endl;
            error_count++;
        }

        test_instance.disableObjectDetection();
        if (test_instance.objectdetection_enabled) {
            std::cerr << "FAILED: disableObjectDetection()" << std::endl;
            error_count++;
        }
    }

    // ---- Test Enable/Disable Semantic Segmentation ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        test_instance.enableSemanticSegmentation();
        if (!test_instance.semanticsegmentation_enabled) {
            std::cerr << "FAILED: enableSemanticSegmentation()" << std::endl;
            error_count++;
        }

        test_instance.disableSemanticSegmentation();
        if (test_instance.semanticsegmentation_enabled) {
            std::cerr << "FAILED: disableSemanticSegmentation()" << std::endl;
            error_count++;
        }
    }

    // ---- Test Enable/Disable Instance Segmentation ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        test_instance.enableInstanceSegmentation();
        if (!test_instance.instancesegmentation_enabled) {
            std::cerr << "FAILED: enableInstanceSegmentation()" << std::endl;
            error_count++;
        }

        test_instance.disableInstanceSegmentation();
        if (test_instance.instancesegmentation_enabled) {
            std::cerr << "FAILED: disableInstanceSegmentation()" << std::endl;
            error_count++;
        }
    }

    // ---- Test labelPrimitives ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        uint sampleUUID = 12345;
        test_instance.labelPrimitives(sampleUUID, "test_label");

        if (test_instance.labelUUIDs.find("test_label") == test_instance.labelUUIDs.end()) {
            std::cerr << "FAILED: labelPrimitives()" << std::endl;
            error_count++;
        }
    }

    // ---- Test labelPrimitives (const char* label) ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        test_instance.labelPrimitives("test_label");

        if (test_instance.labelUUIDs.find("test_label") == test_instance.labelUUIDs.end()) {
            std::cerr << "FAILED: labelPrimitives(const char*)" << std::endl;
            error_count++;
        }
    }

    // ---- Test labelPrimitives (vector<vector<uint>> UUIDs, const char* label) ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        std::vector<std::vector<uint>> uuid_groups = {{123, 456}, {789, 1011}};
        test_instance.labelPrimitives(uuid_groups, "group_test_label");

        if (test_instance.labelUUIDs.find("group_test_label") == test_instance.labelUUIDs.end()) {
            std::cerr << "FAILED: labelPrimitives(vector<vector<uint>>, const char*)" << std::endl;
            error_count++;
        } else if (test_instance.labelUUIDs["group_test_label"].size() != uuid_groups.size()) {
            std::cerr << "FAILED: labelPrimitives - incorrect group size" << std::endl;
            error_count++;
        }
    }

    // ---- Test addSkyDome ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        try {
            test_instance.addSkyDome("plugins/visualizer/textures/SkyDome_clouds.jpg");
        } catch (...) {
            std::cerr << "FAILED: addSkyDome()" << std::endl;
            error_count++;
        }
    }

    // ---- Test getGroupRectangularBBox ---- //
    {
        Context context_test;
        SyntheticAnnotation test_instance(&context_test);

        helios::int4 bbox;
        std::vector<uint> pixels(1920 * 1080 * 3, 255); // Simulate white screen
        uint pixelCount = test_instance.getGroupRectangularBBox(1, pixels, 1920, 1080, bbox);

        if (pixelCount != 0) {
            std::cerr << "FAILED: getGroupRectangularBBox()" << std::endl;
            error_count++;
        }
    }

    // ---- Expanded RGB Encoding Tests ---- //
    {
        SyntheticAnnotation test_instance(nullptr);

        int edgeCases[] = {0, 1, 255, 256, 10000, 16777215}; // Max RGB range
        for (int id : edgeCases) {
            int convertedBack = test_instance.rgb2int(test_instance.int2rgb(id));
            if (id != convertedBack) {
                std::cerr << "FAILED: RGB encoding for value " << id << std::endl;
                error_count++;
            }
        }
    }



    // ---- Final Summary ---- //
    if (error_count == 0) {
        std::cout << "All tests passed." << std::endl;
    } else {
        std::cout << "Failed self-test with " << error_count << " errors." << std::endl;
    }
    return error_count;
}
