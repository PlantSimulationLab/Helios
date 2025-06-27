/** \file "selfTest.cpp" Self-test routines for Plant Architecture plug-in.

Copyright (C) 2016-2025 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "PlantArchitecture.h"

using namespace std;
using namespace helios;

int PlantArchitecture::selfTest() {

    int error_count = 0;
    double errtol = 1e-7;

    // Determine platform-specific null stream
#ifdef _WIN32
    const char *null_device = "NUL";
#else
    const char *null_device = "/dev/null";
#endif

    // Test 1: PlantArchitecture Constructor
    {
        try {
            Context context;
            PlantArchitecture pa_test(&context);
            std::cout << "Test 1 (PlantArchitecture Constructor): Passed." << std::endl;
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 1 (PlantArchitecture Constructor): Failed - " << e.what() << std::endl;
        }
    }

    // Test 2: interpolateTube (float)
    {
        try {
            std::vector<float> P = {0.0f, 1.0f, 2.0f, 3.0f};
            float frac = 0.5f;
            float expected_value = 1.5f;
            float actual_value = PlantArchitecture::interpolateTube(P, frac);
            if (std::abs(actual_value - expected_value) > errtol) {
                error_count++;
                std::cerr << "Test 2 (interpolateTube float): Failed - Expected " << expected_value << ", got " << actual_value << std::endl;
            } else {
                std::cout << "Test 2 (interpolateTube float): Passed." << std::endl;
            }
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 2 (interpolateTube float): Failed - " << e.what() << std::endl;
        }
    }

    // Test 3: interpolateTube (vec3)
    {
        try {
            std::vector<vec3> P = {make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_vec3(2, 2, 2)};
            float frac = 0.5f;
            vec3 expected_value = make_vec3(1, 1, 1); // Midpoint of the first segment
            vec3 actual_value = PlantArchitecture::interpolateTube(P, frac);
            if (std::abs(actual_value.x - expected_value.x) > errtol || std::abs(actual_value.y - expected_value.y) > errtol || std::abs(actual_value.z - expected_value.z) > errtol) {
                error_count++;
                std::cerr << "Test 3 (interpolateTube vec3): Failed - Expected (" << expected_value.x << "," << expected_value.y << "," << expected_value.z << "), got (" << actual_value.x << "," << actual_value.y << "," << actual_value.z << ")"
                          << std::endl;
            } else {
                std::cout << "Test 3 (interpolateTube vec3): Passed." << std::endl;
            }
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 3 (interpolateTube vec3): Failed - " << e.what() << std::endl;
        }
    }

    // Test 4: ShootParameters::defineChildShootTypes (valid input)
    {
        try {
            ShootParameters sp_test;
            std::vector<std::string> labels = {"typeA", "typeB"};
            std::vector<float> probabilities = {0.4f, 0.6f};
            sp_test.defineChildShootTypes(labels, probabilities);
            std::cout << "Test 4 (defineChildShootTypes valid): Passed." << std::endl;
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 4 (defineChildShootTypes valid): Failed - " << e.what() << std::endl;
        }
    }

    // Test 5: ShootParameters::defineChildShootTypes (invalid input - size mismatch)
    {
        // Suppress warnings temporarily
        std::streambuf *old_cerr = std::cerr.rdbuf();
        std::ofstream null_stream(null_device);
        std::cerr.rdbuf(null_stream.rdbuf());

        try {
            ShootParameters sp_test;
            std::vector<std::string> labels = {"typeA", "typeB"};
            std::vector<float> probabilities = {0.4f};
            sp_test.defineChildShootTypes(labels, probabilities);
            error_count++; // Should throw an exception, so if it reaches here, it's a failure
            std::cerr << "Test 5 (defineChildShootTypes size mismatch): Failed - No exception thrown for size mismatch." << std::endl;
        } catch (const std::exception &e) {
            std::cout << "Test 5 (defineChildShootTypes size mismatch): Passed (exception correctly caught)." << std::endl;
        }

        // Restore std::cerr
        std::cerr.rdbuf(old_cerr);
    }

    // Test 6: ShootParameters::defineChildShootTypes (invalid input - empty vectors)
    {
        // Suppress warnings temporarily
        std::streambuf *old_cerr = std::cerr.rdbuf();
        std::ofstream null_stream(null_device);
        std::cerr.rdbuf(null_stream.rdbuf());

        try {
            ShootParameters sp_test;
            std::vector<std::string> labels = {};
            std::vector<float> probabilities = {};
            sp_test.defineChildShootTypes(labels, probabilities);
            error_count++; // Should throw an exception
            std::cerr << "Test 6 (defineChildShootTypes empty vectors): Failed - No exception thrown for empty vectors." << std::endl;
        } catch (const std::exception &e) {
            std::cout << "Test 6 (defineChildShootTypes empty vectors): Passed (exception correctly caught)." << std::endl;
        }

        // Restore std::cerr
        std::cerr.rdbuf(old_cerr);
    }

    // Test 7: ShootParameters::defineChildShootTypes (invalid input - probabilities sum != 1)
    {
        // Suppress warnings temporarily
        std::streambuf *old_cerr = std::cerr.rdbuf();
        std::ofstream null_stream(null_device);
        std::cerr.rdbuf(null_stream.rdbuf());

        try {
            ShootParameters sp_test;
            std::vector<std::string> labels = {"typeA", "typeB"};
            std::vector<float> probabilities = {0.3f, 0.6f}; // Sums to 0.9
            sp_test.defineChildShootTypes(labels, probabilities);
            error_count++; // Should throw an exception
            std::cerr << "Test 7 (defineChildShootTypes sum != 1): Failed - No exception thrown for probabilities not summing to 1." << std::endl;
        } catch (const std::exception &e) {
            std::cout << "Test 7 (defineChildShootTypes sum != 1): Passed (exception correctly caught)." << std::endl;
        }

        // Restore std::cerr
        std::cerr.rdbuf(old_cerr);
    }

    // Test 8: PlantArchitecture::defineShootType
    {
        try {
            Context context;
            PlantArchitecture pa_test(&context);
            ShootParameters sp_define;
            pa_test.defineShootType("newShootType", sp_define);
            // Verify by trying to retrieve it (no direct getter, so rely on no exception)
            std::cout << "Test 8 (defineShootType): Passed." << std::endl;
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 8 (defineShootType): Failed - " << e.what() << std::endl;
        }
    }

    // Test 9: LeafPrototype Constructor
    {
        try {
            Context context;
            std::minstd_rand0 *generator = context.getRandomGenerator();
            LeafPrototype lp_test(generator);
            if (lp_test.subdivisions != 1 || lp_test.unique_prototypes != 1 || lp_test.leaf_offset.x != 0.0f || lp_test.leaf_offset.y != 0.0f || lp_test.leaf_offset.z != 0.0f) {
                error_count++;
                std::cerr << "Test 9 (LeafPrototype Constructor): Failed - Default values not as expected." << std::endl;
            } else {
                std::cout << "Test 9 (LeafPrototype Constructor): Passed." << std::endl;
            }
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 9 (LeafPrototype Constructor): Failed - " << e.what() << std::endl;
        }
    }

    // Test 10: PhytomerParameters Constructor
    {
        try {
            Context context;
            std::minstd_rand0 *generator = context.getRandomGenerator();
            PhytomerParameters pp_test(generator);
            std::cout << "Test 10 (PhytomerParameters Constructor): Passed." << std::endl;
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 10 (PhytomerParameters Constructor): Failed - " << e.what() << std::endl;
        }
    }

    std::vector<std::string> plant_labels{"almond", "apple", "asparagus", "bindweed", "bean", "cheeseweed", "cowpea", "grapevine_VSP", "maize", "olive", "pistachio", "puncturevine", "easternredbud", "rice", "butterlettuce", "sorghum", "soybean", "strawberry", "sugarbeet", "tomato", "walnut", "wheat"};

    auto test_buildPlantFromLibrary = [&error_count]( std::string plant_label) {
        try {
            std::cout << "Building " << plant_label << " plant model..." << std::flush;
            Context context;
            PlantArchitecture plantarchitecture(&context);
            plantarchitecture.loadPlantModelFromLibrary( plant_label );
            plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0,0,0), 5000);
            std::cout << "done." << std::endl;
        }catch (std::exception &e) {
            std::cerr << plant_label << " model failed." << std::endl;
            std::cerr << e.what() << std::endl;
            error_count++;
        }
    };

    for ( auto &plant_label : plant_labels ) {
        test_buildPlantFromLibrary( plant_label );
    }

    if (error_count == 0) {
        std::cout << "All PlantArchitecture self-tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "PlantArchitecture self-tests failed with " << error_count << " errors." << std::endl;
        return error_count;
    }
}
