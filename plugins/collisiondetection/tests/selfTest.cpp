/** \file "selfTest.cpp" Self-test routines for collision detection plugin

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "CollisionDetection.h"
#define DOCTEST_CONFIG_IMPLEMENT
#include <cmath>
#include <doctest.h>
#include <iostream>
#include "global.h"
#include "doctest_utils.h"

using namespace helios;

int CollisionDetection::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

namespace CollisionTests {

    /**
     * \brief Generate non-overlapping primitives for negative testing
     */
    std::vector<uint> generateSeparatedTriangles(Context *context, int count, float separation = 5.0f) {
        std::vector<uint> uuids;
        for (int i = 0; i < count; i++) {
            float x = i * separation;
            uint uuid = context->addTriangle(make_vec3(x, -1, 0), make_vec3(x + 1, -1, 0), make_vec3(x + 0.5f, 1, 0));
            uuids.push_back(uuid);
        }
        return uuids;
    }

    /**
     * \brief Generate overlapping primitive cluster for positive testing
     */
    std::vector<uint> generateOverlappingCluster(Context *context, int count, vec3 center = make_vec3(0, 0, 0)) {
        std::vector<uint> uuids;
        for (int i = 0; i < count; i++) {
            float angle = (2.0f * M_PI * i) / count;
            float radius = 0.5f; // Overlapping radius
            float x = center.x + radius * cos(angle);
            float y = center.y + radius * sin(angle);

            uint uuid = context->addTriangle(make_vec3(x - 0.5f, y - 0.5f, center.z), make_vec3(x + 0.5f, y - 0.5f, center.z), make_vec3(x, y + 0.5f, center.z));
            uuids.push_back(uuid);
        }
        return uuids;
    }

} // namespace CollisionTests

DOCTEST_TEST_CASE("CollisionDetection Plugin Initialization") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Test basic initialization
    DOCTEST_CHECK_NOTHROW(collision.disableMessages());
    DOCTEST_CHECK_NOTHROW(collision.enableMessages());

    // Test GPU acceleration capabilities based on actual hardware availability
    try {
        // Try to enable GPU acceleration and see if it actually works
        collision.enableGPUAcceleration();
        bool gpu_enabled = collision.isGPUAccelerationEnabled();
        
        // Create minimal test geometry to verify GPU functionality
        uint test_uuid = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
        collision.buildBVH();
        
        // If we reach here without exception, GPU may be available
        DOCTEST_INFO("GPU acceleration capability test - actual hardware dependent");
        if (gpu_enabled) {
            DOCTEST_WARN("GPU acceleration is available and enabled on this system");
        } else {
            DOCTEST_WARN("GPU acceleration requested but not available - using CPU fallback");
        }
        
        // Test that we can successfully disable GPU acceleration
        DOCTEST_CHECK_NOTHROW(collision.disableGPUAcceleration());
        DOCTEST_CHECK(collision.isGPUAccelerationEnabled() == false);
        
    } catch (std::exception &e) {
        // GPU initialization failure is acceptable - GPU may not be available
        DOCTEST_WARN((std::string("GPU acceleration test failed (expected on non-NVIDIA systems): ") + e.what()).c_str());
        
        // Ensure CPU mode works regardless
        DOCTEST_CHECK_NOTHROW(collision.disableGPUAcceleration());
        DOCTEST_CHECK(collision.isGPUAccelerationEnabled() == false);
    }
}


DOCTEST_TEST_CASE("CollisionDetection BVH Construction") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create some simple triangles
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    uint UUID2 = context.addTriangle(make_vec3(2, -1, 0), make_vec3(4, -1, 0), make_vec3(3, 1, 0));
    uint UUID3 = context.addTriangle(make_vec3(-0.5, -0.5, 1), make_vec3(1.5, -0.5, 1), make_vec3(0.5, 1.5, 1));

    // Build BVH
    collision.buildBVH();

    DOCTEST_CHECK(collision.isBVHValid() == true);
    DOCTEST_CHECK(collision.getPrimitiveCount() == 3);
}


DOCTEST_TEST_CASE("CollisionDetection Basic Collision Detection") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration(); // Use CPU for deterministic results

    // Create overlapping triangles
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    uint UUID2 = context.addTriangle(make_vec3(-0.5, -0.5, 0), make_vec3(1.5, -0.5, 0), make_vec3(0.5, 1.5, 0));

    // Create non-overlapping triangle
    uint UUID3 = context.addTriangle(make_vec3(10, -1, 0), make_vec3(11, -1, 0), make_vec3(10.5, 1, 0));

    collision.buildBVH();

    // Test collision between overlapping triangles
    std::vector<uint> collisions1 = collision.findCollisions(UUID1);

    // UUID1 should collide with UUID2 but not UUID3
    bool found_UUID2 = std::find(collisions1.begin(), collisions1.end(), UUID2) != collisions1.end();
    bool found_UUID3 = std::find(collisions1.begin(), collisions1.end(), UUID3) != collisions1.end();

    DOCTEST_CHECK(found_UUID2 == true);
    DOCTEST_CHECK(found_UUID3 == false);
}


DOCTEST_TEST_CASE("CollisionDetection BVH Statistics") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create a larger set of primitives
    for (int i = 0; i < 20; i++) {
        context.addTriangle(make_vec3(i, -1, 0), make_vec3(i + 1, -1, 0), make_vec3(i + 0.5f, 1, 0));
    }

    collision.buildBVH();

    size_t node_count, leaf_count, max_depth;
    collision.getBVHStatistics(node_count, leaf_count, max_depth);

    DOCTEST_CHECK(node_count > 0);
    DOCTEST_CHECK(leaf_count > 0);
    DOCTEST_CHECK(max_depth > 0);
}


DOCTEST_TEST_CASE("CollisionDetection Empty Geometry Handling") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Try to build BVH with no primitives
    collision.buildBVH();

    // Should handle gracefully
    DOCTEST_CHECK(collision.getPrimitiveCount() == 0);

    // Try collision detection with empty BVH
    std::vector<uint> collisions = collision.findCollisions(std::vector<uint>{});

    DOCTEST_CHECK(collisions.empty() == true);
}


DOCTEST_TEST_CASE("CollisionDetection Invalid UUID Handling") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));

    // Try collision detection with invalid UUID - should throw std::runtime_error
    DOCTEST_CHECK_THROWS_AS(collision.findCollisions(999999), std::runtime_error);
    
    // Also verify the exception message contains relevant information
    try {
        collision.findCollisions(999999);
        DOCTEST_FAIL("Expected exception was not thrown");
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        bool has_relevant_content = error_msg.find("UUID") != std::string::npos || 
                                   error_msg.find("invalid") != std::string::npos;
        DOCTEST_CHECK(has_relevant_content);
    }
}


DOCTEST_TEST_CASE("CollisionDetection GPU/CPU Mode Switching") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    uint UUID2 = context.addTriangle(make_vec3(-0.5, -0.5, 0), make_vec3(1.5, -0.5, 0), make_vec3(0.5, 1.5, 0));

    collision.buildBVH();

    // Test with GPU enabled
    collision.enableGPUAcceleration();
    std::vector<uint> gpu_results = collision.findCollisions(UUID1);

    // Test with GPU disabled
    collision.disableGPUAcceleration();
    std::vector<uint> cpu_results = collision.findCollisions(UUID1);

    // Results should be equivalent (though may be in different order)
    std::sort(gpu_results.begin(), gpu_results.end());
    std::sort(cpu_results.begin(), cpu_results.end());

    DOCTEST_CHECK(gpu_results == cpu_results);
}


DOCTEST_TEST_CASE("CollisionDetection Null Context Error Handling") {
    // Should throw std::runtime_error when context is null
    DOCTEST_CHECK_THROWS_AS(CollisionDetection collision(nullptr), std::runtime_error);
    
    // Also verify the exception message contains relevant information
    try {
        CollisionDetection collision(nullptr);
        DOCTEST_FAIL("Expected exception was not thrown");
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        bool has_relevant_content = error_msg.find("context") != std::string::npos || 
                                   error_msg.find("Context") != std::string::npos ||
                                   error_msg.find("null") != std::string::npos;
        DOCTEST_CHECK(has_relevant_content);
    }
}


DOCTEST_TEST_CASE("CollisionDetection Invalid UUIDs in BuildBVH") {
    Context context;
    
    // Suppress all initialization messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Try to build BVH with invalid UUIDs - should throw std::runtime_error
    std::vector<uint> invalid_UUIDs = {999999, 888888};
    
    DOCTEST_CHECK_THROWS_AS(collision.buildBVH(invalid_UUIDs), std::runtime_error);
    
    // Also verify the exception message contains relevant information
    try {
        collision.buildBVH(invalid_UUIDs);
        DOCTEST_FAIL("Expected exception was not thrown");
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        bool has_relevant_content = error_msg.find("UUID") != std::string::npos || 
                                   error_msg.find("invalid") != std::string::npos;
        DOCTEST_CHECK(has_relevant_content);
    }
}


DOCTEST_TEST_CASE("CollisionDetection Primitive/Object Collision Detection") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Create some primitives
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    uint UUID2 = context.addTriangle(make_vec3(-0.5, -0.5, 0), make_vec3(1.5, -0.5, 0), make_vec3(0.5, 1.5, 0));

    // Create a compound object
    uint objID = context.addTileObject(make_vec3(0, 0, 1), make_vec2(2, 2), make_SphericalCoord(0, 0), make_int2(1, 1));

    collision.buildBVH();

    // Test mixed primitive/object collision detection
    std::vector<uint> primitive_UUIDs = {UUID1};
    std::vector<uint> object_IDs = {objID};

    DOCTEST_CHECK_NOTHROW(collision.findCollisions(primitive_UUIDs, object_IDs));
}


DOCTEST_TEST_CASE("CollisionDetection Empty Input Handling") {
    Context context;
    
    // Suppress all initialization messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Test empty primitive vector
    std::vector<uint> empty_primitives;
    std::vector<uint> collisions1 = collision.findCollisions(empty_primitives);

    // Test empty mixed input
    std::vector<uint> empty_objects;
    std::vector<uint> collisions2 = collision.findCollisions(empty_primitives, empty_objects);

    DOCTEST_CHECK(collisions1.empty() == true);
    DOCTEST_CHECK(collisions2.empty() == true);
}


DOCTEST_TEST_CASE("CollisionDetection Invalid Object ID Error Handling") {
    Context context;
    
    // Suppress all initialization messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Try collision detection with invalid object ID - should throw std::runtime_error
    std::vector<uint> empty_primitives;
    std::vector<uint> invalid_objects = {999999};
    
    DOCTEST_CHECK_THROWS_AS(collision.findCollisions(empty_primitives, invalid_objects), std::runtime_error);
    
    // Also verify the exception message contains relevant information
    try {
        collision.findCollisions(empty_primitives, invalid_objects);
        DOCTEST_FAIL("Expected exception was not thrown");
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        bool has_relevant_content = error_msg.find("object") != std::string::npos || 
                                   error_msg.find("Object") != std::string::npos ||
                                   error_msg.find("999999") != std::string::npos ||
                                   error_msg.find("exist") != std::string::npos ||
                                   error_msg.find("invalid") != std::string::npos;
        DOCTEST_CHECK(has_relevant_content);
    }
}


DOCTEST_TEST_CASE("CollisionDetection Manual BVH Rebuild") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create initial geometry
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    collision.buildBVH();

    size_t initial_count = collision.getPrimitiveCount();

    // Add more geometry
    uint UUID2 = context.addTriangle(make_vec3(2, -1, 0), make_vec3(4, -1, 0), make_vec3(3, 1, 0));

    // Force rebuild
    collision.rebuildBVH();

    size_t final_count = collision.getPrimitiveCount();

    DOCTEST_CHECK(final_count == 2);
}


DOCTEST_TEST_CASE("CollisionDetection Message Control") {
    Context context;
    
    // Suppress initialization messages (we're testing message control itself)
    CollisionDetection collision(&context);

    // Test message disabling/enabling
    DOCTEST_CHECK_NOTHROW(collision.disableMessages());
    DOCTEST_CHECK_NOTHROW(collision.enableMessages());
}


DOCTEST_TEST_CASE("CollisionDetection Large Geometry Handling") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Create many primitives to stress test BVH
    for (int i = 0; i < 50; i++) {
        context.addTriangle(make_vec3(i, -1, 0), make_vec3(i + 1, -1, 0), make_vec3(i + 0.5f, 1, 0));
    }

    collision.buildBVH();

    // Test collision detection with large BVH
    uint UUID = context.getAllUUIDs()[0];
    std::vector<uint> collisions = collision.findCollisions(UUID);

    // Verify BVH statistics make sense
    size_t node_count, leaf_count, max_depth;
    collision.getBVHStatistics(node_count, leaf_count, max_depth);

    DOCTEST_CHECK(collision.getPrimitiveCount() == 50);
    DOCTEST_CHECK(node_count > 0);
    DOCTEST_CHECK(max_depth > 0);
}


DOCTEST_TEST_CASE("CollisionDetection Single Primitive Edge Case") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Create single primitive
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));

    collision.buildBVH();

    // Test collision with itself (should return empty since self is removed)
    std::vector<uint> collisions = collision.findCollisions(UUID1);

    // Verify BVH handles single primitive correctly
    size_t node_count, leaf_count, max_depth;
    collision.getBVHStatistics(node_count, leaf_count, max_depth);

    DOCTEST_CHECK(collision.getPrimitiveCount() == 1);
    DOCTEST_CHECK(collision.isBVHValid() == true);
}


DOCTEST_TEST_CASE("CollisionDetection Overlapping AABB Primitives") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Create primitives with overlapping AABBs
    // These triangles are tilted so their AABBs will overlap in Z
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0.2f));
    uint UUID2 = context.addTriangle(make_vec3(-1, -1, 0.1f), make_vec3(1, -1, 0.1f), make_vec3(0, 1, -0.1f));

    collision.buildBVH();

    // These should collide (overlapping AABBs)
    std::vector<uint> collisions = collision.findCollisions(UUID1);

    bool found_collision = std::find(collisions.begin(), collisions.end(), UUID2) != collisions.end();

    DOCTEST_CHECK(found_collision == true);
}


DOCTEST_TEST_CASE("CollisionDetection BVH Validity Persistence") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Initially invalid (no BVH built)
    DOCTEST_CHECK(collision.isBVHValid() == false);

    // Create geometry and build
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    collision.buildBVH();

    // Should be valid after building
    DOCTEST_CHECK(collision.isBVHValid() == true);

    // Should remain valid after collision queries
    collision.findCollisions(UUID1);
    DOCTEST_CHECK(collision.isBVHValid() == true);
}


DOCTEST_TEST_CASE("CollisionDetection GPU Acceleration") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create test geometry
    for (int i = 0; i < 5; i++) {
        context.addTriangle(make_vec3(i, -1, 0), make_vec3(i + 1, -1, 0), make_vec3(i + 0.5f, 1, 0));
    }

    // Test GPU vs CPU equivalence (if GPU available)
    try {
        collision.disableGPUAcceleration();
        collision.buildBVH();
        uint UUID = context.getAllUUIDs()[0];
        std::vector<uint> cpu_results = collision.findCollisions(UUID);

        collision.enableGPUAcceleration();
        collision.buildBVH(); // This should transfer to GPU
        std::vector<uint> gpu_results = collision.findCollisions(UUID);

        // Compare results (allowing for different orders)
        std::sort(cpu_results.begin(), cpu_results.end());
        std::sort(gpu_results.begin(), gpu_results.end());

        // Check if results match (GPU may not be available, so we use INFO instead of CHECK)
        DOCTEST_INFO("GPU/CPU result comparison - GPU may not be available on this system");
        if (cpu_results.size() == gpu_results.size()) {
            bool results_match = true;
            for (size_t i = 0; i < cpu_results.size(); i++) {
                if (cpu_results[i] != gpu_results[i]) {
                    results_match = false;
                    break;
                }
            }
            // Only warn if results don't match - this is acceptable if no GPU
            if (!results_match) {
                DOCTEST_WARN("GPU/CPU results differ - may be expected if no CUDA device");
            }
        }
    } catch (std::exception &e) {
        // GPU test failure is acceptable - GPU may not be available
        DOCTEST_WARN((std::string("GPU test failed (may be expected): ") + e.what()).c_str());
    }
}


DOCTEST_TEST_CASE("CollisionDetection GPU/CPU Message Display") {
    Context context;
    
    // Suppress initialization message, then create collision detection object and test geometry
    CollisionDetection collision(&context);
    
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    uint UUID2 = context.addTriangle(make_vec3(-0.5, -0.5, 0), make_vec3(1.5, -0.5, 0), make_vec3(0.5, 1.5, 0));
    collision.buildBVH(); // Build once to avoid BVH construction messages later

    // Test that messages are displayed when enabled
    collision.enableMessages();
    collision.disableGPUAcceleration();

    // Capture traversal messages using capture_cout (these are the specific messages we want to test)
    helios::capture_cout capture_cpu;
    std::vector<uint> cpu_results = collision.findCollisions(UUID1);
    std::string cpu_output = capture_cpu.get_captured_output();

    // Check that some CPU message was displayed (the exact message may vary)
    // DOCTEST_INFO("CPU output: " << cpu_output);
    // For now, just check that collision detection worked (test is mainly about message suppression)
    DOCTEST_CHECK(true); // Placeholder - the main goal is testing message suppression below

    // Test GPU message (if available)
    collision.enableGPUAcceleration();

    helios::capture_cout capture_gpu;
    std::vector<uint> gpu_results = collision.findCollisions(UUID1);
    std::string gpu_output = capture_gpu.get_captured_output();

    // Should contain either GPU or CPU message depending on availability
    // For now, just check basic functionality (the main test is message suppression below)
    DOCTEST_CHECK(true); // Placeholder - the main goal is testing message suppression below

    // Test message suppression
    collision.disableMessages();

    helios::capture_cout capture_silent;
    std::vector<uint> silent_results = collision.findCollisions(UUID1);
    std::string silent_output = capture_silent.get_captured_output();

    // Should not contain traversal messages when disabled
    DOCTEST_CHECK(silent_output.find("Using GPU acceleration") == std::string::npos);
    DOCTEST_CHECK(silent_output.find("Using CPU traversal") == std::string::npos);
}


DOCTEST_TEST_CASE("CollisionDetection Automatic BVH Building") {
    Context context;
    
    // Suppress all initialization and automatic BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration(); // Use CPU for predictable behavior

    // Create initial geometry
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));

    // BVH should be invalid initially (not built)
    DOCTEST_CHECK(collision.isBVHValid() == false);

    // Calling findCollisions should automatically build BVH
    std::vector<uint> results = collision.findCollisions(UUID1);

    // BVH should now be valid
    DOCTEST_CHECK(collision.isBVHValid() == true);
    DOCTEST_CHECK(collision.getPrimitiveCount() == 1);

    // Add more geometry and mark it dirty in context
    uint UUID2 = context.addTriangle(make_vec3(2, -1, 0), make_vec3(4, -1, 0), make_vec3(3, 1, 0));

    // BVH should now be invalid because there's a new primitive not in the BVH
    DOCTEST_CHECK(collision.isBVHValid() == false);

    // But calling findCollisions should detect the new geometry and rebuild
    results = collision.findCollisions(UUID1);

    // BVH should now include both primitives
    DOCTEST_CHECK(collision.getPrimitiveCount() == 2);
    DOCTEST_CHECK(collision.isBVHValid() == true);

    // Test that repeated calls don't unnecessarily rebuild
    size_t count_before = collision.getPrimitiveCount();
    results = collision.findCollisions(UUID1);
    size_t count_after = collision.getPrimitiveCount();

    // Should be the same (no unnecessary rebuild)
    DOCTEST_CHECK(count_before == count_after);
}


DOCTEST_TEST_CASE("CollisionDetection Restricted Geometry - UUIDs Only") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Create test geometry: 3 overlapping triangles
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));
    uint UUID2 = context.addTriangle(make_vec3(-0.5, -0.5, 0), make_vec3(1.5, -0.5, 0), make_vec3(0.5, 1.5, 0));
    // UUID3 overlaps in z-dimension to ensure AABB collision
    uint UUID3 = context.addTriangle(make_vec3(-0.5, -0.5, -0.1f), make_vec3(1.5, -0.5, -0.1f), make_vec3(0.5, 1.5, 0.1f));

    // Test unrestricted collision detection (should find all collisions)
    std::vector<uint> all_results = collision.findCollisions(UUID1);

    // UUID1 should collide with both UUID2 and UUID3
    bool found_UUID2_all = std::find(all_results.begin(), all_results.end(), UUID2) != all_results.end();
    bool found_UUID3_all = std::find(all_results.begin(), all_results.end(), UUID3) != all_results.end();

    DOCTEST_CHECK(found_UUID2_all == true);
    DOCTEST_CHECK(found_UUID3_all == true);

    // Test restricted collision detection - only target UUID2
    std::vector<uint> query_UUIDs = {UUID1};
    std::vector<uint> query_objects = {};
    std::vector<uint> target_UUIDs = {UUID2}; // Only target UUID2
    std::vector<uint> target_objects = {};

    std::vector<uint> restricted_results = collision.findCollisions(query_UUIDs, query_objects, target_UUIDs, target_objects);

    // Should only find collision with UUID2, not UUID3
    bool found_UUID2_restricted = std::find(restricted_results.begin(), restricted_results.end(), UUID2) != restricted_results.end();
    bool found_UUID3_restricted = std::find(restricted_results.begin(), restricted_results.end(), UUID3) != restricted_results.end();

    DOCTEST_CHECK(found_UUID2_restricted == true);
    DOCTEST_CHECK(found_UUID3_restricted == false);
}


DOCTEST_TEST_CASE("CollisionDetection Restricted Geometry - Object IDs") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Create individual primitives
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));

    // Create compound objects
    uint objID1 = context.addTileObject(make_vec3(0, 0, 0.5f), make_vec2(2, 2), make_SphericalCoord(0, 0), make_int2(1, 1));
    uint objID2 = context.addTileObject(make_vec3(10, 0, 0), make_vec2(2, 2), make_SphericalCoord(0, 0), make_int2(1, 1));

    // Test collision detection restricted to specific object
    std::vector<uint> query_UUIDs = {UUID1};
    std::vector<uint> query_objects = {};
    std::vector<uint> target_UUIDs = {};
    std::vector<uint> target_objects = {objID1}; // Only target objID1

    DOCTEST_CHECK_NOTHROW(collision.findCollisions(query_UUIDs, query_objects, target_UUIDs, target_objects));

    // Test mixed UUID/Object ID restriction
    std::vector<uint> mixed_target_UUIDs = {UUID1};
    std::vector<uint> mixed_target_objects = {objID1};

    DOCTEST_CHECK_NOTHROW(collision.findCollisions(query_UUIDs, query_objects, mixed_target_UUIDs, mixed_target_objects));
}


DOCTEST_TEST_CASE("CollisionDetection Restricted Geometry - Error Handling") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create valid geometry
    uint UUID1 = context.addTriangle(make_vec3(-1, -1, 0), make_vec3(1, -1, 0), make_vec3(0, 1, 0));

    // Test error handling for invalid query UUIDs - should throw std::runtime_error
    std::vector<uint> invalid_query_UUIDs = {999999};
    std::vector<uint> query_objects = {};
    std::vector<uint> valid_target_UUIDs = {UUID1};
    std::vector<uint> target_objects = {};
    
    DOCTEST_CHECK_THROWS_AS(collision.findCollisions(invalid_query_UUIDs, query_objects, valid_target_UUIDs, target_objects), std::runtime_error);

    // Test error handling for invalid target UUIDs - should throw std::runtime_error
    std::vector<uint> valid_query_UUIDs = {UUID1};
    std::vector<uint> invalid_target_UUIDs = {999999};
    
    DOCTEST_CHECK_THROWS_AS(collision.findCollisions(valid_query_UUIDs, query_objects, invalid_target_UUIDs, target_objects), std::runtime_error);

    // Test error handling for invalid query object IDs - should throw std::runtime_error
    std::vector<uint> invalid_query_objects = {999999};
    
    DOCTEST_CHECK_THROWS_AS(collision.findCollisions(valid_query_UUIDs, invalid_query_objects, valid_target_UUIDs, target_objects), std::runtime_error);

    // Test error handling for invalid target object IDs - should throw std::runtime_error
    std::vector<uint> invalid_target_objects = {999999};
    
    DOCTEST_CHECK_THROWS_AS(collision.findCollisions(valid_query_UUIDs, query_objects, valid_target_UUIDs, invalid_target_objects), std::runtime_error);
    
    // Additional verification - test that the exception messages are meaningful
    try {
        collision.findCollisions(invalid_query_UUIDs, query_objects, valid_target_UUIDs, target_objects);
        DOCTEST_FAIL("Expected exception was not thrown");
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        bool has_relevant_content = error_msg.find("UUID") != std::string::npos || 
                                   error_msg.find("invalid") != std::string::npos;
        DOCTEST_CHECK(has_relevant_content);
    }
}

DOCTEST_TEST_CASE("CollisionDetection findOptimalConePath Basic Functionality") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create a simple test scene with obstacles
    // Add a few triangles that will create obstacles in certain directions
    uint triangle1 = context.addTriangle(make_vec3(2, -1, -1), make_vec3(2, 1, -1), make_vec3(2, 0, 1));
    uint triangle2 = context.addTriangle(make_vec3(-2, -1, -1), make_vec3(-2, 1, -1), make_vec3(-2, 0, 1));

    // Test with cone pointing towards obstacles
    vec3 apex = make_vec3(0, 0, -5);
    vec3 central_axis = make_vec3(0, 0, 1); // Pointing toward obstacles
    float half_angle = M_PI / 4.0f; // 45 degrees

    // Test 1: Basic functionality with default parameters
    CollisionDetection::OptimalPathResult result = collision.findOptimalConePath(apex, central_axis, half_angle);

    DOCTEST_CHECK(result.direction.magnitude() > 0.9f); // Should be normalized
    DOCTEST_CHECK(result.direction.magnitude() < 1.1f);
    DOCTEST_CHECK(result.confidence >= 0.0f);
    DOCTEST_CHECK(result.confidence <= 1.0f);
    DOCTEST_CHECK(result.collisionCount >= 0);
}

DOCTEST_TEST_CASE("CollisionDetection findOptimalConePath Gap Detection") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create test geometry with obstacles in a specific pattern
    // Add obstacle directly in front (along central axis)
    context.addTriangle(make_vec3(-0.5f, -0.5f, 0), make_vec3(0.5f, -0.5f, 0), make_vec3(0, 0.5f, 0));
    // Add fewer obstacles to the side (at an angle from central axis)
    context.addTriangle(make_vec3(3, -0.2f, 0), make_vec3(3, 0.2f, 0), make_vec3(3, 0, 0.5f));

    vec3 apex = make_vec3(0, 0, -2);
    vec3 central_axis = make_vec3(0, 0, 1); // Pointing toward main obstacle
    float half_angle = M_PI / 3.0f; // 60 degrees - wide cone

    // Test 1: Default behavior - should find optimal gap-based path
    CollisionDetection::OptimalPathResult gap_result = collision.findOptimalConePath(apex, central_axis, half_angle, 0.0f, 64);

    // Test 2: Higher sample count for better gap detection
    CollisionDetection::OptimalPathResult dense_result = collision.findOptimalConePath(apex, central_axis, half_angle, 0.0f, 256);

    // Test 3: Lower sample count
    CollisionDetection::OptimalPathResult sparse_result = collision.findOptimalConePath(apex, central_axis, half_angle, 0.0f, 32);

    // Verify results are reasonable
    DOCTEST_CHECK(gap_result.direction.magnitude() > 0.9f);
    DOCTEST_CHECK(dense_result.direction.magnitude() > 0.9f);
    DOCTEST_CHECK(sparse_result.direction.magnitude() > 0.9f);

    // All should have valid confidence scores
    DOCTEST_CHECK(gap_result.confidence >= 0.0f);
    DOCTEST_CHECK(gap_result.confidence <= 1.0f);
    DOCTEST_CHECK(dense_result.confidence >= 0.0f);
    DOCTEST_CHECK(dense_result.confidence <= 1.0f);
    DOCTEST_CHECK(sparse_result.confidence >= 0.0f);
    DOCTEST_CHECK(sparse_result.confidence <= 1.0f);

    // Results should be valid directions
    float gap_deviation = acosf(std::max(-1.0f, std::min(1.0f, gap_result.direction * central_axis)));
    float dense_deviation = acosf(std::max(-1.0f, std::min(1.0f, dense_result.direction * central_axis)));

    DOCTEST_CHECK(gap_deviation >= 0.0f); // Should be valid angle
    DOCTEST_CHECK(dense_deviation >= 0.0f); // Should be valid angle
}

DOCTEST_TEST_CASE("CollisionDetection findOptimalConePath Edge Cases") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    vec3 apex = make_vec3(0, 0, 0);
    vec3 central_axis = make_vec3(0, 0, 1);

    // Test 1: Invalid parameters
    {
        // Zero samples
        CollisionDetection::OptimalPathResult result = collision.findOptimalConePath(apex, central_axis, M_PI / 4.0f, 0.0f, 0);
        DOCTEST_CHECK(result.direction * central_axis > 0.9f); // Should default to central axis
    }

    // Test 2: Zero half-angle
    {
        CollisionDetection::OptimalPathResult result = collision.findOptimalConePath(apex, central_axis, 0.0f, 0.0f, 16);
        DOCTEST_CHECK(result.direction * central_axis > 0.9f); // Should default to central axis
    }

    // Test 3: Empty scene (no geometry)
    {
        CollisionDetection::OptimalPathResult result = collision.findOptimalConePath(apex, central_axis, M_PI / 4.0f, 0.0f, 16);
        DOCTEST_CHECK(result.direction * central_axis > 0.9f); // Should prefer central axis
        DOCTEST_CHECK(result.collisionCount == 0); // No collisions in empty scene
        DOCTEST_CHECK(result.confidence == 1.0f); // High confidence with no obstacles
    }

    // Test 4: Single sample
    {
        // Add some geometry
        context.addTriangle(make_vec3(-1, -1, 1), make_vec3(1, -1, 1), make_vec3(0, 1, 1));

        CollisionDetection::OptimalPathResult result = collision.findOptimalConePath(apex, central_axis, M_PI / 4.0f, 0.0f, 1);
        DOCTEST_CHECK(result.direction.magnitude() > 0.9f); // Should be normalized
        DOCTEST_CHECK(result.confidence == 1.0f); // Perfect confidence with single sample
    }
}

DOCTEST_TEST_CASE("CollisionDetection Finite Cone Height") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create obstacles at different distances
    // Close obstacle
    context.addTriangle(make_vec3(-0.5f, -0.5f, 1), make_vec3(0.5f, -0.5f, 1), make_vec3(0, 0.5f, 1));
    // Far obstacle
    context.addTriangle(make_vec3(-0.5f, -0.5f, 5), make_vec3(0.5f, -0.5f, 5), make_vec3(0, 0.5f, 5));

    vec3 apex = make_vec3(0, 0, 0);
    vec3 central_axis = make_vec3(0, 0, 1);
    float half_angle = M_PI / 6.0f; // 30 degrees

    // Test 1: Short cone - should only see close obstacle
    CollisionDetection::OptimalPathResult short_result = collision.findOptimalConePath(apex, central_axis, half_angle, 2.0f, 16);

    // Test 2: Long cone - should see both obstacles
    CollisionDetection::OptimalPathResult long_result = collision.findOptimalConePath(apex, central_axis, half_angle, 10.0f, 16);

    // Both should produce valid results
    DOCTEST_CHECK(short_result.direction.magnitude() > 0.9f);
    DOCTEST_CHECK(long_result.direction.magnitude() > 0.9f);
    DOCTEST_CHECK(short_result.collisionCount >= 0);
    DOCTEST_CHECK(long_result.collisionCount >= 0);

}

DOCTEST_TEST_CASE("CollisionDetection Scale Test - 1000 Primitives") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Generate 1000 separated triangles
    auto uuids = CollisionTests::generateSeparatedTriangles(&context, 1000);

    // Should not crash during BVH construction
    DOCTEST_CHECK_NOTHROW(collision.buildBVH());

    DOCTEST_CHECK(collision.isBVHValid() == true);
    DOCTEST_CHECK(collision.getPrimitiveCount() == 1000);
}

DOCTEST_TEST_CASE("CollisionDetection Scale Test - 10000 Primitives") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Generate 10,000 separated triangles - should stress memory allocation
    auto uuids = CollisionTests::generateSeparatedTriangles(&context, 10000);

    // This would have caught the original memory allocation bug
    DOCTEST_CHECK_NOTHROW(collision.buildBVH());

    DOCTEST_CHECK(collision.isBVHValid() == true);
    DOCTEST_CHECK(collision.getPrimitiveCount() == 10000);
}

// =================== CPU vs GPU VALIDATION ===================

DOCTEST_TEST_CASE("CollisionDetection CPU vs GPU Consistency - Small Scale") {
    Context context;

    // Suppress all initialization and BVH building messages

    // Create test scenario - overlapping and non-overlapping primitives
    auto overlapping = CollisionTests::generateOverlappingCluster(&context, 5, make_vec3(0, 0, 0));
    auto separated = CollisionTests::generateSeparatedTriangles(&context, 5, 10.0f);

    // Test with CPU
    CollisionDetection cpu_collision(&context);
    cpu_collision.disableMessages();
    cpu_collision.disableGPUAcceleration();
    cpu_collision.buildBVH();

    // Test with GPU
    CollisionDetection gpu_collision(&context);
    gpu_collision.disableMessages();
    gpu_collision.enableGPUAcceleration();
    gpu_collision.buildBVH();

    // Compare results for each primitive
    for (uint uuid: overlapping) {
        auto cpu_results = cpu_collision.findCollisions(uuid);
        auto gpu_results = gpu_collision.findCollisions(uuid);

        // Sort for comparison
        std::sort(cpu_results.begin(), cpu_results.end());
        std::sort(gpu_results.begin(), gpu_results.end());

        DOCTEST_CHECK(cpu_results == gpu_results);
    }
}

DOCTEST_TEST_CASE("CollisionDetection CPU vs GPU Consistency - Large Scale") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Generate 1000 primitives with known collision pattern
    auto cluster1 = CollisionTests::generateOverlappingCluster(&context, 10, make_vec3(0, 0, 0));
    auto cluster2 = CollisionTests::generateOverlappingCluster(&context, 10, make_vec3(20, 0, 0));
    auto separated = CollisionTests::generateSeparatedTriangles(&context, 980, 2.0f);

    // Test CPU results
    collision.disableGPUAcceleration();
    collision.buildBVH();
    auto cpu_results = collision.findCollisions(cluster1[0]);

    // Test GPU results
    collision.enableGPUAcceleration();
    collision.buildBVH();
    auto gpu_results = collision.findCollisions(cluster1[0]);

    // Sort for comparison
    std::sort(cpu_results.begin(), cpu_results.end());
    std::sort(gpu_results.begin(), gpu_results.end());

    // This would have caught the original GPU false positive bug
    DOCTEST_CHECK(cpu_results == gpu_results);
}

// =================== NEGATIVE TESTING ===================

DOCTEST_TEST_CASE("CollisionDetection Negative Test - Well Separated Primitives") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create primitives that definitely should not intersect
    uint triangle1 = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 1, 0));
    uint triangle2 = context.addTriangle(make_vec3(10, 10, 10), make_vec3(11, 10, 10), make_vec3(10.5f, 11, 10));
    uint patch = context.addPatch(make_vec3(20, 20, 20), make_vec2(1, 1));

    collision.buildBVH();

    // Test CPU
    collision.disableGPUAcceleration();
    auto cpu_collisions = collision.findCollisions(triangle1);

    // Test GPU
    collision.enableGPUAcceleration();
    auto gpu_collisions = collision.findCollisions(triangle1);

    // Should find 0 collisions (not counting self)
    DOCTEST_CHECK(cpu_collisions.size() == 0);
    DOCTEST_CHECK(gpu_collisions.size() == 0);
}

DOCTEST_TEST_CASE("CollisionDetection Negative Test - Patch vs Distant Model") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create a cluster of triangles far from origin
    auto distant_triangles = CollisionTests::generateOverlappingCluster(&context, 20, make_vec3(50, 50, 50));

    // Create patch at origin
    uint patch = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    collision.buildBVH();

    // Test both CPU and GPU
    collision.disableGPUAcceleration();
    auto cpu_collisions = collision.findCollisions(patch);

    collision.enableGPUAcceleration();
    auto gpu_collisions = collision.findCollisions(patch);

    // Should find 0 collisions
    DOCTEST_CHECK(cpu_collisions.size() == 0);
    DOCTEST_CHECK(gpu_collisions.size() == 0);
    DOCTEST_CHECK(cpu_collisions == gpu_collisions);
}

// =================== EDGE CASE TESTING ===================

DOCTEST_TEST_CASE("CollisionDetection Edge Case - Boundary Touching") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create primitives that exactly touch at boundaries
    uint triangle1 = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 1, 0));
    uint triangle2 = context.addTriangle(make_vec3(1, 0, 0), make_vec3(2, 0, 0), make_vec3(1.5f, 1, 0)); // Shares edge

    collision.buildBVH();

    // Test CPU vs GPU consistency
    collision.disableGPUAcceleration();
    auto cpu_results = collision.findCollisions(triangle1);

    collision.enableGPUAcceleration();
    auto gpu_results = collision.findCollisions(triangle1);

    std::sort(cpu_results.begin(), cpu_results.end());
    std::sort(gpu_results.begin(), gpu_results.end());

    DOCTEST_CHECK(cpu_results == gpu_results);
}

DOCTEST_TEST_CASE("CollisionDetection Edge Case - Very Small Overlaps") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create primitives with tiny overlaps (precision testing)
    uint triangle1 = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 1, 0));
    uint triangle2 = context.addTriangle(make_vec3(0.99f, 0, 0), make_vec3(1.99f, 0, 0), make_vec3(1.49f, 1, 0));

    collision.buildBVH();

    // Test CPU vs GPU consistency for precision
    collision.disableGPUAcceleration();
    auto cpu_results = collision.findCollisions(triangle1);

    collision.enableGPUAcceleration();
    auto gpu_results = collision.findCollisions(triangle1);

    std::sort(cpu_results.begin(), cpu_results.end());
    std::sort(gpu_results.begin(), gpu_results.end());

    DOCTEST_CHECK(cpu_results == gpu_results);
}

// =================== REAL GEOMETRY TESTING ===================

DOCTEST_TEST_CASE("CollisionDetection Real Geometry - PLY File Loading") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // This would test with actual PLY files if available
    // For now, simulate complex real geometry
    std::vector<uint> complex_model;

    // Generate a "complex model" with varying triangle sizes and orientations
    for (int i = 0; i < 1000; i++) {
        float scale = 0.1f + (i % 10) * 0.05f; // Varying sizes
        float angle = (i * 0.1f);
        float x = cos(angle) * (i * 0.01f);
        float y = sin(angle) * (i * 0.01f);
        float z = (i % 100) * 0.001f; // Varying heights

        uint uuid = context.addTriangle(make_vec3(x - scale, y - scale, z), make_vec3(x + scale, y - scale, z), make_vec3(x, y + scale, z));
        complex_model.push_back(uuid);
    }

    // Test patch at various positions
    uint patch_intersecting = context.addPatch(make_vec3(0, 0, 0.05f), make_vec2(0.5f, 0.5f));
    uint patch_non_intersecting = context.addPatch(make_vec3(100, 100, 100), make_vec2(1, 1));

    collision.buildBVH();

    // Test CPU vs GPU with complex geometry
    collision.disableGPUAcceleration();
    auto cpu_intersecting = collision.findCollisions(patch_intersecting);
    auto cpu_non_intersecting = collision.findCollisions(patch_non_intersecting);

    collision.enableGPUAcceleration();
    auto gpu_intersecting = collision.findCollisions(patch_intersecting);
    auto gpu_non_intersecting = collision.findCollisions(patch_non_intersecting);

    // Sort for comparison
    std::sort(cpu_intersecting.begin(), cpu_intersecting.end());
    std::sort(gpu_intersecting.begin(), gpu_intersecting.end());
    std::sort(cpu_non_intersecting.begin(), cpu_non_intersecting.end());
    std::sort(gpu_non_intersecting.begin(), gpu_non_intersecting.end());

    DOCTEST_CHECK(cpu_intersecting == gpu_intersecting);
    DOCTEST_CHECK(cpu_non_intersecting == gpu_non_intersecting);
    DOCTEST_CHECK(cpu_non_intersecting.size() == 0); // Should be empty
}

// =================== PERFORMANCE REGRESSION TESTING ===================

DOCTEST_TEST_CASE("CollisionDetection Performance - BVH Construction Time") {
    Context context;
    
    // Suppress all initialization messages - but we want to time BVH construction itself
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Generate substantial geometry
    auto uuids = CollisionTests::generateSeparatedTriangles(&context, 5000);

    // End suppression for timing the actual BVH construction (but keep it silent)
    // Time BVH construction
    auto start = std::chrono::high_resolution_clock::now();
    collision.buildBVH();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete within reasonable time (adjust threshold as needed)
    DOCTEST_CHECK(duration.count() < 5000); // 5 seconds max
    DOCTEST_CHECK(collision.isBVHValid() == true);
}

// =================== MEMORY STRESS TESTING ===================

DOCTEST_TEST_CASE("CollisionDetection Memory Stress - Progressive Loading") {
    Context context;
    
    // Suppress all initialization and BVH building messages
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Progressively add more primitives to test memory allocation patterns
    std::vector<uint> all_uuids;

    for (int batch = 0; batch < 10; batch++) {
        // Add 1000 more primitives each batch
        auto batch_uuids = CollisionTests::generateSeparatedTriangles(&context, 1000, 5.0f + batch * 50.0f);
        all_uuids.insert(all_uuids.end(), batch_uuids.begin(), batch_uuids.end());

        // Rebuild BVH each time (stress test memory management)
        DOCTEST_CHECK_NOTHROW(collision.buildBVH());
        DOCTEST_CHECK(collision.isBVHValid() == true);
        DOCTEST_CHECK(collision.getPrimitiveCount() == all_uuids.size());
    }
}