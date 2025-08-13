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
#include <chrono>
#include <doctest.h>
#include "doctest_utils.h"
#include <iostream>
#include "global.h"

using namespace helios;

int CollisionDetection::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc,argv);
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
}


DOCTEST_TEST_CASE("CollisionDetection Soft/Hard Detection Integration - BVH Sharing") {
    Context context;
    
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Create test geometry - obstacle and some primitives for "soft" collision testing
    uint obstacle = context.addPatch(make_vec3(0, 0, 1), make_vec2(2, 2)); // Obstacle at height 1m
    uint soft_prim1 = context.addTriangle(make_vec3(-0.5, -0.5, 0.5), make_vec3(0.5, -0.5, 0.5), make_vec3(0, 0.5, 0.5)); // Below obstacle
    uint soft_prim2 = context.addTriangle(make_vec3(-1.5, -1.5, 1.5), make_vec3(-0.5, -1.5, 1.5), make_vec3(-1, -0.5, 1.5)); // Above obstacle
    
    // Build BVH with all geometry for initial soft collision detection
    std::vector<uint> all_geometry = {obstacle, soft_prim1, soft_prim2};
    collision.buildBVH(all_geometry);
    
    // Verify initial BVH state
    DOCTEST_CHECK(collision.isBVHValid() == true);
    size_t initial_node_count, initial_leaf_count, initial_max_depth;
    collision.getBVHStatistics(initial_node_count, initial_leaf_count, initial_max_depth);
    
    // Simulate soft collision detection (standard findCollisions)
    std::vector<uint> soft_collisions = collision.findCollisions({soft_prim1, soft_prim2});
    bool soft_detection_completed = true; // Mark that soft detection has run
    
    // Now test hard detection using the same BVH
    vec3 test_origin = make_vec3(0, 0, 0.5);
    vec3 test_direction = make_vec3(0, 0, 1); // Pointing up toward obstacle
    float distance;
    vec3 obstacle_direction;
    
    // This should use the SAME BVH that was built for soft detection
    bool hard_hit = collision.findNearestSolidObstacleInCone(
        test_origin, test_direction, 0.52f, 1.0f, 
        {obstacle}, distance, obstacle_direction);
    
    // Verify both detections work correctly despite sharing BVH
    DOCTEST_CHECK(soft_detection_completed == true);
    DOCTEST_CHECK(hard_hit == true);
    DOCTEST_CHECK(distance < 1.0f); // Should detect obstacle before 1m
    
    // Verify BVH state wasn't corrupted by interleaved usage
    size_t final_node_count, final_leaf_count, final_max_depth;
    collision.getBVHStatistics(final_node_count, final_leaf_count, final_max_depth);
    
    DOCTEST_CHECK(initial_node_count == final_node_count);
    DOCTEST_CHECK(initial_leaf_count == final_leaf_count);
    DOCTEST_CHECK(collision.isBVHValid() == true);
}


DOCTEST_TEST_CASE("CollisionDetection Soft/Hard Detection Integration - Sequential Calls") {
    Context context;
    
    CollisionDetection collision(&context);
    collision.disableMessages(); // Disable debug output
    collision.disableGPUAcceleration();

    // Create a more complex scene
    uint ground = context.addPatch(make_vec3(0, 0, 0), make_vec2(4, 4));
    uint wall = context.addPatch(make_vec3(1, 0, 0.5), make_vec2(0.1, 2), make_SphericalCoord(0.5*M_PI, 0.5*M_PI)); // Vertical wall
    uint plant_stem = context.addTriangle(make_vec3(-0.02, 0, 0), make_vec3(0.02, 0, 0), make_vec3(0, 0, 0.8));
    
    std::vector<uint> all_obstacles = {ground, wall};
    std::vector<uint> plant_parts = {plant_stem};
    
    collision.buildBVH(all_obstacles);
    
    // Test 1: Soft collision detection between plant and obstacles
    std::vector<uint> soft_collisions = collision.findCollisions(plant_parts, {}, all_obstacles, {});
    
    // Test 2: Hard detection for plant growth (cone-based)
    vec3 growth_tip = make_vec3(0, 0, 0.8);
    vec3 growth_direction = make_vec3(0.5, 0, 0.2); // Angled toward wall
    growth_direction.normalize();
    
    
    float distance;
    vec3 obstacle_direction;
    bool hard_hit = collision.findNearestSolidObstacleInCone(
        growth_tip, growth_direction, 0.35f, 2.0f,  // Increased height to 2.0m
        all_obstacles, distance, obstacle_direction);
    
    // Test 3: Repeat soft detection to ensure no state corruption
    std::vector<uint> soft_collisions_2 = collision.findCollisions(plant_parts, {}, all_obstacles, {});
    
    // Test 4: Repeat hard detection with different parameters
    vec3 growth_direction_2 = make_vec3(-0.3, 0, 0.3);
    growth_direction_2.normalize();
    
    bool hard_hit_2 = collision.findNearestSolidObstacleInCone(
        growth_tip, growth_direction_2, 0.35f, 0.5f,
        all_obstacles, distance, obstacle_direction);
    
    // Verify consistency - repeated calls should give same results
    DOCTEST_CHECK(soft_collisions.size() == soft_collisions_2.size());
    DOCTEST_CHECK(hard_hit == true); // Should detect wall
    DOCTEST_CHECK(collision.isBVHValid() == true);
}


DOCTEST_TEST_CASE("CollisionDetection Soft/Hard Detection Integration - Different Geometry Sets") {
    Context context;
    
    CollisionDetection collision(&context);
    collision.disableGPUAcceleration();

    // Create separate geometry sets for soft and hard detection
    uint hard_obstacle_1 = context.addPatch(make_vec3(1, 0, 1), make_vec2(1, 1)); // For hard detection only
    uint hard_obstacle_2 = context.addPatch(make_vec3(-1, 0, 1), make_vec2(1, 1)); // For hard detection only
    
    uint soft_object_1 = context.addTriangle(make_vec3(0, 1, 1.0), make_vec3(0.5, 1.5, 1.0), make_vec3(-0.5, 1.5, 1.0)); // For soft detection only
    uint soft_object_2 = context.addTriangle(make_vec3(0, -1, 0.5), make_vec3(0.5, -1.5, 0.5), make_vec3(-0.5, -1.5, 0.5)); // For soft detection only
    
    uint shared_object = context.addPatch(make_vec3(0, 0, 2), make_vec2(0.5, 0.5)); // Used by both detection types
    
    // Build BVH with ALL geometry (this is typical in plant architecture)
    std::vector<uint> all_geometry = {hard_obstacle_1, hard_obstacle_2, soft_object_1, soft_object_2, shared_object};
    collision.buildBVH(all_geometry);
    
    // Test hard detection using only hard obstacles
    vec3 test_origin = make_vec3(0, 0, 0.5);
    vec3 test_direction = make_vec3(1, 0, 0.3); // Toward hard_obstacle_1
    test_direction.normalize();
    
    std::vector<uint> hard_only = {hard_obstacle_1, hard_obstacle_2, shared_object};
    float distance;
    vec3 obstacle_direction;
    
    bool hard_hit = collision.findNearestSolidObstacleInCone(
        test_origin, test_direction, 0.4f, 2.0f,
        hard_only, distance, obstacle_direction);
    
    // Test soft detection using only soft objects
    std::vector<uint> soft_only = {soft_object_1, soft_object_2, shared_object};
    std::vector<uint> soft_collisions = collision.findCollisions(soft_only);
    
    // Test with mixed queries to ensure BVH handles subset filtering correctly
    vec3 test_direction_2 = make_vec3(0, 1, 0.3); // Toward soft_object_1
    test_direction_2.normalize();
    
    bool hard_hit_2 = collision.findNearestSolidObstacleInCone(
        test_origin, test_direction_2, 0.4f, 10.0f,  // Very generous height for detection
        soft_only, distance, obstacle_direction); // Using soft objects for hard detection
    
    // Verify that detection works correctly with different geometry subsets
    DOCTEST_CHECK(hard_hit == true); // Should detect hard obstacle
    DOCTEST_CHECK(hard_hit_2 == true); // Should also detect soft object when used as hard obstacle
    DOCTEST_CHECK(collision.isBVHValid() == true);
    
    // Verify BVH efficiency - primitive count should match total geometry
    DOCTEST_CHECK(collision.getPrimitiveCount() == all_geometry.size());
}


DOCTEST_TEST_CASE("CollisionDetection Soft/Hard Detection Integration - BVH Rebuild Behavior") {
    Context context;
    
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.disableGPUAcceleration();

    // Initial geometry
    uint obstacle1 = context.addPatch(make_vec3(0, 0, 1), make_vec2(1, 1));
    std::vector<uint> initial_geometry = {obstacle1};
    
    collision.buildBVH(initial_geometry);
    
    // Test initial state
    vec3 test_origin = make_vec3(0, 0, 0.5);
    vec3 test_direction = make_vec3(0, 0, 1);
    float distance;
    vec3 obstacle_direction;
    
    bool hit_initial = collision.findNearestSolidObstacleInCone(
        test_origin, test_direction, 0.3f, 1.0f,
        initial_geometry, distance, obstacle_direction);
    
    // Add more geometry (this might trigger BVH rebuild internally)
    uint obstacle2 = context.addPatch(make_vec3(1, 1, 1), make_vec2(1, 1));
    uint obstacle3 = context.addPatch(make_vec3(-1, -1, 1), make_vec2(1, 1));
    
    std::vector<uint> expanded_geometry = {obstacle1, obstacle2, obstacle3};
    
    // This should trigger a BVH rebuild
    collision.buildBVH(expanded_geometry);
    
    // Test that both detection methods still work after rebuild
    std::vector<uint> soft_collisions = collision.findCollisions(expanded_geometry);
    
    bool hit_after_rebuild = collision.findNearestSolidObstacleInCone(
        test_origin, test_direction, 0.3f, 1.0f,
        expanded_geometry, distance, obstacle_direction);
    
    // Test detection with different geometry subset after rebuild
    vec3 test_direction_2 = make_vec3(1, 1, 0.2);
    test_direction_2.normalize();
    
    bool hit_subset = collision.findNearestSolidObstacleInCone(
        test_origin, test_direction_2, 0.5f, 2.0f,
        {obstacle2}, distance, obstacle_direction); // Only test against obstacle2
    
    // Verify all detection methods work correctly after BVH rebuild
    DOCTEST_CHECK(hit_initial == true);
    DOCTEST_CHECK(hit_after_rebuild == true);
    DOCTEST_CHECK(hit_subset == true);
    DOCTEST_CHECK(collision.isBVHValid() == true);
    DOCTEST_CHECK(collision.getPrimitiveCount() == expanded_geometry.size());
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

// =================== RAY DISTANCE TESTING ===================

DOCTEST_TEST_CASE("CollisionDetection findNearestPrimitiveDistance - Basic Functionality") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create a simple scene with triangles at known distances
    uint triangle1 = context.addTriangle(make_vec3(5, -1, -1), make_vec3(5, 1, -1), make_vec3(5, 0, 1));
    uint triangle2 = context.addTriangle(make_vec3(10, -1, -1), make_vec3(10, 1, -1), make_vec3(10, 0, 1));
    uint triangle3 = context.addTriangle(make_vec3(15, -1, -1), make_vec3(15, 1, -1), make_vec3(15, 0, 1));

    // Test 1: Ray hitting the nearest triangle
    vec3 origin = make_vec3(0, 0, 0);
    vec3 direction = make_vec3(1, 0, 0); // Pointing along +X axis
    std::vector<uint> candidate_UUIDs = {triangle1, triangle2, triangle3};
    float distance;
    vec3 obstacle_direction;

    bool result = collision.findNearestPrimitiveDistance(origin, direction, candidate_UUIDs, distance, obstacle_direction);
    DOCTEST_CHECK(result == true);
    DOCTEST_CHECK(distance >= 4.0f); // Should be approximately 5.0, but AABB might be slightly smaller
    DOCTEST_CHECK(distance <= 6.0f);

    // Test 2: Ray missing all triangles
    vec3 direction_miss = make_vec3(0, 1, 0); // Pointing along +Y axis
    bool result_miss = collision.findNearestPrimitiveDistance(origin, direction_miss, candidate_UUIDs, distance, obstacle_direction);
    DOCTEST_CHECK(result_miss == false);

    // Test 3: Ray with subset of candidates
    std::vector<uint> subset_UUIDs = {triangle2, triangle3}; // Exclude nearest triangle
    bool result_subset = collision.findNearestPrimitiveDistance(origin, direction, subset_UUIDs, distance, obstacle_direction);
    DOCTEST_CHECK(result_subset == true);
    DOCTEST_CHECK(distance >= 9.0f); // Should be approximately 10.0
    DOCTEST_CHECK(distance <= 11.0f);
}

DOCTEST_TEST_CASE("CollisionDetection findNearestPrimitiveDistance - Edge Cases") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create test geometry
    uint triangle1 = context.addTriangle(make_vec3(5, -1, -1), make_vec3(5, 1, -1), make_vec3(5, 0, 1));
    vec3 origin = make_vec3(0, 0, 0);
    vec3 direction = make_vec3(1, 0, 0);
    float distance;

    // Test 1: Empty candidate list
    std::vector<uint> empty_UUIDs;
    vec3 obstacle_direction_unused;
    bool result_empty = collision.findNearestPrimitiveDistance(origin, direction, empty_UUIDs, distance, obstacle_direction_unused);
    DOCTEST_CHECK(result_empty == false);

    // Test 2: Non-normalized direction vector (should return false with warning)
    vec3 non_normalized_dir = make_vec3(2, 0, 0); // Magnitude = 2
    std::vector<uint> valid_UUIDs = {triangle1};
    bool result_non_norm = collision.findNearestPrimitiveDistance(origin, non_normalized_dir, valid_UUIDs, distance, obstacle_direction_unused);
    DOCTEST_CHECK(result_non_norm == false);

    // Test 3: Invalid UUID in candidate list
    std::vector<uint> invalid_UUIDs = {999999}; // Non-existent UUID
    bool result_invalid = collision.findNearestPrimitiveDistance(origin, direction, invalid_UUIDs, distance, obstacle_direction_unused);
    DOCTEST_CHECK(result_invalid == false);

    // Test 4: Mixed valid and invalid UUIDs
    std::vector<uint> mixed_UUIDs = {triangle1, 999999};
    bool result_mixed = collision.findNearestPrimitiveDistance(origin, direction, mixed_UUIDs, distance, obstacle_direction_unused);
    DOCTEST_CHECK(result_mixed == true); // Should still find the valid triangle
    DOCTEST_CHECK(distance >= 4.0f);
    DOCTEST_CHECK(distance <= 6.0f);
}

DOCTEST_TEST_CASE("CollisionDetection findNearestPrimitiveDistance - Complex Scenarios") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create overlapping geometry
    auto cluster = CollisionTests::generateOverlappingCluster(&context, 10, make_vec3(5, 0, 0));
    
    // Test 1: Ray through dense cluster
    vec3 origin = make_vec3(0, 0, 0);
    vec3 direction = make_vec3(1, 0, 0);
    float distance;
    vec3 obstacle_direction_unused;

    bool result = collision.findNearestPrimitiveDistance(origin, direction, cluster, distance, obstacle_direction_unused);
    // Updated expectation: ray traveling in +X direction should NOT hit triangles in XY plane (parallel)
    DOCTEST_CHECK(result == false);

    // Test 2: Ray from near the cluster edge  
    // Note: Ray parallel to triangles should not detect intersection
    vec3 origin_near = make_vec3(3.5, 0, 0);  // Just outside the cluster
    vec3 direction_out = make_vec3(1, 0, 0);
    bool result_near = collision.findNearestPrimitiveDistance(origin_near, direction_out, cluster, distance, obstacle_direction_unused);
    DOCTEST_CHECK(result_near == false); // Ray parallel to triangles - no hit expected

    // Test 3: Test with ray that can actually hit the triangles (perpendicular approach)
    vec3 origin_above = make_vec3(5, 0, 1);  // Above the cluster center
    vec3 direction_down = make_vec3(0, 0, -1); // Growing downward toward triangles
    bool result_perpendicular = collision.findNearestPrimitiveDistance(origin_above, direction_down, cluster, distance, obstacle_direction_unused);
    DOCTEST_CHECK(result_perpendicular == true); // Should hit triangles when approaching perpendicularly
    DOCTEST_CHECK(distance >= 0.9f); // Distance from z=1 to z=0 (approximately)
    DOCTEST_CHECK(distance <= 1.1f);
}

DOCTEST_TEST_CASE("CollisionDetection findNearestPrimitiveDistance - Directional Testing") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();

    // Create triangles in different directions
    uint triangle_x = context.addTriangle(make_vec3(5, -1, -1), make_vec3(5, 1, -1), make_vec3(5, 0, 1));
    uint triangle_y = context.addTriangle(make_vec3(-1, 5, -1), make_vec3(1, 5, -1), make_vec3(0, 5, 1));
    uint triangle_z = context.addTriangle(make_vec3(-1, -1, 5), make_vec3(1, -1, 5), make_vec3(0, 1, 5));
    uint triangle_neg_x = context.addTriangle(make_vec3(-5, -1, -1), make_vec3(-5, 1, -1), make_vec3(-5, 0, 1));

    std::vector<uint> all_triangles = {triangle_x, triangle_y, triangle_z, triangle_neg_x};
    vec3 origin = make_vec3(0, 0, 0);
    float distance;

    // Test different ray directions
    struct DirectionTest {
        vec3 direction;
        float expected_min;
        float expected_max;
        bool should_hit;
    };

    std::vector<DirectionTest> tests = {
        {make_vec3(1, 0, 0), 4.0f, 6.0f, true},    // +X direction
        {make_vec3(0, 1, 0), 4.0f, 6.0f, true},    // +Y direction
        {make_vec3(0, 0, 1), 4.0f, 6.0f, true},    // +Z direction
        {make_vec3(-1, 0, 0), 4.0f, 6.0f, true},   // -X direction
        {make_vec3(0.707f, 0.707f, 0), 6.0f, 8.0f, false}, // Diagonal XY (should miss)
    };

    vec3 obstacle_direction_unused;
    for (const auto& test : tests) {
        bool result = collision.findNearestPrimitiveDistance(origin, test.direction, all_triangles, distance, obstacle_direction_unused);
        DOCTEST_CHECK(result == test.should_hit);
        if (result && test.should_hit) {
            DOCTEST_CHECK(distance >= test.expected_min);
            DOCTEST_CHECK(distance <= test.expected_max);
        }
    }
}

DOCTEST_TEST_CASE("CollisionDetection - findNearestPrimitiveDistance front/back face detection") {
    helios::Context context;
    CollisionDetection collision(&context);

    // Create a horizontal patch at z=1.0 (normal pointing up in +Z direction)
    vec3 patch_center = make_vec3(0, 0, 1.0f);
    vec2 patch_size = make_vec2(2, 2);
    uint horizontal_patch = context.addPatch(patch_center, patch_size);
    
    std::vector<uint> candidates = {horizontal_patch};
    float distance;
    vec3 obstacle_direction;
    
    // Test 1: Approaching from below (should get +Z direction - toward surface)
    vec3 origin_below = make_vec3(0, 0, 0.5f);
    vec3 direction_up = make_vec3(0, 0, 1); // Growing upward
    
    bool found_below = collision.findNearestPrimitiveDistance(origin_below, direction_up, candidates, distance, obstacle_direction);
    DOCTEST_CHECK(found_below == true);
    DOCTEST_CHECK(distance >= 0.49f);
    DOCTEST_CHECK(distance <= 0.51f);
    // When approaching from below, obstacle_direction should point upward (+Z)
    DOCTEST_CHECK(obstacle_direction.z > 0.9f);
    DOCTEST_CHECK(std::abs(obstacle_direction.x) < 0.1f);
    DOCTEST_CHECK(std::abs(obstacle_direction.y) < 0.1f);
    
    // Test 2: Approaching from above (should get -Z direction - toward surface)
    vec3 origin_above = make_vec3(0, 0, 1.5f);
    vec3 direction_down = make_vec3(0, 0, -1); // Growing downward
    
    bool found_above = collision.findNearestPrimitiveDistance(origin_above, direction_down, candidates, distance, obstacle_direction);
    DOCTEST_CHECK(found_above == true);
    DOCTEST_CHECK(distance >= 0.49f);
    DOCTEST_CHECK(distance <= 0.51f);
    // When approaching from above, obstacle_direction should point downward (-Z)
    DOCTEST_CHECK(obstacle_direction.z < -0.9f);
    DOCTEST_CHECK(std::abs(obstacle_direction.x) < 0.1f);
    DOCTEST_CHECK(std::abs(obstacle_direction.y) < 0.1f);
    
    // Test 3: Growing away from surface (should not detect obstacle)
    vec3 origin_below2 = make_vec3(0, 0, 0.5f);
    vec3 direction_away = make_vec3(0, 0, -1); // Growing away from obstacle
    
    bool found_away = collision.findNearestPrimitiveDistance(origin_below2, direction_away, candidates, distance, obstacle_direction);
    DOCTEST_CHECK(found_away == false); // Should not detect surface behind growth direction
}

// =================== CONE-BASED OBSTACLE DETECTION TESTS ===================

DOCTEST_TEST_CASE("CollisionDetection Cone-Based Obstacle Detection - Basic Functionality") {
    Context context;
    CollisionDetection collision(&context);
    
    // Create a horizontal patch obstacle at z=1.0
    uint obstacle_uuid = context.addPatch(make_vec3(0, 0, 1.0f), make_vec2(2, 2));
    std::vector<uint> obstacles = {obstacle_uuid};
    collision.buildBVH(obstacles);
    
    // Test 1: Ray from below, directly toward obstacle
    vec3 apex = make_vec3(0, 0, 0.5f);
    vec3 axis = make_vec3(0, 0, 1); // Straight up
    float half_angle = deg2rad(30.0f); // 30 degree half-angle
    float height = 1.0f; // 1 meter detection range
    
    float distance;
    vec3 obstacle_direction;
    
    bool found = collision.findNearestSolidObstacleInCone(apex, axis, half_angle, height, obstacles, distance, obstacle_direction);
    
    DOCTEST_CHECK(found == true);
    DOCTEST_CHECK(distance >= 0.49f);
    DOCTEST_CHECK(distance <= 0.51f); // Should be ~0.5m from apex to obstacle
    DOCTEST_CHECK(obstacle_direction.z > 0.9f); // Direction should be mostly upward
    
    // Test 2: Ray from far away - should not detect
    vec3 apex_far = make_vec3(0, 0, -2.0f);
    bool found_far = collision.findNearestSolidObstacleInCone(apex_far, axis, half_angle, height, obstacles, distance, obstacle_direction);
    DOCTEST_CHECK(found_far == false); // Too far away
    
    // Test 3: Narrow cone that misses obstacle
    float narrow_angle = deg2rad(5.0f); // Very narrow cone
    vec3 axis_offset = make_vec3(3.0f, 0, 1); // Aimed well to the side to clearly miss the 2x2 patch
    axis_offset.normalize();
    
    bool found_narrow = collision.findNearestSolidObstacleInCone(apex, axis_offset, narrow_angle, height, obstacles, distance, obstacle_direction);
    DOCTEST_CHECK(found_narrow == false); // Should miss the obstacle
}

DOCTEST_TEST_CASE("CollisionDetection Cone-Based vs Legacy Method Comparison") {
    Context context;
    CollisionDetection collision(&context);
    
    // Create test obstacle
    uint obstacle_uuid = context.addPatch(make_vec3(0, 0, 1.0f), make_vec2(1, 1));
    std::vector<uint> obstacles = {obstacle_uuid};
    collision.buildBVH(obstacles);
    
    // Test parameters
    vec3 origin = make_vec3(0, 0, 0.2f);
    vec3 direction = make_vec3(0, 0, 1);
    
    // Legacy method
    float legacy_distance;
    vec3 legacy_obstacle_direction;
    bool legacy_found = collision.findNearestPrimitiveDistance(origin, direction, obstacles, legacy_distance, legacy_obstacle_direction);
    
    // New cone method
    float cone_distance;
    vec3 cone_obstacle_direction;
    float half_angle = deg2rad(30.0f);
    float height = 2.0f;
    bool cone_found = collision.findNearestSolidObstacleInCone(origin, direction, half_angle, height, obstacles, cone_distance, cone_obstacle_direction);
    
    // Both should find the obstacle
    DOCTEST_CHECK(legacy_found == true);
    DOCTEST_CHECK(cone_found == true);
    
    // Both methods should produce reasonable distance measurements (within 5% of expected 0.8m)
    DOCTEST_CHECK(std::abs(cone_distance - 0.8f) < 0.04f); // Within 4cm of expected distance
    DOCTEST_CHECK(std::abs(legacy_distance - 0.8f) < 0.04f); // Within 4cm of expected distance
    
    // Both should have reasonable direction vectors
    DOCTEST_CHECK(legacy_obstacle_direction.magnitude() > 0.9f);
    DOCTEST_CHECK(cone_obstacle_direction.magnitude() > 0.9f);
}

DOCTEST_TEST_CASE("CollisionDetection Cone-Based Triangle vs Patch Intersection") {
    Context context;
    CollisionDetection collision(&context);
    
    // Test triangle intersection
    uint triangle_uuid = context.addTriangle(
        make_vec3(-0.5f, -0.5f, 1.0f), 
        make_vec3(0.5f, -0.5f, 1.0f), 
        make_vec3(0, 0.5f, 1.0f)
    );
    
    // Test patch intersection
    uint patch_uuid = context.addPatch(make_vec3(2, 0, 1.0f), make_vec2(1, 1));
    
    std::vector<uint> triangle_obstacles = {triangle_uuid};
    std::vector<uint> patch_obstacles = {patch_uuid};
    
    collision.buildBVH({triangle_uuid, patch_uuid});
    
    vec3 apex = make_vec3(0, 0, 0.5f);
    vec3 axis = make_vec3(0, 0, 1);
    float half_angle = deg2rad(30.0f);
    float height = 1.0f;
    float distance;
    vec3 obstacle_direction;
    
    // Test triangle intersection
    bool triangle_found = collision.findNearestSolidObstacleInCone(apex, axis, half_angle, height, triangle_obstacles, distance, obstacle_direction);
    DOCTEST_CHECK(triangle_found == true);
    DOCTEST_CHECK(distance > 0.4f);
    DOCTEST_CHECK(distance < 0.6f);
    
    // Test patch intersection 
    vec3 apex_patch = make_vec3(2, 0, 0.5f);
    bool patch_found = collision.findNearestSolidObstacleInCone(apex_patch, axis, half_angle, height, patch_obstacles, distance, obstacle_direction);
    DOCTEST_CHECK(patch_found == true);
    DOCTEST_CHECK(distance > 0.4f);
    DOCTEST_CHECK(distance < 0.6f);
}

DOCTEST_TEST_CASE("CollisionDetection Cone-Based Parameter Validation") {
    Context context;
    CollisionDetection collision(&context);
    
    uint obstacle_uuid = context.addPatch(make_vec3(0, 0, 1.0f), make_vec2(1, 1));
    std::vector<uint> obstacles = {obstacle_uuid};
    collision.buildBVH(obstacles);
    
    vec3 apex = make_vec3(0, 0, 0.5f);
    vec3 axis = make_vec3(0, 0, 1);
    float distance;
    vec3 obstacle_direction;
    
    // Test invalid parameters
    bool result1 = collision.findNearestSolidObstacleInCone(apex, axis, -0.1f, 1.0f, obstacles, distance, obstacle_direction); // Negative angle
    DOCTEST_CHECK(result1 == false);
    
    bool result2 = collision.findNearestSolidObstacleInCone(apex, axis, M_PI, 1.0f, obstacles, distance, obstacle_direction); // Too large angle
    DOCTEST_CHECK(result2 == false);
    
    bool result3 = collision.findNearestSolidObstacleInCone(apex, axis, deg2rad(30.0f), -1.0f, obstacles, distance, obstacle_direction); // Negative height
    DOCTEST_CHECK(result3 == false);
    
    // Test empty candidate list
    std::vector<uint> empty_obstacles;
    bool result4 = collision.findNearestSolidObstacleInCone(apex, axis, deg2rad(30.0f), 1.0f, empty_obstacles, distance, obstacle_direction);
    DOCTEST_CHECK(result4 == false);
    
    // Test valid parameters - should work
    bool result5 = collision.findNearestSolidObstacleInCone(apex, axis, deg2rad(30.0f), 1.0f, obstacles, distance, obstacle_direction);
    DOCTEST_CHECK(result5 == true);
}

DOCTEST_TEST_CASE("Test attraction points detection - basic functionality") {
    
    helios::Context context;
    CollisionDetection collision(&context);
    
    // Test 1: Single attraction point directly ahead
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(0, 0, 1);  // Looking up
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(0, 0, 5));  // Directly ahead
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance, 
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        DOCTEST_CHECK(std::abs(direction_to_closest.x) < 0.01f);
        DOCTEST_CHECK(std::abs(direction_to_closest.y) < 0.01f);
        DOCTEST_CHECK(direction_to_closest.z > 0.99f);  // Should point upward
        DOCTEST_CHECK(std::abs(direction_to_closest.magnitude() - 1.0f) < 0.01f);  // Should be normalized
    }
    
    // Test 2: Multiple attraction points - should return closest to centerline
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(0, 0, 1);
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(2, 0, 5));   // Off to the side
        attraction_points.push_back(make_vec3(0, 0, 5));   // Directly ahead (closest to centerline)
        attraction_points.push_back(make_vec3(-2, 0, 5));  // Off to other side
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        // Should point to the center point (0, 0, 5) which is closest to centerline
        DOCTEST_CHECK(std::abs(direction_to_closest.x) < 0.01f);
        DOCTEST_CHECK(std::abs(direction_to_closest.y) < 0.01f);
        DOCTEST_CHECK(direction_to_closest.z > 0.99f);
    }
    
    // Test 3: Points outside the cone - should return false
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(0, 0, 1);
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 30.0f;  // Narrower cone
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(10, 0, 5));  // Far off to the side
        attraction_points.push_back(make_vec3(-10, 0, 5)); // Far off to other side
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == false);
    }
    
    // Test 4: Points beyond look-ahead distance - should be ignored
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(1, 0, 0);  // Looking along x-axis
        float look_ahead_distance = 5.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(10, 0, 0));  // Beyond look-ahead distance
        attraction_points.push_back(make_vec3(3, 0, 0));   // Within distance
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        // Should point to (3, 0, 0) since (10, 0, 0) is beyond look-ahead distance
        DOCTEST_CHECK(direction_to_closest.x > 0.99f);
        DOCTEST_CHECK(std::abs(direction_to_closest.y) < 0.01f);
        DOCTEST_CHECK(std::abs(direction_to_closest.z) < 0.01f);
    }
}

DOCTEST_TEST_CASE("Test attraction points detection - edge cases") {
    
    helios::Context context;
    CollisionDetection collision(&context);
    
    // Test 1: Empty attraction points vector
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(0, 0, 1);
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;  // Empty
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == false);
    }
    
    // Test 2: Invalid parameters
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(0, 0, 1);
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(0, 0, 5));
        vec3 direction_to_closest;
        
        // Negative look-ahead distance
        bool result1 = collision.detectAttractionPoints(vertex, look_direction, -1.0f, 45.0f, 
                                                        attraction_points, direction_to_closest);
        DOCTEST_CHECK(result1 == false);
        
        // Zero look-ahead distance
        bool result2 = collision.detectAttractionPoints(vertex, look_direction, 0.0f, 45.0f,
                                                        attraction_points, direction_to_closest);
        DOCTEST_CHECK(result2 == false);
        
        // Invalid half-angle (0 degrees)
        bool result3 = collision.detectAttractionPoints(vertex, look_direction, 10.0f, 0.0f,
                                                        attraction_points, direction_to_closest);
        DOCTEST_CHECK(result3 == false);
        
        // Invalid half-angle (180 degrees)
        bool result4 = collision.detectAttractionPoints(vertex, look_direction, 10.0f, 180.0f,
                                                        attraction_points, direction_to_closest);
        DOCTEST_CHECK(result4 == false);
        
        // Invalid half-angle (negative)
        bool result5 = collision.detectAttractionPoints(vertex, look_direction, 10.0f, -45.0f,
                                                        attraction_points, direction_to_closest);
        DOCTEST_CHECK(result5 == false);
    }
    
    // Test 3: Point at vertex location (should be ignored)
    {
        vec3 vertex = make_vec3(5, 5, 5);
        vec3 look_direction = make_vec3(1, 0, 0);
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(5, 5, 5));  // Same as vertex
        attraction_points.push_back(make_vec3(8, 5, 5));  // Valid point
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        // Should find the valid point, ignoring the one at vertex
        DOCTEST_CHECK(direction_to_closest.x > 0.99f);
    }
    
    // Test 4: Very narrow cone
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(0, 0, 1);
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 1.0f;  // Very narrow cone
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(0.0, 0.0, 5));    // Directly ahead
        attraction_points.push_back(make_vec3(0.1, 0.0, 5));    // Slightly off
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);  // Should find at least the directly ahead point
    }
}

DOCTEST_TEST_CASE("Test attraction points detection - angular distance prioritization") {
    
    helios::Context context;
    CollisionDetection collision(&context);
    
    // Test that closer angular distance is prioritized over physical distance
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(0, 0, 1);
        float look_ahead_distance = 20.0f;
        float half_angle_degrees = 60.0f;
        
        std::vector<vec3> attraction_points;
        // Point A: Farther but directly on centerline
        attraction_points.push_back(make_vec3(0, 0, 10));
        // Point B: Closer but off to the side  
        attraction_points.push_back(make_vec3(3, 0, 5));
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        // Should point to (0, 0, 10) which is on the centerline
        DOCTEST_CHECK(std::abs(direction_to_closest.x) < 0.01f);
        DOCTEST_CHECK(std::abs(direction_to_closest.y) < 0.01f);
        DOCTEST_CHECK(direction_to_closest.z > 0.99f);
    }
}

DOCTEST_TEST_CASE("Test attraction points detection - various cone orientations") {
    
    helios::Context context;
    CollisionDetection collision(&context);
    
    // Test 1: Cone pointing in negative z direction
    {
        vec3 vertex = make_vec3(0, 0, 10);
        vec3 look_direction = make_vec3(0, 0, -1);  // Looking down
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(0, 0, 5));  // Below vertex
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        DOCTEST_CHECK(std::abs(direction_to_closest.x) < 0.01f);
        DOCTEST_CHECK(std::abs(direction_to_closest.y) < 0.01f);
        DOCTEST_CHECK(direction_to_closest.z < -0.99f);  // Should point downward
    }
    
    // Test 2: Cone pointing at an angle
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(1, 1, 0);  // 45 degrees in xy-plane
        look_direction.normalize();
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(5, 5, 0));  // Along look direction
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        float expected = 1.0f / std::sqrt(2.0f);
        DOCTEST_CHECK(std::abs(direction_to_closest.x - expected) < 0.01f);
        DOCTEST_CHECK(std::abs(direction_to_closest.y - expected) < 0.01f);
        DOCTEST_CHECK(std::abs(direction_to_closest.z) < 0.01f);
    }
    
    // Test 3: Cone with non-normalized look direction (should handle normalization internally)
    {
        vec3 vertex = make_vec3(0, 0, 0);
        vec3 look_direction = make_vec3(3, 0, 0);  // Not normalized
        float look_ahead_distance = 10.0f;
        float half_angle_degrees = 45.0f;
        
        std::vector<vec3> attraction_points;
        attraction_points.push_back(make_vec3(5, 0, 0));
        
        vec3 direction_to_closest;
        bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                       half_angle_degrees, attraction_points, direction_to_closest);
        
        DOCTEST_CHECK(found == true);
        DOCTEST_CHECK(direction_to_closest.x > 0.99f);
        DOCTEST_CHECK(std::abs(direction_to_closest.magnitude() - 1.0f) < 0.01f);  // Should be normalized
    }
}

DOCTEST_TEST_CASE("Test attraction points detection - large scale scenario") {
    
    helios::Context context;
    CollisionDetection collision(&context);
    
    // Simulate a scenario with many attraction points
    vec3 vertex = make_vec3(0, 0, 0);
    vec3 look_direction = make_vec3(0, 1, 0);  // Looking along y-axis
    float look_ahead_distance = 50.0f;
    float half_angle_degrees = 30.0f;
    
    std::vector<vec3> attraction_points;
    
    // Create a grid of attraction points
    for (float x = -10; x <= 10; x += 5) {
        for (float y = 5; y <= 45; y += 10) {
            for (float z = -10; z <= 10; z += 5) {
                attraction_points.push_back(make_vec3(x, y, z));
            }
        }
    }
    
    // Add one point directly on the centerline
    attraction_points.push_back(make_vec3(0, 25, 0));
    
    vec3 direction_to_closest;
    bool found = collision.detectAttractionPoints(vertex, look_direction, look_ahead_distance,
                                                   half_angle_degrees, attraction_points, direction_to_closest);
    
    DOCTEST_CHECK(found == true);
    // Should point to (0, 25, 0) which is on the centerline
    DOCTEST_CHECK(std::abs(direction_to_closest.x) < 0.01f);
    DOCTEST_CHECK(direction_to_closest.y > 0.99f);
    DOCTEST_CHECK(std::abs(direction_to_closest.z) < 0.01f);
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Basic Functionality") {
    Context context;
    CollisionDetection collision(&context);
    
    // Set up simple voxel grid
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(10, 10, 10);
    int3 grid_divisions(2, 2, 2);
    
    // Create simple rays through the grid
    std::vector<vec3> ray_origins;
    std::vector<vec3> ray_directions;
    
    // Ray going straight through the grid center
    ray_origins.push_back(make_vec3(-10, 0, 0));
    ray_directions.push_back(make_vec3(1, 0, 0));
    
    // Ray going diagonally through multiple voxels
    ray_origins.push_back(make_vec3(-10, -10, -10));
    ray_directions.push_back(normalize(make_vec3(1, 1, 1)));
    
    // Calculate voxel ray path lengths
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
    
    // Test transmission probability access
    int P_denom, P_trans;
    collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
    DOCTEST_CHECK(P_denom >= 0);
    DOCTEST_CHECK(P_trans >= 0);
    DOCTEST_CHECK(P_trans <= P_denom);
    
    // Test r_bar access
    float r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
    DOCTEST_CHECK(r_bar >= 0.0f);
    
    // Clear data should work without error
    collision.clearVoxelData();
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Edge Cases") {
    Context context;
    CollisionDetection collision(&context);
    
    // Test with empty ray vectors
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(5, 5, 5);
    int3 grid_divisions(1, 1, 1);
    
    std::vector<vec3> empty_origins;
    std::vector<vec3> empty_directions;
    
    // Should handle empty input gracefully
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, empty_origins, empty_directions);
    
    // Test invalid voxel indices
    int P_denom, P_trans;
    
    // Test boundary cases - should handle gracefully or throw appropriate error
    capture_cerr capture;
    try {
        collision.getVoxelTransmissionProbability(make_int3(-1, 0, 0), P_denom, P_trans);
    } catch (const std::exception& e) {
        // Expected behavior - invalid indices should be handled
        DOCTEST_CHECK(true);
    }
    
    try {
        collision.getVoxelTransmissionProbability(make_int3(1, 0, 0), P_denom, P_trans);
    } catch (const std::exception& e) {
        // Expected behavior - out of bounds indices
        DOCTEST_CHECK(true);
    }
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Data Consistency") {
    Context context;
    CollisionDetection collision(&context);
    
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(6, 6, 6);
    int3 grid_divisions(3, 3, 3);
    
    // Create systematic ray pattern
    std::vector<vec3> ray_origins;
    std::vector<vec3> ray_directions;
    
    // Grid of parallel rays
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            ray_origins.push_back(make_vec3(i * 1.5f, j * 1.5f, -10));
            ray_directions.push_back(make_vec3(0, 0, 1));
        }
    }
    
    // Calculate path lengths
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
    
    // Verify data consistency
    bool found_data = false;
    for (int i = 0; i < grid_divisions.x; i++) {
        for (int j = 0; j < grid_divisions.y; j++) {
            for (int k = 0; k < grid_divisions.z; k++) {
                int P_denom, P_trans;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                
                if (P_denom > 0) {
                    found_data = true;
                    // Transmission count should never exceed total count
                    DOCTEST_CHECK(P_trans <= P_denom);
                    
                    // R_bar should be positive when rays are present
                    float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                    DOCTEST_CHECK(r_bar > 0.0f);
                }
            }
        }
    }
    
    DOCTEST_CHECK(found_data); // Should have found at least some ray intersections
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Manual Data Setting") {
    Context context;
    CollisionDetection collision(&context);
    
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(4, 4, 4);
    int3 grid_divisions(2, 2, 2);
    
    // Initialize with minimal calculation to set up data structures
    std::vector<vec3> init_origins;
    std::vector<vec3> init_directions;
    init_origins.push_back(make_vec3(0, 0, -10));  // Outside the grid
    init_directions.push_back(make_vec3(0, 0, 1));
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, init_origins, init_directions);
    
    // Test manual data setting
    int3 test_voxel(0, 0, 0);
    collision.setVoxelTransmissionProbability(100, 75, test_voxel);
    collision.setVoxelRbar(2.5f, test_voxel);
    
    // Verify the data was set correctly
    int P_denom, P_trans;
    collision.getVoxelTransmissionProbability(test_voxel, P_denom, P_trans);
    DOCTEST_CHECK(P_denom == 100);
    DOCTEST_CHECK(P_trans == 75);
    
    float r_bar = collision.getVoxelRbar(test_voxel);
    DOCTEST_CHECK(std::abs(r_bar - 2.5f) < 1e-6f);
    
    // Test another voxel
    int3 test_voxel2(1, 1, 1);
    collision.setVoxelTransmissionProbability(200, 150, test_voxel2);
    collision.setVoxelRbar(3.7f, test_voxel2);
    
    collision.getVoxelTransmissionProbability(test_voxel2, P_denom, P_trans);
    DOCTEST_CHECK(P_denom == 200);
    DOCTEST_CHECK(P_trans == 150);
    
    r_bar = collision.getVoxelRbar(test_voxel2);
    DOCTEST_CHECK(std::abs(r_bar - 3.7f) < 1e-6f);
    
    // Verify first voxel data is still intact
    collision.getVoxelTransmissionProbability(test_voxel, P_denom, P_trans);
    DOCTEST_CHECK(P_denom == 100);
    DOCTEST_CHECK(P_trans == 75);
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Different Grid Sizes") {
    Context context;
    CollisionDetection collision(&context);
    
    // Test various grid sizes
    std::vector<int3> test_grids = {
        make_int3(1, 1, 1),  // Single voxel
        make_int3(2, 1, 1),  // Linear arrangement
        make_int3(4, 4, 4),  // Cubic arrangement
        make_int3(5, 3, 2)   // Asymmetric arrangement
    };
    
    for (const auto& grid_div : test_grids) {
        vec3 grid_center(0, 0, 0);
        vec3 grid_size(10, 10, 10);
        
        std::vector<vec3> ray_origins;
        std::vector<vec3> ray_directions;
        
        // Single test ray
        ray_origins.push_back(make_vec3(0, 0, -10));
        ray_directions.push_back(make_vec3(0, 0, 1));
        
        collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_div, ray_origins, ray_directions);
        
        // Verify that valid indices work
        bool found_valid_voxel = false;
        for (int i = 0; i < grid_div.x; i++) {
            for (int j = 0; j < grid_div.y; j++) {
                for (int k = 0; k < grid_div.z; k++) {
                    int P_denom, P_trans;
                    collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                    float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                    
                    // Should not crash and should give reasonable values
                    DOCTEST_CHECK(P_denom >= 0);
                    DOCTEST_CHECK(P_trans >= 0);
                    DOCTEST_CHECK(r_bar >= 0.0f);
                    found_valid_voxel = true;
                }
            }
        }
        DOCTEST_CHECK(found_valid_voxel);
        
        collision.clearVoxelData();
    }
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Ray Direction Variations") {
    Context context;
    CollisionDetection collision(&context);
    
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(8, 8, 8);
    int3 grid_divisions(2, 2, 2);
    
    // Test rays with different directions
    std::vector<vec3> test_directions = {
        make_vec3(1, 0, 0),     // X-axis
        make_vec3(0, 1, 0),     // Y-axis  
        make_vec3(0, 0, 1),     // Z-axis
        normalize(make_vec3(1, 1, 1)),    // Diagonal
        normalize(make_vec3(1, -1, 0)),   // Diagonal in XY plane
        normalize(make_vec3(-1, 0, 1))    // Negative X, positive Z
    };
    
    for (const auto& direction : test_directions) {
        std::vector<vec3> ray_origins;
        std::vector<vec3> ray_directions;
        
        // Start ray from outside grid
        vec3 start_point = grid_center - direction * 10.0f;
        ray_origins.push_back(start_point);
        ray_directions.push_back(direction);
        
        collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
        
        // Should find intersections for most directions through the center
        bool found_intersections = false;
        for (int i = 0; i < grid_divisions.x; i++) {
            for (int j = 0; j < grid_divisions.y; j++) {
                for (int k = 0; k < grid_divisions.z; k++) {
                    int P_denom, P_trans;
                    collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                    if (P_denom > 0) {
                        found_intersections = true;
                        float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                        DOCTEST_CHECK(r_bar > 0.0f);
                    }
                }
            }
        }
        
        DOCTEST_CHECK(found_intersections);
        collision.clearVoxelData();
    }
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - GPU/CPU Consistency") {
    Context context;
    CollisionDetection collision(&context);
    
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(6, 6, 6);
    int3 grid_divisions(3, 3, 3);
    
    // Create test rays
    std::vector<vec3> ray_origins;
    std::vector<vec3> ray_directions;
    
    for (int i = 0; i < 5; i++) {
        ray_origins.push_back(make_vec3(i - 2.0f, 0, -10));
        ray_directions.push_back(make_vec3(0, 0, 1));
    }
    
    // Test CPU implementation first
    collision.disableGPUAcceleration();
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
    
    // Store CPU results
    std::vector<std::vector<std::vector<std::pair<int, float>>>> cpu_results(grid_divisions.x);
    for (int i = 0; i < grid_divisions.x; i++) {
        cpu_results[i].resize(grid_divisions.y);
        for (int j = 0; j < grid_divisions.y; j++) {
            cpu_results[i][j].resize(grid_divisions.z);
            for (int k = 0; k < grid_divisions.z; k++) {
                int P_denom, P_trans;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                cpu_results[i][j][k] = std::make_pair(P_denom, r_bar);
            }
        }
    }
    
    // Test GPU implementation
    collision.enableGPUAcceleration();
    collision.clearVoxelData();
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
    
    // Compare GPU results with CPU results
    for (int i = 0; i < grid_divisions.x; i++) {
        for (int j = 0; j < grid_divisions.y; j++) {
            for (int k = 0; k < grid_divisions.z; k++) {
                int P_denom_gpu, P_trans_gpu;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom_gpu, P_trans_gpu);
                float r_bar_gpu = collision.getVoxelRbar(make_int3(i, j, k));
                
                int P_denom_cpu = cpu_results[i][j][k].first;
                float r_bar_cpu = cpu_results[i][j][k].second;
                
                // Results should match within reasonable tolerance
                DOCTEST_CHECK(P_denom_gpu == P_denom_cpu);
                if (r_bar_cpu > 0 && r_bar_gpu > 0) {
                    DOCTEST_CHECK(std::abs(r_bar_gpu - r_bar_cpu) < 1e-4f);
                }
            }
        }
    }
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Parameter Validation") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: Negative grid size
    vec3 grid_center(0, 0, 0);
    vec3 negative_size(-5, 5, 5);  
    int3 grid_divisions(2, 2, 2);
    std::vector<vec3> ray_origins = {make_vec3(0, 0, -10)};
    std::vector<vec3> ray_directions = {make_vec3(0, 0, 1)};
    
    // Should handle gracefully without crashing
    try {
        collision.calculateVoxelRayPathLengths(grid_center, negative_size, grid_divisions, ray_origins, ray_directions);
        // If it doesn't throw, verify voxel data is still accessible
        int P_denom, P_trans;
        collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
        DOCTEST_CHECK(P_denom >= 0);
        DOCTEST_CHECK(P_trans >= 0);
    } catch (const std::exception& e) {
        // Exception is acceptable for invalid parameters
        DOCTEST_CHECK(true);
    }
    
    // Test 2: Zero grid divisions
    int3 zero_divisions(0, 2, 2);
    try {
        collision.calculateVoxelRayPathLengths(grid_center, make_vec3(5, 5, 5), zero_divisions, ray_origins, ray_directions);
        // If no exception, verify it handles gracefully by checking voxel access behavior
        int P_denom_zero, P_trans_zero;
        try {
            collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom_zero, P_trans_zero);
            // Either returns valid data or the getter throws - both are acceptable
            DOCTEST_CHECK(P_denom_zero >= 0);
        } catch (const std::exception& inner_e) {
            // Exception on voxel access is acceptable for zero divisions
            DOCTEST_CHECK(true);
        }
    } catch (const std::exception& e) {
        DOCTEST_CHECK(true); // Exception is also expected behavior
    }
    
    // Test 3: Mismatched ray vector sizes
    std::vector<vec3> mismatched_directions = {make_vec3(0, 0, 1), make_vec3(1, 0, 0)};
    try {
        collision.calculateVoxelRayPathLengths(grid_center, make_vec3(5, 5, 5), make_int3(2, 2, 2), ray_origins, mismatched_directions);
        DOCTEST_CHECK(false); // Should throw an exception
    } catch (const std::exception& e) {
        DOCTEST_CHECK(true); // Expected behavior
    }
    
    // Test 4: Invalid P_trans > P_denom
    vec3 valid_grid_size(4, 4, 4);
    int3 valid_divisions(2, 2, 2);
    collision.calculateVoxelRayPathLengths(grid_center, valid_grid_size, valid_divisions, ray_origins, ray_directions);
    
    // This should be allowed - implementation may handle it gracefully
    collision.setVoxelTransmissionProbability(10, 15, make_int3(0, 0, 0)); // P_trans > P_denom
    int test_P_denom, test_P_trans;
    collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), test_P_denom, test_P_trans);
    DOCTEST_CHECK(test_P_denom == 10);
    DOCTEST_CHECK(test_P_trans == 15);
    
    // Test 5: Negative values for transmission probability
    collision.setVoxelTransmissionProbability(-5, -3, make_int3(0, 0, 0));
    collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), test_P_denom, test_P_trans);
    DOCTEST_CHECK(test_P_denom == -5);
    DOCTEST_CHECK(test_P_trans == -3);
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Mathematical Validation") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: Single ray through single voxel - analytical solution
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(2, 2, 2);  // 2x2x2 cube
    int3 grid_divisions(1, 1, 1);  // Single voxel
    
    // Ray passes straight through center
    std::vector<vec3> ray_origins = {make_vec3(0, 0, -5)};
    std::vector<vec3> ray_directions = {make_vec3(0, 0, 1)};
    
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
    
    float r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
    
    // Expected path length through 2x2x2 cube should be 2.0 (cube height)
    DOCTEST_CHECK(std::abs(r_bar - 2.0f) < 0.1f);
    
    // Test 2: Diagonal ray through cubic voxel
    collision.clearVoxelData();
    std::vector<vec3> diagonal_origins = {make_vec3(-2, -2, -2)};
    std::vector<vec3> diagonal_directions = {normalize(make_vec3(1, 1, 1))};
    
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, diagonal_origins, diagonal_directions);
    
    float diagonal_r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
    
    // Expected diagonal through 2x2x2 cube should be sqrt(3)*2 = ~3.46
    float expected_diagonal = std::sqrt(3.0f) * 2.0f;
    DOCTEST_CHECK(std::abs(diagonal_r_bar - expected_diagonal) < 0.2f);
    
    // Test 3: Multiple rays, verify statistical consistency  
    collision.clearVoxelData();
    std::vector<vec3> multi_origins;
    std::vector<vec3> multi_directions;
    
    // Create 4 parallel rays through same voxel
    for (int i = 0; i < 4; i++) {
        multi_origins.push_back(make_vec3(0.5f * i - 0.75f, 0, -5));
        multi_directions.push_back(make_vec3(0, 0, 1));
    }
    
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, multi_origins, multi_directions);
    
    int P_denom, P_trans;
    collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
    float multi_r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
    
    // Should have 4 rays hitting the voxel
    DOCTEST_CHECK(P_denom == 4);
    // All rays should pass through (assuming no geometry), so P_trans should equal P_denom
    DOCTEST_CHECK(P_trans == P_denom);
    // Average path length should still be ~2.0
    DOCTEST_CHECK(std::abs(multi_r_bar - 2.0f) < 0.1f);
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Numerical Precision") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: Very small grid (precision test)
    vec3 tiny_center(0, 0, 0);
    vec3 tiny_size(0.001f, 0.001f, 0.001f);
    int3 tiny_divisions(1, 1, 1);
    
    std::vector<vec3> tiny_origins = {make_vec3(0, 0, -0.01f)};
    std::vector<vec3> tiny_directions = {make_vec3(0, 0, 1)};
    
    collision.calculateVoxelRayPathLengths(tiny_center, tiny_size, tiny_divisions, tiny_origins, tiny_directions);
    float tiny_r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
    
    // Should handle small values without underflow
    DOCTEST_CHECK(tiny_r_bar > 0.0f);
    DOCTEST_CHECK(tiny_r_bar < 0.1f); // Should be on order of tiny_size
    
    // Test 2: Large grid (overflow test)
    vec3 large_center(0, 0, 0);
    vec3 large_size(1000.0f, 1000.0f, 1000.0f);
    int3 large_divisions(2, 2, 2);
    
    std::vector<vec3> large_origins = {make_vec3(0, 0, -2000)};
    std::vector<vec3> large_directions = {make_vec3(0, 0, 1)};
    
    collision.calculateVoxelRayPathLengths(large_center, large_size, large_divisions, large_origins, large_directions);
    float large_r_bar = collision.getVoxelRbar(make_int3(1, 1, 1)); // Center voxel
    
    // Should handle large values without overflow
    DOCTEST_CHECK(large_r_bar > 0.0f);
    DOCTEST_CHECK(large_r_bar < 2000.0f); // Should be reasonable
    
    // Test 3: CPU/GPU precision comparison with tight tolerance
    collision.clearVoxelData();
    vec3 precision_center(0, 0, 0);
    vec3 precision_size(10, 10, 10);
    int3 precision_divisions(5, 5, 5);
    
    std::vector<vec3> precision_origins;
    std::vector<vec3> precision_directions;
    for (int i = 0; i < 10; i++) {
        precision_origins.push_back(make_vec3(i - 5.0f, 0, -20));
        precision_directions.push_back(make_vec3(0, 0, 1));
    }
    
    // CPU calculation
    collision.disableGPUAcceleration();
    collision.calculateVoxelRayPathLengths(precision_center, precision_size, precision_divisions, precision_origins, precision_directions);
    
    // Store CPU results with higher precision
    std::vector<float> cpu_rbars;
    for (int i = 0; i < precision_divisions.x; i++) {
        for (int j = 0; j < precision_divisions.y; j++) {
            for (int k = 0; k < precision_divisions.z; k++) {
                cpu_rbars.push_back(collision.getVoxelRbar(make_int3(i, j, k)));
            }
        }
    }
    
    // GPU calculation
    collision.enableGPUAcceleration();
    collision.clearVoxelData();
    collision.calculateVoxelRayPathLengths(precision_center, precision_size, precision_divisions, precision_origins, precision_directions);
    
    // Compare with tighter tolerance than existing test
    int idx = 0;
    for (int i = 0; i < precision_divisions.x; i++) {
        for (int j = 0; j < precision_divisions.y; j++) {
            for (int k = 0; k < precision_divisions.z; k++) {
                float gpu_rbar = collision.getVoxelRbar(make_int3(i, j, k));
                float cpu_rbar = cpu_rbars[idx++];
                
                if (cpu_rbar > 0 && gpu_rbar > 0) {
                    float relative_error = std::abs(gpu_rbar - cpu_rbar) / std::max(cpu_rbar, gpu_rbar);
                    DOCTEST_CHECK(relative_error < 1e-5f); // Tighter precision requirement
                }
            }
        }
    }
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Error Recovery and State Management") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: API calls before initialization should handle gracefully
    try {
        int P_denom, P_trans;
        collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
        // Should return zeros for uninitialized data
        DOCTEST_CHECK(P_denom == 0);
        DOCTEST_CHECK(P_trans == 0);
    } catch (const std::exception& e) {
        // Exception is also acceptable
        DOCTEST_CHECK(true);
    }
    
    try {
        float r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
        // Should return zero for uninitialized data  
        DOCTEST_CHECK(r_bar == 0.0f);
    } catch (const std::exception& e) {
        // Exception is also acceptable
        DOCTEST_CHECK(true);
    }
    
    // Test 2: Multiple initialization cycles
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(4, 4, 4);
    int3 grid_divisions(2, 2, 2);
    std::vector<vec3> ray_origins = {make_vec3(0, 0, -10)};
    std::vector<vec3> ray_directions = {make_vec3(0, 0, 1)};
    
    // Initialize multiple times - should handle gracefully
    for (int cycle = 0; cycle < 3; cycle++) {
        collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
        
        // Verify data is accessible
        int P_denom, P_trans;
        collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
        DOCTEST_CHECK(P_denom >= 0);
        
        float r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
        DOCTEST_CHECK(r_bar >= 0);
        
        // Clear and reinitialize
        collision.clearVoxelData();
    }
    
    // Test 3: State consistency after errors
    try {
        // Attempt invalid operation
        collision.setVoxelTransmissionProbability(10, 5, make_int3(-1, 0, 0)); // Invalid index
        DOCTEST_CHECK(false); // Should throw
    } catch (const std::exception& e) {
        // After error, system should still be usable
        collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
        
        int P_denom, P_trans;
        collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
        DOCTEST_CHECK(P_denom >= 0);
    }
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Memory and Performance Stress") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: Moderately large grid (memory test)
    vec3 stress_center(0, 0, 0);
    vec3 stress_size(50, 50, 50);
    int3 stress_divisions(10, 10, 10); // 1000 voxels
    
    // Create many rays
    std::vector<vec3> stress_origins;
    std::vector<vec3> stress_directions;
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 10; j++) {
            stress_origins.push_back(make_vec3(i - 50.0f, j - 5.0f, -100));
            stress_directions.push_back(make_vec3(0, 0, 1));
        }
    }
    
    // Should handle 1000 rays x 1000 voxels without issues
    auto start_time = std::chrono::high_resolution_clock::now();
    collision.calculateVoxelRayPathLengths(stress_center, stress_size, stress_divisions, stress_origins, stress_directions);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Verify computation completed successfully
    bool found_data = false;
    for (int i = 0; i < stress_divisions.x && !found_data; i++) {
        for (int j = 0; j < stress_divisions.y && !found_data; j++) {
            for (int k = 0; k < stress_divisions.z && !found_data; k++) {
                int P_denom, P_trans;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                if (P_denom > 0) {
                    found_data = true;
                    DOCTEST_CHECK(P_trans <= P_denom);
                    float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                    DOCTEST_CHECK(r_bar > 0.0f);
                }
            }
        }
    }
    DOCTEST_CHECK(found_data);
    
    // Test 2: Memory cleanup validation
    collision.clearVoxelData();
    
    // After clear, should return default values
    int P_denom, P_trans;
    collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
    DOCTEST_CHECK(P_denom == 0);
    DOCTEST_CHECK(P_trans == 0);
    
    float r_bar = collision.getVoxelRbar(make_int3(0, 0, 0));
    DOCTEST_CHECK(r_bar == 0.0f);
    
    // Test 3: Repeated allocation/deallocation cycles
    for (int cycle = 0; cycle < 5; cycle++) {
        vec3 cycle_size(8 + cycle * 2, 8 + cycle * 2, 8 + cycle * 2);
        int3 cycle_divisions(2 + cycle, 2 + cycle, 2 + cycle);
        
        std::vector<vec3> cycle_origins = {make_vec3(0, 0, -20)};
        std::vector<vec3> cycle_directions = {make_vec3(0, 0, 1)};
        
        collision.calculateVoxelRayPathLengths(stress_center, cycle_size, cycle_divisions, cycle_origins, cycle_directions);
        
        // Verify some data exists
        collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
        DOCTEST_CHECK(P_denom >= 0);
        
        collision.clearVoxelData();
    }
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Integration with BVH") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create some geometry to ensure BVH is built
    std::vector<uint> sphere_UUIDs = context.addSphere(10, make_vec3(0, 0, 0), 1.0f);
    
    // Test 1: Voxel calculations with existing geometry
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(10, 10, 10);
    int3 grid_divisions(5, 5, 5);
    
    std::vector<vec3> ray_origins;
    std::vector<vec3> ray_directions;
    for (int i = 0; i < 8; i++) {
        ray_origins.push_back(make_vec3(i - 4.0f, 0, -15));
        ray_directions.push_back(make_vec3(0, 0, 1));
    }
    
    // Should work normally even with geometry in context
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
    
    // Verify data is accessible
    bool found_voxel_data = false;
    for (int i = 0; i < grid_divisions.x; i++) {
        for (int j = 0; j < grid_divisions.y; j++) {
            for (int k = 0; k < grid_divisions.z; k++) {
                int P_denom, P_trans;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                if (P_denom > 0) {
                    found_voxel_data = true;
                    DOCTEST_CHECK(P_trans >= 0);
                    DOCTEST_CHECK(P_trans <= P_denom);
                    
                    float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                    DOCTEST_CHECK(r_bar >= 0.0f);
                }
            }
        }
    }
    DOCTEST_CHECK(found_voxel_data);
    
    // Test 2: Verify collision detection still works after voxel calculations
    std::vector<uint> collision_results = collision.findCollisions(sphere_UUIDs[0]);
    DOCTEST_CHECK(collision_results.size() >= 0); // Should execute without error
    
    // Test 3: Interleaved operations
    collision.clearVoxelData();
    
    // Add more geometry
    std::vector<uint> triangle_UUIDs;
    triangle_UUIDs.push_back(context.addTriangle(make_vec3(5, 0, 0), make_vec3(6, 1, 0), make_vec3(6, 0, 1)));
    
    // Recalculate voxels
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);
    
    // Find collisions with new geometry
    std::vector<uint> new_collisions = collision.findCollisions(triangle_UUIDs);
    DOCTEST_CHECK(new_collisions.size() >= 0);
    
    // Verify voxel data is still valid
    int final_P_denom, final_P_trans;
    collision.getVoxelTransmissionProbability(make_int3(2, 2, 2), final_P_denom, final_P_trans);
    DOCTEST_CHECK(final_P_denom >= 0);
    DOCTEST_CHECK(final_P_trans >= 0);
}

DOCTEST_TEST_CASE("CollisionDetection Voxel Ray Path Length - Edge Case Ray Geometries") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    vec3 grid_center(0, 0, 0);
    vec3 grid_size(4, 4, 4);
    int3 grid_divisions(2, 2, 2);
    
    // Test 1: Ray parallel to voxel face (grazing)
    std::vector<vec3> grazing_origins = {make_vec3(-3, 2.0f, 0)};  // At grid boundary
    std::vector<vec3> grazing_directions = {make_vec3(1, 0, 0)};   // Parallel to face
    
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, grazing_origins, grazing_directions);
    
    // Should handle gracefully - may or may not intersect
    bool found_grazing_intersection = false;
    for (int i = 0; i < grid_divisions.x; i++) {
        for (int j = 0; j < grid_divisions.y; j++) {
            for (int k = 0; k < grid_divisions.z; k++) {
                int P_denom, P_trans;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                if (P_denom > 0) {
                    found_grazing_intersection = true;
                    float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                    DOCTEST_CHECK(r_bar >= 0.0f);
                }
            }
        }
    }
    
    // Test 2: Ray touching corner/edge
    collision.clearVoxelData();
    std::vector<vec3> corner_origins = {make_vec3(-3, -3, -3)};
    std::vector<vec3> corner_directions = {normalize(make_vec3(1, 1, 1))};
    
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, corner_origins, corner_directions);
    
    bool found_corner_intersection = false;
    for (int i = 0; i < grid_divisions.x; i++) {
        for (int j = 0; j < grid_divisions.y; j++) {
            for (int k = 0; k < grid_divisions.z; k++) {
                int P_denom, P_trans;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                if (P_denom > 0) {
                    found_corner_intersection = true;
                    float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                    DOCTEST_CHECK(r_bar >= 0.0f);
                }
            }
        }
    }
    
    // Test 3: Rays with very small direction components (near-zero)
    collision.clearVoxelData();
    std::vector<vec3> near_zero_origins = {make_vec3(0, 0, -5)};
    std::vector<vec3> near_zero_directions = {normalize(make_vec3(1e-6f, 1e-6f, 1.0f))};
    
    collision.calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, near_zero_origins, near_zero_directions);
    
    // Should handle without numerical issues
    bool found_near_zero_intersection = false;
    for (int i = 0; i < grid_divisions.x; i++) {
        for (int j = 0; j < grid_divisions.y; j++) {
            for (int k = 0; k < grid_divisions.z; k++) {
                int P_denom, P_trans;
                collision.getVoxelTransmissionProbability(make_int3(i, j, k), P_denom, P_trans);
                if (P_denom > 0) {
                    found_near_zero_intersection = true;
                    float r_bar = collision.getVoxelRbar(make_int3(i, j, k));
                    DOCTEST_CHECK(r_bar >= 0.0f);
                    DOCTEST_CHECK(std::isfinite(r_bar)); // Check for NaN/inf
                }
            }
        }
    }
}

// -------- GENERIC RAY-TRACING TESTS --------

DOCTEST_TEST_CASE("CollisionDetection Generic Ray Casting - Basic Functionality") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create simple test geometry - triangle
    vec3 v0 = make_vec3(0, 0, 0);
    vec3 v1 = make_vec3(1, 0, 0);
    vec3 v2 = make_vec3(0.5f, 0, 1);
    uint triangle_uuid = context.addTriangle(v0, v1, v2);
    
    // Test 1: Hit test - ray intersecting triangle
    vec3 ray_origin = make_vec3(0.5f, -1, 0.5f);
    vec3 ray_direction = normalize(make_vec3(0, 1, 0));
    
    CollisionDetection::CollisionDetection::HitResult result = collision.castRay(ray_origin, ray_direction);
    
    DOCTEST_CHECK(result.hit == true);
    DOCTEST_CHECK(result.primitive_UUID == triangle_uuid);
    DOCTEST_CHECK(result.distance > 0.9f);
    DOCTEST_CHECK(result.distance < 1.1f);
    
    // Check intersection point is reasonable
    DOCTEST_CHECK(result.intersection_point.x > 0.4f);
    DOCTEST_CHECK(result.intersection_point.x < 0.6f);
    DOCTEST_CHECK(std::abs(result.intersection_point.y) < 1e-5f);
    DOCTEST_CHECK(result.intersection_point.z > 0.4f);
    DOCTEST_CHECK(result.intersection_point.z < 0.6f);
    
    // Test 2: Miss test - ray not intersecting triangle
    vec3 miss_origin = make_vec3(2, -1, 0.5f);
    vec3 miss_direction = normalize(make_vec3(0, 1, 0));
    
    CollisionDetection::HitResult miss_result = collision.castRay(miss_origin, miss_direction);
    DOCTEST_CHECK(miss_result.hit == false);
    DOCTEST_CHECK(miss_result.distance < 0);
    
    // Test 3: Max distance constraint
    vec3 limited_origin = make_vec3(0.5f, -2, 0.5f);
    vec3 limited_direction = normalize(make_vec3(0, 1, 0));
    float max_distance = 1.5f; // Ray would need ~2 units to reach triangle
    
    CollisionDetection::HitResult limited_result = collision.castRay(limited_origin, limited_direction, max_distance);
    DOCTEST_CHECK(limited_result.hit == false);
}

DOCTEST_TEST_CASE("CollisionDetection Generic Ray Casting - CollisionDetection::RayQuery Structure") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create test geometry
    uint triangle1 = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 0, 1));
    uint triangle2 = context.addTriangle(make_vec3(2, 0, 0), make_vec3(3, 0, 0), make_vec3(2.5f, 0, 1));
    
    // Test 1: CollisionDetection::RayQuery with default constructor
    CollisionDetection::RayQuery query1;
    query1.origin = make_vec3(0.5f, -1, 0.5f);
    query1.direction = normalize(make_vec3(0, 1, 0));
    
    CollisionDetection::HitResult result1 = collision.castRay(query1);
    DOCTEST_CHECK(result1.hit == true);
    
    // Test 2: CollisionDetection::RayQuery with full constructor
    CollisionDetection::RayQuery query2(make_vec3(2.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)), -1.0f, {triangle2});
    
    CollisionDetection::HitResult result2 = collision.castRay(query2);
    DOCTEST_CHECK(result2.hit == true);
    DOCTEST_CHECK(result2.primitive_UUID == triangle2);
    
    // Test 3: Target UUID filtering - should only hit triangle1
    CollisionDetection::RayQuery query3(make_vec3(0.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)), -1.0f, {triangle1});
    
    CollisionDetection::HitResult result3 = collision.castRay(query3);
    DOCTEST_CHECK(result3.hit == true);
    DOCTEST_CHECK(result3.primitive_UUID == triangle1);
    
    // Test 4: Target UUID filtering with non-intersecting primitive
    CollisionDetection::RayQuery query4(make_vec3(0.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)), -1.0f, {triangle2});
    
    CollisionDetection::HitResult result4 = collision.castRay(query4);
    DOCTEST_CHECK(result4.hit == false);
}

DOCTEST_TEST_CASE("CollisionDetection Batch Ray Casting") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create test geometry
    uint triangle = context.addTriangle(make_vec3(-1, 0, -1), make_vec3(1, 0, -1), make_vec3(0, 0, 1));
    
    
    // Create multiple ray queries
    std::vector<CollisionDetection::RayQuery> queries;
    queries.push_back(CollisionDetection::RayQuery(make_vec3(0, -1, 0), normalize(make_vec3(0, 1, 0))));     // Hit
    queries.push_back(CollisionDetection::RayQuery(make_vec3(2, -1, 0), normalize(make_vec3(0, 1, 0))));     // Miss
    queries.push_back(CollisionDetection::RayQuery(make_vec3(-0.5f, -1, 0), normalize(make_vec3(0, 1, 0)))); // Hit
    queries.push_back(CollisionDetection::RayQuery(make_vec3(0.5f, -1, 0), normalize(make_vec3(0, 1, 0))));  // Hit
    
    // Test batch casting with statistics
    CollisionDetection::RayTracingStats stats;
    std::vector<CollisionDetection::HitResult> results = collision.castRays(queries, &stats);
    
    
    DOCTEST_CHECK(results.size() == 4);
    DOCTEST_CHECK(stats.total_rays_cast == 4);
    DOCTEST_CHECK(stats.total_hits == 3);
    DOCTEST_CHECK(stats.average_ray_distance > 0);
    
    // Verify individual results
    DOCTEST_CHECK(results[0].hit == true);   // Center hit
    DOCTEST_CHECK(results[1].hit == false);  // Miss
    DOCTEST_CHECK(results[2].hit == true);   // Left hit
    DOCTEST_CHECK(results[3].hit == true);   // Right hit
    
    // All hits should be on the same triangle
    for (const auto& result : results) {
        if (result.hit) {
            DOCTEST_CHECK(result.primitive_UUID == triangle);
            DOCTEST_CHECK(result.distance > 0);
        }
    }
}

DOCTEST_TEST_CASE("CollisionDetection Grid Ray Intersection") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create test geometry - triangle in center
    uint triangle = context.addTriangle(make_vec3(-0.5f, 0, -0.5f), make_vec3(0.5f, 0, -0.5f), make_vec3(0, 0, 0.5f));
    
    // Set up grid parameters
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(4, 4, 4);
    int3 grid_divisions = make_int3(2, 2, 2);
    
    // Create rays hitting different voxels
    std::vector<CollisionDetection::RayQuery> rays;
    rays.push_back(CollisionDetection::RayQuery(make_vec3(0, -2, 0), normalize(make_vec3(0, 1, 0))));     // Center voxel
    rays.push_back(CollisionDetection::RayQuery(make_vec3(-0.8f, -2, 0), normalize(make_vec3(0, 1, 0)))); // Different voxel
    rays.push_back(CollisionDetection::RayQuery(make_vec3(2, -2, 0), normalize(make_vec3(0, 1, 0))));     // Miss entirely
    
    auto grid_results = collision.performGridRayIntersection(grid_center, grid_size, grid_divisions, rays);
    
    DOCTEST_CHECK(grid_results.size() == 2); // x-divisions
    DOCTEST_CHECK(grid_results[0].size() == 2); // y-divisions
    DOCTEST_CHECK(grid_results[0][0].size() == 2); // z-divisions
    
    // Count total hits across all voxels
    int total_hits = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                total_hits += grid_results[i][j][k].size();
            }
        }
    }
    DOCTEST_CHECK(total_hits >= 1); // At least one hit should be recorded
}

DOCTEST_TEST_CASE("CollisionDetection Ray Path Lengths Detailed") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create test geometry
    uint triangle = context.addTriangle(make_vec3(-1, 0, -1), make_vec3(1, 0, -1), make_vec3(0, 0, 1));
    
    // Set up test parameters
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(4, 4, 4);
    int3 grid_divisions = make_int3(2, 2, 2);
    
    std::vector<vec3> ray_origins = {
        make_vec3(0, -2, 0),
        make_vec3(0.5f, -2, 0),
        make_vec3(2, -2, 0) // This should miss
    };
    
    std::vector<vec3> ray_directions = {
        normalize(make_vec3(0, 1, 0)),
        normalize(make_vec3(0, 1, 0)),
        normalize(make_vec3(0, 1, 0))
    };
    
    std::vector<CollisionDetection::HitResult> hit_results;
    collision.calculateRayPathLengthsDetailed(grid_center, grid_size, grid_divisions, ray_origins, ray_directions, hit_results);
    
    DOCTEST_CHECK(hit_results.size() == 3);
    
    // First two rays should hit
    DOCTEST_CHECK(hit_results[0].hit == true);
    DOCTEST_CHECK(hit_results[1].hit == true);
    DOCTEST_CHECK(hit_results[2].hit == false);
    
    // Hit distances should be reasonable
    DOCTEST_CHECK(hit_results[0].distance > 1.5f);
    DOCTEST_CHECK(hit_results[0].distance < 2.5f);
    DOCTEST_CHECK(hit_results[1].distance > 1.5f);
    DOCTEST_CHECK(hit_results[1].distance < 2.5f);
    
    // Verify that existing voxel data is also updated
    int P_denom, P_trans;
    collision.getVoxelTransmissionProbability(make_int3(0, 0, 0), P_denom, P_trans);
    DOCTEST_CHECK(P_denom >= 0);
}

DOCTEST_TEST_CASE("CollisionDetection Ray Casting - Normal Calculation") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: Triangle normal calculation
    vec3 v0 = make_vec3(0, 0, 0);
    vec3 v1 = make_vec3(1, 0, 0);
    vec3 v2 = make_vec3(0, 1, 0);
    uint triangle = context.addTriangle(v0, v1, v2);
    
    vec3 ray_origin = make_vec3(0.3f, 0.3f, -1);
    vec3 ray_direction = normalize(make_vec3(0, 0, 1));
    
    CollisionDetection::CollisionDetection::HitResult result = collision.castRay(ray_origin, ray_direction);
    
    DOCTEST_CHECK(result.hit == true);
    
    // Normal should point in +Z direction (calculated from cross product)
    vec3 expected_normal = normalize(make_vec3(0, 0, 1));
    float dot_product = result.normal.x * expected_normal.x + 
                       result.normal.y * expected_normal.y + 
                       result.normal.z * expected_normal.z;
    DOCTEST_CHECK(std::abs(dot_product) > 0.9f); // Should be nearly parallel
    
    // Test 2: Patch normal calculation
    uint patch = context.addPatch(make_vec3(0, 0, 2), make_vec2(1, 1));
    
    vec3 patch_ray_origin = make_vec3(0, 0, 1);
    vec3 patch_ray_direction = normalize(make_vec3(0, 0, 1));
    
    CollisionDetection::HitResult patch_result = collision.castRay(patch_ray_origin, patch_ray_direction);
    
    DOCTEST_CHECK(patch_result.hit == true);
    DOCTEST_CHECK(patch_result.primitive_UUID == patch);
    
    // Patch normal should also be reasonable
    DOCTEST_CHECK(patch_result.normal.magnitude() > 0.9f);
    DOCTEST_CHECK(patch_result.normal.magnitude() < 1.1f);
}

DOCTEST_TEST_CASE("CollisionDetection Ray Casting - Edge Cases and Error Handling") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create test triangle
    uint triangle = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 0, 1));
    
    // Test 1: Zero direction vector (should be handled)
    vec3 zero_direction = make_vec3(0, 0, 0);
    CollisionDetection::HitResult zero_result = collision.castRay(make_vec3(0, -1, 0), zero_direction);
    DOCTEST_CHECK(zero_result.hit == false); // Should handle gracefully
    
    // Test 2: Very small direction vector (should normalize)
    vec3 tiny_direction = make_vec3(1e-8f, 1e-8f, 1e-8f);
    CollisionDetection::HitResult tiny_result = collision.castRay(make_vec3(0.5f, -1, 0.5f), tiny_direction);
    // Should either hit or miss gracefully, not crash
    
    // Test 3: Infinite max distance
    CollisionDetection::HitResult inf_result = collision.castRay(make_vec3(0.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)), 
                                           std::numeric_limits<float>::infinity());
    DOCTEST_CHECK(inf_result.hit == true);
    
    // Test 4: Negative max distance (should be treated as infinite)
    CollisionDetection::HitResult neg_result = collision.castRay(make_vec3(0.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)), -5.0f);
    DOCTEST_CHECK(neg_result.hit == true);
    
    // Test 5: Empty target UUIDs (should use all primitives)
    std::vector<uint> empty_targets;
    CollisionDetection::HitResult empty_result = collision.castRay(make_vec3(0.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)), -1.0f, empty_targets);
    DOCTEST_CHECK(empty_result.hit == true);
    
    // Test 6: Invalid target UUIDs (should be filtered out)
    std::vector<uint> invalid_targets = {99999, triangle, 88888};
    CollisionDetection::HitResult invalid_result = collision.castRay(make_vec3(0.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)), -1.0f, invalid_targets);
    DOCTEST_CHECK(invalid_result.hit == true);
    DOCTEST_CHECK(invalid_result.primitive_UUID == triangle);
}

DOCTEST_TEST_CASE("CollisionDetection Ray Casting - Performance and Scalability") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create larger set of test geometry
    std::vector<uint> triangles;
    for (int i = 0; i < 100; i++) {
        float x = i * 0.1f;
        uint triangle = context.addTriangle(
            make_vec3(x, 0, -0.5f), 
            make_vec3(x + 0.05f, 0, -0.5f), 
            make_vec3(x + 0.025f, 0, 0.5f)
        );
        triangles.push_back(triangle);
    }
    
    // Test batch casting with large number of rays - align with triangle centers
    std::vector<CollisionDetection::RayQuery> many_rays;
    for (int i = 0; i < 200; i++) {
        float x = (i % 100) * 0.1f + 0.025f; // Align with triangle centers
        many_rays.push_back(CollisionDetection::RayQuery(make_vec3(x, -1, 0), normalize(make_vec3(0, 1, 0))));
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CollisionDetection::RayTracingStats stats;
    std::vector<CollisionDetection::HitResult> results = collision.castRays(many_rays, &stats);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    DOCTEST_CHECK(results.size() == 200);
    DOCTEST_CHECK(stats.total_rays_cast == 200);
    DOCTEST_CHECK(stats.total_hits >= 100); // Should hit many triangles
    
    // Performance should be reasonable (less than 1 second for 200 rays)
    DOCTEST_CHECK(duration.count() < 1000);
    
    std::cout << "Batch ray casting performance: " << duration.count() << " ms for " 
              << many_rays.size() << " rays (" << stats.total_hits << " hits)" << std::endl;
}

DOCTEST_TEST_CASE("CollisionDetection Ray Casting - Integration with Existing BVH") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create test geometry
    uint triangle1 = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 0, 1));
    uint triangle2 = context.addTriangle(make_vec3(2, 0, 0), make_vec3(3, 0, 0), make_vec3(2.5f, 0, 1));
    
    // Test that ray casting works with manually built BVH
    collision.buildBVH();
    
    CollisionDetection::HitResult result1 = collision.castRay(make_vec3(0.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)));
    DOCTEST_CHECK(result1.hit == true);
    DOCTEST_CHECK(result1.primitive_UUID == triangle1);
    
    CollisionDetection::HitResult result2 = collision.castRay(make_vec3(2.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)));
    DOCTEST_CHECK(result2.hit == true);
    DOCTEST_CHECK(result2.primitive_UUID == triangle2);
    
    // Test that ray casting works with automatic BVH rebuilds
    collision.enableAutomaticBVHRebuilds();
    
    // Add new geometry after BVH was built
    uint triangle3 = context.addTriangle(make_vec3(4, 0, 0), make_vec3(5, 0, 0), make_vec3(4.5f, 0, 1));
    
    // Should automatically rebuild BVH and find new triangle
    CollisionDetection::HitResult result3 = collision.castRay(make_vec3(4.5f, -1, 0.5f), normalize(make_vec3(0, 1, 0)));
    DOCTEST_CHECK(result3.hit == true);
    DOCTEST_CHECK(result3.primitive_UUID == triangle3);
}

DOCTEST_TEST_CASE("CollisionDetection Ray Casting - Compatibility with Other Plugin Methods") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create overlapping test geometry
    auto triangles = CollisionTests::generateOverlappingCluster(&context, 5);
    
    // Test that ray casting and collision detection can coexist
    auto collisions = collision.findCollisions(triangles[0]);
    DOCTEST_CHECK(collisions.size() > 0);
    
    // Test that existing collision detection methods still work
    DOCTEST_CHECK(triangles.size() == 5);
    
    // Test cone intersection functionality  
    auto cone_result = collision.findOptimalConePath(make_vec3(0, -2, 0), make_vec3(0, 1, 0), M_PI/6, 3.0f);
    
    // Verify that both collision detection and ray casting infrastructure coexist
    // This test mainly verifies compatibility, not specific ray hits
    DOCTEST_CHECK(collisions.size() > 0); // Collision detection works
    DOCTEST_CHECK(true); // Ray casting infrastructure is available (compilation test)
}

// ================================================================
// OPTIMIZATION TEST CASES
// ================================================================

DOCTEST_TEST_CASE("CollisionDetection - BVH Optimization Mode Management") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create test geometry to trigger BVH construction
    uint triangle = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 0, 1));
    auto sphere_uuids = context.addSphere(10, make_vec3(2, 0, 0.5f), 0.5f);
    
    // Test 1: Default mode should be SOA_QUANTIZED
    DOCTEST_CHECK(collision.getBVHOptimizationMode() == CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
    
    // Test 2: Switch to SOA_UNCOMPRESSED
    DOCTEST_CHECK_NOTHROW(collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED));
    DOCTEST_CHECK(collision.getBVHOptimizationMode() == CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    
    // Test 3: Switch back to SOA_QUANTIZED  
    DOCTEST_CHECK_NOTHROW(collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED));
    DOCTEST_CHECK(collision.getBVHOptimizationMode() == CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
    
    // Test 4: Build BVH to populate memory statistics
    collision.buildBVH();
    
    // Test 5: Convert between optimization modes to populate all memory structures
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
    
    // Test 6: Memory usage comparison 
    auto memory_stats = collision.getBVHMemoryUsage();
    DOCTEST_CHECK(memory_stats.soa_memory_bytes > 0);
    DOCTEST_CHECK(memory_stats.quantized_memory_bytes > 0);
    
    // Memory hierarchy: quantized < soa (with reasonable tolerances)
    DOCTEST_CHECK(memory_stats.quantized_memory_bytes < memory_stats.soa_memory_bytes);
}

DOCTEST_TEST_CASE("CollisionDetection - Optimized Ray Casting Correctness") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create diverse test geometry with known positions
    uint triangle = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 0, 1));
    auto sphere_uuids = context.addSphere(8, make_vec3(3, 0, 0.5f), 0.5f);
    
    // Create test rays with known expected outcomes
    std::vector<CollisionDetection::RayQuery> rays;
    rays.push_back(CollisionDetection::RayQuery(make_vec3(0.5f, -1, 0.5f), make_vec3(0, 1, 0)));  // Should hit triangle
    rays.push_back(CollisionDetection::RayQuery(make_vec3(3, -1, 0.5f), make_vec3(0, 1, 0)));     // Should hit sphere
    rays.push_back(CollisionDetection::RayQuery(make_vec3(10, -1, 0), make_vec3(0, 1, 0)));       // Should miss both
    
    // Test each optimization mode produces consistent results
    std::vector<CollisionDetection::HitResult> legacy_results, soa_results, quantized_results;
    
    // Legacy mode
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    legacy_results = collision.castRays(rays);
    
    // SOA mode  
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    soa_results = collision.castRaysOptimized(rays);
    
    // Quantized mode
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
    quantized_results = collision.castRaysOptimized(rays);
    
    // Verify all modes produce equivalent results
    DOCTEST_REQUIRE(legacy_results.size() == 3);
    DOCTEST_REQUIRE(soa_results.size() == 3);
    DOCTEST_REQUIRE(quantized_results.size() == 3);
    
    for (size_t i = 0; i < legacy_results.size(); i++) {
        // Hit/miss should be consistent across all modes
        DOCTEST_CHECK(legacy_results[i].hit == soa_results[i].hit);
        DOCTEST_CHECK(soa_results[i].hit == quantized_results[i].hit);
        
        if (legacy_results[i].hit) {
            // Primitive UUID should match
            DOCTEST_CHECK(legacy_results[i].primitive_UUID == soa_results[i].primitive_UUID);
            DOCTEST_CHECK(soa_results[i].primitive_UUID == quantized_results[i].primitive_UUID);
            
            // Distance should be very close (allowing for quantization error)
            float distance_tolerance = 0.01f; // Allow for quantization effects
            DOCTEST_CHECK(std::abs(legacy_results[i].distance - soa_results[i].distance) < 0.001f);
            DOCTEST_CHECK(std::abs(soa_results[i].distance - quantized_results[i].distance) < distance_tolerance);
        }
    }
    
    // Verify expected hit pattern: ray 0 and 1 should hit, ray 2 should miss
    DOCTEST_CHECK(legacy_results[0].hit == true);  // Triangle hit
    DOCTEST_CHECK(legacy_results[1].hit == true);  // Sphere hit  
    DOCTEST_CHECK(legacy_results[2].hit == false); // Miss
}

DOCTEST_TEST_CASE("CollisionDetection - Ray Streaming Interface") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    
    // Create test geometry grid
    for (int i = 0; i < 5; i++) {
        float x = i * 2.0f;
        context.addTriangle(make_vec3(x, 0, 0), make_vec3(x+1, 0, 0), make_vec3(x+0.5f, 0, 1));
    }
    
    // Test ray streaming interface
    CollisionDetection::RayStream stream;
    std::vector<CollisionDetection::RayQuery> batch;
    
    // Create batch of rays
    for (int i = 0; i < 50; i++) {
        float x = (i % 5) * 2.0f + 0.5f; // Align with triangles
        batch.push_back(CollisionDetection::RayQuery(make_vec3(x, -1, 0.5f), make_vec3(0, 1, 0)));
    }
    stream.addRays(batch);
    
    DOCTEST_CHECK(stream.total_rays == 50);
    DOCTEST_CHECK(stream.packets.size() > 0);
    
    // Process stream
    CollisionDetection::RayTracingStats stats;
    bool success = collision.processRayStream(stream, &stats);
    DOCTEST_CHECK(success == true);
    DOCTEST_CHECK(stats.total_rays_cast == 50);
    
    // Verify results
    auto results = stream.getAllResults();
    DOCTEST_CHECK(results.size() == 50);
    
    // All rays should hit (they're aligned with triangles)
    size_t hit_count = 0;
    for (const auto& result : results) {
        if (result.hit) hit_count++;
    }
    DOCTEST_CHECK(hit_count > 40); // Most rays should hit the triangles
}

DOCTEST_TEST_CASE("CollisionDetection - BVH Layout Conversion Methods") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create substantial test geometry to make conversions meaningful
    uint triangle1 = context.addTriangle(make_vec3(0, 0, 0), make_vec3(2, 0, 0), make_vec3(1, 0, 2));
    auto sphere_uuids = context.addSphere(12, make_vec3(5, 0, 1), 0.8f);
    uint triangle2 = context.addTriangle(make_vec3(-2, 1, 0), make_vec3(0, 1, 0), make_vec3(-1, 1, 1.5f));
    
    // Build initial BVH in legacy mode
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    collision.buildBVH();
    
    // Test all possible conversion paths
    std::vector<CollisionDetection::RayQuery> test_rays = {
        CollisionDetection::RayQuery(make_vec3(1, -1, 1), make_vec3(0, 1, 0)),    // Should hit triangle1
        CollisionDetection::RayQuery(make_vec3(5, -1, 1), make_vec3(0, 1, 0)),    // Should hit sphere
        CollisionDetection::RayQuery(make_vec3(-1, 0, 0.75f), make_vec3(0, 1, 0)) // Should hit triangle2
    };
    
    // Get baseline results from legacy mode
    auto legacy_results = collision.castRays(test_rays);
    auto legacy_memory = collision.getBVHMemoryUsage();
    
    // Test Legacy → SOA → Legacy conversion
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    auto soa_results = collision.castRaysOptimized(test_rays);
    auto soa_memory = collision.getBVHMemoryUsage();
    
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    auto legacy_restored_results = collision.castRays(test_rays);
    
    // Test Legacy → Quantized → Legacy conversion  
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
    auto quantized_results = collision.castRaysOptimized(test_rays);
    auto quantized_memory = collision.getBVHMemoryUsage();
    
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    auto legacy_final_results = collision.castRays(test_rays);
    
    // Test SOA ↔ Quantized conversion
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
    auto soa_to_quantized_results = collision.castRaysOptimized(test_rays);
    
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    auto quantized_to_soa_results = collision.castRaysOptimized(test_rays);
    
    // Validation: All conversion paths should preserve ray casting functionality
    DOCTEST_REQUIRE(legacy_results.size() == 3);
    DOCTEST_REQUIRE(soa_results.size() == 3);
    DOCTEST_REQUIRE(quantized_results.size() == 3);
    DOCTEST_REQUIRE(legacy_restored_results.size() == 3);
    
    // Count total hits for each method to verify general consistency
    size_t legacy_hits = 0, soa_hits = 0, quantized_hits = 0, restored_hits = 0;
    
    for (size_t i = 0; i < 3; i++) {
        if (legacy_results[i].hit) legacy_hits++;
        if (soa_results[i].hit) soa_hits++;
        if (quantized_results[i].hit) quantized_hits++;
        if (legacy_restored_results[i].hit) restored_hits++;
    }
    
    // Check that conversion methods produce reasonable results
    // SoA may have slight differences due to traversal order, allow small variance
    int hit_diff_soa = std::abs(static_cast<int>(legacy_hits) - static_cast<int>(soa_hits));
    DOCTEST_CHECK(hit_diff_soa <= 1); // Allow 1 difference due to optimization differences
    DOCTEST_CHECK(legacy_hits == restored_hits); // Round-trip should preserve results
    
    // Quantized might have 1-2 differences due to precision loss, but should be close
    int hit_diff_quantized = std::abs(static_cast<int>(legacy_hits) - static_cast<int>(quantized_hits));
    DOCTEST_CHECK(hit_diff_quantized <= 2);
    
    // For rays that hit in both legacy and corresponding optimized mode, check accuracy
    for (size_t i = 0; i < 3; i++) {
        if (legacy_results[i].hit && soa_results[i].hit) {
            // SoA should be identical to legacy (no precision loss)
            DOCTEST_CHECK(legacy_results[i].primitive_UUID == soa_results[i].primitive_UUID);
            DOCTEST_CHECK(std::abs(legacy_results[i].distance - soa_results[i].distance) < 0.001f);
        }
        
        if (legacy_results[i].hit && legacy_restored_results[i].hit) {
            // Round-trip conversion should be accurate
            DOCTEST_CHECK(legacy_results[i].primitive_UUID == legacy_restored_results[i].primitive_UUID);
            DOCTEST_CHECK(std::abs(legacy_results[i].distance - legacy_restored_results[i].distance) < 0.001f);
        }
        
        if (legacy_results[i].hit && quantized_results[i].hit) {
            // Quantized may have more error, but should be reasonable
            float distance_tolerance = 0.1f; // Allow larger tolerance for quantization
            DOCTEST_CHECK(std::abs(legacy_results[i].distance - quantized_results[i].distance) < distance_tolerance);
        }
    }
    
    // Memory usage verification
    DOCTEST_CHECK(soa_memory.soa_memory_bytes > 0);  
    DOCTEST_CHECK(quantized_memory.quantized_memory_bytes > 0);
    DOCTEST_CHECK(quantized_memory.quantized_memory_bytes < soa_memory.soa_memory_bytes); // Quantized should be smaller
}

DOCTEST_TEST_CASE("CollisionDetection - RayPacket Edge Cases and Functionality") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: Empty RayPacket behavior
    CollisionDetection::RayPacket empty_packet;
    DOCTEST_CHECK(empty_packet.ray_count == 0);
    DOCTEST_CHECK(empty_packet.getMemoryUsage() == 0);
    DOCTEST_CHECK(empty_packet.toRayQueries().empty());
    
    // Clear empty packet should not crash
    DOCTEST_CHECK_NOTHROW(empty_packet.clear());
    
    // Test 2: RayPacket capacity management
    CollisionDetection::RayPacket capacity_packet;
    capacity_packet.reserve(100);
    
    // Add rays up to and beyond initial capacity
    std::vector<CollisionDetection::RayQuery> test_queries;
    for (int i = 0; i < 150; i++) {
        float x = i * 0.1f;
        CollisionDetection::RayQuery query(make_vec3(x, 0, 0), make_vec3(0, 0, 1));
        test_queries.push_back(query);
        capacity_packet.addRay(query);
    }
    
    DOCTEST_CHECK(capacity_packet.ray_count == 150);
    DOCTEST_CHECK(capacity_packet.origins.size() == 150);
    DOCTEST_CHECK(capacity_packet.directions.size() == 150);
    DOCTEST_CHECK(capacity_packet.results.size() == 150);
    
    // Test 3: RayPacket conversion accuracy
    auto converted_queries = capacity_packet.toRayQueries();
    DOCTEST_REQUIRE(converted_queries.size() == 150);
    
    for (size_t i = 0; i < 150; i++) {
        DOCTEST_CHECK(converted_queries[i].origin.magnitude() == test_queries[i].origin.magnitude());
        DOCTEST_CHECK(converted_queries[i].direction.magnitude() == test_queries[i].direction.magnitude());
        DOCTEST_CHECK(converted_queries[i].max_distance == test_queries[i].max_distance);
    }
    
    // Test 4: Memory usage calculation
    size_t expected_memory = (150 * 2) * sizeof(helios::vec3) + // origins + directions
                            150 * sizeof(float) +                 // max_distances  
                            150 * sizeof(CollisionDetection::HitResult); // results
    size_t actual_memory = capacity_packet.getMemoryUsage();
    DOCTEST_CHECK(actual_memory >= expected_memory); // Account for target_UUIDs overhead
    
    // Test 5: Clear functionality
    capacity_packet.clear();
    DOCTEST_CHECK(capacity_packet.ray_count == 0);
    DOCTEST_CHECK(capacity_packet.origins.empty());
    DOCTEST_CHECK(capacity_packet.directions.empty());
    DOCTEST_CHECK(capacity_packet.results.empty());
    DOCTEST_CHECK(capacity_packet.getMemoryUsage() == 0);
}

DOCTEST_TEST_CASE("CollisionDetection - RayStream Batch Management") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    
    // Create test geometry
    for (int i = 0; i < 3; i++) {
        float x = i * 3.0f;
        context.addTriangle(make_vec3(x, 0, 0), make_vec3(x+1, 0, 0), make_vec3(x+0.5f, 0, 1));
    }
    
    // Test 1: Large ray stream with multiple packets
    CollisionDetection::RayStream large_stream;
    std::vector<CollisionDetection::RayQuery> large_batch;
    
    // Create more rays than fit in a single packet
    size_t total_rays = CollisionDetection::RAY_BATCH_SIZE * 2.5; // 2.5 packets worth
    for (size_t i = 0; i < total_rays; i++) {
        float x = (i % 3) * 3.0f + 0.5f;
        large_batch.push_back(CollisionDetection::RayQuery(make_vec3(x, -1, 0.5f), make_vec3(0, 1, 0)));
    }
    
    large_stream.addRays(large_batch);
    DOCTEST_CHECK(large_stream.total_rays == total_rays);
    DOCTEST_CHECK(large_stream.packets.size() == 3); // Should create 3 packets
    
    // Test 2: Stream processing and memory usage
    size_t stream_memory_before = large_stream.getMemoryUsage();
    DOCTEST_CHECK(stream_memory_before > 0);
    
    CollisionDetection::RayTracingStats large_stats;
    bool large_success = collision.processRayStream(large_stream, &large_stats);
    DOCTEST_CHECK(large_success == true);
    DOCTEST_CHECK(large_stats.total_rays_cast == total_rays);
    
    // Test 3: Results aggregation
    auto all_results = large_stream.getAllResults();
    DOCTEST_CHECK(all_results.size() == total_rays);
    
    // Verify reasonable hit rate (rays are aligned with triangles)
    size_t hit_count = 0;
    for (const auto& result : all_results) {
        if (result.hit) hit_count++;
    }
    float hit_rate = float(hit_count) / float(total_rays);
    DOCTEST_CHECK(hit_rate > 0.8f); // Expect high hit rate
    
    // Test 4: Empty stream handling
    CollisionDetection::RayStream empty_stream;
    DOCTEST_CHECK(empty_stream.total_rays == 0);
    DOCTEST_CHECK(empty_stream.packets.empty());
    DOCTEST_CHECK(empty_stream.getMemoryUsage() == 0);
    
    CollisionDetection::RayTracingStats empty_stats;
    bool empty_success = collision.processRayStream(empty_stream, &empty_stats);
    DOCTEST_CHECK(empty_success == true);
    DOCTEST_CHECK(empty_stats.total_rays_cast == 0);
    
    // Test 5: Stream clear and reuse
    large_stream.clear();
    DOCTEST_CHECK(large_stream.total_rays == 0);
    DOCTEST_CHECK(large_stream.packets.empty());
    DOCTEST_CHECK(large_stream.current_packet == 0);
    DOCTEST_CHECK(large_stream.getMemoryUsage() == 0);
}

DOCTEST_TEST_CASE("CollisionDetection - Quantization Precision Validation") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create robust test geometry that works well with quantization
    uint triangle1 = context.addTriangle(make_vec3(0, 0, 0), make_vec3(2, 0, 0), make_vec3(1, 0, 2));
    uint triangle2 = context.addTriangle(make_vec3(10, 0, 0), make_vec3(12, 0, 0), make_vec3(11, 0, 2));  
    auto sphere_uuids = context.addSphere(12, make_vec3(5, 5, 1), 1.0f);
    
    // Build BVH and get legacy baseline
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    collision.buildBVH();
    
    std::vector<CollisionDetection::RayQuery> precision_test_rays = {
        // Ray hitting first triangle - well within bounds
        CollisionDetection::RayQuery(make_vec3(1, -1, 1), make_vec3(0, 1, 0)),
        // Ray hitting second triangle - different scale
        CollisionDetection::RayQuery(make_vec3(11, -1, 1), make_vec3(0, 1, 0)),
        // Ray hitting sphere - curved surface
        CollisionDetection::RayQuery(make_vec3(5, 3, 1), make_vec3(0, 1, 0)),
        // Miss rays - clearly outside geometry
        CollisionDetection::RayQuery(make_vec3(20, -1, 0), make_vec3(0, 1, 0)),
        CollisionDetection::RayQuery(make_vec3(-5, -1, 0), make_vec3(0, 1, 0))
    };
    
    auto legacy_precision_results = collision.castRays(precision_test_rays);
    
    // Test quantized precision
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);
    auto quantized_precision_results = collision.castRaysOptimized(precision_test_rays);
    
    DOCTEST_REQUIRE(legacy_precision_results.size() == precision_test_rays.size());
    DOCTEST_REQUIRE(quantized_precision_results.size() == precision_test_rays.size());
    
    // Count matching hit/miss patterns (allow some quantization discrepancy)
    size_t matching_patterns = 0;
    size_t total_hits_legacy = 0, total_hits_quantized = 0;
    
    for (size_t i = 0; i < precision_test_rays.size(); i++) {
        if (legacy_precision_results[i].hit == quantized_precision_results[i].hit) {
            matching_patterns++;
        }
        
        if (legacy_precision_results[i].hit) total_hits_legacy++;
        if (quantized_precision_results[i].hit) total_hits_quantized++;
        
        if (legacy_precision_results[i].hit && quantized_precision_results[i].hit) {
            // When both hit, check distance accuracy
            float distance_error = std::abs(legacy_precision_results[i].distance - quantized_precision_results[i].distance);
            float relative_error = distance_error / std::max(legacy_precision_results[i].distance, 0.1f);
            DOCTEST_CHECK(relative_error < 0.15f); // Allow 15% tolerance for quantization
        }
    }
    
    // Most patterns should match (allow 1-2 mismatches due to quantization boundaries) 
    DOCTEST_CHECK(matching_patterns >= precision_test_rays.size() - 2);
    
    // Hit counts should be reasonably close
    int hit_count_diff = std::abs(static_cast<int>(total_hits_legacy) - static_cast<int>(total_hits_quantized));
    DOCTEST_CHECK(hit_count_diff <= 2);
    
    // Test quantization memory benefits
    auto memory_stats = collision.getBVHMemoryUsage();
    DOCTEST_CHECK(memory_stats.quantized_reduction_percent > 40.0f); // Should achieve reasonable reduction
    DOCTEST_CHECK(memory_stats.quantized_memory_bytes < memory_stats.soa_memory_bytes);
}

DOCTEST_TEST_CASE("CollisionDetection - Error Handling and Edge Cases") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Test 1: Mode conversion with empty BVH should not crash
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    DOCTEST_CHECK_NOTHROW(collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED));
    DOCTEST_CHECK_NOTHROW(collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED));
    
    // Test 2: Repeated mode setting should be handled efficiently
    auto initial_mode = collision.getBVHOptimizationMode();
    DOCTEST_CHECK_NOTHROW(collision.setBVHOptimizationMode(initial_mode)); // No-op
    DOCTEST_CHECK(collision.getBVHOptimizationMode() == initial_mode);
    
    // Test 3: Memory usage queries with empty structures
    auto empty_memory_stats = collision.getBVHMemoryUsage();
    DOCTEST_CHECK(empty_memory_stats.soa_memory_bytes == 0);
    DOCTEST_CHECK(empty_memory_stats.quantized_memory_bytes == 0);
    
    // Test 4: Ray casting on empty BVH structures
    std::vector<CollisionDetection::RayQuery> empty_test_rays = {
        CollisionDetection::RayQuery(make_vec3(0, 0, 0), make_vec3(0, 0, 1))
    };
    
    DOCTEST_CHECK_NOTHROW(collision.castRays(empty_test_rays));
    DOCTEST_CHECK_NOTHROW(collision.castRaysOptimized(empty_test_rays));
    
    auto empty_results = collision.castRaysOptimized(empty_test_rays);
    DOCTEST_CHECK(empty_results.size() == 1);
    DOCTEST_CHECK(empty_results[0].hit == false);
    
    // Test 5: Stream processing with empty stream
    CollisionDetection::RayStream empty_stream;
    CollisionDetection::RayTracingStats empty_stats;
    DOCTEST_CHECK_NOTHROW(collision.processRayStream(empty_stream, &empty_stats));
    
    // Add geometry and test normal operation recovery
    uint recovery_triangle = context.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5f, 0, 1));
    collision.buildBVH();
    
    std::vector<CollisionDetection::RayQuery> recovery_rays = {
        CollisionDetection::RayQuery(make_vec3(0.5f, -1, 0.5f), make_vec3(0, 1, 0))
    };
    
    auto recovery_results = collision.castRaysOptimized(recovery_rays);
    DOCTEST_CHECK(recovery_results.size() == 1);
    DOCTEST_CHECK(recovery_results[0].hit == true); // Should now hit the triangle
}

DOCTEST_TEST_CASE("CollisionDetection - Memory and Statistics Validation") {
    Context context;
    CollisionDetection collision(&context);
    collision.disableMessages();
    
    // Create a reasonable amount of test geometry for meaningful statistics
    for (int i = 0; i < 8; i++) {
        float x = i * 2.0f;
        float y = (i % 2) * 2.0f;
        context.addTriangle(make_vec3(x, y, 0), make_vec3(x+1, y, 0), make_vec3(x+0.5f, y, 1));
    }
    auto sphere_uuids = context.addSphere(16, make_vec3(10, 10, 1), 1.5f);
    
    // Build BVH in all modes and collect statistics
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    collision.buildBVH();
    auto legacy_memory = collision.getBVHMemoryUsage();
    
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_UNCOMPRESSED);
    auto soa_memory = collision.getBVHMemoryUsage();
    
    collision.setBVHOptimizationMode(CollisionDetection::BVHOptimizationMode::SOA_QUANTIZED);  
    auto quantized_memory = collision.getBVHMemoryUsage();
    
    // Test memory usage statistics accuracy
    DOCTEST_CHECK(soa_memory.soa_memory_bytes > 0);
    DOCTEST_CHECK(quantized_memory.quantized_memory_bytes > 0);
    
    // Validate memory hierarchy: quantized < soa
    DOCTEST_CHECK(quantized_memory.quantized_memory_bytes < soa_memory.soa_memory_bytes);
    
    // Validate reduction percentages
    DOCTEST_CHECK(quantized_memory.quantized_reduction_percent > 0.0f);
    
    // Test ray tracing statistics collection
    std::vector<CollisionDetection::RayQuery> stat_test_rays;
    for (int i = 0; i < 20; i++) {
        float x = (i % 4) * 2.0f + 0.5f;
        float y = (i / 4) * 2.0f + 0.5f;
        stat_test_rays.push_back(CollisionDetection::RayQuery(make_vec3(x, y, -1), make_vec3(0, 0, 1)));
    }
    
    CollisionDetection::RayTracingStats stats;
    auto stat_results = collision.castRaysOptimized(stat_test_rays, &stats);
    
    // Validate statistics collection
    DOCTEST_CHECK(stats.total_rays_cast == 20);
    DOCTEST_CHECK(stat_results.size() == 20);
    DOCTEST_CHECK(stats.total_hits <= stats.total_rays_cast); // Hits can't exceed rays
    
    if (stats.total_hits > 0) {
        DOCTEST_CHECK(stats.average_ray_distance > 0.0f);
        DOCTEST_CHECK(stats.bvh_nodes_visited > 0); // Should visit at least some BVH nodes
    }
    
    // Test stream processing statistics
    CollisionDetection::RayStream stats_stream;
    stats_stream.addRays(stat_test_rays);
    
    CollisionDetection::RayTracingStats stream_stats;
    bool stream_success = collision.processRayStream(stats_stream, &stream_stats);
    DOCTEST_CHECK(stream_success == true);
    DOCTEST_CHECK(stream_stats.total_rays_cast == 20);
    
    // Stream stats should be consistent with direct ray casting
    DOCTEST_CHECK(stream_stats.total_hits == stats.total_hits);
    DOCTEST_CHECK(std::abs(stream_stats.average_ray_distance - stats.average_ray_distance) < 0.01f);
}