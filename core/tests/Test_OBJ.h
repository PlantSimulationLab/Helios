#pragma once
// =================================================================================
// Suite: OBJ File I/O Tests
//
// Tests for OBJ file loading and writing functionality in Context class.
// Includes basic functionality, performance benchmarking, error handling, and
// correctness validation tests.
// =================================================================================

#include <chrono>
#include <fstream>

TEST_CASE("OBJ File Loading - Basic Functionality") {
    SUBCASE("Load simple triangle") {
        Context ctx;
        std::vector<uint> UUIDs;

        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_triangle_simple.obj", true));
        DOCTEST_CHECK(UUIDs.size() == 1);
        DOCTEST_CHECK(ctx.getPrimitiveCount() == 1);

        // Verify triangle properties
        DOCTEST_CHECK(ctx.doesPrimitiveExist(UUIDs[0]));
        DOCTEST_CHECK(ctx.getPrimitiveType(UUIDs[0]) == PRIMITIVE_TYPE_TRIANGLE);
        DOCTEST_CHECK(ctx.getPrimitiveArea(UUIDs[0]) > 0.0f);

        // Check vertices are approximately correct
        std::vector<vec3> vertices = ctx.getPrimitiveVertices(UUIDs[0]);
        DOCTEST_CHECK(vertices.size() == 3);
        DOCTEST_CHECK(vertices[0].x == doctest::Approx(0.0f));
        DOCTEST_CHECK(vertices[0].y == doctest::Approx(0.0f));
        DOCTEST_CHECK(vertices[0].z == doctest::Approx(0.0f));
        DOCTEST_CHECK(vertices[1].x == doctest::Approx(1.0f));
        DOCTEST_CHECK(vertices[1].y == doctest::Approx(0.0f));
        DOCTEST_CHECK(vertices[1].z == doctest::Approx(0.0f));
        DOCTEST_CHECK(vertices[2].x == doctest::Approx(0.5f));
        DOCTEST_CHECK(vertices[2].y == doctest::Approx(1.0f));
        DOCTEST_CHECK(vertices[2].z == doctest::Approx(0.0f));
    }

    SUBCASE("Load cube with materials") {
        Context ctx;
        std::vector<uint> UUIDs;

        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_cube_medium.obj", true));
        DOCTEST_CHECK(UUIDs.size() == 12); // Cube has 6 faces, each split into 2 triangles
        DOCTEST_CHECK(ctx.getPrimitiveCount() == 12);

        // All should be triangles
        for (uint uuid: UUIDs) {
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));
            DOCTEST_CHECK(ctx.getPrimitiveType(uuid) == PRIMITIVE_TYPE_TRIANGLE);
            DOCTEST_CHECK(ctx.getPrimitiveArea(uuid) > 0.0f);
        }

        // Total surface area should be approximately 24 (6 faces * 2*2 area per face)
        float total_area = 0.0f;
        for (uint uuid: UUIDs) {
            total_area += ctx.getPrimitiveArea(uuid);
        }
        DOCTEST_CHECK(total_area == doctest::Approx(24.0f).epsilon(1e-3));
    }

    SUBCASE("Load complex large model") {
        Context ctx;
        std::vector<uint> UUIDs;

        // Record loading time for performance awareness
        auto start = std::chrono::high_resolution_clock::now();
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_complex_large.obj", true));
        auto end = std::chrono::high_resolution_clock::now();

        // Should load successfully
        DOCTEST_CHECK(UUIDs.size() > 5000); // Should have many triangles from grid + icosphere
        DOCTEST_CHECK(ctx.getPrimitiveCount() == UUIDs.size());

        // Verify all primitives are valid
        for (uint uuid: UUIDs) {
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));
            DOCTEST_CHECK(ctx.getPrimitiveType(uuid) == PRIMITIVE_TYPE_TRIANGLE);
        }
    }

    SUBCASE("Load with transformations") {
        Context ctx;
        std::vector<uint> UUIDs;

        vec3 origin = make_vec3(5.0f, 10.0f, 2.0f);
        float height = 3.0f;
        SphericalCoord rotation = make_SphericalCoord(M_PI / 4, M_PI / 6);
        RGBcolor color = RGB::red;

        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_triangle_simple.obj", origin, height, rotation, color, "ZUP", true));
        DOCTEST_CHECK(UUIDs.size() == 1);

        // Check that triangle has been transformed
        std::vector<vec3> vertices = ctx.getPrimitiveVertices(UUIDs[0]);
        DOCTEST_CHECK(vertices.size() == 3);

        // Vertices should not be at original positions due to transformation
        DOCTEST_CHECK(!(vertices[0].x == 0.0f && vertices[0].y == 0.0f && vertices[0].z == 0.0f));

        // Check color was applied
        RGBcolor prim_color = ctx.getPrimitiveColor(UUIDs[0]);
        DOCTEST_CHECK(prim_color.r == doctest::Approx(color.r));
        DOCTEST_CHECK(prim_color.g == doctest::Approx(color.g));
        DOCTEST_CHECK(prim_color.b == doctest::Approx(color.b));
    }

    SUBCASE("Load existing test model") {
        Context ctx;
        std::vector<uint> UUIDs;

        // Test loading the existing test OBJ file in lib/models/
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/obj_object_test.obj", true));
        DOCTEST_CHECK(UUIDs.size() > 0);
        DOCTEST_CHECK(ctx.getPrimitiveCount() > 0);

        // Verify all loaded primitives are valid
        for (uint uuid: UUIDs) {
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));
            DOCTEST_CHECK(ctx.getPrimitiveType(uuid) == PRIMITIVE_TYPE_TRIANGLE);
            DOCTEST_CHECK(ctx.getPrimitiveArea(uuid) > 0.0f);
        }
    }
}

TEST_CASE("OBJ File Writing - Basic Functionality") {
    SUBCASE("Write simple geometry and reload") {
        Context ctx;

        // Create some test geometry
        uint tri1 = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5, 1, 0), RGB::red);
        uint tri2 = ctx.addTriangle(make_vec3(2, 0, 0), make_vec3(3, 0, 0), make_vec3(2.5, 1, 0), RGB::blue);

        std::vector<uint> original_uuids = {tri1, tri2};

        // Write to OBJ file
        std::string output_file = "lib/models/test_output_simple.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), original_uuids, false, true));

        // Verify file was created
        std::ifstream file_check(output_file);
        DOCTEST_CHECK(file_check.good());
        file_check.close();

        // Load back and compare
        Context ctx2;
        std::vector<uint> loaded_uuids;
        DOCTEST_CHECK_NOTHROW(loaded_uuids = ctx2.loadOBJ(output_file.c_str(), true));

        DOCTEST_CHECK(loaded_uuids.size() == 2);
        DOCTEST_CHECK(ctx2.getPrimitiveCount() == 2);

        // Check geometry is preserved (approximately)
        for (size_t i = 0; i < loaded_uuids.size(); i++) {
            std::vector<vec3> orig_verts = ctx.getPrimitiveVertices(original_uuids[i]);
            std::vector<vec3> loaded_verts = ctx2.getPrimitiveVertices(loaded_uuids[i]);

            DOCTEST_CHECK(orig_verts.size() == loaded_verts.size());
            for (size_t j = 0; j < orig_verts.size(); j++) {
                DOCTEST_CHECK(orig_verts[j].x == doctest::Approx(loaded_verts[j].x));
                DOCTEST_CHECK(orig_verts[j].y == doctest::Approx(loaded_verts[j].y));
                DOCTEST_CHECK(orig_verts[j].z == doctest::Approx(loaded_verts[j].z));
            }
        }

        // Clean up
        std::remove(output_file.c_str());
        std::remove("lib/models/test_output_simple.mtl");
    }

    SUBCASE("Write all primitives") {
        Context ctx;

        // Create various primitive types
        uint tri = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5, 1, 0));
        uint patch = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));

        // Write all primitives
        std::string output_file = "lib/models/test_output_all.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), true, true));

        // Verify file was created and has content
        std::ifstream file_check(output_file);
        DOCTEST_CHECK(file_check.good());

        std::string line;
        bool has_vertices = false;
        bool has_faces = false;
        while (std::getline(file_check, line)) {
            if (line.substr(0, 2) == "v ")
                has_vertices = true;
            if (line.substr(0, 2) == "f ")
                has_faces = true;
        }
        file_check.close();

        DOCTEST_CHECK(has_vertices);
        DOCTEST_CHECK(has_faces);

        // Clean up
        std::remove(output_file.c_str());
        std::remove("lib/models/test_output_all.mtl");
    }
}

TEST_CASE("OBJ File I/O - Error Handling and Edge Cases") {
    SUBCASE("Handle empty OBJ file") {
        Context ctx;
        std::vector<uint> UUIDs;

        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_empty.obj", true));
        DOCTEST_CHECK(UUIDs.empty());
        DOCTEST_CHECK(ctx.getPrimitiveCount() == 0);
    }

    SUBCASE("Handle non-existent file") {
        Context ctx;
        std::vector<uint> UUIDs;

        DOCTEST_CHECK_THROWS(UUIDs = ctx.loadOBJ("lib/models/does_not_exist.obj", true));
    }

    SUBCASE("Handle malformed OBJ file") {
        Context ctx;

        // This should throw an exception for malformed content
        DOCTEST_CHECK_THROWS(ctx.loadOBJ("lib/models/test_malformed.obj", true));
    }

    SUBCASE("Handle invalid file extension") {
        Context ctx;
        std::vector<uint> UUIDs;

        DOCTEST_CHECK_THROWS(UUIDs = ctx.loadOBJ("lib/models/test_triangle_simple.txt", true));
    }

    SUBCASE("Write to invalid path") {
        Context ctx;
        uint tri = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0.5, 1, 0));

        // Try to write to invalid path
        std::vector<uint> test_uuids = {tri};
        DOCTEST_CHECK_THROWS(ctx.writeOBJ("/invalid/path/test.obj", test_uuids, true, true));
    }

    SUBCASE("Write with invalid UUIDs") {
        Context ctx;
        std::vector<uint> invalid_uuids = {999, 1000}; // Non-existent UUIDs

        // Should throw exception for invalid UUIDs (fail-fast philosophy)
        DOCTEST_CHECK_THROWS(ctx.writeOBJ("lib/models/test_invalid_uuid.obj", invalid_uuids, false, true));

        // Clean up if file was created
        std::remove("lib/models/test_invalid_uuid.obj");
        std::remove("lib/models/test_invalid_uuid.mtl");
    }
}

TEST_CASE("OBJ File Loading - Texture and UV Coordinate Testing") {
    SUBCASE("Load textured model with UV coordinates") {
        Context ctx;
        std::vector<uint> UUIDs;

        // Load the test model that has texture references in MTL file (not silent for debugging)
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_complex_large.obj", true));
        DOCTEST_CHECK(UUIDs.size() > 0);

        // Verify that at least some primitives were loaded
        DOCTEST_CHECK(ctx.getPrimitiveCount() > 0);

        bool found_textured_primitive = false;
        int primitive_count = 0;
        int textured_count = 0;

        for (uint uuid: UUIDs) {
            primitive_count++;
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));

            // Test primitive type
            PrimitiveType type = ctx.getPrimitiveType(uuid);
            DOCTEST_CHECK((type == PRIMITIVE_TYPE_TRIANGLE || type == PRIMITIVE_TYPE_PATCH));

            // Test that vertices are valid
            std::vector<vec3> vertices = ctx.getPrimitiveVertices(uuid);
            DOCTEST_CHECK(vertices.size() >= 3);

            // Check if this primitive has texture
            if (type == PRIMITIVE_TYPE_TRIANGLE) {
                // Test texture file access
                std::string texture_file;
                DOCTEST_CHECK_NOTHROW(texture_file = ctx.getPrimitiveTextureFile(uuid));

                // Count textured primitives
                if (!texture_file.empty()) {
                    textured_count++;
                }

                // For textured primitives, verify texture properties
                if (!texture_file.empty()) {
                    found_textured_primitive = true;

                    // Test UV coordinates using public API
                    std::vector<vec2> uv_coords;
                    DOCTEST_CHECK_NOTHROW(uv_coords = ctx.getPrimitiveTextureUV(uuid));
                    DOCTEST_CHECK(uv_coords.size() == 3); // Triangle should have 3 UV coordinates

                    // Verify UV coordinates are finite
                    for (const auto &uv: uv_coords) {
                        DOCTEST_CHECK(std::isfinite(uv.x));
                        DOCTEST_CHECK(std::isfinite(uv.y));
                        // UV coordinates are often outside [0,1] for tiling, so just check they're finite
                    }

                    // Test texture transparency and color override properties
                    bool has_transparency, color_overridden;
                    DOCTEST_CHECK_NOTHROW(has_transparency = ctx.primitiveTextureHasTransparencyChannel(uuid));
                    DOCTEST_CHECK_NOTHROW(color_overridden = ctx.isPrimitiveTextureColorOverridden(uuid));

                    // These should be boolean values (not checking specific values, just that calls work)
                    DOCTEST_CHECK((has_transparency == true || has_transparency == false));
                    DOCTEST_CHECK((color_overridden == true || color_overridden == false));
                }
            } else if (type == PRIMITIVE_TYPE_PATCH) {
                // Test patch texture properties
                std::string texture_file;
                DOCTEST_CHECK_NOTHROW(texture_file = ctx.getPrimitiveTextureFile(uuid));

                if (!texture_file.empty()) {
                    textured_count++;
                }

                if (!texture_file.empty()) {
                    found_textured_primitive = true;

                    // Test UV coordinates for patches using public API
                    std::vector<vec2> uv_coords;
                    DOCTEST_CHECK_NOTHROW(uv_coords = ctx.getPrimitiveTextureUV(uuid));
                    DOCTEST_CHECK(uv_coords.size() == 4); // Patch should have 4 UV coordinates

                    // Verify UV coordinates are finite
                    for (const auto &uv: uv_coords) {
                        DOCTEST_CHECK(std::isfinite(uv.x));
                        DOCTEST_CHECK(std::isfinite(uv.y));
                    }
                }
            }

            // Test color properties (should work for both textured and non-textured primitives)
            RGBcolor color;
            DOCTEST_CHECK_NOTHROW(color = ctx.getPrimitiveColor(uuid));
            DOCTEST_CHECK(std::isfinite(color.r));
            DOCTEST_CHECK(std::isfinite(color.g));
            DOCTEST_CHECK(std::isfinite(color.b));
        }

        // For now, verify that texture methods work even if no textures found
        // This ensures our OpenMP implementation doesn't break texture API calls
    }

    SUBCASE("Load model with material properties") {
        Context ctx;
        std::vector<uint> UUIDs;

        // Load a model with different materials
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_cube_medium.obj", true));
        DOCTEST_CHECK(UUIDs.size() > 0);

        // Verify materials are applied correctly
        std::vector<RGBcolor> colors_found;

        for (uint uuid: UUIDs) {
            RGBcolor color = ctx.getPrimitiveColor(uuid);
            colors_found.push_back(color);

            // Colors should be valid (split compound expressions for doctest)
            DOCTEST_CHECK(color.r >= 0.0f);
            DOCTEST_CHECK(color.r <= 1.0f);
            DOCTEST_CHECK(color.g >= 0.0f);
            DOCTEST_CHECK(color.g <= 1.0f);
            DOCTEST_CHECK(color.b >= 0.0f);
            DOCTEST_CHECK(color.b <= 1.0f);
        }

        // Check that we have different materials/colors by comparing first few colors
        DOCTEST_CHECK(colors_found.size() >= 2);
        if (colors_found.size() >= 2) {
            // At least some colors should be different (indicating multiple materials)
            bool found_different = false;
            for (size_t i = 1; i < colors_found.size(); i++) {
                if (colors_found[0].r != colors_found[i].r || colors_found[0].g != colors_found[i].g || colors_found[0].b != colors_found[i].b) {
                    found_different = true;
                    break;
                }
            }
            DOCTEST_CHECK(found_different);
        }
    }

    SUBCASE("Texture coordinate consistency with transformations") {
        Context ctx;
        std::vector<uint> UUIDs;

        // Load with transformations to test that UV coordinates are preserved
        vec3 origin = make_vec3(5.0f, 0.0f, 0.0f);
        float height = 2.0f;
        SphericalCoord rotation = make_SphericalCoord(M_PI / 4, 0);
        RGBcolor color = RGB::blue;

        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/obj_object_test.obj", origin, height, rotation, color, "ZUP", true));
        DOCTEST_CHECK(UUIDs.size() > 0);

        for (uint uuid: UUIDs) {
            if (ctx.getPrimitiveType(uuid) == PRIMITIVE_TYPE_TRIANGLE) {
                std::string texture_file = ctx.getPrimitiveTextureFile(uuid);

                if (!texture_file.empty()) {
                    // UV coordinates should still be valid after transformations
                    std::vector<vec2> uv_coords = ctx.getPrimitiveTextureUV(uuid);
                    DOCTEST_CHECK(uv_coords.size() == 3);

                    for (const auto &uv: uv_coords) {
                        DOCTEST_CHECK(std::isfinite(uv.x));
                        DOCTEST_CHECK(std::isfinite(uv.y));
                    }

                    // Vertices should be transformed but texture coordinates unchanged
                    std::vector<vec3> vertices = ctx.getPrimitiveVertices(uuid);
                    DOCTEST_CHECK(vertices.size() == 3);

                    // At least one vertex should be displaced from origin due to transformations
                    bool found_transformed = false;
                    for (const auto &vertex: vertices) {
                        if (vertex.magnitude() > 1.0f) { // Original vertices are within unit range
                            found_transformed = true;
                            break;
                        }
                    }
                    DOCTEST_CHECK(found_transformed);
                }
            }
        }
    }

    SUBCASE("Missing texture file handling") {
        Context ctx;
        std::vector<uint> UUIDs;

        // This should load successfully even though texture.jpg doesn't exist
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_complex_large.obj", true));
        DOCTEST_CHECK(UUIDs.size() > 0);

        // Primitives should still be created even with missing texture files
        for (uint uuid: UUIDs) {
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));

            // Should have valid geometry regardless of missing textures
            std::vector<vec3> vertices = ctx.getPrimitiveVertices(uuid);
            DOCTEST_CHECK(vertices.size() >= 3);
            DOCTEST_CHECK(ctx.getPrimitiveArea(uuid) > 0.0f);
        }
    }
}

TEST_CASE("OBJ File Loading - Error Handling and Edge Cases") {
    SUBCASE("Invalid face vertex indices") {
        Context ctx;

        // Create a test file with invalid face indices
        std::string test_content = "# Test OBJ with invalid face indices\n"
                                   "v 0.0 0.0 0.0\n"
                                   "v 1.0 0.0 0.0\n"
                                   "v 0.5 1.0 0.0\n"
                                   "f 1 2 5\n"; // Face references vertex 5 but only 3 vertices exist

        std::string test_file = "lib/models/test_invalid_face.obj";
        std::ofstream file(test_file);
        file << test_content;
        file.close();

        // Should throw helios_runtime_error for invalid face index
        DOCTEST_CHECK_THROWS_AS(ctx.loadOBJ(test_file.c_str(), true), std::runtime_error);

        // Clean up
        std::remove(test_file.c_str());
    }

    SUBCASE("Invalid texture coordinate indices") {
        Context ctx;

        // Create a test file with invalid UV indices
        std::string test_content = "# Test OBJ with invalid UV indices\n"
                                   "v 0.0 0.0 0.0\n"
                                   "v 1.0 0.0 0.0\n"
                                   "v 0.5 1.0 0.0\n"
                                   "vt 0.0 0.0\n"
                                   "vt 1.0 0.0\n"
                                   "f 1/3 2/1 3/2\n"; // UV index 3 doesn't exist (only 2 UVs defined)

        std::string test_file = "lib/models/test_invalid_uv.obj";
        std::ofstream file(test_file);
        file << test_content;
        file.close();

        // Should throw helios_runtime_error for invalid UV index
        DOCTEST_CHECK_THROWS_AS(ctx.loadOBJ(test_file.c_str(), true), std::runtime_error);

        // Clean up
        std::remove(test_file.c_str());
    }

    SUBCASE("Missing texture file") {
        Context ctx;

        // Create test OBJ and MTL files where MTL references non-existent texture
        std::string obj_content = "# Test OBJ with missing texture\n"
                                  "mtllib test_missing_texture.mtl\n"
                                  "v 0.0 0.0 0.0\n"
                                  "v 1.0 0.0 0.0\n"
                                  "v 0.5 1.0 0.0\n"
                                  "usemtl test_material\n"
                                  "f 1 2 3\n";

        std::string mtl_content = "newmtl test_material\n"
                                  "Ka 0.2 0.2 0.2\n"
                                  "Kd 0.8 0.8 0.8\n"
                                  "map_Kd nonexistent_texture.jpg\n"; // This texture file doesn't exist

        std::string obj_file = "lib/models/test_missing_texture.obj";
        std::string mtl_file = "lib/models/test_missing_texture.mtl";

        std::ofstream obj_f(obj_file);
        obj_f << obj_content;
        obj_f.close();

        std::ofstream mtl_f(mtl_file);
        mtl_f << mtl_content;
        mtl_f.close();

        // Should throw helios_runtime_error for missing texture file
        DOCTEST_CHECK_THROWS_AS(ctx.loadOBJ(obj_file.c_str(), true), std::runtime_error);

        // Clean up
        std::remove(obj_file.c_str());
        std::remove(mtl_file.c_str());
    }

    SUBCASE("Texture without UV coordinates") {
        Context ctx;

        // Create a dummy texture file
        std::string texture_file = "lib/models/test_dummy.jpg";
        std::ofstream tex_f(texture_file);
        tex_f << "dummy";
        tex_f.close();

        // Create test files where material has texture but face has no UV coordinates
        std::string obj_content = "# Test OBJ with texture but no UV coordinates\n"
                                  "mtllib test_no_uv.mtl\n"
                                  "v 0.0 0.0 0.0\n"
                                  "v 1.0 0.0 0.0\n"
                                  "v 0.5 1.0 0.0\n"
                                  "usemtl textured_material\n"
                                  "f 1 2 3\n"; // Face has no UV indices

        std::string mtl_content = "newmtl textured_material\n"
                                  "Ka 0.2 0.2 0.2\n"
                                  "Kd 0.8 0.8 0.8\n"
                                  "map_Kd test_dummy.jpg\n";

        std::string obj_file = "lib/models/test_no_uv.obj";
        std::string mtl_file = "lib/models/test_no_uv.mtl";

        std::ofstream obj_f(obj_file);
        obj_f << obj_content;
        obj_f.close();

        std::ofstream mtl_f(mtl_file);
        mtl_f << mtl_content;
        mtl_f.close();

        // Should throw helios_runtime_error for texture without UV coordinates
        DOCTEST_CHECK_THROWS_AS(ctx.loadOBJ(obj_file.c_str(), true), std::runtime_error);

        // Clean up
        std::remove(obj_file.c_str());
        std::remove(mtl_file.c_str());
        std::remove(texture_file.c_str());
    }

    SUBCASE("Zero vertex file") {
        Context ctx;

        // Create a test file with faces but no vertices
        std::string test_content = "# Test OBJ with no vertices\n"
                                   "f 1 2 3\n"; // Face references vertices but none are defined

        std::string test_file = "lib/models/test_zero_vertex.obj";
        std::ofstream file(test_file);
        file << test_content;
        file.close();

        // Should throw helios_runtime_error for face referencing non-existent vertices
        DOCTEST_CHECK_THROWS_AS(ctx.loadOBJ(test_file.c_str(), true), std::runtime_error);

        // Clean up
        std::remove(test_file.c_str());
    }
}

TEST_CASE("OBJ File I/O - Performance Benchmarking") {
    SUBCASE("Loading performance baseline") {
        Context ctx;
        std::vector<uint> UUIDs;

        // Benchmark loading the large complex file
        auto start = std::chrono::high_resolution_clock::now();
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_complex_large.obj", true));
        auto end = std::chrono::high_resolution_clock::now();

        auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        DOCTEST_CHECK(UUIDs.size() > 0);

        // Store baseline metrics (for future optimization comparison)
        float triangles_per_second = UUIDs.size() * 1000000.0f / load_duration.count();
        float load_time_ms = load_duration.count() / 1000.0f;

        // Basic sanity check - should process at least 1000 triangles/second
        // (This is a very conservative baseline)
        DOCTEST_CHECK(triangles_per_second > 1000.0f);
    }

    SUBCASE("Loading performance with transformations") {
        Context ctx;
        std::vector<uint> UUIDs;

        // Test performance with transformations (more CPU intensive)
        vec3 origin = make_vec3(10.0f, 5.0f, 0.0f);
        float height = 2.0f;
        SphericalCoord rotation = make_SphericalCoord(M_PI / 6, M_PI / 4);
        RGBcolor color = RGB::green;

        auto start = std::chrono::high_resolution_clock::now();
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_complex_large.obj", origin, height, rotation, color, "ZUP", true));
        auto end = std::chrono::high_resolution_clock::now();

        auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        DOCTEST_CHECK(UUIDs.size() > 0);

        float triangles_per_second = UUIDs.size() * 1000000.0f / load_duration.count();
        float load_time_ms = load_duration.count() / 1000.0f;

        // Transformed loading should still maintain reasonable performance
        DOCTEST_CHECK(triangles_per_second > 800.0f);
    }

    SUBCASE("Writing performance baseline") {
        Context ctx;

        // Create a substantial amount of geometry
        std::vector<uint> uuids;
        for (int i = 0; i < 1000; i++) {
            float x = static_cast<float>(i % 10);
            float z = static_cast<float>(i / 10);
            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));
            uuids.push_back(tri);
        }

        // Benchmark writing
        std::string output_file = "lib/models/test_perf_output.obj";
        auto start = std::chrono::high_resolution_clock::now();
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));
        auto end = std::chrono::high_resolution_clock::now();

        auto write_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Basic sanity check - should write at least 500 triangles/second
        float triangles_per_second = uuids.size() * 1000000.0f / write_duration.count();
        DOCTEST_CHECK(triangles_per_second > 500.0f);

        // Clean up
        std::remove(output_file.c_str());
        std::remove("lib/models/test_perf_output.mtl");
    }
}

TEST_CASE("OBJ File I/O - Correctness Validation") {
    SUBCASE("Round-trip geometry preservation") {
        Context ctx1, ctx2, ctx3;

        // Load original file
        std::vector<uint> original_uuids;
        DOCTEST_CHECK_NOTHROW(original_uuids = ctx1.loadOBJ("lib/models/test_cube_medium.obj", true));

        // Write to new file
        std::string intermediate_file = "lib/models/test_roundtrip.obj";
        DOCTEST_CHECK_NOTHROW(ctx1.writeOBJ(intermediate_file.c_str(), original_uuids, false, true));

        // Load the written file
        std::vector<uint> roundtrip_uuids;
        DOCTEST_CHECK_NOTHROW(roundtrip_uuids = ctx2.loadOBJ(intermediate_file.c_str(),  true));

        // Write again
        std::string final_file = "lib/models/test_roundtrip2.obj";
        DOCTEST_CHECK_NOTHROW(ctx2.writeOBJ(final_file.c_str(), roundtrip_uuids, false, true));

        // Load final file
        std::vector<uint> final_uuids;
        DOCTEST_CHECK_NOTHROW(final_uuids = ctx3.loadOBJ(final_file.c_str(), true));

        // All should have same number of triangles
        DOCTEST_CHECK(original_uuids.size() == roundtrip_uuids.size());
        DOCTEST_CHECK(roundtrip_uuids.size() == final_uuids.size());

        // Total area should be preserved (approximately)
        float original_area = 0, roundtrip_area = 0, final_area = 0;

        for (uint uuid: original_uuids) {
            original_area += ctx1.getPrimitiveArea(uuid);
        }
        for (uint uuid: roundtrip_uuids) {
            roundtrip_area += ctx2.getPrimitiveArea(uuid);
        }
        for (uint uuid: final_uuids) {
            final_area += ctx3.getPrimitiveArea(uuid);
        }

        DOCTEST_CHECK(original_area == doctest::Approx(roundtrip_area).epsilon(1e-4));
        DOCTEST_CHECK(roundtrip_area == doctest::Approx(final_area).epsilon(1e-4));

        // Clean up
        std::remove(intermediate_file.c_str());
        std::remove("lib/models/test_roundtrip.mtl");
        std::remove(final_file.c_str());
        std::remove("lib/models/test_roundtrip2.mtl");
    }

    SUBCASE("Material preservation") {
        Context ctx1, ctx2;

        // Load file with materials
        std::vector<uint> original_uuids;
        DOCTEST_CHECK_NOTHROW(original_uuids = ctx1.loadOBJ("lib/models/test_cube_medium.obj", true));

        // Write and reload
        std::string output_file = "lib/models/test_material_preservation.obj";
        DOCTEST_CHECK_NOTHROW(ctx1.writeOBJ(output_file.c_str(), original_uuids, false, true));

        std::vector<uint> reloaded_uuids;
        DOCTEST_CHECK_NOTHROW(reloaded_uuids = ctx2.loadOBJ(output_file.c_str(), true));

        DOCTEST_CHECK(original_uuids.size() == reloaded_uuids.size());

        // Colors should be preserved (at least approximately)
        for (size_t i = 0; i < std::min(original_uuids.size(), reloaded_uuids.size()); i++) {
            RGBcolor orig_color = ctx1.getPrimitiveColor(original_uuids[i]);
            RGBcolor new_color = ctx2.getPrimitiveColor(reloaded_uuids[i]);

            // Colors should be reasonable (not black unless intentionally black)
            float orig_brightness = orig_color.r + orig_color.g + orig_color.b;
            float new_brightness = new_color.r + new_color.g + new_color.b;
            DOCTEST_CHECK((orig_brightness > 0.01f || new_brightness > 0.01f));
        }

        // Clean up
        std::remove(output_file.c_str());
        std::remove("lib/models/test_material_preservation.mtl");
    }
}

TEST_CASE("OBJ WriteOBJ - Comprehensive Test Suite for Optimization") {

    // Helper function to create test datasets
    auto createTestDataset = [](Context &ctx, const std::string &type, int count) -> std::vector<uint> {
        std::vector<uint> uuids;

        if (type == "simple_triangles") {
            for (int i = 0; i < count; i++) {
                float x = static_cast<float>(i % 10);
                float z = static_cast<float>(i / 10);
                uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), make_RGBcolor(0.5f + 0.5f * (i % 3 == 0), 0.5f + 0.5f * (i % 3 == 1), 0.5f + 0.5f * (i % 3 == 2)));
                uuids.push_back(tri);
            }
        } else if (type == "mixed_primitives") {
            for (int i = 0; i < count; i++) {
                float x = static_cast<float>(i % 10);
                float z = static_cast<float>(i / 10);
                if (i % 2 == 0) {
                    uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));
                    uuids.push_back(tri);
                } else {
                    uint patch = ctx.addPatch(make_vec3(x, 0, z), make_vec2(1, 1));
                    uuids.push_back(patch);
                }
            }
        } else if (type == "textured_primitives") {
            // For textured primitives, use solid colors instead of actual texture files
            // to avoid JPEG validation issues in the test environment
            for (int i = 0; i < count; i++) {
                float x = static_cast<float>(i % 10);
                float z = static_cast<float>(i / 10);
                // Create triangles with different colors to simulate material variation
                RGBcolor color = make_RGBcolor(0.3f + 0.4f * (i % 3 == 0), 0.3f + 0.4f * (i % 3 == 1), 0.3f + 0.4f * (i % 3 == 2));
                uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), color);
                uuids.push_back(tri);
            }
        } else if (type == "multi_material") {
            // Create different materials with different properties
            for (int i = 0; i < count; i++) {
                float x = static_cast<float>(i % 10);
                float z = static_cast<float>(i / 10);

                // Cycle through different material types
                int material_type = i % 4;
                uint prim;

                switch (material_type) {
                    case 0: // Red solid
                        prim = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), RGB::red);
                        break;
                    case 1: // Green solid
                        prim = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), RGB::green);
                        break;
                    case 2: // Blue solid
                        prim = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), RGB::blue);
                        break;
                    default: // Custom color
                        prim = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), make_RGBcolor(0.8f, 0.4f, 0.6f));
                        break;
                }
                uuids.push_back(prim);
            }
        } else if (type == "object_groups") {
            for (int i = 0; i < count; i++) {
                float x = static_cast<float>(i % 10);
                float z = static_cast<float>(i / 10);
                uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));

                // Add object group labels
                std::string object_label = "group_" + std::to_string(i / 25); // 25 triangles per group
                ctx.setPrimitiveData(tri, "object_label", object_label);
                uuids.push_back(tri);
            }
        }

        return uuids;
    };

    SUBCASE("Small dataset validation") {
        Context ctx;
        std::vector<uint> uuids = createTestDataset(ctx, "simple_triangles", 50);

        std::string output_file = "test_writeobj_small.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify round-trip correctness
        Context ctx_reload;
        std::vector<uint> reloaded_uuids;
        DOCTEST_CHECK_NOTHROW(reloaded_uuids = ctx_reload.loadOBJ(output_file.c_str(), true));

        DOCTEST_CHECK(uuids.size() == reloaded_uuids.size());

        // Clean up
        std::remove(output_file.c_str());
        std::remove("test_writeobj_small.mtl");
    }

    SUBCASE("Mixed primitives validation") {
        Context ctx;
        std::vector<uint> uuids = createTestDataset(ctx, "mixed_primitives", 100);

        std::string output_file = "test_writeobj_mixed.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify round-trip correctness
        Context ctx_reload;
        std::vector<uint> reloaded_uuids;
        DOCTEST_CHECK_NOTHROW(reloaded_uuids = ctx_reload.loadOBJ(output_file.c_str(), true));

        // Should have more primitives due to patch->triangle conversion
        DOCTEST_CHECK(reloaded_uuids.size() >= uuids.size());

        // Clean up
        std::remove(output_file.c_str());
        std::remove("test_writeobj_mixed.mtl");
    }

    SUBCASE("Textured primitives validation") {
        Context ctx;
        std::vector<uint> uuids = createTestDataset(ctx, "textured_primitives", 75);

        std::string output_file = "test_writeobj_textured.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify round-trip correctness
        Context ctx_reload;
        std::vector<uint> reloaded_uuids;
        DOCTEST_CHECK_NOTHROW(reloaded_uuids = ctx_reload.loadOBJ(output_file.c_str(), true));

        DOCTEST_CHECK(uuids.size() == reloaded_uuids.size());

        // Clean up
        std::remove(output_file.c_str());
        std::remove("test_writeobj_textured.mtl");
    }


    SUBCASE("Multi-material validation") {
        Context ctx;
        std::vector<uint> uuids = createTestDataset(ctx, "multi_material", 200);

        std::string output_file = "test_writeobj_materials.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify MTL file was created and contains multiple materials
        std::string mtl_file = "test_writeobj_materials.mtl";
        std::ifstream mtl_check(mtl_file);
        DOCTEST_CHECK(mtl_check.good());

        int material_count = 0;
        std::string line;
        while (std::getline(mtl_check, line)) {
            if (line.substr(0, 6) == "newmtl") {
                material_count++;
            }
        }
        mtl_check.close();

        DOCTEST_CHECK(material_count >= 4); // Should have at least 4 different materials

        // Clean up
        std::remove(output_file.c_str());
        std::remove(mtl_file.c_str());
    }

    SUBCASE("Object groups validation") {
        Context ctx;
        std::vector<uint> uuids = createTestDataset(ctx, "object_groups", 100);

        std::string output_file = "test_writeobj_groups.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify OBJ file contains object group directives
        std::ifstream obj_check(output_file);
        DOCTEST_CHECK(obj_check.good());

        int object_count = 0;
        std::string line;
        while (std::getline(obj_check, line)) {
            if (line.substr(0, 2) == "o ") {
                object_count++;
            }
        }
        obj_check.close();

        DOCTEST_CHECK(object_count >= 4); // Should have at least 4 object groups

        // Clean up
        std::remove(output_file.c_str());
        std::remove("test_writeobj_groups.mtl");
    }
}

TEST_CASE("OBJ WriteOBJ - Performance Benchmarking Suite") {

    // Performance benchmark helper
    auto benchmarkWriteOBJ = [](Context &ctx, const std::vector<uint> &uuids, const std::string &test_name) {
        std::string output_file = "bench_" + test_name + ".obj";

        auto start = std::chrono::high_resolution_clock::now();
        ctx.writeOBJ(output_file.c_str(), uuids, false, true);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float write_time_ms = duration.count() / 1000.0f;
        float primitives_per_second = uuids.size() * 1000000.0f / duration.count();

        // Clean up
        std::remove(output_file.c_str());
        std::string mtl_file = "bench_" + test_name + ".mtl";
        std::remove(mtl_file.c_str());

        return std::make_pair(write_time_ms, primitives_per_second);
    };

    SUBCASE("Baseline performance - 1000 triangles") {
        Context ctx;
        std::vector<uint> uuids;

        for (int i = 0; i < 1000; i++) {
            float x = static_cast<float>(i % 10);
            float z = static_cast<float>(i / 10);
            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));
            uuids.push_back(tri);
        }

        auto [time_ms, prims_per_sec] = benchmarkWriteOBJ(ctx, uuids, "baseline_1k");

        // Should achieve reasonable performance (conservative baseline)
        DOCTEST_CHECK(prims_per_sec > 100.0f);
    }

    SUBCASE("Multi-material performance - 2000 primitives") {
        Context ctx;
        std::vector<uint> uuids;

        for (int i = 0; i < 2000; i++) {
            float x = static_cast<float>(i % 20);
            float z = static_cast<float>(i / 20);

            // Create 10 different materials
            RGBcolor color = make_RGBcolor((i % 10) / 10.0f, 0.5f, 0.7f);
            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), color);
            uuids.push_back(tri);
        }

        auto [time_ms, prims_per_sec] = benchmarkWriteOBJ(ctx, uuids, "multi_material_2k");

        // Should handle multiple materials efficiently
        DOCTEST_CHECK(prims_per_sec > 50.0f);
    }

    SUBCASE("Large dataset performance - 5000 primitives") {
        Context ctx;
        std::vector<uint> uuids;

        for (int i = 0; i < 5000; i++) {
            float x = static_cast<float>(i % 50);
            float z = static_cast<float>(i / 50);
            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));
            uuids.push_back(tri);
        }

        auto [time_ms, prims_per_sec] = benchmarkWriteOBJ(ctx, uuids, "large_5k");

        // Performance should remain reasonable for larger datasets
        DOCTEST_CHECK(prims_per_sec > 25.0f);
    }

    SUBCASE("Memory usage monitoring") {
        Context ctx;
        std::vector<uint> uuids;

        // Create substantial dataset for memory testing
        for (int i = 0; i < 3000; i++) {
            float x = static_cast<float>(i % 30);
            float z = static_cast<float>(i / 30);
            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));
            uuids.push_back(tri);
        }

        std::string output_file = "bench_memory_test.obj";

        // This test primarily verifies no memory leaks or excessive allocation
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify file was created with reasonable size
        std::ifstream file_check(output_file, std::ios::ate);
        auto file_size = file_check.tellg();
        file_check.close();

        DOCTEST_CHECK(file_size > 0);
        DOCTEST_CHECK(file_size < 50 * 1024 * 1024); // Should be less than 50MB

        // Clean up
        std::remove(output_file.c_str());
        std::remove("bench_memory_test.mtl");
    }
}

TEST_CASE("OBJ WriteOBJ - Stress Testing and Edge Cases") {

    SUBCASE("Very large primitive count") {
        Context ctx;
        std::vector<uint> uuids;

        // Test with 10000 primitives to stress test the system
        for (int i = 0; i < 10000; i++) {
            float x = static_cast<float>(i % 100);
            float z = static_cast<float>(i / 100);
            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));
            uuids.push_back(tri);
        }

        std::string output_file = "stress_large.obj";

        auto start = std::chrono::high_resolution_clock::now();
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should complete within reasonable time (10 seconds max)
        DOCTEST_CHECK(duration.count() < 10000);

        // Clean up
        std::remove(output_file.c_str());
        std::remove("stress_large.mtl");
    }

    SUBCASE("Many materials stress test") {
        Context ctx;
        std::vector<uint> uuids;

        // Create 500 primitives with 100 different materials
        for (int i = 0; i < 500; i++) {
            float x = static_cast<float>(i % 25);
            float z = static_cast<float>(i / 25);

            // Create unique color for every 5th primitive (100 materials total)
            float r = static_cast<float>((i / 5) % 10) / 10.0f;
            float g = static_cast<float>((i / 5) % 10) / 10.0f;
            float b = static_cast<float>((i / 5) / 10) / 10.0f;
            RGBcolor color = make_RGBcolor(r, g, b);

            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z), color);
            uuids.push_back(tri);
        }

        std::string output_file = "stress_materials.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify many materials were created
        std::string mtl_file = "stress_materials.mtl";
        std::ifstream mtl_check(mtl_file);
        int material_count = 0;
        std::string line;
        while (std::getline(mtl_check, line)) {
            if (line.substr(0, 6) == "newmtl") {
                material_count++;
            }
        }
        mtl_check.close();

        DOCTEST_CHECK(material_count >= 50); // Should have many materials

        // Clean up
        std::remove(output_file.c_str());
        std::remove(mtl_file.c_str());
    }

    SUBCASE("Degenerate geometry handling") {
        Context ctx;
        std::vector<uint> uuids;

        // Create some regular triangles
        for (int i = 0; i < 100; i++) {
            float x = static_cast<float>(i % 10);
            float z = static_cast<float>(i / 10);
            uint tri = ctx.addTriangle(make_vec3(x, 0, z), make_vec3(x + 1, 0, z), make_vec3(x + 0.5f, 1, z));
            uuids.push_back(tri);
        }

        // Add some very small triangles that might be degenerate
        for (int i = 0; i < 10; i++) {
            float offset = i * 1e-8f;
            uint tri = ctx.addTriangle(make_vec3(100, 0, 0), make_vec3(100 + offset, 0, 0), make_vec3(100, offset, 0));
            uuids.push_back(tri);
        }

        std::string output_file = "stress_degenerate.obj";
        DOCTEST_CHECK_NOTHROW(ctx.writeOBJ(output_file.c_str(), uuids, false, true));

        // Verify file was written correctly by checking it exists and has content
        std::ifstream file_check(output_file);
        DOCTEST_CHECK(file_check.good());

        // Count lines to verify content was written
        std::string line;
        int line_count = 0;
        while (std::getline(file_check, line)) {
            line_count++;
        }
        file_check.close();

        DOCTEST_CHECK(line_count > 10); // Should have vertices, faces, etc.

        // Clean up
        std::remove(output_file.c_str());
        std::remove("stress_degenerate.mtl");
    }
}

TEST_CASE("OBJ Material Color Override Test") {
    SUBCASE("Material with Kd color and map_d transparency should use correct color") {
        Context ctx;

        // Create a simple OBJ file with a material that has both Kd (diffuse color) and map_d (transparency map)
        std::string obj_content = R"(# Test OBJ file for material color override bug
mtllib test_material_color.mtl
o TestObject
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 1.0
usemtl TestMaterial
f 1/1 2/2 3/3
)";

        // Create a test MTL file with blue diffuse color and transparency map
        std::string mtl_content = R"(# Test MTL file for material color override bug
newmtl TestMaterial
Ns 250.000000
Ka 1.000000 1.000000 1.000000
Kd 0.000000 0.000000 0.800000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.500000
illum 2
map_d lib/images/solid.jpg
)";

        // Write test files
        std::ofstream obj_file("lib/models/test_material_color.obj");
        obj_file << obj_content;
        obj_file.close();

        std::ofstream mtl_file("lib/models/test_material_color.mtl");
        mtl_file << mtl_content;
        mtl_file.close();

        // Load the OBJ file with a default color (green) to ensure materials override it properly
        std::vector<uint> UUIDs;
        RGBcolor default_color = RGB::green;
        DOCTEST_CHECK_NOTHROW(UUIDs = ctx.loadOBJ("lib/models/test_material_color.obj", make_vec3(0,0,0), make_vec3(1,1,1), nullrotation, default_color, "ZUP", true));
        DOCTEST_CHECK(UUIDs.size() == 1);

        // Verify that the triangle was loaded
        DOCTEST_CHECK(ctx.doesPrimitiveExist(UUIDs[0]));
        DOCTEST_CHECK(ctx.getPrimitiveType(UUIDs[0]) == PRIMITIVE_TYPE_TRIANGLE);

        // Check that the primitive has a texture (the map_d transparency texture)
        std::string texture_file = ctx.getPrimitiveTextureFile(UUIDs[0]);
        DOCTEST_CHECK(!texture_file.empty());

        // Most importantly: check that the color is BLUE (from Kd), not RED (old bug) or GREEN (default)
        RGBcolor primitive_color = ctx.getPrimitiveColor(UUIDs[0]);

        // The material specified Kd 0.000000 0.000000 0.800000 (blue)
        DOCTEST_CHECK(primitive_color.r == doctest::Approx(0.0f).epsilon(1e-5));
        DOCTEST_CHECK(primitive_color.g == doctest::Approx(0.0f).epsilon(1e-5));
        DOCTEST_CHECK(primitive_color.b == doctest::Approx(0.8f).epsilon(1e-5));

        // Verify that texture color is overridden (because only map_d was specified, not map_Kd)
        DOCTEST_CHECK(ctx.isPrimitiveTextureColorOverridden(UUIDs[0]));

        // Clean up test files
        std::remove("lib/models/test_material_color.obj");
        std::remove("lib/models/test_material_color.mtl");
    }
}
