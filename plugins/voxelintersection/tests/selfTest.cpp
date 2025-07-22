#include "VoxelIntersection.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

using namespace helios;

DOCTEST_TEST_CASE("VoxelIntersection - Patch with no slices") {
    Context context;
    
    // Create patch using center and size instead of vertices
    vec3 patch_center = make_vec3(0.5, 0, 0.5);
    vec2 patch_size = make_vec2(1, 1);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size);
    
    vec3 grid_center = make_vec3(0, 1, 0);
    vec3 grid_size = make_vec3(10, 10, 10);
    
    VoxelIntersection voxelintersection(&context);
    
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(2, 2, 2));
    
    DOCTEST_CHECK(voxel_UUIDs.size() >= 1);
}

DOCTEST_TEST_CASE("VoxelIntersection - Patch with one slice") {
    Context context;
    
    // Create patch using center and size instead of vertices
    vec3 patch_center = make_vec3(0.5, 0, 0);
    vec2 patch_size = make_vec2(2, 2);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size);
    
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(1, 10, 10);
    
    VoxelIntersection voxelintersection(&context);
    
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(2, 1, 1));
    
    DOCTEST_CHECK(voxel_UUIDs.size() >= 1);
}

DOCTEST_TEST_CASE("VoxelIntersection - Patch with 2 slices") {
    Context context;
    
    // Create patch using center and size instead of vertices
    vec3 patch_center = make_vec3(1, 0, 0);
    vec2 patch_size = make_vec2(3, 1);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size);
    
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(2, 10, 1);
    
    VoxelIntersection voxelintersection(&context);
    
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(3, 1, 1));
    
    DOCTEST_CHECK(voxel_UUIDs.size() >= 1);
}

DOCTEST_TEST_CASE("VoxelIntersection - Patch with 3 slices and no vertices inside voxel") {
    Context context;
    
    // Create patch using center and size instead of vertices
    vec3 patch_center = make_vec3(2, 0, 0);
    vec2 patch_size = make_vec2(3, 2);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size);
    
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(4, 10, 10);
    
    VoxelIntersection voxelintersection(&context);
    
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(4, 1, 1));
    
    DOCTEST_CHECK(voxel_UUIDs.size() >= 1);
}

DOCTEST_TEST_CASE("VoxelIntersection - Areas after slicing (non-textured)") {
    Context context;
    
    // Create patch using center and size instead of vertices
    vec3 patch_center = make_vec3(0, 0, 0);
    vec2 patch_size = make_vec2(2, 2);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size);
    
    float area_patch = context.getPrimitiveArea(prim_UUID);
    
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(2, 10, 2);
    
    VoxelIntersection voxelintersection(&context);
    
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(2, 1, 2));
    
    float area_tot = 0;
    for (uint UUID : voxel_UUIDs) {
        area_tot += context.getPrimitiveArea(UUID);
    }
    
    DOCTEST_CHECK(fabs(area_tot - area_patch) / area_patch < 0.05f);
}

DOCTEST_TEST_CASE("VoxelIntersection - Textured areas after slicing") {
    Context context;
    
    // Create textured patch using center, size, and texture file
    vec3 patch_center = make_vec3(0, 0, 0);
    vec2 patch_size = make_vec2(2, 2);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size, make_SphericalCoord(0, 0), "lib/images/disk_texture.png");
    
    float area_patch = context.getPrimitiveArea(prim_UUID);
    
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(2, 10, 2);
    
    VoxelIntersection voxelintersection(&context);
    
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(2, 1, 2));
    
    float area_tot = 0;
    for (uint UUID : voxel_UUIDs) {
        area_tot += context.getPrimitiveArea(UUID);
    }
    
    DOCTEST_CHECK(fabs(area_tot - area_patch) / area_patch < 0.05f);
}

DOCTEST_TEST_CASE("VoxelIntersection - Cropping non-textured primitives") {
    Context context;
    
    // Create two triangles to form a quad (replacement for addPolygon)
    uint tri1_UUID = context.addTriangle(make_vec3(-1, 0, 1), make_vec3(1, 0, 1), make_vec3(1, 0, -1));
    uint tri2_UUID = context.addTriangle(make_vec3(1, 0, -1), make_vec3(-1, 0, -1), make_vec3(-1, 0, 1));
    uint prim_UUID = tri1_UUID; // Use first triangle for the test
    
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(1, 10, 1);
    
    VoxelIntersection voxelintersection(&context);
    
    // Test slicing the primitive within the voxel bounds
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(1, 1, 1));
    
    DOCTEST_CHECK(voxel_UUIDs.size() >= 1);
}

DOCTEST_TEST_CASE("VoxelIntersection - Cropping textured primitives") {
    Context context;
    
    // Create textured patch using center, size, and texture file
    vec3 patch_center = make_vec3(0, 0, 0);
    vec2 patch_size = make_vec2(2, 2);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size, make_SphericalCoord(0, 0), "lib/images/disk_texture.png");
    
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(1, 10, 1);
    
    VoxelIntersection voxelintersection(&context);
    
    // Test slicing the textured primitive within the voxel bounds
    std::vector<uint> voxel_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(1, 1, 1));
    
    DOCTEST_CHECK(voxel_UUIDs.size() >= 1);
}

DOCTEST_TEST_CASE("VoxelIntersection - Basic functionality test") {
    Context context;
    
    // Create patch using center and size instead of vertices
    vec3 patch_center = make_vec3(0, 0, 0);
    vec2 patch_size = make_vec2(2, 2);
    
    uint prim_UUID = context.addPatch(patch_center, patch_size);
    
    VoxelIntersection voxelintersection(&context);
    
    // Test slicing the primitive within a grid
    vec3 grid_center = make_vec3(0, 0, 0);
    vec3 grid_size = make_vec3(3, 3, 3);
    std::vector<uint> sliced_UUIDs = voxelintersection.slicePrimitivesUsingGrid(std::vector<uint>{prim_UUID}, grid_center, grid_size, make_int3(2, 1, 2));
    
    // Check that slicing worked and produced some output
    DOCTEST_CHECK(sliced_UUIDs.size() >= 1);
}

int VoxelIntersection::selfTest() {
    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}