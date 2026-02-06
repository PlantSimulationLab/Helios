/** intersections.glsl - Ray-primitive intersection tests
 *
 * Supports 6 primitive types:
 * - 0: Patch (quadrilateral)
 * - 1: Triangle
 * - 2: Disk
 * - 3: Tile (textured quadrilateral)
 * - 4: Voxel (axis-aligned box)
 * - 5: BBox (oriented bounding box)
 */

#ifndef INTERSECTIONS_GLSL
#define INTERSECTIONS_GLSL

// Intersection result
struct HitInfo {
    bool hit;
    float t;
    vec2 uv;
    vec3 normal;
    uint prim_index;
};

// Placeholder intersection functions (to be implemented in Phase 1+)

HitInfo intersect_patch(vec3 ray_origin, vec3 ray_dir, uint prim_index) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;
    return hit;
}

HitInfo intersect_triangle(vec3 ray_origin, vec3 ray_dir, uint prim_index) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;
    return hit;
}

HitInfo intersect_disk(vec3 ray_origin, vec3 ray_dir, uint prim_index) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;
    return hit;
}

HitInfo intersect_tile(vec3 ray_origin, vec3 ray_dir, uint prim_index) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;
    return hit;
}

HitInfo intersect_voxel(vec3 ray_origin, vec3 ray_dir, uint prim_index) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;
    return hit;
}

HitInfo intersect_bbox(vec3 ray_origin, vec3 ray_dir, uint prim_index) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;
    return hit;
}

#endif // INTERSECTIONS_GLSL
