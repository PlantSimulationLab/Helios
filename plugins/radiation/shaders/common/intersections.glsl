/** intersections.glsl - Ray-primitive intersection tests
 *
 * Primitive types reaching the software BVH:
 * - 0: Patch (quadrilateral)
 * - 1: Triangle
 * - 3: Tile (textured quadrilateral, intersected as a patch)
 *
 * RadiationModel::buildGeometryData() only ever emits types 0, 1 and 4, and
 * VulkanComputeBackend::updateGeometry() rejects type 4 (voxel) because this backend has no
 * voxel intersector. Types 2 (disk) and 5 (bbox) are never emitted at all.
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

// Ray-patch (quadrilateral) intersection
// Vertices layout: v0, v1, v2, v3 (counter-clockwise)
HitInfo intersect_patch(vec3 ray_origin, vec3 ray_dir, vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;

    // Compute patch plane. The degeneracy test must happen BEFORE normalizing: a zero-area
    // patch gives cross() == 0, normalize() then yields a NaN normal, and the parallel-ray
    // guard below cannot reject it because abs(NaN) < 1e-8 evaluates to false.
    vec3 anchor = v0.xyz;
    vec3 plane_normal = cross(v1.xyz - v0.xyz, v2.xyz - v0.xyz);
    float plane_normal_length = length(plane_normal);
    if (plane_normal_length < 1e-20) {
        return hit; // Degenerate patch (zero area)
    }
    vec3 normal = plane_normal / plane_normal_length;

    // Ray-plane intersection
    float denom = dot(ray_dir, normal);
    if (abs(denom) < 1e-8) {
        return hit; // Ray parallel to patch
    }

    float t = dot(anchor - ray_origin, normal) / denom;

    if (t < 1e-8 || t > 1e8) {
        return hit; // Behind ray or too far
    }

    // Check if intersection point is inside quadrilateral
    vec3 p = ray_origin + ray_dir * t;
    vec3 d = p - anchor;

    vec3 a = v1.xyz - v0.xyz;
    vec3 b = v3.xyz - v0.xyz;

    float ddota = dot(d, a);
    float ddotb = dot(d, b);

    // The squared edge lengths double as the inside-quad bounds and as the UV divisors, so a
    // zero-length edge cannot reach the division: ddota > 0.0 && ddota < 0.0 is never true.
    float a_length_squared = dot(a, a);
    float b_length_squared = dot(b, b);

    if (ddota > 0.0 && ddota < a_length_squared && ddotb > 0.0 && ddotb < b_length_squared) {
        // Inside quad
        hit.hit = true;
        hit.t = t;
        hit.normal = normal;

        // Compute UV coordinates
        hit.uv = vec2(ddota / a_length_squared, ddotb / b_length_squared);
    }

    return hit;
}

// Ray-triangle intersection using Möller-Trumbore algorithm
HitInfo intersect_triangle(vec3 ray_origin, vec3 ray_dir, vec3 v0, vec3 v1, vec3 v2) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(ray_dir, edge2);
    float a = dot(edge1, h);

    if (abs(a) < 1e-8) {
        return hit; // Ray parallel to triangle
    }

    float f = 1.0 / a;
    vec3 s = ray_origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return hit;
    }

    vec3 q = cross(s, edge1);
    float v = f * dot(ray_dir, q);

    if (v < 0.0 || u + v > 1.0) {
        return hit;
    }

    float t = f * dot(edge2, q);

    if (t > 1e-8 && t < 1e8) {
        hit.hit = true;
        hit.t = t;
        hit.uv = vec2(u, v);
        // Guard the normalize: a sliver triangle can pass the determinant test above while
        // cross(edge1, edge2) underflows to zero, and normalize(vec3(0)) is NaN.
        vec3 face_normal = cross(edge1, edge2);
        float face_normal_length = length(face_normal);
        hit.normal = (face_normal_length > 1e-20) ? (face_normal / face_normal_length) : vec3(0.0, 0.0, 1.0);
    }

    return hit;
}

#endif // INTERSECTIONS_GLSL
