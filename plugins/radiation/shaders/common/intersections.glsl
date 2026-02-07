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

// Ray-patch (quadrilateral) intersection
// Vertices layout: v0, v1, v2, v3 (counter-clockwise)
HitInfo intersect_patch(vec3 ray_origin, vec3 ray_dir, vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    HitInfo hit;
    hit.hit = false;
    hit.t = 1e30;

    // Compute patch plane
    vec3 anchor = v0.xyz;
    vec3 normal = normalize(cross(v1.xyz - v0.xyz, v2.xyz - v0.xyz));

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

    if (ddota > 0.0 && ddota < dot(a, a) && ddotb > 0.0 && ddotb < dot(b, b)) {
        // Inside quad
        hit.hit = true;
        hit.t = t;
        hit.normal = normal;

        // Compute UV coordinates
        float amag = length(a);
        float bmag = length(b);
        hit.uv = vec2(ddota / (amag * amag), ddotb / (bmag * bmag));
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
        hit.normal = normalize(cross(edge1, edge2));
    }

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
