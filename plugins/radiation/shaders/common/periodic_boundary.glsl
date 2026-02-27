/** periodic_boundary.glsl - Periodic boundary condition utilities
 *
 * Implements ray-bbox face intersection and coordinate wrapping for periodic
 * boundary conditions. Bbox faces are thin rectangular walls at domain edges
 * that catch rays exiting the domain and wrap them to the opposite side.
 *
 * Architecture: Bboxes are checked SEPARATELY from the BVH (not integrated).
 * After each BVH traversal, this code tests the 2-4 bbox faces for closer hits.
 * This matches the OptiX backend where bboxes are a separate geometry group.
 *
 * Reference: rayHit.cu:41-65 (wrapping logic), rayGeneration.cu:238-246 (wrap loop)
 */

#ifndef PERIODIC_BOUNDARY_GLSL
#define PERIODIC_BOUNDARY_GLSL

// NOTE: The calling shader must declare bbox_verts_buf before including this file:
//   layout(set = 0, binding = 17) readonly buffer BboxVerticesBuffer {
//       float vertices[]; // 12 floats per face (4 vec3 vertices)
//   } bbox_verts_buf;

/**
 * Ray-rectangle intersection for one bbox face.
 *
 * Uses the same plane + point-in-quad algorithm as intersect_patch().
 * Returns t value at intersection, or -1.0 on miss.
 */
float intersect_bbox_face(vec3 ray_origin, vec3 ray_dir, uint face_idx) {
    uint base = face_idx * 12; // 4 vertices * 3 floats each
    vec3 v0 = vec3(bbox_verts_buf.vertices[base + 0], bbox_verts_buf.vertices[base + 1], bbox_verts_buf.vertices[base + 2]);
    vec3 v1 = vec3(bbox_verts_buf.vertices[base + 3], bbox_verts_buf.vertices[base + 4], bbox_verts_buf.vertices[base + 5]);
    vec3 v2 = vec3(bbox_verts_buf.vertices[base + 6], bbox_verts_buf.vertices[base + 7], bbox_verts_buf.vertices[base + 8]);
    vec3 v3 = vec3(bbox_verts_buf.vertices[base + 9], bbox_verts_buf.vertices[base + 10], bbox_verts_buf.vertices[base + 11]);

    vec3 anchor = v0;
    vec3 normal = normalize(cross(v1 - v0, v2 - v0));

    float denom = dot(ray_dir, normal);
    if (abs(denom) < 1e-8) return -1.0; // Ray parallel to face

    float t = dot(anchor - ray_origin, normal) / denom;
    if (t <= 1e-5 || t > 1e8) return -1.0; // Behind ray or too far

    // Check if hit point is inside the rectangle
    vec3 p = ray_origin + ray_dir * t;
    vec3 d = p - anchor;
    vec3 a = v1 - v0;
    vec3 b = v3 - v0;

    float ddota = dot(d, a);
    float ddotb = dot(d, b);

    if (ddota > 0.0 && ddota < dot(a, a) && ddotb > 0.0 && ddotb < dot(b, b)) {
        return t;
    }
    return -1.0;
}

/**
 * Check all bbox faces for periodic boundary intersection and compute wrapped origin.
 *
 * Tests each bbox face against the ray. If the closest bbox hit is nearer than
 * the closest real geometry hit (bvh_closest_t), computes the wrapped ray origin
 * by translating the hit point across the domain.
 *
 * Wrapping logic matches OptiX rayHit.cu:41-65:
 * - Hit at -x boundary → wrap to +x (add width_x - eps)
 * - Hit at +x boundary → wrap to -x (subtract width_x - eps)
 * - Same for Y boundaries
 *
 * Returns true if a periodic boundary was hit (caller should re-trace from wrapped_origin).
 */
bool check_periodic_boundary(
    vec3 ray_origin, vec3 ray_dir, float bvh_closest_t,
    uint periodic_flag_x, uint periodic_flag_y, uint bbox_count,
    float domain_xmin, float domain_xmax, float domain_ymin, float domain_ymax,
    out vec3 wrapped_origin
) {
    float bbox_t = bvh_closest_t; // Only consider bbox hits closer than real geometry
    int closest_face = -1;

    for (uint b = 0; b < bbox_count; ++b) {
        float t = intersect_bbox_face(ray_origin, ray_dir, b);
        if (t > 0.0 && t < bbox_t) {
            bbox_t = t;
            closest_face = int(b);
        }
    }

    if (closest_face < 0) return false; // No bbox hit closer than real geometry

    // Compute hit position on bbox face
    vec3 hit_pos = ray_origin + ray_dir * bbox_t;

    // Compute wrapped position (matches OptiX rayHit.cu:41-65)
    float eps = 1e-5;
    float width_x = domain_xmax - domain_xmin;
    float width_y = domain_ymax - domain_ymin;

    wrapped_origin = hit_pos;
    if (periodic_flag_x == 1 && abs(hit_pos.x - domain_xmin) <= eps) {
        wrapped_origin.x += width_x - eps; // -x boundary → wrap to +x
    } else if (periodic_flag_x == 1 && abs(hit_pos.x - domain_xmax) <= eps) {
        wrapped_origin.x -= width_x - eps; // +x boundary → wrap to -x
    } else if (periodic_flag_y == 1 && abs(hit_pos.y - domain_ymin) <= eps) {
        wrapped_origin.y += width_y - eps; // -y boundary → wrap to +y
    } else if (periodic_flag_y == 1 && abs(hit_pos.y - domain_ymax) <= eps) {
        wrapped_origin.y -= width_y - eps; // +y boundary → wrap to -y
    }

    return true;
}

#endif // PERIODIC_BOUNDARY_GLSL
