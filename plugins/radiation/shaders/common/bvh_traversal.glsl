/** bvh_traversal.glsl - BVH traversal using stack
 *
 * Stack-based traversal with ordered near-child-first descent.
 */

#ifndef BVH_TRAVERSAL_GLSL
#define BVH_TRAVERSAL_GLSL

#include "intersections.glsl"
#include "transforms.glsl"

const uint MAX_STACK_DEPTH = 64; // Sufficient for typical BVH scenes
const uint UINT_MAX = 0xFFFFFFFF;

// BVH node structure (matches BVHNode in BVHBuilder.h)
struct BVHNode {
    vec3 aabb_min;
    vec3 aabb_max;
    uint left_child;
    uint right_child;
    uint prim_count;
    uint prim_type;
    uint first_prim;
    uint padding;
};

// Ray-AABB intersection test
bool intersect_aabb(vec3 ray_origin, vec3 ray_dir_inv, vec3 aabb_min, vec3 aabb_max, float t_max) {
    vec3 t_min_vec = (aabb_min - ray_origin) * ray_dir_inv;
    vec3 t_max_vec = (aabb_max - ray_origin) * ray_dir_inv;

    vec3 t1 = min(t_min_vec, t_max_vec);
    vec3 t2 = max(t_min_vec, t_max_vec);

    float t_near = max(max(t1.x, t1.y), t1.z);
    float t_far = min(min(t2.x, t2.y), t2.z);

    return t_near <= t_far && t_far >= 0.0 && t_near <= t_max;
}

// Forward declarations for intersection tests (defined in intersections.glsl)
HitInfo intersect_patch(vec3 ray_origin, vec3 ray_dir, vec4 v0, vec4 v1, vec4 v2, vec4 v3);
HitInfo intersect_triangle(vec3 ray_origin, vec3 ray_dir, vec3 v0, vec3 v1, vec3 v2);

// BVH traversal with stack-based descent and closest-hit tracking
// NOTE: This function expects buffers to be defined in the calling shader:
//   - bvh_buf (BVHNode array)
//   - prim_indices_buf (primitive indices)
//   - patch_vertices_buf (patch vertex data)
//   - triangle_vertices_buf (triangle vertex data)
// Parameters:
//   origin_prim_idx: Index of primitive shooting the ray (to skip self-intersection)
// Returns: closest hit primitive index or UINT_MAX if miss
// Outputs: closest_t (intersection distance), hit_prim_type
uint traverse_bvh(vec3 ray_origin, vec3 ray_dir, float t_min, uint origin_prim_idx, inout float closest_t, out uint hit_prim_type) {

    // Pre-compute inverse ray direction for AABB tests
    vec3 ray_dir_inv = vec3(1.0) / ray_dir;

    // Stack for BVH traversal (64 elements sufficient for typical scenes)
    uint stack[64];
    int stack_ptr = 0;

    uint closest_prim = UINT_MAX;
    closest_t = 1e30;
    hit_prim_type = 0;

    // Start at root (index 0)
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        // Pop node from stack
        uint node_idx = stack[--stack_ptr];

        // Access BVH node from buffer (buffer defined in calling shader)
        // BVHNode C++ layout (48 bytes): float aabb_min[3], float aabb_max[3], uint32_t × 6
        // In GLSL float buffer: 12 floats total (3 aabb_min + 3 aabb_max + 6 uints-as-floats)
        uint base = node_idx * 12; // 12 floats per BVHNode (48 bytes / 4 bytes per float)
        BVHNode node;
        node.aabb_min = vec3(bvh_buf.data[base + 0], bvh_buf.data[base + 1], bvh_buf.data[base + 2]);
        node.aabb_max = vec3(bvh_buf.data[base + 3], bvh_buf.data[base + 4], bvh_buf.data[base + 5]);
        node.left_child = floatBitsToUint(bvh_buf.data[base + 6]);
        node.right_child = floatBitsToUint(bvh_buf.data[base + 7]);
        node.prim_count = floatBitsToUint(bvh_buf.data[base + 8]);
        node.prim_type = floatBitsToUint(bvh_buf.data[base + 9]);
        node.first_prim = floatBitsToUint(bvh_buf.data[base + 10]);
        // Skip padding at base + 11

        // Test ray against node AABB (early out if no intersection or farther than closest hit)
        if (!intersect_aabb(ray_origin, ray_dir_inv, node.aabb_min, node.aabb_max, closest_t)) {
            continue;
        }

        // Check if leaf node (right_child == UINT_MAX indicates leaf)
        if (node.right_child == UINT_MAX) {
            // Leaf node - test all primitives in this leaf
            for (uint i = 0; i < node.prim_count; ++i) {
                uint prim_idx = prim_indices_buf.indices[node.first_prim + i];
                uint prim_type = node.prim_type;

                // Skip self-intersection (ray hitting the surface it originated from)
                if (prim_idx == origin_prim_idx) {
                    continue;
                }

                // Perform ray-primitive intersection based on type
                HitInfo hit;
                hit.hit = false;

                if (prim_type == 0 || prim_type == 3) { // Patch or Tile
                    // Get patch vertices from buffer (canonical/local space)
                    uint vbase = prim_idx * 12; // 12 floats per patch (4 vec3 vertices)
                    vec3 v0_local = vec3(patch_vertices_buf.vertices[vbase + 0], patch_vertices_buf.vertices[vbase + 1], patch_vertices_buf.vertices[vbase + 2]);
                    vec3 v1_local = vec3(patch_vertices_buf.vertices[vbase + 3], patch_vertices_buf.vertices[vbase + 4], patch_vertices_buf.vertices[vbase + 5]);
                    vec3 v2_local = vec3(patch_vertices_buf.vertices[vbase + 6], patch_vertices_buf.vertices[vbase + 7], patch_vertices_buf.vertices[vbase + 8]);
                    vec3 v3_local = vec3(patch_vertices_buf.vertices[vbase + 9], patch_vertices_buf.vertices[vbase + 10], patch_vertices_buf.vertices[vbase + 11]);

                    // Transform vertices to world space
                    float hit_transform[16];
                    for (uint t = 0; t < 16; ++t) {
                        hit_transform[t] = transform_buf.matrices[prim_idx * 16 + t];
                    }
                    mat4 transform = floats_to_mat4(hit_transform);

                    vec4 v0 = vec4(transform_point(transform, v0_local), 1.0);
                    vec4 v1 = vec4(transform_point(transform, v1_local), 1.0);
                    vec4 v2 = vec4(transform_point(transform, v2_local), 1.0);
                    vec4 v3 = vec4(transform_point(transform, v3_local), 1.0);

                    hit = intersect_patch(ray_origin, ray_dir, v0, v1, v2, v3);

                } else if (prim_type == 1) { // Triangle
                    // Get triangle vertices from buffer (canonical/local space)
                    uint vbase = prim_idx * 9; // 9 floats per triangle (3 vec3 vertices)
                    vec3 v0_local = vec3(triangle_vertices_buf.vertices[vbase + 0], triangle_vertices_buf.vertices[vbase + 1], triangle_vertices_buf.vertices[vbase + 2]);
                    vec3 v1_local = vec3(triangle_vertices_buf.vertices[vbase + 3], triangle_vertices_buf.vertices[vbase + 4], triangle_vertices_buf.vertices[vbase + 5]);
                    vec3 v2_local = vec3(triangle_vertices_buf.vertices[vbase + 6], triangle_vertices_buf.vertices[vbase + 7], triangle_vertices_buf.vertices[vbase + 8]);

                    // Transform vertices to world space
                    float hit_transform[16];
                    for (uint t = 0; t < 16; ++t) {
                        hit_transform[t] = transform_buf.matrices[prim_idx * 16 + t];
                    }
                    mat4 transform = floats_to_mat4(hit_transform);

                    vec3 v0 = transform_point(transform, v0_local);
                    vec3 v1 = transform_point(transform, v1_local);
                    vec3 v2 = transform_point(transform, v2_local);

                    hit = intersect_triangle(ray_origin, ray_dir, v0, v1, v2);

                } else if (prim_type == 2) { // Disk
                    // TODO: Implement disk intersection
                    continue;

                } else if (prim_type == 4) { // Voxel
                    // TODO: Implement voxel intersection
                    continue;
                }

                // Update closest hit if this intersection is nearer
                if (hit.hit && hit.t < closest_t && hit.t > t_min) {
                    closest_t = hit.t;
                    closest_prim = prim_idx;
                    hit_prim_type = prim_type;
                }
            }
        } else {
            // Internal node - push children onto stack
            // Push far child first, then near child (so near child is processed first)
            // For now, simple approach: just push both children
            if (stack_ptr + 2 <= 64) {
                stack[stack_ptr++] = node.right_child;
                stack[stack_ptr++] = node.left_child;
            } else {
                // Stack overflow - shouldn't happen with 64 elements for typical scenes
                // Return current best hit (or miss)
                break;
            }
        }
    }

    return closest_prim;
}

#endif // BVH_TRAVERSAL_GLSL
