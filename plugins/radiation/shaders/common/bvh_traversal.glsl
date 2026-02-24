/** bvh_traversal.glsl - BVH traversal using stack
 *
 * Stack-based traversal with ordered near-child-first descent.
 */

#ifndef BVH_TRAVERSAL_GLSL
#define BVH_TRAVERSAL_GLSL

#include "intersections.glsl"
#include "texture_mask.glsl"

const uint MAX_STACK_DEPTH = 36; // BVH MAX_DEPTH=32, so 36 provides headroom
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
    uint split_axis;
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
// Outputs: closest_t (intersection distance), hit_prim_type, hit_uv (UV coordinates at hit point)
uint traverse_bvh(vec3 ray_origin, vec3 ray_dir, float t_min, uint origin_prim_idx, inout float closest_t, out uint hit_prim_type, out vec2 hit_uv) {

    // Pre-compute inverse ray direction for AABB tests
    vec3 ray_dir_inv = vec3(1.0) / ray_dir;

    // Stack for BVH traversal (36 elements: BVH MAX_DEPTH=32, +4 headroom)
    uint stack[36];
    int stack_ptr = 0;

    uint closest_prim = UINT_MAX;
    closest_t = 1e30;
    hit_prim_type = 0;
    hit_uv = vec2(0.0, 0.0);

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
        node.split_axis = floatBitsToUint(bvh_buf.data[base + 11]);

        // Test ray against node AABB (early out if no intersection or farther than closest hit)
        if (!intersect_aabb(ray_origin, ray_dir_inv, node.aabb_min, node.aabb_max, closest_t)) {
            continue;
        }

        // Check if leaf node (right_child == UINT_MAX indicates leaf)
        if (node.right_child == UINT_MAX) {
            // Leaf node - test all primitives in this leaf
            for (uint i = 0; i < node.prim_count; ++i) {
                uint prim_idx = prim_indices_buf.indices[node.first_prim + i];
                uint prim_type = prim_types_buf.types[prim_idx];

                // Skip self-intersection (ray hitting the surface it originated from)
                if (prim_idx == origin_prim_idx) {
                    continue;
                }

                // Perform ray-primitive intersection based on type
                HitInfo hit;
                hit.hit = false;

                if (prim_type == 0 || prim_type == 3) { // Patch or Tile
                    // Get pre-transformed world-space patch vertices from buffer
                    uint vbase = prim_idx * 12; // 12 floats per patch (4 vec3 vertices)
                    vec4 v0 = vec4(patch_vertices_buf.vertices[vbase + 0], patch_vertices_buf.vertices[vbase + 1], patch_vertices_buf.vertices[vbase + 2], 1.0);
                    vec4 v1 = vec4(patch_vertices_buf.vertices[vbase + 3], patch_vertices_buf.vertices[vbase + 4], patch_vertices_buf.vertices[vbase + 5], 1.0);
                    vec4 v2 = vec4(patch_vertices_buf.vertices[vbase + 6], patch_vertices_buf.vertices[vbase + 7], patch_vertices_buf.vertices[vbase + 8], 1.0);
                    vec4 v3 = vec4(patch_vertices_buf.vertices[vbase + 9], patch_vertices_buf.vertices[vbase + 10], patch_vertices_buf.vertices[vbase + 11], 1.0);

                    hit = intersect_patch(ray_origin, ray_dir, v0, v1, v2, v3);

                } else if (prim_type == 1) { // Triangle
                    // Get pre-transformed world-space triangle vertices from buffer
                    uint vbase = prim_idx * 9; // 9 floats per triangle (3 vec3 vertices)
                    vec3 v0 = vec3(triangle_vertices_buf.vertices[vbase + 0], triangle_vertices_buf.vertices[vbase + 1], triangle_vertices_buf.vertices[vbase + 2]);
                    vec3 v1 = vec3(triangle_vertices_buf.vertices[vbase + 3], triangle_vertices_buf.vertices[vbase + 4], triangle_vertices_buf.vertices[vbase + 5]);
                    vec3 v2 = vec3(triangle_vertices_buf.vertices[vbase + 6], triangle_vertices_buf.vertices[vbase + 7], triangle_vertices_buf.vertices[vbase + 8]);

                    hit = intersect_triangle(ray_origin, ray_dir, v0, v1, v2);

                } else if (prim_type == 2) { // Disk
                    // TODO: Implement disk intersection
                    continue;

                } else if (prim_type == 4) { // Voxel
                    // TODO: Implement voxel intersection
                    continue;
                }

                // Update closest hit if this intersection is nearer AND passes texture mask
                if (hit.hit && hit.t < closest_t && hit.t > t_min) {
                    // Check texture mask at hit point (transparent texels = ray passes through)
                    if (!check_texture_mask(prim_idx, hit.uv, prim_type)) {
                        continue; // Skip this primitive - ray passes through transparent texel
                    }

                    closest_t = hit.t;
                    closest_prim = prim_idx;
                    hit_prim_type = prim_type;
                    hit_uv = hit.uv;
                }
            }
        } else {
            // Internal node - push children onto stack with near-child-first ordering
            // Use split axis and ray direction to determine which child is nearer
            if (stack_ptr + 2 <= MAX_STACK_DEPTH) {
                bool near_is_left = ray_dir_inv[node.split_axis] > 0.0;
                uint near_child = near_is_left ? node.left_child : node.right_child;
                uint far_child = near_is_left ? node.right_child : node.left_child;
                stack[stack_ptr++] = far_child;   // Far child pushed first (processed last)
                stack[stack_ptr++] = near_child;  // Near child pushed last (processed first)
            } else {
                break;
            }
        }
    }

    return closest_prim;
}

#endif // BVH_TRAVERSAL_GLSL
