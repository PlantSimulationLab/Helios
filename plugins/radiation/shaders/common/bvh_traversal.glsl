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

// ========== CWBVH (8-wide BVH with quantized AABBs) Traversal ==========

const uint MAX_CWBVH_STACK_DEPTH = 24;

// Bitmask lookup table: slot_masks[i] = (1 << i) - 1, for popcount-based child indexing
const uint slot_masks[8] = uint[8](0x00u, 0x01u, 0x03u, 0x07u, 0x0Fu, 0x1Fu, 0x3Fu, 0x7Fu);

// CWBVH traversal with 8-wide nodes and quantized AABBs
// Same signature as traverse_bvh() for drop-in replacement.
// NOTE: This function expects bvh_buf to contain CWBVH_Node data (128 bytes/node, 32 floats/node)
//
// Optimized for low register pressure on Apple M2 GPU (Metal via MoltenVK):
// - Phase 1: Read only quantization params + bounds (16 words), run AABB tests, build hit bitmask
// - Phase 2: Read metadata + leaf data only for survivors (deferred reads)
// - No sort arrays — process children in slot order, push internal nodes in near-first order
// - Branchless byte extraction via bitfieldExtract
uint traverse_cwbvh(vec3 ray_origin, vec3 ray_dir, float t_min, uint origin_prim_idx, inout float closest_t, out uint hit_prim_type, out vec2 hit_uv) {

    vec3 ray_dir_inv = vec3(1.0) / ray_dir;

    uint stack[MAX_CWBVH_STACK_DEPTH];
    int stack_ptr = 0;

    uint closest_prim = UINT_MAX;
    closest_t = 1e30;
    hit_prim_type = 0;
    hit_uv = vec2(0.0, 0.0);

    // Start at root (index 0)
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        uint node_idx = stack[--stack_ptr];

        // CWBVH_Node is 128 bytes = 32 floats
        uint base = node_idx * 32u;

        // ---- Phase 1: Read quantization params + bounds (words 0-15 = 64 bytes) ----
        vec3 p = vec3(bvh_buf.data[base], bvh_buf.data[base + 1], bvh_buf.data[base + 2]);
        uint e_packed = floatBitsToUint(bvh_buf.data[base + 3]);

        // Decode exponents: biased IEEE754 exponents -> scale = 2^(biased - 127)
        float scale_x = uintBitsToFloat(((e_packed      ) & 0xFFu) << 23);
        float scale_y = uintBitsToFloat(((e_packed >>  8u) & 0xFFu) << 23);
        float scale_z = uintBitsToFloat(((e_packed >> 16u) & 0xFFu) << 23);

        // Quantized child AABBs: 6 axes x 2 words each = 12 words
        // qmin_x[8]: words 4-5, qmin_y[8]: words 6-7, qmin_z[8]: words 8-9
        // qmax_x[8]: words 10-11, qmax_y[8]: words 12-13, qmax_z[8]: words 14-15
        uint qmin_x_w0 = floatBitsToUint(bvh_buf.data[base + 4]);
        uint qmin_x_w1 = floatBitsToUint(bvh_buf.data[base + 5]);
        uint qmin_y_w0 = floatBitsToUint(bvh_buf.data[base + 6]);
        uint qmin_y_w1 = floatBitsToUint(bvh_buf.data[base + 7]);
        uint qmin_z_w0 = floatBitsToUint(bvh_buf.data[base + 8]);
        uint qmin_z_w1 = floatBitsToUint(bvh_buf.data[base + 9]);
        uint qmax_x_w0 = floatBitsToUint(bvh_buf.data[base + 10]);
        uint qmax_x_w1 = floatBitsToUint(bvh_buf.data[base + 11]);
        uint qmax_y_w0 = floatBitsToUint(bvh_buf.data[base + 12]);
        uint qmax_y_w1 = floatBitsToUint(bvh_buf.data[base + 13]);
        uint qmax_z_w0 = floatBitsToUint(bvh_buf.data[base + 14]);
        uint qmax_z_w1 = floatBitsToUint(bvh_buf.data[base + 15]);

        // Test all 8 children against ray using branchless byte extraction.
        // Build a bitmask of surviving children (no arrays needed).
        uint hit_mask = 0u;

        for (uint i = 0u; i < 8u; i++) {
            // Branchless byte extraction: select word based on child index
            // Children 0-3 use w0, children 4-7 use w1
            uint w0_or_w1 = i >> 2u;  // 0 for children 0-3, 1 for 4-7
            int bit_offset = int((i & 3u) * 8u);

            uint qmin_x_val = bitfieldExtract(w0_or_w1 == 0u ? qmin_x_w0 : qmin_x_w1, bit_offset, 8);
            uint qmin_y_val = bitfieldExtract(w0_or_w1 == 0u ? qmin_y_w0 : qmin_y_w1, bit_offset, 8);
            uint qmin_z_val = bitfieldExtract(w0_or_w1 == 0u ? qmin_z_w0 : qmin_z_w1, bit_offset, 8);
            uint qmax_x_val = bitfieldExtract(w0_or_w1 == 0u ? qmax_x_w0 : qmax_x_w1, bit_offset, 8);
            uint qmax_y_val = bitfieldExtract(w0_or_w1 == 0u ? qmax_y_w0 : qmax_y_w1, bit_offset, 8);
            uint qmax_z_val = bitfieldExtract(w0_or_w1 == 0u ? qmax_z_w0 : qmax_z_w1, bit_offset, 8);

            // Reconstruct child AABB (empty slots have inverted bounds: qmin=255, qmax=0)
            vec3 child_min = p + vec3(float(qmin_x_val) * scale_x, float(qmin_y_val) * scale_y, float(qmin_z_val) * scale_z);
            vec3 child_max = p + vec3(float(qmax_x_val) * scale_x, float(qmax_y_val) * scale_y, float(qmax_z_val) * scale_z);

            // Ray-AABB slab test
            vec3 t1 = (child_min - ray_origin) * ray_dir_inv;
            vec3 t2 = (child_max - ray_origin) * ray_dir_inv;
            vec3 t_lo = min(t1, t2);
            vec3 t_hi = max(t1, t2);
            float t_near_val = max(max(t_lo.x, t_lo.y), t_lo.z);
            float t_far_val = min(min(t_hi.x, t_hi.y), t_hi.z);

            if (t_near_val <= t_far_val && t_far_val >= 0.0 && t_near_val <= closest_t) {
                hit_mask |= (1u << i);
            }
        }

        // p, scales, and bounds words are no longer needed — compiler can free those registers

        if (hit_mask == 0u) continue; // No children hit — skip metadata reads

        // ---- Phase 2: Read metadata only for nodes with survivors (words 16, 18) ----
        uint imask = floatBitsToUint(bvh_buf.data[base + 16]) & 0xFFu;
        uint base_index_child = floatBitsToUint(bvh_buf.data[base + 18]);

        // Process surviving children in slot order.
        // Internal children are pushed to stack; leaf children are intersected immediately.
        // No sort needed — for incoherent diffuse rays, ordering provides minimal benefit.
        // For coherent rays (direct), slot order approximates octant-based spatial ordering.
        uint remaining = hit_mask;
        while (remaining != 0u) {
            uint child_slot = findLSB(remaining);
            remaining &= remaining - 1u; // Clear lowest set bit

            if ((imask & (1u << child_slot)) != 0u) {
                // Internal child: compute array index using popcount
                uint child_array_idx = base_index_child + bitCount(imask & slot_masks[child_slot]);
                if (stack_ptr < int(MAX_CWBVH_STACK_DEPTH)) {
                    stack[stack_ptr++] = child_array_idx;
                }
            } else {
                // ---- Phase 3: Read leaf data on demand (words 20+slot, 28-31) ----
                uint first_prim = floatBitsToUint(bvh_buf.data[base + 20u + child_slot]);

                // Read prim_count and prim_type from packed byte arrays
                uint pc_word = floatBitsToUint(bvh_buf.data[base + 28u + (child_slot >> 2u)]);
                uint pt_word = floatBitsToUint(bvh_buf.data[base + 30u + (child_slot >> 2u)]);
                int bit_off = int((child_slot & 3u) * 8u);
                uint prim_count = bitfieldExtract(pc_word, bit_off, 8);

                if (prim_count == 0u) continue;

                for (uint pi = 0u; pi < prim_count; ++pi) {
                    uint prim_idx = prim_indices_buf.indices[first_prim + pi];
                    uint actual_prim_type = prim_types_buf.types[prim_idx];

                    if (prim_idx == origin_prim_idx) continue;

                    HitInfo hit;
                    hit.hit = false;

                    if (actual_prim_type == 0u || actual_prim_type == 3u) { // Patch or Tile
                        uint vbase = prim_idx * 12u;
                        vec4 v0 = vec4(patch_vertices_buf.vertices[vbase], patch_vertices_buf.vertices[vbase + 1], patch_vertices_buf.vertices[vbase + 2], 1.0);
                        vec4 v1 = vec4(patch_vertices_buf.vertices[vbase + 3], patch_vertices_buf.vertices[vbase + 4], patch_vertices_buf.vertices[vbase + 5], 1.0);
                        vec4 v2 = vec4(patch_vertices_buf.vertices[vbase + 6], patch_vertices_buf.vertices[vbase + 7], patch_vertices_buf.vertices[vbase + 8], 1.0);
                        vec4 v3 = vec4(patch_vertices_buf.vertices[vbase + 9], patch_vertices_buf.vertices[vbase + 10], patch_vertices_buf.vertices[vbase + 11], 1.0);
                        hit = intersect_patch(ray_origin, ray_dir, v0, v1, v2, v3);

                    } else if (actual_prim_type == 1u) { // Triangle
                        uint vbase = prim_idx * 9u;
                        vec3 v0 = vec3(triangle_vertices_buf.vertices[vbase], triangle_vertices_buf.vertices[vbase + 1], triangle_vertices_buf.vertices[vbase + 2]);
                        vec3 v1 = vec3(triangle_vertices_buf.vertices[vbase + 3], triangle_vertices_buf.vertices[vbase + 4], triangle_vertices_buf.vertices[vbase + 5]);
                        vec3 v2 = vec3(triangle_vertices_buf.vertices[vbase + 6], triangle_vertices_buf.vertices[vbase + 7], triangle_vertices_buf.vertices[vbase + 8]);
                        hit = intersect_triangle(ray_origin, ray_dir, v0, v1, v2);

                    } else if (actual_prim_type == 2u) { // Disk
                        continue;
                    } else if (actual_prim_type == 4u) { // Voxel
                        continue;
                    }

                    if (hit.hit && hit.t < closest_t && hit.t > t_min) {
                        if (!check_texture_mask(prim_idx, hit.uv, actual_prim_type)) {
                            continue;
                        }
                        closest_t = hit.t;
                        closest_prim = prim_idx;
                        hit_prim_type = actual_prim_type;
                        hit_uv = hit.uv;
                    }
                }
            }
        }
    }

    return closest_prim;
}

#endif // BVH_TRAVERSAL_GLSL
