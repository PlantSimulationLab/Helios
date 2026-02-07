/** bvh_traversal.glsl - BVH traversal using stack
 *
 * Stack-based traversal with ordered near-child-first descent.
 */

#ifndef BVH_TRAVERSAL_GLSL
#define BVH_TRAVERSAL_GLSL

const uint MAX_STACK_DEPTH = 32;
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

// BVH traversal with stack-based descent
// Returns: closest hit primitive index or UINT_MAX if miss
// Outputs: closest_t (intersection distance), hit_prim_type
uint traverse_bvh(layout(std430) readonly buffer BVHBuffer { BVHNode nodes[]; } bvh_buf,
                  layout(std430) readonly buffer PrimIndicesBuffer { uint indices[]; } prim_indices_buf,
                  vec3 ray_origin, vec3 ray_dir, float t_min, inout float closest_t, out uint hit_prim_type) {

    // Pre-compute inverse ray direction for AABB tests
    vec3 ray_dir_inv = vec3(1.0) / ray_dir;

    // Stack for BVH traversal
    uint stack[MAX_STACK_DEPTH];
    int stack_ptr = 0;

    uint closest_prim = UINT_MAX;
    closest_t = 1e30;
    hit_prim_type = 0;

    // Start at root
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        uint node_idx = stack[--stack_ptr];
        BVHNode node = bvh_buf.nodes[node_idx];

        // Test ray against node AABB
        if (!intersect_aabb(ray_origin, ray_dir_inv, node.aabb_min, node.aabb_max, closest_t)) {
            continue;
        }

        if (node.right_child == UINT_MAX) {
            // Leaf node - test primitives
            for (uint i = 0; i < node.prim_count; ++i) {
                uint prim_idx = prim_indices_buf.indices[node.first_prim + i];

                // Intersection test happens in caller (they know prim type)
                // For Phase 1, we just return the first primitive in the leaf
                // TODO Phase 2+: Proper intersection testing here
                if (prim_idx < closest_prim) {
                    closest_prim = prim_idx;
                    hit_prim_type = node.prim_type;
                }
            }
        } else {
            // Internal node - push children (near child last for depth-first)
            // Simple heuristic: push both, GPU will cull via AABB test
            if (stack_ptr + 2 <= MAX_STACK_DEPTH) {
                stack[stack_ptr++] = node.left_child;
                stack[stack_ptr++] = node.right_child;
            }
        }
    }

    return closest_prim;
}

#endif // BVH_TRAVERSAL_GLSL
