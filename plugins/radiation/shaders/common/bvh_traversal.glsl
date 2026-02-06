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

// Placeholder BVH traversal (to be implemented in Phase 1+)
// Returns: hit primitive index or UINT_MAX if miss
uint traverse_bvh(vec3 ray_origin, vec3 ray_dir, float t_min, float t_max) {
    // TODO: Implement full traversal in Phase 1
    return UINT_MAX; // Miss
}

#endif // BVH_TRAVERSAL_GLSL
