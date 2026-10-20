/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

/** bvh_traversal.glsl - Closest-hit traversal of the 8-wide compressed BVH (CWBVH)
 *
 * Stack-based traversal. Leaf primitives are tested through the leaf intersection records
 * (see intersections.glsl); the calling shader must declare bvh_buf and leaf_records_buf.
 */

#ifndef BVH_TRAVERSAL_GLSL
#define BVH_TRAVERSAL_GLSL

#include "intersections.glsl"
#include "texture_mask.glsl"

const uint UINT_MAX = 0xFFFFFFFF;

// Reciprocal of the ray direction that is safe for axis-aligned rays.
//
// A plain 1.0/ray_dir yields +/-inf for any zero direction component. The slab test then computes
// (plane - origin) * inf; when the origin lies exactly on that slab plane the term is 0*inf = NaN,
// which makes the "t_near <= t_far" comparison evaluate to false and silently prunes a subtree that
// the ray actually passes through. This is common with axis-aligned geometry (flat ground at z=0,
// horizontal leaves). Substituting a large finite reciprocal keeps the term finite (0*large = 0),
// so a ray parallel to a slab is correctly treated as always-inside/always-outside that axis.
vec3 safe_inv_dir(vec3 ray_dir) {
    const float LARGE = 1e30;
    return vec3(abs(ray_dir.x) > 1e-30 ? 1.0 / ray_dir.x : LARGE, abs(ray_dir.y) > 1e-30 ? 1.0 / ray_dir.y : LARGE, abs(ray_dir.z) > 1e-30 ? 1.0 / ray_dir.z : LARGE);
}

// ========== CWBVH (8-wide BVH with quantized AABBs) Traversal ==========

// Traversal stack size.
//
// Each pop can push up to 8 children (7 more entries than it removed), and the BVH8 is at most
// ceil(MAX_DEPTH / 3) levels deep because the collapse folds three BVH2 levels into one BVH8
// level. With MAX_DEPTH = 32 that is 11 levels, so the worst case is 11 * 7 + 1 = 78 entries.
// The previous value of 24 could be exceeded by a deep, unbalanced canopy BVH, and the overflow
// path simply discarded the node - geometry vanished from the trace with no diagnostic.
const uint MAX_CWBVH_STACK_DEPTH = 80;

// Bitmask lookup table: slot_masks[i] = (1 << i) - 1, for popcount-based child indexing
const uint slot_masks[8] = uint[8](0x00u, 0x01u, 0x03u, 0x07u, 0x0Fu, 0x1Fu, 0x3Fu, 0x7Fu);

// CWBVH traversal with 8-wide nodes and quantized AABBs
// NOTE: This function expects bvh_buf to contain CWBVH_Node data (128 bytes/node, 32 floats/node)
//
// Optimized for low register pressure on Apple M2 GPU (Metal via MoltenVK):
// - Step 1: Read only quantization params + bounds (16 words), run AABB tests, build hit bitmask
// - Step 2: Read metadata + leaf data only for survivors (deferred reads)
// - No sort arrays — process children in slot order, push internal nodes in near-first order
// - Branchless byte extraction via bitfieldExtract
// This overload also reports where on the facet the closest hit landed, which the camera shader interpolates a per-vertex quantity with. For a triangle the components are the barycentric weights of the
// second and third vertices; for a patch they run from zero at the first vertex to one at the second and fourth.
uint traverse_cwbvh(vec3 ray_origin, vec3 ray_dir, float t_min, uint origin_prim_idx, inout float closest_t, out vec2 closest_uv) {

    closest_uv = vec2(0.0);


    vec3 ray_dir_inv = safe_inv_dir(ray_dir);

    uint stack[MAX_CWBVH_STACK_DEPTH];
    int stack_ptr = 0;

    uint closest_prim = UINT_MAX;
    closest_t = 1e30;

    // Start at root (index 0)
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        uint node_idx = stack[--stack_ptr];

        // CWBVH_Node is 128 bytes = 32 floats
        uint base = node_idx * 32u;

        // ---- Step 1: Read quantization params + bounds (words 0-15 = 64 bytes) ----
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

        // ---- Step 2: Read metadata only for nodes with survivors (words 16, 18) ----
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
                // No else: the stack is sized for the worst case the builder can produce, and
                // BVHBuilder::convertToCWBVH() refuses to emit a tree deeper than that bound, so
                // this cannot overflow. The check is retained only so that a future change to
                // MAX_DEPTH degrades into dropped nodes rather than out-of-bounds stack writes.
            } else {
                // ---- Step 3: Read leaf data on demand (words 20+slot, 28-31) ----
                uint first_prim = floatBitsToUint(bvh_buf.data[base + 20u + child_slot]);

                // Read prim_count from the packed 16-bit-per-child array occupying words 28-31
                // (CWBVH_Node::child_prim_count). first_prim indexes the leaf intersection records,
                // which carry each primitive's index, type and flags.
                uint pc_word = floatBitsToUint(bvh_buf.data[base + 28u + (child_slot >> 1u)]);
                int bit_off = int((child_slot & 1u) * 16u);
                uint prim_count = bitfieldExtract(pc_word, bit_off, 16);

                if (prim_count == 0u) continue;

                for (uint pi = 0u; pi < prim_count; ++pi) {
                    uint slot = first_prim + pi;
                    vec4 anchor_and_index = leaf_records_buf.records[slot * 4u];
                    uint flags = floatBitsToUint(leaf_records_buf.records[slot * 4u + 1u].w);
                    uint prim_idx = floatBitsToUint(anchor_and_index.w);

                    if (prim_idx == origin_prim_idx || (flags & LEAF_RECORD_NEVER_HIT) != 0u) continue;

                    float hit_t;
                    vec2 hit_uv;
                    if (!intersect_leaf_record(slot, ray_origin, ray_dir, t_min, closest_t, hit_t, hit_uv)) continue;

                    // Transparent texels let the ray pass through
                    uint hit_prim_type = ((flags & LEAF_RECORD_TRIANGLE) != 0u) ? 1u : 0u;
                    if (!check_texture_mask(prim_idx, hit_uv, hit_prim_type)) continue;

                    closest_t = hit_t;
                    closest_prim = prim_idx;
                    closest_uv = hit_uv;
                }
            }
        }
    }

    return closest_prim;
}

// Callers that do not need the surface parameters keep the shorter signature.
uint traverse_cwbvh(vec3 ray_origin, vec3 ray_dir, float t_min, uint origin_prim_idx, inout float closest_t) {
    vec2 unused_uv;
    return traverse_cwbvh(ray_origin, ray_dir, t_min, origin_prim_idx, closest_t, unused_uv);
}

#endif // BVH_TRAVERSAL_GLSL
