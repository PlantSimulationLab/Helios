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

/** intersections.glsl - Ray-primitive intersection against the BVH leaf intersection records
 *
 * Primitive types reaching the software BVH:
 * - 0: Patch (quadrilateral)
 * - 1: Triangle
 * - 3: Tile (textured quadrilateral, intersected as a patch)
 *
 * RadiationModel::buildGeometryData() only ever emits types 0, 1 and 4, and
 * VulkanComputeBackend::updateGeometry() rejects type 4 (voxel) because this backend has no
 * voxel intersector. Types 2 (disk) and 5 (bbox) are never emitted at all.
 *
 * VulkanComputeBackend::updateGeometry() stores one 64-byte record per BVH leaf slot, in the order
 * the leaves reference them, so a leaf's primitives are read contiguously and without a lookup
 * through a primitive index buffer. Record s occupies leaf_records_buf.records[4s .. 4s+3]:
 *
 *   [4s + 0]  xyz = anchor vertex v0 (world space)      w = primitive index (uint bits)
 *   [4s + 1]  xyz = dual edge vector u                  w = flags (uint bits, LEAF_RECORD_*)
 *   [4s + 2]  xyz = dual edge vector v                  w = unused
 *   [4s + 3]  xyz = unit plane normal                   w = unused
 *
 * For a point p in the primitive's plane, (dot(p - v0, u), dot(p - v0, v)) are its coordinates along
 * the primitive's two edges from v0. For a patch or tile the edges run to v1 and v3 and the
 * coordinates are the parametric surface UV, from 0 at v0 to 1 at the far edges. For a triangle the
 * edges run to v1 and v2 and the coordinates are the barycentric weights of v1 and v2. These are the
 * hit coordinates the texture-mask lookup and the camera's per-vertex interpolation expect.
 * Patches and triangles therefore share one test that differs only in the final inside condition,
 * so the traversal loop carries no per-type branch.
 *
 * The calling shader must declare leaf_records_buf (set 0, binding 8) before including this file.
 */

#ifndef INTERSECTIONS_GLSL
#define INTERSECTIONS_GLSL

const uint LEAF_RECORD_TRIANGLE = 1u; //!< Record is a triangle (otherwise a patch or tile)
const uint LEAF_RECORD_NEVER_HIT = 2u; //!< Never intersected: invisible (twosided_flag >= 2) or zero-area primitive

// Ray test against leaf record `slot`. On a hit with t_min < t < t_max, returns true and writes the
// hit distance and the surface coordinates described above.
bool intersect_leaf_record(uint slot, vec3 ray_origin, vec3 ray_dir, float t_min, float t_max, out float t, out vec2 uv) {
    t = 0.0;
    uv = vec2(0.0);

    uint base = slot * 4u;
    vec4 anchor_and_index = leaf_records_buf.records[base];
    vec4 dual_u_and_flags = leaf_records_buf.records[base + 1u];
    vec3 plane_normal = leaf_records_buf.records[base + 3u].xyz;

    float denom = dot(ray_dir, plane_normal);
    if (abs(denom) < 1e-8) {
        return false; // Ray parallel to the primitive
    }

    t = dot(anchor_and_index.xyz - ray_origin, plane_normal) / denom;
    if (t <= t_min || t >= t_max || t <= 1e-8 || t >= 1e8) {
        return false; // Behind the ray, beyond the closest hit so far, or too far
    }

    vec3 offset = ray_origin + ray_dir * t - anchor_and_index.xyz;
    uv = vec2(dot(offset, dual_u_and_flags.xyz), dot(offset, leaf_records_buf.records[base + 2u].xyz));

    bool is_triangle = (floatBitsToUint(dual_u_and_flags.w) & LEAF_RECORD_TRIANGLE) != 0u;
    if (is_triangle) {
        return uv.x >= 0.0 && uv.y >= 0.0 && uv.x + uv.y <= 1.0;
    }
    return uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0;
}

#endif // INTERSECTIONS_GLSL
