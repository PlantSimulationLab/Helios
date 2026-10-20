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

/** texture_mask.glsl - Texture transparency mask checking
 *
 * Requires these buffers to be defined in the calling shader:
 *   - mask_data_buf.mask_data[] (uint, 0=transparent, 1=opaque)
 *   - mask_sizes_buf.mask_sizes[] (ivec2, width x height per mask)
 *   - mask_offsets_buf.mask_offsets[] (uint, starting offset per mask)
 *   - mask_ids_buf.mask_ids[] (int, mask ID per primitive, -1 = no mask)
 *   - uv_data_buf.uv_data[] (vec2, 4 per primitive)
 *   - uv_ids_buf.uv_ids[] (int, -1 = default UVs)
 */

#ifndef TEXTURE_MASK_GLSL
#define TEXTURE_MASK_GLSL

// Texture coordinates of a triangle hit, given the Möller-Trumbore barycentrics.
//
// A masked triangle always carries three explicit UVs: RadiationModel::updateGeometry() rejects one without them.
vec2 triangle_texture_uv(uint prim_idx, vec2 barycentric) {
    // Vulkan: uv_data has 4 vec2 per primitive (all prims, indexed by prim_idx * 4)
    vec2 uv0 = uv_data_buf.uv_data[prim_idx * 4 + 0];
    vec2 uv1 = uv_data_buf.uv_data[prim_idx * 4 + 1];
    vec2 uv2 = uv_data_buf.uv_data[prim_idx * 4 + 2];
    // Interpolate per-vertex UVs: tex_uv = uv0 + beta*(uv1-uv0) + gamma*(uv2-uv0)
    vec2 tex_uv = uv0 + barycentric.x * (uv1 - uv0) + barycentric.y * (uv2 - uv0);
    tex_uv.y = 1.0 - tex_uv.y;
    return tex_uv;
}

// Read one texel of a transparency mask. Single definition so that the ray-origin rejection in
// the direct/diffuse ray-generation shaders and the ray-hit rejection here cannot disagree about
// rounding. Texel i spans texture coordinates i/(size-1) to (i+1)/(size-1), as in
// Texture::computeSolidFraction(), the patch branch below and the OptiX backends.
bool mask_texel_is_opaque(int mask_id, vec2 tex_uv) {
    ivec2 sz = mask_sizes_buf.mask_sizes[mask_id];
    if (sz.x <= 0 || sz.y <= 0) {
        return true; // Degenerate mask carries no information; treat the primitive as solid
    }
    uint offset = mask_offsets_buf.mask_offsets[mask_id];
    int tx = clamp(int(floor(float(sz.x - 1) * tex_uv.x)), 0, sz.x - 1);
    int ty = clamp(int(floor(float(sz.y - 1) * tex_uv.y)), 0, sz.y - 1);
    return mask_data_buf.mask_data[offset + uint(ty) * uint(sz.x) + uint(tx)] != 0;
}

// Check if a parametric UV coordinate on a primitive is opaque (not masked)
// prim_idx: global primitive index
// uv: parametric UV in [0,1] x [0,1] for patches, barycentric coords for triangles
// prim_type: 0=patch, 1=triangle, 2=disk, 3=tile
// Returns true if opaque (solid), false if transparent
bool check_texture_mask(uint prim_idx, vec2 uv, uint prim_type) {
    int mask_id = mask_ids_buf.mask_ids[prim_idx];
    if (mask_id < 0) {
        return true; // No mask - always opaque
    }

    if (prim_type == 1) {
        // Triangle: uv contains barycentric (beta, gamma) from Möller-Trumbore
        return mask_texel_is_opaque(mask_id, triangle_texture_uv(prim_idx, uv));
    }

    // Patch/Tile: rectangular UV mapping. Unlike the triangle branch this maps parametric
    // surface coordinates rather than barycentrics, so it keeps its own floor()-based mapping.
    ivec2 sz = mask_sizes_buf.mask_sizes[mask_id];
    if (sz.x <= 0 || sz.y <= 0) {
        return true; // Degenerate mask carries no information; treat the primitive as solid
    }
    uint offset = mask_offsets_buf.mask_offsets[mask_id];

    int tx, ty;
    int uv_id = uv_ids_buf.uv_ids[prim_idx];
    if (uv_id < 0) {
        // Default UV mapping: direct parametric coordinates
        tx = int(floor(float(sz.x - 1) * uv.x));
        ty = int(floor(float(sz.y - 1) * (1.0 - uv.y)));  // Y-flip (texture top-to-bottom)
    } else {
        // Custom UV mapping: read UVs from buffer
        // Vulkan: uv_data has 4 vec2 per primitive (all prims, indexed by prim_idx * 4)
        vec2 uvmin = uv_data_buf.uv_data[prim_idx * 4 + 0];
        float du = uv_data_buf.uv_data[prim_idx * 4 + 1].x - uvmin.x;
        float dv = uv_data_buf.uv_data[prim_idx * 4 + 2].y - uv_data_buf.uv_data[prim_idx * 4 + 1].y;
        tx = int(floor(float(sz.x - 1) * (uvmin.x + uv.x * du)));
        ty = int(floor(float(sz.y - 1) * (1.0 - uvmin.y - uv.y * dv)));
    }

    // Clamp to valid range
    tx = clamp(tx, 0, sz.x - 1);
    ty = clamp(ty, 0, sz.y - 1);

    return mask_data_buf.mask_data[offset + uint(ty) * uint(sz.x) + uint(tx)] != 0;
}

#endif // TEXTURE_MASK_GLSL
