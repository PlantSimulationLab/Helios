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
// A textured triangle normally carries three explicit UVs. When it does not (uv_ids < 0), the
// per-primitive UV slots were never written and reading them yields (0,0) for all three, which
// collapses the whole triangle onto a single texel - and makes the triangle invisible to every
// ray if that texel happens to be transparent. Fall back to the canonical triangle mapping
// instead, mirroring how the patch branch treats uv_ids < 0 as "default UV mapping".
vec2 triangle_texture_uv(uint prim_idx, vec2 barycentric) {
    vec2 uv0, uv1, uv2;
    if (uv_ids_buf.uv_ids[prim_idx] < 0) {
        uv0 = vec2(0.0, 0.0);
        uv1 = vec2(0.0, 1.0);
        uv2 = vec2(1.0, 1.0);
    } else {
        // Vulkan: uv_data has 4 vec2 per primitive (all prims, indexed by prim_idx * 4)
        uv0 = uv_data_buf.uv_data[prim_idx * 4 + 0];
        uv1 = uv_data_buf.uv_data[prim_idx * 4 + 1];
        uv2 = uv_data_buf.uv_data[prim_idx * 4 + 2];
    }
    // Interpolate per-vertex UVs: tex_uv = uv0 + beta*(uv1-uv0) + gamma*(uv2-uv0)
    vec2 tex_uv = uv0 + barycentric.x * (uv1 - uv0) + barycentric.y * (uv2 - uv0);
    tex_uv.y = 1.0 - tex_uv.y;
    return tex_uv;
}

// Read one texel of a transparency mask. Single definition so that the ray-origin rejection in
// the direct/diffuse ray-generation shaders and the ray-hit rejection here cannot disagree about
// rounding - they previously used floor() without abs() and round() with abs() respectively,
// which put them one texel apart along each axis.
bool mask_texel_is_opaque(int mask_id, vec2 tex_uv) {
    ivec2 sz = mask_sizes_buf.mask_sizes[mask_id];
    if (sz.x <= 0 || sz.y <= 0) {
        return true; // Degenerate mask carries no information; treat the primitive as solid
    }
    uint offset = mask_offsets_buf.mask_offsets[mask_id];
    // round() and abs() match OptiX roundf/fabs (primitiveIntersection.cu:201-202)
    int tx = clamp(int(round(float(sz.x - 1) * abs(tex_uv.x))), 0, sz.x - 1);
    int ty = clamp(int(round(float(sz.y - 1) * abs(tex_uv.y))), 0, sz.y - 1);
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
