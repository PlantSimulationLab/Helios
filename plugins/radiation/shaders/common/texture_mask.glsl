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

    ivec2 sz = mask_sizes_buf.mask_sizes[mask_id];
    uint offset = mask_offsets_buf.mask_offsets[mask_id];

    int tx, ty;

    if (prim_type == 1) {
        // Triangle: uv contains barycentric (beta, gamma) from Möller-Trumbore
        // Interpolate per-vertex UVs: tex_uv = uv0 + beta*(uv1-uv0) + gamma*(uv2-uv0)
        // Vulkan: uv_data has 4 vec2 per primitive (all prims, indexed by prim_idx * 4)
        vec2 uv0 = uv_data_buf.uv_data[prim_idx * 4 + 0];
        vec2 uv1 = uv_data_buf.uv_data[prim_idx * 4 + 1];
        vec2 uv2 = uv_data_buf.uv_data[prim_idx * 4 + 2];
        vec2 tex_uv = uv0 + uv.x * (uv1 - uv0) + uv.y * (uv2 - uv0);
        tex_uv.y = 1.0 - tex_uv.y;
        // Use round() and abs() to match OptiX roundf/fabs (primitiveIntersection.cu:201-202)
        tx = int(round(float(sz.x - 1) * abs(tex_uv.x)));
        ty = int(round(float(sz.y - 1) * abs(tex_uv.y)));
    } else {
        int uv_id = uv_ids_buf.uv_ids[prim_idx];
        // Patch/Tile: rectangular UV mapping
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
    }

    // Clamp to valid range
    tx = clamp(tx, 0, sz.x - 1);
    ty = clamp(ty, 0, sz.y - 1);

    return mask_data_buf.mask_data[offset + ty * sz.x + tx] != 0;
}

#endif // TEXTURE_MASK_GLSL
