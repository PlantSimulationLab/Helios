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

/** spectral_texture.glsl - Classes of spectral textures (Context::setPrimitiveSpectralTexture())
 *
 * A ray leaving a primitive with a spectral texture uses the reflectivity and transmissivity of the class at the point it left
 * from, taken from the palette tables, instead of the primitive's own values (which hold their area-weighted mean).
 *
 * Set 0, binding 19 holds the class maps: the number of primitives P (0 when no primitive has a class map) and of class maps M;
 * then the class map of each primitive (0xFFFFFFFF for none); then the width, height and first texel of each class map; then the
 * classes of every texel, rows from the top, two 16-bit classes per word (the lower half first).
 *
 * Set 1, binding 18 holds the palettes (see RayTracingMaterial::palette_offset): a header of the number of palette entries E (0
 * when no primitive has a spectral texture), the global band count, the camera count, and the starts of the first palette entry of
 * each primitive (0xFFFFFFFF for none), of a flag per global band (1 where rays use the palette), and of the reflectivity,
 * transmissivity, camera-weighted reflectivity and camera-weighted transmissivity tables, as float bits.
 *
 * Set 2, binding 16 receives the radiation reaching each face of each primitive with a spectral texture, per [face (top, bottom)][source
 * slot][primitive][launch band] (see RayTracingResults::spectral_incident), when there are cameras and some primitive has a spectral texture.
 *
 * Set 2, binding 17 holds the camera-weighted outgoing radiation of each class for the camera being traced (see
 * RayTracingBackend::uploadSpectralCameraFlux()): the number of class-flux slots S (0 for none), the launch band count B and the number of
 * primitives P; then the first slot of each primitive (0xFFFFFFFF for none); then the values per [face][slot][band] as float bits.
 *
 * Set 2, binding 19 holds, with camera flux smoothing, the radiation of primitives with a spectral texture on the smoothing vertices (see
 * RayTracingBackend::uploadSpectralVertexIncident()): the layer count L (0 for none: the source slots and then the emitted radiation), the
 * launch band count B, the vertex count V and the camera index; then the values per [face][layer][vertex][band] as float bits.
 *
 * Requires texture_mask.glsl (triangle_texture_uv) and the uv_data_buf / uv_ids_buf buffers.
 */

#ifndef SPECTRAL_TEXTURE_GLSL
#define SPECTRAL_TEXTURE_GLSL

layout(set = 0, binding = 19) readonly buffer SpectralClassMapBuffer {
    uint words[];
} class_map_buf;

layout(set = 1, binding = 18) readonly buffer SpectralPaletteBuffer {
    uint words[];
} palette_buf;

layout(set = 2, binding = 16) buffer SpectralIncidentBuffer {
    float values[];
} spectral_incident_buf;

layout(set = 2, binding = 17) readonly buffer SpectralCameraFluxBuffer {
    uint words[];
} spectral_camera_flux_buf;

layout(set = 2, binding = 19) readonly buffer SpectralVertexBuffer {
    uint words[];
} spectral_vertex_buf;

const uint NO_PALETTE_ENTRY = 0xFFFFFFFFu;

// Texture coordinate of a launch point sp_local in a primitive's canonical space, with v counted from the top of the image, as
// mask_texel_is_opaque() takes it. The same mapping as the ray-origin mask rejection in the direct and diffuse shaders.
vec2 spectral_origin_texel_uv(uint prim_idx, uint prim_type, vec3 sp_local) {
    if (prim_type == 1u) {
        // Canonical triangle: x = gamma, y = beta + gamma. A triangle with a spectral texture always has (u,v) coordinates.
        return triangle_texture_uv(prim_idx, vec2(sp_local.y - sp_local.x, sp_local.x));
    }
    vec2 uv = vec2(sp_local.x + 0.5, sp_local.y + 0.5);
    if (uv_ids_buf.uv_ids[prim_idx] < 0) {
        return vec2(uv.x, 1.0 - uv.y);
    }
    // As the patch branch of check_texture_mask()
    vec2 uvmin = uv_data_buf.uv_data[prim_idx * 4 + 0];
    float du = uv_data_buf.uv_data[prim_idx * 4 + 1].x - uvmin.x;
    float dv = uv_data_buf.uv_data[prim_idx * 4 + 2].y - uv_data_buf.uv_data[prim_idx * 4 + 1].y;
    return vec2(uvmin.x + uv.x * du, 1.0 - uvmin.y - uv.y * dv);
}

// Palette entry of the class at a launch point of a primitive (texel_uv from spectral_origin_texel_uv()), or NO_PALETTE_ENTRY if
// the primitive has no spectral texture or the texel is transparent (class 0). The texel convention is that of the masks.
uint spectral_palette_entry(uint prim_idx, vec2 texel_uv) {
    if (palette_buf.words[0] == 0u || class_map_buf.words[0] == 0u) {
        return NO_PALETTE_ENTRY;
    }
    uint first_entry = palette_buf.words[palette_buf.words[3] + prim_idx];
    uint num_prims = class_map_buf.words[0];
    uint class_map = class_map_buf.words[2u + prim_idx];
    if (first_entry == NO_PALETTE_ENTRY || class_map == 0xFFFFFFFFu) {
        return NO_PALETTE_ENTRY;
    }
    uint info = 2u + num_prims + 3u * class_map;
    int width = int(class_map_buf.words[info]);
    int height = int(class_map_buf.words[info + 1u]);
    uint first_texel = class_map_buf.words[info + 2u];
    int tx = clamp(int(floor(float(width - 1) * texel_uv.x)), 0, width - 1);
    int ty = clamp(int(floor(float(height - 1) * texel_uv.y)), 0, height - 1);
    uint texel = first_texel + uint(ty) * uint(width) + uint(tx);
    uint data_start = 2u + num_prims + 3u * class_map_buf.words[1];
    uint word = class_map_buf.words[data_start + texel / 2u];
    uint texel_class = (texel % 2u == 0u) ? (word & 0xFFFFu) : (word >> 16u);
    return texel_class == 0u ? NO_PALETTE_ENTRY : first_entry + texel_class - 1u;
}

// Whether a ray with palette entry `entry` uses the palette in global band `global_band`
bool spectral_palette_applies(uint entry, uint global_band) {
    return entry != NO_PALETTE_ENTRY && palette_buf.words[palette_buf.words[4] + global_band] != 0u;
}

float spectral_palette_value(uint table, uint slot, uint entry, uint global_band) {
    uint num_entries = palette_buf.words[0];
    uint num_bands = palette_buf.words[1];
    return uintBitsToFloat(palette_buf.words[palette_buf.words[table] + (slot * num_entries + entry) * num_bands + global_band]);
}

float spectral_palette_value_cam(uint table, uint slot, uint entry, uint global_band, uint camera) {
    uint num_entries = palette_buf.words[0];
    uint num_bands = palette_buf.words[1];
    uint num_cameras = palette_buf.words[2];
    return uintBitsToFloat(palette_buf.words[palette_buf.words[table] + ((slot * num_entries + entry) * num_bands + global_band) * num_cameras + camera]);
}

// Reflectivity, transmissivity and their camera-weighted values of the class of palette entry `entry` in material slot `slot`
float spectral_palette_rho(uint slot, uint entry, uint global_band) {
    return spectral_palette_value(5u, slot, entry, global_band);
}
float spectral_palette_tau(uint slot, uint entry, uint global_band) {
    return spectral_palette_value(6u, slot, entry, global_band);
}
float spectral_palette_rho_cam(uint slot, uint entry, uint global_band, uint camera) {
    return spectral_palette_value_cam(7u, slot, entry, global_band, camera);
}
float spectral_palette_tau_cam(uint slot, uint entry, uint global_band, uint camera) {
    return spectral_palette_value_cam(8u, slot, entry, global_band, camera);
}

// Whether primitive prim_idx has a spectral texture (with the palette of the current materials)
bool spectral_texture_on(uint prim_idx) {
    return palette_buf.words[0] != 0u && palette_buf.words[palette_buf.words[3] + prim_idx] != NO_PALETTE_ENTRY;
}

// Class-flux slot of the class a camera ray hits, or NO_PALETTE_ENTRY. hit_uv comes from the traversal: patch-local coordinates in
// [0,1], or the barycentric weights of a triangle's second and third corners.
uint spectral_camera_class_slot(uint prim_idx, uint prim_type, vec2 hit_uv) {
    if (spectral_camera_flux_buf.words[0] == 0u) {
        return NO_PALETTE_ENTRY;
    }
    uint first_slot = spectral_camera_flux_buf.words[3u + prim_idx];
    if (first_slot == NO_PALETTE_ENTRY || !spectral_texture_on(prim_idx)) {
        return NO_PALETTE_ENTRY;
    }
    vec3 sp = prim_type == 1u ? vec3(hit_uv.y, hit_uv.x + hit_uv.y, 0.0) : vec3(hit_uv.x - 0.5, hit_uv.y - 0.5, 0.0);
    uint entry = spectral_palette_entry(prim_idx, spectral_origin_texel_uv(prim_idx, prim_type, sp));
    if (entry == NO_PALETTE_ENTRY) {
        return NO_PALETTE_ENTRY;
    }
    return first_slot + (entry - palette_buf.words[palette_buf.words[3] + prim_idx]);
}

// Camera-weighted outgoing radiation of a class-flux slot from one face in a launch band
float spectral_class_flux(uint slot, bool top_face, uint launch_band) {
    uint num_slots = spectral_camera_flux_buf.words[0];
    uint num_bands = spectral_camera_flux_buf.words[1];
    uint num_prims = spectral_camera_flux_buf.words[2];
    return uintBitsToFloat(spectral_camera_flux_buf.words[3u + num_prims + ((top_face ? 0u : num_slots) + slot) * num_bands + launch_band]);
}

// Whether the smoothing-vertex radiation of primitives with a spectral texture is present
bool spectral_vertex_on() {
    return spectral_vertex_buf.words[0] != 0u;
}

// With camera flux smoothing: the camera-weighted outgoing radiation of the class of class-flux slot `slot` on primitive prim_idx, from one face
// in a launch band, interpolated from the facet's vertices v (v.w < 0 for a triangle) with weights w
float spectral_smoothed_class_flux(uint prim_idx, uint slot, bool top_face, uint launch_band, uint global_band, ivec4 v, vec4 w) {
    uint layers = spectral_vertex_buf.words[0];
    uint num_bands = spectral_vertex_buf.words[1];
    uint num_vertices = spectral_vertex_buf.words[2];
    uint camera = spectral_vertex_buf.words[3];
    uint entry = palette_buf.words[palette_buf.words[3] + prim_idx] + (slot - spectral_camera_flux_buf.words[3u + prim_idx]);
    uint same_face = top_face ? 0u : 1u;
    float flux = 0.0;
    for (uint layer = 0u; layer < layers; layer++) {
        float value[2];
        for (uint face = 0u; face < 2u; face++) {
            uint base = 4u + ((face * layers + layer) * num_vertices) * num_bands + launch_band;
            float sum = w.x * uintBitsToFloat(spectral_vertex_buf.words[base + uint(v.x) * num_bands]) + w.y * uintBitsToFloat(spectral_vertex_buf.words[base + uint(v.y) * num_bands]) +
                        w.z * uintBitsToFloat(spectral_vertex_buf.words[base + uint(v.z) * num_bands]);
            if (v.w >= 0) {
                sum += w.w * uintBitsToFloat(spectral_vertex_buf.words[base + uint(v.w) * num_bands]);
            }
            value[face] = sum;
        }
        if (layer + 1u == layers) {
            flux += value[same_face]; // emitted
        } else {
            flux += spectral_palette_rho_cam(layer, entry, global_band, camera) * value[same_face] + spectral_palette_tau_cam(layer, entry, global_band, camera) * value[1u - same_face];
        }
    }
    return flux;
}

#endif // SPECTRAL_TEXTURE_GLSL
