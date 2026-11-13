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

/** normal_map.glsl - Normal maps (Context::setMaterialNormalMap())
 *
 * A line-for-line port of include/NormalMapMath.h, which specifies the packed buffer layout and every convention: change both together.
 *
 * Requires these buffers to be defined in the calling shader:
 *   - transform_buf.matrices[] (float, 16 per primitive, row-major)
 *   - uv_data_buf.uv_data[] (vec2, 4 per primitive)
 *   - uv_ids_buf.uv_ids[] (int, -1 = default UVs)
 */

#ifndef NORMAL_MAP_GLSL
#define NORMAL_MAP_GLSL

layout(set = 0, binding = 20) readonly buffer NormalMapBuffer {
    uint words[];
} normal_map_buf;

// Slot of each normal-mapped primitive in normal_map_moment_buf (0xFFFFFFFF for none), and [slot][face][launch band][3] the flux-weighted directions of the diffuse
// radiation reaching each face (see RayTracingGeometry::normal_map_slots); one-element placeholders when no primitive has a map. With a background radiation field
// (see buffer_indexing.glsl), its moments are in a second set of slots after those of the scene as it is.
layout(set = 0, binding = 21) readonly buffer NormalMapSlotBuffer {
    uint slots[];
} normal_map_slot_buf;

layout(set = 2, binding = 23) buffer NormalMapMomentBuffer {
    float values[];
} normal_map_moment_buf;

// normal_map_moment_buf averaged onto the smoothing vertices of camera flux smoothing, [field][vertex][face][launch band][3]; a one-element placeholder when there is none
layout(set = 2, binding = 24) readonly buffer VertexNormalMapMomentBuffer {
    float values[];
} vertex_normal_map_moment_buf;

const uint NO_NORMAL_MAP = 0xFFFFu;
const uint NO_NORMAL_MAP_SLOT = 0xFFFFFFFFu;

// Slot of the primitive at prim_idx in the record of the directions of its diffuse radiation, or NO_NORMAL_MAP_SLOT
uint nm_slot(uint prim_idx) {
    if (normal_map_buf.words[0] == 0u || prim_idx >= uint(normal_map_slot_buf.slots.length())) {
        return NO_NORMAL_MAP_SLOT;
    }
    return normal_map_slot_buf.slots[prim_idx];
}

// The change, for a camera, in the scattered light the viewed face of a normal-mapped primitive sends out where its shading normal tilts away from the base normal
// (the facet's or the smoothed one) by the tangential vector m_t (see normalMapDiffuseRelief() in the CUDA backends): (8/3) m_t . (rho M_same - tau M_other), for the
// parts of the moments along the facet (unit normal facet_normal) they were gathered on
// With camera flux smoothing (corners.x >= 0) and moments averaged onto the smoothing vertices, they are interpolated from the facet's corners with the given weights.
// The moments are those of one radiation field: field_slot_offset and field_vertex_offset are the numbers of slots and of vertices before those of that field (zero for the scene as it is).
float nm_diffuse_relief(uint prim_idx, bool face_top, uint band, uint band_count, vec3 shading_normal, vec3 base_normal, vec3 facet_normal, float rho_d, float tau_d, ivec4 corners, vec4 weights,
                        uint field_slot_offset, uint field_vertex_offset) {
    uint slot = nm_slot(prim_idx);
    if (slot == NO_NORMAL_MAP_SLOT) {
        return 0.0;
    }
    slot += field_slot_offset;
    uint same = ((slot * 2u + (face_top ? 0u : 1u)) * band_count + band) * 3u;
    uint other = ((slot * 2u + (face_top ? 1u : 0u)) * band_count + band) * 3u;
    if (max(same, other) + 3u > uint(normal_map_moment_buf.values.length())) {
        return 0.0;
    }
    vec3 m_same = vec3(normal_map_moment_buf.values[same], normal_map_moment_buf.values[same + 1u], normal_map_moment_buf.values[same + 2u]);
    vec3 m_other = vec3(normal_map_moment_buf.values[other], normal_map_moment_buf.values[other + 1u], normal_map_moment_buf.values[other + 2u]);
    if (vertex_normal_map_moment_buf.values.length() > 1 && corners.x >= 0) {
        m_same = vec3(0.0);
        m_other = vec3(0.0);
        uint face_same = face_top ? 0u : 1u;
        for (int c = 0; c < 4; c++) {
            if (corners[c] < 0) {
                continue;
            }
            uint vertex = uint(corners[c]) + field_vertex_offset;
            uint vertex_same = ((vertex * 2u + face_same) * band_count + band) * 3u;
            uint vertex_other = ((vertex * 2u + (1u - face_same)) * band_count + band) * 3u;
            if (max(vertex_same, vertex_other) + 3u > uint(vertex_normal_map_moment_buf.values.length())) {
                return 0.0;
            }
            m_same += weights[c] * vec3(vertex_normal_map_moment_buf.values[vertex_same], vertex_normal_map_moment_buf.values[vertex_same + 1u], vertex_normal_map_moment_buf.values[vertex_same + 2u]);
            m_other += weights[c] * vec3(vertex_normal_map_moment_buf.values[vertex_other], vertex_normal_map_moment_buf.values[vertex_other + 1u], vertex_normal_map_moment_buf.values[vertex_other + 2u]);
        }
    }
    vec3 weighted = rho_d * m_same - tau_d * m_other;
    weighted -= facet_normal * dot(weighted, facet_normal);
    vec3 m_t = shading_normal - base_normal * dot(shading_normal, base_normal);
    return (8.0 / 3.0) * dot(m_t, weighted);
}

// Map index of the primitive at position prim_idx, or NO_NORMAL_MAP
uint nm_map_index(uint prim_idx) {
    if (normal_map_buf.words[0] == 0u) {
        return NO_NORMAL_MAP;
    }
    uint word = normal_map_buf.words[4u + prim_idx / 2u];
    return (prim_idx & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu);
}

// Unit tangent-space normal of a packed 16-bit texel
vec3 nm_decode_texel(uint texel) {
    int xi = int(texel & 0xFFu);
    int yi = int((texel >> 8) & 0xFFu);
    xi = xi >= 128 ? xi - 256 : xi;
    yi = yi >= 128 ? yi - 256 : yi;
    float x = float(xi) / 127.0;
    float y = float(yi) / 127.0;
    float zz = 1.0 - x * x - y * y;
    vec3 n = vec3(x, y, zz > 0.0 ? sqrt(zz) : 0.0);
    float len = length(n);
    return len > 0.0 ? n / len : n;
}

// Normal of texel (column, row from the top) of map m
vec3 nm_texel(uint m, int column, int row) {
    uint P = normal_map_buf.words[0];
    uint header = 4u + (P + 1u) / 2u + 3u * m;
    uint width = normal_map_buf.words[header];
    uint index = uint(row) * width + uint(column);
    uint word = normal_map_buf.words[normal_map_buf.words[2] + normal_map_buf.words[header + 2u] + index / 2u];
    return nm_decode_texel((index & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

// Normal of map m at texture coordinate (u,v): the texel at column floor((W-1)u) and row floor((H-1)(1-v)), clamped to the image
vec3 nm_fetch_nearest(uint m, float u, float v) {
    uint header = 4u + (normal_map_buf.words[0] + 1u) / 2u + 3u * m;
    int width = int(normal_map_buf.words[header]);
    int height = int(normal_map_buf.words[header + 1u]);
    int column = clamp(int(floor(float(width - 1) * u)), 0, width - 1);
    int row = clamp(int(floor(float(height - 1) * (1.0 - v))), 0, height - 1);
    return nm_texel(m, column, row);
}

// Normal of map m bilinearly interpolated at (u,v), texel centres as in nm_fetch_nearest(), clamped to the edge, renormalized
vec3 nm_fetch_bilinear(uint m, float u, float v) {
    uint header = 4u + (normal_map_buf.words[0] + 1u) / 2u + 3u * m;
    int width = int(normal_map_buf.words[header]);
    int height = int(normal_map_buf.words[header + 1u]);
    float fx = float(width - 1) * u - 0.5;
    float fy = float(height - 1) * (1.0 - v) - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float wx = fx - float(x0);
    float wy = fy - float(y0);
    int c0 = clamp(x0, 0, width - 1);
    int c1 = clamp(x0 + 1, 0, width - 1);
    int r0 = clamp(y0, 0, height - 1);
    int r1 = clamp(y0 + 1, 0, height - 1);
    vec3 top = nm_texel(m, c0, r0) * (1.0 - wx) + nm_texel(m, c1, r0) * wx;
    vec3 bottom = nm_texel(m, c0, r1) * (1.0 - wx) + nm_texel(m, c1, r1) * wx;
    vec3 n = top * (1.0 - wy) + bottom * wy;
    float len = length(n);
    return len > 0.0 ? n / len : n;
}

// Edges E1, E2 of the canonical shape of the primitive at prim_idx in world space
void nm_primitive_edges(uint prim_idx, uint prim_type, out vec3 E1, out vec3 E2) {
    uint base = prim_idx * 16u;
    vec3 c0 = vec3(transform_buf.matrices[base + 0u], transform_buf.matrices[base + 4u], transform_buf.matrices[base + 8u]);
    vec3 c1 = vec3(transform_buf.matrices[base + 1u], transform_buf.matrices[base + 5u], transform_buf.matrices[base + 9u]);
    if (prim_type == 1u) {
        E1 = c1;
        E2 = c0 + c1;
    } else {
        E1 = c0;
        E2 = c1;
    }
}

// Unit top normal of the facet, normalize(cross(E1, E2)), as the ray-generation programs compute it
vec3 nm_facet_normal(uint prim_idx, uint prim_type) {
    vec3 E1, E2;
    nm_primitive_edges(prim_idx, prim_type, E1, E2);
    return normalize(cross(E1, E2));
}

// Shading normal on the top face of the primitive at prim_idx at canonical point (sx, sy), about the unit base normal N on the side of the facet's top normal
vec3 nm_shading_normal(uint prim_idx, uint prim_type, uint m, vec3 N, float sx, float sy, bool bilinear) {
    vec3 E1, E2;
    nm_primitive_edges(prim_idx, prim_type, E1, E2);
    bool has_uv = uv_ids_buf.uv_ids[prim_idx] >= 0;
    vec2 uv0 = vec2(0.0), uv1 = vec2(0.0), uv2 = vec2(0.0);
    if (has_uv) {
        uv0 = uv_data_buf.uv_data[prim_idx * 4u + 0u];
        uv1 = uv_data_buf.uv_data[prim_idx * 4u + 1u];
        uv2 = uv_data_buf.uv_data[prim_idx * 4u + 2u];
    }
    // Columns a, b of the texture-coordinate Jacobian
    vec2 a, b;
    if (!has_uv) {
        a = vec2(1.0, 0.0);
        b = prim_type == 1u ? vec2(1.0, 1.0) : vec2(0.0, 1.0);
    } else {
        a = uv1 - uv0;
        b = prim_type == 1u ? uv2 - uv0 : uv2 - uv1;
    }
    // Texture coordinate, v up the image
    float s1 = prim_type == 1u ? sy - sx : sx + 0.5;
    float s2 = prim_type == 1u ? sx : sy + 0.5;
    vec2 uv = uv0 + s1 * a + s2 * b;
    // Tangent frame
    float det = a.x * b.y - a.y * b.x;
    float extent = max(dot(a, a), dot(b, b));
    if (!(abs(det) > 1e-12 * extent) || extent == 0.0) {
        return N;
    }
    vec3 dPdu = (E1 * b.y - E2 * a.y) / det;
    vec3 tangent = dPdu - N * dot(N, dPdu);
    float length2 = dot(tangent, tangent);
    if (!(length2 > 1e-20)) {
        return N;
    }
    vec3 t = tangent / sqrt(length2);
    vec3 bt = cross(N, t) * (det > 0.0 ? 1.0 : -1.0);
    vec3 n_ts = bilinear ? nm_fetch_bilinear(m, uv.x, uv.y) : nm_fetch_nearest(m, uv.x, uv.y);
    return normalize(t * n_ts.x + bt * n_ts.y + N * n_ts.z);
}

// Shadow-terminator softening of Chiang, Li and Burley (2019); both normals oriented to the face, 0 when either cosine is not positive
float nm_terminator_softening(vec3 n_f, vec3 m_f, vec3 L) {
    float mu_m = dot(m_f, L);
    float mu_g = dot(n_f, L);
    float c = dot(n_f, m_f);
    if (mu_m <= 0.0 || mu_g <= 0.0 || c <= 1e-6) {
        return 0.0;
    }
    float G = min(1.0, mu_g / (mu_m * c));
    return G * (1.0 + G * (1.0 - G));
}

#endif // NORMAL_MAP_GLSL
