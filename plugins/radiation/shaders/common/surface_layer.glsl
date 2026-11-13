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

/** surface_layer.glsl - Specular surface layers (see RayTracingMaterial::surface_model)
 *
 * A primitive with a layer reflects a(mu) of the radiation arriving at its surface into the second component of its outgoing
 * radiation, with angular shape a(mu), and its body absorbs, reflects and transmits the rest (rho and tau are those of the
 * body), sending it out in the first component with shape 1 - a(mu). Both components are stored in the outgoing and scattered
 * buffers, the second after the first, and in the camera buffers after all the cameras' first components.
 *
 * The data are packed in one buffer of uint words (VulkanComputeBackend::surface_layer_buffer): a header of the table size
 * (0 when no primitive has a layer), the global band count, and the starts of the band tables and of the albedo tables; then
 * the surface model of each primitive, the albedo table of each model in each band, and the tables as float bits (each the
 * table size values of a(mu), then abar, the refractive index and the roughness).
 */

#ifndef SURFACE_LAYER_GLSL
#define SURFACE_LAYER_GLSL

layout(set = 1, binding = 17) readonly buffer SurfaceLayerBuffer {
    uint words[];
} surface_layer_buf;

// Whether any primitive has a surface layer. The same for every invocation, so subgroup operations may depend on it.
bool surface_layers_enabled() {
    return surface_layer_buf.words[0] != 0u;
}

// Whether primitive prim has a surface layer (in some band)
bool surface_layer_present(uint prim) {
    return surface_layer_buf.words[0] != 0u && surface_layer_buf.words[4u + prim] != 0u;
}

// Start of the albedo table of the surface layer of primitive prim in global band global_band, or -1 if it has none there
int surface_layer_table(uint prim, uint global_band) {
    uint table_size = surface_layer_buf.words[0];
    if (table_size == 0u) {
        return -1;
    }
    uint model = surface_layer_buf.words[4u + prim];
    if (model == 0u) {
        return -1;
    }
    uint table = surface_layer_buf.words[surface_layer_buf.words[2] + (model - 1u) * surface_layer_buf.words[1] + global_band];
    if (table == 0xFFFFFFFFu) {
        return -1;
    }
    return int(surface_layer_buf.words[3] + table * (table_size + 3u));
}

// Directional albedo a(mu) of a surface layer, interpolated linearly in its table
float surface_layer_albedo(int table, float mu) {
    uint size = surface_layer_buf.words[0];
    float x = clamp(mu, 0.0, 1.0) * float(size - 1u);
    uint i = min(uint(x), size - 2u);
    float a0 = uintBitsToFloat(surface_layer_buf.words[uint(table) + i]);
    float a1 = uintBitsToFloat(surface_layer_buf.words[uint(table) + i + 1u]);
    return a0 + (x - float(i)) * (a1 - a0);
}

// Hemispherical mean abar of a surface layer's albedo
float surface_layer_mean_albedo(int table) {
    return uintBitsToFloat(surface_layer_buf.words[uint(table) + surface_layer_buf.words[0]]);
}

// Flux a face sends out at the angle whose cosine is mu_out, from its first and second outgoing components
float surface_layer_outgoing(float first, float second, int table, float mu_out) {
    float a = surface_layer_albedo(table, mu_out);
    float abar = surface_layer_mean_albedo(table);
    return first * (1.0 - a) / (1.0 - abar) + second * a / abar;
}

// Fresnel + GGX reflection of a surface layer from direction to_source to direction to_viewer (unit vectors), in 1/sr, for the
// face whose unit normal is face_normal; 0 unless both are on that face's side. The table holds the layer's refractive index
// and roughness after abar.
float surface_layer_lobe(int table, vec3 face_normal, vec3 to_source, vec3 to_viewer) {
    float mu_s = dot(face_normal, to_source);
    float mu_o = dot(face_normal, to_viewer);
    if (mu_s <= 0.0 || mu_o <= 0.0) {
        return 0.0;
    }
    uint size = surface_layer_buf.words[0];
    float n = uintBitsToFloat(surface_layer_buf.words[uint(table) + size + 1u]);
    float alpha = uintBitsToFloat(surface_layer_buf.words[uint(table) + size + 2u]);
    float alpha2 = alpha * alpha;
    vec3 half_vector = normalize(to_source + to_viewer);
    float mu_h = dot(face_normal, half_vector);
    float D_denominator = mu_h * mu_h * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (3.14159265359 * D_denominator * D_denominator);
    float lambda_s = 0.5 * (-1.0 + sqrt(1.0 + alpha2 * (1.0 - mu_s * mu_s) / (mu_s * mu_s)));
    float lambda_o = 0.5 * (-1.0 + sqrt(1.0 + alpha2 * (1.0 - mu_o * mu_o) / (mu_o * mu_o)));
    float G2 = 1.0 / (1.0 + lambda_s + lambda_o);
    // Unpolarized Fresnel reflectance of the microfacet
    float c = clamp(dot(half_vector, to_source), 0.0, 1.0);
    float cos_t = sqrt(max(0.0, 1.0 - (1.0 - c * c) / (n * n)));
    float rs = (c - n * cos_t) / (c + n * cos_t);
    float rp = (n * c - cos_t) / (n * c + cos_t);
    float F = 0.5 * (rs * rs + rp * rp);
    return F * D * G2 / (4.0 * mu_s * mu_o);
}

#endif // SURFACE_LAYER_GLSL
