/** \file "rayHit.cu" Functions for object intersection.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/


#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "RayTracing.cuh"
#include "BufferIndexing.h"

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable(unsigned int, UUID, attribute UUID, );
// Where on the facet the ray landed, written by every intersection program. See primitiveIntersection.cu for the meaning of each component.
rtDeclareVariable(float, surface_u, attribute surface_u, );
rtDeclareVariable(float, surface_v, attribute surface_v, );
// True distance to a periodic boundary face, which bbox_intersect may report at the ray's minimum distance instead
rtDeclareVariable(float, periodic_face_t, attribute periodic_face_t, );

// ---------------------------------------------------------------------------
// Translucent cover (glass/plastic) optical model
// ---------------------------------------------------------------------------

//! Angular transmittance/reflectance/absorptance of a single dielectric sheet via Fresnel reflection
//! (two interfaces, both polarizations, internal reflections summed in closed form) plus Bouguer
//! absorption (Duffie & Beckman solar-glazing model). Byte-for-byte port of glass_tau_rho_alpha() in
//! optix8/OptiX8DeviceCode.cu and shaders/common/glass_cover.glsl — KEEP NUMERICALLY IDENTICAL.
//! Inputs:  cos_theta = |cos| of incidence angle, n = refractive index (>1), KL = absorption product K*L.
//! Returns: float3(tau, rho, alpha) with tau + rho + alpha == 1.
static __device__ __inline__ float3 glass_tau_rho_alpha(float cos_theta, float n, float KL) {
    cos_theta = fmaxf(1e-4f, fminf(1.f, cos_theta)); // guard grazing/degenerate
    const float theta = acos_safe(cos_theta);
    const float sin_t = sinf(theta);
    const float sin_tr = sin_t / n; // Snell
    const float cos_tr = sqrtf(fmaxf(0.f, 1.f - sin_tr * sin_tr));
    const float theta_r = asin_safe(sin_tr);

    float r_par, r_per;
    if (theta < 1e-3f) {
        const float r0 = ((n - 1.f) / (n + 1.f)) * ((n - 1.f) / (n + 1.f));
        r_par = r0;
        r_per = r0;
    } else {
        const float s_minus = sinf(theta_r - theta);
        const float s_plus = sinf(theta_r + theta);
        const float t_minus = tanf(theta_r - theta);
        const float t_plus = tanf(theta_r + theta);
        r_per = (s_minus * s_minus) / fmaxf(1e-12f, s_plus * s_plus); // perpendicular
        r_par = (t_minus * t_minus) / fmaxf(1e-12f, t_plus * t_plus); // parallel
    }

    const float tau_a = (KL > 0.f) ? expf(-KL / fmaxf(1e-4f, cos_tr)) : 1.f; // Bouguer absorption

    float tau = 0.f, rho = 0.f;
    for (int pol = 0; pol < 2; pol++) {
        const float r = (pol == 0) ? r_per : r_par;
        const float denom = fmaxf(1e-6f, 1.f - (r * tau_a) * (r * tau_a));
        const float tau_i = tau_a * (1.f - r) * (1.f - r) / denom;
        const float rho_i = r * (1.f + tau_a * tau_i);
        tau += 0.5f * tau_i;
        rho += 0.5f * rho_i;
    }
    tau = fmaxf(0.f, fminf(1.f, tau));
    rho = fmaxf(0.f, fminf(1.f, rho));
    float alpha = 1.f - tau - rho;
    if (alpha < 0.f) {
        alpha = 0.f;
    } // numerical guard
    return make_float3(tau, rho, alpha);
}

//! Reconstruct the surface normal of primitive `hit_position` from its transform matrix and type,
//! returning |cos(incidence angle)| and the hit face (face_top = true when the ray strikes the
//! +normal/top side). Mirrors the normal logic in closest_hit_direct/diffuse and the OptiX 8
//! coverCosTheta().
static __device__ __inline__ float coverNormalFaceCos(uint hit_position, const float3 &ray_dir, bool &face_top) {
    float m[16];
    for (uint i = 0; i < 16; i++) {
        m[i] = transform_matrix[optix::make_uint2(i, hit_position)];
    }
    float3 normal = make_float3(0.f, 0.f, 1.f);
    const uint ptype = primitive_type[hit_position];
    if (ptype == 0 || ptype == 3) { // patch or tile
        float3 s0 = make_float3(0, 0, 0), s1 = make_float3(1, 0, 0), s2 = make_float3(0, 1, 0);
        d_transformPoint(m, s0);
        d_transformPoint(m, s1);
        d_transformPoint(m, s2);
        normal = cross(s1 - s0, s2 - s0);
    } else if (ptype == 1) { // triangle
        float3 v0 = make_float3(0, 0, 0), v1 = make_float3(0, 1, 0), v2 = make_float3(1, 1, 0);
        d_transformPoint(m, v0);
        d_transformPoint(m, v1);
        d_transformPoint(m, v2);
        normal = cross(v1 - v0, v2 - v0);
    } else if (ptype == 2) { // disk
        float3 v0 = make_float3(0, 0, 0), v1 = make_float3(1, 0, 0), v2 = make_float3(0, 1, 0);
        d_transformPoint(m, v0);
        d_transformPoint(m, v1);
        d_transformPoint(m, v2);
        normal = cross(v1 - v0, v2 - v0);
    }
    const float nmag = d_magnitude(normal);
    if (nmag < 1e-12f) {
        face_top = true;
        return 1.f;
    }
    normal = normal / nmag;
    face_top = dot(normal, ray_dir) < 0.f;
    return fminf(1.f, fabsf(dot(ray_dir, normal)));
}

//! Whether a primitive is a translucent cover for the rays of this launch. A single ray carries every
//! launched band, so a primitive that is glass in some launched bands but opaque in others cannot be both
//! transmitted and blocked; it is a cover only if it is glass in ALL launched bands, and is otherwise a
//! normal opaque primitive for the whole ray. Glass properties do not depend on the material slot, so the
//! slot-0 slice of the material buffers is read. Mirrors coverInAllLaunchedBands() in
//! optix8/OptiX8DeviceCode.cu.
static __device__ __inline__ bool coverInAllLaunchedBands(uint position) {
    if (glass_enabled == 0u) {
        return false;
    }
    MaterialPropertyIndexer mat_indexer(Nsources + 1, Nprimitives, Nbands_global); // one slot per source plus the diffuse/scatter slot
    bool any_band_launched = false;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {
        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        any_band_launched = true;
        if (is_glass[mat_indexer(0, position, b_global)] == 0.f) {
            return false;
        }
    }
    return any_band_launched;
}

//! Angular (tau, rho, alpha) of a cover primitive in one band, for a ray at |cos(theta)| to its normal.
static __device__ __inline__ float3 coverTauRhoAlpha(uint position, int b_global, float cos_theta) {
    MaterialPropertyIndexer mat_indexer(Nsources + 1, Nprimitives, Nbands_global); // one slot per source plus the diffuse/scatter slot
    size_t mat_ind = mat_indexer(0, position, b_global);
    return glass_tau_rho_alpha(cos_theta, glass_n[mat_ind], glass_KL[mat_ind]);
}

//! Book the radiation arriving at the face of a cover primitive that launched the ray. The cover absorbs
//! alpha(theta) of it and reflects rho(theta) back out of the same face, into both the scattered-energy
//! buffers and every camera's camera-weighted scatter. The transmitted part tau(theta) continues through
//! the cover undeviated; it is delivered to the surfaces behind the cover by their own rays passing through
//! it, so it is not scattered here. A cover's reflectance is spectrally flat within a band, so each camera
//! weights it by that camera's white reference. ind_origin is the cover's [primitive][band] index. With a background field (see d_dualField()), field 1 books only what
//! the cover sends out; what it absorbs is that of the scene as it is. Mirrors depositOnCoverOrigin() in optix8/OptiX8DeviceCode.cu.
static __device__ __forceinline__ void depositOnCoverOrigin(uint origin_position, size_t ind_origin, int b_global, float cos_theta, float strength, const uint field) {
    if (strength <= 0) {
        return;
    }
    float3 tra = coverTauRhoAlpha(origin_position, b_global, cos_theta);

    if (field == 0u) {
        atomicAdd(&radiation_in[ind_origin], strength * tra.z);
        if (face_absorption_enabled && prd.face) {
            atomicAdd(&radiation_in_top[ind_origin], strength * tra.z);
        }
    }

    const size_t ind_slot = ind_origin + (size_t) field * Nprimitives * Nbands_launch;
    if (prd.face) {
        atomicFloatAdd(&scatter_buff_top[ind_slot], strength * tra.y);
    } else {
        atomicFloatAdd(&scatter_buff_bottom[ind_slot], strength * tra.y);
    }

    if (Ncameras > 0) {
        size_t cam_scatter_stride = (size_t) Nflux_slots * Nbands_launch;
        for (unsigned int cam = 0; cam < Ncameras; cam++) {
            float white = white_reference_cam[((size_t) prd.source_ID * Nbands_global + b_global) * Ncameras + cam];
            size_t ind_cam = cam * cam_scatter_stride + ind_slot;
            if (prd.face) {
                atomicFloatAdd(&scatter_buff_top_cam[ind_cam], strength * tra.y * white);
            } else {
                atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], strength * tra.y * white);
            }
        }
    }
}

//! Continue a diffuse, camera or pixel-label ray past the translucent cover it hit. What lies behind the cover is
//! seen attenuated by its transmittance: the ray's throughput (prd.cover_transmittance) is multiplied by tau(theta)
//! in every launched band, and the raygen program re-traces from just beyond the cover. The ray is not
//! refracted. Mirrors continuePastCover() in optix8/OptiX8DeviceCode.cu.
static __device__ __inline__ void continuePastCover(uint hit_position) {
    bool face_unused;
    float cos_theta = coverNormalFaceCos(hit_position, ray.direction, face_unused);

    int b = -1;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {
        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;
        if (b >= HELIOS_MAX_RADIATION_BANDS) {
            break; // host guarantees this never trips
        }
        prd.cover_transmittance[b] *= coverTauRhoAlpha(hit_position, b_global, cos_theta).x;
    }

    prd.hit_cover = true;
    prd.cover_exit = ray.origin + (t_hit + COVER_EXIT_OFFSET) * ray.direction;
}

// ---- Specular surface layer (see RayTracingMaterial::surface_model) ----
// A primitive with a layer reflects a(mu) of the radiation arriving at its surface into the second component of its outgoing
// radiation, with angular shape a(mu), and its body absorbs, reflects and transmits the rest (rho and tau are those of the
// body), sending it out in the first component with shape 1 - a(mu). Both components are stored in the outgoing and
// scattered buffers, the second after the first.

//! Start of the albedo table of the surface layer of the primitive at position pos in global band b_global in surface_layer_tables, or -1 if it has none there
static __device__ __inline__ int surfaceLayerTable(uint pos, uint b_global) {
    if (surface_layer_enabled == 0) {
        return -1;
    }
    const uint model = surface_model[pos];
    if (model == 0) {
        return -1;
    }
    const uint table = surface_model_band_table[(model - 1) * Nbands_global + b_global];
    if (table == UINT_MAX) {
        return -1;
    }
    return int(table * (surface_layer_table_size + 3));
}

//! Directional albedo a(mu) of a surface layer, interpolated linearly in its table
static __device__ __inline__ float surfaceLayerAlbedo(int table, float mu) {
    const float x = fminf(fmaxf(mu, 0.f), 1.f) * float(surface_layer_table_size - 1);
    const uint i = min(uint(x), surface_layer_table_size - 2);
    return surface_layer_tables[table + i] + (x - float(i)) * (surface_layer_tables[table + i + 1] - surface_layer_tables[table + i]);
}

//! Flux the top or bottom face of a primitive sends out at the angle whose cosine is mu_out, from its outgoing components [component][flux slot][band]; ind is the index of the first
static __device__ __inline__ float outgoingFlux(bool top, size_t ind, int table, float mu_out) {
    const float first = top ? radiation_out_top[ind] : radiation_out_bottom[ind];
    if (table < 0) {
        return first;
    }
    const size_t layer_offset = (size_t) Nflux_slots * Nbands_launch;
    const float second = top ? radiation_out_top[layer_offset + ind] : radiation_out_bottom[layer_offset + ind];
    const float a = surfaceLayerAlbedo(table, mu_out);
    const float abar = surface_layer_tables[table + surface_layer_table_size];
    return first * (1.f - a) / (1.f - abar) + second * a / abar;
}

//! Adds what the surface layer of the ray's origin reflects, of its arrival on face prd.face, to that face's second outgoing component, and to every camera's (weighted by its white reference, since the layer reflects all wavelengths
//! alike). ind_origin is the [flux slot][band] index of the origin in the field booked.
static __device__ __inline__ void depositLayerReflection(size_t ind_origin, int b_global, float reflected) {
    const size_t layer_offset = (size_t) Nflux_slots * Nbands_launch;
    if (prd.face) {
        atomicFloatAdd(&scatter_buff_top[layer_offset + ind_origin], reflected);
    } else {
        atomicFloatAdd(&scatter_buff_bottom[layer_offset + ind_origin], reflected);
    }
    for (unsigned int cam = 0; cam < Ncameras; cam++) {
        const float white = white_reference_cam[((size_t) prd.source_ID * Nbands_global + b_global) * Ncameras + cam];
        const size_t ind_cam = (Ncameras + cam) * layer_offset + ind_origin;
        if (prd.face) {
            atomicFloatAdd(&scatter_buff_top_cam[ind_cam], reflected * white);
        } else {
            atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], reflected * white);
        }
    }
}

//! Adds the diffuse radiation that a ray brings to one face of a normal-mapped primitive, times the direction it came from, to the record cameras shade the
//! primitive's scattered light with (see RayTracingGeometry::normal_map_slots and normalMapDiffuseRelief()). Nothing is recorded for other primitives.
//! The background field's moments (field 1, see d_dualField()) are in a second set of slots.
static __device__ __forceinline__ void recordIncidentDirection(uint pos, bool face_top, uint b_launch, float body_strength, const float3 &direction, const uint field) {
    if (normal_map_slots.size() <= pos || body_strength == 0.f) {
        return;
    }
    const uint slot = normal_map_slots[pos];
    if (slot == 0xFFFFFFFFu) {
        return;
    }
    const size_t base = ((size_t(slot + field * normal_map_slot_count) * 2u + (face_top ? 0u : 1u)) * Nbands_launch + b_launch) * 3u;
    if (normal_map_moments.size() < base + 3) {
        return;
    }
    atomicFloatAdd(&normal_map_moments[base], body_strength * direction.x);
    atomicFloatAdd(&normal_map_moments[base + 1], body_strength * direction.y);
    atomicFloatAdd(&normal_map_moments[base + 2], body_strength * direction.z);
}

//! The change, for a camera, in the scattered light the viewed face of a normal-mapped primitive sends out where its shading normal tilts away from the base normal
//! (the facet's, or the smoothed one) by the tangential vector m_t, given the camera-weighted reflectivity and transmissivity of the diffuse slot: (8/3) m_t . (rho
//! M_same - tau M_other) for the parts along the facet (unit normal facet_normal) of the recorded first moments M of what reached each face (see the OptiX 8
//! backend). A uniform sky gives no relief. The moments are those of radiation field `field` (see d_dualField()): the background's are in a second set of slots, and its
//! vertices follow those of the scene as it is.
static __device__ __inline__ float normalMapDiffuseRelief(uint pos, bool face_top, uint b_launch, const float3 &shading_normal, const float3 &base_normal, const float3 &facet_normal, float rho_d, float tau_d, const int *smooth_vertices,
                                                          const float *smooth_weights, uint field) {
    if (normal_map_slots.size() <= pos) {
        return 0.f;
    }
    const uint slot = normal_map_slots[pos];
    if (slot == 0xFFFFFFFFu) {
        return 0.f;
    }
    const size_t field_slot = size_t(slot) + field * normal_map_slot_count;
    const size_t same = ((field_slot * 2u + (face_top ? 0u : 1u)) * Nbands_launch + b_launch) * 3u;
    const size_t other = ((field_slot * 2u + (face_top ? 1u : 0u)) * Nbands_launch + b_launch) * 3u;
    if (normal_map_moments.size() < (same > other ? same : other) + 3) {
        return 0.f;
    }
    float3 m_same = make_float3(normal_map_moments[same], normal_map_moments[same + 1], normal_map_moments[same + 2]);
    float3 m_other = make_float3(normal_map_moments[other], normal_map_moments[other + 1], normal_map_moments[other + 2]);
    // With camera flux smoothing, the moments averaged onto the smoothing vertices, interpolated from the facet's corners (see the OptiX 8 backend)
    if (vertex_normal_map_moments.size() > 1 && smooth_vertices[0] >= 0) {
        m_same = make_float3(0.f, 0.f, 0.f);
        m_other = make_float3(0.f, 0.f, 0.f);
        const uint face_same = face_top ? 0u : 1u;
        for (int c = 0; c < 4; c++) {
            if (smooth_vertices[c] < 0) {
                continue;
            }
            const size_t vertex = size_t(smooth_vertices[c]) + size_t(field) * vertex_field_stride;
            const size_t vertex_same = ((vertex * 2u + face_same) * Nbands_launch + b_launch) * 3u;
            const size_t vertex_other = ((vertex * 2u + (1u - face_same)) * Nbands_launch + b_launch) * 3u;
            if (vertex_normal_map_moments.size() < (vertex_same > vertex_other ? vertex_same : vertex_other) + 3) {
                return 0.f;
            }
            m_same += smooth_weights[c] * make_float3(vertex_normal_map_moments[vertex_same], vertex_normal_map_moments[vertex_same + 1], vertex_normal_map_moments[vertex_same + 2]);
            m_other += smooth_weights[c] * make_float3(vertex_normal_map_moments[vertex_other], vertex_normal_map_moments[vertex_other + 1], vertex_normal_map_moments[vertex_other + 2]);
        }
    }
    float3 weighted = rho_d * m_same - tau_d * m_other;
    weighted = weighted - facet_normal * dot(weighted, facet_normal);
    const float3 m_t = shading_normal - base_normal * dot(shading_normal, base_normal);
    return (8.f / 3.f) * dot(m_t, weighted);
}

//! Fresnel + GGX reflection of a surface layer from direction to_source to direction to_viewer (unit vectors), in 1/sr, for the face whose unit normal is face_normal; 0 unless both are on that face's side
static __device__ __inline__ float surfaceLayerLobe(int table, const float3 &face_normal, const float3 &to_source, const float3 &to_viewer) {
    const float mu_s = dot(face_normal, to_source);
    const float mu_o = dot(face_normal, to_viewer);
    if (mu_s <= 0.f || mu_o <= 0.f) {
        return 0.f;
    }
    const float n = surface_layer_tables[table + surface_layer_table_size + 1];
    const float alpha = surface_layer_tables[table + surface_layer_table_size + 2];
    const float alpha2 = alpha * alpha;
    const float3 half_vector = normalize(to_source + to_viewer);
    const float mu_h = dot(face_normal, half_vector);
    const float D_denominator = mu_h * mu_h * (alpha2 - 1.f) + 1.f;
    const float D = alpha2 / (float(M_PI) * D_denominator * D_denominator);
    const float lambda_s = 0.5f * (-1.f + sqrtf(1.f + alpha2 * (1.f - mu_s * mu_s) / (mu_s * mu_s)));
    const float lambda_o = 0.5f * (-1.f + sqrtf(1.f + alpha2 * (1.f - mu_o * mu_o) / (mu_o * mu_o)));
    const float G2 = 1.f / (1.f + lambda_s + lambda_o);
    // Unpolarized Fresnel reflectance of the microfacet
    const float c = fminf(fmaxf(dot(half_vector, to_source), 0.f), 1.f);
    const float cos_t = sqrtf(fmaxf(0.f, 1.f - (1.f - c * c) / (n * n)));
    const float rs = (c - n * cos_t) / (c + n * cos_t);
    const float rp = (n * c - cos_t) / (n * c + cos_t);
    const float F = 0.5f * (rs * rs + rp * rp);
    return F * D * G2 / (4.f * mu_s * mu_o);
}

//! Flux, in the units of the outgoing buffers, that the surface layer of the primitive at position pos reflects toward to_viewer from the direct radiation of every source reaching point (radiation_specular holds the incident direct flux). camera >=
//! 0 weights each source by that camera's response. The direct flux is that of radiation field `field` (see d_dualField()).
static __device__ __forceinline__ float directLayerReflection(uint pos, uint b_launch, int table, const float3 &face_normal, const float3 &to_viewer, const float3 &point, int camera, const uint field) {
    float reflected = 0.f;
    for (uint s = 0; s < Nsources; s++) {
        float incident = radiation_specular[d_directFluxIndex(s, pos, field, b_launch)];
        if (incident <= 0.f) {
            continue;
        }
        if (camera >= 0) {
            incident *= source_fluxes_cam[((size_t) s * Nbands_launch + b_launch) * Ncameras + uint(camera)];
        }
        // A collimated source or the sun sphere lies in one direction; a finite source is seen in the direction of its centre
        const float3 to_source = (source_types[s] == 0 || source_types[s] == 2) ? normalize(source_positions[s]) : normalize(source_positions[s] - point);
        reflected += incident * surfaceLayerLobe(table, face_normal, to_source, to_viewer);
    }
    return float(M_PI) * reflected; // radiance to flux units
}

//! Flux, in the units of the outgoing buffers, that the surface layer of a normal-mapped primitive that is not smoothed reflects toward to_viewer from the direct radiation of every source, evaluated
//! with the shading normal rather than the facet's. viewed_normal is the facet's own unit normal on the side seen, and shading_normal the shading normal on that side. The incident flux carries the
//! cosine of the facet, so it is rescaled to the cosine of the shading normal; it reached the face toward the source, so only a source on the viewed side counts. The direct flux is that of
//! radiation field `field` (see d_dualField()).
static __device__ __forceinline__ float directLayerReflectionShaded(uint pos, uint b_launch, int table, const float3 &viewed_normal, const float3 &shading_normal, const float3 &to_viewer, const float3 &point, int camera, bool soften,
                                                                    const uint field) {
    float reflected = 0.f;
    for (uint s = 0; s < Nsources; s++) {
        const float3 to_source = d_directionToSource(s, point);
        const float facet_value = dot(viewed_normal, to_source) > 0.f ? radiation_specular[d_directFluxIndex(s, pos, field, b_launch)] : 0.f;
        float incident = d_faceDirect(facet_value, to_source, viewed_normal, shading_normal, soften);
        if (incident <= 0.f) {
            continue;
        }
        if (camera >= 0) {
            incident *= source_fluxes_cam[((size_t) s * Nbands_launch + b_launch) * Ncameras + uint(camera)];
        }
        reflected += incident * surfaceLayerLobe(table, shading_normal, to_source, to_viewer);
    }
    return float(M_PI) * reflected; // radiance to flux units
}

//! Whether the primitive at position pos takes part in camera flux smoothing, whose direct radiation is then recorded for the camera (see RadiationModel::subtractCameraDirectFlux())
static __device__ __inline__ bool isSmoothedPrimitive(uint pos) {
    return smoothing_vertex_indices.size() > 4 * size_t(pos) + 3 && smoothing_vertex_indices[4 * pos] >= 0;
}

//! The outgoing radiation, first component, that the face seen of a normal-mapped primitive that is not smoothed sends out of the direct radiation of every source, evaluated with the shading normal
//! (see RadiationModel::subtractCameraDirectFlux(), which removes the facet's own value from what the camera reads). viewed_normal and shading_normal are on the viewed side. The radiation reaching the
//! face toward a source is reflected by the viewed face if it is that face and transmitted to it otherwise; a surface layer, whose albedo table starts at `layer` (-1 for none), keeps 1 - a(mu) of it in
//! the body. The direct flux is that of radiation field `field` (see d_dualField()).
static __device__ __inline__ float shadedDirectBodyFlux(uint pos, uint b_launch, uint b_global, int layer, const float3 &viewed_normal, const float3 &shading_normal, const float3 &point, bool soften, uint field) {
    CameraMaterialIndexer cam_mat_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);
    float flux = 0.f;
    for (uint s = 0; s < Nsources; s++) {
        const float incident = radiation_specular[d_directFluxIndex(s, pos, field, b_launch)];
        const float3 to_source = d_directionToSource(s, point);
        const bool viewed_face_toward_source = dot(viewed_normal, to_source) > 0.f;
        const size_t material = cam_mat_indexer(s, pos, b_global, camera_ID);
        const float rho = rho_cam[material];
        const float tau = tau_cam[material];
        if (rho > 0.f) {
            const float reaching = d_faceDirect(viewed_face_toward_source ? incident : 0.f, to_source, viewed_normal, shading_normal, soften);
            flux += rho * reaching * (layer >= 0 ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, dot(shading_normal, to_source))) : 1.f);
        }
        if (tau > 0.f) {
            const float reaching = d_faceDirect(viewed_face_toward_source ? 0.f : incident, to_source, -viewed_normal, -shading_normal, soften);
            flux += tau * reaching * (layer >= 0 ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, -dot(shading_normal, to_source))) : 1.f);
        }
    }
    return flux;
}

//! The camera-weighted outgoing radiation of the class of class-flux slot `slot` on a normal-mapped primitive at `position` that is not smoothed, from one face in a launch band, of the sources'
//! direct radiation alone: taken from the facet itself and carried over to the shading normal (see d_faceDirect()). viewed_normal and shading_normal are on the side of `top_face`, and point is where the
//! camera ray hit. soften applies the shadow-terminator softening of a normal map. A surface layer, whose albedo table starts at `layer` (-1 for none), keeps 1 - a(mu) of the radiation reaching each face
//! in the body, as in shadedDirectBodyFlux(). The slot and the direct radiation are those of radiation field `field` (see d_dualField()).
static __device__ __inline__ float d_spectralDirectClassFlux(uint position, uint slot, bool top_face, uint launch_band, uint b_global, uint camera, int layer, const float3 &viewed_normal, const float3 &shading_normal, const float3 &point,
                                                             bool soften, uint field) {
    const uint entry = spectral_palette[spectral_palette[3] + position] + (slot - spectral_camera_flux[3 + d_fluxSlot(position, field)]);
    float flux = 0.f;
    for (uint source_slot = 0; source_slot < Nsources; source_slot++) {
        const float3 to_source = d_directionToSource(source_slot, point);
        const float incident = radiation_specular[d_directFluxIndex(source_slot, position, field, launch_band)];
        const bool viewed_face_toward_source = dot(viewed_normal, to_source) > 0.f;
        const float same_value = d_faceDirect(viewed_face_toward_source ? incident : 0.f, to_source, viewed_normal, shading_normal, soften);
        const float other_value = d_faceDirect(viewed_face_toward_source ? 0.f : incident, to_source, -viewed_normal, -shading_normal, soften);
        const float same_body = layer >= 0 ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, dot(shading_normal, to_source))) : 1.f;
        const float other_body = layer >= 0 ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, -dot(shading_normal, to_source))) : 1.f;
        flux += d_spectralPaletteValueCam(7, source_slot, entry, b_global, camera) * same_value * same_body + d_spectralPaletteValueCam(8, source_slot, entry, b_global, camera) * other_value * other_body;
    }
    return flux;
}

//! Global index of the launched band with launch index b_launch
static __device__ __inline__ uint globalBandIndex(uint b_launch) {
    uint b = 0;
    for (uint b_global = 0; b_global < Nbands_global; b_global++) {
        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        if (b == b_launch) {
            return b_global;
        }
        b++;
    }
    return 0; // unreachable: b_launch is always below the number of launched bands
}

// A scene with a background field (see d_dualField()) traces its direct and diffuse rays as the ray types direct_dual_ray_type and diffuse_dual_ray_type, which have their own
// closest-hit and miss programs (*_dual), so that the programs of a scene without one are those of one field and nothing more.

RT_PROGRAM void closest_hit_direct() {

    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if position is invalid
    if (hit_position == UINT_MAX) {
        return;
    }

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition
        d_handlePeriodicBoundaryHit(prd, ray.origin, ray.direction, periodic_face_t);
    }
};

//! closest_hit_direct with a background field: an excluded primitive that a dual ray passed before it left the central copy of the domain blocks it in the scene as it is
RT_PROGRAM void closest_hit_direct_dual() {

    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if position is invalid
    if (hit_position == UINT_MAX) {
        return;
    }

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition
        // The face is reported no nearer than the ray's minimum distance; periodic_face_t is its true distance
        if (prd.dual_ray && prd.recorded_t < periodic_face_t) {
            prd.excluded_resolved = true;
        }
        d_handlePeriodicBoundaryHit(prd, ray.origin, ray.direction, periodic_face_t);
    }
};

//! Books on the primitive a diffuse ray was launched from the radiation it receives from a surface the ray reaches
/**
 * The surface is the primitive at hit_position, reached at distance hit_t with hit attributes hit_u and hit_v (see primitiveIntersection.cu). The ray books in fields
 * [field_first, field_end) (see d_dualField()): the scene as it is (0) reads the surface's outgoing radiation in field seen_field, the field of the copy of the domain the
 * surface is in, and the background (1) always reads the background's. Without a background field this is the one field, 0. The fields are arguments so that, inlined with
 * constants for a scene without a background field, the code is that of one field. Must match gatherFromSurface() in OptiX8DeviceCode.cu.
 */
static __device__ __forceinline__ void gatherFromSurface(uint origin_position, uint hit_position, float hit_u, float hit_v, float hit_t, const uint field_first, const uint field_end, const uint seen_field) {

    // Create indexers for buffer access
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    MaterialPropertyIndexer mat_indexer(Nsources + 1, Nprimitives, Nbands_global); // one slot per source plus the diffuse/scatter slot
    CameraMaterialIndexer cam_mat_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);
    const size_t slot_stride = (size_t) Nflux_slots * Nbands_launch; // one [flux slot][band] block of the outgoing, scattered and camera buffers

    // Note: hit_position is the primitive that the ray reached (i.e., where energy is coming from), and origin_position the primitive the ray originated from (i.e., where the energy is being recieved)

    // find out if we hit top or bottom surface
    float3 normal;

    float m[16];
    for (uint i = 0; i < 16; i++) {
        m[i] = transform_matrix[optix::make_uint2(i, hit_position)];
    }

    if (primitive_type[hit_position] == 0 || primitive_type[hit_position] == 3) { // hit patch or tile
        float3 s0 = make_float3(0, 0, 0);
        float3 s1 = make_float3(1, 0, 0);
        float3 s2 = make_float3(0, 1, 0);
        d_transformPoint(m, s0);
        d_transformPoint(m, s1);
        d_transformPoint(m, s2);
        normal = cross(s1 - s0, s2 - s0);
    } else if (primitive_type[hit_position] == 1) { // hit triangle
        float3 v0 = make_float3(0, 0, 0);
        d_transformPoint(m, v0);
        float3 v1 = make_float3(0, 1, 0);
        d_transformPoint(m, v1);
        float3 v2 = make_float3(1, 1, 0);
        d_transformPoint(m, v2);
        normal = cross(v1 - v0, v2 - v0);
    } else if (primitive_type[hit_position] == 2) { // hit disk
        float3 v0 = make_float3(0, 0, 0);
        d_transformPoint(m, v0);
        float3 v1 = make_float3(1, 0, 0);
        d_transformPoint(m, v1);
        float3 v2 = make_float3(0, 1, 0);
        d_transformPoint(m, v2);
        normal = cross(v1 - v0, v2 - v0);
    }
    normal = normalize(normal);

    bool face = dot(normal, ray.direction) < 0;

    // A translucent cover books what reaches it with its angular glass properties, not rho/tau.
    bool origin_face_unused;
    const bool origin_is_cover = coverInAllLaunchedBands(origin_position);
    const float origin_cos = origin_is_cover ? coverNormalFaceCos(origin_position, ray.direction, origin_face_unused) : 1.f;

    // The angle-dependent terms of a hit primitive's surface layer see the ray at the angle of its shading normal; a Lambertian surface sends out the same radiance at every angle, so only a
    // layered primitive with a normal map needs the shading normal here
    const uint hit_map = d_normalMapOf(hit_position);
    const bool hit_shaded = hit_map != helios_nm::NO_MAP && d_hasSurfaceLayer(hit_position);
    float3 hit_normal_top = normal;
    float3 hit_shading_top = normal;
    if (hit_shaded) {
        hit_shading_top = d_hitShadingNormalTop(hit_position, hit_map, hit_u, hit_v, false, hit_normal_top);
    }
    const float hit_cos = hit_shaded ? fabsf(dot(hit_shading_top, ray.direction)) : fabsf(dot(normal, ray.direction));

    int b = -1;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {

        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;

        // Use BufferIndexer for radiation buffers: [primitive][band]
        // NOTE: b should match the band's index in the original band_labels array
        // This is correct as long as band_launch_flag isn't modified after launch
        size_t ind_origin = rad_indexer(origin_position, b);

        // A hit primitive with a surface layer sends out its two components, in proportions that depend on the angle
        const int hit_layer = surfaceLayerTable(hit_position, b_global);

        // What the surface sends toward the ray, in each field read
        float outgoing_of_field[2] = {0.f, 0.f};
        bool outgoing_known[2] = {false, false};
        for (uint field = field_first; field < field_end; field++) {
            const uint source_field = field == 1u ? 1u : seen_field;
            if (outgoing_known[source_field]) {
                continue;
            }
            const size_t ind_hit = (size_t) d_fluxSlot(hit_position, source_field) * Nbands_launch + b;
            float outgoing = outgoingFlux(face, ind_hit, hit_layer, hit_layer >= 0 ? hit_cos : 1.f);
            if (hit_layer >= 0 && specular_reflection_enabled != 0) {
                // In the launch that reads the first scattering of the direct radiation, add the layer's reflection lobe of it
                if (hit_shaded) {
                    // The lobe around the shading normal, with the incident flux carried from the facet's cosine to the shading normal's
                    const float3 viewed_normal = face ? normal : -normal;
                    const float3 viewed_shading = dot(hit_normal_top, viewed_normal) > 0.f ? hit_shading_top : -hit_shading_top;
                    outgoing += directLayerReflectionShaded(hit_position, b, hit_layer, viewed_normal, viewed_shading, -ray.direction, ray.origin + hit_t * ray.direction, -1, true, source_field);
                } else {
                    outgoing += directLayerReflection(hit_position, b, hit_layer, face ? normal : -normal, -ray.direction, ray.origin + hit_t * ray.direction, -1, source_field);
                }
            }
            outgoing_of_field[source_field] = outgoing;
            outgoing_known[source_field] = true;
        }

        // Forward scattering estimate (see RadiationModel::runBand()): the flux this ray carries away from the origin along its
        // path, times the fraction of it the surface it reaches scatters again. A cover reflects its reflectance; what it transmits
        // is counted where the ray continues, attenuated by the cover transmittance. A hit surface's own reflectivity is used, not
        // that of a spectral-texture class at the hit point: the estimate need only be unbiased over the scene, not per texel. It is
        // that of the scene as it is (field 0, whose flux slots are the primitives' positions), where that field's ray ends.
        if (forward_scatter_enabled && field_first == 0u) {
            const float launched = outgoingFlux(prd.face, ind_origin, surfaceLayerTable(origin_position, b_global), prd.origin_cos);
            if (launched != 0.f) {
                float rescattered;
                if (coverInAllLaunchedBands(hit_position)) {
                    bool hit_face_unused;
                    rescattered = coverTauRhoAlpha(hit_position, b_global, coverNormalFaceCos(hit_position, ray.direction, hit_face_unused)).y;
                } else {
                    const size_t hit_radprop_ind = mat_indexer(prd.source_ID, hit_position, b_global);
                    rescattered = rho[hit_radprop_ind] + tau[hit_radprop_ind];
                    if (hit_layer >= 0) {
                        const float hit_albedo = surfaceLayerAlbedo(hit_layer, hit_cos);
                        rescattered = hit_albedo + (1.f - hit_albedo) * rescattered;
                    }
                }
                const float path_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;
                atomicAdd(&forward_scatter[ind_origin], fabsf(launched) * prd.strength * path_tau * rescattered);
            }
        }

        for (uint field = field_first; field < field_end; field++) {
            const uint source_field = field == 1u ? 1u : seen_field;
            // The origin's [flux slot][band] index in the field booked
            const size_t ind_slot = ind_origin + (size_t) field * Nprimitives * Nbands_launch;

            double strength = outgoing_of_field[source_field] * prd.strength;

            // Attenuated by any translucent covers the ray passed on its way to the surface.
            strength *= (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;

            if (strength == 0) {
                continue;
            }

            if (origin_is_cover) {
                depositOnCoverOrigin(origin_position, ind_origin, b_global, origin_cos, strength, field);
                continue;
            }

            // Use BufferIndexer for material properties: [source][primitive][band]
            size_t radprop_ind_global = mat_indexer(prd.source_ID, origin_position, b_global);
            float t_rho = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValue(5, prd.source_ID, prd.palette_entry, b_global) : rho[radprop_ind_global];
            float t_tau = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValue(6, prd.source_ID, prd.palette_entry, b_global) : tau[radprop_ind_global];

            // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest
            const int layer = surfaceLayerTable(origin_position, b_global);
            const float layer_albedo = layer >= 0 ? surfaceLayerAlbedo(layer, prd.origin_cos) : 0.f;
            const double body_strength = layer >= 0 ? strength * (1.0 - layer_albedo) : strength;
            if (layer_albedo > 0.f) {
                depositLayerReflection(ind_slot, b_global, strength * layer_albedo);
            }
            recordIncidentDirection(origin_position, prd.face, b, float(body_strength), ray.direction, field);

            // absorption - calculate with defensive check for energy conservation violations
            float absorption_factor = 1.f - t_rho - t_tau;
            float contribution = body_strength * absorption_factor;


#ifndef NDEBUG
            if (absorption_factor < -1e-5f) {
                printf("ERROR: Negative absorption! rho=%.6f, tau=%.6f, origin_UUID=%u\n", t_rho, t_tau, prd.origin_UUID);
                absorption_factor = 0.f;
            }
#endif
            if (field == 0u) { // what is absorbed is that of the scene as it is
                atomicAdd(&radiation_in[ind_origin], contribution);
                if (face_absorption_enabled && prd.face) {
                    atomicAdd(&radiation_in_top[ind_origin], contribution);
                }
            }

            // The gathered flux is negative where a surface radiates less than the reference field (see RadiationModel::runBand())
            if (t_rho > 0 || t_tau > 0) {
                if (prd.face) { // reflection from top, transmission from bottom
                    atomicFloatAdd(&scatter_buff_top[ind_slot], body_strength * t_rho); // reflection
                    atomicFloatAdd(&scatter_buff_bottom[ind_slot], body_strength * t_tau); // transmission
                } else { // reflection from bottom, transmission from top
                    atomicFloatAdd(&scatter_buff_bottom[ind_slot], body_strength * t_rho); // reflection
                    atomicFloatAdd(&scatter_buff_top[ind_slot], body_strength * t_tau); // transmission
                }
            }
            // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
            if (spectral_incident_enabled != 0u && spectral_palette[spectral_palette[3] + origin_position] != 0xFFFFFFFFu && body_strength != 0) {
                const uint face_slot = (prd.face ? 0u : Nsources + 1u) + prd.source_ID;
                atomicFloatAdd(&spectral_incident[(size_t) face_slot * slot_stride + ind_slot], body_strength);
            }
            if (Ncameras > 0) {
                // Camera-weighted scatter, accumulated once per camera into that camera's [flux slot][band]
                // block. This launch runs once per dispatch, not once per camera, so camera_ID is not
                // meaningful here; weighting by a single camera handed every other camera that camera's image.
                size_t cam_scatter_stride = slot_stride;
                for (unsigned int cam = 0; cam < Ncameras; cam++) {
                    // Use BufferIndexer for camera material: [source][primitive][band][camera]
                    size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, cam);
                    float t_rho_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(7, prd.source_ID, prd.palette_entry, b_global, cam) : rho_cam[indc];
                    float t_tau_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(8, prd.source_ID, prd.palette_entry, b_global, cam) : tau_cam[indc];
                    size_t ind_cam = cam * cam_scatter_stride + ind_slot;
                    if ((t_rho_cam > 0 || t_tau_cam > 0) && body_strength != 0) {
                        if (prd.face) { // reflection from top, transmission from bottom
                            atomicFloatAdd(&scatter_buff_top_cam[ind_cam], body_strength * t_rho_cam); // reflection
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], body_strength * t_tau_cam); // transmission
                        } else { // reflection from bottom, transmission from top
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], body_strength * t_rho_cam); // reflection
                            atomicFloatAdd(&scatter_buff_top_cam[ind_cam], body_strength * t_tau_cam); // transmission
                        }
                    }
                }
                // Note: Don't accumulate scattered radiation to radiation_specular
                // Specular should only reflect DIRECT source radiation (accumulated in miss_direct)
            }
        }
    }
}

//! Settles what the scene as it is receives along a dual diffuse ray that passed an excluded primitive nearer than t_end, where its current segment ends
/**
 * A dual ray (PerRayData::dual_ray) passes the primitives excluded from the periodic boundary, so that it reaches what the background field's ray reaches. In the scene as it
 * is the ray ends on the nearest excluded primitive it passed, whose radiation that field receives here, with the transmittance of the covers crossed so far. It must be
 * called before anything else is booked for the segment or a cover's transmittance is applied. Must match resolveRecordedHit() in OptiX8DeviceCode.cu.
 */
static __device__ __inline__ void resolveRecordedHit(uint origin_position, float t_end) {
    if (!prd.dual_ray || prd.excluded_resolved || !(prd.recorded_t < t_end)) {
        return;
    }
    gatherFromSurface(origin_position, prd.recorded_position, prd.recorded_u, prd.recorded_v, prd.recorded_t, 0u, 1u, 0u);
    prd.excluded_resolved = true;
}

//! The diffuse closest-hit program, for a scene with a background field (dual) or without; inlined with a constant, each is its own code. Must match closestHitDiffuse() in OptiX8DeviceCode.cu.
static __device__ __forceinline__ void closestHitDiffuse(const bool dual) {

    // Convert UUIDs to array positions for buffer indexing
    uint origin_position = primitive_positions[prd.origin_UUID];
    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if positions are invalid
    if (origin_position == UINT_MAX || hit_position == UINT_MAX) {
        return;
    }

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        if (dual) {
            // The face is reported no nearer than the ray's minimum distance; periodic_face_t is its true distance
            resolveRecordedHit(origin_position, periodic_face_t);
        }
        d_handlePeriodicBoundaryHit(prd, ray.origin, ray.direction, periodic_face_t);

    } else {

        if (!dual) {
            gatherFromSurface(origin_position, hit_position, surface_u, surface_v, t_hit, 0u, 1u, 0u);
        } else {
            // An excluded primitive the ray passed before this surface is what the scene as it is receives from, in place of the surface
            resolveRecordedHit(origin_position, t_hit);
            gatherFromSurface(origin_position, hit_position, surface_u, surface_v, t_hit, prd.excluded_resolved ? 1u : 0u, prd.dual_ray ? 2u : 1u, d_fieldSeenBy(prd));
        }

        // A translucent cover: its own outgoing radiance (reflection and emission) was received above like that of any
        // surface; continue past it to what lies behind, attenuated by its transmittance.
        if (coverInAllLaunchedBands(hit_position)) {
            continuePastCover(hit_position);
        }
    }
}

RT_PROGRAM void closest_hit_diffuse() {
    closestHitDiffuse(false);
}

//! closest_hit_diffuse with a background field
RT_PROGRAM void closest_hit_diffuse_dual() {
    closestHitDiffuse(true);
}

//! With camera flux smoothing: the direct radiation of every source at the point a camera ray hit, evaluated for this pixel sample with a shadow ray toward one point sampled on each source
/**
 * Mirrors pixelDirectRadiation() in optix8/OptiX8DeviceCode.cu, which describes the method. The shadow ray is of camera_shadow_ray_type, which has the direct ray type's hit programs and a miss program
 * that marks the source reached (miss_camera_shadow). m is the facet's transform, prim_type its primitive type, viewed_normal its normal, base_normal the smoothed normal and shading_normal the normal the
 * camera shades with, all on the side seen; v and w are the facet's smoothing vertices and their weights at the hit point, and entry the palette entry of a spectral-texture class (0xFFFFFFFF for none).
 * Fills the launch bands from band_begin on, at most HELIOS_MAX_RADIATION_BANDS of them, indexed from band_begin.
 */
static __device__ __inline__ void d_pixelDirectRadiation(uint hit_UUID, uint pos, uint prim_type, const float (&m)[16], const float3 &viewed_normal, const float3 &base_normal, const float3 &shading_normal, bool normal_mapped, bool face_top,
                                                         const float3 &point, const float3 &to_viewer, const int v[4], const float w[4], uint entry, uint band_begin, uint &seed, float *body, float *class_flux, float *lobe) {
    const uint band_end = band_begin + HELIOS_MAX_RADIATION_BANDS < Nbands_launch ? band_begin + HELIOS_MAX_RADIATION_BANDS : uint(Nbands_launch);
    for (uint b = 0; b < HELIOS_MAX_RADIATION_BANDS; b++) {
        body[b] = 0.f;
        class_flux[b] = 0.f;
        lobe[b] = 0.f;
    }
    CameraMaterialIndexer cam_mat_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);
    const int twosided = twosided_flag[pos];

    for (uint s = 0; s < Nsources; s++) {
        float3 direction;
        float tmax;
        double factor;
        if (!d_sampleSourceDirection(s, point, seed, direction, tmax, factor)) {
            continue;
        }

        // The face whose shading normal faces the sampled point receives its radiation
        const float mu_viewed = dot(shading_normal, direction);
        if (mu_viewed == 0.f) {
            continue;
        }
        const bool reaches_viewed_face = mu_viewed > 0.f;
        const float sign = reaches_viewed_face ? 1.f : -1.f;
        const float mu = fabsf(mu_viewed);
        const bool receiving_top = reaches_viewed_face == face_top;
        if (twosided == 3 || (!receiving_top && twosided != 1)) {
            continue; // an invisible primitive, or the bottom face of a one-sided one
        }
        const float3 face_normal = sign * viewed_normal;
        const float3 face_base_normal = sign * base_normal;
        const float3 face_shading_normal = sign * shading_normal;

        // Where the facet faces away from the source, start from the point raised onto the smooth surface the vertex normals describe
        float3 origin = point;
        if (dot(face_normal, direction) <= 0.f && (prim_type == 0 || prim_type == 1 || prim_type == 3)) {
            for (int c = 0; c < 4; c++) {
                if (v[c] < 0) {
                    continue;
                }
                float3 corner = prim_type == 1 ? make_float3(c == 2 ? 1.f : 0.f, c == 0 ? 0.f : 1.f, 0.f) : make_float3((c == 1 || c == 2) ? 0.5f : -0.5f, (c == 2 || c == 3) ? 0.5f : -0.5f, 0.f);
                d_transformPoint(m, corner);
                float3 corner_normal = make_float3(vertex_normals[3 * v[c] + 0], vertex_normals[3 * v[c] + 1], vertex_normals[3 * v[c] + 2]);
                if (dot(corner_normal, face_normal) < 0.f) {
                    corner_normal = -corner_normal;
                }
                const float below = dot(point - corner, corner_normal);
                if (below < 0.f) {
                    origin -= (w[c] * below) * corner_normal;
                }
            }
        }

        // Off the receiving face by a few times the rounding error of the hit point (t_hit from the camera), so that the ray cannot strike the facet it leaves, or a coplanar neighbour, from just behind it
        origin += (1e-6f * (1.f + fmaxf(fmaxf(fabsf(point.x), fabsf(point.y)), fabsf(point.z)) + t_hit)) * face_normal;

        // The shadow ray: the direct ray type's hit programs, and a miss program that marks the source reached. A ray that reaches the wrap limit counts as blocked.
        PerRayData shadow;
        shadow.strength = 0.0;
        shadow.origin_UUID = hit_UUID;
        shadow.face = receiving_top;
        shadow.seed = seed;
        shadow.source_ID = (unsigned char) s;
        shadow.hit_periodic_boundary = false;
        // The shadow ray leaves the point in the periodic copy of the domain the camera ray reached it in
        shadow.tile_x = prd.tile_x;
        shadow.tile_y = prd.tile_y;
        shadow.origin_tile_x = prd.tile_x; // the point it leaves is on the copy of the primitive in that copy of the domain
        shadow.origin_tile_y = prd.tile_y;
        shadow.origin_cos = mu;
        shadow.palette_entry = 0xFFFFFFFFu;
        shadow.hit_cover = false;
        shadow.source_reached = false;
        initCoverTransmittance(shadow);
        optix::Ray shadow_ray = optix::make_Ray(origin, direction, camera_shadow_ray_type, 0.f, tmax);
        for (int wrap = 0; wrap < MAX_PERIODIC_WRAPS; ++wrap) {
            rtTrace(top_object, shadow_ray, shadow);
            if (!shadow.hit_periodic_boundary) {
                break;
            }
            shadow_ray.origin = shadow.periodic_hit;
            shadow_ray.tmin = 0.f;
            if (source_types[s] != 0) {
                shadow_ray.tmax = fmaxf(shadow_ray.tmax - shadow.periodic_hit_t, 2e-4f);
            }
            shadow.hit_periodic_boundary = false;
        }
        if (!shadow.source_reached) {
            continue;
        }

        // A normal map softens the shading cosine where it tilts the normal past the smooth surface's terminator
        const float softening = normal_mapped ? helios_nm::terminatorSoftening(d_toNM(face_base_normal), d_toNM(face_shading_normal), d_toNM(direction)) : 1.f;
        const float irradiance_factor = float(factor) * mu * softening;
        if (irradiance_factor <= 0.f) {
            continue;
        }

        int b = -1;
        for (uint b_global = 0; b_global < Nbands_global; b_global++) {
            if (band_launch_flag[b_global] == 0) {
                continue;
            }
            b++;
            if (uint(b) < band_begin || uint(b) >= band_end) {
                continue;
            }
            const uint slot = uint(b) - band_begin;
            const float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? shadow.cover_transmittance[b] : 1.f;
            const float reaching = source_fluxes[s * Nbands_launch + uint(b)] * irradiance_factor * cover_tau;
            if (reaching <= 0.f) {
                continue;
            }
            const size_t material = cam_mat_indexer(s, pos, b_global, camera_ID);
            const int layer = surfaceLayerTable(pos, b_global);
            const float body_fraction = layer >= 0 ? 1.f - surfaceLayerAlbedo(layer, mu) : 1.f;
            body[slot] += (reaches_viewed_face ? rho_cam[material] : tau_cam[material]) * reaching * body_fraction;
            if (entry != 0xFFFFFFFFu) {
                class_flux[slot] += d_spectralPaletteValueCam(reaches_viewed_face ? 7 : 8, s, entry, b_global, camera_ID) * reaching * body_fraction;
            }
            if (layer >= 0 && reaches_viewed_face && specular_reflection_enabled != 0) {
                const float camera_weight = source_fluxes_cam[((size_t) s * Nbands_launch + uint(b)) * Ncameras + camera_ID];
                lobe[slot] += float(M_PI) * reaching * camera_weight * surfaceLayerLobe(layer, shading_normal, direction, to_viewer);
            }
        }
    }
}

RT_PROGRAM void closest_hit_camera() {

    // Convert UUID to array position
    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if position is invalid
    if (hit_position == UINT_MAX) {
        return;
    }

    // For cameras, origin_UUID is actually the pixel index (not a primitive UUID!)
    uint pixel_index = prd.origin_UUID;
    size_t Npixels = camera_resolution_full.x * camera_resolution_full.y;

    // Create indexers
    RadiationBufferIndexer rad_indexer(Npixels, Nbands_launch); // Use Npixels for camera radiation buffer
    SourceFluxIndexer source_flux_indexer(Nsources, Nbands_launch);

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        d_handlePeriodicBoundaryHit(prd, ray.origin, ray.direction, periodic_face_t);

    } else {

        // Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being received)

        // find out if we hit top or bottom surface per each ray
        float3 normal;

        float m[16];
        for (uint i = 0; i < 16; i++) {
            m[i] = transform_matrix[optix::make_uint2(i, hit_position)];
        }

        if (primitive_type[hit_position] == 0 || primitive_type[hit_position] == 3) { // hit patch or tile
            float3 s0 = make_float3(0, 0, 0);
            float3 s1 = make_float3(1, 0, 0);
            float3 s2 = make_float3(0, 1, 0);
            d_transformPoint(m, s0);
            d_transformPoint(m, s1);
            d_transformPoint(m, s2);
            normal = cross(s1 - s0, s2 - s0);
        } else if (primitive_type[hit_position] == 1) { // hit triangle
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(0, 1, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(1, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        } else if (primitive_type[hit_position] == 2) { // hit disk
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(1, 0, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(0, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        }
        normal = normalize(normal);

        bool face = dot(normal, ray.direction) < 0;

        // The normal of the face the camera sees
        const float3 viewed_normal = face ? normal : -normal;

        // With a background radiation field (see d_dualField()), the camera sees the scene as it is in the central copy of the domain, where the primitives excluded from the periodic
        // boundary stand, and the background in every other copy: the field's flux slot of the primitive, and its vertex values after those of the scene as it is. Must match
        // __closesthit__camera in OptiX8DeviceCode.cu.
        const uint field = d_fieldSeenBy(prd);
        const uint flux_position = d_fluxSlot(hit_position, field);
        const size_t vertex_field_offset = size_t(field) * vertex_field_stride * Nbands_launch;


        float3 camera_normal = d_rotatePoint(make_float3(0, 0, 1), -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y);

        // Camera flux smoothing: the shared vertices of the facet that was hit, and the weights to blend them with. Done once for the whole hit rather than once per band. A primitive taking no part in
        // smoothing keeps the single value of the facet behind the pixel.
        int smooth_vertices[4] = {-1, -1, -1, -1};
        float smooth_weights[4] = {0.f, 0.f, 0.f, 0.f};
        if (smoothing_vertex_indices.size() > 4 * hit_position + 3 && vertex_radiation_out_top.size() > 0 && smoothing_vertex_indices[4 * hit_position + 0] >= 0) {
            for (int c = 0; c < 4; c++) {
                smooth_vertices[c] = smoothing_vertex_indices[4 * hit_position + c];
            }
            if (smooth_vertices[3] < 0) {
                // Triangle: barycentric weights, with the attributes holding the weights of the second and third corners.
                smooth_weights[1] = surface_u;
                smooth_weights[2] = surface_v;
                smooth_weights[0] = 1.f - surface_u - surface_v;
            } else {
                // Patch: bilinear weights over the patch-local coordinates, whose corners run (0,0), (1,0), (1,1), (0,1).
                smooth_weights[0] = (1.f - surface_u) * (1.f - surface_v);
                smooth_weights[1] = surface_u * (1.f - surface_v);
                smooth_weights[2] = surface_u * surface_v;
                smooth_weights[3] = (1.f - surface_u) * surface_v;
            }
        }
        const int smooth_v0 = smooth_vertices[0];

        // With smoothing and a specular surface layer: a shading normal interpolated from the facet's vertices, for the layer's angle-dependent terms. Each vertex normal is first turned to the side of the
        // facet, so that the result does not depend on how the facet is wound. The facet's own normal is kept in the rare case where the vertex normals interpolate to nothing on its side.
        const float3 hit_point = ray.origin + t_hit * ray.direction;
        const uint vertex_normal_count = uint(vertex_normals.size() / 3);
        const bool smooth_shading = smooth_v0 >= 0 && vertex_normal_count > 0;
        float3 shading_normal = viewed_normal;
        if (smooth_shading) {
            float3 interpolated = make_float3(0.f, 0.f, 0.f);
            for (int c = 0; c < 4; c++) {
                if (smooth_vertices[c] < 0) {
                    continue;
                }
                float3 vertex_normal = make_float3(vertex_normals[3 * smooth_vertices[c] + 0], vertex_normals[3 * smooth_vertices[c] + 1], vertex_normals[3 * smooth_vertices[c] + 2]);
                if (dot(vertex_normal, viewed_normal) < 0.f) {
                    vertex_normal = -vertex_normal;
                }
                interpolated += smooth_weights[c] * vertex_normal;
            }
            if (dot(interpolated, viewed_normal) > 1e-6f) {
                shading_normal = normalize(interpolated);
            }
        }
        // A normal map tilts the shading normal (the facet's, or the smoothed one) by the texel at the hit point, bilinearly interpolated so that a magnified map does not show its texels
        const uint hit_map = d_normalMapOf(hit_position);
        const bool normal_mapped = hit_map != helios_nm::NO_MAP;
        // The normal the map tilts (the facet's or the smoothed one), on the viewed side
        const float3 base_viewed = shading_normal;
        if (normal_mapped) {
            const uint hit_type = primitive_type[hit_position];
            const float3 base_top = face ? shading_normal : -shading_normal;
            const float3 m_top = d_normalMapShadingNormalAt(hit_type, hit_map, m, uvID[hit_position], base_top, d_canonicalPointFromHit(hit_type, surface_u, surface_v), true);
            shading_normal = face ? m_top : -m_top;
        }
        // The angle-dependent terms (the direct radiation per hit point, a surface layer's viewing angle and lobe) use the shading normal when there is one
        const bool use_shading_normal = smooth_shading || normal_mapped;
        // A normal-mapped primitive's direct radiation is shaded per hit point when the host removed what the direct launch deposited from the outgoing radiation the camera reads. With primitives
        // excluded from the periodic boundary, the host removed it for every primitive but a translucent cover, and it is evaluated per pixel sample with a shadow ray, so that the shadow of an object
        // that does not repeat falls only where that object puts it, not on every periodic copy of the primitives it shades.
        const bool hit_is_cover = coverInAllLaunchedBands(hit_position);
        const bool direct_all = camera_direct_all != 0u && !hit_is_cover;
        const bool shade_direct_per_hit = direct_all || (normal_mapped ? normal_map_direct != 0u : smooth_shading);
        const bool direct_recorded = Nsources > 0 && radiation_specular.size() >= size_t(Nsources) * Nflux_slots * Nbands_launch;

        // The palette entry of the class of a spectral texture at the hit point, or 0xFFFFFFFF
        const uint hit_class_slot = d_spectralCameraClassSlot(hit_position, primitive_type[hit_position], surface_u, surface_v, uvID[hit_position], field);
        const uint hit_entry = hit_class_slot == 0xFFFFFFFFu ? 0xFFFFFFFFu : spectral_palette[spectral_palette[3] + hit_position] + (hit_class_slot - spectral_camera_flux[3 + flux_position]);

        // With smoothing, the sources' direct radiation is evaluated for this pixel sample, with a shadow ray toward each source, so that a cast shadow falls where the geometry puts it within the facet
        // (see d_pixelDirectRadiation()). A translucent cover's own direct radiation stays in the outgoing radiation its vertices carry.
        // It is evaluated for up to HELIOS_MAX_RADIATION_BANDS bands at a time; every group of bands draws the same points on the sources, so all bands see the same shadows.
        const bool per_pixel_direct = camera_direct_per_pixel != 0u && shade_direct_per_hit && (smooth_shading || direct_all) && direct_recorded && !hit_is_cover;
        float pixel_body[HELIOS_MAX_RADIATION_BANDS];
        float pixel_class[HELIOS_MAX_RADIATION_BANDS];
        float pixel_lobe[HELIOS_MAX_RADIATION_BANDS];
        uint pixel_band_begin = 0;
        const uint seed_at_hit = prd.seed;

        double strength;
        for (size_t b = 0; b < Nbands_launch; b++) {
            if (per_pixel_direct && b % HELIOS_MAX_RADIATION_BANDS == 0) {
                uint band_group_seed = seed_at_hit;
                d_pixelDirectRadiation(UUID, hit_position, primitive_type[hit_position], m, viewed_normal, base_viewed, shading_normal, normal_mapped, face, hit_point, -ray.direction, smooth_vertices, smooth_weights, hit_entry, uint(b),
                                       band_group_seed, pixel_body, pixel_class, pixel_lobe);
                prd.seed = band_group_seed;
                pixel_band_begin = uint(b);
            }

            // Check if any light sources are between camera and hit point
            float source_radiance = 0.0f;
            for (uint s = 0; s < Nsources; s++) {
                float flux = source_fluxes[s * Nbands_launch + b];
                if (flux <= 0.0f)
                    continue;

                uint source_type = source_types[s];

                if (source_type == 1) {
                    // SPHERE
                    float radius = source_widths[s].x * 0.5f;
                    float3 oc = ray.origin - source_positions[s];
                    float b = dot(oc, ray.direction);
                    float c = dot(oc, oc) - radius * radius;
                    float discriminant = b * b - c;

                    if (discriminant >= 0.0f) {
                        float t_sphere = -b - sqrtf(discriminant);
                        if (t_sphere > 0.0f && t_sphere < t_hit) {
                            float area = 4.0f * M_PI * radius * radius;
                            source_radiance += (flux / area) / M_PI;
                        }
                    }
                } else if (source_type == 3) {
                    // RECTANGLE
                    float transform[16];
                    d_makeTransformMatrix(source_rotations[s], transform);
                    float3 normal = make_float3(transform[2], transform[6], transform[10]);

                    float denom = dot(ray.direction, normal);
                    if (denom < -1e-6f) {
                        float3 oc = source_positions[s] - ray.origin;
                        float t_rect = dot(oc, normal) / denom;

                        if (t_rect > 0.0f && t_rect < t_hit) {
                            float3 hit_point = ray.origin + t_rect * ray.direction;
                            float3 local_hit = hit_point - source_positions[s];
                            float inv_transform[16];
                            d_invertMatrix(transform, inv_transform);
                            d_transformPoint(inv_transform, local_hit);

                            if (fabsf(local_hit.x) <= source_widths[s].x * 0.5f && fabsf(local_hit.y) <= source_widths[s].y * 0.5f) {
                                float area = source_widths[s].x * source_widths[s].y;
                                float cos_angle = -denom;
                                source_radiance += (flux / area) * cos_angle / M_PI;
                            }
                        }
                    }
                } else if (source_type == 4) {
                    // DISK
                    float transform[16];
                    d_makeTransformMatrix(source_rotations[s], transform);
                    float3 normal = make_float3(transform[2], transform[6], transform[10]);

                    float denom = dot(ray.direction, normal);
                    if (denom < -1e-6f) {
                        float3 oc = source_positions[s] - ray.origin;
                        float t_disk = dot(oc, normal) / denom;

                        if (t_disk > 0.0f && t_disk < t_hit) {
                            float3 hit_point = ray.origin + t_disk * ray.direction;
                            float3 offset = hit_point - source_positions[s];
                            float dist_sq = dot(offset, offset);

                            float radius = source_widths[s].x;
                            if (dist_sq <= radius * radius) {
                                float area = M_PI * radius * radius;
                                float cos_angle = -denom;
                                source_radiance += (flux / area) * cos_angle / M_PI;
                            }
                        }
                    }
                }
            }

            // [flux slot][band]
            size_t ind_hit = (size_t) flux_position * Nbands_launch + b;

            const bool use_top_face = face;

            // Camera flux smoothing: blend the outgoing flux across the facet from the values carried by the vertices it shares with its neighbours, so that a coarsely tessellated curved surface does not
            // render as flat panels with a step at every facet edge.
            if (smooth_v0 >= 0) {
                float outgoing_flux = 0.f;
                for (int c = 0; c < 4; c++) {
                    if (smooth_vertices[c] >= 0) {
                        const size_t index = vertex_field_offset + size_t(smooth_vertices[c]) * Nbands_launch + b;
                        outgoing_flux += smooth_weights[c] * (use_top_face ? vertex_radiation_out_top[index] : vertex_radiation_out_bottom[index]);
                    }
                }
                strength = outgoing_flux * prd.strength;
            } else if (use_top_face) {
                strength = radiation_out_top[ind_hit] * prd.strength;
            } else {
                strength = radiation_out_bottom[ind_hit] * prd.strength;
            }
            // A primitive with a spectral texture is seen in the outgoing radiation of the class at the hit point: flat over the facet, or with
            // camera flux smoothing, interpolated from the facet's vertices
            const bool class_applies = hit_class_slot != 0xFFFFFFFFu && d_spectralPaletteApplies(0u, globalBandIndex(b));
            // With smoothing, the vertices carry the radiation other than the sources' direct radiation, which is evaluated here for this pixel sample. A normal-mapped primitive that is not smoothed
            // carries its facet's direct radiation over to the shading normal.
            if (per_pixel_direct && !class_applies) {
                strength += pixel_body[b - pixel_band_begin] * prd.strength;
            } else if (shade_direct_per_hit && !smooth_shading && direct_recorded && !class_applies) {
                strength += shadedDirectBodyFlux(hit_position, b, globalBandIndex(b), surfaceLayerTable(hit_position, globalBandIndex(b)), viewed_normal, shading_normal, hit_point, normal_mapped, field) * prd.strength;
            }
            if (class_applies) {
                if (smooth_v0 >= 0 && spectral_vertex[0] != 0u) {
                    strength = d_spectralSmoothedClassFlux(hit_position, hit_class_slot, use_top_face, b, globalBandIndex(b), smooth_vertices, smooth_weights, field) * prd.strength;
                    if (per_pixel_direct) {
                        strength += pixel_class[b - pixel_band_begin] * prd.strength;
                    }
                } else {
                    strength = d_spectralClassFlux(hit_class_slot, use_top_face, b) * prd.strength;
                    // When the host left the sources' direct radiation out of the class flux, it is evaluated for this pixel sample, or with a normal map shaded here with the shading normal
                    if (per_pixel_direct) {
                        strength += pixel_class[b - pixel_band_begin] * prd.strength;
                    } else if (normal_mapped && normal_map_direct != 0u && direct_recorded) {
                        strength += d_spectralDirectClassFlux(hit_position, hit_class_slot, use_top_face, b, globalBandIndex(b), camera_ID, surfaceLayerTable(hit_position, globalBandIndex(b)), viewed_normal, shading_normal, hit_point, true, field) *
                                    prd.strength;
                    }
                }
            }
            // A normal map also shades the scattered light (from the sky and other surfaces), by the direction it came from
            if (normal_mapped) {
                const uint b_global = globalBandIndex(b);
                float rho_d, tau_d;
                if (class_applies) {
                    const uint entry = spectral_palette[spectral_palette[3] + hit_position] + (hit_class_slot - spectral_camera_flux[3 + flux_position]);
                    rho_d = d_spectralPaletteValueCam(7, Nsources, entry, b_global, camera_ID);
                    tau_d = d_spectralPaletteValueCam(8, Nsources, entry, b_global, camera_ID);
                } else {
                    CameraMaterialIndexer diffuse_slot_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);
                    const size_t material = diffuse_slot_indexer(Nsources, hit_position, b_global, camera_ID);
                    rho_d = rho_cam[material];
                    tau_d = tau_cam[material];
                }
                const float relief = normalMapDiffuseRelief(hit_position, use_top_face, b, shading_normal, base_viewed, normalize(viewed_normal), rho_d, tau_d, smooth_vertices, smooth_weights, field) * prd.strength;
                strength = strength >= 0 ? fmax(0.0, strength + relief) : strength + relief;
            }
            // A surface with a layer sends out its two components, in proportions that depend on the viewing angle. With smoothing,
            // the second component is interpolated from the facet's vertices like the first, and the viewing angle is taken from the shading normal.
            const int layer = surfaceLayerTable(hit_position, globalBandIndex(b));
            if (layer >= 0) {
                float mu_view;
                float layer_flux;
                if (smooth_shading) {
                    // The second component follows the first, which holds every field's vertices
                    const size_t layer_offset = (size_t) vertex_normal_count * (d_dualField() ? 2u : 1u) * Nbands_launch + vertex_field_offset;
                    layer_flux = 0.f;
                    for (int c = 0; c < 4; c++) {
                        if (smooth_vertices[c] >= 0) {
                            const size_t index = layer_offset + size_t(smooth_vertices[c]) * Nbands_launch + b;
                            layer_flux += smooth_weights[c] * (use_top_face ? vertex_radiation_out_top[index] : vertex_radiation_out_bottom[index]);
                        }
                    }
                } else {
                    const size_t layer_offset = (size_t) Nflux_slots * Nbands_launch;
                    layer_flux = use_top_face ? radiation_out_top[layer_offset + ind_hit] : radiation_out_bottom[layer_offset + ind_hit];
                }
                if (normal_mapped) {
                    // A texel of a map tilted away from the viewer is seen at |m.d| (see the OptiX 8 backend)
                    mu_view = fminf(1.f, fabsf(dot(shading_normal, ray.direction)));
                } else if (use_shading_normal) {
                    // A shading normal can tip away from the viewer at a silhouette, where the layer is seen edge-on
                    mu_view = fminf(1.f, fmaxf(0.f, -dot(shading_normal, ray.direction)));
                } else {
                    mu_view = fabsf(dot(viewed_normal, ray.direction));
                }
                const float a = surfaceLayerAlbedo(layer, mu_view);
                const float abar = surface_layer_tables[layer + surface_layer_table_size];
                strength = strength * (1.f - a) / (1.f - abar) + layer_flux * prd.strength * a / abar;
            }

            // Direct radiation reflected into the camera by a surface layer's microfacet lobe
            double strength_spec = 0;
            if (layer >= 0 && specular_reflection_enabled != 0) {
                if (per_pixel_direct) {
                    strength_spec = pixel_lobe[b - pixel_band_begin] * prd.strength;
                } else if (use_shading_normal) {
                    strength_spec = directLayerReflectionShaded(hit_position, b, layer, viewed_normal, shading_normal, -ray.direction, ray.origin + t_hit * ray.direction, int(camera_ID), normal_mapped, field) * prd.strength;
                } else {
                    strength_spec = directLayerReflection(hit_position, b, layer, viewed_normal, -ray.direction, ray.origin + t_hit * ray.direction, int(camera_ID), field) * prd.strength;
                }
            }

            // absorption
            // For cameras, use pixel index directly (no UUID lookup needed)
            // Use BufferIndexer: [pixel][band]
            size_t ind_camera = rad_indexer(pixel_index, b);

            // Attenuated by any translucent covers the ray passed in front of this surface.
            float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;

            atomicAdd(&radiation_in_camera[ind_camera],
                      cover_tau * ((strength + strength_spec) / M_PI + source_radiance * prd.strength)); // note: pi factor is to convert from flux to intensity assuming surface is Lambertian. We don't multiply by the solid angle by convention to avoid very small numbers. The source radiance is already a radiance, so it is not divided by pi.
        }

        // A translucent cover adds its own outgoing radiance (above) and lets the ray continue to what lies behind it.
        if (coverInAllLaunchedBands(hit_position)) {
            continuePastCover(hit_position);
        }
    }
}

RT_PROGRAM void closest_hit_pixel_label() {

    uint origin_UUID = prd.origin_UUID;

    uint hit_position = primitive_positions[UUID];

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        d_handlePeriodicBoundaryHit(prd, ray.origin, ray.direction, periodic_face_t);
        prd.strength += prd.periodic_hit_t; // the distance travelled to the boundary counts toward the depth

    } else {

        // Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being received)
        // Label the first surface behind any translucent covers: pass through the cover, carrying the distance
        // travelled so far in prd.strength.
        if (coverInAllLaunchedBands(hit_position)) {
            prd.strength += t_hit + COVER_EXIT_OFFSET;
            continuePastCover(hit_position);
            return;
        }

        // Note: We are reserving a value of 0 for the sky, so we will store UUID+1
        camera_pixel_label[origin_UUID] = UUID + 1;

        // The class of the spectral texture at the hit point, found as closest_hit_camera finds it
        if (spectral_palette[0] != 0u && spectral_palette[spectral_palette[3] + hit_position] != 0xFFFFFFFFu) {
            const uint ptype = primitive_type[hit_position];
            const optix::float3 sp = ptype == 1 ? optix::make_float3(surface_v, surface_u + surface_v, 0.f) : optix::make_float3(surface_u - 0.5f, surface_v - 0.5f, 0.f);
            const uint entry = d_spectralPaletteEntry(hit_position, d_originTextureUV(ptype, sp, uvID[hit_position]));
            camera_pixel_class[origin_UUID] = entry == 0xFFFFFFFFu ? 0u : entry - spectral_palette[spectral_palette[3] + hit_position] + 1u;
        }

        float depth = prd.strength + t_hit;
        float3 camera_direction3 = d_rotatePoint(make_float3(1, 0, 0), -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y);
        camera_pixel_depth[origin_UUID] = abs(dot(camera_direction3, ray.direction)) * depth;
    }
}

// ---------------------------------------------------------------------------
// Translucent cover (glass/plastic) any-hit programs
// ---------------------------------------------------------------------------
// A glass cover is not an occluder for direct rays: the ray passes through it (rtIgnoreIntersection)
// attenuated by the Fresnel+Bouguer tau(theta), which accumulates per band in prd.cover_transmittance and is
// applied to the deposited flux in the miss program. Diffuse rays also need the radiance of the surface they
// end on, attenuated only by the covers in front of it, so they pass covers in closest_hit_diffuse instead. What the cover itself absorbs and reflects is booked by the
// cover's own rays (depositOnCoverOrigin), not here: this ray was launched by the surface behind the
// cover, so its strength is normalized to that surface's area and angle. Mirrors coverAnyHitBody() in
// optix8/OptiX8DeviceCode.cu.

RT_PROGRAM void any_hit_direct() {

    if (glass_enabled == 0u) {
        return; // glass model not in use this launch — accept the hit (normal occlusion)
    }

    uint hit_position = primitive_positions[UUID];
    if (hit_position == UINT_MAX) {
        return;
    }
    // The intersection program already skips self-hits (origin_UUID == UUID); guard defensively.
    if (UUID == prd.origin_UUID && prd.tile_x == prd.origin_tile_x && prd.tile_y == prd.origin_tile_y) {
        rtIgnoreIntersection();
        return;
    }

    // A primitive that is not glass in every launched band is a normal (opaque) occluder: accept the hit.
    if (!coverInAllLaunchedBands(hit_position)) {
        return;
    }

    bool face_top;
    float cos_theta = coverNormalFaceCos(hit_position, ray.direction, face_top);

    int b = -1;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {
        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;
        if (b >= HELIOS_MAX_RADIATION_BANDS) {
            break; // host guarantees this never trips
        }

        prd.cover_transmittance[b] *= coverTauRhoAlpha(hit_position, b_global, cos_theta).x; // attenuate the through-beam for this band
    }

    // All launched bands are glass: pass the (attenuated) ray through toward the source.
    rtIgnoreIntersection();
}

//! Add light arriving at the face a ray was launched from to every camera's white reference
/**
 * The white reference is what each camera would record from a spectrally flat, perfectly white surface in place of the primitive: the arriving light weighted by white_reference_cam. Only light
 * straight from a radiation source or the sky is added, so a camera's "auto" white balance neutralizes the light illuminating the scene and not light already coloured by other surfaces. Mirrors
 * accumulateWhiteReference() in optix8/OptiX8DeviceCode.cu.
 */
static __device__ __inline__ void accumulateWhiteReference(size_t ind_origin, int b_global, float strength) {
    if (Ncameras == 0 || strength <= 0) {
        return;
    }
    size_t cam_stride = (size_t) Nprimitives * Nbands_launch;
    for (unsigned int cam = 0; cam < Ncameras; cam++) {
        float white = white_reference_cam[((size_t) prd.source_ID * Nbands_global + b_global) * Ncameras + cam];
        size_t ind_white = cam * cam_stride + ind_origin;
        if (prd.face) {
            atomicFloatAdd(&white_reference_top_cam[ind_white], strength * white);
        } else {
            atomicFloatAdd(&white_reference_bottom_cam[ind_white], strength * white);
        }
    }
}

//! Books the direct radiation a ray delivers to the primitive it was launched from, in fields [0, field_count) but not in field 0 if skip_scene (an excluded primitive blocks the ray there).
//! The fields are arguments so that, inlined with constants for a scene without a background field, the code is that of one field. Must match bookDirectOnOrigin() in OptiX8DeviceCode.cu.
static __device__ __forceinline__ void bookDirectOnOrigin(const uint field_count, const bool skip_scene) {

    // Convert UUID to array position
    uint origin_position = primitive_positions[prd.origin_UUID];
    const size_t slot_stride = (size_t) Nflux_slots * Nbands_launch; // one [flux slot][band] block of the outgoing, scattered and camera buffers

    // Create indexers
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    MaterialPropertyIndexer mat_indexer(Nsources + 1, Nprimitives, Nbands_global); // one slot per source plus the diffuse/scatter slot
    SourceFluxIndexer source_flux_indexer(Nsources, Nbands_launch);
    CameraMaterialIndexer cam_mat_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);

    // A translucent cover books what reaches it with its angular glass properties, not rho/tau.
    bool origin_face_unused;
    const bool origin_is_cover = coverInAllLaunchedBands(origin_position);
    const float origin_cos = origin_is_cover ? coverNormalFaceCos(origin_position, ray.direction, origin_face_unused) : 1.f;

    int b = -1;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {

        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;

        // Use BufferIndexer: [primitive][band]
        size_t ind_origin = rad_indexer(origin_position, b);

        // Use BufferIndexer: [source][primitive][band]
        size_t radprop_ind_global = mat_indexer(prd.source_ID, origin_position, b_global);
        float t_rho = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValue(5, prd.source_ID, prd.palette_entry, b_global) : rho[radprop_ind_global];
        float t_tau = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValue(6, prd.source_ID, prd.palette_entry, b_global) : tau[radprop_ind_global];

        // Use BufferIndexer: [source][band]
        size_t flux_idx = source_flux_indexer(prd.source_ID, b);
        float source_flux = source_fluxes[flux_idx];
        // Attenuation from any translucent covers (glass/plastic) the ray passed through, per band.
        float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;
        double strength = prd.strength * source_flux * cover_tau;

        // A dual ray books what it delivers in the background field too (see d_dualField()). An excluded primitive it passed blocks it
        // in the scene as it is only.
        for (uint field = 0; field < field_count; field++) {
            if (field == 0u && skip_scene) {
                continue;
            }
            // The origin's [flux slot][band] index in the field booked
            const size_t ind_slot = ind_origin + (size_t) field * Nprimitives * Nbands_launch;
            if (origin_is_cover) {
                depositOnCoverOrigin(origin_position, ind_origin, b_global, origin_cos, strength, field);
            } else {
                // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest. The layer reflects a
                // source's direct radiation into its microfacet lobe around the mirror direction, which the rays reading the first
                // scattering of the direct radiation evaluate from the incident flux stored here (see directLayerReflection()).
                const int layer = surfaceLayerTable(origin_position, b_global);
                const float layer_albedo = layer >= 0 ? surfaceLayerAlbedo(layer, prd.origin_cos) : 0.f;
                const double body_strength = layer >= 0 ? strength * (1.0 - layer_albedo) : strength;
                // Also recorded for a primitive with a normal map, whose direct radiation cameras shade per hit point with the shading normal
                if ((layer >= 0 || isSmoothedPrimitive(origin_position) || d_normalMapOf(origin_position) != helios_nm::NO_MAP) && specular_reflection_enabled != 0) {
                    atomicFloatAdd(&radiation_specular[d_directFluxIndex(prd.source_ID, origin_position, field, b)], strength);
                }
                // The reflected energy is also kept in the face's second outgoing component, so that the host counts it in the energy
                // the launch delivering the lobe scatters, or absorbs it in place if that launch never comes; the host clears it
                // before that launch reads the outgoing buffers (cameras use the lobe alone).
                if (layer_albedo > 0.f) {
                    if (prd.face) {
                        atomicFloatAdd(&scatter_buff_top[slot_stride + ind_slot], strength * layer_albedo);
                    } else {
                        atomicFloatAdd(&scatter_buff_bottom[slot_stride + ind_slot], strength * layer_albedo);
                    }
                }

                float absorption = body_strength * (1.f - t_rho - t_tau);

                // absorption: what is absorbed is that of the scene as it is
                if (field == 0u) {
                    atomicAdd(&radiation_in[ind_origin], absorption);
                    if (face_absorption_enabled && prd.face) {
                        atomicAdd(&radiation_in_top[ind_origin], absorption);
                    }
                }

                if (t_rho > 0 || t_tau > 0) {
                    if (prd.face) { // reflection from top, transmission from bottom
                        atomicFloatAdd(&scatter_buff_top[ind_slot], body_strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_bottom[ind_slot], body_strength * t_tau); // transmission
                    } else { // reflection from bottom, transmission from top
                        atomicFloatAdd(&scatter_buff_bottom[ind_slot], body_strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_top[ind_slot], body_strength * t_tau); // transmission
                    }
                }
                // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
                if (spectral_incident_enabled != 0u && spectral_palette[spectral_palette[3] + origin_position] != 0xFFFFFFFFu && body_strength != 0) {
                    const uint face_slot = (prd.face ? 0u : Nsources + 1u) + prd.source_ID;
                    atomicFloatAdd(&spectral_incident[(size_t) face_slot * slot_stride + ind_slot], body_strength);
                }
                if (Ncameras > 0) {
                    // Camera-weighted scatter, accumulated once per camera into that camera's [flux slot][band]
                    // block. This launch runs once per dispatch, not once per camera, so camera_ID is not
                    // meaningful here; weighting by a single camera handed every other camera that camera's image.
                    size_t cam_scatter_stride = slot_stride;
                    for (unsigned int cam = 0; cam < Ncameras; cam++) {
                        // Use BufferIndexer: [source][primitive][band][camera]
                        size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, cam);
                        float t_rho_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(7, prd.source_ID, prd.palette_entry, b_global, cam) : rho_cam[indc];
                        float t_tau_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(8, prd.source_ID, prd.palette_entry, b_global, cam) : tau_cam[indc];
                        size_t ind_cam = cam * cam_scatter_stride + ind_slot;
                        if ((t_rho_cam > 0 || t_tau_cam > 0) && body_strength > 0) {
                            if (prd.face) { // reflection from top, transmission from bottom
                                atomicFloatAdd(&scatter_buff_top_cam[ind_cam], body_strength * t_rho_cam); // reflection
                                atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], body_strength * t_tau_cam); // transmission
                            } else { // reflection from bottom, transmission from top
                                atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], body_strength * t_rho_cam); // reflection
                                atomicFloatAdd(&scatter_buff_top_cam[ind_cam], body_strength * t_tau_cam); // transmission
                            }
                        }
                    }
                }
            }
        }
        if (Ncameras > 0 && !skip_scene) {
            accumulateWhiteReference(ind_origin, b_global, strength);
        }
    }
}

RT_PROGRAM void miss_direct() {
    bookDirectOnOrigin(1u, false);
}

//! miss_direct with a background field: a dual ray books what it delivers in the background field too. An excluded primitive it passed, in the central copy of the domain in
//! an earlier segment (see closest_hit_direct_dual) or in this last one, blocks it in the scene as it is only.
RT_PROGRAM void miss_direct_dual() {
    const bool blocked_in_scene = prd.dual_ray && (prd.excluded_resolved || prd.recorded_t < NO_RECORDED_HIT);
    bookDirectOnOrigin(prd.dual_ray ? 2u : 1u, blocked_in_scene);
}

// Unified device function to evaluate diffuse angular distribution
// Supports three modes with automatic priority-based selection:
//   Priority 1: Power-law (Harrison & Coombes) if K > 0
//   Priority 2: Prague sky model if params.w > 0 (valid normalization)
//   Priority 3: Isotropic (uniform) otherwise
__device__ float evaluateDiffuseAngularDistribution(const float3 &ray_dir, const float3 &peak_dir, float power_law_K, float power_law_norm, const float4 &prague_params) {

    // Priority 1: Power-law (if K > 0)
    if (power_law_K > 0.0f) {
        float psi = acos_safe(dot(peak_dir, ray_dir));
        psi = fmaxf(psi, M_PI / 180.0f); // Avoid singularity at 1 degree
        return powf(psi, -power_law_K) * power_law_norm;
    }

    // Priority 2: Prague (if params.w > 0, indicating valid normalization)
    if (prague_params.w > 0.0f) {
        // Angular distance from sun (degrees)
        float gamma = acos_safe(dot(ray_dir, peak_dir)) * 180.0f / float(M_PI);

        // Zenith angle
        float cos_theta = fmaxf(ray_dir.z, 0.0f);

        // Circumsolar + horizon brightening
        // params: (circ_strength, circ_width, horizon_brightness, normalization)
        // A non-positive circumsolar width means no circumsolar region (a width of zero would give 0/0 = NaN toward the sun)
        float circumsolar_term = (prague_params.y > 0.f) ? (1.0f + prague_params.x * expf(-gamma / prague_params.y)) : 1.0f;
        float pattern = circumsolar_term * (1.0f + (prague_params.z - 1.0f) * (1.0f - cos_theta));

        // Multiply by π to account for cosine-weighted sampling PDF (cos×sin/π)
        // This ensures correct Monte Carlo integration for Prague angular distribution
        return pattern * prague_params.w * M_PI;
    }

    // Priority 3: Isotropic
    // For isotropic diffuse with cosine-weighted sampling (PDF = cos×sin/π):
    // The π from the PDF denominator must appear in the Monte Carlo weight
    return 1.0f;
}

//! Books the sky radiation a diffuse ray that leaves the scene brings to the primitive at origin_position it was launched from, in fields [field_first, field_count).
//! The fields are arguments so that, inlined with constants for a scene without a background field, the code is that of one field. Must match bookSkyOnOrigin() in OptiX8DeviceCode.cu.
static __device__ __forceinline__ void bookSkyOnOrigin(uint origin_position, const uint field_first, const uint field_count) {

    const size_t slot_stride = (size_t) Nflux_slots * Nbands_launch; // one [flux slot][band] block of the outgoing, scattered and camera buffers

    // Create indexers
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    MaterialPropertyIndexer mat_indexer(Nsources + 1, Nprimitives, Nbands_global); // one slot per source plus the diffuse/scatter slot
    CameraMaterialIndexer cam_mat_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);

    // A translucent cover books what reaches it with its angular glass properties, not rho/tau.
    bool origin_face_unused;
    const bool origin_is_cover = coverInAllLaunchedBands(origin_position);
    const float origin_cos = origin_is_cover ? coverNormalFaceCos(origin_position, ray.direction, origin_face_unused) : 1.f;

    // The diffuse sky fills only the upper hemisphere unless enabled below the horizon.
    const bool sees_sky = ray.direction.z >= 0.f || below_horizon_diffuse_enabled != 0;

    int b = -1;
    for (size_t b_global = 0; b_global < Nbands_global; b_global++) {

        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;

        // The ray brings the sky's flux less that of the band's reference blackbody field, which follows the sky fluxes in
        // diffuse_flux: emission is gathered as its departure from that field (see RadiationModel::runBand()).
        const float sky_flux = sees_sky ? diffuse_flux[b] : 0.f;
        const float reference_flux = diffuse_flux[Nbands_launch + b];

        if (sky_flux > 0.f || reference_flux != 0.f) {

            // Use BufferIndexer: [primitive][band]
            size_t ind_origin = rad_indexer(origin_position, b);

            // Use BufferIndexer: [source][primitive][band]
            size_t radprop_ind_global = mat_indexer(prd.source_ID, origin_position, b_global);
            float t_rho = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValue(5, prd.source_ID, prd.palette_entry, b_global) : rho[radprop_ind_global];
            float t_tau = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValue(6, prd.source_ID, prd.palette_entry, b_global) : tau[radprop_ind_global];

            // Attenuation from any translucent covers (glass/plastic) the sky ray passed through, per band.
            float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;

            // Use unified distribution function (supports power-law, Prague, and isotropic modes)
            float fd = sky_flux > 0.f ? evaluateDiffuseAngularDistribution(ray.direction, diffuse_peak_dir[b], diffuse_extinction[b], diffuse_dist_norm[b], sky_radiance_params[b]) : 0.f;

            float sky_strength = fd * sky_flux * prd.strength * cover_tau;
            float strength = sky_strength - reference_flux * prd.strength * cover_tau;

            // A dual ray books the sky in the background field too (see d_dualField()); the scene as it is has already received an
            // excluded primitive's radiation instead if the ray passed one (see resolveRecordedHit()).
            for (uint field = field_first; field < field_count; field++) {
                // The origin's [flux slot][band] index in the field booked
                const size_t ind_slot = ind_origin + (size_t) field * Nprimitives * Nbands_launch;
                if (origin_is_cover) {
                    depositOnCoverOrigin(origin_position, ind_origin, b_global, origin_cos, strength, field);
                } else {
                    // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest
                    const int layer = surfaceLayerTable(origin_position, b_global);
                    const float layer_albedo = layer >= 0 ? surfaceLayerAlbedo(layer, prd.origin_cos) : 0.f;
                    const float body_strength = layer >= 0 ? strength * (1.f - layer_albedo) : strength;
                    if (layer_albedo > 0.f) {
                        depositLayerReflection(ind_slot, b_global, strength * layer_albedo);
                    }
                    recordIncidentDirection(origin_position, prd.face, b, body_strength, ray.direction, field);

                    //  absorption: what is absorbed is that of the scene as it is
                    if (field == 0u) {
                        atomicAdd(&radiation_in[ind_origin], body_strength * (1.f - t_rho - t_tau));
                        if (face_absorption_enabled && prd.face) {
                            atomicAdd(&radiation_in_top[ind_origin], body_strength * (1.f - t_rho - t_tau));
                        }
                    }

                    if (t_rho > 0 || t_tau > 0) {
                        if (prd.face) { // reflection from top, transmission from bottom
                            atomicFloatAdd(&scatter_buff_top[ind_slot], body_strength * t_rho); // reflection
                            atomicFloatAdd(&scatter_buff_bottom[ind_slot], body_strength * t_tau); // transmission
                        } else { // reflection from bottom, transmission from top
                            atomicFloatAdd(&scatter_buff_bottom[ind_slot], body_strength * t_rho); // reflection
                            atomicFloatAdd(&scatter_buff_top[ind_slot], body_strength * t_tau); // transmission
                        }
                    }
                    // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
                    if (spectral_incident_enabled != 0u && spectral_palette[spectral_palette[3] + origin_position] != 0xFFFFFFFFu && body_strength != 0) {
                        const uint face_slot = (prd.face ? 0u : Nsources + 1u) + prd.source_ID;
                        atomicFloatAdd(&spectral_incident[(size_t) face_slot * slot_stride + ind_slot], body_strength);
                    }
                    if (Ncameras > 0) {
                        // Camera-weighted scatter, accumulated once per camera into that camera's [flux slot][band]
                        // block. This launch runs once per dispatch, not once per camera, so camera_ID is not
                        // meaningful here; weighting by a single camera handed every other camera that camera's image.
                        size_t cam_scatter_stride = slot_stride;
                        for (unsigned int cam = 0; cam < Ncameras; cam++) {
                            // Use BufferIndexer: [source][primitive][band][camera]
                            size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, cam);
                            float t_rho_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(7, prd.source_ID, prd.palette_entry, b_global, cam) : rho_cam[indc];
                            float t_tau_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(8, prd.source_ID, prd.palette_entry, b_global, cam) : tau_cam[indc];
                            size_t ind_cam = cam * cam_scatter_stride + ind_slot;
                            if ((t_rho_cam > 0 || t_tau_cam > 0) && prd.strength > 0) {
                                if (prd.face) { // reflection from top, transmission from bottom
                                    atomicFloatAdd(&scatter_buff_top_cam[ind_cam], body_strength * t_rho_cam); // reflection
                                    atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], body_strength * t_tau_cam); // transmission
                                } else { // reflection from bottom, transmission from top
                                    atomicFloatAdd(&scatter_buff_bottom_cam[ind_cam], body_strength * t_rho_cam); // reflection
                                    atomicFloatAdd(&scatter_buff_top_cam[ind_cam], body_strength * t_tau_cam); // transmission
                                }
                            }
                        }
                    }
                }
            }
            if (Ncameras > 0 && field_first == 0u) {
                accumulateWhiteReference(ind_origin, b_global, sky_strength);
                // Note: Don't accumulate diffuse sky radiation to radiation_specular
                // Specular should only reflect DIRECT source radiation (accumulated in miss_direct)
            }
        }
    }
}

RT_PROGRAM void miss_diffuse() {
    bookSkyOnOrigin(primitive_positions[prd.origin_UUID], 0u, 1u);
}

//! miss_diffuse with a background field
RT_PROGRAM void miss_diffuse_dual() {
    const uint origin_position = primitive_positions[prd.origin_UUID];
    // An excluded primitive the ray passed in this segment is what the scene as it is receives from, in place of the sky
    resolveRecordedHit(origin_position, NO_RECORDED_HIT);
    bookSkyOnOrigin(origin_position, prd.excluded_resolved ? 1u : 0u, prd.dual_ray ? 2u : 1u);
}

RT_PROGRAM void miss_camera() {

    // For cameras, origin_UUID is actually the pixel index (not a primitive UUID!)
    uint pixel_index = prd.origin_UUID;
    size_t Npixels = camera_resolution_full.x * camera_resolution_full.y;

    // Create indexer
    RadiationBufferIndexer rad_indexer(Npixels, Nbands_launch); // Use Npixels for camera radiation buffer

    for (size_t b = 0; b < Nbands_launch; b++) {

        float radiance = 0.0f;

        // Check all light sources
        for (uint s = 0; s < Nsources; s++) {
            float flux = source_fluxes[s * Nbands_launch + b];
            if (flux <= 0.0f)
                continue;

            uint source_type = source_types[s];

            if (source_type == 0 || source_type == 2) {
                // COLLIMATED or SUN_SPHERE
                float cos_sun_angle = dot(ray.direction, sun_direction);
                if (cos_sun_angle >= solar_disk_cos_angle && solar_disk_radiance[b] > 0.0f) {
                    radiance += solar_disk_radiance[b];
                }
            } else if (source_type == 1) {
                // SPHERE
                float radius = source_widths[s].x * 0.5f;
                if (d_raySphereIntersect(ray.origin, ray.direction, source_positions[s], radius)) {
                    float area = 4.0f * M_PI * radius * radius;
                    radiance += (flux / area) / M_PI;
                }
            } else if (source_type == 3) {
                // RECTANGLE
                float cos_angle;
                if (d_rayRectangleIntersect(ray.origin, ray.direction, source_positions[s], source_widths[s].x, source_widths[s].y, source_rotations[s], cos_angle)) {
                    float area = source_widths[s].x * source_widths[s].y;
                    radiance += (flux / area) * cos_angle / M_PI;
                }
            } else if (source_type == 4) {
                // DISK
                float cos_angle;
                float radius = source_widths[s].x;
                if (d_rayDiskIntersect(ray.origin, ray.direction, source_positions[s], radius, source_rotations[s], cos_angle)) {
                    float area = M_PI * radius * radius;
                    radiance += (flux / area) * cos_angle / M_PI;
                }
            }
        }

        // The diffuse transfer limits the sky to the upper hemisphere, but the camera continues the sky below the horizon so that a finite
        // domain does not render a black strip between its edge and the horizon: a ray leaving below the horizon sees the sky radiance at the
        // horizon in its own azimuth. With the diffuse sky enabled below the horizon the camera sees the full-sphere sky pattern instead.
        float3 sky_direction = ray.direction;
        if (ray.direction.z < 0.f && below_horizon_diffuse_enabled == 0) {
            const float horizontal_length = sqrtf(ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y);
            // A ray pointing straight down has no azimuth; any horizontal direction is equally valid for this single direction.
            sky_direction = horizontal_length > 1e-6f ? make_float3(ray.direction.x / horizontal_length, ray.direction.y / horizontal_length, 0.f) : make_float3(1.f, 0.f, 0.f);
        }

        // Fallback to sky radiance if no sources visible
        if (radiance <= 0.0f && camera_sky_radiance[b] > 0.f) {
            // Evaluate directional sky radiance using unified distribution function. Prague parameters (normalization w > 0) give the
            // sky's own pattern. Without them the camera sees the diffuse sky that lights the scene, so it takes the band's angular
            // distribution from the diffuse launches: its power law, or isotropic.
            const float4 sky_p = sky_radiance_params[b];
            float power_K = 0.f;
            float power_N = 1.f;
            float3 peak_d = sun_direction;
            if (sky_p.w <= 0.f) {
                power_K = diffuse_extinction[b];
                power_N = diffuse_dist_norm[b];
                peak_d = diffuse_peak_dir[b];
            }
            float angular_weight = evaluateDiffuseAngularDistribution(sky_direction, peak_d, power_K, power_N, sky_p);

            radiance = camera_sky_radiance[b] * angular_weight;
        }

        // Isotropic sky emission/longwave: when band has emission enabled, the user-set diffuse_flux
        // represents hemispherical sky thermal/longwave irradiance. Convert to isotropic radiance.
        if (band_emission_flag[b] != 0u && camera_diffuse_flux[b] > 0.f) {
            radiance += camera_diffuse_flux[b] / M_PI;
        }

        if (radiance > 0.0f) {
            // Accumulate radiance directly (same as surface hits accumulate radiation_out)
            // Units: W/m²/sr
            // Monte Carlo averaging: prd.strength = 1/N_rays
            // Use BufferIndexer: [pixel][band]
            size_t ind_camera = rad_indexer(pixel_index, b);
            // Seen through any translucent covers the ray passed, attenuated by their transmittance.
            float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;
            atomicAdd(&radiation_in_camera[ind_camera], radiance * prd.strength * cover_tau);
        }
    }
}

//! A camera's shadow ray toward a source missed everything: the source reaches the point it was traced from (see d_pixelDirectRadiation())
RT_PROGRAM void miss_camera_shadow() {
    prd.source_reached = true;
}

RT_PROGRAM void miss_pixel_label() {

    camera_pixel_depth[prd.origin_UUID] = -1;
}
