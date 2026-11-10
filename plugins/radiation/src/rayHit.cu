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
//! weights it by that camera's white reference. Mirrors depositOnCoverOrigin() in optix8/OptiX8DeviceCode.cu.
static __device__ __inline__ void depositOnCoverOrigin(uint origin_position, size_t ind_origin, int b_global, float cos_theta, float strength) {
    if (strength <= 0) {
        return;
    }
    float3 tra = coverTauRhoAlpha(origin_position, b_global, cos_theta);

    atomicAdd(&radiation_in[ind_origin], strength * tra.z);
    if (face_absorption_enabled && prd.face) {
        atomicAdd(&radiation_in_top[ind_origin], strength * tra.z);
    }

    if (prd.face) {
        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * tra.y);
    } else {
        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * tra.y);
    }

    if (Ncameras > 0) {
        size_t cam_scatter_stride = (size_t) Nprimitives * Nbands_launch;
        for (unsigned int cam = 0; cam < Ncameras; cam++) {
            float white = white_reference_cam[((size_t) prd.source_ID * Nbands_global + b_global) * Ncameras + cam];
            size_t ind_cam = cam * cam_scatter_stride + ind_origin;
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

//! Flux the top or bottom face of a primitive sends out at the angle whose cosine is mu_out, from its outgoing components
static __device__ __inline__ float outgoingFlux(bool top, size_t ind, int table, float mu_out) {
    const float first = top ? radiation_out_top[ind] : radiation_out_bottom[ind];
    if (table < 0) {
        return first;
    }
    const size_t layer_offset = (size_t) Nprimitives * Nbands_launch;
    const float second = top ? radiation_out_top[layer_offset + ind] : radiation_out_bottom[layer_offset + ind];
    const float a = surfaceLayerAlbedo(table, mu_out);
    const float abar = surface_layer_tables[table + surface_layer_table_size];
    return first * (1.f - a) / (1.f - abar) + second * a / abar;
}

//! Adds what the surface layer of the ray's origin reflects, of its arrival on face prd.face, to that face's second outgoing component, and to every camera's (weighted by its white reference, since the layer reflects all wavelengths alike)
static __device__ __inline__ void depositLayerReflection(size_t ind_origin, int b_global, float reflected) {
    const size_t layer_offset = (size_t) Nprimitives * Nbands_launch;
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
//! 0 weights each source by that camera's response.
static __device__ __inline__ float directLayerReflection(uint pos, uint b_launch, int table, const float3 &face_normal, const float3 &to_viewer, const float3 &point, int camera) {
    float reflected = 0.f;
    for (uint s = 0; s < Nsources; s++) {
        float incident = radiation_specular[((size_t) s * Nprimitives + pos) * Nbands_launch + b_launch];
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

//! With camera flux smoothing: flux, in the units of the outgoing buffers, that the surface layer of the primitive at position pos reflects toward to_viewer from the direct radiation of every source,
//! evaluated with the interpolated shading normal rather than the facet's. viewed_normal is the facet's own unit normal on the side the camera sees, and shading_normal the interpolated one on that side.
//! The incident flux carries the cosine of the facet, so it is rescaled to the cosine of the shading normal; it reached the face toward the source, so only a source on the viewed side counts.
static __device__ __inline__ float directLayerReflectionSmoothed(uint pos, uint b_launch, int table, const float3 &viewed_normal, const float3 &shading_normal, const float3 &to_viewer, const float3 &point, int camera, const int v[4],
                                                                 const float w[4]) {
    float reflected = 0.f;
    for (uint s = 0; s < Nsources; s++) {
        const float3 to_source = d_directionToSource(s, point);
        // What reached the face toward the source; a viewed face turned away from it takes the irradiance its vertices carry (see d_faceDirect())
        const float facet_value = dot(viewed_normal, to_source) > 0.f ? radiation_specular[((size_t) s * Nprimitives + pos) * Nbands_launch + b_launch] : 0.f;
        float incident = d_faceDirect(facet_value, to_source, viewed_normal, shading_normal, s, b_launch, v, w);
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

//! With camera flux smoothing: the outgoing radiation, first component, that the face of the primitive at position pos seen along the camera ray sends out of the direct radiation of every source,
//! evaluated with the shading normal (see RadiationModel::subtractCameraDirectFlux(), which removes the facet's own value from what the vertices carry). viewed_normal and shading_normal are on the viewed
//! side. The radiation reaching the face toward a source is reflected by the viewed face if it is that face and transmitted to it otherwise; a surface layer, whose albedo table starts at `layer` (-1 for
//! none), keeps 1 - a(mu) of it in the body.
static __device__ __inline__ float smoothedDirectBodyFlux(uint pos, uint b_launch, uint b_global, int layer, const float3 &viewed_normal, const float3 &shading_normal, const float3 &point, const int v[4], const float w[4]) {
    CameraMaterialIndexer cam_mat_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);
    float flux = 0.f;
    for (uint s = 0; s < Nsources; s++) {
        const float incident = radiation_specular[((size_t) s * Nprimitives + pos) * Nbands_launch + b_launch];
        const float3 to_source = d_directionToSource(s, point);
        const bool viewed_face_toward_source = dot(viewed_normal, to_source) > 0.f;
        const size_t material = cam_mat_indexer(s, pos, b_global, camera_ID);
        const float rho = rho_cam[material];
        const float tau = tau_cam[material];
        if (rho > 0.f) {
            const float reaching = d_faceDirect(viewed_face_toward_source ? incident : 0.f, to_source, viewed_normal, shading_normal, s, b_launch, v, w);
            flux += rho * reaching * (layer >= 0 ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, dot(shading_normal, to_source))) : 1.f);
        }
        if (tau > 0.f) {
            const float reaching = d_faceDirect(viewed_face_toward_source ? 0.f : incident, to_source, -viewed_normal, -shading_normal, s, b_launch, v, w);
            flux += tau * reaching * (layer >= 0 ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, -dot(shading_normal, to_source))) : 1.f);
        }
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

RT_PROGRAM void closest_hit_direct() {

    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if position is invalid
    if (hit_position == UINT_MAX) {
        return;
    }

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        prd.hit_periodic_boundary = true;

        prd.periodic_hit_t = periodic_face_t;

        float3 ray_origin = ray.origin + periodic_face_t * ray.direction;

        // Tolerance for identifying the face that was hit. The ray continues from exactly one period away, the same point on the
        // opposite face: moving it any further would shift it off its straight path by that much at every wrap.
        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += width_x;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x -= width_x;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += width_y;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y -= width_y;
        }
    }
};

RT_PROGRAM void closest_hit_diffuse() {

    // Convert UUIDs to array positions for buffer indexing
    uint origin_UUID = prd.origin_UUID;
    uint origin_position = primitive_positions[origin_UUID];
    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if positions are invalid
    if (origin_position == UINT_MAX || hit_position == UINT_MAX) {
        return;
    }

    // Create indexers for buffer access
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    MaterialPropertyIndexer mat_indexer(Nsources + 1, Nprimitives, Nbands_global); // one slot per source plus the diffuse/scatter slot
    CameraMaterialIndexer cam_mat_indexer(Nsources + 1, Nprimitives, Nbands_global, Ncameras);

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + periodic_face_t * ray.direction;

        // Tolerance for identifying the face that was hit. The ray continues from exactly one period away, the same point on the
        // opposite face: moving it any further would shift it off its straight path by that much at every wrap.
        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += width_x;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x -= width_x;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += width_y;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y -= width_y;
        }

    } else {

        // Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being recieved)

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
            size_t ind_hit = rad_indexer(hit_position, b);

            // A hit primitive with a surface layer sends out its two components, in proportions that depend on the angle
            const int hit_layer = surfaceLayerTable(hit_position, b_global);
            float outgoing = outgoingFlux(face, ind_hit, hit_layer, hit_layer >= 0 ? fabsf(dot(normal, ray.direction)) : 1.f);
            if (hit_layer >= 0 && specular_reflection_enabled != 0) {
                // In the launch that reads the first scattering of the direct radiation, add the layer's reflection lobe of it
                outgoing += directLayerReflection(hit_position, b, hit_layer, face ? normal : -normal, -ray.direction, ray.origin + t_hit * ray.direction, -1);
            }
            // Forward scattering estimate (see RadiationModel::runBand()): the flux this ray carries away from the origin along its
            // path, times the fraction of it the surface it reaches scatters again. A cover reflects its reflectance; what it transmits
            // is counted where the ray continues, attenuated by the cover transmittance. A hit surface's own reflectivity is used, not
            // that of a spectral-texture class at the hit point: the estimate need only be unbiased over the scene, not per texel.
            if (forward_scatter_enabled) {
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
                            const float hit_albedo = surfaceLayerAlbedo(hit_layer, fabsf(dot(normal, ray.direction)));
                            rescattered = hit_albedo + (1.f - hit_albedo) * rescattered;
                        }
                    }
                    const float path_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;
                    atomicAdd(&forward_scatter[ind_origin], fabsf(launched) * prd.strength * path_tau * rescattered);
                }
            }

            double strength = outgoing * prd.strength;

            // Attenuated by any translucent covers the ray passed on its way to the surface.
            strength *= (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;

            if (strength == 0) {
                continue;
            }

            if (origin_is_cover) {
                depositOnCoverOrigin(origin_position, ind_origin, b_global, origin_cos, strength);
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
                depositLayerReflection(ind_origin, b_global, strength * layer_albedo);
            }

            // absorption - calculate with defensive check for energy conservation violations
            float absorption_factor = 1.f - t_rho - t_tau;
            float contribution = body_strength * absorption_factor;


#ifndef NDEBUG
            if (absorption_factor < -1e-5f) {
                printf("ERROR: Negative absorption! rho=%.6f, tau=%.6f, origin_UUID=%u\n", t_rho, t_tau, origin_UUID);
                absorption_factor = 0.f;
            }
#endif
            atomicAdd(&radiation_in[ind_origin], contribution);
            if (face_absorption_enabled && prd.face) {
                atomicAdd(&radiation_in_top[ind_origin], contribution);
            }

            // The gathered flux is negative where a surface radiates less than the reference field (see RadiationModel::runBand())
            if (t_rho > 0 || t_tau > 0) {
                if (prd.face) { // reflection from top, transmission from bottom
                    atomicFloatAdd(&scatter_buff_top[ind_origin], body_strength * t_rho); // reflection
                    atomicFloatAdd(&scatter_buff_bottom[ind_origin], body_strength * t_tau); // transmission
                } else { // reflection from bottom, transmission from top
                    atomicFloatAdd(&scatter_buff_bottom[ind_origin], body_strength * t_rho); // reflection
                    atomicFloatAdd(&scatter_buff_top[ind_origin], body_strength * t_tau); // transmission
                }
            }
            // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
            if (spectral_incident_enabled != 0u && spectral_palette[spectral_palette[3] + origin_position] != 0xFFFFFFFFu && body_strength != 0) {
                const uint face_slot = (prd.face ? 0u : Nsources + 1u) + prd.source_ID;
                atomicFloatAdd(&spectral_incident[(size_t) face_slot * Nprimitives * Nbands_launch + ind_origin], body_strength);
            }
            if (Ncameras > 0) {
                // Camera-weighted scatter, accumulated once per camera into that camera's [primitive][band]
                // block. This launch runs once per dispatch, not once per camera, so camera_ID is not
                // meaningful here; weighting by a single camera handed every other camera that camera's image.
                size_t cam_scatter_stride = (size_t) Nprimitives * Nbands_launch;
                for (unsigned int cam = 0; cam < Ncameras; cam++) {
                    // Use BufferIndexer for camera material: [source][primitive][band][camera]
                    size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, cam);
                    float t_rho_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(7, prd.source_ID, prd.palette_entry, b_global, cam) : rho_cam[indc];
                    float t_tau_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(8, prd.source_ID, prd.palette_entry, b_global, cam) : tau_cam[indc];
                    size_t ind_cam = cam * cam_scatter_stride + ind_origin;
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

        // A translucent cover: its own outgoing radiance (reflection and emission) was received above like that of any
        // surface; continue past it to what lies behind, attenuated by its transmittance.
        if (coverInAllLaunchedBands(hit_position)) {
            continuePastCover(hit_position);
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

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + periodic_face_t * ray.direction;

        // Tolerance for identifying the face that was hit. The ray continues from exactly one period away, the same point on the
        // opposite face: moving it any further would shift it off its straight path by that much at every wrap.
        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += width_x;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x -= width_x;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += width_y;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y -= width_y;
        }

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

        double strength;
        for (size_t b = 0; b < Nbands_launch; b++) {

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

            // Use BufferIndexer: [primitive][band]
            size_t ind_hit = rad_indexer(hit_position, b);

            const bool use_top_face = face;

            // Camera flux smoothing: blend the outgoing flux across the facet from the values carried by the vertices it shares with its neighbours, so that a coarsely tessellated curved surface does not
            // render as flat panels with a step at every facet edge.
            if (smooth_v0 >= 0) {
                float outgoing_flux = 0.f;
                for (int c = 0; c < 4; c++) {
                    if (smooth_vertices[c] >= 0) {
                        const size_t index = size_t(smooth_vertices[c]) * Nbands_launch + b;
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
            const uint hit_class_slot = d_spectralCameraClassSlot(hit_position, primitive_type[hit_position], surface_u, surface_v, uvID[hit_position]);
            const bool class_applies = hit_class_slot != 0xFFFFFFFFu && d_spectralPaletteApplies(0u, globalBandIndex(b));
            // With smoothing, the vertices carry the radiation other than the sources' direct radiation, which is evaluated here per facet with the shading normal
            if (smooth_shading && Nsources > 0 && radiation_specular.size() >= size_t(Nsources) * Nprimitives * Nbands_launch && !class_applies) {
                strength += smoothedDirectBodyFlux(hit_position, b, globalBandIndex(b), surfaceLayerTable(hit_position, globalBandIndex(b)), viewed_normal, shading_normal, hit_point, smooth_vertices, smooth_weights) * prd.strength;
            }
            if (class_applies) {
                if (smooth_v0 >= 0 && spectral_vertex[0] != 0u) {
                    strength = d_spectralSmoothedClassFlux(hit_position, hit_class_slot, use_top_face, b, globalBandIndex(b), smooth_vertices, smooth_weights, viewed_normal, shading_normal, hit_point) * prd.strength;
                } else {
                    strength = d_spectralClassFlux(hit_class_slot, use_top_face, b) * prd.strength;
                }
            }
            // A surface with a layer sends out its two components, in proportions that depend on the viewing angle. With smoothing,
            // the second component is interpolated from the facet's vertices like the first, and the viewing angle is taken from the shading normal.
            const int layer = surfaceLayerTable(hit_position, globalBandIndex(b));
            if (layer >= 0) {
                float mu_view;
                float layer_flux;
                if (smooth_shading) {
                    const size_t layer_offset = (size_t) vertex_normal_count * Nbands_launch;
                    layer_flux = 0.f;
                    for (int c = 0; c < 4; c++) {
                        if (smooth_vertices[c] >= 0) {
                            const size_t index = layer_offset + size_t(smooth_vertices[c]) * Nbands_launch + b;
                            layer_flux += smooth_weights[c] * (use_top_face ? vertex_radiation_out_top[index] : vertex_radiation_out_bottom[index]);
                        }
                    }
                    // A shading normal can tip away from the viewer at a silhouette, where the layer is seen edge-on
                    mu_view = fminf(1.f, fmaxf(0.f, -dot(shading_normal, ray.direction)));
                } else {
                    const size_t layer_offset = (size_t) Nprimitives * Nbands_launch;
                    layer_flux = use_top_face ? radiation_out_top[layer_offset + ind_hit] : radiation_out_bottom[layer_offset + ind_hit];
                    mu_view = fabsf(dot(viewed_normal, ray.direction));
                }
                const float a = surfaceLayerAlbedo(layer, mu_view);
                const float abar = surface_layer_tables[layer + surface_layer_table_size];
                strength = strength * (1.f - a) / (1.f - abar) + layer_flux * prd.strength * a / abar;
            }

            // Direct radiation reflected into the camera by a surface layer's microfacet lobe
            double strength_spec = 0;
            if (layer >= 0 && specular_reflection_enabled != 0) {
                if (smooth_shading) {
                    strength_spec = directLayerReflectionSmoothed(hit_position, b, layer, viewed_normal, shading_normal, -ray.direction, ray.origin + t_hit * ray.direction, int(camera_ID), smooth_vertices, smooth_weights) * prd.strength;
                } else {
                    strength_spec = directLayerReflection(hit_position, b, layer, viewed_normal, -ray.direction, ray.origin + t_hit * ray.direction, int(camera_ID)) * prd.strength;
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

        prd.hit_periodic_boundary = true;
        prd.strength += periodic_face_t; // the distance travelled to the boundary counts toward the depth

        float3 ray_origin = ray.origin + periodic_face_t * ray.direction;

        // Tolerance for identifying the face that was hit. The ray continues from exactly one period away, the same point on the
        // opposite face: moving it any further would shift it off its straight path by that much at every wrap.
        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += width_x;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x -= width_x;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += width_y;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y -= width_y;
        }

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
    if (UUID == prd.origin_UUID) {
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

RT_PROGRAM void miss_direct() {

    // Convert UUID to array position
    uint origin_position = primitive_positions[prd.origin_UUID];

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

        if (origin_is_cover) {
            depositOnCoverOrigin(origin_position, ind_origin, b_global, origin_cos, strength);
        } else {
            // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest. The layer reflects a
            // source's direct radiation into its microfacet lobe around the mirror direction, which the rays reading the first
            // scattering of the direct radiation evaluate from the incident flux stored here (see directLayerReflection()).
            const int layer = surfaceLayerTable(origin_position, b_global);
            const float layer_albedo = layer >= 0 ? surfaceLayerAlbedo(layer, prd.origin_cos) : 0.f;
            const double body_strength = layer >= 0 ? strength * (1.0 - layer_albedo) : strength;
            if ((layer >= 0 || isSmoothedPrimitive(origin_position)) && specular_reflection_enabled != 0) {
                atomicFloatAdd(&radiation_specular[((size_t) prd.source_ID * Nprimitives + origin_position) * Nbands_launch + b], strength);
            }
            // The reflected energy is also kept in the face's second outgoing component, so that the host counts it in the energy
            // the launch delivering the lobe scatters, or absorbs it in place if that launch never comes; the host clears it
            // before that launch reads the outgoing buffers (cameras use the lobe alone).
            if (layer_albedo > 0.f) {
                const size_t layer_offset = (size_t) Nprimitives * Nbands_launch;
                if (prd.face) {
                    atomicFloatAdd(&scatter_buff_top[layer_offset + ind_origin], strength * layer_albedo);
                } else {
                    atomicFloatAdd(&scatter_buff_bottom[layer_offset + ind_origin], strength * layer_albedo);
                }
            }

            float absorption = body_strength * (1.f - t_rho - t_tau);

            // absorption
            atomicAdd(&radiation_in[ind_origin], absorption);
            if (face_absorption_enabled && prd.face) {
                atomicAdd(&radiation_in_top[ind_origin], absorption);
            }

            if (t_rho > 0 || t_tau > 0) {
                if (prd.face) { // reflection from top, transmission from bottom
                    atomicFloatAdd(&scatter_buff_top[ind_origin], body_strength * t_rho); // reflection
                    atomicFloatAdd(&scatter_buff_bottom[ind_origin], body_strength * t_tau); // transmission
                } else { // reflection from bottom, transmission from top
                    atomicFloatAdd(&scatter_buff_bottom[ind_origin], body_strength * t_rho); // reflection
                    atomicFloatAdd(&scatter_buff_top[ind_origin], body_strength * t_tau); // transmission
                }
            }
            // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
            if (spectral_incident_enabled != 0u && spectral_palette[spectral_palette[3] + origin_position] != 0xFFFFFFFFu && body_strength != 0) {
                const uint face_slot = (prd.face ? 0u : Nsources + 1u) + prd.source_ID;
                atomicFloatAdd(&spectral_incident[(size_t) face_slot * Nprimitives * Nbands_launch + ind_origin], body_strength);
            }
            if (Ncameras > 0) {
                // Camera-weighted scatter, accumulated once per camera into that camera's [primitive][band]
                // block. This launch runs once per dispatch, not once per camera, so camera_ID is not
                // meaningful here; weighting by a single camera handed every other camera that camera's image.
                size_t cam_scatter_stride = (size_t) Nprimitives * Nbands_launch;
                for (unsigned int cam = 0; cam < Ncameras; cam++) {
                    // Use BufferIndexer: [source][primitive][band][camera]
                    size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, cam);
                    float t_rho_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(7, prd.source_ID, prd.palette_entry, b_global, cam) : rho_cam[indc];
                    float t_tau_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(8, prd.source_ID, prd.palette_entry, b_global, cam) : tau_cam[indc];
                    size_t ind_cam = cam * cam_scatter_stride + ind_origin;
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
        if (Ncameras > 0) {
            accumulateWhiteReference(ind_origin, b_global, strength);
        }
    }
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

RT_PROGRAM void miss_diffuse() {

    // Convert UUID to array position
    uint origin_position = primitive_positions[prd.origin_UUID];

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

            if (origin_is_cover) {
                depositOnCoverOrigin(origin_position, ind_origin, b_global, origin_cos, strength);
            } else {
                // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest
                const int layer = surfaceLayerTable(origin_position, b_global);
                const float layer_albedo = layer >= 0 ? surfaceLayerAlbedo(layer, prd.origin_cos) : 0.f;
                const float body_strength = layer >= 0 ? strength * (1.f - layer_albedo) : strength;
                if (layer_albedo > 0.f) {
                    depositLayerReflection(ind_origin, b_global, strength * layer_albedo);
                }

                //  absorption
                atomicAdd(&radiation_in[ind_origin], body_strength * (1.f - t_rho - t_tau));
                if (face_absorption_enabled && prd.face) {
                    atomicAdd(&radiation_in_top[ind_origin], body_strength * (1.f - t_rho - t_tau));
                }

                if (t_rho > 0 || t_tau > 0) {
                    if (prd.face) { // reflection from top, transmission from bottom
                        atomicFloatAdd(&scatter_buff_top[ind_origin], body_strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], body_strength * t_tau); // transmission
                    } else { // reflection from bottom, transmission from top
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], body_strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_top[ind_origin], body_strength * t_tau); // transmission
                    }
                }
                // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
                if (spectral_incident_enabled != 0u && spectral_palette[spectral_palette[3] + origin_position] != 0xFFFFFFFFu && body_strength != 0) {
                    const uint face_slot = (prd.face ? 0u : Nsources + 1u) + prd.source_ID;
                    atomicFloatAdd(&spectral_incident[(size_t) face_slot * Nprimitives * Nbands_launch + ind_origin], body_strength);
                }
                if (Ncameras > 0) {
                    // Camera-weighted scatter, accumulated once per camera into that camera's [primitive][band]
                    // block. This launch runs once per dispatch, not once per camera, so camera_ID is not
                    // meaningful here; weighting by a single camera handed every other camera that camera's image.
                    size_t cam_scatter_stride = (size_t) Nprimitives * Nbands_launch;
                    for (unsigned int cam = 0; cam < Ncameras; cam++) {
                        // Use BufferIndexer: [source][primitive][band][camera]
                        size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, cam);
                        float t_rho_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(7, prd.source_ID, prd.palette_entry, b_global, cam) : rho_cam[indc];
                        float t_tau_cam = d_spectralPaletteApplies(prd.palette_entry, b_global) ? d_spectralPaletteValueCam(8, prd.source_ID, prd.palette_entry, b_global, cam) : tau_cam[indc];
                        size_t ind_cam = cam * cam_scatter_stride + ind_origin;
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
            if (Ncameras > 0) {
                accumulateWhiteReference(ind_origin, b_global, sky_strength);
                // Note: Don't accumulate diffuse sky radiation to radiation_specular
                // Specular should only reflect DIRECT source radiation (accumulated in miss_direct)
            }
        }
    }
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

RT_PROGRAM void miss_pixel_label() {

    camera_pixel_depth[prd.origin_UUID] = -1;
}
