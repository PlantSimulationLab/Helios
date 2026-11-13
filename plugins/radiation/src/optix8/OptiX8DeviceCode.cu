/** \file "OptiX8DeviceCode.cu" OptiX 8.1 device programs for the radiation model.

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

#include <optix.h>
#include <optix_device.h>
#include "OptiX8Math.h"
#include "OptiX8LaunchParams.h"
#include "NormalMapMath.h"

// ---------------------------------------------------------------------------
// Launch params declared as constant memory (filled by optixLaunch)
// ---------------------------------------------------------------------------
extern "C" __constant__ OptiX8LaunchParams params;

// ---------------------------------------------------------------------------
// Device utility functions (adapted from RayTracing.cuh, OptiX-API-agnostic)
// ---------------------------------------------------------------------------

static __forceinline__ __device__ unsigned int lcg(unsigned int &prev) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

static __forceinline__ __device__ float rnd(unsigned int &prev) {
    return (float)lcg(prev) / (float)0x01000000;
}

template<unsigned int N>
static __forceinline__ __device__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0, v1 = val1, s0 = 0;
    for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

__device__ __forceinline__ void atomicFloatAdd(float *address, float val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void d_transformPoint(const float (&T)[16], float3 &v) {
    float3 V;
    V.x = T[0]*v.x + T[1]*v.y + T[2]*v.z  + T[3];
    V.y = T[4]*v.x + T[5]*v.y + T[6]*v.z  + T[7];
    V.z = T[8]*v.x + T[9]*v.y + T[10]*v.z + T[11];
    v = V;
}

__device__ __forceinline__ float3 d_rotatePoint(const float3 &pos, float theta, float phi) {
    float st = sinf(theta), ct = cosf(theta);
    float sp = sinf(phi),   cp = cosf(phi);
    float3 tmp;
    tmp.x =  cp*ct*pos.x + (-sp)*pos.y + cp*st*pos.z;
    tmp.y =  sp*ct*pos.x +   cp*pos.y  + sp*st*pos.z;
    tmp.z = -st*pos.x                  + ct*pos.z;
    return tmp;
}

__device__ __forceinline__ float d_magnitude(const float3 v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

static __forceinline__ __device__ float acos_safe(float x) {
    return acosf(fmaxf(-1.f, fminf(1.f, x)));
}

static __forceinline__ __device__ float asin_safe(float x) {
    return asinf(fmaxf(-1.f, fminf(1.f, x)));
}

// Translucent cover (glass/plastic) optical model: angular transmittance/reflectance/absorptance of a
// single dielectric sheet via Fresnel reflection (two interfaces, both polarizations, internal
// reflections summed in closed form) plus Bouguer absorption. This is the classic solar-glazing model
// (Duffie & Beckman). Inputs: cos_theta = |cos| of the incidence angle, n = refractive index (>1),
// KL = absorption product K*L (0 = lossless). Returns (tau, rho, alpha) with tau+rho+alpha == 1.
static __forceinline__ __device__ float3 glass_tau_rho_alpha(float cos_theta, float n, float KL) {
    cos_theta = fmaxf(1e-4f, fminf(1.f, cos_theta)); // guard grazing/degenerate
    const float theta   = acos_safe(cos_theta);
    const float sin_t   = sinf(theta);
    const float sin_tr  = sin_t / n;                 // Snell
    const float cos_tr  = sqrtf(fmaxf(0.f, 1.f - sin_tr * sin_tr));
    const float theta_r = asin_safe(sin_tr);

    // Fresnel interface reflectance for the two polarizations. As theta -> 0 the sin/tan ratios
    // become 0/0, so fall back to the normal-incidence form r0 = ((n-1)/(n+1))^2.
    float r_par, r_per;
    if (theta < 1e-3f) {
        const float r0 = ((n - 1.f) / (n + 1.f)) * ((n - 1.f) / (n + 1.f));
        r_par = r0;
        r_per = r0;
    } else {
        const float s_minus = sinf(theta_r - theta);
        const float s_plus  = sinf(theta_r + theta);
        const float t_minus = tanf(theta_r - theta);
        const float t_plus  = tanf(theta_r + theta);
        r_per = (s_minus * s_minus) / fmaxf(1e-12f, s_plus * s_plus); // perpendicular
        r_par = (t_minus * t_minus) / fmaxf(1e-12f, t_plus * t_plus); // parallel
    }

    // Bouguer single-pass absorption through the angled path length L/cos_tr.
    const float tau_a = (KL > 0.f) ? expf(-KL / fmaxf(1e-4f, cos_tr)) : 1.f;

    // Per-polarization sheet transmittance/reflectance with internal reflections summed in closed form.
    float tau = 0.f, rho = 0.f;
    #pragma unroll
    for (int pol = 0; pol < 2; pol++) {
        const float r   = (pol == 0) ? r_per : r_par;
        const float denom = fmaxf(1e-6f, 1.f - (r * tau_a) * (r * tau_a));
        const float tau_i = tau_a * (1.f - r) * (1.f - r) / denom;
        const float rho_i = r * (1.f + tau_a * tau_i);
        tau += 0.5f * tau_i;
        rho += 0.5f * rho_i;
    }
    tau = fmaxf(0.f, fminf(1.f, tau));
    rho = fmaxf(0.f, fminf(1.f, rho));
    float alpha = 1.f - tau - rho;
    if (alpha < 0.f) { alpha = 0.f; } // numerical guard
    return make_float3(tau, rho, alpha);
}

// Initialize the per-band translucent-cover transmittance accumulator to 1 (no attenuation yet).
static __forceinline__ __device__ void initCoverTransmittance(PerRayData &prd) {
    #pragma unroll
    for (int i = 0; i < HELIOS_MAX_RADIATION_BANDS; i++) {
        prd.cover_transmittance[i] = 1.f;
    }
}

// Evaluates the diffuse angular distribution (fd factor).
// Priority 1: Power-law (Harrison & Coombes) if extinction > 0
// Priority 2: Prague sky model if sky_params.w > 0
// Priority 3: Isotropic (returns 1.0)
static __device__ float evaluateDiffuseAngularDistribution(const float3 &ray_dir, const float3 &peak_dir,
                                                            float power_law_K, float power_law_norm,
                                                            const float4 &sky_params) {
    if (power_law_K > 0.f) {
        float psi = acos_safe(dot(peak_dir, ray_dir));
        psi = fmaxf(psi, M_PI / 180.f);
        return powf(psi, -power_law_K) * power_law_norm;
    }
    if (sky_params.w > 0.f) {
        float gamma     = acos_safe(dot(ray_dir, peak_dir)) * 180.f / M_PI;
        float cos_theta = fmaxf(ray_dir.z, 0.f);
        // A non-positive circumsolar width means no circumsolar region (a width of zero would give 0/0 = NaN toward the sun)
        float circumsolar_term = (sky_params.y > 0.f) ? (1.f + sky_params.x * expf(-gamma / sky_params.y)) : 1.f;
        float pattern   = circumsolar_term * (1.f + (sky_params.z - 1.f) * (1.f - cos_theta));
        return pattern * sky_params.w * M_PI;
    }
    return 1.f; // isotropic
}

// Load transform matrix for primitive at global position pos
__device__ __forceinline__ void loadTransformMatrix(uint32_t pos, float (&T)[16]) {
    for (int i = 0; i < 16; i++) {
        T[i] = params.transform_matrix[pos * 16 + i];
    }
}

// Sample texture mask at UV (uv_u, uv_v) in [0,1]^2 for mask at index msk_id.
// Returns true if the texel is opaque (intersection should be reported),
// false if transparent (reject the hit).
// Standard texture convention: uv_v=0 maps to the top row (iy=0).
static __forceinline__ __device__ bool sampleMask(int32_t msk_id, float uv_u, float uv_v) {
    if (msk_id < 0) return true; // no mask → always opaque
    const int32_t  width   = params.mask_sizes[msk_id * 2];
    const int32_t  height  = params.mask_sizes[msk_id * 2 + 1];
    const uint32_t offset  = params.mask_offsets[msk_id];
    int ix = (int)(floorf(float(width  - 1) * uv_u));
    int iy = (int)(floorf(float(height - 1) * (1.f - uv_v)));
    ix = max(0, min(ix, width  - 1));
    iy = max(0, min(iy, height - 1));
    return params.mask_data[offset + (uint32_t)(iy * width + ix)] != 0u;
}

static __forceinline__ __device__ bool sampleMask(int32_t msk_id, float2 uv) {
    return sampleMask(msk_id, uv.x, uv.y);
}

// Texture (u,v) of a launch point sp in a primitive's canonical space: [-0.5, 0.5]^2 for a patch or tile, and the triangle
// (0,0)-(0,1)-(1,1) for a triangle. Used by the ray-generation programs both for the mask and for the class of a spectral texture.
static __forceinline__ __device__ float2 originTextureUV(uint32_t pos, uint32_t ptype, const float3 &sp) {
    const bool has_uv = params.uv_IDs[pos] >= 0;
    if (ptype == 1) {
        if (!has_uv) {
            return make_float2(sp.y, sp.x); // (beta + gamma, gamma)
        }
        const float2 uv0 = params.uv_data[pos * 4 + 0];
        const float2 uv1 = params.uv_data[pos * 4 + 1];
        const float2 uv2 = params.uv_data[pos * 4 + 2];
        const float beta  = sp.y - sp.x; // weight at v1
        const float gamma = sp.x;        // weight at v2
        return make_float2(uv0.x + beta * (uv1.x - uv0.x) + gamma * (uv2.x - uv0.x),
                           uv0.y + beta * (uv1.y - uv0.y) + gamma * (uv2.y - uv0.y)); // sampleMask() applies the image-row flip
    }
    if (!has_uv) {
        return make_float2(sp.x + 0.5f, sp.y + 0.5f); // parametric: remap [-0.5, 0.5] to [0, 1]
    }
    const float2 uv0 = params.uv_data[pos * 4 + 0];
    const float2 uv1 = params.uv_data[pos * 4 + 1];
    const float2 uv2 = params.uv_data[pos * 4 + 2];
    return make_float2(uv0.x + (sp.x + 0.5f) * (uv1.x - uv0.x), uv0.y + (sp.y + 0.5f) * (uv2.y - uv0.y));
}

// Palette entry of the class of a primitive's spectral texture at texture coordinate (u,v), with the texel convention of
// sampleMask(), or UINT32_MAX if the primitive has no spectral texture or the texel is transparent (class 0). A ray leaving the
// primitive from that point uses the entry's reflectivity and transmissivity instead of the primitive's own.
static __forceinline__ __device__ uint32_t paletteEntryAt(uint32_t pos, float2 uv) {
    const uint32_t first_entry = params.palette_offset[pos];
    const int32_t  class_map   = params.class_map_IDs[pos];
    if (first_entry == 0xFFFFFFFFu || class_map < 0) return 0xFFFFFFFFu;
    const int32_t width  = params.class_map_sizes[class_map * 2];
    const int32_t height = params.class_map_sizes[class_map * 2 + 1];
    int ix = (int)(floorf(float(width  - 1) * uv.x));
    int iy = (int)(floorf(float(height - 1) * (1.f - uv.y)));
    ix = max(0, min(ix, width  - 1));
    iy = max(0, min(iy, height - 1));
    const uint32_t texel_class = params.class_map_data[params.class_map_offsets[class_map] + (uint32_t)(iy * width + ix)];
    return texel_class == 0u ? 0xFFFFFFFFu : first_entry + texel_class - 1u;
}

// ---------------------------------------------------------------------------
// Normal maps (the conventions are specified in include/NormalMapMath.h)
// ---------------------------------------------------------------------------

//! Reads the packed normal-map words for the shared helios_nm functions
struct NormalMapWords {
    const uint32_t *words;
    __device__ __forceinline__ uint32_t operator[](uint32_t i) const {
        return words[i];
    }
};

static __forceinline__ __device__ helios_nm::Vec3 toNM(const float3 &v) {
    return helios_nm::make(v.x, v.y, v.z);
}
static __forceinline__ __device__ float3 fromNM(const helios_nm::Vec3 &v) {
    return make_float3(v.x, v.y, v.z);
}

//! Normal map of the primitive at position pos, or helios_nm::NO_MAP
static __forceinline__ __device__ uint32_t normalMapOf(uint32_t pos) {
    if (params.normal_maps == nullptr) {
        return helios_nm::NO_MAP;
    }
    return helios_nm::mapIndex(NormalMapWords{params.normal_maps}, pos);
}


//! Shading normal on the top face of the primitive at position pos at canonical point sp (as originTextureUV() takes it), about the unit base normal base_top on the side of the facet's top normal:
//! the nearest texel for ray launches, bilinear for cameras
static __forceinline__ __device__ float3 normalMapShadingNormalAt(uint32_t pos, uint32_t ptype, uint32_t map, const float *T, const float3 &base_top, const float3 &sp, bool bilinear) {
    helios_nm::Vec3 E1, E2;
    helios_nm::primitiveEdges(T, ptype, E1, E2);
    const bool has_uv = params.uv_IDs[pos] >= 0;
    float2 uv0 = make_float2(0.f, 0.f), uv1 = uv0, uv2 = uv0;
    if (has_uv) {
        uv0 = params.uv_data[pos * 4 + 0];
        uv1 = params.uv_data[pos * 4 + 1];
        uv2 = params.uv_data[pos * 4 + 2];
    }
    float ax, ay, bx, by;
    helios_nm::uvJacobian(ptype, has_uv, uv0.x, uv0.y, uv1.x, uv1.y, uv2.x, uv2.y, ax, ay, bx, by);
    const helios_nm::Frame frame = helios_nm::buildFrame(E1, E2, ax, ay, bx, by, toNM(base_top));
    float u, v;
    helios_nm::texcoord(ptype, uv0.x, uv0.y, ax, ay, bx, by, sp.x, sp.y, u, v);
    const NormalMapWords words{params.normal_maps};
    const helios_nm::Vec3 n_ts = bilinear ? helios_nm::fetchBilinear(words, map, u, v) : helios_nm::fetchNearest(words, map, u, v);
    return fromNM(helios_nm::perturb(frame, n_ts));
}

//! The canonical point sp of a hit on the primitive at position pos from its hit attributes (patch-local coordinates plus 0.5, or the barycentric weights of a triangle's second and third corners)
static __forceinline__ __device__ float3 canonicalPointFromHit(uint32_t ptype, float attr_u, float attr_v) {
    return ptype == 1 ? make_float3(attr_v, attr_u + attr_v, 0.f) : make_float3(attr_u - 0.5f, attr_v - 0.5f, 0.f);
}

//! Shading normal on the top face of the primitive at position pos at a hit with attributes attr_u and attr_v (see __intersection__patch), with the facet's top normal returned in normal_top
static __forceinline__ __device__ float3 hitShadingNormalTopAt(uint32_t pos, uint32_t map, bool bilinear, float3 &normal_top, float attr_u, float attr_v) {
    float T[16];
    for (int i = 0; i < 16; i++) {
        T[i] = params.transform_matrix[pos * 16 + i];
    }
    const uint32_t ptype = params.primitive_type[pos];
    helios_nm::Vec3 E1, E2;
    helios_nm::primitiveEdges(T, ptype, E1, E2);
    normal_top = fromNM(helios_nm::normalize(helios_nm::cross(E1, E2)));
    const float3 sp = canonicalPointFromHit(ptype, attr_u, attr_v);
    return normalMapShadingNormalAt(pos, ptype, map, T, normal_top, sp, bilinear);
}

// ---------------------------------------------------------------------------
// Background radiation field (see RayTracingGeometry::periodicDualField()). With primitives excluded from the periodic
// boundary, the buffers of outgoing and scattered radiation, of what cameras read, and of the direct flux hold two fields:
// field 0, the scene as it is, in flux slots [0, Nprimitives), and field 1, the background scene without the excluded
// primitives, in flux slots [Nprimitives, 2 Nprimitives). A ray or camera reads field 0 where it hits the central copy
// of the domain and field 1 in every other copy. Material and geometry buffers are indexed by the primitive's position
// alone. Must match the Vulkan shaders (flux_slot() in common/buffer_indexing.glsl) and RayTracing.cuh (d_fluxSlot()).
// ---------------------------------------------------------------------------

//! Whether a background field is solved alongside the scene as it is
static __forceinline__ __device__ bool dualField() {
    return params.Nflux_slots != params.Nprimitives;
}

//! Flux slot of the primitive at position pos in a field
static __forceinline__ __device__ uint32_t fluxSlot(uint32_t pos, uint32_t field) {
    return pos + field * params.Nprimitives;
}

//! Index of the direct flux of source s reaching the primitive at position pos, in a field: radiation_specular is [source][flux slot][launch band]
static __forceinline__ __device__ size_t directFluxIndex(uint32_t s, uint32_t pos, uint32_t field, uint32_t b_launch) {
    return (size_t(s) * params.Nflux_slots + fluxSlot(pos, field)) * params.Nbands_launch + b_launch;
}

//! Whether the direct and diffuse rays of the primitive at position pos book what they deliver in both fields: there is a background field, and the primitive is in it
static __forceinline__ __device__ bool originBooksBothFields(uint32_t pos) {
    return dualField() && params.periodic_excluded[pos] == 0;
}

//! Sets the background-field members of the payload of a direct or diffuse ray launched from the primitive at position pos
/**
 * Only the *_dual hit and miss programs read them, which only the direct and diffuse rays of a scene with a background field run.
 */
static __forceinline__ __device__ void startDualRay(PerRayDataDual &prd, uint32_t pos) {
    prd.dual_ray = originBooksBothFields(pos);
    prd.excluded_resolved = false;
    prd.recorded_t = 3.0e38f;
    prd.recorded_position = 0u;
    prd.recorded_face = false;
    prd.recorded_u = 0.f;
    prd.recorded_v = 0.f;
}

//! Nothing to set for the payload of a scene without a background field (see the raygen programs, which are written once for both payloads)
static __forceinline__ __device__ void startDualRay(PerRayData &, uint32_t) {
}

//! At the start of a segment, a dual ray has passed no excluded primitive yet
static __forceinline__ __device__ void startDualSegment(PerRayDataDual &prd) {
    prd.recorded_t = 3.0e38f;
}

static __forceinline__ __device__ void startDualSegment(PerRayData &) {
}

//! The payload of a direct or diffuse ray in a scene with a background field (the *_dual programs only)
static __forceinline__ __device__ PerRayDataDual *getPayloadPRDDual() {
    return static_cast<PerRayDataDual *>(getPayloadPRD());
}

//! The field a ray reads where it is: the scene as it is in the central copy of the domain, and the background in every other copy
static __forceinline__ __device__ uint32_t fieldSeenBy(const PerRayData *prd) {
    return (dualField() && (prd->tile_x != 0 || prd->tile_y != 0)) ? 1u : 0u;
}

//! Shading normal on the top face of the primitive at position pos at the current hit (closest-hit programs only), with the facet's top normal returned in normal_top
static __forceinline__ __device__ float3 hitShadingNormalTop(uint32_t pos, uint32_t map, bool bilinear, float3 &normal_top) {
    return hitShadingNormalTopAt(pos, map, bilinear, normal_top, __uint_as_float(optixGetAttribute_2()), __uint_as_float(optixGetAttribute_3()));
}

//! Adds the diffuse radiation that a ray brings to one face of a normal-mapped primitive, times the direction it came from, to the record cameras shade the
//! primitive's scattered light with (see RayTracingGeometry::normal_map_slots and normalMapDiffuseRelief()). Nothing is recorded for other primitives.
static __forceinline__ __device__ void recordIncidentDirection(uint32_t pos, bool face_top, uint32_t b_launch, float body_strength, const float3 &direction, uint32_t field = 0u) {
    if (params.normal_map_moments == nullptr || body_strength == 0.f) {
        return;
    }
    const uint32_t slot = params.normal_map_slots[pos];
    if (slot == 0xFFFFFFFFu) {
        return;
    }
    // The background field's moments are in a second set of slots
    float *moment = params.normal_map_moments + ((size_t(slot + field * params.normal_map_slot_count) * 2u + (face_top ? 0u : 1u)) * params.Nbands_launch + b_launch) * 3u;
    atomicFloatAdd(&moment[0], body_strength * direction.x);
    atomicFloatAdd(&moment[1], body_strength * direction.y);
    atomicFloatAdd(&moment[2], body_strength * direction.z);
}

//! The change, for a camera, in the scattered light the viewed face of a normal-mapped primitive sends out where its shading normal tilts away from the base normal
//! (the facet's, or the smoothed one) by the tangential vector m_t, given the camera-weighted reflectivity and transmissivity of the diffuse slot. The diffuse rays of
//! a primitive gather about its facet normal, as without a map, so that it absorbs what its footprint intercepts; with the radiance reaching each face taken as linear in
//! direction, fitted to the recorded first moment M of what reached it, a facet tilted by m_t would receive (8/3) m_t . M_t more, M_t the part of M along the facet
//! (unit normal facet_normal) the moment was gathered on. The face toward the camera reflects that, and the light it transmits entered the other face, whose shading
//! normal is -m. A uniform sky gives no relief. With camera flux smoothing (smooth_corners[0] >= 0) and the moments averaged onto the smoothing vertices, the moments are interpolated
//! from the facet's corners with smooth_weights, so that the relief is continuous across facets.
static __forceinline__ __device__ float normalMapDiffuseRelief(uint32_t pos, bool face_top, uint32_t b_launch, const float3 &shading_normal, const float3 &base_normal, const float3 &facet_normal, float rho_d, float tau_d,
                                                               const int32_t *smooth_corners, const float *smooth_weights, uint32_t field = 0u) {
    if (params.normal_map_moments == nullptr) {
        return 0.f;
    }
    const uint32_t slot = params.normal_map_slots[pos];
    if (slot == 0xFFFFFFFFu) {
        return 0.f;
    }
    float3 same = make_float3(0.f, 0.f, 0.f), other = make_float3(0.f, 0.f, 0.f);
    const uint32_t face_same = face_top ? 0u : 1u;
    if (params.vertex_normal_map_moments != nullptr && smooth_corners[0] >= 0) {
        for (int c = 0; c < 4; c++) {
            if (smooth_corners[c] < 0) {
                continue;
            }
            const size_t vertex = size_t(smooth_corners[c]) + field * params.vertex_field_stride; // the background field's vertices follow those of the scene as it is
            const float *vertex_same = params.vertex_normal_map_moments + ((vertex * 2u + face_same) * params.Nbands_launch + b_launch) * 3u;
            const float *vertex_other = params.vertex_normal_map_moments + ((vertex * 2u + (1u - face_same)) * params.Nbands_launch + b_launch) * 3u;
            same = make_float3(same.x + smooth_weights[c] * vertex_same[0], same.y + smooth_weights[c] * vertex_same[1], same.z + smooth_weights[c] * vertex_same[2]);
            other = make_float3(other.x + smooth_weights[c] * vertex_other[0], other.y + smooth_weights[c] * vertex_other[1], other.z + smooth_weights[c] * vertex_other[2]);
        }
    } else {
        const size_t field_slot = size_t(slot) + field * params.normal_map_slot_count;
        const float *own_same = params.normal_map_moments + ((field_slot * 2u + face_same) * params.Nbands_launch + b_launch) * 3u;
        const float *own_other = params.normal_map_moments + ((field_slot * 2u + (1u - face_same)) * params.Nbands_launch + b_launch) * 3u;
        same = make_float3(own_same[0], own_same[1], own_same[2]);
        other = make_float3(own_other[0], own_other[1], own_other[2]);
    }
    float3 weighted = make_float3(rho_d * same.x - tau_d * other.x, rho_d * same.y - tau_d * other.y, rho_d * same.z - tau_d * other.z);
    const float normal_part = dot(weighted, facet_normal);
    weighted = make_float3(weighted.x - normal_part * facet_normal.x, weighted.y - normal_part * facet_normal.y, weighted.z - normal_part * facet_normal.z);
    const float along = dot(shading_normal, base_normal);
    const float3 m_t = make_float3(shading_normal.x - along * base_normal.x, shading_normal.y - along * base_normal.y, shading_normal.z - along * base_normal.z);
    return (8.f / 3.f) * dot(m_t, weighted);
}

// Reflectivity, transmissivity and their camera-weighted values of the primitive a ray left, at the point it left from
static __forceinline__ __device__ float originRho(const PerRayData *prd, uint32_t radprop_ind, uint32_t b_global) {
    if (prd->palette_entry != 0xFFFFFFFFu && params.palette_band_active[b_global] != 0u) {
        return params.palette_rho[prd->source_ID * params.Npalette_entries * params.Nbands_global + prd->palette_entry * params.Nbands_global + b_global];
    }
    return params.rho[radprop_ind];
}
static __forceinline__ __device__ float originTau(const PerRayData *prd, uint32_t radprop_ind, uint32_t b_global) {
    if (prd->palette_entry != 0xFFFFFFFFu && params.palette_band_active[b_global] != 0u) {
        return params.palette_tau[prd->source_ID * params.Npalette_entries * params.Nbands_global + prd->palette_entry * params.Nbands_global + b_global];
    }
    return params.tau[radprop_ind];
}
static __forceinline__ __device__ float originRhoCam(const PerRayData *prd, uint32_t cam_ind, uint32_t b_global, uint32_t camera) {
    if (prd->palette_entry != 0xFFFFFFFFu && params.palette_band_active[b_global] != 0u) {
        return params.palette_rho_cam[((prd->source_ID * params.Npalette_entries + prd->palette_entry) * params.Nbands_global + b_global) * params.Ncameras + camera];
    }
    return params.rho_cam[cam_ind];
}
static __forceinline__ __device__ float originTauCam(const PerRayData *prd, uint32_t cam_ind, uint32_t b_global, uint32_t camera) {
    if (prd->palette_entry != 0xFFFFFFFFu && params.palette_band_active[b_global] != 0u) {
        return params.palette_tau_cam[((prd->source_ID * params.Npalette_entries + prd->palette_entry) * params.Nbands_global + b_global) * params.Ncameras + camera];
    }
    return params.tau_cam[cam_ind];
}

// Build a rotation-only 4×4 transform matrix (row-major) from Euler angles (rx, ry, rz)
// Matches OptiX 6's d_makeTransformMatrix convention.
static __forceinline__ __device__ void d_makeTransformMatrix(float3 rotation, float (&T)[16]) {
    float sx = sinf(rotation.x), cx = cosf(rotation.x);
    float sy = sinf(rotation.y), cy = cosf(rotation.y);
    float sz = sinf(rotation.z), cz = cosf(rotation.z);
    T[0]  = cz * cy;             T[1]  = cz * sy * sx - sz * cx; T[2]  = cz * sy * cx + sz * sx; T[3]  = 0.f;
    T[4]  = sz * cy;             T[5]  = sz * sy * sx + cz * cx; T[6]  = sz * sy * cx - cz * sx; T[7]  = 0.f;
    T[8]  = -sy;                 T[9]  = cy * sx;                T[10] = cy * cx;                T[11] = 0.f;
    T[12] = 0.f;                 T[13] = 0.f;                    T[14] = 0.f;                    T[15] = 1.f;
}

// Sample a point uniformly on the unit disk in the xy-plane (z=0).
// Based on Suffern (2007) "Ray Tracing from the Ground Up", Ch. 6 concentric mapping.
static __forceinline__ __device__ void d_sampleDisk(uint32_t &seed, float3 &sample) {
    float Rx = rnd(seed), Ry = rnd(seed);
    float sp_x = -1.f + 2.f * Rx;
    float sp_y = -1.f + 2.f * Ry;
    float r, p;
    if (sp_x > -sp_y) {
        if (sp_x > sp_y) { r = sp_x; p = sp_y / sp_x; }
        else              { r = sp_y; p = 2.f - sp_x / sp_y; }
    } else {
        if (sp_x < sp_y) { r = -sp_x; p = 4.f + sp_y / sp_x; }
        else              { r = -sp_y; p = (sp_y != 0.f) ? 6.f - sp_x / sp_y : 0.f; }
    }
    p *= 0.25f * M_PI;
    sample = make_float3(r * cosf(p), r * sinf(p), 0.f);
}

// Sample a point uniformly on the unit square [-0.5,0.5]^2 in the xy-plane (z=0).
static __forceinline__ __device__ void d_sampleSquare(uint32_t &seed, float3 &sample) {
    sample = make_float3(-0.5f + rnd(seed), -0.5f + rnd(seed), 0.f);
}

// Invert a 4×4 row-major matrix (used for rect/disk source intersection tests).
static __forceinline__ __device__ void d_invertMatrix(const float (&m)[16], float (&minv)[16]) {
    float inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    det = 1.0f / det;
    for (int i = 0; i < 16; i++) minv[i] = inv[i] * det;
}

// Test if ray hits a sphere source (any intersection in front of origin)
static __forceinline__ __device__ bool d_raySphereIntersect(const float3 &ray_origin, const float3 &ray_direction,
                                                            const float3 &sphere_center, float sphere_radius) {
    const float3 oc = make_float3(ray_origin.x - sphere_center.x, ray_origin.y - sphere_center.y,
                                   ray_origin.z - sphere_center.z);
    const float b = dot(oc, ray_direction);
    const float c = dot(oc, oc) - sphere_radius * sphere_radius;
    const float disc = b * b - c;
    if (disc < 0.0f) return false;
    return (-b - sqrtf(disc)) > 0.0f;
}

// Test if ray hits the front face of a rectangular source
static __forceinline__ __device__ bool d_rayRectangleIntersect(const float3 &ray_origin, const float3 &ray_direction,
                                                               const float3 &rect_center, float rect_width, float rect_length,
                                                               const float3 &rect_rotation, float &out_cos_angle) {
    float transform[16];
    d_makeTransformMatrix(rect_rotation, transform);
    const float3 normal = make_float3(transform[2], transform[6], transform[10]);
    const float denom = dot(ray_direction, normal);
    if (denom >= -1e-6f) return false;
    const float3 oc = make_float3(rect_center.x - ray_origin.x, rect_center.y - ray_origin.y,
                                   rect_center.z - ray_origin.z);
    const float t = dot(oc, normal) / denom;
    if (t <= 0.0f) return false;
    float3 hit = make_float3(ray_origin.x + t * ray_direction.x - rect_center.x,
                              ray_origin.y + t * ray_direction.y - rect_center.y,
                              ray_origin.z + t * ray_direction.z - rect_center.z);
    float inv_t[16];
    d_invertMatrix(transform, inv_t);
    d_transformPoint(inv_t, hit);
    if (fabsf(hit.x) > rect_width * 0.5f || fabsf(hit.y) > rect_length * 0.5f) return false;
    out_cos_angle = -denom;
    return true;
}

// Test if ray hits the front face of a disk source
static __forceinline__ __device__ bool d_rayDiskIntersect(const float3 &ray_origin, const float3 &ray_direction,
                                                          const float3 &disk_center, float disk_radius,
                                                          const float3 &disk_rotation, float &out_cos_angle) {
    float transform[16];
    d_makeTransformMatrix(disk_rotation, transform);
    const float3 normal = make_float3(transform[2], transform[6], transform[10]);
    const float denom = dot(ray_direction, normal);
    if (denom >= -1e-6f) return false;
    const float3 oc = make_float3(disk_center.x - ray_origin.x, disk_center.y - ray_origin.y,
                                   disk_center.z - ray_origin.z);
    const float t = dot(oc, normal) / denom;
    if (t <= 0.0f) return false;
    const float3 hit = make_float3(ray_origin.x + t * ray_direction.x - disk_center.x,
                                    ray_origin.y + t * ray_direction.y - disk_center.y,
                                    ray_origin.z + t * ray_direction.z - disk_center.z);
    if (dot(hit, hit) > disk_radius * disk_radius) return false;
    out_cos_angle = -denom;
    return true;
}

// ---------------------------------------------------------------------------
// PerRayData accessor (uses getPRD() from OptiX8LaunchParams.h)
// ---------------------------------------------------------------------------

// getPRD() is defined in OptiX8LaunchParams.h (guarded by #ifdef __CUDACC__)

// ---------------------------------------------------------------------------
// Intersection dispatch: handles all primitive types via primitive_type lookup.
// All hitgroup SBT records reference this single entry point.
// prim_idx = optixGetPrimitiveIndex() is the global AABB index (= global pos).
// UUID is read from params.primitive_uuid[prim_idx] (global pos → UUID array).
// Triangle and patch vertices are derived from the transform matrix using
// canonical-space vertices that match those used in buildAABBs() on the host.
// ---------------------------------------------------------------------------

// Whether a ray is in the periodic copy of the domain of the primitive it was launched from, which it does not hit there. The primitive's
// other periodic copies are hit like any other primitive.
static __forceinline__ __device__ bool inOriginTile(const PerRayData *prd) {
    return prd->tile_x == prd->origin_tile_x && prd->tile_y == prd->origin_tile_y;
}

// Whether a primitive is absent from the periodic copy of the domain a ray is in: a primitive excluded from the periodic boundary
// exists only in the central copy, and nothing exists beyond a limited number of copies. Must match periodic_hidden() in the
// Vulkan shaders and periodicHidden() in RayTracing.cuh.
static __forceinline__ __device__ bool periodicHidden(const PerRayData *prd, uint32_t pos) {
    if (params.periodic_tile_tracking == 0u) {
        return false;
    }
    if ((params.periodic_max_repeats.x >= 0 && abs(prd->tile_x) > params.periodic_max_repeats.x) || (params.periodic_max_repeats.y >= 0 && abs(prd->tile_y) > params.periodic_max_repeats.y)) {
        return true; // outside the field of copies
    }
    return params.periodic_excluded != nullptr && params.periodic_excluded[pos] != 0 && (prd->tile_x != 0 || prd->tile_y != 0);
}

extern "C" __global__ void __intersection__patch() {
    const uint32_t prim_idx = optixGetPrimitiveIndex();
    const uint32_t pos      = prim_idx; // global AABB index == global primitive position
    const uint32_t ptype    = params.primitive_type[pos];

    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float  t_min         = optixGetRayTmin();
    const float  t_max         = optixGetRayTmax();

    const uint32_t uuid = params.primitive_uuid[prim_idx];

    if (ptype == 5) {
        // ---- Bbox face: planar quad intersection (periodic boundary wall) ----
        // Each bbox has 4 world-space vertices stored at bbox_vertices[bbox_local * 4 + v].
        const uint32_t bbox_local = uuid - params.bbox_UUID_base;
        const float3 v0 = params.bbox_vertices[bbox_local * 4 + 0];
        const float3 v1 = params.bbox_vertices[bbox_local * 4 + 1];
        const float3 v2 = params.bbox_vertices[bbox_local * 4 + 2];
        const float3 v3 = params.bbox_vertices[bbox_local * 4 + 3];

        // Quad normal from two edges
        float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 e3 = make_float3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);
        float3 n  = cross(e1, e3);
        float  nd = dot(ray_direction, n);
        if (fabsf(nd) < 1e-8f) return; // ray parallel to face

        // A ray is wrapped only by the face it leaves the domain through: faces come in (min, max) pairs, and vertex 0 of a face
        // minus vertex 0 of the other face of its pair points out of the domain. The face a wrapped ray starts on is therefore
        // never hit again, so the ray can be continued with no minimum distance, and a ray leaving the domain is wrapped even
        // if it starts closer to the face than the minimum distance. Such a hit is reported at t_min, with its true distance
        // in attribute 2.
        const float3 opposite_v0 = params.bbox_vertices[(bbox_local ^ 1u) * 4];
        if (dot(ray_direction, v0 - opposite_v0) <= 0.f) return;

        float t = dot(make_float3(v0.x - ray_origin.x, v0.y - ray_origin.y, v0.z - ray_origin.z), n) / nd;
        if (t > t_max) return;
        // A ray that left the periodic copies continues from beyond the face it left through (see handlePeriodicBoundaryHit()), and
        // this program can be called for it although the ray misses the face's bounding box
        if (params.periodic_tile_tracking != 0u && t < -0.5f * params.periodic_escape_offset)
            return;

        // Hit point must lie within the bounding box of the quad vertices
        float3 hit = make_float3(ray_origin.x + t * ray_direction.x,
                                  ray_origin.y + t * ray_direction.y,
                                  ray_origin.z + t * ray_direction.z);
        const float slack = 1e-4f;
        float mnx = fminf(fminf(v0.x, v1.x), fminf(v2.x, v3.x)) - slack;
        float mxx = fmaxf(fmaxf(v0.x, v1.x), fmaxf(v2.x, v3.x)) + slack;
        float mny = fminf(fminf(v0.y, v1.y), fminf(v2.y, v3.y)) - slack;
        float mxy = fmaxf(fmaxf(v0.y, v1.y), fmaxf(v2.y, v3.y)) + slack;
        float mnz = fminf(fminf(v0.z, v1.z), fminf(v2.z, v3.z)) - slack;
        float mxz = fmaxf(fmaxf(v0.z, v1.z), fmaxf(v2.z, v3.z)) + slack;
        if (hit.x < mnx || hit.x > mxx || hit.y < mny || hit.y > mxy ||
            hit.z < mnz || hit.z > mxz) return;

        // OptiX does not take -0 as a hit distance (the trace ends in an undefined state), which fmaxf() may return for a hit at zero distance
        optixReportIntersection(fabsf(fmaxf(t, t_min)), 0, uuid, 0u, __float_as_uint(t), __float_as_uint(0.f));
        return;
    }

    if (ptype != 0 && ptype != 1 && ptype != 3) return; // only patch, triangle, tile

    if (params.twosided_flag[pos] >= 2) return; // invisible primitive: never intersected

    // A direct or diffuse ray never hits the primitive it was launched from. The minimum ray distance alone does not ensure this
    // far from the origin, where the rounding error of the launch point can exceed it.
    const PerRayData *prd = getPayloadPRD();
    if (prd->skip_origin && params.primitive_positions[prd->origin_UUID] == pos && inOriginTile(prd))
        return;
    if (periodicHidden(prd, pos))
        return;

    float T[16];
    loadTransformMatrix(pos, T);

    if (ptype == 0 || ptype == 3) {
        // ---- Patch: rectangle in canonical [-0.5, 0.5]^2 space ----
        // Normal = cross product of the first two columns of T, the convention of the ray-generation programs (the third column
        // points the other way for a mirrored transform)
        float3 normal = normalize(cross(make_float3(T[0], T[4], T[8]), make_float3(T[1], T[5], T[9])));
        float  nd     = dot(ray_direction, normal);
        if (fabsf(nd) < 1e-8f) return; // ray parallel to patch plane

        // Patch centroid is the translation column of T
        float3 patch_origin = make_float3(T[3], T[7], T[11]);
        float  t = dot(patch_origin - ray_origin, normal) / nd;
        // A hit at zero distance (t_min is 0 on a ray's first segment) is not a hit, as in the OptiX 6.5 and Vulkan backends. It
        // would also be reported as -0 for a ray launched in the plane of the surface, which OptiX does not take as a hit distance:
        // the trace ended in an undefined state, running the direct rays' miss program for a diffuse ray.
        if (t <= t_min || t > t_max) return;

        // Project hit point into patch-local 2D coordinates
        float3 hit_local = ray_origin + t * ray_direction - patch_origin;
        float3 local_x   = make_float3(T[0], T[4], T[8]);
        float3 local_y   = make_float3(T[1], T[5], T[9]);
        float  lx2 = dot(local_x, local_x);
        float  ly2 = dot(local_y, local_y);
        if (lx2 < 1e-12f || ly2 < 1e-12f) return;
        float u = dot(hit_local, local_x) / lx2;
        float v = dot(hit_local, local_y) / ly2;
        if (u < -0.5f || u > 0.5f || v < -0.5f || v > 0.5f) return;

        // Texture mask check
        const int32_t msk_id = params.mask_IDs[pos];
        if (msk_id >= 0) {
            float uv_u, uv_v;
            if (params.uv_IDs[pos] >= 0) {
                // Custom UV: bilinear from stored corner UVs
                float2 uv0 = params.uv_data[pos * 4 + 0]; // UV at (u=0, v=0) corner
                float2 uv1 = params.uv_data[pos * 4 + 1]; // UV at (u=1, v=0) corner
                float2 uv2 = params.uv_data[pos * 4 + 2]; // UV at (u=0, v=1) corner
                float du = uv1.x - uv0.x;
                float dv = uv2.y - uv0.y;
                uv_u = uv0.x + (u + 0.5f) * du;
                uv_v = uv0.y + (v + 0.5f) * dv;
            } else {
                // Parametric UV: remap from [-0.5, 0.5] to [0, 1]
                uv_u = u + 0.5f;
                uv_v = v + 0.5f;
            }
            if (!sampleMask(msk_id, uv_u, uv_v)) return;
        }

        // Report the intersection for BOTH faces. A one-sided primitive (twosided_flag==0) still
        // occludes a ray hitting its back face — matching the canonical OptiX 6 and Vulkan backends.
        // The hit face is recorded in face_attr (1 = front/top, 0 = back/bottom) for downstream
        // energy/face bookkeeping; one-sidedness is enforced at ray launch (raygen), not here.
        uint32_t face_attr = (nd < 0.f) ? 1u : 0u;
        // Carry the patch-local hit position, remapped from [-0.5,0.5] to [0,1], so that the camera program can interpolate a per-vertex quantity bilinearly across the patch. Corner zero of
        // Context::getPrimitiveVertices() sits at (0,0) here, corner one at (1,0), corner two at (1,1) and corner three at (0,1).
        optixReportIntersection(t, 0, uuid, face_attr, __float_as_uint(u + 0.5f), __float_as_uint(v + 0.5f));

    } else {
        // ---- Triangle: canonical vertices (0,0,0), (0,1,0), (1,1,0) ----
        // World-space vertices via T (consistent with buildAABBs canonical vertices)
        const float3 v0 = make_float3(T[3], T[7], T[11]);
        const float3 v1 = make_float3(T[1] + T[3], T[5] + T[7], T[9] + T[11]);
        const float3 v2 = make_float3(T[0] + T[1] + T[3], T[4] + T[5] + T[7], T[8] + T[9] + T[11]);

        // Shirley's ray-triangle intersection (Möller–Trumbore style)
        float a = v0.x - v1.x, b = v0.x - v2.x, c = ray_direction.x, d = v0.x - ray_origin.x;
        float e = v0.y - v1.y, f = v0.y - v2.y, g = ray_direction.y, h = v0.y - ray_origin.y;
        float i = v0.z - v1.z, j = v0.z - v2.z, k = ray_direction.z, l = v0.z - ray_origin.z;

        float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
        float q = g * i - e * k, s = e * j - f * i;

        float tri_denom = a * m + b * q + c * s;
        if (fabsf(tri_denom) < 1e-12f) return;
        float inv_denom = 1.f / tri_denom;

        float e1   = d * m - b * n - c * p;
        float beta = e1 * inv_denom;
        if (beta < 0.f) return;

        float r     = e * l - h * i;
        float e2    = a * n + d * q + c * r;
        float gamma = e2 * inv_denom;
        if (gamma < 0.f || beta + gamma > 1.f) return;

        float e3 = a * p - b * r + d * s;
        float t  = e3 * inv_denom;
        // A hit at zero distance (t_min is 0 on a ray's first segment) is not a hit, as in the OptiX 6.5 and Vulkan backends. It
        // would also be reported as -0 for a ray launched in the plane of the surface, which OptiX does not take as a hit distance:
        // the trace ended in an undefined state, running the direct rays' miss program for a diffuse ray.
        if (t <= t_min || t > t_max) return;

        // Texture mask check (using barycentric UV)
        const int32_t msk_id = params.mask_IDs[pos];
        if (msk_id >= 0) {
            float uv_u, uv_v;
            if (params.uv_IDs[pos] >= 0) {
                // Custom UV: interpolate from stored per-vertex UV
                float2 uv0 = params.uv_data[pos * 4 + 0];
                float2 uv1 = params.uv_data[pos * 4 + 1];
                float2 uv2 = params.uv_data[pos * 4 + 2];
                // beta = weight at v1, gamma = weight at v2, (1-beta-gamma) = weight at v0
                float2 uv = make_float2(uv0.x + beta * (uv1.x - uv0.x) + gamma * (uv2.x - uv0.x),
                                        uv0.y + beta * (uv1.y - uv0.y) + gamma * (uv2.y - uv0.y));
                uv_u = uv.x;
                uv_v = uv.y; // sampleMask() applies the image-row flip; flipping here too mirrored triangle masks
            } else {
                // Parametric UV: use barycentric coordinates directly
                uv_u = beta + gamma; // along u-axis (v1 is at u=1)
                uv_v = gamma;        // along v-axis (v2 is at v=1)
            }
            if (!sampleMask(msk_id, uv_u, uv_v)) return;
        }

        // Face from cross product normal vs ray direction
        float3 edge0   = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 edge1   = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        float3 tri_nrm = make_float3(edge0.y * edge1.z - edge0.z * edge1.y,
                                     edge0.z * edge1.x - edge0.x * edge1.z,
                                     edge0.x * edge1.y - edge0.y * edge1.x);
        // Report the intersection for BOTH faces (see note in the patch branch above): one-sided
        // primitives still occlude back-face hits, matching the canonical OptiX 6 / Vulkan backends.
        uint32_t face_attr = (dot(ray_direction, tri_nrm) < 0.f) ? 1u : 0u;
        // beta and gamma are the barycentric weights of the second and third vertices, already computed above for the texture mask. Passing them on lets the camera program interpolate a per-vertex
        // quantity across the facet instead of holding it constant, at the cost of two attribute slots and no extra arithmetic.
        optixReportIntersection(t, 0, uuid, face_attr, __float_as_uint(beta), __float_as_uint(gamma));
    }
}

extern "C" __global__ void __intersection__disk() {
    // Disk intersection is not yet implemented. Disk primitives are rejected at
    // the host level in updateGeometry(), so this program is never invoked.
}

extern "C" __global__ void __intersection__tile() {
    // Tile intersection is handled by __intersection__patch (ptype == 3).
}

extern "C" __global__ void __intersection__bbox() {
    // Bbox intersection is not yet implemented in this backend.
}

// ---------------------------------------------------------------------------
// Translucent cover (glass/plastic) support
// ---------------------------------------------------------------------------

// Diffuse, camera and pixel-label rays continue past a translucent cover from this distance (m) beyond it.
#define COVER_EXIT_OFFSET 1e-4f
// Number of periodic-boundary hits after which a ray of any type is abandoned (the ray is wrapped at most MAX_PERIODIC_WRAPS - 1
// times): it deposits nothing, as if blocked. Must match MAX_PERIODIC_WRAPS in RayTracing.cuh and the Vulkan shaders.
#define MAX_PERIODIC_WRAPS 1000
// Number of translucent covers a diffuse, camera or pixel-label ray can pass in addition to its periodic-boundary wraps.
#define MAX_COVER_PASSES 32
// Maximum number of segments a diffuse, camera or pixel-label ray is traced in: each translucent cover passed
// and each periodic-boundary wrap starts a new segment. A ray that runs out stops without further contribution.
#define MAX_RAY_SEGMENTS (MAX_PERIODIC_WRAPS + MAX_COVER_PASSES)

// Compute |cos(theta)| between a ray and the surface normal of a primitive.
// Mirrors the normal construction in __intersection__patch (cross product of the first two transform
// columns for patch/tile, edge cross-product for triangle). Returns a value in (0, 1]; degenerate cases return 1.
static __forceinline__ __device__ float coverCosTheta(uint32_t hit_position, const float3 &ray_dir) {
    const uint32_t ptype = params.primitive_type[hit_position];
    float T[16];
    loadTransformMatrix(hit_position, T);
    float3 normal;
    if (ptype == 1) { // triangle
        const float3 v0 = make_float3(T[3], T[7], T[11]);
        const float3 v1 = make_float3(T[1] + T[3], T[5] + T[7], T[9] + T[11]);
        const float3 v2 = make_float3(T[0] + T[1] + T[3], T[4] + T[5] + T[7], T[8] + T[9] + T[11]);
        const float3 e0 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        const float3 e1 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        normal = cross(e0, e1);
    } else { // patch or tile
        normal = cross(make_float3(T[0], T[4], T[8]), make_float3(T[1], T[5], T[9]));
    }
    const float nmag = d_magnitude(normal);
    if (nmag < 1e-12f) return 1.f;
    return fminf(1.f, fabsf(dot(ray_dir, normal)) / nmag);
}

// Whether a primitive is a translucent cover for the rays of this launch. A single ray carries every
// launched band, so a primitive that is glass in some launched bands but opaque in others cannot be both
// transmitted and blocked; it is a cover only if it is glass in ALL launched bands, and is otherwise a
// normal opaque primitive for the whole ray. This matches the OptiX 6 and Vulkan backends. Glass
// properties do not depend on the source, so the source-0 slice of the material buffers is read.
static __forceinline__ __device__ bool coverInAllLaunchedBands(uint32_t position) {
    if (params.is_glass == nullptr || params.band_launch_flag == nullptr) return false;
    bool any_band_launched = false;
    for (uint32_t b_global = 0; b_global < params.Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        any_band_launched = true;
        if (params.is_glass[position * params.Nbands_global + b_global] == 0) return false;
    }
    return any_band_launched;
}

// Angular (tau, rho, alpha) of a cover primitive in one band, for a ray at |cos(theta)| to its normal.
static __forceinline__ __device__ float3 coverTauRhoAlpha(uint32_t position, uint32_t b_global, float cos_theta) {
    const uint32_t mat_ind = position * params.Nbands_global + b_global;
    return glass_tau_rho_alpha(cos_theta, params.glass_n[mat_ind], params.glass_KL[mat_ind]);
}

// Book the radiation arriving at the face of a cover primitive that launched the ray. The cover absorbs
// alpha(theta) of it and reflects rho(theta) back out of the same face, into both the scattered-energy
// buffers and every camera's camera-weighted scatter. The transmitted part tau(theta) continues through
// the cover undeviated; it is delivered to the surfaces behind the cover by their own rays passing through
// it, so it is not scattered here. A cover's reflectance is spectrally flat within a band, so each
// camera weights it by that camera's white reference.
// With a background field (see dualField()), field 1 books only what the cover sends out; what it absorbs is that of the scene as it is.
static __forceinline__ __device__ void depositOnCoverOrigin(const PerRayData *prd, uint32_t origin_position, uint32_t ind_origin, uint32_t b_global, float cos_theta, float strength, uint32_t field = 0u) {
    if (strength <= 0.f) return;
    const float3 tra = coverTauRhoAlpha(origin_position, b_global, cos_theta);

    if (field == 0u) {
        atomicFloatAdd(&params.radiation_in[ind_origin], strength * tra.z);
        if (params.radiation_in_top && prd->face) {
            atomicFloatAdd(&params.radiation_in_top[ind_origin], strength * tra.z);
        }
    }

    const uint32_t ind_slot = ind_origin + field * params.Nprimitives * params.Nbands_launch;
    float *scatter_face = prd->face ? params.scatter_buff_top : params.scatter_buff_bottom;
    atomicFloatAdd(&scatter_face[ind_slot], strength * tra.y);

    if (params.Ncameras > 0 && params.scatter_buff_top_cam) {
        const uint32_t Ncameras   = params.Ncameras;
        const uint32_t cam_stride = params.Nflux_slots * params.Nbands_launch;
        float *scatter_face_cam   = prd->face ? params.scatter_buff_top_cam : params.scatter_buff_bottom_cam;
        for (uint32_t cam = 0; cam < Ncameras; cam++) {
            const float white = params.white_reference_cam[(prd->source_ID * params.Nbands_global + b_global) * Ncameras + cam];
            atomicFloatAdd(&scatter_face_cam[cam * cam_stride + ind_slot], strength * tra.y * white);
        }
    }
}

// Continue a diffuse, camera or pixel-label ray past a translucent cover it hit at distance t_hit. What lies
// behind the cover is seen attenuated by its transmittance: the ray's throughput (cover_transmittance) is
// multiplied by tau(theta) in every launched band, and the raygen program re-traces from just beyond the
// cover. The ray is not refracted.
static __forceinline__ __device__ void continuePastCover(PerRayData *prd, uint32_t hit_position, float t_hit) {
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float  cos_theta     = coverCosTheta(hit_position, ray_direction);

    int b = -1;
    for (uint32_t b_global = 0; b_global < params.Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        b++;
        if (b >= HELIOS_MAX_RADIATION_BANDS) break; // host guarantees this never trips
        prd->cover_transmittance[b] *= coverTauRhoAlpha(hit_position, b_global, cos_theta).x;
    }

    prd->hit_cover  = true;
    prd->cover_exit = ray_origin + ray_direction * (t_hit + COVER_EXIT_OFFSET);
}

// ---------------------------------------------------------------------------
// Miss programs
// ---------------------------------------------------------------------------

//! Add light arriving at the face a ray was launched from to every camera's white reference
/**
 * The white reference is what each camera would record from a spectrally flat, perfectly white surface in place of the primitive: the arriving light weighted by white_reference_cam. Only light
 * straight from a radiation source or the sky is added, so a camera's "auto" white balance neutralizes the light illuminating the scene and not light already coloured by other surfaces.
 */
static __forceinline__ __device__ void accumulateWhiteReference(const PerRayData *prd, uint32_t ind_origin, uint32_t b_global, float strength) {
    if (params.Ncameras == 0 || !params.white_reference_cam || !params.white_reference_top_cam || strength <= 0.f) {
        return;
    }
    const uint32_t Ncameras   = params.Ncameras;
    const uint32_t cam_stride = params.Nprimitives * params.Nbands_launch;
    float *face_white_reference = prd->face ? params.white_reference_top_cam : params.white_reference_bottom_cam;
    for (uint32_t cam = 0; cam < Ncameras; cam++) {
        const float white = params.white_reference_cam[(prd->source_ID * params.Nbands_global + b_global) * Ncameras + cam];
        atomicFloatAdd(&face_white_reference[cam * cam_stride + ind_origin], strength * white);
    }
}

// ---------------------------------------------------------------------------
// Specular surface layer (see RayTracingMaterial::surface_model). A primitive with a layer reflects a(mu) of the radiation
// arriving at its surface into the second component of its outgoing radiation, with angular shape a(mu), and its body
// absorbs, reflects and transmits the rest (rho and tau are those of the body), sending it out in the first component with
// shape 1 - a(mu). Both components are stored in the outgoing and scattered buffers, the second after the first.
// ---------------------------------------------------------------------------

//! Albedo table of the surface layer of the primitive at position pos in global band b_global, or nullptr if it has none there
static __forceinline__ __device__ const float *surfaceLayerTable(uint32_t pos, uint32_t b_global) {
    if (params.surface_model == nullptr) return nullptr;
    const uint32_t model = params.surface_model[pos];
    if (model == 0u) return nullptr;
    const uint32_t table = params.surface_model_band_table[(model - 1u) * params.Nbands_global + b_global];
    if (table == 0xFFFFFFFFu) return nullptr;
    return params.surface_layer_tables + (size_t)table * (params.surface_layer_table_size + 3u);
}

//! Directional albedo a(mu) of a surface layer, interpolated linearly in its table
static __forceinline__ __device__ float surfaceLayerAlbedo(const float *table, float mu) {
    const uint32_t size = params.surface_layer_table_size;
    const float x = fminf(fmaxf(mu, 0.f), 1.f) * float(size - 1u);
    const uint32_t i = min(uint32_t(x), size - 2u);
    return table[i] + (x - float(i)) * (table[i + 1u] - table[i]);
}

//! Flux a face sends out at the angle whose cosine is mu_out, from its two outgoing components [component][flux slot][band]; ind is the index of the first
static __forceinline__ __device__ float outgoingLayerFlux(const float *outgoing, uint32_t ind, const float *table, float mu_out) {
    const uint32_t layer_offset = params.Nflux_slots * params.Nbands_launch;
    const float a = surfaceLayerAlbedo(table, mu_out);
    const float abar = table[params.surface_layer_table_size];
    return outgoing[ind] * (1.f - a) / (1.f - abar) + outgoing[layer_offset + ind] * a / abar;
}

//! Fresnel + GGX reflection of a surface layer from direction to_source to direction to_viewer (unit vectors), in 1/sr, for the face whose unit normal is face_normal; 0 unless both are on that face's side
static __forceinline__ __device__ float surfaceLayerLobe(const float *table, const float3 &face_normal, const float3 &to_source, const float3 &to_viewer) {
    const float mu_s = dot(face_normal, to_source);
    const float mu_o = dot(face_normal, to_viewer);
    if (mu_s <= 0.f || mu_o <= 0.f) return 0.f;
    const uint32_t size = params.surface_layer_table_size;
    const float n = table[size + 1u];
    const float alpha2 = table[size + 2u] * table[size + 2u];
    const float3 half_vector = normalize(make_float3(to_source.x + to_viewer.x, to_source.y + to_viewer.y, to_source.z + to_viewer.z));
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

//! Flux, in the units of the outgoing buffers, that the surface layer of the primitive at position pos reflects toward to_viewer from the direct radiation of every source reaching point (see RadiationModel::runBand(): radiation_specular holds the incident direct flux). camera >= 0 weights each source by that camera's response.
static __forceinline__ __device__ float directLayerReflection(uint32_t pos, uint32_t b_launch, const float *table, const float3 &face_normal, const float3 &to_viewer, const float3 &point, int camera, uint32_t field = 0u) {
    float reflected = 0.f;
    for (uint32_t s = 0; s < params.Nsources; s++) {
        float incident = params.radiation_specular[directFluxIndex(s, pos, field, b_launch)];
        if (incident <= 0.f) continue;
        if (camera >= 0) {
            incident *= params.source_fluxes_cam[(s * params.Nbands_launch + b_launch) * params.Ncameras + uint32_t(camera)];
        }
        // A collimated source or the sun sphere lies in one direction; a finite source is seen in the direction of its centre
        const float3 to_source = (params.source_types[s] == 0 || params.source_types[s] == 2)
                                     ? normalize(params.source_positions[s])
                                     : normalize(make_float3(params.source_positions[s].x - point.x, params.source_positions[s].y - point.y, params.source_positions[s].z - point.z));
        reflected += incident * surfaceLayerLobe(table, face_normal, to_source, to_viewer);
    }
    return float(M_PI) * reflected; // radiance to flux units
}

//! Whether the primitive at position pos takes part in camera flux smoothing, whose direct radiation is then recorded for the camera (see RadiationModel::subtractCameraDirectFlux())
static __forceinline__ __device__ bool isSmoothedPrimitive(uint32_t pos) {
    return params.smoothing_vertex_indices != nullptr && params.smoothing_vertex_indices[4 * pos] >= 0;
}

//! Direction from point to radiation source s: the source's direction for a collimated source or the sun sphere, and toward its centre for a finite source
static __forceinline__ __device__ float3 directionToSource(uint32_t s, const float3 &point) {
    if (params.source_types[s] == 0 || params.source_types[s] == 2) {
        return normalize(params.source_positions[s]);
    }
    return normalize(make_float3(params.source_positions[s].x - point.x, params.source_positions[s].y - point.y, params.source_positions[s].z - point.z));
}

//! One sample of radiation source s seen from the point `origin`, drawn as the direct launch draws it
/**
 * Gives the direction toward the point sampled on the source (the source's direction for a collimated source), the distance to that point (1e38 for a collimated source), and the factor that, times
 * the source's flux and |n . direction| for a surface of normal n, gives the irradiance the sample stands for: the strength of a direct ray before its division by the number of rays. Cameras draw their
 * shadow rays with it too, so that the shadows they evaluate per pixel come from the same distribution of points on the source as the radiation solve, including the penumbra of the sun's disk and of a
 * finite source. Returns false for a sample on the back of a rectangle or disk source, which delivers nothing, and for an unknown source type.
 */
static __forceinline__ __device__ bool sampleSourceDirection(uint32_t s, const float3 &origin, uint32_t &seed, float3 &direction, float &tmax, double &factor) {
    const uint32_t src_type = params.source_types[s];
    if (src_type == 0) { // Collimated source
        direction = normalize(params.source_positions[s]);
        tmax = 1e38f;
        factor = 1.0;
        return true;
    }
    if (src_type == 1 || src_type == 2) { // Sphere source (type 1 = point, type 2 = sphere)
        const float theta_s = acos_safe(1.f - 2.f * rnd(seed));
        const float phi_s = rnd(seed) * 2.f * M_PI;
        const float3 sphere_pt = make_float3(0.5f * params.source_widths[s].x * sinf(theta_s) * cosf(phi_s), 0.5f * params.source_widths[s].x * sinf(theta_s) * sinf(phi_s), 0.5f * params.source_widths[s].x * cosf(theta_s));
        direction = sphere_pt + params.source_positions[s] - origin;
        tmax = d_magnitude(direction);
        direction = normalize(direction);

        // Integrate over the sphere's surface for the strength
        factor = 0.0;
        const uint32_t N = 10;
        for (uint32_t j = 0; j < N; j++) {
            for (uint32_t i = 0; i < N; i++) {
                const float theta = acos_safe(1.f - 2.f * (float(i) + 0.5f) / float(N));
                const float phi = (float(j) + 0.5f) * 2.f * M_PI / float(N);
                const float3 ldir = make_float3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));
                if (dot(ldir, direction) < 0.f) {
                    factor += (double) fabsf(dot(ldir, direction)) / ((double) tmax * (double) tmax) / double(N * N) * (double) params.source_widths[s].x * (double) params.source_widths[s].x;
                }
            }
        }
        return true;
    }
    if (src_type == 3) { // Rectangle source
        float light_transform[16];
        float3 rot3 = params.source_rotations[s];
        d_makeTransformMatrix(rot3, light_transform);

        float3 square_pt;
        d_sampleSquare(seed, square_pt);
        square_pt = make_float3(params.source_widths[s].x * square_pt.x, params.source_widths[s].y * square_pt.y, square_pt.z);
        d_transformPoint(light_transform, square_pt);

        float3 light_dir = make_float3(0.f, 0.f, 1.f);
        d_transformPoint(light_transform, light_dir);

        direction = square_pt + params.source_positions[s] - origin;
        if (dot(direction, light_dir) > 0.f) {
            return false; // the source does not emit from its back
        }
        tmax = d_magnitude(direction);
        direction = normalize(direction);
        factor = (double) fabsf(dot(light_dir, direction)) / ((double) tmax * (double) tmax) * (double) params.source_widths[s].x * (double) params.source_widths[s].y / M_PI;
        return true;
    }
    if (src_type == 4) { // Disk source
        float light_transform[16];
        float3 rot3 = params.source_rotations[s];
        d_makeTransformMatrix(rot3, light_transform);

        float3 disk_pt;
        d_sampleDisk(seed, disk_pt);
        d_transformPoint(light_transform, disk_pt);

        float3 light_dir = make_float3(0.f, 0.f, 1.f);
        d_transformPoint(light_transform, light_dir);

        direction = params.source_widths[s].x * disk_pt + params.source_positions[s] - origin;
        if (dot(direction, light_dir) > 0.f) {
            return false; // the source does not emit from its back
        }
        tmax = d_magnitude(direction);
        direction = normalize(direction);
        factor = (double) fabsf(dot(light_dir, direction)) / ((double) tmax * (double) tmax) * (double) params.source_widths[s].x * (double) params.source_widths[s].x;
        return true;
    }
    return false; // unknown source type
}

//! Direct radiation of a source reaching one face of a normal-mapped facet that is not smoothed, evaluated with the map's shading normal
/**
 * face_normal and face_shading_normal are the facet's normal and the shading normal of that face, and facet_value what the face itself received (see RadiationModel::subtractCameraDirectFlux()). That
 * carries the cosine of the facet's normal, which is replaced by that of the shading normal. A face turned away from the source received nothing. With soften, the rescaled value is multiplied by the
 * shadow-terminator softening of Chiang et al. (see helios_nm::terminatorSoftening()), so that relief does not end in a hard edge where the facet still faces the source but the shading normal barely
 * does. (A smoothed primitive's direct radiation is evaluated per pixel sample instead; see pixelDirectRadiation().)
 */
static __forceinline__ __device__ float faceDirect(float facet_value, const float3 &to_source, const float3 &face_normal, const float3 &face_shading_normal, bool soften) {
    const float mu_facet = dot(face_normal, to_source);
    const float mu_shading = fmaxf(0.f, dot(face_shading_normal, to_source));
    if (mu_shading <= 0.f || mu_facet <= 0.f) {
        return 0.f;
    }
    const float softening = soften ? helios_nm::terminatorSoftening(toNM(face_normal), toNM(face_shading_normal), toNM(to_source)) : 1.f;
    return facet_value * mu_shading / mu_facet * softening;
}

//! The outgoing radiation, first component, that the face seen of a normal-mapped primitive that is not smoothed sends out of the direct radiation of every source, evaluated with the shading normal
/**
 * See RadiationModel::subtractCameraDirectFlux(), which removes the facet's own value from what the camera reads. viewed_normal and shading_normal are on the viewed side. The viewed face reflects what
 * reaches it and passes on what reaches the other face; a surface layer, whose albedo table is `layer` (null for none), keeps 1 - a(mu) of it in the body. The direct radiation a facet records reached
 * the face toward the source.
 */
static __forceinline__ __device__ float shadedDirectBodyFlux(uint32_t pos, uint32_t b_launch, uint32_t b_global, const float *layer, const float3 &viewed_normal, const float3 &shading_normal, const float3 &point, bool soften,
                                                             uint32_t field = 0u) {
    const float3 back_normal = make_float3(-viewed_normal.x, -viewed_normal.y, -viewed_normal.z);
    const float3 back_shading_normal = make_float3(-shading_normal.x, -shading_normal.y, -shading_normal.z);
    float flux = 0.f;
    for (uint32_t s = 0; s < params.Nsources; s++) {
        const float incident = params.radiation_specular[directFluxIndex(s, pos, field, b_launch)];
        const float3 to_source = directionToSource(s, point);
        const bool viewed_face_toward_source = dot(viewed_normal, to_source) > 0.f;
        const size_t material = ((size_t(s) * params.Nprimitives + pos) * params.Nbands_global + b_global) * params.Ncameras + params.camera_ID;
        const float rho = params.rho_cam[material];
        const float tau = params.tau_cam ? params.tau_cam[material] : 0.f;
        if (rho > 0.f) {
            const float reaching = faceDirect(viewed_face_toward_source ? incident : 0.f, to_source, viewed_normal, shading_normal, soften);
            flux += rho * reaching * (layer ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, dot(shading_normal, to_source))) : 1.f);
        }
        if (tau > 0.f) {
            const float reaching = faceDirect(viewed_face_toward_source ? 0.f : incident, to_source, back_normal, back_shading_normal, soften);
            flux += tau * reaching * (layer ? 1.f - surfaceLayerAlbedo(layer, fmaxf(0.f, dot(back_shading_normal, to_source))) : 1.f);
        }
    }
    return flux;
}

//! Flux, in the units of the outgoing buffers, that the surface layer of a normal-mapped primitive that is not smoothed reflects toward to_viewer from the direct radiation of every source
/**
 * Evaluated with the shading normal rather than the facet's. viewed_normal is the facet's own unit normal on the side seen, and shading_normal the shading normal on that side. The incident flux carries
 * the cosine of the facet, so it is rescaled to the cosine of the shading normal; it reached the face toward the source, so only a source on the viewed side counts.
 */
static __forceinline__ __device__ float directLayerReflectionShaded(uint32_t pos, uint32_t b_launch, const float *table, const float3 &viewed_normal, const float3 &shading_normal, const float3 &to_viewer, const float3 &point, int camera,
                                                                    bool soften, uint32_t field = 0u) {
    float reflected = 0.f;
    for (uint32_t s = 0; s < params.Nsources; s++) {
        const float3 to_source = directionToSource(s, point);
        const float facet_value = dot(viewed_normal, to_source) > 0.f ? params.radiation_specular[directFluxIndex(s, pos, field, b_launch)] : 0.f;
        float incident = faceDirect(facet_value, to_source, viewed_normal, shading_normal, soften);
        if (incident <= 0.f)
            continue;
        if (camera >= 0) {
            incident *= params.source_fluxes_cam[(s * params.Nbands_launch + b_launch) * params.Ncameras + uint32_t(camera)];
        }
        reflected += incident * surfaceLayerLobe(table, shading_normal, to_source, to_viewer);
    }
    return float(M_PI) * reflected; // radiance to flux units
}

//! With camera flux smoothing: the direct radiation of every source at the point a camera ray hit, evaluated for this pixel sample with a shadow ray toward one point sampled on each source
/**
 * Whether a source reaches the point is decided here, by the point itself, rather than taken from the direct radiation its facet received, so that a cast shadow falls where the geometry puts it within a
 * facet. What reaches a face is the source's flux times the factor of the sample (see sampleSourceDirection()) times the cosine of the face's shading normal, times the transmittance of any translucent
 * covers on the way. The face of the shading normal toward the source receives it; a one-sided primitive's bottom face receives nothing, as in the direct launch. The shadow ray is traced with the
 * direct ray type, so it is blocked, passes covers and wraps at periodic boundaries exactly as a direct ray does, and reaches the source if it misses everything (__miss__camera_shadow). A ray that
 * reaches the wrap limit counts as blocked.
 *
 * Near the terminator a facet can face away from the source while the shading normal still faces it, and a ray from the flat facet would pass through the surface the facets stand for. Such a ray
 * starts instead from the hit point raised onto the smooth surface the vertex normals describe: the point projected onto the tangent plane of each corner it lies below, blended with the corner weights
 * (Hanika, "Hacking the shadow terminator", Ray Tracing Gems II, 2021). Every shadow ray also starts a few times the rounding error of the hit point (hit_distance from the camera) off the receiving face,
 * so that it cannot strike the facet it leaves, or a coplanar neighbour, from just behind it.
 *
 * Accumulates, for the launch bands from band_begin on (at most HELIOS_MAX_RADIATION_BANDS of them, indexed from band_begin), into `body` what the face seen sends out of that radiation (the primitive's camera-weighted reflectivity of what reaches
 * the face seen and transmissivity of what reaches the other face, with 1 - a(mu) of it for a surface layer), into `class_flux` the same through the camera-weighted spectra of the spectral-texture class `entry` (when it is not UINT32_MAX), and into
 * `lobe` what a surface layer reflects into the camera, in the units of the outgoing buffers. viewed_normal is the facet's normal, base_normal the smoothed normal and shading_normal the normal the camera shades with (the smoothed one tilted by a
 * normal map), all on the side the camera sees; v and w are the facet's smoothing vertices and their weights at the hit point.
 */
static __forceinline__ __device__ void pixelDirectRadiation(uint32_t hit_uuid, uint32_t pos, uint32_t prim_type, const float (&T)[16], const float3 &viewed_normal, const float3 &base_normal, const float3 &shading_normal, bool normal_mapped,
                                                            bool face_top, const float3 &point, float hit_distance, const float3 &to_viewer, const int32_t v[4], const float w[4], uint32_t entry, uint32_t band_begin, uint32_t &seed, float *body,
                                                            float *class_flux, float *lobe) {
    const uint32_t Nbands_l = params.Nbands_launch;
    const uint32_t band_end = band_begin + HELIOS_MAX_RADIATION_BANDS < Nbands_l ? band_begin + HELIOS_MAX_RADIATION_BANDS : Nbands_l;
    for (uint32_t b = 0; b < HELIOS_MAX_RADIATION_BANDS; b++) {
        body[b] = 0.f;
        class_flux[b] = 0.f;
        lobe[b] = 0.f;
    }
    const int8_t twosided = params.twosided_flag[pos];

    for (uint32_t s = 0; s < params.Nsources; s++) {
        float3 direction;
        float tmax;
        double factor;
        if (!sampleSourceDirection(s, point, seed, direction, tmax, factor)) {
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

        // Where the facet faces away from the source, start from the point raised onto the smooth surface
        float3 origin = point;
        if (dot(face_normal, direction) <= 0.f && (prim_type == 0 || prim_type == 1 || prim_type == 3)) {
            for (int c = 0; c < 4; c++) {
                if (v[c] < 0) {
                    continue;
                }
                float3 corner = prim_type == 1 ? make_float3(c == 2 ? 1.f : 0.f, c == 0 ? 0.f : 1.f, 0.f) : make_float3((c == 1 || c == 2) ? 0.5f : -0.5f, (c == 2 || c == 3) ? 0.5f : -0.5f, 0.f);
                d_transformPoint(T, corner);
                float3 corner_normal = make_float3(params.vertex_normals[3 * v[c] + 0], params.vertex_normals[3 * v[c] + 1], params.vertex_normals[3 * v[c] + 2]);
                if (dot(corner_normal, face_normal) < 0.f) {
                    corner_normal = -1.f * corner_normal;
                }
                const float below = dot(point - corner, corner_normal);
                if (below < 0.f) {
                    origin = origin - (w[c] * below) * corner_normal;
                }
            }
        }

        // Off the receiving face by a few times the rounding error of the hit point
        origin = origin + (1e-6f * (1.f + fmaxf(fmaxf(fabsf(point.x), fabsf(point.y)), fabsf(point.z)) + hit_distance)) * face_normal;

        // The shadow ray: the direct ray type's hit programs, and a miss program that marks the source reached
        PerRayData shadow;
        shadow.strength = 0.0;
        shadow.origin_UUID = hit_uuid;
        shadow.face = receiving_top;
        shadow.seed = seed;
        shadow.source_ID = (unsigned char) s;
        shadow.skip_origin = true;
        shadow.hit_periodic_boundary = false;
        // The shadow ray leaves the point in the periodic copy of the domain the camera ray reached it in
        shadow.tile_x = getPayloadPRD()->tile_x;
        shadow.tile_y = getPayloadPRD()->tile_y;
        shadow.origin_tile_x = shadow.tile_x; // the point it leaves is on the copy of the primitive in that copy of the domain
        shadow.origin_tile_y = shadow.tile_y;
        shadow.hit_cover = false;
        shadow.origin_cos = mu;
        shadow.palette_entry = 0xFFFFFFFFu;
        shadow.source_reached = false;
        initCoverTransmittance(shadow);
        float3 current_origin = origin;
        for (int wrap = 0; wrap < MAX_PERIODIC_WRAPS; ++wrap) {
            uint32_t u0, u1;
            packPointer(&shadow, u0, u1);
            optixTrace(params.traversable, current_origin, direction, 0.f, tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0u, 0u, 4u, // SBT: offset=0 (direct hit), stride=0, miss=4 (camera shadow)
                       u0, u1);
            if (!shadow.hit_periodic_boundary) {
                break;
            }
            current_origin = shadow.periodic_hit;
            if (params.source_types[s] != 0) {
                tmax = fmaxf(tmax - shadow.periodic_hit_t, 2e-4f);
            }
            shadow.hit_periodic_boundary = false;
        }
        if (!shadow.source_reached) {
            continue;
        }

        // A normal map softens the shading cosine where it tilts the normal past the smooth surface's terminator (see faceDirect())
        const float softening = normal_mapped ? helios_nm::terminatorSoftening(toNM(face_base_normal), toNM(face_shading_normal), toNM(direction)) : 1.f;
        const float irradiance_factor = float(factor) * mu * softening;
        if (irradiance_factor <= 0.f) {
            continue;
        }

        int b = -1;
        for (uint32_t b_global = 0; b_global < params.Nbands_global; b_global++) {
            if (!params.band_launch_flag[b_global]) {
                continue;
            }
            b++;
            if (uint32_t(b) < band_begin || uint32_t(b) >= band_end) {
                continue;
            }
            const uint32_t slot = uint32_t(b) - band_begin;
            const float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? shadow.cover_transmittance[b] : 1.f;
            const float reaching = params.source_fluxes[s * Nbands_l + uint32_t(b)] * irradiance_factor * cover_tau;
            if (reaching <= 0.f) {
                continue;
            }
            const size_t material = ((size_t(s) * params.Nprimitives + pos) * params.Nbands_global + b_global) * params.Ncameras + params.camera_ID;
            const float *layer = surfaceLayerTable(pos, b_global);
            const float body_fraction = layer ? 1.f - surfaceLayerAlbedo(layer, mu) : 1.f;
            if (reaches_viewed_face) {
                body[slot] += params.rho_cam[material] * reaching * body_fraction;
            } else if (params.tau_cam) {
                body[slot] += params.tau_cam[material] * reaching * body_fraction;
            }
            if (entry != 0xFFFFFFFFu) {
                const size_t ind_cam = ((size_t(s) * params.Npalette_entries + entry) * params.Nbands_global + b_global) * params.Ncameras + params.camera_ID;
                class_flux[slot] += (reaches_viewed_face ? params.palette_rho_cam[ind_cam] : params.palette_tau_cam[ind_cam]) * reaching * body_fraction;
            }
            if (layer && reaches_viewed_face && params.specular_reflection_enabled != 0u) {
                const float camera_weight = params.source_fluxes_cam[(s * Nbands_l + uint32_t(b)) * params.Ncameras + params.camera_ID];
                lobe[slot] += float(M_PI) * reaching * camera_weight * surfaceLayerLobe(layer, shading_normal, direction, to_viewer);
            }
        }
    }
}

//! Global index of the launched band with launch index b_launch
static __forceinline__ __device__ uint32_t globalBandIndex(uint32_t b_launch) {
    uint32_t b = 0;
    for (uint32_t b_global = 0; b_global < params.Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        if (b == b_launch) return b_global;
        b++;
    }
    return 0u; // unreachable: b_launch is always below the number of launched bands
}

//! |cosine| of the angle between a direction and the normal of the primitive at position pos
static __forceinline__ __device__ float primitiveNormalCos(uint32_t pos, const float3 &direction) {
    float T[16];
    loadTransformMatrix(pos, T);
    // The normal of patches and triangles is along the cross product of the first two columns of the transform
    const float3 normal = normalize(cross(make_float3(T[0], T[4], T[8]), make_float3(T[1], T[5], T[9])));
    return fabsf(dot(normal, direction));
}

//! Adds what the surface layer of a primitive reflects, of a ray's arrival on face prd->face, to that face's second outgoing component, and to every camera's (weighted by its white reference, since the layer reflects all wavelengths alike). ind_origin is the [flux slot][band] index of the primitive in the field booked.
static __forceinline__ __device__ void depositLayerReflection(const PerRayData *prd, uint32_t ind_origin, uint32_t b_global, float reflected) {
    const uint32_t layer_offset = params.Nflux_slots * params.Nbands_launch;
    atomicFloatAdd(&(prd->face ? params.scatter_buff_top : params.scatter_buff_bottom)[layer_offset + ind_origin], reflected);
    if (params.Ncameras > 0 && params.scatter_buff_top_cam && params.white_reference_cam) {
        const uint32_t Ncameras = params.Ncameras;
        float *camera_buff = prd->face ? params.scatter_buff_top_cam : params.scatter_buff_bottom_cam;
        for (uint32_t cam = 0; cam < Ncameras; cam++) {
            const float white = params.white_reference_cam[(prd->source_ID * params.Nbands_global + b_global) * Ncameras + cam];
            atomicFloatAdd(&camera_buff[(Ncameras + cam) * layer_offset + ind_origin], reflected * white);
        }
    }
}

static __forceinline__ __device__ void resolveRecordedHit(PerRayDataDual *prd, uint32_t origin_position, float t_end); // defined with __closesthit__diffuse

//! Books the direct radiation a ray delivers to the primitive it was launched from, in fields [0, field_count) but not in field 0 if skip_scene (an excluded primitive blocks the ray there).
//! The fields are arguments so that, inlined with constants for a scene without a background field, the code is that of one field.
static __forceinline__ __device__ void bookDirectOnOrigin(const PerRayData *prd, const uint32_t field_count, const bool skip_scene) {

    const uint32_t origin_position = params.primitive_positions[prd->origin_UUID];
    const uint32_t Nprims          = params.Nprimitives;
    const uint32_t Nbands_global   = params.Nbands_global;
    const uint32_t Nbands_launch   = params.Nbands_launch;
    const uint32_t slot_stride     = params.Nflux_slots * Nbands_launch; // one [flux slot][band] block of the outgoing, scattered and camera buffers

    // A translucent cover books what reaches it with its angular glass properties, not rho/tau.
    const bool  origin_is_cover = coverInAllLaunchedBands(origin_position);
    const float origin_cos      = origin_is_cover ? coverCosTheta(origin_position, optixGetWorldRayDirection()) : 1.f;

    int b = -1;
    for (uint32_t b_global = 0; b_global < Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        b++;

        // radiation_in layout: [prim * Nbands_launch + band_launch]
        const uint32_t ind_origin = origin_position * Nbands_launch + (uint32_t)b;

        // rho/tau layout: [source * Nprims * Nbands_global + prim * Nbands_global + band_global]
        const uint32_t radprop_ind = prd->source_ID * Nprims * Nbands_global
                                   + origin_position * Nbands_global
                                   + b_global;
        const float t_rho = originRho(prd, radprop_ind, b_global);
        const float t_tau = originTau(prd, radprop_ind, b_global);

        // source_fluxes layout: [source * Nbands_launch + band_launch]
        const uint32_t flux_idx  = prd->source_ID * Nbands_launch + (uint32_t)b;
        const float source_flux  = params.source_fluxes[flux_idx];

        // Attenuation from any translucent covers (glass/plastic) the ray passed through, per band.
        const float cover_tau  = (b < HELIOS_MAX_RADIATION_BANDS) ? prd->cover_transmittance[b] : 1.f;

        const double strength  = prd->strength * (double)source_flux * (double)cover_tau;

        // A dual ray books what it delivers in the background field too (see dualField()). An excluded primitive it passed blocks it
        // in the scene as it is only.
        for (uint32_t field = 0; field < field_count; field++) {
            if (field == 0u && skip_scene) continue;
            const uint32_t ind_slot = ind_origin + field * Nprims * Nbands_launch;
            if (origin_is_cover) {
                depositOnCoverOrigin(prd, origin_position, ind_origin, b_global, origin_cos, (float)strength, field);
            } else {
                // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest. The layer reflects a
                // source's direct radiation into its microfacet lobe around the mirror direction, which the rays reading the first
                // scattering of the direct radiation evaluate from the incident flux stored here (see directLayerReflection()).
                const float *layer = surfaceLayerTable(origin_position, b_global);
                const float layer_albedo = layer ? surfaceLayerAlbedo(layer, prd->origin_cos) : 0.f;
                const double body_strength = layer ? strength * (1.0 - layer_albedo) : strength;
                // Also recorded for a primitive with a normal map, whose direct radiation cameras shade per hit point with the shading normal
                if ((layer || isSmoothedPrimitive(origin_position) || normalMapOf(origin_position) != helios_nm::NO_MAP) && params.radiation_specular) {
                    atomicFloatAdd(&params.radiation_specular[directFluxIndex(prd->source_ID, origin_position, field, (uint32_t) b)], (float)strength);
                }
                // The reflected energy is also kept in the face's second outgoing component, so that the host counts it in the energy
                // the launch delivering the lobe scatters, or absorbs it in place if that launch never comes; the host clears it
                // before that launch reads the outgoing buffers (cameras use the lobe alone).
                if (layer_albedo > 0.f) {
                    atomicFloatAdd(&(prd->face ? params.scatter_buff_top : params.scatter_buff_bottom)[slot_stride + ind_slot], (float)(strength * layer_albedo));
                }

                const float absorption = (float)(body_strength * (1.0 - t_rho - t_tau));

                if (field == 0u) { // what is absorbed is that of the scene as it is
                    atomicFloatAdd(&params.radiation_in[ind_origin], absorption);
                    if (params.radiation_in_top && prd->face) {
                        atomicFloatAdd(&params.radiation_in_top[ind_origin], absorption);
                    }
                }

                if (t_rho > 0.f || t_tau > 0.f) {
                    if (prd->face) {
                        atomicFloatAdd(&params.scatter_buff_top[ind_slot],    (float)(body_strength * t_rho));
                        atomicFloatAdd(&params.scatter_buff_bottom[ind_slot], (float)(body_strength * t_tau));
                    } else {
                        atomicFloatAdd(&params.scatter_buff_bottom[ind_slot], (float)(body_strength * t_rho));
                        atomicFloatAdd(&params.scatter_buff_top[ind_slot],    (float)(body_strength * t_tau));
                    }
                }

                // Camera-weighted scatter: mirrors scatter_buff but uses rho_cam/tau_cam, accumulated once per
                // camera into that camera's [prim][band] block. This launch runs once per dispatch, not once per
                // camera, so params.camera_ID is not meaningful here; weighting by a single camera handed every
                // other camera that camera's image.
                // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
                if (params.spectral_incident && params.palette_offset[origin_position] != 0xFFFFFFFFu && body_strength != 0.0) {
                    const uint32_t face_slot = (prd->face ? 0u : params.Nsources + 1u) + prd->source_ID;
                    atomicFloatAdd(&params.spectral_incident[face_slot * slot_stride + ind_slot], (float)body_strength);
                }
                if (params.Ncameras > 0 && params.rho_cam && params.scatter_buff_top_cam) {
                    const uint32_t Ncameras   = params.Ncameras;
                    const uint32_t cam_stride = slot_stride;
                    for (uint32_t cam = 0; cam < Ncameras; cam++) {
                        const uint32_t rc_idx     = prd->source_ID * Nprims * Nbands_global * Ncameras
                                                  + origin_position * Nbands_global * Ncameras
                                                  + b_global * Ncameras + cam;
                        const float t_rho_cam = originRhoCam(prd, rc_idx, b_global, cam);
                        const float t_tau_cam = params.tau_cam ? originTauCam(prd, rc_idx, b_global, cam) : 0.f;
                        const uint32_t ind_cam = cam * cam_stride + ind_slot;
                        if ((t_rho_cam > 0.f || t_tau_cam > 0.f) && body_strength > 0.0) {
                            if (prd->face) {
                                atomicFloatAdd(&params.scatter_buff_top_cam[ind_cam],    (float)(body_strength * t_rho_cam));
                                atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_cam], (float)(body_strength * t_tau_cam));
                            } else {
                                atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_cam], (float)(body_strength * t_rho_cam));
                                atomicFloatAdd(&params.scatter_buff_top_cam[ind_cam],    (float)(body_strength * t_tau_cam));
                            }
                        }
                    }
                }
            }
        }
        if (!skip_scene) {
            accumulateWhiteReference(prd, ind_origin, b_global, (float)strength);
        }
    }
}

// A scene with a background field (see dualField()) traces its direct and diffuse rays with its own hit and miss programs (*_dual, SBT entries
// DUAL_FIELD_SBT_OFFSET further on, chosen in the raygen programs), so that the programs of a scene without one are those of one field and nothing more.

extern "C" __global__ void __miss__direct() {
    bookDirectOnOrigin(getPayloadPRD(), 1u, false);
}

//! __miss__direct with a background field: a dual ray books what it delivers in the background field too. An excluded primitive it passed blocks it in the scene as it is only.
extern "C" __global__ void __miss__direct_dual() {
    const PerRayDataDual *prd = getPayloadPRDDual();
    bookDirectOnOrigin(prd, prd->dual_ray ? 2u : 1u, prd->excluded_resolved);
}

//! Books the sky radiation a diffuse ray that leaves the scene brings to the primitive at origin_position it was launched from, in fields [field_first, field_count).
//! The fields are arguments so that, inlined with constants for a scene without a background field, the code is that of one field.
static __forceinline__ __device__ void bookSkyOnOrigin(const PerRayData *prd, const uint32_t origin_position, const uint32_t field_first, const uint32_t field_count) {

    const uint32_t Nprims        = params.Nprimitives;
    const uint32_t Nbands_global = params.Nbands_global;
    const uint32_t Nbands_launch = params.Nbands_launch;
    const uint32_t slot_stride   = params.Nflux_slots * Nbands_launch; // one [flux slot][band] block of the outgoing, scattered and camera buffers

    if (params.diffuse_flux == nullptr) {
        printf("ERROR (OptiX8 __miss__diffuse): diffuse_flux is null. "
               "launchDiffuseRays() must be given the diffuse flux of each launched band.\n");
        __trap();
    }

    const float3 ray_dir = optixGetWorldRayDirection();

    // The diffuse sky fills only the upper hemisphere unless enabled below the horizon.
    const bool sees_sky = ray_dir.z >= 0.f || params.below_horizon_diffuse_enabled != 0;

    // A translucent cover books what reaches it with its angular glass properties, not rho/tau.
    const bool  origin_is_cover = coverInAllLaunchedBands(origin_position);
    const float origin_cos      = origin_is_cover ? coverCosTheta(origin_position, ray_dir) : 1.f;

    int b = -1;
    for (uint32_t b_global = 0; b_global < Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        b++;

        // The ray brings the sky's flux less that of the band's reference blackbody field, which follows the sky fluxes in
        // diffuse_flux: emission is gathered as its departure from that field (see RadiationModel::runBand()).
        const float sky_flux       = sees_sky ? params.diffuse_flux[b] : 0.f;
        const float reference_flux = params.diffuse_flux[Nbands_launch + b];
        if (sky_flux <= 0.f && reference_flux == 0.f) continue;

        const float4 sky_p   = params.sky_radiance_params  ? params.sky_radiance_params[b]                : make_float4(0.f, 0.f, 0.f, 0.f);
        const float3 peak_d  = params.diffuse_peak_dir     ? params.diffuse_peak_dir[b]                   : make_float3(0.f, 0.f, 1.f);
        const float  power_K = params.diffuse_extinction   ? params.diffuse_extinction[b]                 : 0.f;
        const float  power_N = params.diffuse_dist_norm    ? params.diffuse_dist_norm[b]                  : 1.f;

        const float fd       = sky_flux > 0.f ? evaluateDiffuseAngularDistribution(ray_dir, peak_d, power_K, power_N, sky_p) : 0.f;
        // Attenuation from any translucent covers (glass/plastic) the sky ray passed through, per band.
        const float cover_tau    = (b < HELIOS_MAX_RADIATION_BANDS) ? prd->cover_transmittance[b] : 1.f;
        const float sky_strength = fd * sky_flux * (float)prd->strength * cover_tau;
        const float strength     = sky_strength - reference_flux * (float)prd->strength * cover_tau;

        const uint32_t ind_origin   = origin_position * Nbands_launch + (uint32_t)b;
        const uint32_t radprop_ind  = prd->source_ID * Nprims * Nbands_global
                                    + origin_position * Nbands_global + b_global;
        const float t_rho = originRho(prd, radprop_ind, b_global);
        const float t_tau = originTau(prd, radprop_ind, b_global);

        // A dual ray books the sky in the background field too (see dualField()); the scene as it is has already received an
        // excluded primitive's radiation instead if the ray passed one (see resolveRecordedHit()).
        for (uint32_t field = field_first; field < field_count; field++) {
            const uint32_t ind_slot = ind_origin + field * Nprims * Nbands_launch;
            if (origin_is_cover) {
                depositOnCoverOrigin(prd, origin_position, ind_origin, b_global, origin_cos, strength, field);
            } else {
                // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest
                const float *layer = surfaceLayerTable(origin_position, b_global);
                const float layer_albedo = layer ? surfaceLayerAlbedo(layer, prd->origin_cos) : 0.f;
                const float body_strength = layer ? strength * (1.f - layer_albedo) : strength;
                if (layer_albedo > 0.f) {
                    depositLayerReflection(prd, ind_slot, b_global, strength * layer_albedo);
                }
                recordIncidentDirection(origin_position, prd->face, (uint32_t) b, body_strength, ray_dir, field);

                if (field == 0u) { // what is absorbed is that of the scene as it is
                    atomicFloatAdd(&params.radiation_in[ind_origin], body_strength * (1.f - t_rho - t_tau));
                    if (params.radiation_in_top && prd->face) {
                        atomicFloatAdd(&params.radiation_in_top[ind_origin], body_strength * (1.f - t_rho - t_tau));
                    }
                }

                if (t_rho > 0.f || t_tau > 0.f) {
                    if (prd->face) { // top-face origin
                        atomicFloatAdd(&params.scatter_buff_top[ind_slot],    body_strength * t_rho);
                        atomicFloatAdd(&params.scatter_buff_bottom[ind_slot], body_strength * t_tau);
                    } else {         // bottom-face origin
                        atomicFloatAdd(&params.scatter_buff_bottom[ind_slot], body_strength * t_rho);
                        atomicFloatAdd(&params.scatter_buff_top[ind_slot],    body_strength * t_tau);
                    }
                }

                // Camera-weighted scatter: mirrors scatter_buff but uses rho_cam/tau_cam, accumulated once per
                // camera into that camera's [prim][band] block. This launch runs once per dispatch, not once per
                // camera, so params.camera_ID is not meaningful here; weighting by a single camera handed every
                // other camera that camera's image.
                // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
                if (params.spectral_incident && params.palette_offset[origin_position] != 0xFFFFFFFFu && body_strength != 0.0) {
                    const uint32_t face_slot = (prd->face ? 0u : params.Nsources + 1u) + prd->source_ID;
                    atomicFloatAdd(&params.spectral_incident[face_slot * slot_stride + ind_slot], (float)body_strength);
                }
                if (params.Ncameras > 0 && params.rho_cam && params.scatter_buff_top_cam) {
                    const uint32_t Ncameras   = params.Ncameras;
                    const uint32_t cam_stride = slot_stride;
                    for (uint32_t cam = 0; cam < Ncameras; cam++) {
                        const uint32_t rc_idx     = prd->source_ID * Nprims * Nbands_global * Ncameras
                                                  + origin_position * Nbands_global * Ncameras
                                                  + b_global * Ncameras + cam;
                        const float t_rho_cam = originRhoCam(prd, rc_idx, b_global, cam);
                        const float t_tau_cam = params.tau_cam ? originTauCam(prd, rc_idx, b_global, cam) : 0.f;
                        const uint32_t ind_cam = cam * cam_stride + ind_slot;
                        if ((t_rho_cam > 0.f || t_tau_cam > 0.f) && body_strength != 0.f) {
                            if (prd->face) {
                                atomicFloatAdd(&params.scatter_buff_top_cam[ind_cam],    body_strength * t_rho_cam);
                                atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_cam], body_strength * t_tau_cam);
                            } else {
                                atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_cam], body_strength * t_rho_cam);
                                atomicFloatAdd(&params.scatter_buff_top_cam[ind_cam],    body_strength * t_tau_cam);
                            }
                        }
                    }
                }
            }
        }
        if (field_first == 0u) {
            accumulateWhiteReference(prd, ind_origin, b_global, sky_strength);
        }
    }
}

extern "C" __global__ void __miss__diffuse() {
    PerRayData *prd = getPayloadPRD();

    const uint32_t origin_position = params.primitive_positions[prd->origin_UUID];
    if (origin_position == UINT_MAX) return;

    bookSkyOnOrigin(prd, origin_position, 0u, 1u);
}

//! __miss__diffuse with a background field
extern "C" __global__ void __miss__diffuse_dual() {
    PerRayDataDual *prd = getPayloadPRDDual();

    const uint32_t origin_position = params.primitive_positions[prd->origin_UUID];
    if (origin_position == UINT_MAX) return;

    // An excluded primitive the ray passed in this segment is what the scene as it is receives from, in place of the sky
    resolveRecordedHit(prd, origin_position, 3.0e38f);
    bookSkyOnOrigin(prd, origin_position, prd->excluded_resolved ? 1u : 0u, prd->dual_ray ? 2u : 1u);
}

extern "C" __global__ void __miss__camera() {
    PerRayData *prd          = getPayloadPRD();
    const uint32_t pixel_idx = prd->origin_UUID;
    const uint32_t Nbands_l  = params.Nbands_launch;
    const float3   ray_origin = optixGetWorldRayOrigin();
    const float3   ray_dir    = optixGetWorldRayDirection();

    for (uint32_t b = 0; b < Nbands_l; b++) {
        float radiance = 0.0f;

        for (uint32_t s = 0; s < params.Nsources; s++) {
            const float flux = params.source_fluxes[s * Nbands_l + b];
            if (flux <= 0.0f) continue;

            const uint32_t stype = params.source_types[s];
            if (stype == 0 || stype == 2) {
                // Collimated / sun-sphere: treat as solar disk
                if (params.solar_disk_radiance && params.solar_disk_radiance[b] > 0.0f &&
                    dot(ray_dir, params.sun_direction) >= params.solar_disk_cos_angle) {
                    radiance += params.solar_disk_radiance[b];
                }
            } else if (stype == 1) {
                // Sphere
                if (d_raySphereIntersect(ray_origin, ray_dir, params.source_positions[s],
                                         params.source_widths[s].x * 0.5f)) {
                    const float area = 4.0f * M_PI * params.source_widths[s].x * 0.5f * params.source_widths[s].x * 0.5f;
                    radiance += (flux / area) / M_PI;
                }
            } else if (stype == 3) {
                // Rectangle
                float cos_angle;
                if (d_rayRectangleIntersect(ray_origin, ray_dir, params.source_positions[s],
                                            params.source_widths[s].x, params.source_widths[s].y,
                                            params.source_rotations[s], cos_angle)) {
                    const float area = params.source_widths[s].x * params.source_widths[s].y;
                    radiance += (flux / area) * cos_angle / M_PI;
                }
            } else if (stype == 4) {
                // Disk
                float cos_angle;
                if (d_rayDiskIntersect(ray_origin, ray_dir, params.source_positions[s],
                                       params.source_widths[s].x, params.source_rotations[s], cos_angle)) {
                    const float area = M_PI * params.source_widths[s].x * params.source_widths[s].x;
                    radiance += (flux / area) * cos_angle / M_PI;
                }
            }
        }

        // The diffuse transfer limits the sky to the upper hemisphere, but the camera continues the sky below the horizon so that a finite
        // domain does not render a black strip between its edge and the horizon: a ray leaving below the horizon sees the sky radiance at the
        // horizon in its own azimuth. With the diffuse sky enabled below the horizon the camera sees the full-sphere sky pattern instead.
        float3 sky_direction = ray_dir;
        if (ray_dir.z < 0.f && params.below_horizon_diffuse_enabled == 0) {
            const float horizontal_length = sqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y);
            // A ray pointing straight down has no azimuth; any horizontal direction is equally valid for this single direction.
            sky_direction = horizontal_length > 1e-6f ? make_float3(ray_dir.x / horizontal_length, ray_dir.y / horizontal_length, 0.f)
                                                      : make_float3(1.f, 0.f, 0.f);
        }

        // Sky radiance fallback
        if (radiance <= 0.0f && params.camera_sky_radiance && params.camera_sky_radiance[b] > 0.0f) {
            const float4 sky_p = params.sky_radiance_params ? params.sky_radiance_params[b]
                                                             : make_float4(0.f, 0.f, 0.f, 0.f);
            // Prague parameters (normalization w > 0) give the sky's own pattern. Without them the camera sees the diffuse sky that
            // lights the scene, so it takes the band's angular distribution from the diffuse launches: its power law, or isotropic.
            float  power_K = 0.f;
            float  power_N = 1.f;
            float3 peak_d  = params.sun_direction;
            if (sky_p.w <= 0.f && params.diffuse_extinction) {
                power_K = params.diffuse_extinction[b];
                power_N = params.diffuse_dist_norm ? params.diffuse_dist_norm[b] : 1.f;
                peak_d  = params.diffuse_peak_dir ? params.diffuse_peak_dir[b] : peak_d;
            }
            radiance = params.camera_sky_radiance[b] *
                       evaluateDiffuseAngularDistribution(sky_direction, peak_d, power_K, power_N, sky_p);
        }

        // Isotropic sky emission/longwave: when band has emission enabled, the user-set diffuse_flux
        // represents hemispherical sky thermal/longwave irradiance. Convert to isotropic radiance.
        if (params.band_emission_flag && params.band_emission_flag[b] != 0u &&
            params.camera_diffuse_flux && params.camera_diffuse_flux[b] > 0.0f) {
            radiance += params.camera_diffuse_flux[b] / M_PI;
        }

        if (radiance > 0.0f) {
            // Seen through any translucent covers the ray passed, attenuated by their transmittance.
            const float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd->cover_transmittance[b] : 1.f;
            atomicFloatAdd(&params.radiation_in_camera[pixel_idx * Nbands_l + b],
                           radiance * (float)prd->strength * cover_tau);
        }
    }
}

//! A camera's shadow ray toward a source missed everything: the source reaches the point it was traced from (see pixelDirectRadiation())
extern "C" __global__ void __miss__camera_shadow() {
    getPayloadPRD()->source_reached = true;
}

extern "C" __global__ void __miss__pixel_label() {
    PerRayData *prd = getPayloadPRD();
    if (params.camera_pixel_depth) {
        params.camera_pixel_depth[prd->origin_UUID] = -1.0f;
    }
}

// ---------------------------------------------------------------------------
// Closest-hit: direct radiation
// ---------------------------------------------------------------------------

// Shared helper: populate PerRayData with periodic boundary wrapping info.
// Called from closesthit programs when a bbox (type-5) face is hit.
// hit_uuid identifies which bbox face was hit (bbox_local = hit_uuid - bbox_UUID_base).
// Bbox face ordering in RadiationModel.cpp:
//   x-only:  0=x-min, 1=x-max
//   y-only:  0=y-min, 1=y-max
//   xy:      0=x-min, 1=x-max, 2=y-min, 3=y-max
// Vertex 0 of each face is always the "min" corner, so bbox_vertices[b*4+0] gives
// the face's position (x or y coordinate) without floating-point tolerance issues.
static __forceinline__ __device__ float wrapIntoPeriod(float coordinate, float direction, float lower, float upper);

static __forceinline__ __device__ void handlePeriodicBoundaryHit(PerRayData *prd, uint32_t hit_uuid) {
    const float  t_hit    = __uint_as_float(optixGetAttribute_2()); // true distance to the face (the hit may be reported at t_min)
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float3 hit_pos  = make_float3(ray_orig.x + t_hit * ray_dir.x,
                                         ray_orig.y + t_hit * ray_dir.y,
                                         ray_orig.z + t_hit * ray_dir.z);

    const uint32_t bbox_local = hit_uuid - params.bbox_UUID_base;
    prd->periodic_hit = hit_pos;
    prd->periodic_hit_t = t_hit;

    // Leaving through a min face moves the ray into the next periodic copy toward -x (or -y), and through a max face toward +x (or +y)
    const uint32_t y_base = (params.periodic_flag.x == 1) ? 2u : 0u; // y-faces: if x-periodic exists they are bboxes 2,3; otherwise 0,1
    // vertex 0 of each face gives the face's position
    const float xmin = params.bbox_vertices[0 * 4].x; // x-min face, any vertex .x = xmin
    const float xmax = params.bbox_vertices[1 * 4].x; // x-max face, any vertex .x = xmax
    const float ymin = params.bbox_vertices[y_base * 4].y; // y-min face vertex 0 .y = ymin
    const float ymax = params.bbox_vertices[(y_base + 1) * 4].y; // y-max face vertex 0 .y = ymax
    int step_x = 0;
    int step_y = 0;
    if (params.periodic_flag.x == 1 && bbox_local < 2) {
        // x-faces: bbox 0 = x-min, bbox 1 = x-max
        step_x = (bbox_local == 0) ? -1 : 1;
        prd->periodic_hit.x += (bbox_local == 0) ? (xmax - xmin) : -(xmax - xmin);
    } else {
        step_y = (bbox_local == y_base) ? -1 : 1;
        prd->periodic_hit.y += (bbox_local == y_base) ? (ymax - ymin) : -(ymax - ymin);
    }

    int tile_x = prd->tile_x + step_x;
    int tile_y = prd->tile_y + step_y;
    if (params.periodic_tile_tracking == 0u) {
        // Only the copy counts, for the ray to hit the periodic copies of the primitive it was launched from
        prd->tile_x = tile_x;
        prd->tile_y = tile_y;
        return;
    }

    // Whether the ray leaves the field of copies; it can then hit nothing more
    const auto leaves_field = [&]() {
        return (params.periodic_max_repeats.x >= 0 && abs(tile_x) > params.periodic_max_repeats.x && float(tile_x) * ray_dir.x > 0.f) ||
               (params.periodic_max_repeats.y >= 0 && abs(tile_y) > params.periodic_max_repeats.y && float(tile_y) * ray_dir.y > 0.f);
    };
    // Outside the central copy, which it can no longer return to, only repeated geometry can be hit
    const bool leaves_central_copy = (tile_x != 0 && float(tile_x) * ray_dir.x >= 0.f) || (tile_y != 0 && float(tile_y) * ray_dir.y >= 0.f);
    const float3 wrapped = prd->periodic_hit;
    bool escapes = leaves_field() || (leaves_central_copy && ((wrapped.z > params.periodic_repeated_zbounds.y && ray_dir.z >= 0.f) || (wrapped.z < params.periodic_repeated_zbounds.x && ray_dir.z <= 0.f)));
    if (!escapes && leaves_central_copy && (wrapped.z > params.periodic_repeated_zbounds.y || wrapped.z < params.periodic_repeated_zbounds.x)) {
        // Above (below) the repeated geometry and heading down (up): nothing can be hit before the ray reaches it, so the ray is moved
        // there in one step, into the domain by a whole number of periods, rather than wrapped through every copy on the way
        const float target_z = wrapped.z > params.periodic_repeated_zbounds.y ? params.periodic_repeated_zbounds.y : params.periodic_repeated_zbounds.x;
        const float advance = (target_z - wrapped.z) / ray_dir.z;
        float3 moved = make_float3(wrapped.x + advance * ray_dir.x, wrapped.y + advance * ray_dir.y, target_z);
        if (params.periodic_flag.x == 1) {
            const float unwrapped = moved.x;
            moved.x = wrapIntoPeriod(moved.x, ray_dir.x, xmin, xmax);
            tile_x += __float2int_rn((unwrapped - moved.x) / (xmax - xmin));
        }
        if (params.periodic_flag.y == 1) {
            const float unwrapped = moved.y;
            moved.y = wrapIntoPeriod(moved.y, ray_dir.y, ymin, ymax);
            tile_y += __float2int_rn((unwrapped - moved.y) / (ymax - ymin));
        }
        escapes = leaves_field();
        if (!escapes) {
            prd->periodic_hit = moved;
            prd->periodic_hit_t = t_hit + advance; // the distance to the point the ray continues from
        }
    }
    if (escapes) {
        // The ray continues from just beyond the face it left through, where it hits no face again and leaves the scene, so that it
        // ends as a ray that missed everything
        prd->periodic_hit = hit_pos;
        prd->periodic_hit_t = t_hit;
        prd->periodic_hit.x += float(step_x) * params.periodic_escape_offset;
        prd->periodic_hit.y += float(step_y) * params.periodic_escape_offset;
        return;
    }
    prd->tile_x = tile_x;
    prd->tile_y = tile_y;
}

// Count a direct or diffuse ray abandoned at the periodic wrap limit (read back by the host after a runBand() call's launches)
static __forceinline__ __device__ void countAbandonedRay() {
    if (params.abandoned_ray_count) {
        atomicAdd(params.abandoned_ray_count, 1u);
    }
}

// Move a coordinate into [lower, upper) by a whole number of periods. A point on the lower face of a ray travelling out through
// that face is moved to the upper face instead, so the ray starts inside the domain. Must match wrap_into_period() in the Vulkan
// shaders.
static __forceinline__ __device__ float wrapIntoPeriod(float coordinate, float direction, float lower, float upper) {
    const float period = upper - lower;
    float wrapped = coordinate - period * floorf((coordinate - lower) / period);
    if (wrapped < lower || wrapped >= upper) {
        wrapped = lower; // rounding at a whole number of periods
    }
    if (wrapped == lower && direction < 0.f) {
        wrapped = upper;
    }
    return wrapped;
}

// Move a camera ray that starts above or below the periodic boundary faces to the top or bottom of the faces. The faces span only
// the height of the geometry, so the ray passes nothing it can hit on the way there; it is then moved into the domain laterally
// by a whole number of periods, which is where it would have been wrapped to on the way. A ray that never reaches the faces'
// height range is left where it is. Returns the distance the ray was moved along its direction, and adds the whole periods it was
// moved by laterally to the periodic copy of the domain the ray is in (prd's tile). Must match advance_to_periodic_faces() in the
// Vulkan shaders.
static __forceinline__ __device__ float advanceToPeriodicFaces(float3 &ray_origin, const float3 &ray_direction, PerRayData &prd) {
    if (params.periodic_flag.x != 1 && params.periodic_flag.y != 1) {
        return 0.f;
    }

    // Vertices 0 and 2 of every face are at the bottom and the top of the faces
    const float face_zmin = params.bbox_vertices[0].z;
    const float face_zmax = params.bbox_vertices[2].z;

    float face_z;
    if (ray_origin.z > face_zmax && ray_direction.z < 0.f) {
        face_z = face_zmax;
    } else if (ray_origin.z < face_zmin && ray_direction.z > 0.f) {
        face_z = face_zmin;
    } else {
        return 0.f;
    }

    const float t = (face_z - ray_origin.z) / ray_direction.z;
    float3 moved = make_float3(ray_origin.x + t * ray_direction.x, ray_origin.y + t * ray_direction.y, face_z);
    if (params.periodic_flag.x == 1) {
        // x-faces are bboxes 0 (x-min) and 1 (x-max)
        const float xmin = params.bbox_vertices[0 * 4].x;
        const float xmax = params.bbox_vertices[1 * 4].x;
        const float unwrapped = moved.x;
        moved.x = wrapIntoPeriod(moved.x, ray_direction.x, xmin, xmax);
        prd.tile_x += __float2int_rn((unwrapped - moved.x) / (xmax - xmin));
    }
    if (params.periodic_flag.y == 1) {
        // y-faces follow the x-faces when x is periodic
        const uint32_t y_base = (params.periodic_flag.x == 1) ? 2u : 0u;
        const float ymin = params.bbox_vertices[y_base * 4].y;
        const float ymax = params.bbox_vertices[(y_base + 1) * 4].y;
        const float unwrapped = moved.y;
        moved.y = wrapIntoPeriod(moved.y, ray_direction.y, ymin, ymax);
        prd.tile_y += __float2int_rn((unwrapped - moved.y) / (ymax - ymin));
    }
    ray_origin = moved;
    return t;
}

// ---------------------------------------------------------------------------
// Any-hit: translucent covers
// ---------------------------------------------------------------------------

// Any-hit body for translucent covers on direct rays. A direct ray passing through a cover is
// attenuated by the cover's per-band transmittance and continues toward the source; the
// intersection is ignored so traversal continues. Only whether the ray reaches the source matters,
// so the order in which covers are met does not. (Diffuse rays also need the radiance of the surface
// they end on, attenuated only by the covers in front of it, so they pass covers in __closesthit__diffuse.) What the cover itself absorbs and reflects is
// booked by the cover's own rays (depositOnCoverOrigin), not here: this ray was launched by the
// surface behind the cover, so its strength is normalized to that surface's area and angle.
// Hits on anything that is not a cover are left to the normal closest-hit (occlusion) path.
static __forceinline__ __device__ void coverAnyHitBody() {
    const uint32_t hit_uuid = optixGetAttribute_0();
    const uint32_t hit_position = params.primitive_positions[hit_uuid];
    if (hit_position == UINT_MAX) return;
    if (params.is_glass == nullptr) return; // glass model not configured this launch

    PerRayData *prd = getPayloadPRD();

    // Never let a cover attenuate a ray it launched itself.
    if (hit_uuid == prd->origin_UUID && inOriginTile(prd)) {
        optixIgnoreIntersection();
        return;
    }

    // A primitive that is not glass in every launched band is a normal (opaque) occluder: accept the hit.
    if (!coverInAllLaunchedBands(hit_position)) return;

    const float cos_theta = coverCosTheta(hit_position, optixGetWorldRayDirection());

    int b = -1;
    for (uint32_t b_global = 0; b_global < params.Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        b++;
        if (b >= HELIOS_MAX_RADIATION_BANDS) break; // host guarantees this never trips

        // Attenuate the through-beam for this band.
        prd->cover_transmittance[b] *= coverTauRhoAlpha(hit_position, b_global, cos_theta).x;
    }

    // All launched bands are glass: pass the (attenuated) ray through toward the source/sky.
    optixIgnoreIntersection();
}

// A dual ray (PerRayDataDual::dual_ray) passes the primitives excluded from the periodic boundary, which the background field does not contain, so that
// its closest hit is the background's. Whether the hit is on one is returned; the hit is then ignored, and what it means for the scene as it is is
// kept in the payload: the nearest one passed in the segment (diffuse rays, see resolveRecordedHit()), or that the ray is blocked (direct rays).
// Excluded primitives are reported in the central copy of the domain only (see periodicHidden()). Must match the Vulkan traversal
// (traversal_record_excluded in common/bvh_traversal.glsl) and passExcludedPrimitive() in primitiveIntersection.cu.
static __forceinline__ __device__ bool passExcludedPrimitive(bool blocks_direct) {
    PerRayDataDual *prd = getPayloadPRDDual();
    if (!prd->dual_ray) return false;
    const uint32_t hit_uuid = optixGetAttribute_0();
    if (params.bbox_UUID_base != 0xFFFFFFFFu && hit_uuid >= params.bbox_UUID_base) return false; // a periodic boundary face
    const uint32_t hit_position = params.primitive_positions[hit_uuid];
    if (hit_position >= params.Nprimitives || params.periodic_excluded[hit_position] == 0) return false;
    if (blocks_direct) {
        prd->excluded_resolved = true;
    } else if (optixGetRayTmax() < prd->recorded_t) { // in an any-hit program, the distance of the hit being tested
        prd->recorded_t = optixGetRayTmax();
        prd->recorded_position = hit_position;
        prd->recorded_face = optixGetAttribute_1() == 1u;
        prd->recorded_u = __uint_as_float(optixGetAttribute_2());
        prd->recorded_v = __uint_as_float(optixGetAttribute_3());
    }
    optixIgnoreIntersection();
    return true;
}

extern "C" __global__ void __anyhit__direct() {
    coverAnyHitBody();
}

//! __anyhit__direct with a background field
extern "C" __global__ void __anyhit__direct_dual() {
    if (passExcludedPrimitive(true)) return;
    coverAnyHitBody();
}

//! __anyhit__diffuse with a background field
extern "C" __global__ void __anyhit__diffuse_dual() {
    if (passExcludedPrimitive(false)) return;
    // Translucent covers are passed in __closesthit__diffuse_dual. Here a cover's own ray is only kept from hitting the cover.
    if (params.is_glass == nullptr) return;
    const PerRayData *prd = getPayloadPRD();
    if (optixGetAttribute_0() == prd->origin_UUID && inOriginTile(prd))
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__diffuse() {
    // Translucent covers are passed in __closesthit__diffuse. Here a cover's own ray is only kept from hitting the cover.
    if (params.is_glass == nullptr) return;
    const PerRayData *prd = getPayloadPRD();
    if (optixGetAttribute_0() == prd->origin_UUID && inOriginTile(prd))
        optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__direct() {
    // A direct ray hit an obstacle — it is simply blocked.
    // Exception: bbox (type-5) hits trigger periodic boundary wrapping.
    if (params.periodic_flag.x == 0 && params.periodic_flag.y == 0) return;

    const uint32_t hit_uuid = optixGetAttribute_0();
    if (params.bbox_UUID_base == 0xFFFFFFFFu || hit_uuid < params.bbox_UUID_base) return;

    PerRayData *prd = getPayloadPRD();
    prd->hit_periodic_boundary = true;
    handlePeriodicBoundaryHit(prd, hit_uuid);
}

// ---------------------------------------------------------------------------
// Closest-hit: diffuse radiation
// ---------------------------------------------------------------------------

//! Books on the primitive a diffuse ray was launched from the radiation it receives from a surface the ray reaches
/**
 * The surface is the primitive at hit_position, reached on its top (face_top) or bottom face at distance t_hit with hit attributes attr_u and attr_v (see __intersection__patch).
 * The ray books in fields [field_first, field_end) (see dualField()): the scene as it is (0) reads the surface's outgoing radiation in field seen_field, the field of the copy of the
 * domain the surface is in, and the background (1) always reads the background's. Without a background field this is the one field, 0.
 */
static __forceinline__ __device__ void gatherFromSurface(const PerRayData *prd, uint32_t origin_position, uint32_t hit_position, bool face_top, float attr_u, float attr_v, float t_hit, const uint32_t field_first,
                                                         const uint32_t field_end, const uint32_t seen_field) {
    // A translucent cover books what reaches it with its angular glass properties, not rho/tau.
    const bool  origin_is_cover = coverInAllLaunchedBands(origin_position);
    const float origin_cos      = origin_is_cover ? coverCosTheta(origin_position, optixGetWorldRayDirection()) : 1.f;

    const uint32_t Nprims        = params.Nprimitives;
    const uint32_t Nbands_global = params.Nbands_global;
    const uint32_t Nbands_launch = params.Nbands_launch;
    const uint32_t slot_stride   = params.Nflux_slots * Nbands_launch; // one [flux slot][band] block of the outgoing, scattered and camera buffers

    // The angle-dependent terms of a hit primitive's surface layer see the ray at the angle of its shading normal; a Lambertian surface sends out the same radiance at every angle, so only a
    // layered primitive with a normal map needs the shading normal here
    const uint32_t hit_map = normalMapOf(hit_position);
    const bool hit_shaded = hit_map != helios_nm::NO_MAP && params.surface_model != nullptr && params.surface_model[hit_position] != 0u;
    float3 hit_normal_top = make_float3(0.f, 0.f, 1.f);
    float3 hit_shading_top = hit_normal_top;
    if (hit_shaded) {
        hit_shading_top = hitShadingNormalTopAt(hit_position, hit_map, false, hit_normal_top, attr_u, attr_v);
    }
    // Only a primitive with a surface layer uses the angle, so the others skip reading their transform for it
    const bool hit_has_layer = params.surface_model != nullptr && params.surface_model[hit_position] != 0u;
    const float hit_cos = hit_shaded ? fabsf(dot(hit_shading_top, optixGetWorldRayDirection())) : (hit_has_layer ? primitiveNormalCos(hit_position, optixGetWorldRayDirection()) : 0.f);

    int b = -1;
    for (uint32_t b_global = 0; b_global < Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        b++;

        const uint32_t ind_origin = origin_position * Nbands_launch + (uint32_t)b;

        // Attenuated by any translucent covers the ray passed on its way to the surface.
        const double cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? (double)prd->cover_transmittance[b] : 1.0;
        // A hit primitive with a surface layer sends out its two components, in proportions that depend on the angle
        const float *hit_layer = surfaceLayerTable(hit_position, b_global);

        // What the surface sends toward the ray, in each field read
        float outgoing_of_field[2] = {0.f, 0.f};
        bool  outgoing_known[2] = {false, false};
        for (uint32_t field = field_first; field < field_end; field++) {
            const uint32_t source_field = field == 1u ? 1u : seen_field;
            if (outgoing_known[source_field]) continue;
            const float *hit_outgoing = face_top ? params.radiation_out_top : params.radiation_out_bottom;
            const uint32_t ind_hit = fluxSlot(hit_position, source_field) * Nbands_launch + (uint32_t)b;
            float outgoing = hit_layer ? outgoingLayerFlux(hit_outgoing, ind_hit, hit_layer, hit_cos) : hit_outgoing[ind_hit];
            if (hit_layer && params.specular_reflection_enabled != 0u && params.radiation_specular) {
                const float3 ray_direction = optixGetWorldRayDirection();
                const float3 to_viewer = make_float3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
                float T_hit[16];
                loadTransformMatrix(hit_position, T_hit);
                float3 face_normal = normalize(cross(make_float3(T_hit[0], T_hit[4], T_hit[8]), make_float3(T_hit[1], T_hit[5], T_hit[9])));
                if (dot(face_normal, to_viewer) < 0.f) face_normal = make_float3(-face_normal.x, -face_normal.y, -face_normal.z);
                const float3 ray_origin = optixGetWorldRayOrigin();
                const float3 point = make_float3(ray_origin.x + t_hit * ray_direction.x, ray_origin.y + t_hit * ray_direction.y, ray_origin.z + t_hit * ray_direction.z);
                if (hit_shaded) {
                    // The lobe around the shading normal, with the incident flux carried from the facet's cosine to the shading normal's
                    const float3 viewed_shading = dot(hit_normal_top, face_normal) > 0.f ? hit_shading_top : make_float3(-hit_shading_top.x, -hit_shading_top.y, -hit_shading_top.z);
                    outgoing += directLayerReflectionShaded(hit_position, (uint32_t) b, hit_layer, face_normal, viewed_shading, to_viewer, point, -1, true, source_field);
                } else {
                    outgoing += directLayerReflection(hit_position, (uint32_t) b, hit_layer, face_normal, to_viewer, point, -1, source_field);
                }
            }
            outgoing_of_field[source_field] = outgoing;
            outgoing_known[source_field] = true;
        }

        // Forward scattering estimate (see RadiationModel::runBand()): the flux this ray carries away from the origin along its
        // path, times the fraction of it the surface it reaches scatters again. A cover reflects its reflectance; what it transmits
        // is counted where the ray continues, attenuated by cover_tau. A hit surface's own reflectivity is used, not that of a
        // spectral-texture class at the hit point: the estimate need only be unbiased over the scene, not per texel. It is that of
        // the scene as it is (field 0, whose flux slots are the primitives' positions), where that field's ray ends.
        if (params.forward_scatter && field_first == 0u) {
            const float *origin_outgoing = prd->face ? params.radiation_out_top : params.radiation_out_bottom;
            const float *origin_layer = surfaceLayerTable(origin_position, b_global);
            const float launched = origin_layer ? outgoingLayerFlux(origin_outgoing, ind_origin, origin_layer, prd->origin_cos) : origin_outgoing[ind_origin];
            if (launched != 0.f) {
                float rescattered;
                if (coverInAllLaunchedBands(hit_position)) {
                    rescattered = coverTauRhoAlpha(hit_position, b_global, coverCosTheta(hit_position, optixGetWorldRayDirection())).y;
                } else {
                    const uint32_t hit_radprop_ind = prd->source_ID * Nprims * Nbands_global + hit_position * Nbands_global + b_global;
                    rescattered = params.rho[hit_radprop_ind] + params.tau[hit_radprop_ind];
                    if (hit_layer) {
                        const float hit_albedo = surfaceLayerAlbedo(hit_layer, hit_cos);
                        rescattered = hit_albedo + (1.f - hit_albedo) * rescattered;
                    }
                }
                atomicFloatAdd(&params.forward_scatter[ind_origin], (float)(fabsf(launched) * prd->strength * cover_tau * rescattered));
            }
        }

        for (uint32_t field = field_first; field < field_end; field++) {
            const uint32_t source_field = field == 1u ? 1u : seen_field;
            const uint32_t ind_slot = ind_origin + field * Nprims * Nbands_launch;
            const double strength  = outgoing_of_field[source_field] * prd->strength * cover_tau;
            if (strength == 0.0) continue;

            const uint32_t radprop_ind = prd->source_ID * Nprims * Nbands_global
                                       + origin_position * Nbands_global + b_global;
            const float t_rho = originRho(prd, radprop_ind, b_global);
            const float t_tau = originTau(prd, radprop_ind, b_global);

            if (origin_is_cover) {
                depositOnCoverOrigin(prd, origin_position, ind_origin, b_global, origin_cos, (float)strength, field);
                continue;
            }

            // A surface layer reflects layer_albedo of what arrives; the body absorbs and scatters the rest
            const float *layer = surfaceLayerTable(origin_position, b_global);
            const float layer_albedo = layer ? surfaceLayerAlbedo(layer, prd->origin_cos) : 0.f;
            const double body_strength = layer ? strength * (1.0 - layer_albedo) : strength;
            if (layer_albedo > 0.f) {
                depositLayerReflection(prd, ind_slot, b_global, (float)(strength * layer_albedo));
            }
            recordIncidentDirection(origin_position, prd->face, (uint32_t) b, (float) body_strength, optixGetWorldRayDirection(), field);

            if (field == 0u) { // what is absorbed is that of the scene as it is
                atomicFloatAdd(&params.radiation_in[ind_origin], (float)(body_strength * (1.0 - t_rho - t_tau)));
                if (params.radiation_in_top && prd->face) {
                    atomicFloatAdd(&params.radiation_in_top[ind_origin], (float)(body_strength * (1.0 - t_rho - t_tau)));
                }
            }

            if (t_rho > 0.f || t_tau > 0.f) {
                if (prd->face) { // top-face origin
                    atomicFloatAdd(&params.scatter_buff_top[ind_slot],    (float)(body_strength * t_rho));
                    atomicFloatAdd(&params.scatter_buff_bottom[ind_slot], (float)(body_strength * t_tau));
                } else {         // bottom-face origin
                    atomicFloatAdd(&params.scatter_buff_bottom[ind_slot], (float)(body_strength * t_rho));
                    atomicFloatAdd(&params.scatter_buff_top[ind_slot],    (float)(body_strength * t_tau));
                }
            }

            // Camera-weighted scatter: mirrors scatter_buff but uses rho_cam/tau_cam, accumulated once per
            // camera into that camera's [prim][band] block. This launch runs once per dispatch, not once per
            // camera, so params.camera_ID is not meaningful here; weighting by a single camera handed every
            // other camera that camera's image.
            // What reaches a primitive with a spectral texture, from which each camera computes the outgoing radiation of its classes
            if (params.spectral_incident && params.palette_offset[origin_position] != 0xFFFFFFFFu && body_strength != 0.0) {
                const uint32_t face_slot = (prd->face ? 0u : params.Nsources + 1u) + prd->source_ID;
                atomicFloatAdd(&params.spectral_incident[face_slot * slot_stride + ind_slot], (float)body_strength);
            }
            if (params.Ncameras > 0 && params.rho_cam && params.scatter_buff_top_cam) {
                const uint32_t Ncameras   = params.Ncameras;
                const uint32_t cam_stride = slot_stride;
                for (uint32_t cam = 0; cam < Ncameras; cam++) {
                    const uint32_t rc_idx     = prd->source_ID * Nprims * Nbands_global * Ncameras
                                              + origin_position * Nbands_global * Ncameras
                                              + b_global * Ncameras + cam;
                    const float t_rho_cam = originRhoCam(prd, rc_idx, b_global, cam);
                    const float t_tau_cam = params.tau_cam ? originTauCam(prd, rc_idx, b_global, cam) : 0.f;
                    const uint32_t ind_cam = cam * cam_stride + ind_slot;
                    // The gathered flux is negative where a surface radiates less than the reference field (see RadiationModel::runBand())
                    if ((t_rho_cam > 0.f || t_tau_cam > 0.f) && body_strength != 0.0) {
                        if (prd->face) {
                            atomicFloatAdd(&params.scatter_buff_top_cam[ind_cam],    (float)(body_strength * t_rho_cam));
                            atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_cam], (float)(body_strength * t_tau_cam));
                        } else {
                            atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_cam], (float)(body_strength * t_rho_cam));
                            atomicFloatAdd(&params.scatter_buff_top_cam[ind_cam],    (float)(body_strength * t_tau_cam));
                        }
                    }
                }
            }
    
        }
    }
}

//! Settles what the scene as it is receives along a dual diffuse ray that passed an excluded primitive nearer than t_end, where its current segment ends
/**
 * A dual ray (PerRayDataDual::dual_ray) passes the primitives excluded from the periodic boundary, so that it reaches what the background field's ray reaches. In the scene as it
 * is the ray ends on the nearest excluded primitive it passed, whose radiation that field receives here, with the transmittance of the covers crossed so far. It must be
 * called before anything else is booked for the segment or a cover's transmittance is applied.
 */
static __forceinline__ __device__ void resolveRecordedHit(PerRayDataDual *prd, uint32_t origin_position, float t_end) {
    if (!prd->dual_ray || prd->excluded_resolved || !(prd->recorded_t < t_end)) {
        return;
    }
    gatherFromSurface(prd, origin_position, prd->recorded_position, prd->recorded_face, prd->recorded_u, prd->recorded_v, prd->recorded_t, 0u, 1u, 0u);
    prd->excluded_resolved = true;
}

//! The diffuse closest-hit program, for a scene with a background field (dual) or without; inlined with a constant, each is its own code
static __forceinline__ __device__ void closestHitDiffuse(const bool dual) {
    const uint32_t hit_uuid  = optixGetAttribute_0();
    const bool     face_top  = (optixGetAttribute_1() == 1u);

    PerRayData *prd = getPayloadPRD();
    const uint32_t origin_position = params.primitive_positions[prd->origin_UUID];

    // Periodic boundary: bbox (type-5) hit — wrap ray and return without depositing energy
    if ((params.periodic_flag.x == 1 || params.periodic_flag.y == 1) &&
        params.bbox_UUID_base != 0xFFFFFFFFu && hit_uuid >= params.bbox_UUID_base) {
        if (dual && origin_position != UINT_MAX) {
            // The face is reported no nearer than the ray's minimum distance; attribute 2 holds its true distance
            resolveRecordedHit(getPayloadPRDDual(), origin_position, __uint_as_float(optixGetAttribute_2()));
        }
        prd->hit_periodic_boundary = true;
        handlePeriodicBoundaryHit(prd, hit_uuid);
        return;
    }

    const uint32_t hit_position    = params.primitive_positions[hit_uuid];
    if (origin_position == UINT_MAX || hit_position == UINT_MAX) return;

    if (!dual) {
        gatherFromSurface(prd, origin_position, hit_position, face_top, __uint_as_float(optixGetAttribute_2()), __uint_as_float(optixGetAttribute_3()), optixGetRayTmax(), 0u, 1u, 0u);
    } else {
        PerRayDataDual *dual_prd = getPayloadPRDDual();
        resolveRecordedHit(dual_prd, origin_position, optixGetRayTmax());
        gatherFromSurface(prd, origin_position, hit_position, face_top, __uint_as_float(optixGetAttribute_2()), __uint_as_float(optixGetAttribute_3()), optixGetRayTmax(), dual_prd->excluded_resolved ? 1u : 0u,
                          dual_prd->dual_ray ? 2u : 1u, fieldSeenBy(prd));
    }

    // A translucent cover: its own outgoing radiance (reflection and emission) was received above like that of any
    // surface; continue past it to what lies behind, attenuated by its transmittance.
    if (coverInAllLaunchedBands(hit_position)) {
        continuePastCover(prd, hit_position, optixGetRayTmax());
    }
}

extern "C" __global__ void __closesthit__diffuse() {
    closestHitDiffuse(false);
}

//! __closesthit__diffuse with a background field
extern "C" __global__ void __closesthit__diffuse_dual() {
    closestHitDiffuse(true);
}

// ---------------------------------------------------------------------------
// Closest-hit: camera
// ---------------------------------------------------------------------------

extern "C" __global__ void __closesthit__camera() {
    const uint32_t hit_uuid     = optixGetAttribute_0();
    PerRayData    *prd          = getPayloadPRD();
    const uint32_t hit_position = params.primitive_positions[hit_uuid];

    if (hit_position == 0xFFFFFFFFu) return; // invalid primitive

    // Periodic boundary: treat as transparent wall and re-launch
    if ((params.periodic_flag.x != 0.f || params.periodic_flag.y != 0.f) &&
        params.primitive_type[hit_position] == 5) {
        handlePeriodicBoundaryHit(prd, hit_uuid);
        prd->hit_periodic_boundary = true;
        return;
    }

    const uint32_t pixel_index   = prd->origin_UUID;
    const uint32_t Nbands_l      = params.Nbands_launch;
    const uint32_t Nprims        = params.Nprimitives;
    // With a background radiation field (see dualField()), the camera sees the scene as it is in the central copy of the domain, where the primitives excluded from the periodic
    // boundary stand, and the background in every other copy: the field's flux slot of the primitive, and its vertex values after those of the scene as it is
    const uint32_t field         = fieldSeenBy(prd);
    const uint32_t flux_position = fluxSlot(hit_position, field);
    const size_t   vertex_field_offset = size_t(field) * params.vertex_field_stride * Nbands_l;
    const float    t_hit         = optixGetRayTmax();
    const float3   ray_origin    = optixGetWorldRayOrigin();
    const float3   ray_direction = optixGetWorldRayDirection();

    // Use face attribute from intersection program (correct for both patches and triangles)
    const bool   face_top  = (optixGetAttribute_1() == 1u);

    // Surface normal of the face the camera sees, for the surface layer's angle-dependent reflection
    float T[16];
    loadTransformMatrix(hit_position, T);
    float3 n0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, n0);
    float3 n1 = make_float3(1.f, 0.f, 0.f); d_transformPoint(T, n1);
    float3 n2 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, n2);
    float3 normal = normalize(cross(n1 - n0, n2 - n0));
    // The normal of the face the camera sees
    if (dot(normal, ray_direction) > 0.f) {
        normal = make_float3(-normal.x, -normal.y, -normal.z);
    }

    // Camera flux smoothing: blend the outgoing flux across the facet from the values carried by the vertices it shares with its neighbours, so that a coarsely tessellated curved surface does not render as
    // flat panels with a step at every facet edge. The weights come from where on the facet the ray landed, which the intersection program passed through in attributes 2 and 3.
    int32_t smooth_v0 = -1, smooth_v1 = -1, smooth_v2 = -1, smooth_v3 = -1;
    float   smooth_w0 = 0.f, smooth_w1 = 0.f, smooth_w2 = 0.f, smooth_w3 = 0.f;

    if (params.smoothing_vertex_indices != nullptr && params.vertex_radiation_out_top != nullptr) {
        smooth_v0 = params.smoothing_vertex_indices[hit_position * 4 + 0];
        if (smooth_v0 >= 0) {
            smooth_v1 = params.smoothing_vertex_indices[hit_position * 4 + 1];
            smooth_v2 = params.smoothing_vertex_indices[hit_position * 4 + 2];
            smooth_v3 = params.smoothing_vertex_indices[hit_position * 4 + 3];

            const float attr_u = __uint_as_float(optixGetAttribute_2());
            const float attr_v = __uint_as_float(optixGetAttribute_3());

            if (smooth_v3 < 0) {
                // Triangle: barycentric weights, with attributes holding the weights of the second and third corners.
                smooth_w1 = attr_u;
                smooth_w2 = attr_v;
                smooth_w0 = 1.f - smooth_w1 - smooth_w2;
            } else {
                // Patch: bilinear weights over the patch-local coordinates, whose corners run (0,0), (1,0), (1,1), (0,1).
                smooth_w0 = (1.f - attr_u) * (1.f - attr_v);
                smooth_w1 = attr_u * (1.f - attr_v);
                smooth_w2 = attr_u * attr_v;
                smooth_w3 = (1.f - attr_u) * attr_v;
            }
        }
    }

    // With camera flux smoothing and a specular surface layer: a shading normal interpolated from the facet's vertices, for the layer's angle-dependent terms. Each vertex normal is first turned to the
    // side of the facet, so that the result does not depend on how the facet is wound. The facet's own normal is kept in the rare case where the vertex normals interpolate to nothing on its side.
    const bool smooth_shading = smooth_v0 >= 0 && params.vertex_normals != nullptr;
    float3 shading_normal = normal;
    if (smooth_shading) {
        const int32_t corners[4] = {smooth_v0, smooth_v1, smooth_v2, smooth_v3};
        const float weights[4] = {smooth_w0, smooth_w1, smooth_w2, smooth_w3};
        float3 interpolated = make_float3(0.f, 0.f, 0.f);
        for (int c = 0; c < 4; c++) {
            if (corners[c] < 0)
                continue;
            float3 vertex_normal = make_float3(params.vertex_normals[3 * corners[c] + 0], params.vertex_normals[3 * corners[c] + 1], params.vertex_normals[3 * corners[c] + 2]);
            if (dot(vertex_normal, normal) < 0.f) {
                vertex_normal = make_float3(-vertex_normal.x, -vertex_normal.y, -vertex_normal.z);
            }
            interpolated = make_float3(interpolated.x + weights[c] * vertex_normal.x, interpolated.y + weights[c] * vertex_normal.y, interpolated.z + weights[c] * vertex_normal.z);
        }
        if (dot(interpolated, normal) > 1e-6f) {
            shading_normal = normalize(interpolated);
        }
    }
    // A normal map tilts the shading normal (the facet's, or the smoothed one) by the texel at the hit point, bilinearly interpolated so that a magnified map does not show its texels
    const uint32_t hit_map = normalMapOf(hit_position);
    const bool normal_mapped = hit_map != helios_nm::NO_MAP;
    // The normal the map tilts (the facet's or the smoothed one), on the viewed side
    const float3 base_viewed = shading_normal;
    if (normal_mapped) {
        const uint32_t hit_type = params.primitive_type[hit_position];
        const float3 base_top = face_top ? shading_normal : make_float3(-shading_normal.x, -shading_normal.y, -shading_normal.z);
        const float3 sp = canonicalPointFromHit(hit_type, __uint_as_float(optixGetAttribute_2()), __uint_as_float(optixGetAttribute_3()));
        const float3 m_top = normalMapShadingNormalAt(hit_position, hit_type, hit_map, T, base_top, sp, true);
        shading_normal = face_top ? m_top : make_float3(-m_top.x, -m_top.y, -m_top.z);
    }
    // The angle-dependent terms (the direct radiation per hit point, a surface layer's viewing angle and lobe) use the shading normal when there is one
    const bool use_shading_normal = smooth_shading || normal_mapped;
    // A normal-mapped primitive's direct radiation is shaded per hit point when the host removed what the direct launch deposited from the outgoing radiation the camera reads. With primitives excluded
    // from the periodic boundary, the host removed it for every primitive but a translucent cover, and it is evaluated per pixel sample with a shadow ray, so that the shadow of an object that does not
    // repeat falls only where that object puts it, not on every periodic copy of the primitives it shades.
    const bool hit_is_cover = coverInAllLaunchedBands(hit_position);
    const bool direct_all = params.camera_direct_all != 0u && !hit_is_cover;
    const bool shade_direct_per_hit = direct_all || (normal_mapped ? params.normal_map_direct != 0u : smooth_shading);
    const int32_t smooth_corners[4] = {smooth_v0, smooth_v1, smooth_v2, smooth_v3};
    const float smooth_weights[4] = {smooth_w0, smooth_w1, smooth_w2, smooth_w3};

    const float3 hit_point = make_float3(ray_origin.x + t_hit * ray_direction.x, ray_origin.y + t_hit * ray_direction.y, ray_origin.z + t_hit * ray_direction.z);

    // A primitive with a spectral texture is seen in the outgoing radiation of the class at the hit point, from the hit attributes:
    // the patch-local coordinates plus 0.5, or the barycentric weights of a triangle's second and third corners
    uint32_t hit_class_slot = 0xFFFFFFFFu;
    uint32_t hit_entry = 0xFFFFFFFFu;
    if (params.class_flux != nullptr && params.class_flux_offset[flux_position] != 0xFFFFFFFFu) {
        const uint32_t hit_type = params.primitive_type[hit_position];
        const float attr_u = __uint_as_float(optixGetAttribute_2());
        const float attr_v = __uint_as_float(optixGetAttribute_3());
        const float3 sp = hit_type == 1 ? make_float3(attr_v, attr_u + attr_v, 0.f) : make_float3(attr_u - 0.5f, attr_v - 0.5f, 0.f);
        const uint32_t entry = paletteEntryAt(hit_position, originTextureUV(hit_position, hit_type, sp));
        if (entry != 0xFFFFFFFFu) {
            hit_class_slot = params.class_flux_offset[flux_position] + (entry - params.palette_offset[hit_position]);
            hit_entry = entry;
        }
    }

    // With smoothing, the sources' direct radiation is evaluated for this pixel sample, with a shadow ray toward each source, so that a cast shadow falls where the geometry puts it within the facet (see
    // pixelDirectRadiation()). A translucent cover's own direct radiation stays in the outgoing radiation its vertices carry.
    // It is evaluated for up to HELIOS_MAX_RADIATION_BANDS bands at a time; every group of bands draws the same points on the sources, so all bands see the same shadows.
    const bool per_pixel_direct = params.camera_direct_per_pixel != 0u && shade_direct_per_hit && (smooth_shading || direct_all) && params.Nsources > 0 && !hit_is_cover;
    float pixel_body[HELIOS_MAX_RADIATION_BANDS];
    float pixel_class[HELIOS_MAX_RADIATION_BANDS];
    float pixel_lobe[HELIOS_MAX_RADIATION_BANDS];
    uint32_t pixel_band_begin = 0;
    const uint32_t seed_at_hit = prd->seed;

    for (uint32_t b = 0; b < Nbands_l; b++) {
        if (per_pixel_direct && b % HELIOS_MAX_RADIATION_BANDS == 0) {
            const float3 to_viewer = make_float3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
            uint32_t band_group_seed = seed_at_hit;
            pixelDirectRadiation(hit_uuid, hit_position, params.primitive_type[hit_position], T, normal, base_viewed, shading_normal, normal_mapped, face_top, hit_point, t_hit, to_viewer, smooth_corners, smooth_weights, hit_entry, b, band_group_seed,
                                 pixel_body, pixel_class, pixel_lobe);
            prd->seed = band_group_seed;
            pixel_band_begin = b;
        }
        // Radiance from hit surface (outgoing flux / pi = radiance)
        const uint32_t ind_hit = flux_position * Nbands_l + b;
        float outgoing_flux;
        if (smooth_v0 >= 0) {
            const float *vertex_flux = (face_top ? params.vertex_radiation_out_top : params.vertex_radiation_out_bottom) + vertex_field_offset;
            outgoing_flux = smooth_w0 * vertex_flux[uint32_t(smooth_v0) * Nbands_l + b] + smooth_w1 * vertex_flux[uint32_t(smooth_v1) * Nbands_l + b] + smooth_w2 * vertex_flux[uint32_t(smooth_v2) * Nbands_l + b];
            if (smooth_v3 >= 0) {
                outgoing_flux += smooth_w3 * vertex_flux[uint32_t(smooth_v3) * Nbands_l + b];
            }
        } else {
            outgoing_flux = face_top ? params.radiation_out_top[ind_hit] : params.radiation_out_bottom[ind_hit];
        }
        const bool class_applies = hit_class_slot != 0xFFFFFFFFu && params.palette_band_active[globalBandIndex(b)] != 0u;
        // With smoothing, the vertices carry the radiation other than the sources' direct radiation, which is evaluated here for this pixel sample. A normal-mapped primitive that is not smoothed
        // carries its facet's direct radiation over to the shading normal.
        if (per_pixel_direct && !class_applies) {
            outgoing_flux += pixel_body[b - pixel_band_begin];
        } else if (shade_direct_per_hit && !smooth_shading && params.radiation_specular != nullptr && !class_applies) {
            outgoing_flux += shadedDirectBodyFlux(hit_position, b, globalBandIndex(b), surfaceLayerTable(hit_position, globalBandIndex(b)), normal, shading_normal, hit_point, normal_mapped, field);
        }
        // The class's own outgoing radiation: flat over the facet, or with camera flux smoothing, the class's camera-weighted reflectivity and
        // transmissivity applied to the radiation reaching the two faces, and the emitted radiation, interpolated from the facet's vertices
        if (class_applies) {
            if (smooth_v0 >= 0 && params.vertex_spectral != nullptr) {
                const uint32_t layers = params.vertex_spectral_layers;
                const uint32_t vertex_count = params.vertex_spectral_count;
                auto vertexValue = [&](uint32_t face, uint32_t layer) {
                    const float *values = params.vertex_spectral + size_t(face * layers + layer) * vertex_count * Nbands_l + vertex_field_offset;
                    float value = smooth_w0 * values[uint32_t(smooth_v0) * Nbands_l + b] + smooth_w1 * values[uint32_t(smooth_v1) * Nbands_l + b] + smooth_w2 * values[uint32_t(smooth_v2) * Nbands_l + b];
                    if (smooth_v3 >= 0) {
                        value += smooth_w3 * values[uint32_t(smooth_v3) * Nbands_l + b];
                    }
                    return value;
                };
                const uint32_t same_face = face_top ? 0u : 1u;
                const uint32_t b_global = globalBandIndex(b);
                const uint32_t camera = params.vertex_spectral_camera;
                // The vertices carry the radiation from the sky and other surfaces (layer 0, the last source slot) and the emitted radiation (layer 1). The direct radiation of the sources (slots below
                // Nsources) is evaluated for this pixel sample, through the class's spectra (see pixelDirectRadiation()).
                outgoing_flux = vertexValue(same_face, 1); // emitted
                const size_t ind_cam = ((size_t(params.Nsources) * params.Npalette_entries + hit_entry) * params.Nbands_global + b_global) * params.Ncameras + camera;
                outgoing_flux += params.palette_rho_cam[ind_cam] * vertexValue(same_face, 0) + params.palette_tau_cam[ind_cam] * vertexValue(1u - same_face, 0);
                if (per_pixel_direct) {
                    outgoing_flux += pixel_class[b - pixel_band_begin];
                }
            } else {
                outgoing_flux = params.class_flux[((face_top ? 0u : params.class_flux_slots) + hit_class_slot) * Nbands_l + b];
                // When the host left the sources' direct radiation out of the class flux, it is evaluated for this pixel sample, or with a normal map shaded here with the shading normal
                if (per_pixel_direct) {
                    outgoing_flux += pixel_class[b - pixel_band_begin];
                } else if (normal_mapped && params.normal_map_direct != 0u && params.radiation_specular != nullptr) {
                    const uint32_t b_global = globalBandIndex(b);
                    const float3 back_normal = make_float3(-normal.x, -normal.y, -normal.z);
                    const float3 back_shading_normal = make_float3(-shading_normal.x, -shading_normal.y, -shading_normal.z);
                    // A surface layer keeps 1 - a(mu) of the radiation reaching each face in the body, as in shadedDirectBodyFlux()
                    const float *class_layer = surfaceLayerTable(hit_position, b_global);
                    for (uint32_t slot = 0; slot < params.Nsources; slot++) {
                        const size_t ind_cam = ((size_t(slot) * params.Npalette_entries + hit_entry) * params.Nbands_global + b_global) * params.Ncameras + params.camera_ID;
                        const float3 to_source = directionToSource(slot, hit_point);
                        const float incident = params.radiation_specular[directFluxIndex(slot, hit_position, field, b)];
                        const bool viewed_face_toward_source = dot(normal, to_source) > 0.f;
                        const float same_value = faceDirect(viewed_face_toward_source ? incident : 0.f, to_source, normal, shading_normal, true);
                        const float other_value = faceDirect(viewed_face_toward_source ? 0.f : incident, to_source, back_normal, back_shading_normal, true);
                        const float same_body = class_layer ? 1.f - surfaceLayerAlbedo(class_layer, fmaxf(0.f, dot(shading_normal, to_source))) : 1.f;
                        const float other_body = class_layer ? 1.f - surfaceLayerAlbedo(class_layer, fmaxf(0.f, dot(back_shading_normal, to_source))) : 1.f;
                        outgoing_flux += params.palette_rho_cam[ind_cam] * same_value * same_body + params.palette_tau_cam[ind_cam] * other_value * other_body;
                    }
                }
            }
        }
        // A normal map also shades the scattered light (from the sky and other surfaces), by the direction it came from
        if (normal_mapped && params.normal_map_moments != nullptr) {
            const uint32_t b_global = globalBandIndex(b);
            float rho_d, tau_d;
            if (class_applies) {
                const size_t ind_cam = ((size_t(params.Nsources) * params.Npalette_entries + hit_entry) * params.Nbands_global + b_global) * params.Ncameras + params.camera_ID;
                rho_d = params.palette_rho_cam[ind_cam];
                tau_d = params.palette_tau_cam[ind_cam];
            } else {
                const size_t material = ((size_t(params.Nsources) * Nprims + hit_position) * params.Nbands_global + b_global) * params.Ncameras + params.camera_ID;
                rho_d = params.rho_cam[material];
                tau_d = params.tau_cam ? params.tau_cam[material] : 0.f;
            }
            const float relief = normalMapDiffuseRelief(hit_position, face_top, b, shading_normal, base_viewed, normalize(normal), rho_d, tau_d, smooth_corners, smooth_weights, field);
            outgoing_flux = outgoing_flux >= 0.f ? fmaxf(0.f, outgoing_flux + relief) : outgoing_flux + relief;
        }
        // A surface with a layer sends out its two components, in proportions that depend on the viewing angle. With smoothing, the
        // second component is interpolated from the facet's vertices like the first, and the viewing angle is taken from the shading normal.
        if (const float *layer = surfaceLayerTable(hit_position, globalBandIndex(b))) {
            const uint32_t layer_offset = params.Nflux_slots * Nbands_l;
            float mu_view;
            float layer_flux;
            if (normal_mapped) {
                // A texel of a map tilted away from the viewer is seen at |m.d|, as the rays the primitive sends out are folded into its hemisphere
                mu_view = fminf(1.f, fabsf(dot(shading_normal, ray_direction)));
            } else if (use_shading_normal) {
                // A shading normal can tip away from the viewer at a silhouette, where the layer is seen edge-on
                mu_view = fminf(1.f, fmaxf(0.f, -dot(shading_normal, ray_direction)));
            } else {
                mu_view = fabsf(dot(normal, ray_direction));
            }
            if (smooth_shading) {
                // The second component follows the first, which holds every field's vertices
                const float *vertex_layer_flux = (face_top ? params.vertex_radiation_out_top : params.vertex_radiation_out_bottom) + size_t(params.vertex_normal_count) * (dualField() ? 2u : 1u) * Nbands_l + vertex_field_offset;
                layer_flux = smooth_w0 * vertex_layer_flux[uint32_t(smooth_v0) * Nbands_l + b] + smooth_w1 * vertex_layer_flux[uint32_t(smooth_v1) * Nbands_l + b] + smooth_w2 * vertex_layer_flux[uint32_t(smooth_v2) * Nbands_l + b];
                if (smooth_v3 >= 0) {
                    layer_flux += smooth_w3 * vertex_layer_flux[uint32_t(smooth_v3) * Nbands_l + b];
                }
            } else {
                layer_flux = (face_top ? params.radiation_out_top : params.radiation_out_bottom)[layer_offset + ind_hit];
            }
            const float a = surfaceLayerAlbedo(layer, mu_view);
            const float abar = layer[params.surface_layer_table_size];
            outgoing_flux = outgoing_flux * (1.f - a) / (1.f - abar) + layer_flux * a / abar;
        }
        float strength = (float) prd->strength * outgoing_flux;

        // Radiance of sources between the ray origin and the hit point. It is already a radiance, so unlike the
        // surface's outgoing flux it is not divided by pi below.
        float source_strength = 0.0f;

        // Check sources visible between camera origin and hit point
        for (uint32_t s = 0; s < params.Nsources; s++) {
            const float flux = params.source_fluxes[s * Nbands_l + b];
            if (flux <= 0.0f) continue;

            const uint32_t stype = params.source_types[s];
            float source_radiance = 0.0f;

            if (stype == 1) {
                // Sphere
                const float radius = params.source_widths[s].x * 0.5f;
                const float3 oc = make_float3(ray_origin.x - params.source_positions[s].x,
                                               ray_origin.y - params.source_positions[s].y,
                                               ray_origin.z - params.source_positions[s].z);
                const float bd = dot(oc, ray_direction);
                const float cd = dot(oc, oc) - radius * radius;
                const float disc = bd * bd - cd;
                if (disc >= 0.0f) {
                    const float t_sphere = -bd - sqrtf(disc);
                    if (t_sphere > 0.0f && t_sphere < t_hit) {
                        const float area = 4.0f * M_PI * radius * radius;
                        source_radiance = (flux / area) / M_PI;
                    }
                }
            } else if (stype == 3) {
                // Rectangle
                float trans[16];
                d_makeTransformMatrix(params.source_rotations[s], trans);
                const float3 snormal = make_float3(trans[2], trans[6], trans[10]);
                const float denom = dot(ray_direction, snormal);
                if (denom < -1e-6f) {
                    const float3 oc = make_float3(params.source_positions[s].x - ray_origin.x,
                                                   params.source_positions[s].y - ray_origin.y,
                                                   params.source_positions[s].z - ray_origin.z);
                    const float t_r = dot(oc, snormal) / denom;
                    if (t_r > 0.0f && t_r < t_hit) {
                        float3 hp = make_float3(ray_origin.x + t_r * ray_direction.x - params.source_positions[s].x,
                                                ray_origin.y + t_r * ray_direction.y - params.source_positions[s].y,
                                                ray_origin.z + t_r * ray_direction.z - params.source_positions[s].z);
                        float inv_t[16];
                        d_invertMatrix(trans, inv_t);
                        d_transformPoint(inv_t, hp);
                        if (fabsf(hp.x) <= params.source_widths[s].x * 0.5f &&
                            fabsf(hp.y) <= params.source_widths[s].y * 0.5f) {
                            const float area = params.source_widths[s].x * params.source_widths[s].y;
                            source_radiance = (flux / area) * (-denom) / M_PI;
                        }
                    }
                }
            } else if (stype == 4) {
                // Disk
                float trans[16];
                d_makeTransformMatrix(params.source_rotations[s], trans);
                const float3 snormal = make_float3(trans[2], trans[6], trans[10]);
                const float denom = dot(ray_direction, snormal);
                if (denom < -1e-6f) {
                    const float3 oc = make_float3(params.source_positions[s].x - ray_origin.x,
                                                   params.source_positions[s].y - ray_origin.y,
                                                   params.source_positions[s].z - ray_origin.z);
                    const float t_d = dot(oc, snormal) / denom;
                    if (t_d > 0.0f && t_d < t_hit) {
                        const float3 hp = make_float3(ray_origin.x + t_d * ray_direction.x - params.source_positions[s].x,
                                                       ray_origin.y + t_d * ray_direction.y - params.source_positions[s].y,
                                                       ray_origin.z + t_d * ray_direction.z - params.source_positions[s].z);
                        const float radius = params.source_widths[s].x;
                        if (dot(hp, hp) <= radius * radius) {
                            const float area = M_PI * radius * radius;
                            source_radiance = (flux / area) * (-denom) / M_PI;
                        }
                    }
                }
            }

            if (source_radiance > 0.0f) {
                source_strength += source_radiance * (float)prd->strength;
            }
        }

        // Direct radiation reflected into the camera by a surface layer's microfacet lobe
        float strength_spec = 0.0f;
        if (params.specular_reflection_enabled != 0u && params.radiation_specular) {
            if (const float *layer = surfaceLayerTable(hit_position, globalBandIndex(b))) {
                const float3 point = make_float3(ray_origin.x + t_hit * ray_direction.x, ray_origin.y + t_hit * ray_direction.y, ray_origin.z + t_hit * ray_direction.z);
                const float3 to_viewer = make_float3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
                if (per_pixel_direct) {
                    strength_spec = pixel_lobe[b - pixel_band_begin] * (float) prd->strength;
                } else if (use_shading_normal) {
                    strength_spec = directLayerReflectionShaded(hit_position, b, layer, normal, shading_normal, to_viewer, point, int(params.camera_ID), normal_mapped, field) * (float) prd->strength;
                } else {
                    strength_spec = directLayerReflection(hit_position, b, layer, normal, to_viewer, point, int(params.camera_ID), field) * (float) prd->strength;
                }
            }
        }

        // Accumulate into camera radiation buffer: [pixel][band], attenuated by any translucent covers the
        // ray passed in front of this surface.
        const float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd->cover_transmittance[b] : 1.f;
        atomicFloatAdd(&params.radiation_in_camera[pixel_index * Nbands_l + b],
                       cover_tau * ((strength + strength_spec) / M_PI + source_strength));
    }

    // A translucent cover adds its own outgoing radiance (above) and lets the ray continue to what lies behind it.
    if (coverInAllLaunchedBands(hit_position)) {
        continuePastCover(prd, hit_position, t_hit);
    }
}

// ---------------------------------------------------------------------------
// Closest-hit: pixel label
// ---------------------------------------------------------------------------

extern "C" __global__ void __closesthit__pixel_label() {
    const uint32_t hit_uuid     = optixGetAttribute_0();
    PerRayData    *prd          = getPayloadPRD();
    const uint32_t origin_UUID  = prd->origin_UUID;
    const uint32_t hit_position = params.primitive_positions[hit_uuid];

    // Periodic boundary: treat as transparent wall and re-launch
    if ((params.periodic_flag.x != 0.f || params.periodic_flag.y != 0.f) &&
        hit_position != 0xFFFFFFFFu && params.primitive_type[hit_position] == 5) {
        handlePeriodicBoundaryHit(prd, hit_uuid);
        prd->strength += (double)prd->periodic_hit_t; // the distance travelled to the boundary counts toward the depth
        prd->hit_periodic_boundary = true;
        return;
    }

    // Label the first surface behind any translucent covers: pass through the cover, carrying the distance
    // travelled so far in prd->strength.
    if (hit_position != 0xFFFFFFFFu && coverInAllLaunchedBands(hit_position)) {
        prd->strength += (double)(optixGetRayTmax() + COVER_EXIT_OFFSET);
        continuePastCover(prd, hit_position, optixGetRayTmax());
        return;
    }

    // Store UUID+1 (0 is reserved for sky/miss)
    if (params.camera_pixel_label) {
        params.camera_pixel_label[origin_UUID] = hit_uuid + 1u;
    }

    // The class of the spectral texture at the hit point, from the hit attributes as in __closesthit__camera
    if (params.camera_pixel_class && hit_position != 0xFFFFFFFFu && params.palette_offset[hit_position] != 0xFFFFFFFFu) {
        const uint32_t hit_type = params.primitive_type[hit_position];
        const float attr_u = __uint_as_float(optixGetAttribute_2());
        const float attr_v = __uint_as_float(optixGetAttribute_3());
        const float3 sp = hit_type == 1 ? make_float3(attr_v, attr_u + attr_v, 0.f) : make_float3(attr_u - 0.5f, attr_v - 0.5f, 0.f);
        const uint32_t entry = paletteEntryAt(hit_position, originTextureUV(hit_position, hit_type, sp));
        params.camera_pixel_class[origin_UUID] = entry == 0xFFFFFFFFu ? 0u : entry - params.palette_offset[hit_position] + 1u;
    }

    // Depth: project ray parameter along camera view direction
    if (params.camera_pixel_depth) {
        const float  t_hit    = optixGetRayTmax() + (float)prd->strength; // strength = distance travelled before any translucent covers
        const float3 ray_dir  = optixGetWorldRayDirection();
        const float3 cam_dir  = d_rotatePoint(make_float3(1.f, 0.f, 0.f),
                                              -0.5f * M_PI + params.camera_direction.x,
                                               0.5f * M_PI - params.camera_direction.y);
        params.camera_pixel_depth[origin_UUID] = fabsf(dot(cam_dir, ray_dir)) * t_hit;
    }
}

// ---------------------------------------------------------------------------
// Raygen: direct rays
// ---------------------------------------------------------------------------

//! The program for a scene without a background radiation field (Payload = PerRayData) or with one (PerRayDataDual), whose rays use that scene's hit and miss programs
template <typename Payload>
static __forceinline__ __device__ void raygenDirect() {
    // 3D launch: x = x-strat index [0, dim_x), y = y-strat index [0, dim_y), z = prim
    const uint3    idx        = optixGetLaunchIndex();
    const uint32_t xi         = idx.x;
    const uint32_t yi         = idx.y;
    const uint32_t prim_local = idx.z;

    const uint32_t dim_x  = params.launch_dim_x;
    const uint32_t dim_y  = params.launch_dim_y;
    const uint32_t Nrays  = dim_x * dim_y;
    const uint32_t prim_pos = params.launch_offset + prim_local;

    if (prim_pos >= params.Nprimitives) return;

    const uint32_t ptype = params.primitive_type[prim_pos];
    const int32_t  NX    = params.object_subdivisions[prim_pos * 2];
    const int32_t  NY    = params.object_subdivisions[prim_pos * 2 + 1];

    float T[16];
    loadTransformMatrix(prim_pos, T);

    // Seeded by the global primitive position: rays are launched in batches of at most 65535 primitives, and a batch-local index
    // would give primitives k and k + 65535 identical samples.
    const uint32_t linear_idx = xi + dim_x * yi;
    uint32_t seed = tea<16>(linear_idx + Nrays * prim_pos, params.random_seed);

    for (int jj = 0; jj < NY; jj++) {
        for (int ii = 0; ii < NX; ii++) {

            // UUID for this sub-patch
            const uint32_t UUID = params.primitiveID[prim_pos] + (uint32_t)(jj * NX + ii);

            const float Rx = rnd(seed);
            const float Ry = rnd(seed);

            float3 sp;
            float3 normal;

            if (ptype == 0 || ptype == 3) { // Patch or Tile: canonical space [-0.5, 0.5]^2
                const float dx = 1.0f / float(NX);
                const float dy = 1.0f / float(NY);
                sp.x = -0.5f + ii * dx + (float(xi) + Rx) * dx / float(dim_x);
                sp.y = -0.5f + jj * dy + (float(yi) + Ry) * dy / float(dim_y);
                sp.z = 0.f;

                // Origin mask rejection sampling: only launch from opaque regions (matches OptiX 6 raygen behavior)
                const int32_t msk_orig = params.mask_IDs[prim_pos];
                if (msk_orig >= 0) {
                    for (int attempt = 0; attempt < 10; attempt++) {
                        if (sampleMask(msk_orig, originTextureUV(prim_pos, ptype, sp))) break;
                        sp.x = -0.5f + (ii + rnd(seed)) * dx;
                        sp.y = -0.5f + (jj + rnd(seed)) * dy;
                    }
                }

                float3 v0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, v0);
                float3 v1 = make_float3(1.f, 0.f, 0.f); d_transformPoint(T, v1);
                float3 v2 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, v2);
                normal = normalize(cross(v1 - v0, v2 - v0));

            } else if (ptype == 1) { // Triangle: canonical space (0,0,0)-(0,1,0)-(1,1,0)
                if (Rx < Ry) { sp.x = Rx; sp.y = Ry; }
                else          { sp.x = Ry; sp.y = Rx; }
                sp.z = 0.f;

                // Origin mask rejection sampling
                const int32_t msk_orig_t = params.mask_IDs[prim_pos];
                if (msk_orig_t >= 0) {
                    for (int attempt = 0; attempt < 10; attempt++) {
                        if (sampleMask(msk_orig_t, originTextureUV(prim_pos, ptype, sp))) break;
                        float Rx2 = rnd(seed), Ry2 = rnd(seed);
                        if (Rx2 < Ry2) { sp.x = Rx2; sp.y = Ry2; }
                        else            { sp.x = Ry2; sp.y = Rx2; }
                    }
                }

                float3 v0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, v0);
                float3 v1 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, v1);
                float3 v2 = make_float3(1.f, 1.f, 0.f); d_transformPoint(T, v2);
                normal = normalize(cross(v1 - v0, v2 - v0));

            } else {
                // Unsupported primitive type — should have been caught by updateGeometry().
                printf("ERROR (OptiX8 __raygen__direct): unsupported primitive type %u at index %u\n",
                       ptype, prim_pos);
                __trap();
            }

            // The class of a spectral texture at the launch point, whose reflectivity and transmissivity the rays from it use
            const uint32_t origin_palette_entry = paletteEntryAt(prim_pos, originTextureUV(prim_pos, ptype, sp));

            // The shading normal of a normal map at the launch point, which only a surface layer's angle-dependent split of the light uses here
            const uint32_t origin_map = normalMapOf(prim_pos);
            const bool origin_has_normal = origin_map != helios_nm::NO_MAP && params.surface_model != nullptr && params.surface_model[prim_pos] != 0u;
            float3 origin_shading_normal = normal;
            if (origin_has_normal) {
                origin_shading_normal = normalMapShadingNormalAt(prim_pos, ptype, origin_map, T, normal, sp, false);
            }

            // Transform sample point to world space
            float3 ray_origin = sp;
            d_transformPoint(T, ray_origin);

            // Send a ray toward each source
            for (uint32_t rr = 0; rr < params.Nsources; rr++) {

                const uint32_t src_type = params.source_types[rr];

                float3 ray_direction;
                float  ray_tmax;
                double source_factor;
                if (!sampleSourceDirection(rr, ray_origin, seed, ray_direction, ray_tmax, source_factor)) {
                    continue; // the back of a rectangle or disk source, or an unknown source type
                }
                const double strength = (1.0 / double(dim_x * dim_y)) * (double) fabsf(dot(normal, ray_direction)) * source_factor;

                // With a background field, the direct ray type's dual programs (see __miss__direct_dual)
                const bool dual_scene = Payload::is_dual; // a compile-time constant in each program
                const uint32_t direct_sbt  = dual_scene ? DUAL_FIELD_SBT_OFFSET + 0u : 0u;
                const uint32_t direct_miss = dual_scene ? DUAL_FIELD_SBT_OFFSET + 1u + 0u : 0u;

                Payload prd;
                prd.seed                 = seed;
                prd.origin_UUID          = UUID;
                prd.source_ID            = (unsigned char)rr;
                prd.skip_origin          = true;
                prd.hit_periodic_boundary = false;
                prd.tile_x = 0; // launched from the central copy of the domain
                prd.tile_y = 0;
                prd.origin_tile_x = 0;
                prd.origin_tile_y = 0;
                prd.face                 = (dot(ray_direction, normal) > 0.f);
                // A surface layer sees the light at the angle of the shading normal at the launch point (|m.L|, so that a facet tipped away from the source counts as its mirror image,
                // consistently with the folding of diffuse rays); the beam power the primitive intercepts stays at the geometric cosine
                prd.origin_cos = origin_has_normal ? fabsf(dot(ray_direction, origin_shading_normal)) : fabsf(dot(ray_direction, normal));
                prd.palette_entry        = origin_palette_entry;
                initCoverTransmittance(prd); // translucent-cover attenuation starts at 1 (no covers crossed)
                startDualRay(prd, prim_pos);

                // Strength set above per source type
                prd.strength = strength;

                // Only fire from the top face, or from the bottom face of a two-sided primitive (twosided_flag 1). An invisible
                // primitive (twosided_flag 3) fires no rays.
                const int8_t tsf = params.twosided_flag[prim_pos];
                if (!prd.face && tsf != 1) continue;
                if (tsf == 3) continue;

                uint32_t u0, u1;
                float3 current_origin = ray_origin;

                // A ray that reaches the wrap limit deposits nothing (treated as blocked). After a wrap, a ray toward a finite
                // source only has the rest of the distance to the point it sampled on the source left to travel. Every segment
                // starts with no minimum distance: the primitive the ray was launched from is skipped (skip_origin), and a
                // minimum distance let a ray launched that close to an adjacent surface pass through it where the two meet. A
                // wrapped ray must likewise hit geometry right at the boundary it was wrapped to.
                bool wrap_limit_reached = true;
                float segment_t_min = 0.f;
                for (int wrap = 0; wrap < MAX_PERIODIC_WRAPS; ++wrap) {
                    packPointer(&prd, u0, u1);
                    optixTrace(
                        params.traversable,
                        current_origin,
                        ray_direction,
                        segment_t_min, // tmin
                        ray_tmax,    // tmax
                        0.f,         // time
                        OptixVisibilityMask(255),
                        OPTIX_RAY_FLAG_NONE,
                        direct_sbt,  // SBT offset  (direct hit group)
                        0,           // SBT stride
                        direct_miss, // miss SBT index (direct miss)
                        u0, u1
                    );
                    if (!prd.hit_periodic_boundary) {
                        wrap_limit_reached = false;
                        break;
                    }
                    current_origin = prd.periodic_hit;
                    segment_t_min = 0.f;
                    if (src_type != 0) {
                        ray_tmax = fmaxf(ray_tmax - prd.periodic_hit_t, 2e-4f);
                    }
                    prd.hit_periodic_boundary = false;
                }
                if (wrap_limit_reached) {
                    countAbandonedRay();
                }

                seed = prd.seed;
            }
        }
    }
}

extern "C" __global__ void __raygen__direct() {
    raygenDirect<PerRayData>();
}

//! __raygen__direct for a scene with a background radiation field
extern "C" __global__ void __raygen__direct_dual() {
    raygenDirect<PerRayDataDual>();
}

// ---------------------------------------------------------------------------
// Raygen: diffuse rays
// ---------------------------------------------------------------------------

//! The program for a scene without a background radiation field (Payload = PerRayData) or with one (PerRayDataDual), whose rays use that scene's hit and miss programs
template <typename Payload>
static __forceinline__ __device__ void raygenDiffuse() {
    // 3D launch: x=theta_idx, y=phi_idx, z=prim_local
    const uint3    idx        = optixGetLaunchIndex();
    const uint32_t theta_idx  = idx.x;
    const uint32_t phi_idx    = idx.y;
    const uint32_t prim_local = idx.z;

    const uint32_t dim_x = params.launch_dim_x;
    const uint32_t dim_y = params.launch_dim_y;
    const uint32_t dimxy = dim_x * dim_y;

    const uint32_t prim_pos = params.launch_offset + prim_local;
    if (prim_pos >= params.Nprimitives) return;

    // Skip bottom-face launch for one-sided primitives
    if (params.launch_face == 0 && params.twosided_flag[prim_pos] == 0) return;

    const uint32_t ptype = params.primitive_type[prim_pos];
    const int32_t  NX    = params.object_subdivisions[prim_pos * 2];
    const int32_t  NY    = params.object_subdivisions[prim_pos * 2 + 1];

    float T[16];
    loadTransformMatrix(prim_pos, T);

    // Seed once per ray index, by the global primitive position (see __raygen__direct)
    const uint32_t linear_idx = theta_idx + dim_x * phi_idx;
    uint32_t seed = tea<16>(linear_idx + dimxy * prim_pos, params.random_seed);

    // Stratified cosine-weighted hemisphere sampling
    const float Rt = (theta_idx + rnd(seed)) / float(dim_x);
    const float Rp = (phi_idx   + rnd(seed)) / float(dim_y);
    const float t  = asin_safe(sqrtf(Rt));
    const float p  = 2.f * M_PI * Rp;
    float3 ray_dir_canonical;
    ray_dir_canonical.x = sinf(t) * cosf(p);
    ray_dir_canonical.y = sinf(t) * sinf(p);
    ray_dir_canonical.z = cosf(t);

    for (int jj = 0; jj < NY; jj++) {
        for (int ii = 0; ii < NX; ii++) {

            const uint32_t UUID = params.primitiveID[prim_pos] + (uint32_t)(jj * NX + ii);

            const float Rx = rnd(seed);
            const float Ry = rnd(seed);

            float3 sp;
            float3 normal;

            if (ptype == 0 || ptype == 3) { // Patch or Tile
                const float dx = 1.f / float(NX);
                const float dy = 1.f / float(NY);
                sp.x = -0.5f + (ii + Rx) * dx;
                sp.y = -0.5f + (jj + Ry) * dy;
                sp.z = 0.f;

                // Origin mask rejection sampling
                const int32_t msk_orig_d = params.mask_IDs[prim_pos];
                if (msk_orig_d >= 0) {
                    for (int attempt = 0; attempt < 10; attempt++) {
                        if (sampleMask(msk_orig_d, originTextureUV(prim_pos, ptype, sp))) break;
                        sp.x = -0.5f + (ii + rnd(seed)) * dx;
                        sp.y = -0.5f + (jj + rnd(seed)) * dy;
                    }
                }

                float3 v0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, v0);
                float3 v1 = make_float3(1.f, 0.f, 0.f); d_transformPoint(T, v1);
                float3 v2 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, v2);
                normal = normalize(cross(v1 - v0, v2 - v0));

            } else if (ptype == 1) { // Triangle
                if (Rx < Ry) { sp.x = Rx; sp.y = Ry; }
                else          { sp.x = Ry; sp.y = Rx; }
                sp.z = 0.f;

                // Origin mask rejection sampling
                const int32_t msk_orig_dt = params.mask_IDs[prim_pos];
                if (msk_orig_dt >= 0) {
                    for (int attempt = 0; attempt < 10; attempt++) {
                        if (sampleMask(msk_orig_dt, originTextureUV(prim_pos, ptype, sp))) break;
                        float Rx2 = rnd(seed), Ry2 = rnd(seed);
                        if (Rx2 < Ry2) { sp.x = Rx2; sp.y = Ry2; }
                        else            { sp.x = Ry2; sp.y = Rx2; }
                    }
                }

                float3 v0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, v0);
                float3 v1 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, v1);
                float3 v2 = make_float3(1.f, 1.f, 0.f); d_transformPoint(T, v2);
                normal = normalize(cross(v1 - v0, v2 - v0));

            } else {
                // Unsupported primitive type — should have been caught by updateGeometry().
                printf("ERROR (OptiX8 __raygen__diffuse): unsupported primitive type %u at index %u\n",
                       ptype, prim_pos);
                __trap();
            }

            // Rotate hemisphere direction by primitive normal orientation. A normal map leaves the cosine lobe on the facet normal, so that the primitive gathers what its
            // footprint intercepts (as a relief that is a heightfield does); a surface layer sees each ray at its angle to the shading normal at the launch point.
            const float3 ray_dir = d_rotatePoint(ray_dir_canonical, acos_safe(normal.z), atan2f(normal.y, normal.x));
            const uint32_t origin_map = normalMapOf(prim_pos);
            float origin_lobe_cos = fabsf(dot(ray_dir, normal));
            if (origin_map != helios_nm::NO_MAP && params.surface_model != nullptr && params.surface_model[prim_pos] != 0u) {
                origin_lobe_cos = fabsf(dot(ray_dir, normalMapShadingNormalAt(prim_pos, ptype, origin_map, T, normal, sp, false)));
            }

            // The class of a spectral texture at the launch point, whose reflectivity and transmissivity the rays from it use
            const uint32_t origin_palette_entry = paletteEntryAt(prim_pos, originTextureUV(prim_pos, ptype, sp));

            // Transform origin point to world space
            float3 ray_origin = sp;
            d_transformPoint(T, ray_origin);

            // With a background field, the diffuse ray type's dual programs (see __closesthit__diffuse_dual)
            const bool dual_scene = Payload::is_dual; // a compile-time constant in each program
            const uint32_t diffuse_sbt  = dual_scene ? DUAL_FIELD_SBT_OFFSET + 1u : 1u;
            const uint32_t diffuse_miss = dual_scene ? DUAL_FIELD_SBT_OFFSET + 1u + 1u : 1u;

            Payload prd;
            prd.palette_entry         = origin_palette_entry;
            prd.seed                  = seed;
            prd.origin_UUID           = UUID;
            prd.source_ID             = (unsigned char)params.Nsources; // diffuse/scatter material slot, weighted by the band's combined incident spectrum
            prd.skip_origin           = true;
            prd.hit_periodic_boundary = false;
            prd.strength              = 1.0 / double(dimxy);
            prd.origin_cos = origin_lobe_cos;
            initCoverTransmittance(prd); // translucent-cover attenuation starts at 1 (no covers crossed)
            startDualRay(prd, prim_pos);

            uint32_t u0, u1;

            if (params.launch_face == 1 && params.twosided_flag[prim_pos] != 3) {
                prd.face = true;
                float3 cur_origin = ray_origin;
                // Trace segment by segment: re-trace after wrapping through a periodic boundary and after passing a
                // translucent cover. The first segment starts with no minimum distance, since the primitive the ray was
                // launched from is skipped (skip_origin) and a minimum distance let a ray launched that close to an adjacent
                // surface pass through it where the two meet; a wrapped ray must likewise hit geometry right at the boundary.
                prd.hit_cover = false;
                int wraps = 0;
                float segment_t_min = 0.f;
                prd.tile_x = 0; // launched from the central copy of the domain
                prd.tile_y = 0;
                prd.origin_tile_x = 0;
                prd.origin_tile_y = 0;
                for (int segment = 0; segment < MAX_RAY_SEGMENTS; ++segment) {
                    startDualSegment(prd); // no excluded primitive passed in this segment yet
                    packPointer(&prd, u0, u1);
                    optixTrace(
                        params.traversable,
                        cur_origin, ray_dir,
                        segment_t_min, 1e38f, 0.f,
                        OptixVisibilityMask(255),
                        OPTIX_RAY_FLAG_NONE,
                        diffuse_sbt, 0, diffuse_miss, // SBT offset (diffuse hit), stride=0, miss index (diffuse miss)
                        u0, u1
                    );
                    if (prd.hit_periodic_boundary) {
                        if (++wraps >= MAX_PERIODIC_WRAPS) {
                            countAbandonedRay();
                            break;
                        }
                        cur_origin = prd.periodic_hit;
                        segment_t_min = 0.f;
                        prd.hit_periodic_boundary = false;
                    } else if (prd.hit_cover) {
                        cur_origin = prd.cover_exit;
                        segment_t_min = 1e-4f;
                        prd.hit_cover = false;
                    } else {
                        break;
                    }
                }
            } else if (params.launch_face == 0 && params.twosided_flag[prim_pos] == 1) {
                prd.face = false;
                float3 neg_dir = make_float3(-ray_dir.x, -ray_dir.y, -ray_dir.z);
                float3 cur_origin = ray_origin;
                // Trace segment by segment: re-trace after wrapping through a periodic boundary and after passing a
                // translucent cover. The first segment starts with no minimum distance (see the top-face launch above).
                prd.hit_cover = false;
                int wraps = 0;
                float segment_t_min = 0.f;
                prd.tile_x = 0; // launched from the central copy of the domain
                prd.tile_y = 0;
                prd.origin_tile_x = 0;
                prd.origin_tile_y = 0;
                for (int segment = 0; segment < MAX_RAY_SEGMENTS; ++segment) {
                    startDualSegment(prd); // no excluded primitive passed in this segment yet
                    packPointer(&prd, u0, u1);
                    optixTrace(
                        params.traversable,
                        cur_origin, neg_dir,
                        segment_t_min, 1e38f, 0.f,
                        OptixVisibilityMask(255),
                        OPTIX_RAY_FLAG_NONE,
                        diffuse_sbt, 0, diffuse_miss,
                        u0, u1
                    );
                    if (prd.hit_periodic_boundary) {
                        if (++wraps >= MAX_PERIODIC_WRAPS) {
                            countAbandonedRay();
                            break;
                        }
                        cur_origin = prd.periodic_hit;
                        segment_t_min = 0.f;
                        prd.hit_periodic_boundary = false;
                    } else if (prd.hit_cover) {
                        cur_origin = prd.cover_exit;
                        segment_t_min = 1e-4f;
                        prd.hit_cover = false;
                    } else {
                        break;
                    }
                }
            }

            seed = prd.seed;
        }
    }
}

extern "C" __global__ void __raygen__diffuse() {
    raygenDiffuse<PerRayData>();
}

//! __raygen__diffuse for a scene with a background radiation field
extern "C" __global__ void __raygen__diffuse_dual() {
    raygenDiffuse<PerRayDataDual>();
}

// ---------------------------------------------------------------------------
// Raygen: camera rays
// 3D launch: x=ray_within_pixel [0,anti_samples), y=tile_column, z=tile_row
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__camera() {
    const uint3    idx         = optixGetLaunchIndex();
    const uint32_t ray_idx     = idx.x; // sample index within pixel
    const uint32_t col         = idx.y; // tile column
    const uint32_t row         = idx.z; // tile row

    const uint32_t dim_x  = params.launch_dim_x; // antialiasing_samples
    const uint32_t dim_y  = params.launch_dim_y; // tile_width

    // Global pixel coordinates
    const uint32_t ii = (uint32_t)params.camera_pixel_offset.x + col;
    const uint32_t jj = (uint32_t)params.camera_pixel_offset.y + row;

    // Linear pixel index in the full image
    const uint32_t pixel_index = jj * (uint32_t)params.camera_resolution_full.x + ii;

    // Seed: unique per (ray_idx, col, row)
    const uint32_t linear_idx = dim_x * col + ray_idx;
    uint32_t seed = tea<16>(linear_idx + dim_x * dim_y * row, params.random_seed);

    const float Rx = rnd(seed);
    const float Ry = rnd(seed);

    // Map sub-pixel sample to view-space point on viewplane
    // sp.x = viewplane distance, sp.y/sp.z = horizontal/vertical offsets
    const float multiplier = 1.0f / params.FOV_aspect_ratio;
    float3 sp;
    sp.y = -0.5f + ((float)ii + Rx) / (float)params.camera_resolution_full.x;
    sp.z = ( 0.5f - ((float)jj + Ry) / (float)params.camera_resolution_full.y) * multiplier;
    sp.x = params.camera_viewplane_length;

    // Focal point on focus plane
    const float3 p = make_float3(
        params.camera_focal_length,
        sp.y / params.camera_viewplane_length * params.camera_focal_length,
        sp.z / params.camera_viewplane_length * params.camera_focal_length);

    // Sample lens (pinhole if lens_diameter == 0)
    float3 ray_origin = make_float3(0.f, 0.f, 0.f);
    if (params.camera_lens_diameter > 0.f) {
        float3 disk_sample;
        d_sampleDisk(seed, disk_sample);
        ray_origin = make_float3(0.f, 0.5f * disk_sample.x * params.camera_lens_diameter,
                                      0.5f * disk_sample.y * params.camera_lens_diameter);
    }

    float3 ray_direction = make_float3(p.x - ray_origin.x, p.y - ray_origin.y, p.z - ray_origin.z);

    // Rotate into world space
    const float theta = -0.5f * M_PI + params.camera_direction.x;
    const float phi   =  0.5f * M_PI - params.camera_direction.y;
    ray_origin    = d_rotatePoint(ray_origin,    theta, phi) + params.camera_position;
    ray_direction = d_rotatePoint(ray_direction, theta, phi);
    ray_direction = ray_direction * (1.0f / d_magnitude(ray_direction));

    PerRayData prd;
    prd.strength             = 1.0f / (float)dim_x;
    prd.origin_UUID          = pixel_index;
    prd.skip_origin          = false; // origin_UUID is a pixel index
    prd.face                 = true;
    prd.source_ID            = 0;
    prd.seed                 = seed;
    prd.palette_entry        = 0xFFFFFFFFu; // camera rays leave no primitive
    prd.hit_periodic_boundary = false;
    prd.tile_x = params.camera_tile.x; // the periodic copy of the domain the camera is in
    prd.tile_y = params.camera_tile.y;
    prd.origin_tile_x = prd.tile_x; // not used: the ray skips no primitive
    prd.origin_tile_y = prd.tile_y;
    prd.hit_cover            = false;
    initCoverTransmittance(prd); // camera throughput: no translucent covers crossed yet

    uint32_t p0, p1;
    packPointer(&prd, p0, p1);

    // A wrapped ray continues with no minimum distance, so that it hits geometry right at the boundary it was wrapped to
    const float initial_t_min = 1e-5f;
    const float t_max = 1e30f;
    float t_min = initial_t_min;

    // A ray starting above or below a periodic domain is moved to it in one step
    advanceToPeriodicFaces(ray_origin, ray_direction, prd);

    // Trace segment by segment: the ray is re-launched after wrapping through a periodic boundary and after
    // passing a translucent cover, whose closest-hit has already added the cover's own radiance.
    float3 cur_origin = ray_origin;
    int wraps = 0;
    for (int segment = 0; segment < MAX_RAY_SEGMENTS; segment++) {
        prd.hit_periodic_boundary = false;
        prd.hit_cover             = false;
        optixTrace(params.traversable, cur_origin, ray_direction,
                   t_min, t_max, 0.f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                   2u, 0u, 2u, // SBT: offset=2 (camera hit), stride=0, miss=2
                   p0, p1);
        if (prd.hit_periodic_boundary) {
            if (++wraps >= MAX_PERIODIC_WRAPS) break;
            cur_origin = prd.periodic_hit;
            t_min = 0.f;
        } else if (prd.hit_cover) {
            cur_origin = prd.cover_exit;
            t_min = initial_t_min;
        } else {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Raygen: pixel label rays
// 3D launch: x=1 (always), y=tile_column, z=tile_row
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__pixel_label() {
    const uint3    idx  = optixGetLaunchIndex();
    const uint32_t col  = idx.y;
    const uint32_t row  = idx.z;

    const uint32_t dim_y = params.launch_dim_y; // tile_width

    // Global pixel coordinates
    const uint32_t ii = (uint32_t)params.camera_pixel_offset.x + col;
    const uint32_t jj = (uint32_t)params.camera_pixel_offset.y + row;
    const uint32_t pixel_index = jj * (uint32_t)params.camera_resolution_full.x + ii;

    uint32_t seed = tea<16>(dim_y * row + col, params.random_seed);

    // Center of pixel, no antialiasing jitter
    const float multiplier = 1.0f / params.FOV_aspect_ratio;
    float3 sp;
    sp.y = -0.5f + ((float)ii + 0.5f) / (float)params.camera_resolution_full.x;
    sp.z = ( 0.5f - ((float)jj + 0.5f) / (float)params.camera_resolution_full.y) * multiplier;
    sp.x = params.camera_viewplane_length;

    const float3 p = make_float3(
        params.camera_focal_length,
        sp.y / params.camera_viewplane_length * params.camera_focal_length,
        sp.z / params.camera_viewplane_length * params.camera_focal_length);

    float3 ray_origin    = make_float3(0.f, 0.f, 0.f);
    float3 ray_direction = p;

    const float theta = -0.5f * M_PI + params.camera_direction.x;
    const float phi   =  0.5f * M_PI - params.camera_direction.y;
    ray_origin    = d_rotatePoint(ray_origin,    theta, phi) + params.camera_position;
    ray_direction = d_rotatePoint(ray_direction, theta, phi);
    ray_direction = ray_direction * (1.0f / d_magnitude(ray_direction));

    PerRayData prd;
    prd.strength             = 0.f; // used as the distance travelled before any translucent covers passed
    prd.origin_UUID          = pixel_index;
    prd.skip_origin          = false; // origin_UUID is a pixel index
    prd.face                 = true;
    prd.source_ID            = 0;
    prd.seed                 = seed;
    prd.palette_entry        = 0xFFFFFFFFu; // camera rays leave no primitive
    prd.hit_periodic_boundary = false;
    prd.tile_x = params.camera_tile.x; // the periodic copy of the domain the camera is in
    prd.tile_y = params.camera_tile.y;
    prd.origin_tile_x = prd.tile_x; // not used: the ray skips no primitive
    prd.origin_tile_y = prd.tile_y;
    prd.hit_cover            = false;
    initCoverTransmittance(prd);

    uint32_t p0, p1;
    packPointer(&prd, p0, p1);

    // A wrapped ray continues with no minimum distance, so that it hits geometry right at the boundary it was wrapped to
    const float initial_t_min = 1e-5f;
    const float t_max = 1e30f;
    float t_min = initial_t_min;

    // A ray starting above or below a periodic domain is moved to it in one step; the distance counts toward the depth
    prd.strength = (double) advanceToPeriodicFaces(ray_origin, ray_direction, prd);

    // Trace segment by segment, as __raygen__camera does, so the label is that of the first surface behind
    // any translucent covers.
    float3 cur_origin = ray_origin;
    int wraps = 0;
    bool ray_ended = false;
    for (int segment = 0; segment < MAX_RAY_SEGMENTS; segment++) {
        prd.hit_periodic_boundary = false;
        prd.hit_cover             = false;
        optixTrace(params.traversable, cur_origin, ray_direction,
                   t_min, t_max, 0.f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                   3u, 0u, 3u, // SBT: offset=3 (pixel label hit), stride=0, miss=3
                   p0, p1);
        if (prd.hit_periodic_boundary) {
            if (++wraps >= MAX_PERIODIC_WRAPS) break;
            cur_origin = prd.periodic_hit;
            t_min = 0.f;
        } else if (prd.hit_cover) {
            cur_origin = prd.cover_exit;
            t_min = initial_t_min;
        } else {
            ray_ended = true;
            break;
        }
    }

    // A ray that runs out of wraps or segments is treated as blocked: the pixel shows no primitive, as for a ray that leaves the scene
    if (!ray_ended) {
        if (params.camera_pixel_label) {
            params.camera_pixel_label[pixel_index] = 0u;
        }
        if (params.camera_pixel_depth) {
            params.camera_pixel_depth[pixel_index] = -1.0f;
        }
    }
}
