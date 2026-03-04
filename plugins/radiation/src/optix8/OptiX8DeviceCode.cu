/** \file "OptiX8DeviceCode.cu" OptiX 8.1 device programs for the radiation model.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <optix.h>
#include <optix_device.h>
#include "OptiX8Math.h"
#include "OptiX8LaunchParams.h"

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

// Load transform matrix for primitive at global position pos
__device__ __forceinline__ void loadTransformMatrix(uint32_t pos, float (&T)[16]) {
    for (int i = 0; i < 16; i++) {
        T[i] = params.transform_matrix[pos * 16 + i];
    }
}

// ---------------------------------------------------------------------------
// PerRayData accessor (uses getPRD() from OptiX8LaunchParams.h)
// ---------------------------------------------------------------------------

// getPRD() is defined in OptiX8LaunchParams.h (guarded by #ifdef __CUDACC__)

// ---------------------------------------------------------------------------
// Intersection program: patch (rectangle in canonical [0,1]^2 space)
// ---------------------------------------------------------------------------

extern "C" __global__ void __intersection__patch() {
    const uint32_t prim_idx  = optixGetPrimitiveIndex();
    const uint32_t obj_idx   = optixGetInstanceIndex(); // 0 (single-instance GAS)
    (void)obj_idx;

    // Retrieve the primitive's global position
    // prim_idx is the index into the AABB array, which equals the global primitive position
    const uint32_t pos = prim_idx;

    // Verify this is actually a patch (type 0)
    if (params.primitive_type[pos] != 0) return;

    // Get transform matrix for this primitive
    float T[16];
    loadTransformMatrix(pos, T);

    // Ray in world space
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float  t_min         = optixGetRayTmin();
    const float  t_max         = optixGetRayTmax();

    // Patch normal is +Z in canonical space, transformed to world space
    // Normal = T * (0,0,1,0) = column 2 of rotation part
    float3 normal = make_float3(T[2], T[6], T[10]);
    float  denom  = dot(ray_direction, normal);

    if (fabsf(denom) < 1e-8f) return; // parallel

    // Patch origin is at T * (0,0,0,1) = translation column
    float3 patch_origin = make_float3(T[3], T[7], T[11]);
    float  t = dot(patch_origin - ray_origin, normal) / denom;

    if (t < t_min || t > t_max) return;

    // Compute local 2D coordinates of hit point
    // We need the inverse transform to check if hit is in [0,1]^2
    float3 hit_world = ray_origin + t * ray_direction;
    float3 hit_local = hit_world - patch_origin;

    // Local X axis = T * (1,0,0,0) - T * (0,0,0,0)
    float3 local_x = make_float3(T[0], T[4], T[8]);
    float3 local_y = make_float3(T[1], T[5], T[9]);

    float lx2 = dot(local_x, local_x);
    float ly2 = dot(local_y, local_y);
    if (lx2 < 1e-12f || ly2 < 1e-12f) return;

    float u = dot(hit_local, local_x) / lx2;
    float v = dot(hit_local, local_y) / ly2;

    if (u < -0.5f || u > 0.5f || v < -0.5f || v > 0.5f) return;

    // Solid fraction / texture masking (Phase 7: skip for now, always solid)
    // float solid_frac = params.primitive_solid_fraction[pos];

    // Determine face: top if ray hits front face (denom < 0)
    uint32_t face_attr = (denom < 0.f) ? 1u : 0u;
    // For one-sided primitives, reject back-face hits
    if (!face_attr && !params.twosided_flag[pos]) return;

    // Report intersection: attribute0 = UUID, attribute1 = face
    uint32_t uuid = params.patch_UUIDs[prim_idx];
    optixReportIntersection(t, 0, uuid, face_attr);
}

// ---------------------------------------------------------------------------
// Intersection stubs for other types (Phase 6+)
// ---------------------------------------------------------------------------

extern "C" __global__ void __intersection__triangle() {
    // TODO: Phase 6
}

extern "C" __global__ void __intersection__disk() {
    // TODO: Phase 8
}

extern "C" __global__ void __intersection__tile() {
    // TODO: Phase 11
}

extern "C" __global__ void __intersection__voxel() {
    // TODO: Phase 11
}

extern "C" __global__ void __intersection__bbox() {
    // TODO: Phase 9
}

// ---------------------------------------------------------------------------
// Miss programs
// ---------------------------------------------------------------------------

extern "C" __global__ void __miss__direct() {
    PerRayData *prd = getPayloadPRD();

    const uint32_t origin_position = params.primitive_positions[prd->origin_UUID];
    const uint32_t Nprims          = params.Nprimitives;
    const uint32_t Nbands_global   = params.Nbands_global;
    const uint32_t Nbands_launch   = params.Nbands_launch;

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
        const float t_rho = params.rho[radprop_ind];
        const float t_tau = params.tau[radprop_ind];

        // source_fluxes layout: [source * Nbands_launch + band_launch]
        const uint32_t flux_idx  = prd->source_ID * Nbands_launch + (uint32_t)b;
        const float source_flux  = params.source_fluxes[flux_idx];

        const double strength  = prd->strength * (double)source_flux;
        const float absorption = (float)(strength * (1.0 - t_rho - t_tau));

        atomicFloatAdd(&params.radiation_in[ind_origin], absorption);

        if (t_rho > 0.f || t_tau > 0.f) {
            if (prd->face) {
                atomicFloatAdd(&params.scatter_buff_top[ind_origin],    (float)(strength * t_rho));
                atomicFloatAdd(&params.scatter_buff_bottom[ind_origin], (float)(strength * t_tau));
            } else {
                atomicFloatAdd(&params.scatter_buff_bottom[ind_origin], (float)(strength * t_rho));
                atomicFloatAdd(&params.scatter_buff_top[ind_origin],    (float)(strength * t_tau));
            }
        }
    }
}

extern "C" __global__ void __miss__diffuse() {
    PerRayData *prd = getPayloadPRD();
    (void)prd;
}

extern "C" __global__ void __miss__camera() {
    PerRayData *prd = getPayloadPRD();
    prd->origin_UUID = 0; // miss = sky
}

extern "C" __global__ void __miss__pixel_label() {
    PerRayData *prd = getPayloadPRD();
    prd->origin_UUID = 0; // sky pixel
}

// ---------------------------------------------------------------------------
// Closest-hit: direct radiation
// ---------------------------------------------------------------------------

extern "C" __global__ void __closesthit__direct() {
    // Retrieve hit UUID and face from attributes
    const uint32_t uuid      = optixGetAttribute_0();
    const uint32_t face_uint = optixGetAttribute_1();
    const bool     face_top  = (face_uint == 1u);

    PerRayData *prd = getPayloadPRD();

    // Find array position for this UUID
    const uint32_t pos = params.primitive_positions[uuid];
    if (pos == UINT_MAX) return; // invalid UUID

    // Source ID and band info
    const uint32_t src_id      = prd->source_ID;
    const uint32_t Nprims      = params.Nprimitives;
    const uint32_t Nbands      = params.Nbands_launch;
    const uint32_t Nbands_g    = params.Nbands_global;

    // Accumulate absorbed radiation for each launched band
    for (uint32_t b = 0; b < Nbands_g; b++) {
        if (!params.band_launch_flag[b]) continue;

        // Source flux for this source and band
        const float flux = params.source_fluxes[src_id * Nbands + b];
        if (flux == 0.f) continue;

        // rho and tau for this primitive, source, band
        const uint32_t mat_idx = src_id * Nbands_g * Nprims + b * Nprims + pos;
        const float rho = params.rho[mat_idx];
        const float tau = params.tau[mat_idx];

        // Absorbed = strength * (1 - rho - tau)
        const float absorbed = (float)prd->strength * flux * (1.f - rho - tau);
        atomicFloatAdd(&params.radiation_in[pos * Nbands_g + b], absorbed);

        // Transmit if tau > 0: deposit on opposite face
        if (tau > 0.f) {
            const float transmitted = (float)prd->strength * flux * tau;
            // Transmitted radiation goes to radiation_in of the same prim (both faces)
            atomicFloatAdd(&params.radiation_in[pos * Nbands_g + b], transmitted);
        }
    }
}

// ---------------------------------------------------------------------------
// Closest-hit: diffuse radiation
// ---------------------------------------------------------------------------

extern "C" __global__ void __closesthit__diffuse() {
    // TODO: Phase 2
    const uint32_t uuid      = optixGetAttribute_0();
    const uint32_t face_uint = optixGetAttribute_1();
    (void)uuid; (void)face_uint;
}

// ---------------------------------------------------------------------------
// Closest-hit: camera
// ---------------------------------------------------------------------------

extern "C" __global__ void __closesthit__camera() {
    // TODO: Phase 10
    const uint32_t uuid = optixGetAttribute_0();
    PerRayData *prd = getPayloadPRD();
    prd->origin_UUID = uuid;
}

// ---------------------------------------------------------------------------
// Closest-hit: pixel label
// ---------------------------------------------------------------------------

extern "C" __global__ void __closesthit__pixel_label() {
    const uint32_t uuid = optixGetAttribute_0();
    PerRayData *prd = getPayloadPRD();
    prd->origin_UUID = uuid;
}

// ---------------------------------------------------------------------------
// Raygen: direct rays
// Phase 1: launch rays from primitives toward each source
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__direct() {
    // 3D launch: x = ray index [0, Nrays), y = 0, z = primitive index within launch
    const uint3 idx         = optixGetLaunchIndex();
    const uint32_t ray_index  = idx.x;
    const uint32_t prim_local = idx.z;

    const uint32_t Nrays    = params.launch_dim_x;
    const uint32_t prim_pos = params.launch_offset + prim_local;

    if (prim_pos >= params.Nprimitives) return;

    const uint32_t ptype = params.primitive_type[prim_pos];
    const int32_t  NX    = params.object_subdivisions[prim_pos * 2];
    const int32_t  NY    = params.object_subdivisions[prim_pos * 2 + 1];

    float T[16];
    loadTransformMatrix(prim_pos, T);

    uint32_t seed = tea<16>(ray_index + Nrays * prim_local, params.random_seed);

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
                sp.x = -0.5f + ii * dx + float(ray_index) * dx / float(Nrays) + Rx * dx / float(Nrays);
                sp.y = -0.5f + jj * dy + Ry * dy;
                sp.z = 0.f;

                float3 v0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, v0);
                float3 v1 = make_float3(1.f, 0.f, 0.f); d_transformPoint(T, v1);
                float3 v2 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, v2);
                normal = normalize(cross(v1 - v0, v2 - v0));

            } else if (ptype == 1) { // Triangle: canonical space (0,0,0)-(0,1,0)-(1,1,0)
                if (Rx < Ry) { sp.x = Rx; sp.y = Ry; }
                else          { sp.x = Ry; sp.y = Rx; }
                sp.z = 0.f;

                float3 v0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, v0);
                float3 v1 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, v1);
                float3 v2 = make_float3(1.f, 1.f, 0.f); d_transformPoint(T, v2);
                normal = normalize(cross(v1 - v0, v2 - v0));

            } else {
                // Unsupported primitive type for Phase 1 — skip
                continue;
            }

            // Transform sample point to world space
            float3 ray_origin = sp;
            d_transformPoint(T, ray_origin);

            // Send a ray toward each source
            for (uint32_t rr = 0; rr < params.Nsources; rr++) {

                const uint32_t src_type = params.source_types[rr];

                float3 ray_direction;
                float  ray_tmax;

                if (src_type == 0) { // Collimated source
                    ray_direction = normalize(params.source_positions[rr]);
                    ray_tmax      = 1e38f;
                } else {
                    // Other source types: not implemented in Phase 1
                    continue;
                }

                PerRayData prd;
                prd.seed                 = seed;
                prd.origin_UUID          = UUID;
                prd.source_ID            = (unsigned char)rr;
                prd.hit_periodic_boundary = false;
                prd.face                 = (dot(ray_direction, normal) > 0.f);

                // Strength: Lambert cosine weighting divided by number of rays
                prd.strength = (1.0 / double(Nrays)) * (double)fabsf(dot(normal, ray_direction));

                // Only fire from the face pointing toward source (or two-sided)
                const int8_t tsf = params.twosided_flag[prim_pos];
                if (!prd.face && tsf == 0) continue;
                if (tsf == 3) continue; // reserved flag — skip

                uint32_t u0, u1;
                packPointer(&prd, u0, u1);

                optixTrace(
                    params.traversable,
                    ray_origin,
                    ray_direction,
                    1e-4f,       // tmin
                    ray_tmax,    // tmax
                    0.f,         // time
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_NONE,
                    0,           // SBT offset  (direct hit group = 0)
                    0,           // SBT stride
                    0,           // miss SBT index (direct miss = 0)
                    u0, u1
                );

                seed = prd.seed;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Raygen: diffuse rays (Phase 2)
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__diffuse() {
    // TODO: Phase 2
}

// ---------------------------------------------------------------------------
// Raygen: camera rays (Phase 10)
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__camera() {
    // TODO: Phase 10
}

// ---------------------------------------------------------------------------
// Raygen: pixel label rays (Phase 10)
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__pixel_label() {
    // TODO: Phase 10
}
