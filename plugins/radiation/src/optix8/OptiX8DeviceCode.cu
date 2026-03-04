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

    if (u < 0.f || u > 1.f || v < 0.f || v > 1.f) return;

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
    // Ray missed all geometry → no energy absorbed
    PerRayData *prd = getPayloadPRD();
    // Direct ray hitting no geometry: add strength to sky energy
    // Sky energy accumulation happens in raygen for now
    (void)prd;
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
    // TODO: Phase 1 - implement ray generation from primitives to sources
    // For Phase 0, this is a stub
    (void)params;
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
