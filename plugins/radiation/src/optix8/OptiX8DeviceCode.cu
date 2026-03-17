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

static __forceinline__ __device__ float acos_safe(float x) {
    return acosf(fmaxf(-1.f, fminf(1.f, x)));
}

static __forceinline__ __device__ float asin_safe(float x) {
    return asinf(fmaxf(-1.f, fminf(1.f, x)));
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
        float pattern   = (1.f + sky_params.x * expf(-gamma / sky_params.y))
                        * (1.f + (sky_params.z - 1.f) * (1.f - cos_theta));
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

        float t = dot(make_float3(v0.x - ray_origin.x, v0.y - ray_origin.y, v0.z - ray_origin.z), n) / nd;
        if (t < t_min || t > t_max) return;

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

        optixReportIntersection(t, 0, uuid, 0u);
        return;
    }

    if (ptype != 0 && ptype != 1 && ptype != 3) return; // only patch, triangle, tile

    float T[16];
    loadTransformMatrix(pos, T);

    if (ptype == 0 || ptype == 3) {
        // ---- Patch: rectangle in canonical [-0.5, 0.5]^2 space ----
        // Normal = third column of rotation part of T
        float3 normal = make_float3(T[2], T[6], T[10]);
        float  nd     = dot(ray_direction, normal);
        if (fabsf(nd) < 1e-8f) return; // ray parallel to patch plane

        // Patch centroid is the translation column of T
        float3 patch_origin = make_float3(T[3], T[7], T[11]);
        float  t = dot(patch_origin - ray_origin, normal) / nd;
        if (t < t_min || t > t_max) return;

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

        uint32_t face_attr = (nd < 0.f) ? 1u : 0u;
        if (!face_attr && !params.twosided_flag[pos]) return;
        optixReportIntersection(t, 0, uuid, face_attr);

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
        if (t < t_min || t > t_max) return;

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
                uv_v = 1.f - uv.y; // Y-flip to match OptiX 6 convention
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
        uint32_t face_attr = (dot(ray_direction, tri_nrm) < 0.f) ? 1u : 0u;
        if (!face_attr && !params.twosided_flag[pos]) return;
        optixReportIntersection(t, 0, uuid, face_attr);
    }
}

extern "C" __global__ void __intersection__disk() {
    // Disk intersection is not yet implemented. Disk primitives are rejected at
    // the host level in updateGeometry(), so this program is never invoked.
}

extern "C" __global__ void __intersection__tile() {
    // Tile intersection is handled by __intersection__patch (ptype == 3).
}

extern "C" __global__ void __intersection__voxel() {
    // Voxel intersection is not yet implemented. Voxel primitives are rejected at
    // the host level in updateGeometry(), so this program is never invoked.
}

extern "C" __global__ void __intersection__bbox() {
    // Bbox intersection is not yet implemented in this backend.
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

        // Camera-weighted scatter: mirrors scatter_buff but uses rho_cam/tau_cam
        if (params.Ncameras > 0 && params.rho_cam && params.scatter_buff_top_cam) {
            const uint32_t Ncameras   = params.Ncameras;
            const uint32_t cam_id     = params.camera_ID;
            const uint32_t rc_idx     = prd->source_ID * Nprims * Nbands_global * Ncameras
                                      + origin_position * Nbands_global * Ncameras
                                      + b_global * Ncameras + cam_id;
            const float t_rho_cam = params.rho_cam[rc_idx];
            const float t_tau_cam = params.tau_cam ? params.tau_cam[rc_idx] : 0.f;
            if ((t_rho_cam > 0.f || t_tau_cam > 0.f) && strength > 0.0) {
                if (prd->face) {
                    atomicFloatAdd(&params.scatter_buff_top_cam[ind_origin],    (float)(strength * t_rho_cam));
                    atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_origin], (float)(strength * t_tau_cam));
                } else {
                    atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_origin], (float)(strength * t_rho_cam));
                    atomicFloatAdd(&params.scatter_buff_top_cam[ind_origin],    (float)(strength * t_tau_cam));
                }
            }
        }

        // Accumulate incident radiation for specular for ALL cameras (per source, camera-weighted).
        // Direct rays are launched once (not per camera), so we must populate every camera's
        // slot in radiation_specular here so each camera's closest-hit can read its own data.
        if (params.radiation_specular && params.source_fluxes_cam && strength > 0.0) {
            for (uint32_t cam = 0; cam < params.Ncameras; cam++) {
                // source_fluxes_cam layout: [source][band][camera] (full 3D buffer uploaded in updateSources)
                const uint32_t weight_idx = prd->source_ID * Nbands_launch * params.Ncameras
                                          + (uint32_t)b * params.Ncameras + cam;
                const float camera_weight = params.source_fluxes_cam[weight_idx];
                // radiation_specular layout: [source][camera][primitive][band]
                const uint32_t ind_specular = prd->source_ID * params.Ncameras * Nprims * Nbands_launch
                                            + cam * Nprims * Nbands_launch
                                            + origin_position * Nbands_launch + (uint32_t)b;
                atomicFloatAdd(&params.radiation_specular[ind_specular], (float)(strength * camera_weight));
            }
        }
    }
}

extern "C" __global__ void __miss__diffuse() {
    PerRayData *prd = getPayloadPRD();

    const uint32_t origin_position = params.primitive_positions[prd->origin_UUID];
    if (origin_position == UINT_MAX) return;

    const uint32_t Nprims        = params.Nprimitives;
    const uint32_t Nbands_global = params.Nbands_global;
    const uint32_t Nbands_launch = params.Nbands_launch;

    if (params.diffuse_flux == nullptr) {
        printf("ERROR (OptiX8 __miss__diffuse): diffuse_flux is null. "
               "Call updateDiffuseRadiation() before launchDiffuseRays().\n");
        __trap();
    }

    const float3 ray_dir = optixGetWorldRayDirection();

    int b = -1;
    for (uint32_t b_global = 0; b_global < Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        b++;

        if (params.diffuse_flux[b] <= 0.f) continue;

        const float4 sky_p   = params.sky_radiance_params  ? params.sky_radiance_params[b]                : make_float4(0.f, 0.f, 0.f, 0.f);
        const float3 peak_d  = params.diffuse_peak_dir     ? params.diffuse_peak_dir[b]                   : make_float3(0.f, 0.f, 1.f);
        const float  power_K = params.diffuse_extinction   ? params.diffuse_extinction[b]                 : 0.f;
        const float  power_N = params.diffuse_dist_norm    ? params.diffuse_dist_norm[b]                  : 1.f;

        const float fd       = evaluateDiffuseAngularDistribution(ray_dir, peak_d, power_K, power_N, sky_p);
        const float strength = fd * params.diffuse_flux[b] * (float)prd->strength;

        const uint32_t ind_origin   = origin_position * Nbands_launch + (uint32_t)b;
        const uint32_t radprop_ind  = prd->source_ID * Nprims * Nbands_global
                                    + origin_position * Nbands_global + b_global;
        const float t_rho = params.rho[radprop_ind];
        const float t_tau = params.tau[radprop_ind];

        atomicFloatAdd(&params.radiation_in[ind_origin], strength * (1.f - t_rho - t_tau));

        if (t_rho > 0.f || t_tau > 0.f) {
            if (prd->face) { // top-face origin
                atomicFloatAdd(&params.scatter_buff_top[ind_origin],    strength * t_rho);
                atomicFloatAdd(&params.scatter_buff_bottom[ind_origin], strength * t_tau);
            } else {         // bottom-face origin
                atomicFloatAdd(&params.scatter_buff_bottom[ind_origin], strength * t_rho);
                atomicFloatAdd(&params.scatter_buff_top[ind_origin],    strength * t_tau);
            }
        }

        // Camera-weighted scatter: mirrors scatter_buff but uses rho_cam/tau_cam
        if (params.Ncameras > 0 && params.rho_cam && params.scatter_buff_top_cam) {
            const uint32_t Ncameras   = params.Ncameras;
            const uint32_t cam_id     = params.camera_ID;
            const uint32_t rc_idx     = prd->source_ID * Nprims * Nbands_global * Ncameras
                                      + origin_position * Nbands_global * Ncameras
                                      + b_global * Ncameras + cam_id;
            const float t_rho_cam = params.rho_cam[rc_idx];
            const float t_tau_cam = params.tau_cam ? params.tau_cam[rc_idx] : 0.f;
            if ((t_rho_cam > 0.f || t_tau_cam > 0.f) && strength > 0.f) {
                if (prd->face) {
                    atomicFloatAdd(&params.scatter_buff_top_cam[ind_origin],    strength * t_rho_cam);
                    atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_origin], strength * t_tau_cam);
                } else {
                    atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_origin], strength * t_rho_cam);
                    atomicFloatAdd(&params.scatter_buff_top_cam[ind_origin],    strength * t_tau_cam);
                }
            }
        }
    }
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

        // Sky radiance fallback
        if (radiance <= 0.0f && params.camera_sky_radiance && params.camera_sky_radiance[b] > 0.0f) {
            const float4 sky_p = params.sky_radiance_params ? params.sky_radiance_params[b]
                                                             : make_float4(0.f, 0.f, 0.f, 0.f);
            radiance = params.camera_sky_radiance[b] *
                       evaluateDiffuseAngularDistribution(ray_dir, params.sun_direction, 0.0f, 1.0f, sky_p);
        }

        if (radiance > 0.0f) {
            atomicFloatAdd(&params.radiation_in_camera[pixel_idx * Nbands_l + b],
                           radiance * (float)prd->strength);
        }
    }
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
static __forceinline__ __device__ void handlePeriodicBoundaryHit(PerRayData *prd, uint32_t hit_uuid) {
    const float  t_hit    = optixGetRayTmax();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float3 hit_pos  = make_float3(ray_orig.x + t_hit * ray_dir.x,
                                         ray_orig.y + t_hit * ray_dir.y,
                                         ray_orig.z + t_hit * ray_dir.z);

    const uint32_t bbox_local = hit_uuid - params.bbox_UUID_base;
    prd->periodic_hit = hit_pos;

    if (params.periodic_flag.x == 1 && bbox_local < 2) {
        // x-faces: bbox 0 = x-min, bbox 1 = x-max
        // vertex 0 of each face gives the face's x coordinate
        const float xmin = params.bbox_vertices[0 * 4].x; // x-min face, any vertex .x = xmin
        const float xmax = params.bbox_vertices[1 * 4].x; // x-max face, any vertex .x = xmax
        const float width = xmax - xmin;
        prd->periodic_hit.x += (bbox_local == 0) ? width : -width;
    } else {
        // y-faces: if x-periodic exists they are bboxes 2,3; otherwise 0,1
        const uint32_t y_base = (params.periodic_flag.x == 1) ? 2u : 0u;
        const float ymin = params.bbox_vertices[y_base * 4].y;       // y-min face vertex 0 .y = ymin
        const float ymax = params.bbox_vertices[(y_base + 1) * 4].y; // y-max face vertex 0 .y = ymax
        const float width = ymax - ymin;
        prd->periodic_hit.y += (bbox_local == y_base) ? width : -width;
    }
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

extern "C" __global__ void __closesthit__diffuse() {
    const uint32_t hit_uuid  = optixGetAttribute_0();
    const bool     face_top  = (optixGetAttribute_1() == 1u);

    PerRayData *prd = getPayloadPRD();

    // Periodic boundary: bbox (type-5) hit — wrap ray and return without depositing energy
    if ((params.periodic_flag.x == 1 || params.periodic_flag.y == 1) &&
        params.bbox_UUID_base != 0xFFFFFFFFu && hit_uuid >= params.bbox_UUID_base) {
        prd->hit_periodic_boundary = true;
        handlePeriodicBoundaryHit(prd, hit_uuid);
        return;
    }

    const uint32_t origin_position = params.primitive_positions[prd->origin_UUID];
    const uint32_t hit_position    = params.primitive_positions[hit_uuid];
    if (origin_position == UINT_MAX || hit_position == UINT_MAX) return;

    const uint32_t Nprims        = params.Nprimitives;
    const uint32_t Nbands_global = params.Nbands_global;
    const uint32_t Nbands_launch = params.Nbands_launch;

    int b = -1;
    for (uint32_t b_global = 0; b_global < Nbands_global; b_global++) {
        if (!params.band_launch_flag[b_global]) continue;
        b++;

        const uint32_t ind_origin = origin_position * Nbands_launch + (uint32_t)b;
        const uint32_t ind_hit    = hit_position    * Nbands_launch + (uint32_t)b;

        const double strength = face_top ? params.radiation_out_top[ind_hit]    * prd->strength
                                         : params.radiation_out_bottom[ind_hit] * prd->strength;
        if (strength == 0.0) continue;

        const uint32_t radprop_ind = prd->source_ID * Nprims * Nbands_global
                                   + origin_position * Nbands_global + b_global;
        const float t_rho = params.rho[radprop_ind];
        const float t_tau = params.tau[radprop_ind];

        atomicFloatAdd(&params.radiation_in[ind_origin], (float)(strength * (1.0 - t_rho - t_tau)));

        if (t_rho > 0.f || t_tau > 0.f) {
            if (prd->face) { // top-face origin
                atomicFloatAdd(&params.scatter_buff_top[ind_origin],    (float)(strength * t_rho));
                atomicFloatAdd(&params.scatter_buff_bottom[ind_origin], (float)(strength * t_tau));
            } else {         // bottom-face origin
                atomicFloatAdd(&params.scatter_buff_bottom[ind_origin], (float)(strength * t_rho));
                atomicFloatAdd(&params.scatter_buff_top[ind_origin],    (float)(strength * t_tau));
            }
        }

        // Camera-weighted scatter: mirrors scatter_buff but uses rho_cam/tau_cam
        if (params.Ncameras > 0 && params.rho_cam && params.scatter_buff_top_cam) {
            const uint32_t Ncameras   = params.Ncameras;
            const uint32_t cam_id     = params.camera_ID;
            const uint32_t rc_idx     = prd->source_ID * Nprims * Nbands_global * Ncameras
                                      + origin_position * Nbands_global * Ncameras
                                      + b_global * Ncameras + cam_id;
            const float t_rho_cam = params.rho_cam[rc_idx];
            const float t_tau_cam = params.tau_cam ? params.tau_cam[rc_idx] : 0.f;
            if ((t_rho_cam > 0.f || t_tau_cam > 0.f) && strength > 0.0) {
                if (prd->face) {
                    atomicFloatAdd(&params.scatter_buff_top_cam[ind_origin],    (float)(strength * t_rho_cam));
                    atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_origin], (float)(strength * t_tau_cam));
                } else {
                    atomicFloatAdd(&params.scatter_buff_bottom_cam[ind_origin], (float)(strength * t_rho_cam));
                    atomicFloatAdd(&params.scatter_buff_top_cam[ind_origin],    (float)(strength * t_tau_cam));
                }
            }
        }
    }
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
    const float    t_hit         = optixGetRayTmax();
    const float3   ray_origin    = optixGetWorldRayOrigin();
    const float3   ray_direction = optixGetWorldRayDirection();

    // Use face attribute from intersection program (correct for both patches and triangles)
    const bool   face_top  = (optixGetAttribute_1() == 1u);

    // Compute surface normal for specular reflection (needed for Blinn-Phong half-vector)
    float T[16];
    loadTransformMatrix(hit_position, T);
    float3 n0 = make_float3(0.f, 0.f, 0.f); d_transformPoint(T, n0);
    float3 n1 = make_float3(1.f, 0.f, 0.f); d_transformPoint(T, n1);
    float3 n2 = make_float3(0.f, 1.f, 0.f); d_transformPoint(T, n2);
    float3 normal = normalize(cross(n1 - n0, n2 - n0));
    // Ensure normal points toward the camera (consistent with face_top)
    if (face_top != (dot(normal, ray_direction) < 0.f)) {
        normal = make_float3(-normal.x, -normal.y, -normal.z);
    }

    for (uint32_t b = 0; b < Nbands_l; b++) {
        // Radiance from hit surface (outgoing flux / pi = radiance)
        const uint32_t ind_hit = hit_position * Nbands_l + b;
        float strength = (float)prd->strength *
                         (float)(face_top ? params.radiation_out_top[ind_hit]
                                          : params.radiation_out_bottom[ind_hit]);

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
                strength += source_radiance * (float)prd->strength;
            }
        }

        // Specular contribution (only if enabled and on iteration 0)
        float strength_spec = 0.0f;
        if (params.specular_reflection_enabled > 0 &&
            params.specular_exponent && params.specular_exponent[hit_position] > 0.f &&
            params.scattering_iteration == 0 &&
            params.radiation_specular) {
            for (uint32_t rr = 0; rr < params.Nsources; rr++) {
                const uint32_t ind_spec = rr * params.Ncameras * Nprims * Nbands_l
                                        + params.camera_ID * Nprims * Nbands_l
                                        + hit_position * Nbands_l + b;
                const float spec = params.radiation_specular[ind_spec] * 0.25f;
                if (spec > 0.0f) {
                    float3 light_dir;
                    if (params.source_types[rr] == 0 || params.source_types[rr] == 2) {
                        light_dir = normalize(params.source_positions[rr]);
                    } else {
                        const float3 hp = make_float3(ray_origin.x + t_hit * ray_direction.x,
                                                       ray_origin.y + t_hit * ray_direction.y,
                                                       ray_origin.z + t_hit * ray_direction.z);
                        light_dir = normalize(make_float3(params.source_positions[rr].x - hp.x,
                                                           params.source_positions[rr].y - hp.y,
                                                           params.source_positions[rr].z - hp.z));
                    }
                    const float3 spec_dir = normalize(light_dir - ray_direction);
                    const float exponent  = params.specular_exponent[hit_position];
                    float scale_coeff = 1.0f;
                    if (params.specular_reflection_enabled == 2 && params.specular_scale) {
                        scale_coeff = params.specular_scale[hit_position];
                    }
                    const float cos_spec = fmaxf(0.f, dot(spec_dir, normal));
                    strength_spec += spec * scale_coeff
                                   * powf(cos_spec, exponent) * (exponent + 2.f)
                                   / ((float)params.launch_dim_x * 2.f * M_PI);
                }
            }
        }

        // Accumulate into camera radiation buffer: [pixel][band]
        atomicFloatAdd(&params.radiation_in_camera[pixel_index * Nbands_l + b],
                       (strength + strength_spec) / M_PI);
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
        prd->hit_periodic_boundary = true;
        return;
    }

    // Store UUID+1 (0 is reserved for sky/miss)
    if (params.camera_pixel_label) {
        params.camera_pixel_label[origin_UUID] = hit_uuid + 1u;
    }

    // Depth: project ray parameter along camera view direction
    if (params.camera_pixel_depth) {
        const float  t_hit    = optixGetRayTmax() + (float)prd->strength; // strength=0 for pixel label
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

extern "C" __global__ void __raygen__direct() {
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

    const uint32_t linear_idx = xi + dim_x * yi;
    uint32_t seed = tea<16>(linear_idx + Nrays * prim_local, params.random_seed);

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
                        float uv_u, uv_v;
                        if (params.uv_IDs[prim_pos] >= 0) {
                            float2 uv0 = params.uv_data[prim_pos * 4 + 0];
                            float2 uv1 = params.uv_data[prim_pos * 4 + 1];
                            float2 uv2 = params.uv_data[prim_pos * 4 + 2];
                            uv_u = uv0.x + (sp.x + 0.5f) * (uv1.x - uv0.x);
                            uv_v = uv0.y + (sp.y + 0.5f) * (uv2.y - uv0.y);
                        } else {
                            uv_u = sp.x + 0.5f;
                            uv_v = sp.y + 0.5f;
                        }
                        if (sampleMask(msk_orig, uv_u, uv_v)) break;
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
                        float uv_u, uv_v;
                        if (params.uv_IDs[prim_pos] >= 0) {
                            float2 uv0 = params.uv_data[prim_pos * 4 + 0];
                            float2 uv1 = params.uv_data[prim_pos * 4 + 1];
                            float2 uv2 = params.uv_data[prim_pos * 4 + 2];
                            const float beta  = sp.y - sp.x; // weight at v1
                            const float gamma = sp.x;        // weight at v2
                            float2 uv = make_float2(
                                uv0.x + beta * (uv1.x - uv0.x) + gamma * (uv2.x - uv0.x),
                                uv0.y + beta * (uv1.y - uv0.y) + gamma * (uv2.y - uv0.y));
                            uv_u = uv.x;
                            uv_v = 1.f - uv.y; // Y-flip (matches intersection convention)
                        } else {
                            uv_u = sp.y;  // = beta + gamma
                            uv_v = sp.x;  // = gamma
                        }
                        if (sampleMask(msk_orig_t, uv_u, uv_v)) break;
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

            // Transform sample point to world space
            float3 ray_origin = sp;
            d_transformPoint(T, ray_origin);

            // Send a ray toward each source
            for (uint32_t rr = 0; rr < params.Nsources; rr++) {

                const uint32_t src_type = params.source_types[rr];

                float3 ray_direction;
                float  ray_tmax;
                double strength;

                if (src_type == 0) { // Collimated source
                    ray_direction = normalize(params.source_positions[rr]);
                    ray_tmax      = 1e38f;
                    strength = (1.0 / double(dim_x * dim_y)) * (double)fabsf(dot(normal, ray_direction));

                } else if (src_type == 1 || src_type == 2) { // Sphere source (type 1 = point, type 2 = sphere)
                    float theta_s = acos_safe(1.f - 2.f * rnd(seed));
                    float phi_s   = rnd(seed) * 2.f * M_PI;
                    float3 sphere_pt = make_float3(0.5f * params.source_widths[rr].x * sinf(theta_s) * cosf(phi_s),
                                                   0.5f * params.source_widths[rr].x * sinf(theta_s) * sinf(phi_s),
                                                   0.5f * params.source_widths[rr].x * cosf(theta_s));
                    ray_direction = sphere_pt + params.source_positions[rr] - ray_origin;
                    ray_tmax      = d_magnitude(ray_direction);
                    ray_direction = normalize(ray_direction);

                    // Integrate over sphere surface for strength
                    strength = 0.0;
                    const uint32_t N = 10;
                    for (uint32_t j = 0; j < N; j++) {
                        for (uint32_t i = 0; i < N; i++) {
                            float theta = acos_safe(1.f - 2.f * (float(i) + 0.5f) / float(N));
                            float phi   = (float(j) + 0.5f) * 2.f * M_PI / float(N);
                            float3 ldir = make_float3(sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta));
                            if (dot(ldir, ray_direction) < 0.f) {
                                strength += (1.0 / double(dim_x * dim_y)) * (double)fabsf(dot(normal, ray_direction))
                                          * (double)fabsf(dot(ldir, ray_direction))
                                          / ((double)ray_tmax * (double)ray_tmax)
                                          / double(N * N)
                                          * (double)params.source_widths[rr].x * (double)params.source_widths[rr].x;
                            }
                        }
                    }

                } else if (src_type == 3) { // Rectangle source
                    float light_transform[16];
                    float3 rot3 = params.source_rotations[rr];
                    d_makeTransformMatrix(rot3, light_transform);

                    float3 square_pt;
                    d_sampleSquare(seed, square_pt);
                    square_pt = make_float3(params.source_widths[rr].x * square_pt.x, params.source_widths[rr].y * square_pt.y, square_pt.z);
                    d_transformPoint(light_transform, square_pt);

                    float3 light_dir = make_float3(0.f, 0.f, 1.f);
                    d_transformPoint(light_transform, light_dir);

                    ray_direction = square_pt + params.source_positions[rr] - ray_origin;
                    if (dot(ray_direction, light_dir) > 0.f) continue; // don't emit from back of source

                    ray_tmax      = d_magnitude(ray_direction);
                    ray_direction = normalize(ray_direction);
                    strength = (1.0 / double(dim_x * dim_y))
                             * (double)fabsf(dot(normal, ray_direction))
                             * (double)fabsf(dot(light_dir, ray_direction))
                             / ((double)ray_tmax * (double)ray_tmax)
                             * (double)params.source_widths[rr].x * (double)params.source_widths[rr].y
                             / M_PI;

                } else if (src_type == 4) { // Disk source
                    float light_transform[16];
                    float3 rot3 = params.source_rotations[rr];
                    d_makeTransformMatrix(rot3, light_transform);

                    float3 disk_pt;
                    d_sampleDisk(seed, disk_pt);
                    d_transformPoint(light_transform, disk_pt);

                    float3 light_dir = make_float3(0.f, 0.f, 1.f);
                    d_transformPoint(light_transform, light_dir);

                    ray_direction = params.source_widths[rr].x * disk_pt + params.source_positions[rr] - ray_origin;
                    if (dot(ray_direction, light_dir) > 0.f) continue; // don't emit from back of source

                    ray_tmax      = d_magnitude(ray_direction);
                    ray_direction = normalize(ray_direction);
                    strength = (1.0 / double(dim_x * dim_y))
                             * (double)fabsf(dot(normal, ray_direction))
                             * (double)fabsf(dot(light_dir, ray_direction))
                             / ((double)ray_tmax * (double)ray_tmax)
                             * (double)params.source_widths[rr].x * (double)params.source_widths[rr].x;

                } else {
                    continue; // Unknown source type
                }

                PerRayData prd;
                prd.seed                 = seed;
                prd.origin_UUID          = UUID;
                prd.source_ID            = (unsigned char)rr;
                prd.hit_periodic_boundary = false;
                prd.face                 = (dot(ray_direction, normal) > 0.f);

                // Strength set above per source type
                prd.strength = strength;

                // Only fire from the face pointing toward source (or two-sided)
                const int8_t tsf = params.twosided_flag[prim_pos];
                if (!prd.face && tsf == 0) continue;
                if (tsf == 3) continue; // reserved flag — skip

                uint32_t u0, u1;
                float3 current_origin = ray_origin;

                for (int wrap = 0; wrap < 10; ++wrap) {
                    packPointer(&prd, u0, u1);
                    optixTrace(
                        params.traversable,
                        current_origin,
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
                    if (!prd.hit_periodic_boundary) break;
                    current_origin = prd.periodic_hit;
                    prd.hit_periodic_boundary = false;
                }

                seed = prd.seed;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Raygen: diffuse rays
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__diffuse() {
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

    // Seed once per ray index
    const uint32_t linear_idx = theta_idx + dim_x * phi_idx;
    uint32_t seed = tea<16>(linear_idx + dimxy * prim_local, params.random_seed);

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
                        float uv_u, uv_v;
                        if (params.uv_IDs[prim_pos] >= 0) {
                            float2 uv0 = params.uv_data[prim_pos * 4 + 0];
                            float2 uv1 = params.uv_data[prim_pos * 4 + 1];
                            float2 uv2 = params.uv_data[prim_pos * 4 + 2];
                            uv_u = uv0.x + (sp.x + 0.5f) * (uv1.x - uv0.x);
                            uv_v = uv0.y + (sp.y + 0.5f) * (uv2.y - uv0.y);
                        } else {
                            uv_u = sp.x + 0.5f;
                            uv_v = sp.y + 0.5f;
                        }
                        if (sampleMask(msk_orig_d, uv_u, uv_v)) break;
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
                        float uv_u, uv_v;
                        if (params.uv_IDs[prim_pos] >= 0) {
                            float2 uv0 = params.uv_data[prim_pos * 4 + 0];
                            float2 uv1 = params.uv_data[prim_pos * 4 + 1];
                            float2 uv2 = params.uv_data[prim_pos * 4 + 2];
                            const float beta  = sp.y - sp.x;
                            const float gamma = sp.x;
                            float2 uv = make_float2(
                                uv0.x + beta * (uv1.x - uv0.x) + gamma * (uv2.x - uv0.x),
                                uv0.y + beta * (uv1.y - uv0.y) + gamma * (uv2.y - uv0.y));
                            uv_u = uv.x;
                            uv_v = 1.f - uv.y;
                        } else {
                            uv_u = sp.y;
                            uv_v = sp.x;
                        }
                        if (sampleMask(msk_orig_dt, uv_u, uv_v)) break;
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

            // Rotate hemisphere direction by primitive normal orientation
            float3 ray_dir = d_rotatePoint(ray_dir_canonical,
                                           acos_safe(normal.z),
                                           atan2f(normal.y, normal.x));

            // Transform origin point to world space
            float3 ray_origin = sp;
            d_transformPoint(T, ray_origin);

            PerRayData prd;
            prd.seed                  = seed;
            prd.origin_UUID           = UUID;
            prd.source_ID             = 0;
            prd.hit_periodic_boundary = false;
            prd.strength              = 1.0 / double(dimxy);

            uint32_t u0, u1;

            if (params.launch_face == 1 && params.twosided_flag[prim_pos] != 3) {
                prd.face = true;
                float3 cur_origin = ray_origin;
                for (int wrap = 0; wrap < 10; ++wrap) {
                    packPointer(&prd, u0, u1);
                    optixTrace(
                        params.traversable,
                        cur_origin, ray_dir,
                        1e-4f, 1e38f, 0.f,
                        OptixVisibilityMask(255),
                        OPTIX_RAY_FLAG_NONE,
                        1, 0, 1, // SBT offset=1 (diffuse hit), stride=0, miss index=1 (diffuse miss)
                        u0, u1
                    );
                    if (!prd.hit_periodic_boundary) break;
                    cur_origin = prd.periodic_hit;
                    prd.hit_periodic_boundary = false;
                }
            } else if (params.launch_face == 0 && params.twosided_flag[prim_pos] == 1) {
                prd.face = false;
                float3 neg_dir = make_float3(-ray_dir.x, -ray_dir.y, -ray_dir.z);
                float3 cur_origin = ray_origin;
                for (int wrap = 0; wrap < 10; ++wrap) {
                    packPointer(&prd, u0, u1);
                    optixTrace(
                        params.traversable,
                        cur_origin, neg_dir,
                        1e-4f, 1e38f, 0.f,
                        OptixVisibilityMask(255),
                        OPTIX_RAY_FLAG_NONE,
                        1, 0, 1,
                        u0, u1
                    );
                    if (!prd.hit_periodic_boundary) break;
                    cur_origin = prd.periodic_hit;
                    prd.hit_periodic_boundary = false;
                }
            }

            seed = prd.seed;
        }
    }
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
    prd.face                 = true;
    prd.source_ID            = 0;
    prd.seed                 = seed;
    prd.hit_periodic_boundary = false;

    uint32_t p0, p1;
    packPointer(&prd, p0, p1);

    const float t_min = 1e-5f;
    const float t_max = 1e30f;

    float3 cur_origin = ray_origin;
    for (int wrap = 0; wrap < 10; wrap++) {
        prd.hit_periodic_boundary = false;
        optixTrace(params.traversable, cur_origin, ray_direction,
                   t_min, t_max, 0.f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                   2u, 0u, 2u, // SBT: offset=2 (camera hit), stride=0, miss=2
                   p0, p1);
        if (!prd.hit_periodic_boundary) break;
        cur_origin = prd.periodic_hit;
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
    prd.strength             = 0.f; // used as distance offset (always 0 for pixel label)
    prd.origin_UUID          = pixel_index;
    prd.face                 = true;
    prd.source_ID            = 0;
    prd.seed                 = seed;
    prd.hit_periodic_boundary = false;

    uint32_t p0, p1;
    packPointer(&prd, p0, p1);

    const float t_min = 1e-5f;
    const float t_max = 1e30f;

    float3 cur_origin = ray_origin;
    for (int wrap = 0; wrap < 10; wrap++) {
        prd.hit_periodic_boundary = false;
        optixTrace(params.traversable, cur_origin, ray_direction,
                   t_min, t_max, 0.f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                   3u, 0u, 3u, // SBT: offset=3 (pixel label hit), stride=0, miss=3
                   p0, p1);
        if (!prd.hit_periodic_boundary) break;
        cur_origin = prd.periodic_hit;
    }
}
