/** \file "OptiX8Math.h" Inline float3/float2 math helpers for OptiX 8.1 device code.

    OptiX 8 removed the optixu:: math namespace present in OptiX 6.5.
    This header provides the minimal set of operators and functions needed
    by the Helios radiation model device programs.

    Copyright (C) 2016-2026 Brian Bailey
*/

#ifndef OPTIX8_MATH_H
#define OPTIX8_MATH_H

#include <vector_types.h> // float3, float2, int2, etc. (from CUDA runtime)

// ---------------------------------------------------------------------------
// float3 arithmetic operators
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ __forceinline__ float3 operator*(const float3 &a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ __forceinline__ float3 operator*(float s, const float3 &a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ __forceinline__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ __forceinline__ float3 operator/(const float3 &a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__host__ __device__ __forceinline__ float3 operator-(const float3 &a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__host__ __device__ __forceinline__ float3 &operator+=(float3 &a, const float3 &b) {
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
__host__ __device__ __forceinline__ float3 &operator-=(float3 &a, const float3 &b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}
__host__ __device__ __forceinline__ float3 &operator*=(float3 &a, float s) {
    a.x *= s; a.y *= s; a.z *= s; return a;
}

// ---------------------------------------------------------------------------
// float2 arithmetic operators
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__host__ __device__ __forceinline__ float2 operator-(const float2 &a, const float2 &b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__host__ __device__ __forceinline__ float2 operator*(const float2 &a, float s) {
    return make_float2(a.x * s, a.y * s);
}
__host__ __device__ __forceinline__ float2 operator*(float s, const float2 &a) {
    return make_float2(a.x * s, a.y * s);
}

// ---------------------------------------------------------------------------
// Math functions on float3
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__host__ __device__ __forceinline__ float length(const float3 &v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ __forceinline__ float3 normalize(const float3 &v) {
    float inv_len = 1.0f / length(v);
    return v * inv_len;
}

__host__ __device__ __forceinline__ float dot(const float2 &a, const float2 &b) {
    return a.x * b.x + a.y * b.y;
}

#endif // OPTIX8_MATH_H
