/** \file "CollisionDetection.cu" CUDA source file for GPU-accelerated collision detection

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

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>

#include "helios_vector_types.h"

// Forward declaration to avoid pulling the full helios global header (and its template-heavy transitive includes)
// into the CUDA translation unit. Defined in core/src/global.cpp.
namespace helios {
    void helios_runtime_error(const std::string &error_message);
}

//! Fail-fast wrapper for CUDA runtime calls. Helios policy forbids silent fallbacks: a failed CUDA call throws a
//! helios_runtime_error rather than the legacy fprintf(stderr)+return, which used to leave output arrays partially
//! written. Used by the resident-scene launch path below.
#define HELIOS_CUDA_CHECK(call)                                                                                                                                                                                                                 \
    do {                                                                                                                                                                                                                                        \
        cudaError_t _helios_cuda_err = (call);                                                                                                                                                                                                  \
        if (_helios_cuda_err != cudaSuccess) {                                                                                                                                                                                                  \
            helios::helios_runtime_error(std::string("CUDA error (") + #call + "): " + cudaGetErrorString(_helios_cuda_err));                                                                                                                    \
        }                                                                                                                                                                                                                                       \
    } while (0)

/**
 * \brief Owning handle for a temporary device allocation, freed when the handle goes out of scope.
 *
 * The launch wrappers allocate several per-call buffers and throw (HELIOS_CUDA_CHECK) on any CUDA failure. Holding each
 * buffer in a DeviceBuffer frees every allocation made before the failing call instead of leaking it.
 */
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) {
        allocate(count);
    }
    ~DeviceBuffer() {
        if (device_pointer != nullptr) {
            cudaFree(device_pointer);
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    //! Allocate room for \p count elements, releasing any previous allocation first.
    void allocate(size_t count) {
        if (device_pointer != nullptr) {
            cudaFree(device_pointer);
            device_pointer = nullptr;
        }
        HELIOS_CUDA_CHECK(cudaMalloc(&device_pointer, count * sizeof(T)));
    }

    //! Device pointer, or null if nothing has been allocated.
    T *get() const {
        return device_pointer;
    }

private:
    T *device_pointer = nullptr;
};

/**
 * \brief GPU-friendly BVH node structure (Legacy AoS format)
 *
 * This structure is optimized for GPU memory access patterns
 * and maintains 64-byte alignment for optimal performance.
 */
struct GPUBVHNode {
    float3 aabb_min; //!< Minimum corner of AABB (12 bytes)
    float3 aabb_max; //!< Maximum corner of AABB (12 bytes)
    unsigned int left_child; //!< Index of left child (0xFFFFFFFF if leaf) (4 bytes)
    unsigned int right_child; //!< Index of right child (0xFFFFFFFF if leaf) (4 bytes)
    unsigned int primitive_start; //!< Starting index in primitive array (4 bytes)
    unsigned int primitive_count; //!< Number of primitives (0 for internal nodes) (4 bytes)
    unsigned int is_leaf; //!< 1 if leaf, 0 if internal node (4 bytes)
    unsigned int padding; //!< Padding for alignment (4 bytes)
};

//! Per-thread BVH traversal stack capacity. An iterative DFS that pushes both children holds at most ~one node per tree
//! level, so the capacity must exceed the host builder's MAX_DEPTH (64, see buildBVHRecursive). 128 mirrors the CPU
//! STACK_CAPACITY (CollisionDetection_RayTracing.cpp) and leaves generous headroom; the guarded push in each kernel is a
//! defensive backstop that must never actually trigger for a valid tree. The stack lives in per-thread local memory (it is
//! thread-private, so the previous block-shared array gave no sharing benefit while capping occupancy at 32 KiB/block).
#define BVH_TRAVERSAL_STACK_CAPACITY 128

//! Device flag raised (atomically) by a traversal kernel if a child push would exceed BVH_TRAVERSAL_STACK_CAPACITY. The
//! host launch wrappers reset it before launch and throw helios_runtime_error after the kernel if it is set, mirroring the
//! CPU invariant that a valid SAH tree never overflows the stack (fail-fast instead of silently dropping primitives).
__device__ unsigned int d_bvh_stack_overflow = 0;

/**
 * \brief GPU-optimized SoA BVH structure for warp-efficient traversal
 *
 * This structure stores BVH data in Structure-of-Arrays layout for optimal
 * GPU memory coalescing and SIMD processing within warps.
 */
struct GPUBVHNodesSoA {
    // Hot data: frequently accessed during traversal (separate arrays for coalescing)
    float3 *aabb_mins; //!< Array of AABB minimum corners
    float3 *aabb_maxs; //!< Array of AABB maximum corners
    uint32_t *left_children; //!< Array of left child indices
    uint32_t *right_children; //!< Array of right child indices

    // Cold data: accessed less frequently
    uint32_t *primitive_starts; //!< Array of primitive starting indices
    uint32_t *primitive_counts; //!< Array of primitive counts
    uint8_t *is_leaf_flags; //!< Array of leaf flags (packed)

    size_t node_count; //!< Total number of nodes
};

/**
 * \brief CUDA device function to test AABB intersection
 * \param[in] min1 Minimum corner of first AABB
 * \param[in] max1 Maximum corner of first AABB
 * \param[in] min2 Minimum corner of second AABB
 * \param[in] max2 Maximum corner of second AABB
 * \return True if AABBs intersect
 */
__device__ bool d_aabbIntersect(const float3 &min1, const float3 &max1, const float3 &min2, const float3 &max2) {
    return (min1.x <= max2.x && max1.x >= min2.x) && (min1.y <= max2.y && max1.y >= min2.y) && (min1.z <= max2.z && max1.z >= min2.z);
}

/**
 * \brief CUDA device helper functions for vector operations
 */
__device__ __forceinline__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalize(const float3 &v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-8f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0.0f, 0.0f, 1.0f); // Default up vector
}

__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3 &a, float scalar) {
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

/**
 * \brief Fast ray-triangle intersection using Möller-Trumbore algorithm
 * \param[in] ray_origin Ray starting point
 * \param[in] ray_direction Ray direction vector (normalized)
 * \param[in] v0 First triangle vertex
 * \param[in] v1 Second triangle vertex
 * \param[in] v2 Third triangle vertex
 * \param[in] max_distance Maximum ray distance
 * \param[out] hit_distance Distance to intersection (if hit)
 * \return True if ray intersects triangle within max_distance
 */
__device__ __forceinline__ bool rayTriangleIntersect(const float3 &ray_origin, const float3 &ray_direction, const float3 &v0, const float3 &v1, const float3 &v2, float max_distance, float &hit_distance) {
    const float EPSILON = 1e-5f;  // Match CPU tolerance for consistent GPU/CPU behavior

    // Find vectors for two edges sharing v0
    float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

    // Begin calculating determinant - also used to calculate u parameter
    float3 h = make_float3(ray_direction.y * edge2.z - ray_direction.z * edge2.y, ray_direction.z * edge2.x - ray_direction.x * edge2.z, ray_direction.x * edge2.y - ray_direction.y * edge2.x);

    // If determinant is near zero, ray lies in plane of triangle. The determinant scales with the triangle's area, so it is compared against the normal's
    // length: the test is then on the cosine between ray and normal alone, and a millimetre-scale triangle is not mistaken for a parallel one.
    float a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;
    const float3 triangle_normal = cross(edge1, edge2);
    if (a * a <= EPSILON * EPSILON * dot(triangle_normal, triangle_normal)) {
        return false; // This ray is parallel to this triangle.
    }

    float f = 1.0f / a;
    float3 s = make_float3(ray_origin.x - v0.x, ray_origin.y - v0.y, ray_origin.z - v0.z);
    float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);

    if (u < -EPSILON || u > 1.0f + EPSILON) {
        return false;
    }

    float3 q = make_float3(s.y * edge1.z - s.z * edge1.y, s.z * edge1.x - s.x * edge1.z, s.x * edge1.y - s.y * edge1.x);

    float v = f * (ray_direction.x * q.x + ray_direction.y * q.y + ray_direction.z * q.z);

    if (v < -EPSILON || u + v > 1.0f + EPSILON) {
        return false;
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

    // Match CPU behavior - don't check max_distance here (checked by caller)
    if (t > EPSILON) { // ray intersection
        hit_distance = t;
        return true;
    } else { // This means that there is a line intersection but not a ray intersection.
        return false;
    }
}

//! Sign-preserving reciprocal of a ray direction for slab-method AABB tests. Clamps each near-zero component to
//! +/-PARALLEL_EPS before the reciprocal so an axis-aligned ray lying exactly on a box face computes
//! (bound - origin == 0) * huge == 0 (finite) instead of (bound - origin == 0) * inf == NaN, which would corrupt the
//! fminf/fmaxf slab interval and spuriously reject a box the ray actually crosses. Mirrors the CPU aabbIntersectSoA SSE
//! path (CollisionDetection_RayTracing.cpp).
__device__ __forceinline__ float3 safeRayInvDir(const float3 &dir) {
    constexpr float PARALLEL_EPS = 1e-8f;
    float dx = (fabsf(dir.x) < PARALLEL_EPS) ? copysignf(PARALLEL_EPS, dir.x) : dir.x;
    float dy = (fabsf(dir.y) < PARALLEL_EPS) ? copysignf(PARALLEL_EPS, dir.y) : dir.y;
    float dz = (fabsf(dir.z) < PARALLEL_EPS) ? copysignf(PARALLEL_EPS, dir.z) : dir.z;
    return make_float3(1.0f / dx, 1.0f / dy, 1.0f / dz);
}

__device__ __forceinline__ bool warpRayAABBIntersect(const float3 &ray_origin, const float3 &ray_dir, const float3 &aabb_min, const float3 &aabb_max, float max_dist) {
    // Optimized ray-AABB intersection using slab method
    // Compute intersection distances for each axis (sign-preserving reciprocal avoids 0*inf=NaN for axis-aligned rays).
    float3 inv_dir = safeRayInvDir(ray_dir);

    float3 t_min = make_float3((aabb_min.x - ray_origin.x) * inv_dir.x, (aabb_min.y - ray_origin.y) * inv_dir.y, (aabb_min.z - ray_origin.z) * inv_dir.z);

    float3 t_max = make_float3((aabb_max.x - ray_origin.x) * inv_dir.x, (aabb_max.y - ray_origin.y) * inv_dir.y, (aabb_max.z - ray_origin.z) * inv_dir.z);

    // Handle negative ray directions
    if (ray_dir.x < 0.0f) {
        float temp = t_min.x;
        t_min.x = t_max.x;
        t_max.x = temp;
    }
    if (ray_dir.y < 0.0f) {
        float temp = t_min.y;
        t_min.y = t_max.y;
        t_max.y = temp;
    }
    if (ray_dir.z < 0.0f) {
        float temp = t_min.z;
        t_min.z = t_max.z;
        t_max.z = temp;
    }

    // Find the intersection interval
    float t_enter = fmaxf(fmaxf(t_min.x, t_min.y), t_min.z);
    float t_exit = fminf(fminf(t_max.x, t_max.y), t_max.z);

    // Check if intersection exists and is within ray limits
    return (t_enter <= t_exit) && (t_exit >= 0.0f) && (t_enter <= max_dist);
}

// Device function for ray-triangle intersection (same algorithm as CPU)
__device__ __forceinline__ bool rayTriangleIntersectCPU(const float3 &origin, const float3 &direction, const float3 &v0, const float3 &v1, const float3 &v2, float &distance) {

    // Use same algorithm as CPU: radiation model's triangle_intersect
    const float EPSILON = 1e-5f;  // Match CPU tolerance for consistent GPU/CPU behavior

    float a = v0.x - v1.x, b = v0.x - v2.x, c = direction.x, d = v0.x - origin.x;
    float e = v0.y - v1.y, f = v0.y - v2.y, g = direction.y, h = v0.y - origin.y;
    float i = v0.z - v1.z, j = v0.z - v2.z, k = direction.z, l = v0.z - origin.z;

    float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
    float q = g * i - e * k, s = e * j - f * i;

    float denom = a * m + b * q + c * s;
    // Compared against the triangle normal's length (s is its x-component) so the parallel test depends on the ray's angle, not the triangle's size.
    const float normal_y = i * b - a * j;
    const float normal_z = a * f - e * b;
    if (denom * denom <= EPSILON * EPSILON * (s * s + normal_y * normal_y + normal_z * normal_z)) {
        return false; // Ray is parallel to triangle
    }

    float inv_denom = 1.0f / denom;

    float e1 = d * m - b * n - c * p;
    float beta = e1 * inv_denom;

    if (beta >= -EPSILON) {
        float r = e * l - h * i;
        float e2 = a * n + d * q + c * r;
        float gamma = e2 * inv_denom;

        if (gamma >= -EPSILON && beta + gamma <= 1.0f + EPSILON) {
            float e3 = a * p - b * r + d * s;
            float t = e3 * inv_denom;

            if (t > EPSILON) {
                distance = t;
                return true;
            }
        }
    }
    return false;
}

// Device function for ray-patch intersection (same algorithm as CPU)
__device__ __forceinline__ bool rayPatchIntersect(const float3 &origin, const float3 &direction, const float3 &v0, const float3 &v1, const float3 &v2, const float3 &v3, float &distance) {

    // Calculate patch vectors and normal. Tolerances match the CPU patchIntersect() (CollisionDetection_RayTracing.cpp).
    const float EPSILON = 1e-8f;

    float3 anchor = v0;
    float3 normal = cross(v1 - v0, v2 - v0);
    normal = normalize(normal);

    float3 a = v1 - v0; // First edge vector
    float3 b = v3 - v0; // Second edge vector

    // Ray-plane intersection
    float denom = dot(direction, normal);
    if (fabsf(denom) > EPSILON) { // Not parallel to plane
        float t = dot(anchor - origin, normal) / denom;

        if (t > EPSILON && t < 1e8f) { // Valid intersection distance
            // Find intersection point
            float3 p = origin + direction * t;
            float3 d = p - anchor;

            // Project onto patch coordinate system
            float ddota = dot(d, a);
            float ddotb = dot(d, b);

            // Check if point is within patch bounds
            if (ddota >= -EPSILON && ddota <= dot(a, a) + EPSILON && ddotb >= -EPSILON && ddotb <= dot(b, b) + EPSILON) {

                distance = t;
                return true;
            }
        }
    }
    return false;
}

// Ray-AABB intersection for voxel primitives
__device__ bool rayVoxelIntersect(const float3 &ray_origin, const float3 &ray_direction, const float3 &aabb_min, const float3 &aabb_max, float &distance) {
    const float EPSILON = 1e-6f; // matches the CPU voxel branch of intersectCachedPrimitive()

    // Calculate t values for each slab (sign-preserving reciprocal avoids 0*inf=NaN for axis-aligned rays grazing a face).
    float3 inv_dir = safeRayInvDir(ray_direction);

    float3 t_min = make_float3((aabb_min.x - ray_origin.x) * inv_dir.x, (aabb_min.y - ray_origin.y) * inv_dir.y, (aabb_min.z - ray_origin.z) * inv_dir.z);

    float3 t_max = make_float3((aabb_max.x - ray_origin.x) * inv_dir.x, (aabb_max.y - ray_origin.y) * inv_dir.y, (aabb_max.z - ray_origin.z) * inv_dir.z);

    // Handle negative ray directions
    if (ray_direction.x < 0.0f) {
        float temp = t_min.x;
        t_min.x = t_max.x;
        t_max.x = temp;
    }
    if (ray_direction.y < 0.0f) {
        float temp = t_min.y;
        t_min.y = t_max.y;
        t_max.y = temp;
    }
    if (ray_direction.z < 0.0f) {
        float temp = t_min.z;
        t_min.z = t_max.z;
        t_max.z = temp;
    }

    // Find the intersection interval
    float t_enter = fmaxf(fmaxf(t_min.x, t_min.y), t_min.z);
    float t_exit = fminf(fminf(t_max.x, t_max.y), t_max.z);

    // Check for intersection
    if (t_enter > t_exit || t_exit < 0.0f) {
        return false; // No intersection or behind ray
    }

    // Set distance to entry point (or exit if ray starts inside)
    distance = (t_enter >= EPSILON) ? t_enter : t_exit;

    return distance >= EPSILON;
}

/**
 * \brief Compute the surface normal for a hit primitive, matching the CPU intersectPrimitiveThreadSafe convention.
 *
 * Triangles/patches use normalize(cross(v1-v0, v2-v0)) face-forwarded toward the ray origin (n flipped so it points
 * back at the ray), exactly as the CPU SoA path does, so GPU and CPU normals (and the LiDAR hit_fnorm = dot(dir,n)
 * derived from them) agree. Voxels use the axis-aligned face normal of the hit face (no face-forward, matching CPU).
 * Degenerate/unknown primitives fall back to normalize(-direction).
 */
__device__ __forceinline__ float3 computeHitNormal(int ptype, const float3 *d_primitive_vertices, unsigned int vertex_offset, const float3 &ray_origin, const float3 &ray_direction, float hit_distance) {
    float3 fallback = normalize(make_float3(-ray_direction.x, -ray_direction.y, -ray_direction.z));

    if (ptype == 1 || ptype == 0) { // triangle or patch: cross of first two edges, face-forwarded
        float3 v0 = d_primitive_vertices[vertex_offset + 0];
        float3 v1 = d_primitive_vertices[vertex_offset + 1];
        float3 v2 = d_primitive_vertices[vertex_offset + 2];
        float3 n = cross(v1 - v0, v2 - v0);
        float mag = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        if (mag > 1e-8f) {
            n = make_float3(n.x / mag, n.y / mag, n.z / mag);
            float3 hit_point = make_float3(ray_origin.x + ray_direction.x * hit_distance, ray_origin.y + ray_direction.y * hit_distance, ray_origin.z + ray_direction.z * hit_distance);
            float3 to_origin = make_float3(ray_origin.x - hit_point.x, ray_origin.y - hit_point.y, ray_origin.z - hit_point.z);
            if (dot(n, to_origin) < 0.0f) {
                n = make_float3(-n.x, -n.y, -n.z);
            }
            return n;
        }
        return fallback;
    } else if (ptype == 2) { // voxel: axis-aligned face normal of the hit face
        float3 vmin = d_primitive_vertices[vertex_offset + 0];
        float3 vmax = d_primitive_vertices[vertex_offset + 1];
        float3 hit_point = make_float3(ray_origin.x + ray_direction.x * hit_distance, ray_origin.y + ray_direction.y * hit_distance, ray_origin.z + ray_direction.z * hit_distance);
        float3 center = make_float3((vmin.x + vmax.x) * 0.5f, (vmin.y + vmax.y) * 0.5f, (vmin.z + vmax.z) * 0.5f);
        float3 extent = make_float3((vmax.x - vmin.x) * 0.5f, (vmax.y - vmin.y) * 0.5f, (vmax.z - vmin.z) * 0.5f);
        float3 local = make_float3(hit_point.x - center.x, hit_point.y - center.y, hit_point.z - center.z);
        float rel_x = fabsf(local.x) / extent.x;
        float rel_y = fabsf(local.y) / extent.y;
        float rel_z = fabsf(local.z) / extent.z;
        if (rel_x >= rel_y && rel_x >= rel_z) {
            return make_float3((local.x > 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
        } else if (rel_y >= rel_z) {
            return make_float3(0.0f, (local.y > 0.0f) ? 1.0f : -1.0f, 0.0f);
        } else {
            return make_float3(0.0f, 0.0f, (local.z > 0.0f) ? 1.0f : -1.0f);
        }
    }
    return fallback;
}

//! Sample a texture transparency mask at UV (u,v); returns true if the texel is opaque (hit should count), false if
//! transparent (ray passes through). Mirrors the CPU CollisionDetection::isHitTexelOpaque() texel lookup: wrap UV into
//! [0,1), px = u*width, py = (1-v)*height (mask rows are top-to-bottom, UV y=0 is the bottom), clamp, fetch the byte.
__device__ __forceinline__ bool sampleMaskOpaqueGPU(int mask_id, float u, float v, const unsigned char *d_mask_data, const unsigned int *d_mask_offsets, const int *d_mask_sizes) {
    if (mask_id < 0 || d_mask_data == nullptr) {
        return true; // no mask -> fully solid
    }
    const int width = d_mask_sizes[mask_id * 2];
    const int height = d_mask_sizes[mask_id * 2 + 1];
    if (width <= 0 || height <= 0) {
        return true;
    }
    const unsigned int offset = d_mask_offsets[mask_id];

    u -= floorf(u); // wrap repeat-style mappings into [0,1)
    v -= floorf(v);

    int px = (int) (u * (float) width);
    px = max(0, min(px, width - 1));
    int py = (int) ((1.f - v) * (float) height);
    py = max(0, min(py, height - 1));

    return d_mask_data[offset + (unsigned int) (py * width + px)] != 0u;
}

//! Decide whether a ray-primitive hit lands on an opaque texel. Computes the (u,v) at the hit point with the SAME
//! interpolation as the CPU CollisionDetection::isHitTexelOpaque() (patch: BL->BR / BL->TL basis projection; triangle:
//! barycentric Cramer solve) so the GPU and CPU synthetic scans reject identical texels. Voxels / mask-less primitives
//! and degenerate/missing-UV configurations are treated as solid, exactly as the CPU path does.
__device__ __forceinline__ bool isHitOpaqueGPU(int ptype, const float3 *verts, int mask_id, int uv_id, const float *uv4, const float3 &hit_point, const unsigned char *d_mask_data, const unsigned int *d_mask_offsets, const int *d_mask_sizes) {
    if (mask_id < 0) {
        return true;
    }
    float u, v;
    if (ptype == 0) { // patch: corners (BL, BR, TR, TL)
        const float3 v0 = verts[0];
        const float3 e1 = verts[1] - v0;
        const float3 e2 = verts[3] - v0;
        const float3 d = hit_point - v0;
        const float e1_sq = dot(e1, e1);
        const float e2_sq = dot(e2, e2);
        float s = (e1_sq > 0.f) ? dot(d, e1) / e1_sq : 0.f;
        float t = (e2_sq > 0.f) ? dot(d, e2) / e2_sq : 0.f;
        s = fminf(fmaxf(s, 0.f), 1.f);
        t = fminf(fmaxf(t, 0.f), 1.f);
        if (uv_id >= 0) {
            u = (1.f - s) * (1.f - t) * uv4[0] + s * (1.f - t) * uv4[2] + s * t * uv4[4] + (1.f - s) * t * uv4[6];
            v = (1.f - s) * (1.f - t) * uv4[1] + s * (1.f - t) * uv4[3] + s * t * uv4[5] + (1.f - s) * t * uv4[7];
        } else {
            u = s;
            v = t;
        }
    } else if (ptype == 1) { // triangle
        if (uv_id < 0) {
            return true; // no UVs -> cannot map texel, treat as solid (matches CPU)
        }
        const float3 v0 = verts[0];
        const float3 e1 = verts[1] - v0;
        const float3 e2 = verts[2] - v0;
        const float3 d = hit_point - v0;
        const float dot11 = dot(e1, e1);
        const float dot12 = dot(e1, e2);
        const float dot22 = dot(e2, e2);
        const float dot1d = dot(e1, d);
        const float dot2d = dot(e2, d);
        const float denom = dot11 * dot22 - dot12 * dot12;
        if (fabsf(denom) < 1e-20f) {
            return true; // degenerate triangle
        }
        const float inv = 1.f / denom;
        const float beta = (dot22 * dot1d - dot12 * dot2d) * inv;
        const float gamma = (dot11 * dot2d - dot12 * dot1d) * inv;
        u = uv4[0] + beta * (uv4[2] - uv4[0]) + gamma * (uv4[4] - uv4[0]);
        v = uv4[1] + beta * (uv4[3] - uv4[1]) + gamma * (uv4[5] - uv4[1]);
    } else {
        return true; // voxel / unknown -> solid
    }
    return sampleMaskOpaqueGPU(mask_id, u, v, d_mask_data, d_mask_offsets, d_mask_sizes);
}

__global__ void rayPrimitiveBVHKernel(GPUBVHNode *d_bvh_nodes, unsigned int *d_primitive_indices,
                                      int *d_primitive_types, // Type of each primitive (int for GPU compatibility)
                                      float3 *d_primitive_vertices, // All vertices for all primitives (variable count per primitive)
                                      unsigned int *d_vertex_offsets, // Starting index in vertices array for each primitive
                                      const unsigned char *d_mask_data, // Texture transparency: flattened mask bytes (or null)
                                      const unsigned int *d_mask_offsets, // Per-mask start index into d_mask_data
                                      const int *d_mask_sizes, // Per-mask width/height (2 ints per mask)
                                      const int *d_mask_IDs, // Per-primitive mask index (-1 = none)
                                      const float *d_uv_data, // Per-primitive UVs (4 vec2 = 8 floats per primitive)
                                      const int *d_uv_IDs, // Per-primitive UV flag (-1 = parametric/none)
                                      float3 *d_ray_origins, float3 *d_ray_directions, float *d_ray_max_distances, float uniform_max_distance, int num_rays, int primitive_count, int total_vertex_count, float *d_hit_distances,
                                      unsigned int *d_hit_primitive_ids, unsigned int *d_hit_counts, float3 *d_hit_normals, bool find_closest_hit) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_idx >= num_rays) {
        return;
    }

    // Load ray data. Directions are normalized here (the host passes them raw) so the returned t is a geometric
    // distance, matching the CPU RayQuery semantics. d_ray_max_distances may be null, in which case every ray shares
    // uniform_max_distance (avoids a per-ray host/device array when the caller's cutoff is constant).
    float3 ray_origin = d_ray_origins[ray_idx];
    float3 ray_direction = d_ray_directions[ray_idx];
    float dmag = sqrtf(ray_direction.x * ray_direction.x + ray_direction.y * ray_direction.y + ray_direction.z * ray_direction.z);
    if (dmag > 1e-8f) {
        ray_direction = make_float3(ray_direction.x / dmag, ray_direction.y / dmag, ray_direction.z / dmag);
    }
    float ray_max_distance = (d_ray_max_distances != nullptr) ? d_ray_max_distances[ray_idx] : uniform_max_distance;

    // Initialize hit data
    float closest_hit_distance = ray_max_distance + 1.0f; // Initialize beyond max
    unsigned int hit_primitive_id = 0xFFFFFFFF; // Invalid ID
    unsigned int total_hits = 0;
    int best_vertex_offset = -1; // packed-vertex offset of the current closest hit (for normal computation)
    int best_ptype = -1; // primitive type of the current closest hit

    // Stack-based BVH traversal. Per-thread local stack (thread-private; sized to BVH_TRAVERSAL_STACK_CAPACITY so it exceeds
    // the host builder's MAX_DEPTH). The old block-shared 32-entry-per-thread array silently dropped children once the new
    // deeper SAH trees pushed past depth 31, causing missed/farther hits that diverged from the CPU result.
    unsigned int thread_stack[BVH_TRAVERSAL_STACK_CAPACITY];
    int stack_size = 0;

    // Start from root node. A zero direction has no ray to trace, so it skips the traversal and misses, as on the CPU.
    if (dmag > 1e-8f) {
        thread_stack[0] = 0;
        stack_size = 1;
    }

    // Main traversal loop
    while (stack_size > 0) {
        // Pop node from stack
        stack_size--;
        unsigned int node_idx = thread_stack[stack_size];

        if (node_idx == 0xFFFFFFFF) {
            continue;
        }

        GPUBVHNode node = d_bvh_nodes[node_idx];

        // Test ray-AABB intersection first (early rejection)
        if (!warpRayAABBIntersect(ray_origin, ray_direction, node.aabb_min, node.aabb_max, ray_max_distance)) {
            continue;
        }

        if (node.is_leaf) {
            // Test ray against all triangles in this leaf
            for (unsigned int i = 0; i < node.primitive_count; i++) {
                unsigned int primitive_index = node.primitive_start + i;

                // Bounds check for primitive_index
                if (primitive_index >= primitive_count) {
                    continue; // Skip this primitive, don't exit the entire kernel!
                }

                unsigned int primitive_id = d_primitive_indices[primitive_index];

                // Get primitive type
                int ptype = d_primitive_types[primitive_index];

                // Get primitive vertices starting index
                unsigned int vertex_offset = d_vertex_offsets[primitive_index];

                // Test ray-primitive intersection based on type
                float hit_distance;
                bool hit = false;

                if (ptype == 1) { // PRIMITIVE_TYPE_TRIANGLE
                    if (vertex_offset + 2 >= total_vertex_count) {
                        continue; // Skip this primitive, don't exit the entire kernel!
                    }

                    float3 v0 = d_primitive_vertices[vertex_offset + 0];
                    float3 v1 = d_primitive_vertices[vertex_offset + 1];
                    float3 v2 = d_primitive_vertices[vertex_offset + 2];

                    hit = rayTriangleIntersect(ray_origin, ray_direction, v0, v1, v2, ray_max_distance, hit_distance);

                } else if (ptype == 0) { // PRIMITIVE_TYPE_PATCH
                    if (vertex_offset + 3 >= total_vertex_count) {
                        continue; // Skip this primitive, don't exit the entire kernel!
                    }

                    float3 v0 = d_primitive_vertices[vertex_offset + 0];
                    float3 v1 = d_primitive_vertices[vertex_offset + 1];
                    float3 v2 = d_primitive_vertices[vertex_offset + 2];
                    float3 v3 = d_primitive_vertices[vertex_offset + 3];

                    hit = rayPatchIntersect(ray_origin, ray_direction, v0, v1, v2, v3, hit_distance);

                } else if (ptype == 2) { // PRIMITIVE_TYPE_VOXEL
                    // Voxel intersection using AABB intersection
                    // For voxels, vertices store min/max coordinates: [min.x, min.y, min.z, max.x, max.y, max.z, 0, 0]
                    if (vertex_offset + 1 >= total_vertex_count) {
                        continue; // Skip this primitive, don't exit the entire kernel!
                    }

                    float3 voxel_min = d_primitive_vertices[vertex_offset + 0];
                    float3 voxel_max = d_primitive_vertices[vertex_offset + 1];

                    hit = rayVoxelIntersect(ray_origin, ray_direction, voxel_min, voxel_max, hit_distance);
                }

                // Process hit if found

                // Same acceptance as the CPU intersectCachedPrimitive(): a minimum distance of 1e-6 and a strict max_distance.
                if (hit && hit_distance > 1e-6f && hit_distance < ray_max_distance) {

                    // Texture transparency: if this primitive has a mask and the hit lands on a transparent texel, the ray
                    // passes through it (skip without recording) so traversal continues to geometry behind, matching the
                    // CPU isHitTexelOpaque() path. mask_IDs/uv_data are indexed by primitive_index (BVH order), like vertices.
                    const int mask_id = (d_mask_IDs != nullptr) ? d_mask_IDs[primitive_index] : -1;
                    if (mask_id >= 0) {
                        const float3 hit_point = make_float3(ray_origin.x + ray_direction.x * hit_distance, ray_origin.y + ray_direction.y * hit_distance, ray_origin.z + ray_direction.z * hit_distance);
                        if (!isHitOpaqueGPU(ptype, &d_primitive_vertices[vertex_offset], mask_id, d_uv_IDs[primitive_index], &d_uv_data[primitive_index * 8], hit_point, d_mask_data, d_mask_offsets, d_mask_sizes)) {
                            continue; // transparent texel -> not a hit
                        }
                    }

                    total_hits++;

                    if (find_closest_hit) {
                        // Keep only the closest hit
                        if (hit_distance < closest_hit_distance) {
                            closest_hit_distance = hit_distance;
                            hit_primitive_id = primitive_id;
                            best_vertex_offset = (int) vertex_offset;
                            best_ptype = ptype;
                        }
                    } else {
                        // For collision detection, any hit is sufficient
                        d_hit_distances[ray_idx] = hit_distance;
                        d_hit_primitive_ids[ray_idx] = primitive_id;
                        d_hit_counts[ray_idx] = 1; // Found at least one hit
                        if (d_hit_normals) {
                            d_hit_normals[ray_idx] = computeHitNormal(ptype, d_primitive_vertices, vertex_offset, ray_origin, ray_direction, hit_distance);
                        }
                        return; // Early exit for collision detection
                    }
                }
            }
        } else {
            // Add child nodes to stack (add right child first for left-first traversal). The capacity is sized well above
            // the tree depth; if a push would overflow, raise the device flag (the host throws) rather than silently
            // dropping the child and reporting a wrong/missed hit.
            if (node.right_child != 0xFFFFFFFF) {
                if (stack_size < BVH_TRAVERSAL_STACK_CAPACITY) {
                    thread_stack[stack_size] = node.right_child;
                    stack_size++;
                } else {
                    atomicMax(&d_bvh_stack_overflow, 1u);
                }
            }
            if (node.left_child != 0xFFFFFFFF) {
                if (stack_size < BVH_TRAVERSAL_STACK_CAPACITY) {
                    thread_stack[stack_size] = node.left_child;
                    stack_size++;
                } else {
                    atomicMax(&d_bvh_stack_overflow, 1u);
                }
            }
        }
    }

    // Store final results
    if (find_closest_hit && hit_primitive_id != 0xFFFFFFFF) {
        d_hit_distances[ray_idx] = closest_hit_distance;
        d_hit_primitive_ids[ray_idx] = hit_primitive_id;
        d_hit_counts[ray_idx] = 1;
        if (d_hit_normals) {
            d_hit_normals[ray_idx] = computeHitNormal(best_ptype, d_primitive_vertices, (unsigned int) best_vertex_offset, ray_origin, ray_direction, closest_hit_distance);
        }
    } else if (!find_closest_hit) {
        d_hit_distances[ray_idx] = ray_max_distance + 1.0f; // No hit
        d_hit_primitive_ids[ray_idx] = 0xFFFFFFFF;
        d_hit_counts[ray_idx] = 0;
        if (d_hit_normals) {
            d_hit_normals[ray_idx] = make_float3(0.0f, 0.0f, 0.0f);
        }
    } else {
        // No hit found
        d_hit_distances[ray_idx] = ray_max_distance + 1.0f;
        d_hit_primitive_ids[ray_idx] = 0xFFFFFFFF;
        d_hit_counts[ray_idx] = 0;
        if (d_hit_normals) {
            d_hit_normals[ray_idx] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
}

//! Type code given to a resident primitive that has been deleted from the Context since the scene was uploaded.
//! rayPrimitiveBVHKernel tests only types 0-2, so rays pass through such a primitive, as the CPU traversal does.
#define GPU_PRIMITIVE_TYPE_DELETED (-1)

/**
 * \brief CUDA kernel marking resident primitives as deleted
 * \param[in,out] d_primitive_types Resident per-primitive type codes in BVH order
 * \param[in] d_positions Positions in BVH order of the primitives to mark
 * \param[in] count Number of positions
 */
__global__ void markPrimitivesDeletedKernel(int *d_primitive_types, const unsigned int *d_positions, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        d_primitive_types[d_positions[i]] = GPU_PRIMITIVE_TYPE_DELETED;
    }
}

// Host wrapper functions called from the C++ sources. These must keep C++ linkage (no extern "C"): they raise
// helios_runtime_error, and MSVC's /EHsc assumes extern "C" functions never throw, so it skips the destructors of their
// locals during unwinding and the DeviceBuffer allocations are leaked on every error.

/**
 * \brief Mark primitives of the resident scene as deleted, so that rays pass through them
 * \param[in,out] d_primitive_types Resident per-primitive type codes in BVH order
 * \param[in] h_positions Host array of the positions in BVH order of the primitives to mark
 * \param[in] count Number of positions
 */
void launchMarkPrimitivesDeleted(int *d_primitive_types, const unsigned int *h_positions, int count) {
    if (count == 0) {
        return;
    }
    DeviceBuffer<unsigned int> d_positions(count);
    HELIOS_CUDA_CHECK(cudaMemcpy(d_positions.get(), h_positions, size_t(count) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int num_blocks = (count + threads_per_block - 1) / threads_per_block;
    cudaGetLastError(); // clear an error left by an earlier failed CUDA call so that the check below reports only this launch
    markPrimitivesDeletedKernel<<<num_blocks, threads_per_block>>>(d_primitive_types, d_positions.get(), count);
    HELIOS_CUDA_CHECK(cudaGetLastError());
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * \brief Launch the ray-primitive intersection kernel against scene geometry that is ALREADY resident on the device.
 *
 * Only the per-call ray inputs and hit outputs are allocated, uploaded, and freed here; the BVH, primitive indices,
 * types, packed vertices, and vertex offsets are passed in as device pointers owned by the caller (uploaded once per
 * scan by transferBVHToGPU). This is what keeps chunked synthetic scans from re-uploading the whole scene per chunk.
 * Pass h_hit_normals != nullptr to also read back the per-ray face-forwarded surface normal (flat xyz, 3 floats/ray).
 * CUDA failures throw helios_runtime_error (fail-fast) rather than writing partial results.
 *
 * \param[in] d_bvh_nodes Device BVH nodes (resident)
 * \param[in] node_count Number of BVH nodes
 * \param[in] d_primitive_indices Device primitive indices (resident)
 * \param[in] primitive_count Number of primitives
 * \param[in] d_primitive_types Device per-primitive type codes (resident)
 * \param[in] d_primitive_vertices Device packed primitive vertices (resident)
 * \param[in] d_vertex_offsets Device per-primitive vertex offsets (resident)
 * \param[in] d_mask_data Device texture transparency mask bytes, or null when the scene has no masks (resident)
 * \param[in] d_mask_offsets Device per-mask start index into d_mask_data (resident)
 * \param[in] d_mask_sizes Device per-mask width/height (resident)
 * \param[in] d_mask_IDs Device per-primitive mask index, -1 = none (resident)
 * \param[in] d_uv_data Device per-primitive UVs, 4 vec2 per primitive (resident)
 * \param[in] d_uv_IDs Device per-primitive UV flag, -1 = parametric/none (resident)
 * \param[in] total_vertex_count Number of packed vertices
 * \param[in] h_ray_origins Host ray origins (3 floats per ray)
 * \param[in] h_ray_directions Host ray directions (3 floats per ray)
 * \param[in] h_ray_max_distances Host ray maximum distances
 * \param[in] num_rays Number of rays to process
 * \param[out] h_hit_distances Host array for closest-hit distances
 * \param[out] h_hit_primitive_ids Host array for hit primitive IDs
 * \param[out] h_hit_counts Host array for per-ray hit counts (0 == miss)
 * \param[out] h_hit_normals Host array for per-ray surface normals (3 floats per ray), or nullptr to skip
 * \param[in] find_closest_hit If true, return only the closest hit
 */
void launchRaysOnResidentScene(void *d_bvh_nodes, int node_count, unsigned int *d_primitive_indices, int primitive_count, int *d_primitive_types, float3 *d_primitive_vertices, unsigned int *d_vertex_offsets, const unsigned char *d_mask_data,
                               const unsigned int *d_mask_offsets, const int *d_mask_sizes, const int *d_mask_IDs, const float *d_uv_data, const int *d_uv_IDs, int total_vertex_count, const float *h_ray_origins, const float *h_ray_directions,
                               const float *h_ray_max_distances, float uniform_max_distance, int num_rays, float *h_hit_distances, unsigned int *h_hit_primitive_ids, unsigned int *h_hit_counts, float *h_hit_normals, bool find_closest_hit) {
    if (num_rays == 0) {
        return;
    }

    // Allocate ONLY the per-call ray + result buffers; the geometry pointers are resident and owned by the caller. Each
    // optional host pointer that is null skips its device buffer / copy, so a caller writing straight into its own arrays
    // (the SoA path) drives no extra host staging. d_hit_counts is always allocated because the kernel writes it. The
    // buffers free themselves, including when a CUDA call below throws.
    const size_t ray_data_size = size_t(num_rays) * sizeof(float3);
    const size_t ray_distances_size = size_t(num_rays) * sizeof(float);
    const size_t hit_results_size = size_t(num_rays) * sizeof(unsigned int);

    DeviceBuffer<float3> d_ray_origins(num_rays);
    DeviceBuffer<float3> d_ray_directions(num_rays);
    DeviceBuffer<float> d_hit_distances(num_rays);
    DeviceBuffer<unsigned int> d_hit_primitive_ids(num_rays);
    DeviceBuffer<unsigned int> d_hit_counts(num_rays);
    DeviceBuffer<float> d_ray_max_distances;
    const bool per_ray_max = (h_ray_max_distances != nullptr);
    if (per_ray_max) {
        d_ray_max_distances.allocate(num_rays);
    }
    DeviceBuffer<float3> d_hit_normals;
    const bool want_normals = (h_hit_normals != nullptr);
    if (want_normals) {
        d_hit_normals.allocate(num_rays);
    }

    // helios::vec3 and the caller's flat xyz arrays have the exact byte layout of float3, so the host ray buffers copy
    // straight into the device float3 buffers with no host repack. Directions are normalized in the kernel.
    HELIOS_CUDA_CHECK(cudaMemcpy(d_ray_origins.get(), h_ray_origins, ray_data_size, cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_ray_directions.get(), h_ray_directions, ray_data_size, cudaMemcpyHostToDevice));
    if (per_ray_max) {
        HELIOS_CUDA_CHECK(cudaMemcpy(d_ray_max_distances.get(), h_ray_max_distances, ray_distances_size, cudaMemcpyHostToDevice));
    }

    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((static_cast<long long>(num_rays) + threads_per_block - 1) / threads_per_block);

    // Reset the traversal-stack overflow flag before launch; checked after sync (fail-fast on a deeper-than-expected tree).
    const unsigned int stack_overflow_reset = 0;
    HELIOS_CUDA_CHECK(cudaMemcpyToSymbol(d_bvh_stack_overflow, &stack_overflow_reset, sizeof(unsigned int)));

    // cudaGetLastError() after the launch also returns an error left by an earlier failed CUDA call, which was already raised
    // to the caller. Clear it here so that the launch check reports only this launch.
    cudaGetLastError();
    rayPrimitiveBVHKernel<<<num_blocks, threads_per_block>>>((GPUBVHNode *) d_bvh_nodes, d_primitive_indices, d_primitive_types, d_primitive_vertices, d_vertex_offsets, d_mask_data, d_mask_offsets, d_mask_sizes, d_mask_IDs, d_uv_data, d_uv_IDs,
                                                             d_ray_origins.get(), d_ray_directions.get(), d_ray_max_distances.get(), uniform_max_distance, num_rays, primitive_count, total_vertex_count, d_hit_distances.get(),
                                                             d_hit_primitive_ids.get(), d_hit_counts.get(), d_hit_normals.get(), find_closest_hit);

    HELIOS_CUDA_CHECK(cudaGetLastError()); // launch configuration errors
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize()); // errors raised while the kernel ran

    unsigned int stack_overflow_flag = 0;
    HELIOS_CUDA_CHECK(cudaMemcpyFromSymbol(&stack_overflow_flag, d_bvh_stack_overflow, sizeof(unsigned int)));
    if (stack_overflow_flag != 0) {
        helios::helios_runtime_error("ERROR (CollisionDetection GPU): BVH traversal stack overflow in rayPrimitiveBVHKernel - tree depth exceeded BVH_TRAVERSAL_STACK_CAPACITY (" + std::to_string(BVH_TRAVERSAL_STACK_CAPACITY) +
                                     "). This must not happen for a valid SAH tree; raise BVH_TRAVERSAL_STACK_CAPACITY or check the BVH build.");
    }

    HELIOS_CUDA_CHECK(cudaMemcpy(h_hit_distances, d_hit_distances.get(), ray_distances_size, cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaMemcpy(h_hit_primitive_ids, d_hit_primitive_ids.get(), hit_results_size, cudaMemcpyDeviceToHost));
    if (h_hit_counts != nullptr) {
        HELIOS_CUDA_CHECK(cudaMemcpy(h_hit_counts, d_hit_counts.get(), hit_results_size, cudaMemcpyDeviceToHost));
    }
    if (want_normals) {
        HELIOS_CUDA_CHECK(cudaMemcpy(h_hit_normals, d_hit_normals.get(), ray_data_size, cudaMemcpyDeviceToHost));
    }
}

/**
 * \brief CUDA kernel finding, for each query box, the resident-BVH primitives whose bounding boxes overlap it
 *
 * Each thread processes one query box. The kernel runs twice over the same traversal: with \p d_hit_offsets null it only
 * counts each query's hits into \p d_hit_counts; with \p d_hit_offsets set to the exclusive prefix sums of those counts it
 * writes each query's hits to d_hits[d_hit_offsets[query]...]. The output is therefore sized exactly and never truncated.
 *
 * \param[in] d_nodes Resident BVH nodes
 * \param[in] d_primitive_indices Resident primitive UUIDs in BVH order
 * \param[in] d_primitive_aabb_min Resident primitive bounding-box minimum corners in BVH order
 * \param[in] d_primitive_aabb_max Resident primitive bounding-box maximum corners in BVH order
 * \param[in] d_query_aabb_min Query bounding-box minimum corners
 * \param[in] d_query_aabb_max Query bounding-box maximum corners
 * \param[in] num_queries Number of query boxes
 * \param[in] d_hit_offsets Per-query start index into \p d_hits (fill pass), or null (count pass)
 * \param[out] d_hit_counts Per-query hit counts (written in the count pass only)
 * \param[out] d_hits Hit primitive UUIDs (written in the fill pass only)
 */
__global__ void bvhBoxQueryKernel(const GPUBVHNode *d_nodes, const unsigned int *d_primitive_indices, const float3 *d_primitive_aabb_min, const float3 *d_primitive_aabb_max, const float3 *d_query_aabb_min, const float3 *d_query_aabb_max,
                                  int num_queries, const unsigned long long *d_hit_offsets, unsigned int *d_hit_counts, unsigned int *d_hits) {

    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) {
        return;
    }

    const float3 query_min = d_query_aabb_min[query_idx];
    const float3 query_max = d_query_aabb_max[query_idx];

    const bool fill_pass = (d_hit_offsets != nullptr);
    unsigned int *query_hits = fill_pass ? d_hits + d_hit_offsets[query_idx] : nullptr;
    unsigned int hit_count = 0;

    // Per-thread local stack, sized to BVH_TRAVERSAL_STACK_CAPACITY (well above the SAH tree depth).
    unsigned int thread_stack[BVH_TRAVERSAL_STACK_CAPACITY];
    thread_stack[0] = 0;
    int stack_size = 1;

    while (stack_size > 0) {
        stack_size--;
        const unsigned int node_idx = thread_stack[stack_size];
        if (node_idx == 0xFFFFFFFF) {
            continue;
        }

        const GPUBVHNode node = d_nodes[node_idx];
        if (!d_aabbIntersect(query_min, query_max, node.aabb_min, node.aabb_max)) {
            continue;
        }

        if (node.is_leaf) {
            for (unsigned int i = 0; i < node.primitive_count; i++) {
                const unsigned int primitive_index = node.primitive_start + i;
                if (d_aabbIntersect(query_min, query_max, d_primitive_aabb_min[primitive_index], d_primitive_aabb_max[primitive_index])) {
                    if (fill_pass) {
                        query_hits[hit_count] = d_primitive_indices[primitive_index];
                    }
                    hit_count++;
                }
            }
        } else {
            // An overflow raises the device flag (the host throws) rather than silently dropping a child and under-reporting.
            if (node.left_child != 0xFFFFFFFF) {
                if (stack_size < BVH_TRAVERSAL_STACK_CAPACITY) {
                    thread_stack[stack_size] = node.left_child;
                    stack_size++;
                } else {
                    atomicMax(&d_bvh_stack_overflow, 1u);
                }
            }
            if (node.right_child != 0xFFFFFFFF) {
                if (stack_size < BVH_TRAVERSAL_STACK_CAPACITY) {
                    thread_stack[stack_size] = node.right_child;
                    stack_size++;
                } else {
                    atomicMax(&d_bvh_stack_overflow, 1u);
                }
            }
        }
    }

    if (!fill_pass) {
        d_hit_counts[query_idx] = hit_count;
    }
}

//! Upload the query boxes and run one pass of bvhBoxQueryKernel, throwing on any CUDA error or traversal-stack overflow.
static void runBVHBoxQueryPass(void *d_nodes, unsigned int *d_primitive_indices, void *d_primitive_aabb_min, void *d_primitive_aabb_max, const float *h_query_aabb_min, const float *h_query_aabb_max, int num_queries,
                               const unsigned long long *d_hit_offsets, unsigned int *d_hit_counts, unsigned int *d_hits) {

    // float3 is three packed floats, so the host xyz arrays copy straight into float3 buffers.
    DeviceBuffer<float3> d_query_min(num_queries);
    DeviceBuffer<float3> d_query_max(num_queries);
    HELIOS_CUDA_CHECK(cudaMemcpy(d_query_min.get(), h_query_aabb_min, num_queries * sizeof(float3), cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_query_max.get(), h_query_aabb_max, num_queries * sizeof(float3), cudaMemcpyHostToDevice));

    const unsigned int stack_overflow_reset = 0;
    HELIOS_CUDA_CHECK(cudaMemcpyToSymbol(d_bvh_stack_overflow, &stack_overflow_reset, sizeof(unsigned int)));

    const int block_size = 256;
    const int num_blocks = static_cast<int>((static_cast<long long>(num_queries) + block_size - 1) / block_size);
    // cudaGetLastError() after the launch also returns an error left by an earlier failed CUDA call, which was already raised
    // to the caller. Clear it here so that the launch check reports only this launch.
    cudaGetLastError();
    bvhBoxQueryKernel<<<num_blocks, block_size>>>((const GPUBVHNode *) d_nodes, d_primitive_indices, (const float3 *) d_primitive_aabb_min, (const float3 *) d_primitive_aabb_max, d_query_min.get(), d_query_max.get(), num_queries, d_hit_offsets,
                                                  d_hit_counts, d_hits);
    HELIOS_CUDA_CHECK(cudaGetLastError());
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int stack_overflow_flag = 0;
    HELIOS_CUDA_CHECK(cudaMemcpyFromSymbol(&stack_overflow_flag, d_bvh_stack_overflow, sizeof(unsigned int)));
    if (stack_overflow_flag != 0) {
        helios::helios_runtime_error("ERROR (CollisionDetection GPU): BVH traversal stack overflow in bvhBoxQueryKernel - tree depth exceeded BVH_TRAVERSAL_STACK_CAPACITY (" + std::to_string(BVH_TRAVERSAL_STACK_CAPACITY) +
                                     "). This must not happen for a valid SAH tree; raise BVH_TRAVERSAL_STACK_CAPACITY or check the BVH build.");
    }
}

/**
 * \brief Count, for each query box, the primitives of the resident BVH whose bounding boxes overlap it (first of two passes)
 *
 * \param[in] d_nodes Resident BVH nodes
 * \param[in] d_primitive_indices Resident primitive UUIDs in BVH order
 * \param[in] d_primitive_aabb_min Resident primitive bounding-box minimum corners (float3) in BVH order
 * \param[in] d_primitive_aabb_max Resident primitive bounding-box maximum corners (float3) in BVH order
 * \param[in] h_query_aabb_min Host query minimum corners, xyz per query
 * \param[in] h_query_aabb_max Host query maximum corners, xyz per query
 * \param[in] num_queries Number of query boxes
 * \param[out] h_hit_counts Host array of \p num_queries hit counts
 */
void launchBVHBoxQueryCount(void *d_nodes, unsigned int *d_primitive_indices, void *d_primitive_aabb_min, void *d_primitive_aabb_max, const float *h_query_aabb_min, const float *h_query_aabb_max, int num_queries, unsigned int *h_hit_counts) {
    if (num_queries == 0) {
        return;
    }
    DeviceBuffer<unsigned int> d_hit_counts(num_queries);
    runBVHBoxQueryPass(d_nodes, d_primitive_indices, d_primitive_aabb_min, d_primitive_aabb_max, h_query_aabb_min, h_query_aabb_max, num_queries, nullptr, d_hit_counts.get(), nullptr);
    HELIOS_CUDA_CHECK(cudaMemcpy(h_hit_counts, d_hit_counts.get(), num_queries * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

/**
 * \brief Write, for each query box, the UUIDs of the resident-BVH primitives whose bounding boxes overlap it (second of two passes)
 *
 * \param[in] d_nodes Resident BVH nodes
 * \param[in] d_primitive_indices Resident primitive UUIDs in BVH order
 * \param[in] d_primitive_aabb_min Resident primitive bounding-box minimum corners (float3) in BVH order
 * \param[in] d_primitive_aabb_max Resident primitive bounding-box maximum corners (float3) in BVH order
 * \param[in] h_query_aabb_min Host query minimum corners, xyz per query
 * \param[in] h_query_aabb_max Host query maximum corners, xyz per query
 * \param[in] num_queries Number of query boxes
 * \param[in] h_hit_offsets Host array of \p num_queries start indices into \p h_hits (exclusive prefix sums of launchBVHBoxQueryCount's counts)
 * \param[in] total_hits Total number of hits over all queries (size of \p h_hits)
 * \param[out] h_hits Host array receiving the hit primitive UUIDs, grouped by query
 */
void launchBVHBoxQueryFill(void *d_nodes, unsigned int *d_primitive_indices, void *d_primitive_aabb_min, void *d_primitive_aabb_max, const float *h_query_aabb_min, const float *h_query_aabb_max, int num_queries,
                           const unsigned long long *h_hit_offsets, unsigned long long total_hits, unsigned int *h_hits) {
    if (num_queries == 0 || total_hits == 0) {
        return;
    }
    DeviceBuffer<unsigned long long> d_hit_offsets(num_queries);
    DeviceBuffer<unsigned int> d_hits(total_hits);
    HELIOS_CUDA_CHECK(cudaMemcpy(d_hit_offsets.get(), h_hit_offsets, num_queries * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    runBVHBoxQueryPass(d_nodes, d_primitive_indices, d_primitive_aabb_min, d_primitive_aabb_max, h_query_aabb_min, h_query_aabb_max, num_queries, d_hit_offsets.get(), nullptr, d_hits.get());
    HELIOS_CUDA_CHECK(cudaMemcpy(h_hits, d_hits.get(), total_hits * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

/**
 * \brief Check that the current CUDA device can run this plug-in's kernels
 *
 * A visible device can still be unusable: the build may have no kernel image for its architecture, or it may be busy in
 * exclusive-process mode. Both show up here, before GPU acceleration is chosen, rather than as a failure of every GPU call.
 *
 * \return True if the kernels have an image for the device and a small device allocation succeeds
 */
bool probeCollisionDetectionKernels() {
    cudaFuncAttributes attributes;
    bool usable = cudaFuncGetAttributes(&attributes, bvhBoxQueryKernel) == cudaSuccess && cudaFuncGetAttributes(&attributes, rayPrimitiveBVHKernel) == cudaSuccess;
    if (usable) {
        void *probe_allocation = nullptr;
        usable = cudaMalloc(&probe_allocation, 256) == cudaSuccess;
        if (probe_allocation != nullptr) {
            cudaFree(probe_allocation);
        }
    }
    cudaGetLastError(); // a failed probe is answered by the return value; do not leave it as the thread's last CUDA error
    return usable;
}
