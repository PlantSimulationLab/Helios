/** \file "CollisionDetection_RayTracing.cpp" Ray tracing implementation for collision detection plugin

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <chrono>
#include <functional>
#include <limits>
#include <queue>
#include <stack>
#include <thread>
#include "CollisionDetection.h"

// SIMD headers
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

// MSVC prefetch support
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <algorithm>

#ifdef HELIOS_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

using namespace helios;

#ifdef HELIOS_CUDA_AVAILABLE
// GPU BVH node structure (must match the one in .cu file)
struct GPUBVHNode {
    float3 aabb_min, aabb_max;
    unsigned int left_child, right_child;
    unsigned int primitive_start, primitive_count;
    unsigned int is_leaf, padding;
};

// External CUDA functions
extern "C" {
void launchBVHTraversal(void *h_nodes, int node_count, unsigned int *h_primitive_indices, int primitive_count, float *h_primitive_aabb_min, float *h_primitive_aabb_max, float *h_query_aabb_min, float *h_query_aabb_max, int num_queries,
                        unsigned int *h_results, unsigned int *h_result_counts, int max_results_per_query);
bool launchVoxelRayPathLengths(int num_rays, float *h_ray_origins, float *h_ray_directions, float grid_center_x, float grid_center_y, float grid_center_z, float grid_size_x, float grid_size_y, float grid_size_z, int grid_divisions_x,
                               int grid_divisions_y, int grid_divisions_z, int primitive_count, int *h_voxel_ray_counts, float *h_voxel_path_lengths, int *h_voxel_transmitted, int *h_voxel_hit_before, int *h_voxel_hit_after, int *h_voxel_hit_inside);
// Warp-efficient GPU kernels
void launchWarpEfficientBVH(void *h_bvh_soa_gpu, unsigned int *h_primitive_indices, int primitive_count, float *h_primitive_aabb_min, float *h_primitive_aabb_max, float *h_ray_origins, float *h_ray_directions, float *h_ray_max_distances,
                            int num_rays, unsigned int *h_results, unsigned int *h_result_counts, int max_results_per_ray);
// High-performance ray-primitive intersection kernel against device-resident scene geometry. Only the per-call ray and
// result buffers are uploaded/downloaded here; BVH/primitive geometry is passed in as resident device pointers.
void launchRaysOnResidentScene(void *d_bvh_nodes, int node_count, unsigned int *d_primitive_indices, int primitive_count, int *d_primitive_types, float3 *d_primitive_vertices, unsigned int *d_vertex_offsets, const unsigned char *d_mask_data,
                               const unsigned int *d_mask_offsets, const int *d_mask_sizes, const int *d_mask_IDs, const float *d_uv_data, const int *d_uv_IDs, int total_vertex_count, const float *h_ray_origins, const float *h_ray_directions,
                               const float *h_ray_max_distances, float uniform_max_distance, int num_rays, float *h_hit_distances, unsigned int *h_hit_primitive_ids, unsigned int *h_hit_counts, float *h_hit_normals, bool find_closest_hit);
}

// Helper function to convert helios::vec3 to float3
inline float3 heliosVecToFloat3(const helios::vec3 &v) {
    return make_float3(v.x, v.y, v.z);
}
#endif

// -------- GENERIC RAY-TRACING IMPLEMENTATIONS --------

CollisionDetection::HitResult CollisionDetection::castRay(const RayQuery &ray_query) {
    return castRay(ray_query.origin, ray_query.direction, ray_query.max_distance, ray_query.target_UUIDs);
}

CollisionDetection::HitResult CollisionDetection::castRay(const vec3 &origin, const vec3 &direction, float max_distance, const std::vector<uint> &target_UUIDs) {
    HitResult result;

    // Normalize direction vector
    vec3 ray_direction = direction;
    if (ray_direction.magnitude() < 1e-8f) {
        return result; // Invalid direction
    }
    ray_direction = ray_direction / ray_direction.magnitude();

    // Ensure BVH is current before ray casting (handles automatic rebuilds)
    const_cast<CollisionDetection *>(this)->ensureBVHCurrent();

    // CRITICAL FIX: Use BVH traversal when available for performance
    // This fixes the 10x+ performance regression by replacing brute-force primitive testing
    if (!bvh_nodes.empty()) {
        // Build primitive cache for thread-safe access if not already built (keeps the dense cache in sync too)
        ensurePrimitiveCacheCurrent();

        // Use BVH traversal for both filtered and unfiltered queries - BVH supports UUID filtering
        RayQuery query(origin, ray_direction, max_distance, target_UUIDs);
        return castRayBVHTraversal(query);
    }

    // FALLBACK: Brute-force primitive testing (only when no BVH available or target UUIDs specified)
    // This is the slow path that was causing the performance regression
    std::vector<uint> search_primitives;
    if (target_UUIDs.empty()) {
        // Use cached primitives if available (thread-safe), otherwise use context (thread-unsafe)
        if (!primitive_cache.empty()) {
            // Use cached primitive IDs for thread-safe operation
            search_primitives.reserve(primitive_cache.size());
            for (const auto &cached_pair: primitive_cache) {
                search_primitives.push_back(cached_pair.first);
            }
        } else {
            // Fall back to context call (thread-unsafe)
            search_primitives = context->getAllUUIDs();
        }
    } else {
        search_primitives = target_UUIDs;
    }

    // Find nearest intersection
    float nearest_distance = std::numeric_limits<float>::max();
    if (max_distance > 0) {
        nearest_distance = max_distance;
    }

    uint hit_primitive = 0;
    bool found_intersection = false;

    // Test each primitive for intersection
    for (uint candidate_uuid: search_primitives) {
        float intersection_distance;
        bool hit = false;

        // Use thread-safe cached intersection if available
        if (!primitive_cache.empty()) {
            HitResult primitive_result = intersectPrimitiveThreadSafe(origin, ray_direction, candidate_uuid, max_distance);
            if (primitive_result.hit) {
                hit = true;
                intersection_distance = primitive_result.distance;
            }
        } else {
            // Fall back to thread-unsafe context call
            if (!context->doesPrimitiveExist(candidate_uuid)) {
                continue;
            }
            hit = rayPrimitiveIntersection(origin, ray_direction, candidate_uuid, intersection_distance);

            // Reject hits on transparent texels so the ray passes through to geometry behind.
            if (hit && context->primitiveTextureHasTransparencyChannel(candidate_uuid)) {
                CachedPrimitive temp_cached(context->getPrimitiveType(candidate_uuid), context->getPrimitiveVertices(candidate_uuid));
                temp_cached.transparency_mask = context->getPrimitiveTextureTransparencyData(candidate_uuid);
                temp_cached.texture_size = context->getPrimitiveTextureSize(candidate_uuid);
                temp_cached.uv = context->getPrimitiveTextureUV(candidate_uuid);
                if (!isHitTexelOpaque(temp_cached, origin + ray_direction * intersection_distance)) {
                    hit = false;
                }
            }
        }

        if (hit) {
            // Check distance constraints
            if (intersection_distance > 1e-6f && // Avoid self-intersection
                intersection_distance < nearest_distance) { // Find nearest

                nearest_distance = intersection_distance;
                hit_primitive = candidate_uuid;
                found_intersection = true;
            }
        }
    }

    // Fill result
    if (found_intersection) {
        result.hit = true;
        result.distance = nearest_distance;
        result.primitive_UUID = hit_primitive;
        result.intersection_point = origin + ray_direction * nearest_distance;

        // Calculate surface normal
        try {
            PrimitiveType type = context->getPrimitiveType(hit_primitive);
            std::vector<vec3> vertices = context->getPrimitiveVertices(hit_primitive);

            if (type == PRIMITIVE_TYPE_TRIANGLE && vertices.size() >= 3) {
                // Calculate triangle normal
                vec3 v0 = vertices[0];
                vec3 v1 = vertices[1];
                vec3 v2 = vertices[2];
                vec3 edge1 = v1 - v0;
                vec3 edge2 = v2 - v0;
                result.normal = cross(edge1, edge2);
                result.normal = result.normal / result.normal.magnitude();
            } else if (type == PRIMITIVE_TYPE_PATCH && vertices.size() >= 4) {
                // Calculate patch normal (assuming quad)
                vec3 v0 = vertices[0];
                vec3 v1 = vertices[1];
                vec3 v2 = vertices[2];
                vec3 edge1 = v1 - v0;
                vec3 edge2 = v2 - v0;
                result.normal = cross(edge1, edge2);
                result.normal = result.normal / result.normal.magnitude();
            } else {
                // Default normal (pointing back along ray)
                result.normal = make_vec3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
            }
        } catch (const std::exception &e) {
            // If we can't get surface normal, use default
            result.normal = make_vec3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
        }
    }

    return result;
}


bool CollisionDetection::shouldUseGPU(size_t ray_count) const {
#ifdef HELIOS_CUDA_AVAILABLE
    // Use GPU only for large batches (amortize launch + transfer overhead) on a complex, already-resident scene.
    // CPU is faster for small batches. Mirrors the thresholds the vector dispatch has always used.
    constexpr size_t GPU_BATCH_THRESHOLD = 1000000; // batches >= 1M rays
    constexpr size_t MIN_PRIMITIVES_FOR_GPU = 500; // minimum scene complexity
    return gpu_acceleration_enabled && ray_count >= GPU_BATCH_THRESHOLD && d_bvh_nodes != nullptr && d_primitive_vertices != nullptr && !primitive_indices.empty() && primitive_indices.size() >= MIN_PRIMITIVES_FOR_GPU;
#else
    (void) ray_count;
    return false;
#endif
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRays(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats) {
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());

    // Initialize statistics
    RayTracingStats local_stats;
    local_stats.total_rays_cast = ray_queries.size();

    // Smart CPU/GPU selection via the shared predicate (see shouldUseGPU): GPU only pays off for large batches on a
    // sufficiently complex, GPU-resident scene. Keeping the predicate in one place ensures castRays and castRaysSoA
    // never diverge in how they choose GPU vs CPU.
#ifdef HELIOS_CUDA_AVAILABLE
    if (shouldUseGPU(ray_queries.size())) {
        castRaysGPU(ray_queries, results, local_stats);
    } else {
        castRaysCPU(ray_queries, results, local_stats);
    }
#else
    // Use CPU implementation when GPU not available
    castRaysCPU(ray_queries, results, local_stats);
#endif

    // Copy statistics to output parameter if provided
    if (stats != nullptr) {
        *stats = local_stats;
    }

    return results;
}

void CollisionDetection::castRaysCPU(const std::vector<RayQuery> &ray_queries, std::vector<HitResult> &results, RayTracingStats &stats) {
    // Use optimized batch processing - fail explicitly if there are issues
    results = castRaysOptimized(ray_queries, &stats);
}

#ifdef HELIOS_CUDA_AVAILABLE
void CollisionDetection::castRaysGPU(const std::vector<RayQuery> &ray_queries, std::vector<HitResult> &results, RayTracingStats &stats) {
    // Use the high-performance GPU ray-triangle intersection implementation
    results = castRaysGPU(ray_queries, stats);
}
#endif

std::vector<std::vector<std::vector<std::vector<CollisionDetection::HitResult>>>> CollisionDetection::performGridRayIntersection(const vec3 &grid_center, const vec3 &grid_size, const helios::int3 &grid_divisions,
                                                                                                                                 const std::vector<RayQuery> &ray_queries) {

    // Initialize result grid
    std::vector<std::vector<std::vector<std::vector<HitResult>>>> grid_results;
    grid_results.resize(grid_divisions.x);
    for (int i = 0; i < grid_divisions.x; i++) {
        grid_results[i].resize(grid_divisions.y);
        for (int j = 0; j < grid_divisions.y; j++) {
            grid_results[i][j].resize(grid_divisions.z);
        }
    }

    // Calculate voxel size
    vec3 voxel_size = make_vec3(grid_size.x / float(grid_divisions.x), grid_size.y / float(grid_divisions.y), grid_size.z / float(grid_divisions.z));

    // Process each ray
    for (const auto &query: ray_queries) {
        HitResult hit_result = castRay(query);

        if (hit_result.hit) {
            // Determine which voxel the hit point falls into
            vec3 relative_pos = hit_result.intersection_point - (grid_center - grid_size * 0.5f);

            int voxel_i = int(relative_pos.x / voxel_size.x);
            int voxel_j = int(relative_pos.y / voxel_size.y);
            int voxel_k = int(relative_pos.z / voxel_size.z);

            // Check bounds
            if (voxel_i >= 0 && voxel_i < grid_divisions.x && voxel_j >= 0 && voxel_j < grid_divisions.y && voxel_k >= 0 && voxel_k < grid_divisions.z) {

                grid_results[voxel_i][voxel_j][voxel_k].push_back(hit_result);
            }
        }
    }

    return grid_results;
}

std::vector<std::vector<CollisionDetection::HitResult>> CollisionDetection::calculateVoxelPathLengths(const vec3 &scan_origin, const std::vector<vec3> &ray_directions, const std::vector<vec3> &voxel_centers, const std::vector<vec3> &voxel_sizes) {

    if (ray_directions.empty()) {
        if (printmessages) {
            std::cout << "WARNING (CollisionDetection::calculateVoxelPathLengths): No rays provided" << std::endl;
        }
        return std::vector<std::vector<HitResult>>();
    }

    if (voxel_centers.size() != voxel_sizes.size()) {
        helios_runtime_error("ERROR (CollisionDetection::calculateVoxelPathLengths): voxel_centers and voxel_sizes vectors must have same size");
    }

    if (voxel_centers.empty()) {
        if (printmessages) {
            std::cout << "WARNING (CollisionDetection::calculateVoxelPathLengths): No voxels provided" << std::endl;
        }
        return std::vector<std::vector<HitResult>>();
    }

    const size_t num_rays = ray_directions.size();
    const size_t num_voxels = voxel_centers.size();

    if (printmessages) {
        std::cout << "Calculating voxel path lengths for " << num_rays << " rays through " << num_voxels << " voxels..." << std::endl;
    }

    // Initialize result structure - one vector of HitResults per voxel
    std::vector<std::vector<HitResult>> result(num_voxels);

// OpenMP parallel loop over rays for performance
#pragma omp parallel for schedule(dynamic, 1000)
    for (int ray_idx = 0; ray_idx < static_cast<int>(num_rays); ++ray_idx) {
        const vec3 &ray_direction = ray_directions[ray_idx];

        // Process each voxel for this ray
        for (size_t voxel_idx = 0; voxel_idx < num_voxels; ++voxel_idx) {
            const vec3 &voxel_center = voxel_centers[voxel_idx];
            const vec3 &voxel_size = voxel_sizes[voxel_idx];

            // Calculate voxel AABB from center and size
            const vec3 half_size = voxel_size * 0.5f;
            const vec3 voxel_min = voxel_center - half_size;
            const vec3 voxel_max = voxel_center + half_size;

            // Perform ray-AABB intersection test
            float t_min, t_max;
            if (rayAABBIntersect(scan_origin, ray_direction, voxel_min, voxel_max, t_min, t_max)) {
                // Calculate path length: t_max - max(0, t_min)
                const float path_length = t_max - std::max(0.0f, t_min);

                if (path_length > 1e-6f) { // Only count meaningful intersections
                    // Create HitResult with path length information
                    HitResult hit_result;
                    hit_result.hit = false; // No actual primitive hit, just voxel traversal
                    hit_result.distance = -1.0f; // Not applicable for voxel traversal
                    hit_result.primitive_UUID = 0; // No primitive
                    hit_result.intersection_point = make_vec3(0, 0, 0); // Not applicable
                    hit_result.normal = make_vec3(0, 0, 0); // Not applicable
                    hit_result.path_length = path_length; // This is what we want!

// Thread-safe update of results
#pragma omp critical
                    {
                        result[voxel_idx].push_back(hit_result);
                    }
                }
            }
        }
    }

    if (printmessages) {
        size_t total_intersections = 0;
        for (size_t i = 0; i < num_voxels; ++i) {
            total_intersections += result[i].size();
        }
        std::cout << "Completed voxel path length calculations. Total ray-voxel intersections: " << total_intersections << std::endl;
    }

    return result;
}

void CollisionDetection::calculateRayPathLengthsDetailed(const vec3 &grid_center, const vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<vec3> &ray_origins, const std::vector<vec3> &ray_directions,
                                                         std::vector<HitResult> &hit_results) {

    hit_results.clear();
    hit_results.reserve(ray_origins.size());

    if (ray_origins.size() != ray_directions.size()) {
        helios_runtime_error("ERROR (CollisionDetection::calculateRayPathLengthsDetailed): ray_origins and ray_directions must have the same size");
        return;
    }

    // Also update the existing voxel data structures
    calculateVoxelRayPathLengths(grid_center, grid_size, grid_divisions, ray_origins, ray_directions);

    // Cast each ray and collect detailed results
    for (size_t i = 0; i < ray_origins.size(); i++) {
        RayQuery query(ray_origins[i], ray_directions[i]);
        HitResult result = castRay(query);
        hit_results.push_back(result);
    }
}

// ================================================================
// PHASE 2 OPTIMIZATION METHODS: Structure-of-Arrays & Quantization
// ================================================================

void CollisionDetection::setBVHOptimizationMode(BVHOptimizationMode mode) {
    if (mode == bvh_optimization_mode) {
        return; // No change needed
    }

    BVHOptimizationMode old_mode = bvh_optimization_mode;
    bvh_optimization_mode = mode;

    // Build optimized structures when mode changes
    if (old_mode != mode && !bvh_nodes.empty()) {
        if (printmessages) {
            std::cout << "CollisionDetection: Converting BVH from mode " << static_cast<int>(old_mode) << " to mode " << static_cast<int>(mode) << std::endl;
        }

        // Build the optimized structures immediately
        ensureOptimizedBVH();

        if (printmessages) {
            auto memory_stats = getBVHMemoryUsage();
            std::cout << "CollisionDetection: Memory usage - SoA: " << memory_stats.soa_memory_bytes << " bytes, Quantized: " << memory_stats.quantized_memory_bytes << " bytes (" << memory_stats.quantized_reduction_percent << "% reduction)"
                      << std::endl;
        }
    }
}

CollisionDetection::BVHOptimizationMode CollisionDetection::getBVHOptimizationMode() const {
    return bvh_optimization_mode;
}

void CollisionDetection::convertBVHLayout(BVHOptimizationMode from_mode, BVHOptimizationMode to_mode) {
    // With only SOA_UNCOMPRESSED mode remaining, no conversion needed
    return;
}


std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysOptimized(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats) {
    if (ray_queries.empty()) {
        return {};
    }

    // Ensure BVH is current and optimized structures are available
    ensureBVHCurrent();
    ensureOptimizedBVH();

    // Build primitive cache for high-performance thread-safe primitive intersection (dense cache kept in sync)
    ensurePrimitiveCacheCurrent();

    RayTracingStats local_stats;
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());

    auto start_time = std::chrono::high_resolution_clock::now();

    // Dispatch to appropriate optimized method based on current mode
    switch (bvh_optimization_mode) {
        case BVHOptimizationMode::SOA_UNCOMPRESSED:
            results = castRaysSoA(ray_queries, local_stats);
            break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (stats) {
        *stats = local_stats;
    }

    return results;
}

bool CollisionDetection::processRayStream(RayStream &ray_stream, RayTracingStats *stats) {
    if (ray_stream.packets.empty()) {
        return true;
    }

    RayTracingStats combined_stats;
    bool success = true;

    for (auto &packet: ray_stream.packets) {
        // Convert packet to ray queries
        auto queries = packet.toRayQueries();

        // Process the packet using optimized ray casting
        RayTracingStats packet_stats;
        auto results = castRaysOptimized(queries, &packet_stats);

        if (results.size() != queries.size()) {
            success = false;
            continue;
        }

        // Store results back in packet
        packet.results = std::move(results);

        // Accumulate statistics
        combined_stats.total_rays_cast += packet_stats.total_rays_cast;
        combined_stats.total_hits += packet_stats.total_hits;
        combined_stats.bvh_nodes_visited += packet_stats.bvh_nodes_visited;
        combined_stats.average_ray_distance = (combined_stats.average_ray_distance * (combined_stats.total_rays_cast - packet_stats.total_rays_cast) + packet_stats.average_ray_distance * packet_stats.total_rays_cast) / combined_stats.total_rays_cast;
    }

    if (stats) {
        *stats = combined_stats;
    }

    if (printmessages && success) {
        std::cout << "CollisionDetection: Processed " << ray_stream.packets.size() << " ray packets (" << ray_stream.total_rays << " total rays)" << std::endl;
    }

    return success;
}

CollisionDetection::MemoryUsageStats CollisionDetection::getBVHMemoryUsage() const {
    // Ensure optimized structures are built before calculating memory usage
    const_cast<CollisionDetection *>(this)->ensureOptimizedBVH();

    MemoryUsageStats stats;

    // Calculate SoA memory usage
    stats.soa_memory_bytes = bvh_nodes_soa.getMemoryUsage();

    // With quantized mode removed, set quantized values to 0
    stats.quantized_memory_bytes = 0;
    stats.quantized_reduction_percent = 0.0f;

    return stats;
}

void CollisionDetection::castRaysSoA(const helios::vec3 *origins, const helios::vec3 *directions, size_t count, float max_distance, float *out_distance, helios::vec3 *out_normal, uint *out_primitive_UUID, RayTracingStats *stats) {

    // Low-memory SoA batch cast: no intermediate RayQuery/HitResult vectors. Results are written straight into the
    // caller-owned output arrays; a miss is signalled by out_primitive_UUID == MISS_UUID (out_distance/out_normal are
    // then unspecified). Mirrors the parallel traversal of the vector-based castRaysSoA() but reuses the per-ray kernel
    // castRaySoATraversal() directly.
    constexpr uint MISS_UUID = 0xFFFFFFFFu;

    RayTracingStats local_stats;
    local_stats.total_rays_cast = count;

    if (count == 0) {
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return;
    }

    // Snapshot the external cancellation flag (see the vector-based castRaysSoA overload). When it is already set, or
    // flips mid-trace, the batch short-circuits: the GPU launch is skipped entirely and the CPU loop drains its
    // remaining indices cheaply, marking every un-traced ray as a miss so the caller frees the batch and stops.
    volatile int *const cancel = cancel_flag;
    if (cancel != nullptr && *cancel != 0) {
        for (size_t i = 0; i < count; i++) {
            out_primitive_UUID[i] = MISS_UUID;
        }
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return;
    }

    // Ensure the BVH and primitive cache are current (same prerequisites as castRaysOptimized()). Automatic BVH rebuilds
    // are intentionally NOT toggled here — a caller issuing many batched calls over static geometry should disable
    // automatic rebuilds and buildBVH() once around the whole batch.
    ensureBVHCurrent();

#ifdef HELIOS_CUDA_AVAILABLE
    // GPU fast path: for large batches on a GPU-resident scene, trace on the device and write results straight into the
    // caller arrays. The scene geometry is uploaded once per scan by buildBVH()/transferBVHToGPU() (driven by the
    // LiDAR prepare/finish bracket), so only this batch's ray + result buffers move across the bus here — keeping a
    // chunked synthetic scan from re-uploading the whole scene per chunk. shouldUseGPU() also guards that the scene is
    // actually resident (d_primitive_vertices != nullptr); otherwise we fall through to the CPU traversal below.
    if (shouldUseGPU(count)) {
        // The CPU convention "max_distance <= 0 => unbounded" maps to a large finite kernel cutoff, broadcast to every
        // ray (no per-ray distance array needed).
        const float kernel_max_distance = (max_distance > 0) ? max_distance : std::numeric_limits<float>::max();

        // helios::vec3 is three contiguous floats == float3, so the caller's origins/directions are passed straight to
        // the device with no host repack (directions are normalized in the kernel). Results are written directly into the
        // caller's out_distance / out_primitive_UUID / out_normal — no intermediate host staging. The kernel reports a
        // miss as out_primitive_UUID == 0xFFFFFFFF, which is exactly the SoA MISS_UUID sentinel, and hit_counts is not
        // needed (passed null) since the UUID already distinguishes hit from miss.
        launchRaysOnResidentScene(d_bvh_nodes, d_gpu_node_count, d_primitive_indices, d_gpu_primitive_count, d_primitive_types, (float3 *) d_primitive_vertices, d_vertex_offsets, (const unsigned char *) d_mask_data, d_mask_offsets, d_mask_sizes,
                                  d_mask_IDs, (const float *) d_uv_data, d_uv_IDs, d_gpu_total_vertex_count, reinterpret_cast<const float *>(origins), reinterpret_cast<const float *>(directions), /*h_ray_max_distances=*/nullptr, kernel_max_distance,
                                  static_cast<int>(count), out_distance, out_primitive_UUID, /*h_hit_counts=*/nullptr, reinterpret_cast<float *>(out_normal), true);

        // Stats only: out_distance / out_primitive_UUID / out_normal are already populated by the launch above. The kernel
        // bounds every recorded hit by kernel_max_distance, so a non-sentinel UUID is exactly an in-range hit.
        size_t total_hits = 0;
        double dist_sum = 0.0;
        for (size_t i = 0; i < count; i++) {
            if (out_primitive_UUID[i] != MISS_UUID) {
                total_hits++;
                dist_sum += out_distance[i];
            }
        }
        local_stats.total_hits = total_hits;
        local_stats.average_ray_distance = (total_hits > 0) ? (dist_sum / static_cast<double>(total_hits)) : 0.0;
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return;
    }
#endif

    ensureOptimizedBVH();
    ensurePrimitiveCacheCurrent();

    if (bvh_nodes_soa.node_count == 0) {
        // No SoA BVH: every ray misses.
        for (size_t i = 0; i < count; i++) {
            out_primitive_UUID[i] = MISS_UUID;
        }
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return;
    }

#pragma omp parallel
    {
        RayTracingStats thread_stats = {}; // thread-local statistics

#pragma omp for schedule(guided, 32)
        for (long long i = 0; i < static_cast<long long>(count); ++i) {
            if (cancel != nullptr && *cancel != 0) {
                out_primitive_UUID[i] = MISS_UUID; // run cancelled — drain remaining indices cheaply as misses
                continue;
            }
            RayQuery query(origins[i], directions[i], max_distance); // stack-local; empty target_UUIDs => all primitives
            HitResult result = castRaySoATraversal(query, thread_stats);

            if (result.hit) {
                out_primitive_UUID[i] = result.primitive_UUID;
                out_distance[i] = result.distance;
                out_normal[i] = result.normal;
                thread_stats.total_hits++;
                thread_stats.average_ray_distance += result.distance;
            } else {
                out_primitive_UUID[i] = MISS_UUID;
            }
        }

#pragma omp atomic
        local_stats.total_hits += thread_stats.total_hits;
#pragma omp atomic
        local_stats.average_ray_distance += thread_stats.average_ray_distance;
#pragma omp atomic
        local_stats.bvh_nodes_visited += thread_stats.bvh_nodes_visited;
    }

    if (local_stats.total_hits > 0) {
        local_stats.average_ray_distance /= local_stats.total_hits;
    }

    if (stats != nullptr) {
        *stats = local_stats;
    }
}

void CollisionDetection::castRaysSoA_packets(const helios::vec3 *origins, const helios::vec3 *directions, size_t count, size_t packet_size, float max_distance, float *out_distance, helios::vec3 *out_normal, uint *out_primitive_UUID,
                                             RayTracingStats *stats) {
    constexpr uint MISS_UUID = 0xFFFFFFFFu;

    RayTracingStats local_stats;
    local_stats.total_rays_cast = count;

    if (count == 0) {
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return;
    }

    // packet_size 0 or 1 has no coherence to exploit: defer to the per-ray path (which also covers the GPU fast path).
    if (packet_size <= 1) {
        castRaysSoA(origins, directions, count, max_distance, out_distance, out_normal, out_primitive_UUID, stats);
        return;
    }

    volatile int *const cancel = cancel_flag;
    if (cancel != nullptr && *cancel != 0) {
        for (size_t i = 0; i < count; i++) {
            out_primitive_UUID[i] = MISS_UUID;
        }
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return;
    }

    ensureBVHCurrent();

    // The packet traversal is a CPU-only optimization. On a GPU-resident scene large enough to favor the device, defer
    // to the per-ray path which dispatches to the GPU kernel (per-ray on the GPU is already massively parallel and the
    // packet sharing would not map onto it). shouldUseGPU is false when no GPU build / scene is resident.
    if (shouldUseGPU(count)) {
        castRaysSoA(origins, directions, count, max_distance, out_distance, out_normal, out_primitive_UUID, stats);
        return;
    }

    ensureOptimizedBVH();
    ensurePrimitiveCacheCurrent();

    if (bvh_nodes_soa.node_count == 0) {
        for (size_t i = 0; i < count; i++) {
            out_primitive_UUID[i] = MISS_UUID;
        }
        if (stats != nullptr) {
            *stats = local_stats;
        }
        return;
    }

    const size_t num_packets = (count + packet_size - 1) / packet_size;

#pragma omp parallel
    {
        RayTracingStats thread_stats = {};

#pragma omp for schedule(guided, 8)
        for (long long pkt = 0; pkt < static_cast<long long>(num_packets); ++pkt) {
            const size_t begin = size_t(pkt) * packet_size;
            const size_t end = std::min(begin + packet_size, count);

            if (cancel != nullptr && *cancel != 0) {
                for (size_t i = begin; i < end; i++) {
                    out_primitive_UUID[i] = MISS_UUID;
                }
                continue;
            }

            castPacketSoATraversal(origins, directions, begin, end, max_distance, out_distance, out_normal, out_primitive_UUID, thread_stats);
        }

#pragma omp atomic
        local_stats.total_hits += thread_stats.total_hits;
#pragma omp atomic
        local_stats.average_ray_distance += thread_stats.average_ray_distance;
#pragma omp atomic
        local_stats.bvh_nodes_visited += thread_stats.bvh_nodes_visited;
    }

    if (local_stats.total_hits > 0) {
        local_stats.average_ray_distance /= local_stats.total_hits;
    }
    if (stats != nullptr) {
        *stats = local_stats;
    }
}

void CollisionDetection::castPacketSoATraversal(const helios::vec3 *origins, const helios::vec3 *directions, size_t begin, size_t end, float max_distance, float *out_distance, helios::vec3 *out_normal, uint *out_primitive_UUID,
                                                RayTracingStats &stats) {
    constexpr uint MISS_UUID = 0xFFFFFFFFu;
    const size_t n = end - begin;

    // Per-ray traversal state for this packet (small, stack-allocated). Packets larger than MAX_PACKET_RAYS are split
    // into sub-packets so the fixed-size buffers never overflow; LiDAR pulses are ~10-200 sub-rays so this is rare.
    constexpr size_t MAX_PACKET_RAYS = 256;
    if (n > MAX_PACKET_RAYS) {
        for (size_t sub = begin; sub < end; sub += MAX_PACKET_RAYS) {
            castPacketSoATraversal(origins, directions, sub, std::min(sub + MAX_PACKET_RAYS, end), max_distance, out_distance, out_normal, out_primitive_UUID, stats);
        }
        return;
    }

    const bool use_dense_cache = (primitive_cache_dense.size() == primitive_indices.size());

    // Per-ray running closest-hit distance and best hit so far.
    float closest[MAX_PACKET_RAYS];
    HitResult best[MAX_PACKET_RAYS];
    vec3 ray_origin[MAX_PACKET_RAYS];
    vec3 ray_dir[MAX_PACKET_RAYS];
    const float init_far = (max_distance > 0) ? max_distance : std::numeric_limits<float>::max();
    for (size_t r = 0; r < n; r++) {
        ray_origin[r] = origins[begin + r];
        ray_dir[r] = directions[begin + r];
        closest[r] = init_far;
        // best[r] default-constructed: hit == false
    }

    // Fixed-size shared traversal stack (BVH depth bounded by buildBVHRecursive MAX_DEPTH).
    constexpr int STACK_CAPACITY = 128;
    uint32_t node_stack[STACK_CAPACITY];
    int stack_size = 0;
    node_stack[stack_size++] = 0; // root

    while (stack_size > 0) {
        uint32_t node_idx = node_stack[--stack_size];
        if (node_idx >= bvh_nodes_soa.node_count) {
            continue;
        }

        if (bvh_nodes_soa.is_leaf_flags[node_idx]) {
            // Active-ray mask: only rays whose AABB (with their own running closest distance) intersects this leaf's node
            // can possibly hit a primitive inside it (the node AABB bounds all its primitives). Test the primitive against
            // just those rays, not all n. For a coherent LiDAR pulse only ~1/3 of the sub-rays reach a given leaf, so this
            // avoids ~2/3 of the (expensive) ray-primitive intersections while remaining identical to the per-ray result.
            int active_idx[MAX_PACKET_RAYS];
            int active_count = 0;
            for (size_t r = 0; r < n; r++) {
                if (aabbIntersectSoA(ray_origin[r], ray_dir[r], closest[r], node_idx)) {
                    active_idx[active_count++] = int(r);
                }
            }
            if (active_count == 0) {
                continue;
            }
            stats.bvh_nodes_visited++;

            const uint32_t primitive_start = bvh_nodes_soa.primitive_starts[node_idx];
            const uint32_t primitive_count = bvh_nodes_soa.primitive_counts[node_idx];

            for (uint32_t i = 0; i < primitive_count; ++i) {
                const uint32_t slot = primitive_start + i;
                const uint primitive_id = primitive_indices[slot];

                // Fetch the primitive once for the packet, then test it against the active rays only.
                for (int a = 0; a < active_count; ++a) {
                    const int r = active_idx[a];
                    HitResult pr = use_dense_cache ? intersectCachedPrimitive(ray_origin[r], ray_dir[r], primitive_cache_dense[slot], closest[r]) : intersectPrimitiveThreadSafe(ray_origin[r], ray_dir[r], primitive_id, closest[r]);
                    if (pr.hit && pr.distance < closest[r]) {
                        best[r] = pr;
                        closest[r] = pr.distance;
                    }
                }
            }
        } else {
            // Internal node: descend if ANY ray (using its own closest distance) intersects the node AABB. Early-break —
            // the full active set is only needed at leaves.
            bool any_hit_box = false;
            for (size_t r = 0; r < n; r++) {
                if (aabbIntersectSoA(ray_origin[r], ray_dir[r], closest[r], node_idx)) {
                    any_hit_box = true;
                    break;
                }
            }
            if (!any_hit_box) {
                continue;
            }
            stats.bvh_nodes_visited++;

            const uint32_t left_child = bvh_nodes_soa.left_children[node_idx];
            const uint32_t right_child = bvh_nodes_soa.right_children[node_idx];
            const bool left_valid = (left_child != 0xFFFFFFFF && left_child < bvh_nodes_soa.node_count);
            const bool right_valid = (right_child != 0xFFFFFFFF && right_child < bvh_nodes_soa.node_count);

            if (left_valid && right_valid) {
                // Near-first ordering using the packet's central ray (sub-ray 0 = the beam nominal axis): push the farther
                // child first so the nearer is popped first, tightening the running closest sooner (which then lets the
                // active-ray test above prune more leaves). Affects only traversal order, not the result.
                const float t_left = aabbEntryDistanceSoA(ray_origin[0], ray_dir[0], left_child);
                const float t_right = aabbEntryDistanceSoA(ray_origin[0], ray_dir[0], right_child);
                uint32_t first = left_child, second = right_child;
                if (t_right < t_left) {
                    first = right_child;
                    second = left_child;
                }
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = second; // farther pushed first (popped last)
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = first; // nearer pushed last (popped first)
            } else if (left_valid) {
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = left_child;
            } else if (right_valid) {
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = right_child;
            }
        }
    }

    // Write packet results.
    for (size_t r = 0; r < n; r++) {
        if (best[r].hit) {
            out_primitive_UUID[begin + r] = best[r].primitive_UUID;
            out_distance[begin + r] = best[r].distance;
            out_normal[begin + r] = best[r].normal;
            stats.total_hits++;
            stats.average_ray_distance += best[r].distance;
        } else {
            out_primitive_UUID[begin + r] = MISS_UUID;
        }
    }
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysSoA(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats) {
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());

    if (bvh_nodes_soa.node_count == 0) {
        // Return empty results if no SoA BVH, but still set statistics
        results.resize(ray_queries.size());
        stats.total_rays_cast = ray_queries.size();
        stats.total_hits = 0;
        stats.bvh_nodes_visited = 0;
        stats.average_ray_distance = 0.0f;
        return results;
    }

    stats.total_rays_cast = ray_queries.size();
    stats.total_hits = 0;
    stats.average_ray_distance = 0.0;

    // Resize results vector for parallel access
    results.resize(ray_queries.size());

    // Snapshot the external cancellation flag so every thread reads the same
    // pointer. When it flips non-zero mid-trace, each thread short-circuits its
    // remaining loop indices (OpenMP forbids breaking out of a worksharing loop,
    // so we skip-cheaply instead). The volatile load is re-read each iteration
    // and is effectively free next to a BVH traversal; left-over results[i] for
    // skipped rays stay default-constructed (no hit), which the caller discards.
    volatile int *const cancel = cancel_flag;

// OpenMP parallel ray processing for high-performance SoA traversal
#pragma omp parallel
    {
        RayTracingStats local_stats = {}; // Thread-local statistics

#pragma omp for schedule(guided, 32)
        for (int i = 0; i < static_cast<int>(ray_queries.size()); ++i) {
            if (cancel != nullptr && *cancel != 0) {
                continue; // run cancelled — drain remaining indices cheaply
            }
            HitResult result = castRaySoATraversal(ray_queries[i], local_stats);
            results[i] = result;

            if (result.hit) {
                local_stats.total_hits++;
                local_stats.average_ray_distance += result.distance;
            }
        }

// Combine thread-local statistics atomically
#pragma omp atomic
        stats.total_hits += local_stats.total_hits;

#pragma omp atomic
        stats.average_ray_distance += local_stats.average_ray_distance;

#pragma omp atomic
        stats.bvh_nodes_visited += local_stats.bvh_nodes_visited;
    }

    if (stats.total_hits > 0) {
        stats.average_ray_distance /= stats.total_hits;
    }

    return results;
}

// Optimized BVH traversal methods using Structure-of-Arrays layout

#include "CollisionDetection.h"

using namespace helios;

CollisionDetection::HitResult CollisionDetection::castRaySoATraversal(const RayQuery &query, RayTracingStats &stats) {
    HitResult result;

    if (bvh_nodes_soa.node_count == 0 || bvh_nodes_soa.aabb_mins.empty()) {
        return result; // No BVH built
    }

    // Fixed-size traversal stack (no per-ray heap allocation). An iterative DFS that pushes both children holds at most
    // ~one node per tree level, so the capacity must exceed buildBVHRecursive's MAX_DEPTH (64). 128 leaves generous
    // headroom; the guarded push below is a defensive backstop that must never actually trigger for a valid tree.
    constexpr int STACK_CAPACITY = 128;
    uint32_t node_stack[STACK_CAPACITY];
    int stack_size = 0;
    node_stack[stack_size++] = 0; // Start from root

    // Dense, BVH-leaf-ordered cache must be in lockstep with the SoA leaves' primitive ranges; fall back to the
    // UUID-keyed path if it has not been built (e.g. legacy callers that bypass buildPrimitiveCache()).
    const bool use_dense_cache = (primitive_cache_dense.size() == primitive_indices.size());

    float closest_distance = (query.max_distance > 0) ? query.max_distance : std::numeric_limits<float>::max();

    while (stack_size > 0) {
        uint32_t node_idx = node_stack[--stack_size];
        stats.bvh_nodes_visited++;

        // Bounds check for node index
        if (node_idx >= bvh_nodes_soa.node_count) {
            continue;
        }

        // AABB intersection test using SoA layout
        if (!aabbIntersectSoA(query.origin, query.direction, closest_distance, node_idx)) {
            continue;
        }

        // Check if leaf node
        if (bvh_nodes_soa.is_leaf_flags[node_idx]) {
            // Process primitives in this leaf
            uint32_t primitive_start = bvh_nodes_soa.primitive_starts[node_idx];
            uint32_t primitive_count = bvh_nodes_soa.primitive_counts[node_idx];

            for (uint32_t i = 0; i < primitive_count; ++i) {
                const uint32_t slot = primitive_start + i;
                uint primitive_id = primitive_indices[slot];

                // Skip if not in target list (if specified)
                if (!query.target_UUIDs.empty()) {
                    bool found = false;
                    for (uint target: query.target_UUIDs) {
                        if (primitive_id == target) {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        continue;
                }

                // Hot path: dense-cache slot lookup (no unordered_map find). Falls back to the UUID-keyed
                // intersection when the dense cache is unavailable.
                HitResult primitive_result = use_dense_cache ? intersectCachedPrimitive(query.origin, query.direction, primitive_cache_dense[slot], closest_distance)
                                                             : intersectPrimitiveThreadSafe(query.origin, query.direction, primitive_id, closest_distance);
                if (primitive_result.hit && primitive_result.distance < closest_distance) {
                    result = primitive_result;
                    closest_distance = primitive_result.distance;
                }
            }
        } else {
            // Internal node - push children. Descend the nearer child first (pushed last, popped first) so the
            // farther subtree can be pruned by closest_distance once a near hit is found.
            uint32_t left_child = bvh_nodes_soa.left_children[node_idx];
            uint32_t right_child = bvh_nodes_soa.right_children[node_idx];

            const bool left_valid = (left_child != 0xFFFFFFFF && left_child < bvh_nodes_soa.node_count);
            const bool right_valid = (right_child != 0xFFFFFFFF && right_child < bvh_nodes_soa.node_count);

            if (left_valid && right_valid) {
                // Order by entry distance into each child's AABB so the nearer child is popped first.
                const float t_left = aabbEntryDistanceSoA(query.origin, query.direction, left_child);
                const float t_right = aabbEntryDistanceSoA(query.origin, query.direction, right_child);
                uint32_t first = left_child, second = right_child;
                if (t_right < t_left) {
                    first = right_child;
                    second = left_child;
                }
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = second; // farther child pushed first (popped last)
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = first; // nearer child pushed last (popped first)
            } else if (left_valid) {
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = left_child;
            } else if (right_valid) {
                if (stack_size < STACK_CAPACITY)
                    node_stack[stack_size++] = right_child;
            }
        }
    }

    return result;
}


bool CollisionDetection::aabbIntersectSoA(const helios::vec3 &ray_origin, const helios::vec3 &ray_direction, float max_distance, size_t node_index) const {
    // Direct access to SoA arrays for optimal memory usage
    const vec3 &aabb_min = bvh_nodes_soa.aabb_mins[node_index];
    const vec3 &aabb_max = bvh_nodes_soa.aabb_maxs[node_index];

#ifdef __SSE4_1__
    // SIMD-optimized ray-AABB intersection for better performance
    __m128 ray_orig = _mm_set_ps(0.0f, ray_origin.z, ray_origin.y, ray_origin.x);
    __m128 ray_dir = _mm_set_ps(0.0f, ray_direction.z, ray_direction.y, ray_direction.x);
    __m128 aabb_min_vec = _mm_set_ps(0.0f, aabb_min.z, aabb_min.y, aabb_min.x);
    __m128 aabb_max_vec = _mm_set_ps(0.0f, aabb_max.z, aabb_max.y, aabb_max.x);

    // Compute inverse ray direction. Clamp near-zero direction components away from zero before the reciprocal so an
    // axis-aligned ray lying exactly on a box face computes (bound-origin)*huge == 0 (finite) instead of
    // (bound-origin==0)*inf == NaN, which would corrupt the min/max slab test and spuriously reject the box.
    constexpr float PARALLEL_EPS = 1e-8f;
    __m128 abs_dir = _mm_andnot_ps(_mm_set1_ps(-0.0f), ray_dir); // |ray_dir|
    __m128 too_small = _mm_cmplt_ps(abs_dir, _mm_set1_ps(PARALLEL_EPS));
    // Replace near-zero components with +/-PARALLEL_EPS preserving sign (sign bit of the original component).
    __m128 sign = _mm_and_ps(ray_dir, _mm_set1_ps(-0.0f));
    __m128 clamped = _mm_or_ps(sign, _mm_set1_ps(PARALLEL_EPS));
    __m128 safe_dir = _mm_or_ps(_mm_and_ps(too_small, clamped), _mm_andnot_ps(too_small, ray_dir));
    __m128 inv_dir = _mm_div_ps(_mm_set1_ps(1.0f), safe_dir);

    // Compute t1 and t2 for all axes
    __m128 t1 = _mm_mul_ps(_mm_sub_ps(aabb_min_vec, ray_orig), inv_dir);
    __m128 t2 = _mm_mul_ps(_mm_sub_ps(aabb_max_vec, ray_orig), inv_dir);

    // Get min and max for each axis
    __m128 tmin = _mm_min_ps(t1, t2);
    __m128 tmax = _mm_max_ps(t1, t2);

    // Extract components
    float tmin_vals[4], tmax_vals[4];
    _mm_store_ps(tmin_vals, tmin);
    _mm_store_ps(tmax_vals, tmax);

    float t_near = std::max({tmin_vals[0], tmin_vals[1], tmin_vals[2], 0.0f});
    float t_far = std::min({tmax_vals[0], tmax_vals[1], tmax_vals[2], max_distance});

    return t_near <= t_far;
#else
    // Fallback scalar implementation. Each axis is handled explicitly so that an axis-aligned ray lying exactly on a
    // box face (direction component == 0) does not produce a 0*inf == NaN t-value that would corrupt the slab test
    // and spuriously reject the box. A ray parallel to an axis simply imposes no near/far constraint on that axis as
    // long as its origin is within the slab; if the origin is outside the slab it misses outright.
    float t_near = 0.0f;
    float t_far = max_distance;

    const float origin_xyz[3] = {ray_origin.x, ray_origin.y, ray_origin.z};
    const float dir_xyz[3] = {ray_direction.x, ray_direction.y, ray_direction.z};
    const float min_xyz[3] = {aabb_min.x, aabb_min.y, aabb_min.z};
    const float max_xyz[3] = {aabb_max.x, aabb_max.y, aabb_max.z};

    constexpr float PARALLEL_EPS = 1e-8f;
    for (int axis = 0; axis < 3; axis++) {
        if (std::abs(dir_xyz[axis]) < PARALLEL_EPS) {
            // Ray parallel to this slab: it can only hit the box if its origin lies within the slab bounds.
            if (origin_xyz[axis] < min_xyz[axis] || origin_xyz[axis] > max_xyz[axis]) {
                return false;
            }
            continue; // no t constraint from this axis
        }
        const float inv = 1.0f / dir_xyz[axis];
        float t1 = (min_xyz[axis] - origin_xyz[axis]) * inv;
        float t2 = (max_xyz[axis] - origin_xyz[axis]) * inv;
        if (t1 > t2) {
            std::swap(t1, t2);
        }
        t_near = std::max(t_near, t1);
        t_far = std::min(t_far, t2);
        if (t_near > t_far) {
            return false;
        }
    }

    return t_near <= t_far;
#endif
}

float CollisionDetection::aabbEntryDistanceSoA(const helios::vec3 &ray_origin, const helios::vec3 &ray_direction, size_t node_index) const {
    // Entry distance (t_near, clamped to >=0) of the ray into the node's AABB, used only to order the two
    // children for near-first traversal. Returns +inf when the ray misses the box so a missed child sorts last.
    // Handles axis-aligned (parallel) rays explicitly to avoid 0*inf == NaN corrupting the comparison.
    const vec3 &aabb_min = bvh_nodes_soa.aabb_mins[node_index];
    const vec3 &aabb_max = bvh_nodes_soa.aabb_maxs[node_index];

    const float origin_xyz[3] = {ray_origin.x, ray_origin.y, ray_origin.z};
    const float dir_xyz[3] = {ray_direction.x, ray_direction.y, ray_direction.z};
    const float min_xyz[3] = {aabb_min.x, aabb_min.y, aabb_min.z};
    const float max_xyz[3] = {aabb_max.x, aabb_max.y, aabb_max.z};

    constexpr float PARALLEL_EPS = 1e-8f;
    float t_near = 0.0f;
    float t_far = std::numeric_limits<float>::max();
    for (int axis = 0; axis < 3; axis++) {
        if (std::abs(dir_xyz[axis]) < PARALLEL_EPS) {
            if (origin_xyz[axis] < min_xyz[axis] || origin_xyz[axis] > max_xyz[axis]) {
                return std::numeric_limits<float>::max();
            }
            continue;
        }
        const float inv = 1.0f / dir_xyz[axis];
        float t1 = (min_xyz[axis] - origin_xyz[axis]) * inv;
        float t2 = (max_xyz[axis] - origin_xyz[axis]) * inv;
        if (t1 > t2) {
            std::swap(t1, t2);
        }
        t_near = std::max(t_near, t1);
        t_far = std::min(t_far, t2);
    }

    return (t_near <= t_far) ? t_near : std::numeric_limits<float>::max();
}


// Basic BVH traversal using standard node structure
CollisionDetection::HitResult CollisionDetection::castRayBVHTraversal(const RayQuery &query) {
    HitResult result;

    if (bvh_nodes.empty()) {
        return result; // No BVH built
    }

    // Stack-based traversal using standard BVH nodes
    std::stack<size_t> node_stack;
    node_stack.push(0); // Start from root

    float closest_distance = (query.max_distance > 0) ? query.max_distance : std::numeric_limits<float>::max();

    while (!node_stack.empty()) {
        size_t node_idx = node_stack.top();
        node_stack.pop();

        if (node_idx >= bvh_nodes.size()) {
            if (printmessages) {
                std::cout << "ERROR: Invalid BVH node index " << node_idx << " >= " << bvh_nodes.size() << " nodes" << std::endl;
            }
            result.hit = false;
            return result;
        }

        const BVHNode &node = bvh_nodes[node_idx];

        // AABB intersection test - this provides the early miss detection that was missing
        if (!rayAABBIntersect(query.origin, query.direction, node.aabb_min, node.aabb_max)) {
            continue; // Ray misses this node's bounding box - skip entire subtree
        }

        if (node.is_leaf) {
            // Process primitives in this leaf
            for (uint32_t i = 0; i < node.primitive_count; ++i) {
                if (node.primitive_start + i >= primitive_indices.size()) {
                    if (printmessages) {
                        std::cout << "ERROR: Invalid BVH primitive index " << (node.primitive_start + i) << " >= " << primitive_indices.size() << " primitives" << std::endl;
                    }
                    result.hit = false;
                    return result;
                }

                uint primitive_id = primitive_indices[node.primitive_start + i];

                // Skip if not in target list (if specified)
                if (!query.target_UUIDs.empty()) {
                    bool found = false;
                    for (uint target: query.target_UUIDs) {
                        if (primitive_id == target) {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        continue;
                }

                // Perform primitive intersection test
                HitResult primitive_hit = intersectPrimitive(query, primitive_id);
                if (primitive_hit.hit && primitive_hit.distance < closest_distance) {
                    result = primitive_hit;
                    closest_distance = primitive_hit.distance;
                }
            }
        } else {
            // Internal node - add children to stack for traversal
            if (node.left_child != 0xFFFFFFFF && node.left_child < bvh_nodes.size()) {
                node_stack.push(node.left_child);
            }
            if (node.right_child != 0xFFFFFFFF && node.right_child < bvh_nodes.size()) {
                node_stack.push(node.right_child);
            }
        }
    }

    return result;
}

// Helper method to test ray-AABB intersection
bool CollisionDetection::rayAABBIntersect(const vec3 &ray_origin, const vec3 &ray_direction, const vec3 &aabb_min, const vec3 &aabb_max) const {
    // Robust ray-AABB intersection using slab method with proper axis-aligned ray handling
    const float EPSILON = 1e-8f;

    float tmin = 0.0f; // Ray starts at origin
    float tmax = std::numeric_limits<float>::max(); // No maximum distance limit

    // Handle X slab
    if (std::abs(ray_direction.x) > EPSILON) {
        float inv_dir_x = 1.0f / ray_direction.x;
        float t1 = (aabb_min.x - ray_origin.x) * inv_dir_x;
        float t2 = (aabb_max.x - ray_origin.x) * inv_dir_x;

        float slab_tmin = std::min(t1, t2);
        float slab_tmax = std::max(t1, t2);

        tmin = std::max(tmin, slab_tmin);
        tmax = std::min(tmax, slab_tmax);

        if (tmin > tmax)
            return false; // Early exit if no intersection
    } else {
        // Ray is parallel to X slab - check if ray origin is within X bounds
        if (ray_origin.x < aabb_min.x || ray_origin.x > aabb_max.x) {
            return false;
        }
    }

    // Handle Y slab
    if (std::abs(ray_direction.y) > EPSILON) {
        float inv_dir_y = 1.0f / ray_direction.y;
        float t1 = (aabb_min.y - ray_origin.y) * inv_dir_y;
        float t2 = (aabb_max.y - ray_origin.y) * inv_dir_y;

        float slab_tmin = std::min(t1, t2);
        float slab_tmax = std::max(t1, t2);

        tmin = std::max(tmin, slab_tmin);
        tmax = std::min(tmax, slab_tmax);

        if (tmin > tmax)
            return false; // Early exit if no intersection
    } else {
        // Ray is parallel to Y slab - check if ray origin is within Y bounds
        if (ray_origin.y < aabb_min.y || ray_origin.y > aabb_max.y) {
            return false;
        }
    }

    // Handle Z slab
    if (std::abs(ray_direction.z) > EPSILON) {
        float inv_dir_z = 1.0f / ray_direction.z;
        float t1 = (aabb_min.z - ray_origin.z) * inv_dir_z;
        float t2 = (aabb_max.z - ray_origin.z) * inv_dir_z;

        float slab_tmin = std::min(t1, t2);
        float slab_tmax = std::max(t1, t2);

        tmin = std::max(tmin, slab_tmin);
        tmax = std::min(tmax, slab_tmax);

        if (tmin > tmax)
            return false; // Early exit if no intersection
    } else {
        // Ray is parallel to Z slab - check if ray origin is within Z bounds
        if (ray_origin.z < aabb_min.z || ray_origin.z > aabb_max.z) {
            return false;
        }
    }

    return tmin <= tmax;
}

// Helper method to intersect with individual primitive (reuses existing logic)
CollisionDetection::HitResult CollisionDetection::intersectPrimitive(const RayQuery &query, uint primitive_id) {
    // PERFORMANCE FIX: Use direct context call instead of expensive cached primitive data
    // This avoids the need to build and maintain a primitive cache
    HitResult result;

    float distance;
    if (rayPrimitiveIntersection(query.origin, query.direction, primitive_id, distance)) {
        vec3 intersection_point = query.origin + query.direction * distance;

        // Calculate surface normal directly from context (minimal overhead)
        PrimitiveType type = context->getPrimitiveType(primitive_id);
        std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_id);

        // Reject hits on transparent texels so the ray passes through to geometry behind.
        if (context->primitiveTextureHasTransparencyChannel(primitive_id)) {
            CachedPrimitive temp_cached(type, vertices);
            temp_cached.transparency_mask = context->getPrimitiveTextureTransparencyData(primitive_id);
            temp_cached.texture_size = context->getPrimitiveTextureSize(primitive_id);
            temp_cached.uv = context->getPrimitiveTextureUV(primitive_id);
            if (!isHitTexelOpaque(temp_cached, intersection_point)) {
                return result; // result.hit is still false
            }
        }

        result.hit = true;
        result.distance = distance;
        result.primitive_UUID = primitive_id;
        result.intersection_point = intersection_point;

        if (type == PRIMITIVE_TYPE_TRIANGLE && vertices.size() >= 3) {
            vec3 edge1 = vertices[1] - vertices[0];
            vec3 edge2 = vertices[2] - vertices[0];
            result.normal = cross(edge1, edge2);
            if (result.normal.magnitude() > 1e-8f) {
                result.normal = result.normal / result.normal.magnitude();
            } else {
                result.normal = make_vec3(-query.direction.x, -query.direction.y, -query.direction.z);
            }
        } else if (type == PRIMITIVE_TYPE_PATCH && vertices.size() >= 3) {
            vec3 edge1 = vertices[1] - vertices[0];
            vec3 edge2 = vertices[2] - vertices[0];
            result.normal = cross(edge1, edge2);
            if (result.normal.magnitude() > 1e-8f) {
                result.normal = result.normal / result.normal.magnitude();
            } else {
                result.normal = make_vec3(-query.direction.x, -query.direction.y, -query.direction.z);
            }
        } else {
            // Default normal (opposite to ray direction)
            result.normal = make_vec3(-query.direction.x, -query.direction.y, -query.direction.z);
        }
    }

    return result;
}

bool CollisionDetection::isHitTexelOpaque(const CachedPrimitive &cached, const vec3 &hit_point) const {

    // No transparency mask -> primitive is fully solid.
    if (cached.transparency_mask == nullptr || cached.texture_size.x <= 0 || cached.texture_size.y <= 0) {
        return true;
    }

    const std::vector<vec2> &uvs = cached.uv;

    // Compute the (u,v) texture coordinate at the hit point. The interpolation mirrors the
    // verified logic in LiDARcloud::syntheticScan (sample_hit_color) so that the texel selected
    // here matches the texel later used for hit-point coloring.
    vec2 uv;
    if (cached.type == PRIMITIVE_TYPE_PATCH && cached.vertices.size() >= 4) {
        // Patch corners are (BL, BR, TR, TL); project the hit onto the (BL->BR, BL->TL) basis.
        const vec3 e1 = cached.vertices[1] - cached.vertices[0];
        const vec3 e2 = cached.vertices[3] - cached.vertices[0];
        const vec3 d = hit_point - cached.vertices[0];
        const float e1_sq = e1 * e1;
        const float e2_sq = e2 * e2;
        float s_param = (e1_sq > 0.f) ? (d * e1) / e1_sq : 0.f;
        float t_param = (e2_sq > 0.f) ? (d * e2) / e2_sq : 0.f;
        s_param = std::min(std::max(s_param, 0.f), 1.f);
        t_param = std::min(std::max(t_param, 0.f), 1.f);
        if (uvs.size() == 4) {
            uv = (1.f - s_param) * (1.f - t_param) * uvs[0] + s_param * (1.f - t_param) * uvs[1] + s_param * t_param * uvs[2] + (1.f - s_param) * t_param * uvs[3];
        } else {
            uv = make_vec2(s_param, t_param);
        }
    } else if (cached.type == PRIMITIVE_TYPE_TRIANGLE && cached.vertices.size() >= 3 && uvs.size() == 3) {
        const vec3 e1 = cached.vertices[1] - cached.vertices[0];
        const vec3 e2 = cached.vertices[2] - cached.vertices[0];
        const vec3 d = hit_point - cached.vertices[0];
        const float dot11 = e1 * e1;
        const float dot12 = e1 * e2;
        const float dot22 = e2 * e2;
        const float dot1d = e1 * d;
        const float dot2d = e2 * d;
        const float denom = dot11 * dot22 - dot12 * dot12;
        if (std::fabs(denom) < 1e-20f) {
            return true; // degenerate triangle - cannot map UV, treat as solid
        }
        const float inv_denom = 1.f / denom;
        const float beta = (dot22 * dot1d - dot12 * dot2d) * inv_denom;
        const float gamma = (dot11 * dot2d - dot12 * dot1d) * inv_denom;
        uv = uvs[0] + beta * (uvs[1] - uvs[0]) + gamma * (uvs[2] - uvs[0]);
    } else {
        // Unsupported configuration (e.g. missing UVs) - cannot map the texel, treat as solid.
        return true;
    }

    // Wrap UV into [0,1) so repeat-style mappings sample correctly.
    uv.x -= std::floor(uv.x);
    uv.y -= std::floor(uv.y);

    int px = static_cast<int>(uv.x * static_cast<float>(cached.texture_size.x));
    px = std::min(std::max(px, 0), cached.texture_size.x - 1);
    // Mask rows are stored top-to-bottom (row 0 = top of image); UV y=0 is the bottom.
    int py = static_cast<int>((1.f - uv.y) * static_cast<float>(cached.texture_size.y));
    py = std::min(std::max(py, 0), cached.texture_size.y - 1);

    const std::vector<std::vector<bool>> &mask = *cached.transparency_mask;
    if (py >= static_cast<int>(mask.size()) || px >= static_cast<int>(mask[py].size())) {
        return true; // out-of-bounds guard - treat as solid rather than dropping the hit
    }
    return mask[py][px]; // true => opaque/solid texel
}

CollisionDetection::HitResult CollisionDetection::intersectPrimitiveThreadSafe(const vec3 &origin, const vec3 &direction, uint primitive_id, float max_distance) {
    HitResult result;

    // Check if we have cached primitive data for this primitive
    auto it = primitive_cache.find(primitive_id);
    if (it == primitive_cache.end()) {
        // Primitive not in cache - this shouldn't happen in optimized paths
        // This indicates that the context was modified after the cache was built
        // Fall back to thread-unsafe context call (only safe for sequential code)
        // For parallel regions, this could cause issues, but it's better than crashing
        float distance;
        if (rayPrimitiveIntersection(origin, direction, primitive_id, distance)) {
            vec3 intersection_point = origin + direction * distance;

            // Calculate surface normal for uncached primitive
            PrimitiveType type = context->getPrimitiveType(primitive_id);
            std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_id);

            // Reject hits on transparent texels (build a temporary cache entry from the Context;
            // this fallback path is already thread-unsafe so the extra Context reads are acceptable).
            if (context->primitiveTextureHasTransparencyChannel(primitive_id)) {
                CachedPrimitive temp_cached(type, vertices);
                temp_cached.transparency_mask = context->getPrimitiveTextureTransparencyData(primitive_id);
                temp_cached.texture_size = context->getPrimitiveTextureSize(primitive_id);
                temp_cached.uv = context->getPrimitiveTextureUV(primitive_id);
                if (!isHitTexelOpaque(temp_cached, intersection_point)) {
                    return result; // result.hit is still false
                }
            }

            result.hit = true;
            result.distance = distance;
            result.primitive_UUID = primitive_id;
            result.intersection_point = intersection_point;

            if (type == PRIMITIVE_TYPE_TRIANGLE && vertices.size() >= 3) {
                vec3 edge1 = vertices[1] - vertices[0];
                vec3 edge2 = vertices[2] - vertices[0];
                result.normal = cross(edge1, edge2);
                if (result.normal.magnitude() > 1e-8f) {
                    result.normal = result.normal / result.normal.magnitude();
                    // Ensure normal points towards ray origin (for LiDAR compatibility)
                    vec3 to_origin = origin - result.intersection_point;
                    if (result.normal * to_origin < 0) {
                        result.normal = result.normal * -1.0f;
                    }
                } else {
                    result.normal = make_vec3(-direction.x, -direction.y, -direction.z);
                    result.normal = result.normal / result.normal.magnitude();
                }
            } else if (type == PRIMITIVE_TYPE_PATCH && vertices.size() >= 4) {
                vec3 edge1 = vertices[1] - vertices[0];
                vec3 edge2 = vertices[2] - vertices[0];
                result.normal = cross(edge1, edge2);
                if (result.normal.magnitude() > 1e-8f) {
                    result.normal = result.normal / result.normal.magnitude();
                    // Ensure normal points towards ray origin (for LiDAR compatibility)
                    vec3 to_origin = origin - result.intersection_point;
                    if (result.normal * to_origin < 0) {
                        result.normal = result.normal * -1.0f;
                    }
                } else {
                    result.normal = make_vec3(-direction.x, -direction.y, -direction.z);
                    result.normal = result.normal / result.normal.magnitude();
                }
            } else {
                result.normal = make_vec3(-direction.x, -direction.y, -direction.z);
                result.normal = result.normal / result.normal.magnitude();
            }
        }
        return result;
    }

    // Cache hit: delegate to the shared cached-primitive intersection (same logic as the dense-cache hot path).
    return intersectCachedPrimitive(origin, direction, it->second, max_distance);
}

CollisionDetection::HitResult CollisionDetection::intersectCachedPrimitive(const vec3 &origin, const vec3 &direction, const CachedPrimitive &cached, float max_distance) const {
    HitResult result;

    // Perform intersection test based on primitive type
    if (cached.type == PRIMITIVE_TYPE_TRIANGLE && cached.vertices.size() >= 3) {
        float distance;
        if (triangleIntersect(origin, direction, cached.vertices[0], cached.vertices[1], cached.vertices[2], distance)) {
            if (distance > 1e-6f && (max_distance <= 0 || distance < max_distance)) {
                vec3 intersection_point = origin + direction * distance;

                // Reject hits on transparent texels so the ray passes through to geometry behind.
                if (!isHitTexelOpaque(cached, intersection_point)) {
                    return result; // result.hit is still false
                }

                result.hit = true;
                result.distance = distance;
                result.primitive_UUID = cached.UUID;
                result.intersection_point = intersection_point;

                // Calculate triangle normal
                vec3 edge1 = cached.vertices[1] - cached.vertices[0];
                vec3 edge2 = cached.vertices[2] - cached.vertices[0];
                result.normal = cross(edge1, edge2);
                if (result.normal.magnitude() > 1e-8f) {
                    result.normal = result.normal / result.normal.magnitude();
                    // Ensure normal points towards ray origin (for LiDAR compatibility)
                    vec3 to_origin = origin - result.intersection_point;
                    if (result.normal * to_origin < 0) {
                        result.normal = result.normal * -1.0f;
                    }
                } else {
                    result.normal = make_vec3(-direction.x, -direction.y, -direction.z);
                    result.normal = result.normal / result.normal.magnitude();
                }
            }
        }
    } else if (cached.type == PRIMITIVE_TYPE_PATCH && cached.vertices.size() >= 4) {
        float distance;
        if (patchIntersect(origin, direction, cached.vertices[0], cached.vertices[1], cached.vertices[2], cached.vertices[3], distance)) {
            if (distance > 1e-6f && (max_distance <= 0 || distance < max_distance)) {
                vec3 intersection_point = origin + direction * distance;

                // Reject hits on transparent texels so the ray passes through to geometry behind.
                if (!isHitTexelOpaque(cached, intersection_point)) {
                    return result; // result.hit is still false
                }

                result.hit = true;
                result.distance = distance;
                result.primitive_UUID = cached.UUID;
                result.intersection_point = intersection_point;

                // Calculate patch normal (use v0, v1, v2 like the original code)
                vec3 edge1 = cached.vertices[1] - cached.vertices[0];
                vec3 edge2 = cached.vertices[2] - cached.vertices[0];
                result.normal = cross(edge1, edge2);
                if (result.normal.magnitude() > 1e-8f) {
                    result.normal = result.normal / result.normal.magnitude();
                    // Ensure normal points towards ray origin (for LiDAR compatibility)
                    vec3 to_origin = origin - result.intersection_point;
                    if (result.normal * to_origin < 0) {
                        result.normal = result.normal * -1.0f;
                    }
                } else {
                    result.normal = make_vec3(-direction.x, -direction.y, -direction.z);
                    result.normal = result.normal / result.normal.magnitude();
                }
            }
        }
    } else if (cached.type == PRIMITIVE_TYPE_VOXEL && cached.vertices.size() == 8) {
        // Voxel (AABB) intersection using slab method
        // Calculate AABB from 8 vertices
        vec3 aabb_min = cached.vertices[0];
        vec3 aabb_max = cached.vertices[0];

        for (int i = 1; i < 8; i++) {
            aabb_min.x = std::min(aabb_min.x, cached.vertices[i].x);
            aabb_min.y = std::min(aabb_min.y, cached.vertices[i].y);
            aabb_min.z = std::min(aabb_min.z, cached.vertices[i].z);
            aabb_max.x = std::max(aabb_max.x, cached.vertices[i].x);
            aabb_max.y = std::max(aabb_max.y, cached.vertices[i].y);
            aabb_max.z = std::max(aabb_max.z, cached.vertices[i].z);
        }

        // Ray-AABB intersection using slab method
        float t_near = -std::numeric_limits<float>::max();
        float t_far = std::numeric_limits<float>::max();

        // Check intersection with each slab (X, Y, Z)
        for (int axis = 0; axis < 3; axis++) {
            float ray_dir_component = (axis == 0) ? direction.x : (axis == 1) ? direction.y : direction.z;
            float ray_orig_component = (axis == 0) ? origin.x : (axis == 1) ? origin.y : origin.z;
            float aabb_min_component = (axis == 0) ? aabb_min.x : (axis == 1) ? aabb_min.y : aabb_min.z;
            float aabb_max_component = (axis == 0) ? aabb_max.x : (axis == 1) ? aabb_max.y : aabb_max.z;

            if (std::abs(ray_dir_component) < 1e-8f) {
                // Ray is parallel to slab
                if (ray_orig_component < aabb_min_component || ray_orig_component > aabb_max_component) {
                    return result; // Ray is outside slab and parallel - no intersection
                }
            } else {
                // Calculate intersection distances for this slab
                float t1 = (aabb_min_component - ray_orig_component) / ray_dir_component;
                float t2 = (aabb_max_component - ray_orig_component) / ray_dir_component;

                // Ensure t1 <= t2
                if (t1 > t2) {
                    std::swap(t1, t2);
                }

                // Update near and far intersection distances
                t_near = std::max(t_near, t1);
                t_far = std::min(t_far, t2);

                // Early exit if no intersection possible
                if (t_near > t_far) {
                    return result;
                }
            }
        }

        // Check if intersection is in front of ray origin and within max distance
        if (t_far >= 0.0f) {
            // Use t_near if it's positive (ray starts outside box), otherwise t_far (ray starts inside box)
            float intersection_distance = (t_near >= 1e-6f) ? t_near : t_far;
            if (intersection_distance >= 1e-6f && (max_distance <= 0 || intersection_distance < max_distance)) {
                result.hit = true;
                result.distance = intersection_distance;
                result.primitive_UUID = cached.UUID;
                result.intersection_point = origin + direction * intersection_distance;

                // Calculate normal based on which face was hit
                // Determine which face of the voxel was hit by examining intersection point
                vec3 hit_point = result.intersection_point;
                vec3 box_center = (aabb_min + aabb_max) * 0.5f;
                vec3 box_extent = (aabb_max - aabb_min) * 0.5f;

                // Find which face the hit point is closest to
                vec3 local_hit = hit_point - box_center;
                vec3 abs_local_hit = make_vec3(std::abs(local_hit.x), std::abs(local_hit.y), std::abs(local_hit.z));

                // Determine which axis has the largest relative coordinate (closest to face)
                float rel_x = abs_local_hit.x / box_extent.x;
                float rel_y = abs_local_hit.y / box_extent.y;
                float rel_z = abs_local_hit.z / box_extent.z;

                if (rel_x >= rel_y && rel_x >= rel_z) {
                    // Hit X face
                    result.normal = make_vec3((local_hit.x > 0) ? 1.0f : -1.0f, 0.0f, 0.0f);
                } else if (rel_y >= rel_z) {
                    // Hit Y face
                    result.normal = make_vec3(0.0f, (local_hit.y > 0) ? 1.0f : -1.0f, 0.0f);
                } else {
                    // Hit Z face
                    result.normal = make_vec3(0.0f, 0.0f, (local_hit.z > 0) ? 1.0f : -1.0f);
                }
            }
        }
    }
    // Add other primitive types as needed (DISK, etc.)

    return result;
}

void CollisionDetection::buildPrimitiveCache() {
    primitive_cache.clear();

    // Get all primitive UUIDs from context
    std::vector<uint> all_primitives = context->getAllUUIDs();

    // Cache primitive data for thread-safe access
    for (uint primitive_id: all_primitives) {
        if (context->doesPrimitiveExist(primitive_id)) {
            try {
                PrimitiveType type = context->getPrimitiveType(primitive_id);
                std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_id);

                CachedPrimitive cached(type, vertices);
                cached.UUID = primitive_id;

                // Cache texture transparency data so that ray hits on transparent texels can be
                // rejected during traversal (mirrors the OptiX rtIgnoreIntersection behavior).
                // Reading the Context here keeps the parallel traversal thread-safe. Primitives
                // without a transparency channel (no texture, or e.g. JPEG) keep a null mask and
                // are treated as fully solid, preserving the original behavior.
                if (context->primitiveTextureHasTransparencyChannel(primitive_id)) {
                    cached.transparency_mask = context->getPrimitiveTextureTransparencyData(primitive_id);
                    cached.texture_size = context->getPrimitiveTextureSize(primitive_id);
                    cached.uv = context->getPrimitiveTextureUV(primitive_id);
                }

                primitive_cache[primitive_id] = std::move(cached);
            } catch (const std::exception &e) {
                // Skip this primitive if it no longer exists or can't be accessed
                // This can happen when UUIDs from previous contexts persist
                if (printmessages) {
                    std::cout << "Warning: Skipping primitive " << primitive_id << " in cache build (not accessible: " << e.what() << ")" << std::endl;
                }
                continue;
            }
        }
    }

    // Build the dense, BVH-leaf-ordered cache to match the current primitive_indices ordering.
    rebuildDensePrimitiveCache();
}

void CollisionDetection::rebuildDensePrimitiveCache() {
    // Build the dense, BVH-leaf-ordered cache (slot i <-> primitive_indices[i]) from the UUID-keyed cache so the
    // hot traversal loop indexes it directly without an unordered_map lookup. Must be called whenever
    // primitive_indices is (re)ordered — buildBVH() reorders it in place even when the primitive set is unchanged.
    // Any primitive_indices entry with no cache entry gets a default CachedPrimitive whose empty vertex list makes
    // every intersection test fail safely — the same outcome as the previous find()==end() miss.
    primitive_cache_dense.assign(primitive_indices.size(), CachedPrimitive());
    for (size_t i = 0; i < primitive_indices.size(); i++) {
        auto it = primitive_cache.find(primitive_indices[i]);
        if (it != primitive_cache.end()) {
            primitive_cache_dense[i] = it->second;
        }
    }
}

void CollisionDetection::ensurePrimitiveCacheCurrent() {
    // Build the UUID-keyed cache the first time (or after a primitive-set change clears it). buildPrimitiveCache()
    // also (re)builds the dense cache in the correct order.
    if (primitive_cache.empty()) {
        buildPrimitiveCache();
        return;
    }
    // The UUID-keyed cache is valid but the dense cache may be out of sync with the current primitive_indices order
    // (a BVH rebuild over an unchanged primitive set reorders primitive_indices and clears the dense cache). Rebuild
    // just the cheap dense ordering from the existing map in that case.
    if (primitive_cache_dense.size() != primitive_indices.size()) {
        rebuildDensePrimitiveCache();
    }
}

bool CollisionDetection::triangleIntersect(const vec3 &origin, const vec3 &direction, const vec3 &v0, const vec3 &v1, const vec3 &v2, float &distance) const {
    // Möller-Trumbore triangle intersection algorithm (optimized - no vec3 temporaries)
    // Note: Using 1e-5f to match LiDAR CUDA kernel tolerance for edge-case rays
    const float EPSILON = 1e-5f;

    // Compute triangle edges directly as components (avoid vec3 constructors)
    float edge1_x = v1.x - v0.x, edge1_y = v1.y - v0.y, edge1_z = v1.z - v0.z;
    float edge2_x = v2.x - v0.x, edge2_y = v2.y - v0.y, edge2_z = v2.z - v0.z;

    // Cross product: h = direction × edge2 (computed directly)
    float h_x = direction.y * edge2_z - direction.z * edge2_y;
    float h_y = direction.z * edge2_x - direction.x * edge2_z;
    float h_z = direction.x * edge2_y - direction.y * edge2_x;

    // Dot product: a = edge1 · h
    float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;

    if (a > -EPSILON && a < EPSILON) {
        return false; // Ray is parallel to triangle
    }

    float f = 1.0f / a;

    // Vector s = origin - v0 (computed as components)
    float s_x = origin.x - v0.x, s_y = origin.y - v0.y, s_z = origin.z - v0.z;

    // u = f * (s · h)
    float u = f * (s_x * h_x + s_y * h_y + s_z * h_z);

    if (u < -EPSILON || u > 1.0f + EPSILON) {
        return false;
    }

    // Cross product: q = s × edge1 (computed directly)
    float q_x = s_y * edge1_z - s_z * edge1_y;
    float q_y = s_z * edge1_x - s_x * edge1_z;
    float q_z = s_x * edge1_y - s_y * edge1_x;

    // v = f * (direction · q)
    float v = f * (direction.x * q_x + direction.y * q_y + direction.z * q_z);

    if (v < -EPSILON || u + v > 1.0f + EPSILON) {
        return false;
    }

    // t = f * (edge2 · q) - computed directly as dot product
    float t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);

    if (t > EPSILON) {
        distance = t;
        return true;
    }

    return false; // Line intersection but not ray intersection
}

bool CollisionDetection::patchIntersect(const vec3 &origin, const vec3 &direction, const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &v3, float &distance) const {
    // Patch (quadrilateral) intersection using radiation model algorithm
    const float EPSILON = 1e-8f;

    // Calculate patch vectors and normal (same as radiation model)
    vec3 anchor = v0;
    vec3 normal = cross(v1 - v0, v2 - v0);
    normal.normalize();

    vec3 a = v1 - v0; // First edge vector
    vec3 b = v3 - v0; // Second edge vector

    // Ray-plane intersection
    float denom = direction * normal;
    if (std::abs(denom) > EPSILON) { // Not parallel to plane
        float t = (anchor - origin) * normal / denom;

        if (t > EPSILON && t < 1e8f) { // Valid intersection distance
            // Find intersection point
            vec3 p = origin + direction * t;
            vec3 d = p - anchor;

            // Project onto patch coordinate system
            float ddota = d * a;
            float ddotb = d * b;

            // Check if point is within patch bounds (with epsilon tolerance for edge cases)
            if (ddota >= -EPSILON && ddota <= (a * a) + EPSILON && ddotb >= -EPSILON && ddotb <= (b * b) + EPSILON) {
                distance = t;
                return true;
            }
        }
    }

    return false;
}

#ifdef HELIOS_CUDA_AVAILABLE
#include <vector_types.h> // For make_float3

/**
 * \brief Cast rays using warp-efficient GPU kernels
 * \param[in] ray_queries Vector of ray queries to process
 * \param[out] stats Ray-tracing performance statistics
 * \return Vector of HitResult with optimal GPU performance
 */
std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysGPU(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats) {
    std::vector<HitResult> results;
    results.resize(ray_queries.size());

    if (ray_queries.empty() || !gpu_acceleration_enabled) {
        return castRaysSoA(ray_queries, stats); // Use CPU implementation
    }

    // Ensure the BVH is built. With GPU acceleration enabled, buildBVH() also uploads the scene geometry to the device
    // (transferBVHToGPU) so it stays resident across many ray batches.
    if (bvh_nodes.empty()) {
        buildBVH();
        if (bvh_nodes.empty()) {
            helios_runtime_error("ERROR: BVH construction failed - no geometry available for ray tracing. Ensure primitives are properly added to the collision detection system.");
        }
    }

    // If the caller enabled GPU after the BVH was already built, the scene may not be resident yet; upload it now.
    if (d_bvh_nodes == nullptr || d_primitive_vertices == nullptr) {
        transferBVHToGPU();
    }
    // If the scene still could not be made resident (e.g. GPU disabled at runtime), fall back to the CPU path.
    if (!gpu_acceleration_enabled || d_bvh_nodes == nullptr || d_primitive_vertices == nullptr || primitive_indices.empty()) {
        return castRaysSoA(ray_queries, stats);
    }

    // Prepare host ray data with normalized directions (matching CPU RayQuery semantics). A non-positive max_distance
    // means "unbounded" on the CPU path; translate it to a large finite value so the GPU AABB/closest-hit logic agrees.
    const size_t num_rays = ray_queries.size();
    std::vector<float> ray_origins(num_rays * 3);
    std::vector<float> ray_directions(num_rays * 3);
    std::vector<float> ray_max_distances(num_rays);
    for (size_t i = 0; i < num_rays; i++) {
        vec3 dir = ray_queries[i].direction;
        float mag = dir.magnitude();
        if (mag > 1e-8f) {
            dir = dir / mag;
        }
        ray_origins[i * 3] = ray_queries[i].origin.x;
        ray_origins[i * 3 + 1] = ray_queries[i].origin.y;
        ray_origins[i * 3 + 2] = ray_queries[i].origin.z;
        ray_directions[i * 3] = dir.x;
        ray_directions[i * 3 + 1] = dir.y;
        ray_directions[i * 3 + 2] = dir.z;
        ray_max_distances[i] = (ray_queries[i].max_distance > 0) ? ray_queries[i].max_distance : std::numeric_limits<float>::max();
    }

    std::vector<float> hit_distances(num_rays);
    std::vector<unsigned int> hit_primitive_ids(num_rays);
    std::vector<unsigned int> hit_counts(num_rays);
    std::vector<float> hit_normals(num_rays * 3);

    // Launch against the resident scene: only the ray + result buffers are uploaded/downloaded here. The vector path
    // carries a per-ray max-distance array (RayQuery.max_distance may differ per ray), so uniform_max_distance is unused.
    launchRaysOnResidentScene(d_bvh_nodes, d_gpu_node_count, d_primitive_indices, d_gpu_primitive_count, d_primitive_types, (float3 *) d_primitive_vertices, d_vertex_offsets, (const unsigned char *) d_mask_data, d_mask_offsets, d_mask_sizes,
                              d_mask_IDs, (const float *) d_uv_data, d_uv_IDs, d_gpu_total_vertex_count, ray_origins.data(), ray_directions.data(), ray_max_distances.data(), /*uniform_max_distance=*/0.0f, static_cast<int>(num_rays),
                              hit_distances.data(), hit_primitive_ids.data(), hit_counts.data(), hit_normals.data(), true);

    size_t hit_count = 0;
    for (size_t i = 0; i < num_rays; i++) {
        const float max_d = (ray_queries[i].max_distance > 0) ? ray_queries[i].max_distance : std::numeric_limits<float>::max();
        if (hit_counts[i] > 0 && hit_distances[i] <= max_d) {
            results[i].hit = true;
            results[i].primitive_UUID = hit_primitive_ids[i];
            results[i].distance = hit_distances[i];
            results[i].intersection_point = ray_queries[i].origin + ray_queries[i].direction * hit_distances[i];
            results[i].normal = make_vec3(hit_normals[i * 3], hit_normals[i * 3 + 1], hit_normals[i * 3 + 2]);
            hit_count++;
        } else {
            results[i].hit = false;
            results[i].primitive_UUID = 0;
            results[i].distance = std::numeric_limits<float>::max();
        }
    }

    stats.total_rays_cast = num_rays;
    stats.total_hits = hit_count;

    return results;
}

// ================================================================
// SIMD-OPTIMIZED RAY TRACING METHODS
// ================================================================

uint32_t CollisionDetection::rayAABBIntersectSIMD(const vec3 *ray_origins, const vec3 *ray_directions, const vec3 *aabb_mins, const vec3 *aabb_maxs, float *t_mins, float *t_maxs, int count) {
#ifdef __AVX2__
    if (count == 8) {
        // AVX2 implementation for 8 rays at once
        uint32_t hit_mask = 0;

        for (int i = 0; i < 8; i += 8) {
            // Load 8 ray origins
            __m256 orig_x = _mm256_set_ps(ray_origins[i + 7].x, ray_origins[i + 6].x, ray_origins[i + 5].x, ray_origins[i + 4].x, ray_origins[i + 3].x, ray_origins[i + 2].x, ray_origins[i + 1].x, ray_origins[i + 0].x);
            __m256 orig_y = _mm256_set_ps(ray_origins[i + 7].y, ray_origins[i + 6].y, ray_origins[i + 5].y, ray_origins[i + 4].y, ray_origins[i + 3].y, ray_origins[i + 2].y, ray_origins[i + 1].y, ray_origins[i + 0].y);
            __m256 orig_z = _mm256_set_ps(ray_origins[i + 7].z, ray_origins[i + 6].z, ray_origins[i + 5].z, ray_origins[i + 4].z, ray_origins[i + 3].z, ray_origins[i + 2].z, ray_origins[i + 1].z, ray_origins[i + 0].z);

            // Load 8 ray directions
            __m256 dir_x = _mm256_set_ps(ray_directions[i + 7].x, ray_directions[i + 6].x, ray_directions[i + 5].x, ray_directions[i + 4].x, ray_directions[i + 3].x, ray_directions[i + 2].x, ray_directions[i + 1].x, ray_directions[i + 0].x);
            __m256 dir_y = _mm256_set_ps(ray_directions[i + 7].y, ray_directions[i + 6].y, ray_directions[i + 5].y, ray_directions[i + 4].y, ray_directions[i + 3].y, ray_directions[i + 2].y, ray_directions[i + 1].y, ray_directions[i + 0].y);
            __m256 dir_z = _mm256_set_ps(ray_directions[i + 7].z, ray_directions[i + 6].z, ray_directions[i + 5].z, ray_directions[i + 4].z, ray_directions[i + 3].z, ray_directions[i + 2].z, ray_directions[i + 1].z, ray_directions[i + 0].z);

            // Load 8 AABB mins
            __m256 aabb_min_x = _mm256_set_ps(aabb_mins[i + 7].x, aabb_mins[i + 6].x, aabb_mins[i + 5].x, aabb_mins[i + 4].x, aabb_mins[i + 3].x, aabb_mins[i + 2].x, aabb_mins[i + 1].x, aabb_mins[i + 0].x);
            __m256 aabb_min_y = _mm256_set_ps(aabb_mins[i + 7].y, aabb_mins[i + 6].y, aabb_mins[i + 5].y, aabb_mins[i + 4].y, aabb_mins[i + 3].y, aabb_mins[i + 2].y, aabb_mins[i + 1].y, aabb_mins[i + 0].y);
            __m256 aabb_min_z = _mm256_set_ps(aabb_mins[i + 7].z, aabb_mins[i + 6].z, aabb_mins[i + 5].z, aabb_mins[i + 4].z, aabb_mins[i + 3].z, aabb_mins[i + 2].z, aabb_mins[i + 1].z, aabb_mins[i + 0].z);

            // Load 8 AABB maxs
            __m256 aabb_max_x = _mm256_set_ps(aabb_maxs[i + 7].x, aabb_maxs[i + 6].x, aabb_maxs[i + 5].x, aabb_maxs[i + 4].x, aabb_maxs[i + 3].x, aabb_maxs[i + 2].x, aabb_maxs[i + 1].x, aabb_maxs[i + 0].x);
            __m256 aabb_max_y = _mm256_set_ps(aabb_maxs[i + 7].y, aabb_maxs[i + 6].y, aabb_maxs[i + 5].y, aabb_maxs[i + 4].y, aabb_maxs[i + 3].y, aabb_maxs[i + 2].y, aabb_maxs[i + 1].y, aabb_maxs[i + 0].y);
            __m256 aabb_max_z = _mm256_set_ps(aabb_maxs[i + 7].z, aabb_maxs[i + 6].z, aabb_maxs[i + 5].z, aabb_maxs[i + 4].z, aabb_maxs[i + 3].z, aabb_maxs[i + 2].z, aabb_maxs[i + 1].z, aabb_maxs[i + 0].z);

            // Calculate inverse directions
            __m256 inv_dir_x = _mm256_div_ps(_mm256_set1_ps(1.0f), dir_x);
            __m256 inv_dir_y = _mm256_div_ps(_mm256_set1_ps(1.0f), dir_y);
            __m256 inv_dir_z = _mm256_div_ps(_mm256_set1_ps(1.0f), dir_z);

            // Calculate intersection distances for X axis
            __m256 t1_x = _mm256_mul_ps(_mm256_sub_ps(aabb_min_x, orig_x), inv_dir_x);
            __m256 t2_x = _mm256_mul_ps(_mm256_sub_ps(aabb_max_x, orig_x), inv_dir_x);
            __m256 tmin_x = _mm256_min_ps(t1_x, t2_x);
            __m256 tmax_x = _mm256_max_ps(t1_x, t2_x);

            // Calculate intersection distances for Y axis
            __m256 t1_y = _mm256_mul_ps(_mm256_sub_ps(aabb_min_y, orig_y), inv_dir_y);
            __m256 t2_y = _mm256_mul_ps(_mm256_sub_ps(aabb_max_y, orig_y), inv_dir_y);
            __m256 tmin_y = _mm256_min_ps(t1_y, t2_y);
            __m256 tmax_y = _mm256_max_ps(t1_y, t2_y);

            // Calculate intersection distances for Z axis
            __m256 t1_z = _mm256_mul_ps(_mm256_sub_ps(aabb_min_z, orig_z), inv_dir_z);
            __m256 t2_z = _mm256_mul_ps(_mm256_sub_ps(aabb_max_z, orig_z), inv_dir_z);
            __m256 tmin_z = _mm256_min_ps(t1_z, t2_z);
            __m256 tmax_z = _mm256_max_ps(t1_z, t2_z);

            // Find intersection interval
            __m256 t_min_final = _mm256_max_ps(_mm256_max_ps(tmin_x, tmin_y), tmin_z);
            __m256 t_max_final = _mm256_min_ps(_mm256_min_ps(tmax_x, tmax_y), tmax_z);

            // Store results
            _mm256_store_ps(&t_mins[i], t_min_final);
            _mm256_store_ps(&t_maxs[i], t_max_final);

            // Check for intersection: t_max >= 0 && t_min <= t_max
            __m256 zero = _mm256_set1_ps(0.0f);
            __m256 hits = _mm256_and_ps(_mm256_cmp_ps(t_max_final, zero, _CMP_GE_OS), _mm256_cmp_ps(t_min_final, t_max_final, _CMP_LE_OS));

            // Convert to bitmask
            hit_mask |= _mm256_movemask_ps(hits);
        }

        return hit_mask;
    }
#endif

#ifdef __SSE4_1__
    if (count == 4) {
        // SSE implementation for 4 rays at once
        uint32_t hit_mask = 0;

        for (int i = 0; i < 4; i += 4) {
            // Load 4 ray origins
            __m128 orig_x = _mm_set_ps(ray_origins[i + 3].x, ray_origins[i + 2].x, ray_origins[i + 1].x, ray_origins[i + 0].x);
            __m128 orig_y = _mm_set_ps(ray_origins[i + 3].y, ray_origins[i + 2].y, ray_origins[i + 1].y, ray_origins[i + 0].y);
            __m128 orig_z = _mm_set_ps(ray_origins[i + 3].z, ray_origins[i + 2].z, ray_origins[i + 1].z, ray_origins[i + 0].z);

            // Load 4 ray directions
            __m128 dir_x = _mm_set_ps(ray_directions[i + 3].x, ray_directions[i + 2].x, ray_directions[i + 1].x, ray_directions[i + 0].x);
            __m128 dir_y = _mm_set_ps(ray_directions[i + 3].y, ray_directions[i + 2].y, ray_directions[i + 1].y, ray_directions[i + 0].y);
            __m128 dir_z = _mm_set_ps(ray_directions[i + 3].z, ray_directions[i + 2].z, ray_directions[i + 1].z, ray_directions[i + 0].z);

            // Load 4 AABB mins
            __m128 aabb_min_x = _mm_set_ps(aabb_mins[i + 3].x, aabb_mins[i + 2].x, aabb_mins[i + 1].x, aabb_mins[i + 0].x);
            __m128 aabb_min_y = _mm_set_ps(aabb_mins[i + 3].y, aabb_mins[i + 2].y, aabb_mins[i + 1].y, aabb_mins[i + 0].y);
            __m128 aabb_min_z = _mm_set_ps(aabb_mins[i + 3].z, aabb_mins[i + 2].z, aabb_mins[i + 1].z, aabb_mins[i + 0].z);

            // Load 4 AABB maxs
            __m128 aabb_max_x = _mm_set_ps(aabb_maxs[i + 3].x, aabb_maxs[i + 2].x, aabb_maxs[i + 1].x, aabb_maxs[i + 0].x);
            __m128 aabb_max_y = _mm_set_ps(aabb_maxs[i + 3].y, aabb_maxs[i + 2].y, aabb_maxs[i + 1].y, aabb_maxs[i + 0].y);
            __m128 aabb_max_z = _mm_set_ps(aabb_maxs[i + 3].z, aabb_maxs[i + 2].z, aabb_maxs[i + 1].z, aabb_maxs[i + 0].z);

            // Calculate inverse directions
            __m128 inv_dir_x = _mm_div_ps(_mm_set1_ps(1.0f), dir_x);
            __m128 inv_dir_y = _mm_div_ps(_mm_set1_ps(1.0f), dir_y);
            __m128 inv_dir_z = _mm_div_ps(_mm_set1_ps(1.0f), dir_z);

            // Calculate intersection distances for X axis
            __m128 t1_x = _mm_mul_ps(_mm_sub_ps(aabb_min_x, orig_x), inv_dir_x);
            __m128 t2_x = _mm_mul_ps(_mm_sub_ps(aabb_max_x, orig_x), inv_dir_x);
            __m128 tmin_x = _mm_min_ps(t1_x, t2_x);
            __m128 tmax_x = _mm_max_ps(t1_x, t2_x);

            // Calculate intersection distances for Y axis
            __m128 t1_y = _mm_mul_ps(_mm_sub_ps(aabb_min_y, orig_y), inv_dir_y);
            __m128 t2_y = _mm_mul_ps(_mm_sub_ps(aabb_max_y, orig_y), inv_dir_y);
            __m128 tmin_y = _mm_min_ps(t1_y, t2_y);
            __m128 tmax_y = _mm_max_ps(t1_y, t2_y);

            // Calculate intersection distances for Z axis
            __m128 t1_z = _mm_mul_ps(_mm_sub_ps(aabb_min_z, orig_z), inv_dir_z);
            __m128 t2_z = _mm_mul_ps(_mm_sub_ps(aabb_max_z, orig_z), inv_dir_z);
            __m128 tmin_z = _mm_min_ps(t1_z, t2_z);
            __m128 tmax_z = _mm_max_ps(t1_z, t2_z);

            // Find intersection interval
            __m128 t_min_final = _mm_max_ps(_mm_max_ps(tmin_x, tmin_y), tmin_z);
            __m128 t_max_final = _mm_min_ps(_mm_min_ps(tmax_x, tmax_y), tmax_z);

            // Store results
            _mm_store_ps(&t_mins[i], t_min_final);
            _mm_store_ps(&t_maxs[i], t_max_final);

            // Check for intersection: t_max >= 0 && t_min <= t_max
            __m128 zero = _mm_set1_ps(0.0f);
            __m128 hits = _mm_and_ps(_mm_cmpge_ps(t_max_final, zero), _mm_cmple_ps(t_min_final, t_max_final));

            // Convert to bitmask
            hit_mask |= _mm_movemask_ps(hits);
        }

        return hit_mask;
    }
#endif

    // Fallback to scalar implementation
    uint32_t hit_mask = 0;
    for (int i = 0; i < count; ++i) {
        if (rayAABBIntersect(ray_origins[i], ray_directions[i], aabb_mins[i], aabb_maxs[i], t_mins[i], t_maxs[i])) {
            hit_mask |= (1 << i);
        }
    }
    return hit_mask;
}

void CollisionDetection::traverseBVHSIMD(const vec3 *ray_origins, const vec3 *ray_directions, int count, HitResult *results) {
    if (bvh_nodes.empty()) {
        // Initialize all results as misses
        for (int i = 0; i < count; ++i) {
            results[i] = HitResult();
        }
        return;
    }

    // Determine SIMD batch size based on available instructions
    int simd_batch_size = 1; // Default to scalar
#ifdef __AVX2__
    simd_batch_size = 8;
#elif defined(__SSE4_1__)
    simd_batch_size = 4;
#endif

    // Process rays in SIMD batches
    for (int batch_start = 0; batch_start < count; batch_start += simd_batch_size) {
        int batch_count = std::min(simd_batch_size, count - batch_start);

        // Initialize batch results
        for (int i = 0; i < batch_count; ++i) {
            results[batch_start + i] = HitResult();
        }

        if (batch_count >= simd_batch_size && simd_batch_size > 1) {
            // SIMD-optimized batch processing
            traverseBVHSIMDImpl(&ray_origins[batch_start], &ray_directions[batch_start], batch_count, &results[batch_start]);
        } else {
            // Process remaining rays individually
            for (int i = 0; i < batch_count; ++i) {
                int ray_idx = batch_start + i;
                results[ray_idx] = castRay(RayQuery(ray_origins[ray_idx], ray_directions[ray_idx]));
            }
        }
    }
}

void CollisionDetection::traverseBVHSIMDImpl(const vec3 *ray_origins, const vec3 *ray_directions, int count, HitResult *results) {
    const size_t MAX_STACK_SIZE = 64;

    // Per-ray data structures - using aligned arrays for SIMD efficiency
    alignas(32) uint32_t node_stacks[8][MAX_STACK_SIZE]; // 8 stacks for up to 8 rays
    alignas(32) uint32_t stack_tops[8] = {0}; // Current stack positions
    alignas(32) float closest_distances[8]; // Closest hit distances per ray
    alignas(32) bool ray_active[8]; // Which rays are still active

    // Initialize per-ray state
    for (int i = 0; i < count; ++i) {
        node_stacks[i][0] = 0; // Start with root node
        stack_tops[i] = 1;
        closest_distances[i] = std::numeric_limits<float>::max();
        ray_active[i] = true;
        results[i] = HitResult(); // Initialize as miss
    }

    // Main traversal loop - continue while any ray is active
    while (true) {
        bool any_active = false;
        for (int i = 0; i < count; ++i) {
            if (ray_active[i] && stack_tops[i] > 0) {
                any_active = true;
                break;
            }
        }
        if (!any_active)
            break;

        // Collect next nodes to test for active rays
        alignas(32) vec3 test_aabb_mins[8];
        alignas(32) vec3 test_aabb_maxs[8];
        alignas(32) uint32_t test_node_indices[8];
        alignas(32) int test_ray_indices[8];
        int test_count = 0;

        for (int i = 0; i < count; ++i) {
            if (ray_active[i] && stack_tops[i] > 0) {
                uint32_t node_idx = node_stacks[i][--stack_tops[i]];
                const BVHNode &node = bvh_nodes[node_idx];

                test_aabb_mins[test_count] = node.aabb_min;
                test_aabb_maxs[test_count] = node.aabb_max;
                test_node_indices[test_count] = node_idx;
                test_ray_indices[test_count] = i;
                test_count++;

                if (test_count == count)
                    break; // Batch is full
            }
        }

        if (test_count == 0)
            break;

        // Prepare ray data for SIMD intersection test
        alignas(32) vec3 batch_origins[8];
        alignas(32) vec3 batch_directions[8];
        alignas(32) float t_mins[8];
        alignas(32) float t_maxs[8];

        for (int i = 0; i < test_count; ++i) {
            int ray_idx = test_ray_indices[i];
            batch_origins[i] = ray_origins[ray_idx];
            batch_directions[i] = ray_directions[ray_idx];
        }

        // Perform SIMD AABB intersection test
        uint32_t hit_mask = rayAABBIntersectSIMD(batch_origins, batch_directions, test_aabb_mins, test_aabb_maxs, t_mins, t_maxs, test_count);

        // Process intersection results
        for (int i = 0; i < test_count; ++i) {
            if (!(hit_mask & (1 << i)))
                continue; // Ray missed this AABB

            int ray_idx = test_ray_indices[i];
            uint32_t node_idx = test_node_indices[i];
            const BVHNode &node = bvh_nodes[node_idx];

            if (t_mins[i] > closest_distances[ray_idx])
                continue; // Beyond closest hit

            if (node.is_leaf) {
                // Test primitives in leaf node
                for (uint32_t prim_idx = node.primitive_start; prim_idx < node.primitive_start + node.primitive_count; ++prim_idx) {

                    uint32_t primitive_id = primitive_indices[prim_idx];

                    // Use thread-safe primitive intersection
                    HitResult prim_result = intersectPrimitiveThreadSafe(batch_origins[i], batch_directions[i], primitive_id, closest_distances[ray_idx]);
                    if (prim_result.hit && prim_result.distance < closest_distances[ray_idx]) {
                        closest_distances[ray_idx] = prim_result.distance;
                        results[ray_idx] = prim_result;
                    }
                }
            } else {
                // Add child nodes to stack (if space available)
                if (stack_tops[ray_idx] < MAX_STACK_SIZE - 2) {
                    node_stacks[ray_idx][stack_tops[ray_idx]++] = node.left_child;
                    node_stacks[ray_idx][stack_tops[ray_idx]++] = node.right_child;
                } else {
                    // Stack overflow - mark ray as inactive to prevent infinite loop
                    ray_active[ray_idx] = false;
                }
            }
        }
    }
}
#endif


bool CollisionDetection::rayAABBIntersectPrimitive(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max, float &distance) {
    // Ray-AABB intersection using slab method
    // Optimized version with early termination

    const float EPSILON = 1e-8f;

    // Calculate t values for each slab
    float t_min_x = (aabb_min.x - origin.x) / direction.x;
    float t_max_x = (aabb_max.x - origin.x) / direction.x;

    // Handle negative direction components
    if (direction.x < 0.0f) {
        float temp = t_min_x;
        t_min_x = t_max_x;
        t_max_x = temp;
    }

    float t_min_y = (aabb_min.y - origin.y) / direction.y;
    float t_max_y = (aabb_max.y - origin.y) / direction.y;

    if (direction.y < 0.0f) {
        float temp = t_min_y;
        t_min_y = t_max_y;
        t_max_y = temp;
    }

    // Check for early termination in X-Y
    float t_min = std::max(t_min_x, t_min_y);
    float t_max = std::min(t_max_x, t_max_y);

    if (t_min > t_max) {
        return false; // No intersection
    }

    float t_min_z = (aabb_min.z - origin.z) / direction.z;
    float t_max_z = (aabb_max.z - origin.z) / direction.z;

    if (direction.z < 0.0f) {
        float temp = t_min_z;
        t_min_z = t_max_z;
        t_max_z = temp;
    }

    // Final intersection test
    t_min = std::max(t_min, t_min_z);
    t_max = std::min(t_max, t_max_z);

    if (t_min > t_max || t_max < EPSILON) {
        return false; // No intersection or behind ray
    }

    // Set distance to closest intersection point
    distance = (t_min > EPSILON) ? t_min : t_max;

    return distance > EPSILON;
}
