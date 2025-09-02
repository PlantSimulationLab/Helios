/** \file "CollisionDetection_RayTracing.cpp" Ray tracing implementation for collision detection plugin

    Copyright (C) 2016-2025 Brian Bailey

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
void launchVoxelRayPathLengths(int num_rays, float *h_ray_origins, float *h_ray_directions, float grid_center_x, float grid_center_y, float grid_center_z, float grid_size_x, float grid_size_y, float grid_size_z, int grid_divisions_x,
                               int grid_divisions_y, int grid_divisions_z, int primitive_count, int *h_voxel_ray_counts, float *h_voxel_path_lengths, int *h_voxel_transmitted, int *h_voxel_hit_before, int *h_voxel_hit_after, int *h_voxel_hit_inside);
// Warp-efficient GPU kernels
void launchWarpEfficientBVH(void *h_bvh_soa_gpu, unsigned int *h_primitive_indices, int primitive_count, float *h_primitive_aabb_min, float *h_primitive_aabb_max, float *h_ray_origins, float *h_ray_directions, float *h_ray_max_distances,
                            int num_rays, unsigned int *h_results, unsigned int *h_result_counts, int max_results_per_ray);
// High-performance ray-triangle intersection kernel
void launchRayPrimitiveIntersection(void *h_bvh_nodes, int node_count, unsigned int *h_primitive_indices, int primitive_count, int *h_primitive_types, float3 *h_primitive_vertices, unsigned int *h_vertex_offsets, int total_vertex_count,
                                    float *h_ray_origins, float *h_ray_directions, float *h_ray_max_distances, int num_rays, float *h_hit_distances, unsigned int *h_hit_primitive_ids, unsigned int *h_hit_counts, bool find_closest_hit);
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
        // Build primitive cache for thread-safe access if not already built
        if (primitive_cache.empty()) {
            buildPrimitiveCache();
        }

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


std::vector<CollisionDetection::HitResult> CollisionDetection::castRays(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats) {
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());

    // Initialize statistics
    RayTracingStats local_stats;
    local_stats.total_rays_cast = ray_queries.size();

    // Smart CPU/GPU selection based on ray count and scene complexity
    // Performance analysis shows CPU is faster for small batches (<1000 rays) due to
    // GPU launch overhead, while GPU provides significant speedup for large batches
    // TEMPORARY: Force CPU for LiDAR until GPU normal calculation is implemented
    const size_t GPU_BATCH_THRESHOLD = 1000000; // Use GPU for batches larger than this (effectively disabled)
    const size_t MIN_PRIMITIVES_FOR_GPU = 500; // Keep CPU path for stability

#ifdef HELIOS_CUDA_AVAILABLE
    bool use_gpu = gpu_acceleration_enabled && ray_queries.size() >= GPU_BATCH_THRESHOLD && d_bvh_nodes != nullptr && !primitive_indices.empty() && primitive_indices.size() >= MIN_PRIMITIVES_FOR_GPU;

    if (use_gpu) {
        // Use GPU acceleration for large batches with sufficient scene complexity
        castRaysGPU(ray_queries, results, local_stats);
    } else {
        // Use CPU for small batches or simple scenes
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

    // Build primitive cache for high-performance thread-safe primitive intersection
    if (primitive_cache.empty()) {
        buildPrimitiveCache();
    }

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

// OpenMP parallel ray processing for high-performance SoA traversal
#pragma omp parallel
    {
        RayTracingStats local_stats = {}; // Thread-local statistics

#pragma omp for schedule(guided, 32)
        for (int i = 0; i < static_cast<int>(ray_queries.size()); ++i) {
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

    // Stack-based traversal (more cache-friendly than recursion)
    std::stack<size_t> node_stack;
    node_stack.push(0); // Start from root

    float closest_distance = (query.max_distance > 0) ? query.max_distance : std::numeric_limits<float>::max();

    // Safety limit to prevent infinite loops
    const size_t MAX_TRAVERSAL_STEPS = 10000000; // 10M steps should be more than enough
    size_t traversal_steps = 0;

    while (!node_stack.empty()) {
        // Safety check to prevent infinite loops
        if (++traversal_steps > MAX_TRAVERSAL_STEPS) {
            if (printmessages) {
                std::cout << "WARNING: BVH traversal exceeded maximum steps, terminating ray" << std::endl;
            }
            break;
        }

        size_t node_idx = node_stack.top();
        node_stack.pop();
        stats.bvh_nodes_visited++;

        // Bounds check for node index
        if (node_idx >= bvh_nodes_soa.node_count) {
            if (printmessages) {
                std::cout << "WARNING: Invalid node index " << node_idx << " >= " << bvh_nodes_soa.node_count << std::endl;
            }
            continue;
        }

        // Prefetch data for better cache performance
        if (node_idx + 1 < bvh_nodes_soa.node_count) {
#ifdef __GNUC__
            __builtin_prefetch(&bvh_nodes_soa.aabb_mins[node_idx + 1], 0, 1);
            __builtin_prefetch(&bvh_nodes_soa.aabb_maxs[node_idx + 1], 0, 1);
#elif defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char *>(&bvh_nodes_soa.aabb_mins[node_idx + 1]), _MM_HINT_T1);
            _mm_prefetch(reinterpret_cast<const char *>(&bvh_nodes_soa.aabb_maxs[node_idx + 1]), _MM_HINT_T1);
#endif
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
                uint primitive_id = primitive_indices[primitive_start + i];

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

                // Use high-performance cached primitive intersection
                HitResult primitive_result = intersectPrimitiveThreadSafe(query.origin, query.direction, primitive_id, closest_distance);
                if (primitive_result.hit && primitive_result.distance < closest_distance) {
                    result = primitive_result;
                    closest_distance = primitive_result.distance;
                }
            }
        } else {
            // Internal node - add children to stack
            uint32_t left_child = bvh_nodes_soa.left_children[node_idx];
            uint32_t right_child = bvh_nodes_soa.right_children[node_idx];

            if (left_child != 0xFFFFFFFF && left_child < bvh_nodes_soa.node_count) {
                node_stack.push(left_child);
            }
            if (right_child != 0xFFFFFFFF && right_child < bvh_nodes_soa.node_count) {
                node_stack.push(right_child);
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

    // Compute inverse ray direction
    __m128 inv_dir = _mm_div_ps(_mm_set1_ps(1.0f), ray_dir);

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
    // Fallback scalar implementation
    float inv_dir_x = 1.0f / ray_direction.x;
    float inv_dir_y = 1.0f / ray_direction.y;
    float inv_dir_z = 1.0f / ray_direction.z;

    float t1_x = (aabb_min.x - ray_origin.x) * inv_dir_x;
    float t2_x = (aabb_max.x - ray_origin.x) * inv_dir_x;
    float t1_y = (aabb_min.y - ray_origin.y) * inv_dir_y;
    float t2_y = (aabb_max.y - ray_origin.y) * inv_dir_y;
    float t1_z = (aabb_min.z - ray_origin.z) * inv_dir_z;
    float t2_z = (aabb_max.z - ray_origin.z) * inv_dir_z;

    float tmin_x = std::min(t1_x, t2_x);
    float tmax_x = std::max(t1_x, t2_x);
    float tmin_y = std::min(t1_y, t2_y);
    float tmax_y = std::max(t1_y, t2_y);
    float tmin_z = std::min(t1_z, t2_z);
    float tmax_z = std::max(t1_z, t2_z);

    float t_near = std::max({tmin_x, tmin_y, tmin_z, 0.0f});
    float t_far = std::min({tmax_x, tmax_y, tmax_z, max_distance});

    return t_near <= t_far;
#endif
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
        result.hit = true;
        result.distance = distance;
        result.primitive_UUID = primitive_id;
        result.intersection_point = query.origin + query.direction * distance;

        // Calculate surface normal directly from context (minimal overhead)
        PrimitiveType type = context->getPrimitiveType(primitive_id);
        std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_id);

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
            result.hit = true;
            result.distance = distance;
            result.primitive_UUID = primitive_id;
            result.intersection_point = origin + direction * distance;

            // Calculate surface normal for uncached primitive
            PrimitiveType type = context->getPrimitiveType(primitive_id);
            std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_id);

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

    const CachedPrimitive &cached = it->second;

    // Perform intersection test based on primitive type
    if (cached.type == PRIMITIVE_TYPE_TRIANGLE && cached.vertices.size() >= 3) {
        float distance;
        if (triangleIntersect(origin, direction, cached.vertices[0], cached.vertices[1], cached.vertices[2], distance)) {
            if (distance > 1e-6f && (max_distance <= 0 || distance < max_distance)) {
                result.hit = true;
                result.distance = distance;
                result.primitive_UUID = primitive_id;
                result.intersection_point = origin + direction * distance;

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
                result.hit = true;
                result.distance = distance;
                result.primitive_UUID = primitive_id;
                result.intersection_point = origin + direction * distance;

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
                result.primitive_UUID = primitive_id;
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

                primitive_cache[primitive_id] = CachedPrimitive(type, vertices);
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

    if (printmessages) {
    }
}

bool CollisionDetection::triangleIntersect(const vec3 &origin, const vec3 &direction, const vec3 &v0, const vec3 &v1, const vec3 &v2, float &distance) {
    // Möller-Trumbore triangle intersection algorithm (optimized - no vec3 temporaries)
    const float EPSILON = 1e-8f;

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

bool CollisionDetection::patchIntersect(const vec3 &origin, const vec3 &direction, const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &v3, float &distance) {
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

    auto start = std::chrono::high_resolution_clock::now();

    // Ensure BVH is built for GPU acceleration
    if (bvh_nodes.empty()) {
        if (printmessages) {
        }
        buildBVH();
        if (bvh_nodes.empty()) {
            helios_runtime_error("ERROR: BVH construction failed - no geometry available for ray tracing. Ensure primitives are properly added to the collision detection system.");
        }
    }

    // Prepare ray data for GPU
    std::vector<float> ray_origins(ray_queries.size() * 3);
    std::vector<float> ray_directions(ray_queries.size() * 3);
    std::vector<float> ray_max_distances(ray_queries.size());

    for (size_t i = 0; i < ray_queries.size(); i++) {
        vec3 normalized_dir = ray_queries[i].direction;
        if (normalized_dir.magnitude() > 1e-8f) {
            normalized_dir = normalized_dir / normalized_dir.magnitude();
        }

        ray_origins[i * 3] = ray_queries[i].origin.x;
        ray_origins[i * 3 + 1] = ray_queries[i].origin.y;
        ray_origins[i * 3 + 2] = ray_queries[i].origin.z;

        ray_directions[i * 3] = normalized_dir.x;
        ray_directions[i * 3 + 1] = normalized_dir.y;
        ray_directions[i * 3 + 2] = normalized_dir.z;

        ray_max_distances[i] = ray_queries[i].max_distance;
    }

    // Get all primitives for GPU ray tracing (handle all primitive types like CPU)
    std::vector<uint> all_primitives = context->getAllUUIDs();
    std::vector<unsigned int> primitive_indices(all_primitives.size());
    std::vector<int> primitive_types(all_primitives.size());
    std::vector<float3> primitive_vertices;
    std::vector<unsigned int> vertex_offsets(all_primitives.size());

    size_t triangle_count = 0;
    size_t patch_count = 0;
    size_t voxel_count = 0;

    if (printmessages) {
    }

    // Prepare primitive data arrays for GPU
    size_t vertex_index = 0;
    for (size_t i = 0; i < all_primitives.size(); i++) {
        primitive_indices[i] = all_primitives[i];
        vertex_offsets[i] = vertex_index;

        PrimitiveType ptype = context->getPrimitiveType(all_primitives[i]);
        primitive_types[i] = static_cast<int>(ptype);

        std::vector<vec3> vertices = context->getPrimitiveVertices(all_primitives[i]);

        if (ptype == PRIMITIVE_TYPE_TRIANGLE) {
            triangle_count++;
            if (vertices.size() >= 3) {
                // Add 3 vertices for triangle, pad to 4 for alignment
                for (int v = 0; v < 3; v++) {
                    primitive_vertices.push_back(make_float3(vertices[v].x, vertices[v].y, vertices[v].z));
                }
                primitive_vertices.push_back(make_float3(0, 0, 0)); // Padding
                vertex_index += 4;
            } else {
                if (printmessages) {
                    std::cout << "WARNING: Triangle primitive " << all_primitives[i] << " has " << vertices.size() << " vertices" << std::endl;
                }
                // Add zeros for invalid triangle
                for (int v = 0; v < 4; v++) {
                    primitive_vertices.push_back(make_float3(0, 0, 0));
                }
                vertex_index += 4;
            }
        } else if (ptype == PRIMITIVE_TYPE_PATCH) {
            patch_count++;
            if (vertices.size() >= 4) {
                // Add 4 vertices for patch
                for (int v = 0; v < 4; v++) {
                    primitive_vertices.push_back(make_float3(vertices[v].x, vertices[v].y, vertices[v].z));
                }
                vertex_index += 4;
            } else {
                if (printmessages) {
                    std::cout << "WARNING: Patch primitive " << all_primitives[i] << " has " << vertices.size() << " vertices" << std::endl;
                }
                // Add zeros for invalid patch
                for (int v = 0; v < 4; v++) {
                    primitive_vertices.push_back(make_float3(0, 0, 0));
                }
                vertex_index += 4;
            }
        } else if (ptype == PRIMITIVE_TYPE_VOXEL) {
            voxel_count++;
            // For voxels, compute AABB from 8 corner vertices
            // vertices was already retrieved above, reuse it
            vec3 voxel_min = vertices[0];
            vec3 voxel_max = vertices[0];

            // Find min/max coordinates from all 8 vertices
            for (const auto &vertex: vertices) {
                voxel_min.x = std::min(voxel_min.x, vertex.x);
                voxel_min.y = std::min(voxel_min.y, vertex.y);
                voxel_min.z = std::min(voxel_min.z, vertex.z);
                voxel_max.x = std::max(voxel_max.x, vertex.x);
                voxel_max.y = std::max(voxel_max.y, vertex.y);
                voxel_max.z = std::max(voxel_max.z, vertex.z);
            }

            // Store as: [min.x, min.y, min.z, max.x, max.y, max.z, 0, 0]
            primitive_vertices.push_back(make_float3(voxel_min.x, voxel_min.y, voxel_min.z)); // v0 = min
            primitive_vertices.push_back(make_float3(voxel_max.x, voxel_max.y, voxel_max.z)); // v1 = max
            primitive_vertices.push_back(make_float3(0, 0, 0)); // v2 = padding
            primitive_vertices.push_back(make_float3(0, 0, 0)); // v3 = padding
            vertex_index += 4;
        } else {
            // Unknown primitive type - add padding
            for (int v = 0; v < 4; v++) {
                primitive_vertices.push_back(make_float3(0, 0, 0));
            }
            vertex_index += 4;
        }
    }

    if (printmessages) {
    }

    if (all_primitives.empty()) {
        if (printmessages) {
            std::cout << "No primitives found for GPU ray tracing, falling back to CPU" << std::endl;
        }
        return castRaysSoA(ray_queries, stats);
    }


    // Prepare result arrays
    std::vector<float> hit_distances(ray_queries.size());
    std::vector<unsigned int> hit_primitive_ids(ray_queries.size());
    std::vector<unsigned int> hit_counts(ray_queries.size());

    // Convert CPU BVH nodes to GPU format
    std::vector<GPUBVHNode> gpu_bvh_nodes(bvh_nodes.size());
    for (size_t i = 0; i < bvh_nodes.size(); i++) {
        gpu_bvh_nodes[i].aabb_min = make_float3(bvh_nodes[i].aabb_min.x, bvh_nodes[i].aabb_min.y, bvh_nodes[i].aabb_min.z);
        gpu_bvh_nodes[i].aabb_max = make_float3(bvh_nodes[i].aabb_max.x, bvh_nodes[i].aabb_max.y, bvh_nodes[i].aabb_max.z);
        gpu_bvh_nodes[i].left_child = bvh_nodes[i].left_child;
        gpu_bvh_nodes[i].right_child = bvh_nodes[i].right_child;
        gpu_bvh_nodes[i].primitive_start = bvh_nodes[i].primitive_start;
        gpu_bvh_nodes[i].primitive_count = bvh_nodes[i].primitive_count;
        gpu_bvh_nodes[i].is_leaf = bvh_nodes[i].is_leaf ? 1 : 0;
        gpu_bvh_nodes[i].padding = 0;

        // Debug problematic BVH nodes
        if (bvh_nodes[i].is_leaf && printmessages && i < 3) {
            std::cout << "BVH leaf node " << i << ": primitive_start=" << bvh_nodes[i].primitive_start << ", primitive_count=" << bvh_nodes[i].primitive_count << ", max_index=" << (bvh_nodes[i].primitive_start + bvh_nodes[i].primitive_count - 1)
                      << std::endl;
        }
    }

    // Launch high-performance GPU ray intersection kernel
    launchRayPrimitiveIntersection(gpu_bvh_nodes.data(), // BVH nodes
                                   static_cast<int>(gpu_bvh_nodes.size()), // Node count
                                   primitive_indices.data(), // Primitive indices
                                   static_cast<int>(all_primitives.size()), // Primitive count
                                   primitive_types.data(), // Primitive types
                                   primitive_vertices.data(), // Primitive vertices (all types)
                                   vertex_offsets.data(), // Vertex offsets
                                   static_cast<int>(primitive_vertices.size()),
                                   ray_origins.data(), // Ray origins
                                   ray_directions.data(), // Ray directions
                                   ray_max_distances.data(), // Ray max distances
                                   static_cast<int>(ray_queries.size()), // Number of rays
                                   hit_distances.data(), // Output: hit distances
                                   hit_primitive_ids.data(), // Output: hit primitive IDs
                                   hit_counts.data(), // Output: hit counts
                                   true // Find closest hit
    );

    // Convert GPU results back to HitResult format
    size_t hit_count = 0;
    for (size_t i = 0; i < ray_queries.size(); i++) {
        if (hit_counts[i] > 0 && hit_distances[i] <= ray_queries[i].max_distance) {
            results[i].hit = true;
            results[i].primitive_UUID = hit_primitive_ids[i];
            results[i].distance = hit_distances[i];
            results[i].intersection_point = ray_queries[i].origin + ray_queries[i].direction * hit_distances[i];
            hit_count++;
        } else {
            results[i].hit = false;
            results[i].primitive_UUID = 0;
            results[i].distance = std::numeric_limits<float>::max();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    stats.total_rays_cast = ray_queries.size();
    stats.total_hits = hit_count;
    double rays_per_second = ray_queries.size() / (elapsed / 1000.0);


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
