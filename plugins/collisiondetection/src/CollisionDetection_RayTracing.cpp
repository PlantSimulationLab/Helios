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

#include "CollisionDetection.h"
#include <limits>
#include <functional>
#include <queue>
#include <stack>
#include <chrono>
#include <thread>

// SIMD headers
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
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
void launchVoxelRayPathLengths(int num_rays, float *h_ray_origins, float *h_ray_directions, float grid_center_x, float grid_center_y, float grid_center_z, float grid_size_x, float grid_size_y, float grid_size_z, int grid_divisions_x, int grid_divisions_y, int grid_divisions_z, int *h_voxel_ray_counts, float *h_voxel_path_lengths, int *h_voxel_transmitted);
// Warp-efficient GPU kernels
void launchWarpEfficientBVH(void *h_bvh_soa_gpu, unsigned int *h_primitive_indices, int primitive_count, float *h_primitive_aabb_min, float *h_primitive_aabb_max, float *h_ray_origins, float *h_ray_directions, float *h_ray_max_distances, int num_rays, unsigned int *h_results, unsigned int *h_result_counts, int max_results_per_ray);
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

    // Set up search parameters
    std::vector<uint> search_primitives;
    if (target_UUIDs.empty()) {
        // Use cached primitives if available (thread-safe), otherwise use context (thread-unsafe)
        if (!primitive_cache.empty()) {
            // Use cached primitive IDs for thread-safe operation
            search_primitives.reserve(primitive_cache.size());
            for (const auto& cached_pair : primitive_cache) {
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
    for (uint candidate_uuid : search_primitives) {
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
        } catch (const std::exception& e) {
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

    const size_t GPU_BATCH_THRESHOLD = 1000; // Use GPU for batches larger than this

#ifdef HELIOS_CUDA_AVAILABLE
    if (ray_queries.size() >= GPU_BATCH_THRESHOLD && d_bvh_nodes != nullptr && !primitive_indices.empty()) {
        // Use GPU acceleration for large batches
        castRaysGPU(ray_queries, results, local_stats);
    } else {
        // Use CPU for small batches
        castRaysCPU(ray_queries, results, local_stats);
    }
#else
    // CPU-only fallback
    castRaysCPU(ray_queries, results, local_stats);
#endif

    // Copy statistics to output parameter if provided
    if (stats != nullptr) {
        *stats = local_stats;
    }

    return results;
}

void CollisionDetection::castRaysCPU(const std::vector<RayQuery> &ray_queries, std::vector<HitResult> &results, RayTracingStats &stats) {
    const size_t num_rays = ray_queries.size();

#ifdef _OPENMP
    // Use OpenMP for parallel processing with thread-safe primitive cache
    const size_t OPENMP_THRESHOLD = 100; // Use OpenMP for batches larger than this

    if (num_rays >= OPENMP_THRESHOLD) {
        // Build primitive cache for thread-safe access BEFORE parallel region
        buildPrimitiveCache();

        // Pre-allocate results vector for parallel access
        results.resize(num_rays);

        // Thread-local statistics for reduction
        size_t total_hits = 0;
        double total_distance = 0.0;

#pragma omp parallel for reduction(+:total_hits,total_distance) schedule(dynamic, 16)
        for (size_t i = 0; i < num_rays; i++) {
            HitResult result = castRay(ray_queries[i]);
            results[i] = result;

            // Update thread-local statistics
            if (result.hit) {
                total_hits++;
                total_distance += result.distance;
            }
        }

        // Update global statistics
        stats.total_hits = total_hits;
        stats.average_ray_distance = (total_hits > 0) ? (total_distance / total_hits) : 0.0;
        return;
    }
#endif

    // Fallback to sequential processing - use original push_back approach
    for (const auto &query : ray_queries) {
        HitResult result = castRay(query);
        results.push_back(result);

        // Update statistics
        if (result.hit) {
            stats.total_hits++;
            stats.average_ray_distance += result.distance;
        }
    }

    // Finalize statistics
    if (stats.total_hits > 0) {
        stats.average_ray_distance /= stats.total_hits;
    }
}

#ifdef HELIOS_CUDA_AVAILABLE
void CollisionDetection::castRaysGPU(const std::vector<RayQuery> &ray_queries, std::vector<HitResult> &results, RayTracingStats &stats) {
    const size_t num_rays = ray_queries.size();

    // For very large batches, use parallel CPU processing to avoid GPU memory issues
    // This provides better performance than sequential processing while avoiding CUDA complexity
    const size_t PARALLEL_BATCH_SIZE = 1000;

    if (num_rays >= PARALLEL_BATCH_SIZE) {
        // Process rays in parallel batches using CPU threading
        const size_t num_threads = std::min(size_t(8), num_rays / 100); // Up to 8 threads
        const size_t rays_per_thread = num_rays / num_threads;

        std::vector<std::thread> threads;
        std::vector<std::vector<HitResult>> thread_results(num_threads);
        std::vector<RayTracingStats> thread_stats(num_threads);

        for (size_t t = 0; t < num_threads; t++) {
            size_t start_idx = t * rays_per_thread;
            size_t end_idx = (t == num_threads - 1) ? num_rays : (t + 1) * rays_per_thread;

            threads.emplace_back([this, &ray_queries, &thread_results, &thread_stats, t, start_idx, end_idx]() {
                for (size_t i = start_idx; i < end_idx; i++) {
                    HitResult hit_result = castRay(ray_queries[i]);
                    thread_results[t].push_back(hit_result);

                    if (hit_result.hit) {
                        thread_stats[t].total_hits++;
                        thread_stats[t].average_ray_distance += hit_result.distance;
                    }
                }

                if (thread_stats[t].total_hits > 0) {
                    thread_stats[t].average_ray_distance /= thread_stats[t].total_hits;
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Combine results
        for (size_t t = 0; t < num_threads; t++) {
            results.insert(results.end(), thread_results[t].begin(), thread_results[t].end());
            stats.total_hits += thread_stats[t].total_hits;
            if (thread_stats[t].total_hits > 0) {
                stats.average_ray_distance += thread_stats[t].average_ray_distance * thread_stats[t].total_hits;
            }
        }

        if (stats.total_hits > 0) {
            stats.average_ray_distance /= stats.total_hits;
        }
    } else {
        // For smaller batches, use existing CPU method
        castRaysCPU(ray_queries, results, stats);
    }
}
#endif

std::vector<std::vector<std::vector<std::vector<CollisionDetection::HitResult>>>> CollisionDetection::performGridRayIntersection(
        const vec3 &grid_center, const vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<RayQuery> &ray_queries) {

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
    vec3 voxel_size = make_vec3(grid_size.x / float(grid_divisions.x),
                                grid_size.y / float(grid_divisions.y),
                                grid_size.z / float(grid_divisions.z));

    // Process each ray
    for (const auto &query : ray_queries) {
        HitResult hit_result = castRay(query);

        if (hit_result.hit) {
            // Determine which voxel the hit point falls into
            vec3 relative_pos = hit_result.intersection_point - (grid_center - grid_size * 0.5f);

            int voxel_i = int(relative_pos.x / voxel_size.x);
            int voxel_j = int(relative_pos.y / voxel_size.y);
            int voxel_k = int(relative_pos.z / voxel_size.z);

            // Check bounds
            if (voxel_i >= 0 && voxel_i < grid_divisions.x &&
                voxel_j >= 0 && voxel_j < grid_divisions.y &&
                voxel_k >= 0 && voxel_k < grid_divisions.z) {

                grid_results[voxel_i][voxel_j][voxel_k].push_back(hit_result);
            }
        }
    }

    return grid_results;
}

void CollisionDetection::calculateRayPathLengthsDetailed(const vec3 &grid_center, const vec3 &grid_size,
                                                         const helios::int3 &grid_divisions, const std::vector<vec3> &ray_origins, const std::vector<vec3> &ray_directions,
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

    // Rebuild BVH in the new format if it already exists
    if (!bvh_nodes.empty()) {
        if (printmessages) {
            std::cout << "CollisionDetection: Converting BVH from mode " << static_cast<int>(old_mode)
                      << " to mode " << static_cast<int>(mode) << std::endl;
        }

        convertBVHLayout(old_mode, mode);

        if (printmessages) {
            auto memory_stats = getBVHMemoryUsage();
            std::cout << "CollisionDetection: Memory usage - SoA: " << memory_stats.soa_memory_bytes
                      << " bytes, Quantized: " << memory_stats.quantized_memory_bytes << " bytes ("
                      << memory_stats.quantized_reduction_percent << "% reduction)" << std::endl;
        }
    }
}

CollisionDetection::BVHOptimizationMode CollisionDetection::getBVHOptimizationMode() const {
    return bvh_optimization_mode;
}

void CollisionDetection::convertBVHLayout(BVHOptimizationMode from_mode, BVHOptimizationMode to_mode) {
    if (from_mode == to_mode) return;

    // Convert between SoA formats
    if (from_mode == BVHOptimizationMode::SOA_UNCOMPRESSED) {
        if (to_mode == BVHOptimizationMode::SOA_QUANTIZED) {
            convertSoAToQuantized();
        }
    }
    else if (from_mode == BVHOptimizationMode::SOA_QUANTIZED) {
        if (to_mode == BVHOptimizationMode::SOA_UNCOMPRESSED) {
            convertQuantizedToSoA();
        }
    }
}




void CollisionDetection::convertSoAToQuantized() {
    if (bvh_nodes_soa.node_count == 0) return;

    // Calculate scene bounding box from SoA data
    vec3 scene_min = bvh_nodes_soa.aabb_mins[0];
    vec3 scene_max = bvh_nodes_soa.aabb_maxs[0];

    for (size_t i = 1; i < bvh_nodes_soa.node_count; ++i) {
        scene_min.x = std::min(scene_min.x, bvh_nodes_soa.aabb_mins[i].x);
        scene_min.y = std::min(scene_min.y, bvh_nodes_soa.aabb_mins[i].y);
        scene_min.z = std::min(scene_min.z, bvh_nodes_soa.aabb_mins[i].z);

        scene_max.x = std::max(scene_max.x, bvh_nodes_soa.aabb_maxs[i].x);
        scene_max.y = std::max(scene_max.y, bvh_nodes_soa.aabb_maxs[i].y);
        scene_max.z = std::max(scene_max.z, bvh_nodes_soa.aabb_maxs[i].z);
    }

    bvh_nodes_quantized = QuantizedBVHNodes();
    bvh_nodes_quantized.initializeQuantization(scene_min, scene_max);

    for (size_t i = 0; i < bvh_nodes_soa.node_count; ++i) {
        bvh_nodes_quantized.addNode(
                bvh_nodes_soa.aabb_mins[i], bvh_nodes_soa.aabb_maxs[i],
                bvh_nodes_soa.left_children[i], bvh_nodes_soa.right_children[i],
                bvh_nodes_soa.primitive_starts[i], bvh_nodes_soa.primitive_counts[i],
                bvh_nodes_soa.is_leaf_flags[i] != 0
        );
    }
}


void CollisionDetection::convertQuantizedToSoA() {
    if (bvh_nodes_quantized.node_count == 0) return;

    bvh_nodes_soa.clear();
    bvh_nodes_soa.reserve(bvh_nodes_quantized.node_count);

    for (size_t i = 0; i < bvh_nodes_quantized.node_count; ++i) {
        vec3 aabb_min, aabb_max;
        bvh_nodes_quantized.getAABB(i, aabb_min, aabb_max);

        bvh_nodes_soa.aabb_mins.push_back(aabb_min);
        bvh_nodes_soa.aabb_maxs.push_back(aabb_max);
        bvh_nodes_soa.left_children.push_back(bvh_nodes_quantized.left_children[i]);
        bvh_nodes_soa.right_children.push_back(bvh_nodes_quantized.right_children[i]);

        // Unpack primitive data
        uint32_t packed = bvh_nodes_quantized.primitive_data[i];
        bvh_nodes_soa.primitive_starts.push_back((packed >> 1) & 0x7FFFFFFF);
        bvh_nodes_soa.primitive_counts.push_back((packed >> 16) & 0x7FFF);
        bvh_nodes_soa.is_leaf_flags.push_back((packed & 1) ? 1 : 0);
        bvh_nodes_soa.node_count++;
    }
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysOptimized(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats) {
    if (ray_queries.empty()) {
        return {};
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

        case BVHOptimizationMode::SOA_QUANTIZED:
            results = castRaysQuantized(ray_queries, local_stats);
            break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (printmessages) {
        std::cout << "CollisionDetection: Optimized ray casting completed in "
                  << duration.count() / 1000.0 << " ms using mode "
                  << static_cast<int>(bvh_optimization_mode) << std::endl;
    }

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

    for (auto& packet : ray_stream.packets) {
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
        combined_stats.average_ray_distance = (combined_stats.average_ray_distance *
                                                       (combined_stats.total_rays_cast - packet_stats.total_rays_cast) +
                                               packet_stats.average_ray_distance * packet_stats.total_rays_cast) /
                                              combined_stats.total_rays_cast;
    }

    if (stats) {
        *stats = combined_stats;
    }

    if (printmessages && success) {
        std::cout << "CollisionDetection: Processed " << ray_stream.packets.size()
                  << " ray packets (" << ray_stream.total_rays << " total rays)" << std::endl;
    }

    return success;
}

CollisionDetection::MemoryUsageStats CollisionDetection::getBVHMemoryUsage() const {
    MemoryUsageStats stats;

    // Calculate SoA memory usage
    stats.soa_memory_bytes = bvh_nodes_soa.getMemoryUsage();

    // Calculate quantized memory usage  
    stats.quantized_memory_bytes = bvh_nodes_quantized.getMemoryUsage();

    // Calculate reduction percentage (quantized vs SoA)
    if (stats.soa_memory_bytes > 0) {
        stats.quantized_reduction_percent =
                ((float)(stats.soa_memory_bytes - stats.quantized_memory_bytes) / stats.soa_memory_bytes) * 100.0f;
    }

    return stats;
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysSoA(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats) {
    if (bvh_nodes_soa.node_count == 0) {
        // Fall back to legacy method if SoA not built
        return castRays(ray_queries, &stats);
    }

    const size_t num_rays = ray_queries.size();
    std::vector<HitResult> results;
    stats.total_rays_cast = num_rays;

#ifdef _OPENMP
    // Use OpenMP for parallel SoA ray traversal with thread-safe primitive cache
    const size_t OPENMP_THRESHOLD = 100;

    // Always build primitive cache for SoA mode to ensure thread safety
    buildPrimitiveCache();

    // Check if SIMD optimization should be used for large batches
    const size_t SIMD_THRESHOLD = 32; // Use SIMD for batches of 32+ rays
    bool use_simd = (num_rays >= SIMD_THRESHOLD);

    if (use_simd && num_rays >= OPENMP_THRESHOLD) {
        // Combined OpenMP + SIMD optimization for large batches
        results.resize(num_rays);
        size_t total_hits = 0;
        double total_distance = 0.0;

#pragma omp parallel for reduction(+:total_hits,total_distance) schedule(dynamic, 32)
        for (size_t batch_start = 0; batch_start < num_rays; batch_start += 8) {
            size_t batch_size = std::min(size_t(8), num_rays - batch_start);

            // Prepare ray data for SIMD processing
            alignas(32) vec3 ray_origins[8];
            alignas(32) vec3 ray_directions[8];
            HitResult batch_results[8];

            for (size_t i = 0; i < batch_size; ++i) {
                ray_origins[i] = ray_queries[batch_start + i].origin;
                ray_directions[i] = ray_queries[batch_start + i].direction;
                if (ray_directions[i].magnitude() > 1e-8f) {
                    ray_directions[i] = ray_directions[i] / ray_directions[i].magnitude();
                }
            }

            // Use SIMD-optimized BVH traversal
            traverseBVHSIMD(ray_origins, ray_directions, batch_size, batch_results);

            // Store results and update statistics
            for (size_t i = 0; i < batch_size; ++i) {
                results[batch_start + i] = batch_results[i];
                if (batch_results[i].hit) {
                    total_hits++;
                    total_distance += batch_results[i].distance;
                }
            }
        }

        stats.total_hits = total_hits;
        stats.average_ray_distance = (total_hits > 0) ? (total_distance / total_hits) : 0.0;
        return results;
    } else if (num_rays >= OPENMP_THRESHOLD) {

        results.resize(num_rays); // Pre-allocate for parallel access
        size_t total_hits = 0;
        double total_distance = 0.0;

#pragma omp parallel for reduction(+:total_hits,total_distance) schedule(dynamic, 16)
        for (size_t i = 0; i < num_rays; i++) {
            RayTracingStats thread_stats = {}; // Thread-local stats
            HitResult result = castRaySoATraversal(ray_queries[i], thread_stats);
            results[i] = result;

            if (result.hit) {
                total_hits++;
                total_distance += result.distance;
            }
        }

        stats.total_hits = total_hits;
        stats.average_ray_distance = (total_hits > 0) ? (total_distance / total_hits) : 0.0;
        return results;
    }
#endif

    // Sequential fallback - use push_back approach
    results.reserve(num_rays);
    for (const auto& query : ray_queries) {
        HitResult result = castRaySoATraversal(query, stats);
        results.push_back(result);

        if (result.hit) {
            stats.total_hits++;
        }
    }

    // Calculate average distance for successful hits
    if (stats.total_hits > 0) {
        float total_distance = 0.0f;
        for (const auto& result : results) {
            if (result.hit) {
                total_distance += result.distance;
            }
        }
        stats.average_ray_distance = total_distance / stats.total_hits;
    }

    return results;
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysQuantized(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats) {
    if (bvh_nodes_quantized.node_count == 0) {
        // Fall back to legacy method if quantized BVH not built
        return castRays(ray_queries, &stats);
    }

    const size_t num_rays = ray_queries.size();
    std::vector<HitResult> results;
    stats.total_rays_cast = num_rays;

#ifdef _OPENMP
    // Use OpenMP for parallel quantized ray traversal with thread-safe primitive cache
    const size_t OPENMP_THRESHOLD = 100;

    // Always build primitive cache for Quantized mode to ensure thread safety
    buildPrimitiveCache();

    if (num_rays >= OPENMP_THRESHOLD) {

        results.resize(num_rays); // Pre-allocate for parallel access
        size_t total_hits = 0;
        double total_distance = 0.0;

#pragma omp parallel for reduction(+:total_hits,total_distance) schedule(dynamic, 16)
        for (size_t i = 0; i < num_rays; i++) {
            RayTracingStats thread_stats = {}; // Thread-local stats
            HitResult result = castRayQuantizedTraversal(ray_queries[i], thread_stats);
            results[i] = result;

            if (result.hit) {
                total_hits++;
                total_distance += result.distance;
            }
        }

        stats.total_hits = total_hits;
        stats.average_ray_distance = (total_hits > 0) ? (total_distance / total_hits) : 0.0;
        return results;
    }
#endif

    // Sequential fallback - use push_back approach
    results.reserve(num_rays);
    for (const auto& query : ray_queries) {
        HitResult result = castRayQuantizedTraversal(query, stats);
        results.push_back(result);

        if (result.hit) {
            stats.total_hits++;
        }
    }

    // Calculate average distance for successful hits
    if (stats.total_hits > 0) {
        float total_distance = 0.0f;
        for (const auto& result : results) {
            if (result.hit) {
                total_distance += result.distance;
            }
        }
        stats.average_ray_distance = total_distance / stats.total_hits;
    }

    return results;
}
// ================================================================
// PHASE 2 OPTIMIZATION: SoA and Quantized BVH Traversal Methods
// ================================================================

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

    while (!node_stack.empty()) {
        size_t node_idx = node_stack.top();
        node_stack.pop();
        stats.bvh_nodes_visited++;

        // OPTIMIZATION: AABB intersection test using SoA layout - only loads required data
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
                    for (uint target : query.target_UUIDs) {
                        if (primitive_id == target) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) continue;
                }

                // Perform primitive intersection test
                HitResult primitive_hit = intersectPrimitive(query, primitive_id);
                if (primitive_hit.hit && primitive_hit.distance < closest_distance) {
                    result = primitive_hit;
                    closest_distance = primitive_hit.distance;
                }
            }
        } else {
            // Internal node - add children to stack
            uint32_t left_child = bvh_nodes_soa.left_children[node_idx];
            uint32_t right_child = bvh_nodes_soa.right_children[node_idx];

            if (left_child != 0xFFFFFFFF) {
                node_stack.push(left_child);
            }
            if (right_child != 0xFFFFFFFF) {
                node_stack.push(right_child);
            }
        }
    }

    return result;
}

CollisionDetection::HitResult CollisionDetection::castRayQuantizedTraversal(const RayQuery &query, RayTracingStats &stats) {
    HitResult result;

    if (bvh_nodes_quantized.node_count == 0) {
        return result; // No quantized BVH built
    }

    // Stack-based traversal with quantized data
    std::stack<size_t> node_stack;
    node_stack.push(0); // Start from root

    float closest_distance = (query.max_distance > 0) ? query.max_distance : std::numeric_limits<float>::max();

    while (!node_stack.empty()) {
        size_t node_idx = node_stack.top();
        node_stack.pop();
        stats.bvh_nodes_visited++;

        // OPTIMIZATION: AABB intersection test with quantized coordinates
        if (!aabbIntersectQuantized(query.origin, query.direction, closest_distance, node_idx)) {
            continue;
        }

        // Check if leaf node (packed in primitive_data)
        uint32_t packed_data = bvh_nodes_quantized.primitive_data[node_idx];
        bool is_leaf = (packed_data & 1) != 0;

        if (is_leaf) {
            // Process primitives in this leaf
            uint32_t primitive_start = (packed_data >> 1) & 0x7FFFFFFF;
            uint32_t primitive_count = (packed_data >> 16) & 0x7FFF;

            for (uint32_t i = 0; i < primitive_count; ++i) {
                uint primitive_id = primitive_indices[primitive_start + i];

                // Skip if not in target list (if specified)
                if (!query.target_UUIDs.empty()) {
                    bool found = false;
                    for (uint target : query.target_UUIDs) {
                        if (primitive_id == target) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) continue;
                }

                // Perform primitive intersection test
                HitResult primitive_hit = intersectPrimitive(query, primitive_id);
                if (primitive_hit.hit && primitive_hit.distance < closest_distance) {
                    result = primitive_hit;
                    closest_distance = primitive_hit.distance;
                }
            }
        } else {
            // Internal node - add children to stack
            uint32_t left_child = bvh_nodes_quantized.left_children[node_idx];
            uint32_t right_child = bvh_nodes_quantized.right_children[node_idx];

            if (left_child != 0xFFFFFFFF) {
                node_stack.push(left_child);
            }
            if (right_child != 0xFFFFFFFF) {
                node_stack.push(right_child);
            }
        }
    }

    return result;
}

bool CollisionDetection::aabbIntersectSoA(const helios::vec3& ray_origin, const helios::vec3& ray_direction, float max_distance, size_t node_index) const {
    // OPTIMIZATION: Direct access to SoA arrays - only loads AABB data (24 bytes vs 48 bytes in AoS)
    const vec3& aabb_min = bvh_nodes_soa.aabb_mins[node_index];
    const vec3& aabb_max = bvh_nodes_soa.aabb_maxs[node_index];

    // Optimized ray-AABB intersection using slab method
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
}

bool CollisionDetection::aabbIntersectQuantized(const helios::vec3& ray_origin, const helios::vec3& ray_direction, float max_distance, size_t node_index) const {
    // OPTIMIZATION: Dequantize on-the-fly (trades compute for memory bandwidth)
    vec3 aabb_min, aabb_max;
    bvh_nodes_quantized.getAABB(node_index, aabb_min, aabb_max);

    // Optimized ray-AABB intersection using slab method
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
}

// Helper method to intersect with individual primitive (reuses existing logic)
CollisionDetection::HitResult CollisionDetection::intersectPrimitive(const RayQuery &query, uint primitive_id) {
    // Thread-safe primitive intersection using cached primitive data
    return intersectPrimitiveThreadSafe(query.origin, query.direction, primitive_id, query.max_distance);
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
            }
        }
    }
    // Add other primitive types as needed (VOXEL, DISK, etc.)

    return result;
}

void CollisionDetection::buildPrimitiveCache() {
    primitive_cache.clear();

    // Get all primitive UUIDs from context
    std::vector<uint> all_primitives = context->getAllUUIDs();

    // Cache primitive data for thread-safe access
    for (uint primitive_id : all_primitives) {
        if (context->doesPrimitiveExist(primitive_id)) {
            try {
                PrimitiveType type = context->getPrimitiveType(primitive_id);
                std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_id);

                primitive_cache[primitive_id] = CachedPrimitive(type, vertices);
            } catch (const std::exception& e) {
                // Skip this primitive if it no longer exists or can't be accessed
                // This can happen when UUIDs from previous contexts persist
                if (printmessages) {
                    std::cout << "Warning: Skipping primitive " << primitive_id
                              << " in cache build (not accessible: " << e.what() << ")" << std::endl;
                }
                continue;
            }
        }
    }

    if (printmessages) {
        std::cout << "Built primitive cache with " << primitive_cache.size() << " primitives" << std::endl;
    }
}

bool CollisionDetection::triangleIntersect(const vec3 &origin, const vec3 &direction,
                                           const vec3 &v0, const vec3 &v1, const vec3 &v2, float &distance) {
    // Möller-Trumbore triangle intersection algorithm (thread-safe)
    const float EPSILON = 1e-8f;

    // Helper lambda for dot product
    auto dot = [](const vec3 &a, const vec3 &b) -> float {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = helios::cross(direction, edge2);
    float a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) {
        return false; // Ray is parallel to triangle
    }

    float f = 1.0f / a;
    vec3 s = origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    vec3 q = helios::cross(s, edge1);
    float v = f * dot(direction, q);

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = f * dot(edge2, q);

    if (t > EPSILON) {
        distance = t;
        return true;
    }

    return false; // Line intersection but not ray intersection
}

#ifdef HELIOS_CUDA_AVAILABLE
#include <vector_types.h>  // For make_float3

/**
 * \brief Cast rays using warp-efficient GPU kernels
 * \param[in] ray_queries Vector of ray queries to process
 * \param[out] stats Ray-tracing performance statistics
 * \return Vector of HitResult with optimal GPU performance
 */
std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysGPUPhase3(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats) {
    std::vector<HitResult> results;
    results.resize(ray_queries.size());

    if (ray_queries.empty() || !gpu_acceleration_enabled) {
        return castRaysSoA(ray_queries, stats); // CPU fallback
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Ensure BVH is built and in SoA format for GPU
    setBVHOptimizationMode(BVHOptimizationMode::SOA_UNCOMPRESSED);

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

    // Get all primitives in context
    std::vector<uint> all_primitives = context->getAllUUIDs();

    // Prepare primitive data
    std::vector<float> primitive_aabb_min(all_primitives.size() * 3);
    std::vector<float> primitive_aabb_max(all_primitives.size() * 3);

    for (size_t i = 0; i < all_primitives.size(); i++) {
        vec3 aabb_min, aabb_max;
        context->getPrimitiveBoundingBox(all_primitives[i], aabb_min, aabb_max);

        primitive_aabb_min[i * 3] = aabb_min.x;
        primitive_aabb_min[i * 3 + 1] = aabb_min.y;
        primitive_aabb_min[i * 3 + 2] = aabb_min.z;

        primitive_aabb_max[i * 3] = aabb_max.x;
        primitive_aabb_max[i * 3 + 1] = aabb_max.y;
        primitive_aabb_max[i * 3 + 2] = aabb_max.z;
    }

    // Prepare results arrays
    const int max_results_per_ray = 1; // For collision detection, we typically want first hit
    std::vector<unsigned int> gpu_results(ray_queries.size() * max_results_per_ray, 0);
    std::vector<unsigned int> gpu_result_counts(ray_queries.size(), 0);

    // Get primitive indices
    std::vector<unsigned int> primitive_indices(all_primitives.size());
    for (size_t i = 0; i < all_primitives.size(); i++) {
        primitive_indices[i] = all_primitives[i];
    }

    // Create GPU SoA BVH structure for warp-efficient kernel
    // For now, use a simplified approach that works with existing BVH data
    if (bvh_nodes.empty()) {
        // No BVH built, fall back to CPU
        return castRaysSoA(ray_queries, stats);
    }

    // Create minimal GPU SoA structure on the stack
    struct {
        float3 *aabb_mins;
        float3 *aabb_maxs;
        uint32_t *left_children;
        uint32_t *right_children;
        uint32_t *primitive_starts;
        uint32_t *primitive_counts;
        uint8_t *is_leaf_flags;
        size_t node_count;
    } gpu_bvh_soa;

    // Allocate minimal arrays for demonstration
    std::vector<float3> soa_aabb_mins(bvh_nodes.size());
    std::vector<float3> soa_aabb_maxs(bvh_nodes.size());
    std::vector<uint32_t> soa_left_children(bvh_nodes.size());
    std::vector<uint32_t> soa_right_children(bvh_nodes.size());
    std::vector<uint32_t> soa_primitive_starts(bvh_nodes.size());
    std::vector<uint32_t> soa_primitive_counts(bvh_nodes.size());
    std::vector<uint8_t> soa_is_leaf_flags(bvh_nodes.size());

    // Convert existing BVH to SoA format
    for (size_t i = 0; i < bvh_nodes.size(); i++) {
        soa_aabb_mins[i] = make_float3(bvh_nodes[i].aabb_min.x, bvh_nodes[i].aabb_min.y, bvh_nodes[i].aabb_min.z);
        soa_aabb_maxs[i] = make_float3(bvh_nodes[i].aabb_max.x, bvh_nodes[i].aabb_max.y, bvh_nodes[i].aabb_max.z);
        soa_left_children[i] = bvh_nodes[i].left_child;
        soa_right_children[i] = bvh_nodes[i].right_child;
        soa_primitive_starts[i] = bvh_nodes[i].primitive_start;
        soa_primitive_counts[i] = bvh_nodes[i].primitive_count;
        soa_is_leaf_flags[i] = bvh_nodes[i].is_leaf ? 1 : 0;
    }

    // Set up GPU SoA structure
    gpu_bvh_soa.aabb_mins = soa_aabb_mins.data();
    gpu_bvh_soa.aabb_maxs = soa_aabb_maxs.data();
    gpu_bvh_soa.left_children = soa_left_children.data();
    gpu_bvh_soa.right_children = soa_right_children.data();
    gpu_bvh_soa.primitive_starts = soa_primitive_starts.data();
    gpu_bvh_soa.primitive_counts = soa_primitive_counts.data();
    gpu_bvh_soa.is_leaf_flags = soa_is_leaf_flags.data();
    gpu_bvh_soa.node_count = bvh_nodes.size();

    // Call warp-efficient GPU kernel
    launchWarpEfficientBVH(
            &gpu_bvh_soa,
            primitive_indices.data(),
            static_cast<int>(all_primitives.size()),
            primitive_aabb_min.data(),
            primitive_aabb_max.data(),
            ray_origins.data(),
            ray_directions.data(),
            ray_max_distances.data(),
            static_cast<int>(ray_queries.size()),
            gpu_results.data(),
            gpu_result_counts.data(),
            max_results_per_ray
    );

    // Convert GPU results back to HitResult format
    size_t hit_count = 0;
    for (size_t i = 0; i < ray_queries.size(); i++) {
        if (gpu_result_counts[i] > 0) {
            results[i].hit = true;
            results[i].primitive_UUID = gpu_results[i * max_results_per_ray];
            results[i].distance = ray_queries[i].max_distance; // GPU doesn't return distance yet
            hit_count++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats.total_rays_cast = ray_queries.size();
    stats.total_hits = hit_count;

    return results;
}

// ================================================================
// SIMD-OPTIMIZED RAY TRACING METHODS
// ================================================================

uint32_t CollisionDetection::rayAABBIntersectSIMD(const vec3 *ray_origins, const vec3 *ray_directions,
                                                  const vec3 *aabb_mins, const vec3 *aabb_maxs,
                                                  float *t_mins, float *t_maxs, int count) {
#ifdef __AVX2__
    if (count == 8) {
        // AVX2 implementation for 8 rays at once
        uint32_t hit_mask = 0;

        for (int i = 0; i < 8; i += 8) {
            // Load 8 ray origins
            __m256 orig_x = _mm256_set_ps(ray_origins[i+7].x, ray_origins[i+6].x, ray_origins[i+5].x, ray_origins[i+4].x,
                                          ray_origins[i+3].x, ray_origins[i+2].x, ray_origins[i+1].x, ray_origins[i+0].x);
            __m256 orig_y = _mm256_set_ps(ray_origins[i+7].y, ray_origins[i+6].y, ray_origins[i+5].y, ray_origins[i+4].y,
                                          ray_origins[i+3].y, ray_origins[i+2].y, ray_origins[i+1].y, ray_origins[i+0].y);
            __m256 orig_z = _mm256_set_ps(ray_origins[i+7].z, ray_origins[i+6].z, ray_origins[i+5].z, ray_origins[i+4].z,
                                          ray_origins[i+3].z, ray_origins[i+2].z, ray_origins[i+1].z, ray_origins[i+0].z);

            // Load 8 ray directions
            __m256 dir_x = _mm256_set_ps(ray_directions[i+7].x, ray_directions[i+6].x, ray_directions[i+5].x, ray_directions[i+4].x,
                                         ray_directions[i+3].x, ray_directions[i+2].x, ray_directions[i+1].x, ray_directions[i+0].x);
            __m256 dir_y = _mm256_set_ps(ray_directions[i+7].y, ray_directions[i+6].y, ray_directions[i+5].y, ray_directions[i+4].y,
                                         ray_directions[i+3].y, ray_directions[i+2].y, ray_directions[i+1].y, ray_directions[i+0].y);
            __m256 dir_z = _mm256_set_ps(ray_directions[i+7].z, ray_directions[i+6].z, ray_directions[i+5].z, ray_directions[i+4].z,
                                         ray_directions[i+3].z, ray_directions[i+2].z, ray_directions[i+1].z, ray_directions[i+0].z);

            // Load 8 AABB mins
            __m256 aabb_min_x = _mm256_set_ps(aabb_mins[i+7].x, aabb_mins[i+6].x, aabb_mins[i+5].x, aabb_mins[i+4].x,
                                              aabb_mins[i+3].x, aabb_mins[i+2].x, aabb_mins[i+1].x, aabb_mins[i+0].x);
            __m256 aabb_min_y = _mm256_set_ps(aabb_mins[i+7].y, aabb_mins[i+6].y, aabb_mins[i+5].y, aabb_mins[i+4].y,
                                              aabb_mins[i+3].y, aabb_mins[i+2].y, aabb_mins[i+1].y, aabb_mins[i+0].y);
            __m256 aabb_min_z = _mm256_set_ps(aabb_mins[i+7].z, aabb_mins[i+6].z, aabb_mins[i+5].z, aabb_mins[i+4].z,
                                              aabb_mins[i+3].z, aabb_mins[i+2].z, aabb_mins[i+1].z, aabb_mins[i+0].z);

            // Load 8 AABB maxs
            __m256 aabb_max_x = _mm256_set_ps(aabb_maxs[i+7].x, aabb_maxs[i+6].x, aabb_maxs[i+5].x, aabb_maxs[i+4].x,
                                              aabb_maxs[i+3].x, aabb_maxs[i+2].x, aabb_maxs[i+1].x, aabb_maxs[i+0].x);
            __m256 aabb_max_y = _mm256_set_ps(aabb_maxs[i+7].y, aabb_maxs[i+6].y, aabb_maxs[i+5].y, aabb_maxs[i+4].y,
                                              aabb_maxs[i+3].y, aabb_maxs[i+2].y, aabb_maxs[i+1].y, aabb_maxs[i+0].y);
            __m256 aabb_max_z = _mm256_set_ps(aabb_maxs[i+7].z, aabb_maxs[i+6].z, aabb_maxs[i+5].z, aabb_maxs[i+4].z,
                                              aabb_maxs[i+3].z, aabb_maxs[i+2].z, aabb_maxs[i+1].z, aabb_maxs[i+0].z);

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
            __m256 hits = _mm256_and_ps(_mm256_cmp_ps(t_max_final, zero, _CMP_GE_OS),
                                        _mm256_cmp_ps(t_min_final, t_max_final, _CMP_LE_OS));

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
            __m128 orig_x = _mm_set_ps(ray_origins[i+3].x, ray_origins[i+2].x, ray_origins[i+1].x, ray_origins[i+0].x);
            __m128 orig_y = _mm_set_ps(ray_origins[i+3].y, ray_origins[i+2].y, ray_origins[i+1].y, ray_origins[i+0].y);
            __m128 orig_z = _mm_set_ps(ray_origins[i+3].z, ray_origins[i+2].z, ray_origins[i+1].z, ray_origins[i+0].z);

            // Load 4 ray directions
            __m128 dir_x = _mm_set_ps(ray_directions[i+3].x, ray_directions[i+2].x, ray_directions[i+1].x, ray_directions[i+0].x);
            __m128 dir_y = _mm_set_ps(ray_directions[i+3].y, ray_directions[i+2].y, ray_directions[i+1].y, ray_directions[i+0].y);
            __m128 dir_z = _mm_set_ps(ray_directions[i+3].z, ray_directions[i+2].z, ray_directions[i+1].z, ray_directions[i+0].z);

            // Load 4 AABB mins
            __m128 aabb_min_x = _mm_set_ps(aabb_mins[i+3].x, aabb_mins[i+2].x, aabb_mins[i+1].x, aabb_mins[i+0].x);
            __m128 aabb_min_y = _mm_set_ps(aabb_mins[i+3].y, aabb_mins[i+2].y, aabb_mins[i+1].y, aabb_mins[i+0].y);
            __m128 aabb_min_z = _mm_set_ps(aabb_mins[i+3].z, aabb_mins[i+2].z, aabb_mins[i+1].z, aabb_mins[i+0].z);

            // Load 4 AABB maxs
            __m128 aabb_max_x = _mm_set_ps(aabb_maxs[i+3].x, aabb_maxs[i+2].x, aabb_maxs[i+1].x, aabb_maxs[i+0].x);
            __m128 aabb_max_y = _mm_set_ps(aabb_maxs[i+3].y, aabb_maxs[i+2].y, aabb_maxs[i+1].y, aabb_maxs[i+0].y);
            __m128 aabb_max_z = _mm_set_ps(aabb_maxs[i+3].z, aabb_maxs[i+2].z, aabb_maxs[i+1].z, aabb_maxs[i+0].z);

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
            __m128 hits = _mm_and_ps(_mm_cmpge_ps(t_max_final, zero),
                                     _mm_cmple_ps(t_min_final, t_max_final));

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

void CollisionDetection::traverseBVHSIMD(const vec3 *ray_origins, const vec3 *ray_directions,
                                         int count, HitResult *results) {
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
            traverseBVHSIMDImpl(&ray_origins[batch_start], &ray_directions[batch_start],
                                batch_count, &results[batch_start]);
        } else {
            // Scalar fallback for remaining rays
            for (int i = 0; i < batch_count; ++i) {
                int ray_idx = batch_start + i;
                results[ray_idx] = castRay(RayQuery(ray_origins[ray_idx], ray_directions[ray_idx]));
            }
        }
    }
}

void CollisionDetection::traverseBVHSIMDImpl(const vec3 *ray_origins, const vec3 *ray_directions,
                                             int count, HitResult *results) {
    const size_t MAX_STACK_SIZE = 64;

    // Per-ray data structures - using aligned arrays for SIMD efficiency
    alignas(32) uint32_t node_stacks[8][MAX_STACK_SIZE];  // 8 stacks for up to 8 rays
    alignas(32) uint32_t stack_tops[8] = {0};             // Current stack positions
    alignas(32) float closest_distances[8];               // Closest hit distances per ray
    alignas(32) bool ray_active[8];                       // Which rays are still active

    // Initialize per-ray state
    for (int i = 0; i < count; ++i) {
        node_stacks[i][0] = 0;  // Start with root node
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
        if (!any_active) break;

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

                if (test_count == count) break; // Batch is full
            }
        }

        if (test_count == 0) break;

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
        uint32_t hit_mask = rayAABBIntersectSIMD(batch_origins, batch_directions,
                                                 test_aabb_mins, test_aabb_maxs,
                                                 t_mins, t_maxs, test_count);

        // Process intersection results
        for (int i = 0; i < test_count; ++i) {
            if (!(hit_mask & (1 << i))) continue; // Ray missed this AABB

            int ray_idx = test_ray_indices[i];
            uint32_t node_idx = test_node_indices[i];
            const BVHNode &node = bvh_nodes[node_idx];

            if (t_mins[i] > closest_distances[ray_idx]) continue; // Beyond closest hit

            if (node.is_leaf) {
                // Test primitives in leaf node
                for (uint32_t prim_idx = node.primitive_start;
                     prim_idx < node.primitive_start + node.primitive_count; ++prim_idx) {

                    uint32_t primitive_id = primitive_indices[prim_idx];

                    // Use thread-safe primitive intersection
                    HitResult prim_result = intersectPrimitiveThreadSafe(batch_origins[i], batch_directions[i],
                                                                         primitive_id, closest_distances[ray_idx]);
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
                }
            }
        }
    }
}
#endif