/**
 * @file CollisionDetection_RayTracing.cpp
 * @brief Ray-tracing functionality for the CollisionDetection plugin
 * 
 * This file contains all ray-tracing related methods including:
 * - Ray casting and intersection tests
 * - BVH traversal algorithms (legacy, SoA, and quantized)
 * - Phase 2 optimization methods
 * - Ray streaming interfaces
 * - AABB intersection utilities
 */

#include "CollisionDetection.h"
#include <stack>
#include <thread>
#include <chrono>

using namespace helios;

// ================================================================
// RAY CASTING CORE INTERFACE
// ================================================================

CollisionDetection::HitResult CollisionDetection::castRay(const RayQuery &ray_query) {
    return castRay(ray_query.origin, ray_query.direction, ray_query.max_distance, ray_query.target_UUIDs);
}

CollisionDetection::HitResult CollisionDetection::castRay(const vec3 &origin, const vec3 &direction, float max_distance, const std::vector<uint> &target_UUIDs) {
    HitResult result;
    if (!isBVHValid()) {
        return result;
    }

    // Normalize direction vector
    vec3 normalized_direction = direction;
    float magnitude = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    if (magnitude > 1e-6f) {
        normalized_direction = direction / magnitude;
    } else {
        return result; // Invalid direction
    }

    float closest_distance = (max_distance > 0) ? max_distance : std::numeric_limits<float>::max();
    result.distance = closest_distance;

    // Use optimized traversal if available and enabled
    if (bvh_optimization_mode != BVHOptimizationMode::LEGACY_AOS) {
        RayQuery query(origin, normalized_direction, max_distance, target_UUIDs);
        RayTracingStats stats;
        
        if (bvh_optimization_mode == BVHOptimizationMode::SOA_UNCOMPRESSED) {
            return castRaySoATraversal(query, stats);
        } else if (bvh_optimization_mode == BVHOptimizationMode::SOA_QUANTIZED) {
            return castRayQuantizedTraversal(query, stats);
        }
    }

    // Legacy BVH traversal (stack-based for better performance)
    std::stack<uint> node_stack;
    if (!bvh_nodes.empty()) {
        node_stack.push(0); // Start from root
    }

    while (!node_stack.empty()) {
        uint node_index = node_stack.top();
        node_stack.pop();

        const BVHNode &node = bvh_nodes[node_index];

        // Check ray-AABB intersection
        float t_min, t_max;
        if (!rayAABBIntersect(origin, normalized_direction, node.aabb_min, node.aabb_max, t_min, t_max) ||
            t_min > closest_distance) {
            continue;
        }

        if (node.is_leaf) {
            // Check intersection with primitives in this leaf
            for (uint i = node.primitive_start; i < node.primitive_start + node.primitive_count; ++i) {
                uint primitive_id = primitive_indices[i];
                
                // Skip if not in target list (if specified)
                if (!target_UUIDs.empty()) {
                    bool found = false;
                    for (uint target : target_UUIDs) {
                        if (primitive_id == target) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) continue;
                }

                float distance;
                if (rayPrimitiveIntersection(origin, normalized_direction, primitive_id, distance)) {
                    if (distance < closest_distance && distance >= 0) {
                        result.hit = true;
                        result.distance = distance;
                        result.primitive_UUID = primitive_id;
                        closest_distance = distance;
                    }
                }
            }
        } else {
            // Add children to stack (right child first for depth-first traversal)
            if (node.right_child != 0xFFFFFFFF) {
                node_stack.push(node.right_child);
            }
            if (node.left_child != 0xFFFFFFFF) {
                node_stack.push(node.left_child);
            }
        }
    }

    return result;
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRays(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats) {
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());
    
    RayTracingStats local_stats;
    local_stats.total_rays_cast = ray_queries.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (gpu_acceleration_enabled && bvh_nodes.size() > 100) {
        castRaysGPU(ray_queries, results, local_stats);
    } else {
        castRaysCPU(ray_queries, results, local_stats);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (stats) {
        *stats = local_stats;
    }
    
    return results;
}

void CollisionDetection::castRaysCPU(const std::vector<RayQuery> &ray_queries, std::vector<HitResult> &results, RayTracingStats &stats) {
    results.clear();
    results.reserve(ray_queries.size());
    
    for (const auto &query : ray_queries) {
        HitResult result = castRay(query);
        results.push_back(result);
        
        if (result.hit) {
            stats.total_hits++;
        }
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
            std::cout << "CollisionDetection: Memory usage - Legacy: " << memory_stats.legacy_memory_bytes 
                      << " bytes, SoA: " << memory_stats.soa_memory_bytes 
                      << " bytes, Quantized: " << memory_stats.quantized_memory_bytes << " bytes" << std::endl;
        }
    }
}

CollisionDetection::BVHOptimizationMode CollisionDetection::getBVHOptimizationMode() const {
    return bvh_optimization_mode;
}

void CollisionDetection::convertBVHLayout(BVHOptimizationMode from_mode, BVHOptimizationMode to_mode) {
    if (from_mode == to_mode) return;
    
    // Convert from legacy AoS to SoA formats
    if (from_mode == BVHOptimizationMode::LEGACY_AOS) {
        if (to_mode == BVHOptimizationMode::SOA_UNCOMPRESSED) {
            convertLegacyToSoA();
        } else if (to_mode == BVHOptimizationMode::SOA_QUANTIZED) {
            convertLegacyToQuantized();
        }
    }
    // Convert from SoA uncompressed to other formats  
    else if (from_mode == BVHOptimizationMode::SOA_UNCOMPRESSED) {
        if (to_mode == BVHOptimizationMode::LEGACY_AOS) {
            convertSoAToLegacy();
        } else if (to_mode == BVHOptimizationMode::SOA_QUANTIZED) {
            convertSoAToQuantized();
        }
    }
    // Convert from quantized to other formats
    else if (from_mode == BVHOptimizationMode::SOA_QUANTIZED) {
        if (to_mode == BVHOptimizationMode::LEGACY_AOS) {
            convertQuantizedToLegacy();
        } else if (to_mode == BVHOptimizationMode::SOA_UNCOMPRESSED) {
            convertQuantizedToSoA();
        }
    }
}

void CollisionDetection::convertLegacyToSoA() {
    if (bvh_nodes.empty()) return;
    
    // Initialize SoA data structures
    bvh_nodes_soa.node_count = bvh_nodes.size();
    bvh_nodes_soa.aabb_mins.resize(bvh_nodes.size());
    bvh_nodes_soa.aabb_maxs.resize(bvh_nodes.size());
    bvh_nodes_soa.left_children.resize(bvh_nodes.size());
    bvh_nodes_soa.right_children.resize(bvh_nodes.size());
    bvh_nodes_soa.primitive_starts.resize(bvh_nodes.size());
    bvh_nodes_soa.primitive_counts.resize(bvh_nodes.size());
    bvh_nodes_soa.is_leaf_flags.resize(bvh_nodes.size());
    
    // Convert each node from AoS to SoA
    for (size_t i = 0; i < bvh_nodes.size(); ++i) {
        const BVHNode& aos_node = bvh_nodes[i];
        
        bvh_nodes_soa.aabb_mins[i] = aos_node.aabb_min;
        bvh_nodes_soa.aabb_maxs[i] = aos_node.aabb_max;
        bvh_nodes_soa.left_children[i] = aos_node.left_child;
        bvh_nodes_soa.right_children[i] = aos_node.right_child;
        bvh_nodes_soa.primitive_starts[i] = aos_node.primitive_start;
        bvh_nodes_soa.primitive_counts[i] = aos_node.primitive_count;
        bvh_nodes_soa.is_leaf_flags[i] = aos_node.is_leaf;
    }
}

void CollisionDetection::convertLegacyToQuantized() {
    if (bvh_nodes.empty()) return;
    
    // Calculate scene bounding box for quantization
    vec3 scene_min = bvh_nodes[0].aabb_min;
    vec3 scene_max = bvh_nodes[0].aabb_max;
    
    for (const auto& node : bvh_nodes) {
        scene_min.x = std::min(scene_min.x, node.aabb_min.x);
        scene_min.y = std::min(scene_min.y, node.aabb_min.y);
        scene_min.z = std::min(scene_min.z, node.aabb_min.z);
        scene_max.x = std::max(scene_max.x, node.aabb_max.x);
        scene_max.y = std::max(scene_max.y, node.aabb_max.y);
        scene_max.z = std::max(scene_max.z, node.aabb_max.z);
    }
    
    // Initialize quantized structure
    bvh_nodes_quantized.node_count = bvh_nodes.size();
    bvh_nodes_quantized.initializeQuantization(scene_min, scene_max);
    bvh_nodes_quantized.aabb_mins_x.resize(bvh_nodes.size());
    bvh_nodes_quantized.aabb_mins_y.resize(bvh_nodes.size());
    bvh_nodes_quantized.aabb_mins_z.resize(bvh_nodes.size());
    bvh_nodes_quantized.aabb_maxs_x.resize(bvh_nodes.size());
    bvh_nodes_quantized.aabb_maxs_y.resize(bvh_nodes.size());
    bvh_nodes_quantized.aabb_maxs_z.resize(bvh_nodes.size());
    bvh_nodes_quantized.left_children.resize(bvh_nodes.size());
    bvh_nodes_quantized.right_children.resize(bvh_nodes.size());
    bvh_nodes_quantized.primitive_data.resize(bvh_nodes.size());
    
    // Convert each node to quantized format
    for (size_t i = 0; i < bvh_nodes.size(); ++i) {
        const BVHNode& aos_node = bvh_nodes[i];
        
        // Quantize AABB coordinates manually (no quantizeAABB helper method)
        bvh_nodes_quantized.aabb_mins_x[i] = bvh_nodes_quantized.quantizeCoordinate(aos_node.aabb_min.x, bvh_nodes_quantized.scene_min.x, bvh_nodes_quantized.quantization_scale.x);
        bvh_nodes_quantized.aabb_mins_y[i] = bvh_nodes_quantized.quantizeCoordinate(aos_node.aabb_min.y, bvh_nodes_quantized.scene_min.y, bvh_nodes_quantized.quantization_scale.y);
        bvh_nodes_quantized.aabb_mins_z[i] = bvh_nodes_quantized.quantizeCoordinate(aos_node.aabb_min.z, bvh_nodes_quantized.scene_min.z, bvh_nodes_quantized.quantization_scale.z);
        bvh_nodes_quantized.aabb_maxs_x[i] = bvh_nodes_quantized.quantizeCoordinate(aos_node.aabb_max.x, bvh_nodes_quantized.scene_min.x, bvh_nodes_quantized.quantization_scale.x);
        bvh_nodes_quantized.aabb_maxs_y[i] = bvh_nodes_quantized.quantizeCoordinate(aos_node.aabb_max.y, bvh_nodes_quantized.scene_min.y, bvh_nodes_quantized.quantization_scale.y);
        bvh_nodes_quantized.aabb_maxs_z[i] = bvh_nodes_quantized.quantizeCoordinate(aos_node.aabb_max.z, bvh_nodes_quantized.scene_min.z, bvh_nodes_quantized.quantization_scale.z);
        bvh_nodes_quantized.left_children[i] = aos_node.left_child;
        bvh_nodes_quantized.right_children[i] = aos_node.right_child;
        
        // Pack primitive data: [leaf_flag:1][primitive_count:15][primitive_start:16]
        uint32_t packed_data = aos_node.is_leaf ? 1 : 0;
        packed_data |= (aos_node.primitive_count & 0x7FFF) << 16;
        packed_data |= (aos_node.primitive_start & 0xFFFF) << 1;
        bvh_nodes_quantized.primitive_data[i] = packed_data;
    }
}

void CollisionDetection::convertSoAToLegacy() {
    if (bvh_nodes_soa.node_count == 0) return;
    
    // Clear and resize legacy nodes
    bvh_nodes.clear();
    bvh_nodes.resize(bvh_nodes_soa.node_count);
    
    // Convert each node from SoA to AoS
    for (size_t i = 0; i < bvh_nodes_soa.node_count; ++i) {
        BVHNode& aos_node = bvh_nodes[i];
        
        aos_node.aabb_min = bvh_nodes_soa.aabb_mins[i];
        aos_node.aabb_max = bvh_nodes_soa.aabb_maxs[i];
        aos_node.left_child = bvh_nodes_soa.left_children[i];
        aos_node.right_child = bvh_nodes_soa.right_children[i];
        aos_node.primitive_start = bvh_nodes_soa.primitive_starts[i];
        aos_node.primitive_count = bvh_nodes_soa.primitive_counts[i];
        aos_node.is_leaf = bvh_nodes_soa.is_leaf_flags[i];
    }
}

void CollisionDetection::convertSoAToQuantized() {
    if (bvh_nodes_soa.node_count == 0) return;
    
    // Calculate scene bounding box
    vec3 scene_min, scene_max;
    if (!bvh_nodes_soa.aabb_mins.empty()) {
        scene_min = scene_max = bvh_nodes_soa.aabb_mins[0];
        for (size_t i = 0; i < bvh_nodes_soa.node_count; ++i) {
            const vec3& min_bound = bvh_nodes_soa.aabb_mins[i];
            const vec3& max_bound = bvh_nodes_soa.aabb_maxs[i];
            scene_min.x = std::min(scene_min.x, min_bound.x);
            scene_min.y = std::min(scene_min.y, min_bound.y);
            scene_min.z = std::min(scene_min.z, min_bound.z);
            scene_max.x = std::max(scene_max.x, max_bound.x);
            scene_max.y = std::max(scene_max.y, max_bound.y);
            scene_max.z = std::max(scene_max.z, max_bound.z);
        }
    }
    
    // Initialize quantized structure
    bvh_nodes_quantized.node_count = bvh_nodes_soa.node_count;
    bvh_nodes_quantized.initializeQuantization(scene_min, scene_max);
    bvh_nodes_quantized.aabb_mins_x.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.aabb_mins_y.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.aabb_mins_z.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.aabb_maxs_x.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.aabb_maxs_y.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.aabb_maxs_z.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.left_children.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.right_children.resize(bvh_nodes_soa.node_count);
    bvh_nodes_quantized.primitive_data.resize(bvh_nodes_soa.node_count);
    
    // Convert each node
    for (size_t i = 0; i < bvh_nodes_soa.node_count; ++i) {
        // Quantize AABB coordinates manually
        bvh_nodes_quantized.aabb_mins_x[i] = bvh_nodes_quantized.quantizeCoordinate(bvh_nodes_soa.aabb_mins[i].x, bvh_nodes_quantized.scene_min.x, bvh_nodes_quantized.quantization_scale.x);
        bvh_nodes_quantized.aabb_mins_y[i] = bvh_nodes_quantized.quantizeCoordinate(bvh_nodes_soa.aabb_mins[i].y, bvh_nodes_quantized.scene_min.y, bvh_nodes_quantized.quantization_scale.y);
        bvh_nodes_quantized.aabb_mins_z[i] = bvh_nodes_quantized.quantizeCoordinate(bvh_nodes_soa.aabb_mins[i].z, bvh_nodes_quantized.scene_min.z, bvh_nodes_quantized.quantization_scale.z);
        bvh_nodes_quantized.aabb_maxs_x[i] = bvh_nodes_quantized.quantizeCoordinate(bvh_nodes_soa.aabb_maxs[i].x, bvh_nodes_quantized.scene_min.x, bvh_nodes_quantized.quantization_scale.x);
        bvh_nodes_quantized.aabb_maxs_y[i] = bvh_nodes_quantized.quantizeCoordinate(bvh_nodes_soa.aabb_maxs[i].y, bvh_nodes_quantized.scene_min.y, bvh_nodes_quantized.quantization_scale.y);
        bvh_nodes_quantized.aabb_maxs_z[i] = bvh_nodes_quantized.quantizeCoordinate(bvh_nodes_soa.aabb_maxs[i].z, bvh_nodes_quantized.scene_min.z, bvh_nodes_quantized.quantization_scale.z);
        bvh_nodes_quantized.left_children[i] = bvh_nodes_soa.left_children[i];
        bvh_nodes_quantized.right_children[i] = bvh_nodes_soa.right_children[i];
        
        // Pack primitive data
        uint32_t packed_data = bvh_nodes_soa.is_leaf_flags[i] ? 1 : 0;
        packed_data |= (bvh_nodes_soa.primitive_counts[i] & 0x7FFF) << 16;
        packed_data |= (bvh_nodes_soa.primitive_starts[i] & 0xFFFF) << 1;
        bvh_nodes_quantized.primitive_data[i] = packed_data;
    }
}

void CollisionDetection::convertQuantizedToLegacy() {
    if (bvh_nodes_quantized.node_count == 0) return;
    
    // Clear and resize legacy nodes
    bvh_nodes.clear();
    bvh_nodes.resize(bvh_nodes_quantized.node_count);
    
    // Convert each node from quantized to AoS
    for (size_t i = 0; i < bvh_nodes_quantized.node_count; ++i) {
        BVHNode& aos_node = bvh_nodes[i];
        
        // Dequantize AABB
        bvh_nodes_quantized.getAABB(i, aos_node.aabb_min, aos_node.aabb_max);
        
        aos_node.left_child = bvh_nodes_quantized.left_children[i];
        aos_node.right_child = bvh_nodes_quantized.right_children[i];
        
        // Unpack primitive data
        uint32_t packed_data = bvh_nodes_quantized.primitive_data[i];
        aos_node.is_leaf = (packed_data & 1) != 0;
        aos_node.primitive_count = (packed_data >> 16) & 0x7FFF;
        aos_node.primitive_start = (packed_data >> 1) & 0xFFFF;
    }
}

void CollisionDetection::convertQuantizedToSoA() {
    if (bvh_nodes_quantized.node_count == 0) return;
    
    // Initialize SoA data structures
    bvh_nodes_soa.node_count = bvh_nodes_quantized.node_count;
    bvh_nodes_soa.aabb_mins.resize(bvh_nodes_quantized.node_count);
    bvh_nodes_soa.aabb_maxs.resize(bvh_nodes_quantized.node_count);
    bvh_nodes_soa.left_children.resize(bvh_nodes_quantized.node_count);
    bvh_nodes_soa.right_children.resize(bvh_nodes_quantized.node_count);
    bvh_nodes_soa.primitive_starts.resize(bvh_nodes_quantized.node_count);
    bvh_nodes_soa.primitive_counts.resize(bvh_nodes_quantized.node_count);
    bvh_nodes_soa.is_leaf_flags.resize(bvh_nodes_quantized.node_count);
    
    // Convert each node from quantized to SoA
    for (size_t i = 0; i < bvh_nodes_quantized.node_count; ++i) {
        // Dequantize AABB
        bvh_nodes_quantized.getAABB(i, bvh_nodes_soa.aabb_mins[i], bvh_nodes_soa.aabb_maxs[i]);
        
        bvh_nodes_soa.left_children[i] = bvh_nodes_quantized.left_children[i];
        bvh_nodes_soa.right_children[i] = bvh_nodes_quantized.right_children[i];
        
        // Unpack primitive data
        uint32_t packed_data = bvh_nodes_quantized.primitive_data[i];
        bvh_nodes_soa.is_leaf_flags[i] = (packed_data & 1) != 0;
        bvh_nodes_soa.primitive_counts[i] = (packed_data >> 16) & 0x7FFF;
        bvh_nodes_soa.primitive_starts[i] = (packed_data >> 1) & 0xFFFF;
    }
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysOptimized(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats) {
    std::vector<HitResult> results;
    RayTracingStats local_stats;
    local_stats.total_rays_cast = ray_queries.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (bvh_optimization_mode == BVHOptimizationMode::SOA_UNCOMPRESSED) {
        results = castRaysSoA(ray_queries, local_stats);
    } else if (bvh_optimization_mode == BVHOptimizationMode::SOA_QUANTIZED) {
        results = castRaysQuantized(ray_queries, local_stats);
    } else {
        // Fall back to regular ray casting for LEGACY_AOS
        results = castRays(ray_queries, &local_stats);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (stats) {
        *stats = local_stats;
    }
    
    return results;
}

bool CollisionDetection::processRayStream(RayStream &ray_stream, RayTracingStats *stats) {
    if (ray_stream.packets.empty()) {
        return true; // No rays to process
    }
    
    RayTracingStats local_stats;
    local_stats.total_rays_cast = ray_stream.total_rays;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process each packet in the stream
    for (auto &packet : ray_stream.packets) {
        std::vector<RayQuery> queries = packet.toRayQueries();
        
        RayTracingStats packet_stats;
        std::vector<HitResult> packet_results = castRaysOptimized(queries, &packet_stats);
        
        // Store results back in the packet
        packet.results = packet_results;
        
        // Accumulate statistics
        local_stats.total_hits += packet_stats.total_hits;
        local_stats.bvh_nodes_visited += packet_stats.bvh_nodes_visited;
        local_stats.average_ray_distance += packet_stats.average_ray_distance * packet_stats.total_hits;
    }
    
    // Calculate average ray distance
    if (local_stats.total_hits > 0) {
        local_stats.average_ray_distance /= local_stats.total_hits;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (stats) {
        *stats = local_stats;
    }
    
    return true;
}

CollisionDetection::MemoryUsageStats CollisionDetection::getBVHMemoryUsage() const {
    MemoryUsageStats stats;
    
    // Calculate legacy memory usage
    stats.legacy_memory_bytes = bvh_nodes.size() * sizeof(BVHNode) + 
                               primitive_indices.size() * sizeof(uint);
    
    // Calculate SoA memory usage
    stats.soa_memory_bytes = bvh_nodes_soa.getMemoryUsage();
    
    // Calculate quantized memory usage
    stats.quantized_memory_bytes = bvh_nodes_quantized.getMemoryUsage();
    
    // Calculate reduction percentages
    if (stats.legacy_memory_bytes > 0) {
        stats.soa_reduction_percent = 
            ((float)(stats.legacy_memory_bytes - stats.soa_memory_bytes) / stats.legacy_memory_bytes) * 100.0f;
        stats.quantized_reduction_percent = 
            ((float)(stats.legacy_memory_bytes - stats.quantized_memory_bytes) / stats.legacy_memory_bytes) * 100.0f;
    }
    
    return stats;
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysSoA(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats) {
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());
    
    if (bvh_nodes_soa.node_count == 0) {
        // Return empty results if no SoA BVH
        results.resize(ray_queries.size());
        return results;
    }
    
    for (const auto &query : ray_queries) {
        HitResult result = castRaySoATraversal(query, stats);
        results.push_back(result);
        
        if (result.hit) {
            stats.total_hits++;
            stats.average_ray_distance += result.distance;
        }
    }
    
    if (stats.total_hits > 0) {
        stats.average_ray_distance /= stats.total_hits;
    }
    
    return results;
}

std::vector<CollisionDetection::HitResult> CollisionDetection::castRaysQuantized(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats) {
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());
    
    if (bvh_nodes_quantized.node_count == 0) {
        // Return empty results if no quantized BVH  
        results.resize(ray_queries.size());
        return results;
    }
    
    for (const auto &query : ray_queries) {
        HitResult result = castRayQuantizedTraversal(query, stats);
        results.push_back(result);
        
        if (result.hit) {
            stats.total_hits++;
            stats.average_ray_distance += result.distance;
        }
    }
    
    if (stats.total_hits > 0) {
        stats.average_ray_distance /= stats.total_hits;
    }
    
    return results;
}

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
    // Calculate inverse direction components (avoiding division in inner loops)
    float inv_dir_x = 1.0f / ray_direction.x;
    float inv_dir_y = 1.0f / ray_direction.y;
    float inv_dir_z = 1.0f / ray_direction.z;
    
    // Calculate intersection distances for each axis
    float t1_x = (aabb_min.x - ray_origin.x) * inv_dir_x;
    float t2_x = (aabb_max.x - ray_origin.x) * inv_dir_x;
    float t1_y = (aabb_min.y - ray_origin.y) * inv_dir_y;
    float t2_y = (aabb_max.y - ray_origin.y) * inv_dir_y;
    float t1_z = (aabb_min.z - ray_origin.z) * inv_dir_z;
    float t2_z = (aabb_max.z - ray_origin.z) * inv_dir_z;
    
    // Find min and max t values for each axis
    float tmin_x = std::min(t1_x, t2_x);
    float tmax_x = std::max(t1_x, t2_x);
    float tmin_y = std::min(t1_y, t2_y);
    float tmax_y = std::max(t1_y, t2_y);
    float tmin_z = std::min(t1_z, t2_z);
    float tmax_z = std::max(t1_z, t2_z);
    
    // Calculate final intersection interval
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
    // This method delegates to the existing primitive intersection logic
    // For now, we'll use the single-ray castRay method on individual primitives
    std::vector<uint> single_primitive = {primitive_id};
    RayQuery single_query(query.origin, query.direction, query.max_distance, single_primitive);
    return castRay(single_query);
}

// ================================================================
// AABB INTERSECTION UTILITIES
// ================================================================

bool CollisionDetection::rayAABBIntersect(const vec3 &origin, const vec3 &direction, const vec3 &aabb_min, const vec3 &aabb_max, float &t_min, float &t_max) {
    // Calculate inverse direction to avoid divisions in the loop
    float inv_dir_x = 1.0f / direction.x;
    float inv_dir_y = 1.0f / direction.y;
    float inv_dir_z = 1.0f / direction.z;
    
    // Calculate the intersection distances for each slab
    float t1_x = (aabb_min.x - origin.x) * inv_dir_x;
    float t2_x = (aabb_max.x - origin.x) * inv_dir_x;
    float t1_y = (aabb_min.y - origin.y) * inv_dir_y;
    float t2_y = (aabb_max.y - origin.y) * inv_dir_y;
    float t1_z = (aabb_min.z - origin.z) * inv_dir_z;
    float t2_z = (aabb_max.z - origin.z) * inv_dir_z;
    
    // Calculate the minimum and maximum t values for each axis
    float tmin_x = std::min(t1_x, t2_x);
    float tmax_x = std::max(t1_x, t2_x);
    float tmin_y = std::min(t1_y, t2_y);
    float tmax_y = std::max(t1_y, t2_y);
    float tmin_z = std::min(t1_z, t2_z);
    float tmax_z = std::max(t1_z, t2_z);
    
    // Find the intersection interval
    t_min = std::max({tmin_x, tmin_y, tmin_z});
    t_max = std::min({tmax_x, tmax_y, tmax_z});
    
    // Check if the ray intersects the AABB
    return t_max >= 0 && t_min <= t_max;
}

// ================================================================
// PRIMITIVE INTERSECTION METHODS
// ================================================================

bool CollisionDetection::rayPrimitiveIntersection(const vec3 &origin, const vec3 &direction, uint primitive_UUID, float &distance) const {
    // Get primitive type and vertices
    PrimitiveType type = context->getPrimitiveType(primitive_UUID);
    std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_UUID);
    
    if (vertices.empty()) {
        return false;
    }
    
    bool hit = false;
    float min_distance = std::numeric_limits<float>::max();
    
    if (type == PRIMITIVE_TYPE_TRIANGLE) {
        // Triangle intersection using radiation model algorithm (proven to work)
        if (vertices.size() >= 3) {
            const vec3 &v0 = vertices[0];
            const vec3 &v1 = vertices[1];
            const vec3 &v2 = vertices[2];
            
            // Use the same algorithm as radiation model's triangle_intersect
            float a = v0.x - v1.x, b = v0.x - v2.x, c = direction.x, d = v0.x - origin.x;
            float e = v0.y - v1.y, f = v0.y - v2.y, g = direction.y, h = v0.y - origin.y;
            float i = v0.z - v1.z, j = v0.z - v2.z, k = direction.z, l = v0.z - origin.z;

            float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
            float q = g * i - e * k, s = e * j - f * i;

            float denom = a * m + b * q + c * s;
            if (std::abs(denom) < 1e-8f) {
                return false; // Ray is parallel to triangle
            }
            
            float inv_denom = 1.0f / denom;

            float e1 = d * m - b * n - c * p;
            float beta = e1 * inv_denom;

            if (beta >= 0.0f) {
                float r = e * l - h * i;
                float e2 = a * n + d * q + c * r;
                float gamma = e2 * inv_denom;

                if (gamma >= 0.0f && beta + gamma <= 1.0f) {
                    float e3 = a * p - b * r + d * s;
                    float t = e3 * inv_denom;

                    if (t > 1e-8f && t < min_distance) {
                        min_distance = t;
                        hit = true;
                    }
                }
            }
        }
    } else if (type == PRIMITIVE_TYPE_PATCH) {
        // Patch (quadrilateral) intersection using radiation model algorithm
        if (vertices.size() >= 4) {
            const vec3 &v0 = vertices[0];
            const vec3 &v1 = vertices[1];
            const vec3 &v2 = vertices[2];
            const vec3 &v3 = vertices[3];
            
            // Calculate patch vectors and normal (same as radiation model)
            vec3 anchor = v0;
            vec3 normal = cross(v1 - v0, v2 - v0);
            normal.normalize();
            
            vec3 a = v1 - v0; // First edge vector
            vec3 b = v3 - v0; // Second edge vector
            
            // Ray-plane intersection
            float denom = direction * normal;
            if (std::abs(denom) > 1e-8f) { // Not parallel to plane
                float t = (anchor - origin) * normal / denom;
                
                if (t > 1e-8f && t < 1e8f) { // Valid intersection distance
                    // Find intersection point
                    vec3 p = origin + direction * t;
                    vec3 d = p - anchor;
                    
                    // Project onto patch coordinate system
                    float ddota = d * a;
                    float ddotb = d * b;
                    
                    // Check if point is within patch bounds
                    if (ddota >= 0.0f && ddota <= (a * a) && 
                        ddotb >= 0.0f && ddotb <= (b * b)) {
                        
                        if (t < min_distance) {
                            min_distance = t;
                            hit = true;
                        }
                    }
                }
            }
        }
    }
    
    if (hit) {
        distance = min_distance;
        return true;
    }
    
    return false;
}