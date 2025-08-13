/** \file "CollisionDetection.cu" CUDA source file for GPU-accelerated collision detection

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#include "helios_vector_types.h"

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

/**
 * \brief GPU-optimized SoA BVH structure for warp-efficient traversal
 *
 * This structure stores BVH data in Structure-of-Arrays layout for optimal
 * GPU memory coalescing and SIMD processing within warps.
 */
struct GPUBVHNodesSoA {
    // Hot data: frequently accessed during traversal (separate arrays for coalescing)
    float3 *aabb_mins;          //!< Array of AABB minimum corners
    float3 *aabb_maxs;          //!< Array of AABB maximum corners
    uint32_t *left_children;    //!< Array of left child indices
    uint32_t *right_children;   //!< Array of right child indices
    
    // Cold data: accessed less frequently
    uint32_t *primitive_starts; //!< Array of primitive starting indices
    uint32_t *primitive_counts; //!< Array of primitive counts
    uint8_t *is_leaf_flags;     //!< Array of leaf flags (packed)
    
    size_t node_count;          //!< Total number of nodes
};

/**
 * \brief GPU-optimized quantized BVH structure for memory efficiency
 *
 * Uses 16-bit quantization for 52% memory reduction while maintaining precision.
 */
struct GPUQuantizedBVHNodes {
    // Quantization parameters (constant for all nodes)
    float3 scene_min;           //!< Scene bounding box minimum
    float3 scene_max;           //!< Scene bounding box maximum
    float3 quantization_scale;  //!< Scale factor for quantization
    
    // Quantized AABB data (separate arrays for coalescing)
    uint16_t *aabb_mins_x;      //!< Quantized X coordinates (min corners)
    uint16_t *aabb_mins_y;      //!< Quantized Y coordinates (min corners)
    uint16_t *aabb_mins_z;      //!< Quantized Z coordinates (min corners)
    uint16_t *aabb_maxs_x;      //!< Quantized X coordinates (max corners)
    uint16_t *aabb_maxs_y;      //!< Quantized Y coordinates (max corners)
    uint16_t *aabb_maxs_z;      //!< Quantized Z coordinates (max corners)
    
    // Navigation data
    uint32_t *left_children;    //!< Array of left child indices
    uint32_t *right_children;   //!< Array of right child indices
    uint32_t *primitive_data;   //!< Packed primitive start/count data
    
    size_t node_count;          //!< Total number of nodes
};

/**
 * \brief Ray streaming packet for coherent GPU processing
 */
struct RayPacket {
    float3 origins[32];         //!< Ray origins (warp-sized)
    float3 directions[32];      //!< Ray directions (normalized)
    float max_distances[32];    //!< Maximum ray distances
    uint32_t active_mask;       //!< Bitmask of active rays in packet
    uint32_t packet_id;         //!< Unique packet identifier
};

/**
 * \brief Shared memory cache for BVH nodes (per thread block)
 */
__shared__ extern float shared_bvh_cache[];

// Constants for optimizations
#define WARP_SIZE 32
#define MAX_SHARED_NODES 256
#define SHARED_CACHE_SIZE (MAX_SHARED_NODES * 8 * sizeof(float))  // 8 floats per cached node

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
 * \brief Warp-efficient ray-AABB intersection for GPU optimization
 * \param[in] ray_origin Ray starting point
 * \param[in] ray_dir Ray direction vector (normalized)
 * \param[in] aabb_min AABB minimum corner
 * \param[in] aabb_max AABB maximum corner
 * \param[in] max_dist Maximum ray distance
 * \return True if ray intersects AABB within max_dist
 */
__device__ __forceinline__ bool warpRayAABBIntersect(const float3 &ray_origin, const float3 &ray_dir, 
                                                     const float3 &aabb_min, const float3 &aabb_max, 
                                                     float max_dist) {
    // Optimized ray-AABB intersection using slab method
    // Compute intersection distances for each axis
    float3 inv_dir = make_float3(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);
    
    float3 t_min = make_float3(
        (aabb_min.x - ray_origin.x) * inv_dir.x,
        (aabb_min.y - ray_origin.y) * inv_dir.y,
        (aabb_min.z - ray_origin.z) * inv_dir.z
    );
    
    float3 t_max = make_float3(
        (aabb_max.x - ray_origin.x) * inv_dir.x,
        (aabb_max.y - ray_origin.y) * inv_dir.y,
        (aabb_max.z - ray_origin.z) * inv_dir.z
    );
    
    // Handle negative ray directions
    if (ray_dir.x < 0.0f) { float temp = t_min.x; t_min.x = t_max.x; t_max.x = temp; }
    if (ray_dir.y < 0.0f) { float temp = t_min.y; t_min.y = t_max.y; t_max.y = temp; }
    if (ray_dir.z < 0.0f) { float temp = t_min.z; t_min.z = t_max.z; t_max.z = temp; }
    
    // Find the intersection interval
    float t_enter = fmaxf(fmaxf(t_min.x, t_min.y), t_min.z);
    float t_exit = fminf(fminf(t_max.x, t_max.y), t_max.z);
    
    // Check if intersection exists and is within ray limits
    return (t_enter <= t_exit) && (t_exit >= 0.0f) && (t_enter <= max_dist);
}

/**
 * \brief CUDA kernel for BVH traversal collision detection
 *
 * Each thread processes one query AABB and traverses the BVH to find collisions.
 *
 * \param[in] d_nodes Array of BVH nodes on GPU
 * \param[in] d_primitive_indices Array of primitive indices on GPU
 * \param[in] d_query_aabb_min Array of query AABB minimum corners
 * \param[in] d_query_aabb_max Array of query AABB maximum corners
 * \param[out] d_results Array to store collision results
 * \param[out] d_result_counts Array to store number of results per query
 * \param[in] num_queries Number of queries to process
 * \param[in] max_results_per_query Maximum results to store per query
 */
__global__ void bvhTraversalKernel(GPUBVHNode *d_nodes, unsigned int *d_primitive_indices, float3 *d_primitive_aabb_min, float3 *d_primitive_aabb_max, float3 *d_query_aabb_min, float3 *d_query_aabb_max, unsigned int *d_results,
                                   unsigned int *d_result_counts, int num_queries, int max_results_per_query) {

    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (query_idx >= num_queries)
        return;

    float3 query_min = d_query_aabb_min[query_idx];
    float3 query_max = d_query_aabb_max[query_idx];

    unsigned int result_count = 0;
    unsigned int *query_results = &d_results[query_idx * max_results_per_query];

    // Stack-based traversal using shared memory for better performance
    __shared__ unsigned int node_stack[1024]; // Shared among threads in block
    int stack_size = 0;

    // Each thread gets its own portion of the shared stack
    int thread_stack_start = threadIdx.x * 32; // 32 entries per thread
    unsigned int *thread_stack = &node_stack[thread_stack_start];

    // Start traversal from root node
    thread_stack[0] = 0;
    stack_size = 1;

    while (stack_size > 0 && result_count < max_results_per_query) {

        // Pop node from stack
        stack_size--;
        unsigned int node_idx = thread_stack[stack_size];

        // Check if node index is valid
        if (node_idx == 0xFFFFFFFF)
            continue;

        GPUBVHNode node = d_nodes[node_idx];

        // Test if query AABB intersects node AABB
        if (!d_aabbIntersect(query_min, query_max, node.aabb_min, node.aabb_max)) {
            continue;
        }

        if (node.is_leaf) {
            // Check each primitive in this leaf individually
            for (unsigned int i = 0; i < node.primitive_count && result_count < max_results_per_query; i++) {
                unsigned int primitive_index = node.primitive_start + i;
                unsigned int primitive_id = d_primitive_indices[primitive_index];

                // Get primitive's AABB from pre-computed arrays (using array position, not UUID)
                float3 prim_min = d_primitive_aabb_min[primitive_index];
                float3 prim_max = d_primitive_aabb_max[primitive_index];

                // Only add to results if AABBs actually intersect
                if (d_aabbIntersect(query_min, query_max, prim_min, prim_max)) {
                    query_results[result_count] = primitive_id;
                    result_count++;
                }
            }
        } else {
            // Add child nodes to stack
            if (node.left_child != 0xFFFFFFFF && stack_size < 32) {
                thread_stack[stack_size] = node.left_child;
                stack_size++;
            }
            if (node.right_child != 0xFFFFFFFF && stack_size < 32) {
                thread_stack[stack_size] = node.right_child;
                stack_size++;
            }
        }
    }

    d_result_counts[query_idx] = result_count;
}

/**
 * \brief Warp-efficient BVH traversal kernel for high-performance ray tracing
 * 
 * This kernel processes rays in warp-coherent packets, uses shared memory caching,
 * and implements cooperative fetching for optimal GPU utilization.
 * 
 * \param[in] bvh_soa GPU-optimized Structure-of-Arrays BVH data
 * \param[in] d_primitive_indices Array of primitive indices
 * \param[in] primitive_count Total number of primitives
 * \param[in] d_primitive_aabb_min Array of primitive AABB minimums
 * \param[in] d_primitive_aabb_max Array of primitive AABB maximums
 * \param[in] d_ray_origins Array of ray origins
 * \param[in] d_ray_directions Array of ray directions
 * \param[in] d_ray_max_distances Array of ray maximum distances
 * \param[in] num_rays Total number of rays to process
 * \param[out] d_results Array to store collision results
 * \param[out] d_result_counts Array to store result counts per ray
 * \param[in] max_results_per_ray Maximum results to store per ray
 */
__global__ void warpEfficientBVHKernel(
    GPUBVHNodesSoA bvh_soa,
    unsigned int *d_primitive_indices,
    int primitive_count,
    float3 *d_primitive_aabb_min,
    float3 *d_primitive_aabb_max,
    float3 *d_ray_origins,
    float3 *d_ray_directions,
    float *d_ray_max_distances,
    int num_rays,
    unsigned int *d_results,
    unsigned int *d_result_counts,
    int max_results_per_ray
) {
    // Global thread and warp identifiers
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Each warp processes one ray packet
    if (warp_id >= (num_rays + WARP_SIZE - 1) / WARP_SIZE) {
        return;
    }
    
    // Calculate ray index for this thread
    int ray_idx = warp_id * WARP_SIZE + lane_id;
    
    // Shared memory for BVH node caching (per thread block)
    __shared__ float3 shared_aabb_mins[MAX_SHARED_NODES];
    __shared__ float3 shared_aabb_maxs[MAX_SHARED_NODES];
    __shared__ uint32_t shared_left_children[MAX_SHARED_NODES];
    __shared__ uint32_t shared_right_children[MAX_SHARED_NODES];
    __shared__ uint32_t shared_primitive_starts[MAX_SHARED_NODES];
    __shared__ uint32_t shared_primitive_counts[MAX_SHARED_NODES];
    __shared__ uint8_t shared_is_leaf_flags[MAX_SHARED_NODES];
    
    // Initialize shared memory (cooperative loading)
    for (int i = threadIdx.x; i < MAX_SHARED_NODES && i < bvh_soa.node_count; i += blockDim.x) {
        shared_aabb_mins[i] = bvh_soa.aabb_mins[i];
        shared_aabb_maxs[i] = bvh_soa.aabb_maxs[i];
        shared_left_children[i] = bvh_soa.left_children[i];
        shared_right_children[i] = bvh_soa.right_children[i];
        shared_primitive_starts[i] = bvh_soa.primitive_starts[i];
        shared_primitive_counts[i] = bvh_soa.primitive_counts[i];
        shared_is_leaf_flags[i] = bvh_soa.is_leaf_flags[i];
    }
    
    __syncthreads();
    
    // Process ray if within bounds
    if (ray_idx >= num_rays) {
        return;
    }
    
    // Load ray data
    float3 ray_origin = d_ray_origins[ray_idx];
    float3 ray_direction = d_ray_directions[ray_idx];
    float ray_max_distance = d_ray_max_distances[ray_idx];
    
    unsigned int result_count = 0;
    unsigned int *ray_results = &d_results[ray_idx * max_results_per_ray];
    
    // Warp-coherent traversal stack (using local memory)
    unsigned int traversal_stack[64];
    int stack_size = 0;
    
    // Start traversal from root
    traversal_stack[0] = 0;
    stack_size = 1;
    
    // Main traversal loop
    while (stack_size > 0 && result_count < max_results_per_ray) {
        // Pop node from stack
        stack_size--;
        unsigned int node_idx = traversal_stack[stack_size];
        
        if (node_idx >= bvh_soa.node_count) {
            continue;
        }
        
        // Use shared memory cache if node is cached, otherwise access global memory
        float3 node_aabb_min, node_aabb_max;
        uint32_t left_child, right_child, primitive_start, primitive_count;
        uint8_t is_leaf;
        
        if (node_idx < MAX_SHARED_NODES) {
            // Node is in shared cache
            node_aabb_min = shared_aabb_mins[node_idx];
            node_aabb_max = shared_aabb_maxs[node_idx];
            left_child = shared_left_children[node_idx];
            right_child = shared_right_children[node_idx];
            primitive_start = shared_primitive_starts[node_idx];
            primitive_count = shared_primitive_counts[node_idx];
            is_leaf = shared_is_leaf_flags[node_idx];
        } else {
            // Access global memory with coalesced reads
            node_aabb_min = bvh_soa.aabb_mins[node_idx];
            node_aabb_max = bvh_soa.aabb_maxs[node_idx];
            left_child = bvh_soa.left_children[node_idx];
            right_child = bvh_soa.right_children[node_idx];
            primitive_start = bvh_soa.primitive_starts[node_idx];
            primitive_count = bvh_soa.primitive_counts[node_idx];
            is_leaf = bvh_soa.is_leaf_flags[node_idx];
        }
        
        // Test ray-AABB intersection
        if (!warpRayAABBIntersect(ray_origin, ray_direction, node_aabb_min, node_aabb_max, ray_max_distance)) {
            continue;
        }
        
        if (is_leaf) {
            // Process primitives in leaf node
            for (unsigned int i = 0; i < primitive_count && result_count < max_results_per_ray; i++) {
                unsigned int primitive_index = primitive_start + i;
                if (primitive_index >= primitive_count) break;
                
                unsigned int primitive_id = d_primitive_indices[primitive_index];
                
                // Test ray against primitive AABB
                float3 prim_min = d_primitive_aabb_min[primitive_index];
                float3 prim_max = d_primitive_aabb_max[primitive_index];
                
                if (warpRayAABBIntersect(ray_origin, ray_direction, prim_min, prim_max, ray_max_distance)) {
                    ray_results[result_count] = primitive_id;
                    result_count++;
                }
            }
        } else {
            // Add children to traversal stack (prioritize closer child for coherence)
            if (right_child != 0xFFFFFFFF && stack_size < 63) {
                traversal_stack[stack_size] = right_child;
                stack_size++;
            }
            if (left_child != 0xFFFFFFFF && stack_size < 63) {
                traversal_stack[stack_size] = left_child;
                stack_size++;
            }
        }
    }
    
    // Store result count
    d_result_counts[ray_idx] = result_count;
}

/**
 * \brief CUDA kernel to calculate ray path lengths through a regular voxel grid
 * 
 * This kernel computes the intersection of rays with voxels and accumulates
 * path length statistics for integration with aeriallidar and lidar plugins.
 * 
 * \param[in] num_rays Number of rays to process
 * \param[in] d_ray_origins Array of ray origin points
 * \param[in] d_ray_directions Array of ray direction vectors (normalized)
 * \param[in] grid_center Center of the voxel grid
 * \param[in] grid_size Total size of the voxel grid
 * \param[in] grid_divisions Number of divisions in x, y, z
 * \param[out] d_voxel_ray_counts Flattened array to accumulate ray counts per voxel
 * \param[out] d_voxel_path_lengths Flattened array to accumulate path lengths per voxel
 * \param[out] d_voxel_transmitted Flattened array to count transmitted rays per voxel
 */
__global__ void intersectRegularGridKernel(
    const size_t num_rays,
    float3 *d_ray_origins,
    float3 *d_ray_directions,
    float3 grid_center,
    float3 grid_size,
    int3 grid_divisions,
    int *d_voxel_ray_counts,
    float *d_voxel_path_lengths,
    int *d_voxel_transmitted
) {
    
    size_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= num_rays) {
        return;
    }
    
    float3 ray_origin = d_ray_origins[ray_idx];
    float3 ray_direction = d_ray_directions[ray_idx];
    
    // Calculate voxel size
    float3 voxel_size = make_float3(
        grid_size.x / static_cast<float>(grid_divisions.x),
        grid_size.y / static_cast<float>(grid_divisions.y),
        grid_size.z / static_cast<float>(grid_divisions.z)
    );
    
    float3 grid_min = make_float3(
        grid_center.x - 0.5f * grid_size.x,
        grid_center.y - 0.5f * grid_size.y,
        grid_center.z - 0.5f * grid_size.z
    );
    
    // Test intersection with each voxel in the grid
    for (int i = 0; i < grid_divisions.x; i++) {
        for (int j = 0; j < grid_divisions.y; j++) {
            for (int k = 0; k < grid_divisions.z; k++) {
                
                // Calculate voxel AABB
                float3 voxel_min = make_float3(
                    grid_min.x + i * voxel_size.x,
                    grid_min.y + j * voxel_size.y,
                    grid_min.z + k * voxel_size.z
                );
                
                float3 voxel_max = make_float3(
                    voxel_min.x + voxel_size.x,
                    voxel_min.y + voxel_size.y,
                    voxel_min.z + voxel_size.z
                );
                
                // Ray-AABB intersection test
                float t_min_x, t_max_x, t_min_y, t_max_y, t_min_z, t_max_z;
                
                // X slab
                float inv_dir_x = 1.0f / ray_direction.x;
                if (inv_dir_x >= 0) {
                    t_min_x = (voxel_min.x - ray_origin.x) * inv_dir_x;
                    t_max_x = (voxel_max.x - ray_origin.x) * inv_dir_x;
                } else {
                    t_min_x = (voxel_max.x - ray_origin.x) * inv_dir_x;
                    t_max_x = (voxel_min.x - ray_origin.x) * inv_dir_x;
                }
                
                // Y slab
                float inv_dir_y = 1.0f / ray_direction.y;
                if (inv_dir_y >= 0) {
                    t_min_y = (voxel_min.y - ray_origin.y) * inv_dir_y;
                    t_max_y = (voxel_max.y - ray_origin.y) * inv_dir_y;
                } else {
                    t_min_y = (voxel_max.y - ray_origin.y) * inv_dir_y;
                    t_max_y = (voxel_min.y - ray_origin.y) * inv_dir_y;
                }
                
                // Z slab
                float inv_dir_z = 1.0f / ray_direction.z;
                if (inv_dir_z >= 0) {
                    t_min_z = (voxel_min.z - ray_origin.z) * inv_dir_z;
                    t_max_z = (voxel_max.z - ray_origin.z) * inv_dir_z;
                } else {
                    t_min_z = (voxel_max.z - ray_origin.z) * inv_dir_z;
                    t_max_z = (voxel_min.z - ray_origin.z) * inv_dir_z;
                }
                
                // Find intersection parameters
                float t_enter = fmaxf(fmaxf(t_min_x, t_min_y), t_min_z);
                float t_exit = fminf(fminf(t_max_x, t_max_y), t_max_z);
                
                // Check if ray intersects voxel
                if (t_enter < t_exit && t_exit > 1e-6f) {
                    
                    // Calculate path length through voxel
                    float path_length = t_exit - t_enter;
                    
                    // Handle case where ray starts inside voxel
                    if (t_enter < 0) {
                        path_length = t_exit;
                    }
                    
                    // Calculate flattened voxel index
                    int voxel_idx = i * grid_divisions.y * grid_divisions.z + j * grid_divisions.z + k;
                    
                    // Accumulate statistics using atomic operations
                    atomicAdd(&d_voxel_ray_counts[voxel_idx], 1);
                    atomicAdd(&d_voxel_path_lengths[voxel_idx], path_length);
                    atomicAdd(&d_voxel_transmitted[voxel_idx], 1);
                }
            }
        }
    }
}

// C-style wrapper functions for calling from C++ code

extern "C" {

/**
 * \brief Launch BVH traversal kernel from C++ code
 *
 * \param[in] h_nodes Host array of BVH nodes
 * \param[in] node_count Number of BVH nodes
 * \param[in] h_primitive_indices Host array of primitive indices
 * \param[in] primitive_count Number of primitive indices
 * \param[in] h_query_aabb_min Host array of query AABB minimum corners
 * \param[in] h_query_aabb_max Host array of query AABB maximum corners
 * \param[in] num_queries Number of queries
 * \param[out] h_results Host array for results
 * \param[out] h_result_counts Host array for result counts
 * \param[in] max_results_per_query Maximum results per query
 */
void launchBVHTraversal(void *h_nodes, int node_count, unsigned int *h_primitive_indices, int primitive_count, float *h_primitive_aabb_min, float *h_primitive_aabb_max, float *h_query_aabb_min, float *h_query_aabb_max, int num_queries,
                        unsigned int *h_results, unsigned int *h_result_counts, int max_results_per_query) {

    if (num_queries == 0)
        return;

    // Allocate temporary GPU memory for query data and primitive AABBs
    float3 *d_query_min;
    float3 *d_query_max;
    float3 *d_primitive_min;
    float3 *d_primitive_max;
    unsigned int *d_results;
    unsigned int *d_result_counts;

    size_t query_size = num_queries * sizeof(float3);
    size_t primitive_aabb_size = primitive_count * sizeof(float3);
    size_t results_size = num_queries * max_results_per_query * sizeof(unsigned int);
    size_t counts_size = num_queries * sizeof(unsigned int);

    cudaMalloc((void **) &d_query_min, query_size);
    cudaMalloc((void **) &d_query_max, query_size);
    cudaMalloc((void **) &d_primitive_min, primitive_aabb_size);
    cudaMalloc((void **) &d_primitive_max, primitive_aabb_size);
    cudaMalloc((void **) &d_results, results_size);
    cudaMalloc((void **) &d_result_counts, counts_size);

    // Convert query data to float3 format
    std::vector<float3> query_min_vec(num_queries);
    std::vector<float3> query_max_vec(num_queries);
    for (int i = 0; i < num_queries; i++) {
        query_min_vec[i] = make_float3(h_query_aabb_min[i * 3], h_query_aabb_min[i * 3 + 1], h_query_aabb_min[i * 3 + 2]);
        query_max_vec[i] = make_float3(h_query_aabb_max[i * 3], h_query_aabb_max[i * 3 + 1], h_query_aabb_max[i * 3 + 2]);
    }

    // Convert primitive AABB data to float3 format
    std::vector<float3> primitive_min_vec(primitive_count);
    std::vector<float3> primitive_max_vec(primitive_count);
    for (int i = 0; i < primitive_count; i++) {
        primitive_min_vec[i] = make_float3(h_primitive_aabb_min[i * 3], h_primitive_aabb_min[i * 3 + 1], h_primitive_aabb_min[i * 3 + 2]);
        primitive_max_vec[i] = make_float3(h_primitive_aabb_max[i * 3], h_primitive_aabb_max[i * 3 + 1], h_primitive_aabb_max[i * 3 + 2]);
    }

    // Copy query and primitive AABB data to GPU
    cudaMemcpy(d_query_min, query_min_vec.data(), query_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_max, query_max_vec.data(), query_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primitive_min, primitive_min_vec.data(), primitive_aabb_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primitive_max, primitive_max_vec.data(), primitive_aabb_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_queries + block_size - 1) / block_size;

    bvhTraversalKernel<<<num_blocks, block_size>>>((GPUBVHNode *) h_nodes, (unsigned int *) h_primitive_indices, d_primitive_min, d_primitive_max, d_query_min, d_query_max, d_results, d_result_counts, num_queries, max_results_per_query);

    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Copy results back
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_counts, d_result_counts, counts_size, cudaMemcpyDeviceToHost);

    // Clean up temporary memory
    cudaFree(d_query_min);
    cudaFree(d_query_max);
    cudaFree(d_primitive_min);
    cudaFree(d_primitive_max);
    cudaFree(d_results);
    cudaFree(d_result_counts);
}

/**
 * \brief Launch cone intersection kernel from C++ code
 *
 * \param[in] h_nodes Host array of BVH nodes
 * \param[in] node_count Number of BVH nodes
 * \param[in] h_primitive_indices Host array of primitive indices
 * \param[in] primitive_count Number of primitive indices
 * \param[in] h_cone_origins Host array of cone origins
 * \param[in] h_cone_directions Host array of cone directions
 * \param[in] h_cone_angles Host array of cone angles
 * \param[in] h_max_distances Host array of maximum distances
 * \param[in] num_queries Number of queries
 * \param[out] h_results Host array for results
 * \param[out] h_result_counts Host array for result counts
 * \param[in] max_results_per_query Maximum results per query
 */
void launchConeIntersection(void *h_nodes, int node_count, unsigned int *h_primitive_indices, int primitive_count, float *h_cone_origins, float *h_cone_directions, float *h_cone_angles, float *h_max_distances, int num_queries,
                            unsigned int *h_results, unsigned int *h_result_counts, int max_results_per_query) {

    // Stub implementation
    for (int i = 0; i < num_queries; i++) {
        h_result_counts[i] = 0;
    }
}

/**
 * \brief Launch CUDA kernel for regular grid voxel ray path length calculation
 */
void launchVoxelRayPathLengths(
    int num_rays,
    float *h_ray_origins,
    float *h_ray_directions,
    float grid_center_x, float grid_center_y, float grid_center_z,
    float grid_size_x, float grid_size_y, float grid_size_z,
    int grid_divisions_x, int grid_divisions_y, int grid_divisions_z,
    int *h_voxel_ray_counts,
    float *h_voxel_path_lengths,
    int *h_voxel_transmitted
) {
    
    // Allocate device memory
    float3 *d_ray_origins, *d_ray_directions;
    int *d_voxel_ray_counts, *d_voxel_transmitted;
    float *d_voxel_path_lengths;
    
    size_t ray_data_size = num_rays * 3 * sizeof(float);
    size_t voxel_count = grid_divisions_x * grid_divisions_y * grid_divisions_z;
    size_t voxel_int_size = voxel_count * sizeof(int);
    size_t voxel_float_size = voxel_count * sizeof(float);
    
    // Allocate memory
    cudaMalloc(&d_ray_origins, ray_data_size);
    cudaMalloc(&d_ray_directions, ray_data_size);
    cudaMalloc(&d_voxel_ray_counts, voxel_int_size);
    cudaMalloc(&d_voxel_transmitted, voxel_int_size);
    cudaMalloc(&d_voxel_path_lengths, voxel_float_size);
    
    // Copy input data to device
    cudaMemcpy(d_ray_origins, h_ray_origins, ray_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_directions, h_ray_directions, ray_data_size, cudaMemcpyHostToDevice);
    cudaMemset(d_voxel_ray_counts, 0, voxel_int_size);
    cudaMemset(d_voxel_transmitted, 0, voxel_int_size);
    cudaMemset(d_voxel_path_lengths, 0, voxel_float_size);
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((num_rays + block_size.x - 1) / block_size.x);
    
    float3 grid_center = make_float3(grid_center_x, grid_center_y, grid_center_z);
    float3 grid_size_vec = make_float3(grid_size_x, grid_size_y, grid_size_z);
    int3 grid_divisions_vec = make_int3(grid_divisions_x, grid_divisions_y, grid_divisions_z);
    
    intersectRegularGridKernel<<<grid_size, block_size>>>(
        num_rays,
        d_ray_origins,
        d_ray_directions,
        grid_center,
        grid_size_vec,
        grid_divisions_vec,
        d_voxel_ray_counts,
        d_voxel_path_lengths,
        d_voxel_transmitted
    );
    
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA voxel kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy results back to host
    cudaMemcpy(h_voxel_ray_counts, d_voxel_ray_counts, voxel_int_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxel_path_lengths, d_voxel_path_lengths, voxel_float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxel_transmitted, d_voxel_transmitted, voxel_int_size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_ray_origins);
    cudaFree(d_ray_directions);
    cudaFree(d_voxel_ray_counts);
    cudaFree(d_voxel_transmitted);
    cudaFree(d_voxel_path_lengths);
}

/**
 * \brief Launch warp-efficient BVH traversal kernel with SoA optimizations
 * 
 * This function launches the optimized kernel that uses warp-coherent
 * processing, shared memory caching, and Structure-of-Arrays data layout.
 * 
 * \param[in] h_bvh_soa_gpu Host pointer to GPU-allocated SoA BVH structure
 * \param[in] h_primitive_indices Host array of primitive indices
 * \param[in] primitive_count Number of primitives
 * \param[in] h_primitive_aabb_min Host array of primitive AABB minimums
 * \param[in] h_primitive_aabb_max Host array of primitive AABB maximums
 * \param[in] h_ray_origins Host array of ray origins
 * \param[in] h_ray_directions Host array of ray directions
 * \param[in] h_ray_max_distances Host array of ray maximum distances
 * \param[in] num_rays Number of rays to process
 * \param[out] h_results Host array for results
 * \param[out] h_result_counts Host array for result counts
 * \param[in] max_results_per_ray Maximum results per ray
 */
void launchWarpEfficientBVH(
    void *h_bvh_soa_gpu,
    unsigned int *h_primitive_indices,
    int primitive_count,
    float *h_primitive_aabb_min,
    float *h_primitive_aabb_max,
    float *h_ray_origins,
    float *h_ray_directions,
    float *h_ray_max_distances,
    int num_rays,
    unsigned int *h_results,
    unsigned int *h_result_counts,
    int max_results_per_ray
) {
    if (num_rays == 0) {
        return;
    }
    
    // Cast to proper SoA structure
    GPUBVHNodesSoA *h_bvh_soa = static_cast<GPUBVHNodesSoA*>(h_bvh_soa_gpu);
    if (!h_bvh_soa || h_bvh_soa->node_count == 0) {
        fprintf(stderr, "Invalid BVH SoA structure passed to GPU kernel\n");
        return;
    }
    
    // Declare all variables at the beginning to avoid goto issues
    float3 *d_bvh_aabb_mins = nullptr, *d_bvh_aabb_maxs = nullptr;
    uint32_t *d_bvh_left_children = nullptr, *d_bvh_right_children = nullptr;
    uint32_t *d_bvh_primitive_starts = nullptr, *d_bvh_primitive_counts = nullptr;
    uint8_t *d_bvh_is_leaf_flags = nullptr;
    float3 *d_ray_origins = nullptr, *d_ray_directions = nullptr;
    float *d_ray_max_distances = nullptr;
    float3 *d_primitive_aabb_min = nullptr, *d_primitive_aabb_max = nullptr;
    unsigned int *d_results = nullptr, *d_result_counts = nullptr;
    
    size_t bvh_node_count = h_bvh_soa->node_count;
    size_t bvh_float3_size = bvh_node_count * sizeof(float3);
    size_t bvh_uint32_size = bvh_node_count * sizeof(uint32_t);
    size_t bvh_uint8_size = bvh_node_count * sizeof(uint8_t);
    
    // Allocate GPU memory for BVH data
    cudaError_t err;
    err = cudaMalloc(&d_bvh_aabb_mins, bvh_float3_size);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA malloc error for BVH mins: %s\n", cudaGetErrorString(err)); 
        return; 
    }
    
    err = cudaMalloc(&d_bvh_aabb_maxs, bvh_float3_size);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA malloc error for BVH maxs: %s\n", cudaGetErrorString(err)); 
        cudaFree(d_bvh_aabb_mins); 
        return; 
    }
    
    err = cudaMalloc(&d_bvh_left_children, bvh_uint32_size);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA malloc error for BVH left: %s\n", cudaGetErrorString(err)); 
        cudaFree(d_bvh_aabb_mins); 
        cudaFree(d_bvh_aabb_maxs); 
        return; 
    }
    
    err = cudaMalloc(&d_bvh_right_children, bvh_uint32_size);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA malloc error for BVH right: %s\n", cudaGetErrorString(err)); 
        cudaFree(d_bvh_aabb_mins); 
        cudaFree(d_bvh_aabb_maxs); 
        cudaFree(d_bvh_left_children); 
        return; 
    }
    
    err = cudaMalloc(&d_bvh_primitive_starts, bvh_uint32_size);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA malloc error for BVH starts: %s\n", cudaGetErrorString(err)); 
        cudaFree(d_bvh_aabb_mins); 
        cudaFree(d_bvh_aabb_maxs); 
        cudaFree(d_bvh_left_children); 
        cudaFree(d_bvh_right_children); 
        return; 
    }
    
    err = cudaMalloc(&d_bvh_primitive_counts, bvh_uint32_size);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA malloc error for BVH counts: %s\n", cudaGetErrorString(err)); 
        cudaFree(d_bvh_aabb_mins); 
        cudaFree(d_bvh_aabb_maxs); 
        cudaFree(d_bvh_left_children); 
        cudaFree(d_bvh_right_children); 
        cudaFree(d_bvh_primitive_starts); 
        return; 
    }
    
    err = cudaMalloc(&d_bvh_is_leaf_flags, bvh_uint8_size);
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA malloc error for BVH flags: %s\n", cudaGetErrorString(err)); 
        cudaFree(d_bvh_aabb_mins); 
        cudaFree(d_bvh_aabb_maxs); 
        cudaFree(d_bvh_left_children); 
        cudaFree(d_bvh_right_children); 
        cudaFree(d_bvh_primitive_starts); 
        cudaFree(d_bvh_primitive_counts); 
        return; 
    }
    
    // Copy BVH data to GPU
    cudaMemcpy(d_bvh_aabb_mins, h_bvh_soa->aabb_mins, bvh_float3_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_aabb_maxs, h_bvh_soa->aabb_maxs, bvh_float3_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_left_children, h_bvh_soa->left_children, bvh_uint32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_right_children, h_bvh_soa->right_children, bvh_uint32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_primitive_starts, h_bvh_soa->primitive_starts, bvh_uint32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_primitive_counts, h_bvh_soa->primitive_counts, bvh_uint32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bvh_is_leaf_flags, h_bvh_soa->is_leaf_flags, bvh_uint8_size, cudaMemcpyHostToDevice);
    
    // Create GPU BVH structure with device pointers
    GPUBVHNodesSoA d_bvh_soa;
    d_bvh_soa.aabb_mins = d_bvh_aabb_mins;
    d_bvh_soa.aabb_maxs = d_bvh_aabb_maxs;
    d_bvh_soa.left_children = d_bvh_left_children;
    d_bvh_soa.right_children = d_bvh_right_children;
    d_bvh_soa.primitive_starts = d_bvh_primitive_starts;
    d_bvh_soa.primitive_counts = d_bvh_primitive_counts;
    d_bvh_soa.is_leaf_flags = d_bvh_is_leaf_flags;
    d_bvh_soa.node_count = bvh_node_count;
    
    // Calculate sizes for ray data
    size_t ray_data_size = num_rays * sizeof(float3);
    size_t ray_distances_size = num_rays * sizeof(float);
    size_t primitive_aabb_size = primitive_count * sizeof(float3);
    size_t results_size = num_rays * max_results_per_ray * sizeof(unsigned int);
    size_t counts_size = num_rays * sizeof(unsigned int);
    
    // Allocate GPU memory for ray and primitive data
    cudaMalloc(&d_ray_origins, ray_data_size);
    cudaMalloc(&d_ray_directions, ray_data_size);
    cudaMalloc(&d_ray_max_distances, ray_distances_size);
    cudaMalloc(&d_primitive_aabb_min, primitive_aabb_size);
    cudaMalloc(&d_primitive_aabb_max, primitive_aabb_size);
    cudaMalloc(&d_results, results_size);
    cudaMalloc(&d_result_counts, counts_size);
    
    // Convert and copy ray data to GPU
    std::vector<float3> ray_origins_vec(num_rays);
    std::vector<float3> ray_directions_vec(num_rays);
    for (int i = 0; i < num_rays; i++) {
        ray_origins_vec[i] = make_float3(h_ray_origins[i * 3], h_ray_origins[i * 3 + 1], h_ray_origins[i * 3 + 2]);
        ray_directions_vec[i] = make_float3(h_ray_directions[i * 3], h_ray_directions[i * 3 + 1], h_ray_directions[i * 3 + 2]);
    }
    
    // Convert primitive AABB data
    std::vector<float3> primitive_min_vec(primitive_count);
    std::vector<float3> primitive_max_vec(primitive_count);
    for (int i = 0; i < primitive_count; i++) {
        primitive_min_vec[i] = make_float3(h_primitive_aabb_min[i * 3], h_primitive_aabb_min[i * 3 + 1], h_primitive_aabb_min[i * 3 + 2]);
        primitive_max_vec[i] = make_float3(h_primitive_aabb_max[i * 3], h_primitive_aabb_max[i * 3 + 1], h_primitive_aabb_max[i * 3 + 2]);
    }
    
    // Copy data to GPU
    cudaMemcpy(d_ray_origins, ray_origins_vec.data(), ray_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_directions, ray_directions_vec.data(), ray_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_max_distances, h_ray_max_distances, ray_distances_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primitive_aabb_min, primitive_min_vec.data(), primitive_aabb_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primitive_aabb_max, primitive_max_vec.data(), primitive_aabb_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters for warp efficiency
    int threads_per_block = 256;  // Multiple of warp size (32)
    int num_blocks = (num_rays + threads_per_block - 1) / threads_per_block;
    
    // Launch warp-efficient kernel with shared memory
    size_t shared_memory_size = SHARED_CACHE_SIZE;
    
    warpEfficientBVHKernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
        d_bvh_soa,  // Now passing GPU pointers, not host pointers
        h_primitive_indices,
        primitive_count,
        d_primitive_aabb_min,
        d_primitive_aabb_max,
        d_ray_origins,
        d_ray_directions,
        d_ray_max_distances,
        num_rays,
        d_results,
        d_result_counts,
        max_results_per_ray
    );
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Warp-efficient kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy results back to host
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_counts, d_result_counts, counts_size, cudaMemcpyDeviceToHost);
    
    // Clean up all device memory
    cudaFree(d_ray_origins);
    cudaFree(d_ray_directions);
    cudaFree(d_ray_max_distances);
    cudaFree(d_primitive_aabb_min);
    cudaFree(d_primitive_aabb_max);
    cudaFree(d_results);
    cudaFree(d_result_counts);
    cudaFree(d_bvh_aabb_mins);
    cudaFree(d_bvh_aabb_maxs);
    cudaFree(d_bvh_left_children);
    cudaFree(d_bvh_right_children);
    cudaFree(d_bvh_primitive_starts);
    cudaFree(d_bvh_primitive_counts);
    cudaFree(d_bvh_is_leaf_flags);
}

} // extern "C"
