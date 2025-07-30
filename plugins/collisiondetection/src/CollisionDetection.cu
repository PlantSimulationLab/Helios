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
 * \brief GPU-friendly BVH node structure
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

    // Stub implementation - will be completed in Phase 3
    for (int i = 0; i < num_queries; i++) {
        h_result_counts[i] = 0;
    }
}

} // extern "C"
