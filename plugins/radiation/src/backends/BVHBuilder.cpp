/** \file "BVHBuilder.cpp" BVH construction implementation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "BVHBuilder.h"
#include <algorithm>
#include <limits>
#include <cmath>

// TODO: Add TBB parallelization in future optimization phase
// #include <tbb/parallel_for.h>
// #include <tbb/blocked_range.h>

namespace helios {

    std::vector<BVHNode> BVHBuilder::build(const RayTracingGeometry &geom) {
        geometry = &geom;

        if (geom.primitive_count == 0) {
            return {}; // Empty BVH
        }

        // Step 1: Pre-compute type-specific offset tables (O(N) instead of O(N²))
        type_offsets.resize(geom.primitive_count);
        std::array<uint32_t, 6> type_counters = {0}; // 6 primitive types

        for (size_t i = 0; i < geom.primitive_count; ++i) {
            uint32_t prim_type = geom.primitive_types[i];
            type_offsets[i] = type_counters[prim_type]++;
        }

        // Step 2: Compute AABBs for all primitives (serial for now, TODO: parallelize with TBB)
        primitive_refs.resize(geom.primitive_count);

        for (size_t i = 0; i < geom.primitive_count; ++i) {
            uint32_t prim_type = geom.primitive_types[i];
            primitive_refs[i].prim_index = static_cast<uint32_t>(i);
            primitive_refs[i].prim_type = prim_type;
            primitive_refs[i].bounds = computePrimitiveAABB(static_cast<uint32_t>(i), prim_type);
        }

        // Step 3: Build BVH tree (recursive SAH)
        BuildNode *root = recursiveBuild(primitive_refs, 0, static_cast<uint32_t>(geom.primitive_count), 0);

        if (!root) {
            return {}; // Failed to build
        }

        // Step 4: Flatten tree to contiguous array
        std::vector<BVHNode> nodes;
        nodes.reserve(allocated_nodes.size() * 2); // Estimate size
        flattenTree(root, nodes);

        // Cleanup
        cleanup();
        primitive_refs.clear();
        type_offsets.clear();

        return nodes;
    }

    AABB BVHBuilder::computePrimitiveAABB(uint32_t prim_index, uint32_t prim_type) const {
        switch (prim_type) {
        case 0:
            return computePatchAABB(prim_index);
        case 1:
            return computeTriangleAABB(prim_index);
        case 2:
            return computeDiskAABB(prim_index);
        case 3:
            return computeTileAABB(prim_index);
        case 4:
            return computeVoxelAABB(prim_index);
        case 5:
            return computeBBoxAABB(prim_index);
        default:
            return AABB{{0, 0, 0}, {0, 0, 0}};
        }
    }

    helios::vec3 BVHBuilder::transformPoint(const float *transform, const helios::vec3 &point) const {
        // 4x4 row-major transform matrix (16 floats)
        // [m0  m1  m2  m3 ]   [x]
        // [m4  m5  m6  m7 ] * [y]
        // [m8  m9  m10 m11]   [z]
        // [m12 m13 m14 m15]   [1]

        helios::vec3 result;
        result.x = transform[0] * point.x + transform[1] * point.y + transform[2] * point.z + transform[3];
        result.y = transform[4] * point.x + transform[5] * point.y + transform[6] * point.z + transform[7];
        result.z = transform[8] * point.x + transform[9] * point.y + transform[10] * point.z + transform[11];
        return result;
    }

    AABB BVHBuilder::computePatchAABB(uint32_t prim_index) const {
        // Canonical patch vertices in local space — transform to world space for AABB
        static const helios::vec3 canonical_quad[4] = {
            {-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f},
            {0.5f, 0.5f, 0.f}, {-0.5f, 0.5f, 0.f}
        };

        const float *transform = &geometry->transform_matrices[prim_index * 16];
        AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};

        for (uint32_t v = 0; v < 4; ++v) {
            helios::vec3 world_vertex = transformPoint(transform, canonical_quad[v]);

            bounds.min.x = std::min(bounds.min.x, world_vertex.x);
            bounds.min.y = std::min(bounds.min.y, world_vertex.y);
            bounds.min.z = std::min(bounds.min.z, world_vertex.z);
            bounds.max.x = std::max(bounds.max.x, world_vertex.x);
            bounds.max.y = std::max(bounds.max.y, world_vertex.y);
            bounds.max.z = std::max(bounds.max.z, world_vertex.z);
        }

        return bounds;
    }

    AABB BVHBuilder::computeTriangleAABB(uint32_t prim_index) const {
        // Canonical triangle vertices in local space — transform to world space for AABB
        static const helios::vec3 canonical_tri[3] = {
            {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}
        };

        const float *transform = &geometry->transform_matrices[prim_index * 16];
        AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};

        for (uint32_t v = 0; v < 3; ++v) {
            helios::vec3 world_vertex = transformPoint(transform, canonical_tri[v]);

            bounds.min.x = std::min(bounds.min.x, world_vertex.x);
            bounds.min.y = std::min(bounds.min.y, world_vertex.y);
            bounds.min.z = std::min(bounds.min.z, world_vertex.z);
            bounds.max.x = std::max(bounds.max.x, world_vertex.x);
            bounds.max.y = std::max(bounds.max.y, world_vertex.y);
            bounds.max.z = std::max(bounds.max.z, world_vertex.z);
        }

        return bounds;
    }

    AABB BVHBuilder::computeDiskAABB(uint32_t prim_index) const {
        // O(1) lookup using pre-computed offset table
        uint32_t disk_index = type_offsets[prim_index];

        // Disks: center + radius + normal (already in world space from disk_centers)
        const float *transform = &geometry->transform_matrices[prim_index * 16];
        helios::vec3 center = geometry->disk_centers[disk_index];
        float radius = geometry->disk_radii[disk_index];

        // Apply transform to center
        helios::vec3 world_center = transformPoint(transform, center);

        // Conservative AABB: cube around center with side length = 2*radius
        AABB bounds;
        bounds.min = world_center - helios::vec3(radius, radius, radius);
        bounds.max = world_center + helios::vec3(radius, radius, radius);

        return bounds;
    }

    AABB BVHBuilder::computeTileAABB(uint32_t prim_index) const {
        // Canonical tile vertices in local space — transform to world space for AABB
        static const helios::vec3 canonical_quad[4] = {
            {-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f},
            {0.5f, 0.5f, 0.f}, {-0.5f, 0.5f, 0.f}
        };

        const float *transform = &geometry->transform_matrices[prim_index * 16];
        AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};

        for (uint32_t v = 0; v < 4; ++v) {
            helios::vec3 world_vertex = transformPoint(transform, canonical_quad[v]);

            bounds.min.x = std::min(bounds.min.x, world_vertex.x);
            bounds.min.y = std::min(bounds.min.y, world_vertex.y);
            bounds.min.z = std::min(bounds.min.z, world_vertex.z);
            bounds.max.x = std::max(bounds.max.x, world_vertex.x);
            bounds.max.y = std::max(bounds.max.y, world_vertex.y);
            bounds.max.z = std::max(bounds.max.z, world_vertex.z);
        }

        return bounds;
    }

    AABB BVHBuilder::computeVoxelAABB(uint32_t prim_index) const {
        // O(1) lookup using pre-computed offset table
        uint32_t voxel_index = type_offsets[prim_index];

        // Voxels have 8 vertices
        const float *transform = &geometry->transform_matrices[prim_index * 16];
        AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};

        for (uint32_t v = 0; v < 8; ++v) {
            helios::vec3 vertex = geometry->voxels.vertices[voxel_index * 8 + v];
            helios::vec3 world_vertex = transformPoint(transform, vertex);

            bounds.min.x = std::min(bounds.min.x, world_vertex.x);
            bounds.min.y = std::min(bounds.min.y, world_vertex.y);
            bounds.min.z = std::min(bounds.min.z, world_vertex.z);
            bounds.max.x = std::max(bounds.max.x, world_vertex.x);
            bounds.max.y = std::max(bounds.max.y, world_vertex.y);
            bounds.max.z = std::max(bounds.max.z, world_vertex.z);
        }

        return bounds;
    }

    AABB BVHBuilder::computeBBoxAABB(uint32_t prim_index) const {
        // O(1) lookup using pre-computed offset table
        uint32_t bbox_index = type_offsets[prim_index];

        // BBoxes have 8 vertices
        const float *transform = &geometry->transform_matrices[prim_index * 16];
        AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};

        for (uint32_t v = 0; v < 8; ++v) {
            helios::vec3 vertex = geometry->bboxes.vertices[bbox_index * 8 + v];
            helios::vec3 world_vertex = transformPoint(transform, vertex);

            bounds.min.x = std::min(bounds.min.x, world_vertex.x);
            bounds.min.y = std::min(bounds.min.y, world_vertex.y);
            bounds.min.z = std::min(bounds.min.z, world_vertex.z);
            bounds.max.x = std::max(bounds.max.x, world_vertex.x);
            bounds.max.y = std::max(bounds.max.y, world_vertex.y);
            bounds.max.z = std::max(bounds.max.z, world_vertex.z);
        }

        return bounds;
    }

    BVHBuilder::BuildNode *BVHBuilder::recursiveBuild(std::vector<PrimitiveRef> &refs, uint32_t start, uint32_t end, uint32_t depth) {
        BuildNode *node = new BuildNode();
        allocated_nodes.push_back(node);

        // Compute bounds of all primitives
        AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
        for (uint32_t i = start; i < end; ++i) {
            bounds.expand(refs[i].bounds);
        }
        node->bounds = bounds;

        uint32_t prim_count = end - start;

        // Leaf criteria: few primitives or max depth reached
        if (prim_count <= MAX_PRIMS_PER_LEAF || depth >= MAX_DEPTH) {
            return createLeaf(refs, start, end);
        }

        // Try to partition using SAH
        uint32_t mid, axis;
        if (!partitionSAH(refs, start, end, bounds, mid, axis)) {
            // SAH failed to find good split - create leaf
            return createLeaf(refs, start, end);
        }

        node->split_axis = axis;

        // Recursively build children
        node->left = recursiveBuild(refs, start, mid, depth + 1);
        node->right = recursiveBuild(refs, mid, end, depth + 1);

        return node;
    }

    bool BVHBuilder::partitionSAH(std::vector<PrimitiveRef> &refs, uint32_t start, uint32_t end, const AABB &bounds, uint32_t &mid, uint32_t &axis) {
        uint32_t prim_count = end - start;

        // Find best split axis and position using binned SAH
        float best_cost = std::numeric_limits<float>::infinity();
        uint32_t best_axis = 0;
        uint32_t best_split = 0;

        for (uint32_t test_axis = 0; test_axis < 3; ++test_axis) {
            // Bin primitives along this axis
            struct Bin {
                AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
                uint32_t count = 0;
            };
            Bin bins[NUM_BINS];

            float axis_min = (test_axis == 0) ? bounds.min.x : (test_axis == 1) ? bounds.min.y : bounds.min.z;
            float axis_max = (test_axis == 0) ? bounds.max.x : (test_axis == 1) ? bounds.max.y : bounds.max.z;
            float axis_extent = axis_max - axis_min;

            if (axis_extent < 1e-6f)
                continue; // Degenerate axis

            // Assign primitives to bins
            for (uint32_t i = start; i < end; ++i) {
                helios::vec3 centroid = refs[i].bounds.centroid();
                float centroid_axis = (test_axis == 0) ? centroid.x : (test_axis == 1) ? centroid.y : centroid.z;
                int bin_idx = static_cast<int>(NUM_BINS * (centroid_axis - axis_min) / axis_extent);
                bin_idx = std::min(bin_idx, static_cast<int>(NUM_BINS - 1));

                bins[bin_idx].count++;
                bins[bin_idx].bounds.expand(refs[i].bounds);
            }

            // Incremental SAH cost computation using prefix sums (O(NUM_BINS) instead of O(NUM_BINS^2))
            AABB left_prefix_bounds[NUM_BINS];
            AABB right_prefix_bounds[NUM_BINS];
            uint32_t left_prefix_count[NUM_BINS];
            uint32_t right_prefix_count[NUM_BINS];

            // Forward sweep: cumulative left bounds and counts
            {
                AABB running{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
                uint32_t count = 0;
                for (uint32_t i = 0; i < NUM_BINS - 1; ++i) {
                    running.expand(bins[i].bounds);
                    count += bins[i].count;
                    left_prefix_bounds[i] = running;
                    left_prefix_count[i] = count;
                }
            }

            // Backward sweep: cumulative right bounds and counts
            {
                AABB running{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
                uint32_t count = 0;
                for (int i = NUM_BINS - 1; i >= 1; --i) {
                    running.expand(bins[i].bounds);
                    count += bins[i].count;
                    right_prefix_bounds[i - 1] = running;
                    right_prefix_count[i - 1] = count;
                }
            }

            // Evaluate SAH cost at each split position in a single pass
            float parent_sa = bounds.surfaceArea();
            for (uint32_t split_idx = 0; split_idx < NUM_BINS - 1; ++split_idx) {
                if (left_prefix_count[split_idx] == 0 || right_prefix_count[split_idx] == 0)
                    continue;

                float cost = 1.0f + (left_prefix_bounds[split_idx].surfaceArea() * left_prefix_count[split_idx] + right_prefix_bounds[split_idx].surfaceArea() * right_prefix_count[split_idx]) / parent_sa;

                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = test_axis;
                    best_split = split_idx + 1;
                }
            }
        }

        // Check if SAH split is better than making a leaf
        float leaf_cost = static_cast<float>(prim_count);
        if (best_cost >= leaf_cost) {
            return false; // Leaf is better
        }

        // Partition primitives based on best split
        axis = best_axis;
        float axis_min = (axis == 0) ? bounds.min.x : (axis == 1) ? bounds.min.y : bounds.min.z;
        float axis_max = (axis == 0) ? bounds.max.x : (axis == 1) ? bounds.max.y : bounds.max.z;
        float axis_extent = axis_max - axis_min;
        float split_pos = axis_min + (best_split / static_cast<float>(NUM_BINS)) * axis_extent;

        auto partition_pred = [&](const PrimitiveRef &ref) {
            helios::vec3 centroid = ref.bounds.centroid();
            float centroid_axis = (axis == 0) ? centroid.x : (axis == 1) ? centroid.y : centroid.z;
            return centroid_axis < split_pos;
        };

        PrimitiveRef *mid_ptr = std::partition(&refs[start], &refs[end], partition_pred);
        mid = static_cast<uint32_t>(mid_ptr - &refs[0]);

        // Fallback if partition failed
        if (mid == start || mid == end) {
            mid = (start + end) / 2;
        }

        return true;
    }

    BVHBuilder::BuildNode *BVHBuilder::createLeaf(std::vector<PrimitiveRef> &refs, uint32_t start, uint32_t end) {
        BuildNode *node = new BuildNode();
        allocated_nodes.push_back(node);

        // Compute bounds
        AABB bounds{{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
        for (uint32_t i = start; i < end; ++i) {
            bounds.expand(refs[i].bounds);
        }
        node->bounds = bounds;

        // Store primitive indices
        node->first_prim_offset = static_cast<uint32_t>(primitive_indices.size());
        node->prim_count = end - start;
        node->prim_type = refs[start].prim_type; // Assume homogeneous leaf (all same type)

        for (uint32_t i = start; i < end; ++i) {
            primitive_indices.push_back(refs[i].prim_index);
        }

        return node;
    }

    uint32_t BVHBuilder::flattenTree(BuildNode *node, std::vector<BVHNode> &nodes) {
        BVHNode flat_node{};

        // AABB
        flat_node.aabb_min[0] = node->bounds.min.x;
        flat_node.aabb_min[1] = node->bounds.min.y;
        flat_node.aabb_min[2] = node->bounds.min.z;
        flat_node.aabb_max[0] = node->bounds.max.x;
        flat_node.aabb_max[1] = node->bounds.max.y;
        flat_node.aabb_max[2] = node->bounds.max.z;

        uint32_t my_offset = static_cast<uint32_t>(nodes.size());
        nodes.push_back(flat_node);

        if (node->prim_count > 0) {
            // Leaf node
            nodes[my_offset].prim_count = node->prim_count;
            nodes[my_offset].prim_type = node->prim_type;
            nodes[my_offset].first_prim = node->first_prim_offset;
            nodes[my_offset].left_child = node->first_prim_offset; // Reuse field
            nodes[my_offset].right_child = UINT32_MAX; // Mark as leaf
        } else {
            // Internal node
            nodes[my_offset].prim_count = 0;
            nodes[my_offset].split_axis = node->split_axis;
            nodes[my_offset].left_child = flattenTree(node->left, nodes);
            nodes[my_offset].right_child = flattenTree(node->right, nodes);
        }

        return my_offset;
    }

    void BVHBuilder::cleanup() {
        for (BuildNode *node : allocated_nodes) {
            delete node;
        }
        allocated_nodes.clear();
    }

} // namespace helios
