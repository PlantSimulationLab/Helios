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
#include <queue>

namespace helios {

    std::vector<BVHNode> BVHBuilder::build(const RayTracingGeometry &geom) {
        geometry = &geom;

        if (geom.primitive_count == 0) {
            return {}; // Empty BVH
        }

        // Clear state from previous build (prevents accumulation across multiple updateGeometry calls)
        primitive_indices.clear();

        // Step 1: Pre-compute type-specific offset tables (O(N) instead of O(N²))
        type_offsets.resize(geom.primitive_count);
        std::array<uint32_t, 6> type_counters = {0}; // 6 primitive types

        for (size_t i = 0; i < geom.primitive_count; ++i) {
            uint32_t prim_type = geom.primitive_types[i];
            type_offsets[i] = type_counters[prim_type]++;
        }

        // Step 2: Compute AABBs for all primitives
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
        // Breadth-first layout: nodes are stored level by level so that upper-level
        // nodes (visited by all rays) are contiguous at the start of the buffer and
        // remain cache-resident during traversal.

        // First pass: BFS to assign indices and collect nodes in level order
        std::queue<BuildNode *> bfs_queue;
        std::vector<BuildNode *> bfs_order;
        bfs_queue.push(node);
        while (!bfs_queue.empty()) {
            BuildNode *current = bfs_queue.front();
            bfs_queue.pop();
            bfs_order.push_back(current);
            if (current->prim_count == 0) {
                // Internal node: enqueue children (left then right)
                bfs_queue.push(current->left);
                bfs_queue.push(current->right);
            }
        }

        // Build a map from BuildNode pointer to its BFS index
        std::unordered_map<BuildNode *, uint32_t> node_to_index;
        for (uint32_t i = 0; i < bfs_order.size(); ++i) {
            node_to_index[bfs_order[i]] = i;
        }

        // Second pass: write BVHNode entries in BFS order with correct child indices
        nodes.resize(bfs_order.size());
        for (uint32_t i = 0; i < bfs_order.size(); ++i) {
            BuildNode *current = bfs_order[i];
            BVHNode flat_node{};

            flat_node.aabb_min[0] = current->bounds.min.x;
            flat_node.aabb_min[1] = current->bounds.min.y;
            flat_node.aabb_min[2] = current->bounds.min.z;
            flat_node.aabb_max[0] = current->bounds.max.x;
            flat_node.aabb_max[1] = current->bounds.max.y;
            flat_node.aabb_max[2] = current->bounds.max.z;

            if (current->prim_count > 0) {
                // Leaf node
                flat_node.prim_count = current->prim_count;
                flat_node.prim_type = current->prim_type;
                flat_node.first_prim = current->first_prim_offset;
                flat_node.left_child = current->first_prim_offset; // Reuse field
                flat_node.right_child = UINT32_MAX; // Mark as leaf
            } else {
                // Internal node
                flat_node.prim_count = 0;
                flat_node.split_axis = current->split_axis;
                flat_node.left_child = node_to_index[current->left];
                flat_node.right_child = node_to_index[current->right];
            }

            nodes[i] = flat_node;
        }

        return 0; // Root is always at index 0
    }

    void BVHBuilder::cleanup() {
        for (BuildNode *node : allocated_nodes) {
            delete node;
        }
        allocated_nodes.clear();
    }

    // ========== CWBVH Conversion Implementation ==========

    std::vector<CWBVH_Node> BVHBuilder::convertToCWBVH(const std::vector<BVHNode> &bvh2_nodes) {
        if (bvh2_nodes.empty()) {
            return {};
        }

        // Step 1: Reconstruct tree from flat BVH2
        BuildNode *root = reconstructTree(bvh2_nodes, 0);

        // Step 2: Collapse to BVH8
        BVH8Node *root8 = collapseToBVH8(root);

        // Step 3: Flatten with compression
        std::vector<CWBVH_Node> cwbvh_nodes;
        flattenBVH8(root8, cwbvh_nodes);

        // Cleanup
        for (BVH8Node *node : allocated_bvh8_nodes) {
            delete node;
        }
        allocated_bvh8_nodes.clear();

        return cwbvh_nodes;
    }

    BVHBuilder::BuildNode *BVHBuilder::reconstructTree(const std::vector<BVHNode> &bvh2_nodes, uint32_t node_idx) {
        if (node_idx >= bvh2_nodes.size()) {
            return nullptr;
        }

        const BVHNode &flat_node = bvh2_nodes[node_idx];
        BuildNode *node = new BuildNode();
        allocated_nodes.push_back(node);

        // Copy AABB
        node->bounds.min = helios::make_vec3(flat_node.aabb_min[0], flat_node.aabb_min[1], flat_node.aabb_min[2]);
        node->bounds.max = helios::make_vec3(flat_node.aabb_max[0], flat_node.aabb_max[1], flat_node.aabb_max[2]);

        if (flat_node.right_child == UINT32_MAX) {
            // Leaf node
            node->prim_count = flat_node.prim_count;
            node->prim_type = flat_node.prim_type;
            node->first_prim_offset = flat_node.first_prim;
            node->left = nullptr;
            node->right = nullptr;
        } else {
            // Internal node
            node->prim_count = 0;
            node->split_axis = flat_node.split_axis;
            node->left = reconstructTree(bvh2_nodes, flat_node.left_child);
            node->right = reconstructTree(bvh2_nodes, flat_node.right_child);
        }

        return node;
    }

    BVHBuilder::BVH8Node *BVHBuilder::collapseToBVH8(BuildNode *bvh2_node, int depth) {
        if (!bvh2_node) return nullptr;

        // Helper lambda: Count total primitives in a subtree
        auto countPrimitivesInSubtree = [](BuildNode *node, auto& self) -> uint32_t {
            if (!node) return 0;
            if (node->prim_count > 0) return node->prim_count;
            return self(node->left, self) + self(node->right, self);
        };
        BVH8Node *node8 = new BVH8Node();
        allocated_bvh8_nodes.push_back(node8);

        node8->aabb = bvh2_node->bounds;

        // If already a leaf, create a BVH8 leaf with one child
        if (bvh2_node->prim_count > 0) {
            node8->children[0] = bvh2_node;
            node8->is_leaf[0] = true;
            node8->first_prim[0] = bvh2_node->first_prim_offset;
            node8->prim_count[0] = bvh2_node->prim_count;
            node8->prim_type[0] = bvh2_node->prim_type;
            return node8;
        }

        // Greedy collapse: collect up to 8 BuildNode children from BVH2 subtree
        std::vector<BuildNode *> candidates = {bvh2_node};

        while (candidates.size() < 8) {
            // Find node with most primitives to expand (better load balancing than surface area)
            int best_idx = -1;
            uint32_t max_prim_count = 0;

            for (size_t i = 0; i < candidates.size(); i++) {
                if (candidates[i]->prim_count > 0) continue; // Skip leaves

                // Count total primitives in this subtree (recursive count)
                uint32_t subtree_prims = countPrimitivesInSubtree(candidates[i], countPrimitivesInSubtree);

                if (subtree_prims > max_prim_count) {
                    max_prim_count = subtree_prims;
                    best_idx = static_cast<int>(i);
                }
            }

            if (best_idx == -1) break; // All leaves

            // Expand the node with most primitives
            BuildNode *expand = candidates[best_idx];
            candidates.erase(candidates.begin() + best_idx);
            if (expand->left) candidates.push_back(expand->left);
            if (expand->right) candidates.push_back(expand->right);
        }

        // Assign BuildNode children to octants
        assignChildrenToOctants(node8, candidates);

        return node8;
    }

    void BVHBuilder::assignChildrenToOctants(BVH8Node *node8, std::vector<BuildNode *> &children) {
        helios::vec3 center = node8->aabb.centroid();

        // 8 octant directions (Morton order) - bit 1 = positive direction
        static const helios::vec3 octants[8] = {
            {1, 1, 1},      // 000
            {-1, 1, 1},     // 001 (x negative)
            {1, -1, 1},     // 010 (y negative)
            {-1, -1, 1},    // 011 (x,y negative)
            {1, 1, -1},     // 100 (z negative)
            {-1, 1, -1},    // 101 (x,z negative)
            {1, -1, -1},    // 110 (y,z negative)
            {-1, -1, -1}    // 111 (all negative)
        };

        // Greedy assignment: minimize distance to octant direction
        for (int slot = 0; slot < 8; slot++) {
            if (children.empty()) break;

            int best = -1;
            float best_score = 1e30f;

            for (size_t c = 0; c < children.size(); c++) {
                helios::vec3 child_center = children[c]->bounds.centroid();
                helios::vec3 offset = child_center - center;

                // Negative dot product = child is in this octant's direction
                float score = -(offset.x * octants[slot].x + offset.y * octants[slot].y + offset.z * octants[slot].z);

                if (score < best_score) {
                    best_score = score;
                    best = static_cast<int>(c);
                }
            }

            if (best >= 0) {
                BuildNode *child = children[best];
                node8->children[slot] = child;
                node8->is_leaf[slot] = (child->prim_count > 0);

                if (node8->is_leaf[slot]) {
                    node8->first_prim[slot] = child->first_prim_offset;
                    node8->prim_count[slot] = child->prim_count;
                    node8->prim_type[slot] = child->prim_type;
                }

                children.erase(children.begin() + best);
            }
        }
    }

    void BVHBuilder::compressNode(const BVH8Node *src, CWBVH_Node &dst) {
        // Compute quantization parameters for each axis
        float extent_x = src->aabb.max.x - src->aabb.min.x;
        float extent_y = src->aabb.max.y - src->aabb.min.y;
        float extent_z = src->aabb.max.z - src->aabb.min.z;

        dst.p[0] = src->aabb.min.x;
        dst.p[1] = src->aabb.min.y;
        dst.p[2] = src->aabb.min.z;

        // Compute exponents: ceil(log2(extent / 255))
        int exp_x = (extent_x > 1e-6f) ? static_cast<int>(std::ceil(std::log2(extent_x / 255.0f))) : 0;
        int exp_y = (extent_y > 1e-6f) ? static_cast<int>(std::ceil(std::log2(extent_y / 255.0f))) : 0;
        int exp_z = (extent_z > 1e-6f) ? static_cast<int>(std::ceil(std::log2(extent_z / 255.0f))) : 0;

        // Store IEEE754-biased exponents so the GPU can decode via uintBitsToFloat(byte << 23)
        uint8_t biased_x = static_cast<uint8_t>(std::max(0, std::min(255, 127 + exp_x)));
        uint8_t biased_y = static_cast<uint8_t>(std::max(0, std::min(255, 127 + exp_y)));
        uint8_t biased_z = static_cast<uint8_t>(std::max(0, std::min(255, 127 + exp_z)));
        dst.e_packed = biased_x | (biased_y << 8) | (biased_z << 16);
        dst.imask = 0;

        float scale_x = std::pow(2.0f, static_cast<float>(exp_x));
        float scale_y = std::pow(2.0f, static_cast<float>(exp_y));
        float scale_z = std::pow(2.0f, static_cast<float>(exp_z));

        // Quantize each child's AABB
        for (int child = 0; child < 8; child++) {
            if (!src->children[child]) {
                // Empty slot - set inverted bounds so AABB test fails
                dst.qmin_x[child] = 255;
                dst.qmin_y[child] = 255;
                dst.qmin_z[child] = 255;
                dst.qmax_x[child] = 0;
                dst.qmax_y[child] = 0;
                dst.qmax_z[child] = 0;
                continue;
            }

            BuildNode *child_node = src->children[child];

            // Quantize X axis
            float min_qx = (child_node->bounds.min.x - dst.p[0]) / scale_x;
            float max_qx = (child_node->bounds.max.x - dst.p[0]) / scale_x;
            int qmin_x = std::max(0, std::min(255, static_cast<int>(std::floor(min_qx - 0.001f))));
            int qmax_x = std::max(0, std::min(255, static_cast<int>(std::ceil(max_qx + 0.001f))));
            dst.qmin_x[child] = static_cast<uint8_t>(qmin_x);
            dst.qmax_x[child] = static_cast<uint8_t>(qmax_x);

            // Quantize Y axis
            float min_qy = (child_node->bounds.min.y - dst.p[1]) / scale_y;
            float max_qy = (child_node->bounds.max.y - dst.p[1]) / scale_y;
            int qmin_y = std::max(0, std::min(255, static_cast<int>(std::floor(min_qy - 0.001f))));
            int qmax_y = std::max(0, std::min(255, static_cast<int>(std::ceil(max_qy + 0.001f))));
            dst.qmin_y[child] = static_cast<uint8_t>(qmin_y);
            dst.qmax_y[child] = static_cast<uint8_t>(qmax_y);

            // Quantize Z axis
            float min_qz = (child_node->bounds.min.z - dst.p[2]) / scale_z;
            float max_qz = (child_node->bounds.max.z - dst.p[2]) / scale_z;
            int qmin_z = std::max(0, std::min(255, static_cast<int>(std::floor(min_qz - 0.001f))));
            int qmax_z = std::max(0, std::min(255, static_cast<int>(std::ceil(max_qz + 0.001f))));
            dst.qmin_z[child] = static_cast<uint8_t>(qmin_z);
            dst.qmax_z[child] = static_cast<uint8_t>(qmax_z);

            // Set imask bit if child is internal (not a leaf)
            if (!src->is_leaf[child]) {
                dst.imask |= (1 << child);
            }

            // Store per-child leaf data in extended fields
            dst.child_first_prim[child] = src->first_prim[child];
            dst.child_prim_count[child] = static_cast<uint8_t>(src->prim_count[child]);
            dst.child_prim_type[child] = static_cast<uint8_t>(src->prim_type[child]);
        }
    }

    uint32_t BVHBuilder::flattenBVH8(BVH8Node *root, std::vector<CWBVH_Node> &cwbvh_nodes) {
        // BFS flattening ensures internal children of each parent occupy consecutive
        // array slots, which is required for the GPU popcount-based child indexing:
        //   child_index = base_index_child + bitCount(imask & ((1 << slot) - 1))

        struct QueueEntry {
            BVH8Node *node;
            uint32_t array_index;
        };
        std::queue<QueueEntry> bfs_queue;

        // Place root
        CWBVH_Node root_flat = {};
        compressNode(root, root_flat);
        cwbvh_nodes.push_back(root_flat);
        bfs_queue.push({root, 0});

        while (!bfs_queue.empty()) {
            auto [current, my_idx] = bfs_queue.front();
            bfs_queue.pop();

            // Collect internal children (need recursive collapse to BVH8)
            std::vector<std::pair<int, BVH8Node *>> internal_children;
            for (int i = 0; i < 8; i++) {
                if (current->children[i] && !current->is_leaf[i]) {
                    BVH8Node *child8 = collapseToBVH8(current->children[i]);
                    internal_children.push_back({i, child8});
                }
            }
            if (internal_children.empty()) continue;

            // Children placed contiguously starting at current array end
            uint32_t child_base = static_cast<uint32_t>(cwbvh_nodes.size());
            cwbvh_nodes[my_idx].base_index_child = child_base;

            for (auto &[slot, child8] : internal_children) {
                CWBVH_Node child_flat = {};
                compressNode(child8, child_flat);
                uint32_t child_idx = static_cast<uint32_t>(cwbvh_nodes.size());
                cwbvh_nodes.push_back(child_flat);
                bfs_queue.push({child8, child_idx});
            }
        }

        return 0; // Root is always at index 0
    }

} // namespace helios
