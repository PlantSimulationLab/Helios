/** \file "BVHBuilder.h" CPU-side BVH construction for Vulkan backend.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef BVH_BUILDER_H
#define BVH_BUILDER_H

#include "RayTracingTypes.h"
#include "global.h"
#include <vector>
#include <cstdint>

namespace helios {

    /**
     * @brief BVH node for GPU traversal (48 bytes, flat array)
     *
     * Flat array representation for efficient GPU traversal.
     * Nodes reference children via array indices (not pointers).
     * Leaf nodes contain homogeneous primitive types.
     */
    struct BVHNode {
        float aabb_min[3]; //!< AABB min corner (12 bytes)
        float aabb_max[3]; //!< AABB max corner (12 bytes)
        uint32_t left_child; //!< Left child index (or first_prim if leaf)
        uint32_t right_child; //!< Right child index (UINT32_MAX if leaf)
        uint32_t prim_count; //!< Primitive count (0 = internal node, >0 = leaf)
        uint32_t prim_type; //!< Primitive type (0-5) for leaf nodes
        uint32_t first_prim; //!< Index of first primitive in leaf
        uint32_t padding; //!< Padding to 48 bytes for alignment
    };

    static_assert(sizeof(BVHNode) == 48, "BVHNode must be exactly 48 bytes");

    /**
     * @brief Axis-aligned bounding box
     */
    struct AABB {
        helios::vec3 min;
        helios::vec3 max;

        /**
         * @brief Expand AABB to include another AABB
         */
        void expand(const AABB &other) {
            min.x = std::min(min.x, other.min.x);
            min.y = std::min(min.y, other.min.y);
            min.z = std::min(min.z, other.min.z);
            max.x = std::max(max.x, other.max.x);
            max.y = std::max(max.y, other.max.y);
            max.z = std::max(max.z, other.max.z);
        }

        /**
         * @brief Compute surface area
         */
        float surfaceArea() const {
            helios::vec3 d = max - min;
            return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
        }

        /**
         * @brief Get centroid
         */
        helios::vec3 centroid() const {
            return (min + max) * 0.5f;
        }
    };

    /**
     * @brief Primitive reference for BVH construction
     *
     * Temporary structure used during BVH building.
     * References a primitive by its index, type, and AABB.
     */
    struct PrimitiveRef {
        uint32_t prim_index; //!< Primitive index in geometry arrays
        uint32_t prim_type; //!< Primitive type (0-5)
        AABB bounds; //!< Primitive's AABB in world space
    };

    /**
     * @brief BVH builder using SAH (Surface Area Heuristic)
     *
     * Constructs a high-quality BVH on CPU using:
     * - TBB-parallelized AABB computation
     * - Top-down recursive SAH split (16-bin binned SAH)
     * - Type-homogeneous leaves (each leaf contains one primitive type)
     * - Flat array output for GPU traversal
     */
    class BVHBuilder {
    public:
        /**
         * @brief Build BVH from geometry
         *
         * @param[in] geometry Geometry data (all primitive types)
         * @return Flat array of BVH nodes ready for GPU upload
         *
         * Steps:
         * 1. Compute per-primitive AABBs (TBB parallel)
         * 2. Build BVH tree (top-down recursive SAH)
         * 3. Flatten tree to contiguous array (DFS order)
         */
        std::vector<BVHNode> build(const RayTracingGeometry &geometry);

        /**
         * @brief Get primitive indices for leaves (ordered by tree structure)
         *
         * @return Primitive indices in BVH leaf order
         *
         * Used to reorder geometry data to match BVH leaf layout.
         * Each leaf's primitives are contiguous in this array.
         */
        const std::vector<uint32_t> &getPrimitiveIndices() const {
            return primitive_indices;
        }

    private:
        /**
         * @brief Internal BVH node during construction (tree structure)
         */
        struct BuildNode {
            AABB bounds; //!< AABB of this node
            BuildNode *left = nullptr; //!< Left child
            BuildNode *right = nullptr; //!< Right child
            uint32_t first_prim_offset = 0; //!< Offset in primitive_indices
            uint32_t prim_count = 0; //!< Number of primitives in leaf
            uint32_t prim_type = 0; //!< Primitive type (for leaves)
            uint32_t split_axis = 0; //!< Split axis (0=x, 1=y, 2=z)
        };

        // Temporary data during construction
        std::vector<PrimitiveRef> primitive_refs; //!< Primitive references
        std::vector<uint32_t> primitive_indices; //!< Primitive indices in BVH order
        std::vector<BuildNode *> allocated_nodes; //!< Track allocated nodes for cleanup

        const RayTracingGeometry *geometry = nullptr; //!< Geometry being built

        // BVH construction parameters
        static constexpr uint32_t MAX_PRIMS_PER_LEAF = 4; //!< Max primitives per leaf
        static constexpr uint32_t MAX_DEPTH = 32; //!< Max tree depth
        static constexpr uint32_t NUM_BINS = 16; //!< Number of bins for SAH

        /**
         * @brief Compute AABB for a primitive
         *
         * @param prim_index Primitive index
         * @param prim_type Primitive type (0-5)
         * @return AABB in world space (after applying transform)
         */
        AABB computePrimitiveAABB(uint32_t prim_index, uint32_t prim_type) const;

        /**
         * @brief Compute AABB for patch (type 0)
         */
        AABB computePatchAABB(uint32_t prim_index) const;

        /**
         * @brief Compute AABB for triangle (type 1)
         */
        AABB computeTriangleAABB(uint32_t prim_index) const;

        /**
         * @brief Compute AABB for disk (type 2)
         */
        AABB computeDiskAABB(uint32_t prim_index) const;

        /**
         * @brief Compute AABB for tile (type 3)
         */
        AABB computeTileAABB(uint32_t prim_index) const;

        /**
         * @brief Compute AABB for voxel (type 4)
         */
        AABB computeVoxelAABB(uint32_t prim_index) const;

        /**
         * @brief Compute AABB for bbox (type 5)
         */
        AABB computeBBoxAABB(uint32_t prim_index) const;

        /**
         * @brief Apply transform matrix to vec3
         */
        helios::vec3 transformPoint(const float *transform, const helios::vec3 &point) const;

        /**
         * @brief Recursive BVH build
         *
         * @param refs Primitive references to partition
         * @param start Start index in refs
         * @param end End index in refs (exclusive)
         * @param depth Current tree depth
         * @return BuildNode for this subtree
         */
        BuildNode *recursiveBuild(std::vector<PrimitiveRef> &refs, uint32_t start, uint32_t end, uint32_t depth);

        /**
         * @brief Partition primitives using SAH
         *
         * @param refs Primitive references
         * @param start Start index
         * @param end End index (exclusive)
         * @param bounds AABB of all primitives
         * @param[out] mid Split point
         * @param[out] axis Split axis
         * @return True if split found, false if leaf should be created
         */
        bool partitionSAH(std::vector<PrimitiveRef> &refs, uint32_t start, uint32_t end, const AABB &bounds, uint32_t &mid, uint32_t &axis);

        /**
         * @brief Create leaf node
         */
        BuildNode *createLeaf(std::vector<PrimitiveRef> &refs, uint32_t start, uint32_t end);

        /**
         * @brief Flatten tree to contiguous array (DFS order)
         *
         * @param node Current node
         * @param[out] nodes Output array
         * @return Node index in output array
         */
        uint32_t flattenTree(BuildNode *node, std::vector<BVHNode> &nodes);

        /**
         * @brief Cleanup allocated nodes
         */
        void cleanup();
    };

} // namespace helios

#endif // BVH_BUILDER_H
