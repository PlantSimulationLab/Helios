/** \file "CollisionDetection.h" Primary header file for collision detection plugin

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef COLLISION_DETECTION_H
#define COLLISION_DETECTION_H

#include <set>
#include "Context.h"

/**
 * \brief Class for GPU-accelerated collision detection and spatial queries in Helios
 *
 * The CollisionDetection class provides efficient algorithms for:
 * 1. Primitive/CompoundObject collision detection using BVH acceleration
 * 2. Cone intersection queries from arbitrary points
 * 3. Grid-based intersection queries (replacing voxel primitives)
 * 4. Layout optimization to minimize collisions
 */
class CollisionDetection {
public:
    // -------- CONE COLLISION STRUCTURES --------

    /**
     * \brief Structure representing a cone for collision detection
     */
    struct Cone {
        helios::vec3 apex; //!< Cone apex/tip position
        helios::vec3 axis; //!< Normalized axis direction (apex to base)
        float half_angle; //!< Half-angle of cone in radians
        float height; //!< Height of cone (0 = infinite)
    };

    /**
     * \brief Structure representing the result of optimal cone path detection
     */
    struct OptimalPathResult {
        helios::vec3 direction; //!< Optimal direction vector (normalized)
        int collisionCount; //!< Number of collisions along this path
        float confidence; //!< Confidence score (0-1)
    };

    /**
     * \brief Constructor for CollisionDetection class
     * \param[in] context Pointer to the Helios context
     */
    explicit CollisionDetection(helios::Context *context);

    /**
     * \brief Destructor for CollisionDetection class
     */
    ~CollisionDetection();

    // -------- PRIMITIVE/OBJECT COLLISION DETECTION --------

    /**
     * \brief Find all primitives/objects that collide with a given primitive
     * \param[in] UUID Unique identifier of the query primitive
     * \return Vector of UUIDs that intersect with the query primitive
     */
    std::vector<uint> findCollisions(uint UUID);

    /**
     * \brief Find all primitives/objects that collide with any of the given primitives
     * \param[in] UUIDs Vector of query primitive UUIDs
     * \return Vector of UUIDs that intersect with any of the query primitives
     */
    std::vector<uint> findCollisions(const std::vector<uint> &UUIDs);

    /**
     * \brief Find collisions between primitives and compound objects
     * \param[in] primitive_UUIDs Vector of primitive UUIDs to test
     * \param[in] object_IDs Vector of compound object IDs to test against
     * \return Vector of UUIDs that intersect
     */
    std::vector<uint> findCollisions(const std::vector<uint> &primitive_UUIDs, const std::vector<uint> &object_IDs);

    /**
     * \brief Find collisions restricting both query and target geometry
     * \param[in] query_UUIDs Vector of primitive UUIDs to test for collisions
     * \param[in] query_object_IDs Vector of object IDs to test for collisions
     * \param[in] target_UUIDs Vector of primitive UUIDs to test against (empty = all primitives)
     * \param[in] target_object_IDs Vector of object IDs to test against (empty = all objects)
     * \return Vector of UUIDs from the target set that intersect with any query primitive
     */
    std::vector<uint> findCollisions(const std::vector<uint> &query_UUIDs, const std::vector<uint> &query_object_IDs, const std::vector<uint> &target_UUIDs, const std::vector<uint> &target_object_IDs);

    // -------- CONE INTERSECTION QUERIES --------

    /**
     * \brief Find the radial direction through a cone toward the largest gap
     * \param[in] apex Cone apex position
     * \param[in] centralAxis Central axis of the cone
     * \param[in] half_angle Half-angle of cone in radians
     * \param[in] height Height of cone (0 = infinite)
     * \param[in] initialSamples Number of initial uniform samples (default: 256)
     * \return OptimalPathResult containing the optimal direction and collision count
     */
    OptimalPathResult findOptimalConePath(const helios::vec3 &apex, const helios::vec3 &centralAxis, float half_angle, float height = 0.0f, int initialSamples = 256);

    // -------- GRID-BASED INTERSECTION --------

    /**
     * \brief Calculate which primitives fall within a 3D grid
     * \param[in] grid_center Center point of the 3D grid
     * \param[in] grid_size Size of the grid in x, y, z dimensions
     * \param[in] grid_divisions Number of divisions in x, y, z dimensions
     * \param[in] UUIDs Optional vector of specific UUIDs to test (empty = all primitives)
     */
    void calculateGridIntersection(const helios::vec3 &grid_center, const helios::vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<uint> &UUIDs = {});

    /**
     * \brief Get all grid cells and their contained primitives
     * \return 3D vector indexed as [i][j][k] containing UUIDs in each cell
     */
    std::vector<std::vector<std::vector<std::vector<uint>>>> getGridCells();

    /**
     * \brief Get primitives in a specific grid cell
     * \param[in] i Grid cell index in x-dimension
     * \param[in] j Grid cell index in y-dimension
     * \param[in] k Grid cell index in z-dimension
     * \return Vector of UUIDs in the specified grid cell
     */
    std::vector<uint> getGridIntersections(int i, int j, int k);

    // -------- FEATURE 4: COLLISION MINIMIZATION --------

    /**
     * \brief Optimize layout of primitives/objects to minimize collisions
     * \param[in] UUIDs Vector of primitive/object UUIDs to optimize
     * \param[in] learning_rate Step size for optimization algorithm
     * \param[in] max_iterations Maximum number of optimization iterations
     * \return Final collision count after optimization
     */
    int optimizeLayout(const std::vector<uint> &UUIDs, float learning_rate = 0.01f, int max_iterations = 1000);

    // -------- BVH MANAGEMENT --------

    /**
     * \brief Build or rebuild the BVH acceleration structure
     * \param[in] UUIDs Optional vector of specific UUIDs to include (empty = all primitives)
     */
    void buildBVH(const std::vector<uint> &UUIDs = {});

    /**
     * \brief Force a complete rebuild of the BVH
     */
    void rebuildBVH();

    /**
     * \brief Check if BVH is valid and up-to-date
     * \return True if BVH is valid, false otherwise
     */
    [[nodiscard]] bool isBVHValid() const;

    // -------- GPU ACCELERATION CONTROL --------

    /**
     * \brief Enable GPU acceleration for collision detection
     */
    void enableGPUAcceleration();

    /**
     * \brief Disable GPU acceleration (use CPU-only algorithms)
     */
    void disableGPUAcceleration();

    /**
     * \brief Check if GPU acceleration is enabled
     * \return True if GPU acceleration is enabled
     */
    [[nodiscard]] bool isGPUAccelerationEnabled() const;

    // -------- UTILITY METHODS --------

    /**
     * \brief Disable console output messages
     */
    void disableMessages();

    /**
     * \brief Enable console output messages
     */
    void enableMessages();

    /**
     * \brief Get the number of primitives in the BVH
     * \return Number of primitives
     */
    [[nodiscard]] size_t getPrimitiveCount() const;

    /**
     * \brief Get BVH statistics
     * \param[out] node_count Number of nodes in BVH
     * \param[out] leaf_count Number of leaf nodes in BVH
     * \param[out] max_depth Maximum depth of BVH
     */
    void getBVHStatistics(size_t &node_count, size_t &leaf_count, size_t &max_depth) const;

    /**
     * \brief Self-test routine to verify plugin functionality
     * \return 0 if all tests pass, non-zero otherwise
     */
    static int selfTest(int argc = 0, char** argv = nullptr);

private:
    //! Pointer to the Helios context
    helios::Context *context;

    //! Flag to enable/disable GPU acceleration
    bool gpu_acceleration_enabled;

    //! Flag to enable/disable console messages
    bool printmessages;

    // -------- BVH DATA STRUCTURES --------

    /**
     * \brief BVH node structure for CPU storage
     */
    struct BVHNode {
        helios::vec3 aabb_min; //!< Minimum corner of axis-aligned bounding box
        helios::vec3 aabb_max; //!< Maximum corner of axis-aligned bounding box
        uint left_child; //!< Index of left child node (0xFFFFFFFF if leaf)
        uint right_child; //!< Index of right child node (0xFFFFFFFF if leaf)
        uint primitive_start; //!< Starting index in primitive_indices array
        uint primitive_count; //!< Number of primitives in this node (0 for internal nodes)
        bool is_leaf; //!< True if this is a leaf node

        BVHNode() : aabb_min(0, 0, 0), aabb_max(0, 0, 0), left_child(0xFFFFFFFF), right_child(0xFFFFFFFF), primitive_start(0), primitive_count(0), is_leaf(false) {
        }
    };

    //! Vector of BVH nodes (linearized tree structure)
    std::vector<BVHNode> bvh_nodes;

    //! Primitive indices sorted by BVH construction
    std::vector<uint> primitive_indices;

    //! Cached primitive bounding boxes (optimization for BVH construction)
    std::unordered_map<uint, std::pair<helios::vec3, helios::vec3>> primitive_aabbs_cache;

    //! Grid intersection results [i][j][k] = vector of UUIDs
    std::vector<std::vector<std::vector<std::vector<uint>>>> grid_cells;

    //! Grid parameters
    helios::vec3 grid_center;
    helios::vec3 grid_size;
    helios::int3 grid_divisions;

    // -------- GPU MEMORY POINTERS --------

    //! GPU memory for BVH nodes
    void *d_bvh_nodes;

    //! GPU memory for primitive indices
    uint *d_primitive_indices;

    //! GPU memory allocated flag
    bool gpu_memory_allocated;

    //! Internal tracking of processed UUIDs to avoid redundant BVH rebuilds
    std::set<uint> last_processed_uuids;

    //! Internal tracking of deleted UUIDs that we've already handled
    std::set<uint> last_processed_deleted_uuids;

    // -------- GAP DETECTION DATA STRUCTURES --------

    /**
     * \brief Structure representing a ray sample for gap detection
     */
    struct RaySample {
        helios::vec3 direction; //!< Normalized ray direction
        float distance; //!< Distance to first collision (-1 if no collision)
        bool is_free; //!< True if ray is unobstructed within cone height
    };

    /**
     * \brief Structure representing a detected gap (free space region)
     */
    struct Gap {
        helios::vec3 center_direction; //!< Direction toward gap center
        float angular_size; //!< Angular size in steradians
        float angular_distance; //!< Angular distance from cone centerline
        float score; //!< Fish-eye metric score
        std::vector<int> sample_indices; //!< Indices of ray samples in this gap
    };

    // -------- PRIVATE HELPER METHODS --------

    /**
     * \brief Ensure BVH is current by rebuilding if geometry has changed
     *
     * This method automatically checks if the Context geometry is dirty
     * and rebuilds the BVH as needed. It replaces manual BVH management.
     */
    void ensureBVHCurrent();

    /**
     * \brief Calculate axis-aligned bounding box for a set of primitives
     * \param[in] primitives Vector of primitive UUIDs
     * \param[out] aabb_min Minimum corner of bounding box
     * \param[out] aabb_max Maximum corner of bounding box
     */
    void calculateAABB(const std::vector<uint> &primitives, helios::vec3 &aabb_min, helios::vec3 &aabb_max) const;

    /**
     * \brief Recursively build BVH using Surface Area Heuristic
     * \param[in] node_index Index of current node being built
     * \param[in,out] primitives Vector of primitive UUIDs to partition
     * \param[in] depth Current recursion depth
     */
    void buildBVHRecursive(uint node_index, size_t primitive_start, size_t primitive_count, int depth);

    /**
     * \brief Traverse BVH on CPU to find intersecting primitives
     * \param[in] query_aabb_min Minimum corner of query bounding box
     * \param[in] query_aabb_max Maximum corner of query bounding box
     * \return Vector of intersecting primitive UUIDs
     */
    std::vector<uint> traverseBVH_CPU(const helios::vec3 &query_aabb_min, const helios::vec3 &query_aabb_max);

    /**
     * \brief Traverse BVH on GPU to find intersecting primitives
     * \param[in] query_aabb_min Minimum corner of query bounding box
     * \param[in] query_aabb_max Maximum corner of query bounding box
     * \return Vector of intersecting primitive UUIDs
     */
    std::vector<uint> traverseBVH_GPU(const helios::vec3 &query_aabb_min, const helios::vec3 &query_aabb_max);

    /**
     * \brief Test if two AABBs intersect
     * \param[in] min1 Minimum corner of first AABB
     * \param[in] max1 Maximum corner of first AABB
     * \param[in] min2 Minimum corner of second AABB
     * \param[in] max2 Maximum corner of second AABB
     * \return True if AABBs intersect
     */
    bool aabbIntersect(const helios::vec3 &min1, const helios::vec3 &max1, const helios::vec3 &min2, const helios::vec3 &max2);

    /**
     * \brief Test if a ray intersects an AABB
     * \param[in] origin Ray origin point
     * \param[in] direction Ray direction vector (normalized)
     * \param[in] aabb_min Minimum corner of AABB
     * \param[in] aabb_max Maximum corner of AABB
     * \param[out] t_min Near intersection parameter (if intersects)
     * \param[out] t_max Far intersection parameter (if intersects)
     * \return True if ray intersects AABB
     */
    bool rayAABBIntersect(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max, float &t_min, float &t_max);

    /**
     * \brief Test if a cone intersects an AABB
     * \param[in] cone Cone to test
     * \param[in] aabb_min Minimum corner of AABB
     * \param[in] aabb_max Maximum corner of AABB
     * \return True if cone and AABB intersect
     */
    bool coneAABBIntersect(const Cone &cone, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max);

    /**
     * \brief Test if a cone intersects an AABB
     * \param[in] cone_origin Origin point of the cone
     * \param[in] cone_direction Direction vector of the cone (normalized)
     * \param[in] cone_angle Half-angle of the cone in radians
     * \param[in] max_distance Maximum distance to test
     * \param[in] aabb_min Minimum corner of AABB
     * \param[in] aabb_max Maximum corner of AABB
     * \return True if cone intersects AABB
     */
    bool coneAABBIntersect(const helios::vec3 &cone_origin, const helios::vec3 &cone_direction, float cone_angle, float max_distance, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max);

    /**
     * \brief Count intersections along a ray using BVH traversal
     * \param[in] origin Starting point of the ray
     * \param[in] direction Direction vector of the ray (normalized)
     * \param[in] max_distance Maximum distance to test (negative = infinite)
     * \return Number of primitive intersections along the ray
     */
    int countRayIntersections(const helios::vec3 &origin, const helios::vec3 &direction, float max_distance = -1.0f);

    /**
     * \brief Generate uniform sample directions within a cone
     * \param[in] apex Cone apex position
     * \param[in] central_axis Central axis direction (normalized)
     * \param[in] half_angle Half-angle of cone in radians
     * \param[in] num_samples Number of directions to generate
     * \return Vector of normalized direction vectors
     */
    std::vector<helios::vec3> sampleDirectionsInCone(const helios::vec3 &apex, const helios::vec3 &central_axis, float half_angle, int num_samples);

    // -------- GAP DETECTION HELPER METHODS --------

    /**
     * \brief Detect gaps (free space regions) within a cone using dense ray sampling
     * \param[in] apex Cone apex position
     * \param[in] central_axis Central axis direction (normalized)
     * \param[in] half_angle Half-angle of cone in radians
     * \param[in] height Maximum distance to sample (0 = infinite)
     * \param[in] num_samples Number of ray samples to generate
     * \return Vector of detected gaps
     */
    std::vector<Gap> detectGapsInCone(const helios::vec3 &apex, const helios::vec3 &central_axis, float half_angle, float height, int num_samples);

    /**
     * \brief Calculate angular size (solid angle) of a gap in steradians
     * \param[in] gap_samples Vector of ray samples that form the gap
     * \param[in] central_axis Central axis for reference
     * \return Angular size in steradians
     */
    float calculateGapAngularSize(const std::vector<RaySample> &gap_samples, const helios::vec3 &central_axis);

    /**
     * \brief Score gaps using fish-eye metric (gap size vs angular distance)
     * \param[in,out] gaps Vector of gaps to score
     * \param[in] central_axis Central axis for angular distance calculation
     */
    void scoreGapsByFishEyeMetric(std::vector<Gap> &gaps, const helios::vec3 &central_axis);

    /**
     * \brief Find optimal direction toward the highest-scoring gap
     * \param[in] gaps Vector of detected and scored gaps
     * \param[in] central_axis Fallback direction if no gaps found
     * \return Direction toward optimal gap center
     */
    helios::vec3 findOptimalGapDirection(const std::vector<Gap> &gaps, const helios::vec3 &central_axis);

    /**
     * \brief Allocate GPU memory for BVH
     */
    void allocateGPUMemory();

    /**
     * \brief Free GPU memory
     */
    void freeGPUMemory();

    /**
     * \brief Transfer BVH data from CPU to GPU
     */
    void transferBVHToGPU();

    /**
     * \brief Mark BVH as dirty (needs rebuilding)
     */
    void markBVHDirty();

    /**
     * \brief Validate that all required primitives exist in context
     * \param[in] UUIDs Vector of UUIDs to validate
     * \return True if all UUIDs are valid
     */
    bool validateUUIDs(const std::vector<uint> &UUIDs) const;
};

#endif
