/** \file "CollisionDetection.h" Primary header file for collision detection plugin

    Copyright (C) 2016-2026 Brian Bailey

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

#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
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
     * \brief Angular bin structure for rasterization-based collision detection
     */
    struct AngularBins {
        int theta_divisions; //!< Number of azimuthal divisions (around cone axis)
        int phi_divisions; //!< Number of polar divisions (from apex to cone edge)
        float angular_resolution; //!< Steradians per bin

        // Coverage data using packed bits for memory efficiency
        std::vector<uint8_t> coverage_bits; //!< Packed coverage bits [theta][phi]
        std::vector<float> depth_values; //!< Depth values for covered bins only

        AngularBins(int theta_div, int phi_div) : theta_divisions(theta_div), phi_divisions(phi_div) {
            int total_bins = theta_div * phi_div;
            coverage_bits.resize((total_bins + 7) / 8, 0); // Packed bits
            depth_values.resize(total_bins, std::numeric_limits<float>::max());
            angular_resolution = (2.0f * M_PI) / float(theta_div * phi_div);
        }

        // Fast bit operations for coverage testing
        bool isCovered(int theta, int phi) const {
            int index = theta * phi_divisions + phi;
            return coverage_bits[index >> 3] & (1 << (index & 7));
        }

        void setCovered(int theta, int phi, float depth) {
            int index = theta * phi_divisions + phi;
            coverage_bits[index >> 3] |= (1 << (index & 7));
            depth_values[index] = std::min(depth_values[index], depth);
        }

        void clear() {
            std::fill(coverage_bits.begin(), coverage_bits.end(), 0);
            std::fill(depth_values.begin(), depth_values.end(), std::numeric_limits<float>::max());
        }
    };

    // -------- GENERIC RAY-TRACING STRUCTURES --------

    /**
     * \brief Structure representing a ray query for generic ray-tracing operations
     */
    struct RayQuery {
        helios::vec3 origin; //!< Ray origin point
        helios::vec3 direction; //!< Ray direction vector (should be normalized)
        float max_distance; //!< Maximum ray distance (negative = infinite)
        std::vector<uint> target_UUIDs; //!< Target primitive UUIDs (empty = all primitives)

        RayQuery() : origin(0, 0, 0), direction(0, 0, 1), max_distance(-1.0f) {
        }
        RayQuery(const helios::vec3 &ray_origin, const helios::vec3 &ray_direction, float max_dist = -1.0f, const std::vector<uint> &targets = {}) : origin(ray_origin), direction(ray_direction), max_distance(max_dist), target_UUIDs(targets) {
        }
    };

    /**
     * \brief Structure representing the result of a ray-primitive intersection
     */
    struct HitResult {
        bool hit; //!< True if ray intersected with a primitive
        float distance; //!< Distance to intersection point (only valid if hit = true)
        uint primitive_UUID; //!< UUID of intersected primitive (only valid if hit = true)
        helios::vec3 intersection_point; //!< World coordinates of intersection (only valid if hit = true)
        helios::vec3 normal; //!< Surface normal at intersection (only valid if hit = true)
        float path_length; //!< Unobstructed path length through voxel (t_max - max(0, t_min)) for LiDAR processing

        HitResult() : hit(false), distance(-1.0f), primitive_UUID(0), intersection_point(0, 0, 0), normal(0, 0, 0), path_length(0.0f) {
        }
    };

    /**
     * \brief Structure for batch ray-tracing statistics
     */
    struct RayTracingStats {
        size_t total_rays_cast; //!< Total number of rays processed
        size_t total_hits; //!< Total number of ray-primitive intersections
        size_t bvh_nodes_visited; //!< Total BVH nodes visited during traversal
        float average_ray_distance; //!< Average distance of successful ray hits

        RayTracingStats() : total_rays_cast(0), total_hits(0), bvh_nodes_visited(0), average_ray_distance(0.0f) {
        }
    };

    /**
     * \brief Ray streaming for efficient GPU processing
     * Processes multiple rays simultaneously for better GPU utilization
     */
    static constexpr size_t WARP_SIZE = 32; //!< CUDA warp size for optimal batching
    static constexpr size_t RAY_BATCH_SIZE = 1024; //!< Optimal batch size for ray processing

    /**
     * \brief Optimized ray packet for streaming processing
     * Organizes ray data in Structure-of-Arrays format for better memory access
     */
    struct RayPacket {
        // Ray data (Structure-of-Arrays layout)
        std::vector<helios::vec3> origins; //!< Ray origin points (contiguous)
        std::vector<helios::vec3> directions; //!< Ray direction vectors (contiguous)
        std::vector<float> max_distances; //!< Maximum ray distances (contiguous)
        std::vector<std::vector<uint>> target_UUIDs; //!< Target primitive UUIDs per ray

        // Results (output)
        std::vector<HitResult> results; //!< Intersection results (output)

        size_t ray_count = 0; //!< Number of rays in this packet

        RayPacket() = default;

        /**
         * \brief Initialize packet with specified capacity
         */
        void reserve(size_t capacity) {
            origins.reserve(capacity);
            directions.reserve(capacity);
            max_distances.reserve(capacity);
            target_UUIDs.reserve(capacity);
            results.reserve(capacity);
        }

        /**
         * \brief Add a ray to the packet
         */
        void addRay(const RayQuery &query) {
            origins.push_back(query.origin);
            directions.push_back(query.direction);
            max_distances.push_back(query.max_distance);
            target_UUIDs.push_back(query.target_UUIDs);
            results.emplace_back(); // Initialize empty result
            ray_count++;
        }

        /**
         * \brief Clear all data
         */
        void clear() {
            origins.clear();
            directions.clear();
            max_distances.clear();
            target_UUIDs.clear();
            results.clear();
            ray_count = 0;
        }

        /**
         * \brief Convert packet back to individual RayQuery objects
         */
        [[nodiscard]] std::vector<RayQuery> toRayQueries() const {
            std::vector<RayQuery> queries;
            queries.reserve(ray_count);
            for (size_t i = 0; i < ray_count; ++i) {
                queries.emplace_back(origins[i], directions[i], max_distances[i], target_UUIDs[i]);
            }
            return queries;
        }

        /**
         * \brief Get memory usage of the packet
         */
        [[nodiscard]] size_t getMemoryUsage() const {
            size_t base_memory = (origins.size() + directions.size()) * sizeof(helios::vec3) + max_distances.size() * sizeof(float) + results.size() * sizeof(HitResult);

            // Add memory for target UUIDs
            for (const auto &targets: target_UUIDs) {
                base_memory += targets.size() * sizeof(uint);
            }

            return base_memory;
        }
    };

    /**
     * \brief Streaming ray tracer interface
     * Enables efficient batch processing of ray packets
     */
    struct RayStream {
        std::vector<RayPacket> packets; //!< Collection of ray packets
        size_t current_packet = 0; //!< Current packet being processed
        size_t total_rays = 0; //!< Total rays across all packets

        /**
         * \brief Add rays to the stream, automatically batching into packets
         */
        void addRays(const std::vector<RayQuery> &queries) {
            for (const auto &query: queries) {
                // Create new packet if current one is full
                if (packets.empty() || packets.back().ray_count >= RAY_BATCH_SIZE) {
                    packets.emplace_back();
                    packets.back().reserve(RAY_BATCH_SIZE);
                }

                packets.back().addRay(query);
                total_rays++;
            }
        }

        /**
         * \brief Get all results from processed packets
         */
        [[nodiscard]] std::vector<HitResult> getAllResults() const {
            std::vector<HitResult> all_results;
            all_results.reserve(total_rays);

            for (const auto &packet: packets) {
                all_results.insert(all_results.end(), packet.results.begin(), packet.results.end());
            }

            return all_results;
        }

        /**
         * \brief Clear the stream
         */
        void clear() {
            packets.clear();
            current_packet = 0;
            total_rays = 0;
        }

        /**
         * \brief Get total memory usage of the stream
         */
        [[nodiscard]] size_t getMemoryUsage() const {
            size_t total_memory = 0;
            for (const auto &packet: packets) {
                total_memory += packet.getMemoryUsage();
            }
            return total_memory;
        }
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
     * \param[in] allow_spatial_culling If true, use tree-based BVH spatial culling for efficiency (default: true)
     * \return Vector of UUIDs that intersect with the query primitive
     */
    std::vector<uint> findCollisions(uint UUID, bool allow_spatial_culling = true);

    /**
     * \brief Find all primitives/objects that collide with any of the given primitives
     * \param[in] UUIDs Vector of query primitive UUIDs
     * \param[in] allow_spatial_culling If true, use tree-based BVH spatial culling for efficiency (default: true)
     * \return Vector of UUIDs that intersect with any of the query primitives
     */
    std::vector<uint> findCollisions(const std::vector<uint> &UUIDs, bool allow_spatial_culling = true);

    /**
     * \brief Find collisions between primitives and compound objects
     * \param[in] primitive_UUIDs Vector of primitive UUIDs to test
     * \param[in] object_IDs Vector of compound object IDs to test against
     * \param[in] allow_spatial_culling If true, use tree-based BVH spatial culling for efficiency (default: true)
     * \return Vector of UUIDs that intersect
     */
    std::vector<uint> findCollisions(const std::vector<uint> &primitive_UUIDs, const std::vector<uint> &object_IDs, bool allow_spatial_culling = true);

    /**
     * \brief Find collisions restricting both query and target geometry
     * \param[in] query_UUIDs Vector of primitive UUIDs to test for collisions
     * \param[in] query_object_IDs Vector of object IDs to test for collisions
     * \param[in] target_UUIDs Vector of primitive UUIDs to test against (empty = all primitives)
     * \param[in] target_object_IDs Vector of object IDs to test against (empty = all objects)
     * \param[in] allow_spatial_culling If true, use tree-based BVH spatial culling for efficiency (default: true)
     * \return Vector of UUIDs from the target set that intersect with any query primitive
     */
    std::vector<uint> findCollisions(const std::vector<uint> &query_UUIDs, const std::vector<uint> &query_object_IDs, const std::vector<uint> &target_UUIDs, const std::vector<uint> &target_object_IDs, bool allow_spatial_culling = true);

    // -------- GENERIC RAY-TRACING INTERFACE --------

    /**
     * \brief Cast a single ray and return detailed intersection information
     * \param[in] ray_query Ray parameters (origin, direction, max distance, target UUIDs)
     * \return HitResult containing intersection details
     */
    HitResult castRay(const RayQuery &ray_query);

    /**
     * \brief Cast a single ray with simplified parameters
     * \param[in] origin Ray origin point
     * \param[in] direction Ray direction vector (should be normalized)
     * \param[in] max_distance Maximum ray distance (negative = infinite)
     * \param[in] target_UUIDs Target primitive UUIDs (empty = all primitives)
     * \return HitResult containing intersection details
     */
    HitResult castRay(const helios::vec3 &origin, const helios::vec3 &direction, float max_distance = -1.0f, const std::vector<uint> &target_UUIDs = {});

    /**
     * \brief Cast multiple rays in batch for improved performance
     * \param[in] ray_queries Vector of ray parameters
     * \param[out] stats Optional ray-tracing statistics
     * \return Vector of HitResult corresponding to each input ray
     */
    std::vector<HitResult> castRays(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats = nullptr);

    // -------- OPTIMIZATION METHODS --------

    /**
     * \brief BVH optimization modes for performance improvements
     */
    enum class BVHOptimizationMode {
        SOA_UNCOMPRESSED //!< Structure-of-Arrays, full precision
    };

    /**
     * \brief Set BVH optimization mode for performance improvements
     * \param[in] mode Optimization mode (SOA_UNCOMPRESSED)
     */
    void setBVHOptimizationMode(BVHOptimizationMode mode);

    /**
     * \brief Get current BVH optimization mode
     * \return Current optimization mode
     */
    BVHOptimizationMode getBVHOptimizationMode() const;

    /**
     * \brief Cast rays using optimized Structure-of-Arrays BVH layout
     * \param[in] ray_queries Vector of ray parameters
     * \param[out] stats Optional ray-tracing statistics
     * \return Vector of HitResult corresponding to each input ray
     */
    std::vector<HitResult> castRaysOptimized(const std::vector<RayQuery> &ray_queries, RayTracingStats *stats = nullptr);

#ifdef HELIOS_CUDA_AVAILABLE
    /**
     * \brief Cast rays using GPU acceleration with warp-efficient kernels
     * \param[in] ray_queries Vector of ray queries to process
     * \param[out] stats Ray-tracing performance statistics
     * \return Vector of HitResult with optimal GPU performance
     * \note This method is only available when CUDA is enabled during compilation
     */
    std::vector<HitResult> castRaysGPU(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats);
#endif

    /**
     * \brief Cast ray packets using streaming interface for optimal GPU utilization
     * \param[in] ray_stream Ray stream containing batched ray packets
     * \param[out] stats Optional ray-tracing statistics
     * \return True if all packets processed successfully
     */
    bool processRayStream(RayStream &ray_stream, RayTracingStats *stats = nullptr);

    /**
     * \brief Get memory usage statistics for different BVH layouts
     * \return Structure containing memory usage for each optimization mode
     */
    struct MemoryUsageStats {
        size_t soa_memory_bytes = 0;
        size_t quantized_memory_bytes = 0;
        float quantized_reduction_percent = 0.0f; // Reduction vs SoA
    };
    MemoryUsageStats getBVHMemoryUsage() const;

    /**
     * \brief Perform grid-based ray intersection for voxel operations
     * \param[in] grid_center Center point of the 3D grid
     * \param[in] grid_size Size of the grid in x, y, z dimensions
     * \param[in] grid_divisions Number of divisions in x, y, z dimensions
     * \param[in] ray_queries Vector of rays to intersect with the grid
     * \return 4D vector indexed as [i][j][k][hit_index] containing HitResult objects for each hit in each voxel
     */
    std::vector<std::vector<std::vector<std::vector<HitResult>>>> performGridRayIntersection(const helios::vec3 &grid_center, const helios::vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<RayQuery> &ray_queries);

    /**
     * \brief Calculate ray path lengths through individual voxels for LiDAR processing
     * \param[in] scan_origin Single vec3 LiDAR scanner position
     * \param[in] ray_directions Vector of normalized ray directions (10K-10M rays)
     * \param[in] voxel_centers Vector of voxel center positions (8-1000 voxels)
     * \param[in] voxel_sizes Vector of voxel dimensions (width,height,depth)
     * \return Vector of HitResult vectors, one per voxel, with path_length populated for each ray intersection
     *
     * This method extends performGridRayIntersection to handle individual voxels with specific centers and sizes.
     * Uses ray-AABB intersection with path_length = t_max - max(0, t_min) for each intersection.
     * Performance target: 1M rays × 100 voxels in <2 seconds with OpenMP parallelization.
     */
    std::vector<std::vector<HitResult>> calculateVoxelPathLengths(const helios::vec3 &scan_origin, const std::vector<helios::vec3> &ray_directions, const std::vector<helios::vec3> &voxel_centers, const std::vector<helios::vec3> &voxel_sizes);

    /**
     * \brief Enhanced ray path length calculations with detailed statistics
     * \param[in] grid_center Center point of the 3D voxel grid
     * \param[in] grid_size Size of the grid in x, y, z dimensions
     * \param[in] grid_divisions Number of divisions in x, y, z dimensions
     * \param[in] ray_origins Vector of ray origin points
     * \param[in] ray_directions Vector of ray direction vectors (should be normalized)
     * \param[out] hit_results Vector of detailed hit results for each ray
     */
    void calculateRayPathLengthsDetailed(const helios::vec3 &grid_center, const helios::vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<helios::vec3> &ray_origins, const std::vector<helios::vec3> &ray_directions,
                                         std::vector<HitResult> &hit_results);

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

    // -------- PRIMITIVE SLICING OPERATIONS --------

    /**
     * \brief Slice a primitive (patch or triangle) into multiple triangles using a planar face
     * \param[in] UUID Primitive UUID to slice
     * \param[in] voxel_face_vertices Vector of ≥3 points defining the slicing plane
     * \param[in] warnings Warning aggregator for collecting non-fatal warnings
     * \return Vector of new triangle UUIDs created by slicing
     */
    std::vector<uint> slicePrimitive(uint UUID, const std::vector<helios::vec3> &voxel_face_vertices, helios::WarningAggregator &warnings);

    /**
     * \brief Slice primitives along a regular 3D grid
     * \param[in] UUIDs Primitives to slice
     * \param[in] grid_center Center of voxel grid
     * \param[in] grid_size Grid dimensions (x, y, z)
     * \param[in] grid_divisions Grid cells (x, y, z)
     * \return All resulting triangle UUIDs (sliced + unmodified)
     * \note Creates "cell_ID" primitive data (-1 for outside grid)
     * \note Populates internal grid_cells structure
     */
    std::vector<uint> slicePrimitivesUsingGrid(const std::vector<uint> &UUIDs, const helios::vec3 &grid_center, const helios::vec3 &grid_size, const helios::int3 &grid_divisions);

    /**
     * \brief Calculate voxel-primitive intersections using OpenMP parallelization
     * \param[in] UUIDs Optional primitive UUIDs (empty = all primitives)
     * \note Requires actual voxel primitives in Context
     * \note Creates "inside_UUIDs" primitive data on voxels
     * \note Uses OpenMP for CPU parallel execution
     */
    void calculatePrimitiveVoxelIntersection(const std::vector<uint> &UUIDs = {});

    // -------- GEOMETRIC UTILITY FUNCTIONS --------

    /**
     * \brief Calculate 3D line intersection point
     * \param[in] line1_point Point on first line
     * \param[in] line1_direction Direction of first line
     * \param[in] line2_point Point on second line
     * \param[in] line2_direction Direction of second line
     * \return Intersection point (assumes lines intersect)
     */
    helios::vec3 linesIntersection(const helios::vec3 &line1_point, const helios::vec3 &line1_direction, const helios::vec3 &line2_point, const helios::vec3 &line2_direction) const;

    /**
     * \brief Compare floats with dual tolerance (absolute + relative)
     * \param[in] a First value
     * \param[in] b Second value
     * \param[in] absTol Absolute tolerance
     * \param[in] relTol Relative tolerance
     * \return True if |a-b| ≤ absTol OR |a-b|/max(|a|,|b|) ≤ relTol
     */
    bool approxSame(float a, float b, float absTol, float relTol) const;

    /**
     * \brief Compare vec3's element-wise with absolute tolerance
     */
    bool approxSame(const helios::vec3 &a, const helios::vec3 &b, float absTol) const;

    /**
     * \brief Interpolate UV texture coordinates to slice point
     * \param[in] p1 First edge vertex position
     * \param[in] uv1 First edge vertex UV
     * \param[in] p2 Second edge vertex position
     * \param[in] uv2 Second edge vertex UV
     * \param[in] ps Slice point position on edge
     * \return Interpolated UV coordinates at slice point
     */
    helios::vec2 interpolate_texture_UV_to_slice_point(const helios::vec3 &p1, const helios::vec2 &uv1, const helios::vec3 &p2, const helios::vec2 &uv2, const helios::vec3 &ps) const;

    // -------- VOXEL RAY PATH LENGTH CALCULATIONS --------

    /**
     * \brief Calculate ray path lengths and transmission statistics for a voxel grid
     * \param[in] grid_center Center point of the 3D voxel grid
     * \param[in] grid_size Size of the grid in x, y, z dimensions
     * \param[in] grid_divisions Number of divisions in x, y, z dimensions
     * \param[in] ray_origins Vector of ray origin points
     * \param[in] ray_directions Vector of ray direction vectors (should be normalized)
     */
    void calculateVoxelRayPathLengths(const helios::vec3 &grid_center, const helios::vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<helios::vec3> &ray_origins, const std::vector<helios::vec3> &ray_directions);

    /**
     * \brief Set transmission probability counts for a voxel
     * \param[in] P_denom Number of rays that reached this voxel (denominator)
     * \param[in] P_trans Number of rays that were transmitted through this voxel (numerator)
     * \param[in] ijk Voxel indices (i,j,k)
     */
    void setVoxelTransmissionProbability(int P_denom, int P_trans, const helios::int3 &ijk);

    /**
     * \brief Get transmission probability counts for a voxel
     * \param[in] ijk Voxel indices (i,j,k)
     * \param[out] P_denom Number of rays that reached this voxel
     * \param[out] P_trans Number of rays that were transmitted through this voxel
     */
    void getVoxelTransmissionProbability(const helios::int3 &ijk, int &P_denom, int &P_trans) const;

    /**
     * \brief Set average ray propagation distance for a voxel
     * \param[in] r_bar Average ray propagation distance in meters
     * \param[in] ijk Voxel indices (i,j,k)
     */
    void setVoxelRbar(float r_bar, const helios::int3 &ijk);

    /**
     * \brief Get average ray propagation distance for a voxel
     * \param[in] ijk Voxel indices (i,j,k)
     * \return Average ray propagation distance in meters
     */
    float getVoxelRbar(const helios::int3 &ijk) const;

    /**
     * \brief Get ray classification counts for a voxel (for Beer's law calculations)
     * \param[in] ijk Voxel indices (i,j,k)
     * \param[out] hit_before Number of rays that hit geometry BEFORE entering the voxel
     * \param[out] hit_after Number of rays that hit geometry AFTER entering the voxel
     * \param[out] hit_inside Number of rays that hit geometry INSIDE the voxel
     */
    void getVoxelRayHitCounts(const helios::int3 &ijk, int &hit_before, int &hit_after, int &hit_inside) const;

    /**
     * \brief Get individual ray path lengths for a voxel (for Beer's law calculations)
     * \param[in] ijk Voxel indices (i,j,k)
     * \return Vector of individual path lengths for rays that passed through the voxel
     */
    std::vector<float> getVoxelRayPathLengths(const helios::int3 &ijk) const;

    /**
     * \brief Clear all voxel ray statistics data
     */
    void clearVoxelData();

    // -------- FEATURE 4: COLLISION MINIMIZATION --------

    /**
     * \brief Optimize layout of primitives/objects to minimize collisions
     * \param[in] UUIDs Vector of primitive/object UUIDs to optimize
     * \param[in] learning_rate Step size for optimization algorithm
     * \param[in] max_iterations Maximum number of optimization iterations
     * \return Final collision count after optimization
     */
    int optimizeLayout(const std::vector<uint> &UUIDs, float learning_rate = 0.01f, int max_iterations = 1000);

    // -------- SPATIAL OPTIMIZATION --------

    /**
     * \brief Find collisions within a specified distance threshold
     * \param[in] query_UUIDs Vector of primitive UUIDs to test for collisions
     * \param[in] target_UUIDs Vector of primitive UUIDs to test against
     * \param[in] max_distance Maximum distance for collision consideration
     * \return Vector of collision pairs within distance threshold
     */
    std::vector<std::pair<uint, uint>> findCollisionsWithinDistance(const std::vector<uint> &query_UUIDs, const std::vector<uint> &target_UUIDs, float max_distance);

    /**
     * \brief Set maximum distance for collision detection queries
     * \param[in] distance Maximum distance threshold (meters)
     */
    void setMaxCollisionDistance(float distance);

    /**
     * \brief Get current maximum collision distance
     * \return Current maximum distance threshold
     */
    [[nodiscard]] float getMaxCollisionDistance() const;

    /**
     * \brief Filter geometry by spatial proximity for efficient collision detection
     * \param[in] query_center Center point for spatial filtering
     * \param[in] max_radius Maximum radius for including geometry
     * \param[in] candidate_UUIDs Optional list of candidate UUIDs (empty = all geometry)
     * \return Vector of UUIDs within the specified radius
     */
    std::vector<uint> filterGeometryByDistance(const helios::vec3 &query_center, float max_radius, const std::vector<uint> &candidate_UUIDs = {});

    /**
     * \brief Determine the distance to the nearest primitive in any direction, filtered by direction component
     * \param[in] origin Origin location from which to search
     * \param[in] direction Unit direction vector to filter "forward" surfaces (dot product > 0)
     * \param[in] candidate_UUIDs Vector of primitive UUIDs to consider for intersection
     * \param[out] distance Distance to the nearest primitive (only valid if return is true)
     * \param[out] obstacle_direction Direction from origin to nearest obstacle (only valid if return is true)
     * \return True if a primitive is found that is "in front" of the direction vector, false otherwise
     *
     * This method finds the nearest solid surface in any direction from the origin point, but only
     * considers surfaces that are "in front" of the growth direction. A surface is considered "in front"
     * if the vector from origin to the closest point on the surface has a positive dot product with
     * the input direction vector.
     */
    bool findNearestPrimitiveDistance(const helios::vec3 &origin, const helios::vec3 &direction, const std::vector<uint> &candidate_UUIDs, float &distance, helios::vec3 &obstacle_direction);

    /**
     * \brief Find the nearest solid obstacle within a cone using ray casting
     * \param[in] apex Cone apex position
     * \param[in] axis Central axis of the cone (normalized)
     * \param[in] half_angle Half-angle of cone in radians
     * \param[in] height Maximum detection distance (cone height)
     * \param[in] candidate_UUIDs Vector of primitive UUIDs to consider as solid obstacles
     * \param[out] distance Distance to nearest obstacle (only valid if return is true)
     * \param[out] obstacle_direction Direction from apex to nearest obstacle (only valid if return is true)
     * \param[in] num_rays Number of rays to cast for detection (default: 64)
     * \return True if a solid obstacle is found within the cone, false otherwise
     *
     * This method casts rays within a cone to find the nearest solid obstacle. Unlike the soft
     * collision avoidance, this uses a smaller cone angle and returns the exact distance to the
     * nearest primitive surface using accurate ray-primitive intersection tests.
     */
    bool findNearestSolidObstacleInCone(const helios::vec3 &apex, const helios::vec3 &axis, float half_angle, float height, const std::vector<uint> &candidate_UUIDs, float &distance, helios::vec3 &obstacle_direction, int num_rays = 64);

    //! Find nearest solid obstacle in cone with tree identification for per-tree BVH optimization
    /**
     * \brief Find the nearest solid obstacle within a cone-shaped detection volume with tree identification
     *
     * This method is optimized for per-tree BVH systems by identifying the querying tree through
     * plant primitive UUIDs, enabling efficient spatial filtering of collision candidates.
     *
     * \param[in] apex Location of the cone apex (plant growth point)
     * \param[in] axis Central axis direction of the cone (growth direction, normalized)
     * \param[in] half_angle Half-angle of the cone in radians
     * \param[in] height Height of the cone (detection distance)
     * \param[in] candidate_UUIDs Vector of obstacle primitive UUIDs to check for collisions
     * \param[in] plant_primitives Vector of plant primitive UUIDs to identify the querying tree
     * \param[out] distance Distance to the nearest obstacle surface
     * \param[out] obstacle_direction Direction vector from apex toward the nearest obstacle point
     * \param[in] num_rays Number of rays to cast within the cone for sampling
     * \return True if an obstacle was found within the cone, false otherwise
     */
    bool findNearestSolidObstacleInCone(const helios::vec3 &apex, const helios::vec3 &axis, float half_angle, float height, const std::vector<uint> &candidate_UUIDs, const std::vector<uint> &plant_primitives, float &distance,
                                        helios::vec3 &obstacle_direction, int num_rays = 64);


    // -------- BVH MANAGEMENT --------

    /**
     * \brief Build or rebuild the BVH acceleration structure
     * \param[in] UUIDs Optional vector of specific UUIDs to include (empty = all primitives)
     */
    void buildBVH(const std::vector<uint> &UUIDs = {});

    /**
     * \brief Efficiently update BVH with new geometry (avoids full rebuild when possible)
     * \param[in] UUIDs Vector of UUIDs to include in BVH
     * \param[in] force_rebuild Force complete rebuild even if geometry hasn't changed significantly
     */
    void updateBVH(const std::vector<uint> &UUIDs, bool force_rebuild = false);

    /**
     * \brief Mark specific geometry as static (will be cached for efficiency)
     * \param[in] UUIDs Vector of primitive UUIDs that represent static obstacles
     */
    void setStaticGeometry(const std::vector<uint> &UUIDs);

    /**
     * \brief Force a complete rebuild of the BVH
     */
    void rebuildBVH();

    /**
     * \brief Disable automatic BVH rebuilds (caller must manually manage rebuilds)
     */
    void disableAutomaticBVHRebuilds();

    /**
     * \brief Enable automatic BVH rebuilds (default behavior)
     */
    void enableAutomaticBVHRebuilds();

    /**
     * \brief Enable hierarchical BVH with separate static and dynamic geometry
     */
    void enableHierarchicalBVH();

    /**
     * \brief Disable hierarchical BVH (use single unified BVH)
     */
    void disableHierarchicalBVH();

    /**
     * \brief Build static BVH for obstacles and non-growing geometry
     *
     * Builds the static BVH using geometry set by setStaticGeometry().
     * Should be called after enabling hierarchical BVH and setting static geometry.
     */
    void buildStaticBVH();

    /**
     * \brief Enable tree-based BVH isolation for spatially separated trees
     * \param[in] isolation_distance Spatial distance threshold for tree isolation
     *
     * When enabled, creates separate BVHs for each tree object that are spatially
     * separated by at least isolation_distance. This provides linear scaling for
     * scenarios with multiple non-interacting trees.
     */
    void enableTreeBasedBVH(float isolation_distance = 5.0f);

    /**
     * \brief Disable tree-based BVH isolation
     */
    void disableTreeBasedBVH();

    /**
     * \brief Check if tree-based BVH isolation is enabled
     * \return True if tree-based BVH is enabled, false otherwise
     */
    [[nodiscard]] bool isTreeBasedBVHEnabled() const;

    /**
     * \brief Initialize spatial grid for fast static obstacle lookups
     */
    void initializeObstacleSpatialGrid();

    /**
     * \brief Register a tree object for per-tree BVH isolation
     * \param[in] tree_object_id Object ID of the tree
     * \param[in] tree_primitives Primitive UUIDs belonging to this tree
     *
     * Associates primitives with a specific tree for isolated collision detection.
     * Should be called before plant growth begins.
     */
    void registerTree(uint tree_object_id, const std::vector<uint> &tree_primitives);

    /**
     * \brief Set static obstacle geometry for per-tree collision detection
     * \param[in] obstacle_primitives Primitive UUIDs that represent static obstacles
     *
     * These primitives will be included in ALL tree collision detection queries.
     * Should include ground, buildings, and other permanent obstacles.
     */
    void setStaticObstacles(const std::vector<uint> &obstacle_primitives);

    /**
     * \brief Get relevant geometry for tree-based collision detection
     * \param[in] query_position Position of the collision query
     * \param[in] query_primitives Primitives making the query (to identify source tree)
     * \return Vector of primitive UUIDs that should be considered for collision
     *
     * Returns geometry that includes:
     * - Static obstacles (always relevant)
     * - Source tree's own geometry
     * - Nearby trees within interaction distance
     */
    std::vector<uint> getRelevantGeometryForTree(const helios::vec3 &query_position, const std::vector<uint> &query_primitives = {}, float max_distance = 15.0f);

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
    static int selfTest(int argc, char **argv);

private:
    //! Pointer to the Helios context
    helios::Context *context;

    //! Flag to enable/disable GPU acceleration
    bool gpu_acceleration_enabled;

    //! Flag to enable/disable console messages
    bool printmessages;

    // -------- THREAD-SAFE PRIMITIVE CACHE --------

    /**
     * \brief Cached primitive data for thread-safe intersection tests
     */
    struct CachedPrimitive {
        helios::PrimitiveType type;
        std::vector<helios::vec3> vertices;

        CachedPrimitive() : type(helios::PRIMITIVE_TYPE_TRIANGLE) {
        }
        CachedPrimitive(helios::PrimitiveType t, const std::vector<helios::vec3> &v) : type(t), vertices(v) {
        }
    };

    //! Cache of primitive data for thread-safe access (maps primitive_id -> cached data)
    std::unordered_map<uint, CachedPrimitive> primitive_cache;

    /**
     * \brief Build primitive cache for thread-safe ray casting
     */
    void buildPrimitiveCache();

    /**
     * \brief Thread-safe primitive intersection method using cached data
     */
    HitResult intersectPrimitiveThreadSafe(const helios::vec3 &origin, const helios::vec3 &direction, uint primitive_id, float max_distance);

    /**
     * \brief Fast triangle intersection test (thread-safe)
     */
    bool triangleIntersect(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &v0, const helios::vec3 &v1, const helios::vec3 &v2, float &distance);

    bool patchIntersect(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &v0, const helios::vec3 &v1, const helios::vec3 &v2, const helios::vec3 &v3, float &distance);

    // -------- BVH DATA STRUCTURES --------

    /**
     * \brief Original BVH node structure for CPU storage (legacy support)
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

    /**
     * \brief Structure-of-Arrays (SoA) BVH layout for optimized memory access
     * Improves cache efficiency and enables vectorization
     */
    struct BVHNodesSoA {
        // Hot data: frequently accessed during traversal (cache-friendly grouping)
        std::vector<helios::vec3> aabb_mins; //!< All AABB minimum corners (contiguous)
        std::vector<helios::vec3> aabb_maxs; //!< All AABB maximum corners (contiguous)
        std::vector<uint32_t> left_children; //!< All left child indices (contiguous)
        std::vector<uint32_t> right_children; //!< All right child indices (contiguous)

        // Cold data: accessed less frequently (separate for better cache utilization)
        std::vector<uint32_t> primitive_starts; //!< Starting indices in primitive array
        std::vector<uint32_t> primitive_counts; //!< Number of primitives per leaf node
        std::vector<uint8_t> is_leaf_flags; //!< Leaf node flags (packed to single bytes)

        // Metadata
        size_t node_count = 0; //!< Total number of nodes

        BVHNodesSoA() = default;

        /**
         * \brief Initialize SoA structure with specified capacity
         */
        void reserve(size_t capacity) {
            aabb_mins.reserve(capacity);
            aabb_maxs.reserve(capacity);
            left_children.reserve(capacity);
            right_children.reserve(capacity);
            primitive_starts.reserve(capacity);
            primitive_counts.reserve(capacity);
            is_leaf_flags.reserve(capacity);
        }

        /**
         * \brief Clear all data
         */
        void clear() {
            aabb_mins.clear();
            aabb_maxs.clear();
            left_children.clear();
            right_children.clear();
            primitive_starts.clear();
            primitive_counts.clear();
            is_leaf_flags.clear();
            node_count = 0;
        }

        /**
         * \brief Get memory usage in bytes
         */
        [[nodiscard]] size_t getMemoryUsage() const {
            return (aabb_mins.size() + aabb_maxs.size()) * sizeof(helios::vec3) + (left_children.size() + right_children.size() + primitive_starts.size() + primitive_counts.size()) * sizeof(uint32_t) + is_leaf_flags.size() * sizeof(uint8_t);
        }
    };


    //! Vector of BVH nodes (linearized tree structure) - LEGACY
    std::vector<BVHNode> bvh_nodes;

    //! Index tracking for pre-allocated BVH node array
    size_t next_available_node_index;

    //! Structure-of-Arrays BVH layout
    BVHNodesSoA bvh_nodes_soa;

    //! Current optimization mode
    BVHOptimizationMode bvh_optimization_mode = BVHOptimizationMode::SOA_UNCOMPRESSED;

    //! Primitive indices sorted by BVH construction
    std::vector<uint> primitive_indices;

    //! Cached primitive bounding boxes (optimization for BVH construction)
    std::unordered_map<uint, std::pair<helios::vec3, helios::vec3>> primitive_aabbs_cache;

    //! OPTIMIZATION: Cache validity tracking to avoid rebuilding unchanged primitive AABBs
    std::unordered_set<uint> dirty_primitive_cache;

    //! Grid intersection results [i][j][k] = vector of UUIDs
    std::vector<std::vector<std::vector<std::vector<uint>>>> grid_cells;

    //! Grid parameters
    helios::vec3 grid_center;
    helios::vec3 grid_size;
    helios::int3 grid_divisions;

    // -------- VOXEL RAY STATISTICS DATA --------

    //! Voxel ray count statistics [i][j][k] = number of rays reaching voxel (P_denom)
    std::vector<std::vector<std::vector<int>>> voxel_ray_counts;

    //! Voxel transmission count statistics [i][j][k] = number of rays transmitted (P_trans)
    std::vector<std::vector<std::vector<int>>> voxel_transmitted;

    //! Voxel cumulative ray path lengths [i][j][k] = sum of path lengths for r_bar calculation
    std::vector<std::vector<std::vector<float>>> voxel_path_lengths;

    //! Voxel ray classification counts [i][j][k] = rays hit before entering voxel
    std::vector<std::vector<std::vector<int>>> voxel_hit_before;

    //! Voxel ray classification counts [i][j][k] = rays hit after entering voxel
    std::vector<std::vector<std::vector<int>>> voxel_hit_after;

    //! Voxel ray classification counts [i][j][k] = rays hit inside voxel
    std::vector<std::vector<std::vector<int>>> voxel_hit_inside;

    //! Individual ray path lengths [i][j][k] = vector of individual path lengths
    std::vector<std::vector<std::vector<std::vector<float>>>> voxel_individual_path_lengths;

    // -------- OPTIMIZED FLAT ARRAY DATA (Structure-of-Arrays) --------

    //! Flat array storage for voxel statistics (cache-friendly layout)
    std::vector<int> voxel_ray_counts_flat;
    std::vector<int> voxel_transmitted_flat;
    std::vector<float> voxel_path_lengths_flat;
    std::vector<int> voxel_hit_before_flat;
    std::vector<int> voxel_hit_after_flat;
    std::vector<int> voxel_hit_inside_flat;

    //! Individual path lengths stored as flat arrays with offset indexing
    std::vector<float> voxel_individual_path_lengths_flat;
    std::vector<size_t> voxel_individual_path_offsets; // Start offset for each voxel
    std::vector<size_t> voxel_individual_path_counts; // Count for each voxel

    //! Flag to use optimized flat arrays instead of nested vectors
    bool use_flat_arrays;

    //! Flag to track if voxel data structures are initialized
    bool voxel_data_initialized;

    //! Current voxel grid parameters (for validation)
    helios::vec3 voxel_grid_center;
    helios::vec3 voxel_grid_size;
    helios::int3 voxel_grid_divisions;

    // -------- SPATIAL OPTIMIZATION DATA --------

    //! Maximum distance for collision detection queries (meters)
    float max_collision_distance;

    //! Spatial partitioning hash for fast proximity queries
    std::unordered_map<uint, helios::vec3> primitive_centroids_cache;

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

    //! Cache for static geometry (obstacles, fixed structures)
    std::set<uint> static_geometry_cache;

    //! Last built BVH geometry set (for cache validation)
    std::set<uint> last_bvh_geometry;

    //! Flag to track if BVH needs rebuilding
    bool bvh_dirty;

    //! Flag to track if SoA structures need rebuilding
    bool soa_dirty;

    //! Flag to control automatic BVH rebuilds (default: true)
    bool automatic_bvh_rebuilds;

    //! Flag to skip BVH currency checks during batch operations (performance optimization)
    mutable bool batch_mode_skip_bvh_check;

    // -------- HIERARCHICAL BVH DATA STRUCTURES --------

    //! Flag to enable hierarchical BVH (separate static/dynamic BVHs)
    bool hierarchical_bvh_enabled;

    //! Static BVH nodes (persistent obstacles like ground, buildings)
    std::vector<BVHNode> static_bvh_nodes;

    //! Static BVH primitives
    std::vector<uint> static_bvh_primitives;

    //! Flag indicating if static BVH is built and valid
    bool static_bvh_valid;

    //! Last geometry set used for static BVH
    std::set<uint> last_static_bvh_geometry;

    //! Helper method for hierarchical BVH updates
    void updateHierarchicalBVH(const std::set<uint> &requested_geometry, bool force_rebuild);

    // -------- TREE-BASED BVH DATA STRUCTURES --------

    // Forward declaration
    struct TreeBVH;

    //! Flag to enable per-tree BVH isolation for better scaling
    bool tree_based_bvh_enabled;

    //! Spatial distance threshold for tree isolation (objects closer than this interact)
    float tree_isolation_distance;

    //! Cache of object-to-tree mappings for efficient lookup
    std::unordered_map<uint, uint> object_to_tree_map;

    //! Static obstacle primitives (always included in collision detection)
    std::vector<uint> static_obstacle_primitives;

    //! Spatial hash grid for fast static obstacle lookups to avoid O(N) distance checks
    struct ObstacleSpatialGrid {
        float cell_size;
        std::unordered_map<int64_t, std::vector<uint>> grid_cells;

        [[nodiscard]] int64_t getGridKey(float x, float y) const {
            auto grid_x = static_cast<int32_t>(std::floor(x / cell_size));
            auto grid_y = static_cast<int32_t>(std::floor(y / cell_size));
            return (static_cast<int64_t>(grid_x) << 32) | static_cast<uint32_t>(grid_y);
        }

        [[nodiscard]] std::vector<uint> getRelevantObstacles(const helios::vec3 &position, float radius) const;
    };

    mutable ObstacleSpatialGrid obstacle_spatial_grid;
    bool obstacle_spatial_grid_initialized;

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

    /**
     * \brief Spatial hash grid for efficient gap detection
     */
    struct SpatialHashGrid {
        struct Cell {
            std::vector<size_t> sample_indices; //!< Indices of samples in this cell
        };

        std::vector<std::vector<Cell>> grid; //!< 2D grid for spherical coordinates
        int theta_resolution; //!< Number of theta divisions
        int phi_resolution; //!< Number of phi divisions
        float theta_step; //!< Angular step size in theta
        float phi_step; //!< Angular step size in phi

        explicit SpatialHashGrid(int theta_res = 32, int phi_res = 16) : theta_resolution(theta_res), phi_resolution(phi_res) {
            grid.resize(theta_resolution);
            for (auto &row: grid) {
                row.resize(phi_resolution);
            }
            theta_step = M_PI / theta_resolution;
            phi_step = 2.0f * M_PI / phi_resolution;
        }

        void clear() {
            for (auto &row: grid) {
                for (auto &cell: row) {
                    cell.sample_indices.clear();
                }
            }
        }

        [[nodiscard]] std::pair<int, int> getGridIndex(const helios::vec3 &direction) const {
            // Convert direction to spherical coordinates
            float theta = acosf(std::max(-1.0f, std::min(1.0f, direction.z)));
            float phi = atan2f(direction.y, direction.x);
            if (phi < 0)
                phi += 2.0f * M_PI;

            int theta_idx = std::min((int) (theta / theta_step), theta_resolution - 1);
            int phi_idx = std::min((int) (phi / phi_step), phi_resolution - 1);

            return {theta_idx, phi_idx};
        }

        void addSample(size_t sample_idx, const helios::vec3 &direction) {
            auto [theta_idx, phi_idx] = getGridIndex(direction);
            grid[theta_idx][phi_idx].sample_indices.push_back(sample_idx);
        }

        [[nodiscard]] std::vector<size_t> getNearbyIndices(const helios::vec3 &direction, int radius = 1) const {
            auto [center_theta, center_phi] = getGridIndex(direction);
            std::vector<size_t> nearby_indices;

            for (int dt = -radius; dt <= radius; dt++) {
                for (int dp = -radius; dp <= radius; dp++) {
                    int theta_idx = center_theta + dt;
                    int phi_idx = (center_phi + dp + phi_resolution) % phi_resolution;

                    if (theta_idx >= 0 && theta_idx < theta_resolution) {
                        const auto &cell = grid[theta_idx][phi_idx];
                        nearby_indices.insert(nearby_indices.end(), cell.sample_indices.begin(), cell.sample_indices.end());
                    }
                }
            }

            return nearby_indices;
        }
    };

    /**
     * \brief Tree-based BVH structure for spatial isolation
     */
    struct TreeBVH {
        uint tree_object_id; //!< Object ID of the tree
        helios::vec3 tree_center; //!< Spatial center of the tree
        float tree_radius; //!< Spatial radius of the tree
        std::vector<BVHNode> nodes; //!< BVH nodes for this tree
        std::vector<uint> primitive_indices; //!< Primitive indices for this tree
        BVHNodesSoA soa_structure; //!< Optimized SoA structure for this tree
        bool soa_dirty; //!< Whether SoA structure needs rebuilding

        TreeBVH() : tree_object_id(0), tree_center(0, 0, 0), tree_radius(0), soa_dirty(true) {
        }

        void clear() {
            nodes.clear();
            primitive_indices.clear();
            soa_structure = BVHNodesSoA();
            soa_dirty = true;
        }
    };

    //! Map from tree object ID to tree-specific BVH
    std::unordered_map<uint, TreeBVH> tree_bvh_map;

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
     * \brief Recursively builds the BVH (Bounding Volume Hierarchy) for the given primitives.
     *
     * This function initializes and processes BVH nodes for efficient collision detection
     * or spatial queries, using recursive subdivision based on a surface area heuristic.
     *
     * \param[in] node_index Index of the current BVH node being processed
     * \param[in] primitive_start Starting index of the primitives for the current node
     * \param[in] primitive_count Number of primitives to consider for the current node
     * \param[in] depth Current depth of the recursion
     */
    void buildBVHRecursive(uint node_index, size_t primitive_start, size_t primitive_count, int depth);

    /**
     * \brief Traverse BVH on CPU to find intersecting primitives
     * \param[in] query_aabb_min Minimum corner of query bounding box
     * \param[in] query_aabb_max Maximum corner of query bounding box
     * \return Vector of intersecting primitive UUIDs
     */
    std::vector<uint> traverseBVH_CPU(const helios::vec3 &query_aabb_min, const helios::vec3 &query_aabb_max);

#ifdef HELIOS_CUDA_AVAILABLE
    /**
     * \brief Traverse BVH on GPU to find intersecting primitives
     * \param[in] query_aabb_min Minimum corner of query bounding box
     * \param[in] query_aabb_max Maximum corner of query bounding box
     * \return Vector of intersecting primitive UUIDs
     * \note This method is only available when CUDA is enabled during compilation
     */
    std::vector<uint> traverseBVH_GPU(const helios::vec3 &query_aabb_min, const helios::vec3 &query_aabb_max);
#endif

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
    bool rayAABBIntersect(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max, float &t_min, float &t_max) const;

    /**
     * \brief SIMD-optimized ray-AABB intersection test using AVX2/SSE
     * \param[in] ray_origins Array of ray origins (size 4 or 8 for AVX)
     * \param[in] ray_directions Array of ray directions (size 4 or 8 for AVX)
     * \param[in] aabb_mins Array of AABB minimum points (size 4 or 8 for AVX)
     * \param[in] aabb_maxs Array of AABB maximum points (size 4 or 8 for AVX)
     * \param[out] t_mins Array of near intersection parameters
     * \param[out] t_maxs Array of far intersection parameters
     * \param[in] count Number of rays to process (must be 4 or 8)
     * \return Bitmask indicating which rays intersect their respective AABBs
     */
    uint32_t rayAABBIntersectSIMD(const helios::vec3 *ray_origins, const helios::vec3 *ray_directions, const helios::vec3 *aabb_mins, const helios::vec3 *aabb_maxs, float *t_mins, float *t_maxs, int count);

    /**
     * \brief SIMD-optimized BVH traversal for multiple rays
     * \param[in] ray_origins Array of ray origins
     * \param[in] ray_directions Array of ray directions
     * \param[in] count Number of rays to process
     * \param[out] results Array to store hit results
     */
    void traverseBVHSIMD(const helios::vec3 *ray_origins, const helios::vec3 *ray_directions, int count, HitResult *results);

    /**
     * \brief SIMD implementation helper for BVH traversal
     */
    void traverseBVHSIMDImpl(const helios::vec3 *ray_origins, const helios::vec3 *ray_directions, int count, HitResult *results);

    /**
     * \brief Test if a cone intersects an AABB
     * \param[in] cone Cone to test
     * \param[in] aabb_min Minimum corner of AABB
     * \param[in] aabb_max Maximum corner of AABB
     * \return True if cone and AABB intersect
     */
    bool coneAABBIntersect(const Cone &cone, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max);

    /**
     * \brief Fast cone-AABB intersection test optimized for high-throughput filtering
     * \param[in] cone Cone to test
     * \param[in] aabb_min Minimum corner of AABB
     * \param[in] aabb_max Maximum corner of AABB
     * \return True if cone and AABB intersect
     * \note This version uses aggressive early rejection for 99%+ geometry elimination
     */
    bool coneAABBIntersectFast(const Cone &cone, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max);

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
     * \brief Find nearest primitive intersection along a ray using BVH traversal
     * \param[in] origin Starting point of the ray
     * \param[in] direction Direction vector of the ray (normalized)
     * \param[in] candidate_UUIDs Set of primitive UUIDs to consider (empty = all primitives)
     * \param[out] nearest_distance Distance to nearest intersection (only valid if return is true)
     * \param[in] max_distance Maximum distance to test (negative = infinite)
     * \return True if any intersection found, false otherwise
     */
    bool findNearestRayIntersection(const helios::vec3 &origin, const helios::vec3 &direction, const std::set<uint> &candidate_UUIDs, float &nearest_distance, float max_distance = -1.0f);

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
     * \brief Get candidate primitives within cone region using BVH culling
     * \param[in] apex Cone apex position
     * \param[in] central_axis Central axis direction (normalized)
     * \param[in] half_angle Half-angle of cone in radians
     * \param[in] height Maximum distance to sample
     * \return Vector of primitive UUIDs that could intersect the cone
     */
    std::vector<uint> getCandidatePrimitivesInCone(const helios::vec3 &apex, const helios::vec3 &central_axis, float half_angle, float height);

    /**
     * \brief Get candidate primitives using spatial grid optimization
     * \param[in] cone Cone object for filtering
     * \param[in] apex Cone apex position
     * \param[in] central_axis Central axis direction (normalized)
     * \param[in] half_angle Half-angle of cone in radians
     * \param[in] height Maximum distance to sample
     * \return Vector of primitive UUIDs that could intersect the cone
     */
    std::vector<uint> getCandidatesUsingSpatialGrid(const Cone &cone, const helios::vec3 &apex, const helios::vec3 &central_axis, float half_angle, float height);

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

    // -------- VOXEL RAY PATH LENGTH HELPER METHODS --------

    /**
     * \brief Initialize voxel data structures for given grid parameters
     * \param[in] grid_center Center of voxel grid
     * \param[in] grid_size Size of voxel grid
     * \param[in] grid_divisions Number of divisions in each dimension
     */
    void initializeVoxelData(const helios::vec3 &grid_center, const helios::vec3 &grid_size, const helios::int3 &grid_divisions);

    /**
     * \brief CPU implementation of voxel ray path length calculations
     * \param[in] ray_origins Vector of ray origin points
     * \param[in] ray_directions Vector of ray direction vectors
     */
    void calculateVoxelRayPathLengths_CPU(const std::vector<helios::vec3> &ray_origins, const std::vector<helios::vec3> &ray_directions);

    /**
     * \brief GPU implementation of voxel ray path length calculations
     * \param[in] ray_origins Vector of ray origin points
     * \param[in] ray_directions Vector of ray direction vectors
     */
    void calculateVoxelRayPathLengths_GPU(const std::vector<helios::vec3> &ray_origins, const std::vector<helios::vec3> &ray_directions);

    /**
     * \brief Validate voxel grid indices
     * \param[in] ijk Voxel indices to validate
     * \return True if indices are valid
     */
    bool validateVoxelIndices(const helios::int3 &ijk) const;

    /**
     * \brief Calculate voxel AABB from grid parameters and indices
     * \param[in] ijk Voxel indices
     * \param[out] voxel_min Minimum corner of voxel AABB
     * \param[out] voxel_max Maximum corner of voxel AABB
     */
    void calculateVoxelAABB(const helios::int3 &ijk, helios::vec3 &voxel_min, helios::vec3 &voxel_max) const;

    /**
     * \brief Traverse voxel grid along ray using DDA algorithm
     * \param[in] ray_origin Ray origin point
     * \param[in] ray_direction Ray direction vector (normalized)
     * \return Vector of (voxel_index, path_length) pairs for intersected voxels
     */
    std::vector<std::pair<helios::int3, float>> traverseVoxelGrid(const helios::vec3 &ray_origin, const helios::vec3 &ray_direction) const;

    /**
     * \brief Convert 3D voxel indices to flat array index
     * \param[in] i X index
     * \param[in] j Y index
     * \param[in] k Z index
     * \return Flat array index
     */
    inline size_t flatIndex(int i, int j, int k) const {
        return static_cast<size_t>(i) * static_cast<size_t>(voxel_grid_divisions.y) * static_cast<size_t>(voxel_grid_divisions.z) + static_cast<size_t>(j) * static_cast<size_t>(voxel_grid_divisions.z) + static_cast<size_t>(k);
    }

    /**
     * \brief Convert 3D voxel indices to flat array index (int3 version)
     * \param[in] ijk Voxel indices
     * \return Flat array index
     */
    inline size_t flatIndex(const helios::int3 &ijk) const {
        return flatIndex(ijk.x, ijk.y, ijk.z);
    }

#ifdef HELIOS_CUDA_AVAILABLE
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
#endif

    /**
     * \brief Mark BVH as dirty (needs rebuilding)
     */
    void markBVHDirty();

    /**
     * \brief Perform incremental BVH update for small geometry changes
     * \param[in] added_geometry Set of primitive UUIDs to add
     * \param[in] removed_geometry Set of primitive UUIDs to remove
     * \param[in] final_geometry Final set of all geometry after changes
     */
    void incrementalUpdateBVH(const std::set<uint> &added_geometry, const std::set<uint> &removed_geometry, const std::set<uint> &final_geometry);

    /**
     * \brief Update primitive AABB cache for a single primitive
     * \param[in] uuid Primitive UUID to cache AABB for
     */
    void updatePrimitiveAABBCache(uint uuid);

    /**
     * \brief Perform optimized BVH rebuild using cached primitive AABBs
     * \param[in] final_geometry Set of all geometry to include in BVH
     */
    void optimizedRebuildBVH(const std::set<uint> &final_geometry);

    /**
     * \brief Validate that all required primitives exist in context
     * \param[in] UUIDs Vector of UUIDs to validate
     * \return True if all UUIDs are valid
     */
    bool validateUUIDs(const std::vector<uint> &UUIDs) const;

    /**
     * \brief Test ray-primitive intersection and compute exact intersection distance
     * \param[in] origin Ray origin point
     * \param[in] direction Ray direction vector (normalized)
     * \param[in] primitive_UUID UUID of the primitive to test
     * \param[out] distance Distance to intersection point (only valid if return is true)
     * \return True if ray intersects primitive, false otherwise
     */
    bool rayPrimitiveIntersection(const helios::vec3 &origin, const helios::vec3 &direction, uint primitive_UUID, float &distance) const;

    /**
     * \brief Process ray batch using CPU (sequential processing)
     * \param[in] ray_queries Input ray queries
     * \param[out] results Output hit results
     * \param[out] stats Ray tracing statistics
     */
    void castRaysCPU(const std::vector<RayQuery> &ray_queries, std::vector<HitResult> &results, RayTracingStats &stats);

#ifdef HELIOS_CUDA_AVAILABLE
    /**
     * \brief Process ray batch using GPU (parallel processing)
     * \param[in] ray_queries Input ray queries
     * \param[out] results Output hit results
     * \param[out] stats Ray tracing statistics
     */
    void castRaysGPU(const std::vector<RayQuery> &ray_queries, std::vector<HitResult> &results, RayTracingStats &stats);
#endif

    // -------- OPTIMIZED RAY TRACING PRIVATE METHODS --------

    /**
     * \brief Convert BVH between different layout formats
     */
    void convertBVHLayout(BVHOptimizationMode from_mode, BVHOptimizationMode to_mode);
    void ensureOptimizedBVH(); // Populate optimized BVH structures on demand

    /**
     * \brief Optimized ray casting implementations for different BVH layouts
     */
    std::vector<HitResult> castRaysSoA(const std::vector<RayQuery> &ray_queries, RayTracingStats &stats);

    /**
     * \brief Single ray traversal methods for optimized BVH layouts
     */
    HitResult castRaySoATraversal(const RayQuery &query, RayTracingStats &stats);

    /**
     * \brief Basic BVH traversal (fallback when optimized structures not available)
     * This method provides the critical early miss detection that was missing from brute-force approach
     */
    HitResult castRayBVHTraversal(const RayQuery &query);

    /**
     * \brief Optimized AABB intersection tests
     */
    inline bool aabbIntersectSoA(const helios::vec3 &ray_origin, const helios::vec3 &ray_direction, float max_distance, size_t node_index) const;

    /**
     * \brief Basic ray-AABB intersection test
     * Used by the basic BVH traversal for early miss detection
     */
    bool rayAABBIntersect(const helios::vec3 &ray_origin, const helios::vec3 &ray_direction, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max) const;

    /**
     * \brief Helper method for primitive intersection
     */
    HitResult intersectPrimitive(const RayQuery &query, uint primitive_id);

    /**
     * \brief Ray-AABB intersection test for voxel primitives
     * \param[in] origin Ray origin point
     * \param[in] direction Ray direction vector (should be normalized)
     * \param[in] aabb_min Minimum corner of axis-aligned bounding box
     * \param[in] aabb_max Maximum corner of axis-aligned bounding box
     * \param[out] distance Distance to intersection point
     * \return True if ray intersects the AABB, false otherwise
     */
    bool rayAABBIntersectPrimitive(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &aabb_min, const helios::vec3 &aabb_max, float &distance);

    // -------- RASTERIZATION-BASED COLLISION DETECTION --------

    /**
     * \brief Calculate optimal bin count for angular discretization
     * \param[in] cone_half_angle Cone half-angle in radians
     * \param[in] geometry_count Number of potentially intersecting primitives
     * \return Optimal number of bins for both theta and phi divisions
     */
    int calculateOptimalBinCount(float cone_half_angle, int geometry_count);

    /**
     * \brief Filter primitives using fast cone-AABB intersection with OpenMP parallelization
     * \param[in] cone Cone to test against
     * \param[in] primitive_uuids All primitive UUIDs to test
     * \return Vector of UUIDs that intersect the cone
     */
    std::vector<uint> filterPrimitivesParallel(const Cone &cone, const std::vector<uint> &primitive_uuids);

    /**
     * \brief Project geometry to angular bins using spherical coordinates
     * \param[in] cone Cone parameters for projection
     * \param[in] filtered_uuids Primitives that passed cone-AABB filter
     * \param[out] bins Angular bins to populate with coverage data
     */
    void projectGeometryToBins(const Cone &cone, const std::vector<uint> &filtered_uuids, AngularBins &bins);

    /**
     * \brief Find gaps in coverage map using flood-fill algorithm
     * \param[in] bins Angular bins with coverage data
     * \param[in] cone Cone parameters for scoring
     * \return Vector of gaps sorted by quality score
     */
    std::vector<Gap> findGapsInCoverageMap(const AngularBins &bins, const Cone &cone);

    /**
     * \brief Convert spherical coordinates to bin indices
     * \param[in] theta Azimuthal angle (0 to 2*PI)
     * \param[in] phi Polar angle (0 to cone_half_angle)
     * \param[in] bins Angular bin structure
     * \param[out] theta_bin Theta bin index
     * \param[out] phi_bin Phi bin index
     * \return True if angles are within valid cone range
     */
    bool sphericalCoordsToBinIndices(float theta, float phi, const AngularBins &bins, int &theta_bin, int &phi_bin);

    /**
     * \brief Convert 3D cartesian vector to spherical coordinates relative to cone axis
     * \param[in] vector 3D vector from cone apex
     * \param[in] cone_axis Normalized cone axis direction
     * \param[out] theta Azimuthal angle (0 to 2*PI)
     * \param[out] phi Polar angle (0 to PI)
     * \return Distance from cone apex
     */
    float cartesianToSphericalCone(const helios::vec3 &vector, const helios::vec3 &cone_axis, float &theta, float &phi);

    /**
     * \brief Serial version of geometry projection for small geometry sets or OpenMP fallback
     * \param[in] cone Cone parameters for projection
     * \param[in] filtered_uuids Primitives that passed cone-AABB filter
     * \param[out] bins Angular bins to populate with coverage data
     */
    void projectGeometryToBinsSerial(const Cone &cone, const std::vector<uint> &filtered_uuids, AngularBins &bins);

    /**
     * \brief Flood-fill algorithm to find contiguous gap regions
     * \param[in] bins Angular bins with coverage data
     * \param[in] start_theta Starting theta bin index
     * \param[in] start_phi Starting phi bin index
     * \param[in,out] visited Visited bins tracking array
     * \param[in] cone Cone parameters for gap scoring
     * \return Gap object representing the found contiguous region
     */
    Gap floodFillGap(const AngularBins &bins, int start_theta, int start_phi, std::vector<std::vector<bool>> &visited, const Cone &cone);

    /**
     * \brief Calculate solid angle of a single angular bin
     * \param[in] theta_bin Theta bin index
     * \param[in] phi_bin Phi bin index
     * \param[in] bins Angular bin structure
     * \param[in] cone_half_angle Cone half-angle in radians
     * \return Solid angle in steradians
     */
    float calculateBinSolidAngle(int theta_bin, int phi_bin, const AngularBins &bins, float cone_half_angle);

    /**
     * \brief Convert bin indices to cartesian direction vector
     * \param[in] theta_bin Theta bin index
     * \param[in] phi_bin Phi bin index
     * \param[in] bins Angular bin structure
     * \param[in] cone Cone parameters
     * \return Normalized direction vector
     */
    helios::vec3 binIndicesToCartesian(int theta_bin, int phi_bin, const AngularBins &bins, const Cone &cone);

    /**
     * \brief Add unoccupied neighboring bins to flood-fill queue
     * \param[in] theta Current theta bin index
     * \param[in] phi Current phi bin index
     * \param[in] bins Angular bin structure
     * \param[in] visited Visited bins tracking array
     * \param[in,out] queue Flood-fill queue
     */
    void addUnoccupiedNeighbors(int theta, int phi, const AngularBins &bins, std::vector<std::vector<bool>> &visited, std::queue<std::pair<int, int>> &queue);
};

#endif
