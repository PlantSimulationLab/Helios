/** \file "CollisionDetection.cpp" Source file for collision detection plugin

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
#include <algorithm>
#include <chrono>
#include <thread>

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
// Phase 3: Warp-efficient GPU kernels
void launchWarpEfficientBVH(void *h_bvh_soa_gpu, unsigned int *h_primitive_indices, int primitive_count, float *h_primitive_aabb_min, float *h_primitive_aabb_max, float *h_ray_origins, float *h_ray_directions, float *h_ray_max_distances, int num_rays, unsigned int *h_results, unsigned int *h_result_counts, int max_results_per_ray);
}

// Helper function to convert helios::vec3 to float3
inline float3 heliosVecToFloat3(const helios::vec3 &v) {
    return make_float3(v.x, v.y, v.z);
}
#endif

CollisionDetection::CollisionDetection(helios::Context *a_context) {

    if (a_context == nullptr) {
        helios_runtime_error("ERROR (CollisionDetection::CollisionDetection): Context is null");
    }

    context = a_context;
    printmessages = true;

#ifdef HELIOS_CUDA_AVAILABLE
    gpu_acceleration_enabled = true;
#else
    gpu_acceleration_enabled = false;
#endif

    // Initialize GPU memory pointers
    d_bvh_nodes = nullptr;
    d_primitive_indices = nullptr;
    gpu_memory_allocated = false;

    // Initialize BVH caching variables
    bvh_dirty = true;
    automatic_bvh_rebuilds = true; // Default: allow automatic rebuilds
    
    // Initialize hierarchical BVH variables
    hierarchical_bvh_enabled = false;
    static_bvh_valid = false;

    // Initialize grid parameters
    grid_center = make_vec3(0, 0, 0);
    grid_size = make_vec3(1, 1, 1);
    grid_divisions = helios::make_int3(1, 1, 1);

    // Initialize voxel data parameters
    voxel_data_initialized = false;
    voxel_grid_center = make_vec3(0, 0, 0);
    voxel_grid_size = make_vec3(1, 1, 1);
    voxel_grid_divisions = helios::make_int3(1, 1, 1);

    // Initialize spatial optimization parameters
    max_collision_distance = 10.0f; // Default 10 meter maximum distance

    if (printmessages) {
#ifdef HELIOS_CUDA_AVAILABLE
        std::cout << "CollisionDetection plugin initialized successfully with GPU support." << std::endl;
#else
        std::cout << "CollisionDetection plugin initialized successfully (CPU-only mode)." << std::endl;
#endif
    }
}

CollisionDetection::~CollisionDetection() {
    freeGPUMemory();
}

std::vector<uint> CollisionDetection::findCollisions(uint UUID) {
    return findCollisions(std::vector<uint>{UUID});
}

std::vector<uint> CollisionDetection::findCollisions(const std::vector<uint> &UUIDs) {

    if (UUIDs.empty()) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findCollisions): No UUIDs provided" << std::endl;
        }
        return {};
    }

    // Validate UUIDs - throw exception if any are invalid
    std::vector<uint> valid_UUIDs;
    for (uint uuid : UUIDs) {
        if (context->doesPrimitiveExist(uuid)) {
            valid_UUIDs.push_back(uuid);
        } else {
            if (printmessages) {
                std::cerr << "ERROR (CollisionDetection::findCollisions): Invalid UUID " << uuid << std::endl;
            }
            helios_runtime_error("ERROR (CollisionDetection::findCollisions): Invalid UUID " + std::to_string(uuid) + " provided");
        }
    }

    // Automatically rebuild BVH if geometry has changed or BVH is empty
    ensureBVHCurrent();

    std::vector<uint> all_collisions;

    for (uint UUID: valid_UUIDs) {

        // Get bounding box for query primitive
        if (!context->doesPrimitiveExist(UUID)) {
            continue; // Skip invalid primitive
        }
        vec3 aabb_min, aabb_max;
        context->getPrimitiveBoundingBox(UUID, aabb_min, aabb_max);

        std::vector<uint> collisions;

        if (gpu_acceleration_enabled && gpu_memory_allocated) {
            collisions = traverseBVH_GPU(aabb_min, aabb_max);
        } else {
            collisions = traverseBVH_CPU(aabb_min, aabb_max);
        }

        // Remove the query UUID from results
        collisions.erase(std::remove(collisions.begin(), collisions.end(), UUID), collisions.end());

        // Add to overall results
        all_collisions.insert(all_collisions.end(), collisions.begin(), collisions.end());
    }

    // Remove duplicates
    std::sort(all_collisions.begin(), all_collisions.end());
    all_collisions.erase(std::unique(all_collisions.begin(), all_collisions.end()), all_collisions.end());

    if (printmessages) {
        std::cout << "Found " << all_collisions.size() << " colliding primitives." << std::endl;
    }

    return all_collisions;
}

std::vector<uint> CollisionDetection::findCollisions(const std::vector<uint> &primitive_UUIDs, const std::vector<uint> &object_IDs) {

    if (primitive_UUIDs.empty() && object_IDs.empty()) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findCollisions): No UUIDs or object IDs provided" << std::endl;
        }
        return {};
    }

    // Expand object IDs to their constituent primitive UUIDs
    std::vector<uint> all_test_UUIDs = primitive_UUIDs;

    for (uint ObjID: object_IDs) {
        if (!context->doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (CollisionDetection::findCollisions): Object ID " + std::to_string(ObjID) + " does not exist");
        }

        std::vector<uint> object_UUIDs = context->getObjectPrimitiveUUIDs(ObjID);
        all_test_UUIDs.insert(all_test_UUIDs.end(), object_UUIDs.begin(), object_UUIDs.end());
    }

    return findCollisions(all_test_UUIDs);
}

std::vector<uint> CollisionDetection::findCollisions(const std::vector<uint> &query_UUIDs, const std::vector<uint> &query_object_IDs, const std::vector<uint> &target_UUIDs, const std::vector<uint> &target_object_IDs) {

    if (query_UUIDs.empty() && query_object_IDs.empty()) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findCollisions): No query UUIDs or object IDs provided" << std::endl;
        }
        return {};
    }

    // Expand query objects to their constituent primitive UUIDs
    std::vector<uint> all_query_UUIDs = query_UUIDs;

    for (uint ObjID: query_object_IDs) {
        if (!context->doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (CollisionDetection::findCollisions): Query object ID " + std::to_string(ObjID) + " does not exist");
        }

        std::vector<uint> object_UUIDs = context->getObjectPrimitiveUUIDs(ObjID);
        all_query_UUIDs.insert(all_query_UUIDs.end(), object_UUIDs.begin(), object_UUIDs.end());
    }

    // Validate query UUIDs
    if (!validateUUIDs(all_query_UUIDs)) {
        helios_runtime_error("ERROR (CollisionDetection::findCollisions): One or more invalid query UUIDs provided");
    }

    // Build restricted BVH if target geometry is specified

    if (target_UUIDs.empty() && target_object_IDs.empty()) {
        // Use all geometry in context as target
        ensureBVHCurrent();
    } else {
        // Build restricted BVH with only target geometry
        std::vector<uint> all_target_UUIDs = target_UUIDs;

        for (uint ObjID: target_object_IDs) {
            if (!context->doesObjectExist(ObjID)) {
                helios_runtime_error("ERROR (CollisionDetection::findCollisions): Target object ID " + std::to_string(ObjID) + " does not exist");
            }

            std::vector<uint> object_UUIDs = context->getObjectPrimitiveUUIDs(ObjID);
            all_target_UUIDs.insert(all_target_UUIDs.end(), object_UUIDs.begin(), object_UUIDs.end());
        }

        // Validate target UUIDs
        if (!validateUUIDs(all_target_UUIDs)) {
            helios_runtime_error("ERROR (CollisionDetection::findCollisions): One or more invalid target UUIDs provided");
        }

        // Build BVH with only the target geometry (with caching)
        updateBVH(all_target_UUIDs, false); // Use caching logic instead of direct rebuild
    }

    // Perform collision detection using the same logic as the standard findCollisions
    std::vector<uint> all_collisions;

    for (uint UUID: all_query_UUIDs) {

        // Get bounding box for query primitive
        if (!context->doesPrimitiveExist(UUID)) {
            continue; // Skip invalid primitive
        }
        vec3 aabb_min, aabb_max;
        context->getPrimitiveBoundingBox(UUID, aabb_min, aabb_max);

        std::vector<uint> collisions;

        if (gpu_acceleration_enabled && gpu_memory_allocated) {
            collisions = traverseBVH_GPU(aabb_min, aabb_max);
        } else {
            collisions = traverseBVH_CPU(aabb_min, aabb_max);
        }

        // Remove the query UUID from results
        collisions.erase(std::remove(collisions.begin(), collisions.end(), UUID), collisions.end());

        // Add to overall results
        all_collisions.insert(all_collisions.end(), collisions.begin(), collisions.end());
    }

    // Remove duplicates
    std::sort(all_collisions.begin(), all_collisions.end());
    all_collisions.erase(std::unique(all_collisions.begin(), all_collisions.end()), all_collisions.end());

    return all_collisions;
}

void CollisionDetection::buildBVH(const std::vector<uint> &UUIDs) {

    std::vector<uint> primitives_to_include;

    if (UUIDs.empty()) {
        // Include all primitives in context
        primitives_to_include = context->getAllUUIDs();
    } else {
        primitives_to_include = UUIDs;
    }
    

    if (primitives_to_include.empty()) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::buildBVH): No primitives found to build BVH" << std::endl;
        }
        return;
    }

    // Validate UUIDs - throw exception if any are invalid (only when specific UUIDs are provided)
    std::vector<uint> valid_primitives;
    if (!UUIDs.empty()) {
        // When specific UUIDs are provided, they must all be valid
        for (uint uuid : primitives_to_include) {
            if (context->doesPrimitiveExist(uuid)) {
                valid_primitives.push_back(uuid);
            } else {
                if (printmessages) {
                    std::cerr << "ERROR (CollisionDetection::buildBVH): Invalid UUID " << uuid << std::endl;
                }
                helios_runtime_error("ERROR (CollisionDetection::buildBVH): Invalid UUID " + std::to_string(uuid) + " provided");
            }
        }
    } else {
        // When no specific UUIDs provided (use all), filter out invalid ones
        for (uint uuid : primitives_to_include) {
            if (context->doesPrimitiveExist(uuid)) {
                valid_primitives.push_back(uuid);
            } else if (printmessages) {
                std::cerr << "WARNING (CollisionDetection::buildBVH): Skipping invalid UUID " << uuid << std::endl;
            }
        }
        
        if (valid_primitives.empty()) {
            if (printmessages) {
                std::cerr << "WARNING (CollisionDetection::buildBVH): No valid primitives found after filtering" << std::endl;
            }
            return;
        }
    }
    
    primitives_to_include = valid_primitives;

    // Clear existing BVH
    bvh_nodes.clear();
    primitive_indices.clear();

    // Copy primitives to indices array
    primitive_indices = primitives_to_include;

    // Pre-allocate BVH nodes to avoid excessive resizing
    // For N primitives, we need at most 2*N-1 nodes for a complete binary tree
    size_t max_nodes = std::max(size_t(1), 2 * primitives_to_include.size());
    bvh_nodes.reserve(max_nodes);
    bvh_nodes.resize(1); // Start with root node

    // OPTIMIZATION: Pre-cache bounding boxes with dirty flagging to avoid repeated expensive calculations
    // Only clear cache for primitives that no longer exist or are dirty
    std::unordered_set<uint> current_primitives(primitives_to_include.begin(), primitives_to_include.end());
    
    // Remove cached entries for primitives that no longer exist
    auto cache_it = primitive_aabbs_cache.begin();
    while (cache_it != primitive_aabbs_cache.end()) {
        if (current_primitives.find(cache_it->first) == current_primitives.end()) {
            cache_it = primitive_aabbs_cache.erase(cache_it);
        } else {
            ++cache_it;
        }
    }
    
    // Update only dirty or missing cache entries
    for (uint UUID: primitives_to_include) {
        if (!context->doesPrimitiveExist(UUID)) {
            continue; // Skip invalid primitive
        }
        
        // Only update if not cached or marked as dirty
        bool needs_update = (primitive_aabbs_cache.find(UUID) == primitive_aabbs_cache.end()) ||
                           (dirty_primitive_cache.find(UUID) != dirty_primitive_cache.end());
        
        if (needs_update) {
            vec3 aabb_min, aabb_max;
            context->getPrimitiveBoundingBox(UUID, aabb_min, aabb_max);
            primitive_aabbs_cache[UUID] = {aabb_min, aabb_max};
            dirty_primitive_cache.erase(UUID); // Mark as clean
        }
    }

    // Build BVH recursively starting from root
    buildBVHRecursive(0, 0, primitive_indices.size(), 0);

    // Shrink to actual size used
    bvh_nodes.shrink_to_fit();

    // Transfer to GPU if acceleration is enabled
    if (gpu_acceleration_enabled) {
        transferBVHToGPU();
    }

    // Update internal tracking - record what we have processed
    // This allows us to avoid redundant rebuilds until user marks geometry clean
    std::vector<uint> context_deleted_uuids = context->getDeletedUUIDs();

    // Track ALL primitives that are now in the BVH, not just dirty ones
    // This ensures isBVHValid() works correctly when new geometry is added
    last_processed_uuids.clear();
    last_processed_uuids.insert(primitives_to_include.begin(), primitives_to_include.end());

    last_processed_deleted_uuids.clear();
    last_processed_deleted_uuids.insert(context_deleted_uuids.begin(), context_deleted_uuids.end());

    // Update BVH geometry tracking for caching
    last_bvh_geometry.clear();
    last_bvh_geometry.insert(primitives_to_include.begin(), primitives_to_include.end());
    bvh_dirty = false;

}

void CollisionDetection::rebuildBVH() {
    markBVHDirty();
    buildBVH();
}

void CollisionDetection::disableAutomaticBVHRebuilds() {
    automatic_bvh_rebuilds = false;
    if (printmessages) {
        std::cout << "Disabled automatic BVH rebuilds - caller must manually manage rebuilds" << std::endl;
    }
}

void CollisionDetection::enableAutomaticBVHRebuilds() {
    automatic_bvh_rebuilds = true;
    if (printmessages) {
        std::cout << "Enabled automatic BVH rebuilds (default behavior)" << std::endl;
    }
}

void CollisionDetection::enableHierarchicalBVH() {
    hierarchical_bvh_enabled = true;
    static_bvh_valid = false; // Force rebuild of static BVH
    if (printmessages) {
        std::cout << "Enabled hierarchical BVH (separate static/dynamic geometry)" << std::endl;
    }
}

void CollisionDetection::disableHierarchicalBVH() {
    hierarchical_bvh_enabled = false;
    // Clear static BVH data
    static_bvh_nodes.clear();
    static_bvh_primitives.clear();
    static_bvh_valid = false;
    last_static_bvh_geometry.clear();
    if (printmessages) {
        std::cout << "Disabled hierarchical BVH (using single unified BVH)" << std::endl;
    }
}

void CollisionDetection::updateHierarchicalBVH(const std::set<uint> &requested_geometry, bool force_rebuild) {
    
    // Step 1: Build/update static BVH if needed
    if (!static_bvh_valid || force_rebuild || static_geometry_cache != last_static_bvh_geometry) {
        buildStaticBVH();
    }
    
    // Step 2: Separate dynamic geometry (not in static cache)
    std::vector<uint> dynamic_geometry;
    for (uint uuid : requested_geometry) {
        if (static_geometry_cache.find(uuid) == static_geometry_cache.end()) {
            dynamic_geometry.push_back(uuid);
        }
    }
    
    
    // Step 3: Build dynamic BVH with remaining geometry (this is smaller and faster)
    if (!dynamic_geometry.empty()) {
        buildBVH(dynamic_geometry);  // Use existing buildBVH for dynamic part
    } else {
        // No dynamic geometry - just clear the dynamic BVH
        bvh_nodes.clear();
        primitive_indices.clear();
    }
    
    // Update cache
    last_bvh_geometry = requested_geometry;
    bvh_dirty = false;
}

void CollisionDetection::buildStaticBVH() {
    if (static_geometry_cache.empty()) {
        static_bvh_nodes.clear();
        static_bvh_primitives.clear();
        static_bvh_valid = false;
        if (printmessages) {
            std::cout << "[STATIC BVH] No static geometry defined - cleared static BVH" << std::endl;
        }
        return;
    }
    
    std::vector<uint> static_primitives(static_geometry_cache.begin(), static_geometry_cache.end());
    
    
    // Build BVH for static geometry (reuse existing GPU BVH building logic)
    // For now, we'll store it in the static BVH structures but use same building method
    std::vector<BVHNode> temp_nodes;
    std::vector<uint> temp_primitives;
    
    // Temporarily swap in static storage
    std::swap(bvh_nodes, temp_nodes);
    std::swap(primitive_indices, temp_primitives);
    
    // Build BVH using existing method
    buildBVH(static_primitives);
    
    // Store result in static BVH and restore dynamic BVH
    static_bvh_nodes = bvh_nodes;
    static_bvh_primitives = primitive_indices;
    std::swap(bvh_nodes, temp_nodes);
    std::swap(primitive_indices, temp_primitives);
    
    static_bvh_valid = true;
    last_static_bvh_geometry = static_geometry_cache;
    
}

void CollisionDetection::updateBVH(const std::vector<uint> &UUIDs, bool force_rebuild) {
    // Convert input to set for efficient comparison
    std::set<uint> requested_geometry(UUIDs.begin(), UUIDs.end());
    
    // Check if geometry has changed significantly
    bool geometry_changed = (requested_geometry != last_bvh_geometry) || bvh_dirty;
    
    if (!geometry_changed && !force_rebuild) {
        return;
    }
    
    if (printmessages) {
        std::cout << "BVH update: " << UUIDs.size() << " primitives" << std::endl;
    }
    
    // Use hierarchical BVH approach if enabled
    if (hierarchical_bvh_enabled) {
        updateHierarchicalBVH(requested_geometry, force_rebuild);
        return;
    }
    
    // Determine if we need a full rebuild or can do incremental update
    if (force_rebuild || bvh_nodes.empty()) {
        // Full rebuild required
        buildBVH(UUIDs);
    } else {
        // Check how much geometry has changed
        std::set<uint> added_geometry, removed_geometry;
        
        // Find added geometry (in requested but not in last_bvh_geometry)
        std::set_difference(requested_geometry.begin(), requested_geometry.end(),
                           last_bvh_geometry.begin(), last_bvh_geometry.end(),
                           std::inserter(added_geometry, added_geometry.begin()));
        
        // Find removed geometry (in last_bvh_geometry but not in requested)
        std::set_difference(last_bvh_geometry.begin(), last_bvh_geometry.end(),
                           requested_geometry.begin(), requested_geometry.end(),
                           std::inserter(removed_geometry, removed_geometry.begin()));
        
        // If more than 20% of geometry changed, do full rebuild, otherwise incremental
        size_t total_change = added_geometry.size() + removed_geometry.size();
        size_t current_size = std::max(last_bvh_geometry.size(), requested_geometry.size());
        
        if (current_size == 0 || (float(total_change) / float(current_size)) > 0.2f) {
            if (printmessages) {
                std::cout << "Significant geometry change detected, performing full rebuild" << std::endl;
            }
            buildBVH(UUIDs);
        } else {
            if (printmessages) {
                std::cout << "Minor geometry change detected, performing incremental update (" 
                         << added_geometry.size() << " added, " << removed_geometry.size() << " removed)" << std::endl;
            }
            // Implement incremental update by selective insertion/removal
            incrementalUpdateBVH(added_geometry, removed_geometry, requested_geometry);
        }
    }
    
    // Update tracking
    last_bvh_geometry = requested_geometry;
    bvh_dirty = false;
}

void CollisionDetection::setStaticGeometry(const std::vector<uint> &UUIDs) {
    static_geometry_cache.clear();
    static_geometry_cache.insert(UUIDs.begin(), UUIDs.end());
    
    if (printmessages) {
        std::cout << "Marked " << UUIDs.size() << " primitives as static geometry" << std::endl;
    }
}

void CollisionDetection::ensureBVHCurrent() {
    // If automatic rebuilds are disabled, skip all automatic updates
    if (!automatic_bvh_rebuilds) {
        return;
    }
    
    // If BVH is completely empty, build it with all geometry
    if (bvh_nodes.empty()) {
        if (printmessages) {
            std::cout << "Building initial BVH..." << std::endl;
        }
        buildBVH();
        return;
    }

    // Use two-level dirty tracking pattern (similar to Visualizer plugin)
    // Get dirty UUIDs from Context (but don't clear them - that's for the user)
    std::vector<uint> context_dirty_uuids = context->getDirtyUUIDs(false); // Don't include deleted
    std::vector<uint> context_deleted_uuids = context->getDeletedUUIDs();

    // Convert to sets for efficient comparison
    std::set<uint> current_dirty(context_dirty_uuids.begin(), context_dirty_uuids.end());
    std::set<uint> current_deleted(context_deleted_uuids.begin(), context_deleted_uuids.end());

    // Check if there are new changes since our last processing
    bool has_new_dirty = false;
    bool has_new_deleted = false;

    // Check for new dirty UUIDs we haven't processed
    for (uint uuid: current_dirty) {
        if (last_processed_uuids.find(uuid) == last_processed_uuids.end()) {
            has_new_dirty = true;
            break;
        }
    }

    // Check for new deleted UUIDs we haven't processed
    for (uint uuid: current_deleted) {
        if (last_processed_deleted_uuids.find(uuid) == last_processed_deleted_uuids.end()) {
            has_new_deleted = true;
            break;
        }
    }

    // Only rebuild if there are genuinely new changes since our last processing
    if (has_new_dirty || has_new_deleted) {
        if (printmessages) {
            std::cout << "Geometry has changed since last BVH build, rebuilding..." << std::endl;
        }

        buildBVH(); // This will update our internal tracking
    }

    // Note: We do NOT call context->markGeometryClean() here
    // That should only be done by the user after all plugins have processed the changes
}

bool CollisionDetection::isBVHValid() const {
    if (bvh_nodes.empty()) {
        return false;
    }

    // Check if there are new primitives in the context that aren't in our BVH
    std::vector<uint> all_context_uuids = context->getAllUUIDs();
    for (uint uuid: all_context_uuids) {
        if (last_processed_uuids.find(uuid) == last_processed_uuids.end()) {
            return false; // Found a primitive that's not in our BVH
        }
    }

    // Check for deleted UUIDs we haven't processed
    std::vector<uint> context_deleted_uuids = context->getDeletedUUIDs();
    for (uint uuid: context_deleted_uuids) {
        if (last_processed_deleted_uuids.find(uuid) == last_processed_deleted_uuids.end()) {
            return false; // Found new deleted UUID we haven't processed
        }
    }

    // Check if any primitives in our BVH have been deleted
    for (uint uuid: primitive_indices) {
        if (!context->doesPrimitiveExist(uuid)) {
            return false; // A primitive in our BVH no longer exists
        }
    }

    return true; // BVH is current with respect to geometry changes
}

void CollisionDetection::enableGPUAcceleration() {
    gpu_acceleration_enabled = true;
    if (!bvh_nodes.empty()) {
        transferBVHToGPU();
    }
}

void CollisionDetection::disableGPUAcceleration() {
    gpu_acceleration_enabled = false;
    freeGPUMemory();
}

bool CollisionDetection::isGPUAccelerationEnabled() const {
    return gpu_acceleration_enabled;
}

void CollisionDetection::disableMessages() {
    printmessages = false;
}

void CollisionDetection::enableMessages() {
    printmessages = true;
}

size_t CollisionDetection::getPrimitiveCount() const {
    return primitive_indices.size();
}

void CollisionDetection::getBVHStatistics(size_t &node_count, size_t &leaf_count, size_t &max_depth) const {

    node_count = bvh_nodes.size();
    leaf_count = 0;
    max_depth = 0;

    // Simple traversal to count leaves and find max depth
    std::function<void(uint, size_t)> traverse = [&](uint node_idx, size_t depth) {
        if (node_idx >= bvh_nodes.size())
            return;

        const BVHNode &node = bvh_nodes[node_idx];
        max_depth = std::max(max_depth, depth);

        if (node.is_leaf) {
            leaf_count++;
        } else {
            if (node.left_child != 0xFFFFFFFF) {
                traverse(node.left_child, depth + 1);
            }
            if (node.right_child != 0xFFFFFFFF) {
                traverse(node.right_child, depth + 1);
            }
        }
    };

    if (!bvh_nodes.empty()) {
        traverse(0, 0);
    }
}

void CollisionDetection::calculateAABB(const std::vector<uint> &primitives, vec3 &aabb_min, vec3 &aabb_max) const {

    if (primitives.empty()) {
        aabb_min = make_vec3(0, 0, 0);
        aabb_max = make_vec3(0, 0, 0);
        return;
    }

    // Find first valid primitive for initialization
    size_t first_valid = 0;
    while (first_valid < primitives.size() && !context->doesPrimitiveExist(primitives[first_valid])) {
        first_valid++;
    }
    if (first_valid >= primitives.size()) {
        // No valid primitives found
        aabb_min = make_vec3(0, 0, 0);
        aabb_max = make_vec3(0, 0, 0);
        return;
    }

    // Initialize with first valid primitive's bounding box
    context->getPrimitiveBoundingBox(primitives[first_valid], aabb_min, aabb_max);

    // Expand to include all remaining valid primitives
    for (size_t i = first_valid + 1; i < primitives.size(); i++) {
        if (!context->doesPrimitiveExist(primitives[i])) {
            continue; // Skip invalid primitive
        }
        vec3 prim_min, prim_max;
        context->getPrimitiveBoundingBox(primitives[i], prim_min, prim_max);

        aabb_min.x = std::min(aabb_min.x, prim_min.x);
        aabb_min.y = std::min(aabb_min.y, prim_min.y);
        aabb_min.z = std::min(aabb_min.z, prim_min.z);

        aabb_max.x = std::max(aabb_max.x, prim_max.x);
        aabb_max.y = std::max(aabb_max.y, prim_max.y);
        aabb_max.z = std::max(aabb_max.z, prim_max.z);
    }
}

void CollisionDetection::buildBVHRecursive(uint node_index, size_t primitive_start, size_t primitive_count, int depth) {

    // Ensure node exists (should already be pre-allocated)
    if (node_index >= bvh_nodes.size()) {
        bvh_nodes.resize(node_index + 1);
    }

    BVHNode &node = bvh_nodes[node_index];

    // Calculate bounding box for this node using cached AABBs
    if (primitive_count == 0) {
        node.aabb_min = make_vec3(0, 0, 0);
        node.aabb_max = make_vec3(0, 0, 0);
    } else {
        // Initialize with first primitive's cached AABB
        uint first_uuid = primitive_indices[primitive_start];
        auto it = primitive_aabbs_cache.find(first_uuid);
        if (it == primitive_aabbs_cache.end()) {
            // Handle missing primitive - use zero bounds
            node.aabb_min = make_vec3(0, 0, 0);
            node.aabb_max = make_vec3(0, 0, 0);
            return;
        }
        const auto &first_cached_aabb = it->second;
        node.aabb_min = first_cached_aabb.first;
        node.aabb_max = first_cached_aabb.second;

        // Expand to include all primitives in this node
        for (size_t i = 1; i < primitive_count; i++) {
            uint uuid = primitive_indices[primitive_start + i];
            auto it = primitive_aabbs_cache.find(uuid);
            if (it == primitive_aabbs_cache.end()) {
                continue; // Skip missing primitive
            }
            const auto &cached_aabb = it->second;
            node.aabb_min.x = std::min(node.aabb_min.x, cached_aabb.first.x);
            node.aabb_min.y = std::min(node.aabb_min.y, cached_aabb.first.y);
            node.aabb_min.z = std::min(node.aabb_min.z, cached_aabb.first.z);

            node.aabb_max.x = std::max(node.aabb_max.x, cached_aabb.second.x);
            node.aabb_max.y = std::max(node.aabb_max.y, cached_aabb.second.y);
            node.aabb_max.z = std::max(node.aabb_max.z, cached_aabb.second.z);
        }
    }

    // Stopping criteria - make this a leaf
    const int MAX_PRIMITIVES_PER_LEAF = 4;
    const int MAX_DEPTH = 20;

    if (primitive_count <= MAX_PRIMITIVES_PER_LEAF || depth >= MAX_DEPTH) {
        // Make leaf node
        node.is_leaf = true;
        node.primitive_start = primitive_start;
        node.primitive_count = primitive_count;
        node.left_child = 0xFFFFFFFF;
        node.right_child = 0xFFFFFFFF;
        return;
    }

    // Find best split using Surface Area Heuristic (simplified)
    vec3 extent = node.aabb_max - node.aabb_min;
    int split_axis = 0; // Split along longest axis
    if (extent.y > extent.x)
        split_axis = 1;
    if (extent.z > (split_axis == 0 ? extent.x : extent.y))
        split_axis = 2;

    // Sort primitives along split axis by their centroid using cached AABBs
    std::sort(primitive_indices.begin() + primitive_start, primitive_indices.begin() + primitive_start + primitive_count, [&](uint a, uint b) {
        auto it_a = primitive_aabbs_cache.find(a);
        auto it_b = primitive_aabbs_cache.find(b);
        if (it_a == primitive_aabbs_cache.end() || it_b == primitive_aabbs_cache.end()) {
            return false; // Can't compare missing primitives
        }
        const auto &cached_aabb_a = it_a->second;
        const auto &cached_aabb_b = it_b->second;
        vec3 centroid_a = 0.5f * (cached_aabb_a.first + cached_aabb_a.second);
        vec3 centroid_b = 0.5f * (cached_aabb_b.first + cached_aabb_b.second);
        float a_coord = (split_axis == 0) ? centroid_a.x : (split_axis == 1) ? centroid_a.y : centroid_a.z;
        float b_coord = (split_axis == 0) ? centroid_b.x : (split_axis == 1) ? centroid_b.y : centroid_b.z;
        return a_coord < b_coord;
    });

    // Split in middle
    size_t split_index = primitive_count / 2;

    // Create child nodes by allocating next two available indices
    uint left_child_index = bvh_nodes.size();
    uint right_child_index = bvh_nodes.size() + 1;

    // Resize to accommodate both children at once (more efficient than incremental resizing)
    bvh_nodes.resize(bvh_nodes.size() + 2);

    // Re-get the node reference after potential reallocation
    BVHNode &updated_node = bvh_nodes[node_index];
    updated_node.left_child = left_child_index;
    updated_node.right_child = right_child_index;
    updated_node.is_leaf = false;
    updated_node.primitive_start = 0;
    updated_node.primitive_count = 0;

    // Recursively build child nodes
    buildBVHRecursive(left_child_index, primitive_start, split_index, depth + 1);
    buildBVHRecursive(right_child_index, primitive_start + split_index, primitive_count - split_index, depth + 1);
}

std::vector<uint> CollisionDetection::traverseBVH_CPU(const vec3 &query_aabb_min, const vec3 &query_aabb_max) {

    std::vector<uint> results;

    if (bvh_nodes.empty()) {
        return results;
    }

    // Stack-based traversal to avoid recursion
    std::vector<uint> node_stack;
    node_stack.push_back(0); // Start with root node

    while (!node_stack.empty()) {
        uint node_idx = node_stack.back();
        node_stack.pop_back();

        if (node_idx >= bvh_nodes.size())
            continue;

        const BVHNode &node = bvh_nodes[node_idx];

        // Test if query AABB intersects node AABB
        if (!aabbIntersect(query_aabb_min, query_aabb_max, node.aabb_min, node.aabb_max)) {
            continue;
        }

        if (node.is_leaf) {
            // Check each primitive in this leaf for intersection
            for (uint i = 0; i < node.primitive_count; i++) {
                uint primitive_id = primitive_indices[node.primitive_start + i];

                // Get this primitive's AABB
                if (!context->doesPrimitiveExist(primitive_id)) {
                    continue; // Skip invalid primitive
                }
                vec3 prim_min, prim_max;
                context->getPrimitiveBoundingBox(primitive_id, prim_min, prim_max);

                // Only add to results if AABBs actually intersect
                if (aabbIntersect(query_aabb_min, query_aabb_max, prim_min, prim_max)) {
                    results.push_back(primitive_id);
                }
            }
        } else {
            // Add child nodes to stack for processing
            if (node.left_child != 0xFFFFFFFF) {
                node_stack.push_back(node.left_child);
            }
            if (node.right_child != 0xFFFFFFFF) {
                node_stack.push_back(node.right_child);
            }
        }
    }

    return results;
}

std::vector<uint> CollisionDetection::traverseBVH_GPU(const vec3 &query_aabb_min, const vec3 &query_aabb_max) {
#ifdef HELIOS_CUDA_AVAILABLE
    if (!gpu_memory_allocated) {
        if (printmessages) {
            std::cerr << "WARNING: GPU memory not allocated, falling back to CPU" << std::endl;
        }
        return traverseBVH_CPU(query_aabb_min, query_aabb_max);
    }

    // Prepare single query
    float query_min_array[3] = {query_aabb_min.x, query_aabb_min.y, query_aabb_min.z};
    float query_max_array[3] = {query_aabb_max.x, query_aabb_max.y, query_aabb_max.z};

    // Prepare primitive AABB arrays
    std::vector<float> primitive_min_array(primitive_indices.size() * 3);
    std::vector<float> primitive_max_array(primitive_indices.size() * 3);

    for (size_t i = 0; i < primitive_indices.size(); i++) {
        uint uuid = primitive_indices[i];
        auto it = primitive_aabbs_cache.find(uuid);
        if (it == primitive_aabbs_cache.end()) {
            continue; // Skip missing primitive
        }
        const auto &cached_aabb = it->second;

        primitive_min_array[i * 3] = cached_aabb.first.x;
        primitive_min_array[i * 3 + 1] = cached_aabb.first.y;
        primitive_min_array[i * 3 + 2] = cached_aabb.first.z;

        primitive_max_array[i * 3] = cached_aabb.second.x;
        primitive_max_array[i * 3 + 1] = cached_aabb.second.y;
        primitive_max_array[i * 3 + 2] = cached_aabb.second.z;
    }

    const int max_results = 1000; // Reasonable limit
    std::vector<unsigned int> results(max_results);
    unsigned int result_count = 0;

    // Call CUDA kernel wrapper
    launchBVHTraversal(d_bvh_nodes, bvh_nodes.size(), d_primitive_indices, primitive_indices.size(), primitive_min_array.data(), primitive_max_array.data(), query_min_array, query_max_array, 1, results.data(), &result_count, max_results);

    // Convert to return format
    std::vector<uint> final_results;
    for (unsigned int i = 0; i < result_count; i++) {
        final_results.push_back(results[i]);
    }

    return final_results;
#else
    // CUDA not available, fall back to CPU
    if (printmessages) {
        std::cout << "CUDA not available, falling back to CPU" << std::endl;
    }
    return traverseBVH_CPU(query_aabb_min, query_aabb_max);
#endif
}

bool CollisionDetection::aabbIntersect(const vec3 &min1, const vec3 &max1, const vec3 &min2, const vec3 &max2) {
    return (min1.x <= max2.x && max1.x >= min2.x) && (min1.y <= max2.y && max1.y >= min2.y) && (min1.z <= max2.z && max1.z >= min2.z);
}

bool CollisionDetection::rayAABBIntersect(const vec3 &origin, const vec3 &direction, const vec3 &aabb_min, const vec3 &aabb_max, float &t_min, float &t_max) {

    t_min = 0.0f;
    t_max = std::numeric_limits<float>::max();

    // Check intersection with each pair of parallel planes (X, Y, Z)
    for (int i = 0; i < 3; i++) {
        float dir_component = (i == 0) ? direction.x : (i == 1) ? direction.y : direction.z;
        float orig_component = (i == 0) ? origin.x : (i == 1) ? origin.y : origin.z;
        float min_component = (i == 0) ? aabb_min.x : (i == 1) ? aabb_min.y : aabb_min.z;
        float max_component = (i == 0) ? aabb_max.x : (i == 1) ? aabb_max.y : aabb_max.z;

        if (std::abs(dir_component) < 1e-9f) {
            // Ray is parallel to the slab
            if (orig_component < min_component || orig_component > max_component) {
                return false; // Ray is outside the slab and parallel to it
            }
        } else {
            // Compute intersection t values with the two planes
            float t1 = (min_component - orig_component) / dir_component;
            float t2 = (max_component - orig_component) / dir_component;

            // Make sure t1 is the near intersection and t2 is the far intersection
            if (t1 > t2) {
                std::swap(t1, t2);
            }

            // Update the overall near and far intersection parameters
            t_min = std::max(t_min, t1);
            t_max = std::min(t_max, t2);

            // If the near intersection is farther than the far intersection, no intersection
            if (t_min > t_max) {
                return false;
            }
        }
    }

    // Ray intersects AABB if t_min <= t_max and the intersection is in front of ray origin
    return t_max >= 0.0f;
}

bool CollisionDetection::coneAABBIntersect(const Cone &cone, const vec3 &aabb_min, const vec3 &aabb_max) {
    // Check if apex is inside AABB
    if (cone.apex.x >= aabb_min.x && cone.apex.x <= aabb_max.x && cone.apex.y >= aabb_min.y && cone.apex.y <= aabb_max.y && cone.apex.z >= aabb_min.z && cone.apex.z <= aabb_max.z) {
        return true; // Apex is inside AABB, definite intersection
    }

    // Early rejection using bounding sphere around AABB
    vec3 box_center = 0.5f * (aabb_min + aabb_max);
    vec3 box_half_extents = 0.5f * (aabb_max - aabb_min);
    float box_radius = box_half_extents.magnitude();

    // For infinite cone, check if any AABB corner is inside the cone
    if (cone.height <= 0.0f) {
        // Check all 8 corners of the AABB
        for (int i = 0; i < 8; i++) {
            vec3 corner = make_vec3((i & 1) ? aabb_max.x : aabb_min.x, (i & 2) ? aabb_max.y : aabb_min.y, (i & 4) ? aabb_max.z : aabb_min.z);

            // Vector from apex to corner
            vec3 apex_to_corner = corner - cone.apex;
            float distance_along_axis = apex_to_corner * cone.axis;

            // Point must be in front of apex
            if (distance_along_axis > 0) {
                // Check if point is within cone angle
                float cos_angle = distance_along_axis / apex_to_corner.magnitude();
                if (cos_angle >= cosf(cone.half_angle)) {
                    return true; // This corner is inside the cone
                }
            }
        }

        // Check if cone axis intersects AABB
        // Use ray-AABB intersection with cone axis as ray
        float t_min = 0.0f;
        float t_max = std::numeric_limits<float>::max();

        for (int i = 0; i < 3; i++) {
            float axis_component = (i == 0) ? cone.axis.x : (i == 1) ? cone.axis.y : cone.axis.z;
            float apex_component = (i == 0) ? cone.apex.x : (i == 1) ? cone.apex.y : cone.apex.z;
            float min_component = (i == 0) ? aabb_min.x : (i == 1) ? aabb_min.y : aabb_min.z;
            float max_component = (i == 0) ? aabb_max.x : (i == 1) ? aabb_max.y : aabb_max.z;

            if (std::abs(axis_component) < 1e-6f) {
                // Ray is parallel to slab
                if (apex_component < min_component || apex_component > max_component) {
                    return false; // Ray is outside slab
                }
            } else {
                // Compute intersection t values
                float t1 = (min_component - apex_component) / axis_component;
                float t2 = (max_component - apex_component) / axis_component;

                if (t1 > t2)
                    std::swap(t1, t2);

                t_min = std::max(t_min, t1);
                t_max = std::min(t_max, t2);

                if (t_min > t_max) {
                    return false; // No intersection
                }
            }
        }

        // If we get here, the cone axis intersects the AABB
        // But for narrow cones, we need to check if the AABB is within the cone angle
        if (t_min >= 0 && t_max >= 0) {
            // Check if the intersection region on the axis is within the cone angle
            // Find the closest point on the axis within the intersection region
            float t_check = std::max(0.0f, t_min);
            vec3 axis_point = cone.apex + cone.axis * t_check;

            // Find the closest point in the AABB to this axis point
            vec3 closest_in_box = make_vec3(std::max(aabb_min.x, std::min(axis_point.x, aabb_max.x)), std::max(aabb_min.y, std::min(axis_point.y, aabb_max.y)), std::max(aabb_min.z, std::min(axis_point.z, aabb_max.z)));

            // Check if this closest point is within the cone
            vec3 apex_to_point = closest_in_box - cone.apex;
            float distance_along_axis = apex_to_point * cone.axis;

            if (distance_along_axis > 0) {
                float distance_to_point = apex_to_point.magnitude();
                if (distance_to_point > 0) {
                    float cos_angle = distance_along_axis / distance_to_point;
                    if (cos_angle >= cosf(cone.half_angle)) {
                        return true;
                    }
                }
            }
        }
    } else {
        // Finite cone case
        // Check if any AABB corner is inside the finite cone
        for (int i = 0; i < 8; i++) {
            vec3 corner = make_vec3((i & 1) ? aabb_max.x : aabb_min.x, (i & 2) ? aabb_max.y : aabb_min.y, (i & 4) ? aabb_max.z : aabb_min.z);

            // Vector from apex to corner
            vec3 apex_to_corner = corner - cone.apex;
            float distance_along_axis = apex_to_corner * cone.axis;

            // Check if point is within cone height and in front of apex
            if (distance_along_axis > 0 && distance_along_axis <= cone.height) {
                // Check if point is within cone angle
                float cos_angle = distance_along_axis / apex_to_corner.magnitude();
                if (cos_angle >= cosf(cone.half_angle)) {
                    return true; // This corner is inside the cone
                }
            }
        }

        // Also need to check if cone base intersects AABB
        vec3 base_center = cone.apex + cone.axis * cone.height;
        float base_radius = cone.height * tanf(cone.half_angle);

        // Simple sphere-AABB check for cone base
        vec3 closest_point = make_vec3(std::max(aabb_min.x, std::min(base_center.x, aabb_max.x)), std::max(aabb_min.y, std::min(base_center.y, aabb_max.y)), std::max(aabb_min.z, std::min(base_center.z, aabb_max.z)));

        float dist_sq = (closest_point - base_center).magnitude();
        if (dist_sq <= base_radius) {
            return true; // Base circle intersects AABB
        }
    }

    // More sophisticated tests could be added here for edge cases
    // For now, return false if no intersection found
    return false;
}

void CollisionDetection::allocateGPUMemory() {
#ifdef HELIOS_CUDA_AVAILABLE
    if (gpu_memory_allocated) {
        freeGPUMemory(); // Clean up existing allocation
    }

    if (bvh_nodes.empty() || primitive_indices.empty()) {
        return; // Nothing to allocate
    }

    // Initialize pointers to nullptr for safety
    d_bvh_nodes = nullptr;
    d_primitive_indices = nullptr;

    // Calculate sizes
    size_t bvh_size = bvh_nodes.size() * sizeof(GPUBVHNode);
    size_t indices_size = primitive_indices.size() * sizeof(uint);

    // Validate sizes are reasonable
    if (bvh_size == 0 || indices_size == 0) {
        helios_runtime_error("ERROR: Invalid BVH or primitive data sizes for GPU allocation");
    }

    // Allocate BVH nodes
    cudaError_t err = cudaMalloc(&d_bvh_nodes, bvh_size);
    if (err != cudaSuccess) {
        helios_runtime_error("CUDA error allocating BVH nodes: " + std::string(cudaGetErrorString(err)));
    }

    // Allocate primitive indices
    err = cudaMalloc((void **) &d_primitive_indices, indices_size);
    if (err != cudaSuccess) {
        cudaFree(d_bvh_nodes);
        d_bvh_nodes = nullptr;
        helios_runtime_error("CUDA error allocating primitive indices: " + std::string(cudaGetErrorString(err)));
    }

    // Mark as allocated only after both allocations succeeded
    gpu_memory_allocated = true;

    if (printmessages) {
        std::cout << "Allocated " << (bvh_size + indices_size) / 1024 << " KB GPU memory" << std::endl;
    }
#else
    // CUDA not available, disable GPU acceleration
    gpu_acceleration_enabled = false;
    gpu_memory_allocated = false;
    if (printmessages) {
        std::cout << "CUDA not available, GPU acceleration disabled" << std::endl;
    }
#endif
}

void CollisionDetection::freeGPUMemory() {
#ifdef HELIOS_CUDA_AVAILABLE
    if (!gpu_memory_allocated)
        return;

    if (d_bvh_nodes) {
        cudaFree(d_bvh_nodes);
        d_bvh_nodes = nullptr;
    }

    if (d_primitive_indices) {
        cudaFree(d_primitive_indices);
        d_primitive_indices = nullptr;
    }

    gpu_memory_allocated = false;
#else
    // No GPU memory to free
    gpu_memory_allocated = false;
#endif
}

void CollisionDetection::transferBVHToGPU() {
#ifdef HELIOS_CUDA_AVAILABLE
    if (!gpu_acceleration_enabled || bvh_nodes.empty()) {
        return;
    }

    // Always reallocate GPU memory to handle size changes
    if (gpu_memory_allocated) {
        freeGPUMemory();
    }
    allocateGPUMemory();

    // Verify allocation succeeded
    if (!gpu_memory_allocated || d_bvh_nodes == nullptr || d_primitive_indices == nullptr) {
        helios_runtime_error("ERROR: Failed to allocate GPU memory for BVH transfer");
    }

    // Convert CPU BVH to GPU format
    std::vector<GPUBVHNode> gpu_nodes(bvh_nodes.size());
    for (size_t i = 0; i < bvh_nodes.size(); i++) {
        const BVHNode &cpu_node = bvh_nodes[i];
        GPUBVHNode &gpu_node = gpu_nodes[i];

        gpu_node.aabb_min = heliosVecToFloat3(cpu_node.aabb_min);
        gpu_node.aabb_max = heliosVecToFloat3(cpu_node.aabb_max);
        gpu_node.left_child = cpu_node.left_child;
        gpu_node.right_child = cpu_node.right_child;
        gpu_node.primitive_start = cpu_node.primitive_start;
        gpu_node.primitive_count = cpu_node.primitive_count;
        gpu_node.is_leaf = cpu_node.is_leaf ? 1 : 0;
        gpu_node.padding = 0;
    }

    // Transfer to GPU
    cudaError_t err = cudaMemcpy(d_bvh_nodes, gpu_nodes.data(), gpu_nodes.size() * sizeof(GPUBVHNode), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        helios_runtime_error("CUDA error transferring BVH nodes: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(d_primitive_indices, primitive_indices.data(), primitive_indices.size() * sizeof(uint), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        helios_runtime_error("CUDA error transferring primitive indices: " + std::string(cudaGetErrorString(err)));
    }

    if (printmessages) {
        std::cout << "Transferred BVH to GPU (" << bvh_nodes.size() << " nodes, " << primitive_indices.size() << " primitives)" << std::endl;
    }
#else
    // CUDA not available, disable GPU acceleration
    gpu_acceleration_enabled = false;
    if (printmessages) {
        std::cout << "CUDA not available, using CPU-only mode" << std::endl;
    }
#endif
}

void CollisionDetection::markBVHDirty() {
    // Clear internal tracking so BVH will be rebuilt on next access
    last_processed_uuids.clear();
    last_processed_deleted_uuids.clear();
    last_bvh_geometry.clear();
    bvh_dirty = true;
    // Free GPU memory since BVH will be rebuilt
    freeGPUMemory();
}

void CollisionDetection::incrementalUpdateBVH(const std::set<uint> &added_geometry, const std::set<uint> &removed_geometry, const std::set<uint> &final_geometry) {
    if (printmessages) {
        std::cout << "Performing incremental BVH update: +" << added_geometry.size() << " -" << removed_geometry.size() << " primitives" << std::endl;
    }
    
    // For small changes, it's actually more efficient to do a targeted rebuild than complex tree restructuring
    // True incremental BVH updates require complex rebalancing algorithms
    
    // Convert final geometry to vector for buildBVH
    std::vector<uint> final_primitives(final_geometry.begin(), final_geometry.end());
    
    // Validate new geometries exist
    for (uint uuid : added_geometry) {
        if (!context->doesPrimitiveExist(uuid)) {
            if (printmessages) {
                std::cout << "Warning: Added primitive " << uuid << " does not exist, falling back to full rebuild" << std::endl;
            }
            buildBVH(final_primitives);
            return;
        }
    }
    
    // For now, use optimized rebuild for incremental updates
    // This is still faster than full geometry traversal since we're limiting to specific primitives
    if (printmessages) {
        std::cout << "Performing targeted rebuild with " << final_primitives.size() << " primitives" << std::endl;
    }
    
    buildBVH(final_primitives);
    
    // Update tracking
    last_bvh_geometry = final_geometry;
    bvh_dirty = false;
}

bool CollisionDetection::validateUUIDs(const std::vector<uint> &UUIDs) const {
    bool all_valid = true;
    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            if (printmessages) {
                std::cerr << "WARNING (CollisionDetection::validateUUIDs): Primitive UUID " + std::to_string(UUID) + " does not exist - skipping" << std::endl;
            }
            all_valid = false;
        }
    }
    return all_valid;
}

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

void CollisionDetection::calculateGridIntersection(const vec3 &grid_center, const vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<uint> &UUIDs) {
    if (printmessages) {
        std::cerr << "WARNING: calculateGridIntersection not yet implemented" << std::endl;
    }
}

std::vector<std::vector<std::vector<std::vector<uint>>>> CollisionDetection::getGridCells() {
    if (printmessages) {
        std::cerr << "WARNING: getGridCells not yet implemented" << std::endl;
    }
    return {};
}

std::vector<uint> CollisionDetection::getGridIntersections(int i, int j, int k) {
    if (printmessages) {
        std::cerr << "WARNING: getGridIntersections not yet implemented" << std::endl;
    }
    return {};
}

int CollisionDetection::optimizeLayout(const std::vector<uint> &UUIDs, float learning_rate, int max_iterations) {
    if (printmessages) {
        std::cerr << "WARNING: optimizeLayout not yet implemented" << std::endl;
    }
    return 0;
}

int CollisionDetection::countRayIntersections(const vec3 &origin, const vec3 &direction, float max_distance) {

    int intersection_count = 0;

    if (bvh_nodes.empty()) {
        return intersection_count;
    }

    // OPTIMIZATION: Minimum distance threshold to avoid self-intersection with nearby geometry
    // This prevents plant's own geometry (shoot tips, etc.) from occluding the entire cone view
    float min_distance = 0.05f; // 5cm minimum distance - ignore intersections closer than this

    // Ensure the BVH is current before traversal
    const_cast<CollisionDetection *>(this)->ensureBVHCurrent();

    // Stack-based traversal to avoid recursion
    std::vector<uint> node_stack;
    node_stack.push_back(0); // Start with root node

    while (!node_stack.empty()) {
        uint node_idx = node_stack.back();
        node_stack.pop_back();

        if (node_idx >= bvh_nodes.size())
            continue;

        const BVHNode &node = bvh_nodes[node_idx];

        // Test if ray intersects node AABB
        float t_min, t_max;
        if (!rayAABBIntersect(origin, direction, node.aabb_min, node.aabb_max, t_min, t_max)) {
            continue;
        }

        // Check if intersection is within distance range (both min and max)
        if (t_max < min_distance) {
            continue; // Entire AABB is too close - skip
        }
        if (max_distance > 0.0f && t_min > max_distance) {
            continue; // Entire AABB is too far - skip
        }

        if (node.is_leaf) {
            // Check each primitive in this leaf for ray intersection
            for (uint i = 0; i < node.primitive_count; i++) {
                uint primitive_id = primitive_indices[node.primitive_start + i];

                // Get this primitive's AABB
                if (!context->doesPrimitiveExist(primitive_id)) {
                    continue; // Skip invalid primitive
                }
                vec3 prim_min, prim_max;
                context->getPrimitiveBoundingBox(primitive_id, prim_min, prim_max);

                // Test ray against primitive AABB
                float prim_t_min, prim_t_max;
                if (rayAABBIntersect(origin, direction, prim_min, prim_max, prim_t_min, prim_t_max)) {
                    // Check distance constraints (both min and max)
                    bool within_min_distance = prim_t_min >= min_distance;
                    bool within_max_distance = (max_distance <= 0.0f) || (prim_t_min <= max_distance);
                    
                    if (within_min_distance && within_max_distance) {
                        intersection_count++;
                    }
                }
            }
        } else {
            // Add child nodes to stack for further traversal
            if (node.left_child != 0xFFFFFFFF) {
                node_stack.push_back(node.left_child);
            }
            if (node.right_child != 0xFFFFFFFF) {
                node_stack.push_back(node.right_child);
            }
        }
    }

    return intersection_count;
}

bool CollisionDetection::findNearestRayIntersection(const vec3 &origin, const vec3 &direction, const std::set<uint> &candidate_UUIDs, float &nearest_distance, float max_distance) {
    
    nearest_distance = std::numeric_limits<float>::max();
    bool found_intersection = false;
    
    // Check if we need to traverse both static and dynamic BVHs
    bool check_static_bvh = hierarchical_bvh_enabled && static_bvh_valid && !static_bvh_nodes.empty();
    bool check_dynamic_bvh = !bvh_nodes.empty();
    
    if (!check_static_bvh && !check_dynamic_bvh) {
        return false;
    }
    
    // Debug tracking
    static int ray_call_count = 0;
    ray_call_count++;
    bool should_debug = (ray_call_count % 50) == 1;
    int nodes_visited = 0;
    int primitives_checked = 0;
    int candidates_found_in_bvh = 0;
    
    // Ensure the BVH is current before traversal
    const_cast<CollisionDetection *>(this)->ensureBVHCurrent();
    
    // Lambda function to traverse a BVH and find ray intersections
    auto traverseBVH = [&](const std::vector<BVHNode>& nodes, const std::vector<uint>& primitives, const char* bvh_name) {
        if (nodes.empty()) return;
        
        // Stack-based traversal to avoid recursion
        std::vector<uint> node_stack;
        node_stack.push_back(0); // Start with root node
        
        while (!node_stack.empty()) {
            uint node_idx = node_stack.back();
            node_stack.pop_back();
            
            if (node_idx >= nodes.size()) {
                continue;
            }
            
            const BVHNode &node = nodes[node_idx];
            nodes_visited++;
        
        // Test if ray intersects node AABB
        float t_min, t_max;
        if (!rayAABBIntersect(origin, direction, node.aabb_min, node.aabb_max, t_min, t_max)) {
            continue;
        }
        
        // Check if intersection is within distance range
        if (max_distance > 0.0f && t_min > max_distance) {
            continue; // Entire AABB is too far - skip
        }
        
        // If we've already found a closer intersection than this AABB, skip it
        if (t_min > nearest_distance) {
            continue;
        }
        
        if (node.is_leaf) {
            // Check each primitive in this leaf for ray intersection
            for (uint i = 0; i < node.primitive_count; i++) {
                uint primitive_id = primitives[node.primitive_start + i];
                
                primitives_checked++;
                
                // Skip if this primitive is not in the candidate set (unless candidate set is empty)
                if (!candidate_UUIDs.empty() && candidate_UUIDs.find(primitive_id) == candidate_UUIDs.end()) {
                    continue;
                }
                
                candidates_found_in_bvh++;
                
                // Get this primitive's AABB
                if (!context->doesPrimitiveExist(primitive_id)) {
                    continue; // Skip invalid primitive
                }
                
                vec3 prim_min, prim_max;
                context->getPrimitiveBoundingBox(primitive_id, prim_min, prim_max);
                
                // Test ray against primitive AABB
                float prim_t_min, prim_t_max;
                if (rayAABBIntersect(origin, direction, prim_min, prim_max, prim_t_min, prim_t_max)) {
                    // Check distance constraints
                    bool within_max_distance = (max_distance <= 0.0f) || (prim_t_min <= max_distance);
                    
                    if (within_max_distance && prim_t_min > 0.0f && prim_t_min < nearest_distance) {
                        // For now, we use AABB intersection distance as an approximation
                        // A more accurate implementation would perform exact ray-primitive intersection
                        nearest_distance = prim_t_min;
                        found_intersection = true;
                    }
                }
            }
        } else {
            // Add child nodes to stack for further traversal
            if (node.left_child != 0xFFFFFFFF) {
                node_stack.push_back(node.left_child);
            }
            if (node.right_child != 0xFFFFFFFF) {
                node_stack.push_back(node.right_child);
            }
        }
        }  // End of while loop
    };  // End of lambda
    
    // First, traverse the static BVH if hierarchical BVH is enabled
    if (check_static_bvh) {
        if (should_debug && printmessages) {
            std::cout << "DEBUG: Traversing static BVH..." << std::endl;
        }
        traverseBVH(static_bvh_nodes, static_bvh_primitives, "static");
    }
    
    // Then, traverse the dynamic BVH
    if (check_dynamic_bvh) {
        if (should_debug && printmessages) {
            std::cout << "DEBUG: Traversing dynamic BVH..." << std::endl;
        }
        traverseBVH(bvh_nodes, primitive_indices, "dynamic");
    }
    
    if (should_debug && printmessages) {
        std::cout << "DEBUG findNearestRayIntersection: visited " << nodes_visited << " nodes, checked " 
                  << primitives_checked << " primitives, found " << candidates_found_in_bvh 
                  << " candidates in BVH, intersection found: " << (found_intersection ? "YES" : "NO") << std::endl;
    }
    
    return found_intersection;
}

bool CollisionDetection::findNearestPrimitiveDistance(const vec3 &origin, const vec3 &direction, const std::vector<uint> &candidate_UUIDs, float &distance, vec3 &obstacle_direction) {
    
    if (candidate_UUIDs.empty()) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findNearestPrimitiveDistance): No candidate UUIDs provided" << std::endl;
        }
        return false;
    }
    
    // Validate that direction is normalized
    float dir_magnitude = direction.magnitude();
    if (std::abs(dir_magnitude - 1.0f) > 1e-6f) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findNearestPrimitiveDistance): Direction vector is not normalized (magnitude = " 
                      << dir_magnitude << ")" << std::endl;
        }
        return false;
    }
    
    // Debug output
    static int debug_call_count = 0;
    debug_call_count++;
    bool should_debug = (debug_call_count % 50) == 1;
    
    // Filter out invalid UUIDs
    std::vector<uint> valid_candidates;
    for (uint uuid : candidate_UUIDs) {
        if (context->doesPrimitiveExist(uuid)) {
            valid_candidates.push_back(uuid);
        } else if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findNearestPrimitiveDistance): Skipping invalid UUID " << uuid << std::endl;
        }
    }
    
    
    if (valid_candidates.empty()) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findNearestPrimitiveDistance): No valid candidate UUIDs after filtering" << std::endl;
        }
        return false;
    }
    
    float nearest_distance_found = std::numeric_limits<float>::max();
    vec3 nearest_obstacle_direction;
    bool found_forward_surface = false;
    
    // Check each candidate primitive to find the nearest "forward-facing" surface
    for (uint primitive_id : valid_candidates) {
        // Get primitive normal and a point on the surface using Context methods
        vec3 surface_normal = context->getPrimitiveNormal(primitive_id);
        std::vector<vec3> vertices = context->getPrimitiveVertices(primitive_id);
        
        if (vertices.empty()) {
            continue; // Skip if no vertices
        }
        
        // Use first vertex as a point on the plane
        vec3 point_on_plane = vertices[0];
        
        // Calculate distance from origin to the plane
        vec3 to_origin = origin - point_on_plane;
        float distance_to_plane = to_origin * surface_normal;
        
        // Distance is the absolute value
        float surface_distance = std::abs(distance_to_plane);
        
        // The direction from origin to closest point on surface
        vec3 surface_direction;
        if (distance_to_plane > 0) {
            // Origin is on the positive side of the normal - direction to surface is -normal
            surface_direction = -surface_normal;
        } else {
            // Origin is on the negative side of the normal - direction to surface is +normal  
            surface_direction = surface_normal;
        }
        
        // Check if this surface is "in front" using dot product
        float dot_product = surface_direction * direction;
        
        if (dot_product > 0.0f) { // Surface is in front of origin
            if (surface_distance < nearest_distance_found) {
                nearest_distance_found = surface_distance;
                nearest_obstacle_direction = surface_direction;
                found_forward_surface = true;
            }
        }
    }
    
    if (found_forward_surface) {
        distance = nearest_distance_found;
        obstacle_direction = nearest_obstacle_direction;
        return true;
    }
    
    
    return false;
}

bool CollisionDetection::findNearestSolidObstacleInCone(const vec3 &apex, const vec3 &axis, float half_angle, float height, const std::vector<uint> &candidate_UUIDs, float &distance, vec3 &obstacle_direction, int num_rays) {
    
    if (candidate_UUIDs.empty()) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findNearestSolidObstacleInCone): No candidate UUIDs provided" << std::endl;
        }
        return false;
    }
    
    // Validate input parameters
    if (half_angle <= 0.0f || half_angle > M_PI/2.0f) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findNearestSolidObstacleInCone): Invalid half_angle " << half_angle << std::endl;
        }
        return false;
    }
    
    if (height <= 0.0f) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::findNearestSolidObstacleInCone): Invalid height " << height << std::endl;
        }
        return false;
    }
    
    // Ensure BVH is current
    ensureBVHCurrent();
    
    // Check if BVH is empty
    if (bvh_nodes.empty()) {
        return false; // No geometry to collide with
    }
    
    // Convert candidate UUIDs to set for efficient lookup
    std::set<uint> candidate_set(candidate_UUIDs.begin(), candidate_UUIDs.end());
    
    // Generate ray directions within the cone
    std::vector<vec3> ray_directions = sampleDirectionsInCone(apex, axis, half_angle, num_rays);
    
    
    float nearest_distance = std::numeric_limits<float>::max();
    vec3 nearest_direction;
    bool found_obstacle = false;
    
    // Cast rays and find nearest intersection
    for (const vec3 &ray_dir : ray_directions) {
        // Traverse BVH to find potential intersections along this ray
        // First, create a ray segment AABB for broad phase
        vec3 ray_end = apex + ray_dir * height;
        vec3 ray_aabb_min = make_vec3(
            std::min(apex.x, ray_end.x),
            std::min(apex.y, ray_end.y),
            std::min(apex.z, ray_end.z)
        );
        vec3 ray_aabb_max = make_vec3(
            std::max(apex.x, ray_end.x),
            std::max(apex.y, ray_end.y),
            std::max(apex.z, ray_end.z)
        );
        
        // Add small epsilon to avoid zero-sized AABBs
        vec3 epsilon(1e-6f, 1e-6f, 1e-6f);
        ray_aabb_min = ray_aabb_min - epsilon;
        ray_aabb_max = ray_aabb_max + epsilon;
        
        // Traverse BVH to find primitives that might intersect this ray
        std::vector<uint> potential_primitives;
        std::stack<uint> node_stack;
        node_stack.push(0); // Start from root
        
        if (false && printmessages) { // Disable verbose BVH debug for performance
            std::cout << "Starting BVH traversal for ray " << ray_dir << " from " << apex << " (height: " << height << ")" << std::endl;
            std::cout << "BVH has " << bvh_nodes.size() << " nodes" << std::endl;
        }
        
        while (!node_stack.empty()) {
            uint node_index = node_stack.top();
            node_stack.pop();
            
            if (node_index >= bvh_nodes.size()) {
                if (printmessages) {
                    std::cout << "WARNING: Invalid node index " << node_index << " >= " << bvh_nodes.size() << std::endl;
                }
                continue;
            }
            
            const BVHNode &node = bvh_nodes[node_index];
            
            if (false && printmessages) { // Disable verbose node debug
                std::cout << "Checking node " << node_index << " AABB: min=" << node.aabb_min << ", max=" << node.aabb_max << std::endl;
            }
            
            // Check if ray AABB intersects node AABB
            if (!aabbIntersect(ray_aabb_min, ray_aabb_max, node.aabb_min, node.aabb_max)) {
                if (false && printmessages) { // Disable verbose AABB debug
                    std::cout << "Ray AABB [" << ray_aabb_min << " to " << ray_aabb_max << "] does not intersect node AABB" << std::endl;
                }
                continue;
            }
            
            // Also check if ray actually intersects the node's AABB (more precise test)
            float t_min, t_max;
            if (!rayAABBIntersect(apex, ray_dir, node.aabb_min, node.aabb_max, t_min, t_max)) {
                continue;
            }
            
            // Check if intersection is within cone height
            if (t_min > height) {
                continue;
            }
            
            if (node.is_leaf) {
                // Add primitives from this leaf node
                for (uint i = 0; i < node.primitive_count; ++i) {
                    uint prim_idx = primitive_indices[node.primitive_start + i];
                    if (candidate_set.find(prim_idx) != candidate_set.end()) {
                        potential_primitives.push_back(prim_idx);
                    }
                }
            } else {
                // Add child nodes to stack
                if (node.left_child != 0xFFFFFFFF) {
                    node_stack.push(node.left_child);
                }
                if (node.right_child != 0xFFFFFFFF) {
                    node_stack.push(node.right_child);
                }
            }
        }
        
        // Test ray against each potential primitive
        
        for (uint prim_uuid : potential_primitives) {
            // Check if this primitive is in our candidate set
            bool is_candidate = candidate_set.find(prim_uuid) != candidate_set.end();
            
            
            if (is_candidate) {
                float intersection_distance;
                if (rayPrimitiveIntersection(apex, ray_dir, prim_uuid, intersection_distance)) {
                    // Check if intersection is within cone height
                    if (intersection_distance <= height && intersection_distance < nearest_distance) {
                        nearest_distance = intersection_distance;
                        nearest_direction = ray_dir;
                        found_obstacle = true;
                    }
                }
            }
        }
        
    }
    
    if (found_obstacle) {
        distance = nearest_distance;
        obstacle_direction = nearest_direction;
        return true;
    }
    
    return false;
}

bool CollisionDetection::detectAttractionPoints(const vec3 &vertex, const vec3 &look_direction, float look_ahead_distance, float half_angle_degrees, const std::vector<vec3> &attraction_points, vec3 &direction_to_closest) {
    
    // Validate input parameters
    if (attraction_points.empty()) {
        return false;
    }
    
    if (look_ahead_distance <= 0.0f) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::detectAttractionPoints): Invalid look-ahead distance (<= 0)" << std::endl;
        }
        return false;
    }
    
    if (half_angle_degrees <= 0.0f || half_angle_degrees >= 180.0f) {
        if (printmessages) {
            std::cerr << "WARNING (CollisionDetection::detectAttractionPoints): Invalid half-angle (must be in range (0, 180) degrees)" << std::endl;
        }
        return false;
    }
    
    // Convert half-angle to radians
    float half_angle_rad = half_angle_degrees * M_PI / 180.0f;
    
    // Normalize look direction
    vec3 axis = look_direction;
    axis.normalize();
    
    // Variables to track the closest attraction point
    bool found_any = false;
    float min_angular_distance = std::numeric_limits<float>::max();
    vec3 closest_point;
    
    // Check each attraction point
    for (const vec3 &point : attraction_points) {
        // Calculate vector from vertex to attraction point
        vec3 to_point = point - vertex;
        float distance_to_point = to_point.magnitude();
        
        // Skip if point is at the vertex or beyond look-ahead distance
        if (distance_to_point < 1e-6f || distance_to_point > look_ahead_distance) {
            continue;
        }
        
        // Normalize the direction to the point
        vec3 direction_to_point = to_point;
        direction_to_point.normalize();
        
        // Calculate angle between look direction and direction to point
        float cos_angle = axis * direction_to_point;
        
        // Clamp to handle numerical precision issues
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
        
        float angle = std::acos(cos_angle);
        
        // Check if point is within the perception cone
        if (angle <= half_angle_rad) {
            found_any = true;
            
            // Check if this is the closest to the centerline
            if (angle < min_angular_distance) {
                min_angular_distance = angle;
                closest_point = point;
            }
        }
    }
    
    // If we found any attraction points, calculate the direction to the closest one
    if (found_any) {
        direction_to_closest = closest_point - vertex;
        direction_to_closest.normalize();
        return true;
    }
    
    return false;
}

std::vector<helios::vec3> CollisionDetection::sampleDirectionsInCone(const vec3 &apex, const vec3 &central_axis, float half_angle, int num_samples) {

    std::vector<vec3> directions;
    directions.reserve(num_samples);

    if (num_samples <= 0 || half_angle <= 0.0f) {
        return directions;
    }

    // Normalize the central axis
    vec3 axis = central_axis;
    axis.normalize();

    // Create an orthonormal basis with the central axis as the primary axis
    vec3 u, v;
    if (std::abs(axis.z) < 0.9f) {
        u = cross(axis, make_vec3(0, 0, 1));
    } else {
        u = cross(axis, make_vec3(1, 0, 0));
    }
    u.normalize();
    v = cross(axis, u);
    v.normalize();

    // Generate uniform samples within the cone using rejection sampling on hemisphere
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

    int samples_generated = 0;
    int max_attempts = num_samples * 10; // Prevent infinite loops
    int attempts = 0;

    while (samples_generated < num_samples && attempts < max_attempts) {
        attempts++;

        // Generate uniform sample on unit hemisphere using spherical coordinates
        float u1 = uniform_dist(gen);
        float u2 = uniform_dist(gen);

        // Use stratified sampling for better distribution
        if (samples_generated > 0) {
            float stratum_u1 = (float) samples_generated / (float) num_samples;
            float stratum_u2 = uniform_dist(gen);
            u1 = (stratum_u1 + u1 / (float) num_samples);
            if (u1 > 1.0f)
                u1 -= 1.0f;
        }

        // Convert to spherical coordinates
        // For uniform sampling within cone, we need:
        // cos(theta) uniformly distributed between cos(half_angle) and 1
        float cos_half_angle = cosf(half_angle);
        float cos_theta = cos_half_angle + u1 * (1.0f - cos_half_angle);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        float phi = 2.0f * M_PI * u2;

        // Convert to Cartesian coordinates in local coordinate system
        float x = sin_theta * cosf(phi);
        float y = sin_theta * sinf(phi);
        float z = cos_theta;

        // Transform from local coordinates to world coordinates
        vec3 local_direction = make_vec3(x, y, z);
        vec3 world_direction = u * local_direction.x + v * local_direction.y + axis * local_direction.z;
        world_direction.normalize();

        // Verify the direction is within the cone (numerical precision check)
        float dot_product = world_direction * axis;
        if (dot_product >= cos_half_angle - 1e-6f) {
            directions.push_back(world_direction);
            samples_generated++;
        }
    }

    // If we couldn't generate enough samples, fill with the central axis
    while (directions.size() < (size_t) num_samples) {
        directions.push_back(axis);
    }

    return directions;
}

CollisionDetection::OptimalPathResult CollisionDetection::findOptimalConePath(const vec3 &apex, const vec3 &centralAxis, float half_angle, float height, int initialSamples) {

    OptimalPathResult result;
    result.direction = centralAxis;
    result.direction.normalize();
    result.collisionCount = 0;
    result.confidence = 0.0f;

    // Validate input parameters
    if (initialSamples <= 0 || half_angle <= 0.0f || half_angle > M_PI) {
        if (printmessages) {
            std::cerr << "WARNING: Invalid parameters for findOptimalConePath" << std::endl;
        }
        return result;
    }

    if (bvh_nodes.empty()) {
        // No geometry to collide with, central axis is optimal
        result.confidence = 1.0f;
        return result;
    }

    // Use fish-eye camera gap detection algorithm
    std::vector<Gap> detected_gaps = detectGapsInCone(apex, centralAxis, half_angle, height, initialSamples);

    if (detected_gaps.empty()) {
        // No gaps found, fall back to central axis
        if (printmessages) {
            std::cerr << "WARNING: No gaps detected in cone, using central axis" << std::endl;
        }
        result.confidence = 0.1f;
        return result;
    }

    // Score gaps using fish-eye metric
    scoreGapsByFishEyeMetric(detected_gaps, centralAxis);

    // Find optimal direction toward highest-scoring gap
    result.direction = findOptimalGapDirection(detected_gaps, centralAxis);

    // Count collisions along optimal direction for reporting
    float max_distance = (height > 0.0f) ? height : -1.0f;
    result.collisionCount = countRayIntersections(apex, result.direction, max_distance);

    // Calculate confidence based on gap quality
    if (!detected_gaps.empty()) {
        // Higher confidence for larger, well-defined gaps
        const Gap &best_gap = detected_gaps[0]; // Assuming first is best after sorting
        result.confidence = std::min(1.0f, best_gap.angular_size * 10.0f); // Scale angular size to confidence
    }

    return result;
}

// -------- GAP DETECTION IMPLEMENTATION --------

std::vector<CollisionDetection::Gap> CollisionDetection::detectGapsInCone(const vec3 &apex, const vec3 &central_axis, float half_angle, float height, int num_samples) {

    std::vector<Gap> gaps;

    // Generate dense ray samples within the cone
    std::vector<vec3> sample_directions = sampleDirectionsInCone(apex, central_axis, half_angle, num_samples);

    if (sample_directions.empty()) {
        return gaps;
    }

    // Cast rays and build ray sample data
    std::vector<RaySample> ray_samples;
    ray_samples.reserve(sample_directions.size());

    float max_distance = (height > 0.0f) ? height : -1.0f;

    for (const vec3 &direction: sample_directions) {
        RaySample sample;
        sample.direction = direction;

        // Find distance to first collision
        int collision_count = countRayIntersections(apex, direction, max_distance);

        if (collision_count == 0) {
            sample.distance = (max_distance > 0.0f) ? max_distance : 1000.0f; // Large value for "infinite"
            sample.is_free = true;
        } else {
            // Calculate actual distance by doing detailed ray casting
            sample.distance = 0.0f; // For now, just mark as blocked
            sample.is_free = false;
        }

        ray_samples.push_back(sample);
    }

    // Debug: Count free vs blocked samples
    int free_count = 0, blocked_count = 0;
    for (const auto& sample : ray_samples) {
        if (sample.is_free) free_count++;
        else blocked_count++;
    }

    // Use a more sophisticated gap detection approach based on contiguous free regions
    // First, sort samples by angular position relative to central axis to identify contiguous regions
    std::vector<std::pair<float, size_t>> angular_positions;
    for (size_t i = 0; i < ray_samples.size(); ++i) {
        if (ray_samples[i].is_free) {
            // Calculate angular position in cone-relative coordinates
            float dot_product = ray_samples[i].direction * central_axis;
            dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
            float angular_from_center = acosf(dot_product);
            angular_positions.push_back({angular_from_center, i});
        }
    }

    if (angular_positions.empty()) {
        return gaps; // No free samples
    }

    // Sort by angular position
    std::sort(angular_positions.begin(), angular_positions.end());

    // Find contiguous free regions (gaps) with minimum size threshold
    std::vector<bool> processed(ray_samples.size(), false);
    float min_gap_angular_size = half_angle * 0.05f; // 5% of cone angle minimum gap size
    
    for (size_t start = 0; start < angular_positions.size(); ++start) {
        size_t start_idx = angular_positions[start].second;
        if (processed[start_idx]) continue;

        Gap new_gap;
        new_gap.sample_indices.push_back(start_idx);
        processed[start_idx] = true;

        // Extend gap by finding nearby free samples using k-nearest neighbor approach
        std::vector<float> distances_to_start;
        for (size_t j = 0; j < ray_samples.size(); ++j) {
            if (j != start_idx && ray_samples[j].is_free && !processed[j]) {
                float dot_product = ray_samples[start_idx].direction * ray_samples[j].direction;
                dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
                float angular_distance = acosf(dot_product);
                distances_to_start.push_back(angular_distance);
            } else {
                distances_to_start.push_back(999.0f); // Large value for excluded samples
            }
        }

        // Add nearby samples to gap using adaptive threshold
        float sample_density = 2.0f * half_angle / sqrtf((float)num_samples);
        float adaptive_threshold = sample_density * 3.0f; // 3x sample spacing for connection
        
        for (size_t j = 0; j < ray_samples.size(); ++j) {
            if (j != start_idx && ray_samples[j].is_free && !processed[j] && 
                distances_to_start[j] < adaptive_threshold) {
                new_gap.sample_indices.push_back(j);
                processed[j] = true;
            }
        }

        // Only keep gaps that meet minimum size requirements
        if (new_gap.sample_indices.size() >= 5) { // Require at least 5 samples for a valid gap
            gaps.push_back(new_gap);
        }
    }

    // Calculate gap properties
    for (Gap &gap: gaps) {
        // Calculate gap center direction (average of constituent directions)
        vec3 center(0, 0, 0);
        for (int idx: gap.sample_indices) {
            center = center + ray_samples[idx].direction;
        }
        center = center / (float) gap.sample_indices.size();
        center.normalize();
        gap.center_direction = center;

        // Calculate angular size
        std::vector<RaySample> gap_samples;
        for (int idx: gap.sample_indices) {
            gap_samples.push_back(ray_samples[idx]);
        }
        gap.angular_size = calculateGapAngularSize(gap_samples, central_axis);

        // Calculate angular distance from central axis
        float dot_product = gap.center_direction * central_axis;
        dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
        gap.angular_distance = acosf(dot_product);
    }

    // OPTIMIZATION: Spatial filtering - remove gaps that are too far from central axis
    // This reduces the number of gaps passed to the expensive scoring function
    if (gaps.size() > 10) {
        float max_angular_distance = half_angle * 0.8f; // Only consider gaps within 80% of cone angle
        
        auto it = std::remove_if(gaps.begin(), gaps.end(), [max_angular_distance](const Gap& gap) {
            return gap.angular_distance > max_angular_distance;
        });
        
        gaps.erase(it, gaps.end());
        
        // If filtering removed too many gaps, keep at least the closest ones
        if (gaps.size() < 3 && gaps.size() > 0) {
            // Sort by angular distance and keep the closest ones
            std::partial_sort(gaps.begin(), gaps.begin() + std::min(size_t(3), gaps.size()), gaps.end(),
                            [](const Gap& a, const Gap& b) { return a.angular_distance < b.angular_distance; });
        }
    }

    return gaps;
}

float CollisionDetection::calculateGapAngularSize(const std::vector<RaySample> &gap_samples, const vec3 &central_axis) {

    if (gap_samples.empty()) {
        return 0.0f;
    }

    // Find the angular extent of the gap by finding min/max angles
    float min_angle = M_PI;
    float max_angle = 0.0f;

    for (const RaySample &sample: gap_samples) {
        float dot_product = sample.direction * central_axis;
        dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
        float angle = acosf(dot_product);

        min_angle = std::min(min_angle, angle);
        max_angle = std::max(max_angle, angle);
    }

    // Simple approximation: angular size as solid angle
    float angular_width = max_angle - min_angle;

    // Convert to approximate solid angle (steradians)
    // This is a rough approximation: solid_angle ≈ π * (angular_width)^2
    float solid_angle = M_PI * angular_width * angular_width;

    return solid_angle;
}

void CollisionDetection::scoreGapsByFishEyeMetric(std::vector<Gap> &gaps, const vec3 &central_axis) {

    // Early exit for small gap counts - full sort is fine
    if (gaps.size() <= 10) {
        for (Gap &gap: gaps) {
            // Fish-eye metric: prefer larger gaps closer to central axis
            // Gap size component (logarithmic scaling for larger gaps)
            float size_score = log(1.0f + gap.angular_size * 100.0f); // Scale up angular size
            // Distance penalty (exponential penalty for gaps far from center)
            float distance_penalty = exp(gap.angular_distance * 2.0f);
            // Combined score (higher is better)
            gap.score = size_score / distance_penalty;
        }
        // Sort gaps by score (highest first)
        std::sort(gaps.begin(), gaps.end(), [](const Gap &a, const Gap &b) { return a.score > b.score; });
        return;
    }
    
    // OPTIMIZATION: For large gap counts, use partial sorting
    // We only need the top 3-5 gaps for collision avoidance
    const size_t max_gaps_needed = std::min(size_t(5), gaps.size());
    
    // Calculate scores for all gaps
    for (Gap &gap: gaps) {
        // Fish-eye metric: prefer larger gaps closer to central axis

        // Gap size component (logarithmic scaling for larger gaps)
        float size_score = log(1.0f + gap.angular_size * 100.0f); // Scale up angular size

        // Distance penalty (exponential penalty for gaps far from center)
        float distance_penalty = exp(gap.angular_distance * 2.0f);

        // Combined score (higher is better)
        gap.score = size_score / distance_penalty;
    }
    
    // Use partial_sort to get only the top N gaps - O(n log k) instead of O(n log n)
    std::partial_sort(gaps.begin(), gaps.begin() + max_gaps_needed, gaps.end(), 
                     [](const Gap &a, const Gap &b) { return a.score > b.score; });
    
    // Resize to keep only the top gaps to avoid processing unnecessary gaps later
    gaps.resize(max_gaps_needed);
}

helios::vec3 CollisionDetection::findOptimalGapDirection(const std::vector<Gap> &gaps, const vec3 &central_axis) {

    if (gaps.empty()) {
        // No gaps found, return central axis
        vec3 result = central_axis;
        result.normalize();
        return result;
    }

    // Return direction toward the highest-scoring gap
    const Gap &best_gap = gaps[0];
    return best_gap.center_direction;
}

// -------- SPATIAL OPTIMIZATION METHODS --------

std::vector<std::pair<uint, uint>> CollisionDetection::findCollisionsWithinDistance(const std::vector<uint> &query_UUIDs, const std::vector<uint> &target_UUIDs, float max_distance) {
    
    std::vector<std::pair<uint, uint>> collision_pairs;

    
    // Update primitive centroids cache for target geometry
    for (uint target_id : target_UUIDs) {
        if (primitive_centroids_cache.find(target_id) == primitive_centroids_cache.end()) {
            // Calculate and cache centroid for this primitive
            std::vector<vec3> vertices = context->getPrimitiveVertices(target_id);
            if (!vertices.empty()) {
                vec3 centroid = make_vec3(0, 0, 0);
                for (const vec3& vertex : vertices) {
                    centroid = centroid + vertex;
                }
                centroid = centroid / float(vertices.size());
                primitive_centroids_cache[target_id] = centroid;
            }
        }
    }
    
    // For each query primitive, find nearby targets within distance
    for (uint query_id : query_UUIDs) {
        // Get query centroid
        std::vector<vec3> query_vertices = context->getPrimitiveVertices(query_id);
        if (query_vertices.empty()) continue;
        
        vec3 query_centroid = make_vec3(0, 0, 0);
        for (const vec3& vertex : query_vertices) {
            query_centroid = query_centroid + vertex;
        }
        query_centroid = query_centroid / float(query_vertices.size());
        
        // Check distance to each target
        for (uint target_id : target_UUIDs) {
            if (query_id == target_id) continue; // Skip self-collision
            
            auto target_centroid_it = primitive_centroids_cache.find(target_id);
            if (target_centroid_it != primitive_centroids_cache.end()) {
                vec3 target_centroid = target_centroid_it->second;
                float distance = (query_centroid - target_centroid).magnitude();
                
                if (distance <= max_distance) {
                    // Within distance threshold, check for actual collision
                    std::vector<uint> single_query = {query_id};
                    std::vector<uint> single_target = {target_id};
                    std::vector<uint> empty_objects;
                    
                    std::vector<uint> collisions = findCollisions(single_query, empty_objects, single_target, empty_objects);
                    if (!collisions.empty()) {
                        collision_pairs.push_back(std::make_pair(query_id, target_id));
                    }
                }
            }
        }
    }
    
    if (printmessages) {
        std::cout << "Found " << collision_pairs.size() << " collision pairs within distance threshold" << std::endl;
    }
    
    return collision_pairs;
}

void CollisionDetection::setMaxCollisionDistance(float distance) {
    if (distance <= 0.0f) {
        helios_runtime_error("ERROR (CollisionDetection::setMaxCollisionDistance): Distance must be positive");
    }
    
    max_collision_distance = distance;
    
    if (printmessages) {
        std::cout << "Set maximum collision distance to " << distance << " meters" << std::endl;
    }
}

float CollisionDetection::getMaxCollisionDistance() const {
    return max_collision_distance;
}

std::vector<uint> CollisionDetection::filterGeometryByDistance(const helios::vec3 &query_center, float max_radius, const std::vector<uint> &candidate_UUIDs) {
    
    std::vector<uint> filtered_UUIDs;
    
    if (printmessages) {
        //std::cout << "Filtering geometry within radius " << max_radius << " of center " << query_center << std::endl;
    }
    
    // Get list of candidates (either provided or all primitives)
    std::vector<uint> candidates;
    if (candidate_UUIDs.empty()) {
        candidates = context->getAllUUIDs();
    } else {
        candidates = candidate_UUIDs;
    }
    
    // Filter candidates by distance
    for (uint candidate_id : candidates) {
        // Skip if primitive doesn't exist
        if (!context->doesPrimitiveExist(candidate_id)) {
            continue;
        }
        
        // Get primitive centroid (calculate if not cached)
        vec3 centroid;
        auto cache_it = primitive_centroids_cache.find(candidate_id);
        if (cache_it != primitive_centroids_cache.end()) {
            centroid = cache_it->second;
        } else {
            // Calculate and cache centroid
            std::vector<vec3> vertices = context->getPrimitiveVertices(candidate_id);
            if (vertices.empty()) continue;
            
            centroid = make_vec3(0, 0, 0);
            for (const vec3& vertex : vertices) {
                centroid = centroid + vertex;
            }
            centroid = centroid / float(vertices.size());
            primitive_centroids_cache[candidate_id] = centroid;
        }
        
        // Check distance from query center
        float distance = (query_center - centroid).magnitude();
        if (distance <= max_radius) {
            filtered_UUIDs.push_back(candidate_id);
        }
    }
    
    if (printmessages) {
        //std::cout << "Filtered " << candidates.size() << " candidates to " << filtered_UUIDs.size() << " within radius" << std::endl;
    }
    
    return filtered_UUIDs;
}

// -------- VOXEL RAY PATH LENGTH CALCULATIONS --------

void CollisionDetection::calculateVoxelRayPathLengths(const vec3 &grid_center, const vec3 &grid_size, const helios::int3 &grid_divisions, const std::vector<vec3> &ray_origins, const std::vector<vec3> &ray_directions) {

    if (ray_origins.size() != ray_directions.size()) {
        helios_runtime_error("ERROR (CollisionDetection::calculateVoxelRayPathLengths): ray_origins and ray_directions vectors must have same size");
    }

    if (ray_origins.empty()) {
        if (printmessages) {
            std::cout << "WARNING (CollisionDetection::calculateVoxelRayPathLengths): No rays provided" << std::endl;
        }
        return;
    }

    // Initialize voxel data structures for the given grid
    initializeVoxelData(grid_center, grid_size, grid_divisions);

    if (printmessages) {
        std::cout << "Calculating voxel ray path lengths for " << ray_origins.size() << " rays on " 
                  << grid_divisions.x << "x" << grid_divisions.y << "x" << grid_divisions.z << " voxel grid..." << std::endl;
    }

    // Choose GPU or CPU implementation based on acceleration setting
    if (isGPUAccelerationEnabled()) {
        calculateVoxelRayPathLengths_GPU(ray_origins, ray_directions);
    } else {
        calculateVoxelRayPathLengths_CPU(ray_origins, ray_directions);
    }

    if (printmessages) {
        std::cout << "Voxel ray path length calculation completed." << std::endl;
    }
}

void CollisionDetection::setVoxelTransmissionProbability(int P_denom, int P_trans, const helios::int3 &ijk) {
    if (!validateVoxelIndices(ijk)) {
        helios_runtime_error("ERROR (CollisionDetection::setVoxelTransmissionProbability): Invalid voxel indices");
    }
    
    if (!voxel_data_initialized) {
        helios_runtime_error("ERROR (CollisionDetection::setVoxelTransmissionProbability): Voxel data not initialized. Call calculateVoxelRayPathLengths first.");
    }

    voxel_ray_counts[ijk.x][ijk.y][ijk.z] = P_denom;
    voxel_transmitted[ijk.x][ijk.y][ijk.z] = P_trans;
}

void CollisionDetection::getVoxelTransmissionProbability(const helios::int3 &ijk, int &P_denom, int &P_trans) const {
    if (!validateVoxelIndices(ijk)) {
        helios_runtime_error("ERROR (CollisionDetection::getVoxelTransmissionProbability): Invalid voxel indices");
    }
    
    if (!voxel_data_initialized) {
        P_denom = 0;
        P_trans = 0;
        return;
    }

    P_denom = voxel_ray_counts[ijk.x][ijk.y][ijk.z];
    P_trans = voxel_transmitted[ijk.x][ijk.y][ijk.z];
}

void CollisionDetection::setVoxelRbar(float r_bar, const helios::int3 &ijk) {
    if (!validateVoxelIndices(ijk)) {
        helios_runtime_error("ERROR (CollisionDetection::setVoxelRbar): Invalid voxel indices");
    }
    
    if (!voxel_data_initialized) {
        helios_runtime_error("ERROR (CollisionDetection::setVoxelRbar): Voxel data not initialized. Call calculateVoxelRayPathLengths first.");
    }

    // Store r_bar * ray_count so that getVoxelRbar returns the correct value when it divides
    int ray_count = voxel_ray_counts[ijk.x][ijk.y][ijk.z];
    if (ray_count == 0) {
        ray_count = 1; // Set to 1 to avoid division by zero
        voxel_ray_counts[ijk.x][ijk.y][ijk.z] = 1;
    }
    voxel_path_lengths[ijk.x][ijk.y][ijk.z] = r_bar * static_cast<float>(ray_count);
}

float CollisionDetection::getVoxelRbar(const helios::int3 &ijk) const {
    if (!validateVoxelIndices(ijk)) {
        helios_runtime_error("ERROR (CollisionDetection::getVoxelRbar): Invalid voxel indices");
    }
    
    if (!voxel_data_initialized) {
        return 0.0f;
    }

    int ray_count = voxel_ray_counts[ijk.x][ijk.y][ijk.z];
    if (ray_count == 0) {
        return 0.0f;
    }

    // If this was set directly via setVoxelRbar, return it as-is
    // If this accumulated from ray calculations, compute the average
    return voxel_path_lengths[ijk.x][ijk.y][ijk.z] / static_cast<float>(ray_count);
}

void CollisionDetection::clearVoxelData() {
    voxel_ray_counts.clear();
    voxel_transmitted.clear();
    voxel_path_lengths.clear();
    voxel_data_initialized = false;
    
    if (printmessages) {
        std::cout << "Voxel data cleared." << std::endl;
    }
}

// -------- VOXEL RAY PATH LENGTH HELPER METHODS --------

void CollisionDetection::initializeVoxelData(const vec3 &grid_center, const vec3 &grid_size, const helios::int3 &grid_divisions) {
    
    // Check if we need to reinitialize (grid parameters changed)
    bool need_reinit = !voxel_data_initialized || 
                      (grid_center - voxel_grid_center).magnitude() > 1e-6 ||
                      (grid_size - voxel_grid_size).magnitude() > 1e-6 ||
                      grid_divisions.x != voxel_grid_divisions.x ||
                      grid_divisions.y != voxel_grid_divisions.y ||
                      grid_divisions.z != voxel_grid_divisions.z;
    
    if (!need_reinit) {
        // Just clear existing data but keep structure
        for (int i = 0; i < grid_divisions.x; i++) {
            for (int j = 0; j < grid_divisions.y; j++) {
                for (int k = 0; k < grid_divisions.z; k++) {
                    voxel_ray_counts[i][j][k] = 0;
                    voxel_transmitted[i][j][k] = 0;
                    voxel_path_lengths[i][j][k] = 0.0f;
                }
            }
        }
        return;
    }
    
    // Store grid parameters
    voxel_grid_center = grid_center;
    voxel_grid_size = grid_size;
    voxel_grid_divisions = grid_divisions;
    
    // Resize and initialize voxel data structures
    voxel_ray_counts.resize(grid_divisions.x);
    voxel_transmitted.resize(grid_divisions.x);
    voxel_path_lengths.resize(grid_divisions.x);
    
    for (int i = 0; i < grid_divisions.x; i++) {
        voxel_ray_counts[i].resize(grid_divisions.y);
        voxel_transmitted[i].resize(grid_divisions.y);
        voxel_path_lengths[i].resize(grid_divisions.y);
        
        for (int j = 0; j < grid_divisions.y; j++) {
            voxel_ray_counts[i][j].resize(grid_divisions.z, 0);
            voxel_transmitted[i][j].resize(grid_divisions.z, 0);
            voxel_path_lengths[i][j].resize(grid_divisions.z, 0.0f);
        }
    }
    
    voxel_data_initialized = true;
    
    if (printmessages) {
        std::cout << "Initialized voxel data structures: " << grid_divisions.x << "x" << grid_divisions.y << "x" << grid_divisions.z << " grid" << std::endl;
    }
}

bool CollisionDetection::validateVoxelIndices(const helios::int3 &ijk) const {
    return (ijk.x >= 0 && ijk.x < voxel_grid_divisions.x &&
            ijk.y >= 0 && ijk.y < voxel_grid_divisions.y &&
            ijk.z >= 0 && ijk.z < voxel_grid_divisions.z);
}

void CollisionDetection::calculateVoxelAABB(const helios::int3 &ijk, vec3 &voxel_min, vec3 &voxel_max) const {
    vec3 voxel_size = make_vec3(voxel_grid_size.x / static_cast<float>(voxel_grid_divisions.x),
                               voxel_grid_size.y / static_cast<float>(voxel_grid_divisions.y),
                               voxel_grid_size.z / static_cast<float>(voxel_grid_divisions.z));
    
    vec3 grid_min = voxel_grid_center - 0.5f * voxel_grid_size;
    
    voxel_min = grid_min + make_vec3(static_cast<float>(ijk.x) * voxel_size.x,
                                    static_cast<float>(ijk.y) * voxel_size.y,
                                    static_cast<float>(ijk.z) * voxel_size.z);
    
    voxel_max = voxel_min + voxel_size;
}

void CollisionDetection::calculateVoxelRayPathLengths_CPU(const std::vector<vec3> &ray_origins, const std::vector<vec3> &ray_directions) {
    
    if (printmessages) {
        std::cout << "Using CPU implementation for voxel ray path length calculations..." << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Ensure BVH is current
    ensureBVHCurrent();
    
    const int num_rays = static_cast<int>(ray_origins.size());
    
    // Use OpenMP for parallel processing
    #pragma omp parallel for schedule(dynamic)
    for (int ray_idx = 0; ray_idx < num_rays; ray_idx++) {
        const vec3 &ray_origin = ray_origins[ray_idx];
        const vec3 &ray_direction = ray_directions[ray_idx];
        
        // For each ray, check intersection with each voxel
        for (int i = 0; i < voxel_grid_divisions.x; i++) {
            for (int j = 0; j < voxel_grid_divisions.y; j++) {
                for (int k = 0; k < voxel_grid_divisions.z; k++) {
                    
                    helios::int3 voxel_idx = helios::make_int3(i, j, k);
                    vec3 voxel_min, voxel_max;
                    calculateVoxelAABB(voxel_idx, voxel_min, voxel_max);
                    
                    // Test ray-voxel intersection
                    float t_min, t_max;
                    if (rayAABBIntersect(ray_origin, ray_direction, voxel_min, voxel_max, t_min, t_max)) {
                        
                        // Ensure intersection is in forward direction and valid
                        if (t_max > 1e-6 && t_min < t_max) {
                            
                            // Calculate path length through voxel
                            float path_length = t_max - t_min;
                            if (t_min < 0) {
                                // Ray starts inside voxel
                                path_length = t_max;
                            }
                            
                            // Thread-safe accumulation using atomic operations
                            #pragma omp atomic
                            voxel_ray_counts[i][j][k]++;
                            
                            #pragma omp atomic
                            voxel_path_lengths[i][j][k] += path_length;
                            
                            // For now, assume all rays are transmitted (no interception testing)
                            #pragma omp atomic
                            voxel_transmitted[i][j][k]++;
                        }
                    }
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (printmessages) {
        std::cout << "CPU voxel ray path length calculation completed in " << duration.count() << " ms" << std::endl;
        
        // Report some statistics
        int total_ray_voxel_intersections = 0;
        for (int i = 0; i < voxel_grid_divisions.x; i++) {
            for (int j = 0; j < voxel_grid_divisions.y; j++) {
                for (int k = 0; k < voxel_grid_divisions.z; k++) {
                    total_ray_voxel_intersections += voxel_ray_counts[i][j][k];
                }
            }
        }
        std::cout << "Total ray-voxel intersections: " << total_ray_voxel_intersections << std::endl;
    }
}

void CollisionDetection::calculateVoxelRayPathLengths_GPU(const std::vector<vec3> &ray_origins, const std::vector<vec3> &ray_directions) {
#ifdef HELIOS_CUDA_AVAILABLE
    if (printmessages) {
        std::cout << "Using GPU implementation for voxel ray path length calculations..." << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int num_rays = static_cast<int>(ray_origins.size());
    const int total_voxels = voxel_grid_divisions.x * voxel_grid_divisions.y * voxel_grid_divisions.z;
    
    // Prepare data for GPU kernel
    std::vector<float> h_ray_origins(num_rays * 3);
    std::vector<float> h_ray_directions(num_rays * 3);
    std::vector<int> h_voxel_ray_counts(total_voxels, 0);
    std::vector<float> h_voxel_path_lengths(total_voxels, 0.0f);
    std::vector<int> h_voxel_transmitted(total_voxels, 0);
    
    // Convert ray data to flat arrays
    for (int i = 0; i < num_rays; i++) {
        h_ray_origins[i * 3 + 0] = ray_origins[i].x;
        h_ray_origins[i * 3 + 1] = ray_origins[i].y;
        h_ray_origins[i * 3 + 2] = ray_origins[i].z;
        
        h_ray_directions[i * 3 + 0] = ray_directions[i].x;
        h_ray_directions[i * 3 + 1] = ray_directions[i].y;
        h_ray_directions[i * 3 + 2] = ray_directions[i].z;
    }
    
    // Launch CUDA kernel
    launchVoxelRayPathLengths(
        num_rays,
        h_ray_origins.data(),
        h_ray_directions.data(),
        voxel_grid_center.x, voxel_grid_center.y, voxel_grid_center.z,
        voxel_grid_size.x, voxel_grid_size.y, voxel_grid_size.z,
        voxel_grid_divisions.x, voxel_grid_divisions.y, voxel_grid_divisions.z,
        h_voxel_ray_counts.data(),
        h_voxel_path_lengths.data(),
        h_voxel_transmitted.data()
    );
    
    // Copy results back to class data structures
    for (int i = 0; i < voxel_grid_divisions.x; i++) {
        for (int j = 0; j < voxel_grid_divisions.y; j++) {
            for (int k = 0; k < voxel_grid_divisions.z; k++) {
                int flat_idx = i * voxel_grid_divisions.y * voxel_grid_divisions.z + j * voxel_grid_divisions.z + k;
                voxel_ray_counts[i][j][k] = h_voxel_ray_counts[flat_idx];
                voxel_path_lengths[i][j][k] = h_voxel_path_lengths[flat_idx];
                voxel_transmitted[i][j][k] = h_voxel_transmitted[flat_idx];
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (printmessages) {
        std::cout << "GPU voxel ray path length calculation completed in " << duration.count() << " ms" << std::endl;
        
        // Report some statistics
        int total_ray_voxel_intersections = 0;
        for (const auto& count : h_voxel_ray_counts) {
            total_ray_voxel_intersections += count;
        }
        std::cout << "Total ray-voxel intersections: " << total_ray_voxel_intersections << std::endl;
    }
    
#else
    if (printmessages) {
        std::cout << "CUDA not available, using CPU implementation..." << std::endl;
    }
    calculateVoxelRayPathLengths_CPU(ray_origins, ray_directions);
#endif
}

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
        // Use all primitives in context
        search_primitives = context->getAllUUIDs();
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
        if (!context->doesPrimitiveExist(candidate_uuid)) {
            continue;
        }
        
        float intersection_distance;
        if (rayPrimitiveIntersection(origin, ray_direction, candidate_uuid, intersection_distance)) {
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
    // Process each ray sequentially (automatic BVH rebuilds should be disabled by caller for performance)
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
    
    bvh_nodes_soa.clear();
    bvh_nodes_soa.reserve(bvh_nodes.size());
    
    for (const auto& node : bvh_nodes) {
        bvh_nodes_soa.aabb_mins.push_back(node.aabb_min);
        bvh_nodes_soa.aabb_maxs.push_back(node.aabb_max);
        bvh_nodes_soa.left_children.push_back(node.left_child);
        bvh_nodes_soa.right_children.push_back(node.right_child);
        bvh_nodes_soa.primitive_starts.push_back(node.primitive_start);
        bvh_nodes_soa.primitive_counts.push_back(node.primitive_count);
        bvh_nodes_soa.is_leaf_flags.push_back(node.is_leaf ? 1 : 0);
        bvh_nodes_soa.node_count++;
    }
    
    if (printmessages) {
        std::cout << "CollisionDetection: Converted " << bvh_nodes_soa.node_count << " nodes to SoA layout" << std::endl;
    }
}

void CollisionDetection::convertLegacyToQuantized() {
    if (bvh_nodes.empty()) return;
    
    // Calculate scene bounding box
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
    
    // Initialize quantized BVH
    bvh_nodes_quantized = QuantizedBVHNodes();
    bvh_nodes_quantized.initializeQuantization(scene_min, scene_max);
    
    // Convert each node
    for (const auto& node : bvh_nodes) {
        bvh_nodes_quantized.addNode(
            node.aabb_min, node.aabb_max,
            node.left_child, node.right_child,
            node.primitive_start, node.primitive_count,
            node.is_leaf
        );
    }
    
    if (printmessages) {
        std::cout << "CollisionDetection: Converted " << bvh_nodes_quantized.node_count 
                  << " nodes to quantized layout with scene bounds: " 
                  << scene_min << " to " << scene_max << std::endl;
    }
}

void CollisionDetection::convertSoAToLegacy() {
    if (bvh_nodes_soa.node_count == 0) return;
    
    bvh_nodes.clear();
    bvh_nodes.reserve(bvh_nodes_soa.node_count);
    
    for (size_t i = 0; i < bvh_nodes_soa.node_count; ++i) {
        BVHNode node;
        node.aabb_min = bvh_nodes_soa.aabb_mins[i];
        node.aabb_max = bvh_nodes_soa.aabb_maxs[i];
        node.left_child = bvh_nodes_soa.left_children[i];
        node.right_child = bvh_nodes_soa.right_children[i];
        node.primitive_start = bvh_nodes_soa.primitive_starts[i];
        node.primitive_count = bvh_nodes_soa.primitive_counts[i];
        node.is_leaf = bvh_nodes_soa.is_leaf_flags[i] != 0;
        bvh_nodes.push_back(node);
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

void CollisionDetection::convertQuantizedToLegacy() {
    if (bvh_nodes_quantized.node_count == 0) return;
    
    bvh_nodes.clear();
    bvh_nodes.reserve(bvh_nodes_quantized.node_count);
    
    for (size_t i = 0; i < bvh_nodes_quantized.node_count; ++i) {
        BVHNode node;
        bvh_nodes_quantized.getAABB(i, node.aabb_min, node.aabb_max);
        node.left_child = bvh_nodes_quantized.left_children[i];
        node.right_child = bvh_nodes_quantized.right_children[i];
        
        // Unpack primitive data
        uint32_t packed = bvh_nodes_quantized.primitive_data[i];
        node.primitive_start = (packed >> 1) & 0x7FFFFFFF;
        node.primitive_count = (packed >> 16) & 0x7FFF;
        node.is_leaf = (packed & 1) != 0;
        
        bvh_nodes.push_back(node);
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
        case BVHOptimizationMode::LEGACY_AOS:
            results = castRays(ray_queries, &local_stats); // Fall back to original method
            break;
            
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
        // Fall back to legacy method if SoA not built
        return castRays(ray_queries, &stats);
    }
    
    stats.total_rays_cast = ray_queries.size();
    
    // Process each ray using SoA traversal
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
    std::vector<HitResult> results;
    results.reserve(ray_queries.size());
    
    if (bvh_nodes_quantized.node_count == 0) {
        // Fall back to legacy method if quantized BVH not built
        return castRays(ray_queries, &stats);
    }
    
    stats.total_rays_cast = ray_queries.size();
    
    // Process each ray using quantized traversal
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
    // This method delegates to the existing primitive intersection logic
    // For now, we'll use the single-ray castRay method on individual primitives
    std::vector<uint> single_primitive = {primitive_id};
    RayQuery single_query(query.origin, query.direction, query.max_distance, single_primitive);
    return castRay(single_query);
}

#ifdef HELIOS_CUDA_AVAILABLE
#include <vector_types.h>  // For make_float3

/**
 * \brief Phase 3: Cast rays using warp-efficient GPU kernels
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
    
    // Call Phase 3 warp-efficient GPU kernel
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
#endif