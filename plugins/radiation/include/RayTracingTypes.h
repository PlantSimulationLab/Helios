/** \file "RayTracingTypes.h" Backend-agnostic data structures for ray tracing.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef RAYTRACING_TYPES_H
#define RAYTRACING_TYPES_H

#include "Context.h"
#include "IndexTypes.h"
#include <vector>

namespace helios {

    /**
     * @brief Geometry data for a specific primitive type
     */
    struct PrimitiveTypeGeometry {
        std::vector<helios::vec3> vertices; //!< Vertex positions (layout depends on type)
        std::vector<uint> UUIDs; //!< Primitive UUIDs for this type
        size_t count = 0; //!< Number of primitives of this type
    };

    /**
     * @brief Geometry data for ray tracing (backend-agnostic)
     *
     * All geometry data for 6 primitive types: patch, triangle, disk, tile, voxel, bbox.
     * Data is organized per-primitive and per-type.
     */
    struct RayTracingGeometry {

        // Per-primitive data (indexed by primitive index 0..primitive_count-1)
        std::vector<float> transform_matrices; //!< Transform matrices (16 floats per primitive, row-major)
        std::vector<uint> primitive_types; //!< Primitive type: 0=patch, 1=triangle, 2=disk, 3=tile, 4=voxel, 5=bbox
        std::vector<uint> primitive_UUIDs; //!< UUID for each primitive (for result mapping)
        std::vector<uint> primitive_IDs; //!< Primitive ID within parent object (for sub-patch calculations)
        std::vector<uint> object_IDs; //!< Parent object ID for each primitive
        std::vector<helios::int2> object_subdivisions; //!< Object subdivisions (for mapping)
        std::vector<char> twosided_flags; //!< Two-sided flag (1=two-sided, 0=one-sided)
        std::vector<float> solid_fractions; //!< Solid fraction (0-1, for porous surfaces)

        // UUID-to-position lookup table (indexed by UUID value)
        //!< Maps UUID to array position: primitive_positions[UUID] = array_index
        //!< Size: max_UUID + 1 (sparse array, entries for non-existent UUIDs = UINT_MAX)
        //!< Enables CUDA to convert UUID to array position for buffer indexing
        std::vector<uint> primitive_positions;

        // Per-type geometry data
        PrimitiveTypeGeometry patches; //!< Patch data (4 vertices per patch)
        PrimitiveTypeGeometry triangles; //!< Triangle data (3 vertices per triangle)
        PrimitiveTypeGeometry tiles; //!< Tile data (4 vertices per tile)
        PrimitiveTypeGeometry voxels; //!< Voxel data (8 vertices per voxel)
        PrimitiveTypeGeometry bboxes; //!< Bounding box data (8 vertices per bbox)

        // Disk-specific data (disks don't use vertex arrays)
        std::vector<helios::vec3> disk_centers; //!< Disk center positions
        std::vector<float> disk_radii; //!< Disk radii
        std::vector<helios::vec3> disk_normals; //!< Disk normal vectors
        std::vector<uint> disk_UUIDs; //!< Disk UUIDs
        size_t disk_count = 0; //!< Number of disks

        // Texture/masking data
        std::vector<bool> mask_data; //!< Flattened 3D mask data [mask_id][y][x]
        std::vector<helios::int2> mask_sizes; //!< Size (width, height) for each mask
        std::vector<int> mask_IDs; //!< Mask ID for each primitive (-1 if no mask)
        std::vector<helios::vec2> uv_data; //!< Flattened 2D UV coordinates [primitive][vertex]
        std::vector<int> uv_IDs; //!< UV data start index for each primitive (-1 if no UVs)

        // Periodic boundary condition
        helios::vec2 periodic_flag; //!< Periodic boundary flags (x, y directions)

        // Counts
        size_t primitive_count = 0; //!< Total number of primitives (all types)
        size_t patch_count = 0; //!< Number of patches
        size_t triangle_count = 0; //!< Number of triangles
        size_t tile_count = 0; //!< Number of tiles
        size_t voxel_count = 0; //!< Number of voxels
        size_t bbox_count = 0; //!< Number of bounding boxes
        uint bbox_UUID_base = 0; //!< Starting UUID for bboxes (max_real_UUID + 1)

        // UUID ↔ Position mapping utility (CPU-side type-safe conversion)
        UUIDPositionMapper mapper;

        // ========== Helper Methods for Backend Implementation ==========

        /**
         * @brief Get expected buffer size for per-primitive data
         * @return primitive_count
         *
         * Use when allocating buffers indexed by position (0..primitive_count-1).
         * CRITICAL: Use this instead of Nobjects to avoid buffer undersizing bugs!
         *
         * Buffers to size with this:
         * - transform_matrices: size = getPerPrimitiveBufferSize() * 16
         * - object_subdivisions: size = getPerPrimitiveBufferSize()
         * - primitive_IDs: size = getPerPrimitiveBufferSize()  ← NOT Nobjects!
         * - twosided_flags, solid_fractions, object_IDs: size = getPerPrimitiveBufferSize()
         *
         * Historical bug (commit 53ca9687d): primitive_IDs sized by Nobjects instead of Nprimitives
         */
        size_t getPerPrimitiveBufferSize() const {
            return primitive_count;
        }

        /**
         * @brief Get expected buffer size for UUID→position lookup table
         * @return max(primitive_UUIDs) + 1
         *
         * The lookup table is a SPARSE array indexed by UUID value.
         * Must be sized to accommodate the highest UUID, not the primitive count!
         *
         * Example: If UUIDs are [10, 42, 100], size must be 101, not 3.
         * Entries for non-existent UUIDs contain UINT_MAX.
         */
        size_t getUUIDLookupBufferSize() const {
            if (primitive_UUIDs.empty())
                return 0;
            return *std::max_element(primitive_UUIDs.begin(), primitive_UUIDs.end()) + 1;
        }

        /**
         * @brief Validate geometry buffer consistency (debug builds only)
         *
         * Comprehensive validation of all indexing invariants.
         * Catches buffer sizing errors BEFORE GPU upload, preventing crashes and silent corruption.
         *
         * Validates:
         * - All per-primitive buffers have size == primitive_count
         * - All per-type buffers match their type counts
         * - UUID→position mapping is bidirectionally consistent
         * - primitive_positions table is correctly sized (max_UUID + 1)
         * - Object subdivisions are valid (>= 1)
         *
         * Call this at the START of every backend's updateGeometry() method.
         *
         * @throws helios_runtime_error with detailed message if validation fails
         * @note Compiled out in release builds (#ifndef NDEBUG) - zero production cost
         *
         * Historical bugs this prevents:
         * - Buffer sized by Nobjects instead of Nprimitives (commit 53ca9687d)
         * - Wrong primitive_positions size
         * - Inconsistent type counts
         */
        void validate() const;
    };

    /**
     * @brief Material properties for ray tracing
     *
     * Indexing: materials are indexed as [source * Nbands * Nprims + band * Nprims + prim]
     * For camera materials: [camera * Nbands * Nprims + band * Nprims + prim]
     */
    struct RayTracingMaterial {
        std::vector<float> reflectivity; //!< Reflectivity per [source][band][primitive]
        std::vector<float> transmissivity; //!< Transmissivity per [source][band][primitive]
        std::vector<float> reflectivity_cam; //!< Camera-weighted reflectivity
        std::vector<float> transmissivity_cam; //!< Camera-weighted transmissivity
        std::vector<float> specular_exponent; //!< Specular reflection exponent per primitive
        std::vector<float> specular_scale; //!< Specular reflection scale coefficient per primitive

        size_t num_bands = 0; //!< Number of spectral bands
        size_t num_sources = 0; //!< Number of radiation sources
        size_t num_primitives = 0; //!< Number of primitives
        size_t num_cameras = 0; //!< Number of cameras
    };

    /**
     * @brief Radiation source definition
     */
    struct RayTracingSource {
        helios::vec3 position; //!< Position (for point sources) or direction (for collimated)
        helios::vec3 rotation; //!< Rotation angles (Euler angles in radians)
        helios::vec2 width; //!< Width/size (for rectangle/disk sources)
        uint type; //!< Type: 0=collimated, 1=sphere, 2=sun_sphere, 3=rectangle, 4=disk
        std::vector<float> fluxes; //!< Flux per band
        std::vector<float> fluxes_cam; //!< Camera-weighted flux per band
    };

    /**
     * @brief Launch parameters for ray tracing
     */
    struct RayTracingLaunchParams {
        // Launch dimensions
        uint launch_offset = 0; //!< Starting primitive index
        uint launch_count = 0; //!< Number of primitives to launch
        uint rays_per_primitive = 0; //!< Number of rays per primitive
        uint random_seed = 0; //!< Random seed for ray generation

        // Band configuration
        uint current_band = 0; //!< Current band index (for single-band launches)
        uint num_bands_global = 0; //!< Total number of bands in simulation
        uint num_bands_launch = 0; //!< Number of bands in this launch
        std::vector<bool> band_launch_flag; //!< Flag indicating which bands to launch

        // Scattering
        uint scattering_iteration = 0; //!< Current scattering iteration (0 = first bounce)
        uint max_scatters = 0; //!< Maximum number of scattering events

        // Face selection (for diffuse rays)
        uint launch_face = 0; //!< Face to launch from: 0=top, 1=bottom

        // Camera-specific parameters
        uint camera_id = 0; //!< Camera ID
        helios::vec3 camera_position; //!< Camera position
        helios::vec2 camera_direction; //!< Camera direction (spherical: elevation, azimuth)
        float camera_focal_length = 0.0f; //!< Camera focal length (working distance)
        float camera_lens_diameter = 0.0f; //!< Camera lens diameter (0 = pinhole)
        float camera_fov_aspect = 0.0f; //!< Field of view aspect ratio (HFOV/VFOV)
        helios::int2 camera_resolution; //!< Camera resolution (width, height)
        float camera_HFOV = 0.0f; //!< Horizontal field of view (radians)
        uint antialiasing_samples = 1; //!< Antialiasing samples per pixel
        helios::int2 camera_pixel_offset; //!< Pixel offset for tiled rendering
        float camera_viewplane_length = 0.0f; //!< Viewplane length computed from effective_HFOV/zoom
        float camera_pixel_solid_angle = 0.0f; //!< Pixel solid angle in steradians
        helios::int2 camera_resolution_full; //!< Full resolution for tiled rendering (global coordinates)

        // Diffuse radiation parameters (for diffuse ray launches)
        std::vector<float> diffuse_flux; //!< Diffuse flux per band
        std::vector<float> diffuse_extinction; //!< Diffuse extinction coefficient per band
        std::vector<helios::vec3> diffuse_peak_dir; //!< Peak direction for diffuse distribution per band
        std::vector<float> diffuse_dist_norm; //!< Diffuse distribution normalization per band
        std::vector<helios::vec4> sky_radiance_params; //!< Prague sky model params per band (circ_str, circ_width, horiz_bright, norm)

        // Emission/outgoing radiation (for diffuse ray launches)
        std::vector<float> radiation_out_top; //!< Emitted/outgoing radiation (top face) per [primitive][band]
        std::vector<float> radiation_out_bottom; //!< Emitted/outgoing radiation (bottom face) per [primitive][band]

        // Flags
        uint specular_reflection_enabled = 0; //!< Specular reflection mode: 0=disabled, 1=default 0.25 scale, 2=user-defined scale
    };

    /**
     * @brief Output data from ray tracing
     *
     * Indexing: results are indexed as [primitive * Nbands + band]
     * For specular: [source * Ncameras * Nprims * Nbands + camera * Nprims * Nbands + prim * Nbands + band]
     */
    struct RayTracingResults {
        std::vector<float> radiation_in; //!< Absorbed radiation per [primitive][band]
        std::vector<float> radiation_out_top; //!< Emitted radiation (top face) per [primitive][band]
        std::vector<float> radiation_out_bottom; //!< Emitted radiation (bottom face) per [primitive][band]
        std::vector<float> scatter_buff_top; //!< Scattered radiation buffer (top) for multi-bounce
        std::vector<float> scatter_buff_bottom; //!< Scattered radiation buffer (bottom) for multi-bounce
        std::vector<float> scatter_buff_top_cam; //!< Camera-weighted scattered radiation buffer (top) per [primitive][band]
        std::vector<float> scatter_buff_bottom_cam; //!< Camera-weighted scattered radiation buffer (bottom) per [primitive][band]
        std::vector<float> radiation_specular; //!< Specular reflection per [source][camera][primitive][band]
        std::vector<float> sky_energy; //!< Energy absorbed by sky per band

        size_t num_primitives = 0; //!< Number of primitives
        size_t num_bands = 0; //!< Number of bands
        size_t num_sources = 0; //!< Number of sources
        size_t num_cameras = 0; //!< Number of cameras
    };

} // namespace helios

#endif // RAYTRACING_TYPES_H
