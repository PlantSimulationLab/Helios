/** \file "RayTracingTypes.h" Backend-agnostic data structures for ray tracing.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

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
     * All geometry data for the primitive types: patch, triangle, disk, tile, bbox (voxels are not traced).
     * Data is organized per-primitive and per-type.
     */
    struct RayTracingGeometry {

        // Per-primitive data (indexed by primitive index 0..primitive_count-1)
        std::vector<float> transform_matrices; //!< Transform matrices (16 floats per primitive, row-major)
        std::vector<uint> primitive_types; //!< Primitive type: 0=patch, 1=triangle, 2=disk, 3=tile, 5=bbox (4 is unused)
        std::vector<uint> primitive_UUIDs; //!< UUID for each primitive (for result mapping)
        std::vector<uint> primitive_IDs; //!< Primitive ID within parent object (for sub-patch calculations)
        std::vector<uint> object_IDs; //!< Parent object ID for each primitive
        std::vector<helios::int2> object_subdivisions; //!< Object subdivisions (for mapping)
        std::vector<char> twosided_flags; //!< Two-sided flag (1=two-sided, 0=one-sided)
        std::vector<float> solid_fractions; //!< Solid fraction (0-1, for porous surfaces)
        std::vector<float> primitive_areas; //!< One-sided surface area of each primitive (host side only: totals the power launched in scattering iterations)

        // Camera flux smoothing topology (empty unless smoothing is enabled)
        //!< Shared vertex index for each corner of each primitive, four entries per primitive.
        //!< Entry 4*p is negative for a primitive that takes no part in smoothing, and entry 4*p+3 is negative for a triangle, which has only three corners.
        std::vector<int> smoothing_vertex_indices;
        //!< Number of distinct shared vertices the indices above refer to, across every participating object.
        size_t smoothing_vertex_count = 0;

        // UUID-to-position lookup table (indexed by UUID value)
        //!< Maps UUID to array position: primitive_positions[UUID] = array_index
        //!< Size: max_UUID + 1 (sparse array, entries for non-existent UUIDs = UINT_MAX)
        //!< Enables CUDA to convert UUID to array position for buffer indexing
        std::vector<uint> primitive_positions;

        // Per-type geometry data
        PrimitiveTypeGeometry patches; //!< Patch data (4 vertices per patch)
        PrimitiveTypeGeometry triangles; //!< Triangle data (3 vertices per triangle)
        PrimitiveTypeGeometry tiles; //!< Tile data (4 vertices per tile)
        PrimitiveTypeGeometry bboxes; //!< Bounding box data (4 vertices per bbox face)

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

        // Spectral texture class maps (Context::setPrimitiveSpectralTexture()), laid over a primitive with the same (u,v) mapping and
        // texel convention as its mask. A ray leaving a primitive with a class map takes the reflectivity and transmissivity of the
        // class at its launch point from the palette tables in RayTracingMaterial.
        std::vector<uint16_t> class_map_data; //!< Flattened class maps [class map][row from the top][column]
        std::vector<helios::int2> class_map_sizes; //!< Size (width, height) of each class map
        std::vector<int> class_map_IDs; //!< Class map of each primitive (-1 if none)

        // Normal maps (Context::setMaterialNormalMap() and setMaterialBumpMap()), laid over a primitive with the same (u,v) mapping and texel convention as its
        // mask, already decoded to unit tangent-space normals with their strength applied (see include/NormalMapMath.h for how they are used).
        std::vector<uint16_t> normal_map_texels; //!< Flattened normals [map][row from the top][column], each uint8(x) | uint8(y) << 8 with x and y signed 8-bit normalized
        std::vector<helios::int2> normal_map_sizes; //!< Size (width, height) of each normal map
        std::vector<int> normal_map_IDs; //!< Normal map of each primitive (-1 if none); empty when no primitive has one
        //! Slot of each normal-mapped primitive in the record of the direction of the diffuse radiation it gathers (UINT32_MAX for none); empty when no primitive has a map.
        //! The diffuse launches accumulate, per slot, face (top, then bottom) and launch band, the flux-weighted sum of the directions the radiation came from (3 values),
        //! with which cameras shade a mapped primitive's scattered light per texel (see RayTracingBackend::zeroRadiationBuffers()).
        std::vector<uint32_t> normal_map_slots;
        size_t normal_map_slot_count = 0; //!< Number of slots in normal_map_slots

        // Periodic boundary condition
        helios::vec2 periodic_flag; //!< Periodic boundary flags (x, y directions)
        //! Per-primitive flag (indexed by position): 1 if the primitive is excluded from the periodic boundary condition and exists only in the central copy of the domain; empty when none is
        std::vector<uint8_t> periodic_excluded;
        //! Number of periodic copies of the domain on each side of the central one in x and y; -1 for no limit. A ray that leaves the copies continues as one that leaves the scene.
        helios::int2 periodic_max_repeats = helios::make_int2(-1, -1);
        //! z-range (min, max) of the geometry that repeats. A ray outside the central copy that is above it and heading up, or below it and heading down, can hit nothing more.
        helios::vec2 periodic_repeated_zbounds = helios::make_vec2(0.f, 0.f);
        //! Distance beyond a periodic boundary face from which a ray that leaves the periodic copies continues, far enough that it cannot hit a boundary face again
        float periodic_escape_offset = 0.f;

        //! Whether rays must track which periodic copy of the domain they are in (some primitives are excluded from the periodic boundary, or the number of copies is limited)
        [[nodiscard]] bool periodicTileTracking() const {
            return (periodic_flag.x == 1 || periodic_flag.y == 1) && (!periodic_excluded.empty() || periodic_max_repeats.x >= 0 || periodic_max_repeats.y >= 0);
        }

        //! Whether the radiation is solved for two fields at once: the scene as it is, and the "background" scene without the primitives excluded from the periodic boundary
        /**
         * Periodic copies of a primitive are the same primitive, so they all show its one outgoing radiation, which near an excluded primitive carries that primitive's shade and
         * scattered light. The background field is what the repeated geometry sends out where no excluded primitive stands; rays and cameras read it for every copy of the domain
         * but the central one. See fluxSlotCount().
         */
        [[nodiscard]] bool periodicDualField() const {
            return (periodic_flag.x == 1 || periodic_flag.y == 1) && !periodic_excluded.empty();
        }

        //! Number of per-primitive slots in the buffers of outgoing radiation and of what cameras read
        /**
         * The primitive count, or twice it with a background field (see periodicDualField()): slot p holds primitive p in the scene as it is, and slot primitive_count + p
         * holds it in the background field (zero for an excluded primitive, which the background does not contain). Buffers of absorbed radiation have no background slots.
         */
        [[nodiscard]] size_t fluxSlotCount() const {
            return periodicDualField() ? 2 * primitive_count : primitive_count;
        }

        // Counts
        size_t primitive_count = 0; //!< Total number of primitives (all types)
        size_t patch_count = 0; //!< Number of patches
        size_t triangle_count = 0; //!< Number of triangles
        size_t tile_count = 0; //!< Number of tiles
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
         */
        void validate() const;
    };

    /**
     * @brief Material properties for ray tracing
     *
     * Indexing: materials are indexed as [slot * Nprims * Nbands + prim * Nbands + band]
     * (i.e. MaterialPropertyIndexer(Nsources + 1, Nprims, Nbands)(slot, prim, band); band is the
     * fastest-varying dimension). There are Nsources + 1 slots: slot s < Nsources holds the properties
     * weighted by source s's spectrum, read by that source's direct rays, and slot Nsources holds the
     * properties weighted by the band's combined incident spectrum, read by diffuse, scattered and
     * emitted rays.
     * For camera materials: CameraMaterialIndexer(Nsources + 1, Nprims, Nbands, Ncameras).
     */
    struct RayTracingMaterial {
        std::vector<float> reflectivity; //!< Reflectivity per [slot][primitive][band]
        std::vector<float> transmissivity; //!< Transmissivity per [slot][primitive][band]
        std::vector<float> reflectivity_cam; //!< Camera-weighted reflectivity
        std::vector<float> transmissivity_cam; //!< Camera-weighted transmissivity
        //! Camera-weighted reflectivity of a spectrally flat, perfectly white surface, per [slot][band][camera] (band is the global band index)
        /**
         * This is the value reflectivity_cam takes for a reflectivity spectrum equal to 1 at every wavelength. The direct and diffuse launches weight the light arriving at each primitive by it to
         * accumulate the white reference that a camera's "auto" white balance is computed from.
         */
        std::vector<float> white_reference_cam;
        //! Whether any primitive has a surface layer, whose reflection lobe of the direct radiation needs the direct flux reaching each primitive (radiation_specular, allocated only then)
        bool specular_reflection_enabled = false;

        // Translucent cover (glass/plastic) material. When is_glass != 0 for a [slot][primitive][band]
        // entry, the angular transmittance/reflectance is computed on-device from glass_n and glass_KL
        // via the Fresnel+Bouguer model, overriding reflectivity/transmissivity for that entry.
        std::vector<float> glass_n; //!< Refractive index per [slot][primitive][band] (0 = not glass)
        std::vector<float> glass_KL; //!< Absorption product K*L per [slot][primitive][band] (0 = lossless)
        std::vector<char> is_glass; //!< Glass-material flag per [slot][primitive][band] (0 = off, 1 = on)

        // Specular surface layer (primitive or material data "specular_roughness" and "specular_n"). A primitive with a surface
        // layer reflects a(mu) of the radiation reaching it at its surface, where a(mu) is the directional albedo of a Fresnel +
        // GGX microfacet layer, and its body reflects, transmits and emits the rest diffusely. Its reflectivity and
        // transmissivity above (and the camera-weighted ones) then hold those of the body, rho_b and tau_b, so that the total
        // reflectance is R(mu) = a(mu) + rho_b (1 - a(mu)), the transmittance T(mu) = tau_b (1 - a(mu)) and the emissivity
        // eps(mu) = (1 - rho_b - tau_b)(1 - a(mu)). Each face sends out two components: one with angular shape 1 - a(mu)
        // (body reflection, transmission and emission) and one with shape a(mu) (the surface layer's reflection).
        std::vector<uint32_t> surface_model; //!< Surface model per primitive: 0 = Lambertian (no surface layer), m > 0 = model m - 1
        //! Albedo table of surface model m in global band b, [m * num_bands + b], or UINT32_MAX where that model has no layer in the band
        std::vector<uint32_t> surface_model_band_table;
        //! Directional albedo tables: each holds surface_layer_table_size values of a(mu) at mu = i / (surface_layer_table_size - 1), followed by its hemispherical mean abar = 2 int a(mu) mu dmu, the layer's refractive index and its roughness
        std::vector<float> surface_layer_tables;
        uint32_t surface_layer_table_size = 65; //!< Number of mu values in each albedo table
        //! Whether any primitive has a surface layer in some band. The outgoing and scattered radiation buffers then hold a second component (see RayTracingResults).
        bool surface_layer_enabled = false;

        // Spectral texture palettes. The palettes of every primitive with a class map are laid end to end; class k of a primitive's
        // class map is entry palette_offset[primitive] + k - 1. A ray leaving such a primitive from a texel of class k > 0 uses that
        // entry instead of the primitive's own reflectivity and transmissivity, which hold its area-weighted mean.
        std::vector<float> palette_reflectivity; //!< Reflectivity per [slot][entry][band]
        std::vector<float> palette_transmissivity; //!< Transmissivity per [slot][entry][band]
        std::vector<float> palette_reflectivity_cam; //!< Camera-weighted reflectivity per [slot][entry][band][camera]
        std::vector<float> palette_transmissivity_cam; //!< Camera-weighted transmissivity per [slot][entry][band][camera]
        std::vector<uint32_t> palette_offset; //!< First palette entry of each primitive (UINT32_MAX if it has no class map)
        //! Per global band: 1 where rays use the palette, 0 where they use the primitive's own values (bands with emission, whose emissivity is per primitive)
        std::vector<uint32_t> palette_band_active;
        size_t num_palette_entries = 0; //!< Number of palette entries

        size_t num_bands = 0; //!< Number of spectral bands
        size_t num_sources = 0; //!< Number of radiation sources (the material arrays have one more slot than this)
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
        std::vector<float> fluxes_cam; //!< Camera weights of the source flux [launched band][camera], set before each direct launch when the model has cameras (empty otherwise)
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
        //! Accumulate the forward scattering estimate of this launch (see RayTracingBackend::getForwardScatterResults()); set for scattering launches only
        bool forward_scatter_enabled = false;

        // Face selection (for diffuse rays)
        uint launch_face = 0; //!< Face to launch from: 1 = top (the side the normal points to), 0 = bottom

        // Camera-specific parameters
        uint camera_id = 0; //!< Camera ID
        helios::vec3 camera_position; //!< Camera position
        helios::vec2 camera_direction; //!< Camera direction (spherical: elevation, azimuth)
        float camera_focal_length = 0.0f; //!< Camera focal length (working distance)
        float camera_lens_diameter = 0.0f; //!< Camera lens diameter (0 = pinhole)
        float camera_fov_aspect = 0.0f; //!< Field of view aspect ratio (HFOV/VFOV)
        helios::int2 camera_resolution; //!< Camera resolution (width, height)
        uint antialiasing_samples = 1; //!< Antialiasing samples per pixel
        helios::int2 camera_pixel_offset; //!< Pixel offset for tiled rendering
        float camera_viewplane_length = 0.0f; //!< Viewplane length computed from effective_HFOV/zoom
        helios::int2 camera_resolution_full; //!< Full resolution for tiled rendering (global coordinates)
        //! Periodic copy of the domain the camera is in (x, y): the whole periods its position was moved by to bring it into the domain (see RayTracingGeometry::periodicTileTracking())
        helios::int2 camera_tile = helios::make_int2(0, 0);

        // Diffuse radiation parameters (for diffuse ray launches)
        //! Diffuse sky flux per launched band, followed by the flux of each launched band's reference blackbody field (2 x Nbands_launch values)
        /**
         * A diffuse ray that leaves the scene receives the sky flux less the reference flux: emission is gathered as its
         * departure from a uniform blackbody field (see RadiationModel::runBand()). A band without one has a reference flux of 0.
         */
        std::vector<float> diffuse_flux;
        std::vector<float> diffuse_extinction; //!< Diffuse extinction coefficient per band
        std::vector<helios::vec3> diffuse_peak_dir; //!< Peak direction for diffuse distribution per band
        std::vector<float> diffuse_dist_norm; //!< Diffuse distribution normalization per band
        std::vector<helios::vec4> sky_radiance_params; //!< Prague sky model params per band (circ_str, circ_width, horiz_bright, norm)

        // Emission/outgoing radiation (for diffuse ray launches)
        std::vector<float> radiation_out_top; //!< Emitted/outgoing radiation (top face) per [primitive][band]
        std::vector<float> radiation_out_bottom; //!< Emitted/outgoing radiation (bottom face) per [primitive][band]

        // Outgoing radiation resampled onto the shared mesh vertices, for camera flux smoothing (empty unless smoothing is enabled)
        std::vector<float> vertex_radiation_out_top; //!< Area-weighted mean outgoing radiation (top face) per [component][vertex][band]
        std::vector<float> vertex_radiation_out_bottom; //!< Area-weighted mean outgoing radiation (bottom face) per [component][vertex][band]

        // Flags
        uint specular_reflection_enabled =
                0; //!< 1 = this launch evaluates the surface layers' reflection lobe of the direct radiation (the launch reading its first scattering, and camera launches); for a direct launch, that surface layers are present
        uint below_horizon_diffuse_enabled = 0; //!< 1 = diffuse sky radiation also reaches rays that leave the domain below the horizon (diffuse and camera rays)
        //! Camera launches: 1 if the outgoing radiation uploaded for normal-mapped primitives leaves out what the direct launch deposited, which the camera then shades per hit point with the shading normal
        uint normal_map_direct = 0;
        //! Camera launches: 1 if the camera evaluates the sources' direct radiation on smoothed primitives per pixel sample, which the host left out of what their vertices carry (camera flux smoothing, a
        //! radiation source, and a direct launch that deposited the direct radiation the host removed)
        uint camera_direct_per_pixel = 0;
        //! Camera launches: 1 if the camera evaluates the direct radiation of every primitive, not only smoothed ones, per pixel sample (primitives are excluded from the periodic boundary), the host having
        //! left it out of the outgoing radiation of every primitive but a translucent cover
        uint camera_direct_all = 0;
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
        std::vector<float> scatter_buff_top_cam; //!< Camera-weighted scattered radiation buffer (top) per [camera][primitive][band]
        std::vector<float> scatter_buff_bottom_cam; //!< Camera-weighted scattered radiation buffer (bottom) per [camera][primitive][band]
        std::vector<float> radiation_specular; //!< Direct flux reaching primitives with a surface layer, per [source][primitive][band]
        //! Radiation reaching each face of each primitive with a spectral texture, per [face (top, then bottom)][source slot][primitive][launch band], from which cameras
        //! compute the outgoing radiation of each class. Empty unless there are cameras and some primitive has a spectral texture.
        std::vector<float> spectral_incident;

        size_t num_primitives = 0; //!< Number of primitives
        size_t num_bands = 0; //!< Number of bands
        size_t num_sources = 0; //!< Number of sources
        size_t num_cameras = 0; //!< Number of cameras
    };

    //! Directional albedo of a specular surface layer: a dielectric interface of refractive index n with GGX microfacet roughness alpha
    /**
     * a(mu) is the fraction of the radiation arriving at the angle whose cosine is mu that the layer reflects (Fresnel
     * reflectance of the microfacets with the height-correlated Smith masking-shadowing term), integrated over the reflected
     * hemisphere.
     * \param[in] refractive_index Refractive index of the layer (>= 1)
     * \param[in] roughness GGX roughness alpha (> 0)
     * \param[in] table_size Number of mu values, equally spaced from 0 to 1 (>= 2)
     * \return table_size values of a(mu), followed by the hemispherical mean abar = 2 int_0^1 a(mu) mu dmu
     */
    std::vector<float> computeSurfaceLayerAlbedoTable(float refractive_index, float roughness, uint32_t table_size);

    //! The class maps of a geometry packed into one array of words, for backends that read them from a single buffer
    /**
     * The layout is described in shaders/common/spectral_texture.glsl (used by the Vulkan and OptiX 6 backends): the number of
     * primitives (0 when none has a class map) and of class maps, the class map of each primitive (0xFFFFFFFF for none), the width,
     * height and first texel of each class map, then two 16-bit classes per word.
     * \param[in] geometry Geometry whose class_map_data, class_map_sizes and class_map_IDs are packed
     */
    std::vector<uint32_t> packSpectralClassMaps(const RayTracingGeometry &geometry);

    //! The normal maps of a geometry packed into one array of words, the layout every backend reads (described in include/NormalMapMath.h)
    /**
     * Word 0 is 0 when no primitive has a normal map, and the array then holds only the four header words.
     * \param[in] geometry Geometry whose normal_map_* fields are packed
     */
    std::vector<uint32_t> packNormalMaps(const RayTracingGeometry &geometry);

    //! The spectral texture palettes of a material packed into one array of words, for backends that read them from a single buffer
    /**
     * The layout is described in shaders/common/spectral_texture.glsl: a header of 9 words (the number of palette entries, 0 when no
     * primitive has a spectral texture; the global band count; the camera count; and the starts of the per-primitive offsets, of the
     * per-band flags and of the four tables), then those arrays, the tables as float bits.
     * \param[in] materials Material whose palette_* fields are packed
     */
    std::vector<uint32_t> packSpectralPalettes(const RayTracingMaterial &materials);

    //! The camera-weighted outgoing radiation of the classes of spectral textures packed into one array of words, for backends that read it from a single buffer
    /**
     * The layout is described in shaders/common/spectral_texture.glsl: the number of class-flux slots (0 when there are none), the launch band count and the
     * number of primitives; the first slot of each primitive; then the values per [face][slot][band] as float bits.
     * \param[in] class_flux_offset First class-flux slot of each primitive (see RayTracingBackend::uploadSpectralCameraFlux()), or empty for none
     * \param[in] class_flux Values per [face][slot][launch band]
     * \param[in] launch_band_count Number of launched bands
     */
    std::vector<uint32_t> packSpectralCameraFlux(const std::vector<uint32_t> &class_flux_offset, const std::vector<float> &class_flux, size_t launch_band_count);

    //! Pack the vertex values of RayTracingBackend::uploadSpectralVertexIncident() for the Vulkan and OptiX 6 devices
    /**
     * Layout: the layer count L (2, or 0 for none), the launch band count B, the smoothing vertex count V and the camera index; then the values per
     * [face][layer][vertex][band] as float bits.
     */
    std::vector<uint32_t> packSpectralVertexIncident(const std::vector<float> &vertex_values, uint32_t layer_count, size_t launch_band_count, uint32_t camera);

} // namespace helios

#endif // RAYTRACING_TYPES_H
