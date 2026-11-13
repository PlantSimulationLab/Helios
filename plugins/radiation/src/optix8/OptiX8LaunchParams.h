/** \file "OptiX8LaunchParams.h" Shared host/device launch parameters struct for OptiX 8.1 backend.

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

#ifndef OPTIX8_LAUNCH_PARAMS_H
#define OPTIX8_LAUNCH_PARAMS_H

// This header is included by both host (.cpp) and device (.cu) code.
// Only use types compatible with both CUDA and C++.
#include <stdint.h>

// CUDA vector types (float3, float2, int2, float4, ...) - available in both host and device contexts
#include <vector_types.h>

// OptiX traversable handle type (host and device)
#include <optix.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

typedef unsigned int uint;

// Maximum number of radiation bands that may be launched together. Used to size the
// per-band translucent-cover transmittance accumulator in the payload (the payload lives in a
// per-thread pool, not registers, so this array adds no register pressure). The host validates
// that the launch band count does not exceed this cap (fail-fast) before launching.
#ifndef HELIOS_MAX_RADIATION_BANDS
#define HELIOS_MAX_RADIATION_BANDS 32
#endif

// SBT layout. Hit groups: 0 direct, 1 diffuse, 2 camera, 3 pixel label, then the direct and diffuse hit groups of a scene with a
// background radiation field (RayTracingGeometry::periodicDualField()) at DUAL_FIELD_SBT_OFFSET + 0 and + 1. Miss programs: 0 direct,
// 1 diffuse, 2 camera, 3 pixel label, 4 camera shadow, then the direct and diffuse miss programs of such a scene at
// DUAL_FIELD_SBT_OFFSET + 1 + 0 and + 1 + 1.
#define DUAL_FIELD_SBT_OFFSET 4u

// ---------------------------------------------------------------------------
// Per-ray payload.  Passed by pointer-in-two-registers.
// Allocated in a per-thread pool on the device.
// ---------------------------------------------------------------------------
struct PerRayData {
    static constexpr bool is_dual = false; //!< Whether this is the payload of a scene with a background radiation field (PerRayDataDual)
    double strength;           //!< Ray energy
    uint   origin_UUID;        //!< UUID of origin primitive (0 = launched from source/camera)
    bool   face;               //!< Top face = true, bottom face = false
    uint   seed;               //!< LCG random seed
    unsigned char source_ID;   //!< Source index, or Nsources (the diffuse/scatter material slot) for diffuse rays (max 255 sources)
    bool   skip_origin;        //!< The ray never hits the primitive origin_UUID in the periodic copy it left from (direct and diffuse rays; camera and pixel-label rays carry a pixel index there)
    bool   hit_periodic_boundary;
    float3 periodic_hit;       //!< World-space hit on the periodic wall
    float  periodic_hit_t;     //!< Distance from the segment origin to the periodic wall it hit
    int    tile_x;             //!< Periodic copy of the domain the ray is in, x (0 = central; counted whenever a boundary is periodic)
    int    tile_y;             //!< Periodic copy of the domain the ray is in, y
    int    origin_tile_x;      //!< Periodic copy of the domain the origin primitive the ray skips is in (skip_origin), x
    int    origin_tile_y;      //!< Periodic copy of the domain the origin primitive the ray skips is in, y
    //! Per-launch-band transmittance accumulated as the ray passes through translucent covers
    //! (glass/plastic). Initialised to 1 in the raygen programs; multiplied by the Fresnel+Bouguer
    //! tau(theta) in the any-hit program (direct) or the closest-hit programs (diffuse, camera);
    //! applied to what the ray delivers past the covers. Indexed by launch-band b.
    float  cover_transmittance[HELIOS_MAX_RADIATION_BANDS];
    bool   hit_cover;          //!< Diffuse/camera/pixel-label ray hit a translucent cover and continues past it from cover_exit
    float3 cover_exit;         //!< World-space point just beyond the cover where the ray continues
    float  origin_cos;         //!< |cosine| of the angle between a direct or diffuse ray and the normal of the primitive it left
    uint32_t palette_entry;    //!< Palette entry of the class at the point the ray left, or UINT32_MAX to use the origin primitive's own values
    bool   source_reached;     //!< Set when a camera's shadow ray reaches the source it was traced toward (__miss__camera_shadow)
};

// ---------------------------------------------------------------------------
// Payload of the direct and diffuse rays of a scene with a background radiation field (RayTracingGeometry::periodicDualField()).
// Those rays are launched by the *_dual raygen programs and handled by the *_dual hit and miss programs, which alone know of
// the members added here; every other program sees a PerRayData. A scene without a background field never uses it, so its
// rays carry nothing for it.
// ---------------------------------------------------------------------------
struct PerRayDataDual : PerRayData {
    static constexpr bool is_dual = true;
    //! The ray books what it delivers in both fields (its primitive is not excluded from the periodic boundary). Such a ray passes the excluded primitives, which the
    //! background field does not contain, and records the nearest one it would have hit in each segment (the any-hit programs).
    bool   dual_ray;
    //! A dual ray's delivery to the scene as it is (field 0) is settled by an excluded primitive: a direct ray is blocked by it, and a diffuse ray has received its radiation
    bool   excluded_resolved;
    bool   recorded_face;      //!< Whether the top face of the recorded primitive was hit
    float  recorded_t;         //!< Distance to the nearest excluded primitive passed in the current segment (infinite if none)
    uint   recorded_position;  //!< Its position
    float  recorded_u;         //!< Its hit attributes 2 and 3 (see __intersection__patch)
    float  recorded_v;
};

// ---------------------------------------------------------------------------
// SBT hit-group record data (per primitive type).
// ---------------------------------------------------------------------------
struct HitGroupData {
    float3*   vertices; //!< Device pointer to type-specific vertex array (layout varies by type)
    uint32_t* UUIDs;    //!< Device pointer to type-specific UUID array
    uint32_t  prim_type; //!< 0=patch, 1=triangle, 2=disk, 3=tile, 5=bbox
};

// ---------------------------------------------------------------------------
// Main launch parameters structure.
// Passed from host to device via optixLaunch(d_params, ...).
// Accessible on device as: extern "C" __constant__ OptiX8LaunchParams params;
// ---------------------------------------------------------------------------
struct OptiX8LaunchParams {
    // Traversal handle
    OptixTraversableHandle traversable;

    // ---- Per-ray-data pool (size = launch_dim_x * launch_dim_y) ----
    PerRayData* prd_pool;

    // ---- Geometry buffers (device pointers) ----
    float*    transform_matrix;      //!< [prim * 16 + elem], row-major 4x4
    uint32_t* primitive_type;        //!< 0=patch,1=tri,2=disk,3=tile,5=bbox
    uint32_t* primitive_positions;   //!< UUID→array-index sparse lookup (size=max_UUID+1)
    uint32_t* primitiveID;           //!< First primitive index in parent object
    uint32_t* objectID;              //!< Parent object ID for each primitive
    int32_t*  object_subdivisions;   //!< 2 ints per prim: [subdiv_x, subdiv_y]
    int8_t*   twosided_flag;         //!< 1=two-sided, 0=one-sided
    float*    primitive_solid_fraction;

    // ---- Global UUID lookup (indexed by global primitive position) ----
    uint32_t* primitive_uuid;        //!< primitive_uuid[global_pos] = UUID of that primitive

    // ---- Per-type geometry (also in SBT HitGroupData, mirrored here for easy access) ----
    float3*   patch_vertices;        //!< [patch_idx * 4 + corner_idx]
    uint32_t* patch_UUIDs;
    float3*   triangle_vertices;     //!< [tri_idx * 3 + vertex_idx]
    uint32_t* triangle_UUIDs;
    float3*   disk_centers;
    float*    disk_radii;
    float3*   disk_normals;
    uint32_t* disk_UUIDs;
    float3*   tile_vertices;         //!< [tile_idx * 4 + corner_idx]
    uint32_t* tile_UUIDs;
    float3*   bbox_vertices;         //!< [bbox_idx * 4 + corner_idx] (4 vertices per planar face)
    uint8_t*  periodic_excluded;     //!< [prim]: 1 if the primitive exists only in the central periodic copy of the domain; null when none is excluded
    uint32_t* bbox_UUIDs;

    // ---- Material buffers ----
    float*    rho;        //!< Reflectivity [source * Nprims * Nbands_global + prim * Nbands_global + band]
    float*    tau;        //!< Transmissivity (same layout)
    float*    rho_cam;    //!< Camera-weighted reflectivity
    float*    tau_cam;    //!< Camera-weighted transmissivity
    float*    white_reference_cam; //!< Camera-weighted reflectivity of a white surface [source * Nbands_global * Ncameras + band_global * Ncameras + camera]

    // ---- Translucent cover (glass/plastic) material buffers (same layout as rho/tau) ----
    float*    glass_n;    //!< Refractive index [source * Nprims * Nbands_global + prim * Nbands_global + band]
    float*    glass_KL;   //!< Bouguer absorption product K*L (same layout)
    int8_t*   is_glass;   //!< 1 if this [source][prim][band] uses the glass model, else 0 (same layout)

    // ---- Specular surface layer (see RayTracingMaterial::surface_model); null when no primitive has one ----
    uint32_t* surface_model;            //!< [prim]: 0 = Lambertian, m > 0 = surface model m - 1
    uint32_t* surface_model_band_table; //!< [(m - 1) * Nbands_global + band]: albedo table, or UINT32_MAX for no layer in the band
    float*    surface_layer_tables;     //!< [table * (surface_layer_table_size + 3) + i]: a(mu) at mu = i / (size - 1), then abar, n and alpha

    // ---- Radiation energy buffers ----
    float*    radiation_in;           //!< [prim * Nbands_global + band]
    float*    radiation_in_top;       //!< Absorbed radiation that arrived on the top face [prim * Nbands_launch + band]; null unless face absorption tracking is enabled
    float*    radiation_out_top;      //!< [prim * Nbands_global + band]
    float*    radiation_out_bottom;   //!< [prim * Nbands_global + band]
    int32_t*  smoothing_vertex_indices;  //!< Shared vertex index per facet corner, 4 per primitive; entry 4*prim negative if the primitive is not smoothed, entry 4*prim+3 negative for a triangle
    float*    vertex_radiation_out_top;    //!< [component][vertex][band], null when camera flux smoothing is disabled
    float*    vertex_radiation_out_bottom; //!< [component][vertex][band], null when camera flux smoothing is disabled
    float*    vertex_normals;              //!< Unit shading normal per smoothing vertex [vertex * 3]; null unless smoothing is enabled and there is a source or some primitive has a surface layer
    uint32_t  vertex_normal_count;         //!< Number of smoothing vertices in vertex_normals (and in each component of vertex_radiation_out_*), 0 when null
    float*    scatter_buff_top;       //!< [prim * Nbands_global + band]
    float*    scatter_buff_bottom;    //!< [prim * Nbands_global + band]
    float*    forward_scatter;        //!< Forward scattering estimate [prim * Nbands_launch + band]; null except in scattering launches
    float*    radiation_in_camera;    //!< [prim * Nbands_launch + band]
    float*    scatter_buff_top_cam;   //!< [cam * Nprims * Nbands_launch + prim * Nbands_launch + band]
    float*    scatter_buff_bottom_cam; //!< same layout as scatter_buff_top_cam
    float*    white_reference_top_cam;    //!< Each camera's white reference, accumulated over all the launches of a runBand(); same layout as scatter_buff_top_cam
    float*    white_reference_bottom_cam; //!< same layout as white_reference_top_cam
    float*    radiation_specular;     //!< Direct flux reaching primitives with a surface layer [src*Nprims*Nbands + prim*Nbands + band]

    // ---- Camera output buffers ----
    uint32_t* camera_pixel_label;
    float*    camera_pixel_depth;

    // ---- Ray counters ----
    uint32_t* abandoned_ray_count;    //!< [1]: direct and diffuse rays abandoned at the periodic wrap limit; null until zeroed by the host

    // ---- Source buffers ----
    float3*   source_positions;
    float3*   source_rotations;
    float2*   source_widths;
    uint32_t* source_types;
    float*    source_fluxes;          //!< [source * Nbands_launch + band]
    float*    source_fluxes_cam;      //!< [source * Nbands_launch * Ncameras + ...]

    // ---- Diffuse/sky buffers ----
    float*    diffuse_flux;           //!< [band]
    float*    diffuse_extinction;     //!< [band]
    float3*   diffuse_peak_dir;       //!< [band]
    float*    diffuse_dist_norm;      //!< [band]
    float4*   sky_radiance_params;    //!< [band]: (circumsolar_str, circumsolar_width, horiz_coeff, zenith_scale)
    float*    camera_sky_radiance;    //!< [band]
    float*    solar_disk_radiance;    //!< [band]
    float*    camera_diffuse_flux;    //!< [band]: hemispherical diffuse flux for camera longwave/emission sky miss
    uint32_t* band_emission_flag;     //!< [band]: 1 if band has emission enabled (gates camera longwave sky sampling)

    // ---- Band launch flags ----
    bool*     band_launch_flag;       //!< [Nbands_global]: which bands to process this launch

    // ---- Texture/mask buffers ----
    uint8_t*  mask_data;              //!< Flat mask bytes: 1=opaque, 0=transparent; layout [mask0_pixels][mask1_pixels]...
    uint32_t* mask_offsets;           //!< mask_offsets[k] = start byte index in mask_data for mask k
    int32_t*  mask_sizes;             //!< 2 ints per mask: mask_sizes[k*2]=width, mask_sizes[k*2+1]=height
    int32_t*  mask_IDs;               //!< Per-primitive mask ID (-1 = no mask), indexed by global pos
    float2*   uv_data;                //!< UV coords: uv_data[pos*4 + vertex_idx] (4 vertices per primitive)
    int32_t*  uv_IDs;                 //!< Per-primitive: -1 = use parametric UV, >=0 = has custom UV in uv_data

    // ---- Spectral texture class maps and palettes (see RayTracingGeometry::class_map_data and RayTracingMaterial::palette_offset) ----
    uint16_t* class_map_data;         //!< Flat class maps, rows from the top: [map0_texels][map1_texels]...
    uint32_t* class_map_offsets;      //!< class_map_offsets[k] = start index in class_map_data of class map k
    int32_t*  class_map_sizes;        //!< 2 ints per class map: width, height
    int32_t*  class_map_IDs;          //!< Per-primitive class map (-1 = none), indexed by global pos

    // ---- Normal maps (see include/NormalMapMath.h for the packed layout) ----
    const uint32_t* normal_maps;      //!< Packed normal maps, or nullptr when no primitive has one
    uint32_t  normal_map_direct;      //!< Camera launches: 1 if the host removed the direct launch's deposit from normal-mapped primitives' outgoing radiation, so the camera shades it per hit point
    uint32_t  camera_direct_per_pixel; //!< Camera launches: 1 if the camera evaluates the direct radiation of smoothed primitives per pixel sample (RayTracingLaunchParams::camera_direct_per_pixel)
    uint32_t  camera_direct_all;     //!< Camera launches: 1 if it does so for every primitive but a translucent cover (RayTracingLaunchParams::camera_direct_all)
    const uint32_t* normal_map_slots; //!< Slot of each normal-mapped primitive in normal_map_moments (UINT32_MAX for none), or nullptr when no primitive has a map
    float*    normal_map_moments;     //!< [slot][face][launch band][3]: flux-weighted sum of the directions the diffuse radiation reaching each face came from
    const float* vertex_normal_map_moments; //!< [smoothing vertex][face][launch band][3]: normal_map_moments averaged onto the smoothing vertices; null unless smoothing and maps
    uint32_t* palette_offset;         //!< Per-primitive first palette entry (UINT32_MAX = no spectral texture), indexed by global pos
    float*    palette_rho;            //!< Palette reflectivity [source * Npalette_entries * Nbands_global + entry * Nbands_global + band]
    float*    palette_tau;            //!< Palette transmissivity (same layout)
    float*    palette_rho_cam;        //!< Camera-weighted palette reflectivity [((source * Npalette_entries + entry) * Nbands_global + band) * Ncameras + camera]
    float*    palette_tau_cam;        //!< Camera-weighted palette transmissivity (same layout)
    uint32_t* palette_band_active;    //!< [Nbands_global]: 1 where rays use the palette, 0 where they use the primitive's own values
    uint32_t  Npalette_entries;       //!< Number of palette entries (0 when no primitive has a spectral texture)
    float*    spectral_incident;      //!< Radiation reaching primitives with a spectral texture [face (top, bottom)][source slot][prim][band_launch], or null without cameras
    uint32_t* class_flux_offset;      //!< Per-primitive first class-flux slot for the camera being traced (UINT32_MAX = none), or null
    float*    class_flux;             //!< Camera-weighted outgoing radiation [face][class-flux slot][band_launch] (see RayTracingBackend::uploadSpectralCameraFlux())
    uint32_t  class_flux_slots;       //!< Number of class-flux slots
    uint32_t* camera_pixel_class;     //!< Class of the spectral texture seen in each pixel by the pixel-label launch (0 = none), or null
    float*    vertex_spectral;        //!< Smoothing-vertex radiation of primitives with a spectral texture [face][layer][vertex][band] (see RayTracingBackend::uploadSpectralVertexIncident()), or null
    uint32_t  vertex_spectral_layers; //!< Number of layers (source slots plus the emitted radiation), 0 for none
    uint32_t  vertex_spectral_count;  //!< Number of smoothing vertices in vertex_spectral
    uint32_t  vertex_spectral_camera; //!< Camera index for the camera-weighted palette values

    // ---- Scalar parameters ----
    uint32_t  Nprimitives;
    uint32_t  Nflux_slots;           //!< Slots per primitive dimension of the outgoing, scattered, camera, direct-flux and spectral buffers: Nprimitives, or 2 Nprimitives with a background field (slot Nprimitives + p), see RayTracingGeometry::fluxSlotCount()
    uint32_t  vertex_field_stride;   //!< Number of smoothing vertices; the background field's vertex values are at vertex_field_stride + vertex (set with the vertex outgoing radiation)
    uint32_t  normal_map_slot_count; //!< Number of normal-map slots; the background field's moments are in slots normal_map_slot_count + slot
    uint32_t  Nsources;
    uint32_t  Ncameras;
    uint32_t  Nbands_global;
    uint32_t  Nbands_launch;
    uint32_t  launch_offset;         //!< Starting primitive index for this launch
    uint32_t  launch_count;          //!< Primitives to process in this launch
    uint32_t  rays_per_primitive;
    uint32_t  random_seed;
    uint32_t  launch_face;           //!< 1=top face, 0=bottom face (diffuse)
    uint32_t  scattering_iteration;
    uint32_t  specular_reflection_enabled; //!< 1 = this launch reads the surface layers' reflection lobe of the direct radiation
    uint32_t  surface_layer_table_size; //!< Number of mu values in each surface-layer albedo table
    uint32_t  below_horizon_diffuse_enabled; //!< 1 = the diffuse sky also reaches rays leaving below the horizon
    uint32_t  camera_ID;
    uint32_t  bbox_UUID_base;        //!< Starting UUID for bboxes

    float2    periodic_flag;         //!< (x_periodic, y_periodic)
    uint32_t  periodic_tile_tracking; //!< 1 if rays track the periodic copy of the domain they are in (excluded primitives or a limited number of copies)
    int2      periodic_max_repeats;  //!< Periodic copies on each side of the central domain (x, y); -1 for no limit
    float2    periodic_repeated_zbounds; //!< z-range of the geometry that repeats
    float     periodic_escape_offset; //!< Distance beyond a face from which a ray leaving the periodic copies continues
    int2      camera_tile;           //!< Periodic copy of the domain the camera is in
    float3    sun_direction;
    float     solar_disk_cos_angle;
    float2    camera_direction;      //!< (elevation, azimuth) in radians
    float3    camera_position;
    float     camera_lens_diameter;
    float     camera_focal_length;
    float     FOV_aspect_ratio;
    int2      camera_resolution;
    float     camera_viewplane_length;
    int2      camera_pixel_offset;
    int2      camera_resolution_full;

    // ---- Launch grid ----
    uint32_t  launch_dim_x;
    uint32_t  launch_dim_y;
};

// ---------------------------------------------------------------------------
// Helper: pack/unpack 64-bit pointer into two 32-bit OptiX payload registers
// Device-only (requires OptiX device header)
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &u0, uint32_t &u1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    u0 = static_cast<uint32_t>(uptr >> 32);
    u1 = static_cast<uint32_t>(uptr & 0xFFFFFFFFULL);
}

static __forceinline__ __device__ void *unpackPointer(uint32_t u0, uint32_t u1) {
    const uint64_t uptr = (static_cast<uint64_t>(u0) << 32) | static_cast<uint64_t>(u1);
    return reinterpret_cast<void *>(uptr);
}

static __forceinline__ __device__ PerRayData *getPayloadPRD() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData *>(unpackPointer(u0, u1));
}
#endif // __CUDACC__

#endif // OPTIX8_LAUNCH_PARAMS_H
