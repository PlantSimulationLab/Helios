/** \file "OptiX8LaunchParams.h" Shared host/device launch parameters struct for OptiX 8.1 backend.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

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

// ---------------------------------------------------------------------------
// Per-ray payload.  Passed by pointer-in-two-registers.
// Allocated in a per-thread pool on the device.
// ---------------------------------------------------------------------------
struct PerRayData {
    double strength;           //!< Ray energy
    uint   origin_UUID;        //!< UUID of origin primitive (0 = launched from source/camera)
    bool   face;               //!< Top face = true, bottom face = false
    uint   seed;               //!< LCG random seed
    unsigned char source_ID;   //!< Source index (max 255 sources)
    bool   hit_periodic_boundary;
    float3 periodic_hit;       //!< World-space hit on the periodic wall
};

// ---------------------------------------------------------------------------
// SBT hit-group record data (per primitive type).
// ---------------------------------------------------------------------------
struct HitGroupData {
    float3*   vertices; //!< Device pointer to type-specific vertex array (layout varies by type)
    uint32_t* UUIDs;    //!< Device pointer to type-specific UUID array
    uint32_t  prim_type; //!< 0=patch, 1=triangle, 2=disk, 3=tile, 4=voxel, 5=bbox
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
    uint32_t* primitive_type;        //!< 0=patch,1=tri,2=disk,3=tile,4=voxel,5=bbox
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
    float3*   voxel_vertices;        //!< [voxel_idx * 8 + corner_idx]
    uint32_t* voxel_UUIDs;
    float3*   bbox_vertices;         //!< [bbox_idx * 4 + corner_idx] (4 vertices per planar face)
    uint32_t* bbox_UUIDs;

    // ---- Material buffers ----
    float*    rho;        //!< Reflectivity [source * Nbands_global * Nprims + band * Nprims + prim]
    float*    tau;        //!< Transmissivity (same layout)
    float*    rho_cam;    //!< Camera-weighted reflectivity
    float*    tau_cam;    //!< Camera-weighted transmissivity
    float*    specular_exponent;
    float*    specular_scale;

    // ---- Radiation energy buffers ----
    float*    radiation_in;           //!< [prim * Nbands_global + band]
    float*    radiation_out_top;      //!< [prim * Nbands_global + band]
    float*    radiation_out_bottom;   //!< [prim * Nbands_global + band]
    float*    scatter_buff_top;       //!< [prim * Nbands_global + band]
    float*    scatter_buff_bottom;    //!< [prim * Nbands_global + band]
    float*    radiation_in_camera;    //!< [prim * Nbands_launch + band]
    float*    scatter_buff_top_cam;   //!< [prim * Nbands_launch + band]
    float*    scatter_buff_bottom_cam;
    float*    radiation_specular;     //!< [src*Ncams*Nprims*Nbands + cam*Nprims*Nbands + prim*Nbands + band]
    float*    Rsky;                   //!< Sky energy absorbed [band]

    // ---- Camera output buffers ----
    uint32_t* camera_pixel_label;
    float*    camera_pixel_depth;

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

    // ---- Band launch flags ----
    bool*     band_launch_flag;       //!< [Nbands_global]: which bands to process this launch

    // ---- Texture/mask buffers ----
    uint8_t*  mask_data;              //!< Flat mask bytes: 1=opaque, 0=transparent; layout [mask0_pixels][mask1_pixels]...
    uint32_t* mask_offsets;           //!< mask_offsets[k] = start byte index in mask_data for mask k
    int32_t*  mask_sizes;             //!< 2 ints per mask: mask_sizes[k*2]=width, mask_sizes[k*2+1]=height
    int32_t*  mask_IDs;               //!< Per-primitive mask ID (-1 = no mask), indexed by global pos
    float2*   uv_data;                //!< UV coords: uv_data[pos*4 + vertex_idx] (4 vertices per primitive)
    int32_t*  uv_IDs;                 //!< Per-primitive: -1 = use parametric UV, >=0 = has custom UV in uv_data

    // ---- Scalar parameters ----
    uint32_t  Nprimitives;
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
    uint32_t  specular_reflection_enabled;
    uint32_t  camera_ID;
    uint32_t  bbox_UUID_base;        //!< Starting UUID for bboxes

    float2    periodic_flag;         //!< (x_periodic, y_periodic)
    float3    sun_direction;
    float     solar_disk_cos_angle;
    float2    camera_direction;      //!< (elevation, azimuth) in radians
    float3    camera_position;
    float     camera_lens_diameter;
    float     camera_focal_length;
    float     FOV_aspect_ratio;
    float     camera_HFOV;
    int2      camera_resolution;
    float     camera_viewplane_length;
    float     camera_pixel_solid_angle;
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
