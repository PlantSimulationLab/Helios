/** \file "RayTracing.cu.h" This file contains definitions and helper functions for CUDA/OptiX routines

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

#ifndef RAYTRACING_CUH
#define RAYTRACING_CUH

#include <stdint.h>
#include "NormalMapMath.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int uint;

//! Maximum number of radiation bands supported by the per-ray translucent-cover transmittance
//! accumulator (PerRayData::cover_transmittance). Must match HELIOS_MAX_RADIATION_BANDS in
//! OptiX8LaunchParams.h and GLASS_MAX_BANDS in shaders/common/glass_cover.glsl; the host fails fast
//! if the launched band count exceeds it.
#ifndef HELIOS_MAX_RADIATION_BANDS
#define HELIOS_MAX_RADIATION_BANDS 32
#endif

// launch parameters
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, random_seed, , );
rtDeclareVariable(unsigned int, launch_offset, , );
rtDeclareVariable(unsigned int, camera_pixel_offset_x, , );
rtDeclareVariable(unsigned int, camera_pixel_offset_y, , );
rtDeclareVariable(optix::int2, camera_resolution_full, , );
rtDeclareVariable(unsigned int, launch_face, , );
rtDeclareVariable(unsigned int, Nbands_launch, , );
rtDeclareVariable(unsigned int, Nbands_global, , );
rtBuffer<bool, 1> band_launch_flag;

// ray types
rtDeclareVariable(unsigned int, direct_ray_type, , );
rtDeclareVariable(unsigned int, diffuse_ray_type, , );
rtDeclareVariable(unsigned int, camera_ray_type, , );
rtDeclareVariable(unsigned int, pixel_label_ray_type, , );
rtDeclareVariable(unsigned int, camera_shadow_ray_type, , ); // A camera's shadow ray toward a source (see closest_hit_camera)
// The direct and diffuse rays of a scene with a background radiation field (see d_dualField()), which have their own closest-hit and miss programs (*_dual in rayHit.cu), so
// that the programs of a scene without one are those of one field and nothing more. Their payload alone carries the background-field members of PerRayData.
rtDeclareVariable(unsigned int, direct_dual_ray_type, , );
rtDeclareVariable(unsigned int, diffuse_dual_ray_type, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );

// texture-related buffers
rtBuffer<bool, 3> maskdata;
rtBuffer<int2, 1> masksize;
rtBuffer<int, 1> maskID;
rtBuffer<float2, 2> uvdata;
rtBuffer<int, 1> uvID;

// Object ID for a primitive
rtBuffer<uint, 1> objectID;
rtBuffer<int2, 1> object_subdivisions;

// ID for first primitive in an object
rtBuffer<uint, 1> primitiveID;

// UUID to array position lookup (indexed by UUID, returns array position)
// Size: max_UUID + 1 (sparse array)
// Usage: uint position = primitive_positions[UUID];
rtBuffer<uint, 1> primitive_positions;

// Radiation sources buffers
rtDeclareVariable(unsigned int, Nsources, , );
rtBuffer<float, 1> source_fluxes;
rtBuffer<float, 1> source_fluxes_cam; // Camera weights of the source fluxes, for the surface layers' reflection of direct radiation [source * Nbands * Ncameras]
rtBuffer<float3, 1> source_positions;
rtBuffer<float3, 1> source_rotations;
rtBuffer<float2, 1> source_widths;
rtBuffer<unsigned int, 1> source_types;

// Diffuse radiation
rtBuffer<float, 1> diffuse_flux;
rtBuffer<float, 1> diffuse_extinction;
rtBuffer<float3, 1> diffuse_peak_dir;
rtBuffer<float, 1> diffuse_dist_norm;

// Atmospheric sky radiance parameters (for camera miss rays)
rtBuffer<float4, 1> sky_radiance_params; // Per-band: (circumsolar_strength, circumsolar_width, horizon_coeff, zenith_scale)
rtBuffer<float, 1> camera_sky_radiance; // Per-band: base sky radiance (W/m²/sr) for camera atmospheric model
rtBuffer<float, 1> camera_diffuse_flux; // Per-band: hemispherical diffuse sky flux for camera longwave/emission sky miss
rtBuffer<unsigned int, 1> band_emission_flag; // Per-band: 1 if emission is enabled (gates camera longwave sky sampling)
rtDeclareVariable(float3, sun_direction, , ); // Sun direction vector for sky radiance evaluation

// Solar disk rendering for camera (lens flare support)
rtBuffer<float, 1> solar_disk_radiance; // Per-band: solar disk radiance (W/m²/sr) when looking directly at sun
rtDeclareVariable(float, solar_disk_cos_angle, , ); // Cosine of solar angular radius (~0.265°, cos ≈ 0.99999)

//--- Patches ---//
rtBuffer<float3, 2> patch_vertices;
rtBuffer<unsigned int, 1> patch_UUID;
//--- Triangles ---//
rtBuffer<float3, 2> triangle_vertices;
rtBuffer<unsigned int, 1> triangle_UUID;
//--- Disks ---//
// To-Do: disks not finished
rtBuffer<unsigned int, 1> disk_UUID;
//--- Tiles ---//
rtBuffer<float3, 2> tile_vertices;
rtBuffer<unsigned int, 1> tile_UUID;
//--- Bounding Box ---//
rtBuffer<float3, 2> bbox_vertices;
rtBuffer<unsigned int, 1> bbox_UUID;

// Primitive data
rtDeclareVariable(unsigned int, Nprimitives, , );
// Slots per primitive dimension of the outgoing, scattered, camera, direct-flux and spectral buffers: Nprimitives, or 2 Nprimitives with a background radiation field
// (slot Nprimitives + p holds primitive p in that field; see RayTracingGeometry::fluxSlotCount()). Nprimitives itself also locates the periodic boundary faces.
rtDeclareVariable(unsigned int, Nflux_slots, , );
rtDeclareVariable(unsigned int, normal_map_slot_count, , ); // Number of normal-map slots; the background field's moments are in slots normal_map_slot_count + slot
rtDeclareVariable(unsigned int, vertex_field_stride, , ); // Number of smoothing vertices; the background field's vertex values are at vertex_field_stride + vertex (set with the vertex outgoing radiation)
rtDeclareVariable(unsigned int, bbox_UUID_base, , ); // Starting UUID for bboxes (max_real_UUID + 1)
rtDeclareVariable(float2, periodic_flag, , );
rtDeclareVariable(unsigned int, periodic_tile_tracking, , ); // 1 if rays track the periodic copy of the domain they are in (excluded primitives or a limited number of copies)
rtDeclareVariable(unsigned int, periodic_excluded_present, , ); // 1 if periodic_excluded holds a flag for every primitive position
rtDeclareVariable(int2, periodic_max_repeats, , ); // Periodic copies on each side of the central domain (x, y); -1 for no limit
rtDeclareVariable(float2, periodic_repeated_zbounds, , ); // z-range of the geometry that repeats
rtDeclareVariable(float, periodic_escape_offset, , ); // Distance beyond a face from which a ray leaving the periodic copies continues
rtDeclareVariable(int2, camera_tile, , ); // Periodic copy of the domain the camera is in
rtBuffer<char, 1> periodic_excluded; // [position]: 1 if the primitive exists only in the central periodic copy of the domain
rtBuffer<char, 1> twosided_flag;
rtBuffer<float, 2> transform_matrix;
rtBuffer<unsigned int, 1> primitive_type;
rtBuffer<float, 1> primitive_solid_fraction;

rtBuffer<float, 1> rho, tau;
rtBuffer<float, 1> rho_cam, tau_cam;
// Camera-weighted reflectivity of a spectrally flat, perfectly white surface [source * Nbands_global * Ncameras + band_global * Ncameras + camera]
rtBuffer<float, 1> white_reference_cam;

// Translucent cover (glass/plastic) material. Same layout as rho/tau:
// [source * Nprimitives * Nbands_global + primitive * Nbands_global + band_global].
// When is_glass != 0 for an entry, the angular transmittance/reflectance is computed on-device from
// glass_n (refractive index) and glass_KL (Bouguer absorption product K*L) via the Fresnel+Bouguer
// model, overriding the constant rho/tau for that entry. glass_enabled is a cheap gate: when 0 (no
// primitive uses the glass model) the any-hit programs early-return so non-glass scenes pay no cost.
rtBuffer<float, 1> glass_n, glass_KL, is_glass;
rtDeclareVariable(unsigned int, glass_enabled, , );

// Specular surface layer (see RayTracingMaterial::surface_model). surface_layer_enabled is 0 when no primitive has one; the
// buffers are then one-element placeholders and the outgoing and scattered buffers hold a single component.
rtBuffer<unsigned int, 1> surface_model; // [primitive]: 0 = Lambertian, m > 0 = surface model m - 1
rtBuffer<unsigned int, 1> surface_model_band_table; // [(m - 1) * Nbands_global + band]: albedo table, or UINT_MAX for no layer in the band
rtBuffer<float, 1> surface_layer_tables; // [table * (surface_layer_table_size + 3) + i]: a(mu) at mu = i / (size - 1), then abar, n and alpha
rtDeclareVariable(unsigned int, surface_layer_enabled, , );
rtDeclareVariable(unsigned int, surface_layer_table_size, , );

// Spectral textures (Context::setPrimitiveSpectralTexture()): the class maps and the palettes, each packed into one buffer in the
// layout described in shaders/common/spectral_texture.glsl (see helios::packSpectralClassMaps() and helios::packSpectralPalettes())
rtBuffer<unsigned int, 1> spectral_class_maps;
rtBuffer<unsigned int, 1> spectral_palette;
// Normal maps (Context::setMaterialNormalMap()), packed as described in include/NormalMapMath.h; word 0 is 0 when no primitive has one
rtBuffer<unsigned int, 1> normal_maps;
// Camera launches: 1 if the host removed the direct launch's deposit from normal-mapped primitives' outgoing radiation, so the camera shades it per hit point
rtDeclareVariable(unsigned int, normal_map_direct, , );
rtDeclareVariable(unsigned int, camera_direct_per_pixel, , ); // Camera launches: 1 if smoothed primitives' direct radiation is evaluated per pixel sample
rtDeclareVariable(unsigned int, camera_direct_all, , ); // Camera launches: 1 if every primitive's (but a translucent cover's) direct radiation is evaluated per pixel sample
// Slot of each normal-mapped primitive in normal_map_moments (0xFFFFFFFF for none), and [slot][face][launch band][3] the flux-weighted directions of the diffuse radiation
// reaching each face (see RayTracingGeometry::normal_map_slots); one-element placeholders when no primitive has a map. With a background radiation field, its moments
// follow in a second set of slots.
rtBuffer<unsigned int, 1> normal_map_slots;
rtBuffer<float, 1> normal_map_moments;
// The radiation reaching primitives with a spectral texture [face][source slot][flux slot][launch band], booked when spectral_incident_enabled is 1, and the
// packed camera-weighted outgoing radiation of each class for the camera being traced (see helios::packSpectralCameraFlux())
rtBuffer<float, 1> spectral_incident;
rtDeclareVariable(unsigned int, spectral_incident_enabled, , );
rtBuffer<unsigned int, 1> spectral_camera_flux;
// With camera flux smoothing, the radiation of primitives with a spectral texture on the smoothing vertices, packed (see helios::packSpectralVertexIncident())
rtBuffer<unsigned int, 1> spectral_vertex;
// Class of the spectral texture seen in each pixel by the pixel-label launch (0 = none), sized when some primitive has a spectral texture
rtBuffer<unsigned int, 1> camera_pixel_class;
rtBuffer<float, 1> vertex_normals; // Unit shading normal per smoothing vertex [vertex * 3], for camera flux smoothing; a one-element placeholder when there are none
rtBuffer<float, 1> vertex_normal_map_moments; // normal_map_moments averaged onto the smoothing vertices [field][vertex][face][band_launch][3]; a one-element placeholder when there is none
rtBuffer<float, 1> radiation_specular; // Direct flux reaching primitives with a surface layer, and with camera flux smoothing every smoothed primitive [source][flux slot][band_launch]

// ---------------------------------------------------------------------------
// Background radiation field (see RayTracingGeometry::periodicDualField()). With primitives excluded from the periodic
// boundary, the buffers of outgoing and scattered radiation, of what cameras read, and of the direct flux hold two fields:
// field 0, the scene as it is, in flux slots [0, Nprimitives), and field 1, the background scene without the excluded
// primitives, in flux slots [Nprimitives, 2 Nprimitives). A ray or camera reads field 0 where it hits the central copy
// of the domain and field 1 in every other copy. Material and geometry buffers are indexed by the primitive's position
// alone. Must match OptiX8DeviceCode.cu (fluxSlot()) and the Vulkan shaders (flux_slot() in common/buffer_indexing.glsl).
// ---------------------------------------------------------------------------

//! Whether a background field is solved alongside the scene as it is
__device__ __forceinline__ bool d_dualField() {
    return Nflux_slots != Nprimitives;
}

//! Flux slot of the primitive at position pos in a field
__device__ __forceinline__ uint d_fluxSlot(uint pos, uint field) {
    return pos + field * Nprimitives;
}

//! Index of the direct flux of source s reaching the primitive at position pos, in a field: radiation_specular is [source][flux slot][launch band]
__device__ __forceinline__ size_t d_directFluxIndex(uint s, uint pos, uint field, uint b_launch) {
    return (size_t(s) * Nflux_slots + d_fluxSlot(pos, field)) * Nbands_launch + b_launch;
}

//! Texture (u,v) of a launch point sp in a primitive's canonical space, as d_sampleTexture_patch() and d_sampleTexture_triangle() compute it
__device__ optix::float2 d_originTextureUV(uint ptype, const optix::float3 &sp, int uv_ID) {
    if (ptype == 1) {
        // Barycentric weights of the second and third vertices of the canonical triangle (0,0,0), (0,1,0), (1,1,0)
        const float beta = sp.y - sp.x;
        const float gamma = sp.x;
        if (uv_ID == -1) { // no (u,v) coordinates, and so no spectral texture either: parametric, as in OptiX 8
            return optix::make_float2(sp.y, sp.x);
        }
        const optix::float2 uv0 = uvdata[optix::make_uint2(0, uv_ID)];
        const optix::float2 uv1 = uvdata[optix::make_uint2(1, uv_ID)];
        const optix::float2 uv2 = uvdata[optix::make_uint2(2, uv_ID)];
        return uv0 + beta * (uv1 - uv0) + gamma * (uv2 - uv0);
    }
    const optix::float2 uv = optix::make_float2(sp.x + 0.5f, sp.y + 0.5f);
    if (uv_ID == -1) {
        return uv;
    }
    const optix::float2 uvmin = uvdata[optix::make_uint2(0, uv_ID)];
    const float du = uvdata[optix::make_uint2(1, uv_ID)].x - uvmin.x;
    const float dv = uvdata[optix::make_uint2(2, uv_ID)].y - uvdata[optix::make_uint2(1, uv_ID)].y;
    return optix::make_float2(uvmin.x + uv.x * du, uvmin.y + uv.y * dv);
}

// ---------------------------------------------------------------------------
// Normal maps (the conventions are specified in include/NormalMapMath.h)
// ---------------------------------------------------------------------------

//! Reads the packed normal-map words for the shared helios_nm functions
struct NormalMapWords {
    __device__ __forceinline__ unsigned int operator[](unsigned int i) const {
        return normal_maps[i];
    }
};

__device__ __forceinline__ helios_nm::Vec3 d_toNM(const optix::float3 &v) {
    return helios_nm::make(v.x, v.y, v.z);
}
__device__ __forceinline__ optix::float3 d_fromNM(const helios_nm::Vec3 &v) {
    return optix::make_float3(v.x, v.y, v.z);
}

//! Normal map of the primitive at position pos, or helios_nm::NO_MAP
__device__ __forceinline__ uint d_normalMapOf(uint position) {
    return helios_nm::mapIndex(NormalMapWords(), position);
}

//! Whether the primitive at position pos has a specular surface layer
__device__ __forceinline__ bool d_hasSurfaceLayer(uint position) {
    return surface_layer_enabled != 0u && surface_model[position] != 0u;
}

//! Shading normal on the top face of a primitive at canonical point sp (see d_originTextureUV()), about the unit base normal base_top on the side of the facet's top normal: the nearest texel for ray launches, bilinear for cameras
__device__ optix::float3 d_normalMapShadingNormalAt(uint ptype, uint map, const float *T, int uv_ID, const optix::float3 &base_top, const optix::float3 &sp, bool bilinear) {
    helios_nm::Vec3 E1, E2;
    helios_nm::primitiveEdges(T, ptype, E1, E2);
    optix::float2 uv0 = optix::make_float2(0.f, 0.f), uv1 = uv0, uv2 = uv0;
    if (uv_ID != -1) {
        uv0 = uvdata[optix::make_uint2(0, uv_ID)];
        uv1 = uvdata[optix::make_uint2(1, uv_ID)];
        uv2 = uvdata[optix::make_uint2(2, uv_ID)];
    }
    float ax, ay, bx, by;
    helios_nm::uvJacobian(ptype, uv_ID != -1, uv0.x, uv0.y, uv1.x, uv1.y, uv2.x, uv2.y, ax, ay, bx, by);
    const helios_nm::Frame frame = helios_nm::buildFrame(E1, E2, ax, ay, bx, by, d_toNM(base_top));
    float u, v;
    helios_nm::texcoord(ptype, uv0.x, uv0.y, ax, ay, bx, by, sp.x, sp.y, u, v);
    const NormalMapWords words;
    const helios_nm::Vec3 n_ts = bilinear ? helios_nm::fetchBilinear(words, map, u, v) : helios_nm::fetchNearest(words, map, u, v);
    return d_fromNM(helios_nm::perturb(frame, n_ts));
}

//! The canonical point of a hit from the hit attributes (patch-local coordinates in [0,1]^2, or the barycentric weights of a triangle's second and third corners)
__device__ __forceinline__ optix::float3 d_canonicalPointFromHit(uint ptype, float surface_u, float surface_v) {
    return ptype == 1 ? optix::make_float3(surface_v, surface_u + surface_v, 0.f) : optix::make_float3(surface_u - 0.5f, surface_v - 0.5f, 0.f);
}

//! Shading normal on the top face of the primitive at position pos at a hit with attributes (surface_u, surface_v), with the facet's top normal returned in normal_top
__device__ optix::float3 d_hitShadingNormalTop(uint position, uint map, float surface_u, float surface_v, bool bilinear, optix::float3 &normal_top) {
    float T[16];
    for (uint i = 0; i < 16; i++) {
        T[i] = transform_matrix[optix::make_uint2(i, position)];
    }
    const uint ptype = primitive_type[position];
    helios_nm::Vec3 E1, E2;
    helios_nm::primitiveEdges(T, ptype, E1, E2);
    normal_top = d_fromNM(helios_nm::normalize(helios_nm::cross(E1, E2)));
    return d_normalMapShadingNormalAt(ptype, map, T, uvID[position], normal_top, d_canonicalPointFromHit(ptype, surface_u, surface_v), bilinear);
}

//! Palette entry of the class of a primitive's spectral texture at texture coordinate (u,v), with the texel convention of the masks, or 0xFFFFFFFF if the primitive has none or the texel is transparent (class 0)
__device__ uint d_spectralPaletteEntry(uint position, optix::float2 uv) {
    if (spectral_palette[0] == 0u || spectral_class_maps[0] == 0u) {
        return 0xFFFFFFFFu;
    }
    const uint first_entry = spectral_palette[spectral_palette[3] + position];
    const uint num_prims = spectral_class_maps[0];
    const uint class_map = spectral_class_maps[2 + position];
    if (first_entry == 0xFFFFFFFFu || class_map == 0xFFFFFFFFu) {
        return 0xFFFFFFFFu;
    }
    const uint info = 2 + num_prims + 3 * class_map;
    const int width = int(spectral_class_maps[info]);
    const int height = int(spectral_class_maps[info + 1]);
    int ix = int(floorf(float(width - 1) * uv.x));
    int iy = int(floorf(float(height - 1) * (1.f - uv.y)));
    ix = max(0, min(ix, width - 1));
    iy = max(0, min(iy, height - 1));
    const uint texel = spectral_class_maps[info + 2] + uint(iy) * uint(width) + uint(ix);
    const uint word = spectral_class_maps[2 + num_prims + 3 * spectral_class_maps[1] + texel / 2];
    const uint texel_class = (texel % 2 == 0) ? (word & 0xFFFFu) : (word >> 16);
    return texel_class == 0u ? 0xFFFFFFFFu : first_entry + texel_class - 1u;
}

//! Class-flux slot, in a radiation field (see d_dualField()), of the class a camera ray hits on the primitive at `position`, from the hit attributes (patch-local coordinates in [0,1], or the barycentric
//! weights of a triangle's second and third corners), or 0xFFFFFFFF. The packed buffer holds one first slot per flux slot.
__device__ uint d_spectralCameraClassSlot(uint position, uint ptype, float surface_u, float surface_v, int uv_ID, uint field) {
    if (spectral_camera_flux[0] == 0u || spectral_palette[0] == 0u) {
        return 0xFFFFFFFFu;
    }
    const uint first_slot = spectral_camera_flux[3 + d_fluxSlot(position, field)];
    const uint first_entry = spectral_palette[spectral_palette[3] + position];
    if (first_slot == 0xFFFFFFFFu || first_entry == 0xFFFFFFFFu) {
        return 0xFFFFFFFFu;
    }
    const optix::float3 sp = ptype == 1 ? optix::make_float3(surface_v, surface_u + surface_v, 0.f) : optix::make_float3(surface_u - 0.5f, surface_v - 0.5f, 0.f);
    const uint entry = d_spectralPaletteEntry(position, d_originTextureUV(ptype, sp, uv_ID));
    return entry == 0xFFFFFFFFu ? 0xFFFFFFFFu : first_slot + (entry - first_entry);
}

//! Camera-weighted outgoing radiation of a class-flux slot from one face in a launch band
__device__ float d_spectralClassFlux(uint slot, bool top_face, uint launch_band) {
    const uint num_slots = spectral_camera_flux[0];
    const uint num_bands = spectral_camera_flux[1];
    const uint num_prims = spectral_camera_flux[2];
    return __uint_as_float(spectral_camera_flux[3 + num_prims + ((top_face ? 0u : num_slots) + slot) * num_bands + launch_band]);
}

//! Value of camera-weighted palette table `table` (7 reflectivity, 8 transmissivity) for material slot `slot`
__device__ float d_spectralPaletteValueCam(uint table, uint slot, uint entry, uint b_global, uint camera);

//! Direction from point to radiation source s: the source's direction for a collimated source or the sun sphere, and toward its centre for a finite source
__device__ optix::float3 d_directionToSource(uint s, const optix::float3 &point) {
    if (source_types[s] == 0 || source_types[s] == 2) {
        return optix::normalize(source_positions[s]);
    }
    return optix::normalize(source_positions[s] - point);
}

//! Direct radiation of a source reaching one face of a normal-mapped facet that is not smoothed, evaluated with the map's shading normal
/**
 * face_normal and face_shading_normal are the facet's normal and the shading normal of that face, and facet_value what the face itself received. That carries the cosine of the facet's normal, which is
 * replaced by that of the shading normal. A face turned away from the source received nothing. With soften, the rescaled value is multiplied by the shadow-terminator softening of Chiang et al. (see
 * helios_nm::terminatorSoftening()). (A smoothed primitive's direct radiation is evaluated per pixel sample instead; see d_pixelDirectRadiation() in rayHit.cu.)
 */
__device__ float d_faceDirect(float facet_value, const optix::float3 &to_source, const optix::float3 &face_normal, const optix::float3 &face_shading_normal, bool soften) {
    const float mu_facet = optix::dot(face_normal, to_source);
    const float mu_shading = fmaxf(0.f, optix::dot(face_shading_normal, to_source));
    if (mu_shading <= 0.f || mu_facet <= 0.f) {
        return 0.f;
    }
    const float softening = soften ? helios_nm::terminatorSoftening(d_toNM(face_normal), d_toNM(face_shading_normal), d_toNM(to_source)) : 1.f;
    return facet_value * mu_shading / mu_facet * softening;
}

//! Value of layer `layer` of spectral_vertex on face `face` (0 top, 1 bottom) in a launch band, interpolated from the facet's vertices v with weights w, in a radiation field (see d_dualField()).
//! The values are packed [face][layer][field][vertex][band]: the background field's vertices follow those of the scene as it is.
__device__ float d_spectralVertexValue(uint face, uint layer, uint launch_band, const int v[4], const float w[4], uint field) {
    const uint layers = spectral_vertex[0];
    const uint num_bands = spectral_vertex[1];
    const uint num_vertices = spectral_vertex[2]; // of every field
    const size_t base = 4 + (size_t(face * layers + layer) * num_vertices + size_t(field) * vertex_field_stride) * num_bands + launch_band;
    float sum = 0.f;
    for (int c = 0; c < 4; c++) {
        if (v[c] >= 0) {
            sum += w[c] * __uint_as_float(spectral_vertex[base + size_t(v[c]) * num_bands]);
        }
    }
    return sum;
}

//! With camera flux smoothing: the camera-weighted outgoing radiation of the class of class-flux slot `slot` on the primitive at `position`, from one face in a launch band, of all but the sources'
//! direct radiation: the radiation from the sky and other surfaces (the last source slot, layer 0 of spectral_vertex) and the emitted radiation (layer 1), interpolated from the facet's vertices v
//! (v[3] < 0 for a triangle) with weights w. The direct radiation of the sources is evaluated per pixel sample (see d_pixelDirectRadiation() in rayHit.cu). The slot and the vertex values are those
//! of radiation field `field` (see d_dualField()).
__device__ float d_spectralSmoothedClassFlux(uint position, uint slot, bool top_face, uint launch_band, uint b_global, const int v[4], const float w[4], uint field) {
    const uint camera = spectral_vertex[3];
    const uint entry = spectral_palette[spectral_palette[3] + position] + (slot - spectral_camera_flux[3 + d_fluxSlot(position, field)]);
    const uint same_face = top_face ? 0u : 1u;
    float flux = d_spectralVertexValue(same_face, 1, launch_band, v, w, field); // emitted
    // The radiation from the sky and other surfaces (the last source slot)
    const float sky_same = d_spectralVertexValue(same_face, 0, launch_band, v, w, field);
    const float sky_other = d_spectralVertexValue(1 - same_face, 0, launch_band, v, w, field);
    return flux + d_spectralPaletteValueCam(7, Nsources, entry, b_global, camera) * sky_same + d_spectralPaletteValueCam(8, Nsources, entry, b_global, camera) * sky_other;
}

//! Whether a ray with palette entry `entry` uses the palette in global band `b_global`
__device__ bool d_spectralPaletteApplies(uint entry, uint b_global) {
    return entry != 0xFFFFFFFFu && spectral_palette[spectral_palette[4] + b_global] != 0u;
}

//! Value of palette table `table` (5 reflectivity, 6 transmissivity) for material slot `slot`
__device__ float d_spectralPaletteValue(uint table, uint slot, uint entry, uint b_global) {
    return __uint_as_float(spectral_palette[spectral_palette[table] + (slot * spectral_palette[0] + entry) * spectral_palette[1] + b_global]);
}

//! Value of camera-weighted palette table `table` (7 reflectivity, 8 transmissivity) for material slot `slot`
__device__ float d_spectralPaletteValueCam(uint table, uint slot, uint entry, uint b_global, uint camera) {
    return __uint_as_float(spectral_palette[spectral_palette[table] + ((slot * spectral_palette[0] + entry) * spectral_palette[1] + b_global) * spectral_palette[2] + camera]);
}

rtDeclareVariable(unsigned int, specular_reflection_enabled, , ); // 1 = this launch evaluates the surface layers' reflection lobe of the direct radiation
rtDeclareVariable(unsigned int, scattering_iteration, , ); // Current scattering iteration (0-based)
rtDeclareVariable(unsigned int, below_horizon_diffuse_enabled, , ); // 1 = the diffuse sky also reaches rays leaving below the horizon

// Output buffers
rtBuffer<float, 1> radiation_in;
// Absorbed radiation that arrived on the top face [primitive * Nbands_launch + band], written only when face_absorption_enabled is set (a one-element placeholder otherwise)
rtBuffer<float, 1> radiation_in_top;
rtDeclareVariable(unsigned int, face_absorption_enabled, , );
// Forward scattering estimate [primitive * Nbands_launch + band], written only when forward_scatter_enabled is set (scattering launches)
rtBuffer<float, 1> forward_scatter;
rtDeclareVariable(unsigned int, forward_scatter_enabled, , );
rtBuffer<float, 1> radiation_in_camera;
rtBuffer<float, 1> radiation_out_top;
rtBuffer<float, 1> radiation_out_bottom;
rtBuffer<int, 1> smoothing_vertex_indices; // Shared vertex index per facet corner, 4 per primitive; entry 4*prim is negative when the primitive is not smoothed, entry 4*prim+3 is negative for a triangle
rtBuffer<float, 1> vertex_radiation_out_top; // Outgoing radiation resampled onto the shared mesh vertices [component][field][vertex][band_launch]
rtBuffer<float, 1> vertex_radiation_out_bottom;
rtBuffer<float, 1> scatter_buff_top;
rtBuffer<float, 1> scatter_buff_bottom;
rtBuffer<float, 1> scatter_buff_top_cam;
rtBuffer<float, 1> scatter_buff_bottom_cam;
// Each camera's white reference [camera * Nprimitives * Nbands_launch + primitive * Nbands_launch + band], accumulated over all the launches of a runBand()
rtBuffer<float, 1> white_reference_top_cam;
rtBuffer<float, 1> white_reference_bottom_cam;

// Camera variables
rtBuffer<unsigned int, 1> camera_pixel_label;
rtBuffer<float, 1> camera_pixel_depth;

// Ray counters
rtBuffer<unsigned int, 1> abandoned_ray_count; // [1]: direct and diffuse rays abandoned at the periodic wrap limit
rtDeclareVariable(unsigned int, camera_ID, , );
rtDeclareVariable(unsigned int, Ncameras, , );
rtDeclareVariable(float3, camera_position, , );
rtDeclareVariable(float2, camera_direction, , );
rtDeclareVariable(float, camera_lens_diameter, , );
rtDeclareVariable(float, FOV_aspect_ratio, , );
rtDeclareVariable(float, camera_focal_length, , ); // Focal plane distance (working distance for ray generation), NOT lens optical focal length
rtDeclareVariable(float, camera_viewplane_length, , );


void queryGPUMemory(void);

/** OptiX ray payload - contains data for each ray traversal. */
struct PerRayData {

    //! "strength" or amount of energy associated with the ray.
    double strength;
    //! UUID of primitive from which ray originated
    uint origin_UUID;
    //! Face of primitive from which ray was launched (true=top, false=bottom)
    bool face;
    //! Seed for curand random number generator
    uint seed;
    //! Number of periodic boundary intersections for ray
    // unsigned char periodic_depth;
    //! Numerical identifier for radiation source corresponding to each ray
    /**
     * Diffuse, scattered and emitted rays carry Nsources, the index of the diffuse/scatter material slot.
     * \note The data type limits to a maximum of 255 radiation sources
     */
    unsigned char source_ID;
    //! Flag to determine if ray hit a periodic boundary
    bool hit_periodic_boundary;
    //! Periodic copy of the domain the ray is in (0 = central; counted whenever a boundary is periodic)
    int tile_x;
    int tile_y;
    //! Periodic copy of the domain of the primitive the ray was launched from (origin_UUID), which the ray does not hit in that copy
    int origin_tile_x;
    int origin_tile_y;
    optix::float3 periodic_hit;
    //! Distance from the segment origin to the periodic boundary it hit
    float periodic_hit_t;
    //! Per-launch-band transmittance accumulated as the ray passes through translucent covers
    //! (glass/plastic). Initialised to 1 in the raygen programs; multiplied by the Fresnel+Bouguer
    //! tau(theta) in the any-hit program (direct) or the closest-hit programs (diffuse, camera);
    //! applied to what the ray delivers past the covers. Indexed by launch-band b.
    float cover_transmittance[HELIOS_MAX_RADIATION_BANDS];
    //! |cosine| of the angle between a direct or diffuse ray and the normal of the primitive it left
    float origin_cos;
    //! Palette entry of the class of a spectral texture at the point the ray left, or 0xFFFFFFFF to use the origin primitive's own values
    uint palette_entry;
    //! Flag set when a diffuse/camera/pixel-label ray hit a translucent cover and continues past it from cover_exit
    bool hit_cover;
    //! World-space point just beyond the cover where the ray continues
    optix::float3 cover_exit;
    //! Set when a camera's shadow ray reaches the source it was traced toward (miss_camera_shadow)
    bool source_reached;

    // The members below are set and read only for rays of the direct_dual_ray_type and diffuse_dual_ray_type, the direct and diffuse rays of a scene with a background
    // radiation field (see d_dualField()); rays of every other type leave them unset.

    //! The ray books what it delivers in both fields (its primitive is not excluded from the periodic boundary). Such a ray passes the excluded primitives, which the
    //! background field does not contain, and records the nearest one it would have hit in each segment (see passExcludedPrimitive() in primitiveIntersection.cu).
    bool dual_ray;
    //! A dual ray's delivery to the scene as it is (field 0) is settled by an excluded primitive: a direct ray is blocked by it, and a diffuse ray has received its radiation
    bool excluded_resolved;
    //! Distance to the nearest excluded primitive passed in the current segment (infinite if none)
    float recorded_t;
    //! Its position
    uint recorded_position;
    //! Its hit attributes surface_u and surface_v (see primitiveIntersection.cu)
    float recorded_u;
    float recorded_v;
};

//! Distance of the recorded excluded primitive of a dual ray that has passed none in its current segment
#define NO_RECORDED_HIT 3.0e38f

//! Sets the background-field members of the payload of a direct or diffuse ray of a scene with a background field, launched from the primitive at position pos: it books what
//! it delivers in both fields if the primitive is in the background field. Must match startDualRay() in OptiX8DeviceCode.cu.
static __device__ __inline__ void d_startDualRay(PerRayData &p, uint pos) {
    p.dual_ray = d_dualField() && periodic_excluded[pos] == 0;
    p.excluded_resolved = false;
    p.recorded_t = NO_RECORDED_HIT;
    p.recorded_position = 0u;
    p.recorded_u = 0.f;
    p.recorded_v = 0.f;
}

//! The field a ray reads where it is: the scene as it is in the central copy of the domain, and the background in every other copy. Must match fieldSeenBy() in OptiX8DeviceCode.cu.
static __device__ __inline__ uint d_fieldSeenBy(const PerRayData &p) {
    return (d_dualField() && (p.tile_x != 0 || p.tile_y != 0)) ? 1u : 0u;
}

//! Diffuse, camera and pixel-label rays continue past a translucent cover from this distance (m) beyond it.
#define COVER_EXIT_OFFSET 1e-4f
//! Number of periodic-boundary hits after which a ray of any type is abandoned (the ray is wrapped at most MAX_PERIODIC_WRAPS - 1
//! times): it deposits nothing, as if blocked. Must match MAX_PERIODIC_WRAPS in OptiX8DeviceCode.cu and the Vulkan shaders.
#define MAX_PERIODIC_WRAPS 1000
//! Number of translucent covers a diffuse, camera or pixel-label ray can pass in addition to its periodic-boundary wraps.
#define MAX_COVER_PASSES 32
//! Maximum number of segments a diffuse, camera or pixel-label ray is traced in: each translucent cover passed
//! and each periodic-boundary wrap starts a new segment. A ray that runs out stops without further contribution.
#define MAX_RAY_SEGMENTS (MAX_PERIODIC_WRAPS + MAX_COVER_PASSES)

static __device__ void init_state(PerRayData *prd);

//! Reset the per-band translucent-cover transmittance accumulator to 1 (no covers crossed yet).
//! Called in the raygen programs before each ray is traced.
static __device__ __inline__ void initCoverTransmittance(PerRayData &p) {
    for (int i = 0; i < HELIOS_MAX_RADIATION_BANDS; i++) {
        p.cover_transmittance[i] = 1.f;
    }
}

__device__ __inline__ void atomicFloatAdd(float *address, float val) {
#if __CUDA_ARCH__ >= 200
    atomicAdd(address, val);
#else
    float old = val;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f)
        ;
#endif
} // end atomic float add

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ __inline__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// -- random number generation -- //

template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev) {
    return ((float) lcg(prev) / (float) 0x01000000);
}

/** Function to rotate a 3D vector given spherical angles elevation and azimuth ON THE GPU. */
__device__ float3 d_rotatePoint(const float3 &position, const float &theta, const float &phi) {

    /** @param[in] "position" 3D coordinate of point to be rotated. */
    /** @param[in] "theta" elevation angle of rotation. */
    /** @param[in] "phi" azimuthal angle of rotation. */

    float Ry[3][3], Rz[3][3];

    float st = sinf(theta);
    float ct = cosf(theta);

    float sp = sinf(phi);
    float cp = cosf(phi);

    // Setup the rotation matrix, this matrix is based off of the rotation matrix used in glRotatef.
    Ry[0][0] = ct;
    Ry[0][1] = 0.f;
    Ry[0][2] = st;
    Ry[1][0] = 0.f;
    Ry[1][1] = 1.f;
    Ry[1][2] = 0.f;
    Ry[2][0] = -st;
    Ry[2][1] = 0.f;
    Ry[2][2] = ct;

    Rz[0][0] = cp;
    Rz[0][1] = -sp;
    Rz[0][2] = 0.f;
    Rz[1][0] = sp;
    Rz[1][1] = cp;
    Rz[1][2] = 0.f;
    Rz[2][0] = 0.f;
    Rz[2][1] = 0.f;
    Rz[2][2] = 1.f;

    // Multiply Ry*Rz

    float rotMat[3][3] = {0.f};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                rotMat[i][j] = rotMat[i][j] + Rz[i][k] * Ry[k][j];
            }
        }
    }

    // Multiply the rotation matrix with the position vector.
    float3 tmp;
    tmp.x = rotMat[0][0] * position.x + rotMat[0][1] * position.y + rotMat[0][2] * position.z;
    tmp.y = rotMat[1][0] * position.x + rotMat[1][1] * position.y + rotMat[1][2] * position.z;
    tmp.z = rotMat[2][0] * position.x + rotMat[2][1] * position.y + rotMat[2][2] * position.z;

    return tmp;
}

__device__ void d_makeTransformMatrix(optix::float3 rotation, float (&T)[16]) {

    T[0] = cosf(rotation.z) * cosf(rotation.y);
    T[1] = cosf(rotation.z) * sinf(rotation.y) * sinf(rotation.x) - sinf(rotation.z) * cosf(rotation.x);
    T[2] = cosf(rotation.z) * sinf(rotation.y) * cosf(rotation.x) + sinf(rotation.z) * sinf(rotation.x);
    T[3] = 0.f;
    T[4] = sinf(rotation.z) * cosf(rotation.y);
    T[5] = sinf(rotation.z) * sinf(rotation.y) * sinf(rotation.x) + cosf(rotation.z) * cosf(rotation.x);
    T[6] = sinf(rotation.z) * sinf(rotation.y) * cosf(rotation.x) - cosf(rotation.z) * sinf(rotation.x);
    T[7] = 0.f;
    T[8] = -sinf(rotation.y);
    T[9] = cosf(rotation.y) * sinf(rotation.x);
    T[10] = cosf(rotation.y) * cosf(rotation.x);
    T[11] = 0.f;
    T[12] = 0.f;
    T[13] = 0.f;
    T[14] = 0.f;
    T[15] = 1.f;
}

__device__ void d_invertMatrix(const float *m, float *minv) {

    float inv[16], det;

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    // if (det == 0)
    // return false;

    det = 1.0 / det;

    for (int i = 0; i < 16; i++)
        minv[i] = inv[i] * det;
}


//! Function to transform a 3D point based on current affine transformation matrix on the GPU
__device__ void d_transformPoint(const float (&transform_matrix)[16], float3 &v) {

    float3 V;

    V.x = transform_matrix[0] * v.x + transform_matrix[1] * v.y + transform_matrix[2] * v.z + transform_matrix[3];
    V.y = transform_matrix[4] * v.x + transform_matrix[5] * v.y + transform_matrix[6] * v.z + transform_matrix[7];
    V.z = transform_matrix[8] * v.x + transform_matrix[9] * v.y + transform_matrix[10] * v.z + transform_matrix[11];

    v = V;
}

//! Function to calculate the magnitude of float3 vector on the GPU
__device__ float d_magnitude(const float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

//! Function to calculate the surface area of a primitive
__device__ float d_calculatePrimitiveArea(const uint primitive_type, const float (&transform_matrix)[16]) {

    if (primitive_type == 0 || primitive_type == 3) { // Patch or AlphaMask

        float3 s0 = make_float3(0, 0, 0);
        float3 s1 = make_float3(1, 1, 0);
        float3 s2 = make_float3(0, 1, 0);
        d_transformPoint(transform_matrix, s0);
        d_transformPoint(transform_matrix, s1);
        d_transformPoint(transform_matrix, s2);
        return d_magnitude(s2 - s0) * d_magnitude(s2 - s1);

    } else if (primitive_type == 1) { // Triangle

        float3 v0 = make_float3(0, 0, 0);
        d_transformPoint(transform_matrix, v0);
        float3 v1 = make_float3(0, 1, 0);
        d_transformPoint(transform_matrix, v1);
        float3 v2 = make_float3(1, 1, 0);
        d_transformPoint(transform_matrix, v2);
        float3 A(v1 - v0);
        float3 B(v2 - v0);
        float3 C(v2 - v1);
        float a = d_magnitude(A);
        float b = d_magnitude(B);
        float c = d_magnitude(C);
        float s = 0.5f * (a + b + c);
        return sqrtf(s * (s - a) * (s - b) * (s - c));

    } else if (primitive_type == 2) { // Disk

        float r0 = transform_matrix[0];
        float r1 = transform_matrix[5];
        return M_PI * r0 * r1;
    }

    return 0;
}

__device__ void d_matMult33(float *ml, float *mr, float *m) {

    for (int i = 0; i < 9; i++) {
        m[i] = 0.f;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                m[i * 3 + j] = m[i * 3 + j] + ml[i * 3 + k] * mr[k * 3 + j];
            }
        }
    }
}

__device__ void d_matMult44(float *ml, float *mr, float *m) {

    for (int i = 0; i < 16; i++) {
        m[i] = 0.f;
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                m[i * 4 + j] = m[i * 4 + j] + ml[i * 4 + k] * mr[k * 4 + j];
            }
        }
    }
}

__device__ void d_sampleDisk(uint &seed, optix::float3 &sample) {

    // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

    optix::float3 sp;

    float Rx = rnd(seed);
    float Ry = rnd(seed);

    // first map sample point to rectangle [-1,1] [-1,1]
    sp.x = -1.f + 2.f * Rx;
    sp.y = -1.f + 2.f * Ry;

    float r, p;

    if (sp.x > -sp.y) {
        if (sp.x > sp.y) {
            r = sp.x;
            p = sp.y / sp.x;
        } else {
            r = sp.y;
            p = 2.f - sp.x / sp.y;
        }
    } else {
        if (sp.x < sp.y) {
            r = -sp.x;
            p = 4.f + sp.y / sp.x;
        } else {
            r = -sp.y;
            if (sp.y != 0.f) { // avoid division by zero at origin
                p = 6.f - sp.x / sp.y;
            } else {
                p = 0.f;
            }
        }
    }
    p *= 0.25f * M_PI;

    // find x,y point on unit disk
    sp.x = r * cos(p);
    sp.y = r * sin(p);
    sp.z = 0.f;

    sample = sp;
}

__device__ void d_sampleSquare(uint &seed, optix::float3 &sample) {

    optix::float3 sp;

    float Rx = rnd(seed);
    float Ry = rnd(seed);

    sp.x = -0.5f + Rx;
    sp.y = -0.5f + Ry;
    sp.z = 0.f;

    sample = sp;
}

__device__ bool d_sampleTexture_patch(optix::float3 &sp, const optix::int2 &subpatch_index, const optix::float2 &subpatch_size, PerRayData &prd, int mask_ID, int uv_ID) {

    int2 sz = masksize[mask_ID];
    uint3 ind;

    bool solid = false;
    int count = 0;
    while (!solid) {
        count++;

        float2 uv = make_float2((sp.x + 0.5f), (sp.y + 0.5f));
        if (uv_ID == -1) { // does not have custom (u,v) coordinates
            ind = make_uint3(floorf(float(sz.x - 1) * uv.x), floorf(float(sz.y - 1) * (1.f - uv.y)), mask_ID);
        } else { // has custom (u,v) coordinates
            float2 uvmin = uvdata[make_uint2(0, uv_ID)];
            float2 duv;
            duv.x = uvdata[make_uint2(1, uv_ID)].x - uvdata[make_uint2(0, uv_ID)].x;
            duv.y = uvdata[make_uint2(2, uv_ID)].y - uvdata[make_uint2(1, uv_ID)].y;
            ind = make_uint3(floorf(float(sz.x - 1) * (uvmin.x + uv.x * duv.x)), floorf(float(sz.y - 1) * (1.f - uvmin.y - uv.y * duv.y)), mask_ID);
        }
        solid = maskdata[ind];
        if (!solid) {
            if (count > 10) {
                break;
            }
            sp.x = -0.5f + (subpatch_index.x + rnd(prd.seed)) * subpatch_size.x;
            sp.y = -0.5f + (subpatch_index.y + rnd(prd.seed)) * subpatch_size.y;
        }
    }

    return solid;
}

__device__ bool d_sampleTexture_triangle(optix::float3 &sp, PerRayData &prd, int mask_ID, int uv_ID) {

    int2 sz = masksize[mask_ID];

    float2 uv0 = uvdata[make_uint2(0, uv_ID)];
    float2 uv1 = uvdata[make_uint2(1, uv_ID)];
    float2 uv2 = uvdata[make_uint2(2, uv_ID)];

    bool solid = false;
    int count = 0;
    while (!solid) {
        count++;

        // Barycentric weights of the second and third vertices of the canonical triangle (0,0,0), (0,1,0), (1,1,0)
        float beta = sp.y - sp.x;
        float gamma = sp.x;

        float2 uv = uv0 + beta * (uv1 - uv0) + gamma * (uv2 - uv0);

        uint3 ind = make_uint3(floorf(float(sz.x - 1) * uv.x), floorf(float(sz.y - 1) * (1.f - uv.y)), mask_ID);

        if (uv.x < 0 || uv.x > 1.f || uv.y < 0 || uv.y > 1.f) {
            solid = true;
        } else {
            solid = maskdata[ind];
        }
        if (!solid) {
            if (count > 10) {
                break;
            }
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);
            if (Rx < Ry) {
                sp.x = Rx;
                sp.y = Ry;
            } else {
                sp.x = Ry;
                sp.y = Rx;
            }
        }
    }

    return solid;
}

static __host__ __device__ __inline__ float acos_safe(float x) {
    if (x < -1.0)
        x = -1.0;
    else if (x > 1.0)
        x = 1.0;
    return acos(x);
}

static __host__ __device__ __inline__ float asin_safe(float x) {
    if (x < -1.0)
        x = -1.0;
    else if (x > 1.0)
        x = 1.0;
    return asin(x);
}

//! One sample of radiation source s seen from the point `origin`, drawn as the direct launch draws it
/**
 * Gives the direction toward the point sampled on the source (the source's direction for a collimated source), the distance to that point (RT_DEFAULT_MAX for a collimated source), and the factor that,
 * times the source's flux and |n . direction| for a surface of normal n, gives the irradiance the sample stands for: the strength of a direct ray before its division by the number of rays. Cameras draw
 * their shadow rays with it too, so that the shadows they evaluate per pixel come from the same distribution of points on the source as the radiation solve, including the penumbra of the sun's disk and
 * of a finite source. Returns false for a sample on the back of a rectangle or disk source, which delivers nothing, and for an unknown source type.
 */
__device__ bool d_sampleSourceDirection(uint s, const optix::float3 &origin, uint &seed, optix::float3 &direction, float &tmax, double &factor) {
    const uint source_type = source_types[s];
    if (source_type == 0) { // collimated source
        direction = optix::normalize(source_positions[s]);
        tmax = RT_DEFAULT_MAX;
        factor = 1.0;
        return true;
    }
    if (source_type == 1 || source_type == 2) { // sphere source
        // sample point on surface of sphere
        const float theta_s = acos_safe(1.f - 2.f * rnd(seed));
        const float phi_s = rnd(seed) * 2.f * M_PI;
        const optix::float3 sphere_point = 0.5 * source_widths[s].x * optix::make_float3(sin(theta_s) * cos(phi_s), sin(theta_s) * sin(phi_s), cos(theta_s));
        direction = sphere_point + source_positions[s] - origin;
        tmax = d_magnitude(direction);
        direction = optix::normalize(direction);
        factor = 0.0;
        const uint N = 10;
        for (uint j = 0; j < N; j++) {
            for (uint i = 0; i < N; i++) {
                const float theta = acos_safe(1.f - 2.f * (float(i) + 0.5f) / float(N));
                const float phi = (float(j) + 0.5f) * 2.f * M_PI / float(N);
                const optix::float3 light_direction = optix::make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
                if (optix::dot(light_direction, direction) < 0) {
                    factor += fabs(optix::dot(light_direction, direction)) / (double(tmax) * double(tmax)) / double(N * N) * source_widths[s].x * source_widths[s].x;
                }
            }
        }
        return true;
    }
    if (source_type == 3) { // rectangle source
        float light_transform[16];
        d_makeTransformMatrix(source_rotations[s], light_transform);
        optix::float3 square_point;
        d_sampleSquare(seed, square_point);
        square_point = optix::make_float3(source_widths[s].x * square_point.x, source_widths[s].y * square_point.y, square_point.z);
        d_transformPoint(light_transform, square_point);
        optix::float3 light_direction = optix::make_float3(0, 0, 1);
        d_transformPoint(light_transform, light_direction);
        direction = square_point + source_positions[s] - origin;
        if (optix::dot(direction, light_direction) > 0.f) { // the ray goes toward the source, so the dot product is negative when the source faces the point
            return false;
        }
        tmax = d_magnitude(direction);
        direction = optix::normalize(direction);
        factor = fabs(optix::dot(light_direction, direction)) / (double(tmax) * double(tmax)) * source_widths[s].x * source_widths[s].y / M_PI;
        return true;
    }
    if (source_type == 4) { // disk source
        float light_transform[16];
        d_makeTransformMatrix(source_rotations[s], light_transform);
        optix::float3 disk_point;
        d_sampleDisk(seed, disk_point);
        d_transformPoint(light_transform, disk_point);
        optix::float3 light_direction = optix::make_float3(0, 0, 1);
        d_transformPoint(light_transform, light_direction);
        direction = source_widths[s].x * disk_point + source_positions[s] - origin;
        if (optix::dot(direction, light_direction) > 0.f) {
            return false;
        }
        tmax = d_magnitude(direction);
        direction = optix::normalize(direction);
        factor = fabs(optix::dot(light_direction, direction)) / (double(tmax) * double(tmax)) * source_widths[s].x * source_widths[s].x;
        return true;
    }
    return false; // unknown source type
}

// ----- auxiliary helpers --------------------------------------
static __forceinline__ __device__ uint32_t reverseBits32(uint32_t v) {
    // bitwise reverse (Van-der-Corput direction numbers for dim0)
    v = (v << 16) | (v >> 16);
    v = ((v & 0x00ff00ffU) << 8) | ((v & 0xff00ff00U) >> 8);
    v = ((v & 0x0f0f0f0fU) << 4) | ((v & 0xf0f0f0f0U) >> 4);
    v = ((v & 0x33333333U) << 2) | ((v & 0xccccccccU) >> 2);
    v = ((v & 0x55555555U) << 1) | ((v & 0xaaaaaaaaU) >> 1);
    return v;
}

// Simple PCG-style 32-bit hash (acts as an Owen scramble)
static __forceinline__ __device__ uint32_t pcgHash(uint32_t v) {
    v ^= v >> 17;
    v *= 0xed5ad4bbU;
    v ^= v >> 11;
    v *= 0xac4c1b51U;
    v ^= v >> 15;
    v *= 0x31848babU;
    v ^= v >> 14;
    return v;
}

// Convert uint32→float in [0,1)
static __forceinline__ __device__ float uint32ToUnitFloat(uint32_t v) {
    return v * 2.3283064365386963e-10f; // 1 / 2^32
}

// Sobol dimension-0 (Van der Corput) with Owen scramble
static __forceinline__ __device__ float sobolDim0(uint32_t idx, uint32_t scramble) {
    return uint32ToUnitFloat(reverseBits32(idx) ^ scramble);
}

// Sobol dimension-1 (direction numbers from poly x^2+x+1)
static __forceinline__ __device__ float sobolDim1(uint32_t idx, uint32_t scramble) {
    uint32_t v = 1u << 31; // MSB first
    uint32_t res = scramble;
    while (idx) {
        if (idx & 1)
            res ^= v;
        idx >>= 1;
        v ^= v >> 1; // Gray-code update
    }
    return uint32ToUnitFloat(res);
}

// ----- public entry point -------------------------------------
static __forceinline__ __device__ float2 sobol2D(uint32_t sampleIdx) {
    // Independent Owen scrambles for each dimension
    uint32_t scramble0 = pcgHash(sampleIdx);
    uint32_t scramble1 = pcgHash(sampleIdx ^ 0x9e3779b9U); // golden-ratio XOR

    float x = sobolDim0(sampleIdx, scramble0); // dimension 0
    float y = sobolDim1(sampleIdx, scramble1); // dimension 1
    return make_float2(x, y);
}

// simple LCG → xorshift → float conversion (1 cycle, 4 ops)
__forceinline__ __device__ float uint32_to_unit_float(uint32_t v) {
    return v * 2.3283064365386963e-10f; // 1 / 2^32
}
__forceinline__ __device__ float2 hash2D(uint32_t s) {
    s ^= s * 0x6c50b47cu;
    s ^= s >> 17;
    float u = uint32_to_unit_float(s);

    s ^= s * 0xb82f1e52u;
    s ^= s >> 17;
    float v = uint32_to_unit_float(s);

    return make_float2(u, v);
}

// Ray-source intersection helpers for camera rendering

//! Test if ray intersects a spherical radiation source
/**
 * \param[in] ray_origin Origin point of the ray
 * \param[in] ray_direction Direction vector of the ray (must be normalized)
 * \param[in] sphere_center Center position of the sphere
 * \param[in] sphere_radius Radius of the sphere
 * \return True if ray intersects sphere in front of origin
 */
__device__ __forceinline__ bool d_raySphereIntersect(const optix::float3 &ray_origin, const optix::float3 &ray_direction, const optix::float3 &sphere_center, float sphere_radius) {
    optix::float3 oc = ray_origin - sphere_center;
    float b = optix::dot(oc, ray_direction);
    float c = optix::dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - c;

    if (discriminant < 0.0f)
        return false;

    float t = -b - sqrtf(discriminant);
    return t > 0.0f; // Hit if in front of camera
}

//! Test if ray intersects a rectangular radiation source
/**
 * \param[in] ray_origin Origin point of the ray
 * \param[in] ray_direction Direction vector of the ray (must be normalized)
 * \param[in] rect_center Center position of the rectangle
 * \param[in] rect_width Width of the rectangle (x dimension)
 * \param[in] rect_length Length of the rectangle (y dimension)
 * \param[in] rect_rotation Rotation angles (rx, ry, rz) of the rectangle
 * \param[out] out_cos_angle Cosine of angle between ray and rectangle normal
 * \return True if ray intersects front face of rectangle in front of origin
 */
__device__ __forceinline__ bool d_rayRectangleIntersect(const optix::float3 &ray_origin, const optix::float3 &ray_direction, const optix::float3 &rect_center, float rect_width, float rect_length, const optix::float3 &rect_rotation,
                                                        float &out_cos_angle) {
    // Build transform matrix
    float transform[16];
    d_makeTransformMatrix(rect_rotation, transform);

    // Get normal vector (transformed +Z axis)
    optix::float3 normal = optix::make_float3(transform[2], transform[6], transform[10]);

    // Ray-plane intersection
    float denom = optix::dot(ray_direction, normal);
    if (denom >= -1e-6f)
        return false; // Parallel or back-facing

    optix::float3 oc = rect_center - ray_origin;
    float t = optix::dot(oc, normal) / denom;
    if (t <= 0.0f)
        return false; // Behind camera

    // Hit point in world space
    optix::float3 hit_point = ray_origin + t * ray_direction;

    // Transform to local coordinates
    optix::float3 local_hit = hit_point - rect_center;
    float inv_transform[16];
    d_invertMatrix(transform, inv_transform);
    d_transformPoint(inv_transform, local_hit);

    // Point-in-rectangle test
    if (fabsf(local_hit.x) > rect_width * 0.5f)
        return false;
    if (fabsf(local_hit.y) > rect_length * 0.5f)
        return false;

    out_cos_angle = -denom; // Cosine of angle between ray and normal
    return true;
}

//! Test if ray intersects a disk radiation source
/**
 * \param[in] ray_origin Origin point of the ray
 * \param[in] ray_direction Direction vector of the ray (must be normalized)
 * \param[in] disk_center Center position of the disk
 * \param[in] disk_radius Radius of the disk
 * \param[in] disk_rotation Rotation angles (rx, ry, rz) of the disk
 * \param[out] out_cos_angle Cosine of angle between ray and disk normal
 * \return True if ray intersects front face of disk in front of origin
 */
__device__ __forceinline__ bool d_rayDiskIntersect(const optix::float3 &ray_origin, const optix::float3 &ray_direction, const optix::float3 &disk_center, float disk_radius, const optix::float3 &disk_rotation, float &out_cos_angle) {
    // Build transform matrix
    float transform[16];
    d_makeTransformMatrix(disk_rotation, transform);

    // Get normal vector (transformed +Z axis)
    optix::float3 normal = optix::make_float3(transform[2], transform[6], transform[10]);

    // Ray-plane intersection
    float denom = optix::dot(ray_direction, normal);
    if (denom >= -1e-6f)
        return false; // Parallel or back-facing

    optix::float3 oc = disk_center - ray_origin;
    float t = optix::dot(oc, normal) / denom;
    if (t <= 0.0f)
        return false; // Behind camera

    // Hit point in world space
    optix::float3 hit_point = ray_origin + t * ray_direction;

    // Point-in-disk test (distance from center)
    optix::float3 offset = hit_point - disk_center;
    float dist_sq = optix::dot(offset, offset);
    if (dist_sq > disk_radius * disk_radius)
        return false;

    out_cos_angle = -denom; // Cosine of angle between ray and normal
    return true;
}


// ----- Periodic boundary -----

// Move a coordinate into [lower, upper) by a whole number of periods. A point on the lower face of a ray travelling out through
// that face is moved to the upper face instead, so the ray starts inside the domain. Must match wrap_into_period() in the Vulkan
// shaders.
static __device__ __inline__ float wrapIntoPeriod(float coordinate, float direction, float lower, float upper) {
    const float period = upper - lower;
    float wrapped = coordinate - period * floorf((coordinate - lower) / period);
    if (wrapped < lower || wrapped >= upper) {
        wrapped = lower; // rounding at a whole number of periods
    }
    if (wrapped == lower && direction < 0.f) {
        wrapped = upper;
    }
    return wrapped;
}

// Low and high corners of the periodic domain in x and y. bbox_vertices is indexed (vertex, face): vertex 0 of face 0 and vertex 1
// of face 1 are at the low and high corners.
static __device__ __inline__ optix::float2 periodicXBounds() {
    return optix::make_float2(bbox_vertices[optix::make_uint2(0, 0)].x, bbox_vertices[optix::make_uint2(1, 1)].x);
}
static __device__ __inline__ optix::float2 periodicYBounds() {
    return optix::make_float2(bbox_vertices[optix::make_uint2(0, 0)].y, bbox_vertices[optix::make_uint2(1, 1)].y);
}

// Whether a primitive is absent from the periodic copy of the domain a ray is in: a primitive excluded from the periodic boundary
// exists only in the central copy, and nothing exists beyond a limited number of copies. Must match periodicHidden() in the OptiX 8
// device code and periodic_hidden() in the Vulkan shaders.
static __device__ __inline__ bool periodicHidden(const PerRayData &ray_data, uint position) {
    if (periodic_tile_tracking == 0u) {
        return false;
    }
    if ((periodic_max_repeats.x >= 0 && abs(ray_data.tile_x) > periodic_max_repeats.x) || (periodic_max_repeats.y >= 0 && abs(ray_data.tile_y) > periodic_max_repeats.y)) {
        return true; // outside the field of copies
    }
    return periodic_excluded_present != 0u && periodic_excluded[position] != 0 && (ray_data.tile_x != 0 || ray_data.tile_y != 0);
}

// A ray hit a periodic boundary face at distance face_t along it: set where it continues (periodic_hit, at periodic_hit_t from its
// origin), wrapped to the opposite face. When rays track the periodic copy they are in, this also updates the copy, moves a ray that
// is above (below) all repeated geometry and heading down (up), and can no longer return to the central copy, down (up) to it in one
// step, and lets a ray that can hit nothing more continue from just beyond the face it left through, where it hits no face again and
// leaves the scene as a ray that missed everything. Must match handlePeriodicBoundaryHit() in the OptiX 8 device code and
// check_periodic_boundary() in the Vulkan shaders.
static __device__ __inline__ void d_handlePeriodicBoundaryHit(PerRayData &ray_data, const optix::float3 &ray_origin, const optix::float3 &ray_direction, float face_t) {
    ray_data.hit_periodic_boundary = true;
    ray_data.periodic_hit_t = face_t;
    const optix::float3 hit_pos = ray_origin + face_t * ray_direction;

    // Tolerance for identifying the face that was hit. The ray continues from exactly one period away, the same point on the
    // opposite face: moving it any further would shift it off its straight path by that much at every wrap.
    const float eps = 1e-5f;
    const optix::float2 xbounds = periodicXBounds();
    const optix::float2 ybounds = periodicYBounds();
    const float width_x = xbounds.y - xbounds.x;
    const float width_y = ybounds.y - ybounds.x;

    // Leaving through a low face moves the ray into the next periodic copy toward -x (or -y), and through a high face toward +x (or +y)
    int step_x = 0;
    int step_y = 0;
    ray_data.periodic_hit = hit_pos;
    if (periodic_flag.x == 1 && fabs(hit_pos.x - xbounds.x) <= eps) { //-x facing boundary
        ray_data.periodic_hit.x += width_x;
        step_x = -1;
    } else if (periodic_flag.x == 1 && fabs(hit_pos.x - xbounds.y) <= eps) { //+x facing boundary
        ray_data.periodic_hit.x -= width_x;
        step_x = 1;
    } else if (periodic_flag.y == 1 && fabs(hit_pos.y - ybounds.x) <= eps) { //-y facing boundary
        ray_data.periodic_hit.y += width_y;
        step_y = -1;
    } else if (periodic_flag.y == 1 && fabs(hit_pos.y - ybounds.y) <= eps) { //+y facing boundary
        ray_data.periodic_hit.y -= width_y;
        step_y = 1;
    }

    int tile_x = ray_data.tile_x + step_x;
    int tile_y = ray_data.tile_y + step_y;
    if (periodic_tile_tracking == 0u) {
        // Only the copy counts, for the ray to hit the periodic copies of the primitive it was launched from
        ray_data.tile_x = tile_x;
        ray_data.tile_y = tile_y;
        return;
    }

    // Whether the ray leaves the field of copies; it can then hit nothing more
    bool leaves_field = (periodic_max_repeats.x >= 0 && abs(tile_x) > periodic_max_repeats.x && float(tile_x) * ray_direction.x > 0.f) || (periodic_max_repeats.y >= 0 && abs(tile_y) > periodic_max_repeats.y && float(tile_y) * ray_direction.y > 0.f);
    // Outside the central copy, which it can no longer return to, only repeated geometry can be hit
    const bool leaves_central_copy = (tile_x != 0 && float(tile_x) * ray_direction.x >= 0.f) || (tile_y != 0 && float(tile_y) * ray_direction.y >= 0.f);
    const optix::float3 wrapped = ray_data.periodic_hit;
    bool escapes = leaves_field || (leaves_central_copy && ((wrapped.z > periodic_repeated_zbounds.y && ray_direction.z >= 0.f) || (wrapped.z < periodic_repeated_zbounds.x && ray_direction.z <= 0.f)));
    if (!escapes && leaves_central_copy && (wrapped.z > periodic_repeated_zbounds.y || wrapped.z < periodic_repeated_zbounds.x)) {
        // Above (below) the repeated geometry and heading down (up): nothing can be hit before the ray reaches it, so the ray is moved
        // there in one step, into the domain by a whole number of periods, rather than wrapped through every copy on the way
        const float target_z = wrapped.z > periodic_repeated_zbounds.y ? periodic_repeated_zbounds.y : periodic_repeated_zbounds.x;
        const float advance = (target_z - wrapped.z) / ray_direction.z;
        optix::float3 moved = optix::make_float3(wrapped.x + advance * ray_direction.x, wrapped.y + advance * ray_direction.y, target_z);
        if (periodic_flag.x == 1) {
            const float unwrapped = moved.x;
            moved.x = wrapIntoPeriod(moved.x, ray_direction.x, xbounds.x, xbounds.y);
            tile_x += __float2int_rn((unwrapped - moved.x) / width_x);
        }
        if (periodic_flag.y == 1) {
            const float unwrapped = moved.y;
            moved.y = wrapIntoPeriod(moved.y, ray_direction.y, ybounds.x, ybounds.y);
            tile_y += __float2int_rn((unwrapped - moved.y) / width_y);
        }
        escapes = (periodic_max_repeats.x >= 0 && abs(tile_x) > periodic_max_repeats.x && float(tile_x) * ray_direction.x > 0.f) || (periodic_max_repeats.y >= 0 && abs(tile_y) > periodic_max_repeats.y && float(tile_y) * ray_direction.y > 0.f);
        if (!escapes) {
            ray_data.periodic_hit = moved;
            ray_data.periodic_hit_t = face_t + advance; // the distance to the point the ray continues from
        }
    }
    if (escapes) {
        ray_data.periodic_hit = hit_pos;
        ray_data.periodic_hit_t = face_t;
        ray_data.periodic_hit.x += float(step_x) * periodic_escape_offset;
        ray_data.periodic_hit.y += float(step_y) * periodic_escape_offset;
        return;
    }
    ray_data.tile_x = tile_x;
    ray_data.tile_y = tile_y;
}

#endif
