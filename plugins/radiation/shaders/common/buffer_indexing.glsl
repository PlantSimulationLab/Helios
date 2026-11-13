/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

/** buffer_indexing.glsl - Multi-dimensional buffer indexing helpers
 *
 * Matches BufferIndexing.h CPU-side indexing.
 */

#ifndef BUFFER_INDEXING_GLSL
#define BUFFER_INDEXING_GLSL

// Index into [Nprims][Nbands] buffer (primitive-major, matches CPU BufferIndexer2D)
// CPU formula: primitive * Nbands + band
uint index_prim_band(uint prim, uint band, uint Nbands) {
    return prim * Nbands + band;
}

// Index into [Nsources][Nprims][Nbands] buffer (source-major, primitive-major, band-minor)
// Matches CPU MaterialPropertyIndexer: source * (Nprims * Nbands) + prim * Nbands + band
uint index_source_band_prim(uint source, uint band, uint prim, uint Nbands, uint Nprims) {
    return source * (Nprims * Nbands) + prim * Nbands + band;
}

// Index into [Ncameras * Nbands * Nprims] buffer (camera-major)
uint index_camera_band_prim(uint camera, uint band, uint prim, uint Nbands, uint Nprims) {
    return camera * (Nbands * Nprims) + band * Nprims + prim;
}

// ---------------------------------------------------------------------------
// Background radiation field (see RayTracingGeometry::periodicDualField()). With primitives excluded from the periodic boundary, the buffers of outgoing and scattered radiation, of what cameras read,
// and of the direct flux hold two fields: field 0, the scene as it is, in flux slots [0, primitive_count), and field 1, the background scene without the excluded primitives, in flux slots
// [primitive_count, 2 primitive_count). A ray or camera reads field 0 where it hits the central copy of the domain and field 1 in every other copy. Material and geometry buffers are indexed by the
// primitive's position alone. The including shader must declare periodic_info_buf with the field layout words (see VulkanComputeBackend::periodic_info) and pc.primitive_count and pc.band_count.
// Must match fluxSlot() in OptiX8DeviceCode.cu and d_fluxSlot() in RayTracing.cuh.
// ---------------------------------------------------------------------------

// Whether a background field is solved alongside the scene as it is
bool dual_field() {
    return periodic_info_buf.flux_field_count == 2u;
}

// Flux slot of primitive prim in a field
uint flux_slot(uint prim, uint field) {
    return prim + field * pc.primitive_count;
}

// Values in one [flux slot][launch band] block of the outgoing, scattered, camera and spectral buffers
uint flux_slot_stride() {
    return periodic_info_buf.flux_slot_count * pc.band_count;
}

// Index of the direct flux of source s reaching primitive prim, in a field: radiation_specular is [source][flux slot][launch band]
uint direct_flux_index(uint s, uint prim, uint field, uint band) {
    return (s * periodic_info_buf.flux_slot_count + flux_slot(prim, field)) * pc.band_count + band;
}

// The field a ray reads in the periodic copy `tile` of the domain: the scene as it is in the central copy, and the background in every other copy. Must match fieldSeenBy() in OptiX8DeviceCode.cu.
uint field_seen_in_tile(ivec2 tile) {
    return (dual_field() && tile != ivec2(0)) ? 1u : 0u;
}

// Index into [band * pixel_y * pixel_x] buffer (row-major pixels)
uint index_pixel(uint band, uint pixel_x, uint pixel_y, uint resolution_x, uint resolution_y) {
    return band * (resolution_y * resolution_x) + pixel_y * resolution_x + pixel_x;
}

#endif // BUFFER_INDEXING_GLSL
