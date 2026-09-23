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

// Index into [band * pixel_y * pixel_x] buffer (row-major pixels)
uint index_pixel(uint band, uint pixel_x, uint pixel_y, uint resolution_x, uint resolution_y) {
    return band * (resolution_y * resolution_x) + pixel_y * resolution_x + pixel_x;
}

#endif // BUFFER_INDEXING_GLSL
