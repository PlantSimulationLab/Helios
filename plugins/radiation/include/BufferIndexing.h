/** \file "BufferIndexing.h" Type-safe buffer indexing utilities for multi-dimensional arrays.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_BUFFER_INDEXING_H
#define HELIOS_BUFFER_INDEXING_H

#include <cstddef>

/**
 * @brief Type-safe buffer indexing utilities for multi-dimensional arrays
 *
 * This header provides zero-overhead indexing abstractions that eliminate manual
 * index calculations and prevent indexing errors. All indexers use row-major order
 * (C-style) where the last dimension varies fastest.
 *
 * Example usage:
 * @code
 * // Instead of: size_t ind = prim * Nbands + band;
 * BufferIndexer2D indexer(Nprimitives, Nbands);
 * size_t ind = indexer(prim, band);
 *
 * // 3D example:
 * MaterialPropertyIndexer indexer3d(Nsources, Nprimitives, Nbands);
 * size_t ind = indexer3d(source, prim, band);
 * @endcode
 *
 * Benefits:
 * - Eliminates manual index calculations
 * - Self-documenting code (named indexer types)
 * - Compile-time argument count checking
 * - Zero runtime overhead (inline functions)
 * - Works in both CPU and GPU code
 */

// Support both CUDA and CPU compilation
#ifdef __CUDACC__
#define HELIOS_HOST_DEVICE __host__ __device__
#else
#define HELIOS_HOST_DEVICE
#endif

/**
 * @brief 2D buffer indexer: [dim0][dim1]
 *
 * Computes linear index as: dim0_idx * dim1_size + dim1_idx
 *
 * Common usage patterns:
 * - Radiation buffers: [primitive][band]
 * - Source fluxes: [source][band]
 * - Camera pixels: [row][col]
 */
class BufferIndexer2D {
private:
    size_t dim1_size;

public:
    /**
     * @brief Construct 2D indexer
     * @param dim0_size Size of first dimension (stored for documentation/future validation)
     * @param dim1_size Size of second dimension
     */
    HELIOS_HOST_DEVICE
    explicit BufferIndexer2D(size_t dim0_size, size_t dim1_size)
        : dim1_size(dim1_size) {
        // dim0_size parameter exists for self-documentation and future validation
        (void)dim0_size;
    }

    /**
     * @brief Compute linear index for 2D access
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @return Linear array index
     */
    HELIOS_HOST_DEVICE inline
    size_t operator()(size_t i, size_t j) const {
        return i * dim1_size + j;
    }

    /** @brief Get size of second dimension */
    HELIOS_HOST_DEVICE inline
    size_t getDim1Size() const { return dim1_size; }
};

/**
 * @brief 3D buffer indexer: [dim0][dim1][dim2]
 *
 * Computes linear index as: dim0_idx * (dim1_size * dim2_size) + dim1_idx * dim2_size + dim2_idx
 *
 * Common usage patterns:
 * - Material properties: [source][primitive][band]
 * - Source camera fluxes: [source][band][camera]
 */
class BufferIndexer3D {
private:
    size_t dim1_size;
    size_t dim2_size;
    size_t dim1_dim2_product;  // Precomputed for efficiency

public:
    /**
     * @brief Construct 3D indexer
     * @param dim0_size Size of first dimension (stored for documentation/future validation)
     * @param dim1_size Size of second dimension
     * @param dim2_size Size of third dimension
     */
    HELIOS_HOST_DEVICE
    explicit BufferIndexer3D(size_t dim0_size, size_t dim1_size, size_t dim2_size)
        : dim1_size(dim1_size)
        , dim2_size(dim2_size)
        , dim1_dim2_product(dim1_size * dim2_size) {
        (void)dim0_size;
    }

    /**
     * @brief Compute linear index for 3D access
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @param k Index in third dimension
     * @return Linear array index
     */
    HELIOS_HOST_DEVICE inline
    size_t operator()(size_t i, size_t j, size_t k) const {
        return i * dim1_dim2_product + j * dim2_size + k;
    }

    /** @brief Get size of second dimension */
    HELIOS_HOST_DEVICE inline
    size_t getDim1Size() const { return dim1_size; }

    /** @brief Get size of third dimension */
    HELIOS_HOST_DEVICE inline
    size_t getDim2Size() const { return dim2_size; }
};

/**
 * @brief 4D buffer indexer: [dim0][dim1][dim2][dim3]
 *
 * Computes linear index as:
 *   dim0_idx * (dim1 * dim2 * dim3) + dim1_idx * (dim2 * dim3) + dim2_idx * dim3 + dim3_idx
 *
 * Common usage patterns:
 * - Camera material properties: [source][primitive][band][camera]
 * - Specular radiation: [source][camera][primitive][band]
 */
class BufferIndexer4D {
private:
    size_t dim1_size;
    size_t dim2_size;
    size_t dim3_size;
    size_t dim1_dim2_dim3_product;  // dim1 * dim2 * dim3
    size_t dim2_dim3_product;        // dim2 * dim3

public:
    /**
     * @brief Construct 4D indexer
     * @param dim0_size Size of first dimension (stored for documentation/future validation)
     * @param dim1_size Size of second dimension
     * @param dim2_size Size of third dimension
     * @param dim3_size Size of fourth dimension
     */
    HELIOS_HOST_DEVICE
    explicit BufferIndexer4D(size_t dim0_size, size_t dim1_size, size_t dim2_size, size_t dim3_size)
        : dim1_size(dim1_size)
        , dim2_size(dim2_size)
        , dim3_size(dim3_size)
        , dim1_dim2_dim3_product(dim1_size * dim2_size * dim3_size)
        , dim2_dim3_product(dim2_size * dim3_size) {
        (void)dim0_size;
    }

    /**
     * @brief Compute linear index for 4D access
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @param k Index in third dimension
     * @param l Index in fourth dimension
     * @return Linear array index
     */
    HELIOS_HOST_DEVICE inline
    size_t operator()(size_t i, size_t j, size_t k, size_t l) const {
        return i * dim1_dim2_dim3_product + j * dim2_dim3_product + k * dim3_size + l;
    }

    /** @brief Get size of second dimension */
    HELIOS_HOST_DEVICE inline
    size_t getDim1Size() const { return dim1_size; }

    /** @brief Get size of third dimension */
    HELIOS_HOST_DEVICE inline
    size_t getDim2Size() const { return dim2_size; }

    /** @brief Get size of fourth dimension */
    HELIOS_HOST_DEVICE inline
    size_t getDim3Size() const { return dim3_size; }
};

// ========== Named Type Aliases for Self-Documenting Code ==========

/**
 * @brief Indexer for radiation buffers: [primitive][band]
 *
 * Used for:
 * - radiation_in, radiation_out_top, radiation_out_bottom
 * - scatter_buff_top, scatter_buff_bottom
 * - sky_energy
 */
using RadiationBufferIndexer = BufferIndexer2D;

/**
 * @brief Indexer for source flux buffers: [source][band]
 *
 * Used for:
 * - source_fluxes (direct radiation)
 */
using SourceFluxIndexer = BufferIndexer2D;

/**
 * @brief Indexer for material property buffers: [source][primitive][band]
 *
 * Used for:
 * - rho (reflectivity)
 * - tau (transmissivity)
 *
 * Indexing: material_indexer(source_ID, primitive_position, band_global)
 */
using MaterialPropertyIndexer = BufferIndexer3D;

/**
 * @brief Indexer for source-camera flux buffers: [source][band][camera]
 *
 * Used for:
 * - source_fluxes_cam (camera-weighted source fluxes)
 */
using SourceCameraFluxIndexer = BufferIndexer3D;

/**
 * @brief Indexer for camera material property buffers: [source][primitive][band][camera]
 *
 * Used for:
 * - rho_cam (camera-weighted reflectivity)
 * - tau_cam (camera-weighted transmissivity)
 */
using CameraMaterialIndexer = BufferIndexer4D;

/**
 * @brief Indexer for specular radiation buffers: [source][camera][primitive][band]
 *
 * Note: Different dimension order than camera materials!
 *
 * Used for:
 * - radiation_specular (specular reflection per source and camera)
 */
using SpecularRadiationIndexer = BufferIndexer4D;

#endif // HELIOS_BUFFER_INDEXING_H
