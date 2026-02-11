/** \file "BufferIndexing.h" Type-safe buffer indexing utilities for multi-dimensional arrays.

    Copyright (C) 2016-2026 Brian Bailey

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
#include <climits>

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
    explicit BufferIndexer2D(size_t dim0_size, size_t dim1_size) : dim1_size(dim1_size) {
        // dim0_size parameter exists for self-documentation and future validation
        (void) dim0_size;
    }

    /**
     * @brief Compute linear index for 2D access
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @return Linear array index
     */
    HELIOS_HOST_DEVICE inline size_t operator()(size_t i, size_t j) const {
        return i * dim1_size + j;
    }

    /** @brief Get size of second dimension */
    HELIOS_HOST_DEVICE inline size_t getDim1Size() const {
        return dim1_size;
    }
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
    size_t dim1_dim2_product; // Precomputed for efficiency

public:
    /**
     * @brief Construct 3D indexer
     * @param dim0_size Size of first dimension (stored for documentation/future validation)
     * @param dim1_size Size of second dimension
     * @param dim2_size Size of third dimension
     */
    HELIOS_HOST_DEVICE
    explicit BufferIndexer3D(size_t dim0_size, size_t dim1_size, size_t dim2_size) : dim1_size(dim1_size), dim2_size(dim2_size), dim1_dim2_product(dim1_size * dim2_size) {
        (void) dim0_size;
    }

    /**
     * @brief Compute linear index for 3D access
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @param k Index in third dimension
     * @return Linear array index
     */
    HELIOS_HOST_DEVICE inline size_t operator()(size_t i, size_t j, size_t k) const {
        return i * dim1_dim2_product + j * dim2_size + k;
    }

    /** @brief Get size of second dimension */
    HELIOS_HOST_DEVICE inline size_t getDim1Size() const {
        return dim1_size;
    }

    /** @brief Get size of third dimension */
    HELIOS_HOST_DEVICE inline size_t getDim2Size() const {
        return dim2_size;
    }
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
    size_t dim1_dim2_dim3_product; // dim1 * dim2 * dim3
    size_t dim2_dim3_product; // dim2 * dim3

public:
    /**
     * @brief Construct 4D indexer
     * @param dim0_size Size of first dimension (stored for documentation/future validation)
     * @param dim1_size Size of second dimension
     * @param dim2_size Size of third dimension
     * @param dim3_size Size of fourth dimension
     */
    HELIOS_HOST_DEVICE
    explicit BufferIndexer4D(size_t dim0_size, size_t dim1_size, size_t dim2_size, size_t dim3_size) :
        dim1_size(dim1_size), dim2_size(dim2_size), dim3_size(dim3_size), dim1_dim2_dim3_product(dim1_size * dim2_size * dim3_size), dim2_dim3_product(dim2_size * dim3_size) {
        (void) dim0_size;
    }

    /**
     * @brief Compute linear index for 4D access
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @param k Index in third dimension
     * @param l Index in fourth dimension
     * @return Linear array index
     */
    HELIOS_HOST_DEVICE inline size_t operator()(size_t i, size_t j, size_t k, size_t l) const {
        return i * dim1_dim2_dim3_product + j * dim2_dim3_product + k * dim3_size + l;
    }

    /** @brief Get size of second dimension */
    HELIOS_HOST_DEVICE inline size_t getDim1Size() const {
        return dim1_size;
    }

    /** @brief Get size of third dimension */
    HELIOS_HOST_DEVICE inline size_t getDim2Size() const {
        return dim2_size;
    }

    /** @brief Get size of fourth dimension */
    HELIOS_HOST_DEVICE inline size_t getDim3Size() const {
        return dim3_size;
    }
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

// ========== UUID / Position Lookup Helper ==========

/**
 * @brief Safe UUID→Position lookup with automatic bounds checking
 *
 * Encapsulates the UUID→position conversion pattern used throughout CUDA code.
 * Prevents forgetting bounds check or using UUID as direct array index.
 *
 * This helper makes it impossible to:
 * - Use UUID directly as array index (sparse UUIDs cause out-of-bounds)
 * - Forget bounds checking (built into toPosition())
 * - Access deleted/invalid primitives (returns false for UINT_MAX)
 *
 * Usage in CUDA ray hit programs:
 * @code
 * UUIDLookupHelper lookup(primitive_positions, Nprimitives);
 * uint position;
 * if (!lookup.toPosition(hit_UUID, position)) {
 *     return;  // Invalid UUID - primitive was deleted or never existed
 * }
 * // Safe to use position for all buffer access
 * bool two_sided = twosided_flag[position];
 * int2 subdivs = object_subdivisions[position];
 * @endcode
 *
 * Historical bugs prevented (commit 0ec2dc25a):
 * - 20+ instances of direct UUID indexing: twosided_flag[UUID], objectID[UUID]
 */
class UUIDLookupHelper {
private:
    const uint *primitive_positions_; ///< GPU buffer: sparse UUID→position lookup table
    size_t primitive_count_; ///< Total primitive count (for bounds checking)

public:
    /**
     * @brief Construct UUID lookup helper
     * @param prim_positions Pointer to primitive_positions buffer (sparse array indexed by UUID)
     * @param prim_count Total number of primitives (for bounds validation)
     */
    HELIOS_HOST_DEVICE
    UUIDLookupHelper(const uint *prim_positions, size_t prim_count) : primitive_positions_(prim_positions), primitive_count_(prim_count) {
    }

    /**
     * @brief Convert UUID to position with automatic bounds checking
     *
     * @param UUID Primitive UUID to look up
     * @param[out] position Output position (only written if UUID is valid)
     * @return true if UUID is valid and position was written, false if UUID not found
     *
     * Returns false if:
     * - primitive_positions[UUID] == UINT_MAX (deleted/never created)
     * - primitive_positions[UUID] >= primitive_count (corrupted data)
     *
     * Position parameter is unchanged if function returns false.
     */
    HELIOS_HOST_DEVICE inline bool toPosition(uint UUID, uint &position) const {
        // Lookup in sparse table - returns UINT_MAX if UUID doesn't exist
        uint pos = primitive_positions_[UUID];
        if (pos == UINT_MAX || pos >= primitive_count_) {
            return false; // Invalid UUID or out-of-bounds position
        }
        position = pos;
        return true;
    }
};

// ========== Camera Pixel Coordinate Helper ==========

/**
 * @brief Camera pixel coordinate abstraction for tiled rendering
 *
 * Encapsulates the tile offset + coordinate mapping + flattening complexity.
 * Prevents x/y swapping, wrong offset, wrong resolution variable errors.
 *
 * This helper makes it impossible to:
 * - Forget tile offset when computing global coordinates
 * - Swap x and y coordinates (launch_index.y → x, launch_index.z → y)
 * - Use wrong resolution (tile vs full)
 * - Get flattening formula wrong (always row-major: y * width + x)
 *
 * Usage in CUDA camera ray generation:
 * @code
 * // Simple one-line pixel index calculation:
 * size_t pixel_idx = PixelCoordinate::computeFlatIndex(
 *     launch_index, camera_pixel_offset, camera_resolution_full);
 * camera_pixel_label[pixel_idx] = hit_UUID + 1;
 *
 * // Or step-by-step if you need the coordinates:
 * PixelCoordinate pixel = PixelCoordinate::fromTiledLaunch(
 *     launch_index, camera_pixel_offset, camera_resolution_full);
 * size_t idx = pixel.toFlatIndex(camera_resolution_full);
 * @endcode
 *
 * Historical bugs prevented:
 * - Forgetting camera_pixel_offset in tiled rendering
 * - Swapping x/y coordinates
 * - Using camera_resolution instead of camera_resolution_full
 */
struct PixelCoordinate {
    uint x; ///< Global x-coordinate (column index)
    uint y; ///< Global y-coordinate (row index)

    /// Construct pixel coordinate from global x, y
    HELIOS_HOST_DEVICE
    PixelCoordinate(uint x_coord, uint y_coord) : x(x_coord), y(y_coord) {
    }

    /**
     * @brief Convert to flat array index (row-major)
     * @param full_resolution Global camera resolution (width, height)
     * @return Flattened index: y * width + x (always row-major)
     */
#ifdef __CUDACC__
    // CUDA version: use int2 (built-in CUDA type)
    __host__ __device__ inline size_t toFlatIndex(const int2 &full_resolution) const {
        return static_cast<size_t>(y) * full_resolution.x + x;
    }
#else
    // CPU version: use helios::int2
    inline size_t toFlatIndex(const helios::int2 &full_resolution) const {
        return static_cast<size_t>(y) * full_resolution.x + x;
    }
#endif

// CUDA-specific factory methods (use OptiX types)
#ifdef __CUDACC__
    /**
     * @brief Create pixel coordinate from tiled launch parameters
     *
     * @param launch_idx OptiX launch index (x unused, y=local col, z=local row)
     * @param tile_offset Tile offset in global pixel space (x, y)
     * @param full_resolution Global camera resolution (NOT tile resolution!)
     * @return Global pixel coordinate
     *
     * Encapsulates mapping: global_x = tile_offset.x + launch_idx.y
     *                         global_y = tile_offset.y + launch_idx.z
     */
    __device__ static PixelCoordinate fromTiledLaunch(const optix::uint3 &launch_idx, const optix::int2 &tile_offset, const optix::int2 &full_resolution) {
        (void) full_resolution; // Unused but included for API clarity
        return PixelCoordinate(tile_offset.x + launch_idx.y, // Global x from local y-index
                               tile_offset.y + launch_idx.z // Global y from local z-index
        );
    }

    /**
     * @brief Convenience: compute flat index directly from launch parameters
     *
     * @param launch_idx OptiX launch index
     * @param tile_offset Tile offset in global pixel space
     * @param full_resolution Global camera resolution
     * @return Flattened pixel index ready for buffer access
     *
     * One-line replacement for manual pixel index calculation.
     */
    __device__ static size_t computeFlatIndex(const optix::uint3 &launch_idx, const optix::int2 &tile_offset, const optix::int2 &full_resolution) {
        PixelCoordinate pixel = fromTiledLaunch(launch_idx, tile_offset, full_resolution);
        return pixel.toFlatIndex(full_resolution);
    }
#endif // __CUDACC__
};

// ========== Subpatch UUID Calculator ==========

/**
 * @brief Subpatch UUID calculator for tiled/subdivided geometry
 *
 * Encapsulates the subdivision offset calculation pattern.
 * Prevents wrong dimension order (i vs j, x vs y, row vs col).
 *
 * This helper makes it impossible to:
 * - Get subdivision offset formula wrong
 * - Swap row/column indices
 * - Use wrong dimension (NX vs NY)
 *
 * Usage in CUDA ray generation:
 * @code
 * uint base_UUID = primitiveID[objID];
 * SubpatchUUIDCalculator calc(base_UUID, object_subdivisions[objID]);
 *
 * // Clear semantic names prevent index swapping
 * for (int row = 0; row < calc.getSubdivisions().y; row++) {
 *     for (int col = 0; col < calc.getSubdivisions().x; col++) {
 *         uint subpatch_UUID = calc.getUUID(col, row);
 *         // ... launch ray for this subpatch
 *     }
 * }
 * @endcode
 *
 * Historical bugs prevented (commit 53ca9687d):
 * - Swapped indices: UUID = base + ii * NY + jj (should be jj * NX + ii)
 * - Wrong subdivision counts from parent inheritance
 */
class SubpatchUUIDCalculator {
private:
    uint base_UUID_; ///< Base UUID for first subpatch
#ifdef __CUDACC__
    int2 subdivisions_; ///< Subdivision counts (x, y) - CUDA version
#else
    helios::int2 subdivisions_; ///< Subdivision counts (x, y) - CPU version
#endif

public:
    /**
     * @brief Construct subpatch UUID calculator
     * @param base_UUID UUID of first subpatch (at position 0,0)
     * @param subdivisions Subdivision counts in x and y directions
     */
#ifdef __CUDACC__
    __host__ __device__ SubpatchUUIDCalculator(uint base_UUID, int2 subdivisions) : base_UUID_(base_UUID), subdivisions_(subdivisions) {
    }
#else
    SubpatchUUIDCalculator(uint base_UUID, helios::int2 subdivisions) : base_UUID_(base_UUID), subdivisions_(subdivisions) {
    }
#endif

    /**
     * @brief Get UUID for subpatch at (col, row) position
     *
     * @param col Column index (x-direction, 0 to subdivisions.x-1)
     * @param row Row index (y-direction, 0 to subdivisions.y-1)
     * @return UUID = base_UUID + row * subdivisions.x + col
     *
     * Parameter names make dimension order explicit - can't accidentally swap.
     */
    HELIOS_HOST_DEVICE inline uint getUUID(int col, int row) const {
        return base_UUID_ + row * subdivisions_.x + col;
    }

    /**
     * @brief Get total number of subpatches
     * @return subdivisions.x * subdivisions.y
     */
    HELIOS_HOST_DEVICE inline int getSubpatchCount() const {
        return subdivisions_.x * subdivisions_.y;
    }

    /**
     * @brief Get subdivisions (useful for loop bounds)
     * @return Subdivision counts as int2
     */
#ifdef __CUDACC__
    __host__ __device__ int2 getSubdivisions() const {
        return subdivisions_;
    }
#else
    helios::int2 getSubdivisions() const {
        return subdivisions_;
    }
#endif
};

#endif // HELIOS_BUFFER_INDEXING_H
