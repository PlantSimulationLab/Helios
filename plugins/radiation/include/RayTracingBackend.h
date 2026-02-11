/** \file "RayTracingBackend.h" Abstract interface for ray tracing backends.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef RAYTRACING_BACKEND_H
#define RAYTRACING_BACKEND_H

#include "RayTracingTypes.h"
#include <memory>
#include <string>

namespace helios {

    /**
     * @brief Abstract interface for ray-tracing backends
     *
     * This interface provides a backend-agnostic API for ray tracing operations.
     * Implementations include:
     * - OptiX6Backend: OptiX 6.5 implementation (current)
     * - OptiX7Backend: OptiX 7.7 implementation (future - Phase 2)
     * - VulkanBackend: Vulkan ray-tracing implementation (future - Phase 3)
     *
     * The interface is designed to support:
     * - 4 ray types: direct, diffuse, camera, pixel_label
     * - 6 primitive types: patch, triangle, disk, tile, voxel, bbox
     * - Multi-band spectral radiation
     * - Multiple radiation sources
     * - Camera rendering with depth and labeling
     * - Scattering/multi-bounce radiation
     */
    class RayTracingBackend {
    public:
        virtual ~RayTracingBackend() = default;

        // ========== Lifecycle Management ==========

        /**
         * @brief Initialize the ray tracing backend
         *
         * Creates the ray tracing context, allocates buffers, compiles/loads programs,
         * and sets up the acceleration structure.
         */
        virtual void initialize() = 0;

        /**
         * @brief Shutdown the ray tracing backend
         *
         * Destroys the context, frees all buffers and resources.
         */
        virtual void shutdown() = 0;

        // ========== Geometry Management ==========

        /**
         * @brief Update geometry data in the ray tracing backend
         *
         * @param[in] geometry Geometry data for all primitive types
         *
         * This uploads vertex data, transform matrices, UUIDs, texture masks,
         * and all other geometry-related data to the GPU.
         */
        virtual void updateGeometry(const RayTracingGeometry &geometry) = 0;

        /**
         * @brief Build or rebuild the acceleration structure
         *
         * Constructs the BVH or other acceleration structure for efficient ray traversal.
         * Should be called after updateGeometry() or when geometry changes.
         */
        virtual void buildAccelerationStructure() = 0;

        // ========== Material/Optical Properties ==========

        /**
         * @brief Update material properties (reflectivity, transmissivity, specular)
         *
         * @param[in] materials Material data for all primitives, sources, and bands
         */
        virtual void updateMaterials(const RayTracingMaterial &materials) = 0;

        // ========== Radiation Sources ==========

        /**
         * @brief Update radiation source definitions
         *
         * @param[in] sources Vector of radiation sources (collimated, sphere, sun, rectangle, disk)
         */
        virtual void updateSources(const std::vector<RayTracingSource> &sources) = 0;

        // ========== Diffuse/Sky Radiation ==========

        /**
         * @brief Update diffuse (ambient) radiation parameters
         *
         * @param[in] flux Diffuse flux per band
         * @param[in] extinction Extinction coefficient per band
         * @param[in] peak_dir Peak direction for diffuse distribution per band
         * @param[in] dist_norm Distribution normalization factor per band
         * @param[in] sky_energy Energy absorbed by sky per band
         */
        virtual void updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm, const std::vector<float> &sky_energy) = 0;

        /**
         * @brief Update sky radiance model parameters (for camera rendering)
         *
         * @param[in] sky_radiance_params Prague sky model parameters per band (vec4)
         * @param[in] camera_sky_radiance Sky radiance for cameras per band
         * @param[in] sun_direction Direction to the sun
         * @param[in] solar_disk_radiance Solar disk radiance per band
         * @param[in] solar_disk_cos_angle Cosine of solar disk angular radius
         */
        virtual void updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction, const std::vector<float> &solar_disk_radiance,
                                    float solar_disk_cos_angle) = 0;

        // ========== Ray Launching ==========

        /**
         * @brief Launch direct (unscattered) rays to radiation sources
         *
         * @param[in] params Launch parameters (dimensions, seed, band config, etc.)
         */
        virtual void launchDirectRays(const RayTracingLaunchParams &params) = 0;

        /**
         * @brief Launch diffuse (hemispherically-distributed) rays
         *
         * @param[in] params Launch parameters including diffuse distribution parameters
         */
        virtual void launchDiffuseRays(const RayTracingLaunchParams &params) = 0;

        /**
         * @brief Launch camera rays for image rendering
         *
         * @param[in] params Launch parameters including camera position, direction, FOV, etc.
         */
        virtual void launchCameraRays(const RayTracingLaunchParams &params) = 0;

        /**
         * @brief Launch pixel labeling rays (for primitive identification in images)
         *
         * @param[in] params Launch parameters with camera configuration
         */
        virtual void launchPixelLabelRays(const RayTracingLaunchParams &params) = 0;

        // ========== Results Retrieval ==========

        /**
         * @brief Get radiation simulation results
         *
         * @param[out] results Structure containing radiation_in, radiation_out, scatter buffers, etc.
         *
         * Retrieves absorbed and emitted radiation for all primitives and bands.
         */
        virtual void getRadiationResults(RayTracingResults &results) = 0;

        /**
         * @brief Get camera rendering results
         *
         * @param[out] pixel_data Flattened pixel values [band][pixel_y][pixel_x]
         * @param[out] pixel_labels Primitive UUID for each pixel (0 if sky/miss)
         * @param[out] pixel_depths Depth value for each pixel
         * @param[in] camera_id Camera identifier
         * @param[in] resolution Camera resolution (width, height)
         */
        virtual void getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels, std::vector<float> &pixel_depths, uint camera_id, const helios::int2 &resolution) = 0;

        // ========== Buffer Management Utilities ==========

        /**
         * @brief Zero all radiation result buffers (radiation_in, radiation_out, radiation_specular)
         *
         * @param[in] launch_band_count Number of bands being launched in current iteration
         *
         * Camera scatter buffers are sized by launch_band_count (not global band count) to match
         * per-launch accumulation semantics. All other buffers use the global band count.
         */
        virtual void zeroRadiationBuffers(size_t launch_band_count) = 0;

        /**
         * @brief Zero scatter buffers (scatter_buff_top, scatter_buff_bottom)
         */
        virtual void zeroScatterBuffers() = 0;

        /**
         * @brief Zero camera pixel label and depth buffers
         *
         * @param[in] resolution Full camera resolution (width, height)
         *
         * Must be called before launching pixel label rays for multiple tiles.
         * Each tile launch accumulates results without zeroing, so this must
         * be called once before the tile loop to initialize the buffers.
         */
        virtual void zeroCameraPixelBuffers(const helios::int2 &resolution) = 0;

        /**
         * @brief Copy scatter buffer contents to radiation buffers
         *
         * Used for multi-bounce radiation: scattering iteration results are accumulated
         * into the main radiation buffers.
         */
        virtual void copyScatterToRadiation() = 0;

        /**
         * @brief Upload radiation_out buffers directly (without launching rays)
         *
         * @param[in] radiation_out_top Top-face outgoing radiation [Nprims * Nbands]
         * @param[in] radiation_out_bottom Bottom-face outgoing radiation [Nprims * Nbands]
         *
         * Used to set outgoing radiation for scattering iterations.
         */
        virtual void uploadRadiationOut(const std::vector<float> &radiation_out_top, const std::vector<float> &radiation_out_bottom) = 0;

        /**
         * @brief Upload camera scatter buffers
         *
         * @param[in] scatter_top_cam Scattered radiation for cameras (top surface) indexed by [primitive * Nbands_launch + band]
         * @param[in] scatter_bottom_cam Scattered radiation for cameras (bottom surface)
         *
         * Used to upload emission energy that cameras should see before scatter iterations.
         */
        virtual void uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam, const std::vector<float> &scatter_bottom_cam) = 0;

        /**
         * @brief Zero camera scatter buffers (scatter_buff_top_cam, scatter_buff_bottom_cam)
         *
         * @param[in] launch_band_count Number of bands being launched in current iteration
         *
         * Buffers are sized by launch_band_count to prevent double-counting when accumulating
         * camera scatter across multiple ray launches.
         */
        virtual void zeroCameraScatterBuffers(size_t launch_band_count) = 0;

        /**
         * @brief Upload source flux values for launched bands
         *
         * @param[in] fluxes Source flux values indexed by [source * Nbands_launch + band]
         *
         * Used to set per-band source fluxes before direct ray launch.
         */
        virtual void uploadSourceFluxes(const std::vector<float> &fluxes) = 0;

        // ========== Diagnostics ==========

        /**
         * @brief Query and report GPU memory usage
         */
        virtual void queryGPUMemory() const = 0;

        /**
         * @brief Get the name of the backend implementation
         *
         * @return Backend name (e.g., "OptiX 6.5", "OptiX 7.7", "Vulkan")
         */
        virtual std::string getBackendName() const = 0;

    protected:
        /**
         * @brief Validate geometry before upload (debug builds only)
         *
         * Backend implementers MUST call this at the START of updateGeometry().
         * Catches buffer sizing errors before GPU upload, preventing crashes.
         *
         * @param geometry Geometry to validate
         * @throws helios_runtime_error if validation fails with detailed error message
         * @note Compiled out in release builds (#ifndef NDEBUG) - zero cost
         *
         * This validation prevents historical bugs like:
         * - primitive_IDs sized by Nobjects instead of Nprimitives (commit 53ca9687d)
         * - Wrong primitive_positions size
         * - Inconsistent type-specific buffer sizes
         */
        void validateGeometryBeforeUpload(const RayTracingGeometry &geometry) const {
#ifndef NDEBUG
            geometry.validate();
#endif
        }

    public:
        // ========== Factory Method ==========

        /**
         * @brief Create a ray tracing backend instance
         *
         * @param[in] backend_type Backend type string: "optix6", "optix7", "vulkan"
         * @return Unique pointer to backend instance
         *
         * Factory method that selects and instantiates the appropriate backend.
         * Throws helios_runtime_error if backend type is unknown or unavailable.
         */
        static std::unique_ptr<RayTracingBackend> create(const std::string &backend_type);
    };

} // namespace helios

#endif // RAYTRACING_BACKEND_H
