/** \file "OptiX8Backend.h" OptiX 8.1 ray tracing backend implementation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef OPTIX8_BACKEND_H
#define OPTIX8_BACKEND_H

#include "RayTracingBackend.h"
#include "../optix8/OptiX8LaunchParams.h"

// CUDA runtime
#include <cuda_runtime.h>

// OptiX 8.1 host API
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

// Standard library
#include <vector>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t rc = call;                                                                    \
        if (rc != cudaSuccess) {                                                                  \
            helios_runtime_error("CUDA error in " + std::string(__FILE__) + ":" +                \
                                 std::to_string(__LINE__) + ": " + cudaGetErrorString(rc));       \
        }                                                                                         \
    } while (0)

// OptiX error checking macro
#define OPTIX_CHECK(call)                                                                         \
    do {                                                                                          \
        OptixResult rc = call;                                                                    \
        if (rc != OPTIX_SUCCESS) {                                                                \
            helios_runtime_error("OptiX error in " + std::string(__FILE__) + ":" +               \
                                 std::to_string(__LINE__) + ": " + optixGetErrorName(rc) +        \
                                 " (" + optixGetErrorString(rc) + ")");                           \
        }                                                                                         \
    } while (0)


namespace helios {

    /**
     * @brief OptiX 8.1 implementation of the ray tracing backend.
     *
     * Uses the modern OptiX 8 pipeline/SBT model. Compatible with NVIDIA
     * drivers >= 560 (required for OptiX 8.0+). Automatically selected when
     * HELIOS_HAVE_OPTIX8 is defined at compile time.
     *
     * Architecture:
     * - Single GAS with 6 AABB custom intersection programs (one per primitive type)
     * - Single pipeline: 4 raygen + 4 closesthit + 4 miss + 6 intersection programs
     * - Launch params struct passed to optixLaunch for all data sharing
     * - PerRayData via pointer-in-two-payload-registers pattern
     */
    class OptiX8Backend : public RayTracingBackend {
    public:
        OptiX8Backend();
        ~OptiX8Backend() override;

        /**
         * @brief Lightweight probe to check if OptiX 8 hardware is available
         *
         * Checks CUDA device availability and OptiX driver initialization without
         * allocating persistent GPU resources.
         *
         * @return true if a CUDA device exists and optixInit() succeeds
         */
        static bool probe() noexcept;

        // Lifecycle
        void initialize() override;
        void shutdown() override;

        // Geometry management
        void updateGeometry(const RayTracingGeometry &geometry) override;
        void buildAccelerationStructure() override;

        // Material/optical properties
        void updateMaterials(const RayTracingMaterial &materials) override;

        // Radiation sources
        void updateSources(const std::vector<RayTracingSource> &sources) override;

        // Diffuse/sky radiation
        void updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction,
                                    const std::vector<helios::vec3> &peak_dir,
                                    const std::vector<float> &dist_norm,
                                    const std::vector<float> &sky_energy) override;

        void updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params,
                            const std::vector<float> &camera_sky_radiance,
                            const helios::vec3 &sun_direction,
                            const std::vector<float> &solar_disk_radiance,
                            float solar_disk_cos_angle) override;

        // Ray launching
        void launchDirectRays(const RayTracingLaunchParams &params) override;
        void launchDiffuseRays(const RayTracingLaunchParams &params) override;
        void launchCameraRays(const RayTracingLaunchParams &params) override;
        void launchPixelLabelRays(const RayTracingLaunchParams &params) override;

        // Results retrieval
        void getRadiationResults(RayTracingResults &results) override;
        void getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels,
                              std::vector<float> &pixel_depths, uint camera_id,
                              const helios::int2 &resolution) override;

        // Buffer utilities
        void zeroRadiationBuffers(size_t launch_band_count) override;
        void zeroScatterBuffers() override;
        void zeroCameraPixelBuffers(const helios::int2 &resolution) override;
        void copyScatterToRadiation() override;
        void uploadRadiationOut(const std::vector<float> &radiation_out_top,
                                const std::vector<float> &radiation_out_bottom) override;
        void uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam,
                                        const std::vector<float> &scatter_bottom_cam) override;
        void zeroCameraScatterBuffers(size_t launch_band_count) override;
        void uploadSourceFluxes(const std::vector<float> &fluxes) override;
        void uploadSourceFluxesCam(const std::vector<float> &fluxes_cam) override;

        // Diagnostics
        void queryGPUMemory() const override;
        std::string getBackendName() const override { return "OptiX 8.1"; }

    private:
        // ---- OptiX pipeline objects ----
        OptixDeviceContext        optix_context    = nullptr;
        OptixModule               optix_module     = nullptr;
        OptixPipeline             optix_pipeline   = nullptr;
        CUstream                  cuda_stream      = nullptr;

        // Program groups (4 raygen + 4 miss + 4 closest-hit + 6 intersection)
        OptixProgramGroup pg_raygen_direct       = nullptr;
        OptixProgramGroup pg_raygen_diffuse      = nullptr;
        OptixProgramGroup pg_raygen_camera       = nullptr;
        OptixProgramGroup pg_raygen_pixel_label  = nullptr;

        OptixProgramGroup pg_miss_direct         = nullptr;
        OptixProgramGroup pg_miss_diffuse        = nullptr;
        OptixProgramGroup pg_miss_camera         = nullptr;
        OptixProgramGroup pg_miss_pixel_label    = nullptr;

        OptixProgramGroup pg_hit_direct          = nullptr;
        OptixProgramGroup pg_hit_diffuse         = nullptr;
        OptixProgramGroup pg_hit_camera          = nullptr;
        OptixProgramGroup pg_hit_pixel_label     = nullptr;

        // ---- GAS / acceleration structure ----
        OptixTraversableHandle    gas_handle       = 0;
        CUdeviceptr               d_gas_output     = 0;

        // ---- SBT ----
        OptixShaderBindingTable   sbt              = {};
        CUdeviceptr               d_raygen_records = 0;
        CUdeviceptr               d_miss_records   = 0;
        CUdeviceptr               d_hitgroup_records= 0;

        // Per-raygen device pointers (byte offsets into d_raygen_records; set in buildSBT)
        CUdeviceptr               d_raygen_record_direct      = 0;
        CUdeviceptr               d_raygen_record_diffuse     = 0;
        CUdeviceptr               d_raygen_record_camera      = 0;
        CUdeviceptr               d_raygen_record_pixel_label = 0;

        // ---- Launch params (device copy) ----
        OptiX8LaunchParams        h_params         = {};
        CUdeviceptr               d_params         = 0;

        // ---- Geometry device buffers ----
        CUdeviceptr d_transform_matrix       = 0;
        CUdeviceptr d_primitive_type         = 0;
        CUdeviceptr d_primitive_positions    = 0;
        CUdeviceptr d_primitiveID            = 0;
        CUdeviceptr d_objectID               = 0;
        CUdeviceptr d_object_subdivisions    = 0;
        CUdeviceptr d_twosided_flag          = 0;
        CUdeviceptr d_primitive_solid_fraction = 0;

        // Per-type geometry
        CUdeviceptr d_patch_vertices         = 0;
        CUdeviceptr d_patch_UUIDs            = 0;
        CUdeviceptr d_triangle_vertices      = 0;
        CUdeviceptr d_triangle_UUIDs         = 0;
        CUdeviceptr d_disk_centers           = 0;
        CUdeviceptr d_disk_radii             = 0;
        CUdeviceptr d_disk_normals           = 0;
        CUdeviceptr d_disk_UUIDs             = 0;
        CUdeviceptr d_tile_vertices          = 0;
        CUdeviceptr d_tile_UUIDs             = 0;
        CUdeviceptr d_voxel_vertices         = 0;
        CUdeviceptr d_voxel_UUIDs            = 0;
        CUdeviceptr d_bbox_vertices          = 0;
        CUdeviceptr d_bbox_UUIDs             = 0;

        // Global UUID array: d_primitive_uuid[global_pos] = UUID
        CUdeviceptr d_primitive_uuid_arr     = 0;

        // AABB buffer for GAS build (one AABB per primitive)
        CUdeviceptr d_aabbs                  = 0;

        // ---- Material device buffers ----
        CUdeviceptr d_rho                    = 0;
        CUdeviceptr d_tau                    = 0;
        CUdeviceptr d_rho_cam                = 0;
        CUdeviceptr d_tau_cam                = 0;
        CUdeviceptr d_specular_exponent      = 0;
        CUdeviceptr d_specular_scale         = 0;

        // ---- Radiation energy device buffers ----
        CUdeviceptr d_radiation_in           = 0;
        CUdeviceptr d_radiation_out_top      = 0;
        CUdeviceptr d_radiation_out_bottom   = 0;
        CUdeviceptr d_scatter_buff_top       = 0;
        CUdeviceptr d_scatter_buff_bottom    = 0;
        CUdeviceptr d_radiation_in_camera    = 0;
        CUdeviceptr d_scatter_buff_top_cam   = 0;
        CUdeviceptr d_scatter_buff_bottom_cam= 0;
        CUdeviceptr d_radiation_specular     = 0;
        CUdeviceptr d_Rsky                   = 0;

        // ---- Camera device buffers ----
        CUdeviceptr d_camera_pixel_label     = 0;
        CUdeviceptr d_camera_pixel_depth     = 0;

        // ---- Source device buffers ----
        CUdeviceptr d_source_positions       = 0;
        CUdeviceptr d_source_rotations       = 0;
        CUdeviceptr d_source_widths          = 0;
        CUdeviceptr d_source_types           = 0;
        CUdeviceptr d_source_fluxes          = 0;
        CUdeviceptr d_source_fluxes_cam      = 0;

        // ---- Diffuse/sky device buffers ----
        CUdeviceptr d_diffuse_flux           = 0;
        CUdeviceptr d_diffuse_extinction     = 0;
        CUdeviceptr d_diffuse_peak_dir       = 0;
        CUdeviceptr d_diffuse_dist_norm      = 0;
        CUdeviceptr d_sky_radiance_params    = 0;
        CUdeviceptr d_camera_sky_radiance    = 0;
        CUdeviceptr d_solar_disk_radiance    = 0;

        // ---- Band launch flag buffer ----
        CUdeviceptr d_band_launch_flag       = 0;

        // ---- Texture/mask device buffers ----
        CUdeviceptr d_mask_data              = 0; //!< flat uint8: 1=opaque, 0=transparent
        CUdeviceptr d_mask_offsets           = 0; //!< cumulative per-mask start index in mask_data
        CUdeviceptr d_mask_sizes             = 0; //!< 2 ints per mask: width, height
        CUdeviceptr d_mask_IDs               = 0; //!< per-primitive mask ID (-1 = no mask)
        CUdeviceptr d_uv_data                = 0; //!< uv_data[pos*4 + vtx]: flat UV, 4 verts per prim
        CUdeviceptr d_uv_IDs                 = 0; //!< per-primitive: -1=parametric, >=0=custom UV

        // ---- State ----
        bool     is_initialized          = false;
        size_t   current_primitive_count = 0;
        size_t   current_patch_count     = 0;
        size_t   current_triangle_count  = 0;
        size_t   current_disk_count      = 0;
        size_t   current_tile_count      = 0;
        size_t   current_voxel_count     = 0;
        size_t   current_bbox_count      = 0;
        size_t   current_source_count    = 0;
        size_t   current_band_count      = 0;
        size_t   current_camera_count    = 0;
        size_t   current_launch_band_count = 0;   //!< Nbands_launch used for camera buffers
        uint32_t current_camera_launch_id  = 0xFFFFFFFFu; //!< Camera ID from last launchCameraRays

        // ---- Private helper methods ----

        /// Free a device pointer if non-null, then set it to 0
        void freeCUdeviceptr(CUdeviceptr &ptr);

        /// Free all geometry-related device buffers
        void freeGeometryBuffers();

        /// Free all material-related device buffers
        void freeMaterialBuffers();

        /// Compute host-side AABBs for all primitives and upload to d_aabbs
        void buildAABBs(const RayTracingGeometry &geometry);

        /// Build or rebuild the GAS from the current d_aabbs buffer
        void buildGAS(uint32_t Nprimitives);

        /// Populate the SBT records after pipeline creation
        void buildSBT();

        /// Fill h_params from stored device pointers and upload to d_params
        void uploadLaunchParams();

        /// Apply RayTracingLaunchParams scalar fields to h_params
        void applyLaunchParams(const RayTracingLaunchParams &params);

        /// Reallocate a device buffer to a new size (frees old allocation)
        void reallocDevice(CUdeviceptr &ptr, size_t bytes);

        /// Download a device float buffer to a host vector
        std::vector<float> downloadFloat(CUdeviceptr ptr, size_t count) const;

        /// Download a device uint buffer to a host vector
        std::vector<uint32_t> downloadUInt32(CUdeviceptr ptr, size_t count) const;

        /// Locate the compiled PTX/OptixIR file for device programs
        std::string findDeviceCodeFile() const;
    };

} // namespace helios

#endif // OPTIX8_BACKEND_H
