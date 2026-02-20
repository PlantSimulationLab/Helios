/** \file "VulkanComputeBackend.h" Vulkan compute shader ray tracing backend.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef VULKAN_COMPUTE_BACKEND_H
#define VULKAN_COMPUTE_BACKEND_H

#include "RayTracingBackend.h"
#include "VulkanDevice.h"
#include "BVHBuilder.h"
#include <vulkan/vulkan.h>

// Suppress nullability warnings from VMA header on macOS
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif
#include <vk_mem_alloc.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <memory>
#include <unordered_map>

namespace helios {

    /**
     * @brief Vulkan compute shader ray tracing backend
     *
     * Uses software BVH traversal in compute shaders (no VK_KHR_ray_tracing required).
     * Supports AMD, Intel, and Apple Silicon GPUs via Vulkan or MoltenVK.
     *
     * Performance target: 5-10x slower than OptiX (hardware RT).
     */
    class VulkanComputeBackend : public RayTracingBackend {
    public:
        VulkanComputeBackend();

        /**
         * @brief Constructor for test mode with shared device
         *
         * @param external_device Non-owning pointer to shared VulkanDevice (must be pre-initialized)
         *
         * INTERNAL TEST-ONLY CONSTRUCTOR
         * Used by test suite to share single VulkanDevice across all test cases.
         * Precondition: external_device must outlive this backend (satisfied by static singleton).
         */
        explicit VulkanComputeBackend(VulkanDevice* external_device);

        ~VulkanComputeBackend() override;

        // ========== Lifecycle Management ==========
        void initialize() override;
        void shutdown() override;

        // ========== Geometry Management ==========
        void updateGeometry(const RayTracingGeometry &geometry) override;
        void buildAccelerationStructure() override;

        // ========== Material/Optical Properties ==========
        void updateMaterials(const RayTracingMaterial &materials) override;

        // ========== Radiation Sources ==========
        void updateSources(const std::vector<RayTracingSource> &sources) override;

        // ========== Diffuse/Sky Radiation ==========
        void updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm,
                                    const std::vector<float> &sky_energy) override;

        void updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction, const std::vector<float> &solar_disk_radiance,
                            float solar_disk_cos_angle) override;

        // ========== Ray Launching ==========
        void launchDirectRays(const RayTracingLaunchParams &params) override;
        void launchDiffuseRays(const RayTracingLaunchParams &params) override;
        void launchCameraRays(const RayTracingLaunchParams &params) override;
        void launchPixelLabelRays(const RayTracingLaunchParams &params) override;

        // ========== Results Retrieval ==========
        void getRadiationResults(RayTracingResults &results) override;
        void getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels, std::vector<float> &pixel_depths, uint camera_id, const helios::int2 &resolution) override;

        // ========== Buffer Management Utilities ==========
        void zeroRadiationBuffers(size_t launch_band_count) override;
        void zeroScatterBuffers() override;
        void zeroCameraPixelBuffers(const helios::int2 &resolution) override;
        void copyScatterToRadiation() override;
        void uploadRadiationOut(const std::vector<float> &radiation_out_top, const std::vector<float> &radiation_out_bottom) override;
        void uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam, const std::vector<float> &scatter_bottom_cam) override;
        void zeroCameraScatterBuffers(size_t launch_band_count) override;
        void uploadSourceFluxes(const std::vector<float> &fluxes) override;

        // ========== Diagnostics ==========
        void queryGPUMemory() const override;
        std::string getBackendName() const override {
            return "Vulkan Compute (software BVH)";
        }

    private:
        // Vulkan device - either owned (production) or borrowed (test shared device)
        VulkanDevice* device;
        bool owns_device;  // true = we own device, false = borrowed from test singleton
        BVHBuilder bvh_builder;

        // BVH data
        std::vector<BVHNode> bvh_nodes;

        // Vulkan buffers (managed by VMA)
        struct Buffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDeviceSize size = 0;
        };

        // Geometry buffers (Set 0)
        Buffer bvh_buffer;
        Buffer primitive_indices_buffer;
        Buffer transform_matrices_buffer;
        Buffer primitive_types_buffer;
        Buffer primitive_uuids_buffer;
        Buffer primitive_positions_buffer;
        Buffer object_subdivisions_buffer;
        Buffer twosided_flag_buffer;
        Buffer patch_vertices_buffer;
        Buffer triangle_vertices_buffer;
        Buffer normal_buffer; //!< Pre-computed world-space normals (vec3 per primitive)
        Buffer mask_data_buffer;    //!< Flat uint mask texels: 0=transparent, 1=opaque
        Buffer mask_sizes_buffer;   //!< ivec2 (width, height) per unique mask texture
        Buffer mask_offsets_buffer; //!< uint offset into mask_data for start of each mask
        Buffer mask_IDs_buffer;     //!< int per primitive: mask index (-1 = no mask)
        Buffer uv_data_buffer;      //!< vec2 UV coords: 4 per primitive (flat, indexed by prim_idx*4)
        Buffer uv_IDs_buffer;       //!< int per primitive: -1 = default UVs, >=0 = has custom UVs
        Buffer bbox_vertices_buffer; //!< Bbox face vertices for periodic boundaries (12 floats per face)

        // Material/Source buffers (Set 1)
        Buffer source_positions_buffer;
        Buffer source_types_buffer;
        Buffer source_rotations_buffer;
        Buffer source_widths_buffer;
        Buffer source_fluxes_buffer;
        Buffer reflectivity_buffer;
        Buffer transmissivity_buffer;
        // TODO Phase 2+: Add diffuse params, specular

        // Result buffers (Set 2)
        Buffer radiation_in_buffer;
        Buffer radiation_out_buffer; // Phase 1 unified buffer (deprecated in Phase 2+)
        Buffer radiation_out_top_buffer;    // Phase 2+ separate top face
        Buffer radiation_out_bottom_buffer; // Phase 2+ separate bottom face
        Buffer scatter_top_buffer;
        Buffer scatter_bottom_buffer;
        Buffer camera_radiation_buffer;      //!< Camera pixel radiation values
        Buffer camera_pixel_label_buffer;    //!< Camera pixel UUID labels
        Buffer camera_pixel_depth_buffer;    //!< Camera pixel depth values
        Buffer camera_scatter_top_buffer;    //!< Camera-weighted scatter (top face)
        Buffer camera_scatter_bottom_buffer; //!< Camera-weighted scatter (bottom face)

        // Sky parameter buffers (Set 3) - Phase 2+
        Buffer diffuse_flux_buffer;
        Buffer diffuse_peak_dir_buffer;
        Buffer diffuse_extinction_buffer;
        Buffer diffuse_dist_norm_buffer;
        Buffer sky_radiance_params_buffer;
        Buffer camera_sky_radiance_buffer;   //!< Zenith sky radiance for camera miss shader
        Buffer solar_disk_radiance_buffer;   //!< Solar disk radiance for camera miss shader

        // Debug/profiling buffers (Set 4)
        Buffer debug_counters_buffer;

        // Descriptor sets
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_geometry = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_materials = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_results = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_sky = VK_NULL_HANDLE; // Phase 2+: Sky parameters
        VkDescriptorSetLayout set_layout_debug = VK_NULL_HANDLE; // Debug counters
        VkDescriptorSet set_geometry = VK_NULL_HANDLE;
        VkDescriptorSet set_materials = VK_NULL_HANDLE;
        VkDescriptorSet set_results = VK_NULL_HANDLE;
        VkDescriptorSet set_sky = VK_NULL_HANDLE; // Phase 2+: Sky parameters
        VkDescriptorSet set_debug = VK_NULL_HANDLE; // Debug counters

        // Cached sky/sun parameters (for camera push constants)
        helios::vec3 cached_sun_direction;
        float cached_solar_disk_cos_angle = 0.0f;

        // Compute pipelines
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline_direct = VK_NULL_HANDLE;
        VkPipeline pipeline_diffuse = VK_NULL_HANDLE;
        VkPipeline pipeline_camera = VK_NULL_HANDLE;
        VkPipeline pipeline_pixel_label = VK_NULL_HANDLE;

        // Command resources - separate for transfer and compute to avoid synchronization issues
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer transfer_command_buffer = VK_NULL_HANDLE;  // For buffer uploads/downloads
        VkCommandBuffer compute_command_buffer = VK_NULL_HANDLE;   // For compute shader dispatches
        VkFence transfer_fence = VK_NULL_HANDLE; // For synchronizing buffer operations
        VkFence compute_fence = VK_NULL_HANDLE;  // For synchronizing compute operations

        // Timestamp queries for GPU profiling
        VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
        float timestamp_period = 1.0f; // Nanoseconds per timestamp unit

        // Geometry cache
        size_t primitive_count = 0;
        size_t band_count = 0;  // Global band count (material buffer stride)
        size_t source_count = 0;

        // Periodic boundary state
        uint32_t bbox_count = 0;       //!< Number of bbox faces (0-4)
        float periodic_flag_x = 0;     //!< 1.0 if periodic in X direction
        float periodic_flag_y = 0;     //!< 1.0 if periodic in Y direction
        float domain_bounds[4] = {};   //!< Domain bounds: xmin, xmax, ymin, ymax

        // Per-launch band tracking (for radiation I/O buffers)
        uint32_t launch_band_count = 0;  // Current runBand() band count (set by zeroRadiationBuffers)
        std::vector<uint32_t> launch_to_global_band;  // Maps launch band index → global band index

        // Descriptor set update tracking
        bool descriptors_dirty = false;

        /**
         * @brief Create Vulkan buffer with VMA
         */
        Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage mem_usage);

        /**
         * @brief Destroy Vulkan buffer
         */
        void destroyBuffer(Buffer &buffer);

        /**
         * @brief Upload data to buffer
         */
        void uploadBufferData(Buffer &buffer, const void *data, size_t size);

        /**
         * @brief Download data from buffer
         */
        void downloadBufferData(const Buffer &buffer, void *data, size_t size);

        /**
         * @brief Zero buffer contents
         */
        void zeroBuffer(Buffer &buffer);

        /**
         * @brief Create descriptor sets
         */
        void createDescriptorSets();

        /**
         * @brief Create compute pipelines
         */
        void createPipelines();

        /**
         * @brief Load SPIR-V shader from file
         */
        VkShaderModule loadShader(const std::string &filename);

        /**
         * @brief Create command pool and allocate command buffer
         */
        void createCommandResources();

        /**
         * @brief Update descriptor set bindings with current buffers
         */
        void updateDescriptorSets();
    };

} // namespace helios

#endif // VULKAN_COMPUTE_BACKEND_H
