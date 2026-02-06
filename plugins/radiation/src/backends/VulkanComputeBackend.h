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
#include <vk_mem_alloc.h>
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
        // Vulkan device and allocator
        std::unique_ptr<VulkanDevice> device;
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
        // TODO: Add per-type vertex buffers, mask data, UV data

        // Material buffers (Set 1)
        Buffer reflectivity_buffer;
        Buffer transmissivity_buffer;
        // TODO: Add source data, diffuse params, specular params

        // Result buffers (Set 2)
        Buffer radiation_in_buffer;
        Buffer radiation_out_buffer;
        Buffer scatter_top_buffer;
        Buffer scatter_bottom_buffer;
        // TODO: Add camera pixel buffers

        // Descriptor sets
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_geometry = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_materials = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_results = VK_NULL_HANDLE;
        VkDescriptorSet set_geometry = VK_NULL_HANDLE;
        VkDescriptorSet set_materials = VK_NULL_HANDLE;
        VkDescriptorSet set_results = VK_NULL_HANDLE;

        // Compute pipelines
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline_direct = VK_NULL_HANDLE;
        VkPipeline pipeline_diffuse = VK_NULL_HANDLE;
        VkPipeline pipeline_camera = VK_NULL_HANDLE;
        VkPipeline pipeline_pixel_label = VK_NULL_HANDLE;

        // Command buffer
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;

        // Geometry cache
        size_t primitive_count = 0;
        size_t band_count = 0;

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
    };

} // namespace helios

#endif // VULKAN_COMPUTE_BACKEND_H
