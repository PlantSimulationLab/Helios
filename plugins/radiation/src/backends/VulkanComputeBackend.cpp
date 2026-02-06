/** \file "VulkanComputeBackend.cpp" Vulkan compute backend implementation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "VulkanComputeBackend.h"
#include <fstream>
#include <iostream>
#include <cstring>

namespace helios {

    VulkanComputeBackend::VulkanComputeBackend() {
        device = std::make_unique<VulkanDevice>();
    }

    VulkanComputeBackend::~VulkanComputeBackend() {
        shutdown();
    }

    void VulkanComputeBackend::initialize() {
        // Initialize Vulkan device
        device->initialize(false); // Validation off for release

        // Create command resources
        createCommandResources();

        std::cout << "Vulkan compute backend initialized successfully." << std::endl;
    }

    void VulkanComputeBackend::shutdown() {
        if (!device || device->getDevice() == VK_NULL_HANDLE) {
            return; // Already shutdown
        }

        VkDevice vk_device = device->getDevice();

        // Wait for device idle
        vkDeviceWaitIdle(vk_device);

        // Destroy pipelines
        if (pipeline_direct != VK_NULL_HANDLE)
            vkDestroyPipeline(vk_device, pipeline_direct, nullptr);
        if (pipeline_diffuse != VK_NULL_HANDLE)
            vkDestroyPipeline(vk_device, pipeline_diffuse, nullptr);
        if (pipeline_camera != VK_NULL_HANDLE)
            vkDestroyPipeline(vk_device, pipeline_camera, nullptr);
        if (pipeline_pixel_label != VK_NULL_HANDLE)
            vkDestroyPipeline(vk_device, pipeline_pixel_label, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(vk_device, pipeline_layout, nullptr);

        // Destroy descriptor pool (also frees sets)
        if (descriptor_pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(vk_device, descriptor_pool, nullptr);
        if (set_layout_geometry != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(vk_device, set_layout_geometry, nullptr);
        if (set_layout_materials != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(vk_device, set_layout_materials, nullptr);
        if (set_layout_results != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(vk_device, set_layout_results, nullptr);

        // Destroy buffers
        destroyBuffer(bvh_buffer);
        destroyBuffer(primitive_indices_buffer);
        destroyBuffer(transform_matrices_buffer);
        destroyBuffer(primitive_types_buffer);
        destroyBuffer(primitive_uuids_buffer);
        destroyBuffer(primitive_positions_buffer);
        destroyBuffer(reflectivity_buffer);
        destroyBuffer(transmissivity_buffer);
        destroyBuffer(radiation_in_buffer);
        destroyBuffer(radiation_out_buffer);
        destroyBuffer(scatter_top_buffer);
        destroyBuffer(scatter_bottom_buffer);

        // Destroy command resources
        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(vk_device, command_pool, nullptr);

        // Shutdown device
        device->shutdown();
    }

    void VulkanComputeBackend::updateGeometry(const RayTracingGeometry &geometry) {
        validateGeometryBeforeUpload(geometry);

        primitive_count = geometry.primitive_count;

        if (primitive_count == 0) {
            return; // Empty geometry
        }

        // Build BVH on CPU
        std::cout << "Building BVH for " << primitive_count << " primitives..." << std::endl;
        bvh_nodes = bvh_builder.build(geometry);
        std::cout << "BVH built with " << bvh_nodes.size() << " nodes." << std::endl;

        // Upload BVH
        if (!bvh_nodes.empty()) {
            if (bvh_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(bvh_buffer);
            }
            bvh_buffer = createBuffer(bvh_nodes.size() * sizeof(BVHNode), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(bvh_buffer, bvh_nodes.data(), bvh_nodes.size() * sizeof(BVHNode));
        }

        // Upload primitive indices
        const auto &prim_indices = bvh_builder.getPrimitiveIndices();
        if (!prim_indices.empty()) {
            if (primitive_indices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(primitive_indices_buffer);
            }
            primitive_indices_buffer = createBuffer(prim_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(primitive_indices_buffer, prim_indices.data(), prim_indices.size() * sizeof(uint32_t));
        }

        // Upload transform matrices
        if (!geometry.transform_matrices.empty()) {
            if (transform_matrices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(transform_matrices_buffer);
            }
            transform_matrices_buffer = createBuffer(geometry.transform_matrices.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(transform_matrices_buffer, geometry.transform_matrices.data(), geometry.transform_matrices.size() * sizeof(float));
        }

        // Upload primitive types
        if (!geometry.primitive_types.empty()) {
            if (primitive_types_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(primitive_types_buffer);
            }
            primitive_types_buffer = createBuffer(geometry.primitive_types.size() * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(primitive_types_buffer, geometry.primitive_types.data(), geometry.primitive_types.size() * sizeof(uint));
        }

        // Upload primitive UUIDs
        if (!geometry.primitive_UUIDs.empty()) {
            if (primitive_uuids_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(primitive_uuids_buffer);
            }
            primitive_uuids_buffer = createBuffer(geometry.primitive_UUIDs.size() * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(primitive_uuids_buffer, geometry.primitive_UUIDs.data(), geometry.primitive_UUIDs.size() * sizeof(uint));
        }

        // Upload UUID→position lookup
        if (!geometry.primitive_positions.empty()) {
            if (primitive_positions_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(primitive_positions_buffer);
            }
            primitive_positions_buffer = createBuffer(geometry.primitive_positions.size() * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(primitive_positions_buffer, geometry.primitive_positions.data(), geometry.primitive_positions.size() * sizeof(uint));
        }

        std::cout << "Geometry uploaded successfully." << std::endl;
    }

    void VulkanComputeBackend::buildAccelerationStructure() {
        // No-op: BVH is built in updateGeometry()
    }

    // ========== Stub implementations (to be completed in later phases) ==========

    void VulkanComputeBackend::updateMaterials(const RayTracingMaterial &materials) {
        band_count = materials.num_bands;
        // TODO: Upload material buffers in Phase 2+
    }

    void VulkanComputeBackend::updateSources(const std::vector<RayTracingSource> &sources) {
        // TODO: Upload source data in Phase 1+
    }

    void VulkanComputeBackend::updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm,
                                                      const std::vector<float> &sky_energy) {
        // TODO: Upload diffuse radiation params in Phase 2+
    }

    void VulkanComputeBackend::updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction,
                                              const std::vector<float> &solar_disk_radiance, float solar_disk_cos_angle) {
        // TODO: Upload sky model params in Phase 7+
    }

    void VulkanComputeBackend::launchDirectRays(const RayTracingLaunchParams &params) {
        // TODO: Implement in Phase 1
    }

    void VulkanComputeBackend::launchDiffuseRays(const RayTracingLaunchParams &params) {
        // TODO: Implement in Phase 2
    }

    void VulkanComputeBackend::launchCameraRays(const RayTracingLaunchParams &params) {
        // TODO: Implement in Phase 7
    }

    void VulkanComputeBackend::launchPixelLabelRays(const RayTracingLaunchParams &params) {
        // TODO: Implement in Phase 7
    }

    void VulkanComputeBackend::getRadiationResults(RayTracingResults &results) {
        // TODO: Download result buffers in Phase 1+
    }

    void VulkanComputeBackend::getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels, std::vector<float> &pixel_depths, uint camera_id, const helios::int2 &resolution) {
        // TODO: Download camera buffers in Phase 7
    }

    void VulkanComputeBackend::zeroRadiationBuffers(size_t launch_band_count) {
        // TODO: Zero radiation buffers in Phase 1
    }

    void VulkanComputeBackend::zeroScatterBuffers() {
        // TODO: Zero scatter buffers in Phase 4
    }

    void VulkanComputeBackend::zeroCameraPixelBuffers(const helios::int2 &resolution) {
        // TODO: Zero camera pixel buffers in Phase 7
    }

    void VulkanComputeBackend::copyScatterToRadiation() {
        // TODO: Copy scatter→radiation in Phase 4
    }

    void VulkanComputeBackend::uploadRadiationOut(const std::vector<float> &radiation_out_top, const std::vector<float> &radiation_out_bottom) {
        // TODO: Upload radiation_out in Phase 3
    }

    void VulkanComputeBackend::uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam, const std::vector<float> &scatter_bottom_cam) {
        // TODO: Upload camera scatter in Phase 7
    }

    void VulkanComputeBackend::zeroCameraScatterBuffers(size_t launch_band_count) {
        // TODO: Zero camera scatter in Phase 7
    }

    void VulkanComputeBackend::uploadSourceFluxes(const std::vector<float> &fluxes) {
        // TODO: Upload source fluxes in Phase 1
    }

    void VulkanComputeBackend::queryGPUMemory() const {
        // TODO: Query Vulkan memory budget in Phase 8
        std::cout << "Vulkan memory query not yet implemented." << std::endl;
    }

    // ========== Helper methods ==========

    VulkanComputeBackend::Buffer VulkanComputeBackend::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage mem_usage) {
        Buffer buffer;
        buffer.size = size;

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = mem_usage;

        VkResult result = vmaCreateBuffer(device->getAllocator(), &buffer_info, &alloc_info, &buffer.buffer, &buffer.allocation, nullptr);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createBuffer): Failed to create buffer. VkResult: " + std::to_string(result));
        }

        return buffer;
    }

    void VulkanComputeBackend::destroyBuffer(Buffer &buffer) {
        if (buffer.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(device->getAllocator(), buffer.buffer, buffer.allocation);
            buffer.buffer = VK_NULL_HANDLE;
            buffer.allocation = VK_NULL_HANDLE;
            buffer.size = 0;
        }
    }

    void VulkanComputeBackend::uploadBufferData(Buffer &buffer, const void *data, size_t size) {
        // Create staging buffer
        Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

        // Map and copy
        void *mapped;
        vmaMapMemory(device->getAllocator(), staging.allocation, &mapped);
        std::memcpy(mapped, data, size);
        vmaUnmapMemory(device->getAllocator(), staging.allocation);

        // Copy staging → device buffer
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        VkBufferCopy copy_region{};
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, staging.buffer, buffer.buffer, 1, &copy_region);

        vkEndCommandBuffer(command_buffer);

        // Submit and wait
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(device->getComputeQueue());

        // Cleanup staging
        destroyBuffer(staging);
    }

    void VulkanComputeBackend::downloadBufferData(const Buffer &buffer, void *data, size_t size) {
        // Create staging buffer
        Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

        // Copy device → staging
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        VkBufferCopy copy_region{};
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, buffer.buffer, staging.buffer, 1, &copy_region);

        vkEndCommandBuffer(command_buffer);

        // Submit and wait
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(device->getComputeQueue());

        // Map and read
        void *mapped;
        vmaMapMemory(device->getAllocator(), staging.allocation, &mapped);
        std::memcpy(data, mapped, size);
        vmaUnmapMemory(device->getAllocator(), staging.allocation);

        // Cleanup staging
        destroyBuffer(staging);
    }

    void VulkanComputeBackend::zeroBuffer(Buffer &buffer) {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        vkCmdFillBuffer(command_buffer, buffer.buffer, 0, buffer.size, 0);

        vkEndCommandBuffer(command_buffer);

        // Submit and wait
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(device->getComputeQueue());
    }

    void VulkanComputeBackend::createCommandResources() {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = device->getComputeQueueFamily();
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkResult result = vkCreateCommandPool(device->getDevice(), &pool_info, nullptr, &command_pool);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to create command pool. VkResult: " + std::to_string(result));
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(device->getDevice(), &alloc_info, &command_buffer);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to allocate command buffer. VkResult: " + std::to_string(result));
        }
    }

    void VulkanComputeBackend::createDescriptorSets() {
        // TODO: Implement in Phase 1
    }

    void VulkanComputeBackend::createPipelines() {
        // TODO: Implement in Phase 1
    }

    VkShaderModule VulkanComputeBackend::loadShader(const std::string &filename) {
        // TODO: Implement in Phase 1
        return VK_NULL_HANDLE;
    }

} // namespace helios
