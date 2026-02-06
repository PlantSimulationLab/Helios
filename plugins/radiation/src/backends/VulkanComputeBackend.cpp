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

        // Create descriptor sets and pipelines
        createDescriptorSets();
        createPipelines();

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
        if (transfer_fence != VK_NULL_HANDLE)
            vkDestroyFence(vk_device, transfer_fence, nullptr);
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
            size_t expected_size = primitive_count * 16;
            if (geometry.transform_matrices.size() != expected_size) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateGeometry): transform_matrices size mismatch. Expected " + std::to_string(expected_size) + " floats (16 per primitive), got " +
                                     std::to_string(geometry.transform_matrices.size()));
            }

            if (transform_matrices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(transform_matrices_buffer);
            }
            transform_matrices_buffer = createBuffer(geometry.transform_matrices.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(transform_matrices_buffer, geometry.transform_matrices.data(), geometry.transform_matrices.size() * sizeof(float));
        }

        // Upload primitive types
        if (!geometry.primitive_types.empty()) {
            if (geometry.primitive_types.size() != primitive_count) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateGeometry): primitive_types size mismatch. Expected " + std::to_string(primitive_count) + " entries, got " +
                                     std::to_string(geometry.primitive_types.size()));
            }

            if (primitive_types_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(primitive_types_buffer);
            }
            primitive_types_buffer = createBuffer(geometry.primitive_types.size() * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(primitive_types_buffer, geometry.primitive_types.data(), geometry.primitive_types.size() * sizeof(uint));
        }

        // Upload primitive UUIDs
        if (!geometry.primitive_UUIDs.empty()) {
            if (geometry.primitive_UUIDs.size() != primitive_count) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateGeometry): primitive_UUIDs size mismatch. Expected " + std::to_string(primitive_count) + " entries, got " +
                                     std::to_string(geometry.primitive_UUIDs.size()));
            }

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
        // Query VMA statistics
        VmaTotalStatistics stats;
        vmaCalculateStatistics(device->getAllocator(), &stats);

        std::cout << "========== Vulkan Memory Usage ==========" << std::endl;
        std::cout << "Allocated blocks: " << stats.total.statistics.blockCount << std::endl;
        std::cout << "Allocated memory: " << (stats.total.statistics.allocationBytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Used memory: " << (stats.total.statistics.blockBytes / 1024.0 / 1024.0) << " MB" << std::endl;

        // Query physical device memory properties
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(device->getPhysicalDevice(), &mem_props);

        std::cout << "\nTotal device memory heaps: " << mem_props.memoryHeapCount << std::endl;
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            std::cout << "  Heap " << i << ": " << (mem_props.memoryHeaps[i].size / 1024.0 / 1024.0 / 1024.0) << " GB";
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                std::cout << " (device-local)";
            }
            std::cout << std::endl;
        }
        std::cout << "==========================================" << std::endl;
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

        // Optimization: Use dedicated memory for large GPU-only buffers (>64 MB)
        if (mem_usage == VMA_MEMORY_USAGE_GPU_ONLY && size > 64 * 1024 * 1024) {
            alloc_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        }

        // Optimization: Keep staging buffers persistently mapped for faster CPU access
        if (mem_usage == VMA_MEMORY_USAGE_CPU_ONLY) {
            alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        }

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

        // Add memory barrier to ensure transfer completes before compute shader access
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

        vkEndCommandBuffer(command_buffer);

        // Submit with fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkResetFences(device->getDevice(), 1, &transfer_fence);
        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, transfer_fence);

        // Wait for fence
        vkWaitForFences(device->getDevice(), 1, &transfer_fence, VK_TRUE, UINT64_MAX);

        // SAFETY: Staging buffer destroyed only after fence signals (GPU copy complete)
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

        // Add memory barrier to ensure compute shader writes complete before transfer
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

        VkBufferCopy copy_region{};
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, buffer.buffer, staging.buffer, 1, &copy_region);

        vkEndCommandBuffer(command_buffer);

        // Submit with fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkResetFences(device->getDevice(), 1, &transfer_fence);
        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, transfer_fence);

        // Wait for fence
        vkWaitForFences(device->getDevice(), 1, &transfer_fence, VK_TRUE, UINT64_MAX);

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

        // Submit with fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkResetFences(device->getDevice(), 1, &transfer_fence);
        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, transfer_fence);

        // Wait for fence
        vkWaitForFences(device->getDevice(), 1, &transfer_fence, VK_TRUE, UINT64_MAX);
    }

    void VulkanComputeBackend::createCommandResources() {
        VkDevice vk_device = device->getDevice();

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = device->getComputeQueueFamily();
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkResult result = vkCreateCommandPool(vk_device, &pool_info, nullptr, &command_pool);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to create command pool. VkResult: " + std::to_string(result));
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(vk_device, &alloc_info, &command_buffer);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to allocate command buffer. VkResult: " + std::to_string(result));
        }

        // Create fence for buffer operation synchronization
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = 0; // Start unsignaled

        result = vkCreateFence(vk_device, &fence_info, nullptr, &transfer_fence);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to create transfer fence. VkResult: " + std::to_string(result));
        }
    }

    void VulkanComputeBackend::createDescriptorSets() {
        VkDevice vk_device = device->getDevice();

        // ========== Create Descriptor Set Layouts ==========

        // Set 0: Geometry buffers (changes on geometry update)
        std::vector<VkDescriptorSetLayoutBinding> geometry_bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // BVH nodes
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Primitive indices
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Transform matrices
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Primitive types
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Primitive UUIDs
            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Primitive positions
            // TODO Phase 1+: Add vertex buffers, mask data, UV data
        };

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(geometry_bindings.size());
        layout_info.pBindings = geometry_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_geometry) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create geometry descriptor set layout");
        }

        // Set 1: Material buffers (changes per simulation) - minimal for Phase 0
        std::vector<VkDescriptorSetLayoutBinding> material_bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Reflectivity
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Transmissivity
            // TODO Phase 1+: Add source data, diffuse params, specular
        };

        layout_info.bindingCount = static_cast<uint32_t>(material_bindings.size());
        layout_info.pBindings = material_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_materials) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create material descriptor set layout");
        }

        // Set 2: Result buffers (read/write, zeroed per-launch)
        std::vector<VkDescriptorSetLayoutBinding> result_bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // radiation_in
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // radiation_out
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // scatter_top
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // scatter_bottom
            // TODO Phase 7: Add camera buffers
        };

        layout_info.bindingCount = static_cast<uint32_t>(result_bindings.size());
        layout_info.pBindings = result_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_results) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create result descriptor set layout");
        }

        // ========== Create Descriptor Pool ==========

        std::vector<VkDescriptorPoolSize> pool_sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20}, // Enough for all 3 sets
        };

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.maxSets = 3; // geometry, materials, results

        if (vkCreateDescriptorPool(vk_device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create descriptor pool");
        }

        // ========== Allocate Descriptor Sets ==========

        VkDescriptorSetLayout layouts[] = {set_layout_geometry, set_layout_materials, set_layout_results};

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool;
        alloc_info.descriptorSetCount = 3;
        alloc_info.pSetLayouts = layouts;

        VkDescriptorSet sets[3];
        if (vkAllocateDescriptorSets(vk_device, &alloc_info, sets) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to allocate descriptor sets");
        }

        set_geometry = sets[0];
        set_materials = sets[1];
        set_results = sets[2];

        std::cout << "Descriptor sets created successfully." << std::endl;
    }

    void VulkanComputeBackend::createPipelines() {
        VkDevice vk_device = device->getDevice();

        // ========== Create Pipeline Layout ==========

        VkDescriptorSetLayout set_layouts[] = {set_layout_geometry, set_layout_materials, set_layout_results};

        // Push constants (128 bytes max for MoltenVK compatibility)
        const uint32_t push_constant_size = 128;

        // Validate against device limits
        const VkPhysicalDeviceProperties &props = device->getDeviceProperties();
        if (push_constant_size > props.limits.maxPushConstantsSize) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createPipelines): Push constant size (" + std::to_string(push_constant_size) + " bytes) exceeds device limit (" +
                                 std::to_string(props.limits.maxPushConstantsSize) + " bytes)");
        }

        // Warn if approaching MoltenVK's 128-byte limit
        if (device->isMoltenVK() && push_constant_size > 96) {
            std::cout << "WARNING: Push constants (" << push_constant_size << " bytes) are close to MoltenVK's 128-byte limit. Consider moving large parameters to UBO." << std::endl;
        }

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 3;
        pipeline_layout_info.pSetLayouts = set_layouts;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;

        if (vkCreatePipelineLayout(vk_device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createPipelines): Failed to create pipeline layout");
        }

        // ========== Load Shaders ==========

        // Shader paths (relative to build directory)
        std::string shader_dir = "plugins/radiation/";

        VkShaderModule shader_direct = loadShader(shader_dir + "direct_raygen.spv");
        VkShaderModule shader_diffuse = loadShader(shader_dir + "diffuse_raygen.spv");
        VkShaderModule shader_camera = loadShader(shader_dir + "camera_raygen.spv");
        VkShaderModule shader_pixel_label = loadShader(shader_dir + "pixel_label_raygen.spv");

        // ========== Create Compute Pipelines ==========

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.layout = pipeline_layout;

        // Direct ray pipeline
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = shader_direct;
        pipeline_info.stage.pName = "main";

        if (vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_direct) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createPipelines): Failed to create direct ray pipeline");
        }

        // Diffuse ray pipeline
        pipeline_info.stage.module = shader_diffuse;
        if (vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_diffuse) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createPipelines): Failed to create diffuse ray pipeline");
        }

        // Camera ray pipeline
        pipeline_info.stage.module = shader_camera;
        if (vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_camera) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createPipelines): Failed to create camera ray pipeline");
        }

        // Pixel label pipeline
        pipeline_info.stage.module = shader_pixel_label;
        if (vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_pixel_label) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createPipelines): Failed to create pixel label pipeline");
        }

        // Cleanup shader modules (no longer needed after pipeline creation)
        vkDestroyShaderModule(vk_device, shader_direct, nullptr);
        vkDestroyShaderModule(vk_device, shader_diffuse, nullptr);
        vkDestroyShaderModule(vk_device, shader_camera, nullptr);
        vkDestroyShaderModule(vk_device, shader_pixel_label, nullptr);

        std::cout << "Compute pipelines created successfully." << std::endl;
    }

    VkShaderModule VulkanComputeBackend::loadShader(const std::string &filename) {
        // Read SPIR-V file
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            helios_runtime_error("ERROR (VulkanComputeBackend::loadShader): Failed to open shader file: " + filename);
        }

        size_t file_size = file.tellg();
        if (file_size == 0) {
            helios_runtime_error("ERROR (VulkanComputeBackend::loadShader): Shader file is empty: " + filename);
        }
        if (file_size % 4 != 0) {
            helios_runtime_error("ERROR (VulkanComputeBackend::loadShader): Invalid SPIR-V file size (not multiple of 4 bytes): " + filename);
        }

        std::vector<uint32_t> code(file_size / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char *>(code.data()), file_size);
        file.close();

        // Validate SPIR-V magic number (0x07230203)
        if (code.empty() || code[0] != 0x07230203) {
            helios_runtime_error("ERROR (VulkanComputeBackend::loadShader): Invalid SPIR-V magic number in: " + filename + ". Expected 0x07230203, got 0x" + std::to_string(code.empty() ? 0 : code[0]));
        }

        // Create shader module
        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size() * sizeof(uint32_t);
        create_info.pCode = code.data();

        VkShaderModule shader_module;
        VkResult result = vkCreateShaderModule(device->getDevice(), &create_info, nullptr, &shader_module);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::loadShader): Failed to create shader module from: " + filename + " (VkResult: " + std::to_string(result) + ")");
        }

        return shader_module;
    }

} // namespace helios
