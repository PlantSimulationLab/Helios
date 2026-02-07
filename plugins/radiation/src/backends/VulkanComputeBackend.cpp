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
        destroyBuffer(object_subdivisions_buffer);
        destroyBuffer(twosided_flag_buffer);
        destroyBuffer(patch_vertices_buffer);
        destroyBuffer(triangle_vertices_buffer);
        destroyBuffer(source_positions_buffer);
        destroyBuffer(source_types_buffer);
        destroyBuffer(source_rotations_buffer);
        destroyBuffer(source_widths_buffer);
        destroyBuffer(source_fluxes_buffer);
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

        // Upload object subdivisions
        if (!geometry.object_subdivisions.empty()) {
            if (geometry.object_subdivisions.size() != primitive_count) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateGeometry): object_subdivisions size mismatch. Expected " + std::to_string(primitive_count) + " entries, got " +
                                     std::to_string(geometry.object_subdivisions.size()));
            }

            if (object_subdivisions_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(object_subdivisions_buffer);
            }
            object_subdivisions_buffer = createBuffer(geometry.object_subdivisions.size() * sizeof(helios::int2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(object_subdivisions_buffer, geometry.object_subdivisions.data(), geometry.object_subdivisions.size() * sizeof(helios::int2));
        }

        // Upload twosided flags
        if (!geometry.twosided_flags.empty()) {
            if (geometry.twosided_flags.size() != primitive_count) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateGeometry): twosided_flags size mismatch. Expected " + std::to_string(primitive_count) + " entries, got " +
                                     std::to_string(geometry.twosided_flags.size()));
            }

            if (twosided_flag_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(twosided_flag_buffer);
            }
            // Convert char to uint for GPU (easier access)
            std::vector<uint> twosided_uint(geometry.twosided_flags.begin(), geometry.twosided_flags.end());
            twosided_flag_buffer = createBuffer(twosided_uint.size() * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(twosided_flag_buffer, twosided_uint.data(), twosided_uint.size() * sizeof(uint));
        }

        // Upload patch vertices
        if (!geometry.patches.vertices.empty()) {
            if (patch_vertices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(patch_vertices_buffer);
            }
            patch_vertices_buffer = createBuffer(geometry.patches.vertices.size() * sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(patch_vertices_buffer, geometry.patches.vertices.data(), geometry.patches.vertices.size() * sizeof(helios::vec3));
        }

        // Upload triangle vertices
        if (!geometry.triangles.vertices.empty()) {
            if (triangle_vertices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(triangle_vertices_buffer);
            }
            triangle_vertices_buffer = createBuffer(geometry.triangles.vertices.size() * sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(triangle_vertices_buffer, geometry.triangles.vertices.data(), geometry.triangles.vertices.size() * sizeof(helios::vec3));
        }

        descriptors_dirty = true;  // Geometry changed, need descriptor update
        std::cout << "Geometry uploaded successfully." << std::endl;
    }

    void VulkanComputeBackend::buildAccelerationStructure() {
        // No-op: BVH is built in updateGeometry()
    }

    // ========== Stub implementations (to be completed in later phases) ==========

    void VulkanComputeBackend::updateMaterials(const RayTracingMaterial &materials) {
        band_count = materials.num_bands;

        if (primitive_count == 0) {
            return; // No geometry uploaded yet
        }

        size_t expected_size = primitive_count * band_count;

        // Upload reflectivity buffer
        if (!materials.reflectivity.empty()) {
            if (materials.reflectivity.size() != expected_size) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateMaterials): reflectivity size mismatch. Expected " + std::to_string(expected_size) + " entries (Nprims * Nbands), got " +
                                     std::to_string(materials.reflectivity.size()));
            }

            if (reflectivity_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(reflectivity_buffer);
            }
            reflectivity_buffer = createBuffer(materials.reflectivity.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(reflectivity_buffer, materials.reflectivity.data(), materials.reflectivity.size() * sizeof(float));
        }

        // Upload transmissivity buffer
        if (!materials.transmissivity.empty()) {
            if (materials.transmissivity.size() != expected_size) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateMaterials): transmissivity size mismatch. Expected " + std::to_string(expected_size) + " entries (Nprims * Nbands), got " +
                                     std::to_string(materials.transmissivity.size()));
            }

            if (transmissivity_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(transmissivity_buffer);
            }
            transmissivity_buffer = createBuffer(materials.transmissivity.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(transmissivity_buffer, materials.transmissivity.data(), materials.transmissivity.size() * sizeof(float));
        }

        descriptors_dirty = true;  // Materials changed, need descriptor update
        std::cout << "Materials uploaded successfully (" << band_count << " bands)." << std::endl;
    }

    void VulkanComputeBackend::updateSources(const std::vector<RayTracingSource> &sources) {
        source_count = sources.size();

        if (source_count == 0) {
            return; // No sources
        }

        // Extract source data from structs
        std::vector<helios::vec3> positions;
        std::vector<uint> types;
        std::vector<helios::vec3> rotations;
        std::vector<helios::vec2> widths;

        positions.reserve(source_count);
        types.reserve(source_count);
        rotations.reserve(source_count);
        widths.reserve(source_count);

        for (const auto &source : sources) {
            positions.push_back(source.position);
            types.push_back(source.type);
            rotations.push_back(source.rotation);
            widths.push_back(source.width);
        }

        // Upload source positions
        if (source_positions_buffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(source_positions_buffer);
        }
        source_positions_buffer = createBuffer(positions.size() * sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadBufferData(source_positions_buffer, positions.data(), positions.size() * sizeof(helios::vec3));

        // Upload source types
        if (source_types_buffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(source_types_buffer);
        }
        source_types_buffer = createBuffer(types.size() * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadBufferData(source_types_buffer, types.data(), types.size() * sizeof(uint));

        // Upload source rotations
        if (source_rotations_buffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(source_rotations_buffer);
        }
        source_rotations_buffer = createBuffer(rotations.size() * sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadBufferData(source_rotations_buffer, rotations.data(), rotations.size() * sizeof(helios::vec3));

        // Upload source widths
        if (source_widths_buffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(source_widths_buffer);
        }
        source_widths_buffer = createBuffer(widths.size() * sizeof(helios::vec2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadBufferData(source_widths_buffer, widths.data(), widths.size() * sizeof(helios::vec2));

        std::cout << "Sources uploaded successfully (" << source_count << " sources)." << std::endl;
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
        if (primitive_count == 0 || source_count == 0) {
            return; // No geometry or sources
        }

        // Update descriptor sets only if buffers changed
        if (descriptors_dirty) {
            updateDescriptorSets();
            descriptors_dirty = false;
        }

        VkDevice vk_device = device->getDevice();

        // Record command buffer
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        // Bind pipeline
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_direct);

        // Bind descriptor sets
        VkDescriptorSet sets[] = {set_geometry, set_materials, set_results};
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 3, sets, 0, nullptr);

        // Push constants
        struct PushConstants {
            uint launch_offset;
            uint launch_count;
            uint rays_per_primitive;
            uint random_seed;
            uint current_band;
            uint band_count;
            uint source_count;
            uint primitive_count;
        } push_constants;

        push_constants.launch_offset = params.launch_offset;
        push_constants.launch_count = params.launch_count;
        push_constants.rays_per_primitive = params.rays_per_primitive;
        push_constants.random_seed = params.random_seed;
        push_constants.current_band = params.current_band;
        push_constants.band_count = band_count;
        push_constants.source_count = source_count;
        push_constants.primitive_count = primitive_count;

        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);

        // Dispatch compute shader
        // Workgroup size = 256 threads (from shader local_size_x)
        uint workgroup_count = (params.launch_count + 255) / 256;
        vkCmdDispatch(command_buffer, workgroup_count, 1, 1);

        // Memory barrier to ensure writes complete before next launch or readback
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        vkEndCommandBuffer(command_buffer);

        // Submit command buffer with fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkResetFences(vk_device, 1, &transfer_fence);
        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, transfer_fence);

        // Wait for completion
        vkWaitForFences(vk_device, 1, &transfer_fence, VK_TRUE, UINT64_MAX);
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
        if (primitive_count == 0 || band_count == 0) {
            return; // No results to download
        }

        size_t buffer_size = primitive_count * band_count;

        // Resize output vectors
        results.radiation_in.resize(buffer_size);
        results.radiation_out_top.resize(buffer_size);
        results.radiation_out_bottom.resize(buffer_size);
        results.num_primitives = primitive_count;
        results.num_bands = band_count;

        // Download radiation_in buffer
        if (radiation_in_buffer.buffer != VK_NULL_HANDLE) {
            downloadBufferData(radiation_in_buffer, results.radiation_in.data(), buffer_size * sizeof(float));
        } else {
            std::fill(results.radiation_in.begin(), results.radiation_in.end(), 0.0f);
        }

        // Phase 1: radiation_out and scatter buffers not used (no emission, no scattering)
        std::fill(results.radiation_out_top.begin(), results.radiation_out_top.end(), 0.0f);
        std::fill(results.radiation_out_bottom.begin(), results.radiation_out_bottom.end(), 0.0f);

        // Initialize scatter buffers (needed by RadiationModel even if not used in Phase 1)
        results.scatter_buff_top.resize(buffer_size, 0.0f);
        results.scatter_buff_bottom.resize(buffer_size, 0.0f);
        results.scatter_buff_top_cam.resize(buffer_size, 0.0f);
        results.scatter_buff_bottom_cam.resize(buffer_size, 0.0f);
    }

    void VulkanComputeBackend::getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels, std::vector<float> &pixel_depths, uint camera_id, const helios::int2 &resolution) {
        // TODO: Download camera buffers in Phase 7
    }

    void VulkanComputeBackend::zeroRadiationBuffers(size_t launch_band_count) {
        if (primitive_count == 0 || band_count == 0) {
            return; // No geometry or bands
        }

        size_t buffer_size = primitive_count * band_count;

        // Create or resize radiation_in buffer
        if (radiation_in_buffer.buffer == VK_NULL_HANDLE || radiation_in_buffer.size != buffer_size * sizeof(float)) {
            if (radiation_in_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_in_buffer);
            }
            radiation_in_buffer = createBuffer(buffer_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }

        // Create or resize radiation_out buffer
        if (radiation_out_buffer.buffer == VK_NULL_HANDLE || radiation_out_buffer.size != buffer_size * sizeof(float)) {
            if (radiation_out_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_buffer);
            }
            radiation_out_buffer = createBuffer(buffer_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }

        // Zero both buffers
        zeroBuffer(radiation_in_buffer);
        zeroBuffer(radiation_out_buffer);

        descriptors_dirty = true;  // Result buffers created/changed
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
        if (fluxes.empty() || source_count == 0) {
            return; // No fluxes or sources
        }

        // Fluxes are indexed by [source * Nbands_launch + band]
        // For Phase 1, we expect Nsources * Nbands_launch entries
        size_t expected_size = source_count * band_count; // Assuming single-band launch for Phase 1

        if (fluxes.size() != expected_size && fluxes.size() != source_count) {
            // Allow single-band upload (size = Nsources) or full upload (size = Nsources * Nbands)
            if (fluxes.size() != source_count) {
                helios_runtime_error("ERROR (VulkanComputeBackend::uploadSourceFluxes): fluxes size mismatch. Expected " + std::to_string(source_count) + " (single band) or " +
                                     std::to_string(expected_size) + " (all bands), got " + std::to_string(fluxes.size()));
            }
        }

        if (source_fluxes_buffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(source_fluxes_buffer);
        }
        source_fluxes_buffer = createBuffer(fluxes.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadBufferData(source_fluxes_buffer, fluxes.data(), fluxes.size() * sizeof(float));
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
            {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Object subdivisions
            {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Twosided flags
            {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Patch vertices
            {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // Triangle vertices
            // TODO Phase 2+: Add disk, tile, voxel, bbox vertices, mask data, UV data
        };

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(geometry_bindings.size());
        layout_info.pBindings = geometry_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_geometry) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create geometry descriptor set layout");
        }

        // Set 1: Material/Source buffers (changes per simulation)
        std::vector<VkDescriptorSetLayoutBinding> material_bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Source positions
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Source types
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Source rotations
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Source widths
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Source fluxes
            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Reflectivity
            {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Transmissivity
            // TODO Phase 2+: Add diffuse params, specular
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
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 25}, // Enough for all 3 sets (10 geometry + 7 material + 4 result + margin)
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

    void VulkanComputeBackend::updateDescriptorSets() {
        VkDevice vk_device = device->getDevice();

        std::vector<VkWriteDescriptorSet> descriptor_writes;

        // ========== Set 0: Geometry Buffers ==========

        VkDescriptorBufferInfo bvh_info{};
        bvh_info.buffer = bvh_buffer.buffer;
        bvh_info.offset = 0;
        bvh_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo prim_indices_info{};
        prim_indices_info.buffer = primitive_indices_buffer.buffer;
        prim_indices_info.offset = 0;
        prim_indices_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo transform_info{};
        transform_info.buffer = transform_matrices_buffer.buffer;
        transform_info.offset = 0;
        transform_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo prim_types_info{};
        prim_types_info.buffer = primitive_types_buffer.buffer;
        prim_types_info.offset = 0;
        prim_types_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo prim_uuids_info{};
        prim_uuids_info.buffer = primitive_uuids_buffer.buffer;
        prim_uuids_info.offset = 0;
        prim_uuids_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo prim_positions_info{};
        prim_positions_info.buffer = primitive_positions_buffer.buffer;
        prim_positions_info.offset = 0;
        prim_positions_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo obj_subdivisions_info{};
        obj_subdivisions_info.buffer = object_subdivisions_buffer.buffer;
        obj_subdivisions_info.offset = 0;
        obj_subdivisions_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo twosided_info{};
        twosided_info.buffer = twosided_flag_buffer.buffer;
        twosided_info.offset = 0;
        twosided_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo patch_vertices_info{};
        patch_vertices_info.buffer = patch_vertices_buffer.buffer;
        patch_vertices_info.offset = 0;
        patch_vertices_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo triangle_vertices_info{};
        triangle_vertices_info.buffer = triangle_vertices_buffer.buffer;
        triangle_vertices_info.offset = 0;
        triangle_vertices_info.range = VK_WHOLE_SIZE;

        // Only add descriptor writes for non-null buffers
        if (bvh_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bvh_info;
            descriptor_writes.push_back(write);
        }

        if (primitive_indices_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 1;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &prim_indices_info;
            descriptor_writes.push_back(write);
        }

        if (transform_matrices_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 2;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &transform_info;
            descriptor_writes.push_back(write);
        }

        if (primitive_types_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 3;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &prim_types_info;
            descriptor_writes.push_back(write);
        }

        if (primitive_uuids_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 4;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &prim_uuids_info;
            descriptor_writes.push_back(write);
        }

        if (primitive_positions_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 5;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &prim_positions_info;
            descriptor_writes.push_back(write);
        }

        if (object_subdivisions_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 6;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &obj_subdivisions_info;
            descriptor_writes.push_back(write);
        }

        if (twosided_flag_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 7;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &twosided_info;
            descriptor_writes.push_back(write);
        }

        if (patch_vertices_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 8;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &patch_vertices_info;
            descriptor_writes.push_back(write);
        }

        if (triangle_vertices_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 9;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &triangle_vertices_info;
            descriptor_writes.push_back(write);
        }

        // ========== Set 1: Material/Source Buffers ==========

        VkDescriptorBufferInfo source_pos_info{};
        source_pos_info.buffer = source_positions_buffer.buffer;
        source_pos_info.offset = 0;
        source_pos_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo source_types_info{};
        source_types_info.buffer = source_types_buffer.buffer;
        source_types_info.offset = 0;
        source_types_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo source_rot_info{};
        source_rot_info.buffer = source_rotations_buffer.buffer;
        source_rot_info.offset = 0;
        source_rot_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo source_widths_info{};
        source_widths_info.buffer = source_widths_buffer.buffer;
        source_widths_info.offset = 0;
        source_widths_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo source_fluxes_info{};
        source_fluxes_info.buffer = source_fluxes_buffer.buffer;
        source_fluxes_info.offset = 0;
        source_fluxes_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo reflectivity_info{};
        reflectivity_info.buffer = reflectivity_buffer.buffer;
        reflectivity_info.offset = 0;
        reflectivity_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo transmissivity_info{};
        transmissivity_info.buffer = transmissivity_buffer.buffer;
        transmissivity_info.offset = 0;
        transmissivity_info.range = VK_WHOLE_SIZE;

        if (source_positions_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &source_pos_info;
            descriptor_writes.push_back(write);
        }

        if (source_types_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 1;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &source_types_info;
            descriptor_writes.push_back(write);
        }

        if (source_rotations_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 2;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &source_rot_info;
            descriptor_writes.push_back(write);
        }

        if (source_widths_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 3;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &source_widths_info;
            descriptor_writes.push_back(write);
        }

        if (source_fluxes_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 4;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &source_fluxes_info;
            descriptor_writes.push_back(write);
        }

        if (reflectivity_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 5;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &reflectivity_info;
            descriptor_writes.push_back(write);
        }

        if (transmissivity_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 6;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &transmissivity_info;
            descriptor_writes.push_back(write);
        }

        // ========== Set 2: Result Buffers ==========

        VkDescriptorBufferInfo rad_in_info{};
        rad_in_info.buffer = radiation_in_buffer.buffer;
        rad_in_info.offset = 0;
        rad_in_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo rad_out_info{};
        rad_out_info.buffer = radiation_out_buffer.buffer;
        rad_out_info.offset = 0;
        rad_out_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo scatter_top_info{};
        scatter_top_info.buffer = scatter_top_buffer.buffer;
        scatter_top_info.offset = 0;
        scatter_top_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo scatter_bottom_info{};
        scatter_bottom_info.buffer = scatter_bottom_buffer.buffer;
        scatter_bottom_info.offset = 0;
        scatter_bottom_info.range = VK_WHOLE_SIZE;

        if (radiation_in_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &rad_in_info;
            descriptor_writes.push_back(write);
        }

        if (radiation_out_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 1;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &rad_out_info;
            descriptor_writes.push_back(write);
        }

        if (scatter_top_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 2;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &scatter_top_info;
            descriptor_writes.push_back(write);
        }

        if (scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 3;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &scatter_bottom_info;
            descriptor_writes.push_back(write);
        }

        // Apply all descriptor writes
        if (!descriptor_writes.empty()) {
            vkUpdateDescriptorSets(vk_device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
        }
    }

} // namespace helios
