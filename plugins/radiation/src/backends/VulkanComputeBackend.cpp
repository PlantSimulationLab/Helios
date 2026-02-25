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

    VulkanComputeBackend::VulkanComputeBackend()
        : device(new VulkanDevice()), owns_device(true) {
        // Production mode: own the device
    }

    VulkanComputeBackend::VulkanComputeBackend(VulkanDevice* external_device)
        : device(external_device), owns_device(false) {
        // Test mode: borrow pre-initialized device from test singleton
        if (!device) {
            helios_runtime_error("ERROR (VulkanComputeBackend): external_device cannot be nullptr");
        }
    }

    VulkanComputeBackend::~VulkanComputeBackend() {
        shutdown();
        if (owns_device) {
            delete device;
        }
    }

    void VulkanComputeBackend::initialize() {
        // Determine whether to enable validation layers
        // Default: disabled for tests (20-30% faster), enabled in debug builds
        // Can be explicitly controlled via HELIOS_VULKAN_VALIDATION environment variable
        bool enable_validation = false;

        #ifndef NDEBUG
            // Debug builds: enable validation by default
            enable_validation = true;
        #endif

        // Allow environment variable override
        const char* validation_env = std::getenv("HELIOS_VULKAN_VALIDATION");
        if (validation_env != nullptr) {
            std::string val(validation_env);
            if (val == "1" || val == "true" || val == "TRUE") {
                enable_validation = true;
            } else if (val == "0" || val == "false" || val == "FALSE") {
                enable_validation = false;
            }
        }

        // Initialize Vulkan device (only if we own it - shared device already initialized)
        if (owns_device) {
            device->initialize(enable_validation);
        }

        // Create command resources
        createCommandResources();

        // Create descriptor sets and pipelines
        createDescriptorSets();
        createPipelines();
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
        if (set_layout_sky != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(vk_device, set_layout_sky, nullptr);
        if (set_layout_debug != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(vk_device, set_layout_debug, nullptr);

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
        destroyBuffer(normal_buffer);
        destroyBuffer(mask_data_buffer);
        destroyBuffer(mask_sizes_buffer);
        destroyBuffer(mask_offsets_buffer);
        destroyBuffer(mask_IDs_buffer);
        destroyBuffer(uv_data_buffer);
        destroyBuffer(uv_IDs_buffer);
        destroyBuffer(source_positions_buffer);
        destroyBuffer(source_types_buffer);
        destroyBuffer(source_rotations_buffer);
        destroyBuffer(source_widths_buffer);
        destroyBuffer(source_fluxes_buffer);
        destroyBuffer(source_fluxes_cam_buffer);
        destroyBuffer(reflectivity_buffer);
        destroyBuffer(transmissivity_buffer);
        destroyBuffer(specular_exponent_buffer);
        destroyBuffer(specular_scale_buffer);
        destroyBuffer(radiation_in_buffer);
        destroyBuffer(radiation_out_top_buffer);
        destroyBuffer(radiation_out_bottom_buffer);
        destroyBuffer(scatter_top_buffer);
        destroyBuffer(scatter_bottom_buffer);
        destroyBuffer(camera_radiation_buffer);
        destroyBuffer(camera_pixel_label_buffer);
        destroyBuffer(camera_pixel_depth_buffer);
        destroyBuffer(camera_scatter_top_buffer);
        destroyBuffer(camera_scatter_bottom_buffer);
        destroyBuffer(radiation_specular_buffer);
        destroyBuffer(diffuse_flux_buffer);
        destroyBuffer(diffuse_peak_dir_buffer);
        destroyBuffer(diffuse_extinction_buffer);
        destroyBuffer(diffuse_dist_norm_buffer);
        destroyBuffer(sky_radiance_params_buffer);
        destroyBuffer(camera_sky_radiance_buffer);
        destroyBuffer(solar_disk_radiance_buffer);
        destroyBuffer(debug_counters_buffer);
        destroyBuffer(bbox_vertices_buffer);

        // Destroy command resources
        if (transfer_fence != VK_NULL_HANDLE)
            vkDestroyFence(vk_device, transfer_fence, nullptr);
        if (compute_fence != VK_NULL_HANDLE)
            vkDestroyFence(vk_device, compute_fence, nullptr);
        if (timestamp_query_pool != VK_NULL_HANDLE)
            vkDestroyQueryPool(vk_device, timestamp_query_pool, nullptr);
        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(vk_device, command_pool, nullptr);

        // Only shutdown device if we own it (shared device managed by singleton)
        if (owns_device) {
            device->shutdown();
        }
    }

    void VulkanComputeBackend::updateGeometry(const RayTracingGeometry &geometry) {
        validateGeometryBeforeUpload(geometry);

        primitive_count = geometry.primitive_count;

        if (primitive_count == 0) {
            return; // Empty geometry
        }

        // Build BVH2 on CPU, then convert to CWBVH (8-wide with quantized AABBs)
        bvh_nodes = bvh_builder.build(geometry);
        std::vector<CWBVH_Node> cwbvh_nodes = bvh_builder.convertToCWBVH(bvh_nodes);

        // Upload CWBVH to GPU
        if (!cwbvh_nodes.empty()) {
            if (bvh_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(bvh_buffer);
            }
            bvh_buffer = createBuffer(cwbvh_nodes.size() * sizeof(CWBVH_Node), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(bvh_buffer, cwbvh_nodes.data(), cwbvh_nodes.size() * sizeof(CWBVH_Node));
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

        // Build pre-transformed (world-space) vertex buffers indexed by global primitive index.
        // Vertices are transformed from canonical local space to world space using each primitive's
        // transform matrix. This eliminates per-ray matrix loads and transform_point() calls in the
        // BVH traversal shader inner loop — the single biggest GPU performance optimization.
        {
            static const helios::vec3 canonical_quad[4] = {
                {-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f},
                {0.5f, 0.5f, 0.f}, {-0.5f, 0.5f, 0.f}
            };
            static const helios::vec3 canonical_tri[3] = {
                {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}
            };

            // Patch/tile vertex buffer: 4 vec3s per primitive slot, indexed by global prim_idx
            std::vector<helios::vec3> patch_verts(primitive_count * 4, helios::make_vec3(0.f, 0.f, 0.f));
            // Triangle vertex buffer: 3 vec3s per primitive slot, indexed by global prim_idx
            std::vector<helios::vec3> tri_verts(primitive_count * 3, helios::make_vec3(0.f, 0.f, 0.f));

            for (size_t i = 0; i < primitive_count; ++i) {
                uint prim_type = geometry.primitive_types[i];
                const float *transform = &geometry.transform_matrices[i * 16];

                if (prim_type == 0 || prim_type == 3) { // Patch or Tile
                    for (int v = 0; v < 4; ++v) {
                        const helios::vec3 &p = canonical_quad[v];
                        helios::vec3 world_v;
                        world_v.x = transform[0] * p.x + transform[1] * p.y + transform[2] * p.z + transform[3];
                        world_v.y = transform[4] * p.x + transform[5] * p.y + transform[6] * p.z + transform[7];
                        world_v.z = transform[8] * p.x + transform[9] * p.y + transform[10] * p.z + transform[11];
                        patch_verts[i * 4 + v] = world_v;
                    }
                } else if (prim_type == 1) { // Triangle
                    for (int v = 0; v < 3; ++v) {
                        const helios::vec3 &p = canonical_tri[v];
                        helios::vec3 world_v;
                        world_v.x = transform[0] * p.x + transform[1] * p.y + transform[2] * p.z + transform[3];
                        world_v.y = transform[4] * p.x + transform[5] * p.y + transform[6] * p.z + transform[7];
                        world_v.z = transform[8] * p.x + transform[9] * p.y + transform[10] * p.z + transform[11];
                        tri_verts[i * 3 + v] = world_v;
                    }
                }
            }

            if (patch_vertices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(patch_vertices_buffer);
            }
            patch_vertices_buffer = createBuffer(patch_verts.size() * sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(patch_vertices_buffer, patch_verts.data(), patch_verts.size() * sizeof(helios::vec3));

            if (triangle_vertices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(triangle_vertices_buffer);
            }
            triangle_vertices_buffer = createBuffer(tri_verts.size() * sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(triangle_vertices_buffer, tri_verts.data(), tri_verts.size() * sizeof(helios::vec3));
        }

        // Pre-compute world-space normals for each primitive.
        // Eliminates per-thread get_patch_normal() / get_triangle_normal() (3 transform_point + cross product)
        // in both direct and diffuse shaders, and also eliminates the per-hit normal computation in diffuse.
        {
            static const helios::vec3 patch_v0 = {0.f, 0.f, 0.f};
            static const helios::vec3 patch_v1 = {1.f, 0.f, 0.f};
            static const helios::vec3 patch_v2 = {0.f, 1.f, 0.f};
            static const helios::vec3 tri_v0 = {0.f, 0.f, 0.f};
            static const helios::vec3 tri_v1 = {0.f, 1.f, 0.f};
            static const helios::vec3 tri_v2 = {1.f, 1.f, 0.f};

            std::vector<helios::vec3> normals(primitive_count);

            for (size_t i = 0; i < primitive_count; ++i) {
                const float *t = &geometry.transform_matrices[i * 16];
                uint prim_type = geometry.primitive_types[i];

                // Choose canonical vertices based on primitive type
                helios::vec3 cv0, cv1, cv2;
                if (prim_type == 0 || prim_type == 2 || prim_type == 3) { // Patch, Disk, or Tile
                    cv0 = patch_v0;
                    cv1 = patch_v1;
                    cv2 = patch_v2;
                } else if (prim_type == 1) { // Triangle
                    cv0 = tri_v0;
                    cv1 = tri_v1;
                    cv2 = tri_v2;
                } else {
                    normals[i] = helios::make_vec3(0.f, 0.f, 1.f);
                    continue;
                }

                // Transform canonical vertices to world space
                helios::vec3 w0, w1, w2;
                w0.x = t[0] * cv0.x + t[1] * cv0.y + t[2] * cv0.z + t[3];
                w0.y = t[4] * cv0.x + t[5] * cv0.y + t[6] * cv0.z + t[7];
                w0.z = t[8] * cv0.x + t[9] * cv0.y + t[10] * cv0.z + t[11];
                w1.x = t[0] * cv1.x + t[1] * cv1.y + t[2] * cv1.z + t[3];
                w1.y = t[4] * cv1.x + t[5] * cv1.y + t[6] * cv1.z + t[7];
                w1.z = t[8] * cv1.x + t[9] * cv1.y + t[10] * cv1.z + t[11];
                w2.x = t[0] * cv2.x + t[1] * cv2.y + t[2] * cv2.z + t[3];
                w2.y = t[4] * cv2.x + t[5] * cv2.y + t[6] * cv2.z + t[7];
                w2.z = t[8] * cv2.x + t[9] * cv2.y + t[10] * cv2.z + t[11];

                // cross(w1 - w0, w2 - w0) then normalize
                helios::vec3 e1 = w1 - w0;
                helios::vec3 e2 = w2 - w0;
                helios::vec3 n = cross(e1, e2);
                float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
                if (len > 1e-8f) {
                    normals[i] = n / len;
                } else {
                    normals[i] = helios::make_vec3(0.f, 0.f, 1.f);
                }
            }

            if (normal_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(normal_buffer);
            }
            normal_buffer = createBuffer(normals.size() * sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(normal_buffer, normals.data(), normals.size() * sizeof(helios::vec3));
        }

        // Upload texture mask and UV data
        {
            // Convert mask_data from vector<bool> to flat uint32_t array (0 or 1 per texel)
            std::vector<uint32_t> mask_offsets;
            uint32_t current_offset = 0;
            for (const auto &sz : geometry.mask_sizes) {
                mask_offsets.push_back(current_offset);
                current_offset += static_cast<uint32_t>(sz.x) * static_cast<uint32_t>(sz.y);
            }

            std::vector<uint32_t> mask_data_uint;
            if (current_offset > 0) {
                if (geometry.mask_data.size() < current_offset) {
                    helios_runtime_error("ERROR (VulkanComputeBackend::updateGeometry): mask_data size mismatch. Expected " + std::to_string(current_offset) + " texels, got " +
                                        std::to_string(geometry.mask_data.size()));
                }
                mask_data_uint.resize(current_offset, 0);
                for (size_t i = 0; i < current_offset; ++i) {
                    mask_data_uint[i] = geometry.mask_data[i] ? 1 : 0;
                }

            }

            // Reformat UV data: allocate 4 vec2 per primitive (flat array)
            std::vector<helios::vec2> uv_flat(primitive_count * 4, helios::make_vec2(0.f, 0.f));
            size_t uv_read_offset = 0;
            for (size_t p = 0; p < primitive_count; ++p) {
                if (!geometry.uv_IDs.empty() && geometry.uv_IDs[p] >= 0) {
                    // This primitive has custom UVs - read next 4 from uv_data
                    for (int v = 0; v < 4 && uv_read_offset < geometry.uv_data.size(); ++v) {
                        uv_flat[p * 4 + v] = geometry.uv_data[uv_read_offset++];
                    }
                }
                // If uv_IDs[p] == -1, keep default zeros (unused, will use parametric UVs in shader)
            }


            // Upload mask_data_buffer
            if (mask_data_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(mask_data_buffer);
            }
            if (!mask_data_uint.empty()) {
                mask_data_buffer = createBuffer(mask_data_uint.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                uploadBufferData(mask_data_buffer, mask_data_uint.data(), mask_data_uint.size() * sizeof(uint32_t));
            } else {
                // Keep placeholder buffer if no mask data
                mask_data_buffer = createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            }

            // Upload mask_sizes_buffer
            if (mask_sizes_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(mask_sizes_buffer);
            }
            if (!geometry.mask_sizes.empty()) {
                mask_sizes_buffer = createBuffer(geometry.mask_sizes.size() * sizeof(helios::int2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                uploadBufferData(mask_sizes_buffer, geometry.mask_sizes.data(), geometry.mask_sizes.size() * sizeof(helios::int2));
            } else {
                mask_sizes_buffer = createBuffer(sizeof(int32_t) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            }

            // Upload mask_offsets_buffer
            if (mask_offsets_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(mask_offsets_buffer);
            }
            if (!mask_offsets.empty()) {
                mask_offsets_buffer = createBuffer(mask_offsets.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                uploadBufferData(mask_offsets_buffer, mask_offsets.data(), mask_offsets.size() * sizeof(uint32_t));
            } else {
                mask_offsets_buffer = createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            }

            // Upload mask_IDs_buffer
            if (mask_IDs_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(mask_IDs_buffer);
            }
            if (!geometry.mask_IDs.empty()) {
                // Convert to int32_t
                std::vector<int32_t> mask_IDs_int32(geometry.mask_IDs.begin(), geometry.mask_IDs.end());
                mask_IDs_buffer = createBuffer(mask_IDs_int32.size() * sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                uploadBufferData(mask_IDs_buffer, mask_IDs_int32.data(), mask_IDs_int32.size() * sizeof(int32_t));
            } else {
                mask_IDs_buffer = createBuffer(sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            }

            // Upload uv_data_buffer
            if (uv_data_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(uv_data_buffer);
            }
            if (!uv_flat.empty()) {
                uv_data_buffer = createBuffer(uv_flat.size() * sizeof(helios::vec2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                uploadBufferData(uv_data_buffer, uv_flat.data(), uv_flat.size() * sizeof(helios::vec2));
            } else {
                uv_data_buffer = createBuffer(sizeof(float) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            }

            // Upload uv_IDs_buffer
            if (uv_IDs_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(uv_IDs_buffer);
            }
            if (!geometry.uv_IDs.empty()) {
                // Convert to int32_t
                std::vector<int32_t> uv_IDs_int32(geometry.uv_IDs.begin(), geometry.uv_IDs.end());
                uv_IDs_buffer = createBuffer(uv_IDs_int32.size() * sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                uploadBufferData(uv_IDs_buffer, uv_IDs_int32.data(), uv_IDs_int32.size() * sizeof(int32_t));
            } else {
                uv_IDs_buffer = createBuffer(sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            }
        }

        // Upload periodic boundary bbox data
        {
            bbox_count = geometry.bbox_count;
            periodic_flag_x = geometry.periodic_flag.x;
            periodic_flag_y = geometry.periodic_flag.y;

            if (bbox_vertices_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(bbox_vertices_buffer);
            }

            if (bbox_count > 0 && !geometry.bboxes.vertices.empty()) {
                // Flatten vec3 vertices to float array (4 vertices per face, 3 floats each = 12 floats per face)
                std::vector<float> bbox_verts_flat;
                bbox_verts_flat.reserve(bbox_count * 12);
                for (size_t i = 0; i < bbox_count * 4; ++i) {
                    bbox_verts_flat.push_back(geometry.bboxes.vertices[i].x);
                    bbox_verts_flat.push_back(geometry.bboxes.vertices[i].y);
                    bbox_verts_flat.push_back(geometry.bboxes.vertices[i].z);
                }
                bbox_vertices_buffer = createBuffer(bbox_verts_flat.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                uploadBufferData(bbox_vertices_buffer, bbox_verts_flat.data(), bbox_verts_flat.size() * sizeof(float));

                // Compute domain bounds from all bbox vertices (min/max across all faces)
                float xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f;
                for (size_t i = 0; i < bbox_count * 4; ++i) {
                    xmin = std::min(xmin, geometry.bboxes.vertices[i].x);
                    xmax = std::max(xmax, geometry.bboxes.vertices[i].x);
                    ymin = std::min(ymin, geometry.bboxes.vertices[i].y);
                    ymax = std::max(ymax, geometry.bboxes.vertices[i].y);
                }
                domain_bounds[0] = xmin;
                domain_bounds[1] = xmax;
                domain_bounds[2] = ymin;
                domain_bounds[3] = ymax;
            } else {
                // No bboxes - create placeholder buffer
                bbox_vertices_buffer = createBuffer(sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                domain_bounds[0] = domain_bounds[1] = domain_bounds[2] = domain_bounds[3] = 0.f;
            }
        }

        descriptors_dirty = true;  // Geometry changed, need descriptor update
    }

    void VulkanComputeBackend::buildAccelerationStructure() {
        // No-op: BVH is built in updateGeometry()
    }

    void VulkanComputeBackend::updateMaterials(const RayTracingMaterial &materials) {
        band_count = materials.num_bands;

        if (primitive_count == 0) {
            return; // No geometry uploaded yet
        }

        // Material buffers are indexed as [source * Nbands * Nprims + band * Nprims + prim]
        // Use materials.num_sources for validation (source_count may not be set yet if updateSources hasn't been called)
        size_t expected_size = materials.num_sources * band_count * primitive_count;

        // Upload reflectivity buffer
        if (!materials.reflectivity.empty()) {
            if (materials.reflectivity.size() != expected_size) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateMaterials): reflectivity size mismatch. Expected " + std::to_string(expected_size) + " entries (Nsources * Nprims * Nbands), got " +
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
                helios_runtime_error("ERROR (VulkanComputeBackend::updateMaterials): transmissivity size mismatch. Expected " + std::to_string(expected_size) + " entries (Nsources * Nprims * Nbands), got " +
                                     std::to_string(materials.transmissivity.size()));
            }

            if (transmissivity_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(transmissivity_buffer);
            }
            transmissivity_buffer = createBuffer(materials.transmissivity.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(transmissivity_buffer, materials.transmissivity.data(), materials.transmissivity.size() * sizeof(float));
        }

        // Upload specular exponent buffer (per primitive)
        if (!materials.specular_exponent.empty()) {
            if (materials.specular_exponent.size() != primitive_count) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateMaterials): specular_exponent size mismatch. Expected " + std::to_string(primitive_count) + " entries (Nprims), got " +
                                     std::to_string(materials.specular_exponent.size()));
            }

            if (specular_exponent_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(specular_exponent_buffer);
            }
            specular_exponent_buffer = createBuffer(materials.specular_exponent.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(specular_exponent_buffer, materials.specular_exponent.data(), materials.specular_exponent.size() * sizeof(float));
        }

        // Upload specular scale buffer (per primitive)
        if (!materials.specular_scale.empty()) {
            if (materials.specular_scale.size() != primitive_count) {
                helios_runtime_error("ERROR (VulkanComputeBackend::updateMaterials): specular_scale size mismatch. Expected " + std::to_string(primitive_count) + " entries (Nprims), got " +
                                     std::to_string(materials.specular_scale.size()));
            }

            if (specular_scale_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(specular_scale_buffer);
            }
            specular_scale_buffer = createBuffer(materials.specular_scale.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            uploadBufferData(specular_scale_buffer, materials.specular_scale.data(), materials.specular_scale.size() * sizeof(float));
        }

        descriptors_dirty = true;  // Materials changed, need descriptor update
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

    }

    void VulkanComputeBackend::updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm,
                                                      const std::vector<float> &sky_energy) {
        // Intentional no-op: the Vulkan backend uploads diffuse radiation parameters directly
        // in launchDiffuseRays() from the RayTracingLaunchParams struct, rather than caching
        // them here. This method is required by the RayTracingBackend interface but is not
        // called by RadiationModel.
    }

    void VulkanComputeBackend::updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction,
                                              const std::vector<float> &solar_disk_radiance, float solar_disk_cos_angle) {
        // Store sun parameters for push constants
        cached_sun_direction = sun_direction;
        cached_solar_disk_cos_angle = solar_disk_cos_angle;

        // Upload sky_radiance_params to existing buffer (already used by diffuse shader)
        if (!sky_radiance_params.empty()) {
            size_t params_size = sky_radiance_params.size() * sizeof(helios::vec4);
            if (sky_radiance_params_buffer.buffer == VK_NULL_HANDLE || sky_radiance_params_buffer.size != params_size) {
                if (sky_radiance_params_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(sky_radiance_params_buffer);
                }
                sky_radiance_params_buffer = createBuffer(params_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                descriptors_dirty = true;
            }
            uploadBufferData(sky_radiance_params_buffer, sky_radiance_params.data(), params_size);
        }

        // Upload camera_sky_radiance (zenith sky radiance for camera miss shader)
        if (!camera_sky_radiance.empty()) {
            size_t sky_size = camera_sky_radiance.size() * sizeof(float);
            if (camera_sky_radiance_buffer.buffer == VK_NULL_HANDLE || camera_sky_radiance_buffer.size != sky_size) {
                if (camera_sky_radiance_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(camera_sky_radiance_buffer);
                }
                camera_sky_radiance_buffer = createBuffer(sky_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                descriptors_dirty = true;
            }
            uploadBufferData(camera_sky_radiance_buffer, camera_sky_radiance.data(), sky_size);
        }

        // Upload solar_disk_radiance (solar disk radiance for camera miss shader)
        if (!solar_disk_radiance.empty()) {
            size_t solar_size = solar_disk_radiance.size() * sizeof(float);
            if (solar_disk_radiance_buffer.buffer == VK_NULL_HANDLE || solar_disk_radiance_buffer.size != solar_size) {
                if (solar_disk_radiance_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(solar_disk_radiance_buffer);
                }
                solar_disk_radiance_buffer = createBuffer(solar_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                descriptors_dirty = true;
            }
            uploadBufferData(solar_disk_radiance_buffer, solar_disk_radiance.data(), solar_size);
        }
    }

    void VulkanComputeBackend::launchDirectRays(const RayTracingLaunchParams &params) {
        if (primitive_count == 0 || source_count == 0) {
            return; // No geometry or sources
        }

        // Build band mapping (same logic as diffuse)
        launch_to_global_band.clear();
        for (uint32_t g = 0; g < band_count; g++) {
            if (!params.band_launch_flag.empty() && params.band_launch_flag[g]) {
                launch_to_global_band.push_back(g);
            }
        }
        if (launch_to_global_band.empty()) {
            for (uint32_t g = 0; g < launch_band_count; g++) {
                launch_to_global_band.push_back(g);
            }
        }

        // Upload band mapping to GPU buffer
        uploadBufferData(band_map_buffer, launch_to_global_band.data(),
                         launch_to_global_band.size() * sizeof(uint32_t));

        // Ensure scatter and radiation_out buffers exist (direct shader writes scatter buffers)
        size_t buf_size = primitive_count * launch_band_count * sizeof(float);
        VkBufferUsageFlags scatter_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (scatter_top_buffer.buffer == VK_NULL_HANDLE || scatter_top_buffer.size != buf_size) {
            if (scatter_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(scatter_top_buffer);
            }
            scatter_top_buffer = createBuffer(buf_size, scatter_usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }
        if (scatter_bottom_buffer.buffer == VK_NULL_HANDLE || scatter_bottom_buffer.size != buf_size) {
            if (scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(scatter_bottom_buffer);
            }
            scatter_bottom_buffer = createBuffer(buf_size, scatter_usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Ensure radiation_out buffers exist (needed for descriptor set completeness)
        VkBufferUsageFlags rad_out_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (radiation_out_top_buffer.buffer == VK_NULL_HANDLE || radiation_out_top_buffer.size != buf_size) {
            if (radiation_out_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_top_buffer);
            }
            radiation_out_top_buffer = createBuffer(buf_size, rad_out_usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(radiation_out_top_buffer);
            descriptors_dirty = true;
        }
        if (radiation_out_bottom_buffer.buffer == VK_NULL_HANDLE || radiation_out_bottom_buffer.size != buf_size) {
            if (radiation_out_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_bottom_buffer);
            }
            radiation_out_bottom_buffer = createBuffer(buf_size, rad_out_usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(radiation_out_bottom_buffer);
            descriptors_dirty = true;
        }

        // Zero scatter buffers before each direct launch (runBand does not zero them before direct)
        zeroBuffer(scatter_top_buffer);
        zeroBuffer(scatter_bottom_buffer);

        // Update descriptor sets only if buffers changed
        if (descriptors_dirty) {
            updateDescriptorSets();
            descriptors_dirty = false;
        }
        VkDevice vk_device = device->getDevice();

        // Record COMPUTE command buffer
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(compute_command_buffer, &begin_info);

        // Bind pipeline
        vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_direct);

        // Bind descriptor sets
        VkDescriptorSet sets[] = {set_geometry, set_materials, set_results};
        vkCmdBindDescriptorSets(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 3, sets, 0, nullptr);

        // Push constants (expanded for 3D dispatch with 2D primitive tiling)
        struct PushConstants {
            uint launch_offset;
            uint launch_count;
            uint rays_per_primitive;
            uint random_seed;
            uint current_band;
            uint band_count;
            uint source_count;
            uint primitive_count;
            uint debug_mode;  // 1 = enable bounds checking, 0 = production
            uint launch_dim_x;    // Grid dimension X for stratified sampling
            uint launch_dim_y;    // Grid dimension Y for stratified sampling
            uint prim_tiles_y;    // Number of primitive tiles in Y dimension
            uint prims_per_tile;  // Primitives per tile (65535 max)
            uint material_band_count;  // Global band count (for material buffers)
            uint periodic_flag_x;   // 1 if periodic in X direction
            uint periodic_flag_y;   // 1 if periodic in Y direction
            uint bbox_count;         // Number of bbox faces (0-4)
            float domain_xmin;       // Domain bounds for periodic wrapping
            float domain_xmax;
            float domain_ymin;
            float domain_ymax;
            uint specular_reflection_enabled;  // 0=disabled, 1=default scale, 2=user scale
        } push_constants;

        // Compute 2D grid dimensions for stratified sampling (matches OptiX)
        uint32_t launch_dim_x = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(params.rays_per_primitive))));
        uint32_t launch_dim_y = launch_dim_x;

        push_constants.launch_offset = params.launch_offset;
        push_constants.launch_count = params.launch_count;
        push_constants.rays_per_primitive = params.rays_per_primitive;
        push_constants.random_seed = params.random_seed;
        push_constants.current_band = params.current_band;
        push_constants.band_count = launch_band_count;  // Use launch band count (not global)
        push_constants.source_count = source_count;
        push_constants.primitive_count = primitive_count;
        push_constants.launch_dim_x = launch_dim_x;
        push_constants.launch_dim_y = launch_dim_y;
        push_constants.material_band_count = band_count;  // Global band count for material indexing

        // Enable debug bounds checking (can be disabled in production builds)
        #ifdef HELIOS_DEBUG
            push_constants.debug_mode = 1;
        #else
            push_constants.debug_mode = 0;
        #endif

        // Periodic boundary parameters
        push_constants.periodic_flag_x = static_cast<uint32_t>(periodic_flag_x);
        push_constants.periodic_flag_y = static_cast<uint32_t>(periodic_flag_y);
        push_constants.bbox_count = bbox_count;
        push_constants.domain_xmin = domain_bounds[0];
        push_constants.domain_xmax = domain_bounds[1];
        push_constants.domain_ymin = domain_bounds[2];
        push_constants.domain_ymax = domain_bounds[3];
        push_constants.specular_reflection_enabled = params.specular_reflection_enabled;

        // 3D dispatch with 2D primitive tiling to avoid sub-batching
        // Tile primitives into Y dimension when count exceeds 65535 to use full Vulkan dispatch space
        const uint32_t WG_X = 8; // Must match shader local_size_x
        const uint32_t WG_Y = 32; // Must match shader local_size_y
        const uint32_t MAX_PRIMS_PER_TILE = 65535;

        uint32_t dispatch_x = (launch_dim_x + WG_X - 1) / WG_X;
        uint32_t dispatch_y_rays = (launch_dim_y + WG_Y - 1) / WG_Y;

        // Compute primitive tiling
        uint32_t prims_per_tile = std::min(params.launch_count, MAX_PRIMS_PER_TILE);
        uint32_t prim_tiles_y = (params.launch_count + MAX_PRIMS_PER_TILE - 1) / MAX_PRIMS_PER_TILE;

        uint32_t dispatch_y = dispatch_y_rays * prim_tiles_y;
        uint32_t dispatch_z = prims_per_tile;

        push_constants.prim_tiles_y = prim_tiles_y;
        push_constants.prims_per_tile = prims_per_tile;

        vkCmdPushConstants(compute_command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(compute_command_buffer, dispatch_x, dispatch_y, dispatch_z);

        // Buffer memory barrier to ensure storage buffer writes are visible for readback
        // CRITICAL: Use buffer-specific barrier instead of global barrier for MoltenVK compatibility
        VkBufferMemoryBarrier buffer_barriers[3];

        // Radiation_in buffer barrier
        buffer_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barriers[0].pNext = nullptr;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        buffer_barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        buffer_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[0].buffer = radiation_in_buffer.buffer;
        buffer_barriers[0].offset = 0;
        buffer_barriers[0].size = VK_WHOLE_SIZE;

        // Scatter_top buffer barrier (direct shader writes scatter)
        buffer_barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barriers[1].pNext = nullptr;
        buffer_barriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        buffer_barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        buffer_barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[1].buffer = scatter_top_buffer.buffer;
        buffer_barriers[1].offset = 0;
        buffer_barriers[1].size = VK_WHOLE_SIZE;

        // Scatter_bottom buffer barrier
        buffer_barriers[2].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barriers[2].pNext = nullptr;
        buffer_barriers[2].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        buffer_barriers[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        buffer_barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[2].buffer = scatter_bottom_buffer.buffer;
        buffer_barriers[2].offset = 0;
        buffer_barriers[2].size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr,  // No global memory barriers
                             3, buffer_barriers,  // Buffer-specific barriers
                             0, nullptr);  // No image barriers

        vkEndCommandBuffer(compute_command_buffer);

        // Submit command buffer with COMPUTE fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &compute_command_buffer;

        vkResetFences(vk_device, 1, &compute_fence);
        VkResult result = vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, compute_fence);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::launchDirectRays): vkQueueSubmit failed. VkResult: " + std::to_string(result));
        }

        // Wait for compute to complete (no timeout - large scenes can take minutes)
        VkResult wait_result;
        do {
            wait_result = vkWaitForFences(vk_device, 1, &compute_fence, VK_TRUE, 1000000000ULL);
            if (wait_result != VK_SUCCESS && wait_result != VK_TIMEOUT) {
                helios_runtime_error("ERROR (VulkanComputeBackend::launchDirectRays): vkWaitForFences failed. VkResult: " + std::to_string(wait_result));
            }
        } while (wait_result == VK_TIMEOUT);
    }

    void VulkanComputeBackend::launchDiffuseRays(const RayTracingLaunchParams &params) {
        if (primitive_count == 0) {
            return; // No geometry
        }

        // Ensure radiation_out_top/bottom buffers exist (required by shader)
        size_t rad_out_size = primitive_count * launch_band_count * sizeof(float);
        if (radiation_out_top_buffer.buffer == VK_NULL_HANDLE || radiation_out_top_buffer.size != rad_out_size) {
            if (radiation_out_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_top_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            radiation_out_top_buffer = createBuffer(rad_out_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(radiation_out_top_buffer);
            descriptors_dirty = true;
        }

        if (radiation_out_bottom_buffer.buffer == VK_NULL_HANDLE || radiation_out_bottom_buffer.size != rad_out_size) {
            if (radiation_out_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_bottom_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            radiation_out_bottom_buffer = createBuffer(rad_out_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(radiation_out_bottom_buffer);
            descriptors_dirty = true;
        }

        // Ensure scatter_top/bottom buffers exist (required by shader and barriers)
        if (scatter_top_buffer.buffer == VK_NULL_HANDLE || scatter_top_buffer.size != rad_out_size) {
            if (scatter_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(scatter_top_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            scatter_top_buffer = createBuffer(rad_out_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(scatter_top_buffer);
            descriptors_dirty = true;
        }
        if (scatter_bottom_buffer.buffer == VK_NULL_HANDLE || scatter_bottom_buffer.size != rad_out_size) {
            if (scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(scatter_bottom_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            scatter_bottom_buffer = createBuffer(rad_out_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(scatter_bottom_buffer);
            descriptors_dirty = true;
        }

        // Upload radiation_out_top and radiation_out_bottom if provided
        if (!params.radiation_out_top.empty() && !params.radiation_out_bottom.empty()) {
            uploadRadiationOut(params.radiation_out_top, params.radiation_out_bottom);
        }

        // Upload diffuse sky parameters
        if (!params.diffuse_flux.empty()) {
            // Diffuse flux buffer
            size_t flux_size = params.diffuse_flux.size() * sizeof(float);
            if (diffuse_flux_buffer.buffer == VK_NULL_HANDLE || diffuse_flux_buffer.size != flux_size) {
                if (diffuse_flux_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(diffuse_flux_buffer);
                }
                diffuse_flux_buffer = createBuffer(flux_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                descriptors_dirty = true;
            }
            uploadBufferData(diffuse_flux_buffer, params.diffuse_flux.data(), flux_size);
        }

        if (!params.diffuse_peak_dir.empty()) {
            // Diffuse peak direction buffer
            size_t peak_dir_size = params.diffuse_peak_dir.size() * sizeof(helios::vec3);
            if (diffuse_peak_dir_buffer.buffer == VK_NULL_HANDLE || diffuse_peak_dir_buffer.size != peak_dir_size) {
                if (diffuse_peak_dir_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(diffuse_peak_dir_buffer);
                }
                diffuse_peak_dir_buffer = createBuffer(peak_dir_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                descriptors_dirty = true;
            }
            uploadBufferData(diffuse_peak_dir_buffer, params.diffuse_peak_dir.data(), peak_dir_size);
        }

        if (!params.diffuse_extinction.empty()) {
            // Diffuse extinction buffer
            size_t extinction_size = params.diffuse_extinction.size() * sizeof(float);
            if (diffuse_extinction_buffer.buffer == VK_NULL_HANDLE || diffuse_extinction_buffer.size != extinction_size) {
                if (diffuse_extinction_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(diffuse_extinction_buffer);
                }
                diffuse_extinction_buffer = createBuffer(extinction_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                descriptors_dirty = true;
            }
            uploadBufferData(diffuse_extinction_buffer, params.diffuse_extinction.data(), extinction_size);
        }

        if (!params.diffuse_dist_norm.empty()) {
            // Diffuse distribution normalization buffer
            size_t dist_norm_size = params.diffuse_dist_norm.size() * sizeof(float);
            if (diffuse_dist_norm_buffer.buffer == VK_NULL_HANDLE || diffuse_dist_norm_buffer.size != dist_norm_size) {
                if (diffuse_dist_norm_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(diffuse_dist_norm_buffer);
                }
                diffuse_dist_norm_buffer = createBuffer(dist_norm_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
                descriptors_dirty = true;
            }
            uploadBufferData(diffuse_dist_norm_buffer, params.diffuse_dist_norm.data(), dist_norm_size);
        }

        // Ensure all sky parameter buffers exist (required by shader even if empty/zero)
        // Only create if null - don't resize (that's handled by upload sections above)
        if (diffuse_flux_buffer.buffer == VK_NULL_HANDLE) {
            size_t flux_size = band_count * sizeof(float);
            diffuse_flux_buffer = createBuffer(flux_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(diffuse_flux_buffer);
            descriptors_dirty = true;
        }

        if (diffuse_peak_dir_buffer.buffer == VK_NULL_HANDLE) {
            size_t peak_dir_size = band_count * sizeof(helios::vec3);
            diffuse_peak_dir_buffer = createBuffer(peak_dir_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(diffuse_peak_dir_buffer);
            descriptors_dirty = true;
        }

        if (diffuse_extinction_buffer.buffer == VK_NULL_HANDLE) {
            size_t extinction_size = band_count * sizeof(float);
            diffuse_extinction_buffer = createBuffer(extinction_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(diffuse_extinction_buffer);
            descriptors_dirty = true;
        }

        if (diffuse_dist_norm_buffer.buffer == VK_NULL_HANDLE) {
            size_t dist_norm_size = band_count * sizeof(float);
            diffuse_dist_norm_buffer = createBuffer(dist_norm_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(diffuse_dist_norm_buffer);
            descriptors_dirty = true;
        }

        // Sky radiance params (Prague model) - ensure properly sized and zeroed
        // TODO Phase 3+: Upload actual sky_radiance_params from updateSkyModel
        {
            size_t sky_params_size = launch_band_count * sizeof(helios::vec4);
            if (sky_radiance_params_buffer.buffer == VK_NULL_HANDLE || sky_radiance_params_buffer.size != sky_params_size) {
                if (sky_radiance_params_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(sky_radiance_params_buffer);
                }
                sky_radiance_params_buffer = createBuffer(sky_params_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
                descriptors_dirty = true;
            }
            zeroBuffer(sky_radiance_params_buffer);
        }

        // Create debug counters buffer if needed (5 uint32_t counters)
        // MUST be before descriptor update check
        if (debug_counters_buffer.buffer == VK_NULL_HANDLE) {
            debug_counters_buffer = createBuffer(5 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            zeroBuffer(debug_counters_buffer);
            descriptors_dirty = true;
        } else {
            // Zero counters before each dispatch
            zeroBuffer(debug_counters_buffer);
        }

        // Update descriptor sets only if buffers changed
        if (descriptors_dirty) {
            updateDescriptorSets();
            descriptors_dirty = false;
        }
        VkDevice vk_device = device->getDevice();

        // Compute 2D grid dimensions for stratified hemisphere sampling (matches OptiX: ceil(sqrt(rays_per_primitive)))
        uint32_t launch_dim_x = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(params.rays_per_primitive))));
        uint32_t launch_dim_y = launch_dim_x;

        // Use the launch_face specified by the caller (RadiationModel already loops over faces)
        uint32_t launch_face = params.launch_face;

        // Build band mapping: launch band index → global band index
        launch_to_global_band.clear();
        for (uint32_t g = 0; g < band_count; g++) {
            if (!params.band_launch_flag.empty() && params.band_launch_flag[g]) {
                launch_to_global_band.push_back(g);
            }
        }
        if (launch_to_global_band.empty()) {
            // Fallback: identity mapping (all bands active)
            for (uint32_t g = 0; g < launch_band_count; g++) {
                launch_to_global_band.push_back(g);
            }
        }

        // Upload band mapping to GPU buffer
        uploadBufferData(band_map_buffer, launch_to_global_band.data(),
                         launch_to_global_band.size() * sizeof(uint32_t));

        // Push constants struct (expanded for diffuse rays)
        struct PushConstants {
            uint32_t launch_offset;
            uint32_t launch_count;
            uint32_t rays_per_primitive;
            uint32_t random_seed;
            uint32_t band_count;
            uint32_t source_count;
            uint32_t primitive_count;
            uint32_t launch_face;     // 0 = bottom, 1 = top
            uint32_t launch_dim_x;    // Grid dimension X
            uint32_t launch_dim_y;    // Grid dimension Y
            uint32_t material_band_count;  // Global band count (for material buffers)
            uint32_t periodic_flag_x;   // 1 if periodic in X direction
            uint32_t periodic_flag_y;   // 1 if periodic in Y direction
            uint32_t bbox_count;         // Number of bbox faces (0-4)
            float domain_xmin;       // Domain bounds for periodic wrapping
            float domain_xmax;
            float domain_ymin;
            float domain_ymax;
        } push_constants;

        // Initialize invariant push constants
        push_constants.rays_per_primitive = params.rays_per_primitive;
        push_constants.random_seed = params.random_seed;
        push_constants.band_count = launch_band_count;  // Use launch band count (not global)
        push_constants.material_band_count = band_count;  // Global band count for material indexing
        push_constants.source_count = source_count;
        push_constants.primitive_count = primitive_count;
        push_constants.launch_face = launch_face;
        push_constants.launch_dim_x = launch_dim_x;
        push_constants.launch_dim_y = launch_dim_y;

        // Periodic boundary parameters
        push_constants.periodic_flag_x = static_cast<uint32_t>(periodic_flag_x);
        push_constants.periodic_flag_y = static_cast<uint32_t>(periodic_flag_y);
        push_constants.bbox_count = bbox_count;
        push_constants.domain_xmin = domain_bounds[0];
        push_constants.domain_xmax = domain_bounds[1];
        push_constants.domain_ymin = domain_bounds[2];
        push_constants.domain_ymax = domain_bounds[3];

        // 3D dispatch: X/Y = ray grid, Z = primitives (up to 65535 per dispatch)
        const uint32_t WG_X = 8;  // Must match shader local_size_x
        const uint32_t WG_Y = 32; // Must match shader local_size_y

        uint32_t dispatch_x = (launch_dim_x + WG_X - 1) / WG_X;
        uint32_t dispatch_y = (launch_dim_y + WG_Y - 1) / WG_Y;
        uint32_t dispatch_z = params.launch_count;

        push_constants.launch_offset = params.launch_offset;
        push_constants.launch_count = params.launch_count;

        {
            // Record COMPUTE command buffer
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(compute_command_buffer, &begin_info);

            // Reset timestamp queries for this command buffer
            vkCmdResetQueryPool(compute_command_buffer, timestamp_query_pool, 0, 2);

            // Bind pipeline
            vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_diffuse);

            // Bind descriptor sets (geometry, materials, results, sky, debug)
            VkDescriptorSet sets[] = {set_geometry, set_materials, set_results, set_sky, set_debug};
            vkCmdBindDescriptorSets(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 5, sets, 0, nullptr);

            // Write timestamp before dispatch (measures GPU start time)
            vkCmdWriteTimestamp(compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestamp_query_pool, 0);

            // Single dispatch — all primitives via 2D tiling, all bands handled inside shader
            vkCmdPushConstants(compute_command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
            vkCmdDispatch(compute_command_buffer, dispatch_x, dispatch_y, dispatch_z);

            // Write timestamp after dispatch (measures GPU end time)
            vkCmdWriteTimestamp(compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestamp_query_pool, 1);

            // Final buffer memory barriers to ensure storage buffer writes are visible
            VkBufferMemoryBarrier buffer_barriers[4];

            // Radiation_in buffer barrier
            buffer_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            buffer_barriers[0].pNext = nullptr;
            buffer_barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            buffer_barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            buffer_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[0].buffer = radiation_in_buffer.buffer;
            buffer_barriers[0].offset = 0;
            buffer_barriers[0].size = VK_WHOLE_SIZE;

            // Scatter_top buffer barrier
            buffer_barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            buffer_barriers[1].pNext = nullptr;
            buffer_barriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            buffer_barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            buffer_barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[1].buffer = scatter_top_buffer.buffer;
            buffer_barriers[1].offset = 0;
            buffer_barriers[1].size = VK_WHOLE_SIZE;

            // Scatter_bottom buffer barrier
            buffer_barriers[2].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            buffer_barriers[2].pNext = nullptr;
            buffer_barriers[2].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            buffer_barriers[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            buffer_barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[2].buffer = scatter_bottom_buffer.buffer;
            buffer_barriers[2].offset = 0;
            buffer_barriers[2].size = VK_WHOLE_SIZE;

            // Radiation_out_top buffer barrier (read-only for diffuse rays)
            buffer_barriers[3].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            buffer_barriers[3].pNext = nullptr;
            buffer_barriers[3].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            buffer_barriers[3].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            buffer_barriers[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            buffer_barriers[3].buffer = radiation_out_top_buffer.buffer;
            buffer_barriers[3].offset = 0;
            buffer_barriers[3].size = VK_WHOLE_SIZE;

            vkCmdPipelineBarrier(compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, nullptr,  // No global memory barriers
                                 4, buffer_barriers,  // Buffer-specific barriers
                                 0, nullptr);  // No image barriers

            vkEndCommandBuffer(compute_command_buffer);

            // Submit command buffer with COMPUTE fence
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &compute_command_buffer;

            vkResetFences(vk_device, 1, &compute_fence);
            VkResult result = vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, compute_fence);
            if (result != VK_SUCCESS) {
                helios_runtime_error("ERROR (VulkanComputeBackend::launchDiffuseRays): vkQueueSubmit failed. VkResult: " + std::to_string(result));
            }

            // Wait for compute to complete (no timeout - large scenes can take minutes)
            VkResult wait_result;
            do {
                wait_result = vkWaitForFences(vk_device, 1, &compute_fence, VK_TRUE, 1000000000ULL);
                if (wait_result != VK_SUCCESS && wait_result != VK_TIMEOUT) {
                    helios_runtime_error("ERROR (VulkanComputeBackend::launchDiffuseRays): vkWaitForFences failed. VkResult: " + std::to_string(wait_result));
                }
            } while (wait_result == VK_TIMEOUT);
        }
    }

    void VulkanComputeBackend::launchCameraRays(const RayTracingLaunchParams &params) {
        if (primitive_count == 0) {
            return; // No geometry
        }

        // Build band mapping (same logic as direct/diffuse)
        launch_to_global_band.clear();
        for (uint32_t g = 0; g < band_count; g++) {
            if (!params.band_launch_flag.empty() && params.band_launch_flag[g]) {
                launch_to_global_band.push_back(g);
            }
        }
        if (launch_to_global_band.empty()) {
            for (uint32_t g = 0; g < launch_band_count; g++) {
                launch_to_global_band.push_back(g);
            }
        }

        // Ensure camera_radiation_buffer exists at FULL resolution
        size_t total_pixels = size_t(params.camera_resolution_full.x) * size_t(params.camera_resolution_full.y);
        size_t radiation_size = total_pixels * launch_band_count * sizeof(float);
        if (camera_radiation_buffer.buffer == VK_NULL_HANDLE || camera_radiation_buffer.size != radiation_size) {
            if (camera_radiation_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(camera_radiation_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            camera_radiation_buffer = createBuffer(radiation_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Zero camera radiation buffer before tile loop (matches OptiX line 713)
        // CRITICAL: Only zero if this is the first tile (pixel_offset == 0,0)
        // Subsequent tiles accumulate into the same buffer
        if (params.camera_pixel_offset.x == 0 && params.camera_pixel_offset.y == 0) {
            zeroBuffer(camera_radiation_buffer);
        }

        // Update descriptor sets if buffers changed
        if (descriptors_dirty) {
            updateDescriptorSets();
            descriptors_dirty = false;
        }

        VkDevice vk_device = device->getDevice();

        // Record COMPUTE command buffer
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(compute_command_buffer, &begin_info);

        // Bind pipeline
        vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_camera);

        // Bind descriptor sets
        VkDescriptorSet sets[] = {set_geometry, set_materials, set_results, set_sky, set_debug};
        vkCmdBindDescriptorSets(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 5, sets, 0, nullptr);

        // Build push constants
        struct PushConstants {
            helios::vec3 camera_position;      // 12 bytes
            float viewplane_length;            // 4 bytes
            float camera_direction_x;          // 4 bytes
            float camera_direction_y;          // 4 bytes
            float focal_length;                // 4 bytes
            float lens_diameter;               // 4 bytes
            float fov_aspect_ratio;            // 4 bytes
            uint32_t resolution_x;             // 4 bytes
            uint32_t resolution_y;             // 4 bytes
            uint32_t resolution_full_x;        // 4 bytes
            uint32_t resolution_full_y;        // 4 bytes
            uint32_t pixel_offset_x;           // 4 bytes
            uint32_t pixel_offset_y;           // 4 bytes
            uint32_t antialiasing_samples;     // 4 bytes
            uint32_t random_seed;              // 4 bytes
            uint32_t band_count;               // 4 bytes
            uint32_t source_count;             // 4 bytes
            uint32_t primitive_count;          // 4 bytes
            helios::vec3 sun_direction;        // 12 bytes
            float solar_disk_cos_angle;        // 4 bytes
            uint32_t periodic_flag_x;          // 4 bytes
            uint32_t periodic_flag_y;          // 4 bytes
            uint32_t bbox_count;               // 4 bytes
            float domain_xmin;                 // 4 bytes
            float domain_xmax;                 // 4 bytes
            float domain_ymin;                 // 4 bytes
            float domain_ymax;                 // 4 bytes
            uint32_t specular_reflection_enabled;  // 4 bytes
        } push_constants{};  // Total: 128 bytes

        push_constants.camera_position = params.camera_position;
        push_constants.viewplane_length = params.camera_viewplane_length;
        push_constants.camera_direction_x = params.camera_direction.x;
        push_constants.camera_direction_y = params.camera_direction.y;
        push_constants.focal_length = params.camera_focal_length;
        push_constants.lens_diameter = params.camera_lens_diameter;
        push_constants.fov_aspect_ratio = params.camera_fov_aspect;
        push_constants.resolution_x = params.camera_resolution.x;
        push_constants.resolution_y = params.camera_resolution.y;
        push_constants.resolution_full_x = params.camera_resolution_full.x;
        push_constants.resolution_full_y = params.camera_resolution_full.y;
        push_constants.pixel_offset_x = params.camera_pixel_offset.x;
        push_constants.pixel_offset_y = params.camera_pixel_offset.y;
        push_constants.antialiasing_samples = params.antialiasing_samples;
        push_constants.random_seed = params.random_seed;
        push_constants.band_count = launch_band_count;
        push_constants.source_count = source_count;
        push_constants.primitive_count = primitive_count;
        push_constants.sun_direction = cached_sun_direction;
        push_constants.solar_disk_cos_angle = cached_solar_disk_cos_angle;
        push_constants.periodic_flag_x = static_cast<uint32_t>(periodic_flag_x);
        push_constants.periodic_flag_y = static_cast<uint32_t>(periodic_flag_y);
        push_constants.bbox_count = bbox_count;
        push_constants.domain_xmin = domain_bounds[0];
        push_constants.domain_xmax = domain_bounds[1];
        push_constants.domain_ymin = domain_bounds[2];
        push_constants.domain_ymax = domain_bounds[3];
        push_constants.specular_reflection_enabled = params.specular_reflection_enabled;

        vkCmdPushConstants(compute_command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);

        // Compute dispatch dimensions (workgroup size 16x16x1)
        const uint32_t WG_X = 16;
        const uint32_t WG_Y = 16;
        uint32_t dispatch_x = (params.camera_resolution.x + WG_X - 1) / WG_X;
        uint32_t dispatch_y = (params.camera_resolution.y + WG_Y - 1) / WG_Y;
        uint32_t dispatch_z = params.antialiasing_samples;

        vkCmdDispatch(compute_command_buffer, dispatch_x, dispatch_y, dispatch_z);

        // Buffer memory barrier for camera_radiation_buffer
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = camera_radiation_buffer.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &barrier, 0, nullptr);

        vkEndCommandBuffer(compute_command_buffer);

        // Submit command buffer
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &compute_command_buffer;

        vkResetFences(vk_device, 1, &compute_fence);
        VkResult result = vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, compute_fence);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::launchCameraRays): vkQueueSubmit failed. VkResult: " + std::to_string(result));
        }

        // Wait for compute to complete (no timeout - large scenes can take minutes)
        VkResult wait_result;
        do {
            wait_result = vkWaitForFences(vk_device, 1, &compute_fence, VK_TRUE, 1000000000ULL);
            if (wait_result != VK_SUCCESS && wait_result != VK_TIMEOUT) {
                helios_runtime_error("ERROR (VulkanComputeBackend::launchCameraRays): vkWaitForFences failed. VkResult: " + std::to_string(wait_result));
            }
        } while (wait_result == VK_TIMEOUT);
    }

    void VulkanComputeBackend::launchPixelLabelRays(const RayTracingLaunchParams &params) {
        if (primitive_count == 0) {
            return; // No geometry
        }

        // Update descriptor sets if buffers changed
        if (descriptors_dirty) {
            updateDescriptorSets();
            descriptors_dirty = false;
        }

        VkDevice vk_device = device->getDevice();

        // Record COMPUTE command buffer
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(compute_command_buffer, &begin_info);

        // Bind pipeline
        vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_pixel_label);

        // Bind descriptor sets
        VkDescriptorSet sets[] = {set_geometry, set_materials, set_results, set_sky, set_debug};
        vkCmdBindDescriptorSets(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 5, sets, 0, nullptr);

        // Build push constants (same layout as camera_raygen)
        struct PushConstants {
            helios::vec3 camera_position;
            float viewplane_length;
            float camera_direction_x;
            float camera_direction_y;
            float focal_length;
            float lens_diameter;
            float fov_aspect_ratio;
            uint32_t resolution_x;
            uint32_t resolution_y;
            uint32_t resolution_full_x;
            uint32_t resolution_full_y;
            uint32_t pixel_offset_x;
            uint32_t pixel_offset_y;
            uint32_t antialiasing_samples;
            uint32_t random_seed;
            uint32_t band_count;
            uint32_t source_count;
            uint32_t primitive_count;
            helios::vec3 sun_direction;
            float solar_disk_cos_angle;
            uint32_t periodic_flag_x;
            uint32_t periodic_flag_y;
            uint32_t bbox_count;
            float domain_xmin;
            float domain_xmax;
            float domain_ymin;
            float domain_ymax;
            uint32_t padding;
        } push_constants{};

        push_constants.camera_position = params.camera_position;
        push_constants.viewplane_length = params.camera_viewplane_length;
        push_constants.camera_direction_x = params.camera_direction.x;
        push_constants.camera_direction_y = params.camera_direction.y;
        push_constants.focal_length = params.camera_focal_length;
        push_constants.lens_diameter = params.camera_lens_diameter;
        push_constants.fov_aspect_ratio = params.camera_fov_aspect;
        push_constants.resolution_x = params.camera_resolution.x;
        push_constants.resolution_y = params.camera_resolution.y;
        push_constants.resolution_full_x = params.camera_resolution_full.x;
        push_constants.resolution_full_y = params.camera_resolution_full.y;
        push_constants.pixel_offset_x = params.camera_pixel_offset.x;
        push_constants.pixel_offset_y = params.camera_pixel_offset.y;
        push_constants.antialiasing_samples = 1;  // No AA for pixel label
        push_constants.random_seed = params.random_seed;
        push_constants.band_count = launch_band_count;
        push_constants.source_count = source_count;
        push_constants.primitive_count = primitive_count;
        push_constants.sun_direction = cached_sun_direction;
        push_constants.solar_disk_cos_angle = cached_solar_disk_cos_angle;
        push_constants.periodic_flag_x = static_cast<uint32_t>(periodic_flag_x);
        push_constants.periodic_flag_y = static_cast<uint32_t>(periodic_flag_y);
        push_constants.bbox_count = bbox_count;
        push_constants.domain_xmin = domain_bounds[0];
        push_constants.domain_xmax = domain_bounds[1];
        push_constants.domain_ymin = domain_bounds[2];
        push_constants.domain_ymax = domain_bounds[3];
        push_constants.padding = 0;

        vkCmdPushConstants(compute_command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);

        // Compute dispatch dimensions (workgroup size 16x16x1, no AA dimension)
        const uint32_t WG_X = 16;
        const uint32_t WG_Y = 16;
        uint32_t dispatch_x = (params.camera_resolution.x + WG_X - 1) / WG_X;
        uint32_t dispatch_y = (params.camera_resolution.y + WG_Y - 1) / WG_Y;
        uint32_t dispatch_z = 1;  // No antialiasing

        vkCmdDispatch(compute_command_buffer, dispatch_x, dispatch_y, dispatch_z);

        // Buffer memory barriers for pixel label and depth buffers
        VkBufferMemoryBarrier barriers[2];

        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = nullptr;
        barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = camera_pixel_label_buffer.buffer;
        barriers[0].offset = 0;
        barriers[0].size = VK_WHOLE_SIZE;

        barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[1].pNext = nullptr;
        barriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].buffer = camera_pixel_depth_buffer.buffer;
        barriers[1].offset = 0;
        barriers[1].size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 2, barriers, 0, nullptr);

        vkEndCommandBuffer(compute_command_buffer);

        // Submit command buffer
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &compute_command_buffer;

        vkResetFences(vk_device, 1, &compute_fence);
        VkResult result = vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, compute_fence);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::launchPixelLabelRays): vkQueueSubmit failed. VkResult: " + std::to_string(result));
        }

        // Wait for compute to complete (no timeout - large scenes can take minutes)
        VkResult wait_result;
        do {
            wait_result = vkWaitForFences(vk_device, 1, &compute_fence, VK_TRUE, 1000000000ULL);
            if (wait_result != VK_SUCCESS && wait_result != VK_TIMEOUT) {
                helios_runtime_error("ERROR (VulkanComputeBackend::launchPixelLabelRays): vkWaitForFences failed. VkResult: " + std::to_string(wait_result));
            }
        } while (wait_result == VK_TIMEOUT);
    }

    void VulkanComputeBackend::getRadiationResults(RayTracingResults &results) {
        if (primitive_count == 0 || launch_band_count == 0) {
            return; // No results to download
        }

        size_t buffer_size = primitive_count * launch_band_count;

        // Resize output vectors
        results.radiation_in.resize(buffer_size);
        results.radiation_out_top.resize(buffer_size);
        results.radiation_out_bottom.resize(buffer_size);
        results.num_primitives = primitive_count;
        results.num_bands = launch_band_count;

        // Download radiation_in buffer
        if (radiation_in_buffer.buffer != VK_NULL_HANDLE) {
            // WORKAROUND for MoltenVK: Direct map instead of transfer command buffer
            // This works around coherency issues with compute shader writes
            vkQueueWaitIdle(device->getComputeQueue()); // Ensure compute is done

            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), radiation_in_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                // Invalidate mapped memory range to ensure coherency with GPU writes
                // Use VMA's invalidate which handles nonCoherentAtomSize alignment automatically
                vmaInvalidateAllocation(device->getAllocator(), radiation_in_buffer.allocation, 0, VK_WHOLE_SIZE);

                std::memcpy(results.radiation_in.data(), mapped, buffer_size * sizeof(float));
                vmaUnmapMemory(device->getAllocator(), radiation_in_buffer.allocation);

                // Check for shader error codes (negative values)
                if (results.radiation_in[0] < 0.0f) {
                    std::string error_msg = "ERROR (VulkanComputeBackend): Compute shader reported error. Code: " + std::to_string(results.radiation_in[0]);
                    if (results.radiation_in[0] == -999.0f) {
                        error_msg += " (Invalid subdivisions: NX=" + std::to_string(results.radiation_in[1]) + " NY=" + std::to_string(results.radiation_in[2]) + ")";
                    } else if (results.radiation_in[0] == -998.0f) {
                        error_msg += " (Subdivisions too large: NX=" + std::to_string(results.radiation_in[1]) + " NY=" + std::to_string(results.radiation_in[2]) + ")";
                    } else if (results.radiation_in[0] == -997.0f) {
                        error_msg += " (Zero rays per dimension)";
                    } else if (results.radiation_in[0] == -996.0f) {
                        error_msg += " (Too many rays per dimension: " + std::to_string(results.radiation_in[1]) + ")";
                    }
                    helios_runtime_error(error_msg);
                }

            } else {
                downloadBufferData(radiation_in_buffer, results.radiation_in.data(), buffer_size * sizeof(float));
            }
        } else {
            std::fill(results.radiation_in.begin(), results.radiation_in.end(), 0.0f);
        }

        // Download radiation_out_top buffer
        if (radiation_out_top_buffer.buffer != VK_NULL_HANDLE) {
            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), radiation_out_top_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                vmaInvalidateAllocation(device->getAllocator(), radiation_out_top_buffer.allocation, 0, VK_WHOLE_SIZE);
                std::memcpy(results.radiation_out_top.data(), mapped, buffer_size * sizeof(float));
                vmaUnmapMemory(device->getAllocator(), radiation_out_top_buffer.allocation);
            }
        } else {
            std::fill(results.radiation_out_top.begin(), results.radiation_out_top.end(), 0.0f);
        }

        // Download radiation_out_bottom buffer
        if (radiation_out_bottom_buffer.buffer != VK_NULL_HANDLE) {
            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), radiation_out_bottom_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                vmaInvalidateAllocation(device->getAllocator(), radiation_out_bottom_buffer.allocation, 0, VK_WHOLE_SIZE);
                std::memcpy(results.radiation_out_bottom.data(), mapped, buffer_size * sizeof(float));
                vmaUnmapMemory(device->getAllocator(), radiation_out_bottom_buffer.allocation);
            }
        } else {
            std::fill(results.radiation_out_bottom.begin(), results.radiation_out_bottom.end(), 0.0f);
        }

        // Download scatter_top buffer
        results.scatter_buff_top.resize(buffer_size);
        if (scatter_top_buffer.buffer != VK_NULL_HANDLE) {
            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), scatter_top_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                vmaInvalidateAllocation(device->getAllocator(), scatter_top_buffer.allocation, 0, VK_WHOLE_SIZE);
                std::memcpy(results.scatter_buff_top.data(), mapped, buffer_size * sizeof(float));
                vmaUnmapMemory(device->getAllocator(), scatter_top_buffer.allocation);
            }
        } else {
            std::fill(results.scatter_buff_top.begin(), results.scatter_buff_top.end(), 0.0f);
        }

        // Download scatter_bottom buffer
        results.scatter_buff_bottom.resize(buffer_size);
        if (scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), scatter_bottom_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                vmaInvalidateAllocation(device->getAllocator(), scatter_bottom_buffer.allocation, 0, VK_WHOLE_SIZE);
                std::memcpy(results.scatter_buff_bottom.data(), mapped, buffer_size * sizeof(float));
                vmaUnmapMemory(device->getAllocator(), scatter_bottom_buffer.allocation);
            }
        } else {
            std::fill(results.scatter_buff_bottom.begin(), results.scatter_buff_bottom.end(), 0.0f);
        }

        // Camera scatter buffers: use regular scatter as proxy for camera-weighted scatter.
        // This is exact when camera spectral response is uniform (1.0 across all wavelengths).
        // For non-uniform camera responses, Vulkan shaders would need dedicated camera scatter
        // buffers with camera-weighted materials (rho_cam, tau_cam), matching OptiX (rayHit.cu:223-236).
        results.scatter_buff_top_cam = results.scatter_buff_top;
        results.scatter_buff_bottom_cam = results.scatter_buff_bottom;
    }

    void VulkanComputeBackend::getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels, std::vector<float> &pixel_depths, uint camera_id, const helios::int2 &resolution) {
        size_t total_pixels = size_t(resolution.x) * size_t(resolution.y);
        if (total_pixels == 0) {
            return; // No pixels
        }

        // Download camera_radiation_buffer (pixel_data)
        pixel_data.resize(total_pixels * launch_band_count);
        if (camera_radiation_buffer.buffer != VK_NULL_HANDLE && !pixel_data.empty()) {
            vkQueueWaitIdle(device->getComputeQueue()); // Ensure compute is done

            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), camera_radiation_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                vmaInvalidateAllocation(device->getAllocator(), camera_radiation_buffer.allocation, 0, VK_WHOLE_SIZE);
                std::memcpy(pixel_data.data(), mapped, pixel_data.size() * sizeof(float));
                vmaUnmapMemory(device->getAllocator(), camera_radiation_buffer.allocation);
            } else {
                downloadBufferData(camera_radiation_buffer, pixel_data.data(), pixel_data.size() * sizeof(float));
            }
        } else {
            std::fill(pixel_data.begin(), pixel_data.end(), 0.0f);
        }

        // Download camera_pixel_label_buffer (pixel_labels)
        pixel_labels.resize(total_pixels);
        if (camera_pixel_label_buffer.buffer != VK_NULL_HANDLE && !pixel_labels.empty()) {
            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), camera_pixel_label_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                vmaInvalidateAllocation(device->getAllocator(), camera_pixel_label_buffer.allocation, 0, VK_WHOLE_SIZE);
                std::memcpy(pixel_labels.data(), mapped, pixel_labels.size() * sizeof(uint));
                vmaUnmapMemory(device->getAllocator(), camera_pixel_label_buffer.allocation);
            } else {
                downloadBufferData(camera_pixel_label_buffer, pixel_labels.data(), pixel_labels.size() * sizeof(uint));
            }
        } else {
            std::fill(pixel_labels.begin(), pixel_labels.end(), 0u);
        }

        // Download camera_pixel_depth_buffer (pixel_depths)
        pixel_depths.resize(total_pixels);
        if (camera_pixel_depth_buffer.buffer != VK_NULL_HANDLE && !pixel_depths.empty()) {
            void *mapped;
            VkResult result = vmaMapMemory(device->getAllocator(), camera_pixel_depth_buffer.allocation, &mapped);
            if (result == VK_SUCCESS) {
                vmaInvalidateAllocation(device->getAllocator(), camera_pixel_depth_buffer.allocation, 0, VK_WHOLE_SIZE);
                std::memcpy(pixel_depths.data(), mapped, pixel_depths.size() * sizeof(float));
                vmaUnmapMemory(device->getAllocator(), camera_pixel_depth_buffer.allocation);
            } else {
                downloadBufferData(camera_pixel_depth_buffer, pixel_depths.data(), pixel_depths.size() * sizeof(float));
            }
        } else {
            std::fill(pixel_depths.begin(), pixel_depths.end(), 0.0f);
        }
    }

    void VulkanComputeBackend::zeroRadiationBuffers(size_t launch_band_count_param) {
        if (primitive_count == 0 || band_count == 0) {
            return; // No geometry or bands
        }

        // Store launch band count for this runBand() call
        launch_band_count = static_cast<uint32_t>(launch_band_count_param);

        // Create or resize band_map buffer [launch_band_count × uint32]
        size_t band_map_size = launch_band_count * sizeof(uint32_t);
        if (band_map_buffer.buffer == VK_NULL_HANDLE || band_map_buffer.size != band_map_size) {
            if (band_map_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(band_map_buffer);
            }
            band_map_buffer = createBuffer(band_map_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            descriptors_dirty = true;
        }

        size_t buffer_size = primitive_count * launch_band_count;

        // Create or resize radiation_in buffer
        // CRITICAL: Use AUTO_PREFER_HOST for coherent memory (matches working Vulkan compute examples)
        // This ensures HOST_VISIBLE | HOST_COHERENT memory which works reliably on MoltenVK
        if (radiation_in_buffer.buffer == VK_NULL_HANDLE || radiation_in_buffer.size != buffer_size * sizeof(float)) {
            if (radiation_in_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_in_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            radiation_in_buffer = createBuffer(buffer_size * sizeof(float), usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        }

        // Create or resize radiation_specular buffer [source * primitive * band]
        // Only create if source_count > 0 (specular requires sources)
        if (source_count > 0) {
            size_t specular_buffer_size = source_count * primitive_count * launch_band_count;
            if (radiation_specular_buffer.buffer == VK_NULL_HANDLE || radiation_specular_buffer.size != specular_buffer_size * sizeof(float)) {
                if (radiation_specular_buffer.buffer != VK_NULL_HANDLE) {
                    destroyBuffer(radiation_specular_buffer);
                }
                VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                radiation_specular_buffer = createBuffer(specular_buffer_size * sizeof(float), usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            }
            zeroBuffer(radiation_specular_buffer);
        }

        // Zero radiation buffers
        zeroBuffer(radiation_in_buffer);

        descriptors_dirty = true;  // Result buffers created/changed
    }

    void VulkanComputeBackend::zeroScatterBuffers() {
        if (primitive_count == 0 || launch_band_count == 0) {
            return; // No geometry or bands
        }

        size_t buffer_size = primitive_count * launch_band_count;

        // Create or resize scatter_top buffer
        if (scatter_top_buffer.buffer == VK_NULL_HANDLE || scatter_top_buffer.size != buffer_size * sizeof(float)) {
            if (scatter_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(scatter_top_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            scatter_top_buffer = createBuffer(buffer_size * sizeof(float), usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        }

        // Create or resize scatter_bottom buffer
        if (scatter_bottom_buffer.buffer == VK_NULL_HANDLE || scatter_bottom_buffer.size != buffer_size * sizeof(float)) {
            if (scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(scatter_bottom_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            scatter_bottom_buffer = createBuffer(buffer_size * sizeof(float), usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        }

        // Zero both buffers
        zeroBuffer(scatter_top_buffer);
        zeroBuffer(scatter_bottom_buffer);

        descriptors_dirty = true;  // Scatter buffers created/changed
    }

    void VulkanComputeBackend::zeroCameraPixelBuffers(const helios::int2 &resolution) {
        size_t total_pixels = size_t(resolution.x) * size_t(resolution.y);
        if (total_pixels == 0) {
            return; // No pixels
        }

        // Create or resize camera_pixel_label_buffer
        VkDeviceSize label_size = total_pixels * sizeof(uint32_t);
        if (camera_pixel_label_buffer.buffer == VK_NULL_HANDLE || camera_pixel_label_buffer.size != label_size) {
            if (camera_pixel_label_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(camera_pixel_label_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            camera_pixel_label_buffer = createBuffer(label_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Create or resize camera_pixel_depth_buffer
        VkDeviceSize depth_size = total_pixels * sizeof(float);
        if (camera_pixel_depth_buffer.buffer == VK_NULL_HANDLE || camera_pixel_depth_buffer.size != depth_size) {
            if (camera_pixel_depth_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(camera_pixel_depth_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            camera_pixel_depth_buffer = createBuffer(depth_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Zero both buffers
        zeroBuffer(camera_pixel_label_buffer);
        zeroBuffer(camera_pixel_depth_buffer);
    }

    void VulkanComputeBackend::copyScatterToRadiation() {
        if (primitive_count == 0 || band_count == 0) {
            return; // No geometry or bands
        }

        size_t buffer_size = primitive_count * band_count * sizeof(float);

        // Create radiation_out_top/bottom buffers if they don't exist
        if (radiation_out_top_buffer.buffer == VK_NULL_HANDLE || radiation_out_top_buffer.size != buffer_size) {
            if (radiation_out_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_top_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            radiation_out_top_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        if (radiation_out_bottom_buffer.buffer == VK_NULL_HANDLE || radiation_out_bottom_buffer.size != buffer_size) {
            if (radiation_out_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_bottom_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            radiation_out_bottom_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Wait for any pending compute work and invalidate scatter buffers before reading
        vkQueueWaitIdle(device->getComputeQueue());

        // Copy scatter_top → radiation_out_top
        vmaInvalidateAllocation(device->getAllocator(), scatter_top_buffer.allocation, 0, VK_WHOLE_SIZE);
        void *src_top = nullptr;
        void *dst_top = nullptr;
        vmaMapMemory(device->getAllocator(), scatter_top_buffer.allocation, &src_top);
        vmaMapMemory(device->getAllocator(), radiation_out_top_buffer.allocation, &dst_top);
        memcpy(dst_top, src_top, buffer_size);
        vmaFlushAllocation(device->getAllocator(), radiation_out_top_buffer.allocation, 0, VK_WHOLE_SIZE);
        vmaUnmapMemory(device->getAllocator(), scatter_top_buffer.allocation);
        vmaUnmapMemory(device->getAllocator(), radiation_out_top_buffer.allocation);

        // Copy scatter_bottom → radiation_out_bottom
        vmaInvalidateAllocation(device->getAllocator(), scatter_bottom_buffer.allocation, 0, VK_WHOLE_SIZE);
        void *src_bottom = nullptr;
        void *dst_bottom = nullptr;
        vmaMapMemory(device->getAllocator(), scatter_bottom_buffer.allocation, &src_bottom);
        vmaMapMemory(device->getAllocator(), radiation_out_bottom_buffer.allocation, &dst_bottom);
        memcpy(dst_bottom, src_bottom, buffer_size);
        vmaFlushAllocation(device->getAllocator(), radiation_out_bottom_buffer.allocation, 0, VK_WHOLE_SIZE);
        vmaUnmapMemory(device->getAllocator(), scatter_bottom_buffer.allocation);
        vmaUnmapMemory(device->getAllocator(), radiation_out_bottom_buffer.allocation);
    }

    void VulkanComputeBackend::uploadRadiationOut(const std::vector<float> &radiation_out_top, const std::vector<float> &radiation_out_bottom) {
        if (radiation_out_top.empty() || radiation_out_bottom.empty()) {
            return; // No data to upload
        }

        size_t buffer_size = radiation_out_top.size() * sizeof(float);

        // Create radiation_out_top buffer if needed
        if (radiation_out_top_buffer.buffer == VK_NULL_HANDLE || radiation_out_top_buffer.size != buffer_size) {
            if (radiation_out_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_top_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            radiation_out_top_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Create radiation_out_bottom buffer if needed
        if (radiation_out_bottom_buffer.buffer == VK_NULL_HANDLE || radiation_out_bottom_buffer.size != buffer_size) {
            if (radiation_out_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(radiation_out_bottom_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            radiation_out_bottom_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Upload data using direct mapping (HOST_VISIBLE buffers, matching copyScatterToRadiation approach)
        void *dst_top = nullptr;
        void *dst_bottom = nullptr;
        VkResult result_top = vmaMapMemory(device->getAllocator(), radiation_out_top_buffer.allocation, &dst_top);
        VkResult result_bottom = vmaMapMemory(device->getAllocator(), radiation_out_bottom_buffer.allocation, &dst_bottom);

        if (result_top == VK_SUCCESS) {
            std::memcpy(dst_top, radiation_out_top.data(), buffer_size);
            vmaFlushAllocation(device->getAllocator(), radiation_out_top_buffer.allocation, 0, VK_WHOLE_SIZE);
            vmaUnmapMemory(device->getAllocator(), radiation_out_top_buffer.allocation);
        }
        if (result_bottom == VK_SUCCESS) {
            std::memcpy(dst_bottom, radiation_out_bottom.data(), buffer_size);
            vmaFlushAllocation(device->getAllocator(), radiation_out_bottom_buffer.allocation, 0, VK_WHOLE_SIZE);
            vmaUnmapMemory(device->getAllocator(), radiation_out_bottom_buffer.allocation);
        }
        if (result_top != VK_SUCCESS || result_bottom != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::uploadRadiationOut): Failed to map radiation output buffers.");
        }
    }

    void VulkanComputeBackend::uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam, const std::vector<float> &scatter_bottom_cam) {
        if (scatter_top_cam.empty() || scatter_bottom_cam.empty()) {
            return; // No data to upload
        }

        size_t buffer_size = scatter_top_cam.size() * sizeof(float);

        // Create or resize camera_scatter_top_buffer
        if (camera_scatter_top_buffer.buffer == VK_NULL_HANDLE || camera_scatter_top_buffer.size != buffer_size) {
            if (camera_scatter_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(camera_scatter_top_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            camera_scatter_top_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Create or resize camera_scatter_bottom_buffer
        if (camera_scatter_bottom_buffer.buffer == VK_NULL_HANDLE || camera_scatter_bottom_buffer.size != buffer_size) {
            if (camera_scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(camera_scatter_bottom_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            camera_scatter_bottom_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Upload data to both buffers
        uploadBufferData(camera_scatter_top_buffer, scatter_top_cam.data(), buffer_size);
        uploadBufferData(camera_scatter_bottom_buffer, scatter_bottom_cam.data(), buffer_size);
    }

    void VulkanComputeBackend::zeroCameraScatterBuffers(size_t launch_band_count_param) {
        if (primitive_count == 0 || launch_band_count_param == 0) {
            return; // No geometry or bands
        }

        size_t buffer_size = primitive_count * launch_band_count_param * sizeof(float);

        // Create or resize camera_scatter_top_buffer
        if (camera_scatter_top_buffer.buffer == VK_NULL_HANDLE || camera_scatter_top_buffer.size != buffer_size) {
            if (camera_scatter_top_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(camera_scatter_top_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            camera_scatter_top_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Create or resize camera_scatter_bottom_buffer
        if (camera_scatter_bottom_buffer.buffer == VK_NULL_HANDLE || camera_scatter_bottom_buffer.size != buffer_size) {
            if (camera_scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
                destroyBuffer(camera_scatter_bottom_buffer);
            }
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            camera_scatter_bottom_buffer = createBuffer(buffer_size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            descriptors_dirty = true;
        }

        // Zero both buffers
        zeroBuffer(camera_scatter_top_buffer);
        zeroBuffer(camera_scatter_bottom_buffer);
    }

    void VulkanComputeBackend::uploadSourceFluxes(const std::vector<float> &fluxes) {
        if (fluxes.empty() || source_count == 0) {
            return; // No fluxes or sources
        }

        // Fluxes are indexed by [source * Nbands_launch + band]
        // We expect Nsources * Nbands_launch entries
        size_t expected_size = source_count * launch_band_count;

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

        descriptors_dirty = true;  // New buffer created, descriptors need update
    }

    void VulkanComputeBackend::uploadSourceFluxesCam(const std::vector<float> &fluxes_cam) {
        if (fluxes_cam.empty() || source_count == 0) {
            return; // No camera weights or sources
        }

        // Camera spectral response weights indexed by [source * Nbands_launch + band]
        size_t expected_size = source_count * launch_band_count;

        if (fluxes_cam.size() != expected_size) {
            helios_runtime_error("ERROR (VulkanComputeBackend::uploadSourceFluxesCam): fluxes_cam size mismatch. Expected " + std::to_string(expected_size) +
                                 " (Nsources * Nbands_launch), got " + std::to_string(fluxes_cam.size()));
        }

        if (source_fluxes_cam_buffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(source_fluxes_cam_buffer);
        }
        source_fluxes_cam_buffer = createBuffer(fluxes_cam.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadBufferData(source_fluxes_cam_buffer, fluxes_cam.data(), fluxes_cam.size() * sizeof(float));

        descriptors_dirty = true; // New buffer uploaded, descriptors need update
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

        // CRITICAL for MoltenVK: Request host-coherent memory for compute shader result buffers
        // This matches working Vulkan compute examples that use HOST_VISIBLE | HOST_COHERENT
        if (mem_usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST) {
            alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }

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
        vkBeginCommandBuffer(transfer_command_buffer, &begin_info);

        VkBufferCopy copy_region{};
        copy_region.size = size;
        vkCmdCopyBuffer(transfer_command_buffer, staging.buffer, buffer.buffer, 1, &copy_region);

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

        vkCmdPipelineBarrier(transfer_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

        vkEndCommandBuffer(transfer_command_buffer);

        // Submit with fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &transfer_command_buffer;

        vkResetFences(device->getDevice(), 1, &transfer_fence);
        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, transfer_fence);

        // Polling-based timeout (MoltenVK doesn't respect timeout parameter)
        const int max_attempts = 50;
        const uint64_t poll_interval_ns = 100000000ULL; // 100ms
        bool completed = false;

        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            VkResult result = vkWaitForFences(device->getDevice(), 1, &transfer_fence, VK_TRUE, poll_interval_ns);
            if (result == VK_SUCCESS) {
                completed = true;
                break;
            } else if (result != VK_TIMEOUT) {
                helios_runtime_error("ERROR (VulkanComputeBackend::uploadBufferData): vkWaitForFences failed. VkResult: " + std::to_string(result));
            }
        }

        if (!completed) {
            helios_runtime_error("ERROR (VulkanComputeBackend::uploadBufferData): GPU buffer upload timed out after 5 seconds. Buffer size: " + std::to_string(size));
        }

        // SAFETY: Staging buffer destroyed only after fence signals (GPU copy complete)
        destroyBuffer(staging);
    }

    void VulkanComputeBackend::downloadBufferData(const Buffer &buffer, void *data, size_t size) {
        // CRITICAL: Wait for ALL GPU work to complete before downloading
        // This ensures compute shader writes are visible to the transfer operation
        vkQueueWaitIdle(device->getComputeQueue());

        // Create staging buffer
        Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

        // Copy device → staging
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(transfer_command_buffer, &begin_info);

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

        vkCmdPipelineBarrier(transfer_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

        VkBufferCopy copy_region{};
        copy_region.size = size;
        vkCmdCopyBuffer(transfer_command_buffer, buffer.buffer, staging.buffer, 1, &copy_region);

        vkEndCommandBuffer(transfer_command_buffer);

        // Submit with fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &transfer_command_buffer;

        vkResetFences(device->getDevice(), 1, &transfer_fence);
        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, transfer_fence);

        // Polling-based timeout (MoltenVK doesn't respect timeout parameter)
        const int max_attempts = 50;
        const uint64_t poll_interval_ns = 100000000ULL; // 100ms
        bool completed = false;

        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            VkResult result = vkWaitForFences(device->getDevice(), 1, &transfer_fence, VK_TRUE, poll_interval_ns);
            if (result == VK_SUCCESS) {
                completed = true;
                break;
            } else if (result != VK_TIMEOUT) {
                helios_runtime_error("ERROR (VulkanComputeBackend::downloadBufferData): vkWaitForFences failed. VkResult: " + std::to_string(result));
            }
        }

        if (!completed) {
            helios_runtime_error("ERROR (VulkanComputeBackend::downloadBufferData): GPU buffer download timed out after 5 seconds. Buffer size: " + std::to_string(size));
        }

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
        vkBeginCommandBuffer(transfer_command_buffer, &begin_info);

        vkCmdFillBuffer(transfer_command_buffer, buffer.buffer, 0, buffer.size, 0);

        vkEndCommandBuffer(transfer_command_buffer);

        // Submit with fence
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &transfer_command_buffer;

        vkResetFences(device->getDevice(), 1, &transfer_fence);
        vkQueueSubmit(device->getComputeQueue(), 1, &submit_info, transfer_fence);

        // Polling-based timeout (MoltenVK doesn't respect timeout parameter)
        const int max_attempts = 50;
        const uint64_t poll_interval_ns = 100000000ULL; // 100ms
        bool completed = false;

        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            VkResult result = vkWaitForFences(device->getDevice(), 1, &transfer_fence, VK_TRUE, poll_interval_ns);
            if (result == VK_SUCCESS) {
                completed = true;
                break;
            } else if (result != VK_TIMEOUT) {
                helios_runtime_error("ERROR (VulkanComputeBackend::zeroBuffer): vkWaitForFences failed. VkResult: " + std::to_string(result));
            }
        }

        if (!completed) {
            helios_runtime_error("ERROR (VulkanComputeBackend::zeroBuffer): GPU buffer clear timed out after 5 seconds. Buffer size: " + std::to_string(buffer.size));
        }
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

        // Allocate TWO command buffers: one for transfers, one for compute
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 2;

        VkCommandBuffer buffers[2];
        result = vkAllocateCommandBuffers(vk_device, &alloc_info, buffers);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to allocate command buffers. VkResult: " + std::to_string(result));
        }

        transfer_command_buffer = buffers[0];
        compute_command_buffer = buffers[1];

        // Create fences for synchronization
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start signaled so first wait succeeds

        result = vkCreateFence(vk_device, &fence_info, nullptr, &transfer_fence);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to create transfer fence. VkResult: " + std::to_string(result));
        }

        result = vkCreateFence(vk_device, &fence_info, nullptr, &compute_fence);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to create compute fence. VkResult: " + std::to_string(result));
        }

        // Create timestamp query pool for GPU profiling (2 queries: start and end)
        VkQueryPoolCreateInfo query_pool_info{};
        query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_pool_info.queryCount = 2;

        result = vkCreateQueryPool(vk_device, &query_pool_info, nullptr, &timestamp_query_pool);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createCommandResources): Failed to create timestamp query pool. VkResult: " + std::to_string(result));
        }

        // Get timestamp period (nanoseconds per tick) for converting timestamps to time
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device->getPhysicalDevice(), &props);
        timestamp_period = props.limits.timestampPeriod;
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
            {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // World-space normals
            {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Mask data (uint)
            {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Mask sizes (ivec2)
            {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Mask offsets (uint)
            {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Mask IDs (int)
            {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // UV data (vec2)
            {16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // UV IDs (int)
            {17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Bbox vertices (periodic boundary)
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
            {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Specular exponent
            {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Specular scale
            {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Source fluxes (camera-weighted)
            {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Band map
        };

        layout_info.bindingCount = static_cast<uint32_t>(material_bindings.size());
        layout_info.pBindings = material_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_materials) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create material descriptor set layout");
        }

        // Set 2: Result buffers (read/write, zeroed per-launch)
        std::vector<VkDescriptorSetLayoutBinding> result_bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // radiation_in
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // radiation_out_top
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // radiation_out_bottom
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // scatter_top
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // scatter_bottom
            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // camera_radiation
            {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // camera_pixel_label
            {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // camera_pixel_depth
            {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // camera_scatter_top
            {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // camera_scatter_bottom
            {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // radiation_specular
        };

        layout_info.bindingCount = static_cast<uint32_t>(result_bindings.size());
        layout_info.pBindings = result_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_results) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create result descriptor set layout");
        }

        // Set 3: Sky parameters (read-only diffuse parameters)
        std::vector<VkDescriptorSetLayoutBinding> sky_bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // diffuse_flux
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // diffuse_peak_dir
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // diffuse_extinction
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // diffuse_dist_norm
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // sky_radiance_params
            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // camera_sky_radiance
            {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // solar_disk_radiance
        };

        layout_info.bindingCount = static_cast<uint32_t>(sky_bindings.size());
        layout_info.pBindings = sky_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_sky) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create sky descriptor set layout");
        }

        // Set 4: Debug counters (profiling/diagnostics)
        std::vector<VkDescriptorSetLayoutBinding> debug_bindings = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // debug_counters
        };

        layout_info.bindingCount = static_cast<uint32_t>(debug_bindings.size());
        layout_info.pBindings = debug_bindings.data();

        if (vkCreateDescriptorSetLayout(vk_device, &layout_info, nullptr, &set_layout_debug) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create debug descriptor set layout");
        }

        // ========== Create Descriptor Pool ==========

        std::vector<VkDescriptorPoolSize> pool_sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 62}, // All sets: 18 geo + 11 mat + 11 result + 7 sky + 1 debug + margin
        };

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.maxSets = 5; // geometry, materials, results, sky, debug

        if (vkCreateDescriptorPool(vk_device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to create descriptor pool");
        }

        // ========== Allocate Descriptor Sets ==========

        VkDescriptorSetLayout layouts[] = {set_layout_geometry, set_layout_materials, set_layout_results, set_layout_sky, set_layout_debug};

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool;
        alloc_info.descriptorSetCount = 5;
        alloc_info.pSetLayouts = layouts;

        VkDescriptorSet sets[5];
        if (vkAllocateDescriptorSets(vk_device, &alloc_info, sets) != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createDescriptorSets): Failed to allocate descriptor sets");
        }

        set_geometry = sets[0];
        set_materials = sets[1];
        set_results = sets[2];
        set_sky = sets[3];
        set_debug = sets[4];

        // ========== Create placeholder sky parameter buffers ==========
        // MoltenVK requires all descriptor buffers to exist before pipeline creation
        // to determine argument buffer resource base types during shader compilation.
        // These will be resized properly when launchDiffuseRays() is first called.

        const size_t placeholder_size = sizeof(float); // Minimal 1-element buffer
        // All placeholders include TRANSFER_DST for zeroing - VMA may reuse freed memory with stale data

        diffuse_flux_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(diffuse_flux_buffer);
        diffuse_peak_dir_buffer = createBuffer(sizeof(helios::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(diffuse_peak_dir_buffer);
        diffuse_extinction_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(diffuse_extinction_buffer);
        diffuse_dist_norm_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(diffuse_dist_norm_buffer);
        sky_radiance_params_buffer = createBuffer(sizeof(helios::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(sky_radiance_params_buffer);
        camera_sky_radiance_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(camera_sky_radiance_buffer);
        solar_disk_radiance_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(solar_disk_radiance_buffer);

        // ========== Create placeholder camera result buffers ==========
        // MoltenVK requires these before pipeline creation (camera shaders reference them)
        camera_radiation_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        zeroBuffer(camera_radiation_buffer);
        camera_pixel_label_buffer = createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        zeroBuffer(camera_pixel_label_buffer);
        camera_pixel_depth_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        zeroBuffer(camera_pixel_depth_buffer);
        camera_scatter_top_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        zeroBuffer(camera_scatter_top_buffer);
        camera_scatter_bottom_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        zeroBuffer(camera_scatter_bottom_buffer);

        // ========== Create placeholder specular buffers ==========
        // MoltenVK requires these before pipeline creation (camera/direct shaders reference them)
        specular_exponent_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(specular_exponent_buffer);
        specular_scale_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(specular_scale_buffer);
        source_fluxes_cam_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(source_fluxes_cam_buffer);
        radiation_specular_buffer = createBuffer(placeholder_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        zeroBuffer(radiation_specular_buffer);

        // ========== Create placeholder mask/UV buffers ==========
        // Same requirement as sky parameters — needed before pipeline creation for MoltenVK
        mask_data_buffer = createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        mask_sizes_buffer = createBuffer(sizeof(int32_t) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        mask_offsets_buffer = createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        mask_IDs_buffer = createBuffer(sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uv_data_buffer = createBuffer(sizeof(float) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uv_IDs_buffer = createBuffer(sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        bbox_vertices_buffer = createBuffer(sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // Update descriptor set 3 (sky parameters) with placeholder buffers
        // Note: Geometry/material/result buffers don't exist yet, so we only update set 3
        VkDescriptorBufferInfo diffuse_flux_info{};
        diffuse_flux_info.buffer = diffuse_flux_buffer.buffer;
        diffuse_flux_info.offset = 0;
        diffuse_flux_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo diffuse_peak_dir_info{};
        diffuse_peak_dir_info.buffer = diffuse_peak_dir_buffer.buffer;
        diffuse_peak_dir_info.offset = 0;
        diffuse_peak_dir_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo diffuse_extinction_info{};
        diffuse_extinction_info.buffer = diffuse_extinction_buffer.buffer;
        diffuse_extinction_info.offset = 0;
        diffuse_extinction_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo diffuse_dist_norm_info{};
        diffuse_dist_norm_info.buffer = diffuse_dist_norm_buffer.buffer;
        diffuse_dist_norm_info.offset = 0;
        diffuse_dist_norm_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo sky_radiance_params_info{};
        sky_radiance_params_info.buffer = sky_radiance_params_buffer.buffer;
        sky_radiance_params_info.offset = 0;
        sky_radiance_params_info.range = VK_WHOLE_SIZE;

        std::vector<VkWriteDescriptorSet> descriptor_writes;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set_sky;
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &diffuse_flux_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 1;
        write.pBufferInfo = &diffuse_peak_dir_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 2;
        write.pBufferInfo = &diffuse_extinction_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 3;
        write.pBufferInfo = &diffuse_dist_norm_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 4;
        write.pBufferInfo = &sky_radiance_params_info;
        descriptor_writes.push_back(write);

        VkDescriptorBufferInfo camera_sky_radiance_info{};
        camera_sky_radiance_info.buffer = camera_sky_radiance_buffer.buffer;
        camera_sky_radiance_info.offset = 0;
        camera_sky_radiance_info.range = VK_WHOLE_SIZE;

        write.dstBinding = 5;
        write.pBufferInfo = &camera_sky_radiance_info;
        descriptor_writes.push_back(write);

        VkDescriptorBufferInfo solar_disk_radiance_info{};
        solar_disk_radiance_info.buffer = solar_disk_radiance_buffer.buffer;
        solar_disk_radiance_info.offset = 0;
        solar_disk_radiance_info.range = VK_WHOLE_SIZE;

        write.dstBinding = 6;
        write.pBufferInfo = &solar_disk_radiance_info;
        descriptor_writes.push_back(write);

        // Descriptor writes for camera result placeholder buffers (set_results bindings 5-9)
        VkDescriptorBufferInfo camera_radiation_info{};
        camera_radiation_info.buffer = camera_radiation_buffer.buffer;
        camera_radiation_info.offset = 0;
        camera_radiation_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_pixel_label_info{};
        camera_pixel_label_info.buffer = camera_pixel_label_buffer.buffer;
        camera_pixel_label_info.offset = 0;
        camera_pixel_label_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_pixel_depth_info{};
        camera_pixel_depth_info.buffer = camera_pixel_depth_buffer.buffer;
        camera_pixel_depth_info.offset = 0;
        camera_pixel_depth_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_scatter_top_info{};
        camera_scatter_top_info.buffer = camera_scatter_top_buffer.buffer;
        camera_scatter_top_info.offset = 0;
        camera_scatter_top_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_scatter_bottom_info{};
        camera_scatter_bottom_info.buffer = camera_scatter_bottom_buffer.buffer;
        camera_scatter_bottom_info.offset = 0;
        camera_scatter_bottom_info.range = VK_WHOLE_SIZE;

        write.dstSet = set_results;
        write.dstBinding = 5;
        write.pBufferInfo = &camera_radiation_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 6;
        write.pBufferInfo = &camera_pixel_label_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 7;
        write.pBufferInfo = &camera_pixel_depth_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 8;
        write.pBufferInfo = &camera_scatter_top_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 9;
        write.pBufferInfo = &camera_scatter_bottom_info;
        descriptor_writes.push_back(write);

        // Descriptor writes for mask/UV placeholder buffers (set_geometry bindings 11-16)
        VkDescriptorBufferInfo mask_data_info{};
        mask_data_info.buffer = mask_data_buffer.buffer;
        mask_data_info.offset = 0;
        mask_data_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo mask_sizes_info{};
        mask_sizes_info.buffer = mask_sizes_buffer.buffer;
        mask_sizes_info.offset = 0;
        mask_sizes_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo mask_offsets_info{};
        mask_offsets_info.buffer = mask_offsets_buffer.buffer;
        mask_offsets_info.offset = 0;
        mask_offsets_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo mask_IDs_info{};
        mask_IDs_info.buffer = mask_IDs_buffer.buffer;
        mask_IDs_info.offset = 0;
        mask_IDs_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo uv_data_info{};
        uv_data_info.buffer = uv_data_buffer.buffer;
        uv_data_info.offset = 0;
        uv_data_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo uv_IDs_info{};
        uv_IDs_info.buffer = uv_IDs_buffer.buffer;
        uv_IDs_info.offset = 0;
        uv_IDs_info.range = VK_WHOLE_SIZE;

        write.dstSet = set_geometry;
        write.dstBinding = 11;
        write.pBufferInfo = &mask_data_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 12;
        write.pBufferInfo = &mask_sizes_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 13;
        write.pBufferInfo = &mask_offsets_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 14;
        write.pBufferInfo = &mask_IDs_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 15;
        write.pBufferInfo = &uv_data_info;
        descriptor_writes.push_back(write);

        write.dstBinding = 16;
        write.pBufferInfo = &uv_IDs_info;
        descriptor_writes.push_back(write);

        VkDescriptorBufferInfo bbox_verts_info{};
        bbox_verts_info.buffer = bbox_vertices_buffer.buffer;
        bbox_verts_info.offset = 0;
        bbox_verts_info.range = VK_WHOLE_SIZE;

        write.dstBinding = 17;
        write.pBufferInfo = &bbox_verts_info;
        descriptor_writes.push_back(write);

        vkUpdateDescriptorSets(vk_device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
    }

    void VulkanComputeBackend::createPipelines() {
        VkDevice vk_device = device->getDevice();

        // ========== Create Pipeline Layout ==========

        VkDescriptorSetLayout set_layouts[] = {set_layout_geometry, set_layout_materials, set_layout_results, set_layout_sky, set_layout_debug};

        // Push constants (128 bytes max for MoltenVK compatibility)
        const uint32_t push_constant_size = 128;

        // Validate against device limits
        const VkPhysicalDeviceProperties &props = device->getDeviceProperties();
        if (push_constant_size > props.limits.maxPushConstantsSize) {
            helios_runtime_error("ERROR (VulkanComputeBackend::createPipelines): Push constant size (" + std::to_string(push_constant_size) + " bytes) exceeds device limit (" +
                                 std::to_string(props.limits.maxPushConstantsSize) + " bytes)");
        }

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 5; // geometry, materials, results, sky, debug
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

        VkDescriptorBufferInfo normal_info{};
        normal_info.buffer = normal_buffer.buffer;
        normal_info.offset = 0;
        normal_info.range = VK_WHOLE_SIZE;

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

        if (normal_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 10;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &normal_info;
            descriptor_writes.push_back(write);
        }

        // Mask/UV texture data buffers (bindings 11-16)
        VkDescriptorBufferInfo mask_data_info{};
        mask_data_info.buffer = mask_data_buffer.buffer;
        mask_data_info.offset = 0;
        mask_data_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo mask_sizes_info{};
        mask_sizes_info.buffer = mask_sizes_buffer.buffer;
        mask_sizes_info.offset = 0;
        mask_sizes_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo mask_offsets_info{};
        mask_offsets_info.buffer = mask_offsets_buffer.buffer;
        mask_offsets_info.offset = 0;
        mask_offsets_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo mask_IDs_info{};
        mask_IDs_info.buffer = mask_IDs_buffer.buffer;
        mask_IDs_info.offset = 0;
        mask_IDs_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo uv_data_info{};
        uv_data_info.buffer = uv_data_buffer.buffer;
        uv_data_info.offset = 0;
        uv_data_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo uv_IDs_info{};
        uv_IDs_info.buffer = uv_IDs_buffer.buffer;
        uv_IDs_info.offset = 0;
        uv_IDs_info.range = VK_WHOLE_SIZE;

        if (mask_data_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 11;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &mask_data_info;
            descriptor_writes.push_back(write);
        }

        if (mask_sizes_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 12;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &mask_sizes_info;
            descriptor_writes.push_back(write);
        }

        if (mask_offsets_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 13;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &mask_offsets_info;
            descriptor_writes.push_back(write);
        }

        if (mask_IDs_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 14;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &mask_IDs_info;
            descriptor_writes.push_back(write);
        }

        if (uv_data_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 15;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &uv_data_info;
            descriptor_writes.push_back(write);
        }

        if (uv_IDs_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 16;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &uv_IDs_info;
            descriptor_writes.push_back(write);
        }

        VkDescriptorBufferInfo bbox_verts_info{};
        bbox_verts_info.buffer = bbox_vertices_buffer.buffer;
        bbox_verts_info.offset = 0;
        bbox_verts_info.range = VK_WHOLE_SIZE;

        if (bbox_vertices_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_geometry;
            write.dstBinding = 17;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bbox_verts_info;
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

        // Specular property buffers
        VkDescriptorBufferInfo specular_exponent_info{};
        specular_exponent_info.buffer = specular_exponent_buffer.buffer;
        specular_exponent_info.offset = 0;
        specular_exponent_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo specular_scale_info{};
        specular_scale_info.buffer = specular_scale_buffer.buffer;
        specular_scale_info.offset = 0;
        specular_scale_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo source_fluxes_cam_info{};
        source_fluxes_cam_info.buffer = source_fluxes_cam_buffer.buffer;
        source_fluxes_cam_info.offset = 0;
        source_fluxes_cam_info.range = VK_WHOLE_SIZE;

        if (specular_exponent_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 7;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &specular_exponent_info;
            descriptor_writes.push_back(write);
        }

        if (specular_scale_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 8;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &specular_scale_info;
            descriptor_writes.push_back(write);
        }

        if (source_fluxes_cam_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 9;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &source_fluxes_cam_info;
            descriptor_writes.push_back(write);
        }

        VkDescriptorBufferInfo band_map_info{};
        band_map_info.buffer = band_map_buffer.buffer;
        band_map_info.offset = 0;
        band_map_info.range = VK_WHOLE_SIZE;

        if (band_map_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_materials;
            write.dstBinding = 10;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &band_map_info;
            descriptor_writes.push_back(write);
        }

        // ========== Set 2: Result Buffers ==========

        VkDescriptorBufferInfo rad_in_info{};
        rad_in_info.buffer = radiation_in_buffer.buffer;
        rad_in_info.offset = 0;
        rad_in_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo rad_out_top_info{};
        rad_out_top_info.buffer = radiation_out_top_buffer.buffer;
        rad_out_top_info.offset = 0;
        rad_out_top_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo rad_out_bottom_info{};
        rad_out_bottom_info.buffer = radiation_out_bottom_buffer.buffer;
        rad_out_bottom_info.offset = 0;
        rad_out_bottom_info.range = VK_WHOLE_SIZE;

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

        if (radiation_out_top_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 1;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &rad_out_top_info;
            descriptor_writes.push_back(write);
        } else {
            // radiation_out_top_buffer not yet allocated; descriptor update skipped
        }

        if (radiation_out_bottom_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 2;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &rad_out_bottom_info;
            descriptor_writes.push_back(write);
        }

        if (scatter_top_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 3;
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
            write.dstBinding = 4;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &scatter_bottom_info;
            descriptor_writes.push_back(write);
        }

        // Camera result buffers (bindings 5-9)
        VkDescriptorBufferInfo camera_radiation_info{};
        camera_radiation_info.buffer = camera_radiation_buffer.buffer;
        camera_radiation_info.offset = 0;
        camera_radiation_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_pixel_label_info{};
        camera_pixel_label_info.buffer = camera_pixel_label_buffer.buffer;
        camera_pixel_label_info.offset = 0;
        camera_pixel_label_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_pixel_depth_info{};
        camera_pixel_depth_info.buffer = camera_pixel_depth_buffer.buffer;
        camera_pixel_depth_info.offset = 0;
        camera_pixel_depth_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_scatter_top_info{};
        camera_scatter_top_info.buffer = camera_scatter_top_buffer.buffer;
        camera_scatter_top_info.offset = 0;
        camera_scatter_top_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo camera_scatter_bottom_info{};
        camera_scatter_bottom_info.buffer = camera_scatter_bottom_buffer.buffer;
        camera_scatter_bottom_info.offset = 0;
        camera_scatter_bottom_info.range = VK_WHOLE_SIZE;

        if (camera_radiation_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 5;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &camera_radiation_info;
            descriptor_writes.push_back(write);
        }

        if (camera_pixel_label_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 6;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &camera_pixel_label_info;
            descriptor_writes.push_back(write);
        }

        if (camera_pixel_depth_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 7;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &camera_pixel_depth_info;
            descriptor_writes.push_back(write);
        }

        if (camera_scatter_top_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 8;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &camera_scatter_top_info;
            descriptor_writes.push_back(write);
        }

        if (camera_scatter_bottom_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 9;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &camera_scatter_bottom_info;
            descriptor_writes.push_back(write);
        }

        VkDescriptorBufferInfo radiation_specular_info{};
        radiation_specular_info.buffer = radiation_specular_buffer.buffer;
        radiation_specular_info.offset = 0;
        radiation_specular_info.range = VK_WHOLE_SIZE;

        if (radiation_specular_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_results;
            write.dstBinding = 10;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &radiation_specular_info;
            descriptor_writes.push_back(write);
        }

        // ========== Set 3: Sky Parameters ==========

        VkDescriptorBufferInfo diffuse_flux_info{};
        diffuse_flux_info.buffer = diffuse_flux_buffer.buffer;
        diffuse_flux_info.offset = 0;
        diffuse_flux_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo diffuse_peak_dir_info{};
        diffuse_peak_dir_info.buffer = diffuse_peak_dir_buffer.buffer;
        diffuse_peak_dir_info.offset = 0;
        diffuse_peak_dir_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo diffuse_extinction_info{};
        diffuse_extinction_info.buffer = diffuse_extinction_buffer.buffer;
        diffuse_extinction_info.offset = 0;
        diffuse_extinction_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo diffuse_dist_norm_info{};
        diffuse_dist_norm_info.buffer = diffuse_dist_norm_buffer.buffer;
        diffuse_dist_norm_info.offset = 0;
        diffuse_dist_norm_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo sky_radiance_params_info{};
        sky_radiance_params_info.buffer = sky_radiance_params_buffer.buffer;
        sky_radiance_params_info.offset = 0;
        sky_radiance_params_info.range = VK_WHOLE_SIZE;

        if (diffuse_flux_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_sky;
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &diffuse_flux_info;
            descriptor_writes.push_back(write);
        }

        if (diffuse_peak_dir_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_sky;
            write.dstBinding = 1;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &diffuse_peak_dir_info;
            descriptor_writes.push_back(write);
        }

        if (diffuse_extinction_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_sky;
            write.dstBinding = 2;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &diffuse_extinction_info;
            descriptor_writes.push_back(write);
        }

        if (diffuse_dist_norm_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_sky;
            write.dstBinding = 3;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &diffuse_dist_norm_info;
            descriptor_writes.push_back(write);
        }

        if (sky_radiance_params_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_sky;
            write.dstBinding = 4;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &sky_radiance_params_info;
            descriptor_writes.push_back(write);
        }

        // Camera sky buffers (bindings 5-6)
        VkDescriptorBufferInfo camera_sky_radiance_info{};
        camera_sky_radiance_info.buffer = camera_sky_radiance_buffer.buffer;
        camera_sky_radiance_info.offset = 0;
        camera_sky_radiance_info.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo solar_disk_radiance_info{};
        solar_disk_radiance_info.buffer = solar_disk_radiance_buffer.buffer;
        solar_disk_radiance_info.offset = 0;
        solar_disk_radiance_info.range = VK_WHOLE_SIZE;

        if (camera_sky_radiance_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_sky;
            write.dstBinding = 5;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &camera_sky_radiance_info;
            descriptor_writes.push_back(write);
        }

        if (solar_disk_radiance_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_sky;
            write.dstBinding = 6;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &solar_disk_radiance_info;
            descriptor_writes.push_back(write);
        }

        // ========== Set 4: Debug Counters ==========

        VkDescriptorBufferInfo debug_counters_info{};
        debug_counters_info.buffer = debug_counters_buffer.buffer;
        debug_counters_info.offset = 0;
        debug_counters_info.range = VK_WHOLE_SIZE;

        if (debug_counters_buffer.buffer != VK_NULL_HANDLE) {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set_debug;
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &debug_counters_info;
            descriptor_writes.push_back(write);
        }

        // Apply all descriptor writes
        if (!descriptor_writes.empty()) {
            vkUpdateDescriptorSets(vk_device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
        }
    }

} // namespace helios
