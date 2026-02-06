/** \file "VulkanDevice.h" Vulkan device and instance management for ray tracing backend.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef VULKAN_DEVICE_H
#define VULKAN_DEVICE_H

#include "global.h"
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

#include <vector>
#include <string>

namespace helios {

    /**
     * @brief Manages Vulkan instance, physical device, logical device, and memory allocator
     *
     * Handles:
     * - Vulkan instance creation with required extensions (MoltenVK-aware)
     * - Physical device selection (prefer discrete GPU, fall back to integrated)
     * - Logical device creation with compute queue
     * - VMA allocator initialization for efficient GPU memory management
     * - Feature detection (atomic floats, etc.)
     */
    class VulkanDevice {
    public:
        VulkanDevice();
        ~VulkanDevice();

        // Delete copy/move constructors - this class manages Vulkan resources
        VulkanDevice(const VulkanDevice &) = delete;
        VulkanDevice &operator=(const VulkanDevice &) = delete;
        VulkanDevice(VulkanDevice &&) = delete;
        VulkanDevice &operator=(VulkanDevice &&) = delete;

        /**
         * @brief Initialize Vulkan instance, device, and allocator
         *
         * @param enable_validation Enable Vulkan validation layers (for debugging)
         *
         * Creates instance, selects physical device, creates logical device, and initializes VMA.
         */
        void initialize(bool enable_validation = false);

        /**
         * @brief Shutdown and destroy all Vulkan resources
         *
         * Destroys VMA allocator, logical device, and instance in correct order.
         */
        void shutdown();

        /**
         * @brief Check if device supports atomic float operations
         *
         * @return True if GL_EXT_shader_atomic_float is supported
         *
         * Used to select optimized shader variant (atomic floats vs atomicCompSwap fallback).
         */
        bool supportsAtomicFloat() const { return has_atomic_float; }

        /**
         * @brief Check if running on MoltenVK (macOS/iOS)
         *
         * @return True if MoltenVK driver detected
         *
         * Used for platform-specific workarounds (push constant limits, etc.).
         */
        bool isMoltenVK() const { return is_moltenvk; }

        /**
         * @brief Get Vulkan instance
         */
        VkInstance getInstance() const { return instance; }

        /**
         * @brief Get physical device
         */
        VkPhysicalDevice getPhysicalDevice() const { return physical_device; }

        /**
         * @brief Get logical device
         */
        VkDevice getDevice() const { return device; }

        /**
         * @brief Get compute queue
         */
        VkQueue getComputeQueue() const { return compute_queue; }

        /**
         * @brief Get compute queue family index
         */
        uint32_t getComputeQueueFamily() const { return compute_queue_family; }

        /**
         * @brief Get VMA allocator
         */
        VmaAllocator getAllocator() const { return allocator; }

        /**
         * @brief Get device properties (name, limits, etc.)
         */
        const VkPhysicalDeviceProperties &getDeviceProperties() const { return device_properties; }

        /**
         * @brief Print device information (name, vendor, driver version)
         */
        void printDeviceInfo() const;

    private:
        // Vulkan handles
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue compute_queue = VK_NULL_HANDLE;
        uint32_t compute_queue_family = 0;

        // Memory allocator
        VmaAllocator allocator = VK_NULL_HANDLE;

        // Device properties and features
        VkPhysicalDeviceProperties device_properties{};
        bool has_atomic_float = false;
        bool is_moltenvk = false;

        // Optional: validation layers
        VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

        /**
         * @brief Create Vulkan instance with required extensions
         */
        void createInstance(bool enable_validation);

        /**
         * @brief Select best physical device (prefer discrete GPU)
         */
        void selectPhysicalDevice();

        /**
         * @brief Create logical device with compute queue
         */
        void createLogicalDevice();

        /**
         * @brief Initialize VMA allocator
         */
        void createAllocator();

        /**
         * @brief Setup debug messenger for validation layers
         */
        void setupDebugMessenger();

        /**
         * @brief Find queue family with compute support
         *
         * @return Queue family index, or UINT32_MAX if not found
         */
        uint32_t findComputeQueueFamily(VkPhysicalDevice device_candidate) const;

        /**
         * @brief Score physical device (discrete > integrated > other)
         *
         * @return Score (higher is better), or 0 if unsuitable
         */
        uint32_t scorePhysicalDevice(VkPhysicalDevice device_candidate) const;

        /**
         * @brief Check if device supports required features and extensions
         */
        bool isDeviceSuitable(VkPhysicalDevice device_candidate) const;

        /**
         * @brief Detect MoltenVK driver
         */
        void detectMoltenVK();

        /**
         * @brief Detect atomic float support
         */
        void detectAtomicFloat();
    };

} // namespace helios

#endif // VULKAN_DEVICE_H
