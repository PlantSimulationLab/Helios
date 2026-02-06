/** \file "VulkanDevice.cpp" Vulkan device and instance management implementation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "VulkanDevice.h"
#include <algorithm>
#include <cstring>
#include <iostream>

// Suppress nullability warnings from VMA header on macOS
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace helios {

    // ========== Debug messenger callback ==========

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                         const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {
        std::string severity_str;
        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            severity_str = "ERROR";
        } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            severity_str = "WARNING";
        } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
            severity_str = "INFO";
        } else {
            severity_str = "VERBOSE";
        }

        // Use std::cout instead of std::cerr (Helios convention)
        std::cout << "[Vulkan " << severity_str << "] " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

    // ========== VulkanDevice implementation ==========

    VulkanDevice::VulkanDevice() = default;

    VulkanDevice::~VulkanDevice() {
        shutdown();
    }

    void VulkanDevice::initialize(bool enable_validation) {
        createInstance(enable_validation);
        selectPhysicalDevice();
        createLogicalDevice();
        createAllocator();

        detectMoltenVK();
        detectAtomicFloat();

        printDeviceInfo();
    }

    void VulkanDevice::shutdown() {
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
        }

        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }

        if (debug_messenger != VK_NULL_HANDLE) {
            auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
            if (func != nullptr) {
                func(instance, debug_messenger, nullptr);
            }
            debug_messenger = VK_NULL_HANDLE;
        }

        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }

    void VulkanDevice::createInstance(bool enable_validation) {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Helios Radiation Plugin";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 3, 0);
        app_info.pEngineName = "Helios";
        app_info.engineVersion = VK_MAKE_VERSION(1, 3, 0);
        app_info.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        // Required extensions (none for headless compute)
        std::vector<const char *> extensions;

        // Validation layers for debugging
        std::vector<const char *> validation_layers;
        if (enable_validation) {
            validation_layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // MoltenVK requires VK_KHR_portability_enumeration on macOS
#ifdef __APPLE__
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();

        VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanDevice::createInstance): Failed to create Vulkan instance. "
                                 "Make sure Vulkan SDK is installed. VkResult code: " +
                                 std::to_string(result));
        }

        if (enable_validation) {
            setupDebugMessenger();
        }
    }

    void VulkanDevice::setupDebugMessenger() {
        VkDebugUtilsMessengerCreateInfoEXT create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create_info.pfnUserCallback = debugCallback;

        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance, &create_info, nullptr, &debug_messenger);
        }
    }

    void VulkanDevice::selectPhysicalDevice() {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

        if (device_count == 0) {
            helios_runtime_error("ERROR (VulkanDevice::selectPhysicalDevice): No Vulkan-capable GPU found. "
                                 "Ensure your GPU drivers support Vulkan 1.1 or higher.");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

        // Score devices and select the best one
        uint32_t best_score = 0;
        VkPhysicalDevice best_device = VK_NULL_HANDLE;

        for (const auto &device_candidate : devices) {
            if (!isDeviceSuitable(device_candidate)) {
                continue;
            }

            uint32_t score = scorePhysicalDevice(device_candidate);
            if (score > best_score) {
                best_score = score;
                best_device = device_candidate;
            }
        }

        if (best_device == VK_NULL_HANDLE) {
            helios_runtime_error("ERROR (VulkanDevice::selectPhysicalDevice): No suitable Vulkan device found. "
                                 "GPU must support compute queues and Vulkan 1.1.");
        }

        physical_device = best_device;
        vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    }

    bool VulkanDevice::isDeviceSuitable(VkPhysicalDevice device_candidate) const {
        // Must have compute queue
        uint32_t queue_family = findComputeQueueFamily(device_candidate);
        if (queue_family == UINT32_MAX) {
            return false;
        }

        // Must support Vulkan 1.1 minimum
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device_candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_1) {
            return false;
        }

        return true;
    }

    uint32_t VulkanDevice::scorePhysicalDevice(VkPhysicalDevice device_candidate) const {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device_candidate, &props);

        uint32_t score = 0;

        // Discrete GPUs have a huge performance advantage
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 10000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score += 1000;
        }

        // Prefer newer Vulkan API versions
        score += VK_VERSION_MAJOR(props.apiVersion) * 100 + VK_VERSION_MINOR(props.apiVersion) * 10;

        return score;
    }

    uint32_t VulkanDevice::findComputeQueueFamily(VkPhysicalDevice device_candidate) const {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &queue_family_count, queue_families.data());

        for (uint32_t i = 0; i < queue_family_count; i++) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                return i;
            }
        }

        return UINT32_MAX;
    }

    void VulkanDevice::createLogicalDevice() {
        compute_queue_family = findComputeQueueFamily(physical_device);

        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = compute_queue_family;
        queue_create_info.queueCount = 1;
        float queue_priority = 1.0f;
        queue_create_info.pQueuePriorities = &queue_priority;

        VkPhysicalDeviceFeatures device_features{};
        // No special features required for baseline compute

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.pQueueCreateInfos = &queue_create_info;
        create_info.queueCreateInfoCount = 1;
        create_info.pEnabledFeatures = &device_features;

        // Required device extensions
        std::vector<const char *> device_extensions;

#ifdef __APPLE__
        // MoltenVK requires portability subset extension
        device_extensions.push_back("VK_KHR_portability_subset");
#endif

        create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        create_info.ppEnabledExtensionNames = device_extensions.data();

        VkResult result = vkCreateDevice(physical_device, &create_info, nullptr, &device);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanDevice::createLogicalDevice): Failed to create logical device. "
                                 "VkResult code: " +
                                 std::to_string(result));
        }

        vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);
    }

    void VulkanDevice::createAllocator() {
        VmaAllocatorCreateInfo allocator_info{};
        allocator_info.vulkanApiVersion = VK_API_VERSION_1_1;
        allocator_info.physicalDevice = physical_device;
        allocator_info.device = device;
        allocator_info.instance = instance;

        VkResult result = vmaCreateAllocator(&allocator_info, &allocator);
        if (result != VK_SUCCESS) {
            helios_runtime_error("ERROR (VulkanDevice::createAllocator): Failed to create VMA allocator. "
                                 "VkResult code: " +
                                 std::to_string(result));
        }
    }

    void VulkanDevice::detectMoltenVK() {
        // Method 1: Check for Apple vendor ID (0x106B)
        if (device_properties.vendorID == 0x106B) {
            is_moltenvk = true;
            return;
        }

        // Method 2: Check for MoltenVK-specific extension
        uint32_t ext_count;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, exts.data());

        for (const auto &ext : exts) {
            if (std::strcmp(ext.extensionName, "VK_MVK_moltenvk") == 0) {
                is_moltenvk = true;
                return;
            }
        }

        // Method 3: Fallback to device name check
        const char *device_name = device_properties.deviceName;
        is_moltenvk = (std::strstr(device_name, "MoltenVK") != nullptr);
    }

    void VulkanDevice::detectAtomicFloat() {
        // First check if VK_EXT_shader_atomic_float extension is available
        uint32_t ext_count;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, exts.data());

        bool has_extension = false;
        for (const auto &ext : exts) {
            if (std::strcmp(ext.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0) {
                has_extension = true;
                break;
            }
        }

        if (!has_extension) {
            has_atomic_float = false;
            return; // Extension not available, use atomicCompSwap fallback
        }

        // Extension present, query features
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &atomic_float_features;

        vkGetPhysicalDeviceFeatures2(physical_device, &features2);

        has_atomic_float = atomic_float_features.shaderBufferFloat32AtomicAdd;
    }

    void VulkanDevice::printDeviceInfo() const {
        std::cout << "========== Vulkan Device Information ==========" << std::endl;
        std::cout << "Device name: " << device_properties.deviceName << std::endl;
        std::cout << "Vendor ID: 0x" << std::hex << device_properties.vendorID << std::dec << std::endl;
        std::cout << "Device ID: 0x" << std::hex << device_properties.deviceID << std::dec << std::endl;
        std::cout << "Driver version: " << VK_VERSION_MAJOR(device_properties.driverVersion) << "." << VK_VERSION_MINOR(device_properties.driverVersion) << "." << VK_VERSION_PATCH(device_properties.driverVersion)
                  << std::endl;
        std::cout << "Vulkan API version: " << VK_VERSION_MAJOR(device_properties.apiVersion) << "." << VK_VERSION_MINOR(device_properties.apiVersion) << "." << VK_VERSION_PATCH(device_properties.apiVersion)
                  << std::endl;

        std::string device_type_str;
        switch (device_properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            device_type_str = "Discrete GPU";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            device_type_str = "Integrated GPU";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            device_type_str = "Virtual GPU";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            device_type_str = "CPU";
            break;
        default:
            device_type_str = "Other";
            break;
        }

        std::cout << "Device type: " << device_type_str << std::endl;
        std::cout << "Atomic float support: " << (has_atomic_float ? "YES" : "NO (using atomicCompSwap fallback)") << std::endl;
        std::cout << "MoltenVK detected: " << (is_moltenvk ? "YES" : "NO") << std::endl;
        std::cout << "===============================================" << std::endl;
    }

} // namespace helios
