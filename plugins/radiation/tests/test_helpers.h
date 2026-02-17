/** \file "test_helpers.h" Test-only utilities for radiation test suite.
 *
 *    Copyright (C) 2016-2026 Brian Bailey
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, version 2.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 */

#pragma once
#include "VulkanDevice.h"
#include <memory>
#include <cstdlib>

namespace helios {

/**
 * @brief Shared Vulkan device singleton for test process
 *
 * INTERNAL TEST-ONLY HELPER - NOT FOR PRODUCTION USE
 *
 * Purpose: Work around NVIDIA driver bug where >25 vkCreateInstance/vkDestroyInstance
 * cycles in same process cause VK_ERROR_INCOMPATIBLE_DRIVER.
 *
 * Solution: Share single VulkanDevice across all 78 tests in the binary.
 * Device created on first use, destroyed at process exit via atexit().
 *
 * Thread Safety: Not thread-safe (tests run sequentially in selfTest.cpp).
 *
 * Validation: Disabled for test performance (20-30% faster).
 *
 * References:
 * - https://forums.developer.nvidia.com/t/issue-with-repeated-instance-creation-in-one-process/176978
 * - https://github.com/qdrant/qdrant/pull/6031
 */
class TestVulkanDeviceManager {
public:
    /**
     * @brief Get shared test VulkanDevice instance
     *
     * @return Non-owning pointer to shared VulkanDevice (valid until process exit)
     *
     * First call initializes device without validation. Subsequent calls return same instance.
     * Device is destroyed at process exit via registered atexit() handler.
     */
    static VulkanDevice* getSharedDevice() {
        static std::unique_ptr<VulkanDevice> shared_device;
        static bool initialized = false;

        if (!initialized) {
            shared_device = std::make_unique<VulkanDevice>();
            shared_device->initialize(false);  // No validation for test performance
            initialized = true;

            // Register cleanup at process exit for clean driver state
            std::atexit([]() {
                if (shared_device) {
                    shared_device->shutdown();
                    shared_device.reset();
                }
            });
        }

        return shared_device.get();
    }

    // Disable construction - static-only interface
    TestVulkanDeviceManager() = delete;
    TestVulkanDeviceManager(const TestVulkanDeviceManager&) = delete;
    TestVulkanDeviceManager& operator=(const TestVulkanDeviceManager&) = delete;
};

}  // namespace helios
