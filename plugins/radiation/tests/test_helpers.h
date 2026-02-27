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
#include <iostream>
#include <stdexcept>

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
     * If Vulkan is not available (e.g., CI runners without GPUs), initialization
     * fails gracefully and isVulkanAvailable() returns false.
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
         * @return Non-owning pointer to shared VulkanDevice, or nullptr if Vulkan is unavailable
         *
         * First call attempts to initialize device without validation. If initialization
         * fails (no GPU, no driver, etc.), returns nullptr and all subsequent calls also
         * return nullptr. Device is destroyed at process exit via registered atexit() handler.
         */
        static VulkanDevice *getSharedDevice() {
            static std::unique_ptr<VulkanDevice> shared_device;
            static bool initialized = false;
            static bool available = false;

            if (!initialized) {
                initialized = true;
                try {
                    shared_device = std::make_unique<VulkanDevice>();
                    shared_device->initialize(false); // No validation for test performance
                    available = true;

                    // Register cleanup at process exit for clean driver state
                    std::atexit([]() {
                        if (shared_device) {
                            shared_device->shutdown();
                            shared_device.reset();
                        }
                    });
                } catch (const std::exception &e) {
                    std::cout << "[Vulkan Test] No Vulkan device available: " << e.what() << std::endl;
                    std::cout << "[Vulkan Test] GPU-dependent radiation tests will be skipped." << std::endl;
                    shared_device.reset();
                    available = false;
                }
            }

            return available ? shared_device.get() : nullptr;
        }

        /**
         * @brief Check if a functional Vulkan device is available for testing
         *
         * @return True if Vulkan device was successfully initialized
         *
         * Triggers device initialization on first call if not already attempted.
         */
        static bool isVulkanAvailable() {
            return getSharedDevice() != nullptr;
        }

        // Disable construction - static-only interface
        TestVulkanDeviceManager() = delete;
        TestVulkanDeviceManager(const TestVulkanDeviceManager &) = delete;
        TestVulkanDeviceManager &operator=(const TestVulkanDeviceManager &) = delete;
    };

} // namespace helios
