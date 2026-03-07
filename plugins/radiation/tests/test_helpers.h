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

#ifdef HELIOS_HAVE_VULKAN
#include "VulkanDevice.h"
#endif

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
#ifdef HELIOS_HAVE_VULKAN
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

                // Allow forcing GPU tests to be skipped (e.g., for local CI simulation)
                // Usage: HELIOS_NO_GPU=1 ./run_tests.sh --test radiation
                const char *no_gpu = std::getenv("HELIOS_NO_GPU");
                if (no_gpu && std::string(no_gpu) != "0") {
                    std::cout << "[Vulkan Test] HELIOS_NO_GPU is set. Skipping GPU tests." << std::endl;
                    available = false;
                    return nullptr;
                }

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
#endif // HELIOS_HAVE_VULKAN

        /**
         * @brief Check if a functional Vulkan device is available for testing
         *
         * @return True if Vulkan device was successfully initialized and no runtime failure has occurred
         *
         * Triggers device initialization on first call if not already attempted.
         * Also returns false if a GPU runtime failure (e.g., VK_ERROR_DEVICE_LOST)
         * has been detected during test execution.
         */
        static bool isVulkanAvailable() {
#ifdef HELIOS_HAVE_VULKAN
            if (gpu_runtime_failed) {
                return false;
            }
            return getSharedDevice() != nullptr;
#else
            return false;
#endif
        }

        /**
         * @brief Mark GPU as unavailable due to a runtime failure (e.g., VK_ERROR_DEVICE_LOST)
         *
         * Called when a GPU test encounters a runtime failure that indicates the device
         * cannot actually execute compute work. Once marked, all subsequent GPU tests skip.
         */
        static void markGPURuntimeFailed() {
            gpu_runtime_failed = true;
        }

        // Disable construction - static-only interface
        TestVulkanDeviceManager() = delete;
        TestVulkanDeviceManager(const TestVulkanDeviceManager &) = delete;
        TestVulkanDeviceManager &operator=(const TestVulkanDeviceManager &) = delete;

    private:
        static inline bool gpu_runtime_failed = false;
    };

} // namespace helios

// ============================================================================
// GPU_TEST_CASE macro: wraps a doctest test case with GPU availability check
// and runtime DEVICE_LOST detection.
//
// Usage (replaces DOCTEST_TEST_CASE + SKIP_IF_NO_GPU):
//   GPU_TEST_CASE("test name") {
//       // test body — no SKIP_IF_NO_GPU() needed
//   }
//
// Behavior:
//   1. If GPU is known unavailable (no driver, HELIOS_NO_GPU, or previous DEVICE_LOST),
//      the test immediately reports SKIPPED and returns.
//   2. Otherwise, runs the test body inside a try/catch. If the body throws an exception
//      containing "DEVICE_LOST" or "VkResult: -4", marks the GPU as failed at runtime
//      so all subsequent GPU tests skip instead of repeating the same failure.
//   3. Any other exception is re-thrown for doctest to handle normally.
// ============================================================================

#define GPU_TEST_CASE_IMPL(name, unique_inner)                                                              \
    static void unique_inner();                                                                             \
    DOCTEST_TEST_CASE(name) {                                                                               \
        using helios::RadiationModelTestHelper;                                                             \
        using helios::TestVulkanDeviceManager;                                                              \
        if (!RadiationModelTestHelper::isGPUAvailable()) {                                                  \
            DOCTEST_MESSAGE("SKIPPED: No GPU/Vulkan device available");                                     \
            return;                                                                                         \
        }                                                                                                   \
        try {                                                                                               \
            unique_inner();                                                                                 \
        } catch (const std::exception &_gpu_ex) {                                                           \
            std::string _gpu_msg(_gpu_ex.what());                                                           \
            if (_gpu_msg.find("DEVICE_LOST") != std::string::npos ||                                        \
                _gpu_msg.find("VkResult: -4") != std::string::npos ||                                       \
                _gpu_msg.find("VkResult: -9") != std::string::npos) {                                       \
                TestVulkanDeviceManager::markGPURuntimeFailed();                                             \
                std::cout << "[Vulkan Test] GPU device lost: " << _gpu_msg << std::endl;                    \
                std::cout << "[Vulkan Test] Remaining GPU tests will be skipped." << std::endl;             \
                DOCTEST_MESSAGE("SKIPPED: GPU device lost during execution");                               \
                return;                                                                                     \
            }                                                                                               \
            throw;                                                                                          \
        }                                                                                                   \
    }                                                                                                       \
    static void unique_inner()

#define GPU_TEST_CASE(name) \
    GPU_TEST_CASE_IMPL(name, DOCTEST_ANONYMOUS(DOCTEST_ANON_GPU_FUNC_))
