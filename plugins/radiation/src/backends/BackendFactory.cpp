/** \file "BackendFactory.cpp" Factory for creating ray tracing backend instances.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RayTracingBackend.h"

// Conditionally include backend headers based on compile-time definitions
#ifdef HELIOS_HAVE_OPTIX8
#include "OptiX8Backend.h"
#endif

#ifdef HELIOS_HAVE_OPTIX
#include "OptiX6Backend.h"
#endif

#ifdef HELIOS_HAVE_VULKAN
#include "VulkanComputeBackend.h"
#endif

namespace helios {

    std::unique_ptr<RayTracingBackend> RayTracingBackend::create(const std::string &backend_type) {

        // Auto-detect: probe compiled-in backends in priority order at runtime.
        // Note: probe() verifies driver/hardware availability but not full initialization
        // capability (e.g., missing device code files, insufficient VRAM). If initialize()
        // fails after a successful probe, the error propagates directly — no fallthrough
        // to the next backend.
        if (backend_type == "auto") {
#ifdef HELIOS_HAVE_OPTIX8
            if (OptiX8Backend::probe()) {
                return std::make_unique<OptiX8Backend>();
            }
#endif
#ifdef HELIOS_HAVE_OPTIX
            if (OptiX6Backend::probe()) {
                return std::make_unique<OptiX6Backend>();
            }
#endif
#ifdef HELIOS_HAVE_VULKAN
            if (VulkanComputeBackend::probe()) {
                return std::make_unique<VulkanComputeBackend>();
            }
#endif
            std::string compiled_backends;
#ifdef HELIOS_HAVE_OPTIX8
            compiled_backends += "OptiX 8.1 (NVIDIA drivers >= 560)";
#endif
#ifdef HELIOS_HAVE_OPTIX
            if (!compiled_backends.empty()) compiled_backends += ", ";
            compiled_backends += "OptiX 6.5 (NVIDIA drivers < 590)";
#endif
#ifdef HELIOS_HAVE_VULKAN
            if (!compiled_backends.empty()) compiled_backends += ", ";
            compiled_backends += "Vulkan Compute (AMD/Intel/Apple Silicon)";
#endif
            if (compiled_backends.empty()) {
                compiled_backends = "(none)";
            }

            helios_runtime_error(
                "ERROR (RayTracingBackend::create): No compatible GPU backend found.\n\n"
                "The radiation plugin requires a GPU with one of the following:\n"
                "  - NVIDIA GPU with CUDA drivers (for OptiX backend)\n"
                "  - AMD, Intel, or Apple Silicon GPU with Vulkan support\n\n"
                "Backends compiled into this build: " + compiled_backends + "\n"
                "All failed hardware probing. Ensure GPU drivers are installed and up to date.\n\n"
                "To diagnose:\n"
                "  - NVIDIA: run 'nvidia-smi' to verify driver is loaded\n"
                "  - Vulkan: run 'vulkaninfo --summary' to verify Vulkan support"
            );
        }

#ifdef HELIOS_HAVE_OPTIX8
        // OptiX 8.1 backend (explicit request)
        if (backend_type == "optix8" || backend_type == "OptiX8") {
            return std::make_unique<OptiX8Backend>();
        }
#endif

#ifdef HELIOS_HAVE_OPTIX8
        // "optix" / "optix6" auto-selects: prefer OptiX 8 on modern drivers
        if (backend_type == "optix" || backend_type == "OptiX" || backend_type == "optix6" || backend_type == "OptiX6") {
            return std::make_unique<OptiX8Backend>();
        }
#elif defined(HELIOS_HAVE_OPTIX)
        // OptiX 6.5 backend (legacy drivers < 560)
        if (backend_type == "optix6" || backend_type == "OptiX6" || backend_type == "optix" || backend_type == "OptiX") {
            return std::make_unique<OptiX6Backend>();
        }
#endif

#ifdef HELIOS_HAVE_VULKAN
        // Vulkan compute backend (software BVH traversal)
        if (backend_type == "vulkan_compute" || backend_type == "vulkan" || backend_type == "Vulkan") {
            return std::make_unique<VulkanComputeBackend>();
        }
#endif

        // Unknown backend type - provide helpful error message listing available backends
        std::string available_backends;
#ifdef HELIOS_HAVE_OPTIX8
        available_backends += "'optix8' (OptiX 8.1), 'optix6' (auto-selects OptiX 8.1)";
#elif defined(HELIOS_HAVE_OPTIX)
        available_backends += "'optix6' (OptiX 6.5)";
#endif
#ifdef HELIOS_HAVE_VULKAN
        if (!available_backends.empty()) {
            available_backends += ", ";
        }
        available_backends += "'vulkan_compute' (Vulkan compute shaders)";
#endif

        helios_runtime_error("ERROR (RayTracingBackend::create): Unknown ray tracing backend type '" + backend_type +
                             "'. "
                             "Supported backends: " +
                             available_backends);

        // Unreachable, but silence compiler warning
        return nullptr;
    }

    bool probeAnyGPUBackend() noexcept {
#ifdef HELIOS_HAVE_OPTIX8
        if (OptiX8Backend::probe()) return true;
#endif
#ifdef HELIOS_HAVE_OPTIX
        if (OptiX6Backend::probe()) return true;
#endif
#ifdef HELIOS_HAVE_VULKAN
        if (VulkanComputeBackend::probe()) return true;
#endif
        return false;
    }

} // namespace helios
