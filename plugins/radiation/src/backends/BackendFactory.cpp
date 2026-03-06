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

} // namespace helios
