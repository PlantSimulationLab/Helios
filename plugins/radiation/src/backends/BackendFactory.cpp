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
#include "OptiX6Backend.h"

// Future backend implementations
// #include "OptiX7Backend.h"
// #include "VulkanBackend.h"

namespace helios {

    std::unique_ptr<RayTracingBackend> RayTracingBackend::create(const std::string &backend_type) {

        // OptiX 6.5 backend (current implementation)
        if (backend_type == "optix6" || backend_type == "OptiX6" || backend_type == "optix" || backend_type == "OptiX") {
            return std::make_unique<OptiX6Backend>();
        }

        // Future backends:

        // OptiX 7.7 backend
        // else if (backend_type == "optix7" || backend_type == "OptiX7") {
        //     return std::make_unique<OptiX7Backend>();
        // }

        // Vulkan ray-tracing backend
        // else if (backend_type == "vulkan" || backend_type == "Vulkan") {
        //     return std::make_unique<VulkanBackend>();
        // }

        // Unknown backend type
        else {
            helios_runtime_error("ERROR (RayTracingBackend::create): Unknown ray tracing backend type '" + backend_type +
                                 "'. "
                                 "Supported backends: 'optix6' (OptiX 6.5)");
        }

        // Unreachable, but silence compiler warning
        return nullptr;
    }

} // namespace helios
