/** \file "test_device_memory.cpp" Test-only CUDA device-memory utilities for the collision detection test suite.
 *
 *    Copyright (C) 2016-2026 Brian Bailey
 *
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 2.1 of the License, or (at your option) any later version.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */

#include "test_device_memory.h"
#include <cuda_runtime.h>
#include "global.h"

namespace CollisionTests {

    size_t freeDeviceBytes() {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            helios::helios_runtime_error("ERROR (CollisionTests::freeDeviceBytes): cudaMemGetInfo failed (" + std::string(cudaGetErrorString(err)) + ").");
        }
        return free_bytes;
    }

    DeviceMemoryHog::DeviceMemoryHog() {
        size_t block_bytes = size_t(1) << 30;
        while (block_bytes >= 256) {
            void *block = nullptr;
            if (cudaMalloc(&block, block_bytes) == cudaSuccess) {
                blocks.push_back(block);
            } else {
                block_bytes /= 2;
            }
        }
        cudaGetLastError();
    }

    DeviceMemoryHog::~DeviceMemoryHog() {
        for (void *block: blocks) {
            cudaFree(block);
        }
    }

} // namespace CollisionTests
