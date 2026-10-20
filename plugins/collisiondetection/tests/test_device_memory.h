/** \file "test_device_memory.h" Test-only CUDA device-memory utilities for the collision detection test suite.
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

#pragma once

// Kept free of CUDA headers: cuda_runtime.h declares global int3/make_int3, which clash with helios::int3 in selfTest.cpp.

#include <cstddef>
#include <vector>

namespace CollisionTests {

    //! Free device memory in bytes, as reported by the CUDA driver
    size_t freeDeviceBytes();

    /**
     * \brief Holds (almost) all free device memory for its lifetime, so that the next device allocation fails
     * \note Allocates in halving block sizes down to 256 bytes, then clears the expected final allocation failure so it does
     * not linger as the thread's last CUDA error.
     */
    class DeviceMemoryHog {
    public:
        DeviceMemoryHog();
        ~DeviceMemoryHog();
        DeviceMemoryHog(const DeviceMemoryHog &) = delete;
        DeviceMemoryHog &operator=(const DeviceMemoryHog &) = delete;

    private:
        std::vector<void *> blocks;
    };

} // namespace CollisionTests
