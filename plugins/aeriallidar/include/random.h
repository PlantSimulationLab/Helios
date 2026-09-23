/** \file "random.h" Device-side pseudo-random number generation for the aerial LiDAR plug-in.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#ifndef HELIOS_AERIALLIDAR_RANDOM_H
#define HELIOS_AERIALLIDAR_RANDOM_H

//! Hash two 32-bit values into a per-thread seed using N rounds of the Tiny Encryption Algorithm
/**
 * TEA as described in Wheeler & Needham (1994), "TEA, a Tiny Encryption Algorithm", with the fixed 128-bit key used for GPU
 * seeding by Zafar, Olano & Curtis (2010), "GPU random numbers via the tiny encryption algorithm", Proc. HPG 2010.
 * \param[in] val0 First value to hash (e.g., sub-pulse index)
 * \param[in] val1 Second value to hash (e.g., global ray index)
 * \return First word of the encrypted pair, used as an LCG seed
 */
template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1) {
    const unsigned int delta = 0x9e3779b9u; // TEA key-schedule constant, floor(2^32 / golden ratio)
    const unsigned int key[4] = {0xa341316cu, 0xc8013ea4u, 0xad90777du, 0x7e95761eu};

    unsigned int word0 = val0;
    unsigned int word1 = val1;
    unsigned int sum = 0;
    for (unsigned int round = 0; round < N; round++) {
        sum += delta;
        word0 += ((word1 << 4) + key[0]) ^ (word1 + sum) ^ ((word1 >> 5) + key[1]);
        word1 += ((word0 << 4) + key[2]) ^ (word0 + sum) ^ ((word0 >> 5) + key[3]);
    }
    return word0;
}

//! Advance a linear congruential generator and return a 24-bit random integer in [0, 2^24)
/**
 * Uses the multiplier and increment of the "quick and dirty" generator in Press et al., Numerical Recipes in C (2nd ed.), Sec. 7.1.
 * \param[inout] state Generator state, updated in place
 */
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &state) {
    state = 1664525u * state + 1013904223u;
    return state & 0x00FFFFFFu;
}

//! Return a uniformly distributed random float in [0, 1)
/**
 * \param[inout] state Generator state, updated in place
 */
static __host__ __device__ __inline__ float rnd(unsigned int &state) {
    return static_cast<float>(lcg(state)) / static_cast<float>(0x01000000u);
}

#endif // HELIOS_AERIALLIDAR_RANDOM_H
