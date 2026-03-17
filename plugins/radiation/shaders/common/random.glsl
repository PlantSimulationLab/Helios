/** random.glsl - Random number generation (TEA + LCG)
 *
 * Matches CUDA implementation for consistency.
 */

#ifndef RANDOM_GLSL
#define RANDOM_GLSL

// TEA (Tiny Encryption Algorithm) - used as hash function for seeding
uint tea(uint val0, uint val1) {
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;

    for(uint n = 0; n < 16; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Linear Congruential Generator state
struct RNGState {
    uint seed;
};

// Initialize RNG state (matches CUDA: tea(index, seed))
RNGState rng_init(uint seed, uint index) {
    RNGState state;
    state.seed = tea(index, seed);
    return state;
}

// Generate random uint [0, 2^24-1] (matches CUDA LCG with 24-bit mask)
uint rng_uint(inout RNGState state) {
    state.seed = (1664525u * state.seed + 1013904223u);
    return state.seed & 0x00FFFFFFu;
}

// Generate random float [0, 1)
float rng_float(inout RNGState state) {
    return float(rng_uint(state)) / 16777216.0;
}

// Generate random float [min, max)
float rng_range(inout RNGState state, float min_val, float max_val) {
    return min_val + rng_float(state) * (max_val - min_val);
}

#endif // RANDOM_GLSL
