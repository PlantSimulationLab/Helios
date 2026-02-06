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

// Initialize RNG state
RNGState rng_init(uint seed, uint index) {
    RNGState state;
    state.seed = tea(seed, index);
    return state;
}

// Generate random uint [0, 2^32-1]
uint rng_uint(inout RNGState state) {
    state.seed = (1103515245u * state.seed + 12345u);
    return state.seed;
}

// Generate random float [0, 1)
float rng_float(inout RNGState state) {
    return float(rng_uint(state)) * (1.0 / 4294967296.0);
}

// Generate random float [min, max)
float rng_range(inout RNGState state, float min_val, float max_val) {
    return min_val + rng_float(state) * (max_val - min_val);
}

#endif // RANDOM_GLSL
