/** material.glsl - Material property lookups
 *
 * Helpers for accessing material buffers with correct indexing.
 */

#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

// Placeholder material functions (to be implemented in Phase 2+)

float get_reflectivity(uint source, uint band, uint prim, uint Nbands, uint Nprims) {
    // TODO: Read from reflectivity buffer
    return 0.0;
}

float get_transmissivity(uint source, uint band, uint prim, uint Nbands, uint Nprims) {
    // TODO: Read from transmissivity buffer
    return 0.0;
}

float get_specular_exponent(uint prim) {
    // TODO: Read from specular_exponent buffer
    return 10.0;
}

float get_specular_scale(uint prim) {
    // TODO: Read from specular_scale buffer
    return 0.0;
}

#endif // MATERIAL_GLSL
