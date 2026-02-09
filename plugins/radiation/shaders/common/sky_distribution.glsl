/** sky_distribution.glsl - Sky angular distribution models
 *
 * Implements three sky radiance distribution models:
 * 1. Isotropic (uniform hemispherical radiation)
 * 2. Power-law (Harrison & Coombes circumsolar model)
 * 3. Prague sky model (Perez et al. - circumsolar + horizon brightening)
 */

#ifndef SKY_DISTRIBUTION_GLSL
#define SKY_DISTRIBUTION_GLSL

const float PI = 3.14159265359;
const float DEG_TO_RAD = PI / 180.0;
const float RAD_TO_DEG = 180.0 / PI;

// Safe acos that clamps input to [-1, 1] to avoid NaN
float acos_safe(float x) {
    return acos(clamp(x, -1.0, 1.0));
}

// Isotropic sky distribution (uniform)
// Returns: 1.0 (constant radiance)
float sky_isotropic() {
    // For isotropic diffuse with cosine-weighted sampling (PDF = cos×sin/π):
    // The π from the PDF denominator appears in the Monte Carlo weight
    // This ensures proper hemispherical integration
    return 1.0;
}

// Power-law angular distribution (Harrison & Coombes)
// Models circumsolar brightening as angular distance from sun
//
// Parameters:
//   ray_dir: Ray direction (normalized)
//   peak_dir: Sun direction (normalized)
//   K: Power-law exponent (controls circumsolar concentration)
//   norm: Normalization factor (pre-computed to ensure hemisphere integrates to π)
//
// Returns: Angular distribution weight
float sky_power_law(vec3 ray_dir, vec3 peak_dir, float K, float norm) {
    // Angular distance from sun (radians)
    float psi = acos_safe(dot(peak_dir, ray_dir));

    // Avoid singularity near psi = 0 by clamping to 1 degree
    psi = max(psi, DEG_TO_RAD);

    // Power-law: psi^(-K) × normalization
    return pow(psi, -K) * norm;
}

// Prague sky model (Perez et al.)
// Combines circumsolar brightening and horizon brightening
//
// Parameters:
//   ray_dir: Ray direction (normalized, z-up coordinate system)
//   peak_dir: Sun direction (normalized)
//   params: vec4(circ_strength, circ_width, horizon_brightness, normalization)
//     - circ_strength: Circumsolar region strength multiplier
//     - circ_width: Circumsolar region angular width (degrees)
//     - horizon_brightness: Horizon brightening factor (1.0 = none, >1.0 = brightening)
//     - normalization: Pre-computed normalization factor
//
// Returns: Angular distribution weight
float sky_prague(vec3 ray_dir, vec3 peak_dir, vec4 params) {
    // Unpack parameters
    float circ_strength = params.x;
    float circ_width = params.y;
    float horizon_brightness = params.z;
    float normalization = params.w;

    // Angular distance from sun (degrees)
    float gamma = acos_safe(dot(ray_dir, peak_dir)) * RAD_TO_DEG;

    // Zenith angle (z-component is cos(theta) in z-up coordinates)
    float cos_theta = max(0.0, ray_dir.z);

    // Circumsolar brightening: exponential decay from sun
    float circ_term = 1.0 + circ_strength * exp(-gamma / circ_width);

    // Horizon brightening: increases as cos_theta → 0 (horizon)
    float horizon_term = 1.0 + (horizon_brightness - 1.0) * (1.0 - cos_theta);

    // Combined pattern
    float pattern = circ_term * horizon_term;

    // Multiply by π to account for cosine-weighted sampling PDF (cos×sin/π)
    // This ensures correct Monte Carlo integration for Prague angular distribution
    return pattern * normalization * PI;
}

// Unified sky distribution evaluator
// Selects model based on parameter validity (priority: power-law > prague > isotropic)
//
// Parameters:
//   ray_dir: Ray direction (normalized)
//   peak_dir: Sun direction (normalized)
//   power_law_K: Power-law exponent (>0 enables power-law model)
//   power_law_norm: Power-law normalization
//   prague_params: Prague model parameters (w > 0 enables Prague model)
//
// Returns: Angular distribution weight for Monte Carlo integration
float evaluate_sky_distribution(vec3 ray_dir, vec3 peak_dir, float power_law_K, float power_law_norm, vec4 prague_params) {

    // Priority 1: Power-law (if K > 0)
    if (power_law_K > 0.0) {
        return sky_power_law(ray_dir, peak_dir, power_law_K, power_law_norm);
    }

    // Priority 2: Prague (if params.w > 0, indicating valid normalization)
    if (prague_params.w > 0.0) {
        return sky_prague(ray_dir, peak_dir, prague_params);
    }

    // Priority 3: Isotropic (fallback)
    return sky_isotropic();
}

#endif // SKY_DISTRIBUTION_GLSL
