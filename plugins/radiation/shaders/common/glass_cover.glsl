// Translucent cover (glass/plastic) optical model — GLSL port of glass_tau_rho_alpha in
// OptiX8DeviceCode.cu. Angular transmittance/reflectance/absorptance of a single dielectric sheet via
// Fresnel reflection (two interfaces, both polarizations, internal reflections summed in closed form)
// plus Bouguer absorption (Duffie & Beckman solar-glazing model).
//
// Inputs:  cos_theta = |cos| of incidence angle, n = refractive index (>1), KL = absorption product K*L.
// Returns: vec3(tau, rho, alpha) with tau + rho + alpha == 1.
#ifndef GLASS_COVER_GLSL
#define GLASS_COVER_GLSL

// Maximum number of radiation bands supported for the per-band cover-transmittance accumulator.
// Must match HELIOS_MAX_RADIATION_BANDS in OptiX8LaunchParams.h; the host fails fast if exceeded.
#define GLASS_MAX_BANDS 32

vec3 glass_tau_rho_alpha(float cos_theta, float n, float KL) {
    cos_theta = max(1e-4, min(1.0, cos_theta)); // guard grazing/degenerate
    float theta   = acos(clamp(cos_theta, -1.0, 1.0));
    float sin_t   = sin(theta);
    float sin_tr  = sin_t / n;                    // Snell
    float cos_tr  = sqrt(max(0.0, 1.0 - sin_tr * sin_tr));
    float theta_r = asin(clamp(sin_tr, -1.0, 1.0));

    float r_par, r_per;
    if (theta < 1e-3) {
        float r0 = ((n - 1.0) / (n + 1.0)) * ((n - 1.0) / (n + 1.0));
        r_par = r0;
        r_per = r0;
    } else {
        float s_minus = sin(theta_r - theta);
        float s_plus  = sin(theta_r + theta);
        float t_minus = tan(theta_r - theta);
        float t_plus  = tan(theta_r + theta);
        r_per = (s_minus * s_minus) / max(1e-12, s_plus * s_plus);
        r_par = (t_minus * t_minus) / max(1e-12, t_plus * t_plus);
    }

    float tau_a = (KL > 0.0) ? exp(-KL / max(1e-4, cos_tr)) : 1.0;

    float tau = 0.0;
    float rho = 0.0;
    for (int pol = 0; pol < 2; pol++) {
        float r     = (pol == 0) ? r_per : r_par;
        float denom = max(1e-6, 1.0 - (r * tau_a) * (r * tau_a));
        float tau_i = tau_a * (1.0 - r) * (1.0 - r) / denom;
        float rho_i = r * (1.0 + tau_a * tau_i);
        tau += 0.5 * tau_i;
        rho += 0.5 * rho_i;
    }
    tau = clamp(tau, 0.0, 1.0);
    rho = clamp(rho, 0.0, 1.0);
    float alpha = max(0.0, 1.0 - tau - rho);
    return vec3(tau, rho, alpha);
}

#endif // GLASS_COVER_GLSL
