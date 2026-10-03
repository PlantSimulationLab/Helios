/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

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

// Diffuse, camera and pixel-label rays continue past a translucent cover from this distance (m) beyond it.
// Must match COVER_EXIT_OFFSET in OptiX8DeviceCode.cu and RayTracing.cuh.
#define COVER_EXIT_OFFSET 1e-4
// Maximum number of segments a camera or pixel-label ray is traced in: each translucent cover passed and each
// periodic-boundary wrap starts a new segment. Must match MAX_RAY_SEGMENTS in the OptiX backends.
#define MAX_RAY_SEGMENTS 32
// Maximum number of periodic-boundary wraps of a camera or pixel-label ray.
#define MAX_PERIODIC_WRAPS 10

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

// The helpers below read the translucent-cover material buffers, which the calling shader must declare before including
// this file: band_map_buf.map[] (set 1, binding 10), is_glass_buf.is_glass[] (11), glass_n_buf.glass_n[] (12) and
// glass_KL_buf.glass_KL[] (13). They index the source-0 slice [prim * material_band_count + global_band], as glass
// properties do not depend on the source.

// Whether a primitive is a translucent cover for the rays of this launch. A single ray carries every launched band, so a
// primitive that is glass in some launched bands but opaque in others cannot be both transmitted and blocked; it is a
// cover only if it is glass in ALL launched bands, and is otherwise a normal opaque primitive for the whole ray. This
// matches the OptiX 8 and OptiX 6 backends.
bool glass_cover_all_bands(uint prim, uint band_count, uint material_band_count) {
    if (band_count == 0u) {
        return false;
    }
    for (uint band = 0u; band < band_count; ++band) {
        if (is_glass_buf.is_glass[prim * material_band_count + band_map_buf.map[band]] == 0u) {
            return false;
        }
    }
    return true;
}

// Angular vec3(tau, rho, alpha) of a cover primitive in launched band `band`, for a ray at |cos(theta)| to its normal.
vec3 glass_cover_tau_rho_alpha(uint prim, uint band, uint material_band_count, float cos_theta) {
    uint gmi = prim * material_band_count + band_map_buf.map[band];
    return glass_tau_rho_alpha(cos_theta, glass_n_buf.glass_n[gmi], glass_KL_buf.glass_KL[gmi]);
}

#endif // GLASS_COVER_GLSL
