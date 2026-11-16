/** \file "EnergyBalanceSolver.h" Surface energy balance residual and per-surface solve, shared by the CPU and CUDA solvers.

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

#ifndef ENERGYBALANCESOLVER_H
#define ENERGYBALANCESOLVER_H

#include "EnergyBalanceModel.h"

//! Total conductance to water vapor of a surface, mol m⁻² s⁻¹
/**
 * The stomata of each face and that face's boundary layer act in series, and the two faces act in parallel. A face with no
 * stomata or no boundary-layer conductance has no vapor pathway, which also avoids a 0/0 in the series expression.
 * \param[in] gbv_front Boundary-layer conductance to water vapor of the front face
 * \param[in] gbv_back Boundary-layer conductance to water vapor of the back face
 * \param[in] gS Stomatal conductance of the surface
 * \param[in] stomatal_sidedness Fraction of the stomatal conductance on the front face
 */
ENERGYBALANCE_HOST_DEVICE inline float moistureConductance(float gbv_front, float gbv_back, float gS, float stomatal_sidedness) {
    float gM = 0.f;
    const float gS_front = gS * stomatal_sidedness;
    const float gS_back = gS * (1.f - stomatal_sidedness);
    if (gbv_front != 0.f && gS_front != 0.f) {
        gM += gbv_front * gS_front / (gbv_front + gS_front);
    }
    if (gbv_back != 0.f && gS_back != 0.f) {
        gM += gbv_back * gS_back / (gbv_back + gS_back);
    }
    return gM;
}

//! Inputs of the energy balance of one surface
struct SurfaceEnergyBalanceTerms {
    float R; //!< Absorbed radiation, W m⁻²
    float Qother; //!< Other heat flux leaving the surface, W m⁻²
    float eps; //!< Emissivity
    float Ta; //!< Air temperature, K
    float ea; //!< Air vapor pressure, Pa
    float pressure; //!< Atmospheric pressure, Pa
    float gH; //!< Conductance to heat, mol m⁻² s⁻¹
    float gbv_front; //!< Boundary-layer conductance to water vapor of the front face, mol m⁻² s⁻¹
    float gbv_back; //!< Boundary-layer conductance to water vapor of the back face, mol m⁻² s⁻¹
    float gS; //!< Stomatal conductance, mol m⁻² s⁻¹
    unsigned int Nsides; //!< Number of emitting faces (1 or 2)
    float stomatal_sidedness; //!< Fraction of the stomatal conductance on the front face
    float heatcapacity; //!< Heat capacity per unit area, J m⁻² K⁻¹
    float surfacehumidity; //!< Relative humidity at the evaporating surface
    float dt; //!< Time step for heat storage, s (0 for steady state)
    float Tprev; //!< Surface temperature at the previous time step, K
};

//! Energy balance residual of a surface at temperature \p T, W m⁻² (absorbed minus emitted, sensible, latent, other and stored)
ENERGYBALANCE_HOST_DEVICE inline float energyBalanceResidual(float T, const SurfaceEnergyBalanceTerms &terms) {

    // Outgoing emission flux
    const float Rout = float(terms.Nsides) * terms.eps * 5.67e-8F * T * T * T * T;

    // Sensible heat flux (see Campbell and Norman Eq. 6.8)
    const float QH = cp_air_mol * terms.gH * (T - terms.Ta);

    // Latent heat flux
    const float es = esat_Pa(T);
    const float gM = moistureConductance(terms.gbv_front, terms.gbv_back, terms.gS, terms.stomatal_sidedness);
    const float QL = gM * lambda_mol * (es * terms.surfacehumidity - terms.ea) / terms.pressure;

    // Storage heat flux
    float storage = 0.f;
    if (terms.dt > 0) {
        storage = terms.heatcapacity * (T - terms.Tprev) / terms.dt;
    }

    return terms.R - Rout - QH - QL - terms.Qother - storage;
}

//! Solve the energy balance of one surface for its temperature by the secant method
/**
 * \param[in] terms Inputs of the surface's energy balance
 * \param[in] T_initial First guess of the surface temperature, K
 * \param[out] T Surface temperature, K: the last iterate if the solve did not converge
 * \return True if the relative change of the last secant step fell below the tolerance within the iteration limit
 */
ENERGYBALANCE_HOST_DEVICE inline bool solveSurfaceEnergyBalance(const SurfaceEnergyBalanceTerms &terms, float T_initial, float &T) {
    const float err_max = 0.0001f;
    const unsigned int max_iter = 100;

    // Initial guesses. The second guess must differ from the first, otherwise the two residuals are identical and the secant
    // update below divides by zero (this happens when the supplied initial temperature equals the second guess of 400 K).
    float T_old = T_initial;
    float T_old_old = 400.f;
    if (T_old_old == T_old) {
        T_old_old = T_old + 100.f;
    }

    float resid_old = energyBalanceResidual(T_old, terms);
    float resid_old_old = energyBalanceResidual(T_old_old, terms);

    // Initialize the result to the current best guess so that a degenerate early exit (identical residuals) never leaves the
    // surface temperature uninitialized.
    T = T_old;
    float err = 100.f;
    unsigned int iter = 0;
    while (err > err_max && iter < max_iter) {

        if (resid_old == resid_old_old) { // secant denominator would be zero
            err = 0;
            break;
        }

        T = fabsf((T_old_old * resid_old - T_old * resid_old_old) / (resid_old - resid_old_old));

        const float resid = energyBalanceResidual(T, terms);

        // Relative change of the newest secant step (measures the latest iterate, not the previous one)
        err = fabsf(T - T_old) / fabsf(T_old);

        resid_old_old = resid_old;
        resid_old = resid;

        T_old_old = T_old;
        T_old = T;

        iter++;
    }

    return !(err > err_max);
}

#endif
