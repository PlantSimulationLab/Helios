/** \file "EnergyBalanceModel.cpp" Energy balance model plugin declarations.

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

#include "EnergyBalanceModel.h"
#include "../include/EnergyBalanceModel.h"

#include "global.h"

#ifdef HELIOS_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace helios;

EnergyBalanceModel::EnergyBalanceModel(helios::Context *__context) {

    // All default values set here

    temperature_default = 300; // Kelvin

    wind_speed_default = 1.f; // m/s

    air_temperature_default = 300; // Kelvin

    air_humidity_default = 0.5; // %

    pressure_default = 101000; // Pa

    gS_default = 0.; // mol/m^2-s

    Qother_default = 0; // W/m^2

    heatcapacity_default = 0; // J/m^2-oC

    surface_humidity_default = 1; //(unitless)

    air_energy_balance_enabled = false;

    message_flag = true; // print messages to screen by default

    // Copy pointer to the context
    context = __context;

#ifdef HELIOS_CUDA_AVAILABLE
    // Initialize GPU acceleration flags
    gpu_acceleration_enabled = true; // Try to use GPU by default

    // Perform runtime GPU detection
    initializeGPUAcceleration();
#endif
}

void EnergyBalanceModel::run() {
    run(context->getAllUUIDs());
}

void EnergyBalanceModel::run(float dt) {
    run(context->getAllUUIDs(), dt);
}

void EnergyBalanceModel::run(const std::vector<uint> &UUIDs) {
    run(UUIDs, 0.f);
}


void EnergyBalanceModel::run(const std::vector<uint> &UUIDs, float dt) {

    if (message_flag) {
        std::cout << "Running energy balance model..." << std::flush;
    }

    // Check that some primitives exist in the context
    if (UUIDs.empty()) {
        std::cerr << "WARNING (EnergyBalanceModel::run): No primitives have been added to the context.  There is nothing to simulate. Exiting..." << std::endl;
        return;
    }

    // Results are written to the Context in parallel, so a primitive listed twice would be written by two threads at once. Every UUID is first checked to exist, which bounds the largest UUID and therefore the size of the table used to
    // find duplicates.
    uint max_UUID = 0;
    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::run): Primitive UUID " + std::to_string(UUID) + " does not exist in the Context.");
        }
        max_UUID = std::max(max_UUID, UUID);
    }
    std::vector<bool> UUID_listed(size_t(max_UUID) + 1, false);
    for (uint UUID: UUIDs) {
        if (UUID_listed[UUID]) {
            helios_runtime_error("ERROR (EnergyBalanceModel::run): Primitive UUID " + std::to_string(UUID) + " appears more than once in the UUIDs passed to run(). Each primitive must be listed only once.");
        }
        UUID_listed[UUID] = true;
    }

    if (canopy_airspace_enabled) {
        if (dt > 0.f) {
            helios_runtime_error("ERROR (EnergyBalanceModel::run): The canopy airspace model solves for a steady state and cannot be used with the dynamic form of run() that takes a timestep. Either call run() without a timestep argument, or "
                                 "disable the canopy airspace model with disableCanopyAirspaceModel().");
        }
        evaluateCanopyAirspace(UUIDs);
    } else {
        evaluateSurfaceEnergyBalance(UUIDs, dt);
    }

    if (message_flag) {
        std::cout << "done." << std::endl;
    }
}

#ifdef HELIOS_CUDA_AVAILABLE
void EnergyBalanceModel::initializeGPUAcceleration() {
    // Check if CUDA is actually available at runtime
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        // CUDA not available - disable GPU acceleration and fall back to CPU
        if (message_flag) {
            std::cout << "INFO (EnergyBalanceModel): CUDA runtime unavailable (" << cudaGetErrorString(err) << "). Using OpenMP CPU implementation." << std::endl;
        }
        gpu_acceleration_enabled = false;
        return;
    }

    // GPU available
    if (message_flag) {
        std::cout << "INFO (EnergyBalanceModel): GPU acceleration enabled (" << deviceCount << " device(s) found)." << std::endl;
    }
    gpu_acceleration_enabled = true;
}
#endif

void EnergyBalanceModel::evaluateSurfaceEnergyBalance(const std::vector<uint> &UUIDs, float dt) {
    SurfaceEnergyBalanceInputs inputs;
    gatherSurfaceEnergyBalanceInputs(UUIDs, inputs);

    std::vector<float> T;
    solveSurfaceTemperatures(inputs, dt, T);

    writeSurfaceEnergyBalanceOutputs(UUIDs, inputs, T, dt);
}

void EnergyBalanceModel::solveSurfaceTemperatures(const SurfaceEnergyBalanceInputs &inputs, float dt, std::vector<float> &T) {
#ifdef HELIOS_CUDA_AVAILABLE
    if (gpu_acceleration_enabled) {
        solveSurfaceTemperatures_GPU(inputs, dt, T);
    } else {
        solveSurfaceTemperatures_CPU(inputs, dt, T);
    }
#else
    // CPU-only build: always use OpenMP implementation
    solveSurfaceTemperatures_CPU(inputs, dt, T);
#endif

    for (float &temperature: T) {
        if (!std::isfinite(temperature)) { // catches NaN and infinities
            temperature = temperature_default;
        }
    }
}

#ifdef HELIOS_CUDA_AVAILABLE
void EnergyBalanceModel::enableGPUAcceleration() {
    // Try to enable GPU, but respect hardware availability
    initializeGPUAcceleration();
}

void EnergyBalanceModel::disableGPUAcceleration() {
    gpu_acceleration_enabled = false;
}

bool EnergyBalanceModel::isGPUAccelerationEnabled() const {
    return gpu_acceleration_enabled;
}
#endif

void EnergyBalanceModel::evaluateAirEnergyBalance(float dt_sec, float time_advance_sec) {
    evaluateAirEnergyBalance(context->getAllUUIDs(), dt_sec, time_advance_sec);
}

void EnergyBalanceModel::evaluateAirEnergyBalance(const std::vector<uint> &UUIDs, float dt_sec, float time_advance_sec) {

    if (dt_sec <= 0) {
        helios_runtime_error("ERROR (EnergyBalanceModel::evaluateAirEnergyBalance): dt_sec must be greater than zero to run the air energy balance.");
    }

    // Create warning aggregator
    helios::WarningAggregator warnings;
    warnings.setEnabled(message_flag);

    float air_temperature_reference = air_temperature_default; // Default air temperature in Kelvin
    if (context->doesGlobalDataExist("air_temperature_reference") && context->getGlobalDataType("air_temperature_reference") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_temperature_reference", air_temperature_reference);
    }

    float air_humidity_reference = air_humidity_default; // Default air relative humidity
    if (context->doesGlobalDataExist("air_humidity_reference") && context->getGlobalDataType("air_humidity_reference") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_humidity_reference", air_humidity_reference);
    }

    float wind_speed_reference = wind_speed_default; // Default wind speed in m/s
    if (context->doesGlobalDataExist("wind_speed_reference") && context->getGlobalDataType("wind_speed_reference") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("wind_speed_reference", wind_speed_reference);
    }

    // Variables to be set externally via global data

    float Patm;
    if (context->doesGlobalDataExist("air_pressure") && context->getGlobalDataType("air_pressure") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_pressure", Patm);
    } else {
        Patm = pressure_default;
    }

    float air_temperature_average;
    if (context->doesGlobalDataExist("air_temperature_average") && context->getGlobalDataType("air_temperature_average") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_temperature_average", air_temperature_average);
    } else {
        air_temperature_average = air_temperature_reference;
    }

    float air_moisture_average;
    if (context->doesGlobalDataExist("air_moisture_average") && context->getGlobalDataType("air_moisture_average") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_moisture_average", air_moisture_average);
    } else {
        air_moisture_average = air_humidity_reference * esat_Pa(air_temperature_reference) / Patm;
    }

    float air_temperature_ABL = 0; // initialize to 0 as a flag so that if it's not available from global data, we'll initialize it below
    if (context->doesGlobalDataExist("air_temperature_ABL") && context->getGlobalDataType("air_temperature_ABL") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_temperature_ABL", air_temperature_ABL);
    }

    float air_moisture_ABL;
    if (context->doesGlobalDataExist("air_moisture_ABL") && context->getGlobalDataType("air_moisture_ABL") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_moisture_ABL", air_moisture_ABL);
    } else {
        air_moisture_ABL = air_humidity_reference * esat_Pa(air_temperature_reference) / Patm;
    }

    // Get dimensions of canopy volume
    if (canopy_dimensions == make_vec3(0, 0, 0)) {
        vec2 xbounds, ybounds, zbounds;
        context->getDomainBoundingBox(UUIDs, xbounds, ybounds, zbounds);
        canopy_dimensions = make_vec3(xbounds.y - xbounds.x, ybounds.y - ybounds.x, zbounds.y - zbounds.x);
    }
    assert(canopy_dimensions.x > 0 && canopy_dimensions.y > 0 && canopy_dimensions.z > 0);
    if (canopy_height_m == 0) {
        canopy_height_m = canopy_dimensions.z; // Set the canopy height if not already set
    }
    if (reference_height_m == 0) {
        reference_height_m = canopy_height_m; // Set the reference height if not already set
    }

    std::cout << "Read in initial values. Ta = " << air_temperature_average << "; e = " << air_moisture_average << "; " << air_moisture_average / esat_Pa(air_temperature_reference) * Patm << std::endl;

    std::cout << "Canopy dimensions: " << canopy_dimensions.x << " x " << canopy_dimensions.y << " x " << canopy_height_m << std::endl;

    float displacement_height = 0.67f * canopy_height_m;
    float zo_m = 0.1f * canopy_height_m; // Roughness length for momentum transfer (m)
    float zo_h = 0.01f * canopy_height_m; // Roughness length for heat transfer (m)

    float abl_height_m = 1000.f;

    float time = 0;
    float air_temperature_old = 0;
    float air_moisture_old = 0;
    float air_temperature_ABL_old = 0;
    float air_moisture_ABL_old = 0;
    while (time < time_advance_sec) {
        float dt_actual = dt_sec;
        if (time + dt_actual > time_advance_sec) {
            dt_actual = time_advance_sec - time; // Adjust the time step to not exceed the total time
        }

        std::cout << "Air moisture average: " << air_moisture_average << std::endl;

        // Update the surface energy balance
        this->run(UUIDs);

        // Calculate temperature source term
        float sensible_source_flux_W_m3;
        context->calculatePrimitiveDataAreaWeightedSum(UUIDs, "sensible_flux", sensible_source_flux_W_m3); // units: Watts
        sensible_source_flux_W_m3 /= canopy_dimensions.x * canopy_dimensions.y * canopy_height_m; // Convert to W/m^3

        // Calculate moisture source term
        float moisture_source_flux_W_m3;
        context->calculatePrimitiveDataAreaWeightedSum(UUIDs, "latent_flux", moisture_source_flux_W_m3); // units: Watts
        moisture_source_flux_W_m3 /= canopy_dimensions.x * canopy_dimensions.y * canopy_height_m; // Convert to W/m^3
        float moisture_source_flux_mol_s_m3 = moisture_source_flux_W_m3 / lambda_mol; // Convert to mol/s/m^3

        float rho_air_mol_m3 = Patm / (R * air_temperature_average);
        ; // Molar density of air (mol/m^3).

        // --- canopy / ABL interface & implicit‐Euler update ---

        // 1) canopy‐scale source fluxes per unit ground area
        float H_s = sensible_source_flux_W_m3 * canopy_height_m; // W m⁻²
        float E_s = moisture_source_flux_mol_s_m3 * canopy_height_m; // mol m⁻² s⁻¹

        // 2) neutral friction velocity for Obukhov length
        float u_star_neutral = von_Karman_constant * wind_speed_reference / std::log((reference_height_m - displacement_height) / zo_m);

        // 3) Obukhov length L [m]
        float rho_air_kg_m3 = rho_air_mol_m3 * 0.02896f;
        float cp_air_kg = cp_air_mol / 0.02896f;
        float L = 1e6f;
        if (std::fabs(H_s) > 1e-6f) {
            L = -rho_air_kg_m3 * cp_air_kg * std::pow(u_star_neutral, 3) * air_temperature_reference / (von_Karman_constant * 9.81f * H_s);
        }

        // 4) stability functions (Businger–Dyer)
        auto psi_unstable_u = [&](float zeta) {
            float x = std::pow(1.f - 16.f * zeta, 0.25f);
            return 2.f * std::log((1.f + x) / 2.f) + std::log((1.f + x * x) / 2.f) - 2.f * std::atan(x) + 0.5f * M_PI;
        };
        auto psi_unstable_h = [&](float zeta) { return 2.f * std::log((1.f + std::sqrt(1.f - 16.f * zeta)) / 2.f); };
        auto psi_stable = [&](float zeta) { return -5.f * zeta; };

        float zeta_r = (reference_height_m - displacement_height) / L;
        float psi_m_r = (zeta_r < 0.f) ? psi_unstable_u(zeta_r) : psi_stable(zeta_r);
        float psi_h_r = (zeta_r < 0.f) ? psi_unstable_h(zeta_r) : psi_stable(zeta_r);

        // 5) stability‐corrected friction velocity
        float u_star = von_Karman_constant * wind_speed_reference / (std::log((reference_height_m - displacement_height) / zo_m) - psi_m_r);

        // 6) Monin–Obukhov scales θ_* and q_*
        float theta_star = H_s / (rho_air_mol_m3 * cp_air_mol * u_star);
        float q_star = E_s / (rho_air_mol_m3 * u_star);

        // 7) corrected log‐law lengths
        float ln_m_corr = std::log((reference_height_m - displacement_height) / zo_m) - psi_m_r;
        float ln_h_corr = std::log((reference_height_m - displacement_height) / zo_h) - psi_h_r;

        // 8) aerodynamic conductance ga [mol m⁻² s⁻¹]
        float ga = std::pow(von_Karman_constant, 2) * wind_speed_reference / (ln_m_corr * ln_h_corr) * rho_air_mol_m3;

        // 9) interface values at canopy top (Monin–Obukhov log-law)
        float air_moisture_reference = esat_Pa(air_temperature_reference) / Patm * air_humidity_reference;
        float T_int = air_temperature_reference + theta_star / von_Karman_constant * ln_h_corr;
        float x_int = air_moisture_reference + q_star / von_Karman_constant * ln_h_corr;

        // Cap relative humidity to 100%
        float x_sat = esat_Pa(T_int) / Patm;
        if (x_int > x_sat) {
            // std::cerr << "WARNING (EnergyBalanceModel::evaluateAirEnergyBalance): Air moisture exceeds saturation. Capping to saturation value." << std::endl;
            x_int = 0.99f * x_sat;
        }

        // 10) entrainment conductance g_e (mol m⁻² s⁻¹)
        float H_canopy = sensible_source_flux_W_m3 * canopy_height_m;
        float g_e;
        if (air_temperature_ABL == 0.f) {
            // first‐step equilibrium initialization
            float w_star0 = H_canopy > 0.f ? std::pow(9.81f * H_canopy * canopy_height_m / (rho_air_mol_m3 * cp_air_mol * air_temperature_reference), 1.f / 3.f) : 0.f;
            const float A_e = 0.02f;
            float w_e0 = A_e * w_star0;
            g_e = rho_air_mol_m3 * w_e0;
            // initialize ABL state
            air_temperature_ABL = (ga * air_temperature_average + g_e * air_temperature_reference) / (ga + g_e);
            air_moisture_ABL = (ga * air_moisture_average + g_e * air_moisture_reference) / (ga + g_e);
        } else {
            // ongoing entrainment
            float w_star = 0.f;
            if (H_canopy > 0.f) {
                w_star = std::pow(9.81f * H_canopy * canopy_height_m / (rho_air_mol_m3 * cp_air_mol * air_temperature_ABL), 1.f / 3.f);
            }
            float u_star_shear = von_Karman_constant * wind_speed_reference / std::log((reference_height_m - displacement_height) / zo_m);
            float w_e = std::fmax(0.01f * w_star, 0.005f * u_star_shear);
            g_e = rho_air_mol_m3 * w_e;
        }

        // if ( g_e>0.05 ) {
        //     std::cerr << "Capping g_e value of " << g_e << std::endl;
        //     g_e = 0.05f;
        // }
        g_e = 2.0;

        // 11) canopy→ABL fluxes
        float sensible_upper_flux_W_m2 = cp_air_mol * ga * (air_temperature_average - T_int);

        // 12) update canopy‐air temperature (implicit Euler)
        // numerator: T^n + (Δt / (ρ c_p h_c)) * [H_s + c_p g_a T_int]
        // denominator: 1 + Δt·g_a/(ρ h_c)
        float numerator_temp = air_temperature_average + dt_actual / (rho_air_mol_m3 * cp_air_mol * canopy_height_m) * (H_s + cp_air_mol * ga * T_int);
        float denominator_temp = 1.f + dt_actual * ga / (rho_air_mol_m3 * canopy_height_m);
        air_temperature_average = numerator_temp / denominator_temp;

        // 13) update canopy‐air moisture (implicit Euler)

        // --- canopy implicit‐Euler moisture update ---
        // update: ρh (x_new - x_old)/dt = E_s - E_top
        float E_top = ga * (air_moisture_average - x_int);
        float f = std::pow(1.f - (1.f - 0.622f) * air_moisture_average, 2) / 0.622f;
        float numer_can_x = air_moisture_average + dt_actual * f / (rho_air_mol_m3 * canopy_height_m) * (E_s + ga * x_int);
        float denom_can_x = 1.f + dt_actual * f * ga / (rho_air_mol_m3 * canopy_height_m);
        air_moisture_average = numer_can_x / denom_can_x;

        // 14) update ABL‐air temperature (implicit Euler)
        float denom_abl_T = 1.f + dt_actual * (ga + g_e) / (rho_air_mol_m3 * cp_air_mol * abl_height_m);
        float num_abl_T = air_temperature_ABL + dt_actual / (rho_air_mol_m3 * cp_air_mol * abl_height_m) * (sensible_upper_flux_W_m2 + cp_air_mol * g_e * (air_temperature_reference - air_temperature_ABL));
        air_temperature_ABL = num_abl_T / denom_abl_T;

        // 15) update ABL‐air moisture (implicit Euler)
        // source into ABL storage
        //  ρ_ab·h_ab · (x_ab_new - x_ab_old)/dt = E_top - E_e
        float E_e = g_e * (air_moisture_ABL - air_moisture_reference);
        float numer_abl_x = air_moisture_ABL + dt_actual / (rho_air_mol_m3 * abl_height_m) * (E_top - E_e);
        float denom_abl_x = 1.f + dt_actual * (ga + g_e) / (rho_air_mol_m3 * abl_height_m);
        air_moisture_ABL = numer_abl_x / denom_abl_x;

        // Cap relative humidity to 100%
        float esat = esat_Pa(air_temperature_average) / Patm;
        if (air_moisture_average > esat) {
            warnings.addWarning("air_moisture_exceeds_saturation", "Air moisture exceeds saturation. Capping to saturation value.");
            air_moisture_average = 0.99f * esat;
        }
        esat = esat_Pa(air_temperature_ABL) / Patm;
        if (air_moisture_ABL > esat) {
            air_moisture_ABL = 0.99f * esat;
        }

        // Check that timestep is not too large based on aerodynamic resistance
        if (dt_actual > 0.5f * canopy_height_m * rho_air_mol_m3 / ga) {
            warnings.addWarning("timestep_too_large", "Time step is too large. The air energy balance may not converge properly.");
        }

        float heat_from_Htop = dt_actual / (rho_air_mol_m3 * cp_air_mol * abl_height_m) * sensible_upper_flux_W_m2;
        float heat_from_entrainment = dt_actual * g_e * (air_temperature_reference - air_temperature_ABL) / (rho_air_mol_m3 * abl_height_m);

        std::cerr << "ABL ENERGY UPDATE:\n"
                  << "  T_ABL_n          = " << air_temperature_ABL << "\n"
                  << "  T_ref            = " << air_temperature_reference << "\n"
                  << "  g_e              = " << g_e << " mol/m²/s\n"
                  << "  Heat from H_top  = " << heat_from_Htop << " K\n"
                  << "  Heat from H_e    = " << heat_from_entrainment << " K\n"
                  << "  Numerator        = " << num_abl_T << " K\n"
                  << "  Denominator      = " << denom_abl_T << "\n"
                  << "  T_ABL_new        = " << (num_abl_T / denom_abl_T) << "\n";


#ifdef HELIOS_DEBUG

        // -- check energy consistency -- //
        H_s = sensible_source_flux_W_m3 * canopy_height_m; // W m⁻², canopy source
        float H_top = cp_air_mol * ga * (air_temperature_average - T_int); // W m⁻², canopy→ABL
        float H_e = cp_air_mol * g_e * (air_temperature_ABL - air_temperature_reference); // W m⁻², ABL→free atm

        E_s = moisture_source_flux_mol_s_m3 * canopy_height_m; // mol m⁻² s⁻¹, canopy source
        E_top = ga * (air_moisture_average - x_int); // mol m⁻² s⁻¹, canopy→ABL
        E_e = g_e * (air_moisture_ABL - air_moisture_reference); // mol m⁻² s⁻¹, ABL→free atm

        // Print a concise table of key values
        std::cout << std::fixed << std::setprecision(5) << "step=" << time << "  T_c=" << air_temperature_average << "  T_int=" << T_int << "  T_b=" << air_temperature_ABL << "  H_s=" << H_s << "  H_top=" << H_top << "  H_e=" << H_e
                  << "  E_s=" << E_s << "  E_top=" << E_top << "  E_e=" << E_e << "  psi_m=" << psi_m_r << "  psi_h=" << psi_h_r << "  u*=" << u_star << "  L=" << L << std::endl;

        // temperature balance

        const float tol = 1e-5f;
        float S_can = rho_air_mol_m3 * cp_air_mol * canopy_height_m * (air_temperature_average - air_temperature_old) / dt_actual;
        if (fabs(H_s - H_top - S_can) > tol)
            std::cerr << "Canopy budget error: " << (H_s - H_top - S_can) << "\n";

        float S_abl = rho_air_mol_m3 * cp_air_mol * abl_height_m * (air_temperature_ABL - air_temperature_ABL_old) / dt_actual;
        if (fabs(H_top - H_e - S_abl) > tol)
            std::cerr << "ABL budget error: " << (H_top - H_e - S_abl) << "\n";

        // moisture balance per unit ground area

        S_can = rho_air_mol_m3 * canopy_height_m * (air_moisture_average - air_moisture_old) / dt_actual;

        const float tol_m = 1e-6f;
        if (std::fabs(E_s - E_top - S_can) > tol_m) {
            std::cerr << "CANOPY MOISTURE BUDGET ERROR: " << "E_s - E_up - S_can = " << E_s - E_top - S_can << " mol/m2/s\n";
        }

        S_abl = rho_air_mol_m3 * abl_height_m * (air_moisture_ABL - air_moisture_ABL_old) / dt_actual;

        if (std::fabs(E_top - E_e - S_abl) > tol_m) {
            std::cerr << "ABL MOISTURE BUDGET ERROR: " << "E_up - E_e - S_abl = " << E_top - E_e - S_abl << " mol/m2/s\n";
        }


#endif

        // Set primitive data
        context->setPrimitiveData(UUIDs, "air_temperature", air_temperature_average);
        float air_humidity = air_moisture_average * Patm / esat_Pa(air_temperature_average);
        context->setPrimitiveData(UUIDs, "air_humidity", air_humidity);

        // Set global data
        context->setGlobalData("air_temperature_average", air_temperature_average);
        context->setGlobalData("air_humidity_average", air_humidity);
        context->setGlobalData("air_moisture_average", air_moisture_average);

        context->setGlobalData("air_temperature_ABL", air_temperature_ABL);
        float air_humidity_ABL = air_moisture_ABL * Patm / esat_Pa(air_temperature_ABL);
        context->setGlobalData("air_humidity_ABL", air_humidity_ABL);

        context->setGlobalData("air_temperature_interface", T_int);
        float air_humidity_interface = x_int * Patm / esat_Pa(T_int);
        context->setGlobalData("air_humidity_interface", air_humidity_interface);

        context->setGlobalData("aerodynamic_resistance", rho_air_mol_m3 / ga);

        std::cout << "Computed air temperature: " << air_temperature_average << " " << air_temperature_old << std::endl;
        std::cout << "Computed air humidity: " << air_humidity << std::endl;
        std::cout << "Computed air moisture: " << air_moisture_average << " " << air_moisture_old << std::endl;

        if (time > 0 && std::abs(air_temperature_average - air_temperature_old) / air_temperature_old < 0.003f && std::abs(air_moisture_average - air_moisture_old) / air_moisture_old < 0.003f) {
            // If the air temperature and humidity have not changed significantly, we can stop iterating
            std::cout << "Converged" << std::endl;
            break;
        }
        air_temperature_old = air_temperature_average;
        air_moisture_old = air_moisture_average;
        air_temperature_ABL_old = air_temperature_ABL;
        air_moisture_ABL_old = air_moisture_ABL;

        time += dt_actual;
    }

    // Report aggregated warnings
    warnings.report(std::cerr);
}

void EnergyBalanceModel::enableMessages() {
    message_flag = true;
}

void EnergyBalanceModel::disableMessages() {
    message_flag = false;
}

void EnergyBalanceModel::addRadiationBand(const char *band) {
    if (std::find(radiation_bands.begin(), radiation_bands.end(), band) == radiation_bands.end()) { // only add band if it doesn't exist
        radiation_bands.emplace_back(band);
    }
}

void EnergyBalanceModel::addRadiationBand(const std::vector<std::string> &bands) {
    for (auto &band: bands) {
        addRadiationBand(band.c_str());
    }
}

void EnergyBalanceModel::enableAirEnergyBalance() {
    enableAirEnergyBalance(0, 0);
}

void EnergyBalanceModel::enableAirEnergyBalance(float canopy_height_m, float reference_height_m) {

    if (canopy_height_m < 0) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableAirEnergyBalance): Canopy height must be greater than or equal to zero.");
    } else if (canopy_height_m != 0 && reference_height_m < canopy_height_m) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableAirEnergyBalance): Reference height must be greater than or equal to canopy height.");
    }

    if (canopy_airspace_enabled) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableAirEnergyBalance): The canopy airspace model is already enabled. These two models both determine the within-canopy air state and share the canopy and reference heights, so only one "
                             "may be active at a time.");
    }

    air_energy_balance_enabled = true;
    this->canopy_height_m = canopy_height_m;
    this->reference_height_m = reference_height_m;
}

//! Solve a tridiagonal linear system using the Thomas algorithm
/**
 * The canopy airspace layers form a chain in which each layer exchanges only with its immediate neighbors, so the resulting system is tridiagonal and can be solved directly in a single forward and backward sweep.
 * \param[in] sub Sub-diagonal coefficients. The first element is unused.
 * \param[in] diag Diagonal coefficients.
 * \param[in] super Super-diagonal coefficients. The last element is unused.
 * \param[in] rhs Right-hand side of the system.
 * \param[out] solution Solution vector, resized to match the system.
 */
static void solveTridiagonal(const std::vector<float> &sub, const std::vector<float> &diag, const std::vector<float> &super, const std::vector<float> &rhs, std::vector<float> &solution) {

    size_t N = diag.size();
    solution.resize(N);

    std::vector<float> super_modified(N);
    std::vector<float> rhs_modified(N);

    if (diag.at(0) == 0.f) {
        helios_runtime_error("ERROR (EnergyBalanceModel): Canopy airspace resistance network is singular. This indicates that a layer has no conductance to its neighbors or to the leaves it contains.");
    }

    // Forward sweep
    super_modified.at(0) = super.at(0) / diag.at(0);
    rhs_modified.at(0) = rhs.at(0) / diag.at(0);
    for (size_t j = 1; j < N; j++) {
        float denominator = diag.at(j) - sub.at(j) * super_modified.at(j - 1);
        if (denominator == 0.f) {
            helios_runtime_error("ERROR (EnergyBalanceModel): Canopy airspace resistance network is singular at layer " + std::to_string(j) + ". This indicates that a layer has no conductance to its neighbors or to the leaves it contains.");
        }
        super_modified.at(j) = super.at(j) / denominator;
        rhs_modified.at(j) = (rhs.at(j) - sub.at(j) * rhs_modified.at(j - 1)) / denominator;
    }

    // Backward substitution
    solution.at(N - 1) = rhs_modified.at(N - 1);
    for (size_t j = N - 1; j-- > 0;) {
        solution.at(j) = rhs_modified.at(j) - super_modified.at(j) * solution.at(j + 1);
    }
}

void EnergyBalanceModel::evaluateCanopyAirspace(const std::vector<uint> &UUIDs) {

    helios::WarningAggregator warnings;
    warnings.setEnabled(message_flag);

    // -- Above-canopy boundary condition -- //

    float air_temperature_reference = air_temperature_default;
    if (context->doesGlobalDataExist("air_temperature_reference") && context->getGlobalDataType("air_temperature_reference") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_temperature_reference", air_temperature_reference);
    }

    float air_humidity_reference = air_humidity_default;
    if (context->doesGlobalDataExist("air_humidity_reference") && context->getGlobalDataType("air_humidity_reference") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_humidity_reference", air_humidity_reference);
    }

    float wind_speed_reference = wind_speed_default;
    if (context->doesGlobalDataExist("wind_speed_reference") && context->getGlobalDataType("wind_speed_reference") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("wind_speed_reference", wind_speed_reference);
    }

    if (wind_speed_reference <= 0.f) {
        helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Reference wind speed must be greater than zero for the canopy airspace model, since aerodynamic resistance is inversely proportional to wind speed. Set global data "
                             "'wind_speed_reference' to a positive value.");
    }

    float Patm = pressure_default;
    if (context->doesGlobalDataExist("air_pressure") && context->getGlobalDataType("air_pressure") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_pressure", Patm);
    }

    // Layer assignment and the canopy base height are cached, so verify the geometry they describe is still present. A primitive deleted since enableCanopyAirspaceModel() would otherwise fail deep inside the reduction.
    for (uint UUID: canopy_airspace_UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Canopy primitive UUID " + std::to_string(UUID) +
                                 " no longer exists in the Context. If the geometry changed since the canopy airspace model was enabled, call enableCanopyAirspaceModel() again with the current primitive UUIDs.");
        }
    }
    for (uint UUID: canopy_airspace_ground_UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Ground primitive UUID " + std::to_string(UUID) +
                                 " no longer exists in the Context. If the geometry changed since the canopy airspace model was enabled, call enableCanopyAirspaceModel() again with the current primitive UUIDs.");
        }
    }

    // The canopy and the soil are both part of the airspace solution, so their surface energy balance must be solved alongside it. A primitive that the airspace drives but that run() never solves would keep a surface temperature that does not
    // correspond to the air it is exchanging with.
    std::unordered_set<uint> solved_UUIDs(UUIDs.begin(), UUIDs.end());
    for (uint UUID: canopy_airspace_UUIDs) {
        if (solved_UUIDs.find(UUID) == solved_UUIDs.end()) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Canopy primitive UUID " + std::to_string(UUID) +
                                 " was given to enableCanopyAirspaceModel() but is not included in the primitives passed to run(). Every primitive exchanging with the canopy airspace must also have its surface energy balance solved.");
        }
    }
    for (uint UUID: canopy_airspace_ground_UUIDs) {
        if (solved_UUIDs.find(UUID) == solved_UUIDs.end()) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Ground primitive UUID " + std::to_string(UUID) +
                                 " was given to enableCanopyAirspaceModel() but is not included in the primitives passed to run(). The soil exchanges heat with the lowest canopy layer, so its surface energy balance must be solved alongside it.");
        }
    }

    if (!canopy_airspace_layers_assigned) {
        assignCanopyAirspaceLayers();
    }

    size_t Ncanopy = canopy_airspace_UUIDs.size();
    uint Nlayers = canopy_airspace_layer_count;

    // -- Resistance network geometry -- //

    float rho_air_mol_m3 = Patm / (R * air_temperature_reference);

    float exp_half_LAI = std::exp(-0.5f * canopy_airspace_LAI);
    float zo_m = canopy_height_m * exp_half_LAI * (1.f - exp_half_LAI);
    float displacement_height = canopy_height_m * (1.f - 2.f / canopy_airspace_LAI * (1.f - exp_half_LAI));
    float attenuation_coefficient = 0.5f * canopy_airspace_LAI;

    // Turbulent conductances of the resistance network, which depend on atmospheric stability through the inverse Obukhov length. They begin at neutral stability and, if stability correction is enabled, are updated on each iteration
    // from the fluxes leaving the canopy.
    float inverse_obukhov_length = 0.f;
    float aerodynamic_resistance_s_m = 0.f;
    float friction_velocity = 0.f;
    float g_a = 0.f;
    // Conductance between layer j and layer j+1, evaluated at the shared layer boundary. Sized Nlayers so that the last entry, which has no layer above it, is unused.
    std::vector<float> g_layer(Nlayers, 0.f);

    auto updateTurbulentConductances = [&]() {
        // Aerodynamic resistance between the canopy top and the reference height, converted from a resistance per unit ground area to a molar conductance.
        aerodynamic_resistance_s_m = calculateAerodynamicResistance(wind_speed_reference, inverse_obukhov_length, friction_velocity);
        g_a = rho_air_mol_m3 / aerodynamic_resistance_s_m;

        // Conductance between adjacent canopy airspace layers, from the eddy diffusivity for heat divided by the distance between layer midpoints. Turbulent mixing is strongly damped with depth into the canopy, so the diffusivity decays
        // exponentially from its canopy-top value with the same attenuation coefficient used for the wind profile (Cionco 1972; the K/dz discretization follows standard multilayer canopy practice, e.g. Bonan et al. 2018). The
        // canopy-top value is the surface-layer diffusivity there, so that it is continuous with the aerodynamic resistance above the canopy.
        float eddy_diffusivity_canopy_top = von_Karman_constant * friction_velocity * (canopy_height_m - displacement_height) / calculateStabilityFunctionHeat((canopy_height_m - displacement_height) * inverse_obukhov_length);
        for (uint j = 0; j + 1 < Nlayers; j++) {
            float boundary_height = canopy_airspace_layer_top_m.at(j);
            float separation = canopy_airspace_layer_midpoint_m.at(j + 1) - canopy_airspace_layer_midpoint_m.at(j);
            if (separation <= 0.f) {
                helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Canopy airspace layers " + std::to_string(j) + " and " + std::to_string(j + 1) +
                                     " have coincident midpoints, so the conductance between them is undefined. This indicates that leaf area is concentrated at a single height and cannot be divided into the requested number of layers.");
            }
            float eddy_diffusivity = eddy_diffusivity_canopy_top * std::exp(-attenuation_coefficient * (1.f - boundary_height / canopy_height_m));
            g_layer.at(j) = rho_air_mol_m3 * eddy_diffusivity / separation;
        }
    };
    updateTurbulentConductances();

    // Wind speed at the canopy top, obtained by extrapolating the reference wind speed down to the canopy height with the neutral log law. The leaf boundary-layer conductances are fixed before the iteration begins, so the wind profile
    // that sets them is not updated as the stability evolves.
    float wind_speed_canopy_top = wind_speed_reference * std::log((canopy_height_m - displacement_height) / zo_m) / std::log((reference_height_m - displacement_height) / zo_m);
    wind_speed_canopy_top = std::fmax(wind_speed_canopy_top, 0.01f);

    // Wind speed near the soil surface, which sets the soil boundary-layer conductance. The conductance itself is taken from the soil surface energy balance on each iteration, so that the heat the soil releases is exactly the heat the
    // bottom layer receives.
    bool soil_node_present = !canopy_airspace_ground_UUIDs.empty();
    float wind_speed_soil = wind_speed_canopy_top * std::exp(-attenuation_coefficient);

    // The ground primitives represent the soil surface beneath the whole canopy footprint, so soil fluxes enter the network per unit area of soil. This keeps the soil flux density independent of how far the ground geometry extends
    // beyond the canopy.
    float soil_area = 0.f;
    if (soil_node_present) {
        soil_area = context->sumPrimitiveSurfaceArea(canopy_airspace_ground_UUIDs);
        if (soil_area <= 0.f) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): The ground primitives given to enableCanopyAirspaceModel() have zero total surface area, so they cannot exchange heat or water vapor with the canopy airspace.");
        }
    }

    // Water vapor diffuses more readily than heat, so conductances to moisture are larger by the ratio of diffusivities.
    constexpr float moisture_conductance_ratio = 1.08f;

    updateCanopyWindProfile(wind_speed_canopy_top);

    // -- Initialize the airspace state to the above-canopy boundary condition -- //

    float moisture_reference = air_humidity_reference * esat_Pa(air_temperature_reference) / Patm;

    std::vector<float> T_ac(Nlayers, air_temperature_reference);
    std::vector<float> x_ac(Nlayers, moisture_reference);
    // Warm-start from the previous solution so that repeated calls across a diurnal simulation converge in fewer iterations. The per-layer profile is preferred over the canopy mean because it preserves the vertical structure, and it is
    // used only when its length matches the current layer count so that a profile left over from a different configuration is ignored.
    if (context->doesGlobalDataExist("canopy_air_temperature_layers") && context->getGlobalDataType("canopy_air_temperature_layers") == helios::HELIOS_TYPE_FLOAT) {
        std::vector<float> T_previous_layers;
        context->getGlobalData("canopy_air_temperature_layers", T_previous_layers);
        if (T_previous_layers.size() == Nlayers) {
            T_ac = T_previous_layers;
        }
    }

    // Seed the primitive data that the surface energy balance inputs are gathered from.
    std::vector<float> canopy_air_temperature(Ncanopy);
    std::vector<float> canopy_air_humidity(Ncanopy);
    for (size_t u = 0; u < Ncanopy; u++) {
        uint layer = canopy_airspace_layer_index.at(u);
        canopy_air_temperature.at(u) = T_ac.at(layer);
        canopy_air_humidity.at(u) = std::fmin(x_ac.at(layer) * Patm / esat_Pa(T_ac.at(layer)), 1.f);
    }
    context->setPrimitiveData(canopy_airspace_UUIDs, "air_temperature", canopy_air_temperature);
    context->setPrimitiveData(canopy_airspace_UUIDs, "air_humidity", canopy_air_humidity);
    if (soil_node_present) {
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_temperature", T_ac.front());
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_humidity", std::fmin(x_ac.front() * Patm / esat_Pa(T_ac.front()), 1.f));
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "wind_speed", wind_speed_soil);
    }

    // Only the air temperature and humidity of the canopy and ground primitives change between iterations, so the surface energy balance inputs are gathered from the Context once and the airspace state is then applied to them directly.
    SurfaceEnergyBalanceInputs inputs;
    gatherSurfaceEnergyBalanceInputs(UUIDs, inputs);

    // Position of each UUID within the surface energy balance inputs.
    std::unordered_map<uint, size_t> solve_index_of_UUID;
    solve_index_of_UUID.reserve(UUIDs.size());
    for (size_t s = 0; s < UUIDs.size(); s++) {
        solve_index_of_UUID.emplace(UUIDs.at(s), s);
    }

    // Airspace layer whose air drives each entry of the surface energy balance inputs, or -1 for primitives outside the airspace. The ground exchanges with the bottom layer, and takes precedence for a primitive in both sets as when
    // the airspace state was written to the primitive data.
    std::vector<int> airspace_layer_of_solve_index(UUIDs.size(), -1);
    {
        std::unordered_map<uint, int> layer_of_UUID;
        for (size_t u = 0; u < Ncanopy; u++) {
            layer_of_UUID[canopy_airspace_UUIDs.at(u)] = int(canopy_airspace_layer_index.at(u));
        }
        for (uint UUID: canopy_airspace_ground_UUIDs) {
            layer_of_UUID[UUID] = 0;
        }
        for (size_t s = 0; s < UUIDs.size(); s++) {
            auto it = layer_of_UUID.find(UUIDs.at(s));
            if (it != layer_of_UUID.end()) {
                airspace_layer_of_solve_index.at(s) = it->second;
            }
        }
    }

    std::vector<size_t> canopy_solve_index(Ncanopy);
    std::vector<float> canopy_primitive_area(Ncanopy);
    for (size_t u = 0; u < Ncanopy; u++) {
        canopy_solve_index.at(u) = solve_index_of_UUID.at(canopy_airspace_UUIDs.at(u));
        canopy_primitive_area.at(u) = context->getPrimitiveArea(canopy_airspace_UUIDs.at(u));
    }
    std::vector<size_t> ground_solve_index(canopy_airspace_ground_UUIDs.size());
    std::vector<float> ground_primitive_area(canopy_airspace_ground_UUIDs.size());
    for (size_t u = 0; u < canopy_airspace_ground_UUIDs.size(); u++) {
        ground_solve_index.at(u) = solve_index_of_UUID.at(canopy_airspace_ground_UUIDs.at(u));
        ground_primitive_area.at(u) = context->getPrimitiveArea(canopy_airspace_ground_UUIDs.at(u));
    }

    // -- Outer fixed-point iteration between the surface energy balance and the airspace state -- //

    // Ground area basis for the leaf area weighting. The leaf area index relates total one-sided leaf area to ground area, so the ground area follows from the total canopy area.
    float total_canopy_area = context->sumPrimitiveSurfaceArea(canopy_airspace_UUIDs);
    float ground_area = total_canopy_area / canopy_airspace_LAI;

    std::vector<float> T_ac_previous(Nlayers);
    std::vector<float> x_ac_previous(Nlayers);

    // Tridiagonal system coefficients: sub-diagonal, diagonal, super-diagonal, and right-hand side.
    std::vector<float> sub(Nlayers), diag(Nlayers), super(Nlayers), rhs(Nlayers);

    // Total conductance to water vapor of a surface, and the vapor mole fraction at its evaporating surface. Water vapor leaves through the stomata (or soil pores) and the boundary layer in series, and the boundary-layer conductance is the
    // conductance summed over both faces of a two-sided primitive whereas vapor leaving a face only crosses that face's boundary layer. These are the same expressions the surface energy balance uses to compute the latent flux, so that the water
    // weighted into the airspace matches the water the surface actually released.
    auto evaluateMoistureSource = [&](size_t s, float T_surface, float &g_M, float &x_surface) {
        float g_bl = inputs.gH.at(s);
        float gS = inputs.gS.at(s);
        float stomatal_sidedness = inputs.stomatal_sidedness.at(s);

        g_M = 0.f;
        if (g_bl != 0.f && gS != 0.f) {
            float g_bv = moisture_conductance_ratio * g_bl / float(inputs.Nsides.at(s));
            g_M = g_bv * gS * (stomatal_sidedness / (g_bv + gS * stomatal_sidedness) + (1.f - stomatal_sidedness) / (g_bv + gS * (1.f - stomatal_sidedness)));
        }

        // The evaporating surface is at the relative humidity 'surface_humidity' (saturated by default) at the surface temperature.
        x_surface = inputs.surfacehumidity.at(s) * esat_Pa(T_surface) / Patm;
    };

    std::vector<float> T_surface;
    std::vector<float> humidity_ac(Nlayers);

    uint iteration = 0;
    bool converged = false;
    while (iteration < canopy_airspace_max_iterations && !converged) {

        T_ac_previous = T_ac;
        x_ac_previous = x_ac;

        // 1) Update leaf temperatures given the current airspace state. The inputs gathered before the first iteration already hold the seeded airspace state.
        if (iteration > 0) {
            for (size_t s = 0; s < UUIDs.size(); s++) {
                int layer = airspace_layer_of_solve_index.at(s);
                if (layer >= 0) {
                    inputs.Ta.at(s) = T_ac.at(layer);
                    inputs.ea.at(s) = humidity_ac.at(layer) * esat_Pa(T_ac.at(layer));
                }
            }
        }
        solveSurfaceTemperatures(inputs, 0.f, T_surface);

        // The previous solution is the initial guess for the next one.
        for (size_t s = 0; s < UUIDs.size(); s++) {
            inputs.To.at(s) = (T_surface.at(s) == 0) ? 300.f : T_surface.at(s);
        }

        // 2) Reduce leaf temperature and boundary-layer conductance onto the airspace layers, weighting each leaf by its boundary-layer conductance and its area fraction of the ground.
        std::vector<float> sum_gA_T(Nlayers, 0.f);
        std::vector<float> sum_gA(Nlayers, 0.f);
        std::vector<float> sum_gA_e(Nlayers, 0.f);
        std::vector<float> sum_gA_moisture(Nlayers, 0.f);

        for (size_t u = 0; u < Ncanopy; u++) {
            size_t s = canopy_solve_index.at(u);
            uint layer = canopy_airspace_layer_index.at(u);

            float T_leaf = T_surface.at(s);
            float g_bl = inputs.gH.at(s);

            float area_fraction = canopy_primitive_area.at(u) / ground_area;
            float weight = g_bl * area_fraction;

            sum_gA.at(layer) += weight;
            sum_gA_T.at(layer) += weight * T_leaf;

            // Water vapor leaves the substomatal airspace through the stomata and the boundary layer in series, so the moisture source is weighted by the total conductance to moisture rather than by the boundary-layer conductance alone. A leaf
            // with closed stomata contributes no moisture to the canopy airspace.
            float g_M, x_surface;
            evaluateMoistureSource(s, T_leaf, g_M, x_surface);
            float moisture_weight = g_M * area_fraction;

            sum_gA_moisture.at(layer) += moisture_weight;
            sum_gA_e.at(layer) += moisture_weight * x_surface;
        }

        // 3) Reduce the soil surface onto the bottom layer in the same way, using the conductances the soil surface energy balance itself used, so that the heat and water vapor the soil releases are exactly what the bottom layer receives.
        // Each ground primitive is weighted by its fraction of the soil area, giving conductances per unit ground area.
        float g_soil = 0.f;
        float g_soil_T = 0.f;
        float g_soil_moisture = 0.f;
        float g_soil_x = 0.f;
        if (soil_node_present) {
            for (size_t u = 0; u < canopy_airspace_ground_UUIDs.size(); u++) {
                size_t s = ground_solve_index.at(u);

                float T_soil = T_surface.at(s);
                float g_bl = inputs.gH.at(s);

                float area_fraction = ground_primitive_area.at(u) / soil_area;
                g_soil += g_bl * area_fraction;
                g_soil_T += g_bl * area_fraction * T_soil;

                float g_M, x_surface;
                evaluateMoistureSource(s, T_soil, g_M, x_surface);
                g_soil_moisture += g_M * area_fraction;
                g_soil_x += g_M * area_fraction * x_surface;
            }
        }

        // 4) Solve the tridiagonal system for layer air temperature. Layer 0 is the bottom layer, exchanging with the soil below; layer (Nlayers-1) is the top layer, exchanging with the reference air above.
        for (uint j = 0; j < Nlayers; j++) {
            // g_layer[j] is the conductance across the boundary between layer j and layer j+1, so the link below layer j is g_layer[j-1].
            float g_below = (j == 0) ? g_soil : g_layer.at(j - 1);
            float g_above = (j == Nlayers - 1) ? g_a : g_layer.at(j);

            sub.at(j) = (j == 0) ? 0.f : -g_layer.at(j - 1);
            super.at(j) = (j == Nlayers - 1) ? 0.f : -g_layer.at(j);
            diag.at(j) = sum_gA.at(j) + g_below + g_above;
            rhs.at(j) = sum_gA_T.at(j);

            if (j == 0) {
                rhs.at(j) += g_soil_T;
            }
            if (j == Nlayers - 1) {
                rhs.at(j) += g_a * air_temperature_reference;
            }
        }
        solveTridiagonal(sub, diag, super, rhs, T_ac);

        // 5) Solve the tridiagonal system for layer moisture mole fraction. The bottom layer exchanges water vapor with the soil, which evaporates according to the moisture conductance of the ground primitives (zero by default).
        // Turbulent transport moves heat and water vapor with the same eddies, so the layer-to-layer and layer-to-reference conductances are the same for both. The ratio of vapor to heat diffusivity applies only to the molecular surface
        // boundary layers, and is already carried in the leaf and soil weighting above.
        for (uint j = 0; j < Nlayers; j++) {
            float g_above = (j == Nlayers - 1) ? g_a : g_layer.at(j);
            float g_below = (j == 0) ? g_soil_moisture : g_layer.at(j - 1);

            sub.at(j) = (j == 0) ? 0.f : -g_layer.at(j - 1);
            super.at(j) = (j == Nlayers - 1) ? 0.f : -g_layer.at(j);
            diag.at(j) = sum_gA_moisture.at(j) + g_below + g_above;
            rhs.at(j) = sum_gA_e.at(j);

            if (j == 0) {
                rhs.at(j) += g_soil_x;
            }
            if (j == Nlayers - 1) {
                rhs.at(j) += g_a * moisture_reference;
            }
        }
        solveTridiagonal(sub, diag, super, rhs, x_ac);

        // 6) Bound the updated airspace state. It is applied to the surface energy balance inputs at the start of the next iteration.
        for (uint j = 0; j < Nlayers; j++) {
            float x_saturation = esat_Pa(T_ac.at(j)) / Patm;
            if (x_ac.at(j) > x_saturation) {
                warnings.addWarning("canopy_airspace_supersaturated", "Canopy airspace moisture exceeded saturation and was capped. This can occur when transpiration is high relative to ventilation.");
                x_ac.at(j) = x_saturation;
            }
            if (x_ac.at(j) < 0.f) {
                x_ac.at(j) = 0.f;
            }
        }

        // Relative humidity of each layer, clamped to physical bounds so that every consumer of this state sees the same value.
        for (uint j = 0; j < Nlayers; j++) {
            humidity_ac.at(j) = std::fmin(std::fmax(x_ac.at(j) * Patm / esat_Pa(T_ac.at(j)), 0.f), 1.f);
        }

        // 7) Test for convergence of the airspace state. Moisture is tested alongside temperature because the two are
        // coupled through the saturation vapor pressure at the leaf surface and can converge at different rates.
        float max_change_T = 0.f;
        float max_change_x = 0.f;
        for (uint j = 0; j < Nlayers; j++) {
            max_change_T = std::fmax(max_change_T, std::fabs(T_ac.at(j) - T_ac_previous.at(j)));
            max_change_x = std::fmax(max_change_x, std::fabs(x_ac.at(j) - x_ac_previous.at(j)));
        }
        // The moisture tolerance is expressed as the change in vapor mole fraction that corresponds to the temperature
        // tolerance, so that a single user-facing tolerance controls both.
        float moisture_tolerance = canopy_airspace_tolerance_K * (esat_Pa(air_temperature_reference + 0.5f) - esat_Pa(air_temperature_reference - 0.5f)) / Patm;
        if (max_change_T < canopy_airspace_tolerance_K && max_change_x < moisture_tolerance) {
            converged = true;
        }

        // 8) Update the atmospheric stability from the heat and water vapor leaving the top layer for the reference air, and with it the turbulent conductances for the next iteration. The update is under-relaxed because stability
        // and the fluxes reinforce each other: a stable canopy is more weakly coupled to the air above, which makes it cooler still. The state is converged only once the stability has also settled.
        if (canopy_airspace_stability_correction) {
            // Water vapor adds to the buoyancy of the air through the virtual temperature, T_v = T(1 + 0.378 x) for vapor mole fraction x.
            float virtual_temperature_difference = T_ac.back() * (1.f + 0.378f * x_ac.back()) - air_temperature_reference * (1.f + 0.378f * moisture_reference);
            float inverse_obukhov_length_target = calculateInverseObukhovLength(wind_speed_reference, virtual_temperature_difference, air_temperature_reference);
            float inverse_obukhov_length_updated = inverse_obukhov_length + 0.5f * (inverse_obukhov_length_target - inverse_obukhov_length);
            if (std::fabs(inverse_obukhov_length_updated - inverse_obukhov_length) * (reference_height_m - displacement_height) > 1e-3f) {
                converged = false;
            }
            // The conductances are left unchanged after the final iteration so that the reported resistance is the one the final airspace state was solved with.
            if (!converged && iteration + 1 < canopy_airspace_max_iterations) {
                inverse_obukhov_length = inverse_obukhov_length_updated;
                updateTurbulentConductances();
            }
        }

        iteration++;
    }

    // -- Write the converged state to primitive data -- //

    // Surface temperatures and fluxes are those of the last surface energy balance solution, driven by the airspace state that preceded it.
    writeSurfaceEnergyBalanceOutputs(UUIDs, inputs, T_surface, 0.f);

    for (size_t u = 0; u < Ncanopy; u++) {
        uint layer = canopy_airspace_layer_index.at(u);
        canopy_air_temperature.at(u) = T_ac.at(layer);
        canopy_air_humidity.at(u) = humidity_ac.at(layer);
    }
    context->setPrimitiveData(canopy_airspace_UUIDs, "air_temperature", canopy_air_temperature);
    context->setPrimitiveData(canopy_airspace_UUIDs, "air_humidity", canopy_air_humidity);

    // The soil exchanges heat with the lowest canopy layer, so its own energy balance is driven by that layer's air rather than by the air above the canopy. The wind speed near the soil surface is the one already used to form the soil
    // conductance.
    if (soil_node_present) {
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_temperature", T_ac.front());
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_humidity", humidity_ac.front());
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "wind_speed", wind_speed_soil);
    }

    float stability_parameter = (reference_height_m - displacement_height) * inverse_obukhov_length;
    if (stability_parameter >= 0.999f * canopy_airspace_stability_parameter_max) {
        warnings.addWarning("canopy_airspace_stability_limit_stable", "Canopy airspace stability parameter reached its stable limit of " + std::to_string(canopy_airspace_stability_parameter_max) +
                                                                          ". The canopy is effectively decoupled from the reference air, beyond the range in which Monin-Obukhov similarity theory applies, so the aerodynamic resistance is a lower bound.");
    } else if (stability_parameter <= 0.999f * canopy_airspace_stability_parameter_min) {
        warnings.addWarning("canopy_airspace_stability_limit_unstable", "Canopy airspace stability parameter reached its unstable limit of " + std::to_string(canopy_airspace_stability_parameter_min) + ", approaching free convection.");
    }

    if (!converged) {
        warnings.addWarning("canopy_airspace_not_converged", "Canopy airspace model did not converge within " + std::to_string(canopy_airspace_max_iterations) +
                                                                     " iterations. Results may be unreliable. Consider increasing the maximum number of iterations via setCanopyAirspaceConvergence().");
    }

    // -- Write diagnostic output to global data -- //

    float T_ac_mean = 0.f;
    float x_ac_mean = 0.f;
    for (uint j = 0; j < Nlayers; j++) {
        T_ac_mean += T_ac.at(j);
        x_ac_mean += x_ac.at(j);
    }
    T_ac_mean /= float(Nlayers);
    x_ac_mean /= float(Nlayers);

    context->setGlobalData("canopy_air_temperature", T_ac_mean);
    context->setGlobalData("canopy_air_humidity", x_ac_mean * Patm / esat_Pa(T_ac_mean));
    context->setGlobalData("canopy_air_temperature_layers", T_ac);
    context->setGlobalData("aerodynamic_resistance", aerodynamic_resistance_s_m);
    context->setGlobalData("friction_velocity", friction_velocity);
    context->setGlobalData("canopy_stability_parameter", stability_parameter);
    context->setGlobalData("canopy_airspace_iterations", iteration);

    std::vector<float> humidity_layers(Nlayers);
    for (uint j = 0; j < Nlayers; j++) {
        humidity_layers.at(j) = x_ac.at(j) * Patm / esat_Pa(T_ac.at(j));
    }
    context->setGlobalData("canopy_air_humidity_layers", humidity_layers);

    warnings.report(std::cerr);
}

void EnergyBalanceModel::assignCanopyAirspaceLayers() {

    size_t Ncanopy = canopy_airspace_UUIDs.size();

    canopy_airspace_layer_index.assign(Ncanopy, 0);

    // Determine the vertical extent of the canopy from the primitives themselves, so that layers span the leaf area rather than the whole domain.
    vec2 xbounds, ybounds, zbounds;
    context->getDomainBoundingBox(canopy_airspace_UUIDs, xbounds, ybounds, zbounds);
    canopy_airspace_base_height_m = zbounds.x;

    if (canopy_airspace_layer_count == 1) {
        canopy_airspace_layer_midpoint_m.assign(1, 0.5f * canopy_height_m);
        canopy_airspace_layer_top_m.assign(1, canopy_height_m);
        canopy_airspace_layers_assigned = true;
        return;
    }

    // Sort primitives by centroid height so that equal-area layers can be cut from the bottom upward.
    std::vector<std::pair<float, size_t>> height_index(Ncanopy);
    std::vector<float> area(Ncanopy);
    float total_area = 0.f;
    for (size_t u = 0; u < Ncanopy; u++) {
        uint UUID = canopy_airspace_UUIDs.at(u);
        std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);
        float height = 0.f;
        for (const vec3 &vertex: vertices) {
            height += vertex.z;
        }
        height /= float(vertices.size());
        height_index.at(u) = std::make_pair(height, u);
        area.at(u) = context->getPrimitiveArea(UUID);
        total_area += area.at(u);
    }

    if (total_area <= 0.f) {
        helios_runtime_error("ERROR (EnergyBalanceModel::assignCanopyAirspaceLayers): Total area of canopy primitives is zero, so canopy airspace layers of equal leaf area cannot be defined.");
    }

    std::sort(height_index.begin(), height_index.end());

    // Walk upward through the sorted primitives, advancing to the next layer each time an equal share of the total leaf area has been accumulated.
    float area_per_layer = total_area / float(canopy_airspace_layer_count);
    float area_accumulated = 0.f;
    uint layer = 0;
    std::vector<float> layer_area(canopy_airspace_layer_count, 0.f);
    for (const auto &[height, u]: height_index) {
        canopy_airspace_layer_index.at(u) = layer;
        area_accumulated += area.at(u);
        layer_area.at(layer) += area.at(u);
        // A single primitive may carry more area than one layer's share, so advance past every threshold it crosses rather than only one.
        while (area_accumulated >= area_per_layer * float(layer + 1) && layer + 1 < canopy_airspace_layer_count) {
            layer++;
        }
    }

    // A layer containing no leaf area has no source term, and the solver would silently interpolate a temperature through it. Fail rather than return a profile that looks resolved but is not.
    for (uint j = 0; j < canopy_airspace_layer_count; j++) {
        if (layer_area.at(j) <= 0.f) {
            helios_runtime_error("ERROR (EnergyBalanceModel::assignCanopyAirspaceLayers): Canopy airspace layer " + std::to_string(j) + " of " + std::to_string(canopy_airspace_layer_count) +
                                 " contains no leaf area. This happens when the number of layers is too large for the canopy geometry, either because there are too few canopy primitives (" + std::to_string(Ncanopy) +
                                 " were given) or because leaf area is distributed too unevenly with height. Reduce the number of layers.");
        }
    }

    // Record the height spanned by each layer. Because layers hold equal leaf area rather than equal depth, their thicknesses differ and are needed to convert eddy diffusivity into a conductance between layer midpoints.
    canopy_airspace_layer_top_m.assign(canopy_airspace_layer_count, canopy_height_m);
    canopy_airspace_layer_midpoint_m.assign(canopy_airspace_layer_count, 0.f);
    float layer_bottom = 0.f;
    for (uint j = 0; j < canopy_airspace_layer_count; j++) {
        // The top of a layer is the height of the highest primitive it contains, referenced to the canopy base.
        float layer_top = layer_bottom;
        for (const auto &[height, u]: height_index) {
            if (canopy_airspace_layer_index.at(u) == j) {
                layer_top = std::fmax(layer_top, height - canopy_airspace_base_height_m);
            }
        }
        layer_top = std::fmin(std::fmax(layer_top, layer_bottom), canopy_height_m);
        canopy_airspace_layer_top_m.at(j) = layer_top;
        canopy_airspace_layer_midpoint_m.at(j) = 0.5f * (layer_bottom + layer_top);
        layer_bottom = layer_top;
    }
    // The uppermost layer extends to the nominal canopy top so that the profile spans the full canopy depth.
    canopy_airspace_layer_top_m.back() = canopy_height_m;
    canopy_airspace_layer_midpoint_m.back() = 0.5f * (canopy_airspace_layer_top_m.at(canopy_airspace_layer_count - 2) + canopy_height_m);

    canopy_airspace_layers_assigned = true;
}

float EnergyBalanceModel::calculateAerodynamicResistance(float wind_speed_reference_m_s, float inverse_obukhov_length_per_m, float &friction_velocity_m_s) const {

    // Roughness length for momentum and zero-plane displacement height following Perrier (1982).
    float exp_half_LAI = std::exp(-0.5f * canopy_airspace_LAI);
    float zo_m = canopy_height_m * exp_half_LAI * (1.f - exp_half_LAI);
    float displacement_height = canopy_height_m * (1.f - 2.f / canopy_airspace_LAI * (1.f - exp_half_LAI));

    // Guard against a degenerate roughness geometry, which would otherwise produce a division by zero or a negative resistance.
    if (zo_m <= 0.f || displacement_height >= canopy_height_m) {
        helios_runtime_error("ERROR (EnergyBalanceModel::calculateAerodynamicResistance): Canopy roughness geometry is degenerate for a canopy height of " + std::to_string(canopy_height_m) + " m and a leaf area index of " +
                             std::to_string(canopy_airspace_LAI) + ". Check that the canopy height and leaf area index are physically reasonable.");
    }

    // Aerodynamic resistance between the canopy top and the reference height following Perrier (1975).
    float ln_momentum = std::log((reference_height_m - displacement_height) / zo_m);
    float ln_heat = std::log((reference_height_m - displacement_height) / (canopy_height_m - displacement_height));

    // Both logarithms vanish as the reference height approaches the canopy top, which would make the canopy perfectly coupled to the reference air. Require a physically meaningful separation rather than returning a near-zero resistance.
    if (ln_momentum <= 0.f || ln_heat <= 0.f) {
        helios_runtime_error("ERROR (EnergyBalanceModel::calculateAerodynamicResistance): The reference height (" + std::to_string(reference_height_m) + " m) is too close to the canopy top (" + std::to_string(canopy_height_m) +
                             " m) for the aerodynamic resistance to be defined, given a leaf area index of " + std::to_string(canopy_airspace_LAI) + ". Increase the reference height so that it lies within the surface layer above the canopy.");
    }

    // Monin-Obukhov stability corrections integrated between the same lower and upper heights as the logarithms. Momentum is integrated from the roughness length and heat from the canopy top, both to the reference height.
    float zeta_reference = (reference_height_m - displacement_height) * inverse_obukhov_length_per_m;
    float profile_momentum = ln_momentum - calculateStabilityCorrectionMomentum(zeta_reference) + calculateStabilityCorrectionMomentum(zo_m * inverse_obukhov_length_per_m);
    float profile_heat = ln_heat - calculateStabilityCorrectionHeat(zeta_reference) + calculateStabilityCorrectionHeat((canopy_height_m - displacement_height) * inverse_obukhov_length_per_m);

    friction_velocity_m_s = von_Karman_constant * wind_speed_reference_m_s / profile_momentum;

    return profile_heat / (von_Karman_constant * friction_velocity_m_s);
}

float EnergyBalanceModel::calculateInverseObukhovLength(float wind_speed_reference_m_s, float virtual_temperature_difference_K, float air_temperature_K) const {

    float exp_half_LAI = std::exp(-0.5f * canopy_airspace_LAI);
    float displacement_height = canopy_height_m * (1.f - 2.f / canopy_airspace_LAI * (1.f - exp_half_LAI));

    float inverse_obukhov_length_min = canopy_airspace_stability_parameter_min / (reference_height_m - displacement_height);
    float inverse_obukhov_length_max = canopy_airspace_stability_parameter_max / (reference_height_m - displacement_height);

    // The kinematic buoyancy flux is the virtual temperature difference divided by the aerodynamic resistance, and both it and the friction velocity depend on the stability. Iterating with the temperature difference held fixed,
    // rather than the flux, is equivalent to iterating on the bulk Richardson number and has a solution under stable conditions, where a fixed downward flux can exceed what the suppressed turbulence is able to carry.
    float inverse_obukhov_length = 0.f;
    for (int iteration = 0; iteration < 100; iteration++) {
        float friction_velocity;
        float aerodynamic_resistance = calculateAerodynamicResistance(wind_speed_reference_m_s, inverse_obukhov_length, friction_velocity);
        float buoyancy_flux_K_m_s = virtual_temperature_difference_K / aerodynamic_resistance;
        float updated = -von_Karman_constant * 9.81f * buoyancy_flux_K_m_s / (air_temperature_K * std::pow(friction_velocity, 3));
        updated = std::fmin(std::fmax(updated, inverse_obukhov_length_min), inverse_obukhov_length_max);
        bool converged = std::fabs(updated - inverse_obukhov_length) * (reference_height_m - displacement_height) < 1e-5f;
        inverse_obukhov_length = updated;
        if (converged) {
            return inverse_obukhov_length;
        }
    }

    helios_runtime_error("ERROR (EnergyBalanceModel::calculateInverseObukhovLength): The Obukhov length did not converge for a reference wind speed of " + std::to_string(wind_speed_reference_m_s) + " m/s and a canopy-to-reference virtual temperature difference of " +
                         std::to_string(virtual_temperature_difference_K) + " K. Disable stability correction with disableCanopyAirspaceStabilityCorrection() to proceed assuming neutral stability.");
    return 0.f;
}

float EnergyBalanceModel::calculateStabilityCorrectionMomentum(float zeta) {
    if (zeta < 0.f) {
        // Paulson (1970)
        float x = std::pow(1.f - 16.f * zeta, 0.25f);
        return 2.f * std::log(0.5f * (1.f + x)) + std::log(0.5f * (1.f + x * x)) - 2.f * std::atan(x) + 0.5f * float(M_PI);
    }
    // Beljaars and Holtslag (1991)
    constexpr float a = 1.f, b = 2.f / 3.f, c = 5.f, d = 0.35f;
    return -(a * zeta + b * (zeta - c / d) * std::exp(-d * zeta) + b * c / d);
}

float EnergyBalanceModel::calculateStabilityCorrectionHeat(float zeta) {
    if (zeta < 0.f) {
        // Paulson (1970)
        return 2.f * std::log(0.5f * (1.f + std::sqrt(1.f - 16.f * zeta)));
    }
    // Beljaars and Holtslag (1991)
    constexpr float a = 1.f, b = 2.f / 3.f, c = 5.f, d = 0.35f;
    return 1.f - std::pow(1.f + 2.f * a * zeta / 3.f, 1.5f) - b * (zeta - c / d) * std::exp(-d * zeta) - b * c / d;
}

float EnergyBalanceModel::calculateStabilityFunctionHeat(float zeta) {
    if (zeta < 0.f) {
        // Businger-Dyer
        return 1.f / std::sqrt(1.f - 16.f * zeta);
    }
    // Beljaars and Holtslag (1991), phi_h = 1 - zeta * dpsi_h/dzeta
    constexpr float a = 1.f, b = 2.f / 3.f, c = 5.f, d = 0.35f;
    return 1.f + zeta * (a * std::sqrt(1.f + 2.f * a * zeta / 3.f) + b * std::exp(-d * zeta) * (1.f + c - d * zeta));
}

float EnergyBalanceModel::calculateGroundBoundaryLayerConductance(float wind_speed_m_s) {
    // Eq. A17 of Kustas and Norman (1999), in m/s, converted to a molar conductance with the molar density of air at standard temperature and pressure (41.56 mol/m^3), matching the "Ground" model of the boundary-layer conductance
    // plug-in.
    return (0.004f + 0.012f * wind_speed_m_s) * 41.56f;
}

void EnergyBalanceModel::updateCanopyWindProfile(float wind_speed_canopy_top_m_s) {

    // Exponential within-canopy wind profile of Cionco (1972), with an attenuation coefficient of one half of the leaf area index.
    float attenuation_coefficient = 0.5f * canopy_airspace_LAI;

    for (uint UUID: canopy_airspace_UUIDs) {
        std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);
        float height = 0.f;
        for (const vec3 &vertex: vertices) {
            height += vertex.z;
        }
        height = height / float(vertices.size()) - canopy_airspace_base_height_m;

        // Clamp to the canopy top so that primitives extending above the nominal canopy height do not receive wind speeds greater than that at the canopy top.
        float height_fraction = std::fmin(height / canopy_height_m, 1.f);
        height_fraction = std::fmax(height_fraction, 0.f);

        float wind_speed = wind_speed_canopy_top_m_s * std::exp(-attenuation_coefficient * (1.f - height_fraction));
        context->setPrimitiveData(UUID, "wind_speed", wind_speed);
    }
}

void EnergyBalanceModel::enableCanopyAirspaceModel(const std::vector<uint> &canopy_UUIDs, const std::vector<uint> &ground_UUIDs, float canopy_height_m, float reference_height_m, float leaf_area_index, uint num_layers) {

    if (canopy_UUIDs.empty()) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): No canopy primitives were given. The canopy airspace model requires at least one canopy primitive to exchange heat and moisture with the air.");
    } else if (canopy_height_m <= 0) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): Canopy height must be greater than zero.");
    } else if (reference_height_m <= canopy_height_m) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): Reference height (" + std::to_string(reference_height_m) + " m) must be greater than the canopy height (" + std::to_string(canopy_height_m) +
                             " m). The reference height is where the above-canopy air temperature, humidity, and wind speed are measured.");
    } else if (leaf_area_index < 0.5f) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): Leaf area index must be at least 0.5. The Perrier roughness relations used to compute aerodynamic resistance are not valid for sparser canopies, and a value of " +
                             std::to_string(leaf_area_index) + " was given.");
    } else if (num_layers == 0) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): Number of canopy airspace layers must be at least 1. Use num_layers=1 for a single within-canopy node.");
    }

    for (uint UUID: canopy_UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): Canopy primitive UUID " + std::to_string(UUID) + " does not exist in the Context.");
        }
    }
    for (uint UUID: ground_UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): Ground primitive UUID " + std::to_string(UUID) + " does not exist in the Context.");
        }
    }

    if (air_energy_balance_enabled) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableCanopyAirspaceModel): The air energy balance model is already enabled. These two models both determine the within-canopy air state and share the canopy and reference heights, so only "
                             "one may be active at a time.");
    }

    canopy_airspace_enabled = true;
    canopy_airspace_UUIDs = canopy_UUIDs;
    canopy_airspace_ground_UUIDs = ground_UUIDs;
    this->canopy_height_m = canopy_height_m;
    this->reference_height_m = reference_height_m;
    canopy_airspace_LAI = leaf_area_index;
    canopy_airspace_layer_count = num_layers;
    canopy_airspace_layers_assigned = false;

    // The airspace solution weights leaf temperatures by their boundary-layer conductance, so this output must be available in the Context.
    if (std::find(output_prim_data.begin(), output_prim_data.end(), "boundarylayer_conductance_out") == output_prim_data.end()) {
        output_prim_data.emplace_back("boundarylayer_conductance_out");
    }
}

void EnergyBalanceModel::disableCanopyAirspaceModel() {
    canopy_airspace_enabled = false;
    canopy_airspace_layers_assigned = false;
}

void EnergyBalanceModel::setCanopyAirspaceConvergence(float tolerance_K, uint max_iterations) {

    if (tolerance_K <= 0) {
        helios_runtime_error("ERROR (EnergyBalanceModel::setCanopyAirspaceConvergence): Convergence tolerance must be greater than zero.");
    } else if (max_iterations == 0) {
        helios_runtime_error("ERROR (EnergyBalanceModel::setCanopyAirspaceConvergence): Maximum number of iterations must be at least 1.");
    }

    canopy_airspace_tolerance_K = tolerance_K;
    canopy_airspace_max_iterations = max_iterations;
}

void EnergyBalanceModel::enableCanopyAirspaceStabilityCorrection() {
    canopy_airspace_stability_correction = true;
}

void EnergyBalanceModel::disableCanopyAirspaceStabilityCorrection() {
    canopy_airspace_stability_correction = false;
}

void EnergyBalanceModel::optionalOutputPrimitiveData(const char *label) {

    if (strcmp(label, "boundarylayer_conductance_out") == 0 || strcmp(label, "vapor_pressure_deficit") == 0 || strcmp(label, "storage_flux") == 0 || strcmp(label, "net_radiation_flux") == 0) {
        output_prim_data.emplace_back(label);
    } else {
        std::cerr << "WARNING (EnergyBalanceModel::optionalOutputPrimitiveData): unknown output primitive data '" << label << "' will be ignored." << std::endl;
    }
}

void EnergyBalanceModel::printDefaultValueReport() const {
    printDefaultValueReport(context->getAllUUIDs());
}

void EnergyBalanceModel::printDefaultValueReport(const std::vector<uint> &UUIDs) const {

    size_t assumed_default_TL = 0;
    size_t assumed_default_U = 0;
    size_t assumed_default_L = 0;
    size_t assumed_default_p = 0;
    size_t assumed_default_Ta = 0;
    size_t assumed_default_rh = 0;
    size_t assumed_default_gH = 0;
    size_t assumed_default_gH_ground = 0;
    size_t assumed_default_gs = 0;
    size_t assumed_default_Qother = 0;
    size_t assumed_default_heatcapacity = 0;
    size_t assumed_default_fs = 0;
    size_t twosided_0 = 0;
    size_t twosided_1 = 0;
    size_t Ne_1 = 0;
    size_t Ne_2 = 0;

    size_t Nprimitives = UUIDs.size();

    std::unordered_set<uint> canopy_airspace_ground_set;
    if (canopy_airspace_enabled) {
        canopy_airspace_ground_set.insert(canopy_airspace_ground_UUIDs.begin(), canopy_airspace_ground_UUIDs.end());
    }

    for (uint UUID: UUIDs) {

        // surface temperature (K)
        if (!context->doesPrimitiveDataExist(UUID, "temperature") || context->getPrimitiveDataType("temperature") != HELIOS_TYPE_FLOAT) {
            assumed_default_TL++;
        }

        // air pressure (Pa)
        if (!context->doesPrimitiveDataExist(UUID, "air_pressure") || context->getPrimitiveDataType("air_pressure") != HELIOS_TYPE_FLOAT) {
            assumed_default_p++;
        }

        // air temperature (K)
        if (!context->doesPrimitiveDataExist(UUID, "air_temperature") || context->getPrimitiveDataType("air_temperature") != HELIOS_TYPE_FLOAT) {
            assumed_default_Ta++;
        }

        // air humidity
        if (!context->doesPrimitiveDataExist(UUID, "air_humidity") || context->getPrimitiveDataType("air_humidity") != HELIOS_TYPE_FLOAT) {
            assumed_default_rh++;
        }

        // boundary-layer conductance to heat
        if (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") || context->getPrimitiveDataType("boundarylayer_conductance") != HELIOS_TYPE_FLOAT) {
            if (canopy_airspace_ground_set.count(UUID) > 0) {
                assumed_default_gH_ground++;
            } else {
                assumed_default_gH++;
            }
        }

        // wind speed
        if (!context->doesPrimitiveDataExist(UUID, "wind_speed") || context->getPrimitiveDataType("wind_speed") != HELIOS_TYPE_FLOAT) {
            assumed_default_U++;
        }

        // object length
        if (!context->doesPrimitiveDataExist(UUID, "object_length") || context->getPrimitiveDataType("object_length") != HELIOS_TYPE_FLOAT) {
            assumed_default_L++;
        }

        // moisture conductance
        if (!context->doesPrimitiveDataExist(UUID, "moisture_conductance") || context->getPrimitiveDataType("moisture_conductance") != HELIOS_TYPE_FLOAT) {
            assumed_default_gs++;
        }

        // Heat capacity
        if (!context->doesPrimitiveDataExist(UUID, "heat_capacity") || context->getPrimitiveDataType("heat_capacity") != HELIOS_TYPE_FLOAT) {
            assumed_default_heatcapacity++;
        }

        //"Other" heat fluxes
        if (!context->doesPrimitiveDataExist(UUID, "other_surface_flux") || context->getPrimitiveDataType("other_surface_flux") != HELIOS_TYPE_FLOAT) {
            assumed_default_Qother++;
        }

        // two-sided flag - check material first, then primitive data
        uint twosided = context->getPrimitiveTwosidedFlag(UUID, 1);
        if (twosided == 0) {
            twosided_0++;
        } else {
            twosided_1++;
        }

        // number of evaporating faces
        if (context->doesPrimitiveDataExist(UUID, "evaporating_faces") && context->getPrimitiveDataType("evaporating_faces") == HELIOS_TYPE_UINT) {
            uint Ne;
            context->getPrimitiveData(UUID, "evaporating_faces", Ne);
            if (Ne == 1) {
                Ne_1++;
            } else if (Ne == 2) {
                Ne_2++;
            }
        } else {
            Ne_1++;
        }

        // Surface humidity
        if (!context->doesPrimitiveDataExist(UUID, "surface_humidity") || context->getPrimitiveDataType("surface_humidity") != HELIOS_TYPE_FLOAT) {
            assumed_default_fs++;
        }
    }

    std::cout << "--- Energy Balance Model Default Value Report ---" << std::endl;

    std::cout << "surface temperature (initial guess): " << assumed_default_TL << " of " << Nprimitives << " used default value of " << temperature_default
              << " because "
                 "temperature"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "air pressure: " << assumed_default_p << " of " << Nprimitives << " used default value of " << pressure_default
              << " because "
                 "air_pressure"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "air temperature: " << assumed_default_Ta << " of " << Nprimitives << " used default value of " << air_temperature_default
              << " because "
                 "air_temperature"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "air humidity: " << assumed_default_rh << " of " << Nprimitives << " used default value of " << air_humidity_default
              << " because "
                 "air_humidity"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "boundary-layer conductance: " << assumed_default_gH << " of " << Nprimitives
              << " calculated boundary-layer conductance from Polhausen equation because "
                 "boundarylayer_conductance"
                 " primitive data did not exist"
              << std::endl;
    if (canopy_airspace_enabled && !canopy_airspace_ground_UUIDs.empty()) {
        std::cout << "boundary-layer conductance (canopy airspace ground): " << assumed_default_gH_ground << " of " << Nprimitives
                  << " calculated boundary-layer conductance from the Kustas and Norman (1999) bare-soil relation because "
                     "boundarylayer_conductance"
                     " primitive data did not exist"
                  << std::endl;
    }
    if (assumed_default_gH > 0) {
        std::cout << "  - wind speed: " << assumed_default_U << " of " << assumed_default_gH << " using Polhausen equation used default value of " << wind_speed_default
                  << " because "
                     "wind_speed"
                     " primitive data did not exist"
                  << std::endl;
        std::cout << "  - object length: " << assumed_default_L << " of " << assumed_default_gH
                  << " using Polhausen equation used the primitive/object length/area to calculate object length because "
                     "object_length"
                     " primitive data did not exist"
                  << std::endl;
    }
    std::cout << "moisture conductance: " << assumed_default_gs << " of " << Nprimitives << " used default value of " << gS_default
              << " because "
                 "moisture_conductance"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "surface humidity: " << assumed_default_fs << " of " << Nprimitives << " used default value of " << surface_humidity_default
              << " because "
                 "surface_humidity"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "two-sided flag: " << twosided_0 << " of " << Nprimitives << " used two-sided flag=0; " << twosided_1 << " of " << Nprimitives << " used two-sided flag=1 (default)" << std::endl;
    std::cout << "evaporating faces: " << Ne_1 << " of " << Nprimitives << " used Ne = 1 (default); " << Ne_2 << " of " << Nprimitives << " used Ne = 2" << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;
}

// Helper function: CPU version of energy balance evaluation
static inline float evaluateEnergyBalance_CPU(float T, float R, float Qother, float eps, float Ta, float ea, float pressure, float gH, float gS, uint Nsides, float stomatal_sidedness, float heatcapacity, float surfacehumidity, float dt,
                                              float Tprev) {

    // Outgoing emission flux
    float Rout = float(Nsides) * eps * 5.67e-8F * T * T * T * T;

    // Sensible heat flux
    float QH = cp_air_mol * gH * (T - Ta);

    // Latent heat flux
    float es = esat_Pa(T);
    // A zero boundary-layer conductance (gH) or zero stomatal conductance (gS) means there is no vapor
    // pathway, so gM is zero. Guarding both cases also avoids a 0/0 in the series-conductance expression
    // when gH == 0 and the stomatal sidedness makes one of the denominators vanish.
    float gM = 0.f;
    if (gH != 0.f && gS != 0.f) {
        // gH is summed over all faces, but vapor leaving a face only crosses that face's boundary layer
        float gbv = 1.08f * gH / float(Nsides);
        gM = gbv * gS * (stomatal_sidedness / (gbv + gS * stomatal_sidedness) + (1.f - stomatal_sidedness) / (gbv + gS * (1.f - stomatal_sidedness)));
    }

    float QL = gM * lambda_mol * (es * surfacehumidity - ea) / pressure;

    // Storage heat flux
    float storage = 0.f;
    if (dt > 0) {
        storage = heatcapacity * (T - Tprev) / dt;
    }

    // Residual
    return R - Rout - QH - QL - Qother - storage;
}

void EnergyBalanceModel::gatherSurfaceEnergyBalanceInputs(const std::vector<uint> &UUIDs, SurfaceEnergyBalanceInputs &inputs) {

    // Create warning aggregator for default value warnings
    helios::WarningAggregator warnings;
    warnings.setEnabled(message_flag);

    if (radiation_bands.empty()) {
        helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation bands were found.");
    }

    const uint Nprimitives = UUIDs.size();

    // The type of a primitive data label is the same for every primitive, so it is checked once per label here rather than once per primitive. A label that no primitive carries is not looked up on each primitive at all.
    std::unordered_set<std::string> existing_labels;
    {
        std::vector<std::string> labels = context->listAllPrimitiveDataLabels();
        existing_labels.insert(labels.begin(), labels.end());
    }
    auto isPrimitiveDataLabelOfType = [&](const std::string &label, helios::HeliosDataType type) { return existing_labels.find(label) != existing_labels.end() && context->getPrimitiveDataType(label.c_str()) == type; };

    // Absorbed radiation flux labels, summed over all bands for each primitive below
    std::vector<std::string> flux_labels;
    for (const auto &band: radiation_bands) {
        std::string flux_label = "radiation_flux_" + band;
        if (existing_labels.find(flux_label) != existing_labels.end() && context->getPrimitiveDataType(flux_label.c_str()) != helios::HELIOS_TYPE_FLOAT) {
            helios_runtime_error("ERROR (EnergyBalanceModel::run): Radiation primitive data for band " + band + " does not have the correct type of 'float'");
        }
        flux_labels.push_back(flux_label);
    }

    // Determine which radiation band(s) govern the emitted thermal radiation. The RadiationModel records per-band emission-enabled state in global data 'emission_enabled_<band>', and the wavelength bounds of each band that
    // has them in 'wavelength_bounds_<band>'. A band without bounds emits eps*sigma*T^4, and a band with bounds emits only its share of the blackbody spectrum, eps*sigma*T^4*F(T), so the emission of the surface is the sum over
    // the emission-enabled bands. When no emission-enabled information is available (e.g. radiation fluxes were set manually without running the RadiationModel), all bands added to the energy balance are treated as candidates
    // for a single broadband emissivity, for backward compatibility.
    struct EmissionBand {
        std::string label;
        std::string emissivity_label;
        bool emissivity_given = false;
        // The RadiationModel treats surfaces as black in an emission band whose scattering depth is 0, emitting sigma*T^4 regardless of the emissivity
        bool emits_as_blackbody = false;
        bool has_wavelength_bounds = false;
        helios::vec2 wavelength_bounds_nm;
    };
    std::vector<EmissionBand> emission_bands;
    bool emission_info_available = false;
    // Global data written by the RadiationModel that exists but has the wrong type was not written by it, and would otherwise silently select a different emission model
    auto requireGlobalDataType = [&](const std::string &label, helios::HeliosDataType type, const std::string &type_name) {
        if (context->doesGlobalDataExist(label.c_str()) && context->getGlobalDataType(label.c_str()) != type) {
            helios_runtime_error("ERROR (EnergyBalanceModel::run): Global data '" + label + "' must be of type " + type_name + ", as written by RadiationModel::runBand(), but it has a different type. Remove it or set it with the correct type.");
        }
    };
    for (const auto &band: radiation_bands) {
        std::string gdlabel = "emission_enabled_" + band;
        requireGlobalDataType(gdlabel, helios::HELIOS_TYPE_UINT, "uint");
        if (context->doesGlobalDataExist(gdlabel.c_str())) {
            emission_info_available = true;
            uint enabled = 0;
            context->getGlobalData(gdlabel.c_str(), enabled);
            if (enabled == 1) {
                EmissionBand emission_band;
                emission_band.label = band;
                emission_band.emissivity_label = "emissivity_" + band;
                emission_band.emissivity_given = isPrimitiveDataLabelOfType(emission_band.emissivity_label, helios::HELIOS_TYPE_FLOAT);
                std::string scattering_depth_label = "scattering_depth_" + band;
                requireGlobalDataType(scattering_depth_label, helios::HELIOS_TYPE_UINT, "uint");
                if (context->doesGlobalDataExist(scattering_depth_label.c_str())) {
                    uint scattering_depth = 0;
                    context->getGlobalData(scattering_depth_label.c_str(), scattering_depth);
                    emission_band.emits_as_blackbody = scattering_depth == 0;
                }
                std::string bounds_label = "wavelength_bounds_" + band;
                requireGlobalDataType(bounds_label, helios::HELIOS_TYPE_VEC2, "vec2");
                if (context->doesGlobalDataExist(bounds_label.c_str())) {
                    emission_band.has_wavelength_bounds = true;
                    context->getGlobalData(bounds_label.c_str(), emission_band.wavelength_bounds_nm);
                }
                emission_bands.push_back(emission_band);
            }
        }
    }
    // Bounded emission bands that overlap would count the emission at their shared wavelengths twice
    for (size_t b = 0; b < emission_bands.size(); b++) {
        for (size_t c = b + 1; c < emission_bands.size(); c++) {
            const EmissionBand &band_b = emission_bands.at(b);
            const EmissionBand &band_c = emission_bands.at(c);
            if (band_b.has_wavelength_bounds && band_c.has_wavelength_bounds && band_b.wavelength_bounds_nm.x < band_c.wavelength_bounds_nm.y && band_c.wavelength_bounds_nm.x < band_b.wavelength_bounds_nm.y) {
                helios_runtime_error("ERROR (EnergyBalanceModel::run): Emission is enabled for radiation bands '" + band_b.label + "' (" + std::to_string(band_b.wavelength_bounds_nm.x) + "-" + std::to_string(band_b.wavelength_bounds_nm.y) +
                                     " nm) and '" + band_c.label + "' (" + std::to_string(band_c.wavelength_bounds_nm.x) + "-" + std::to_string(band_c.wavelength_bounds_nm.y) +
                                     " nm), whose wavelength bounds overlap, so the thermal emission at the shared wavelengths would be counted twice. Either make the bands' bounds adjacent, or remove one band from the energy balance model.");
            }
        }
    }

    // Range of the summed blackbody fraction of the emission bands that is accepted as accounting for the blackbody spectrum once. A small shortfall is unavoidable with bounded bands, whose bounds cannot reach zero and infinite
    // wavelength; an excess means that some emission is counted twice.
    constexpr float min_blackbody_fraction_sum = 0.95f;
    constexpr float max_blackbody_fraction_sum = 1.01f;

    // Without emission-enabled information: emissivity labels of all bands, in band order, that some primitive defines as a float
    std::vector<std::string> emissivity_labels;
    if (!emission_info_available) {
        for (const auto &band: radiation_bands) {
            std::string emissivity_label = "emissivity_" + band;
            if (isPrimitiveDataLabelOfType(emissivity_label, helios::HELIOS_TYPE_FLOAT)) {
                emissivity_labels.push_back(emissivity_label);
            }
        }
    }

    const bool temperature_given = isPrimitiveDataLabelOfType("temperature", helios::HELIOS_TYPE_FLOAT);
    const bool air_temperature_given = isPrimitiveDataLabelOfType("air_temperature", helios::HELIOS_TYPE_FLOAT);
    const bool air_humidity_given = isPrimitiveDataLabelOfType("air_humidity", helios::HELIOS_TYPE_FLOAT);
    const bool air_pressure_given = isPrimitiveDataLabelOfType("air_pressure", helios::HELIOS_TYPE_FLOAT);
    const bool stomatal_sidedness_given = isPrimitiveDataLabelOfType("stomatal_sidedness", helios::HELIOS_TYPE_FLOAT);
    const bool evaporating_faces_given = isPrimitiveDataLabelOfType("evaporating_faces", helios::HELIOS_TYPE_UINT);
    const bool boundarylayer_conductance_given = isPrimitiveDataLabelOfType("boundarylayer_conductance", helios::HELIOS_TYPE_FLOAT);
    const bool wind_speed_given = isPrimitiveDataLabelOfType("wind_speed", helios::HELIOS_TYPE_FLOAT);
    const bool object_length_given = isPrimitiveDataLabelOfType("object_length", helios::HELIOS_TYPE_FLOAT);
    const bool moisture_conductance_given = isPrimitiveDataLabelOfType("moisture_conductance", helios::HELIOS_TYPE_FLOAT);
    const bool other_surface_flux_given = isPrimitiveDataLabelOfType("other_surface_flux", helios::HELIOS_TYPE_FLOAT);
    const bool heat_capacity_given = isPrimitiveDataLabelOfType("heat_capacity", helios::HELIOS_TYPE_FLOAT);
    const bool surface_humidity_given = isPrimitiveDataLabelOfType("surface_humidity", helios::HELIOS_TYPE_FLOAT);

    //---- Set up temperature solution ----//

    inputs.R.resize(Nprimitives);
    inputs.To.resize(Nprimitives);
    inputs.Qother.resize(Nprimitives);
    inputs.eps.resize(Nprimitives);
    inputs.Ta.resize(Nprimitives);
    inputs.ea.resize(Nprimitives);
    inputs.pressure.resize(Nprimitives);
    inputs.gH.resize(Nprimitives);
    inputs.gS.resize(Nprimitives);
    inputs.Nsides.resize(Nprimitives);
    inputs.stomatal_sidedness.resize(Nprimitives);
    inputs.heatcapacity.resize(Nprimitives);
    inputs.surfacehumidity.resize(Nprimitives);

    bool calculated_blconductance_used = false;
    bool primitive_length_used = false;

    // Ground primitives of the canopy airspace model receive a bare-soil boundary-layer conductance by default (see below).
    std::unordered_set<uint> canopy_airspace_ground_set;
    if (canopy_airspace_enabled) {
        canopy_airspace_ground_set.insert(canopy_airspace_ground_UUIDs.begin(), canopy_airspace_ground_UUIDs.end());
    }

    // Primitives are read in parallel. Only const Context queries are made inside the loop. An exception cannot propagate out of an OpenMP parallel region, so the first one thrown is held and rethrown after the region.
    std::exception_ptr gather_exception = nullptr;

    // Missing-data warnings can occur for every primitive, so they are counted per thread and passed to the warning aggregator after the loop rather than contending for its lock on every primitive. Warnings that report an invalid
    // input value are rare and are added to the (thread-safe) aggregator directly.
    int missing_surface_temperature_count = 0;
    int missing_air_temperature_count = 0;
    int missing_air_humidity_count = 0;
    int missing_wind_speed_count = 0;
    int missing_emissivity_count = 0;
    int multiple_emissivity_bands_count = 0;

#ifdef USE_OPENMP
#pragma omp parallel
#endif
    {
        // Characteristic length of each compound object, shared by all of its primitives. The area of an object is a sum over all of its primitives, so it is computed once per object (per thread) rather than once per primitive.
        std::unordered_map<uint, float> object_length_cache;

#ifdef USE_OPENMP
#pragma omp for schedule(static) reduction(|| : calculated_blconductance_used, primitive_length_used)                                                                                                                                                    \
        reduction(+ : missing_surface_temperature_count, missing_air_temperature_count, missing_air_humidity_count, missing_wind_speed_count, missing_emissivity_count, multiple_emissivity_bands_count)
#endif
        for (int u = 0; u < (int) Nprimitives; u++) {
            try {
                uint p = UUIDs.at(u);

                // Absorbed radiation summed over all bands
                inputs.R[u] = 0.f;
                for (size_t b = 0; b < flux_labels.size(); b++) {
                    if (!context->doesPrimitiveDataExist(p, flux_labels[b].c_str())) {
                        helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation was found in the context for band " + radiation_bands.at(b) + ". Did you run the radiation model for this band?");
                    }
                    float R;
                    context->getPrimitiveData(p, flux_labels[b].c_str(), R);
                    inputs.R[u] += R;
                }

                // Initial guess for surface temperature
                if (temperature_given && context->doesPrimitiveDataExist(p, "temperature")) {
                    context->getPrimitiveData(p, "temperature", inputs.To[u]);
                } else {
                    inputs.To[u] = temperature_default;
                    missing_surface_temperature_count++;
                }
                if (inputs.To[u] == 0) {
                    inputs.To[u] = 300;
                }

                // Air temperature
                if (air_temperature_given && context->doesPrimitiveDataExist(p, "air_temperature")) {
                    context->getPrimitiveData(p, "air_temperature", inputs.Ta[u]);
                    if (inputs.Ta[u] < 250.f) {
                        warnings.addWarning("air_temperature_likely_celsius",
                                            "Value of " + std::to_string(inputs.Ta[u]) + " in 'air_temperature' is very small - should be in Kelvin, using default (" + std::to_string(air_temperature_default) + " K)");
                        inputs.Ta[u] = air_temperature_default;
                    }
                } else {
                    inputs.Ta[u] = air_temperature_default;
                    missing_air_temperature_count++;
                }

                // Air relative humidity
                float hr;
                if (air_humidity_given && context->doesPrimitiveDataExist(p, "air_humidity")) {
                    context->getPrimitiveData(p, "air_humidity", hr);
                    if (hr > 1.f) {
                        warnings.addWarning("air_humidity_out_of_range_high", "Value of " + std::to_string(hr) + " in 'air_humidity' > 1.0 - should be fractional [0,1], using default (" + std::to_string(air_humidity_default) + ")");
                        hr = air_humidity_default;
                    } else if (hr < 0.f) {
                        warnings.addWarning("air_humidity_out_of_range_low", "Value of " + std::to_string(hr) + " in 'air_humidity' < 0.0 - should be fractional [0,1], using default (" + std::to_string(air_humidity_default) + ")");
                        hr = air_humidity_default;
                    }
                } else {
                    hr = air_humidity_default;
                    missing_air_humidity_count++;
                }

                // Air vapor pressure
                float esat = esat_Pa(inputs.Ta[u]);
                inputs.ea[u] = hr * esat;

                // Air pressure
                if (air_pressure_given && context->doesPrimitiveDataExist(p, "air_pressure")) {
                    context->getPrimitiveData(p, "air_pressure", inputs.pressure[u]);
                    if (inputs.pressure[u] < 10000.f) {
                        if (message_flag) {
                            warnings.addWarning("air_pressure_out_of_range", "Value of " + std::to_string(inputs.pressure[u]) +
                                                                                     " given in 'air_pressure' primitive data is very small. "
                                                                                     "Values should be given in units of Pascals. Assuming default value of " +
                                                                                     std::to_string(pressure_default));
                        }
                        inputs.pressure[u] = pressure_default;
                    }
                } else {
                    inputs.pressure[u] = pressure_default;
                }

                // Number of sides emitting radiation
                uint twosided_flag = context->getPrimitiveTwosidedFlag(p, 1);
                inputs.Nsides[u] = (twosided_flag == 0) ? 1 : 2;

                // Number of evaporating/transpiring faces
                inputs.stomatal_sidedness[u] = 0.f;
                if (inputs.Nsides[u] == 2 && stomatal_sidedness_given && context->doesPrimitiveDataExist(p, "stomatal_sidedness")) {
                    context->getPrimitiveData(p, "stomatal_sidedness", inputs.stomatal_sidedness[u]);
                } else if (inputs.Nsides[u] == 2 && evaporating_faces_given && context->doesPrimitiveDataExist(p, "evaporating_faces")) {
                    uint flag;
                    context->getPrimitiveData(p, "evaporating_faces", flag);
                    if (flag == 1) {
                        inputs.stomatal_sidedness[u] = 0.f;
                    } else if (flag == 2) {
                        inputs.stomatal_sidedness[u] = 0.5f;
                    }
                }

                // Boundary-layer conductance to heat
                if (boundarylayer_conductance_given && context->doesPrimitiveDataExist(p, "boundarylayer_conductance")) {
                    context->getPrimitiveData(p, "boundarylayer_conductance", inputs.gH[u]);
                } else {
                    // Wind speed
                    float U;
                    if (wind_speed_given && context->doesPrimitiveDataExist(p, "wind_speed")) {
                        context->getPrimitiveData(p, "wind_speed", U);
                    } else {
                        U = wind_speed_default;
                        missing_wind_speed_count++;
                    }

                    if (canopy_airspace_ground_set.count(p) > 0) {
                        // The flat-plate relation below scales with the size of the primitive, which is arbitrary for a tiled soil surface. Ground primitives beneath the canopy airspace instead use the bare-soil conductance of Kustas and Norman
                        // (1999), evaluated at the within-canopy wind speed near the soil that the airspace model writes to 'wind_speed'.
                        inputs.gH[u] = calculateGroundBoundaryLayerConductance(U);
                    } else {
                        // Characteristic size of primitive
                        float L;
                        uint objID = context->getPrimitiveParentObjectID(p);
                        if (object_length_given && context->doesPrimitiveDataExist(p, "object_length")) {
                            context->getPrimitiveData(p, "object_length", L);
                            if (L == 0) {
                                L = sqrt(context->getPrimitiveArea(p));
                                primitive_length_used = true;
                            }
                        } else if (objID > 0) {
                            auto cached = object_length_cache.find(objID);
                            if (cached != object_length_cache.end()) {
                                L = cached->second;
                            } else {
                                L = sqrt(context->getObjectArea(objID));
                                object_length_cache.emplace(objID, L);
                            }
                        } else {
                            L = sqrt(context->getPrimitiveArea(p));
                            primitive_length_used = true;
                        }

                        inputs.gH[u] = 0.135f * sqrt(U / L) * float(inputs.Nsides[u]);
                    }
                    calculated_blconductance_used = true;
                }

                // Moisture conductance
                if (moisture_conductance_given && context->doesPrimitiveDataExist(p, "moisture_conductance")) {
                    context->getPrimitiveData(p, "moisture_conductance", inputs.gS[u]);
                } else {
                    inputs.gS[u] = gS_default;
                }

                // Other fluxes
                if (other_surface_flux_given && context->doesPrimitiveDataExist(p, "other_surface_flux")) {
                    context->getPrimitiveData(p, "other_surface_flux", inputs.Qother[u]);
                } else {
                    inputs.Qother[u] = Qother_default;
                }

                // Object heat capacity
                if (heat_capacity_given && context->doesPrimitiveDataExist(p, "heat_capacity")) {
                    context->getPrimitiveData(p, "heat_capacity", inputs.heatcapacity[u]);
                } else {
                    inputs.heatcapacity[u] = heatcapacity_default;
                }

                // Surface humidity
                if (surface_humidity_given && context->doesPrimitiveDataExist(p, "surface_humidity")) {
                    context->getPrimitiveData(p, "surface_humidity", inputs.surfacehumidity[u]);
                } else {
                    inputs.surfacehumidity[u] = surface_humidity_default;
                }

                // Emissivity for the emitted thermal radiation term, which multiplies sigma*T^4. If no band emits, the surface is taken as a blackbody.
                inputs.eps[u] = 1.f;
                if (emission_info_available) {
                    if (!emission_bands.empty()) {
                        // Effective emissivity summed over the emission bands, sum_b(eps_b*F_b(T)), with F_b = 1 for a band without wavelength bounds. The blackbody fractions are evaluated at the initial surface temperature,
                        // the temperature at which the RadiationModel computed the emission that the absorbed fluxes contain.
                        float effective_emissivity = 0.f;
                        float blackbody_fraction_sum = 0.f;
                        bool emissivity_missing = false;
                        for (const auto &emission_band: emission_bands) {
                            // A band that emits as a blackbody keeps an emissivity of 1; the RadiationModel already warns that its radiative properties are ignored without scattering
                            float emissivity = 1.f;
                            if (!emission_band.emits_as_blackbody) {
                                if (emission_band.emissivity_given && context->doesPrimitiveDataExist(p, emission_band.emissivity_label.c_str())) {
                                    context->getPrimitiveData(p, emission_band.emissivity_label.c_str(), emissivity);
                                } else {
                                    emissivity_missing = true;
                                }
                            }
                            float blackbody_fraction = 1.f;
                            if (emission_band.has_wavelength_bounds) {
                                if (!std::isfinite(inputs.To[u]) || inputs.To[u] <= 0.f) {
                                    helios_runtime_error("ERROR (EnergyBalanceModel::run): Primitive " + std::to_string(p) + " has a surface temperature of " + std::to_string(inputs.To[u]) +
                                                         " K in primitive data 'temperature', which is not a valid absolute temperature, so the share of the blackbody spectrum emitted in radiation band '" + emission_band.label +
                                                         "' cannot be computed. Surface temperatures must be given in Kelvin.");
                                }
                                blackbody_fraction = blackbodyBandFraction(emission_band.wavelength_bounds_nm.x, emission_band.wavelength_bounds_nm.y, inputs.To[u]);
                            }
                            effective_emissivity += emissivity * blackbody_fraction;
                            blackbody_fraction_sum += blackbody_fraction;
                        }
                        if (blackbody_fraction_sum < min_blackbody_fraction_sum || blackbody_fraction_sum > max_blackbody_fraction_sum) {
                            std::string band_list;
                            for (const auto &emission_band: emission_bands) {
                                band_list += (band_list.empty() ? "'" : ", '") + emission_band.label + "'";
                                if (emission_band.has_wavelength_bounds) {
                                    band_list += " (" + std::to_string(emission_band.wavelength_bounds_nm.x) + "-" + std::to_string(emission_band.wavelength_bounds_nm.y) + " nm)";
                                } else {
                                    band_list += " (no wavelength bounds, i.e. broadband)";
                                }
                            }
                            helios_runtime_error(
                                    "ERROR (EnergyBalanceModel::run): The radiation bands with emission enabled, " + band_list + ", together emit " + std::to_string(blackbody_fraction_sum) + " times the blackbody emission sigma*T^4 of primitive " +
                                    std::to_string(p) + " at " + std::to_string(inputs.To[u]) + " K. The energy balance emits only what these bands emit in the RadiationModel, so they must account for the blackbody spectrum once (a sum between " +
                                    std::to_string(min_blackbody_fraction_sum) + " and " + std::to_string(max_blackbody_fraction_sum) +
                                    "). Use a single longwave band without wavelength bounds, or bounded bands that together span the thermal spectrum (e.g. 3-100 um). A thermal camera band such as 8-14 um should not be added to the energy balance model "
                                    "alongside the longwave band, nor used as its only longwave band. A shortwave band added to the energy balance model should have its emission disabled with RadiationModel::disableEmission().");
                        }
                        inputs.eps[u] = effective_emissivity;
                        if (emissivity_missing) {
                            missing_emissivity_count++;
                        }
                    }
                } else {
                    // Without emission-enabled information, use the first band that defines an emissivity for this primitive.
                    int emissivity_defined_count = 0;
                    for (const auto &emissivity_label: emissivity_labels) {
                        if (context->doesPrimitiveDataExist(p, emissivity_label.c_str())) {
                            if (emissivity_defined_count == 0) {
                                context->getPrimitiveData(p, emissivity_label.c_str(), inputs.eps[u]);
                            }
                            emissivity_defined_count++;
                        }
                    }
                    if (emissivity_defined_count == 0) {
                        missing_emissivity_count++;
                    } else if (emissivity_defined_count > 1) {
                        multiple_emissivity_bands_count++;
                    }
                }
            } catch (...) {
#ifdef USE_OPENMP
#pragma omp critical(energybalance_gather_exception)
#endif
                {
                    if (!gather_exception) {
                        gather_exception = std::current_exception();
                    }
                }
            }
        }
    }

    if (gather_exception) {
        std::rethrow_exception(gather_exception);
    }

    if (warnings.isEnabled()) {
        const std::vector<std::tuple<const char *, std::string, int>> counted_warnings = {
                {"missing_surface_temperature", "Primitive data 'temperature' not set, using default (" + std::to_string(temperature_default) + " K)", missing_surface_temperature_count},
                {"missing_air_temperature", "Primitive data 'air_temperature' not set, using default (" + std::to_string(air_temperature_default) + " K)", missing_air_temperature_count},
                {"missing_air_humidity", "Primitive data 'air_humidity' not set, using default (" + std::to_string(air_humidity_default) + ")", missing_air_humidity_count},
                {"missing_wind_speed", "Primitive data 'wind_speed' not set, using default (" + std::to_string(wind_speed_default) + " m/s)", missing_wind_speed_count},
                {"missing_emissivity", "Emissivity not set for the emitting radiation band(s), using default (1.0)", missing_emissivity_count},
                {"multiple_emissivity_bands", "Emissivity was defined for more than one radiation band. Thermal emission uses a single emissivity; the value from the first band will be used.", multiple_emissivity_bands_count}};
        for (const auto &[category, message, count]: counted_warnings) {
            warnings.addWarning(category, message, size_t(count));
        }
    }

    // Report all accumulated warnings
    warnings.report(std::cout, true); // true = compact mode

    // Enable output for calculated boundary-layer conductance if used
    if (calculated_blconductance_used) {
        auto it = find(output_prim_data.begin(), output_prim_data.end(), "boundarylayer_conductance_out");
        if (it == output_prim_data.end()) {
            output_prim_data.emplace_back("boundarylayer_conductance_out");
        }
    }

    // Warning about primitive length usage
    if (message_flag && primitive_length_used) {
        std::cout << "WARNING (EnergyBalanceModel::run): The length of a primitive that is not a member of a compound object "
                  << "was used to calculate the boundary-layer conductance. This often results in incorrect values because "
                  << "the length should be that of the object (e.g., leaf, stem) not the primitive. "
                  << "Make sure this is what you intended." << std::endl;
    }
}

void EnergyBalanceModel::solveSurfaceTemperatures_CPU(const SurfaceEnergyBalanceInputs &inputs, float dt, std::vector<float> &T) {

    const uint Nprimitives = inputs.To.size();

    const std::vector<float> &To = inputs.To;
    const std::vector<float> &R = inputs.R;
    const std::vector<float> &Qother = inputs.Qother;
    const std::vector<float> &eps = inputs.eps;
    const std::vector<float> &Ta = inputs.Ta;
    const std::vector<float> &ea = inputs.ea;
    const std::vector<float> &pressure = inputs.pressure;
    const std::vector<float> &gH = inputs.gH;
    const std::vector<float> &gS = inputs.gS;
    const std::vector<uint> &Nsides = inputs.Nsides;
    const std::vector<float> &stomatal_sidedness = inputs.stomatal_sidedness;
    const std::vector<float> &heatcapacity = inputs.heatcapacity;
    const std::vector<float> &surfacehumidity = inputs.surfacehumidity;

// Issue warning if OpenMP not available (use aggregator to avoid spam)
#ifndef USE_OPENMP
    static bool openmp_warning_issued = false;
    if (message_flag && !openmp_warning_issued) {
        std::cout << "WARNING (EnergyBalanceModel): OpenMP not available. Using serial CPU implementation. "
                  << "Performance will be significantly slower. Consider installing OpenMP for parallel execution." << std::endl;
        openmp_warning_issued = true;
    }
#endif

    // Allocate result array
    T.resize(Nprimitives);

// Thread-local storage for convergence warnings
#ifdef USE_OPENMP
    int num_threads = omp_get_max_threads();
    std::vector<int> thread_convergence_failures(num_threads, 0);
#else
    int convergence_failures = 0;
#endif

// Parallel loop over primitives
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int p = 0; p < (int) Nprimitives; p++) {

        // Secant method parameters
        const float err_max = 0.0001f;
        const uint max_iter = 100;

        // Initial guesses. The second guess must differ from the first, otherwise the two residuals are
        // identical and the secant update below divides by zero (this happens when the supplied initial
        // temperature equals the hardcoded second guess of 400 K).
        float T_old = To[p];
        float T_old_old = 400.f;
        if (T_old_old == T_old) {
            T_old_old = T_old + 100.f;
        }

        // Evaluate residual at initial guesses
        float resid_old = evaluateEnergyBalance_CPU(T_old, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p]);
        float resid_old_old = evaluateEnergyBalance_CPU(T_old_old, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p]);

        // Secant method iteration. Initialize the result to the current best guess so that a degenerate
        // early exit (identical residuals) never leaves the surface temperature uninitialized.
        float T_new = T_old;
        float resid = 100;
        float err = resid;
        uint iter = 0;
        while (err > err_max && iter < max_iter) {

            if (resid_old == resid_old_old) {
                err = 0;
                break;
            }

            T_new = fabs((T_old_old * resid_old - T_old * resid_old_old) / (resid_old - resid_old_old));

            resid = evaluateEnergyBalance_CPU(T_new, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p]);

            // Relative change of the newest secant step (measures the latest iterate, not the previous one)
            err = fabs(T_new - T_old) / fabs(T_old);

            resid_old_old = resid_old;
            resid_old = resid;

            T_old_old = T_old;
            T_old = T_new;

            iter++;
        }

        // Track convergence failures (thread-safe)
        if (err > err_max) {
#ifdef USE_OPENMP
            int thread_id = omp_get_thread_num();
            thread_convergence_failures[thread_id]++;
#else
            convergence_failures++;
#endif
        }

        T[p] = T_new;
    }

// Report convergence warnings
#ifdef USE_OPENMP
    int total_failures = 0;
    for (int i = 0; i < num_threads; i++) {
        total_failures += thread_convergence_failures[i];
    }
    if (total_failures > 0 && message_flag) {
        std::cout << "WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge for " << total_failures << " primitives." << std::endl;
    }
#else
    if (convergence_failures > 0 && message_flag) {
        std::cout << "WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge for " << convergence_failures << " primitives." << std::endl;
    }
#endif
}

void EnergyBalanceModel::writeSurfaceEnergyBalanceOutputs(const std::vector<uint> &UUIDs, const SurfaceEnergyBalanceInputs &inputs, const std::vector<float> &T, float dt) {

    const size_t Nprimitives = UUIDs.size();

    // Outputs are assembled into vectors and written with one bulk Context call per label, which validates the label once rather than once per primitive.
    std::vector<float> sensible_flux(Nprimitives);
    std::vector<float> latent_flux(Nprimitives);
    std::vector<float> surface_esat(Nprimitives);

    for (size_t u = 0; u < Nprimitives; u++) {
        sensible_flux[u] = cp_air_mol * inputs.gH[u] * (T[u] - inputs.Ta[u]);

        surface_esat[u] = esat_Pa(T[u]);
        float gM = 0.f;
        if (inputs.gH[u] != 0.f && inputs.gS[u] != 0.f) {
            float gbv = 1.08f * inputs.gH[u] / float(inputs.Nsides[u]); // per-face boundary-layer conductance to vapor
            gM = gbv * inputs.gS[u] * (inputs.stomatal_sidedness[u] / (gbv + inputs.gS[u] * inputs.stomatal_sidedness[u]) + (1.f - inputs.stomatal_sidedness[u]) / (gbv + inputs.gS[u] * (1.f - inputs.stomatal_sidedness[u])));
        }
        latent_flux[u] = lambda_mol * gM * (surface_esat[u] * inputs.surfacehumidity[u] - inputs.ea[u]) / inputs.pressure[u];
    }

    context->setPrimitiveData(UUIDs, "temperature", T);
    context->setPrimitiveData(UUIDs, "sensible_flux", sensible_flux);
    context->setPrimitiveData(UUIDs, "latent_flux", latent_flux);

    std::vector<float> optional_output(Nprimitives);
    for (const auto &data_label: output_prim_data) {
        if (data_label == "boundarylayer_conductance_out") {
            context->setPrimitiveData(UUIDs, "boundarylayer_conductance_out", inputs.gH);
        } else if (data_label == "vapor_pressure_deficit") {
            for (size_t u = 0; u < Nprimitives; u++) {
                optional_output[u] = (surface_esat[u] - inputs.ea[u]) / inputs.pressure[u];
            }
            context->setPrimitiveData(UUIDs, "vapor_pressure_deficit", optional_output);
        } else if (data_label == "net_radiation_flux") {
            for (size_t u = 0; u < Nprimitives; u++) {
                optional_output[u] = inputs.R[u] - float(inputs.Nsides[u]) * inputs.eps[u] * 5.67e-8F * std::pow(T[u], 4);
            }
            context->setPrimitiveData(UUIDs, "net_radiation_flux", optional_output);
        } else if (data_label == "storage_flux") {
            for (size_t u = 0; u < Nprimitives; u++) {
                optional_output[u] = 0.f;
                if (dt > 0) {
                    optional_output[u] = inputs.heatcapacity[u] * (T[u] - inputs.To[u]) / dt;
                }
            }
            context->setPrimitiveData(UUIDs, "storage_flux", optional_output);
        }
    }
}
