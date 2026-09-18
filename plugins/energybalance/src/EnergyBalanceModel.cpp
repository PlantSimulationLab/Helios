/** \file "EnergyBalanceModel.cpp" Energy balance model plugin declarations.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "EnergyBalanceModel.h"
#include <iomanip>
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
#ifdef HELIOS_CUDA_AVAILABLE
    // Santanello-Friedl G is an algebraic function of the CURRENT trial soil
    // temperature through Rn_s(Ts).  The CPU residual below evaluates G(Ts)
    // inside each root-function evaluation.  Until the CUDA residual is patched
    // with the same term, canopy-airspace solves that include ground must use
    // the CPU residual to remain physically identical.
    if (canopy_airspace_enabled && !canopy_airspace_ground_UUIDs.empty()) {
        evaluateSurfaceEnergyBalance_CPU(UUIDs, dt);
    } else if (gpu_acceleration_enabled) {
        evaluateSurfaceEnergyBalance_GPU(UUIDs, dt);
    } else {
        evaluateSurfaceEnergyBalance_CPU(UUIDs, dt);
    }
#else
    evaluateSurfaceEnergyBalance_CPU(UUIDs, dt);
#endif
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
        helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Reference wind speed must be greater than zero.");
    }

    float Patm = pressure_default;
    if (context->doesGlobalDataExist("air_pressure") && context->getGlobalDataType("air_pressure") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("air_pressure", Patm);
    }

    for (uint UUID: canopy_airspace_UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Canopy primitive UUID " + std::to_string(UUID) + " no longer exists.");
        }
    }
    for (uint UUID: canopy_airspace_ground_UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Ground primitive UUID " + std::to_string(UUID) + " no longer exists.");
        }
    }

    if (!canopy_airspace_layers_assigned) {
        assignCanopyAirspaceLayers();
    }

    const size_t Ncanopy = canopy_airspace_UUIDs.size();
    const uint Nlayers = canopy_airspace_layer_count;
    const bool soil_node_present = !canopy_airspace_ground_UUIDs.empty();

    // -- Aerodynamic / within-canopy conductances -- //
    const float rho_air_mol_m3 = Patm / (R * air_temperature_reference);
    const float aerodynamic_resistance_s_m = calculateAerodynamicResistance(wind_speed_reference);
    const float g_a = rho_air_mol_m3 / aerodynamic_resistance_s_m;

    // ---------------------------------------------------------------------
    // TSEB-neutral aerodynamic geometry.
    //
    // Match the pyTSEB configuration used by the manuscript baseline:
    //     d0  = 0.65 * h_c
    //     z0M = 0.125 * h_c
    //
    // With kB = 0, z0H = z0M.  The same d0/z0M geometry is used for
    // aerodynamic resistance and for extrapolating measured reference wind
    // down to canopy-top wind.
    // ---------------------------------------------------------------------
    const float displacement_height = 0.65f * canopy_height_m;
    const float zo_m = 0.125f * canopy_height_m;

    if (!(zo_m > 0.f) ||
        !(reference_height_m > displacement_height) ||
        !(canopy_height_m > displacement_height)) {
        helios_runtime_error(
            "ERROR (EnergyBalanceModel::evaluateCanopyAirspace): "
            "Invalid TSEB aerodynamic geometry. Require h_c > d0, "
            "z_ref > d0, and z0M > 0.");
    }

    const float log_wind_reference =
        std::log((reference_height_m - displacement_height) / zo_m);
    const float log_wind_canopy_top =
        std::log((canopy_height_m - displacement_height) / zo_m);

    if (!(log_wind_reference > 0.f) || !(log_wind_canopy_top > 0.f)) {
        helios_runtime_error(
            "ERROR (EnergyBalanceModel::evaluateCanopyAirspace): "
            "Invalid logarithmic wind-profile geometry for TSEB d0/z0M.");
    }

    float wind_speed_canopy_top =
        wind_speed_reference *
        log_wind_canopy_top /
        log_wind_reference;

    wind_speed_canopy_top = std::fmax(wind_speed_canopy_top, 0.01f);

    const float friction_velocity =
        von_Karman_constant * wind_speed_reference /
        log_wind_reference;

    const float eddy_diffusivity_canopy_top =
        von_Karman_constant *
        friction_velocity *
        (canopy_height_m - displacement_height);

    // Export the exact TSEB aerodynamic geometry used by the airspace model.
    context->setGlobalData("canopy_displacement_height", displacement_height);
    context->setGlobalData("canopy_roughness_length_momentum", zo_m);

    // ---------------------------------------------------------------------
    // Within-canopy wind attenuation.
    //
    // canopy_wind_model = 0 : original Helios/Cionco
    //                         a = 0.5 * LAI
    //
    // canopy_wind_model = 1 : Goudriaan formulation used by TSEB
    //                         a = 0.28 * LAI^(2/3)
    //                                   * h_c^(1/3)
    //                                   * leaf_width^(-1/3)
    //
    // main.cpp sets canopy_wind_model and canopy_leaf_width_m.
    // ---------------------------------------------------------------------
    uint canopy_wind_model = 0;
    if (context->doesGlobalDataExist("canopy_wind_model") &&
        context->getGlobalDataType("canopy_wind_model") == helios::HELIOS_TYPE_UINT) {
        context->getGlobalData("canopy_wind_model", canopy_wind_model);
    }

    float attenuation_coefficient = 0.5f * canopy_airspace_LAI;

    if (canopy_wind_model == 1) {
        float canopy_leaf_width_m = NAN;

        if (!context->doesGlobalDataExist("canopy_leaf_width_m") ||
            context->getGlobalDataType("canopy_leaf_width_m") != helios::HELIOS_TYPE_FLOAT) {
            helios_runtime_error(
                "ERROR (EnergyBalanceModel::evaluateCanopyAirspace): "
                "Goudriaan wind model requested, but canopy_leaf_width_m "
                "is missing or is not a float.");
        }

        context->getGlobalData("canopy_leaf_width_m", canopy_leaf_width_m);

        if (!(canopy_leaf_width_m > 0.f) || !std::isfinite(canopy_leaf_width_m)) {
            helios_runtime_error(
                "ERROR (EnergyBalanceModel::evaluateCanopyAirspace): "
                "Goudriaan wind model requires canopy_leaf_width_m > 0.");
        }

        attenuation_coefficient =
            0.28f *
            std::pow(canopy_airspace_LAI, 2.f / 3.f) *
            std::pow(canopy_height_m, 1.f / 3.f) *
            std::pow(canopy_leaf_width_m, -1.f / 3.f);
    } else if (canopy_wind_model != 0) {
        helios_runtime_error(
            "ERROR (EnergyBalanceModel::evaluateCanopyAirspace): "
            "canopy_wind_model must be 0 (Cionco) or 1 (Goudriaan).");
    }

    // Export the exact attenuation coefficient used by this calculation.
    context->setGlobalData("canopy_wind_attenuation_coefficient",
                           attenuation_coefficient);

    std::vector<float> g_layer(Nlayers, 0.f);
    for (uint j = 0; j + 1 < Nlayers; j++) {
        const float boundary_height = canopy_airspace_layer_top_m.at(j);
        const float separation = canopy_airspace_layer_midpoint_m.at(j + 1) - canopy_airspace_layer_midpoint_m.at(j);
        if (separation <= 0.f) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Coincident airspace layer midpoints.");
        }
        const float eddy_diffusivity = eddy_diffusivity_canopy_top *
                                       std::exp(-attenuation_coefficient * (1.f - boundary_height / canopy_height_m));
        g_layer.at(j) = rho_air_mol_m3 * eddy_diffusivity / separation;
    }

    const float wind_speed_soil = wind_speed_canopy_top * std::exp(-attenuation_coefficient);
    constexpr float moisture_conductance_ratio = 1.08f;

    // Assign the within-canopy wind speed to each canopy primitive.
    // updateCanopyWindProfile() writes primitive data "wind_speed".
    updateCanopyWindProfile(wind_speed_canopy_top);

    // ---------------------------------------------------------------------
    // Export wind diagnostics used by this exact EnergyBalance calculation.
    //
    // canopy_top_wind_speed : neutral-profile wind at canopy top
    // soil_wind_speed       : attenuated wind used in the KN99 soil resistance
    // canopy_wind_speed     : canopy-area-weighted mean of primitive wind speeds
    // ---------------------------------------------------------------------
    double canopy_wind_area_sum = 0.0;
    double canopy_area_sum = 0.0;

    for (uint UUID : canopy_airspace_UUIDs) {
        float primitive_wind_speed = NAN;
        context->getPrimitiveData(UUID, "wind_speed", primitive_wind_speed);

        const float primitive_area = context->getPrimitiveArea(UUID);

        if (std::isfinite(primitive_wind_speed) && primitive_area > 0.f) {
            canopy_wind_area_sum +=
                static_cast<double>(primitive_wind_speed) *
                static_cast<double>(primitive_area);
            canopy_area_sum += static_cast<double>(primitive_area);
        }
    }

    const float canopy_wind_speed =
        (canopy_area_sum > 0.0)
            ? static_cast<float>(canopy_wind_area_sum / canopy_area_sum)
            : NAN;

    context->setGlobalData("canopy_wind_speed", canopy_wind_speed);
    context->setGlobalData("canopy_top_wind_speed", wind_speed_canopy_top);
    context->setGlobalData("soil_wind_speed", wind_speed_soil);

    const float moisture_reference = air_humidity_reference * esat_Pa(air_temperature_reference) / Patm;

    // State variables of the coupled airspace problem.
    //
    // Thermal state:
    //   T_ac  -> the TSEB-like canopy-air node temperature.
    //
    // Moisture state:
    //   x_ac  -> canopy-air water-vapor mole fraction.
    //
    // These are the ONLY variables used to determine convergence of the
    // outer fixed-point iteration. H and LE are diagnosed afterwards from
    // the converged state; they are not independently forced.
    std::vector<float> T_ac(Nlayers, air_temperature_reference);
    std::vector<float> x_ac(Nlayers, moisture_reference);

    // Warm-start temperature from the previous call when available.
    if (context->doesGlobalDataExist("canopy_air_temperature_layers") &&
        context->getGlobalDataType("canopy_air_temperature_layers") == helios::HELIOS_TYPE_FLOAT) {
        std::vector<float> T_previous_layers;
        context->getGlobalData("canopy_air_temperature_layers", T_previous_layers);
        if (T_previous_layers.size() == Nlayers) {
            T_ac = T_previous_layers;
        }
    }

    // Warm-start moisture from the previous layer relative humidity using the
    // current layer temperature.  Previously x_ac was reset to the above-canopy
    // reference every call, which needlessly increased the number of iterations.
    if (context->doesGlobalDataExist("canopy_air_humidity_layers") &&
        context->getGlobalDataType("canopy_air_humidity_layers") == helios::HELIOS_TYPE_FLOAT) {
        std::vector<float> RH_previous_layers;
        context->getGlobalData("canopy_air_humidity_layers", RH_previous_layers);
        if (RH_previous_layers.size() == Nlayers) {
            for (uint j = 0; j < Nlayers; j++) {
                const float RH = std::fmin(std::fmax(RH_previous_layers.at(j), 0.f), 1.f);
                x_ac.at(j) = RH * esat_Pa(T_ac.at(j)) / Patm;
            }
        }
    }

    // Ground-area basis used by every canopy/soil reduction below.
    const float total_canopy_area = context->sumPrimitiveSurfaceArea(canopy_airspace_UUIDs);
    const float ground_area = total_canopy_area / canopy_airspace_LAI;
    if (!(ground_area > 0.f)) {
        helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Ground area inferred from canopy area/LAI is not positive.");
    }

    auto write_air_state_to_primitives = [&](const std::vector<float>& Tair, const std::vector<float>& xair) {
        for (size_t u = 0; u < Ncanopy; u++) {
            const uint layer = canopy_airspace_layer_index.at(u);
            const float RH = std::fmin(std::fmax(xair.at(layer) * Patm / esat_Pa(Tair.at(layer)), 0.f), 1.f);
            context->setPrimitiveData(canopy_airspace_UUIDs.at(u), "air_temperature", Tair.at(layer));
            context->setPrimitiveData(canopy_airspace_UUIDs.at(u), "air_humidity", RH);
            // Keep the primitive surface-EB vapor calculation on the same
            // pressure basis as the airspace moisture network.
            context->setPrimitiveData(canopy_airspace_UUIDs.at(u), "air_pressure", Patm);
        }
        if (soil_node_present) {
            const float RHs = std::fmin(std::fmax(xair.front() * Patm / esat_Pa(Tair.front()), 0.f), 1.f);
            context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_temperature", Tair.front());
            context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_humidity", RHs);
            context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_pressure", Patm);
            context->setPrimitiveData(canopy_airspace_ground_UUIDs, "wind_speed", wind_speed_soil);
        }
    };

    // Make sure the very first surface solve sees the initialized airspace state.
    write_air_state_to_primitives(T_ac, x_ac);

    // Ground heat flux G is not an externally fixed forcing.  For airspace-ground
    // primitives it is evaluated directly inside the soil surface-energy-balance
    // residual from the CURRENT trial soil net radiation:
    //
    //   G(Ts) = A_SF * Rn_s(Ts) * cos[2*pi*(seconds_from_noon + 10800)/74000].
    //
    // Therefore G converges automatically with Ts and the coupled airspace state;
    // there is no separate G fixed-point update in this outer loop.

    // Soil sensible-heat resistance: use the same Kustas & Norman (1999) formulation
    // used by TSEB, and force that SAME conductance into the ground primitive surface EB.
    //
    //   R_s = 1 / [0.0038 * max(T_s - T_ac, 0)^(1/3) + 0.012 * U_s]   [s m-1]
    //   g_H,s = rho_mol / R_s                                             [mol m-2 s-1]
    //
    // A single bulk R_s is evaluated from the area-weighted soil temperature and the
    // lowest canopy-air temperature. Every ground primitive then receives this same g_H,s.
    // At convergence, both the primitive ground EB and the airspace resistance network
    // therefore use exactly the same Kustas-Norman soil resistance.
    float soil_resistance_KN_s_m = NAN;
    float soil_heat_conductance_KN_mol_m2_s = NAN;
    float soil_temperature_KN_K = NAN;

    auto update_KN_soil_conductance = [&](float Tac_K) {
        if (!soil_node_present) {
            soil_resistance_KN_s_m = NAN;
            soil_heat_conductance_KN_mol_m2_s = NAN;
            soil_temperature_KN_K = NAN;
            return;
        }

        float soil_T_area_sum = 0.f;
        const float soil_area = context->sumPrimitiveSurfaceArea(canopy_airspace_ground_UUIDs);
        if (!(soil_area > 0.f)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Ground primitive area is not positive.");
        }
        // On the very first canopy-airspace iteration, ground primitives may not yet
        // have a solved surface temperature.  Use Tac as the initial bulk-soil
        // temperature for the KN99 resistance; subsequent iterations use the solved Ts.
        bool ground_temperature_available = false;
        for (uint UUID : canopy_airspace_ground_UUIDs) {
            if (context->doesPrimitiveDataExist(UUID, "temperature") &&
                context->getPrimitiveDataType("temperature") == helios::HELIOS_TYPE_FLOAT) {
                ground_temperature_available = true;
                break;
            }
        }

        if (ground_temperature_available) {
            context->calculatePrimitiveDataAreaWeightedSum(canopy_airspace_ground_UUIDs,
                                                           "temperature",
                                                           soil_T_area_sum);
            soil_temperature_KN_K = soil_T_area_sum / soil_area;
        } else {
            soil_temperature_KN_K = Tac_K;
        }

        const float dT_positive_K = std::fmax(soil_temperature_KN_K - Tac_K, 0.f);
        const float conductance_velocity_m_s =
            0.0038f * std::cbrt(dT_positive_K) + 0.012f * wind_speed_soil;

        if (!(conductance_velocity_m_s > 0.f)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Kustas-Norman soil heat conductance is not positive.");
        }

        soil_resistance_KN_s_m = 1.f / conductance_velocity_m_s;
        soil_heat_conductance_KN_mol_m2_s = rho_air_mol_m3 * conductance_velocity_m_s;

        // evaluateSurfaceEnergyBalance() gives primitive data 'boundarylayer_conductance'
        // precedence over its internally calculated plate boundary-layer conductance.
        // Thus the primitive sensible flux and the airspace network use the same R_s.
        context->setPrimitiveData(canopy_airspace_ground_UUIDs,
                                  "boundarylayer_conductance",
                                  soil_heat_conductance_KN_mol_m2_s);
    };

    std::vector<float> sub(Nlayers), diag(Nlayers), super(Nlayers), rhs(Nlayers);
    std::vector<float> T_ac_previous(Nlayers), x_ac_previous(Nlayers);

    // Diagnostics retained from the final converged iteration.
    float Hc_primitive = NAN, Hs_primitive = NAN, LEc_primitive = NAN, LEs_primitive = NAN;
    float Hc_network = NAN, Hs_network = NAN, H_atmosphere = NAN;
    float LEc_network = NAN, LEs_network = NAN, LE_atmosphere = NAN;
    float Hc_sync = NAN, Hs_sync = NAN, LEc_sync = NAN, LEs_sync = NAN;
    float sensible_closure = NAN, latent_closure = NAN;
    float Tc_effective = NAN, Ts_effective = NAN;
    float Gc_effective = NAN, Gs_effective = NAN;

    const float flux_tolerance_W_m2 = 0.1f;
    uint iteration = 0;
    bool converged = false;

    // Picard under-relaxation.  The coupled mapping
    //   (Tac,xac) -> surface EB -> resistance network -> (Tac,xac)
    // is strongly nonlinear because Ts changes R_s and both Ts/Tc change
    // vapor pressure.  A full (relaxation=1) update can oscillate indefinitely.
    // Under-relaxation changes only the numerical path, not the converged
    // physical fixed point.
    constexpr float airspace_relaxation = 0.5f;

    while (iteration < canopy_airspace_max_iterations && !converged) {

        T_ac_previous = T_ac;
        x_ac_previous = x_ac;

        // One surface-energy-balance solve per outer iteration. Canopy AND ground primitives
        // therefore use exactly the air state from T_ac_previous/x_ac_previous.
        // Update the Kustas-Norman soil resistance first because R_s depends on T_s - T_ac.
        // The T_s used here is the current (previous-iteration) ground surface temperature;
        // the fixed-point iteration updates it until T_s, T_ac and R_s are mutually consistent.
        update_KN_soil_conductance(T_ac_previous.front());
        evaluateSurfaceEnergyBalance(UUIDs, 0.f);

        std::vector<double> sum_gA_T(Nlayers, 0.0);
        std::vector<double> sum_gA(Nlayers, 0.0);
        std::vector<double> sum_gA_dT_old(Nlayers, 0.0);
        std::vector<double> sum_gA_dT_new(Nlayers, 0.0);
        std::vector<float> sum_gM_xs(Nlayers, 0.f);
        std::vector<float> sum_gM(Nlayers, 0.f);

        Hc_primitive = 0.f;
        LEc_primitive = 0.f;

        // -- Exact canopy reduction of the primitive sensible and latent formulations -- //
        for (size_t u = 0; u < Ncanopy; u++) {
            const uint UUID = canopy_airspace_UUIDs.at(u);
            const uint layer = canopy_airspace_layer_index.at(u);
            const float area_fraction = context->getPrimitiveArea(UUID) / ground_area;

            float T_surface;
            context->getPrimitiveData(UUID, "temperature", T_surface);

            float gH;
            if (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out")) {
                helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Missing boundarylayer_conductance_out on canopy primitive.");
            }
            context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);

            const double gA =
                static_cast<double>(gH) * static_cast<double>(area_fraction);

            sum_gA.at(layer) += gA;
            sum_gA_T.at(layer) += gA * static_cast<double>(T_surface);

            // Direct temperature-difference accumulation avoids subtracting
            // two large, nearly equal weighted-temperature sums.
            sum_gA_dT_old.at(layer) +=
                gA * (static_cast<double>(T_surface) -
                      static_cast<double>(T_ac_previous.at(layer)));

            float gS = gS_default;
            if (context->doesPrimitiveDataExist(UUID, "moisture_conductance") && context->getPrimitiveDataType("moisture_conductance") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(UUID, "moisture_conductance", gS);
            }

            float stomatal_sidedness = 0.f;
            const uint twosided_flag = context->getPrimitiveTwosidedFlag(UUID, 1);
            if (twosided_flag != 0) {
                if (context->doesPrimitiveDataExist(UUID, "stomatal_sidedness") && context->getPrimitiveDataType("stomatal_sidedness") == helios::HELIOS_TYPE_FLOAT) {
                    context->getPrimitiveData(UUID, "stomatal_sidedness", stomatal_sidedness);
                } else if (context->doesPrimitiveDataExist(UUID, "evaporating_faces") && context->getPrimitiveDataType("evaporating_faces") == helios::HELIOS_TYPE_UINT) {
                    uint evaporating_faces;
                    context->getPrimitiveData(UUID, "evaporating_faces", evaporating_faces);
                    stomatal_sidedness = (evaporating_faces == 2) ? 0.5f : 0.f;
                }
            }

            float gM = 0.f;
            if (gH != 0.f && gS != 0.f) {
                gM = moisture_conductance_ratio * gH * gS *
                     (stomatal_sidedness / (moisture_conductance_ratio * gH + gS * stomatal_sidedness) +
                      (1.f - stomatal_sidedness) / (moisture_conductance_ratio * gH + gS * (1.f - stomatal_sidedness)));
            }

            float surface_humidity = surface_humidity_default;
            if (context->doesPrimitiveDataExist(UUID, "surface_humidity") && context->getPrimitiveDataType("surface_humidity") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(UUID, "surface_humidity", surface_humidity);
            }
            const float x_surface = surface_humidity * esat_Pa(T_surface) / Patm;
            sum_gM.at(layer) += gM * area_fraction;
            sum_gM_xs.at(layer) += gM * area_fraction * x_surface;

            float qh = 0.f, ql = 0.f;
            context->getPrimitiveData(UUID, "sensible_flux", qh);
            context->getPrimitiveData(UUID, "latent_flux", ql);
            Hc_primitive += qh * context->getPrimitiveArea(UUID) / ground_area;
            LEc_primitive += ql * context->getPrimitiveArea(UUID) / ground_area;
        }

        // -- Exact soil reduction of the SAME primitive conductances used by the surface EB -- //
        double soil_sum_gA = 0.0;
        double soil_sum_gA_T = 0.0;
        double soil_sum_gA_dT_old = 0.0;
        double soil_sum_gA_dT_new = 0.0;
        float soil_sum_gM = 0.f;
        float soil_sum_gM_xs = 0.f;
        Hs_primitive = 0.f;
        LEs_primitive = 0.f;

        if (soil_node_present) {
            for (uint UUID: canopy_airspace_ground_UUIDs) {
                const float area_fraction = context->getPrimitiveArea(UUID) / ground_area;

                float T_surface;
                context->getPrimitiveData(UUID, "temperature", T_surface);

                float gH;
                if (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out")) {
                    helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Missing boundarylayer_conductance_out on ground primitive.");
                }
                context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);
                const double gA =
                    static_cast<double>(gH) * static_cast<double>(area_fraction);

                soil_sum_gA += gA;
                soil_sum_gA_T += gA * static_cast<double>(T_surface);
                soil_sum_gA_dT_old +=
                    gA * (static_cast<double>(T_surface) -
                          static_cast<double>(T_ac_previous.front()));

                float gS = gS_default;
                if (context->doesPrimitiveDataExist(UUID, "moisture_conductance") && context->getPrimitiveDataType("moisture_conductance") == helios::HELIOS_TYPE_FLOAT) {
                    context->getPrimitiveData(UUID, "moisture_conductance", gS);
                }

                float stomatal_sidedness = 0.f;
                const uint twosided_flag = context->getPrimitiveTwosidedFlag(UUID, 1);
                if (twosided_flag != 0) {
                    if (context->doesPrimitiveDataExist(UUID, "stomatal_sidedness") && context->getPrimitiveDataType("stomatal_sidedness") == helios::HELIOS_TYPE_FLOAT) {
                        context->getPrimitiveData(UUID, "stomatal_sidedness", stomatal_sidedness);
                    } else if (context->doesPrimitiveDataExist(UUID, "evaporating_faces") && context->getPrimitiveDataType("evaporating_faces") == helios::HELIOS_TYPE_UINT) {
                        uint evaporating_faces;
                        context->getPrimitiveData(UUID, "evaporating_faces", evaporating_faces);
                        stomatal_sidedness = (evaporating_faces == 2) ? 0.5f : 0.f;
                    }
                }

                float gM = 0.f;
                if (gH != 0.f && gS != 0.f) {
                    gM = moisture_conductance_ratio * gH * gS *
                         (stomatal_sidedness / (moisture_conductance_ratio * gH + gS * stomatal_sidedness) +
                          (1.f - stomatal_sidedness) / (moisture_conductance_ratio * gH + gS * (1.f - stomatal_sidedness)));
                }

                float surface_humidity = surface_humidity_default;
                if (context->doesPrimitiveDataExist(UUID, "surface_humidity") && context->getPrimitiveDataType("surface_humidity") == helios::HELIOS_TYPE_FLOAT) {
                    context->getPrimitiveData(UUID, "surface_humidity", surface_humidity);
                }
                const float x_surface = surface_humidity * esat_Pa(T_surface) / Patm;
                soil_sum_gM += gM * area_fraction;
                soil_sum_gM_xs += gM * area_fraction * x_surface;

                float qh = 0.f, ql = 0.f;
                context->getPrimitiveData(UUID, "sensible_flux", qh);
                context->getPrimitiveData(UUID, "latent_flux", ql);
                Hs_primitive += qh * context->getPrimitiveArea(UUID) / ground_area;
                LEs_primitive += ql * context->getPrimitiveArea(UUID) / ground_area;
            }
        }

        // Solve temperature network. The soil link uses the exact conductance-weighted
        // primitive reduction, not a separate empirical bulk g_soil.
        for (uint j = 0; j < Nlayers; j++) {
            const float g_below =
                (j == 0)
                    ? (soil_node_present ? static_cast<float>(soil_sum_gA) : 0.f)
                    : g_layer.at(j - 1);
            const float g_above = (j == Nlayers - 1) ? g_a : g_layer.at(j);
            sub.at(j) = (j == 0) ? 0.f : -g_layer.at(j - 1);
            super.at(j) = (j == Nlayers - 1) ? 0.f : -g_layer.at(j);
            diag.at(j) =
                static_cast<float>(sum_gA.at(j)) + g_below + g_above;
            rhs.at(j) = static_cast<float>(sum_gA_T.at(j));
            if (j == 0 && soil_node_present) {
                rhs.at(j) += static_cast<float>(soil_sum_gA_T);
            }
            if (j == Nlayers - 1) {
                rhs.at(j) += g_a * air_temperature_reference;
            }
        }
        std::vector<float> T_ac_candidate;
        solveTridiagonal(sub, diag, super, rhs, T_ac_candidate);

        // Under-relax the nonlinear thermal fixed-point update.
        for (uint j = 0; j < Nlayers; j++) {
            T_ac.at(j) = T_ac_previous.at(j) +
                         airspace_relaxation *
                         (T_ac_candidate.at(j) - T_ac_previous.at(j));
        }

        // Solve vapor network using the exact primitive gM and surface_humidity formulation.
        for (uint j = 0; j < Nlayers; j++) {
            const float g_below = (j == 0) ? (soil_node_present ? soil_sum_gM : 0.f) : g_layer.at(j - 1);
            const float g_above = (j == Nlayers - 1) ? g_a : g_layer.at(j);
            sub.at(j) = (j == 0) ? 0.f : -g_layer.at(j - 1);
            super.at(j) = (j == Nlayers - 1) ? 0.f : -g_layer.at(j);
            diag.at(j) = sum_gM.at(j) + g_below + g_above;
            rhs.at(j) = sum_gM_xs.at(j);
            if (j == 0 && soil_node_present) {
                rhs.at(j) += soil_sum_gM_xs;
            }
            if (j == Nlayers - 1) {
                rhs.at(j) += g_a * moisture_reference;
            }
        }
        std::vector<float> x_ac_candidate;
        solveTridiagonal(sub, diag, super, rhs, x_ac_candidate);

        // Under-relax the nonlinear moisture fixed-point update, then enforce
        // physical bounds using the relaxed air temperature.
        for (uint j = 0; j < Nlayers; j++) {
            x_ac.at(j) = x_ac_previous.at(j) +
                         airspace_relaxation *
                         (x_ac_candidate.at(j) - x_ac_previous.at(j));
            const float x_sat = esat_Pa(T_ac.at(j)) / Patm;
            x_ac.at(j) = std::fmin(std::fmax(x_ac.at(j), 0.f), x_sat);
        }

        // Build direct sensible-temperature differences at the NEW relaxed
        // air state. This is numerically stable even for very large canopies.
        for (uint j = 0; j < Nlayers; j++) {
            sum_gA_dT_new.at(j) =
                sum_gA_T.at(j) -
                sum_gA.at(j) * static_cast<double>(T_ac.at(j));
        }
        if (soil_node_present) {
            soil_sum_gA_dT_new =
                soil_sum_gA_T -
                soil_sum_gA * static_cast<double>(T_ac.front());
        }

        // Network fluxes evaluated with the NEW network state.
        Hc_network = 0.f;
        LEc_network = 0.f;
        for (uint j = 0; j < Nlayers; j++) {
            Hc_network += static_cast<float>(
                static_cast<double>(cp_air_mol) * sum_gA_dT_new.at(j));
            LEc_network += lambda_mol * (sum_gM_xs.at(j) - sum_gM.at(j) * x_ac.at(j));
        }
        Hs_network = soil_node_present
                         ? static_cast<float>(
                               static_cast<double>(cp_air_mol) *
                               soil_sum_gA_dT_new)
                         : 0.f;
        LEs_network = soil_node_present ? lambda_mol * (soil_sum_gM_xs - soil_sum_gM * x_ac.front()) : 0.f;
        H_atmosphere = cp_air_mol * g_a * (T_ac.back() - air_temperature_reference);
        LE_atmosphere = lambda_mol * g_a * (x_ac.back() - moisture_reference);
        sensible_closure = Hc_network + Hs_network - H_atmosphere;
        latent_closure = LEc_network + LEs_network - LE_atmosphere;

        // Same-state primitive/network identity. These are evaluated at the OLD air state,
        // i.e. the exact state that produced the primitive fluxes in this iteration.
        float Hc_same_state = 0.f, LEc_same_state = 0.f;
        for (uint j = 0; j < Nlayers; j++) {
            Hc_same_state += static_cast<float>(
                static_cast<double>(cp_air_mol) * sum_gA_dT_old.at(j));
            LEc_same_state += lambda_mol * (sum_gM_xs.at(j) - sum_gM.at(j) * x_ac_previous.at(j));
        }
        const float Hs_same_state =
            soil_node_present
                ? static_cast<float>(
                      static_cast<double>(cp_air_mol) *
                      soil_sum_gA_dT_old)
                : 0.f;
        const float LEs_same_state = soil_node_present ? lambda_mol * (soil_sum_gM_xs - soil_sum_gM * x_ac_previous.front()) : 0.f;

        Hc_sync = Hc_same_state - Hc_primitive;
        LEc_sync = LEc_same_state - LEc_primitive;
        Hs_sync = Hs_same_state - Hs_primitive;
        LEs_sync = LEs_same_state - LEs_primitive;

        // At a common fixed point the primitive source fluxes must also equal transport to the atmosphere.
        const float H_atm_old = cp_air_mol * g_a * (T_ac_previous.back() - air_temperature_reference);
        const float LE_atm_old = lambda_mol * g_a * (x_ac_previous.back() - moisture_reference);
        const float primitive_H_network_residual = Hc_primitive + Hs_primitive - H_atm_old;
        const float primitive_LE_network_residual = LEc_primitive + LEs_primitive - LE_atm_old;

        // Changes in the RELAXED state are useful for diagnostics, but they are
        // not the correct fixed-point residual when under-relaxation is used.
        float max_change_T = 0.f;
        float max_change_x = 0.f;
        float max_fixedpoint_T = 0.f;
        float max_fixedpoint_x = 0.f;
        for (uint j = 0; j < Nlayers; j++) {
            max_change_T = std::fmax(max_change_T, std::fabs(T_ac.at(j) - T_ac_previous.at(j)));
            max_change_x = std::fmax(max_change_x, std::fabs(x_ac.at(j) - x_ac_previous.at(j)));

            // Raw network candidate minus the state that generated the surface EB.
            // This is the actual residual of the TSEB-like fixed-point equation.
            max_fixedpoint_T = std::fmax(max_fixedpoint_T,
                                         std::fabs(T_ac_candidate.at(j) - T_ac_previous.at(j)));
            max_fixedpoint_x = std::fmax(max_fixedpoint_x,
                                         std::fabs(x_ac_candidate.at(j) - x_ac_previous.at(j)));
        }

        const float moisture_tolerance_base =
            canopy_airspace_tolerance_K *
            (esat_Pa(air_temperature_reference + 0.5f) -
             esat_Pa(air_temperature_reference - 0.5f)) / Patm;

        // Feed the newly relaxed state to the next surface-EB iteration.
        write_air_state_to_primitives(T_ac, x_ac);

        // ---------------------------------------------------------------
        // Convergence criterion
        // ---------------------------------------------------------------
        //
        // Converge the STATE fixed point, not H or LE themselves.
        //
        // With one airspace layer, the raw temperature residual is exactly
        // related to the network sensible-heat residual by
        //
        //   H_residual = cp * (Gc + Gs + ga) * (Tac_candidate - Tac_old).
        //
        // Likewise for vapor:
        //
        //   LE_residual = lambda * (GMc + GMs + ga) * (xac_candidate - xac_old).
        //
        // Therefore use a state tolerance that is tight enough to imply the
        // requested 0.1 W m-2 transport closure, while still remaining a
        // STATE-based convergence criterion.
        float temperature_tolerance = canopy_airspace_tolerance_K;
        float moisture_tolerance = moisture_tolerance_base;

        if (Nlayers == 1) {
            const double Gthermal_total =
                sum_gA.front() +
                (soil_node_present ? soil_sum_gA : 0.0) +
                static_cast<double>(g_a);
            if (Gthermal_total > 0.f) {
                temperature_tolerance = std::fmin(
                    temperature_tolerance,
                    static_cast<float>(
                        static_cast<double>(flux_tolerance_W_m2) /
                        (static_cast<double>(cp_air_mol) *
                         Gthermal_total)));
            }

            const float Gmoisture_total = sum_gM.front() +
                                          (soil_node_present ? soil_sum_gM : 0.f) +
                                          g_a;
            if (Gmoisture_total > 0.f) {
                moisture_tolerance = std::fmin(
                    moisture_tolerance,
                    flux_tolerance_W_m2 / (lambda_mol * Gmoisture_total));
            }
        }

        const bool temperature_converged =
            max_fixedpoint_T < temperature_tolerance;
        const bool moisture_converged =
            max_fixedpoint_x < moisture_tolerance;
        const bool state_converged =
            temperature_converged && moisture_converged;

        // Diagnostic checks only.  These do not control convergence.
        const bool sensible_sync_ok =
            std::fabs(Hc_sync) < flux_tolerance_W_m2 &&
            std::fabs(Hs_sync) < flux_tolerance_W_m2;
        const bool latent_sync_ok =
            std::fabs(LEc_sync) < flux_tolerance_W_m2 &&
            std::fabs(LEs_sync) < flux_tolerance_W_m2;
        const bool transport_balance_ok =
            std::fabs(primitive_H_network_residual) < flux_tolerance_W_m2 &&
            std::fabs(primitive_LE_network_residual) < flux_tolerance_W_m2;

        // G is algebraic in the CURRENT trial Ts through Rn_s(Ts), so no
        // independent G convergence test is needed.
        converged = state_converged;

        // Concise diagnostic.  The iteration stops from dT/dx only, while the
        // flux diagnostics show whether the converged implementation is internally
        // consistent.
        if (iteration < 5 || iteration % 5 == 0 || converged ||
            iteration + 1 == canopy_airspace_max_iterations) {
            std::cerr
                << "AIRSPACE iter=" << iteration
                << " | dT_relaxed=" << max_change_T
                << " | dx_relaxed=" << max_change_x
                << " | dT_fixed=" << max_fixedpoint_T
                << " | dx_fixed=" << max_fixedpoint_x
                << " | Ttol=" << temperature_tolerance
                << " | xtol=" << moisture_tolerance
                << " | Tconv=" << temperature_converged
                << " | xconv=" << moisture_converged
                << " | Hc_sync=" << Hc_sync
                << " | Hs_sync=" << Hs_sync
                << " | LEc_sync=" << LEc_sync
                << " | LEs_sync=" << LEs_sync
                << " | H_balance=" << primitive_H_network_residual
                << " | LE_balance=" << primitive_LE_network_residual
                << " | Hsync_ok=" << sensible_sync_ok
                << " | LEsync_ok=" << latent_sync_ok
                << " | transport_ok=" << transport_balance_ok
                << std::endl;
        }

        iteration++;
    }

    if (!converged) {
        warnings.addWarning("canopy_airspace_not_converged",
                            "Canopy airspace model did not reach the coupled T_ac/x_ac state fixed point within " +
                            std::to_string(canopy_airspace_max_iterations) + " iterations.");
    }

    // Final surface solve at the converged FINAL air state.
    //
    // IMPORTANT: explicitly write the converged T_ac and x_ac to every canopy
    // and soil primitive BEFORE recomputing Kustas-Norman R_s and BEFORE the
    // final primitive surface-energy-balance solve.  This removes any possible
    // one-iteration lag between the state used by the primitive EB and the
    // state used by the final network reduction.
    write_air_state_to_primitives(T_ac, x_ac);

    update_KN_soil_conductance(T_ac.front());
    evaluateSurfaceEnergyBalance(UUIDs, 0.f);

    // Recompute primitive totals at the final state for diagnostics/output.
    Hc_primitive = LEc_primitive = Hs_primitive = LEs_primitive = 0.f;
    context->calculatePrimitiveDataAreaWeightedSum(canopy_airspace_UUIDs, "sensible_flux", Hc_primitive);
    context->calculatePrimitiveDataAreaWeightedSum(canopy_airspace_UUIDs, "latent_flux", LEc_primitive);
    Hc_primitive /= ground_area;
    LEc_primitive /= ground_area;
    if (soil_node_present) {
        context->calculatePrimitiveDataAreaWeightedSum(canopy_airspace_ground_UUIDs, "sensible_flux", Hs_primitive);
        context->calculatePrimitiveDataAreaWeightedSum(canopy_airspace_ground_UUIDs, "latent_flux", LEs_primitive);
        Hs_primitive /= ground_area;
        LEs_primitive /= ground_area;
    }

    // Exact final-state reduction. Recompute both heat and vapor source terms after the
    // final surface solve, so all exported primitive/network diagnostics refer to the SAME state.
    double Gc = 0.0, GcT = 0.0, Gs = 0.0, GsT = 0.0;
    double Hc_direct_sum = 0.0, Hs_direct_sum = 0.0;
    float canopy_Ta_min = std::numeric_limits<float>::infinity();
    float canopy_Ta_max = -std::numeric_limits<float>::infinity();
    float soil_Ta_min = std::numeric_limits<float>::infinity();
    float soil_Ta_max = -std::numeric_limits<float>::infinity();
    float canopy_max_abs_dTa = 0.f;
    float soil_max_abs_dTa = 0.f;
    float GMc = 0.f, GMcXs = 0.f, GMs = 0.f, GMsXs = 0.f;

    auto primitive_moisture_terms = [&](uint UUID, float gH, float Tsrf, float &gM_out, float &xs_out) {
        float gS = gS_default;
        if (context->doesPrimitiveDataExist(UUID, "moisture_conductance") && context->getPrimitiveDataType("moisture_conductance") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "moisture_conductance", gS);
        }
        float stomatal_sidedness = 0.f;
        const uint twosided_flag = context->getPrimitiveTwosidedFlag(UUID, 1);
        if (twosided_flag != 0) {
            if (context->doesPrimitiveDataExist(UUID, "stomatal_sidedness") && context->getPrimitiveDataType("stomatal_sidedness") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(UUID, "stomatal_sidedness", stomatal_sidedness);
            } else if (context->doesPrimitiveDataExist(UUID, "evaporating_faces") && context->getPrimitiveDataType("evaporating_faces") == helios::HELIOS_TYPE_UINT) {
                uint evaporating_faces;
                context->getPrimitiveData(UUID, "evaporating_faces", evaporating_faces);
                stomatal_sidedness = (evaporating_faces == 2) ? 0.5f : 0.f;
            }
        }
        gM_out = 0.f;
        if (gH != 0.f && gS != 0.f) {
            gM_out = moisture_conductance_ratio * gH * gS *
                     (stomatal_sidedness / (moisture_conductance_ratio * gH + gS * stomatal_sidedness) +
                      (1.f - stomatal_sidedness) / (moisture_conductance_ratio * gH + gS * (1.f - stomatal_sidedness)));
        }
        float surface_humidity = surface_humidity_default;
        if (context->doesPrimitiveDataExist(UUID, "surface_humidity") && context->getPrimitiveDataType("surface_humidity") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "surface_humidity", surface_humidity);
        }
        xs_out = surface_humidity * esat_Pa(Tsrf) / Patm;
    };

    for (uint UUID: canopy_airspace_UUIDs) {
        float gH, Tsrf;
        context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);
        context->getPrimitiveData(UUID, "temperature", Tsrf);
        const float af = context->getPrimitiveArea(UUID) / ground_area;
        const double gA =
            static_cast<double>(gH) * static_cast<double>(af);
        Gc += gA;
        GcT += gA * static_cast<double>(Tsrf);
        Hc_direct_sum +=
            gA * (static_cast<double>(Tsrf) -
                  static_cast<double>(T_ac.front()));

        float Ta_primitive = T_ac.front();
        if (context->doesPrimitiveDataExist(UUID, "air_temperature") &&
            context->getPrimitiveDataType("air_temperature") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "air_temperature", Ta_primitive);
        }
        canopy_Ta_min = std::fmin(canopy_Ta_min, Ta_primitive);
        canopy_Ta_max = std::fmax(canopy_Ta_max, Ta_primitive);
        canopy_max_abs_dTa = std::fmax(
            canopy_max_abs_dTa,
            std::fabs(Ta_primitive - T_ac.front()));

        float gM, xs;
        primitive_moisture_terms(UUID, gH, Tsrf, gM, xs);
        GMc += gM * af;
        GMcXs += gM * af * xs;
    }
    if (soil_node_present) {
        for (uint UUID: canopy_airspace_ground_UUIDs) {
            float gH, Tsrf;
            context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);
            context->getPrimitiveData(UUID, "temperature", Tsrf);
            const float af = context->getPrimitiveArea(UUID) / ground_area;
            const double gA =
                static_cast<double>(gH) * static_cast<double>(af);
            Gs += gA;
            GsT += gA * static_cast<double>(Tsrf);
            Hs_direct_sum +=
                gA * (static_cast<double>(Tsrf) -
                      static_cast<double>(T_ac.front()));

            float Ta_primitive = T_ac.front();
            if (context->doesPrimitiveDataExist(UUID, "air_temperature") &&
                context->getPrimitiveDataType("air_temperature") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(UUID, "air_temperature", Ta_primitive);
            }
            soil_Ta_min = std::fmin(soil_Ta_min, Ta_primitive);
            soil_Ta_max = std::fmax(soil_Ta_max, Ta_primitive);
            soil_max_abs_dTa = std::fmax(
                soil_max_abs_dTa,
                std::fabs(Ta_primitive - T_ac.front()));

            float gM, xs;
            primitive_moisture_terms(UUID, gH, Tsrf, gM, xs);
            GMs += gM * af;
            GMsXs += gM * af * xs;
        }
    }

    Gc_effective = static_cast<float>(Gc);
    Gs_effective = static_cast<float>(Gs);
    Tc_effective = (Gc > 0.0)
                       ? static_cast<float>(GcT / Gc)
                       : air_temperature_reference;
    Ts_effective = (Gs > 0.0)
                       ? static_cast<float>(GsT / Gs)
                       : air_temperature_reference;

    // Export the exact bulk surface-temperature nodes used by the final
    // one-layer canopy-airspace resistance network. These are the temperatures
    // paired with Gc/Gs (and therefore Rx/Rs) in the Tac reconstruction.
    context->setGlobalData(
        "canopy_airspace_canopy_temperature",
        Tc_effective);

    context->setGlobalData(
        "canopy_airspace_soil_temperature",
        Ts_effective);

    float Hc_sync_from_air_temperature = NAN;
    float Hs_sync_from_air_temperature = NAN;

    if (Nlayers == 1) {
        Hc_network = static_cast<float>(
            static_cast<double>(cp_air_mol) * Hc_direct_sum);
        Hs_network = static_cast<float>(
            static_cast<double>(cp_air_mol) * Hs_direct_sum);
        H_atmosphere = cp_air_mol * g_a * (T_ac.front() - air_temperature_reference);
        LEc_network = lambda_mol * (GMcXs - GMc * x_ac.front());
        LEs_network = lambda_mol * (GMsXs - GMs * x_ac.front());
        LE_atmosphere = lambda_mol * g_a * (x_ac.front() - moisture_reference);
        Hc_sync = Hc_network - Hc_primitive;
        Hs_sync = Hs_network - Hs_primitive;

        // If primitive sensible heat is QH = cp*gH*(Ts-Ta_primitive), then
        // these expressions MUST reproduce Hc_sync/Hs_sync exactly.  A match
        // identifies a primitive-air-temperature mismatch; a non-match points
        // instead to gH/area/flux bookkeeping.
        double canopy_Ta_difference_sum = 0.0;
        for (uint UUID : canopy_airspace_UUIDs) {
            float gH = 0.f;
            float Ta_primitive = T_ac.front();
            context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);
            if (context->doesPrimitiveDataExist(UUID, "air_temperature")) {
                context->getPrimitiveData(UUID, "air_temperature", Ta_primitive);
            }
            const double af =
                static_cast<double>(context->getPrimitiveArea(UUID)) /
                static_cast<double>(ground_area);
            canopy_Ta_difference_sum +=
                static_cast<double>(gH) * af *
                (static_cast<double>(Ta_primitive) -
                 static_cast<double>(T_ac.front()));
        }

        double soil_Ta_difference_sum = 0.0;
        for (uint UUID : canopy_airspace_ground_UUIDs) {
            float gH = 0.f;
            float Ta_primitive = T_ac.front();
            context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);
            if (context->doesPrimitiveDataExist(UUID, "air_temperature")) {
                context->getPrimitiveData(UUID, "air_temperature", Ta_primitive);
            }
            const double af =
                static_cast<double>(context->getPrimitiveArea(UUID)) /
                static_cast<double>(ground_area);
            soil_Ta_difference_sum +=
                static_cast<double>(gH) * af *
                (static_cast<double>(Ta_primitive) -
                 static_cast<double>(T_ac.front()));
        }

        Hc_sync_from_air_temperature = static_cast<float>(
            static_cast<double>(cp_air_mol) *
            canopy_Ta_difference_sum);
        Hs_sync_from_air_temperature = static_cast<float>(
            static_cast<double>(cp_air_mol) *
            soil_Ta_difference_sum);
        LEc_sync = LEc_network - LEc_primitive;
        LEs_sync = LEs_network - LEs_primitive;
        sensible_closure = Hc_network + Hs_network - H_atmosphere;
        latent_closure = LEc_network + LEs_network - LE_atmosphere;
    }

    // -----------------------------------------------------------------
    // FINAL PHYSICS CONSISTENCY CHECKS
    // -----------------------------------------------------------------
    // These are deliberately evaluated AFTER state convergence.
    //
    // 1) primitive H vs resistance-network H
    // 2) primitive LE vs vapor-network LE
    // 3) network transport closure to the atmosphere
    //
    // A failure here means an implementation/reduction inconsistency.  It
    // does NOT mean Tac or xac should be forced until the mismatch disappears.
    const bool final_H_sync_ok =
        std::fabs(Hc_sync) < flux_tolerance_W_m2 &&
        std::fabs(Hs_sync) < flux_tolerance_W_m2;
    const bool final_LE_sync_ok =
        std::fabs(LEc_sync) < flux_tolerance_W_m2 &&
        std::fabs(LEs_sync) < flux_tolerance_W_m2;
    const bool final_H_closure_ok =
        std::fabs(sensible_closure) < flux_tolerance_W_m2;
    const bool final_LE_closure_ok =
        std::fabs(latent_closure) < flux_tolerance_W_m2;

    std::cerr << std::setprecision(9)
        << "AIRSPACE FINAL"
        << " | iterations=" << iteration
        << " | Hc_sync=" << Hc_sync
        << " | Hs_sync=" << Hs_sync
        << " | LEc_sync=" << LEc_sync
        << " | LEs_sync=" << LEs_sync
        << " | H_closure=" << sensible_closure
        << " | LE_closure=" << latent_closure
        << " | Hc_sync_from_Ta=" << Hc_sync_from_air_temperature
        << " | Hs_sync_from_Ta=" << Hs_sync_from_air_temperature
        << " | canopy_Ta_min=" << canopy_Ta_min
        << " | canopy_Ta_max=" << canopy_Ta_max
        << " | Tac=" << T_ac.front()
        << " | canopy_max_abs_dTa=" << canopy_max_abs_dTa
        << " | soil_Ta_min=" << soil_Ta_min
        << " | soil_Ta_max=" << soil_Ta_max
        << " | soil_max_abs_dTa=" << soil_max_abs_dTa
        << " | Hsync_ok=" << final_H_sync_ok
        << " | LEsync_ok=" << final_LE_sync_ok
        << " | Hclosure_ok=" << final_H_closure_ok
        << " | LEclosure_ok=" << final_LE_closure_ok
        << std::endl;

    // Export pass/fail flags as uints for downstream diagnostics.
    context->setGlobalData("canopy_airspace_H_sync_ok", static_cast<uint>(final_H_sync_ok));
    context->setGlobalData("canopy_airspace_LE_sync_ok", static_cast<uint>(final_LE_sync_ok));
    context->setGlobalData("canopy_airspace_H_closure_ok", static_cast<uint>(final_H_closure_ok));
    context->setGlobalData("canopy_airspace_LE_closure_ok", static_cast<uint>(final_LE_closure_ok));
    context->setGlobalData("canopy_airspace_Hc_sync_from_air_temperature", Hc_sync_from_air_temperature);
    context->setGlobalData("canopy_airspace_Hs_sync_from_air_temperature", Hs_sync_from_air_temperature);
    context->setGlobalData("canopy_airspace_canopy_max_abs_dTa", canopy_max_abs_dTa);
    context->setGlobalData("canopy_airspace_soil_max_abs_dTa", soil_max_abs_dTa);

    if (Nlayers == 1) {
        // For one layer, this is the exact resistance-network reconstruction of Tac:
        // Tac = (Gc*Tc + Gs*Ts + ga*Ta)/(Gc+Gs+ga).
        const float Tac_reconstructed = (Gc * Tc_effective + Gs * Ts_effective + g_a * air_temperature_reference) /
                                        (Gc + Gs + g_a);
        context->setGlobalData("canopy_airspace_Tac_reconstructed", Tac_reconstructed);
        context->setGlobalData("canopy_airspace_Tac_reconstruction_error", T_ac.front() - Tac_reconstructed);

        // Keep the numerically stable canopy/soil reductions in double precision
        // internally, but export resistances as FLOAT globals so main.cpp can read
        // them consistently with the rest of the Helios/TSEB interface.
        const float canopy_resistance_s_m =
            (Gc > 0.0)
                ? static_cast<float>(
                      static_cast<double>(rho_air_mol_m3) / Gc)
                : NAN;

        const float soil_resistance_effective_s_m =
            (Gs > 0.0)
                ? static_cast<float>(
                      static_cast<double>(rho_air_mol_m3) / Gs)
                : NAN;

        context->setGlobalData(
            "canopy_airspace_canopy_resistance",
            canopy_resistance_s_m);

        // Raw Kustas-Norman soil resistance used by the ground surface EB and by the
        // resistance network. For a ground tiling whose total area equals ground_area,
        // this is also rho_mol/Gs exactly.
        context->setGlobalData(
            "canopy_airspace_soil_resistance",
            soil_resistance_KN_s_m);

        context->setGlobalData(
            "canopy_airspace_soil_resistance_effective",
            soil_resistance_effective_s_m);

        context->setGlobalData(
            "canopy_airspace_soil_heat_conductance_KN",
            soil_heat_conductance_KN_mol_m2_s);

        context->setGlobalData(
            "canopy_airspace_soil_temperature_KN",
            soil_temperature_KN_K);
    }

    float T_ac_mean = 0.f, x_ac_mean = 0.f;
    for (uint j = 0; j < Nlayers; j++) {
        T_ac_mean += T_ac.at(j);
        x_ac_mean += x_ac.at(j);
    }
    T_ac_mean /= float(Nlayers);
    x_ac_mean /= float(Nlayers);

    context->setGlobalData("canopy_air_temperature", T_ac_mean);
    context->setGlobalData("canopy_air_humidity", x_ac_mean * Patm / esat_Pa(T_ac_mean));

    // Air state immediately above the soil is the lowest canopy-airspace node.
    // For Nlayers = 1 this is identical to the single canopy-air temperature
    // and humidity.  For Nlayers > 1 it remains the bottom-layer air state
    // that is actually used for soil-air exchange in the resistance network.
    const float soil_air_temperature = T_ac.front();
    const float soil_air_humidity =
        x_ac.front() * Patm / esat_Pa(soil_air_temperature);

    context->setGlobalData("soil_air_temperature", soil_air_temperature);
    context->setGlobalData("soil_air_humidity", soil_air_humidity);

    context->setGlobalData("canopy_air_temperature_layers", T_ac);
    context->setGlobalData("aerodynamic_resistance", aerodynamic_resistance_s_m);
    context->setGlobalData("canopy_airspace_iterations", iteration);

    std::vector<float> humidity_layers(Nlayers);
    for (uint j = 0; j < Nlayers; j++) {
        humidity_layers.at(j) = x_ac.at(j) * Patm / esat_Pa(T_ac.at(j));
    }
    context->setGlobalData("canopy_air_humidity_layers", humidity_layers);

    context->setGlobalData("canopy_airspace_canopy_sensible_flux", Hc_network);
    context->setGlobalData("canopy_airspace_soil_sensible_flux", Hs_network);
    context->setGlobalData("canopy_airspace_atmospheric_sensible_flux", H_atmosphere);
    context->setGlobalData("canopy_airspace_canopy_latent_flux", LEc_network);
    context->setGlobalData("canopy_airspace_soil_latent_flux", LEs_network);
    context->setGlobalData("canopy_airspace_atmospheric_latent_flux", LE_atmosphere);

    context->setGlobalData("canopy_airspace_canopy_sensible_sync_error", Hc_sync);
    context->setGlobalData("canopy_airspace_soil_sensible_sync_error", Hs_sync);
    context->setGlobalData("canopy_airspace_canopy_latent_sync_error", LEc_sync);
    context->setGlobalData("canopy_airspace_soil_latent_sync_error", LEs_sync);
    context->setGlobalData("canopy_airspace_sensible_closure_error", sensible_closure);
    context->setGlobalData("canopy_airspace_latent_closure_error", latent_closure);

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

float EnergyBalanceModel::calculateAerodynamicResistance(float wind_speed_reference_m_s) const {

    // ---------------------------------------------------------------------
    // Neutral aerodynamic resistance matched to the TSEB configuration.
    //
    //     d0  = 0.65 * h_c
    //     z0M = 0.125 * h_c
    //     z0H = z0M / exp(kB)
    //
    // main.cpp currently sets canopy_kB = 0, therefore z0H = z0M.
    //
    //     R_A = ln((z_u-d0)/z0M) * ln((z_T-d0)/z0H)
    //           ------------------------------------------------
    //                         kappa^2 U_ref
    // ---------------------------------------------------------------------
    if (!(wind_speed_reference_m_s > 0.f)) {
        helios_runtime_error(
            "ERROR (EnergyBalanceModel::calculateAerodynamicResistance): "
            "Reference wind speed must be greater than zero.");
    }

    const float displacement_height = 0.65f * canopy_height_m;
    const float zo_m = 0.125f * canopy_height_m;

    float temperature_reference_height_m = reference_height_m;
    if (context->doesGlobalDataExist("air_temperature_reference_height") &&
        context->getGlobalDataType("air_temperature_reference_height") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData(
            "air_temperature_reference_height",
            temperature_reference_height_m);
    }

    float kB = 0.f;
    if (context->doesGlobalDataExist("canopy_kB") &&
        context->getGlobalDataType("canopy_kB") == helios::HELIOS_TYPE_FLOAT) {
        context->getGlobalData("canopy_kB", kB);
    }

    const float zo_h = zo_m / std::exp(kB);

    if (!(zo_m > 0.f) ||
        !(zo_h > 0.f) ||
        !(reference_height_m > displacement_height) ||
        !(temperature_reference_height_m > displacement_height)) {
        helios_runtime_error(
            "ERROR (EnergyBalanceModel::calculateAerodynamicResistance): "
            "Invalid TSEB aerodynamic geometry.");
    }

    const float ln_momentum =
        std::log((reference_height_m - displacement_height) / zo_m);

    const float ln_heat =
        std::log((temperature_reference_height_m - displacement_height) / zo_h);

    if (!(ln_momentum > 0.f) || !(ln_heat > 0.f)) {
        helios_runtime_error(
            "ERROR (EnergyBalanceModel::calculateAerodynamicResistance): "
            "Invalid logarithmic terms for TSEB aerodynamic resistance.");
    }

    return ln_momentum * ln_heat /
           (von_Karman_constant *
            von_Karman_constant *
            wind_speed_reference_m_s);
}

void EnergyBalanceModel::updateCanopyWindProfile(float wind_speed_canopy_top_m_s) {

    // Use the same attenuation formulation as evaluateCanopyAirspace() so the
    // primitive canopy wind speeds, soil wind speed, and inter-layer mixing
    // are all based on one consistent profile.
    uint canopy_wind_model = 0;
    if (context->doesGlobalDataExist("canopy_wind_model") &&
        context->getGlobalDataType("canopy_wind_model") == helios::HELIOS_TYPE_UINT) {
        context->getGlobalData("canopy_wind_model", canopy_wind_model);
    }

    float attenuation_coefficient = 0.5f * canopy_airspace_LAI;

    if (canopy_wind_model == 1) {
        float canopy_leaf_width_m = NAN;

        if (!context->doesGlobalDataExist("canopy_leaf_width_m") ||
            context->getGlobalDataType("canopy_leaf_width_m") != helios::HELIOS_TYPE_FLOAT) {
            helios_runtime_error(
                "ERROR (EnergyBalanceModel::updateCanopyWindProfile): "
                "Goudriaan wind model requested, but canopy_leaf_width_m "
                "is missing or is not a float.");
        }

        context->getGlobalData("canopy_leaf_width_m", canopy_leaf_width_m);

        if (!(canopy_leaf_width_m > 0.f) || !std::isfinite(canopy_leaf_width_m)) {
            helios_runtime_error(
                "ERROR (EnergyBalanceModel::updateCanopyWindProfile): "
                "Goudriaan wind model requires canopy_leaf_width_m > 0.");
        }

        attenuation_coefficient =
            0.28f *
            std::pow(canopy_airspace_LAI, 2.f / 3.f) *
            std::pow(canopy_height_m, 1.f / 3.f) *
            std::pow(canopy_leaf_width_m, -1.f / 3.f);
    } else if (canopy_wind_model != 0) {
        helios_runtime_error(
            "ERROR (EnergyBalanceModel::updateCanopyWindProfile): "
            "canopy_wind_model must be 0 (Cionco) or 1 (Goudriaan).");
    }

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

    // Ground heat flux G is calculated internally from the current trial soil net
    // radiation in the surface-energy-balance residual. Ensure final Rn and G are
    // written so the simulation can aggregate the exact converged soil balance.
    if (!ground_UUIDs.empty() && std::find(output_prim_data.begin(), output_prim_data.end(), "storage_flux") == output_prim_data.end()) {
        output_prim_data.emplace_back("storage_flux");
    }
    if (!ground_UUIDs.empty() && std::find(output_prim_data.begin(), output_prim_data.end(), "net_radiation_flux") == output_prim_data.end()) {
        output_prim_data.emplace_back("net_radiation_flux");
    }

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
    size_t assumed_default_gs = 0;
    size_t assumed_default_Qother = 0;
    size_t assumed_default_heatcapacity = 0;
    size_t assumed_default_fs = 0;
    size_t twosided_0 = 0;
    size_t twosided_1 = 0;
    size_t Ne_1 = 0;
    size_t Ne_2 = 0;

    size_t Nprimitives = UUIDs.size();

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
            assumed_default_gH++;
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
                                              float Tprev, bool is_airspace_ground, float santanello_friedl_phase, float santanello_friedl_amplitude) {

    // Outgoing emission flux
    float Rout = float(Nsides) * eps * 5.67e-8F * T * T * T * T;

    // Sensible heat flux
    float QH = cp_air_mol * gH * (T - Ta);

    // Latent heat flux
    // Use the exact same saturation-vapor-pressure function as the
    // canopy-airspace vapor network.  Using a different approximation here
    // prevents LE_primitive and LE_network from becoming identical even at
    // the same T/x state.
    float es = esat_Pa(T);
    // A zero boundary-layer conductance (gH) or zero stomatal conductance (gS) means there is no vapor
    // pathway, so gM is zero. Guarding both cases also avoids a 0/0 in the series-conductance expression
    // when gH == 0 and the stomatal sidedness makes one of the denominators vanish.
    float gM = 0.f;
    if (gH != 0.f && gS != 0.f) {
        gM = 1.08f * gH * gS * (stomatal_sidedness / (1.08f * gH + gS * stomatal_sidedness) + (1.f - stomatal_sidedness) / (1.08f * gH + gS * (1.f - stomatal_sidedness)));
    }

    float QL = gM * lambda_mol * (es * surfacehumidity - ea) / pressure;

    // Dynamic material heat storage (zero in the steady canopy-airspace solve).
    float storage = 0.f;
    if (dt > 0) {
        storage = heatcapacity * (T - Tprev) / dt;
    }

    // Ground heat flux G following the Santanello-Friedl formulation used by
    // the TSEB baseline.  It MUST use soil NET radiation at the current trial
    // temperature, not absorbed radiation alone:
    //
    //   Rn_soil(T) = R_absorbed - eps*sigma*T^4
    //   G(T) = 0.35 * Rn_soil(T) * cos[2*pi*(t_from_noon + 10800)/74000]
    //
    // For canopy and all non-airspace-ground primitives G = 0.
    float ground_heat_flux = 0.f;
    if (is_airspace_ground) {
        const float Rnet_soil = R - Rout;
        ground_heat_flux = santanello_friedl_amplitude * Rnet_soil * santanello_friedl_phase;
    }

    // Residual.  For an airspace ground primitive this is exactly
    // Rn_s - H_s - LE_s - G = 0 (plus any explicitly supplied non-G Qother).
    return R - Rout - QH - QL - Qother - ground_heat_flux - storage;
}
void EnergyBalanceModel::evaluateSurfaceEnergyBalance_CPU(const std::vector<uint> &UUIDs, float dt) {

    // Create warning aggregator for default value warnings
    helios::WarningAggregator warnings;
    warnings.setEnabled(message_flag);

    //---- Sum up to get total absorbed radiation across all bands ----//

    if (radiation_bands.empty()) {
        helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation bands were found.");
    }

    const uint Nprimitives = UUIDs.size();

    std::vector<float> Rn(Nprimitives, 0);

    // Accumulate radiation across bands
    for (int b = 0; b < radiation_bands.size(); b++) {
        for (size_t u = 0; u < Nprimitives; u++) {
            size_t p = UUIDs.at(u);

            char str[50];
            snprintf(str, sizeof(str), "radiation_flux_%s", radiation_bands.at(b).c_str());
            if (!context->doesPrimitiveDataExist(p, str)) {
                helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation was found in the context for band " + std::string(radiation_bands.at(b)) + ". Did you run the radiation model for this band?");
            } else if (context->getPrimitiveDataType(str) != helios::HELIOS_TYPE_FLOAT) {
                helios_runtime_error("ERROR (EnergyBalanceModel::run): Radiation primitive data for band " + std::string(radiation_bands.at(b)) + " does not have the correct type of 'float'");
            }
            float R;
            context->getPrimitiveData(p, str, R);
            Rn.at(u) += R;
        }
    }

    // Determine which radiation band(s) govern the emitted thermal radiation. The RadiationModel records
    // per-band emission-enabled state in global data 'emission_enabled_<band>'. In a properly configured
    // run emission is enabled for exactly one band, and that band's emissivity is used. If more than one
    // band has emission enabled we warn and use the first. When no emission-enabled information is
    // available (e.g. radiation fluxes were set manually without running the RadiationModel), all bands
    // added to the energy balance are treated as candidates for backward compatibility.
    std::vector<std::string> emissivity_candidate_bands;
    {
        std::vector<std::string> emission_enabled_bands;
        bool emission_info_available = false;
        for (const auto &band: radiation_bands) {
            std::string gdlabel = "emission_enabled_" + band;
            if (context->doesGlobalDataExist(gdlabel.c_str()) && context->getGlobalDataType(gdlabel.c_str()) == helios::HELIOS_TYPE_UINT) {
                emission_info_available = true;
                uint enabled = 0;
                context->getGlobalData(gdlabel.c_str(), enabled);
                if (enabled == 1) {
                    emission_enabled_bands.push_back(band);
                }
            }
        }
        if (emission_info_available) {
            if (emission_enabled_bands.size() > 1) {
                warnings.addWarning("multiple_emission_bands", "Emission is enabled for more than one radiation band. Thermal emission uses a single emissivity; the emissivity of the first emission-enabled band will be used.");
            }
            emissivity_candidate_bands = emission_enabled_bands; // may be empty => no modeled band emits, blackbody default used
        } else {
            emissivity_candidate_bands = radiation_bands;
        }
    }

    //---- Set up temperature solution ----//

    // Allocate arrays for inputs
    std::vector<float> To(Nprimitives);
    std::vector<float> R(Nprimitives);
    std::vector<float> Qother(Nprimitives);
    std::vector<float> eps(Nprimitives);
    std::vector<float> Ta(Nprimitives);
    std::vector<float> ea(Nprimitives);
    std::vector<float> pressure(Nprimitives);
    std::vector<float> gH(Nprimitives);
    std::vector<float> gS(Nprimitives);
    std::vector<uint> Nsides(Nprimitives);
    std::vector<float> stomatal_sidedness(Nprimitives);
    std::vector<float> heatcapacity(Nprimitives);
    std::vector<float> surfacehumidity(Nprimitives);
    std::vector<unsigned char> is_airspace_ground(Nprimitives, 0);

    // Santanello-Friedl parameters.  For canopy-airspace ground, G is evaluated
    // directly inside evaluateEnergyBalance_CPU() at each trial Ts using the
    // corresponding trial soil net radiation.
    float santanello_friedl_phase = 0.f;
    float santanello_friedl_amplitude = 0.35f;
    if (canopy_airspace_enabled && !canopy_airspace_ground_UUIDs.empty()) {
        float seconds_from_solar_noon = NAN;
        if (context->doesGlobalDataExist("santanello_friedl_seconds_from_solar_noon") &&
            context->getGlobalDataType("santanello_friedl_seconds_from_solar_noon") == helios::HELIOS_TYPE_FLOAT) {
            context->getGlobalData("santanello_friedl_seconds_from_solar_noon", seconds_from_solar_noon);
        }
        if (!std::isfinite(seconds_from_solar_noon)) {
            helios_runtime_error("ERROR (EnergyBalanceModel::evaluateSurfaceEnergyBalance_CPU): canopy-airspace ground requires global float data 'santanello_friedl_seconds_from_solar_noon'.");
        }
        if (context->doesGlobalDataExist("santanello_friedl_amplitude") &&
            context->getGlobalDataType("santanello_friedl_amplitude") == helios::HELIOS_TYPE_FLOAT) {
            context->getGlobalData("santanello_friedl_amplitude", santanello_friedl_amplitude);
        }
        santanello_friedl_phase = std::cos(2.f * float(M_PI) *
                                           (seconds_from_solar_noon + 10800.f) / 74000.f);
    }
    std::unordered_set<uint> airspace_ground_set(canopy_airspace_ground_UUIDs.begin(), canopy_airspace_ground_UUIDs.end());

    bool calculated_blconductance_used = false;
    bool primitive_length_used = false;

    // Data preparation loop - same as CUDA version
    for (uint u = 0; u < Nprimitives; u++) {
        size_t p = UUIDs.at(u);

        // Initial guess for surface temperature
        if (context->doesPrimitiveDataExist(p, "temperature") && context->getPrimitiveDataType("temperature") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "temperature", To[u]);
        } else {
            To[u] = temperature_default;
            warnings.addWarning("missing_surface_temperature", "Primitive data 'temperature' not set, using default (" + std::to_string(temperature_default) + " K)");
        }
        if (To[u] == 0) {
            To[u] = 300;
        }

        // Air temperature
        if (context->doesPrimitiveDataExist(p, "air_temperature") && context->getPrimitiveDataType("air_temperature") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "air_temperature", Ta[u]);
            if (Ta[u] < 250.f) {
                warnings.addWarning("air_temperature_likely_celsius", "Value of " + std::to_string(Ta[u]) + " in 'air_temperature' is very small - should be in Kelvin, using default (" + std::to_string(air_temperature_default) + " K)");
                Ta[u] = air_temperature_default;
            }
        } else {
            Ta[u] = air_temperature_default;
            warnings.addWarning("missing_air_temperature", "Primitive data 'air_temperature' not set, using default (" + std::to_string(air_temperature_default) + " K)");
        }

        // Air relative humidity
        float hr;
        if (context->doesPrimitiveDataExist(p, "air_humidity") && context->getPrimitiveDataType("air_humidity") == helios::HELIOS_TYPE_FLOAT) {
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
            warnings.addWarning("missing_air_humidity", "Primitive data 'air_humidity' not set, using default (" + std::to_string(air_humidity_default) + ")");
        }

        // Air vapor pressure
        float esat = esat_Pa(Ta[u]);
        ea[u] = hr * esat;

        // Air pressure
        if (context->doesPrimitiveDataExist(p, "air_pressure") && context->getPrimitiveDataType("air_pressure") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "air_pressure", pressure[u]);
            if (pressure[u] < 10000.f) {
                if (message_flag) {
                    warnings.addWarning("air_pressure_out_of_range", "Value of " + std::to_string(pressure[u]) +
                                                                             " given in 'air_pressure' primitive data is very small. "
                                                                             "Values should be given in units of Pascals. Assuming default value of " +
                                                                             std::to_string(pressure_default));
                }
                pressure[u] = pressure_default;
            }
        } else {
            pressure[u] = pressure_default;
        }

        // Number of sides emitting radiation
        uint twosided_flag = context->getPrimitiveTwosidedFlag(p, 1);
        Nsides[u] = (twosided_flag == 0) ? 1 : 2;

        // Number of evaporating/transpiring faces
        stomatal_sidedness[u] = 0.f;
        if (Nsides[u] == 2 && context->doesPrimitiveDataExist(p, "stomatal_sidedness") && context->getPrimitiveDataType("stomatal_sidedness") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "stomatal_sidedness", stomatal_sidedness[u]);
        } else if (Nsides[u] == 2 && context->doesPrimitiveDataExist(p, "evaporating_faces") && context->getPrimitiveDataType("evaporating_faces") == helios::HELIOS_TYPE_UINT) {
            uint flag;
            context->getPrimitiveData(p, "evaporating_faces", flag);
            if (flag == 1) {
                stomatal_sidedness[u] = 0.f;
            } else if (flag == 2) {
                stomatal_sidedness[u] = 0.5f;
            }
        }

        // Boundary-layer conductance to heat
        if (context->doesPrimitiveDataExist(p, "boundarylayer_conductance") && context->getPrimitiveDataType("boundarylayer_conductance") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "boundarylayer_conductance", gH[u]);
        } else {
            // Wind speed
            float U;
            if (context->doesPrimitiveDataExist(p, "wind_speed") && context->getPrimitiveDataType("wind_speed") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(p, "wind_speed", U);
            } else {
                U = wind_speed_default;
                warnings.addWarning("missing_wind_speed", "Primitive data 'wind_speed' not set, using default (" + std::to_string(wind_speed_default) + " m/s)");
            }

            // Characteristic size of primitive
            float L;
            if (context->doesPrimitiveDataExist(p, "object_length") && context->getPrimitiveDataType("object_length") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(p, "object_length", L);
                if (L == 0) {
                    L = sqrt(context->getPrimitiveArea(p));
                    primitive_length_used = true;
                }
            } else if (context->getPrimitiveParentObjectID(p) > 0) {
                uint objID = context->getPrimitiveParentObjectID(p);
                L = sqrt(context->getObjectArea(objID));
            } else {
                L = sqrt(context->getPrimitiveArea(p));
                primitive_length_used = true;
            }

            gH[u] = 0.135f * sqrt(U / L) * float(Nsides[u]);
            calculated_blconductance_used = true;
        }

        // Moisture conductance
        if (context->doesPrimitiveDataExist(p, "moisture_conductance") && context->getPrimitiveDataType("moisture_conductance") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "moisture_conductance", gS[u]);
        } else {
            gS[u] = gS_default;
        }

        // Identify soil primitives coupled to the canopy-airspace model.
        // Their Santanello-Friedl G is calculated INSIDE the residual from the
        // current trial Rn_s(Ts), so it must not also be supplied in Qother.
        is_airspace_ground[u] = (canopy_airspace_enabled && airspace_ground_set.find(p) != airspace_ground_set.end()) ? 1 : 0;

        if (is_airspace_ground[u]) {
            Qother[u] = 0.f;
        } else if (context->doesPrimitiveDataExist(p, "other_surface_flux") && context->getPrimitiveDataType("other_surface_flux") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "other_surface_flux", Qother[u]);
        } else {
            Qother[u] = Qother_default;
        }

        // Object heat capacity
        if (context->doesPrimitiveDataExist(p, "heat_capacity") && context->getPrimitiveDataType("heat_capacity") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "heat_capacity", heatcapacity[u]);
        } else {
            heatcapacity[u] = heatcapacity_default;
        }

        // Surface humidity
        if (context->doesPrimitiveDataExist(p, "surface_humidity") && context->getPrimitiveDataType("surface_humidity") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "surface_humidity", surfacehumidity[u]);
        } else {
            surfacehumidity[u] = surface_humidity_default;
        }

        // Emissivity for the emitted thermal radiation term. Use the first candidate band that defines
        // an emissivity for this primitive; default to 1.0 (blackbody) if none do.
        eps[u] = 1.f;
        int emissivity_defined_count = 0;
        for (const auto &band: emissivity_candidate_bands) {
            std::string elabel = "emissivity_" + band;
            if (context->doesPrimitiveDataExist(p, elabel.c_str()) && context->getPrimitiveDataType(elabel.c_str()) == helios::HELIOS_TYPE_FLOAT) {
                if (emissivity_defined_count == 0) {
                    context->getPrimitiveData(p, elabel.c_str(), eps[u]);
                }
                emissivity_defined_count++;
            }
        }
        if (emissivity_defined_count == 0 && !emissivity_candidate_bands.empty()) {
            warnings.addWarning("missing_emissivity", "Emissivity not set for the emitting radiation band(s), using default (1.0)");
        } else if (emissivity_candidate_bands.size() > 1 && emissivity_defined_count > 1) {
            warnings.addWarning("multiple_emissivity_bands", "Emissivity was defined for more than one radiation band. Thermal emission uses a single emissivity; the value from the first band will be used.");
        }

        // Net absorbed radiation
        R[u] = Rn.at(u);
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

//---- Solve energy balance using OpenMP ----//

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
    std::vector<float> T(Nprimitives);

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
        float resid_old = evaluateEnergyBalance_CPU(T_old, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p], is_airspace_ground[p] != 0, santanello_friedl_phase, santanello_friedl_amplitude);
        float resid_old_old = evaluateEnergyBalance_CPU(T_old_old, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p], is_airspace_ground[p] != 0, santanello_friedl_phase, santanello_friedl_amplitude);

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

            resid = evaluateEnergyBalance_CPU(T_new, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p], is_airspace_ground[p] != 0, santanello_friedl_phase, santanello_friedl_amplitude);

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

    //---- Write results back to Context ----//

    for (uint u = 0; u < Nprimitives; u++) {
        size_t UUID = UUIDs.at(u);

        if (!std::isfinite(T[u])) { // catches NaN and infinities
            T[u] = temperature_default;
        }

        context->setPrimitiveData(UUID, "temperature", T[u]);

        float QH = cp_air_mol * gH[u] * (T[u] - Ta[u]);
        context->setPrimitiveData(UUID, "sensible_flux", QH);

        float es = esat_Pa(T[u]);
        float gM = 0.f;
        if (gH[u] != 0.f && gS[u] != 0.f) {
            gM = 1.08f * gH[u] * gS[u] * (stomatal_sidedness[u] / (1.08f * gH[u] + gS[u] * stomatal_sidedness[u]) + (1.f - stomatal_sidedness[u]) / (1.08f * gH[u] + gS[u] * (1.f - stomatal_sidedness[u])));
        }
        float QL = lambda_mol * gM * (es * surfacehumidity[u] - ea[u]) / pressure[u];
        context->setPrimitiveData(UUID, "latent_flux", QL);

        for (const auto &data_label: output_prim_data) {
            if (data_label == "boundarylayer_conductance_out") {
                context->setPrimitiveData(UUID, "boundarylayer_conductance_out", gH[u]);
            } else if (data_label == "vapor_pressure_deficit") {
                float vpd = (es - ea[u]) / pressure[u];
                context->setPrimitiveData(UUID, "vapor_pressure_deficit", vpd);
            } else if (data_label == "net_radiation_flux") {
                float Rnet = R[u] - float(Nsides[u]) * eps[u] * 5.67e-8F * std::pow(T[u], 4);
                context->setPrimitiveData(UUID, "net_radiation_flux", Rnet);
            } else if (data_label == "storage_flux") {
                float storage = 0.f;
                if (dt > 0) {
                    storage = heatcapacity[u] * (T[u] - To[u]) / dt;
                }
                float G_report = 0.f;
                if (airspace_ground_set.find(UUID) != airspace_ground_set.end()) {
                    const float Rnet_final = R[u] - float(Nsides[u]) * eps[u] * 5.67e-8F * std::pow(T[u], 4);
                    G_report = santanello_friedl_amplitude * Rnet_final * santanello_friedl_phase;
                    context->setPrimitiveData(UUID, "ground_heat_flux", G_report);
                }
                context->setPrimitiveData(UUID, "storage_flux", storage + G_report);
            }
        }
    }
}



