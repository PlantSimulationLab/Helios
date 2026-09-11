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
    if (gpu_acceleration_enabled) {
        evaluateSurfaceEnergyBalance_GPU(UUIDs, dt);
    } else {
        evaluateSurfaceEnergyBalance_CPU(UUIDs, dt);
    }
#else
    // CPU-only build: always use OpenMP implementation
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

    // Aerodynamic resistance between the canopy top and the reference height, converted from a resistance per unit ground area to a molar conductance.
    float rho_air_mol_m3 = Patm / (R * air_temperature_reference);
    float aerodynamic_resistance_s_m = calculateAerodynamicResistance(wind_speed_reference);
    float g_a = rho_air_mol_m3 / aerodynamic_resistance_s_m;

    // Wind speed at the canopy top, obtained by extrapolating the reference wind speed down to the canopy height with the same log law used for the aerodynamic resistance.
    float exp_half_LAI = std::exp(-0.5f * canopy_airspace_LAI);
    float zo_m = canopy_height_m * exp_half_LAI * (1.f - exp_half_LAI);
    float displacement_height = canopy_height_m * (1.f - 2.f / canopy_airspace_LAI * (1.f - exp_half_LAI));
    float wind_speed_canopy_top = wind_speed_reference * std::log((canopy_height_m - displacement_height) / zo_m) / std::log((reference_height_m - displacement_height) / zo_m);
    wind_speed_canopy_top = std::fmax(wind_speed_canopy_top, 0.01f);

    // Conductance between adjacent canopy airspace layers, from the eddy diffusivity for heat divided by the distance between layer midpoints. Turbulent mixing is strongly damped with depth into the canopy, so the diffusivity decays
    // exponentially from its canopy-top value with the same attenuation coefficient used for the wind profile (Cionco 1972; the K/dz discretization follows standard multilayer canopy practice, e.g. Bonan et al. 2018).
    float friction_velocity = von_Karman_constant * wind_speed_reference / std::log((reference_height_m - displacement_height) / zo_m);
    float eddy_diffusivity_canopy_top = von_Karman_constant * friction_velocity * (canopy_height_m - displacement_height);
    float attenuation_coefficient = 0.5f * canopy_airspace_LAI;

    // Conductance between layer j and layer j+1, evaluated at the shared layer boundary. Sized Nlayers so that the last entry, which has no layer above it, is unused.
    std::vector<float> g_layer(Nlayers, 0.f);
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

    // Conductance between the bottom layer and the soil surface. The soil has its own roughness sublayer, so this uses the ground boundary-layer conductance of Kustas and Norman (1999) evaluated at the wind speed near the soil surface rather
    // than the within-canopy turbulent diffusivity.
    bool soil_node_present = !canopy_airspace_ground_UUIDs.empty();
    float wind_speed_soil = wind_speed_canopy_top * std::exp(-attenuation_coefficient);
    float g_soil = 0.166f + 0.5f * wind_speed_soil;

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

    // Seed the primitive data that the surface energy balance reads on its first pass.
    for (size_t u = 0; u < Ncanopy; u++) {
        uint layer = canopy_airspace_layer_index.at(u);
        context->setPrimitiveData(canopy_airspace_UUIDs.at(u), "air_temperature", T_ac.at(layer));
        context->setPrimitiveData(canopy_airspace_UUIDs.at(u), "air_humidity", std::fmin(x_ac.at(layer) * Patm / esat_Pa(T_ac.at(layer)), 1.f));
    }
    if (soil_node_present) {
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_temperature", T_ac.front());
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_humidity", std::fmin(x_ac.front() * Patm / esat_Pa(T_ac.front()), 1.f));
        context->setPrimitiveData(canopy_airspace_ground_UUIDs, "wind_speed", wind_speed_soil);
    }

    // -- Outer fixed-point iteration between the surface energy balance and the airspace state -- //

    // Ground area basis for the leaf area weighting. The leaf area index relates total one-sided leaf area to ground area, so the ground area follows from the total canopy area.
    float total_canopy_area = context->sumPrimitiveSurfaceArea(canopy_airspace_UUIDs);
    float ground_area = total_canopy_area / canopy_airspace_LAI;

    std::vector<float> T_ac_previous(Nlayers);
    std::vector<float> x_ac_previous(Nlayers);

    // Tridiagonal system coefficients: sub-diagonal, diagonal, super-diagonal, and right-hand side.
    std::vector<float> sub(Nlayers), diag(Nlayers), super(Nlayers), rhs(Nlayers);

    uint iteration = 0;
    bool converged = false;
    while (iteration < canopy_airspace_max_iterations && !converged) {

        T_ac_previous = T_ac;
        x_ac_previous = x_ac;

        // 1) Update leaf temperatures given the current airspace state.
        evaluateSurfaceEnergyBalance(UUIDs, 0.f);

        // The airspace solution weights each leaf by its boundary-layer conductance, so verify once that the surface energy balance produced it rather than testing every primitive on every iteration.
        if (iteration == 0) {
            for (uint UUID: canopy_airspace_UUIDs) {
                if (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out")) {
                    helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): Primitive data 'boundarylayer_conductance_out' was not produced by the surface energy balance for primitive UUID " + std::to_string(UUID) +
                                         ", so leaf contributions to the canopy airspace cannot be weighted. Check that this primitive is included in the set of UUIDs passed to run().");
                }
            }
        }

        // 2) Reduce leaf temperature and boundary-layer conductance onto the airspace layers, weighting each leaf by its boundary-layer conductance and its area fraction of the ground.
        std::vector<float> sum_gA_T(Nlayers, 0.f);
        std::vector<float> sum_gA(Nlayers, 0.f);
        std::vector<float> sum_gA_e(Nlayers, 0.f);
        std::vector<float> sum_gA_moisture(Nlayers, 0.f);

        for (size_t u = 0; u < Ncanopy; u++) {
            uint UUID = canopy_airspace_UUIDs.at(u);
            uint layer = canopy_airspace_layer_index.at(u);

            float T_leaf;
            context->getPrimitiveData(UUID, "temperature", T_leaf);

            float g_bl;
            context->getPrimitiveData(UUID, "boundarylayer_conductance_out", g_bl);

            float area_fraction = context->getPrimitiveArea(UUID) / ground_area;
            float weight = g_bl * area_fraction;

            sum_gA.at(layer) += weight;
            sum_gA_T.at(layer) += weight * T_leaf;

            // Water vapor leaves the substomatal airspace through the stomata and the boundary layer in series, so the moisture source is weighted by the total conductance to moisture rather than by the boundary-layer conductance alone. A leaf
            // with closed stomata contributes no moisture to the canopy airspace.
            float gS;
            if (context->doesPrimitiveDataExist(UUID, "moisture_conductance") && context->getPrimitiveDataType("moisture_conductance") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(UUID, "moisture_conductance", gS);
            } else {
                gS = gS_default;
            }

            // 'boundarylayer_conductance_out' is the conductance summed over both faces of a two-sided primitive, whereas water vapor leaves through the stomata-bearing faces only. The sidedness partition below is the same expression the
            // surface energy balance uses to compute the latent flux, so that the moisture weighted into the airspace matches the water the leaf actually transpired.
            float stomatal_sidedness = 0.f;
            uint twosided_flag = context->getPrimitiveTwosidedFlag(UUID, 1);
            if (twosided_flag != 0) {
                if (context->doesPrimitiveDataExist(UUID, "stomatal_sidedness") && context->getPrimitiveDataType("stomatal_sidedness") == helios::HELIOS_TYPE_FLOAT) {
                    context->getPrimitiveData(UUID, "stomatal_sidedness", stomatal_sidedness);
                } else if (context->doesPrimitiveDataExist(UUID, "evaporating_faces") && context->getPrimitiveDataType("evaporating_faces") == helios::HELIOS_TYPE_UINT) {
                    uint evaporating_faces;
                    context->getPrimitiveData(UUID, "evaporating_faces", evaporating_faces);
                    stomatal_sidedness = (evaporating_faces == 2) ? 0.5f : 0.f;
                }
            }

            float g_M = 0.f;
            if (g_bl != 0.f && gS != 0.f) {
                g_M = moisture_conductance_ratio * g_bl * gS * (stomatal_sidedness / (moisture_conductance_ratio * g_bl + gS * stomatal_sidedness) + (1.f - stomatal_sidedness) / (moisture_conductance_ratio * g_bl + gS * (1.f - stomatal_sidedness)));
            }
            float moisture_weight = g_M * area_fraction;

            // The substomatal airspace is assumed saturated at the leaf surface temperature.
            sum_gA_moisture.at(layer) += moisture_weight;
            sum_gA_e.at(layer) += moisture_weight * esat_Pa(T_leaf) / Patm;
        }

        // 3) Soil surface temperature, area-weighted over the ground primitives.
        float T_soil = air_temperature_reference;
        if (soil_node_present) {
            float soil_area_weighted_sum;
            context->calculatePrimitiveDataAreaWeightedSum(canopy_airspace_ground_UUIDs, "temperature", soil_area_weighted_sum);
            float soil_area = context->sumPrimitiveSurfaceArea(canopy_airspace_ground_UUIDs);
            if (soil_area <= 0.f) {
                helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): The ground primitives given to enableCanopyAirspaceModel() have zero total surface area, so the soil node temperature cannot be determined.");
            }
            T_soil = soil_area_weighted_sum / soil_area;
            // A missing 'temperature' label produces an area-weighted sum of zero rather than an error, which would drive the bottom layer toward absolute zero.
            if (T_soil < 100.f) {
                helios_runtime_error("ERROR (EnergyBalanceModel::evaluateCanopyAirspace): The area-weighted soil surface temperature evaluated to " + std::to_string(T_soil) +
                                     " K, which is not physical. Check that primitive data 'temperature' is set in Kelvin for every ground primitive given to enableCanopyAirspaceModel().");
            }
        }

        // 4) Solve the tridiagonal system for layer air temperature. Layer 0 is the bottom layer, exchanging with the soil below; layer (Nlayers-1) is the top layer, exchanging with the reference air above.
        for (uint j = 0; j < Nlayers; j++) {
            // g_layer[j] is the conductance across the boundary between layer j and layer j+1, so the link below layer j is g_layer[j-1].
            float g_below = (j == 0) ? (soil_node_present ? g_soil : 0.f) : g_layer.at(j - 1);
            float g_above = (j == Nlayers - 1) ? g_a : g_layer.at(j);

            sub.at(j) = (j == 0) ? 0.f : -g_layer.at(j - 1);
            super.at(j) = (j == Nlayers - 1) ? 0.f : -g_layer.at(j);
            diag.at(j) = sum_gA.at(j) + g_below + g_above;
            rhs.at(j) = sum_gA_T.at(j);

            if (j == 0 && soil_node_present) {
                rhs.at(j) += g_soil * T_soil;
            }
            if (j == Nlayers - 1) {
                rhs.at(j) += g_a * air_temperature_reference;
            }
        }
        solveTridiagonal(sub, diag, super, rhs, T_ac);

        // 5) Solve the tridiagonal system for layer moisture mole fraction. Evaporation from the soil surface is assumed to be zero, so the bottom layer has no moisture exchange with the soil.
        // Turbulent transport moves heat and water vapor with the same eddies, so the layer-to-layer and layer-to-reference conductances are the same for both. The ratio of vapor to heat diffusivity applies only to the molecular leaf
        // boundary layer, and is already carried in the leaf weighting above.
        for (uint j = 0; j < Nlayers; j++) {
            float g_above = (j == Nlayers - 1) ? g_a : g_layer.at(j);
            float g_below = (j == 0) ? 0.f : g_layer.at(j - 1);

            sub.at(j) = (j == 0) ? 0.f : -g_layer.at(j - 1);
            super.at(j) = (j == Nlayers - 1) ? 0.f : -g_layer.at(j);
            diag.at(j) = sum_gA_moisture.at(j) + g_below + g_above;
            rhs.at(j) = sum_gA_e.at(j);

            if (j == Nlayers - 1) {
                rhs.at(j) += g_a * moisture_reference;
            }
        }
        solveTridiagonal(sub, diag, super, rhs, x_ac);

        // 6) Write the updated airspace state back to the primitives that the surface energy balance reads.
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
        std::vector<float> humidity_ac(Nlayers);
        for (uint j = 0; j < Nlayers; j++) {
            humidity_ac.at(j) = std::fmin(std::fmax(x_ac.at(j) * Patm / esat_Pa(T_ac.at(j)), 0.f), 1.f);
        }

        for (size_t u = 0; u < Ncanopy; u++) {
            uint layer = canopy_airspace_layer_index.at(u);
            context->setPrimitiveData(canopy_airspace_UUIDs.at(u), "air_temperature", T_ac.at(layer));
            context->setPrimitiveData(canopy_airspace_UUIDs.at(u), "air_humidity", humidity_ac.at(layer));
        }

        // The soil exchanges heat with the lowest canopy layer, so its own energy balance must be driven by that layer's air rather than by the air above the canopy. The wind speed near the soil surface is the one already used to form the
        // soil conductance.
        if (soil_node_present) {
            context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_temperature", T_ac.front());
            context->setPrimitiveData(canopy_airspace_ground_UUIDs, "air_humidity", humidity_ac.front());
            context->setPrimitiveData(canopy_airspace_ground_UUIDs, "wind_speed", wind_speed_soil);
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

        iteration++;
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

float EnergyBalanceModel::calculateAerodynamicResistance(float wind_speed_reference_m_s) const {

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

    return ln_momentum * ln_heat / (von_Karman_constant * von_Karman_constant * wind_speed_reference_m_s);
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
                                              float Tprev) {

    // Outgoing emission flux
    float Rout = float(Nsides) * eps * 5.67e-8F * T * T * T * T;

    // Sensible heat flux
    float QH = cp_air_mol * gH * (T - Ta);

    // Latent heat flux
    float es = 611.0f * expf(17.502f * (T - 273.f) / (T - 273.f + 240.97f));
    // A zero boundary-layer conductance (gH) or zero stomatal conductance (gS) means there is no vapor
    // pathway, so gM is zero. Guarding both cases also avoids a 0/0 in the series-conductance expression
    // when gH == 0 and the stomatal sidedness makes one of the denominators vanish.
    float gM = 0.f;
    if (gH != 0.f && gS != 0.f) {
        gM = 1.08f * gH * gS * (stomatal_sidedness / (1.08f * gH + gS * stomatal_sidedness) + (1.f - stomatal_sidedness) / (1.08f * gH + gS * (1.f - stomatal_sidedness)));
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

        // Other fluxes
        if (context->doesPrimitiveDataExist(p, "other_surface_flux") && context->getPrimitiveDataType("other_surface_flux") == helios::HELIOS_TYPE_FLOAT) {
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
                context->setPrimitiveData(UUID, "storage_flux", storage);
            }
        }
    }
}
