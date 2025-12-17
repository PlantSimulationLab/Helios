/** \file "EnergyBalanceModel.cpp" Energy balance model plugin declarations.

    Copyright (C) 2016-2025 Brian Bailey

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
    gpu_acceleration_enabled = true;  // Try to use GPU by default

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

    evaluateSurfaceEnergyBalance(UUIDs, dt);

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
            std::cout << "INFO (EnergyBalanceModel): CUDA runtime unavailable ("
                      << cudaGetErrorString(err)
                      << "). Using OpenMP CPU implementation." << std::endl;
        }
        gpu_acceleration_enabled = false;
        return;
    }

    // GPU available
    if (message_flag) {
        std::cout << "INFO (EnergyBalanceModel): GPU acceleration enabled ("
                  << deviceCount << " device(s) found)." << std::endl;
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

    air_energy_balance_enabled = true;
    this->canopy_height_m = canopy_height_m;
    this->reference_height_m = reference_height_m;
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
static inline float evaluateEnergyBalance_CPU(float T, float R, float Qother, float eps,
                                               float Ta, float ea, float pressure, float gH,
                                               float gS, uint Nsides, float stomatal_sidedness,
                                               float heatcapacity, float surfacehumidity,
                                               float dt, float Tprev) {

    // Outgoing emission flux
    float Rout = float(Nsides) * eps * 5.67e-8F * T * T * T * T;

    // Sensible heat flux
    float QH = cp_air_mol * gH * (T - Ta);

    // Latent heat flux
    float es = 611.0f * expf(17.502f * (T - 273.f) / (T - 273.f + 240.97f));
    float gM = 1.08f * gH * gS * (stomatal_sidedness / (1.08f * gH + gS * stomatal_sidedness) +
                                  (1.f - stomatal_sidedness) / (1.08f * gH + gS * (1.f - stomatal_sidedness)));
    if (gH == 0 && gS == 0) {
        gM = 0;
    }

    float QL = gM * lambda_mol * (es - ea * surfacehumidity) / pressure;

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
    std::vector<float> emissivity(Nprimitives);

    for (size_t u = 0; u < Nprimitives; u++) {
        emissivity.at(u) = 1.f;
    }

    // Accumulate radiation across bands
    for (int b = 0; b < radiation_bands.size(); b++) {
        for (size_t u = 0; u < Nprimitives; u++) {
            size_t p = UUIDs.at(u);

            char str[50];
            snprintf(str, sizeof(str), "radiation_flux_%s", radiation_bands.at(b).c_str());
            if (!context->doesPrimitiveDataExist(p, str)) {
                helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation was found in the context for band " +
                                    std::string(radiation_bands.at(b)) +
                                    ". Did you run the radiation model for this band?");
            } else if (context->getPrimitiveDataType(str) != helios::HELIOS_TYPE_FLOAT) {
                helios_runtime_error("ERROR (EnergyBalanceModel::run): Radiation primitive data for band " +
                                    std::string(radiation_bands.at(b)) +
                                    " does not have the correct type of 'float'");
            }
            float R;
            context->getPrimitiveData(p, str, R);
            Rn.at(u) += R;

            snprintf(str, sizeof(str), "emissivity_%s", radiation_bands.at(b).c_str());
            if (context->doesPrimitiveDataExist(p, str) && context->getPrimitiveDataType(str) == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(p, str, emissivity.at(u));
            } else {
                warnings.addWarning("missing_emissivity",
                    "Primitive data 'emissivity_" + std::string(radiation_bands.at(b)) +
                    "' not set, using default (1.0)");
            }
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
        if (context->doesPrimitiveDataExist(p, "temperature") &&
            context->getPrimitiveDataType("temperature") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "temperature", To[u]);
        } else {
            To[u] = temperature_default;
            warnings.addWarning("missing_surface_temperature",
                "Primitive data 'temperature' not set, using default (" +
                std::to_string(temperature_default) + " K)");
        }
        if (To[u] == 0) {
            To[u] = 300;
        }

        // Air temperature
        if (context->doesPrimitiveDataExist(p, "air_temperature") &&
            context->getPrimitiveDataType("air_temperature") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "air_temperature", Ta[u]);
            if (Ta[u] < 250.f) {
                warnings.addWarning("air_temperature_likely_celsius",
                    "Value of " + std::to_string(Ta[u]) +
                    " in 'air_temperature' is very small - should be in Kelvin, using default (" +
                    std::to_string(air_temperature_default) + " K)");
                Ta[u] = air_temperature_default;
            }
        } else {
            Ta[u] = air_temperature_default;
            warnings.addWarning("missing_air_temperature",
                "Primitive data 'air_temperature' not set, using default (" +
                std::to_string(air_temperature_default) + " K)");
        }

        // Air relative humidity
        float hr;
        if (context->doesPrimitiveDataExist(p, "air_humidity") &&
            context->getPrimitiveDataType("air_humidity") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "air_humidity", hr);
            if (hr > 1.f) {
                warnings.addWarning("air_humidity_out_of_range_high",
                    "Value of " + std::to_string(hr) +
                    " in 'air_humidity' > 1.0 - should be fractional [0,1], using default (" +
                    std::to_string(air_humidity_default) + ")");
                hr = air_humidity_default;
            } else if (hr < 0.f) {
                warnings.addWarning("air_humidity_out_of_range_low",
                    "Value of " + std::to_string(hr) +
                    " in 'air_humidity' < 0.0 - should be fractional [0,1], using default (" +
                    std::to_string(air_humidity_default) + ")");
                hr = air_humidity_default;
            }
        } else {
            hr = air_humidity_default;
            warnings.addWarning("missing_air_humidity",
                "Primitive data 'air_humidity' not set, using default (" +
                std::to_string(air_humidity_default) + ")");
        }

        // Air vapor pressure
        float esat = esat_Pa(Ta[u]);
        ea[u] = hr * esat;

        // Air pressure
        if (context->doesPrimitiveDataExist(p, "air_pressure") &&
            context->getPrimitiveDataType("air_pressure") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "air_pressure", pressure[u]);
            if (pressure[u] < 10000.f) {
                if (message_flag) {
                    std::cout << "WARNING (EnergyBalanceModel::run): Value of " << pressure[u]
                              << " given in 'air_pressure' primitive data is very small. "
                              << "Values should be given in units of Pascals. Assuming default value of "
                              << pressure_default << std::endl;
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
        if (Nsides[u] == 2 && context->doesPrimitiveDataExist(p, "stomatal_sidedness") &&
            context->getPrimitiveDataType("stomatal_sidedness") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "stomatal_sidedness", stomatal_sidedness[u]);
        } else if (Nsides[u] == 2 && context->doesPrimitiveDataExist(p, "evaporating_faces") &&
                   context->getPrimitiveDataType("evaporating_faces") == helios::HELIOS_TYPE_UINT) {
            uint flag;
            context->getPrimitiveData(p, "evaporating_faces", flag);
            if (flag == 1) {
                stomatal_sidedness[u] = 0.f;
            } else if (flag == 2) {
                stomatal_sidedness[u] = 0.5f;
            }
        }

        // Boundary-layer conductance to heat
        if (context->doesPrimitiveDataExist(p, "boundarylayer_conductance") &&
            context->getPrimitiveDataType("boundarylayer_conductance") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "boundarylayer_conductance", gH[u]);
        } else {
            // Wind speed
            float U;
            if (context->doesPrimitiveDataExist(p, "wind_speed") &&
                context->getPrimitiveDataType("wind_speed") == helios::HELIOS_TYPE_FLOAT) {
                context->getPrimitiveData(p, "wind_speed", U);
            } else {
                U = wind_speed_default;
                warnings.addWarning("missing_wind_speed",
                    "Primitive data 'wind_speed' not set, using default (" +
                    std::to_string(wind_speed_default) + " m/s)");
            }

            // Characteristic size of primitive
            float L;
            if (context->doesPrimitiveDataExist(p, "object_length") &&
                context->getPrimitiveDataType("object_length") == helios::HELIOS_TYPE_FLOAT) {
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
        if (context->doesPrimitiveDataExist(p, "moisture_conductance") &&
            context->getPrimitiveDataType("moisture_conductance") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "moisture_conductance", gS[u]);
        } else {
            gS[u] = gS_default;
        }

        // Other fluxes
        if (context->doesPrimitiveDataExist(p, "other_surface_flux") &&
            context->getPrimitiveDataType("other_surface_flux") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "other_surface_flux", Qother[u]);
        } else {
            Qother[u] = Qother_default;
        }

        // Object heat capacity
        if (context->doesPrimitiveDataExist(p, "heat_capacity") &&
            context->getPrimitiveDataType("heat_capacity") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "heat_capacity", heatcapacity[u]);
        } else {
            heatcapacity[u] = heatcapacity_default;
        }

        // Surface humidity
        if (context->doesPrimitiveDataExist(p, "surface_humidity") &&
            context->getPrimitiveDataType("surface_humidity") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(p, "surface_humidity", surfacehumidity[u]);
        } else {
            surfacehumidity[u] = surface_humidity_default;
        }

        // Emissivity
        eps[u] = emissivity.at(u);

        // Net absorbed radiation
        R[u] = Rn.at(u);
    }

    // Report all accumulated warnings
    warnings.report(std::cout, true);  // true = compact mode

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
    for (int p = 0; p < (int)Nprimitives; p++) {

        // Secant method parameters
        const float err_max = 0.0001f;
        const uint max_iter = 100;

        // Initial guesses
        float T_old_old = To[p];
        float T_old = T_old_old;
        T_old_old = 400.f;

        // Evaluate residual at initial guesses
        float resid_old = evaluateEnergyBalance_CPU(T_old, R[p], Qother[p], eps[p], Ta[p], ea[p],
                                                     pressure[p], gH[p], gS[p], Nsides[p],
                                                     stomatal_sidedness[p], heatcapacity[p],
                                                     surfacehumidity[p], dt, To[p]);
        float resid_old_old = evaluateEnergyBalance_CPU(T_old_old, R[p], Qother[p], eps[p], Ta[p], ea[p],
                                                         pressure[p], gH[p], gS[p], Nsides[p],
                                                         stomatal_sidedness[p], heatcapacity[p],
                                                         surfacehumidity[p], dt, To[p]);

        // Secant method iteration
        float T_new;
        float resid = 100;
        float err = resid;
        uint iter = 0;
        while (err > err_max && iter < max_iter) {

            if (resid_old == resid_old_old) {
                err = 0;
                break;
            }

            T_new = fabs((T_old_old * resid_old - T_old * resid_old_old) / (resid_old - resid_old_old));

            resid = evaluateEnergyBalance_CPU(T_new, R[p], Qother[p], eps[p], Ta[p], ea[p],
                                              pressure[p], gH[p], gS[p], Nsides[p],
                                              stomatal_sidedness[p], heatcapacity[p],
                                              surfacehumidity[p], dt, To[p]);

            resid_old_old = resid_old;
            resid_old = resid;

            err = fabs(T_old - T_old_old) / fabs(T_old_old);

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
        std::cout << "WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge for "
                  << total_failures << " primitives." << std::endl;
    }
    #else
    if (convergence_failures > 0 && message_flag) {
        std::cout << "WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge for "
                  << convergence_failures << " primitives." << std::endl;
    }
    #endif

    //---- Write results back to Context ----//

    for (uint u = 0; u < Nprimitives; u++) {
        size_t UUID = UUIDs.at(u);

        if (T[u] != T[u]) {  // NaN check
            T[u] = temperature_default;
        }

        context->setPrimitiveData(UUID, "temperature", T[u]);

        float QH = cp_air_mol * gH[u] * (T[u] - Ta[u]);
        context->setPrimitiveData(UUID, "sensible_flux", QH);

        float es = esat_Pa(T[u]);
        float gM = 1.08f * gH[u] * gS[u] *
                   (stomatal_sidedness[u] / (1.08f * gH[u] + gS[u] * stomatal_sidedness[u]) +
                    (1.f - stomatal_sidedness[u]) / (1.08f * gH[u] + gS[u] * (1.f - stomatal_sidedness[u])));
        if (gH[u] == 0 && gS[u] == 0) {
            gM = 0;
        }
        float QL = lambda_mol * gM * (es - ea[u]) / pressure[u];
        context->setPrimitiveData(UUID, "latent_flux", QL);

        for (const auto &data_label : output_prim_data) {
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
