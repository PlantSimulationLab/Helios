/** \file "EnergyBalanceModel.h" Primary header file for energy balance model.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef ENERGYBALANCEMODEL
#define ENERGYBALANCEMODEL

#include "Context.h"

// -- constants -- //

//! specific heat capacity of air at constant pressure, J mol⁻¹ K⁻¹
constexpr float cp_air_mol = 29.25f;
//! latent heat of vaporization of water at 300 K, J mol⁻¹
constexpr float lambda_mol = 44000.f;
//! von Karman constant, dimensionless
constexpr float von_Karman_constant = 0.41f;
//! Gas constant, J mol⁻¹ K⁻¹
constexpr float R = 8.314462618f;

inline float esat_Pa(float T_K) {
    float Tc = T_K - 273.15f;
    // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in Kelvin, and result is in Pascals
    return 611.0f * expf(17.502f * Tc / (Tc + 240.97f));
}

//! Energy balance model class
/** This model computes surface temperatures based on a local energy balance */
class EnergyBalanceModel {
public:
    //! Constructor
    /**
     * \param[in] context Pointer to the Helios context
     */
    EnergyBalanceModel(helios::Context *context);

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    int selfTest();

    //! Enable standard output from this plug-in (default)
    void enableMessages();

    //! Disable standard output from this plug-in
    void disableMessages();

    //! Function to run the energy balance model for all primitives in the Context
    void run();

    //! Function to run the dynamic /non steady state energy balance model for one timestep of length "dt" seconds
    /**
     * \param[in] dt Time step in seconds.
     */
    void run(float dt);

    //! Function to run the energy balance model for a select set of primitives
    /**
     * \param[in] UUIDs Unique universal identifiers (UUIDs) for primitives that should be included in energy balance calculations. All other primitives will be skipped by the model.
     */
    void run(const std::vector<uint> &UUIDs);

    //! Function to run the energy balance model for a select set of primitives for one timestep of length "dt" seconds
    /**
     * \param[in] UUIDs  Unique universal identifiers (UUIDs) for primitives that should be included in energy balance calculations. All other primitives will be skipped by the model.
     * \param[in] dt Time step in seconds.
     */
    void run(const std::vector<uint> &UUIDs, float dt);

    //! Add the label of a radiation band in the RadiationModel plug-in that should be used in calculation of the absorbed all-wave radiation flux
    /**
     * \param[in] band Name of radiation band (e.g., PAR, NIR, LW, etc.)
     */
    void addRadiationBand(const char *band);

    //! Add the labels of radiation bands in the RadiationModel plug-in that should be used in calculation of the absorbed all-wave radiation flux
    /**
     * \param[in] bands Vector of names of radiation bands (e.g., PAR, NIR, LW, etc.)
     */
    void addRadiationBand(const std::vector<std::string> &bands);

    //! Enable the air energy balance model, which computes the average air temperature and water vapor mole fraction based on the energy balance of the air layer in the canopy
    /**
     * Calling this version of enableAirEnergyBalance() will compute the canopy height based on a bounding box of all primitives in the Context, or the UUIDs based to evaluateAirEnergyBalance().
     * It also assumes that the reference height for ambient air temperature, humidity, and wind speed is at the canopy top.
     * This routine sets primitive data 'air_temperature' and 'air_humidity' to the average air temperature and humidity in the canopy for all primitives.
     */
    void enableAirEnergyBalance();

    //! Enable the air energy balance model, which computes the average air temperature and water vapor mole fraction based on the energy balance of the air layer in the canopy
    /**
     * This routine sets primitive data 'air_temperature' and 'air_humidity' to the average air temperature and humidity in the canopy for all primitives.
     * \param[in] canopy_height_m Height of the canopy in meters.
     * \param[in] reference_height_m Height at which the ambient air temperature, humidity, and wind speed are measured in meters.
     */
    void enableAirEnergyBalance(float canopy_height_m, float reference_height_m);

    //! Advance the air energy balance over time for all primitives in the Context
    /**
     * \param[in] dt_sec Time step in seconds.
     * \param[in] time_advance_sec Total time to advance the model in seconds (T must be greater than or equal to dt).
     */
    void evaluateAirEnergyBalance(float dt_sec, float time_advance_sec);

    //! Advance the air energy balance over time for primitives specified by UUIDs
    /**
     * \param[in] UUIDs Universal unique identifiers for primitives that should be included in the air energy balance calculations.
     * \param[in] dt_sec Time step in seconds.
     * \param[in] time_advance_sec Total time to advance the model in seconds (T must be greater than or equal to dt).
     */
    void evaluateAirEnergyBalance(const std::vector<uint> &UUIDs, float dt_sec, float time_advance_sec);

    //! Add optional output primitive data values to the Context
    /**
     * \param[in] label Name of primitive data (e.g., vapor_pressure_deficit)
     */
    void optionalOutputPrimitiveData(const char *label);

    //! Print a report detailing usage of default input values for all primitives in the Context
    void printDefaultValueReport() const;

    //! Print a report detailing usage of default input values based on a subset of primitive UUIDs
    /**
     * \param[in] UUIDs Universal unique identifiers for report
     */
    void printDefaultValueReport(const std::vector<uint> &UUIDs) const;

private:
    void evaluateSurfaceEnergyBalance(const std::vector<uint> &UUIDs, float dt);

    //! Copy of a pointer to the context
    helios::Context *context;

    //! Default surface temperature if it was not specified in the context
    float temperature_default;

    //! Default wind speed if it was not specified in the context
    float wind_speed_default;

    //! Default air temperature if it was not specified in the context
    float air_temperature_default;

    //! Default air relative humidity if it was not specified in the context
    float air_humidity_default;

    //! Default air pressure if it was not specified in the context
    float pressure_default;

    //! Default stomatal conductance if it was not specified in the context
    float gS_default;

    //! Default heat capacity if it was not specified in the context
    float heatcapacity_default;

    //! Default "other" flux if it was not specified in the context
    float Qother_default;

    //! Default surface humidity if it was not specified in the context
    float surface_humidity_default;

    bool air_energy_balance_enabled;

    //! Dimensions of the canopy (x, y, z) in meters
    helios::vec3 canopy_dimensions;

    float canopy_height_m;
    float reference_height_m;

    //! Flag controlling whether messages are printed to standard output
    bool message_flag;

    //! Names of radiation bands to be included in absorbed all-wave radiation flux
    std::vector<std::string> radiation_bands;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;
};

#endif
