/** \file "EnergyBalanceModel.h" Primary header file for energy balance model.

    Copyright (C) 2016-2026 Brian Bailey

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
    static int selfTest(int argc = 0, char **argv = nullptr);

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

    //! Enable the canopy airspace model, which resolves within-canopy air temperature and humidity from a resistance network
    /**
     * This model computes the within-canopy (aerodynamic) air temperature \f$T_{ac}\f$ and vapor pressure \f$e_{ac}\f$ that surround leaves, allowing the canopy to feed back on the air that drives its own transpiration. The canopy airspace is
     * discretized into \p num_layers vertical layers of equal leaf area index, each of which exchanges sensible heat and water vapor with the leaves it contains, with its neighboring layers, and (for the bottom and top layers) with the soil
     * surface and the above-canopy reference air. Setting \p num_layers to 1 yields a single within-canopy node whose neighbors are the reference air above and the soil below.
     *
     * Unlike \ref enableAirEnergyBalance(), this model imposes the measured above-canopy air temperature and humidity as a fixed boundary condition at the canopy top rather than evolving a prognostic atmospheric boundary layer. It is therefore
     * appropriate for canopies subject to advection and mesoscale forcing (e.g., an orchard block) where the assumption of an infinite horizontal canopy does not hold.
     *
     * When enabled, \ref run() iterates the surface energy balance and the airspace solution to convergence, since leaf temperature and the airspace state are mutually dependent. Because the model solves for a steady state, it is not
     * compatible with the dynamic (non-steady-state) forms of \ref run() that take a timestep argument.
     *
     * This routine sets primitive data 'air_temperature', 'air_humidity', and 'wind_speed' for all primitives in \p canopy_UUIDs, and adds 'boundarylayer_conductance_out' to the optional outputs because the airspace solution requires it.
     *
     * The above-canopy boundary condition is read from global data 'air_temperature_reference', 'air_humidity_reference', and 'wind_speed_reference' if present, and otherwise falls back to this model's default values.
     *
     * \param[in] canopy_UUIDs Universal unique identifiers for canopy (leaf) primitives that exchange heat and moisture with the canopy airspace.
     * \param[in] ground_UUIDs Universal unique identifiers for ground primitives forming the soil node beneath the canopy. May be empty, in which case there is no exchange with the soil surface.
     * \param[in] canopy_height_m Height of the canopy in meters.
     * \param[in] reference_height_m Height at which the above-canopy air temperature, humidity, and wind speed are measured in meters. Must be greater than the canopy height.
     * \param[in] leaf_area_index One-sided leaf area index of the canopy on a ground-area basis (m² leaf area per m² ground area).
     * \param[in] num_layers Number of vertical canopy airspace layers of equal leaf area index. A value of 1 gives a single within-canopy node.
     */
    void enableCanopyAirspaceModel(const std::vector<uint> &canopy_UUIDs, const std::vector<uint> &ground_UUIDs, float canopy_height_m, float reference_height_m, float leaf_area_index, uint num_layers);

    //! Disable the canopy airspace model
    /**
     * Subsequent calls to \ref run() will perform a single surface energy balance pass using whatever 'air_temperature' and 'air_humidity' primitive data are currently set.
     */
    void disableCanopyAirspaceModel();

    //! Set convergence criteria for the canopy airspace model iteration
    /**
     * \param[in] tolerance_K Convergence tolerance for canopy airspace temperature in Kelvin. Iteration stops when the maximum change in any layer's air temperature falls below this value. Default is 0.01 K.
     * \param[in] max_iterations Maximum number of iterations of the coupled surface energy balance and airspace solution. Default is 50.
     */
    void setCanopyAirspaceConvergence(float tolerance_K, uint max_iterations);

    //! Add optional output primitive data values to the Context
    /**
     * \param[in] label Name of primitive data (e.g., vapor_pressure_deficit)
     */
    void optionalOutputPrimitiveData(const char *label);

#ifdef HELIOS_CUDA_AVAILABLE
    //! Enable GPU acceleration for energy balance calculations
    /**
     * Attempts to enable GPU acceleration. If GPU is not available at runtime,
     * this will have no effect and CPU mode will be used.
     */
    void enableGPUAcceleration();

    //! Disable GPU acceleration and force CPU mode
    /**
     * Forces the use of OpenMP CPU implementation even if GPU is available.
     * Useful for testing, benchmarking, or when CPU performance is preferred.
     */
    void disableGPUAcceleration();

    //! Check if GPU acceleration is currently enabled
    /**
     * \return True if GPU acceleration is enabled and available, false otherwise
     */
    bool isGPUAccelerationEnabled() const;
#endif

    //! Print a report detailing usage of default input values for all primitives in the Context
    void printDefaultValueReport() const;

    //! Print a report detailing usage of default input values based on a subset of primitive UUIDs
    /**
     * \param[in] UUIDs Universal unique identifiers for report
     */
    void printDefaultValueReport(const std::vector<uint> &UUIDs) const;

private:
    void evaluateSurfaceEnergyBalance(const std::vector<uint> &UUIDs, float dt);

    //! Solve the coupled surface energy balance and canopy airspace state to convergence
    /**
     * \param[in] UUIDs Universal unique identifiers for primitives included in the surface energy balance.
     */
    void evaluateCanopyAirspace(const std::vector<uint> &UUIDs);

    //! Assign canopy primitives to vertical airspace layers containing equal leaf area
    /**
     * Layer 0 is the bottom layer and layer (num_layers-1) is the top layer. Primitives are sorted by the height of their centroid above the base of the canopy and divided such that each layer contains an equal one-sided leaf area.
     */
    void assignCanopyAirspaceLayers();

    //! Write the within-canopy wind speed profile to primitive data 'wind_speed'
    /**
     * Applies the exponential within-canopy profile of Cionco (1972), U(z) = U(h_c)*exp[-a*(1-z/h_c)], where the attenuation coefficient a is one half of the leaf area index.
     * \param[in] wind_speed_canopy_top_m_s Wind speed at the top of the canopy in m/s.
     */
    void updateCanopyWindProfile(float wind_speed_canopy_top_m_s);

    //! Calculate aerodynamic resistance above the canopy following Perrier (1975)
    /**
     * \param[in] wind_speed_reference_m_s Wind speed at the reference height in m/s.
     * \return Aerodynamic resistance between the canopy top and the reference height in s/m.
     */
    [[nodiscard]] float calculateAerodynamicResistance(float wind_speed_reference_m_s) const;

    //! Flag indicating whether the canopy airspace model is enabled
    bool canopy_airspace_enabled = false;

    //! Canopy primitives exchanging heat and moisture with the canopy airspace
    std::vector<uint> canopy_airspace_UUIDs;

    //! Ground primitives forming the soil node beneath the canopy airspace
    std::vector<uint> canopy_airspace_ground_UUIDs;

    //! Number of vertical canopy airspace layers
    uint canopy_airspace_layer_count = 1;

    //! One-sided leaf area index of the canopy on a ground-area basis
    float canopy_airspace_LAI = 0.f;

    //! Convergence tolerance for canopy airspace temperature, Kelvin
    float canopy_airspace_tolerance_K = 0.01f;

    //! Maximum number of canopy airspace iterations
    uint canopy_airspace_max_iterations = 50;

    //! Index of the airspace layer containing each primitive in canopy_airspace_UUIDs
    std::vector<uint> canopy_airspace_layer_index;

    //! Height of the base of the canopy above which layers are assigned, meters
    float canopy_airspace_base_height_m = 0.f;

    //! Height of the midpoint of each canopy airspace layer above the canopy base, meters
    std::vector<float> canopy_airspace_layer_midpoint_m;

    //! Height of the upper boundary of each canopy airspace layer above the canopy base, meters
    std::vector<float> canopy_airspace_layer_top_m;

    //! Flag indicating whether layer assignment is current
    bool canopy_airspace_layers_assigned = false;

#ifdef HELIOS_CUDA_AVAILABLE
    //! GPU implementation using CUDA
    void evaluateSurfaceEnergyBalance_GPU(const std::vector<uint> &UUIDs, float dt);

    //! Check if GPU is available at runtime and initialize
    void initializeGPUAcceleration();
#endif

    //! CPU implementation using OpenMP
    void evaluateSurfaceEnergyBalance_CPU(const std::vector<uint> &UUIDs, float dt);

    //! Copy of a pointer to the context
    helios::Context *context;

#ifdef HELIOS_CUDA_AVAILABLE
    //! Flag to enable/disable GPU acceleration
    bool gpu_acceleration_enabled = false;
#endif

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

    float canopy_height_m = 0.f;
    float reference_height_m = 0.f;

    //! Flag controlling whether messages are printed to standard output
    bool message_flag;

    //! Names of radiation bands to be included in absorbed all-wave radiation flux
    std::vector<std::string> radiation_bands;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;
};

#endif
