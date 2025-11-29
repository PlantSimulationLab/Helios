/** \file "SolarPosition.h" Primary header file for solar position model plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SOLARPOSITION
#define SOLARPOSITION

#include "Context.h"
#include <memory>

// Forward declaration is in PragueSkyModelInterface.h
#include "PragueSkyModelInterface.h"

class SolarPosition {
public:
    //! Solar position model default constructor. Initializes location based on the location set in the Context.
    /**
     * \param[in] context_ptr Pointer to the Helios context
     */
    explicit SolarPosition(helios::Context *context_ptr);

    //! Solar position model constructor
    /**
     * \param[in] context_ptr Pointer to the Helios context
     * \param[in] UTC_hrs Hours from Coordinated Universal Time (UTC) for location of interest.  Convention is that UTC is positive moving Westward.
     * \param[in] latitude_deg Latitude in degrees for location of interest.  Convention is latitude is positive for Northern hemisphere.
     * \param[in] longitude_deg Longitude in degrees for location of interest. Convention is longitude is positive for Western hemisphere.
     */
    SolarPosition(float UTC_hrs, float latitude_deg, float longitude_deg, helios::Context *context_ptr);

    //! Function to perform a self-test of model functions
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Get the approximate time of sunrise at the current location
    [[nodiscard]] helios::Time getSunriseTime() const;

    //! Get the approximate time of sunset at the current location
    [[nodiscard]] helios::Time getSunsetTime() const;

    //! Get the current sun elevation angle in radians for the current location. The sun angle is computed based on the current time and date set in the Helios context
    [[nodiscard]] float getSunElevation() const;

    //! Get the current sun zenithal angle in radians for the current location. The sun angle is computed based on the current time and date set in the Helios context
    [[nodiscard]] float getSunZenith() const;

    //! Get the current sun azimuthal angle in radians for the current location. The sun angle is computed based on the current time and date set in the Helios context
    [[nodiscard]] float getSunAzimuth() const;

    //! Get a unit vector pointing toward the sun for the current location. The sun angle is computed based on the current time and date set in the Helios context
    [[nodiscard]] helios::vec3 getSunDirectionVector() const;

    //! Get a spherical coordinate vector pointing toward the sun for the current location. The sun angle is computed based on the current time and date set in the Helios context
    [[nodiscard]] helios::SphericalCoord getSunDirectionSpherical() const;

    //! Override solar position calculation based on time in the Context by using a prescribed solar position
    /**
     * \param[in] sundirection SphericalCoord giving the direction of the sun
     */
    void setSunDirection(const helios::SphericalCoord &sundirection);

    //! Get the solar radiation flux perpendicular to the sun direction.
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (AOD at 500 nm). Typical values: 0.02 (very clear), 0.05 (clear), 0.1 (hazy)
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \return Global solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     * \deprecated Use setAtmosphericConditions() and the parameter-free getSolarFlux() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getSolarFlux() instead")]]
    [[nodiscard]] float getSolarFlux(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Get the photosynthetically active (PAR) component of solar radiation flux perpendicular to the sun direction.
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (AOD at 500 nm). Typical values: 0.02 (very clear), 0.05 (clear), 0.1 (hazy)
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \return Global solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     * \deprecated Use setAtmosphericConditions() and the parameter-free getSolarFluxPAR() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getSolarFluxPAR() instead")]]
    [[nodiscard]] float getSolarFluxPAR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Get the near-infrared (NIR) component of solar radiation flux perpendicular to the sun direction.
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (AOD at 500 nm). Typical values: 0.02 (very clear), 0.05 (clear), 0.1 (hazy)
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \return Global solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     * \deprecated Use setAtmosphericConditions() and the parameter-free getSolarFluxNIR() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getSolarFluxNIR() instead")]]
    [[nodiscard]] float getSolarFluxNIR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Get the fraction of solar radiation flux that is diffuse
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (AOD at 500 nm). Typical values: 0.02 (very clear), 0.05 (clear), 0.1 (hazy)
     * \return Fraction of global radiation that is diffuse
     * \deprecated Use setAtmosphericConditions() and the parameter-free getDiffuseFraction() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getDiffuseFraction() instead")]]
    [[nodiscard]] float getDiffuseFraction(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Calculate the ambient (sky) longwave radiation flux
    /**
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \return Ambient longwave flux in W/m^2
     * \note The longwave flux model is based on <a href="http://onlinelibrary.wiley.com/doi/10.1002/qj.49712253306/full">Prata (1996)</a>.
     * \deprecated Use setAtmosphericConditions() and the parameter-free getAmbientLongwaveFlux() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getAmbientLongwaveFlux() instead")]]
    [[nodiscard]] float getAmbientLongwaveFlux(float temperature_K, float humidity_rel) const;

    //! Calculate the turbidity value based on a timeseries of net radiation measurements
    /**
     * \param[in] timeseries_shortwave_flux_label_Wm2 Label of the timeseries variable in the Helios context that contains the net radiation flux measurements
     * \return Turbidity value
     * \note The net radiation flux measurements contained in the timeseries should be global shortwave radiation flux on a horizontal plane in W/m^2. The data should contain at least one day with clear sky conditions.
     */
    [[nodiscard]] float calibrateTurbidityFromTimeseries(const std::string &timeseries_shortwave_flux_label_Wm2) const;

    //! Enable calibration of solar flux and diffuse fraction when possibility of clouds are present against measured solar flux data
    /**
     * \param[in] timeseries_shortwave_flux_label_Wm2 Label for timeseries data field containing measured total shortwave flux data (W/m^2).
     */
    void enableCloudCalibration(const std::string &timeseries_shortwave_flux_label_Wm2);

    //! Disable calibration of solar flux and diffuse fraction for clouds
    void disableCloudCalibration();

    //! Set atmospheric conditions globally in the Context
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals (must be > 0)
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin (must be > 0)
     * \param[in] humidity_rel Air relative humidity near the ground surface (must be between 0 and 1)
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (β), representing aerosol optical depth (AOD) at 500 nm reference wavelength (must be >= 0). Typical values: 0.02 (very clear sky), 0.05 (clear sky), 0.1 (light haze), 0.2-0.3 (hazy), >0.4 (very hazy/polluted). <b>Note:</b> This is NOT Linke turbidity, which uses a different scale (typically 2-6).
     * \note This method sets global data in the Context with labels: atmosphere_pressure_Pa, atmosphere_temperature_K, atmosphere_humidity_rel, atmosphere_turbidity
     * \note Input values are validated. Invalid values will throw helios_runtime_error.
     */
    void setAtmosphericConditions(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity);

    //! Get atmospheric conditions from the Context
    /**
     * \param[out] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[out] temperature_K Air temperature near the ground surface in Kelvin
     * \param[out] humidity_rel Air relative humidity near the ground surface
     * \param[out] turbidity Ångström's aerosol turbidity coefficient (β), representing aerosol optical depth (AOD) at 500 nm
     * \note If atmospheric conditions have not been set via setAtmosphericConditions(), reasonable defaults are used: pressure=101325 Pa (1 atm), temperature=300 K (27°C), humidity=0.5 (50%), turbidity=0.02 (clear sky AOD)
     */
    void getAtmosphericConditions(float &pressure_Pa, float &temperature_K, float &humidity_rel, float &turbidity) const;

    //! Get the solar radiation flux perpendicular to the sun direction using atmospheric conditions from the Context.
    /**
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \note Atmospheric conditions must be set using \ref setAtmosphericConditions() before calling this method. If not set, default values are used with a warning.
     * \return Global solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     */
    [[nodiscard]] float getSolarFlux() const;

    //! Get the photosynthetically active (PAR) component of solar radiation flux perpendicular to the sun direction using atmospheric conditions from the Context.
    /**
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \note Atmospheric conditions must be set using \ref setAtmosphericConditions() before calling this method. If not set, default values are used with a warning.
     * \return PAR component of solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     */
    [[nodiscard]] float getSolarFluxPAR() const;

    //! Get the near-infrared (NIR) component of solar radiation flux perpendicular to the sun direction using atmospheric conditions from the Context.
    /**
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \note Atmospheric conditions must be set using \ref setAtmosphericConditions() before calling this method. If not set, default values are used with a warning.
     * \return NIR component of solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     */
    [[nodiscard]] float getSolarFluxNIR() const;

    //! Get the fraction of solar radiation flux that is diffuse using atmospheric conditions from the Context.
    /**
     * \note Atmospheric conditions must be set using \ref setAtmosphericConditions() before calling this method. If not set, default values are used with a warning.
     * \return Fraction of global radiation that is diffuse
     */
    [[nodiscard]] float getDiffuseFraction() const;

    //! Calculate the ambient (sky) longwave radiation flux using atmospheric conditions from the Context.
    /**
     * \note The longwave flux model is based on <a href="http://onlinelibrary.wiley.com/doi/10.1002/qj.49712253306/full">Prata (1996)</a>.
     * \note Atmospheric conditions must be set using \ref setAtmosphericConditions() before calling this method. If not set, default values are used with a warning.
     * \return Ambient longwave flux in W/m^2
     */
    [[nodiscard]] float getAmbientLongwaveFlux() const;

    //! Calculate direct beam solar spectrum and store in global data
    /**
     * \param[in] label User-defined label for storing spectral data in Context global data
     * \param[in] resolution_nm Wavelength resolution in nm (default: 1 nm). Must be >= 1 nm and <= 2300 nm.
     * \note Computes spectral irradiance normal to sun direction from 300-2600 nm using the SSolar-GOA model (Cachorro et al. 2022).
     * \note Stores result in Context global data as std::vector<helios::vec2> (wavelength_nm, W/m²/nm) with the specified label.
     */
    void calculateDirectSolarSpectrum(const std::string& label, float resolution_nm = 1.0f);

    //! Calculate diffuse solar spectrum and store in global data
    /**
     * \param[in] label User-defined label for storing spectral data in Context global data
     * \param[in] resolution_nm Wavelength resolution in nm (default: 1 nm). Must be >= 1 nm and <= 2300 nm.
     * \note Computes diffuse spectral irradiance on horizontal surface from 300-2600 nm using the SSolar-GOA model (Cachorro et al. 2022).
     * \note Stores result in Context global data as std::vector<helios::vec2> (wavelength_nm, W/m²/nm) with the specified label.
     */
    void calculateDiffuseSolarSpectrum(const std::string& label, float resolution_nm = 1.0f);

    //! Calculate global (total) solar spectrum and store in global data
    /**
     * \param[in] label User-defined label for storing spectral data in Context global data
     * \param[in] resolution_nm Wavelength resolution in nm (default: 1 nm). Must be >= 1 nm and <= 2300 nm.
     * \note Computes global spectral irradiance on horizontal surface from 300-2600 nm using the SSolar-GOA model (Cachorro et al. 2022).
     * \note Stores result in Context global data as std::vector<helios::vec2> (wavelength_nm, W/m²/nm) with the specified label.
     */
    void calculateGlobalSolarSpectrum(const std::string& label, float resolution_nm = 1.0f);

    // ===== Prague Sky Model =====

    //! Enable Prague sky model for atmospheric sky radiance computation
    /**
     * \note The Prague Sky Model computes physically-based sky radiance distributions accounting for Rayleigh and Mie scattering.
     * \note Uses the reduced dataset (360-1480 nm, ~27 MB) from plugins/solarposition/lib/prague_sky_model/PragueSkyModelReduced.dat
     * \note Once enabled, call updatePragueSkyModel() to compute and store spectral-angular parameters in Context global data.
     */
    void enablePragueSkyModel();

    //! Check if Prague sky model is enabled
    /**
     * \return True if Prague model has been enabled via enablePragueSkyModel()
     */
    [[nodiscard]] bool isPragueSkyModelEnabled() const;

    //! Update Prague sky model and store spectral-angular parameters in Context
    /**
     * \param[in] ground_albedo Ground reflectance [0-1] (default: 0.33 for vegetation)
     * \note Must call after solar position changes or atmospheric conditions change.
     * \note Reads turbidity from Context atmospheric conditions (set via setAtmosphericConditions). If not set, uses default 0.02 (clear sky).
     * \note Computationally intensive (~1100 model queries with OpenMP parallelization). Use lazy evaluation via pragueSkyModelNeedsUpdate() to avoid unnecessary updates.
     * \note Stores the following in Context global data:
     *   - "prague_sky_spectral_params": vec<float> of size 1350 (225 wavelengths × 6 parameters)
     *   - "prague_sky_sun_direction": vec3 sun direction
     *   - "prague_sky_visibility_km": float visibility in km
     *   - "prague_sky_ground_albedo": float ground albedo
     *   - "prague_sky_valid": int validity flag (1=valid, 0=invalid)
     */
    void updatePragueSkyModel(float ground_albedo = 0.33f);

    //! Check if Prague sky model update is needed based on changed conditions
    /**
     * \param[in] ground_albedo Current ground albedo
     * \param[in] sun_tolerance Threshold for sun direction changes (default: 0.01 ≈ 0.57°)
     * \param[in] turbidity_tolerance Relative threshold for turbidity (default: 0.02 = 2%)
     * \param[in] albedo_tolerance Threshold for albedo changes (default: 0.05 = 5%)
     * \return True if updatePragueSkyModel() should be called
     * \note This method enables lazy evaluation to avoid expensive Prague updates when conditions haven't changed significantly.
     * \note Reads turbidity from Context atmospheric conditions for comparison.
     */
    [[nodiscard]] bool pragueSkyModelNeedsUpdate(float ground_albedo = 0.33f,
                                                  float sun_tolerance = 0.01f,
                                                  float turbidity_tolerance = 0.02f,
                                                  float albedo_tolerance = 0.05f) const;

private:
    helios::Context *context;

    float UTC;
    float latitude;
    float longitude;

    bool issolarpositionoverridden = false;
    helios::SphericalCoord sun_direction;

    std::string cloudcalibrationlabel;

    // Prague Sky Model members
    std::unique_ptr<helios::PragueSkyModelInterface> prague_model;
    bool prague_enabled = false;

    // Cached values for Prague lazy evaluation
    helios::vec3 cached_sun_direction;
    float cached_turbidity = -1.0f;
    float cached_albedo = -1.0f;

    [[nodiscard]] helios::SphericalCoord calculateSunDirection(const helios::Time &time, const helios::Date &date) const;

    // Prague Sky Model helper methods
    void fitAngularParametersAtWavelength(float wavelength, float visibility_km,
                                           float albedo, const helios::vec3& sun_dir,
                                           float& L_zenith, float& circ_str,
                                           float& circ_width, float& horiz_bright,
                                           float& normalization);

    [[nodiscard]] float fitCircumsolarWidth(float L1, float L2, float h1, float h2,
                                            float gamma1, float gamma2) const;

    [[nodiscard]] float computeAngularNormalization(float circ_str, float circ_width,
                                                    float horiz_bright) const;

    [[nodiscard]] helios::vec3 rotateDirectionTowardZenith(const helios::vec3& dir, float angle_rad) const;

    [[nodiscard]] helios::vec3 getDirectionAwayFromSun(const helios::vec3& sun_dir, float zenith_angle_deg) const;

    void GueymardSolarModel(float pressure, float temperature, float humidity, float turbidity, float &Eb_PAR, float &Eb_NIR, float &fdiff) const;

    void applyCloudCalibration(float &R_calc_Wm2, float &fdiff_calc) const;

    static float turbidityResidualFunction(float turbidity, std::vector<float> &parameters, const void *a_solarpositionmodel);

    // ===== SSolar-GOA Spectral Solar Radiation Model =====

    //! Spectral data for SSolar-GOA model (TOA spectrum and absorption coefficients)
    struct SpectralData {
        std::vector<float> wavelengths_nm;      // 300-2600 nm (2301 points)
        std::vector<float> toa_irradiance;      // Top-of-atmosphere irradiance in W/m²/nm
        std::vector<float> h2o_coef;            // Water vapor absorption coefficient
        std::vector<float> h2o_exp;             // Water vapor absorption exponent
        std::vector<float> o3_xsec;             // Ozone cross section in cm²/molecule
        std::vector<float> o2_coef;             // Oxygen absorption coefficient

        //! Load spectral data from directory
        void loadFromDirectory(const std::string& data_path);

        //! Linear interpolation for wavelength data
        [[nodiscard]] static float interpolate(const std::vector<float>& x, const std::vector<float>& y, float x_val);
    };

    //! Calculate geometric factor (Earth-Sun distance correction) for a given Julian day
    [[nodiscard]] float calculateGeometricFactor(int julian_day) const;

    //! Calculate Rayleigh scattering transmittance (Bates formula + Sobolev approximation)
    void calculateRayleighTransmittance(
        const std::vector<float>& wavelengths_um,
        float mu0,
        float pressure_ratio,
        std::vector<float>& tdir,
        std::vector<float>& tglb,
        std::vector<float>& tdif,
        std::vector<float>& atm_albedo
    ) const;

    //! Calculate aerosol extinction transmittance (Ångström law + Ambartsumian solution)
    void calculateAerosolTransmittance(
        const std::vector<float>& wavelengths_um,
        float mu0,
        float alpha,
        float beta,
        float w0,
        float g,
        std::vector<float>& tdir,
        std::vector<float>& tglb,
        std::vector<float>& tdif,
        std::vector<float>& atm_albedo
    ) const;

    //! Calculate combined Rayleigh-aerosol mixture transmittance with coupling
    void calculateMixtureTransmittance(
        const std::vector<float>& wavelengths_um,
        float mu0,
        float pressure_Pa,
        float turbidity_beta,
        float angstrom_alpha,
        float aerosol_ssa,
        float aerosol_g,
        bool coupling,
        std::vector<float>& tglb,
        std::vector<float>& tdir,
        std::vector<float>& tdif,
        std::vector<float>& atm_albedo
    ) const;

    //! Calculate water vapor absorption transmittance
    void calculateWaterVaporTransmittance(
        const SpectralData& data,
        float mu0,
        float water_vapor_cm,
        std::vector<float>& transmittance
    ) const;

    //! Calculate ozone absorption transmittance
    void calculateOzoneTransmittance(
        const SpectralData& data,
        float mu0,
        float ozone_DU,
        std::vector<float>& transmittance
    ) const;

    //! Calculate oxygen absorption transmittance
    void calculateOxygenTransmittance(
        const SpectralData& data,
        float mu0,
        std::vector<float>& transmittance
    ) const;

    //! Helper method to calculate all three spectral components
    void calculateSpectralIrradianceComponents(
        std::vector<helios::vec2>& global_spectrum,
        std::vector<helios::vec2>& direct_spectrum,
        std::vector<helios::vec2>& diffuse_spectrum,
        float resolution_nm
    ) const;
};

#endif
