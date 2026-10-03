/** \file "SolarPosition.h" Primary header file for solar position model plug-in.

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
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (β), the aerosol optical depth at 1 um. See \ref TurbidityDefinition.
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \return Global solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     * \deprecated Use setAtmosphericConditions() and the parameter-free getSolarFlux() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getSolarFlux() instead")]] [[nodiscard]] float getSolarFlux(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Get the photosynthetically active (PAR) component of solar radiation flux perpendicular to the sun direction.
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (β), the aerosol optical depth at 1 um. See \ref TurbidityDefinition.
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \return Global solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     * \deprecated Use setAtmosphericConditions() and the parameter-free getSolarFluxPAR() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getSolarFluxPAR() instead")]] [[nodiscard]] float getSolarFluxPAR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Get the near-infrared (NIR) component of solar radiation flux perpendicular to the sun direction.
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (β), the aerosol optical depth at 1 um. See \ref TurbidityDefinition.
     * \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith() function.
     * \note The solar flux model is based on <a href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>.
     * \return Global solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
     * \deprecated Use setAtmosphericConditions() and the parameter-free getSolarFluxNIR() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getSolarFluxNIR() instead")]] [[nodiscard]] float getSolarFluxNIR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Get the fraction of solar radiation flux that is diffuse
    /**
     * \param[in] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (β), the aerosol optical depth at 1 um. See \ref TurbidityDefinition.
     * \return Fraction of global radiation that is diffuse
     * \deprecated Use setAtmosphericConditions() and the parameter-free getDiffuseFraction() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getDiffuseFraction() instead")]] [[nodiscard]] float getDiffuseFraction(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const;

    //! Calculate the ambient (sky) longwave radiation flux
    /**
     * \param[in] temperature_K Air temperature near the ground surface in Kelvin
     * \param[in] humidity_rel Air relative humidity near the ground surface
     * \return Ambient longwave flux in W/m^2
     * \note The longwave flux model is based on <a href="http://onlinelibrary.wiley.com/doi/10.1002/qj.49712253306/full">Prata (1996)</a>.
     * \deprecated Use setAtmosphericConditions() and the parameter-free getAmbientLongwaveFlux() instead
     */
    [[deprecated("Use setAtmosphericConditions() and parameter-free getAmbientLongwaveFlux() instead")]] [[nodiscard]] float getAmbientLongwaveFlux(float temperature_K, float humidity_rel) const;

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
     * \param[in] humidity_rel Air relative humidity near the ground surface (must be between 0 and 1; the spectral solar irradiance, sensor atmosphere and thermal atmosphere methods, which derive the precipitable water from it, require a value greater than 0)
     * \param[in] turbidity Ångström's aerosol turbidity coefficient (β), the aerosol optical depth at 1 um (must be >= 0). The aerosol optical depth at 550 nm is about 2.18 β for the Ångström exponent of 1.3 assumed by the
     * models. See \ref TurbidityDefinition. <b>Note:</b> This is NOT Linke turbidity, which uses a different scale.
     * \note This method sets global data in the Context with labels: atmosphere_pressure_Pa, atmosphere_temperature_K, atmosphere_humidity_rel, atmosphere_turbidity
     * \note Input values are validated. Invalid values will throw helios_runtime_error.
     */
    void setAtmosphericConditions(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity);

    //! Get atmospheric conditions from the Context
    /**
     * \param[out] pressure_Pa Atmospheric pressure near ground surface in Pascals
     * \param[out] temperature_K Air temperature near the ground surface in Kelvin
     * \param[out] humidity_rel Air relative humidity near the ground surface
     * \param[out] turbidity Ångström's aerosol turbidity coefficient (β), the aerosol optical depth at 1 um
     * \note If atmospheric conditions have not been set via setAtmosphericConditions(), reasonable defaults are used: pressure=101325 Pa (1 atm), temperature=300 K (27°C), humidity=0.5 (50%), turbidity=0.02
     */
    void getAtmosphericConditions(float &pressure_Pa, float &temperature_K, float &humidity_rel, float &turbidity) const;

    //! Set the total column ozone, overriding the climatological value
    /**
     * \param[in] ozone_DU Total column ozone in Dobson units (must be > 0).
     * \note This method sets global data in the Context with label atmosphere_ozone_DU. See \ref getOzoneColumn().
     */
    void setOzoneColumn(float ozone_DU);

    //! Get the total column ozone used by the solar radiation models
    /**
     * The ozone column set with \ref setOzoneColumn(), if any. Otherwise the monthly zonal-mean climatology of the NASA SBUV Merged Ozone Data Set averaged over 2005-2024, interpolated linearly in latitude between the
     * centers of its 5-degree latitude bands and in the day of the year between the middles of the months (see \ref OzoneColumn).
     * \return Total column ozone in Dobson units.
     * \note Next to a latitude band or month without data, the value of the band or month containing the location or date is used rather than interpolating toward the missing one. The climatology has no data in
     * high-latitude winter months or poleward of 80 degrees; for those months and latitudes an error is raised, and the ozone column must be set with \ref setOzoneColumn().
     */
    [[nodiscard]] float getOzoneColumn() const;

    //! Set the albedo of the ground surrounding the scene
    /**
     * The ground albedo sets how much light is reflected back and forth between the ground and the atmosphere, which increases the global and diffuse irradiance, most strongly at short wavelengths.
     * \param[in] albedo Broadband ground albedo (must be between 0 and 1).
     * \note This method sets global data in the Context with label atmosphere_ground_albedo.
     * \note The ground albedo affects the global and diffuse spectra of the spectral solar model (\ref calculateGlobalSolarSpectrum(), \ref calculateDiffuseSolarSpectrum()) and the ground irradiance and adjacency radiance of
     * \ref calculateSensorAtmosphereSpectra(); without cloud calibration it does not affect the direct beam. It does not affect the broadband \ref getSolarFlux() family of methods, and the Prague sky model takes its ground albedo as an argument of
     * \ref updatePragueSkyModel().
     */
    void setGroundAlbedo(float albedo);

    //! Get the albedo of the ground surrounding the scene
    /**
     * \return Ground albedo set by \ref setGroundAlbedo(), or 0.2 if it has not been set.
     */
    [[nodiscard]] float getGroundAlbedo() const;

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
     * \note The flux is integrated over all longwave wavelengths. For the sky flux within a thermal band with wavelength bounds, use \ref getThermalSkyFlux().
     * \return Ambient longwave flux in W/m^2
     */
    [[nodiscard]] float getAmbientLongwaveFlux() const;

    //! Calculate direct beam solar spectrum and store in global data
    /**
     * \param[in] label User-defined label for storing spectral data in Context global data
     * \param[in] resolution_nm Wavelength resolution in nm (default: 1 nm). Must be >= 1 nm and <= 2300 nm.
     * \note The irradiance is that of the direct solar beam on a surface normal to the sun direction.
     * \note The spectrum is computed at 1 nm from 300 to 2600 nm with the spectral model described in \ref SpectralIrradianceTheory, which follows the SSolar-GOA model of Cachorro et al. (2022) with the corrections
     * and data listed there. Atmospheric conditions are taken from \ref setAtmosphericConditions(), the ozone column from \ref getOzoneColumn() and the ground albedo from \ref setGroundAlbedo(); cloud calibration is
     * applied if enabled with \ref enableCloudCalibration().
     * \note Stores result in Context global data as std::vector<helios::vec2> (wavelength_nm, W/m²/nm) with the specified label.
     * \note An error is raised if the sun is at or below the horizon; if the relative humidity set with \ref setAtmosphericConditions() is not greater than zero; if no ozone column was set with
     * \ref setOzoneColumn() and the climatology has no data for the location and date (see \ref getOzoneColumn()); or if a gas path along the solar beam exceeds the range of the gas transmittance tables: a water
     * vapor path (precipitable water times air mass) above 320 g/cm² (possible only for precipitable water above about 8.4 cm with the sun close to the horizon), an ozone path (ozone column times air mass) above
     * 25 atm-cm (an ozone column above about 660 DU with the sun at the horizon), or an air mass times the ratio of surface pressure to 1013.25 hPa above 37.99 (a surface pressure above about 1015 hPa with the
     * sun within a fraction of a degree of the horizon).
     */
    void calculateDirectSolarSpectrum(const std::string &label, float resolution_nm = 1.0f);

    //! Calculate diffuse solar spectrum and store in global data
    /**
     * \param[in] label User-defined label for storing spectral data in Context global data
     * \param[in] resolution_nm Wavelength resolution in nm (default: 1 nm). Must be >= 1 nm and <= 2300 nm.
     * \note The irradiance is the diffuse (sky) irradiance on a horizontal surface, including light reflected by the ground and scattered back down by the atmosphere.
     * \note The spectrum is computed at 1 nm from 300 to 2600 nm with the spectral model described in \ref SpectralIrradianceTheory, which follows the SSolar-GOA model of Cachorro et al. (2022) with the corrections
     * and data listed there. Atmospheric conditions are taken from \ref setAtmosphericConditions(), the ozone column from \ref getOzoneColumn() and the ground albedo from \ref setGroundAlbedo(); cloud calibration is
     * applied if enabled with \ref enableCloudCalibration().
     * \note Stores result in Context global data as std::vector<helios::vec2> (wavelength_nm, W/m²/nm) with the specified label.
     * \note An error is raised if the sun is at or below the horizon; if the relative humidity set with \ref setAtmosphericConditions() is not greater than zero; if no ozone column was set with
     * \ref setOzoneColumn() and the climatology has no data for the location and date (see \ref getOzoneColumn()); or if a gas path along the solar beam exceeds the range of the gas transmittance tables: a water
     * vapor path (precipitable water times air mass) above 320 g/cm² (possible only for precipitable water above about 8.4 cm with the sun close to the horizon), an ozone path (ozone column times air mass) above
     * 25 atm-cm (an ozone column above about 660 DU with the sun at the horizon), or an air mass times the ratio of surface pressure to 1013.25 hPa above 37.99 (a surface pressure above about 1015 hPa with the
     * sun within a fraction of a degree of the horizon).
     */
    void calculateDiffuseSolarSpectrum(const std::string &label, float resolution_nm = 1.0f);

    //! Calculate global (total) solar spectrum and store in global data
    /**
     * \param[in] label User-defined label for storing spectral data in Context global data
     * \param[in] resolution_nm Wavelength resolution in nm (default: 1 nm). Must be >= 1 nm and <= 2300 nm.
     * \note The irradiance is the global (direct plus diffuse) irradiance on a horizontal surface.
     * \note The spectrum is computed at 1 nm from 300 to 2600 nm with the spectral model described in \ref SpectralIrradianceTheory, which follows the SSolar-GOA model of Cachorro et al. (2022) with the corrections
     * and data listed there. Atmospheric conditions are taken from \ref setAtmosphericConditions(), the ozone column from \ref getOzoneColumn() and the ground albedo from \ref setGroundAlbedo(); cloud calibration is
     * applied if enabled with \ref enableCloudCalibration().
     * \note Stores result in Context global data as std::vector<helios::vec2> (wavelength_nm, W/m²/nm) with the specified label.
     * \note An error is raised if the sun is at or below the horizon; if the relative humidity set with \ref setAtmosphericConditions() is not greater than zero; if no ozone column was set with
     * \ref setOzoneColumn() and the climatology has no data for the location and date (see \ref getOzoneColumn()); or if a gas path along the solar beam exceeds the range of the gas transmittance tables: a water
     * vapor path (precipitable water times air mass) above 320 g/cm² (possible only for precipitable water above about 8.4 cm with the sun close to the horizon), an ozone path (ozone column times air mass) above
     * 25 atm-cm (an ozone column above about 660 DU with the sun at the horizon), or an air mass times the ratio of surface pressure to 1013.25 hPa above 37.99 (a surface pressure above about 1015 hPa with the
     * sun within a fraction of a degree of the horizon).
     */
    void calculateGlobalSolarSpectrum(const std::string &label, float resolution_nm = 1.0f);

    //! Calculate the spectral atmospheric quantities needed to simulate imagery from a sensor above the atmosphere (e.g., a satellite)
    /**
     * Radiance reaching the sensor is modeled as \f$L_{sensor} = L_{path} + T_{dir}^{\uparrow} L_{surface} + L_{adj}\f$, where \f$L_{surface}\f$ is the radiance leaving the scene. The atmosphere is described by a look-up
     * table computed with the 6S radiative transfer code (continental aerosol), which gives both the upward path to the sensor and the downward solar illumination of the scene. For the two to be consistent, the scene
     * should be illuminated with the "_direct_irradiance" and "_diffuse_irradiance" spectra stored by this method rather than those of \ref calculateDirectSolarSpectrum() and \ref calculateDiffuseSolarSpectrum().
     *
     * The following spectra are stored in Context global data as std::vector<helios::vec2> (wavelength_nm, value), each labeled with the given label followed by a suffix:
     *   - "_path_radiance": radiance scattered by the atmosphere into the sensor without reaching the ground, in W/m²/sr/nm
     *   - "_adjacency_radiance": radiance reflected by the surrounding ground (albedo from \ref setGroundAlbedo()) and scattered by the atmosphere into the sensor, in W/m²/sr/nm
     *   - "_upward_direct_transmittance": transmittance of the surface-leaving radiance straight to the sensor, including gas absorption
     *   - "_upward_diffuse_transmittance": transmittance of surface-leaving radiance scattered into the sensor direction, including gas absorption
     *   - "_spherical_albedo": spherical albedo of the atmosphere
     *   - "_direct_irradiance": direct solar irradiance at the ground, normal to the sun direction, in W/m²/nm
     *   - "_diffuse_irradiance": diffuse solar irradiance at the ground on a horizontal surface, in W/m²/nm
     *   - "_global_irradiance": global solar irradiance at the ground on a horizontal surface, in W/m²/nm
     * The unit vector toward the sensor is also stored, as helios::vec3 global data with the suffix "_direction_to_sensor".
     * \param[in] label Prefix of the labels under which the spectra are stored in Context global data.
     * \param[in] direction_to_sensor Vector pointing from the scene toward the sensor (need not be normalized). Its zenith angle must not exceed 60 degrees.
     * \param[in] resolution_nm Wavelength resolution in nm (default: 1 nm). Must be >= 1 nm and <= 2300 nm.
     * \note Atmospheric conditions are taken from \ref setAtmosphericConditions() and the ground albedo from \ref setGroundAlbedo(). The turbidity is converted to aerosol optical depth at 550 nm with an Ångström exponent of 1.3.
     * \note The look-up table covers solar zenith angles up to 70 degrees, view zenith angles up to 60 degrees, aerosol optical depths at 550 nm up to 1.2 (turbidity up to about 0.55) and surface pressures from 701.2 to
     * 1013 hPa (ground elevations up to about 3 km). Pressures up to 1050 hPa are extrapolated linearly. Its gas tables cover water vapor paths, the column water vapor times \f$1/\mu_s + 1/\mu_v\f$, up to 320 g/cm².
     * Conditions outside these ranges, and a sun below the horizon, raise an error.
     * \note The sensor is taken to be above the atmosphere. The model assumes clear sky and cannot be used with cloud calibration enabled.
     */
    void calculateSensorAtmosphereSpectra(const std::string &label, const helios::vec3 &direction_to_sensor, float resolution_nm = 1.0f);

    //! Calculate the thermal infrared atmospheric quantities needed to simulate thermal imagery from a sensor above the atmosphere (e.g., a satellite)
    /**
     * Radiance reaching the sensor at a thermal wavelength is modeled as \f$L_{sensor} = \tau L_{surface} + L^{\uparrow}\f$, where \f$L_{surface}\f$ is the radiance leaving the scene (emitted plus reflected sky radiance), \f$\tau\f$ the
     * atmospheric transmittance and \f$L^{\uparrow}\f$ the radiance emitted upward by the atmosphere. These are interpolated from a look-up table computed with the libRadtran radiative transfer code (REPTRAN absorption
     * parameterization, clear sky without aerosol; see \ref SensorThermalAtmosphereTheory) as averages over 5 cm⁻¹ wavenumber bins between 650 and 1820 cm⁻¹. The following spectra are stored in Context global data as
     * std::vector<helios::vec2> (wavelength of the bin center in nm, value), with 234 points from 5502 to 15326 nm, each labeled with the given label followed by a suffix:
     *   - "_thermal_transmittance": atmospheric transmittance from the ground to the sensor
     *   - "_thermal_upwelling_radiance": radiance emitted by the atmosphere toward the sensor, in W/m²/sr/nm
     *   - "_thermal_downwelling_radiance": sky irradiance on a horizontal surface at the ground divided by \f$\pi\f$, in W/m²/sr/nm, independent of the view direction. A Lambertian surface of emissivity \f$\varepsilon\f$
     *     reflects \f$(1-\varepsilon)L^{\downarrow}\f$.
     *
     * The column water vapor (float, in g/cm²) is stored with the suffix "_thermal_water_vapor_cm", and the unit vector toward the sensor, as helios::vec3, with the suffix "_direction_to_sensor".
     * \param[in] label Prefix of the labels under which the results are stored in Context global data.
     * \param[in] direction_to_sensor Vector pointing from the scene toward the sensor (need not be normalized). Its zenith angle must not exceed 60 degrees.
     * \note The atmospheric profile is built from the surface air temperature, humidity and pressure set with \ref setAtmosphericConditions() and the ozone column of \ref getOzoneColumn(). The column water vapor, derived from
     * the temperature and humidity, must lie within 0.1-7.5 g/cm², the pressure within 701.2-1050 hPa (pressures above 1013 hPa are extrapolated) and the ozone column within 150-450 DU. The supported air temperatures
     * depend on the pressure: 246.2-321.7 K at 1013 hPa. Conditions outside these ranges raise an error.
     * \note The model assumes clear sky and does not depend on the position of the sun, so night-time imagery can be simulated. Reflected sunlight, which is significant below about 5 um, is not included.
     */
    void calculateSensorThermalAtmosphere(const std::string &label, const helios::vec3 &direction_to_sensor);

    //! Calculate the sky longwave irradiance on a horizontal surface within a thermal band
    /**
     * This is the band-specific counterpart of \ref getAmbientLongwaveFlux(): the downward irradiance emitted by a clear-sky atmosphere between two wavelengths, integrated from the downwelling radiance of the thermal
     * atmosphere look-up table (see \ref calculateSensorThermalAtmosphere()). It is the diffuse flux to give an emission band with the same wavelength bounds (RadiationModel::setDiffuseRadiationFlux()), so that the sky
     * radiance the scene reflects is consistent with the thermal sensor atmosphere.
     * \param[in] wavelength_min_nm Lower wavelength bound of the band in nm.
     * \param[in] wavelength_max_nm Upper wavelength bound of the band in nm.
     * \return Sky longwave irradiance within the band on a horizontal surface, in W/m².
     * \note Both bounds must lie within 5502-15326 nm, the range of the tabulated spectrum. The atmospheric conditions must lie within the ranges given for \ref calculateSensorThermalAtmosphere().
     */
    [[nodiscard]] float getThermalSkyFlux(float wavelength_min_nm, float wavelength_max_nm) const;

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
    [[nodiscard]] bool pragueSkyModelNeedsUpdate(float ground_albedo = 0.33f, float sun_tolerance = 0.01f, float turbidity_tolerance = 0.02f, float albedo_tolerance = 0.05f) const;

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
    void fitAngularParametersAtWavelength(float wavelength, float visibility_km, float albedo, const helios::vec3 &sun_dir, float &L_zenith, float &circ_str, float &circ_width, float &horiz_bright, float &normalization);

    [[nodiscard]] float fitCircumsolarWidth(float L1, float L2, float h1, float h2, float gamma1, float gamma2) const;

    [[nodiscard]] float computeAngularNormalization(float circ_str, float circ_width, float horiz_bright) const;

    [[nodiscard]] helios::vec3 rotateDirectionTowardZenith(const helios::vec3 &dir, float angle_rad) const;

    [[nodiscard]] helios::vec3 getDirectionAwayFromSun(const helios::vec3 &sun_dir, float zenith_angle_deg) const;

    void GueymardSolarModel(float pressure, float temperature, float humidity, float turbidity, float &Eb_PAR, float &Eb_NIR, float &fdiff) const;

    //! Calibrate the clear-sky direct-normal fluxes and diffuse fraction against a measured all-wave flux
    /**
     * The cloud attenuation factor is computed from the ALL-WAVE (PAR+NIR) flux and then applied to
     * each band, so that the spectral partitioning between PAR and NIR is preserved and the two
     * bands continue to sum to the broadband total.
     * \param[in,out] Eb_PAR_Wm2 Clear-sky PAR flux perpendicular to the sun (W/m^2); scaled in place.
     * \param[in,out] Eb_NIR_Wm2 Clear-sky NIR flux perpendicular to the sun (W/m^2); scaled in place.
     * \param[in,out] fdiff_calc Clear-sky diffuse fraction; replaced by the cloud-calibrated value.
     */
    void applyCloudCalibration(float &Eb_PAR_Wm2, float &Eb_NIR_Wm2, float &fdiff_calc) const;

    static float turbidityResidualFunction(float turbidity, std::vector<float> &parameters, const void *a_solarpositionmodel);

    // ===== SSolar-GOA Spectral Solar Radiation Model =====

    //! Extraterrestrial solar spectrum used by the spectral solar irradiance model: the Wehrli (1985) spectrum averaged over 1 nm bins when loaded (see loadFromFile())
    struct SpectralData {
        std::vector<float> wavelengths_nm; // 300-2600 nm (2301 points)
        std::vector<float> toa_irradiance; // Extraterrestrial irradiance at the mean Earth-Sun distance in W/m²/nm

        //! Load the Wehrli (1985) spectrum from its National Laboratory of the Rockies distribution (after a header, each line holds a wavelength in nm, the spectral irradiance in W/m²/nm and the cumulative irradiance) and average it
        //! over 1 nm bins centered on 300-2600 nm, each source value taken to apply halfway to the neighbouring wavelengths
        void loadFromFile(const std::string &path);
    };

    //! Ångström wavelength exponent of the aerosol optical depth in the spectral solar irradiance model; also converts the turbidity to the aerosol optical depth at 550 nm in calculateSensorAtmosphereSpectra()
    static constexpr float ssolar_angstrom_alpha = 1.3f;
    //! Aerosol single-scattering albedo in the spectral solar irradiance model: that of the 6S continental aerosol at 550 nm (6S User Guide, Version 3, Part 2, p. 120, Table 3)
    static constexpr float ssolar_aerosol_ssa = 0.893f;
    //! Aerosol scattering asymmetry parameter in the spectral solar irradiance model: that of the 6S continental aerosol at 550 nm (6S User Guide, Version 3, Part 2, p. 120, Table 3)
    static constexpr float ssolar_aerosol_g = 0.634f;

    //! Get the extraterrestrial solar spectrum, loading it from disk on the first call
    [[nodiscard]] static const SpectralData &getSSolarSpectralData();

    //! One table of the atmospheric look-up table used by calculateSensorAtmosphereSpectra(), interpolated multilinearly in its axes
    struct AtmosphereTable {
        std::vector<std::string> axis_names;
        //! Values of each axis, each strictly increasing or strictly decreasing
        std::vector<std::vector<float>> axis_values;
        //! Value up to which each axis may be linearly extrapolated beyond its largest tabulated value (equal to the largest value if no extrapolation is allowed)
        std::vector<float> axis_extrapolation_limits;
        //! Table values in row-major order (the last axis varies fastest)
        std::vector<float> data;

        //! Interpolate the table at a point
        /**
         * \param[in] point Coordinates along each axis, in axis order.
         * \param[in] table_name Name of the table, used in error messages.
         * \return Interpolated value.
         */
        [[nodiscard]] float interpolate(const std::vector<float> &point, const std::string &table_name) const;

        //! Interpolate the table at a point in all axes but the last, returning the values at every node of the last axis
        /**
         * \param[in] point Coordinates along each axis except the last, in axis order.
         * \param[in] table_name Name of the table, used in error messages.
         * \return Interpolated values, one per node of the last axis.
         */
        [[nodiscard]] std::vector<float> interpolateLastAxis(const std::vector<float> &point, const std::string &table_name) const;

    private:
        //! For each of the first point.size() axes, find the lower index of the interval bracketing the point's coordinate and the weight of the upper node, raising an error outside the axis range
        void bracket(const std::vector<float> &point, const std::string &table_name, std::vector<size_t> &lower_index, std::vector<float> &upper_weight) const;
    };

    //! Atmospheric look-up table computed with the 6S radiative transfer code (see plugins/solarposition/utilities/generate_atmosphere_lut.py)
    struct AtmosphereLUT {
        std::string provenance;
        std::map<std::string, AtmosphereTable> tables;

        //! Load the look-up table from its binary file
        void loadFromFile(const std::string &path);

        //! Get a table by name
        [[nodiscard]] const AtmosphereTable &getTable(const std::string &name) const;
    };

    //! Get the atmospheric look-up table, loading it from disk on the first call
    [[nodiscard]] static const AtmosphereLUT &getAtmosphereLUT();

    //! Interpolate the scattering quantity of an atmospheric look-up table at every wavelength of the SSolar-GOA grid
    /**
     * The table is interpolated in its non-wavelength axes at each of its wavelength nodes, then log-log interpolated between nodes, since scattering quantities vary close to a power law with wavelength.
     * \param[in] table Table whose first axis is wavelength in nm.
     * \param[in] table_name Name of the table, used in error messages.
     * \param[in] point Coordinates along the remaining axes.
     * \param[in] wavelengths_nm Wavelengths at which to evaluate the table.
     * \return Values at each wavelength.
     */
    [[nodiscard]] static std::vector<float> interpolateScatteringSpectrum(const AtmosphereTable &table, const std::string &table_name, const std::vector<float> &point, const std::vector<float> &wavelengths_nm);

    //! Interpolate a gas transmittance table of the atmospheric look-up table at every wavelength of the SSolar-GOA grid
    /**
     * The absorber amount is interpolated in log space. Amounts below the smallest tabulated amount are interpolated linearly toward a transmittance of one at zero amount.
     * \param[in] table Table with axes (wavelength in nm, absorber amount along the path).
     * \param[in] table_name Name of the table, used in error messages.
     * \param[in] absorber_amount Absorber amount along the path, in the units of the table's second axis.
     * \param[in] wavelengths_nm Wavelengths at which to evaluate the table.
     * \return Transmittance at each wavelength.
     */
    [[nodiscard]] static std::vector<float> interpolateGasTransmittance(const AtmosphereTable &table, const std::string &table_name, float absorber_amount, const std::vector<float> &wavelengths_nm);

    //! Calculate column precipitable water from near-surface air temperature and relative humidity through the dew point, with the empirical relation used by this plugin's SSolar-GOA implementation (attributed in that
    //! implementation to Viswanadham 1981)
    /**
     * \param[in] temperature_K Near-surface air temperature in Kelvin.
     * \param[in] humidity_rel Near-surface relative humidity (greater than 0 and at most 1; an error is raised otherwise).
     * \return Precipitable water in cm.
     */
    [[nodiscard]] static float calculatePrecipitableWater(float temperature_K, float humidity_rel);

    //! Get the monthly zonal-mean total ozone climatology, loading it from disk on the first call
    /**
     * \return Total column ozone in Dobson units for each of the 36 latitude bands 5 degrees wide (southernmost first) and each month; NaN where the climatology has no data.
     */
    [[nodiscard]] static const std::vector<std::array<float, 12>> &getOzoneClimatology();

    //! Calculate the square of the ratio of the mean to the actual Earth-Sun distance, which scales extraterrestrial irradiance, with the Fourier series of Spencer (1971)
    /**
     * \param[in] julian_day Day of the year (1-366).
     * \return Earth-Sun distance factor (dimensionless, 0.967-1.035).
     */
    [[nodiscard]] float calculateGeometricFactor(int julian_day) const;

    //! Calculate the Rayleigh optical depth of the atmospheric column with Eq. 7 of Cachorro et al. (2022), which they attribute to Gueymard (1995)
    /**
     * \param[in] wavelength_um Wavelength in micrometers.
     * \param[in] pressure_ratio Surface pressure divided by standard sea-level pressure (101325 Pa).
     * \return Rayleigh optical depth (dimensionless).
     */
    [[nodiscard]] static float calculateRayleighOpticalDepth(float wavelength_um, float pressure_ratio);

    //! Calculate the aerosol optical depth of the atmospheric column from the Ångström law
    /**
     * \param[in] wavelength_um Wavelength in micrometers.
     * \param[in] alpha Ångström wavelength exponent.
     * \param[in] beta Ångström turbidity coefficient (aerosol optical depth at 1 um).
     * \return Aerosol optical depth (dimensionless).
     */
    [[nodiscard]] static float calculateAerosolOpticalDepth(float wavelength_um, float alpha, float beta);

    //! Resample a spectrum on a uniform wavelength grid (e.g. 300-2600 nm at 1 nm) to a coarser resolution: output wavelengths start at the first native wavelength and step by resolution_nm, each taking the nearest native point
    [[nodiscard]] static std::vector<helios::vec2> downsampleSpectrum(const std::vector<helios::vec2> &spectrum, float resolution_nm);

    //! Atmospheric state at which the thermal atmosphere look-up table is interpolated
    struct ThermalAtmosphereState {
        //! Column water vapor in g/cm²
        float water_vapor_cm;
        //! Position of the atmospheric profile in the table's climatology of standard atmospheres (see SensorThermalAtmosphereTheory)
        float climate_coordinate;
        float pressure_hPa;
        float ozone_DU;
    };

    //! Get the thermal atmosphere look-up table, loading it from disk on the first call
    [[nodiscard]] static const AtmosphereLUT &getThermalAtmosphereLUT();

    //! Atmospheric state for the thermal atmosphere look-up table from the conditions set with setAtmosphericConditions() and the ozone column, raising an error if it lies outside the table
    /**
     * \param[in] calling_function Name of the public method, used in error messages.
     */
    [[nodiscard]] ThermalAtmosphereState getThermalAtmosphereState(const std::string &calling_function) const;

    //! Map a surface air temperature to the climate coordinate of the thermal atmosphere look-up table
    /**
     * \param[in] climate_temperature The table "climate_temperature_K" (surface air temperature against pressure and climate coordinate).
     * \param[in] air_temperature_K Surface air temperature in K.
     * \param[in] pressure_hPa Surface pressure in hPa.
     * \param[in] calling_function Name of the public method, used in error messages.
     * \return Climate coordinate.
     */
    [[nodiscard]] static float calculateThermalClimateCoordinate(const AtmosphereTable &climate_temperature, float air_temperature_K, float pressure_hPa, const std::string &calling_function);

    //! Downwelling sky radiance spectrum of the thermal atmosphere look-up table, as (wavelength in nm, radiance in W/m²/sr/nm)
    [[nodiscard]] static std::vector<helios::vec2> calculateThermalDownwellingRadiance(const ThermalAtmosphereState &state);

    //! Calculate the relative optical air mass with the formula of Kasten & Young (1989)
    /**
     * \param[in] solar_zenith_rad Solar zenith angle in radians (0 to pi/2).
     * \return Optical air mass relative to the vertical path (1 overhead, 37.9 at the horizon).
     */
    [[nodiscard]] static float calculateRelativeAirMass(float solar_zenith_rad);

    //! Calculate the exponential integral E_n(x) (Abramowitz & Stegun 1964, Eqs. 5.1.11, 5.1.14, 5.1.22 and 5.1.23)
    /**
     * \param[in] order Order n (n >= 1).
     * \param[in] x Argument (x >= 0; x > 0 for n = 1).
     * \return E_n(x).
     */
    [[nodiscard]] static double calculateExponentialIntegral(int order, double x);

    //! Calculate the spherical albedo of a purely Rayleigh-scattering layer with the closed form of 6S subroutine CSALBR
    /**
     * \param[in] rayleigh_optical_depth Rayleigh optical depth of the layer.
     * \return Spherical albedo (the reflectance of the layer for isotropic illumination from below).
     */
    [[nodiscard]] static double calculateRayleighSphericalAlbedo(double rayleigh_optical_depth);

    //! Calculate the reflectance of a homogeneous layer over a black surface under diffuse illumination, in the two-stream approximation with the hemispheric closure (Heng & Kitzmann 2017; Heng et al. 2014)
    /**
     * \param[in] optical_depth Extinction optical depth of the layer.
     * \param[in] single_scattering_albedo Single-scattering albedo of the layer (0-1).
     * \param[in] asymmetry Scattering asymmetry parameter of the layer.
     * \return Reflectance.
     */
    [[nodiscard]] static double calculateTwoStreamDiffuseReflectance(double optical_depth, double single_scattering_albedo, double asymmetry);

    //! Calculate the total (direct plus diffuse) scattering transmittance of a homogeneous Rayleigh-aerosol layer over a black surface, Eqs. 14-15 of Cachorro et al. (2022) with the square root in k
    /**
     * \param[in] optical_depth Extinction optical depth of the layer.
     * \param[in] single_scattering_albedo Single-scattering albedo of the layer (0-1).
     * \param[in] asymmetry Scattering asymmetry parameter of the layer.
     * \param[in] air_mass Relative optical air mass of the solar path.
     * \return Transmittance.
     */
    [[nodiscard]] static double calculateMixedLayerTransmittance(double optical_depth, double single_scattering_albedo, double asymmetry, double air_mass);

    //! Calculate the global horizontal, direct normal and diffuse horizontal spectral irradiance (see \ref SpectralIrradianceTheory), applying cloud calibration if it is enabled
    /**
     * \param[out] global_spectrum Global horizontal irradiance (wavelength in nm, W/m²/nm).
     * \param[out] direct_spectrum Direct irradiance normal to the sun direction (wavelength in nm, W/m²/nm).
     * \param[out] diffuse_spectrum Diffuse horizontal irradiance (wavelength in nm, W/m²/nm).
     * \param[in] resolution_nm Output wavelength resolution in nm (1-2300).
     */
    void calculateSpectralIrradianceComponents(std::vector<helios::vec2> &global_spectrum, std::vector<helios::vec2> &direct_spectrum, std::vector<helios::vec2> &diffuse_spectrum, float resolution_nm) const;

    //! Apply cloud calibration to clear-sky spectral irradiance using a measured broadband flux
    /**
     * Normalizes the modeled clear-sky spectrum so that its integral matches the measured broadband
     * global horizontal irradiance, then applies the empirical spectral modifiers of Bird, Riordan &
     * Myers (1987) to account for the preferential transmission of short wavelengths through cloud.
     * See \ref SolarFluxClouds for the model description and its limitations.
     *
     * The spectra must be supplied at the model's native 1 nm resolution (before any downsampling),
     * since the normalization integrates over them.
     * \param[in,out] global_spectrum Global horizontal spectral irradiance (W/m^2/nm).
     * \param[in,out] direct_spectrum Direct normal spectral irradiance (W/m^2/nm).
     * \param[in,out] diffuse_spectrum Diffuse horizontal spectral irradiance (W/m^2/nm).
     * \param[in] mu0 Cosine of the solar zenith angle.
     */
    void applySpectralCloudCalibration(std::vector<helios::vec2> &global_spectrum, std::vector<helios::vec2> &direct_spectrum, std::vector<helios::vec2> &diffuse_spectrum, float mu0) const;

    //! Estimate the diffuse fraction of global horizontal irradiance from the clearness index
    /**
     * Uses the hourly correlation of Erbs et al. (1982), Solar Energy 28(4):293-302, fit to
     * pyrheliometer and pyranometer data from four U.S. locations. The diffuse fraction rises toward
     * unity under thick cloud and falls to 0.165 under clear skies.
     * \param[in] clearness_index Ratio of measured global horizontal irradiance to extraterrestrial horizontal irradiance.
     * \return Diffuse fraction of global horizontal irradiance, in the range [0,1].
     */
    [[nodiscard]] static float diffuseFractionFromClearnessIndex(float clearness_index);

    //! Calculate the extraterrestrial irradiance on a horizontal surface
    /**
     * \return Extraterrestrial horizontal irradiance in W/m^2, or zero when the sun is below the horizon.
     */
    [[nodiscard]] float getExtraterrestrialHorizontalFlux() const;

    //! Integrate a spectral irradiance curve over wavelength using the trapezoidal rule
    /**
     * \param[in] spectrum Spectral irradiance as (wavelength_nm, W/m^2/nm) pairs, ordered by wavelength.
     * \return Integrated irradiance in W/m^2.
     */
    [[nodiscard]] static float integrateSpectrum(const std::vector<helios::vec2> &spectrum);
};

#endif
