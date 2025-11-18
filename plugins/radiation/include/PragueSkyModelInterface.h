/**
 * \file PragueSkyModelInterface.h
 * \brief Helios wrapper interface for the Prague Sky Model
 *
 * This class provides a Helios-friendly interface to the Prague Sky Model,
 * handling coordinate system conversions, parameter mappings, and spectral
 * integration for camera bands.
 *
 * Copyright 2025 Helios
 * Integrates Prague Sky Model (Apache 2.0) by Charles University
 */

#ifndef HELIOS_PRAGUE_SKY_MODEL_INTERFACE_H
#define HELIOS_PRAGUE_SKY_MODEL_INTERFACE_H

#include "Context.h"
#include <string>
#include <vector>
#include <memory>

// Forward declare Prague model to avoid including full header
class PragueSkyModel;

namespace helios {

/**
 * \brief Wrapper class providing Helios-friendly interface to Prague Sky Model
 *
 * This class:
 * - Manages Prague model instance lifecycle
 * - Loads reduced dataset on initialization
 * - Handles coordinate system conversions (Helios ↔ Prague)
 * - Converts turbidity (AOD) to visibility (km)
 * - Integrates sky radiance over camera spectral response
 * - Converts Prague exceptions to helios_runtime_error
 */
class PragueSkyModelInterface {
public:
    /**
     * \brief Constructor - creates uninitialized interface
     */
    PragueSkyModelInterface();

    /**
     * \brief Destructor - cleans up Prague model instance
     */
    ~PragueSkyModelInterface();

    /**
     * \brief Initialize the Prague Sky Model with dataset file
     *
     * \param[in] dataset_path Absolute path to Prague dataset (.dat file)
     *
     * \note Fails with helios_runtime_error if:
     *       - Dataset file not found
     *       - Dataset read error
     *       - Already initialized
     */
    void initialize(const std::string& dataset_path);

    /**
     * \brief Check if model has been initialized
     * \return true if initialize() was called successfully
     */
    bool isInitialized() const;

    /**
     * \brief Get sky radiance at a single wavelength and direction
     *
     * \param[in] view_direction Viewing direction (normalized vector)
     * \param[in] sun_direction Sun direction (normalized vector from origin to sun)
     * \param[in] wavelength_nm Wavelength in nanometers [360-1480 nm]
     * \param[in] visibility_km Atmospheric visibility in kilometers [20-131.8 km]
     * \param[in] albedo Ground albedo [0-1] (optional, uses dataset default if <0)
     *
     * \return Sky radiance in W/m²/sr/nm
     *
     * \note Throws helios_runtime_error if:
     *       - Model not initialized
     *       - Wavelength outside dataset range
     *       - Visibility outside dataset range
     */
    float getSkyRadiance(const vec3& view_direction,
                         const vec3& sun_direction,
                         float wavelength_nm,
                         float visibility_km,
                         float albedo = -1.0f) const;

    /**
     * \brief Compute sky radiance integrated over camera spectral response
     *
     * Integrates: ∫ L(λ) × R(λ) dλ / ∫ R(λ) dλ
     * where L(λ) is sky radiance and R(λ) is camera spectral response.
     *
     * \param[in] view_direction Viewing direction (normalized vector)
     * \param[in] sun_direction Sun direction (normalized vector)
     * \param[in] visibility_km Atmospheric visibility in kilometers
     * \param[in] camera_response Camera spectral response [(lambda_nm, response), ...]
     *            Must be sorted by wavelength. Response values are unitless (0-1).
     * \param[in] albedo Ground albedo [0-1] (optional, uses dataset default if <0)
     *
     * \return Integrated sky radiance in W/m²/sr
     *
     * \note Uses trapezoidal integration with automatic wavelength sampling
     */
    float computeIntegratedSkyRadiance(const vec3& view_direction,
                                        const vec3& sun_direction,
                                        float visibility_km,
                                        const std::vector<vec2>& camera_response,
                                        float albedo = -1.0f) const;

    /**
     * \brief Convert Helios turbidity (AOD) to Prague visibility (km)
     *
     * Uses Koschmieder formula: V ≈ 3.9 / turbidity
     * Clamps result to Prague dataset range [20, 131.8 km]
     *
     * \param[in] turbidity Ångström aerosol optical depth at 500 nm
     * \return Visibility in kilometers
     */
    static float turbidityToVisibility(float turbidity);

    /**
     * \brief Get available data ranges from loaded dataset
     *
     * \param[out] min_wavelength_nm Minimum wavelength (nm)
     * \param[out] max_wavelength_nm Maximum wavelength (nm)
     * \param[out] min_visibility_km Minimum visibility (km)
     * \param[out] max_visibility_km Maximum visibility (km)
     * \param[out] min_elevation_deg Minimum solar elevation (degrees)
     * \param[out] max_elevation_deg Maximum solar elevation (degrees)
     *
     * \note Throws helios_runtime_error if model not initialized
     */
    void getAvailableRanges(float& min_wavelength_nm,
                            float& max_wavelength_nm,
                            float& min_visibility_km,
                            float& max_visibility_km,
                            float& min_elevation_deg,
                            float& max_elevation_deg) const;

private:
    /**
     * \brief Convert Helios sun direction vector to Prague elevation and azimuth angles
     *
     * Helios: sun_direction is unit vector from origin toward sun
     * Prague: elevation = angle above horizon [degrees], azimuth = from North [degrees]
     *
     * \param[in] sun_dir Sun direction (Helios convention)
     * \param[out] elevation_deg Solar elevation in degrees (0 = horizon, 90 = zenith)
     * \param[out] azimuth_deg Solar azimuth in degrees (0 = North, increasing clockwise)
     */
    void sunDirectionToAngles(const vec3& sun_dir,
                              double& elevation_deg,
                              double& azimuth_deg) const;

    std::unique_ptr<PragueSkyModel> model;  ///< Pointer to Prague model instance (opaque)
    bool initialized;                        ///< Initialization flag
    std::string dataset_path;                ///< Path to loaded dataset file
};

} // namespace helios

#endif // HELIOS_PRAGUE_SKY_MODEL_INTERFACE_H
