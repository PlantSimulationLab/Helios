/**
 * \file PragueSkyModelInterface.cpp
 * \brief Implementation of Helios wrapper for Prague Sky Model
 *
 * Copyright 2025 Helios
 * Integrates Prague Sky Model (Apache 2.0) by Charles University
 */

#include "PragueSkyModelInterface.h"
#include "PragueSkyModel.h"
#include "global.h"
#include <cmath>
#include <algorithm>
#include <sstream>

using namespace helios;

// Constructor
PragueSkyModelInterface::PragueSkyModelInterface()
    : model(nullptr), initialized(false) {
}

// Destructor
PragueSkyModelInterface::~PragueSkyModelInterface() {
    // unique_ptr automatically cleans up
}

// Initialize the Prague Sky Model
void PragueSkyModelInterface::initialize(const std::string& dataset_path) {
    if (initialized) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::initialize): Model already initialized.");
    }

    // Create Prague model instance
    model = std::make_unique<PragueSkyModel>();

    // Try to initialize with dataset file
    // Prague model throws exceptions, we need to catch and convert to helios_runtime_error
    try {
        model->initialize(dataset_path);
    } catch (const PragueSkyModel::DatasetNotFoundException& e) {
        std::ostringstream oss;
        oss << "ERROR (PragueSkyModelInterface::initialize): Prague dataset file not found: "
            << dataset_path << "\n"
            << "Please ensure PragueSkyModelReduced.dat is present in plugins/radiation/spectral_data/prague_sky_model/\n"
            << "The reduced dataset should be ~26 MB.";
        helios_runtime_error(oss.str());
    } catch (const PragueSkyModel::DatasetReadException& e) {
        std::ostringstream oss;
        oss << "ERROR (PragueSkyModelInterface::initialize): Failed to read Prague dataset: "
            << e.what();
        helios_runtime_error(oss.str());
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ERROR (PragueSkyModelInterface::initialize): Unexpected error initializing Prague model: "
            << e.what();
        helios_runtime_error(oss.str());
    }

    this->dataset_path = dataset_path;
    initialized = true;
}

// Check if initialized
bool PragueSkyModelInterface::isInitialized() const {
    return initialized;
}

// Convert Helios sun direction to Prague elevation and azimuth
void PragueSkyModelInterface::sunDirectionToAngles(const vec3& sun_dir,
                                                    double& elevation_deg,
                                                    double& azimuth_deg) const {
    // Helios: sun_dir is unit vector from origin toward sun
    // Z component gives elevation: sun_dir.z = sin(elevation)
    // X,Y components give azimuth

    // Convert to spherical coordinates
    SphericalCoord sun_sphere = cart2sphere(sun_dir);

    // sun_sphere.zenith is angle from vertical (radians)
    // Prague elevation is angle from horizontal (degrees)
    double zenith_rad = sun_sphere.zenith;
    elevation_deg = 90.0 - (zenith_rad * 180.0 / M_PI);

    // sun_sphere.azimuth is in radians
    // Prague expects azimuth from North, increasing clockwise
    // Helios azimuth convention needs verification, but typically:
    // 0 = +X axis, increasing counterclockwise when viewed from above
    // Prague: 0 = North (+Y), increasing clockwise
    // Conversion: prague_azimuth = 90 - helios_azimuth
    double helios_azimuth_deg = sun_sphere.azimuth * 180.0 / M_PI;
    azimuth_deg = 90.0 - helios_azimuth_deg;

    // Normalize azimuth to [0, 360)
    while (azimuth_deg < 0.0) azimuth_deg += 360.0;
    while (azimuth_deg >= 360.0) azimuth_deg -= 360.0;
}

// Get sky radiance at single wavelength
float PragueSkyModelInterface::getSkyRadiance(const vec3& view_direction,
                                               const vec3& sun_direction,
                                               float wavelength_nm,
                                               float visibility_km,
                                               float albedo) const {
    if (!initialized) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::getSkyRadiance): Model not initialized. Call initialize() first.");
    }

    // Convert sun direction to elevation and azimuth
    double solar_elevation_deg, solar_azimuth_deg;
    sunDirectionToAngles(sun_direction, solar_elevation_deg, solar_azimuth_deg);

    // Set default albedo if not provided
    if (albedo < 0.0f) {
        // Use default from reduced dataset (0.33 - typical vegetation)
        albedo = 0.33f;
    }

    // Create Prague Vector3 for view direction
    PragueSkyModel::Vector3 prague_view_dir(view_direction.x, view_direction.y, view_direction.z);

    // Observer at ground level (altitude = 0)
    PragueSkyModel::Vector3 view_point(0.0, 0.0, 0.0);

    // Compute Prague parameters
    PragueSkyModel::Parameters params;
    try {
        params = model->computeParameters(view_point,
                                          prague_view_dir,
                                          solar_elevation_deg * M_PI / 180.0, // Convert to radians
                                          solar_azimuth_deg * M_PI / 180.0,
                                          visibility_km,
                                          albedo);
    } catch (const PragueSkyModel::NotInitializedException& e) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::getSkyRadiance): Prague model not initialized.");
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ERROR (PragueSkyModelInterface::getSkyRadiance): Error computing Prague parameters: "
            << e.what();
        helios_runtime_error(oss.str());
    }

    // Query sky radiance at wavelength
    double radiance_Wm2_sr_nm = 0.0;
    try {
        radiance_Wm2_sr_nm = model->skyRadiance(params, wavelength_nm);
    } catch (const PragueSkyModel::NotInitializedException& e) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::getSkyRadiance): Prague model not initialized.");
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ERROR (PragueSkyModelInterface::getSkyRadiance): Error querying sky radiance: "
            << e.what();
        helios_runtime_error(oss.str());
    }

    return static_cast<float>(radiance_Wm2_sr_nm);
}

// Compute integrated sky radiance over camera spectral response
float PragueSkyModelInterface::computeIntegratedSkyRadiance(const vec3& view_direction,
                                                             const vec3& sun_direction,
                                                             float visibility_km,
                                                             const std::vector<vec2>& camera_response,
                                                             float albedo) const {
    if (!initialized) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::computeIntegratedSkyRadiance): Model not initialized.");
    }

    if (camera_response.empty()) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::computeIntegratedSkyRadiance): Empty camera response.");
    }

    // Get wavelength range
    float lambda_min = camera_response.front().x;
    float lambda_max = camera_response.back().x;

    if (lambda_min >= lambda_max) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::computeIntegratedSkyRadiance): Invalid wavelength range.");
    }

    // Determine number of wavelength samples for integration
    // Use adaptive sampling: more samples for broader bands
    float wavelength_span = lambda_max - lambda_min;
    int n_samples = std::max(10, std::min(50, static_cast<int>(wavelength_span / 20.0f)));

    float dlambda = wavelength_span / static_cast<float>(n_samples);

    // Integrate: ∫ L(λ) × R(λ) dλ
    // This gives the band-specific radiance weighted by camera spectral response
    double integrated_radiance = 0.0;

    for (int i = 0; i < n_samples; ++i) {
        // Sample at midpoint of interval
        float lambda = lambda_min + (i + 0.5f) * dlambda;

        // Interpolate camera response at this wavelength
        float response = interp1(camera_response, lambda);

        if (response <= 0.0f) {
            continue; // Skip wavelengths with zero response
        }

        // Query Prague model for sky radiance at this wavelength
        float sky_rad = getSkyRadiance(view_direction, sun_direction, lambda, visibility_km, albedo);

        // Integrate: ∫ L(λ) × R(λ) dλ
        // L(λ) is in W/m²/sr/nm, R(λ) is unitless (0-1), dλ is in nm
        // Result has units: W/m²/sr
        integrated_radiance += sky_rad * response * dlambda;
    }

    return static_cast<float>(integrated_radiance);
}

// Convert turbidity to visibility
float PragueSkyModelInterface::turbidityToVisibility(float turbidity) {
    // Koschmieder formula: V = 3.912 / (β × (λ/500)^α)
    // For 500 nm and typical conditions: V ≈ 3.9 / turbidity
    // where turbidity is Ångström AOD at 500 nm

    if (turbidity <= 0.0f) {
        // Very clear conditions - return maximum visibility
        return 131.8f;
    }

    float visibility_km = 3.9f / turbidity;

    // Clamp to Prague dataset range [20, 131.8 km]
    visibility_km = std::max(20.0f, std::min(131.8f, visibility_km));

    return visibility_km;
}

// Get available data ranges
void PragueSkyModelInterface::getAvailableRanges(float& min_wavelength_nm,
                                                  float& max_wavelength_nm,
                                                  float& min_visibility_km,
                                                  float& max_visibility_km,
                                                  float& min_elevation_deg,
                                                  float& max_elevation_deg) const {
    if (!initialized) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::getAvailableRanges): Model not initialized.");
    }

    try {
        PragueSkyModel::AvailableData available = model->getAvailableData();

        // Wavelength range
        min_wavelength_nm = static_cast<float>(available.channelStart);
        max_wavelength_nm = static_cast<float>(available.channelStart +
                                                available.channelWidth * (available.channels - 1));

        // Visibility range
        min_visibility_km = static_cast<float>(available.visibilityMin);
        max_visibility_km = static_cast<float>(available.visibilityMax);

        // Elevation range (convert from radians to degrees)
        min_elevation_deg = static_cast<float>(available.elevationMin * 180.0 / M_PI);
        max_elevation_deg = static_cast<float>(available.elevationMax * 180.0 / M_PI);

    } catch (const PragueSkyModel::NotInitializedException& e) {
        helios_runtime_error("ERROR (PragueSkyModelInterface::getAvailableRanges): Prague model not initialized.");
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ERROR (PragueSkyModelInterface::getAvailableRanges): Error querying available data: "
            << e.what();
        helios_runtime_error(oss.str());
    }
}
