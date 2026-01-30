/** \file "RadiationModel.h" Primary header file for radiation transport model.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef RADIATION_MODEL
#define RADIATION_MODEL

#include "CameraCalibration.h"
#include "Context.h"
#include "RayTracingBackend.h"
#include "json.hpp"

#include <utility>

//! Properties defining a radiation camera
struct CameraProperties {

    bool operator!=(const CameraProperties &rhs) const {
        return !(rhs == *this);
    }

    //! Camera sensor resolution (number of pixels) in the horizontal (.x) and vertical (.y) directions
    helios::int2 camera_resolution;

    //! Distance from the viewing plane to the focal plane (working distance for ray generation)
    float focal_plane_distance;

    //! Camera lens optical focal length in meters (characteristic of the lens, used for f-number and aperture calculations). This is the physical focal length, not the 35mm equivalent.
    float lens_focal_length;

    //! Diameter of the camera lens (lens_diameter = 0 gives a 'pinhole' camera with everything in focus)
    float lens_diameter;

    //! Camera horizontal field of view in degrees
    float HFOV;

    //! Ratio of camera horizontal field of view to vertical field of view (HFOV/VFOV). DEPRECATED: This parameter is auto-calculated from camera_resolution to ensure square pixels. Setting it explicitly will be ignored with a warning.
    float FOV_aspect_ratio;

    //! Physical sensor width in mm (default 35mm full-frame)
    float sensor_width_mm;

    //! Camera model name (e.g., "Nikon D700", "Canon EOS 5D")
    std::string model;

    //! Lens make/manufacturer (e.g., "Canon", "Nikon")
    std::string lens_make;

    //! Lens model name (e.g., "AF-S NIKKOR 50mm f/1.8G")
    std::string lens_model;

    //! Lens specification (e.g., "50mm f/1.8", "18-55mm f/3.5-5.6")
    std::string lens_specification;

    //! Exposure mode: "auto" (automatic exposure), "ISOXXX" (ISO-based, e.g., "ISO100"), or "manual" (no automatic exposure scaling). ISO mode is calibrated to match auto-exposure at reference settings (ISO 100, 1/125s, f/2.8) for typical Helios
    //! scenes.
    std::string exposure;

    //! Camera shutter speed in seconds (used for ISO-based exposure calculations). Example: 1/125 second = 0.008
    float shutter_speed;

    //! White balance mode: "auto" (automatic white balance using spectral response) or "off" (no white balance correction)
    std::string white_balance;

    /**! \brief Camera optical zoom multiplier (1.0 = no zoom, 2.0 = 2x zoom, etc.)
     *
     * This parameter scales the horizontal field of view (HFOV) during rendering: effective_HFOV = HFOV / camera_zoom.
     * The HFOV parameter represents the reference field of view at zoom=1.0. When zoom > 1.0, the effective FOV
     * becomes narrower (telephoto effect). When zoom < 1.0, the effective FOV becomes wider.
     *
     * IMPORTANT: All camera metadata exported to JSON (focal_length, sensor dimensions, etc.) reflect the REFERENCE
     * state at zoom=1.0, not the zoomed state. The zoom value itself is written to metadata as "zoom" so users
     * can reconstruct the effective parameters.
     *
     * Example: HFOV=60° with camera_zoom=2.0 renders with an effective HFOV of 30° (2x optical zoom).
     */
    float camera_zoom;

    CameraProperties() {
        camera_resolution = helios::make_int2(512, 512);
        focal_plane_distance = 1;
        lens_focal_length = 0.05; // 50mm default
        lens_diameter = 0.05;
        FOV_aspect_ratio = 0.f; // Sentinel value: 0 means auto-calculate from camera_resolution
        HFOV = 20.f;
        sensor_width_mm = 35.f;
        model = "generic";
        exposure = "auto";
        shutter_speed = 1.f / 125.f; // 1/125 second (standard default)
        white_balance = "auto";
        camera_zoom = 1.0f;
    }

    bool operator==(const CameraProperties &rhs) const {
        return camera_resolution == rhs.camera_resolution && focal_plane_distance == rhs.focal_plane_distance && lens_focal_length == rhs.lens_focal_length && lens_diameter == rhs.lens_diameter && FOV_aspect_ratio == rhs.FOV_aspect_ratio &&
               HFOV == rhs.HFOV && sensor_width_mm == rhs.sensor_width_mm && model == rhs.model && lens_make == rhs.lens_make && lens_model == rhs.lens_model && lens_specification == rhs.lens_specification && exposure == rhs.exposure &&
               shutter_speed == rhs.shutter_speed && white_balance == rhs.white_balance && camera_zoom == rhs.camera_zoom;
    }
};

//! Properties defining lens flare rendering parameters
struct LensFlareProperties {

    //! Number of aperture blades (affects starburst pattern). 6 blades produces 6-pointed star, 8 blades produces 8-pointed star, etc.
    int aperture_blade_count = 6;

    //! Anti-reflective coating efficiency (0.0-1.0). Higher values reduce ghost intensity. Typical modern coatings are 0.96-0.99.
    float coating_efficiency = 0.96f;

    //! Scale factor for ghost reflection intensity (0.0-1.0+). Default 1.0 uses physically-derived intensity.
    float ghost_intensity = 1.0f;

    //! Scale factor for starburst/diffraction pattern intensity (0.0-1.0+). Default 1.0 uses physically-derived intensity.
    float starburst_intensity = 1.0f;

    //! Minimum normalized pixel intensity (0.0-1.0) required to generate lens flare. Pixels below this threshold are ignored.
    float intensity_threshold = 0.8f;

    //! Number of ghost reflections to render. More ghosts increase realism but also computation time.
    int ghost_count = 5;

    bool operator==(const LensFlareProperties &rhs) const {
        return aperture_blade_count == rhs.aperture_blade_count && coating_efficiency == rhs.coating_efficiency && ghost_intensity == rhs.ghost_intensity && starburst_intensity == rhs.starburst_intensity &&
               intensity_threshold == rhs.intensity_threshold && ghost_count == rhs.ghost_count;
    }

    bool operator!=(const LensFlareProperties &rhs) const {
        return !(rhs == *this);
    }
};

//! Data object for a radiation camera
struct RadiationCamera {

    // Constructor
    RadiationCamera(std::string initlabel, const std::vector<std::string> &band_label, const helios::vec3 &initposition, const helios::vec3 &initlookat, const CameraProperties &camera_properties, uint initantialiasing_samples) :
        label(std::move(initlabel)), band_labels(band_label), position(initposition), lookat(initlookat), antialiasing_samples(initantialiasing_samples) {
        for (const auto &band: band_label) {
            band_spectral_response[band] = "uniform";
        }
        focal_length = camera_properties.focal_plane_distance; // working distance for ray generation
        lens_focal_length = camera_properties.lens_focal_length; // optical focal length for aperture
        resolution = camera_properties.camera_resolution;
        lens_diameter = camera_properties.lens_diameter;
        HFOV_degrees = camera_properties.HFOV;
        FOV_aspect_ratio = camera_properties.FOV_aspect_ratio;
        sensor_width_mm = camera_properties.sensor_width_mm;
        model = camera_properties.model;
        lens_make = camera_properties.lens_make;
        lens_model = camera_properties.lens_model;
        lens_specification = camera_properties.lens_specification;
        exposure = camera_properties.exposure;
        shutter_speed = camera_properties.shutter_speed;
        white_balance = camera_properties.white_balance;
        camera_zoom = camera_properties.camera_zoom;
    }

    // Label for camera array
    std::string label;
    // Cartesian (x,y,z) position of camera array center
    helios::vec3 position;
    // Direction camera is pointed (normal vector of camera surface). This vector will automatically be normalized
    helios::vec3 lookat;
    // Physical dimensions of the camera lens
    float lens_diameter;
    // Resolution of camera sub-divisions (i.e., pixels)
    helios::int2 resolution;
    // camera focal length (working distance for ray generation)
    float focal_length;
    // lens optical focal length (for f-number and aperture calculations). This is the physical focal length, not the 35mm equivalent.
    float lens_focal_length;
    // camera horizontal field of view (degrees)
    float HFOV_degrees;
    // Ratio of camera horizontal field of view to vertical field of view
    float FOV_aspect_ratio;
    // Physical sensor width in mm
    float sensor_width_mm;
    // Camera model name
    std::string model;
    // Lens make/manufacturer
    std::string lens_make;
    // Lens model name
    std::string lens_model;
    // Lens specification
    std::string lens_specification;
    // Exposure mode: "auto", "ISOXXX" (e.g., "ISO100"), or "manual"
    std::string exposure;
    // Camera shutter speed in seconds
    float shutter_speed;
    // White balance mode: "auto" or "off"
    std::string white_balance;
    // Camera optical zoom multiplier
    float camera_zoom;
    // Camera type (rgb, spectral, or thermal)
    std::string camera_type;
    // Number of antialiasing samples per pixel
    uint antialiasing_samples;

    std::vector<std::string> band_labels;

    std::map<std::string, std::string> band_spectral_response;

    std::map<std::string, std::vector<float>> pixel_data;

    std::vector<uint> pixel_label_UUID;
    std::vector<float> pixel_depth;

    //! Flag indicating whether lens flare rendering is enabled for this camera
    bool lens_flare_enabled = false;

    //! Lens flare rendering properties
    LensFlareProperties lens_flare_properties;

    //! Normalize all pixel data in the camera such that the maximum pixel value is 1.0 and the minimum is 0.0 (no clamping applied)
    void normalizePixels();

    //! Apply auto-exposure scaling to image data to scale the average luminance to a target value.
    /**
     * Computes the image’s mean luminance and applies a single uniform gain so that the scene-average luminance becomes the specified grey_target value.
     *
     * \param[in] target [optional] Target average luminance value. Default is 18%.
     */

    //! Apply auto-white balancing to image data based on Gray World assumption using Minkowski mean
    /**
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] p [optional] Minkowski mean parameter. Default is 5.0.
     */
    void whiteBalance(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float p = 5.0);

    //! Apply Gray Edge white balancing algorithm
    /**
     * Uses edge information to estimate illuminant, assuming edge differences are achromatic on average.
     * Works better than Gray World for vegetation and textured scenes.
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] derivative_order [optional] Order of derivative (1 or 2). Default is 1.
     * \param[in] p [optional] Minkowski norm parameter. Default is 5.0.
     */
    void whiteBalanceGrayEdge(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, int derivative_order = 1, float p = 5.0);

    //! Apply White Patch white balancing algorithm
    /**
     * Assumes brightest pixels in the scene represent white objects under the illuminant.
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] percentile [optional] Percentile of brightest pixels to use. Default is 0.99 (top 1%).
     */
    void whiteBalanceWhitePatch(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float percentile = 0.99f);


    //! Apply spectral-based white balance using integrated camera response curves
    /**
     * Normalizes image channels based on the integrated spectral response of each camera band.
     * This method assumes a flat light source spectrum and normalizes each channel such that
     * an object with flat spectral reflectance appears correctly white balanced.
     * Each channel is multiplied by the reciprocal of its integrated spectral response.
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] context Pointer to Helios context for accessing spectral data
     */
    void whiteBalanceSpectral(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, helios::Context *context);

    //! Apply Reinhard tone mapping curve to image data
    /**
     * The Reinhard curve applies a simple global tone mapping to compress high dynamic range into displayable range while preserving chroma.
     *
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     */
    void reinhardToneMapping(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label);

    //! Apply image gain
    /**
     * Computes the specified percentile of the per-pixel maximum channel values and multiplies all channels by the reciprocal of that percentile so that it is mapped to full white (1.0).
     *
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] percentile [optional] Percentile to use for gain computation (e.g., 0.9 for 90th percentile).
     */
    void applyGain(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float percentile = 0.95f);

    //! Applies HDR toning to the specified color bands based on the local adaptation method
    /**
     * \param[in] red_band_label Label for the red color band to be processed.
     * \param[in] green_band_label Label for the green color band to be processed.
     * \param[in] blue_band_label Label for the blue color band to be processed.
     */
    void adjustSBC(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation, float brightness, float contrast);

    //! Apply the color correction matrix to image data
    /**
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     */
    // void applyCCM(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label);

    //! Apply gamma compression to image data
    /**
     * Applies the standard sRGB electro-optical transfer function to each channel of a linear-light image—clamping negatives to zero and limiting outputs to [0,1]—thereby encoding the data into display-ready sRGB space.  This final step ensures that
     * pixel values map correctly to human‐perceived brightness on typical monitors.
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     */
    void gammaCompress(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label);

    /**
     * \brief Performs global histogram equalization on the specified color bands.
     *
     * \param[in] red_band_label Label for the red band to be processed.
     * \param[in] green_band_label Label for the green band to be processed.
     * \param[in] blue_band_label Label for the blue band to be processed.
     */
    void globalHistogramEqualization(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label);

    //! Apply percentile-based auto-exposure to optimize scene brightness
    /**
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] gain_multiplier Additional gain factor to apply after auto-exposure
     */
    void autoExposure(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float gain_multiplier);

    //! Apply camera exposure based on the camera's exposure setting
    /**
     * Applies exposure scaling to pixel_data based on the camera's exposure mode:
     * - "auto": Automatic exposure using percentile-based normalization (18% gray target for RGB, per-band for spectral)
     * - "ISOXXX" (e.g., "ISO100"): Fixed exposure based on ISO, shutter speed, and aperture settings.
     *   Calibrated to match auto-exposure at reference settings (ISO 100, 1/125s, f/2.8) for typical Helios scenes.
     *   Higher ISO/longer shutter/wider aperture → proportionally brighter.
     * - "manual": No automatic exposure scaling applied
     *
     * This method should be called after rendering is complete and pixel_data is populated.
     * \param[in] context Pointer to Helios context for accessing spectral data
     */
    void applyCameraExposure(helios::Context *context);

    //! Apply automatic white balance correction to camera image data
    /**
     * Applies white balance correction based on the camera's white_balance setting:
     * - "auto": Applies spectral white balance using camera spectral response curves
     * - "off": No white balance correction applied
     *
     * White balance is automatically skipped for single-channel (grayscale) images.
     * For multi-channel images, white balance is applied simultaneously to all channels.
     *
     * This method should be called after rendering and exposure adjustment are complete.
     * \param[in] context Pointer to Helios context for accessing spectral data
     */
    void applyCameraWhiteBalance(helios::Context *context);

    //! Adjust brightness and contrast of image data
    /**
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] brightness Brightness adjustment factor (1.0 = no change)
     * \param[in] contrast Contrast adjustment factor (1.0 = no change)
     */
    void adjustBrightnessContrast(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float brightness, float contrast);

    //! Adjust color saturation of image data
    /**
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] saturation Saturation adjustment factor (1.0 = no change, 0.0 = grayscale, >1.0 = more saturated)
     */
    void adjustSaturation(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation);

    //! Converts a linear color value to sRGB color space.
    /**
     * \param[in] x Input value to be converted. Values > 1.0 are clipped to white (1.0).
     * \return Corresponding value in the sRGB color space, clamped to [0.0, 1.0].
     */
    static float lin_to_srgb(float x) noexcept {
        // Clamp negative values to 0, bright values > 1.0 to white (1.0)
        if (x <= 0.0f)
            return 0.0f;
        if (x >= 1.0f)
            return 1.0f; // Bright pixels clipped to white
        return (x <= 0.0031308f) ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    }

    //! Converts an sRGB color component to its linear representation.
    /**
     * \param[in] v sRGB color component in the range [0, 1]
     * \return Corresponding linear color component in the range [0, 1]
     */
    static float srgb_to_lin(float v) noexcept {
        return (v <= 0.04045f) ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    }

private:
    //! Computes the luminance of a color given its red, green, and blue components.
    /**
     * \param[in] red Red component of the color.
     * \param[in] green Green component of the color.
     * \param[in] blue Blue component of the color.
     * \return The luminance value calculated as a weighted sum of the red, green, and blue components.
     */
    static float luminance(float red, float green, float blue) noexcept {
        return 0.2126f * red + 0.7152f * green + 0.0722f * blue;
    }
};

//! Struct to store camera tile information for tiled rendering
struct CameraTile {
    helios::int2 resolution; //!< Tile dimensions (width, height)
    helios::int2 offset; //!< Tile offset in full image (x, y)
};

//! Metadata for radiation camera image export
struct CameraMetadata {

    //! Full path to the associated image file
    std::string path;

    //! Camera intrinsic properties
    struct CameraProperties {
        int height; //!< Image height in pixels
        int width; //!< Image width in pixels
        int channels; //!< Number of spectral bands (1=grayscale, 3=RGB)
        std::string type; //!< Camera type: "rgb" (3-channel), "spectral" (1-N channel), or "thermal" (1 channel thermal)
        float focal_length; //!< Optical focal length in mm (physical focal length, not 35mm equivalent)
        std::string aperture; //!< Aperture f-stop (e.g., "f/2.8" or "pinhole")
        float sensor_width; //!< Physical sensor width in mm
        float sensor_height; //!< Physical sensor height in mm
        std::string model; //!< Camera model name (e.g., "Nikon D700", "generic")
        std::string lens_make; //!< Lens make/manufacturer (e.g., "Canon", "Nikon")
        std::string lens_model; //!< Lens model name (e.g., "AF-S NIKKOR 50mm f/1.8G")
        std::string lens_specification; //!< Lens specification (e.g., "50mm f/1.8", "18-55mm f/3.5-5.6")
        std::string exposure; //!< Exposure mode: "auto", "ISOXXX" (e.g., "ISO100"), or "manual"
        float shutter_speed; //!< Shutter speed in seconds (e.g., 0.008 for 1/125s)
        std::string white_balance; //!< White balance mode: "auto" or "off"
        float camera_zoom; //!< Camera optical zoom multiplier (1.0 = no zoom, 2.0 = 2x zoom)
    } camera_properties;

    //! Geographic location properties
    struct LocationProperties {
        float latitude; //!< Latitude in degrees (+N/-S)
        float longitude; //!< Longitude in degrees (+E/-W)
    } location_properties;

    //! Image acquisition properties
    struct AcquisitionProperties {
        std::string date; //!< Acquisition date (YYYY-MM-DD format)
        std::string time; //!< Acquisition time (HH:MM:SS format)
        float UTC_offset; //!< UTC offset in hours
        float camera_height_m; //!< Camera height above ground in meters
        float camera_angle_deg; //!< Camera tilt angle from horizontal (0=horizontal, 90=down)
        std::string light_source; //!< Lighting type: "sunlight", "artificial", "mixed", or "none"
    } acquisition_properties;

    //! Image processing corrections applied to the image
    struct ImageProcessingProperties {
        float saturation_adjustment = 1.f; //!< Saturation adjustment factor (1.0 = no change)
        float brightness_adjustment = 1.f; //!< Brightness adjustment factor (1.0 = no change)
        float contrast_adjustment = 1.f; //!< Contrast adjustment factor (1.0 = no change)
        std::string color_space = "linear"; //!< Output color space: "linear" or "sRGB"
    } image_processing;

    //! Agronomic properties derived from plant architecture data
    struct AgronomicProperties {
        std::vector<std::string> plant_species; //!< List of unique plant species visible in image (from object data 'plant_name')
        std::vector<int> plant_count; //!< Number of plants per species, parallel to plant_species (from unique 'plantID' values)
        std::vector<float> plant_height_m; //!< Average height of plants per species in meters, parallel to plant_species (from object data 'plant_height')
        std::vector<float> plant_age_days; //!< Average age of plants per species in days, parallel to plant_species (from object data 'age')
        std::vector<std::string> plant_stage; //!< Most common phenological stage per species, parallel to plant_species (from object data 'phenology_stage')
        std::vector<float> leaf_area_m2; //!< Total visible leaf area per species in square meters, parallel to plant_species (computed from leaf primitive areas)
        std::string weed_pressure; //!< Weed pressure classification: "low" (0-20%), "moderate" (21-40%), "high" (>40%), or empty if no data
    } agronomic_properties;
};

//! Properties defining a radiation band
struct RadiationBand {

    //! Constructor
    explicit RadiationBand(std::string a_label, size_t directRayCount_default, size_t diffuseRayCount_default, float diffuseFlux_default, uint scatteringDepth_default, float minScatterEnergy_default) : label(std::move(a_label)) {
        directRayCount = directRayCount_default;
        diffuseRayCount = diffuseRayCount_default;
        diffuseFlux = diffuseFlux_default;
        scatteringDepth = scatteringDepth_default;
        minScatterEnergy = minScatterEnergy_default;
        diffuseExtinction = 0.f;
        diffuseDistNorm = 1.f;
        emissionFlag = true;
        wavebandBounds = helios::make_vec2(0, 0);
    }

    //! Label for band
    std::string label;

    //! Number of direct rays launched per element
    size_t directRayCount;

    //! Number of diffuse rays launched per element
    size_t diffuseRayCount;

    //! Diffuse component of radiation flux integrated over wave band
    float diffuseFlux;

    //! Distribution coefficient of ambient diffuse radiation for wave band
    float diffuseExtinction;

    //! Direction of peak in ambient diffuse radiation
    helios::vec3 diffusePeakDir;

    //! Diffuse distribution normalization factor
    float diffuseDistNorm;

    //! Prague sky model angular parameters for diffuse distribution
    //! (circumsolar_strength, circumsolar_width, horizon_brightness, normalization)
    //! If normalization (w component) == 0, Prague not active for this band
    helios::vec4 diffusePragueParams = helios::make_vec4(0, 0, 0, 0);

    //! Spectral distribution of diffuse radiation flux for wave band
    std::vector<helios::vec2> diffuse_spectrum;

    //! Scattering depth for wave band
    uint scatteringDepth;

    //! Minimum energy for scattering for wave band
    float minScatterEnergy;

    //! Flag that determines if emission calculations are performed for wave band
    bool emissionFlag;

    //! Waveband range of band
    helios::vec2 wavebandBounds;
};

//! Possible types of radiation sources
enum RadiationSourceType { RADIATION_SOURCE_TYPE_COLLIMATED = 0, RADIATION_SOURCE_TYPE_SPHERE = 1, RADIATION_SOURCE_TYPE_SUN_SPHERE = 2, RADIATION_SOURCE_TYPE_RECTANGLE = 3, RADIATION_SOURCE_TYPE_DISK = 4 };

//! Radiation source data object
struct RadiationSource {
public:
    //! Constructor for collimated radiation source
    explicit RadiationSource(const helios::vec3 &position) : source_position(position) {
        source_type = RADIATION_SOURCE_TYPE_COLLIMATED;

        // initialize other unused variables
        source_position_scaling_factor = 1.f;
        source_flux_scaling_factor = 1.f;
    };

    //! Constructor for spherical radiation source
    RadiationSource(const helios::vec3 &position, float width) : source_position(position) {
        source_type = RADIATION_SOURCE_TYPE_SPHERE;

        source_width = helios::make_vec2(width, width);

        // initialize other unused variables
        source_position_scaling_factor = 1.f;
        source_flux_scaling_factor = 1.f;
    };

    //! Constructor for sun sphere radiation source
    RadiationSource(const helios::vec3 &position, float position_scaling_factor, float width, float flux_scaling_factor) :
        source_position(position), source_position_scaling_factor(position_scaling_factor), source_flux_scaling_factor(flux_scaling_factor) {
        source_type = RADIATION_SOURCE_TYPE_SUN_SPHERE;
        source_width = helios::make_vec2(width, width);
    };


    //! Constructor for rectangular radiation source
    RadiationSource(const helios::vec3 &position, const helios::vec2 &size, const helios::vec3 &rotation) : source_position(position), source_width(size), source_rotation(rotation) {
        source_type = RADIATION_SOURCE_TYPE_RECTANGLE;

        // initialize other unused variables
        source_position_scaling_factor = 1.f;
        source_flux_scaling_factor = 1.f;
    };

    //! Constructor for disk radiation source
    RadiationSource(const helios::vec3 &position, float width, const helios::vec3 &rotation) : source_position(position), source_rotation(rotation) {
        source_type = RADIATION_SOURCE_TYPE_DISK;

        source_width = helios::make_vec2(width, width);

        // initialize other unused variables
        source_position_scaling_factor = 1.f;
        source_flux_scaling_factor = 1.f;
    };

    //! Positions of radiation source
    helios::vec3 source_position;

    //! Source position factors used to scale position in case of a sun sphere source
    float source_position_scaling_factor;

    //! Spectral distribution of radiation source. Each element of the vector is a wavelength band, where .x is the wavelength in nm and .y is the spectral flux in W/m^2/nm.
    std::vector<helios::vec2> source_spectrum;

    std::string source_spectrum_label = "none";

    //! Cached version number of source spectrum for change detection
    uint64_t source_spectrum_version = 0;

    //! Widths for each radiation source (N/A for collimated sources)
    helios::vec2 source_width;

    //! Rotation (rx,ry,rz) for radiation source (area sources only)
    helios::vec3 source_rotation;

    //! Source flux factors used to scale flux in case of a sun sphere source
    float source_flux_scaling_factor;

    //! Types of all radiation sources
    RadiationSourceType source_type;

    //! Fluxes of radiation source for all bands
    std::map<std::string, float> source_fluxes;
};

//! Radiation transport model plugin
class RadiationModel {
public:
    //! Default constructor
    explicit RadiationModel(helios::Context *context);

    //! Destructor
    ~RadiationModel();

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed
     */
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Disable/silence status messages
    /**
     * \note Error messages are still displayed.
     */
    void disableMessages();

    //! Enable status messages
    void enableMessages();

    //! Add optional output primitive data values to the Context
    /**
     * \param[in] label Name of primitive data (e.g., "reflectivity", "transmissivity")
     */
    void optionalOutputPrimitiveData(const char *label);

    //! Sets variable directRayCount, the number of rays to be used in direct radiation model.
    /**
     * \param[in] label Label used to reference the band
     * \param[in] N Number of rays
     * \note Default is 100 rays/primitive.
     */
    void setDirectRayCount(const std::string &label, size_t N);

    //! Sets variable diffuseRayCount, the number of rays to be used in diffuse (ambient) radiation model.
    /**
     * \param[in] label Label used to reference the band
     * \param[in] N Number of rays
     * \note Default is 1000 rays/primitive.
     */
    void setDiffuseRayCount(const std::string &label, size_t N);

    //! Diffuse (ambient) radiation flux
    /**
     * Diffuse component of radiation incident on a horizontal surface above all geometry in the domain.
     * \param[in] label Label used to reference the band
     * \param[in] flux Radiative flux
     */
    void setDiffuseRadiationFlux(const std::string &label, float flux);

    //! Extinction coefficient of diffuse ambient radiation
    /**
     * The angular distribution of diffuse ambient radiation is computed according to N = Psi^-K, where Psi is the angle between the distribution peak (usually the sun direction) and the ambient direction, and K is the extinction coefficient. When
     * K=0 the ambient distribution is uniform, which is the default setting
     * \param[in] label Label used to reference the radiative band
     * \param[in] K Extinction coefficient value
     * \param[in] peak_dir Unit vector pointing in the direction of the peak in diffuse radiation (this is usually the sun direction)
     */
    void setDiffuseRadiationExtinctionCoeff(const std::string &label, float K, const helios::vec3 &peak_dir);

    //! Extinction coefficient of diffuse ambient radiation
    /**
     * The angular distribution of diffuse ambient radiation is computed according to N = Psi^-K, where Psi is the angle between the distribution peak (usually the sun direction) and the ambient direction, and K is the extinction coefficient. When
     * K=0 the ambient distribution is uniform, which is the default setting
     * \param[in] label Label used to reference the radiative band
     * \param[in] K Extinction coefficient value
     * \param[in] peak_dir Spherical direction of the peak in diffuse radiation (elevation and azimuth angles in radians, this is usually the sun direction)
     */
    void setDiffuseRadiationExtinctionCoeff(const std::string &label, float K, const helios::SphericalCoord &peak_dir);

    //! Scale the global diffuse spectrum so its integral equals the specified value (=∫Sdλ)
    /**
     * Scales the global diffuse spectrum (set via setDiffuseSpectrum()) so that its integral over all wavelengths
     * equals the specified value. The scaled spectrum is applied to all existing radiation bands and will be
     * inherited by any bands created subsequently. This should be called after setDiffuseSpectrum().
     * \param[in] spectrum_integral Desired integration of spectral flux distribution across all wavelengths (=∫Sdλ)
     */
    void setDiffuseSpectrumIntegral(float spectrum_integral);

    //! Scale the global diffuse spectrum based on a prescribed integral between two wavelengths (=∫Sdλ)
    /**
     * Scales the global diffuse spectrum (set via setDiffuseSpectrum()) so that its integral between the specified
     * wavelengths equals the given value. The entire spectrum is scaled uniformly. The scaled spectrum is applied
     * to all existing radiation bands and will be inherited by any bands created subsequently.
     * \param[in] spectrum_integral Desired integration of spectral flux distribution between the specified wavelengths (=∫Sdλ)
     * \param[in] wavelength_min Lower bounding wavelength for integration range (nm)
     * \param[in] wavelength_max Upper bounding wavelength for integration range (nm)
     */
    void setDiffuseSpectrumIntegral(float spectrum_integral, float wavelength_min, float wavelength_max);

    //! Set the integral of the diffuse spectral flux distribution across all possible wavelengths (=∫Sdλ)
    /**
     * \param[in] band_label Label used to reference the band
     * \param[in] spectrum_integral Integration of source spectral flux distribution across all possible wavelengths (=∫Sdλ)
     * \note This function will call setDiffuseFlux() for all bands to update source fluxes based on the new spectrum integral
     */
    void setDiffuseSpectrumIntegral(const std::string &band_label, float spectrum_integral);

    //! Scale the source spectral flux distribution based on a prescribed integral between two wavelengths (=∫Sdλ)
    /**
     * \param[in] band_label Label used to reference the band
     * \param[in] spectrum_integral Integration of source spectral flux distribution between two wavelengths (=∫Sdλ)
     * \param[in] wavelength_min Lower bounding wavelength for wave band
     * \param[in] wavelength_max Upper bounding wavelength for wave band
     * \note This function will call setDiffuseFlux() for all bands to update source fluxes based on the new spectrum integral
     */
    void setDiffuseSpectrumIntegral(const std::string &band_label, float spectrum_integral, float wavelength_min, float wavelength_max);

    //! Add a spectral radiation band to the model
    /**
     * \param[in] label Label used to reference the band
     */
    void addRadiationBand(const std::string &label);

    //! Add a spectral radiation band to the model with explicit specification of the spectral wave band
    /**
     * \param[in] label Label used to reference the band
     * \param[in] wavelength_min Lower bounding wavelength for wave band
     * \param[in] wavelength_max Upper bounding wavelength for wave band
     */
    void addRadiationBand(const std::string &label, float wavelength_min, float wavelength_max);

    //! Copy a spectral radiation band based on a previously created band
    /**
     * \param[in] old_label Label of old radiation band to be copied
     * \param[in] new_label Label of new radiation band to be created
     */
    void copyRadiationBand(const std::string &old_label, const std::string &new_label);

    //! Copy a spectral radiation band based on a previously created band and explicitly set new band wavelength range
    /**
     * \param[in] old_label Label of old radiation band to be copied
     * \param[in] new_label Label of new radiation band to be created
     * \param[in] wavelength_min Lower bounding wavelength for wave band
     * \param[in] wavelength_max Upper bounding wavelength for wave band
     */
    void copyRadiationBand(const std::string &old_label, const std::string &new_label, float wavelength_min, float wavelength_max);

    //! Check if a radiation band exists based on its label
    /**
     * \param[in] label Label used to reference the band
     */
    bool doesBandExist(const std::string &label) const;

    //! Disable emission calculations for all primitives in this band.
    /**
     * \param[in] label Label used to reference the band
     */
    void disableEmission(const std::string &label);

    //! Enable emission calculations for all primitives in this band.
    /**
     * \param[in] label Label used to reference the band
     */
    void enableEmission(const std::string &label);

    //! Add an external source of collimated radiation (i.e., source at infinite distance with parallel rays) assuming the default direction of (0,0,1)
    /**
     * \return Source identifier
     */
    uint addCollimatedRadiationSource();

    //! Add an external source of collimated radiation (i.e., source at infinite distance with parallel rays)
    /**
     * \param[in] direction Spherical coordinate pointing toward the radiation source (elevation and azimuth angles in radians)
     * \return Source identifier
     */
    uint addCollimatedRadiationSource(const helios::SphericalCoord &direction);

    //! Add an external source of collimated radiation (i.e., source at infinite distance with parallel rays)
    /**
     * \param[in] direction unit vector pointing toward the radiation source
     * \return Source identifier
     */
    uint addCollimatedRadiationSource(const helios::vec3 &direction);

    //! Add an external source of radiation that emits from the surface of a sphere.
    /**
     * \param[in] position (x,y,z) position of the center of the sphere radiation source
     * \param[in] radius Radius of the sphere radiation source
     * \return Source identifier
     */
    uint addSphereRadiationSource(const helios::vec3 &position, float radius);

    //! Add a sphere radiation source that models the sun assuming the default direction of (0,0,1)
    /**
     * \return Source identifier
     */
    uint addSunSphereRadiationSource();

    //! Add a sphere radiation source that models the sun
    /**
     * \param[in] sun_direction Spherical coordinate pointing towards the sun
     * \return Source identifier
     */
    uint addSunSphereRadiationSource(const helios::SphericalCoord &sun_direction);

    //! Add a sphere radiation source that models the sun
    /**
     * \param[in] sun_direction Unit vector pointing towards the sun
     * \return Source identifier
     */
    uint addSunSphereRadiationSource(const helios::vec3 &sun_direction);

    //! Add planar rectangular radiation source
    /**
     * \param[in] position  (x,y,z) position of the center of the rectangular radiation source
     * \param[in] size Length (.x) and width (.y) of rectangular source
     * \param[in] rotation_rad Rotation of the source in radians about the x- y- and z- axes (the sign of the rotation angle follows right-hand rule)
     * \return Source identifier
     */
    uint addRectangleRadiationSource(const helios::vec3 &position, const helios::vec2 &size, const helios::vec3 &rotation_rad);

    //! Add planar circular radiation source
    /**
     * \param[in] position  (x,y,z) position of the center of the disk radiation source
     * \param[in] radius Radius of disk source
     * \param[in] rotation_rad Rotation of the source in radians about the x- y- and z- axes (the sign of the rotation angle follows right-hand rule)
     * \return Source identifier
     */
    uint addDiskRadiationSource(const helios::vec3 &position, float radius, const helios::vec3 &rotation_rad);

    //! Delete an existing radiation source (any type)
    /**
     * \param[in] sourceID Identifier of radiation source
     */
    void deleteRadiationSource(uint sourceID);

    //! Set the integral of the source spectral flux distribution across all possible wavelengths (=∫Sdλ)
    /**
     * \param[in] source_ID ID of source
     * \param[in] source_integral Integration of source spectral flux distribution across all possible wavelengths (=∫Sdλ)
     * \note This function will call setSourceFlux() for all bands to update source fluxes based on the new spectrum integral
     */
    void setSourceSpectrumIntegral(uint source_ID, float source_integral);

    //! Scale the source spectral flux distribution based on a prescribed integral between two wavelengths (=∫Sdλ)
    /**
     * \param[in] source_ID ID of source
     * \param[in] source_integral Integration of source spectral flux distribution between two wavelengths (=∫Sdλ)
     * \param[in] wavelength_min Lower bounding wavelength for wave band
     * \param[in] wavelength_max Upper bounding wavelength for wave band
     * \note This function will call setSourceFlux() for all bands to update source fluxes based on the new spectrum integral
     */
    void setSourceSpectrumIntegral(uint source_ID, float source_integral, float wavelength_min, float wavelength_max);

    //! Set the flux of radiation source for this band.
    /**
     * \param[in] source_ID Identifier of radiation source
     * \param[in] band_label Label used to reference the band
     * \param[in] flux Radiative flux normal to the direction of radiation propagation
     */
    void setSourceFlux(uint source_ID, const std::string &band_label, float flux);

    //! Set the flux of multiple radiation sources for this band.
    /**
     * \param[in] source_ID Vector of radiation source identifiers
     * \param[in] band_label Label used to reference the band
     * \param[in] flux Radiative flux normal to the direction of radiation propagation
     */
    void setSourceFlux(const std::vector<uint> &source_ID, const std::string &band_label, float flux);

    //! Get the flux of radiation source for this band
    /**
     * \param[in] source_ID Identifier of radiation source
     * \param[in] band_label Label used to reference the band
     * \return Radiative flux normal to the direction of radiation propagation
     */
    float getSourceFlux(uint source_ID, const std::string &band_label) const;

    //! Set the position/direction of radiation source based on a Cartesian vector
    /**
     * \param[in] source_ID Identifier of radiation source
     * \param[in] position If point source - (x,y,z) position of the radiation source. If collimated source - (nx,ny,nz) unit vector pointing toward the source.
     */
    void setSourcePosition(uint source_ID, const helios::vec3 &position);

    //! Set the position/direction of radiation source based on a spherical vector
    /**
     * \param[in] source_ID Identifier of radiation source
     * \param[in] position If point source - (radius,elevation,azimuth) position of the radiation source (elevation and azimuth angles in radians). If collimated source - (elevation,azimuth) vector pointing toward the source (elevation and azimuth
     * angles in radians, radius is ignored).
     */
    void setSourcePosition(uint source_ID, const helios::SphericalCoord &position);

    //! Get the position/direction of radiation source
    /**
     * \param[in] source_ID Identifier of radiation source
     * \return If point source - (x,y,z) position of the radiation source. If collimated source - (nx,ny,nz) unit vector pointing toward the source.
     */
    helios::vec3 getSourcePosition(uint source_ID) const;

    //! Set the spectral distribution of a radiation source according to a vector of wavelength-intensity pairs.
    /**
     * \param[in] source_ID Identifier of radiation source.
     * \param[in] spectrum Vector containing spectral intensity data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity (.y).
     */
    void setSourceSpectrum(uint source_ID, const std::vector<helios::vec2> &spectrum);

    //! Set the spectral distribution of multiple radiation sources according to a vector of wavelength-intensity pairs.
    /**
     * \param[in] source_ID Vector of radiation source identifiers.
     * \param[in] spectrum Vector containing spectral intensity data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity (.y).
     */
    void setSourceSpectrum(const std::vector<uint> &source_ID, const std::vector<helios::vec2> &spectrum);

    //! Set the spectral distribution of a radiation source based on global data of wavelength-intensity pairs.
    /**
     * \param[in] source_ID Identifier of radiation source.
     * \param[in] spectrum_label Label of global data containing spectral intensity data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity (.y).
     */
    void setSourceSpectrum(uint source_ID, const std::string &spectrum_label);

    //! Set the spectral distribution of multiple radiation sources based on global data of wavelength-intensity pairs.
    /**
     * \param[in] source_ID Vector of radiation source identifiers.
     * \param[in] spectrum_label Label of global data containing spectral intensity data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity (.y).
     */
    void setSourceSpectrum(const std::vector<uint> &source_ID, const std::string &spectrum_label);

    //! Set the spectral distribution of diffuse ambient radiation for all bands based on global data of wavelength-intensity pairs.
    /**
     * \param[in] spectrum_label Label of global data containing spectral intensity data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity (.y).
     * \note For emission-enabled bands, getDiffuseFlux() returns 0 since diffuse sky radiation is not relevant for thermal bands. Use setDiffuseRadiationFlux() to manually set flux for emission bands if needed.
     */
    void setDiffuseSpectrum(const std::string &spectrum_label);

    //! Get the diffuse flux for a given band
    /**
     * \param[in] band_label Label used to reference the band
     * \return Diffuse flux for the band
     */
    float getDiffuseFlux(const std::string &band_label) const;

    //! Add a 3D model of the light source (rectangular, disk, and sphere) to the Context for visualization purposes
    void enableLightModelVisualization();

    //! Remove the 3D model of the light source from the Context
    void disableLightModelVisualization();

    //! Add a 3D model of the camera to the Context for visualization purposes
    void enableCameraModelVisualization();

    //! Remove the 3D model of the camera from the Context
    void disableCameraModelVisualization();

    //! Integrate a spectral distribution between two wavelength bounds
    /**
     * \param[in] object_spectrum Vector containing spectral data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity/reflectivity (.y).
     * \param[in] wavelength_min Wavelength for lower bounds of integration
     * \param[in] wavelength_max Wavelength for upper bounds of integration
     * \return Integral of spectral data from wavelength_min to wavelength_max
     */
    float integrateSpectrum(const std::vector<helios::vec2> &object_spectrum, float wavelength_min, float wavelength_max) const;

    //! Integrate a spectral distribution across all wavelengths
    /**
     * \param[in] object_spectrum Vector containing spectral data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity/reflectivity (.y).
     * \return Integral of spectral data from minimum to maximum wavelength
     */
    float integrateSpectrum(const std::vector<helios::vec2> &object_spectrum) const;

    //! Integrate the product of a radiation source spectral distribution with specified spectral data between two wavelength bounds
    /**
     * \param[in] source_ID Identifier of a radiation source.
     * \param[in] object_spectrum Vector containing spectral data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity/reflectivity (.y).
     * \param[in] wavelength_min Wavelength for lower bounds of integration
     * \param[in] wavelength_max Wavelength for upper bounds of integration
     * \return Integral of product of source energy spectrum and spectral data from minimum to maximum wavelength
     */
    float integrateSpectrum(uint source_ID, const std::vector<helios::vec2> &object_spectrum, float wavelength_min, float wavelength_max) const;

    //! Integrate the product of a radiation source spectral distribution, surface spectral data, and camera spectral response across all wavelengths
    /**
     * \param[in] source_ID Identifier of a radiation source.
     * \param[in] object_spectrum Vector containing surface spectral data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity/reflectivity (.y).
     * \param[in] camera_spectrum Vector containing camera spectral response data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity/reflectivity (.y).
     * \return Integral of product of a radiation source spectral distribution, surface spectral data, and camera spectral response across all wavelengths
     */
    float integrateSpectrum(uint source_ID, const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum) const;

    //! Integrate the product of surface spectral data and camera spectral response across all wavelengths
    /**
     * \param[in] object_spectrum Vector containing surface spectral data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity/reflectivity (.y).
     * \param[in] camera_spectrum Vector containing camera spectral response data. Each index of "spectrum" gives the wavelength (.x) and spectral intensity/reflectivity (.y).
     * \return Integral of product of a radiation source spectral distribution, surface spectral data, and camera spectral response across all wavelengths
     */
    float integrateSpectrum(const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum) const;

    //! Integrate a source spectral distribution between two wavelength bounds
    /**
     * \param[in] source_ID Identifier of a radiation source.
     * \param[in] wavelength_min Wavelength for lower bounds of integration
     * \param[in] wavelength_max Wavelength for upper bounds of integration
     * \return Integral of spectral data from wavelength_min to wavelength_max
     */
    float integrateSourceSpectrum(uint source_ID, float wavelength_min, float wavelength_max) const;

    //! Scale an entire spectrum by a constant factor. Creates new global data for scaled spectrum.
    /**
     * \param[in] existing_global_data_label Label of global data containing spectral data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity/reflectivity/transmissivity (.y).
     * \param[in] new_global_data_label Label of new global data to be created containing scaled spectral data (type of vec2).
     * \param[in] scale_factor Scaling factor.
     */
    void scaleSpectrum(const std::string &existing_global_data_label, const std::string &new_global_data_label, float scale_factor) const;

    //! Scale an entire spectrum by a constant factor. Performs scaling in-place.
    /**
     * \param[in] global_data_label Label of global data containing spectral data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity/reflectivity/transmissivity (.y).
     * \param[in] scale_factor Scaling factor.
     */
    void scaleSpectrum(const std::string &global_data_label, float scale_factor) const;

    //! Scale an entire spectrum by a random factor following a uniform distribution.
    /**
     * \param[in] existing_global_data_label Label of global data containing spectral data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity/reflectivity/transmissivity (.y).
     * \param[in] new_global_data_label Label of new global data to be created containing scaled spectral data (type of vec2).
     * \param[in] minimum_scale_factor Scaling factor minimum value in uniform distribution.
     * \param[in] maximum_scale_factor Scaling factor maximum value in uniform distribution.
     */
    void scaleSpectrumRandomly(const std::string &existing_global_data_label, const std::string &new_global_data_label, float minimum_scale_factor, float maximum_scale_factor) const;

    //! Blend one or more spectra together into a new spectrum
    /**
     * \param[in] new_spectrum_label Label for new spectrum global data, which is created by blending the input spectra.
     * \param[in] spectrum_labels Vector of global data labels for spectra to be blended.
     * \param[in] weights Vector of weights for each spectrum to be blended. The weights must sum to 1.0.
     * \note The input spectra can have different sizes, but must have matching wavelength values across all spectra. The output spectra will be the size of the overlapping portion of wavelengths.
     */
    void blendSpectra(const std::string &new_spectrum_label, const std::vector<std::string> &spectrum_labels, const std::vector<float> &weights) const;

    //! Blend one or more spectra together into a new spectrum, with random weights assigned to each input spectrum
    /**
     * \param[in] new_spectrum_label Label for new spectrum global data, which is created by blending the input spectra.
     * \param[in] spectrum_labels Vector of global data labels for spectra to be blended.
     * \note The input spectra can have different sizes, but must have matching wavelength values across all spectra. The output spectra will be the size of the overlapping portion of wavelengths.
     */
    void blendSpectraRandomly(const std::string &new_spectrum_label, const std::vector<std::string> &spectrum_labels) const;

    //! Configure automatic spectral interpolation based on primitive data values
    /**
     * This function sets up automatic interpolation between different spectra based on the value of a primitive data field. When \ref updateRadiativeProperties() is called, for each primitive specified,
     * it will query the value of the primitive data field specified by primitive_data_query_label, perform nearest-neighbor interpolation to find the closest spectrum from the spectra vector,
     * and set the primitive data field specified by primitive_data_radprop_label to the label of the selected spectrum.
     * \param[in] primitive_UUIDs Vector of primitive UUIDs to apply interpolation to
     * \param[in] spectra Vector of global data labels containing spectral data (type std::vector<helios::vec2>). Each label must reference valid global data.
     * \param[in] values Vector of primitive data values mapping to each spectrum. Must be the same length as spectra vector.
     * \param[in] primitive_data_query_label Name of existing primitive data field to query for interpolation (e.g., "age")
     * \param[in] primitive_data_radprop_label Name of primitive data field to set with interpolated spectrum label (e.g., "reflectivity_spectrum" or "transmissivity_spectrum")
     * \note This function must be called before \ref updateRadiativeProperties(). The interpolation uses nearest-neighbor selection based on the absolute distance between the queried value and the provided mapping values.
     */
    void interpolateSpectrumFromPrimitiveData(const std::vector<uint> &primitive_UUIDs, const std::vector<std::string> &spectra, const std::vector<float> &values, const std::string &primitive_data_query_label,
                                              const std::string &primitive_data_radprop_label);

    //! Configure automatic spectral interpolation based on object data values
    /**
     * This function sets up automatic interpolation between different spectra based on the value of an object data field. When \ref updateRadiativeProperties() is called, for each object specified,
     * it will query the value of the object data field specified by object_data_query_label, perform nearest-neighbor interpolation to find the closest spectrum from the spectra vector,
     * and set the primitive data field specified by primitive_data_radprop_label to the label of the selected spectrum for all primitives belonging to that object.
     * \param[in] object_IDs Vector of object IDs to apply interpolation to
     * \param[in] spectra Vector of global data labels containing spectral data (type std::vector<helios::vec2>). Each label must reference valid global data.
     * \param[in] values Vector of object data values mapping to each spectrum. Must be the same length as spectra vector.
     * \param[in] object_data_query_label Name of existing object data field to query for interpolation (e.g., "age")
     * \param[in] primitive_data_radprop_label Name of primitive data field to set with interpolated spectrum label (e.g., "reflectivity_spectrum" or "transmissivity_spectrum")
     * \note This function must be called before \ref updateRadiativeProperties(). The interpolation uses nearest-neighbor selection based on the absolute distance between the queried value and the provided mapping values.
     * \note Although this function reads object data, it sets primitive data because radiative properties are defined per-primitive. All primitives belonging to the object will have their primitive data set.
     */
    void interpolateSpectrumFromObjectData(const std::vector<uint> &object_IDs, const std::vector<std::string> &spectra, const std::vector<float> &values, const std::string &object_data_query_label, const std::string &primitive_data_radprop_label);

    //! Set the number of scattering iterations for a certain band
    /**
     * \param[in] label Label used to reference the band
     * \param[in] depth Number of scattering iterations (depth=0 turns scattering off)
     */
    void setScatteringDepth(const std::string &label, uint depth);

    //! Set the energy threshold used to terminate scattering iterations. Scattering iterations are terminated when the maximum to-be-scattered energy among all primitives is less than "energy"
    /**
     * \param[in] label Label used to reference the band
     * \param[in] energy Energy threshold
     */
    void setMinScatterEnergy(const std::string &label, uint energy);

    //! Use a periodic boundary condition in one or more lateral directions
    /**
     * \param[in] boundary Lateral direction to enforce periodic boundary - choices are "x" (periodic only in x-direction), "y" (periodic only in y-direction), or "xy" (periodic in both x- and y-directions).
     * \note This method should be called prior to calling RadiationModel::updateGeometry(), otherwise the boundary condition will not be enforced.
     */
    void enforcePeriodicBoundary(const std::string &boundary);

    //! Add a radiation camera sensor
    /**
     * \param[in] camera_label A label that will be used to refer to the camera (e.g., "thermal", "multispectral", "NIR", etc.).
     * \param[in] band_label Labels for radiation bands to include in camera.
     * \param[in] position Cartesian (x,y,z) location of the camera sensor.
     * \param[in] lookat Cartesian (x,y,z) position at which the camera is pointed. The vector (lookat-position) is perpendicular to the camera face.
     * \param[in] camera_properties 'CameraProperties' struct containing intrinsic camera parameters.
     * \param[in] antialiasing_samples Number of ray samples per pixel. More samples will decrease noise/aliasing in the image, but will take longer to run.
     */
    void addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::vec3 &lookat, const CameraProperties &camera_properties, uint antialiasing_samples);

    //! Add a radiation camera sensor
    /**
     * \param[in] camera_label A label that will be used to refer to the camera (e.g., "thermal", "multispectral", "NIR", etc.).
     * \param[in] band_label Labels for radiation bands to include in camera.
     * \param[in] position Cartesian (x,y,z) location of the camera sensor.
     * \param[in] viewing_direction Spherical direction in which the camera is pointed (elevation and azimuth angles in radians).
     * \param[in] camera_properties 'CameraProperties' struct containing intrinsic camera parameters.
     * \param[in] antialiasing_samples Number of ray samples per pixel. More samples will decrease noise/aliasing in the image, but will take longer to run.
     */
    void addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::SphericalCoord &viewing_direction, const CameraProperties &camera_properties,
                            uint antialiasing_samples);


    //! Set the spectral response of a camera band based on reference to global data. This function version uses all the global data array to calculate the spectral response.
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \param[in] band_label Label for the radiation band.
     * \param[in] global_data Label for global data containing camera spectral response data. This should be of type vec2 and contain wavelength-quantum efficiency pairs.
     * \note If global data in the standard camera spectral library is referenced, the library will be automatically loaded.
     */
    void setCameraSpectralResponse(const std::string &camera_label, const std::string &band_label, const std::string &global_data);

    //! Set the camera spectral response based on a camera available in the standard camera spectral library (radiation/spectral_data/camera_spectral_library.xml).
    /**
     * Consult the documentation for available cameras in the library, or examine the file radiation/spectral_data/camera_spectral_library.xml.
     * The naming convention is that the response data for the band starts with the camera model (e.g., "iPhone11") followed by an underscore, then the band label.
     * In order for the response to be applied to the camera, the bands must all exist. For example, for iPhone11, there must exist bands "red", "green", and "blue".
     *
     * \param[in] camera_label Label for the camera to be set.
     * \param[in] camera_library_name Name of the camera in the standard camera spectral library (e.g., "iPhone11", "NikonD700", etc.).
     */
    void setCameraSpectralResponseFromLibrary(const std::string &camera_label, const std::string &camera_library_name);

    //! Add a radiation camera sensor loading all properties from the camera library
    /**
     * This method loads camera intrinsic parameters (resolution, field of view, sensor size)
     * and spectral response data from the camera library XML file. The camera is created with
     * the specified position and viewing direction.
     *
     * Available cameras can be found in plugins/radiation/camera_library/camera_library.xml.
     * The library includes cameras such as: Canon_20D, Nikon_D700, Nikon_D50, iPhone11, iPhone12ProMAX.
     *
     * Each camera in the library defines spectral bands (typically "red", "green", and "blue" for RGB cameras).
     * If these bands do not already exist in the radiation model, they will be automatically created
     * with emission disabled and scattering depth set to 3.
     *
     * \param[in] camera_label A label that will be used to refer to the camera instance.
     * \param[in] library_camera_label Label of the camera in the library (e.g., "Canon_20D", "iPhone11").
     * \param[in] position Cartesian (x,y,z) location of the camera sensor.
     * \param[in] lookat Cartesian (x,y,z) position at which the camera is pointed.
     * \param[in] antialiasing_samples Number of ray samples per pixel (minimum 1).
     */
    void addRadiationCameraFromLibrary(const std::string &camera_label, const std::string &library_camera_label, const helios::vec3 &position, const helios::vec3 &lookat, uint antialiasing_samples);

    //! Add a radiation camera sensor loading all properties from the camera library with custom band names
    /**
     * This overload allows specifying custom band labels instead of using the default labels from the
     * camera library XML file. This is useful when you want to use different band names than those
     * defined in the library (e.g., using "R", "G", "B" instead of "red", "green", "blue").
     *
     * The custom band labels are mapped to the spectral responses in the order they appear in the
     * camera library XML file. For example, if the XML defines spectral responses in order
     * "red", "green", "blue" and you provide band_labels = {"R", "G", "B"}, then "R" will use
     * the "red" spectral response, "G" will use "green", and "B" will use "blue".
     *
     * \param[in] camera_label A label that will be used to refer to the camera instance.
     * \param[in] library_camera_label Label of the camera in the library (e.g., "Canon_20D", "iPhone11").
     * \param[in] position Cartesian (x,y,z) location of the camera sensor.
     * \param[in] lookat Cartesian (x,y,z) position at which the camera is pointed.
     * \param[in] antialiasing_samples Number of ray samples per pixel (minimum 1).
     * \param[in] band_labels Custom band labels to use. Must have the same number of elements as there
     *                        are spectral responses defined in the camera library entry. The order must
     *                        correspond to the order of spectral_response elements in the XML.
     */
    void addRadiationCameraFromLibrary(const std::string &camera_label, const std::string &library_camera_label, const helios::vec3 &position, const helios::vec3 &lookat, uint antialiasing_samples, const std::vector<std::string> &band_labels);

    //! Set the position of the radiation camera.
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \param[in] position Cartesian coordinate of camera position.
     */
    void setCameraPosition(const std::string &camera_label, const helios::vec3 &position);

    //! Get the position of the radiation camera.
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \return Cartesian coordinate of camera position.
     */
    helios::vec3 getCameraPosition(const std::string &camera_label) const;

    //! Set the position the radiation camera is pointed toward (used to calculate camera orientation)
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \param[in] lookat Cartesian coordinate of location camera is pointed toward.
     */
    void setCameraLookat(const std::string &camera_label, const helios::vec3 &lookat);

    //! Get the position the radiation camera is pointed toward (used to calculate camera orientation)
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \return Cartesian coordinate of location camera is pointed toward.
     */
    helios::vec3 getCameraLookat(const std::string &camera_label) const;

    //! Set the orientation of the radiation camera based on a Cartesian vector
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \param[in] direction Cartesian vector defining the orientation of the camera.
     */
    void setCameraOrientation(const std::string &camera_label, const helios::vec3 &direction);

    //! Set the orientation of the radiation camera based on a spherical coordinate
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \param[in] direction Spherical coordinate defining the orientation of the camera (elevation and azimuth angles in radians).
     */
    void setCameraOrientation(const std::string &camera_label, const helios::SphericalCoord &direction);

    //! Get the orientation of the radiation camera based on a spherical coordinate
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \return Spherical coordinate defining the orientation of the camera (elevation and azimuth angles in radians).
     */
    helios::SphericalCoord getCameraOrientation(const std::string &camera_label) const;

    //! Get the intrinsic parameters of an existing radiation camera
    /**
     * This method returns the current intrinsic optical and geometric parameters of a camera
     * including resolution, horizontal field of view, lens diameter, focal plane distance,
     * sensor width, model name, and the auto-calculated FOV aspect ratio.
     *
     * \param[in] camera_label Label identifying the camera to query. Camera must exist.
     * \return CameraProperties struct containing the current intrinsic parameters.
     * \note This method does not return camera position, lookat direction, or spectral band configuration.
     * \note The returned FOV_aspect_ratio is the auto-calculated value ensuring square pixels.
     * \note Throws helios_runtime_error if camera_label does not exist.
     */
    CameraProperties getCameraParameters(const std::string &camera_label) const;

    //! Update intrinsic parameters of an existing radiation camera
    /**
     * This method updates the intrinsic optical and geometric parameters of an existing camera.
     * The camera position, lookat direction, and spectral bands are preserved. All fields in
     * CameraProperties can be updated including resolution, HFOV, lens diameter, focal plane
     * distance, sensor width, and model name. If resolution changes, pixel buffers will be
     * reallocated on the next call to runRadiationImaging().
     *
     * \param[in] camera_label Label identifying the camera to update. Camera must already exist.
     * \param[in] camera_properties CameraProperties struct containing the new intrinsic parameters.
     * \note This method preserves the camera's position, lookat direction, and spectral band configuration.
     * \note The FOV_aspect_ratio field in camera_properties is ignored and recalculated from resolution.
     * \note Throws helios_runtime_error if camera_label does not exist.
     */
    void updateCameraParameters(const std::string &camera_label, const CameraProperties &camera_properties);

    //! Get the labels for all radiation cameras that have been added to the radiation model
    /**
     * \return Vector of strings corresponding to each camera label.
     */
    std::vector<std::string> getAllCameraLabels();

    //! Enable automatic JSON metadata file writing for a camera
    /**
     * \param[in] camera_label Label for the camera to enable metadata writing for.
     * \note After calling this method, writeCameraImage() will automatically create a JSON metadata file alongside the image.
     * \note Metadata is automatically populated from camera properties and simulation context. Use getCameraMetadata() and setCameraMetadata() to customize.
     */
    void enableCameraMetadata(const std::string &camera_label);

    //! Enable automatic JSON metadata file writing for multiple cameras
    /**
     * \param[in] camera_labels Vector of camera labels to enable metadata writing for.
     * \note After calling this method, writeCameraImage() will automatically create a JSON metadata file alongside the image for each camera.
     * \note Metadata is automatically populated from camera properties and simulation context. Use getCameraMetadata() and setCameraMetadata() to customize.
     */
    void enableCameraMetadata(const std::vector<std::string> &camera_labels);

    //! Get the current metadata for a camera
    /**
     * \param[in] camera_label Label for the camera to get metadata for.
     * \return CameraMetadata struct containing current metadata for the camera.
     * \note Metadata is automatically populated when cameras are added. This method allows retrieval for inspection or modification.
     */
    CameraMetadata getCameraMetadata(const std::string &camera_label) const;

    //! Set metadata for a camera to be automatically written with images
    /**
     * \param[in] camera_label Label for the camera to set metadata for.
     * \param[in] metadata CameraMetadata struct containing metadata to be written with camera images.
     * \note When writeCameraImage() is called for this camera, a JSON metadata file will be automatically created alongside the image.
     */
    void setCameraMetadata(const std::string &camera_label, const CameraMetadata &metadata);

    //! Enable lens flare rendering for a camera
    /**
     * Enables physically-based lens flare effects including ghost reflections and starburst diffraction patterns.
     * Lens flare is applied as a post-processing step after the main radiation calculations.
     * \param[in] camera_label Label for the camera to enable lens flare for.
     * \note Use setCameraLensFlareProperties() to customize lens flare appearance.
     */
    void enableCameraLensFlare(const std::string &camera_label);

    //! Disable lens flare rendering for a camera
    /**
     * \param[in] camera_label Label for the camera to disable lens flare for.
     */
    void disableCameraLensFlare(const std::string &camera_label);

    //! Check if lens flare rendering is enabled for a camera
    /**
     * \param[in] camera_label Label for the camera to check.
     * \return true if lens flare is enabled, false otherwise.
     */
    [[nodiscard]] bool isCameraLensFlareEnabled(const std::string &camera_label) const;

    //! Set lens flare rendering properties for a camera
    /**
     * \param[in] camera_label Label for the camera to configure.
     * \param[in] properties LensFlareProperties struct containing the desired settings.
     * \note Lens flare must be enabled separately using enableCameraLensFlare().
     */
    void setCameraLensFlareProperties(const std::string &camera_label, const LensFlareProperties &properties);

    //! Get the current lens flare properties for a camera
    /**
     * \param[in] camera_label Label for the camera to get properties for.
     * \return LensFlareProperties struct containing current settings.
     */
    [[nodiscard]] LensFlareProperties getCameraLensFlareProperties(const std::string &camera_label) const;

    //! Adds all geometric primitives from the Context to OptiX
    /**
     * This function should be called anytime Context geometry is created or modified
     * \note \ref RadiationModel::updateGeometry() must be called before simulation can be run
     */
    void updateGeometry();

    //! Adds certain geometric primitives from the Context to OptiX as specified by a list of UUIDs
    /**
     * This function should be called anytime Context geometry is created or modified
     * \param[in] UUIDs Vector of universal unique identifiers of Context primitives to be updated
     * \note \ref RadiationModel::updateGeometry() must be called before simulation can be run
     */
    void updateGeometry(const std::vector<uint> &UUIDs);

    //! Run the simulation for a single radiative band
    /**
     * \param[in] label Label used to reference the band (e.g., "PAR")
     * \note Before running the band simulation, you must 1) add at least one radiative band to the simulation (see \ref RadiationModel::addRadiationBand()), 2) update the Context geometry in the model (see \ref RadiationModel::updateGeometry()),
     * and 3) update radiative properties in the model (see RadiationModel::updateRadiativeProperties()).
     */
    void runBand(const std::string &label);

    //! Run the simulation for a multiple radiative bands
    /**
     * \param[in] labels Label used to reference the band (e.g., "PAR")
     * \note Before running the band simulation, you must 1) add at least one radiative band to the simulation (see \ref RadiationModel::addRadiationBand()), 2) update the Context geometry in the model (see \ref RadiationModel::updateGeometry()),
     * and 3) update radiative properties in the model (see RadiationModel::updateRadiativeProperties()).
     */
    void runBand(const std::vector<std::string> &labels);

    //! Get the total absorbed radiation flux summed over all bands for each primitive
    std::vector<float> getTotalAbsorbedFlux();

    //! Get the radiative energy lost to the sky (surroundings)
    float getSkyEnergy();

    //! Calculate G(theta) (i.e., projected area fraction) for a group of primitives given a certain viewing direction
    /**
     * \param[in] context Pointer to Helios context
     * \param[in] view_direction Viewing direction for projected area
     * \return Projected area fraction G(theta)
     */
    float calculateGtheta(helios::Context *context, helios::vec3 view_direction);

    void setCameraCalibration(CameraCalibration *CameraCalibration);

    //! Update the camera response for a given camera based on color board
    /**
     * \param[in] orginalcameralabel Label of camera to be used for simulation
     * \param[in] sourcelabels_raw Vector of labels of source spectra to be used for simulation
     * \param[in] cameraresponselabels Vector of labels of camera spectral responses
     * \param[in] wavelengthrange Wavelength range of the camera
     * \param[in] truevalues True image values of the color board
     * \param[in] calibratedmark Mark of the calibrated camera
     */
    void updateCameraResponse(const std::string &orginalcameralabel, const std::vector<std::string> &sourcelabels_raw, const std::vector<std::string> &cameraresponselabels, helios::vec2 &wavelengthrange,
                              const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark);

    //! Get the scale factor of the camera response for a given camera
    /**
     * \param[in] orginalcameralabel Label of camera to be used for simulation
     * \param[in] cameraresponselabels Vector of labels of camera spectral responses
     * \param[in] bandlabels Vector of labels of radiation bands to be used for simulation
     * \param[in] sourcelabels Vector of labels of source spectra to be used for simulation
     * \param[in] wavelengthrange Wavelength range of the camera
     * \param[in] truevalues True image values of the color board
     * \return scale factor
     */
    float getCameraResponseScale(const std::string &orginalcameralabel, const std::vector<std::string> &cameraresponselabels, const std::vector<std::string> &bandlabels, const std::vector<std::string> &sourcelabels, helios::vec2 &wavelengthrange,
                                 const std::vector<std::vector<float>> &truevalues);

    //! Run radiation imaging simulation
    /**
     * \param[in] cameralabel Label of camera to be used for simulation
     * \param[in] sourcelabels Vector of labels of source spectra to be used for simulation
     * \param[in] bandlabels Vector of labels of radiation bands to be used for simulation
     * \param[in] cameraresponselabels Vector of labels of camera spectral responses
     * \param[in] wavelengthrange Wavelength range of spectra
     * \param[in] fluxscale Scale factor for source flux
     * \param[in] diffusefactor Diffuse factor for diffuse radiation
     * \param[in] scatteringdepth Number of scattering events to simulate
     */
    void runRadiationImaging(const std::string &cameralabel, const std::vector<std::string> &sourcelabels, const std::vector<std::string> &bandlabels, const std::vector<std::string> &cameraresponselabels, helios::vec2 wavelengthrange,
                             float fluxscale = 1, float diffusefactor = 0.0005, uint scatteringdepth = 4);

    //! Run radiation imaging simulation
    /**
     * \param[in] cameralabels Vector of camera labels to be used for simulation
     * \param[in] sourcelabels Vector of labels of source spectra to be used for simulation
     * \param[in] bandlabels Vector of labels of radiation bands to be used for simulation
     * \param[in] cameraresponselabels Vector of labels of camera spectral responses
     * \param[in] wavelengthrange Wavelength range of spectra
     * \param[in] fluxscale Scale factor for source flux
     * \param[in] diffusefactor Diffuse factor for diffuse radiation
     * \param[in] scatteringdepth Number of scattering events to simulate
     */
    void runRadiationImaging(const std::vector<std::string> &cameralabels, const std::vector<std::string> &sourcelabels, const std::vector<std::string> &bandlabels, const std::vector<std::string> &cameraresponselabels, helios::vec2 wavelengthrange,
                             float fluxscale = 1, float diffusefactor = 0.0005, uint scatteringdepth = 4);

    //! Apply camera image corrections including brightness, contrast, saturation, and gamma compression
    /**
     * This only applies to RGB cameras. This pipeline applies post-processing steps including
     * brightness/contrast adjustment, saturation adjustment, and gamma compression.
     * The parameters are saved to camera metadata if metadata export is enabled.
     *
     * \param[in] cameralabel Label of camera to be used for processing
     * \param[in] red_band_label Label of the red band
     * \param[in] green_band_label Label of the green band
     * \param[in] blue_band_label Label of the blue band
     * \param[in] saturation_adjustment [optional] Adjustment factor for saturation (default is 1.0, which means no adjustment)
     * \param[in] brightness_adjustment [optional] Adjustment factor for brightness (default is 1.0, which means no adjustment)
     * \param[in] contrast_adjustment [optional] Adjustment factor for contrast (default is 1.0, which means no adjustment)
     */
    void applyCameraImageCorrections(const std::string &cameralabel, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation_adjustment = 1.f, float brightness_adjustment = 1.f,
                                     float contrast_adjustment = 1.f);

    //! \deprecated Use applyCameraImageCorrections() instead
    [[deprecated("Use applyCameraImageCorrections() instead")]]
    void applyImageProcessingPipeline(const std::string &cameralabel, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation_adjustment = 1.f, float brightness_adjustment = 1.f,
                                      float contrast_adjustment = 1.f, float gain_adjustment = 1.f);

    //! Apply pre-computed color correction matrix to camera data
    /**
     * \param[in] camera_label Label of the camera to apply color correction to
     * \param[in] red_band_label Label for red channel band data
     * \param[in] green_band_label Label for green channel band data
     * \param[in] blue_band_label Label for blue channel band data
     * \param[in] ccm_file_path Path to XML file containing color correction matrix
     */
    void applyCameraColorCorrectionMatrix(const std::string &camera_label, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, const std::string &ccm_file_path);


    //! Write camera data for one or more bands to a JPEG image
    /**
     * \param[in] camera Label for camera to be queried
     * \param[in] bands Vector of labels for radiative bands to be written
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] frame [optional] A frame count number to be appended to the output image file (e.g., camera_thermal_00001.jpeg). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     * \param[in] flux_to_pixel_conversion [optional] A factor to convert radiative flux to 8-bit pixel values (0-255). By default, this value is 1.0, which means that the pixel values will be equal to the radiative flux. If the radiative flux is
     * very large or very small, it may be necessary to scale the flux to a more appropriate range for the image.
     * \return Name of the output image file that was written
     */
    std::string writeCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1, float flux_to_pixel_conversion = 1.f);

    //! Write normalized camera data (maximum value is 1) for one or more bands to a JPEG image
    /**
     * \param[in] camera Label for camera to be queried
     * \param[in] bands Vector of labels for radiative bands to be written
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path Path to directory where images should be saved
     * \param[in] frame [optional] A frame count number to be appended to the output image file (e.g., camera_thermal_00001.jpeg). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     * \return Name of the output image file that was written
     */
    std::string writeNormCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1);

    //! Write camera data for one band to an ASCII text file
    /**
     * \param[in] camera Label for camera to be queried
     * \param[in] band Label for radiative band to be written
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] frame [optional] A frame count number to be appended to the output image file (e.g., camera_thermal_00001.jpeg). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     */
    void writeCameraImageData(const std::string &camera, const std::string &band, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1);

    //! Write image pixel labels to text file based on primitive data. Primitive data must have type 'float', 'double', 'uint', or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] primitive_data_label Name of the primitive data label
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] frame [optional] A frame count number to be appended to the output file (e.g., camera_thermal_00001.txt). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     * \param[in] padvalue Pad value for the empty pixels
     */
    void writePrimitiveDataLabelMap(const std::string &cameralabel, const std::string &primitive_data_label, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1, float padvalue = NAN);

    //! Write image pixel labels to text file based on object data. Object data must have type 'float', 'double', 'uint', or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] object_data_label Name of the object data label
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] frame [optional] A frame count number to be appended to the output file (e.g., camera_thermal_00001.txt). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     * \param[in] padvalue Pad value for the empty pixels
     */
    void writeObjectDataLabelMap(const std::string &cameralabel, const std::string &object_data_label, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1, float padvalue = NAN);

    //! Write depth image data to text file
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] frame [optional] A frame count number to be appended to the output image file (e.g., camera_depth_00001.txt). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     */
    void writeDepthImageData(const std::string &cameralabel, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1);

    //! Write depth image file, with grayscale normalized to the minimum and maximum depth values
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] max_depth Maximum depth value for normalization (e.g., the depth of the sky)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] frame [optional] A frame count number to be appended to the output image file (e.g., camera_depth_00001.txt). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     */
    void writeNormDepthImage(const std::string &cameralabel, const std::string &imagefile_base, float max_depth, const std::string &image_path = "./", int frame = -1);

    //! Write bounding boxes based on primitive data labels (Ultralytic's YOLO format). Primitive data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] primitive_data_label Name of the primitive data label. Primitive data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] imagefile_base Name for base of output files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] append_label_file [optional] If true, the label file will be appended to the existing file. If false, the label file will be overwritten. By default, it is false.
     * \param[in] frame [optional] A frame count number to be appended to the output file (e.g., camera_thermal_00001.txt). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     */
    [[deprecated]]
    void writeImageBoundingBoxes(const std::string &cameralabel, const std::string &primitive_data_label, uint object_class_ID, const std::string &imagefile_base, const std::string &image_path = "./", bool append_label_file = false, int frame = -1);

    //! Write bounding boxes based on object data labels (Ultralytic's YOLO format). Object data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] object_data_label Name of the object data label. Object data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] imagefile_base Name for base of output files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] append_label_file [optional] If true, the label file will be appended to the existing file. If false, the label file will be overwritten. By default, it is false.
     * \param[in] frame [optional] A frame count number to be appended to the output file (e.g., camera_thermal_00001.txt). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     */
    [[deprecated]]
    void writeImageBoundingBoxes_ObjectData(const std::string &cameralabel, const std::string &object_data_label, uint object_class_ID, const std::string &imagefile_base, const std::string &image_path = "./", bool append_label_file = false,
                                            int frame = -1);

    //! Write bounding boxes based on primitive data labels (Ultralytic's YOLO format). Primitive data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] primitive_data_label Name of the primitive data label. Primitive data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] image_file Name for base of output files (will also include the camera label and a frame number in the file name)
     * \param[in] classes_txt_file [optional] Name of text file to write class names. By default, it is "classes.txt".
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \note The lengths of primitive_data_label and object_class_ID vectors must be the same.
     */
    void writeImageBoundingBoxes(const std::string &cameralabel, const std::string &primitive_data_label, const uint &object_class_ID, const std::string &image_file, const std::string &classes_txt_file = "classes.txt",
                                 const std::string &image_path = "./");

    //! Write bounding boxes based on primitive data labels (Ultralytic's YOLO format). Primitive data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] primitive_data_label Name of the primitive data label. Primitive data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] image_file Name for base of output files (will also include the camera label and a frame number in the file name)
     * \param[in] classes_txt_file [optional] Name of text file to write class names. By default, it is "classes.txt".
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \note The lengths of primitive_data_label and object_class_ID vectors must be the same.
     */
    void writeImageBoundingBoxes(const std::string &cameralabel, const std::vector<std::string> &primitive_data_label, const std::vector<uint> &object_class_ID, const std::string &image_file, const std::string &classes_txt_file = "classes.txt",
                                 const std::string &image_path = "./");

    //! Write bounding boxes based on object data labels (Ultralytic's YOLO format). Object data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] object_data_label Name of the object data label. Object data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] image_file Name for base of output files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] classes_txt_file [optional] Name of text file to write class names. By default, it is "classes.txt".
     * \note The lengths of object_data_label and object_class_ID vectors must be the same.
     */
    void writeImageBoundingBoxes_ObjectData(const std::string &cameralabel, const std::string &object_data_label, const uint &object_class_ID, const std::string &image_file, const std::string &classes_txt_file = "classes.txt",
                                            const std::string &image_path = "./");

    //! Write bounding boxes based on object data labels (Ultralytic's YOLO format). Object data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] object_data_label Name of the object data label. Object data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] image_file Name for base of output files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] classes_txt_file [optional] Name of text file to write class names. By default, it is "classes.txt".
     * \note The lengths of object_data_label and object_class_ID vectors must be the same.
     */
    void writeImageBoundingBoxes_ObjectData(const std::string &cameralabel, const std::vector<std::string> &object_data_label, const std::vector<uint> &object_class_ID, const std::string &image_file,
                                            const std::string &classes_txt_file = "classes.txt", const std::string &image_path = "./");

    //! Write segmentation masks for primitive data in COCO JSON format. Primitive data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] primitive_data_label Name of the primitive data label. Object data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] json_filename Name of the output JSON file. Can include a relative path. If no extension is provided, ".json" will be added.
     * \param[in] image_file Name of the image file corresponding to these labels
     * \param[in] data_attribute_labels [optional] Vector of primitive or object data labels to calculate mean values within each mask and write as attributes. If empty or data doesn't exist, no attributes are added. By default, it is an empty
     * vector.
     * \param[in] append_file [optional] If true, the data will be appended to the existing COCO JSON file. If false, a new file will be created. By default, it is false.
     * \note The lengths of primitive_data_label and object_class_ID vectors must be the same.
     */
    void writeImageSegmentationMasks(const std::string &cameralabel, const std::string &primitive_data_label, const uint &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                     const std::vector<std::string> &data_attribute_labels = {}, bool append_file = false);

    //! Write segmentation masks for primitive data in COCO JSON format. Primitive data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] primitive_data_label Name of the primitive data label. Object data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] json_filename Name of the output JSON file. Can include a relative path. If no extension is provided, ".json" will be added.
     * \param[in] image_file Name of the image file corresponding to these labels
     * \param[in] data_attribute_labels [optional] Vector of primitive or object data labels to calculate mean values within each mask and write as attributes. If empty or data doesn't exist, no attributes are added. By default, it is an empty
     * vector.
     * \param[in] append_file [optional] If true, the data will be appended to the existing COCO JSON file. If false, a new file will be created. By default, it is false.
     * \note The lengths of primitive_data_label and object_class_ID vectors must be the same.
     */
    void writeImageSegmentationMasks(const std::string &cameralabel, const std::vector<std::string> &primitive_data_label, const std::vector<uint> &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                     const std::vector<std::string> &data_attribute_labels = {}, bool append_file = false);

    //! Write segmentation masks for object data in COCO JSON format. Object data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] object_data_label Name of the object data label. Object data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] json_filename Name of the output JSON file. Can include a relative path. If no extension is provided, ".json" will be added.
     * \param[in] image_file Name of the image file corresponding to these labels
     * \param[in] data_attribute_labels [optional] Vector of primitive or object data labels to calculate mean values within each mask and write as attributes. If empty or data doesn't exist, no attributes are added. By default, it is an empty
     * vector.
     * \param[in] append_file [optional] If true, the data will be appended to the existing COCO JSON file. If false, a new file will be created. By default, it is false.
     * \note The lengths of object_data_label and object_class_ID vectors must be the same.
     */
    void writeImageSegmentationMasks_ObjectData(const std::string &cameralabel, const std::string &object_data_label, const uint &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                                const std::vector<std::string> &data_attribute_labels = {}, bool append_file = false);

    //! Write segmentation masks for object data in COCO JSON format. Object data must have type of 'uint' or 'int'.
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] object_data_label Name of the object data label. Object data must have type of 'uint' or 'int'.
     * \param[in] object_class_ID Object class ID to write for the labels in this group.
     * \param[in] json_filename Name of the output JSON file. Can include a relative path. If no extension is provided, ".json" will be added.
     * \param[in] image_file Name of the image file corresponding to these labels
     * \param[in] data_attribute_labels [optional] Vector of primitive or object data labels to calculate mean values within each mask and write as attributes. If empty or data doesn't exist, no attributes are added. By default, it is an empty
     * vector.
     * \param[in] append_file [optional] If true, the data will be appended to the existing COCO JSON file. If false, a new file will be created. By default, it is false.
     * \note The lengths of object_data_label and object_class_ID vectors must be the same.
     */
    void writeImageSegmentationMasks_ObjectData(const std::string &cameralabel, const std::vector<std::string> &object_data_label, const std::vector<uint> &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                                const std::vector<std::string> &data_attribute_labels = {}, bool append_file = false);

private:
    // Helper functions for COCO JSON handling
    std::pair<nlohmann::json, int> initializeCOCOJsonWithImageId(const std::string &filename, bool append_file, const std::string &cameralabel, const helios::int2 &camera_resolution, const std::string &image_file);
    nlohmann::json initializeCOCOJson(const std::string &filename, bool append_file, const std::string &cameralabel, const helios::int2 &camera_resolution, const std::string &image_file);
    void addCategoryToCOCO(nlohmann::json &coco_json, const std::vector<uint> &object_class_ID, const std::vector<std::string> &category_name);
    void writeCOCOJson(const nlohmann::json &coco_json, const std::string &filename);

    // Helper functions for mask generation and boundary tracing
    std::map<int, std::vector<std::vector<bool>>> generateLabelMasks(const std::string &cameralabel, const std::string &data_label, bool use_object_data);
    std::pair<int, int> findStartingBoundaryPixel(const std::vector<std::vector<bool>> &mask, const helios::int2 &camera_resolution);
    std::vector<std::pair<int, int>> traceBoundaryMoore(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &camera_resolution);
    std::vector<std::pair<int, int>> traceBoundarySimple(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &camera_resolution);
    std::vector<std::map<std::string, std::vector<float>>> generateAnnotationsFromMasks(const std::map<int, std::vector<std::vector<bool>>> &label_masks, uint object_class_ID, const helios::int2 &camera_resolution, int image_id);

    // Helper functions for camera metadata export
    std::string detectLightingType() const;
    float calculateCameraTiltAngle(const helios::vec3 &position, const helios::vec3 &lookat) const;
    void computeAgronomicProperties(const std::string &camera_label, CameraMetadata::AgronomicProperties &props) const;
    void populateCameraMetadata(const std::string &camera_label, CameraMetadata &metadata) const;
    std::string writeCameraMetadataFile(const std::string &camera_label, const std::string &output_path = "./") const;

public:
    //! Set padding value for pixels do not have valid values
    /**
     * \param[in] cameralabel Label of target camera
     * \param[in] bandlabels Vector of labels of radiation bands to be used for simulation
     * \param[in] padvalues Vector of padding values for each band
     */
    void setPadValue(const std::string &cameralabel, const std::vector<std::string> &bandlabels, const std::vector<float> &padvalues);

    //! Calibrate camera
    /**
     * \param[in] orginalcameralabel Label of camera to be used for simulation
     * \param[in] sourcelabels Labels of source fluxes
     * \param[in] cameraresponselabels Labels of camera spectral responses
     * \param[in] bandlabels Labels of radiation bands
     * \param[in] scalefactor Scale factor for calibrated camera spectral response
     * \param[in] truevalues True image values of the color board
     * \param[in] calibratedmark Mark of the calibrated camera spectral response
     */
    void calibrateCamera(const std::string &orginalcameralabel, const std::vector<std::string> &sourcelabels, const std::vector<std::string> &cameraresponselabels, const std::vector<std::string> &bandlabels, const float scalefactor,
                         const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark);

    //! Calibrate camera
    /**
     * \param[in] originalcameralabel Label of camera to be used for simulation
     * \param[in] scalefactor Scale factor for calibrated camera spectral response
     * \param[in] truevalues True image values of the color board
     * \param[in] calibratedmark Mark of the calibrated camera spectral response
     */
    void calibrateCamera(const std::string &originalcameralabel, const float scalefactor, const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark);

    //! Color correction algorithm types for auto-calibration
    enum class ColorCorrectionAlgorithm {
        DIAGONAL_ONLY, //!< Simple diagonal scaling (white balance only)
        MATRIX_3X3_AUTO, //!< 3x3 matrix with automatic fallback to diagonal if unstable
        MATRIX_3X3_FORCE //!< Force 3x3 matrix calculation even if potentially unstable
    };

    //! Auto-calibrate camera image using colorboard reference values
    /**
     * \param[in] camera_label Label of the camera that generated the image
     * \param[in] red_band_label Label for red channel band data
     * \param[in] green_band_label Label for green channel band data
     * \param[in] blue_band_label Label for blue channel band data
     * \param[in] output_file_path Path where corrected image will be written
     * \param[in] print_quality_report If true, prints calibration quality metrics to console
     * \param[in] algorithm Color correction algorithm to use (defaults to 3x3 matrix with auto-fallback)
     * \param[in] ccm_export_file_path Optional path to export the computed color correction matrix to XML file
     * \return Path to the written corrected image file
     */
    std::string autoCalibrateCameraImage(const std::string &camera_label, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, const std::string &output_file_path,
                                         bool print_quality_report = false, ColorCorrectionAlgorithm algorithm = ColorCorrectionAlgorithm::MATRIX_3X3_AUTO, const std::string &ccm_export_file_path = "");

    //! Helper function to export color correction matrix to XML file (public for testing)
    void exportColorCorrectionMatrixXML(const std::string &file_path, const std::string &camera_label, const std::vector<std::vector<float>> &matrix, const std::string &source_image_path, const std::string &colorboard_type, float average_delta_e);

    //! Helper function to load color correction matrix from XML file (public for testing)
    std::vector<std::vector<float>> loadColorCorrectionMatrixXML(const std::string &file_path, std::string &camera_label_out);

    //! Get camera pixel data for a specific band
    std::vector<float> getCameraPixelData(const std::string &camera_label, const std::string &band_label);

    //! Set camera pixel data for a specific band
    void setCameraPixelData(const std::string &camera_label, const std::string &band_label, const std::vector<float> &pixel_data);

    //! Query GPU memory available via backend abstraction layer
    /**
     * Phase 1: Integration test method - proves backend is accessible and functional.
     * This method uses the backend instead of direct OptiX calls.
     */
    void queryBackendGPUMemory() const;

    //! Test helper: Build geometry data and return primitive count (Phase 1 testing)
    /**
     * Phase 1: Testing helper to verify buildGeometryData() works correctly.
     * Returns the number of primitives extracted from Context.
     */
    size_t testBuildGeometryData();

    //! Test helper: Get backend pointer for direct testing (Phase 1 only)
    helios::RayTracingBackend *getBackend();

    //! Test helper: Get geometry data reference (Phase 1 only)
    helios::RayTracingGeometry &getGeometryData();

    //! Test helper: Get material data reference (Phase 1 only)
    helios::RayTracingMaterial &getMaterialData();

    //! Test helper: Get source data reference (Phase 1 only)
    std::vector<helios::RayTracingSource> &getSourceData();

    //! Test helper: Build all backend data (Phase 1 testing only)
    void testBuildAllBackendData();

protected:
    //! Flag to determine if status messages are output to the screen
    bool message_flag;

    //! Specular reflection mode: 0=disabled, 1=default scale (0.25), 2=user scale
    uint specular_reflection_mode = 0;

    //! Pointer to the context
    helios::Context *context;

    CameraCalibration *cameracalibration;
    bool calibration_flag = false;

    //! Helper function to get current date and time as string
    std::string getCurrentDateTime();

    //! Pointers to current primitive geometry
    std::vector<uint> primitiveID;

    //! UUIDs currently added from the Context
    std::vector<uint> context_UUIDs;

    //! UUID-to-array-position mapping (UUID → array index)
    //! Enables O(1) lookup of array position from UUID value
    std::unordered_map<uint, size_t> uuid_to_position;

    //! Array-position-to-UUID mapping (array index → UUID)
    //! For reverse lookups: position_to_uuid[array_position] = UUID
    std::vector<uint> position_to_uuid;

    // --- Radiation Band Variables --- //

    std::map<std::string, RadiationBand> radiation_bands;

    //! Global diffuse spectrum applied to all bands (set via setDiffuseSpectrum)
    std::vector<helios::vec2> global_diffuse_spectrum;

    //! Label for global diffuse spectrum in Context global data
    std::string global_diffuse_spectrum_label = "none";

    //! Cached version number of global diffuse spectrum for change detection
    uint64_t global_diffuse_spectrum_version = 0;

    std::map<std::string, bool> scattering_iterations_needed;

    // --- radiation source variables --- //

    std::vector<RadiationSource> radiation_sources;

    // --- Camera Variables --- //

    //! Radiation cameras
    std::map<std::string, RadiationCamera> cameras;

    //! Camera metadata for JSON export
    std::map<std::string, CameraMetadata> camera_metadata;

    //! Set of cameras with metadata JSON writing enabled
    std::set<std::string> metadata_enabled_cameras;

    //! Primitive spectral reflectivity data references
    std::map<std::string, std::vector<uint>> spectral_reflectivity_data;

    //! Primitive spectral transmissivity data references
    std::map<std::string, std::vector<uint>> spectral_transmissivity_data;

    //! Storage for spectral interpolation configurations
    struct SpectrumInterpolationConfig {
        std::unordered_set<uint> primitive_UUIDs; // Primitive UUIDs to apply this config to
        std::unordered_set<uint> object_IDs; // Object IDs to apply this config to
        std::vector<std::string> spectra_labels; // Global data labels for spectra
        std::vector<float> mapping_values; // Values corresponding to each spectrum
        std::string query_data_label; // Primitive/object data to query (e.g., "age")
        std::string target_data_label; // Primitive data to set (e.g., "reflectivity_spectrum")
    };

    std::vector<SpectrumInterpolationConfig> spectrum_interpolation_configs;

    std::vector<helios::vec2> generateGaussianCameraResponse(float FWHM, float mu, float centrawavelength, const helios::int2 &wavebandrange);

    // --- Constants and Defaults --- //

    //! Steffan Boltzmann Constant
    float sigma = 5.6703744E-8;

    //! Default primitive reflectivity
    float rho_default;

    //! Default primitive transmissivity
    float tau_default;

    //! Default primitive emissivity
    float eps_default;

    //! Default primitive attenuation coefficient
    float kappa_default;

    //! Default primitive scattering coefficient
    float sigmas_default;

    //! Default primitive temperature
    float temperature_default;

    //! Default number of rays to be used in direct radiation model.
    size_t directRayCount_default;

    //! Default number of rays to be used in diffuse radiation model.
    size_t diffuseRayCount_default;

    //! Default diffuse radiation flux
    float diffuseFlux_default;

    //! Default minimum energy for scattering
    float minScatterEnergy_default;

    //! Default scattering depth
    uint scatteringDepth_default;

    // --- Functions --- //

    //! Creates OptiX context and creates all associated variables, buffers, geometry, acceleration structures, etc. needed for radiation ray tracing.
    void initializeOptiX();

    //! Sets radiative properties for all primitives
    /** DEPRECATED: This function is no longer required - material properties are automatically updated when runBand() is called.
        Handles spectrum-based material loading and camera spectral response weighting.
        Called internally during runBand() when spectrum interpolation is configured.
        \note This is a private method called automatically - users should not call this directly.
    */
    void updateRadiativeProperties();

    //! Update atmospheric sky radiance model for camera radiation calculations
    /** This function computes spectral sky radiance based on atmospheric conditions from SolarPosition plugin.
     *  Integrates atmospheric sky spectrum weighted by camera spectral response for each band.
     *  Uses analytical atmospheric physics model (independent implementation based on Rayleigh/Mie scattering).
     *
     *  Reads atmospheric parameters from Context global data:
     *  - atmosphere_pressure_Pa: Atmospheric pressure in Pascals
     *  - atmosphere_temperature_K: Air temperature in Kelvin
     *  - atmosphere_humidity_rel: Relative humidity (0-1)
     *  - atmosphere_turbidity: Ångström's aerosol turbidity coefficient (AOD at 500nm)
     *
     * \param[in] band_labels Labels for all bands to be launched
     * \param[in] camera Reference to the camera being traced
     * \note Turbidity uses the same convention as SolarPosition plugin (AOD at 500nm, NOT Linke turbidity)
     * \note If camera spectral response is "uniform", wavelength bounds from the band must be set
     * \return Vector of base sky radiance values (W/m²/sr) for each band, to be used for camera rendering
     */
    std::vector<float> updateAtmosphericSkyModel(const std::vector<std::string> &band_labels, const RadiationCamera &camera);

    //! Update Prague sky model angular parameters for general diffuse radiation
    /**
     * \param[in] band_labels Labels for all bands to be launched
     * \note Reads Prague spectral parameters from Context global data (set by SolarPosition plugin)
     * \note Skips bands where power-law extinction is already set (priority 1)
     * \note If Prague data unavailable, parameters remain at zero (isotropic distribution used)
     */
    void updatePragueParametersForGeneralDiffuse(const std::vector<std::string> &band_labels);

    //! Load Context global data corresponding to spectral data
    /**
     * \param[in] global_data_label Label for global data containing spectral data
     * \return Vector of vec2 data containing spectral data (.x is wavelength in nanometers, .y is the spectral value)
     */
    std::vector<helios::vec2> loadSpectralData(const std::string &global_data_label) const;

    /// void updateFluxesFromSpectra( uint SourceID );


    void buildLightModelGeometry(uint sourceID);

    void buildCameraModelGeometry(const std::string &cameralabel);

    void updateLightModelPosition(uint sourceID, const helios::vec3 &delta_position);

    void updateCameraModelPosition(const std::string &cameralabel);

    //! Build camera launch parameters from camera settings
    /**
     * @brief Build camera launch parameters from camera settings
     * @param camera Camera configuration
     * @param camera_id Camera index
     * @param antialiasing_samples Antialiasing sample count
     * @param tile_resolution Tile resolution (or full resolution if no tiling)
     * @param tile_offset Tile offset (0,0 if no tiling)
     * @return Launch parameters struct ready for backend
     */
    helios::RayTracingLaunchParams buildCameraLaunchParams(const RadiationCamera &camera, uint camera_id, uint antialiasing_samples, const helios::int2 &tile_resolution, const helios::int2 &tile_offset);

    //! Compute camera tiles for large renders
    /**
     * @brief Compute camera tiles for large renders
     * @param camera Camera to tile
     * @param maxRays Maximum rays per launch
     * @return Vector of tiles (single tile if no tiling needed)
     */
    std::vector<CameraTile> computeCameraTiles(const RadiationCamera &camera, size_t maxRays);

    //! Phase 1: Build backend-agnostic geometry data from Context primitives
    /**
     * Extracts geometry from all Context primitives and populates geometry_data structure.
     * This data can then be uploaded to the backend via backend->updateGeometry().
     */
    void buildGeometryData();

    //! Extract texture mask and UV data for all primitives
    /**
     * Iterates through primitives with transparency textures, extracts mask data
     * and UV coordinates, and populates the texture-related fields in geometry_data.
     * Called internally by buildGeometryData().
     */
    void buildTextureData();

    //! Build UUID-to-array-position mapping from geometry_data
    //! Must be called after buildGeometryData() and before buildMaterialData()
    void buildUUIDMapping();

    //! Phase 1: Build backend-agnostic material data from Context primitive data
    void buildMaterialData();

    //! Phase 1: Build backend-agnostic source data from radiation_sources
    void buildSourceData();

    //! UUIDs for source 3D object models (for visualization). Key is the source ID, value is a vector of UUIDs for the source model.
    std::map<uint, std::vector<uint>> source_model_UUIDs;
    //! UUIDs for camera 3D object models (for visualization). Key is the camera label, value is a vector of UUIDs for the camera model.
    std::map<std::string, std::vector<uint>> camera_model_UUIDs;

    /* Phase 1: Backend abstraction layer (for incremental OptiX code replacement) */

    //! Ray tracing backend (will replace direct OptiX usage)
    std::unique_ptr<helios::RayTracingBackend> backend;

    //! Backend-agnostic geometry data (built from Context, uploaded to backend)
    helios::RayTracingGeometry geometry_data;

    //! Backend-agnostic material data (built from Context, uploaded to backend)
    helios::RayTracingMaterial material_data;

    //! Backend-agnostic source data (built from radiation_sources, uploaded to backend)
    std::vector<helios::RayTracingSource> source_data;


    //! Flag indicating whether geometry has been built
    bool isgeometryinitialized;

    //! Periodic boundary condition flags (x, y)
    helios::vec2 periodic_flag;

    bool radiativepropertiesneedupdate = true;

    std::vector<bool> isbandpropertyinitialized;

    bool islightvisualizationenabled = false;
    bool iscameravisualizationenabled = false;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;

    std::vector<std::string> spectral_library_files;

    // Helper methods for Prague Sky Model spectral integration from Context
    float integrateOverResponse(const std::vector<float> &wavelengths, const std::vector<float> &values, const std::vector<helios::vec2> &camera_response) const;

    float weightedAverageOverResponse(const std::vector<float> &wavelengths, const std::vector<float> &param_values, const std::vector<float> &weight_values, const std::vector<helios::vec2> &camera_response) const;

    float computeAngularNormalization(float circ_str, float circ_width, float horiz_bright) const;
};

#endif
