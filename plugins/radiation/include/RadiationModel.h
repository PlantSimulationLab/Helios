/** \file "RadiationModel.h" Primary header file for radiation transport model.

    Copyright (C) 2016-2025 Brian Bailey

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

// NVIDIA OptiX Includes
#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_vector_functions.h>
#include <optixu/optixu_vector_types.h>

#include <utility>

//! Properties defining a radiation camera
struct CameraProperties {

    bool operator!=(const CameraProperties &rhs) const {
        return !(rhs == *this);
    }

    //! Camera sensor resolution (number of pixels) in the horizontal (.x) and vertical (.y) directions
    helios::int2 camera_resolution;

    //! Distance from the viewing plane to the focal plane
    float focal_plane_distance;

    //! Diameter of the camera lens (lens_diameter = 0 gives a 'pinhole' camera with everything in focus)
    float lens_diameter;

    //! Camera horizontal field of view in degrees
    float HFOV;

    //! Physical dimensions of the pixel array sensor in the horizontal (.x) and vertical (.y) directions
    float FOV_aspect_ratio;

    // The color correction matrix is a 4×3 linear transform that maps a camera’s raw sensor RGB values into a standardized color space by compensating for its spectral sensitivities and channel cross-talk.
    std::array<float, 12> color_correction_matrix;

    CameraProperties() {
        camera_resolution = helios::make_int2(512, 512);
        focal_plane_distance = 1;
        lens_diameter = 0.05;
        FOV_aspect_ratio = 1.f;
        HFOV = 20.f;
        FOV_aspect_ratio = 1.f;
        color_correction_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    }

    bool operator==(const CameraProperties &rhs) const {
        return camera_resolution == rhs.camera_resolution && focal_plane_distance == rhs.focal_plane_distance && lens_diameter == rhs.lens_diameter && FOV_aspect_ratio == rhs.FOV_aspect_ratio && HFOV == rhs.HFOV &&
               color_correction_matrix == rhs.color_correction_matrix;
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
        focal_length = camera_properties.focal_plane_distance;
        resolution = camera_properties.camera_resolution;
        lens_diameter = camera_properties.lens_diameter;
        HFOV_degrees = camera_properties.HFOV;
        FOV_aspect_ratio = camera_properties.FOV_aspect_ratio;
        color_correction_matrix = camera_properties.color_correction_matrix;
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
    // camera focal length.
    float focal_length;
    // camera horizontal field of view (degrees)
    float HFOV_degrees;
    // Ratio of camera horizontal field of view to vertical field of view
    float FOV_aspect_ratio;
    // The color correction matrix is a 4×3 linear transform that maps a camera’s raw sensor RGB values into a standardized color space by compensating for its spectral sensitivities and channel cross-talk.
    std::array<float, 12> color_correction_matrix;
    // Number of antialiasing samples per pixel
    uint antialiasing_samples;

    std::vector<std::string> band_labels;

    std::map<std::string, std::string> band_spectral_response;

    std::map<std::string, std::vector<float>> pixel_data;

    std::vector<uint> pixel_label_UUID;
    std::vector<float> pixel_depth;

    //! Normalize all pixel data in the camera such that the maximum pixel value is 1.0 and the minimum is 0.0 (no clamping applied)
    void normalizePixels();

    //! Apply auto-exposure scaling to image data to scale the average luminance to a target value.
    /**
     * Computes the image’s mean luminance and applies a single uniform gain so that the scene-average luminance becomes the specified grey_target value.
     *
     * \param[in] target [optional] Target average luminance value. Default is 18%.
     */
    void scaleToGreyTarget(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float target = 0.18f);

    //! Apply auto-white balancing to image data based on Minkowski mean
    /**
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     * \param[in] p [optional] Minkowski mean parameter. Default is 5.0.
     */
    void whiteBalance(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float p = 5.0);

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
    void HDRToning(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label);

    //! Apply the color correction matrix to image data
    /**
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     */
    void applyCCM(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label);

    //! Apply gamma compression to image data
    /**
     * Applies the standard sRGB electro-optical transfer function to each channel of a linear-light image—clamping negatives to zero and limiting outputs to [0,1]—thereby encoding the data into display-ready sRGB space.  This final step ensures that
     * pixel values map correctly to human‐perceived brightness on typical monitors.
     * \param[in] red_band_label Label for red channel band
     * \param[in] green_band_label Label for green channel band
     * \param[in] blue_band_label Label for blue channel band
     */
    void gammaCompress(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label);

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

    //! Converts a value to the sRGB color space.
    /**
     * \param[in] x Input value to be converted, expected in the range [0.0, 1.0].
     * \return Corresponding value in the sRGB color space.
     */
    static float toSRGB(float x) noexcept {
        x = std::fminf(x, 1.0f);
        return (x <= 0.0031308f) ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.0f) - 0.055f;
    }

    //! Applies a bilateral filter to the input data.
    /**
     * \param[in] in The input vector containing image data.
     * \param[out] out The output vector where the filtered image data will be stored.
     * \param[in] w Width of the image.
     * \param[in] h Height of the image.
     * \param[in] ss Spatial standard deviation for the filter.
     * \param[in] sr Intensity standard deviation for the filter.
     */
    void bilateralFilter(const std::vector<float>& in, std::vector<float>& out, int w, int h, float ss, float sr);

    //! Applies a fast bilateral filter to the input data.
    /**
     * \param[in] in Input vector containing the source data to be filtered.
     * \param[out] out Output vector to store the filtered data.
     * \param[in] w Width of the input data.
     * \param[in] h Height of the input data.
     * \param[in] ss Spatial sigma, controlling the spatial smoothing.
     * \param[in] sr Range sigma, controlling the intensity similarity smoothing.
     * \param[in] df Downsampling factor for optimization.
     */
    void fastBilateralFilter(const std::vector<float> &in, std::vector<float> &out, int w, int h, float ss, float sr, int df);

    void gaussianBlur(const std::vector<float>& in, std::vector<float>& out, int w, int h, float sigma);

    //! Resizes a source image to the specified dimensions.
    /**
     * \param[in] src The source image data as a 1D array of float values.
     * \param[out] dst The destination image data as a 1D array of float values.
     * \param[in] src_w The width of the source image.
     * \param[in] src_h The height of the source image.
     * \param[in] dst_w The width of the destination image.
     * \param[in] dst_h The height of the destination image.
     */
    void resizeImage(const std::vector<float> &src, std::vector<float> &dst, int src_w, int src_h, int dst_w, int dst_h);

    //! Performs a smoothstep interpolation between two edge values
    /**
     * \param[in] edge0 The lower edge of the interpolation range.
     * \param[in] edge1 The upper edge of the interpolation range.
     * \param[in] x The input value to interpolate, expected to be between edge0 and edge1.
     * \return The interpolated value, ranging from 0 to 1, based on the input value x.
     */
    float smoothstep(float edge0, float edge1, float x);
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
    /** \return 0 if test was successful, 1 if test failed */
    int selfTest();

    //! Disable/silence status messages
    /** \note Error messages are still displayed. */
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
    /** Diffuse component of radiation incident on a horizontal surface above all geometry in the domain.
     * \param[in] label Label used to reference the band
     * \param[in] flux Radiative flux
     */
    void setDiffuseRadiationFlux(const std::string &label, float flux);

    //! Extinction coefficient of diffuse ambient radiation
    /** The angular distribution of diffuse ambient radiation is computed according to N = Psi^-K, where Psi is the angle between the distribution peak (usually the sun direction) and the ambient direction, and K is the extinction coefficient. When
     * K=0 the ambient distribution is uniform, which is the default setting
     * \param[in] label Label used to reference the radiative band
     * \param[in] K Extinction coefficient value
     * \param[in] peak_dir Unit vector pointing in the direction of the peak in diffuse radiation (this is usually the sun direction)
     */
    void setDiffuseRadiationExtinctionCoeff(const std::string &label, float K, const helios::vec3 &peak_dir);

    //! Extinction coefficient of diffuse ambient radiation
    /** The angular distribution of diffuse ambient radiation is computed according to N = Psi^-K, where Psi is the angle between the distribution peak (usually the sun direction) and the ambient direction, and K is the extinction coefficient. When
     * K=0 the ambient distribution is uniform, which is the default setting
     * \param[in] label Label used to reference the radiative band
     * \param[in] K Extinction coefficient value
     * \param[in] peak_dir Spherical direction of the peak in diffuse radiation (this is usually the sun direction)
     */
    void setDiffuseRadiationExtinctionCoeff(const std::string &label, float K, const helios::SphericalCoord &peak_dir);

    //! Set the integral of the diffuse spectral flux distribution across all possible wavelengths FOR ALL EXISTING BANDS (=∫Sdλ)
    /**
     * \param[in] spectrum_integral Integration of source spectral flux distribution across all possible wavelengths (=∫Sdλ)
     * \note This function will call setDiffuseFlux() for all bands to update source fluxes based on the new spectrum integral
     */
    void setDiffuseSpectrumIntegral(float spectrum_integral);

    //! Scale the source spectral flux distribution based on a prescribed integral between two wavelengths FOR ALL EXISTING BANDS (=∫Sdλ)
    /**
     * \param[in] spectrum_integral Integration of source spectral flux distribution between two wavelengths (=∫Sdλ)
     * \param[in] wavelength_min Lower bounding wavelength for wave band
     * \param[in] wavelength_max Upper bounding wavelength for wave band
     * \note This function will call setDiffuseFlux() for all bands to update source fluxes based on the new spectrum integral
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
     * \param[in] direction Spherical coordinate pointing toward the radiation source
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
     * \param[in] rotation Rotation of the source in radians about the x- y- and z- axes (the sign of the rotation angle follows right-hand rule)
     * \return Source identifier
     */
    uint addRectangleRadiationSource(const helios::vec3 &position, const helios::vec2 &size, const helios::vec3 &rotation);

    //! Add planar circular radiation source
    /**
     * \param[in] position  (x,y,z) position of the center of the disk radiation source
     * \param[in] radius Radius of disk source
     * \param[in] rotation Rotation of the source in radians about the x- y- and z- axes (the sign of the rotation angle follows right-hand rule)
     * \return Source identifier
     */
    uint addDiskRadiationSource(const helios::vec3 &position, float radius, const helios::vec3 &rotation);

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
     * \param[in] position If point source - (radius,elevation,azimuth) position of the radiation source. If collimated source - (elevation,azimuth) vector pointing toward the source (radius is ignored).
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

    //! Set the spectral distribution of diffuse ambient radiation FOR A SINGLE BAND based on global data of wavelength-intensity pairs.
    /**
     * \param[in] band_label Label used to reference the band
     * \param[in] spectrum_label Label of global data containing spectral intensity data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity (.y).
     */
    void setDiffuseSpectrum(const std::string &band_label, const std::string &spectrum_label);

    //! Set the spectral distribution of diffuse ambient radiation FOR MULTIPLE BANDS based on global data of wavelength-intensity pairs.
    /**
     * \param[in] band_labels List of labels used to reference the bands
     * \param[in] spectrum_label Label of global data containing spectral intensity data (type of vec2). Each index of the global data gives the wavelength (.x) and spectral intensity (.y).
     */
    void setDiffuseSpectrum(const std::vector<std::string> &band_labels, const std::string &spectrum_label);

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
     * \param[in] viewing_direction Spherical direction in which the camera is pointed.
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
     * \param[in] direction Spherical coordinate defining the orientation of the camera.
     */
    void setCameraOrientation(const std::string &camera_label, const helios::SphericalCoord &direction);

    //! Get the orientation of the radiation camera based on a spherical coordinate
    /**
     * \param[in] camera_label Label for the camera to be set.
     * \return Spherical coordinate defining the orientation of the camera.
     */
    helios::SphericalCoord getCameraOrientation(const std::string &camera_label) const;

    //! Get the labels for all radiation cameras that have been added to the radiation model
    /**
     * \return Vector of strings corresponding to each camera label.
     */
    std::vector<std::string> getAllCameraLabels();

    //! Adds all geometric primitives from the Context to OptiX
    /**
     * This function should be called anytime Context geometry is created or modified
     * \note \ref helios::Context::updateGeometry() must be called before simulation can be run
     */
    void updateGeometry();

    //! Adds certain geometric primitives from the Context to OptiX as specified by a list of UUIDs
    /**
     * This function should be called anytime Context geometry is created or modified
     * \param[in] UUIDs Vector of universal unique identifiers of Context primitives to be updated
     * \note \ref helios::Context::updateGeometry() must be called before simulation can be run
     */
    void updateGeometry(const std::vector<uint> &UUIDs);

    //! Run the simulation for a single radiative band
    /**
     * \param[in] label Label used to reference the band (e.g., "PAR")
     * \note Before running the band simulation, you must 1) add at least one radiative band to the simulation (see \ref RadiationModel::addRadiationBand()), 2) update the Context geometry in the model (see \ref helios::Context::updateGeometry()),
     * and 3) update radiative properties in the model (see \ref RadiationModel::updateRadiativeProperties()).
     */
    void runBand(const std::string &label);

    //! Run the simulation for a multiple radiative bands
    /**
     * \param[in] labels Label used to reference the band (e.g., "PAR")
     * \note Before running the band simulation, you must 1) add at least one radiative band to the simulation (see \ref RadiationModel::addRadiationBand()), 2) update the Context geometry in the model (see \ref helios::Context::updateGeometry()),
     * and 3) update radiative properties in the model (see \ref RadiationModel::updateRadiativeProperties()).
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

    //! Apply a digital camera-like processing pipeline to the camera image
    /**
     * This only applies to RGB cameras.
     *
     * \param[in] cameralabel Label of camera to be used for processing
     * \param[in] red_band_label Label of the red band
     * \param[in] green_band_label Label of the green band
     * \param[in] blue_band_label Label of the blue band
     * \param[in] apply_HDR_toning Whether to apply HDR toning to the image
     */
    void applyImageProcessingPipeline(const std::string &cameralabel, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, bool apply_HDR_toning = true);

    //! Write camera data for one or more bands to a JPEG image
    /**
     * \param[in] camera Label for camera to be queried
     * \param[in] bands Vector of labels for radiative bands to be written
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path [optional] Path to directory where images should be saved. By default, it will be placed in the current working directory.
     * \param[in] frame [optional] A frame count number to be appended to the output image file (e.g., camera_thermal_00001.jpeg). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     * \param[in] flux_to_pixel_conversion [optional] A factor to convert radiative flux to 8-bit pixel values (0-255). By default, this value is 1.0, which means that the pixel values will be equal to the radiative flux. If the radiative flux is
     * very large or very small, it may be necessary to scale the flux to a more appropriate range for the image.
     */
    void writeCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1, float flux_to_pixel_conversion = 1.f);

    //! Write normalized camera data (maximum value is 1) for one or more bands to a JPEG image
    /**
     * \param[in] camera Label for camera to be queried
     * \param[in] bands Vector of labels for radiative bands to be written
     * \param[in] imagefile_base Name for base of output image JPEG files (will also include the camera label and a frame number in the file name)
     * \param[in] image_path Path to directory where images should be saved
     * \param[in] frame [optional] A frame count number to be appended to the output image file (e.g., camera_thermal_00001.jpeg). By default, the frame count will be omitted from the file name. This value must be less than or equal to 99,999.
     */
    void writeNormCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path = "./", int frame = -1);

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
    void writeImageBoundingBoxes_ObjectData(const std::string &cameralabel, const std::string &object_data_label, uint object_class_ID, const std::string &imagefile_base, const std::string &image_path = "./", bool append_label_file = false,
                                            int frame = -1);

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


protected:
    //! Flag to determine if status messages are output to the screen
    bool message_flag;

    //! Pointer to the context
    helios::Context *context;

    CameraCalibration *cameracalibration;
    bool calibration_flag = false;

    //! Pointers to current primitive geometry
    std::vector<uint> primitiveID;

    //! UUIDs currently added from the Context
    std::vector<uint> context_UUIDs;

    // --- Radiation Band Variables --- //

    std::map<std::string, RadiationBand> radiation_bands;

    std::map<std::string, bool> scattering_iterations_needed;

    // --- radiation source variables --- //

    std::vector<RadiationSource> radiation_sources;

    //! Number of external radiation sources
    RTvariable Nsources_RTvariable;

    //! (x,y,z) positions of external radiation sources - RTbuffer object
    RTbuffer source_positions_RTbuffer;
    //! (x,y,z) positions of external radiation sources - RTvariable
    RTvariable source_positions_RTvariable;

    //! Types of radiation sources - RTbuffer object
    RTbuffer source_types_RTbuffer;
    //! Types radiation sources - RTvariable
    RTvariable source_types_RTvariable;

    //! Fluxes of external radiation sources - RTbuffer object
    RTbuffer source_fluxes_RTbuffer;
    //! Fluxes of external radiation sources - RTvariable
    RTvariable source_fluxes_RTvariable;

    //! Widths of external radiation sources - RTbuffer object
    RTbuffer source_widths_RTbuffer;
    //! Widths of external radiation sources - RTvariable
    RTvariable source_widths_RTvariable;


    //! Rotations (rx,ry,rz) of external radiation sources - RTbuffer object
    RTbuffer source_rotations_RTbuffer;
    //! Rotations (rx,ry,rz) of external radiation sources - RTvariable
    RTvariable source_rotations_RTvariable;

    // --- Camera Variables --- //

    //! Radiation cameras
    std::map<std::string, RadiationCamera> cameras;

    //! Positions of radiation camera center points - RTvariable
    RTvariable camera_position_RTvariable;

    //! Radiation camera viewing directions - RTvariable
    RTvariable camera_direction_RTvariable;

    //! Radiation camera lens size - RTvariable
    RTvariable camera_lens_diameter_RTvariable;

    //! Radiation FOV aspect ratio - RTvariable
    RTvariable FOV_aspect_RTvariable;

    //! Radiation camera focal length - RTvariable
    RTvariable camera_focal_length_RTvariable;

    //! Radiation camera distance between lens and sensor plane - RTvariable
    RTvariable camera_viewplane_length_RTvariable;

    //! Number of radiation cameras
    RTvariable Ncameras_RTvariable;

    //! Current radiation camera index
    RTvariable camera_ID_RTvariable;

    //! Primitive spectral reflectivity data references
    std::map<std::string, std::vector<uint>> spectral_reflectivity_data;

    //! Primitive spectral transmissivity data references
    std::map<std::string, std::vector<uint>> spectral_transmissivity_data;

    std::vector<helios::vec2> generateGaussianCameraResponse(float FWHM, float mu, float centrawavelength, const helios::int2 &wavebanrange);

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
    /** This function should be called anytime primitive radiative properties are modified. If radiative properties were not set in the Context, default radiative properties will be applied (black body).
        \note \ref RadiationModel::updateRadiativeProperties() must be called before simulation can be run
    */
    void updateRadiativeProperties();

    //! Load Context global data corresponding to spectral data
    /**
     * \param[in] global_data_label Label for global data containing spectral data
     * \return Vector of vec2 data containing spectral data (.x is wavelength in nanometers, .y is the spectral value)
     */
    std::vector<helios::vec2> loadSpectralData(const std::string &global_data_label) const;

    /// void updateFluxesFromSpectra( uint SourceID );

    //! Get 1D array of data for an OptiX buffer of floats
    /**
        \param[in] buffer OptiX buffer object corresponding to 1D array of data
    */
    std::vector<float> getOptiXbufferData(RTbuffer buffer);

    //! Get 1D array of data for an OptiX buffer of doubles
    /**
        \param[in] buffer OptiX buffer object corresponding to 1D array of data
    */
    std::vector<double> getOptiXbufferData_d(RTbuffer buffer);

    //! Get 1D array of data for an OptiX buffer of unsigned ints
    /**
        \param[in] buffer OptiX buffer object corresponding to 1D array of data
    */
    std::vector<uint> getOptiXbufferData_ui(RTbuffer buffer);

    void addBuffer(const char *name, RTbuffer &buffer, RTvariable &variable, RTbuffertype type, RTformat format, size_t dimension);

    //! Set size of 1D buffer and initialize all elements to zero.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] bsize length of buffer.
     */
    void zeroBuffer1D(RTbuffer &buffer, size_t bsize);

    //! Copy contents of one buffer to another
    /**
     * \param[in] buffer OptiX buffer to copy FROM.
     * \param[out] buffer_copy OptiX buffer to copy TO.
     */
    void copyBuffer1D(RTbuffer &buffer, RTbuffer &buffer_copy);

    //! Set size of 1D buffer and initialize all elements based on a 1D array of doubles.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dd(RTbuffer &buffer, const std::vector<double> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Df(RTbuffer &buffer, const std::vector<float> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type float2.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dfloat2(RTbuffer &buffer, const std::vector<optix::float2> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type float3.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dfloat3(RTbuffer &buffer, const std::vector<optix::float3> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type float4.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dfloat4(RTbuffer &buffer, const std::vector<optix::float4> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type int.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Di(RTbuffer &buffer, const std::vector<int> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type unsigned int.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dui(RTbuffer &buffer, const std::vector<uint> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type int2.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dint2(RTbuffer &buffer, const std::vector<optix::int2> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type int3.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dint3(RTbuffer &buffer, const std::vector<optix::int3> &array);
    //! Set size of 1D buffer and initialize all elements based on a 1D array of type char.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 1D array used to initialize buffer.
     */
    void initializeBuffer1Dchar(RTbuffer &buffer, const std::vector<char> &array);
    //! Set size of 2D buffer and initialize all elements to zero.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] size length of buffer.
     */
    void zeroBuffer2D(RTbuffer &buffer, optix::int2 bsize);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of doubles.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dd(RTbuffer &buffer, const std::vector<std::vector<double>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Df(RTbuffer &buffer, const std::vector<std::vector<float>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dfloat2(RTbuffer &buffer, const std::vector<std::vector<optix::float2>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dfloat3(RTbuffer &buffer, const std::vector<std::vector<optix::float3>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dfloat4(RTbuffer &buffer, const std::vector<std::vector<optix::float4>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Di(RTbuffer &buffer, const std::vector<std::vector<int>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dui(RTbuffer &buffer, const std::vector<std::vector<uint>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dint2(RTbuffer &buffer, const std::vector<std::vector<optix::int2>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dint3(RTbuffer &buffer, const std::vector<std::vector<optix::int3>> &array);
    //! Set size of 2D buffer and initialize all elements based on a 2D array of floats.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 2D array used to initialize buffer.
     */
    void initializeBuffer2Dbool(RTbuffer &buffer, const std::vector<std::vector<bool>> &array);

    //! Set size of 3D buffer and initialize all elements based on a 3D array.
    /**
     * \param[inout] buffer OptiX buffer to be initialized.
     * \param[in] array 3D array used to initialize buffer.
     */
    template<typename anytype>
    void initializeBuffer3D(RTbuffer &buffer, const std::vector<std::vector<std::vector<anytype>>> &array);

    void buildLightModelGeometry(uint sourceID);

    void buildCameraModelGeometry(const std::string &cameralabel);

    void updateLightModelPosition(uint sourceID, const helios::vec3 &delta_position);

    void updateCameraModelPosition(const std::string &cameralabel);

    //! UUIDs for source 3D object models (for visualization). Key is the source ID, value is a vector of UUIDs for the source model.
    std::map<uint, std::vector<uint>> source_model_UUIDs;
    //! UUIDs for camera 3D object models (for visualization). Key is the camera label, value is a vector of UUIDs for the camera model.
    std::map<std::string, std::vector<uint>> camera_model_UUIDs;

    /* Primary RT API objects */

    //! OptiX context object
    RTcontext OptiX_Context;
    //! OptiX ray generation program handle for direct radiation
    RTprogram direct_raygen;
    //! OptiX ray generation program handle for diffuse radiation
    RTprogram diffuse_raygen;

    //! OptiX ray generation program handle for radiation cameras
    RTprogram camera_raygen;
    //! OptiX ray generation program handle for radiation camera pixel labeling
    RTprogram pixel_label_raygen;

    /* Variables */

    RTgeometrygroup base_geometry_group;

    //! Random number generator seed
    RTvariable random_seed_RTvariable;

    //! Primitive offset used for tiling ray launches
    RTvariable launch_offset_RTvariable;

    //! Flag designating which face of the primitive the launch is for
    RTvariable launch_face_RTvariable;

    //! Maximum scattering depth
    RTvariable max_scatters_RTvariable;
    RTbuffer max_scatters_RTbuffer;

    //! Number of radiative bands in the radiation model
    RTvariable Nbands_global_RTvariable;

    //! Number of radiative bands in the current launch
    RTvariable Nbands_launch_RTvariable;

    //! Flag to disable launches for certain bands
    RTvariable band_launch_flag_RTvariable;
    RTbuffer band_launch_flag_RTbuffer;

    //! Number of Context primitives
    RTvariable Nprimitives_RTvariable;

    //! Flux of ambient/diffuse radiation
    RTvariable diffuse_flux_RTvariable;
    RTbuffer diffuse_flux_RTbuffer;

    //! Diffuse distribution coefficient of diffuse ambient radiation
    RTvariable diffuse_extinction_RTvariable;
    RTbuffer diffuse_extinction_RTbuffer;

    //! Direction of peak diffuse radiation
    RTvariable diffuse_peak_dir_RTvariable;
    RTbuffer diffuse_peak_dir_RTbuffer;

    //! Diffuse distribution normalization factor
    RTvariable diffuse_dist_norm_RTvariable;
    RTbuffer diffuse_dist_norm_RTbuffer;

    //! Radiation emission flag
    RTvariable emission_flag_RTvariable;
    RTbuffer emission_flag_RTbuffer;

    //! Periodic boundary condition
    helios::vec2 periodic_flag;
    RTvariable periodic_flag_RTvariable;

    //! Energy absorbed by the "sky"
    RTvariable Rsky_RTvariable;

    //! Primitive reflectivity - RTbuffer
    RTbuffer rho_RTbuffer;
    //! Primitive reflectivity - RTvariable
    RTvariable rho_RTvariable;
    //! Primitive transmissivity - RTbuffer
    RTbuffer tau_RTbuffer;
    //! Primitive transmissivity - RTvariable
    RTvariable tau_RTvariable;

    //! Primitive reflectivity weighted by camera response - RTbuffer
    RTbuffer rho_cam_RTbuffer;
    //! Primitive reflectivity weighted by camera response - RTvariable
    RTvariable rho_cam_RTvariable;
    //! Primitive transmissivity weighted by camera response - RTbuffer
    RTbuffer tau_cam_RTbuffer;
    //! Primitive transmissivity weighted by camera response - RTvariable
    RTvariable tau_cam_RTvariable;

    //! Primitive specular reflection exponent - RTbuffer
    RTbuffer specular_exponent_RTbuffer;
    //! Primitive specular reflection exponent - RTvariable
    RTvariable specular_exponent_RTvariable;

    //! Primitive specular reflection scale coefficient - RTbuffer
    RTbuffer specular_scale_RTbuffer;
    //! Primitive specular reflection scale coefficient - RTvariable
    RTvariable specular_scale_RTvariable;

    //! Flag indicating whether specular reflection is enabled - RTvariable
    /**
     * = 0 -> specular reflection is disabled
     * = 1 -> specular reflection is enabled with scale coefficient of 1.0
     * = 2 -> specular reflection is enabled with per-primitive scale coefficient specified
     */
    RTvariable specular_reflection_enabled_RTvariable;

    //! Primitive type - RTbuffer object
    RTbuffer primitive_type_RTbuffer;
    //! Primitive type - RTvariable
    RTvariable primitive_type_RTvariable;

    //! Primitive area - RTbuffer object
    RTbuffer primitive_solid_fraction_RTbuffer;
    //! Primitive area - RTvariable
    RTvariable primitive_solid_fraction_RTvariable;

    //! Primitive UUIDs - RTbuffer object
    RTbuffer patch_UUID_RTbuffer;
    RTbuffer triangle_UUID_RTbuffer;
    RTbuffer disk_UUID_RTbuffer;
    RTbuffer tile_UUID_RTbuffer;
    RTbuffer voxel_UUID_RTbuffer;
    RTbuffer bbox_UUID_RTbuffer;
    //! Primitive UUIDs - RTvariable object
    RTvariable patch_UUID_RTvariable;
    RTvariable triangle_UUID_RTvariable;
    RTvariable disk_UUID_RTvariable;
    RTvariable tile_UUID_RTvariable;
    RTvariable voxel_UUID_RTvariable;
    RTvariable bbox_UUID_RTvariable;

    //! Mapping UUIDs to object IDs - RTbuffer object
    RTbuffer objectID_RTbuffer;
    //! Mapping UUIDs to object IDs - RTvariable object
    RTvariable objectID_RTvariable;

    //! Mapping object IDs to UUIDs - RTbuffer object
    RTbuffer primitiveID_RTbuffer;
    //! Mapping object IDs to UUIDs - RTvariable object
    RTvariable primitiveID_RTvariable;

    //! Primitive two-sided flag - RTbuffer object
    RTbuffer twosided_flag_RTbuffer;
    //! Primitive two-sided flag - RTvariable
    RTvariable twosided_flag_RTvariable;

    //! Radiative flux lost to the sky - RTbuffer object
    RTbuffer Rsky_RTbuffer;

    //-- Patch Buffers --//
    RTbuffer patch_vertices_RTbuffer;
    RTvariable patch_vertices_RTvariable;

    //-- Triangle Buffers --//
    RTbuffer triangle_vertices_RTbuffer;
    RTvariable triangle_vertices_RTvariable;

    //-- Disk Buffers --//
    RTbuffer disk_centers_RTbuffer;
    RTvariable disk_centers_RTvariable;
    RTbuffer disk_radii_RTbuffer;
    RTvariable disk_radii_RTvariable;
    RTbuffer disk_normals_RTbuffer;
    RTvariable disk_normals_RTvariable;

    //-- Tile Buffers --//
    RTbuffer tile_vertices_RTbuffer;
    RTvariable tile_vertices_RTvariable;

    //-- Voxel Buffers --//
    RTbuffer voxel_vertices_RTbuffer;
    RTvariable voxel_vertices_RTvariable;

    //-- Bounding Box Buffers --//
    RTbuffer bbox_vertices_RTbuffer;
    RTvariable bbox_vertices_RTvariable;

    //-- Object Buffers --//
    RTbuffer object_subdivisions_RTbuffer;
    RTvariable object_subdivisions_RTvariable;

    /* Output Buffers */

    //! Primitive affine transformation matrix - RTbuffer object
    RTbuffer transform_matrix_RTbuffer;
    //! Primitive affine transformation matrix - RTvariable
    RTvariable transform_matrix_RTvariable;
    //! Primitive temperatures - RTbuffer object
    RTbuffer primitive_emission_RTbuffer;
    //! Primitive temperatures - RTvariable
    RTvariable primitive_emission_RTvariable;

    //! Incoming radiative energy for each object - RTbuffer object
    RTbuffer radiation_in_RTbuffer;
    //! Incoming radiative energy for each object - RTvariable
    RTvariable radiation_in_RTvariable;
    //! Outgoing radiative energy (reflected/emitted) for top surface of each object - RTbuffer object
    RTbuffer radiation_out_top_RTbuffer;
    //! Outgoing radiative energy (reflected/emitted) for top surface each object - RTvariable
    RTvariable radiation_out_top_RTvariable;
    //! Outgoing radiative energy (reflected/emitted) for bottom surface of each object - RTbuffer object
    RTbuffer radiation_out_bottom_RTbuffer;
    //! Outgoing radiative energy (reflected/emitted) for bottom surface each object - RTvariable
    RTvariable radiation_out_bottom_RTvariable;
    //! "to-be-scattered" radiative energy (reflected/emitted) for top surface of each object - RTbuffer object
    RTbuffer scatter_buff_top_RTbuffer;
    //! "to-be-scattered" radiative energy (reflected/emitted) for top surface each object - RTvariable
    RTvariable scatter_buff_top_RTvariable;
    //! "to-be-scattered" radiative energy (reflected/emitted) for bottom surface of each object - RTbuffer object
    RTbuffer scatter_buff_bottom_RTbuffer;
    //! "to-be-scattered" radiative energy (reflected/emitted) for bottom surface each object - RTvariable
    RTvariable scatter_buff_bottom_RTvariable;

    //! Incoming radiative energy for each camera pixel - RTbuffer
    RTbuffer radiation_in_camera_RTbuffer;
    //! Incoming radiative energy for each camera pixel - RTvariable
    RTvariable radiation_in_camera_RTvariable;

    //! Camera "to-be-scattered" radiative energy (reflected/emitted) for top surface of each object - RTbuffer object
    RTbuffer scatter_buff_top_cam_RTbuffer;
    //! Camera "to-be-scattered" radiative energy (reflected/emitted) for top surface each object - RTvariable
    RTvariable scatter_buff_top_cam_RTvariable;
    //! Camera "to-be-scattered" radiative energy (reflected/emitted) for bottom surface of each object - RTbuffer object
    RTbuffer scatter_buff_bottom_cam_RTbuffer;
    //! Camera "to-be-scattered" radiative energy (reflected/emitted) for bottom surface each object - RTvariable
    RTvariable scatter_buff_bottom_cam_RTvariable;

    //! Pixel label primitive ID - RTbuffer
    RTbuffer camera_pixel_label_RTbuffer;
    //! Pixel label primitive ID - RTvariable
    RTvariable camera_pixel_label_RTvariable;

    //! Pixel depth - RTbuffer
    RTbuffer camera_pixel_depth_RTbuffer;
    //! Pixel depth - RTvariable
    RTvariable camera_pixel_depth_RTvariable;

    //! Mask data for texture masked Patches - RTbuffer object
    RTbuffer maskdata_RTbuffer;
    //! Mask data for texture masked Patches - RTvariable
    RTvariable maskdata_RTvariable;
    //! Size of mask data for texture masked Patches - RTbuffer object
    RTbuffer masksize_RTbuffer;
    //! Size of mask data for texture masked Patches - RTvariable object
    RTvariable masksize_RTvariable;
    //! ID of mask data (0...Nmasks-1) - RTbuffer object
    RTbuffer maskID_RTbuffer;
    //! ID of mask data (0...Nmasks-1) - RTvariable object
    RTvariable maskID_RTvariable;
    //! uv data for textures - RTbuffer object
    RTbuffer uvdata_RTbuffer;
    //! uv data for textures - RTvariable
    RTvariable uvdata_RTvariable;
    //! ID of uv data (0...Nuv-1) - RTbuffer object
    RTbuffer uvID_RTbuffer;
    //! ID of uv data (0...Nuv-1) - RTvariable
    RTvariable uvID_RTvariable;


    /* Ray Types */

    //! Handle to OptiX ray type for direct radiation rays.
    RTvariable direct_ray_type_RTvariable;
    //! Handle to OptiX ray type for diffuse radiation rays.
    RTvariable diffuse_ray_type_RTvariable;

    // Handle to OptiX ray type for camera rays
    RTvariable camera_ray_type_RTvariable;
    // Handle to OptiX ray type for camera pixel labeling rays
    RTvariable pixel_label_ray_type_RTvariable;

    //! OptiX Ray Types
    enum RayType { RAYTYPE_DIRECT = 0, RAYTYPE_DIFFUSE = 1, RAYTYPE_CAMERA = 2, RAYTYPE_PIXEL_LABEL = 3 };

    /* OptiX Geometry Structures */
    RTgeometry patch;
    RTgeometry triangle;
    RTgeometry disk;
    RTgeometry tile;
    RTgeometry voxel;
    RTgeometry bbox;
    RTmaterial patch_material;
    RTmaterial triangle_material;
    RTmaterial disk_material;
    RTmaterial tile_material;
    RTmaterial voxel_material;
    RTmaterial bbox_material;

    RTgroup top_level_group;
    RTacceleration top_level_acceleration;
    RTvariable top_object;
    RTacceleration geometry_acceleration;


    //! Flag indicating whether geometry has been built
    bool isgeometryinitialized;

    bool radiativepropertiesneedupdate = true;

    std::vector<bool> isbandpropertyinitialized;

    bool islightvisualizationenabled = false;
    bool iscameravisualizationenabled = false;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;

    std::vector<std::string> spectral_library_files;
};

void sutilHandleError(RTcontext context, RTresult code, const char *file, int line);

void sutilReportError(const char *message);

/* assumes current scope has Context variable named 'OptiX_Context' */
#define RT_CHECK_ERROR(func)                                                                                                                                                                                                                             \
    do {                                                                                                                                                                                                                                                 \
        RTresult code = func;                                                                                                                                                                                                                            \
        if (code != RT_SUCCESS)                                                                                                                                                                                                                          \
            sutilHandleError(OptiX_Context, code, __FILE__, __LINE__);                                                                                                                                                                                   \
    } while (0)

#endif
