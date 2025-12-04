/** \file "LensFlare.h" Header file for lens flare rendering algorithms.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef LENS_FLARE_H
#define LENS_FLARE_H

#include "Context.h"
#include <complex>
#include <map>
#include <string>
#include <vector>

// Forward declaration - full definition is in RadiationModel.h
// LensFlareProperties must be defined before this header is included
struct LensFlareProperties;

//! Class for computing physically-based lens flare effects
/**
 * Implements lens flare rendering using two components:
 * 1. Ghost reflections - Internal lens reflections computed using Fresnel equations
 * 2. Starburst diffraction - Aperture diffraction pattern computed via FFT
 */
class LensFlare {
public:
    //! Constructor
    /**
     * \param[in] lens_flare_props Lens flare rendering properties
     * \param[in] resolution Camera resolution (width, height)
     */
    LensFlare(const LensFlareProperties &lens_flare_props, helios::int2 resolution);

    //! Apply lens flare effect to image data
    /**
     * Applies both ghost reflections and starburst diffraction to the camera image.
     * \param[in,out] pixel_data Map of band labels to pixel intensity vectors
     * \param[in] resolution Image resolution (width, height)
     */
    void apply(std::map<std::string, std::vector<float>> &pixel_data, helios::int2 resolution);

    //! Find bright pixels that will generate lens flare
    /**
     * \param[in] pixel_data Map of band labels to pixel intensity vectors
     * \param[in] resolution Image resolution
     * \param[in] threshold Minimum intensity threshold (0-1)
     * \return Vector of (x, y, intensity) tuples for bright pixels
     */
    std::vector<std::tuple<int, int, float>> findBrightPixels(const std::map<std::string, std::vector<float>> &pixel_data, helios::int2 resolution, float threshold) const;

private:
    //! Generate the starburst kernel via FFT of aperture mask
    /**
     * Creates a diffraction pattern by computing the FFT of a polygon aperture shape.
     * The number of spikes equals the number of aperture blades.
     */
    void generateStarburstKernel();

    //! Generate aperture mask for FFT
    /**
     * Creates an image of an N-sided polygon representing the aperture shape.
     * \param[out] mask Output aperture mask (size kernel_size_ x kernel_size_)
     */
    void generateApertureMask(std::vector<float> &mask) const;

    //! Apply starburst effect to bright pixels
    /**
     * \param[in,out] pixel_data Image data to modify
     * \param[in] resolution Image resolution
     * \param[in] bright_pixels List of bright pixel locations and intensities
     */
    void applyStarburst(std::map<std::string, std::vector<float>> &pixel_data, helios::int2 resolution, const std::vector<std::tuple<int, int, float>> &bright_pixels);

    //! Apply ghost reflection effect to bright pixels
    /**
     * Renders ghost reflections based on Fresnel reflection physics.
     * \param[in,out] pixel_data Image data to modify
     * \param[in] resolution Image resolution
     * \param[in] bright_pixels List of bright pixel locations and intensities
     */
    void applyGhosts(std::map<std::string, std::vector<float>> &pixel_data, helios::int2 resolution, const std::vector<std::tuple<int, int, float>> &bright_pixels);

    //! Compute Fresnel reflectance at a surface interface
    /**
     * Uses Schlick's approximation for unpolarized light.
     * \param[in] cos_theta Cosine of incident angle
     * \param[in] n1 Refractive index of first medium (typically air = 1.0)
     * \param[in] n2 Refractive index of second medium (typically glass = 1.5)
     * \return Reflectance value in range [0, 1]
     */
    static float fresnelReflectance(float cos_theta, float n1 = 1.0f, float n2 = 1.5f);

    //! Compute 2D FFT using simple DFT algorithm
    /**
     * Computes discrete Fourier transform for small kernel sizes.
     * \param[in] input Input data (real values)
     * \param[out] output Output data (complex values)
     * \param[in] size Width/height of square input (must be power of 2)
     */
    static void fft2D(const std::vector<float> &input, std::vector<std::complex<float>> &output, int size);

    //! Compute 1D FFT using Cooley-Tukey algorithm
    /**
     * \param[in,out] data Complex data to transform in-place
     * \param[in] size Size of data (must be power of 2)
     * \param[in] inverse If true, compute inverse FFT
     */
    static void fft1D(std::vector<std::complex<float>> &data, int size, bool inverse = false);

    //! Render a soft disc (ghost artifact) at a position
    /**
     * \param[in,out] channel Single channel pixel data to modify
     * \param[in] resolution Image resolution
     * \param[in] center_x X coordinate of disc center
     * \param[in] center_y Y coordinate of disc center
     * \param[in] radius Disc radius in pixels
     * \param[in] intensity Peak intensity at disc center
     */
    void renderSoftDisc(std::vector<float> &channel, helios::int2 resolution, float center_x, float center_y, float radius, float intensity) const;

    //! Lens flare properties
    LensFlareProperties props_;

    //! Camera resolution
    helios::int2 resolution_;

    //! Pre-computed starburst kernel (magnitude of FFT)
    std::vector<float> starburst_kernel_;

    //! Starburst kernel size (power of 2)
    int kernel_size_ = 128;

    //! Ghost scale factors (position multipliers relative to screen center)
    static constexpr float ghost_scales_[] = {0.3f, 0.5f, 0.8f, 1.2f, 1.6f};

    //! Ghost size factors (radius multipliers)
    static constexpr float ghost_sizes_[] = {0.02f, 0.03f, 0.015f, 0.025f, 0.04f};
};

#endif // LENS_FLARE_H
