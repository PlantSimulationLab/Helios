/** \file "LensFlare.cpp" Implementation of lens flare rendering algorithms.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RadiationModel.h" // Must be included first for LensFlareProperties definition
#include "LensFlare.h"
#include <algorithm>
#include <cmath>

using namespace helios;

// Define static constexpr arrays
constexpr float LensFlare::ghost_scales_[];
constexpr float LensFlare::ghost_sizes_[];

LensFlare::LensFlare(const LensFlareProperties &lens_flare_props, int2 resolution) : props_(lens_flare_props), resolution_(resolution) {
    // Determine kernel size based on image resolution (use power of 2)
    int max_dim = std::max(resolution.x, resolution.y);
    kernel_size_ = 64; // Minimum kernel size
    while (kernel_size_ < max_dim / 4 && kernel_size_ < 256) {
        kernel_size_ *= 2;
    }

    // Pre-compute the starburst kernel
    generateStarburstKernel();
}

void LensFlare::apply(std::map<std::string, std::vector<float>> &pixel_data, int2 resolution) {
    if (pixel_data.empty()) {
        return;
    }

    // Find bright pixels that will generate lens flare
    auto bright_pixels = findBrightPixels(pixel_data, resolution, props_.intensity_threshold);

    if (bright_pixels.empty()) {
        return; // No bright pixels, nothing to do
    }

    // Apply starburst diffraction pattern
    if (props_.starburst_intensity > 0.0f) {
        applyStarburst(pixel_data, resolution, bright_pixels);
    }

    // Apply ghost reflections
    if (props_.ghost_intensity > 0.0f) {
        applyGhosts(pixel_data, resolution, bright_pixels);
    }
}

std::vector<std::tuple<int, int, float>> LensFlare::findBrightPixels(const std::map<std::string, std::vector<float>> &pixel_data, int2 resolution, float threshold) const {
    std::vector<std::tuple<int, int, float>> bright_pixels;

    if (pixel_data.empty()) {
        return bright_pixels;
    }

    int num_pixels = resolution.x * resolution.y;

    // Get the first band to determine pixel count
    const auto &first_band = pixel_data.begin()->second;
    if (static_cast<int>(first_band.size()) != num_pixels) {
        return bright_pixels;
    }

    // Calculate per-pixel maximum intensity across all bands
    std::vector<float> max_intensity(num_pixels, 0.0f);
    for (const auto &band_pair: pixel_data) {
        const auto &band_data = band_pair.second;
        for (int i = 0; i < num_pixels; ++i) {
            max_intensity[i] = std::max(max_intensity[i], band_data[i]);
        }
    }

    // Find pixels above threshold
    for (int j = 0; j < resolution.y; ++j) {
        for (int i = 0; i < resolution.x; ++i) {
            int idx = j * resolution.x + i;
            if (max_intensity[idx] >= threshold) {
                bright_pixels.emplace_back(i, j, max_intensity[idx]);
            }
        }
    }

    // Sort by intensity (brightest first) and limit count for performance
    std::sort(bright_pixels.begin(), bright_pixels.end(), [](const auto &a, const auto &b) { return std::get<2>(a) > std::get<2>(b); });

    // Limit to top 100 bright pixels to maintain performance
    constexpr size_t max_bright_pixels = 100;
    if (bright_pixels.size() > max_bright_pixels) {
        bright_pixels.resize(max_bright_pixels);
    }

    return bright_pixels;
}

void LensFlare::generateStarburstKernel() {
    // Generate aperture mask
    std::vector<float> aperture_mask;
    generateApertureMask(aperture_mask);

    // Compute 2D FFT
    std::vector<std::complex<float>> fft_result;
    fft2D(aperture_mask, fft_result, kernel_size_);

    // Extract magnitude and normalize
    starburst_kernel_.resize(kernel_size_ * kernel_size_);
    float max_magnitude = 0.0f;
    for (size_t i = 0; i < fft_result.size(); ++i) {
        float mag = std::abs(fft_result[i]);
        starburst_kernel_[i] = mag;
        max_magnitude = std::max(max_magnitude, mag);
    }

    // Normalize kernel to [0, 1]
    if (max_magnitude > 0.0f) {
        for (float &val: starburst_kernel_) {
            val /= max_magnitude;
        }
    }

    // Apply radial window to fade at edges (eliminates square box artifact)
    float center = static_cast<float>(kernel_size_) / 2.0f;
    float max_dist = center * 1.414f; // sqrt(2) * center = corner distance
    for (int y = 0; y < kernel_size_; ++y) {
        for (int x = 0; x < kernel_size_; ++x) {
            size_t i = y * kernel_size_ + x;
            float dx = x - center;
            float dy = y - center;
            float dist = std::sqrt(dx * dx + dy * dy);
            float normalized_dist = std::min(dist / center, 1.0f); // Normalize to aperture radius

            // Smooth cosine window: 1 at center, 0 at radius
            float window = (normalized_dist < 1.0f) ? (0.5f + 0.5f * std::cos(normalized_dist * M_PI)) : 0.0f;
            starburst_kernel_[i] *= window;
        }
    }

    // Apply power falloff to make spikes more pronounced
    for (float &val: starburst_kernel_) {
        val = std::pow(val, 0.5f); // Square root to enhance spikes
    }
}

void LensFlare::generateApertureMask(std::vector<float> &mask) const {
    mask.resize(kernel_size_ * kernel_size_, 0.0f);

    int blade_count = props_.aperture_blade_count;
    float center = static_cast<float>(kernel_size_) / 2.0f;
    float radius = center * 0.8f; // Aperture radius (80% of half-size)

    // Generate N-gon vertices
    std::vector<std::pair<float, float>> vertices(blade_count);
    for (int i = 0; i < blade_count; ++i) {
        float angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(blade_count);
        // Rotate by 90 degrees so one point is at top
        angle += static_cast<float>(M_PI) / 2.0f;
        vertices[i] = {center + radius * std::cos(angle), center + radius * std::sin(angle)};
    }

    // Fill polygon using scanline algorithm
    for (int y = 0; y < kernel_size_; ++y) {
        for (int x = 0; x < kernel_size_; ++x) {
            // Point-in-polygon test using ray casting
            float px = static_cast<float>(x) + 0.5f;
            float py = static_cast<float>(y) + 0.5f;

            int crossings = 0;
            for (int i = 0; i < blade_count; ++i) {
                int j = (i + 1) % blade_count;
                float x1 = vertices[i].first, y1 = vertices[i].second;
                float x2 = vertices[j].first, y2 = vertices[j].second;

                // Check if ray from (px, py) going right crosses edge
                if ((y1 <= py && y2 > py) || (y2 <= py && y1 > py)) {
                    // Compute x coordinate of intersection
                    float t = (py - y1) / (y2 - y1);
                    float x_intersect = x1 + t * (x2 - x1);
                    if (px < x_intersect) {
                        crossings++;
                    }
                }
            }

            // Odd number of crossings = inside polygon
            if (crossings % 2 == 1) {
                mask[y * kernel_size_ + x] = 1.0f;
            }
        }
    }

    // Apply soft edge anti-aliasing (optional smoothing)
    // Not implemented for simplicity - the FFT handles most aliasing
}

void LensFlare::applyStarburst(std::map<std::string, std::vector<float>> &pixel_data, int2 resolution, const std::vector<std::tuple<int, int, float>> &bright_pixels) {
    if (starburst_kernel_.empty() || bright_pixels.empty()) {
        return;
    }

    float intensity_scale = props_.starburst_intensity * 0.05f; // Reduced for subtler effect
    int half_kernel = kernel_size_ / 2;

    // Apply starburst kernel centered on each bright pixel
    for (const auto &bright_pixel: bright_pixels) {
        int bx = std::get<0>(bright_pixel);
        int by = std::get<1>(bright_pixel);
        float pixel_intensity = std::get<2>(bright_pixel);

        // Extract color ratios from the source pixel
        int pixel_idx = by * resolution.x + bx;
        std::map<std::string, float> color_ratios;
        float total = 0.0f;
        for (const auto &[band_label, band_data]: pixel_data) {
            float value = band_data[pixel_idx];
            color_ratios[band_label] = value;
            total += value;
        }
        // Normalize color ratios
        if (total > 0.0f) {
            for (auto &[band_label, ratio]: color_ratios) {
                ratio /= total;
            }
        }

        // Scale starburst by pixel brightness relative to threshold
        float brightness_factor = (pixel_intensity - props_.intensity_threshold) / (1.0f - props_.intensity_threshold + 0.001f);
        brightness_factor = std::clamp(brightness_factor, 0.0f, 1.0f);
        float scaled_intensity = intensity_scale * brightness_factor * pixel_intensity;

        // Add starburst contribution to each band with source color
        for (auto &[band_label, band_data]: pixel_data) {
            float band_scale = color_ratios[band_label] * scaled_intensity;

            for (int ky = 0; ky < kernel_size_; ++ky) {
                for (int kx = 0; kx < kernel_size_; ++kx) {
                    // Map kernel coordinate to image coordinate (centered on bright pixel)
                    int ix = bx + kx - half_kernel;
                    int iy = by + ky - half_kernel;

                    // Skip pixels outside image bounds
                    if (ix < 0 || ix >= resolution.x || iy < 0 || iy >= resolution.y) {
                        continue;
                    }

                    int img_idx = iy * resolution.x + ix;
                    int kern_idx = ky * kernel_size_ + kx;

                    // Add kernel contribution with color (additive blending)
                    band_data[img_idx] += starburst_kernel_[kern_idx] * band_scale;
                }
            }
        }
    }
}

void LensFlare::applyGhosts(std::map<std::string, std::vector<float>> &pixel_data, int2 resolution, const std::vector<std::tuple<int, int, float>> &bright_pixels) {
    if (bright_pixels.empty()) {
        return;
    }

    float center_x = static_cast<float>(resolution.x) / 2.0f;
    float center_y = static_cast<float>(resolution.y) / 2.0f;

    // Base reflectance from coating efficiency
    // Each ghost involves 2 reflections, so intensity = (1 - coating_efficiency)^2
    float coating_reflectance = 1.0f - props_.coating_efficiency;
    float base_reflectance = coating_reflectance * coating_reflectance * props_.ghost_intensity;

    int num_ghosts = std::min(props_.ghost_count, static_cast<int>(sizeof(ghost_scales_) / sizeof(ghost_scales_[0])));

    for (const auto &bright_pixel: bright_pixels) {
        int bx = std::get<0>(bright_pixel);
        int by = std::get<1>(bright_pixel);
        float pixel_intensity = std::get<2>(bright_pixel);

        // Extract color from source pixel
        int pixel_idx = by * resolution.x + bx;
        std::map<std::string, float> source_color;
        for (const auto &[band_label, band_data]: pixel_data) {
            source_color[band_label] = band_data[pixel_idx];
        }

        // Vector from center to bright pixel
        float dx = static_cast<float>(bx) - center_x;
        float dy = static_cast<float>(by) - center_y;

        // Distance from center (for Fresnel calculation)
        float dist_from_center = std::sqrt(dx * dx + dy * dy);
        float max_dist = std::sqrt(center_x * center_x + center_y * center_y);
        float normalized_dist = dist_from_center / max_dist;

        // Approximate incident angle for Fresnel (0 at center, increases toward edges)
        float cos_theta = std::sqrt(1.0f - normalized_dist * normalized_dist * 0.25f);
        float fresnel = fresnelReflectance(cos_theta);

        // Render each ghost
        for (int g = 0; g < num_ghosts; ++g) {
            // Ghost position is reflected through center with scaling
            float ghost_x = center_x - ghost_scales_[g] * dx;
            float ghost_y = center_y - ghost_scales_[g] * dy;

            // Ghost radius based on image diagonal
            float image_diagonal = std::sqrt(static_cast<float>(resolution.x * resolution.x + resolution.y * resolution.y));
            float ghost_radius = image_diagonal * ghost_sizes_[g];

            // Ghost base intensity (physical attenuation only)
            float ghost_base = base_reflectance * fresnel * std::exp(-ghost_scales_[g] * 0.5f);

            // Separate luminance and chrominance (production approach)
            // Step 1: Find max channel as luminance proxy
            float luminance = 0.0f;
            for (const auto &[band_label, val]: source_color) {
                luminance = std::max(luminance, val);
            }

            // Step 2: Extract normalized chrominance (color ratios)
            std::map<std::string, float> chrominance;
            for (const auto &[band_label, val]: source_color) {
                chrominance[band_label] = val / std::max(luminance, 1e-6f);
            }

            // Step 3: Compress luminance only with Reinhard
            float compressed_luminance = luminance / (1.0f + luminance);

            // Step 4: Apply ghost physics attenuation and visibility scaling
            // compressed_luminance ≈ 1.0 for bright sources, ghost_base ≈ 5e-5
            // Reduced to 20% of previous strength for subtler effect
            float ghost_luminance = compressed_luminance * ghost_base * 40.0f;

            // Apply to each band: chrominance × compressed luminance
            for (auto &[band_label, band_data]: pixel_data) {
                // Recombine: color ratio × compressed brightness
                float band_intensity = chrominance[band_label] * ghost_luminance;

                // Render without chromatic aberration for now (can add back later as optional)
                renderSoftDisc(band_data, resolution, ghost_x, ghost_y, ghost_radius, band_intensity);
            }
        }
    }
}

float LensFlare::fresnelReflectance(float cos_theta, float n1, float n2) {
    // Schlick's approximation for unpolarized light
    float r0 = ((n1 - n2) / (n1 + n2));
    r0 = r0 * r0;
    float one_minus_cos = 1.0f - cos_theta;
    float one_minus_cos_5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
    return r0 + (1.0f - r0) * one_minus_cos_5;
}

void LensFlare::renderSoftDisc(std::vector<float> &channel, int2 resolution, float center_x, float center_y, float radius, float intensity) const {
    if (intensity <= 0.0f || radius <= 0.0f) {
        return;
    }

    // Bounding box for the disc
    int x_min = std::max(0, static_cast<int>(center_x - radius - 1.0f));
    int x_max = std::min(resolution.x - 1, static_cast<int>(center_x + radius + 1.0f));
    int y_min = std::max(0, static_cast<int>(center_y - radius - 1.0f));
    int y_max = std::min(resolution.y - 1, static_cast<int>(center_y + radius + 1.0f));

    float radius_sq = radius * radius;

    // Solid disc with smooth falloff from bright center to transparent edge
    for (int y = y_min; y <= y_max; ++y) {
        for (int x = x_min; x <= x_max; ++x) {
            float dx = static_cast<float>(x) + 0.5f - center_x;
            float dy = static_cast<float>(y) + 0.5f - center_y;
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq <= radius_sq) {
                float dist_ratio = std::sqrt(dist_sq) / radius;

                // Smooth falloff: 1.0 at center, 0.0 at edge
                // Using 1 - r² for gradual falloff
                float falloff = 1.0f - dist_ratio * dist_ratio;
                falloff = std::max(0.0f, falloff);

                int idx = y * resolution.x + x;
                channel[idx] += intensity * falloff;
            }
        }
    }
}

void LensFlare::fft2D(const std::vector<float> &input, std::vector<std::complex<float>> &output, int size) {
    // Initialize output with input values
    output.resize(size * size);
    for (int i = 0; i < size * size; ++i) {
        output[i] = std::complex<float>(input[i], 0.0f);
    }

    // Apply FFT to each row
    std::vector<std::complex<float>> row(size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            row[x] = output[y * size + x];
        }
        fft1D(row, size);
        for (int x = 0; x < size; ++x) {
            output[y * size + x] = row[x];
        }
    }

    // Apply FFT to each column
    std::vector<std::complex<float>> col(size);
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            col[y] = output[y * size + x];
        }
        fft1D(col, size);
        for (int y = 0; y < size; ++y) {
            output[y * size + x] = col[y];
        }
    }

    // FFT shift to center the zero-frequency component
    int half = size / 2;
    for (int y = 0; y < half; ++y) {
        for (int x = 0; x < half; ++x) {
            // Swap quadrants
            std::swap(output[y * size + x], output[(y + half) * size + (x + half)]);
            std::swap(output[y * size + (x + half)], output[(y + half) * size + x]);
        }
    }
}

void LensFlare::fft1D(std::vector<std::complex<float>> &data, int size, bool inverse) {
    // Bit-reversal permutation
    int n = size;
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // Cooley-Tukey FFT
    for (int len = 2; len <= n; len *= 2) {
        float angle = 2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        if (inverse) {
            angle = -angle;
        }
        std::complex<float> wn(std::cos(angle), std::sin(angle));

        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int jj = 0; jj < len / 2; ++jj) {
                std::complex<float> u = data[i + jj];
                std::complex<float> t = w * data[i + jj + len / 2];
                data[i + jj] = u + t;
                data[i + jj + len / 2] = u - t;
                w *= wn;
            }
        }
    }

    // Scale for inverse FFT
    if (inverse) {
        for (auto &val: data) {
            val /= static_cast<float>(n);
        }
    }
}
