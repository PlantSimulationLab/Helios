/** \file "RadiationModel.cpp" Primary source file for radiation transport model.

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

#include "RadiationModel.h"
#include "BufferIndexing.h"
#include <climits>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

using namespace helios;

RadiationModel::RadiationModel(helios::Context *context_a) {

    context = context_a;

    // Asset directory registration removed - now using HELIOS_BUILD resolution

    // All default values set here

    message_flag = true;

    directRayCount_default = 100;
    diffuseRayCount_default = 1000;

    diffuseFlux_default = -1.f;

    minScatterEnergy_default = 0.f;
    scatteringDepth_default = 3;

    rho_default = 0.f;
    tau_default = 0.f;
    eps_default = 1.f;

    kappa_default = 1.f;
    sigmas_default = 0.f;

    temperature_default = 300;

    periodic_flag = make_vec2(0, 0);
    periodic_xbounds = make_vec2(0, 0);
    periodic_ybounds = make_vec2(0, 0);

    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/camera_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/light_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/soil_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/leaf_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/bark_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/fruit_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/solar_spectrum_ASTMG173.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/DGK_DKK_colorboard.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/cie_colorimetry.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Datacolor_SpyderCHECKR_24_colorboard.xml").string());

    // Initialize backend abstraction layer with runtime hardware detection
    backend = helios::RayTracingBackend::create("auto");
    backend->initialize();

    if (message_flag) {
        std::string backend_name = backend->getBackendName();
        std::cout << "Radiation model initialized with " << backend_name << " backend";
        if (backend_name.find("Vulkan") != std::string::npos) {
            std::cout << " - WARNING: radiation model may be slow depending on your GPU (NVIDIA+OptiX backend recommended)";
        } else {
            std::cout << ".";
        }
        std::cout << std::endl;
    }
}

RadiationModel::RadiationModel(helios::Context *context_a, bool skip_backend_init) {
    context = context_a;

    // Initialize all default values (same as main constructor)
    message_flag = true;
    directRayCount_default = 100;
    diffuseRayCount_default = 1000;
    diffuseFlux_default = -1.f;
    minScatterEnergy_default = 0.f;
    scatteringDepth_default = 3;
    rho_default = 0.f;
    tau_default = 0.f;
    eps_default = 1.f;
    kappa_default = 1.f;
    sigmas_default = 0.f;
    temperature_default = 300;
    periodic_flag = make_vec2(0, 0);
    periodic_xbounds = make_vec2(0, 0);
    periodic_ybounds = make_vec2(0, 0);

    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/camera_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/light_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/soil_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/leaf_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/bark_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/fruit_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/solar_spectrum_ASTMG173.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/DGK_DKK_colorboard.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/cie_colorimetry.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Datacolor_SpyderCHECKR_24_colorboard.xml").string());

    // Skip backend creation - will be injected by caller
}

RadiationModel RadiationModel::createWithBackend(helios::Context *context, std::unique_ptr<helios::RayTracingBackend> backend) {
    RadiationModel model(context, true); // Use private constructor, skip backend init

    // Inject the provided backend
    model.backend = std::move(backend);

    return model; // Uses move constructor
}

bool RadiationModel::isGPUBackendAvailable() {
    // Single source of truth: probeAnyGPUBackend() caches its result process-wide and honors
    // the HELIOS_NO_GPU veto, so what is reported here can never diverge from what
    // RayTracingBackend::create("auto") does when the constructor runs.
    return helios::probeAnyGPUBackend();
}

RadiationModel::~RadiationModel() {
    // Backend's unique_ptr will automatically clean up OptiX context
}

void RadiationModel::disableMessages() {
    message_flag = false;
}

void RadiationModel::enableMessages() {
    message_flag = true;
}

std::string RadiationModel::getBackendName() const {
    if (backend) {
        return backend->getBackendName();
    }
    return "none";
}

void RadiationModel::optionalOutputPrimitiveData(const char *label) {

    if (strcmp(label, "reflectivity") == 0 || strcmp(label, "transmissivity") == 0) {
        output_prim_data.emplace_back(label);
    } else {
        std::cout << "WARNING (RadiationModel::optionalOutputPrimitiveData): unknown output primitive data " << label << std::endl;
    }
}

void RadiationModel::setDirectRayCount(const std::string &label, size_t N) {
    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::setDirectRayCount): Cannot set ray count for band '" + label + "' because it is not a valid band.");
    }
    radiation_bands.at(label).directRayCount = N;
}

void RadiationModel::setDiffuseRayCount(const std::string &label, size_t N) {
    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseRayCount): Cannot set ray count for band '" + label + "' because it is not a valid band.");
    }
    radiation_bands.at(label).diffuseRayCount = N;
}

void RadiationModel::setDiffuseRadiationFlux(const std::string &label, float flux) {
    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseRadiationFlux): Cannot set flux value for band '" + label + "' because it is not a valid band.");
    }
    radiation_bands.at(label).diffuseFlux = flux;
}

void RadiationModel::setDiffuseRadiationExtinctionCoeff(const std::string &label, float K, const SphericalCoord &peak_dir) {
    setDiffuseRadiationExtinctionCoeff(label, K, sphere2cart(peak_dir));
}

void RadiationModel::setDiffuseRadiationExtinctionCoeff(const std::string &label, float K, const vec3 &peak_dir) {
    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseRadiationExtinctionCoeff): Cannot set diffuse extinction value for band '" + label + "' because it is not a valid band.");
    }

    vec3 dir = peak_dir;
    dir.normalize();

    int N = 100;
    float norm = 0.f;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            float theta = 0.5f * M_PI / float(N) * (0.5f + float(i));
            float phi = 2.f * M_PI / float(N) * (0.5f + float(j));
            vec3 n = sphere2cart(make_SphericalCoord(0.5f * M_PI - theta, phi));

            float psi = acos_safe(n * dir);
            float fd;
            if (psi < M_PI / 180.f) {
                fd = powf(M_PI / 180.f, -K);
            } else {
                fd = powf(psi, -K);
            }

            norm += fd * cosf(theta) * sinf(theta) * M_PI / float(N * N);
            // note: the multipication factors are dtheta*dphi/pi = (0.5*pi/N)*(2*pi/N)/pi = pi/N^2
        }
    }

    radiation_bands.at(label).diffuseExtinction = K;
    radiation_bands.at(label).diffusePeakDir = dir;
    radiation_bands.at(label).diffuseDistNorm = 1.f / norm;
}

void RadiationModel::enableBelowHorizonDiffuseRadiation() {
    below_horizon_diffuse_enabled = true;
}

void RadiationModel::disableBelowHorizonDiffuseRadiation() {
    below_horizon_diffuse_enabled = false;
}

void RadiationModel::setDiffuseSpectrumIntegral(float spectrum_integral) {

    if (spectrum_integral < 0) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Spectrum integral must be non-negative.");
    } else if (global_diffuse_spectrum.empty()) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Global diffuse spectrum has not been set. Call setDiffuseSpectrum() first.");
    }

    // Scale the global spectrum
    float current_integral = integrateSpectrum(global_diffuse_spectrum);
    if (current_integral > 0) {
        float scale_factor = spectrum_integral / current_integral;
        for (vec2 &wavelength: global_diffuse_spectrum) {
            wavelength.y *= scale_factor;
        }
    }

    // Apply scaled spectrum to all existing bands
    for (auto &band: radiation_bands) {
        band.second.diffuse_spectrum = global_diffuse_spectrum;
    }

    radiativepropertiesneedupdate = true;
}

void RadiationModel::setDiffuseSpectrumIntegral(float spectrum_integral, float wavelength1, float wavelength2) {

    if (spectrum_integral < 0) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Spectrum integral must be non-negative.");
    } else if (global_diffuse_spectrum.empty()) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Global diffuse spectrum has not been set. Call setDiffuseSpectrum() first.");
    }

    // Scale the global spectrum based on the integral within the specified wavelength range
    float current_integral = integrateSpectrum(global_diffuse_spectrum, wavelength1, wavelength2);
    if (current_integral > 0) {
        float scale_factor = spectrum_integral / current_integral;
        for (vec2 &wavelength: global_diffuse_spectrum) {
            wavelength.y *= scale_factor;
        }
    }

    // Apply scaled spectrum to all existing bands
    for (auto &band: radiation_bands) {
        band.second.diffuse_spectrum = global_diffuse_spectrum;
    }

    radiativepropertiesneedupdate = true;
}

void RadiationModel::setDiffuseSpectrumIntegral(const std::string &band_label, float spectrum_integral) {

    if (spectrum_integral < 0) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Source integral must be non-negative.");
    } else if (!doesBandExist(band_label)) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Cannot set integral for band '" + band_label + "' because it is not a valid band.");
    } else if (radiation_bands.at(band_label).diffuse_spectrum.empty()) {
        std::cerr << "WARNING (RadiationModel::setDiffuseSpectrumIntegral): Diffuse spectral distribution has not been set for radiation band '" + band_label + "'. Cannot set its integral." << std::endl;
        return;
    }

    float current_integral = integrateSpectrum(radiation_bands.at(band_label).diffuse_spectrum);

    for (vec2 &wavelength: radiation_bands.at(band_label).diffuse_spectrum) {
        wavelength.y *= spectrum_integral / current_integral;
    }

    radiativepropertiesneedupdate = true;
}

void RadiationModel::setDiffuseSpectrumIntegral(const std::string &band_label, float spectrum_integral, float wavelength1, float wavelength2) {

    if (spectrum_integral < 0) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Source integral must be non-negative.");
    } else if (!doesBandExist(band_label)) {
        helios_runtime_error("ERROR (RadiationModel::setDiffuseSpectrumIntegral): Cannot set integral for band '" + band_label + "' because it is not a valid band.");
    }

    float current_integral = integrateSpectrum(radiation_bands.at(band_label).diffuse_spectrum, wavelength1, wavelength2);

    for (vec2 &wavelength: radiation_bands.at(band_label).diffuse_spectrum) {
        wavelength.y *= spectrum_integral / current_integral;
    }

    radiativepropertiesneedupdate = true;
}

void RadiationModel::addRadiationBand(const std::string &label) {

    if (radiation_bands.find(label) != radiation_bands.end()) {
        std::cerr << "WARNING (RadiationModel::addRadiationBand): Radiation band " << label << " has already been added. Skipping this call to addRadiationBand()." << std::endl;
        return;
    }

    RadiationBand band(label, directRayCount_default, diffuseRayCount_default, diffuseFlux_default, scatteringDepth_default, minScatterEnergy_default);

    // Apply global diffuse spectrum if one was set
    if (!global_diffuse_spectrum.empty()) {
        band.diffuse_spectrum = global_diffuse_spectrum;
    }

    radiation_bands.emplace(label, band);

    // Initialize all radiation source fluxes
    for (auto &source: radiation_sources) {
        source.source_fluxes[label] = -1.f;
    }

    radiativepropertiesneedupdate = true;
}

void RadiationModel::addRadiationBand(const std::string &label, float wavelength1, float wavelength2) {

    if (radiation_bands.find(label) != radiation_bands.end()) {
        std::cerr << "WARNING (RadiationModel::addRadiationBand): Radiation band " << label << " has already been added. Skipping this call to addRadiationBand()." << std::endl;
        return;
    } else if (wavelength1 < 0.f) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationBand): The lower wavelength bound of band " + label + " must not be negative, but " + std::to_string(wavelength1) + " nm was given.");
    } else if (wavelength1 > wavelength2) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationBand): The upper wavelength bound for a band must be greater than the lower bound.");
    } else if (wavelength2 - wavelength1 < 1) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationBand): The waveband range of a radiation band must be at least 1 nm.");
    }

    RadiationBand band(label, directRayCount_default, diffuseRayCount_default, diffuseFlux_default, scatteringDepth_default, minScatterEnergy_default);

    band.wavebandBounds = make_vec2(wavelength1, wavelength2);

    // Apply global diffuse spectrum if one was set
    if (!global_diffuse_spectrum.empty()) {
        band.diffuse_spectrum = global_diffuse_spectrum;
    }

    radiation_bands.emplace(label, band);

    // Initialize all radiation source fluxes
    for (auto &source: radiation_sources) {
        source.source_fluxes[label] = -1.f;
    }

    radiativepropertiesneedupdate = true;
}

void RadiationModel::copyRadiationBand(const std::string &old_label, const std::string &new_label) {

    if (!doesBandExist(old_label)) {
        helios_runtime_error("ERROR (RadiationModel::copyRadiationBand): Cannot copy band " + old_label + " because it does not exist.");
    }

    vec2 waveBounds = radiation_bands.at(old_label).wavebandBounds;

    copyRadiationBand(old_label, new_label, waveBounds.x, waveBounds.y);
}

void RadiationModel::copyRadiationBand(const std::string &old_label, const std::string &new_label, float wavelength_min, float wavelength_max) {

    if (!doesBandExist(old_label)) {
        helios_runtime_error("ERROR (RadiationModel::copyRadiationBand): Cannot copy band " + old_label + " because it does not exist.");
    }

    // (0,0) copies a band without wavelength bounds; any other bounds are validated as in addRadiationBand()
    if (wavelength_min != 0.f || wavelength_max != 0.f) {
        if (wavelength_min < 0.f) {
            helios_runtime_error("ERROR (RadiationModel::copyRadiationBand): The lower wavelength bound of band " + new_label + " must not be negative, but " + std::to_string(wavelength_min) + " nm was given.");
        } else if (wavelength_max - wavelength_min < 1.f) {
            helios_runtime_error("ERROR (RadiationModel::copyRadiationBand): The waveband range of band " + new_label + " must be at least 1 nm, with the upper bound greater than the lower bound.");
        }
    }

    RadiationBand band = radiation_bands.at(old_label);
    band.label = new_label;
    band.wavebandBounds = make_vec2(wavelength_min, wavelength_max);

    radiation_bands.emplace(new_label, band);

    // copy source fluxes
    for (auto &source: radiation_sources) {
        source.source_fluxes[new_label] = source.source_fluxes.at(old_label);
    }

    radiativepropertiesneedupdate = true;
}

bool RadiationModel::doesBandExist(const std::string &label) const {
    if (radiation_bands.find(label) == radiation_bands.end()) {
        return false;
    } else {
        return true;
    }
}

void RadiationModel::disableEmission(const std::string &label) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::disableEmission): Cannot disable emission for band '" + label + "' because it is not a valid band.");
    }

    radiation_bands.at(label).emissionFlag = false;
}

void RadiationModel::enableEmission(const std::string &label) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::enableEmission): Cannot disable emission for band '" + label + "' because it is not a valid band.");
    }

    radiation_bands.at(label).emissionFlag = true;
}

void RadiationModel::enableEmissionReferenceField(const std::string &label) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::enableEmissionReferenceField): Cannot enable the emission reference field for band '" + label + "' because it is not a valid band.");
    }

    radiation_bands.at(label).emissionReferenceFieldFlag = true;
}

void RadiationModel::disableEmissionReferenceField(const std::string &label) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::disableEmissionReferenceField): Cannot disable the emission reference field for band '" + label + "' because it is not a valid band.");
    }

    radiation_bands.at(label).emissionReferenceFieldFlag = false;
}

// --- Solar-induced chlorophyll fluorescence (SIF) v2 ---
//
// v2 replaces the v1 25%/75% fixed red/far-red split with a physically correct
// spectrum-at-the-source model. Per-leaf fluorescence is computed via the
// Fluspect-B kernel (Vilfan et al. 2016, ported in FluspectB.cpp) driven by
// absorbed PAR integrated across auto-generated excitation bands (400-750 nm)
// and the van der Tol (2014) rate-coefficient quantum yield Phi_F. The result
// is integrated over user-defined emission bands (any wavelength range the
// user picks when adding the band). Users interact with SIF by adding a
// SIFCamera; the emission bands it references are flagged as SIF-sourcing.

namespace {
    // Round a float to 5 decimals so tiny numerical differences don't prevent
    // cache hits for canonically-identical biochemistry.
    float fp_round5(float x) {
        return std::round(x * 1e5f) * 1e-5f;
    }
} // namespace

bool RadiationModel::FluspectCacheKey::operator==(const FluspectCacheKey &o) const noexcept {
    return biochem_label == o.biochem_label && excitation_step_nm == o.excitation_step_nm;
}

std::size_t RadiationModel::FluspectCacheKeyHash::operator()(const FluspectCacheKey &k) const noexcept {
    std::size_t h = std::hash<std::string>{}(k.biochem_label);
    std::uint32_t bits;
    std::memcpy(&bits, &k.excitation_step_nm, sizeof(bits));
    h ^= static_cast<std::size_t>(bits) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

void RadiationModel::ensureFluspectOptiparLoaded() {
    if (fluspect_optipar_loaded) {
        return;
    }
    const std::filesystem::path p = helios::resolveFilePath("plugins/radiation/spectral_data/fluspect_B_optipar.xml");
    helios::loadFluspectOptipar(p.string(), fluspect_optipar);
    fluspect_optipar_loaded = true;
}

const helios::FluspectKernel *RadiationModel::getOrComputeFluspectKernel(uint UUID, float excitation_step_nm) {
    // Label-based biochemistry lookup: the leaf must have a "fluspect_spectrum"
    // string primitive-data field that points at a corresponding
    // "fluspect_biochem_<label>" global data entry (authored by LeafOptics::run()
    // or directly by the user). This mirrors how "reflectivity_spectrum" works.
    if (!context->doesPrimitiveDataExist(UUID, "fluspect_spectrum")) {
        return nullptr;
    }
    std::string biochem_label;
    context->getPrimitiveData(UUID, "fluspect_spectrum", biochem_label);

    // Resolve and validate the global-data biochemistry vector the label points at.
    if (!context->doesGlobalDataExist(biochem_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::getOrComputeFluspectKernel): primitive " + std::to_string(UUID) +
                             " has fluspect_spectrum = '" + biochem_label + "' but that global data does not exist. "
                             "Either call LeafOptics::run() to author the biochemistry, or manually setGlobalData("
                             "\"" + biochem_label + "\", std::vector<float>{Cab, Cca, Cw, Cdm, Cs, Cant, Cp, Cbc, N, V2Z, fqe}).");
    }
    if (context->getGlobalDataType(biochem_label.c_str()) != HELIOS_TYPE_FLOAT) {
        helios_runtime_error("ERROR (RadiationModel::getOrComputeFluspectKernel): global data '" + biochem_label +
                             "' is not a float vector — must be std::vector<float> with 11 elements.");
    }
    std::vector<float> biochem_vec;
    context->getGlobalData(biochem_label.c_str(), biochem_vec);
    if (biochem_vec.size() != 11) {
        helios_runtime_error("ERROR (RadiationModel::getOrComputeFluspectKernel): global data '" + biochem_label +
                             "' has " + std::to_string(biochem_vec.size()) + " elements but must have exactly 11 "
                             "(Cab, Cca, Cw, Cdm, Cs, Cant, Cp, Cbc, N, V2Z, fqe).");
    }

    // Cache key: the global-data label + excitation step. A label is not a fixed identity -- the values behind it can be rewritten
    // (LeafOptics::run() reuses its nitrogen-bin labels with new pigment values every time it rebuilds its bins) -- so a cached kernel is
    // reused only while the biochemistry it was computed from is unchanged. fqe (the last field) does not enter the kernel, so a change to
    // it alone does not force a recompute.
    FluspectCacheKey key{biochem_label, fp_round5(excitation_step_nm)};
    const std::vector<float> kernel_biochem(biochem_vec.begin(), biochem_vec.begin() + 10);
    auto it = fluspect_cache.find(key);
    if (it != fluspect_cache.end()) {
        if (fluspect_cache_biochem.at(key) == kernel_biochem) {
            return &it->second;
        }
        fluspect_cache.erase(it);
    }

    helios::FluspectBiochemistry biochem;
    biochem.Cab  = biochem_vec[0];
    biochem.Cca  = biochem_vec[1];
    biochem.Cw   = biochem_vec[2];
    biochem.Cdm  = biochem_vec[3];
    biochem.Cs   = biochem_vec[4];
    biochem.Cant = biochem_vec[5];
    biochem.Cp   = biochem_vec[6];
    biochem.Cbc  = biochem_vec[7];
    biochem.N    = biochem_vec[8];
    biochem.V2Z  = biochem_vec[9];
    // fqe scales the entire kernel linearly; we factor it out by setting fqe=1 in
    // the kernel compute and multiplying by the per-leaf Phi_F × fqe at emission
    // time. The user's calibrated fqe scalar from biochem_vec[10] is applied in
    // computeSIFEmission alongside the van der Tol yield.
    biochem.fqe  = 1.f;

    ensureFluspectOptiparLoaded();
    helios::FluspectKernel kernel = helios::computeFluspectKernel(biochem, fluspect_optipar, excitation_step_nm);
    fluspect_cache_biochem[key] = kernel_biochem;
    auto [inserted_it, _] = fluspect_cache.emplace(key, std::move(kernel));
    return &inserted_it->second;
}

RadiationModel::ExcitationSet &RadiationModel::ensureExcitationSet(float bin_width_nm, uint scattering_depth) {
    const float key = fp_round5(bin_width_nm);
    auto it = excitation_sets.find(key);
    if (it != excitation_sets.end()) {
        // Set already exists. If this caller requests a deeper scattering depth than
        // the set's current value, upgrade the existing bands to that depth — any
        // camera bound to this set benefits from the more accurate APAR. Never
        // downgrade (another camera may have requested the larger value already).
        if (scattering_depth > it->second.scattering_depth) {
            it->second.scattering_depth = scattering_depth;
            for (const auto &bname : it->second.band_labels) {
                setScatteringDepth(bname, scattering_depth);
            }
            // Emission computed at the shallower depth is stale
            it->second.populated = false;
        }
        return it->second;
    }
    // Create new excitation set covering 400-750 nm at bin_width_nm resolution.
    ExcitationSet set;
    set.bin_width_nm = bin_width_nm;
    set.scattering_depth = scattering_depth;
    constexpr float ex_min = 400.f;
    constexpr float ex_max = 750.f;
    // Number of bins such that the last band ends exactly at ex_max (using a
    // bin width that may not divide the range evenly: the final bin is capped
    // at ex_max so all excitation within 400-750 is covered).
    const int n_bins = static_cast<int>(std::ceil((ex_max - ex_min) / bin_width_nm));
    set.band_labels.reserve(n_bins);
    set.band_min_nm.reserve(n_bins);
    set.band_max_nm.reserve(n_bins);
    for (int i = 0; i < n_bins; ++i) {
        float wmin = ex_min + i * bin_width_nm;
        float wmax = std::min(ex_max, wmin + bin_width_nm);
        std::ostringstream oss;
        oss << "_SIF_exc_" << bin_width_nm << "_" << wmin << "_" << wmax;
        const std::string label = oss.str();
        if (!doesBandExist(label)) {
            addRadiationBand(label, wmin, wmax);
            // Internal excitation bands are pure absorbers — disable emission.
            // Scattering depth follows the requesting SIF camera's
            // excitation_scattering_depth (default 0 for speed; users opt in to
            // scattering for more accurate APAR under high leaf rho/tau).
            disableEmission(label);
            setScatteringDepth(label, scattering_depth);
            // Auto-propagate band flux from any source that has a spectrum set.
            // Without this, the auto-generated excitation bands would have a
            // -1 sentinel flux (initialized in addRadiationBand) even if the user
            // had set a broadband spectrum on their source — so the excitation
            // ray trace would produce zero APAR.
            for (uint sid = 0; sid < radiation_sources.size(); ++sid) {
                const auto &src = radiation_sources.at(sid);
                if (!src.source_spectrum.empty()) {
                    const float band_flux = integrateSpectrum(src.source_spectrum, wmin, wmax);
                    setSourceFlux(sid, label, band_flux);
                }
            }
        }
        set.band_labels.push_back(label);
        set.band_min_nm.push_back(wmin);
        set.band_max_nm.push_back(wmax);
    }
    auto [inserted_it, _] = excitation_sets.emplace(key, std::move(set));
    return inserted_it->second;
}

namespace {
    //! Trapezoidal integral over [wavelength_min, wavelength_max] of a spectrum tabulated on the wavelength grid, linearly interpolated within grid intervals
    double integrateTabulatedSpectrum(const std::vector<float> &wavelengths, const std::vector<double> &spectrum, double wavelength_min, double wavelength_max) {
        double total = 0.0;
        for (size_t i = 0; i + 1 < wavelengths.size(); ++i) {
            const double w0 = wavelengths[i];
            const double w1 = wavelengths[i + 1];
            if (w1 < wavelength_min) {
                continue;
            }
            if (w0 > wavelength_max) {
                break;
            }
            const double lo = std::max(w0, wavelength_min);
            const double hi = std::min(w1, wavelength_max);
            if (hi <= lo) {
                continue;
            }
            const double spectrum_lo = spectrum[i] + (lo - w0) / (w1 - w0) * (spectrum[i + 1] - spectrum[i]);
            const double spectrum_hi = spectrum[i] + (hi - w0) / (w1 - w0) * (spectrum[i + 1] - spectrum[i]);
            total += 0.5 * (spectrum_lo + spectrum_hi) * (hi - lo);
        }
        return total;
    }
} // namespace

void RadiationModel::populateExcitationSet(ExcitationSet &exc, const std::vector<std::string> &launch_band_labels, const std::vector<float> &radiation_in, const std::vector<float> &radiation_in_top) {
    // After the excitation bands have been ray-traced, radiation_flux_<band> holds each primitive's absorbed flux. The Fluspect-B
    // matrices act on the excitation flux incident on each face of the leaf (leaf absorption is already inside them), so the incident
    // flux is recovered here:
    //  - Scattering depth 0: energy a primitive would scatter is retained in its radiation_flux (see runBand), so the reported
    //    flux already is the incident flux.
    //  - Scattering depth >= 1: the reported flux is absorbed, so it is divided by the absorptance 1 - rho - tau that the ray
    //    tracer applied to the primitive in that band. That includes the iterations after the last one traced, which runBand()
    //    sums as a geometric series and reports as absorption too.
    // The incident flux is split between the faces in proportion to the absorbed flux that arrived on each face (radiation_in_top
    // versus radiation_in). Retained scattered energy is split the same way; that is exact because rho and tau are properties of the
    // primitive, not of a face, so each face absorbs the same fraction of what arrives on it.
    //
    // Emission is linear in the fluorescence yield, which is not known until photosynthesis has run, so what is stored per leaf and
    // per emission band bound to this bin width is the yield-free band emission of each face; computeSIFEmission() scales it. The
    // internal radiation_flux_<band> primitive data is then cleared (internal bands shouldn't pollute the user namespace).
    exc.base_emission.clear();
    exc.biochem_label_versions.clear();
    exc.leaf_biochem_label.clear();
    std::unordered_map<std::string, uint> biochem_label_index;
    const std::vector<uint> all_UUIDs = context->getAllUUIDs();
    const size_t n_bands = exc.band_labels.size();
    const size_t Nbands_launch = launch_band_labels.size();

    // Launch slot of each excitation band in the radiation_in buffers
    std::vector<size_t> launch_slot(n_bands);
    for (size_t b = 0; b < n_bands; ++b) {
        auto slot_it = std::find(launch_band_labels.begin(), launch_band_labels.end(), exc.band_labels[b]);
        if (slot_it == launch_band_labels.end()) {
            helios_runtime_error("ERROR (RadiationModel::populateExcitationSet): Excitation band '" + exc.band_labels[b] + "' was not part of the launch. This is an internal inconsistency.");
        }
        launch_slot[b] = std::distance(launch_band_labels.begin(), slot_it);
    }

    // material_data is laid out [slot][primitive][band], with primitives in context_UUIDs order and bands in radiation_bands order. Slots 0..Nsources-1
    // hold each source's properties; the diffuse/scatter slot after them is not used here.
    const size_t Nsources = radiation_sources.size();
    const size_t Nprimitives = context_UUIDs.size();
    const size_t Nbands = radiation_bands.size();
    if (radiation_in.size() < Nprimitives * Nbands_launch || radiation_in_top.size() < Nprimitives * Nbands_launch) {
        helios_runtime_error("ERROR (RadiationModel::populateExcitationSet): The absorbed radiation read back from the ray tracer does not cover every primitive and launched band. This is an internal inconsistency.");
    }
    MaterialPropertyIndexer mat_idx(Nsources + 1, Nprimitives, Nbands);
    std::vector<size_t> material_band_index(n_bands);
    for (size_t b = 0; b < n_bands; ++b) {
        material_band_index[b] = std::distance(radiation_bands.begin(), radiation_bands.find(exc.band_labels[b]));
    }
    std::unordered_map<uint, size_t> primitive_index;
    for (size_t u = 0; u < Nprimitives; ++u) {
        primitive_index.emplace(context_UUIDs[u], u);
    }

    // Emission bands whose excitation resolution is this set's
    std::vector<std::pair<std::string, helios::vec2>> emission_bands;
    for (const auto &[band_label, bin_width]: sif_band_bin_width) {
        if (std::abs(bin_width - exc.bin_width_nm) < 1e-5f) {
            emission_bands.emplace_back(band_label, radiation_bands.at(band_label).wavebandBounds);
            exc.base_emission[band_label];
        }
    }

    const float step = exc.bin_width_nm;
    std::vector<float> incident_top(n_bands);
    std::vector<float> incident_bottom(n_bands);
    for (uint UUID: all_UUIDs) {
        // Only track primitives we care about: leaves with a fluspect_spectrum label.
        if (!context->doesPrimitiveDataExist(UUID, "fluspect_spectrum")) {
            continue;
        }
        // Record the biochemistry the emission is computed from, so that a later change to it is detected (isExcitationEmissionCurrent)
        std::string biochem_label;
        context->getPrimitiveData(UUID, "fluspect_spectrum", biochem_label);
        auto [label_it, new_label] = biochem_label_index.try_emplace(biochem_label, static_cast<uint>(exc.biochem_label_versions.size()));
        if (new_label) {
            exc.biochem_label_versions.emplace_back(biochem_label, context->getGlobalDataVersion(biochem_label.c_str()));
        }
        exc.leaf_biochem_label[UUID] = label_it->second;

        std::fill(incident_top.begin(), incident_top.end(), 0.f);
        std::fill(incident_bottom.begin(), incident_bottom.end(), 0.f);
        bool any_incident = false;
        for (size_t b = 0; b < n_bands; ++b) {
            const std::string prop = "radiation_flux_" + exc.band_labels[b];
            float absorbed = 0.f;
            if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                context->getPrimitiveData(UUID, prop.c_str(), absorbed);
            }
            if (absorbed <= 0.f) {
                continue;
            }
            auto index_it = primitive_index.find(UUID);
            if (index_it == primitive_index.end()) {
                helios_runtime_error("ERROR (RadiationModel::populateExcitationSet): Primitive " + std::to_string(UUID) + " received SIF excitation flux but is not in the ray-traced geometry. This is an internal inconsistency.");
            }
            const size_t u = index_it->second;
            float incident = absorbed;
            if (exc.scattering_depth > 0) {
                // Absorptance of this primitive in this band, weighted by each source's flux in the band (rho and tau are
                // source-spectrum weighted, so they can differ slightly between sources).
                float weighted_absorptance = 0.f;
                float weight_sum = 0.f;
                for (size_t source = 0; source < Nsources; ++source) {
                    const size_t ind = mat_idx(source, u, material_band_index[b]);
                    const float absorptance = 1.f - material_data.reflectivity[ind] - material_data.transmissivity[ind];
                    const float weight = getSourceFlux(source, exc.band_labels[b]);
                    weighted_absorptance += weight * absorptance;
                    weight_sum += weight;
                }
                if (weight_sum <= 0.f || weighted_absorptance <= 0.f) {
                    helios_runtime_error("ERROR (RadiationModel::populateExcitationSet): Leaf primitive " + std::to_string(UUID) + " absorbed SIF excitation flux in band '" + exc.band_labels[b] +
                                         "', but its absorptance (1 - reflectivity - transmissivity) in that band is not positive, so the incident excitation flux that drives the Fluspect-B "
                                         "fluorescence matrices cannot be recovered. Check the reflectivity_spectrum and transmissivity_spectrum assigned to leaves with 'fluspect_spectrum' data.");
                }
                incident = absorbed * weight_sum / weighted_absorptance;
            }

            const size_t launch_index = u * Nbands_launch + launch_slot[b];
            const float absorbed_all_faces = radiation_in[launch_index];
            if (absorbed_all_faces <= 0.f) {
                helios_runtime_error("ERROR (RadiationModel::populateExcitationSet): Leaf primitive " + std::to_string(UUID) + " received SIF excitation flux in band '" + exc.band_labels[b] +
                                     "' but absorbed none of it, so the face it arrived on cannot be determined. Its absorptance (1 - reflectivity - transmissivity) in that band must be positive; "
                                     "check the reflectivity_spectrum and transmissivity_spectrum assigned to leaves with 'fluspect_spectrum' data.");
            }
            const float top_fraction = std::clamp(radiation_in_top[launch_index] / absorbed_all_faces, 0.f, 1.f);
            incident_top[b] = incident * top_fraction;
            incident_bottom[b] = incident * (1.f - top_fraction);
            any_incident = true;
        }

        if (!any_incident || emission_bands.empty()) {
            for (const auto &emission_band: emission_bands) {
                exc.base_emission[emission_band.first][UUID] = helios::make_vec2(0.f, 0.f);
            }
            continue;
        }

        // Apply the Fluspect-B matrices to the incident excitation flux of each excitation band (as SCOPE's RTMf does), producing
        // per-emission-wavelength spectra on the kernel's 4 nm wlf grid for each face. The backward matrix Mb gives the emission from
        // the face that received the excitation and the forward matrix Mf the emission from the opposite face, so
        // F_top = Mb * I_top + Mf * I_bottom and F_bottom = Mf * I_top + Mb * I_bottom.
        //  - Each kernel column carries the excitation grid spacing as a quadrature weight (FluspectB.h), so dividing by the
        //    spacing gives the emission per nm of excitation, which multiplies the band-integrated incident flux.
        //  - The kernel is evaluated at the band centre by linear interpolation between the neighbouring wle columns. A final band
        //    narrower than the bin width can have its centre beyond the last column; it uses the last column.
        //  - The matrices convert photons to photons (fqe is a photon yield), so energy flux is scaled by lambda_e / lambda_f.
        const helios::FluspectKernel *kernel = getOrComputeFluspectKernel(UUID, step);
        const auto &wle = kernel->wle;
        const auto &wlf = kernel->wlf;
        std::vector<double> F_top(wlf.size(), 0.0);
        std::vector<double> F_bottom(wlf.size(), 0.0);
        for (size_t b = 0; b < n_bands; ++b) {
            if (incident_top[b] == 0.f && incident_bottom[b] == 0.f) {
                continue;
            }
            const double band_center = 0.5 * (exc.band_min_nm[b] + exc.band_max_nm[b]);
            const double column_position = std::clamp((band_center - wle.front()) / step, 0.0, double(wle.size() - 1));
            const size_t j0 = std::min(static_cast<size_t>(column_position), wle.size() - 2);
            const double w1 = column_position - double(j0);
            const double w0 = 1.0 - w1;
            for (size_t i = 0; i < wlf.size(); ++i) {
                const double scale = band_center / (step * wlf[i]);
                const double Mf = w0 * kernel->Mf[i][j0] + w1 * kernel->Mf[i][j0 + 1];
                const double Mb = w0 * kernel->Mb[i][j0] + w1 * kernel->Mb[i][j0 + 1];
                F_top[i] += scale * (Mb * incident_top[b] + Mf * incident_bottom[b]);
                F_bottom[i] += scale * (Mf * incident_top[b] + Mb * incident_bottom[b]);
            }
        }
        for (const auto &[band_label, bounds]: emission_bands) {
            const double emission_top = integrateTabulatedSpectrum(wlf, F_top, bounds.x, bounds.y);
            const double emission_bottom = integrateTabulatedSpectrum(wlf, F_bottom, bounds.x, bounds.y);
            exc.base_emission[band_label][UUID] = helios::make_vec2(static_cast<float>(emission_top), static_cast<float>(emission_bottom));
        }
    }
    // Clear primitive data for internal excitation bands from ALL primitives.
    for (uint UUID: all_UUIDs) {
        for (const auto &band_label: exc.band_labels) {
            const std::string prop = "radiation_flux_" + band_label;
            if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                context->clearPrimitiveData(UUID, prop.c_str());
            }
        }
    }
    exc.populated = true;
}

void RadiationModel::runExcitationBands() {
    // Run any excitation set that isn't already populated. All internal bands for a set are launched together via runBand(vector),
    // which fills the set's per-leaf emission at the end of the launch (populateExcitationSet).
    //
    // The per-band "scattering disabled" warning from runBand() is suppressed for
    // "_SIF_exc_*" labels. Users who want more accurate APAR under high leaf
    // rho/tau should set SIFCameraProperties::excitation_scattering_depth >= 1.
    for (auto &kv : excitation_sets) {
        ExcitationSet &exc = kv.second;
        if (exc.populated) {
            continue;
        }
        runBand(exc.band_labels);
        if (!exc.populated) {
            helios_runtime_error("ERROR (RadiationModel::runExcitationBands): The SIF excitation bands at " + std::to_string(exc.bin_width_nm) + " nm were run but their per-leaf emission was not computed. This is an internal inconsistency.");
        }
    }
}

bool RadiationModel::isExcitationEmissionCurrent(const ExcitationSet &exc, const std::string &emission_band) const {
    // The per-leaf emission is computed when the excitation bands are run, for the emission bands, leaves and leaf biochemistry that existed
    // then. An emission band bound later (a new SIF camera), a leaf given biochemistry later, a leaf moved to another biochemistry label, or a
    // label whose values were rewritten (LeafOptics::run() reuses its nitrogen-bin labels) makes it stale, so the bands must be run again.
    auto band_it = exc.base_emission.find(emission_band);
    if (band_it == exc.base_emission.end()) {
        return false;
    }
    for (const auto &[biochem_label, version]: exc.biochem_label_versions) {
        if (context->getGlobalDataVersion(biochem_label.c_str()) != version) {
            return false;
        }
    }
    std::string biochem_label;
    for (uint UUID: context->getAllUUIDs()) {
        if (!context->doesPrimitiveDataExist(UUID, "fluspect_spectrum")) {
            continue;
        }
        auto leaf_it = exc.leaf_biochem_label.find(UUID);
        if (leaf_it == exc.leaf_biochem_label.end() || band_it->second.find(UUID) == band_it->second.end()) {
            return false;
        }
        context->getPrimitiveData(UUID, "fluspect_spectrum", biochem_label);
        if (biochem_label != exc.biochem_label_versions.at(leaf_it->second).first) {
            return false;
        }
    }
    return true;
}

void RadiationModel::computeSIFEmission(const std::string &emission_band) {
    // Populate sif_emission_buffer[emission_band] (top face) and sif_emission_buffer_bottom[emission_band] (bottom face) for every leaf
    // primitive, scaling the yield-free face emission computed from the excitation bands by the van der Tol (2014) quantum yield.

    if (radiation_bands.find(emission_band) == radiation_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): band '" + emission_band + "' does not exist.");
    }

    // Make sure excitation bands have been run this dispatch.
    runExcitationBands();

    auto &buf_top = sif_emission_buffer[emission_band];
    auto &buf_bot = sif_emission_buffer_bottom[emission_band];
    buf_top.clear();
    buf_bot.clear();

    // Look up the authoritative excitation bin width for this emission band.
    // addSIFCamera guarantees sif_band_bin_width[emission_band] exists for every band
    // in sif_emission_bands (otherwise how did we get here).
    auto bw_it = sif_band_bin_width.find(emission_band);
    if (bw_it == sif_band_bin_width.end()) {
        helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): band '" + emission_band + "' is flagged as SIF but has no excitation bin width registered. This is an internal inconsistency.");
    }
    const float step = bw_it->second;
    // Find the excitation set that matches this bin width. ensureExcitationSet
    // is called from addSIFCamera for every camera's bin width, so the set
    // must exist.
    ExcitationSet *matched = nullptr;
    for (auto &kv : excitation_sets) {
        if (std::abs(kv.second.bin_width_nm - step) < 1e-5f) {
            matched = &kv.second;
            break;
        }
    }
    if (!matched) {
        helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): no excitation set found for bin width " + std::to_string(step) + " nm (band '" + emission_band + "').");
    }
    if (!isExcitationEmissionCurrent(*matched, emission_band)) {
        matched->populated = false;
        runExcitationBands();
        if (!isExcitationEmissionCurrent(*matched, emission_band)) {
            helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): The SIF excitation bands were run again for band '" + emission_band + "', but it still has no per-leaf emission. This is an internal inconsistency.");
        }
    }
    const auto &base_emission = matched->base_emission.at(emission_band);

    const std::vector<uint> all_UUIDs = context->getAllUUIDs();
    size_t n_applied = 0;
    size_t n_skipped_no_biochem_has_etr = 0;  // primitives with electron_transport_ratio but no biochemistry label
    size_t n_skipped_has_biochem_no_etr = 0;  // leaves with biochemistry but no electron_transport_ratio
    for (uint UUID : all_UUIDs) {
        const bool has_biochem = context->doesPrimitiveDataExist(UUID, "fluspect_spectrum");
        const bool has_etr = context->doesPrimitiveDataExist(UUID, "electron_transport_ratio");
        if (!has_etr) {
            if (has_biochem) ++n_skipped_has_biochem_no_etr;
            continue;
        }
        if (!has_biochem) {
            ++n_skipped_no_biochem_has_etr;
            continue;
        }

        float electron_transport_ratio = 0.f;
        context->getPrimitiveData(UUID, "electron_transport_ratio", electron_transport_ratio);
        float T_leaf_K = temperature_default;
        if (context->doesPrimitiveDataExist(UUID, "temperature")) {
            context->getPrimitiveData(UUID, "temperature", T_leaf_K);
            if (T_leaf_K <= 0.f) {
                helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): Leaf primitive " + std::to_string(UUID) + " has 'temperature' primitive data of " + std::to_string(T_leaf_K) +
                                     " K. The fluorescence yield needs a leaf temperature in Kelvin greater than zero.");
            }
        }
        const float Phi_F = calculateFluorescenceYield(electron_transport_ratio, T_leaf_K);
        context->setPrimitiveData(UUID, "fluorescence_yield", Phi_F);

        // Retrieve the user's fqe calibration scalar from the global-data biochemistry
        // vector (11th field). The kernel was computed with fqe=1, so the authoritative
        // fqe multiplies Phi_F here to produce the final emission scaling.
        float fqe = 1.f;
        {
            std::string biochem_label;
            context->getPrimitiveData(UUID, "fluspect_spectrum", biochem_label);
            if (context->doesGlobalDataExist(biochem_label.c_str())) {
                std::vector<float> biochem_vec;
                context->getGlobalData(biochem_label.c_str(), biochem_vec);
                if (biochem_vec.size() >= 11) fqe = biochem_vec[10];
            }
        }
        const float phi_F_scaled = Phi_F * fqe;

        const helios::vec2 &face_emission = base_emission.at(UUID);
        buf_top[UUID] = phi_F_scaled * face_emission.x;
        buf_bot[UUID] = phi_F_scaled * face_emission.y;
        ++n_applied;
    }

    // A leaf with biochemistry is one the user asked to fluoresce; without electron_transport_ratio its fluorescence yield is
    // unknown, and skipping it would leave it silently dark in the SIF image.
    if (n_skipped_has_biochem_no_etr > 0) {
        helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): " + std::to_string(n_skipped_has_biochem_no_etr) + " primitives have leaf biochemistry ('fluspect_spectrum') but no 'electron_transport_ratio' primitive data, so their fluorescence yield cannot be computed for SIF band '" +
                             emission_band + "'. Run PhotosynthesisModel::run() with optionalOutputPrimitiveData(\"electron_transport_ratio\") before runBand(), using the Farquhar or C4 model (the empirical model has no electron transport). " +
                             "Only assign leaf biochemistry to primitives that should fluoresce.");
    }
    if (n_applied == 0 && message_flag) {
        std::cerr << "WARNING (RadiationModel::computeSIFEmission): SIF-flagged band '" << emission_band << "' will emit zero - no primitive has leaf biochemistry ('fluspect_spectrum')";
        if (n_skipped_no_biochem_has_etr > 0) {
            std::cerr << " (" << n_skipped_no_biochem_has_etr << " primitives have 'electron_transport_ratio' but no biochemistry)";
        }
        std::cerr << ". Call LeafOptics::run() to author leaf biochemistry for the leaves that should fluoresce." << std::endl;
    }
}

// --- SIF camera public API ---

void RadiationModel::addSIFCamera(const std::string &camera_label, const std::vector<std::string> &emission_band_labels, const vec3 &position, const vec3 &lookat, const SIFCameraProperties &camera_properties, uint antialiasing_samples) {

    if (emission_band_labels.empty()) {
        helios_runtime_error("ERROR (RadiationModel::addSIFCamera): emission_band_labels cannot be empty.");
    }
    for (const auto &band : emission_band_labels) {
        if (!doesBandExist(band)) {
            helios_runtime_error("ERROR (RadiationModel::addSIFCamera): band '" + band + "' does not exist. Add it with addRadiationBand() before calling addSIFCamera().");
        }
        // Enable emission on the band if not already — Fluspect sourced via emission loop.
        enableEmission(band);
        // Flag as SIF-sourcing: computeSIFEmission will populate sif_emission_buffer for it.
        sif_emission_bands.insert(band);
        // Bind this band to the camera's excitation bin width. If the band was already
        // bound to a different bin width by a prior camera, that's a user error — each
        // emission band must have a single authoritative excitation resolution so that
        // sif_emission_buffer[band] holds one consistent source flux per leaf.
        auto bw_it = sif_band_bin_width.find(band);
        if (bw_it == sif_band_bin_width.end()) {
            sif_band_bin_width[band] = camera_properties.excitation_bin_width_nm;
        } else if (std::abs(bw_it->second - camera_properties.excitation_bin_width_nm) > 1e-5f) {
            helios_runtime_error("ERROR (RadiationModel::addSIFCamera): emission band '" + band + "' is already bound to excitation_bin_width_nm=" + std::to_string(bw_it->second) +
                                 " by a prior SIF camera, but this camera's excitation_bin_width_nm=" + std::to_string(camera_properties.excitation_bin_width_nm) +
                                 ". Each SIF emission band can be bound to only one excitation resolution. "
                                 "Either use a separate band per camera or match excitation_bin_width_nm.");
        }
    }
    if (camera_properties.excitation_bin_width_nm <= 0.f) {
        helios_runtime_error("ERROR (RadiationModel::addSIFCamera): excitation_bin_width_nm must be > 0.");
    }
    ensureExcitationSet(camera_properties.excitation_bin_width_nm, camera_properties.excitation_scattering_depth);

    // --- Diagnostic scan: flag likely-silent SIF setup mistakes at camera-addition time ---
    // This catches the common pitfalls before the user invests compute in a bad run:
    //   - No radiation source has a spectrum set (excitation bands will pick up zero flux).
    //   - Leaves are present but none have Cab (none will fluoresce).
    //   - Leaves have biochemistry but lack electron_transport_ratio (checked at runBand() time, where it is an error).
    if (message_flag) {
        // (a) Source spectrum check: at least one source must have a spectrum covering the
        //     Fluspect-B excitation range (400-750 nm). We only flag the missing-spectrum
        //     case — not the partial-coverage case (that's the user's prerogative).
        bool any_source_has_spectrum = false;
        for (const auto &src : radiation_sources) {
            if (!src.source_spectrum.empty()) {
                any_source_has_spectrum = true;
                break;
            }
        }
        if (!any_source_has_spectrum) {
            std::cerr << "WARNING (RadiationModel::addSIFCamera): Camera '" << camera_label
                      << "' added, but no radiation source has a spectrum set. Auto-generated excitation "
                         "bands will receive zero flux, so SIF emission from all leaves will be zero. "
                         "Call setSourceSpectrum(source_ID, \"solar_spectrum_direct_ASTMG173\") (or similar) "
                         "on at least one source before runBand()."
                      << std::endl;
        }

        // (b) Leaf biochemistry coverage. We only look at primitives present at camera-add
        //     time; users may add more leaves later. 'electron_transport_ratio' is NOT
        //     checked here because it is normally populated later by
        //     PhotosynthesisModel::run() — a setup-time check would fire for every correctly-
        //     sequenced pipeline (addSIFCamera → photomodel.run() → runBand). The equivalent
        //     runtime check in computeSIFEmission() errors if it is still missing at dispatch.
        const auto all_UUIDs = context->getAllUUIDs();
        const size_t n_prims = all_UUIDs.size();
        size_t n_with_biochem = 0;
        for (uint UUID : all_UUIDs) {
            if (context->doesPrimitiveDataExist(UUID, "fluspect_spectrum")) {
                ++n_with_biochem;
            }
        }
        if (n_prims > 0 && n_with_biochem == 0) {
            std::cerr << "WARNING (RadiationModel::addSIFCamera): Camera '" << camera_label
                      << "' added to a scene with " << n_prims << " primitives, but none have "
                         "'fluspect_spectrum' primitive data. Helios cannot identify fluorescing leaves "
                         "without a biochemistry label. Call LeafOptics::run(UUIDs, properties, \"my_label\") "
                         "to author leaf biochemistry, or manually "
                         "setGlobalData(\"fluspect_biochem_<label>\", std::vector<float>{Cab, Cca, Cw, Cdm, "
                         "Cs, Cant, Cp, Cbc, N, V2Z, fqe}) and setPrimitiveData(UUIDs, \"fluspect_spectrum\", "
                         "\"fluspect_biochem_<label>\")."
                      << std::endl;
        }
    }

    // Delegate to addRadiationCamera for the geometric/pixel setup.
    addRadiationCamera(camera_label, emission_band_labels, position, lookat, camera_properties, antialiasing_samples);

    // Tag the camera as a SIF camera via its label in a private set.
    sif_cameras.insert(camera_label);
}

void RadiationModel::addSIFCamera(const std::string &camera_label, const std::vector<std::string> &emission_band_labels, const vec3 &position, const SphericalCoord &viewing_direction,
                                   const SIFCameraProperties &camera_properties, uint antialiasing_samples) {
    // Convert spherical direction to lookat point.
    const vec3 dir = sphere2cart(viewing_direction);
    addSIFCamera(camera_label, emission_band_labels, position, position + dir, camera_properties, antialiasing_samples);
}

bool RadiationModel::isSIFCamera(const std::string &camera_label) const {
    return sif_cameras.find(camera_label) != sif_cameras.end();
}

float RadiationModel::calculateFluorescenceYield(float electron_transport_ratio, float T_leaf_K) {
    // Van der Tol et al. (2014) rate-coefficient model of steady-state fluorescence (also SCOPE's biochemical.m).
    // All rate coefficients are relative to K_F + K_D = 1 at 25 C, so K_N is the PAM-fluorometry NPQ.
    constexpr float kF = 0.05f; // fluorescence
    constexpr float kP = 4.f; // photochemistry with all PSII reaction centres open
    // Thermal dissipation, constant up to ~26 C and rising linearly above it (Section 5 of the paper).
    const float kD = std::max(0.03f * (T_leaf_K - 273.15f) + 0.0773f, 0.87f);

    // Maximum (dark-adapted, K_N = 0) photochemical yield, and the steady-state yield it is reduced to by feedback from carbon metabolism (Eq. 16).
    const float Phi_P0 = kP / (kF + kD + kP);
    const float relative_light_saturation = helios::clamp(electron_transport_ratio, 0.f, 1.f);
    const float Phi_P = Phi_P0 * relative_light_saturation;

    // Non-photochemical quenching as a function of the degree of light saturation x (Eqs. 14 and 19), using the fit to the unstressed cotton data
    // (K_N0 = 2.48, alpha = 2.83, beta = 0.114), which is also SCOPE's default.
    constexpr float kN0 = 2.48f;
    constexpr float kN_alpha = 2.83f;
    constexpr float kN_beta = 0.114f;
    const float x = 1.f - relative_light_saturation;
    const float x_alpha = std::pow(x, kN_alpha);
    const float kN = kN0 * (1.f + kN_beta) * x_alpha / (kN_beta + x_alpha);

    // Steady-state fluorescence yield: the light-adapted maximum yield Fm' reduced by the fraction of excitons used in photochemistry (Eq. 17).
    return kF / (kF + kD + kN) * (1.f - Phi_P);
}

uint RadiationModel::addCollimatedRadiationSource() {

    return addCollimatedRadiationSource(make_vec3(0, 0, 1));
}

uint RadiationModel::addCollimatedRadiationSource(const SphericalCoord &direction) {
    return addCollimatedRadiationSource(sphere2cart(direction));
}

uint RadiationModel::addCollimatedRadiationSource(const vec3 &direction) {

    if (direction.magnitude() == 0) {
        helios_runtime_error("ERROR (RadiationModel::addCollimatedRadiationSource): Invalid collimated source direction. Direction vector should not have length of zero.");
    }

    uint Nsources = radiation_sources.size() + 1;
    if (Nsources > 255) { // source IDs are stored in 8 bits, and index Nsources is the diffuse/scatter material slot
        helios_runtime_error("ERROR (RadiationModel::addCollimatedRadiationSource): A maximum of 255 radiation sources are allowed.");
    }

    bool warn_multiple_suns = false;
    for (auto &source: radiation_sources) {
        if (source.source_type == RADIATION_SOURCE_TYPE_COLLIMATED || source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
            warn_multiple_suns = true;
        }
    }
    if (warn_multiple_suns) {
        std::cerr << "WARNING (RadiationModel::addCollimatedRadiationSource): Multiple sun sources have been added to the radiation model. This may lead to unintended behavior." << std::endl;
    }

    RadiationSource collimated_source(direction);

    // initialize fluxes
    for (const auto &band: radiation_bands) {
        collimated_source.source_fluxes[band.first] = -1.f;
    }

    radiation_sources.emplace_back(collimated_source);

    radiativepropertiesneedupdate = true;

    return Nsources - 1;
}

uint RadiationModel::addSphereRadiationSource(const vec3 &position, float radius) {

    if (radius <= 0) {
        helios_runtime_error("ERROR (RadiationModel::addSphereRadiationSource): Spherical radiation source radius must be positive.");
    }

    uint Nsources = radiation_sources.size() + 1;
    if (Nsources > 255) { // source IDs are stored in 8 bits, and index Nsources is the diffuse/scatter material slot
        helios_runtime_error("ERROR (RadiationModel::addSphereRadiationSource): A maximum of 255 radiation sources are allowed.");
    }

    RadiationSource sphere_source(position, 2.f * fabsf(radius));

    // initialize fluxes
    for (const auto &band: radiation_bands) {
        sphere_source.source_fluxes[band.first] = -1.f;
    }

    radiation_sources.emplace_back(sphere_source);

    uint sourceID = Nsources - 1;

    if (islightvisualizationenabled) {
        buildLightModelGeometry(sourceID);
    }

    radiativepropertiesneedupdate = true;

    return sourceID;
}

uint RadiationModel::addSunSphereRadiationSource() {
    return addSunSphereRadiationSource(make_vec3(0, 0, 1));
}

uint RadiationModel::addSunSphereRadiationSource(const SphericalCoord &sun_direction) {
    return addSunSphereRadiationSource(sphere2cart(sun_direction));
}

uint RadiationModel::addSunSphereRadiationSource(const vec3 &sun_direction) {

    uint Nsources = radiation_sources.size() + 1;
    if (Nsources > 255) { // source IDs are stored in 8 bits, and index Nsources is the diffuse/scatter material slot
        helios_runtime_error("ERROR (RadiationModel::addSunSphereRadiationSource): A maximum of 255 radiation sources are allowed.");
    }

    bool warn_multiple_suns = false;
    for (auto &source: radiation_sources) {
        if (source.source_type == RADIATION_SOURCE_TYPE_COLLIMATED || source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
            warn_multiple_suns = true;
        }
    }
    if (warn_multiple_suns) {
        std::cerr << "WARNING (RadiationModel::addSunSphereRadiationSource): Multiple sun sources have been added to the radiation model. This may lead to unintended behavior." << std::endl;
    }

    RadiationSource sphere_source(150e9 * sun_direction / sun_direction.magnitude(), 150e9, 2.f * 695.5e6, sigma * powf(5700, 4) / 1288.437f);

    // initialize fluxes
    for (const auto &band: radiation_bands) {
        sphere_source.source_fluxes[band.first] = -1.f;
    }

    radiation_sources.emplace_back(sphere_source);

    radiativepropertiesneedupdate = true;

    return Nsources - 1;
}

uint RadiationModel::addRectangleRadiationSource(const vec3 &position, const vec2 &size, const vec3 &rotation_rad) {

    if (size.x <= 0 || size.y <= 0) {
        helios_runtime_error("ERROR (RadiationModel::addRectangleRadiationSource): Radiation source size must be positive.");
    }

    uint Nsources = radiation_sources.size() + 1;
    if (Nsources > 255) { // source IDs are stored in 8 bits, and index Nsources is the diffuse/scatter material slot
        helios_runtime_error("ERROR (RadiationModel::addRectangleRadiationSource): A maximum of 255 radiation sources are allowed.");
    }

    RadiationSource rectangle_source(position, size, rotation_rad);

    // initialize fluxes
    for (const auto &band: radiation_bands) {
        rectangle_source.source_fluxes[band.first] = -1.f;
    }

    radiation_sources.emplace_back(rectangle_source);

    uint sourceID = Nsources - 1;

    if (islightvisualizationenabled) {
        buildLightModelGeometry(sourceID);
    }

    radiativepropertiesneedupdate = true;

    return sourceID;
}

uint RadiationModel::addDiskRadiationSource(const vec3 &position, float radius, const vec3 &rotation_rad) {

    if (radius <= 0) {
        helios_runtime_error("ERROR (RadiationModel::addDiskRadiationSource): Disk radiation source radius must be positive.");
    }

    uint Nsources = radiation_sources.size() + 1;
    if (Nsources > 255) { // source IDs are stored in 8 bits, and index Nsources is the diffuse/scatter material slot
        helios_runtime_error("ERROR (RadiationModel::addDiskRadiationSource): A maximum of 255 radiation sources are allowed.");
    }

    RadiationSource disk_source(position, radius, rotation_rad);

    // initialize fluxes
    for (const auto &band: radiation_bands) {
        disk_source.source_fluxes[band.first] = -1.f;
    }

    radiation_sources.emplace_back(disk_source);

    uint sourceID = Nsources - 1;

    if (islightvisualizationenabled) {
        buildLightModelGeometry(sourceID);
    }

    radiativepropertiesneedupdate = true;

    return sourceID;
}

void RadiationModel::deleteRadiationSource(uint sourceID) {

    if (sourceID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::deleteRadiationSource): Source ID out of bounds. Only " + std::to_string(radiation_sources.size() - 1) + " radiation sources have been created.");
    }

    radiation_sources.erase(radiation_sources.begin() + sourceID);

    radiativepropertiesneedupdate = true;
}

void RadiationModel::setSourceSpectrumIntegral(uint source_ID, float source_integral) {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrumIntegral): Source ID out of bounds. Only " + std::to_string(radiation_sources.size() - 1) + " radiation sources have been created.");
    } else if (source_integral < 0) {
        helios_runtime_error("ERROR (RadiationModel::setSourceIntegral): Source integral must be non-negative.");
    }

    float current_integral = integrateSpectrum(radiation_sources.at(source_ID).source_spectrum);

    for (vec2 &wavelength: radiation_sources.at(source_ID).source_spectrum) {
        wavelength.y *= source_integral / current_integral;
    }
}

void RadiationModel::setSourceSpectrumIntegral(uint source_ID, float source_integral, float wavelength1, float wavelength2) {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrumIntegral): Source ID out of bounds. Only " + std::to_string(radiation_sources.size() - 1) + " radiation sources have been created.");
    } else if (source_integral < 0) {
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrumIntegral): Source integral must be non-negative.");
    } else if (radiation_sources.at(source_ID).source_spectrum.empty()) {
        std::cout << "WARNING (RadiationModel::setSourceSpectrumIntegral): Spectral distribution has not been set for radiation source. Cannot set its integral." << std::endl;
        return;
    }

    RadiationSource &source = radiation_sources.at(source_ID);

    float old_integral = integrateSpectrum(source.source_spectrum, wavelength1, wavelength2);

    for (vec2 &wavelength: source.source_spectrum) {
        wavelength.y *= source_integral / old_integral;
    }
}

void RadiationModel::setSourceFlux(uint source_ID, const std::string &label, float flux) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::setSourceFlux): Cannot add set source flux for band '" + label + "' because it is not a valid band.");
    } else if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::setSourceFlux): Source ID out of bounds. Only " + std::to_string(radiation_sources.size() - 1) + " radiation sources have been created.");
    } else if (flux < 0) {
        helios_runtime_error("ERROR (RadiationModel::setSourceFlux): Source flux must be non-negative.");
    }

    radiation_sources.at(source_ID).source_fluxes[label] = flux * radiation_sources.at(source_ID).source_flux_scaling_factor;
}

void RadiationModel::setSourceFlux(const std::vector<uint> &source_ID, const std::string &band_label, float flux) {
    for (auto ID: source_ID) {
        setSourceFlux(ID, band_label, flux);
    }
}

float RadiationModel::getSourceFlux(uint source_ID, const std::string &label) const {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::getSourceFlux): Cannot get source flux for band '" + label + "' because it is not a valid band.");
    } else if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::getSourceFlux): Source ID out of bounds. Only " + std::to_string(radiation_sources.size() - 1) + " radiation sources have been created.");
    } else if (radiation_sources.at(source_ID).source_fluxes.find(label) == radiation_sources.at(source_ID).source_fluxes.end()) {
        helios_runtime_error("ERROR (RadiationModel::getSourceFlux): Cannot get flux for source #" + std::to_string(source_ID) + " because radiative band '" + label + "' does not exist.");
    }

    const RadiationSource &source = radiation_sources.at(source_ID);

    if (!source.source_spectrum.empty() && source.source_fluxes.at(label) < 0.f) { // source spectrum was specified (and not overridden by setting source flux manually)
        vec2 wavebounds = radiation_bands.at(label).wavebandBounds;
        if (!radiation_bands.at(label).hasWavebandBounds()) {
            wavebounds = make_vec2(source.source_spectrum.front().x, source.source_spectrum.back().x);
        }
        return integrateSpectrum(source.source_spectrum, wavebounds.x, wavebounds.y) * source.source_flux_scaling_factor;
    } else if (source.source_fluxes.at(label) < 0.f) {
        return 0;
    }

    return source.source_fluxes.at(label);
}

void RadiationModel::setSourceSpectrum(uint source_ID, const std::vector<helios::vec2> &spectrum) {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Cannot add radiation spectra for this source because it is not a valid radiation source ID.\n");
    }

    // validate spectrum
    for (auto s = 0; s < spectrum.size(); s++) {
        // check that wavelengths are monotonic
        if (s > 0 && spectrum.at(s).x <= spectrum.at(s - 1).x) {
            helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Source spectral data validation failed. Wavelengths must increase monotonically.");
        }
        // check that wavelength is within a reasonable range
        if (spectrum.at(s).x < 0 || spectrum.at(s).x > 100000) {
            helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Source spectral data validation failed. Wavelength value of " + std::to_string(spectrum.at(s).x) + " appears to be erroneous.");
        }
        // check that flux is non-negative
        if (spectrum.at(s).y < 0) {
            helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Source spectral data validation failed. Flux value at wavelength of " + std::to_string(spectrum.at(s).x) + " appears is negative.");
        }
    }

    radiation_sources.at(source_ID).source_spectrum = spectrum;

    radiativepropertiesneedupdate = true;
}

void RadiationModel::setSourceSpectrum(const std::vector<uint> &source_ID, const std::vector<helios::vec2> &spectrum) {
    for (auto ID: source_ID) {
        setSourceSpectrum(ID, spectrum);
    }
}

void RadiationModel::setSourceSpectrum(uint source_ID, const std::string &spectrum_label) {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Cannot add radiation spectra for this source because it is not a valid radiation source ID.\n");
    }

    std::vector<vec2> spectrum = loadSpectralData(spectrum_label);

    radiation_sources.at(source_ID).source_spectrum = spectrum;
    radiation_sources.at(source_ID).source_spectrum_label = spectrum_label;
    radiation_sources.at(source_ID).source_spectrum_version = context->getGlobalDataVersion(spectrum_label.c_str());

    radiativepropertiesneedupdate = true;
}

void RadiationModel::setSourceSpectrum(const std::vector<uint> &source_ID, const std::string &spectrum_label) {
    for (auto ID: source_ID) {
        setSourceSpectrum(ID, spectrum_label);
    }
}

void RadiationModel::setDiffuseSpectrum(const std::string &spectrum_label) {

    std::vector<vec2> spectrum;

    // standard solar spectrum
    if (spectrum_label == "ASTMG173") {
        spectrum = loadSpectralData("solar_spectrum_diffuse_ASTMG173");
        global_diffuse_spectrum_label = "solar_spectrum_diffuse_ASTMG173";
    } else {
        spectrum = loadSpectralData(spectrum_label);
        global_diffuse_spectrum_label = spectrum_label;
    }

    // Store globally so new bands will also get this spectrum
    global_diffuse_spectrum = spectrum;
    global_diffuse_spectrum_version = context->getGlobalDataVersion(global_diffuse_spectrum_label.c_str());

    // Apply to all existing bands
    for (auto &band_pair: radiation_bands) {
        band_pair.second.diffuse_spectrum = spectrum;
    }

    radiativepropertiesneedupdate = true;
}

float RadiationModel::getDiffuseFlux(const std::string &band_label) const {

    if (!doesBandExist(band_label)) {
        helios_runtime_error("ERROR (RadiationModel::getDiffuseFlux): Cannot get diffuse flux for band '" + band_label + "' because it is not a valid band.");
    }

    const RadiationBand &band = radiation_bands.at(band_label);

    // For emission-enabled bands: spectra are not relevant, only use manual flux
    if (band.emissionFlag) {
        if (band.diffuseFlux >= 0.f) {
            return band.diffuseFlux;
        }
        return 0.f;
    }

    // For non-emission bands: check manual flux first, then spectrum
    if (band.diffuseFlux >= 0.f) {
        return band.diffuseFlux;
    }

    const std::vector<vec2> &spectrum = band.diffuse_spectrum;
    if (!spectrum.empty()) {
        vec2 wavebounds = band.wavebandBounds;
        if (!band.hasWavebandBounds()) {
            wavebounds = make_vec2(spectrum.front().x, spectrum.back().x);
        }
        return integrateSpectrum(spectrum, wavebounds.x, wavebounds.y);
    }

    return 0.f;
}

namespace {
    // Delete the model primitives that still exist. The user may have deleted some of them through the Context already.
    void deleteModelPrimitives(helios::Context *context, const std::vector<uint> &UUIDs) {
        std::vector<uint> existing;
        existing.reserve(UUIDs.size());
        for (uint UUID: UUIDs) {
            if (context->doesPrimitiveExist(UUID)) {
                existing.push_back(UUID);
            }
        }
        context->deletePrimitive(existing);
    }
} // namespace

void RadiationModel::enableLightModelVisualization() {
    // rebuild from scratch so that calling this twice does not leave a duplicate set of models behind
    for (auto &UUIDs: source_model_UUIDs) {
        deleteModelPrimitives(context, UUIDs.second);
    }
    source_model_UUIDs.clear();

    islightvisualizationenabled = true;

    // build the geometry of any existing sources at this point
    for (int s = 0; s < radiation_sources.size(); s++) {
        buildLightModelGeometry(s);
    }
}

void RadiationModel::disableLightModelVisualization() {
    islightvisualizationenabled = false;
    for (auto &UUIDs: source_model_UUIDs) {
        deleteModelPrimitives(context, UUIDs.second);
    }
    source_model_UUIDs.clear();
}

void RadiationModel::enableCameraModelVisualization() {
    for (auto &UUIDs: camera_model_UUIDs) {
        deleteModelPrimitives(context, UUIDs.second);
    }
    camera_model_UUIDs.clear();

    iscameravisualizationenabled = true;

    // build the geometry of any existing cameras at this point
    for (auto &cam: cameras) {
        buildCameraModelGeometry(cam.first);
    }
}

void RadiationModel::disableCameraModelVisualization() {
    iscameravisualizationenabled = false;
    for (auto &UUIDs: camera_model_UUIDs) {
        deleteModelPrimitives(context, UUIDs.second);
    }
    camera_model_UUIDs.clear();
}

void RadiationModel::buildLightModelGeometry(uint sourceID) {

    assert(sourceID < radiation_sources.size());

    RadiationSource source = radiation_sources.at(sourceID);
    if (source.source_type == RADIATION_SOURCE_TYPE_SPHERE) {
        source_model_UUIDs[sourceID] = context->loadOBJ("plugins/radiation/camera_light_models/SphereLightSource.obj", true);
    } else if (source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
        source_model_UUIDs[sourceID] = context->loadOBJ("plugins/radiation/camera_light_models/SphereLightSource.obj", true);
    } else if (source.source_type == RADIATION_SOURCE_TYPE_DISK) {
        source_model_UUIDs[sourceID] = context->loadOBJ("plugins/radiation/camera_light_models/DiskLightSource.obj", true);
        context->scalePrimitive(source_model_UUIDs.at(sourceID), make_vec3(source.source_width.x, source.source_width.y, 0.05f * source.source_width.x));
        std::vector<uint> UUIDs_arrow = context->loadOBJ("plugins/radiation/camera_light_models/Arrow.obj", true);
        source_model_UUIDs.at(sourceID).insert(source_model_UUIDs.at(sourceID).begin(), UUIDs_arrow.begin(), UUIDs_arrow.end());
        context->scalePrimitive(UUIDs_arrow, make_vec3(1, 1, 1) * 0.25f * source.source_width.x);
    } else if (source.source_type == RADIATION_SOURCE_TYPE_RECTANGLE) {
        source_model_UUIDs[sourceID] = context->loadOBJ("plugins/radiation/camera_light_models/RectangularLightSource.obj", true);
        context->scalePrimitive(source_model_UUIDs.at(sourceID), make_vec3(source.source_width.x, source.source_width.y, fmin(0.05f * (source.source_width.x + source.source_width.y), 0.5f * fmin(source.source_width.x, source.source_width.y))));
        std::vector<uint> UUIDs_arrow = context->loadOBJ("plugins/radiation/camera_light_models/Arrow.obj", true);
        source_model_UUIDs.at(sourceID).insert(source_model_UUIDs.at(sourceID).begin(), UUIDs_arrow.begin(), UUIDs_arrow.end());
        context->scalePrimitive(UUIDs_arrow, make_vec3(1, 1, 1) * 0.15f * (source.source_width.x + source.source_width.y));
    } else {
        return;
    }

    if (source.source_type == RADIATION_SOURCE_TYPE_SPHERE) {
        context->scalePrimitive(source_model_UUIDs.at(sourceID), make_vec3(source.source_width.x, source.source_width.x, source.source_width.x));
        context->translatePrimitive(source_model_UUIDs.at(sourceID), source.source_position);
    } else if (source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
        vec3 center;
        float radius;
        context->getDomainBoundingSphere(center, radius);
        context->scalePrimitive(source_model_UUIDs.at(sourceID), make_vec3(1, 1, 1) * 0.1f * radius);
        vec3 sunvec = source.source_position;
        sunvec.normalize();
        context->translatePrimitive(source_model_UUIDs.at(sourceID), center + sunvec * radius);
    } else {
        context->rotatePrimitive(source_model_UUIDs.at(sourceID), source.source_rotation.x, "x");
        context->rotatePrimitive(source_model_UUIDs.at(sourceID), source.source_rotation.y, "y");
        context->rotatePrimitive(source_model_UUIDs.at(sourceID), source.source_rotation.z, "z");
        context->translatePrimitive(source_model_UUIDs.at(sourceID), source.source_position);
    }

    context->setPrimitiveData(source_model_UUIDs.at(sourceID), "invisibility_flag", uint(1)); // source model does not interact with radiation field
}

void RadiationModel::buildCameraModelGeometry(const std::string &cameralabel) {

    assert(cameras.find(cameralabel) != cameras.end());

    RadiationCamera camera = cameras.at(cameralabel);

    vec3 viewvec = camera.lookat - camera.position;
    SphericalCoord viewsph = cart2sphere(viewvec);

    camera_model_UUIDs[cameralabel] = context->loadOBJ("plugins/radiation/camera_light_models/Camera.obj", true);

    context->rotatePrimitive(camera_model_UUIDs.at(cameralabel), viewsph.elevation, "x");
    context->rotatePrimitive(camera_model_UUIDs.at(cameralabel), -viewsph.azimuth, "z");

    context->translatePrimitive(camera_model_UUIDs.at(cameralabel), camera.position);

    context->setPrimitiveData(camera_model_UUIDs.at(cameralabel), "invisibility_flag", uint(1)); // camera model does not interact with radiation field
}

void RadiationModel::updateLightModelPosition(uint sourceID, const helios::vec3 &delta_position) {

    assert(sourceID < radiation_sources.size());

    RadiationSource source = radiation_sources.at(sourceID);

    if (source.source_type != RADIATION_SOURCE_TYPE_SPHERE && source.source_type != RADIATION_SOURCE_TYPE_DISK && source.source_type != RADIATION_SOURCE_TYPE_RECTANGLE) {
        return;
    }

    context->translatePrimitive(source_model_UUIDs.at(sourceID), delta_position);
}

void RadiationModel::updateCameraModelPosition(const std::string &cameralabel) {

    assert(cameras.find(cameralabel) != cameras.end());

    context->deletePrimitive(camera_model_UUIDs.at(cameralabel));
    buildCameraModelGeometry(cameralabel);
}

namespace {
    //! Trapezoidal integral of integrand(wavelength, spectrum value) over the part of a tabulated spectrum lying between two wavelength bounds.
    /**
     * The spectrum is interpolated linearly to the bounds, so a bound that falls between grid points contributes only the portion of its segment
     * inside the bounds. Wavelengths outside the tabulated range contribute nothing.
     */
    template<typename Integrand>
    float integrateSpectrumBetweenBounds(const std::vector<helios::vec2> &spectrum, float wavelength1, float wavelength2, Integrand integrand) {
        float integral = 0.f;
        for (size_t i = 0; i + 1 < spectrum.size(); i++) {
            const float x0 = spectrum.at(i).x;
            const float x1 = spectrum.at(i + 1).x;
            if (x1 <= wavelength1) {
                continue;
            }
            if (x0 >= wavelength2) {
                break;
            }
            const float lower = std::max(x0, wavelength1);
            const float upper = std::min(x1, wavelength2);
            const float slope = (spectrum.at(i + 1).y - spectrum.at(i).y) / (x1 - x0);
            const float y_lower = spectrum.at(i).y + slope * (lower - x0);
            const float y_upper = spectrum.at(i).y + slope * (upper - x0);
            integral += 0.5f * (integrand(lower, y_lower) + integrand(upper, y_upper)) * (upper - lower);
        }
        return integral;
    }

    //! Integrals over a wavelength interval of an object spectrum weighted by a weighting spectrum and a camera response
    struct WeightedSpectrumIntegral {
        //! Integral of object x weight x camera
        float weighted = 0.f;
        //! Integral of the weight alone
        float weight = 0.f;
    };

    //! Integrate object(λ)·weight(λ)·camera(λ) and weight(λ) over [wavelength1, wavelength2]
    /**
     * Each spectrum is linear between its tabulated points and zero outside its tabulated range; an empty spectrum stands for a
     * value of 1 at every wavelength. The product is integrated on the union of the three grids, so the weight integral is exact
     * and equal to integrateSpectrum(weight, wavelength1, wavelength2), and a narrow weight spectrum is resolved even when the
     * object spectrum is coarse.
     */
    WeightedSpectrumIntegral integrateWeightedSpectrum(const std::vector<helios::vec2> &object, const std::vector<helios::vec2> &weight, const std::vector<helios::vec2> &camera, float wavelength1, float wavelength2) {
        WeightedSpectrumIntegral result;
        result.weight = weight.empty() ? wavelength2 - wavelength1 : integrateSpectrumBetweenBounds(weight, wavelength1, wavelength2, [](float, float value) { return value; });

        const std::vector<const std::vector<helios::vec2> *> spectra = {&object, &weight, &camera};
        float lower = wavelength1;
        float upper = wavelength2;
        for (const auto *spectrum: spectra) {
            if (!spectrum->empty()) {
                lower = std::max(lower, spectrum->front().x);
                upper = std::min(upper, spectrum->back().x);
            }
        }
        if (upper <= lower) {
            return result; // the spectra do not overlap within the bounds, so the product is zero everywhere
        }

        std::vector<float> grid = {lower, upper};
        for (const auto *spectrum: spectra) {
            for (const helios::vec2 &point: *spectrum) {
                if (point.x > lower && point.x < upper) {
                    grid.push_back(point.x);
                }
            }
        }
        std::sort(grid.begin(), grid.end());
        grid.erase(std::unique(grid.begin(), grid.end()), grid.end());

        std::vector<helios::vec2> weight_times_camera;
        weight_times_camera.reserve(grid.size());
        for (float wavelength: grid) {
            const float weight_value = weight.empty() ? 1.f : helios::interp1(weight, wavelength);
            const float camera_value = camera.empty() ? 1.f : helios::interp1(camera, wavelength);
            weight_times_camera.push_back(helios::make_vec2(wavelength, weight_value * camera_value));
        }
        result.weighted = integrateSpectrumBetweenBounds(weight_times_camera, lower, upper, [&](float wavelength, float weight_camera_value) { return (object.empty() ? 1.f : helios::interp1(object, wavelength)) * weight_camera_value; });
        return result;
    }

    //! Format a wavelength for an error message without trailing zeros, e.g. "800" or "512.5"
    std::string wavelengthString(float wavelength) {
        std::ostringstream stream;
        stream << wavelength;
        return stream.str();
    }

    //! Format a wavelength interval for an error message, e.g. "800 to 900 nm"
    std::string wavelengthRangeString(float wavelength1, float wavelength2) {
        return wavelengthString(wavelength1) + " to " + wavelengthString(wavelength2) + " nm";
    }

    //! Whether a band is one of the internal SIF excitation bands that ensureExcitationSet() creates
    bool isSIFExcitationBand(const std::string &band_label) {
        return band_label.compare(0, 9, "_SIF_exc_") == 0;
    }

    //! Format a spectrum's tabulated range for an error message, e.g. "800 to 900 nm"
    std::string spectrumRangeString(const std::vector<helios::vec2> &spectrum) {
        return wavelengthRangeString(spectrum.front().x, spectrum.back().x);
    }
} // namespace

float RadiationModel::integrateSpectrum(uint source_ID, const std::vector<helios::vec2> &object_spectrum, float wavelength1, float wavelength2) const {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum was not set for source ID. Make sure to set its spectrum using setSourceSpectrum() function.");
    } else if (object_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    } else if (wavelength1 > wavelength2 || wavelength1 == wavelength2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Lower wavelength bound must be less than the upper wavelength bound.");
    }

    const std::vector<helios::vec2> &source_spectrum = radiation_sources.at(source_ID).source_spectrum;
    if (source_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Source " + std::to_string(source_ID) + " has no spectrum. Set one with setSourceSpectrum() before weighting a spectrum by it.");
    }

    const WeightedSpectrumIntegral integral = integrateWeightedSpectrum(object_spectrum, source_spectrum, {}, wavelength1, wavelength2);
    if (integral.weight <= 0.f) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): The spectrum of source " + std::to_string(source_ID) + " (tabulated " + spectrumRangeString(source_spectrum) + ") has no energy from " +
                             wavelengthRangeString(wavelength1, wavelength2) + ", so there is nothing to weight by.");
    }
    return integral.weighted / integral.weight;
}

float RadiationModel::integrateSpectrum(const std::vector<helios::vec2> &object_spectrum, float wavelength1, float wavelength2) const {

    if (object_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    } else if (wavelength1 > wavelength2 || wavelength1 == wavelength2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Lower wavelength bound must be less than the upper wavelength bound.");
    }

    return integrateSpectrumBetweenBounds(object_spectrum, wavelength1, wavelength2, [](float, float value) { return value; });
}

float RadiationModel::integrateSpectrum(const std::vector<helios::vec2> &object_spectrum) const {
    float wavelength1 = object_spectrum.at(0).x;
    float wavelength2 = object_spectrum.at(object_spectrum.size() - 1).x;
    float E = RadiationModel::integrateSpectrum(object_spectrum, wavelength1, wavelength2);
    return E;
}

float RadiationModel::integrateSpectrum(uint source_ID, const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum) const {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum was not set for source ID. Make sure to set its spectrum using setSourceSpectrum() function.");
    } else if (object_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    } else if (camera_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Camera spectral response must have at least 2 wavelengths.");
    }

    const std::vector<helios::vec2> &source_spectrum = radiation_sources.at(source_ID).source_spectrum;
    if (source_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Source " + std::to_string(source_ID) + " has no spectrum. Set one with setSourceSpectrum() before weighting a spectrum by it.");
    }

    // Integrate over the wavelengths where both the source and the camera are tabulated
    const float wavelength1 = std::max(source_spectrum.front().x, camera_spectrum.front().x);
    const float wavelength2 = std::min(source_spectrum.back().x, camera_spectrum.back().x);
    const WeightedSpectrumIntegral integral = wavelength2 > wavelength1 ? integrateWeightedSpectrum(object_spectrum, source_spectrum, camera_spectrum, wavelength1, wavelength2) : WeightedSpectrumIntegral{};
    if (integral.weight <= 0.f) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): The spectrum of source " + std::to_string(source_ID) + " (tabulated " + spectrumRangeString(source_spectrum) +
                             ") has no energy where the camera spectral response is tabulated (" + spectrumRangeString(camera_spectrum) + "), so there is nothing to weight by.");
    }
    return integral.weighted / integral.weight;
}

float RadiationModel::integrateSpectrum(const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum) const {

    if (object_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    } else if (camera_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Camera spectral response must have at least 2 wavelengths.");
    }

    // The camera response is the weight: a camera-response-weighted average of the object spectrum
    const WeightedSpectrumIntegral integral = integrateWeightedSpectrum(object_spectrum, camera_spectrum, {}, camera_spectrum.front().x, camera_spectrum.back().x);
    if (integral.weight <= 0.f) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): The camera spectral response (tabulated " + spectrumRangeString(camera_spectrum) + ") is zero everywhere, so there is nothing to weight by.");
    }
    return integral.weighted / integral.weight;
}

float RadiationModel::integrateSourceSpectrum(uint source_ID, float wavelength1, float wavelength2) const {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::integrateSourceSpectrum): Radiation spectrum was not set for source ID. Make sure to set its spectrum using setSourceSpectrum() function.");
    } else if (wavelength1 > wavelength2 || wavelength1 == wavelength2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSourceSpectrum): Lower wavelength bound must be less than the upper wavelength bound.");
    }

    return integrateSpectrum(radiation_sources.at(source_ID).source_spectrum, wavelength1, wavelength2);
}

void RadiationModel::scaleSpectrum(const std::string &existing_global_data_label, const std::string &new_global_data_label, float scale_factor) const {

    std::vector<helios::vec2> spectrum = loadSpectralData(existing_global_data_label);

    for (helios::vec2 &s: spectrum) {
        s.y *= scale_factor;
    }

    context->setGlobalData(new_global_data_label.c_str(), spectrum);
}

void RadiationModel::scaleSpectrum(const std::string &global_data_label, float scale_factor) const {

    std::vector<vec2> spectrum = loadSpectralData(global_data_label);

    for (vec2 &s: spectrum) {
        s.y *= scale_factor;
    }

    context->setGlobalData(global_data_label.c_str(), spectrum);
}

void RadiationModel::scaleSpectrumRandomly(const std::string &existing_global_data_label, const std::string &new_global_data_label, float minimum_scale_factor, float maximum_scale_factor) const {

    scaleSpectrum(existing_global_data_label, new_global_data_label, context->randu(minimum_scale_factor, maximum_scale_factor));
}


void RadiationModel::blendSpectra(const std::string &new_spectrum_label, const std::vector<std::string> &spectrum_labels, const std::vector<float> &weights) const {

    if (spectrum_labels.size() != weights.size()) {
        helios_runtime_error("ERROR (RadiationModel::blendSpectra): number of spectra and weights must be equal");
    } else if (fabsf(sum(weights) - 1.f) > 1e-5f) {
        helios_runtime_error("ERROR (RadiationModel::blendSpectra): weights must sum to 1");
    }

    std::vector<vec2> new_spectrum;
    uint spectrum_size = 0;

    std::vector<std::vector<vec2>> spectrum(spectrum_labels.size());

    uint lambda_start = 0;
    uint lambda_end = 0;
    for (uint i = 0; i < spectrum_labels.size(); i++) {

        spectrum.at(i) = loadSpectralData(spectrum_labels.at(i));

        if (i == 0) {
            lambda_start = spectrum.at(i).front().x;
            lambda_end = spectrum.at(i).back().x;
        } else {
            if (spectrum.at(i).front().x > lambda_start) {
                lambda_start = spectrum.at(i).front().x;
            }
            if (spectrum.at(i).back().x < lambda_end) {
                lambda_end = spectrum.at(i).back().x;
            }
        }
    }

    spectrum_size = lambda_end - lambda_start + 1;
    new_spectrum.resize(spectrum_size);
    for (uint j = 0; j < spectrum_size; j++) {
        new_spectrum.at(j) = make_vec2(lambda_start + j, 0);
    }

    // trim front
    for (uint i = 0; i < spectrum_labels.size(); i++) {
        for (uint j = 0; j < spectrum.at(i).size(); j++) {

            if (spectrum.at(i).at(j).x >= lambda_start) {
                if (j > 0) {
                    spectrum.at(i).erase(spectrum.at(i).begin(), spectrum.at(i).begin() + j);
                }
                break;
            }
        }
    }

    // trim back
    for (uint i = 0; i < spectrum_labels.size(); i++) {
        for (int j = spectrum.at(i).size() - 1; j <= 0; j--) {

            if (spectrum.at(i).at(j).x <= lambda_end) {
                if (j < spectrum.at(i).size() - 1) {
                    spectrum.at(i).erase(spectrum.at(i).begin() + j + 1, spectrum.at(i).end());
                }
                break;
            }
        }
    }

    for (uint i = 0; i < spectrum_labels.size(); i++) {
        for (uint j = 0; j < spectrum_size; j++) {
            assert(new_spectrum.at(j).x == spectrum.at(i).at(j).x);
            new_spectrum.at(j).y += weights.at(i) * spectrum.at(i).at(j).y;
        }
    }

    context->setGlobalData(new_spectrum_label.c_str(), new_spectrum);
}

void RadiationModel::blendSpectraRandomly(const std::string &new_spectrum_label, const std::vector<std::string> &spectrum_labels) const {

    std::vector<float> weights;
    weights.resize(spectrum_labels.size());
    for (uint i = 0; i < spectrum_labels.size(); i++) {
        weights.at(i) = context->randu();
    }
    float sum_weights = sum(weights);
    for (uint i = 0; i < spectrum_labels.size(); i++) {
        weights.at(i) /= sum_weights;
    }

    blendSpectra(new_spectrum_label, spectrum_labels, weights);
}

void RadiationModel::interpolateSpectrumFromPrimitiveData(const std::vector<uint> &primitive_UUIDs, const std::vector<std::string> &spectra, const std::vector<float> &values, const std::string &primitive_data_query_label,
                                                          const std::string &primitive_data_radprop_label) {

    // Validate that spectra and values have the same length
    if (spectra.size() != values.size()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromPrimitiveData): The 'spectra' vector (size=" + std::to_string(spectra.size()) + ") and 'values' vector (size=" + std::to_string(values.size()) +
                             ") must have the same length.");
    }

    // Validate that vectors are not empty
    if (spectra.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromPrimitiveData): The 'spectra' and 'values' vectors cannot be empty.");
    }

    // Validate that primitive_UUIDs is not empty
    if (primitive_UUIDs.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromPrimitiveData): The 'primitive_UUIDs' vector cannot be empty.");
    }

    // Validate that query and target data labels are not empty
    if (primitive_data_query_label.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromPrimitiveData): The 'primitive_data_query_label' cannot be empty.");
    }

    if (primitive_data_radprop_label.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromPrimitiveData): The 'primitive_data_radprop_label' cannot be empty.");
    }

    // Search for existing config with matching query and target labels
    SpectrumInterpolationConfig *existing_config = nullptr;
    for (auto &config: spectrum_interpolation_configs) {
        if (config.query_data_label == primitive_data_query_label && config.target_data_label == primitive_data_radprop_label) {
            existing_config = &config;
            break;
        }
    }

    if (existing_config != nullptr) {
        // Check if spectra/values match the existing config
        bool spectra_match = (existing_config->spectra_labels == spectra && existing_config->mapping_values == values);

        if (spectra_match) {
            // Merge UUIDs into existing config (unordered_set handles duplicates automatically)
            existing_config->primitive_UUIDs.insert(primitive_UUIDs.begin(), primitive_UUIDs.end());
        } else {
            // Replace entire config with new spectra/values and UUIDs
            existing_config->spectra_labels = spectra;
            existing_config->mapping_values = values;
            existing_config->primitive_UUIDs.clear();
            existing_config->primitive_UUIDs.insert(primitive_UUIDs.begin(), primitive_UUIDs.end());
        }
    } else {
        // Create new config
        SpectrumInterpolationConfig config;
        config.primitive_UUIDs.insert(primitive_UUIDs.begin(), primitive_UUIDs.end());
        config.spectra_labels = spectra;
        config.mapping_values = values;
        config.query_data_label = primitive_data_query_label;
        config.target_data_label = primitive_data_radprop_label;

        spectrum_interpolation_configs.push_back(config);
    }
}

void RadiationModel::interpolateSpectrumFromObjectData(const std::vector<uint> &object_IDs, const std::vector<std::string> &spectra, const std::vector<float> &values, const std::string &object_data_query_label,
                                                       const std::string &primitive_data_radprop_label) {

    // Validate that spectra and values have the same length
    if (spectra.size() != values.size()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromObjectData): The 'spectra' vector (size=" + std::to_string(spectra.size()) + ") and 'values' vector (size=" + std::to_string(values.size()) + ") must have the same length.");
    }

    // Validate that vectors are not empty
    if (spectra.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromObjectData): The 'spectra' and 'values' vectors cannot be empty.");
    }

    // Validate that object_IDs is not empty
    if (object_IDs.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromObjectData): The 'object_IDs' vector cannot be empty.");
    }

    // Validate that query and target data labels are not empty
    if (object_data_query_label.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromObjectData): The 'object_data_query_label' cannot be empty.");
    }

    if (primitive_data_radprop_label.empty()) {
        helios_runtime_error("ERROR (RadiationModel::interpolateSpectrumFromObjectData): The 'primitive_data_radprop_label' cannot be empty.");
    }

    // Search for existing config with matching query and target labels
    SpectrumInterpolationConfig *existing_config = nullptr;
    for (auto &config: spectrum_interpolation_configs) {
        if (config.query_data_label == object_data_query_label && config.target_data_label == primitive_data_radprop_label) {
            existing_config = &config;
            break;
        }
    }

    if (existing_config != nullptr) {
        // Check if spectra/values match the existing config
        bool spectra_match = (existing_config->spectra_labels == spectra && existing_config->mapping_values == values);

        if (spectra_match) {
            // Merge object IDs into existing config (unordered_set handles duplicates automatically)
            existing_config->object_IDs.insert(object_IDs.begin(), object_IDs.end());
        } else {
            // Replace entire config with new spectra/values and object IDs
            existing_config->spectra_labels = spectra;
            existing_config->mapping_values = values;
            existing_config->object_IDs.clear();
            existing_config->object_IDs.insert(object_IDs.begin(), object_IDs.end());
        }
    } else {
        // Create new config
        SpectrumInterpolationConfig config;
        config.object_IDs.insert(object_IDs.begin(), object_IDs.end());
        config.spectra_labels = spectra;
        config.mapping_values = values;
        config.query_data_label = object_data_query_label;
        config.target_data_label = primitive_data_radprop_label;

        spectrum_interpolation_configs.push_back(config);
    }
}

void RadiationModel::setSourcePosition(uint source_ID, const vec3 &position) {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::setSourcePosition): Source ID out of bounds. Only " + std::to_string(radiation_sources.size() - 1) + " radiation sources.");
    }

    vec3 old_position = radiation_sources.at(source_ID).source_position;

    if (radiation_sources.at(source_ID).source_type == RADIATION_SOURCE_TYPE_COLLIMATED) {
        radiation_sources.at(source_ID).source_position = position / position.magnitude();
    } else {
        radiation_sources.at(source_ID).source_position = position * radiation_sources.at(source_ID).source_position_scaling_factor;
    }

    if (islightvisualizationenabled) {
        updateLightModelPosition(source_ID, radiation_sources.at(source_ID).source_position - old_position);
    }
}

void RadiationModel::setSourcePosition(uint source_ID, const SphericalCoord &position) {
    setSourcePosition(source_ID, sphere2cart(position));
}

helios::vec3 RadiationModel::getSourcePosition(uint source_ID) const {
    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::getSourcePosition): Source ID does not exist.");
    }
    return radiation_sources.at(source_ID).source_position;
}

void RadiationModel::setScatteringDepth(const std::string &label, uint depth) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::setScatteringDepth): Cannot set scattering depth for band '" + label + "' because it is not a valid band.");
    }
    radiation_bands.at(label).scatteringDepth = depth;
}

void RadiationModel::setMinScatterEnergy(const std::string &label, float energy) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (setMinScatterEnergy): Cannot set minimum scattering energy for band '" + label + "' because it is not a valid band.");
    }
    if (energy < 0.f || std::isnan(energy)) {
        helios_runtime_error("ERROR (setMinScatterEnergy): The minimum scattering energy for band '" + label + "' must be zero (never stop early) or positive, but " + std::to_string(energy) + " was given.");
    }
    radiation_bands.at(label).minScatterEnergy = energy;
}

void RadiationModel::enableScatteringRayReduction() {
    scattering_ray_reduction = true;
}

void RadiationModel::disableScatteringRayReduction() {
    scattering_ray_reduction = false;
}

void RadiationModel::enforcePeriodicBoundary(const std::string &boundary) {

    if (boundary == "x") {

        periodic_flag.x = 1;

    } else if (boundary == "y") {

        periodic_flag.y = 1;

    } else if (boundary == "xy") {

        periodic_flag.x = 1;
        periodic_flag.y = 1;

    } else {

        std::cout << "WARNING (RadiationModel::enforcePeriodicBoundary()): unknown boundary of '" << boundary << "'. Possible choices are x, y, or xy." << std::endl;
    }
}

void RadiationModel::updateGeometry() {
    updateGeometry(context->getAllUUIDs());
    // A full-Context build is the default; clear the subset latch set by the overload below so
    // that subsequent geometry changes are picked up automatically.
    isgeometrysubset = false;
}

bool RadiationModel::needsGeometryRebuild() const {

    if (!isgeometryinitialized) {
        return true;
    }

    // The user explicitly restricted the model to a subset of primitives. Rebuilding from the full
    // Context would silently discard that choice, so leave the existing geometry alone.
    if (isgeometrysubset) {
        return false;
    }

    // Primitives added, deleted, moved, or reshaped since the last build leave the acceleration structure stale. The count
    // catches additions; the shape revision catches the rest, including a deletion paired with an addition that leaves the
    // count unchanged. Primitive data writes (including this model's own flux output) do not advance the shape revision.
    return context->getPrimitiveCount() != geometry_build_primitive_count || context->getShapeRevision() != geometry_build_shape_revision;
}


void RadiationModel::enableCameraFluxSmoothing(float crease_angle_degrees) {
    if (crease_angle_degrees < 0.f || crease_angle_degrees > 180.f) {
        helios_runtime_error("ERROR (RadiationModel::enableCameraFluxSmoothing): Crease angle of " + std::to_string(crease_angle_degrees) +
                             " degrees is out of range. The crease angle must be between 0 and 180 degrees, where 0 treats every mesh edge as a crease and 180 treats none of them as one.");
    }
    cameraflux_smoothing_enabled = true;
    cameraflux_smoothing_crease_angle = crease_angle_degrees;
    // Smoothing records the direct radiation reaching each primitive, which the materials update allocates for
    radiativepropertiesneedupdate = true;

    // The topology is uploaded with the geometry, so toggling smoothing after geometry has already been built has to repeat that upload.
    if (isgeometryinitialized) {
        uploadCameraFluxSmoothingTopology();
    }
}

void RadiationModel::disableCameraFluxSmoothing() {
    cameraflux_smoothing_enabled = false;
    radiativepropertiesneedupdate = true;
    if (isgeometryinitialized) {
        uploadCameraFluxSmoothingTopology();
    }
}

void RadiationModel::uploadCameraFluxSmoothingTopology() {
    buildCameraFluxSmoothingTopology();

    // Re-uploading the geometry releases the buffers the acceleration structure was built from, so it is rebuilt here rather than left pointing at storage that has been freed. Toggling smoothing is a
    // deliberate one-off, so paying for a rebuild is preferable to carrying a second upload path that only writes this one buffer.
    backend->updateGeometry(geometry_data);
    backend->buildAccelerationStructure();
}

bool RadiationModel::isCameraFluxSmoothingEnabled() const {
    return cameraflux_smoothing_enabled;
}

float RadiationModel::getCameraFluxSmoothingCreaseAngle() const {
    return cameraflux_smoothing_crease_angle;
}

void RadiationModel::buildCameraFluxSmoothingTopology() {

    smoothing_vertex_indices.clear();
    smoothing_vertex_area.clear();
    smoothing_primitive_area.clear();
    smoothing_vertex_normals.clear();
    smoothing_facet_normal.clear();
    smoothing_facet_centroid.clear();
    smoothing_vertex_count = 0;
    // A rebuilt topology invalidates whatever normals the backend holds; the backend releases them with the geometry upload that follows (see RayTracingBackend::uploadVertexNormals()).
    smoothing_vertex_normals_uploaded = false;
    vertex_direct_irradiance_uploaded = false;
    geometry_data.smoothing_vertex_indices.clear();
    geometry_data.smoothing_vertex_count = 0;

    if (!cameraflux_smoothing_enabled) {
        return;
    }

    const size_t primitive_count = context_UUIDs.size();
    if (primitive_count == 0) {
        return;
    }

    smoothing_vertex_indices.assign(4 * primitive_count, -1);
    smoothing_primitive_area.assign(primitive_count, 0.f);
    smoothing_facet_normal.assign(primitive_count, helios::make_vec3(0, 0, 0));
    smoothing_facet_centroid.assign(primitive_count, helios::make_vec3(0, 0, 0));

    // Gather the primitives of each parent object together, so that the per-object topology is fetched once rather than once per facet.
    std::map<uint, std::vector<size_t>> positions_by_object;
    for (size_t position = 0; position < primitive_count; position++) {
        const uint UUID = context_UUIDs.at(position);
        smoothing_primitive_area.at(position) = context->getPrimitiveArea(UUID);

        const uint parent_ObjID = context->getPrimitiveParentObjectID(UUID);
        if (parent_ObjID == 0 || !context->doesObjectExist(parent_ObjID)) {
            continue; // a primitive belonging to no object has no neighbours to share a vertex with
        }
        positions_by_object[parent_ObjID].push_back(position);
    }

    const float crease_angle_radians_mesh = deg2rad(cameraflux_smoothing_crease_angle);

    std::vector<float> vertex_area;
    std::vector<helios::vec3> vertex_normal_sum;
    size_t next_global_vertex = 0;

    for (const auto &object_entry: positions_by_object) {
        const uint parent_ObjID = object_entry.first;
        const std::vector<size_t> &positions = object_entry.second;

        if (!context->doesObjectHaveSharedVertexTopology(parent_ObjID)) {
            continue; // a box, a disk, or a mesh with no face table: its faces are flat, or its facets are not known to meet
        }

        // Every coincident vertex is shared, across the rings of a sphere and the segments of a tube or cone as well as around them. What smoothing averages onto the vertices is the radiation other than the
        // direct radiation of the sources, which the camera evaluates per facet with the shading normal (see subtractCameraDirectFlux()); that remainder has no sharp shadow edges to preserve.
        const size_t local_vertex_count = context->getObjectSharedVertexCount(parent_ObjID, helios::WELD_FULL);
        if (local_vertex_count == 0) {
            continue; // the object could not locate its own facets, so no topology is available
        }

        std::vector<uint> object_UUIDs;
        object_UUIDs.reserve(positions.size());
        for (size_t position: positions) {
            object_UUIDs.push_back(context_UUIDs.at(position));
        }

        const std::vector<std::vector<int>> local_indices = context->getObjectPrimitiveSharedVertexIndices(parent_ObjID, object_UUIDs, helios::WELD_FULL);

        // A curved object is a tessellation of a smooth surface, so every edge of it is an artifact of the tessellation and none of them is a crease. Only a mesh can have an edge that is genuinely there.
        const bool object_can_have_creases = (context->getObjectType(parent_ObjID) == helios::OBJECT_TYPE_POLYMESH);
        const float crease_angle_radians = object_can_have_creases ? crease_angle_radians_mesh : float(M_PI);

        // Facet normals, for the crease test and the vertex shading normals. Computed from the corner positions rather than read from any stored vertex normal, which smoothing does not use.
        std::vector<helios::vec3> facet_normal(object_UUIDs.size(), helios::make_vec3(0, 0, 0));
        for (size_t k = 0; k < object_UUIDs.size(); k++) {
            if (local_indices.at(k).empty()) {
                continue;
            }
            const std::vector<helios::vec3> corners = context->getPrimitiveVertices(object_UUIDs.at(k));
            if (corners.size() < 3) {
                continue;
            }
            const helios::vec3 facet_cross = cross(corners.at(1) - corners.at(0), corners.at(2) - corners.at(0));
            const float cross_magnitude = facet_cross.magnitude();
            if (cross_magnitude > 0.f) {
                facet_normal.at(k) = facet_cross / cross_magnitude;
            }
            helios::vec3 centroid = helios::make_vec3(0, 0, 0);
            for (const helios::vec3 &corner: corners) {
                centroid += corner;
            }
            smoothing_facet_normal.at(positions.at(k)) = facet_normal.at(k);
            smoothing_facet_centroid.at(positions.at(k)) = centroid / float(corners.size());
        }

        // Which facets meet at each of the object's shared vertices.
        std::vector<std::vector<size_t>> vertex_to_facets(local_vertex_count);
        for (size_t k = 0; k < local_indices.size(); k++) {
            for (int local_vertex: local_indices.at(k)) {
                if (local_vertex >= 0 && size_t(local_vertex) < local_vertex_count) {
                    vertex_to_facets.at(size_t(local_vertex)).push_back(k);
                }
            }
        }

        // Assign each smooth group of facets around a vertex its own slot in the global buffer. Facets meeting at more than the crease angle end up in different groups, so a genuine edge stays sharp
        // instead of the lit side bleeding into the shaded one. Each facet also records whether its normal must be flipped to agree with the group's first facet, so that oppositely wound facets sharing a
        // vertex (possible on a curved object, whose crease angle admits any pair) add to the vertex normal instead of cancelling.
        std::vector<std::map<size_t, std::pair<int, float>>> facet_vertex_slot(local_vertex_count);

        for (size_t local_vertex = 0; local_vertex < local_vertex_count; local_vertex++) {
            const std::vector<size_t> &incident_facets = vertex_to_facets.at(local_vertex);
            if (incident_facets.empty()) {
                continue; // a lattice slot no facet reaches; allocating a buffer entry for it would be wasted
            }

            std::vector<std::vector<size_t>> smooth_groups;
            for (size_t incident_facet: incident_facets) {
                bool placed = false;
                for (std::vector<size_t> &group: smooth_groups) {
                    for (size_t member_facet: group) {
                        if (acos_safe(facet_normal.at(incident_facet) * facet_normal.at(member_facet)) <= crease_angle_radians) {
                            group.push_back(incident_facet);
                            placed = true;
                            break;
                        }
                    }
                    if (placed) {
                        break;
                    }
                }
                if (!placed) {
                    smooth_groups.push_back({incident_facet});
                }
            }

            for (const std::vector<size_t> &group: smooth_groups) {
                const int global_vertex = int(next_global_vertex);
                next_global_vertex++;
                vertex_area.push_back(0.f);
                vertex_normal_sum.push_back(helios::make_vec3(0, 0, 0));
                const helios::vec3 &reference_normal = facet_normal.at(group.front());
                for (size_t member_facet: group) {
                    const float orientation = (facet_normal.at(member_facet) * reference_normal < 0.f) ? -1.f : 1.f;
                    facet_vertex_slot.at(local_vertex)[member_facet] = std::make_pair(global_vertex, orientation);
                }
            }
        }

        // Record where each facet's corners landed, and how much area meets at each shared vertex.
        for (size_t k = 0; k < local_indices.size(); k++) {
            const std::vector<int> &corners = local_indices.at(k);
            if (corners.size() != 3 && corners.size() != 4) {
                continue; // the object does not describe this facet
            }

            // Resolve every corner before writing any of them. The camera program tells a triangle from a patch by whether the fourth entry is negative, so a partly-resolved patch would be read as a
            // triangle and interpolated with the wrong weights - a plausible-looking but wrong image rather than a visible failure.
            int resolved[4] = {-1, -1, -1, -1};
            float orientation[4] = {1.f, 1.f, 1.f, 1.f};
            bool all_corners_resolved = true;
            for (size_t c = 0; c < corners.size(); c++) {
                const int local_vertex = corners.at(c);
                if (local_vertex < 0 || size_t(local_vertex) >= local_vertex_count) {
                    all_corners_resolved = false;
                    break;
                }
                auto slot = facet_vertex_slot.at(size_t(local_vertex)).find(k);
                if (slot == facet_vertex_slot.at(size_t(local_vertex)).end()) {
                    all_corners_resolved = false;
                    break;
                }
                resolved[c] = slot->second.first;
                orientation[c] = slot->second.second;
            }
            if (!all_corners_resolved) {
                continue; // leave the facet unsmoothed, so the camera keeps its per-primitive value
            }

            const size_t position = positions.at(k);
            const float facet_area = smoothing_primitive_area.at(position);
            for (size_t c = 0; c < corners.size(); c++) {
                smoothing_vertex_indices.at(4 * position + c) = resolved[c];
                vertex_area.at(size_t(resolved[c])) += facet_area;
                vertex_normal_sum.at(size_t(resolved[c])) += orientation[c] * facet_area * facet_normal.at(k);
            }
        }
    }

    smoothing_vertex_count = next_global_vertex;
    smoothing_vertex_area = vertex_area;

    // A vertex no facet resolved onto keeps a zero normal; no camera hit reads it, because no facet refers to that slot.
    smoothing_vertex_normals.assign(3 * smoothing_vertex_count, 0.f);
    for (size_t vertex = 0; vertex < smoothing_vertex_count; vertex++) {
        const float magnitude = vertex_normal_sum.at(vertex).magnitude();
        if (magnitude > 0.f) {
            smoothing_vertex_normals.at(3 * vertex + 0) = vertex_normal_sum.at(vertex).x / magnitude;
            smoothing_vertex_normals.at(3 * vertex + 1) = vertex_normal_sum.at(vertex).y / magnitude;
            smoothing_vertex_normals.at(3 * vertex + 2) = vertex_normal_sum.at(vertex).z / magnitude;
        }
    }

    geometry_data.smoothing_vertex_indices = smoothing_vertex_indices;
    geometry_data.smoothing_vertex_count = smoothing_vertex_count;
}

void RadiationModel::accumulateCameraFluxAtVertices(const std::vector<float> &flux_top, const std::vector<float> &flux_bottom, size_t band_count, size_t component_count, std::vector<float> &vertex_flux_top,
                                                    std::vector<float> &vertex_flux_bottom) const {

    vertex_flux_top.assign(component_count * smoothing_vertex_count * band_count, 0.f);
    vertex_flux_bottom.assign(component_count * smoothing_vertex_count * band_count, 0.f);

    if (smoothing_vertex_count == 0) {
        return;
    }

    const size_t primitive_count = context_UUIDs.size();

    // A topology built against a different primitive ordering would attribute each facet's flux to whichever vertices now occupy those slots, producing a plausible but wrong image with no other symptom.
    if (smoothing_vertex_indices.size() != 4 * primitive_count || smoothing_primitive_area.size() != primitive_count) {
        helios_runtime_error("ERROR (RadiationModel::accumulateCameraFluxAtVertices): The camera flux smoothing topology describes " + std::to_string(smoothing_vertex_indices.size() / 4) +
                             " primitives but the geometry holds " + std::to_string(primitive_count) +
                             ". The topology is built by updateGeometry() and is only valid for the primitive ordering that build produced. Call updateGeometry() after changing the geometry.");
    }
    if (flux_top.size() < component_count * primitive_count * band_count || flux_bottom.size() < component_count * primitive_count * band_count) {
        helios_runtime_error("ERROR (RadiationModel::accumulateCameraFluxAtVertices): The outgoing radiation buffers hold " + std::to_string(flux_top.size()) + " entries but " + std::to_string(component_count * primitive_count * band_count) +
                             " are required for " + std::to_string(component_count) + " component(s) of " + std::to_string(primitive_count) + " primitives across " + std::to_string(band_count) + " bands.");
    }
    const size_t primitive_block = primitive_count * band_count;
    const size_t vertex_block = smoothing_vertex_count * band_count;
    // The second component, a surface layer's reflection, is averaged over the facets that have a layer only, so that the zeros of their neighbours without one do not dim it near a seam between the two
    std::vector<float> layered_area;
    if (component_count > 1) {
        if (material_data.surface_model.size() != primitive_count) {
            helios_runtime_error("ERROR (RadiationModel::accumulateCameraFluxAtVertices): The surface models describe " + std::to_string(material_data.surface_model.size()) + " primitives but the geometry holds " + std::to_string(primitive_count) +
                                 ". The materials must be updated after the geometry.");
        }
        layered_area.assign(smoothing_vertex_count, 0.f);
    }

    for (size_t position = 0; position < primitive_count; position++) {
        if (smoothing_vertex_indices.at(4 * position) < 0) {
            continue; // this primitive takes no part in smoothing
        }
        const float facet_area = smoothing_primitive_area.at(position);
        if (facet_area <= 0.f) {
            continue;
        }

        for (size_t c = 0; c < 4; c++) {
            const int global_vertex = smoothing_vertex_indices.at(4 * position + c);
            if (global_vertex < 0) {
                continue; // a triangle, which has only three corners
            }
            const bool has_layer = component_count > 1 && material_data.surface_model.at(position) != 0;
            if (has_layer) {
                layered_area.at(size_t(global_vertex)) += facet_area;
            }
            for (size_t component = 0; component < component_count; component++) {
                if (component > 0 && !has_layer) {
                    continue;
                }
                for (size_t b = 0; b < band_count; b++) {
                    vertex_flux_top.at(component * vertex_block + size_t(global_vertex) * band_count + b) += facet_area * flux_top.at(component * primitive_block + position * band_count + b);
                    vertex_flux_bottom.at(component * vertex_block + size_t(global_vertex) * band_count + b) += facet_area * flux_bottom.at(component * primitive_block + position * band_count + b);
                }
            }
        }
    }

    for (size_t vertex = 0; vertex < smoothing_vertex_count; vertex++) {
        const float total_area = smoothing_vertex_area.at(vertex);
        if (total_area <= 0.f) {
            continue; // no facet meets here, so the slot is never read; leaving it at zero avoids dividing by nothing
        }
        for (size_t component = 0; component < component_count; component++) {
            const float component_area = component == 0 ? total_area : layered_area.at(vertex);
            if (component_area <= 0.f) {
                continue; // no facet with a layer meets here, so no camera reads the layer's component at this vertex
            }
            for (size_t b = 0; b < band_count; b++) {
                vertex_flux_top.at(component * vertex_block + vertex * band_count + b) /= component_area;
                vertex_flux_bottom.at(component * vertex_block + vertex * band_count + b) /= component_area;
            }
        }
    }
}

void RadiationModel::subtractCameraDirectFlux(std::vector<float> &camera_scatter_top, std::vector<float> &camera_scatter_bottom, const std::vector<float> &direct_flux, uint camera_index, const std::vector<char> &band_launch_flag) const {
    const size_t primitive_count = context_UUIDs.size();
    const size_t source_count = source_data.size();
    const size_t global_band_count = band_launch_flag.size();
    std::vector<size_t> launch_to_global;
    for (size_t g = 0; g < global_band_count; g++) {
        if (band_launch_flag.at(g) != 0) {
            launch_to_global.push_back(g);
        }
    }
    const size_t launch_band_count = launch_to_global.size();
    const size_t camera_count = cameras.size();

    if (direct_flux.size() != source_count * primitive_count * launch_band_count) {
        helios_runtime_error("ERROR (RadiationModel::subtractCameraDirectFlux): The direct radiation reaching the primitives holds " + std::to_string(direct_flux.size()) + " values, but " + std::to_string(source_count) + " source(s), " +
                             std::to_string(primitive_count) + " primitives and " + std::to_string(launch_band_count) + " launched band(s) require " + std::to_string(source_count * primitive_count * launch_band_count) + ".");
    }
    if (camera_scatter_top.size() < primitive_count * launch_band_count || camera_scatter_bottom.size() != camera_scatter_top.size()) {
        helios_runtime_error("ERROR (RadiationModel::subtractCameraDirectFlux): The camera's outgoing radiation holds " + std::to_string(camera_scatter_top.size()) + " and " + std::to_string(camera_scatter_bottom.size()) + " values, but " +
                             std::to_string(primitive_count * launch_band_count) + " are required for each face.");
    }
    if (smoothing_vertex_indices.size() != 4 * primitive_count || smoothing_facet_normal.size() != primitive_count || smoothing_facet_centroid.size() != primitive_count) {
        helios_runtime_error("ERROR (RadiationModel::subtractCameraDirectFlux): The camera flux smoothing topology describes a different number of primitives than the geometry's " + std::to_string(primitive_count) +
                             ". Call updateGeometry() after changing the geometry.");
    }
    const size_t camera_material_size = (source_count + 1) * primitive_count * global_band_count * camera_count;
    if (material_data.reflectivity_cam.size() != camera_material_size || material_data.transmissivity_cam.size() != camera_material_size) {
        helios_runtime_error("ERROR (RadiationModel::subtractCameraDirectFlux): The camera-weighted reflectivity and transmissivity hold " + std::to_string(material_data.reflectivity_cam.size()) + " and " +
                             std::to_string(material_data.transmissivity_cam.size()) + " values, but " + std::to_string(camera_material_size) + " are required. The materials must be updated for the current cameras.");
    }
    const bool layers_present = material_data.surface_layer_enabled && material_data.surface_model.size() == primitive_count;
    const size_t table_size = material_data.surface_layer_table_size;
    CameraMaterialIndexer camera_material_index(source_count + 1, primitive_count, global_band_count, camera_count);

    for (size_t position = 0; position < primitive_count; position++) {
        if (smoothing_vertex_indices.at(4 * position) < 0) {
            continue; // not smoothed, so the camera reads this primitive's own value and nothing is averaged
        }
        const helios::vec3 &facet_normal = smoothing_facet_normal.at(position);
        for (size_t s = 0; s < source_count; s++) {
            // A collimated source and the sun sphere lie in one direction; a finite source is seen from the facet's centroid
            const bool directional = source_data.at(s).type == RADIATION_SOURCE_TYPE_COLLIMATED || source_data.at(s).type == RADIATION_SOURCE_TYPE_SUN_SPHERE;
            helios::vec3 to_source = directional ? source_data.at(s).position : source_data.at(s).position - smoothing_facet_centroid.at(position);
            to_source.normalize();
            const float mu_facet = facet_normal * to_source;
            if (mu_facet == 0.f) {
                continue; // edge-on, so no face is lit and the camera adds nothing back either
            }
            const bool lit_top = mu_facet > 0.f;
            for (size_t b = 0; b < launch_band_count; b++) {
                const float incident = direct_flux.at((s * primitive_count + position) * launch_band_count + b);
                if (incident <= 0.f) {
                    continue;
                }
                const size_t global_band = launch_to_global.at(b);
                // A surface layer reflects a(mu) of the arriving radiation at its surface, so the body receives the rest
                float body_fraction = 1.f;
                if (layers_present && material_data.surface_model.at(position) != 0) {
                    const uint32_t table = material_data.surface_model_band_table.at((size_t(material_data.surface_model.at(position)) - 1) * global_band_count + global_band);
                    if (table != UINT32_MAX) {
                        const float *albedo = &material_data.surface_layer_tables.at(size_t(table) * (table_size + 3));
                        const float x = std::min(1.f, std::fabs(mu_facet)) * float(table_size - 1);
                        const size_t i = std::min(size_t(x), table_size - 2);
                        body_fraction = 1.f - (albedo[i] + (x - float(i)) * (albedo[i + 1] - albedo[i]));
                    }
                }
                const size_t material = camera_material_index(s, position, global_band, camera_index);
                const float reflected = material_data.reflectivity_cam.at(material) * body_fraction * incident;
                const float transmitted = material_data.transmissivity_cam.at(material) * body_fraction * incident;
                const size_t index = position * launch_band_count + b;
                (lit_top ? camera_scatter_top : camera_scatter_bottom).at(index) -= reflected;
                (lit_top ? camera_scatter_bottom : camera_scatter_top).at(index) -= transmitted;
            }
        }
    }
}

void RadiationModel::accumulateDirectIrradianceAtVertices(const std::vector<float> &direct_flux, size_t launch_band_count, std::vector<float> &vertex_irradiance) const {
    const size_t primitive_count = context_UUIDs.size();
    const size_t source_count = source_data.size();
    vertex_irradiance.assign(source_count * smoothing_vertex_count * launch_band_count, 0.f);
    if (smoothing_vertex_count == 0 || source_count == 0) {
        return;
    }
    if (direct_flux.size() != source_count * primitive_count * launch_band_count) {
        helios_runtime_error("ERROR (RadiationModel::accumulateDirectIrradianceAtVertices): The direct radiation reaching the primitives holds " + std::to_string(direct_flux.size()) + " values, but " + std::to_string(source_count) + " source(s), " +
                             std::to_string(primitive_count) + " primitives and " + std::to_string(launch_band_count) + " launched band(s) require " + std::to_string(source_count * primitive_count * launch_band_count) + ".");
    }
    if (smoothing_vertex_indices.size() != 4 * primitive_count || smoothing_facet_normal.size() != primitive_count || smoothing_primitive_area.size() != primitive_count) {
        helios_runtime_error("ERROR (RadiationModel::accumulateDirectIrradianceAtVertices): The camera flux smoothing topology describes a different number of primitives than the geometry's " + std::to_string(primitive_count) +
                             ". Call updateGeometry() after changing the geometry.");
    }
    std::vector<float> weight(vertex_irradiance.size(), 0.f);
    for (size_t position = 0; position < primitive_count; position++) {
        const float facet_area = smoothing_primitive_area.at(position);
        if (smoothing_vertex_indices.at(4 * position) < 0 || facet_area <= 0.f) {
            continue;
        }
        for (size_t s = 0; s < source_count; s++) {
            const bool directional = source_data.at(s).type == RADIATION_SOURCE_TYPE_COLLIMATED || source_data.at(s).type == RADIATION_SOURCE_TYPE_SUN_SPHERE;
            helios::vec3 to_source = directional ? source_data.at(s).position : source_data.at(s).position - smoothing_facet_centroid.at(position);
            to_source.normalize();
            const float mu_facet = std::fabs(smoothing_facet_normal.at(position) * to_source);
            if (mu_facet < 0.01f) {
                continue;
            }
            for (size_t b = 0; b < launch_band_count; b++) {
                const float incident = direct_flux.at((s * primitive_count + position) * launch_band_count + b);
                if (incident <= 0.f) {
                    continue; // turned away from the source or occluded, so it says nothing about what its neighbours' turned-away faces would receive
                }
                for (size_t c = 0; c < 4; c++) {
                    const int vertex = smoothing_vertex_indices.at(4 * position + c);
                    if (vertex < 0) {
                        continue;
                    }
                    const size_t index = (s * smoothing_vertex_count + size_t(vertex)) * launch_band_count + b;
                    vertex_irradiance.at(index) += facet_area * incident / mu_facet;
                    weight.at(index) += facet_area;
                }
            }
        }
    }
    for (size_t index = 0; index < vertex_irradiance.size(); index++) {
        if (weight.at(index) > 0.f) {
            vertex_irradiance.at(index) /= weight.at(index);
        }
    }
}

void RadiationModel::accumulateSpectralIncidentAtVertices(const std::vector<float> &spectral_incident, const std::vector<float> &spectral_emitted, size_t slot_count, size_t band_count, std::vector<float> &vertex_values) const {
    const size_t primitive_count = context_UUIDs.size();
    const size_t layer_count = 2; // the last source slot (the sky and other surfaces) and the emitted radiation
    vertex_values.assign(2 * layer_count * smoothing_vertex_count * band_count, 0.f);
    if (smoothing_vertex_count == 0) {
        return;
    }
    if (smoothing_vertex_indices.size() != 4 * primitive_count || smoothing_primitive_area.size() != primitive_count || material_data.palette_offset.size() != primitive_count) {
        helios_runtime_error("ERROR (RadiationModel::accumulateSpectralIncidentAtVertices): The camera flux smoothing topology or the spectral textures describe a different number of primitives than the geometry's " +
                             std::to_string(primitive_count) + ". Call updateGeometry() after changing the geometry.");
    }
    if (spectral_incident.size() != 2 * slot_count * primitive_count * band_count || spectral_emitted.size() != 2 * primitive_count * band_count) {
        helios_runtime_error("ERROR (RadiationModel::accumulateSpectralIncidentAtVertices): The radiation reaching and leaving primitives with a spectral texture has " + std::to_string(spectral_incident.size()) + " and " +
                             std::to_string(spectral_emitted.size()) + " values, but " + std::to_string(2 * slot_count * primitive_count * band_count) + " and " + std::to_string(2 * primitive_count * band_count) + " are required.");
    }
    auto value = [&](size_t face, size_t layer, size_t position, size_t band) {
        return layer == 0 ? spectral_incident[((face * slot_count + slot_count - 1) * primitive_count + position) * band_count + band] : spectral_emitted[(face * primitive_count + position) * band_count + band];
    };
    std::vector<float> textured_area(smoothing_vertex_count, 0.f);
    for (size_t position = 0; position < primitive_count; position++) {
        const float facet_area = smoothing_primitive_area.at(position);
        if (smoothing_vertex_indices.at(4 * position) < 0 || material_data.palette_offset.at(position) == UINT32_MAX || facet_area <= 0.f) {
            continue; // takes no part in smoothing, or has no spectral texture
        }
        for (size_t c = 0; c < 4; c++) {
            const int vertex = smoothing_vertex_indices.at(4 * position + c);
            if (vertex < 0) {
                continue; // a triangle, which has only three corners
            }
            textured_area.at(size_t(vertex)) += facet_area;
            for (size_t face = 0; face < 2; face++) {
                for (size_t layer = 0; layer < layer_count; layer++) {
                    float *destination = &vertex_values[((face * layer_count + layer) * smoothing_vertex_count + size_t(vertex)) * band_count];
                    for (size_t band = 0; band < band_count; band++) {
                        destination[band] += facet_area * value(face, layer, position, band);
                    }
                }
            }
        }
    }
    for (size_t face = 0; face < 2; face++) {
        for (size_t layer = 0; layer < layer_count; layer++) {
            for (size_t vertex = 0; vertex < smoothing_vertex_count; vertex++) {
                if (textured_area[vertex] <= 0.f) {
                    continue; // no primitive with a spectral texture meets here, so no camera reads the slot
                }
                float *destination = &vertex_values[((face * layer_count + layer) * smoothing_vertex_count + vertex) * band_count];
                for (size_t band = 0; band < band_count; band++) {
                    destination[band] /= textured_area[vertex];
                }
            }
        }
    }
}

void RadiationModel::sumCameraWhiteReference(RadiationCamera &camera, uint camera_index, const std::vector<std::string> &band_labels, const std::vector<float> &white_reference_top_cam,
                                             const std::vector<float> &white_reference_bottom_cam) const {
    const size_t Nprimitives = context_UUIDs.size();
    const size_t Nbands_launch = band_labels.size();
    const size_t camera_block_offset = camera_index * Nprimitives * Nbands_launch;
    if (white_reference_top_cam.size() < camera_block_offset + Nprimitives * Nbands_launch || white_reference_bottom_cam.size() != white_reference_top_cam.size()) {
        helios_runtime_error("ERROR (RadiationModel::sumCameraWhiteReference): The white reference buffers hold " + std::to_string(white_reference_top_cam.size()) + " values, too few for camera '" + camera.label + "' at position " +
                             std::to_string(camera_index) + " with " + std::to_string(Nprimitives) + " primitives and " + std::to_string(Nbands_launch) + " launched band(s).");
    }

    // Number of pixels whose centre ray hit each primitive. Labels are UUID+1, with 0 for the sky; the bounding-box primitives of a periodic boundary are not surfaces and have positions past the last
    // primitive.
    std::vector<size_t> pixel_count(Nprimitives, 0);
    for (uint label: camera.pixel_label_UUID) {
        if (label == 0) {
            continue;
        }
        const uint UUID = label - 1;
        if (UUID >= geometry_data.primitive_positions.size()) {
            helios_runtime_error("ERROR (RadiationModel::sumCameraWhiteReference): Camera '" + camera.label + "' has a pixel labelled with primitive UUID " + std::to_string(UUID) + ", which is not in the ray-tracing geometry.");
        }
        const uint position = geometry_data.primitive_positions.at(UUID);
        if (position < Nprimitives) {
            pixel_count.at(position)++;
        }
    }

    std::vector<double> band_totals(Nbands_launch, 0.0);
    for (size_t position = 0; position < Nprimitives; position++) {
        if (pixel_count.at(position) == 0) {
            continue;
        }
        const uint UUID = context_UUIDs.at(position);
        const std::vector<helios::vec3> vertices = context->getPrimitiveVertices(UUID);
        helios::vec3 centroid;
        for (const auto &vertex: vertices) {
            centroid += vertex;
        }
        centroid = centroid / float(vertices.size());
        const bool camera_sees_top_face = context->getPrimitiveNormal(UUID) * (camera.position - centroid) > 0.f;
        const std::vector<float> &face_white_reference = camera_sees_top_face ? white_reference_top_cam : white_reference_bottom_cam;
        for (size_t b = 0; b < Nbands_launch; b++) {
            band_totals.at(b) += double(pixel_count.at(position)) * face_white_reference.at(camera_block_offset + position * Nbands_launch + b);
        }
    }

    for (size_t b = 0; b < Nbands_launch; b++) {
        if (std::find(camera.band_labels.begin(), camera.band_labels.end(), band_labels.at(b)) != camera.band_labels.end()) {
            camera.white_reference_band_totals[band_labels.at(b)] = band_totals.at(b);
        }
    }
}

void RadiationModel::updateGeometry(const std::vector<uint> &UUIDs) {

    if (message_flag) {
        std::cout << "Updating geometry in radiation transport model..." << std::flush;
    }

    // Upload geometry through backend abstraction layer
    buildGeometryData(UUIDs);
    buildUUIDMapping(); // Build UUID↔position mapping for efficient indexing

    // CRITICAL: context_UUIDs must match GPU buffer ordering (primitive_UUIDs_ordered)
    // Emission data is indexed by position, which corresponds to primitive_UUIDs order
    context_UUIDs = geometry_data.primitive_UUIDs;

    // The smoothing topology names primitives by their position in the buffer just built, so it is only meaningful for this ordering and must be rebuilt whenever the ordering is.
    buildCameraFluxSmoothingTopology();

    backend->updateGeometry(geometry_data);
    backend->buildAccelerationStructure();

    radiativepropertiesneedupdate = true;
    isgeometryinitialized = true;

    // Record the Context state this build corresponds to, so that geometry added, deleted, or modified later
    // is detected by needsGeometryRebuild(). A build covering fewer primitives than the Context
    // holds is an explicit user-specified subset, which must not be silently overwritten.
    geometry_build_primitive_count = context->getPrimitiveCount();
    geometry_build_shape_revision = context->getShapeRevision();
    isgeometrysubset = UUIDs.size() < geometry_build_primitive_count;

    if (message_flag) {
        std::cout << "done." << std::endl;
    }
}

void RadiationModel::updateRadiativeProperties() {

    // Possible scenarios for specifying a primitive's radiative properties
    // 1. If primitive data of form reflectivity_band/transmissivity_band is given, this value is used and overrides any other option.
    // 2. If primitive data of form reflectivity_spectrum/transmissivity_spectrum is given that references global data containing spectral reflectivity/transmissivity:
    //    2a. If radiation source spectrum was not given, assume source spectral intensity is constant over band and calculate using primitive spectrum
    //    2b. If radiation source spectrum was given, calculate using both source and primitive spectrum.

    // Create warning aggregator
    helios::WarningAggregator warnings;
    warnings.setEnabled(message_flag);

    if (message_flag) {
        std::cout << "Updating radiative properties..." << std::flush;
    }

    uint Nbands = radiation_bands.size(); // number of radiative bands
    uint Nsources = radiation_sources.size();
    uint Ncameras = cameras.size();
    size_t Nobjects = primitiveID.size();
    size_t Nprimitives = context_UUIDs.size();

    scattering_iterations_needed.clear();
    for (auto &band: radiation_bands) {
        scattering_iterations_needed[band.first] = false;
    }
    spectral_weighting_errors.clear();

    float eps;

    std::string prop;
    std::vector<std::string> band_labels;
    for (auto &band: radiation_bands) {
        band_labels.push_back(band.first);
    }

    // Allocate flat arrays directly in material_data to avoid nested vector overhead and redundant copies. There is one
    // material slot per radiation source, read by that source's direct rays, plus one diffuse/scatter slot at index Nsources,
    // read by diffuse, scattered and emitted rays (see the diffuse-slot weights below).
    material_data.num_primitives = Nprimitives;
    material_data.num_bands = Nbands;
    material_data.num_sources = Nsources;
    material_data.num_cameras = Ncameras;

    const uint Nslots = Nsources + 1;
    const uint diffuse_slot = Nsources;

    size_t mat_size = (size_t) Nslots * Nprimitives * Nbands;
    material_data.reflectivity.assign(mat_size, rho_default);
    material_data.transmissivity.assign(mat_size, tau_default);

    // Translucent cover (glass/plastic) material arrays. Default: not glass (is_glass=0), lossless (KL=0).
    material_data.glass_n.assign(mat_size, 0.f);
    material_data.glass_KL.assign(mat_size, 0.f);
    material_data.is_glass.assign(mat_size, 0);

    if (Ncameras > 0) {
        size_t cam_size = (size_t) Nslots * Nprimitives * Nbands * Ncameras;
        material_data.reflectivity_cam.assign(cam_size, rho_default);
        material_data.transmissivity_cam.assign(cam_size, tau_default);
    } else {
        material_data.reflectivity_cam.clear();
        material_data.transmissivity_cam.clear();
    }

    MaterialPropertyIndexer mat_idx(Nslots, Nprimitives, Nbands);
    CameraMaterialIndexer cam_idx(Nslots, Nprimitives, Nbands, Ncameras);

    // Cache all unique camera spectral responses for all cameras and bands
    std::vector<std::vector<std::vector<helios::vec2>>> camera_response_unique;
    camera_response_unique.resize(Ncameras);
    if (Ncameras > 0) {
        uint cam = 0;
        for (const auto &camera: cameras) {

            camera_response_unique.at(cam).resize(Nbands);

            for (uint b = 0; b < Nbands; b++) {

                if (camera.second.band_spectral_response.find(band_labels.at(b)) == camera.second.band_spectral_response.end()) {
                    continue;
                }

                std::string camera_response = camera.second.band_spectral_response.at(band_labels.at(b));

                // "uniform" (or no response) means a flat response. Any other name must be a spectrum: loadSpectralData() loads it from the radiation plug-in's
                // spectral libraries if it is not already in the Context, and raises an error naming it if it is found nowhere or is not a valid spectrum.
                if (!camera_response.empty() && camera_response != "uniform") {
                    camera_response_unique.at(cam).at(b) = loadSpectralData(camera_response);
                }
            }
            cam++;
        }
    }

    // A camera-weighted value is normalized by the integral of the camera response (see cameraWeightedValue below), so a response must have some
    // sensitivity somewhere.
    for (uint cam = 0; cam < Ncameras; cam++) {
        for (uint b = 0; b < Nbands; b++) {
            const std::vector<helios::vec2> &response = camera_response_unique.at(cam).at(b);
            if (!response.empty() && (response.size() < 2 || integrateSpectrum(response) <= 0.f)) {
                helios_runtime_error("ERROR (RadiationModel::updateRadiativeProperties): The spectral response of camera #" + std::to_string(cam) + " in band '" + band_labels.at(b) +
                                     "' is zero everywhere (or has fewer than 2 wavelengths), so the camera cannot see anything in that band. Check the response data set with setCameraSpectralResponse().");
            }
        }
    }

    // Spectral weighting of reflectivity and transmissivity spectra. The radiation arriving in a band comes from "weighting contributors": each
    // radiation source, then the sky diffuse radiation, then a spectrally flat weighting that stands in when a band receives no source or sky
    // radiation at all (e.g. an emission-only band). A contributor's spectrum is its source spectrum or the band's diffuse spectrum; one without a
    // spectrum is spectrally flat. Every spectrum is zero outside its tabulated range.
    //
    // Direct rays use their own source's weighting. Diffuse, scattered and emitted rays use the diffuse slot, weighted by the band's combined
    // incident spectrum W(λ) = Σ_c F_c Ŝ_c(λ), where F_c is contributor c's band flux and Ŝ_c its spectrum normalized to integrate to 1 over the
    // band. Since ∫ρW / ∫W = Σ_c F_c ρ_c / Σ_c F_c, with ρ_c the value weighted by contributor c alone, the diffuse slot is the flux-weighted mean of
    // the per-contributor values. This is uniform over the scene: a shaded surface gets the same diffuse-slot value as a sunlit one.
    const uint sky_contributor = Nsources;
    const uint flat_contributor = Nsources + 1;
    const uint Ncontributors = Nsources + 2;
    const std::vector<helios::vec2> flat_weighting;
    auto contributorSpectrum = [&](uint contributor, uint b) -> const std::vector<helios::vec2> & {
        if (contributor < Nsources) {
            return radiation_sources.at(contributor).source_spectrum;
        } else if (contributor == sky_contributor && !radiation_bands.at(band_labels.at(b)).emissionFlag) {
            // In an emitting (thermal) band the diffuse spectrum does not describe the sky radiation (see getDiffuseFlux()), so it is flat there
            return radiation_bands.at(band_labels.at(b)).diffuse_spectrum;
        }
        return flat_weighting;
    };
    // Contributors with the same spectrum (e.g. every source without one) share their integrals through the cache below
    auto contributorSpectrumID = [&](uint contributor, uint b) -> std::string {
        if (contributorSpectrum(contributor, b).empty()) {
            return "flat";
        }
        return contributor < Nsources ? "source" + std::to_string(contributor) : "sky";
    };

    std::vector<helios::vec2> band_bounds(Nbands);
    std::vector<bool> band_has_bounds(Nbands);
    // Whether the contributor's spectrum has any energy within the band. Only bands with wavelength bounds are integrated, and a flat weighting always has energy.
    std::vector<std::vector<bool>> contributor_has_band_energy(Nbands, std::vector<bool>(Ncontributors, true));
    // A contributor with flux in a band where its spectrum has no energy leaves nothing to weight a spectrum by. This is raised when a primitive's
    // properties in that band are weighted by spectra; with only constant properties, the contributor's spectrum is never used.
    std::vector<std::string> band_weighting_error(Nbands);
    // Weight of each contributor in a band's diffuse slot: its band flux, normalized so the weights sum to 1
    std::vector<std::vector<float>> diffuse_slot_weight(Nbands, std::vector<float>(Ncontributors, 0.f));
    // Whether any contributor with flux in the band has a spectrum. Camera-weighted values of spectrally flat contributors are then normalized
    // the same way as those of contributors with a spectrum (see cameraWeightedValue), so the two can be mixed in one image and one diffuse slot.
    std::vector<bool> band_has_spectral_flux(Nbands, false);
    for (uint b = 0; b < Nbands; b++) {
        const std::string &band = band_labels.at(b);
        band_bounds.at(b) = radiation_bands.at(band).wavebandBounds;
        band_has_bounds.at(b) = radiation_bands.at(band).hasWavebandBounds();
        const std::string band_description = "band '" + band + "'" + (band_has_bounds.at(b) ? " (" + wavelengthRangeString(band_bounds.at(b).x, band_bounds.at(b).y) + ")" : "");

        std::vector<float> band_flux(flat_contributor);
        for (uint c = 0; c < flat_contributor; c++) {
            band_flux.at(c) = c < Nsources ? getSourceFlux(c, band) : getDiffuseFlux(band);
        }

        if (band_has_bounds.at(b)) {
            for (uint c = 0; c < flat_contributor; c++) {
                const std::vector<helios::vec2> &spectrum = contributorSpectrum(c, b);
                if (spectrum.empty() || integrateSpectrum(spectrum, band_bounds.at(b).x, band_bounds.at(b).y) > 0.f) {
                    continue;
                }
                contributor_has_band_energy.at(b).at(c) = false;
                if (band_flux.at(c) > 0.f && band_weighting_error.at(b).empty()) {
                    if (c < Nsources) {
                        band_weighting_error.at(b) = "ERROR (RadiationModel::updateRadiativeProperties): Radiation source " + std::to_string(c) + " has a flux of " + std::to_string(band_flux.at(c)) + " W/m^2 in " + band_description +
                                                     ", but its spectrum (tabulated " + spectrumRangeString(spectrum) + ") has no energy in that band, so surface reflectivity and transmissivity spectra cannot be weighted by it. " +
                                                     (isSIFExcitationBand(band) ? "This is an internal SIF excitation band, whose source fluxes are taken from the source spectra when the SIF camera is added; the source's spectrum was changed "
                                                                                  "afterwards. Set the source spectrum before adding the SIF camera."
                                                                                : "Either set the source's flux in this band to 0 using setSourceFlux(), or give the source a spectrum that covers the band.");
                    } else {
                        band_weighting_error.at(b) =
                                "ERROR (RadiationModel::updateRadiativeProperties): The diffuse radiation flux in " + band_description + " is " + std::to_string(band_flux.at(c)) + " W/m^2, but the diffuse spectrum (tabulated " +
                                spectrumRangeString(spectrum) +
                                ") has no energy in that band, so surface reflectivity and transmissivity spectra cannot be weighted by it. Either set the band's diffuse flux to 0 using setDiffuseRadiationFlux(), or set a diffuse "
                                "spectrum that covers the band using setDiffuseSpectrum().";
                    }
                }
            }
        }

        float total_flux = 0.f;
        for (uint c = 0; c < flat_contributor; c++) {
            if (band_flux.at(c) > 0.f) {
                total_flux += band_flux.at(c);
                if (!contributorSpectrum(c, b).empty()) {
                    band_has_spectral_flux.at(b) = true;
                }
            }
        }
        if (total_flux > 0.f) {
            for (uint c = 0; c < flat_contributor; c++) {
                if (band_flux.at(c) > 0.f) {
                    diffuse_slot_weight.at(b).at(c) = band_flux.at(c) / total_flux;
                }
            }
        } else {
            // No source or sky radiation in the band: its diffuse slot carries only emission, and ρ/τ are the unweighted band averages
            diffuse_slot_weight.at(b).at(flat_contributor) = 1.f;
        }
    }

    // Flux-weighted mean of per-contributor values for a band's diffuse slot
    auto diffuseSlotValue = [&](uint b, const std::function<float(uint)> &contributor_value) -> float {
        float value = 0.f;
        for (uint c = 0; c < Ncontributors; c++) {
            if (diffuse_slot_weight.at(b).at(c) > 0.f) {
                value += diffuse_slot_weight.at(b).at(c) * contributor_value(c);
            }
        }
        return value;
    };

    // Weighted value of a reflectivity/transmissivity spectrum in a band with wavelength bounds, for one contributor: ∫ρ·W / ∫W over the band,
    // with W the contributor's spectrum (1 for a spectrally flat one). Only used for contributors with energy in the band, so ∫W > 0.
    auto bandWeightedValue = [&](const std::vector<helios::vec2> &object_spectrum, uint b, uint contributor) -> float {
        const WeightedSpectrumIntegral integral = integrateWeightedSpectrum(object_spectrum, contributorSpectrum(contributor, b), {}, band_bounds.at(b).x, band_bounds.at(b).y);
        return integral.weighted / integral.weight;
    };

    // Camera-weighted value of a reflectivity/transmissivity spectrum for one contributor and camera. A contributor with a spectrum W gives
    // ∫ρ·W·C / ∫W over the wavelengths where both W and the camera response C are tabulated, which keeps the magnitude of the response. A
    // spectrally flat contributor is taken as a unit spectrum over the response, ∫ρ·C / ∫dλ, when the band has any flux with a spectrum, so
    // that it is on the same scale; in a band where no flux has a spectrum it gives the camera-weighted average ∫ρ·C / ∫C, whose white
    // reference is 1. An empty object spectrum stands for a spectrally flat, perfectly white surface, whose value is the white reference.
    auto cameraWeightedValue = [&](const std::vector<helios::vec2> &object_spectrum, uint b, uint contributor, uint cam) -> float {
        const std::vector<helios::vec2> &response = camera_response_unique.at(cam).at(b);
        const std::vector<helios::vec2> &weighting = contributorSpectrum(contributor, b);
        if (weighting.empty()) {
            const WeightedSpectrumIntegral integral =
                    band_has_spectral_flux.at(b) ? integrateWeightedSpectrum(object_spectrum, {}, response, response.front().x, response.back().x) : integrateWeightedSpectrum(object_spectrum, response, {}, response.front().x, response.back().x);
            return integral.weighted / integral.weight; // responses with fewer than 2 points or all zeros are rejected above, so integral.weight > 0
        }
        const float wavelength1 = std::max(weighting.front().x, response.front().x);
        const float wavelength2 = std::min(weighting.back().x, response.back().x);
        const WeightedSpectrumIntegral integral = wavelength2 > wavelength1 ? integrateWeightedSpectrum(object_spectrum, weighting, response, wavelength1, wavelength2) : WeightedSpectrumIntegral{};
        if (integral.weight <= 0.f) {
            // The camera responds to none of this contributor's radiation, so every camera-weighted quantity it produces, including the
            // white reference, is genuinely zero.
            return 0.f;
        }
        return integral.weighted / integral.weight;
    };

    // Camera-weighted reflectivity of a spectrally flat, perfectly white surface, per [slot][band][camera]: the value rho_cam below takes for a
    // reflectivity spectrum of 1. Without a camera response it is 1.
    if (Ncameras > 0) {
        material_data.white_reference_cam.assign((size_t) Nslots * Nbands * Ncameras, 1.f);
        std::vector<float> contributor_white_reference(Ncontributors);
        for (uint b = 0; b < Nbands; b++) {
            for (uint cam = 0; cam < Ncameras; cam++) {
                if (camera_response_unique.at(cam).at(b).empty()) {
                    continue;
                }
                for (uint c = 0; c < Ncontributors; c++) {
                    contributor_white_reference.at(c) = cameraWeightedValue({}, b, c, cam);
                }
                for (uint s = 0; s < Nsources; s++) {
                    material_data.white_reference_cam.at(((size_t) s * Nbands + b) * Ncameras + cam) = contributor_white_reference.at(s);
                }
                material_data.white_reference_cam.at(((size_t) diffuse_slot * Nbands + b) * Ncameras + cam) = diffuseSlotValue(b, [&](uint c) { return contributor_white_reference.at(c); });
            }
        }
    } else {
        material_data.white_reference_cam.clear();
    }

    // Spectral integration cache to avoid redundant computations
    std::unordered_map<std::string, float> spectral_integration_cache;

    // Look up a spectral integral in the cache, computing and storing it on a miss (thread-safe)
    auto cachedSpectralValue = [&](const std::string &cache_key, const std::function<float()> &compute) -> float {
        bool found = false;
        float result = 0.f;
#ifdef USE_OPENMP
#pragma omp critical(radiation_spectral_cache)
#endif
        {
            auto cache_it = spectral_integration_cache.find(cache_key);
            if (cache_it != spectral_integration_cache.end()) {
                found = true;
                result = cache_it->second;
            }
        }
        if (found) {
            return result;
        }
        result = compute();
#ifdef USE_OPENMP
#pragma omp critical(radiation_spectral_cache)
#endif
        {
            spectral_integration_cache[cache_key] = result;
        }
        return result;
    };

    // Apply spectral interpolation based on primitive data values
    for (const auto &config: spectrum_interpolation_configs) {
        // Validate that all spectra in this config exist in global data and have correct type
        for (const auto &spectrum_label: config.spectra_labels) {
            if (!context->doesGlobalDataExist(spectrum_label.c_str())) {
                helios_runtime_error("ERROR (RadiationModel::updateRadiativeProperties): Spectral interpolation config references global data '" + spectrum_label + "' which does not exist.");
            }
            if (context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_VEC2) {
                helios_runtime_error("ERROR (RadiationModel::updateRadiativeProperties): Spectral interpolation config references global data '" + spectrum_label + "' which must be of type HELIOS_TYPE_VEC2 (std::vector<helios::vec2>).");
            }
        }

        for (uint uuid: config.primitive_UUIDs) {
            // Check if primitive still exists in context (it may have been deleted)
            if (!context->doesPrimitiveExist(uuid)) {
                continue;
            }

            // Check if the query data exists for this primitive and has correct type
            if (context->doesPrimitiveDataExist(uuid, config.query_data_label.c_str())) {
                // Check that query data is of type float
                if (context->getPrimitiveDataType(config.query_data_label.c_str()) != helios::HELIOS_TYPE_FLOAT) {
                    helios_runtime_error("ERROR (RadiationModel::updateRadiativeProperties): Primitive data '" + config.query_data_label + "' for UUID " + std::to_string(uuid) + " must be of type HELIOS_TYPE_FLOAT for spectral interpolation.");
                }

                // Get the query value
                float query_value;
                context->getPrimitiveData(uuid, config.query_data_label.c_str(), query_value);

                // Perform nearest-neighbor interpolation
                size_t nearest_idx = 0;
                float min_distance = std::abs(query_value - config.mapping_values[0]);
                for (size_t i = 1; i < config.mapping_values.size(); i++) {
                    float distance = std::abs(query_value - config.mapping_values[i]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        nearest_idx = i;
                    }
                }

                // Set the target primitive data to the selected spectrum label
                context->setPrimitiveData(uuid, config.target_data_label.c_str(), config.spectra_labels[nearest_idx]);
            }
        }

        // Apply spectral interpolation based on object data values
        for (uint objID: config.object_IDs) {
            // Check if object still exists in context (it may have been deleted)
            if (!context->doesObjectExist(objID)) {
                continue;
            }

            // Check if the query data exists for this object and has correct type
            if (context->doesObjectDataExist(objID, config.query_data_label.c_str())) {
                // Check that query data is of type float
                if (context->getObjectDataType(config.query_data_label.c_str()) != helios::HELIOS_TYPE_FLOAT) {
                    helios_runtime_error("ERROR (RadiationModel::updateRadiativeProperties): Object data '" + config.query_data_label + "' for object ID " + std::to_string(objID) + " must be of type HELIOS_TYPE_FLOAT for spectral interpolation.");
                }

                // Get the query value
                float query_value;
                context->getObjectData(objID, config.query_data_label.c_str(), query_value);

                // Perform nearest-neighbor interpolation
                size_t nearest_idx = 0;
                float min_distance = std::abs(query_value - config.mapping_values.at(0));
                for (size_t i = 1; i < config.mapping_values.size(); i++) {
                    float distance = std::abs(query_value - config.mapping_values.at(i));
                    if (distance < min_distance) {
                        min_distance = distance;
                        nearest_idx = i;
                    }
                }

                // Get object's primitive UUIDs and set their primitive data using vector overload
                std::vector<uint> prim_uuids = context->getObjectPrimitiveUUIDs(objID);
                context->setPrimitiveData(prim_uuids, config.target_data_label.c_str(), config.spectra_labels.at(nearest_idx));
            }
        }
    }

    // Cache all unique primitive reflectivity and transmissivity spectra before assigning to primitives

    // first, figure out all of the spectra referenced by all primitives and store it in "surface_spectra" to avoid having to load it again
    std::map<std::string, std::vector<helios::vec2>> surface_spectra_rho;
    std::map<std::string, std::vector<helios::vec2>> surface_spectra_tau;
    for (size_t u = 0; u < Nprimitives; u++) {

        uint UUID = context_UUIDs.at(u);

        // A primitive with a spectral texture takes its spectra from its palette (collected below), and its own spectrum data are not used
        if (context->primitiveHasSpectralTexture(UUID)) {
            continue;
        }

        if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
            if (context->getPrimitiveDataType("reflectivity_spectrum") == HELIOS_TYPE_STRING) {
                std::string spectrum_label;
                context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);

                // get the spectral reflectivity data and store it in surface_spectra to avoid having to load it again. loadSpectralData() loads a spectrum from the
                // radiation plug-in's spectral libraries if it is not already in the Context, and raises an error naming it if it is found nowhere or is not a valid spectrum.
                if (!spectrum_label.empty() && surface_spectra_rho.find(spectrum_label) == surface_spectra_rho.end()) {
                    surface_spectra_rho.emplace(spectrum_label, loadSpectralData(spectrum_label));
                }
            }
        }

        if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum")) {
            if (context->getPrimitiveDataType("transmissivity_spectrum") == HELIOS_TYPE_STRING) {
                std::string spectrum_label;
                context->getPrimitiveData(UUID, "transmissivity_spectrum", spectrum_label);

                // get the spectral transmissivity data and store it in surface_spectra to avoid having to load it again (see the reflectivity above)
                if (!spectrum_label.empty() && surface_spectra_tau.find(spectrum_label) == surface_spectra_tau.end()) {
                    surface_spectra_tau.emplace(spectrum_label, loadSpectralData(spectrum_label));
                }
            }
        }
    }

    // The palettes of primitives with a spectral texture, laid end to end (see RayTracingMaterial::palette_offset). Their spectra are
    // integrated with those of the primitives, so a class has exactly the values a primitive with the same spectra has.
    std::map<std::string, uint32_t> palette_first_entry;
    std::vector<const helios::SpectralTextureClass *> palette_entries;
    std::vector<uint32_t> primitive_palette_offset(Nprimitives, UINT32_MAX);
    auto loadPaletteSpectrum = [&](const std::string &spectrum_label, const std::string &palette_label, std::map<std::string, std::vector<helios::vec2>> &spectra) {
        if (spectrum_label.empty() || spectra.find(spectrum_label) != spectra.end()) {
            return;
        }
        try {
            spectra.emplace(spectrum_label, loadSpectralData(spectrum_label));
        } catch (const std::runtime_error &error) {
            helios_runtime_error("ERROR (RadiationModel::runBand): Spectral palette '" + palette_label + "' of a primitive's spectral texture names the spectrum '" + spectrum_label + "', which could not be used: " + error.what());
        }
    };
    for (size_t u = 0; u < Nprimitives; u++) {
        const uint UUID = context_UUIDs.at(u);
        if (!context->primitiveHasSpectralTexture(UUID)) {
            continue;
        }
        // The class map is part of the geometry. A geometry built from a list of primitives is not rebuilt automatically, so it can be out of date.
        const std::string class_map_file = context->getPrimitiveSpectralTextureFile(UUID);
        const int class_map_ID = geometry_data.class_map_IDs.at(u);
        if (class_map_ID < 0 || spectral_class_map_files.at(class_map_ID) != class_map_file) {
            helios_runtime_error("ERROR (RadiationModel::runBand): Primitive UUID " + std::to_string(UUID) + " has a spectral texture (class map '" + class_map_file +
                                 "') that it did not have when the radiation model's geometry was built. Call RadiationModel::updateGeometry() again after giving primitives spectral textures.");
        }
        const std::string palette_label = context->getPrimitiveSpectralPalette(UUID);
        auto palette_it = palette_first_entry.find(palette_label);
        if (palette_it == palette_first_entry.end()) {
            palette_it = palette_first_entry.emplace(palette_label, uint32_t(palette_entries.size())).first;
            for (const helios::SpectralTextureClass &spectral_class: context->getSpectralPalette(palette_label)) {
                palette_entries.push_back(&spectral_class);
                loadPaletteSpectrum(spectral_class.reflectivity_spectrum, palette_label, surface_spectra_rho);
                loadPaletteSpectrum(spectral_class.transmissivity_spectrum, palette_label, surface_spectra_tau);
            }
        }
        primitive_palette_offset.at(u) = palette_it->second;
    }

    // second, calculate unique values of rho and tau for every band and weighting contributor: rho_unique[label][band][contributor] and
    // rho_cam_unique[label][band][contributor][camera]. The entry after the last contributor holds the band's diffuse-slot value.
    const uint unique_diffuse_index = Ncontributors;
    std::map<std::string, std::vector<std::vector<float>>> rho_unique;
    std::map<std::string, std::vector<std::vector<float>>> tau_unique;

    std::map<std::string, std::vector<std::vector<std::vector<float>>>> rho_cam_unique;
    std::map<std::string, std::vector<std::vector<std::vector<float>>>> tau_cam_unique;

    // A spectrum that does not overlap a band at all has no defined value in it: [label][band], raised only if a primitive uses the spectrum there
    std::map<std::string, std::vector<std::string>> rho_band_error;
    std::map<std::string, std::vector<std::string>> tau_band_error;

    auto computeUniqueValues = [&](const std::string &spectrum_label, const std::vector<helios::vec2> &spectrum, const std::string &property, float default_value, std::vector<std::vector<float>> &values,
                                   std::vector<std::vector<std::vector<float>>> &cam_values, std::vector<std::string> &band_errors) {
        values.assign(Nbands, std::vector<float>(Ncontributors + 1, default_value));
        if (Ncameras > 0) {
            cam_values.assign(Nbands, std::vector<std::vector<float>>(Ncontributors + 1, std::vector<float>(Ncameras, default_value)));
        }
        band_errors.assign(Nbands, "");
        if (spectrum.empty()) {
            return; // the referenced global data does not exist; this was warned about above and the default value is used
        }

        for (uint b = 0; b < Nbands; b++) {
            const std::string &band = band_labels.at(b);

            // The internal SIF excitation bands tile a fixed wavelength range in narrow bins, so bins beyond the end of a surface spectrum are
            // expected; like the part of any band a spectrum does not cover, they take the spectrum as zero rather than being an error.
            if (band_has_bounds.at(b) && !isSIFExcitationBand(band) && (spectrum.back().x <= band_bounds.at(b).x || spectrum.front().x >= band_bounds.at(b).y)) {
                band_errors.at(b) = "ERROR (RadiationModel::updateRadiativeProperties): The " + property + " spectrum '" + spectrum_label + "' is tabulated from " + spectrumRangeString(spectrum) + ", which does not overlap band '" + band + "' (" +
                                    wavelengthRangeString(band_bounds.at(b).x, band_bounds.at(b).y) + "), so its " + property +
                                    " in that band is undefined. Either extend the spectrum to cover the band, or set a constant value for the band with primitive data \"" + property + "_" + band + "\".";
                continue;
            }

            for (uint c = 0; c < Ncontributors; c++) {
                if (!contributor_has_band_energy.at(b).at(c)) {
                    // Nothing to weight by. A contributor with flux here is rejected (band_weighting_error), so this one has no flux in the band: its
                    // direct rays carry no energy and it has no weight in the diffuse slot. The value is never used, so it is not computed.
                    continue;
                }
                const std::string contributor_id = contributorSpectrumID(c, b);

                float &value = values.at(b).at(c);
                if (band_has_bounds.at(b)) {
                    value = cachedSpectralValue(spectrum_label + "|" + std::to_string(b) + "|" + contributor_id, [&]() { return bandWeightedValue(spectrum, b, c); });
                }
                // Without wavelength bounds a spectrum cannot be integrated over the band, so the default stands unless the cameras give a value below

                if (Ncameras > 0) {
                    float normalized_camera_sum = 0.f;
                    uint Nresponding_cameras = 0;
                    for (uint cam = 0; cam < Ncameras; cam++) {
                        if (camera_response_unique.at(cam).at(b).empty()) {
                            cam_values.at(b).at(c).at(cam) = value;
                        } else {
                            const float camera_value = cachedSpectralValue(spectrum_label + "|" + std::to_string(b) + "|" + contributor_id + "|cam" + std::to_string(cam), [&]() { return cameraWeightedValue(spectrum, b, c, cam); });
                            cam_values.at(b).at(c).at(cam) = camera_value;
                            const float white_reference = cachedSpectralValue("white|" + std::to_string(b) + "|" + contributor_id + "|cam" + std::to_string(cam), [&]() { return cameraWeightedValue({}, b, c, cam); });
                            if (white_reference > 0.f) {
                                normalized_camera_sum += camera_value / white_reference;
                                Nresponding_cameras++;
                            }
                        }
                    }

                    // Without wavelength bounds, the camera-weighted value averaged over the cameras stands in for the band value. This lets regular
                    // scattering work when only reflectivity and transmissivity spectra and camera responses are given. A camera-weighted value keeps
                    // the magnitude of the response, so it is divided by the white reference to give a reflectivity or transmissivity.
                    if (!band_has_bounds.at(b) && value == default_value && Nresponding_cameras > 0) {
                        value = normalized_camera_sum / float(Nresponding_cameras);
                    }
                }
            }

            values.at(b).at(unique_diffuse_index) = diffuseSlotValue(b, [&](uint c) { return values.at(b).at(c); });
            for (uint cam = 0; cam < Ncameras; cam++) {
                cam_values.at(b).at(unique_diffuse_index).at(cam) = diffuseSlotValue(b, [&](uint c) { return cam_values.at(b).at(c).at(cam); });
            }
        }
    };

    // Convert maps to vectors for OpenMP indexing, and create every map entry before the parallel loops so the maps are not modified inside them
    std::vector<std::pair<std::string, std::vector<helios::vec2>>> spectra_rho_vector(surface_spectra_rho.begin(), surface_spectra_rho.end());
    for (const auto &spectrum: spectra_rho_vector) {
        rho_unique[spectrum.first];
        rho_cam_unique[spectrum.first];
        rho_band_error[spectrum.first];
    }
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int spectrum_idx = 0; spectrum_idx < (int) spectra_rho_vector.size(); spectrum_idx++) {
        const auto &spectrum = spectra_rho_vector[spectrum_idx];
        computeUniqueValues(spectrum.first, spectrum.second, "reflectivity", rho_default, rho_unique.at(spectrum.first), rho_cam_unique.at(spectrum.first), rho_band_error.at(spectrum.first));
    }

    std::vector<std::pair<std::string, std::vector<helios::vec2>>> spectra_tau_vector(surface_spectra_tau.begin(), surface_spectra_tau.end());
    for (const auto &spectrum: spectra_tau_vector) {
        tau_unique[spectrum.first];
        tau_cam_unique[spectrum.first];
        tau_band_error[spectrum.first];
    }
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int spectrum_idx = 0; spectrum_idx < (int) spectra_tau_vector.size(); spectrum_idx++) {
        const auto &spectrum = spectra_tau_vector[spectrum_idx];
        computeUniqueValues(spectrum.first, spectrum.second, "transmissivity", tau_default, tau_unique.at(spectrum.first), tau_cam_unique.at(spectrum.first), tau_band_error.at(spectrum.first));
    }

    // Palette tables: the value of each class in each slot and band, from the same unique values the primitives use, clamped to [0, 1]
    // as a primitive's are. A class without a transmissivity spectrum transmits nothing.
    const size_t Nentries = palette_entries.size();
    material_data.num_palette_entries = Nentries;
    material_data.palette_offset = primitive_palette_offset;
    material_data.palette_reflectivity.assign((size_t) Nslots * Nentries * Nbands, 0.f);
    material_data.palette_transmissivity.assign((size_t) Nslots * Nentries * Nbands, 0.f);
    material_data.palette_reflectivity_cam.assign((size_t) Nslots * Nentries * Nbands * Ncameras, 0.f);
    material_data.palette_transmissivity_cam.assign((size_t) Nslots * Nentries * Nbands * Ncameras, 0.f);
    MaterialPropertyIndexer palette_idx(Nslots, Nentries, Nbands);
    CameraMaterialIndexer palette_cam_idx(Nslots, Nentries, Nbands, Ncameras);
    material_data.palette_band_active.assign(Nbands, 1);
    {
        uint b = 0;
        for (const auto &band: band_labels) {
            // A band with emission uses each primitive's emissivity, so its rays use the primitive's own (mean) values; the classes must then agree in it (checked below)
            if (radiation_bands.at(band).emissionFlag && sif_emission_bands.count(band) == 0) {
                material_data.palette_band_active.at(b) = 0;
            }
            for (size_t entry = 0; entry < Nentries; entry++) {
                const helios::SpectralTextureClass &spectral_class = *palette_entries.at(entry);
                for (const auto *property: {&spectral_class.reflectivity_spectrum, &spectral_class.transmissivity_spectrum}) {
                    const bool is_rho = property == &spectral_class.reflectivity_spectrum;
                    const std::string &label = *property;
                    if (label.empty()) {
                        continue;
                    }
                    for (uint slot = 0; slot < Nslots; slot++) {
                        const uint unique_index = slot == diffuse_slot ? unique_diffuse_index : slot;
                        float value = (is_rho ? rho_unique : tau_unique).at(label).at(b).at(unique_index);
                        if (value < 0.f || value > 1.f) {
                            warnings.addWarning(is_rho ? "reflectivity_palette_clamped" : "transmissivity_palette_clamped",
                                                std::string(is_rho ? "Reflectivity" : "Transmissivity") + " of spectrum '" + label + "' in a spectral palette must be between 0 and 1. Clamping it for band " + band + ".");
                            value = std::clamp(value, 0.f, 1.f);
                        }
                        (is_rho ? material_data.palette_reflectivity : material_data.palette_transmissivity)[palette_idx(slot, entry, b)] = value;
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            (is_rho ? material_data.palette_reflectivity_cam : material_data.palette_transmissivity_cam)[palette_cam_idx(slot, entry, b, cam)] = (is_rho ? rho_cam_unique : tau_cam_unique).at(label).at(b).at(unique_index).at(cam);
                        }
                    }
                }
            }
            b++;
        }
    }

    // Area-weighted mean over the classes of primitive u's class map of a palette table, held as the primitive's own value. With one
    // class the class's value is used as it is, so that it is bit-identical to a primitive given the same spectra.
    auto paletteMean = [&](size_t u, const std::function<float(size_t entry)> &entry_value) -> float {
        const std::vector<std::pair<uint16_t, float>> &fractions = spectral_class_fractions.at(u);
        if (fractions.size() == 1) {
            return entry_value(primitive_palette_offset.at(u) + fractions.front().first - 1);
        }
        double mean = 0;
        for (const auto &fraction: fractions) {
            mean += double(fraction.second) * entry_value(primitive_palette_offset.at(u) + fraction.first - 1);
        }
        return float(mean);
    };

    // Spectral weighting errors of the classes primitive u covers, recorded as for a primitive's own spectra (runBand() raises them only if the band is run)
    auto recordPaletteBandErrors = [&](size_t u, uint b, const std::string &band, bool is_rho) {
        for (const auto &fraction: spectral_class_fractions.at(u)) {
            const helios::SpectralTextureClass &spectral_class = *palette_entries.at(primitive_palette_offset.at(u) + fraction.first - 1);
            const std::string &label = is_rho ? spectral_class.reflectivity_spectrum : spectral_class.transmissivity_spectrum;
            if (label.empty()) {
                continue;
            }
            const std::string &band_error = (is_rho ? rho_band_error : tau_band_error).at(label).at(b);
            const std::string &error = !band_error.empty() ? band_error : band_weighting_error.at(b);
            if (!error.empty()) {
                spectral_weighting_errors.emplace(band, error);
            }
        }
    };

    for (size_t u = 0; u < Nprimitives; u++) {

        uint UUID = context_UUIDs.at(u);

        helios::PrimitiveType type = context->getPrimitiveType(UUID);

        if (type == helios::PRIMITIVE_TYPE_VOXEL) {

        } else { // other than voxels

            // A primitive with a spectral texture takes its spectra from its palette, ahead of its own spectrum data
            const bool has_spectral_texture = primitive_palette_offset.at(u) != UINT32_MAX;
            auto rejectConstantProperty = [&](const std::string &label) {
                helios_runtime_error("ERROR (RadiationModel::runBand): Primitive UUID " + std::to_string(UUID) + " has a spectral texture and also the primitive data '" + label + "', which would set one value for the whole primitive. Remove '" +
                                     label + "', or remove the spectral texture with Context::clearPrimitiveSpectralTexture().");
            };

            // Reflectivity

            // check for primitive data of form "reflectivity_spectrum" that can be used to calculate reflectivity
            std::string spectrum_label;
            if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
                if (context->getPrimitiveDataType("reflectivity_spectrum") == HELIOS_TYPE_STRING) {
                    context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);
                }
            }

            uint b = 0;
            for (const auto &band: band_labels) {

                // check for primitive data of form "reflectivity_bandname"
                prop = "reflectivity_" + band;

                float rho_s = rho_default;
                const bool has_band_rho = context->doesPrimitiveDataExist(UUID, prop.c_str());
                if (has_band_rho) {
                    // Where the palette applies, a constant would hide the texture. Elsewhere (bands with emission) rays use the primitive's own values, which a constant may set.
                    if (has_spectral_texture && material_data.palette_band_active.at(b) != 0) {
                        rejectConstantProperty(prop);
                    }
                    context->getPrimitiveData(UUID, prop.c_str(), rho_s);
                }
                const bool rho_from_palette = has_spectral_texture && !has_band_rho;
                if (rho_from_palette) {
                    recordPaletteBandErrors(u, b, band, true);
                }

                // A value set for the band (whatever its value, including 0), or no spectrum, applies to every slot; otherwise the spectrum's weighted values are used
                const bool use_constant_rho = has_band_rho || spectrum_label.empty() || !context->doesGlobalDataExist(spectrum_label.c_str()) || rho_unique.find(spectrum_label) == rho_unique.end();
                if (!use_constant_rho && !has_spectral_texture) {
                    // Recorded rather than raised here: runBand() raises it only if this band is run
                    const std::string &error = !rho_band_error.at(spectrum_label).at(b).empty() ? rho_band_error.at(spectrum_label).at(b) : band_weighting_error.at(b);
                    if (!error.empty()) {
                        spectral_weighting_errors.emplace(band, error);
                    }
                }

                for (uint slot = 0; slot < Nslots; slot++) {
                    float &rho_val = material_data.reflectivity[mat_idx(slot, u, b)];
                    const uint unique_index = slot == diffuse_slot ? unique_diffuse_index : slot;

                    if (rho_from_palette) {
                        rho_val = paletteMean(u, [&](size_t entry) { return material_data.palette_reflectivity[palette_idx(slot, entry, b)]; });
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.reflectivity_cam[cam_idx(slot, u, b, cam)] = paletteMean(u, [&](size_t entry) { return material_data.palette_reflectivity_cam[palette_cam_idx(slot, entry, b, cam)]; });
                        }
                    } else if (use_constant_rho) {
                        rho_val = rho_s;

                        // cameras: a constant reflectivity is flat across the band, so each camera sees it weighted by its
                        // response to a white surface, as it would a reflectivity spectrum flat at the same value
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.reflectivity_cam[cam_idx(slot, u, b, cam)] = rho_s * material_data.white_reference_cam.at(((size_t) slot * Nbands + b) * Ncameras + cam);
                        }
                    } else {
                        rho_val = rho_unique.at(spectrum_label).at(b).at(unique_index);
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.reflectivity_cam[cam_idx(slot, u, b, cam)] = rho_cam_unique.at(spectrum_label).at(b).at(unique_index).at(cam);
                        }
                    }

                    // error checking
                    if (rho_val < 0) {
                        rho_val = 0.f;
                        warnings.addWarning("reflectivity_negative_clamped", "Reflectivity cannot be less than 0. Clamping to 0 for band " + band + ".");
                    } else if (rho_val > 1.f) {
                        rho_val = 1.f;
                        warnings.addWarning("reflectivity_exceeded_clamped", "Reflectivity cannot be greater than 1. Clamping to 1 for band " + band + ".");
                    }
                    if (rho_val != 0) {
                        scattering_iterations_needed.at(band) = true;
                    }
                    if (slot < Nsources) {
                        for (auto &odata: output_prim_data) {
                            if (odata == "reflectivity") {
                                context->setPrimitiveData(UUID, ("reflectivity_" + std::to_string(slot) + "_" + band).c_str(), rho_val);
                            }
                        }
                    }
                }
                b++;
            }

            // Transmissivity

            // check for primitive data of form "transmissivity_spectrum" that can be used to calculate transmissivity
            spectrum_label.resize(0);
            if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum")) {
                if (context->getPrimitiveDataType("transmissivity_spectrum") == HELIOS_TYPE_STRING) {
                    context->getPrimitiveData(UUID, "transmissivity_spectrum", spectrum_label);
                }
            }

            b = 0;
            for (const auto &band: band_labels) {

                // check for primitive data of form "transmissivity_bandname"
                prop = "transmissivity_" + band;

                float tau_s = tau_default;
                const bool has_band_tau = context->doesPrimitiveDataExist(UUID, prop.c_str());
                if (has_band_tau) {
                    if (has_spectral_texture && material_data.palette_band_active.at(b) != 0) {
                        rejectConstantProperty(prop);
                    }
                    context->getPrimitiveData(UUID, prop.c_str(), tau_s);
                }
                const bool tau_from_palette = has_spectral_texture && !has_band_tau;
                if (tau_from_palette) {
                    recordPaletteBandErrors(u, b, band, false);
                }

                const bool use_constant_tau = has_band_tau || spectrum_label.empty() || !context->doesGlobalDataExist(spectrum_label.c_str()) || tau_unique.find(spectrum_label) == tau_unique.end();
                if (!use_constant_tau && !has_spectral_texture) {
                    // Recorded rather than raised here: runBand() raises it only if this band is run
                    const std::string &error = !tau_band_error.at(spectrum_label).at(b).empty() ? tau_band_error.at(spectrum_label).at(b) : band_weighting_error.at(b);
                    if (!error.empty()) {
                        spectral_weighting_errors.emplace(band, error);
                    }
                }

                for (uint slot = 0; slot < Nslots; slot++) {
                    float &tau_val = material_data.transmissivity[mat_idx(slot, u, b)];
                    const uint unique_index = slot == diffuse_slot ? unique_diffuse_index : slot;

                    if (tau_from_palette) {
                        tau_val = paletteMean(u, [&](size_t entry) { return material_data.palette_transmissivity[palette_idx(slot, entry, b)]; });
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.transmissivity_cam[cam_idx(slot, u, b, cam)] = paletteMean(u, [&](size_t entry) { return material_data.palette_transmissivity_cam[palette_cam_idx(slot, entry, b, cam)]; });
                        }
                    } else if (use_constant_tau) {
                        tau_val = tau_s;

                        // cameras: a constant transmissivity is flat across the band (see reflectivity above)
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.transmissivity_cam[cam_idx(slot, u, b, cam)] = tau_s * material_data.white_reference_cam.at(((size_t) slot * Nbands + b) * Ncameras + cam);
                        }
                    } else {
                        tau_val = tau_unique.at(spectrum_label).at(b).at(unique_index);
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.transmissivity_cam[cam_idx(slot, u, b, cam)] = tau_cam_unique.at(spectrum_label).at(b).at(unique_index).at(cam);
                        }
                    }

                    // error checking
                    if (tau_val < 0) {
                        tau_val = 0.f;
                        warnings.addWarning("transmissivity_negative_clamped", "Transmissivity cannot be less than 0. Clamping to 0 for band " + band + ".");
                    } else if (tau_val > 1.f) {
                        tau_val = 1.f;
                        warnings.addWarning("transmissivity_exceeded_clamped", "Transmissivity cannot be greater than 1. Clamping to 1 for band " + band + ".");
                    }
                    if (tau_val != 0) {
                        scattering_iterations_needed.at(band) = true;
                    }
                    if (slot < Nsources) {
                        for (auto &odata: output_prim_data) {
                            if (odata == "transmissivity") {
                                context->setPrimitiveData(UUID, ("transmissivity_" + std::to_string(slot) + "_" + band).c_str(), tau_val);
                            }
                        }
                    }
                }
                b++;
            }

            // Translucent cover (glass/plastic) material.
            // Presence of primitive data "glass_n_<band>" activates the Fresnel+Bouguer angular
            // transmittance model for this primitive+band. Optional "glass_KL_<band>" (default 0 =
            // lossless) sets the Bouguer absorption product K*L. When active, the angular rho/tau are
            // computed on-device, overriding any constant reflectivity_<band>/transmissivity_<band>.
            b = 0;
            for (const auto &band: band_labels) {

                prop = "glass_n_" + band;
                if (!context->doesPrimitiveDataExist(UUID, prop.c_str()) || context->getPrimitiveDataType(prop.c_str()) != HELIOS_TYPE_FLOAT) {
                    b++;
                    continue;
                }
                if (has_spectral_texture) {
                    rejectConstantProperty(prop);
                }

                float n_s = 0.f;
                context->getPrimitiveData(UUID, prop.c_str(), n_s);

                if (n_s < 1.f) {
                    warnings.addWarning("glass_n_invalid", "Refractive index glass_n_" + band + " must be >= 1. Ignoring glass material for this band.");
                    b++;
                    continue;
                }

                float KL_s = 0.f;
                std::string KL_prop = "glass_KL_" + band;
                if (context->doesPrimitiveDataExist(UUID, KL_prop.c_str()) && context->getPrimitiveDataType(KL_prop.c_str()) == HELIOS_TYPE_FLOAT) {
                    context->getPrimitiveData(UUID, KL_prop.c_str(), KL_s);
                    if (KL_s < 0.f) {
                        KL_s = 0.f;
                        warnings.addWarning("glass_KL_negative_clamped", "Absorption glass_KL_" + band + " cannot be less than 0. Clamping to 0.");
                    }
                }

                // Warn (once) if a constant reflectivity/transmissivity was also set for this band: glass wins.
                if (context->doesPrimitiveDataExist(UUID, ("reflectivity_" + band).c_str()) || context->doesPrimitiveDataExist(UUID, ("transmissivity_" + band).c_str())) {
                    warnings.addWarning("glass_overrides_constant", "Primitive has both glass_n_" + band + " and a constant reflectivity/transmissivity for band " + band +
                                                                            ". The glass (Fresnel+Bouguer) model takes precedence; the constant value is ignored.");
                }

                for (uint slot = 0; slot < Nslots; slot++) {
                    material_data.is_glass[mat_idx(slot, u, b)] = 1;
                    material_data.glass_n[mat_idx(slot, u, b)] = n_s;
                    material_data.glass_KL[mat_idx(slot, u, b)] = KL_s;
                }

                // Glass requires the scattering machinery to be active so the diffuse-sky / reflected
                // bookkeeping runs (the unscattered direct/sky attenuation itself is independent of depth).
                scattering_iterations_needed.at(band) = true;
                b++;
            }

            // Emissivity (only for error checking)

            b = 0;
            for (const auto &band: band_labels) {

                prop = "emissivity_" + band;

                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                    context->getPrimitiveData(UUID, prop.c_str(), eps);
                } else {
                    eps = eps_default;
                }

                if (eps < 0) {
                    eps = 0.f;
                    warnings.addWarning("emissivity_negative_clamped", "Emissivity cannot be less than 0. Clamping to 0 for band " + band + ".");
                } else if (eps > 1.f) {
                    eps = 1.f;
                    warnings.addWarning("emissivity_exceeded_clamped", "Emissivity cannot be greater than 1. Clamping to 1 for band " + band + ".");
                }
                if (eps != 1) {
                    scattering_iterations_needed.at(band) = true;
                }

                assert(doesBandExist(band));

                const bool is_sif_band = sif_emission_bands.count(band) > 0;

                if (has_spectral_texture) {
                    const std::vector<std::pair<uint16_t, float>> &fractions = spectral_class_fractions.at(u);
                    const size_t first_entry = primitive_palette_offset.at(u) + fractions.front().first - 1;
                    const bool palette_applies = material_data.palette_band_active.at(b) != 0;
                    const bool rho_from_palette = !context->doesPrimitiveDataExist(UUID, ("reflectivity_" + band).c_str());
                    const bool tau_from_palette = !context->doesPrimitiveDataExist(UUID, ("transmissivity_" + band).c_str());
                    for (const auto &fraction: fractions) {
                        const size_t entry = primitive_palette_offset.at(u) + fraction.first - 1;
                        for (uint slot = 0; slot < Nslots; slot++) {
                            const float class_rho = material_data.palette_reflectivity[palette_idx(slot, entry, b)];
                            const float class_tau = material_data.palette_transmissivity[palette_idx(slot, entry, b)];
                            // Rays use each class's own values where the palette applies, so each class must conserve energy, not only their mean
                            if (palette_applies && class_rho + class_tau > 1.f) {
                                helios_runtime_error("ERROR (RadiationModel): reflectivity and transmissivity must sum to less than or equal to 1 to ensure energy conservation. Primitive #" + std::to_string(UUID) +
                                                     " has a spectral texture whose class " + std::to_string(fraction.first) + " has rho=" + std::to_string(class_rho) + " and tau=" + std::to_string(class_tau) + " in band " + band +
                                                     ". Correct the spectra of that class in its palette.");
                            }
                            // In a band with emission, a primitive emits with one emissivity, which must agree with its reflectivity and transmissivity everywhere
                            // on it, so the classes must not differ in whatever the primitive takes from its palette there (rays then use the primitive's own values)
                            if (!palette_applies && !is_sif_band &&
                                ((rho_from_palette && class_rho != material_data.palette_reflectivity[palette_idx(slot, first_entry, b)]) ||
                                 (tau_from_palette && class_tau != material_data.palette_transmissivity[palette_idx(slot, first_entry, b)]))) {
                                helios_runtime_error(
                                        "ERROR (RadiationModel::runBand): Primitive UUID " + std::to_string(UUID) + " has a spectral texture whose classes " + std::to_string(fractions.front().first) + " and " + std::to_string(fraction.first) +
                                        " have different reflectivity or transmissivity in band " + band +
                                        ", which has emission. A spectral texture cannot yet vary in a band with emission, since each primitive has one emissivity. Disable emission for the band (RadiationModel::disableEmission()), give the classes the same spectra in it, or give the primitive constant \"reflectivity_" +
                                        band + "\" and \"transmissivity_" + band + "\" data for the band.");
                            }
                        }
                    }
                }

                for (uint slot = 0; slot < Nslots; slot++) {
                    float &rho_val = material_data.reflectivity[mat_idx(slot, u, b)];
                    float &tau_val = material_data.transmissivity[mat_idx(slot, u, b)];

                    // Glass primitives compute angle-dependent rho/tau on-device via the Fresnel+Bouguer
                    // model, so the constant ε+ρ+τ=1 conservation constraint does not apply here.
                    if (material_data.is_glass[mat_idx(slot, u, b)] != 0) {
                        continue;
                    }

                    if (is_sif_band) {
                        // SIF bands source their emission from the Fluspect-B per-leaf
                        // kernel (see computeSIFEmission), not from epsilon*sigma*T^4.
                        // Epsilon on a SIF band is therefore irrelevant — the Stefan-
                        // Boltzmann ε+ρ+τ=1 conservation constraint does not apply. We
                        // still enforce rho+tau ≤ 1 (physically required regardless of
                        // the emission mechanism).
                        if (tau_val + rho_val > 1.f) {
                            helios_runtime_error("ERROR (RadiationModel): reflectivity and transmissivity must sum to less than or equal to 1 to ensure energy conservation. Band " + band + ", Primitive #" + std::to_string(UUID) +
                                                 ": tau=" + std::to_string(tau_val) + ", rho=" + std::to_string(rho_val) + ".");
                        }
                    } else if (radiation_bands.at(band).emissionFlag) { // emission enabled
                        if (eps != 1.f && rho_val == 0 && tau_val == 0) {
                            rho_val = 1.f - eps;
                            // cameras see this reflectivity too, weighted by their response to a white surface as a constant reflectivity is
                            for (uint cam = 0; cam < Ncameras; cam++) {
                                material_data.reflectivity_cam[cam_idx(slot, u, b, cam)] = rho_val * material_data.white_reference_cam.at(((size_t) slot * Nbands + b) * Ncameras + cam);
                            }
                        } else if (eps + tau_val + rho_val != 1.f && eps > 0.f) {
                            helios_runtime_error("ERROR (RadiationModel): emissivity, transmissivity, and reflectivity must sum to 1 to ensure energy conservation. Band " + band + ", Primitive #" + std::to_string(UUID) + ": eps=" +
                                                 std::to_string(eps) + ", tau=" + std::to_string(tau_val) + ", rho=" + std::to_string(rho_val) + ". It is also possible that you forgot to disable emission for this band.");
                        } else if (radiation_bands.at(band).scatteringDepth == 0 && eps != 1.f) {
                            // Without scattering the surface is treated as black in this band. eps itself is left alone: every slot is checked against the primitive's emissivity.
                            rho_val = 0.f;
                            tau_val = 0.f;
                        }
                    } else if (tau_val + rho_val > 1.f) {
                        helios_runtime_error("ERROR (RadiationModel): transmissivity and reflectivity cannot sum to greater than 1 ensure energy conservation. Band " + band + ", Primitive #" + std::to_string(UUID) + ": eps=" + std::to_string(eps) +
                                             ", tau=" + std::to_string(tau_val) + ", rho=" + std::to_string(rho_val) + ". It is also possible that you forgot to disable emission for this band.");
                    }
                }
                b++;
            }
        }
    }

    // Without scattering every surface is black: what a primitive reflects or transmits is absorbed in the flux solution, so cameras
    // must not see it either. Cameras still see what the surface emits (with an emissivity of 1, as the emission launch uses).
    if (Ncameras > 0) {
        for (uint b = 0; b < Nbands; b++) {
            if (radiation_bands.at(band_labels.at(b)).scatteringDepth != 0) {
                continue;
            }
            for (uint slot = 0; slot < Nslots; slot++) {
                for (uint cam = 0; cam < Ncameras; cam++) {
                    for (size_t u = 0; u < Nprimitives; u++) {
                        material_data.reflectivity_cam[cam_idx(slot, u, b, cam)] = 0.f;
                        material_data.transmissivity_cam[cam_idx(slot, u, b, cam)] = 0.f;
                    }
                    for (size_t entry = 0; entry < Nentries; entry++) {
                        material_data.palette_reflectivity_cam[palette_cam_idx(slot, entry, b, cam)] = 0.f;
                        material_data.palette_transmissivity_cam[palette_cam_idx(slot, entry, b, cam)] = 0.f;
                    }
                }
            }
        }
    }

    updateSurfaceLayers(warnings);

    // A surface layer reflects a source's direct radiation into its microfacet lobe, which needs the direct flux reaching each
    // primitive with a layer (radiation_specular). Camera flux smoothing needs the same for every smoothed primitive, to evaluate
    // its direct radiation with the shading normal (see subtractCameraDirectFlux()).
    specular_reflection_mode = material_data.surface_layer_enabled ? 1 : 0;
    material_data.specular_reflection_enabled = material_data.surface_layer_enabled || (cameraflux_smoothing_enabled && smoothing_vertex_count > 0);

    backend->updateMaterials(material_data);

    radiativepropertiesneedupdate = false;

    if (message_flag) {
        std::cout << "done\n";
    }

    // Report aggregated warnings
    warnings.report(std::cerr);
}

void RadiationModel::updateSurfaceLayers(helios::WarningAggregator &warnings) {

    const size_t Nprimitives = context_UUIDs.size();
    const uint Nbands = radiation_bands.size();
    const uint Nslots = radiation_sources.size() + 1;
    const uint Ncameras = cameras.size();
    const uint32_t table_size = material_data.surface_layer_table_size;
    std::vector<std::string> band_labels;
    for (const auto &band: radiation_bands) {
        band_labels.push_back(band.first);
    }

    material_data.surface_model.assign(Nprimitives, 0);
    material_data.surface_model_band_table.clear();
    material_data.surface_layer_tables.clear();
    material_data.surface_layer_enabled = false;

    MaterialPropertyIndexer mat_idx(Nslots, Nprimitives, Nbands);
    CameraMaterialIndexer cam_idx(Nslots, Nprimitives, Nbands, Ncameras);

    // A surface model is a roughness and a refractive index in each band; each distinct (refractive index, roughness) pair has one albedo table
    std::map<std::vector<float>, uint32_t> model_index;
    std::map<std::pair<float, float>, uint32_t> table_index;

    // Reads a surface-layer parameter of a primitive. Material data takes precedence over primitive data, as elsewhere in Helios.
    auto readParameter = [&](uint UUID, const std::string &label, float &value) -> bool {
        const Material &material = context->getMaterial(context->getPrimitiveMaterialID(UUID));
        const bool on_primitive = context->doesPrimitiveDataExist(UUID, label.c_str());
        if (material.doesMaterialDataExist(label.c_str())) {
            // The default material and the automatically generated ones (labels beginning "__") are shared by every primitive with the same color and texture
            if (material.label.rfind("__", 0) == 0) {
                helios_runtime_error("ERROR (RadiationModel): Surface-layer data '" + label + "' is set on the automatically generated material '" + material.label + "' of primitive #" + std::to_string(UUID) +
                                     ". Automatic and default materials are shared by every primitive with the same color and texture, so the data would apply to unrelated primitives. Assign a named material "
                                     "(Context::addMaterial() and Context::assignMaterialToPrimitive()) and set the data on it, or set it as primitive data.");
            }
            if (material.getMaterialDataType(label.c_str()) != HELIOS_TYPE_FLOAT) {
                helios_runtime_error("ERROR (RadiationModel): Material data '" + label + "' of material '" + material.label + "' must be of type float.");
            }
            material.getMaterialData(label.c_str(), value);
            if (on_primitive) {
                warnings.addWarning("surface_layer_data_shadowed", "Primitive #" + std::to_string(UUID) + " has '" + label + "' both as primitive data and in its material '" + material.label + "'; the material value is used.");
            }
            return true;
        }
        if (on_primitive) {
            if (context->getPrimitiveDataType(label.c_str()) != HELIOS_TYPE_FLOAT) {
                helios_runtime_error("ERROR (RadiationModel): Primitive data '" + label + "' must be of type float.");
            }
            context->getPrimitiveData(UUID, label.c_str(), value);
            return true;
        }
        return false;
    };

    for (size_t u = 0; u < Nprimitives; u++) {
        const uint UUID = context_UUIDs.at(u);

        // The camera-only Blinn-Phong highlight these controlled added energy on top of the diffuse reflection
        for (const char *legacy_label: {"specular_exponent", "specular_scale"}) {
            if (context->doesPrimitiveDataExist(UUID, legacy_label) || context->getMaterial(context->getPrimitiveMaterialID(UUID)).doesMaterialDataExist(legacy_label)) {
                helios_runtime_error("ERROR (RadiationModel): Primitive #" + std::to_string(UUID) + " has '" + std::string(legacy_label) +
                                     "' data, which is no longer supported. Specular reflection is now modelled by a surface layer that conserves energy and appears in the fluxes as well as in "
                                     "camera images: set 'specular_roughness' (GGX roughness from 0.05 to 1; a Blinn-Phong exponent k corresponds roughly to sqrt(2 / (k + 2))) and optionally 'specular_n' "
                                     "(refractive index, default 1.5) instead, and remove the old data.");
            }
        }

        float roughness;
        if (!readParameter(UUID, "specular_roughness", roughness)) {
            continue;
        }
        // The layer's body values would have to be derived for each class of the palette
        if (material_data.palette_offset.at(u) != UINT32_MAX) {
            helios_runtime_error(
                    "ERROR (RadiationModel): Primitive #" + std::to_string(UUID) +
                    " has a spectral texture and a specular surface layer (specular_roughness), which cannot yet be combined. Remove 'specular_roughness' from the primitive and its material, or remove the spectral texture with Context::clearPrimitiveSpectralTexture().");
        }
        // Below this roughness the reflection lobe is so narrow that the rays gathering it are too noisy to be useful
        if (!(roughness >= 0.05f && roughness <= 1.f)) {
            helios_runtime_error("ERROR (RadiationModel): The specular_roughness of primitive #" + std::to_string(UUID) + " is " + std::to_string(roughness) + ", but it must be between 0.05 and 1.");
        }
        float refractive_index_all = 1.5f;
        readParameter(UUID, "specular_n", refractive_index_all);

        std::vector<float> model_key{roughness};
        for (uint b = 0; b < Nbands; b++) {
            float refractive_index = refractive_index_all;
            readParameter(UUID, "specular_n_" + band_labels.at(b), refractive_index);
            if (!(refractive_index >= 1.f && refractive_index <= 3.f)) {
                helios_runtime_error("ERROR (RadiationModel): The refractive index of the surface layer of primitive #" + std::to_string(UUID) + " in band '" + band_labels.at(b) + "' is " + std::to_string(refractive_index) +
                                     ", but it must be between 1 (no layer) and 3 (specular_n or specular_n_" + band_labels.at(b) + ").");
            }
            // Without scattering every surface is black, so there is no layer to reflect anything
            if (radiation_bands.at(band_labels.at(b)).scatteringDepth == 0) {
                refractive_index = 1.f;
            }
            if (refractive_index > 1.f) {
                for (uint slot = 0; slot < Nslots; slot++) {
                    if (material_data.is_glass.at(mat_idx(slot, u, b)) != 0) {
                        helios_runtime_error("ERROR (RadiationModel): Primitive #" + std::to_string(UUID) + " is a translucent cover in band '" + band_labels.at(b) + "' (glass_n_" + band_labels.at(b) +
                                             ") and also has a specular surface layer (specular_roughness). A primitive can have only one of the two.");
                    }
                }
            }
            model_key.push_back(refractive_index);
        }

        auto model_it = model_index.find(model_key);
        if (model_it == model_index.end()) {
            model_it = model_index.emplace(model_key, uint32_t(model_index.size())).first;
            for (uint b = 0; b < Nbands; b++) {
                const float refractive_index = model_key.at(b + 1);
                if (refractive_index == 1.f) {
                    material_data.surface_model_band_table.push_back(UINT32_MAX);
                    continue;
                }
                auto table_it = table_index.find({refractive_index, roughness});
                if (table_it == table_index.end()) {
                    table_it = table_index.emplace(std::make_pair(refractive_index, roughness), uint32_t(table_index.size())).first;
                    const std::vector<float> table = computeSurfaceLayerAlbedoTable(refractive_index, roughness, table_size);
                    material_data.surface_layer_tables.insert(material_data.surface_layer_tables.end(), table.begin(), table.end());
                    // The microfacet lobe of the direct radiation is evaluated from the layer's parameters
                    material_data.surface_layer_tables.push_back(refractive_index);
                    material_data.surface_layer_tables.push_back(roughness);
                }
                material_data.surface_model_band_table.push_back(table_it->second);
            }
        }
        material_data.surface_model.at(u) = model_it->second + 1;

        // The reflectivity and transmissivity are the totals at normal incidence. The body's are what remains after the layer:
        // rho = a(1) + rho_b (1 - a(1)) and tau = tau_b (1 - a(1)).
        for (uint b = 0; b < Nbands; b++) {
            const float *table = surfaceLayerTable(u, b);
            if (table == nullptr) {
                continue;
            }
            material_data.surface_layer_enabled = true;
            const float normal_albedo = table[table_size - 1];
            for (uint slot = 0; slot < Nslots; slot++) {
                float &rho = material_data.reflectivity.at(mat_idx(slot, u, b));
                float &tau = material_data.transmissivity.at(mat_idx(slot, u, b));
                if (rho < normal_albedo - 1e-6f) {
                    helios_runtime_error("ERROR (RadiationModel): The reflectivity of primitive #" + std::to_string(UUID) + " in band '" + band_labels.at(b) + "' is " + std::to_string(rho) + ", less than the " + std::to_string(normal_albedo) +
                                         " that its specular surface layer (refractive index " + std::to_string(model_key.at(b + 1)) +
                                         ") alone reflects at normal incidence. The reflectivity is the total reflectance at normal incidence, so it cannot be smaller than that of the surface layer. In an emission band the "
                                         "emissivity likewise cannot exceed 1 minus that value. Lower the refractive index for this band with specular_n_" +
                                         band_labels.at(b) + " (1 removes the layer), or raise the reflectivity.");
                }
                rho = std::max(0.f, rho - normal_albedo) / (1.f - normal_albedo);
                tau = tau / (1.f - normal_albedo);
                for (uint cam = 0; cam < Ncameras; cam++) {
                    // The layer reflects all wavelengths alike, so a camera sees it as it sees a white surface
                    const float white = material_data.white_reference_cam.at(((size_t) slot * Nbands + b) * Ncameras + cam);
                    float &rho_cam = material_data.reflectivity_cam.at(cam_idx(slot, u, b, cam));
                    float &tau_cam = material_data.transmissivity_cam.at(cam_idx(slot, u, b, cam));
                    if (rho_cam < normal_albedo * white - 1e-6f) {
                        helios_runtime_error("ERROR (RadiationModel): The reflectivity of primitive #" + std::to_string(UUID) + " in band '" + band_labels.at(b) + "', weighted by the spectral response of camera #" + std::to_string(cam) +
                                             ", is less than its specular surface layer alone reflects at normal incidence (" + std::to_string(normal_albedo) +
                                             "). Its reflectivity spectrum falls below that value within the camera's response; lower the refractive index for this band with specular_n_" + band_labels.at(b) + ".");
                    }
                    rho_cam = std::max(0.f, rho_cam - normal_albedo * white) / (1.f - normal_albedo);
                    tau_cam = tau_cam / (1.f - normal_albedo);
                }
            }
        }
    }
}

const float *RadiationModel::surfaceLayerTable(size_t u, size_t b_global) const {
    const uint32_t model = material_data.surface_model.empty() ? 0 : material_data.surface_model.at(u);
    if (model == 0) {
        return nullptr;
    }
    const uint32_t table = material_data.surface_model_band_table.at((size_t) (model - 1) * radiation_bands.size() + b_global);
    if (table == UINT32_MAX) {
        return nullptr;
    }
    return material_data.surface_layer_tables.data() + (size_t) table * (material_data.surface_layer_table_size + 3);
}

void RadiationModel::applyCameraAtmosphere(RadiationCamera &camera, const std::vector<std::string> &launched_band_labels) {
    // The direction from which the atmosphere spectra were computed must match the camera's viewing direction to within this angle
    const float max_direction_mismatch_deg = 1.f;

    const std::string &atmosphere = camera.atmosphere_label;
    const std::string calculate_hint = " Call SolarPosition::calculateSensorAtmosphereSpectra(\"" + atmosphere + "\", ...) before runBand().";
    auto load_atmosphere_spectrum = [&](const std::string &suffix) {
        const std::string label = atmosphere + suffix;
        if (!context->doesGlobalDataExist(label.c_str()) || context->getGlobalDataType(label.c_str()) != helios::HELIOS_TYPE_VEC2) {
            helios_runtime_error("ERROR (RadiationModel::runBand): The sensor atmosphere spectrum '" + label + "' for camera '" + camera.label + "' does not exist." + calculate_hint);
        }
        return loadSpectralData(label);
    };
    const std::string direction_label = atmosphere + "_direction_to_sensor";
    if (!context->doesGlobalDataExist(direction_label.c_str()) || context->getGlobalDataType(direction_label.c_str()) != helios::HELIOS_TYPE_VEC3) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The sensor direction '" + direction_label + "' for camera '" + camera.label + "' does not exist. Call SolarPosition::calculateSensorAtmosphereSpectra(\"" + atmosphere +
                             "\", ...) for reflective bands or SolarPosition::calculateSensorThermalAtmosphere(\"" + atmosphere + "\", ...) for emission bands before runBand().");
    }
    helios::vec3 direction_to_sensor;
    context->getGlobalData(direction_label.c_str(), direction_to_sensor);
    helios::vec3 camera_direction = camera.position - camera.lookat;
    camera_direction.normalize();
    const float direction_mismatch_deg = helios::rad2deg(helios::acos_safe(camera_direction * direction_to_sensor));
    if (direction_mismatch_deg > max_direction_mismatch_deg) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The sensor atmosphere '" + atmosphere + "' was computed for a direction " + std::to_string(direction_mismatch_deg) + " degrees away from the viewing direction of camera '" +
                             camera.label + "'. Recompute it with SolarPosition::calculateSensorAtmosphereSpectra() or SolarPosition::calculateSensorThermalAtmosphere() using the direction from the camera's look-at point toward its position.");
    }

    // Radiance from the sky has no surface below it for the atmosphere model to act on
    const size_t sky_pixels = std::count_if(camera.pixel_depth.begin(), camera.pixel_depth.end(), [](float depth) { return depth < 0.f; });
    if (sky_pixels > 0) {
        helios_runtime_error("ERROR (RadiationModel::runBand): " + std::to_string(sky_pixels) + " pixels of camera '" + camera.label +
                             "' see the sky, but a sensor atmosphere is enabled for it. The scene must fill the camera's field of view.");
    }

    for (const std::string &band: launched_band_labels) {
        // runBand() fills pixel data for every launched band, including bands the camera does not use
        if (std::find(camera.band_labels.begin(), camera.band_labels.end(), band) == camera.band_labels.end() || camera.pixel_data.find(band) == camera.pixel_data.end()) {
            continue;
        }
        if (radiation_bands.at(band).emissionFlag) {
            applyCameraThermalAtmosphere(camera, band);
            continue;
        }

        const std::vector<helios::vec2> path_radiance = load_atmosphere_spectrum("_path_radiance");
        const std::vector<helios::vec2> adjacency_radiance = load_atmosphere_spectrum("_adjacency_radiance");
        const std::vector<helios::vec2> upward_direct_transmittance = load_atmosphere_spectrum("_upward_direct_transmittance");
        const std::vector<helios::vec2> global_irradiance = load_atmosphere_spectrum("_global_irradiance");
        const float atmosphere_min_wavelength = path_radiance.front().x;
        const float atmosphere_max_wavelength = path_radiance.back().x;

        // Camera spectral response, zero outside its tabulated range; a band without a response spectrum has a uniform response of one
        const std::vector<helios::vec2> response = loadCameraBandResponse(camera, band);
        auto response_at = [&](float wavelength) {
            if (response.empty()) {
                return 1.f;
            }
            if (wavelength < response.front().x || wavelength > response.back().x) {
                return 0.f;
            }
            return interp1(response, wavelength);
        };

        helios::vec2 bounds = radiation_bands.at(band).wavebandBounds;
        if (!radiation_bands.at(band).hasWavebandBounds()) {
            if (response.empty()) {
                helios_runtime_error("ERROR (RadiationModel::runBand): Band '" + band + "' of camera '" + camera.label + "' has neither wavelength bounds nor a spectral response, so the sensor atmosphere cannot be integrated over it.");
            }
            bounds = helios::make_vec2(response.front().x, response.back().x);
        }
        if (bounds.x < atmosphere_min_wavelength || bounds.y > atmosphere_max_wavelength) {
            helios_runtime_error("ERROR (RadiationModel::runBand): Band '" + band + "' of camera '" + camera.label + "' spans " + std::to_string(bounds.x) + "-" + std::to_string(bounds.y) + " nm, outside the " +
                                 std::to_string(atmosphere_min_wavelength) + "-" + std::to_string(atmosphere_max_wavelength) + " nm covered by the sensor atmosphere spectra.");
        }

        // Band radiances are integrals of spectral radiance times the (unnormalized) spectral response, as camera pixel values are
        const float band_path_radiance = integrateSpectrumBetweenBounds(path_radiance, bounds.x, bounds.y, [&](float wavelength, float value) { return value * response_at(wavelength); });
        const float band_adjacency_radiance = integrateSpectrumBetweenBounds(adjacency_radiance, bounds.x, bounds.y, [&](float wavelength, float value) { return value * response_at(wavelength); });
        const float irradiance_weight = integrateSpectrumBetweenBounds(global_irradiance, bounds.x, bounds.y, [&](float wavelength, float value) { return value * response_at(wavelength); });
        if (irradiance_weight <= 0.f) {
            helios_runtime_error("ERROR (RadiationModel::runBand): The spectral response of band '" + band + "' of camera '" + camera.label + "' is zero over the band, so the sensor atmosphere cannot be applied.");
        }
        const float band_direct_transmittance =
                integrateSpectrumBetweenBounds(global_irradiance, bounds.x, bounds.y, [&](float wavelength, float value) { return value * response_at(wavelength) * interp1(upward_direct_transmittance, wavelength); }) / irradiance_weight;

        std::vector<float> &pixels = camera.pixel_data.at(band);
        for (float &pixel: pixels) {
            pixel = band_path_radiance + band_direct_transmittance * pixel + band_adjacency_radiance;
        }
        const std::string data_label = "camera_" + camera.label + "_" + band;
        context->setGlobalData(data_label.c_str(), pixels);
    }
}

std::vector<helios::vec2> RadiationModel::loadCameraBandResponse(const RadiationCamera &camera, const std::string &band) {
    auto response_it = camera.band_spectral_response.find(band);
    if (response_it != camera.band_spectral_response.end() && !response_it->second.empty() && response_it->second != "uniform" && context->doesGlobalDataExist(response_it->second.c_str()) &&
        context->getGlobalDataType(response_it->second.c_str()) == helios::HELIOS_TYPE_VEC2) {
        return loadSpectralData(response_it->second);
    }
    return {};
}

void RadiationModel::bandIntegrationGrid(const helios::vec2 &bounds, const std::vector<const std::vector<helios::vec2> *> &spectra, const std::vector<helios::vec2> &response, std::vector<float> &wavelengths, std::vector<float> &weights) {
    // Largest wavelength step of the grid, fine enough for the Planck spectrum to be integrated by the trapezoidal rule to better than 1e-4
    const float max_integration_step_nm = 10.f;

    // The band bounds and every tabulated wavelength of the spectra and the response within them
    std::vector<float> nodes = {bounds.x, bounds.y};
    std::vector<const std::vector<helios::vec2> *> node_spectra = spectra;
    node_spectra.push_back(&response);
    for (const std::vector<helios::vec2> *spectrum: node_spectra) {
        for (const helios::vec2 &point: *spectrum) {
            if (point.x > bounds.x && point.x < bounds.y) {
                nodes.push_back(point.x);
            }
        }
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    wavelengths.clear();
    for (size_t i = 0; i + 1 < nodes.size(); i++) {
        const int Nsteps = std::max(1, int(std::ceil((nodes.at(i + 1) - nodes.at(i)) / max_integration_step_nm)));
        for (int k = 0; k < Nsteps; k++) {
            wavelengths.push_back(nodes.at(i) + (nodes.at(i + 1) - nodes.at(i)) * float(k) / float(Nsteps));
        }
    }
    wavelengths.push_back(nodes.back());

    // Trapezoidal weight of each grid point times the spectral response there, which is zero outside its tabulated range
    weights.assign(wavelengths.size(), 0.f);
    for (size_t i = 0; i < wavelengths.size(); i++) {
        const float step_below = i == 0 ? 0.f : wavelengths.at(i) - wavelengths.at(i - 1);
        const float step_above = i + 1 == wavelengths.size() ? 0.f : wavelengths.at(i + 1) - wavelengths.at(i);
        float response_value = 1.f;
        if (!response.empty()) {
            response_value = wavelengths.at(i) < response.front().x || wavelengths.at(i) > response.back().x ? 0.f : interp1(response, wavelengths.at(i));
        }
        weights.at(i) = 0.5f * (step_below + step_above) * response_value;
    }
}

void RadiationModel::applyCameraThermalAtmosphere(RadiationCamera &camera, const std::string &band) {
    // Brightness temperatures over which band radiance is tabulated for converting pixels to temperature
    const float min_brightness_temperature = 150.f;
    const float max_brightness_temperature = 400.f;
    const float brightness_temperature_step = 0.25f;
    const float stefan_boltzmann = 5.670374419e-8f;

    const std::string &atmosphere = camera.atmosphere_label;
    auto load_thermal_spectrum = [&](const std::string &suffix) {
        const std::string label = atmosphere + suffix;
        if (!context->doesGlobalDataExist(label.c_str()) || context->getGlobalDataType(label.c_str()) != helios::HELIOS_TYPE_VEC2) {
            helios_runtime_error("ERROR (RadiationModel::runBand): The thermal sensor atmosphere '" + label + "' for emission band '" + band + "' of camera '" + camera.label + "' does not exist. Call SolarPosition::calculateSensorThermalAtmosphere(\"" +
                                 atmosphere + "\", ...) before runBand().");
        }
        return loadSpectralData(label);
    };
    const std::vector<helios::vec2> transmittance = load_thermal_spectrum("_thermal_transmittance");
    const std::vector<helios::vec2> upwelling_radiance = load_thermal_spectrum("_thermal_upwelling_radiance");

    const helios::vec2 bounds = radiation_bands.at(band).wavebandBounds;
    if (!radiation_bands.at(band).hasWavebandBounds()) {
        helios_runtime_error("ERROR (RadiationModel::runBand): Emission band '" + band + "' of camera '" + camera.label + "' has no wavelength bounds, so its pixels are broadband radiance to which the thermal sensor atmosphere cannot be applied.");
    }
    if (bounds.x < transmittance.front().x || bounds.y > transmittance.back().x) {
        helios_runtime_error("ERROR (RadiationModel::runBand): Emission band '" + band + "' of camera '" + camera.label + "' spans " + std::to_string(bounds.x) + "-" + std::to_string(bounds.y) + " nm, outside the " +
                             std::to_string(transmittance.front().x) + "-" + std::to_string(transmittance.back().x) + " nm covered by the thermal sensor atmosphere.");
    }

    // Integration grid over the band, weighted by the camera's spectral response (uniform if it has none)
    const std::vector<helios::vec2> response = loadCameraBandResponse(camera, band);
    std::vector<float> grid_wavelengths;
    std::vector<float> grid_weights;
    bandIntegrationGrid(bounds, {&transmittance, &upwelling_radiance}, response, grid_wavelengths, grid_weights);
    std::vector<float> grid_transmittance(grid_wavelengths.size());
    float band_upwelling_radiance = 0.f;
    float response_integral = 0.f;
    for (size_t i = 0; i < grid_wavelengths.size(); i++) {
        grid_transmittance.at(i) = interp1(transmittance, grid_wavelengths.at(i));
        band_upwelling_radiance += grid_weights.at(i) * interp1(upwelling_radiance, grid_wavelengths.at(i));
        response_integral += grid_weights.at(i);
    }
    if (!(response_integral > 0.f)) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The spectral response of emission band '" + band + "' of camera '" + camera.label + "' is zero over the band, so the thermal sensor atmosphere cannot be applied.");
    }

    // Pixels hold the band's in-band radiance, the emission of a blackbody at the pixel's brightness temperature with no spectral response (RadiationModel emits sigma*T^4 times the in-band blackbody fraction). Tabulate
    // that radiance against temperature, together with the radiance the sensor receives from the same blackbody through the atmosphere, weighted by the spectral response like the upwelling radiance.
    std::vector<float> table_band_radiance;
    std::vector<float> table_transmitted_radiance;
    for (float temperature = min_brightness_temperature; temperature <= max_brightness_temperature + 1e-3f; temperature += brightness_temperature_step) {
        const float band_fraction = blackbodyBandFraction(bounds.x, bounds.y, temperature);
        if (!(band_fraction > 0.f)) {
            helios_runtime_error("ERROR (RadiationModel::runBand): Emission band '" + band + "' of camera '" + camera.label + "' holds no blackbody emission at " + std::to_string(temperature) +
                                 " K, so the thermal sensor atmosphere cannot be applied to it.");
        }
        float transmitted_radiance = 0.f;
        for (size_t i = 0; i < grid_wavelengths.size(); i++) {
            transmitted_radiance += grid_weights.at(i) * grid_transmittance.at(i) * blackbodySpectralRadiance(grid_wavelengths.at(i), temperature);
        }
        table_band_radiance.push_back(stefan_boltzmann * powf(temperature, 4) * band_fraction / float(M_PI));
        table_transmitted_radiance.push_back(transmitted_radiance);
    }

    std::vector<float> &pixels = camera.pixel_data.at(band);
    for (float &pixel: pixels) {
        if (pixel < table_band_radiance.front() || pixel > table_band_radiance.back()) {
            helios_runtime_error("ERROR (RadiationModel::runBand): A pixel of emission band '" + band + "' of camera '" + camera.label + "' has radiance " + std::to_string(pixel) + " W/m^2/sr, outside the range of brightness temperatures (" +
                                 std::to_string(min_brightness_temperature) + "-" + std::to_string(max_brightness_temperature) + " K) supported by the thermal sensor atmosphere.");
        }
        const size_t upper = std::distance(table_band_radiance.begin(), std::lower_bound(table_band_radiance.begin(), table_band_radiance.end(), pixel));
        const size_t lower = upper == 0 ? 0 : upper - 1;
        const float weight = upper == lower ? 0.f : (pixel - table_band_radiance[lower]) / (table_band_radiance[upper] - table_band_radiance[lower]);
        pixel = table_transmitted_radiance[lower] + weight * (table_transmitted_radiance[upper] - table_transmitted_radiance[lower]) + band_upwelling_radiance;
    }
    const std::string data_label = "camera_" + camera.label + "_" + band;
    context->setGlobalData(data_label.c_str(), pixels);
}

void RadiationModel::computeCameraSky(const std::vector<std::string> &band_labels, const RadiationCamera &camera, uint camera_index, bool diffuse_rays_launched, std::vector<float> &camera_sky_radiance,
                                      std::vector<helios::vec4> &sky_pattern_params) const {

    size_t Nbands_launch = band_labels.size();
    camera_sky_radiance.assign(Nbands_launch, 0.0f);
    sky_pattern_params.assign(Nbands_launch, helios::make_vec4(0.f, 0.f, 0.f, 0.f));

    int prague_valid = 0;
    if (context->doesGlobalDataExist("prague_sky_valid")) {
        context->getGlobalData("prague_sky_valid", prague_valid);
    }

    if (prague_valid == 1 && !context->doesGlobalDataExist("prague_sky_spectral_params")) {
        helios_runtime_error("ERROR (RadiationModel::computeCameraSky): Global data \"prague_sky_valid\" is set, but \"prague_sky_spectral_params\" does not exist. Call SolarPosition::updatePragueSkyModel() to compute the Prague sky model, or remove \"prague_sky_valid\" to use the diffuse sky.");
    }

    if (prague_valid != 1) {
        // Without the Prague sky model the camera sees the diffuse sky that lights the scene (setDiffuseRadiationFlux()). The diffuse launches
        // deliver radiance flux * f(direction) / pi from each direction, where f is the band's angular distribution (isotropic, or the power law
        // of setDiffuseRadiationExtinctionCoeff()) normalized so that a horizontal surface receives the band's diffuse flux. The camera weights
        // this light as it weights diffuse light reflected by a perfectly white surface (its diffuse-slot white reference), so a white surface
        // under this sky looks as bright as the sky itself. The pattern parameters stay zero, so the camera miss programs evaluate the band's
        // own angular distribution.
        if (!diffuse_rays_launched) {
            return; // no diffuse light reached the scene, so there is no sky to see
        }
        const size_t Nbands_global = radiation_bands.size();
        const size_t Ncameras = cameras.size();
        const size_t diffuse_slot = radiation_sources.size();
        for (size_t b = 0; b < Nbands_launch; b++) {
            const std::string &band_label = band_labels[b];
            const RadiationBand &band = radiation_bands.at(band_label);
            // Emission bands see the sky's longwave flux through their own path in the camera miss programs
            if (band.emissionFlag || std::find(camera.band_labels.begin(), camera.band_labels.end(), band_label) == camera.band_labels.end()) {
                continue;
            }
            const float flux = getDiffuseFlux(band_label);
            if (flux <= 0.f) {
                continue;
            }
            float white_reference = 1.f;
            if (!material_data.white_reference_cam.empty()) {
                const size_t global_band = std::distance(radiation_bands.begin(), radiation_bands.find(band_label));
                const size_t index = (diffuse_slot * Nbands_global + global_band) * Ncameras + camera_index;
                if (index >= material_data.white_reference_cam.size()) {
                    helios_runtime_error("ERROR (RadiationModel::computeCameraSky): The camera white references were computed for a different set of sources, bands or cameras than camera '" + camera.label +
                                         "' is being rendered with. Call updateGeometry() after adding sources, bands or cameras.");
                }
                white_reference = material_data.white_reference_cam.at(index);
            }
            camera_sky_radiance[b] = flux * white_reference / float(M_PI);
        }
        return;
    }

    // Prague Sky Model (SolarPosition::updatePragueSkyModel()): spectral sky radiance from brute-force atmospheric simulations
    // (Wilkie et al. 2021, Vévoda et al. 2022), integrated over this camera's spectral responses

    std::vector<float> spectral_params;
    context->getGlobalData("prague_sky_spectral_params", spectral_params);

    const int params_per_wavelength = 6;
    const int n_wavelengths = spectral_params.size() / params_per_wavelength;

    std::vector<float> wavelengths(n_wavelengths);
    std::vector<float> L_zenith_spectrum(n_wavelengths);
    std::vector<float> circ_str_spectrum(n_wavelengths);
    std::vector<float> circ_width_spectrum(n_wavelengths);
    std::vector<float> horiz_bright_spectrum(n_wavelengths);

    for (int i = 0; i < n_wavelengths; ++i) {
        int base = i * params_per_wavelength;
        wavelengths[i] = spectral_params[base + 0];
        L_zenith_spectrum[i] = spectral_params[base + 1];
        circ_str_spectrum[i] = spectral_params[base + 2];
        circ_width_spectrum[i] = spectral_params[base + 3];
        horiz_bright_spectrum[i] = spectral_params[base + 4];
    }

    // --- Process each band ---
    for (size_t b = 0; b < Nbands_launch; b++) {
        const std::string &band_label = band_labels[b];
        if (radiation_bands.find(band_label) == radiation_bands.end()) {
            continue;
        }

        const RadiationBand &band = radiation_bands.at(band_label);

        // Skip thermal/longwave bands - Prague Sky Model only handles shortwave radiation
        if (band.emissionFlag) {
            continue;
        }

        // A band this camera does not render has no sky for it
        if (std::find(camera.band_labels.begin(), camera.band_labels.end(), band_label) == camera.band_labels.end()) {
            continue;
        }

        // Get camera spectral response for this band
        std::string spectral_response_label = "uniform";
        if (camera.band_spectral_response.find(band_label) != camera.band_spectral_response.end()) {
            spectral_response_label = camera.band_spectral_response.at(band_label);
            if (spectral_response_label.empty() || trim_whitespace(spectral_response_label).empty()) {
                spectral_response_label = "uniform";
            }
        }

        // Get camera spectral response data
        std::vector<helios::vec2> camera_response;

        if (spectral_response_label == "uniform") {
            helios::vec2 wavelength_range = band.wavebandBounds;

            if (!band.hasWavebandBounds()) {
                bool bounds_inferred = false;

                if (band_label == "red" || band_label == "R") {
                    wavelength_range = helios::make_vec2(620.f, 750.f);
                    bounds_inferred = true;
                } else if (band_label == "green" || band_label == "G") {
                    wavelength_range = helios::make_vec2(495.f, 570.f);
                    bounds_inferred = true;
                } else if (band_label == "blue" || band_label == "B") {
                    wavelength_range = helios::make_vec2(450.f, 495.f);
                    bounds_inferred = true;
                }

                if (!bounds_inferred) {
                    if (!band.diffuse_spectrum.empty()) {
                        wavelength_range.x = band.diffuse_spectrum.front().x;
                        wavelength_range.y = band.diffuse_spectrum.back().x;
                    } else {
                        helios_runtime_error("ERROR (RadiationModel::computeCameraSky): Camera '" + camera.label + "' band '" + band_label + "' has uniform spectral response but no wavelength bounds set.");
                    }
                }
            }

            camera_response.push_back(helios::make_vec2(wavelength_range.x, 1.0f));
            camera_response.push_back(helios::make_vec2(wavelength_range.y, 1.0f));

        } else {
            camera_response = loadSpectralData(spectral_response_label);

            if (camera_response.empty()) {
                helios_runtime_error("ERROR (RadiationModel::computeCameraSky): Camera spectral response '" + spectral_response_label + "' not found for camera '" + camera.label + "' band '" + band_label + "'.");
            }
        }

        // Integrate radiance and weight-average angular parameters over camera response
        // L_zenith: Integrate to get W/m²/sr (band-integrated radiance)
        float integrated_L_zenith = integrateOverResponse(wavelengths, L_zenith_spectrum, camera_response);

        // Angular parameters: Weighted average (unitless quantities)
        // Weight by L_zenith(λ) × R(λ) to get radiance-weighted average
        float integrated_circ_str = weightedAverageOverResponse(wavelengths, circ_str_spectrum, L_zenith_spectrum, camera_response);
        float integrated_circ_width = weightedAverageOverResponse(wavelengths, circ_width_spectrum, L_zenith_spectrum, camera_response);
        float integrated_horiz_bright = weightedAverageOverResponse(wavelengths, horiz_bright_spectrum, L_zenith_spectrum, camera_response);

        // Recompute normalization from averaged angular parameters
        float integrated_norm = computeAngularNormalization(integrated_circ_str, integrated_circ_width, integrated_horiz_bright);

        // The camera miss programs render the sky as camera_sky_radiance * evaluateDiffuseAngularDistribution(), and for these
        // parameters that distribution returns pattern * normalization * pi -- the form the diffuse launches need, whose
        // cosine-weighted integral over the hemisphere is pi. The sky radiance is L_zenith * pattern, so the base radiance divides
        // out both the normalization and pi. The parameters uploaded with it must be these same ones, so that the normalization
        // the device multiplies by is the one divided out here.
        camera_sky_radiance[b] = integrated_L_zenith / (integrated_norm * float(M_PI));
        sky_pattern_params[b] = helios::make_vec4(integrated_circ_str, integrated_circ_width, integrated_horiz_bright, integrated_norm);
    }
}

void RadiationModel::updatePragueParametersForGeneralDiffuse(const std::vector<std::string> &band_labels) {
    // Update Prague sky model angular parameters for general diffuse radiation
    // Reads spectral parameters from Context (set by SolarPosition::updatePragueSkyModel())
    // Integrates over band spectral response to get band-averaged parameters

    // Check Prague data availability
    int prague_valid = 0;
    if (!context->doesGlobalDataExist("prague_sky_valid") || (context->getGlobalData("prague_sky_valid", prague_valid), prague_valid != 1)) {
        // No Prague data - leave params at zero (will use power-law or isotropic)
        return;
    }

    // Read spectral parameters from Context
    std::vector<float> spectral_params;
    context->getGlobalData("prague_sky_spectral_params", spectral_params);

    // Parse into wavelength-resolved arrays
    const int params_per_wavelength = 6;
    const int n_wavelengths = spectral_params.size() / params_per_wavelength;

    std::vector<float> wavelengths(n_wavelengths);
    std::vector<float> L_zenith_spectrum(n_wavelengths);
    std::vector<float> circ_str_spectrum(n_wavelengths);
    std::vector<float> circ_width_spectrum(n_wavelengths);
    std::vector<float> horiz_bright_spectrum(n_wavelengths);
    std::vector<float> norm_spectrum(n_wavelengths);

    for (int i = 0; i < n_wavelengths; ++i) {
        int base = i * params_per_wavelength;
        wavelengths[i] = spectral_params[base + 0];
        L_zenith_spectrum[i] = spectral_params[base + 1];
        circ_str_spectrum[i] = spectral_params[base + 2];
        circ_width_spectrum[i] = spectral_params[base + 3];
        horiz_bright_spectrum[i] = spectral_params[base + 4];
        norm_spectrum[i] = spectral_params[base + 5];
    }

    // Get sun direction
    helios::vec3 sun_dir;
    context->getGlobalData("prague_sky_sun_direction", sun_dir);

    // Process each band
    for (const auto &label: band_labels) {
        RadiationBand &band = radiation_bands.at(label);

        // SKIP if user has explicitly set power-law (priority 1)
        if (band.diffuseExtinction > 0.0f) {
            continue;
        }

        // Integrate Prague parameters over band spectrum
        std::vector<helios::vec2> band_spectrum = band.diffuse_spectrum;
        if (band_spectrum.empty()) {
            // Use waveband bounds if no detailed spectrum
            float lambda_min = band.wavebandBounds.x;
            float lambda_max = band.wavebandBounds.y;
            if (band.hasWavebandBounds()) {
                band_spectrum = {{lambda_min, 1.0f}, {lambda_max, 1.0f}};
            }
        }

        if (band_spectrum.empty()) {
            // No spectral info - skip Prague for this band
            continue;
        }

        // Weighted integration (weight by L_zenith for physical consistency)
        float int_circ_str = weightedAverageOverResponse(wavelengths, circ_str_spectrum, L_zenith_spectrum, band_spectrum);
        float int_circ_width = weightedAverageOverResponse(wavelengths, circ_width_spectrum, L_zenith_spectrum, band_spectrum);
        float int_horiz_bright = weightedAverageOverResponse(wavelengths, horiz_bright_spectrum, L_zenith_spectrum, band_spectrum);

        // Recompute normalization from integrated parameters
        float int_norm = computeAngularNormalization(int_circ_str, int_circ_width, int_horiz_bright);

        // Store in RadiationBand
        band.diffusePragueParams = helios::make_vec4(int_circ_str, int_circ_width, int_horiz_bright, int_norm);
        band.diffusePeakDir = sun_dir;
    }
}

float RadiationModel::integrateOverResponse(const std::vector<float> &wavelengths, const std::vector<float> &values, const std::vector<helios::vec2> &camera_response) const {

    if (wavelengths.empty() || camera_response.empty()) {
        return 0.0f;
    }

    // CRITICAL: This integrates spectral radiance L(λ) in W/m²/sr/nm over wavelength
    // to produce band-integrated radiance in W/m²/sr (same as Prague computeIntegratedSkyRadiance)
    double integrated_radiance = 0.0;

    // Trapezoidal integration over camera response wavelength range
    for (size_t i = 0; i < camera_response.size() - 1; ++i) {
        float lambda1 = camera_response[i].x;
        float lambda2 = camera_response[i + 1].x;

        // Skip if outside spectral data range
        if (lambda2 < wavelengths.front() || lambda1 > wavelengths.back()) {
            continue;
        }

        float r1 = camera_response[i].y;
        float r2 = camera_response[i + 1].y;

        // Interpolate spectral values at these wavelengths using linear interpolation
        float v1, v2;

        // Interpolate v1 at lambda1
        if (lambda1 <= wavelengths.front()) {
            v1 = values.front();
        } else if (lambda1 >= wavelengths.back()) {
            v1 = values.back();
        } else {
            auto it = std::lower_bound(wavelengths.begin(), wavelengths.end(), lambda1);
            size_t idx = std::distance(wavelengths.begin(), it);
            if (idx == 0)
                idx = 1;
            float t = (lambda1 - wavelengths[idx - 1]) / (wavelengths[idx] - wavelengths[idx - 1]);
            v1 = values[idx - 1] + t * (values[idx] - values[idx - 1]);
        }

        // Interpolate v2 at lambda2
        if (lambda2 <= wavelengths.front()) {
            v2 = values.front();
        } else if (lambda2 >= wavelengths.back()) {
            v2 = values.back();
        } else {
            auto it = std::lower_bound(wavelengths.begin(), wavelengths.end(), lambda2);
            size_t idx = std::distance(wavelengths.begin(), it);
            if (idx == 0)
                idx = 1;
            float t = (lambda2 - wavelengths[idx - 1]) / (wavelengths[idx] - wavelengths[idx - 1]);
            v2 = values[idx - 1] + t * (values[idx] - values[idx - 1]);
        }

        float dlambda = lambda2 - lambda1;

        // Integrate: ∫ L(λ) × R(λ) dλ
        // L(λ) in W/m²/sr/nm, R(λ) unitless, dλ in nm → result in W/m²/sr
        integrated_radiance += 0.5 * (v1 * r1 + v2 * r2) * dlambda;
    }

    // Return band-integrated radiance in W/m²/sr (matches Prague computeIntegratedSkyRadiance)
    return static_cast<float>(integrated_radiance);
}

float RadiationModel::weightedAverageOverResponse(const std::vector<float> &wavelengths, const std::vector<float> &param_values, const std::vector<float> &weight_values, const std::vector<helios::vec2> &camera_response) const {

    if (wavelengths.empty() || camera_response.empty()) {
        return 0.0f;
    }

    // CRITICAL: Angular parameters are unitless - compute radiance-weighted average
    // Formula: Σ param(λ) × L(λ) × R(λ) dλ / Σ L(λ) × R(λ) dλ
    double weighted_sum = 0.0;
    double total_weight = 0.0;

    for (size_t i = 0; i < camera_response.size() - 1; ++i) {
        float lambda1 = camera_response[i].x;
        float lambda2 = camera_response[i + 1].x;

        if (lambda2 < wavelengths.front() || lambda1 > wavelengths.back()) {
            continue;
        }

        float r1 = camera_response[i].y;
        float r2 = camera_response[i + 1].y;

        // Interpolate parameter values
        float p1, p2;
        if (lambda1 <= wavelengths.front()) {
            p1 = param_values.front();
        } else if (lambda1 >= wavelengths.back()) {
            p1 = param_values.back();
        } else {
            auto it = std::lower_bound(wavelengths.begin(), wavelengths.end(), lambda1);
            size_t idx = std::distance(wavelengths.begin(), it);
            if (idx == 0)
                idx = 1;
            float t = (lambda1 - wavelengths[idx - 1]) / (wavelengths[idx] - wavelengths[idx - 1]);
            p1 = param_values[idx - 1] + t * (param_values[idx] - param_values[idx - 1]);
        }

        if (lambda2 <= wavelengths.front()) {
            p2 = param_values.front();
        } else if (lambda2 >= wavelengths.back()) {
            p2 = param_values.back();
        } else {
            auto it = std::lower_bound(wavelengths.begin(), wavelengths.end(), lambda2);
            size_t idx = std::distance(wavelengths.begin(), it);
            if (idx == 0)
                idx = 1;
            float t = (lambda2 - wavelengths[idx - 1]) / (wavelengths[idx] - wavelengths[idx - 1]);
            p2 = param_values[idx - 1] + t * (param_values[idx] - param_values[idx - 1]);
        }

        // Interpolate weight (radiance) values
        float w1, w2;
        if (lambda1 <= wavelengths.front()) {
            w1 = weight_values.front();
        } else if (lambda1 >= wavelengths.back()) {
            w1 = weight_values.back();
        } else {
            auto it = std::lower_bound(wavelengths.begin(), wavelengths.end(), lambda1);
            size_t idx = std::distance(wavelengths.begin(), it);
            if (idx == 0)
                idx = 1;
            float t = (lambda1 - wavelengths[idx - 1]) / (wavelengths[idx] - wavelengths[idx - 1]);
            w1 = weight_values[idx - 1] + t * (weight_values[idx] - weight_values[idx - 1]);
        }

        if (lambda2 <= wavelengths.front()) {
            w2 = weight_values.front();
        } else if (lambda2 >= wavelengths.back()) {
            w2 = weight_values.back();
        } else {
            auto it = std::lower_bound(wavelengths.begin(), wavelengths.end(), lambda2);
            size_t idx = std::distance(wavelengths.begin(), it);
            if (idx == 0)
                idx = 1;
            float t = (lambda2 - wavelengths[idx - 1]) / (wavelengths[idx] - wavelengths[idx - 1]);
            w2 = weight_values[idx - 1] + t * (weight_values[idx] - weight_values[idx - 1]);
        }

        float dlambda = lambda2 - lambda1;

        // Weighted average: Σ param × weight × response × dλ
        weighted_sum += 0.5 * (p1 * w1 * r1 + p2 * w2 * r2) * dlambda;
        total_weight += 0.5 * (w1 * r1 + w2 * r2) * dlambda;
    }

    // Return weighted average (unitless)
    if (total_weight > 1e-10) {
        return static_cast<float>(weighted_sum / total_weight);
    }
    return 0.0f;
}

float RadiationModel::computeAngularNormalization(float circ_str, float circ_width, float horiz_bright) const {
    // Numerical integration of angular pattern over hemisphere
    const int N = 50;
    float integral = 0.0f;

    // Sun at zenith for normalization calculation
    helios::vec3 sun_dir = make_vec3(0, 0, 1);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float theta = 0.5f * float(M_PI) * (i + 0.5f) / N; // 0 to π/2
            float phi = 2.0f * float(M_PI) * (j + 0.5f) / N; // 0 to 2π

            helios::vec3 dir = sphere2cart(make_SphericalCoord(0.5f * float(M_PI) - theta, phi));

            // Angular distance from sun (degrees) - matches GPU calculation
            float cos_gamma = std::max(-1.0f, std::min(1.0f, dir.x * sun_dir.x + dir.y * sun_dir.y + dir.z * sun_dir.z));
            float gamma = std::acos(cos_gamma) * 180.0f / float(M_PI);

            // Compute angular pattern (same as GPU: rayHit.cu evaluateDiffuseAngularDistribution)
            float cos_theta = std::max(0.0f, dir.z);
            float horizon_term = 1.0f + (horiz_bright - 1.0f) * (1.0f - cos_theta);
            float circ_term = 1.0f + circ_str * std::exp(-gamma / circ_width);

            float pattern = circ_term * horizon_term;

            // Solid angle element: sin(θ) dθ dφ
            integral += pattern * std::cos(theta) * std::sin(theta) * (float(M_PI) / (2.0f * N)) * (2.0f * float(M_PI) / N);
        }
    }

    return 1.0f / std::max(integral, 1e-10f);
}

void RadiationModel::applySpectralClassMaps() {
    applySpectralClassMaps(context->getAllUUIDs());
}

void RadiationModel::applySpectralClassMaps(const std::vector<uint> &UUIDs) {

    // The primitives of each class map
    std::map<std::string, std::vector<uint>> UUIDs_by_class_map;
    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID) || !context->doesPrimitiveDataExist(UUID, "spectral_class_map")) {
            continue;
        }
        if (context->getPrimitiveDataType("spectral_class_map") != HELIOS_TYPE_STRING) {
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The primitive data \"spectral_class_map\" must be a string (the path of a class map).");
        }
        std::string class_map_file;
        context->getPrimitiveData(UUID, "spectral_class_map", class_map_file);
        UUIDs_by_class_map[class_map_file].push_back(UUID);
    }

    for (const auto &[class_map_file, map_UUIDs]: UUIDs_by_class_map) {

        // ---- The class definitions ---- //
        if (!std::filesystem::exists(class_map_file)) {
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The class map '" + class_map_file + "' given by the primitive data \"spectral_class_map\" does not exist.");
        }
        const std::string definitions_file = std::filesystem::path(class_map_file).replace_extension(".xml").string();
        if (!std::filesystem::exists(definitions_file)) {
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The class map '" + class_map_file + "' has no class definitions. They must be in the XML file '" + definitions_file +
                                 "' (see the radiation plugin documentation, \"Spectral Textures\").");
        }
        pugi::xml_document document;
        const pugi::xml_parse_result parsed = document.load_file(definitions_file.c_str());
        if (!parsed) {
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The class definitions '" + definitions_file + "' could not be read: " + std::string(parsed.description()) + ".");
        }
        const pugi::xml_node map_node = document.child("helios").child("surface_class_map");
        if (!map_node) {
            if (document.child("helios").child("leaf_class_map")) {
                // A leaf's classes are derived from the leaf's properties by LeafOptics, so the leaves of a plant given whole are left to it
                if (message_flag) {
                    std::cout << "RadiationModel::applySpectralClassMaps: " << map_UUIDs.size() << " primitives carry the leaf class map " << class_map_file << ", which is for LeafOptics::run(); they are left as they are." << std::endl;
                }
                continue;
            }
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The class definitions '" + definitions_file + "' have no <surface_class_map> element inside <helios>.");
        }
        const std::string reflectivity_label = map_node.attribute("reflectivity_spectrum").as_string();
        const std::string transmissivity_label = map_node.attribute("transmissivity_spectrum").as_string();
        const std::string provenance = map_node.attribute("provenance").as_string();
        if (reflectivity_label.empty()) {
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The class definitions '" + definitions_file + "' do not name a reflectivity spectrum (the reflectivity_spectrum attribute of <surface_class_map>).");
        }
        std::map<int, float> scales_by_id;
        for (const pugi::xml_node class_node: map_node.children("class")) {
            const pugi::xml_attribute id = class_node.attribute("id");
            const pugi::xml_attribute scale = class_node.attribute("reflectance_scale");
            if (!id || !scale) {
                helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): A class in '" + definitions_file + "' lacks the attribute id or reflectance_scale.");
            }
            const float value = scale.as_float(-1.f);
            if (!(value > 0.f) || !std::isfinite(value)) {
                helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): Class " + std::string(id.value()) + " in '" + definitions_file + "' has a reflectance_scale that is not a positive number.");
            }
            if (!scales_by_id.emplace(id.as_int(0), value).second) {
                helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): Class " + std::string(id.value()) + " is defined twice in '" + definitions_file + "'.");
            }
        }
        std::vector<float> scales;
        for (const auto &[id, scale]: scales_by_id) {
            if (id != int(scales.size()) + 1) {
                helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The classes in '" + definitions_file + "' must be numbered 1, 2, 3 and so on without gaps, but class " + std::to_string(scales.size() + 1) + " is missing.");
            }
            scales.push_back(scale);
        }
        if (scales.empty()) {
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The class definitions '" + definitions_file + "' define no classes.");
        }
        uint width, height;
        std::vector<uint16_t> classes;
        helios::readPNGSingleChannel(class_map_file, width, height, classes);
        const uint16_t highest = classes.empty() ? 0 : *std::max_element(classes.begin(), classes.end());
        if (highest > scales.size()) {
            helios_runtime_error("ERROR (RadiationModel::applySpectralClassMaps): The class map '" + class_map_file + "' has class " + std::to_string(highest) + ", but '" + definitions_file + "' defines only " + std::to_string(scales.size()) + " classes.");
        }

        // ---- The spectra of the classes ---- //
        const std::vector<helios::vec2> reflectivity = loadSpectralData(reflectivity_label);
        std::vector<helios::vec2> transmissivity;
        if (!transmissivity_label.empty()) {
            transmissivity = loadSpectralData(transmissivity_label);
        }
        const std::string map_name = std::filesystem::path(class_map_file).stem().string();
        std::vector<helios::SpectralTextureClass> palette;
        for (size_t k = 0; k < scales.size(); k++) {
            const std::string suffix = "_" + map_name + "_class" + std::to_string(k + 1);
            // A class brighter than the texture's mean scales the spectrum up; reflectivity and transmissivity are each held at 1 at most, and together at 1
            std::vector<helios::vec2> class_tau = transmissivity;
            for (helios::vec2 &point: class_tau) {
                point.y = std::fmin(point.y * scales.at(k), 1.f);
            }
            std::vector<helios::vec2> class_rho = reflectivity;
            for (helios::vec2 &point: class_rho) {
                const float tau = class_tau.empty() ? 0.f : interp1(class_tau, point.x);
                point.y = std::fmin(point.y * scales.at(k), 1.f - tau);
            }
            helios::SpectralTextureClass spectral_class;
            spectral_class.reflectivity_spectrum = reflectivity_label + suffix;
            context->setGlobalData(spectral_class.reflectivity_spectrum.c_str(), class_rho);
            if (!transmissivity_label.empty()) {
                spectral_class.transmissivity_spectrum = transmissivity_label + suffix;
                context->setGlobalData(spectral_class.transmissivity_spectrum.c_str(), class_tau);
            }
            spectral_class.provenance = provenance.empty() ? reflectivity_label + " scaled by " + std::to_string(scales.at(k)) : provenance;
            palette.push_back(spectral_class);
        }
        const std::string palette_label = "surface_" + map_name;
        context->addSpectralPalette(palette_label, palette);
        context->setPrimitiveSpectralTexture(map_UUIDs, class_map_file, palette_label);

        if (message_flag) {
            std::cout << "RadiationModel::applySpectralClassMaps: " << map_UUIDs.size() << " primitives given the spectral texture " << class_map_file << " (" << scales.size() << " classes of " << reflectivity_label << ")." << std::endl;
        }
    }
}

std::vector<helios::vec2> RadiationModel::loadSpectralData(const std::string &global_data_label) const {

    std::vector<helios::vec2> spectrum;

    if (!context->doesGlobalDataExist(global_data_label.c_str())) {

        // check if spectral data exists in any of the library files
        bool data_found = false;
        for (const auto &file: spectral_library_files) {
            if (Context::scanXMLForTag(file, "globaldata_vec2", global_data_label)) {
                context->loadXML(file.c_str(), true);
                data_found = true;
                break;
            }
        }

        if (!data_found) {
            helios_runtime_error("ERROR (RadiationModel::loadSpectralData): Global data for spectrum '" + global_data_label + "' could not be found.");
        }
    }

    if (context->getGlobalDataType(global_data_label.c_str()) != HELIOS_TYPE_VEC2) {
        helios_runtime_error("ERROR (RadiationModel::loadSpectralData): Global data for spectrum '" + global_data_label + "' is not of type HELIOS_TYPE_VEC2.");
    }

    context->getGlobalData(global_data_label.c_str(), spectrum);

    // validate spectrum
    if (spectrum.empty()) {
        helios_runtime_error("ERROR (RadiationModel::loadSpectralData): Global data for spectrum '" + global_data_label + "' is empty.");
    }
    for (auto s = 0; s < spectrum.size(); s++) {
        // check that wavelengths are monotonic
        if (s > 0 && spectrum.at(s).x <= spectrum.at(s - 1).x) {
            helios_runtime_error("ERROR (RadiationModel::loadSpectralData): Source spectral data validation failed. Wavelengths must increase monotonically.");
        }
        // check that wavelength is within a reasonable range
        if (spectrum.at(s).x < 0 || spectrum.at(s).x > 100000) {
            helios_runtime_error("ERROR (RadiationModel::loadSpectralData): Source spectral data validation failed. Wavelength value of " + std::to_string(spectrum.at(s).x) + " appears to be erroneous.");
        }
        // check that flux is non-negative
        if (spectrum.at(s).y < 0) {
            helios_runtime_error("ERROR (RadiationModel::loadSpectralData): Source spectral data validation failed. Flux value at wavelength of " + std::to_string(spectrum.at(s).x) + " appears is negative.");
        }
    }

    return spectrum;
}

void RadiationModel::runBand(const std::string &label) {
    std::vector<std::string> labels{label};
    runBand(labels);
}

//! Add one launch's camera-weighted scatter, downloaded from the backend, into the per-camera accumulators of runBand()
/**
 * The accumulators hold one [primitive][band] block per camera. A download of any other size means the backend's camera-scatter buffers are out of step with the
 * current camera set, and adding it element by element would read or write out of bounds, so it is an error.
 */
static void accumulateCameraScatter(const helios::RayTracingResults &launch_results, std::vector<float> &scatter_top_cam, std::vector<float> &scatter_bottom_cam, std::vector<float> &spectral_incident) {
    // The radiation reaching primitives with a spectral texture, from which cameras compute the outgoing radiation of each class
    if (launch_results.spectral_incident.size() != spectral_incident.size()) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The ray-tracing backend returned " + std::to_string(launch_results.spectral_incident.size()) + " values of the radiation reaching primitives with a spectral texture, but " +
                             std::to_string(spectral_incident.size()) + " are required.");
    }
    for (size_t i = 0; i < spectral_incident.size(); i++) {
        spectral_incident[i] += launch_results.spectral_incident[i];
    }
    if (launch_results.scatter_buff_top_cam.size() != scatter_top_cam.size() || launch_results.scatter_buff_bottom_cam.size() != scatter_bottom_cam.size()) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The ray-tracing backend returned " + std::to_string(launch_results.scatter_buff_top_cam.size()) +
                             " camera-weighted scatter values, but the current cameras, primitives and launched bands require " + std::to_string(scatter_top_cam.size()) +
                             ". The backend's camera-scatter buffers were not sized for the current camera set.");
    }
    for (size_t i = 0; i < scatter_top_cam.size(); i++) {
        scatter_top_cam[i] += launch_results.scatter_buff_top_cam[i];
        scatter_bottom_cam[i] += launch_results.scatter_buff_bottom_cam[i];
    }
}

void RadiationModel::runBand(const std::vector<std::string> &label) {

    //----- VERIFICATIONS -----//

    // Invalidate cached excitation APAR if radiative properties have changed since the
    // last populate (e.g., user changed source spectrum, source flux, geometry, etc.).
    // When `radiativepropertiesneedupdate` is true, `updateRadiativeProperties()` will
    // be rerun during this dispatch, so any previously-cached APAR is stale.
    if (radiativepropertiesneedupdate) {
        for (auto &kv : excitation_sets) {
            kv.second.populated = false;
        }
    }

    // Optimisation: if the user dispatches one or more regular bands and has at least one
    // SIF camera registered, piggy-back the auto-generated excitation bands onto this
    // dispatch. This merges PAR + excitation into a single ray-trace pass, saving a full
    // dispatch round-trip.
    //
    // Guards:
    //   - Don't piggy-back if any band in this dispatch is an SIF emission band — the
    //     emission-dispatch path has its own pre-hook (below) that handles excitation.
    //   - Don't piggy-back if any band is already an internal "_SIF_exc_*" band (we are
    //     already inside the recursive excitation dispatch — prevents infinite recursion).
    //   - Don't piggy-back if no SIF cameras are registered (no excitation sets exist).
    //   - Skip sets that are already populated (APAR is cached).
    //
    // Piggy-backed sets are added to the dispatch list here but NOT marked populated until
    // populateExcitationSet() runs post-dispatch (below).
    std::vector<std::string> effective_label = label;
    if (!excitation_sets.empty()) {
        bool label_has_sif_emission = false;
        bool label_has_excitation_band = false;
        for (const auto &b : label) {
            if (sif_emission_bands.count(b) > 0) label_has_sif_emission = true;
            if (b.size() >= 9 && b.compare(0, 9, "_SIF_exc_") == 0) label_has_excitation_band = true;
        }
        if (!label_has_sif_emission && !label_has_excitation_band) {
            for (auto &kv : excitation_sets) {
                ExcitationSet &exc = kv.second;
                if (exc.populated) continue;
                for (const auto &eb : exc.band_labels) {
                    effective_label.push_back(eb);
                }
            }
        }
    }

    // We need the band label strings to appear in the same order as in radiation_bands map.
    // this is because that was the order in which radiative properties were laid out when updateRadiativeProperties() was called
    std::vector<std::string> band_labels;
    for (auto &band: radiation_bands) {
        if (std::find(effective_label.begin(), effective_label.end(), band.first) != effective_label.end()) {
            band_labels.push_back(band.first);
        }
    }

    // Persist per-band emission-enabled state, scattering depth, and wavelength bounds to global data so that downstream plug-ins (e.g. the energy balance model) can identify which bands emit thermal radiation, with what
    // emissivity (a band without scattering emits as a blackbody), and what share of the blackbody spectrum each emits.
    for (const auto &band: radiation_bands) {
        context->setGlobalData(("emission_enabled_" + band.first).c_str(), (uint) (band.second.emissionFlag ? 1 : 0));
        context->setGlobalData(("scattering_depth_" + band.first).c_str(), band.second.scatteringDepth);
        const std::string bounds_label = "wavelength_bounds_" + band.first;
        if (band.second.hasWavebandBounds()) {
            context->setGlobalData(bounds_label.c_str(), band.second.wavebandBounds);
        } else if (context->doesGlobalDataExist(bounds_label.c_str())) {
            context->clearGlobalData(bounds_label.c_str());
        }
    }

    // Launches that include SIF excitation bands (run directly, or piggy-backed above) also record which face of each primitive absorbed
    // the radiation, which Fluspect-B needs to assign its backward and forward emission to the right faces. Other launches skip it.
    bool track_face_absorption = false;
    for (const auto &b: band_labels) {
        if (b.size() >= 9 && b.compare(0, 9, "_SIF_exc_") == 0) {
            track_face_absorption = true;
            break;
        }
    }

    // Check to make sure some geometry was added to the context
    if (context->getPrimitiveCount() == 0) {
        std::cerr << "WARNING (RadiationModel::runBand): No geometry was added to the context. There is nothing to simulate...exiting." << std::endl;
        return;
    }

    // Build the acceleration structure if it has never been built, or rebuild it if primitives
    // were added to or deleted from the Context since the last build. Without the rebuild the
    // trace would silently run against stale geometry.
    if (needsGeometryRebuild()) {
        updateGeometry();
    }

    // Check that all the bands passed to the runBand() method exist
    for (const std::string &band: label) {
        if (!doesBandExist(band)) {
            helios_runtime_error("ERROR (RadiationModel::runBand): Cannot run band " + band + " because it is not a valid band. Use addRadiationBand() function to add the band.");
        }
    }

    // if there are no radiation sources in the simulation, add at least one but with zero fluxes
    if (radiation_sources.empty()) {
        addCollimatedRadiationSource();
    }

    // --- SIF v2: prepare per-leaf emission for any SIF-flagged bands in this launch ---
    //
    // If any band in this dispatch is a user-defined SIF emission band, we need to:
    //   1. Run the auto-generated excitation bands (once per dispatch), to capture
    //      per-leaf APAR across 400-750 nm.
    //   2. Invoke computeSIFEmission() for each SIF-flagged emission band, populating
    //      sif_emission_buffer[band]. The regular emission loop (below) will then read
    //      sif_emission_buffer when it encounters the band and bypass Stefan-Boltzmann.
    //
    // Recursion guard: runExcitationBands() recursively calls runBand() for the internal
    // "_SIF_exc_*" bands — those bands are NOT in sif_emission_bands, so the hook is a no-op
    // inside the recursion. Also guard against running SIF hook while running a purely-excitation
    // band set (first band starts with underscore prefix).
    bool dispatch_has_sif_band = false;
    for (const auto &b : band_labels) {
        if (sif_emission_bands.count(b) > 0) {
            dispatch_has_sif_band = true;
            break;
        }
    }
    if (dispatch_has_sif_band) {
        // Soft override: SIF bands need emission enabled so that the Fluspect-derived
        // source flux is traced through the emission loop. If the user has disabled
        // emission on a SIF band (directly via disableEmission, or indirectly via
        // setSourceFlux/spectral configuration that implicitly silences emission),
        // re-enable it for this dispatch and warn once. This avoids the silent
        // zero-output failure mode while preserving the user's other controls.
        {
            helios::WarningAggregator sif_warn;
            sif_warn.setEnabled(message_flag);
            for (const auto &b : band_labels) {
                if (sif_emission_bands.count(b) > 0) {
                    auto &band = radiation_bands.at(b);
                    if (!band.emissionFlag) {
                        sif_warn.addWarning("sif_emission_reenabled",
                                            "Band '" + b + "' is bound to a SIF camera but has emission disabled. "
                                            "Re-enabling emission for this dispatch — SIF cameras require emission "
                                            "enabled so that Fluspect-B source flux is traced through the emission "
                                            "loop. Remove the disableEmission(\"" + b + "\") call to silence this warning.");
                        band.emissionFlag = true;
                    }
                }
            }
            sif_warn.report(std::cerr);
        }

        // Run any excitation sets that aren't already populated. Piggy-back in the
        // runBand() preamble may have pre-populated some or all sets during a prior
        // non-SIF dispatch in this frame. When radiativepropertiesneedupdate is true
        // (a user changed something between runBand calls), the preamble above already
        // reset populated=false, so we'll pick up fresh APAR here.
        runExcitationBands();
        for (const auto &b : band_labels) {
            if (sif_emission_bands.count(b) > 0) {
                computeSIFEmission(b);
            }
        }
    }

    // Check if any source spectra have changed in global data and reload if necessary
    for (auto &source: radiation_sources) {
        if (!source.source_spectrum_label.empty() && source.source_spectrum_label != "none") {
            uint64_t current_version = context->getGlobalDataVersion(source.source_spectrum_label.c_str());
            if (current_version != source.source_spectrum_version) {
                // Reload spectrum from global data
                source.source_spectrum = loadSpectralData(source.source_spectrum_label);
                source.source_spectrum_version = current_version;
                radiativepropertiesneedupdate = true;
            }
        }
    }

    // Check if global diffuse spectrum has changed and reload if necessary
    if (!global_diffuse_spectrum_label.empty() && global_diffuse_spectrum_label != "none") {
        uint64_t current_version = context->getGlobalDataVersion(global_diffuse_spectrum_label.c_str());
        if (current_version != global_diffuse_spectrum_version) {
            // Reload diffuse spectrum from global data
            global_diffuse_spectrum = loadSpectralData(global_diffuse_spectrum_label);
            global_diffuse_spectrum_version = current_version;
            // Also update all band diffuse spectra
            for (auto &band_pair: radiation_bands) {
                band_pair.second.diffuse_spectrum = global_diffuse_spectrum;
            }
            radiativepropertiesneedupdate = true;
        }
    }

    // Rebuild the material properties from the Context on every dispatch, so that primitive data edited between
    // calls (reflectivity_<band>, *_spectrum, glass_*, specular_*) is picked up without an explicit call. This is
    // the only material builder: it caches the spectral integrations per unique spectrum and applies the documented
    // precedence (a per-band value overrides a spectrum), and it uploads the result to the backend itself.
    updateRadiativeProperties();

    // Surface spectra that could not be weighted in a band are an error only if that band is being run
    for (const std::string &band: band_labels) {
        auto error_it = spectral_weighting_errors.find(band);
        if (error_it != spectral_weighting_errors.end()) {
            helios_runtime_error(error_it->second);
        }
    }

    // Upload sources to backend (always use new path)
    buildSourceData();
    backend->updateSources(source_data);

    // Prepare launch parameters (these will be passed to backend via RayTracingLaunchParams)
    size_t Nbands_launch = band_labels.size();
    size_t Nbands_global = radiation_bands.size();

    // Build band launch flags
    std::vector<char> band_launch_flag(Nbands_global);
    uint bb = 0;
    for (auto &band: radiation_bands) {
        if (std::find(band_labels.begin(), band_labels.end(), band.first) != band_labels.end()) {
            band_launch_flag.at(bb) = 1;
        }
        bb++;
    }

    // Get dimensions
    size_t Nobjects = primitiveID.size();
    size_t Nprimitives = context_UUIDs.size();
    uint Nsources = radiation_sources.size();
    uint Ncameras = cameras.size();

    // Note: Atmospheric sky radiance model is updated per-camera (see camera trace loop below)
    // This allows us to use camera-specific spectral responses for each band

    // Set scattering depth for each band
    std::vector<uint> scattering_depth(Nbands_launch);
    bool scatteringenabled = false;
    for (auto b = 0; b < Nbands_launch; b++) {
        scattering_depth.at(b) = radiation_bands.at(band_labels.at(b)).scatteringDepth;
        if (scattering_depth.at(b) > 0) {
            scatteringenabled = true;
        }
    }

    // Issue warning if rho>0, tau>0, or eps<1 for any band with scatteringDepth=0.
    // Internal SIF excitation bands (labels start with "_SIF_exc_") are silenced here
    // because LeafOptics automatically sets per-band rho/tau on leaves for the full
    // spectrum — a per-band warning would produce ~35 lines of noise. Instead,
    // runExcitationBands() emits ONE consolidated warning before dispatch covering
    // the entire excitation set.
    helios::WarningAggregator scattering_disabled_warnings;
    for (int b = 0; b < Nbands_launch; b++) {
        const std::string &bname = band_labels.at(b);
        const bool is_sif_excitation_band = bname.size() >= 9 && bname.compare(0, 9, "_SIF_exc_") == 0;
        if (scattering_depth.at(b) == 0 && scattering_iterations_needed.at(bname) && !is_sif_excitation_band) {
            scattering_disabled_warnings.addWarning("scattering_disabled_for_band",
                                                    "Surface radiative properties for band " + bname +
                                                            " are set to non-default values, but scattering iterations are disabled. Surface radiative properties will be ignored unless scattering depth is non-zero.");
        }
    }
    scattering_disabled_warnings.report(std::cerr);

    // Set diffuse flux for each band
    std::vector<float> diffuse_flux(Nbands_launch);
    bool diffuseenabled = false;
    for (auto b = 0; b < Nbands_launch; b++) {
        diffuse_flux.at(b) = getDiffuseFlux(band_labels.at(b));
        if (diffuse_flux.at(b) > 0.f) {
            diffuseenabled = true;
        }
    }
    // NOTE: diffuse_flux now passed to backend via launch params, not uploaded here

    // Update Prague parameters for general diffuse (if available in Context)
    // This must be done before uploading diffuse parameters to GPU
    if (diffuseenabled) {
        updatePragueParametersForGeneralDiffuse(band_labels);
    }

    // Set diffuse extinction coefficient for each band
    std::vector<float> diffuse_extinction(Nbands_launch, 0);
    if (diffuseenabled) {
        for (auto b = 0; b < Nbands_launch; b++) {
            diffuse_extinction.at(b) = radiation_bands.at(band_labels.at(b)).diffuseExtinction;
        }
    }
    // NOTE: diffuse_extinction now passed to backend via launch params, not uploaded here

    // Set diffuse distribution normalization factor for each band
    std::vector<float> diffuse_dist_norm(Nbands_launch, 0);
    if (diffuseenabled) {
        for (auto b = 0; b < Nbands_launch; b++) {
            diffuse_dist_norm.at(b) = radiation_bands.at(band_labels.at(b)).diffuseDistNorm;
        }
    }
    // NOTE: diffuse_dist_norm now passed to backend via launch params, not uploaded here

    // Set diffuse distribution peak direction for each band
    std::vector<helios::vec3> diffuse_peak_dir(Nbands_launch);
    if (diffuseenabled) {
        for (auto b = 0; b < Nbands_launch; b++) {
            helios::vec3 peak_dir = radiation_bands.at(band_labels.at(b)).diffusePeakDir;
            diffuse_peak_dir.at(b) = helios::make_vec3(peak_dir.x, peak_dir.y, peak_dir.z);
        }
    }
    // NOTE: diffuse_peak_dir now passed to backend via launch params, not uploaded here

    // Upload Prague parameters for general diffuse (reuses camera buffer)
    // This allows general diffuse to use Prague sky model if available
    std::vector<helios::vec4> prague_params(Nbands_launch);
    if (diffuseenabled) {
        for (auto b = 0; b < Nbands_launch; b++) {
            const auto &params = radiation_bands.at(band_labels.at(b)).diffusePragueParams;
            prague_params.at(b) = helios::make_vec4(params.x, params.y, params.z, params.w);
        }
        // Prague params are passed to the backend with each diffuse launch; each camera uploads its own with updateSkyModel()
    }

    // Determine whether emission is enabled for any band
    bool emissionenabled = false;
    for (auto b = 0; b < Nbands_launch; b++) {
        if (radiation_bands.at(band_labels.at(b)).emissionFlag) {
            emissionenabled = true;
        }
    }

    // Figure out the maximum direct ray count for all bands in this run and use this as the launch size
    size_t directRayCount = 0;
    for (const auto &band: label) {
        if (radiation_bands.at(band).directRayCount > directRayCount) {
            directRayCount = radiation_bands.at(band).directRayCount;
        }
    }

    // Figure out the maximum diffuse ray count for all bands in this run and use this as the launch size
    size_t diffuseRayCount = 0;
    for (const auto &band: label) {
        if (radiation_bands.at(band).diffuseRayCount > diffuseRayCount) {
            diffuseRayCount = radiation_bands.at(band).diffuseRayCount;
        }
    }

    // Figure out the maximum diffuse ray count for all bands in this run and use this as the launch size
    size_t scatteringDepth = 0;
    for (const auto &band: label) {
        if (radiation_bands.at(band).scatteringDepth > scatteringDepth) {
            scatteringDepth = radiation_bands.at(band).scatteringDepth;
        }
    }

    // Zero radiation buffers via backend
    backend->zeroRadiationBuffers(Nbands_launch, track_face_absorption);

    // Direct and diffuse rays launched by this call, and the count of those abandoned at the periodic wrap limit (reported below)
    backend->zeroAbandonedRayCount();
    size_t launched_rays = 0;

    // Size and zero the camera-weighted scatter buffers for the current camera set before the first launch. Every direct and
    // diffuse launch accumulates one block per camera whether or not this band scatters, so buffers left at the size of an
    // earlier camera set (a camera registered since the last call) would be indexed past their end.
    if (Ncameras > 0) {
        backend->zeroCameraScatterBuffers(Nbands_launch);
    }

    // Primitives with a specular surface layer send out two components of outgoing radiation (see
    // RayTracingMaterial::surface_model). The outgoing, scattered and retired buffers then hold the second component after the
    // first, starting at layer_offset, and the camera buffers after all the cameras' first components.
    const size_t outgoing_components = material_data.surface_layer_enabled ? 2 : 1;
    const size_t layer_offset = Nprimitives * Nbands_launch;

    std::vector<float> TBS_top, TBS_bottom;
    TBS_top.resize(Nbands_launch * Nprimitives, 0);
    TBS_bottom = TBS_top;

    std::map<std::string, std::vector<std::vector<float>>> radiation_in_camera;

    size_t maxRays = 1024 * 1024 * 1024; // maximum number of total rays in a launch

    // ***** DIRECT LAUNCH FROM ALL RADIATION SOURCES ***** //

    helios::int3 launch_dim_dir;

    bool rundirect = false;
    for (uint s = 0; s < Nsources; s++) {
        for (uint b = 0; b < Nbands_launch; b++) {
            if (getSourceFlux(s, band_labels.at(b)) > 0.f) {
                rundirect = true;
                break;
            }
        }
    }

    // Keep the device-side source_fluxes buffer consistent with Nsources for EVERY launch type.
    // runBand() guarantees at least one source exists (a default collimated source is added above
    // when none were created), so Nsources >= 1 here. The miss/closesthit device programs iterate
    // [0, Nsources) and dereference source_fluxes[s*Nbands_launch + b] unconditionally. When no
    // source has positive flux (rundirect == false) the direct-pass block below is skipped, but a
    // camera trace still launches (e.g. an emission-only longwave band sampling the sky on a miss).
    // Uploading the per-band fluxes here (zeros/unset values are harmlessly skipped device-side by
    // the `flux <= 0` guard) prevents source_fluxes from being left null while Nsources > 0, which
    // would otherwise cause an illegal memory access in the camera kernel.
    if (Nsources > 0) {
        std::vector<std::vector<float>> source_flux_values(Nsources, std::vector<float>(Nbands_launch, 0.f));
        for (uint s = 0; s < Nsources; s++) {
            for (uint b = 0; b < Nbands_launch; b++) {
                source_flux_values.at(s).at(b) = getSourceFlux(s, band_labels.at(b));
            }
        }
        backend->uploadSourceFluxes(flatten(source_flux_values));
    }

    if (Nsources > 0 && rundirect) {

        // update radiation source buffers

        std::vector<std::vector<float>> fluxes; // first index is the source, second index is the band (only those passed to runBand() function)
        fluxes.resize(Nsources);
        std::vector<helios::vec3> positions(Nsources);
        std::vector<helios::vec2> widths(Nsources);
        std::vector<helios::vec3> rotations(Nsources);
        std::vector<uint> types(Nsources);

        size_t s = 0;
        for (const auto &source: radiation_sources) {

            fluxes.at(s).resize(Nbands_launch);

            // Iterate over Nbands_launch (== band_labels.size()), NOT label.size(). When SIF cameras
            // are registered, runBand() appends auto-generated excitation bands to band_labels, so
            // Nbands_launch > label.size(). Using label.size() would leave the trailing band slots at
            // zero and upload zero direct flux for those bands.
            for (size_t b = 0; b < Nbands_launch; b++) {
                fluxes.at(s).at(b) = getSourceFlux(s, band_labels.at(b));
            }

            positions.at(s) = helios::make_vec3(source.source_position.x, source.source_position.y, source.source_position.z);
            widths.at(s) = helios::make_vec2(source.source_width.x, source.source_width.y);
            rotations.at(s) = helios::make_vec3(source.source_rotation.x, source.source_rotation.y, source.source_rotation.z);
            types.at(s) = source.source_type;

            s++;
        }

        // Upload band-specific source fluxes to backend buffer (indexed by Nbands_launch, not Nbands_global)
        backend->uploadSourceFluxes(flatten(fluxes));
        // Note: positions, widths, rotations, types are uploaded once in buildSourceData()
        // Only fluxes need per-launch update because they depend on which bands are being run

        // Camera response weighting factors for specular reflection (if cameras exist). A specularly reflected source is seen as a spectrally
        // white reflector would see it, so each source's factor is its white reference, computed in updateRadiativeProperties() with the same
        // integral and normalization as its camera-weighted reflectivity: ∫(source_spectrum × camera_response) / ∫(source_spectrum) over the
        // wavelengths where both are tabulated, or 1 without a camera response. This must be done before ray tracing so the weights are
        // available during miss_direct().
        if (Ncameras > 0) {
            std::vector<float> source_fluxes_cam;
            source_fluxes_cam.resize(Nsources * Nbands_launch * Ncameras, 1.0f);

            std::vector<size_t> global_band_index(Nbands_launch);
            for (uint b = 0; b < Nbands_launch; b++) {
                global_band_index.at(b) = std::distance(radiation_bands.begin(), radiation_bands.find(band_labels.at(b)));
            }

            for (uint s = 0; s < Nsources; s++) {
                for (uint cam = 0; cam < Ncameras; cam++) {
                    for (uint b = 0; b < Nbands_launch; b++) {
                        source_fluxes_cam[s * Nbands_launch * Ncameras + b * Ncameras + cam] = material_data.white_reference_cam.at(((size_t) s * Nbands_global + global_band_index.at(b)) * Ncameras + cam);
                    }
                }
            }

            // Update source_data with camera-weighted fluxes and re-upload to backend
            for (uint s = 0; s < Nsources; s++) {
                source_data[s].fluxes_cam.clear();
                for (uint b = 0; b < Nbands_launch; b++) {
                    for (uint cam = 0; cam < Ncameras; cam++) {
                        source_data[s].fluxes_cam.push_back(source_fluxes_cam[s * Nbands_launch * Ncameras + b * Ncameras + cam]);
                    }
                }
            }
            backend->updateSources(source_data);
        }

        // -- Ray Trace (Using Backend) -- //

        if (message_flag) {
            std::cout << "Performing primary direct radiation ray trace for bands ";
            for (const auto &band: label) {
                std::cout << band << ", ";
            }
            std::cout << "..." << std::flush;
        }

        // Launch direct rays through backend
        helios::RayTracingLaunchParams params;
        params.launch_offset = 0;
        params.launch_count = Nprimitives; // Launch all primitives at once
        params.rays_per_primitive = directRayCount;
        params.random_seed = drawLaunchSeed();
        params.num_bands_global = Nbands_global;
        params.num_bands_launch = Nbands_launch;
        // For the direct launch the flag asks for the direct flux reaching primitives to be recorded: those with a surface layer, and with camera flux smoothing, every smoothed one
        params.specular_reflection_enabled = (specular_reflection_mode != 0 || (cameraflux_smoothing_enabled && smoothing_vertex_count > 0)) ? 1 : 0;

        // Use the band_launch_flag already built above (lines 3375-3383)
        std::vector<bool> band_flags(band_launch_flag.begin(), band_launch_flag.end());
        params.band_launch_flag = band_flags;

        backend->launchDirectRays(params);
        launched_rays += size_t(Nprimitives) * directRayCount * Nsources;

        if (message_flag) {
            std::cout << "done." << std::endl;
        }

    } // end direct source launch

    // --- Extract scattered energy from direct rays for diffuse/emission and scattering ---//
    // This needs to happen BEFORE diffuse/emission block so scattered direct energy
    // is available for both diffuse/emission (via flux_top/flux_bottom) and scattering (via radiation_out)
    std::vector<float> flux_top, flux_bottom;
    flux_top.resize(outgoing_components * Nbands_launch * Nprimitives, 0);
    flux_bottom = flux_top;

    // Reference blackbody field for emission bands. If every surface and the sky radiated at one temperature, the radiation
    // field would be uniform and each face would absorb exactly (1 - rho - tau) times its blackbody emissive power, whatever the
    // geometry. Emission is therefore traced as its departure from such a field at the area-weighted mean emissive power of the
    // scene (reference_flux, W/m^2 per band), and the sky as its departure from the reference radiance; what each face absorbs
    // from the reference field is added back exactly (reference_absorption). The gathered estimate then carries noise only in
    // proportion to the temperature differences in the scene, and an isothermal scene is in exact equilibrium at any ray count.
    // It is enabled per band with enableEmissionReferenceField(); a band that does not use it has a reference_flux of 0.
    std::vector<float> reference_flux(Nbands_launch, 0.f);
    std::vector<float> reference_absorption(Nbands_launch * Nprimitives, 0.f);

    // scatteringDepth is per band, but all bands in the dispatch share one scattering loop that runs to the deepest band's
    // depth. When a band's scattering is done it is "retired": the outgoing energy it was about to launch is zeroed in the
    // outgoing buffers, so the band is still traced but carries no energy. For a band with a depth of 0 (or one stopped by its
    // threshold before any scattering launch) that energy is moved into these accumulators and credited back as absorbed at the
    // end; a band that has scattered gets the summed tail below instead.
    //
    // A band must NOT be retired by clearing its entry in band_launch_flag. That array defines the band -> launch slot
    // mapping ([primitive * Nbands_launch + slot], slot = position among the set flags) that the host and every backend
    // recompute independently, so changing it part way through the dispatch renumbers the slots of the remaining bands.
    std::vector<float> retired_top(outgoing_components * Nbands_launch * Nprimitives, 0.f);
    std::vector<float> retired_bottom(outgoing_components * Nbands_launch * Nprimitives, 0.f);
    std::vector<char> band_retired(Nbands_launch, 0);
    // Retires a band, moving the given outgoing components (all by default) to the retired accumulators
    auto retireBand = [&](size_t launch_slot, std::vector<float> &outgoing_top, std::vector<float> &outgoing_bottom, size_t retired_components = SIZE_MAX) {
        for (size_t component = 0; component < std::min(retired_components, outgoing_components); component++) {
            for (size_t p = 0; p < Nprimitives; p++) {
                const size_t ind = component * layer_offset + p * Nbands_launch + launch_slot;
                retired_top.at(ind) += outgoing_top.at(ind);
                retired_bottom.at(ind) += outgoing_bottom.at(ind);
                outgoing_top.at(ind) = 0.f;
                outgoing_bottom.at(ind) = 0.f;
            }
        }
        band_retired.at(launch_slot) = 1;
    };

    // A band with a scattering depth of at least 1 does not absorb the energy still to be scattered after its last
    // scattering iteration. Absorbing it where it sits overestimated absorption by about omega times the band's last
    // interception, which decays only as q^depth, where q, the fraction of an iteration's launched power that is scattered
    // again, is about 0.5 for leaves with rho = tau = 0.4. The remaining iterations are instead summed as a geometric
    // series: each primitive receives its absorption from the band's last launch times q/(1-q). Bands with a depth of 0
    // absorb what they intercept, as before.
    //
    // q is the forward estimate of the band's last launch (RayTracingBackend::getForwardScatterResults()): each primitive's
    // launched power times the mean fraction that the surfaces its own rays reach scatter again, over the launched power.
    // The ratio of the gathered power after the launch to the launched power has the same expectation, but every primitive
    // gathers it from its own rays, so a large primitive with few rays (a single ground patch in a late iteration) can be
    // credited from one hit on a small bright leaf with more power than all the leaves launched. That ratio exceeded 1 in
    // sparse canopies. The forward estimate weights each ray by the power of the primitive that launched it, and cannot
    // exceed the largest re-scattered fraction in the scene.
    std::vector<float> scattering_tail; // [primitive][band], allocated when the first tail is added
    std::vector<char> band_has_scattering_tail(Nbands_launch, 0);
    std::vector<double> launched_power_previous(Nbands_launch, 0.0); // power each band launched in the last scattering iteration
    std::vector<double> forward_q(Nbands_launch, -1.0); // forward estimate of q from each band's last scattering launch; negative until one has run
    std::vector<double> forward_q_previous(Nbands_launch, -1.0); // forward estimate of q from each band's second-to-last scattering launch; negative until two have run
    std::vector<double> forward_q_third_last(Nbands_launch, -1.0); // forward estimate of q from each band's third-to-last scattering launch; negative until three have run
    std::vector<float> absorbed_before_last_launch; // radiation_in just before the last scattering launch
    std::vector<float> absorbed_before_previous_launch; // radiation_in just before the second-to-last scattering launch
    uint launched_scattering_iterations = 0;
    if (geometry_data.primitive_areas.size() != Nprimitives) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The geometry holds " + std::to_string(geometry_data.primitive_areas.size()) + " primitive areas for " + std::to_string(Nprimitives) +
                             " primitives. Call updateGeometry() after adding or removing primitives.");
    }
    // Power each band would launch from the outgoing buffers, and the largest outgoing flux of any primitive face, both in
    // absolute value
    auto measureOutgoingEnergy = [&](const std::vector<float> &outgoing_top, const std::vector<float> &outgoing_bottom, std::vector<double> &power, std::vector<float> &max_flux) {
        power.assign(Nbands_launch, 0.0);
        max_flux.assign(Nbands_launch, 0.f);
        for (size_t p = 0; p < Nprimitives; p++) {
            const double area = geometry_data.primitive_areas[p];
            for (size_t b = 0; b < Nbands_launch; b++) {
                for (size_t component = 0; component < outgoing_components; component++) {
                    const size_t ind = component * layer_offset + p * Nbands_launch + b;
                    // Outgoing fluxes are negative where a surface radiates less than the reference field (see reference_flux)
                    const float top = fabsf(outgoing_top[ind]);
                    const float bottom = fabsf(outgoing_bottom[ind]);
                    power[b] += area * (double(top) + double(bottom));
                    max_flux[b] = std::max(max_flux[b], std::max(top, bottom));
                }
            }
        }
    };
    // Camera scatter accumulation vectors (declare early for use throughout ray tracing).
    // Laid out [camera][primitive][band]: the direct, diffuse and scattering launches run once per dispatch, not
    // once per camera, so they accumulate the scattered energy weighted by every camera's spectral response at
    // the same time, and each camera is later handed its own slice as the radiance it integrates. A single
    // [primitive][band] accumulator weighted by one camera's response gave every other camera that camera's image.
    const size_t camera_scatter_stride = Nprimitives * Nbands_launch;
    std::vector<float> scatter_top_cam;
    std::vector<float> scatter_bottom_cam;
    if (Ncameras > 0) {
        scatter_top_cam.resize(outgoing_components * Ncameras * camera_scatter_stride, 0.0f);
        scatter_bottom_cam.resize(outgoing_components * Ncameras * camera_scatter_stride, 0.0f);
    }
    // The radiation reaching primitives with a spectral texture, per [face][source slot][primitive][band] (see RayTracingResults::spectral_incident)
    std::vector<float> spectral_incident;
    // The radiation each face emits, per [face][primitive][band]: a camera sees it from every class of a primitive with a spectral texture
    std::vector<float> spectral_emitted;
    if (Ncameras > 0 && material_data.num_palette_entries > 0) {
        spectral_incident.resize(2 * (Nsources + 1) * camera_scatter_stride, 0.0f);
        spectral_emitted.resize(2 * camera_scatter_stride, 0.0f);
    }
    // What the cameras would see of the radiation left in flight after each band's last scattering iteration, summed as for the
    // absorbed flux (see addScatteringTail below), laid out as scatter_top_cam / scatter_bottom_cam and spectral_incident; and those
    // accumulators just before the last two scattering launches, from which what those launches added is measured
    std::vector<float> camera_tail_top, camera_tail_bottom, spectral_incident_tail;
    std::vector<float> camera_before_last_top, camera_before_last_bottom, camera_before_previous_top, camera_before_previous_bottom;
    std::vector<float> spectral_incident_before_last, spectral_incident_before_previous;
    helios::RayTracingResults last_scattering_results; // what the backend returned after the last scattering launch, when there are cameras

    // The ratio q need not be the same from one iteration to the next. Between a sheet of leaves and the ground below it alternates:
    // everything the ground scatters up reaches the sheet, but much of what the sheet scatters escapes upward. A series in the last
    // launch's q alone then over- or underestimates the tail depending on whether the depth is even or odd (by +400% at a depth of 2).
    // Such a series is instead summed over pairs of iterations: what the last two launches absorbed together, times q2/(1-q2), where
    // q2 is the product of their forward estimates. This is exact when the ratio alternates between two values, and equals the
    // single-launch series when it is constant (since A_k = q A_{k-1}, (A_{k-1} + A_k) q^2/(1-q^2) = A_k q/(1-q)). In a canopy the
    // ratio does not alternate but rises over the first iterations, as the radiation left is increasingly radiation that has been
    // scattered many times; the last launch's q is then the better estimate, and the product of the last two underestimates the tail.
    // The two cases are told apart by whether the change in q between the last three launches reverses. Where the test is confused by
    // noise, the estimates are nearly equal and so are the two sums. With only two launches there is nothing to test, and the series
    // is summed over pairs, which is far less wrong in a canopy (-2%) than the single-launch series is in an alternating scene.
    // The factor q/(1-q) (or q2/(1-q2)) by which a band's tail scales what its last launch (or last two launches) added, and whether the
    // series is summed over the last two launches
    auto scatteringTailFactor = [&](size_t launch_slot, uint launched_iterations, bool &sum_over_pairs) {
        if (forward_q.at(launch_slot) < 0.0) {
            helios_runtime_error("ERROR (RadiationModel::runBand): No forward scattering estimate was measured for band '" + band_labels.at(launch_slot) + "' before its scattering tail was summed. This is a bug in the radiation model.");
        }
        const double q_last = forward_q.at(launch_slot);
        const double q_previous = forward_q_previous.at(launch_slot);
        const double q_third_last = forward_q_third_last.at(launch_slot);
        const bool ratio_alternates = q_third_last < 0.0 || (q_last - q_previous) * (q_previous - q_third_last) < 0.0;
        sum_over_pairs = launched_iterations >= 2 && q_previous >= 0.0 && !absorbed_before_previous_launch.empty() && ratio_alternates;
        const double q = sum_over_pairs ? q_previous * q_last : q_last;
        if (q >= 1.0) {
            helios_runtime_error("ERROR (RadiationModel::runBand): The surfaces reached by band '" + band_labels.at(launch_slot) + "' after scattering iteration " + std::to_string(launched_iterations) + " scatter again " + std::to_string(q) +
                                 " of the energy that reaches them, so the iterations that would follow do not decrease and cannot be summed. Some surfaces in the scene have a reflectivity plus transmissivity of 1 or "
                                 "more; reduce it below 1.");
        }
        return float(q / (1.0 - q));
    };

    // Adds the tail of a band to the absorbed flux
    auto addScatteringTail = [&](size_t launch_slot, const std::vector<float> &absorbed_after_last_launch, double power_left, uint launched_iterations) {
        band_has_scattering_tail.at(launch_slot) = 1;
        if (power_left <= 0.0) {
            return; // nothing left in flight
        }
        bool sum_over_pairs = false;
        const float tail_factor = scatteringTailFactor(launch_slot, launched_iterations, sum_over_pairs);
        const std::vector<float> &absorbed_before_summed_launches = sum_over_pairs ? absorbed_before_previous_launch : absorbed_before_last_launch;
        if (scattering_tail.empty()) {
            scattering_tail.assign(Nbands_launch * Nprimitives, 0.f);
        }
        for (size_t p = 0; p < Nprimitives; p++) {
            const size_t ind = p * Nbands_launch + launch_slot;
            scattering_tail[ind] += tail_factor * (absorbed_after_last_launch[ind] - absorbed_before_summed_launches[ind]);
        }
    };

    // Adds the tail of a band to what the cameras see. Cameras see the radiation the primitives scatter, which falls off from one
    // iteration to the next as what they absorb does, so the same series gives what they would see of the iterations that were not
    // traced. It must be added before the cameras are rendered, with the accumulators as they were after the band's last launch.
    auto addCameraScatteringTail = [&](size_t launch_slot, double power_left, uint launched_iterations) {
        if (Ncameras == 0 || power_left <= 0.0) {
            return;
        }
        bool sum_over_pairs = false;
        const float tail_factor = scatteringTailFactor(launch_slot, launched_iterations, sum_over_pairs);
        const std::vector<float> &camera_before_top = sum_over_pairs ? camera_before_previous_top : camera_before_last_top;
        const std::vector<float> &camera_before_bottom = sum_over_pairs ? camera_before_previous_bottom : camera_before_last_bottom;
        if (camera_tail_top.empty()) {
            camera_tail_top.assign(scatter_top_cam.size(), 0.f);
            camera_tail_bottom.assign(scatter_bottom_cam.size(), 0.f);
        }
        for (size_t slice = 0; slice < outgoing_components * Ncameras; slice++) {
            for (size_t p = 0; p < Nprimitives; p++) {
                const size_t ind = slice * camera_scatter_stride + p * Nbands_launch + launch_slot;
                camera_tail_top[ind] += tail_factor * (scatter_top_cam[ind] - camera_before_top[ind]);
                camera_tail_bottom[ind] += tail_factor * (scatter_bottom_cam[ind] - camera_before_bottom[ind]);
            }
        }
        if (!spectral_incident.empty()) {
            const std::vector<float> &spectral_before = sum_over_pairs ? spectral_incident_before_previous : spectral_incident_before_last;
            if (spectral_incident_tail.empty()) {
                spectral_incident_tail.assign(spectral_incident.size(), 0.f);
            }
            for (size_t slice = 0; slice < 2 * (Nsources + 1); slice++) {
                for (size_t p = 0; p < Nprimitives; p++) {
                    const size_t ind = slice * camera_scatter_stride + p * Nbands_launch + launch_slot;
                    spectral_incident_tail[ind] += tail_factor * (spectral_incident[ind] - spectral_before[ind]);
                }
            }
        }
    };

    if (scatteringenabled && rundirect) {
        // Get scattered energy from direct rays for primary diffuse/emission
        helios::RayTracingResults scatter_results;
        backend->getRadiationResults(scatter_results);
        flux_top = scatter_results.scatter_buff_top;
        flux_bottom = scatter_results.scatter_buff_bottom;

        // Accumulate camera scatter from direct rays
        if (Ncameras > 0) {
            accumulateCameraScatter(scatter_results, scatter_top_cam, scatter_bottom_cam, spectral_incident);
            // Zero GPU camera scatter buffers to prevent double-counting on next iteration
            backend->zeroCameraScatterBuffers(Nbands_launch);
        }

        // Bands with no scattering keep their scattered direct energy rather than launching it
        for (size_t b = 0; b < Nbands_launch; b++) {
            if (scattering_depth.at(b) == 0) {
                retireBand(b, flux_top, flux_bottom);
            }
        }

        // The second outgoing component holds what surface layers reflected of the direct radiation, which the primary
        // diffuse/emission launch delivers through their reflection lobe instead. Without that launch it stays, so that the first
        // scattering launch, which then delivers the lobe, counts it in the energy it scatters.
        if (outgoing_components == 2 && (emissionenabled || diffuseenabled)) {
            std::fill(flux_top.begin() + std::ptrdiff_t(layer_offset), flux_top.begin() + std::ptrdiff_t(2 * layer_offset), 0.f);
            std::fill(flux_bottom.begin() + std::ptrdiff_t(layer_offset), flux_bottom.begin() + std::ptrdiff_t(2 * layer_offset), 0.f);
        }

        // Each face launches what it scattered: a one-sided primitive's reflected energy leaves its top face and its transmitted
        // energy its bottom face, as in every later scattering iteration. Copying the sum into both faces, as was done here,
        // made the first launch emit up to twice the energy one-sided primitives scattered.

        // Upload scattered energy to backend's radiation_out buffers for scattering iterations
        backend->uploadRadiationOut(flux_top, flux_bottom);
        backend->zeroScatterBuffers();
    }

    // The reference blackbody field (see reference_flux above) needs a band that scatters: without scattering every surface is
    // black and absorbs all it intercepts, so an isothermal scene is already in exact equilibrium. A translucent cover has an angular emissivity that a uniform field does not satisfy face by face, and a SIF band's
    // emission is not thermal.
    for (size_t b = 0; b < Nbands_launch; b++) {
        const RadiationBand &band = radiation_bands.at(band_labels.at(b));
        if (!band.emissionReferenceFieldFlag) {
            continue;
        }
        std::string problem;
        if (!band.emissionFlag) {
            problem = "emission is disabled for the band";
        } else if (scattering_depth.at(b) == 0) {
            problem = "its scattering depth is 0 (set it to at least 1 with setScatteringDepth(); without scattering every surface is black and an isothermal scene is already in exact equilibrium)";
        } else if (sif_emission_bands.count(band_labels.at(b)) > 0) {
            problem = "it is a SIF emission band, whose emission is not thermal";
        } else if (band.diffuseRayCount == 0) {
            problem = "its diffuse ray count is 0, so no emission would be traced to balance the absorption of the reference field that is added to every surface (set it with setDiffuseRayCount())";
        } else {
            MaterialPropertyIndexer cover_mat_idx(Nsources + 1, Nprimitives, Nbands_global);
            const size_t b_global = std::distance(radiation_bands.begin(), radiation_bands.find(band_labels.at(b)));
            for (size_t u = 0; u < Nprimitives; u++) {
                if (material_data.is_glass.at(cover_mat_idx(Nsources, u, b_global)) != 0) {
                    problem = "it contains translucent cover primitives (glass_n_" + band_labels.at(b) + " primitive data, e.g. UUID " + std::to_string(context_UUIDs.at(u)) + "), whose angular emissivity a uniform field does not satisfy";
                    break;
                }
            }
        }
        if (!problem.empty()) {
            helios_runtime_error("ERROR (RadiationModel::runBand): The emission reference field is enabled for band '" + band_labels.at(b) + "', but " + problem + ". Call disableEmissionReferenceField(\"" + band_labels.at(b) + "\") for this band.");
        }
    }

    // --- Diffuse/Emission launch ---- //

    if (emissionenabled || diffuseenabled) {

        // add any emitted energy to the outgoing energy buffer
        if (emissionenabled) {
            // Update primitive outgoing emission
            float eps, temperature;

            // Create indexer for emission flux buffers
            RadiationBufferIndexer emission_indexer(Nprimitives, Nbands_launch);
            MaterialPropertyIndexer reference_mat_idx(Nsources + 1, Nprimitives, Nbands_global);
            CameraMaterialIndexer reference_cam_idx(Nsources + 1, Nprimitives, Nbands_global, Ncameras);

            for (auto b = 0; b < Nbands_launch; b++) {
                //\todo For emissivity and twosided_flag, this should be done in updateRadiativeProperties() to avoid having to do it on every runBand() call
                if (radiation_bands.at(band_labels.at(b)).emissionFlag) {
                    std::string prop = "emissivity_" + band_labels.at(b);
                    // SIF bands: per-primitive source flux comes from computeSIFEmission(), with
                    // separate top/bottom buffers (each face emits Fluspect-B's backward spectrum of its own excitation plus the forward spectrum of the other face's).
                    auto sif_band_it = sif_emission_buffer.find(band_labels.at(b));
                    auto sif_band_bot_it = sif_emission_buffer_bottom.find(band_labels.at(b));
                    const bool have_sif_band = (sif_band_it != sif_emission_buffer.end());
                    const bool have_sif_band_bot = (sif_band_bot_it != sif_emission_buffer_bottom.end());
                    // A band with wavelength bounds emits only the in-band portion of the Planck spectrum; a band without bounds (both zero) emits the full sigma*T^4. A band may start at 0 nm.
                    const helios::vec2 emission_wavebounds = radiation_bands.at(band_labels.at(b)).wavebandBounds;
                    const bool emission_band_has_bounds = radiation_bands.at(band_labels.at(b)).hasWavebandBounds();
                    const size_t b_global = std::distance(radiation_bands.begin(), radiation_bands.find(band_labels.at(b)));

                    // Blackbody (in-band) emissive power of each primitive at its temperature
                    std::vector<float> blackbody_power;
                    if (!have_sif_band) {
                        blackbody_power.resize(Nprimitives);
                        for (size_t u = 0; u < Nprimitives; u++) {
                            uint p = context_UUIDs.at(u);
                            if (context->doesPrimitiveDataExist(p, "temperature")) {
                                context->getPrimitiveData(p, "temperature", temperature);
                                if (temperature < 0) {
                                    temperature = temperature_default;
                                }
                            } else {
                                temperature = temperature_default;
                            }
                            blackbody_power.at(u) = sigma * pow(temperature, 4);
                            if (emission_band_has_bounds && temperature > 0.f) {
                                blackbody_power.at(u) *= blackbodyBandFraction(emission_wavebounds.x, emission_wavebounds.y, temperature);
                            }
                        }
                    }

                    // Reference blackbody field (see reference_flux above), if enabled for the band
                    if (radiation_bands.at(band_labels.at(b)).emissionReferenceFieldFlag && !have_sif_band) {
                        double weighted_power = 0.0;
                        double visible_area = 0.0;
                        for (size_t u = 0; u < Nprimitives; u++) {
                            if (geometry_data.twosided_flags.at(u) != 3) {
                                weighted_power += geometry_data.primitive_areas.at(u) * double(blackbody_power.at(u));
                                visible_area += geometry_data.primitive_areas.at(u);
                            }
                        }
                        if (visible_area > 0.0) {
                            reference_flux.at(b) = float(weighted_power / visible_area);
                        }
                    }
                    const float reference_power = reference_flux.at(b);

                    for (size_t u = 0; u < Nprimitives; u++) {
                        // Use BufferIndexer: [primitive][band]
                        size_t ind = emission_indexer(u, b);
                        uint p = context_UUIDs.at(u);
                        float out_top;
                        float sif_bottom_flux = 0.f;  // bottom-face emission (if SIF)
                        bool used_sif = false;
                        if (have_sif_band) {
                            auto sif_uuid_it = sif_band_it->second.find(p);
                            if (sif_uuid_it != sif_band_it->second.end()) {
                                out_top = sif_uuid_it->second;
                                used_sif = true;
                            }
                        }
                        if (used_sif && have_sif_band_bot) {
                            auto sif_bot_it = sif_band_bot_it->second.find(p);
                            if (sif_bot_it != sif_band_bot_it->second.end()) {
                                sif_bottom_flux = sif_bot_it->second;
                            }
                        }
                        if (!used_sif) {
                            // SIF bands are visible/red wavelengths (~680-760 nm). Stefan–Boltzmann
                            // (broadband σT⁴) is only physically meaningful for thermal IR. Primitives
                            // without an SIF source flux must not emit into SIF bands at room temperature.
                            if (have_sif_band) {
                                out_top = 0.f;
                            } else {
                                if (context->doesPrimitiveDataExist(p, prop.c_str())) {
                                    context->getPrimitiveData(p, prop.c_str(), eps);
                                } else {
                                    eps = eps_default;
                                }
                                if (scattering_depth.at(b) == 0 && eps != 1.f) {
                                    eps = 1.f;
                                }
                                out_top = eps * blackbody_power.at(u);
                            }
                        }

                        // Check twosided_flag - check material first, then primitive data. A two-sided primitive emits from its
                        // bottom face too: only its own emitted energy, NOT the top face's accumulated scattered-direct energy that
                        // flux_top already holds, which would be re-emitted from the bottom face and violate energy conservation.
                        // A two-sided SIF primitive emits its own bottom-face flux (physically distinct emission lobes).
                        float out_bottom = 0.f;
                        if (context->getPrimitiveTwosidedFlag(p, 1) != 0) {
                            out_bottom = used_sif ? sif_bottom_flux : out_top;
                        }

                        // A face with a specular surface layer emits eps_b (1 - a(mu)) times the blackbody radiance, with
                        // eps_b = eps / (1 - a(1)) for the emissivity eps at normal incidence: its emitted flux is
                        // eps (1 - abar) / (1 - a(1)) of the blackbody's, all in the first outgoing component. A SIF flux is already
                        // the flux leaving the face.
                        const float *layer_table = surfaceLayerTable(u, b_global);
                        const float normal_albedo = layer_table != nullptr ? layer_table[material_data.surface_layer_table_size - 1] : 0.f;
                        const float mean_albedo = layer_table != nullptr ? layer_table[material_data.surface_layer_table_size] : 0.f;
                        if (layer_table != nullptr && !used_sif) {
                            out_top *= (1.f - mean_albedo) / (1.f - normal_albedo);
                            out_bottom *= (1.f - mean_albedo) / (1.f - normal_albedo);
                        }

                        if (!spectral_emitted.empty()) {
                            spectral_emitted.at(ind) += out_top;
                            spectral_emitted.at(camera_scatter_stride + ind) += out_bottom;
                        }

                        if (reference_power == 0.f) {
                            flux_top.at(ind) += out_top;
                            flux_bottom.at(ind) += out_bottom;
                            // Emission is not weighted by a spectral response, so every camera receives the same value.
                            for (size_t cam = 0; cam < Ncameras; cam++) {
                                scatter_top_cam[cam * camera_scatter_stride + ind] += out_top;
                                scatter_bottom_cam[cam * camera_scatter_stride + ind] += out_bottom;
                            }
                        } else {
                            // Under the reference field every face leaves the reference flux: (1 - abar) of it in the first
                            // component and abar in the second (abar = 0 without a surface layer). A face's share of it that is not
                            // reflected or transmitted from the faces that gather (a one-sided primitive's bottom face gathers
                            // nothing) is the emission consistent with that field; only the departure from it is traced. rho and
                            // tau are those of the body for a primitive with a surface layer.
                            const char twosided_flag = geometry_data.twosided_flags.at(u);
                            const float gathers_top = twosided_flag != 3 ? 1.f : 0.f;
                            const float gathers_bottom = twosided_flag == 1 ? 1.f : 0.f;
                            const float rho = material_data.reflectivity.at(reference_mat_idx(Nsources, u, b_global));
                            const float tau = material_data.transmissivity.at(reference_mat_idx(Nsources, u, b_global));
                            const float body_reference = reference_power * (1.f - mean_albedo);
                            const float layer_reference = reference_power * mean_albedo;
                            flux_top.at(ind) += out_top - body_reference * (1.f - rho * gathers_top - tau * gathers_bottom);
                            flux_bottom.at(ind) += out_bottom - body_reference * (1.f - rho * gathers_bottom - tau * gathers_top);
                            if (outgoing_components == 2) {
                                flux_top.at(layer_offset + ind) -= layer_reference * (1.f - gathers_top);
                                flux_bottom.at(layer_offset + ind) -= layer_reference * (1.f - gathers_bottom);
                            }
                            reference_absorption.at(ind) = body_reference * (1.f - rho - tau) * (gathers_top + gathers_bottom);
                            // Cameras read their own outgoing buffers, which hold the true outgoing flux: the emission, plus the
                            // reference field each face reflects and transmits (the traced departure from it is added as gathered).
                            // The surface layer reflects all wavelengths alike, so a camera weights it by its white reference.
                            for (size_t cam = 0; cam < Ncameras; cam++) {
                                const float rho_cam = material_data.reflectivity_cam.at(reference_cam_idx(Nsources, u, b_global, cam));
                                const float tau_cam = material_data.transmissivity_cam.at(reference_cam_idx(Nsources, u, b_global, cam));
                                scatter_top_cam[cam * camera_scatter_stride + ind] += out_top + body_reference * (rho_cam * gathers_top + tau_cam * gathers_bottom);
                                scatter_bottom_cam[cam * camera_scatter_stride + ind] += out_bottom + body_reference * (rho_cam * gathers_bottom + tau_cam * gathers_top);
                                if (outgoing_components == 2) {
                                    const float white = material_data.white_reference_cam.at(((size_t) Nsources * Nbands_global + b_global) * Ncameras + cam);
                                    scatter_top_cam[(Ncameras + cam) * camera_scatter_stride + ind] += layer_reference * white * gathers_top;
                                    scatter_bottom_cam[(Ncameras + cam) * camera_scatter_stride + ind] += layer_reference * white * gathers_bottom;
                                }
                            }
                        }
                    }
                }
            }
        }

        // The device camera-scatter buffers are write-only accumulators: every device-side use is an
        // atomicFloatAdd and nothing reads them back during a launch. They must therefore be ZERO
        // going into each launch, so that the download afterwards yields only that launch's own
        // contribution, which is what the three accumulate sites below add into scatter_top_cam.
        //
        // Uploading the host accumulator here instead seeded the device buffer with the emitted flux
        // that scatter_top_cam already held, so the next download returned base + new and the base was
        // counted twice. On the first runBand() of an emission-only scene that upload was silently
        // dropped -- with no sources, rundirect is false, so the earlier zeroCameraScatterBuffers()
        // call is skipped and the OptiX8 device pointers are still null, which makes its
        // uploadCameraScatterBuffers() (since removed) a no-op -- so only the second and later calls
        // doubled. Zeroing
        // here allocates the buffers as well, which removes that first-call/later-call asymmetry.
        //
        // What the cameras actually read is radiation_out, uploaded from the same host accumulator
        // further below; nothing depends on the device camera-scatter buffer holding the base.
        if (Ncameras > 0) {
            backend->zeroCameraScatterBuffers(Nbands_launch);
        }

        // Note: radiation_specular_RTbuffer is populated on GPU via atomicFloatAdd during ray tracing, don't overwrite it here

        // Compute diffuse launch dimension
        size_t n = ceil(sqrt(double(diffuseRayCount)));
        uint rays_per_primitive = n * n;

        if (message_flag) {
            std::cout << "Performing primary diffuse radiation ray trace for bands ";
            for (const auto &band: label) {
                std::cout << band << " ";
            }
            std::cout << "..." << std::flush;
        }

        // Build launch parameters for diffuse rays
        // Note: OptiX 6.5-specific batching (maxRays limit) should be handled inside the OptiX backend,
        // not here in RadiationModel. Vulkan and other backends can launch all primitives at once.
        helios::RayTracingLaunchParams params;
        params.launch_offset = 0;
        params.launch_count = Nprimitives; // Launch all primitives at once (backend handles batching if needed)
        params.rays_per_primitive = rays_per_primitive;
        params.random_seed = drawLaunchSeed();
        params.current_band = 0;
        params.num_bands_global = Nbands_global;
        params.num_bands_launch = Nbands_launch;
        std::vector<bool> band_flags(band_launch_flag.begin(), band_launch_flag.end());
        params.band_launch_flag = band_flags;
        params.scattering_iteration = 0;
        params.radiation_out_top = flux_top;
        params.radiation_out_bottom = flux_bottom;
        // This launch reads the first scattering of the direct radiation, so it also evaluates the surface layers' reflection lobe of it
        params.specular_reflection_enabled = (rundirect && scatteringenabled) ? specular_reflection_mode : 0;

        // Pass diffuse radiation parameters to backend: the sky flux of each band, followed by its reference flux
        params.diffuse_flux = diffuse_flux;
        params.diffuse_flux.insert(params.diffuse_flux.end(), reference_flux.begin(), reference_flux.end());
        params.below_horizon_diffuse_enabled = below_horizon_diffuse_enabled ? 1 : 0;
        params.diffuse_extinction = diffuse_extinction;
        params.diffuse_dist_norm = diffuse_dist_norm;
        // Convert helios::vec3 to helios::vec3
        std::vector<helios::vec3> peak_dirs(diffuse_peak_dir.size());
        for (size_t i = 0; i < diffuse_peak_dir.size(); i++) {
            peak_dirs[i] = helios::make_vec3(diffuse_peak_dir[i].x, diffuse_peak_dir[i].y, diffuse_peak_dir[i].z);
        }
        params.diffuse_peak_dir = peak_dirs;
        params.sky_radiance_params = prague_params;

        // Top surface launch
        params.launch_face = 1;
        backend->launchDiffuseRays(params);

        // Bottom surface launch
        params.launch_face = 0;
        backend->launchDiffuseRays(params);
        launched_rays += 2 * size_t(Nprimitives) * rays_per_primitive;

        // Retrieve and accumulate camera scatter from primary diffuse
        if (Ncameras > 0) {
            helios::RayTracingResults primary_results;
            backend->getRadiationResults(primary_results);
            accumulateCameraScatter(primary_results, scatter_top_cam, scatter_bottom_cam, spectral_incident);
            // Zero GPU camera scatter buffers to prevent double-counting on next iteration
            backend->zeroCameraScatterBuffers(Nbands_launch);
        }

        if (message_flag) {
            std::cout << "done." << std::endl;
        }
    }

    // After primary diffuse, prepare scatter_buff for scattering iterations
    // When direct rays ran, scatter_buff was already copied to radiation_out at line 3710
    // For emission/diffuse without direct rays, we need to do this now
    if (scatteringenabled && (emissionenabled || diffuseenabled) && !rundirect) {
        backend->copyScatterToRadiation();
        backend->zeroScatterBuffers();
    }

    if (scatteringenabled && (emissionenabled || diffuseenabled || rundirect)) {

        for (auto b = 0; b < Nbands_launch; b++) {
            diffuse_flux.at(b) = 0.f;
        }
        // NOTE: diffuse_flux zeroed for scattering, passed to backend via launch params

        size_t n = ceil(sqrt(double(diffuseRayCount)));
        const uint rays_per_primitive = n * n;
        // Fewest diffuse rays per primitive a later iteration is reduced to (see enableScatteringRayReduction())
        const uint min_scattering_rays = std::min(rays_per_primitive, 16u);
        std::vector<double> launched_power(Nbands_launch, 0.0);
        std::vector<double> launched_power_first(Nbands_launch, 0.0);
        std::vector<float> max_outgoing_flux(Nbands_launch, 0.f);

        for (uint s = 0; s < scatteringDepth; s++) {

            // Copy scatter buffers to radiation_out when needed
            // For s=0 with direct plus emission and/or diffuse: the primary diffuse/emission launch has already propagated the scattered direct
            // energy (and any emission) held in radiation_out, and the scatter buffers now hold what that launch scattered - including the
            // scattered sky radiation. That is the energy to launch next; re-launching radiation_out would count it twice and drop the rest.
            // For s=0 with direct only: no primary launch ran, and radiation_out still holds the scattered direct energy to be launched.
            // For s>0: scatter from previous iteration needs to be copied for next iteration
            if (s > 0 || ((emissionenabled || diffuseenabled) && rundirect)) {
                backend->copyScatterToRadiation();
            }
            backend->zeroScatterBuffers();

            // Extract radiation_out to ensure it's uploaded for scattering rays
            helios::RayTracingResults scatter_results;
            backend->getRadiationResults(scatter_results);
            std::vector<float> flux_top_scatter = std::move(scatter_results.radiation_out_top);
            std::vector<float> flux_bottom_scatter = std::move(scatter_results.radiation_out_bottom);
            measureOutgoingEnergy(flux_top_scatter, flux_bottom_scatter, launched_power, max_outgoing_flux);
            if (s == 0) {
                launched_power_first = launched_power;
            }

            // With direct radiation only, the first scattering launch also delivers the surface layers' reflection lobe of the
            // direct radiation, whose energy is in the second outgoing component (see above). A band that stops before that launch
            // keeps it there until it is known whether the launch will happen.
            const bool layer_lobe_pending = s == 0 && rundirect && !(emissionenabled || diffuseenabled) && specular_reflection_mode != 0;
            std::vector<size_t> retired_before_lobe;

            // Retire the bands whose scattering is done: their depth is exhausted, their largest outgoing flux is below their
            // threshold, or they have nothing left to scatter.
            for (size_t b = 0; b < Nbands_launch; b++) {
                const std::string &bname = band_labels.at(b);
                // Internal SIF excitation bands ("_SIF_exc_*") get piggy-backed onto user dispatches (e.g. PAR) that may have a higher
                // scatteringDepth. Suppress the per-band log for those - with ~35 excitation bins they would flood stdout.
                const bool is_sif_excitation_band = bname.size() >= 9 && bname.compare(0, 9, "_SIF_exc_") == 0;
                const bool report = message_flag && !is_sif_excitation_band && band_retired.at(b) == 0;
                if (scattering_depth.at(b) == 0) {
                    // Retired on every remaining iteration, not just the first: a band retired before the primary diffuse/emission
                    // launch still receives the energy that launch scattered, which arrives in the outgoing buffers at s=0.
                    retireBand(b, flux_top_scatter, flux_bottom_scatter);
                    if (report) {
                        std::cout << "Skipping band " << bname << " for scattering launch " << s + 1 << std::endl;
                    }
                    continue;
                }
                if (band_retired.at(b) != 0) {
                    continue; // its outgoing energy was zeroed when it was retired, so none has arrived since
                }
                const float min_scatter_energy = radiation_bands.at(bname).minScatterEnergy;
                const bool depth_done = s >= scattering_depth.at(b);
                const bool converged = launched_power.at(b) <= 0.0 || (min_scatter_energy > 0.f && max_outgoing_flux.at(b) < min_scatter_energy);
                if (!depth_done && !converged) {
                    continue;
                }
                if (s == 0) {
                    // Below its threshold before any scattering launch, so there is no earlier iteration to extrapolate from: the
                    // little energy left is absorbed where it is.
                    retireBand(b, flux_top_scatter, flux_bottom_scatter, layer_lobe_pending ? 1 : SIZE_MAX);
                    if (layer_lobe_pending) {
                        retired_before_lobe.push_back(b);
                    }
                } else {
                    addScatteringTail(b, scatter_results.radiation_in, launched_power.at(b), s);
                    addCameraScatteringTail(b, launched_power.at(b), s);
                    for (size_t component = 0; component < outgoing_components; component++) {
                        for (size_t p = 0; p < Nprimitives; p++) {
                            flux_top_scatter[component * layer_offset + p * Nbands_launch + b] = 0.f;
                            flux_bottom_scatter[component * layer_offset + p * Nbands_launch + b] = 0.f;
                        }
                    }
                    band_retired.at(b) = 1;
                }
                if (report) {
                    const char *reason = depth_done ? "" : (launched_power.at(b) <= 0.0 ? " (nothing left to scatter)" : " (below the minimum scattering energy)");
                    std::cout << "Scattering for band " << bname << " finished after " << s << " iteration(s)" << reason << std::endl;
                }
            }
            if (std::all_of(band_retired.begin(), band_retired.end(), [](char retired) { return retired != 0; })) {
                // Every band is done, so further launches would carry no energy, and no launch delivers the surface layers' lobe: what
                // they reflected of the direct radiation is absorbed in place, like the rest of the retired energy
                for (size_t b: retired_before_lobe) {
                    for (size_t p = 0; p < Nprimitives; p++) {
                        const size_t ind = layer_offset + p * Nbands_launch + b;
                        retired_top.at(ind) += flux_top_scatter.at(ind);
                        retired_bottom.at(ind) += flux_bottom_scatter.at(ind);
                    }
                }
                break;
            }
            if (layer_lobe_pending) {
                // This launch delivers the surface layers' reflection of the direct radiation through their lobe, so it must not read
                // it from the outgoing buffers too
                std::fill(flux_top_scatter.begin() + std::ptrdiff_t(layer_offset), flux_top_scatter.begin() + std::ptrdiff_t(2 * layer_offset), 0.f);
                std::fill(flux_bottom_scatter.begin() + std::ptrdiff_t(layer_offset), flux_bottom_scatter.begin() + std::ptrdiff_t(2 * layer_offset), 0.f);
            }

            // Later iterations carry less energy and so contribute less noise: give them rays in proportion to that energy. A
            // band's last two launches are weighted by 1/(1-q2), since the summed tail scales what they absorb by q2/(1-q2) on top.
            uint rays_this_iteration = rays_per_primitive;
            if (scattering_ray_reduction && s > 0) {
                double energy_fraction = 0.0;
                for (size_t b = 0; b < Nbands_launch; b++) {
                    if (band_retired.at(b) == 0 && launched_power_first.at(b) > 0.0) {
                        double weight = 1.0;
                        if ((s + 1 == scattering_depth.at(b) || s + 2 == scattering_depth.at(b)) && forward_q.at(b) >= 0.0) {
                            // The forward estimates of the band's previous launches stand for the q2 of its last two
                            const double q = forward_q_previous.at(b) >= 0.0 ? forward_q_previous.at(b) * forward_q.at(b) : forward_q.at(b) * forward_q.at(b);
                            if (q >= 1.0) {
                                energy_fraction = 1.0; // not decaying: trace the full ray count
                                continue;
                            }
                            weight = 1.0 / (1.0 - q);
                        }
                        energy_fraction = std::max(energy_fraction, weight * launched_power.at(b) / launched_power_first.at(b));
                    }
                }
                const size_t reduced_rays = std::max(size_t(min_scattering_rays), size_t(std::ceil(rays_per_primitive * std::min(1.0, energy_fraction))));
                const size_t n_reduced = std::ceil(std::sqrt(double(reduced_rays)));
                rays_this_iteration = n_reduced * n_reduced;
            }

            launched_power_previous = launched_power;
            absorbed_before_previous_launch = std::move(absorbed_before_last_launch);
            absorbed_before_last_launch = std::move(scatter_results.radiation_in);
            if (Ncameras > 0) {
                camera_before_previous_top = std::move(camera_before_last_top);
                camera_before_previous_bottom = std::move(camera_before_last_bottom);
                camera_before_last_top = scatter_top_cam;
                camera_before_last_bottom = scatter_bottom_cam;
                spectral_incident_before_previous = std::move(spectral_incident_before_last);
                spectral_incident_before_last = spectral_incident;
            }
            launched_scattering_iterations = s + 1;

            if (message_flag) {
                std::cout << "Performing scattering ray trace (iteration " << s + 1 << " of " << scatteringDepth << ", " << rays_this_iteration << " rays per primitive)..." << std::flush;
            }

            // Build launch parameters for scattering diffuse rays
            // Launch all primitives at once (backend handles batching if needed)
            helios::RayTracingLaunchParams params;
            params.launch_offset = 0;
            params.launch_count = Nprimitives;
            params.rays_per_primitive = rays_this_iteration;
            params.random_seed = drawLaunchSeed();
            params.current_band = 0;
            params.num_bands_global = Nbands_global;
            params.num_bands_launch = Nbands_launch;
            // Same flags as the primary launches for the whole dispatch - see the note on band retirement above
            std::vector<bool> band_flags(band_launch_flag.begin(), band_launch_flag.end());
            params.band_launch_flag = band_flags;
            params.scattering_iteration = s;
            params.forward_scatter_enabled = true;
            // Without a primary diffuse/emission launch, the first scattering launch reads the first scattering of the direct
            // radiation, so it evaluates the surface layers' reflection lobe of it
            params.specular_reflection_enabled = (s == 0 && rundirect && !(emissionenabled || diffuseenabled)) ? specular_reflection_mode : 0;

            // Pass diffuse radiation parameters to backend: no sky and no reference field, which enter only the primary launch
            params.diffuse_flux.assign(2 * Nbands_launch, 0.f);
            params.below_horizon_diffuse_enabled = below_horizon_diffuse_enabled ? 1 : 0;
            params.diffuse_extinction = diffuse_extinction;
            params.diffuse_dist_norm = diffuse_dist_norm;
            // Convert helios::vec3 to helios::vec3
            std::vector<helios::vec3> peak_dirs(diffuse_peak_dir.size());
            for (size_t i = 0; i < diffuse_peak_dir.size(); i++) {
                peak_dirs[i] = helios::make_vec3(diffuse_peak_dir[i].x, diffuse_peak_dir[i].y, diffuse_peak_dir[i].z);
            }
            params.diffuse_peak_dir = peak_dirs;
            params.sky_radiance_params = prague_params;

            // Set radiation_out for scattering rays
            params.radiation_out_top = std::move(flux_top_scatter);
            params.radiation_out_bottom = std::move(flux_bottom_scatter);

            // Top surface launch
            params.launch_face = 1;
            backend->launchDiffuseRays(params);

            // Bottom surface launch
            params.launch_face = 0;
            backend->launchDiffuseRays(params);
            launched_rays += 2 * size_t(Nprimitives) * rays_this_iteration;

            // The forward estimate of q for each band: the power its next generation will carry over the power just launched
            std::vector<float> forward_scatter;
            backend->getForwardScatterResults(forward_scatter, Nbands_launch);
            for (size_t b = 0; b < Nbands_launch; b++) {
                double forward_power = 0.0;
                for (size_t p = 0; p < Nprimitives; p++) {
                    forward_power += double(geometry_data.primitive_areas[p]) * double(forward_scatter[p * Nbands_launch + b]);
                }
                forward_q_third_last.at(b) = forward_q_previous.at(b);
                forward_q_previous.at(b) = forward_q.at(b);
                forward_q.at(b) = launched_power.at(b) > 0.0 ? forward_power / launched_power.at(b) : 0.0;
            }

            // Accumulate camera scatter from this scattering iteration
            if (Ncameras > 0) {
                helios::RayTracingResults post_launch;
                backend->getRadiationResults(post_launch);
                accumulateCameraScatter(post_launch, scatter_top_cam, scatter_bottom_cam, spectral_incident);
                // Zero GPU camera scatter buffers to prevent double-counting on next iteration
                backend->zeroCameraScatterBuffers(Nbands_launch);
                // Kept for the cameras' scattering tail of the bands still scattering when the loop ends
                last_scattering_results = std::move(post_launch);
            }

            if (message_flag) {
                std::cout << "\r                                                                                                                           \r" << std::flush;
            }
        }

        if (message_flag) {
            std::cout << "Performing scattering ray trace...done." << std::endl;
        }
    }

    // **** CAMERA RAY TRACE **** //
    if (Ncameras > 0) {

        // Add what the cameras would see of the radiation left in flight after the last scattering iteration. Bands that finished
        // inside the scattering loop were summed there; those still scattering when it ended, which ran to the deepest scattering
        // depth, are summed here from what the last launch left, as their absorbed flux is below.
        if (!absorbed_before_last_launch.empty() && !last_scattering_results.scatter_buff_top.empty()) {
            std::vector<double> power_left;
            std::vector<float> max_flux_left;
            measureOutgoingEnergy(last_scattering_results.scatter_buff_top, last_scattering_results.scatter_buff_bottom, power_left, max_flux_left);
            for (size_t b = 0; b < Nbands_launch; b++) {
                if (scattering_depth.at(b) > 0 && band_retired.at(b) == 0) {
                    addCameraScatteringTail(b, power_left.at(b), launched_scattering_iterations);
                }
            }
        }
        for (size_t i = 0; i < camera_tail_top.size(); i++) {
            scatter_top_cam[i] += camera_tail_top[i];
            scatter_bottom_cam[i] += camera_tail_bottom[i];
        }
        for (size_t i = 0; i < spectral_incident_tail.size(); i++) {
            spectral_incident[i] += spectral_incident_tail[i];
        }

        // Setup solar disk rendering for cameras (enables lens flare effects)
        // Find sun-like sources (collimated or sun_sphere) and compute solar disk radiance
        vec3 sun_dir(0, 0, 1); // Default zenith
        std::vector<float> solar_radiances(Nbands_launch, 0.0f);
        bool has_sun_source = false;

        for (size_t s = 0; s < radiation_sources.size(); s++) {
            const RadiationSource &source = radiation_sources.at(s);
            if (source.source_type == RADIATION_SOURCE_TYPE_COLLIMATED || source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
                // Get sun direction from source position (normalized)
                sun_dir = source.source_position;
                sun_dir.normalize();
                has_sun_source = true;

                // Compute solar disk radiance for each band
                for (size_t b = 0; b < Nbands_launch; b++) {
                    float flux = getSourceFlux(s, band_labels.at(b));

                    if (source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
                        // For sun sphere: flux is surface exitance (σT⁴)
                        // Radiance as seen from Earth: L = F_surface / π
                        solar_radiances[b] = flux / M_PI;
                    } else {
                        // For collimated: flux is irradiance at Earth
                        // Solar solid angle: π × (4.63e-3)² ≈ 6.74×10⁻⁵ sr
                        const float solar_solid_angle = 6.74e-5f;
                        solar_radiances[b] = flux / solar_solid_angle;
                    }
                }
                break; // Use first sun-like source found
            }
        }

        // With no sun source emitting in these bands (runBand() adds a zero-flux source at the zenith to a model without sources), only the Prague data say where the
        // sun is, and the camera's circumsolar brightening must surround it as the diffuse launches' does
        int prague_valid = 0;
        if (context->doesGlobalDataExist("prague_sky_valid")) {
            context->getGlobalData("prague_sky_valid", prague_valid);
        }
        const bool sun_emits = std::any_of(solar_radiances.begin(), solar_radiances.end(), [](float radiance) { return radiance > 0.f; });
        if (prague_valid == 1 && !sun_emits) {
            context->getGlobalData("prague_sky_sun_direction", sun_dir);
        }

        // Cameras are rendered whether or not any band scatters: in a band with a scattering depth of 0 every surface is black, so a camera
        // sees what the surfaces emit (and the sky and sources behind them), but nothing they reflect or transmit
        if (emissionenabled || diffuseenabled || rundirect) {
            // re-set diffuse radiation fluxes (will be passed via launch params)
            if (diffuseenabled) {
                for (auto b = 0; b < Nbands_launch; b++) {
                    diffuse_flux.at(b) = getDiffuseFlux(band_labels.at(b));
                }
            }

            size_t n = ceil(sqrt(double(diffuseRayCount)));

            // Per-band diffuse flux and emission-flag vectors for camera longwave/emission sky sampling, the same for every camera.
            // For emission bands, getDiffuseFlux() returns the user-set sky thermal flux (W/m²). The shortwave sky is computed per
            // camera in the loop below, since each camera sees it through its own spectral response.
            std::vector<float> camera_diffuse_flux(Nbands_launch, 0.f);
            std::vector<uint32_t> band_emission_flag(Nbands_launch, 0u);
            for (size_t b = 0; b < Nbands_launch; b++) {
                camera_diffuse_flux[b] = getDiffuseFlux(band_labels[b]);
                band_emission_flag[b] = radiation_bands.at(band_labels[b]).emissionFlag ? 1u : 0u;
            }

            // Each camera's white reference: what the camera would record from a white surface in place of each primitive, lit by the light arriving there straight from a source or the sky.
            // The backend accumulates it over all the direct and diffuse launches above, and each camera's "auto" white balance is computed from it (see sumCameraWhiteReference()).
            std::vector<float> white_reference_top_cam;
            std::vector<float> white_reference_bottom_cam;
            backend->getWhiteReferenceResults(white_reference_top_cam, white_reference_bottom_cam);
            if (white_reference_top_cam.size() != Ncameras * camera_scatter_stride || white_reference_bottom_cam.size() != Ncameras * camera_scatter_stride) {
                helios_runtime_error("ERROR (RadiationModel::runBand): The ray-tracing backend returned " + std::to_string(white_reference_top_cam.size()) + " white reference values, but " + std::to_string(Ncameras) +
                                     " camera(s), " + std::to_string(Nprimitives) + " primitives and " + std::to_string(Nbands_launch) + " launched band(s) require " + std::to_string(Ncameras * camera_scatter_stride) +
                                     ". The backend's white reference buffers were not sized for the current camera set.");
            }

            // With camera flux smoothing, the radiation reaching and leaving each face of a primitive with a spectral texture, averaged onto the
            // shared vertices, so that cameras see each class shaded smoothly across the facets. The same for every camera.
            const bool smooth_spectral_textures = cameraflux_smoothing_enabled && smoothing_vertex_count > 0 && !spectral_incident.empty();
            std::vector<float> spectral_vertex_values;
            if (smooth_spectral_textures) {
                accumulateSpectralIncidentAtVertices(spectral_incident, spectral_emitted, Nsources + 1, Nbands_launch, spectral_vertex_values);
            }

            // With smoothing, the camera evaluates the direct radiation of the sources per facet with the interpolated shading normal, and the angle-dependent terms of a specular surface layer likewise. The
            // direct radiation reaching each primitive is fetched once for all cameras, and the vertex normals are held by the backend only while smoothing is enabled and there is a source or a layer.
            const bool camera_rescales_direct = cameraflux_smoothing_enabled && smoothing_vertex_count > 0 && Nsources > 0;
            std::vector<float> direct_flux;
            if (camera_rescales_direct) {
                backend->getDirectFluxResults(direct_flux, Nbands_launch);
                std::vector<float> vertex_irradiance;
                accumulateDirectIrradianceAtVertices(direct_flux, Nbands_launch, vertex_irradiance);
                backend->uploadVertexDirectIrradiance(vertex_irradiance);
                vertex_direct_irradiance_uploaded = true;
            } else if (vertex_direct_irradiance_uploaded) {
                backend->uploadVertexDirectIrradiance({});
                vertex_direct_irradiance_uploaded = false;
            }
            const bool camera_needs_vertex_normals = cameraflux_smoothing_enabled && smoothing_vertex_count > 0 && (outgoing_components == 2 || camera_rescales_direct);
            if (camera_needs_vertex_normals != smoothing_vertex_normals_uploaded) {
                backend->uploadVertexNormals(camera_needs_vertex_normals ? smoothing_vertex_normals : std::vector<float>{});
                smoothing_vertex_normals_uploaded = camera_needs_vertex_normals;
            }

            uint cam = 0;
            for (auto &camera: cameras) {

                // Skip cameras whose bands don't intersect the current dispatch. Without this,
                // every runBand() call iterates every camera even if the camera is bound to
                // bands that aren't being dispatched, wasting a full camera launch on a no-op.
                // This is especially visible for SIF cameras when the user calls runBand("PAR"):
                // the SIF camera (bound to SIF_red/SIF_farred) doesn't need to render anything
                // for the PAR dispatch.
                bool camera_in_dispatch = false;
                for (const auto &camera_band: camera.second.band_labels) {
                    if (std::find(band_labels.begin(), band_labels.end(), camera_band) != band_labels.end()) {
                        camera_in_dispatch = true;
                        break;
                    }
                }
                if (!camera_in_dispatch) {
                    ++cam;
                    continue;
                }

                // This camera's sky: its radiance and angular pattern weighted by this camera's spectral responses. Uploading them here
                // is safe for the diffuse launches, which upload their own sky parameters with every launch.
                std::vector<float> camera_sky_radiance;
                std::vector<helios::vec4> sky_pattern_params;
                computeCameraSky(band_labels, camera.second, cam, diffuseenabled && diffuseRayCount > 0, camera_sky_radiance, sky_pattern_params);
                backend->updateSkyModel(sky_pattern_params, camera_sky_radiance, sun_dir, solar_radiances,
                                        has_sun_source ? 0.999989f : 0.0f, // solar_disk_cos_angle
                                        camera_diffuse_flux, band_emission_flag);

                // Hand this camera its own slice of the accumulated camera-weighted scatter as the radiance it
                // integrates. Cameras read radiation_out during hits, and what a camera integrates is the scattered
                // radiance weighted by its own spectral response with the emitted flux folded in, which is a different
                // quantity both from the primitive's band-weighted outgoing flux and from another camera's slice.
                // Cleared first, so that a camera never reads the class radiation of an earlier dispatch
                backend->uploadSpectralCameraFlux({}, {});
                backend->uploadSpectralVertexIncident({}, 0, cam);
                // Each component of this camera's slice, laid out as the outgoing buffers are: [component][primitive][band]
                std::vector<float> camera_scatter_top, camera_scatter_bottom;
                for (size_t component = 0; component < outgoing_components; component++) {
                    const size_t begin = (component * Ncameras + cam) * camera_scatter_stride;
                    camera_scatter_top.insert(camera_scatter_top.end(), scatter_top_cam.begin() + std::ptrdiff_t(begin), scatter_top_cam.begin() + std::ptrdiff_t(begin + camera_scatter_stride));
                    camera_scatter_bottom.insert(camera_scatter_bottom.end(), scatter_bottom_cam.begin() + std::ptrdiff_t(begin), scatter_bottom_cam.begin() + std::ptrdiff_t(begin + camera_scatter_stride));
                }
                backend->uploadRadiationOut(camera_scatter_top, camera_scatter_bottom);

                // A primitive with a spectral texture is seen in the outgoing radiation of the class it is hit at, rather than its own
                std::vector<uint32_t> class_flux_offset;
                std::vector<float> class_flux;
                if (!spectral_incident.empty()) {
                    computeSpectralCameraFlux(spectral_incident, spectral_emitted, cam, band_launch_flag, class_flux_offset, class_flux);
                }
                backend->uploadSpectralCameraFlux(class_flux_offset, class_flux);
                if (smooth_spectral_textures) {
                    backend->uploadSpectralVertexIncident(spectral_vertex_values, 2u, cam);
                }

                // Resample the same field this camera is about to read onto the shared mesh vertices.
                if (cameraflux_smoothing_enabled && smoothing_vertex_count > 0) {
                    std::vector<float> vertex_flux_top, vertex_flux_bottom;
                    if (camera_rescales_direct) {
                        std::vector<float> remainder_top = camera_scatter_top;
                        std::vector<float> remainder_bottom = camera_scatter_bottom;
                        subtractCameraDirectFlux(remainder_top, remainder_bottom, direct_flux, cam, band_launch_flag);
                        accumulateCameraFluxAtVertices(remainder_top, remainder_bottom, Nbands_launch, outgoing_components, vertex_flux_top, vertex_flux_bottom);
                    } else {
                        accumulateCameraFluxAtVertices(camera_scatter_top, camera_scatter_bottom, Nbands_launch, outgoing_components, vertex_flux_top, vertex_flux_bottom);
                    }
                    backend->uploadVertexRadiationOut(vertex_flux_top, vertex_flux_bottom);
                }

                // Validate antialiasing samples don't exceed maximum
                if (camera.second.antialiasing_samples > maxRays) {
                    helios_runtime_error("ERROR (runBand): Camera '" + camera.second.label + "' antialiasing samples (" + std::to_string(camera.second.antialiasing_samples) + ") exceeds OptiX maximum launch size (" + std::to_string(maxRays) +
                                         "). Reduce antialiasing samples.");
                }

                // Compute tiling if needed
                std::vector<CameraTile> tiles = computeCameraTiles(camera.second, maxRays);

                if (message_flag && tiles.size() > 1) {
                    std::cout << "Camera '" << camera.second.label << "' requires " << tiles.size() << " tiles" << std::endl;
                }

                // Upload camera spectral response weights for specular reflection
                // Extract per-camera slice from source_data[s].fluxes_cam
                // fluxes_cam is indexed as [band * Ncameras + cam], we need [source * band]
                std::vector<float> cam_weights(Nsources * Nbands_launch, 1.0f);
                for (uint s = 0; s < Nsources; s++) {
                    for (uint b = 0; b < Nbands_launch; b++) {
                        if (!source_data[s].fluxes_cam.empty() && source_data[s].fluxes_cam.size() == Nbands_launch * Ncameras) {
                            cam_weights[s * Nbands_launch + b] = source_data[s].fluxes_cam[b * Ncameras + cam];
                        }
                    }
                }
                backend->uploadSourceFluxesCam(cam_weights);

                // Launch camera rays (tiled or full)
                for (size_t tile_idx = 0; tile_idx < tiles.size(); tile_idx++) {
                    const auto &tile = tiles[tile_idx];

                    // Build params for this tile
                    helios::RayTracingLaunchParams params = buildCameraLaunchParams(camera.second, cam, camera.second.antialiasing_samples, tile.resolution, tile.offset);

                    // Set band parameters (CRITICAL for materials!)
                    params.num_bands_launch = Nbands_launch;
                    params.num_bands_global = Nbands_global;
                    params.random_seed = drawLaunchSeed();
                    std::vector<bool> band_flags(band_launch_flag.begin(), band_launch_flag.end());
                    params.band_launch_flag = band_flags;

                    // Progress message
                    if (message_flag) {
                        if (tiles.size() == 1) {
                            std::cout << "Performing scattering radiation camera ray trace for camera " << camera.second.label << "..." << std::flush;
                        } else {
                            std::cout << "Performing scattering radiation camera ray trace for camera " << camera.second.label << " (tile " << (tile_idx + 1) << " of " << tiles.size() << ")..." << std::flush;
                        }
                    }

                    // Launch through backend
                    backend->launchCameraRays(params);

                    if (message_flag) {
                        if (tiles.size() > 1) {
                            std::cout << "\r" << std::string(120, ' ') << "\r" << std::flush;
                        } else {
                            std::cout << "done." << std::endl;
                        }
                    }
                }

                if (message_flag && tiles.size() > 1) {
                    std::cout << "Performing scattering radiation camera ray trace for camera " << camera.second.label << "...done." << std::endl;
                }

                // Get results from backend
                std::vector<float> radiation_camera;
                std::vector<uint> dummy_labels;
                std::vector<float> dummy_depths;
                backend->getCameraResults(radiation_camera, dummy_labels, dummy_depths, cam, camera.second.resolution);

                // Process pixel data (KEEP EXISTING LOGIC)
                std::string camera_label = camera.second.label;

                for (auto b = 0; b < Nbands_launch; b++) {

                    camera.second.pixel_data[band_labels.at(b)].resize(camera.second.resolution.x * camera.second.resolution.y);

                    std::string data_label = "camera_" + camera_label + "_" + band_labels.at(b);

                    for (auto p = 0; p < camera.second.resolution.x * camera.second.resolution.y; p++) {
                        camera.second.pixel_data.at(band_labels.at(b)).at(p) = radiation_camera.at(p * Nbands_launch + b);
                    }

                    context->setGlobalData(data_label.c_str(), camera.second.pixel_data.at(band_labels.at(b)));
                    camera.second.bands_rendered_since_processing.insert(band_labels.at(b));
                }

                //--- Pixel Labeling Trace ---//

                // Compute tiling for pixel labeling (no antialiasing, 1 ray per pixel)
                RadiationCamera pixel_label_camera = camera.second;
                pixel_label_camera.antialiasing_samples = 1;
                std::vector<CameraTile> pixel_tiles = computeCameraTiles(pixel_label_camera, maxRays);

                // Zero camera pixel buffers once before tile loop
                backend->zeroCameraPixelBuffers(camera.second.resolution);

                // Launch pixel label rays (tiled or full)
                for (size_t tile_idx = 0; tile_idx < pixel_tiles.size(); tile_idx++) {
                    const auto &tile = pixel_tiles[tile_idx];

                    // Build params (reuse buildCameraLaunchParams, antialiasing=1)
                    helios::RayTracingLaunchParams params = buildCameraLaunchParams(pixel_label_camera, cam,
                                                                                    1, // No antialiasing for pixel labeling
                                                                                    tile.resolution, tile.offset);

                    // The launched bands decide which primitives are translucent covers that the pixel-label ray
                    // passes through, exactly as for the camera rays above.
                    params.num_bands_launch = Nbands_launch;
                    params.num_bands_global = Nbands_global;
                    params.band_launch_flag = std::vector<bool>(band_launch_flag.begin(), band_launch_flag.end());

                    // Progress message
                    if (message_flag) {
                        if (pixel_tiles.size() == 1) {
                            std::cout << "Performing camera pixel labeling ray trace for camera " << camera.second.label << "..." << std::flush;
                        } else {
                            std::cout << "Performing camera pixel labeling ray trace for camera " << camera.second.label << " (tile " << (tile_idx + 1) << " of " << pixel_tiles.size() << ")..." << std::flush;
                        }
                    }

                    // Launch through backend
                    backend->launchPixelLabelRays(params);

                    if (message_flag) {
                        if (pixel_tiles.size() > 1) {
                            std::cout << "\r" << std::string(120, ' ') << "\r" << std::flush;
                        } else {
                            std::cout << "done." << std::endl;
                        }
                    }
                }

                if (message_flag && pixel_tiles.size() > 1) {
                    std::cout << "Performing camera pixel labeling ray trace for camera " << camera.second.label << "...done." << std::endl;
                }

                // Get pixel label results
                std::vector<float> dummy_pixel_data;
                backend->getCameraResults(dummy_pixel_data, camera.second.pixel_label_UUID, camera.second.pixel_depth, cam, camera.second.resolution);

                // Pixel labels from GPU already contain Helios UUID+1 (1-indexed, 0=sky).
                // No conversion needed - the intersection programs store actual primitive UUIDs
                // directly from the geometry UUID buffers (patch_UUID, triangle_UUID, etc.).

                // Store results in context (KEEP EXISTING LOGIC)
                std::string data_label = "camera_" + camera_label + "_pixel_UUID";
                context->setGlobalData(data_label.c_str(), camera.second.pixel_label_UUID);

                data_label = "camera_" + camera_label + "_pixel_depth";
                context->setGlobalData(data_label.c_str(), camera.second.pixel_depth);

                // The class of the spectral texture seen in each pixel, for segmentation masks by class (writeImageSegmentationMasksBySpectralClass())
                data_label = "camera_" + camera_label + "_pixel_spectral_class";
                if (material_data.num_palette_entries > 0) {
                    std::vector<uint> pixel_classes;
                    backend->getCameraPixelSpectralClasses(pixel_classes, camera.second.resolution);
                    context->setGlobalData(data_label.c_str(), pixel_classes);
                } else if (context->doesGlobalDataExist(data_label.c_str())) {
                    context->clearGlobalData(data_label.c_str());
                }

                sumCameraWhiteReference(camera.second, cam, band_labels, white_reference_top_cam, white_reference_bottom_cam);

                cam++;
            }
        } else {
            // if scattering is not enabled or all sources have zero flux, we still need to zero the camera buffers
            for (auto &camera: cameras) {
                for (auto b = 0; b < Nbands_launch; b++) {
                    camera.second.pixel_data[band_labels.at(b)].resize(camera.second.resolution.x * camera.second.resolution.y);

                    std::string data_label = "camera_" + camera.second.label + "_" + band_labels.at(b);

                    for (auto p = 0; p < camera.second.resolution.x * camera.second.resolution.y; p++) {
                        camera.second.pixel_data.at(band_labels.at(b)).at(p) = 0.f;
                    }
                    context->setGlobalData(data_label.c_str(), camera.second.pixel_data.at(band_labels.at(b)));
                }
            }
        }
    }

    // Convert camera images to radiance at a sensor above the atmosphere, before exposure rescales them
    for (auto &camera: cameras) {
        if (!camera.second.atmosphere_label.empty()) {
            applyCameraAtmosphere(camera.second, band_labels);
        }
    }

    // Process the image of each camera whose bands have all been rendered since it was last processed: white balance, colour correction and exposure act on the whole
    // image, so a camera rendered one band at a time is processed once its last band is rendered, and a camera not rendered at all is left as it is.
    std::vector<std::string> cameras_to_process;
    for (const auto &camera: cameras) {
        bool all_bands_rendered = !camera.second.band_labels.empty();
        for (const std::string &band: camera.second.band_labels) {
            all_bands_rendered = all_bands_rendered && camera.second.bands_rendered_since_processing.count(band) > 0 && camera.second.pixel_data.count(band) > 0;
        }
        if (all_bands_rendered) {
            cameras_to_process.push_back(camera.first);
        }
    }

    // Identify each camera's red, green and blue bands (RadCamRGBBands), which white balance, colour correction and exposure all act on
    std::map<std::string, std::pair<std::vector<std::string>, std::string>> camera_rgb_bands;
    for (const std::string &camera_label: cameras_to_process) {
        std::string not_rgb_reason;
        std::vector<std::string> rgb_bands = cameraRGBBands(cameras.at(camera_label), not_rgb_reason);
        camera_rgb_bands[camera_label] = {std::move(rgb_bands), not_rgb_reason};
    }

    // Apply camera white balance based on each camera's white_balance setting
    helios::WarningAggregator white_balance_warnings;
    white_balance_warnings.setEnabled(message_flag);
    for (const std::string &camera_label: cameras_to_process) {
        const std::string white_balance_skipped = cameras.at(camera_label).applyCameraWhiteBalance(camera_rgb_bands.at(camera_label).first);
        if (!white_balance_skipped.empty()) {
            white_balance_warnings.addWarning("camera_white_balance_not_applied", white_balance_skipped);
        }
    }
    white_balance_warnings.report(std::cerr);

    // Convert each camera's image from its own RGB to linear sRGB (CameraProperties::color_correction), after white balance
    for (const std::string &camera_label: cameras_to_process) {
        applyAutomaticColorCorrection(cameras.at(camera_label), camera_rgb_bands.at(camera_label).first, camera_rgb_bands.at(camera_label).second);
    }

    // Apply camera exposure based on each camera's exposure setting. Exposure is one scale factor, so it commutes with white balance and colour correction;
    // it comes after them so that auto exposure meters the final image, on which its target was calibrated (CameraProperties::exposure_target).
    for (const std::string &camera_label: cameras_to_process) {
        RadiationCamera &camera = cameras.at(camera_label);
        camera.applyCameraExposure(context, camera_rgb_bands.at(camera_label).first, effectiveCameraType(camera));
        camera.bands_rendered_since_processing.clear();
        camera.image_processed = true;
    }

    // deposit any energy that is left to make sure we satisfy conservation of energy

    // Extract ALL results from backend instead of old OptiX buffers
    helios::RayTracingResults results;
    backend->getRadiationResults(results);

    std::vector<float> radiation_flux_data = results.radiation_in;
    std::vector<float> radiation_in_top;
    if (track_face_absorption) {
        backend->getRadiationInTopResults(radiation_in_top);
    }

    // Bands still scattering when the scattering loop ended ran to the deepest scattering depth: what their last launch
    // scattered is what is left in flight.
    if (!absorbed_before_last_launch.empty()) {
        std::vector<double> power_left;
        std::vector<float> max_flux_left;
        measureOutgoingEnergy(results.scatter_buff_top, results.scatter_buff_bottom, power_left, max_flux_left);
        for (size_t b = 0; b < Nbands_launch; b++) {
            if (scattering_depth.at(b) > 0 && band_retired.at(b) == 0) {
                addScatteringTail(b, results.radiation_in, power_left.at(b), launched_scattering_iterations);
                const std::string &bname = band_labels.at(b);
                if (message_flag && !(bname.size() >= 9 && bname.compare(0, 9, "_SIF_exc_") == 0)) {
                    std::cout << "Scattering for band " << bname << " finished after " << launched_scattering_iterations << " iteration(s)" << std::endl;
                }
            }
        }
    }

    // Extract scatter buffer data from backend results
    TBS_top = results.scatter_buff_top;
    TBS_bottom = results.scatter_buff_bottom;
    for (size_t ind = 0; ind < retired_top.size(); ind++) {
        TBS_top.at(ind) += retired_top.at(ind);
        TBS_bottom.at(ind) += retired_bottom.at(ind);
    }

    // A ray that crosses the periodic boundary MAX_PERIODIC_WRAPS times without hitting anything or leaving the domain is treated as
    // blocked. This takes rays nearly parallel to the ground in a domain that is narrow compared with its height, and is negligible
    // in practice; report it when it is not, since the fluxes are then biased low.
    if (launched_rays > 0) {
        const size_t abandoned_rays = backend->getAbandonedRayCount();
        if (double(abandoned_rays) > 1e-4 * double(launched_rays)) {
            std::cerr << "WARNING (RadiationModel::runBand): " << abandoned_rays << " of the " << launched_rays << " rays launched for band(s) ";
            for (size_t b = 0; b < band_labels.size(); b++) {
                std::cerr << (b > 0 ? ", " : "") << band_labels.at(b);
            }
            std::cerr << " (" << 100. * double(abandoned_rays) / double(launched_rays)
                      << "%) crossed the periodic boundary the maximum number of times without hitting anything or leaving the domain, and were treated as blocked. This happens "
                         "for rays nearly parallel to the ground (e.g., a sun close to the horizon) in a domain that is narrow compared with its height; widening the periodic "
                         "domain reduces it."
                      << std::endl;
        }
    }

    std::vector<uint> UUIDs_context_all = context->getAllUUIDs();

    // Create indexer for result extraction
    RadiationBufferIndexer result_indexer(Nprimitives, Nbands_launch);

    for (auto b = 0; b < Nbands_launch; b++) {

        std::string prop = "radiation_flux_" + band_labels.at(b);
        std::vector<float> R(Nprimitives);
        for (size_t u = 0; u < Nprimitives; u++) {
            // Use BufferIndexer: [primitive][band]
            size_t ind = result_indexer(u, b);
            if (band_has_scattering_tail.at(b)) {
                R.at(u) = radiation_flux_data.at(ind) + (scattering_tail.empty() ? 0.f : scattering_tail.at(ind));
            } else {
                R.at(u) = radiation_flux_data.at(ind) + TBS_top.at(ind) + TBS_bottom.at(ind);
                if (outgoing_components == 2) {
                    R.at(u) += TBS_top.at(layer_offset + ind) + TBS_bottom.at(layer_offset + ind);
                }
            }
            R.at(u) += reference_absorption.at(ind); // absorbed from the reference field, which the gathered flux excludes
        }
        context->setPrimitiveData(context_UUIDs, prop.c_str(), R);

        if (UUIDs_context_all.size() != Nprimitives) {
            for (uint UUID: UUIDs_context_all) {
                if (context->doesPrimitiveExist(UUID) && !context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                    context->setPrimitiveData(UUID, prop.c_str(), 0.f);
                }
            }
        }
    }

    // Finalize every SIF excitation set launched here (run directly by runExcitationBands(), or piggy-backed): now that
    // radiation_flux_<_SIF_exc_*> primitive data has been populated, compute each leaf's emission from it and clear the internal
    // primitive data. A subsequent runBand() on SIF emission bands will then skip re-running excitation (populateExcitationSet marks
    // populated=true).
    if (track_face_absorption) {
        for (auto &kv: excitation_sets) {
            ExcitationSet &exc = kv.second;
            const bool launched = std::all_of(exc.band_labels.begin(), exc.band_labels.end(), [&](const std::string &b) { return std::find(band_labels.begin(), band_labels.end(), b) != band_labels.end(); });
            if (launched) {
                populateExcitationSet(exc, band_labels, radiation_flux_data, radiation_in_top);
            }
        }
    }
}

float RadiationModel::getSkyEnergy() {
    helios_runtime_error("ERROR (RadiationModel::getSkyEnergy): The radiative energy lost to the sky is not computed by any ray-tracing backend, so this method has no value to "
                         "return. It is deprecated and will be removed in v2.0.");
    return 0.f; // unreachable
}

std::vector<float> RadiationModel::getTotalAbsorbedFlux() {

    std::vector<float> total_flux;
    total_flux.resize(context->getPrimitiveCount(), 0.f);

    for (const auto &band: radiation_bands) {

        std::string label = band.first;

        for (size_t u = 0; u < context_UUIDs.size(); u++) {

            uint p = context_UUIDs.at(u);

            std::string str = "radiation_flux_" + label;

            float R;
            context->getPrimitiveData(p, str.c_str(), R);
            total_flux.at(u) += R;
        }
    }

    return total_flux;
}


float RadiationModel::calculateGtheta(helios::Context *context, vec3 view_direction) {

    vec3 dir = view_direction;
    dir.normalize();

    float Gtheta = 0;
    float total_area = 0;
    for (std::size_t u = 0; u < primitiveID.size(); u++) {

        uint UUID = context_UUIDs.at(primitiveID.at(u));

        vec3 normal = context->getPrimitiveNormal(UUID);
        float area = context->getPrimitiveArea(UUID);

        Gtheta += fabsf(normal * dir) * area;

        total_area += area;
    }

    return Gtheta / total_area;
}

void RadiationModel::exportColorCorrectionMatrixXML(const std::string &file_path, const std::string &camera_label, const std::vector<std::vector<float>> &matrix, const std::string &source_image_path, const std::string &colorboard_type,
                                                    float average_delta_e) {

    std::ofstream file(file_path);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::exportColorCorrectionMatrixXML): Failed to open file for writing: " + file_path);
    }

    // Determine matrix type (3x3 or 4x3)
    std::string matrix_type = "3x3";
    if (matrix.size() == 4 || (matrix.size() >= 3 && matrix[0].size() == 4)) {
        matrix_type = "4x3";
    }

    // Write XML header with informative comments
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    file << "<!-- Camera Color Correction Matrix -->" << std::endl;
    file << "<!-- Source Image: " << source_image_path << " -->" << std::endl;
    file << "<!-- Camera Label: " << camera_label << " -->" << std::endl;
    file << "<!-- Colorboard Type: " << colorboard_type << " -->" << std::endl;
    if (average_delta_e >= 0.0f) {
        file << "<!-- Average Delta E: " << std::fixed << std::setprecision(2) << average_delta_e << " -->" << std::endl;
    }
    file << "<!-- Matrix Type: " << matrix_type << " -->" << std::endl;
    file << "<!-- Generated: " << getCurrentDateTime() << " -->" << std::endl;

    // Write matrix data
    file << "<helios>" << std::endl;
    file << "  <ColorCorrectionMatrix camera_label=\"" << camera_label << "\" matrix_type=\"" << matrix_type << "\">" << std::endl;

    for (size_t i = 0; i < matrix.size(); i++) {
        file << "    <row>";
        for (size_t j = 0; j < matrix[i].size(); j++) {
            file << std::fixed << std::setprecision(6) << matrix[i][j];
            if (j < matrix[i].size() - 1) {
                file << " ";
            }
        }
        file << "</row>" << std::endl;
    }

    file << "  </ColorCorrectionMatrix>" << std::endl;
    file << "</helios>" << std::endl;

    file.close();
}

uint RadiationModel::drawLaunchSeed() {
    // Use the raw generator output rather than a std:: distribution: the minstd_rand0 sequence is fixed by the C++
    // standard, whereas distributions are implementation-defined and would give a different seed on each platform.
    return uint((*context->getRandomGenerator())());
}

std::string RadiationModel::getCurrentDateTime() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<std::vector<float>> RadiationModel::loadColorCorrectionMatrixXML(const std::string &file_path, std::string &camera_label_out) {

    std::ifstream file(file_path);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::loadColorCorrectionMatrixXML): Failed to open file for reading: " + file_path);
    }

    std::vector<std::vector<float>> matrix;
    std::string line;
    bool in_matrix = false;
    std::string matrix_type = "";

    while (std::getline(file, line)) {
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        // Look for ColorCorrectionMatrix opening tag
        if (line.find("<ColorCorrectionMatrix") != std::string::npos) {
            in_matrix = true;

            // Extract camera_label attribute
            size_t camera_start = line.find("camera_label=\"");
            if (camera_start != std::string::npos) {
                camera_start += 14; // Length of "camera_label=\""
                size_t camera_end = line.find("\"", camera_start);
                if (camera_end != std::string::npos) {
                    camera_label_out = line.substr(camera_start, camera_end - camera_start);
                }
            }

            // Extract matrix_type attribute
            size_t type_start = line.find("matrix_type=\"");
            if (type_start != std::string::npos) {
                type_start += 13; // Length of "matrix_type=\""
                size_t type_end = line.find("\"", type_start);
                if (type_end != std::string::npos) {
                    matrix_type = line.substr(type_start, type_end - type_start);
                }
            }
            continue;
        }

        // Look for ColorCorrectionMatrix closing tag
        if (line.find("</ColorCorrectionMatrix>") != std::string::npos) {
            in_matrix = false;
            break;
        }

        // Parse row data
        if (in_matrix && line.find("<row>") != std::string::npos && line.find("</row>") != std::string::npos) {
            // Extract content between <row> and </row>
            size_t start = line.find("<row>") + 5;
            size_t end = line.find("</row>");
            std::string row_data = line.substr(start, end - start);

            // Parse float values from row
            std::vector<float> row;
            std::istringstream iss(row_data);
            float value;
            while (iss >> value) {
                row.push_back(value);
            }

            if (!row.empty()) {
                matrix.push_back(row);
            }
        }
    }

    file.close();

    // Validate loaded matrix
    if (matrix.empty()) {
        helios_runtime_error("ERROR (RadiationModel::loadColorCorrectionMatrixXML): No matrix data found in file: " + file_path);
    }

    if (matrix.size() != 3) {
        helios_runtime_error("ERROR (RadiationModel::loadColorCorrectionMatrixXML): Invalid matrix size. Expected 3 rows, found " + std::to_string(matrix.size()) + " rows in file: " + file_path);
    }

    // Validate matrix type consistency
    bool is_3x3 = (matrix[0].size() == 3 && matrix[1].size() == 3 && matrix[2].size() == 3);
    bool is_4x3 = (matrix[0].size() == 4 && matrix[1].size() == 4 && matrix[2].size() == 4);

    if (!is_3x3 && !is_4x3) {
        helios_runtime_error("ERROR (RadiationModel::loadColorCorrectionMatrixXML): Invalid matrix dimensions. All rows must have either 3 or 4 elements. File: " + file_path);
    }

    // Check matrix type attribute matches actual dimensions
    if (!matrix_type.empty()) {
        if ((matrix_type == "3x3" && !is_3x3) || (matrix_type == "4x3" && !is_4x3)) {
            helios_runtime_error("ERROR (RadiationModel::loadColorCorrectionMatrixXML): Matrix type attribute ('" + matrix_type + "') does not match actual matrix dimensions in file: " + file_path);
        }
    }

    return matrix;
}

std::string RadiationModel::autoCalibrateCameraImage(const std::string &camera_label, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, const std::string &output_file_path,
                                                     bool print_quality_report, ColorCorrectionAlgorithm algorithm, const std::string &ccm_export_file_path) {

    // Step 1: Validate camera exists and get pixel UUID data
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Camera '" + camera_label + "' does not exist. Make sure the camera was added to the radiation model.");
    }
    requireCompleteCameraImage(camera_label, "autoCalibrateCameraImage");

    // Get camera pixel UUID data from global data (needed for segmentation)
    std::string pixel_UUID_label = "camera_" + camera_label + "_pixel_UUID";
    if (!context->doesGlobalDataExist(pixel_UUID_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Camera pixel UUID data '" + pixel_UUID_label + "' does not exist for camera '" + camera_label + "'. Make sure the radiation model has been run.");
    }

    // Step 2: Detect all colorboard types using CameraCalibration helper
    CameraCalibration calibration(context);
    std::vector<std::string> colorboard_types;
    try {
        colorboard_types = calibration.detectColorBoardTypes();
    } catch (const std::exception &e) {
        helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Failed to detect colorboard types. " + std::string(e.what()));
    }

    // Step 3: Check that the requested bands belong to the camera and have been rendered
    const RadiationCamera &camera = cameras.at(camera_label);
    for (const std::string &band_label: {red_band_label, green_band_label, blue_band_label}) {
        if (std::find(camera.band_labels.begin(), camera.band_labels.end(), band_label) == camera.band_labels.end()) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Band '" + band_label + "' not found in camera '" + camera_label + "'.");
        }
        if (camera.pixel_data.find(band_label) == camera.pixel_data.end()) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Image data for camera '" + camera_label + "', band '" + band_label + "' has not been created. Call runBand() with this band after the camera was added.");
        }
    }
    std::vector<float> red_data = camera.pixel_data.at(red_band_label);
    std::vector<float> green_data = camera.pixel_data.at(green_band_label);
    std::vector<float> blue_data = camera.pixel_data.at(blue_band_label);

    // Calibrate the camera's own RGB. An image that was colour corrected automatically (CameraProperties::color_correction) has that correction undone on this copy, so that the
    // fitted matrix, like any matrix given to applyCameraColorCorrectionMatrix(), applies to the camera's own RGB, and does not depend on whether the camera corrects its images.
    const std::vector<float> &automatic_matrix = cameras.at(camera_label).applied_color_correction_matrix;
    if (!automatic_matrix.empty()) {
        std::string not_rgb_reason;
        const std::vector<std::string> corrected_bands = cameraRGBBands(cameras.at(camera_label), not_rgb_reason);
        if (corrected_bands != std::vector<std::string>{red_band_label, green_band_label, blue_band_label}) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Camera '" + camera_label + "' was asked to calibrate bands '" + red_band_label + "', '" + green_band_label + "' and '" + blue_band_label +
                                 "' as red, green and blue, but its image was colour corrected automatically with '" + corrected_bands.at(0) + "', '" + corrected_bands.at(1) + "' and '" + corrected_bands.at(2) +
                                 "' as red, green and blue. Pass those bands, or set CameraProperties::color_correction to \"off\" for this camera and render again.");
        }
        const std::vector<float> &m = automatic_matrix;
        const double determinant = double(m[0]) * (double(m[4]) * m[8] - double(m[5]) * m[7]) - double(m[1]) * (double(m[3]) * m[8] - double(m[5]) * m[6]) + double(m[2]) * (double(m[3]) * m[7] - double(m[4]) * m[6]);
        if (!(std::fabs(determinant) > 1e-8)) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): The colour correction matrix applied automatically to camera '" + camera_label +
                                 "' cannot be inverted, so the camera's own RGB cannot be recovered for calibration. Set CameraProperties::color_correction to \"off\" for this camera and render again.");
        }
        // Inverse by the adjugate
        const double inverse[9] = {(double(m[4]) * m[8] - double(m[5]) * m[7]) / determinant, (double(m[2]) * m[7] - double(m[1]) * m[8]) / determinant, (double(m[1]) * m[5] - double(m[2]) * m[4]) / determinant,
                                   (double(m[5]) * m[6] - double(m[3]) * m[8]) / determinant, (double(m[0]) * m[8] - double(m[2]) * m[6]) / determinant, (double(m[2]) * m[3] - double(m[0]) * m[5]) / determinant,
                                   (double(m[3]) * m[7] - double(m[4]) * m[6]) / determinant, (double(m[1]) * m[6] - double(m[0]) * m[7]) / determinant, (double(m[0]) * m[4] - double(m[1]) * m[3]) / determinant};
        for (size_t p = 0; p < red_data.size(); p++) {
            const double r = red_data[p];
            const double g = green_data[p];
            const double b = blue_data[p];
            red_data[p] = float(inverse[0] * r + inverse[1] * g + inverse[2] * b);
            green_data[p] = float(inverse[3] * r + inverse[4] * g + inverse[5] * b);
            blue_data[p] = float(inverse[6] * r + inverse[7] * g + inverse[8] * b);
        }
    }

    // Step 4: Average the camera's RGB over the pixels of each colour board patch in view. The reference colour of a patch is the colorimetric linear sRGB (CIE D65) of the
    // reflectance spectrum it was rendered with, so that the calibration and the render share one definition of each patch's colour.
    std::vector<uint> pixel_UUIDs;
    context->getGlobalData(pixel_UUID_label.c_str(), pixel_UUIDs);
    const int2 camera_resolution = camera.resolution;
    struct PatchSamples {
        helios::vec3 rgb_sum = make_vec3(0, 0, 0);
        int pixel_count = 0;
        std::string spectrum_label;
    };
    std::map<std::pair<std::string, uint>, PatchSamples> patches; // keyed by (colour board type, patch index)
    for (size_t pixel = 0; pixel < pixel_UUIDs.size() && pixel < red_data.size(); pixel++) {
        if (pixel_UUIDs[pixel] == 0) {
            continue; // sky
        }
        const uint UUID = pixel_UUIDs[pixel] - 1; // 1-based
        if (!context->doesPrimitiveExist(UUID)) {
            continue;
        }
        for (const std::string &colorboard_type: colorboard_types) {
            const std::string colorboard_data_label = "colorboard_" + colorboard_type;
            if (!context->doesPrimitiveDataExist(UUID, colorboard_data_label.c_str())) {
                continue;
            }
            uint patch_index;
            context->getPrimitiveData(UUID, colorboard_data_label.c_str(), patch_index);
            PatchSamples &samples = patches[{colorboard_type, patch_index}];
            if (samples.spectrum_label.empty()) {
                if (!context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
                    helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Patch " + std::to_string(patch_index) + " of the " + colorboard_type +
                                         " colour board has no \"reflectivity_spectrum\" primitive data, so its reference colour is unknown. Add colour boards with CameraCalibration, which sets it.");
                }
                context->getPrimitiveData(UUID, "reflectivity_spectrum", samples.spectrum_label);
            }
            samples.rgb_sum = samples.rgb_sum + make_vec3(red_data[pixel], green_data[pixel], blue_data[pixel]);
            samples.pixel_count++;
        }
    }

    // Patches seen by too few pixels are left out: their mean is dominated by pixels that straddle the patch edges
    const int minimum_patch_pixels = 10;
    std::vector<helios::vec3> measured_rgb, reference_rgb;
    std::vector<std::string> patch_names;
    std::vector<float> patch_weights;
    for (const auto &[key, samples]: patches) {
        if (samples.pixel_count <= minimum_patch_pixels) {
            continue;
        }
        measured_rgb.push_back(samples.rgb_sum / float(samples.pixel_count));
        const helios::vec3 reference = colorimetricLinearSRGB(loadSpectralData(samples.spectrum_label));
        reference_rgb.push_back(reference);
        patch_names.push_back(key.first + " patch " + std::to_string(key.second) + " (" + samples.spectrum_label + ")");
        // Neutral patches (CIELAB chroma below 5) count double, so that greys stay grey
        const CameraCalibration::LabColor reference_lab = calibration.linearRgbToLab(reference);
        patch_weights.push_back(std::sqrt(reference_lab.a * reference_lab.a + reference_lab.b * reference_lab.b) < 5.f ? 2.f : 1.f);
    }
    const size_t visible_patches = measured_rgb.size();

    // Step 5: Fit the matrix M minimizing the weighted squared error of M * measured - reference, in linear RGB
    std::vector<std::vector<float>> correction_matrix(3, std::vector<float>(3, 0.f));
    std::string algorithm_name;
    const auto fit_diagonal = [&]() {
        if (visible_patches < 1) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): No colour board patch is seen by more than " + std::to_string(minimum_patch_pixels) + " pixels of camera '" + camera_label +
                                 "', so no correction can be fitted. Move the camera closer to the board or increase its resolution.");
        }
        for (int k = 0; k < 3; k++) {
            double measured_squared = 0, measured_reference = 0;
            for (size_t i = 0; i < visible_patches; i++) {
                const double m = k == 0 ? measured_rgb[i].x : k == 1 ? measured_rgb[i].y : measured_rgb[i].z;
                const double t = k == 0 ? reference_rgb[i].x : k == 1 ? reference_rgb[i].y : reference_rgb[i].z;
                measured_squared += patch_weights[i] * m * m;
                measured_reference += patch_weights[i] * m * t;
            }
            if (!(measured_squared > 0)) {
                helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Camera '" + camera_label + "' recorded nothing in its " + (k == 0 ? red_band_label : k == 1 ? green_band_label : blue_band_label) +
                                     " band over the colour board patches in view, so no correction can be fitted. Check that the board is lit in that band.");
            }
            for (int j = 0; j < 3; j++) {
                correction_matrix[k][j] = j == k ? float(measured_reference / measured_squared) : 0.f;
            }
        }
    };
    // Returns false if the patches in view do not determine a full matrix (they span too few distinct colours)
    const auto fit_full = [&]() {
        if (visible_patches < 6) {
            return false;
        }
        double MtM[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double MtT[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}; // MtT[j][k] = sum of w * measured_j * reference_k
        for (size_t i = 0; i < visible_patches; i++) {
            const double m[3] = {measured_rgb[i].x, measured_rgb[i].y, measured_rgb[i].z};
            const double t[3] = {reference_rgb[i].x, reference_rgb[i].y, reference_rgb[i].z};
            for (int j = 0; j < 3; j++) {
                for (int l = 0; l < 3; l++) {
                    MtM[j][l] += patch_weights[i] * m[j] * m[l];
                    MtT[j][l] += patch_weights[i] * m[j] * t[l];
                }
            }
        }
        // The normal equations are shared by the three rows of the matrix: row k solves MtM * x = (column k of MtT). Solved with the inverse by the adjugate.
        const double determinant = MtM[0][0] * (MtM[1][1] * MtM[2][2] - MtM[1][2] * MtM[2][1]) - MtM[0][1] * (MtM[1][0] * MtM[2][2] - MtM[1][2] * MtM[2][0]) + MtM[0][2] * (MtM[1][0] * MtM[2][1] - MtM[1][1] * MtM[2][0]);
        const double scale = MtM[0][0] * MtM[1][1] * MtM[2][2];
        if (!(scale > 0) || !(std::fabs(determinant) > 1e-6 * scale)) {
            return false;
        }
        const double inverse[3][3] = {{(MtM[1][1] * MtM[2][2] - MtM[1][2] * MtM[2][1]) / determinant, (MtM[0][2] * MtM[2][1] - MtM[0][1] * MtM[2][2]) / determinant, (MtM[0][1] * MtM[1][2] - MtM[0][2] * MtM[1][1]) / determinant},
                                      {(MtM[1][2] * MtM[2][0] - MtM[1][0] * MtM[2][2]) / determinant, (MtM[0][0] * MtM[2][2] - MtM[0][2] * MtM[2][0]) / determinant, (MtM[0][2] * MtM[1][0] - MtM[0][0] * MtM[1][2]) / determinant},
                                      {(MtM[1][0] * MtM[2][1] - MtM[1][1] * MtM[2][0]) / determinant, (MtM[0][1] * MtM[2][0] - MtM[0][0] * MtM[2][1]) / determinant, (MtM[0][0] * MtM[1][1] - MtM[0][1] * MtM[1][0]) / determinant}};
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
                correction_matrix[k][j] = float(inverse[j][0] * MtT[0][k] + inverse[j][1] * MtT[1][k] + inverse[j][2] * MtT[2][k]);
            }
        }
        return true;
    };

    if (algorithm == ColorCorrectionAlgorithm::DIAGONAL_ONLY) {
        fit_diagonal();
        algorithm_name = "Diagonal scaling (white balance only)";
    } else if (algorithm == ColorCorrectionAlgorithm::MATRIX_3X3_FORCE) {
        if (!fit_full()) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): The " + std::to_string(visible_patches) + " colour board patches seen by camera '" + camera_label +
                                 "' do not determine a 3x3 matrix (at least 6 patches of distinct colours are needed). Bring more of the board into view, or use ColorCorrectionAlgorithm::MATRIX_3X3_AUTO or DIAGONAL_ONLY.");
        }
        algorithm_name = "3x3 matrix (forced)";
    } else {
        // A full matrix unless the patches do not determine one, or it has implausibly large entries (a sign that the patches in view span too few colours)
        bool full_matrix_usable = fit_full();
        for (int k = 0; k < 3 && full_matrix_usable; k++) {
            for (int j = 0; j < 3; j++) {
                full_matrix_usable = full_matrix_usable && std::fabs(correction_matrix[k][j]) <= 5.f * std::fabs(correction_matrix[k][k]) + 5.f;
            }
        }
        if (full_matrix_usable) {
            algorithm_name = "3x3 matrix";
        } else {
            fit_diagonal();
            algorithm_name = "Diagonal scaling (the patches in view did not determine a stable 3x3 matrix)";
            if (message_flag) {
                std::cerr << "WARNING (RadiationModel::autoCalibrateCameraImage): The " << visible_patches << " colour board patches seen by camera '" << camera_label
                          << "' did not determine a stable 3x3 matrix, so a diagonal correction (white balance only) was fitted instead." << std::endl;
            }
        }
    }

    // Step 6: Colour error of each patch after correction (CIEDE2000)
    std::vector<double> patch_delta_E(visible_patches);
    double total_delta_E = 0;
    for (size_t i = 0; i < visible_patches; i++) {
        const helios::vec3 &m = measured_rgb[i];
        const helios::vec3 corrected = make_vec3(correction_matrix[0][0] * m.x + correction_matrix[0][1] * m.y + correction_matrix[0][2] * m.z, correction_matrix[1][0] * m.x + correction_matrix[1][1] * m.y + correction_matrix[1][2] * m.z,
                                                 correction_matrix[2][0] * m.x + correction_matrix[2][1] * m.y + correction_matrix[2][2] * m.z);
        patch_delta_E[i] = calibration.deltaE2000(calibration.linearRgbToLab(corrected), calibration.linearRgbToLab(reference_rgb[i]));
        total_delta_E += patch_delta_E[i];
    }
    const double mean_delta_E = total_delta_E / double(visible_patches);

    if (print_quality_report) {
        std::cout << "\n========== COLOR CALIBRATION QUALITY REPORT ==========" << std::endl;
        std::cout << "Camera: " << camera_label << std::endl;
        std::cout << "Patches analyzed: " << visible_patches << std::endl;
        std::cout << "Algorithm used: " << algorithm_name << std::endl;
        std::cout << "Color correction matrix (linear camera RGB to linear sRGB):" << std::endl;
        for (int k = 0; k < 3; k++) {
            std::cout << "  [" << std::fixed << std::setprecision(4) << std::setw(9) << correction_matrix[k][0] << " " << std::setw(9) << correction_matrix[k][1] << " " << std::setw(9) << correction_matrix[k][2] << "]" << std::endl;
        }
        std::cout << "\nPer-patch CIEDE2000 after correction:" << std::endl;
        for (size_t i = 0; i < visible_patches; i++) {
            std::cout << "  " << patch_names[i] << ": " << std::fixed << std::setprecision(2) << patch_delta_E[i] << std::endl;
        }
        std::cout << "Mean CIEDE2000: " << std::fixed << std::setprecision(2) << mean_delta_E << std::endl;
        std::cout << "======================================================\n" << std::endl;
    }

    // Step 7: Write the corrected image, sRGB encoded, with the same pixel ordering as writeCameraImage()
    std::vector<helios::RGBcolor> corrected_pixels(red_data.size());
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            const int source_index = j * camera_resolution.x + i;
            const float r = red_data[source_index];
            const float g = green_data[source_index];
            const float b = blue_data[source_index];
            const float corrected_r = correction_matrix[0][0] * r + correction_matrix[0][1] * g + correction_matrix[0][2] * b;
            const float corrected_g = correction_matrix[1][0] * r + correction_matrix[1][1] * g + correction_matrix[1][2] * b;
            const float corrected_b = correction_matrix[2][0] * r + correction_matrix[2][1] * g + correction_matrix[2][2] * b;
            const uint ii = camera_resolution.x - i - 1; // horizontal flip
            const uint jj = camera_resolution.y - j - 1; // vertical flip
            corrected_pixels[jj * camera_resolution.x + ii] =
                    make_RGBcolor(RadiationCamera::lin_to_srgb(std::fmaxf(0.f, corrected_r)), RadiationCamera::lin_to_srgb(std::fmaxf(0.f, corrected_g)), RadiationCamera::lin_to_srgb(std::fmaxf(0.f, corrected_b)));
        }
    }

    std::string output_path = output_file_path;
    if (output_path.empty()) {
        output_path = "auto_calibrated_" + camera_label + ".jpg";
    }
    try {
        helios::writeJPEG(output_path, camera_resolution.x, camera_resolution.y, corrected_pixels);
        std::cout << "Wrote corrected image to: " << output_path << std::endl;
    } catch (const std::exception &e) {
        helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Failed to write corrected image. " + std::string(e.what()));
    }

    // Step 8: Export the matrix if requested
    if (!ccm_export_file_path.empty()) {
        std::string colorboard_types_str;
        for (size_t i = 0; i < colorboard_types.size(); i++) {
            colorboard_types_str += (i > 0 ? ", " : "") + colorboard_types[i];
        }
        try {
            exportColorCorrectionMatrixXML(ccm_export_file_path, camera_label, correction_matrix, output_path, colorboard_types_str, float(mean_delta_E));
        } catch (const std::exception &e) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Failed to export CCM to XML. " + std::string(e.what()));
        }
        std::cout << "Exported color correction matrix to: " << ccm_export_file_path << std::endl;
    }

    return output_path;
}

void RadiationModel::applyCameraColorCorrectionMatrix(const std::string &camera_label, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, const std::string &ccm_file_path) {

    // Step 1: Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Camera '" + camera_label + "' does not exist. Make sure the camera was added to the radiation model.");
    }
    requireCompleteCameraImage(camera_label, "applyCameraColorCorrectionMatrix");
    if (!cameras.at(camera_label).applied_color_correction_matrix.empty()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Camera '" + camera_label +
                             "' already colour corrected its current image automatically (its CameraProperties::color_correction is \"auto\" and it has spectral responses), so applying another matrix would "
                             "correct the colours twice. Set CameraProperties::color_correction to \"off\" for this camera (e.g. with updateCameraParameters()) and render again to apply your own matrix.");
    }

    // Step 2: Validate band labels exist in camera
    auto &camera_bands = cameras.at(camera_label).band_labels;
    if (std::find(camera_bands.begin(), camera_bands.end(), red_band_label) == camera_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Red band '" + red_band_label + "' not found in camera '" + camera_label + "'.");
    }
    if (std::find(camera_bands.begin(), camera_bands.end(), green_band_label) == camera_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Green band '" + green_band_label + "' not found in camera '" + camera_label + "'.");
    }
    if (std::find(camera_bands.begin(), camera_bands.end(), blue_band_label) == camera_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Blue band '" + blue_band_label + "' not found in camera '" + camera_label + "'.");
    }

    // Step 3: Load color correction matrix from XML file
    std::string loaded_camera_label;
    std::vector<std::vector<float>> correction_matrix;
    try {
        correction_matrix = loadColorCorrectionMatrixXML(ccm_file_path, loaded_camera_label);
    } catch (const std::exception &e) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Failed to load CCM from XML file. " + std::string(e.what()));
    }

    // Step 4: Validate matrix dimensions (should be 3x3 or 4x3)
    if (correction_matrix.size() != 3) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Invalid matrix dimensions. Expected 3x3 or 4x3 matrix, got " + std::to_string(correction_matrix.size()) + " rows.");
    }

    bool is_3x3 = (correction_matrix[0].size() == 3);
    bool is_4x3 = (correction_matrix[0].size() == 4);

    if (!is_3x3 && !is_4x3) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Invalid matrix dimensions. Expected 3x3 or 4x3 matrix, got " + std::to_string(correction_matrix.size()) + "x" + std::to_string(correction_matrix[0].size()) +
                             " matrix.");
    }

    // Step 5: Get camera data (same approach as applyImageProcessingPipeline)
    std::vector<float> &red_data = cameras.at(camera_label).pixel_data.at(red_band_label);
    std::vector<float> &green_data = cameras.at(camera_label).pixel_data.at(green_band_label);
    std::vector<float> &blue_data = cameras.at(camera_label).pixel_data.at(blue_band_label);

    int2 camera_resolution = cameras.at(camera_label).resolution;
    size_t pixel_count = red_data.size();

    // Step 6: Apply color correction matrix to all pixels in-place
    for (size_t i = 0; i < pixel_count; i++) {
        float r = red_data[i];
        float g = green_data[i];
        float b = blue_data[i];

        // Apply color correction matrix (3x3 or 4x3)
        if (is_3x3) {
            // Standard 3x3 matrix transformation
            red_data[i] = correction_matrix[0][0] * r + correction_matrix[0][1] * g + correction_matrix[0][2] * b;
            green_data[i] = correction_matrix[1][0] * r + correction_matrix[1][1] * g + correction_matrix[1][2] * b;
            blue_data[i] = correction_matrix[2][0] * r + correction_matrix[2][1] * g + correction_matrix[2][2] * b;
        } else {
            // 4x3 matrix transformation with affine offset
            red_data[i] = correction_matrix[0][0] * r + correction_matrix[0][1] * g + correction_matrix[0][2] * b + correction_matrix[0][3];
            green_data[i] = correction_matrix[1][0] * r + correction_matrix[1][1] * g + correction_matrix[1][2] * b + correction_matrix[1][3];
            blue_data[i] = correction_matrix[2][0] * r + correction_matrix[2][1] * g + correction_matrix[2][2] * b + correction_matrix[2][3];
        }
    }

    if (message_flag) {
        std::cout << "Applied color correction matrix from '" << ccm_file_path << "' to camera '" << camera_label << "'" << std::endl;
        std::cout << "Matrix type: " << (is_3x3 ? "3x3" : "4x3") << ", processed " << pixel_count << " pixels" << std::endl;
    }
}

std::vector<std::vector<float>> RadiationModel::getCameraColorCorrectionMatrix(const std::string &camera_label) const {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraColorCorrectionMatrix): Camera '" + camera_label + "' does not exist.");
    }
    std::string undefined_reason;
    const std::vector<float> matrix = computeColorCorrectionMatrix(cameras.at(camera_label), undefined_reason);
    if (matrix.empty()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraColorCorrectionMatrix): " + undefined_reason);
    }
    return {{matrix[0], matrix[1], matrix[2]}, {matrix[3], matrix[4], matrix[5]}, {matrix[6], matrix[7], matrix[8]}};
}

namespace {
    //! Whether a camera band has a spectral response (anything but no response or "uniform")
    bool cameraBandHasSpectralResponse(const RadiationCamera &camera, const std::string &band) {
        const auto response = camera.band_spectral_response.find(band);
        return response != camera.band_spectral_response.end() && !trim_whitespace(response->second).empty() && response->second != "uniform";
    }

    //! The colour a band is named for: 0 for "red" or "r", 1 for "green" or "g", 2 for "blue" or "b" (in any case), -1 for any other name
    int bandNameColour(const std::string &band) {
        std::string name = trim_whitespace(band);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (name == "red" || name == "r") {
            return 0;
        } else if (name == "green" || name == "g") {
            return 1;
        } else if (name == "blue" || name == "b") {
            return 2;
        }
        return -1;
    }

    //! Spectral class of a band from the response-weighted mean wavelength of its spectral response: 0 red, 1 green, 2 blue, 3 near-infrared, 4 ultraviolet
    /**
     * The boundaries sit between the responses of real cameras: the camera library's blue responses are centred at 460-490 nm, green at 525-555 nm and red at 588-640 nm, and its near-infrared
     * responses at 745 nm and above.
     */
    int spectralClass(float mean_wavelength) {
        if (mean_wavelength < 400.f) {
            return 4;
        } else if (mean_wavelength < 505.f) {
            return 2;
        } else if (mean_wavelength < 570.f) {
            return 1;
        } else if (mean_wavelength <= 700.f) {
            return 0;
        }
        return 3;
    }

    const char *spectralClassName(int spectral_class) {
        static const char *names[] = {"red", "green", "blue", "near-infrared", "ultraviolet"};
        return names[spectral_class];
    }
} // namespace

std::string RadiationModel::effectiveCameraType(const RadiationCamera &camera) const {
    if (!camera.camera_type.empty()) {
        return camera.camera_type;
    }
    // Emission is enabled by default, so it is left enabled in many visible bands where it is incidental; a band is a thermal band only if its wavelength bounds
    // also lie in the thermal infrared. A band without bounds may be either, so a camera with one is not taken to be a thermal camera.
    const float thermal_infrared_min_nm = 3000.f;
    const bool all_bands_thermal = !camera.band_labels.empty() && std::all_of(camera.band_labels.begin(), camera.band_labels.end(), [&](const std::string &band) {
        auto band_it = radiation_bands.find(band);
        return band_it != radiation_bands.end() && band_it->second.emissionFlag && sif_emission_bands.count(band) == 0 && band_it->second.hasWavebandBounds() && band_it->second.wavebandBounds.x >= thermal_infrared_min_nm;
    });
    return all_bands_thermal ? "thermal" : "";
}

std::vector<std::string> RadiationModel::cameraRGBBands(const RadiationCamera &camera, std::string &not_rgb_reason) const {
    not_rgb_reason.clear();
    // Thermal and multispectral cameras, and cameras with fewer than three bands, are not RGB cameras
    const std::string camera_type = effectiveCameraType(camera);
    if ((!camera_type.empty() && camera_type != "rgb") || camera.band_labels.size() < 3) {
        return {};
    }
    const std::vector<std::string> &bands = camera.band_labels;
    const size_t Nbands = bands.size();

    // The colour each band is named for, and the band named for each colour (empty if none, or more than one)
    std::vector<int> name_colour(Nbands);
    std::string named_band[3];
    int named_count[3] = {0, 0, 0};
    for (size_t b = 0; b < Nbands; b++) {
        name_colour[b] = bandNameColour(bands[b]);
        if (name_colour[b] >= 0) {
            named_count[name_colour[b]]++;
            named_band[name_colour[b]] = bands[b];
        }
    }
    for (int c = 0; c < 3; c++) {
        if (named_count[c] != 1) {
            named_band[c].clear();
        }
    }
    const bool all_colours_named = !named_band[0].empty() && !named_band[1].empty() && !named_band[2].empty();
    // A camera that is evidently meant to be an RGB camera gets a reason when it cannot be treated as one
    const bool expected_rgb = camera_type == "rgb" || Nbands == 3 || all_colours_named;

    // The spectral class of each band with a spectral response, from the response-weighted mean wavelength of the response
    std::vector<int> band_class(Nbands, -1);
    std::vector<float> band_mean_wavelength(Nbands, 0.f);
    bool all_bands_have_responses = true;
    for (size_t b = 0; b < Nbands; b++) {
        if (!cameraBandHasSpectralResponse(camera, bands[b])) {
            all_bands_have_responses = false;
            continue;
        }
        const std::vector<helios::vec2> response = loadSpectralData(camera.band_spectral_response.at(bands[b]));
        double weighted_wavelength = 0;
        double total = 0;
        for (size_t i = 1; i < response.size(); i++) {
            const double width = response[i].x - response[i - 1].x;
            total += 0.5 * width * (response[i].y + response[i - 1].y);
            weighted_wavelength += 0.5 * width * (response[i].y * response[i].x + response[i - 1].y * response[i - 1].x);
        }
        if (!(total > 0)) {
            all_bands_have_responses = false; // a response that is zero everywhere is rejected when the radiative properties are updated
            continue;
        }
        band_mean_wavelength[b] = float(weighted_wavelength / total);
        band_class[b] = spectralClass(band_mean_wavelength[b]);
    }

    // Bands named red, green and blue whose responses are red, green and blue in a different order are a mistake in setting up the camera
    if (all_colours_named) {
        int index_of[3];
        for (int c = 0; c < 3; c++) {
            index_of[c] = int(std::find(bands.begin(), bands.end(), named_band[c]) - bands.begin());
        }
        const int class_red = band_class[index_of[0]], class_green = band_class[index_of[1]], class_blue = band_class[index_of[2]];
        const bool distinct_visible = class_red >= 0 && class_red < 3 && class_green >= 0 && class_green < 3 && class_blue >= 0 && class_blue < 3 && class_red != class_green && class_red != class_blue &&
                                      class_green != class_blue;
        if (distinct_visible && (class_red != 0 || class_green != 1 || class_blue != 2)) {
            std::string mismatches;
            for (int c = 0; c < 3; c++) {
                const int b = index_of[c];
                if (band_class[b] != c) {
                    mismatches += (mismatches.empty() ? "" : "; ") + std::string("band '") + bands[b] + "' has the spectral response '" + camera.band_spectral_response.at(bands[b]) + "', centred at " +
                                  std::to_string(int(std::round(band_mean_wavelength[b]))) + " nm, which is " + spectralClassName(band_class[b]);
                }
            }
            helios_runtime_error("ERROR (RadiationModel::cameraRGBBands): The red, green and blue bands of camera '" + camera.label + "' have red, green and blue spectral responses, but not in the bands named for them: " + mismatches +
                                 ". Check the responses given to setCameraSpectralResponse() for this camera.");
        }
    }

    if (all_bands_have_responses) {
        // Each band's colour comes from its response. Where several bands have the same colour (e.g. a multispectral camera's two green bands), the one named for that colour is taken.
        std::vector<std::string> rgb(3);
        std::string problems;
        for (int c = 0; c < 3; c++) {
            std::vector<std::string> candidates;
            for (size_t b = 0; b < Nbands; b++) {
                if (band_class[b] == c) {
                    candidates.push_back(bands[b]);
                }
            }
            if (candidates.size() == 1) {
                rgb[c] = candidates.front();
            } else if (candidates.size() > 1 && !named_band[c].empty() && std::find(candidates.begin(), candidates.end(), named_band[c]) != candidates.end()) {
                rgb[c] = named_band[c];
            } else if (candidates.empty()) {
                problems += std::string(problems.empty() ? "" : "; ") + "no band responds to " + spectralClassName(c);
            } else {
                std::string list;
                for (const std::string &candidate: candidates) {
                    list += (list.empty() ? "'" : ", '") + candidate + "'";
                }
                problems += std::string(problems.empty() ? "" : "; ") + "bands " + list + " all respond to " + spectralClassName(c) + ", and none of them is named " + spectralClassName(c);
            }
        }
        if (problems.empty()) {
            return rgb;
        }
        if (expected_rgb) {
            std::string centres;
            for (size_t b = 0; b < Nbands; b++) {
                centres += (centres.empty() ? "" : ", ") + std::string("'") + bands[b] + "' at " + std::to_string(int(std::round(band_mean_wavelength[b]))) + " nm (" + spectralClassName(band_class[b]) + ")";
            }
            not_rgb_reason = "Camera '" + camera.label + "' is not an RGB camera: judged by the response-weighted mean wavelengths of its spectral responses (" + centres + "), " + problems + ".";
        }
        return {};
    }

    // Without a spectral response for every band, the bands named red, green and blue are taken, or else the bands of a three-band camera in order
    if (all_colours_named) {
        return {named_band[0], named_band[1], named_band[2]};
    } else if (Nbands == 3) {
        return bands;
    }
    if (expected_rgb) {
        not_rgb_reason = "Camera '" + camera.label + "' has " + std::to_string(Nbands) +
                         " bands, which are not named red, green and blue and do not all have spectral responses, so its red, green and blue bands cannot be identified. Name them \"red\", \"green\" and \"blue\", or give every band a spectral response with setCameraSpectralResponse().";
    }
    return {};
}

namespace {
    //! A tabulated spectrum at a wavelength, linearly interpolated and zero outside its range
    float spectrumAt(const std::vector<helios::vec2> &spectrum, float wavelength) {
        if (spectrum.empty() || wavelength < spectrum.front().x || wavelength > spectrum.back().x) {
            return 0.f;
        }
        const auto upper = std::lower_bound(spectrum.begin(), spectrum.end(), wavelength, [](const helios::vec2 &point, float w) { return point.x < w; });
        if (upper == spectrum.begin()) {
            return upper->y;
        }
        const auto lower = upper - 1;
        return lower->y + (wavelength - lower->x) / (upper->x - lower->x) * (upper->y - lower->y);
    }
} // namespace

const RadiationModel::ColorCorrectionTrainingData &RadiationModel::colorCorrectionTrainingData() const {
    if (color_correction_training) {
        return *color_correction_training;
    }

    // The spectra are read into a Context of their own, so that the user's Context does not fill up with the libraries' hundreds of spectra
    helios::Context library;
    library.loadXML(helios::resolvePluginAsset("radiation", "spectral_data/cie_colorimetry.xml").string().c_str(), true);
    for (const char *file: {"color_board/Calibrite_ColorChecker_Classic_colorboard.xml", "color_board/DGK_DKK_colorboard.xml", "color_board/Datacolor_SpyderCHECKR_24_colorboard.xml", "leaf_surface_spectral_library.xml",
                            "soil_surface_spectral_library.xml", "bark_surface_spectral_library.xml", "fruit_surface_spectral_library.xml"}) {
        library.loadXML(helios::resolvePluginAsset("radiation", std::string("spectral_data/") + file).string().c_str(), true);
    }

    auto training = std::make_unique<ColorCorrectionTrainingData>();
    for (int wavelength = 300; wavelength <= 830; wavelength++) {
        training->wavelengths.push_back(float(wavelength));
    }
    const auto resample = [&](const std::string &label) {
        std::vector<helios::vec2> spectrum;
        library.getGlobalData(label.c_str(), spectrum);
        std::vector<float> values;
        values.reserve(training->wavelengths.size());
        for (float wavelength: training->wavelengths) {
            values.push_back(spectrumAt(spectrum, wavelength));
        }
        return values;
    };
    training->illuminant = resample("CIE_D65");
    training->xbar = resample("CIE1931_2deg_xbar");
    training->ybar = resample("CIE1931_2deg_ybar");
    training->zbar = resample("CIE1931_2deg_zbar");
    for (const std::string &label: library.listGlobalData()) {
        // Only reflectance spectra: the colour boards' patches and the libraries' "*reflectivity*" spectra. The leaf library's transmittance spectra are not the colour of a surface the camera sees.
        const bool is_reflectance = label.rfind("ColorReference_", 0) == 0 || label.find("reflectivity") != std::string::npos;
        if (is_reflectance && library.getGlobalDataType(label.c_str()) == helios::HELIOS_TYPE_VEC2) {
            training->reflectances.push_back(resample(label));
        }
    }
    if (training->reflectances.size() < 3) {
        helios_runtime_error("ERROR (RadiationModel::colorCorrectionTrainingData): Found only " + std::to_string(training->reflectances.size()) +
                             " reflectance spectra in the radiation plug-in's spectral libraries; at least 3 are needed to fit a colour correction matrix. The plug-in's spectral_data folder appears incomplete.");
    }
    color_correction_training = std::move(training);
    return *color_correction_training;
}

std::vector<float> RadiationModel::computeColorCorrectionMatrix(const RadiationCamera &camera, std::string &undefined_reason) const {
    std::string not_rgb_reason;
    const std::vector<std::string> rgb_bands = cameraRGBBands(camera, not_rgb_reason);
    if (rgb_bands.empty()) {
        undefined_reason = not_rgb_reason.empty() ? "Camera '" + camera.label + "' is not an RGB camera, so it has no colour correction matrix." : not_rgb_reason + " It has no colour correction matrix.";
        return {};
    }
    const ColorCorrectionTrainingData &training = colorCorrectionTrainingData();
    const size_t Nwavelengths = training.wavelengths.size();

    std::vector<std::vector<float>> response(3);
    for (int k = 0; k < 3; k++) {
        const auto response_label = camera.band_spectral_response.find(rgb_bands[k]);
        if (!cameraBandHasSpectralResponse(camera, rgb_bands[k])) {
            undefined_reason = "Camera '" + camera.label + "' has no spectral response for band '" + rgb_bands[k] +
                               "'. Its colour correction matrix is derived from the spectral responses of its red, green and blue bands; set them with setCameraSpectralResponse().";
            return {};
        }
        const std::vector<helios::vec2> spectrum = loadSpectralData(response_label->second);
        for (float wavelength: training.wavelengths) {
            response[k].push_back(spectrumAt(spectrum, wavelength));
        }
    }

    // What the camera records from a perfect white, by which each channel is white balanced, and the luminance of that white
    double white[3] = {0, 0, 0};
    double white_luminance = 0;
    for (size_t i = 0; i < Nwavelengths; i++) {
        for (int k = 0; k < 3; k++) {
            white[k] += double(training.illuminant[i]) * response[k][i];
        }
        white_luminance += double(training.illuminant[i]) * training.ybar[i];
    }
    for (int k = 0; k < 3; k++) {
        if (white[k] <= 0) {
            undefined_reason =
                    "The spectral response of camera '" + camera.label + "' in band '" + rgb_bands[k] + "' is zero over the visible range (300-830 nm), so the band records no colour and no colour correction matrix can be derived for the camera.";
            return {};
        }
    }

    // Least squares for the matrix M taking the camera's white-balanced RGB c of each training reflectance to its colorimetric linear sRGB t, with
    // every row of M summing to 1 so that white stays white. Substituting M[k][2] = 1 - M[k][0] - M[k][1] leaves, for each row k, the two unknowns
    // (M[k][0], M[k][1]) in the normal equations of (c0 - c2, c1 - c2) . m = t_k - c2.
    static constexpr double XYZ_to_linear_sRGB[3][3] = {{3.2404542, -1.5371385, -0.4985314}, {-0.9692660, 1.8760108, 0.0415560}, {0.0556434, -0.2040259, 1.0572252}};
    double AtA[2][2] = {{0, 0}, {0, 0}};
    double Atb[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    for (const std::vector<float> &reflectance: training.reflectances) {
        double c[3] = {0, 0, 0};
        double XYZ[3] = {0, 0, 0};
        for (size_t i = 0; i < Nwavelengths; i++) {
            const double light = double(reflectance[i]) * training.illuminant[i];
            for (int k = 0; k < 3; k++) {
                c[k] += light * response[k][i];
            }
            XYZ[0] += light * training.xbar[i];
            XYZ[1] += light * training.ybar[i];
            XYZ[2] += light * training.zbar[i];
        }
        for (int k = 0; k < 3; k++) {
            c[k] /= white[k];
            XYZ[k] /= white_luminance;
        }
        const double a[2] = {c[0] - c[2], c[1] - c[2]};
        AtA[0][0] += a[0] * a[0];
        AtA[0][1] += a[0] * a[1];
        AtA[1][1] += a[1] * a[1];
        for (int k = 0; k < 3; k++) {
            const double t = XYZ_to_linear_sRGB[k][0] * XYZ[0] + XYZ_to_linear_sRGB[k][1] * XYZ[1] + XYZ_to_linear_sRGB[k][2] * XYZ[2];
            Atb[k][0] += a[0] * (t - c[2]);
            Atb[k][1] += a[1] * (t - c[2]);
        }
    }
    // The determinant relative to the product of the diagonal is 1 - r^2, with r the correlation of the two fitted colour differences: 0 when the
    // camera's channels do not distinguish colours (e.g. the same response in every band), in which case the matrix is undefined, and far from 0
    // for a real colour camera (0.3-0.6 for the cameras in the library).
    const double determinant = AtA[0][0] * AtA[1][1] - AtA[0][1] * AtA[0][1];
    if (!(determinant > 1e-4 * AtA[0][0] * AtA[1][1])) {
        undefined_reason = "The red, green and blue spectral responses of camera '" + camera.label + "' do not distinguish colours (e.g. the same response in every band), so no colour correction matrix can be derived for it.";
        return {};
    }
    std::vector<float> matrix(9);
    for (int k = 0; k < 3; k++) {
        const double m0 = (AtA[1][1] * Atb[k][0] - AtA[0][1] * Atb[k][1]) / determinant;
        const double m1 = (AtA[0][0] * Atb[k][1] - AtA[0][1] * Atb[k][0]) / determinant;
        matrix[3 * k] = float(m0);
        matrix[3 * k + 1] = float(m1);
        matrix[3 * k + 2] = float(1.0 - m0 - m1);
    }
    return matrix;
}

helios::vec3 RadiationModel::colorimetricLinearSRGB(const std::vector<helios::vec2> &reflectance) const {
    const ColorCorrectionTrainingData &training = colorCorrectionTrainingData();
    double XYZ[3] = {0, 0, 0};
    double white_luminance = 0;
    for (size_t i = 0; i < training.wavelengths.size(); i++) {
        const double light = double(spectrumAt(reflectance, training.wavelengths[i])) * training.illuminant[i];
        XYZ[0] += light * training.xbar[i];
        XYZ[1] += light * training.ybar[i];
        XYZ[2] += light * training.zbar[i];
        white_luminance += double(training.illuminant[i]) * training.ybar[i];
    }
    for (double &component: XYZ) {
        component /= white_luminance;
    }
    return make_vec3(float(3.2404542 * XYZ[0] - 1.5371385 * XYZ[1] - 0.4985314 * XYZ[2]), float(-0.9692660 * XYZ[0] + 1.8760108 * XYZ[1] + 0.0415560 * XYZ[2]),
                     float(0.0556434 * XYZ[0] - 0.2040259 * XYZ[1] + 1.0572252 * XYZ[2]));
}

void RadiationModel::applyAutomaticColorCorrection(RadiationCamera &camera, const std::vector<std::string> &rgb_bands, const std::string &not_rgb_reason) const {
    // Called by runBand() once the camera's image is complete; the matrix is recorded afresh for each image
    camera.applied_color_correction_matrix.clear();
    if (camera.color_correction == "off") {
        return;
    }
    // A camera that looks like an RGB camera but whose bands are not red, green and blue (e.g. a colour-infrared camera) is reported
    if (rgb_bands.empty()) {
        if (!not_rgb_reason.empty() && message_flag) {
            std::cerr << "WARNING (RadiationModel::runBand): The images of camera '" << camera.label << "' were not colour corrected or white balanced, and were exposed band by band. " << not_rgb_reason
                      << " Set CameraProperties::color_correction to \"off\" for this camera to silence this warning." << std::endl;
        }
        return;
    }
    // Only an RGB camera with a spectral response in each of its red, green and blue bands has a colour correction to apply
    if (!cameraBandHasSpectralResponse(camera, rgb_bands[0]) || !cameraBandHasSpectralResponse(camera, rgb_bands[1]) || !cameraBandHasSpectralResponse(camera, rgb_bands[2])) {
        return;
    }

    std::string undefined_reason;
    const std::vector<float> matrix = computeColorCorrectionMatrix(camera, undefined_reason);
    if (matrix.empty()) {
        if (message_flag) {
            std::cerr << "WARNING (RadiationModel::runBand): The images of camera '" << camera.label << "' were not colour corrected. " << undefined_reason
                      << " Set CameraProperties::color_correction to \"off\" for this camera to turn the correction off." << std::endl;
        }
        return;
    }
    std::vector<float> &red = camera.pixel_data.at(rgb_bands[0]);
    std::vector<float> &green = camera.pixel_data.at(rgb_bands[1]);
    std::vector<float> &blue = camera.pixel_data.at(rgb_bands[2]);
    for (size_t p = 0; p < red.size(); p++) {
        const float r = red[p];
        const float g = green[p];
        const float b = blue[p];
        red[p] = matrix[0] * r + matrix[1] * g + matrix[2] * b;
        green[p] = matrix[3] * r + matrix[4] * g + matrix[5] * b;
        blue[p] = matrix[6] * r + matrix[7] * g + matrix[8] * b;
    }
    camera.applied_color_correction_matrix = matrix;
}

void RadiationModel::requireCompleteCameraImage(const std::string &camera_label, const std::string &caller) const {
    const RadiationCamera &camera = cameras.at(camera_label);
    if (!camera.image_processed || camera.bands_rendered_since_processing.empty()) {
        return;
    }
    std::string rendered, not_rendered;
    for (const std::string &band: camera.band_labels) {
        std::string &list = camera.bands_rendered_since_processing.count(band) > 0 ? rendered : not_rendered;
        list += (list.empty() ? "'" : ", '") + band + "'";
    }
    helios_runtime_error("ERROR (RadiationModel::" + caller + "): The image of camera '" + camera_label + "' is incomplete: band(s) " + rendered + " were rendered again since the image was last processed, but band(s) " +
                         not_rendered + " were not. Their pixel data were already white balanced, colour corrected and exposed, so the image would mix processed and unprocessed bands. Render the remaining band(s) with runBand() "
                         "(the image is processed once every band has been rendered), or render all of the camera's bands together.");
}

std::vector<float> RadiationModel::getCameraPixelData(const std::string &camera_label, const std::string &band_label) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraPixelData): Camera '" + camera_label + "' does not exist.");
    }

    auto &camera_pixel_data = cameras.at(camera_label).pixel_data;
    if (camera_pixel_data.find(band_label) == camera_pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraPixelData): Band '" + band_label + "' does not exist in camera '" + camera_label + "'.");
    }
    requireCompleteCameraImage(camera_label, "getCameraPixelData");

    return camera_pixel_data.at(band_label);
}

std::vector<float> RadiationModel::getCameraPixelBrightnessTemperature(const std::string &camera_label, const std::string &band_label) {
    const std::string caller = "ERROR (RadiationModel::getCameraPixelBrightnessTemperature): ";
    const float stefan_boltzmann = 5.670374419e-8f;
    // Temperature step of the table of band radiance against temperature that is inverted; linear interpolation in it is exact to well under 1e-3 K
    const float table_step_K = 0.01f;
    // Brightness temperatures are sought within this range
    const float max_brightness_temperature = 1e5f;

    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error(caller + "Camera '" + camera_label + "' does not exist.");
    }
    const RadiationCamera &camera = cameras.at(camera_label);
    if (camera.pixel_data.find(band_label) == camera.pixel_data.end()) {
        helios_runtime_error(caller + "Band '" + band_label + "' does not exist in camera '" + camera_label + "', or the camera has not been rendered in it with runBand().");
    }
    requireCompleteCameraImage(camera_label, "getCameraPixelBrightnessTemperature");
    const RadiationBand &band = radiation_bands.at(band_label);
    if (!band.emissionFlag || sif_emission_bands.count(band_label) > 0) {
        helios_runtime_error(caller + "Band '" + band_label + "' of camera '" + camera_label + "' is not a thermal emission band, so its pixels have no brightness temperature. Brightness temperatures are defined for bands with emission enabled (RadiationModel::enableEmission()).");
    }
    const bool exposure_unscaled = camera.exposure == "manual" || (camera.exposure == "auto" && effectiveCameraType(camera) == "thermal");
    if (!exposure_unscaled) {
        helios_runtime_error(caller + "The pixels of camera '" + camera_label + "' were scaled by its exposure setting ('" + camera.exposure +
                             "'), so they are not radiances and have no brightness temperature. Automatic exposure scales every camera that is not a thermal camera (a library camera of type \"thermal\", or a camera whose bands all have emission enabled and wavelength bounds starting at 3000 nm or above), and ISO exposure "
                             "scales every camera. Set CameraProperties::exposure to \"manual\" for this camera.");
    }

    // Radiance of a blackbody at temperature T as the pixels hold it
    const bool through_atmosphere = !camera.atmosphere_label.empty();
    std::vector<float> grid_wavelengths;
    std::vector<float> grid_weights;
    if (through_atmosphere) {
        // Above the atmosphere: the Planck spectrum weighted by the camera's spectral response, integrated as the atmosphere conversion integrates it
        bandIntegrationGrid(band.wavebandBounds, {}, loadCameraBandResponse(camera, band_label), grid_wavelengths, grid_weights);
    }
    auto band_radiance = [&](float temperature) {
        if (temperature <= 0.f) {
            return 0.f;
        }
        if (through_atmosphere) {
            float radiance = 0.f;
            for (size_t i = 0; i < grid_wavelengths.size(); i++) {
                radiance += grid_weights.at(i) * blackbodySpectralRadiance(grid_wavelengths.at(i), temperature);
            }
            return radiance;
        }
        const float band_fraction = band.hasWavebandBounds() ? blackbodyBandFraction(band.wavebandBounds.x, band.wavebandBounds.y, temperature) : 1.f;
        return stefan_boltzmann * powf(temperature, 4) * band_fraction / float(M_PI);
    };

    const std::vector<float> &pixels = camera.pixel_data.at(band_label);
    float max_pixel = 0.f;
    float min_positive_pixel = std::numeric_limits<float>::max();
    for (size_t i = 0; i < pixels.size(); i++) {
        if (!std::isfinite(pixels.at(i))) {
            helios_runtime_error(caller + "Pixel " + std::to_string(i) + " of band '" + band_label + "' of camera '" + camera_label + "' is not finite (" + std::to_string(pixels.at(i)) + ").");
        }
        if (pixels.at(i) < 0.f) {
            helios_runtime_error(caller + "Pixel " + std::to_string(i) + " of band '" + band_label + "' of camera '" + camera_label + "' has a negative radiance (" + std::to_string(pixels.at(i)) + " W/m^2/sr), which no temperature emits.");
        }
        max_pixel = std::max(max_pixel, pixels.at(i));
        if (pixels.at(i) > 0.f) {
            min_positive_pixel = std::min(min_positive_pixel, pixels.at(i));
        }
    }
    std::vector<float> brightness_temperature(pixels.size(), 0.f);
    if (max_pixel == 0.f) {
        return brightness_temperature;
    }

    // A band without wavelength bounds seen without an atmosphere holds sigma*T^4/pi, which is inverted directly
    if (!through_atmosphere && !band.hasWavebandBounds()) {
        for (size_t i = 0; i < pixels.size(); i++) {
            brightness_temperature.at(i) = powf(float(M_PI) * pixels.at(i) / stefan_boltzmann, 0.25f);
        }
        return brightness_temperature;
    }

    // Otherwise the band radiance is tabulated against temperature over a range that holds every pixel, and inverted by linear interpolation
    float upper_temperature = 400.f;
    while (band_radiance(upper_temperature) < max_pixel) {
        upper_temperature *= 2.f;
        if (upper_temperature > max_brightness_temperature) {
            helios_runtime_error(caller + "A pixel of band '" + band_label + "' of camera '" + camera_label + "' has a radiance of " + std::to_string(max_pixel) + " W/m^2/sr, more than a blackbody emits in the band below " +
                                 std::to_string(max_brightness_temperature) + " K.");
        }
    }
    float lower_temperature = 150.f;
    while (lower_temperature > 1.f && band_radiance(lower_temperature) > min_positive_pixel) {
        lower_temperature *= 0.5f;
    }
    if (band_radiance(lower_temperature) > min_positive_pixel) {
        lower_temperature = 0.f;
    }
    const size_t Ntable = size_t(std::ceil((upper_temperature - lower_temperature) / table_step_K)) + 1;
    std::vector<float> table_temperature(Ntable);
    std::vector<float> table_radiance(Ntable);
    for (size_t k = 0; k < Ntable; k++) {
        table_temperature.at(k) = lower_temperature + (upper_temperature - lower_temperature) * float(k) / float(Ntable - 1);
        table_radiance.at(k) = band_radiance(table_temperature.at(k));
    }
    for (size_t i = 0; i < pixels.size(); i++) {
        if (pixels.at(i) == 0.f) {
            continue;
        }
        const size_t upper = std::distance(table_radiance.begin(), std::lower_bound(table_radiance.begin(), table_radiance.end(), pixels.at(i)));
        const size_t lower = upper == 0 ? 0 : upper - 1;
        const float weight = upper == lower ? 0.f : (pixels.at(i) - table_radiance.at(lower)) / (table_radiance.at(upper) - table_radiance.at(lower));
        brightness_temperature.at(i) = table_temperature.at(lower) + weight * (table_temperature.at(upper) - table_temperature.at(lower));
    }
    return brightness_temperature;
}

void RadiationModel::setCameraPixelData(const std::string &camera_label, const std::string &band_label, const std::vector<float> &pixel_data) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::setCameraPixelData): Camera '" + camera_label + "' does not exist.");
    }

    cameras.at(camera_label).pixel_data[band_label] = pixel_data;
}

// ========== Backend Integration Methods ==========

void RadiationModel::queryBackendGPUMemory() const {
    if (backend) {
        backend->queryGPUMemory();
    } else {
        std::cout << "Backend not initialized - cannot query GPU memory." << std::endl;
    }
}


helios::RayTracingLaunchParams RadiationModel::buildCameraLaunchParams(const RadiationCamera &camera, uint camera_id, uint antialiasing_samples, const helios::int2 &tile_resolution, const helios::int2 &tile_offset) {

    helios::RayTracingLaunchParams params;

    // Camera position and orientation
    params.camera_position = camera.position;
    helios::SphericalCoord dir = cart2sphere(camera.lookat - camera.position);

    // A periodic domain repeats laterally, so a camera outside of it sees the same scene as a camera moved into it by a whole number of
    // periods. Wrap the ray origin into the domain, where the periodic boundary faces can wrap its rays.
    if (periodic_flag.x == 1) {
        const float period_x = periodic_xbounds.y - periodic_xbounds.x;
        params.camera_position.x -= period_x * std::floor((params.camera_position.x - periodic_xbounds.x) / period_x);
    }
    if (periodic_flag.y == 1) {
        const float period_y = periodic_ybounds.y - periodic_ybounds.x;
        params.camera_position.y -= period_y * std::floor((params.camera_position.y - periodic_ybounds.x) / period_y);
    }
    params.camera_direction = helios::make_vec2(dir.zenith, dir.azimuth);

    // Camera optical properties
    params.camera_focal_length = camera.focal_length;
    params.camera_lens_diameter = camera.lens_diameter;
    params.camera_fov_aspect = camera.FOV_aspect_ratio;

    // Resolution and tiling
    params.camera_resolution = tile_resolution;
    params.camera_resolution_full = camera.resolution;
    params.camera_pixel_offset = tile_offset;
    params.antialiasing_samples = antialiasing_samples;
    params.camera_id = camera_id;

    // Compute effective HFOV with zoom
    float effective_HFOV = camera.HFOV_degrees / camera.camera_zoom;
    params.camera_viewplane_length = 0.5f / tanf(0.5f * effective_HFOV * M_PI / 180.f);

    // Explicitly set scattering iteration for cameras (always iteration 0 for specular)
    params.scattering_iteration = 0;

    // Set specular reflection mode from auto-detection
    params.specular_reflection_enabled = specular_reflection_mode;
    params.below_horizon_diffuse_enabled = below_horizon_diffuse_enabled ? 1 : 0;

    return params;
}

std::vector<CameraTile> RadiationModel::computeCameraTiles(const RadiationCamera &camera, size_t maxRays) {

    std::vector<CameraTile> tiles;

    size_t total_rays = size_t(camera.antialiasing_samples) * size_t(camera.resolution.x) * size_t(camera.resolution.y);

    // No tiling needed
    if (total_rays <= maxRays) {
        tiles.push_back({camera.resolution, helios::make_int2(0, 0)});
        return tiles;
    }

    // Calculate tile dimensions
    size_t rays_per_row = size_t(camera.antialiasing_samples) * size_t(camera.resolution.x);
    size_t max_rows_per_tile = floor(float(maxRays) / float(rays_per_row));

    if (max_rows_per_tile == 0) {
        // 2D tiling - even one row is too large
        size_t max_pixels_per_tile = floor(float(maxRays) / float(camera.antialiasing_samples));

        float aspect = float(camera.resolution.x) / float(camera.resolution.y);
        size_t tile_width = round(sqrt(max_pixels_per_tile * aspect));
        size_t tile_height = floor(float(max_pixels_per_tile) / float(tile_width));

        tile_width = std::min(tile_width, size_t(camera.resolution.x));
        tile_height = std::min(tile_height, size_t(camera.resolution.y));

        int Ntiles_x = ceil(float(camera.resolution.x) / float(tile_width));
        int Ntiles_y = ceil(float(camera.resolution.y) / float(tile_height));

        for (int ty = 0; ty < Ntiles_y; ty++) {
            for (int tx = 0; tx < Ntiles_x; tx++) {
                size_t offset_x = tx * tile_width;
                size_t offset_y = ty * tile_height;
                size_t width_this = std::min(tile_width, camera.resolution.x - offset_x);
                size_t height_this = std::min(tile_height, camera.resolution.y - offset_y);

                tiles.push_back({helios::make_int2(width_this, height_this), helios::make_int2(offset_x, offset_y)});
            }
        }
    } else {
        // 1D tiling - tile along height only
        size_t rows_per_tile = std::min(max_rows_per_tile, size_t(camera.resolution.y));
        int Ntiles = ceil(float(camera.resolution.y) / float(rows_per_tile));

        for (int t = 0; t < Ntiles; t++) {
            size_t offset_y = t * rows_per_tile;
            size_t height_this = std::min(rows_per_tile, camera.resolution.y - offset_y);

            tiles.push_back({helios::make_int2(camera.resolution.x, height_this), helios::make_int2(0, offset_y)});
        }
    }

    return tiles;
}

void RadiationModel::buildGeometryData(const std::vector<uint> &UUIDs) {
    // Build backend-agnostic geometry data from Context primitives
    // This extracts all geometry information needed by the ray tracing backend

    // Filter out invalid/zero-area primitives (same as old updateGeometry). The areas are kept, indexed by UUID, so the
    // per-primitive pass below does not look them up in the Context a second time.
    std::vector<uint> valid_UUIDs;
    std::unordered_map<uint, float> valid_area;
    valid_area.reserve(UUIDs.size());
    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID))
            continue;

        if (context->getPrimitiveType(UUID) == helios::PRIMITIVE_TYPE_VOXEL) {
            helios_runtime_error("ERROR (RadiationModel::updateGeometry): Primitive UUID " + std::to_string(UUID) +
                                 " is a voxel, which the radiation model cannot trace. Delete the voxels from the Context, or pass updateGeometry() a list of UUIDs that leaves them out.");
        }

        float area = context->getPrimitiveArea(UUID);
        uint parentID = context->getPrimitiveParentObjectID(UUID);
        if ((area == 0 || std::isnan(area)) && context->getObjectType(parentID) != helios::OBJECT_TYPE_TILE) {
            continue;
        }
        valid_UUIDs.push_back(UUID);
        valid_area.emplace(UUID, area);
    }

    if (valid_UUIDs.empty()) {
        geometry_data = helios::RayTracingGeometry(); // Empty geometry
        return;
    }

    // Reorder primitives by parent object (same ordering as old code)
    std::vector<uint> objID_all = context->getUniquePrimitiveParentObjectIDs(valid_UUIDs, true);
    std::vector<uint> primitive_UUIDs_ordered;
    std::vector<float> primitive_areas_ordered;

    for (uint objID: objID_all) {
        std::vector<uint> prim_UUIDs = context->getObjectPrimitiveUUIDs(objID);
        if (objID == 0) {
            // Standalone primitives (parentID=0) come from unordered_map iteration,
            // which has non-deterministic ordering. Sort by UUID for reproducibility.
            std::sort(prim_UUIDs.begin(), prim_UUIDs.end());
        }
        for (uint UUID: prim_UUIDs) {
            if (!context->doesPrimitiveExist(UUID)) {
                continue;
            }
            auto valid = valid_area.find(UUID);
            if (valid != valid_area.end()) {
                primitive_UUIDs_ordered.push_back(UUID);
                primitive_areas_ordered.push_back(valid->second);
            }
        }
    }

    size_t Nprimitives = primitive_UUIDs_ordered.size();
    geometry_data.primitive_count = Nprimitives;

    // Clear and allocate per-primitive arrays (important when updateGeometry is called multiple times)
    geometry_data.transform_matrices.clear();
    geometry_data.transform_matrices.resize(Nprimitives * 16);
    geometry_data.primitive_types.clear();
    // Initialize to UINT_MAX as sentinel - prevents uninitialized entries from matching type==0 (patch)
    geometry_data.primitive_types.resize(Nprimitives, UINT_MAX);
    geometry_data.primitive_UUIDs = primitive_UUIDs_ordered;
    geometry_data.primitive_IDs.clear();
    geometry_data.primitive_IDs.resize(Nprimitives); // Will be populated after primitiveID_indices is built
    geometry_data.object_IDs.clear();
    geometry_data.object_IDs.resize(Nprimitives);
    geometry_data.object_subdivisions.clear();
    geometry_data.object_subdivisions.resize(Nprimitives);
    geometry_data.twosided_flags.clear();
    geometry_data.twosided_flags.resize(Nprimitives);
    geometry_data.solid_fractions.clear();
    geometry_data.solid_fractions.resize(Nprimitives);
    geometry_data.primitive_areas = std::move(primitive_areas_ordered);

    // Clear type-specific arrays
    geometry_data.patches.vertices.clear();
    geometry_data.patches.UUIDs.clear();
    geometry_data.triangles.vertices.clear();
    geometry_data.triangles.UUIDs.clear();
    geometry_data.disk_centers.clear();
    geometry_data.disk_radii.clear();
    geometry_data.disk_normals.clear();
    geometry_data.disk_UUIDs.clear();
    geometry_data.tiles.vertices.clear();
    geometry_data.tiles.UUIDs.clear();
    geometry_data.bboxes.vertices.clear();
    geometry_data.bboxes.UUIDs.clear();

    // Track object IDs for compound objects
    uint current_objID = 0;
    uint last_parentID = 99999;

    std::vector<uint> primitiveID_indices; // Maps primitives to their "object" index

    for (size_t u = 0; u < Nprimitives; u++) {
        uint UUID = primitive_UUIDs_ordered[u];
        uint parentID = context->getPrimitiveParentObjectID(UUID);

        if (last_parentID != parentID || parentID == 0 || context->getObjectType(parentID) == helios::OBJECT_TYPE_TILE || context->getObjectType(parentID) == helios::OBJECT_TYPE_ADAPTIVE_TILE) {
            primitiveID_indices.push_back(u);
            last_parentID = parentID;
            current_objID++;
        } else {
            last_parentID = parentID;
        }

        geometry_data.object_IDs[u] = current_objID - 1;
    }

    size_t Nobjects = primitiveID_indices.size();

    // Populate primitiveID for runBand() compatibility
    primitiveID = primitiveID_indices;

    // For backend: primitiveID[position] must return the UUID for that primitive
    // Sized by Nprimitives (all primitives including subpatches), not Nobjects (object entries only)
    std::vector<uint> primitiveID_for_backend(Nprimitives);
    for (size_t i = 0; i < Nprimitives; i++) {
        primitiveID_for_backend[i] = primitive_UUIDs_ordered[i];
    }

    // Copy corrected primitiveID mapping to geometry_data for backend upload
    geometry_data.primitive_IDs = primitiveID_for_backend;

    // Populate geometry for each primitive
    size_t patch_idx = 0, tri_idx = 0, disk_idx = 0, bbox_idx = 0;

    // Iterate over ALL primitives to set per-primitive data
    // (not just Nobjects, which only has one entry per object group)
    for (size_t prim_idx = 0; prim_idx < Nprimitives; prim_idx++) {
        uint UUID = primitive_UUIDs_ordered[prim_idx];

        // Transform matrix
        float m[16];
        uint parentID = context->getPrimitiveParentObjectID(UUID);
        helios::PrimitiveType type = context->getPrimitiveType(UUID);

        // Solid fraction
        geometry_data.solid_fractions[prim_idx] = context->getPrimitiveSolidFraction(UUID);

        // Two-sided flag. Store the raw value (0=one-sided, 1=two-sided, 2=transparent, 3=invisible) rather than collapsing to a
        // boolean, so the backends receive the full semantics. "invisibility_flag" primitive data = 1 is stored as 3 and takes
        // precedence over both the material and primitive-data twosided_flag.
        uint twosided_flag = context->getPrimitiveTwosidedFlag(UUID, 1);
        if (context->doesPrimitiveDataExist(UUID, "invisibility_flag")) {
            uint invisibility_flag;
            context->getPrimitiveData(UUID, "invisibility_flag", invisibility_flag);
            if (invisibility_flag == 1) {
                twosided_flag = 3;
            } else if (invisibility_flag != 0) {
                helios_runtime_error("ERROR (RadiationModel::buildGeometryData): Primitive " + std::to_string(UUID) + " has \"invisibility_flag\" primitive data set to " + std::to_string(invisibility_flag) +
                                     ". Valid values are 0 (visible) or 1 (invisible to the radiation model).");
            }
        }
        geometry_data.twosided_flags[prim_idx] = static_cast<char>(twosided_flag);

        if (parentID > 0 && context->getObjectType(parentID) == helios::OBJECT_TYPE_TILE) {
            // Tile subpatch: treat as individual patch for both OptiX and Vulkan backends.
            // Each subpatch gets its own world-space vertices in the patch geometry,
            // its own transform matrix, and type=0 (patch). tile_count will be 0.
            geometry_data.primitive_types[prim_idx] = 0; // patch

            context->getPrimitiveTransformationMatrix(UUID, m);
            memcpy(&geometry_data.transform_matrices[prim_idx * 16], m, 16 * sizeof(float));

            std::vector<vec3> verts = context->getPrimitiveVertices(UUID);
            for (const auto &v: verts) {
                geometry_data.patches.vertices.push_back(v);
            }

            geometry_data.object_subdivisions[prim_idx] = helios::make_int2(1, 1);
            geometry_data.patches.UUIDs.push_back(UUID);
            patch_idx++;

        } else if (type == helios::PRIMITIVE_TYPE_PATCH) {
            geometry_data.primitive_types[prim_idx] = 0; // patch

            context->getPrimitiveTransformationMatrix(UUID, m);
            memcpy(&geometry_data.transform_matrices[prim_idx * 16], m, 16 * sizeof(float));

            std::vector<vec3> verts = context->getPrimitiveVertices(UUID);
            for (const auto &v: verts) {
                geometry_data.patches.vertices.push_back(v);
            }

            geometry_data.object_subdivisions[prim_idx] = helios::make_int2(1, 1);

            // FIX: Add UUID inline to ensure consistent ordering with vertices
            geometry_data.patches.UUIDs.push_back(UUID);

            patch_idx++;

        } else if (type == helios::PRIMITIVE_TYPE_TRIANGLE) {
            geometry_data.primitive_types[prim_idx] = 1; // triangle

            context->getPrimitiveTransformationMatrix(UUID, m);
            memcpy(&geometry_data.transform_matrices[prim_idx * 16], m, 16 * sizeof(float));

            std::vector<vec3> verts = context->getPrimitiveVertices(UUID);
            for (const auto &v: verts) {
                geometry_data.triangles.vertices.push_back(v);
            }

            geometry_data.object_subdivisions[prim_idx] = helios::make_int2(1, 1);
            geometry_data.triangles.UUIDs.push_back(UUID); // Store actual UUID, not position
            tri_idx++;
        }
    }

    // Set counts
    geometry_data.patch_count = patch_idx;
    geometry_data.triangle_count = tri_idx;
    geometry_data.disk_count = disk_idx;
    geometry_data.tile_count = 0; // Tile subpatches are treated as individual patches

    // ========== Periodic Boundary Bboxes ==========
    // Create bbox geometry for periodic boundary conditions
    // Each bbox face is a rectangular boundary at domain edge

    // Get domain bounding box
    vec2 xbounds, ybounds, zbounds;
    context->getDomainBoundingBox(xbounds, ybounds, zbounds);

    // The boundary faces span only the height of the geometry, not that of a camera above or below it: every ray would otherwise
    // have to be wrapped through the empty layer up to the camera. A camera ray starting outside the faces' height range is
    // instead moved to the top or bottom of the faces in one step by the ray-generation programs, which start it exactly at that
    // height. The vertical margin keeps the geometry far enough from that starting point to be hit past the rays' minimum
    // distance, also for a ray straight down onto flat geometry, and scales with the coordinates so that it stays above the float
    // spacing far from z = 0.
    const float z_margin = 1e-4f * std::max(1.f, std::max(std::fabs(zbounds.x), std::fabs(zbounds.y)));

    // The lateral faces lie exactly on the edges of the geometry. Rays are wrapped by the distance between opposite faces, so any
    // padding here would leave a gap of twice its width between the periodic copies of the scene, through which rays would pass.
    zbounds.x -= z_margin;
    zbounds.y += z_margin;

    // The rays are wrapped by the width of these bounds, so cameras must be wrapped by the same period
    periodic_xbounds = xbounds;
    periodic_ybounds = ybounds;

    // Bbox UUIDs must not collide with real primitive UUIDs
    // Use max_UUID + 1 as base (not Nprimitives, which can cause collisions with sparse UUIDs)
    uint max_UUID = geometry_data.primitive_UUIDs.empty() ? 0 : *std::max_element(geometry_data.primitive_UUIDs.begin(), geometry_data.primitive_UUIDs.end());
    uint bbox_UUID_base = max_UUID + 1;

    // Create bbox faces based on periodic flags
    if (periodic_flag.x == 1) {
        // -x facing boundary (4 vertices: counter-clockwise from bottom-left)
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.x, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.y, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.y, zbounds.y));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.x, zbounds.y));
        geometry_data.bboxes.UUIDs.push_back(bbox_UUID_base + bbox_idx);
        bbox_idx++;

        // +x facing boundary
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.x, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.y, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.y, zbounds.y));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.x, zbounds.y));
        geometry_data.bboxes.UUIDs.push_back(bbox_UUID_base + bbox_idx);
        bbox_idx++;
    }

    if (periodic_flag.y == 1) {
        // -y facing boundary
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.x, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.x, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.x, zbounds.y));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.x, zbounds.y));
        geometry_data.bboxes.UUIDs.push_back(bbox_UUID_base + bbox_idx);
        bbox_idx++;

        // +y facing boundary
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.y, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.y, zbounds.x));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.y, ybounds.y, zbounds.y));
        geometry_data.bboxes.vertices.push_back(vec3(xbounds.x, ybounds.y, zbounds.y));
        geometry_data.bboxes.UUIDs.push_back(bbox_UUID_base + bbox_idx);
        bbox_idx++;
    }

    // Update bbox count and UUID base
    geometry_data.bbox_count = bbox_idx;
    if (bbox_idx > 0) {
        geometry_data.bbox_UUID_base = bbox_UUID_base;
    } else {
        // No bboxes: set sentinel value so GPU knows all UUIDs are real primitives
        geometry_data.bbox_UUID_base = UINT_MAX;
    }

    // NOTE: Bbox primitive data is NOT included in the shared geometry arrays
    // Bboxes are OptiX-specific constructs for periodic boundaries
    // OptiX backend will build bbox data internally from bbox_count and bbox_UUID_base
    // This keeps the geometry data compatible with non-OptiX backends (Vulkan, etc.)

    // Periodic boundary condition
    geometry_data.periodic_flag = periodic_flag;

    // Extract texture mask and UV data for primitives with transparency textures
    buildTextureData();

    // Build primitive_positions lookup table for GPU UUID→position conversion
    // Size by max UUID to create sparse lookup table (includes bbox UUIDs now that they don't collide)
    // Clear first to remove stale mappings from deleted primitives
    geometry_data.primitive_positions.clear();
    if (!geometry_data.primitive_UUIDs.empty()) {
        uint max_UUID = *std::max_element(geometry_data.primitive_UUIDs.begin(), geometry_data.primitive_UUIDs.end());

        // Expand to include bbox UUIDs if present (they now use max_UUID+1 base, so no collisions)
        uint bbox_max_UUID = max_UUID;
        if (geometry_data.bbox_count > 0) {
            bbox_max_UUID = geometry_data.bbox_UUID_base + geometry_data.bbox_count - 1;
        }

        geometry_data.primitive_positions.resize(bbox_max_UUID + 1, UINT_MAX); // UINT_MAX = invalid/unused

        // Map real primitive UUIDs
        for (size_t i = 0; i < geometry_data.primitive_count; i++) {
            uint UUID = geometry_data.primitive_UUIDs[i];
            geometry_data.primitive_positions[UUID] = i; // Map UUID → array position
        }

        // Map bbox UUIDs to their positions (after real primitives)
        // Now safe because bbox_UUID_base = max_UUID + 1 (no collisions)
        if (geometry_data.bbox_count > 0) {
            for (size_t i = 0; i < geometry_data.bbox_count; i++) {
                uint bbox_UUID = geometry_data.bbox_UUID_base + i;
                geometry_data.primitive_positions[bbox_UUID] = geometry_data.primitive_count + i;
            }
        }
    }
}

void RadiationModel::computeSpectralCameraFlux(const std::vector<float> &spectral_incident, const std::vector<float> &emitted_flux, uint camera, const std::vector<char> &band_launch_flag, std::vector<uint32_t> &class_flux_offset,
                                               std::vector<float> &class_flux) const {
    const size_t Nprimitives = context_UUIDs.size();
    const size_t Nslots = radiation_sources.size() + 1;
    const size_t Nentries = material_data.num_palette_entries;
    const size_t Ncameras = material_data.num_cameras;
    const size_t Nbands_global = band_launch_flag.size();
    std::vector<size_t> global_band;
    for (size_t b_global = 0; b_global < Nbands_global; b_global++) {
        if (band_launch_flag.at(b_global)) {
            global_band.push_back(b_global);
        }
    }
    const size_t Nbands_launch = global_band.size();
    if (spectral_incident.size() != 2 * Nslots * Nprimitives * Nbands_launch) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The radiation reaching primitives with a spectral texture has " + std::to_string(spectral_incident.size()) + " values, but " + std::to_string(2 * Nslots * Nprimitives * Nbands_launch) +
                             " are required.");
    }
    if (emitted_flux.size() != 2 * Nprimitives * Nbands_launch) {
        helios_runtime_error("ERROR (RadiationModel::runBand): The radiation emitted by primitives with a spectral texture has " + std::to_string(emitted_flux.size()) + " values, but " + std::to_string(2 * Nprimitives * Nbands_launch) +
                             " are required.");
    }

    // Each primitive with a spectral texture has one class-flux slot for each class of its palette
    class_flux_offset.assign(Nprimitives, UINT32_MAX);
    std::vector<uint32_t> palette_size(Nprimitives, 0);
    uint32_t class_flux_slots = 0;
    for (size_t u = 0; u < Nprimitives; u++) {
        if (material_data.palette_offset.at(u) == UINT32_MAX) {
            continue;
        }
        palette_size.at(u) = uint32_t(context->getSpectralPalette(context->getPrimitiveSpectralPalette(context_UUIDs.at(u))).size());
        class_flux_offset.at(u) = class_flux_slots;
        class_flux_slots += palette_size.at(u);
    }

    // A class seen on face F sends out what F emits (the same for every class: in a band where the palette applies, only fluorescence),
    // plus what it reflects of the radiation reaching F and transmits of the radiation reaching the other face G, weighted by this camera's
    // response for the slot each arrived in: the camera-weighted scatter a primitive with the class's spectra would accumulate
    CameraMaterialIndexer palette_cam_idx(Nslots, Nentries, Nbands_global, Ncameras);
    auto incident = [&](size_t face, size_t slot, size_t u, size_t b) { return spectral_incident[((face * Nslots + slot) * Nprimitives + u) * Nbands_launch + b]; };
    class_flux.assign(2 * size_t(class_flux_slots) * Nbands_launch, 0.f);
    for (size_t u = 0; u < Nprimitives; u++) {
        for (uint32_t k = 0; k < palette_size.at(u); k++) {
            const size_t entry = material_data.palette_offset.at(u) + k;
            const size_t class_slot = class_flux_offset.at(u) + k;
            for (size_t b = 0; b < Nbands_launch; b++) {
                double top = emitted_flux[u * Nbands_launch + b];
                double bottom = emitted_flux[(Nprimitives + u) * Nbands_launch + b];
                for (size_t slot = 0; slot < Nslots; slot++) {
                    const double rho_cam = material_data.palette_reflectivity_cam.at(palette_cam_idx(slot, entry, global_band.at(b), camera));
                    const double tau_cam = material_data.palette_transmissivity_cam.at(palette_cam_idx(slot, entry, global_band.at(b), camera));
                    top += rho_cam * incident(0, slot, u, b) + tau_cam * incident(1, slot, u, b);
                    bottom += rho_cam * incident(1, slot, u, b) + tau_cam * incident(0, slot, u, b);
                }
                class_flux.at(class_slot * Nbands_launch + b) = float(top);
                class_flux.at((size_t(class_flux_slots) + class_slot) * Nbands_launch + b) = float(bottom);
            }
        }
    }
}


void RadiationModel::buildTextureData() {
    // Extract texture mask and UV data for all primitives with transparency textures, and the class maps of primitives with spectral textures

    size_t Nobjects = geometry_data.primitive_count;

    // Clear any previous texture data (important when updateGeometry is called multiple times)
    geometry_data.mask_data.clear();
    geometry_data.mask_sizes.clear();
    geometry_data.uv_data.clear();
    geometry_data.class_map_data.clear();
    geometry_data.class_map_sizes.clear();

    // Initialize with -1 (no texture)
    geometry_data.mask_IDs.clear();
    geometry_data.mask_IDs.resize(Nobjects, -1);
    geometry_data.uv_IDs.clear();
    geometry_data.uv_IDs.resize(Nobjects, -1);
    geometry_data.class_map_IDs.clear();
    geometry_data.class_map_IDs.resize(Nobjects, -1);
    spectral_class_fractions.assign(Nobjects, {});
    spectral_class_map_files.clear();

    // Cache to avoid duplicate mask data for primitives using the same texture file
    std::map<std::string, int> texture_to_mask_idx;
    // The same for class maps, and for the class fractions of primitives that share a class map and (u,v) coordinates (such as the leaves of one plant)
    std::map<std::string, int> class_map_to_idx;
    std::map<std::pair<int, std::vector<float>>, std::vector<std::pair<uint16_t, float>>> class_fraction_cache;

    for (size_t prim_idx = 0; prim_idx < Nobjects; prim_idx++) {
        uint UUID = geometry_data.primitive_UUIDs[prim_idx];

        const bool has_spectral_texture = context->primitiveHasSpectralTexture(UUID);
        std::string texture_file = context->getPrimitiveTextureFile(UUID);
        const bool has_mask = !texture_file.empty() && context->primitiveTextureHasTransparencyChannel(UUID);

        if (has_mask) {
            // Check cache for existing mask from same texture file
            int mask_idx;
            auto cache_it = texture_to_mask_idx.find(texture_file);
            if (cache_it != texture_to_mask_idx.end()) {
                // Reuse existing mask
                mask_idx = cache_it->second;
            } else {
                // New texture - extract mask data
                const std::vector<std::vector<bool>> *trans_data = context->getPrimitiveTextureTransparencyData(UUID);
                helios::int2 tex_size = context->getPrimitiveTextureSize(UUID);

                mask_idx = static_cast<int>(geometry_data.mask_sizes.size());
                texture_to_mask_idx[texture_file] = mask_idx;

                // Flatten 2D bool array to 1D (row-major: [y][x])
                // Backend expects: for each mask m, iterate [y][x] order
                for (int y = 0; y < tex_size.y; y++) {
                    for (int x = 0; x < tex_size.x; x++) {
                        geometry_data.mask_data.push_back((*trans_data)[y][x]);
                    }
                }
                geometry_data.mask_sizes.push_back(tex_size);
            }

            geometry_data.mask_IDs[prim_idx] = mask_idx;
        }

        if (!has_mask && !has_spectral_texture) {
            continue; // No mask or class map - mask_ID and class_map_ID stay -1
        }

        // Extract UV coordinates for this primitive
        // uv_IDs stores the position index (not offset), used to access uvdata[vertex][position] in CUDA
        std::vector<helios::vec2> uvs = context->getPrimitiveTextureUV(UUID);
        if (uvs.empty() && context->getPrimitiveType(UUID) == helios::PRIMITIVE_TYPE_TRIANGLE) {
            helios_runtime_error("ERROR (RadiationModel::updateGeometry): Triangle UUID " + std::to_string(UUID) + " has a texture with a transparency channel ('" + texture_file +
                                 "') but no texture (u,v) coordinates, so its transparency mask cannot be applied. Add the triangle with Context::addTriangle() giving the texture file and (u,v) coordinates.");
        }
        if (!uvs.empty()) {
            geometry_data.uv_IDs[prim_idx] = static_cast<int>(prim_idx); // Store position index for CUDA 2D buffer access
            for (const auto &uv: uvs) {
                geometry_data.uv_data.push_back(uv);
            }
            // Pad to 4 vertices if needed (CUDA expects max 4 vertices per primitive)
            size_t start_idx = geometry_data.uv_data.size() - uvs.size();
            while (geometry_data.uv_data.size() - start_idx < 4) {
                geometry_data.uv_data.push_back(uvs.back());
            }
        }
        // If uvs is empty, uv_ID stays -1 and CUDA will use default UV mapping

        if (has_spectral_texture) {
            const std::string class_map_file = context->getPrimitiveSpectralTextureFile(UUID);
            helios::int2 class_map_size;
            const std::vector<uint16_t> &classes = context->getSpectralTextureClassMap(class_map_file, class_map_size);
            int class_map_idx;
            auto cache_it = class_map_to_idx.find(class_map_file);
            if (cache_it != class_map_to_idx.end()) {
                class_map_idx = cache_it->second;
            } else {
                class_map_idx = static_cast<int>(geometry_data.class_map_sizes.size());
                class_map_to_idx[class_map_file] = class_map_idx;
                geometry_data.class_map_data.insert(geometry_data.class_map_data.end(), classes.begin(), classes.end());
                geometry_data.class_map_sizes.push_back(class_map_size);
                spectral_class_map_files.push_back(class_map_file);
            }
            geometry_data.class_map_IDs[prim_idx] = class_map_idx;

            // A patch without (u,v) coordinates maps the whole image, as the backends assume
            if (uvs.empty()) {
                uvs = {helios::make_vec2(0, 0), helios::make_vec2(1, 0), helios::make_vec2(1, 1), helios::make_vec2(0, 1)};
            }
            std::vector<float> uv_key;
            for (const helios::vec2 &uv: uvs) {
                uv_key.push_back(uv.x);
                uv_key.push_back(uv.y);
            }
            auto fraction_it = class_fraction_cache.find({class_map_idx, uv_key});
            if (fraction_it == class_fraction_cache.end()) {
                fraction_it = class_fraction_cache.emplace(std::make_pair(class_map_idx, uv_key), helios::classMapAreaFractions(classes, class_map_size, uvs)).first;
            }
            if (fraction_it->second.empty()) {
                helios_runtime_error(
                        "ERROR (RadiationModel::updateGeometry): Primitive UUID " + std::to_string(UUID) + " has a spectral texture whose class map '" + class_map_file +
                        "' has only class 0 (transparent texels) within the primitive's (u,v) coordinates, as the radiation model samples it, so the primitive has no class to take its reflectivity and transmissivity from. Check its (u,v) coordinates, or remove the primitive.");
            }
            spectral_class_fractions[prim_idx] = fraction_it->second;
        }
    }
}

size_t RadiationModel::testBuildGeometryData() {
    buildGeometryData(context->getAllUUIDs());
    return geometry_data.primitive_count;
}

void RadiationModel::buildUUIDMapping() {
    // Build bidirectional UUID ↔ array position mapping
    // This enables efficient conversion between UUID values and array indices

    uuid_to_position.clear();
    position_to_uuid.clear();

    // geometry_data.primitive_UUIDs is already ordered by object
    // Build mapping from this ordered list
    for (size_t i = 0; i < geometry_data.primitive_count; i++) {
        uint UUID = geometry_data.primitive_UUIDs[i];
        uuid_to_position[UUID] = i;
        position_to_uuid.push_back(UUID);
    }

    // Build type-safe mapper (new indexing system)
    // Provides compile-time safety for UUID/position conversions
    geometry_data.mapper.build(geometry_data.primitive_UUIDs);
}


void RadiationModel::buildSourceData() {
    // Build backend-agnostic source data from radiation_sources

    source_data.clear();
    source_data.reserve(radiation_sources.size());

    for (size_t s = 0; s < radiation_sources.size(); s++) {
        const auto &src = radiation_sources[s];
        helios::RayTracingSource backend_src;
        backend_src.position = src.source_position;
        backend_src.rotation = src.source_rotation;
        backend_src.width = src.source_width;
        backend_src.type = src.source_type;
        // The source fluxes are uploaded for each launch (uploadSourceFluxes()), and the camera weights in fluxes_cam are set before each direct launch
        // when the model has cameras, since both depend on the bands launched.

        source_data.push_back(backend_src);
    }
}


helios::RayTracingBackend *RadiationModel::getBackend() {
    return backend.get();
}

helios::RayTracingGeometry &RadiationModel::getGeometryData() {
    return geometry_data;
}

helios::RayTracingMaterial &RadiationModel::getMaterialData() {
    return material_data;
}

std::vector<helios::RayTracingSource> &RadiationModel::getSourceData() {
    return source_data;
}


void RadiationModel::testBuildAllBackendData() {
    buildGeometryData(context->getAllUUIDs());
    updateRadiativeProperties();
    buildSourceData();
}
