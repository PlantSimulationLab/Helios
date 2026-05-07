/** \file "RadiationModel.cpp" Primary source file for radiation transport model.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

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

    minScatterEnergy_default = 0.1;
    scatteringDepth_default = 0;

    rho_default = 0.f;
    tau_default = 0.f;
    eps_default = 1.f;

    kappa_default = 1.f;
    sigmas_default = 0.f;

    temperature_default = 300;

    periodic_flag = make_vec2(0, 0);

    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/camera_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/light_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/soil_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/leaf_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/bark_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/fruit_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/solar_spectrum_ASTMG173.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/DGK_DKK_colorboard.xml").string());

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
    minScatterEnergy_default = 0.1;
    scatteringDepth_default = 0;
    rho_default = 0.f;
    tau_default = 0.f;
    eps_default = 1.f;
    kappa_default = 1.f;
    sigmas_default = 0.f;
    temperature_default = 300;
    periodic_flag = make_vec2(0, 0);

    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/camera_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/light_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/soil_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/leaf_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/bark_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/fruit_surface_spectral_library.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/solar_spectrum_ASTMG173.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml").string());
    spectral_library_files.push_back(helios::resolvePluginAsset("radiation", "spectral_data/color_board/DGK_DKK_colorboard.xml").string());

    // Skip backend creation - will be injected by caller
}

RadiationModel RadiationModel::createWithBackend(helios::Context *context, std::unique_ptr<helios::RayTracingBackend> backend) {
    RadiationModel model(context, true); // Use private constructor, skip backend init

    // Inject the provided backend
    model.backend = std::move(backend);

    return model; // Uses move constructor
}

bool RadiationModel::isGPUBackendAvailable() {
    static bool checked = false;
    static bool available = false;

    if (checked) {
        return available;
    }
    checked = true;

    // Allow forcing unavailability for local CI simulation
    const char *no_gpu = std::getenv("HELIOS_NO_GPU");
    if (no_gpu && std::string(no_gpu) != "0") {
        available = false;
        return false;
    }

    // Lightweight probe without constructing a full backend
    available = helios::probeAnyGPUBackend();

    return available;
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

    // Cache key: the global-data label + excitation step. No need to hash the
    // underlying biochemistry floats — the label is the authoritative identity.
    FluspectCacheKey key{biochem_label, fp_round5(excitation_step_nm)};
    auto it = fluspect_cache.find(key);
    if (it != fluspect_cache.end()) {
        return &it->second;
    }

    // Cache miss — resolve the global-data biochemistry vector, build the struct,
    // compute the Fluspect-B kernel.
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

void RadiationModel::populateExcitationAPAR(ExcitationSet &exc) {
    // After excitation bands have been ray-traced, their radiation_flux_<band>
    // primitive data holds per-leaf absorbed flux. Copy into apar_buffer and
    // clear the primitive data (internal bands shouldn't pollute user namespace).
    exc.apar_buffer.clear();
    const std::vector<uint> all_UUIDs = context->getAllUUIDs();
    const size_t n_bands = exc.band_labels.size();
    for (uint UUID : all_UUIDs) {
        // Only track primitives we care about: leaves with a fluspect_spectrum label.
        if (!context->doesPrimitiveDataExist(UUID, "fluspect_spectrum")) {
            continue;
        }
        std::vector<float> row(n_bands, 0.f);
        for (size_t b = 0; b < n_bands; ++b) {
            const std::string prop = "radiation_flux_" + exc.band_labels[b];
            if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                context->getPrimitiveData(UUID, prop.c_str(), row[b]);
            }
        }
        exc.apar_buffer.emplace(UUID, std::move(row));
    }
    // Clear primitive data for internal excitation bands from ALL primitives.
    for (uint UUID : all_UUIDs) {
        for (const auto &band_label : exc.band_labels) {
            const std::string prop = "radiation_flux_" + band_label;
            if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                context->clearPrimitiveData(UUID, prop.c_str());
            }
        }
    }
    exc.populated = true;
}

void RadiationModel::runExcitationBands() {
    // Run any excitation set that isn't already populated. All internal bands
    // for a set are launched together via runBand(vector), then their per-leaf
    // absorbed flux is stashed into apar_buffer.
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
        populateExcitationAPAR(exc);
    }
}

void RadiationModel::computeSIFEmission(const std::string &emission_band) {
    // Populate sif_emission_buffer[emission_band] (top face) and
    // sif_emission_buffer_bottom[emission_band] (bottom face) for every leaf
    // primitive, integrating the Fluspect-B kernel against per-excitation-band
    // APAR and scaling by the van der Tol (2014) quantum yield.

    auto band_it = radiation_bands.find(emission_band);
    if (band_it == radiation_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): band '" + emission_band + "' does not exist.");
    }
    const float em_min = band_it->second.wavebandBounds.x;
    const float em_max = band_it->second.wavebandBounds.y;

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
    const ExcitationSet *matched = nullptr;
    for (const auto &kv : excitation_sets) {
        if (std::abs(kv.second.bin_width_nm - step) < 1e-5f) {
            matched = &kv.second;
            break;
        }
    }
    if (!matched) {
        helios_runtime_error("ERROR (RadiationModel::computeSIFEmission): no excitation set found for bin width " + std::to_string(step) + " nm (band '" + emission_band + "').");
    }

    const std::vector<uint> all_UUIDs = context->getAllUUIDs();
    size_t n_applied = 0;
    size_t n_skipped_no_biochem_has_etr = 0;  // leaves with J/Jmax but no biochemistry label
    size_t n_skipped_has_biochem_no_etr = 0;  // leaves with biochemistry but no J/Jmax
    for (uint UUID : all_UUIDs) {
        const bool has_biochem = context->doesPrimitiveDataExist(UUID, "fluspect_spectrum");
        const bool has_etr = context->doesPrimitiveDataExist(UUID, "electron_transport_ratio");
        if (!has_etr) {
            if (has_biochem) ++n_skipped_has_biochem_no_etr;
            continue;
        }
        const helios::FluspectKernel *kernel = getOrComputeFluspectKernel(UUID, step);
        if (!kernel) {
            // getOrComputeFluspectKernel returns nullptr iff fluspect_spectrum is missing.
            if (has_etr) ++n_skipped_no_biochem_has_etr;
            continue;
        }

        float J_over_Jmax = 0.f;
        context->getPrimitiveData(UUID, "electron_transport_ratio", J_over_Jmax);
        float T_leaf_K = 298.15f;
        if (context->doesPrimitiveDataExist(UUID, "temperature")) {
            context->getPrimitiveData(UUID, "temperature", T_leaf_K);
            if (T_leaf_K <= 0.f) T_leaf_K = 298.15f;
        }
        const float Phi_F = calculateFluorescenceYield(J_over_Jmax, T_leaf_K);
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

        // Integrate kernel × APAR across excitation grid → per-emission-wavelength spectrum.
        // Then integrate that spectrum across [em_min, em_max] → source flux for this band.
        //
        // The kernel grid is 4 nm spacing in wlf (emission); we use trapezoidal integration
        // over the intersection with [em_min, em_max]. We build two spectra (top = Mf,
        // bottom = Mb) then integrate each.
        const auto &wle = kernel->wle;
        const auto &wlf = kernel->wlf;

        auto apar_it = matched->apar_buffer.find(UUID);
        if (apar_it == matched->apar_buffer.end()) continue;
        const std::vector<float> &apar_bands = apar_it->second;

        // For each excitation sub-band, find its index in the kernel's wle grid and
        // accumulate apar[b] * kernel_col[em] into per-em contributions.
        // The kernel wle is defined at the midpoints of 400-750 at 'step' spacing. We map
        // each band_index to wle by matching band center to wle[j].
        std::vector<double> F_top(wlf.size(), 0.0);
        std::vector<double> F_bot(wlf.size(), 0.0);
        for (size_t b = 0; b < matched->band_labels.size(); ++b) {
            const float band_center = 0.5f * (matched->band_min_nm[b] + matched->band_max_nm[b]);
            // Locate the closest wle index.
            size_t j_best = 0;
            float best_delta = std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < wle.size(); ++j) {
                const float d = std::abs(wle[j] - band_center);
                if (d < best_delta) {
                    best_delta = d;
                    j_best = j;
                }
            }
            // APAR in band b is the absorbed flux on this primitive, W/m².
            const double apar = apar_bands[b];
            if (apar == 0.0) continue;
            for (size_t i = 0; i < wlf.size(); ++i) {
                F_top[i] += apar * kernel->Mf[i][j_best];
                F_bot[i] += apar * kernel->Mb[i][j_best];
            }
        }

        // Multiply by Phi_F × fqe (since we build the kernel with fqe=1 internally, the
        // quantum-yield scaling and user fqe calibration both happen here).
        for (size_t i = 0; i < wlf.size(); ++i) {
            F_top[i] *= phi_F_scaled;
            F_bot[i] *= phi_F_scaled;
        }

        // Integrate F_top/F_bot over [em_min, em_max] via trapezoid over the kernel's wlf grid.
        auto integrate = [&](const std::vector<double> &F) -> double {
            double total = 0.0;
            for (size_t i = 0; i + 1 < wlf.size(); ++i) {
                const float w0 = wlf[i];
                const float w1 = wlf[i + 1];
                if (w1 < em_min) continue;
                if (w0 > em_max) break;
                // Clip to band
                const double lo = std::max<double>(w0, em_min);
                const double hi = std::min<double>(w1, em_max);
                if (hi <= lo) continue;
                // Linear interp of F between w0 and w1 over [lo, hi]. Integrated trapezoid
                // of a linear function equals average × width.
                const double frac_lo = (lo - w0) / (w1 - w0);
                const double frac_hi = (hi - w0) / (w1 - w0);
                const double F_lo = F[i] + frac_lo * (F[i + 1] - F[i]);
                const double F_hi = F[i] + frac_hi * (F[i + 1] - F[i]);
                total += 0.5 * (F_lo + F_hi) * (hi - lo);
            }
            return total;
        };

        const double emit_top = integrate(F_top);
        const double emit_bot = integrate(F_bot);
        buf_top[UUID] = static_cast<float>(emit_top);
        buf_bot[UUID] = static_cast<float>(emit_bot);
        ++n_applied;
    }

    if (n_applied == 0 && message_flag) {
        if (n_skipped_has_biochem_no_etr == 0 && n_skipped_no_biochem_has_etr == 0) {
            std::cerr << "WARNING (RadiationModel::computeSIFEmission): SIF-flagged band '" << emission_band
                      << "' will emit zero — no primitives have both 'fluspect_spectrum' (leaf biochemistry "
                         "label) and 'electron_transport_ratio' (J/Jmax) primitive data. Call "
                         "LeafOptics::run() to author leaf biochemistry and PhotosynthesisModel::run() "
                         "with optionalOutputPrimitiveData(\"electron_transport_ratio\") before runBand()."
                      << std::endl;
        } else if (n_skipped_has_biochem_no_etr > 0 && n_skipped_no_biochem_has_etr == 0) {
            std::cerr << "WARNING (RadiationModel::computeSIFEmission): SIF-flagged band '" << emission_band
                      << "' will emit zero — " << n_skipped_has_biochem_no_etr << " primitives have "
                         "'fluspect_spectrum' but lack 'electron_transport_ratio'. Run PhotosynthesisModel::run() "
                         "with optionalOutputPrimitiveData(\"electron_transport_ratio\") before runBand()."
                      << std::endl;
        } else if (n_skipped_no_biochem_has_etr > 0 && n_skipped_has_biochem_no_etr == 0) {
            std::cerr << "WARNING (RadiationModel::computeSIFEmission): SIF-flagged band '" << emission_band
                      << "' will emit zero — " << n_skipped_no_biochem_has_etr << " primitives have "
                         "'electron_transport_ratio' but lack 'fluspect_spectrum'. Call LeafOptics::run() "
                         "to author leaf biochemistry for those primitives."
                      << std::endl;
        } else {
            std::cerr << "WARNING (RadiationModel::computeSIFEmission): SIF-flagged band '" << emission_band
                      << "' will emit zero — " << n_skipped_has_biochem_no_etr << " primitives have "
                         "'fluspect_spectrum' but lack 'electron_transport_ratio', and "
                      << n_skipped_no_biochem_has_etr << " have 'electron_transport_ratio' but lack "
                         "'fluspect_spectrum'. No primitive has both required fields."
                      << std::endl;
        }
    } else if (n_applied > 0 && n_skipped_has_biochem_no_etr > 0 && message_flag) {
        // Non-fatal partial-coverage case: some leaves produced SIF but others silently didn't.
        std::cerr << "WARNING (RadiationModel::computeSIFEmission): band '" << emission_band << "': "
                  << n_skipped_has_biochem_no_etr << " primitives with 'fluspect_spectrum' were silently "
                     "skipped because they lack 'electron_transport_ratio'. ("
                  << n_applied << " primitives emitted SIF normally.)"
                  << std::endl;
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
    //   - Leaves have Cab but lack electron_transport_ratio (Phi_F can't be computed → zero SIF).
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
        //     runtime check in computeSIFEmission() fires if J/Jmax is still missing at dispatch.
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

float RadiationModel::calculateFluorescenceYield(float J_over_Jmax, float T_leaf_K) {
    // Van der Tol et al. (2014), Eq. 6–10: rate-coefficient model for Φ_F.
    // kF: constant fluorescence rate. kD: thermal (constitutive) dissipation, linear in T.
    // kN: non-photochemical quenching (NPQ), modeled as a piecewise-linear function of x = 1 − J/Jmax.
    // kP: photochemistry rate, back-solved from Φ_P = J/Jmax * Φ_P_max.
    constexpr float kF = 0.05f;

    const float T_C = T_leaf_K - 273.15f;
    const float kD = std::max(0.03f * T_C + 0.0773f, 0.87f);

    const float x = helios::clamp(1.f - J_over_Jmax, 0.f, 1.f);
    float kN;
    if (x < 0.2f) {
        kN = 0.f;
    } else if (x < 0.6f) {
        kN = 2.0f * (x - 0.2f) / 0.4f;
    } else {
        kN = 2.0f + 4.0f * (x - 0.6f) / 0.4f;
    }

    constexpr float Phi_P_max = 0.85f;
    // Clamp strictly below 1 so (1 − Φ_P) never reaches zero.
    const float Phi_P = helios::clamp(J_over_Jmax * Phi_P_max, 0.f, 0.84f);
    const float kP = (kF + kD + kN) * Phi_P / (1.f - Phi_P);

    return kF / (kF + kD + kP + kN);
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
    if (Nsources > 256) {
        helios_runtime_error("ERROR (RadiationModel::addCollimatedRadiationSource): A maximum of 256 radiation sources are allowed.");
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
    if (Nsources > 256) {
        helios_runtime_error("ERROR (RadiationModel::addSphereRadiationSource): A maximum of 256 radiation sources are allowed.");
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
    if (Nsources > 256) {
        helios_runtime_error("ERROR (RadiationModel::addSunSphereRadiationSource): A maximum of 256 radiation sources are allowed.");
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
    if (Nsources > 256) {
        helios_runtime_error("ERROR (RadiationModel::addRectangleRadiationSource): A maximum of 256 radiation sources are allowed.");
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
    if (Nsources > 256) {
        helios_runtime_error("ERROR (RadiationModel::addDiskRadiationSource): A maximum of 256 radiation sources are allowed.");
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
        if (wavebounds == make_vec2(0, 0)) {
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
        if (wavebounds == make_vec2(0, 0)) {
            wavebounds = make_vec2(spectrum.front().x, spectrum.back().x);
        }
        return integrateSpectrum(spectrum, wavebounds.x, wavebounds.y);
    }

    return 0.f;
}

void RadiationModel::enableLightModelVisualization() {
    islightvisualizationenabled = true;

    // build the geometry of any existing sources at this point
    for (int s = 0; s < radiation_sources.size(); s++) {
        buildLightModelGeometry(s);
    }
}

void RadiationModel::disableLightModelVisualization() {
    islightvisualizationenabled = false;
    for (auto &UUIDs: source_model_UUIDs) {
        context->deletePrimitive(UUIDs.second);
    }
}

void RadiationModel::enableCameraModelVisualization() {
    iscameravisualizationenabled = true;

    // build the geometry of any existing cameras at this point
    for (auto &cam: cameras) {
        buildCameraModelGeometry(cam.first);
    }
}

void RadiationModel::disableCameraModelVisualization() {
    iscameravisualizationenabled = false;
    for (auto &UUIDs: camera_model_UUIDs) {
        context->deletePrimitive(UUIDs.second);
    }
}

void RadiationModel::buildLightModelGeometry(uint sourceID) {

    assert(sourceID < radiation_sources.size());

    RadiationSource source = radiation_sources.at(sourceID);
    if (source.source_type == RADIATION_SOURCE_TYPE_SPHERE) {
        source_model_UUIDs[sourceID] = context->loadOBJ("SphereLightSource.obj", true);
    } else if (source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
        source_model_UUIDs[sourceID] = context->loadOBJ("SphereLightSource.obj", true);
    } else if (source.source_type == RADIATION_SOURCE_TYPE_DISK) {
        source_model_UUIDs[sourceID] = context->loadOBJ("DiskLightSource.obj", true);
        context->scalePrimitive(source_model_UUIDs.at(sourceID), make_vec3(source.source_width.x, source.source_width.y, 0.05f * source.source_width.x));
        std::vector<uint> UUIDs_arrow = context->loadOBJ("Arrow.obj", true);
        source_model_UUIDs.at(sourceID).insert(source_model_UUIDs.at(sourceID).begin(), UUIDs_arrow.begin(), UUIDs_arrow.end());
        context->scalePrimitive(UUIDs_arrow, make_vec3(1, 1, 1) * 0.25f * source.source_width.x);
    } else if (source.source_type == RADIATION_SOURCE_TYPE_RECTANGLE) {
        source_model_UUIDs[sourceID] = context->loadOBJ("RectangularLightSource.obj", true);
        context->scalePrimitive(source_model_UUIDs.at(sourceID), make_vec3(source.source_width.x, source.source_width.y, fmin(0.05f * (source.source_width.x + source.source_width.y), 0.5f * fmin(source.source_width.x, source.source_width.y))));
        std::vector<uint> UUIDs_arrow = context->loadOBJ("Arrow.obj", true);
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

    context->setPrimitiveData(source_model_UUIDs.at(sourceID), "twosided_flag", uint(3)); // source model does not interact with radiation field
}

void RadiationModel::buildCameraModelGeometry(const std::string &cameralabel) {

    assert(cameras.find(cameralabel) != cameras.end());

    RadiationCamera camera = cameras.at(cameralabel);

    vec3 viewvec = camera.lookat - camera.position;
    SphericalCoord viewsph = cart2sphere(viewvec);

    camera_model_UUIDs[cameralabel] = context->loadOBJ("Camera.obj", true);

    context->rotatePrimitive(camera_model_UUIDs.at(cameralabel), viewsph.elevation, "x");
    context->rotatePrimitive(camera_model_UUIDs.at(cameralabel), -viewsph.azimuth, "z");

    context->translatePrimitive(camera_model_UUIDs.at(cameralabel), camera.position);

    context->setPrimitiveData(camera_model_UUIDs.at(cameralabel), "twosided_flag", uint(3)); // camera model does not interact with radiation field
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

float RadiationModel::integrateSpectrum(uint source_ID, const std::vector<helios::vec2> &object_spectrum, float wavelength1, float wavelength2) const {

    if (source_ID >= radiation_sources.size()) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum was not set for source ID. Make sure to set its spectrum using setSourceSpectrum() function.");
    } else if (object_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    } else if (wavelength1 > wavelength2 || wavelength1 == wavelength2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Lower wavelength bound must be less than the upper wavelength bound.");
    }

    std::vector<helios::vec2> source_spectrum = radiation_sources.at(source_ID).source_spectrum;

    int istart = 0;
    int iend = (int) object_spectrum.size() - 1;
    for (auto i = 0; i < object_spectrum.size() - 1; i++) {

        if (object_spectrum.at(i).x <= wavelength1 && object_spectrum.at(i + 1).x > wavelength1) {
            istart = i;
        }
        if (object_spectrum.at(i).x <= wavelength2 && object_spectrum.at(i + 1).x > wavelength2) {
            iend = i + 1;
            break;
        }
    }

    float E = 0;
    float Etot = 0;
    for (auto i = istart; i < iend; i++) {

        float x0 = object_spectrum.at(i).x;
        float Esource0 = interp1(source_spectrum, object_spectrum.at(i).x);
        float Eobject0 = object_spectrum.at(i).y;

        float x1 = object_spectrum.at(i + 1).x;
        float Eobject1 = object_spectrum.at(i + 1).y;
        float Esource1 = interp1(source_spectrum, object_spectrum.at(i + 1).x);

        E += 0.5f * (Eobject0 * Esource0 + Eobject1 * Esource1) * (x1 - x0);
        Etot += 0.5f * (Esource1 + Esource0) * (x1 - x0);
    }

    return E / Etot;
}

float RadiationModel::integrateSpectrum(const std::vector<helios::vec2> &object_spectrum, float wavelength1, float wavelength2) const {

    if (object_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    } else if (wavelength1 > wavelength2 || wavelength1 == wavelength2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Lower wavelength bound must be less than the upper wavelength bound.");
    }

    int istart = 1;
    int iend = (int) object_spectrum.size() - 1;
    for (auto i = 0; i < object_spectrum.size() - 1; i++) {

        if (object_spectrum.at(i).x <= wavelength1 && object_spectrum.at(i + 1).x > wavelength1) {
            istart = i;
        }
        if (object_spectrum.at(i).x <= wavelength2 && object_spectrum.at(i + 1).x > wavelength2) {
            iend = i + 1;
            break;
        }
    }

    float E = 0;
    for (auto i = istart; i < iend; i++) {
        float E0 = object_spectrum.at(i).y;
        float x0 = object_spectrum.at(i).x;
        float E1 = object_spectrum.at(i + 1).y;
        float x1 = object_spectrum.at(i + 1).x;
        E += (E0 + E1) * (x1 - x0) * 0.5f;
    }

    return E;
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
    }

    std::vector<helios::vec2> source_spectrum = radiation_sources.at(source_ID).source_spectrum;

    float E = 0;
    float Etot = 0;
    for (auto i = 1; i < object_spectrum.size(); i++) {

        if (object_spectrum.at(i).x <= source_spectrum.front().x || object_spectrum.at(i).x <= camera_spectrum.front().x) {
            continue;
        }
        if (object_spectrum.at(i).x > source_spectrum.back().x || object_spectrum.at(i).x > camera_spectrum.back().x) {
            break;
        }
        float x1 = object_spectrum.at(i).x;
        float Eobject1 = object_spectrum.at(i).y;
        float Esource1 = interp1(source_spectrum, x1);
        float Ecamera1 = interp1(camera_spectrum, x1);


        float x0 = object_spectrum.at(i - 1).x;
        float Eobject0 = object_spectrum.at(i - 1).y;
        float Esource0 = interp1(source_spectrum, x0);
        float Ecamera0 = interp1(camera_spectrum, x0);

        E += 0.5f * ((Eobject1 * Esource1 * Ecamera1) + (Eobject0 * Ecamera0 * Esource0)) * (x1 - x0);
        Etot += 0.5f * (Esource1 + Esource0) * (x1 - x0);
    }


    return E / Etot;
}

float RadiationModel::integrateSpectrum(const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum) const {

    if (object_spectrum.size() < 2) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    }

    float E = 0;
    float Etot = 0;
    for (auto i = 1; i < object_spectrum.size(); i++) {

        if (object_spectrum.at(i).x <= camera_spectrum.front().x) {
            continue;
        }
        if (object_spectrum.at(i).x > camera_spectrum.back().x) {
            break;
        }

        float x1 = object_spectrum.at(i).x;
        float Eobject1 = object_spectrum.at(i).y;
        float Ecamera1 = interp1(camera_spectrum, x1);


        float x0 = object_spectrum.at(i - 1).x;
        float Eobject0 = object_spectrum.at(i - 1).y;
        float Ecamera0 = interp1(camera_spectrum, x0);

        E += 0.5f * ((Eobject1 * Ecamera1) + (Eobject0 * Ecamera0)) * (x1 - x0);
        Etot += 0.5f * (Ecamera1 + Ecamera0) * (x1 - x0);
    }

    return E / Etot;
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

void RadiationModel::setMinScatterEnergy(const std::string &label, uint energy) {

    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (setMinScatterEnergy): Cannot set minimum scattering energy for band '" + label + "' because it is not a valid band.");
    }
    radiation_bands.at(label).minScatterEnergy = energy;
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

    backend->updateGeometry(geometry_data);
    backend->buildAccelerationStructure();

    radiativepropertiesneedupdate = true;
    isgeometryinitialized = true;

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

    float eps;

    std::string prop;
    std::vector<std::string> band_labels;
    for (auto &band: radiation_bands) {
        band_labels.push_back(band.first);
    }

    // Allocate flat arrays directly in material_data to avoid nested vector overhead and redundant copies
    material_data.num_primitives = Nprimitives;
    material_data.num_bands = Nbands;
    material_data.num_sources = Nsources;
    material_data.num_cameras = Ncameras;

    size_t mat_size = (size_t)Nsources * Nprimitives * Nbands;
    material_data.reflectivity.assign(mat_size, rho_default);
    material_data.transmissivity.assign(mat_size, tau_default);

    if (Ncameras > 0) {
        size_t cam_size = (size_t)Nsources * Nprimitives * Nbands * Ncameras;
        material_data.reflectivity_cam.assign(cam_size, rho_default);
        material_data.transmissivity_cam.assign(cam_size, tau_default);
    } else {
        material_data.reflectivity_cam.clear();
        material_data.transmissivity_cam.clear();
    }

    MaterialPropertyIndexer mat_idx(Nsources, Nprimitives, Nbands);
    CameraMaterialIndexer cam_idx(Nsources, Nprimitives, Nbands, Ncameras);

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

                if (!camera_response.empty()) {

                    if (!context->doesGlobalDataExist(camera_response.c_str())) {
                        if (camera_response != "uniform") {
                            warnings.addWarning("missing_camera_response", "Camera spectral response \"" + camera_response + "\" does not exist. Assuming a uniform spectral response.");
                        }
                    } else if (context->getGlobalDataType(camera_response.c_str()) == helios::HELIOS_TYPE_VEC2) {

                        std::vector<helios::vec2> data = loadSpectralData(camera_response.c_str());

                        camera_response_unique.at(cam).at(b) = data;

                    } else if (context->getGlobalDataType(camera_response.c_str()) != helios::HELIOS_TYPE_VEC2 && context->getGlobalDataType(camera_response.c_str()) != helios::HELIOS_TYPE_STRING) {
                        camera_response.clear();
                        std::cout << "WARNING (RadiationModel::runBand): Camera spectral response \"" << camera_response << "\" is not of type HELIOS_TYPE_VEC2 or HELIOS_TYPE_STRING. Assuming a uniform spectral response..." << std::endl;
                    }
                }
            }
            cam++;
        }
    }

    // Spectral integration cache to avoid redundant computations
    std::unordered_map<std::string, float> spectral_integration_cache;

#ifdef USE_OPENMP
    // Temporary cache for this thread group (will be merged later)
    std::unordered_map<std::string, float> temp_spectral_cache;
#endif

    // Helper function to create cache keys for spectral integrations
    auto createCacheKey = [](const std::string &spectrum_label, uint source_id, uint band_id, uint camera_id, const std::string &type) -> std::string {
        return spectrum_label + "_" + std::to_string(source_id) + "_" + std::to_string(band_id) + "_" + std::to_string(camera_id) + "_" + type;
    };

    // Helper function to get from cache (thread-safe)
    auto getCachedValue = [&](const std::string &cache_key, bool &found) -> float {
        float result = 0.0f;
        found = false;

#ifdef USE_OPENMP
#pragma omp critical
        {
#endif
            // Check shared cache
            auto cache_it = spectral_integration_cache.find(cache_key);
            if (cache_it != spectral_integration_cache.end()) {
                found = true;
                result = cache_it->second;
            }
#ifdef USE_OPENMP
        }
#endif
        return result;
    };

    // Helper function to store in cache (thread-safe)
    auto setCachedValue = [&](const std::string &cache_key, float value) {
#ifdef USE_OPENMP
#pragma omp critical
        {
#endif
            spectral_integration_cache[cache_key] = value;
#ifdef USE_OPENMP
        }
#endif
    };

    // Helper function for cached interpolation (thread-safe)
    auto cachedInterp1 = [&](const std::vector<helios::vec2> &spectrum, float wavelength, const std::string &spectrum_id) -> float {
        // Create cache key for this specific interpolation
        std::string cache_key = "interp_" + spectrum_id + "_" + std::to_string(wavelength);

        bool found = false;
        float cached_result = getCachedValue(cache_key, found);
        if (found) {
            return cached_result;
        }

        // Perform interpolation and cache result
        float result = interp1(spectrum, wavelength);
        setCachedValue(cache_key, result);
        return result;
    };

    // Cached version of integrateSpectrum with source spectrum
    auto cachedIntegrateSpectrumWithSource = [&](uint source_ID, const std::vector<helios::vec2> &object_spectrum, float wavelength1, float wavelength2, const std::string &object_spectrum_id) -> float {
        if (source_ID >= radiation_sources.size() || object_spectrum.size() < 2 || wavelength1 >= wavelength2) {
            return 0.0f; // Handle edge cases gracefully
        }

        std::vector<helios::vec2> source_spectrum = radiation_sources.at(source_ID).source_spectrum;
        std::string source_id = "source_" + std::to_string(source_ID);

        int istart = 0;
        int iend = (int) object_spectrum.size() - 1;
        for (auto i = 0; i < object_spectrum.size() - 1; i++) {
            if (object_spectrum.at(i).x <= wavelength1 && object_spectrum.at(i + 1).x > wavelength1) {
                istart = i;
            }
            if (object_spectrum.at(i).x <= wavelength2 && object_spectrum.at(i + 1).x > wavelength2) {
                iend = i + 1;
                break;
            }
        }

        float E = 0;
        float Etot = 0;
        for (auto i = istart; i < iend; i++) {
            float x0 = object_spectrum.at(i).x;
            float Esource0 = cachedInterp1(source_spectrum, x0, source_id);
            float Eobject0 = object_spectrum.at(i).y;

            float x1 = object_spectrum.at(i + 1).x;
            float Eobject1 = object_spectrum.at(i + 1).y;
            float Esource1 = cachedInterp1(source_spectrum, x1, source_id);

            E += 0.5f * (Eobject0 * Esource0 + Eobject1 * Esource1) * (x1 - x0);
            Etot += 0.5f * (Esource1 + Esource0) * (x1 - x0);
        }

        return (Etot != 0.0f) ? E / Etot : 0.0f;
    };

    // Cached version of integrateSpectrum with source and camera spectra
    auto cachedIntegrateSpectrumWithSourceAndCamera = [&](uint source_ID, const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum, uint camera_index, uint band_index,
                                                          const std::string &object_spectrum_id) -> float {
        if (source_ID >= radiation_sources.size() || object_spectrum.size() < 2) {
            return 0.0f;
        }

        std::vector<helios::vec2> source_spectrum = radiation_sources.at(source_ID).source_spectrum;
        std::string source_id = "source_" + std::to_string(source_ID);
        std::string camera_id = "camera_" + std::to_string(camera_index) + "_band_" + std::to_string(band_index); // Include band for unique cache key per band

        float E = 0;
        float Etot = 0;
        for (auto i = 1; i < object_spectrum.size(); i++) {
            if (object_spectrum.at(i).x <= source_spectrum.front().x || object_spectrum.at(i).x <= camera_spectrum.front().x) {
                continue;
            }
            if (object_spectrum.at(i).x > source_spectrum.back().x || object_spectrum.at(i).x > camera_spectrum.back().x) {
                break;
            }

            float x1 = object_spectrum.at(i).x;
            float Eobject1 = object_spectrum.at(i).y;
            float Esource1 = cachedInterp1(source_spectrum, x1, source_id);
            float Ecamera1 = cachedInterp1(camera_spectrum, x1, camera_id);

            float x0 = object_spectrum.at(i - 1).x;
            float Eobject0 = object_spectrum.at(i - 1).y;
            float Esource0 = cachedInterp1(source_spectrum, x0, source_id);
            float Ecamera0 = cachedInterp1(camera_spectrum, x0, camera_id);

            E += 0.5f * ((Eobject1 * Esource1 * Ecamera1) + (Eobject0 * Ecamera0 * Esource0)) * (x1 - x0);
            Etot += 0.5f * (Esource1 + Esource0) * (x1 - x0);
        }

        return (Etot != 0.0f) ? E / Etot : 0.0f;
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

        if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
            if (context->getPrimitiveDataType("reflectivity_spectrum") == HELIOS_TYPE_STRING) {
                std::string spectrum_label;
                context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);

                // get the spectral reflectivity data and store it in surface_spectra to avoid having to load it again
                if (surface_spectra_rho.find(spectrum_label) == surface_spectra_rho.end()) {
                    if (!context->doesGlobalDataExist(spectrum_label.c_str())) {
                        if (!spectrum_label.empty()) {
                            warnings.addWarning("missing_reflectivity_spectrum", "Primitive spectral reflectivity \"" + spectrum_label + "\" does not exist. Using default reflectivity of 0.");
                        }
                        std::vector<helios::vec2> data;
                        surface_spectra_rho.emplace(spectrum_label, data);
                    } else if (context->getGlobalDataType(spectrum_label.c_str()) == HELIOS_TYPE_VEC2) {

                        std::vector<helios::vec2> data = loadSpectralData(spectrum_label.c_str());
                        surface_spectra_rho.emplace(spectrum_label, data);

                    } else if (context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_VEC2 && context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_STRING) {
                        spectrum_label.clear();
                        std::cout << "WARNING (RadiationModel::runBand): Object spectral reflectivity \"" << spectrum_label << "\" is not of type HELIOS_TYPE_VEC2 or HELIOS_TYPE_STRING. Assuming a uniform spectral distribution..." << std::flush;
                    }
                }
            }
        }

        if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum")) {
            if (context->getPrimitiveDataType("transmissivity_spectrum") == HELIOS_TYPE_STRING) {
                std::string spectrum_label;
                context->getPrimitiveData(UUID, "transmissivity_spectrum", spectrum_label);

                // get the spectral transmissivity data and store it in surface_spectra to avoid having to load it again
                if (surface_spectra_tau.find(spectrum_label) == surface_spectra_tau.end()) {
                    if (!context->doesGlobalDataExist(spectrum_label.c_str())) {
                        if (!spectrum_label.empty()) {
                            warnings.addWarning("missing_transmissivity_spectrum", "Primitive spectral transmissivity \"" + spectrum_label + "\" does not exist. Using default transmissivity of 0.");
                        }
                        std::vector<helios::vec2> data;
                        surface_spectra_tau.emplace(spectrum_label, data);
                    } else if (context->getGlobalDataType(spectrum_label.c_str()) == HELIOS_TYPE_VEC2) {

                        std::vector<helios::vec2> data = loadSpectralData(spectrum_label.c_str());
                        surface_spectra_tau.emplace(spectrum_label, data);

                    } else if (context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_VEC2 && context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_STRING) {
                        spectrum_label.clear();
                        std::cout << "WARNING (RadiationModel::runBand): Object spectral transmissivity \"" << spectrum_label << "\" is not of type HELIOS_TYPE_VEC2 or HELIOS_TYPE_STRING. Assuming a uniform spectral distribution..." << std::flush;
                    }
                }
            }
        }
    }

    // second, calculate unique values of rho and tau for all sources and bands
    std::map<std::string, std::vector<std::vector<float>>> rho_unique;
    std::map<std::string, std::vector<std::vector<float>>> tau_unique;

    std::map<std::string, std::vector<std::vector<std::vector<float>>>> rho_cam_unique;
    std::map<std::string, std::vector<std::vector<std::vector<float>>>> tau_cam_unique;

    std::vector<std::vector<float>> empty;
    empty.resize(Nbands);
    for (uint b = 0; b < Nbands; b++) {
        empty.at(b).resize(Nsources, 0);
    }
    std::vector<std::vector<std::vector<float>>> empty_cam;
    if (Ncameras > 0) {
        empty_cam.resize(Nbands);
        for (uint b = 0; b < Nbands; b++) {
            empty_cam.at(b).resize(Nsources);
            for (uint s = 0; s < Nsources; s++) {
                empty_cam.at(b).at(s).resize(Ncameras, 0);
            }
        }
    }

    // Convert maps to vectors for OpenMP indexing
    std::vector<std::pair<std::string, std::vector<helios::vec2>>> spectra_rho_vector(surface_spectra_rho.begin(), surface_spectra_rho.end());

    // Pre-initialize all map entries before parallel processing to avoid race conditions
    for (const auto &spectrum: spectra_rho_vector) {
        rho_unique[spectrum.first] = empty;
        if (Ncameras > 0) {
            rho_cam_unique[spectrum.first] = empty_cam;
        }
    }

    // Process reflectivity spectra with OpenMP parallelization
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int spectrum_idx = 0; spectrum_idx < (int) spectra_rho_vector.size(); spectrum_idx++) {
        const auto &spectrum = spectra_rho_vector[spectrum_idx];

        for (uint b = 0; b < Nbands; b++) {
            std::string band = band_labels.at(b);

            for (uint s = 0; s < Nsources; s++) {

                // integrate with caching
                auto band_it = radiation_bands.find(band);
                if (band_it != radiation_bands.end() && band_it->second.wavebandBounds.x != 0 && band_it->second.wavebandBounds.y != 0 && !spectrum.second.empty()) {
                    if (!radiation_sources.at(s).source_spectrum.empty()) {
                        std::string cache_key = createCacheKey(spectrum.first, s, b, 0, "rho_source");
                        bool found;
                        float cached_result = getCachedValue(cache_key, found);
                        if (found) {
                            rho_unique[spectrum.first][b][s] = cached_result;
                        } else {
                            float result = cachedIntegrateSpectrumWithSource(s, spectrum.second, band_it->second.wavebandBounds.x, band_it->second.wavebandBounds.y, spectrum.first);
                            setCachedValue(cache_key, result);
                            rho_unique[spectrum.first][b][s] = result;
                        }
                    } else {
                        // source spectrum not provided, assume source intensity is constant over the band
                        std::string cache_key = createCacheKey(spectrum.first, s, b, 0, "rho_no_source");
                        bool found;
                        float cached_result = getCachedValue(cache_key, found);
                        if (found) {
                            rho_unique[spectrum.first][b][s] = cached_result;
                        } else {
                            float result = integrateSpectrum(spectrum.second, band_it->second.wavebandBounds.x, band_it->second.wavebandBounds.y) / (band_it->second.wavebandBounds.y - band_it->second.wavebandBounds.x);
                            setCachedValue(cache_key, result);
                            rho_unique[spectrum.first][b][s] = result;
                        }
                    }
                } else {
                    // No wavelength bounds, can't integrate spectrum without camera response
                    // Set to default for now, will use camera average if available
                    rho_unique[spectrum.first][b][s] = rho_default;
                }

                // cameras
                if (Ncameras > 0) {
                    uint cam = 0;
                    float rho_cam_sum_for_averaging = 0.f;
                    for (const auto &camera: cameras) {

                        if (camera_response_unique.at(cam).at(b).empty()) {
                            rho_cam_unique[spectrum.first][b][s][cam] = rho_unique[spectrum.first][b][s];
                        } else {

                            // integrate with caching
                            if (!spectrum.second.empty()) {
                                if (!radiation_sources.at(s).source_spectrum.empty()) {
                                    std::string cache_key = createCacheKey(spectrum.first, s, b, cam, "rho_cam_source");
                                    bool found;
                                    float cached_result = getCachedValue(cache_key, found);
                                    if (found) {
                                        rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = cached_result;
                                        rho_cam_sum_for_averaging += cached_result;
                                    } else {
                                        float result = cachedIntegrateSpectrumWithSourceAndCamera(s, spectrum.second, camera_response_unique.at(cam).at(b), cam, b, spectrum.first);
                                        setCachedValue(cache_key, result);
                                        rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = result;
                                        rho_cam_sum_for_averaging += result;
                                    }
                                } else {
                                    std::string cache_key = createCacheKey(spectrum.first, s, b, cam, "rho_cam_no_source");
                                    bool found;
                                    float cached_result = getCachedValue(cache_key, found);
                                    if (found) {
                                        rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = cached_result;
                                        rho_cam_sum_for_averaging += cached_result;
                                    } else {
                                        float result = integrateSpectrum(spectrum.second, camera_response_unique.at(cam).at(b));
                                        setCachedValue(cache_key, result);
                                        rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = result;
                                        rho_cam_sum_for_averaging += result;
                                    }
                                }
                            } else {
                                rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = rho_default;
                            }
                        }

                        cam++;
                    }

                    // CRITICAL FIX: If wavelength bounds weren't set but camera integration produced values,
                    // use camera average as the base reflectivity. This allows regular scatter to work
                    // when only reflectivity_spectrum + camera response are provided.
                    if (rho_unique[spectrum.first][b][s] == rho_default && rho_cam_sum_for_averaging > 0 && cam > 0) {
                        rho_unique[spectrum.first][b][s] = rho_cam_sum_for_averaging / float(cam);
                    }
                }
            }
        }
    }

    // Convert tau spectra to vector for OpenMP indexing
    std::vector<std::pair<std::string, std::vector<helios::vec2>>> spectra_tau_vector(surface_spectra_tau.begin(), surface_spectra_tau.end());

    // Pre-initialize all map entries before parallel processing to avoid race conditions
    for (const auto &spectrum: spectra_tau_vector) {
        tau_unique[spectrum.first] = empty;
        if (Ncameras > 0) {
            tau_cam_unique[spectrum.first] = empty_cam;
        }
    }

    // Process transmissivity spectra with OpenMP parallelization
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int spectrum_idx = 0; spectrum_idx < (int) spectra_tau_vector.size(); spectrum_idx++) {
        const auto &spectrum = spectra_tau_vector[spectrum_idx];

        for (uint b = 0; b < Nbands; b++) {
            std::string band = band_labels.at(b);

            for (uint s = 0; s < Nsources; s++) {

                // integrate with caching
                auto band_it = radiation_bands.find(band);
                if (band_it != radiation_bands.end() && band_it->second.wavebandBounds.x != 0 && band_it->second.wavebandBounds.y != 0 && !spectrum.second.empty()) {
                    if (!radiation_sources.at(s).source_spectrum.empty()) {
                        std::string cache_key = createCacheKey(spectrum.first, s, b, 0, "tau_source");
                        bool found;
                        float cached_result = getCachedValue(cache_key, found);
                        if (found) {
                            tau_unique[spectrum.first][b][s] = cached_result;
                        } else {
                            float result = cachedIntegrateSpectrumWithSource(s, spectrum.second, band_it->second.wavebandBounds.x, band_it->second.wavebandBounds.y, spectrum.first);
                            setCachedValue(cache_key, result);
                            tau_unique[spectrum.first][b][s] = result;
                        }
                    } else {
                        std::string cache_key = createCacheKey(spectrum.first, s, b, 0, "tau_no_source");
                        bool found;
                        float cached_result = getCachedValue(cache_key, found);
                        if (found) {
                            tau_unique[spectrum.first][b][s] = cached_result;
                        } else {
                            float result = integrateSpectrum(spectrum.second, band_it->second.wavebandBounds.x, band_it->second.wavebandBounds.y) / (band_it->second.wavebandBounds.y - band_it->second.wavebandBounds.x);
                            setCachedValue(cache_key, result);
                            tau_unique[spectrum.first][b][s] = result;
                        }
                    }
                } else {
                    tau_unique[spectrum.first][b][s] = tau_default;
                }

                // cameras
                if (Ncameras > 0) {
                    uint cam = 0;
                    for (const auto &camera: cameras) {

                        if (camera_response_unique.at(cam).at(b).empty()) {

                            tau_cam_unique[spectrum.first][b][s][cam] = tau_unique[spectrum.first][b][s];

                        } else {

                            // integrate with caching
                            if (!spectrum.second.empty()) {
                                if (!radiation_sources.at(s).source_spectrum.empty()) {
                                    std::string cache_key = createCacheKey(spectrum.first, s, b, cam, "tau_cam_source");
                                    bool found;
                                    float cached_result = getCachedValue(cache_key, found);
                                    if (found) {
                                        tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = cached_result;
                                    } else {
                                        float result = cachedIntegrateSpectrumWithSourceAndCamera(s, spectrum.second, camera_response_unique.at(cam).at(b), cam, b, spectrum.first);
                                        setCachedValue(cache_key, result);
                                        tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = result;
                                    }
                                } else {
                                    std::string cache_key = createCacheKey(spectrum.first, s, b, cam, "tau_cam_no_source");
                                    bool found;
                                    float cached_result = getCachedValue(cache_key, found);
                                    if (found) {
                                        tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = cached_result;
                                    } else {
                                        float result = integrateSpectrum(spectrum.second, camera_response_unique.at(cam).at(b));
                                        setCachedValue(cache_key, result);
                                        tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = result;
                                    }
                                }
                            } else {
                                tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = tau_default;
                            }
                        }

                        cam++;
                    }
                }
            }
        }
    }

    for (size_t u = 0; u < Nprimitives; u++) {

        uint UUID = context_UUIDs.at(u);

        helios::PrimitiveType type = context->getPrimitiveType(UUID);

        if (type == helios::PRIMITIVE_TYPE_VOXEL) {

        } else { // other than voxels

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
                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                    context->getPrimitiveData(UUID, prop.c_str(), rho_s);
                }

                for (uint s = 0; s < Nsources; s++) {
                    float &rho_val = material_data.reflectivity[mat_idx(s, u, b)];

                    // if reflectivity was manually set, or a spectrum was given and the global data exists
                    if (rho_s != rho_default || spectrum_label.empty() || !context->doesGlobalDataExist(spectrum_label.c_str()) || rho_unique.find(spectrum_label) == rho_unique.end()) {

                        rho_val = rho_s;

                        // cameras
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.reflectivity_cam[cam_idx(s, u, b, cam)] = rho_s;
                        }

                        // use spectrum
                    } else {

                        rho_val = rho_unique.at(spectrum_label).at(b).at(s);

                        // cameras
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.reflectivity_cam[cam_idx(s, u, b, cam)] = rho_cam_unique.at(spectrum_label).at(b).at(s).at(cam);
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
                    for (auto &odata: output_prim_data) {
                        if (odata == "reflectivity") {
                            context->setPrimitiveData(UUID, ("reflectivity_" + std::to_string(s) + "_" + band).c_str(), rho_val);
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
                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                    context->getPrimitiveData(UUID, prop.c_str(), tau_s);
                }

                for (uint s = 0; s < Nsources; s++) {
                    float &tau_val = material_data.transmissivity[mat_idx(s, u, b)];

                    // if transmissivity was manually set, or a spectrum was given and the global data exists
                    if (tau_s != tau_default || spectrum_label.empty() || !context->doesGlobalDataExist(spectrum_label.c_str()) || tau_unique.find(spectrum_label) == tau_unique.end()) {

                        tau_val = tau_s;

                        // cameras
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.transmissivity_cam[cam_idx(s, u, b, cam)] = tau_s;
                        }

                    } else {

                        tau_val = tau_unique.at(spectrum_label).at(b).at(s);

                        // cameras
                        for (uint cam = 0; cam < Ncameras; cam++) {
                            material_data.transmissivity_cam[cam_idx(s, u, b, cam)] = tau_cam_unique.at(spectrum_label).at(b).at(s).at(cam);
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
                    for (auto &odata: output_prim_data) {
                        if (odata == "transmissivity") {
                            context->setPrimitiveData(UUID, ("transmissivity_" + std::to_string(s) + "_" + band).c_str(), tau_val);
                        }
                    }
                }
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

                for (uint s = 0; s < Nsources; s++) {
                    float &rho_val = material_data.reflectivity[mat_idx(s, u, b)];
                    float &tau_val = material_data.transmissivity[mat_idx(s, u, b)];

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
                        } else if (eps + tau_val + rho_val != 1.f && eps > 0.f) {
                            helios_runtime_error("ERROR (RadiationModel): emissivity, transmissivity, and reflectivity must sum to 1 to ensure energy conservation. Band " + band + ", Primitive #" + std::to_string(UUID) + ": eps=" +
                                                 std::to_string(eps) + ", tau=" + std::to_string(tau_val) + ", rho=" + std::to_string(rho_val) + ". It is also possible that you forgot to disable emission for this band.");
                        } else if (radiation_bands.at(band).scatteringDepth == 0 && eps != 1.f) {
                            eps = 1.f;
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

    // Specular reflection properties
    material_data.specular_exponent.resize(Nprimitives, -1.f);
    material_data.specular_scale.resize(Nprimitives, 0.f);

    bool specular_exponent_specified = false;
    bool specular_scale_specified = false;

    for (size_t u = 0; u < Nprimitives; u++) {
        uint UUID = context_UUIDs.at(u);

        if (context->doesPrimitiveDataExist(UUID, "specular_exponent") && context->getPrimitiveDataType("specular_exponent") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "specular_exponent", material_data.specular_exponent.at(u));
            if (material_data.specular_exponent.at(u) >= 0.f) {
                specular_exponent_specified = true;
            }
        }

        if (context->doesPrimitiveDataExist(UUID, "specular_scale") && context->getPrimitiveDataType("specular_scale") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "specular_scale", material_data.specular_scale.at(u));
            if (material_data.specular_scale.at(u) > 0.f) {
                specular_scale_specified = true;
            }
        }
    }

    // Auto-enable specular reflection if specular properties are specified on any primitive
    if (specular_exponent_specified) {
        if (specular_scale_specified) {
            specular_reflection_mode = 2; // Mode 2: use primitive specular_scale
        } else {
            specular_reflection_mode = 1; // Mode 1: use default 0.25 scale
        }
    } else {
        specular_reflection_mode = 0; // Disabled
    }

    backend->updateMaterials(material_data);

    radiativepropertiesneedupdate = false;

    if (message_flag) {
        std::cout << "done\n";
    }

    // Report aggregated warnings
    warnings.report(std::cerr);
}

std::vector<float> RadiationModel::updateAtmosphericSkyModel(const std::vector<std::string> &band_labels, const RadiationCamera &camera) {
    // Prague Sky Model implementation for atmospheric sky radiance
    // Uses validated spectral radiance from brute-force atmospheric simulations
    // (Wilkie et al. 2021, Vévoda et al. 2022)

    size_t Nbands_launch = band_labels.size();
    std::vector<float> sky_base_radiances(Nbands_launch, 0.0f);

    // Only run atmospheric sky model if user has explicitly enabled it by setting atmospheric parameters
    // This prevents the model from running with default values in tests/scripts that don't want it
    bool has_atmospheric_data =
            context->doesGlobalDataExist("atmosphere_pressure_Pa") || context->doesGlobalDataExist("atmosphere_temperature_K") || context->doesGlobalDataExist("atmosphere_humidity_rel") || context->doesGlobalDataExist("atmosphere_turbidity");

    if (!has_atmospheric_data) {
        // No atmospheric parameters set - return zeros (camera will use user-set diffuse flux or 0)
        return sky_base_radiances;
    }

    // Read atmospheric parameters from Context global data (set by SolarPosition plugin)
    // Default values match SolarPosition::getAtmosphericConditions() defaults
    float pressure_Pa = 101325.f; // Standard atmosphere (1 atm)
    float temperature_K = 300.f; // 27°C
    float humidity_rel = 0.5f; // 50% relative humidity
    float turbidity = 0.02f; // Clear sky - Ångström's aerosol turbidity coefficient (AOD at 500nm)

    if (context->doesGlobalDataExist("atmosphere_pressure_Pa")) {
        context->getGlobalData("atmosphere_pressure_Pa", pressure_Pa);
    }
    if (context->doesGlobalDataExist("atmosphere_temperature_K")) {
        context->getGlobalData("atmosphere_temperature_K", temperature_K);
    }
    if (context->doesGlobalDataExist("atmosphere_humidity_rel")) {
        context->getGlobalData("atmosphere_humidity_rel", humidity_rel);
    }
    if (context->doesGlobalDataExist("atmosphere_turbidity")) {
        context->getGlobalData("atmosphere_turbidity", turbidity);
    }

    // --- Check Prague data availability from Context ---
    int prague_valid = 0;
    if (context->doesGlobalDataExist("prague_sky_valid")) {
        context->getGlobalData("prague_sky_valid", prague_valid);
    }

    // Get sun direction from first radiation source (assumed to be sun)
    helios::vec3 sun_dir(0, 0, 1); // Default zenith
    if (!radiation_sources.empty()) {
        sun_dir = radiation_sources[0].source_position;
        sun_dir.normalize();
    }

    // Compute per-band sky radiance parameters
    std::vector<helios::vec4> sky_params(Nbands_launch);

    // Check if Prague data is available
    bool use_prague_fallback = (prague_valid != 1);
    if (use_prague_fallback) {
        // Will use Rayleigh sky fallback - warn user once
        std::cerr << "WARNING (RadiationModel::updateAtmosphericSkyModel): "
                  << "Prague sky model data not available in Context. "
                  << "Using simple Rayleigh sky fallback. "
                  << "Call SolarPosition::updatePragueSkyModel() for accurate sky radiance." << std::endl;
    }

    // Prepare spectral data (either Prague or Rayleigh fallback)
    std::vector<float> wavelengths;
    std::vector<float> L_zenith_spectrum;
    std::vector<float> circ_str_spectrum;
    std::vector<float> circ_width_spectrum;
    std::vector<float> horiz_bright_spectrum;
    std::vector<float> norm_spectrum;

    if (use_prague_fallback) {
        // --- Create simple Rayleigh sky spectrum (λ^-4 dependence) ---
        // 360-750 nm at 10 nm spacing (visible range only for fallback)
        const int n_wavelengths = 40; // (750-360)/10 + 1
        wavelengths.resize(n_wavelengths);
        L_zenith_spectrum.resize(n_wavelengths);
        circ_str_spectrum.resize(n_wavelengths);
        circ_width_spectrum.resize(n_wavelengths);
        horiz_bright_spectrum.resize(n_wavelengths);
        norm_spectrum.resize(n_wavelengths);

        const float L_base = 0.4f; // W/m²/sr/nm at 550 nm (typical clear sky zenith)
        const float lambda_ref = 550.0f; // Reference wavelength

        for (int i = 0; i < n_wavelengths; ++i) {
            float lambda = 360.0f + i * 10.0f;
            wavelengths[i] = lambda;

            // Rayleigh scattering: L(λ) ∝ λ^-4 (blue sky)
            float rayleigh_factor = std::pow(lambda_ref / lambda, 4.0f);
            L_zenith_spectrum[i] = L_base * rayleigh_factor;

            // Simple angular parameters (no strong circumsolar for fallback)
            circ_str_spectrum[i] = 0.5f;
            circ_width_spectrum[i] = 20.0f;
            horiz_bright_spectrum[i] = 1.8f;
            norm_spectrum[i] = 0.7f;
        }
    } else {
        // --- Read spectral parameters from Context ---
        std::vector<float> spectral_params;
        context->getGlobalData("prague_sky_spectral_params", spectral_params);

        const int params_per_wavelength = 6;
        const int n_wavelengths = spectral_params.size() / params_per_wavelength;

        // Parse into structured format
        wavelengths.resize(n_wavelengths);
        L_zenith_spectrum.resize(n_wavelengths);
        circ_str_spectrum.resize(n_wavelengths);
        circ_width_spectrum.resize(n_wavelengths);
        horiz_bright_spectrum.resize(n_wavelengths);
        norm_spectrum.resize(n_wavelengths);

        for (int i = 0; i < n_wavelengths; ++i) {
            int base = i * params_per_wavelength;
            wavelengths[i] = spectral_params[base + 0];
            L_zenith_spectrum[i] = spectral_params[base + 1];
            circ_str_spectrum[i] = spectral_params[base + 2];
            circ_width_spectrum[i] = spectral_params[base + 3];
            horiz_bright_spectrum[i] = spectral_params[base + 4];
            norm_spectrum[i] = spectral_params[base + 5];
        }
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

            if (wavelength_range.x <= 0.f || wavelength_range.y <= 0.f) {
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
                        helios_runtime_error("ERROR (RadiationModel::updateAtmosphericSkyModel): Camera '" + camera.label + "' band '" + band_label + "' has uniform spectral response but no wavelength bounds set.");
                    }
                }
            }

            camera_response.push_back(helios::make_vec2(wavelength_range.x, 1.0f));
            camera_response.push_back(helios::make_vec2(wavelength_range.y, 1.0f));

        } else {
            camera_response = loadSpectralData(spectral_response_label);

            if (camera_response.empty()) {
                helios_runtime_error("ERROR (RadiationModel::updateAtmosphericSkyModel): Camera spectral response '" + spectral_response_label + "' not found for camera '" + camera.label + "' band '" + band_label + "'.");
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

        // CRITICAL: GPU multiplies by normalization (see rayHit.cu:evaluateSkyRadiance)
        // Since normalization < 1 (typically 0.6-0.7), this darkens the sky
        // Pre-divide by normalization so it cancels: GPU does (L/norm) × pattern × norm = L × pattern
        float base_radiance_for_gpu = integrated_L_zenith / std::max(integrated_norm, 0.1f);

        sky_base_radiances[b] = base_radiance_for_gpu;
        sky_params[b] = helios::make_vec4(integrated_circ_str, integrated_circ_width, integrated_horiz_bright, integrated_norm);
    }

    // Sky parameters will be uploaded to backend via updateSkyModel()
    return sky_base_radiances;
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
            if (lambda_min > 0 && lambda_max > lambda_min) {
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
    // populateExcitationAPAR() runs post-dispatch (below).
    std::vector<std::string> effective_label = label;
    std::vector<ExcitationSet *> piggybacked_sets;
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
                piggybacked_sets.push_back(&exc);
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

    // Check to make sure some geometry was added to the context
    if (context->getPrimitiveCount() == 0) {
        std::cerr << "WARNING (RadiationModel::runBand): No geometry was added to the context. There is nothing to simulate...exiting." << std::endl;
        return;
    }

    // Check to make sure geometry was built in OptiX
    if (!isgeometryinitialized) {
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

    if (radiativepropertiesneedupdate) {
        // Use old material path (handles spectrum interpolation)
        updateRadiativeProperties();
        // DON'T call backend->updateMaterials() - old code already uploaded via direct OptiX calls
    } else {
        // Use new backend path (per-band materials only)
        buildMaterialData();
        backend->updateMaterials(material_data);
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
    for (int b = 0; b < Nbands_launch; b++) {
        const std::string &bname = band_labels.at(b);
        const bool is_sif_excitation_band = bname.size() >= 9 && bname.compare(0, 9, "_SIF_exc_") == 0;
        if (scattering_depth.at(b) == 0 && scattering_iterations_needed.at(bname) && !is_sif_excitation_band) {
            std::cout << "WARNING (RadiationModel::runBand): Surface radiative properties for band " << bname
                      << " are set to non-default values, but scattering iterations are disabled. Surface radiative properties will be ignored unless scattering depth is non-zero." << std::endl;
        }
    }

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

    // Initialize camera sky radiance buffer to zeros (will be set per-camera if atmospheric model is used)
    std::vector<float> camera_sky_radiance(Nbands_launch, 0.0f);

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
        // Prague params will be uploaded to backend via updateSkyModel() during scattering
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
    backend->zeroRadiationBuffers(Nbands_launch);

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

            for (auto b = 0; b < label.size(); b++) {
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

        // Compute camera response weighting factors for specular reflection (if cameras exist)
        // Factor = ∫(source_spectrum × camera_response) / ∫(source_spectrum)
        // This must be done before ray tracing so the weights are available during miss_direct()
        if (Ncameras > 0) {
            std::vector<float> source_fluxes_cam;
            source_fluxes_cam.resize(Nsources * Nbands_launch * Ncameras, 1.0f);

            for (uint s = 0; s < Nsources; s++) {
                const RadiationSource &source = radiation_sources.at(s);

                uint cam = 0;
                for (const auto &camera: cameras) {
                    for (uint b = 0; b < Nbands_launch; b++) {
                        std::string band_label = band_labels.at(b);

                        // Default weighting factor (no camera response)
                        float weight = 1.0f;

                        // Check if camera has spectral response for this band
                        if (camera.second.band_spectral_response.find(band_label) != camera.second.band_spectral_response.end()) {
                            std::string response_label = camera.second.band_spectral_response.at(band_label);

                            if (!response_label.empty() && response_label != "uniform" && context->doesGlobalDataExist(response_label.c_str()) && context->getGlobalDataType(response_label.c_str()) == helios::HELIOS_TYPE_VEC2 &&
                                source.source_spectrum.size() > 0) {

                                // Load camera spectral response
                                std::vector<helios::vec2> camera_response;
                                context->getGlobalData(response_label.c_str(), camera_response);

                                // Get band wavelength range
                                helios::vec2 wavelength_range = radiation_bands.at(band_label).wavebandBounds;

                                // If no wavelength bounds, use overlapping range of source and camera
                                if (wavelength_range.x == 0 && wavelength_range.y == 0) {
                                    wavelength_range.x = fmax(source.source_spectrum.front().x, camera_response.front().x);
                                    wavelength_range.y = fmin(source.source_spectrum.back().x, camera_response.back().x);
                                }

                                // Integrate source_spectrum × camera_response over band
                                // Note: integrateSpectrum already returns ratio: ∫(source × camera) / ∫(source)
                                weight = integrateSpectrum(s, camera_response, wavelength_range.x, wavelength_range.y);
                            }
                        }

                        source_fluxes_cam[s * Nbands_launch * Ncameras + b * Ncameras + cam] = weight;
                    }
                    cam++;
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
        params.random_seed = std::chrono::system_clock::now().time_since_epoch().count();
        params.num_bands_global = Nbands_global;
        params.num_bands_launch = Nbands_launch;
        params.specular_reflection_enabled = specular_reflection_mode;

        // Use the band_launch_flag already built above (lines 3375-3383)
        std::vector<bool> band_flags(band_launch_flag.begin(), band_launch_flag.end());
        params.band_launch_flag = band_flags;

        // Pre-allocate camera scatter buffers before direct launch so __miss__direct can fill them.
        // This ensures current_launch_band_count > 0 so getRadiationResults downloads them after.
        if (Ncameras > 0 && scatteringenabled) {
            backend->zeroCameraScatterBuffers(Nbands_launch);
        }

        backend->launchDirectRays(params);

        if (message_flag) {
            std::cout << "done." << std::endl;
        }

    } // end direct source launch

    // --- Extract scattered energy from direct rays for diffuse/emission and scattering ---//
    // This needs to happen BEFORE diffuse/emission block so scattered direct energy
    // is available for both diffuse/emission (via flux_top/flux_bottom) and scattering (via radiation_out)
    std::vector<float> flux_top, flux_bottom;
    flux_top.resize(Nbands_launch * Nprimitives, 0);
    flux_bottom = flux_top;

    // Camera scatter accumulation vectors (declare early for use throughout ray tracing)
    std::vector<float> scatter_top_cam;
    std::vector<float> scatter_bottom_cam;
    if (Ncameras > 0) {
        scatter_top_cam.resize(Nprimitives * Nbands_launch, 0.0f);
        scatter_bottom_cam.resize(Nprimitives * Nbands_launch, 0.0f);
    }

    if (scatteringenabled && rundirect) {
        // Get scattered energy from direct rays for primary diffuse/emission
        helios::RayTracingResults scatter_results;
        backend->getRadiationResults(scatter_results);
        flux_top = scatter_results.scatter_buff_top;
        flux_bottom = scatter_results.scatter_buff_bottom;

        // Accumulate camera scatter from direct rays
        if (Ncameras > 0) {
            for (size_t i = 0; i < scatter_results.scatter_buff_top_cam.size(); i++) {
                scatter_top_cam[i] += scatter_results.scatter_buff_top_cam[i];
                scatter_bottom_cam[i] += scatter_results.scatter_buff_bottom_cam[i];
            }
            // Zero GPU camera scatter buffers to prevent double-counting on next iteration
            backend->zeroCameraScatterBuffers(Nbands_launch);
        }

        // For one-sided primitives, make scattered energy accessible from both faces
        // This is necessary because scattering rays can hit from either direction
        RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);

        for (size_t i = 0; i < Nprimitives; i++) {
            uint UUID = context_UUIDs.at(i);
            uint twosided = context->getPrimitiveTwosidedFlag(UUID, 1);

            if (twosided == 0) { // one-sided primitive - combine top+bottom scattered energy
                for (size_t b = 0; b < Nbands_launch; b++) {
                    size_t ind = rad_indexer(i, b);
                    float total = flux_top[ind] + flux_bottom[ind];
                    flux_top[ind] = total;
                    flux_bottom[ind] = total;
                }
            }
        }

        // Upload scattered energy to backend's radiation_out buffers for scattering iterations
        backend->uploadRadiationOut(flux_top, flux_bottom);
        backend->zeroScatterBuffers();
    }

    // --- Diffuse/Emission launch ---- //

    if (emissionenabled || diffuseenabled) {

        // add any emitted energy to the outgoing energy buffer
        if (emissionenabled) {
            // Update primitive outgoing emission
            float eps, temperature;

            // Create indexer for emission flux buffers
            RadiationBufferIndexer emission_indexer(Nprimitives, Nbands_launch);

            for (auto b = 0; b < Nbands_launch; b++) {
                //\todo For emissivity and twosided_flag, this should be done in updateRadiativeProperties() to avoid having to do it on every runBand() call
                if (radiation_bands.at(band_labels.at(b)).emissionFlag) {
                    std::string prop = "emissivity_" + band_labels.at(b);
                    // SIF bands: per-primitive source flux comes from computeSIFEmission(), with
                    // separate top/bottom buffers populated from Fluspect-B's Mf/Mb kernels.
                    auto sif_band_it = sif_emission_buffer.find(band_labels.at(b));
                    auto sif_band_bot_it = sif_emission_buffer_bottom.find(band_labels.at(b));
                    const bool have_sif_band = (sif_band_it != sif_emission_buffer.end());
                    const bool have_sif_band_bot = (sif_band_bot_it != sif_emission_buffer_bottom.end());
                    for (size_t u = 0; u < Nprimitives; u++) {
                        // Use BufferIndexer: [primitive][band]
                        size_t ind = emission_indexer(u, b);
                        uint p = context_UUIDs.at(u);
                        float out_top;
                        float sif_bottom_flux = 0.f;  // Mb-sourced bottom-face emission (if SIF)
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
                                if (context->doesPrimitiveDataExist(p, "temperature")) {
                                    context->getPrimitiveData(p, "temperature", temperature);
                                    if (temperature < 0) {
                                        temperature = temperature_default;
                                    }
                                } else {
                                    temperature = temperature_default;
                                }
                                out_top = sigma * eps * pow(temperature, 4);
                            }
                        }
                        flux_top.at(ind) += out_top;
                        if (Ncameras > 0) {
                            scatter_top_cam[ind] += out_top;
                        }
                        // Check twosided_flag - check material first, then primitive data
                        uint twosided_flag = context->getPrimitiveTwosidedFlag(p, 1);
                        if (twosided_flag != 0) { // If two-sided, emit from bottom face too
                            if (used_sif) {
                                // SIF two-sided: use Mb-derived bottom flux rather than copying
                                // the Mf-derived top flux. Physically distinct emission lobes.
                                flux_bottom.at(ind) += sif_bottom_flux;
                                if (Ncameras > 0) {
                                    scatter_bottom_cam[ind] += sif_bottom_flux;
                                }
                            } else {
                                flux_bottom.at(ind) += flux_top.at(ind);
                                if (Ncameras > 0) {
                                    scatter_bottom_cam[ind] += out_top;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upload camera scatter buffers accumulated from emission, direct rays, and primary diffuse
        // Camera scatter is accumulated on CPU from GPU after each ray launch
        if (Ncameras > 0) {
            backend->uploadCameraScatterBuffers(scatter_top_cam, scatter_bottom_cam);
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
        params.random_seed = std::chrono::system_clock::now().time_since_epoch().count();
        params.current_band = 0;
        params.num_bands_global = Nbands_global;
        params.num_bands_launch = Nbands_launch;
        std::vector<bool> band_flags(band_launch_flag.begin(), band_launch_flag.end());
        params.band_launch_flag = band_flags;
        params.scattering_iteration = 0;
        params.max_scatters = scatteringDepth;
        params.radiation_out_top = flux_top;
        params.radiation_out_bottom = flux_bottom;

        // Pass diffuse radiation parameters to backend
        params.diffuse_flux = diffuse_flux;
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

        // Retrieve and accumulate camera scatter from primary diffuse
        if (Ncameras > 0) {
            helios::RayTracingResults primary_results;
            backend->getRadiationResults(primary_results);
            for (size_t i = 0; i < primary_results.scatter_buff_top_cam.size(); i++) {
                scatter_top_cam[i] += primary_results.scatter_buff_top_cam[i];
                scatter_bottom_cam[i] += primary_results.scatter_buff_bottom_cam[i];
            }
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
        uint rays_per_primitive = n * n;

        uint s;
        // FIX: Use a copy of band_launch_flag for scattering so modifications don't affect primary launch indices
        std::vector<char> scatter_band_flags = band_launch_flag;

        for (s = 0; s < scatteringDepth; s++) {
            if (message_flag) {
                std::cout << "Performing scattering ray trace (iteration " << s + 1 << " of " << scatteringDepth << ")..." << std::flush;
            }

            int b = -1;
            int active_bands = 0;
            for (uint b_global = 0; b_global < Nbands_global; b_global++) {

                if (scatter_band_flags.at(b_global) == 0) {
                    continue;
                }
                b++;

                const std::string &bname = band_labels.at(b);
                uint depth = radiation_bands.at(bname).scatteringDepth;
                if (s + 1 > depth) {
                    // Internal SIF excitation bands ("_SIF_exc_*") get piggy-backed onto user
                    // dispatches (e.g. PAR) that may have a higher scatteringDepth, so they hit
                    // this "skip" path on every scattering iteration. Suppress the per-band log
                    // for those — with ~35 excitation bins × scatteringDepth iterations they
                    // would flood stdout (and break the in-progress "Performing scattering ray
                    // trace (iteration N of M)..." line, which has no trailing newline).
                    const bool is_sif_excitation_band = bname.size() >= 9 && bname.compare(0, 9, "_SIF_exc_") == 0;
                    if (message_flag && !is_sif_excitation_band) {
                        std::cout << "Skipping band " << bname << " for scattering launch " << s + 1 << std::endl;
                    }
                    scatter_band_flags.at(b_global) = 0; // FIX: Modify copy, not original
                } else {
                    active_bands++;
                }
            }

            // Copy scatter buffers to radiation_out when needed
            // For s=0 with emission+direct: primary diffuse uploaded emission+scatter via params, but we need to copy scatter to avoid double-counting emission on next iteration
            // For s>0: scatter from previous iteration needs to be copied for next iteration
            if (s > 0 || (emissionenabled && rundirect)) {
                backend->copyScatterToRadiation();
            }
            backend->zeroScatterBuffers();

            // Extract radiation_out to ensure it's uploaded for scattering rays
            helios::RayTracingResults scatter_results;
            backend->getRadiationResults(scatter_results);
            std::vector<float> flux_top_scatter = scatter_results.radiation_out_top;
            std::vector<float> flux_bottom_scatter = scatter_results.radiation_out_bottom;

            // Build launch parameters for scattering diffuse rays
            // Launch all primitives at once (backend handles batching if needed)
            helios::RayTracingLaunchParams params;
            params.launch_offset = 0;
            params.launch_count = Nprimitives;
            params.rays_per_primitive = rays_per_primitive;
            params.random_seed = std::chrono::system_clock::now().time_since_epoch().count();
            params.current_band = 0;
            params.num_bands_global = Nbands_global;
            params.num_bands_launch = Nbands_launch;
            std::vector<bool> band_flags(scatter_band_flags.begin(), scatter_band_flags.end()); // FIX: Use scatter copy
            params.band_launch_flag = band_flags;
            params.scattering_iteration = s;
            params.max_scatters = scatteringDepth;

            // Pass diffuse radiation parameters to backend
            params.diffuse_flux = diffuse_flux;
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
            params.radiation_out_top = flux_top_scatter;
            params.radiation_out_bottom = flux_bottom_scatter;

            // Top surface launch
            params.launch_face = 1;
            backend->launchDiffuseRays(params);

            // Bottom surface launch
            params.launch_face = 0;
            backend->launchDiffuseRays(params);

            // Accumulate camera scatter from this scattering iteration
            if (Ncameras > 0) {
                helios::RayTracingResults post_launch;
                backend->getRadiationResults(post_launch);
                for (size_t i = 0; i < post_launch.scatter_buff_top_cam.size(); i++) {
                    scatter_top_cam[i] += post_launch.scatter_buff_top_cam[i];
                    scatter_bottom_cam[i] += post_launch.scatter_buff_bottom_cam[i];
                }
                // Zero GPU camera scatter buffers to prevent double-counting on next iteration
                backend->zeroCameraScatterBuffers(Nbands_launch);
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

        // Upload accumulated camera scatter to radiation_out for cameras to read
        // scatter_top_cam contains camera-weighted scattered energy from all ray types
        // Cameras read from radiation_out during hits, so we upload camera scatter there
        if (Ncameras > 0 && scatteringenabled) {
            backend->uploadRadiationOut(scatter_top_cam, scatter_bottom_cam);
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

        if (scatteringenabled && (emissionenabled || diffuseenabled || rundirect)) {
            // re-set diffuse radiation fluxes (will be passed via launch params)
            if (diffuseenabled) {
                for (auto b = 0; b < Nbands_launch; b++) {
                    diffuse_flux.at(b) = getDiffuseFlux(band_labels.at(b));
                }
            }

            size_t n = ceil(sqrt(double(diffuseRayCount)));

            // Upload sky model parameters to backend (for camera rendering)

            if (!cameras.empty() && prague_params.size() == Nbands_launch) {
                // Get sky radiances for first camera (already computed above)
                std::vector<float> sky_for_backend = updateAtmosphericSkyModel(band_labels, cameras.begin()->second);

                // Upload to backend
                backend->updateSkyModel(prague_params, sky_for_backend, sun_dir, solar_radiances,
                                        has_sun_source ? 0.999989f : 0.0f // solar_disk_cos_angle
                );
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
                    params.random_seed = std::chrono::system_clock::now().time_since_epoch().count();
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

    // Apply camera exposure based on each camera's exposure setting
    for (auto &camera: cameras) {
        camera.second.applyCameraExposure(context);
    }

    // Apply camera white balance based on each camera's white_balance setting
    for (auto &camera: cameras) {
        camera.second.applyCameraWhiteBalance(context);
    }

    // deposit any energy that is left to make sure we satisfy conservation of energy

    // Extract ALL results from backend instead of old OptiX buffers
    helios::RayTracingResults results;
    backend->getRadiationResults(results);

    std::vector<float> radiation_flux_data = results.radiation_in;

    // Extract scatter buffer data from backend results
    TBS_top = results.scatter_buff_top;
    TBS_bottom = results.scatter_buff_bottom;

    std::vector<uint> UUIDs_context_all = context->getAllUUIDs();

    // Create indexer for result extraction
    RadiationBufferIndexer result_indexer(Nprimitives, Nbands_launch);

    for (auto b = 0; b < Nbands_launch; b++) {

        std::string prop = "radiation_flux_" + band_labels.at(b);
        std::vector<float> R(Nprimitives);
        for (size_t u = 0; u < Nprimitives; u++) {
            // Use BufferIndexer: [primitive][band]
            size_t ind = result_indexer(u, b);
            R.at(u) = radiation_flux_data.at(ind) + TBS_top.at(ind) + TBS_bottom.at(ind);
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

    // Finalize any piggy-backed SIF excitation sets: now that radiation_flux_<_SIF_exc_*>
    // primitive data has been populated, copy it into the ExcitationSet apar_buffer and
    // clear the internal primitive data. A subsequent runBand() on SIF emission bands
    // will then skip re-running excitation (populateExcitationAPAR marks populated=true).
    for (ExcitationSet *exc : piggybacked_sets) {
        populateExcitationAPAR(*exc);
    }
}

float RadiationModel::getSkyEnergy() {

    helios::RayTracingResults results;
    backend->getRadiationResults(results);

    float Rsky = 0.f;
    for (size_t i = 0; i < results.sky_energy.size(); i++) {
        Rsky += results.sky_energy.at(i);
    }
    return Rsky;
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

    // Step 3: Get reference Lab values for all detected colorboards
    std::vector<CameraCalibration::LabColor> reference_lab_values;
    std::vector<std::string> colorboard_type_per_patch; // Track which colorboard each patch belongs to

    for (const auto &colorboard_type: colorboard_types) {
        std::vector<CameraCalibration::LabColor> current_reference_values;

        if (colorboard_type == "DGK") {
            current_reference_values = calibration.getReferenceLab_DGK();
        } else if (colorboard_type == "Calibrite") {
            current_reference_values = calibration.getReferenceLab_Calibrite();
        } else if (colorboard_type == "SpyderCHECKR") {
            current_reference_values = calibration.getReferenceLab_SpyderCHECKR();
        } else {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Unsupported colorboard type '" + colorboard_type + "'.");
        }

        // Add to combined list
        reference_lab_values.insert(reference_lab_values.end(), current_reference_values.begin(), current_reference_values.end());

        // Track which colorboard type each patch belongs to
        for (size_t i = 0; i < current_reference_values.size(); i++) {
            colorboard_type_per_patch.push_back(colorboard_type);
        }
    }

    // Step 4: Generate segmentation masks for all colorboard patches
    std::vector<uint> pixel_UUIDs;
    context->getGlobalData(pixel_UUID_label.c_str(), pixel_UUIDs);
    int2 camera_resolution = cameras.at(camera_label).resolution;

    // Create segmentation masks by finding pixels that belong to colorboard patches
    std::map<int, std::vector<std::vector<bool>>> patch_masks;
    int global_patch_idx = 0; // Global patch index across all colorboards

    for (const auto &colorboard_type: colorboard_types) {
        // Get the number of patches for this colorboard type
        int num_patches = 0;
        if (colorboard_type == "DGK") {
            num_patches = 18;
        } else if (colorboard_type == "Calibrite" || colorboard_type == "SpyderCHECKR") {
            num_patches = 24;
        }

        // Generate masks for each patch in this colorboard
        for (int local_patch_idx = 0; local_patch_idx < num_patches; local_patch_idx++) {
            std::vector<std::vector<bool>> mask(camera_resolution.y, std::vector<bool>(camera_resolution.x, false));

            // Find pixels that correspond to this colorboard patch
            for (int y = 0; y < camera_resolution.y; y++) {
                for (int x = 0; x < camera_resolution.x; x++) {
                    int pixel_index = y * camera_resolution.x + x;
                    uint pixel_UUID = pixel_UUIDs[pixel_index];

                    if (pixel_UUID > 0) { // Valid primitive
                        pixel_UUID--; // Convert from 1-based to 0-based indexing

                        // Check if this primitive belongs to this specific colorboard patch
                        std::string colorboard_data_label = "colorboard_" + colorboard_type;
                        if (context->doesPrimitiveDataExist(pixel_UUID, colorboard_data_label.c_str())) {
                            uint patch_id;
                            context->getPrimitiveData(pixel_UUID, colorboard_data_label.c_str(), patch_id);
                            // Patch indices are 0-based, compare directly
                            if ((int) patch_id == local_patch_idx) {
                                mask[y][x] = true;
                            }
                        }
                    }
                }
            }

            patch_masks[global_patch_idx] = mask;
            global_patch_idx++;
        }
    }

    // Step 5: Extract RGB colors from processed camera data (same source as writeCameraImage)
    // Use the same data source that writeCameraImage() uses: cameras.pixel_data
    std::vector<float> red_data, green_data, blue_data;

    // Check if bands exist in camera
    auto &camera_bands = cameras.at(camera_label).band_labels;
    if (std::find(camera_bands.begin(), camera_bands.end(), red_band_label) == camera_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Red band '" + red_band_label + "' not found in camera '" + camera_label + "'.");
    }
    if (std::find(camera_bands.begin(), camera_bands.end(), green_band_label) == camera_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Green band '" + green_band_label + "' not found in camera '" + camera_label + "'.");
    }
    if (std::find(camera_bands.begin(), camera_bands.end(), blue_band_label) == camera_bands.end()) {
        helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Blue band '" + blue_band_label + "' not found in camera '" + camera_label + "'.");
    }

    // Read processed camera data (same as writeCameraImage uses)
    red_data = cameras.at(camera_label).pixel_data.at(red_band_label);
    green_data = cameras.at(camera_label).pixel_data.at(green_band_label);
    blue_data = cameras.at(camera_label).pixel_data.at(blue_band_label);

    // Check data range and normalize if needed
    float max_r = *std::max_element(red_data.begin(), red_data.end());
    float max_g = *std::max_element(green_data.begin(), green_data.end());
    float max_b = *std::max_element(blue_data.begin(), blue_data.end());

    // Normalize camera data to [0,1] range if values are > 1
    float scale_factor = 1.0f;
    if (max_r > 1.0f || max_g > 1.0f || max_b > 1.0f) {
        scale_factor = 1.0f / std::max({max_r, max_g, max_b});

        for (size_t i = 0; i < red_data.size(); i++) {
            red_data[i] *= scale_factor;
            green_data[i] *= scale_factor;
            blue_data[i] *= scale_factor;
        }
    }

    std::vector<helios::vec3> measured_rgb_values;
    int visible_patches = 0;

    for (const auto &[patch_idx, mask]: patch_masks) {
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        int pixel_count = 0;

        // Average RGB values over all pixels in this patch
        for (int y = 0; y < camera_resolution.y; y++) {
            for (int x = 0; x < camera_resolution.x; x++) {
                if (mask[y][x]) {
                    int pixel_index = y * camera_resolution.x + x;
                    sum_r += red_data[pixel_index];
                    sum_g += green_data[pixel_index];
                    sum_b += blue_data[pixel_index];
                    pixel_count++;
                }
            }
        }

        if (pixel_count > 10) { // Only consider patches with sufficient pixels
            helios::vec3 avg_rgb = make_vec3(sum_r / pixel_count, sum_g / pixel_count, sum_b / pixel_count);
            measured_rgb_values.push_back(avg_rgb);

            visible_patches++;
        } else {
            // Add placeholder for missing patch
            measured_rgb_values.push_back(make_vec3(0, 0, 0));
        }
    }

    // Convert measured RGB to Lab and calculate correction matrix
    std::vector<CameraCalibration::LabColor> measured_lab_values;
    for (const auto &rgb: measured_rgb_values) {
        if (rgb.magnitude() > 0) { // Only process non-zero values
            measured_lab_values.push_back(calibration.rgbToLab(rgb));
        }
    }

    // Calculate color correction matrix based on selected algorithm
    std::vector<std::vector<float>> correction_matrix = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};

    // Report which algorithm is being used
    std::string algorithm_name;
    switch (algorithm) {
        case ColorCorrectionAlgorithm::DIAGONAL_ONLY:
            algorithm_name = "Diagonal scaling (white balance only)";
            break;
        case ColorCorrectionAlgorithm::MATRIX_3X3_AUTO:
            algorithm_name = "3x3 matrix with auto-fallback to diagonal";
            break;
        case ColorCorrectionAlgorithm::MATRIX_3X3_FORCE:
            algorithm_name = "3x3 matrix (forced)";
            break;
    }

    if (measured_lab_values.size() >= 6 && reference_lab_values.size() >= 6) {
        // Convert reference Lab back to RGB for matrix fitting
        std::vector<helios::vec3> target_rgb;

        for (size_t i = 0; i < reference_lab_values.size(); i++) {
            CameraCalibration::LabColor ref_lab = reference_lab_values[i];
            helios::vec3 ref_rgb = calibration.labToRgb(ref_lab);
            target_rgb.push_back(ref_rgb);
        }

        // Build matrices for least squares: M * Measured = Target
        // We need to solve for M where M is 3x3 matrix
        // This becomes: M = Target * Measured^T * (Measured * Measured^T)^(-1)

        // Collect valid patches for matrix calculation
        std::vector<helios::vec3> valid_measured, valid_target;
        std::vector<float> patch_weights;

        for (size_t i = 0; i < std::min(measured_rgb_values.size(), target_rgb.size()); i++) {
            if (measured_rgb_values[i].magnitude() > 0.01f) {
                valid_measured.push_back(measured_rgb_values[i]);
                valid_target.push_back(target_rgb[i]);

                // Perceptually-weighted patch selection based on colour-science best practices
                float weight = 1.0f;

                // Neutral patches (white to black series) - highest priority for white balance
                if (i >= 18 && i <= 23) {
                    // White and light grays get highest weight (most visually important)
                    if (i == 18)
                        weight = 5.0f; // White patch - critical for white balance
                    else if (i == 19)
                        weight = 4.0f; // Light gray - very important for tone mapping
                    else
                        weight = 3.0f; // Darker grays - important for contrast
                }
                // Primary color patches - important for color accuracy
                else if (i == 14 || i == 13 || i == 12)
                    weight = 3.0f; // Red, Green, Blue primaries

                // Skin tone approximates - patches that represent common skin tones
                else if (i == 3 || i == 10)
                    weight = 2.5f; // Foliage and Yellow Green (skin-like hues)

                // Well-lit patches get higher weight based on luminance
                helios::vec3 measured_rgb = measured_rgb_values[i];
                float luminance = 0.299f * measured_rgb.x + 0.587f * measured_rgb.y + 0.114f * measured_rgb.z;

                // Boost weight for brighter patches (they're more visually prominent)
                if (luminance > 0.6f)
                    weight *= 1.5f;
                else if (luminance < 0.2f)
                    weight *= 0.7f; // Reduce weight for very dark patches

                // Additional quality checks for measured RGB values
                // Reduce weight for patches that seem poorly measured (extreme values)
                if (measured_rgb.magnitude() > 1.2f || measured_rgb.magnitude() < 0.1f) {
                    weight *= 0.5f; // Reduce weight for suspicious measurements
                }

                patch_weights.push_back(weight);
            }
        }

        if (valid_measured.size() >= 6 && algorithm != ColorCorrectionAlgorithm::DIAGONAL_ONLY) {

            // Compute robustly regularized weighted least squares solution
            // Use adaptive regularization to prevent extreme matrix coefficients
            bool matrix_valid = true;

            // Try moderate regularization values - avoid extreme regularization that creates bad matrices
            std::vector<float> lambda_values = {0.01f, 0.05f, 0.1f, 0.15f, 0.2f};
            int lambda_attempt = 0;

            while (lambda_attempt < lambda_values.size()) {
                float lambda = lambda_values[lambda_attempt];
                matrix_valid = true;

                for (int row = 0; row < 3; row++) {
                    // Build weighted normal equations with adaptive regularization
                    float ATA[3][3] = {{0}}; // A^T * W * A + λI
                    float ATb[3] = {0}; // A^T * W * b

                    for (size_t i = 0; i < valid_measured.size(); i++) {
                        float weight = patch_weights[i];
                        helios::vec3 m = valid_measured[i]; // measured RGB
                        float target_val = (row == 0) ? valid_target[i].x : (row == 1) ? valid_target[i].y : valid_target[i].z;

                        // Update normal equations
                        ATA[0][0] += weight * m.x * m.x;
                        ATA[0][1] += weight * m.x * m.y;
                        ATA[0][2] += weight * m.x * m.z;
                        ATA[1][0] += weight * m.y * m.x;
                        ATA[1][1] += weight * m.y * m.y;
                        ATA[1][2] += weight * m.y * m.z;
                        ATA[2][0] += weight * m.z * m.x;
                        ATA[2][1] += weight * m.z * m.y;
                        ATA[2][2] += weight * m.z * m.z;

                        ATb[0] += weight * m.x * target_val;
                        ATb[1] += weight * m.y * target_val;
                        ATb[2] += weight * m.z * target_val;
                    }

                    // Add color-preserving regularization
                    // Stronger regularization on diagonal (preserves primary colors)
                    // Weaker regularization on off-diagonal (allows some color mixing)
                    float diag_reg = lambda * 2.0f; // Stronger on diagonal
                    float offdiag_reg = lambda * 0.5f; // Weaker on off-diagonal

                    ATA[0][0] += diag_reg; // Red preservation
                    ATA[1][1] += diag_reg; // Green preservation
                    ATA[2][2] += diag_reg; // Blue preservation

                    // Light off-diagonal regularization to prevent extreme color mixing
                    ATA[0][1] += offdiag_reg;
                    ATA[1][0] += offdiag_reg;
                    ATA[0][2] += offdiag_reg;
                    ATA[2][0] += offdiag_reg;
                    ATA[1][2] += offdiag_reg;
                    ATA[2][1] += offdiag_reg;

                    // Solve regularized 3x3 system using Cramer's rule
                    float det = ATA[0][0] * (ATA[1][1] * ATA[2][2] - ATA[1][2] * ATA[2][1]) - ATA[0][1] * (ATA[1][0] * ATA[2][2] - ATA[1][2] * ATA[2][0]) + ATA[0][2] * (ATA[1][0] * ATA[2][1] - ATA[1][1] * ATA[2][0]);

                    if (fabs(det) < 1e-3f) {
                        if (algorithm != ColorCorrectionAlgorithm::MATRIX_3X3_FORCE) {
                            matrix_valid = false;
                            break;
                        }
                    }

                    float inv_det = 1.0f / det;
                    correction_matrix[row][0] = inv_det * (ATb[0] * (ATA[1][1] * ATA[2][2] - ATA[1][2] * ATA[2][1]) - ATb[1] * (ATA[0][1] * ATA[2][2] - ATA[0][2] * ATA[2][1]) + ATb[2] * (ATA[0][1] * ATA[1][2] - ATA[0][2] * ATA[1][1]));

                    correction_matrix[row][1] = inv_det * (ATb[1] * (ATA[0][0] * ATA[2][2] - ATA[0][2] * ATA[2][0]) - ATb[0] * (ATA[1][0] * ATA[2][2] - ATA[1][2] * ATA[2][0]) + ATb[2] * (ATA[1][0] * ATA[0][2] - ATA[1][2] * ATA[0][0]));

                    correction_matrix[row][2] = inv_det * (ATb[2] * (ATA[0][0] * ATA[1][1] - ATA[0][1] * ATA[1][0]) - ATb[0] * (ATA[1][0] * ATA[2][1] - ATA[1][1] * ATA[2][0]) + ATb[1] * (ATA[0][0] * ATA[2][1] - ATA[0][1] * ATA[2][0]));
                }

                // Validate computed matrix elements are reasonable
                if (matrix_valid) {
                    bool elements_reasonable = true;
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            if (fabs(correction_matrix[i][j]) > 5.0f) {
                                if (algorithm != ColorCorrectionAlgorithm::MATRIX_3X3_FORCE) {
                                    elements_reasonable = false;
                                    break;
                                }
                            }
                        }
                        if (!elements_reasonable)
                            break;
                    }
                    matrix_valid = elements_reasonable;
                }

                if (matrix_valid) {
                    break; // Success!
                } else {
                    lambda_attempt++;
                }
            }

            if (!matrix_valid) {

                // Enhanced perceptually-weighted diagonal correction
                // Calculate weighted averages for each color channel
                float total_weight = 0.0f;
                helios::vec3 weighted_correction = make_vec3(0, 0, 0);

                for (size_t i = 0; i < valid_measured.size(); i++) {
                    float weight = patch_weights[i];
                    helios::vec3 measured = valid_measured[i];
                    helios::vec3 target = valid_target[i];

                    // Calculate per-channel correction factors
                    if (measured.x > 0.01f && measured.y > 0.01f && measured.z > 0.01f) {
                        helios::vec3 channel_correction = make_vec3(target.x / measured.x, target.y / measured.y, target.z / measured.z);

                        weighted_correction.x += weight * channel_correction.x;
                        weighted_correction.y += weight * channel_correction.y;
                        weighted_correction.z += weight * channel_correction.z;
                        total_weight += weight;
                    }
                }

                if (total_weight > 0.1f) {
                    // Apply weighted average correction factors
                    correction_matrix[0][0] = weighted_correction.x / total_weight;
                    correction_matrix[1][1] = weighted_correction.y / total_weight;
                    correction_matrix[2][2] = weighted_correction.z / total_weight;

                    // Apply conservative limits
                    correction_matrix[0][0] = std::max(0.5f, std::min(2.0f, correction_matrix[0][0]));
                    correction_matrix[1][1] = std::max(0.5f, std::min(2.0f, correction_matrix[1][1]));
                    correction_matrix[2][2] = std::max(0.5f, std::min(2.0f, correction_matrix[2][2]));
                }
            }

            if (!matrix_valid || algorithm == ColorCorrectionAlgorithm::DIAGONAL_ONLY) {
                // Use diagonal correction using white patch
                correction_matrix = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};

                // Enhanced perceptually-weighted diagonal correction using all valid patches
                // This is more robust than using just the white patch
                if (valid_measured.size() > 0 && patch_weights.size() == valid_measured.size()) {
                    float total_weight = 0.0f;
                    helios::vec3 weighted_measured_avg = make_vec3(0, 0, 0);
                    helios::vec3 weighted_target_avg = make_vec3(0, 0, 0);

                    // Compute weighted averages using perceptual patch weights
                    for (size_t i = 0; i < valid_measured.size(); i++) {
                        float weight = patch_weights[i];
                        weighted_measured_avg = weighted_measured_avg + weight * valid_measured[i];
                        weighted_target_avg = weighted_target_avg + weight * valid_target[i];
                        total_weight += weight;
                    }

                    if (total_weight > 0) {
                        weighted_measured_avg = weighted_measured_avg / total_weight;
                        weighted_target_avg = weighted_target_avg / total_weight;

                        if (weighted_measured_avg.x > 0.05f && weighted_measured_avg.y > 0.05f && weighted_measured_avg.z > 0.05f) {
                            correction_matrix[0][0] = weighted_target_avg.x / weighted_measured_avg.x;
                            correction_matrix[1][1] = weighted_target_avg.y / weighted_measured_avg.y;
                            correction_matrix[2][2] = weighted_target_avg.z / weighted_measured_avg.z;

                            // Apply conservative limits for stability
                            correction_matrix[0][0] = std::max(0.5f, std::min(2.0f, correction_matrix[0][0]));
                            correction_matrix[1][1] = std::max(0.5f, std::min(2.0f, correction_matrix[1][1]));
                            correction_matrix[2][2] = std::max(0.5f, std::min(2.0f, correction_matrix[2][2]));

                        } else {
                            // Fallback to original white-patch method
                            size_t white_idx = 18;
                            if (white_idx < measured_rgb_values.size() && white_idx < target_rgb.size()) {
                                helios::vec3 measured_white = measured_rgb_values[white_idx];
                                helios::vec3 target_white = target_rgb[white_idx];
                                if (measured_white.x > 0.05f && measured_white.y > 0.05f && measured_white.z > 0.05f) {
                                    correction_matrix[0][0] = std::max(0.5f, std::min(2.0f, target_white.x / measured_white.x));
                                    correction_matrix[1][1] = std::max(0.5f, std::min(2.0f, target_white.y / measured_white.y));
                                    correction_matrix[2][2] = std::max(0.5f, std::min(2.0f, target_white.z / measured_white.z));
                                }
                            }
                        }
                    }
                } else {
                    // Original simple approach as ultimate fallback
                    size_t white_idx = 18;
                    if (white_idx < measured_rgb_values.size() && white_idx < target_rgb.size()) {
                        helios::vec3 measured_white = measured_rgb_values[white_idx];
                        helios::vec3 target_white = target_rgb[white_idx];
                        if (measured_white.x > 0.05f && measured_white.y > 0.05f && measured_white.z > 0.05f) {
                            correction_matrix[0][0] = std::max(0.5f, std::min(2.0f, target_white.x / measured_white.x));
                            correction_matrix[1][1] = std::max(0.5f, std::min(2.0f, target_white.y / measured_white.y));
                            correction_matrix[2][2] = std::max(0.5f, std::min(2.0f, target_white.z / measured_white.z));
                        }
                    }
                }
            }
        } else if (algorithm == ColorCorrectionAlgorithm::DIAGONAL_ONLY) {
            // Apply diagonal correction using white patch
            size_t white_idx = 18;
            if (white_idx < measured_rgb_values.size() && white_idx < target_rgb.size() && measured_rgb_values[white_idx].magnitude() > 0) {

                helios::vec3 measured_white = measured_rgb_values[white_idx];
                helios::vec3 target_white = target_rgb[white_idx];

                if (measured_white.x > 0.05f && measured_white.y > 0.05f && measured_white.z > 0.05f) {
                    correction_matrix[0][0] = target_white.x / measured_white.x;
                    correction_matrix[1][1] = target_white.y / measured_white.y;
                    correction_matrix[2][2] = target_white.z / measured_white.z;

                    // Apply limits
                    correction_matrix[0][0] = std::max(0.5f, std::min(2.0f, correction_matrix[0][0]));
                    correction_matrix[1][1] = std::max(0.5f, std::min(2.0f, correction_matrix[1][1]));
                    correction_matrix[2][2] = std::max(0.5f, std::min(2.0f, correction_matrix[2][2]));
                }
            }
        } else {
            std::cout << "Insufficient valid patches (" << valid_measured.size() << " available), using identity matrix" << std::endl;
        }
    } else if (algorithm == ColorCorrectionAlgorithm::DIAGONAL_ONLY && measured_lab_values.size() > 0) {
        // Apply diagonal correction using white patch
        size_t white_idx = 18;
        if (white_idx < measured_rgb_values.size() && white_idx < reference_lab_values.size() && measured_rgb_values[white_idx].magnitude() > 0) {

            CameraCalibration::LabColor ref_lab = reference_lab_values[white_idx];
            helios::vec3 target_white = calibration.labToRgb(ref_lab);
            helios::vec3 measured_white = measured_rgb_values[white_idx];

            if (measured_white.x > 0.05f && measured_white.y > 0.05f && measured_white.z > 0.05f) {
                correction_matrix[0][0] = target_white.x / measured_white.x;
                correction_matrix[1][1] = target_white.y / measured_white.y;
                correction_matrix[2][2] = target_white.z / measured_white.z;

                // Apply limits
                correction_matrix[0][0] = std::max(0.5f, std::min(2.0f, correction_matrix[0][0]));
                correction_matrix[1][1] = std::max(0.5f, std::min(2.0f, correction_matrix[1][1]));
                correction_matrix[2][2] = std::max(0.5f, std::min(2.0f, correction_matrix[2][2]));
            }
        }
    } else {
        std::cout << "Insufficient patches for correction (" << measured_lab_values.size() << " available), using identity matrix" << std::endl;
    }

    // Generate quality of fit report if requested
    if (print_quality_report) {
        std::cout << "\n========== COLOR CALIBRATION QUALITY REPORT ==========" << std::endl;
        std::cout << "Colorboard types: ";
        for (size_t i = 0; i < colorboard_types.size(); i++) {
            std::cout << colorboard_types[i];
            if (i < colorboard_types.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
        std::cout << "Number of patches analyzed: " << visible_patches << std::endl;
        std::cout << "Algorithm used: " << algorithm_name << std::endl;

        // Display matrix conditioning information
        bool is_diagonal_only = true;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j && fabs(correction_matrix[i][j]) > 1e-6f) {
                    is_diagonal_only = false;
                    break;
                }
            }
            if (!is_diagonal_only)
                break;
        }

        if (is_diagonal_only) {
            std::cout << "Color correction factors applied: R=" << correction_matrix[0][0] << ", G=" << correction_matrix[1][1] << ", B=" << correction_matrix[2][2] << std::endl;
            std::cout << "Matrix type: Diagonal (white balance only)" << std::endl;
        } else {
            std::cout << "Full 3x3 color correction matrix applied:" << std::endl;
            for (int i = 0; i < 3; i++) {
                std::cout << "[" << std::fixed << std::setprecision(4);
                for (int j = 0; j < 3; j++) {
                    std::cout << std::setw(8) << correction_matrix[i][j];
                    if (j < 2)
                        std::cout << " ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "Matrix type: Full 3x3 (corrects color casts and chromatic errors)" << std::endl;

            // Calculate matrix determinant for conditioning info
            float det = correction_matrix[0][0] * (correction_matrix[1][1] * correction_matrix[2][2] - correction_matrix[1][2] * correction_matrix[2][1]) -
                        correction_matrix[0][1] * (correction_matrix[1][0] * correction_matrix[2][2] - correction_matrix[1][2] * correction_matrix[2][0]) +
                        correction_matrix[0][2] * (correction_matrix[1][0] * correction_matrix[2][1] - correction_matrix[1][1] * correction_matrix[2][0]);

            std::cout << "Matrix determinant: " << std::scientific << std::setprecision(3) << det << std::endl;
            if (fabs(det) > 0.1f) {
                std::cout << "Matrix conditioning: Good (well-conditioned)" << std::endl;
            } else if (fabs(det) > 0.01f) {
                std::cout << "Matrix conditioning: Fair (moderately conditioned)" << std::endl;
            } else {
                std::cout << "Matrix conditioning: Poor (ill-conditioned)" << std::endl;
            }
            std::cout << std::fixed; // Reset formatting
        }

        // Calculate and display quality metrics for each patch
        double total_delta_e = 0.0;
        int valid_patches = 0;

        std::cout << "\nPer-patch analysis (after correction):" << std::endl;
        std::cout << "Patch | Corrected RGB       | Reference RGB       | Delta E " << std::endl;
        std::cout << "------|--------------------|--------------------|---------" << std::endl;

        for (size_t i = 0; i < std::min(measured_rgb_values.size(), reference_lab_values.size()); i++) {
            if (measured_rgb_values[i].magnitude() > 0) {
                // Apply color correction to measured RGB values (with optional affine terms)
                helios::vec3 measured_rgb = measured_rgb_values[i];
                float corrected_r = correction_matrix[0][0] * measured_rgb.x + correction_matrix[0][1] * measured_rgb.y + correction_matrix[0][2] * measured_rgb.z;
                float corrected_g = correction_matrix[1][0] * measured_rgb.x + correction_matrix[1][1] * measured_rgb.y + correction_matrix[1][2] * measured_rgb.z;
                float corrected_b = correction_matrix[2][0] * measured_rgb.x + correction_matrix[2][1] * measured_rgb.y + correction_matrix[2][2] * measured_rgb.z;


                // Clamp corrected values to [0,1] - DISABLED FOR TESTING
                // corrected_r = std::max(0.0f, std::min(1.0f, corrected_r));
                // corrected_g = std::max(0.0f, std::min(1.0f, corrected_g));
                // corrected_b = std::max(0.0f, std::min(1.0f, corrected_b));

                helios::vec3 corrected_rgb = make_vec3(corrected_r, corrected_g, corrected_b);

                // Convert corrected RGB to Lab
                CameraCalibration::LabColor corrected_lab = calibration.rgbToLab(corrected_rgb);
                CameraCalibration::LabColor reference_lab = reference_lab_values[i];

                // Calculate Delta E between corrected and reference
                // Use ΔE2000 for better perceptual color difference assessment
                double delta_E = calibration.deltaE2000(corrected_lab, reference_lab);

                std::cout << std::setw(5) << i << " | " << std::fixed << std::setprecision(3) << "(" << std::setw(5) << corrected_rgb.x << "," << std::setw(5) << corrected_rgb.y << "," << std::setw(5) << corrected_rgb.z << ") | ";

                helios::vec3 ref_rgb = calibration.labToRgb(reference_lab);
                std::cout << "(" << std::setw(5) << ref_rgb.x << "," << std::setw(5) << ref_rgb.y << "," << std::setw(5) << ref_rgb.z << ") | " << std::setw(7) << delta_E << std::endl;

                total_delta_e += delta_E;
                valid_patches++;
            }
        }

        // Overall statistics
        double mean_delta_e = total_delta_e / valid_patches;
        std::cout << "\n========== OVERALL CALIBRATION QUALITY ==========" << std::endl;
        std::cout << "Mean Delta E: " << std::fixed << std::setprecision(2) << mean_delta_e << std::endl;

        std::cout << "======================================================\n" << std::endl;
    }

    // Step 7: Apply correction to entire image with same pixel ordering as writeCameraImage
    std::vector<helios::RGBcolor> corrected_pixels;
    corrected_pixels.resize(red_data.size());

    // Apply correction using the same pixel transformation as writeCameraImage uses
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            // Get pixel from source data (no flip)
            int source_index = j * camera_resolution.x + i;
            float r = red_data[source_index];
            float g = green_data[source_index];
            float b = blue_data[source_index];

            // Apply correction matrix (with optional affine terms)
            float corrected_r = correction_matrix[0][0] * r + correction_matrix[0][1] * g + correction_matrix[0][2] * b;
            float corrected_g = correction_matrix[1][0] * r + correction_matrix[1][1] * g + correction_matrix[1][2] * b;
            float corrected_b = correction_matrix[2][0] * r + correction_matrix[2][1] * g + correction_matrix[2][2] * b;


            // Clamp values to [0,1] - DISABLED FOR TESTING
            // corrected_r = std::max(0.0f, std::min(1.0f, corrected_r));
            // corrected_g = std::max(0.0f, std::min(1.0f, corrected_g));
            // corrected_b = std::max(0.0f, std::min(1.0f, corrected_b));

            // Apply same coordinate transformation as writeCameraImage
            uint ii = camera_resolution.x - i - 1; // Horizontal flip
            uint jj = camera_resolution.y - j - 1; // Vertical flip
            uint dest_index = jj * camera_resolution.x + ii;

            corrected_pixels[dest_index] = make_RGBcolor(corrected_r, corrected_g, corrected_b);
        }
    }

    // Step 8: Write corrected image using writeJPEG
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

    // Export CCM to XML file if requested
    if (!ccm_export_file_path.empty()) {
        try {
            // Calculate quality metrics for export (even if not printed)
            double total_delta_e = 0.0;
            int valid_patches = 0;

            for (size_t i = 0; i < std::min(measured_rgb_values.size(), reference_lab_values.size()); i++) {
                if (measured_rgb_values[i].magnitude() > 0) {
                    // Apply color correction to measured RGB values
                    helios::vec3 measured_rgb = measured_rgb_values[i];
                    float corrected_r = correction_matrix[0][0] * measured_rgb.x + correction_matrix[0][1] * measured_rgb.y + correction_matrix[0][2] * measured_rgb.z;
                    float corrected_g = correction_matrix[1][0] * measured_rgb.x + correction_matrix[1][1] * measured_rgb.y + correction_matrix[1][2] * measured_rgb.z;
                    float corrected_b = correction_matrix[2][0] * measured_rgb.x + correction_matrix[2][1] * measured_rgb.y + correction_matrix[2][2] * measured_rgb.z;

                    helios::vec3 corrected_rgb = make_vec3(corrected_r, corrected_g, corrected_b);

                    // Convert corrected RGB to Lab
                    CameraCalibration::LabColor corrected_lab = calibration.rgbToLab(corrected_rgb);
                    CameraCalibration::LabColor reference_lab = reference_lab_values[i];

                    // Calculate Delta E between corrected and reference
                    double delta_E = calibration.deltaE2000(corrected_lab, reference_lab);
                    total_delta_e += delta_E;
                    valid_patches++;
                }
            }

            double mean_delta_e = (valid_patches > 0) ? (total_delta_e / valid_patches) : -1.0;

            // Export CCM to XML file
            // Concatenate all colorboard types into a single string
            std::string colorboard_types_str;
            for (size_t i = 0; i < colorboard_types.size(); i++) {
                colorboard_types_str += colorboard_types[i];
                if (i < colorboard_types.size() - 1) {
                    colorboard_types_str += ", ";
                }
            }
            exportColorCorrectionMatrixXML(ccm_export_file_path, camera_label, correction_matrix, output_path, colorboard_types_str, (float) mean_delta_e);

            std::cout << "Exported color correction matrix to: " << ccm_export_file_path << std::endl;
        } catch (const std::exception &e) {
            helios_runtime_error("ERROR (RadiationModel::autoCalibrateCameraImage): Failed to export CCM to XML. " + std::string(e.what()));
        }
    }

    return output_path;
}

void RadiationModel::applyCameraColorCorrectionMatrix(const std::string &camera_label, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, const std::string &ccm_file_path) {

    // Step 1: Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraColorCorrectionMatrix): Camera '" + camera_label + "' does not exist. Make sure the camera was added to the radiation model.");
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

std::vector<float> RadiationModel::getCameraPixelData(const std::string &camera_label, const std::string &band_label) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraPixelData): Camera '" + camera_label + "' does not exist.");
    }

    auto &camera_pixel_data = cameras.at(camera_label).pixel_data;
    if (camera_pixel_data.find(band_label) == camera_pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraPixelData): Band '" + band_label + "' does not exist in camera '" + camera_label + "'.");
    }

    return camera_pixel_data.at(band_label);
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
    params.camera_HFOV = effective_HFOV * M_PI / 180.0f;
    params.camera_viewplane_length = 0.5f / tanf(0.5f * effective_HFOV * M_PI / 180.f);

    // Compute pixel solid angle
    float HFOV_rad = effective_HFOV * M_PI / 180.f;
    float VFOV_rad = HFOV_rad / camera.FOV_aspect_ratio;
    float pixel_angle_h = HFOV_rad / float(camera.resolution.x);
    float pixel_angle_v = VFOV_rad / float(camera.resolution.y);
    params.camera_pixel_solid_angle = pixel_angle_h * pixel_angle_v;

    // Explicitly set scattering iteration for cameras (always iteration 0 for specular)
    params.scattering_iteration = 0;

    // Set specular reflection mode from auto-detection
    params.specular_reflection_enabled = specular_reflection_mode;

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

    // Filter out invalid/zero-area primitives (same as old updateGeometry)
    std::vector<uint> valid_UUIDs;
    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID))
            continue;

        float area = context->getPrimitiveArea(UUID);
        uint parentID = context->getPrimitiveParentObjectID(UUID);
        if ((area == 0 || std::isnan(area)) && context->getObjectType(parentID) != helios::OBJECT_TYPE_TILE) {
            continue;
        }
        valid_UUIDs.push_back(UUID);
    }

    if (valid_UUIDs.empty()) {
        geometry_data = helios::RayTracingGeometry(); // Empty geometry
        return;
    }

    // Reorder primitives by parent object (same ordering as old code)
    std::vector<uint> objID_all = context->getUniquePrimitiveParentObjectIDs(valid_UUIDs, true);
    std::vector<uint> primitive_UUIDs_ordered;
    std::unordered_set<uint> valid_set(valid_UUIDs.begin(), valid_UUIDs.end());

    for (uint objID: objID_all) {
        std::vector<uint> prim_UUIDs = context->getObjectPrimitiveUUIDs(objID);
        if (objID == 0) {
            // Standalone primitives (parentID=0) come from unordered_map iteration,
            // which has non-deterministic ordering. Sort by UUID for reproducibility.
            std::sort(prim_UUIDs.begin(), prim_UUIDs.end());
        }
        for (uint UUID: prim_UUIDs) {
            if (context->doesPrimitiveExist(UUID) && valid_set.find(UUID) != valid_set.end()) {
                primitive_UUIDs_ordered.push_back(UUID);
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
    geometry_data.voxels.vertices.clear();
    geometry_data.voxels.UUIDs.clear();
    geometry_data.bboxes.vertices.clear();
    geometry_data.bboxes.UUIDs.clear();

    // Track object IDs for compound objects
    uint current_objID = 0;
    uint last_parentID = 99999;

    std::vector<uint> primitiveID_indices; // Maps primitives to their "object" index

    for (size_t u = 0; u < Nprimitives; u++) {
        uint UUID = primitive_UUIDs_ordered[u];
        uint parentID = context->getPrimitiveParentObjectID(UUID);

        if (last_parentID != parentID || parentID == 0 || context->getObjectType(parentID) == helios::OBJECT_TYPE_TILE) {
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
    size_t patch_idx = 0, tri_idx = 0, disk_idx = 0, voxel_idx = 0, bbox_idx = 0;

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

        // Two-sided flag
        geometry_data.twosided_flags[prim_idx] = context->getPrimitiveTwosidedFlag(UUID, 1) ? 1 : 0;

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

        } else if (type == helios::PRIMITIVE_TYPE_VOXEL) {
            geometry_data.primitive_types[prim_idx] = 4; // voxel

            context->getPrimitiveTransformationMatrix(UUID, m);
            memcpy(&geometry_data.transform_matrices[prim_idx * 16], m, 16 * sizeof(float));

            std::vector<vec3> verts = context->getPrimitiveVertices(UUID);
            for (const auto &v: verts) {
                geometry_data.voxels.vertices.push_back(v);
            }

            geometry_data.object_subdivisions[prim_idx] = helios::make_int2(1, 1);
            geometry_data.voxels.UUIDs.push_back(UUID); // Store actual UUID, not position
            voxel_idx++;
        }
    }

    // Set counts
    geometry_data.patch_count = patch_idx;
    geometry_data.triangle_count = tri_idx;
    geometry_data.disk_count = disk_idx;
    geometry_data.tile_count = 0; // Tile subpatches are treated as individual patches
    geometry_data.voxel_count = voxel_idx;

    // ========== Periodic Boundary Bboxes ==========
    // Create bbox geometry for periodic boundary conditions
    // Each bbox face is a rectangular boundary at domain edge

    // Get domain bounding box
    vec2 xbounds, ybounds, zbounds;
    context->getDomainBoundingBox(xbounds, ybounds, zbounds);

    // Validate camera positions if periodic boundaries enabled
    if (periodic_flag.x == 1 || periodic_flag.y == 1) {
        if (!cameras.empty()) {
            for (auto &camera: cameras) {
                vec3 camerapos = camera.second.position;
                if (camerapos.x < xbounds.x || camerapos.x > xbounds.y || camerapos.y < ybounds.x || camerapos.y > ybounds.y) {
                    std::cout << "WARNING (RadiationModel::buildGeometryData): camera position is outside of the domain bounding box. Disabling periodic boundary conditions." << std::endl;
                    periodic_flag.x = 0;
                    periodic_flag.y = 0;
                    break;
                }
                // Extend z-bounds to include camera
                if (camerapos.z < zbounds.x) {
                    zbounds.x = camerapos.z;
                }
                if (camerapos.z > zbounds.y) {
                    zbounds.y = camerapos.z;
                }
            }
        }
    }

    // Expand bounds slightly to ensure bbox faces are outside geometry
    xbounds.x -= 1e-5;
    xbounds.y += 1e-5;
    ybounds.x -= 1e-5;
    ybounds.y += 1e-5;
    zbounds.x -= 1e-5;
    zbounds.y += 1e-5;

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

void RadiationModel::buildTextureData() {
    // Extract texture mask and UV data for all primitives with transparency textures

    size_t Nobjects = geometry_data.primitive_count;

    // Clear any previous texture data (important when updateGeometry is called multiple times)
    geometry_data.mask_data.clear();
    geometry_data.mask_sizes.clear();
    geometry_data.uv_data.clear();

    // Initialize with -1 (no texture)
    geometry_data.mask_IDs.clear();
    geometry_data.mask_IDs.resize(Nobjects, -1);
    geometry_data.uv_IDs.clear();
    geometry_data.uv_IDs.resize(Nobjects, -1);

    // Cache to avoid duplicate mask data for primitives using the same texture file
    std::map<std::string, int> texture_to_mask_idx;

    for (size_t prim_idx = 0; prim_idx < Nobjects; prim_idx++) {
        uint UUID = geometry_data.primitive_UUIDs[prim_idx];

        // Check if primitive has a texture file
        std::string texture_file = context->getPrimitiveTextureFile(UUID);
        if (texture_file.empty()) {
            continue; // No texture - mask_ID stays -1
        }

        // Check if texture has transparency channel (alpha)
        if (!context->primitiveTextureHasTransparencyChannel(UUID)) {
            continue; // No transparency - mask_ID stays -1 (e.g., JPEG files)
        }

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

        // Extract UV coordinates for this primitive
        // uv_IDs stores the position index (not offset), used to access uvdata[vertex][position] in CUDA
        std::vector<helios::vec2> uvs = context->getPrimitiveTextureUV(UUID);
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

static void validateAndCorrectMaterialProperties(float &rho, float &tau, float eps, bool emission_enabled, uint scattering_depth, const std::string &band_label, uint UUID, bool is_sif_band = false,
                                                  helios::WarningAggregator *warnings = nullptr) {
    // Helper function to enforce energy conservation constraints on material properties
    // Mirrors the validation logic from updateRadiativeProperties() (lines 2672-2686)

    // 1. Clamp rho and tau to [0,1] with warnings for out-of-range values
    if (rho < 0.f || rho > 1.f) {
        if (warnings) {
            warnings->addWarning("material_property_clamping", "Reflectivity out of range [0,1] for band " + band_label + ", primitive #" + std::to_string(UUID) + ": rho=" + std::to_string(rho) + ". Clamping to valid range.");
        }
        rho = std::max(0.f, std::min(1.f, rho));
    }

    if (tau < 0.f || tau > 1.f) {
        if (warnings) {
            warnings->addWarning("material_property_clamping", "Transmissivity out of range [0,1] for band " + band_label + ", primitive #" + std::to_string(UUID) + ": tau=" + std::to_string(tau) + ". Clamping to valid range.");
        }
        tau = std::max(0.f, std::min(1.f, tau));
    }

    // SIF-flagged bands bypass the Stefan-Boltzmann ε+ρ+τ=1 conservation constraint
    // because their emission is sourced from the Fluspect-B per-leaf kernel (see
    // computeSIFEmission) rather than ε·σ·T⁴. Epsilon is not consulted for SIF bands.
    // We still enforce the non-emission-style constraint ρ+τ ≤ 1 (physically required
    // regardless of the emission mechanism).
    if (is_sif_band) {
        if (rho + tau > 1.f) {
            helios_runtime_error(std::string("ERROR (RadiationModel): reflectivity and transmissivity must sum to less than or equal to 1 to ensure energy conservation. Band ") + band_label + ", Primitive #" +
                                 std::to_string(UUID) + ": tau=" + std::to_string(tau) + ", rho=" + std::to_string(rho) + ".");
        }
        return;
    }

    // 2. Apply emission-specific constraints
    if (emission_enabled) {
        // Special case: blackbody emission (scatteringDepth=0 requires eps=1, rho=0, tau=0)
        if (scattering_depth == 0 && eps != 1.f) {
            if (warnings && (rho != 0.f || tau != 0.f)) {
                warnings->addWarning("blackbody_override", "Band " + band_label + " has emission with scatteringDepth=0, " + "enforcing blackbody behavior (eps=1, rho=0, tau=0) for primitive #" + std::to_string(UUID));
            }
            rho = 0.f;
            tau = 0.f;
        }
        // General emission case: check energy conservation (eps + rho + tau = 1)
        else if (eps != 1.f && rho == 0 && tau == 0) {
            // Auto-correct: set rho = 1 - eps
            rho = 1.f - eps;
        } else if (std::abs(eps + rho + tau - 1.f) > 1e-5f && eps > 0.f) {
            // Cannot auto-correct, throw error
            helios_runtime_error(std::string("ERROR (RadiationModel): emissivity, transmissivity, and reflectivity ") + "must sum to 1 to ensure energy conservation. Band " + band_label + ", Primitive #" + std::to_string(UUID) +
                                 ": eps=" + std::to_string(eps) + ", tau=" + std::to_string(tau) + ", rho=" + std::to_string(rho) + ". It is also possible that you forgot to disable emission for this band.");
        }
    } else {
        // 3. Non-emission case: rho + tau must be ≤ 1
        if (rho + tau > 1.f) {
            helios_runtime_error(std::string("ERROR (RadiationModel): transmissivity and reflectivity cannot sum to ") + "greater than 1 to ensure energy conservation. Band " + band_label + ", Primitive #" + std::to_string(UUID) +
                                 ": eps=" + std::to_string(eps) + ", tau=" + std::to_string(tau) + ", rho=" + std::to_string(rho) + ". It is also possible that you forgot to disable emission for this band.");
        }
    }
}

void RadiationModel::buildMaterialData() {
    // Build backend-agnostic material data from Context primitive data

    // Warning aggregator for energy conservation issues
    helios::WarningAggregator warnings;

    size_t Nprims = geometry_data.primitive_count;
    size_t Nbands = radiation_bands.size();
    size_t Nsources = radiation_sources.size();

    material_data.num_primitives = Nprims;
    material_data.num_bands = Nbands;
    material_data.num_sources = Nsources;
    material_data.num_cameras = cameras.size();

    // Allocate arrays (indexed as [source][primitive][band] using MaterialPropertyIndexer)
    // NOTE: Bboxes don't need material properties (they only wrap rays for periodic boundaries)
    size_t total_size = Nsources * Nbands * Nprims;
    material_data.reflectivity.resize(total_size, 0.0f);
    material_data.transmissivity.resize(total_size, 0.0f);
    material_data.specular_exponent.resize(Nprims, -1.0f); // Default -1 means disabled
    material_data.specular_scale.resize(Nprims, 0.0f);

    // Create indexer for material properties: [source][primitive][band]
    MaterialPropertyIndexer mat_indexer(Nsources, Nprims, Nbands);

    // Cache unique spectral data to avoid redundant loads
    std::map<std::string, std::vector<helios::vec2>> unique_rho_spectra;
    std::map<std::string, std::vector<helios::vec2>> unique_tau_spectra;

    for (size_t p = 0; p < Nprims; p++) {
        uint UUID = geometry_data.primitive_UUIDs[p];

        // Cache reflectivity spectra
        if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
            std::string spectrum_label;
            context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);
            if (unique_rho_spectra.find(spectrum_label) == unique_rho_spectra.end()) {
                // Only load if spectrum exists in global data
                if (context->doesGlobalDataExist(spectrum_label.c_str())) {
                    unique_rho_spectra[spectrum_label] = loadSpectralData(spectrum_label);
                }
            }
        }

        // Cache transmissivity spectra
        if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum")) {
            std::string spectrum_label;
            context->getPrimitiveData(UUID, "transmissivity_spectrum", spectrum_label);
            if (unique_tau_spectra.find(spectrum_label) == unique_tau_spectra.end()) {
                // Only load if spectrum exists in global data
                if (context->doesGlobalDataExist(spectrum_label.c_str())) {
                    unique_tau_spectra[spectrum_label] = loadSpectralData(spectrum_label);
                }
            }
        }
    }

    // Extract material properties from Context primitives
    size_t b_idx = 0;
    for (const auto &band_pair: radiation_bands) {
        std::string band_label = band_pair.second.label;

        for (size_t s = 0; s < Nsources; s++) {
            for (size_t p = 0; p < Nprims; p++) {
                uint UUID = geometry_data.primitive_UUIDs[p];

                // Use BufferIndexer for safe, verifiable indexing
                // Note: p is already the array position, so we use p directly (not UUID)
                size_t idx = mat_indexer(s, p, b_idx);

                // Get reflectivity - try spectrum first, then per-band label
                float rho = rho_default;

                if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
                    // Spectrum-based reflectivity
                    std::string spectrum_label;
                    context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);

                    // Get spectrum from cache
                    if (unique_rho_spectra.find(spectrum_label) != unique_rho_spectra.end()) {
                        const std::vector<helios::vec2> &spectrum = unique_rho_spectra.at(spectrum_label);

                        // Get band wavelength bounds
                        helios::vec2 wavebounds = band_pair.second.wavebandBounds;

                        // Only require wavelength bounds if band performs scattering/absorption
                        // Emission-only bands (scatteringDepth==0) use Stefan-Boltzmann and don't need spectral integration
                        // Ray launches for emission don't require wavelength bounds since emission properties are wavelength-independent
                        bool needs_spectral_integration = (band_pair.second.scatteringDepth > 0);

                        if (needs_spectral_integration && wavebounds.x == 0 && wavebounds.y == 0) {
                            helios_runtime_error("ERROR (RadiationModel::buildMaterialData): Band '" + band_label + "' has no wavelength bounds - required for spectral integration");
                        }

                        // Integrate spectrum over band wavelength range (only if bounds are defined)
                        if (wavebounds.x != 0 || wavebounds.y != 0) {
                            if (!radiation_sources[s].source_spectrum.empty()) {
                                // Weight by source spectrum
                                rho = integrateSpectrum(s, spectrum, wavebounds.x, wavebounds.y);
                            } else {
                                // Uniform integration (divide by wavelength range to normalize)
                                rho = integrateSpectrum(spectrum, wavebounds.x, wavebounds.y) / (wavebounds.y - wavebounds.x);
                            }
                        }
                        // else: emission-only band, rho remains at default value (should be 0 for blackbody)
                    }
                } else {
                    // Per-band reflectivity (backward compatibility)
                    std::string rho_label = "reflectivity_" + band_label;
                    if (context->doesPrimitiveDataExist(UUID, rho_label.c_str())) {
                        context->getPrimitiveData(UUID, rho_label.c_str(), rho);
                    }
                }

                // Get transmissivity - try spectrum first, then per-band label
                float tau = tau_default;

                if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum")) {
                    // Spectrum-based transmissivity
                    std::string spectrum_label;
                    context->getPrimitiveData(UUID, "transmissivity_spectrum", spectrum_label);

                    // Get spectrum from cache
                    if (unique_tau_spectra.find(spectrum_label) != unique_tau_spectra.end()) {
                        const std::vector<helios::vec2> &spectrum = unique_tau_spectra.at(spectrum_label);

                        // Get band wavelength bounds
                        helios::vec2 wavebounds = band_pair.second.wavebandBounds;

                        // Only require wavelength bounds if band performs scattering/absorption
                        // Emission-only bands (scatteringDepth==0) use Stefan-Boltzmann and don't need spectral integration
                        // Ray launches for emission don't require wavelength bounds since emission properties are wavelength-independent
                        bool needs_spectral_integration = (band_pair.second.scatteringDepth > 0);

                        if (needs_spectral_integration && wavebounds.x == 0 && wavebounds.y == 0) {
                            helios_runtime_error("ERROR (RadiationModel::buildMaterialData): Band '" + band_label + "' has no wavelength bounds - required for spectral integration");
                        }

                        // Integrate spectrum over band wavelength range (only if bounds are defined)
                        if (wavebounds.x != 0 || wavebounds.y != 0) {
                            if (!radiation_sources[s].source_spectrum.empty()) {
                                // Weight by source spectrum
                                tau = integrateSpectrum(s, spectrum, wavebounds.x, wavebounds.y);
                            } else {
                                // Uniform integration
                                tau = integrateSpectrum(spectrum, wavebounds.x, wavebounds.y) / (wavebounds.y - wavebounds.x);
                            }
                        }
                        // else: emission-only band, tau remains at default value (should be 0 for blackbody)
                    }
                } else {
                    // Per-band transmissivity (backward compatibility)
                    std::string tau_label = "transmissivity_" + band_label;
                    if (context->doesPrimitiveDataExist(UUID, tau_label.c_str())) {
                        context->getPrimitiveData(UUID, tau_label.c_str(), tau);
                    }
                }

                // Get emissivity for validation
                float eps = eps_default;
                std::string eps_label = "emissivity_" + band_label;
                if (context->doesPrimitiveDataExist(UUID, eps_label.c_str())) {
                    context->getPrimitiveData(UUID, eps_label.c_str(), eps);
                }

                // Validate and correct material properties to ensure energy conservation.
                // SIF-flagged bands skip the Stefan-Boltzmann ε+ρ+τ=1 check because their
                // emission is sourced from Fluspect-B, not ε·σ·T⁴.
                const RadiationBand &band = band_pair.second;
                const bool is_sif_band = sif_emission_bands.count(band_label) > 0;
                validateAndCorrectMaterialProperties(rho, tau, eps, band.emissionFlag, band.scatteringDepth, band_label, UUID, is_sif_band, &warnings);

                // Store validated properties
                material_data.reflectivity[idx] = rho;
                material_data.transmissivity[idx] = tau;
            }
        }
        b_idx++;
    }

    // NOTE: Bboxes don't need material properties - they only wrap rays for periodic boundaries
    // Material buffers are sized for real primitives only (Nprims), not including bboxes

    // Load specular reflection properties from primitive data
    bool specular_exponent_specified = false;
    bool specular_scale_specified = false;

    for (size_t p = 0; p < Nprims; p++) {
        uint UUID = geometry_data.primitive_UUIDs[p];

        if (context->doesPrimitiveDataExist(UUID, "specular_exponent") && context->getPrimitiveDataType("specular_exponent") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "specular_exponent", material_data.specular_exponent.at(p));
            if (material_data.specular_exponent.at(p) >= 0.f) {
                specular_exponent_specified = true;
            }
        }

        if (context->doesPrimitiveDataExist(UUID, "specular_scale") && context->getPrimitiveDataType("specular_scale") == helios::HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "specular_scale", material_data.specular_scale.at(p));
            if (material_data.specular_scale.at(p) > 0.f) {
                specular_scale_specified = true;
            }
        }
    }

    // Auto-enable specular reflection if specular properties are specified on any primitive
    if (specular_exponent_specified) {
        if (specular_scale_specified) {
            specular_reflection_mode = 2; // Mode 2: use primitive specular_scale
        } else {
            specular_reflection_mode = 1; // Mode 1: use default 0.25 scale
        }
    } else {
        specular_reflection_mode = 0; // Disabled
    }

    // Report any accumulated warnings
    warnings.report();
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

        // Flatten flux arrays - use getSourceFlux() to handle -1.f sentinel values
        backend_src.fluxes.clear();
        backend_src.fluxes_cam.clear();
        for (const auto &band_pair: radiation_bands) {
            std::string band_label = band_pair.second.label;
            // Use getSourceFlux() which properly handles -1.f sentinel (returns 0 or integrates spectrum)
            float flux = getSourceFlux(s, band_label);
            backend_src.fluxes.push_back(flux);
            backend_src.fluxes_cam.push_back(flux); // Same for now
        }

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
    buildMaterialData();
    buildSourceData();
}
