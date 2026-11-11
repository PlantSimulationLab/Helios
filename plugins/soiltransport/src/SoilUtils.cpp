/** \file "SoilUtils.cpp" Implementation of soil utility functions.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilUtils.h"
#include "global.h"
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sched.h>
#endif

// ============================================================================
// Thread count
// ============================================================================

int physicalCoreCount() {
    static const int count = [] {
#if defined(_WIN32)
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        std::vector<char> buf(len);
        if (len == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
            return 0;
        int cores = 0;
        for (DWORD off = 0; off < len; off += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off)->Size)
            cores++;
        return cores;
#elif defined(__APPLE__)
        int cores = 0;
        size_t size = sizeof(cores);
        return (sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) == 0) ? cores : 0;
#elif defined(__linux__)
        // Distinct (package, core) pairs among the CPUs this process may run on
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
            return 0;
        std::set<std::pair<int, int>> cores;
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if (!CPU_ISSET(cpu, &mask))
                continue;
            const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
            std::ifstream fpkg(dir + "physical_package_id"), fcore(dir + "core_id");
            int pkg = -1, core = -1;
            if (!(fpkg >> pkg) || !(fcore >> core))
                return 0;
            cores.insert({pkg, core});
        }
        return static_cast<int>(cores.size());
#else
        return 0;
#endif
    }();
    return count;
}

int resolveThreadCount(int requested) {
    if (requested > 0)
        return requested;
#ifdef _OPENMP
    const int omp_max = std::max(1, omp_get_max_threads());
#else
    const int omp_max = 1;
#endif
    if (std::getenv("OMP_NUM_THREADS") != nullptr) // an explicit user/environment choice wins
        return omp_max;
    const int cores = physicalCoreCount();
    return (cores > 0) ? std::min(omp_max, cores) : omp_max;
}

// ============================================================================
// Thomas algorithm implementation
// ============================================================================

void thomasSolve(float *a, float *b, float *c, float *d, float *x, uint n) {
    // Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    // Note: a[0] and c[n-1] are not used
    // WARNING: Modifies c and d in-place!

    if (n == 0)
        return;
    if (n == 1) {
        x[0] = d[0] / b[0];
        return;
    }

    uint nm1 = n - 1;

    // Forward sweep - modify c and d
    c[0] /= b[0];
    d[0] /= b[0];

    for (uint i = 1; i < nm1; i++) {
        float denom = b[i] - a[i] * c[i - 1];
        c[i] /= denom;
        d[i] = (d[i] - a[i] * d[i - 1]) / denom;
    }

    // Last row
    d[nm1] = (d[nm1] - a[nm1] * d[nm1 - 1]) / (b[nm1] - a[nm1] * c[nm1 - 1]);

    // Back substitution
    x[nm1] = d[nm1];
    for (int i = static_cast<int>(nm1) - 1; i >= 0; i--) {
        x[i] = d[i] - c[i] * x[i + 1];
    }
}

void thomasSolveSafe(const float *a, const float *b, const float *c, const float *d, float *x, uint n) {
    // Allocate temporary arrays to avoid modifying inputs
    std::vector<float> c_temp(n);
    std::vector<float> d_temp(n);

    std::memcpy(c_temp.data(), c, n * sizeof(float));
    std::memcpy(d_temp.data(), d, n * sizeof(float));

    if (n == 0)
        return;
    if (n == 1) {
        x[0] = d_temp[0] / b[0];
        return;
    }

    uint nm1 = n - 1;

    // Forward sweep
    c_temp[0] /= b[0];
    d_temp[0] /= b[0];

    for (uint i = 1; i < nm1; i++) {
        float denom = b[i] - a[i] * c_temp[i - 1];
        c_temp[i] /= denom;
        d_temp[i] = (d_temp[i] - a[i] * d_temp[i - 1]) / denom;
    }

    // Last row
    d_temp[nm1] = (d_temp[nm1] - a[nm1] * d_temp[nm1 - 1]) / (b[nm1] - a[nm1] * c_temp[nm1 - 1]);

    // Back substitution
    x[nm1] = d_temp[nm1];
    for (int i = static_cast<int>(nm1) - 1; i >= 0; i--) {
        x[i] = d_temp[i] - c_temp[i] * x[i + 1];
    }
}


// ============================================================================
// Soil property library - van Genuchten parameters from Carsel & Parrish (1988)
// ============================================================================

// Format: {theta_r, theta_s, alpha [1/m], n, K_s [m/s]}
static const std::map<std::string, SoilProperties> SOIL_LIBRARY = {
        {"sand", SoilProperties(0.045f, 0.43f, 14.5f, 2.68f, 8.25e-5f)},
        {"loamy_sand", SoilProperties(0.057f, 0.41f, 12.4f, 2.28f, 4.05e-5f)},
        {"sandy_loam", SoilProperties(0.065f, 0.41f, 7.5f, 1.89f, 1.23e-5f)},
        {"loam", SoilProperties(0.078f, 0.43f, 3.6f, 1.56f, 2.89e-6f)},
        {"silt", SoilProperties(0.034f, 0.46f, 1.6f, 1.37f, 6.94e-7f)},
        {"silt_loam", SoilProperties(0.067f, 0.45f, 2.0f, 1.41f, 1.25e-6f)},
        {"sandy_clay_loam", SoilProperties(0.100f, 0.39f, 5.9f, 1.48f, 3.64e-6f)},
        {"clay_loam", SoilProperties(0.095f, 0.41f, 1.9f, 1.31f, 7.22e-7f)},
        {"silty_clay_loam", SoilProperties(0.089f, 0.43f, 1.0f, 1.23f, 1.94e-7f)},
        {"sandy_clay", SoilProperties(0.100f, 0.38f, 2.7f, 1.23f, 3.33e-7f)},
        {"silty_clay", SoilProperties(0.070f, 0.36f, 0.5f, 1.09f, 5.56e-8f)},
        {"clay", SoilProperties(0.068f, 0.38f, 0.8f, 1.09f, 5.56e-7f)},
};

SoilProperties getSoilPropertiesFromTexture(const std::string &texture) {
    // Convert to lowercase for case-insensitive lookup
    std::string lower_texture = texture;
    std::transform(lower_texture.begin(), lower_texture.end(), lower_texture.begin(), ::tolower);

    // Replace spaces with underscores
    std::replace(lower_texture.begin(), lower_texture.end(), ' ', '_');

    auto it = SOIL_LIBRARY.find(lower_texture);
    if (it != SOIL_LIBRARY.end()) {
        return it->second;
    }

    std::string available;
    for (const auto &entry: SOIL_LIBRARY) {
        available += (available.empty() ? "" : ", ") + entry.first;
    }
    helios::helios_runtime_error("ERROR (getSoilPropertiesFromTexture): Unknown soil texture '" + texture + "'. Available textures: " + available + ".");
    return {}; // not reached
}

SoilProperties toModifiedVanGenuchten(const SoilProperties &vg, float h_e, bool match_standard_curves) {
    if (vg.model != SoilModel::VAN_GENUCHTEN)
        helios::helios_runtime_error("ERROR (toModifiedVanGenuchten): the input must be a standard van Genuchten soil.");
    if (!(h_e < 0.0f))
        helios::helios_runtime_error("ERROR (toModifiedVanGenuchten): the air-entry head h_e must be negative.");
    SoilProperties p = SoilProperties::modifiedVanGenuchten(vg.theta_r, vg.theta_s, vg.alpha, vg.n, vg.K_s, h_e);
    if (match_standard_curves) { // the standard curves truncated at h_e: saturated values = standard values at h_e
        double Sc, Fk;
        airEntryScales(p, Sc, Fk); // S_c and F_K depend only on alpha, n and h_e
        p.theta_s = (float) (vg.theta_r + (vg.theta_s - vg.theta_r) * Sc);
        p.K_s = (float) (vg.K_s / Fk);
    }
    return p;
}

SoilProperties getSoilPropertiesFromTexture(const std::string &texture, SoilModel model, bool match_standard_curves) {
    SoilProperties p = getSoilPropertiesFromTexture(texture);
    if (model == SoilModel::MODIFIED_VAN_GENUCHTEN) {
        p = toModifiedVanGenuchten(p, kDefaultAirEntryHead, match_standard_curves);
    } else if (model == SoilModel::GARDNER) {
        helios::helios_runtime_error("ERROR (getSoilPropertiesFromTexture): there is no texture library for SoilModel::GARDNER; use SoilProperties::gardner().");
    }
    return p;
}

std::vector<std::string> getAvailableSoilTextures() {
    std::vector<std::string> textures;
    textures.reserve(SOIL_LIBRARY.size());
    for (const auto &pair: SOIL_LIBRARY) {
        textures.push_back(pair.first);
    }
    return textures;
}

bool isSoilTextureAvailable(const std::string &texture) {
    std::string lower_texture = texture;
    std::transform(lower_texture.begin(), lower_texture.end(), lower_texture.begin(), ::tolower);
    std::replace(lower_texture.begin(), lower_texture.end(), ' ', '_');

    return SOIL_LIBRARY.find(lower_texture) != SOIL_LIBRARY.end();
}

void validateSoilProperties(const SoilProperties &props, const std::string &caller) {
    std::string problem;
    const bool van_genuchten = props.model == SoilModel::VAN_GENUCHTEN || props.model == SoilModel::MODIFIED_VAN_GENUCHTEN;
    if (!(props.theta_r >= 0.0f))
        problem = "theta_r must be >= 0";
    else if (!(props.theta_s > props.theta_r))
        problem = "theta_s must be greater than theta_r";
    else if (!(props.theta_s <= 1.0f))
        problem = "theta_s must be <= 1";
    else if (!(props.alpha > 0.0f))
        problem = "alpha must be positive";
    else if (!(props.K_s > 0.0f))
        problem = "K_s must be positive";
    else if (van_genuchten && !(props.n > 1.0f))
        problem = "the van Genuchten n must be greater than 1";
    else if (props.model == SoilModel::MODIFIED_VAN_GENUCHTEN && !(props.h_e < 0.0f))
        problem = "the air-entry head h_e of SoilModel::MODIFIED_VAN_GENUCHTEN must be negative";
    else if (!(props.kr_spline_start > 0.0f && props.kr_spline_start <= 1.0f))
        problem = "kr_spline_start must be in (0, 1]";
    if (!problem.empty()) {
        helios::helios_runtime_error("ERROR (" + caller + "): invalid soil properties: " + problem + " (theta_r = " + std::to_string(props.theta_r) + ", theta_s = " + std::to_string(props.theta_s) + ", alpha = " + std::to_string(props.alpha) +
                                     " 1/m, n = " + std::to_string(props.n) + ", K_s = " + std::to_string(props.K_s) + " m/s).");
    }
}


// ============================================================================
// Philip sorptivity (exact, via the Boltzmann similarity ODE)
// ============================================================================

namespace {
    //! Integrate the absorption ODE from h=0 to h=h_init for a trial surface flux F0.
    //! Returns F at h_init (>0 if F0 too large), or a negative sentinel if F reached 0 early
    //! (F0 too small). On a successful pass `lambda_end` receives lambda at h_init.
    float integrateAbsorption(const SoilProperties &props, float h_init, float F0, int n_steps) {
        double dh = static_cast<double>(h_init) / n_steps; // negative step (0 -> h_init)
        double lambda = 0.0, F = F0;
        auto deriv = [&](double h, double l, double f, double &dl, double &dF) {
            float K = computeK(static_cast<float>(h), props);
            float C = computeC(static_cast<float>(h), props, 1e-9f);
            dl = (f > 1e-30) ? -static_cast<double>(K) / f : -1e30;
            dF = 0.5 * l * static_cast<double>(C);
        };
        for (int i = 0; i < n_steps; i++) {
            double h = dh * i;
            double a1, b1, a2, b2, a3, b3, a4, b4;
            deriv(h, lambda, F, a1, b1);
            deriv(h + 0.5 * dh, lambda + 0.5 * dh * a1, F + 0.5 * dh * b1, a2, b2);
            deriv(h + 0.5 * dh, lambda + 0.5 * dh * a2, F + 0.5 * dh * b2, a3, b3);
            deriv(h + dh, lambda + dh * a3, F + dh * b3, a4, b4);
            lambda += dh / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4);
            F += dh / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4);
            if (F <= 0.0)
                return -1.0f; // F reached zero before h_init -> F0 too small
        }
        return static_cast<float>(F); // F at h_init (>0 -> F0 too large)
    }
} // namespace

float sorptivity(const SoilProperties &props, float h_init, int n_steps) {
    if (h_init >= 0.0f || props.K_s <= 0.0f)
        return 0.0f;
    // Shoot on F(0): bisection. F<0 sentinel => F0 too small; F>0 => F0 too large.
    float lo = 0.0f, hi = 1.0f; // upper bound comfortably exceeds any physical S/2
    for (int k = 0; k < 60; k++) {
        float F0 = 0.5f * (lo + hi);
        float Fend = integrateAbsorption(props, h_init, F0, n_steps);
        if (Fend < 0.0f) {
            lo = F0; // hit zero early -> need larger F0
        } else {
            hi = F0; // still positive at h_init -> F0 too large
        }
    }
    return 2.0f * 0.5f * (lo + hi); // S = 2 F(0)
}
