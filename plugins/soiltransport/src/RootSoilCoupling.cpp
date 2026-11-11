/** \file "RootSoilCoupling.cpp" Implementation of root-soil coupling.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RootSoilCoupling.h"
#include "SoilSolver3D.h"
#include <cmath>
#include <algorithm>
#include <numeric>

RootSoilCoupling::RootSoilCoupling(SoilGrid *grid) : grid_(grid), solver3d_(nullptr), L_r_(1e-9f) { // Default root conductivity
    if (grid == nullptr) {
        helios::helios_runtime_error("ERROR (RootSoilCoupling::RootSoilCoupling): the soil grid is null.");
    }
    nx_ = grid->getNx();
    ny_ = grid->getNy();
    nz_ = grid->getNz();
    dx_ = grid->getDx();
    dy_ = grid->getDy();
    dz_ = grid->getDz();
}

RootSoilCoupling::RootSoilCoupling(SoilSolver3D *solver3d, float origin_x, float origin_y) : grid_(nullptr), solver3d_(solver3d), L_r_(1e-9f) {
    if (solver3d == nullptr) {
        helios::helios_runtime_error("ERROR (RootSoilCoupling::RootSoilCoupling): the 3D solver is null.");
    }
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    nx_ = solver3d->nx();
    ny_ = solver3d->ny();
    nz_ = solver3d->nz();
    dx_ = static_cast<float>(solver3d->dx());
    dy_ = static_cast<float>(solver3d->dy());
    dz_ = static_cast<float>(solver3d->dz());
}

void RootSoilCoupling::columnCenter(uint i, uint j, float &x, float &y) const {
    if (grid_ != nullptr) {
        grid_->getColumnPosition(i, j, x, y);
        return;
    }
    x = origin_x_ + (static_cast<float>(i) + 0.5f) * dx_;
    y = origin_y_ + (static_cast<float>(j) + 0.5f) * dy_;
}

float RootSoilCoupling::layerDepth(uint col_idx, uint k) const {
    if (grid_ != nullptr) {
        return grid_->getColumn(col_idx).getLayerDepth(k);
    }
    return (static_cast<float>(k) + 0.5f) * dz_; // the 3D solver's grid is uniform
}

RootSoilCoupling::~RootSoilCoupling() = default;

// ============================================================================
// Statistical root distributions
// ============================================================================

void RootSoilCoupling::addPlant(uint plant_id, const RootDistributionParams &params) {
    std::map<uint, std::vector<float>> density = densityFromParams(plant_id, params); // raises an error before anything changes
    plant_root_params_[plant_id] = params;
    plant_weibull_params_.erase(plant_id);
    cumulative_uptake_[plant_id] = 0.0f;
    setPlantRoots(plant_id, std::move(density));
}

void RootSoilCoupling::removePlant(uint plant_id) {
    std::set<uint> affected;
    clearPlantSinks(plant_id, affected);
    plant_root_params_.erase(plant_id);
    plant_weibull_params_.erase(plant_id);
    root_density_maps_.erase(plant_id);
    cumulative_uptake_.erase(plant_id);
    plant_transpiration_.erase(plant_id);
    uptake_flux_.erase(plant_id);
    applyColumns(affected); // the plant's sinks leave the soil now
}

void RootSoilCoupling::requireRoots(uint plant_id, const std::string &function) const {
    if (root_density_maps_.find(plant_id) == root_density_maps_.end()) {
        helios::helios_runtime_error("ERROR (RootSoilCoupling::" + function + "): plant " + std::to_string(plant_id) + " has no root distribution. Give it one first with SoilTransportModel::setRootDistribution() or setRootDistributionWeibull().");
    }
}

void RootSoilCoupling::normalizeDensity(uint plant_id, std::map<uint, std::vector<float>> &density, double total_root_density) {
    // Normalize globally so the sum over every cell is 1: each column's sum is then its fraction of the total root
    // length (the between-column uptake weight). Per-column normalization would make every rooted column draw the full
    // transpiration demand.
    if (!(total_root_density > 0.0)) {
        helios::helios_runtime_error("ERROR (RootSoilCoupling): the root distribution of plant " + std::to_string(plant_id) + " places no roots inside the soil domain. Check that the plant's position and root extent lie within the domain.");
    }
    for (auto &[col_idx, column_density]: density) {
        for (float &d: column_density) {
            d = static_cast<float>(d / total_root_density);
        }
    }
}

void RootSoilCoupling::setPlantRoots(uint plant_id, std::map<uint, std::vector<float>> &&density) {
    root_density_maps_[plant_id] = std::move(density);
    if (plant_transpiration_.find(plant_id) != plant_transpiration_.end()) {
        std::set<uint> affected;
        refreshPlantSinks(plant_id, affected); // the roots moved: so do the plant's sinks
        applyColumns(affected);
    }
}

void RootSoilCoupling::updatePlantRoots(uint plant_id, const RootDistributionParams &params) {
    std::map<uint, std::vector<float>> density = densityFromParams(plant_id, params); // raises an error before anything changes
    plant_root_params_[plant_id] = params;
    plant_weibull_params_.erase(plant_id);
    setPlantRoots(plant_id, std::move(density));
}

// ============================================================================
// Explicit root primitives
// ============================================================================

void RootSoilCoupling::setRootPrimitives(uint plant_id, const std::vector<uint> &root_UUIDs) {
    helios::helios_runtime_error("ERROR (RootSoilCoupling::setRootPrimitives): mapping explicit root primitives onto the soil grid is not implemented yet "
                                 "(plant " +
                                 std::to_string(plant_id) + ", " + std::to_string(root_UUIDs.size()) + " primitives). Describe the plant's root zone with addPlant() (RootDistributionParams) or setWeibullDistribution() instead.");
}

// ============================================================================
// Water uptake
// ============================================================================

void RootSoilCoupling::setRootRadialConductivity(float L_r) {
    L_r_ = L_r;
}

float RootSoilCoupling::stressFactor(float h) const {
    if (h >= h_onset_) {
        return 1.0f;
    }
    if (h <= h_wilting_) {
        return 0.0f;
    }
    return (h - h_wilting_) / (h_onset_ - h_wilting_);
}

float RootSoilCoupling::cellHead(uint col_idx, uint k) const {
    // In full-3D mode the column grid is not advanced, so read the 3D solver's head
    // (its k = nz-1 is the surface).
    if (solver3d_) {
        return static_cast<float>(solver3d_->getPressureHead(col_idx % nx_, col_idx / nx_, nz_ - 1 - k));
    }
    return grid_->getColumn(col_idx).getPressureHead(k);
}

void RootSoilCoupling::setWaterStressParameters(float h_onset, float h_wilting, float omega_c) {
    if (!(h_wilting < h_onset) || h_onset > 0.0f || !(omega_c > 0.0f && omega_c <= 1.0f)) {
        helios::helios_runtime_error("ERROR (RootSoilCoupling::setWaterStressParameters): need h_wilting < h_onset <= 0 and 0 < omega_c <= 1.");
    }
    h_onset_ = h_onset;
    h_wilting_ = h_wilting;
    omega_c_ = omega_c;
}

void RootSoilCoupling::computeRootUptake(uint plant_id, float transpiration_demand) {
    requireRoots(plant_id, "computeRootUptake");
    plant_transpiration_[plant_id] = transpiration_demand;
    // Only this plant's sinks change, so only the columns it roots in (before or now) are updated: setting the demand
    // of every plant each step costs one pass over each plant's roots, not one over every plant's roots per plant.
    std::set<uint> affected;
    refreshPlantSinks(plant_id, affected);
    applyColumns(affected);
}

void RootSoilCoupling::computeAllRootUptake() {
    std::set<uint> affected;
    for (const auto &[plant_id, demand]: plant_transpiration_) {
        refreshPlantSinks(plant_id, affected);
    }
    applyColumns(affected);
}

void RootSoilCoupling::refreshPlantSinks(uint plant_id, std::set<uint> &affected) {
    // transpiration_demand is the transpiration FLUX per unit ground area [m/s]; the total
    // plant uptake is transpiration_demand * (domain ground area) = T * n_columns * dx*dy.
    // Each root cell's share is its root fraction f times its water-stress factor alpha(h),
    // renormalized so uptake shifts toward unstressed roots (compensation). The n_columns factor
    // makes the result grid-independent.
    const float transpiration_demand = plant_transpiration_.at(plant_id);
    const auto &density_map = root_density_maps_.at(plant_id);
    const double n_col = static_cast<double>(nx_) * static_cast<double>(ny_);

    // Stress-weighted root fractions, and the root-zone stress index omega = sum(alpha f) / sum(f)
    std::map<uint, std::vector<float>> weighted;
    double sum_f = 0.0, sum_af = 0.0;
    for (const auto &[col_idx, f]: density_map) {
        std::vector<float> &wf = weighted[col_idx];
        wf.assign(f.size(), 0.0f);
        for (uint k = 0; k < f.size(); k++) {
            if (f[k] <= 0.0f) {
                continue;
            }
            wf[k] = f[k] * stressFactor(cellHead(col_idx, k));
            sum_f += f[k];
            sum_af += wf[k];
        }
    }
    const double omega = sum_f > 0.0 ? sum_af / sum_f : 0.0;
    const float supply_fraction = static_cast<float>(std::min(1.0, omega / static_cast<double>(omega_c_)));
    const float uptake = transpiration_demand * supply_fraction;
    uptake_flux_[plant_id] = uptake;
    const double norm = sum_af > 0.0 ? 1.0 / sum_af : 0.0;

    // Replace the plant's sinks: column index -> per-layer uptake [m/s per unit column ground area], layer 0 at the surface
    clearPlantSinks(plant_id, affected);
    std::map<uint, std::vector<double>> &sinks = plant_sinks_[plant_id];
    for (const auto &[col_idx, wf]: weighted) {
        std::vector<double> &sink = sinks[col_idx];
        sink.assign(nz_, 0.0);
        for (uint k = 0; k < wf.size(); k++) {
            sink[k] = static_cast<double>(uptake) * n_col * wf[k] * norm;
        }
        column_plants_[col_idx].insert(plant_id);
        affected.insert(col_idx);
    }
}

void RootSoilCoupling::clearPlantSinks(uint plant_id, std::set<uint> &affected) {
    auto it = plant_sinks_.find(plant_id);
    if (it == plant_sinks_.end()) {
        return;
    }
    for (const auto &[col_idx, sink]: it->second) {
        column_plants_[col_idx].erase(plant_id);
        affected.insert(col_idx);
    }
    plant_sinks_.erase(it);
}

void RootSoilCoupling::applyColumns(const std::set<uint> &columns) {
    // Each column's sink is the sum over the plants rooted in it, so plants whose roots share cells all draw their water.
    // It is summed afresh from the plants' sinks (not adjusted by differences), so no round-off accumulates.
    for (uint col_idx: columns) {
        std::vector<double> total(nz_, 0.0);
        auto plants = column_plants_.find(col_idx);
        if (plants != column_plants_.end()) {
            for (uint plant_id: plants->second) {
                const std::vector<double> &sink = plant_sinks_.at(plant_id).at(col_idx);
                for (uint k = 0; k < nz_; k++) {
                    total[k] += sink[k];
                }
            }
            if (plants->second.empty()) {
                column_plants_.erase(plants);
            }
        }
        if (solver3d_) {
            // Full-3D solver: a per-cell volumetric sink [1/s] = column-layer uptake / dz. (Note the depth flip: column
            // layer k=0 is the surface, but in the 3D solver k = nz-1 is the surface.)
            const uint i = col_idx % nx_, j = col_idx / nx_;
            for (uint k = 0; k < nz_; k++) {
                solver3d_->setUptake(i, j, nz_ - 1 - k, total[k] / static_cast<double>(dz_));
            }
            continue;
        }
        // Column grid (1D / quasi-3D): the column's transpiration flux is the sum of its layer uptakes, and the layer
        // uptakes are its root profile (the column normalizes it). A column with no roots left takes up no water.
        const double col_transpiration = std::accumulate(total.begin(), total.end(), 0.0);
        SoilColumn &column = grid_->getColumn(col_idx);
        column.setTranspiration(static_cast<float>(col_transpiration));
        if (col_transpiration > 0.0) {
            column.setRootDistribution(std::vector<float>(total.begin(), total.end()));
        }
    }
}

// ============================================================================
// Output for plant hydraulics coupling
// ============================================================================

float RootSoilCoupling::getSoilWaterPotential(uint plant_id) const {
    requireRoots(plant_id, "getSoilWaterPotential");
    auto it = root_density_maps_.find(plant_id);

    // Root-zone potential seen by the plant: weighted by root density times the water-stress factor,
    // so roots in soil too dry to take up water (whose heads diverge toward -infinity) do not
    // dominate. Falls back to plain root-density weighting when no root cell is active.
    double num_active = 0.0, den_active = 0.0, num_all = 0.0, den_all = 0.0;
    for (const auto &[col_idx, f]: it->second) {
        for (uint k = 0; k < f.size(); k++) {
            if (f[k] <= 0.0f) {
                continue;
            }
            const float h = cellHead(col_idx, k);
            const double w = f[k] * stressFactor(h);
            num_active += w * h;
            den_active += w;
            num_all += f[k] * h;
            den_all += f[k];
        }
    }
    if (den_active > 0.0) {
        return static_cast<float>(num_active / den_active);
    }
    return static_cast<float>(num_all / den_all); // den_all > 0: a root distribution always has roots in the domain
}

float RootSoilCoupling::getCumulativeRootUptake(uint plant_id) const {
    requireRoots(plant_id, "getCumulativeRootUptake");
    auto it = cumulative_uptake_.find(plant_id);
    return it != cumulative_uptake_.end() ? it->second : 0.0f; // no uptake yet
}

float RootSoilCoupling::getRootUptakeFlux(uint plant_id) const {
    requireRoots(plant_id, "getRootUptakeFlux");
    auto it = uptake_flux_.find(plant_id);
    return it != uptake_flux_.end() ? it->second : 0.0f; // no transpiration demand set yet
}

void RootSoilCoupling::accumulateUptake(float dt) {
    for (const auto &[plant_id, flux]: uptake_flux_) {
        cumulative_uptake_[plant_id] += flux * dt;
    }
}

// ============================================================================
// Internal methods
// ============================================================================

std::map<uint, std::vector<float>> RootSoilCoupling::densityFromParams(uint plant_id, const RootDistributionParams &params) const {
    std::map<uint, std::vector<float>> density; // column -> raw per-layer root density, normalized below
    double total_density = 0.0;
    for (uint j = 0; j < ny_; j++) {
        for (uint i = 0; i < nx_; i++) {
            float col_x, col_y;
            columnCenter(i, j, col_x, col_y);
            const float r = std::sqrt(std::pow(col_x - params.plant_x, 2.0f) + std::pow(col_y - params.plant_y, 2.0f)); // distance from the plant
            if (r > params.r_max) {
                continue; // outside the root zone
            }
            const uint col_idx = j * nx_ + i;
            std::vector<float> col_density(nz_, 0.0f);
            float col_total = 0.0f;
            for (uint k = 0; k < nz_; k++) {
                const float z = layerDepth(col_idx, k);
                if (z > params.z_max) {
                    break; // below the root zone
                }
                col_density[k] = rootDensityFunction(r, z, params);
                col_total += col_density[k];
            }
            if (col_total > 0.0f) {
                density[col_idx] = std::move(col_density);
                total_density += col_total;
            }
        }
    }
    normalizeDensity(plant_id, density, total_density);
    return density;
}

void RootSoilCoupling::setWeibullDistribution(uint plant_id, const WeibullRootParameters &params) {
    if (!(params.shape_z > 0.0f) || !(params.scale_z > 0.0f) || !(params.z_max > 0.0f) || (params.radial && (!(params.shape_r > 0.0f) || !(params.scale_r > 0.0f) || !(params.r_max > 0.0f)))) {
        helios::helios_runtime_error("ERROR (RootSoilCoupling::setWeibullDistribution): the Weibull shapes, scales and maximum extents of plant " + std::to_string(plant_id) + " must be positive.");
    }
    std::map<uint, std::vector<float>> density = densityFromWeibull(plant_id, params); // raises an error before anything changes
    plant_weibull_params_[plant_id] = params;
    plant_root_params_.erase(plant_id);
    setPlantRoots(plant_id, std::move(density));
}

double RootSoilCoupling::weibullSurvival(double x, double shape, double scale) {
    if (x <= 0.0) {
        return 1.0;
    }
    return std::exp(-std::pow(x / scale, shape));
}

float RootSoilCoupling::weibullPDF(float x, float shape, float scale) {
    if (x < 0.0f || shape <= 0.0f || scale <= 0.0f) {
        return 0.0f;
    }
    const float xl = x / scale;
    return (shape / scale) * std::pow(xl, shape - 1.0f) * std::exp(-std::pow(xl, shape));
}

std::map<uint, std::vector<float>> RootSoilCoupling::densityFromWeibull(uint plant_id, const WeibullRootParameters &p) const {
    // Half the diagonal of a column: the radial span a column covers around its centre
    const double half_diagonal = 0.5 * std::sqrt(static_cast<double>(dx_) * dx_ + static_cast<double>(dy_) * dy_);

    // Build raw (un-normalized) per-cell root length density, then normalize globally so the sum over every cell is 1.
    // Each column's sum is then its fraction of the total root length (the between-column uptake weight), and the
    // relative values within a column give its vertical profile.
    std::map<uint, std::vector<float>> density;
    double total = 0.0;
    for (uint j = 0; j < ny_; j++) {
        for (uint i = 0; i < nx_; i++) {
            const uint col = j * nx_ + i;

            // Radial weight (3D only); 1D is horizontally uniform.
            float radial_w = 1.0f;
            if (p.radial) {
                float col_x, col_y;
                columnCenter(i, j, col_x, col_y);
                const double r = std::sqrt(static_cast<double>(col_x - p.plant_x) * (col_x - p.plant_x) + static_cast<double>(col_y - p.plant_y) * (col_y - p.plant_y));
                if (r > p.r_max) {
                    continue; // outside the root zone
                }
                // Mean of the radial density over the radial span the column covers. Unlike its value at the centre this
                // is finite for shape < 1 at the plant's own column (the density is infinite at r = 0), and it equals the
                // value at the centre to second order elsewhere. Taken as a difference of the survival function
                // exp(-(r/lambda)^k), in double, so the far tail keeps its (small) weight instead of rounding to 0.
                const double r_inner = std::max(r - half_diagonal, 0.0), r_outer = r + half_diagonal;
                radial_w = static_cast<float>((weibullSurvival(r_inner, p.shape_r, p.scale_r) - weibullSurvival(r_outer, p.shape_r, p.scale_r)) / (r_outer - r_inner));
            }
            if (radial_w <= 0.0f) {
                continue;
            }

            std::vector<float> col_density(nz_, 0.0f);
            float col_sum = 0.0f;
            for (uint k = 0; k < nz_; k++) {
                const float z = layerDepth(col, k);
                if (z > p.z_max) {
                    break; // below the root zone
                }
                const float rld = radial_w * weibullPDF(z, p.shape_z, p.scale_z);
                col_density[k] = rld;
                col_sum += rld;
            }
            if (col_sum > 0.0f) {
                density[col] = std::move(col_density);
                total += static_cast<double>(col_sum);
            }
        }
    }
    normalizeDensity(plant_id, density, total);
    return density;
}

float RootSoilCoupling::rootDensityFunction(float r, float z, const RootDistributionParams &params) const {
    // Vrugt et al. (2001) root distribution function
    // RLD(r,z) = RLD_max * (1 - z/z_max)^pz * (1 - r/r_max)^pr * exp(-(pz/z_star)|z - z_star| - (pr/r_star)|r - r_star|)

    if (z < 0.0f || z > params.z_max || r < 0.0f || r > params.r_max) {
        return 0.0f;
    }

    float z_term = std::pow(1.0f - z / params.z_max, params.pz);
    float r_term = std::pow(1.0f - r / params.r_max, params.pr);

    // Exponential decay terms
    float exp_z = (params.z_star > 0.0f) ? std::exp(-(params.pz / params.z_star) * std::fabs(z - params.z_star)) : 1.0f;
    float exp_r = (params.r_star > 0.0f) ? std::exp(-(params.pr / params.r_star) * std::fabs(r - params.r_star)) : 1.0f;

    return z_term * r_term * exp_z * exp_r;
}
