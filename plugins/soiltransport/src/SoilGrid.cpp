/** \file "SoilGrid.cpp" Implementation of soil column grid with parallel solving.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilGrid.h"
#include <cmath>
#include <algorithm>
#include "global.h"
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

SoilGrid::SoilGrid(uint nx, uint ny, uint nz, float dx, float dy, float dz, const SoilProperties &props) :
    nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz), origin_x_(0.0f), origin_y_(0.0f), origin_z_(0.0f), horizontal_coupling_enabled_(false), horizontal_coupling_interval_(10), step_count_(0), splitting_type_(SplittingType::SEQUENTIAL) {

    if (nx == 0 || ny == 0 || nz == 0) {
        helios::helios_runtime_error("ERROR (SoilGrid::SoilGrid): grid dimensions must be positive.");
    }
    if (!(dx > 0.0f) || !(dy > 0.0f) || !(dz > 0.0f)) {
        helios::helios_runtime_error("ERROR (SoilGrid::SoilGrid): the grid spacing must be positive (got dx = " + std::to_string(dx) + ", dy = " + std::to_string(dy) + ", dz = " + std::to_string(dz) + " m).");
    }

    n_columns_ = nx * ny;

    // Create columns
    columns_.reserve(n_columns_);
    for (uint col = 0; col < n_columns_; col++) {
        columns_.push_back(std::make_unique<SoilColumn>(nz, dz, props));
    }

    // Allocate working arrays for horizontal diffusion
    h_flat_.resize(nz * n_columns_);
    K_flat_.resize(nz * n_columns_);
    C_flat_.resize(nz * n_columns_);
    theta_flat_.resize(nz * n_columns_);
    h_new_.resize(nz * n_columns_);
}

SoilGrid::~SoilGrid() = default;

// ============================================================================
// Domain setup
// ============================================================================

void SoilGrid::setOrigin(float origin_x, float origin_y, float origin_z) {
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    origin_z_ = origin_z;
}

void SoilGrid::setInitialCondition(float h_init) {
    for (auto &col: columns_) {
        col->setInitialCondition(h_init);
    }
}

void SoilGrid::setInitialCondition(const std::vector<float> &h_profile) {
    if (h_profile.size() != nz_ * n_columns_) {
        helios::helios_runtime_error("ERROR (SoilGrid::setInitialCondition): initial condition size mismatch.");
    }

    // h_profile is in layer-major order: h[k * n_columns + col]
    for (uint col = 0; col < n_columns_; col++) {
        std::vector<float> col_h(nz_);
        for (uint k = 0; k < nz_; k++) {
            col_h[k] = h_profile[k * n_columns_ + col];
        }
        columns_[col]->setInitialCondition(col_h);
    }
}

void SoilGrid::setInitialConditionHydrostatic(float water_table_depth) {
    for (auto &col: columns_) {
        col->setInitialConditionHydrostatic(water_table_depth);
    }
}

void SoilGrid::setSoilProperties(uint i, uint j, const SoilProperties &props) {
    uint idx = columnIndex(i, j);
    if (idx >= n_columns_) {
        helios::helios_runtime_error("ERROR (SoilGrid::setSoilProperties): column index out of range.");
    }
    // Keeps the column's state, boundary conditions, sources and solver parameters
    columns_[idx]->setLayerSoilProperties(0, nz_, props);
}

void SoilGrid::setTopBoundary(BCType type, float value) {
    for (auto &col: columns_) {
        col->setTopBoundary(type, value);
    }
}

void SoilGrid::setTopBoundary(uint i, uint j, BCType type, float value) {
    getColumn(i, j).setTopBoundary(type, value);
}

void SoilGrid::setBottomBoundary(BCType type, float value) {
    for (auto &col: columns_) {
        col->setBottomBoundary(type, value);
    }
}

void SoilGrid::setBottomBoundary(uint i, uint j, BCType type, float value) {
    getColumn(i, j).setBottomBoundary(type, value);
}

// ============================================================================
// Solver control
// ============================================================================

void SoilGrid::setSolverParameters(const SolverParameters &params) {
    solver_params_ = params;
    for (auto &col: columns_) {
        col->setSolverParameters(params);
    }
}

void SoilGrid::enableHorizontalCoupling(uint interval) {
    if (interval < 1) {
        helios::helios_runtime_error("ERROR (SoilGrid::enableHorizontalCoupling): the coupling interval must be at least 1 vertical step.");
    }
    horizontal_coupling_enabled_ = true;
    horizontal_coupling_interval_ = interval;
}

void SoilGrid::disableHorizontalCoupling() {
    horizontal_coupling_enabled_ = false;
}

void SoilGrid::setSplittingType(SplittingType type) {
    splitting_type_ = type;
}

uint SoilGrid::solve(float dt) {
    if (!(dt > 0.0f)) {
        helios::helios_runtime_error("ERROR (SoilGrid::solve): the timestep must be positive (got " + std::to_string(dt) + " s).");
    }
    if (horizontal_coupling_enabled_) {
        return solveQuasi3D(dt);
    }

    // Pure 1D: solve all columns independently
    solveVerticalColumns(dt);

    // Count converged columns
    uint converged = 0;
    for (const auto &col: columns_) {
        if (col->didConverge()) {
            converged++;
        }
    }
    return converged;
}

uint SoilGrid::solveQuasi3D(float dt) {
    uint converged = 0;
    step_count_++;

    // Columns that failed in the first vertical half-step of Strang splitting (their state was restored, so a later
    // half-step converging does not make the whole step a success)
    std::vector<char> failed_first_half(n_columns_, 0);
    if (splitting_type_ == SplittingType::STRANG) {
        // Strang splitting: V(dt/2) -> H(dt) -> V(dt/2)
        solveVerticalColumns(dt / 2.0f);
        for (uint c = 0; c < n_columns_; c++)
            failed_first_half[c] = columns_[c]->didConverge() ? 0 : 1;

        if (step_count_ % horizontal_coupling_interval_ == 0) {
            flattenState();
            solveHorizontalDiffusion(dt * static_cast<float>(horizontal_coupling_interval_));
            unflattenState();
        }

        solveVerticalColumns(dt / 2.0f);
    } else {
        // Sequential splitting: V(dt) -> H(dt)
        solveVerticalColumns(dt);

        if (step_count_ % horizontal_coupling_interval_ == 0) {
            flattenState();
            solveHorizontalDiffusion(dt * static_cast<float>(horizontal_coupling_interval_));
            unflattenState();
        }
    }

    // Count converged
    for (uint c = 0; c < n_columns_; c++) {
        if (columns_[c]->didConverge() && !failed_first_half[c]) {
            converged++;
        }
    }
    return converged;
}

float SoilGrid::solveAdaptive(float target_time, float &dt, uint *steps_taken) {
    float t = 0.0f;
    uint steps = 0;
    dt = std::max(solver_params_.dt_min, std::min(dt, solver_params_.dt_max));

    // Per-column state buffers for save/restore (heads) and the error proxy (theta^n).
    std::vector<std::vector<float>> h_saved(n_columns_);
    std::vector<std::vector<float>> theta_saved(n_columns_);

    while (t < target_time) {
        const float dt_step = std::min(dt, target_time - t);

        // Snapshot the whole grid so the step can be rejected as a unit. theta^n is
        // read from the cached constitutive state (no h_to_theta recompute).
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int c = 0; c < static_cast<int>(n_columns_); c++) {
            h_saved[c] = columns_[c]->getPressureHeads();
            std::vector<float> &ts = theta_saved[c];
            ts.resize(nz_);
            for (uint k = 0; k < nz_; k++)
                ts[k] = columns_[c]->getWaterContentCached(k);
        }

        // Trial split-step (vertical columns + horizontal coupling).
        const uint converged = solve(dt_step);

        if (converged < n_columns_) { // a column failed: reject and retry smaller
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int c = 0; c < static_cast<int>(n_columns_); c++) {
                columns_[c]->setInitialCondition(h_saved[c]);
            }
            if (dt_step <= solver_params_.dt_min * 1.01f) {
                break; // failed at the minimum step: stop, and report the time actually advanced
            }
            dt = std::max(dt * solver_params_.dt_reduction_min, solver_params_.dt_min);
            continue;
        }

        // Accepted. Size the next step from the largest per-step relative change in
        // water content across the whole grid, |Δθ|/(θs-θr) (bounded error proxy).
        // theta^{n+1} is read from the cached constitutive state (no recompute).
        // (per-thread maxima merged in a critical section: MSVC's OpenMP 2.0 has no max reduction)
        float rel = 0.0f;
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            float rel_thread = 0.0f;
#ifdef _OPENMP
#pragma omp for
#endif
            for (int c = 0; c < static_cast<int>(n_columns_); c++) {
                for (uint k = 0; k < nz_; k++) {
                    const SoilProperties &p = columns_[c]->getLayerSoilProperties(k);
                    const float range = p.theta_s - p.theta_r;
                    if (range > 1e-12f) {
                        rel_thread = std::max(rel_thread, std::fabs(columns_[c]->getWaterContentCached(k) - theta_saved[c][k]) / range);
                    }
                }
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            rel = std::max(rel, rel_thread);
        }

        t += dt_step;
        steps++;
        float factor = (rel > 1e-9f) ? solver_params_.dt_safety * std::sqrt(solver_params_.dt_target_rel_change / rel) : solver_params_.dt_growth_max;
        factor = std::max(solver_params_.dt_reduction_min, std::min(factor, solver_params_.dt_growth_max));
        dt = std::max(solver_params_.dt_min, std::min(dt * factor, solver_params_.dt_max));
    }

    if (steps_taken)
        *steps_taken = steps;
    return t;
}

// ============================================================================
// State access
// ============================================================================

SoilColumn &SoilGrid::getColumn(uint i, uint j) {
    if (i >= nx_ || j >= ny_) {
        helios::helios_runtime_error("ERROR (SoilGrid::getColumn): column index out of range.");
    }
    return *columns_[columnIndex(i, j)];
}

const SoilColumn &SoilGrid::getColumn(uint i, uint j) const {
    if (i >= nx_ || j >= ny_) {
        helios::helios_runtime_error("ERROR (SoilGrid::getColumn): column index out of range.");
    }
    return *columns_[columnIndex(i, j)];
}

SoilColumn &SoilGrid::getColumn(uint col_idx) {
    if (col_idx >= n_columns_) {
        helios::helios_runtime_error("ERROR (SoilGrid::getColumn): column index out of range.");
    }
    return *columns_[col_idx];
}

const SoilColumn &SoilGrid::getColumn(uint col_idx) const {
    if (col_idx >= n_columns_) {
        helios::helios_runtime_error("ERROR (SoilGrid::getColumn): column index out of range.");
    }
    return *columns_[col_idx];
}

float SoilGrid::getPressureHead(uint i, uint j, uint k) const {
    return getColumn(i, j).getPressureHead(k);
}

std::vector<float> SoilGrid::getAllPressureHeads() const {
    std::vector<float> h(nz_ * n_columns_);
    for (uint col = 0; col < n_columns_; col++) {
        const auto &heads = columns_[col]->getPressureHeads();
        for (uint k = 0; k < nz_; k++) {
            h[k * n_columns_ + col] = heads[k];
        }
    }
    return h;
}

float SoilGrid::getWaterContent(uint i, uint j, uint k) const {
    return getColumn(i, j).getWaterContent(k);
}

double SoilGrid::getTotalWaterStorage() const {
    double total = 0.0;
    for (const auto &col: columns_) {
        total += col->getTotalWaterStorage() * static_cast<double>(dx_) * static_cast<double>(dy_);
    }
    return total;
}

void SoilGrid::getColumnPosition(uint i, uint j, float &x, float &y) const {
    x = origin_x_ + (static_cast<float>(i) + 0.5f) * dx_;
    y = origin_y_ + (static_cast<float>(j) + 0.5f) * dy_;
}

bool SoilGrid::findColumn(float x, float y, uint &i, uint &j) const {
    float local_x = x - origin_x_;
    float local_y = y - origin_y_;

    // Closed domain: a point on the far x or y edge belongs to the last column. The tolerance (in columns) absorbs the
    // round-off of a point placed exactly on the edge.
    const float fx = local_x / dx_, fy = local_y / dy_, tolerance = 1e-4f;
    if (fx < -tolerance || fy < -tolerance || fx > static_cast<float>(nx_) + tolerance || fy > static_cast<float>(ny_) + tolerance)
        return false;

    i = std::min(static_cast<uint>(std::max(fx, 0.0f)), nx_ - 1);
    j = std::min(static_cast<uint>(std::max(fy, 0.0f)), ny_ - 1);
    return true;
}

float SoilGrid::getPressureHeadAtPosition(float x, float y, float z) const {
    // Find layer (z is depth from surface, positive downward). The surface and the bottom belong to the top and bottom layers.
    uint i, j;
    const float fz = (origin_z_ - z) / dz_, tolerance = 1e-4f;
    if (!findColumn(x, y, i, j) || fz < -tolerance || fz > static_cast<float>(nz_) + tolerance) {
        helios::helios_runtime_error("ERROR (SoilGrid::getPressureHeadAtPosition): the point (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ") is outside the grid.");
    }
    const uint k = std::min(static_cast<uint>(std::max(fz, 0.0f)), nz_ - 1);

    return getColumn(i, j).getPressureHead(k);
}

// ============================================================================
// Root coupling
// ============================================================================

void SoilGrid::setTranspiration(uint i, uint j, float transpiration) {
    getColumn(i, j).setTranspiration(transpiration);
}

void SoilGrid::setRootDistribution(uint i, uint j, const std::vector<float> &root_dist) {
    getColumn(i, j).setRootDistribution(root_dist);
}

void SoilGrid::addSource(uint i, uint j, uint layer, float flux) {
    getColumn(i, j).addSource(layer, flux);
}

void SoilGrid::clearSources() {
    for (auto &col: columns_) {
        col->clearSources();
    }
}

// ============================================================================
// Internal methods
// ============================================================================

void SoilGrid::solveVerticalColumns(float dt) {
    // Solve all columns in parallel using OpenMP
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int col = 0; col < static_cast<int>(n_columns_); col++) {
        columns_[col]->solve(dt);
    }
}

void SoilGrid::flattenState() {
    // Extract head + constitutive values from all columns into flat arrays for the
    // horizontal step. K, C, and theta are read from the columns' cached constitutive
    // state (already computed during the vertical solve) rather than re-evaluated, so
    // the flatten costs no transcendentals.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int col = 0; col < static_cast<int>(n_columns_); col++) {
        const auto &heads = columns_[col]->getPressureHeads();
        for (uint k = 0; k < nz_; k++) {
            uint idx = k * n_columns_ + col;
            h_flat_[idx] = heads[k];
            K_flat_[idx] = columns_[col]->getConductivity(k);
            C_flat_[idx] = columns_[col]->getCapacity(k);
            theta_flat_[idx] = columns_[col]->getWaterContentCached(k);
        }
    }
}

void SoilGrid::unflattenState() {
    // Update columns with new pressure heads after horizontal diffusion. Each column
    // is independent (setInitialCondition refreshes only its own constitutive cache),
    // so this scatter + constitutive refresh parallelizes over columns.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int col = 0; col < static_cast<int>(n_columns_); col++) {
        std::vector<float> col_h(nz_);
        for (uint k = 0; k < nz_; k++) {
            col_h[k] = h_new_[k * n_columns_ + col];
        }
        columns_[col]->setInitialCondition(col_h);
    }
}

void SoilGrid::solveHorizontalDiffusion(float dt) {
    // Implicit (Peaceman-Rachford ADI) horizontal diffusion in WATER-CONTENT form, per
    // layer: d(theta)/dt = div(D grad theta), D = K/C, no-flow lateral BCs. Discretizing
    // the FLUX of theta makes it exactly mass conservative (total theta per layer
    // telescopes), and the constant time term 2/dt keeps it unconditionally stable --
    // unlike head-form diffusion, whose K/C -> infinity at saturation makes the split
    // scheme blow up to full saturation. K, C (hence D) are frozen at flatten(); theta^n
    // is the cached theta_flat_.
    //
    // Interblock D uses an ARITHMETIC mean, not geometric: a geometric mean of a wet
    // (high-D) cell and a dry (tiny-D) neighbour is dominated by the dry side, which
    // chokes flux out of a saturated cell and leaves a dry ring around the emitter. The
    // saturation cap on D (via C_floor) bounds the wet-side diffusivity so the arithmetic
    // mean does not over-diffuse.
    const float dx2 = dx_ * dx_;
    const float dy2 = dy_ * dy_;
    // Specific-storage floor on C so the saturated diffusivity D=K/C stays bounded
    // (only affects near-saturated cells; unsaturated C is orders of magnitude larger).
    const float C_floor = std::max(solver_params_.C_regularization, 1e-5f);
    const float beta = 2.0f / dt; // time term (constant -> strictly diagonally dominant)

    // Quasi-3D cannot robustly handle saturation. As a cell approaches theta_s the
    // capacity C = d(theta)/dh -> 0, so the horizontal "diffusivity" K/C blows up and
    // the operator-split coupling over-mixes or diverges; no physical specific-storage
    // floor de-stiffens it at a vertically-sized timestep. Fail loudly with actionable
    // guidance rather than silently corrupting the solution. (h_flat_ holds the current
    // post-vertical-solve heads, populated by flattenState() just before this call.)
    {
        // A cell is saturated at h >= its saturation head (0, or the air-entry head of the modified model)
        auto saturated = [&](uint idx) { return h_flat_[idx] >= (float) saturationHead(columns_[idx % n_columns_]->getLayerSoilProperties(idx / n_columns_)); };
        int any_saturated = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : any_saturated)
#endif
        for (int idx = 0; idx < static_cast<int>(h_flat_.size()); idx++)
            any_saturated += (int) saturated(static_cast<uint>(idx));
        if (any_saturated) {
            uint si = 0, sj = 0, sk = 0;
            float sh = 0.0f;
            for (uint idx = 0; idx < h_flat_.size(); idx++) {
                if (saturated(idx)) {
                    const uint col = idx % n_columns_;
                    sk = idx / n_columns_;
                    si = col % nx_;
                    sj = col / nx_;
                    sh = h_flat_[idx];
                    break;
                }
            }
            helios::helios_runtime_error("ERROR (SoilTransportModel, SOLVER_QUASI_3D): a soil cell reached saturation (i=" + std::to_string(si) + ", j=" + std::to_string(sj) + ", k=" + std::to_string(sk) + ", pressure head = " + std::to_string(sh) +
                                         " m, at or above its saturation head).\n"
                                         "  The quasi-3D solver cannot robustly handle saturated conditions: near saturation the\n"
                                         "  specific capacity C = d(theta)/dh -> 0, so the horizontal coupling's diffusivity K/C\n"
                                         "  becomes stiff and the operator-split scheme over-mixes or diverges. No physical\n"
                                         "  specific-storage floor removes this at a vertically-sized timestep.\n"
                                         "  How to proceed:\n"
                                         "    - Use the full 3D solver:  setSolverType(SolverType::SOLVER_3D)  (it solves the\n"
                                         "      coupled, saturating flow correctly).\n"
                                         "    - Or keep the soil unsaturated: reduce the applied forcing (boundary flux or\n"
                                         "      volumetric source), distribute it over more cells, or coarsen the mesh so no\n"
                                         "      single cell saturates.\n"
                                         "  Quasi-3D is intended for gently-varying, unsaturated lateral flow.");
        }
    }

    const std::vector<float> &theta = theta_flat_;
    std::vector<float> D(h_flat_.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int idx = 0; idx < static_cast<int>(D.size()); idx++) {
        D[idx] = K_flat_[idx] / std::max(C_flat_[idx], C_floor);
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k_signed = 0; k_signed < static_cast<int>(nz_); k_signed++) {
        const uint k = static_cast<uint>(k_signed);
        const uint base = k * n_columns_;
        std::vector<float> tstar(n_columns_);
        const uint nmax = std::max(nx_, ny_);
        std::vector<float> a(nmax), b(nmax), c(nmax), d(nmax), x(nmax);

        // ---- x-implicit half-step: theta^n -> tstar ----
        for (uint j = 0; j < ny_; j++) {
            for (uint i = 0; i < nx_; i++) {
                const uint idx = base + j * nx_ + i;
                const float Dc = D[idx];
                const float Dxm = (i > 0) ? 0.5f * (D[idx - 1] + Dc) : 0.0f;
                const float Dxp = (i < nx_ - 1) ? 0.5f * (Dc + D[idx + 1]) : 0.0f;
                a[i] = -Dxm / dx2;
                c[i] = -Dxp / dx2;
                b[i] = beta + (Dxm + Dxp) / dx2;
                const float tC = theta[idx];
                const float Dym = (j > 0) ? 0.5f * (D[idx - nx_] + Dc) : 0.0f;
                const float Dyp = (j < ny_ - 1) ? 0.5f * (Dc + D[idx + nx_]) : 0.0f;
                const float tym = (j > 0) ? theta[idx - nx_] : tC;
                const float typ = (j < ny_ - 1) ? theta[idx + nx_] : tC;
                const float Ly = (Dyp * (typ - tC) - Dym * (tC - tym)) / dy2;
                d[i] = beta * tC + Ly;
            }
            thomasSolveSafe(a.data(), b.data(), c.data(), d.data(), x.data(), nx_);
            for (uint i = 0; i < nx_; i++)
                tstar[j * nx_ + i] = x[i];
        }

        // ---- y-implicit half-step: tstar -> theta^{n+1}, then convert to head ----
        for (uint i = 0; i < nx_; i++) {
            for (uint j = 0; j < ny_; j++) {
                const uint lidx = j * nx_ + i;
                const uint idx = base + lidx;
                const float Dc = D[idx];
                const float Dym = (j > 0) ? 0.5f * (D[idx - nx_] + Dc) : 0.0f;
                const float Dyp = (j < ny_ - 1) ? 0.5f * (Dc + D[idx + nx_]) : 0.0f;
                a[j] = -Dym / dy2;
                c[j] = -Dyp / dy2;
                b[j] = beta + (Dym + Dyp) / dy2;
                const float tC = tstar[lidx];
                const float Dxm = (i > 0) ? 0.5f * (D[idx - 1] + Dc) : 0.0f;
                const float Dxp = (i < nx_ - 1) ? 0.5f * (Dc + D[idx + 1]) : 0.0f;
                const float txm = (i > 0) ? tstar[lidx - 1] : tC;
                const float txp = (i < nx_ - 1) ? tstar[lidx + 1] : tC;
                const float Lx = (Dxp * (txp - tC) - Dxm * (tC - txm)) / dx2;
                d[j] = beta * tC + Lx;
            }
            thomasSolveSafe(a.data(), b.data(), c.data(), d.data(), x.data(), ny_);
            // Convert the diffused water content back to pressure head for the columns.
            for (uint j = 0; j < ny_; j++) {
                const uint col = j * nx_ + i;
                const uint idx = base + col;
                const SoilProperties &p = columns_[col]->getLayerSoilProperties(k);
                const float tr = p.theta_r, ts = p.theta_s;
                const float eps = 1e-6f * (ts - tr);
                const float th = std::min(ts, std::max(tr + eps, x[j])); // keep in (theta_r, theta_s]
                h_new_[idx] = theta_to_h(th, p);
            }
        }
    }
    // Final result is in h_new_ (read by unflattenState()).
}

float SoilGrid::computeMaxHorizontalGradient() const {
    float max_grad = 0.0f;

    for (uint k = 0; k < nz_; k++) {
        for (uint j = 0; j < ny_; j++) {
            for (uint i = 0; i < nx_; i++) {
                uint idx = k * n_columns_ + j * nx_ + i;

                if (i < nx_ - 1) {
                    float grad_x = std::fabs(h_flat_[idx + 1] - h_flat_[idx]) / dx_;
                    max_grad = std::max(max_grad, grad_x);
                }
                if (j < ny_ - 1) {
                    float grad_y = std::fabs(h_flat_[idx + nx_] - h_flat_[idx]) / dy_;
                    max_grad = std::max(max_grad, grad_y);
                }
            }
        }
    }

    return max_grad;
}
