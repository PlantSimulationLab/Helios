/** \file "SoilColumn.cpp" Implementation of 1D soil column solver.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilColumn.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include "global.h"

SoilColumn::SoilColumn(uint n_layers, float layer_thickness, const SoilProperties &props) :
    n_layers_(n_layers), top_bc_type_(BCType::NEUMANN), bottom_bc_type_(BCType::FREE_DRAINAGE), top_bc_value_(0.0f), bottom_bc_value_(0.0f), transpiration_(0.0f), cumulative_root_uptake_(0.0f), converged_(false), adaptive_stats_{} {

    if (n_layers < 2) {
        helios::helios_runtime_error("ERROR (SoilColumn::SoilColumn): a column needs at least 2 layers (got " + std::to_string(n_layers) + ").");
    }
    if (!(layer_thickness > 0.0f)) {
        helios::helios_runtime_error("ERROR (SoilColumn::SoilColumn): the layer thickness must be positive (got " + std::to_string(layer_thickness) + " m).");
    }
    validateSoilProperties(props, "SoilColumn::SoilColumn");

    // Initialize uniform soil properties for all layers
    props_.resize(n_layers, props);

    // Initialize uniform grid
    dz_.resize(n_layers, layer_thickness);

    // Compute layer center depths (z positive downward)
    z_.resize(n_layers);
    float depth = 0.0f;
    for (uint k = 0; k < n_layers; k++) {
        depth += dz_[k] / 2.0f;
        z_[k] = depth;
        depth += dz_[k] / 2.0f;
    }

    // Initialize state arrays
    h_.resize(n_layers, -1.0f);
    theta_.resize(n_layers);
    K_.resize(n_layers);
    C_.resize(n_layers);

    // Initialize working arrays
    a_.resize(n_layers, 0.0f);
    b_.resize(n_layers, 0.0f);
    c_.resize(n_layers, 0.0f);
    d_.resize(n_layers, 0.0f);
    x_.resize(n_layers, 0.0f);
    h_old_.resize(n_layers, 0.0f);

    // Initialize root distribution (uniform by default)
    root_dist_.resize(n_layers, 1.0f / static_cast<float>(n_layers));

    // No volumetric sources by default
    source_.resize(n_layers, 0.0f);

    layer_volume_.assign(dz_.begin(), dz_.end());
    face_area_.assign(n_layers + 1, 1.0);

    // Compute initial constitutive values
    updateConstitutive();
}

SoilColumn::SoilColumn(const std::vector<float> &layer_thicknesses, const SoilProperties &props) :
    n_layers_(layer_thicknesses.size()), dz_(layer_thicknesses), top_bc_type_(BCType::NEUMANN), bottom_bc_type_(BCType::FREE_DRAINAGE), top_bc_value_(0.0f), bottom_bc_value_(0.0f), transpiration_(0.0f), cumulative_root_uptake_(0.0f),
    converged_(false), adaptive_stats_{} {

    if (n_layers_ < 2) {
        helios::helios_runtime_error("ERROR (SoilColumn::SoilColumn): a column needs at least 2 layers (got " + std::to_string(n_layers_) + ").");
    }
    for (float thickness: layer_thicknesses) {
        if (!(thickness > 0.0f)) {
            helios::helios_runtime_error("ERROR (SoilColumn::SoilColumn): every layer thickness must be positive (got " + std::to_string(thickness) + " m).");
        }
    }
    validateSoilProperties(props, "SoilColumn::SoilColumn");

    // Initialize uniform soil properties for all layers
    props_.resize(n_layers_, props);

    // Compute layer center depths
    z_.resize(n_layers_);
    float depth = 0.0f;
    for (uint k = 0; k < n_layers_; k++) {
        depth += dz_[k] / 2.0f;
        z_[k] = depth;
        depth += dz_[k] / 2.0f;
    }

    // Initialize state arrays
    h_.resize(n_layers_, -1.0f);
    theta_.resize(n_layers_);
    K_.resize(n_layers_);
    C_.resize(n_layers_);

    // Initialize working arrays
    a_.resize(n_layers_, 0.0f);
    b_.resize(n_layers_, 0.0f);
    c_.resize(n_layers_, 0.0f);
    d_.resize(n_layers_, 0.0f);
    x_.resize(n_layers_, 0.0f);
    h_old_.resize(n_layers_, 0.0f);

    // Initialize root distribution
    root_dist_.resize(n_layers_, 1.0f / static_cast<float>(n_layers_));

    // No volumetric sources by default
    source_.resize(n_layers_, 0.0f);

    layer_volume_.assign(dz_.begin(), dz_.end());
    face_area_.assign(n_layers_ + 1, 1.0);

    updateConstitutive();
}

SoilColumn::~SoilColumn() = default;

// ============================================================================
// Initial and boundary conditions
// ============================================================================

void SoilColumn::setInitialCondition(float h_init) {
    std::fill(h_.begin(), h_.end(), h_init);
    updateConstitutive();
}

void SoilColumn::setInitialCondition(const std::vector<float> &h_profile) {
    if (h_profile.size() != n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::setInitialCondition): initial condition size mismatch.");
    }
    h_ = h_profile;
    updateConstitutive();
}

void SoilColumn::setInitialConditionHydrostatic(float water_table_depth) {
    // Hydrostatic: h = -(z_wt - z) where z_wt is water table depth
    // At water table, h = 0; above water table, h < 0
    for (uint k = 0; k < n_layers_; k++) {
        h_[k] = z_[k] - water_table_depth; // Negative above water table
    }
    updateConstitutive();
}

void SoilColumn::setTopBoundary(BCType type, float value) {
    if (type == BCType::ATMOSPHERIC)
        helios::helios_runtime_error("ERROR (SoilColumn::setTopBoundary): use setTopBoundaryAtmospheric() for an atmospheric boundary.");
    top_bc_type_ = type;
    top_bc_value_ = value;
}

void SoilColumn::setTopBoundaryAtmospheric(float potential_flux, float limiting_head) {
    top_bc_type_ = BCType::ATMOSPHERIC;
    top_bc_value_ = potential_flux;
    top_limit_head_ = limiting_head;
}

double SoilColumn::topFlux(double h0, double K0, double dK0, double &dq_dh0, bool *limited, int mode, double *roundoff) const {
    dq_dh0 = 0.0;
    if (roundoff)
        *roundoff = 0.0;
    if (limited)
        *limited = false;
    if (top_bc_type_ == BCType::NEUMANN)
        return top_bc_value_;
    if (top_bc_type_ != BCType::DIRICHLET && top_bc_type_ != BCType::ATMOSPHERIC)
        return 0.0;
    int w = static_cast<int>(solver_params_.K_weighting);
    if (w < 0)
        w = 1; // KWeighting::DEFAULT -> arithmetic
    const double g = gravity();
    const double hb = (top_bc_type_ == BCType::DIRICHLET) ? (double) top_bc_value_ : (double) top_limit_head_, L = 0.5 * dz_[0];
    double Kb, Cb, dA, dB;
    computeKandC<double>(hb, props_[0], Kb, Cb, 1e-12, (double) solver_params_.K_min); // the boundary carries the cells' conductivity floor
    const double Kf = interblockKDeriv(w, Kb, K0, hb + g * L, h0, props_[0].K_s, props_[0].K_s, dA, dB);
    const double grad = (hb - h0) / L + g, q = Kf * grad;
    const double dq = -Kf / L + dB * dK0 * grad;
    const double ro = kNewtonRoundoffScale * Kf * (std::fabs(hb + g * L) + std::fabs(h0)) / L; // see kNewtonRoundoffScale
    if (top_bc_type_ == BCType::ATMOSPHERIC) {
        const double q_pot = top_bc_value_;
        const bool lim = (mode >= 0) ? (mode == 1) : ((q_pot >= 0.0) ? (q < q_pot) : (q > q_pot)); // the soil cannot take / deliver q_pot
        if (!lim)
            return q_pot;
        if (limited)
            *limited = true;
    }
    dq_dh0 = dq;
    if (roundoff)
        *roundoff = ro;
    return q;
}

float SoilColumn::getTopFlux() const {
    double th, K, C, dK, dq;
    constitutiveAll(h_[0], props_[0], (double) solver_params_.C_regularization, (double) solver_params_.K_min, th, K, C, dK);
    return (float) topFlux(h_[0], K, dK, dq, nullptr, top_atm_applied_);
}

void SoilColumn::setBottomBoundary(BCType type, float value) {
    bottom_bc_type_ = type;
    bottom_bc_value_ = value;
}

void SoilColumn::setRadialGeometry(float inner_radius) {
    if (!(inner_radius > 0.0f)) {
        helios::helios_runtime_error("ERROR (SoilColumn::setRadialGeometry): inner radius must be positive.");
    }
    const double r0 = inner_radius;
    double r = r0;
    for (uint k = 0; k < n_layers_; k++) {
        const double r1 = r + dz_[k];
        face_area_[k] = r / r0;
        layer_volume_[k] = (r1 * r1 - r * r) / (2.0 * r0);
        r = r1;
    }
    face_area_[n_layers_] = r / r0;
    radial_ = true;
}

void SoilColumn::setRootDistribution(const std::vector<float> &root_distribution) {
    if (root_distribution.size() != n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::setRootDistribution): root distribution size mismatch.");
    }
    root_dist_ = root_distribution;

    // Normalize to sum to 1.0
    float sum = std::accumulate(root_dist_.begin(), root_dist_.end(), 0.0f);
    if (sum > 0.0f) {
        for (auto &r: root_dist_) {
            r /= sum;
        }
    }
}

void SoilColumn::setTranspiration(float transpiration) {
    transpiration_ = transpiration;
}

void SoilColumn::setSource(uint layer, float flux) {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::setSource): layer index out of range.");
    }
    source_[layer] = flux;
}

void SoilColumn::addSource(uint layer, float flux) {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::addSource): layer index out of range.");
    }
    source_[layer] += flux;
}

void SoilColumn::clearSources() {
    std::fill(source_.begin(), source_.end(), 0.0f);
}

void SoilColumn::setLayerSoilProperties(uint layer, const SoilProperties &props) {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::setLayerSoilProperties): layer index out of range.");
    }
    validateSoilProperties(props, "SoilColumn::setLayerSoilProperties");
    props_[layer] = props;
    nk_at_h_dbl_ = false;
    updateConstitutive();
}

void SoilColumn::setLayerSoilProperties(uint start_layer, uint end_layer, const SoilProperties &props) {
    if (start_layer >= n_layers_ || end_layer > n_layers_ || start_layer >= end_layer) {
        helios::helios_runtime_error("ERROR (SoilColumn::setLayerSoilProperties): invalid layer range.");
    }
    validateSoilProperties(props, "SoilColumn::setLayerSoilProperties");
    for (uint k = start_layer; k < end_layer; k++) {
        props_[k] = props;
    }
    nk_at_h_dbl_ = false;
    updateConstitutive();
}

const SoilProperties &SoilColumn::getLayerSoilProperties(uint layer) const {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::getLayerSoilProperties): layer index out of range.");
    }
    return props_[layer];
}

void SoilColumn::setSolverParameters(const SolverParameters &params) {
    solver_params_ = params;
    nk_at_h_dbl_ = false; // C_regularization / K_min enter the cached constitutive values
    updateConstitutive();
}

// ============================================================================
// Solver
// ============================================================================

uint SoilColumn::solve(float dt) {
    if (!(dt > 0.0f)) {
        helios::helios_runtime_error("ERROR (SoilColumn::solve): the timestep must be positive (got " + std::to_string(dt) + " s).");
    }
    const bool newton = solver_params_.nonlinear_method == NonlinearMethod::NEWTON_ANALYTIC;
    const std::vector<float> h0 = h_, theta0 = theta_;
    const std::vector<double> hd0 = h_dbl_;
    const float uptake0 = cumulative_root_uptake_;
    auto restore = [&]() {
        h_ = h0;
        theta_ = theta0;
        h_dbl_ = hd0;
        nk_at_h_dbl_ = false;
        cumulative_root_uptake_ = uptake0;
        updateConstitutive();
    };
    const uint it = solveAtmospheric(dt);
    if (it > 0)
        return it;
    restore(); // a failed step leaves the state unchanged (the last Picard iterate can be far off)
    if (!newton || in_adaptive_ || !solver_params_.substep_on_failure || dt <= solver_params_.dt_min * 1.01f)
        return 0;
    // Rather than fail the interval, sub-step it adaptively (e.g. a layer pinned at the saturation kink of a soil
    // with n < 2); if even that cannot cover it, leave the state unchanged. Sub-steps stop at 1/1000 of the interval:
    // an interval that needs smaller ones usually has no solution (e.g. a flux the soil can no longer deliver), and
    // halving down to dt_min would cost many failed solves before the caller can react.
    float sub = 0.25f * dt;
    uint steps = 0;
    const float dt_min = solver_params_.dt_min;
    solver_params_.dt_min = std::max(dt_min, (float) kRescueMinFraction * dt);
    const float advanced = solveAdaptive(dt, sub, &steps);
    solver_params_.dt_min = dt_min;
    if (advanced >= dt * (1.0f - 1e-6f)) {
        converged_ = true;
        return std::max(1u, steps);
    }
    restore();
    converged_ = false;
    return 0;
}

uint SoilColumn::solveAtmospheric(float dt) {
    top_atm_applied_ = -1;
    if (top_bc_type_ != BCType::ATMOSPHERIC)
        return solveStep(dt);
    // Atmospheric boundary: the branch (potential flux or limiting head) is fixed for the step, chosen from the
    // state at its start, and checked against the end state; if it was the wrong one the step is redone with the
    // other. Letting the iterations switch branches (min/max of the two fluxes) stalls the solver where the
    // surface just reaches the limiting head.
    auto limitedAt = [&]() {
        double th, K, C, dK, dq;
        bool lim;
        constitutiveAll(h_[0], props_[0], (double) solver_params_.C_regularization, (double) solver_params_.K_min, th, K, C, dK);
        topFlux(h_[0], K, dK, dq, &lim);
        return lim ? 1 : 0;
    };
    const std::vector<float> h0 = h_, theta0 = theta_;
    const std::vector<double> hd0 = h_dbl_;
    const float uptake0 = cumulative_root_uptake_;
    int mode = limitedAt();
    uint it = 0;
    for (int attempt = 0; attempt < 2; attempt++) {
        top_atm_mode_ = mode;
        it = solveStep(dt);
        top_atm_mode_ = -1;
        top_atm_applied_ = mode;
        if (it == 0 || limitedAt() == mode || attempt == 1)
            break; // failed (the caller retries), consistent, or at the switching point either way
        h_ = h0; // wrong branch: redo the step with the other one
        theta_ = theta0;
        h_dbl_ = hd0;
        nk_at_h_dbl_ = false;
        cumulative_root_uptake_ = uptake0;
        updateConstitutive();
        mode = 1 - mode;
    }
    return it;
}

uint SoilColumn::solveStep(float dt) {
    if (solver_params_.nonlinear_method == NonlinearMethod::NEWTON_ANALYTIC) {
        // Newton is fast and robust in general; on the rare step where it stalls (e.g. a layer pinned at
        // the saturation kink of a soil with n < 2) fall back to Modified Picard for this step.
        const uint it = solveNewton(dt);
        if (it > 0)
            return it;
        h_dbl_.clear();
    }

    const bool newton_fallback = solver_params_.nonlinear_method == NonlinearMethod::NEWTON_ANALYTIC;
    converged_ = false;

    // Water content at the start of the time step (theta^n); constant over the Picard iterations
    std::vector<float> theta_n(n_layers_);
    for (uint k = 0; k < n_layers_; k++)
        theta_n[k] = h_to_theta(h_[k], props_[k]);
    const std::vector<float> h_start = h_;
    float omega_adapt = 1.0f, max_dh_prev = 0.0f;

    // Modified Picard iteration
    for (uint iter = 0; iter < solver_params_.max_picard_iterations; iter++) {
        // Store previous iteration (for convergence check)
        h_old_ = h_;

        // Update constitutive relationships at current h
        updateConstitutive();

        // Assemble and solve the tridiagonal system (theta^n fixed for correct implicit Euler)
        assembleTridiagonal(dt, theta_n);

        // Make copies for Thomas algorithm (which modifies c and d)
        std::vector<float> c_copy = c_;
        std::vector<float> d_copy = d_;

        thomasSolve(a_.data(), b_.data(), c_copy.data(), d_copy.data(), x_.data(), n_layers_);

        // A non-finite iterate can never converge (and NaN slips through every tolerance
        // comparison): fail the step and restore the start-of-step state.
        bool finite = true;
        for (uint k = 0; k < n_layers_; k++)
            finite = finite && std::isfinite(x_[k]);
        if (!finite) {
            h_ = h_start;
            updateConstitutive();
            return 0;
        }

        // Guard against overshoot in dry layers: when the raw head step exceeds the layer's own
        // |h| (or 1 m), apply it in water-content space instead -- the linear solve implies
        // theta = theta^m + C^m (x - h^m); map that back through the retention curve, clamped to
        // (theta_r, theta_s). In dry soil C ~ 0, so the raw update can overshoot by orders of
        // magnitude and oscillate. Ordinary steps are left alone (mapping every step slows
        // Picard convergence on wetting fronts). The fixed point is unchanged.
        for (uint k = 0; k < n_layers_; k++) {
            if (h_[k] >= (float) saturationHead(props_[k]) || props_[k].model == SoilModel::GARDNER || props_[k].model == SoilModel::LINEAR)
                continue;
            if (std::fabs(x_[k] - h_[k]) <= std::max(1.0f, std::fabs(h_[k])))
                continue;
            const float th = theta_[k] + C_[k] * (x_[k] - h_[k]);
            const float ts = props_[k].theta_s, tr = props_[k].theta_r;
            if (th < ts)
                x_[k] = theta_to_h(std::max(th, tr + 1e-6f * (ts - tr)), props_[k]);
        }

        // Update h with solution using underrelaxation
        // h_new = omega * x + (1 - omega) * h_old
        // omega = 1.0 gives standard Picard, omega < 1 adds damping
        float omega = std::min(solver_params_.picard_omega, omega_adapt);
        if (omega >= 1.0f) {
            h_ = x_;
        } else {
            for (uint k = 0; k < n_layers_; k++) {
                h_[k] = omega * x_[k] + (1.0f - omega) * h_old_[k];
            }
        }

        // Converged when, in every layer, the head change is within picard_tolerance (relative to
        // that layer's own |h|, so very dry layers are not held to an unreachable absolute 1e-6 m)
        // AND the water-content change is below kThetaTol. The water-content test is what makes
        // convergence mean "water balance satisfied": a head-only test, or one scaled by the
        // column-wide max |h|, accepts iterates where one runaway cell has absurd h. An unsaturated
        // layer whose head change moves a negligible amount of water (C |dh| << kThetaTol) also
        // passes the head test: where C is tiny, single-precision quantization of theta makes the
        // iterate cycle at the ~1e-6 m level and a pure head tolerance can never be met.
        constexpr float kThetaTol = 1e-6f;
        bool h_ok = true;
        float max_dtheta = 0.0f;
        for (uint k = 0; k < n_layers_; k++) {
            const float dh = std::fabs(h_[k] - h_old_[k]);
            if (dh > solver_params_.picard_tolerance * std::max(1.0f, std::fabs(h_[k])) && !(h_[k] < (float) saturationHead(props_[k]) && C_[k] * dh < 0.1f * kThetaTol))
                h_ok = false;
            max_dtheta = std::max(max_dtheta, std::fabs(h_to_theta(h_[k], props_[k]) - theta_[k]));
        }

        // Adaptive under-relaxation: once the iteration stops contracting (the largest head change
        // fails to shrink), damp subsequent updates. Sharp retention curves (large alpha*n, air entry
        // of millimetres) otherwise make Modified Picard oscillate between two states.
        float max_dh = 0.0f;
        for (uint k = 0; k < n_layers_; k++)
            max_dh = std::max(max_dh, std::fabs(h_[k] - h_old_[k]));
        if (iter > 1 && max_dh > 0.9f * max_dh_prev)
            omega_adapt = std::max(0.25f, 0.7f * omega_adapt);
        max_dh_prev = max_dh;

        if (h_ok && max_dtheta < kThetaTol) {
            // As a fallback for a failed Newton step, also require the column's water balance to close (in double
            // precision): damped single-precision iterates can stop changing well away from the solution.
            if (newton_fallback) {
                std::vector<double> hd(h_.begin(), h_.end()), thn(theta_n.begin(), theta_n.end()), F, scale;
                assembleNewton(hd, thn, dt, F, scale, nullptr, nullptr, nullptr);
                nk_at_h_dbl_ = false;
                double imbalance = 0.0, capacity = 0.0;
                for (uint k = 0; k < n_layers_; k++) {
                    imbalance += F[k];
                    capacity += layer_volume_[k];
                }
                if (!(std::fabs(imbalance) * dt <= kFallbackBalanceTol * capacity)) {
                    h_ = h_start;
                    updateConstitutive();
                    return 0;
                }
            }
            converged_ = true;
            updateWaterContent();

            // Accumulate root uptake
            for (uint k = 0; k < n_layers_; k++) {
                cumulative_root_uptake_ += transpiration_ * root_dist_[k] * dt;
            }

            return iter + 1;
        }
    }

    // Did not converge - still update theta for mass balance tracking
    updateConstitutive();
    updateWaterContent();

    return 0; // 0 indicates non-convergence
}

// ============================================================================
// Newton-Raphson (NonlinearMethod::NEWTON_ANALYTIC, the default)
// ============================================================================

// Residual (per unit ground area [m/s]) of every layer and its tridiagonal Jacobian, in double precision.
// Layer k: dz (theta - theta^n)/dt + q_out(k) - q_in(k) + sink - source = 0, with downward face flux
// q = Kf ((h_above - h_below)/L + 1) between layers at centre distance L, Kf from the selected weighting.
// constitutive_current: nk_* already hold the constitutive values at h (skips their evaluation).
void SoilColumn::assembleNewton(const std::vector<double> &h, const std::vector<double> &theta_n, double dt, std::vector<double> &F, std::vector<double> &scale, std::vector<double> *lo, std::vector<double> *di, std::vector<double> *up,
                                bool constitutive_current) const {
    const uint N = n_layers_;
    int w = static_cast<int>(solver_params_.K_weighting);
    if (w < 0)
        w = 1; // KWeighting::DEFAULT -> arithmetic
    std::vector<double> &th = nk_th_, &K = nk_K_, &C = nk_C_, &dK = nk_dK_;
    if (!constitutive_current) {
        th.resize(N);
        K.resize(N);
        C.resize(N);
        dK.resize(N);
        for (uint k = 0; k < N; k++)
            constitutiveAll(h[k], props_[k], (double) solver_params_.C_regularization, (double) solver_params_.K_min, th[k], K[k], C[k], dK[k]);
    }
    F.assign(N, 0.0);
    scale.assign(N, 0.0);
    if (di) {
        lo->assign(N, 0.0);
        di->assign(N, 0.0);
        up->assign(N, 0.0);
    }
    const double idt = 1.0 / dt;
    for (uint k = 0; k < N; k++) {
        const double V = layer_volume_[k];
        const double sto = V * (th[k] - theta_n[k]) * idt;
        const double ext = (double) source_[k] - (double) transpiration_ * root_dist_[k];
        F[k] += sto - ext;
        scale[k] += std::fabs(V * th[k] * idt) + std::fabs(V * theta_n[k] * idt) + std::fabs(ext);
        if (di)
            (*di)[k] += V * C[k] * idt;
    }
    // Interior faces: between k (above) and k+1 (below); downward flux leaves k and enters k+1.
    const double g = gravity(); // gravity term of the flux
    for (uint k = 0; k + 1 < N; k++) {
        const double L = 0.5 * (dz_[k] + dz_[k + 1]);
        const double Ha = h[k] + g * L, Hb = h[k + 1]; // total heads relative to the lower centre
        double dA, dB;
        const double Kf = interblockKDeriv(w, K[k], K[k + 1], Ha, Hb, props_[k].K_s, props_[k + 1].K_s, dA, dB);
        const double grad = (h[k] - h[k + 1]) / L + g, A = face_area_[k + 1];
        const double q = A * Kf * grad;
        // Round-off floor of q: its head difference cancels terms of size |H| (matters for very thin layers)
        const double qr = std::fabs(q) + kNewtonRoundoffScale * A * Kf * (std::fabs(Ha) + std::fabs(Hb)) / L;
        F[k] += q;
        F[k + 1] -= q;
        scale[k] += qr;
        scale[k + 1] += qr;
        if (di) {
            const double dq_da = A * (Kf / L + dA * dK[k] * grad), dq_db = A * (-Kf / L + dB * dK[k + 1] * grad);
            (*di)[k] += dq_da;
            (*up)[k] += dq_db;
            (*lo)[k + 1] -= dq_da;
            (*di)[k + 1] -= dq_db;
        }
    }
    // Top face (into layer 0).
    {
        double dq, ro;
        const double q = face_area_[0] * topFlux(h[0], K[0], dK[0], dq, nullptr, top_atm_mode_, &ro);
        F[0] -= q;
        scale[0] += std::fabs(q) + face_area_[0] * ro;
        if (di)
            (*di)[0] -= face_area_[0] * dq;
    }
    // Bottom face (out of layer N-1).
    const uint b = N - 1;
    const double Ab = face_area_[N];
    if (bottom_bc_type_ == BCType::NEUMANN) {
        F[b] += Ab * bottom_bc_value_; // positive out of the domain
        scale[b] += std::fabs(Ab * bottom_bc_value_);
    } else if (bottom_bc_type_ == BCType::DIRICHLET) {
        const double hb = bottom_bc_value_, L = 0.5 * dz_[b];
        double Kb, Cb, dA, dB;
        computeKandC<double>(hb, props_[b], Kb, Cb, 1e-12, (double) solver_params_.K_min);
        const double Kf = interblockKDeriv(w, K[b], Kb, h[b] + g * L, hb, props_[b].K_s, props_[b].K_s, dA, dB);
        const double grad = (h[b] - hb) / L + g, q = Ab * Kf * grad;
        F[b] += q;
        scale[b] += std::fabs(q) + kNewtonRoundoffScale * Ab * Kf * (std::fabs(h[b] + g * L) + std::fabs(hb)) / L;
        if (di)
            (*di)[b] += Ab * (Kf / L + dA * dK[b] * grad);
    } else if (bottom_bc_type_ == BCType::FREE_DRAINAGE) {
        F[b] += Ab * g * K[b];
        scale[b] += Ab * g * K[b];
        if (di)
            (*di)[b] += Ab * g * dK[b];
    }
}

// Same algorithm as SoilSolver3D::solveNewtonAnalytic(): exact (Thomas) Newton steps on the tridiagonal
// Jacobian, the update applied in water-content space for unsaturated layers with the saturation change
// per iteration capped, a two-pass backtracking line search, and convergence when every layer's water
// imbalance is negligible and its head (or the water its head change moves) has settled.
uint SoilColumn::solveNewton(float dt_f) {
    converged_ = false;
    const uint N = n_layers_;
    const double dt = dt_f;
    // Work in double precision across steps: start from the double-precision state of the previous Newton
    // step unless the (float) heads were changed in between (initial conditions, restores). Rounding the
    // converged heads to float each step leaves a residual floor that sharp retention curves cannot get below.
    bool reuse = h_dbl_.size() == N;
    for (uint k = 0; k < N && reuse; k++)
        reuse = (float) h_dbl_[k] == h_[k];
    // The previous step's end state is this step's start state: when its constitutive values were kept
    // (nk_at_h_dbl_), theta^n and the first assembly need no new evaluations.
    const bool cached = reuse && nk_at_h_dbl_ && nk_th_.size() == N;
    nk_at_h_dbl_ = false;
    std::vector<double> &h = nw_h_, &theta_n = nw_theta_n_, &F = nw_F_, &Ft = nw_Ft_, &Fb = nw_Fb_, &scale = nw_scale_, &scale_t = nw_scale_t_, &sb = nw_sb_;
    std::vector<double> &lo = nw_lo_, &di = nw_di_, &up = nw_up_, &lo_t = nw_lo_t_, &di_t = nw_di_t_, &up_t = nw_up_t_;
    std::vector<double> &delta = nw_delta_, &ht = nw_ht_, &hb = nw_hb_, &cl = nw_cl_, &dl = nw_dl_, &th_h = nw_th_h_, &C_h = nw_C_h_;
    h.resize(N);
    theta_n.resize(N);
    for (std::vector<double> *v: {&delta, &ht, &hb, &cl, &dl})
        v->resize(N);
    for (uint k = 0; k < N; k++) {
        h[k] = reuse ? h_dbl_[k] : (double) h_[k];
        theta_n[k] = cached ? nk_th_[k] : waterContent<double>(h[k], props_[k]);
    }
    const double theta_tol = 1e-7, rel_tol = kNewtonRelTol, kMaxDSe = 0.2;
    const double tolA = theta_tol / dt; // per-step water imbalance [m/s per m of layer], scaled by dz below
    auto norm = [](const std::vector<double> &x) {
        double s = 0.0;
        for (double v: x)
            s += v * v;
        return std::sqrt(s);
    };
    assembleNewton(h, theta_n, dt, F, scale, &lo, &di, &up, cached);
    double fnorm = norm(F);
    double hist[6]; // residual norms of the last 6 iterations (stall test)
    int creep = 0; // consecutive damped iterations with little progress
    bool dh_ok = false;
    const uint max_iter = std::max<uint>(solver_params_.max_picard_iterations, 30);
    for (uint it = 0; it < max_iter; it++) {
        if (!std::isfinite(fnorm))
            break;
        if (it >= 6 && fnorm > 0.99 * hist[it % 6])
            break; // stalled
        hist[it % 6] = fnorm;
        bool res_ok = true, strict_ok = true;
        for (uint k = 0; k < N && res_ok; k++) {
            res_ok = std::fabs(F[k]) <= std::max(tolA * layer_volume_[k], rel_tol * scale[k]);
            strict_ok = strict_ok && std::fabs(F[k]) <= rel_tol * scale[k];
        }
        bool accept = res_ok && (it == 0 ? strict_ok : dh_ok);
        // Newton step: J delta = -F (Thomas algorithm; J is tridiagonal)
        if (!accept) {
            for (uint k = 0; k < N; k++) {
                cl[k] = up[k];
                dl[k] = -F[k];
            }
            for (uint k = 1; k < N; k++) {
                const double m = lo[k] / di[k - 1];
                di[k] -= m * cl[k - 1];
                dl[k] -= m * dl[k - 1];
            }
            delta[N - 1] = dl[N - 1] / di[N - 1];
            for (int k = (int) N - 2; k >= 0; k--)
                delta[k] = (dl[k] - cl[k] * delta[k + 1]) / di[k];
            // Residual already acceptable and the next correction below the head tolerance: apply it and
            // stop, without assembling once more just to confirm the head has settled.
            if (res_ok && it > 0) {
                bool small = true;
                for (uint k = 0; k < N && small; k++)
                    small = std::fabs(delta[k]) <= solver_params_.picard_tolerance * std::max(1.0, std::fabs(h[k])) || (h[k] < saturationHead(props_[k]) && nk_C_[k] * std::fabs(delta[k]) < 0.1 * theta_tol);
                if (small) {
                    for (uint k = 0; k < N; k++)
                        h[k] += delta[k];
                    // h moved: evaluate its constitutive values once (kept for the next step)
                    for (uint k = 0; k < N; k++)
                        constitutiveAll(h[k], props_[k], (double) solver_params_.C_regularization, (double) solver_params_.K_min, nk_th_[k], nk_K_[k], nk_C_[k], nk_dK_[k]);
                    accept = true;
                }
            }
        }
        if (accept) { // nk_* hold the constitutive values at h
            h_dbl_ = h;
            nk_at_h_dbl_ = true;
            for (uint k = 0; k < N; k++) {
                h_[k] = (float) h[k];
                theta_[k] = (float) nk_th_[k];
                K_[k] = (float) nk_K_[k];
                C_[k] = (float) nk_C_[k];
            }
            converged_ = true;
            for (uint k = 0; k < N; k++)
                cumulative_root_uptake_ += transpiration_ * root_dist_[k] * dt_f;
            return std::max(1u, it);
        }
        th_h = nk_th_; // at h (the last assembly); trials overwrite nk_*
        C_h = nk_C_;
        auto trial = [&](double lambda, bool mapped) {
            for (uint k = 0; k < N; k++) {
                const double d = lambda * delta[k];
                if (!mapped) {
                    ht[k] = h[k] + d;
                    continue;
                }
                const SoilProperties &p = props_[k];
                const double tr = p.theta_r, ts = p.theta_s, range = ts - tr, hs = saturationHead(p); // saturated for h >= hs
                if (h[k] < hs) {
                    // Tiny steps that stay unsaturated (typical of long runs of short fixed steps): the mapped
                    // and raw updates agree to second order, so skip the retention-curve inversion. Looser
                    // thresholds (1e-2) cost an extra iteration on wetting fronts.
                    const double hn = h[k] + d;
                    if (hn < hs && std::fabs(d) <= 1e-4 * std::max(1.0, -h[k]) && std::fabs(C_h[k] * d) <= 1e-4 * range) {
                        ht[k] = hn;
                        continue;
                    }
                    const double Se0 = (th_h[k] - tr) / range;
                    double Se = (th_h[k] + C_h[k] * d - tr) / range;
                    Se = std::min(std::max(Se, Se0 - kMaxDSe), Se0 + kMaxDSe);
                    ht[k] = (Se >= 1.0) ? hs : pressureHeadOfTheta(tr + std::max(Se, 1e-9) * range, p);
                } else {
                    const double hn = h[k] + d;
                    ht[k] = (hn < hs) ? (h[k] > hs ? hs : std::max(hn, pressureHeadOfTheta(tr + (1.0 - kMaxDSe) * range, p))) : hn;
                }
            }
        };
        double best = INFINITY;
        bool decreased = false, jac_current = false;
        for (int pass = 0; pass < 2 && !decreased; pass++) {
            double lambda = 1.0;
            for (int ls = 0; ls < 8; ls++) {
                trial(lambda, pass == 0);
                // The full mapped step is usually accepted: assemble its Jacobian now so it need not be
                // re-assembled after acceptance.
                const bool first = pass == 0 && ls == 0;
                if (first)
                    assembleNewton(ht, theta_n, dt, Ft, scale_t, &lo_t, &di_t, &up_t);
                else
                    assembleNewton(ht, theta_n, dt, Ft, scale_t, nullptr, nullptr, nullptr);
                const double ftn = norm(Ft);
                if (ftn < best) {
                    best = ftn;
                    hb = ht;
                    Fb = Ft;
                    sb = scale_t;
                }
                if (ftn <= (1.0 - 1e-4 * lambda) * fnorm) {
                    decreased = true;
                    jac_current = first;
                    break;
                }
                lambda *= 0.5;
            }
        }
        if (!std::isfinite(best))
            break;
        dh_ok = true;
        for (uint k = 0; k < N; k++) {
            const double dhk = std::fabs(hb[k] - h[k]);
            if (dhk > solver_params_.picard_tolerance * std::max(1.0, std::fabs(h[k])) && !(h[k] < saturationHead(props_[k]) && C_h[k] * dhk < 0.1 * theta_tol))
                dh_ok = false;
        }
        const double fnorm_prev = fnorm;
        h.swap(hb);
        if (jac_current) {
            F.swap(Fb);
            scale.swap(sb);
            lo.swap(lo_t);
            di.swap(di_t);
            up.swap(up_t);
            fnorm = best;
        } else {
            assembleNewton(h, theta_n, dt, F, scale, &lo, &di, &up);
            fnorm = norm(F);
        }
        // Creeping: several damped iterations in a row that each remove under 10% of the residual. This is how the
        // iteration behaves when the step has no solution (e.g. a flux the soil can no longer deliver); fail early
        // so the caller can shorten the step.
        creep = (!jac_current && fnorm > kNewtonCreepRatio * fnorm_prev) ? creep + 1 : 0;
        if (creep >= kNewtonCreepIterations)
            break;
    }
    updateConstitutive(); // not converged: h_ unchanged
    return 0;
}

// ============================================================================
// State access
// ============================================================================

float SoilColumn::getColumnDepth() const {
    return std::accumulate(dz_.begin(), dz_.end(), 0.0f);
}

float SoilColumn::getPressureHead(uint layer) const {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::getPressureHead): layer index out of range.");
    }
    return h_[layer];
}

float SoilColumn::getWaterContent(uint layer) const {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::getWaterContent): layer index out of range.");
    }
    return h_to_theta(h_[layer], props_[layer]);
}

std::vector<float> SoilColumn::getWaterContents() const {
    std::vector<float> theta(n_layers_);
    for (uint k = 0; k < n_layers_; k++) {
        theta[k] = h_to_theta(h_[k], props_[k]);
    }
    return theta;
}

float SoilColumn::getConductivity(uint layer) const {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::getConductivity): layer index out of range.");
    }
    return K_[layer];
}

double SoilColumn::getTotalWaterStorage() const {
    // In double precision, from the double-precision heads that Newton carries between steps when they are current
    // (the same test solveNewton() uses); summing single-precision water contents loses the balance of long runs.
    bool double_heads = h_dbl_.size() == n_layers_;
    for (uint k = 0; k < n_layers_ && double_heads; k++)
        double_heads = (float) h_dbl_[k] == h_[k];
    double storage = 0.0;
    for (uint k = 0; k < n_layers_; k++) {
        const double h = double_heads ? h_dbl_[k] : (double) h_[k];
        storage += waterContent<double>(h, props_[k]) * layer_volume_[k];
    }
    return storage;
}

float SoilColumn::getLayerDepth(uint layer) const {
    if (layer >= n_layers_) {
        helios::helios_runtime_error("ERROR (SoilColumn::getLayerDepth): layer index out of range.");
    }
    return z_[layer];
}

float SoilColumn::getWeightedSoilPotential() const {
    // Uptake-weighted soil water potential
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (uint k = 0; k < n_layers_; k++) {
        float uptake = root_dist_[k] * transpiration_;
        numerator += uptake * h_[k];
        denominator += uptake;
    }

    if (denominator > 0.0f) {
        return numerator / denominator;
    } else {
        // No uptake - return average potential in root zone
        float sum_h = 0.0f;
        float sum_weight = 0.0f;
        for (uint k = 0; k < n_layers_; k++) {
            if (root_dist_[k] > 0.0f) {
                sum_h += h_[k] * root_dist_[k];
                sum_weight += root_dist_[k];
            }
        }
        return sum_weight > 0.0f ? sum_h / sum_weight : h_[0];
    }
}

// ============================================================================
// Internal methods
// ============================================================================

void SoilColumn::updateConstitutive() {
    // Fused K(h) and C(h) from shared intermediates (one Se evaluation for both,
    // ~3 pow + 1 sqrt instead of ~6 pow when computed separately) — the same trick
    // the full-3D solver uses. This is the constitutive hot path (every Picard
    // iteration of every column), so the saving compounds across the whole solve.
    for (uint k = 0; k < n_layers_; k++) {
        theta_[k] = h_to_theta(h_[k], props_[k]);
        computeKandC<float>(h_[k], props_[k], K_[k], C_[k], solver_params_.C_regularization, solver_params_.K_min);
    }
}

void SoilColumn::assembleTridiagonal(float dt, const std::vector<float> &theta_n) {
    float idt = 1.0f / dt;

    // Mass-conservative Modified Picard (Celia et al., 1990): the storage term
    // (theta^{m+1} - theta^n)/dt is linearized about the current iterate h^m as
    // C^m (h^{m+1} - h^m)/dt + (theta^m - theta^n)/dt. The C^m h^{m+1} part sits on the diagonal
    // (b_), the rest on the RHS. The pressure-head form C (h^{m+1} - h^n)/dt is NOT conservative:
    // in dry soil C ~ 0, so a small influx drives h to unphysical values.
    std::vector<float> &storage = storage_rhs_;
    storage.resize(n_layers_);
    for (uint k = 0; k < n_layers_; k++)
        storage[k] = (float) layer_volume_[k] * idt * (C_[k] * h_[k] - (theta_[k] - theta_n[k]));

    int K_method = static_cast<int>(solver_params_.K_weighting);
    if (K_method < 0)
        K_method = 1; // KWeighting::DEFAULT -> arithmetic
    const float g = (float) gravity(); // gravity term of the flux

    // Interblock conductivity between layer ka and the layer kb directly below it. Upstream-type
    // weightings pick the donor by total head (h + elevation), so upward flow (evaporation,
    // capillary rise) takes the lower layer's K rather than always the upper one.
    auto faceK = [&](uint ka, uint kb) -> float {
        if (K_method == 2 || K_method == 4) {
            const bool down = h_[ka] + g * 0.5f * (dz_[ka] + dz_[kb]) >= h_[kb];
            const uint up = down ? ka : kb;
            if (K_method == 2)
                return K_[up];
            const float Ksa = props_[ka].K_s, Ksb = props_[kb].K_s;
            const float Ksh = (Ksa + Ksb > 0.0f) ? 2.0f * Ksa * Ksb / (Ksa + Ksb) : 0.0f;
            return Ksh * K_[up] / props_[up].K_s;
        }
        return interblockK_weighted(K_[ka], K_[kb], K_method);
    };

    // Interior nodes (k = 1 to n_layers-2)
    for (uint k = 1; k < n_layers_ - 1; k++) {
        float dz = (float) layer_volume_[k]; // storage weight (layer thickness unless radial)
        float dz_above = 0.5f * (dz_[k - 1] + dz_[k]);
        float dz_below = 0.5f * (dz_[k] + dz_[k + 1]);

        // Interblock conductivities with specified weighting
        // For downward flow: upstream is the cell above
        float K_above = (float) face_area_[k] * faceK(k - 1, k);
        float K_below = (float) face_area_[k + 1] * faceK(k, k + 1);

        // Sink term (root uptake): the layer's extraction FLUX [m/s] (depth of water per
        // time per unit ground area), = column transpiration [m/s] x its root fraction.
        // It enters the per-area flux balance directly (NO *dz) -- exactly like the
        // Neumann boundary fluxes top_bc_value_/bottom_bc_value_ do.
        float sink = transpiration_ * root_dist_[k];

        // Tridiagonal coefficients
        // a[k] * h[k-1] + b[k] * h[k] + c[k] * h[k+1] = d[k]
        a_[k] = -K_above / dz_above;
        c_[k] = -K_below / dz_below;
        b_[k] = dz * C_[k] * idt - a_[k] - c_[k];

        // RHS: Modified-Picard storage term + gravity flux terms + sink
        // Gravity: inflow from above = K_above, outflow below = K_below
        // Net gravity contribution = K_above - K_below (positive = net inflow)
        d_[k] = storage[k] + g * (K_above - K_below) - sink;
    }

    // Top boundary (k = 0)
    {
        uint k = 0;
        float dz = (float) layer_volume_[k]; // storage weight (layer thickness unless radial)
        float dz_below = 0.5f * (dz_[k] + dz_[k + 1]);
        float K_below = (float) face_area_[1] * faceK(k, k + 1);
        float sink = transpiration_ * root_dist_[k];

        a_[k] = 0.0f; // No node above
        c_[k] = -K_below / dz_below;

        // An atmospheric boundary is a Neumann (potential flux) or Dirichlet (limiting head) boundary, whichever
        // limits at the current iterate
        BCType type = top_bc_type_;
        float value = top_bc_value_;
        if (type == BCType::ATMOSPHERIC) {
            double th0, K0, C0, dK0, dq;
            bool limited;
            constitutiveAll(h_[0], props_[0], (double) solver_params_.C_regularization, (double) solver_params_.K_min, th0, K0, C0, dK0);
            topFlux(h_[0], K0, dK0, dq, &limited, top_atm_mode_);
            type = limited ? BCType::DIRICHLET : BCType::NEUMANN;
            value = limited ? top_limit_head_ : top_bc_value_;
        }

        if (type == BCType::NEUMANN) {
            // Flux BC: q_top = top_bc_value_ (positive into soil)
            // Gravity flux: no inflow from above, outflow to below = K_below
            // Net gravity contribution = 0 - K_below = -K_below
            b_[k] = dz * C_[k] * idt - c_[k];
            d_[k] = storage[k] + (float) face_area_[0] * value - g * K_below - sink;
        } else if (type == BCType::DIRICHLET) {
            // Pressure head BC: h_top = value
            float K_top = computeK(value, props_[0]) + solver_params_.K_min; // the cells' conductivity floor applies at the boundary too
            // For Dirichlet BC, use upstream weighting (K_top is upstream for infiltration)
            float K_above = (float) face_area_[0] * interblockK_weighted(K_top, K_[k], K_method);
            float dz_above = dz_[k] / 2.0f; // Half cell to boundary

            b_[k] = dz * C_[k] * idt + K_above / dz_above - c_[k];
            // Gravity: inflow from above = K_above, outflow to below = K_below
            d_[k] = storage[k] + K_above / dz_above * value + g * (K_above - K_below) - sink;
        } else {
            // Free drainage at top doesn't make physical sense, treat as zero flux
            // Gravity: no inflow from above, outflow to below = K_below
            b_[k] = dz * C_[k] * idt - c_[k];
            d_[k] = storage[k] - g * K_below - sink;
        }
    }

    // Bottom boundary (k = n_layers - 1)
    {
        uint k = n_layers_ - 1;
        float dz = (float) layer_volume_[k]; // storage weight (layer thickness unless radial)
        float dz_above = 0.5f * (dz_[k - 1] + dz_[k]);
        float K_above = (float) face_area_[k] * faceK(k - 1, k);
        float sink = transpiration_ * root_dist_[k];

        c_[k] = 0.0f; // No node below
        a_[k] = -K_above / dz_above;

        if (bottom_bc_type_ == BCType::NEUMANN) {
            // Flux BC: q_bottom = bottom_bc_value_ (positive out of domain)
            // Gravity: inflow from above = K_above, outflow = bottom_bc_value_
            b_[k] = dz * C_[k] * idt - a_[k];
            d_[k] = storage[k] + g * K_above - (float) face_area_[n_layers_] * bottom_bc_value_ - sink;
        } else if (bottom_bc_type_ == BCType::DIRICHLET) {
            // Pressure head BC
            float K_bot = computeK(bottom_bc_value_, props_[n_layers_ - 1]) + solver_params_.K_min;
            float K_below = (float) face_area_[n_layers_] * interblockK_weighted(K_[k], K_bot, K_method);
            float dz_below = dz_[k] / 2.0f;

            // Gravity: inflow from above = K_above, outflow to below = K_below
            b_[k] = dz * C_[k] * idt - a_[k] + K_below / dz_below;
            d_[k] = storage[k] + K_below / dz_below * bottom_bc_value_ + g * (K_above - K_below) - sink;
        } else { // FREE_DRAINAGE
            // Unit gradient: dh/dz = 0, so flux = K (gravity only)
            // Gravity flux: inflow from above = K_above, outflow at bottom = K_[k]
            // Net gravity contribution = K_above - K_[k] (positive when inflow > outflow)
            b_[k] = dz * C_[k] * idt - a_[k];
            d_[k] = storage[k] + g * (K_above - (float) face_area_[n_layers_] * K_[k]) - sink;
        }
    }

    // Volumetric sources (drip emitters): a flux [m/s] injected into each layer,
    // entering the per-area flux balance exactly like the Neumann top influx
    // top_bc_value_ does (positive = into soil). Zero for layers without an emitter.
    for (uint k = 0; k < n_layers_; k++) {
        d_[k] += source_[k];
    }
}

float SoilColumn::computeResidual() const {
    float max_residual = 0.0f;
    for (uint k = 0; k < n_layers_; k++) {
        float res = a_[k] * (k > 0 ? h_[k - 1] : 0.0f) + b_[k] * h_[k] + c_[k] * (k < n_layers_ - 1 ? h_[k + 1] : 0.0f) - d_[k];
        max_residual = std::max(max_residual, std::fabs(res));
    }
    return max_residual;
}

void SoilColumn::updateWaterContent() {
    for (uint k = 0; k < n_layers_; k++) {
        theta_[k] = h_to_theta(h_[k], props_[k]);
    }
}

// ============================================================================
// Adaptive timestepping
// ============================================================================

float SoilColumn::solveAdaptive(float target_time, float &dt, uint *steps_taken) {
    struct AdaptiveScope { // lets solve() know a failed step will be retried with a smaller dt
        bool &flag;
        explicit AdaptiveScope(bool &f) : flag(f) {
            flag = true;
        }
        ~AdaptiveScope() {
            flag = false;
        }
    } scope(in_adaptive_);
    // Initialize statistics
    adaptive_stats_ = AdaptiveSolveStats{};
    adaptive_stats_.min_dt_used = dt;
    adaptive_stats_.max_dt_used = dt;

    float time_advanced = 0.0f;

    // Clamp initial dt to bounds
    dt = std::max(solver_params_.dt_min, std::min(dt, solver_params_.dt_max));

    bool rejected = false;
    while (time_advanced < target_time) {
        // Adjust dt if we're close to target time
        float remaining = target_time - time_advanced;
        float dt_step = std::min(dt, remaining);

        // Save state in case we need to reject the timestep
        std::vector<float> h_saved = h_;
        std::vector<float> theta_saved = theta_;
        float cumulative_uptake_saved = cumulative_root_uptake_;

        // Attempt to solve
        uint iterations = solve(dt_step);

        if (iterations == 0) {
            // Failed to converge
            adaptive_stats_.failed_steps++;

            // Already at the minimum dt: stop and return the time actually advanced, rather than let the
            // clock run on with a state that did not advance.
            if (dt_step <= solver_params_.dt_min * 1.01f) {
                h_ = h_saved;
                theta_ = theta_saved;
                cumulative_root_uptake_ = cumulative_uptake_saved;
                updateConstitutive();
                break;
            }

            // Restore state to retry with a smaller dt
            h_ = h_saved;
            theta_ = theta_saved;
            cumulative_root_uptake_ = cumulative_uptake_saved;
            updateConstitutive();

            // Reduce dt, but cap the reduction so a single slow step can't crater dt.
            float reduction = std::max(solver_params_.dt_failure_factor, solver_params_.dt_reduction_min);
            dt = std::max(dt * reduction, solver_params_.dt_min);
            rejected = true;

            continue; // Retry with smaller dt
        }

        // Successful step - accept it
        time_advanced += dt_step;
        adaptive_stats_.total_steps++;
        adaptive_stats_.total_iterations += iterations;
        adaptive_stats_.min_dt_used = std::min(adaptive_stats_.min_dt_used, dt_step);
        adaptive_stats_.max_dt_used = std::max(adaptive_stats_.max_dt_used, dt_step);

        // ----- Relative error-based step-size control -----
        // Size the next dt from the largest per-step change in water content, normalized
        // by the soil's dynamic range (θs - θr). This is a bounded, tolerance-independent
        // error proxy that replaces the iteration-count heuristic (which was fragile: it
        // coupled dt control to the absolute Picard tolerance and collapsed dt to dt_min
        // on dry, sharp-front problems).
        float rel_change = 0.0f;
        for (uint k = 0; k < n_layers_; k++) {
            float range = props_[k].theta_s - props_[k].theta_r;
            if (range > 1e-12f) {
                rel_change = std::max(rel_change, std::fabs(theta_[k] - theta_saved[k]) / range);
            }
        }

        // Error controller: drive rel_change toward dt_target_rel_change. The 1/2 exponent
        // gives a smooth, stable response for this first-order (backward Euler) scheme.
        float factor;
        if (rel_change > 1e-9f) {
            factor = solver_params_.dt_safety * std::sqrt(solver_params_.dt_target_rel_change / rel_change);
        } else {
            // Negligible change (quasi-steady) - allow maximum growth
            factor = solver_params_.dt_growth_max;
        }

        // Iteration-count safety: don't grow dt when the solve is already near the iteration
        // ceiling, to avoid pushing the next step into non-convergence.
        if (iterations >= (3 * solver_params_.max_picard_iterations) / 4) {
            factor = std::min(factor, 1.0f);
        }

        // Do not grow straight back into a step size that just failed.
        if (rejected)
            factor = std::min(factor, 1.0f);
        rejected = false;

        // Cap growth and reduction in both directions (no single step can crater dt).
        factor = std::max(solver_params_.dt_reduction_min, std::min(factor, solver_params_.dt_growth_max));

        // Clamp dt to bounds
        dt = std::max(solver_params_.dt_min, std::min(dt * factor, solver_params_.dt_max));
    }

    // Compute average iterations
    if (adaptive_stats_.total_steps > 0) {
        adaptive_stats_.avg_iterations = static_cast<float>(adaptive_stats_.total_iterations) / static_cast<float>(adaptive_stats_.total_steps);
    }

    if (steps_taken)
        *steps_taken = adaptive_stats_.total_steps;
    return time_advanced;
}
