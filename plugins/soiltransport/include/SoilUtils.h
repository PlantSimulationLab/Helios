/** \file "SoilUtils.h" Utility functions and data structures for soil water transport.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SOILUTILS_H
#define SOILUTILS_H

#include "global.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

//! Soil constitutive model
enum class SoilModel {
    VAN_GENUCHTEN, //!< Mualem-van Genuchten model (default)
    GARDNER, //!< Gardner exponential model: K(h) = K_s * exp(alpha*h) (linearizes Richards)
    //! Mualem-van Genuchten with an air-entry head h_e < 0 (Ippisch, Vogel & Bastian 2006): saturated for
    //! h >= h_e, and for h < h_e  S_e = S_e,vG(h) / S_c,  K = K_s S_e^(1/2) [(1 - (1 - (S_e S_c)^(1/m))^m) /
    //! (1 - (1 - S_c^(1/m))^m)]^2  with  S_c = S_e,vG(h_e). For n < 2 the standard model's dK/dh is unbounded
    //! as h -> 0, which can stall the nonlinear solver at saturation; here it stays finite.
    MODIFIED_VAN_GENUCHTEN,
    //! Linear model: S_e = 1 + alpha h (0 below h = -1/alpha) and K = K_s S_e, so C = alpha (theta_s - theta_r) is
    //! constant. Nearly incompressible soils and verification against linear-diffusion solutions.
    LINEAR
};

//! Soil hydraulic properties (van Genuchten-Mualem or Gardner exponential model)
struct SoilProperties {
    //! Residual volumetric water content [m³/m³]
    float theta_r;
    //! Saturated volumetric water content [m³/m³]
    float theta_s;
    //! Pore-size parameter [1/m] (van Genuchten alpha, or Gardner exponential rate)
    float alpha;
    //! van Genuchten n parameter [-] (unused for the Gardner model)
    float n;
    //! Saturated hydraulic conductivity [m/s]
    float K_s;
    //! Constitutive model selector
    SoilModel model = SoilModel::VAN_GENUCHTEN;
    //! Air-entry pressure head [m] (< 0) of SoilModel::MODIFIED_VAN_GENUCHTEN (unused by the other models)
    float h_e = 0.0f;
    //! Effective saturation above which the van Genuchten relative conductivity is replaced by a cubic spline that
    //! rises to 1 at saturation with zero slope (1 = off). Keeps dK/dh finite at saturation when n < 2; e.g. 0.9999.
    //! SoilModel::VAN_GENUCHTEN only.
    float kr_spline_start = 1.0f;

    //! Calculate van Genuchten m parameter from n
    float m() const {
        return 1.0f - 1.0f / n;
    }

    //! Default constructor with sand-like properties (van Genuchten)
    SoilProperties() : theta_r(0.045f), theta_s(0.43f), alpha(14.5f), n(2.68f), K_s(8.25e-5f) {
    }

    //! Constructor with all van Genuchten parameters
    SoilProperties(float _theta_r, float _theta_s, float _alpha, float _n, float _K_s) : theta_r(_theta_r), theta_s(_theta_s), alpha(_alpha), n(_n), K_s(_K_s) {
    }

    //! Named constructor for a van Genuchten soil
    static SoilProperties vanGenuchten(float theta_r, float theta_s, float alpha, float n, float K_s) {
        return SoilProperties(theta_r, theta_s, alpha, n, K_s);
    }

    //! Named constructor for a modified (air-entry) van Genuchten soil; h_e [m] is the air-entry head (< 0)
    static SoilProperties modifiedVanGenuchten(float theta_r, float theta_s, float alpha, float n, float K_s, float h_e) {
        SoilProperties p(theta_r, theta_s, alpha, n, K_s);
        p.model = SoilModel::MODIFIED_VAN_GENUCHTEN;
        p.h_e = h_e;
        return p;
    }

    //! Named constructor for a linear soil: S_e = 1 + alpha h, K = K_s S_e (n is unused)
    static SoilProperties linear(float theta_r, float theta_s, float alpha, float K_s) {
        SoilProperties p(theta_r, theta_s, alpha, 2.0f, K_s);
        p.model = SoilModel::LINEAR;
        return p;
    }

    //! Named constructor for a Gardner exponential soil (n is unused)
    static SoilProperties gardner(float theta_r, float theta_s, float alpha, float K_s) {
        SoilProperties p(theta_r, theta_s, alpha, 2.0f, K_s);
        p.model = SoilModel::GARDNER;
        return p;
    }
};

//! State variables for a single soil cell
struct SoilCellState {
    //! Pressure head [m] (negative in unsaturated soil)
    float h;
    //! Volumetric water content [m³/m³]
    float theta;
    //! Hydraulic conductivity [m/s]
    float K;
    //! Specific moisture capacity [1/m]
    float C;
    //! Root water uptake rate [1/s]
    float S_root;

    SoilCellState() : h(-1.0f), theta(0.3f), K(0.0f), C(0.0f), S_root(0.0f) {
    }
};

//! Interblock hydraulic conductivity weighting scheme (face conductivity between cells)
enum class KWeighting {
    DEFAULT = -1, //!< the recommended default for every solver: arithmetic
    GEOMETRIC = 0, //!< geometric mean, sqrt(K1*K2)
    ARITHMETIC = 1, //!< arithmetic mean, (K1+K2)/2 -- the default: most accurate for infiltration and drip irrigation
    UPSTREAM = 2, //!< donor cell by total head — best for infiltration into dry soil
    HARMONIC = 3, //!< harmonic mean, 2*K1*K2/(K1+K2) (standard but slow on sharp dry fronts)
    UPSTREAM_KR = 4 //!< harmonic mean of saturated K times the donor cell's relative permeability (k_r = K/K_s)
};

//! Nonlinear iteration strategy for the 3D solver's per-step solve.
enum class NonlinearMethod {
    PICARD = 0, //!< Modified-Picard only: robust, linear convergence, cheap SPD CG
    NEWTON = 1, //!< Newton-Krylov (JFNK): quadratic near the root; ~35% faster on stiff dry fronts
    PICARD_NEWTON = 2, //!< Picard warmup then Newton finish: robust globalization + quadratic endgame
    NGS_FAS = 3, //!< Nonlinear Gauss-Seidel + FAS multigrid (lagged K): robust at sharp dry fronts, no Krylov
    M_SCHEME = 4, //!< Faithful adaptive M-scheme (Javed/Mitra/Pop 2025): Kirchhoff transform + certified estimator. HOMOGENEOUS soils ONLY; CPU only
    NEWTON_ANALYTIC = 5 //!< (default) Newton-Raphson with an assembled analytic Jacobian + line search (column and 3D solvers; CPU and GPU)
};

//! Nonlinear-solver constants shared by the 1D column and the 3D solver (CPU and GPU), so both apply the same rules.
//! Converged residuals: |F| <= max(theta_tol / dt, kNewtonRelTol x scale), where a cell's scale is the sum of the
//! magnitudes of its terms.
constexpr double kNewtonRelTol = 1e-9;
//! A face flux is computed from a head difference, so its round-off is ~eps x Kf (|H_a| + |H_b|) / L. Adding
//! kNewtonRoundoffScale x that to the scale keeps the residual test within reach of double precision; it matters
//! only for cells far thinner than the head change across them.
constexpr double kNewtonRoundoffScale = 1e-6;
//! Newton fails early after kNewtonCreepIterations consecutive damped (backtracked) iterations that each remove
//! less than 1 - kNewtonCreepRatio (2%) of the residual: how it behaves when the step has no solution. Stricter
//! settings also cut off legitimately slow steps (with 5 iterations and 10%, Vázquez-Gasty et al. Test 1 took twice
//! the steps).
constexpr int kNewtonCreepIterations = 10;
constexpr double kNewtonCreepRatio = 0.98;
//! solve() rescues a failed step by sub-stepping it, down to kRescueMinFraction of the step (not dt_min): an
//! interval that needs smaller sub-steps usually has no solution, and the caller should shorten the step.
constexpr double kRescueMinFraction = 1e-3;
//! A Picard step run as the fallback of a failed Newton step must also close the domain water balance: net
//! imbalance x dt at most kFallbackBalanceTol x domain volume (a mean water-content error of 1e-6).
constexpr double kFallbackBalanceTol = 1e-6;
//! Floor on |alpha h| in the van Genuchten-Mualem dK/dh (conductivityDerivative(), the GPU d_dKdh()). For n < 2 the
//! slope ~ |alpha h|^(n-2) is unbounded as h -> 0-; the floor only keeps it finite. Inside the floored band the
//! Newton Jacobian no longer matches the residual (by (x / floor)^(n-2)), and Newton overshoots there and
//! converges linearly, so the band must stay far below any head a cell can settle at.
constexpr double kDKdhFloor = 1e-12;

//! Solver configuration parameters
struct SolverParameters {
    //! Maximum Picard iterations per timestep
    uint max_picard_iterations = 20;
    //! Include gravity in the flux (false: horizontal flow or capillarity-only problems). Column solvers only; the 3D
    //! solver requires gravity.
    bool enable_gravity = true;
    //! solve(dt): when a step fails, sub-step it (down to kRescueMinFraction of dt) before reporting the failure.
    //! Keeps fixed-step callers going past hard steps; set false when the caller shortens failed steps itself, so a
    //! step that has no solution (e.g. a flux the soil can no longer deliver) fails at once. Same as
    //! SoilSolver3D::Params::substep_on_failure.
    bool substep_on_failure = true;
    //! Convergence tolerance for Picard iteration
    float picard_tolerance = 1e-5f;
    //! Minimum timestep [s]
    float dt_min = 0.01f;
    //! Maximum timestep [s]
    float dt_max = 3600.0f;
    //! Horizontal coupling interval for quasi-3D: couple every N vertical steps.
    //! 1 = couple every step (most accurate sequential splitting). Larger values
    //! amortize the horizontal solve for speed but delay lateral flow.
    uint horizontal_coupling_interval = 1;
    //! Regularization value for specific capacity at saturation
    float C_regularization = 1e-10f;
    //! Additive hydraulic-conductivity floor [m/s] for the 3D solver (0 = off). A small baseline
    //! (e.g. 1e-9) keeps near-residual (K->0) soil weakly coupled, easing sharp wetting fronts into
    //! dry soil (drip irrigation after a drydown). Physically ~ residual film / vapor conductivity.
    float K_min = 0.0f;
    //! Underrelaxation factor for Picard iteration (0 < omega <= 1)
    //! Use 0.5-0.7 for difficult problems (sharp wetting fronts, dry soil)
    float picard_omega = 1.0f;
    //! Interblock conductivity weighting. DEFAULT is arithmetic for every solver: in grid-convergence
    //! tests it was the most accurate for infiltration and drip irrigation and never failed to converge.
    //! KWeighting::UPSTREAM_KR is the better choice for sharply heterogeneous (e.g. perched-water) media.
    KWeighting K_weighting = KWeighting::DEFAULT;
    //! Anderson-acceleration depth for the 3D solver's Modified-Picard iteration (0 = off).
    //! Mixes the last N iterates to shorten Picard's slow linear tail on stiff (dry-front)
    //! steps while keeping the cheap CG inner solve. 2-3 is typical; 0 = plain Picard.
    int anderson_depth = 0;
    //! Nonlinear strategy for the 3D solver and the 1D column. NEWTON_ANALYTIC (default) is Newton
    //! with exact derivatives, a capped-saturation line search and a Picard fallback on stalled steps;
    //! PICARD is cheapest per iteration but converges slowly on dry fronts. See NonlinearMethod.
    NonlinearMethod nonlinear_method = NonlinearMethod::NEWTON_ANALYTIC;
    //! CPU threads for the 3D solver (0 = automatic). Automatic uses OMP_NUM_THREADS when it is set,
    //! otherwise one thread per physical core: the solver is memory-bound, so hyperthreads add no speed,
    //! and leaving them free keeps the solver from stalling when other processes are busy.
    int num_threads = 0;
    //! 3D solver: second-order BDF2 time integration instead of backward Euler (default false). BDF2 is more accurate
    //! for a given step, but it does not conserve water when sources or boundary fluxes change, e.g. an emitter switched
    //! on or off (about 1% of the delivered water), and with variable steps (adaptive stepping, or sub-stepping of a
    //! failed step) the loss depends on the step ratios. Use it only with fixed steps and smooth forcing. Same as
    //! SoilSolver3D::Params::bdf2.
    bool bdf2 = false;

    // ---- Adaptive timestepping parameters ----
    //! Enable adaptive timestepping
    bool adaptive_dt = false;
    //! Target per-step relative change in water content, |Δθ|/(θs-θr), used to size dt
    //! This is a bounded, tolerance-independent error proxy. Smaller = more accurate/smaller dt.
    float dt_target_rel_change = 0.03f;
    //! Safety factor applied to the error-controller dt estimate (0 < safety <= 1)
    float dt_safety = 0.9f;
    //! Maximum factor by which dt may grow in a single step
    float dt_growth_max = 2.0f;
    //! Minimum factor for a single-step dt reduction (caps how far one slow step can crater dt)
    float dt_reduction_min = 0.25f;
    //! Factor to decrease dt after a failed (non-converged) step
    float dt_failure_factor = 0.5f;

    // ---- Deprecated adaptive parameters (retained for API compatibility; no longer used by
    //      the relative-error step controller in SoilColumn::solveAdaptive) ----
    //! \deprecated Target Picard iteration count (superseded by dt_target_rel_change)
    uint target_iterations = 5;
    //! \deprecated Fixed dt growth factor (superseded by error controller)
    float dt_growth_factor = 1.25f;
    //! \deprecated Fixed dt shrink factor (superseded by error controller)
    float dt_shrink_factor = 0.8f;
    //! \deprecated Maximum consecutive dt reductions (superseded by dt_reduction_min cap)
    uint max_dt_reductions = 10;
    //! \deprecated Minimum iterations before allowing dt increase
    uint min_iterations_for_increase = 2;
};

//! Boundary condition types
enum class BCType {
    DIRICHLET, //!< Specified pressure head [m]
    NEUMANN, //!< Specified flux [m/s] (positive = into soil)
    FREE_DRAINAGE, //!< Unit gradient at bottom (gravity drainage)
    //! Potential flux limited by a surface head ("atmospheric" boundary, as in HYDRUS): the potential flux is applied
    //! unless the flux with the surface held at the limiting head is smaller. SoilColumn top only; set it with
    //! SoilColumn::setTopBoundaryAtmospheric().
    ATMOSPHERIC
};

//! Solver types
enum class SolverType {
    SOLVER_1D, //!< Independent 1D columns (fastest)
    SOLVER_QUASI_3D, //!< 1D columns with periodic horizontal coupling
    SOLVER_3D //!< Full 3D Richards equation
};

//! Operator splitting types for quasi-3D
enum class SplittingType {
    SEQUENTIAL, //!< First-order: V(dt) then H(dt)
    STRANG //!< Second-order: V(dt/2) then H(dt) then V(dt/2)
};

//! Soil-domain visualization coloring mode (see SoilTransportModel::addSoilVisualization)
enum class SoilColorMode {
    WATER_CONTENT, //!< color the soil block by volumetric water content theta [m³/m³]
    PRESSURE_HEAD, //!< color the soil block by pressure head h [m]
    TEXTURE //!< plain dirt texture (no field coloring)
};

//! Quantity recorded by a soil point sensor (see SoilTransportModel::addSensor)
enum class SoilMeasurement {
    WATER_CONTENT, //!< volumetric water content theta [m³/m³]
    PRESSURE_HEAD //!< pressure head h [m]
};

//! How a fractional sub-region is used in SoilTransportModel::addSoilVisualization
enum class SoilRegion {
    KEEP, //!< render the sub-box (the rest of the domain is hidden) — slices/zoom
    REMOVE //!< render the domain with the (corner-anchored) sub-box cut out — cutaway
};


// ============================================================================
// Van Genuchten-Mualem constitutive functions
// ============================================================================

//! Pressure head [m] at and above which the soil is saturated: 0, or the air-entry head h_e of
//! SoilModel::MODIFIED_VAN_GENUCHTEN
inline double saturationHead(const SoilProperties &p) {
    return (p.model == SoilModel::MODIFIED_VAN_GENUCHTEN && p.h_e < 0.0f) ? (double) p.h_e : 0.0;
}

//! Relative conductivity near saturation with SoilProperties::kr_spline_start (S1): for S_e > S1 a cubic Hermite spline
//! from (S1, k_r(S1), dk_r/dS_e(S1)) to (1, 1, 0). Returns false (kr, dkr_dSe untouched) where it does not apply.
template<typename T>
inline bool krSaturationSpline(const SoilProperties &p, T Se, T &kr, T &dkr_dSe) {
    const T S1 = T(p.kr_spline_start);
    if (p.model != SoilModel::VAN_GENUCHTEN || !(S1 < T(1)) || !(Se > S1))
        return false;
    const T m = T(1) - T(1) / T(p.n);
    const T w = T(1) - std::pow(S1, T(1) / m), B = T(1) - std::pow(w, m), sq = std::sqrt(S1);
    const T y0 = sq * B * B; // standard k_r and its slope at S1
    const T m0 = T(0.5) / sq * B * B + T(2) * sq * B * std::pow(w, m - T(1)) * std::pow(S1, T(1) / m - T(1));
    const T D = T(1) - S1, t = (std::min(Se, T(1)) - S1) / D, t2 = t * t, t3 = t2 * t;
    kr = (T(2) * t3 - T(3) * t2 + T(1)) * y0 + (t3 - T(2) * t2 + t) * D * m0 + (T(3) * t2 - T(2) * t3);
    dkr_dSe = ((T(6) * t2 - T(6) * t) * y0 + (T(3) * t2 - T(4) * t + T(1)) * D * m0 + (T(6) * t - T(6) * t2)) / D;
    return true;
}

//! Air-entry scale factors of SoilModel::MODIFIED_VAN_GENUCHTEN: S_c = S_e,vG(h_e) and
//! F_K = 1 / (sqrt(S_c) (1 - (1 - S_c^(1/m))^m)^2). Below h_e the modified model is the standard one rescaled:
//! S_e = S_e,vG / S_c, C = C_vG / S_c and K = F_K K_vG (so that K = K_s at h_e). Both are 1 for the other models.
template<typename T>
inline void airEntryScales(const SoilProperties &p, T &Sc, T &Fk) {
    Sc = Fk = T(1);
    if (p.model != SoilModel::MODIFIED_VAN_GENUCHTEN || !(p.h_e < 0.0f))
        return;
    const T n = T(p.n), m = T(1) - T(1) / n;
    const T xn = std::pow(-T(p.alpha) * T(p.h_e), n), base = T(1) + xn;
    Sc = std::pow(base, -m);
    const T Bc = T(1) - std::pow(xn / base, m); // 1 - (1 - S_c^(1/m))^m
    Fk = T(1) / (std::sqrt(Sc) * Bc * Bc);
}

//! Compute effective saturation from pressure head
/**
 * \param[in] h Pressure head [m] (negative in unsaturated soil)
 * \param[in] alpha van Genuchten alpha [1/m]
 * \param[in] n van Genuchten n [-]
 * \param[in] m van Genuchten m = 1 - 1/n [-]
 * \return Effective saturation S_e [0, 1]
 */
inline float computeSe(float h, float alpha, float n, float m) {
    if (h >= 0.0f)
        return 1.0f;
    float ah = fabs(alpha * h); // CRITICAL: use fabs() not abs()
    return powf(1.0f + powf(ah, n), -m);
}

//! Compute effective saturation from pressure head using SoilProperties
inline float computeSe(float h, const SoilProperties &props) {
    if (h >= (float) saturationHead(props))
        return 1.0f;
    if (props.model == SoilModel::GARDNER)
        return expf(props.alpha * h); // S_e = exp(alpha*h)
    if (props.model == SoilModel::LINEAR)
        return fmaxf(1.0f + props.alpha * h, 0.0f);
    float Sc, Fk;
    airEntryScales(props, Sc, Fk);
    return computeSe(h, props.alpha, props.n, props.m()) / Sc;
}

//! Compute volumetric water content from pressure head
/**
 * \param[in] h Pressure head [m]
 * \param[in] props Soil hydraulic properties
 * \return Volumetric water content theta [m³/m³]
 */
inline float h_to_theta(float h, const SoilProperties &props) {
    float Se = computeSe(h, props);
    return props.theta_r + (props.theta_s - props.theta_r) * Se;
}

//! Compute pressure head from volumetric water content
/**
 * \param[in] theta Volumetric water content [m³/m³]
 * \param[in] props Soil hydraulic properties
 * \return Pressure head h [m]
 */
inline float theta_to_h(float theta, const SoilProperties &props) {
    if (theta >= props.theta_s)
        return 0.0f;
    if (theta <= props.theta_r)
        return -1e10f; // Very dry

    float Se = (theta - props.theta_r) / (props.theta_s - props.theta_r);
    if (props.model == SoilModel::GARDNER)
        return logf(Se) / props.alpha; // S_e = exp(alpha*h)
    if (props.model == SoilModel::LINEAR)
        return (Se - 1.0f) / props.alpha;
    float Sc, Fk;
    airEntryScales(props, Sc, Fk);
    Se *= Sc; // standard van Genuchten S_e
    float m = props.m();
    float im = 1.0f / m;
    float in = 1.0f / props.n;
    return -(1.0f / props.alpha) * powf(powf(Se, -im) - 1.0f, in);
}

//! Compute unsaturated hydraulic conductivity from pressure head
/**
 * Van Genuchten-Mualem model: K(Se) = K_s * Se^0.5 * [1 - (1 - Se^(1/m))^m]^2
 * \param[in] h Pressure head [m]
 * \param[in] props Soil hydraulic properties
 * \return Hydraulic conductivity K [m/s]
 */
inline float computeK(float h, const SoilProperties &props) {
    if (h >= (float) saturationHead(props))
        return props.K_s;

    if (props.model == SoilModel::GARDNER)
        return props.K_s * expf(props.alpha * h); // K = K_s exp(alpha*h)
    if (props.model == SoilModel::LINEAR)
        return props.K_s * fmaxf(1.0f + props.alpha * h, 0.0f);

    float Sc, Fk;
    airEntryScales(props, Sc, Fk);
    float Se = computeSe(h, props.alpha, props.n, props.m()); // standard van Genuchten S_e
    float kr, dkr;
    if (krSaturationSpline(props, Se, kr, dkr))
        return props.K_s * kr;
    float m = props.m();
    float im = 1.0f / m;
    float term = 1.0f - powf(1.0f - powf(Se, im), m);
    return props.K_s * Fk * sqrtf(Se) * term * term;
}

//! Compute specific moisture capacity (dθ/dh) with regularization
/**
 * C(h) = α*m*n*(θ_s - θ_r) * |αh|^(n-1) * [1 + |αh|^n]^(-(m+1))
 * \param[in] h Pressure head [m]
 * \param[in] props Soil hydraulic properties
 * \param[in] C_min Minimum capacity for regularization (default 1e-10)
 * \return Specific moisture capacity C [1/m]
 */
inline float computeC(float h, const SoilProperties &props, float C_min = 1e-10f) {
    if (h >= (float) saturationHead(props))
        return C_min; // Regularization at saturation

    if (props.model == SoilModel::GARDNER) // C = alpha*(theta_s-theta_r)*exp(alpha*h)
        return fmaxf(props.alpha * (props.theta_s - props.theta_r) * expf(props.alpha * h), C_min);
    if (props.model == SoilModel::LINEAR)
        return fmaxf((1.0f + props.alpha * h > 0.0f) ? props.alpha * (props.theta_s - props.theta_r) : 0.0f, C_min);

    float alpha = props.alpha;
    float n = props.n;
    float m = props.m();
    float ah = fabs(alpha * h); // CRITICAL: use fabs() not abs()
    float ahn = powf(ah, n);
    float ahn1 = powf(ah, n - 1.0f);
    float denom = powf(1.0f + ahn, m + 1.0f);

    float Sc, Fk;
    airEntryScales(props, Sc, Fk);
    float C = alpha * m * n * (props.theta_s - props.theta_r) * ahn1 / denom / Sc;

    return fmaxf(C, C_min); // Regularization to prevent division by zero
}

//! Fused hydraulic conductivity + specific moisture capacity (templated scalar type)
/**
 * Computes K(h) and C(h) together from shared intermediates, minimizing transcendental
 * calls (van Genuchten: 3 pow + 1 sqrt for both, vs ~7 pow when computed separately;
 * uses the identities Se^{1/m}=1/(1+|ah|^n), |ah|^{n-1}=|ah|^n/|ah|, (1+|ah|^n)^{m+1}=(1+|ah|^n)/Se).
 * Templated so callers can request double precision (e.g. the 3D solver) or float.
 * \param[in] h Pressure head [m]
 * \param[in] props Soil hydraulic properties
 * \param[out] K Hydraulic conductivity [m/s]
 * \param[out] C Specific moisture capacity [1/m]
 * \param[in] C_min Regularization floor for C
 * \param[in] K_min Additive conductivity floor [m/s]. Adds a small baseline K so the dry zone
 *   (where the van Genuchten K -> 0) stays weakly coupled -- improves conditioning of sharp
 *   wetting fronts into near-residual soil. Additive (not max), so dK/dh is unchanged. 0 = off.
 */
template<typename T>
inline void computeKandC(T h, const SoilProperties &props, T &K, T &C, T C_min = T(1e-12), T K_min = T(0)) {
    const T Ks = T(props.K_s), al = T(props.alpha), tr = T(props.theta_r), ts = T(props.theta_s);
    if (h >= T(saturationHead(props))) {
        K = Ks + K_min;
        C = C_min;
        return;
    } // saturated
    if (props.model == SoilModel::GARDNER) {
        T e = std::exp(al * h);
        K = Ks * e + K_min;
        T c = al * (ts - tr) * e;
        C = (c < C_min) ? C_min : c;
        return;
    }
    if (props.model == SoilModel::LINEAR) {
        const T Se = T(1) + al * h;
        K = Ks * (Se > T(0) ? Se : T(0)) + K_min;
        const T c = Se > T(0) ? al * (ts - tr) : T(0);
        C = (c < C_min) ? C_min : c;
        return;
    }
    const T n = T(props.n), m = T(1) - T(1) / n;
    T ah = std::fabs(al * h);
    T ahn = std::pow(ah, n);
    T base = T(1) + ahn;
    T Se = std::pow(base, -m);
    T bracket = T(1) - std::pow(T(1) - T(1) / base, m); // 1 - (1 - Se^{1/m})^m
    T Sc, Fk;
    airEntryScales(props, Sc, Fk);
    T kr, dkr;
    K = (krSaturationSpline(props, Se, kr, dkr) ? Ks * kr : Ks * Fk * std::sqrt(Se) * bracket * bracket) + K_min;
    T c = (ts - tr) * m * n * al * (ahn / ah) * Se / base / Sc; // |ah|^{n-1}=ahn/ah, base^{m+1}=base/Se
    C = (c < C_min) ? C_min : c;
}

//! Supply-limited surface evaporation reduction factor beta in (0, 1]
/**
 * Gates potential (atmospheric-demand) evaporation by what a drying soil surface can
 * actually supply: actual evaporation E_act = beta * E_pot. Uses a dry-surface-layer
 * (DSL) vapor-diffusion resistance after Swenson & Lawrence (2014) (the CLM4.5 scheme):
 * as the surface dries a thin air-filled layer forms and vapor must diffuse across it,
 * so the soil resistance r_soil rises and beta = r_aero/(r_aero + r_soil) falls toward 0.
 *
 *   L_dsl = D_max * (exp((1 - theta/theta_s)^5) - 1) / (e - 1)      [m]
 *   D_eff = D_v * air^(7/3) / theta_s^2   (Millington-Quirk tortuosity, air = theta_s - theta)
 *   r_soil = L_dsl / D_eff,   beta = r_aero / (r_aero + r_soil)
 *
 * \param[in] theta Surface volumetric water content [m^3/m^3]
 * \param[in] props Soil hydraulic properties (uses theta_s, theta_r)
 * \param[in] r_aero Aerodynamic resistance [s/m] (default 50)
 * \param[in] D_max Maximum dry-surface-layer thickness [m] (default 0.015)
 * \return Reduction factor beta in (0, 1]; ~1 for a wet surface, -> 0 as theta -> theta_r
 */
inline float surfaceEvaporationFactor(float theta, const SoilProperties &props, float r_aero = 50.0f, float D_max = 0.015f) {
    const float ts = props.theta_s;
    const float th = fminf(fmaxf(theta, props.theta_r), ts);
    if (th >= ts)
        return 1.0f; // saturated surface: no dry layer, no supply limit
    const float L = D_max * (expf(powf(1.0f - th / ts, 5.0f)) - 1.0f) / 1.71828183f; // (e - 1)
    const float Dv = 2.12e-5f; // vapor diffusivity in free air [m^2/s]
    const float air = fmaxf(ts - th, 1e-4f); // air-filled porosity
    const float D_eff = Dv * powf(air, 7.0f / 3.0f) / (ts * ts);
    const float r_soil = (D_eff > 0.0f) ? L / D_eff : 0.0f;
    return r_aero / (r_aero + r_soil);
}

//! Volumetric water content from pressure head (templated scalar type)
template<typename T>
inline T waterContent(T h, const SoilProperties &props) {
    const T tr = T(props.theta_r), ts = T(props.theta_s), al = T(props.alpha);
    if (h >= T(saturationHead(props)))
        return ts;
    T Se;
    if (props.model == SoilModel::GARDNER) {
        Se = std::exp(al * h);
    } else if (props.model == SoilModel::LINEAR) {
        Se = std::max(T(1) + al * h, T(0));
    } else {
        T n = T(props.n), m = T(1) - T(1) / n, Sc, Fk;
        airEntryScales(props, Sc, Fk);
        Se = std::pow(T(1) + std::pow(std::fabs(al * h), n), -m) / Sc;
    }
    return tr + (ts - tr) * Se;
}

//! Kirchhoff transform + b/B splitting for a HOMOGENEOUS soil (the M-scheme of Javed, Mitra & Pop 2025,
//! arXiv:2508.07420). Tabulates the matric flux potential w = Phi(psi), the integral of K from -inf to psi, and a
//! splitting variable s such that  u = b(s) = theta  and  w_tilde = B(s) = w / w_sat, with b' = dtheta/ds,
//! B' = dw_tilde/ds, b'+B' = 1 and both in [0,1]. The diffusion Delta w = w_sat * Delta w_tilde becomes a
//! CONSTANT-COEFFICIENT Laplacian (coefficient D0 = w_sat), which is the M-scheme's solver advantage.
//!
//! IMPORTANT: Phi is material-dependent, so a single table is valid only for a SINGLE soil. The M-scheme
//! is therefore restricted to HOMOGENEOUS soil properties (w is discontinuous across material interfaces).
struct KirchhoffTable {
    SoilProperties props;
    std::vector<double> s_, theta_, wt_, psi_, bp_, Bp_; // node arrays, strictly increasing in s_
    double w_sat = 0.0; //!< Phi(0) = matric flux potential at saturation [m^2/s]  (== D0, the Laplacian coeff)
    double s_sat = 0.0; //!< s at saturation (psi = 0); s > s_sat is the ponded/saturated branch
    double theta_s_ = 0.0, theta_r_ = 0.0, Ks_ = 0.0;
    bool built = false;

    //! Tabulate the transform for one homogeneous soil. N nodes uniform in effective saturation Se in
    //! [Se_min, 1] (clusters resolution at the wet end where the Kirchhoff integral accumulates).
    void build(const SoilProperties &p, int N = 2000, double Se_min = 1e-6) {
        props = p;
        theta_s_ = p.theta_s;
        theta_r_ = p.theta_r;
        Ks_ = p.K_s;
        const double tr = p.theta_r, ts = p.theta_s, al = p.alpha;
        s_.assign(N, 0.0);
        theta_.assign(N, 0.0);
        wt_.assign(N, 0.0);
        psi_.assign(N, 0.0);
        bp_.assign(N, 0.0);
        Bp_.assign(N, 0.0);
        auto psiOfSe = [&](double Se) -> double {
            if (Se >= 1.0)
                return 0.0;
            if (p.model == SoilModel::GARDNER)
                return std::log(Se) / al; // Se = exp(al*psi)
            if (p.model == SoilModel::LINEAR)
                return (Se - 1.0) / al;
            const double m = 1.0 - 1.0 / p.n;
            return -(1.0 / al) * std::pow(std::pow(Se, -1.0 / m) - 1.0, 1.0 / p.n); // van Genuchten inverse
        };
        // First pass: psi, theta, w (unscaled Kirchhoff potential via trapezoid), and the slopes C, K.
        std::vector<double> w(N, 0.0), Cn(N, 0.0), Kn(N, 0.0);
        double psi_prev = psiOfSe(Se_min), w_acc = 0.0;
        for (int k = 0; k < N; k++) {
            const double Se = Se_min + (1.0 - Se_min) * (double) k / (N - 1);
            const double psi = psiOfSe(Se);
            double K, C;
            computeKandC<double>(psi, p, K, C); // K(psi) [m/s], C = dtheta/dpsi [1/m]
            if (k > 0) {
                double Kp, Cp;
                computeKandC<double>(psi_prev, p, Kp, Cp);
                w_acc += 0.5 * (K + Kp) * (psi - psi_prev); // \int K dpsi (K>=0, psi increasing -> w increasing)
            }
            theta_[k] = tr + (ts - tr) * Se;
            psi_[k] = psi;
            w[k] = w_acc;
            Cn[k] = C;
            Kn[k] = K;
            psi_prev = psi;
        }
        w_sat = (w[N - 1] > 0.0) ? w[N - 1] : 1e-30;
        // Second pass: scale w_tilde = w/w_sat, slopes b' = (dtheta)/(dtheta+dw_tilde), and cumulative s.
        for (int k = 0; k < N; k++) {
            wt_[k] = w[k] / w_sat;
            const double dth = Cn[k]; // dtheta/dpsi = C  (actual, consistent with s = cumsum(dtheta+dw_tilde))
            const double dwt = Kn[k] / w_sat; // dw_tilde/dpsi = K / w_sat
            const double denom = dth + dwt;
            bp_[k] = (denom > 0.0) ? dth / denom : 0.0; // b' = dtheta/ds
            Bp_[k] = (denom > 0.0) ? dwt / denom : 1.0; // B' = dw_tilde/ds
        }
        s_[0] = 0.0;
        for (int k = 1; k < N; k++)
            s_[k] = s_[k - 1] + (theta_[k] - theta_[k - 1]) + (wt_[k] - wt_[k - 1]);
        s_sat = s_[N - 1];
        built = true;
    }

    //! Largest node index with s_[idx] <= s (for interpolation); clamped to [0, N-2].
    int locate(double s) const {
        int lo = 0, hi = (int) s_.size() - 1;
        if (s <= s_[0])
            return 0;
        if (s >= s_[hi])
            return hi - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            (s_[mid] <= s ? lo : hi) = mid;
        }
        return lo;
    }
    static double lerp(double a, double b, double t) {
        return a + (b - a) * t;
    }

    double b(double s) const { // theta(s)
        if (s >= s_sat)
            return theta_s_;
        int k = locate(s);
        double t = (s - s_[k]) / (s_[k + 1] - s_[k]);
        return lerp(theta_[k], theta_[k + 1], t);
    }
    double B(double s) const { // w_tilde(s); ponded branch grows linearly (b'=0, B'=1)
        if (s >= s_sat)
            return 1.0 + (s - s_sat);
        int k = locate(s);
        double t = (s - s_[k]) / (s_[k + 1] - s_[k]);
        return lerp(wt_[k], wt_[k + 1], t);
    }
    double bprime(double s) const {
        if (s >= s_sat)
            return 0.0;
        int k = locate(s);
        double t = (s - s_[k]) / (s_[k + 1] - s_[k]);
        return lerp(bp_[k], bp_[k + 1], t);
    }
    double Bprime(double s) const {
        if (s >= s_sat)
            return 1.0;
        int k = locate(s);
        double t = (s - s_[k]) / (s_[k + 1] - s_[k]);
        return lerp(Bp_[k], Bp_[k + 1], t);
    }
    //! Inverse maps for initialization / recovery.
    double sOfTheta(double theta) const {
        if (theta >= theta_s_)
            return s_sat;
        if (theta <= theta_[0])
            return s_[0];
        int lo = 0, hi = (int) theta_.size() - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            (theta_[mid] <= theta ? lo : hi) = mid;
        }
        double t = (theta - theta_[lo]) / (theta_[lo + 1] - theta_[lo]);
        return lerp(s_[lo], s_[lo + 1], t);
    }
    double sOfWtilde(double wt) const {
        if (wt >= 1.0)
            return s_sat + (wt - 1.0);
        if (wt <= wt_[0])
            return s_[0];
        int lo = 0, hi = (int) wt_.size() - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            (wt_[mid] <= wt ? lo : hi) = mid;
        }
        double t = (wt - wt_[lo]) / (wt_[lo + 1] - wt_[lo]);
        return lerp(s_[lo], s_[lo + 1], t);
    }
    //! Pressure head psi(s) (for the gravity term and BC handling).
    double psiOf(double s) const {
        if (s >= s_sat)
            return (B(s) - 1.0) * w_sat / Ks_; // ponded: w = w_sat + Ks*psi
        int k = locate(s);
        double t = (s - s_[k]) / (s_[k + 1] - s_[k]);
        return lerp(psi_[k], psi_[k + 1], t);
    }
    //! s for a given pressure head (initialization from h, and Dirichlet-BC handling).
    double sOfPsi(double psi) const {
        if (psi >= 0.0)
            return s_sat + (Ks_ / w_sat) * psi; // ponded branch: s - s_sat = w_tilde - 1 = (Ks/w_sat) psi
        if (psi <= psi_[0])
            return s_[0];
        int lo = 0, hi = (int) psi_.size() - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            (psi_[mid] <= psi ? lo : hi) = mid;
        }
        double t = (psi - psi_[lo]) / (psi_[lo + 1] - psi_[lo]);
        return lerp(s_[lo], s_[lo + 1], t);
    }
    //! Scaled Kirchhoff potential w_tilde at a given pressure head (for Dirichlet w_tilde boundary values).
    double wtildeOfPsi(double psi) const {
        if (psi >= 0.0)
            return 1.0 + (Ks_ / w_sat) * psi;
        return B(sOfPsi(psi));
    }
};

//! Compute interblock hydraulic conductivity (geometric mean)
/**
 * \param[in] K1 Conductivity of first cell [m/s]
 * \param[in] K2 Conductivity of second cell [m/s]
 * \return Interblock conductivity [m/s]
 */
inline float interblockK(float K1, float K2) {
    return sqrtf(K1 * K2);
}

//! Compute interblock conductivity with specified weighting
/**
 * \param[in] K1 Conductivity of first (upstream) cell [m/s]
 * \param[in] K2 Conductivity of second (downstream) cell [m/s]
 * \param[in] method 0=geometric, 1=arithmetic, 2=upstream, 3=harmonic
 * \return Interblock conductivity [m/s]
 */
inline float interblockK_weighted(float K1, float K2, int method) {
    switch (method) {
        case 1: // Arithmetic mean
            return 0.5f * (K1 + K2);
        case 2: // Upstream weighting (use K1, the upstream value)
        case 4: // upstream k_r; at a boundary face both sides share the cell's Ks, so this is the upstream K too
            return K1;
        case 3: // Harmonic mean
            return (K1 + K2 > 0.0f) ? (2.0f * K1 * K2 / (K1 + K2)) : 0.0f;
        default: // Geometric mean
            return sqrtf(K1 * K2);
    }
}


// ============================================================================
// Derived infiltration parameters (Green-Ampt / Philip)
//
// These are soil properties derived from the van Genuchten-Mualem functions, not
// free parameters: they let analytical infiltration models (Green-Ampt, Philip) be
// parameterized consistently with the numerical solver.
// ============================================================================

//! dK/dh [1/s] for the soil's constitutive model (0 when saturated)
/**
 * For van Genuchten-Mualem with n < 2 the true slope ~ |alpha h|^(n-2) is unbounded as h -> 0-, so
 * |alpha h| is floored at kDKdhFloor for the derivative (K itself is unaffected). Used by the Newton solvers.
 */
inline double conductivityDerivative(double h, const SoilProperties &p) {
    if (h >= saturationHead(p))
        return 0.0;
    const double Ks = p.K_s, al = p.alpha;
    if (p.model == SoilModel::GARDNER)
        return Ks * al * std::exp(al * h);
    if (p.model == SoilModel::LINEAR)
        return (1.0 + al * h > 0.0) ? Ks * al : 0.0;
    const double n = p.n, m = 1.0 - 1.0 / n;
    const double x = std::max(-al * h, kDKdhFloor); // |alpha h|
    const double xn = std::pow(x, n), base = 1.0 + xn;
    const double Se = std::pow(base, -m);
    const double w = xn / base; // 1 - Se^(1/m)
    const double B = 1.0 - std::pow(w, m);
    const double dSe_dh = m * n * al * (xn / x) * Se / base;
    const double dB_dSe = std::pow(w, m - 1.0) * std::pow(base, -(1.0 - m)); // w^(m-1) Se^(1/m-1)
    const double sq = std::sqrt(Se);
    double kr, dkr;
    if (krSaturationSpline(p, std::pow(1.0 + std::pow(-al * h, n), -m), kr, dkr)) // exact S_e (x is floored above)
        return Ks * dkr * m * n * al * std::pow(-al * h, n - 1.0) * std::pow(1.0 + std::pow(-al * h, n), -m - 1.0);
    double Sc, Fk;
    airEntryScales(p, Sc, Fk);
    return Fk * Ks * (0.5 / sq * B * B + 2.0 * sq * B * dB_dSe) * dSe_dh;
}

//! Stored water per unit volume W(h) = theta + Ss * S_w * h (S_w = theta / theta_s) and dW/dh, from
//! theta and C = dtheta/dh. With Ss = 0 these are theta and C.
inline void addSpecificStorage(double h, double Ss, double theta_s, double &theta, double &C) {
    if (Ss == 0.0)
        return;
    const double Sw = theta / theta_s, dSw = C / theta_s;
    theta += Ss * Sw * h;
    C += Ss * (Sw + h * dSw);
}

inline void constitutiveAll(double h, const SoilProperties &p, double C_min, double K_min, double &theta, double &K, double &C, double &dK) {
    const double tr = p.theta_r, ts = p.theta_s, Ks = p.K_s, al = p.alpha;
    if (h >= saturationHead(p)) {
        theta = ts;
        K = Ks + K_min;
        C = C_min;
        dK = 0.0;
        return;
    }
    if (p.model == SoilModel::GARDNER) {
        const double e = std::exp(al * h);
        theta = tr + (ts - tr) * e;
        K = Ks * e + K_min;
        C = std::max(al * (ts - tr) * e, C_min);
        dK = Ks * al * e;
        return;
    }
    if (p.model == SoilModel::LINEAR) {
        const double Se = 1.0 + al * h;
        const bool wet = Se > 0.0;
        theta = tr + (ts - tr) * (wet ? Se : 0.0);
        K = Ks * (wet ? Se : 0.0) + K_min;
        C = std::max(wet ? al * (ts - tr) : 0.0, C_min);
        dK = wet ? Ks * al : 0.0;
        return;
    }
    const double n = p.n, m = 1.0 - 1.0 / n;
    const double x = -al * h, xn = std::pow(x, n), base = 1.0 + xn;
    const double Se = std::pow(base, -m); // standard van Genuchten S_e (rescaled by S_c for the air-entry model)
    const double w = xn / base, wm = std::pow(w, m), B = 1.0 - wm, sq = std::sqrt(Se);
    double Sc, Fk;
    airEntryScales(p, Sc, Fk);
    theta = tr + (ts - tr) * Se / Sc;
    K = Fk * Ks * sq * B * B + K_min;
    const double dSe_dh = m * n * al * (xn / x) * Se / base;
    C = std::max((ts - tr) * dSe_dh / Sc, C_min);
    double kr, dkr;
    if (krSaturationSpline(p, Se, kr, dkr)) {
        K = Ks * kr + K_min;
        dK = Ks * dkr * dSe_dh;
    } else if (x < 1e-4) {
        dK = conductivityDerivative(h, p); // derivative floored near saturation (see conductivityDerivative)
    } else {
        const double dB_dSe = (wm / w) / (base * Se); // w^(m-1) * base^(m-1)
        dK = Fk * Ks * (0.5 / sq * B * B + 2.0 * sq * B * dB_dSe) * dSe_dh;
    }
}

//! Pressure head [m] at water content theta (double-precision inverse retention curve); theta in (theta_r, theta_s)
inline double pressureHeadOfTheta(double theta, const SoilProperties &p) {
    const double tr = p.theta_r, ts = p.theta_s; // in double, as waterContent<double> (a float theta_s - theta_r skews Se)
    const double Se = (theta - tr) / (ts - tr);
    if (p.model == SoilModel::GARDNER)
        return std::log(Se) / p.alpha;
    if (p.model == SoilModel::LINEAR)
        return (Se - 1.0) / p.alpha;
    const double n = p.n, m = 1.0 - 1.0 / n;
    double Sc, Fk;
    airEntryScales(p, Sc, Fk);
    return -std::pow(std::pow(Se * Sc, -1.0 / m) - 1.0, 1.0 / n) / p.alpha;
}

//! Pressure head [m] at the inflection point of the retention curve (where C = dtheta/dh peaks), or 0 if C
//! peaks at saturation (Gardner). On the wet side of it C falls toward 0 at saturation, so a linearization in
//! water content underestimates drainage; on the dry side it overestimates it.
inline double inflectionHead(const SoilProperties &p) {
    if (p.model == SoilModel::GARDNER || p.model == SoilModel::LINEAR)
        return 0.0;
    const double n = p.n, m = 1.0 - 1.0 / n;
    return std::min(-std::pow(m, 1.0 / n) / p.alpha, saturationHead(p)); // van Genuchten: (alpha |h|)^n = m
}

//! Interblock conductivity between cells a and b and its partial derivatives with respect to Ka and Kb
/**
 * \param[in] weighting 0 geometric, 1 arithmetic, 2 upstream K, 3 harmonic, 4 upstream relative permeability
 *            (harmonic-mean saturated K times the donor cell's k_r = K/Ks)
 * \param[in] Ha,Hb Total heads of the two cells (upstream weightings pick the donor by total head)
 * \param[in] Ksa,Ksb Saturated conductivities of the two cells
 */
inline double interblockKDeriv(int weighting, double Ka, double Kb, double Ha, double Hb, double Ksa, double Ksb, double &dA, double &dB) {
    switch (weighting) {
        case 0: {
            const double g = std::sqrt(Ka * Kb);
            dA = (Ka > 0.0) ? 0.5 * g / Ka : 0.0;
            dB = (Kb > 0.0) ? 0.5 * g / Kb : 0.0;
            return g;
        }
        case 1:
            dA = dB = 0.5;
            return 0.5 * (Ka + Kb);
        case 2:
            if (Ha - Hb > 0.0 || (Ha == Hb && Ka >= Kb)) {
                dA = 1.0;
                dB = 0.0;
                return Ka;
            }
            dA = 0.0;
            dB = 1.0;
            return Kb;
        case 4: {
            const double Ksh = (Ksa + Ksb > 0.0) ? 2.0 * Ksa * Ksb / (Ksa + Ksb) : 0.0;
            if (Ha - Hb > 0.0 || (Ha == Hb && Ka / Ksa >= Kb / Ksb)) {
                dA = Ksh / Ksa;
                dB = 0.0;
                return Ksh * Ka / Ksa;
            }
            dA = 0.0;
            dB = Ksh / Ksb;
            return Ksh * Kb / Ksb;
        }
        default: {
            const double s = Ka + Kb;
            if (s <= 0.0) {
                dA = dB = 0.0;
                return 0.0;
            }
            dA = 2.0 * Kb * Kb / (s * s);
            dB = 2.0 * Ka * Ka / (s * s);
            return 2.0 * Ka * Kb / s;
        }
    }
}

//! Green-Ampt / Bouwer effective capillary drive at the wetting front
/**
 * psi_f = integral_{h_init}^{0} k_r(h) dh, where k_r(h) = K(h)/K_s is the relative
 * hydraulic conductivity. This is the Bouwer (1966) / Morel-Seytoux effective
 * capillary drive (a soil property computed directly from the constitutive functions).
 *
 * \param[in] props Soil hydraulic properties
 * \param[in] h_init Initial (dry) pressure head [m] (negative)
 * \param[in] n_steps Number of trapezoidal integration intervals (default 2000)
 * \return Effective capillary drive psi_f [m] (positive)
 */
inline float effectiveCapillaryDrive(const SoilProperties &props, float h_init, int n_steps = 2000) {
    if (h_init >= 0.0f || props.K_s <= 0.0f)
        return 0.0f;
    float dh = -h_init / static_cast<float>(n_steps); // integrate upward from h_init to 0
    double integral = 0.0;
    float kr_prev = computeK(h_init, props) / props.K_s;
    for (int i = 1; i <= n_steps; i++) {
        float h = h_init + dh * static_cast<float>(i);
        float kr = computeK(h, props) / props.K_s;
        integral += 0.5 * (static_cast<double>(kr_prev) + kr) * dh;
        kr_prev = kr;
    }
    return static_cast<float>(integral);
}

//! Philip sorptivity for infiltration from a saturated surface into uniform soil
/**
 * The exact (Philip 1957) sorptivity, obtained by solving the Boltzmann similarity ODE for
 * one-dimensional absorption. Solved in head space (where K and C are bounded for van
 * Genuchten) as the boundary-value problem
 *     dlambda/dh = -K(h)/F,   dF/dh = (lambda/2) C(h),
 * with lambda(0) = 0 and F(h_init) = 0, shooting on F(0); the sorptivity is S = 2 F(0).
 *
 * \param[in] props Soil hydraulic properties
 * \param[in] h_init Initial (dry) pressure head [m] (negative)
 * \param[in] n_steps Number of integration steps over [h_init, 0] (default 100000)
 * \return Sorptivity S [m s^-1/2]
 */
float sorptivity(const SoilProperties &props, float h_init, int n_steps = 100000);


// ============================================================================
// Thomas algorithm (tridiagonal solver)
// ============================================================================

//! Solve tridiagonal system using Thomas algorithm
/**
 * Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
 * Note: a[0] and c[n-1] are not used
 *
 * WARNING: This function modifies the input arrays c and d in-place!
 *
 * \param[in] a Lower diagonal (size n, a[0] unused)
 * \param[in] b Main diagonal (size n)
 * \param[in,out] c Upper diagonal (size n, c[n-1] unused, modified in-place)
 * \param[in,out] d Right-hand side (size n, modified in-place)
 * \param[out] x Solution vector (size n)
 * \param[in] n Number of unknowns
 */
void thomasSolve(float *a, float *b, float *c, float *d, float *x, uint n);

//! Number of physical CPU cores available to this process (hyperthread siblings counted once), or 0 if unknown
int physicalCoreCount();

//! OpenMP team size for a requested thread count (0 = automatic: OMP_NUM_THREADS if set, else physical cores)
int resolveThreadCount(int requested);

//! Safe version that doesn't modify input arrays
void thomasSolveSafe(const float *a, const float *b, const float *c, const float *d, float *x, uint n);


// ============================================================================
// Soil property library
// ============================================================================

//! Get soil properties from texture name
/**
 * Returns van Genuchten parameters from Carsel & Parrish (1988). Raises an error for a texture that is not in the
 * library (see isSoilTextureAvailable()).
 * \param[in] texture Soil texture name (e.g., "sand", "loam", "clay")
 * \return SoilProperties for the given texture
 */
SoilProperties getSoilPropertiesFromTexture(const std::string &texture);

//! Default air-entry head [m] of the texture library for SoilModel::MODIFIED_VAN_GENUCHTEN: -2 cm, the value of
//! HYDRUS's "van Genuchten-Mualem with an air-entry value of -2 cm" option (Vogel & Cislerova 1988; Simunek et al.)
constexpr float kDefaultAirEntryHead = -0.02f;

//! Convert standard van Genuchten-Mualem parameters to the air-entry (modified) model
/**
 * theta_r, alpha and n are kept. With match_standard_curves (the default) the modified model is the standard one
 * truncated at h_e: theta_s and K_s become the standard theta and K at h_e, so theta(h) and K(h) equal the standard
 * curves exactly for every h < h_e, and only the band h_e <= h < 0, where the standard K drops steeply for n < 2,
 * becomes saturated. theta_s drops by (theta_s - theta_r)(1 - S_c) (0.002 for loam, at most 0.009 for sand at
 * h_e = -2 cm) and K_s by the factor F_K (see airEntryScales(); 1.3 for sand to about 10 for clay). Without it
 * theta_s and K_s are kept as the saturated values (HYDRUS's "air-entry value of -2 cm" convention), which raises
 * the unsaturated K by F_K and theta slightly.
 * \param[in] vg Standard van Genuchten soil
 * \param[in] h_e Air-entry head [m] (< 0)
 * \param[in] match_standard_curves Truncate the standard curves at h_e (true) or keep theta_s and K_s (false)
 */
SoilProperties toModifiedVanGenuchten(const SoilProperties &vg, float h_e = kDefaultAirEntryHead, bool match_standard_curves = true);

//! Get soil hydraulic properties for a USDA texture class and constitutive model
/**
 * SoilModel::VAN_GENUCHTEN returns the Carsel & Parrish (1988) class averages. SoilModel::MODIFIED_VAN_GENUCHTEN
 * converts them with toModifiedVanGenuchten() at h_e = kDefaultAirEntryHead (-2 cm, as HYDRUS): by default the
 * Carsel & Parrish curves truncated at h_e (identical below it, saturated above it). No class averages
 * refitted for the air-entry model have been published for the 12 USDA classes (Schaap & van Genuchten 2006 give
 * 4 texture groups at -4 cm). There is no texture library for SoilModel::GARDNER.
 * \param[in] texture Soil texture name (e.g., "sand", "loam", "clay")
 * \param[in] model Constitutive model
 * \param[in] match_standard_curves For the modified model: see toModifiedVanGenuchten()
 * \return SoilProperties for the given texture and model
 */
SoilProperties getSoilPropertiesFromTexture(const std::string &texture, SoilModel model, bool match_standard_curves = true);

//! Get list of available soil texture names
std::vector<std::string> getAvailableSoilTextures();

//! Check if a soil texture is available in the library
bool isSoilTextureAvailable(const std::string &texture);

//! Raise an error if soil hydraulic properties are not physical
/**
 * Requires 0 <= theta_r < theta_s <= 1, alpha > 0, K_s > 0 and 0 < kr_spline_start <= 1, plus n > 1 for the van Genuchten
 * models and h_e < 0 for SoilModel::MODIFIED_VAN_GENUCHTEN.
 * \param[in] props Soil hydraulic properties to check
 * \param[in] caller Name of the calling function, used in the error message
 */
void validateSoilProperties(const SoilProperties &props, const std::string &caller);


#endif // SOILUTILS_H
