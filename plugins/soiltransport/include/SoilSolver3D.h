/** \file "SoilSolver3D.h" Full 3D structured-grid Richards-equation solver.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SOILSOLVER3D_H
#define SOILSOLVER3D_H

#include "SoilUtils.h"
#include <cstdint>
#include <functional>
#include <vector>

//! Identifier for one of the six domain boundary faces.
enum class Face3D {
    XMIN, //!< -x face (i = 0)
    XMAX, //!< +x face (i = Nx-1)
    YMIN, //!< -y face (j = 0)
    YMAX, //!< +y face (j = Ny-1)
    ZMIN, //!< -z face (k = 0), bottom
    ZMAX //!< +z face (k = Nz-1), top / surface
};

//! Opaque CUDA device-state for SoilSolver3D's GPU backend (defined in SoilSolver3D.cu).
struct SoilSolver3DGpuState;

//! Full 3D Richards-equation solver on a structured grid
/**
 * Solves the mixed-form Richards equation in 3D,
 *   d theta(h)/dt = div[ K(h) grad(h + z) ] + s,
 * on a cell-centered finite-volume grid (7-point stencil). The elevation z is
 * positive upward, so k = 0 is the bottom of the domain and k = Nz-1 is the top.
 *
 * Time integration is backward Euler with a mass-conservative Modified Picard
 * iteration (Celia et al., 1990). Each Picard iteration forms a symmetric
 * positive-definite linear system that is solved matrix-free with a
 * Jacobi-preconditioned conjugate gradient method. All inner kernels (operator
 * apply, constitutive evaluation, vector operations) are OpenMP-parallel.
 *
 * Boundary conditions are specified per face; the default is no-flow
 * (Neumann with zero flux).
 */
class SoilSolver3D {
public:
    //! Constructor
    /**
     * \param[in] Nx,Ny,Nz Number of cells in each direction
     * \param[in] dx,dy,dz Cell spacings [m]
     * \param[in] props Default (uniform) soil hydraulic properties
     */
    SoilSolver3D(uint Nx, uint Ny, uint Nz, double dx, double dy, double dz, const SoilProperties &props);

    //! Destructor (releases GPU resources if a CUDA backend was used)
    ~SoilSolver3D();

    //! Whether a usable CUDA device is available: false in builds without CUDA, without a device that can run this
    //! build's kernels, or when the environment variable HELIOS_NO_GPU is set (to anything but "0")
    static bool isGPUAvailable();

    //! Not copyable: the solver owns its GPU resources
    SoilSolver3D(const SoilSolver3D &) = delete;
    SoilSolver3D &operator=(const SoilSolver3D &) = delete;

    // ------------------------------------------------------------------
    // Setup
    // ------------------------------------------------------------------

    //! Set soil properties for a single cell (heterogeneous media)
    void setSoilProperties(uint i, uint j, uint k, const SoilProperties &props);

    //! Soil properties of a cell (k = 0 is the bottom layer)
    /**
     * \param[in] i Cell index in x
     * \param[in] j Cell index in y
     * \param[in] k Cell index in z (0 = bottom)
     * \return The cell's soil hydraulic properties
     */
    [[nodiscard]] const SoilProperties &getSoilProperties(uint i, uint j, uint k) const;

    //! Set uniform initial pressure head [m]
    void setInitialCondition(double h);

    //! Set initial pressure head per cell [m] (size Nx*Ny*Nz, x-fastest)
    void setInitialCondition(const std::vector<double> &h);

    //! Set per-cell vertical geometry for terrain-following or variably layered grids
    /**
     * By default cell (i,j,k) is a dx x dy x dz box centred at z = (k + 0.5) dz. This replaces each
     * cell's centre elevation and thickness (lateral spacing stays dx, dy), so layers may slope and vary
     * in thickness, as in a terrain-following grid. Gravity uses the actual centre elevations; vertical
     * faces use the centre-to-centre distance and lateral faces an area of dy (or dx) times the mean
     * thickness of the two cells. Supported by NonlinearMethod::NEWTON_ANALYTIC (CPU and GPU) only.
     * \param[in] z_center Cell-centre elevation [m] per cell (size Nx*Ny*Nz, x-fastest); must increase with k
     * \param[in] thickness Cell thickness [m] per cell (size Nx*Ny*Nz), > 0
     */
    void setCellGeometry(const std::vector<double> &z_center, const std::vector<double> &thickness);

    //! Hold the pressure head of a cell fixed [m] (a specified-head cell, e.g. a canal or lake stage)
    /**
     * The cell's head stays at h for every step and the water it exchanges with its neighbours is
     * counted as external inflow in the mass balance. Supported by NEWTON_ANALYTIC (CPU and GPU) only.
     */
    void setFixedHead(uint i, uint j, uint k, double h);

    //! Release all specified-head cells
    void clearFixedHeads();

    //! Set the specific storage Ss [1/m] per cell (elastic storage of water and matrix; default 0)
    /**
     * Stored water per unit volume becomes theta(h) + Ss * S_w * h with S_w = theta / theta_s, so a
     * saturated cell stores Ss per metre of pressure-head rise. Typical values are 1e-6 to 1e-4 1/m.
     * Supported by NonlinearMethod::NEWTON_ANALYTIC (CPU and GPU) only.
     */
    void setSpecificStorage(const std::vector<double> &Ss);

    //! Set a boundary condition on a face
    /**
     * \param[in] face Which boundary face
     * \param[in] type DIRICHLET (head), NEUMANN (flux, positive = into domain), or FREE_DRAINAGE
     * \param[in] value Pressure head [m] for Dirichlet, flux [m/s] for Neumann (ignored otherwise)
     */
    void setBoundary(Face3D face, BCType type, double value = 0.0);

    //! Set a volumetric source/sink rate for a cell [1/s] (positive = source)
    void setSource(uint i, uint j, uint k, double rate);
    //! Accumulate into the volumetric source at a cell [1/s] (for spreading an emitter over a footprint).
    void addSource(uint i, uint j, uint k, double rate);

    //! Clear all sources
    void clearSources();

    //! Set the root water uptake (volumetric sink) rate for a cell [1/s] (positive = removal)
    /**
     * Kept separate from setSource() so root uptake and external sources (e.g. drip
     * emitters) can be set independently. Enters the equation and the mass balance as a
     * negative source.
     */
    void setUptake(uint i, uint j, uint k, double rate);

    //! Clear all root uptake sinks
    void clearUptake();

    //! Set the surface evaporation (volumetric sink) rate for a cell [1/s] (positive = removal)
    /**
     * Separate from setUptake()/setSource() so surface evaporation (typically applied to
     * the top cells) is independent of root uptake and external sources. Enters the
     * equation and the mass balance as a negative source.
     */
    void setEvaporation(uint i, uint j, uint k, double rate);

    //! Clear all surface evaporation sinks
    void clearEvaporation();

    // ------------------------------------------------------------------
    // Solver control
    // ------------------------------------------------------------------

    //! Nonlinear iteration strategy for each timestep.
    enum class NonlinearMethod {
        PICARD = 0, //!< Modified-Picard only: robust, linear convergence, SPD CG
        NEWTON = 1, //!< Newton-Krylov only: quadratic near the root, costlier per step
        PICARD_NEWTON = 2, //!< Picard warmup then Newton finish: robust globalization + quadratic endgame
        NGS_FAS = 3, //!< Nonlinear Gauss-Seidel (lagged K -> pointwise scalar Newton) [Gu & Wang 2026]
        M_SCHEME = 4, //!< Faithful adaptive M-scheme (Javed/Mitra/Pop 2025): Kirchhoff b/B splitting + certified
                      //!< a-posteriori estimator (Algorithm 1 / "MAdap"). HOMOGENEOUS soils only; CPU only.
        NEWTON_ANALYTIC = 5 //!< (default) Newton-Raphson on an assembled analytic Jacobian (incl. dK/dh), ILU(0)-BiCGSTAB
                            //!< inner solve, water-content-space update + backtracking line search; falls back to
                            //!< Picard on a stalled fixed-dt step. Runs on the CPU and the GPU.
    };

    struct Params {
        uint picard_max = 100; //!< Max Modified Picard iterations per step
        double picard_tol = 1e-6; //!< Picard convergence tol on max |dh| [m]
        uint cg_max = 2000; //!< Max CG iterations per linear solve
        double cg_tol = 1e-9; //!< CG relative residual tolerance
        double C_min = 1e-12; //!< Capacity regularization floor [1/m]
        double K_min = 0.0; //!< Additive conductivity floor [m/s] (0 = off); eases dry wetting fronts
        //! Use 2nd-order BDF2 (with a backward-Euler bootstrap) instead of backward Euler. Off by default: BDF2 does not
        //! conserve water when sources or boundary fluxes change, e.g. an emitter switched on or off (about 1% of the
        //! delivered water), and with variable steps (adaptive stepping, or sub-stepping of a failed step) the loss depends
        //! on the step ratios. Backward Euler conserves water to the solver tolerance.
        bool bdf2 = false;
        bool use_multigrid = true; //!< Use geometric-multigrid CG preconditioner (else Jacobi)
        bool use_gpu = false; //!< Run the solve on the GPU (requires a CUDA build); else CPU/OpenMP
        //! Interblock conductivity weighting: 0=geometric, 1=arithmetic, 2=upstream, 3=harmonic,
        //! 4=upstream relative permeability (harmonic-mean saturated K times the donor cell's k_r).
        //! Arithmetic (default) is the most accurate for infiltration and drip irrigation; harmonic
        //! nearly blocks infiltration into dry soil on practical grids; upstream k_r suits sharply
        //! heterogeneous media.
        int K_weighting = 1;
        //! Anderson acceleration depth for the Modified-Picard fixed point (0 = plain Picard).
        //! Mixes the last m iterates via a small least-squares to shorten Picard's linear tail
        //! on stiff steps; keeps the cheap SPD CG inner solve. 2-3 is typical.
        int anderson_m = 0;
        //! Newton-Krylov (JFNK) controls. The Modified-Picard linear solve is reused as the
        //! preconditioner; jacvec supplies the true matrix-free Jacobian action.
        NonlinearMethod nonlinear_method = NonlinearMethod::NEWTON_ANALYTIC; //!< Per-step nonlinear strategy
        uint newton_max = 50; //!< Max outer Newton iterations per step
        double newton_rtol = 1e-10; //!< Outer convergence: ||R|| <= newton_rtol * ||R_0||
        double newton_theta_tol = 1e-7; //!< NEWTON_ANALYTIC: max per-cell residual in water-content units [-]
        //! NEWTON_ANALYTIC: if > 0, also accept a step once the largest head change in an iteration falls
        //! below this [m], even if the residual test is not yet met (a head-change closure criterion).
        //! 0 (default) requires the residual test. The mass-balance residual records any resulting defect.
        double newton_dvclose = 0.0;
        //! NEWTON_ANALYTIC: skip the line search (take the full truncated Newton step) while the residual
        //! L2 norm [1/s] is below this (backtracking only while the residual is large). 0 (default)
        //! always line-searches.
        double newton_backtrack_limit = 0.0;
        uint bicgstab_max = 500; //!< NEWTON_ANALYTIC: max BiCGSTAB iterations per Newton step
        int gmres_restart = 30; //!< GMRES Krylov subspace size before restart
        //! Hard cap on GMRES inner iterations per Newton step (inexact Newton). The Jacobian is
        //! nonsymmetric and only preconditioned by the symmetric Picard V-cycle, so on stiff
        //! (near-residual, K->0) states it can grind to cg_max without this cap -- thousands of
        //! host-synced inner iters on the GPU. Capping keeps each Newton step cheap; the outer
        //! Newton loop + line search + Eisenstat-Walker absorb the inexactness (and the adaptive
        //! dt shrinks if a step can't make progress). 40-100 is typical.
        int newton_gmres_max = 50;
        //! Newton GMRES preconditioner cost: number of multigrid V-cycles (or Jacobi sweeps)
        //! applied per preconditioning step. A cheap approximate A^{-1} (1-2 sweeps) is far more
        //! efficient than a fully converged Picard solve inside every Krylov iteration.
        int newton_precond_sweeps = 1;
        //! PICARD_NEWTON adaptive trigger: the Picard phase is identical in cost to plain Picard
        //! (cheap max|dh| check, no residual eval) and only hands off to Newton when Picard's
        //! convergence stalls -- i.e. the per-iteration reduction ratio max|dh|_k / max|dh|_{k-1}
        //! stays above this for newton_stall_count consecutive iterations. Fast steps never
        //! trigger (they converge with small ratios) so they pay zero Newton overhead.
        double newton_stall_ratio = 0.3;
        int newton_stall_count = 1;
        //! NGS_FAS: stabilization parameter xi >= 0 (temporal diagonal shift, Gu & Wang 2026 eq 2.7).
        //! Larger = more robust but more temporal damping; 0 works when the storage term already
        //! gives diagonal dominance. Max NGS sweeps reuses picard_max; tolerance reuses picard_tol.
        double ngs_xi = 0.0;
        //! M_SCHEME (faithful, Kirchhoff): the dimensionless stabilization mu = M*tau in the linearization
        //! coefficient L = clamp(phi' + mu, 2*mu, 1+eps). mu=0 -> Newton, large mu -> L-scheme. Until the
        //! certified estimator (Algorithm 1) is wired in, this fixed value is used each iteration.
        double m_mu = 1e-3; //!< INITIAL M-scheme stabilization mu (iteration-0 seed); the certified eq-4.12
                            //!< estimator + Algorithm-1 sweep adapt mu from iteration 1 onward (not fixed)
        double m_eps = 1e-6; //!< the 1+eps cap on the M-scheme linearization coefficient
        bool m_mscheme_mg = false; //!< M_SCHEME inner solve: geometric multigrid (true) vs Jacobi-PCG (false). MG cuts inner
                                   //!< iters ~7x but doesn't beat Jacobi on wall time (the outer loop is the bottleneck), so OFF
        int mg_pre = 2; //!< Pre-smoothing sweeps per multigrid level
        int mg_post = 2; //!< Post-smoothing sweeps per multigrid level
        int mg_coarse_sweeps = 30; //!< Smoothing sweeps on the coarsest level
        int mg_min_dim = 4; //!< Stop coarsening a dimension at this size
        // Adaptive timestepping (used by solveAdaptive):
        double dt_target_rel_change = 0.03; //!< Target per-step max |Δθ|/(θs-θr)
        double dt_min = 1e-3; //!< Minimum timestep [s]
        double dt_max = 3600.0; //!< Maximum timestep [s]
        double dt_growth_max = 2.0; //!< Max single-step dt growth factor
        double dt_reduction_min = 0.25; //!< Min single-step dt reduction factor
        double dt_safety = 0.9; //!< Safety factor on the dt error controller
        //! NEWTON_ANALYTIC solve(dt): when a step fails, sub-step it (down to kRescueMinFraction of dt) before
        //! reporting the failure. Set false when the caller shortens failed steps itself. Same as
        //! SolverParameters::substep_on_failure (1D column).
        bool substep_on_failure = true;
        //! CPU threads (0 = automatic: OMP_NUM_THREADS if set, else one per physical core). The solver is
        //! memory-bound, so hyperthreads add no speed; leaving them free also keeps it from stalling at
        //! OpenMP barriers when other processes are busy. The block-Jacobi ILU uses one block per thread.
        int num_threads = 0;
    };
    //! Set the solver parameters
    /**
     * Raises an error if Params::use_gpu is set in a build without CUDA or without a usable device (see isGPUAvailable()),
     * or together with NonlinearMethod::M_SCHEME, which runs on the CPU only.
     * \param[in] p Solver parameters
     */
    void setParams(const Params &p);
    const Params &getParams() const {
        return params_;
    }

    //! Advance one timestep dt [s] (backward Euler / BDF2 per Params::bdf2).
    /**
     * With NonlinearMethod::NEWTON_ANALYTIC a step that fails to converge is sub-stepped adaptively
     * (solveAdaptive()) over the same interval; if even that fails, the state is left unchanged.
     * \param[in] dt Timestep [s]
     * \param[out] cg_iterations Optional: total CG iterations used this step
     * \return Number of nonlinear iterations (0 if it failed to converge)
     */
    uint solve(double dt, uint *cg_iterations = nullptr);

    //! Advance one timestep dt [s] with an inexact Newton-Krylov (JFNK) solve: matrix-free
    //! Jacobian (jacvec) + GMRES preconditioned by one Modified-Picard linear solve (cg), with
    //! an Armijo line search and Eisenstat-Walker forcing. Mass-conservative and history-advancing
    //! exactly like solve(); reaches the same converged state but with (near-)quadratic outer
    //! convergence. \param[out] gmres_iterations Optional: total inner GMRES iterations.
    //! \return Number of outer Newton iterations (0 if it failed to converge).
    uint solveNewton(double dt, uint *gmres_iterations = nullptr);

    //! Advance one timestep dt [s] with Newton-Raphson on an explicitly assembled 7-point Jacobian.
    /**
     * The Jacobian is analytic, including the dK/dh terms that Modified Picard drops, so convergence
     * is quadratic near the root. The nonsymmetric system is solved with ILU(0)-preconditioned
     * BiCGSTAB (exact in 1D, where ILU(0) of a tridiagonal matrix is its LU factorization).
     * Unsaturated cells take the update in water-content space (theta + C*delta mapped back through
     * the retention curve), which keeps iterates physical where C ~ 0, and a backtracking line search
     * on the residual globalizes the iteration. Converged when every cell's head change is within
     * picard_tol (relative to max(1,|h|)) and its residual, in water-content units, is below
     * newton_theta_tol. Mass-conservative and history-advancing exactly like solve().
     * \param[out] linear_iterations Optional: total inner BiCGSTAB iterations.
     * \return Number of Newton iterations (0 if it failed to converge).
     */
    uint solveNewtonAnalytic(double dt, uint *linear_iterations = nullptr);

    //! Advance one timestep dt [s] with the Picard->Newton switchover: cheap Modified-Picard
    //! iterations globalize from far out, then Newton-Krylov finishes with quadratic convergence
    //! once in the basin (or once Picard stalls). Captures Newton's endgame only when it pays.
    //! \param[out] inner_iterations Optional: total inner linear-solve iterations (CG + GMRES).
    //! \return Number of outer iterations (0 if it failed to converge).
    uint solveNonlinearSwitch(double dt, uint *inner_iterations = nullptr);

    //! Advance one step with the Nonlinear Gauss-Seidel scheme (Gu & Wang, 2026): the diffusion
    //! operator is lagged to the start-of-step head (fixed M), so the only nonlinearity is the
    //! pointwise theta(psi); each cell is a scalar monotone Newton solve. No global Krylov solve,
    //! so it does not suffer the ill-conditioned-Jacobian blowup at sharp dry fronts. Semi-implicit
    //! (first-order in the K-nonlinearity). \return NGS sweeps taken (0 if not converged).
    uint solveNGSFAS(double dt, uint *sweeps = nullptr);

    //! Advance one step with the faithful M-scheme (Javed/Mitra/Pop 2025, arXiv:2508.07420). Works in the
    //! Kirchhoff b/B splitting (u=theta=b(s), w_tilde=w/w_sat=B(s)); eliminates s to an SPD constant-coefficient
    //! screened-Laplacian solve for w_tilde, lagged gravity, then recovers s/theta. HOMOGENEOUS soils ONLY
    //! (the Kirchhoff potential is material-dependent -> discontinuous across interfaces). \return iterations.
    uint solveMScheme(double dt, uint *cg_iterations = nullptr);
    KirchhoffTable kirchhoff_; // cached transform (built on first M-scheme solve)
    bool kirchhoff_built_ = false;
    std::vector<double> s_ms_; // splitting variable s (persists across steps for the M-scheme)
    double mscheme_mu_min_ = 0.0, mscheme_mu_max_ = 0.0; // diagnostic: range of estimator-selected mu (last solve)

    //! Advance to target_time [s] with adaptive timestepping (sized by per-step Δθ).
    /**
     * \param[in] target_time Total time to advance [s]
     * \param[in,out] dt Current/next timestep [s], updated for reuse across calls
     * \param[out] steps_taken Optional: number of accepted steps
     * \return Actual time advanced [s]
     */
    double solveAdaptive(double target_time, double &dt, uint *steps_taken = nullptr);

    //! Reset the BDF2 history (call after a discontinuous change in BCs/state).
    void resetHistory() {
        have_history_ = false;
    }

    // GPU nonlinear-solver instrumentation: per-phase outer iterations and GMRES inner
    // iterations accumulated across solveGPU calls since resetGpuStats(). Diagnoses where the
    // device adaptive Picard->Newton spends its work (Picard warmup vs Newton GMRES inner blowup).
    void resetGpuStats() {
        gpu_picard_outer_ = gpu_newton_outer_ = gpu_picard_gmres_ = gpu_newton_gmres_ = gpu_ls_evals_ = 0;
    }
    uint64_t gpuPicardOuter() const {
        return gpu_picard_outer_;
    } //!< Picard-phase outer iterations
    uint64_t gpuNewtonOuter() const {
        return gpu_newton_outer_;
    } //!< Newton-phase outer iterations
    uint64_t gpuPicardGmres() const {
        return gpu_picard_gmres_;
    } //!< GMRES inner iters in the Picard phase (SPD solve)
    uint64_t gpuNewtonGmres() const {
        return gpu_newton_gmres_;
    } //!< GMRES inner iters in the Newton phase (Jacobian solve)
    uint64_t gpuLineSearchEvals() const {
        return gpu_ls_evals_;
    } //!< residual evals spent in Newton line searches
    double mschemeMuMin() const {
        return mscheme_mu_min_;
    } //!< smallest mu the M-scheme estimator selected (last solve)
    double mschemeMuMax() const {
        return mscheme_mu_max_;
    } //!< largest  mu the M-scheme estimator selected (last solve)

    // ------------------------------------------------------------------
    // Newton-Krylov support (JFNK)
    // ------------------------------------------------------------------

    //! Evaluate the discrete nonlinear Richards residual R(h) at an arbitrary head
    //! field, per cell (same per-volume scaling as the assembled system: R = A(h)*h - b(h)).
    /** This is the quantity the Modified-Picard iteration drives to zero, written
        directly as a self-contained kernel (local K per cell) so it does NOT touch the
        solver's frozen preconditioner state (K_, C_, b_, hm_, diag_). It is the building
        block for matrix-free Jacobian-vector products (J v ~ (R(h+e v) - R(h))/e) and for
        a Newton-Krylov solve preconditioned by one Modified-Picard linear solve.
        \param[in] h Pressure-head field [m], length numCells().
        \param[out] F Residual, resized to numCells(). F = 0 (to tolerance) at a converged step.
        \param[in] a0,a1,a2 Time-discretization weights (dθ/dt ~ a0 θ^{n+1}+a1 θ^n+a2 θ^{n-1}).
        Uses the current theta_n_/theta_nm1_ history, so it is consistent only with the step
        those weights belong to. */
    void residual(const std::vector<double> &h, std::vector<double> &F, double a0, double a1, double a2) const;

    //! Compute the time-discretization weights (a0,a1,a2) the next solve(dt) would use,
    //! given the current BDF2/BE history state. Lets callers evaluate residual() consistently.
    void timeWeights(double dt, double &a0, double &a1, double &a2) const;

    //! Matrix-free Jacobian-vector product J(h) v ~ (R(h+e v) - R(h))/e by one-sided finite
    //! difference, e auto-scaled from |h|,|v|. F0 MUST equal residual(h,...) (pass it in so it
    //! is evaluated once per Newton step and reused across Krylov iterations). Captures the
    //! full Jacobian including the dK/dh terms the Modified-Picard operator drops; the
    //! dtheta/dh part is exact because residual() is written in theta(h), not the capacity.
    void jacvec(const std::vector<double> &h, const std::vector<double> &F0, const std::vector<double> &v, std::vector<double> &Jv, double a0, double a1, double a2) const;

    //! L2 norm of the nonlinear residual at each outer iteration of the most recent
    //! solveNewton() (front = initial residual, back = converged). A sharply falling ratio
    //! ||R_{k+1}||/||R_k|| -> 0 is the signature of (near-)quadratic Newton convergence;
    //! a roughly constant ratio would indicate it has degenerated to a linear (Picard-like) rate.
    const std::vector<double> &getNewtonResidualHistory() const {
        return newton_resid_history_;
    }

    //! Action of the Modified-Picard operator A(h) v = a0 C(h) v - L_{K(h)} v with K,C frozen
    //! at h: the SPD M-matrix the CG inner solve uses, and the natural preconditioner for
    //! jacvec. NOTE: freezes K_/C_/hm_ at h as a side effect.
    void picardApply(const std::vector<double> &h, const std::vector<double> &v, std::vector<double> &Av, double a0);

    //! Right-preconditioned restarted GMRES(restart) for a general (possibly nonsymmetric)
    //! matrix-free operator. Solves apply(x) = b to ‖b - A x‖ <= tol*‖b‖.
    /** \param[in] apply   y = A x (e.g. jacvec at the current Newton iterate, or picardApply).
        \param[in] precond z = M^{-1} r (e.g. one cg() Picard solve); pass identity for none.
        \param[in] b       right-hand side. \param[in,out] x  initial guess / solution.
        \param[in] restart Krylov subspace size before restart. \param[in] maxit Max total inner
        iterations. \param[in] tol Relative residual tolerance.
        \return Total inner (matrix-vector) iterations performed. */
    uint gmres(const std::function<void(const std::vector<double> &, std::vector<double> &)> &apply, const std::function<void(const std::vector<double> &, std::vector<double> &)> &precond, const std::vector<double> &b, std::vector<double> &x,
               int restart, int maxit, double tol);

    //! GPU residual R(h) (CUDA builds only): device kResidual with the current theta_n_/theta_nm1_/props_
    //! and the net source src_-uptake_-evap_, per Params::K_weighting. Diagnostic/test wrapper that
    //! uploads host state and downloads F; the device-resident GPU Newton path reuses the kernel.
    void residualGPU(const std::vector<double> &h, std::vector<double> &F, double a0, double a1, double a2);
    //! GPU matrix-free Jacobian-vector product (CUDA builds only): one-sided FD of kResidual.
    void jacvecGPU(const std::vector<double> &h, const std::vector<double> &v, std::vector<double> &Jv, double a0, double a1, double a2);

    // ------------------------------------------------------------------
    // State access
    // ------------------------------------------------------------------

    uint nx() const {
        return Nx_;
    }
    uint ny() const {
        return Ny_;
    }
    uint nz() const {
        return Nz_;
    }
    //! Cell spacing in x [m]
    double dx() const {
        return dx_;
    }
    //! Cell spacing in y [m]
    double dy() const {
        return dy_;
    }
    //! Cell spacing in z [m]
    double dz() const {
        return dz_;
    }
    uint numCells() const {
        return Nx_ * Ny_ * Nz_;
    }
    uint index(uint i, uint j, uint k) const {
        return i + Nx_ * (j + Ny_ * k);
    }

    double getPressureHead(uint i, uint j, uint k) const {
        return h_[index(i, j, k)];
    }
    double getWaterContent(uint i, uint j, uint k) const;
    const std::vector<double> &getPressureHeads() const {
        return h_;
    }

    //! Total water storage in the domain [m^3]
    double getTotalWaterStorage() const;

    //! Net water volume that has entered through sources + boundaries since the
    //! last resetMassBalance() [m^3] (positive = into the domain). Compare with
    //! the change in total storage for a mass-balance check. Note: under BDF2 this
    //! is integrated with a first-order quadrature, so it differs from the actual
    //! storage change by the integrator's O(dt^2) truncation, NOT a conservation
    //! error. For the true conservation check use getMassBalanceResidual().
    double getCumulativeInflux() const {
        return cumulative_influx_;
    }

    //! Integrator-consistent mass-conservation residual since resetMassBalance() [m^3].
    /**
     * Per accepted step the scheme enforces, summed over cells,
     *   Vcell*(a0 theta^{n+1} + a1 theta^n + a2 theta^{n-1}) = external influx rate,
     * with the SAME BDF2/BE weights (a0,a1,a2) the solver used. This accumulates the
     * defect of that identity times dt, so it is ~0 (to Picard/CG tolerance) for a
     * conservative scheme regardless of the time integrator. This is the meaningful
     * "is water conserved" number; getCumulativeInflux() vs storage change is not,
     * because the two are integrated with different-order quadratures under BDF2.
     */
    double getMassBalanceResidual() const {
        return cumulative_balance_residual_;
    }

    //! Reset the mass-balance accumulators (influx and conservation residual).
    void resetMassBalance() {
        cumulative_influx_ = 0.0;
        cumulative_balance_residual_ = 0.0;
    }

    //! Total Modified-Picard iterations across all accepted/rejected steps since the
    //! counter was last reset. Useful for diagnosing solver effort / Anderson speedup.
    uint64_t getTotalPicardIterations() const {
        return total_picard_iters_;
    }
    //! Reset the cumulative Picard-iteration counter.
    void resetPicardCounter() {
        total_picard_iters_ = 0;
    }

private:
    uint64_t total_picard_iters_ = 0; // cumulative Modified-Picard iterations
    std::vector<double> newton_resid_history_; // per-iter ||R|| of the last solveNewton()

    // NEWTON_ANALYTIC: 7-point Jacobian (diagonal + coefficient on each neighbor), its ILU(0)
    // pivots, and per-cell constitutive values at the current iterate.
    std::vector<double> jd_, jxm_, jxp_, jym_, jyp_, jzm_, jzp_, ilu_d_;
    std::vector<double> nk_K_, nk_dK_, nk_C_, nk_theta_, nk_scale_, nk_ext_;
    void assembleNewton(const std::vector<double> &h, std::vector<double> &F, double a0, double a1, double a2, bool with_jacobian);
    uint solveStep(double dt, uint *cg_iterations = nullptr); // one step of solve() without the sub-stepping recovery
    bool in_adaptive_ = false; // true while solveAdaptive() drives solve() (failed steps get a smaller dt)
    std::vector<int> ilu_blk_; // block-Jacobi ILU(0): first row (or column) of each lateral slab block (+ end)
    int ilu_dim_ = 1, ilu_blk_n_ = 0, ilu_blk_threads_ = 0; // slab dimension (0 = x, 1 = y); grid size and thread count the blocks were built for
    int nthreads_ = resolveThreadCount(0); // OpenMP team size for every parallel loop (from Params::num_threads)
    void buildIluBlocks();
    void ilu0Factor();
    void ilu0Solve(const std::vector<double> &r, std::vector<double> &z) const;
    void jacobianApply(const std::vector<double> &v, std::vector<double> &out) const;
    uint bicgstab(const std::vector<double> &b, std::vector<double> &x, double rtol, uint maxit);
    // GPU instrumentation counters (accumulated in the .cu device solve paths).
    uint64_t gpu_picard_outer_ = 0, gpu_newton_outer_ = 0, gpu_picard_gmres_ = 0, gpu_newton_gmres_ = 0, gpu_ls_evals_ = 0;
    double cumulative_influx_ = 0.0; // net water in via sources + boundaries [m^3]
    double cumulative_balance_residual_ = 0.0; // integrator-consistent conservation defect [m^3]
    double stepInflux(); // net external influx rate at the converged state [m^3/s] (head-form flux)
    double stepInfluxMScheme() const; // same, but with the M-scheme's Kirchhoff diffusive boundary flux

    uint Nx_, Ny_, Nz_;
    double dx_, dy_, dz_, Vcell_;
    std::vector<double> zc_, thk_; // per-cell centre elevation and thickness [m] (uniform by default)
    bool uniform_geometry_ = true; // false once setCellGeometry() is used (NEWTON_ANALYTIC only)
    bool picard_fallback_ = false; // a Picard step is running as the fallback of a failed NEWTON_ANALYTIC step
    std::vector<double> fixed_h_; // specified pressure head per cell [m]; NaN where the head is free
    std::vector<double> ss_; // specific storage per cell [1/m]
    std::vector<double> h_infl_; // head at each cell's retention-curve inflection [m] (Newton trial-step switch)
    std::vector<double> h_sat_; // head at and above which each cell is saturated [m] (0, or the air-entry head)
    bool has_ss_ = false;
    double storedWater(int c, double h) const; // theta + Ss * S_w * h
    uint n_fixed_ = 0;
    double cellVolume(int c) const {
        return dx_ * dy_ * thk_[c];
    }
    std::vector<double> h_; // pressure head at current time
    std::vector<SoilProperties> props_; // per-cell soil properties
    std::vector<double> src_; // per-cell volumetric source [1/s] (e.g. emitters)
    std::vector<double> uptake_; // per-cell root water uptake sink [1/s] (positive = removal)
    std::vector<double> evap_; // per-cell surface evaporation sink [1/s] (positive = removal)

    struct BC {
        BCType type = BCType::NEUMANN; // default no-flow
        double value = 0.0;
    };
    BC bc_[6]; // indexed by Face3D

    Params params_;

    // Working arrays (allocated once)
    std::vector<double> K_, C_, theta_m_, theta_n_, theta_nm1_, b_, diag_;
    std::vector<double> hm_, r_, z_, p_, Ap_, u_;

    // Anderson-acceleration scratch (history of fixed-point residual/value differences)
    std::vector<double> aa_f_, aa_fprev_, aa_gprev_; // current residual, previous residual & g
    std::vector<std::vector<double>> aa_dF_, aa_dG_; // difference columns (up to anderson_m)
    //! One Picard fixed-point update with optional Anderson mixing; returns the next iterate
    //! into hm_. f = u_ - hm_ (the Picard increment) is supplied; on the first call (or
    //! anderson_m==0) it falls back to plain Picard (hm_ = u_).
    void andersonUpdate(int N, int mAA, bool &have_prev);

    // BDF2 history
    double dt_prev_ = 0.0;
    bool have_history_ = false; // true once theta_nm1_ holds a valid previous step

    void updateKC(const std::vector<double> &hm);
    //! Commit a converged step: h_ holds the converged head and theta_n_/theta_nm1_ are still
    //! pre-swap. Accumulates the mass-balance bookkeeping then advances the BDF2 history.
    void commitStep(double dt, double a0, double a1, double a2);
    //! Cheap approximate A^{-1} r for the Newton GMRES preconditioner: params_.newton_precond_sweeps
    //! multigrid V-cycles (or Jacobi sweeps) about the frozen Modified-Picard operator at hm_.
    void applyPicardPrecond(const std::vector<double> &r, std::vector<double> &z, double a0);
    // Time-discretized system uses dθ/dt ≈ a0 θ^{n+1} + a1 θ^n + a2 θ^{n-1}.
    void buildSystem(const std::vector<double> &hm, double a0, double a1, double a2);
    void matvec(const std::vector<double> &v, std::vector<double> &out, double a0) const;
    //! Interblock conductivity for a face per Params::K_weighting. Ka/Kb are the cell
    //! conductivities; Ha/Hb their TOTAL heads (pressure head + elevation) for the upstream
    //! option. Symmetric in (a,b) for every option, so the CG matrix stays symmetric.
    double faceCond(double Ka, double Kb, double Ha, double Hb, double Ksa, double Ksb) const;
    double bcFaceK(int face, double Kc, double hc, const SoilProperties &pc) const;
    uint cg(std::vector<double> &u, double a0);

    // Geometric-multigrid preconditioner -----------------------------------
    struct MGLevel {
        int Nx, Ny, Nz;
        double dx, dy, dz;
        bool dir[6]; // face is Dirichlet
        std::vector<double> K, Cdt, cfx, cfy, cfz, diag; // coeffs + 7-point stencil
        std::vector<double> u, b, res; // work vectors
        std::vector<double> fas; // FAS: restricted solution I*psi (NGS_FAS only)
        std::vector<SoilProperties> props; // per-cell soil (NGS_FAS; coarse = restricted)
        int N() const {
            return Nx * Ny * Nz;
        }
        int idx(int i, int j, int k) const {
            return i + Nx * (j + Ny * k);
        }
    };
    std::vector<MGLevel> mg_;
    void mgBuildStencil(MGLevel &L) const;
    void mgBuild(double a0); // build hierarchy from K_, a0*C_
    void mgBuildMScheme(double D0, const std::vector<double> &react); // M-scheme: constant-coef D0 Laplacian + reaction diagonal
    void mgMatvec(const MGLevel &L, const std::vector<double> &u, std::vector<double> &out) const;
    void mgSmooth(MGLevel &L, std::vector<double> &u, const std::vector<double> &b, int sweeps) const;
    void mgVcycle(int lvl);
    void mgApply(std::vector<double> &z, const std::vector<double> &r); // z = M^{-1} r
    // FAS multigrid for the NGS scheme (Gu & Wang 2026): nonlinear operator N_L(u)=a0*theta(u)+xi*u
    // +diffusion, NGS smoother (pointwise scalar Newton), reusing the mg_ hierarchy's lagged face
    // coefficients and full-weighting/piecewise-constant transfers. Uniform soil props (props_[0]).
    void ngsSweepLevel(const MGLevel &L, std::vector<double> &u, const std::vector<double> &b, double a0, double xi, int sweeps) const;
    void ngsOperatorLevel(const MGLevel &L, const std::vector<double> &u, std::vector<double> &out, double a0, double xi) const;
    void mgVcycleFAS(int lvl, double a0, double xi);

    // CUDA backend (Jacobi-preconditioned matrix-free CG on the GPU) ---------
    // Mirrors solve()'s Modified Picard + CG exactly; selected via Params::use_gpu.
    SoilSolver3DGpuState *gpu_ = nullptr; // lazily allocated device state (nullptr on CPU-only builds)
    //! Launch a trivial kernel to check that the device can run this build's kernels (false on CPU-only builds)
    static bool gpuKernelsRun();
    //! Free all device state (after a GPU error it may be half-built); the next GPU solve allocates it again
    void releaseGpuState();
    bool gpu_props_dirty_ = true; // soil props changed since last GPU upload -> re-upload (and rebuild coarse FAS props)
#ifdef HELIOS_CUDA_AVAILABLE
    // One full timestep on the GPU. theta_n_ must already hold theta^n; handles
    // the Picard loop, history bookkeeping, and writes back h_/theta on accept.
    uint solveGPU(double dt, double a0, double a1, double a2, uint *cg_iterations);
    // Device-resident Newton-Krylov step (dispatched from solveGPU when nonlinear_method==NEWTON).
    uint solveNewtonGPU(double dt, double a0, double a1, double a2);
    uint solveNewtonAnalyticGPU(double dt, double a0, double a1, double a2, uint *linear_iterations);
    // Device-resident adaptive Picard->Newton (dispatched when nonlinear_method==PICARD_NEWTON).
    uint solveNonlinearSwitchGPU(double dt, double a0, double a1, double a2);
    // Device-resident NGS-FAS (lagged-K nonlinear multigrid; dispatched when nonlinear_method==NGS_FAS).
    uint solveNGSFASGPU(double dt, double a0, double a1, double a2);
    void mgVcycleFASGPU(int lvl, double a0, double xi); // FAS V-cycle (red-black NGS smoother)
    // GPU geometric-multigrid preconditioner (device-resident V-cycle).
    void mgSetupGPU(); // build level hierarchy + allocate (once)
    void mgBuildGPU(double a0); // fill per-level coefficients each Picard iteration
    void mgVcycleGPU(int lvl); // recursive V-cycle (launches kernels, no host sync)
    void mgApplyGPU(double *z, const double *r); // z = M^{-1} r via one V-cycle
#endif
};

#endif // SOILSOLVER3D_H
