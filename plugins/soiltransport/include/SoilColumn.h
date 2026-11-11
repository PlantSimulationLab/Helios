/** \file "SoilColumn.h" 1D soil column solver for Richards equation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SOILCOLUMN_H
#define SOILCOLUMN_H

#include "SoilUtils.h"
#include <vector>

//! 1D soil column solver for variably saturated flow (Richards equation)
/**
 * Solves the 1D Richards equation using implicit finite differences
 * with Modified Picard iteration for mass conservation.
 *
 * The governing equation is:
 * ∂θ/∂t = ∂/∂z[K(h)(∂h/∂z + 1)] - S_root
 *
 * where z is positive downward from the soil surface.
 */
class SoilColumn {
public:
    //! Constructor
    /**
     * \param[in] n_layers Number of vertical soil layers
     * \param[in] layer_thickness Thickness of each layer [m]
     * \param[in] props Soil hydraulic properties (uniform for column)
     */
    SoilColumn(uint n_layers, float layer_thickness, const SoilProperties &props);

    //! Constructor with variable layer thicknesses
    /**
     * \param[in] layer_thicknesses Vector of layer thicknesses [m]
     * \param[in] props Soil hydraulic properties
     */
    SoilColumn(const std::vector<float> &layer_thicknesses, const SoilProperties &props);

    //! Destructor
    ~SoilColumn();

    // ========================================================================
    // Initial and boundary conditions
    // ========================================================================

    //! Set uniform initial pressure head
    /**
     * \param[in] h_init Initial pressure head [m]
     */
    void setInitialCondition(float h_init);

    //! Set initial pressure head profile
    /**
     * \param[in] h_profile Vector of pressure heads for each layer [m]
     */
    void setInitialCondition(const std::vector<float> &h_profile);

    //! Set hydrostatic initial condition
    /**
     * \param[in] water_table_depth Depth to water table from surface [m]
     */
    void setInitialConditionHydrostatic(float water_table_depth);

    //! Set top boundary condition
    /**
     * \param[in] type Boundary condition type
     * \param[in] value Pressure head [m] for Dirichlet, flux [m/s] for Neumann
     */
    void setTopBoundary(BCType type, float value);

    //! Set a head-limited ("atmospheric") top boundary, as in HYDRUS
    /**
     * The potential flux is applied unless the flux with the surface held at the limiting head is smaller: for
     * infiltration (potential_flux > 0) the soil then takes what it can with the surface ponded at limiting_head
     * (e.g. 0), and for evaporation (potential_flux < 0) it delivers what it can with the surface dried to
     * limiting_head (e.g. -100 m). The face conductivity is that of a Dirichlet face at limiting_head.
     * \param[in] potential_flux Potential flux [m/s] (positive into the soil)
     * \param[in] limiting_head Limiting surface pressure head [m]
     */
    void setTopBoundaryAtmospheric(float potential_flux, float limiting_head);

    //! Water flux through the top boundary at the current state [m/s] (positive into the soil)
    /** For an atmospheric boundary this is the flux actually applied (potential or head-limited). */
    float getTopFlux() const;

    //! Set bottom boundary condition
    /**
     * \param[in] type Boundary condition type
     * \param[in] value Pressure head [m] for Dirichlet, flux [m/s] for Neumann
     */
    void setBottomBoundary(BCType type, float value);

    //! Treat the column as concentric shells around a cylinder (e.g. a root) of radius inner_radius
    /**
     * Layer depths become radial distances from the cylinder surface: layer k is the shell between
     * r = inner_radius + (thickness of the layers before it) and that radius plus its own thickness. The top
     * boundary is the cylinder surface and the bottom boundary the outer radius. Face fluxes scale with
     * the face radius and layer storage with the shell area, so flow is radial; it is taken as horizontal,
     * so gravity is not applied. Boundary fluxes, layer sources and root uptake, getTopFlux() and
     * getTotalWaterStorage() are per unit area of the cylinder surface, except a bottom Neumann flux,
     * which is per unit area of the outer surface. Call it before solving.
     * \param[in] inner_radius Cylinder radius [m] (> 0)
     */
    void setRadialGeometry(float inner_radius);

    //! Set root water uptake distribution (normalized)
    /**
     * \param[in] root_distribution Normalized root length density for each layer
     *            (should sum to 1.0)
     */
    void setRootDistribution(const std::vector<float> &root_distribution);

    //! Set total transpiration rate for root uptake
    /**
     * \param[in] transpiration Total transpiration rate [m/s]
     */
    void setTranspiration(float transpiration);

    //! Get the column's transpiration (root water uptake) flux [m/s]
    float getTranspiration() const {
        return transpiration_;
    }

    // ========================================================================
    // Volumetric source/sink terms (e.g. drip emitters)
    // ========================================================================

    //! Set the volumetric source for a layer (overwrites) [m/s]
    /**
     * The source is a flux per unit horizontal area injected into the layer
     * (positive = water added), i.e. an emitter discharge [m^3/s] divided by the
     * column footprint area [m^2]. It enters the layer water balance exactly like a
     * Neumann influx and persists across solves until changed or cleared.
     * \param[in] layer Layer index (0 = top / surface)
     * \param[in] flux Source flux [m/s] (positive into the soil)
     */
    void setSource(uint layer, float flux);

    //! Add to the volumetric source for a layer [m/s] (accumulates emitters)
    void addSource(uint layer, float flux);

    //! Clear all per-layer source terms
    void clearSources();

    // ========================================================================
    // Heterogeneous soil support
    // ========================================================================

    //! Set soil properties for a single layer
    /**
     * \param[in] layer Layer index (0 = top)
     * \param[in] props Soil properties for this layer
     */
    void setLayerSoilProperties(uint layer, const SoilProperties &props);

    //! Set soil properties for a range of layers
    /**
     * \param[in] start_layer First layer index (inclusive)
     * \param[in] end_layer Last layer index (exclusive)
     * \param[in] props Soil properties for these layers
     */
    void setLayerSoilProperties(uint start_layer, uint end_layer, const SoilProperties &props);

    //! Get soil properties for a specific layer
    /**
     * \param[in] layer Layer index (0 = top)
     * \return Soil properties for this layer
     */
    const SoilProperties &getLayerSoilProperties(uint layer) const;

    // ========================================================================
    // Solver control
    // ========================================================================

    //! Set solver parameters
    void setSolverParameters(const SolverParameters &params);

    //! Solve for one timestep
    /**
     * Advances the solution by dt using Modified Picard iteration.
     * \param[in] dt Timestep [s]
     * \return Number of Picard iterations used (0 if failed to converge)
     */
    uint solve(float dt);

    //! Solve with adaptive timestepping
    /**
     * Advances the solution by target_time using adaptive timestep control.
     * The timestep is adjusted based on Picard iteration convergence:
     * - Increased when convergence is fast (few iterations)
     * - Decreased when convergence is slow (many iterations)
     * - Halved and retried when convergence fails
     *
     * \param[in] target_time Total time to advance [s]
     * \param[in,out] dt Current timestep [s], updated for next call
     * \param[out] steps_taken Number of timesteps taken (optional)
     * \return Actual time advanced [s] (may be less than target_time if solver fails)
     */
    float solveAdaptive(float target_time, float &dt, uint *steps_taken = nullptr);

    //! Get statistics from last adaptive solve
    struct AdaptiveSolveStats {
        uint total_steps; //!< Total timesteps taken
        uint failed_steps; //!< Number of failed steps (retried with smaller dt)
        uint total_iterations; //!< Total Picard iterations across all steps
        float min_dt_used; //!< Minimum dt used during solve
        float max_dt_used; //!< Maximum dt used during solve
        float avg_iterations; //!< Average iterations per successful step
    };
    AdaptiveSolveStats getAdaptiveSolveStats() const {
        return adaptive_stats_;
    }

    // ========================================================================
    // State access
    // ========================================================================

    //! Get number of layers
    uint getNumLayers() const {
        return n_layers_;
    }

    //! Get total column depth [m]
    float getColumnDepth() const;

    //! Get pressure head at a layer
    /**
     * \param[in] layer Layer index (0 = top)
     * \return Pressure head [m]
     */
    float getPressureHead(uint layer) const;

    //! Get all pressure heads
    const std::vector<float> &getPressureHeads() const {
        return h_;
    }

    //! Get water content at a layer
    float getWaterContent(uint layer) const;

    //! Get all water contents
    std::vector<float> getWaterContents() const;

    //! Get hydraulic conductivity at a layer
    float getConductivity(uint layer) const;

    //! Get cached specific moisture capacity at a layer (no recompute) [1/m]
    float getCapacity(uint layer) const {
        return C_[layer];
    }

    //! Get cached water content at a layer (no recompute) [m³/m³]
    /** Returns the stored theta_ from the last constitutive update, avoiding the
        transcendental h_to_theta re-evaluation done by getWaterContent(). */
    float getWaterContentCached(uint layer) const {
        return theta_[layer];
    }

    //! Get total water storage in column [m]
    double getTotalWaterStorage() const;

    //! Get depth to a layer center [m]
    float getLayerDepth(uint layer) const;

    //! Get soil properties (returns first layer for backward compatibility)
    const SoilProperties &getSoilProperties() const {
        return props_[0];
    }

    //! Get weighted average soil water potential for root zone
    /**
     * Returns uptake-weighted soil water potential for coupling with plant hydraulics.
     * \return Weighted soil water potential [m]
     */
    float getWeightedSoilPotential() const;

    //! Get cumulative root water uptake [m]
    float getCumulativeRootUptake() const {
        return cumulative_root_uptake_;
    }

    //! Check if last solve converged
    bool didConverge() const {
        return converged_;
    }

private:
    // Grid parameters
    uint n_layers_;
    std::vector<float> dz_; // Layer thicknesses [m]
    std::vector<float> z_; // Layer center depths [m]
    // Geometry weights (Cartesian: layer_volume_ = dz_, face_area_ = 1; see setRadialGeometry())
    std::vector<double> layer_volume_; // layer volume per unit top-boundary area [m]
    std::vector<double> face_area_; // face area per unit top-boundary area [-], faces 0 (top) .. n_layers (bottom)
    bool radial_ = false; // setRadialGeometry(): horizontal radial flow, no gravity
    double gravity() const {
        return (solver_params_.enable_gravity && !radial_) ? 1.0 : 0.0;
    }

    // Soil properties (one per layer for heterogeneous support)
    std::vector<SoilProperties> props_;

    // State variables
    std::vector<float> h_; // Pressure head [m]
    std::vector<float> theta_; // Water content [m³/m³]
    std::vector<float> K_; // Hydraulic conductivity [m/s]
    std::vector<float> C_; // Specific capacity [1/m]

    // Boundary conditions
    BCType top_bc_type_;
    BCType bottom_bc_type_;
    float top_bc_value_;
    float bottom_bc_value_;
    float top_limit_head_ = 0.0f; // limiting surface head [m] of an atmospheric top boundary
    int top_atm_mode_ = -1; // atmospheric boundary branch: -1 choose (min/max), 0 potential flux, 1 limiting head
    int top_atm_applied_ = -1; // branch used by the last accepted step

    // Root uptake
    std::vector<float> root_dist_; // Normalized root distribution
    float transpiration_; // Total transpiration [m/s]
    float cumulative_root_uptake_; // Cumulative uptake [m]

    // Volumetric source terms (flux into each layer) [m/s], positive = into soil
    std::vector<float> source_;

    // Solver parameters
    SolverParameters solver_params_;
    bool converged_;

    // Adaptive solve statistics
    AdaptiveSolveStats adaptive_stats_;

    // Working arrays for tridiagonal solve
    std::vector<float> a_, b_, c_, d_, x_;
    std::vector<float> h_old_; // Previous iteration values
    std::vector<float> storage_rhs_; // Scratch: Modified-Picard storage RHS per layer
    std::vector<double> h_dbl_; // Newton: double-precision heads carried between steps (see solveNewton)
    mutable std::vector<double> nk_th_, nk_K_, nk_C_, nk_dK_; // Newton: constitutive values of the last assembly
    bool nk_at_h_dbl_ = false; // Newton: nk_* hold the constitutive values at h_dbl_ (reused by the next step)
    // Newton work arrays (members so a step does not allocate)
    std::vector<double> nw_h_, nw_theta_n_, nw_F_, nw_Ft_, nw_Fb_, nw_scale_, nw_scale_t_, nw_sb_, nw_lo_, nw_di_, nw_up_, nw_lo_t_, nw_di_t_, nw_up_t_, nw_delta_, nw_ht_, nw_hb_, nw_cl_, nw_dl_, nw_th_h_, nw_C_h_;

    // Internal methods
    void updateConstitutive();
    void assembleTridiagonal(float dt, const std::vector<float> &theta_n);
    uint solveNewton(float dt);
    //! Flux [m/s] into layer 0 through the top boundary and its derivative with respect to h0, for layer-0 head h0
    //! with conductivity K0 and dK/dh dK0; limited (if given) reports whether an atmospheric boundary is head-limited.
    //! mode: atmospheric branch (-1 whichever limits, 0 potential flux, 1 limiting head)
    double topFlux(double h0, double K0, double dK0, double &dq_dh0, bool *limited = nullptr, int mode = -1, double *roundoff = nullptr) const;
    //! One step of solve() with the atmospheric branch (if any) fixed by top_atm_mode_
    uint solveStep(float dt);
    //! One step of solve() with the atmospheric branch chosen and checked (no sub-stepping)
    uint solveAtmospheric(float dt);
    bool in_adaptive_ = false; // solveAdaptive() is driving solve(): failed steps are retried with a smaller dt
    void assembleNewton(const std::vector<double> &h, const std::vector<double> &theta_n, double dt, std::vector<double> &F, std::vector<double> &scale, std::vector<double> *lo, std::vector<double> *di, std::vector<double> *up,
                        bool constitutive_current = false) const;
    float computeResidual() const;
    void updateWaterContent();
};

#endif // SOILCOLUMN_H
