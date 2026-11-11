/** \file "SoilTransportModel.h" Main plugin class for soil water transport simulation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SOILTRANSPORTMODEL_H
#define SOILTRANSPORTMODEL_H

#include "Context.h"
#include "SoilUtils.h"
#include "SoilColumn.h"
#include "SoilGrid.h"
#include "SoilSolver3D.h"
#include "RootSoilCoupling.h"
#include <memory>

//! Soil water transport model for variably saturated flow
/**
 * Solves the Richards equation for variably saturated water flow in soil,
 * with optional coupling to plant root systems.
 *
 * Features:
 * - 1D (column-based), quasi-3D (operator splitting), or full 3D solver modes
 * - Van Genuchten-Mualem hydraulic functions
 * - Statistical root distributions for water uptake
 * - CUDA GPU acceleration with OpenMP CPU fallback
 *
 * Coupling with PlantHydraulics:
 * - Returns soil water potential via getSoilWaterPotential()
 * - Accepts transpiration demand via setTranspiration()
 * - Lagged coupling permitted (run at different timesteps)
 */
class SoilTransportModel {
public:
    //! Constructor
    /**
     * \param[in] context Pointer to the Helios context
     */
    explicit SoilTransportModel(helios::Context *context);

    //! Destructor
    ~SoilTransportModel();

    //! Self-test routine (runs the plugin's doctest unit-test suite)
    /**
     * \param[in] argc Argument count forwarded to doctest
     * \param[in] argv Argument values forwarded to doctest
     * \return 0 if all tests pass, non-zero if any fail
     */
    static int selfTest(int argc = 0, char **argv = nullptr);

    // ========================================================================
    // Domain setup
    // ========================================================================

    //! Define soil domain with uniform grid
    /**
     * The domain, grid resolution and solver type must be set before the domain is set up, which happens in
     * initializeDomain() or in the first call that needs it (e.g. setInitialCondition()); changing them afterwards raises
     * an error.
     * \param[in] origin Domain origin (x, y, z) where z is top surface [m]
     * \param[in] extent Domain extent (dx, dy, dz) total size [m]
     */
    void defineSoilDomain(const helios::vec3 &origin, const helios::vec3 &extent);

    //! Set grid resolution
    /**
     * \param[in] nx Number of columns in x direction
     * \param[in] ny Number of columns in y direction
     * \param[in] nz Number of vertical layers
     */
    void setGridResolution(uint nx, uint ny, uint nz);

    //! Set single-column resolution (for 1D simulations)
    /**
     * \param[in] nz Number of vertical layers
     */
    void setColumnResolution(uint nz);

    //! Set column grid dimensions (for quasi-3D)
    /**
     * \param[in] nx Number of columns in x
     * \param[in] ny Number of columns in y
     */
    void setColumnGrid(uint nx, uint ny);

    //! Initialize the domain (call after setting resolution and extent)
    void initializeDomain();

    // ========================================================================
    // Soil properties
    // ========================================================================

    //! Set uniform soil properties for entire domain
    /**
     * \param[in] props Soil hydraulic properties
     */
    void setSoilProperties(const SoilProperties &props);

    //! Set soil properties from texture name
    /**
     * Keeps the current initial and boundary conditions. Raises an error for a texture that is not in the library
     * (see isSoilTextureAvailable()).
     * \param[in] texture Soil texture name (e.g., "sand", "loam", "clay")
     */
    void setSoilPropertiesFromTexture(const std::string &texture);

    //! Set soil properties from texture name for a given constitutive model
    /**
     * \param[in] texture Soil texture name (e.g., "sand", "loam", "clay")
     * \param[in] model SoilModel::VAN_GENUCHTEN or SoilModel::MODIFIED_VAN_GENUCHTEN (see getSoilPropertiesFromTexture())
     * \param[in] match_standard_curves For the modified model: see toModifiedVanGenuchten()
     */
    void setSoilPropertiesFromTexture(const std::string &texture, SoilModel model, bool match_standard_curves = true);

    //! Set soil properties for a specific column
    /**
     * Sets up the domain if it has not been set up yet, and keeps the column's initial and boundary conditions.
     * \param[in] i Column index in x
     * \param[in] j Column index in y
     * \param[in] props Soil hydraulic properties
     */
    void setSoilProperties(uint i, uint j, const SoilProperties &props);

    // ========================================================================
    // Initial and boundary conditions
    // ========================================================================

    //! Set uniform initial pressure head
    /**
     * \param[in] h_init Initial pressure head [m]
     */
    void setInitialCondition(float h_init);

    //! Set hydrostatic initial condition
    /**
     * \param[in] water_table_depth Depth to water table from surface [m]
     */
    void setInitialConditionHydrostatic(float water_table_depth);

    //! Set top boundary flux (Neumann BC)
    /**
     * \param[in] flux Surface flux [m/s] (positive = infiltration into soil)
     */
    void setTopBoundaryFlux(float flux);

    //! Apply supply-limited surface evaporation from a potential (atmospheric-demand) rate
    /**
     * For each surface cell the actual evaporation is E_act = beta(theta_surf) * E_pot,
     * where beta is the dry-surface-layer reduction factor (see surfaceEvaporationFactor):
     * a wet surface evaporates near the potential rate, a drying surface less, falling to
     * ~0 as it approaches residual water content. The actual evaporation is applied as a
     * per-cell sink at the surface (and is included in the mass balance). Call once per
     * step with the current potential rate. Each surface cell uses its own soil properties. Requires the full-3D solver
     * (raises an error otherwise).
     * \param[in] potential_rate Potential evaporation flux [m/s] (atmospheric demand)
     * \param[in] aerodynamic_resistance Aerodynamic resistance [s/m] (default 50; lower = stronger evaporation)
     * \return Total actual evaporation volume rate over the surface [m^3/s]
     */
    float setSurfaceEvaporation(float potential_rate, float aerodynamic_resistance = 50.0f);

    //! Set top boundary pressure head (Dirichlet BC)
    /**
     * \param[in] h Pressure head at surface [m]
     */
    void setTopBoundaryPressure(float h);

    //! Set bottom boundary to free drainage (unit gradient)
    void setBottomBoundaryFreeDrainage();

    //! Set bottom boundary pressure head (Dirichlet BC)
    /**
     * \param[in] h Pressure head at bottom [m]
     */
    void setBottomBoundaryPressure(float h);

    //! Set bottom boundary flux (Neumann BC)
    /**
     * \param[in] flux Bottom flux [m/s] (positive = out of domain)
     */
    void setBottomBoundaryFlux(float flux);

    //! Add a point water input (e.g. a drip emitter) at the soil surface
    /**
     * Injects water at a fixed volumetric flow rate as a source term; emitters persist across run() calls until
     * clearSurfaceEmitters() is called. With the full 3D solver the discharge is spread over the surface cells within a
     * circular wetting footprint of the given radius (about the wetted bulb of a real emitter), so the per-cell source
     * rate does not depend on the grid resolution; a single-cell point source would make it grow as the grid is refined
     * and collapse the adaptive timestep. With the column solvers (SOLVER_1D, SOLVER_QUASI_3D) the discharge enters the
     * surface layer of the column containing (x, y), and the radius is not used. Raises an error for a point outside the
     * domain.
     *
     * \param[in] x World x location of the emitter [m]
     * \param[in] y World y location of the emitter [m]
     * \param[in] flow_rate Emitter discharge [m^3/s] (positive = into the soil)
     * \param[in] radius Radius of the wetting footprint [m] (full 3D solver only)
     */
    void addSurfaceEmitter(float x, float y, float flow_rate, float radius = 0.1f);

    //! Remove all surface emitters / cell sources
    void clearSurfaceEmitters();

    // ========================================================================
    // Root coupling
    // ========================================================================

    //! Enable root water uptake
    void enableRootUptake();

    //! Set root primitives for a plant from Helios UUIDs (not implemented yet)
    /**
     * \note Mapping explicit root geometry onto the soil grid is not implemented yet, so this always raises
     *       an error. Use setRootDistribution() or setRootDistributionWeibull() to describe the root zone.
     * \param[in] plant_id Plant identifier
     * \param[in] root_UUIDs Vector of primitive UUIDs representing roots
     */
    void setRootPrimitives(uint plant_id, const std::vector<uint> &root_UUIDs);

    //! Set statistical root distribution for a plant
    /**
     * \param[in] plant_id Plant identifier
     * \param[in] params Root distribution parameters
     */
    void setRootDistribution(uint plant_id, const RootDistributionParams &params);

    //! Set a Weibull root length distribution for a plant (1D depth, or 3D depth x radius)
    /**
     * Defines the root length distribution used to partition transpiration demand into
     * per-cell root water uptake. Use radial=false for a 1D depth profile (horizontally
     * uniform) or radial=true for a 3D depth x radius profile about the plant base. Call
     * setTranspiration() afterward to drive the uptake.
     * \param[in] plant_id Plant identifier
     * \param[in] params Weibull distribution parameters
     */
    void setRootDistributionWeibull(uint plant_id, const WeibullRootParameters &params);

    //! Set root radial conductivity
    /**
     * \param[in] L_r Root radial conductivity [m/(s·MPa)]
     */
    void setRootRadialConductivity(float L_r);

    //! Set the root water-stress (Feddes) and uptake-compensation (Jarvis) parameters
    /**
     * Each root cell's uptake is scaled by alpha(h): 1 for h >= h_onset, falling linearly to 0 at
     * h_wilting, so roots stop extracting from soil that is too dry. Uptake is redistributed toward
     * unstressed roots, so the full transpiration demand is met while the root-weighted mean alpha is
     * at least omega_c; below that, uptake is reduced in proportion (see getRootUptakeFlux()).
     * \param[in] h_onset Pressure head where water stress begins [m] (default -4)
     * \param[in] h_wilting Pressure head where uptake stops [m] (default -150, about -1.5 MPa)
     * \param[in] omega_c Critical root-zone stress index, 0-1 (default 0.5; 1 disables compensation)
     */
    void setRootWaterStressParameters(float h_onset, float h_wilting, float omega_c);

    //! Set transpiration demand for a plant
    /**
     * The plant needs a root distribution (setRootDistribution() or setRootDistributionWeibull()) first; otherwise an
     * error is raised.
     * \param[in] plant_id Plant identifier
     * \param[in] transpiration Transpiration flux per unit ground area of the soil domain [m/s] (a plant's volumetric
     *            transpiration rate [m^3/s] divided by the domain's footprint area)
     */
    void setTranspiration(uint plant_id, float transpiration);

    // ========================================================================
    // Solver configuration
    // ========================================================================

    //! Set solver type
    /**
     * Switching to or from SOLVER_3D raises an error once the domain has been set up, and switching to SOLVER_3D raises
     * one if SolverParameters::enable_gravity is false (column solvers only).
     * \param[in] type Solver type (1D, quasi-3D, or full 3D)
     */
    void setSolverType(SolverType type);

    //! Enable horizontal coupling for quasi-3D
    /**
     * \param[in] interval Couple every N vertical timesteps
     */
    void enableHorizontalCoupling(uint interval = 10);

    //! Set solver parameters
    /**
     * Checks the whole parameter set before storing any of it, so a rejected set leaves the model unchanged. Raises an
     * error for horizontal_coupling_interval < 1, dt_min <= 0 or dt_max < dt_min, enable_gravity = false with SOLVER_3D,
     * and NonlinearMethod::M_SCHEME (CPU only) after enableGPU(). With the automatic GPU choice, M_SCHEME simply runs on
     * the CPU.
     * \param[in] params Solver parameters
     */
    void setSolverParameters(const SolverParameters &params);

    //! Run the full 3D solver on the GPU
    /**
     * By default the choice is automatic: the 3D solver uses the GPU when one is available (see isGPUAvailable()) and the
     * nonlinear method runs on it (NonlinearMethod::M_SCHEME is CPU only). Calling this requires the GPU, overriding the
     * automatic choice. Raises an error if this build has no CUDA support, no usable CUDA device is found, or the
     * nonlinear method is M_SCHEME.
     */
    void enableGPU();

    //! Whether a usable CUDA device is available to the soil solver
    /**
     * False in builds without CUDA, when no CUDA device that can run this build's kernels is found, or when the
     * environment variable HELIOS_NO_GPU is set
     * (to anything but "0").
     */
    static bool isGPUAvailable();

    //! Run the full 3D solver on the CPU (OpenMP) even when a GPU is available
    void disableGPU();

    //! Whether the full 3D solver runs on the GPU (with the automatic choice, whether a usable GPU is available and the nonlinear method runs on it)
    bool isGPUEnabled() const;

    // ========================================================================
    // Run simulation
    // ========================================================================

    //! Run simulation for one timestep
    /**
     * Raises an error if the solver cannot complete the step, typically because a boundary or root-uptake flux exceeds
     * what the soil can deliver or absorb. The simulation clock and the cumulative root uptake then count only the time
     * that was advanced: none with fixed stepping (the state is left as it was), or the part completed before the failure
     * with adaptive stepping (the state is at the end of that part). An error raised part-way through a step for another
     * reason (e.g. a quasi-3D cell reaching saturation, or a GPU failure) leaves the state and the clock as they were
     * before the call.
     * \param[in] dt Timestep [s]
     */
    void run(float dt);

    // ========================================================================
    // Output for coupling
    // ========================================================================

    //! Get weighted soil water potential for a plant
    /**
     * Returns uptake-weighted soil water potential for PlantHydraulics coupling. Raises an error for a plant without a
     * root distribution.
     * \param[in] plant_id Plant identifier
     * \return Weighted soil water potential as a pressure head [m]. PlantHydraulicsModel works in MPa: multiply by
     *         9.80665e-3 MPa/m (rho_w g) before passing it to PlantHydraulicsModel::setSoilWaterPotentialOfPlant().
     */
    float getSoilWaterPotential(uint plant_id) const;

    //! Get water content at a world position
    /**
     * Points on the domain boundary (e.g. the soil surface) read the boundary cell; a point outside the domain raises an
     * error.
     * \param[in] position World coordinates [m]
     * \return Volumetric water content [m³/m³]
     */
    float getWaterContent(const helios::vec3 &position) const;

    //! Get pressure head at a world position
    /**
     * Points on the domain boundary (e.g. the soil surface) read the boundary cell; a point outside the domain raises an
     * error.
     * \param[in] position World coordinates [m]
     * \return Pressure head [m]
     */
    float getPressureHead(const helios::vec3 &position) const;

    // ========================================================================
    // Point sensors
    // ========================================================================

    //! Add a point sensor that records water content or pressure head over time
    /**
     * Sampling is a direct O(1) point query at the stored position (the position maps to
     * its containing cell by arithmetic, never a search), so repeated recording is cheap.
     * Each call to recordSensors() appends the current measured value and the simulation
     * time to the sensor's time series. A position outside the domain raises an error; one on its boundary (e.g. the
     * soil surface) reads the boundary cell.
     * \param[in] position World coordinates of the sensor [m]
     * \param[in] quantity Quantity to measure (WATER_CONTENT or PRESSURE_HEAD)
     * \return Sensor id (index) used to retrieve its data
     */
    uint addSensor(const helios::vec3 &position, SoilMeasurement quantity);

    //! Sample every sensor at the current soil state and append to its time series
    /** Call once per output step (e.g. after each run()). */
    void recordSensors();

    //! Number of sensors that have been added
    uint getSensorCount() const;

    //! Recorded measurement time series for a sensor [theta in m³/m³, or head in m]
    const std::vector<float> &getSensorData(uint sensor_id) const;

    //! Simulation times [s] corresponding to each entry of getSensorData()
    const std::vector<float> &getSensorTimes(uint sensor_id) const;

    //! World position of a sensor [m]
    helios::vec3 getSensorPosition(uint sensor_id) const;

    //! Quantity a sensor measures
    SoilMeasurement getSensorQuantity(uint sensor_id) const;

    //! Clear the recorded time series of all sensors (keeps the sensors themselves)
    void clearSensorData();

    //! Write all sensor time series to a CSV file
    /**
     * One row per recorded sample: a "time_s" column followed by one column per sensor
     * (labeled with its quantity and position). Assumes all sensors were recorded
     * together (same number of samples).
     * \param[in] filename Output file path
     */
    void writeSensorData(const std::string &filename) const;

    // ========================================================================
    // Solver diagnostics (stiffness / cost)
    // ========================================================================

    //! Adaptive sub-steps taken in the last run() call (1 if non-adaptive)
    /** A large value together with a tiny getTimestep() means the solver is stiff
        (e.g. a sharp wetting front into dry soil) and the run is grinding. */
    uint getLastSubstepCount() const {
        return last_step_count_;
    }

    //! GPU nonlinear-solver instrumentation passthroughs (0 on CPU/column modes). Reset, then
    //! read after a run() to see where the device adaptive Picard->Newton spent its work.
    void resetSolverStats();
    uint64_t solverPicardOuter() const;
    uint64_t solverNewtonOuter() const;
    uint64_t solverPicardGmres() const;
    uint64_t solverNewtonGmres() const;

    //! Current adaptive timestep [s] the controller has settled on
    /** Collapsing toward dt_min signals a stiff front; recovering toward dt_max means
        the front has relaxed. Returns the 3D or column-grid step depending on mode. */
    float getTimestep() const;

    //! Total Modified-Picard iterations the 3D solver has spent since the last
    //! resetSolverIterationCount() (0 in column modes). Lower for the same run =
    //! cheaper convergence (e.g. Anderson acceleration shortening the Picard tail).
    uint64_t getSolverIterationCount() const;

    //! Reset the cumulative Picard-iteration counter (3D solver only).
    void resetSolverIterationCount();

    //! Get total water storage in domain [m³]
    double getTotalWaterStorage() const;

    //! Net water volume that has entered through emitters + boundaries since the
    //! start of the run [m^3] (3D solver only). With backward Euler (the default) it
    //! equals the change in storage to the solver tolerance. With BDF2
    //! (SolverParameters::bdf2) it does not: BDF2 gains or loses water whenever
    //! the inflow changes. See resetMassBalance(). Raises an error for the column solvers, which do not track it.
    double getCumulativeBoundaryInflux() const;

    //! Mass-balance residual of the time integrator since the run start [m^3] (3D
    //! solver only): its storage term minus the inflow, ~0 to the solver tolerance.
    //! With backward Euler (the default) the storage term is the change in stored
    //! water; with BDF2 it is BDF2's weighted combination of three time levels, so
    //! a ~0 residual does not mean that the stored water matches the inflow
    //! (compare getTotalWaterStorage() with getCumulativeBoundaryInflux()). Raises an error for the column solvers,
    //! which do not track it.
    double getMassBalanceResidual() const;

    //! Reset the mass-balance accumulators (influx + conservation residual)
    void resetMassBalance();

    //! Get cumulative root water uptake for a plant [m]
    /**
     * Raises an error for a plant without a root distribution.
     */
    float getCumulativeRootUptake(uint plant_id) const;

    //! Get the root water uptake flux currently applied for a plant [m/s per unit ground area]
    /**
     * Equal to the transpiration demand set with setTranspiration() unless the root zone is too dry to
     * supply it (see setRootWaterStressParameters()); 0 before a transpiration demand is set. Raises an error for a
     * plant without a root distribution.
     */
    float getRootUptakeFlux(uint plant_id) const;

    // ========================================================================
    // Visualization
    // ========================================================================

    //! Build a visualization of the whole soil domain into the model's Context
    /**
     * Creates a 3D block spanning the soil domain whose outer faces are colored
     * by the chosen field (volumetric water content or pressure head) sampled
     * from the current soil state — initial conditions or solver output — or
     * shown with a plain dirt texture. Display with the Visualizer plugin. The
     * geometry is added to the Context the model was constructed with; the
     * returned UUIDs let you exclude it from a physics pass (e.g. radiation) if
     * needed.
     *
     * \param[in] mode Coloring mode (WATER_CONTENT, PRESSURE_HEAD, or TEXTURE).
     * \return UUIDs of the created primitives.
     */
    std::vector<uint> addSoilVisualization(SoilColorMode mode) const;

    //! Build a soil visualization cut to a fraction of the domain from the origin corner
    /**
     * Each component of \p keep_fraction is in (0,1] and is measured from the
     * domain origin along the domain extent (for z, the origin is the surface).
     * Cutting an axis below 1 exposes the interior cross-section at that plane.
     * Equivalent to addSoilVisualization(mode, {0,0,0}, keep_fraction).
     *
     * \param[in] mode Coloring mode.
     * \param[in] keep_fraction Fraction of each axis to keep (anchored at origin).
     * \return UUIDs of the created primitives.
     */
    std::vector<uint> addSoilVisualization(SoilColorMode mode, const helios::vec3 &keep_fraction) const;

    //! Build a soil visualization of a fractional sub-box of the domain, kept or cut out
    /**
     * \p fraction_min and \p fraction_max are per-axis fractions in [0,1] of the
     * domain extent, measured from the origin corner, defining the lower and
     * upper corners of the sub-box.
     *
     * With \p region == KEEP the sub-box is rendered (the rest hidden); a
     * zero-thickness axis (min == max) collapses it to a cross-section slice
     * plane. With \p region == REMOVE the sub-box is cut out of the full domain
     * (a cutaway) — the sub-box should be anchored at a domain corner (each axis
     * fraction touches 0 or 1) for a clean notch exposing the interior. The whole
     * visualization is colored on a single shared range so faces stay consistent.
     *
     * Every primitive carries int primitive data "soil_cut_face": 1 for virtual
     * cross-section faces exposed by the slice/cutaway (off the domain boundary),
     * 0 otherwise, so callers can select the cut faces (e.g. with
     * Context::filterPrimitivesByData) to render them differently from the outer faces.
     *
     * \param[in] mode Coloring mode.
     * \param[in] fraction_min Lower-corner fractions (per axis, from origin).
     * \param[in] fraction_max Upper-corner fractions (per axis, from origin).
     * \param[in] region KEEP the sub-box, or REMOVE it (cutaway). Default KEEP.
     * \return UUIDs of the created primitives.
     */
    std::vector<uint> addSoilVisualization(SoilColorMode mode, const helios::vec3 &fraction_min, const helios::vec3 &fraction_max, SoilRegion region = SoilRegion::KEEP) const;

    //! Refresh the field data on an existing soil visualization in place (no rebuild)
    /**
     * Re-samples the current soil field onto the faces created by a prior
     * addSoilVisualization() call and rewrites their "pressure_head"/"water_content"
     * primitive data, reusing the stored per-face sample points. The geometry is left
     * untouched, so for an animation this avoids deleting/recreating thousands of
     * primitives (and the associated GPU buffer churn) on every frame.
     * \param[in] UUIDs The primitive UUIDs returned by addSoilVisualization()
     * \param[in] mode Field to sample (WATER_CONTENT or PRESSURE_HEAD; TEXTURE is a no-op)
     */
    void updateSoilVisualization(const std::vector<uint> &UUIDs, SoilColorMode mode) const;

    // ========================================================================
    // Message control
    // ========================================================================

    //! Enable standard output messages
    void enableMessages();

    //! Disable standard output messages
    void disableMessages();

private:
    helios::Context *context_;

    // Domain parameters
    helios::vec3 origin_;
    helios::vec3 extent_;
    uint nx_, ny_, nz_;
    bool domain_initialized_;

    // Default soil properties
    SoilProperties default_props_;

    // Solver configuration
    SolverType solver_type_;
    SolverParameters solver_params_;
    //! Where the full 3D solver runs: AUTO (default) uses the GPU when isGPUAvailable() and the nonlinear method runs on it
    enum class GPUChoice { AUTO, ON, OFF };
    GPUChoice gpu_choice_ = GPUChoice::AUTO;
    bool root_uptake_enabled_;

    // Point sensors (repeated point queries cached over time)
    struct SoilSensor {
        helios::vec3 position;
        SoilMeasurement quantity;
        std::vector<float> values; // recorded measurements
        std::vector<float> times; // simulation time [s] at each record
    };
    std::vector<SoilSensor> sensors_;
    double sim_time_; // accumulated simulation time [s], advanced by run()
    uint last_step_count_; // adaptive sub-steps taken in the last run() call

    // Core components
    std::unique_ptr<SoilGrid> grid_;
    std::unique_ptr<SoilSolver3D> solver3d_; // full 3D solver (SOLVER_3D mode)
    std::unique_ptr<RootSoilCoupling> root_coupling_;

    // Full-3D mode helpers
    bool use3D() const {
        return solver_type_ == SolverType::SOLVER_3D;
    }
    //! Map SolverParameters onto a 3D solver's Params (the solver checks them before taking them)
    void applySolver3DParams(SoilSolver3D &solver, const SolverParameters &params, bool use_gpu) const;
    //! Whether the 3D solver runs on the GPU for these parameters and GPU choice
    static bool useGPU(const SolverParameters &params, GPUChoice choice);
    //! Cell containing a world position: column (i, j) and layer counted down from the surface. The domain is a closed
    //! box, so a point on its boundary belongs to the boundary cell. Returns false for a point outside the domain.
    bool locateCell(const helios::vec3 &position, uint &i, uint &j, uint &layer) const;
    //! As locateCell(), but raises an error naming the calling function for a point outside the domain
    void locateCellOrFail(const helios::vec3 &position, uint &i, uint &j, uint &layer, const std::string &function) const;
    //! Raise an error if the domain has already been set up (the grid can no longer change)
    void requireDomainNotSetUp(const std::string &function) const;
    //! Raise an error if the domain has not been set up yet
    void requireDomainSetUp(const std::string &function) const;
    double dt3d_; // persistent adaptive timestep for the 3D solver [s]
    float dtgrid_; // persistent adaptive timestep for the column grid (quasi-3D/1D) [s]

    // Message flag
    bool message_flag_;
};

#endif // SOILTRANSPORTMODEL_H
