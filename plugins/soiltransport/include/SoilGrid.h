/** \file "SoilGrid.h" Grid of soil columns with parallel solving and horizontal coupling.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SOILGRID_H
#define SOILGRID_H

#include "SoilColumn.h"
#include <vector>
#include <memory>

//! Grid of soil columns for quasi-3D simulation
/**
 * Manages a 2D grid of SoilColumn objects. Columns can be solved independently
 * in parallel, with optional periodic horizontal diffusion coupling for quasi-3D
 * simulation.
 *
 * Memory layout is layer-major for GPU coalescing:
 *   index = layer * n_columns + column
 * where column = j * nx + i for a 2D grid.
 */
class SoilGrid {
public:
    //! Constructor for uniform grid
    /**
     * \param[in] nx Number of columns in x direction
     * \param[in] ny Number of columns in y direction
     * \param[in] nz Number of vertical layers per column
     * \param[in] dx Grid spacing in x [m]
     * \param[in] dy Grid spacing in y [m]
     * \param[in] dz Vertical layer thickness [m]
     * \param[in] props Soil properties (uniform for all columns)
     */
    SoilGrid(uint nx, uint ny, uint nz, float dx, float dy, float dz, const SoilProperties &props);

    //! Destructor
    ~SoilGrid();

    // ========================================================================
    // Domain setup
    // ========================================================================

    //! Set domain origin in world coordinates
    /**
     * \param[in] origin_x x-coordinate of the domain origin [m]
     * \param[in] origin_y y-coordinate of the domain origin [m]
     * \param[in] origin_z z-coordinate of the domain origin, the top surface elevation [m]
     */
    void setOrigin(float origin_x, float origin_y, float origin_z);

    //! Set uniform initial condition for all columns
    /**
     * \param[in] h_init Initial pressure head [m]
     */
    void setInitialCondition(float h_init);

    //! Set initial condition from 3D array
    /**
     * \param[in] h_profile Pressure head values in layer-major order
     */
    void setInitialCondition(const std::vector<float> &h_profile);

    //! Set hydrostatic initial condition
    /**
     * \param[in] water_table_depth Depth to water table from surface [m]
     */
    void setInitialConditionHydrostatic(float water_table_depth);

    //! Set soil properties for a specific column
    /**
     * \param[in] i Column index in x
     * \param[in] j Column index in y
     * \param[in] props Soil properties
     */
    void setSoilProperties(uint i, uint j, const SoilProperties &props);

    //! Set uniform top boundary condition for all columns
    void setTopBoundary(BCType type, float value);

    //! Set top boundary for a specific column
    void setTopBoundary(uint i, uint j, BCType type, float value);

    //! Set uniform bottom boundary condition for all columns
    void setBottomBoundary(BCType type, float value);

    //! Set bottom boundary for a specific column
    void setBottomBoundary(uint i, uint j, BCType type, float value);

    // ========================================================================
    // Solver control
    // ========================================================================

    //! Set solver parameters for all columns
    void setSolverParameters(const SolverParameters &params);

    //! Enable horizontal coupling for quasi-3D simulation
    /**
     * \param[in] interval Couple every N vertical timesteps
     */
    void enableHorizontalCoupling(uint interval = 10);

    //! Disable horizontal coupling (pure 1D)
    void disableHorizontalCoupling();

    //! Set operator splitting type for quasi-3D
    void setSplittingType(SplittingType type);

    //! Solve all columns for one timestep
    /**
     * \param[in] dt Timestep [s]
     * \return Number of columns that converged
     */
    uint solve(float dt);

    //! Solve with quasi-3D (operator splitting)
    /**
     * \param[in] dt Timestep [s]
     * \return Number of columns that converged
     */
    uint solveQuasi3D(float dt);

    //! Advance the whole grid to target_time with adaptive timestepping
    /**
     * Takes trial split-steps (vertical + horizontal coupling) over the entire grid
     * and accepts/resizes/rejects each step as a unit, so every column stays on the
     * same clock (required for the operator-split horizontal coupling). The step size
     * is controlled by the per-step max relative water-content change across all
     * columns, and a step is rejected (and retried smaller) if any column fails to
     * converge. Assumes coupling every step (horizontal_coupling_interval == 1).
     * \param[in] target_time Total time to advance [s]
     * \param[in,out] dt Current/next timestep [s], updated for reuse across calls
     * \param[out] steps_taken Optional: number of accepted steps
     * \return Actual time advanced [s]
     */
    float solveAdaptive(float target_time, float &dt, uint *steps_taken = nullptr);

    // ========================================================================
    // State access
    // ========================================================================

    //! Get grid dimensions
    uint getNx() const {
        return nx_;
    }
    uint getNy() const {
        return ny_;
    }
    uint getNz() const {
        return nz_;
    }
    uint getNumColumns() const {
        return n_columns_;
    }

    //! Get grid spacing
    float getDx() const {
        return dx_;
    }
    float getDy() const {
        return dy_;
    }
    float getDz() const {
        return dz_;
    }

    //! Get a specific column
    SoilColumn &getColumn(uint i, uint j);
    const SoilColumn &getColumn(uint i, uint j) const;

    //! Get column by linear index
    SoilColumn &getColumn(uint col_idx);
    const SoilColumn &getColumn(uint col_idx) const;

    //! Get pressure head at a specific location
    /**
     * \param[in] i Column index in x
     * \param[in] j Column index in y
     * \param[in] k Layer index (0 = top)
     * \return Pressure head [m]
     */
    float getPressureHead(uint i, uint j, uint k) const;

    //! Get all pressure heads (layer-major order)
    std::vector<float> getAllPressureHeads() const;

    //! Get water content at a specific location
    float getWaterContent(uint i, uint j, uint k) const;

    //! Get total water storage in domain [m³]
    double getTotalWaterStorage() const;

    //! Get world position for column center
    /**
     * \param[in] i Column index in x
     * \param[in] j Column index in y
     * \param[out] x World x coordinate [m]
     * \param[out] y World y coordinate [m]
     */
    void getColumnPosition(uint i, uint j, float &x, float &y) const;

    //! Find column containing a world position
    /**
     * \param[in] x World x coordinate [m]
     * \param[in] y World y coordinate [m]
     * \param[out] i Column index in x
     * \param[out] j Column index in y
     * \return true if position is within domain
     */
    bool findColumn(float x, float y, uint &i, uint &j) const;

    //! Get pressure head at world position (with interpolation)
    /**
     * \param[in] x World x coordinate [m]
     * \param[in] y World y coordinate [m]
     * \param[in] z World z coordinate [m]
     * \return Pressure head [m]. A point outside the grid raises an error; one on its boundary reads the boundary cell.
     */
    float getPressureHeadAtPosition(float x, float y, float z) const;

    // ========================================================================
    // Root coupling
    // ========================================================================

    //! Set transpiration for a specific column
    void setTranspiration(uint i, uint j, float transpiration);

    //! Set root distribution for a specific column
    void setRootDistribution(uint i, uint j, const std::vector<float> &root_dist);

    // ========================================================================
    // Volumetric source/sink terms (e.g. drip emitters)
    // ========================================================================

    //! Add a volumetric source to a column's layer (flux into the layer) [m/s]
    /**
     * \param[in] i Column index in x
     * \param[in] j Column index in y
     * \param[in] layer Layer index (0 = top / surface)
     * \param[in] flux Source flux [m/s] (positive into the soil)
     */
    void addSource(uint i, uint j, uint layer, float flux);

    //! Clear all volumetric source terms in every column
    void clearSources();

private:
    // Grid dimensions
    uint nx_, ny_, nz_;
    uint n_columns_;
    float dx_, dy_, dz_;

    // Domain origin
    float origin_x_, origin_y_, origin_z_;

    // Soil columns
    std::vector<std::unique_ptr<SoilColumn>> columns_;

    // Solver parameters
    SolverParameters solver_params_;
    bool horizontal_coupling_enabled_;
    uint horizontal_coupling_interval_;
    uint step_count_;
    SplittingType splitting_type_;

    // Working arrays for horizontal diffusion
    std::vector<float> h_flat_; // Flattened pressure heads [nz * n_columns]
    std::vector<float> K_flat_; // Flattened conductivities (cached from columns)
    std::vector<float> C_flat_; // Flattened capacities (cached from columns)
    std::vector<float> theta_flat_; // Flattened water contents (cached from columns)
    std::vector<float> h_new_; // Updated values after diffusion

    // Internal methods
    uint columnIndex(uint i, uint j) const {
        return j * nx_ + i;
    }
    uint flatIndex(uint i, uint j, uint k) const {
        return k * n_columns_ + columnIndex(i, j);
    }

    void solveVerticalColumns(float dt);
    void solveHorizontalDiffusion(float dt);
    void flattenState();
    void unflattenState();
    float computeMaxHorizontalGradient() const;
};

#endif // SOILGRID_H
