/** \file "RootSoilCoupling.h" Root-soil interface for water uptake.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef ROOTSOILCOUPLING_H
#define ROOTSOILCOUPLING_H

#include "SoilUtils.h"
#include "SoilGrid.h"
#include "Context.h"
#include <vector>
#include <map>

class SoilSolver3D; // full-3D solver; root uptake is applied to it as a volumetric sink

//! Parameters for statistical root distribution (Vrugt et al. 2001)
struct RootDistributionParams {
    //! Plant position in world coordinates [m]
    float plant_x, plant_y, plant_z;
    //! Maximum rooting depth [m]
    float z_max;
    //! Maximum radial extent [m]
    float r_max;
    //! Depth of maximum root density [m]
    float z_star;
    //! Radial location of maximum density [m]
    float r_star;
    //! Depth shape parameter [-]
    float pz;
    //! Radial shape parameter [-]
    float pr;
    //! Total root length [m]
    float total_root_length;

    RootDistributionParams() : plant_x(0.0f), plant_y(0.0f), plant_z(0.0f), z_max(1.5f), r_max(1.0f), z_star(0.3f), r_star(0.0f), pz(1.0f), pr(1.0f), total_root_length(1000.0f) {
    }
};

//! Parameters for a Weibull root length distribution (1D depth, or 3D depth x radius)
/**
 * Root length density follows a Weibull profile with depth (and, in 3D, radius from the
 * plant base): RLD(x) is proportional to (k/lambda)(x/lambda)^(k-1) exp(-(x/lambda)^k).
 * The shape k controls the profile: k<1 concentrates roots near the origin (surface /
 * trunk), k=1 is exponential decay, k>1 gives a peak away from the origin. In 3D the
 * depth profile is multiplied by an analogous Weibull in radius about (plant_x, plant_y).
 * The distribution is normalized to fractions internally; total_root_length is retained
 * for reference / future flux scaling.
 */
struct WeibullRootParameters {
    //! Plant base position in world coordinates [m] (radial origin; used when radial=true)
    float plant_x, plant_y;
    //! Maximum rooting depth [m]
    float z_max;
    //! Depth Weibull shape k_z [-]
    float shape_z;
    //! Depth Weibull scale lambda_z [m]
    float scale_z;
    //! Include radial variation: true = 3D (depth x radius), false = 1D (depth only, horizontally uniform)
    bool radial;
    //! Maximum radial extent [m] (used when radial=true)
    float r_max;
    //! Radial Weibull shape k_r [-] (used when radial=true)
    float shape_r;
    //! Radial Weibull scale lambda_r [m] (used when radial=true)
    float scale_r;
    //! Total root length [m] (informational; uptake uses normalized fractions)
    float total_root_length;

    WeibullRootParameters() : plant_x(0.0f), plant_y(0.0f), z_max(1.5f), shape_z(1.5f), scale_z(0.3f), radial(false), r_max(1.0f), shape_r(1.0f), scale_r(0.5f), total_root_length(1000.0f) {
    }
};

//! Root-soil coupling interface for water uptake calculations
/**
 * Distributes each plant's transpiration over the soil cells of its root zone, described by a statistical root
 * distribution (Vrugt et al. 2001, or Weibull). Explicit root geometry from Helios primitives is not supported yet.
 */
class RootSoilCoupling {
public:
    //! Constructor for a column grid
    /**
     * \param[in] grid Pointer to the soil grid
     */
    explicit RootSoilCoupling(SoilGrid *grid);

    //! Constructor for the full-3D solver
    /**
     * \param[in] solver3d Pointer to the 3D solver (its grid defines the columns and layers)
     * \param[in] origin_x World x coordinate of the grid's minimum-x edge [m]
     * \param[in] origin_y World y coordinate of the grid's minimum-y edge [m]
     */
    RootSoilCoupling(SoilSolver3D *solver3d, float origin_x, float origin_y);

    //! Destructor
    ~RootSoilCoupling();

    // ========================================================================
    // Statistical root distributions
    // ========================================================================

    //! Add a plant with statistical root distribution
    /**
     * Raises an error if the distribution places no roots inside the soil domain.
     * \param[in] plant_id Unique identifier for the plant
     * \param[in] params Root distribution parameters
     */
    void addPlant(uint plant_id, const RootDistributionParams &params);

    //! Remove a plant
    void removePlant(uint plant_id);

    //! Update root distribution for a plant
    void updatePlantRoots(uint plant_id, const RootDistributionParams &params);

    //! Set a Weibull root length distribution for a plant (1D depth, or 3D depth x radius)
    /**
     * Populates the per-cell root length fractions used to partition transpiration into
     * root water uptake. Pair with a transpiration demand (computeRootUptake / the
     * front-end setTranspiration) to drive uptake. Raises an error if the distribution places no roots
     * inside the soil domain.
     * \param[in] plant_id Plant identifier
     * \param[in] params Weibull distribution parameters
     */
    void setWeibullDistribution(uint plant_id, const WeibullRootParameters &params);

    // ========================================================================
    // Explicit root primitives
    // ========================================================================

    //! Set root primitives for a plant from Helios UUIDs (not implemented yet)
    /**
     * Mapping explicit root geometry onto the soil grid is not implemented yet, so this always raises an error.
     * Describe the root zone with addPlant() or setWeibullDistribution() instead.
     * \param[in] plant_id Plant identifier
     * \param[in] root_UUIDs Vector of primitive UUIDs representing roots
     */
    void setRootPrimitives(uint plant_id, const std::vector<uint> &root_UUIDs);

    // ========================================================================
    // Water uptake
    // ========================================================================

    //! Set root radial conductivity
    /**
     * \param[in] L_r Root radial conductivity [m/(s·MPa)]
     */
    void setRootRadialConductivity(float L_r);

    //! Set a plant's transpiration demand and update the root water uptake of every plant in the soil
    /**
     * The sinks of all plants are summed, so plants whose roots share soil cells all take up water.
     * Raises an error for a plant without a root distribution.
     * \param[in] plant_id Plant identifier
     * \param[in] transpiration_demand The plant's transpiration flux per unit ground area of the soil domain [m/s]
     */
    void computeRootUptake(uint plant_id, float transpiration_demand);

    //! Recompute the root water uptake of every plant from the current soil heads and apply it to the soil
    void computeAllRootUptake();

    // ========================================================================
    // Output for plant hydraulics coupling
    // ========================================================================

    //! Get weighted soil water potential for a plant
    /**
     * Returns the root-zone soil water potential for PlantHydraulics coupling, weighted by root
     * density times the Feddes water-stress factor of each cell, i.e. the potential of the soil the
     * roots are actually drawing from. Falls back to root-density weighting if no cell is active.
     * Raises an error for a plant without a root distribution.
     * \param[in] plant_id Plant identifier
     * \return Weighted soil water potential [m]
     */
    float getSoilWaterPotential(uint plant_id) const;

    //! Get cumulative root water uptake for a plant [m]
    /**
     * Raises an error for a plant without a root distribution.
     */
    float getCumulativeRootUptake(uint plant_id) const;

    //! Get the root water uptake flux currently applied for a plant [m/s per unit ground area]
    /**
     * Equal to the transpiration demand unless the root zone is too dry to supply it (see
     * setWaterStressParameters()); 0 before a transpiration demand is set. Raises an error for a plant without
     * a root distribution.
     */
    float getRootUptakeFlux(uint plant_id) const;

    //! Set the root water-stress (Feddes) and compensation (Jarvis) parameters
    /**
     * Each root cell's uptake is scaled by alpha(h): 1 for h >= h_onset, falling linearly to 0 at
     * h_wilting. Uptake is redistributed toward unstressed roots (compensation), so the full demand is
     * met while the root-weighted mean alpha is at least omega_c; below that, uptake is reduced in
     * proportion to (mean alpha) / omega_c.
     * \param[in] h_onset Pressure head where water stress begins [m] (default -4)
     * \param[in] h_wilting Pressure head where uptake stops [m] (default -150)
     * \param[in] omega_c Critical root-zone stress index for compensation, 0-1 (default 0.5; 1 = no compensation)
     */
    void setWaterStressParameters(float h_onset, float h_wilting, float omega_c);

    //! Add one timestep of uptake to each plant's cumulative total
    /**
     * \param[in] dt Timestep that was just simulated [s]
     */
    void accumulateUptake(float dt);

private:
    SoilGrid *grid_; // column grid (null when coupled to the 3D solver only)
    // Geometry of the soil grid: columns (nx_ x ny_), layers (nz_, layer 0 at the surface) and spacing [m]
    uint nx_ = 0, ny_ = 0, nz_ = 0;
    float dx_ = 0.0f, dy_ = 0.0f, dz_ = 0.0f;
    float origin_x_ = 0.0f, origin_y_ = 0.0f; // minimum-x and minimum-y edges [m] (3D solver only; the grid has its own)
    SoilSolver3D *solver3d_; // when set, root uptake is applied to the full-3D solver

    // Root radial conductivity
    float L_r_; // [m/(s·MPa)]

    // Statistical root distributions
    std::map<uint, RootDistributionParams> plant_root_params_;

    // Weibull root length distributions
    std::map<uint, WeibullRootParameters> plant_weibull_params_;

    // Root density maps (computed from params or primitives)
    // Maps plant_id -> [column_idx -> normalized_root_fraction]
    std::map<uint, std::map<uint, std::vector<float>>> root_density_maps_;

    // Transpiration per plant
    std::map<uint, float> plant_transpiration_;

    // Cumulative uptake per plant
    std::map<uint, float> cumulative_uptake_;

    // Root water stress (Feddes) and compensation (Jarvis)
    float h_onset_ = -4.0f; // [m]
    float h_wilting_ = -150.0f; // [m]
    float omega_c_ = 0.5f; // [-]
    std::map<uint, float> uptake_flux_; // applied uptake per plant [m/s per unit ground area]
    // Each plant's sinks: plant -> column -> per-layer uptake [m/s per unit column ground area] (layer 0 at the surface)
    std::map<uint, std::map<uint, std::vector<double>>> plant_sinks_;
    std::map<uint, std::set<uint>> column_plants_; // column -> plants with a sink in it

    //! Feddes water-stress factor for a pressure head [m]
    float stressFactor(float h) const;
    //! Current pressure head of a root cell (column layer k = 0 at the surface)
    float cellHead(uint col_idx, uint k) const;
    //! Recompute a plant's stress-weighted sinks from its demand and the current heads; adds the columns whose sink changed to affected
    void refreshPlantSinks(uint plant_id, std::set<uint> &affected);
    //! Remove a plant's sinks; adds the columns it had sinks in to affected
    void clearPlantSinks(uint plant_id, std::set<uint> &affected);
    //! Apply the summed sinks of every plant to the given columns of the 3D solver or the column grid
    void applyColumns(const std::set<uint> &columns);
    //! Store a plant's new root density map, and move its sinks if it has a transpiration demand
    void setPlantRoots(uint plant_id, std::map<uint, std::vector<float>> &&density);
    //! Raise an error if the plant has no root distribution
    void requireRoots(uint plant_id, const std::string &function) const;
    //! Normalize a root density map to sum to 1; raises an error if it places no roots inside the domain
    static void normalizeDensity(uint plant_id, std::map<uint, std::vector<float>> &density, double total_root_density);

    // Root density maps from the distribution parameters (normalized; raise an error if no roots are in the domain)
    std::map<uint, std::vector<float>> densityFromParams(uint plant_id, const RootDistributionParams &params) const;
    std::map<uint, std::vector<float>> densityFromWeibull(uint plant_id, const WeibullRootParameters &params) const;
    float rootDensityFunction(float r, float z, const RootDistributionParams &params) const;
    //! Weibull probability density (k/lambda)(x/lambda)^(k-1) exp(-(x/lambda)^k); 0 for x<0 or invalid params
    static float weibullPDF(float x, float shape, float scale);
    //! Weibull survival function exp(-(x/lambda)^k), one minus the cumulative distribution; 1 for x <= 0
    static double weibullSurvival(double x, double shape, double scale);
    //! World x-y position of a column's centre [m]
    void columnCenter(uint i, uint j, float &x, float &y) const;
    //! Depth of a layer's centre below the surface [m] (layer 0 at the surface)
    float layerDepth(uint col_idx, uint k) const;
};

#endif // ROOTSOILCOUPLING_H
