/** \file "SoilTransportModel.cpp" Implementation of main plugin class.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilTransportModel.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace helios;

SoilTransportModel::SoilTransportModel(helios::Context *context) :
    context_(context), origin_(make_vec3(0, 0, 0)), extent_(make_vec3(1, 1, 1)), nx_(1), ny_(1), nz_(20), domain_initialized_(false), solver_type_(SolverType::SOLVER_1D), root_uptake_enabled_(false), message_flag_(true), dt3d_(0.0), dtgrid_(0.0f),
    sim_time_(0.0), last_step_count_(0) {

    // Set default soil properties (loam)
    default_props_ = getSoilPropertiesFromTexture("loam");
}

SoilTransportModel::~SoilTransportModel() = default;

// ============================================================================
// Domain setup
// ============================================================================

void SoilTransportModel::requireDomainNotSetUp(const std::string &function) const {
    if (domain_initialized_) {
        helios_runtime_error("ERROR (SoilTransportModel::" + function +
                             "): the soil domain has already been set up, by initializeDomain() or by the first call that needed it "
                             "(e.g. setInitialCondition(), a boundary condition, a root distribution or a sensor). Changing the grid or switching to or from SOLVER_3D "
                             "would discard that setup, so call defineSoilDomain(), setGridResolution(), setColumnResolution(), setColumnGrid() and setSolverType() first.");
    }
}

void SoilTransportModel::requireDomainSetUp(const std::string &function) const {
    if (!domain_initialized_) {
        helios_runtime_error("ERROR (SoilTransportModel::" + function +
                             "): the soil domain has not been set up yet. Call initializeDomain(), or set an initial condition "
                             "(e.g. setInitialCondition()), first.");
    }
}

void SoilTransportModel::defineSoilDomain(const vec3 &origin, const vec3 &extent) {
    requireDomainNotSetUp("defineSoilDomain");
    if (!(extent.x > 0.0f) || !(extent.y > 0.0f) || !(extent.z > 0.0f)) {
        helios_runtime_error("ERROR (SoilTransportModel::defineSoilDomain): every component of the domain extent must be positive (got " + std::to_string(extent.x) + ", " + std::to_string(extent.y) + ", " + std::to_string(extent.z) + " m).");
    }
    origin_ = origin;
    extent_ = extent;
}

void SoilTransportModel::setGridResolution(uint nx, uint ny, uint nz) {
    requireDomainNotSetUp("setGridResolution");
    if (nx < 1 || ny < 1 || nz < 1) {
        helios_runtime_error("ERROR (SoilTransportModel::setGridResolution): the grid needs at least 1 cell in each direction (got " + std::to_string(nx) + " x " + std::to_string(ny) + " x " + std::to_string(nz) + ").");
    }
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
}

void SoilTransportModel::setColumnResolution(uint nz) {
    requireDomainNotSetUp("setColumnResolution");
    if (nz < 1) {
        helios_runtime_error("ERROR (SoilTransportModel::setColumnResolution): a column needs at least 1 layer.");
    }
    nz_ = nz;
}

void SoilTransportModel::setColumnGrid(uint nx, uint ny) {
    requireDomainNotSetUp("setColumnGrid");
    if (nx < 1 || ny < 1) {
        helios_runtime_error("ERROR (SoilTransportModel::setColumnGrid): the column grid needs at least 1 column in each direction (got " + std::to_string(nx) + " x " + std::to_string(ny) + ").");
    }
    nx_ = nx;
    ny_ = ny;
}

void SoilTransportModel::initializeDomain() {
    if (domain_initialized_) {
        return;
    }

    if (!use3D() && nz_ < 2) {
        helios_runtime_error("ERROR (SoilTransportModel::initializeDomain): the column solvers (SOLVER_1D, SOLVER_QUASI_3D) need at least 2 layers, but the grid has " + std::to_string(nz_) +
                             ". Use more layers (setGridResolution() or setColumnResolution()), or SOLVER_3D.");
    }

    // Compute grid spacing
    float dx = extent_.x / static_cast<float>(nx_);
    float dy = extent_.y / static_cast<float>(ny_);
    float dz = extent_.z / static_cast<float>(nz_);

    // Build everything first and commit only once nothing can fail, so an error leaves the model without a domain
    // (and without a half-built solver from a previous mode)
    std::unique_ptr<SoilSolver3D> solver3d;
    std::unique_ptr<SoilGrid> grid;
    std::unique_ptr<RootSoilCoupling> root_coupling;
    if (use3D()) {
        // Full 3D solver. It uses z elevation upward with k = 0 at the bottom, so the front-end "top" maps to ZMAX.
        // Root uptake is applied to it as a volumetric sink.
        solver3d = std::make_unique<SoilSolver3D>(nx_, ny_, nz_, dx, dy, dz, default_props_);
        applySolver3DParams(*solver3d, solver_params_, useGPU(solver_params_, gpu_choice_));
        root_coupling = std::make_unique<RootSoilCoupling>(solver3d.get(), origin_.x, origin_.y);
    } else {
        // Column grid (1D / quasi-3D)
        grid = std::make_unique<SoilGrid>(nx_, ny_, nz_, dx, dy, dz, default_props_);
        grid->setOrigin(origin_.x, origin_.y, origin_.z);
        grid->setSolverParameters(solver_params_);
        // Enable horizontal coupling here if quasi-3D was selected before the domain
        // existed (setSolverType's grid_ guard would otherwise have skipped it, leaving
        // the grid in pure-1D mode with no lateral flow).
        if (solver_type_ == SolverType::SOLVER_QUASI_3D) {
            grid->enableHorizontalCoupling(solver_params_.horizontal_coupling_interval);
        }
        root_coupling = std::make_unique<RootSoilCoupling>(grid.get());
    }
    solver3d_ = std::move(solver3d);
    grid_ = std::move(grid);
    root_coupling_ = std::move(root_coupling);
    domain_initialized_ = true;

    if (message_flag_) {
        std::cout << "SoilTransportModel: Initialized domain with " << nx_ << "x" << ny_ << "x" << nz_ << " grid" << (use3D() ? " (full 3D solver)" : "") << std::endl;
    }
}

// ============================================================================
// Soil properties
// ============================================================================

void SoilTransportModel::setSoilProperties(const SoilProperties &props) {
    validateSoilProperties(props, "SoilTransportModel::setSoilProperties");
    default_props_ = props;
    if (grid_) {
        for (uint j = 0; j < ny_; j++) {
            for (uint i = 0; i < nx_; i++) {
                grid_->setSoilProperties(i, j, props);
            }
        }
    }
    if (solver3d_) {
        for (uint k = 0; k < nz_; k++)
            for (uint j = 0; j < ny_; j++)
                for (uint i = 0; i < nx_; i++)
                    solver3d_->setSoilProperties(i, j, k, props);
    }
}

void SoilTransportModel::setSoilPropertiesFromTexture(const std::string &texture) {
    setSoilPropertiesFromTexture(texture, SoilModel::VAN_GENUCHTEN);
}

void SoilTransportModel::setSoilPropertiesFromTexture(const std::string &texture, SoilModel model, bool match_standard_curves) {
    default_props_ = getSoilPropertiesFromTexture(texture, model, match_standard_curves); // raises an error for an unknown texture

    if (grid_) {
        for (uint j = 0; j < ny_; j++) {
            for (uint i = 0; i < nx_; i++) {
                grid_->setSoilProperties(i, j, default_props_);
            }
        }
    }
    if (solver3d_) {
        for (uint k = 0; k < nz_; k++)
            for (uint j = 0; j < ny_; j++)
                for (uint i = 0; i < nx_; i++)
                    solver3d_->setSoilProperties(i, j, k, default_props_);
    }
}

void SoilTransportModel::setSoilProperties(uint i, uint j, const SoilProperties &props) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (i >= nx_ || j >= ny_) {
        helios_runtime_error("ERROR (SoilTransportModel::setSoilProperties): column (" + std::to_string(i) + ", " + std::to_string(j) + ") is outside the " + std::to_string(nx_) + " x " + std::to_string(ny_) + " column grid.");
    }
    validateSoilProperties(props, "SoilTransportModel::setSoilProperties");
    if (grid_)
        grid_->setSoilProperties(i, j, props);
    if (solver3d_) {
        for (uint k = 0; k < nz_; k++)
            solver3d_->setSoilProperties(i, j, k, props);
    }
}

// ============================================================================
// Initial and boundary conditions
// ============================================================================

void SoilTransportModel::setInitialCondition(float h_init) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (grid_)
        grid_->setInitialCondition(h_init);
    if (solver3d_)
        solver3d_->setInitialCondition(static_cast<double>(h_init));
}

void SoilTransportModel::setInitialConditionHydrostatic(float water_table_depth) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (grid_)
        grid_->setInitialConditionHydrostatic(water_table_depth);
    if (solver3d_) {
        // Hydrostatic equilibrium: h + z = z_wt (h = 0 at the water table). The
        // 3D solver measures elevation z from the bottom (k = 0), and the water
        // table sits at elevation (extent.z - water_table_depth).
        const double dz = extent_.z / static_cast<double>(nz_);
        const double z_wt = extent_.z - static_cast<double>(water_table_depth);
        std::vector<double> h0(solver3d_->numCells());
        for (uint k = 0; k < nz_; k++) {
            double h = z_wt - (k + 0.5) * dz;
            for (uint j = 0; j < ny_; j++)
                for (uint i = 0; i < nx_; i++)
                    h0[solver3d_->index(i, j, k)] = h;
        }
        solver3d_->setInitialCondition(h0);
    }
}

void SoilTransportModel::setTopBoundaryFlux(float flux) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (grid_)
        grid_->setTopBoundary(BCType::NEUMANN, flux);
    // Front-end flux is positive for infiltration (into soil); the 3D solver's
    // Neumann value is positive into the domain, so the sign carries over at ZMAX.
    if (solver3d_)
        solver3d_->setBoundary(Face3D::ZMAX, BCType::NEUMANN, static_cast<double>(flux));
}

float SoilTransportModel::setSurfaceEvaporation(float potential_rate, float aerodynamic_resistance) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (!solver3d_) {
        helios_runtime_error("ERROR (SoilTransportModel::setSurfaceEvaporation): surface evaporation is only supported by the full 3D solver. Call setSolverType(SolverType::SOLVER_3D) "
                             "before setting up the domain.");
    }
    // For each surface cell (k = nz-1 is the top), gate the potential rate by the
    // dry-surface-layer factor of its current water content, then apply E_act as a
    // volumetric sink [1/s] = flux [m/s] / cell thickness.
    const float dx = extent_.x / static_cast<float>(nx_);
    const float dy = extent_.y / static_cast<float>(ny_);
    const float dz = extent_.z / static_cast<float>(nz_);
    double total = 0.0; // total actual evaporation volume rate [m^3/s]
    for (uint j = 0; j < ny_; j++) {
        for (uint i = 0; i < nx_; i++) {
            const float theta = static_cast<float>(solver3d_->getWaterContent(i, j, nz_ - 1));
            const float beta = surfaceEvaporationFactor(theta, solver3d_->getSoilProperties(i, j, nz_ - 1), aerodynamic_resistance);
            const float e_act = beta * potential_rate; // actual evaporation flux [m/s]
            solver3d_->setEvaporation(i, j, nz_ - 1, static_cast<double>(e_act) / dz);
            total += static_cast<double>(e_act) * dx * dy;
        }
    }
    return static_cast<float>(total);
}

void SoilTransportModel::setTopBoundaryPressure(float h) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (grid_)
        grid_->setTopBoundary(BCType::DIRICHLET, h);
    if (solver3d_)
        solver3d_->setBoundary(Face3D::ZMAX, BCType::DIRICHLET, static_cast<double>(h));
}

void SoilTransportModel::setBottomBoundaryFreeDrainage() {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (grid_)
        grid_->setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    if (solver3d_)
        solver3d_->setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE);
}

void SoilTransportModel::setBottomBoundaryPressure(float h) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (grid_)
        grid_->setBottomBoundary(BCType::DIRICHLET, h);
    if (solver3d_)
        solver3d_->setBoundary(Face3D::ZMIN, BCType::DIRICHLET, static_cast<double>(h));
}

void SoilTransportModel::setBottomBoundaryFlux(float flux) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (grid_)
        grid_->setBottomBoundary(BCType::NEUMANN, flux);
    // Front-end bottom flux is positive for outflow; the 3D solver's Neumann
    // value is positive into the domain, so the sign flips at ZMIN.
    if (solver3d_)
        solver3d_->setBoundary(Face3D::ZMIN, BCType::NEUMANN, -static_cast<double>(flux));
}

void SoilTransportModel::addSurfaceEmitter(float x, float y, float flow_rate, float radius) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    if (!std::isfinite(flow_rate)) {
        helios_runtime_error("ERROR (SoilTransportModel::addSurfaceEmitter): the emitter flow rate must be finite (got " + std::to_string(flow_rate) + " m^3/s).");
    }
    if (!(radius >= 0.0f) || !std::isfinite(radius)) {
        helios_runtime_error("ERROR (SoilTransportModel::addSurfaceEmitter): the wetting radius must be a finite value >= 0 (got " + std::to_string(radius) + " m).");
    }
    const float dx = extent_.x / static_cast<float>(nx_);
    const float dy = extent_.y / static_cast<float>(ny_);
    uint emitter_i, emitter_j, surface_layer;
    if (!locateCell(make_vec3(x, y, origin_.z), emitter_i, emitter_j, surface_layer)) {
        helios_runtime_error("ERROR (SoilTransportModel::addSurfaceEmitter): the emitter at (" + std::to_string(x) + ", " + std::to_string(y) + ") is outside the soil domain.");
    }

    if (solver3d_) {
        // Full 3D solver: spread the discharge over the surface cells whose centres fall within
        // `radius` of (x,y), so the per-cell volumetric source rate [1/s] = (flow/n_cells)/cell_vol
        // stays ~flow/(pi r^2 dz) -- independent of horizontal resolution. A single-cell point
        // source instead scales as 1/(dx dy), exploding as the grid refines and collapsing dt.
        const float dz = extent_.z / static_cast<float>(nz_);
        const int ci = static_cast<int>(emitter_i);
        const int cj = static_cast<int>(emitter_j);
        const float r2 = radius * radius;
        const int ri = std::max(0, static_cast<int>(std::ceil(radius / dx)));
        const int rj = std::max(0, static_cast<int>(std::ceil(radius / dy)));
        std::vector<std::pair<int, int>> cells;
        for (int di = -ri; di <= ri; di++) {
            for (int dj = -rj; dj <= rj; dj++) {
                const int i = ci + di, j = cj + dj;
                if (i < 0 || j < 0 || i >= static_cast<int>(nx_) || j >= static_cast<int>(ny_))
                    continue;
                const float cx = origin_.x + (static_cast<float>(i) + 0.5f) * dx;
                const float cy = origin_.y + (static_cast<float>(j) + 0.5f) * dy;
                if ((cx - x) * (cx - x) + (cy - y) * (cy - y) <= r2)
                    cells.emplace_back(i, j);
            }
        }
        if (cells.empty())
            cells.emplace_back(ci, cj); // radius smaller than a cell: fall back to the point
        const double rate = static_cast<double>(flow_rate) / (static_cast<double>(cells.size()) * dx * dy * dz);
        for (const auto &c: cells)
            solver3d_->addSource(static_cast<uint>(c.first), static_cast<uint>(c.second), nz_ - 1, rate);
    } else {
        // Column grid (1D / quasi-3D): inject into the surface layer (layer 0) of the
        // containing column as a flux [m/s] = discharge / column footprint area.
        // Accumulates, so multiple emitters in one column add up.
        const float flux = flow_rate / (dx * dy);
        grid_->addSource(emitter_i, emitter_j, 0, flux);
    }
}

void SoilTransportModel::clearSurfaceEmitters() {
    if (solver3d_)
        solver3d_->clearSources();
    if (grid_)
        grid_->clearSources();
}

// ============================================================================
// Root coupling
// ============================================================================

void SoilTransportModel::enableRootUptake() {
    root_uptake_enabled_ = true;
}

void SoilTransportModel::setRootPrimitives(uint plant_id, const std::vector<uint> &root_UUIDs) {
    helios::helios_runtime_error("ERROR (SoilTransportModel::setRootPrimitives): mapping explicit root primitives onto the soil grid is not implemented yet "
                                 "(plant " +
                                 std::to_string(plant_id) + ", " + std::to_string(root_UUIDs.size()) + " primitives). Describe the plant's root zone with setRootDistribution() or setRootDistributionWeibull() instead.");
}

void SoilTransportModel::setRootDistribution(uint plant_id, const RootDistributionParams &params) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    root_coupling_->addPlant(plant_id, params);
    root_uptake_enabled_ = true;
}

void SoilTransportModel::setRootDistributionWeibull(uint plant_id, const WeibullRootParameters &params) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    root_coupling_->setWeibullDistribution(plant_id, params);
    root_uptake_enabled_ = true;
}

void SoilTransportModel::setRootRadialConductivity(float L_r) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    root_coupling_->setRootRadialConductivity(L_r);
}

void SoilTransportModel::setTranspiration(uint plant_id, float transpiration) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    root_coupling_->computeRootUptake(plant_id, transpiration);
}

// ============================================================================
// Solver configuration
// ============================================================================

void SoilTransportModel::setSolverType(SolverType type) {
    // Switching into or out of full-3D mode changes which solver backs the domain
    if ((type == SolverType::SOLVER_3D) != use3D()) {
        requireDomainNotSetUp("setSolverType");
    }
    if (type == SolverType::SOLVER_3D && !solver_params_.enable_gravity) {
        helios_runtime_error("ERROR (SoilTransportModel::setSolverType): SolverParameters::enable_gravity = false is supported by the column solvers only, not SOLVER_3D.");
    }
    solver_type_ = type;
    if (grid_) {
        if (type == SolverType::SOLVER_QUASI_3D) {
            grid_->enableHorizontalCoupling(solver_params_.horizontal_coupling_interval);
        } else {
            grid_->disableHorizontalCoupling();
        }
    }
}

void SoilTransportModel::enableHorizontalCoupling(uint interval) {
    if (interval < 1) {
        helios_runtime_error("ERROR (SoilTransportModel::enableHorizontalCoupling): the coupling interval must be at least 1 vertical step.");
    }
    if (use3D()) {
        requireDomainNotSetUp("enableHorizontalCoupling"); // switches to SOLVER_QUASI_3D
    }
    solver_params_.horizontal_coupling_interval = interval;
    solver_type_ = SolverType::SOLVER_QUASI_3D;
    if (grid_) {
        grid_->enableHorizontalCoupling(interval);
    }
}

void SoilTransportModel::setSolverParameters(const SolverParameters &params) {
    // Validate everything before storing anything, so a rejected parameter set leaves the model unchanged
    if (params.horizontal_coupling_interval < 1) {
        helios_runtime_error("ERROR (SoilTransportModel::setSolverParameters): SolverParameters::horizontal_coupling_interval must be at least 1 vertical step.");
    }
    if (!(params.dt_min > 0.0f) || !(params.dt_max >= params.dt_min)) {
        helios_runtime_error("ERROR (SoilTransportModel::setSolverParameters): need 0 < dt_min <= dt_max (got dt_min = " + std::to_string(params.dt_min) + " s, dt_max = " + std::to_string(params.dt_max) + " s).");
    }
    if (use3D() && !params.enable_gravity) {
        helios_runtime_error("ERROR (SoilTransportModel::setSolverParameters): SolverParameters::enable_gravity = false is supported by the column solvers only, not SOLVER_3D.");
    }
    if (use3D() && gpu_choice_ == GPUChoice::ON && params.nonlinear_method == NonlinearMethod::M_SCHEME) {
        helios_runtime_error("ERROR (SoilTransportModel::setSolverParameters): NonlinearMethod::M_SCHEME runs on the CPU only, but enableGPU() asked for the GPU. Call disableGPU() first.");
    }
    if (solver3d_) {
        applySolver3DParams(*solver3d_, params, useGPU(params, gpu_choice_)); // the solver checks its parameters before taking them
    }
    if (grid_) {
        grid_->setSolverParameters(params);
    }
    solver_params_ = params;
}

bool SoilTransportModel::useGPU(const SolverParameters &params, GPUChoice choice) {
    if (choice != GPUChoice::AUTO) {
        return choice == GPUChoice::ON;
    }
    // Automatic: the GPU when one is available and the nonlinear method runs on it (M_SCHEME is CPU only). The method is
    // checked first so the GPU is probed only when it could be used.
    return params.nonlinear_method != NonlinearMethod::M_SCHEME && isGPUAvailable();
}

void SoilTransportModel::applySolver3DParams(SoilSolver3D &solver, const SolverParameters &solver_params, bool use_gpu) const {
    if (!solver_params.enable_gravity)
        helios::helios_runtime_error("ERROR (SoilTransportModel::setSolverParameters): SolverParameters::enable_gravity = false is supported by the column solvers only, not SOLVER_3D.");
    if (use_gpu && solver_params.nonlinear_method == NonlinearMethod::M_SCHEME)
        helios::helios_runtime_error("ERROR (SoilTransportModel::setSolverParameters): NonlinearMethod::M_SCHEME runs on the CPU only, but enableGPU() asked for the GPU. Call "
                                     "disableGPU() first.");
    SoilSolver3D::Params p = solver.getParams();
    p.picard_max = solver_params.max_picard_iterations;
    p.picard_tol = solver_params.picard_tolerance;
    p.C_min = solver_params.C_regularization;
    p.K_min = solver_params.K_min; // additive conductivity floor (eases dry wetting fronts)
    p.dt_target_rel_change = solver_params.dt_target_rel_change;
    p.dt_safety = solver_params.dt_safety;
    p.dt_growth_max = solver_params.dt_growth_max;
    p.dt_reduction_min = solver_params.dt_reduction_min;
    p.dt_min = solver_params.dt_min;
    p.dt_max = solver_params.dt_max;
    {
        const int kw = static_cast<int>(solver_params.K_weighting);
        p.K_weighting = (kw < 0) ? 1 : kw; // KWeighting::DEFAULT -> arithmetic
    }
    p.anderson_m = solver_params.anderson_depth; // Anderson acceleration depth (0 = plain Picard)
    p.nonlinear_method = static_cast<SoilSolver3D::NonlinearMethod>(solver_params.nonlinear_method); // see NonlinearMethod (default NEWTON_ANALYTIC)
    p.use_gpu = use_gpu; // run the 3D solve on the GPU (CUDA builds)
    p.num_threads = solver_params.num_threads; // 0 = automatic (one thread per physical core)
    p.substep_on_failure = solver_params.substep_on_failure;
    p.bdf2 = solver_params.bdf2; // backward Euler unless BDF2 is requested (see SolverParameters::bdf2)
    solver.setParams(p);
}

bool SoilTransportModel::locateCell(const vec3 &position, uint &i, uint &j, uint &layer) const {
    // Fractional cell coordinates. origin_.z is the top surface, and layers are counted down from it.
    const float fx = (position.x - origin_.x) / extent_.x * static_cast<float>(nx_);
    const float fy = (position.y - origin_.y) / extent_.y * static_cast<float>(ny_);
    const float fz = (origin_.z - position.z) / extent_.z * static_cast<float>(nz_);
    // The domain is a closed box: a point on its boundary (e.g. on the soil surface) belongs to the boundary cell.
    // The tolerance (in cells) absorbs the round-off of a point placed exactly on the boundary.
    const float tolerance = 1e-4f;
    auto inside = [&](float f, uint n) { return f >= -tolerance && f <= static_cast<float>(n) + tolerance; };
    if (!inside(fx, nx_) || !inside(fy, ny_) || !inside(fz, nz_)) {
        return false;
    }
    auto cellIndex = [](float f, uint n) { return std::min(static_cast<uint>(std::max(f, 0.0f)), n - 1); };
    i = cellIndex(fx, nx_);
    j = cellIndex(fy, ny_);
    layer = cellIndex(fz, nz_);
    return true;
}

void SoilTransportModel::locateCellOrFail(const vec3 &position, uint &i, uint &j, uint &layer, const std::string &function) const {
    if (!locateCell(position, i, j, layer)) {
        helios_runtime_error("ERROR (SoilTransportModel::" + function + "): the point (" + std::to_string(position.x) + ", " + std::to_string(position.y) + ", " + std::to_string(position.z) + ") is outside the soil domain, which spans x = [" +
                             std::to_string(origin_.x) + ", " + std::to_string(origin_.x + extent_.x) + "], y = [" + std::to_string(origin_.y) + ", " + std::to_string(origin_.y + extent_.y) + "] and z = [" + std::to_string(origin_.z - extent_.z) +
                             ", " + std::to_string(origin_.z) + "] (z = " + std::to_string(origin_.z) + " is the surface).");
    }
}

bool SoilTransportModel::isGPUAvailable() {
    return SoilSolver3D::isGPUAvailable();
}

void SoilTransportModel::enableGPU() {
#ifdef HELIOS_CUDA_AVAILABLE
    if (!isGPUAvailable()) {
        helios_runtime_error("ERROR (SoilTransportModel::enableGPU): no usable CUDA device was found (or HELIOS_NO_GPU is set). Check that an NVIDIA GPU and its driver are installed, "
                             "or run the 3D solver on the CPU by not calling enableGPU().");
    }
    if (use3D() && solver_params_.nonlinear_method == NonlinearMethod::M_SCHEME) {
        helios_runtime_error("ERROR (SoilTransportModel::enableGPU): the nonlinear method is NonlinearMethod::M_SCHEME, which runs on the CPU only.");
    }
    if (solver3d_)
        applySolver3DParams(*solver3d_, solver_params_, true);
    gpu_choice_ = GPUChoice::ON;
    if (message_flag_) {
        std::cout << "SoilTransportModel: GPU acceleration enabled" << std::endl;
    }
#else
    helios_runtime_error("ERROR (SoilTransportModel::enableGPU): this build of Helios was compiled without CUDA, so the soil solver can only run on the CPU. "
                         "Install the CUDA toolkit and rebuild to use the GPU.");
#endif
}

void SoilTransportModel::disableGPU() {
    if (solver3d_)
        applySolver3DParams(*solver3d_, solver_params_, false);
    gpu_choice_ = GPUChoice::OFF;
}

bool SoilTransportModel::isGPUEnabled() const {
    return useGPU(solver_params_, gpu_choice_);
}

// ============================================================================
// Run simulation
// ============================================================================

void SoilTransportModel::run(float dt) {
    if (!(dt > 0.0f)) {
        helios_runtime_error("ERROR (SoilTransportModel::run): the timestep must be positive (got " + std::to_string(dt) + " s).");
    }
    if (!domain_initialized_) {
        initializeDomain();
    }
    root_coupling_->computeAllRootUptake(); // refresh the stress-weighted uptake from the current heads

    // An error raised part-way through a step (e.g. quasi-3D saturation, or a CUDA failure) leaves the model as it was
    // before this call: the 3D solver restores itself, and the column grid is restored from this snapshot.
    std::vector<std::vector<float>> heads_at_start;
    if (grid_) {
        heads_at_start.resize(grid_->getNumColumns());
        for (uint c = 0; c < grid_->getNumColumns(); c++)
            heads_at_start[c] = grid_->getColumn(c).getPressureHeads();
    }
    // Time actually advanced. The clock and the cumulative root uptake follow it, so a step that could not be completed
    // does not count time the soil never advanced.
    double advanced = 0.0;
    try {
        // The full 3D solver runs on the CPU or GPU per SoilSolver3D::Params::use_gpu
        // (set in applySolver3DParams), so it handles its own backend dispatch.
        if (use3D()) {
            if (solver_params_.adaptive_dt) {
                // Sub-step adaptively to advance a total of dt; dt3d_ persists the
                // controller's chosen step size across run() calls.
                if (dt3d_ <= 0.0)
                    dt3d_ = std::min(static_cast<double>(dt), static_cast<double>(solver_params_.dt_max));
                uint steps = 0;
                advanced = solver3d_->solveAdaptive(static_cast<double>(dt), dt3d_, &steps);
                last_step_count_ = steps;
            } else {
                // A failed step leaves the 3D solver's state unchanged
                advanced = solver3d_->solve(static_cast<double>(dt)) > 0 ? static_cast<double>(dt) : 0.0;
                last_step_count_ = 1;
            }
        } else if (solver_params_.adaptive_dt) {
            // Column grid (1D / quasi-3D) always runs on the CPU; only the 3D solver has a GPU backend.
            // Grid-level adaptive stepping: the whole grid advances together (required by
            // the operator-split horizontal coupling). dtgrid_ persists the controller's
            // chosen step across run() calls.
            if (dtgrid_ <= 0.0f)
                dtgrid_ = std::min(dt, solver_params_.dt_max);
            uint steps = 0;
            advanced = grid_->solveAdaptive(dt, dtgrid_, &steps);
            last_step_count_ = steps;
        } else {
            // Fixed stepping: sub-step into vertical steps of at most dt_max so we don't
            // advance the whole interval in one coarse step, and so the quasi-3D horizontal
            // coupling (every horizontal_coupling_interval vertical steps) fires within a call.
            // A step in which any column fails is undone for the whole grid, so the columns stay at one time.
            const float dt_sub = std::min(dt, std::max(solver_params_.dt_max, 1e-6f));
            const uint n_columns = grid_->getNumColumns();
            std::vector<std::vector<float>> heads_before(n_columns);
            float t = 0.0f;
            uint steps = 0;
            while (t < dt) {
                const float step = std::min(dt_sub, dt - t);
                for (uint c = 0; c < n_columns; c++)
                    heads_before[c] = grid_->getColumn(c).getPressureHeads();
                if (grid_->solve(step) < n_columns) {
                    for (uint c = 0; c < n_columns; c++)
                        grid_->getColumn(c).setInitialCondition(heads_before[c]);
                    break;
                }
                t += step;
                steps++;
            }
            advanced = static_cast<double>(t);
            last_step_count_ = steps;
        }
    } catch (...) {
        if (grid_) {
            for (uint c = 0; c < grid_->getNumColumns(); c++)
                grid_->getColumn(c).setInitialCondition(heads_at_start[c]);
        }
        throw;
    }

    sim_time_ += advanced; // simulation clock, used by sensor time stamps
    root_coupling_->accumulateUptake(static_cast<float>(advanced));
    if (advanced < 0.999999 * static_cast<double>(dt)) {
        helios_runtime_error("ERROR (SoilTransportModel::run): the soil solver could not complete the step: it advanced " + std::to_string(advanced) + " s of " + std::to_string(dt) +
                             " s, and the soil state is at the end of the time it advanced. The step usually has no solution because a boundary or root-uptake flux exceeds what "
                             "the soil can deliver or absorb (e.g. evaporation from soil near its residual water content). Reduce the flux, or use a shorter timestep "
                             "(SolverParameters::dt_min / adaptive_dt).");
    }
}

// ============================================================================
// Output for coupling
// ============================================================================

float SoilTransportModel::getSoilWaterPotential(uint plant_id) const {
    requireDomainSetUp("getSoilWaterPotential (plant " + std::to_string(plant_id) + ")");
    return root_coupling_->getSoilWaterPotential(plant_id);
}

float SoilTransportModel::getWaterContent(const vec3 &position) const {
    requireDomainSetUp("getWaterContent");
    uint i, j, layer;
    locateCellOrFail(position, i, j, layer, "getWaterContent");
    if (solver3d_) {
        return static_cast<float>(solver3d_->getWaterContent(i, j, nz_ - 1 - layer)); // the 3D solver's k = 0 is the bottom
    }
    return grid_->getWaterContent(i, j, layer); // uses the column's own soil properties
}

float SoilTransportModel::getPressureHead(const vec3 &position) const {
    requireDomainSetUp("getPressureHead");
    uint i, j, layer;
    locateCellOrFail(position, i, j, layer, "getPressureHead");
    if (solver3d_) {
        return static_cast<float>(solver3d_->getPressureHead(i, j, nz_ - 1 - layer));
    }
    return grid_->getPressureHead(i, j, layer);
}

// ============================================================================
// Point sensors
// ============================================================================

uint SoilTransportModel::addSensor(const vec3 &position, SoilMeasurement quantity) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    uint i, j, layer;
    locateCellOrFail(position, i, j, layer, "addSensor");
    SoilSensor s;
    s.position = position;
    s.quantity = quantity;
    sensors_.push_back(std::move(s));
    return static_cast<uint>(sensors_.size() - 1);
}

void SoilTransportModel::recordSensors() {
    for (SoilSensor &s: sensors_) {
        const float v = (s.quantity == SoilMeasurement::WATER_CONTENT) ? getWaterContent(s.position) : getPressureHead(s.position);
        s.values.push_back(v);
        s.times.push_back(static_cast<float>(sim_time_));
    }
}

uint SoilTransportModel::getSensorCount() const {
    return static_cast<uint>(sensors_.size());
}

const std::vector<float> &SoilTransportModel::getSensorData(uint sensor_id) const {
    if (sensor_id >= sensors_.size()) {
        helios_runtime_error("ERROR (SoilTransportModel::getSensorData): sensor id " + std::to_string(sensor_id) + " out of range.");
    }
    return sensors_[sensor_id].values;
}

const std::vector<float> &SoilTransportModel::getSensorTimes(uint sensor_id) const {
    if (sensor_id >= sensors_.size()) {
        helios_runtime_error("ERROR (SoilTransportModel::getSensorTimes): sensor id " + std::to_string(sensor_id) + " out of range.");
    }
    return sensors_[sensor_id].times;
}

vec3 SoilTransportModel::getSensorPosition(uint sensor_id) const {
    if (sensor_id >= sensors_.size()) {
        helios_runtime_error("ERROR (SoilTransportModel::getSensorPosition): sensor id " + std::to_string(sensor_id) + " out of range.");
    }
    return sensors_[sensor_id].position;
}

SoilMeasurement SoilTransportModel::getSensorQuantity(uint sensor_id) const {
    if (sensor_id >= sensors_.size()) {
        helios_runtime_error("ERROR (SoilTransportModel::getSensorQuantity): sensor id " + std::to_string(sensor_id) + " out of range.");
    }
    return sensors_[sensor_id].quantity;
}

void SoilTransportModel::clearSensorData() {
    for (SoilSensor &s: sensors_) {
        s.values.clear();
        s.times.clear();
    }
}

void SoilTransportModel::writeSensorData(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out) {
        helios_runtime_error("ERROR (SoilTransportModel::writeSensorData): could not open '" + filename + "' for writing.");
    }
    // Header: time + one column per sensor (quantity + position).
    out << "time_s";
    for (size_t s = 0; s < sensors_.size(); s++) {
        const SoilSensor &sen = sensors_[s];
        const char *q = (sen.quantity == SoilMeasurement::WATER_CONTENT) ? "theta" : "head";
        out << "," << q << "_x" << sen.position.x << "_y" << sen.position.y << "_z" << sen.position.z;
    }
    out << "\n";
    // Rows: one per recorded sample. Use the first sensor's time stamps.
    const size_t n = sensors_.empty() ? 0 : sensors_[0].times.size();
    for (size_t i = 0; i < n; i++) {
        out << sensors_[0].times[i];
        for (const SoilSensor &sen: sensors_) {
            out << "," << (i < sen.values.size() ? sen.values[i] : 0.0f);
        }
        out << "\n";
    }
    out.close();
    if (message_flag_) {
        std::cout << "SoilTransportModel: wrote " << n << " samples for " << sensors_.size() << " sensor(s) to '" << filename << "'." << std::endl;
    }
}

float SoilTransportModel::getTimestep() const {
    return use3D() ? static_cast<float>(dt3d_) : dtgrid_;
}

uint64_t SoilTransportModel::getSolverIterationCount() const {
    return solver3d_ ? solver3d_->getTotalPicardIterations() : 0;
}

void SoilTransportModel::resetSolverStats() {
    if (solver3d_)
        solver3d_->resetGpuStats();
}
uint64_t SoilTransportModel::solverPicardOuter() const {
    return solver3d_ ? solver3d_->gpuPicardOuter() : 0;
}
uint64_t SoilTransportModel::solverNewtonOuter() const {
    return solver3d_ ? solver3d_->gpuNewtonOuter() : 0;
}
uint64_t SoilTransportModel::solverPicardGmres() const {
    return solver3d_ ? solver3d_->gpuPicardGmres() : 0;
}
uint64_t SoilTransportModel::solverNewtonGmres() const {
    return solver3d_ ? solver3d_->gpuNewtonGmres() : 0;
}

void SoilTransportModel::resetSolverIterationCount() {
    if (solver3d_)
        solver3d_->resetPicardCounter();
}

double SoilTransportModel::getTotalWaterStorage() const {
    requireDomainSetUp("getTotalWaterStorage");
    if (solver3d_) {
        return solver3d_->getTotalWaterStorage();
    }
    return grid_->getTotalWaterStorage();
}

double SoilTransportModel::getCumulativeBoundaryInflux() const {
    requireDomainSetUp("getCumulativeBoundaryInflux");
    if (!solver3d_) {
        helios_runtime_error("ERROR (SoilTransportModel::getCumulativeBoundaryInflux): the cumulative boundary influx is only tracked by the full 3D solver (SolverType::SOLVER_3D).");
    }
    return solver3d_->getCumulativeInflux();
}

double SoilTransportModel::getMassBalanceResidual() const {
    requireDomainSetUp("getMassBalanceResidual");
    if (!solver3d_) {
        helios_runtime_error("ERROR (SoilTransportModel::getMassBalanceResidual): the mass balance is only tracked by the full 3D solver (SolverType::SOLVER_3D).");
    }
    return solver3d_->getMassBalanceResidual();
}

void SoilTransportModel::resetMassBalance() {
    if (solver3d_)
        solver3d_->resetMassBalance();
}

void SoilTransportModel::setRootWaterStressParameters(float h_onset, float h_wilting, float omega_c) {
    if (!domain_initialized_) {
        initializeDomain();
    }
    root_coupling_->setWaterStressParameters(h_onset, h_wilting, omega_c);
}

float SoilTransportModel::getRootUptakeFlux(uint plant_id) const {
    requireDomainSetUp("getRootUptakeFlux (plant " + std::to_string(plant_id) + ")");
    return root_coupling_->getRootUptakeFlux(plant_id);
}

float SoilTransportModel::getCumulativeRootUptake(uint plant_id) const {
    requireDomainSetUp("getCumulativeRootUptake (plant " + std::to_string(plant_id) + ")");
    return root_coupling_->getCumulativeRootUptake(plant_id);
}

// ============================================================================
// Message control
// ============================================================================

void SoilTransportModel::enableMessages() {
    message_flag_ = true;
}

void SoilTransportModel::disableMessages() {
    message_flag_ = false;
}
