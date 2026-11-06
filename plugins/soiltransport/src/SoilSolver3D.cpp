/** \file "SoilSolver3D.cpp" Implementation of the 3D Richards-equation solver.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilSolver3D.h"
#include "global.h"
#include <cmath>
#include <algorithm>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef HELIOS_CUDA_AVAILABLE
#include <cuda_runtime.h> // cudaGetDeviceCount (isGPUAvailable)
#endif

namespace {
    //! Harmonic mean of two conductivities (zero if both zero).
    inline double hmean(double a, double b) {
        return (a + b > 0.0) ? 2.0 * a * b / (a + b) : 0.0;
    }

    //! Hydraulic conductivity in double precision at head h for the given soil, including the additive conductivity
    //! floor K_min (Params::K_min) that every cell's conductivity carries.
    inline double Kof(double h, const SoilProperties &p, double K_min) {
        double K, C;
        computeKandC<double>(h, p, K, C, 1e-12, K_min);
        return K;
    }
    //! Below this many cells the OpenMP fork/join overhead exceeds the work in a per-cell loop.
    constexpr int kOmpMinCells = 8192;


    //! theta, K, C and dK/dh at head h from one set of shared intermediates (3 pow + 1 sqrt for
    //! van Genuchten-Mualem, vs ~9 pow when computed separately). Values match waterContent(),
    //! computeKandC() and dKdh().
} // namespace

SoilSolver3D::SoilSolver3D(uint Nx, uint Ny, uint Nz, double dx, double dy, double dz, const SoilProperties &props) : Nx_(Nx), Ny_(Ny), Nz_(Nz), dx_(dx), dy_(dy), dz_(dz), Vcell_(dx * dy * dz) {
    if (Nx == 0 || Ny == 0 || Nz == 0)
        helios::helios_runtime_error("ERROR (SoilSolver3D::SoilSolver3D): grid dimensions must be positive.");
    if (!(dx > 0.0) || !(dy > 0.0) || !(dz > 0.0))
        helios::helios_runtime_error("ERROR (SoilSolver3D::SoilSolver3D): the cell spacing must be positive (got dx = " + std::to_string(dx) + ", dy = " + std::to_string(dy) + ", dz = " + std::to_string(dz) + " m).");
    validateSoilProperties(props, "SoilSolver3D::SoilSolver3D");
    uint N = numCells();
    h_.assign(N, -1.0);
    props_.assign(N, props);
    h_infl_.assign(N, inflectionHead(props));
    h_sat_.assign(N, saturationHead(props));
    zc_.resize(N);
    thk_.assign(N, dz);
    for (uint c = 0; c < N; c++)
        zc_[c] = (c / (Nx * Ny) + 0.5) * dz;
    fixed_h_.assign(N, std::numeric_limits<double>::quiet_NaN());
    ss_.assign(N, 0.0);
    src_.assign(N, 0.0);
    uptake_.assign(N, 0.0);
    evap_.assign(N, 0.0);
    K_.resize(N);
    C_.resize(N);
    theta_m_.resize(N);
    theta_n_.resize(N);
    theta_nm1_.resize(N);
    b_.resize(N);
    diag_.resize(N);
    hm_.resize(N);
    r_.resize(N);
    z_.resize(N);
    p_.resize(N);
    Ap_.resize(N);
    u_.resize(N);
}

#ifndef HELIOS_CUDA_AVAILABLE
// On CUDA builds the destructor and the kernel probe live in SoilSolver3D.cu.
SoilSolver3D::~SoilSolver3D() = default;

bool SoilSolver3D::gpuKernelsRun() {
    return false;
}

void SoilSolver3D::releaseGpuState() {
}
#endif

bool SoilSolver3D::isGPUAvailable() {
#ifdef HELIOS_CUDA_AVAILABLE
    // HELIOS_NO_GPU hides the GPU (as for the other GPU plug-ins), e.g. to reproduce a machine without one
    const char *no_gpu = std::getenv("HELIOS_NO_GPU");
    if (no_gpu != nullptr && std::string(no_gpu) != "0") {
        return false;
    }
    // Probed once: a device must exist and be able to run this build's kernels
    static const bool device_usable = [] {
        int device_count = 0;
        const cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            cudaGetLastError(); // answered by the return value; do not leave it as the thread's last CUDA error
            return false;
        }
        return device_count > 0 && gpuKernelsRun();
    }();
    return device_usable;
#else
    return false;
#endif
}

void SoilSolver3D::setSoilProperties(uint i, uint j, uint k, const SoilProperties &props) {
    if (i >= Nx_ || j >= Ny_ || k >= Nz_) {
        helios::helios_runtime_error("ERROR (SoilSolver3D::setSoilProperties): cell (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ") is outside the " + std::to_string(Nx_) + " x " + std::to_string(Ny_) + " x " +
                                     std::to_string(Nz_) + " grid.");
    }
    validateSoilProperties(props, "SoilSolver3D::setSoilProperties");
    props_[index(i, j, k)] = props;
    h_infl_[index(i, j, k)] = inflectionHead(props);
    h_sat_[index(i, j, k)] = saturationHead(props);
    gpu_props_dirty_ = true; // force a GPU re-upload (and coarse-level FAS props rebuild) on the next solve
    kirchhoff_built_ = false; // the M-scheme's Kirchhoff table is built for the soil; rebuild it on the next solve
}

const SoilProperties &SoilSolver3D::getSoilProperties(uint i, uint j, uint k) const {
    if (i >= Nx_ || j >= Ny_ || k >= Nz_) {
        helios::helios_runtime_error("ERROR (SoilSolver3D::getSoilProperties): cell (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ") is outside the " + std::to_string(Nx_) + " x " + std::to_string(Ny_) + " x " +
                                     std::to_string(Nz_) + " grid.");
    }
    return props_[index(i, j, k)];
}

void SoilSolver3D::setInitialCondition(double h) {
    std::fill(h_.begin(), h_.end(), h);
}

void SoilSolver3D::setInitialCondition(const std::vector<double> &h) {
    if (h.size() != numCells())
        helios::helios_runtime_error("ERROR (SoilSolver3D::setInitialCondition): size mismatch.");
    h_ = h;
}

void SoilSolver3D::setCellGeometry(const std::vector<double> &z_center, const std::vector<double> &thickness) {
    const uint N = numCells();
    if (z_center.size() != N || thickness.size() != N)
        helios::helios_runtime_error("ERROR (SoilSolver3D::setCellGeometry): z_center and thickness must have one entry per cell.");
    for (uint c = 0; c < N; c++) {
        if (!(thickness[c] > 0.0))
            helios::helios_runtime_error("ERROR (SoilSolver3D::setCellGeometry): cell thicknesses must be positive.");
        if (c >= Nx_ * Ny_ && !(z_center[c] > z_center[c - Nx_ * Ny_]))
            helios::helios_runtime_error("ERROR (SoilSolver3D::setCellGeometry): cell-centre elevation must increase with k.");
    }
    zc_ = z_center;
    thk_ = thickness;
    uniform_geometry_ = false;
}

void SoilSolver3D::setFixedHead(uint i, uint j, uint k, double h) {
    if (i >= Nx_ || j >= Ny_ || k >= Nz_)
        helios::helios_runtime_error("ERROR (SoilSolver3D::setFixedHead): cell index out of range.");
    const uint c = index(i, j, k);
    if (std::isnan(fixed_h_[c]))
        n_fixed_++;
    fixed_h_[c] = h;
    h_[c] = h;
}

void SoilSolver3D::setSpecificStorage(const std::vector<double> &Ss) {
    if (Ss.size() != numCells())
        helios::helios_runtime_error("ERROR (SoilSolver3D::setSpecificStorage): need one value per cell.");
    for (double v: Ss)
        if (v < 0.0)
            helios::helios_runtime_error("ERROR (SoilSolver3D::setSpecificStorage): specific storage must be non-negative.");
    ss_ = Ss;
    has_ss_ = std::any_of(ss_.begin(), ss_.end(), [](double v) { return v > 0.0; });
}

void SoilSolver3D::clearFixedHeads() {
    std::fill(fixed_h_.begin(), fixed_h_.end(), std::numeric_limits<double>::quiet_NaN());
    n_fixed_ = 0;
}

void SoilSolver3D::setBoundary(Face3D face, BCType type, double value) {
    if (type == BCType::ATMOSPHERIC)
        helios::helios_runtime_error("ERROR (SoilSolver3D::setBoundary): BCType::ATMOSPHERIC is supported by SoilColumn only.");
    bc_[static_cast<int>(face)] = {type, value};
}

void SoilSolver3D::setSource(uint i, uint j, uint k, double rate) {
    src_[index(i, j, k)] = rate;
}

void SoilSolver3D::addSource(uint i, uint j, uint k, double rate) {
    src_[index(i, j, k)] += rate;
}

void SoilSolver3D::clearSources() {
    std::fill(src_.begin(), src_.end(), 0.0);
}

void SoilSolver3D::setUptake(uint i, uint j, uint k, double rate) {
    uptake_[index(i, j, k)] = rate;
}

void SoilSolver3D::clearUptake() {
    std::fill(uptake_.begin(), uptake_.end(), 0.0);
}

void SoilSolver3D::setEvaporation(uint i, uint j, uint k, double rate) {
    evap_[index(i, j, k)] = rate;
}

void SoilSolver3D::clearEvaporation() {
    std::fill(evap_.begin(), evap_.end(), 0.0);
}

double SoilSolver3D::faceCond(double Ka, double Kb, double Ha, double Hb, double Ksa, double Ksb) const {
    switch (params_.K_weighting) {
        case 4: { // upstream relative permeability: harmonic-mean saturated K times the donor cell's k_r
            const double dH = Ha - Hb;
            const bool a_up = dH > 0.0 || (dH == 0.0 && Ka / Ksa >= Kb / Ksb);
            return hmean(Ksa, Ksb) * (a_up ? Ka / Ksa : Kb / Ksb);
        }
        case 0:
            return std::sqrt(Ka * Kb); // geometric mean
        case 1:
            return 0.5 * (Ka + Kb); // arithmetic mean
        case 2: { // upstream: donor cell by total head
            const double dH = Ha - Hb;
            if (dH > 0.0)
                return Ka;
            if (dH < 0.0)
                return Kb;
            return std::fmax(Ka, Kb); // symmetric tie-break (keeps the matrix symmetric)
        }
        default:
            return (Ka + Kb > 0.0) ? 2.0 * Ka * Kb / (Ka + Kb) : 0.0; // harmonic
    }
}

// Conductivity of a Dirichlet boundary face: the boundary is treated as a virtual cell with head h_bc
// half a cell beyond the face (same soil as the cell), and the face uses the selected interblock
// weighting, with flow direction set by total head. (Harmonic weighting would block infiltration from a
// wet boundary into dry soil for every other choice, since hmean(K_wet, K_dry) ~ K_dry.)
double SoilSolver3D::bcFaceK(int face, double Kc, double hc, const SoilProperties &pc) const {
    const double hbc = bc_[face].value;
    const double off = (face == 4) ? -0.5 * dz_ : (face == 5 ? 0.5 * dz_ : 0.0); // boundary-node elevation - z_c
    return faceCond(Kc, Kof(hbc, pc, params_.K_min), hc, hbc + off, pc.K_s, pc.K_s);
}

double SoilSolver3D::getWaterContent(uint i, uint j, uint k) const {
    uint c = index(i, j, k);
    return waterContent<double>(h_[c], props_[c]);
}

double SoilSolver3D::storedWater(int c, double h) const {
    double theta = waterContent<double>(h, props_[c]);
    if (ss_[c] != 0.0)
        theta += ss_[c] * theta / props_[c].theta_s * h;
    return theta;
}

double SoilSolver3D::getTotalWaterStorage() const {
    double s = 0.0;
    for (uint c = 0; c < numCells(); c++)
        s += storedWater(c, h_[c]) * cellVolume(c);
    return s;
}

// Net external water influx RATE [m^3/s] at the converged state h_, summing
// sources and boundary-face fluxes exactly as buildSystem() applies them. For a
// Dirichlet face the influx is cf*(h_bc - h_cell) (+ gravity on z-faces), since
// buildSystem adds cf to the diagonal and cf*h_bc (+gravity) to the RHS.
double SoilSolver3D::stepInflux() {
    if (!uniform_geometry_ || n_fixed_ > 0) { // general geometry / specified heads: the Newton assembly's inflow
        std::vector<double> F;
        assembleNewton(h_, F, 0.0, 0.0, 0.0, false);
        double sum = 0.0;
        for (uint c = 0; c < numCells(); c++)
            sum += nk_ext_[c] * cellVolume(c);
        return sum; // [m^3/s]
    }
    const int Nx = Nx_, Ny = Ny_, Nz = Nz_, NxNy = Nx * Ny, N = (int) numCells();
    const double idx2 = 1.0 / (dx_ * dx_), idy2 = 1.0 / (dy_ * dy_), idz2 = 1.0 / (dz_ * dz_);
    const double idx = 1.0 / dx_, idy = 1.0 / dy_, idz = 1.0 / dz_;
    double sum = 0.0; // sum over cells of per-volume external rate [1/s]
#pragma omp parallel for reduction(+ : sum) schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const SoilProperties &pc = props_[c];
        double Kc, Cc;
        computeKandC<double>(h_[c], pc, Kc, Cc, params_.C_min, params_.K_min);
        double q = src_[c] - uptake_[c] - evap_[c]; // root uptake + surface evaporation enter as negative sources
        if (i == 0) {
            if (bc_[0].type == BCType::DIRICHLET) {
                double cf = 2.0 * bcFaceK(0, Kc, h_[c], pc) * idx2;
                q += cf * (bc_[0].value - h_[c]);
            } else if (bc_[0].type == BCType::NEUMANN)
                q += bc_[0].value * idx;
        }
        if (i == Nx - 1) {
            if (bc_[1].type == BCType::DIRICHLET) {
                double cf = 2.0 * bcFaceK(1, Kc, h_[c], pc) * idx2;
                q += cf * (bc_[1].value - h_[c]);
            } else if (bc_[1].type == BCType::NEUMANN)
                q += bc_[1].value * idx;
        }
        if (j == 0) {
            if (bc_[2].type == BCType::DIRICHLET) {
                double cf = 2.0 * bcFaceK(2, Kc, h_[c], pc) * idy2;
                q += cf * (bc_[2].value - h_[c]);
            } else if (bc_[2].type == BCType::NEUMANN)
                q += bc_[2].value * idy;
        }
        if (j == Ny - 1) {
            if (bc_[3].type == BCType::DIRICHLET) {
                double cf = 2.0 * bcFaceK(3, Kc, h_[c], pc) * idy2;
                q += cf * (bc_[3].value - h_[c]);
            } else if (bc_[3].type == BCType::NEUMANN)
                q += bc_[3].value * idy;
        }
        if (k == 0) { // bottom
            if (bc_[4].type == BCType::DIRICHLET) {
                double cf = 2.0 * bcFaceK(4, Kc, h_[c], pc) * idz2;
                q += cf * (bc_[4].value - h_[c]) + cf * (-dz_ / 2.0);
            } else if (bc_[4].type == BCType::NEUMANN)
                q += bc_[4].value * idz;
            else if (bc_[4].type == BCType::FREE_DRAINAGE)
                q += -Kc * idz;
        }
        if (k == Nz - 1) { // top
            if (bc_[5].type == BCType::DIRICHLET) {
                double cf = 2.0 * bcFaceK(5, Kc, h_[c], pc) * idz2;
                q += cf * (bc_[5].value - h_[c]) + cf * (dz_ / 2.0);
            } else if (bc_[5].type == BCType::NEUMANN)
                q += bc_[5].value * idz;
            else if (bc_[5].type == BCType::FREE_DRAINAGE)
                q += Kc * idz;
        }
        sum += q;
    }
    return sum * Vcell_; // [m^3/s]
}

double SoilSolver3D::stepInfluxMScheme() const {
    // Net external influx for the M-scheme, using its OWN Kirchhoff diffusive boundary flux (2*D0*(w_tilde_bc
    // - w_tilde_c)) instead of the head-form hmean(K)*(h_bc - h_c). Gravity (K*z-hat at z-faces), Neumann,
    // free-drainage and the source/sink terms are identical to the head form. Lets getMassBalanceResidual()
    // report the M-scheme's true conservation defect (~0, since the scheme is fully implicit + FV-conservative
    // in the Kirchhoff discretization) rather than a spurious Kirchhoff-vs-head mismatch.
    const KirchhoffTable &kt = kirchhoff_;
    const double D0 = kt.w_sat;
    const int Nx = Nx_, Ny = Ny_, Nz = Nz_, NxNy = Nx * Ny, N = (int) numCells();
    const double idx2 = 1.0 / (dx_ * dx_), idy2 = 1.0 / (dy_ * dy_), idz2 = 1.0 / (dz_ * dz_);
    const double idx = 1.0 / dx_, idy = 1.0 / dy_, idz = 1.0 / dz_;
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const SoilProperties &pc = props_[c];
        double Kc, Cc;
        computeKandC<double>(h_[c], pc, Kc, Cc, params_.C_min, params_.K_min);
        const double wt_c = kt.wtildeOfPsi(h_[c]);
        double q = src_[c] - uptake_[c] - evap_[c];
        // Kirchhoff diffusive flux at Dirichlet faces; gravity at z Dirichlet faces matches the head form
        // (cf_grav*(±dz/2) with cf_grav = 2*hmean(Kc,K_bc)/d^2 == hmean*idz).
        if (i == 0) {
            if (bc_[0].type == BCType::DIRICHLET)
                q += 2.0 * D0 * idx2 * (kt.wtildeOfPsi(bc_[0].value) - wt_c);
            else if (bc_[0].type == BCType::NEUMANN)
                q += bc_[0].value * idx;
        }
        if (i == Nx - 1) {
            if (bc_[1].type == BCType::DIRICHLET)
                q += 2.0 * D0 * idx2 * (kt.wtildeOfPsi(bc_[1].value) - wt_c);
            else if (bc_[1].type == BCType::NEUMANN)
                q += bc_[1].value * idx;
        }
        if (j == 0) {
            if (bc_[2].type == BCType::DIRICHLET)
                q += 2.0 * D0 * idy2 * (kt.wtildeOfPsi(bc_[2].value) - wt_c);
            else if (bc_[2].type == BCType::NEUMANN)
                q += bc_[2].value * idy;
        }
        if (j == Ny - 1) {
            if (bc_[3].type == BCType::DIRICHLET)
                q += 2.0 * D0 * idy2 * (kt.wtildeOfPsi(bc_[3].value) - wt_c);
            else if (bc_[3].type == BCType::NEUMANN)
                q += bc_[3].value * idy;
        }
        if (k == 0) { // bottom
            if (bc_[4].type == BCType::DIRICHLET)
                q += 2.0 * D0 * idz2 * (kt.wtildeOfPsi(bc_[4].value) - wt_c) - hmean(Kc, Kof(bc_[4].value, pc, params_.K_min)) * idz;
            else if (bc_[4].type == BCType::NEUMANN)
                q += bc_[4].value * idz;
            else if (bc_[4].type == BCType::FREE_DRAINAGE)
                q += -Kc * idz;
        }
        if (k == Nz - 1) { // top
            if (bc_[5].type == BCType::DIRICHLET)
                q += 2.0 * D0 * idz2 * (kt.wtildeOfPsi(bc_[5].value) - wt_c) + hmean(Kc, Kof(bc_[5].value, pc, params_.K_min)) * idz;
            else if (bc_[5].type == BCType::NEUMANN)
                q += bc_[5].value * idz;
            else if (bc_[5].type == BCType::FREE_DRAINAGE)
                q += Kc * idz;
        }
        sum += q;
    }
    return sum * Vcell_; // [m^3/s]
}

void SoilSolver3D::updateKC(const std::vector<double> &hm) {
    const int N = static_cast<int>(numCells());
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        double K, C;
        computeKandC<double>(hm[c], props_[c], K, C, params_.C_min, params_.K_min);
        K_[c] = K;
        C_[c] = C;
        theta_m_[c] = waterContent<double>(hm[c], props_[c]);
    }
}

// A v = a0*C*v - L_K v, with Dirichlet faces contributing to the diagonal.
void SoilSolver3D::matvec(const std::vector<double> &v, std::vector<double> &out, double a0) const {
    const int N = static_cast<int>(numCells());
    const int Nx = Nx_, Ny = Ny_, Nz = Nz_, NxNy = Nx * Ny;
    const double idx2 = 1.0 / (dx_ * dx_), idy2 = 1.0 / (dy_ * dy_), idz2 = 1.0 / (dz_ * dz_);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const double Kc = K_[c];
        double acc = C_[c] * a0 * v[c];
        const double Hc = hm_[c]; // total head h + z; z cancels on x/y faces, +/- dz_ on z faces
        // -x / +x
        if (i > 0) {
            double cf = faceCond(Kc, K_[c - 1], Hc, hm_[c - 1], props_[c].K_s, props_[c - 1].K_s) * idx2;
            acc += cf * (v[c] - v[c - 1]);
        } else if (bc_[0].type == BCType::DIRICHLET)
            acc += 2.0 * bcFaceK(0, Kc, Hc, props_[c]) * idx2 * v[c];
        if (i < Nx - 1) {
            double cf = faceCond(Kc, K_[c + 1], Hc, hm_[c + 1], props_[c].K_s, props_[c + 1].K_s) * idx2;
            acc += cf * (v[c] - v[c + 1]);
        } else if (bc_[1].type == BCType::DIRICHLET)
            acc += 2.0 * bcFaceK(1, Kc, Hc, props_[c]) * idx2 * v[c];
        // -y / +y
        if (j > 0) {
            double cf = faceCond(Kc, K_[c - Nx], Hc, hm_[c - Nx], props_[c].K_s, props_[c - Nx].K_s) * idy2;
            acc += cf * (v[c] - v[c - Nx]);
        } else if (bc_[2].type == BCType::DIRICHLET)
            acc += 2.0 * bcFaceK(2, Kc, Hc, props_[c]) * idy2 * v[c];
        if (j < Ny - 1) {
            double cf = faceCond(Kc, K_[c + Nx], Hc, hm_[c + Nx], props_[c].K_s, props_[c + Nx].K_s) * idy2;
            acc += cf * (v[c] - v[c + Nx]);
        } else if (bc_[3].type == BCType::DIRICHLET)
            acc += 2.0 * bcFaceK(3, Kc, Hc, props_[c]) * idy2 * v[c];
        // -z (bottom) / +z (top): neighbor elevation differs by dz_
        if (k > 0) {
            double cf = faceCond(Kc, K_[c - NxNy], Hc, hm_[c - NxNy] - dz_, props_[c].K_s, props_[c - NxNy].K_s) * idz2;
            acc += cf * (v[c] - v[c - NxNy]);
        } else if (bc_[4].type == BCType::DIRICHLET)
            acc += 2.0 * bcFaceK(4, Kc, Hc, props_[c]) * idz2 * v[c];
        if (k < Nz - 1) {
            double cf = faceCond(Kc, K_[c + NxNy], Hc, hm_[c + NxNy] + dz_, props_[c].K_s, props_[c + NxNy].K_s) * idz2;
            acc += cf * (v[c] - v[c + NxNy]);
        } else if (bc_[5].type == BCType::DIRICHLET)
            acc += 2.0 * bcFaceK(5, Kc, Hc, props_[c]) * idz2 * v[c];
        out[c] = acc;
    }
}

// Fill b_ (RHS) and diag_ (for Jacobi preconditioner) in one pass.
void SoilSolver3D::buildSystem(const std::vector<double> &hm, double a0, double a1, double a2) {
    const int N = static_cast<int>(numCells());
    const int Nx = Nx_, Ny = Ny_, Nz = Nz_, NxNy = Nx * Ny;
    const double idx2 = 1.0 / (dx_ * dx_), idy2 = 1.0 / (dy_ * dy_), idz2 = 1.0 / (dz_ * dz_);
    const double idz = 1.0 / dz_, idx = 1.0 / dx_, idy = 1.0 / dy_;
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const double Kc = K_[c];
        const SoilProperties &pc = props_[c];
        double d = C_[c] * a0;
        double rhs = C_[c] * a0 * hm[c] - (a0 * theta_m_[c] + a1 * theta_n_[c] + a2 * theta_nm1_[c]) + src_[c] - uptake_[c] - evap_[c];

        const double Hc = hm[c]; // total head (h + z); z cancels on x/y faces, +/- dz_ on z faces
        // -x
        if (i > 0)
            d += faceCond(Kc, K_[c - 1], Hc, hm[c - 1], props_[c].K_s, props_[c - 1].K_s) * idx2;
        else if (bc_[0].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(0, Kc, Hc, pc) * idx2;
            d += cf;
            rhs += cf * bc_[0].value;
        } else if (bc_[0].type == BCType::NEUMANN)
            rhs += bc_[0].value * idx;
        // +x
        if (i < Nx - 1)
            d += faceCond(Kc, K_[c + 1], Hc, hm[c + 1], props_[c].K_s, props_[c + 1].K_s) * idx2;
        else if (bc_[1].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(1, Kc, Hc, pc) * idx2;
            d += cf;
            rhs += cf * bc_[1].value;
        } else if (bc_[1].type == BCType::NEUMANN)
            rhs += bc_[1].value * idx;
        // -y
        if (j > 0)
            d += faceCond(Kc, K_[c - Nx], Hc, hm[c - Nx], props_[c].K_s, props_[c - Nx].K_s) * idy2;
        else if (bc_[2].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(2, Kc, Hc, pc) * idy2;
            d += cf;
            rhs += cf * bc_[2].value;
        } else if (bc_[2].type == BCType::NEUMANN)
            rhs += bc_[2].value * idy;
        // +y
        if (j < Ny - 1)
            d += faceCond(Kc, K_[c + Nx], Hc, hm[c + Nx], props_[c].K_s, props_[c + Nx].K_s) * idy2;
        else if (bc_[3].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(3, Kc, Hc, pc) * idy2;
            d += cf;
            rhs += cf * bc_[3].value;
        } else if (bc_[3].type == BCType::NEUMANN)
            rhs += bc_[3].value * idy;
        // -z (bottom): gravity term present on z-faces; neighbor elevation = z - dz_
        if (k > 0) {
            double Kz = faceCond(Kc, K_[c - NxNy], Hc, hm[c - NxNy] - dz_, props_[c].K_s, props_[c - NxNy].K_s);
            d += Kz * idz2;
            rhs += -Kz * idz;
        } else if (bc_[4].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(4, Kc, Hc, pc) * idz2;
            d += cf;
            rhs += cf * bc_[4].value + cf * (-dz_ / 2.0);
        } else if (bc_[4].type == BCType::NEUMANN)
            rhs += bc_[4].value * idz;
        else if (bc_[4].type == BCType::FREE_DRAINAGE)
            rhs += -Kc * idz; // unit gradient, flux = K out the bottom
        // +z (top): neighbor elevation = z + dz_
        if (k < Nz - 1) {
            double Kz = faceCond(Kc, K_[c + NxNy], Hc, hm[c + NxNy] + dz_, props_[c].K_s, props_[c + NxNy].K_s);
            d += Kz * idz2;
            rhs += Kz * idz;
        } else if (bc_[5].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(5, Kc, Hc, pc) * idz2;
            d += cf;
            rhs += cf * bc_[5].value + cf * (dz_ / 2.0);
        } else if (bc_[5].type == BCType::NEUMANN)
            rhs += bc_[5].value * idz;
        else if (bc_[5].type == BCType::FREE_DRAINAGE)
            rhs += Kc * idz;

        diag_[c] = d;
        b_[c] = rhs;
    }
}

void SoilSolver3D::timeWeights(double dt, double &a0, double &a1, double &a2) const {
    if (params_.bdf2 && have_history_ && dt_prev_ > 0.0) {
        const double r = dt / dt_prev_; // variable-step BDF2
        a0 = (1.0 + 2.0 * r) / ((1.0 + r) * dt);
        a1 = -(1.0 + r) / dt;
        a2 = (r * r) / ((1.0 + r) * dt);
    } else { // backward Euler (also the BDF2 bootstrap step)
        a0 = 1.0 / dt;
        a1 = -1.0 / dt;
        a2 = 0.0;
    }
}

// Nonlinear residual R(h) = A(h) h - b(h), computed directly (the a0*C*h Picard
// linearization terms cancel between A h and b, so C does not appear -- the residual
// is in theta(h), not the chord capacity). Every term is transcribed from matvec()
// and buildSystem() so the discretization is identical by construction; the a0*C term
// is simply omitted from both sides. Self-contained: local K per cell, reads only h
// and the theta_n_/theta_nm1_ history, writes nothing the preconditioner depends on.
void SoilSolver3D::residual(const std::vector<double> &h, std::vector<double> &F, double a0, double a1, double a2) const {
    const int N = static_cast<int>(numCells());
    const int Nx = Nx_, Ny = Ny_, Nz = Nz_, NxNy = Nx * Ny;
    const double idx2 = 1.0 / (dx_ * dx_), idy2 = 1.0 / (dy_ * dy_), idz2 = 1.0 / (dz_ * dz_);
    const double idx = 1.0 / dx_, idy = 1.0 / dy_, idz = 1.0 / dz_;
    F.resize(N);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const SoilProperties &pc = props_[c];
        double Kc, Cc;
        computeKandC<double>(h[c], pc, Kc, Cc, params_.C_min, params_.K_min);
        const double hc = h[c];
        // Storage rate a0 theta(h_c) + a1 theta^n + a2 theta^{n-1}  (= -(the -(...) in b)).
        double R = a0 * waterContent<double>(hc, pc) + a1 * theta_n_[c] + a2 * theta_nm1_[c];
        // External sources: b held (+src - uptake - evap), so R = A h - b contributes the negatives.
        R += -src_[c] + uptake_[c] + evap_[c];

        double Kn, Cn; // neighbor conductivity scratch (recomputed locally per face)
        // -x / +x
        if (i > 0) {
            computeKandC<double>(h[c - 1], props_[c - 1], Kn, Cn, params_.C_min, params_.K_min);
            R += faceCond(Kc, Kn, hc, h[c - 1], pc.K_s, props_[c - 1].K_s) * idx2 * (hc - h[c - 1]);
        } else if (bc_[0].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(0, Kc, hc, pc) * idx2;
            R += cf * (hc - bc_[0].value);
        } else if (bc_[0].type == BCType::NEUMANN)
            R += -bc_[0].value * idx;
        if (i < Nx - 1) {
            computeKandC<double>(h[c + 1], props_[c + 1], Kn, Cn, params_.C_min, params_.K_min);
            R += faceCond(Kc, Kn, hc, h[c + 1], pc.K_s, props_[c + 1].K_s) * idx2 * (hc - h[c + 1]);
        } else if (bc_[1].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(1, Kc, hc, pc) * idx2;
            R += cf * (hc - bc_[1].value);
        } else if (bc_[1].type == BCType::NEUMANN)
            R += -bc_[1].value * idx;
        // -y / +y
        if (j > 0) {
            computeKandC<double>(h[c - Nx], props_[c - Nx], Kn, Cn, params_.C_min, params_.K_min);
            R += faceCond(Kc, Kn, hc, h[c - Nx], pc.K_s, props_[c - Nx].K_s) * idy2 * (hc - h[c - Nx]);
        } else if (bc_[2].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(2, Kc, hc, pc) * idy2;
            R += cf * (hc - bc_[2].value);
        } else if (bc_[2].type == BCType::NEUMANN)
            R += -bc_[2].value * idy;
        if (j < Ny - 1) {
            computeKandC<double>(h[c + Nx], props_[c + Nx], Kn, Cn, params_.C_min, params_.K_min);
            R += faceCond(Kc, Kn, hc, h[c + Nx], pc.K_s, props_[c + Nx].K_s) * idy2 * (hc - h[c + Nx]);
        } else if (bc_[3].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(3, Kc, hc, pc) * idy2;
            R += cf * (hc - bc_[3].value);
        } else if (bc_[3].type == BCType::NEUMANN)
            R += -bc_[3].value * idy;
        // -z (bottom): neighbor elevation z - dz; gravity term lived in b's RHS as -Kz*idz.
        if (k > 0) {
            computeKandC<double>(h[c - NxNy], props_[c - NxNy], Kn, Cn, params_.C_min, params_.K_min);
            const double Kz = faceCond(Kc, Kn, hc, h[c - NxNy] - dz_, pc.K_s, props_[c - NxNy].K_s);
            R += Kz * idz2 * (hc - h[c - NxNy]) + Kz * idz;
        } else if (bc_[4].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(4, Kc, hc, pc) * idz2;
            R += cf * (hc - bc_[4].value) + cf * (dz_ / 2.0);
        } else if (bc_[4].type == BCType::NEUMANN)
            R += -bc_[4].value * idz;
        else if (bc_[4].type == BCType::FREE_DRAINAGE)
            R += Kc * idz;
        // +z (top): neighbor elevation z + dz; gravity term lived in b's RHS as +Kz*idz.
        if (k < Nz - 1) {
            computeKandC<double>(h[c + NxNy], props_[c + NxNy], Kn, Cn, params_.C_min, params_.K_min);
            const double Kz = faceCond(Kc, Kn, hc, h[c + NxNy] + dz_, pc.K_s, props_[c + NxNy].K_s);
            R += Kz * idz2 * (hc - h[c + NxNy]) - Kz * idz;
        } else if (bc_[5].type == BCType::DIRICHLET) {
            double cf = 2.0 * bcFaceK(5, Kc, hc, pc) * idz2;
            R += cf * (hc - bc_[5].value) - cf * (dz_ / 2.0);
        } else if (bc_[5].type == BCType::NEUMANN)
            R += -bc_[5].value * idz;
        else if (bc_[5].type == BCType::FREE_DRAINAGE)
            R += -Kc * idz;

        F[c] = R;
    }
}

// Matrix-free J(h) v by one-sided finite difference of the nonlinear residual. The
// increment e = sqrt(eps_mach)*(1+||h||)/||v|| (Knoll & Keyes 2004) balances FD truncation
// against round-off. F0 = R(h) is supplied so it is reused across Krylov iterations.
void SoilSolver3D::jacvec(const std::vector<double> &h, const std::vector<double> &F0, const std::vector<double> &v, std::vector<double> &Jv, double a0, double a1, double a2) const {
    const int N = static_cast<int>(numCells());
    double vnorm2 = 0.0, hnorm2 = 0.0;
#pragma omp parallel for reduction(+ : vnorm2, hnorm2) schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        vnorm2 += v[c] * v[c];
        hnorm2 += h[c] * h[c];
    }
    const double vnorm = std::sqrt(vnorm2);
    Jv.resize(N);
    if (vnorm < 1e-300) {
        std::fill(Jv.begin(), Jv.end(), 0.0);
        return;
    }
    const double bb = 1.49011611938e-8; // ~sqrt(double machine epsilon)
    const double eps = bb * (1.0 + std::sqrt(hnorm2)) / vnorm;
    std::vector<double> hp(N), Fp;
    for (int c = 0; c < N; c++)
        hp[c] = h[c] + eps * v[c];
    residual(hp, Fp, a0, a1, a2);
    const double inv = 1.0 / eps;
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        Jv[c] = (Fp[c] - F0[c]) * inv;
}

// Action of the frozen Modified-Picard operator A(h) v = a0 C v - L_K v. Freezes K_/C_ at
// h (updateKC) and points hm_ at h so matvec() uses the correct total heads for upstream
// weighting, then applies the existing SPD matvec.
void SoilSolver3D::picardApply(const std::vector<double> &h, const std::vector<double> &v, std::vector<double> &Av, double a0) {
    updateKC(h);
    hm_ = h;
    Av.resize(numCells());
    matvec(v, Av, a0);
}

// Right-preconditioned restarted GMRES(m). Solves A x = b with A = apply(), M^{-1} = precond().
// Standard Arnoldi (modified Gram-Schmidt) + Givens rotations; the rotated rhs |g[j+1]| tracks
// the true residual norm so we can stop without an extra matvec. H is stored column-major,
// H[i + (m+1)*j] = entry (i,j).
uint SoilSolver3D::gmres(const std::function<void(const std::vector<double> &, std::vector<double> &)> &apply, const std::function<void(const std::vector<double> &, std::vector<double> &)> &precond, const std::vector<double> &b,
                         std::vector<double> &x, int restart, int maxit, double tol) {
    const int N = static_cast<int>(numCells());
    const int m = std::max(1, restart);
    double bnorm = 0.0;
    for (int c = 0; c < N; c++)
        bnorm += b[c] * b[c];
    bnorm = std::sqrt(bnorm);
    if (bnorm < 1e-300) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }
    const double thresh = tol * bnorm;

    std::vector<std::vector<double>> V(m + 1, std::vector<double>(N));
    std::vector<double> H((m + 1) * m, 0.0), cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0);
    std::vector<double> w(N), Mv(N), Ax(N), tmp(N);

    uint total_inner = 0;
    while (total_inner < (uint) maxit) {
        apply(x, Ax); // r0 = b - A x0
        double beta = 0.0;
        for (int c = 0; c < N; c++) {
            V[0][c] = b[c] - Ax[c];
            beta += V[0][c] * V[0][c];
        }
        beta = std::sqrt(beta);
        if (beta <= thresh)
            break;
        const double inv_beta = 1.0 / beta;
        for (int c = 0; c < N; c++)
            V[0][c] *= inv_beta;
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int j = 0;
        for (; j < m && total_inner < (uint) maxit; j++) {
            total_inner++;
            precond(V[j], Mv); // right preconditioning: w = A M^{-1} v_j
            apply(Mv, w);
            for (int i = 0; i <= j; i++) { // modified Gram-Schmidt
                double hij = 0.0;
                for (int c = 0; c < N; c++)
                    hij += w[c] * V[i][c];
                H[i + (m + 1) * j] = hij;
                for (int c = 0; c < N; c++)
                    w[c] -= hij * V[i][c];
            }
            double hnext = 0.0;
            for (int c = 0; c < N; c++)
                hnext += w[c] * w[c];
            hnext = std::sqrt(hnext);
            H[(j + 1) + (m + 1) * j] = hnext;
            if (hnext > 1e-300) {
                const double inv_h = 1.0 / hnext;
                for (int c = 0; c < N; c++)
                    V[j + 1][c] = w[c] * inv_h;
            }
            for (int i = 0; i < j; i++) { // apply previous Givens rotations to the new column
                const double t = cs[i] * H[i + (m + 1) * j] + sn[i] * H[(i + 1) + (m + 1) * j];
                H[(i + 1) + (m + 1) * j] = -sn[i] * H[i + (m + 1) * j] + cs[i] * H[(i + 1) + (m + 1) * j];
                H[i + (m + 1) * j] = t;
            }
            const double h_jj = H[j + (m + 1) * j], h_j1 = H[(j + 1) + (m + 1) * j]; // new rotation
            double denom = std::sqrt(h_jj * h_jj + h_j1 * h_j1);
            if (denom < 1e-300)
                denom = 1e-300;
            cs[j] = h_jj / denom;
            sn[j] = h_j1 / denom;
            H[j + (m + 1) * j] = cs[j] * h_jj + sn[j] * h_j1;
            H[(j + 1) + (m + 1) * j] = 0.0;
            g[j + 1] = -sn[j] * g[j]; // update rotated rhs
            g[j] = cs[j] * g[j];
            if (std::fabs(g[j + 1]) <= thresh) {
                j++;
                break;
            }
        }

        const int k = j; // completed columns this cycle
        std::vector<double> y(k, 0.0); // back-substitute H(0:k,0:k) y = g
        for (int i = k - 1; i >= 0; i--) {
            double s = g[i];
            for (int l = i + 1; l < k; l++)
                s -= H[i + (m + 1) * l] * y[l];
            y[i] = s / H[i + (m + 1) * i];
        }
        std::fill(tmp.begin(), tmp.end(), 0.0); // x += M^{-1} (V_k y)
        for (int i = 0; i < k; i++)
            for (int c = 0; c < N; c++)
                tmp[c] += y[i] * V[i][c];
        precond(tmp, Mv);
        for (int c = 0; c < N; c++)
            x[c] += Mv[c];

        apply(x, Ax); // true-residual convergence check before another restart
        double rn = 0.0;
        for (int c = 0; c < N; c++) {
            const double rr = b[c] - Ax[c];
            rn += rr * rr;
        }
        if (std::sqrt(rn) <= thresh)
            break;
    }
    return total_inner;
}

// Jacobi-preconditioned conjugate gradient. Solves A u = b_; u in/out.
uint SoilSolver3D::cg(std::vector<double> &u, double a0) {
    const int N = static_cast<int>(numCells());
    auto applyPrecond = [&]() { // z_ = M^{-1} r_  (multigrid V-cycle or Jacobi)
        if (params_.use_multigrid) {
            mgApply(z_, r_);
        } else {
#pragma omp parallel for schedule(static) num_threads(nthreads_)
            for (int c = 0; c < N; c++)
                z_[c] = r_[c] / diag_[c];
        }
    };
    matvec(u, Ap_, a0);
    double bnorm = 0.0, rz = 0.0;
#pragma omp parallel for reduction(+ : bnorm) schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        bnorm += b_[c] * b_[c];
    bnorm = std::sqrt(bnorm) + 1e-300;
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        r_[c] = b_[c] - Ap_[c];
    applyPrecond(); // z_ = M^{-1} r_
#pragma omp parallel for reduction(+ : rz) schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        p_[c] = z_[c];
        rz += r_[c] * z_[c];
    }
    uint it = 0;
    for (; it < params_.cg_max; it++) {
        matvec(p_, Ap_, a0);
        double pAp = 0.0;
#pragma omp parallel for reduction(+ : pAp) schedule(static) num_threads(nthreads_)
        for (int c = 0; c < N; c++)
            pAp += p_[c] * Ap_[c];
        double alpha = rz / (pAp + 1e-300);
        double rnorm = 0.0;
#pragma omp parallel for reduction(+ : rnorm) schedule(static) num_threads(nthreads_)
        for (int c = 0; c < N; c++) {
            u[c] += alpha * p_[c];
            r_[c] -= alpha * Ap_[c];
            rnorm += r_[c] * r_[c];
        }
        if (std::sqrt(rnorm) / bnorm < params_.cg_tol) {
            it++;
            break;
        }
        applyPrecond(); // z_ = M^{-1} r_
        double rznew = 0.0;
#pragma omp parallel for reduction(+ : rznew) schedule(static) num_threads(nthreads_)
        for (int c = 0; c < N; c++)
            rznew += r_[c] * z_[c];
        double beta = rznew / (rz + 1e-300);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
        for (int c = 0; c < N; c++)
            p_[c] = z_[c] + beta * p_[c];
        rz = rznew;
    }
    return it;
}

void SoilSolver3D::setParams(const Params &p) {
    // Hardware-independent first, so the message names the actual problem on any machine
    if (p.use_gpu && p.nonlinear_method == NonlinearMethod::M_SCHEME)
        helios::helios_runtime_error("ERROR (SoilSolver3D::setParams): NonlinearMethod::M_SCHEME runs on the CPU only; set use_gpu = false or choose another nonlinear method.");
#ifndef HELIOS_CUDA_AVAILABLE
    if (p.use_gpu)
        helios::helios_runtime_error("ERROR (SoilSolver3D::setParams): Params::use_gpu is set, but this build of Helios was compiled without CUDA. Install the CUDA toolkit and rebuild, or "
                                     "leave use_gpu false.");
#else
    if (p.use_gpu && !isGPUAvailable())
        helios::helios_runtime_error("ERROR (SoilSolver3D::setParams): Params::use_gpu is set, but no usable CUDA device was found (or HELIOS_NO_GPU is set). Leave use_gpu false to run on "
                                     "the CPU.");
#endif
    params_ = p;
    nthreads_ = resolveThreadCount(p.num_threads);
}

uint SoilSolver3D::solve(double dt, uint *cg_iterations) {
    if (!(dt > 0.0))
        helios::helios_runtime_error("ERROR (SoilSolver3D::solve): the timestep must be positive (got " + std::to_string(dt) + " s).");
    // Start-of-step state: every failure path below restores it, whatever the nonlinear method (the Picard, NGS-FAS and
    // M-scheme iterations otherwise leave their last, unconverged iterate behind).
    const std::vector<double> h0 = h_, tn0 = theta_n_, tnm10 = theta_nm1_;
    const double dt_prev0 = dt_prev_, influx0 = cumulative_influx_, resid0 = cumulative_balance_residual_;
    const bool hist0 = have_history_;
    auto restore = [&]() {
        h_ = h0;
        theta_n_ = tn0;
        theta_nm1_ = tnm10;
        dt_prev_ = dt_prev0;
        have_history_ = hist0;
        cumulative_influx_ = influx0;
        cumulative_balance_residual_ = resid0;
    };
    try {
        const uint it = solveStep(dt, cg_iterations);
        if (it > 0)
            return it;
        restore();
        if (params_.nonlinear_method != NonlinearMethod::NEWTON_ANALYTIC || !params_.substep_on_failure || dt <= params_.dt_min)
            return 0;
        // The step failed even after the fallbacks (typically a cell pinned just below saturation in a soil with
        // n < 2, where dK/dh is unbounded). Rather than skip the interval, sub-step it adaptively; if even that
        // cannot cover it, restore the start-of-step state and report the failure.
        const uint64_t iters0 = total_picard_iters_;
        // Sub-steps stop at kRescueMinFraction of the interval (as in SoilColumn): one that needs smaller ones usually
        // has no solution, and halving down to dt_min would cost many failed solves before the caller can react.
        struct DtMinScope { // raises Params::dt_min for the rescue, and restores it however the rescue ends
            double &dt_min;
            const double saved;
            DtMinScope(double &value, double raised) : dt_min(value), saved(value) {
                dt_min = raised;
            }
            ~DtMinScope() {
                dt_min = saved;
            }
        } dt_min_scope(params_.dt_min, std::max(params_.dt_min, kRescueMinFraction * dt));
        double sub = 0.25 * dt;
        const double advanced = solveAdaptive(dt, sub);
        if (advanced >= dt * (1.0 - 1e-12))
            return static_cast<uint>(std::max<uint64_t>(1, total_picard_iters_ - iters0));
        restore();
        return 0;
    } catch (...) {
        restore(); // e.g. a CUDA failure part-way through the step
        releaseGpuState(); // device state may be half-built; it is rebuilt on the next GPU solve
        throw;
    }
}

uint SoilSolver3D::solveStep(double dt, uint *cg_iterations) {
    const bool general_grid = !uniform_geometry_ || n_fixed_ > 0 || has_ss_;
    if (general_grid && params_.nonlinear_method != NonlinearMethod::NEWTON_ANALYTIC)
        helios::helios_runtime_error("ERROR (SoilSolver3D::solve): setCellGeometry(), setFixedHead() and setSpecificStorage() require NonlinearMethod::NEWTON_ANALYTIC.");
    // Dispatch to the Newton-Krylov or Picard->Newton path when selected (CPU only; the GPU
    // backend below is Picard-only). PICARD (default) falls through to the loop unchanged.
    if (!params_.use_gpu) {
        if (params_.nonlinear_method == NonlinearMethod::NEWTON)
            return solveNewton(dt, cg_iterations);
        if (params_.nonlinear_method == NonlinearMethod::PICARD_NEWTON)
            return solveNonlinearSwitch(dt, cg_iterations);
        if (params_.nonlinear_method == NonlinearMethod::NGS_FAS)
            return solveNGSFAS(dt, cg_iterations);
        if (params_.nonlinear_method == NonlinearMethod::M_SCHEME)
            return solveMScheme(dt, cg_iterations);
        if (params_.nonlinear_method == NonlinearMethod::NEWTON_ANALYTIC) {
            // Newton is fast where it converges. On the rare step where it stalls (a sharp wetting front
            // crossing several cells in one long step, or a cell pinned at the saturation kink of a soil
            // with n < 2) fall back to the globally robust Modified-Picard iteration below for this step,
            // starting again from h^n. Inside solveAdaptive() the fallback is given a small budget: where
            // Picard would also crawl, cutting dt is the cheaper recovery. (The Picard paths assume a
            // uniform grid without specified heads or specific storage.)
            const uint it = solveNewtonAnalytic(dt, cg_iterations);
            if (it > 0 || general_grid)
                return it;
            const std::vector<double> h_start = h_;
            const Params saved = params_;
            if (in_adaptive_) {
                params_.picard_max = std::min<uint>(params_.picard_max, 8);
                params_.cg_max = std::min<uint>(params_.cg_max, 100);
            }
            params_.nonlinear_method = NonlinearMethod::PICARD;
            picard_fallback_ = true;
            const uint pit = solveStep(dt, cg_iterations);
            picard_fallback_ = false;
            params_ = saved;
            if (pit == 0)
                h_ = h_start; // like a failed Newton step, a failed step leaves the state unchanged
            return pit;
        }
    }
    const int N = static_cast<int>(numCells());
    // theta at the start of the step (theta^n)
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        theta_n_[c] = storedWater(c, h_[c]);

    // Time-discretization coefficients:  dθ/dt ≈ a0 θ^{n+1} + a1 θ^n + a2 θ^{n-1}.
    double a0, a1, a2;
    if (params_.bdf2 && have_history_ && dt_prev_ > 0.0) {
        double r = dt / dt_prev_; // variable-step BDF2
        a0 = (1.0 + 2.0 * r) / ((1.0 + r) * dt);
        a1 = -(1.0 + r) / dt;
        a2 = (r * r) / ((1.0 + r) * dt);
    } else { // backward Euler (also the BDF2 bootstrap step)
        a0 = 1.0 / dt;
        a1 = -1.0 / dt;
        a2 = 0.0;
    }

#ifdef HELIOS_CUDA_AVAILABLE
    if (params_.use_gpu) {
        const int N = (int) numCells();
        std::vector<double> tn(theta_n_), tnm1(theta_nm1_); // pre-solve theta^n, theta^{n-1} (solveGPU advances them)
        const std::vector<double> h_start = h_;
        const double dt_prev0 = dt_prev_;
        const bool hist0 = have_history_;
        uint res = solveGPU(dt, a0, a1, a2, cg_iterations);
        const bool fallback = picard_fallback_;
        picard_fallback_ = false;
        if (res != 0) { // instantaneous discrete mass-balance residual (quadrature-free), matching commitStep()
            double smass = 0.0, volume = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : smass, volume) num_threads(nthreads_)
            for (int c = 0; c < N; c++) {
                smass += (a0 * storedWater(c, h_[c]) + a1 * tn[c] + a2 * tnm1[c]) * cellVolume(c);
                volume += cellVolume(c);
            }
            const double influx = stepInflux();
            // A converged Picard fallback of a failed Newton step must also close the water balance (as on the CPU)
            if (fallback && !(std::fabs(smass - influx) * dt <= kFallbackBalanceTol * volume)) {
                h_ = h_start;
                theta_n_ = tn;
                theta_nm1_ = tnm1;
                dt_prev_ = dt_prev0;
                have_history_ = hist0;
                return 0;
            }
            cumulative_influx_ += influx * dt;
            cumulative_balance_residual_ += (smass - influx) * dt;
        }
        return res;
    }
#endif

    hm_ = h_; // Picard iterate
    uint total_cg = 0;
    const int mAA = std::min(std::max(0, params_.anderson_m), 8); // Anderson depth (0 = plain Picard)
    if (mAA > 0) {
        aa_dF_.clear();
        aa_dG_.clear();
        aa_f_.assign(N, 0.0);
    }
    bool aa_have_prev = false;
    for (uint p = 0; p < params_.picard_max; p++) {
        updateKC(hm_);
        buildSystem(hm_, a0, a1, a2);
        if (params_.use_multigrid)
            mgBuild(a0);
        u_ = hm_;
        total_cg += cg(u_, a0); // u_ = G(hm_): the linearized (modified-Picard) solve
        bool finite = true;
        for (int c = 0; c < N && finite; c++)
            finite = std::isfinite(u_[c]);
        if (!finite) // NaN slips through the max|dh| test below; a non-finite iterate never converges
            break;
        double dmax = 0.0;
        if (mAA > 0) {
            for (int c = 0; c < N; c++) {
                aa_f_[c] = u_[c] - hm_[c];
                dmax = std::max(dmax, std::fabs(aa_f_[c]));
            }
        } else {
            for (int c = 0; c < N; c++)
                dmax = std::max(dmax, std::fabs(u_[c] - hm_[c]));
        }
        if (dmax < params_.picard_tol) {
            hm_ = u_; // accept the exact linearized solve (mass-conservative), not the AA extrapolation
            h_ = hm_; // converged head; set before stepInflux()/residual read it
            // Mass-balance bookkeeping. At the converged state the scheme enforces,
            // summed over cells, Vcell*(a0 theta^{n+1} + a1 theta^n + a2 theta^{n-1})
            // = external influx rate, with the same (a0,a1,a2) used this step. Compute
            // the BDF2/BE storage rate here (while theta_n_=theta^n, theta_nm1_=theta^{n-1}
            // are still pre-swap) and accumulate both the influx and the conservation
            // defect (~0 to Picard tolerance regardless of integrator).
            double smass = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : smass) num_threads(nthreads_)
            for (int c = 0; c < N; c++)
                smass += a0 * waterContent<double>(h_[c], props_[c]) + a1 * theta_n_[c] + a2 * theta_nm1_[c];
            smass *= Vcell_;
            const double influx = stepInflux();
            // As the fallback of a failed Newton step, also require the water balance to close (as in SoilColumn):
            // damped or extrapolated iterates can stop changing well away from the solution.
            if (picard_fallback_ && !(std::fabs(smass - influx) * dt <= kFallbackBalanceTol * Vcell_ * N))
                break;
            cumulative_influx_ += influx * dt;
            cumulative_balance_residual_ += (smass - influx) * dt;
            // Advance history (theta^{n-1} <- theta^n, theta^n <- theta^{n+1}).
            theta_nm1_.swap(theta_n_);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
            for (int c = 0; c < N; c++)
                theta_n_[c] = waterContent<double>(hm_[c], props_[c]);
            dt_prev_ = dt;
            have_history_ = true;
            if (cg_iterations)
                *cg_iterations = total_cg;
            return p + 1;
        }
        andersonUpdate(N, mAA, aa_have_prev); // form the next iterate (plain Picard if mAA==0)
    }
    bool finite = true;
    for (int c = 0; c < N && finite; c++)
        finite = std::isfinite(hm_[c]);
    if (finite)
        h_ = hm_; // best effort (does not advance history)
    if (cg_iterations)
        *cg_iterations = total_cg;
    return 0; // did not converge
}

// Mass-balance bookkeeping + BDF2 history advance for a converged step. h_ holds the
// converged head; theta_n_/theta_nm1_ are still pre-swap (theta^n, theta^{n-1}). Shared by
// the Newton and Picard->Newton paths so their conservation accounting is identical to solve().
void SoilSolver3D::commitStep(double dt, double a0, double a1, double a2) {
    const int N = static_cast<int>(numCells());
    double smass = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : smass) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        smass += (a0 * storedWater(c, h_[c]) + a1 * theta_n_[c] + a2 * theta_nm1_[c]) * cellVolume(c);
    const double influx = stepInflux();
    cumulative_influx_ += influx * dt;
    cumulative_balance_residual_ += (smass - influx) * dt;
    theta_nm1_.swap(theta_n_);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        theta_n_[c] = storedWater(c, h_[c]);
    dt_prev_ = dt;
    have_history_ = true;
}

// Cheap approximate A^{-1} r for the GMRES preconditioner: a few multigrid V-cycles (or Jacobi
// sweeps) about the frozen Modified-Picard operator. Using an inexact preconditioner -- instead
// of a fully converged cg solve inside every Krylov iteration -- is what makes JFNK affordable.
// Requires the frozen operator (updateKC + buildSystem + mgBuild) to be current at hm_.
void SoilSolver3D::applyPicardPrecond(const std::vector<double> &r, std::vector<double> &z, double a0) {
    const int N = static_cast<int>(numCells());
    const int sweeps = std::max(1, params_.newton_precond_sweeps);
    z.assign(N, 0.0);
    std::vector<double> Az, res, dz;
    for (int s = 0; s < sweeps; s++) {
        if (s == 0) {
            if (params_.use_multigrid) {
                mgApply(z, r); // z = M^{-1} r (one V-cycle)
            } else {
#pragma omp parallel for schedule(static) num_threads(nthreads_)
                for (int c = 0; c < N; c++)
                    z[c] = r[c] / diag_[c]; // Jacobi
            }
        } else { // stationary correction: z += M^{-1}(r - A z)
            Az.resize(N);
            res.resize(N);
            dz.resize(N);
            matvec(z, Az, a0);
            for (int c = 0; c < N; c++)
                res[c] = r[c] - Az[c];
            if (params_.use_multigrid) {
                mgApply(dz, res);
                for (int c = 0; c < N; c++)
                    z[c] += dz[c];
            } else {
#pragma omp parallel for schedule(static) num_threads(nthreads_)
                for (int c = 0; c < N; c++)
                    z[c] += res[c] / diag_[c];
            }
        }
    }
}

// Inexact Newton-Krylov step. Outer Newton on the nonlinear residual R(h)=0; inner GMRES
// solves J(h) delta = -R with the matrix-free Jacobian (jacvec) preconditioned by one frozen
// Modified-Picard linear solve (cg). Armijo line search globalizes; Eisenstat-Walker forcing
// tightens the inner tol as the residual falls (recovering quadratic convergence near the root).
uint SoilSolver3D::solveNewton(double dt, uint *gmres_iterations) {
    const int N = static_cast<int>(numCells());
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        theta_n_[c] = waterContent<double>(h_[c], props_[c]); // theta^n

    double a0, a1, a2;
    timeWeights(dt, a0, a1, a2);

    auto l2 = [&](const std::vector<double> &x) {
        double s = 0.0;
        for (int c = 0; c < N; c++)
            s += x[c] * x[c];
        return std::sqrt(s);
    };

    std::vector<double> h = h_; // Newton iterate
    std::vector<double> F, Ft, delta(N), rhs(N), htrial(N);
    uint total_gmres = 0;

    residual(h, F, a0, a1, a2);
    double fnorm = l2(F);
    const double ftol = params_.newton_rtol * fnorm + 1e-300;
    double eta = 0.1; // Eisenstat-Walker forcing term (inner relative tolerance)
    newton_resid_history_.clear();

    for (uint it = 0; it < params_.newton_max; it++) {
        newton_resid_history_.push_back(fnorm);
        if (fnorm <= ftol) { // converged
#pragma omp parallel for schedule(static) num_threads(nthreads_)
            for (int c = 0; c < N; c++)
                h_[c] = h[c];
            commitStep(dt, a0, a1, a2);
            if (gmres_iterations)
                *gmres_iterations = total_gmres;
            return it == 0 ? 1u : it;
        }

        // Freeze the Modified-Picard preconditioner operator A at the current iterate h.
        updateKC(h);
        buildSystem(h, a0, a1, a2);
        hm_ = h; // matvec()/cg() read hm_ for the upstream total heads
        if (params_.use_multigrid)
            mgBuild(a0);

        // Inner solve: J(h) delta = -F, GMRES with the true matrix-free Jacobian and the
        // Picard solve as preconditioner. jacvec is self-contained so it never clobbers the
        // frozen K_/C_/diag_/hm_ the preconditioner depends on.
        auto applyJ = [&](const std::vector<double> &v, std::vector<double> &Jv) { jacvec(h, F, v, Jv, a0, a1, a2); };
        auto precM = [&](const std::vector<double> &r, std::vector<double> &z) { applyPicardPrecond(r, z, a0); };
        for (int c = 0; c < N; c++)
            rhs[c] = -F[c];
        std::fill(delta.begin(), delta.end(), 0.0);
        total_gmres += gmres(applyJ, precM, rhs, delta, params_.gmres_restart, params_.newton_gmres_max, eta);

        // Armijo backtracking line search on ||R|| (globalization -> Picard-like robustness).
        double lambda = 1.0, ftnorm = fnorm;
        bool ok = false;
        for (int ls = 0; ls < 8; ls++) {
            for (int c = 0; c < N; c++)
                htrial[c] = h[c] + lambda * delta[c];
            residual(htrial, Ft, a0, a1, a2);
            ftnorm = l2(Ft);
            if (ftnorm < (1.0 - 1e-4 * lambda) * fnorm) {
                ok = true;
                break;
            }
            lambda *= 0.5;
        }
        h.swap(htrial); // accept (best effort even if Armijo never satisfied)
        F.swap(Ft);
        const double fnorm_prev = fnorm;
        fnorm = ftnorm;
        if (fnorm_prev > 1e-300) { // Eisenstat-Walker choice 2
            const double e = 0.9 * (fnorm / fnorm_prev) * (fnorm / fnorm_prev);
            eta = std::min(0.9, std::max(1e-6, e));
        }
        if (!ok && lambda <= (1.0 / 256.0)) { // line search collapsed: no progress possible
            h_ = h;
            if (gmres_iterations)
                *gmres_iterations = total_gmres;
            return 0;
        }
    }
    h_ = h; // best effort (no history advance)
    if (gmres_iterations)
        *gmres_iterations = total_gmres;
    return 0; // did not converge
}

// ============================================================================
// Newton-Raphson with an assembled analytic Jacobian (NEWTON_ANALYTIC)
// ============================================================================

// Residual F(h) and, optionally, its 7-point Jacobian. F is the same discrete residual as
// residual() (storage rate + face fluxes + sources, per unit volume [1/s]); every face flux is
// written as Kf * g * (H_c - H_n) in total head H = h + z, so d/dh_c and d/dh_n follow directly,
// including the dKf/dh terms. Constitutive values are evaluated once per cell rather than per face.
void SoilSolver3D::assembleNewton(const std::vector<double> &h, std::vector<double> &F, double a0, double a1, double a2, bool with_jacobian) {
    const int N = static_cast<int>(numCells());
    const int Nx = Nx_, Ny = Ny_, Nz = Nz_, NxNy = Nx * Ny;
    const double idx2 = 1.0 / (dx_ * dx_), idy2 = 1.0 / (dy_ * dy_);
    const double idx = 1.0 / dx_, idy = 1.0 / dy_;
    const int w = params_.K_weighting;
    const bool par = N > kOmpMinCells;
    F.resize(N);
    nk_K_.resize(N);
    nk_dK_.resize(N);
    nk_C_.resize(N);
    nk_theta_.resize(N);
    nk_scale_.resize(N);
    nk_ext_.resize(N);
    if (with_jacobian) {
        for (auto *v: {&jd_, &jxm_, &jxp_, &jym_, &jyp_, &jzm_, &jzp_})
            v->assign(N, 0.0);
    }

#pragma omp parallel for schedule(static) if (par) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        constitutiveAll(h[c], props_[c], params_.C_min, params_.K_min, nk_theta_[c], nk_K_[c], nk_C_[c], nk_dK_[c]);
        addSpecificStorage(h[c], ss_[c], props_[c].theta_s, nk_theta_[c], nk_C_[c]);
    }

    // Per unit volume of cell c, a face with area A at centre distance L contributes A / (L V_c) times
    // Kf (H_c - H_n). With per-cell centre elevations z_c and thicknesses t_c (uniform dx, dy): vertical
    // faces A/(L V) = 1 / (L t_c), lateral faces (area dy * mean thickness, distance dx) give
    // mean(t_c, t_n) / (dx^2 t_c). Total head H = h + z_c, so sloping layers carry their gravity term.
#pragma omp parallel for schedule(static) if (par) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const SoilProperties &pc = props_[c];
        const double hc = h[c], Kc = nk_K_[c], dKc = nk_dK_[c], tc = thk_[c], zc = zc_[c];
        const double ext_src = src_[c] - uptake_[c] - evap_[c];
        double R = a0 * nk_theta_[c] + a1 * theta_n_[c] + a2 * theta_nm1_[c] - ext_src;
        double scale = std::fabs(a0 * nk_theta_[c]) + std::fabs(a1 * theta_n_[c]) + std::fabs(a2 * theta_nm1_[c]) + std::fabs(ext_src);
        double ext = ext_src; // external inflow rate per unit volume (sources + boundary faces)
        double interior_out = 0.0; // net outflow to neighbouring cells (what a specified-head cell supplies)
        double Jd = a0 * nk_C_[c];

        // Interior face to neighbor n with geometric factor g.
        auto face = [&](int n, double g, double &Jn) {
            const double Hn = h[n] + (zc_[n] - zc), dH = hc - Hn;
            double dA, dB;
            const double Kf = interblockKDeriv(w, Kc, nk_K_[n], hc, Hn, pc.K_s, props_[n].K_s, dA, dB);
            R += g * Kf * dH;
            interior_out += g * Kf * dH;
            scale += std::fabs(g * Kf * dH) + kNewtonRoundoffScale * g * Kf * (std::fabs(hc) + std::fabs(Hn));
            Jd += g * (Kf + dA * dKc * dH);
            Jn = g * (-Kf + dB * nk_dK_[n] * dH);
        };
        // Dirichlet face at half-cell distance; off = z_c - z_boundary_node (head units).
        auto dirichlet = [&](double hbc, double g, double off) {
            double dA, dB;
            const double Kf = interblockKDeriv(w, Kc, Kof(hbc, pc, params_.K_min), hc, hbc - off, pc.K_s, pc.K_s, dA, dB);
            const double cf = 2.0 * Kf * g, dcf = 2.0 * g * dA * dKc;
            R += cf * (hc - hbc + off);
            ext -= cf * (hc - hbc + off);
            scale += std::fabs(cf * (hc - hbc + off)) + kNewtonRoundoffScale * cf * (std::fabs(hc) + std::fabs(hbc - off));
            Jd += cf + dcf * (hc - hbc + off);
        };
        auto neumann = [&](double q, double per_len) {
            R -= q * per_len;
            ext += q * per_len;
        };
        auto lateral = [&](int n, double ig2) { return 0.5 * (tc + thk_[n]) * ig2 / tc; };
        auto vertical = [&](int n) { return 1.0 / (std::fabs(zc_[n] - zc) * tc); };

        double Jxm = 0.0, Jxp = 0.0, Jym = 0.0, Jyp = 0.0, Jzm = 0.0, Jzp = 0.0;
        if (i > 0)
            face(c - 1, lateral(c - 1, idx2), Jxm);
        else if (bc_[0].type == BCType::DIRICHLET)
            dirichlet(bc_[0].value, idx2, 0.0);
        else if (bc_[0].type == BCType::NEUMANN)
            neumann(bc_[0].value, idx);
        if (i < Nx - 1)
            face(c + 1, lateral(c + 1, idx2), Jxp);
        else if (bc_[1].type == BCType::DIRICHLET)
            dirichlet(bc_[1].value, idx2, 0.0);
        else if (bc_[1].type == BCType::NEUMANN)
            neumann(bc_[1].value, idx);
        if (j > 0)
            face(c - Nx, lateral(c - Nx, idy2), Jym);
        else if (bc_[2].type == BCType::DIRICHLET)
            dirichlet(bc_[2].value, idy2, 0.0);
        else if (bc_[2].type == BCType::NEUMANN)
            neumann(bc_[2].value, idy);
        if (j < Ny - 1)
            face(c + Nx, lateral(c + Nx, idy2), Jyp);
        else if (bc_[3].type == BCType::DIRICHLET)
            dirichlet(bc_[3].value, idy2, 0.0);
        else if (bc_[3].type == BCType::NEUMANN)
            neumann(bc_[3].value, idy);
        if (k > 0)
            face(c - NxNy, vertical(c - NxNy), Jzm);
        else if (bc_[4].type == BCType::DIRICHLET)
            dirichlet(bc_[4].value, 1.0 / (tc * tc), 0.5 * tc);
        else if (bc_[4].type == BCType::NEUMANN)
            neumann(bc_[4].value, 1.0 / tc);
        else if (bc_[4].type == BCType::FREE_DRAINAGE) {
            R += Kc / tc;
            ext -= Kc / tc;
            Jd += dKc / tc;
        }
        if (k < Nz - 1)
            face(c + NxNy, vertical(c + NxNy), Jzp);
        else if (bc_[5].type == BCType::DIRICHLET)
            dirichlet(bc_[5].value, 1.0 / (tc * tc), -0.5 * tc);
        else if (bc_[5].type == BCType::NEUMANN)
            neumann(bc_[5].value, 1.0 / tc);
        else if (bc_[5].type == BCType::FREE_DRAINAGE) {
            R += -Kc / tc;
            ext += Kc / tc;
            Jd += -dKc / tc;
        }

        if (n_fixed_ > 0 && !std::isnan(fixed_h_[c])) {
            // Specified-head cell: the row becomes h_c = h_fixed. Its storage is constant, so the water
            // budget of the domain sees only what it passes to its neighbours (its own recharge and boundary
            // faces are absorbed by the specified head).
            nk_ext_[c] = interior_out;
            F[c] = hc - fixed_h_[c];
            nk_scale_[c] = 1.0;
            if (with_jacobian) {
                jd_[c] = 1.0;
                jxm_[c] = jxp_[c] = jym_[c] = jyp_[c] = jzm_[c] = jzp_[c] = 0.0;
            }
            continue;
        }
        F[c] = R;
        nk_scale_[c] = scale;
        nk_ext_[c] = ext;
        if (with_jacobian) {
            jd_[c] = Jd;
            jxm_[c] = Jxm;
            jxp_[c] = Jxp;
            jym_[c] = Jym;
            jyp_[c] = Jyp;
            jzm_[c] = Jzm;
            jzp_[c] = Jzp;
        }
    }
}

// ILU(0) of the 7-point Jacobian in natural (x-fastest) ordering. With zero fill the L and U
// off-diagonals are the matrix entries themselves; only the pivots change. Exact LU in 1D.
// Large grids use block-Jacobi ILU(0) with one block per thread: the grid is split into slabs along y
// (or x when Ny is too small), keeping every vertical column whole, and the couplings across slab
// interfaces are dropped. Vertical coupling usually dominates (layers are thin relative to their lateral
// extent), so cutting columns -- z-slabs -- would discard the strongest connections. Grids below
// kOmpMinCells use a single block, i.e. plain (exact-in-1D) ILU(0).
void SoilSolver3D::buildIluBlocks() {
    int nblk = 1;
#ifdef _OPENMP
    if ((int) numCells() > kOmpMinCells)
        nblk = nthreads_;
#endif
    ilu_dim_ = (Ny_ >= Nx_) ? 1 : 0; // split along the longer lateral dimension
    const int L = ilu_dim_ == 1 ? (int) Ny_ : (int) Nx_;
    nblk = std::max(1, std::min(nblk, L / 2));
    ilu_blk_.assign(nblk + 1, 0);
    for (int b = 0; b <= nblk; b++)
        ilu_blk_[b] = (int) ((long) L * b / nblk); // block b covers rows (or columns) [ilu_blk_[b], ilu_blk_[b+1])
    ilu_blk_n_ = (int) numCells();
    ilu_blk_threads_ = nthreads_;
}

void SoilSolver3D::ilu0Factor() {
    const int N = static_cast<int>(numCells()), Nx = Nx_, NxNy = Nx_ * Ny_;
    if (ilu_blk_.empty() || ilu_blk_n_ != N || ilu_blk_threads_ != nthreads_)
        buildIluBlocks();
    ilu_d_.resize(N);
    const int nblk = (int) ilu_blk_.size() - 1;
    const bool by_y = ilu_dim_ == 1;
#pragma omp parallel for schedule(static, 1) if (nblk > 1) num_threads(nthreads_)
    for (int b = 0; b < nblk; b++) {
        const int a0 = ilu_blk_[b], a1 = ilu_blk_[b + 1];
        const int j0 = by_y ? a0 : 0, j1 = by_y ? a1 : (int) Ny_, i0 = by_y ? 0 : a0, i1 = by_y ? Nx : a1;
        for (int k = 0; k < (int) Nz_; k++)
            for (int j = j0; j < j1; j++)
                for (int i = i0; i < i1; i++) {
                    const int c = i + Nx * j + NxNy * k;
                    double d = jd_[c];
                    if (i > i0)
                        d -= jxm_[c] * jxp_[c - 1] / ilu_d_[c - 1];
                    if (j > j0)
                        d -= jym_[c] * jyp_[c - Nx] / ilu_d_[c - Nx];
                    if (k > 0)
                        d -= jzm_[c] * jzp_[c - NxNy] / ilu_d_[c - NxNy];
                    if (std::fabs(d) < 1e-300)
                        d = (d < 0.0) ? -1e-300 : 1e-300;
                    ilu_d_[c] = d;
                }
    }
}

void SoilSolver3D::ilu0Solve(const std::vector<double> &r, std::vector<double> &z) const {
    const int N = static_cast<int>(numCells()), Nx = Nx_, NxNy = Nx_ * Ny_;
    z.resize(N);
    const int nblk = (int) ilu_blk_.size() - 1;
    const bool by_y = ilu_dim_ == 1;
#pragma omp parallel for schedule(static, 1) if (nblk > 1) num_threads(nthreads_)
    for (int b = 0; b < nblk; b++) {
        const int a0 = ilu_blk_[b], a1 = ilu_blk_[b + 1];
        const int j0 = by_y ? a0 : 0, j1 = by_y ? a1 : (int) Ny_, i0 = by_y ? 0 : a0, i1 = by_y ? Nx : a1;
        for (int k = 0; k < (int) Nz_; k++) // forward: unit-lower L with L_cn = J_cn / d_n
            for (int j = j0; j < j1; j++)
                for (int i = i0; i < i1; i++) {
                    const int c = i + Nx * j + NxNy * k;
                    double y = r[c];
                    if (i > i0)
                        y -= jxm_[c] * z[c - 1] / ilu_d_[c - 1];
                    if (j > j0)
                        y -= jym_[c] * z[c - Nx] / ilu_d_[c - Nx];
                    if (k > 0)
                        y -= jzm_[c] * z[c - NxNy] / ilu_d_[c - NxNy];
                    z[c] = y;
                }
        for (int k = (int) Nz_ - 1; k >= 0; k--) // backward: U with pivots d and off-diagonals J_cn
            for (int j = j1 - 1; j >= j0; j--)
                for (int i = i1 - 1; i >= i0; i--) {
                    const int c = i + Nx * j + NxNy * k;
                    double y = z[c];
                    if (i < i1 - 1)
                        y -= jxp_[c] * z[c + 1];
                    if (j < j1 - 1)
                        y -= jyp_[c] * z[c + Nx];
                    if (k < (int) Nz_ - 1)
                        y -= jzp_[c] * z[c + NxNy];
                    z[c] = y / ilu_d_[c];
                }
    }
}

void SoilSolver3D::jacobianApply(const std::vector<double> &v, std::vector<double> &out) const {
    const int N = static_cast<int>(numCells()), Nx = Nx_, NxNy = Nx_ * Ny_;
    out.resize(N);
#pragma omp parallel for schedule(static) if (N > kOmpMinCells) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        double s = jd_[c] * v[c];
        if (jxm_[c] != 0.0)
            s += jxm_[c] * v[c - 1];
        if (jxp_[c] != 0.0)
            s += jxp_[c] * v[c + 1];
        if (jym_[c] != 0.0)
            s += jym_[c] * v[c - Nx];
        if (jyp_[c] != 0.0)
            s += jyp_[c] * v[c + Nx];
        if (jzm_[c] != 0.0)
            s += jzm_[c] * v[c - NxNy];
        if (jzp_[c] != 0.0)
            s += jzp_[c] * v[c + NxNy];
        out[c] = s;
    }
}

// Right-preconditioned BiCGSTAB (van der Vorst 1992) with the ILU(0) factors. x in/out.
uint SoilSolver3D::bicgstab(const std::vector<double> &b, std::vector<double> &x, double rtol, uint maxit) {
    const int N = static_cast<int>(numCells());
    const bool par = N > kOmpMinCells;
    auto dot = [&](const std::vector<double> &a, const std::vector<double> &c) {
        double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static) if (par) num_threads(nthreads_)
        for (int i = 0; i < N; i++)
            s += a[i] * c[i];
        return s;
    };
    std::vector<double> r(N), rhat(N), p(N, 0.0), v(N, 0.0), phat(N), s(N), shat(N), t(N);
    jacobianApply(x, t);
    for (int i = 0; i < N; i++)
        r[i] = b[i] - t[i];
    rhat = r;
    const double bnorm = std::sqrt(dot(b, b));
    if (bnorm < 1e-300)
        return 0;
    const double thresh = rtol * bnorm;
    if (std::sqrt(dot(r, r)) <= thresh)
        return 0;
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    uint it = 0;
    while (it < maxit) {
        it++;
        const double rho_new = dot(rhat, r);
        if (std::fabs(rho_new) < 1e-300)
            break; // breakdown
        const double beta = (rho_new / rho) * (alpha / omega);
#pragma omp parallel for schedule(static) if (par) num_threads(nthreads_)
        for (int i = 0; i < N; i++)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        ilu0Solve(p, phat);
        jacobianApply(phat, v);
        alpha = rho_new / dot(rhat, v);
#pragma omp parallel for schedule(static) if (par) num_threads(nthreads_)
        for (int i = 0; i < N; i++)
            s[i] = r[i] - alpha * v[i];
        if (std::sqrt(dot(s, s)) <= thresh) {
            for (int i = 0; i < N; i++)
                x[i] += alpha * phat[i];
            break;
        }
        ilu0Solve(s, shat);
        jacobianApply(shat, t);
        const double tt = dot(t, t);
        omega = (tt > 1e-300) ? dot(t, s) / tt : 0.0;
#pragma omp parallel for schedule(static) if (par) num_threads(nthreads_)
        for (int i = 0; i < N; i++) {
            x[i] += alpha * phat[i] + omega * shat[i];
            r[i] = s[i] - omega * t[i];
        }
        if (std::sqrt(dot(r, r)) <= thresh || omega == 0.0)
            break;
        rho = rho_new;
    }
    return it;
}

uint SoilSolver3D::solveNewtonAnalytic(double dt, uint *linear_iterations) {
    const int N = static_cast<int>(numCells());
#pragma omp parallel for schedule(static) if (N > kOmpMinCells) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        theta_n_[c] = storedWater(c, h_[c]); // stored water at t^n

    double a0, a1, a2;
    timeWeights(dt, a0, a1, a2);

    // Residual norm with each cell's residual (per unit volume) weighted by its volume relative to the mean, i.e.
    // in water-flux units as in SoilColumn. On a uniform grid this is the plain L2 norm; on a graded grid it keeps
    // very thin cells, which hold almost no water, from dominating the line search.
    double vbar = 0.0;
    for (int c = 0; c < N; c++)
        vbar += cellVolume(c);
    vbar /= N;
    auto l2 = [&](const std::vector<double> &x) {
        double s = 0.0;
        for (int c = 0; c < N; c++) {
            const double v = x[c] * (uniform_geometry_ ? 1.0 : cellVolume(c) / vbar);
            s += v * v;
        }
        return std::sqrt(s);
    };

    if (n_fixed_ > 0)
        for (int c = 0; c < N; c++)
            if (!std::isnan(fixed_h_[c]))
                h_[c] = fixed_h_[c];
    std::vector<double> h = h_, F, Ft, delta(N), rhs(N), htrial(N), hbest(N), Fbest, theta_h, C_h;
    uint total_lin = 0;
    newton_resid_history_.clear();
    double eta = 1e-2; // inner relative tolerance (Eisenstat-Walker)
    bool dh_ok = false;
    double max_dh = INFINITY;
    int creep = 0; // consecutive damped iterations with little progress (see kNewtonCreepIterations)

    assembleNewton(h, F, a0, a1, a2, true);
    double fnorm = l2(F);
    for (uint it = 0; it < params_.newton_max; it++) {
        newton_resid_history_.push_back(fnorm);
        if (!std::isfinite(fnorm))
            break;
        // Stalled (less than 1% reduction over the last 6 iterations): give up early so the caller can
        // fall back to Picard or cut dt instead of spending newton_max iterations.
        if (it >= 6 && fnorm > 0.99 * newton_resid_history_[it - 6])
            break;
        // Each cell's residual must be small either in water-content units (its per-step water
        // imbalance) or relative to the gross flux through it. The relative test is what
        // terminates very long (quasi-steady) steps, where a0 ~ 1/dt makes the water-content
        // measure amplify floating-point round-off in large, nearly cancelling face fluxes.
        // res_ok: every cell's per-step water imbalance (theta_tol * a0 ~ theta_tol / dt) or its residual
        // relative to the gross flux through it is negligible. Before the first iteration only the strict
        // relative test can accept the step: on a short step the per-step tolerance is loose enough to
        // pass an unchanged state, and time would advance with the state frozen.
        const double tolA = params_.newton_theta_tol * a0;
        bool res_ok = true, strict_ok = true;
        for (int c = 0; c < N && res_ok; c++) {
            res_ok = std::fabs(F[c]) <= std::max(tolA, kNewtonRelTol * nk_scale_[c]);
            strict_ok = strict_ok && std::fabs(F[c]) <= kNewtonRelTol * nk_scale_[c];
        }
        if ((res_ok && (it == 0 ? strict_ok : dh_ok)) || (it > 0 && max_dh < params_.newton_dvclose)) {
            h_ = h;
            commitStep(dt, a0, a1, a2);
            if (linear_iterations)
                *linear_iterations = total_lin;
            return std::max(1u, it);
        }

        ilu0Factor();
        for (int c = 0; c < N; c++)
            rhs[c] = -F[c];
        std::fill(delta.begin(), delta.end(), 0.0);
        total_lin += bicgstab(rhs, delta, eta, params_.bicgstab_max);

        // The line search re-evaluates constitutive values at trial points, so keep theta and C at h.
        theta_h.assign(nk_theta_.begin(), nk_theta_.end());
        C_h.assign(nk_C_.begin(), nk_C_.end());

        // Trial point h(lambda). Each cell's effective-saturation change per iteration is capped at
        // kMaxDSe (as PFLOTRAN/TOUGH2 truncate Newton updates): unsaturated cells step in
        // water-content space (theta + C*lambda*delta, mapped back through the retention curve), so a
        // cell with C ~ 0 cannot be thrown to an absurd head; a cell reaching saturation stops at
        // h = 0; a saturated cell that starts to drain stops at Se = 1 - kMaxDSe. Without the cap,
        // an incompressible (Ss = 0) saturated column jumps straight to bone-dry heads and
        // individual cells then flip between saturated and dry on alternate iterations.
        // Cells with specific storage that are saturated or on the wet side of the retention curve's
        // inflection take the plain head step instead: their stored water has no kink at saturation, and
        // the water-content map there is ill-conditioned (Se = 0.9999 a few cm below saturation, so any
        // wetting step snaps the cell to h = 0) and holds cells at h = 0 for an extra iteration whenever
        // they cross it. Around a water table that made Newton crawl.
        constexpr double kMaxDSe = 0.2;
        auto trial = [&](double lambda, bool mapped) {
#pragma omp parallel for schedule(static) if (N > kOmpMinCells) num_threads(nthreads_)
            for (int c = 0; c < N; c++) {
                const double d = lambda * delta[c];
                if (n_fixed_ > 0 && !std::isnan(fixed_h_[c])) {
                    htrial[c] = fixed_h_[c];
                    continue;
                }
                if (!mapped || (ss_[c] != 0.0 && h[c] > h_infl_[c])) {
                    htrial[c] = h[c] + d;
                    continue;
                }
                const SoilProperties &p = props_[c];
                const double ts = p.theta_s, tr = p.theta_r, range = ts - tr, hs = h_sat_[c]; // saturated for h >= hs
                if (h[c] < hs) {
                    // Map through the retention curve alone: theta_h and C_h hold the stored water
                    // W = theta (1 + Ss h / theta_s) and dW/dh, so remove the specific-storage part first
                    // (otherwise the trial point does not return to h as lambda -> 0 and backtracking fails).
                    double th = theta_h[c], Cc = C_h[c];
                    if (ss_[c] != 0.0) {
                        const double f = 1.0 + ss_[c] * h[c] / ts;
                        th /= f;
                        Cc = (Cc - ss_[c] * th / ts) / f;
                    }
                    // Tiny steps that stay unsaturated: the mapped and plain updates agree to second order,
                    // and inverting the retention curve would add round-off (~1e-16 theta / C) that a nearly
                    // converged Newton step cannot get below.
                    if (h[c] + d < hs && std::fabs(d) <= 1e-4 * std::max(1.0, -h[c]) && std::fabs(Cc * d) <= 1e-4 * range) {
                        htrial[c] = h[c] + d;
                        continue;
                    }
                    const double Se0 = (th - tr) / range;
                    double Se = (th + Cc * d - tr) / range;
                    Se = std::min(std::max(Se, Se0 - kMaxDSe), Se0 + kMaxDSe);
                    htrial[c] = (Se >= 1.0) ? hs : pressureHeadOfTheta(tr + std::max(Se, 1e-9) * range, p);
                } else {
                    const double hn = h[c] + d;
                    // A saturated cell that starts to drain stops at the saturation head first (where the
                    // retention and conductivity curves have a kink), then takes capped unsaturated steps from there.
                    htrial[c] = (hn < hs) ? (h[c] > hs ? hs : std::max(hn, pressureHeadOfTheta(tr + (1.0 - kMaxDSe) * range, p))) : hn;
                }
            }
        };

        // Backtracking line search on ||F||, first along the truncated/mapped path, then (if that
        // finds no decrease) along the plain Newton step in head: near saturation in soils with
        // n -> 1 the retention curve is so curved that the mapped step stops being a descent
        // direction even though the head step is. If neither decreases ||F||, the best trial is
        // still taken and the outer loop continues. Below
        // newton_backtrack_limit the full truncated step is taken without a line search.
        const bool full_step = fnorm < params_.newton_backtrack_limit;
        double lambda = 1.0, best = INFINITY;
        bool decreased = false, jac_current = false;
        for (int pass = 0; pass < 2 && !decreased; pass++) {
            lambda = 1.0;
            for (int ls = 0; ls < 8; ls++) {
                trial(lambda, pass == 0);
                // The full mapped step is usually accepted: assemble its Jacobian now so it need not
                // be re-assembled after acceptance.
                const bool first = pass == 0 && ls == 0;
                assembleNewton(htrial, Ft, a0, a1, a2, first);
                const double ftn = l2(Ft);
                if (ftn < best) {
                    best = ftn;
                    hbest = htrial;
                    Fbest.swap(Ft);
                }
                if (full_step || ftn <= (1.0 - 1e-4 * lambda) * fnorm) {
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
        max_dh = 0.0;
        for (int c = 0; c < N; c++) {
            // A cell's head has settled when its change is within picard_tol (relative to max(1,|h|)) or
            // moves a negligible amount of water: in very dry cells C ~ 0, so the head can wander far
            // with no effect on the water balance and a pure head test is never met.
            const double dhc = std::fabs(hbest[c] - h[c]);
            max_dh = std::max(max_dh, dhc);
            if (dhc > params_.picard_tol * std::max(1.0, std::fabs(h[c])) && !(h[c] < h_sat_[c] && C_h[c] * dhc < 0.1 * params_.newton_theta_tol))
                dh_ok = false;
        }
        h.swap(hbest);
        const double fnorm_prev = fnorm;
        if (jac_current) {
            F.swap(Fbest);
            fnorm = best;
        } else {
            assembleNewton(h, F, a0, a1, a2, true); // Jacobian at the accepted iterate
            fnorm = l2(F);
        }
        if (fnorm_prev > 1e-300) { // Eisenstat-Walker choice 2
            const double e = 0.9 * (fnorm / fnorm_prev) * (fnorm / fnorm_prev);
            eta = std::min(1e-2, std::max(1e-10, e));
        }
        // Creeping (as in SoilColumn): fail early so the caller can shorten the step
        creep = (!jac_current && fnorm > kNewtonCreepRatio * fnorm_prev) ? creep + 1 : 0;
        if (creep >= kNewtonCreepIterations)
            break;
    }
    if (linear_iterations)
        *linear_iterations = total_lin;
    return 0; // did not converge; h_ unchanged so the caller can retry with a smaller dt
}

// Adaptive Picard -> Newton. Phase A is plain Modified-Picard: ONE cg solve per iteration and
// a cheap max|dh| convergence/stall check -- identical in cost to solve(), so steps that Picard
// handles well (the common case) pay zero Newton overhead. It hands off to Phase B (Newton-Krylov
// with line search + Eisenstat-Walker forcing, preconditioned by a cheap V-cycle) only when
// Picard's per-iteration reduction ratio stalls (>= newton_stall_ratio for newton_stall_count
// consecutive iters) -- i.e. exactly the dry-front rewetting steps where Picard's linear rate
// crawls. Both phases use the same max|dh| < picard_tol convergence criterion as solve().
uint SoilSolver3D::solveNonlinearSwitch(double dt, uint *inner_iterations) {
    const int N = static_cast<int>(numCells());
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        theta_n_[c] = waterContent<double>(h_[c], props_[c]); // theta^n

    double a0, a1, a2;
    timeWeights(dt, a0, a1, a2);

    std::vector<double> h = h_, F, Ft, u(N), delta(N), rhs(N), htrial(N);
    uint inner_total = 0;
    newton_resid_history_.clear();

    bool newton_phase = false;
    double eta = 0.1, dmax_prev = 0.0;
    int slow = 0;
    const double stall_ratio = params_.newton_stall_ratio;
    const int stall_count = std::max(1, params_.newton_stall_count);
    const uint outer_max = params_.picard_max + params_.newton_max;

    for (uint it = 0; it < outer_max; it++) {
        // Freeze the Modified-Picard operator A at the current iterate: Phase A solves with it,
        // Phase B uses it as the GMRES preconditioner.
        updateKC(h);
        buildSystem(h, a0, a1, a2);
        hm_ = h;
        if (params_.use_multigrid)
            mgBuild(a0);

        if (!newton_phase) { // ---- Phase A: one cheap Modified-Picard update ----
            u = h;
            inner_total += cg(u, a0); // u = A^{-1} b
            double dmax = 0.0;
            for (int c = 0; c < N; c++)
                dmax = std::max(dmax, std::fabs(u[c] - h[c]));
            h.swap(u); // accept
            if (dmax < params_.picard_tol) { // converged (same criterion as solve())
#pragma omp parallel for schedule(static) num_threads(nthreads_)
                for (int c = 0; c < N; c++)
                    h_[c] = h[c];
                commitStep(dt, a0, a1, a2);
                if (inner_iterations)
                    *inner_iterations = inner_total;
                return it + 1;
            }
            // Stall detection on the cheap max|dh| ratio (no residual eval).
            if (it >= 1) {
                const double ratio = dmax / (dmax_prev + 1e-300);
                if (ratio > stall_ratio) {
                    if (++slow >= stall_count)
                        newton_phase = true;
                } else
                    slow = 0;
            }
            dmax_prev = dmax;
        } else { // ---- Phase B: one Newton-Krylov step ----
            residual(h, F, a0, a1, a2); // residual only enters here, on stalling steps
            double fnorm = 0.0;
            for (int c = 0; c < N; c++)
                fnorm += F[c] * F[c];
            fnorm = std::sqrt(fnorm);
            newton_resid_history_.push_back(fnorm);
            auto applyJ = [&](const std::vector<double> &v, std::vector<double> &Jv) { jacvec(h, F, v, Jv, a0, a1, a2); };
            auto precM = [&](const std::vector<double> &r, std::vector<double> &z) { applyPicardPrecond(r, z, a0); };
            for (int c = 0; c < N; c++)
                rhs[c] = -F[c];
            std::fill(delta.begin(), delta.end(), 0.0);
            inner_total += gmres(applyJ, precM, rhs, delta, params_.gmres_restart, params_.newton_gmres_max, eta);

            double lambda = 1.0, ftnorm = fnorm;
            bool ok = false;
            for (int ls = 0; ls < 8; ls++) {
                for (int c = 0; c < N; c++)
                    htrial[c] = h[c] + lambda * delta[c];
                residual(htrial, Ft, a0, a1, a2);
                ftnorm = 0.0;
                for (int c = 0; c < N; c++)
                    ftnorm += Ft[c] * Ft[c];
                ftnorm = std::sqrt(ftnorm);
                if (ftnorm < (1.0 - 1e-4 * lambda) * fnorm) {
                    ok = true;
                    break;
                }
                lambda *= 0.5;
            }
            double dmax = 0.0;
            for (int c = 0; c < N; c++)
                dmax = std::max(dmax, std::fabs(htrial[c] - h[c]));
            h.swap(htrial);
            if (fnorm > 1e-300) { // Eisenstat-Walker choice 2
                const double e = 0.9 * (ftnorm / fnorm) * (ftnorm / fnorm);
                eta = std::min(0.9, std::max(1e-6, e));
            }
            if (ok && dmax < params_.picard_tol) { // converged (max head change below tol)
#pragma omp parallel for schedule(static) num_threads(nthreads_)
                for (int c = 0; c < N; c++)
                    h_[c] = h[c];
                commitStep(dt, a0, a1, a2);
                if (inner_iterations)
                    *inner_iterations = inner_total;
                return it + 1;
            }
            if (!ok && lambda <= (1.0 / 256.0)) { // line search collapsed
                h_ = h;
                if (inner_iterations)
                    *inner_iterations = inner_total;
                return 0;
            }
        }
    }
    h_ = h; // best effort (no history advance)
    if (inner_iterations)
        *inner_iterations = inner_total;
    return 0; // did not converge
}

// Nonlinear Gauss-Seidel (Gu & Wang, 2026). The diffusion operator is LAGGED to the start-of-step
// head (K frozen at h^n), so the interblock coupling is a fixed matrix and the ONLY nonlinearity is
// the pointwise theta(psi). Each cell is therefore a scalar monotone equation
//   a0*theta(psi_c) + (xi + sum_f cf_f) psi_c = f_c
// solved by a few Newton steps; neighbours use the newest (Gauss-Seidel) values. No global Krylov
// solve, so no ill-conditioned-Jacobian blowup at sharp dry fronts. Backward Euler (BDF1) in time.
uint SoilSolver3D::solveNGSFAS(double dt, uint *sweeps) {
    const int N = static_cast<int>(numCells());
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++)
        theta_n_[c] = waterContent<double>(h_[c], props_[c]); // theta^n
    const double a0 = 1.0 / dt, a1 = -1.0 / dt; // backward Euler
    const double xi = params_.ngs_xi;

    updateKC(h_); // LAG the diffusion operator: K_[c] = K(h^n), C_[c] = C(h^n)
    buildSystem(h_, a0, a1, 0.0); // fills diag_ (needed by mgBuild's L0.diag)
    mgBuild(a0); // build the multigrid hierarchy with the LAGGED face coefficients

    MGLevel &L0 = mg_[0];
    // Fine-grid RHS g_h = a0*theta^n + xi*h^n + M_h h^n - residual(h^n). At psi=h^n the storage in
    // residual() cancels, leaving the lagged flux + gravity + sinks + BC terms with correct signs.
    std::vector<double> Rn, Mh(N);
    residual(h_, Rn, a0, a1, 0.0);
    mgMatvec(L0, h_, Mh); // (Cdt + diffusion) h^n
    for (int c = 0; c < N; c++)
        L0.b[c] = a0 * theta_n_[c] + xi * h_[c] + (Mh[c] - L0.Cdt[c] * h_[c]) - Rn[c];
    L0.u = h_; // initial iterate

    std::vector<double> uprev(N);
    for (uint cyc = 0; cyc < params_.picard_max; cyc++) {
        uprev = L0.u;
        mgVcycleFAS(0, a0, xi);
        double dmax = 0.0;
        for (int c = 0; c < N; c++)
            dmax = std::max(dmax, std::fabs(L0.u[c] - uprev[c]));
        if (dmax < params_.picard_tol) {
            h_ = L0.u;
            commitStep(dt, a0, a1, 0.0);
            if (sweeps)
                *sweeps = cyc + 1;
            return cyc + 1;
        }
    }
    h_ = L0.u; // best effort
    if (sweeps)
        *sweeps = 0;
    return 0; // did not converge
}

uint SoilSolver3D::solveMScheme(double dt, uint *cg_iterations) {
    // Faithful M-scheme (Javed/Mitra/Pop 2025) in the Kirchhoff b/B splitting. Per outer iteration we
    // eliminate s and solve the SPD, CONSTANT-COEFFICIENT screened Laplacian  [ (L_b/L_B)/tau - D0*Lap ] w_tilde
    // = rhs  (D0 = w_sat), with the gravity advection lagged at theta^{i-1}, then recover s and theta. HOMOGENEOUS
    // soils only (one Kirchhoff table). mu = M*tau is chosen each iteration by the certified estimator (below).
    const int N = (int) numCells();
    const int Nx = (int) Nx_, Ny = (int) Ny_, Nz = (int) Nz_, NxNy = Nx * Ny;

    // HOMOGENEOUS-soil requirement: the Kirchhoff potential w = Phi(theta) is material-dependent, so a single
    // transform table is valid only for one soil (w is discontinuous across material interfaces). Reject
    // heterogeneous properties up front rather than silently solving the wrong system.
    if (props_[0].model == SoilModel::MODIFIED_VAN_GENUCHTEN)
        helios::helios_runtime_error("ERROR (SoilSolver3D::solveMScheme): the M-scheme's Kirchhoff table assumes saturation at h = 0 and does not "
                                     "support SoilModel::MODIFIED_VAN_GENUCHTEN. Use NonlinearMethod::NEWTON_ANALYTIC (the default).");
    for (int c = 1; c < N; c++) {
        const SoilProperties &p0 = props_[0];
        const SoilProperties &pc = props_[c];
        if (pc.model != p0.model || pc.h_e != p0.h_e || pc.theta_r != p0.theta_r || pc.theta_s != p0.theta_s || pc.alpha != p0.alpha || pc.n != p0.n || pc.K_s != p0.K_s) {
            helios::helios_runtime_error("ERROR (SoilSolver3D::solveMScheme): the M-scheme (NonlinearMethod::M_SCHEME) uses the "
                                         "Kirchhoff transform and is only valid for HOMOGENEOUS soil properties. Heterogeneous "
                                         "cells were detected. Use NonlinearMethod::NGS_FAS, PICARD, PICARD_NEWTON, or NEWTON "
                                         "for heterogeneous soils.");
        }
    }
    if (!kirchhoff_built_) {
        kirchhoff_.build(props_[0]);
        kirchhoff_built_ = true;
    }
    const KirchhoffTable &kt = kirchhoff_;
    mscheme_mu_min_ = 0.0;
    mscheme_mu_max_ = 0.0; // reset the estimator's mu-range diagnostic
    const double D0 = kt.w_sat; // constant Laplacian (soil-moisture diffusivity scale) [m^2/s]
    const double a0 = 1.0 / dt; // backward Euler
    double mu = std::max(0.0, params_.m_mu); // M-scheme stabilization mu = M*tau; adapted by the estimator below
    const double cap = 1.0 + params_.m_eps;
    const double idx2 = 1.0 / (dx_ * dx_), idy2 = 1.0 / (dy_ * dy_), idz2 = 1.0 / (dz_ * dz_);

    // theta^n and the starting s^n (from the current head; s encodes both theta and psi 1:1).
    if ((int) s_ms_.size() != N)
        s_ms_.assign(N, 0.0);
    std::vector<double> s(N), thetaN(N);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        s[c] = kt.sOfPsi(h_[c]);
        thetaN[c] = kt.b(s[c]);
        theta_n_[c] = thetaN[c];
    }

    // Dirichlet w_tilde boundary values (bc index: 0 XMIN,1 XMAX,2 YMIN,3 YMAX,4 ZMIN bottom,5 ZMAX top).
    auto isDir = [&](int f) { return bc_[f].type == BCType::DIRICHLET; };
    auto isNeu = [&](int f) { return bc_[f].type == BCType::NEUMANN; };
    double wt_bc[6];
    for (int f = 0; f < 6; f++)
        wt_bc[f] = isDir(f) ? kt.wtildeOfPsi(bc_[f].value) : 0.0;
    auto Kof = [&](double psi) {
        double K, C;
        computeKandC<double>(psi, props_[0], K, C, params_.C_min, params_.K_min);
        return K;
    };
    auto hmean = [](double a, double b) { return (a + b > 0.0) ? 2.0 * a * b / (a + b) : 0.0; };

    std::vector<double> wt(N), LB(N), Adiag(N), rhs(N), Av(N), rcg(N), zcg(N), pcg(N);
    std::vector<double> LbLBv(N), bsp(N), Bsp(N), Kcur(N); // per-cell: L_b/L_B, b(s^{i-1}), B(s^{i-1}), K(s^{i-1})
    for (int c = 0; c < N; c++)
        wt[c] = kt.B(s[c]);
    std::vector<double> sprev(N), wtprev(N);
    uint total_cg = 0, result = 0;

    for (uint it = 0; it < params_.picard_max; it++) {
        sprev = s;
        wtprev = wt;
        // Assemble Adiag, rhs, and L_B from s^{i-1}.
#pragma omp parallel for schedule(static) num_threads(nthreads_)
        for (int c = 0; c < N; c++) {
            const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
            const double psi_c = kt.psiOf(s[c]);
            const double Kc = Kof(psi_c);
            const double bp = kt.bprime(s[c]), Bp = kt.Bprime(s[c]);
            const double Lb = std::min(std::max(bp + mu, 2.0 * mu), cap);
            double Lcap = std::min(std::max(Bp + mu, 2.0 * mu), cap);
            if (Lcap < 1e-300)
                Lcap = 1e-300;
            LB[c] = Lcap;
            const double LbLB = Lb / Lcap;
            LbLBv[c] = LbLB;
            bsp[c] = kt.b(s[c]);
            Bsp[c] = kt.B(s[c]);
            Kcur[c] = Kc; // for s-recovery + estimator
            // Lagged gravity divergence G = (K_top - K_bot)/dz with harmonic face K (or BC).
            double Ktop, Kbot;
            if (k < Nz - 1)
                Ktop = hmean(Kc, Kof(kt.psiOf(s[c + NxNy])));
            else
                Ktop = isDir(5) ? hmean(Kc, Kof(bc_[5].value)) : Kc;
            if (k > 0)
                Kbot = hmean(Kc, Kof(kt.psiOf(s[c - NxNy])));
            else
                Kbot = isDir(4) ? hmean(Kc, Kof(bc_[4].value)) : Kc; // free drainage: outflow = Kc
            const double G = (Ktop - Kbot) / dz_;
            const double f = src_[c] - uptake_[c] - evap_[c];
            double d = LbLB * a0;
            double rb = G + f + (thetaN[c] - kt.b(s[c]) + LbLB * kt.B(s[c])) * a0;
            // Diffusion (constant-coefficient D0 Laplacian): interior +D0/d^2; Dirichlet faces ghost (2x) -> rhs;
            // Neumann faces add their flux to rhs; free-drainage faces contribute no diffusion (zero w_tilde-Neumann).
            auto face = [&](bool interior, int f, double id2) {
                if (interior)
                    d += D0 * id2;
                else if (isDir(f)) {
                    d += 2.0 * D0 * id2;
                    rb += 2.0 * D0 * id2 * wt_bc[f];
                } else if (isNeu(f))
                    rb += bc_[f].value * std::sqrt(id2); // specified flux into the cell [m/s]/d
            };
            face(i > 0, 0, idx2);
            face(i < Nx - 1, 1, idx2);
            face(j > 0, 2, idy2);
            face(j < Ny - 1, 3, idy2);
            face(k > 0, 4, idz2);
            face(k < Nz - 1, 5, idz2);
            Adiag[c] = d;
            rhs[c] = rb;
        }
        // Solve A w_tilde = rhs  (A = Adiag - D0*Laplacian_offdiag), Jacobi-preconditioned CG, warm-started at wt.
        auto matvec = [&](const std::vector<double> &v, std::vector<double> &out) {
#pragma omp parallel for schedule(static) num_threads(nthreads_)
            for (int c = 0; c < N; c++) {
                const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
                double acc = Adiag[c] * v[c];
                if (i > 0)
                    acc -= D0 * idx2 * v[c - 1];
                if (i < Nx - 1)
                    acc -= D0 * idx2 * v[c + 1];
                if (j > 0)
                    acc -= D0 * idy2 * v[c - Nx];
                if (j < Ny - 1)
                    acc -= D0 * idy2 * v[c + Nx];
                if (k > 0)
                    acc -= D0 * idz2 * v[c - NxNy];
                if (k < Nz - 1)
                    acc -= D0 * idz2 * v[c + NxNy];
                out[c] = acc;
            }
        };
        // Preconditioner: geometric multigrid V-cycle on the constant-coefficient screened Laplacian
        // (the ideal MG target) when use_multigrid, else Jacobi. The MG hierarchy uses the current
        // reaction diagonal react = (L_b/L_B)/tau.
        // Inner preconditioner. NOTE (measured): geometric MG on this screened Laplacian cuts the inner CG
        // count ~7x but does NOT improve wall time -- the M-scheme's bottleneck is the OUTER nonlinear
        // iteration count, not the inner solve (that's exactly why the nonlinear-multigrid NGS-FAS wins).
        // So plain Jacobi-PCG (cheap per iter) is the better inner solver here; MG is available via the
        // m_mscheme_mg flag (rebuilt per outer iter, since the reaction diagonal changes) for experiments.
        const bool use_mg = params_.m_mscheme_mg;
        if (use_mg) {
            std::vector<double> react(N);
            for (int c = 0; c < N; c++)
                react[c] = LbLBv[c] * a0;
            mgBuildMScheme(D0, react);
        }
        auto precond = [&](const std::vector<double> &rin, std::vector<double> &zout) {
            if (use_mg)
                mgApply(zout, rin);
            else
                for (int c = 0; c < N; c++)
                    zout[c] = rin[c] / Adiag[c];
        };
        matvec(wt, Av);
        double bnorm = 0.0;
        for (int c = 0; c < N; c++) {
            rcg[c] = rhs[c] - Av[c];
            bnorm += rhs[c] * rhs[c];
        }
        precond(rcg, zcg);
        double rz = 0.0;
        for (int c = 0; c < N; c++) {
            pcg[c] = zcg[c];
            rz += rcg[c] * zcg[c];
        }
        bnorm = std::sqrt(bnorm) + 1e-300;
        uint inner = 0;
        for (; inner < params_.cg_max; inner++) {
            matvec(pcg, Av);
            double pAp = 0.0;
            for (int c = 0; c < N; c++)
                pAp += pcg[c] * Av[c];
            const double alpha = rz / (pAp + 1e-300);
            double rnorm2 = 0.0;
            for (int c = 0; c < N; c++) {
                wt[c] += alpha * pcg[c];
                rcg[c] -= alpha * Av[c];
                rnorm2 += rcg[c] * rcg[c];
            }
            if (std::sqrt(rnorm2) / bnorm < params_.cg_tol) {
                inner++;
                break;
            }
            precond(rcg, zcg);
            double rznew = 0.0;
            for (int c = 0; c < N; c++)
                rznew += rcg[c] * zcg[c];
            const double beta = rznew / (rz + 1e-300);
            for (int c = 0; c < N; c++)
                pcg[c] = zcg[c] + beta * pcg[c];
            rz = rznew;
        }
        total_cg += inner;
        // Recover s^i and the linearized u^i = b(s^{i-1}) + (L_b/L_B)(w_tilde^i - B(s^{i-1})).
        std::vector<double> ulin(N);
        double dmax = 0.0;
        for (int c = 0; c < N; c++) {
            const double ds = (wt[c] - Bsp[c]) / LB[c];
            s[c] = sprev[c] + ds;
            ulin[c] = bsp[c] + LbLBv[c] * (wt[c] - Bsp[c]);
            dmax = std::max(dmax, std::fabs(ds));
        }
        if (dmax < params_.picard_tol) {
            result = it + 1;
            break;
        }

        // --- Certified a-posteriori estimator (eq 4.12) + M-sweep (Algorithm 1) ---
        // Choose mu for the NEXT iteration: the smallest candidate whose predicted linearization error eta(mu)
        // does not exceed the current energy-norm error E_lin -> drives mu -> 0 (Newton) as it becomes safe.
        // All norms dimensionless in our scaled variables: reaction L_b L_B |ds|^2, diffusion tau*D0|grad dw|^2,
        // advection (tau/D0)|dK|^2. Adaptation starts at it>=1 (the advection term needs s^{i-1} and s^{i-2}).
        if (it >= 1) {
            std::vector<double> rw(N), ru(N), bpA(N), BpA(N);
            double Elin2 = 0.0, adv = 0.0;
            for (int c = 0; c < N; c++) {
                const double Lb = LbLBv[c] * LB[c];
                Elin2 += Lb * LB[c] * (s[c] - sprev[c]) * (s[c] - sprev[c]);
                rw[c] = wt[c] - kt.B(s[c]); // w_tilde residual at s^i
                ru[c] = ulin[c] - kt.b(s[c]); // theta residual at s^i
                bpA[c] = kt.bprime(s[c]);
                BpA[c] = kt.Bprime(s[c]); // mu-independent (precompute for the sweep)
                const double dK = Kof(kt.psiOf(s[c])) - Kcur[c]; // F(theta^i) - F(theta^{i-1})
                adv += dK * dK;
            }
            adv *= (dt / D0);
            double diff2 = 0.0; // tau*D0 |grad(dw_tilde)|^2 over +faces
            for (int c = 0; c < N; c++) {
                const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
                const double dwc = wt[c] - wtprev[c];
                if (i < Nx - 1) {
                    double e = (wt[c + 1] - wtprev[c + 1]) - dwc;
                    diff2 += e * e * idx2;
                }
                if (j < Ny - 1) {
                    double e = (wt[c + Nx] - wtprev[c + Nx]) - dwc;
                    diff2 += e * e * idy2;
                }
                if (k < Nz - 1) {
                    double e = (wt[c + NxNy] - wtprev[c + NxNy]) - dwc;
                    diff2 += e * e * idz2;
                }
            }
            const double Elin = std::sqrt(Elin2 + dt * D0 * diff2);
            double mu_new = 1e-2; // fallback: most stable candidate if none is certified safe
            for (int jj = -10; jj <= -2; jj++) {
                const double mc = std::pow(10.0, jj);
                double etap = 0.0, etam = 0.0;
                for (int c = 0; c < N; c++) {
                    const double Lb = std::min(std::max(bpA[c] + mc, 2.0 * mc), cap);
                    double Lc = std::min(std::max(BpA[c] + mc, 2.0 * mc), cap);
                    if (Lc < 1e-300)
                        Lc = 1e-300;
                    const double rr = std::sqrt(Lb / Lc), a = rr * rw[c], b = ru[c] / rr;
                    etap += (a + b) * (a + b);
                    etam += (a - b) * (a - b);
                }
                const double eta = 0.5 * (std::sqrt(etap + adv) + std::sqrt(etam + adv));
                if (eta <= Elin) {
                    mu_new = mc;
                    break;
                } // smallest (most Newton-like) certified-safe mu
            }
            mu = mu_new;
            mscheme_mu_min_ = (mscheme_mu_min_ == 0.0) ? mu : std::min(mscheme_mu_min_, mu);
            mscheme_mu_max_ = std::max(mscheme_mu_max_, mu);
        }
    }

    for (int c = 0; c < N; c++)
        h_[c] = kt.psiOf(s[c]); // recover the pressure head from the splitting variable
    s_ms_ = s;
    if (result) {
        // Mass-balance bookkeeping using the M-scheme's OWN (Kirchhoff) boundary fluxes -- consistent with the
        // flux this scheme enforces, so getMassBalanceResidual() reports the true conservation defect (~0, the
        // scheme is fully implicit + FV-conservative) rather than a spurious Kirchhoff-vs-head mismatch.
        // Backward Euler here (a0 = 1/dt, a1 = -1/dt); theta_n_ still holds theta^n (pre-swap).
        double smass = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : smass) num_threads(nthreads_)
        for (int c = 0; c < N; c++)
            smass += a0 * (waterContent<double>(h_[c], props_[c]) - theta_n_[c]); // BE: a1 = -a0
        smass *= Vcell_;
        const double influx = stepInfluxMScheme();
        cumulative_influx_ += influx * dt;
        cumulative_balance_residual_ += (smass - influx) * dt;
        // Advance history (theta^{n-1} <- theta^n, theta^n <- theta(h)).
        theta_nm1_.swap(theta_n_);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
        for (int c = 0; c < N; c++)
            theta_n_[c] = waterContent<double>(h_[c], props_[c]);
        dt_prev_ = dt;
        have_history_ = true;
    }
    if (cg_iterations)
        *cg_iterations = total_cg;
    return result;
}

void SoilSolver3D::andersonUpdate(int N, int mAA, bool &have_prev) {
    // Plain Picard: next iterate is just the linearized solve.
    if (mAA <= 0) {
        hm_ = u_;
        return;
    }
    // Append the difference columns (Delta f, Delta g) from the previous iterate.
    if (have_prev) {
        std::vector<double> dfc(N), dgc(N);
        for (int c = 0; c < N; c++) {
            dfc[c] = aa_f_[c] - aa_fprev_[c];
            dgc[c] = u_[c] - aa_gprev_[c];
        }
        aa_dF_.push_back(std::move(dfc));
        aa_dG_.push_back(std::move(dgc));
        if (static_cast<int>(aa_dF_.size()) > mAA) {
            aa_dF_.erase(aa_dF_.begin());
            aa_dG_.erase(aa_dG_.begin());
        }
    }
    aa_fprev_ = aa_f_; // f_k
    aa_gprev_ = u_; // g_k = G(x_k)
    have_prev = true;

    const int mc = static_cast<int>(aa_dF_.size());
    if (mc == 0) { // first step: nothing to mix yet
        hm_ = u_;
        return;
    }

    // Least-squares: gamma = argmin || f_k - dF gamma ||  via normal equations
    // (dF^T dF) gamma = dF^T f_k. mc is tiny (<= mAA <= 8), so a small dense solve.
    double M[8][8] = {{0.0}}, rhs[8] = {0.0}, gamma[8] = {0.0};
    for (int i = 0; i < mc; i++) {
        for (int jj = i; jj < mc; jj++) {
            double s = 0.0;
            for (int c = 0; c < N; c++)
                s += aa_dF_[i][c] * aa_dF_[jj][c];
            M[i][jj] = s;
            M[jj][i] = s;
        }
        double rr = 0.0;
        for (int c = 0; c < N; c++)
            rr += aa_dF_[i][c] * aa_f_[c];
        rhs[i] = rr;
    }
    // Tikhonov regularization keeps near-collinear histories well-posed.
    double tr = 0.0;
    for (int i = 0; i < mc; i++)
        tr += M[i][i];
    const double lambda = 1e-10 * (tr / mc) + 1e-300;
    for (int i = 0; i < mc; i++)
        M[i][i] += lambda;
    // Gaussian elimination with partial pivoting.
    for (int i = 0; i < mc; i++) {
        int piv = i;
        for (int r = i + 1; r < mc; r++)
            if (std::fabs(M[r][i]) > std::fabs(M[piv][i]))
                piv = r;
        if (piv != i) {
            for (int cc = 0; cc < mc; cc++)
                std::swap(M[i][cc], M[piv][cc]);
            std::swap(rhs[i], rhs[piv]);
        }
        if (std::fabs(M[i][i]) < 1e-300) { // singular -> fall back to plain Picard
            hm_ = u_;
            return;
        }
        for (int r = i + 1; r < mc; r++) {
            const double f = M[r][i] / M[i][i];
            for (int cc = i; cc < mc; cc++)
                M[r][cc] -= f * M[i][cc];
            rhs[r] -= f * rhs[i];
        }
    }
    for (int i = mc - 1; i >= 0; i--) {
        double s = rhs[i];
        for (int cc = i + 1; cc < mc; cc++)
            s -= M[i][cc] * gamma[cc];
        gamma[i] = s / M[i][i];
    }
    // Accelerated iterate: x_{k+1} = g(x_k) - dG gamma.
    for (int c = 0; c < N; c++) {
        double acc = u_[c];
        for (int i = 0; i < mc; i++)
            acc -= gamma[i] * aa_dG_[i][c];
        hm_[c] = acc;
    }
}

double SoilSolver3D::solveAdaptive(double target_time, double &dt, uint *steps_taken) {
    const int N = static_cast<int>(numCells());
    struct AdaptiveScope { // lets solve() know a failed step will be retried with a smaller dt
        bool &flag;
        explicit AdaptiveScope(bool &f) : flag(f) {
            flag = true;
        }
        ~AdaptiveScope() {
            flag = false;
        }
    } scope(in_adaptive_);
    double t = 0.0;
    uint steps = 0;
    dt = std::min(std::max(dt, params_.dt_min), params_.dt_max);
    std::vector<double> h_saved;
    bool rejected = false;
    // Start-of-call state: an error part-way through (e.g. a CUDA failure) leaves the solver as it was before the call
    const std::vector<double> h_start = h_, theta_n_start = theta_n_, theta_nm1_start = theta_nm1_;
    const double dt_prev_start = dt_prev_, influx_start = cumulative_influx_, residual_start = cumulative_balance_residual_;
    const bool history_start = have_history_;
    try {
        while (target_time - t > 1e-9 * target_time) { // a round-off remainder is not another step
            double dt_step = std::min(dt, target_time - t);
            h_saved = h_;
            uint pit = solveStep(dt_step);
            total_picard_iters_ += pit;
            if (pit == 0) { // non-convergence: reject, shrink dt, retry
                h_ = h_saved;
                have_history_ = false; // discard suspect history
                if (dt_step <= params_.dt_min * 1.01) {
                    // Already at dt_min: give the step one last try with the fixed-dt safeguards (Newton's
                    // Picard fallback), and if that fails too, stop and return the time actually advanced
                    // rather than letting the clock run on with a frozen state.
                    in_adaptive_ = false;
                    pit = solveStep(dt_step);
                    in_adaptive_ = true;
                    total_picard_iters_ += pit;
                    if (pit == 0) {
                        h_ = h_saved;
                        break;
                    }
                } else {
                    dt = std::max(dt * params_.dt_reduction_min, params_.dt_min);
                    rejected = true;
                    continue;
                }
            }
            // Relative water-content change over the accepted step (bounded error proxy).
            double rel = 0.0;
            for (int c = 0; c < N; c++) {
                double range = (double) props_[c].theta_s - (double) props_[c].theta_r;
                if (range > 1e-12)
                    rel = std::max(rel, std::fabs(theta_n_[c] - theta_nm1_[c]) / range);
            }
            t += dt_step;
            steps++;
            double factor = (rel > 1e-12) ? params_.dt_safety * std::sqrt(params_.dt_target_rel_change / rel) : params_.dt_growth_max;
            factor = std::max(params_.dt_reduction_min, std::min(factor, params_.dt_growth_max));
            if (rejected) // do not grow straight back into a step size that just failed
                factor = std::min(factor, 1.0);
            rejected = false;
            dt = std::max(params_.dt_min, std::min(dt * factor, params_.dt_max));
        }
    } catch (...) {
        h_ = h_start;
        theta_n_ = theta_n_start;
        theta_nm1_ = theta_nm1_start;
        dt_prev_ = dt_prev_start;
        have_history_ = history_start;
        cumulative_influx_ = influx_start;
        cumulative_balance_residual_ = residual_start;
        releaseGpuState();
        throw;
    }
    if (steps_taken)
        *steps_taken = steps;
    return t;
}

// ============================================================================
// Geometric multigrid preconditioner (re-discretization, semi-coarsening)
// ============================================================================

void SoilSolver3D::mgBuildStencil(MGLevel &L) const {
    const int Nx = L.Nx, Ny = L.Ny, Nz = L.Nz, NxNy = Nx * Ny, N = L.N();
    const double ix2 = 1.0 / (L.dx * L.dx), iy2 = 1.0 / (L.dy * L.dy), iz2 = 1.0 / (L.dz * L.dz);
    L.cfx.assign(N, 0.0);
    L.cfy.assign(N, 0.0);
    L.cfz.assign(N, 0.0);
    L.diag.assign(N, 0.0);
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        if (i < Nx - 1)
            L.cfx[c] = hmean(L.K[c], L.K[c + 1]) * ix2;
        if (j < Ny - 1)
            L.cfy[c] = hmean(L.K[c], L.K[c + Nx]) * iy2;
        if (k < Nz - 1)
            L.cfz[c] = hmean(L.K[c], L.K[c + NxNy]) * iz2;
    }
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        double d = L.Cdt[c];
        d += (i < Nx - 1 ? L.cfx[c] : 0.0) + (i > 0 ? L.cfx[c - 1] : 0.0);
        d += (j < Ny - 1 ? L.cfy[c] : 0.0) + (j > 0 ? L.cfy[c - Nx] : 0.0);
        d += (k < Nz - 1 ? L.cfz[c] : 0.0) + (k > 0 ? L.cfz[c - NxNy] : 0.0);
        if (i == 0 && L.dir[0])
            d += 2.0 * L.K[c] * ix2;
        if (i == Nx - 1 && L.dir[1])
            d += 2.0 * L.K[c] * ix2;
        if (j == 0 && L.dir[2])
            d += 2.0 * L.K[c] * iy2;
        if (j == Ny - 1 && L.dir[3])
            d += 2.0 * L.K[c] * iy2;
        if (k == 0 && L.dir[4])
            d += 2.0 * L.K[c] * iz2;
        if (k == Nz - 1 && L.dir[5])
            d += 2.0 * L.K[c] * iz2;
        L.diag[c] = d;
    }
}

void SoilSolver3D::mgBuild(double a0) {
    const bool need_props = (params_.nonlinear_method == NonlinearMethod::NGS_FAS); // per-level soil only for NGS-FAS
    mg_.clear();
    MGLevel L0;
    L0.Nx = Nx_;
    L0.Ny = Ny_;
    L0.Nz = Nz_;
    L0.dx = dx_;
    L0.dy = dy_;
    L0.dz = dz_;
    for (int f = 0; f < 6; f++)
        L0.dir[f] = (bc_[f].type == BCType::DIRICHLET);
    L0.K = K_;
    L0.Cdt.resize(numCells());
    for (uint c = 0; c < numCells(); c++)
        L0.Cdt[c] = a0 * C_[c];
    mgBuildStencil(L0);
    L0.diag = diag_; // use the solver's exact fine-grid diagonal
    L0.u.assign(L0.N(), 0.0);
    L0.b.assign(L0.N(), 0.0);
    L0.res.assign(L0.N(), 0.0);
    L0.fas.assign(L0.N(), 0.0);
    if (need_props)
        L0.props = props_; // fine grid: exact per-cell soil
    mg_.push_back(std::move(L0));

    while (true) {
        const MGLevel &F = mg_.back();
        int cnx = (F.Nx > params_.mg_min_dim) ? (F.Nx + 1) / 2 : F.Nx;
        int cny = (F.Ny > params_.mg_min_dim) ? (F.Ny + 1) / 2 : F.Ny;
        int cnz = (F.Nz > params_.mg_min_dim) ? (F.Nz + 1) / 2 : F.Nz;
        if (cnx == F.Nx && cny == F.Ny && cnz == F.Nz)
            break; // cannot coarsen further
        const int fx = (cnx < F.Nx) ? 2 : 1, fy = (cny < F.Ny) ? 2 : 1, fz = (cnz < F.Nz) ? 2 : 1;
        MGLevel C;
        C.Nx = cnx;
        C.Ny = cny;
        C.Nz = cnz;
        C.dx = F.dx * fx;
        C.dy = F.dy * fy;
        C.dz = F.dz * fz;
        for (int f = 0; f < 6; f++)
            C.dir[f] = F.dir[f];
        C.K.assign(C.N(), 0.0);
        C.Cdt.assign(C.N(), 0.0);
        if (need_props)
            C.props.assign(C.N(), SoilProperties());
        for (int c = 0; c < C.N(); c++) {
            const int kc = c / (C.Nx * C.Ny), rr = c - kc * (C.Nx * C.Ny), jc = rr / C.Nx, ic = rr - jc * C.Nx;
            double sk = 0, sc = 0;
            int cnt = 0;
            double str = 0, sts = 0, sal = 0, sn = 0, sKs = 0, she = 0; // averaged retention/conductivity params
            int model0 = 0;
            for (int dk = 0; dk < fz; dk++)
                for (int dj = 0; dj < fy; dj++)
                    for (int di = 0; di < fx; di++) {
                        int fi = fx * ic + di, fj = fy * jc + dj, fk = fz * kc + dk;
                        if (fi < F.Nx && fj < F.Ny && fk < F.Nz) {
                            int f = F.idx(fi, fj, fk);
                            sk += F.K[f];
                            sc += F.Cdt[f];
                            cnt++;
                            if (need_props) {
                                const SoilProperties &pf = F.props[f];
                                str += pf.theta_r;
                                sts += pf.theta_s;
                                sal += pf.alpha;
                                sn += pf.n;
                                sKs += pf.K_s;
                                she += pf.h_e;
                                model0 = (int) pf.model;
                            }
                        }
                    }
            C.K[c] = sk / cnt;
            C.Cdt[c] = sc / cnt;
            if (need_props) { // restricted effective soil (FAS: affects convergence rate, not the fine-grid solution)
                SoilProperties &pc = C.props[c];
                pc.theta_r = (float) (str / cnt);
                pc.theta_s = (float) (sts / cnt);
                pc.alpha = (float) (sal / cnt);
                pc.n = (float) (sn / cnt);
                pc.K_s = (float) (sKs / cnt);
                pc.h_e = (float) (she / cnt);
                pc.model = (SoilModel) model0;
            }
        }
        mgBuildStencil(C);
        C.u.assign(C.N(), 0.0);
        C.b.assign(C.N(), 0.0);
        C.res.assign(C.N(), 0.0);
        C.fas.assign(C.N(), 0.0);
        mg_.push_back(std::move(C));
    }
}

void SoilSolver3D::mgBuildMScheme(double D0, const std::vector<double> &react) {
    // Geometric-multigrid hierarchy for the M-scheme's SPD screened Laplacian [ react - D0*Lap ]. Setting
    // K = D0 (constant) makes mgBuildStencil's hmean(K,K)/d^2 = D0/d^2 (the constant-coefficient Laplacian),
    // and Cdt = react supplies the reaction diagonal; the Dirichlet 2*D0/d^2 handling matches solveMScheme's
    // Adiag exactly. Coarse levels: K averages to D0 (stays constant), react restricts (averaged).
    mg_.clear();
    MGLevel L0;
    L0.Nx = Nx_;
    L0.Ny = Ny_;
    L0.Nz = Nz_;
    L0.dx = dx_;
    L0.dy = dy_;
    L0.dz = dz_;
    for (int f = 0; f < 6; f++)
        L0.dir[f] = (bc_[f].type == BCType::DIRICHLET);
    L0.K.assign(numCells(), D0);
    L0.Cdt = react;
    mgBuildStencil(L0);
    L0.u.assign(L0.N(), 0.0);
    L0.b.assign(L0.N(), 0.0);
    L0.res.assign(L0.N(), 0.0);
    L0.fas.assign(L0.N(), 0.0);
    mg_.push_back(std::move(L0));
    while (true) {
        const MGLevel &F = mg_.back();
        int cnx = (F.Nx > params_.mg_min_dim) ? (F.Nx + 1) / 2 : F.Nx;
        int cny = (F.Ny > params_.mg_min_dim) ? (F.Ny + 1) / 2 : F.Ny;
        int cnz = (F.Nz > params_.mg_min_dim) ? (F.Nz + 1) / 2 : F.Nz;
        if (cnx == F.Nx && cny == F.Ny && cnz == F.Nz)
            break;
        const int fx = (cnx < F.Nx) ? 2 : 1, fy = (cny < F.Ny) ? 2 : 1, fz = (cnz < F.Nz) ? 2 : 1;
        MGLevel C;
        C.Nx = cnx;
        C.Ny = cny;
        C.Nz = cnz;
        C.dx = F.dx * fx;
        C.dy = F.dy * fy;
        C.dz = F.dz * fz;
        for (int f = 0; f < 6; f++)
            C.dir[f] = F.dir[f];
        C.K.assign(C.N(), 0.0);
        C.Cdt.assign(C.N(), 0.0);
        for (int c = 0; c < C.N(); c++) {
            const int kc = c / (C.Nx * C.Ny), rr = c - kc * (C.Nx * C.Ny), jc = rr / C.Nx, ic = rr - jc * C.Nx;
            double sk = 0, sc = 0;
            int cnt = 0;
            for (int dk = 0; dk < fz; dk++)
                for (int dj = 0; dj < fy; dj++)
                    for (int di = 0; di < fx; di++) {
                        int fi = fx * ic + di, fj = fy * jc + dj, fk = fz * kc + dk;
                        if (fi < F.Nx && fj < F.Ny && fk < F.Nz) {
                            int f = F.idx(fi, fj, fk);
                            sk += F.K[f];
                            sc += F.Cdt[f];
                            cnt++;
                        }
                    }
            C.K[c] = sk / cnt;
            C.Cdt[c] = sc / cnt;
        }
        mgBuildStencil(C);
        C.u.assign(C.N(), 0.0);
        C.b.assign(C.N(), 0.0);
        C.res.assign(C.N(), 0.0);
        C.fas.assign(C.N(), 0.0);
        mg_.push_back(std::move(C));
    }
}

void SoilSolver3D::mgMatvec(const MGLevel &L, const std::vector<double> &u, std::vector<double> &out) const {
    const int Nx = L.Nx, Ny = L.Ny, Nz = L.Nz, NxNy = Nx * Ny, N = L.N();
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        double a = L.diag[c] * u[c];
        if (i < Nx - 1)
            a -= L.cfx[c] * u[c + 1];
        if (i > 0)
            a -= L.cfx[c - 1] * u[c - 1];
        if (j < Ny - 1)
            a -= L.cfy[c] * u[c + Nx];
        if (j > 0)
            a -= L.cfy[c - Nx] * u[c - Nx];
        if (k < Nz - 1)
            a -= L.cfz[c] * u[c + NxNy];
        if (k > 0)
            a -= L.cfz[c - NxNy] * u[c - NxNy];
        out[c] = a;
    }
}

void SoilSolver3D::mgSmooth(MGLevel &L, std::vector<double> &u, const std::vector<double> &b, int sweeps) const {
    const double omega = 0.8;
    const int N = L.N();
    for (int s = 0; s < sweeps; s++) {
        mgMatvec(L, u, L.res);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
        for (int c = 0; c < N; c++)
            u[c] += omega * (b[c] - L.res[c]) / L.diag[c];
    }
}

void SoilSolver3D::mgVcycle(int lvl) {
    MGLevel &L = mg_[lvl];
    if (lvl == (int) mg_.size() - 1) {
        mgSmooth(L, L.u, L.b, params_.mg_coarse_sweeps);
        return;
    }
    mgSmooth(L, L.u, L.b, params_.mg_pre);
    mgMatvec(L, L.u, L.res);
    const int N = L.N();
    for (int c = 0; c < N; c++)
        L.res[c] = L.b[c] - L.res[c];
    MGLevel &C = mg_[lvl + 1];
    const int fx = (C.Nx < L.Nx) ? 2 : 1, fy = (C.Ny < L.Ny) ? 2 : 1, fz = (C.Nz < L.Nz) ? 2 : 1;
    for (int c = 0; c < C.N(); c++) { // restrict residual (full weighting / average)
        const int kc = c / (C.Nx * C.Ny), rr = c - kc * (C.Nx * C.Ny), jc = rr / C.Nx, ic = rr - jc * C.Nx;
        double s = 0;
        int cnt = 0;
        for (int dk = 0; dk < fz; dk++)
            for (int dj = 0; dj < fy; dj++)
                for (int di = 0; di < fx; di++) {
                    int fi = fx * ic + di, fj = fy * jc + dj, fk = fz * kc + dk;
                    if (fi < L.Nx && fj < L.Ny && fk < L.Nz) {
                        s += L.res[L.idx(fi, fj, fk)];
                        cnt++;
                    }
                }
        C.b[c] = (cnt > 0) ? s / cnt : 0.0;
    }
    std::fill(C.u.begin(), C.u.end(), 0.0);
    mgVcycle(lvl + 1);
    for (int c = 0; c < L.N(); c++) { // prolong correction (piecewise constant) and add
        const int kf = c / (L.Nx * L.Ny), rr = c - kf * (L.Nx * L.Ny), jf = rr / L.Nx, iff = rr - jf * L.Nx;
        L.u[c] += C.u[C.idx(iff / fx, jf / fy, kf / fz)];
    }
    mgSmooth(L, L.u, L.b, params_.mg_post);
}

// --- FAS multigrid for the NGS scheme (Gu & Wang, 2026) ---------------------------------------
// N_L(u) = a0*theta(u) + xi*u + [diffusion], with diffusion = (diag-Cdt)*u - sum cf*u_nb (the
// gravity-free symmetric operator; gravity/sinks/BCs live in the level RHS). Uniform soil props.
void SoilSolver3D::ngsOperatorLevel(const MGLevel &L, const std::vector<double> &u, std::vector<double> &out, double a0, double xi) const {
    const int Nx = L.Nx, Ny = L.Ny, Nz = L.Nz, NxNy = Nx * Ny, N = L.N();
    out.resize(N);
#pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int c = 0; c < N; c++) {
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        double diff = (L.diag[c] - L.Cdt[c]) * u[c]; // diffusion diagonal m_ii * u_c
        if (i < Nx - 1)
            diff -= L.cfx[c] * u[c + 1];
        if (i > 0)
            diff -= L.cfx[c - 1] * u[c - 1];
        if (j < Ny - 1)
            diff -= L.cfy[c] * u[c + Nx];
        if (j > 0)
            diff -= L.cfy[c - Nx] * u[c - Nx];
        if (k < Nz - 1)
            diff -= L.cfz[c] * u[c + NxNy];
        if (k > 0)
            diff -= L.cfz[c - NxNy] * u[c - NxNy];
        out[c] = a0 * waterContent<double>(u[c], L.props[c]) + xi * u[c] + diff;
    }
}

// One or more lexicographic NGS sweeps solving N_L(u) = b: each cell is the scalar monotone
// equation a0*theta(u_c) + (xi + m_ii) u_c = b_c + sum cf*u_nb, by a few damped Newton steps.
void SoilSolver3D::ngsSweepLevel(const MGLevel &L, std::vector<double> &u, const std::vector<double> &b, double a0, double xi, int sweeps) const {
    const int Nx = L.Nx, Ny = L.Ny, Nz = L.Nz, NxNy = Nx * Ny, N = L.N();
    const double Cmin = params_.C_min, Kmin = params_.K_min;
    for (int s = 0; s < sweeps; s++) {
        for (int c = 0; c < N; c++) { // sequential Gauss-Seidel (newest neighbour values)
            const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
            const SoilProperties &p0 = L.props[c];
            const double mii = L.diag[c] - L.Cdt[c];
            double coup = 0.0;
            if (i < Nx - 1)
                coup += L.cfx[c] * u[c + 1];
            if (i > 0)
                coup += L.cfx[c - 1] * u[c - 1];
            if (j < Ny - 1)
                coup += L.cfy[c] * u[c + Nx];
            if (j > 0)
                coup += L.cfy[c - Nx] * u[c - Nx];
            if (k < Nz - 1)
                coup += L.cfz[c] * u[c + NxNy];
            if (k > 0)
                coup += L.cfz[c - NxNy] * u[c - NxNy];
            const double rhs = b[c] + coup, dia = xi + mii;
            double psi = u[c];
            for (int nit = 0; nit < 4; nit++) {
                double Kd, Cap;
                computeKandC<double>(psi, p0, Kd, Cap, Cmin, Kmin);
                const double dpsi = (a0 * waterContent<double>(psi, p0) + dia * psi - rhs) / (a0 * Cap + dia);
                psi -= dpsi;
                if (std::fabs(dpsi) < 1e-13)
                    break;
            }
            u[c] = psi;
        }
    }
}

// Recursive FAS V-cycle: pre-smooth, restrict (solution + nonlinear defect), build the FAS coarse
// RHS g_H = N_H(I psi) + I(g_h - N_h(u)), recurse, prolong the coarse correction, post-smooth.
void SoilSolver3D::mgVcycleFAS(int lvl, double a0, double xi) {
    MGLevel &L = mg_[lvl];
    if (lvl == (int) mg_.size() - 1) {
        ngsSweepLevel(L, L.u, L.b, a0, xi, params_.mg_coarse_sweeps);
        return;
    }
    ngsSweepLevel(L, L.u, L.b, a0, xi, params_.mg_pre);
    ngsOperatorLevel(L, L.u, L.res, a0, xi); // res = N_L(u)
    const int NL = L.N();
    for (int c = 0; c < NL; c++)
        L.res[c] = L.b[c] - L.res[c]; // res = defect d
    MGLevel &C = mg_[lvl + 1];
    const int fx = (C.Nx < L.Nx) ? 2 : 1, fy = (C.Ny < L.Ny) ? 2 : 1, fz = (C.Nz < L.Nz) ? 2 : 1;
    for (int c = 0; c < C.N(); c++) { // restrict solution -> C.fas, defect -> C.res
        const int kc = c / (C.Nx * C.Ny), rr = c - kc * (C.Nx * C.Ny), jc = rr / C.Nx, ic = rr - jc * C.Nx;
        double su = 0, sd = 0;
        int cnt = 0;
        for (int dk = 0; dk < fz; dk++)
            for (int dj = 0; dj < fy; dj++)
                for (int di = 0; di < fx; di++) {
                    int fi = fx * ic + di, fj = fy * jc + dj, fk = fz * kc + dk;
                    if (fi < L.Nx && fj < L.Ny && fk < L.Nz) {
                        int f = L.idx(fi, fj, fk);
                        su += L.u[f];
                        sd += L.res[f];
                        cnt++;
                    }
                }
        C.fas[c] = (cnt > 0) ? su / cnt : 0.0;
        C.res[c] = (cnt > 0) ? sd / cnt : 0.0;
    }
    ngsOperatorLevel(C, C.fas, C.b, a0, xi); // C.b = N_C(I psi)
    for (int c = 0; c < C.N(); c++)
        C.b[c] += C.res[c]; // + restricted defect
    C.u = C.fas; // FAS initial guess = restricted solution
    mgVcycleFAS(lvl + 1, a0, xi);
    for (int c = 0; c < NL; c++) { // prolong correction e = C.u - C.fas (piecewise constant) and add
        const int kf = c / (L.Nx * L.Ny), rr = c - kf * (L.Nx * L.Ny), jf = rr / L.Nx, iff = rr - jf * L.Nx;
        const int cc = C.idx(iff / fx, jf / fy, kf / fz);
        L.u[c] += C.u[cc] - C.fas[cc];
    }
    ngsSweepLevel(L, L.u, L.b, a0, xi, params_.mg_post);
}

void SoilSolver3D::mgApply(std::vector<double> &z, const std::vector<double> &r) {
    mg_[0].b = r;
    std::fill(mg_[0].u.begin(), mg_[0].u.end(), 0.0);
    mgVcycle(0);
    z = mg_[0].u;
}
