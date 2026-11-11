/** \file "SoilSolver3D.cu" CUDA backend for the 3D Richards-equation solver.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

// GPU implementation of SoilSolver3D::solve(). It mirrors the CPU Modified-Picard
// + matrix-free CG path in SoilSolver3D.cpp exactly (same 7-point stencil, interblock
// faces per Params::K_weighting, gravity and boundary handling), in double
// precision, so the two backends must agree to the linear-solver tolerance.
// Selected at runtime via Params::use_gpu. The geometric-multigrid preconditioner
// is CPU-only; on the GPU the same linear system is solved with Jacobi-CG, which
// converges to the identical solution (only the iteration count differs).

#ifdef HELIOS_CUDA_AVAILABLE

#include "SoilSolver3D.h"
#include "global.h"

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <utility>
#include <algorithm>

namespace {

    constexpr int TPB = 256; // threads per block for element-wise kernels
    constexpr int RB = 256; // fixed number of blocks for reductions

#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)
    //! Raise an error for a failed CUDA call. Kernel execution errors are sticky, so they surface at the next checked call.
    inline void gpuAssert(cudaError_t code, const char *file, int line) {
        if (code != cudaSuccess) {
            cudaGetLastError(); // clear a non-sticky error so a caller that recovers starts clean
            helios::helios_runtime_error("ERROR (SoilSolver3D, GPU): CUDA call failed: " + std::string(cudaGetErrorString(code)) + " (" + std::string(file) + ":" + std::to_string(line) +
                                         "). The GPU may be out of memory, or its driver may have failed. Call SoilTransportModel::disableGPU() (or set "
                                         "SoilSolver3D::Params::use_gpu = false) to run the 3D solver on the CPU.");
        }
    }
//! Download from the device after checking for an unreported kernel-launch error: launch failures (e.g. an invalid
//! configuration) are not sticky and are reported only by cudaGetLastError().
#define CUDA_CHECK_READBACK(ans)                                                                                                                                                                                                                         \
    do {                                                                                                                                                                                                                                                 \
        CUDA_CHECK(cudaGetLastError());                                                                                                                                                                                                                  \
        CUDA_CHECK(ans);                                                                                                                                                                                                                                 \
    } while (0)

    //! Device-side mirror of SoilProperties (POD, double).
    struct GpuSoil {
        int model; // 0 = van Genuchten, 1 = Gardner, 2 = modified (air-entry) van Genuchten, 3 = linear
        double alpha, n, theta_r, theta_s, K_s;
        double h_s; // saturated for h >= h_s: 0, or the air-entry head
        double Sc, Fk; // air-entry scales (see airEntryScales() in SoilUtils.h): 1 unless model 2
        double S1, kr1, dkr1; // relative-conductivity spline start (SoilProperties::kr_spline_start; 1 = off) and k_r, dk_r/dS_e there
    };

    //! k_r(S1) and dk_r/dS_e(S1) of the standard van Genuchten model (start of the near-saturation spline)
    __host__ __device__ inline void krSplineStart(double S1, double n, double &kr1, double &dkr1) {
        const double m = 1.0 - 1.0 / n, w = 1.0 - pow(S1, 1.0 / m), B = 1.0 - pow(w, m), sq = sqrt(S1);
        kr1 = sq * B * B;
        dkr1 = 0.5 / sq * B * B + 2.0 * sq * B * pow(w, m - 1.0) * pow(S1, 1.0 / m - 1.0);
    }

    //! Near-saturation spline of k_r (mirrors krSaturationSpline() in SoilUtils.h); false where it does not apply
    __device__ inline bool d_krSpline(const GpuSoil &p, double Se, double &kr, double &dkr) {
        if (p.model != 0 || !(p.S1 < 1.0) || !(Se > p.S1))
            return false;
        const double D = 1.0 - p.S1, t = (fmin(Se, 1.0) - p.S1) / D, t2 = t * t, t3 = t2 * t;
        kr = (2.0 * t3 - 3.0 * t2 + 1.0) * p.kr1 + (t3 - 2.0 * t2 + t) * D * p.dkr1 + (3.0 * t2 - 2.0 * t3);
        dkr = ((6.0 * t2 - 6.0 * t) * p.kr1 + (3.0 * t2 - 4.0 * t + 1.0) * D * p.dkr1 + (6.0 * t - 6.0 * t2)) / D;
        return true;
    }

    GpuSoil toGpuSoil(const SoilProperties &p) {
        GpuSoil g{(int) p.model, (double) p.alpha, (double) p.n, (double) p.theta_r, (double) p.theta_s, (double) p.K_s, saturationHead(p), 1.0, 1.0, (double) p.kr_spline_start, 0.0, 0.0};
        airEntryScales(p, g.Sc, g.Fk);
        if (g.model == 0 && g.S1 < 1.0)
            krSplineStart(g.S1, g.n, g.kr1, g.dkr1);
        return g;
    }

    //! Grid geometry passed to kernels by value.
    struct GpuDims {
        int Nx, Ny, Nz;
        double dx, dy, dz;
    };

    //! Boundary conditions passed to kernels by value (indexed by Face3D).
    struct GpuBC {
        int type[6]; // 0 = Dirichlet, 1 = Neumann, 2 = free drainage
        double val[6];
    };

    // --- Device constitutive functions (exact double-precision mirrors of
    //     computeKandC<double> and waterContent<double> in SoilUtils.h) -----------

    // Additive conductivity floor [m/s], set once per solve via cudaMemcpyToSymbol. Mirrors the CPU
    // computeKandC K_min: lifts the near-residual (K->0) baseline so dry wetting fronts stay coupled.
    __constant__ double d_Kmin = 0.0;

    __device__ inline void d_KandC(double h, const GpuSoil &p, double Cmin, double &K, double &C) {
        const double Ks = p.K_s, al = p.alpha, tr = p.theta_r, ts = p.theta_s;
        if (h >= p.h_s) {
            K = Ks + d_Kmin;
            C = Cmin;
            return;
        }
        if (p.model == 1) { // Gardner
            double e = exp(al * h);
            K = Ks * e + d_Kmin;
            double c = al * (ts - tr) * e;
            C = (c < Cmin) ? Cmin : c;
            return;
        }
        if (p.model == 3) { // linear
            const double Se = 1.0 + al * h;
            K = Ks * fmax(Se, 0.0) + d_Kmin;
            const double c = Se > 0.0 ? al * (ts - tr) : 0.0;
            C = (c < Cmin) ? Cmin : c;
            return;
        }
        const double n = p.n, m = 1.0 - 1.0 / n;
        double ah = fabs(al * h);
        double ahn = pow(ah, n);
        double base = 1.0 + ahn;
        double Se = pow(base, -m);
        double bracket = 1.0 - pow(1.0 - 1.0 / base, m);
        double kr, dkr;
        K = (d_krSpline(p, Se, kr, dkr) ? Ks * kr : Ks * p.Fk * sqrt(Se) * bracket * bracket) + d_Kmin;
        double c = (ts - tr) * m * n * al * (ahn / ah) * Se / base / p.Sc;
        C = (c < Cmin) ? Cmin : c;
    }

    __device__ inline double d_theta(double h, const GpuSoil &p) {
        const double tr = p.theta_r, ts = p.theta_s, al = p.alpha;
        if (h >= p.h_s)
            return ts;
        double Se;
        if (p.model == 1) {
            Se = exp(al * h);
        } else if (p.model == 3) {
            Se = fmax(1.0 + al * h, 0.0);
        } else {
            double n = p.n, m = 1.0 - 1.0 / n;
            Se = pow(1.0 + pow(fabs(al * h), n), -m) / p.Sc;
        }
        return tr + (ts - tr) * Se;
    }

    __device__ inline double d_Kof(double h, const GpuSoil &p) {
        double K, C;
        d_KandC(h, p, 1e-30, K, C);
        return K;
    }

    __device__ inline double d_hmean(double a, double b) {
        return (a + b > 0.0) ? 2.0 * a * b / (a + b) : 0.0;
    }

    __constant__ int d_Kweight = 3; // Params::K_weighting: 0 geometric, 1 arithmetic, 2 upstream K, 3 harmonic, 4 upstream k_r

    // Interblock conductivity per Params::K_weighting and its partial derivatives with respect to the two
    // cell conductivities; mirrors the CPU faceCond()/faceCondDeriv(). Ha/Hb are total heads (the caller
    // gravity-adjusts the neighbour by +/- dz on z-faces); symmetric weightings ignore them. Ksa/Ksb are
    // the saturated conductivities (used by upstream relative-permeability weighting).
    __device__ inline double d_faceCondD(double Ka, double Kb, double Ha, double Hb, double Ksa, double Ksb, double &dA, double &dB) {
        switch (d_Kweight) {
            case 0: {
                const double gm = sqrt(Ka * Kb);
                dA = (Ka > 0.0) ? 0.5 * gm / Ka : 0.0;
                dB = (Kb > 0.0) ? 0.5 * gm / Kb : 0.0;
                return gm;
            }
            case 1:
                dA = dB = 0.5;
                return 0.5 * (Ka + Kb);
            case 2: {
                const double dH = Ha - Hb;
                if (dH > 0.0 || (dH == 0.0 && Ka >= Kb)) {
                    dA = 1.0;
                    dB = 0.0;
                    return Ka;
                }
                dA = 0.0;
                dB = 1.0;
                return Kb;
            }
            case 4: {
                const double Ksh = d_hmean(Ksa, Ksb);
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
                const double sm = Ka + Kb;
                if (sm <= 0.0) {
                    dA = dB = 0.0;
                    return 0.0;
                }
                dA = 2.0 * Kb * Kb / (sm * sm);
                dB = 2.0 * Ka * Ka / (sm * sm);
                return 2.0 * Ka * Kb / sm;
            }
        }
    }

    __device__ inline double d_faceCond(double Ka, double Kb, double Ha, double Hb, double Ksa, double Ksb) {
        double dA, dB;
        return d_faceCondD(Ka, Kb, Ha, Hb, Ksa, Ksb, dA, dB);
    }

    // Dirichlet boundary-face conductivity; mirrors the CPU bcFaceK(): the boundary is a virtual cell
    // (same soil) half a cell beyond the face, weighted like an interior face by total head.
    __device__ inline double d_bcFaceK(int face, double Kc, double hc, const GpuSoil &pc, const GpuBC &bc, double dz) {
        const double hbc = bc.val[face];
        const double off = (face == 4) ? -0.5 * dz : (face == 5 ? 0.5 * dz : 0.0);
        return d_faceCond(Kc, d_Kof(hbc, pc), hc, hbc + off, pc.K_s, pc.K_s);
    }

    // --- Kernels -----------------------------------------------------------------

    __global__ void kUpdateKC(const double *hm, const GpuSoil *props, double Cmin, int N, double *K, double *C, double *theta_m) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        double k, cc;
        d_KandC(hm[c], props[c], Cmin, k, cc);
        K[c] = k;
        C[c] = cc;
        theta_m[c] = d_theta(hm[c], props[c]);
    }

    // A v = a0*C*v - L_K v, with Dirichlet faces contributing to the diagonal.
    __global__ void kMatvec(const double *v, double *out, const double *K, const double *C, const GpuSoil *props, const double *hm, double a0, GpuDims g, GpuBC bc, int N) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int Nx = g.Nx, Ny = g.Ny, Nz = g.Nz, NxNy = Nx * Ny;
        const double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy), idz2 = 1.0 / (g.dz * g.dz);
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const double Kc = K[c], Hc = hm[c]; // Hc = linearization-point head (donor decision for upstream)
        double acc = C[c] * a0 * v[c];
        if (i > 0) {
            double cf = d_faceCond(Kc, K[c - 1], Hc, hm[c - 1], props[c].K_s, props[c - 1].K_s) * idx2;
            acc += cf * (v[c] - v[c - 1]);
        } else if (bc.type[0] == 0)
            acc += 2.0 * d_bcFaceK(0, Kc, Hc, props[c], bc, g.dz) * idx2 * v[c];
        if (i < Nx - 1) {
            double cf = d_faceCond(Kc, K[c + 1], Hc, hm[c + 1], props[c].K_s, props[c + 1].K_s) * idx2;
            acc += cf * (v[c] - v[c + 1]);
        } else if (bc.type[1] == 0)
            acc += 2.0 * d_bcFaceK(1, Kc, Hc, props[c], bc, g.dz) * idx2 * v[c];
        if (j > 0) {
            double cf = d_faceCond(Kc, K[c - Nx], Hc, hm[c - Nx], props[c].K_s, props[c - Nx].K_s) * idy2;
            acc += cf * (v[c] - v[c - Nx]);
        } else if (bc.type[2] == 0)
            acc += 2.0 * d_bcFaceK(2, Kc, Hc, props[c], bc, g.dz) * idy2 * v[c];
        if (j < Ny - 1) {
            double cf = d_faceCond(Kc, K[c + Nx], Hc, hm[c + Nx], props[c].K_s, props[c + Nx].K_s) * idy2;
            acc += cf * (v[c] - v[c + Nx]);
        } else if (bc.type[3] == 0)
            acc += 2.0 * d_bcFaceK(3, Kc, Hc, props[c], bc, g.dz) * idy2 * v[c];
        if (k > 0) {
            double cf = d_faceCond(Kc, K[c - NxNy], Hc, hm[c - NxNy] - g.dz, props[c].K_s, props[c - NxNy].K_s) * idz2;
            acc += cf * (v[c] - v[c - NxNy]);
        } else if (bc.type[4] == 0)
            acc += 2.0 * d_bcFaceK(4, Kc, Hc, props[c], bc, g.dz) * idz2 * v[c];
        if (k < Nz - 1) {
            double cf = d_faceCond(Kc, K[c + NxNy], Hc, hm[c + NxNy] + g.dz, props[c].K_s, props[c + NxNy].K_s) * idz2;
            acc += cf * (v[c] - v[c + NxNy]);
        } else if (bc.type[5] == 0)
            acc += 2.0 * d_bcFaceK(5, Kc, Hc, props[c], bc, g.dz) * idz2 * v[c];
        out[c] = acc;
    }

    // Fill b (RHS) and diag (Jacobi preconditioner) in one pass; mirrors buildSystem().
    __global__ void kBuildSystem(const double *hm, const double *K, const double *C, const double *theta_m, const double *theta_n, const double *theta_nm1, const double *src, const GpuSoil *props, double a0, double a1, double a2, GpuDims g, GpuBC bc,
                                 int N, double *diag, double *b) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int Nx = g.Nx, Ny = g.Ny, Nz = g.Nz, NxNy = Nx * Ny;
        const double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy), idz2 = 1.0 / (g.dz * g.dz);
        const double idz = 1.0 / g.dz, idx = 1.0 / g.dx, idy = 1.0 / g.dy;
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const double Kc = K[c];
        const GpuSoil pc = props[c];
        double d = C[c] * a0;
        double rhs = C[c] * a0 * hm[c] - (a0 * theta_m[c] + a1 * theta_n[c] + a2 * theta_nm1[c]) + src[c];
        const double Hc = hm[c]; // donor head for upstream weighting (symmetric/geometric/arithmetic ignore it)
        // -x
        if (i > 0)
            d += d_faceCond(Kc, K[c - 1], Hc, hm[c - 1], props[c].K_s, props[c - 1].K_s) * idx2;
        else if (bc.type[0] == 0) {
            double cf = 2.0 * d_bcFaceK(0, Kc, Hc, pc, bc, g.dz) * idx2;
            d += cf;
            rhs += cf * bc.val[0];
        } else if (bc.type[0] == 1)
            rhs += bc.val[0] * idx;
        // +x
        if (i < Nx - 1)
            d += d_faceCond(Kc, K[c + 1], Hc, hm[c + 1], props[c].K_s, props[c + 1].K_s) * idx2;
        else if (bc.type[1] == 0) {
            double cf = 2.0 * d_bcFaceK(1, Kc, Hc, pc, bc, g.dz) * idx2;
            d += cf;
            rhs += cf * bc.val[1];
        } else if (bc.type[1] == 1)
            rhs += bc.val[1] * idx;
        // -y
        if (j > 0)
            d += d_faceCond(Kc, K[c - Nx], Hc, hm[c - Nx], props[c].K_s, props[c - Nx].K_s) * idy2;
        else if (bc.type[2] == 0) {
            double cf = 2.0 * d_bcFaceK(2, Kc, Hc, pc, bc, g.dz) * idy2;
            d += cf;
            rhs += cf * bc.val[2];
        } else if (bc.type[2] == 1)
            rhs += bc.val[2] * idy;
        // +y
        if (j < Ny - 1)
            d += d_faceCond(Kc, K[c + Nx], Hc, hm[c + Nx], props[c].K_s, props[c + Nx].K_s) * idy2;
        else if (bc.type[3] == 0) {
            double cf = 2.0 * d_bcFaceK(3, Kc, Hc, pc, bc, g.dz) * idy2;
            d += cf;
            rhs += cf * bc.val[3];
        } else if (bc.type[3] == 1)
            rhs += bc.val[3] * idy;
        // -z (bottom): gravity term present on z-faces
        if (k > 0) {
            double Kz = d_faceCond(Kc, K[c - NxNy], Hc, hm[c - NxNy] - g.dz, props[c].K_s, props[c - NxNy].K_s);
            d += Kz * idz2;
            rhs += -Kz * idz;
        } else if (bc.type[4] == 0) {
            double cf = 2.0 * d_bcFaceK(4, Kc, Hc, pc, bc, g.dz) * idz2;
            d += cf;
            rhs += cf * bc.val[4] + cf * (-g.dz / 2.0);
        } else if (bc.type[4] == 1)
            rhs += bc.val[4] * idz;
        else if (bc.type[4] == 2)
            rhs += -Kc * idz; // free drainage: flux = K out the bottom
        // +z (top)
        if (k < Nz - 1) {
            double Kz = d_faceCond(Kc, K[c + NxNy], Hc, hm[c + NxNy] + g.dz, props[c].K_s, props[c + NxNy].K_s);
            d += Kz * idz2;
            rhs += Kz * idz;
        } else if (bc.type[5] == 0) {
            double cf = 2.0 * d_bcFaceK(5, Kc, Hc, pc, bc, g.dz) * idz2;
            d += cf;
            rhs += cf * bc.val[5] + cf * (g.dz / 2.0);
        } else if (bc.type[5] == 1)
            rhs += bc.val[5] * idz;
        else if (bc.type[5] == 2)
            rhs += Kc * idz;
        diag[c] = d;
        b[c] = rhs;
    }

    __global__ void kWaterContent(const double *h, const GpuSoil *props, int N, double *theta) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            theta[c] = d_theta(h[c], props[c]);
    }

    // Nonlinear residual R(h), evaluated self-contained (K computed inline per cell). src is the net
    // external rate src - uptake - evap, folded in by the caller. Mirrors the CPU residual().
    __global__ void kResidual(const double *h, const GpuSoil *props, const double *theta_n, const double *theta_nm1, const double *src, double a0, double a1, double a2, double Cmin, GpuDims g, GpuBC bc, int N, double *F) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int Nx = g.Nx, Ny = g.Ny, Nz = g.Nz, NxNy = Nx * Ny;
        const double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy), idz2 = 1.0 / (g.dz * g.dz);
        const double idx = 1.0 / g.dx, idy = 1.0 / g.dy, idz = 1.0 / g.dz;
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const GpuSoil pc = props[c];
        double Kc, Cc;
        d_KandC(h[c], pc, Cmin, Kc, Cc);
        const double hc = h[c];
        double R = a0 * d_theta(hc, pc) + a1 * theta_n[c] + a2 * theta_nm1[c] - src[c]; // storage - source
        double Kn, Cn;
        // -x / +x
        if (i > 0) {
            d_KandC(h[c - 1], props[c - 1], Cmin, Kn, Cn);
            R += d_faceCond(Kc, Kn, hc, h[c - 1], props[c].K_s, props[c - 1].K_s) * idx2 * (hc - h[c - 1]);
        } else if (bc.type[0] == 0)
            R += 2.0 * d_bcFaceK(0, Kc, hc, pc, bc, g.dz) * idx2 * (hc - bc.val[0]);
        else if (bc.type[0] == 1)
            R += -bc.val[0] * idx;
        if (i < Nx - 1) {
            d_KandC(h[c + 1], props[c + 1], Cmin, Kn, Cn);
            R += d_faceCond(Kc, Kn, hc, h[c + 1], props[c].K_s, props[c + 1].K_s) * idx2 * (hc - h[c + 1]);
        } else if (bc.type[1] == 0)
            R += 2.0 * d_bcFaceK(1, Kc, hc, pc, bc, g.dz) * idx2 * (hc - bc.val[1]);
        else if (bc.type[1] == 1)
            R += -bc.val[1] * idx;
        // -y / +y
        if (j > 0) {
            d_KandC(h[c - Nx], props[c - Nx], Cmin, Kn, Cn);
            R += d_faceCond(Kc, Kn, hc, h[c - Nx], props[c].K_s, props[c - Nx].K_s) * idy2 * (hc - h[c - Nx]);
        } else if (bc.type[2] == 0)
            R += 2.0 * d_bcFaceK(2, Kc, hc, pc, bc, g.dz) * idy2 * (hc - bc.val[2]);
        else if (bc.type[2] == 1)
            R += -bc.val[2] * idy;
        if (j < Ny - 1) {
            d_KandC(h[c + Nx], props[c + Nx], Cmin, Kn, Cn);
            R += d_faceCond(Kc, Kn, hc, h[c + Nx], props[c].K_s, props[c + Nx].K_s) * idy2 * (hc - h[c + Nx]);
        } else if (bc.type[3] == 0)
            R += 2.0 * d_bcFaceK(3, Kc, hc, pc, bc, g.dz) * idy2 * (hc - bc.val[3]);
        else if (bc.type[3] == 1)
            R += -bc.val[3] * idy;
        // -z (bottom): gravity on z-faces (b held -Kz*idz -> residual gets +Kz*idz)
        if (k > 0) {
            d_KandC(h[c - NxNy], props[c - NxNy], Cmin, Kn, Cn);
            double Kz = d_faceCond(Kc, Kn, hc, h[c - NxNy] - g.dz, props[c].K_s, props[c - NxNy].K_s);
            R += Kz * idz2 * (hc - h[c - NxNy]) + Kz * idz;
        } else if (bc.type[4] == 0) {
            double cf = 2.0 * d_bcFaceK(4, Kc, hc, pc, bc, g.dz) * idz2;
            R += cf * (hc - bc.val[4]) + cf * (g.dz / 2.0);
        } else if (bc.type[4] == 1)
            R += -bc.val[4] * idz;
        else if (bc.type[4] == 2)
            R += Kc * idz; // free drainage
        // +z (top)
        if (k < Nz - 1) {
            d_KandC(h[c + NxNy], props[c + NxNy], Cmin, Kn, Cn);
            double Kz = d_faceCond(Kc, Kn, hc, h[c + NxNy] + g.dz, props[c].K_s, props[c + NxNy].K_s);
            R += Kz * idz2 * (hc - h[c + NxNy]) - Kz * idz;
        } else if (bc.type[5] == 0) {
            double cf = 2.0 * d_bcFaceK(5, Kc, hc, pc, bc, g.dz) * idz2;
            R += cf * (hc - bc.val[5]) - cf * (g.dz / 2.0);
        } else if (bc.type[5] == 1)
            R += -bc.val[5] * idz;
        else if (bc.type[5] == 2)
            R += -Kc * idz;
        F[c] = R;
    }

    // --- NEWTON_ANALYTIC kernels (mirror SoilSolver3D::assembleNewton and friends) ----------------

    //! dK/dh with |alpha h| floored at kDKdhFloor for the derivative only (mirrors the CPU conductivityDerivative()).
    __device__ inline double d_dKdh(double h, const GpuSoil &p) {
        if (h >= p.h_s)
            return 0.0;
        const double Ks = p.K_s, al = p.alpha;
        if (p.model == 1)
            return Ks * al * exp(al * h);
        if (p.model == 3)
            return (1.0 + al * h > 0.0) ? Ks * al : 0.0;
        const double n = p.n, m = 1.0 - 1.0 / n;
        const double x = fmax(-al * h, kDKdhFloor);
        const double xn = pow(x, n), base = 1.0 + xn, Se = pow(base, -m);
        const double w = xn / base, B = 1.0 - pow(w, m);
        const double dSe_dh = m * n * al * (xn / x) * Se / base;
        const double dB_dSe = pow(w, m - 1.0) * pow(base, -(1.0 - m));
        const double sq = sqrt(Se);
        double kr, dkr;
        const double xe = -al * h, Se_e = pow(1.0 + pow(xe, n), -m); // exact S_e (x is floored above)
        if (d_krSpline(p, Se_e, kr, dkr))
            return Ks * dkr * m * n * al * pow(xe, n - 1.0) * pow(1.0 + pow(xe, n), -m - 1.0);
        return p.Fk * Ks * (0.5 / sq * B * B + 2.0 * sq * B * dB_dSe) * dSe_dh;
    }

    //! Pressure head at water content theta (inverse retention curve; mirrors the CPU hOfTheta()).
    __device__ inline double d_hOfTheta(double theta, const GpuSoil &p) {
        const double Se = (theta - p.theta_r) / (p.theta_s - p.theta_r);
        if (p.model == 1)
            return log(Se) / p.alpha;
        if (p.model == 3)
            return (Se - 1.0) / p.alpha;
        const double n = p.n, m = 1.0 - 1.0 / n;
        return -pow(pow(Se * p.Sc, -1.0 / m) - 1.0, 1.0 / n) / p.alpha;
    }

    //! theta, K, C, dK/dh at h; with specific storage Ss, theta and C become the stored water
    //! W = theta + Ss * S_w * h and dW/dh (mirrors constitutiveAll() + addSpecificStorage()).
    __global__ void kNkConstitutive(const double *h, const GpuSoil *props, const double *ss, double Cmin, int N, double *theta, double *K, double *C, double *dK) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const GpuSoil p = props[c];
        const double hc = h[c];
        double k, cc;
        d_KandC(hc, p, Cmin, k, cc);
        double th = d_theta(hc, p);
        if (ss[c] != 0.0) {
            const double Sw = th / p.theta_s, dSw = cc / p.theta_s;
            th += ss[c] * Sw * hc;
            cc += ss[c] * (Sw + hc * dSw);
        }
        K[c] = k;
        C[c] = cc;
        theta[c] = th;
        dK[c] = d_dKdh(hc, p);
    }

    __global__ void kStoredWater(const double *h, const GpuSoil *props, const double *ss, int N, double *W) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const double th = d_theta(h[c], props[c]);
        W[c] = th + ss[c] * th / props[c].theta_s * h[c];
    }

    //! Residual F, per-cell gross-flux scale and (optionally) the 7-point Jacobian J (7 arrays of N:
    //! diagonal, -x, +x, -y, +y, -z, +z). Mirrors SoilSolver3D::assembleNewton(), including per-cell
    //! centre elevation zc / thickness thk (terrain-following grids) and specified-head cells (fixedh,
    //! NaN where free). Every face flux is Kf * g * (H_c - H_n) in total head H = h + z.
    __global__ void kNkAssemble(const double *h, const GpuSoil *props, const double *theta, const double *K, const double *C, const double *dK, const double *theta_n, const double *theta_nm1, const double *src, const double *zc, const double *thk,
                                const double *fixedh, double a0, double a1, double a2, GpuDims g, GpuBC bc, int withJ, int N, double *F, double *scale, double *J) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int Nx = g.Nx, Ny = g.Ny, Nz = g.Nz, NxNy = Nx * Ny;
        const double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy);
        const int k = c / NxNy, rem = c - k * NxNy, j = rem / Nx, i = rem - j * Nx;
        const GpuSoil pc = props[c];
        const double hc = h[c], Kc = K[c], dKc = dK[c], tc = thk[c], z = zc[c];
        double R = a0 * theta[c] + a1 * theta_n[c] + a2 * theta_nm1[c] - src[c];
        double sc = fabs(a0 * theta[c]) + fabs(a1 * theta_n[c]) + fabs(a2 * theta_nm1[c]) + fabs(src[c]);
        double Jd = a0 * C[c];
        double Jn[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const int nb[6] = {c - 1, c + 1, c - Nx, c + Nx, c - NxNy, c + NxNy};
        const bool has[6] = {i > 0, i < Nx - 1, j > 0, j < Ny - 1, k > 0, k < Nz - 1};
        for (int f = 0; f < 6; f++) {
            if (has[f]) {
                const int n = nb[f];
                const double gf = (f < 4) ? 0.5 * (tc + thk[n]) * (f < 2 ? idx2 : idy2) / tc : 1.0 / (fabs(zc[n] - z) * tc);
                const double Hn = h[n] + (zc[n] - z), dH = hc - Hn;
                double dA, dB;
                const double Kf = d_faceCondD(Kc, K[n], hc, Hn, pc.K_s, props[n].K_s, dA, dB);
                R += gf * Kf * dH;
                sc += fabs(gf * Kf * dH) + kNewtonRoundoffScale * gf * Kf * (fabs(hc) + fabs(Hn));
                Jd += gf * (Kf + dA * dKc * dH);
                Jn[f] = gf * (-Kf + dB * dK[n] * dH);
            } else if (bc.type[f] == 0) { // Dirichlet, half-cell distance
                const double gf = (f < 2) ? idx2 : (f < 4 ? idy2 : 1.0 / (tc * tc));
                const double off = (f == 4) ? 0.5 * tc : (f == 5 ? -0.5 * tc : 0.0); // z_c - z_boundary_node
                const double hbc = bc.val[f];
                double dA, dB;
                const double Kf = d_faceCondD(Kc, d_Kof(hbc, pc), hc, hbc - off, pc.K_s, pc.K_s, dA, dB);
                const double cf = 2.0 * Kf * gf, dH = hc - hbc + off;
                R += cf * dH;
                sc += fabs(cf * dH) + kNewtonRoundoffScale * cf * (fabs(hc) + fabs(hbc - off));
                Jd += cf + 2.0 * gf * dA * dKc * dH;
            } else if (bc.type[f] == 1) {
                R += -bc.val[f] * ((f < 2) ? 1.0 / g.dx : (f < 4 ? 1.0 / g.dy : 1.0 / tc));
            } else if (bc.type[f] == 2 && f >= 4) { // free drainage (z faces)
                const double sgn = (f == 4) ? 1.0 : -1.0;
                R += sgn * Kc / tc;
                Jd += sgn * dKc / tc;
            }
        }
        if (!isnan(fixedh[c])) { // specified-head cell: identity row
            F[c] = hc - fixedh[c];
            scale[c] = 1.0;
            if (withJ) {
                J[c] = 1.0;
                for (int f = 0; f < 6; f++)
                    J[(f + 1) * N + c] = 0.0;
            }
            return;
        }
        F[c] = R;
        scale[c] = sc;
        if (withJ) {
            J[c] = Jd;
            for (int f = 0; f < 6; f++)
                J[(f + 1) * N + c] = Jn[f];
        }
    }


    //! Vertical-line block-Jacobi preconditioner for the Newton Jacobian: one thread per (i,j) column solves
    //! that column's tridiagonal (diagonal, -z, +z couplings) exactly by the Thomas algorithm, dropping the
    //! lateral couplings. On thin-layered grids vertical coupling dominates, so this is nearly exact.
    //! work holds Nz_max scratch per column (the modified super-diagonal).
    __global__ void kNkLinePrecond(const double *J, const double *r, GpuDims g, int N, double *work, double *z) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int NxNy = g.Nx * g.Ny;
        if (col >= NxNy)
            return;
        const double *jd = J, *jzm = J + 5 * (size_t) N, *jzp = J + 6 * (size_t) N;
        // forward sweep
        double denom = jd[col];
        double cprev = jzp[col] / denom;
        work[col] = cprev;
        z[col] = r[col] / denom;
        for (int k = 1; k < g.Nz; k++) {
            const int c = col + k * NxNy;
            denom = jd[c] - jzm[c] * cprev;
            cprev = (k < g.Nz - 1) ? jzp[c] / denom : 0.0;
            work[c] = cprev;
            z[c] = (r[c] - jzm[c] * z[c - NxNy]) / denom;
        }
        // back substitution
        for (int k = g.Nz - 2; k >= 0; k--) {
            const int c = col + k * NxNy;
            z[c] -= work[c] * z[c + NxNy];
        }
    }

    __global__ void kNkJacApply(const double *J, const double *v, GpuDims g, int N, double *out) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int Nx = g.Nx, NxNy = g.Nx * g.Ny;
        const int nb[6] = {c - 1, c + 1, c - Nx, c + Nx, c - NxNy, c + NxNy};
        double s = J[c] * v[c];
        for (int f = 0; f < 6; f++) {
            const double a = J[(f + 1) * N + c];
            if (a != 0.0)
                s += a * v[nb[f]];
        }
        out[c] = s;
    }

    //! Trial point h(lambda): water-content-space update capped at kMaxDSe for unsaturated cells (or the
    //! plain head step when mapped == 0); mirrors the CPU trial() lambda.
    __global__ void kNkTrial(const double *h, const double *delta, const double *theta_h, const double *C_h, const GpuSoil *props, const double *fixedh, const double *ss, double lambda, int mapped, int N, double *htrial) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        if (!isnan(fixedh[c])) {
            htrial[c] = fixedh[c];
            return;
        }
        const double d = lambda * delta[c];
        const GpuSoil p = props[c];
        // Specific-storage cells that are saturated or on the wet side of the retention-curve inflection
        // ((alpha |h|)^n < m for van Genuchten) take the plain head step, as on the CPU.
        const bool vg = p.model == 0 || p.model == 2;
        const bool wet_ss = ss[c] != 0.0 && (vg ? (h[c] >= p.h_s || pow(p.alpha * fabs(h[c]), p.n) < 1.0 - 1.0 / p.n) : h[c] > 0.0);
        if (!mapped || wet_ss) {
            htrial[c] = h[c] + d;
            return;
        }
        const double kMaxDSe = 0.2;
        const double tr = p.theta_r, ts = p.theta_s, range = ts - tr, hs = p.h_s; // saturated for h >= hs
        if (h[c] < hs) {
            // Map through the retention curve alone: remove the specific-storage part of theta_h and C_h
            double th = theta_h[c], Cc = C_h[c];
            if (ss[c] != 0.0) {
                const double f = 1.0 + ss[c] * h[c] / ts;
                th /= f;
                Cc = (Cc - ss[c] * th / ts) / f;
            }
            // Tiny steps that stay unsaturated take the plain update (no inversion round-off), as on the CPU
            if (h[c] + d < hs && fabs(d) <= 1e-4 * fmax(1.0, -h[c]) && fabs(Cc * d) <= 1e-4 * range) {
                htrial[c] = h[c] + d;
                return;
            }
            const double Se0 = (th - tr) / range;
            double Se = (th + Cc * d - tr) / range;
            Se = fmin(fmax(Se, Se0 - kMaxDSe), Se0 + kMaxDSe);
            htrial[c] = (Se >= 1.0) ? hs : d_hOfTheta(tr + fmax(Se, 1e-9) * range, p);
        } else {
            const double hn = h[c] + d;
            htrial[c] = (hn < hs) ? (h[c] > hs ? hs : fmax(hn, d_hOfTheta(tr + (1.0 - kMaxDSe) * range, p))) : hn;
        }
    }

    //! out[c] = |F| / max(tolA, tolR * scale): > 1 where the cell's residual test fails.
    //! out = F x thk / mean thickness: residual per unit volume weighted by relative cell volume (uniform dx, dy)
    __global__ void kNkVolumeWeight(const double *F, const double *thk, double inv_tbar, int N, double *out) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            out[c] = F[c] * thk[c] * inv_tbar;
    }

    __global__ void kNkResRatio(const double *F, const double *scale, double tolA, double tolR, int N, double *out) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            out[c] = fabs(F[c]) / fmax(tolA, tolR * scale[c]);
    }

    //! out[c] > 1 where the cell's head-change test fails (mirrors the CPU dh_ok test): the change exceeds
    //! tol * max(1, |h|) and, in an unsaturated cell, moves a non-negligible amount of water (C |dh|).
    __global__ void kNkDhRatio(const double *hn, const double *h, const double *C, const GpuSoil *props, double tol, double theta_tol, int N, double *out) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const double dh = fabs(hn[c] - h[c]);
        out[c] = (h[c] < props[c].h_s && C[c] * dh < 0.1 * theta_tol) ? 0.0 : dh / (tol * fmax(1.0, fabs(h[c])));
    }

    __global__ void kPartialMax(const double *a, int N, double *partial) {
        __shared__ double sh[TPB];
        double mx = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
            mx = fmax(mx, a[i]); // NaN-propagating via the caller's isfinite() check on the norm
        sh[threadIdx.x] = mx;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st)
                sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + st]);
            __syncthreads();
        }
        if (threadIdx.x == 0)
            partial[blockIdx.x] = sh[0];
    }


    __global__ void kJacobi(const double *r, const double *diag, int N, double *z) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            z[c] = r[c] / diag[c];
    }

    __global__ void kSub(const double *a, const double *b, int N, double *out) { // out = a - b
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            out[c] = a[c] - b[c];
    }

    __global__ void kAxpy(double *y, const double *x, double s, int N) { // y += s*x
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            y[c] += s * x[c];
    }

    __global__ void kPaxpy(double *p, const double *z, double beta, int N) { // p = z + beta*p
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            p[c] = z[c] + beta * p[c];
    }

    __global__ void kPartialDot(const double *a, const double *b, int N, double *partial) {
        __shared__ double s[TPB];
        double sum = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
            sum += a[i] * b[i];
        s[threadIdx.x] = sum;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st)
                s[threadIdx.x] += s[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            partial[blockIdx.x] = s[0];
    }

    __global__ void kPartialMaxAbsDiff(const double *a, const double *b, int N, double *partial) {
        __shared__ double s[TPB];
        double mx = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
            mx = fmax(mx, fabs(a[i] - b[i]));
        s[threadIdx.x] = mx;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st)
                s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + st]);
            __syncthreads();
        }
        if (threadIdx.x == 0)
            partial[blockIdx.x] = s[0];
    }

    // --- Device-resident CG scalar kernels --------------------------------------
    // Reduce the RB per-block partials to a single device scalar (single block, no
    // atomics — portable across GPU architectures). Launch with <<<1, RB>>>.
    __global__ void kFinalReduceSum(const double *partial, int n, double *out) {
        __shared__ double sh[RB];
        int t = threadIdx.x;
        sh[t] = (t < n) ? partial[t] : 0.0;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (t < st)
                sh[t] += sh[t + st];
            __syncthreads();
        }
        if (t == 0)
            *out = sh[0];
    }

    // alpha = rz / (pAp + eps)   (all operands device-resident)
    __global__ void kAlpha(const double *rz, const double *pAp, double *alpha) {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            *alpha = *rz / (*pAp + 1e-300);
    }

    // beta = rznew / (rz + eps); then rz <- rznew   (order matters)
    __global__ void kBetaShift(const double *rznew, double *rz, double *beta) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *beta = *rznew / (*rz + 1e-300);
            *rz = *rznew;
        }
    }

    __global__ void kAxpyDev(double *y, const double *x, const double *s, double sign, int N) { // y += sign*(*s)*x
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            y[c] += sign * (*s) * x[c];
    }

    __global__ void kPaxpyDev(double *p, const double *z, const double *beta, int N) { // p = z + (*beta)*p
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            p[c] = z[c] + (*beta) * p[c];
    }

    __global__ void kCopy(const double *src, double *dst, int N) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            dst[c] = src[c];
    }

    __global__ void kScale(const double *in, double s, int N, double *out) { // out = s*in
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            out[c] = s * in[c];
    }

    // --- Geometric-multigrid kernels (device V-cycle preconditioner) ------------
    // Geometry/Dirichlet flags for a multigrid level, passed to kernels by value.
    struct MGGeom {
        int Nx, Ny, Nz;
        double ix2, iy2, iz2;
        int dir[6];
    };

    // Face conductances cfx/cfy/cfz from K (harmonic mean). Mirrors mgBuildStencil.
    __global__ void kMGFaceCoef(const double *K, MGGeom gm, int N, double *cfx, double *cfy, double *cfz) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int Nx = gm.Nx, Ny = gm.Ny, Nz = gm.Nz, NxNy = Nx * Ny;
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        cfx[c] = (i < Nx - 1) ? d_hmean(K[c], K[c + 1]) * gm.ix2 : 0.0;
        cfy[c] = (j < Ny - 1) ? d_hmean(K[c], K[c + Nx]) * gm.iy2 : 0.0;
        cfz[c] = (k < Nz - 1) ? d_hmean(K[c], K[c + NxNy]) * gm.iz2 : 0.0;
    }

    // Diagonal = Cdt + neighbor face conductances + Dirichlet 2K/Δ². Mirrors mgBuildStencil.
    __global__ void kMGDiag(const double *K, const double *Cdt, const double *cfx, const double *cfy, const double *cfz, MGGeom gm, int N, double *diag) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int Nx = gm.Nx, Ny = gm.Ny, Nz = gm.Nz, NxNy = Nx * Ny;
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        double d = Cdt[c];
        d += (i < Nx - 1 ? cfx[c] : 0.0) + (i > 0 ? cfx[c - 1] : 0.0);
        d += (j < Ny - 1 ? cfy[c] : 0.0) + (j > 0 ? cfy[c - Nx] : 0.0);
        d += (k < Nz - 1 ? cfz[c] : 0.0) + (k > 0 ? cfz[c - NxNy] : 0.0);
        if (i == 0 && gm.dir[0])
            d += 2.0 * K[c] * gm.ix2;
        if (i == Nx - 1 && gm.dir[1])
            d += 2.0 * K[c] * gm.ix2;
        if (j == 0 && gm.dir[2])
            d += 2.0 * K[c] * gm.iy2;
        if (j == Ny - 1 && gm.dir[3])
            d += 2.0 * K[c] * gm.iy2;
        if (k == 0 && gm.dir[4])
            d += 2.0 * K[c] * gm.iz2;
        if (k == Nz - 1 && gm.dir[5])
            d += 2.0 * K[c] * gm.iz2;
        diag[c] = d;
    }

    // Average fine-block K and Cdt into a coarse cell (semi-coarsening: fx,fy,fz in {1,2}).
    __global__ void kMGAverage(const double *fK, const double *fCdt, int fNx, int fNy, int fNz, int cNx, int cNy, int cNz, int fx, int fy, int fz, int cN, double *cK, double *cCdt) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= cN)
            return;
        const int cNxNy = cNx * cNy;
        const int kc = c / cNxNy, rr = c - kc * cNxNy, jc = rr / cNx, ic = rr - jc * cNx;
        double sk = 0.0, sc = 0.0;
        int cnt = 0;
        for (int dk = 0; dk < fz; dk++)
            for (int dj = 0; dj < fy; dj++)
                for (int di = 0; di < fx; di++) {
                    int fi = fx * ic + di, fj = fy * jc + dj, fk = fz * kc + dk;
                    if (fi < fNx && fj < fNy && fk < fNz) {
                        int f = fi + fNx * (fj + fNy * fk);
                        sk += fK[f];
                        sc += fCdt[f];
                        cnt++;
                    }
                }
        cK[c] = sk / cnt;
        cCdt[c] = sc / cnt;
    }

    // MG matvec: out = A u using the level's 7-point stencil. Mirrors mgMatvec.
    __global__ void kMGMatvec(const double *u, const double *cfx, const double *cfy, const double *cfz, const double *diag, int Nx, int Ny, int Nz, int N, double *out) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int NxNy = Nx * Ny;
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        double a = diag[c] * u[c];
        if (i < Nx - 1)
            a -= cfx[c] * u[c + 1];
        if (i > 0)
            a -= cfx[c - 1] * u[c - 1];
        if (j < Ny - 1)
            a -= cfy[c] * u[c + Nx];
        if (j > 0)
            a -= cfy[c - Nx] * u[c - Nx];
        if (k < Nz - 1)
            a -= cfz[c] * u[c + NxNy];
        if (k > 0)
            a -= cfz[c - NxNy] * u[c - NxNy];
        out[c] = a;
    }

    // Weighted-Jacobi update: u += omega*(b - res)/diag, with res = A u precomputed.
    __global__ void kMGJacobi(double *u, const double *b, const double *res, const double *diag, double omega, int N) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            u[c] += omega * (b[c] - res[c]) / diag[c];
    }

    // Restrict fine residual into coarse b (block average). Mirrors mgVcycle restriction.
    __global__ void kMGRestrict(const double *fres, int fNx, int fNy, int fNz, int cNx, int cNy, int cNz, int fx, int fy, int fz, int cN, double *cb) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= cN)
            return;
        const int cNxNy = cNx * cNy;
        const int kc = c / cNxNy, rr = c - kc * cNxNy, jc = rr / cNx, ic = rr - jc * cNx;
        double s = 0.0;
        int cnt = 0;
        for (int dk = 0; dk < fz; dk++)
            for (int dj = 0; dj < fy; dj++)
                for (int di = 0; di < fx; di++) {
                    int fi = fx * ic + di, fj = fy * jc + dj, fk = fz * kc + dk;
                    if (fi < fNx && fj < fNy && fk < fNz) {
                        s += fres[fi + fNx * (fj + fNy * fk)];
                        cnt++;
                    }
                }
        cb[c] = (cnt > 0) ? s / cnt : 0.0;
    }

    // Prolong coarse correction into fine u (piecewise constant, additive). Mirrors mgVcycle prolongation.
    __global__ void kMGProlong(const double *cu, int cNx, int cNy, int cNz, int fNx, int fNy, int fNz, int fx, int fy, int fz, int fN, double *fu) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= fN)
            return;
        const int fNxNy = fNx * fNy;
        const int kf = c / fNxNy, rr = c - kf * fNxNy, jf = rr / fNx, iff = rr - jf * fNx;
        fu[c] += cu[(iff / fx) + cNx * ((jf / fy) + cNy * (kf / fz))];
    }

    // --- NGS-FAS kernels (nonlinear Gauss-Seidel / full-approximation scheme) -----

    __global__ void kAdd(const double *a, const double *b, int N, double *out) { // out = a + b
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            out[c] = a[c] + b[c];
    }

    // Average the per-cell soil properties from the fine to the coarse level (arithmetic mean of the
    // van Genuchten / Gardner parameters; model taken from a representative fine cell). Mirrors the
    // CPU mgBuild coarsening so the FAS coarse operators see the same upscaled retention.
    __global__ void kMGAverageProps(const GpuSoil *fp, int fNx, int fNy, int fNz, int cNx, int cNy, int cNz, int fx, int fy, int fz, int cN, GpuSoil *cp) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= cN)
            return;
        const int cNxNy = cNx * cNy;
        const int kc = c / cNxNy, rr = c - kc * cNxNy, jc = rr / cNx, ic = rr - jc * cNx;
        double al = 0, n = 0, tr = 0, ts = 0, ks = 0, hs = 0, S1 = 1.0;
        int cnt = 0, model = 0;
        for (int dk = 0; dk < fz; dk++)
            for (int dj = 0; dj < fy; dj++)
                for (int di = 0; di < fx; di++) {
                    int fi = fx * ic + di, fj = fy * jc + dj, fk = fz * kc + dk;
                    if (fi < fNx && fj < fNy && fk < fNz) {
                        const GpuSoil &p = fp[fi + fNx * (fj + fNy * fk)];
                        if (cnt == 0) {
                            model = p.model; // representative model and spline start
                            S1 = p.S1;
                        }
                        al += p.alpha;
                        n += p.n;
                        tr += p.theta_r;
                        ts += p.theta_s;
                        ks += p.K_s;
                        hs += p.h_s;
                        cnt++;
                    }
                }
        double inv = (cnt > 0) ? 1.0 / cnt : 0.0;
        cp[c] = {model, al * inv, n * inv, tr * inv, ts * inv, ks * inv, hs * inv, 1.0, 1.0, S1, 0.0, 0.0};
        if (model == 0 && S1 < 1.0)
            krSplineStart(S1, cp[c].n, cp[c].kr1, cp[c].dkr1);
        if (model == 2 && cp[c].h_s < 0.0) { // air-entry scales of the averaged soil (as airEntryScales())
            const double mm = 1.0 - 1.0 / cp[c].n, xn = pow(-cp[c].alpha * cp[c].h_s, cp[c].n), base = 1.0 + xn;
            cp[c].Sc = pow(base, -mm);
            const double Bc = 1.0 - pow(xn / base, mm);
            cp[c].Fk = 1.0 / (sqrt(cp[c].Sc) * Bc * Bc);
        }
    }

    // One red/black nonlinear Gauss-Seidel sweep over cells of the given color (7-point stencil ->
    // two independent colors). Each cell is a scalar monotone equation a0*theta(psi)+(xi+m_ii)*psi = rhs
    // solved by 1-4 Newton steps; neighbours use newest available values.
    __global__ void kNGSSweep(int color, double *u, const double *b, const double *cfx, const double *cfy, const double *cfz, const double *diag, const double *Cdt, const GpuSoil *props, double Cmin, double a0, double xi, int Nx, int Ny, int Nz,
                              int N) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int NxNy = Nx * Ny;
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        if (((i + j + k) & 1) != color)
            return;
        const double mii = diag[c] - Cdt[c];
        double coup = 0.0;
        if (i < Nx - 1)
            coup += cfx[c] * u[c + 1];
        if (i > 0)
            coup += cfx[c - 1] * u[c - 1];
        if (j < Ny - 1)
            coup += cfy[c] * u[c + Nx];
        if (j > 0)
            coup += cfy[c - Nx] * u[c - Nx];
        if (k < Nz - 1)
            coup += cfz[c] * u[c + NxNy];
        if (k > 0)
            coup += cfz[c - NxNy] * u[c - NxNy];
        const double rhs = b[c] + coup, dia = xi + mii;
        const GpuSoil p0 = props[c];
        double psi = u[c];
        for (int nit = 0; nit < 4; nit++) {
            double K, Cap;
            d_KandC(psi, p0, Cmin, K, Cap);
            const double dpsi = (a0 * d_theta(psi, p0) + dia * psi - rhs) / (a0 * Cap + dia);
            psi -= dpsi;
            if (fabs(dpsi) < 1e-13)
                break;
        }
        u[c] = psi;
    }

    // Nonlinear operator N_L(u) = a0*theta(u) + xi*u + [(diag-Cdt)*u - sum cf*u_nb]  (the FAS defect uses this).
    __global__ void kNGSOperator(const double *u, double *out, const double *cfx, const double *cfy, const double *cfz, const double *diag, const double *Cdt, const GpuSoil *props, double a0, double xi, int Nx, int Ny, int Nz, int N) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= N)
            return;
        const int NxNy = Nx * Ny;
        const int k = c / NxNy, r = c - k * NxNy, j = r / Nx, i = r - j * Nx;
        double diff = (diag[c] - Cdt[c]) * u[c];
        if (i < Nx - 1)
            diff -= cfx[c] * u[c + 1];
        if (i > 0)
            diff -= cfx[c - 1] * u[c - 1];
        if (j < Ny - 1)
            diff -= cfy[c] * u[c + Nx];
        if (j > 0)
            diff -= cfy[c - Nx] * u[c - Nx];
        if (k < Nz - 1)
            diff -= cfz[c] * u[c + NxNy];
        if (k > 0)
            diff -= cfz[c - NxNy] * u[c - NxNy];
        out[c] = a0 * d_theta(u[c], props[c]) + xi * u[c] + diff;
    }

    // Fine-grid RHS g_h = a0*theta^n + xi*h^n + (M_h h^n - Cdt*h^n) - residual(h^n).
    __global__ void kNGSFineRHS(const double *theta_n, const double *h, const double *Mh, const double *Cdt, const double *Rn, double a0, double xi, int N, double *b) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < N)
            b[c] = a0 * theta_n[c] + xi * h[c] + (Mh[c] - Cdt[c] * h[c]) - Rn[c];
    }

} // namespace

// --- Device-state container --------------------------------------------------

// One geometric-multigrid level (device arrays + geometry).
struct MGLevelGpu {
    int Nx, Ny, Nz;
    double dx, dy, dz;
    int dir[6]; // face is Dirichlet (1/0)
    int fx, fy, fz; // coarsening factor from the next-finer level (1 or 2)
    double *K = nullptr, *Cdt = nullptr, *cfx = nullptr, *cfy = nullptr, *cfz = nullptr, *diag = nullptr, *u = nullptr, *b = nullptr, *res = nullptr;
    double *fas = nullptr; // FAS restricted solution I*psi (NGS_FAS only)
    GpuSoil *props = nullptr; // per-level soil (fine = device props; coarse = restricted)
    int N() const {
        return Nx * Ny * Nz;
    }
};

struct SoilSolver3DGpuState {
    int N = 0;
    double *h = nullptr, *hm = nullptr, *u = nullptr;
    double *K = nullptr, *C = nullptr, *theta_m = nullptr, *theta_n = nullptr, *theta_nm1 = nullptr;
    double *b = nullptr, *diag = nullptr, *r = nullptr, *z = nullptr, *p = nullptr, *Ap = nullptr, *src = nullptr;
    double *F = nullptr, *Fp = nullptr, *hp = nullptr; // Newton-Krylov: residual, perturbed residual, perturbed head
    std::vector<double *> gV; // GMRES Krylov basis (gRestart+1 device vectors)
    double *gw = nullptr, *gMv = nullptr, *gAx = nullptr, *gtmp = nullptr; // GMRES work vectors
    int gRestart = 0; // current basis capacity (0 = not yet allocated)
    bool props_uploaded = false; // uploaded on first use; re-uploaded when setSoilProperties() dirties them
    bool mg_props_built = false; // per-level FAS props; rebuilt when props change
    GpuSoil *props = nullptr;
    double *partial = nullptr; // device reduction scratch (RB)
    double *scal = nullptr; // device CG scalars (8): rz,pAp,rnorm2,bnorm2,rznew,alpha,beta
    std::vector<double> host_partial; // host reduction scratch (RB)
    std::vector<MGLevelGpu> mg; // multigrid hierarchy (empty until set up)
    // NEWTON_ANALYTIC (allocated on first use): constitutive values at the iterate, residuals, flux scale,
    // 7-point Jacobian (7*N), Newton step, trial/best heads, theta and C saved at the iterate, BiCGSTAB work.
    double *nk_theta = nullptr, *nk_K = nullptr, *nk_C = nullptr, *nk_dK = nullptr, *nk_F = nullptr, *nk_Ft = nullptr, *nk_scale = nullptr, *nk_J = nullptr;
    double *nk_delta = nullptr, *nk_htrial = nullptr, *nk_hbest = nullptr, *nk_Fbest = nullptr, *nk_thetah = nullptr, *nk_Ch = nullptr, *nk_tmp = nullptr;
    double *bi_r = nullptr, *bi_rhat = nullptr, *bi_p = nullptr, *bi_v = nullptr, *bi_phat = nullptr, *bi_s = nullptr, *bi_shat = nullptr, *bi_t = nullptr;
    double *nk_zc = nullptr, *nk_thk = nullptr, *nk_fixed = nullptr, *nk_ss = nullptr; // per-cell geometry, specified heads, Ss
    double *nk_tmp2 = nullptr; // line-preconditioner scratch
};

namespace {
    inline void allocState(SoilSolver3DGpuState &s, int N) {
        s.N = N;
        size_t bytes = (size_t) N * sizeof(double);
        for (double **pp: {&s.h, &s.hm, &s.u, &s.K, &s.C, &s.theta_m, &s.theta_n, &s.theta_nm1, &s.b, &s.diag, &s.r, &s.z, &s.p, &s.Ap, &s.src, &s.F, &s.Fp, &s.hp})
            CUDA_CHECK(cudaMalloc((void **) pp, bytes));
        CUDA_CHECK(cudaMalloc((void **) &s.props, (size_t) N * sizeof(GpuSoil)));
        CUDA_CHECK(cudaMalloc((void **) &s.partial, RB * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **) &s.scal, 8 * sizeof(double)));
        s.host_partial.assign(RB, 0.0);
    }
    inline void freeState(SoilSolver3DGpuState &s) {
        for (double *p: {s.h, s.hm, s.u, s.K, s.C, s.theta_m, s.theta_n, s.theta_nm1, s.b, s.diag, s.r, s.z, s.p, s.Ap, s.src, s.F, s.Fp, s.hp, s.partial, s.scal})
            cudaFree(p);
        for (double *p: s.gV)
            cudaFree(p);
        for (double *p: {s.gw, s.gMv, s.gAx, s.gtmp})
            cudaFree(p);
        cudaFree(s.props);
        for (double *p: {s.nk_theta, s.nk_K, s.nk_C,    s.nk_dK, s.nk_F, s.nk_Ft,   s.nk_scale, s.nk_J,    s.nk_delta, s.nk_htrial, s.nk_hbest, s.nk_Fbest, s.nk_thetah, s.nk_Ch,
                         s.nk_tmp,   s.bi_r, s.bi_rhat, s.bi_p,  s.bi_v, s.bi_phat, s.bi_s,     s.bi_shat, s.bi_t,     s.nk_zc,     s.nk_thk,   s.nk_fixed, s.nk_ss,     s.nk_tmp2})
            cudaFree(p);
        for (auto &L: s.mg) {
            for (double *p: {L.K, L.Cdt, L.cfx, L.cfy, L.cfz, L.diag, L.u, L.b, L.res, L.fas})
                cudaFree(p);
            cudaFree(L.props);
        }
    }
    inline void allocNewtonState(SoilSolver3DGpuState &s, int N) {
        if (s.nk_F)
            return;
        size_t bytes = (size_t) N * sizeof(double);
        for (double **pp: {&s.nk_theta, &s.nk_K,    &s.nk_C, &s.nk_dK, &s.nk_F,    &s.nk_Ft, &s.nk_scale, &s.nk_delta, &s.nk_htrial, &s.nk_hbest, &s.nk_Fbest, &s.nk_thetah, &s.nk_Ch,  &s.nk_tmp,
                           &s.bi_r,     &s.bi_rhat, &s.bi_p, &s.bi_v,  &s.bi_phat, &s.bi_s,  &s.bi_shat,  &s.bi_t,     &s.nk_zc,     &s.nk_thk,   &s.nk_fixed, &s.nk_ss,     &s.nk_tmp2})
            CUDA_CHECK(cudaMalloc((void **) pp, bytes));
        CUDA_CHECK(cudaMalloc((void **) &s.nk_J, 7 * bytes));
    }

    //! Build a kernel MGGeom (geometry + Dirichlet flags) from a multigrid level.
    inline MGGeom mgGeomOf(const MGLevelGpu &L) {
        MGGeom gm;
        gm.Nx = L.Nx;
        gm.Ny = L.Ny;
        gm.Nz = L.Nz;
        gm.ix2 = 1.0 / (L.dx * L.dx);
        gm.iy2 = 1.0 / (L.dy * L.dy);
        gm.iz2 = 1.0 / (L.dz * L.dz);
        for (int f = 0; f < 6; f++)
            gm.dir[f] = L.dir[f];
        return gm;
    }

    //! Reduce the per-block partials on the host (sum or max).
    inline double reduceHost(SoilSolver3DGpuState &s, bool maximum) {
        CUDA_CHECK_READBACK(cudaMemcpy(s.host_partial.data(), s.partial, RB * sizeof(double), cudaMemcpyDeviceToHost));
        double acc = maximum ? 0.0 : 0.0;
        for (int i = 0; i < RB; i++)
            acc = maximum ? fmax(acc, s.host_partial[i]) : acc + s.host_partial[i];
        return acc;
    }
} // namespace

namespace {
    __global__ void kProbe(int *out) {
        *out = 1;
    }
} // namespace

bool SoilSolver3D::gpuKernelsRun() {
    // A visible device may still be unable to run this build's kernels (no kernel image for its architecture, or a
    // device in exclusive-process mode); launch a trivial kernel to find out. Errors here are answered by the return value.
    int *device_flag = nullptr;
    if (cudaMalloc(&device_flag, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    kProbe<<<1, 1>>>(device_flag);
    int host_flag = 0;
    const bool ran = cudaGetLastError() == cudaSuccess && cudaMemcpy(&host_flag, device_flag, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess && host_flag == 1;
    cudaFree(device_flag);
    cudaGetLastError();
    return ran;
}

void SoilSolver3D::releaseGpuState() {
    if (gpu_) {
        freeState(*gpu_); // never throws: cudaFree results are not checked
        delete gpu_;
        gpu_ = nullptr;
    }
    cudaGetLastError(); // do not leave a non-sticky error behind
}

SoilSolver3D::~SoilSolver3D() {
    releaseGpuState();
}

void SoilSolver3D::mgSetupGPU() {
    SoilSolver3DGpuState &s = *gpu_;
    if (!s.mg.empty())
        return;
    // Level 0 = the fine grid.
    MGLevelGpu L0;
    L0.Nx = (int) Nx_;
    L0.Ny = (int) Ny_;
    L0.Nz = (int) Nz_;
    L0.dx = dx_;
    L0.dy = dy_;
    L0.dz = dz_;
    for (int f = 0; f < 6; f++)
        L0.dir[f] = (bc_[f].type == BCType::DIRICHLET) ? 1 : 0;
    L0.fx = 1;
    L0.fy = 1;
    L0.fz = 1;
    s.mg.push_back(L0);
    // Per-dimension semi-coarsening (mirrors the CPU mgBuild hierarchy).
    while (true) {
        const MGLevelGpu &F = s.mg.back();
        int cnx = (F.Nx > params_.mg_min_dim) ? (F.Nx + 1) / 2 : F.Nx;
        int cny = (F.Ny > params_.mg_min_dim) ? (F.Ny + 1) / 2 : F.Ny;
        int cnz = (F.Nz > params_.mg_min_dim) ? (F.Nz + 1) / 2 : F.Nz;
        if (cnx == F.Nx && cny == F.Ny && cnz == F.Nz)
            break;
        MGLevelGpu C;
        C.fx = (cnx < F.Nx) ? 2 : 1;
        C.fy = (cny < F.Ny) ? 2 : 1;
        C.fz = (cnz < F.Nz) ? 2 : 1;
        C.Nx = cnx;
        C.Ny = cny;
        C.Nz = cnz;
        C.dx = F.dx * C.fx;
        C.dy = F.dy * C.fy;
        C.dz = F.dz * C.fz;
        for (int f = 0; f < 6; f++)
            C.dir[f] = F.dir[f];
        s.mg.push_back(C);
    }
    const bool need_props = (params_.nonlinear_method == NonlinearMethod::NGS_FAS);
    for (auto &L: s.mg) {
        size_t bytes = (size_t) L.N() * sizeof(double);
        for (double **pp: {&L.K, &L.Cdt, &L.cfx, &L.cfy, &L.cfz, &L.diag, &L.u, &L.b, &L.res})
            CUDA_CHECK(cudaMalloc((void **) pp, bytes));
        if (need_props) { // FAS needs per-level retention (smoother) and a restricted-solution buffer
            CUDA_CHECK(cudaMalloc((void **) &L.fas, bytes));
            CUDA_CHECK(cudaMalloc((void **) &L.props, (size_t) L.N() * sizeof(GpuSoil)));
        }
    }
}

void SoilSolver3D::mgBuildGPU(double a0) {
    SoilSolver3DGpuState &s = *gpu_;
    if (s.mg.empty())
        mgSetupGPU();
    // Refresh Dirichlet flags from the current bc_ (mgSetupGPU captures them only once, but setBoundary()
    // can change BC types between solves; the coarse stencils must see the current flags). Cheap.
    for (int f = 0; f < 6; f++)
        s.mg[0].dir[f] = (bc_[f].type == BCType::DIRICHLET) ? 1 : 0;
    for (size_t l = 1; l < s.mg.size(); l++)
        for (int f = 0; f < 6; f++)
            s.mg[l].dir[f] = s.mg[l - 1].dir[f];
    // Level 0 reuses the solver's K and exact diagonal; Cdt = a0 * C.
    MGLevelGpu &L0 = s.mg[0];
    int N0 = L0.N(), blk0 = (N0 + TPB - 1) / TPB;
    CUDA_CHECK(cudaMemcpyAsync(L0.K, s.K, (size_t) N0 * sizeof(double), cudaMemcpyDeviceToDevice, 0));
    kScale<<<blk0, TPB>>>(s.C, a0, N0, L0.Cdt);
    CUDA_CHECK(cudaMemcpyAsync(L0.diag, s.diag, (size_t) N0 * sizeof(double), cudaMemcpyDeviceToDevice, 0));
    MGGeom gm0 = mgGeomOf(L0);
    kMGFaceCoef<<<blk0, TPB>>>(L0.K, gm0, N0, L0.cfx, L0.cfy, L0.cfz);
    // Coarser levels: average K/Cdt then re-discretize the stencil.
    for (size_t l = 1; l < s.mg.size(); l++) {
        MGLevelGpu &F = s.mg[l - 1];
        MGLevelGpu &C = s.mg[l];
        int cN = C.N(), blk = (cN + TPB - 1) / TPB;
        kMGAverage<<<blk, TPB>>>(F.K, F.Cdt, F.Nx, F.Ny, F.Nz, C.Nx, C.Ny, C.Nz, C.fx, C.fy, C.fz, cN, C.K, C.Cdt);
        MGGeom gm = mgGeomOf(C);
        kMGFaceCoef<<<blk, TPB>>>(C.K, gm, cN, C.cfx, C.cfy, C.cfz);
        kMGDiag<<<blk, TPB>>>(C.K, C.Cdt, C.cfx, C.cfy, C.cfz, gm, cN, C.diag);
    }
    // Per-level retention for the FAS smoother (constant in time -> build once). Fine = device props,
    // coarse = arithmetic-averaged (mirrors the CPU mgBuild coarsening).
    if (params_.nonlinear_method == NonlinearMethod::NGS_FAS && !s.mg_props_built) {
        CUDA_CHECK(cudaMemcpyAsync(s.mg[0].props, s.props, (size_t) N0 * sizeof(GpuSoil), cudaMemcpyDeviceToDevice, 0));
        for (size_t l = 1; l < s.mg.size(); l++) {
            MGLevelGpu &F = s.mg[l - 1];
            MGLevelGpu &C = s.mg[l];
            int cN = C.N(), blk = (cN + TPB - 1) / TPB;
            kMGAverageProps<<<blk, TPB>>>(F.props, F.Nx, F.Ny, F.Nz, C.Nx, C.Ny, C.Nz, C.fx, C.fy, C.fz, cN, C.props);
        }
        s.mg_props_built = true;
    }
}

void SoilSolver3D::mgVcycleGPU(int lvl) {
    SoilSolver3DGpuState &s = *gpu_;
    MGLevelGpu &L = s.mg[lvl];
    int N = L.N(), blk = (N + TPB - 1) / TPB;
    const double omega = 0.8;
    auto smooth = [&](int sweeps) { // weighted-Jacobi sweeps
        for (int sw = 0; sw < sweeps; sw++) {
            kMGMatvec<<<blk, TPB>>>(L.u, L.cfx, L.cfy, L.cfz, L.diag, L.Nx, L.Ny, L.Nz, N, L.res);
            kMGJacobi<<<blk, TPB>>>(L.u, L.b, L.res, L.diag, omega, N);
        }
    };
    if (lvl == (int) s.mg.size() - 1) {
        smooth(params_.mg_coarse_sweeps);
        return;
    } // coarsest
    smooth(params_.mg_pre);
    kMGMatvec<<<blk, TPB>>>(L.u, L.cfx, L.cfy, L.cfz, L.diag, L.Nx, L.Ny, L.Nz, N, L.res);
    kSub<<<blk, TPB>>>(L.b, L.res, N, L.res); // res = b - A u
    MGLevelGpu &C = s.mg[lvl + 1];
    int cN = C.N(), cblk = (cN + TPB - 1) / TPB;
    kMGRestrict<<<cblk, TPB>>>(L.res, L.Nx, L.Ny, L.Nz, C.Nx, C.Ny, C.Nz, C.fx, C.fy, C.fz, cN, C.b);
    CUDA_CHECK(cudaMemsetAsync(C.u, 0, (size_t) cN * sizeof(double), 0));
    mgVcycleGPU(lvl + 1);
    kMGProlong<<<blk, TPB>>>(C.u, C.Nx, C.Ny, C.Nz, L.Nx, L.Ny, L.Nz, C.fx, C.fy, C.fz, N, L.u);
    smooth(params_.mg_post);
}

void SoilSolver3D::mgApplyGPU(double *z, const double *r) {
    SoilSolver3DGpuState &s = *gpu_;
    MGLevelGpu &L0 = s.mg[0];
    int N0 = L0.N(), blk0 = (N0 + TPB - 1) / TPB;
    kCopy<<<blk0, TPB>>>(r, L0.b, N0); // b_0 = r
    CUDA_CHECK(cudaMemsetAsync(L0.u, 0, (size_t) N0 * sizeof(double), 0)); // u_0 = 0
    mgVcycleGPU(0);
    kCopy<<<blk0, TPB>>>(L0.u, z, N0); // z = u_0
}

// FAS V-cycle for the lagged-K nonlinear system (red-black NGS smoother). Mirrors the CPU mgVcycleFAS:
// pre-smooth, form the nonlinear defect d = b - N_L(u), restrict the solution (-> C.fas) and the defect,
// set the coarse RHS N_C(I psi) + I d, recurse from the restricted solution, prolong the FAS correction
// (u_C - I psi), post-smooth.
void SoilSolver3D::mgVcycleFASGPU(int lvl, double a0, double xi) {
    SoilSolver3DGpuState &s = *gpu_;
    MGLevelGpu &L = s.mg[lvl];
    const double Cmin = params_.C_min;
    int N = L.N(), blk = (N + TPB - 1) / TPB;
    auto sweep = [&](MGLevelGpu &Lv, int sweeps) { // red-black nonlinear Gauss-Seidel
        int Nv = Lv.N(), b = (Nv + TPB - 1) / TPB;
        for (int sw = 0; sw < sweeps; sw++) {
            kNGSSweep<<<b, TPB>>>(0, Lv.u, Lv.b, Lv.cfx, Lv.cfy, Lv.cfz, Lv.diag, Lv.Cdt, Lv.props, Cmin, a0, xi, Lv.Nx, Lv.Ny, Lv.Nz, Nv);
            kNGSSweep<<<b, TPB>>>(1, Lv.u, Lv.b, Lv.cfx, Lv.cfy, Lv.cfz, Lv.diag, Lv.Cdt, Lv.props, Cmin, a0, xi, Lv.Nx, Lv.Ny, Lv.Nz, Nv);
        }
    };
    if (lvl == (int) s.mg.size() - 1) {
        sweep(L, params_.mg_coarse_sweeps);
        return;
    } // coarsest
    sweep(L, params_.mg_pre);
    kNGSOperator<<<blk, TPB>>>(L.u, L.res, L.cfx, L.cfy, L.cfz, L.diag, L.Cdt, L.props, a0, xi, L.Nx, L.Ny, L.Nz, N); // res = N_L(u)
    kSub<<<blk, TPB>>>(L.b, L.res, N, L.res); // res = defect d = b - N_L(u)
    MGLevelGpu &C = s.mg[lvl + 1];
    int cN = C.N(), cblk = (cN + TPB - 1) / TPB;
    kMGRestrict<<<cblk, TPB>>>(L.u, L.Nx, L.Ny, L.Nz, C.Nx, C.Ny, C.Nz, C.fx, C.fy, C.fz, cN, C.fas); // C.fas = I psi
    kMGRestrict<<<cblk, TPB>>>(L.res, L.Nx, L.Ny, L.Nz, C.Nx, C.Ny, C.Nz, C.fx, C.fy, C.fz, cN, C.res); // C.res = I d
    kNGSOperator<<<cblk, TPB>>>(C.fas, C.b, C.cfx, C.cfy, C.cfz, C.diag, C.Cdt, C.props, a0, xi, C.Nx, C.Ny, C.Nz, cN); // C.b = N_C(I psi)
    kAdd<<<cblk, TPB>>>(C.b, C.res, cN, C.b); // C.b += I d
    kCopy<<<cblk, TPB>>>(C.fas, C.u, cN); // FAS initial guess u_C = I psi
    mgVcycleFASGPU(lvl + 1, a0, xi);
    kSub<<<cblk, TPB>>>(C.u, C.fas, cN, C.res); // correction e = u_C - I psi (reuse C.res)
    kMGProlong<<<blk, TPB>>>(C.res, C.Nx, C.Ny, C.Nz, L.Nx, L.Ny, L.Nz, C.fx, C.fy, C.fz, N, L.u); // L.u += P e
    sweep(L, params_.mg_post);
}

// NGS-FAS step on the GPU. State (h, theta_n, theta_nm1, net src, props) is already uploaded by solveGPU.
uint SoilSolver3D::solveNGSFASGPU(double dt, double /*a0_in*/, double /*a1_in*/, double /*a2_in*/) {
    // NGS-FAS is semi-implicit backward-Euler by construction (lagged K, BDF1), matching the CPU
    // solveNGSFAS which forces a0=1/dt, a1=-1/dt, a2=0 regardless of the BDF2 coefficients solve()
    // computes once history exists. Override here so the GPU path uses the same discretization.
    SoilSolver3DGpuState &s = *gpu_;
    const int N = s.N, blk = (N + TPB - 1) / TPB;
    const double a0 = 1.0 / dt, a1 = -1.0 / dt, a2 = 0.0;
    const double Cmin = params_.C_min, xi = params_.ngs_xi;
    GpuDims g{(int) Nx_, (int) Ny_, (int) Nz_, dx_, dy_, dz_};
    GpuBC bc;
    for (int f = 0; f < 6; f++) {
        bc.type[f] = (int) bc_[f].type;
        bc.val[f] = bc_[f].value;
    }

    kWaterContent<<<blk, TPB>>>(s.h, s.props, N, s.theta_n); // theta^n = theta(h^n)
    // Lag the diffusion operator at h^n; fill diag (mgBuildGPU's L0 reuses it); build the hierarchy.
    kUpdateKC<<<blk, TPB>>>(s.h, s.props, Cmin, N, s.K, s.C, s.theta_m);
    kBuildSystem<<<blk, TPB>>>(s.h, s.K, s.C, s.theta_m, s.theta_n, s.theta_nm1, s.src, s.props, a0, a1, a2, g, bc, N, s.diag, s.b);
    mgBuildGPU(a0);
    MGLevelGpu &L0 = s.mg[0];

    // Fine-grid RHS g_h = a0*theta^n + xi*h^n + (M_h h^n - Cdt*h^n) - residual(h^n).
    kResidual<<<blk, TPB>>>(s.h, s.props, s.theta_n, s.theta_nm1, s.src, a0, a1, a2, Cmin, g, bc, N, s.F); // Rn
    kMGMatvec<<<blk, TPB>>>(s.h, L0.cfx, L0.cfy, L0.cfz, L0.diag, L0.Nx, L0.Ny, L0.Nz, N, s.z); // Mh = M_h h^n
    kNGSFineRHS<<<blk, TPB>>>(s.theta_n, s.h, s.z, L0.Cdt, s.F, a0, xi, N, L0.b);
    kCopy<<<blk, TPB>>>(s.h, L0.u, N); // initial iterate u = h^n

    uint result = 0;
    for (uint cyc = 0; cyc < params_.picard_max; cyc++) {
        kCopy<<<blk, TPB>>>(L0.u, s.hp, N); // uprev = u
        mgVcycleFASGPU(0, a0, xi);
        kPartialMaxAbsDiff<<<RB, TPB>>>(L0.u, s.hp, N, s.partial);
        double dmax = reduceHost(s, true);
        if (dmax < params_.picard_tol) {
            result = cyc + 1;
            break;
        }
    }
    kCopy<<<blk, TPB>>>(L0.u, s.h, N); // h = u (converged or best-effort)

    if (result != 0) { // advance history: theta^{n-1} <- theta^n, theta^n <- theta(h)
        std::swap(s.theta_n, s.theta_nm1);
        kWaterContent<<<blk, TPB>>>(s.h, s.props, N, s.theta_n);
    }
    CUDA_CHECK_READBACK(cudaMemcpy(h_.data(), s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
    if (result != 0) {
        CUDA_CHECK_READBACK(cudaMemcpy(theta_n_.data(), s.theta_n, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_READBACK(cudaMemcpy(theta_nm1_.data(), s.theta_nm1, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        dt_prev_ = dt;
        have_history_ = true;
    }
    return result;
}

uint SoilSolver3D::solveGPU(double dt, double a0, double a1, double a2, uint *cg_iterations) {
    const int N = (int) numCells();
    const int eblocks = (N + TPB - 1) / TPB; // grid for element-wise kernels

    // Lazily allocate device state on first use.
    if (!gpu_) {
        gpu_ = new SoilSolver3DGpuState();
        try {
            allocState(*gpu_, N);
        } catch (...) {
            releaseGpuState(); // no half-allocated state left behind
            throw;
        }
    }
    SoilSolver3DGpuState &s = *gpu_;

    // Pack and upload host state for this step.
    GpuDims g{(int) Nx_, (int) Ny_, (int) Nz_, dx_, dy_, dz_};
    GpuBC bc;
    for (int f = 0; f < 6; f++) {
        bc.type[f] = (int) bc_[f].type;
        bc.val[f] = bc_[f].value;
    }
    if (!s.props_uploaded || gpu_props_dirty_) { // upload once, then only when setSoilProperties() changed them
        std::vector<GpuSoil> hp(N);
        for (int c = 0; c < N; c++)
            hp[c] = toGpuSoil(props_[c]);
        CUDA_CHECK(cudaMemcpy(s.props, hp.data(), (size_t) N * sizeof(GpuSoil), cudaMemcpyHostToDevice));
        s.props_uploaded = true;
        s.mg_props_built = false; // coarse-level FAS retention is derived from props -> rebuild it too
        gpu_props_dirty_ = false;
    }
    CUDA_CHECK(cudaMemcpy(s.h, h_.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.theta_n, theta_n_.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.theta_nm1, theta_nm1_.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    // Net external source: src - uptake - evap (the GPU kernels carry only one 'src' array, so
    // fold the root-uptake and evaporation sinks in here exactly as buildSystem()/residual() do).
    {
        std::vector<double> net(N);
        for (int c = 0; c < N; c++)
            net[c] = src_[c] - uptake_[c] - evap_[c];
        CUDA_CHECK(cudaMemcpy(s.src, net.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    }

    const double Cmin = params_.C_min;
    CUDA_CHECK(cudaMemcpyToSymbol(d_Kmin, &params_.K_min, sizeof(double))); // conductivity floor (before dispatch)
    {
        int kw = params_.K_weighting;
        CUDA_CHECK(cudaMemcpyToSymbol(d_Kweight, &kw, sizeof(int)));
    } // interblock K weighting
    CUDA_CHECK(cudaMemcpy(s.hm, s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // hm = h

    // Newton paths (state is uploaded above; they do their own epilogue).
    picard_fallback_ = false;
    if (params_.nonlinear_method == NonlinearMethod::NEWTON_ANALYTIC) {
        const uint it = solveNewtonAnalyticGPU(dt, a0, a1, a2, cg_iterations);
        if (it > 0 || in_adaptive_ || !uniform_geometry_ || n_fixed_ > 0 || has_ss_) // the Picard path assumes a uniform grid, no Ss
            return it;
        // Stalled on a caller-fixed dt: fall back to Modified Picard from h^n (as on the CPU). solveStep() checks
        // the water balance of a converged fallback step (picard_fallback_).
        CUDA_CHECK(cudaMemcpy(s.hm, s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        picard_fallback_ = true;
    }
    if (params_.nonlinear_method == NonlinearMethod::NEWTON) {
        return solveNewtonGPU(dt, a0, a1, a2);
    }
    if (params_.nonlinear_method == NonlinearMethod::PICARD_NEWTON) {
        return solveNonlinearSwitchGPU(dt, a0, a1, a2);
    }
    if (params_.nonlinear_method == NonlinearMethod::NGS_FAS) {
        return solveNGSFASGPU(dt, a0, a1, a2);
    }

    uint total_cg = 0;
    uint result = 0;
    for (uint pic = 0; pic < params_.picard_max; pic++) {
        kUpdateKC<<<eblocks, TPB>>>(s.hm, s.props, Cmin, N, s.K, s.C, s.theta_m);
        kBuildSystem<<<eblocks, TPB>>>(s.hm, s.K, s.C, s.theta_m, s.theta_n, s.theta_nm1, s.src, s.props, a0, a1, a2, g, bc, N, s.diag, s.b);
        if (params_.use_multigrid)
            mgBuildGPU(a0); // rebuild MG coefficients (K changed this Picard iter)
        CUDA_CHECK(cudaMemcpy(s.u, s.hm, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // u = hm

        // --- Jacobi-preconditioned CG (device-resident scalars) -------------
        // alpha/beta and all dot products reduce to device scalars and feed the
        // next kernels with no host round-trip, removing 2 of the 3 host syncs
        // per iteration. The residual test still reads one 8-byte scalar each
        // iteration so the convergence point is caught exactly (checking less
        // often overshoots it: the recursive residual drifts and CG grinds on).
        constexpr int CG_CHECK = 1;
        double *d_rz = s.scal + 0;
        double *d_pAp = s.scal + 1;
        double *d_rnorm2 = s.scal + 2;
        double *d_bnorm2 = s.scal + 3;
        double *d_rznew = s.scal + 4;
        double *d_alpha = s.scal + 5;
        double *d_beta = s.scal + 6;
        auto ddot = [&](const double *a, const double *b, double *out) { // *out = a . b (device-resident)
            kPartialDot<<<RB, TPB>>>(a, b, N, s.partial);
            kFinalReduceSum<<<1, RB>>>(s.partial, RB, out);
        };
        auto precond = [&]() { // z = M^{-1} r  (geometric multigrid V-cycle or Jacobi)
            if (params_.use_multigrid)
                mgApplyGPU(s.z, s.r);
            else
                kJacobi<<<eblocks, TPB>>>(s.r, s.diag, N, s.z);
        };
        kMatvec<<<eblocks, TPB>>>(s.u, s.Ap, s.K, s.C, s.props, s.hm, a0, g, bc, N);
        ddot(s.b, s.b, d_bnorm2);
        double bnorm2_h = 0.0;
        CUDA_CHECK_READBACK(cudaMemcpy(&bnorm2_h, d_bnorm2, sizeof(double), cudaMemcpyDeviceToHost));
        double bnorm = std::sqrt(bnorm2_h) + 1e-300;
        kSub<<<eblocks, TPB>>>(s.b, s.Ap, N, s.r); // r = b - A u
        precond(); // z = M^{-1} r
        CUDA_CHECK(cudaMemcpy(s.p, s.z, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // p = z
        ddot(s.r, s.z, d_rz);
        uint it = 0;
        for (; it < params_.cg_max; it++) {
            kMatvec<<<eblocks, TPB>>>(s.p, s.Ap, s.K, s.C, s.props, s.hm, a0, g, bc, N);
            ddot(s.p, s.Ap, d_pAp);
            kAlpha<<<1, 1>>>(d_rz, d_pAp, d_alpha); // alpha = rz / pAp
            kAxpyDev<<<eblocks, TPB>>>(s.u, s.p, d_alpha, 1.0, N); // u += alpha p
            kAxpyDev<<<eblocks, TPB>>>(s.r, s.Ap, d_alpha, -1.0, N); // r -= alpha A p
            if ((it % CG_CHECK) == CG_CHECK - 1 || it == params_.cg_max - 1) {
                ddot(s.r, s.r, d_rnorm2);
                double rnorm2_h = 0.0;
                CUDA_CHECK_READBACK(cudaMemcpy(&rnorm2_h, d_rnorm2, sizeof(double), cudaMemcpyDeviceToHost));
                if (std::sqrt(rnorm2_h) / bnorm < params_.cg_tol) {
                    it++;
                    break;
                }
            }
            precond(); // z = M^{-1} r
            ddot(s.r, s.z, d_rznew);
            kBetaShift<<<1, 1>>>(d_rznew, d_rz, d_beta); // beta = rznew/rz; rz <- rznew
            kPaxpyDev<<<eblocks, TPB>>>(s.p, s.z, d_beta, N); // p = z + beta p
        }
        total_cg += it;

        // --- Picard convergence test: max |u - hm| --------------------------
        kPartialMaxAbsDiff<<<RB, TPB>>>(s.u, s.hm, N, s.partial);
        double dmax = reduceHost(s, true);
        CUDA_CHECK(cudaMemcpy(s.hm, s.u, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // hm = u
        if (dmax < params_.picard_tol) {
            // Accept: advance history (theta^{n-1} <- theta^n, theta^n <- theta^{n+1}).
            std::swap(s.theta_n, s.theta_nm1);
            kWaterContent<<<eblocks, TPB>>>(s.hm, s.props, N, s.theta_n);
            CUDA_CHECK(cudaMemcpy(s.h, s.hm, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
            result = pic + 1;
            break;
        }
    }

    if (result == 0 && params_.nonlinear_method == NonlinearMethod::NEWTON_ANALYTIC) {
        // A failed Picard fallback of the Newton solver leaves the state unchanged, as on the CPU (the caller
        // sub-steps the interval or reports the failure); its last iterate can be far off.
        if (cg_iterations)
            *cg_iterations = total_cg;
        return 0;
    }
    if (result == 0) { // non-convergence: best-effort h, do not advance history
        CUDA_CHECK(cudaMemcpy(s.h, s.hm, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Copy results back to host.
    CUDA_CHECK_READBACK(cudaMemcpy(h_.data(), s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
    if (result != 0) {
        CUDA_CHECK_READBACK(cudaMemcpy(theta_n_.data(), s.theta_n, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_READBACK(cudaMemcpy(theta_nm1_.data(), s.theta_nm1, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        dt_prev_ = dt;
        have_history_ = true;
    }
    if (cg_iterations)
        *cg_iterations = total_cg;
    return result;
}

// --- Newton-Krylov GPU building blocks (keystone: residual + matrix-free Jacobian) ----------
// Test/diagnostic wrappers: upload the current host state, run the device kernels, download the
// result. They use props/theta_n_/theta_nm1_/src_ from the current host state. The device-
// resident solver loop (gmresGPU/solveNewtonGPU) will reuse kResidual without per-call transfers.

namespace {
    // Device dot product a.b reduced to a host scalar (reuses the CG reduction scratch).
    inline double gpuDot(SoilSolver3DGpuState &s, const double *a, const double *b, int N) {
        kPartialDot<<<RB, TPB>>>(a, b, N, s.partial);
        kFinalReduceSum<<<1, RB>>>(s.partial, RB, s.scal);
        double h = 0.0;
        CUDA_CHECK_READBACK(cudaMemcpy(&h, s.scal, sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }
    // Device residual R(h) -> dF.
    inline void gpuResidual(SoilSolver3DGpuState &s, const double *d_h, double *d_F, double a0, double a1, double a2, double Cmin, GpuDims g, GpuBC bc, int N, int eblocks) {
        kResidual<<<eblocks, TPB>>>(d_h, s.props, s.theta_n, s.theta_nm1, s.src, a0, a1, a2, Cmin, g, bc, N, d_F);
    }
    // Device matrix-free Jacobian-vector product J(h) v ~ (R(h+eps v)-R(h))/eps. d_F0 = R(h) (precomputed).
    inline void gpuJacvec(SoilSolver3DGpuState &s, const double *d_h, const double *d_F0, const double *d_v, double *d_Jv, double a0, double a1, double a2, double Cmin, GpuDims g, GpuBC bc, int N, int eblocks) {
        const double vnorm = std::sqrt(gpuDot(s, d_v, d_v, N));
        if (vnorm < 1e-300) {
            CUDA_CHECK(cudaMemset(d_Jv, 0, (size_t) N * sizeof(double)));
            return;
        }
        const double hnorm = std::sqrt(gpuDot(s, d_h, d_h, N));
        const double eps = 1.49011611938e-8 * (1.0 + hnorm) / vnorm;
        CUDA_CHECK(cudaMemcpy(s.hp, d_h, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        kAxpy<<<eblocks, TPB>>>(s.hp, d_v, eps, N); // hp = h + eps*v
        gpuResidual(s, s.hp, s.Fp, a0, a1, a2, Cmin, g, bc, N, eblocks); // Fp = R(h+eps v)
        kSub<<<eblocks, TPB>>>(s.Fp, d_F0, N, d_Jv); // Jv = Fp - F0
        kScale<<<eblocks, TPB>>>(d_Jv, 1.0 / eps, N, d_Jv); // Jv /= eps
    }
    // Right-preconditioned restarted GMRES(restart) on device pointers. apply(x,y): y=A x; precond(r,z):
    // z=M^{-1} r. Krylov basis is device-resident; Arnoldi dots reduce to host, Givens/Hessenberg on host.
    inline uint gmresGPU(SoilSolver3DGpuState &s, const std::function<void(const double *, double *)> &apply, const std::function<void(const double *, double *)> &precond, const double *d_b, double *d_x, int restart, int maxit, double tol, int N,
                         int eblocks) {
        const int m = std::max(1, restart);
        const size_t bytes = (size_t) N * sizeof(double);
        // Persistent basis + work vectors (allocated once, grown if a larger restart is requested).
        if (s.gRestart < m) {
            for (double *p: s.gV)
                cudaFree(p);
            s.gV.assign(m + 1, nullptr);
            for (int i = 0; i <= m; i++)
                CUDA_CHECK(cudaMalloc((void **) &s.gV[i], bytes));
            if (!s.gw) {
                CUDA_CHECK(cudaMalloc((void **) &s.gw, bytes));
                CUDA_CHECK(cudaMalloc((void **) &s.gMv, bytes));
                CUDA_CHECK(cudaMalloc((void **) &s.gAx, bytes));
                CUDA_CHECK(cudaMalloc((void **) &s.gtmp, bytes));
            }
            s.gRestart = m;
        }
        std::vector<double *> &V = s.gV;
        double *w = s.gw, *Mv = s.gMv, *Ax = s.gAx, *tmp = s.gtmp;

        const double bnorm = std::sqrt(gpuDot(s, d_b, d_b, N));
        uint total = 0;
        if (bnorm >= 1e-300) {
            const double thresh = tol * bnorm;
            std::vector<double> H((m + 1) * m, 0.0), cs(m, 0.0), sn(m, 0.0), gg(m + 1, 0.0);
            while (total < (uint) maxit) {
                apply(d_x, Ax);
                kSub<<<eblocks, TPB>>>(d_b, Ax, N, V[0]); // V0 = b - A x
                double beta = std::sqrt(gpuDot(s, V[0], V[0], N));
                if (beta <= thresh)
                    break;
                kScale<<<eblocks, TPB>>>(V[0], 1.0 / beta, N, V[0]);
                std::fill(gg.begin(), gg.end(), 0.0);
                gg[0] = beta;
                int j = 0;
                for (; j < m && total < (uint) maxit; j++) {
                    total++;
                    precond(V[j], Mv);
                    apply(Mv, w);
                    for (int i = 0; i <= j; i++) {
                        double hij = gpuDot(s, w, V[i], N);
                        H[i + (m + 1) * j] = hij;
                        kAxpy<<<eblocks, TPB>>>(w, V[i], -hij, N); // w -= hij V_i
                    }
                    double hnext = std::sqrt(gpuDot(s, w, w, N));
                    H[(j + 1) + (m + 1) * j] = hnext;
                    if (hnext > 1e-300)
                        kScale<<<eblocks, TPB>>>(w, 1.0 / hnext, N, V[j + 1]);
                    else
                        CUDA_CHECK(cudaMemset(V[j + 1], 0, bytes));
                    for (int i = 0; i < j; i++) {
                        double t = cs[i] * H[i + (m + 1) * j] + sn[i] * H[(i + 1) + (m + 1) * j];
                        H[(i + 1) + (m + 1) * j] = -sn[i] * H[i + (m + 1) * j] + cs[i] * H[(i + 1) + (m + 1) * j];
                        H[i + (m + 1) * j] = t;
                    }
                    double hjj = H[j + (m + 1) * j], hj1 = H[(j + 1) + (m + 1) * j];
                    double den = std::sqrt(hjj * hjj + hj1 * hj1);
                    if (den < 1e-300)
                        den = 1e-300;
                    cs[j] = hjj / den;
                    sn[j] = hj1 / den;
                    H[j + (m + 1) * j] = cs[j] * hjj + sn[j] * hj1;
                    H[(j + 1) + (m + 1) * j] = 0.0;
                    gg[j + 1] = -sn[j] * gg[j];
                    gg[j] = cs[j] * gg[j];
                    if (std::fabs(gg[j + 1]) <= thresh) {
                        j++;
                        break;
                    }
                }
                const int kk = j;
                std::vector<double> y(kk, 0.0);
                for (int i = kk - 1; i >= 0; i--) {
                    double sm = gg[i];
                    for (int l = i + 1; l < kk; l++)
                        sm -= H[i + (m + 1) * l] * y[l];
                    y[i] = sm / H[i + (m + 1) * i];
                }
                CUDA_CHECK(cudaMemset(tmp, 0, bytes));
                for (int i = 0; i < kk; i++)
                    kAxpy<<<eblocks, TPB>>>(tmp, V[i], y[i], N);
                precond(tmp, Mv);
                kAxpy<<<eblocks, TPB>>>(d_x, Mv, 1.0, N); // x += M^{-1}(V y)
                apply(d_x, Ax);
                kSub<<<eblocks, TPB>>>(d_b, Ax, N, w);
                if (std::sqrt(gpuDot(s, w, w, N)) <= thresh)
                    break;
            }
        } else {
            CUDA_CHECK(cudaMemset(d_x, 0, bytes));
        }
        return total; // basis is persistent (freed in freeState)
    }
} // namespace

namespace {
    inline void uploadResidualState(SoilSolver3DGpuState &s, int N, const std::vector<SoilProperties> &props, const std::vector<double> &theta_n, const std::vector<double> &theta_nm1, const std::vector<double> &src) {
        std::vector<GpuSoil> hp(N);
        for (int c = 0; c < N; c++)
            hp[c] = toGpuSoil(props[c]);
        CUDA_CHECK(cudaMemcpy(s.props, hp.data(), (size_t) N * sizeof(GpuSoil), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.theta_n, theta_n.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.theta_nm1, theta_nm1.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.src, src.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    }
} // namespace

void SoilSolver3D::residualGPU(const std::vector<double> &h, std::vector<double> &F, double a0, double a1, double a2) {
    const int N = (int) numCells();
    const int eblocks = (N + TPB - 1) / TPB;
    if (!gpu_) {
        gpu_ = new SoilSolver3DGpuState();
        try {
            allocState(*gpu_, N);
        } catch (...) {
            releaseGpuState(); // no half-allocated state left behind
            throw;
        }
    }
    SoilSolver3DGpuState &s = *gpu_;
    GpuDims g{(int) Nx_, (int) Ny_, (int) Nz_, dx_, dy_, dz_};
    GpuBC bc;
    for (int f = 0; f < 6; f++) {
        bc.type[f] = (int) bc_[f].type;
        bc.val[f] = bc_[f].value;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_Kmin, &params_.K_min, sizeof(double)));
    {
        int kw = params_.K_weighting;
        CUDA_CHECK(cudaMemcpyToSymbol(d_Kweight, &kw, sizeof(int)));
    } // interblock K weighting
    std::vector<double> net_src(N); // fold sinks in exactly as CPU residual()/buildSystem() do
    for (int c = 0; c < N; c++)
        net_src[c] = src_[c] - uptake_[c] - evap_[c];
    uploadResidualState(s, N, props_, theta_n_, theta_nm1_, net_src);
    CUDA_CHECK(cudaMemcpy(s.h, h.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    kResidual<<<eblocks, TPB>>>(s.h, s.props, s.theta_n, s.theta_nm1, s.src, a0, a1, a2, params_.C_min, g, bc, N, s.F);
    F.resize(N);
    CUDA_CHECK_READBACK(cudaMemcpy(F.data(), s.F, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
}

void SoilSolver3D::jacvecGPU(const std::vector<double> &h, const std::vector<double> &v, std::vector<double> &Jv, double a0, double a1, double a2) {
    const int N = (int) numCells();
    const int eblocks = (N + TPB - 1) / TPB;
    if (!gpu_) {
        gpu_ = new SoilSolver3DGpuState();
        try {
            allocState(*gpu_, N);
        } catch (...) {
            releaseGpuState(); // no half-allocated state left behind
            throw;
        }
    }
    SoilSolver3DGpuState &s = *gpu_;
    GpuDims g{(int) Nx_, (int) Ny_, (int) Nz_, dx_, dy_, dz_};
    GpuBC bc;
    for (int f = 0; f < 6; f++) {
        bc.type[f] = (int) bc_[f].type;
        bc.val[f] = bc_[f].value;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_Kmin, &params_.K_min, sizeof(double)));
    {
        int kw = params_.K_weighting;
        CUDA_CHECK(cudaMemcpyToSymbol(d_Kweight, &kw, sizeof(int)));
    } // interblock K weighting
    std::vector<double> net_src(N); // fold sinks in exactly as CPU residual()/buildSystem() do
    for (int c = 0; c < N; c++)
        net_src[c] = src_[c] - uptake_[c] - evap_[c];
    uploadResidualState(s, N, props_, theta_n_, theta_nm1_, net_src);
    CUDA_CHECK(cudaMemcpy(s.h, h.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.r, v.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice)); // v -> s.r (scratch)
    // FD increment eps = sqrt(eps_mach)*(1+||h||)/||v|| (host-side, from inputs).
    double hnorm2 = 0.0, vnorm2 = 0.0;
    for (int c = 0; c < N; c++) {
        hnorm2 += h[c] * h[c];
        vnorm2 += v[c] * v[c];
    }
    const double vnorm = std::sqrt(vnorm2);
    Jv.assign(N, 0.0);
    if (vnorm < 1e-300)
        return;
    const double eps = 1.49011611938e-8 * (1.0 + std::sqrt(hnorm2)) / vnorm;
    kCopy<<<eblocks, TPB>>>(s.h, s.hp, N); // hp = h
    kAxpy<<<eblocks, TPB>>>(s.hp, s.r, eps, N); // hp += eps*v
    kResidual<<<eblocks, TPB>>>(s.h, s.props, s.theta_n, s.theta_nm1, s.src, a0, a1, a2, params_.C_min, g, bc, N, s.F); // R(h)
    kResidual<<<eblocks, TPB>>>(s.hp, s.props, s.theta_n, s.theta_nm1, s.src, a0, a1, a2, params_.C_min, g, bc, N, s.Fp); // R(h+eps v)
    kSub<<<eblocks, TPB>>>(s.Fp, s.F, N, s.z); // z = Fp - F
    kScale<<<eblocks, TPB>>>(s.z, 1.0 / eps, N, s.u); // u = z / eps
    CUDA_CHECK_READBACK(cudaMemcpy(Jv.data(), s.u, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
}

// Device-resident inexact Newton-Krylov step. Assumes solveGPU() has uploaded props/h/theta_n/
// theta_nm1/src. Mirrors the CPU solveNewton(): matrix-free Jacobian (gpuJacvec) + GMRES
// preconditioned by one multigrid V-cycle, Armijo line search, Eisenstat-Walker forcing. Does its
// own copy-back epilogue (advancing the BDF2 history on convergence). Returns outer iterations.
uint SoilSolver3D::solveNewtonGPU(double dt, double a0, double a1, double a2) {
    const int N = (int) numCells();
    const int eblocks = (N + TPB - 1) / TPB;
    SoilSolver3DGpuState &s = *gpu_;
    GpuDims g{(int) Nx_, (int) Ny_, (int) Nz_, dx_, dy_, dz_};
    GpuBC bc;
    for (int f = 0; f < 6; f++) {
        bc.type[f] = (int) bc_[f].type;
        bc.val[f] = bc_[f].value;
    }
    const double Cmin = params_.C_min;

    gpuResidual(s, s.h, s.F, a0, a1, a2, Cmin, g, bc, N, eblocks); // F = R(h)
    double fnorm = std::sqrt(gpuDot(s, s.F, s.F, N));
    const double ftol = params_.newton_rtol * fnorm + 1e-300;
    double eta = 0.1;
    uint result = 0;
    for (uint it = 0; it < params_.newton_max; it++) {
        if (fnorm <= ftol) {
            result = (it == 0) ? 1u : it;
            break;
        }
        // Freeze the Modified-Picard preconditioner operator at the current iterate.
        kUpdateKC<<<eblocks, TPB>>>(s.h, s.props, Cmin, N, s.K, s.C, s.theta_m);
        kBuildSystem<<<eblocks, TPB>>>(s.h, s.K, s.C, s.theta_m, s.theta_n, s.theta_nm1, s.src, s.props, a0, a1, a2, g, bc, N, s.diag, s.b);
        if (params_.use_multigrid)
            mgBuildGPU(a0);
        auto apply = [&](const double *v, double *Jv) { gpuJacvec(s, s.h, s.F, v, Jv, a0, a1, a2, Cmin, g, bc, N, eblocks); };
        auto precond = [&](const double *r, double *z) {
            if (params_.use_multigrid)
                mgApplyGPU(z, r);
            else
                kJacobi<<<eblocks, TPB>>>(r, s.diag, N, z);
        };
        kScale<<<eblocks, TPB>>>(s.F, -1.0, N, s.r); // rhs = -F
        CUDA_CHECK(cudaMemset(s.u, 0, (size_t) N * sizeof(double))); // delta = 0
        gmresGPU(s, apply, precond, s.r, s.u, params_.gmres_restart, params_.newton_gmres_max, eta, N, eblocks);
        // Armijo backtracking line search on ||R||.
        double lambda = 1.0, ftnorm = fnorm;
        bool ok = false;
        for (int ls = 0; ls < 8; ls++) {
            CUDA_CHECK(cudaMemcpy(s.hp, s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
            kAxpy<<<eblocks, TPB>>>(s.hp, s.u, lambda, N); // hp = h + lambda*delta
            gpuResidual(s, s.hp, s.Fp, a0, a1, a2, Cmin, g, bc, N, eblocks);
            ftnorm = std::sqrt(gpuDot(s, s.Fp, s.Fp, N));
            if (ftnorm < (1.0 - 1e-4 * lambda) * fnorm) {
                ok = true;
                break;
            }
            lambda *= 0.5;
        }
        CUDA_CHECK(cudaMemcpy(s.h, s.hp, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // accept
        CUDA_CHECK(cudaMemcpy(s.F, s.Fp, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        const double fp = fnorm;
        fnorm = ftnorm;
        if (fp > 1e-300) {
            const double e = 0.9 * (fnorm / fp) * (fnorm / fp);
            eta = std::min(0.9, std::max(1e-6, e));
        }
        if (!ok && lambda <= (1.0 / 256.0)) {
            result = 0;
            break;
        } // line search collapsed
    }

    // Epilogue (mirror solveGPU): copy back; on convergence advance the BDF2 history.
    if (result != 0) {
        std::swap(s.theta_n, s.theta_nm1);
        kWaterContent<<<eblocks, TPB>>>(s.h, s.props, N, s.theta_n);
    }
    CUDA_CHECK_READBACK(cudaMemcpy(h_.data(), s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
    if (result != 0) {
        CUDA_CHECK_READBACK(cudaMemcpy(theta_n_.data(), s.theta_n, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_READBACK(cudaMemcpy(theta_nm1_.data(), s.theta_nm1, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        dt_prev_ = dt;
        have_history_ = true;
    }
    return result;
}

// Device-resident adaptive Picard->Newton (mirrors CPU solveNonlinearSwitch). Phase A is a
// Modified-Picard step: solve A u = b with gmresGPU on the SPD kMatvec operator (reusing the
// validated GMRES rather than touching the existing CG path), cheap max|dh| convergence/stall
// check. It hands off to Phase B (Newton-Krylov) once Picard's per-iter reduction ratio stalls.
// Both phases use max|dh| < picard_tol convergence. Assumes solveGPU() uploaded the state.
uint SoilSolver3D::solveNonlinearSwitchGPU(double dt, double a0, double a1, double a2) {
    const int N = (int) numCells();
    const int eblocks = (N + TPB - 1) / TPB;
    SoilSolver3DGpuState &s = *gpu_;
    GpuDims g{(int) Nx_, (int) Ny_, (int) Nz_, dx_, dy_, dz_};
    GpuBC bc;
    for (int f = 0; f < 6; f++) {
        bc.type[f] = (int) bc_[f].type;
        bc.val[f] = bc_[f].value;
    }
    const double Cmin = params_.C_min;

    auto precond = [&](const double *r, double *z) {
        if (params_.use_multigrid)
            mgApplyGPU(z, r);
        else
            kJacobi<<<eblocks, TPB>>>(r, s.diag, N, z);
    };

    bool newton_phase = false;
    double eta = 0.1, dmax_prev = 0.0;
    int slow = 0;
    const double stall_ratio = params_.newton_stall_ratio;
    const int stall_count = std::max(1, params_.newton_stall_count);
    const uint outer_max = params_.picard_max + params_.newton_max;
    uint result = 0;

    for (uint it = 0; it < outer_max; it++) {
        // Freeze the Modified-Picard operator A at the current iterate (both phases need it).
        kUpdateKC<<<eblocks, TPB>>>(s.h, s.props, Cmin, N, s.K, s.C, s.theta_m);
        kBuildSystem<<<eblocks, TPB>>>(s.h, s.K, s.C, s.theta_m, s.theta_n, s.theta_nm1, s.src, s.props, a0, a1, a2, g, bc, N, s.diag, s.b);
        if (params_.use_multigrid)
            mgBuildGPU(a0);

        if (!newton_phase) { // ---- Phase A: one Modified-Picard update (solve A u = b) ----
            auto applyA = [&](const double *x, double *y) { kMatvec<<<eblocks, TPB>>>(x, y, s.K, s.C, s.props, s.hm, a0, g, bc, N); };
            CUDA_CHECK(cudaMemcpy(s.u, s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // warm start u = h
            gpu_picard_outer_++;
            gpu_picard_gmres_ += gmresGPU(s, applyA, precond, s.b, s.u, params_.gmres_restart, (int) params_.cg_max, params_.cg_tol, N, eblocks);
            kPartialMaxAbsDiff<<<RB, TPB>>>(s.u, s.h, N, s.partial);
            double dmax = reduceHost(s, true);
            CUDA_CHECK(cudaMemcpy(s.h, s.u, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // accept
            if (dmax < params_.picard_tol) {
                result = it + 1;
                break;
            }
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
            gpuResidual(s, s.h, s.F, a0, a1, a2, Cmin, g, bc, N, eblocks);
            double fnorm = std::sqrt(gpuDot(s, s.F, s.F, N));
            auto applyJ = [&](const double *v, double *Jv) { gpuJacvec(s, s.h, s.F, v, Jv, a0, a1, a2, Cmin, g, bc, N, eblocks); };
            kScale<<<eblocks, TPB>>>(s.F, -1.0, N, s.r); // rhs = -F
            CUDA_CHECK(cudaMemset(s.u, 0, (size_t) N * sizeof(double)));
            gpu_newton_outer_++;
            gpu_newton_gmres_ += gmresGPU(s, applyJ, precond, s.r, s.u, params_.gmres_restart, params_.newton_gmres_max, eta, N, eblocks);
            double lambda = 1.0, ftnorm = fnorm;
            bool ok = false;
            for (int ls = 0; ls < 8; ls++) {
                CUDA_CHECK(cudaMemcpy(s.hp, s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
                kAxpy<<<eblocks, TPB>>>(s.hp, s.u, lambda, N);
                gpuResidual(s, s.hp, s.Fp, a0, a1, a2, Cmin, g, bc, N, eblocks);
                ftnorm = std::sqrt(gpuDot(s, s.Fp, s.Fp, N));
                gpu_ls_evals_++;
                if (ftnorm < (1.0 - 1e-4 * lambda) * fnorm) {
                    ok = true;
                    break;
                }
                lambda *= 0.5;
            }
            kPartialMaxAbsDiff<<<RB, TPB>>>(s.hp, s.h, N, s.partial); // update norm = max|lambda*delta|
            double dmax = reduceHost(s, true);
            CUDA_CHECK(cudaMemcpy(s.h, s.hp, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice)); // accept
            if (fnorm > 1e-300) {
                const double e = 0.9 * (ftnorm / fnorm) * (ftnorm / fnorm);
                eta = std::min(0.9, std::max(1e-6, e));
            }
            if (ok && dmax < params_.picard_tol) {
                result = it + 1;
                break;
            }
            if (!ok && lambda <= (1.0 / 256.0)) {
                result = 0;
                break;
            }
        }
    }

    if (result != 0) {
        std::swap(s.theta_n, s.theta_nm1);
        kWaterContent<<<eblocks, TPB>>>(s.h, s.props, N, s.theta_n);
    }
    CUDA_CHECK_READBACK(cudaMemcpy(h_.data(), s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
    if (result != 0) {
        CUDA_CHECK_READBACK(cudaMemcpy(theta_n_.data(), s.theta_n, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_READBACK(cudaMemcpy(theta_nm1_.data(), s.theta_nm1, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
        dt_prev_ = dt;
        have_history_ = true;
    }
    return result;
}

// NEWTON_ANALYTIC on the GPU: same algorithm as solveNewtonAnalytic() (assembled analytic Jacobian,
// saturation-capped water-content-space update, two-pass backtracking line search, per-cell residual and
// head-change convergence), with the device-resident state uploaded by solveGPU(). The linear solve is
// BiCGSTAB preconditioned by one geometric-multigrid V-cycle of the Modified-Picard operator frozen at the
// iterate (ILU's sequential triangular solves do not map to the GPU).
uint SoilSolver3D::solveNewtonAnalyticGPU(double dt, double a0, double a1, double a2, uint *linear_iterations) {
    SoilSolver3DGpuState &s = *gpu_;
    const int N = (int) numCells();
    const int eb = (N + TPB - 1) / TPB;
    allocNewtonState(s, N);
    GpuDims g{(int) Nx_, (int) Ny_, (int) Nz_, dx_, dy_, dz_};
    GpuBC bc;
    for (int f = 0; f < 6; f++) {
        bc.type[f] = (int) bc_[f].type;
        bc.val[f] = bc_[f].value;
    }
    const double Cmin = params_.C_min;
    CUDA_CHECK(cudaMemcpy(s.nk_zc, zc_.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.nk_thk, thk_.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.nk_fixed, fixed_h_.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.nk_ss, ss_.data(), (size_t) N * sizeof(double), cudaMemcpyHostToDevice));
    double *h = s.hm; // Newton iterate (starts at h^n)
    auto norm = [&](const double *v) { return std::sqrt(gpuDot(s, v, v, N)); };
    // Nonlinear residual norm, volume-weighted as on the CPU (plain L2 on a uniform grid)
    double tbar = 0.0;
    for (int c = 0; c < N; c++)
        tbar += thk_[c];
    tbar /= N;
    auto fnormOf = [&](const double *F) {
        if (uniform_geometry_)
            return norm(F);
        kNkVolumeWeight<<<eb, TPB>>>(F, s.nk_thk, 1.0 / tbar, N, s.nk_tmp2);
        return norm(s.nk_tmp2);
    };
    auto devMax = [&](const double *v) {
        kPartialMax<<<RB, TPB>>>(v, N, s.partial);
        return reduceHost(s, true);
    };
    auto assemble = [&](const double *hh, double *F, bool withJ) {
        kNkConstitutive<<<eb, TPB>>>(hh, s.props, s.nk_ss, Cmin, N, s.nk_theta, s.nk_K, s.nk_C, s.nk_dK);
        kNkAssemble<<<eb, TPB>>>(hh, s.props, s.nk_theta, s.nk_K, s.nk_C, s.nk_dK, s.theta_n, s.theta_nm1, s.src, s.nk_zc, s.nk_thk, s.nk_fixed, a0, a1, a2, g, bc, withJ ? 1 : 0, N, F, s.nk_scale, s.nk_J);
    };
    // z = M^{-1} r. Flat cells (layers thinner than half the lateral spacing) or per-cell geometry, where
    // vertical coupling dominates and the uniform-grid multigrid hierarchy does not represent the operator:
    // vertical-line block Jacobi on the Jacobian itself. Otherwise one multigrid V-cycle of the Picard
    // operator frozen at the iterate (built in the loop).
    const bool line_precond = !uniform_geometry_ || dz_ <= 0.5 * std::min(dx_, dy_);
    const int cb = (int) ((Nx_ * Ny_ + TPB - 1) / TPB);
    auto precond = [&](const double *r, double *z) {
        if (line_precond)
            kNkLinePrecond<<<cb, TPB>>>(s.nk_J, r, g, N, s.nk_tmp2, z);
        else
            mgApplyGPU(z, r);
    };
    auto bicgstab = [&](const double *b, double *x, double rtol, uint maxit) -> uint {
        CUDA_CHECK(cudaMemsetAsync(x, 0, (size_t) N * sizeof(double), 0));
        CUDA_CHECK(cudaMemcpy(s.bi_r, b, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s.bi_rhat, b, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemsetAsync(s.bi_p, 0, (size_t) N * sizeof(double), 0));
        CUDA_CHECK(cudaMemsetAsync(s.bi_v, 0, (size_t) N * sizeof(double), 0));
        const double bnorm = norm(b);
        if (bnorm < 1e-300)
            return 0;
        const double thresh = rtol * bnorm;
        double rho = 1.0, alpha = 1.0, omega = 1.0;
        uint it = 0;
        while (it < maxit) {
            it++;
            const double rho_new = gpuDot(s, s.bi_rhat, s.bi_r, N);
            if (std::fabs(rho_new) < 1e-300)
                break;
            const double beta = (rho_new / rho) * (alpha / omega);
            kAxpy<<<eb, TPB>>>(s.bi_p, s.bi_v, -omega, N); // p -= omega v
            kPaxpy<<<eb, TPB>>>(s.bi_p, s.bi_r, beta, N); // p = r + beta p
            precond(s.bi_p, s.bi_phat);
            kNkJacApply<<<eb, TPB>>>(s.nk_J, s.bi_phat, g, N, s.bi_v);
            alpha = rho_new / gpuDot(s, s.bi_rhat, s.bi_v, N);
            CUDA_CHECK(cudaMemcpy(s.bi_s, s.bi_r, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
            kAxpy<<<eb, TPB>>>(s.bi_s, s.bi_v, -alpha, N); // s = r - alpha v
            if (norm(s.bi_s) <= thresh) {
                kAxpy<<<eb, TPB>>>(x, s.bi_phat, alpha, N);
                break;
            }
            precond(s.bi_s, s.bi_shat);
            kNkJacApply<<<eb, TPB>>>(s.nk_J, s.bi_shat, g, N, s.bi_t);
            const double tt = gpuDot(s, s.bi_t, s.bi_t, N);
            omega = (tt > 1e-300) ? gpuDot(s, s.bi_t, s.bi_s, N) / tt : 0.0;
            kAxpy<<<eb, TPB>>>(x, s.bi_phat, alpha, N);
            kAxpy<<<eb, TPB>>>(x, s.bi_shat, omega, N);
            CUDA_CHECK(cudaMemcpy(s.bi_r, s.bi_s, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
            kAxpy<<<eb, TPB>>>(s.bi_r, s.bi_t, -omega, N); // r = s - omega t
            if (norm(s.bi_r) <= thresh || omega == 0.0)
                break;
            rho = rho_new;
        }
        return it;
    };

    newton_resid_history_.clear();
    uint total_lin = 0;
    double eta = 1e-2;
    bool dh_ok = false;
    double max_dh = INFINITY;
    int creep = 0; // consecutive damped iterations with little progress (see kNewtonCreepIterations)
    assemble(h, s.nk_F, true);
    double fnorm = fnormOf(s.nk_F);
    for (uint it = 0; it < params_.newton_max; it++) {
        newton_resid_history_.push_back(fnorm);
        if (!std::isfinite(fnorm))
            break;
        if (it >= 6 && fnorm > 0.99 * newton_resid_history_[it - 6])
            break; // stalled
        kNkResRatio<<<eb, TPB>>>(s.nk_F, s.nk_scale, params_.newton_theta_tol * a0, kNewtonRelTol, N, s.nk_tmp);
        const bool res_ok = devMax(s.nk_tmp) <= 1.0;
        bool strict_ok = false; // before the first iteration only the strict relative test may accept
        if (it == 0 && res_ok) {
            kNkResRatio<<<eb, TPB>>>(s.nk_F, s.nk_scale, 0.0, kNewtonRelTol, N, s.nk_tmp);
            strict_ok = devMax(s.nk_tmp) <= 1.0;
        }
        if ((res_ok && (it == 0 ? strict_ok : dh_ok)) || (it > 0 && max_dh < params_.newton_dvclose)) {
            CUDA_CHECK(cudaMemcpy(s.h, h, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
            std::swap(s.theta_n, s.theta_nm1);
            kStoredWater<<<eb, TPB>>>(s.h, s.props, s.nk_ss, N, s.theta_n);
            CUDA_CHECK_READBACK(cudaMemcpy(h_.data(), s.h, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK_READBACK(cudaMemcpy(theta_n_.data(), s.theta_n, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK_READBACK(cudaMemcpy(theta_nm1_.data(), s.theta_nm1, (size_t) N * sizeof(double), cudaMemcpyDeviceToHost));
            dt_prev_ = dt;
            have_history_ = true;
            if (linear_iterations)
                *linear_iterations = total_lin;
            return std::max(1u, it);
        }

        if (!line_precond) { // freeze the Picard operator at the iterate for the multigrid preconditioner
            kUpdateKC<<<eb, TPB>>>(h, s.props, Cmin, N, s.K, s.C, s.theta_m);
            kBuildSystem<<<eb, TPB>>>(h, s.K, s.C, s.theta_m, s.theta_n, s.theta_nm1, s.src, s.props, a0, a1, a2, g, bc, N, s.diag, s.b);
            mgBuildGPU(a0);
        }
        kScale<<<eb, TPB>>>(s.nk_F, -1.0, N, s.nk_tmp); // rhs = -F
        total_lin += bicgstab(s.nk_tmp, s.nk_delta, eta, params_.bicgstab_max);

        CUDA_CHECK(cudaMemcpy(s.nk_thetah, s.nk_theta, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s.nk_Ch, s.nk_C, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        const bool full_step = fnorm < params_.newton_backtrack_limit;
        double best = INFINITY;
        bool decreased = false, jac_current = false;
        for (int pass = 0; pass < 2 && !decreased; pass++) {
            double lambda = 1.0;
            for (int ls = 0; ls < 8; ls++) {
                kNkTrial<<<eb, TPB>>>(h, s.nk_delta, s.nk_thetah, s.nk_Ch, s.props, s.nk_fixed, s.nk_ss, lambda, pass == 0 ? 1 : 0, N, s.nk_htrial);
                const bool first = pass == 0 && ls == 0;
                assemble(s.nk_htrial, s.nk_Ft, first);
                const double ftn = fnormOf(s.nk_Ft);
                if (ftn < best) {
                    best = ftn;
                    std::swap(s.nk_htrial, s.nk_hbest);
                    std::swap(s.nk_Ft, s.nk_Fbest);
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
        kNkDhRatio<<<eb, TPB>>>(s.nk_hbest, h, s.nk_Ch, s.props, params_.picard_tol, params_.newton_theta_tol, N, s.nk_tmp);
        dh_ok = devMax(s.nk_tmp) <= 1.0;
        kPartialMaxAbsDiff<<<RB, TPB>>>(s.nk_hbest, h, N, s.partial);
        max_dh = reduceHost(s, true);
        CUDA_CHECK(cudaMemcpy(h, s.nk_hbest, (size_t) N * sizeof(double), cudaMemcpyDeviceToDevice));
        const double fnorm_prev = fnorm;
        if (jac_current) {
            std::swap(s.nk_F, s.nk_Fbest);
            fnorm = best;
        } else {
            assemble(h, s.nk_F, true);
            fnorm = fnormOf(s.nk_F);
        }
        if (fnorm_prev > 1e-300) {
            const double e = 0.9 * (fnorm / fnorm_prev) * (fnorm / fnorm_prev);
            eta = std::min(1e-2, std::max(1e-10, e));
        }
        creep = (!jac_current && fnorm > kNewtonCreepRatio * fnorm_prev) ? creep + 1 : 0; // as on the CPU
        if (creep >= kNewtonCreepIterations)
            break;
    }
    if (linear_iterations)
        *linear_iterations = total_lin;
    return 0; // not converged; s.h and the host state are unchanged
}


#endif // HELIOS_CUDA_AVAILABLE
