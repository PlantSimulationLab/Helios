/** \file "EnergyBalanceModel.cu" Energy balance model plugin declarations (CUDA kernels).

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#ifdef HELIOS_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include "EnergyBalanceModel.h"

using namespace std;

#define CUDA_CHECK_ERROR(ans)                                                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                                                                                                                                                                                            \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ float evaluateEnergyBalance(float T, float R, float Qother, float eps, float Ta, float ea, float pressure, float gH, float gS, uint Nsides, float stomatal_sidedness, float heatcapacity, float surfacehumidity, float dt, float Tprev) {

    // Outgoing emission flux
    float Rout = float(Nsides) * eps * 5.67e-8F * T * T * T * T;

    // Sensible heat flux
    float QH = cp_air_mol * gH * (T - Ta); // (see Campbell and Norman Eq. 6.8)

    // Latent heat flux
    // Same expression as esat_Pa(), which is host-only. The latent flux reported after the solve uses esat_Pa(), so the two must agree for the reported fluxes to close the energy balance.
    float es = 611.0f * expf(17.502f * (T - 273.15f) / (T - 273.15f + 240.97f));
    // A zero boundary-layer conductance (gH) or zero stomatal conductance (gS) means there is no vapor
    // pathway, so gM is zero. Guarding both cases also avoids a 0/0 in the series-conductance expression
    // when gH == 0 and the stomatal sidedness makes one of the denominators vanish.
    float gM = 0.f;
    if (gH != 0.f && gS != 0.f) {
        // gH is summed over all faces, but vapor leaving a face only crosses that face's boundary layer
        float gbv = 1.08f * gH / float(Nsides);
        gM = gbv * gS * (stomatal_sidedness / (gbv + gS * stomatal_sidedness) + (1.f - stomatal_sidedness) / (gbv + gS * (1.f - stomatal_sidedness)));
    }

    float QL = gM * lambda_mol * (es * surfacehumidity - ea) / pressure;

    // Storage heat flux
    float storage = 0.f;
    if (dt > 0) {
        storage = heatcapacity * (T - Tprev) / dt;
    }

    // Residual
    return R - Rout - QH - QL - Qother - storage;
}

__global__ void solveEnergyBalance(uint Nprimitives, float *To, float *R, float *Qother, float *eps, float *Ta, float *ea, float *pressure, float *gH, float *gS, uint *Nsides, float *stomatal_sidedness, float *TL, float *heatcapacity,
                                   float *surfacehumidity, float dt) {

    uint p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= Nprimitives) {
        return;
    }

    float err_max = 0.0001;
    uint max_iter = 100;

    // Initial guesses. The second guess must differ from the first, otherwise the two residuals are
    // identical and the secant update below divides by zero (this happens when the supplied initial
    // temperature equals the hardcoded second guess of 400 K).
    float T_old = To[p];
    float T_old_old = 400.f;
    if (T_old_old == T_old) {
        T_old_old = T_old + 100.f;
    }

    // Initialize the result to the current best guess so that a degenerate early exit (identical
    // residuals) never leaves the surface temperature uninitialized.
    float T = T_old;

    float resid_old = evaluateEnergyBalance(T_old, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p]);
    float resid_old_old = evaluateEnergyBalance(T_old_old, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p]);

    float resid = 100;
    float err = resid;
    uint iter = 0;
    while (err > err_max && iter < max_iter) {

        if (resid_old == resid_old_old) { // secant denominator would be zero
            err = 0;
            break;
        }

        T = fabs((T_old_old * resid_old - T_old * resid_old_old) / (resid_old - resid_old_old));

        resid = evaluateEnergyBalance(T, R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p]);

        // Relative change of the newest secant step (measures the latest iterate, not the previous one)
        err = fabs(T - T_old) / fabs(T_old);

        resid_old_old = resid_old;
        resid_old = resid;

        T_old_old = T_old;
        T_old = T;

        iter++;
    }

    if (err > err_max) {
        printf("WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge.\n");
    }

    TL[p] = T;
}

void EnergyBalanceModel::solveSurfaceTemperatures_GPU(const SurfaceEnergyBalanceInputs &inputs, float dt, std::vector<float> &T) {

    const uint Nprimitives = inputs.To.size();
    const size_t float_bytes = Nprimitives * sizeof(float);

    // Copy one input array to a newly allocated device array
    auto copyToDevice = [](const auto &host_vector) {
        using value_type = typename std::decay_t<decltype(host_vector)>::value_type;
        value_type *device_array;
        CUDA_CHECK_ERROR(cudaMalloc((void **) &device_array, host_vector.size() * sizeof(value_type)));
        CUDA_CHECK_ERROR(cudaMemcpy(device_array, host_vector.data(), host_vector.size() * sizeof(value_type), cudaMemcpyHostToDevice));
        return device_array;
    };

    float *d_To = copyToDevice(inputs.To);
    float *d_R = copyToDevice(inputs.R);
    float *d_Qother = copyToDevice(inputs.Qother);
    float *d_eps = copyToDevice(inputs.eps);
    float *d_Ta = copyToDevice(inputs.Ta);
    float *d_ea = copyToDevice(inputs.ea);
    float *d_pressure = copyToDevice(inputs.pressure);
    float *d_gH = copyToDevice(inputs.gH);
    float *d_gS = copyToDevice(inputs.gS);
    uint *d_Nsides = copyToDevice(inputs.Nsides);
    float *d_stomatal_sidedness = copyToDevice(inputs.stomatal_sidedness);
    float *d_heatcapacity = copyToDevice(inputs.heatcapacity);
    float *d_surfacehumidity = copyToDevice(inputs.surfacehumidity);

    float *d_T;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_T, float_bytes));

    // launch kernel
    dim3 dimBlock(64, 1);
    dim3 dimGrid(ceil(Nprimitives / 64.f));

    solveEnergyBalance<<<dimGrid, dimBlock>>>(Nprimitives, d_To, d_R, d_Qother, d_eps, d_Ta, d_ea, d_pressure, d_gH, d_gS, d_Nsides, d_stomatal_sidedness, d_T, d_heatcapacity, d_surfacehumidity, dt);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    T.resize(Nprimitives);
    CUDA_CHECK_ERROR(cudaMemcpy(T.data(), d_T, float_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_To));
    CUDA_CHECK_ERROR(cudaFree(d_R));
    CUDA_CHECK_ERROR(cudaFree(d_Qother));
    CUDA_CHECK_ERROR(cudaFree(d_eps));
    CUDA_CHECK_ERROR(cudaFree(d_Ta));
    CUDA_CHECK_ERROR(cudaFree(d_ea));
    CUDA_CHECK_ERROR(cudaFree(d_pressure));
    CUDA_CHECK_ERROR(cudaFree(d_gH));
    CUDA_CHECK_ERROR(cudaFree(d_gS));
    CUDA_CHECK_ERROR(cudaFree(d_Nsides));
    CUDA_CHECK_ERROR(cudaFree(d_stomatal_sidedness));
    CUDA_CHECK_ERROR(cudaFree(d_heatcapacity));
    CUDA_CHECK_ERROR(cudaFree(d_surfacehumidity));
    CUDA_CHECK_ERROR(cudaFree(d_T));
}

#endif // HELIOS_CUDA_AVAILABLE
