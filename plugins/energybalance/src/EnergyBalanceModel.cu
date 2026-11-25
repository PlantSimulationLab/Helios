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
#include "EnergyBalanceSolver.h"

using namespace std;

#define CUDA_CHECK_ERROR(ans)                                                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                                                                                                                                                                                            \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        helios::helios_runtime_error("ERROR (EnergyBalanceModel::solveSurfaceTemperatures_GPU): CUDA call failed: " + std::string(cudaGetErrorString(code)) + " (" + std::string(file) + ":" + std::to_string(line) +
                                     "). Disable GPU acceleration with EnergyBalanceModel::disableGPUAcceleration() to solve on the CPU.");
    }
}

//! Solve the energy balance of each primitive (one thread per primitive) with the same per-surface solve as the CPU
__global__ void solveEnergyBalance(uint Nprimitives, float *To, float *R, float *Qother, float *eps, float *Ta, float *ea, float *pressure, float *gH, float *gbv_front, float *gbv_back, float *gS, uint *Nsides, float *stomatal_sidedness, float *TL,
                                   float *heatcapacity, float *surfacehumidity, float dt, uint8_t *converged) {

    uint p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p >= Nprimitives) {
        return;
    }

    const SurfaceEnergyBalanceTerms terms{R[p], Qother[p], eps[p], Ta[p], ea[p], pressure[p], gH[p], gbv_front[p], gbv_back[p], gS[p], Nsides[p], stomatal_sidedness[p], heatcapacity[p], surfacehumidity[p], dt, To[p]};
    float T;
    converged[p] = solveSurfaceEnergyBalance(terms, To[p], T) ? 1 : 0;
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
    float *d_gbv_front = copyToDevice(inputs.gbv_front);
    float *d_gbv_back = copyToDevice(inputs.gbv_back);
    float *d_gS = copyToDevice(inputs.gS);
    uint *d_Nsides = copyToDevice(inputs.Nsides);
    float *d_stomatal_sidedness = copyToDevice(inputs.stomatal_sidedness);
    float *d_heatcapacity = copyToDevice(inputs.heatcapacity);
    float *d_surfacehumidity = copyToDevice(inputs.surfacehumidity);

    float *d_T;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_T, float_bytes));
    uint8_t *d_converged;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_converged, Nprimitives * sizeof(uint8_t)));

    // launch kernel
    dim3 dimBlock(64, 1);
    dim3 dimGrid(ceil(Nprimitives / 64.f));

    solveEnergyBalance<<<dimGrid, dimBlock>>>(Nprimitives, d_To, d_R, d_Qother, d_eps, d_Ta, d_ea, d_pressure, d_gH, d_gbv_front, d_gbv_back, d_gS, d_Nsides, d_stomatal_sidedness, d_T, d_heatcapacity, d_surfacehumidity, dt, d_converged);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    T.resize(Nprimitives);
    CUDA_CHECK_ERROR(cudaMemcpy(T.data(), d_T, float_bytes, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> converged(Nprimitives);
    CUDA_CHECK_ERROR(cudaMemcpy(converged.data(), d_converged, Nprimitives * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_To));
    CUDA_CHECK_ERROR(cudaFree(d_R));
    CUDA_CHECK_ERROR(cudaFree(d_Qother));
    CUDA_CHECK_ERROR(cudaFree(d_eps));
    CUDA_CHECK_ERROR(cudaFree(d_Ta));
    CUDA_CHECK_ERROR(cudaFree(d_ea));
    CUDA_CHECK_ERROR(cudaFree(d_pressure));
    CUDA_CHECK_ERROR(cudaFree(d_gH));
    CUDA_CHECK_ERROR(cudaFree(d_gbv_front));
    CUDA_CHECK_ERROR(cudaFree(d_gbv_back));
    CUDA_CHECK_ERROR(cudaFree(d_gS));
    CUDA_CHECK_ERROR(cudaFree(d_Nsides));
    CUDA_CHECK_ERROR(cudaFree(d_stomatal_sidedness));
    CUDA_CHECK_ERROR(cudaFree(d_heatcapacity));
    CUDA_CHECK_ERROR(cudaFree(d_surfacehumidity));
    CUDA_CHECK_ERROR(cudaFree(d_T));
    CUDA_CHECK_ERROR(cudaFree(d_converged));

    const size_t convergence_failures = Nprimitives - std::count(converged.begin(), converged.end(), uint8_t(1));
    if (convergence_failures > 0 && message_flag) {
        std::cout << "WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge for " << convergence_failures << " primitives." << std::endl;
    }
}

#endif // HELIOS_CUDA_AVAILABLE
