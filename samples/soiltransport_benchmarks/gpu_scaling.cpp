// GPU scaling test: one-hour drip-irrigation soil simulation (the pistachio per-hour workload --
// spread emitters into dry loam, adaptive PICARD_NEWTON on the GPU) at increasing resolution n^3.
// Prints "n N wall_s" rows for a LaTeX table. A small warmup run absorbs one-time CUDA init cost.
#include "SoilTransportModel.h"
#include <cstdio>
#include <chrono>
using namespace helios;

static double run_one(int n) {
    Context ctx;
    SoilTransportModel soil(&ctx);
    soil.disableMessages();
    soil.defineSoilDomain(make_vec3(-1.5f, -1.5f, 0.f), make_vec3(3.f, 3.f, 1.5f));
    soil.setGridResolution((uint)n, (uint)n, (uint)n);
    soil.setSolverType(SolverType::SOLVER_3D);
    soil.setSoilPropertiesFromTexture("loam");
    soil.setInitialCondition(-1.0f);
    soil.setTopBoundaryFlux(0.f);
    soil.setBottomBoundaryFreeDrainage();
    for (int e = 0; e < 10; e++) soil.addSurfaceEmitter(0.5f, -1.1f + 0.25f * e, 2.1e-7f);  // spread footprint
    SolverParameters sp;
    sp.adaptive_dt = true;
    sp.dt_max = 3600.f;
    sp.K_weighting = KWeighting::HARMONIC;
    sp.nonlinear_method = NonlinearMethod::PICARD_NEWTON;
    soil.setSolverParameters(sp);
    soil.enableGPU();
    auto t0 = std::chrono::steady_clock::now();
    soil.run(3600.f);  // one simulated hour
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    const int ns[] = {32, 48, 64, 96, 128, 160, 192, 256};
    run_one(24);  // warmup (CUDA context + allocator), not timed
    printf("# n  N=n^3  wall_s\n");
    for (int n : ns) {
        double w = run_one(n);
        printf("RESULT %d %lld %.3f\n", n, (long long)n * n * n, w);
        fflush(stdout);
    }
    return 0;
}
