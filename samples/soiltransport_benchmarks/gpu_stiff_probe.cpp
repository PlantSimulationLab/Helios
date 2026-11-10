// Pinpoint the pistachio drydown cost: a single big dt=3600 solve into drying soil (evap + root
// uptake sinks) at 96^3 on GPU, instrumented. Then the same hour at dt_max=600 (smaller steps).
// Tells us whether big steps escalate to Newton and whether capping dt_max is the fix.
#include "SoilTransportModel.h"
#include "SoilSolver3D.h"
#include <cstdio>
#include <chrono>
using namespace helios;

static void scenario(const char* tag, double dt_max) {
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(96, 96, 96, 3.0 / 96, 3.0 / 96, 1.5 / 96, g);
    SoilSolver3D::Params p = s.getParams();
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::PICARD_NEWTON;
    p.use_gpu = true;
    p.use_multigrid = true;
    p.K_weighting = 3;
    p.dt_max = dt_max;
    s.setParams(p);
    s.setInitialCondition(-1.0);
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 0.0);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    // Surface evaporation (top layer) + a root-uptake zone -- the drying sinks active during drydown.
    for (uint i = 0; i < 96; i++)
        for (uint j = 0; j < 96; j++) s.setEvaporation(i, j, 95u, 1e-6);
    for (uint k = 70; k < 92; k++)
        for (uint i = 40; i < 56; i++)
            for (uint j = 40; j < 56; j++) s.setUptake(i, j, k, 1e-6);

    s.resetGpuStats();
    double dt = std::min(60.0, dt_max);
    uint substeps = 0;
    auto t0 = std::chrono::steady_clock::now();
    s.solveAdaptive(3600.0, dt, &substeps);  // one drydown hour
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto avg = [](uint64_t a, uint64_t b) { return b ? (double)a / b : 0.0; };
    printf("%-16s wall=%6.2f s  substeps=%4u  dt_end=%7.1f  | PICARD o=%llu g=%llu(%.1f/s)  NEWTON o=%llu g=%llu(%.1f/s)  ls=%llu\n",
           tag, secs, substeps, dt,
           (unsigned long long)s.gpuPicardOuter(), (unsigned long long)s.gpuPicardGmres(), avg(s.gpuPicardGmres(), s.gpuPicardOuter()),
           (unsigned long long)s.gpuNewtonOuter(), (unsigned long long)s.gpuNewtonGmres(), avg(s.gpuNewtonGmres(), s.gpuNewtonOuter()),
           (unsigned long long)s.gpuLineSearchEvals());
}

int main() {
    printf("=== 96^3 drydown hour (evap + root uptake), PICARD_NEWTON GPU ===\n");
    scenario("dt_max=3600", 3600.0);
    scenario("dt_max=600", 600.0);
    scenario("dt_max=120", 120.0);
    return 0;
}
