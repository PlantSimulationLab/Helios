// Quick benchmark: does Anderson acceleration shorten the Picard tail on a
// stiff dry-soil infiltration step? Compares wall time + adaptive substeps for
// anderson_depth = 0 (plain Picard), 2, 3 under both harmonic (stiff) and
// upstream interblock K weighting.
#include "SoilTransportModel.h"
#include <chrono>
#include <iostream>

using namespace helios;

static void run_case(int anderson, KWeighting kw, const char* kw_name, float h_init) {
    Context context;
    SoilTransportModel soil(&context);
    soil.disableMessages();
    soil.defineSoilDomain(make_vec3(0.f, 0.f, 0.f), make_vec3(1.f, 1.f, 1.f));
    soil.setGridResolution(30, 30, 30);
    soil.setSolverType(SolverType::SOLVER_3D);
    soil.setSoilPropertiesFromTexture("loam");
    soil.setInitialCondition(h_init);
    soil.setTopBoundaryPressure(-0.05f);        // near-saturated ponding -> sharp stiff front
    soil.setBottomBoundaryFreeDrainage();

    SolverParameters sp;
    sp.adaptive_dt = true;
    sp.dt_max = 3600.f;
    sp.K_weighting = kw;
    sp.anderson_depth = anderson;
    soil.setSolverParameters(sp);

    soil.resetSolverIterationCount();
    auto t0 = std::chrono::steady_clock::now();
    long total_substeps = 0;
    for (int h = 0; h < 12; h++) {
        soil.run(3600.f);
        total_substeps += soil.getLastSubstepCount();
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << "  K=" << kw_name << "  anderson=" << anderson
              << "  ->  " << secs << " s, " << total_substeps << " substeps, "
              << soil.getSolverIterationCount() << " Picard its, storage="
              << soil.getTotalWaterStorage() << " m^3\n";
}

static void sweep(const char* title, float h_init) {
    std::cout << title << "\n";
    std::cout << "-- harmonic K (stiffest) --\n";
    run_case(0, KWeighting::HARMONIC, "harmonic", h_init);
    run_case(2, KWeighting::HARMONIC, "harmonic", h_init);
    run_case(3, KWeighting::HARMONIC, "harmonic", h_init);
    std::cout << "-- upstream K --\n";
    run_case(0, KWeighting::UPSTREAM, "upstream", h_init);
    run_case(2, KWeighting::UPSTREAM, "upstream", h_init);
    run_case(3, KWeighting::UPSTREAM, "upstream", h_init);
}

int main() {
    std::cout << "=== Anderson acceleration: dry-soil infiltration (12 h, 30^3 loam) ===\n";
    sweep("### moderately dry: h_init = -5 m ###", -5.0f);
    sweep("### bone dry (near residual): h_init = -100 m ###", -100.0f);
    return 0;
}
