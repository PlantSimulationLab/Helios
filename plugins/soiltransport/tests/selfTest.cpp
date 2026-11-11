/** \file "selfTest.cpp" Unit tests for SoilTransportModel plugin.

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
#include "SoilUtils.h"
#include "SoilColumn.h"
#include "SoilGrid.h"
#include "SoilSolver3D.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "doctest_utils.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#include <type_traits>

using namespace helios;

namespace {
    //! Whether a test's GPU variant can run. Without a usable CUDA device it records a skip, or, on a CUDA build with
    //! HELIOS_REQUIRE_GPU set, fails the test instead of skipping it.
    bool gpuTestsCanRun(const std::string &test_description) {
        if (SoilTransportModel::isGPUAvailable()) {
            return true;
        }
#ifdef HELIOS_CUDA_AVAILABLE
        helios::requireGPUOrFail(test_description + " needs a usable CUDA device.");
#endif
        DOCTEST_MESSAGE("SKIPPED (no usable CUDA device): " << test_description);
        return false;
    }
} // namespace

// ============================================================================
// SoilUtils tests
// ============================================================================

DOCTEST_TEST_CASE("SoilUtils - van Genuchten functions") {

    DOCTEST_SUBCASE("computeSe - saturation at h=0") {
        SoilProperties props; // Default sand-like
        float Se = computeSe(0.0f, props);
        DOCTEST_CHECK(Se == doctest::Approx(1.0f));
    }

    DOCTEST_SUBCASE("computeSe - unsaturated") {
        SoilProperties props(0.045f, 0.43f, 14.5f, 2.68f, 8.25e-5f); // Sand
        float Se = computeSe(-1.0f, props);
        DOCTEST_CHECK(Se > 0.0f);
        DOCTEST_CHECK(Se < 1.0f);
    }

    DOCTEST_SUBCASE("computeSe - very dry") {
        SoilProperties props;
        float Se = computeSe(-100.0f, props);
        DOCTEST_CHECK(Se > 0.0f);
        DOCTEST_CHECK(Se < 0.01f); // Should be very low
    }

    DOCTEST_SUBCASE("h_to_theta - saturated") {
        SoilProperties props(0.045f, 0.43f, 14.5f, 2.68f, 8.25e-5f);
        float theta = h_to_theta(0.0f, props);
        DOCTEST_CHECK(theta == doctest::Approx(props.theta_s));
    }

    DOCTEST_SUBCASE("h_to_theta - unsaturated") {
        SoilProperties props(0.045f, 0.43f, 14.5f, 2.68f, 8.25e-5f);
        float theta = h_to_theta(-1.0f, props);
        DOCTEST_CHECK(theta > props.theta_r);
        DOCTEST_CHECK(theta < props.theta_s);
    }

    DOCTEST_SUBCASE("theta_to_h - round trip") {
        SoilProperties props(0.078f, 0.43f, 3.6f, 1.56f, 2.89e-6f); // Loam
        float h_orig = -2.0f;
        float theta = h_to_theta(h_orig, props);
        float h_back = theta_to_h(theta, props);
        DOCTEST_CHECK(h_back == doctest::Approx(h_orig).epsilon(0.01));
    }

    DOCTEST_SUBCASE("computeK - saturated") {
        SoilProperties props(0.045f, 0.43f, 14.5f, 2.68f, 8.25e-5f);
        float K = computeK(0.0f, props);
        DOCTEST_CHECK(K == doctest::Approx(props.K_s));
    }

    DOCTEST_SUBCASE("computeK - unsaturated less than K_s") {
        SoilProperties props(0.045f, 0.43f, 14.5f, 2.68f, 8.25e-5f);
        float K = computeK(-1.0f, props);
        DOCTEST_CHECK(K > 0.0f);
        DOCTEST_CHECK(K < props.K_s);
    }

    DOCTEST_SUBCASE("computeC - regularization at saturation") {
        SoilProperties props;
        float C = computeC(0.0f, props);
        DOCTEST_CHECK(C > 0.0f); // Should be small but positive (regularization)
        DOCTEST_CHECK(C < 1e-5f);
    }

    DOCTEST_SUBCASE("computeC - unsaturated positive") {
        SoilProperties props(0.078f, 0.43f, 3.6f, 1.56f, 2.89e-6f);
        float C = computeC(-1.0f, props);
        DOCTEST_CHECK(C > 0.0f);
    }
}

DOCTEST_TEST_CASE("SoilUtils - Thomas algorithm") {

    DOCTEST_SUBCASE("Simple 3x3 system") {
        // Solve: [2 -1 0; -1 2 -1; 0 -1 2] * x = [1; 0; 1]
        // Solution: x = [1; 1; 1]
        float a[] = {0, -1, -1};
        float b[] = {2, 2, 2};
        float c[] = {-1, -1, 0};
        float d[] = {1, 0, 1};
        float x[3];

        thomasSolve(a, b, c, d, x, 3);

        DOCTEST_CHECK(x[0] == doctest::Approx(1.0f).epsilon(0.001));
        DOCTEST_CHECK(x[1] == doctest::Approx(1.0f).epsilon(0.001));
        DOCTEST_CHECK(x[2] == doctest::Approx(1.0f).epsilon(0.001));
    }

    DOCTEST_SUBCASE("Single element") {
        float a[] = {0};
        float b[] = {2};
        float c[] = {0};
        float d[] = {4};
        float x[1];

        thomasSolve(a, b, c, d, x, 1);

        DOCTEST_CHECK(x[0] == doctest::Approx(2.0f));
    }

    DOCTEST_SUBCASE("thomasSolveSafe doesn't modify inputs") {
        float a[] = {0, -1, -1};
        float b[] = {2, 2, 2};
        float c[] = {-1, -1, 0};
        float d[] = {1, 0, 1};
        float c_orig[] = {-1, -1, 0};
        float d_orig[] = {1, 0, 1};
        float x[3];

        thomasSolveSafe(a, b, c, d, x, 3);

        // Check inputs not modified
        for (int i = 0; i < 3; i++) {
            DOCTEST_CHECK(c[i] == c_orig[i]);
            DOCTEST_CHECK(d[i] == d_orig[i]);
        }
    }
}

DOCTEST_TEST_CASE("SoilUtils - Soil library") {

    DOCTEST_SUBCASE("Known soil types available") {
        DOCTEST_CHECK(isSoilTextureAvailable("sand"));
        DOCTEST_CHECK(isSoilTextureAvailable("loam"));
        DOCTEST_CHECK(isSoilTextureAvailable("clay"));
        DOCTEST_CHECK(isSoilTextureAvailable("silt_loam"));
    }

    DOCTEST_SUBCASE("Case insensitive lookup") {
        DOCTEST_CHECK(isSoilTextureAvailable("SAND"));
        DOCTEST_CHECK(isSoilTextureAvailable("Sand"));
        DOCTEST_CHECK(isSoilTextureAvailable("LOAM"));
    }

    DOCTEST_SUBCASE("Space to underscore") {
        DOCTEST_CHECK(isSoilTextureAvailable("sandy loam"));
        DOCTEST_CHECK(isSoilTextureAvailable("clay loam"));
    }

    DOCTEST_SUBCASE("Unknown soil raises an error") {
        DOCTEST_CHECK(!isSoilTextureAvailable("unknown_type"));
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(getSoilPropertiesFromTexture("unknown_type")), doctest::Contains("Unknown soil texture"), std::runtime_error);
    }

    DOCTEST_SUBCASE("Sand properties reasonable") {
        SoilProperties sand = getSoilPropertiesFromTexture("sand");
        DOCTEST_CHECK(sand.theta_r < 0.1f);
        DOCTEST_CHECK(sand.theta_s > 0.3f);
        DOCTEST_CHECK(sand.K_s > 1e-5f); // High conductivity
        DOCTEST_CHECK(sand.alpha > 10.0f); // Low air-entry pressure
    }

    DOCTEST_SUBCASE("Clay properties reasonable") {
        SoilProperties clay = getSoilPropertiesFromTexture("clay");
        DOCTEST_CHECK(clay.K_s < 1e-5f); // Low conductivity
        DOCTEST_CHECK(clay.n < 1.5f); // Broad pore-size distribution
    }
}

// ============================================================================
// SoilColumn tests
// ============================================================================

DOCTEST_TEST_CASE("SoilColumn - Construction") {

    DOCTEST_SUBCASE("Uniform grid") {
        SoilProperties props = getSoilPropertiesFromTexture("loam");
        SoilColumn col(20, 0.05f, props);

        DOCTEST_CHECK(col.getNumLayers() == 20);
        DOCTEST_CHECK(col.getColumnDepth() == doctest::Approx(1.0f));
    }

    DOCTEST_SUBCASE("Variable grid") {
        std::vector<float> dz = {0.01f, 0.02f, 0.05f, 0.1f, 0.2f};
        SoilProperties props = getSoilPropertiesFromTexture("loam");
        SoilColumn col(dz, props);

        DOCTEST_CHECK(col.getNumLayers() == 5);
        DOCTEST_CHECK(col.getColumnDepth() == doctest::Approx(0.38f));
    }
}

DOCTEST_TEST_CASE("SoilColumn - Initial conditions") {

    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilColumn col(10, 0.1f, props);

    DOCTEST_SUBCASE("Uniform initial condition") {
        col.setInitialCondition(-2.0f);
        for (uint k = 0; k < col.getNumLayers(); k++) {
            DOCTEST_CHECK(col.getPressureHead(k) == doctest::Approx(-2.0f));
        }
    }

    DOCTEST_SUBCASE("Hydrostatic initial condition") {
        col.setInitialConditionHydrostatic(0.5f); // Water table at 0.5m
        // Above water table should be negative, below positive
        DOCTEST_CHECK(col.getPressureHead(0) < 0.0f); // Top is unsaturated
    }
}

DOCTEST_TEST_CASE("SoilColumn - Boundary conditions") {

    SoilProperties props = getSoilPropertiesFromTexture("sand");
    SoilColumn col(20, 0.05f, props);
    col.setInitialCondition(-0.5f);

    DOCTEST_SUBCASE("Zero flux conserves mass") {
        col.setTopBoundary(BCType::NEUMANN, 0.0f);
        col.setBottomBoundary(BCType::NEUMANN, 0.0f);

        float initial_storage = col.getTotalWaterStorage();

        // Run for several timesteps
        for (int i = 0; i < 100; i++) {
            col.solve(10.0f); // 10 second timesteps
        }

        float final_storage = col.getTotalWaterStorage();

        // Storage should be conserved (within numerical tolerance)
        DOCTEST_CHECK(final_storage == doctest::Approx(initial_storage).epsilon(0.001));
    }
}

DOCTEST_TEST_CASE("SoilColumn - Solver convergence") {

    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilColumn col(20, 0.05f, props);
    col.setInitialCondition(-1.0f);
    col.setTopBoundary(BCType::NEUMANN, 0.0f);
    col.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);

    DOCTEST_SUBCASE("Converges on reasonable timestep") {
        uint iters = col.solve(60.0f); // 1 minute timestep
        DOCTEST_CHECK(col.didConverge());
        DOCTEST_CHECK(iters > 0);
        DOCTEST_CHECK(iters < 20); // Should converge in reasonable iterations
    }
}

// ============================================================================
// SoilGrid tests
// ============================================================================

DOCTEST_TEST_CASE("SoilGrid - Construction") {

    SoilProperties props = getSoilPropertiesFromTexture("loam");

    DOCTEST_SUBCASE("3x3x10 grid") {
        SoilGrid grid(3, 3, 10, 1.0f, 1.0f, 0.1f, props);

        DOCTEST_CHECK(grid.getNx() == 3);
        DOCTEST_CHECK(grid.getNy() == 3);
        DOCTEST_CHECK(grid.getNz() == 10);
        DOCTEST_CHECK(grid.getNumColumns() == 9);
    }
}

DOCTEST_TEST_CASE("SoilGrid - State access") {

    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(2, 2, 5, 1.0f, 1.0f, 0.2f, props);
    grid.setInitialCondition(-1.5f);

    DOCTEST_SUBCASE("Access by index") {
        float h = grid.getPressureHead(1, 1, 2);
        DOCTEST_CHECK(h == doctest::Approx(-1.5f));
    }

    DOCTEST_SUBCASE("Column access") {
        const SoilColumn &col = grid.getColumn(0, 1);
        DOCTEST_CHECK(col.getNumLayers() == 5);
    }

    DOCTEST_SUBCASE("Position lookup") {
        uint i, j;
        grid.setOrigin(0.0f, 0.0f, 0.0f);
        bool found = grid.findColumn(0.5f, 0.5f, i, j);
        DOCTEST_CHECK(found);
        DOCTEST_CHECK(i == 0);
        DOCTEST_CHECK(j == 0);

        found = grid.findColumn(1.5f, 1.5f, i, j);
        DOCTEST_CHECK(found);
        DOCTEST_CHECK(i == 1);
        DOCTEST_CHECK(j == 1);
    }
}

DOCTEST_TEST_CASE("SoilGrid - Parallel solving") {

    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(4, 4, 10, 0.5f, 0.5f, 0.1f, props);
    grid.setInitialCondition(-1.0f);
    grid.setTopBoundary(BCType::NEUMANN, 0.0f);
    grid.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);

    DOCTEST_SUBCASE("All columns converge") {
        uint converged = grid.solve(60.0f);
        DOCTEST_CHECK(converged == grid.getNumColumns());
    }
}

DOCTEST_TEST_CASE("SoilGrid - Quasi-3D mode") {

    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(5, 5, 10, 0.2f, 0.2f, 0.1f, props);
    grid.setInitialCondition(-1.0f);
    grid.setTopBoundary(BCType::NEUMANN, 0.0f);
    grid.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);

    DOCTEST_SUBCASE("Horizontal coupling can be enabled") {
        grid.enableHorizontalCoupling(5);
        uint converged = grid.solve(60.0f);
        DOCTEST_CHECK(converged > 0);
    }
}

// ============================================================================
// Integration test - Full plugin
// ============================================================================

DOCTEST_TEST_CASE("SoilTransportModel - Basic usage") {

    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();

    DOCTEST_SUBCASE("Domain initialization") {
        model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(2, 2, 1));
        model.setGridResolution(4, 4, 20);
        model.setSoilPropertiesFromTexture("sandy_loam");
        model.setInitialCondition(-0.5f);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryFreeDrainage();

        // Should run without error
        model.run(60.0f);
    }

    DOCTEST_SUBCASE("Water content query") {
        model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
        model.setGridResolution(2, 2, 10);
        model.setSoilPropertiesFromTexture("loam");
        model.setInitialCondition(-1.0f);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryFreeDrainage();

        float wc = model.getWaterContent(make_vec3(0.5f, 0.5f, -0.5f));
        DOCTEST_CHECK(wc > 0.0f);
        DOCTEST_CHECK(wc < 1.0f);
    }
}

// ============================================================================
// Self-test entry point
// ============================================================================

DOCTEST_TEST_CASE("SoilTransportModel - Mass conservation over time") {
    // Full-plugin run path: zero-flux boundaries must conserve water over many steps.
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();

    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(1, 1, 20);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-1.0f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFlux(0.0f);

    float initial_storage = model.getTotalWaterStorage();

    for (int i = 0; i < 100; i++) {
        model.run(10.0f);
    }

    float final_storage = model.getTotalWaterStorage();
    float relative_error = std::fabs(final_storage - initial_storage) / initial_storage;

    DOCTEST_CHECK(relative_error <= 0.01f);
}

DOCTEST_TEST_CASE("SoilUtils - effective capillary drive and sorptivity") {
    SoilProperties sl = getSoilPropertiesFromTexture("sandy_loam");

    float psi_f = effectiveCapillaryDrive(sl, -2.0f);
    DOCTEST_CHECK(psi_f > 0.0f);
    DOCTEST_CHECK(psi_f < 1.0f); // bounded, order ~1/alpha
    // k_r is ~0 in very dry soil, so deepening the initial suction barely changes psi_f.
    DOCTEST_CHECK(effectiveCapillaryDrive(sl, -5.0f) == doctest::Approx(psi_f).epsilon(0.05));

    float S = sorptivity(sl, -2.0f);
    DOCTEST_CHECK(S > 0.0f);
    // Drier initial soil has greater capillary uptake capacity -> larger sorptivity.
    DOCTEST_CHECK(sorptivity(sl, -3.0f) >= sorptivity(sl, -0.5f));

    // Saturated initial state -> no capillary drive, no sorptivity.
    DOCTEST_CHECK(effectiveCapillaryDrive(sl, 0.0f) == 0.0f);
    DOCTEST_CHECK(sorptivity(sl, 0.0f) == 0.0f);
}

// ============================================================================
// 3D solver (SoilSolver3D) verification cases
// ============================================================================

DOCTEST_TEST_CASE("SoilSolver3D - Gardner steady infiltration (exact, 1x1xNz)") {
    // The 3D solver must reproduce the exact Gardner steady profile along z.
    uint Nz = 50;
    double dz = 0.01;
    SoilProperties g = SoilProperties::gardner(0.06f, 0.40f, 2.0f, 1e-5f);
    double Ks = g.K_s, al = g.alpha, q = 0.5 * Ks;
    SoilSolver3D s(1, 1, Nz, 0.01, 0.01, dz, g);
    std::vector<double> h0(Nz);
    for (uint k = 0; k < Nz; k++)
        h0[k] = -((k + 0.5) * dz); // hydrostatic IC
    s.setInitialCondition(h0);
    s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, 0.0); // water table
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q); // infiltration flux

    std::vector<double> hp;
    for (int n = 0; n < 100000; n++) {
        hp = s.getPressureHeads();
        DOCTEST_REQUIRE(s.solve(60.0) > 0);
        double dm = 0.0;
        for (uint k = 0; k < Nz; k++)
            dm = std::max(dm, std::fabs(s.getPressureHead(0, 0, k) - hp[k]));
        if (dm < 1e-10 && n > 50)
            break;
    }
    auto h_exact = [&](double z) { return std::log(((Ks - q) * std::exp(-al * z) + q) / Ks) / al; };
    double err = 0.0;
    for (uint k = 0; k < Nz; k++)
        err = std::max(err, std::fabs(s.getPressureHead(0, 0, k) - h_exact((k + 0.5) * dz)));
    DOCTEST_CHECK(err < 1e-4);
}

DOCTEST_TEST_CASE("SoilSolver3D - nonlinear residual vanishes at the converged state") {
    // residual() is the keystone of the Newton-Krylov path: it must be ~0 (to solver
    // tolerance) exactly at a converged step and large at a perturbed state. A sign error
    // in any face/gravity/BC/storage term would leave the converged residual non-zero.
    // The 3D infiltration-into-drainage box exercises every term: x/y interior + no-flux
    // sides, z interior with gravity, a Neumann top, and a free-drainage bottom.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(4, 4, 6, 0.02, 0.02, 0.02, g);
    s.setInitialCondition(std::vector<double>(s.numCells(), -0.5));
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 0.2 * g.K_s); // gentle infiltration
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0); // gravity drainage out the bottom
    // x/y faces default to zero-flux.

    const double dt = 60.0;
    std::vector<double> hp;
    for (int n = 0; n < 5000; n++) {
        hp = s.getPressureHeads();
        DOCTEST_REQUIRE(s.solve(dt) > 0);
        double dm = 0.0;
        const std::vector<double> &hn = s.getPressureHeads();
        for (uint c = 0; c < s.numCells(); c++)
            dm = std::max(dm, std::fabs(hn[c] - hp[c]));
        if (dm < 1e-11 && n > 20)
            break; // steady state reached
    }

    double a0, a1, a2;
    s.timeWeights(dt, a0, a1, a2);
    const std::vector<double> hstar = s.getPressureHeads();

    std::vector<double> F;
    s.residual(hstar, F, a0, a1, a2);
    double rcvg = 0.0;
    for (double v: F)
        rcvg = std::max(rcvg, std::fabs(v));

    std::vector<double> hpert = hstar;
    for (double &v: hpert)
        v += 0.05; // 5 cm uniform shift off the solution
    s.residual(hpert, F, a0, a1, a2);
    double rpert = 0.0;
    for (double v: F)
        rpert = std::max(rpert, std::fabs(v));

    DOCTEST_CHECK(rpert > 1e-4); // the perturbation genuinely excites the residual
    DOCTEST_CHECK(rcvg < 1e-4 * rpert); // converged residual >= 4 orders below it (sign-error catch)
}

DOCTEST_TEST_CASE("SoilSolver3D - matrix-free Jacobian-vector product") {
    // jacvec() must equal the assembled Modified-Picard operator when the dropped Newton
    // terms vanish (saturated: dK/dh = 0), and differ measurably when they don't (unsaturated).
    // Because residual() is written in theta(h), jacvec - picardApply isolates exactly the
    // dK/dh nonsymmetric Jacobian terms.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(5, 5, 6, 0.02, 0.02, 0.02, g);
    const uint N = s.numCells();
    const double a0 = 1.0 / 60.0, a1 = -1.0 / 60.0, a2 = 0.0;

    std::vector<double> v(N); // deterministic, varied test direction
    for (uint c = 0; c < N; c++)
        v[c] = 0.01 * std::sin(0.7 * (double) c + 1.0);

    auto reldiff = [&](const std::vector<double> &h) {
        std::vector<double> F0, Jv, Av;
        s.residual(h, F0, a0, a1, a2);
        s.jacvec(h, F0, v, Jv, a0, a1, a2);
        s.picardApply(h, v, Av, a0);
        double num = 0.0, den = 0.0;
        for (uint c = 0; c < N; c++) {
            num = std::max(num, std::fabs(Jv[c] - Av[c]));
            den = std::max(den, std::fabs(Av[c]));
        }
        return std::make_pair(num / (den + 1e-300), den);
    };

    { // Saturated (h > 0): K = Ks constant => true Jacobian == Picard operator
        std::vector<double> h(N, 0.1);
        std::pair<double, double> r = reldiff(h);
        DOCTEST_CHECK(r.second > 0.0); // operator is non-trivial
        DOCTEST_CHECK(r.first < 1e-5); // FD Jv matches the assembled operator to FD precision
    }
    { // Unsaturated (steep K(h)): the dropped dK/dh terms are present and non-trivial
        std::vector<double> h(N);
        for (uint c = 0; c < N; c++)
            h[c] = -0.5 - 0.5 * std::sin(0.3 * (double) c);
        std::pair<double, double> r = reldiff(h);
        DOCTEST_CHECK(r.first > 1e-3); // Newton terms make J differ measurably from Picard
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - GMRES recovers the solution of an SPD system") {
    // Isolate the Krylov machinery: drive GMRES with the SPD Modified-Picard operator and a
    // known right-hand side b = A x_true, and confirm it recovers x_true. This validates
    // Arnoldi/Givens/restart independently of the nonlinear (jacvec) and preconditioner pieces.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(6, 6, 6, 0.02, 0.02, 0.02, g);
    const uint N = s.numCells();
    const double a0 = 1.0 / 60.0;
    std::vector<double> h(N, -0.5); // unsaturated -> A = a0 C + L_K is SPD, well-conditioned

    std::vector<double> x_true(N);
    for (uint c = 0; c < N; c++)
        x_true[c] = std::sin(0.5 * (double) c) + 0.3;
    std::vector<double> b;
    s.picardApply(h, x_true, b, a0); // b = A x_true

    auto apply = [&](const std::vector<double> &xx, std::vector<double> &yy) { s.picardApply(h, xx, yy, a0); };
    auto ident = [&](const std::vector<double> &r, std::vector<double> &z) { z = r; };

    std::vector<double> x(N, 0.0);
    uint it = s.gmres(apply, ident, b, x, 40, 400, 1e-12);

    double err = 0.0, scale = 0.0;
    for (uint c = 0; c < N; c++) {
        err = std::max(err, std::fabs(x[c] - x_true[c]));
        scale = std::max(scale, std::fabs(x_true[c]));
    }
    DOCTEST_CHECK(err / scale < 1e-8); // GMRES recovers the known solution
    DOCTEST_CHECK(it <= N); // SPD system converges within N iterations (typically far fewer)
}

DOCTEST_TEST_CASE("SoilSolver3D - Newton-Krylov matches Picard, converges quadratically") {
    // solveNewton() must reach the SAME state as the Modified-Picard solve() (mass-conservative),
    // in fewer outer iterations, and with a near-quadratic residual drop near the root.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    auto setup = [&](SoilSolver3D &s) {
        s.setInitialCondition(std::vector<double>(s.numCells(), -2.0)); // dry
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05); // near-saturated top -> stiff front
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    };
    SoilSolver3D sp(6, 6, 8, 0.02, 0.02, 0.02, g); // Picard reference
    SoilSolver3D sn(6, 6, 8, 0.02, 0.02, 0.02, g); // Newton-Krylov
    setup(sp);
    setup(sn);
    sp.resetMassBalance();
    sn.resetMassBalance();

    const double dt = 30.0;
    uint picard_total = 0, newton_total = 0;
    for (int n = 0; n < 20; n++) {
        uint pi = sp.solve(dt);
        DOCTEST_REQUIRE(pi > 0);
        picard_total += pi;
        uint ni = sn.solveNewton(dt);
        DOCTEST_REQUIRE(ni > 0);
        newton_total += ni;
    }

    // (1) Same converged solution as Picard.
    double dmax = 0.0, scale = 0.0;
    const std::vector<double> &hp = sp.getPressureHeads();
    const std::vector<double> &hn = sn.getPressureHeads();
    for (uint c = 0; c < sp.numCells(); c++) {
        dmax = std::max(dmax, std::fabs(hp[c] - hn[c]));
        scale = std::max(scale, std::fabs(hp[c]));
    }
    DOCTEST_CHECK(dmax / scale < 1e-6);

    // (2) Newton converges in few outer iterations per step (quadratic -> ~3-4/step here).
    //     Note: this dt is not stiff enough for Newton to beat Picard on cost; Newton's edge
    //     is on stiff steps where Picard's linear rate stalls. Here we only require it is not
    //     pathological (the orchestrator, not pure Newton, is what wins on cost).
    DOCTEST_CHECK(newton_total <= 6 * 20);
    (void) picard_total;

    // (3) Mass conservation preserved by the Newton bookkeeping.
    DOCTEST_CHECK(std::fabs(sn.getMassBalanceResidual()) < 1e-6 * std::fabs(sn.getCumulativeInflux()) + 1e-12);

    // (4) Quadratic convergence signature: run one stiff step and confirm the residual-drop
    //     ratio collapses toward zero (Newton) rather than staying constant (Picard).
    SoilSolver3D sq(6, 6, 8, 0.02, 0.02, 0.02, g);
    setup(sq);
    DOCTEST_REQUIRE(sq.solveNewton(dt) > 0);
    const std::vector<double> &rh = sq.getNewtonResidualHistory();
    DOCTEST_REQUIRE(rh.size() >= 3);
    double ratio_first = rh[1] / (rh[0] + 1e-300);
    double ratio_last = rh[rh.size() - 1] / (rh[rh.size() - 2] + 1e-300);
    DOCTEST_CHECK(ratio_last < 0.1 * ratio_first); // ratio collapses => super-linear/quadratic
}

DOCTEST_TEST_CASE("SoilSolver3D - nonlinear method dispatch: Newton & Picard-Newton match Picard") {
    // The NonlinearMethod dispatch (NEWTON, PICARD_NEWTON) must reach the same converged,
    // mass-conservative state as the default Modified-Picard path on a stiff infiltration.
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    auto setup = [&](SoilSolver3D &s, NM m) {
        s.setInitialCondition(std::vector<double>(s.numCells(), -3.0)); // dry
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05); // near-saturated top
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        SoilSolver3D::Params p = s.getParams();
        p.nonlinear_method = m;
        s.setParams(p);
        s.resetMassBalance();
    };
    SoilSolver3D sp(6, 6, 8, 0.02, 0.02, 0.02, g);
    SoilSolver3D sn(6, 6, 8, 0.02, 0.02, 0.02, g);
    SoilSolver3D sw(6, 6, 8, 0.02, 0.02, 0.02, g);
    setup(sp, NM::PICARD);
    setup(sn, NM::NEWTON);
    setup(sw, NM::PICARD_NEWTON);

    const double dt = 120.0;
    for (int n = 0; n < 15; n++) {
        DOCTEST_REQUIRE(sp.solve(dt) > 0);
        DOCTEST_REQUIRE(sn.solve(dt) > 0);
        DOCTEST_REQUIRE(sw.solve(dt) > 0);
    }

    double dn = 0.0, dw = 0.0, sc = 0.0;
    const std::vector<double> &hp = sp.getPressureHeads();
    for (uint c = 0; c < sp.numCells(); c++) {
        dn = std::max(dn, std::fabs(hp[c] - sn.getPressureHeads()[c]));
        dw = std::max(dw, std::fabs(hp[c] - sw.getPressureHeads()[c]));
        sc = std::max(sc, std::fabs(hp[c]));
    }
    DOCTEST_CHECK(dn / sc < 1e-6); // Newton matches Picard
    DOCTEST_CHECK(dw / sc < 1e-6); // Picard-Newton matches Picard
    DOCTEST_CHECK(std::fabs(sn.getMassBalanceResidual()) < 1e-6 * std::fabs(sn.getCumulativeInflux()) + 1e-12);
    DOCTEST_CHECK(std::fabs(sw.getMassBalanceResidual()) < 1e-6 * std::fabs(sw.getCumulativeInflux()) + 1e-12);
}

DOCTEST_TEST_CASE("SoilSolver3D - analytic Newton reproduces the exact Gardner steady profile") {
    // NEWTON_ANALYTIC must converge to the exact Gardner steady solution for downward (infiltration) and
    // upward (evaporation) flux, using long steps that march straight to steady state.
    uint Nz = 50;
    double dz = 0.01;
    SoilProperties g = SoilProperties::gardner(0.06f, 0.40f, 2.0f, 1e-5f);
    const double Ks = g.K_s, al = g.alpha;
    for (double q: {0.5 * Ks, -0.2 * Ks}) {
        SoilSolver3D s(1, 1, Nz, 0.01, 0.01, dz, g);
        std::vector<double> h0(Nz);
        for (uint k = 0; k < Nz; k++)
            h0[k] = -((k + 0.5) * dz);
        s.setInitialCondition(h0);
        s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, 0.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
        SoilSolver3D::Params p = s.getParams();
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
        s.setParams(p);
        for (int n = 0; n < 40; n++)
            DOCTEST_REQUIRE(s.solve(1e5) > 0);
        auto h_exact = [&](double z) { return std::log(((Ks - q) * std::exp(-al * z) + q) / Ks) / al; };
        double err = 0.0;
        for (uint k = 0; k < Nz; k++)
            err = std::max(err, std::fabs(s.getPressureHead(0, 0, k) - h_exact((k + 0.5) * dz)));
        DOCTEST_CHECK(err < 1e-4);
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - analytic Newton matches Picard under every K weighting") {
    // The assembled Jacobian path (faceCondDeriv) and the Picard path (faceCond) discretize the same
    // equations, so for every interblock weighting -- including upstream relative permeability (4) -- they must reach the
    // same mass-conservative state on a stiff, layered infiltration problem.
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilProperties sand = getSoilPropertiesFromTexture("sand");
    for (int w = 0; w <= 4; w++) {
        auto setup = [&](SoilSolver3D &s, NM m) {
            for (uint k = 0; k < 4; k++)
                for (uint j = 0; j < 6; j++)
                    for (uint i = 0; i < 6; i++)
                        s.setSoilProperties(i, j, k, sand); // sand below loam: Ks contrast at the interface
            s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
            s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
            s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
            SoilSolver3D::Params p = s.getParams();
            p.nonlinear_method = m;
            p.K_weighting = w;
            p.picard_tol = 1e-9;
            p.picard_max = 400;
            s.setParams(p);
            s.resetMassBalance();
        };
        SoilSolver3D sp(6, 6, 8, 0.02, 0.02, 0.02, loam), sn(6, 6, 8, 0.02, 0.02, 0.02, loam);
        setup(sp, NM::PICARD);
        setup(sn, NM::NEWTON_ANALYTIC);
        uint newton_total = 0;
        for (int n = 0; n < 15; n++) {
            DOCTEST_REQUIRE(sp.solve(60.0) > 0);
            const uint it = sn.solve(60.0);
            DOCTEST_REQUIRE(it > 0);
            newton_total += it;
        }
        double dmax = 0.0, scale = 0.0;
        for (uint c = 0; c < sp.numCells(); c++) {
            dmax = std::max(dmax, std::fabs(sp.getPressureHeads()[c] - sn.getPressureHeads()[c]));
            scale = std::max(scale, std::fabs(sp.getPressureHeads()[c]));
        }
        DOCTEST_CHECK(dmax / scale < 1e-5);
        DOCTEST_CHECK(newton_total <= 8 * 15);
        DOCTEST_CHECK(std::fabs(sn.getMassBalanceResidual()) < 1e-6 * std::fabs(sn.getCumulativeInflux()) + 1e-12);
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - analytic Newton converges quadratically") {
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(6, 6, 8, 0.02, 0.02, 0.02, g);
    s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
    s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    SoilSolver3D::Params p = s.getParams();
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
    s.setParams(p);
    const uint it = s.solve(120.0);
    DOCTEST_REQUIRE(it > 0);
    DOCTEST_CHECK(it <= 8);
    // Quadratic endgame: after the first iteration some step reduces ||R|| by more than 1e3
    // (a linearly convergent Picard iteration contracts by a roughly constant factor instead).
    const std::vector<double> &rh = s.getNewtonResidualHistory();
    DOCTEST_REQUIRE(rh.size() >= 3);
    double best_ratio = 1.0;
    for (size_t i = 2; i < rh.size(); i++)
        best_ratio = std::min(best_ratio, rh[i] / (rh[i - 1] + 1e-300));
    DOCTEST_CHECK(best_ratio < 1e-3);
}

DOCTEST_TEST_CASE("SoilSolver3D - analytic Newton falls back to Picard on a stalled fixed-dt step") {
    // One long step of a sharp wetting front into dry loam: Newton stalls, and solve() must still return a
    // converged, mass-conservative step by falling back to Modified Picard.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D sn(6, 6, 8, 0.02, 0.02, 0.02, g), sp(6, 6, 8, 0.02, 0.02, 0.02, g);
    for (SoilSolver3D *s: {&sn, &sp}) {
        s->setInitialCondition(std::vector<double>(s->numCells(), -2.0));
        s->setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
        s->setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s->resetMassBalance();
    }
    SoilSolver3D::Params p = sn.getParams();
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
    sn.setParams(p);
    DOCTEST_REQUIRE(sn.solve(3600.0) > 0);
    DOCTEST_REQUIRE(sp.solve(3600.0) > 0);
    double dmax = 0.0;
    for (uint c = 0; c < sn.numCells(); c++)
        dmax = std::max(dmax, std::fabs(sn.getPressureHeads()[c] - sp.getPressureHeads()[c]));
    DOCTEST_CHECK(dmax < 1e-5);
    DOCTEST_CHECK(std::fabs(sn.getMassBalanceResidual()) < 1e-6 * std::fabs(sn.getCumulativeInflux()) + 1e-12);
}


#ifdef HELIOS_CUDA_AVAILABLE
DOCTEST_TEST_CASE("SoilSolver3D - relative-conductivity spline: GPU matches CPU on ponded clay") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - relative-conductivity spline: GPU matches CPU on ponded clay'"))
        return;
    SoilProperties clay = getSoilPropertiesFromTexture("clay");
    clay.kr_spline_start = 0.9999f;
    auto run = [&](bool gpu) {
        SoilSolver3D s(12, 12, 10, 0.05, 0.05, 0.05, clay);
        s.setInitialCondition(-1.0);
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, 0.0); // ponded surface
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        SoilSolver3D::Params p = s.getParams();
        p.use_gpu = gpu;
        p.bdf2 = false;
        s.setParams(p);
        for (int n = 0; n < 12; n++)
            DOCTEST_REQUIRE(s.solve(600.0) > 0);
        return s.getPressureHeads();
    };
    const std::vector<double> hc = run(false), hg = run(true);
    double err = 0.0;
    for (size_t c = 0; c < hc.size(); c++)
        err = std::max(err, std::fabs(hc[c] - hg[c]));
    DOCTEST_CHECK(err < 1e-4);
}

DOCTEST_TEST_CASE("SoilSolver3D - modified van Genuchten: GPU matches CPU on a drip emitter") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - modified van Genuchten: GPU matches CPU on a drip emitter'"))
        return;
    // A drip emitter into loam on the air-entry model, NEWTON_ANALYTIC on both backends.
    const SoilProperties mv = getSoilPropertiesFromTexture("loam", SoilModel::MODIFIED_VAN_GENUCHTEN);
    auto run = [&](bool gpu) {
        SoilSolver3D s(16, 16, 12, 0.05, 0.05, 0.05, mv);
        s.setInitialCondition(-1.0);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        for (uint j = 7; j <= 8; j++)
            for (uint i = 7; i <= 8; i++)
                s.setSource(i, j, 11, 1e-3); // ~17 x K_s into the 4 surface cells: they saturate
        SoilSolver3D::Params p = s.getParams();
        p.use_gpu = gpu;
        p.bdf2 = false;
        s.setParams(p);
        for (int n = 0; n < 12; n++)
            DOCTEST_REQUIRE(s.solve(300.0) > 0);
        return s.getPressureHeads();
    };
    const std::vector<double> hc = run(false), hg = run(true);
    double err = 0.0, hmax = -1e30;
    for (size_t c = 0; c < hc.size(); c++) {
        err = std::max(err, std::fabs(hc[c] - hg[c]));
        hmax = std::max(hmax, hc[c]);
    }
    DOCTEST_CHECK(hmax >= saturationHead(mv)); // the emitter cells saturated
    DOCTEST_CHECK(err < 1e-4);
}
#endif

DOCTEST_TEST_CASE("SoilSolver3D - a failed fixed-dt step leaves the state unchanged") {
    // With Newton and its Picard fallback both capped at one iteration, the stiff step above cannot
    // converge: solve() must report the failure and keep the start-of-step heads (the last Picard iterate
    // of a failed step can be wildly wrong).
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(6, 6, 8, 0.02, 0.02, 0.02, g);
    s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
    s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    SoilSolver3D::Params p = s.getParams();
    p.newton_max = 1;
    p.picard_max = 1;
    s.setParams(p);
    const std::vector<double> h0 = s.getPressureHeads();
    DOCTEST_CHECK(s.solve(3600.0) == 0);
    DOCTEST_CHECK(s.getPressureHeads() == h0);
}

DOCTEST_TEST_CASE("SoilSolver3D - a stalled fixed-dt step is recovered by sub-stepping") {
    // Daily recharge fills a closed column with specific storage until the last unsaturated cell at the top
    // must saturate within one step, after which the column can only take water by compression. Newton stalls
    // on that step and grids with specific storage have no Picard fallback: solve() must sub-step the day
    // rather than fail, and still conserve water.
    const uint nz = 100;
    const double dz = 0.04;
    SoilProperties g(0.02f, 0.35f, 1.6f, 2.7f, 1e-5f);
    SoilSolver3D s(1, 1, nz, 1.0, 1.0, dz, g);
    std::vector<double> h0(nz);
    for (uint k = 0; k < nz; k++)
        h0[k] = 2.0 - (k + 0.5) * dz;
    s.setInitialCondition(h0);
    s.setSpecificStorage(std::vector<double>(nz, 1e-5));
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 2e-7);
    SoilSolver3D::Params p = s.getParams();
    p.bdf2 = false;
    s.setParams(p);
    s.resetMassBalance();
    const double W0 = s.getTotalWaterStorage();
    for (int n = 0; n < 30; n++)
        DOCTEST_REQUIRE(s.solve(86400.0) > 0);
    DOCTEST_CHECK(s.getTotalWaterStorage() - W0 == doctest::Approx(2e-7 * 30 * 86400.0).epsilon(1e-6));
    DOCTEST_CHECK(std::fabs(s.getMassBalanceResidual()) < 1e-9);
}

DOCTEST_TEST_CASE("SoilSolver3D - analytic Newton converges around a water table with specific storage") {
    // A 2D slab with a sloping water table, recharge on half its surface and a specified-head outlet, in a
    // soil with a sharp air entry (Se = 0.9999 a few cm above the water table). Mapping the near-saturated
    // cells' updates through the retention curve snaps them to h = 0 and holds cells at the kink whenever
    // they cross it: Newton then stalled on most hourly steps. With specific storage they step in head.
    const uint nx = 20, nz = 50;
    const double dz = 0.08;
    SoilProperties g(0.02f, 0.35f, 1.6f, 2.7f, 1e-5f);
    SoilSolver3D s(nx, 1, nz, 1.0, 1.0, dz, g);
    std::vector<double> h0(s.numCells());
    for (uint k = 0; k < nz; k++)
        for (uint i = 0; i < nx; i++)
            h0[s.index(i, 0, k)] = 1.0 + 2.0 * i / nx - (k + 0.5) * dz;
    s.setInitialCondition(h0);
    s.setSpecificStorage(std::vector<double>(s.numCells(), 1e-5));
    for (uint i = 0; i < nx / 2; i++)
        s.setSource(i, 0, nz - 1, 3e-7 / dz);
    for (uint k = 0; k < nz / 4; k++)
        s.setFixedHead(nx - 1, 0, k, 0.5 - (k + 0.5) * dz);
    SoilSolver3D::Params p = s.getParams();
    p.bdf2 = false;
    s.setParams(p);
    s.resetMassBalance();
    uint total = 0;
    for (int n = 0; n < 48; n++) {
        const uint it = s.solve(3600.0);
        DOCTEST_REQUIRE(it > 0);
        total += it;
    }
    DOCTEST_CHECK(total <= 48 * 6);
    DOCTEST_CHECK(std::fabs(s.getMassBalanceResidual()) < 1e-9);
}

DOCTEST_TEST_CASE("SoilSolver3D - analytic Newton on a layered column with near-singular clay (n = 1.09)") {
    // A reduced version of the MODFLOW 6 Richards Test 1 column (Vazquez-Gasty et al. 2026): coarse sand,
    // a thin clay band with n = 1.09 (K has an infinite slope at saturation) and silt, recharge on top,
    // water table at the base, started from no-flow hydrostatic and marched to steady state with long
    // adaptive steps. Steady state must be reached with water conserved.
    const uint Nz = 200;
    const double dz = 6.0 / Nz, day = 86400.0, recharge = 9.448e-4 / day;
    SoilProperties sand(0.02f, 0.31f, 16.0f, 2.9f, (float) (86.4 / day));
    SoilProperties clay(0.068f, 0.38f, 0.8f, 1.09f, (float) (0.048 / day));
    SoilProperties silt(0.034f, 0.46f, 1.6f, 1.37f, (float) (0.06 / day));
    SoilSolver3D s(1, 1, Nz, 1.0, 1.0, dz, sand);
    std::vector<double> h0(Nz);
    for (uint k = 0; k < Nz; k++) {
        const double depth = 6.0 - (k + 0.5) * dz;
        if (depth > 2.0 && depth < 2.3)
            s.setSoilProperties(0, 0, k, clay);
        if (depth > 4.0 && depth < 4.6)
            s.setSoilProperties(0, 0, k, silt);
        h0[k] = -((k + 0.5) * dz);
    }
    s.setInitialCondition(h0);
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, recharge);
    s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, 0.0);
    SoilSolver3D::Params p = s.getParams();
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
    p.K_weighting = 4;
    p.bdf2 = false;
    p.dt_target_rel_change = 1.0; // march to steady state (no accuracy limit on dt)
    p.dt_max = 365.0 * day;
    s.setParams(p);
    s.resetMassBalance();
    double dt = 60.0;
    s.solveAdaptive(100.0 * 365.0 * day, dt);
    const std::vector<double> h_end = s.getPressureHeads();
    DOCTEST_REQUIRE(s.solve(365.0 * day) > 0);
    double change = 0.0;
    for (uint k = 0; k < Nz; k++) {
        DOCTEST_REQUIRE(std::isfinite(h_end[k]));
        change = std::max(change, std::fabs(s.getPressureHead(0, 0, k) - h_end[k]));
    }
    DOCTEST_CHECK(change < 1e-4); // steady
    DOCTEST_CHECK(s.getPressureHead(0, 0, Nz - 1) < 0.0); // the top stays unsaturated under this recharge
    // Water balance defect relative to the total recharge that passed through the column (the net
    // influx, i.e. the storage change, is much smaller than the throughput).
    DOCTEST_CHECK(std::fabs(s.getMassBalanceResidual()) < 1e-6 * recharge * 101.0 * 365.0 * day);
}

DOCTEST_TEST_CASE("SoilSolver3D - per-cell geometry equal to the uniform grid reproduces it") {
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    auto setup = [&](SoilSolver3D &s) {
        s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s.setBoundary(Face3D::XMIN, BCType::NEUMANN, 1e-7);
    };
    SoilSolver3D a(5, 4, 6, 0.03, 0.03, 0.02, g), b(5, 4, 6, 0.03, 0.03, 0.02, g);
    setup(a);
    setup(b);
    std::vector<double> zc(b.numCells()), thk(b.numCells(), 0.02);
    for (uint c = 0; c < b.numCells(); c++)
        zc[c] = (c / 20 + 0.5) * 0.02;
    b.setCellGeometry(zc, thk);
    for (int n = 0; n < 10; n++) {
        DOCTEST_REQUIRE(a.solve(120.0) > 0);
        DOCTEST_REQUIRE(b.solve(120.0) > 0);
    }
    double d = 0.0;
    for (uint c = 0; c < a.numCells(); c++)
        d = std::max(d, std::fabs(a.getPressureHeads()[c] - b.getPressureHeads()[c]));
    DOCTEST_CHECK(d < 1e-9);
}

DOCTEST_TEST_CASE("SoilSolver3D - hydrostatic state on a sloping, variably layered grid stays at rest") {
    // A terrain-following grid (surface sloping 3% in x, layers thickening with depth). With no-flow
    // boundaries and pressure head h = H0 - z_c everywhere, total head is uniform: nothing should move.
    const uint nx = 12, ny = 4, nz = 8;
    SoilProperties g = getSoilPropertiesFromTexture("sandy_loam");
    SoilSolver3D s(nx, ny, nz, 1.0, 1.0, 0.1, g);
    std::vector<double> zc(s.numCells()), thk(s.numCells()), h0(s.numCells());
    const double t[nz] = {0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.1}; // bottom to top
    for (uint j = 0; j < ny; j++)
        for (uint i = 0; i < nx; i++) {
            double z = 0.03 * i; // sloping base
            for (uint k = 0; k < nz; k++) {
                const uint c = s.index(i, j, k);
                thk[c] = t[k];
                zc[c] = z + 0.5 * t[k];
                z += t[k];
                h0[c] = 1.5 - zc[c]; // water table at z = 1.5
            }
        }
    s.setCellGeometry(zc, thk);
    s.setInitialCondition(h0);
    s.resetMassBalance();
    for (int n = 0; n < 5; n++)
        DOCTEST_REQUIRE(s.solve(86400.0) > 0);
    double d = 0.0;
    for (uint c = 0; c < s.numCells(); c++)
        d = std::max(d, std::fabs(s.getPressureHeads()[c] - h0[c]));
    DOCTEST_CHECK(d < 1e-8);
}

DOCTEST_TEST_CASE("SoilSolver3D - specified-head cells and specific storage conserve water") {
    // A sloping saturated-unsaturated slab drains to a specified-head column, with specific storage and
    // recharge: the stored-water change must equal the net inflow (recharge minus what the specified
    // heads take up), and the specified heads must not move.
    const uint nx = 10, ny = 3, nz = 6;
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(nx, ny, nz, 2.0, 2.0, 0.2, g);
    std::vector<double> zc(s.numCells()), thk(s.numCells(), 0.2), h0(s.numCells()), ss(s.numCells(), 1e-4);
    for (uint c = 0; c < s.numCells(); c++) {
        const uint k = c / (nx * ny), i = c % nx;
        zc[c] = 0.05 * (nx - i) + (k + 0.5) * 0.2;
        h0[c] = 0.8 - zc[c];
    }
    s.setCellGeometry(zc, thk);
    s.setSpecificStorage(ss);
    s.setInitialCondition(h0);
    SoilSolver3D::Params p = s.getParams();
    p.bdf2 = false; // with backward Euler the cumulative influx integrates exactly to the storage change
    s.setParams(p);
    for (uint j = 0; j < ny; j++)
        for (uint k = 0; k < 3; k++)
            s.setFixedHead(nx - 1, j, k, 0.3 - zc[s.index(nx - 1, j, k)]);
    for (uint j = 0; j < ny; j++)
        for (uint i = 0; i < nx; i++)
            s.setSource(i, j, nz - 1, 2e-8 / 0.2);
    s.resetMassBalance();
    const double W0 = s.getTotalWaterStorage();
    for (int n = 0; n < 20; n++)
        DOCTEST_REQUIRE(s.solve(3600.0) > 0);
    const double dW = s.getTotalWaterStorage() - W0;
    DOCTEST_CHECK(std::fabs(dW - s.getCumulativeInflux()) < 1e-6 * (std::fabs(dW) + 1e-3));
    DOCTEST_CHECK(std::fabs(s.getMassBalanceResidual()) < 1e-9);
    for (uint j = 0; j < ny; j++)
        DOCTEST_CHECK(std::fabs(s.getPressureHead(nx - 1, j, 0) - (0.3 - zc[s.index(nx - 1, j, 0)])) < 1e-12);
}

DOCTEST_TEST_CASE("SoilSolver3D - specific storage of a saturated column") {
    // A fully saturated, closed column with Ss: raising the pressure by injecting water stores Ss * dh per
    // unit volume. Inject a known volume and check the uniform pressure rise.
    const uint nz = 10;
    const double Ss = 1e-4, q = 1e-6, T = 1000.0;
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(1, 1, nz, 1.0, 1.0, 0.1, g);
    std::vector<double> h0(nz);
    for (uint k = 0; k < nz; k++)
        h0[k] = 2.0 - (k + 0.5) * 0.1; // saturated hydrostatic
    s.setInitialCondition(h0);
    s.setSpecificStorage(std::vector<double>(nz, Ss));
    SoilSolver3D::Params p = s.getParams();
    p.bdf2 = false;
    s.setParams(p);
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
    DOCTEST_REQUIRE(s.solve(T) > 0);
    // injected q*T per unit area into a 1 m column: dh = q*T / (Ss * L) at hydrostatic equilibrium
    const double dh_expected = q * T / (Ss * 1.0);
    double mean_dh = 0.0;
    for (uint k = 0; k < nz; k++)
        mean_dh += (s.getPressureHead(0, 0, k) - h0[k]) / nz;
    DOCTEST_CHECK(std::fabs(mean_dh - dh_expected) < 1e-3 * dh_expected);
}

DOCTEST_TEST_CASE("SoilColumn - recharge into very dry coarse soil stays physical and conserves water") {
    // Regression: the pressure-head (non-conservative) storage form with a column-relative convergence
    // test accepted a top-layer head of ~1e7 m for a small recharge into dry sand (C ~ 0) at long steps.
    const uint nz = 200;
    const float dz = 0.03f;
    SoilProperties sand(0.02f, 0.31f, 16.0f, 2.9f, 1e-3f);
    SoilColumn column(nz, dz, sand);
    std::vector<float> h0(nz);
    for (uint k = 0; k < nz; k++)
        h0[k] = -(nz - k - 0.5f) * dz; // water table at the base, top ~6 m of suction
    column.setInitialCondition(h0);
    column.setTopBoundary(BCType::NEUMANN, 1.1e-8f);
    column.setBottomBoundary(BCType::DIRICHLET, 0.0f);
    const float storage0 = column.getTotalWaterStorage();
    for (int n = 0; n < 24; n++) {
        DOCTEST_REQUIRE(column.solve(3600.0f) > 0);
        const float ht = column.getPressureHead(0);
        DOCTEST_CHECK(std::isfinite(ht));
        DOCTEST_CHECK(ht < 0.0f); // ~1 mm/d cannot saturate sand
        DOCTEST_CHECK(ht > -6.0f); // and the top cannot dry below its initial suction
    }
    // The base starts in hydrostatic equilibrium with the water table, and the wetting reaches only the
    // top few centimetres in a day, so the stored water rises by exactly the applied recharge.
    const float applied = 1.1e-8f * 24.0f * 3600.0f;
    DOCTEST_CHECK(std::fabs(column.getTotalWaterStorage() - storage0 - applied) < 0.02f * applied);
}

DOCTEST_TEST_CASE("SoilColumn - upstream weighting follows the flow direction") {
    // Steady upward flow (evaporation from a water table) from a coarse layer into a fine one. Upstream
    // weighting must take the donor's K -- the lower, coarse layer -- at the interface. Taking the upper
    // layer's K regardless of flow direction carries the flux with a conductivity ~100x too small.
    const uint nz = 40;
    const float dz = 0.01f;
    SoilProperties fine = SoilProperties::gardner(0.06f, 0.40f, 2.0f, 1e-6f);
    SoilProperties coarse = SoilProperties::gardner(0.04f, 0.35f, 2.0f, 1e-4f);
    const double q = -0.5e-6; // upward, below the fine layer's Ks
    SoilColumn column(nz, dz, coarse);
    column.setLayerSoilProperties(0, nz / 2, fine); // layer 0 is the top
    std::vector<float> h0(nz);
    for (uint k = 0; k < nz; k++)
        h0[k] = -(nz - k - 0.5f) * dz;
    column.setInitialCondition(h0);
    column.setTopBoundary(BCType::NEUMANN, (float) q);
    column.setBottomBoundary(BCType::DIRICHLET, 0.0f);
    SolverParameters p;
    p.K_weighting = KWeighting::UPSTREAM;
    p.max_picard_iterations = 200;
    column.setSolverParameters(p);
    for (int n = 0; n < 5000; n++)
        DOCTEST_REQUIRE(column.solve(600.0f) > 0);
    const uint a = nz / 2 - 1, b = nz / 2; // fine above, coarse below the interface
    const double ha = column.getPressureHead(a), hb = column.getPressureHead(b);
    const double K_donor = coarse.K_s * std::exp(coarse.alpha * hb);
    const double q_face = K_donor * ((ha - hb) / dz + 1.0); // downward flux across the interface
    DOCTEST_CHECK(std::fabs(q_face - q) < 1e-2 * std::fabs(q)); // steady: every face carries q
}

DOCTEST_TEST_CASE("SoilColumn - upstream relative-permeability weighting matches the 3D solver") {
    // KWeighting::UPSTREAM_KR (harmonic-mean Ks times upstream relative permeability) must give the same
    // layered steady profile in the 1D column and the 3D solver (K_weighting = 4).
    const uint nz = 60;
    const double dz = 0.02, q = 2e-7;
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilProperties sand = getSoilPropertiesFromTexture("sand");
    SoilColumn column(nz, (float) dz, loam);
    column.setLayerSoilProperties(nz / 2, nz, sand); // column layer 0 is the top: sand below
    column.setInitialCondition(-1.0f);
    column.setTopBoundary(BCType::NEUMANN, (float) q);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    SolverParameters cp;
    cp.K_weighting = KWeighting::UPSTREAM_KR;
    cp.max_picard_iterations = 200;
    column.setSolverParameters(cp);
    SoilSolver3D s(1, 1, nz, 1.0, 1.0, dz, loam);
    for (uint k = 0; k < nz / 2; k++)
        s.setSoilProperties(0, 0, k, sand); // 3D k = 0 is the bottom
    s.setInitialCondition(-1.0);
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    SoilSolver3D::Params p = s.getParams();
    p.K_weighting = 4;
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
    s.setParams(p);
    for (int n = 0; n < 2000; n++)
        DOCTEST_REQUIRE(column.solve(3600.0f) > 0);
    for (int n = 0; n < 200; n++)
        DOCTEST_REQUIRE(s.solve(36000.0) > 0);
    double err = 0.0;
    for (uint k = 0; k < nz; k++)
        err = std::max(err, std::fabs((double) column.getPressureHead(k) - s.getPressureHead(0, 0, nz - 1 - k)));
    DOCTEST_CHECK(err < 1e-3);
}

DOCTEST_TEST_CASE("SoilColumn - radial geometry: steady saturated flow follows the logarithmic profile") {
    // Saturated (constant K) radial flow between two held heads: h = h_i + (h_o - h_i) ln(r/r_i) / ln(r_o/r_i),
    // with the flux through the inner surface K (h_i - h_o) / (r_i ln(r_o/r_i)) per unit inner area.
    const double ri = 1e-3, ro = 0.05, hi = 1.0, ho = 0.5;
    const uint N = 200;
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilColumn column(N, (float) ((ro - ri) / N), loam);
    column.setRadialGeometry((float) ri);
    column.setInitialCondition(0.75f);
    column.setTopBoundary(BCType::DIRICHLET, (float) hi);
    column.setBottomBoundary(BCType::DIRICHLET, (float) ho);
    for (int n = 0; n < 3; n++)
        DOCTEST_REQUIRE(column.solve(1e6f) > 0);
    double err = 0.0;
    for (uint k = 0; k < N; k++) {
        const double r = ri + (k + 0.5) * (ro - ri) / N;
        err = std::max(err, std::fabs(column.getPressureHead(k) - (hi + (ho - hi) * std::log(r / ri) / std::log(ro / ri))));
    }
    DOCTEST_CHECK(err < 2e-3); // cell-centred, second order; the error concentrates at the inner surface
    const double q_exact = loam.K_s * (hi - ho) / (ri * std::log(ro / ri));
    DOCTEST_CHECK(column.getTopFlux() == doctest::Approx(q_exact).epsilon(0.01));
}

DOCTEST_TEST_CASE("SoilColumn - radial geometry conserves water and ignores gravity") {
    const double ri = 2e-4, ro = 6e-3, q = 0.1 * 0.01 / 86400.0; // root uptake 0.1 cm/day at the root surface
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    // No fluxes: a uniform radial column stays uniform (gravity would drain a vertical one)
    SoilColumn still(40, (float) ((ro - ri) / 40), loam);
    still.setRadialGeometry((float) ri);
    still.setInitialCondition(-1.0f);
    still.setTopBoundary(BCType::NEUMANN, 0.0f);
    still.setBottomBoundary(BCType::NEUMANN, 0.0f);
    DOCTEST_REQUIRE(still.solve(86400.0f) > 0);
    for (uint k = 0; k < 40; k++)
        DOCTEST_CHECK(still.getPressureHead(k) == doctest::Approx(-1.0).epsilon(1e-6));
    // Uptake at the root surface for 5 days: storage (per unit root-surface area) falls by q t
    SoilColumn column(60, (float) ((ro - ri) / 60), loam);
    column.setRadialGeometry((float) ri);
    column.setInitialCondition(-1.0f);
    column.setTopBoundary(BCType::NEUMANN, (float) -q);
    column.setBottomBoundary(BCType::NEUMANN, 0.0f);
    const double s0 = column.getTotalWaterStorage();
    float dt = 3600.0f;
    const float t = column.solveAdaptive(5.0f * 86400.0f, dt);
    DOCTEST_REQUIRE(t == doctest::Approx(5.0 * 86400.0));
    DOCTEST_CHECK((s0 - column.getTotalWaterStorage()) == doctest::Approx(q * t).epsilon(1e-5));
    // Shell volumes: the storage of a uniform column is theta x (ro^2 - ri^2) / (2 ri)
    DOCTEST_CHECK(still.getTotalWaterStorage() == doctest::Approx(h_to_theta(-1.0f, loam) * (ro * ro - ri * ri) / (2.0 * ri)).epsilon(1e-5));
    DOCTEST_CHECK_THROWS(column.setRadialGeometry(0.0f));
}

DOCTEST_TEST_CASE("Soil models - linear soil") {
    // S_e = 1 + alpha h (clipped to [0, 1]), K = K_s S_e
    const SoilProperties lin = SoilProperties::linear(0.05f, 0.45f, 0.5f, 1e-6f);
    for (double h: {-3.0, -2.0, -1.5, -1.0, -0.4, 0.0, 0.3}) {
        const double Se = std::min(std::max(1.0 + 0.5 * h, 0.0), 1.0);
        double K, C;
        computeKandC<double>(h, lin, K, C);
        DOCTEST_CHECK(waterContent<double>(h, lin) == doctest::Approx(0.05 + 0.4 * Se).epsilon(1e-6));
        DOCTEST_CHECK(K == doctest::Approx(1e-6 * Se).epsilon(1e-6));
        if (Se > 0.0 && Se < 1.0) {
            DOCTEST_CHECK(C == doctest::Approx(0.4 * 0.5).epsilon(1e-6));
            DOCTEST_CHECK(pressureHeadOfTheta(0.05 + 0.4 * Se, lin) == doctest::Approx(h).epsilon(1e-6));
        }
    }
    // The 1D column and the 3D solver agree on a linear-soil infiltration
    const uint nz = 50;
    const double dz = 0.02, q = 5e-7;
    SoilColumn column(nz, (float) dz, lin);
    column.setInitialCondition(-1.5f);
    column.setTopBoundary(BCType::NEUMANN, (float) q);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    SoilSolver3D s(1, 1, nz, 1.0, 1.0, dz, lin);
    s.setInitialCondition(-1.5);
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    SoilSolver3D::Params p = s.getParams();
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
    p.bdf2 = false; // backward Euler, as the column
    s.setParams(p);
    for (int n = 0; n < 48; n++) {
        DOCTEST_REQUIRE(column.solve(1800.0f) > 0);
        DOCTEST_REQUIRE(s.solve(1800.0) > 0);
    }
    double err = 0.0;
    for (uint k = 0; k < nz; k++)
        err = std::max(err, std::fabs((double) column.getWaterContent(k) - s.getWaterContent(0, 0, nz - 1 - k)));
    DOCTEST_CHECK(err < 1e-6);
    DOCTEST_CHECK(column.getWaterContent(0) > h_to_theta(-1.5f, lin) + 0.01f); // water has entered
}

namespace {
    //! The 3D solver set up as a 1 x 1 x n vertical column with the given layer thicknesses (top layer first,
    //! as SoilColumn), on the CPU or the GPU, backward Euler (as SoilColumn).
    void setupColumn3D(SoilSolver3D &s, const std::vector<float> &dz_top_first, bool gpu) {
        const uint n = dz_top_first.size();
        std::vector<double> zc(n), thk(n);
        double z = 0.0;
        for (uint k = 0; k < n; k++) { // 3D k = 0 is the bottom
            thk[k] = dz_top_first[n - 1 - k];
            zc[k] = z + 0.5 * thk[k];
            z += thk[k];
        }
        bool uniform = true;
        for (float d: dz_top_first)
            uniform = uniform && d == dz_top_first[0];
        if (!uniform)
            s.setCellGeometry(zc, thk);
        SoilSolver3D::Params p = s.getParams();
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
        p.bdf2 = false;
        p.use_gpu = gpu;
        s.setParams(p);
    }
    std::vector<bool> solverBackends() {
#ifdef HELIOS_CUDA_AVAILABLE
        if (gpuTestsCanRun("The GPU variant of a CPU/GPU test")) {
            return {false, true};
        }
#endif
        return {false};
    }
} // namespace

DOCTEST_TEST_CASE("Column and 3D solvers - cells far thinner than the head change converge alike") {
    // DuMux's annulus grid laid vertically: 300 cells over 5.8 mm, growing downward by 1/0.95 (the top ones
    // ~1e-10 m), clay held at -150 m at the top, an upward flux from below. Across the thin cells the flux's
    // round-off exceeds 1e-9 of it, so Newton converges only with the round-off allowance in its residual test.
    const int N = 300;
    const double gr = 0.95, L = 5.8e-3, height = (1.0 - gr) / (1.0 - std::pow(gr, N));
    std::vector<float> dz(N);
    for (int i = 0; i < N; i++)
        dz[i] = (float) (L * height * std::pow(gr, N - i - 1));
    const SoilProperties clay(0.1f, 0.4f, 1.0f, 1.1f, 1.1574e-6f);
    const double q = 1.157e-8; // [m/s] upward through the base
    SoilColumn column(dz, clay);
    column.setInitialCondition(-100.0f);
    column.setTopBoundary(BCType::DIRICHLET, -150.0f);
    column.setBottomBoundary(BCType::NEUMANN, (float) -q); // positive out of the domain: -q is an inflow
    for (int n = 0; n < 4; n++)
        DOCTEST_REQUIRE(column.solve(1e9f) > 0);
    for (bool gpu: solverBackends()) {
        SoilSolver3D s(1, 1, N, 1.0, 1.0, L / N, clay);
        setupColumn3D(s, dz, gpu);
        s.setInitialCondition(-100.0);
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -150.0);
        s.setBoundary(Face3D::ZMIN, BCType::NEUMANN, q); // positive into the soil
        for (int n = 0; n < 4; n++)
            DOCTEST_REQUIRE(s.solve(1e9) > 0);
        double err = 0.0;
        for (int k = 0; k < N; k++)
            err = std::max(err, std::fabs((double) column.getPressureHead(k) - s.getPressureHead(0, 0, N - 1 - k)));
        DOCTEST_CAPTURE(gpu);
        DOCTEST_CHECK(err < 1e-4); // the column reports float heads (~1e-5 m at 150 m)
    }
}

DOCTEST_TEST_CASE("Column and 3D solvers - drying to the wilting point under a fixed demand") {
    // Evaporation demand from loam until the surface head reaches -150 m; steps of up to 6 h, a step that fails or
    // overshoots the wilting point by more than 0.1% is halved. Near the end the demand exceeds what the soil can
    // deliver within a step, so steps have no solution: they must fail quickly and leave the state unchanged, and
    // a fallback step must not create water. The column and the 3D solver (CPU, GPU) must agree.
    const uint nz = 60;
    const float dz = 0.005f;
    const double E = 5e-3 / 86400.0, h_wilt = -150.0; // 5 mm/day
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    struct Result {
        double t, balance;
    };
    auto drive = [&](auto solve, auto head0, auto save, auto restore, auto storage) {
        const double s0 = storage();
        double t = 0.0, dt = 21600.0;
        for (int tries = 0; tries < 2000; tries++) {
            save();
            if (!solve(dt) || head0() < 1.001 * h_wilt) {
                restore();
                dt *= 0.5;
                continue;
            }
            t += dt;
            if (head0() <= h_wilt)
                break;
            dt = std::min(2.0 * dt, 21600.0);
        }
        return Result{t, (s0 - storage() - E * t) / (E * t)};
    };
    SoilColumn column(nz, dz, loam);
    SolverParameters cp;
    cp.substep_on_failure = false; // this driver shortens failed steps itself
    column.setSolverParameters(cp);
    column.setInitialCondition(-1.0f);
    column.setTopBoundary(BCType::NEUMANN, (float) -E);
    column.setBottomBoundary(BCType::NEUMANN, 0.0f);
    std::vector<float> hc;
    const Result rc = drive([&](double dt) { return column.solve((float) dt) > 0; }, [&] { return (double) column.getPressureHead(0); }, [&] { hc = column.getPressureHeads(); },
                            [&] { column.setInitialCondition(hc); }, [&] { return (double) column.getTotalWaterStorage(); });
    DOCTEST_CHECK(column.getPressureHead(0) <= h_wilt);
    DOCTEST_CHECK(std::fabs(rc.balance) < 1e-5);
    // Past the wilting point a long step has no solution: without sub-stepping it fails and leaves the state as it was
    {
        const std::vector<float> before = column.getPressureHeads();
        DOCTEST_CHECK(column.solve(21600.0f) == 0);
        DOCTEST_CHECK(column.getPressureHeads() == before);
    }
    for (bool gpu: solverBackends()) {
        SoilSolver3D s(1, 1, nz, 1.0, 1.0, dz, loam);
        setupColumn3D(s, std::vector<float>(nz, dz), gpu);
        {
            SoilSolver3D::Params p = s.getParams();
            p.substep_on_failure = false;
            s.setParams(p);
        }
        s.setInitialCondition(-1.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, -E);
        s.setBoundary(Face3D::ZMIN, BCType::NEUMANN, 0.0);
        std::vector<double> h3;
        const auto t0 = std::chrono::steady_clock::now();
        const Result r3 = drive([&](double dt) { return s.solve(dt) > 0; }, [&] { return s.getPressureHead(0, 0, nz - 1); }, [&] { h3 = s.getPressureHeads(); }, [&] { s.setInitialCondition(h3); },
                                [&] { return s.getTotalWaterStorage(); });
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        DOCTEST_CAPTURE(gpu);
        DOCTEST_CAPTURE(rc.t);
        DOCTEST_CAPTURE(r3.t);
        DOCTEST_CAPTURE(wall);
        DOCTEST_CHECK(s.getPressureHead(0, 0, nz - 1) <= h_wilt);
        DOCTEST_CHECK(r3.t == doctest::Approx(rc.t).epsilon(1e-3));
        DOCTEST_CHECK(std::fabs(r3.balance) < 1e-5);
        DOCTEST_CHECK(wall < 10.0); // failed steps are cheap (it took minutes when they were not)
        const std::vector<double> before = s.getPressureHeads();
        DOCTEST_CHECK(s.solve(21600.0) == 0);
        DOCTEST_CHECK(s.getPressureHeads() == before);
    }
}

DOCTEST_TEST_CASE("SoilUtils - dK/dh matches the slope of K(h) next to saturation") {
    // For van Genuchten-Mualem with n < 2, dK/dh ~ |alpha h|^(n-2) grows without bound as h -> 0-. The Newton
    // solvers' Jacobian must still follow the K(h) of the residual there, down to heads a cell can settle at
    // (micrometres below saturation under drip irrigation), or Newton overshoots and converges only linearly.
    const SoilProperties loam(0.078f, 0.43f, 3.6f, 1.56f, 2.889e-6f), clay(0.1f, 0.4f, 1.0f, 1.1f, 1.1574e-6f);
    for (const SoilProperties &p: {loam, clay}) {
        for (double h: {-1e-2, -1e-3, -1e-4, -1e-5, -3e-6, -1e-7}) {
            const double dh = 1e-3 * std::fabs(h);
            double Kp, Km, C;
            computeKandC<double>(h + dh, p, Kp, C);
            computeKandC<double>(h - dh, p, Km, C);
            const double slope = (Kp - Km) / (2.0 * dh);
            DOCTEST_CAPTURE(p.n);
            DOCTEST_CAPTURE(h);
            DOCTEST_CHECK(conductivityDerivative(h, p) == doctest::Approx(slope).epsilon(1e-3));
        }
    }
}

DOCTEST_TEST_CASE("Column and 3D solvers - Newton converges micrometres below saturation") {
    // Steady downward flow through loam (n = 1.56) at the conductivity of h* = -3 um: the column sits a few
    // micrometres below saturation, where dK/dh is large and changes fast. Starting slightly off h*, the first step
    // settles the column and every later step must converge in a few Newton iterations on every solver (with a
    // Jacobian that misses dK/dh there, full Newton steps overshoot to the mirror image and the residual falls
    // only ~1% per iteration: 12-23 iterations a step).
    const uint nz = 50;
    const float dz = 0.02f;
    const SoilProperties loam(0.078f, 0.43f, 3.6f, 1.56f, 2.889e-6f);
    const double hstar = -3e-6, h0 = -2e-5, dt = 900.0;
    double q, C;
    computeKandC<double>(hstar, loam, q, C);
    auto max_it = [](int n) { return n == 0 ? 10u : 3u; }; // the first step starts away from h*
    SoilColumn column(nz, dz, loam);
    column.setInitialCondition((float) h0);
    column.setTopBoundary(BCType::NEUMANN, (float) q); // positive into the soil at the top
    column.setBottomBoundary(BCType::DIRICHLET, (float) hstar);
    for (int n = 0; n < 6; n++) {
        const uint it = column.solve((float) dt);
        DOCTEST_CAPTURE(n);
        DOCTEST_CHECK(it > 0);
        DOCTEST_CHECK(it <= max_it(n));
    }
    for (bool gpu: solverBackends()) {
        SoilSolver3D s(1, 1, nz, 1.0, 1.0, dz, loam);
        setupColumn3D(s, std::vector<float>(nz, dz), gpu);
        s.setInitialCondition(h0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
        s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, hstar);
        for (int n = 0; n < 6; n++) {
            const uint it = s.solve(dt);
            DOCTEST_CAPTURE(gpu);
            DOCTEST_CAPTURE(n);
            DOCTEST_CHECK(it > 0);
            DOCTEST_CHECK(it <= max_it(n));
        }
        double err = 0.0;
        for (uint k = 0; k < nz; k++)
            err = std::max(err, std::fabs((double) column.getPressureHead(k) - s.getPressureHead(0, 0, nz - 1 - k)));
        DOCTEST_CHECK(err < 1e-8);
    }
}

DOCTEST_TEST_CASE("SoilColumn - soil property changes between Newton steps take effect") {
    // The Newton solver carries the end-of-step constitutive values into the next step. Changing the soil
    // properties must discard them: the run must continue exactly like a fresh column with the new soil
    // started from the same heads.
    const uint nz = 40;
    const float dz = 0.025f;
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilProperties sand = getSoilPropertiesFromTexture("sand");
    auto setup = [&](SoilColumn &c) {
        c.setTopBoundary(BCType::NEUMANN, 1e-7f);
        c.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    };
    SoilColumn a(nz, dz, loam);
    a.setInitialCondition(-0.5f);
    setup(a);
    for (int n = 0; n < 20; n++)
        DOCTEST_REQUIRE(a.solve(600.0f) > 0);
    std::vector<float> h_switch(nz);
    for (uint k = 0; k < nz; k++)
        h_switch[k] = a.getPressureHead(k);
    a.setLayerSoilProperties(0, nz, sand);

    SoilColumn b(nz, dz, sand);
    b.setInitialCondition(h_switch);
    setup(b);
    for (int n = 0; n < 5; n++) {
        DOCTEST_REQUIRE(a.solve(600.0f) > 0);
        DOCTEST_REQUIRE(b.solve(600.0f) > 0);
    }
    double err = 0.0;
    for (uint k = 0; k < nz; k++) {
        err = std::max(err, std::fabs((double) a.getPressureHead(k) - b.getPressureHead(k)) / std::max(1.0, std::fabs((double) b.getPressureHead(k))));
        // Reported water content and conductivity belong to the reported head
        DOCTEST_CHECK(a.getWaterContent(k) == doctest::Approx(h_to_theta(a.getPressureHead(k), sand)).epsilon(1e-5));
        DOCTEST_CHECK(a.getConductivity(k) == doctest::Approx(computeK(a.getPressureHead(k), sand)).epsilon(1e-4));
    }
    DOCTEST_CHECK(err < 1e-4);
}

DOCTEST_TEST_CASE("SoilSolver3D - thread count") {
    DOCTEST_CHECK(resolveThreadCount(3) == 3);
    const int automatic = resolveThreadCount(0);
    DOCTEST_CHECK(automatic >= 1);
    DOCTEST_CHECK(physicalCoreCount() >= 0);
    DOCTEST_CHECK(physicalCoreCount() <= (int) std::max(1u, std::thread::hardware_concurrency()));

    // Large enough for multi-block ILU (one block per thread): the block count changes the preconditioner,
    // never the converged answer.
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    auto run = [&](int threads) {
        SoilSolver3D s(24, 24, 16, 0.05, 0.05, 0.05, loam);
        s.setInitialCondition(-1.0);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s.setSource(12, 12, 15, 2e-5);
        SoilSolver3D::Params p = s.getParams();
        p.num_threads = threads;
        s.setParams(p);
        for (int n = 0; n < 6; n++)
            DOCTEST_REQUIRE(s.solve(600.0) > 0);
        return s.getPressureHeads();
    };
    const std::vector<double> h1 = run(1), h4 = run(4), h0 = run(0);
    double d4 = 0.0, d0 = 0.0;
    for (size_t c = 0; c < h1.size(); c++) {
        d4 = std::max(d4, std::fabs(h4[c] - h1[c]));
        d0 = std::max(d0, std::fabs(h0[c] - h1[c]));
    }
    DOCTEST_CHECK(d4 < 1e-5);
    DOCTEST_CHECK(d0 < 1e-5);
}

DOCTEST_TEST_CASE("SoilUtils - modified (air-entry) van Genuchten constitutive functions") {
    const SoilProperties vg = getSoilPropertiesFromTexture("loam"); // n = 1.56 < 2
    const SoilProperties mv = SoilProperties::modifiedVanGenuchten(vg.theta_r, vg.theta_s, vg.alpha, vg.n, vg.K_s, -0.02f);
    const double he = mv.h_e;
    DOCTEST_CHECK(saturationHead(mv) == doctest::Approx(he));
    DOCTEST_CHECK(saturationHead(vg) == 0.0);

    // Saturated at and above the air-entry head; continuous theta and K at h_e
    double th, K, C, dK;
    for (double h: {he, 0.5 * he, 0.0, 0.3}) {
        constitutiveAll(h, mv, 1e-12, 0.0, th, K, C, dK);
        DOCTEST_CHECK(th == doctest::Approx(mv.theta_s));
        DOCTEST_CHECK(K == doctest::Approx(mv.K_s));
        DOCTEST_CHECK(dK == 0.0);
    }
    constitutiveAll(he - 1e-9, mv, 1e-12, 0.0, th, K, C, dK);
    DOCTEST_CHECK(th == doctest::Approx(mv.theta_s).epsilon(1e-6));
    DOCTEST_CHECK(K == doctest::Approx(mv.K_s).epsilon(1e-5));

    // Below h_e: Se = Se_vG / Sc (equivalently theta rescaled), and the double/float/templated paths agree
    for (double h: {-0.03, -0.1, -0.5, -2.0, -20.0}) {
        constitutiveAll(h, mv, 1e-12, 0.0, th, K, C, dK);
        double Sc, Fk;
        airEntryScales(mv, Sc, Fk);
        const double Se_vg = (waterContent<double>(h, vg) - vg.theta_r) / (vg.theta_s - vg.theta_r);
        DOCTEST_CHECK(th == doctest::Approx(vg.theta_r + (vg.theta_s - vg.theta_r) * Se_vg / Sc).epsilon(1e-12));
        DOCTEST_CHECK(th == doctest::Approx(waterContent<double>(h, mv)).epsilon(1e-12));
        DOCTEST_CHECK(th == doctest::Approx(h_to_theta((float) h, mv)).epsilon(1e-5));
        DOCTEST_CHECK(K == doctest::Approx(computeK((float) h, mv)).epsilon(1e-4));
        DOCTEST_CHECK(C == doctest::Approx(computeC((float) h, mv)).epsilon(1e-4));
        double K2, C2;
        computeKandC<double>(h, mv, K2, C2);
        DOCTEST_CHECK(K2 == doctest::Approx(K).epsilon(1e-12));
        DOCTEST_CHECK(C2 == doctest::Approx(C).epsilon(1e-12));
        DOCTEST_CHECK(conductivityDerivative(h, mv) == doctest::Approx(dK).epsilon(1e-10));
        // derivatives against central differences
        const double eps = 1e-6 * std::fabs(h);
        double thp, Kp, Cp, dKp, thm, Km, Cm, dKm;
        constitutiveAll(h + eps, mv, 1e-12, 0.0, thp, Kp, Cp, dKp);
        constitutiveAll(h - eps, mv, 1e-12, 0.0, thm, Km, Cm, dKm);
        DOCTEST_CHECK(C == doctest::Approx((thp - thm) / (2 * eps)).epsilon(1e-5));
        DOCTEST_CHECK(dK == doctest::Approx((Kp - Km) / (2 * eps)).epsilon(1e-5));
        // inverse retention curve round trip (double and float)
        DOCTEST_CHECK(pressureHeadOfTheta(th, mv) == doctest::Approx(h).epsilon(1e-9));
        DOCTEST_CHECK(theta_to_h((float) th, mv) == doctest::Approx(h).epsilon(1e-3));
    }

    // The point of the model: for n < 2 the standard dK/dh grows without bound toward saturation, the
    // modified one stays bounded up to the air-entry head
    double dK_mod_max = 0.0;
    for (double h = he - 1e-6; h > he - 0.05; h -= 1e-4)
        dK_mod_max = std::max(dK_mod_max, conductivityDerivative(h, mv));
    double th_v, K_v, C_v, dK_far, dK_near;
    constitutiveAll(-1e-2, vg, 1e-12, 0.0, th_v, K_v, C_v, dK_far);
    constitutiveAll(-1e-3, vg, 1e-12, 0.0, th_v, K_v, C_v, dK_near);
    DOCTEST_CHECK(dK_near > 2.0 * dK_far); // standard model: steepening toward h = 0
    DOCTEST_CHECK(std::isfinite(dK_mod_max));
    DOCTEST_CHECK(dK_mod_max < 1e3 * mv.K_s); // bounded (finite, O(K_s / |h_e|))

    // With h_e -> 0 the modified model reduces to the standard one
    const SoilProperties near0 = SoilProperties::modifiedVanGenuchten(vg.theta_r, vg.theta_s, vg.alpha, vg.n, vg.K_s, -1e-7f);
    for (double h: {-0.1, -1.0})
        DOCTEST_CHECK(computeK((float) h, near0) == doctest::Approx(computeK((float) h, vg)).epsilon(1e-3));
    DOCTEST_CHECK(inflectionHead(mv) <= he);
}

DOCTEST_TEST_CASE("SoilUtils - relative-conductivity spline near saturation") {
    SoilProperties vg = getSoilPropertiesFromTexture("clay"); // n = 1.09
    SoilProperties sp = vg;
    sp.kr_spline_start = 0.9999f;
    const double S1 = sp.kr_spline_start, m = 1.0 - 1.0 / vg.n;
    const double h1 = -std::pow(std::pow(S1, -1.0 / m) - 1.0, 1.0 / vg.n) / vg.alpha; // head at S_e = S1
    // Off by default; identical to the standard model below the spline start
    double th_v, K_v, C_v, dK_v, th_s, K_s, C_s, dK_s;
    for (double h: {2.0 * h1, -0.1, -1.0, -10.0}) {
        constitutiveAll(h, vg, 1e-12, 0.0, th_v, K_v, C_v, dK_v);
        constitutiveAll(h, sp, 1e-12, 0.0, th_s, K_s, C_s, dK_s);
        DOCTEST_CHECK(K_s == doctest::Approx(K_v).epsilon(1e-12));
        DOCTEST_CHECK(dK_s == doctest::Approx(dK_v).epsilon(1e-12));
        DOCTEST_CHECK(th_s == th_v);
    }
    // K and dK/dh continuous at the spline start
    double Ka, Kb, dKa, dKb, th, C;
    constitutiveAll(h1 * (1.0 + 1e-7), sp, 1e-12, 0.0, th, Ka, C, dKa); // just drier than S1
    constitutiveAll(h1 * (1.0 - 1e-7), sp, 1e-12, 0.0, th, Kb, C, dKb); // just wetter
    DOCTEST_CHECK(Kb == doctest::Approx(Ka).epsilon(1e-5));
    DOCTEST_CHECK(dKb == doctest::Approx(dKa).epsilon(1e-3));
    // K reaches K_s at saturation, monotone, with a bounded slope; derivatives match finite differences
    double Kprev = 0.0, dKmax = 0.0;
    for (int i = 1; i <= 200; i++) {
        const double h = h1 * (1.0 - i / 200.5);
        constitutiveAll(h, sp, 1e-12, 0.0, th, K_s, C, dK_s);
        DOCTEST_CHECK(K_s >= Kprev);
        Kprev = K_s;
        dKmax = std::max(dKmax, dK_s);
        if (i % 40 == 0) {
            const double e = 1e-6 * std::fabs(h);
            double Kp, Km, d1, d2;
            constitutiveAll(h + e, sp, 1e-12, 0.0, th, Kp, C, d1);
            constitutiveAll(h - e, sp, 1e-12, 0.0, th, Km, C, d2);
            DOCTEST_CHECK(dK_s == doctest::Approx((Kp - Km) / (2 * e)).epsilon(1e-4));
            DOCTEST_CHECK(conductivityDerivative(h, sp) == doctest::Approx(dK_s).epsilon(1e-6));
            DOCTEST_CHECK(computeK((float) h, sp) == doctest::Approx(K_s).epsilon(1e-4));
        }
    }
    DOCTEST_CHECK(Kprev == doctest::Approx(sp.K_s).epsilon(1e-3));
    DOCTEST_CHECK(std::isfinite(dKmax));
    constitutiveAll(-1e-9, sp, 1e-12, 0.0, th, K_s, C, dK_s);
    DOCTEST_CHECK(dK_s < 1e3 * sp.K_s / std::fabs(h1)); // bounded at saturation
}

DOCTEST_TEST_CASE("SoilSolver3D - relative-conductivity spline: ponded clay matches the 1D column") {
    // Ponded infiltration into clay (n = 1.09) with the near-saturation spline: both solvers converge every step
    // and agree.
    const uint nz = 40;
    const double dz = 0.025, dt = 120.0; // short enough that neither solver sub-steps, so both integrate alike
    SoilProperties clay = getSoilPropertiesFromTexture("clay");
    clay.kr_spline_start = 0.9999f;
    SoilColumn column(nz, (float) dz, clay);
    column.setInitialCondition(-1.0f);
    column.setTopBoundary(BCType::DIRICHLET, 0.0f);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    SoilSolver3D s(1, 1, nz, 1.0, 1.0, dz, clay);
    s.setInitialCondition(-1.0);
    s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, 0.0);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    SoilSolver3D::Params p = s.getParams();
    p.bdf2 = false;
    s.setParams(p);
    for (int n = 0; n < 60; n++) {
        DOCTEST_REQUIRE(column.solve((float) dt) > 0);
        DOCTEST_REQUIRE(s.solve(dt) > 0);
    }
    double err = 0.0; // water content: heads at a front this sharp swing by mm for tiny shifts
    for (uint k = 0; k < nz; k++)
        err = std::max(err, std::fabs((double) column.getWaterContent(k) - s.getWaterContent(0, 0, nz - 1 - k)));
    MESSAGE("max |theta_column - theta_3D| = " << err);
    DOCTEST_CHECK(err < 1e-4);
}

DOCTEST_TEST_CASE("SoilUtils - modified van Genuchten texture library") {
    for (const std::string &t: getAvailableSoilTextures()) {
        const SoilProperties vg = getSoilPropertiesFromTexture(t);
        const SoilProperties mv = getSoilPropertiesFromTexture(t, SoilModel::MODIFIED_VAN_GENUCHTEN);
        DOCTEST_CHECK(mv.model == SoilModel::MODIFIED_VAN_GENUCHTEN);
        DOCTEST_CHECK(mv.h_e == doctest::Approx(-0.02f));
        DOCTEST_CHECK(mv.theta_r == vg.theta_r);
        DOCTEST_CHECK(mv.alpha == vg.alpha);
        DOCTEST_CHECK(mv.n == vg.n);
        // default: the Carsel & Parrish curves truncated at h_e -> saturated values are the standard ones at h_e
        DOCTEST_CHECK(mv.theta_s == doctest::Approx(h_to_theta(mv.h_e, vg)).epsilon(1e-6));
        DOCTEST_CHECK(mv.K_s == doctest::Approx(computeK(mv.h_e, vg)).epsilon(1e-5));
        // HYDRUS convention: the saturated values are kept
        const SoilProperties hy = getSoilPropertiesFromTexture(t, SoilModel::MODIFIED_VAN_GENUCHTEN, false);
        DOCTEST_CHECK(hy.theta_s == vg.theta_s);
        DOCTEST_CHECK(hy.K_s == vg.K_s);
        DOCTEST_CHECK(getSoilPropertiesFromTexture(t, SoilModel::VAN_GENUCHTEN).model == SoilModel::VAN_GENUCHTEN);
    }
    DOCTEST_CHECK_THROWS(getSoilPropertiesFromTexture("loam", SoilModel::GARDNER));
}

DOCTEST_TEST_CASE("SoilUtils - conversion to the modified van Genuchten model truncates the standard curves") {
    for (const char *t: {"sand", "loam", "clay"}) {
        const SoilProperties vg = getSoilPropertiesFromTexture(t);
        const SoilProperties mv = toModifiedVanGenuchten(vg, -0.02f);
        for (double h: {-0.0201, -0.05, -0.3, -1.0, -10.0, -150.0}) {
            double th_v, K_v, C_v, dK_v, th_m, K_m, C_m, dK_m;
            constitutiveAll(h, vg, 1e-12, 0.0, th_v, K_v, C_v, dK_v);
            constitutiveAll(h, mv, 1e-12, 0.0, th_m, K_m, C_m, dK_m);
            DOCTEST_CHECK(th_m == doctest::Approx(th_v).epsilon(1e-6));
            DOCTEST_CHECK(K_m == doctest::Approx(K_v).epsilon(1e-5));
            DOCTEST_CHECK(C_m == doctest::Approx(C_v).epsilon(1e-5));
            DOCTEST_CHECK(dK_m == doctest::Approx(dK_v).epsilon(1e-5));
        }
        double th, K, C, dK; // above h_e: saturated at the standard values at h_e
        constitutiveAll(-0.01, mv, 1e-12, 0.0, th, K, C, dK);
        DOCTEST_CHECK(th == doctest::Approx(waterContent<double>(-0.02, vg)).epsilon(1e-6));
        DOCTEST_CHECK(K == doctest::Approx(computeK(-0.02f, vg)).epsilon(1e-5));
    }
    DOCTEST_CHECK_THROWS(toModifiedVanGenuchten(getSoilPropertiesFromTexture("loam"), 0.0f));
    DOCTEST_CHECK_THROWS(toModifiedVanGenuchten(SoilProperties::gardner(0.05f, 0.4f, 2.0f, 1e-5f)));
}

DOCTEST_TEST_CASE("SoilSolver3D - modified van Genuchten: saturating infiltration matches the 1D column and conserves water") {
    // Infiltration at 3 x K_s into dry loam (n = 1.56) forces the top to saturate. With the air-entry model
    // both solvers converge every step, agree, and conserve water.
    const uint nz = 40;
    const double dz = 0.025, dt = 300.0;
    const SoilProperties mv = getSoilPropertiesFromTexture("loam", SoilModel::MODIFIED_VAN_GENUCHTEN);
    const double q = 3.0 * mv.K_s;
    SoilColumn column(nz, (float) dz, mv);
    column.setInitialCondition(-1.0f);
    column.setTopBoundary(BCType::NEUMANN, (float) q);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    SoilSolver3D s(1, 1, nz, 1.0, 1.0, dz, mv);
    s.setInitialCondition(-1.0);
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    SoilSolver3D::Params p = s.getParams();
    p.bdf2 = false;
    s.setParams(p);
    s.resetMassBalance();
    const double W0 = s.getTotalWaterStorage();
    for (int n = 0; n < 72; n++) { // 6 h
        DOCTEST_REQUIRE(column.solve((float) dt) > 0);
        DOCTEST_REQUIRE(s.solve(dt) > 0);
    }
    DOCTEST_CHECK(s.getPressureHead(0, 0, nz - 1) >= saturationHead(mv)); // the top did saturate
    double err = 0.0;
    for (uint k = 0; k < nz; k++)
        err = std::max(err, std::fabs((double) column.getPressureHead(k) - s.getPressureHead(0, 0, nz - 1 - k)));
    DOCTEST_CHECK(err < 1e-3);
    DOCTEST_CHECK(std::fabs(s.getTotalWaterStorage() - W0 - s.getCumulativeInflux()) < 1e-6 * std::fabs(s.getCumulativeInflux()));
}

DOCTEST_TEST_CASE("SoilUtils - Kirchhoff transform + b/B splitting (M-scheme foundation)") {
    // The KirchhoffTable underpins the faithful (paper-true) M-scheme. Validate the splitting invariants
    // and, for Gardner, the closed-form transform  w = (Ks/alpha) e^{alpha psi}  (=> w_sat = Ks/alpha and
    // b' = B' = 1/2 everywhere, since the scaled theta- and w-slopes are equal for Gardner).
    auto checkInvariants = [&](const KirchhoffTable &t) {
        const int N = (int) t.s_.size();
        for (int k = 1; k < N; k++) {
            DOCTEST_CHECK(t.s_[k] > t.s_[k - 1]); // s strictly increasing
            DOCTEST_CHECK(t.theta_[k] >= t.theta_[k - 1] - 1e-12); // theta non-decreasing
            DOCTEST_CHECK(t.wt_[k] >= t.wt_[k - 1] - 1e-12); // w_tilde non-decreasing
        }
        for (int k = 0; k < N; k++) {
            DOCTEST_CHECK(t.bp_[k] >= -1e-9);
            DOCTEST_CHECK(t.bp_[k] <= 1.0 + 1e-9);
            DOCTEST_CHECK(t.Bp_[k] >= -1e-9);
            DOCTEST_CHECK(t.Bp_[k] <= 1.0 + 1e-9);
            DOCTEST_CHECK(std::fabs(t.bp_[k] + t.Bp_[k] - 1.0) < 1e-9); // b' + B' = 1
        }
    };

    // van Genuchten loam.
    KirchhoffTable kl;
    kl.build(getSoilPropertiesFromTexture("loam"));
    DOCTEST_REQUIRE(kl.built);
    checkInvariants(kl);
    // Round-trips through the inverse maps.
    for (double frac: {0.1, 0.3, 0.6, 0.9}) {
        double s = kl.s_[0] + frac * (kl.s_sat - kl.s_[0]);
        DOCTEST_CHECK(std::fabs(kl.b(kl.sOfTheta(kl.b(s))) - kl.b(s)) < 1e-6);
        DOCTEST_CHECK(std::fabs(kl.B(kl.sOfWtilde(kl.B(s))) - kl.B(s)) < 1e-6);
    }
    DOCTEST_CHECK(std::fabs(kl.psiOf(kl.s_sat)) < 1e-6); // s_sat <-> psi = 0

    // Gardner closed form.
    SoilProperties gard;
    gard.model = SoilModel::GARDNER;
    gard.theta_r = 0.05f;
    gard.theta_s = 0.45f;
    gard.alpha = 2.0f;
    gard.K_s = 1e-5f;
    gard.n = 2.0f;
    KirchhoffTable kg;
    kg.build(gard);
    checkInvariants(kg);
    DOCTEST_CHECK(std::fabs(kg.w_sat - gard.K_s / gard.alpha) / (gard.K_s / gard.alpha) < 3e-3); // w_sat = Ks/alpha (trapezoid quadrature)
    // Gardner: dtheta/dpsi = alpha*(ts-tr)*e, dw_tilde/dpsi = K/w_sat = alpha*e, so b' = (ts-tr)/((ts-tr)+1) const.
    const double bp_gard = (gard.theta_s - gard.theta_r) / ((gard.theta_s - gard.theta_r) + 1.0);
    for (int k = 1; k < (int) kg.bp_.size() - 1; k++)
        DOCTEST_CHECK(std::fabs(kg.bp_[k] - bp_gard) < 3e-3); // b' constant (Gardner; ~0.1% trapezoid error in w_sat)
}

DOCTEST_TEST_CASE("SoilSolver3D - NGS (lagged-K nonlinear Gauss-Seidel) correctness") {
    // The NGS scheme (Gu & Wang 2026) lags K to the start-of-step head, so each cell is a scalar
    // monotone Newton solve. Validate: (1) it conserves mass in a closed box, and (2) it reaches
    // the exact Gardner steady profile (at steady state lagged-K == current-K, so it must agree).
    // No multigrid yet, so it converges slowly -> small grids + generous sweep budget.

    { // (1) mass conservation, closed (zero-flux) box
        SoilSolver3D m(5, 5, 6, 0.02, 0.02, 0.02, getSoilPropertiesFromTexture("loam"));
        auto p = m.getParams();
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NGS_FAS;
        p.K_weighting = 3; // NGS's lagged operator is built with harmonic-mean faces
        p.picard_max = 4000; // NGS-only is slow without FAS
        m.setParams(p);
        std::vector<double> h0(m.numCells());
        for (uint k = 0; k < 6; k++)
            for (uint j = 0; j < 5; j++)
                for (uint i = 0; i < 5; i++)
                    h0[m.index(i, j, k)] = (k >= 3) ? -0.3 : -1.2;
        m.setInitialCondition(h0);
        double s0 = m.getTotalWaterStorage();
        for (int n = 0; n < 40; n++)
            DOCTEST_REQUIRE(m.solve(60.0) > 0);
        DOCTEST_CHECK(std::fabs(m.getTotalWaterStorage() - s0) / s0 < 1e-4);
    }

    { // (2) Gardner steady infiltration -> exact profile
        uint Nz = 20;
        double dz = 0.01;
        SoilProperties g = SoilProperties::gardner(0.06f, 0.40f, 2.0f, 1e-5f);
        double Ks = g.K_s, al = g.alpha, q = 0.5 * Ks;
        SoilSolver3D s(1, 1, Nz, 0.01, 0.01, dz, g);
        auto p = s.getParams();
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NGS_FAS;
        p.picard_max = 6000;
        s.setParams(p);
        std::vector<double> h0(Nz);
        for (uint k = 0; k < Nz; k++)
            h0[k] = -((k + 0.5) * dz);
        s.setInitialCondition(h0);
        s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, 0.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
        std::vector<double> hp;
        for (int n = 0; n < 4000; n++) {
            hp = s.getPressureHeads();
            DOCTEST_REQUIRE(s.solve(60.0) > 0);
            double dm = 0.0;
            for (uint k = 0; k < Nz; k++)
                dm = std::max(dm, std::fabs(s.getPressureHead(0, 0, k) - hp[k]));
            if (dm < 1e-10 && n > 20)
                break;
        }
        auto h_exact = [&](double z) { return std::log(((Ks - q) * std::exp(-al * z) + q) / Ks) / al; };
        double err = 0.0;
        for (uint k = 0; k < Nz; k++)
            err = std::max(err, std::fabs(s.getPressureHead(0, 0, k) - h_exact((k + 0.5) * dz)));
        DOCTEST_CHECK(err < 1e-3); // lagged-K reaches the same steady state as fully implicit
    }

    { // (3) HETEROGENEOUS (two-layer) box: NGS-FAS must converge + conserve mass with per-cell soil.
        const uint Nx = 5, Ny = 5, Nz = 8;
        SoilSolver3D s(Nx, Ny, Nz, 0.02, 0.02, 0.02, getSoilPropertiesFromTexture("loam"));
        SoilProperties loam = getSoilPropertiesFromTexture("loam"), clay = getSoilPropertiesFromTexture("clay");
        for (uint k = 0; k < Nz; k++)
            for (uint j = 0; j < Ny; j++)
                for (uint i = 0; i < Nx; i++)
                    s.setSoilProperties(i, j, k, (k >= Nz / 2) ? loam : clay); // clay below, loam above
        auto p = s.getParams();
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NGS_FAS;
        p.picard_max = 4000;
        s.setParams(p);
        std::vector<double> h0(s.numCells(), -0.5);
        s.setInitialCondition(h0);
        double s0 = s.getTotalWaterStorage();
        for (int n = 0; n < 30; n++)
            DOCTEST_REQUIRE(s.solve(60.0) > 0); // must converge each step
        DOCTEST_CHECK(std::fabs(s.getTotalWaterStorage() - s0) / s0 < 1e-3); // conserves (closed box)
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - mass conservation (closed box)") {
    // Zero-flux on all six faces: total storage must be conserved.
    SoilSolver3D m(5, 5, 8, 0.02, 0.02, 0.02, getSoilPropertiesFromTexture("loam"));
    std::vector<double> h0(m.numCells());
    for (uint k = 0; k < 8; k++)
        for (uint j = 0; j < 5; j++)
            for (uint i = 0; i < 5; i++)
                h0[m.index(i, j, k)] = (k >= 4) ? -0.3 : -1.5;
    m.setInitialCondition(h0);
    double s0 = m.getTotalWaterStorage();
    for (int n = 0; n < 100; n++)
        DOCTEST_REQUIRE(m.solve(60.0) > 0);
    double rel = std::fabs(m.getTotalWaterStorage() - s0) / s0;
    DOCTEST_CHECK(rel < 1e-4);
}

DOCTEST_TEST_CASE("SoilSolver3D - lateral x/y isotropy") {
    // A centred wet blob must diffuse symmetrically under x<->y swap (dx == dy).
    SoilSolver3D s(7, 7, 6, 0.02, 0.02, 0.02, getSoilPropertiesFromTexture("sandy_loam"));
    std::vector<double> h0(s.numCells(), -1.5);
    h0[s.index(3, 3, 3)] = -0.2;
    s.setInitialCondition(h0);
    for (int n = 0; n < 30; n++)
        DOCTEST_REQUIRE(s.solve(30.0) > 0);
    double asym = 0.0;
    for (uint k = 0; k < 6; k++)
        for (uint j = 0; j < 7; j++)
            for (uint i = 0; i < 7; i++)
                asym = std::max(asym, std::fabs(s.getPressureHead(i, j, k) - s.getPressureHead(j, i, k)));
    DOCTEST_CHECK(asym < 1e-5);
}

DOCTEST_TEST_CASE("SoilSolver3D - BDF2 + adaptive timestepping") {
    // Adaptive BDF2 must agree with fixed backward Euler on a transient, in far fewer steps.
    SoilProperties sl = getSoilPropertiesFromTexture("sandy_loam");
    double q = 0.3 * sl.K_s, T = 3600.0;
    uint Nz = 40;
    double dz = 0.025;
    auto setup = [&](SoilSolver3D &s) {
        s.setInitialCondition(-3.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE);
    };

    SoilSolver3D be(1, 1, Nz, 0.025, 0.025, dz, sl);
    {
        auto p = be.getParams();
        p.bdf2 = false;
        be.setParams(p);
    }
    setup(be);
    int steps_be = 0;
    for (double t = 0; t < T - 1e-9; t += 5.0) {
        DOCTEST_REQUIRE(be.solve(std::min(5.0, T - t)) > 0);
        steps_be++;
    }

    SoilSolver3D ad(1, 1, Nz, 0.025, 0.025, dz, sl);
    {
        auto p = ad.getParams();
        p.bdf2 = true;
        p.dt_max = 600.0;
        ad.setParams(p);
    }
    setup(ad);
    double dt = 5.0;
    uint steps_ad = 0;
    ad.solveAdaptive(T, dt, &steps_ad);

    double maxdiff = 0.0;
    for (uint k = 0; k < Nz; k++)
        maxdiff = std::max(maxdiff, std::fabs(be.getPressureHead(0, 0, k) - ad.getPressureHead(0, 0, k)));
    DOCTEST_CHECK(maxdiff < 1e-2); // agrees with backward Euler (to the integrators' temporal error at the front)
    DOCTEST_CHECK(steps_ad < (uint) steps_be / 3); // adaptive uses far fewer steps
}

DOCTEST_TEST_CASE("SoilSolver3D - multigrid matches Jacobi, fewer iterations") {
    // Geometric-multigrid and Jacobi preconditioning must give the same solution;
    // multigrid should take substantially fewer CG iterations on a 3D grid.
    SoilProperties g = SoilProperties::gardner(0.06f, 0.40f, 2.0f, 1e-5f);
    int n = 16;
    double d = 0.5 / n;
    auto build = [&](bool mg, std::vector<double> &out_h, uint &cg) {
        SoilSolver3D s(n, n, n, d, d, d, g);
        std::vector<double> h0(s.numCells());
        for (uint k = 0; k < (uint) n; k++)
            for (uint j = 0; j < (uint) n; j++)
                for (uint i = 0; i < (uint) n; i++)
                    h0[s.index(i, j, k)] = -0.1 - 0.1 * (double) k / n;
        s.setInitialCondition(h0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 0.5 * g.K_s);
        s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, 0.0);
        auto p = s.getParams();
        p.use_multigrid = mg;
        p.bdf2 = false;
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::PICARD; // the CG preconditioner is a Picard-path feature
        s.setParams(p);
        uint tot = 0, step = 0;
        for (int t = 0; t < 3; t++) {
            DOCTEST_REQUIRE(s.solve(600.0, &step) > 0);
            tot += step;
        }
        cg = tot;
        out_h = s.getPressureHeads();
    };
    std::vector<double> h_jac, h_mg;
    uint cg_jac = 0, cg_mg = 0;
    build(false, h_jac, cg_jac);
    build(true, h_mg, cg_mg);
    double maxdiff = 0.0;
    for (size_t c = 0; c < h_jac.size(); c++)
        maxdiff = std::max(maxdiff, std::fabs(h_jac[c] - h_mg[c]));
    DOCTEST_CHECK(maxdiff < 1e-6); // same solution
    DOCTEST_CHECK(cg_mg < cg_jac / 2); // multigrid uses far fewer CG iterations
}

#ifdef HELIOS_CUDA_AVAILABLE
DOCTEST_TEST_CASE("SoilSolver3D - GPU analytic Newton matches CPU, including per-cell geometry") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU analytic Newton matches CPU, including per-cell geometry'"))
        return;
    for (bool geom: {false, true}) {
        auto runit = [&](bool gpu) {
            SoilProperties g = getSoilPropertiesFromTexture("sandy_loam");
            SoilSolver3D s(16, 12, 10, 0.5, 0.5, 0.1, g);
            std::vector<double> zc(s.numCells()), thk(s.numCells(), 0.1), h0(s.numCells());
            for (uint c = 0; c < s.numCells(); c++) {
                const uint k = c / (16 * 12), i = c % 16;
                zc[c] = (geom ? 0.02 * i : 0.0) + (k + 0.5) * 0.1;
                h0[c] = (geom ? 0.32 : 0.0) + 0.3 - zc[c];
            }
            if (geom) {
                s.setCellGeometry(zc, thk);
                s.setFixedHead(15, 6, 0, 0.2 - zc[s.index(15, 6, 0)]);
            }
            s.setInitialCondition(h0);
            s.setSource(3, 3, 9, 2e-5);
            auto p = s.getParams();
            p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
            p.K_weighting = 4;
            p.use_gpu = gpu;
            s.setParams(p);
            for (int n = 0; n < 12; n++)
                DOCTEST_REQUIRE(s.solve(600.0) > 0);
            return s.getPressureHeads();
        };
        const std::vector<double> hc = runit(false), hg = runit(true);
        double d = 0.0;
        for (size_t c = 0; c < hc.size(); c++)
            d = std::max(d, std::fabs(hc[c] - hg[c]));
        DOCTEST_CHECK(d < 1e-6);
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU backend matches CPU") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU backend matches CPU'"))
        return;
    // The CUDA backend mirrors the CPU Picard + Jacobi-CG path exactly, so the
    // two must agree to the linear-solver tolerance on the same transient.
    SoilProperties sl = getSoilPropertiesFromTexture("sandy_loam");
    const uint Nx = 12, Ny = 12, Nz = 20;
    const double d = 0.025, q = 0.3 * sl.K_s, T = 1800.0, dt = 30.0;
    struct Scenario {
        BCType bottom;
        double botval;
    };
    Scenario scenarios[] = {{BCType::FREE_DRAINAGE, 0.0}, {BCType::DIRICHLET, 0.0}};
    // Validate both preconditioners: Jacobi-CG and the geometric-multigrid V-cycle.
    for (bool mg: {false, true}) {
        for (auto &sc: scenarios) {
            auto runit = [&](bool gpu, std::vector<double> &hout) {
                SoilSolver3D s(Nx, Ny, Nz, d, d, d, sl);
                auto p = s.getParams();
                p.bdf2 = true;
                p.use_multigrid = mg; // same preconditioner on GPU and CPU
                p.use_gpu = gpu;
                s.setParams(p);
                s.setInitialCondition(-3.0);
                s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
                s.setBoundary(Face3D::ZMIN, sc.bottom, sc.botval);
                for (double t = 0; t < T - 1e-9; t += dt)
                    DOCTEST_REQUIRE(s.solve(std::min(dt, T - t)) > 0);
                hout = s.getPressureHeads();
            };
            std::vector<double> hcpu, hgpu;
            runit(false, hcpu);
            runit(true, hgpu);
            double md = 0.0;
            for (size_t c = 0; c < hcpu.size(); c++)
                md = std::max(md, std::fabs(hcpu[c] - hgpu[c]));
            // Both solve the same linear system to the same tolerance; they differ
            // only by float reduction order / host-vs-device libm and the Picard gate.
            DOCTEST_CHECK(md < 1e-3);
        }
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU matches CPU under each K_weighting") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU matches CPU under each K_weighting'"))
        return;
    // Regression for the GPU hardcoding harmonic: the device kMatvec/kBuildSystem/kResidual now read
    // d_faceCond per Params::K_weighting (geometric/arithmetic/upstream/harmonic), matching the CPU
    // faceCond(). Without the fix, arithmetic/upstream/geometric CPU-vs-GPU diverge (the rest of the
    // suite forces harmonic and never catches it). Infiltration into dry soil where weighting matters.
    SoilProperties sl = getSoilPropertiesFromTexture("sandy_loam");
    const uint Nx = 8, Ny = 8, Nz = 16;
    const double d = 0.025, q = 0.4 * sl.K_s, dt = 30.0;
    for (int kw: {0, 1, 2, 3, 4}) { // geometric, arithmetic, upstream, harmonic, upstream relative permeability
        auto runit = [&](bool gpu, std::vector<double> &hout) {
            SoilSolver3D s(Nx, Ny, Nz, d, d, d, sl);
            auto p = s.getParams();
            p.K_weighting = kw;
            p.use_multigrid = true;
            p.use_gpu = gpu;
            s.setParams(p);
            s.setInitialCondition(-3.0); // dry -> sharp front where geom/arith/upstream/harmonic differ
            s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
            s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
            for (int n = 0; n < 20; n++)
                DOCTEST_REQUIRE(s.solve(dt) > 0);
            hout = s.getPressureHeads();
        };
        std::vector<double> hcpu, hgpu;
        runit(false, hcpu);
        runit(true, hgpu);
        double md = 0.0;
        for (size_t c = 0; c < hcpu.size(); c++)
            md = std::max(md, std::fabs(hcpu[c] - hgpu[c]));
        DOCTEST_CHECK(md < 1e-3); // same discretization on both backends for this weighting
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU picks up setSoilProperties / setBoundary changes after a solve") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU picks up setSoilProperties / setBoundary changes after a solve'"))
        return;
    // Regression for the once-only GPU props upload and the cached multigrid Dirichlet flags: mutating
    // soil properties or boundary types AFTER a GPU solve must take effect (props_uploaded / mg dir flags
    // were previously frozen on first use). Compare a fresh-from-scratch CPU run to a GPU run that is
    // reconfigured mid-life, on the same final configuration.
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilProperties clay = getSoilPropertiesFromTexture("clay");
    const uint Nx = 6, Ny = 6, Nz = 12;
    const double d = 0.03, dt = 120.0;
    auto finalConfig = [&](SoilSolver3D &s) { // the configuration both runs must end up solving
        for (uint k = 0; k < Nz; k++)
            for (uint j = 0; j < Ny; j++)
                for (uint i = 0; i < Nx; i++)
                    s.setSoilProperties(i, j, k, (k >= Nz / 2) ? clay : loam);
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    };
    // Reference: a GPU solver configured correctly from the start.
    std::vector<double> href;
    {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, loam);
        auto p = s.getParams();
        p.use_gpu = true;
        p.use_multigrid = true;
        p.bdf2 = false;
        s.setParams(p);
        s.setInitialCondition(-1.0);
        finalConfig(s);
        for (int n = 0; n < 8; n++)
            DOCTEST_REQUIRE(s.solve(dt) > 0);
        href = s.getPressureHeads();
    }
    // GPU solver that runs first with the WRONG props/BC, then is reconfigured and reset.
    std::vector<double> hreconf;
    {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, clay); // wrong uniform props
        auto p = s.getParams();
        p.use_gpu = true;
        p.use_multigrid = true;
        p.bdf2 = false;
        s.setParams(p);
        s.setInitialCondition(-1.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 0.0); // wrong BC type
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        DOCTEST_REQUIRE(s.solve(dt) > 0); // one GPU solve -> uploads props + caches MG flags
        finalConfig(s); // now mutate props AND a BC type
        s.setInitialCondition(-1.0); // reset state, redo from the same IC
        for (int n = 0; n < 8; n++)
            DOCTEST_REQUIRE(s.solve(dt) > 0);
        hreconf = s.getPressureHeads();
    }
    double md = 0.0;
    for (size_t c = 0; c < href.size(); c++)
        md = std::max(md, std::fabs(href[c] - hreconf[c]));
    DOCTEST_CHECK(md < 1e-3); // the reconfigured GPU run must match the correctly-configured one
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU residual and jacvec match CPU") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU residual and jacvec match CPU'"))
        return;
    // Keystone for the GPU Newton-Krylov path: the device residual (kResidual) and the matrix-free
    // Jacobian (one-sided FD of kResidual) must match the validated CPU residual()/jacvec(). Both
    // sides use harmonic weighting (the GPU operator is harmonic-only).
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(6, 6, 8, 0.02, 0.02, 0.02, g);
    auto p = s.getParams();
    p.K_weighting = 3; // harmonic, matching the GPU operator
    s.setParams(p);
    s.setInitialCondition(std::vector<double>(s.numCells(), -1.0));
    s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.1);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    const double dt = 30.0;
    for (int n = 0; n < 4; n++)
        DOCTEST_REQUIRE(s.solve(dt) > 0); // populate BDF2 history

    double a0, a1, a2;
    s.timeWeights(dt, a0, a1, a2);
    const uint N = s.numCells();
    auto maxrel = [&](const std::vector<double> &A, const std::vector<double> &B) {
        double num = 0.0, den = 0.0;
        for (uint c = 0; c < N; c++) {
            num = std::max(num, std::fabs(A[c] - B[c]));
            den = std::max(den, std::fabs(A[c]));
        }
        return num / (den + 1e-300);
    };

    // (1) residual at several non-trivial states (O(1) residuals -> well-conditioned relative test).
    for (double shift: {0.05, -0.2, 0.5}) {
        std::vector<double> h = s.getPressureHeads();
        for (double &v: h)
            v += shift;
        std::vector<double> Fcpu, Fgpu;
        s.residual(h, Fcpu, a0, a1, a2);
        s.residualGPU(h, Fgpu, a0, a1, a2);
        DOCTEST_CHECK(maxrel(Fcpu, Fgpu) < 1e-9); // pointwise double math -> near machine
    }

    // (2) matrix-free Jacobian-vector product matches to FD precision.
    std::vector<double> h = s.getPressureHeads();
    std::vector<double> v(N);
    for (uint c = 0; c < N; c++)
        v[c] = 0.01 * std::sin(0.7 * (double) c + 1.0);
    std::vector<double> F0, Jcpu, Jgpu;
    s.residual(h, F0, a0, a1, a2);
    s.jacvec(h, F0, v, Jcpu, a0, a1, a2);
    s.jacvecGPU(h, v, Jgpu, a0, a1, a2);
    DOCTEST_CHECK(maxrel(Jcpu, Jgpu) < 1e-4); // FD amplifies pointwise diffs by ~1/eps
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU Newton matches CPU Newton") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU Newton matches CPU Newton'"))
        return;
    // Device-resident Newton-Krylov (kResidual + gmresGPU + V-cycle precond + line search) must
    // reach the same converged state as the CPU solveNewton() on a mild transient where pure
    // Newton is well-behaved. Harmonic weighting both sides; multigrid preconditioner both sides.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    const uint Nx = 8, Ny = 8, Nz = 12;
    const double d = 0.02, dt = 30.0;
    auto runit = [&](bool gpu, std::vector<double> &hout) {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, g);
        auto p = s.getParams();
        p.K_weighting = 3; // harmonic (the GPU operator)
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON;
        p.use_gpu = gpu;
        p.use_multigrid = true;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        for (int n = 0; n < 15; n++)
            DOCTEST_REQUIRE(s.solve(dt) > 0);
        hout = s.getPressureHeads();
    };
    std::vector<double> hcpu, hgpu;
    runit(false, hcpu);
    runit(true, hgpu);
    double md = 0.0;
    for (size_t c = 0; c < hcpu.size(); c++)
        md = std::max(md, std::fabs(hcpu[c] - hgpu[c]));
    DOCTEST_CHECK(md < 1e-3); // same solution to device/host libm + reduction-order + gate
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU Picard-Newton matches CPU") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU Picard-Newton matches CPU'"))
        return;
    // Device-resident adaptive Picard->Newton must reach the same converged state as the CPU
    // PICARD_NEWTON on a stiff infiltration where Picard stalls and the switch engages. The
    // escalation point can differ slightly (FP), but both converge to the same discrete solution.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    const uint Nx = 8, Ny = 8, Nz = 12;
    const double d = 0.02, dt = 300.0;
    auto runit = [&](bool gpu, std::vector<double> &hout) {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, g);
        auto p = s.getParams();
        p.K_weighting = 3; // harmonic (the GPU operator)
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::PICARD_NEWTON;
        p.use_gpu = gpu;
        p.use_multigrid = true;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), -3.0)); // dry -> stiff rewetting
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.02);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        for (int n = 0; n < 12; n++)
            DOCTEST_REQUIRE(s.solve(dt) > 0);
        hout = s.getPressureHeads();
    };
    std::vector<double> hcpu, hgpu;
    runit(false, hcpu);
    runit(true, hgpu);
    double md = 0.0;
    for (size_t c = 0; c < hcpu.size(); c++)
        md = std::max(md, std::fabs(hcpu[c] - hgpu[c]));
    DOCTEST_CHECK(md < 1e-3);
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU NGS-FAS matches CPU NGS-FAS") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU NGS-FAS matches CPU NGS-FAS'"))
        return;
    // Device-resident NGS-FAS (red-black nonlinear Gauss-Seidel + FAS V-cycle) must reach the same
    // converged state as the CPU NGS_FAS (lexicographic NGS). The smoother ORDERING differs (red-black
    // vs sequential), so the iteration paths differ, but both drive the same fine-grid lagged-K system
    // to its unique solution. Stiff infiltration into dry soil (the case NGS-FAS was built for).
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    const uint Nx = 8, Ny = 8, Nz = 12;
    const double d = 0.02, dt = 300.0;
    auto runit = [&](bool gpu, std::vector<double> &hout) {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, g);
        auto p = s.getParams();
        p.K_weighting = 3; // harmonic (the GPU operator)
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NGS_FAS;
        p.use_gpu = gpu;
        p.use_multigrid = true;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), -3.0)); // dry -> stiff rewetting
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.02);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        for (int n = 0; n < 12; n++)
            DOCTEST_REQUIRE(s.solve(dt) > 0);
        hout = s.getPressureHeads();
    };
    std::vector<double> hcpu, hgpu;
    runit(false, hcpu);
    runit(true, hgpu);
    double md = 0.0;
    for (size_t c = 0; c < hcpu.size(); c++)
        md = std::max(md, std::fabs(hcpu[c] - hgpu[c]));
    DOCTEST_CHECK(md < 1e-3); // same discrete lagged-K solution (different smoother ordering)
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU NGS-FAS heterogeneous matches CPU") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU NGS-FAS heterogeneous matches CPU'"))
        return;
    // Per-level restricted props on the device: a two-layer clay-over-loam box must converge and
    // match the CPU NGS-FAS (which validates the kMGAverageProps coarse-retention upscaling on GPU).
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilProperties clay = getSoilPropertiesFromTexture("clay");
    const uint Nx = 6, Ny = 6, Nz = 12;
    const double d = 0.03, dt = 300.0;
    auto runit = [&](bool gpu, std::vector<double> &hout) {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, loam);
        for (uint k = 0; k < Nz; k++)
            for (uint j = 0; j < Ny; j++)
                for (uint i = 0; i < Nx; i++)
                    s.setSoilProperties(i, j, k, (k >= Nz / 2) ? clay : loam); // clay over loam (z up)
        auto p = s.getParams();
        p.K_weighting = 3;
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::NGS_FAS;
        p.use_gpu = gpu;
        p.use_multigrid = true;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), -2.5));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.02);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        for (int n = 0; n < 10; n++)
            DOCTEST_REQUIRE(s.solve(dt) > 0);
        hout = s.getPressureHeads();
    };
    std::vector<double> hcpu, hgpu;
    runit(false, hcpu);
    runit(true, hgpu);
    double md = 0.0;
    for (size_t c = 0; c < hcpu.size(); c++)
        md = std::max(md, std::fabs(hcpu[c] - hgpu[c]));
    DOCTEST_CHECK(md < 1e-3);
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU sources and sinks match CPU") {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU sources and sinks match CPU'"))
        return;
    // The GPU kernels carry only one 'src' array; solveGPU must fold the uptake/evap sinks into it
    // (src - uptake - evap) exactly as the CPU buildSystem/residual do. Without that, the GPU
    // silently ignores root uptake + evaporation. This sets all three nonzero and checks GPU == CPU.
    SoilProperties g = getSoilPropertiesFromTexture("loam");
    const uint Nx = 8, Ny = 8, Nz = 10;
    const double d = 0.02, dt = 60.0;
    auto runit = [&](bool gpu, std::vector<double> &hout) {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, g);
        auto p = s.getParams();
        p.K_weighting = 3;
        p.use_gpu = gpu;
        p.use_multigrid = true;
        s.setParams(p);
        s.setInitialCondition(-0.5);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 0.0);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        // Gentle source (emitter), root uptake through the middle, surface evaporation -- small
        // enough to converge at fixed dt, large enough that ignoring the sinks shifts h by >1e-3.
        s.setSource(4, 4, Nz - 1, 1e-4); // gentle drip [1/s]
        for (uint k = 2; k < Nz - 2; k++)
            s.setUptake(3, 3, k, 5e-7); // root column sink [1/s]
        for (uint i = 0; i < Nx; i++)
            for (uint j = 0; j < Ny; j++)
                s.setEvaporation(i, j, Nz - 1, 3e-7); // surface evap sink
        for (int n = 0; n < 12; n++)
            DOCTEST_REQUIRE(s.solve(dt) > 0);
        hout = s.getPressureHeads();
    };
    std::vector<double> hcpu, hgpu;
    runit(false, hcpu);
    runit(true, hgpu);
    double md = 0.0, scale = 0.0;
    for (size_t c = 0; c < hcpu.size(); c++) {
        md = std::max(md, std::fabs(hcpu[c] - hgpu[c]));
        scale = std::max(scale, std::fabs(hcpu[c]));
    }
    DOCTEST_CHECK(md / scale < 1e-3); // GPU now applies source + uptake + evap, matching CPU
}

DOCTEST_TEST_CASE("SoilSolver3D - GPU scaling benchmark" * doctest::test_suite("scaling") * doctest::skip()) {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - GPU scaling benchmark'"))
        return;
    // GPU-only scaling: multigrid vs Jacobi preconditioning head-to-head, driven
    // by the BDF2 adaptive timestepper over a fixed simulated window. Both
    // preconditioners take the same adaptive steps (the controller keys off Δθ),
    // so it stays a fair comparison. Skipped by default (~3 min). Run with:
    //   ./soiltransport_tests --no-skip --test-suite=scaling
    SoilProperties sl = getSoilPropertiesFromTexture("sandy_loam");
    const double q = 0.3 * sl.K_s;
    const double T = 180.0; // simulated time to advance adaptively [s] (matches the fixed-step window)
    std::vector<int> sizes = {32, 48, 64, 96, 128, 164, 192, 256};

    auto setup = [&](SoilSolver3D &s, bool mg) {
        auto p = s.getParams();
        p.bdf2 = true;
        p.use_multigrid = mg;
        p.use_gpu = true;
        p.cg_max = 6000;
        s.setParams(p);
        s.setInitialCondition(-1.0); // moderately dry: first step converges at fine resolution
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE);
    };
    // Absorb one-time CUDA context / kernel-JIT init before any measurement.
    {
        SoilSolver3D warm(8, 8, 8, 0.01, 0.01, 0.01, sl);
        setup(warm, true);
        warm.solve(1.0);
    }

    auto timeit = [&](int n, bool mg, double &ms, uint &steps_out) {
        double d = 0.5 / n;
        double best = 1e30;
        uint best_steps = 0;
        for (int rep = 0; rep < 2; rep++) { // min-of-2 to reduce timing noise
            SoilSolver3D s(n, n, n, d, d, d, sl);
            setup(s, mg);
            double dt_a = 1.0; // initial step guess; the controller adapts from here
            uint st = 0;
            auto t0 = std::chrono::steady_clock::now();
            double adv = s.solveAdaptive(T, dt_a, &st);
            auto t1 = std::chrono::steady_clock::now();
            DOCTEST_REQUIRE(adv >= T - 1.0); // advanced the full window
            double e = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (e < best) {
                best = e;
                best_steps = st;
            }
        }
        ms = best;
        steps_out = best_steps;
    };

    const char *csv_path = "gpu_mg_vs_jacobi_adaptive.csv";
    std::ofstream csv(csv_path);
    csv << "grid_n,cells,sim_time_s,gpu_jacobi_ms,gpu_mg_ms,mg_over_jac,jacobi_steps,mg_steps,winner\n";

    std::printf("\n  grid       cells     GPU-Jac ms   GPU-MG ms   MG/Jac   Jac steps   MG steps   winner\n");
    for (int n: sizes) {
        double jac_ms = 0, mg_ms = 0;
        uint jac_steps = 0, mg_steps = 0;
        timeit(n, false, jac_ms, jac_steps);
        timeit(n, true, mg_ms, mg_steps);
        double cells = (double) n * n * n;
        const char *winner = (mg_ms < jac_ms) ? "MG" : "Jacobi";
        std::printf("  %3d^3   %9.0f   %9.1f   %9.1f   %6.2fx   %8u   %8u   %s\n", n, cells, jac_ms, mg_ms, mg_ms / jac_ms, jac_steps, mg_steps, winner);
        std::fflush(stdout);
        csv << n << ',' << (long long) cells << ',' << T << ',' << jac_ms << ',' << mg_ms << ',' << (mg_ms / jac_ms) << ',' << jac_steps << ',' << mg_steps << ',' << winner << '\n';
        csv.flush();
    }
    csv.close();
    std::printf("  wrote %s\n", csv_path);
    DOCTEST_CHECK(true);
}
#endif

DOCTEST_TEST_CASE("SoilSolver3D - M-scheme tracks upstream Picard (homogeneous infiltration)") {
    // The faithful Kirchhoff M-scheme must converge on a homogeneous dry-front infiltration and
    // reproduce the same physical solution as the head-form solver. Its exact-integral Kirchhoff flux is
    // closest to UPSTREAM weighting (harmonic badly under-infiltrates dry fronts), so we validate against
    // upstream Picard: total infiltrated water within a few percent, profiles close.
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties soil = getSoilPropertiesFromTexture("loam");
    const uint Nx = 6, Ny = 6, Nz = 30;
    const double d = 0.03, dt = 600.0;
    const int nsteps = 25;
    double massrel_m = 1.0; // M-scheme conservation residual (set in run)
    auto run = [&](NM method, int kw, std::vector<double> &th, double &water, bool &ok) {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, soil);
        auto p = s.getParams();
        p.nonlinear_method = method;
        p.use_multigrid = true;
        p.K_weighting = kw;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s.resetMassBalance();
        ok = true;
        for (int n = 0; n < nsteps; n++)
            if (s.solve(dt) == 0)
                ok = false;
        th.resize(s.numCells());
        water = 0.0;
        for (uint c = 0; c < s.numCells(); c++) {
            th[c] = s.getWaterContent(c % Nx, (c / Nx) % Ny, c / (Nx * Ny));
            water += th[c];
        }
        water *= d * d * d;
        if (method == NM::M_SCHEME)
            massrel_m = std::fabs(s.getMassBalanceResidual()) / (std::fabs(s.getCumulativeInflux()) + 1e-30);
    };
    std::vector<double> thh, thu, thm;
    double wh, wu, wm;
    bool okh, oku, okm;
    run(NM::PICARD, 3, thh, wh, okh); // harmonic Picard (badly under-infiltrates the dry front)
    run(NM::PICARD, 2, thu, wu, oku); // upstream Picard (closest head-form approximation of the infiltration flux)
    run(NM::M_SCHEME, 3, thm, wm, okm); // Kirchhoff M-scheme (exact-integral flux)
    // Discretization-robust checks (integral quantity; pointwise profiles differ O(h) between the Kirchhoff
    // and head-form schemes). The M-scheme must converge and track the ACCURATE (upstream) weighting, landing
    // far from the inaccurate (harmonic) one -- i.e. it gets the dry-front infiltration physics right.
    DOCTEST_CHECK(okm);
    DOCTEST_CHECK(std::fabs(wm - wu) / wu < 0.10); // total water within 10% of upstream Picard
    DOCTEST_CHECK(std::fabs(wm - wu) < std::fabs(wm - wh)); // and much closer to upstream than to harmonic
    // Fully-implicit + FV-conservative in its own Kirchhoff flux: getMassBalanceResidual() (now consistent
    // with the M-scheme's boundary fluxes) must report ~solver-tolerance conservation, not a spurious mismatch.
    DOCTEST_CHECK(massrel_m < 1e-3);
}

DOCTEST_TEST_CASE("SoilSolver3D - M-scheme rejects heterogeneous soil (Kirchhoff limitation)") {
    // The faithful M-scheme uses the Kirchhoff transform, valid only for one soil. Heterogeneous properties
    // must be rejected up front (a single transform would silently solve the wrong system across interfaces).
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilProperties clay = getSoilPropertiesFromTexture("clay");
    SoilSolver3D s(4, 4, 8, 0.03, 0.03, 0.03, loam);
    for (uint k = 0; k < 8; k++)
        for (uint j = 0; j < 4; j++)
            for (uint i = 0; i < 4; i++)
                s.setSoilProperties(i, j, k, (k >= 4) ? clay : loam); // two-layer
    auto p = s.getParams();
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::M_SCHEME;
    s.setParams(p);
    s.setInitialCondition(-1.0);
    s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
    helios::capture_cerr cerr_capture; // swallow the (expected) error message
    DOCTEST_CHECK_THROWS(s.solve(300.0));
}

DOCTEST_TEST_CASE("SoilSolver3D - M-scheme probe (homogeneous, vs Picard)" * doctest::skip()) {
    // On-demand probe for the faithful Kirchhoff M-scheme. Run with:
    //   ./soiltransport_tests --no-skip -tc="*M-scheme probe*"
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties soil = getSoilPropertiesFromTexture("loam");
    const uint Nx = 16, Ny = 16, Nz = 40;
    const double d = 0.025, dt = 600.0;
    const int nsteps = 40; // long enough for a real wetting front to develop in the bulk
    auto run = [&](NM method, int kw, std::vector<double> &th, uint &outer, bool &ok, double &water) {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, soil);
        auto p = s.getParams();
        p.nonlinear_method = method;
        p.use_multigrid = true;
        p.K_weighting = kw;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        outer = 0;
        ok = true;
        for (int n = 0; n < nsteps; n++) {
            uint it = s.solve(dt);
            if (it == 0)
                ok = false;
            outer += it;
        }
        th.resize(s.numCells());
        water = 0.0;
        for (uint c = 0; c < s.numCells(); c++) {
            th[c] = s.getWaterContent(c % Nx, (c / Nx) % Ny, c / (Nx * Ny));
            water += th[c];
        }
        water *= d * d * d; // total stored water [m^3]
    };
    std::vector<double> thp, thu, thm;
    uint op = 0, ou = 0, om = 0;
    bool okp = true, oku = true, okm = true;
    double wp, wu, wm;
    double mu_lo = 0, mu_hi = 0;
    run(NM::PICARD, 3, thp, op, okp, wp); // harmonic Picard (under-estimates dry infiltration)
    run(NM::PICARD, 2, thu, ou, oku, wu); // upstream Picard (closest to the exact infiltration flux)
    { // M-scheme: also capture the estimator's selected mu-range on the last step
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, soil);
        auto p = s.getParams();
        p.nonlinear_method = NM::M_SCHEME;
        p.use_multigrid = true;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), -2.0));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -0.05);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        om = 0;
        okm = true;
        for (int n = 0; n < nsteps; n++) {
            uint it = s.solve(dt);
            if (it == 0)
                okm = false;
            om += it;
        }
        thm.resize(s.numCells());
        wm = 0.0;
        for (uint c = 0; c < s.numCells(); c++) {
            thm[c] = s.getWaterContent(c % Nx, (c / Nx) % Ny, c / (Nx * Ny));
            wm += thm[c];
        }
        wm *= d * d * d;
        mu_lo = s.mschemeMuMin();
        mu_hi = s.mschemeMuMax();
    }
    double md = 0.0, mu_ = 0.0;
    for (size_t c = 0; c < thp.size(); c++) {
        md = std::max(md, std::fabs(thp[c] - thm[c]));
        mu_ = std::max(mu_, std::fabs(thu[c] - thm[c]));
    }
    std::printf("\n=== M-scheme probe (loam, %ux%ux%u, %d steps dt=%.0f) ===\n", Nx, Ny, Nz, nsteps, dt);
    std::printf("  PICARD(harm) : outer=%u  conv=%s  water=%.5f m^3\n", op, okp ? "ok" : "FAILED", wp);
    std::printf("  PICARD(upw)  : outer=%u  conv=%s  water=%.5f m^3\n", ou, oku ? "ok" : "FAILED", wu);
    std::printf("  M_SCHEME     : outer=%u  conv=%s  water=%.5f m^3  estimator mu in [%.1e, %.1e]\n", om, okm ? "ok" : "FAILED", wm, mu_lo, mu_hi);
    std::printf("  max|theta_M - theta_Picard(harm)| = %.3e ;  vs Picard(upwind) = %.3e\n", md, mu_);
    std::printf("  -- (lower max-diff vs upwind confirms the Kirchhoff exact-integral flux tracks the better weighting) --\n");
    // Depth profile (column-averaged theta by layer; k=Nz-1 is the wet top).
    std::printf("  depth profile theta(z)  [k: Picard(harm)  Picard(upw)  M_scheme]\n");
    for (int k = (int) Nz - 1; k >= 0; k -= 4) {
        double ap = 0, au = 0, am = 0;
        int cnt = 0;
        for (uint j = 0; j < Ny; j++)
            for (uint i = 0; i < Nx; i++) {
                uint c = i + Nx * (j + Ny * k);
                ap += thp[c];
                au += thu[c];
                am += thm[c];
                cnt++;
            }
        std::printf("    k=%2d : %.4f  %.4f  %.4f\n", k, ap / cnt, au / cnt, am / cnt);
    }
    std::printf("\n");
    DOCTEST_CHECK(okm);
}

DOCTEST_TEST_CASE("SoilSolver3D - A/B nonlinear-solver benchmark" * doctest::skip()) {
    // On-demand CPU harness comparing the nonlinear solvers head-to-head on a stiff infiltration into
    // dry soil (the day-8 wetting-front regime). Skipped by default; run with:
    //   ./soiltransport_tests --no-skip -tc="*A/B nonlinear*"
    // Reports per method: outer iterations (Picard/M sweeps; NGS_FAS V-cycle sweeps), total CG iters
    // (Picard/M only -- NGS has no Krylov), wall time, and the converged-solution distance from Picard.
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties soil = getSoilPropertiesFromTexture("loam");
    const uint Nx = 32, Ny = 32, Nz = 40;
    const double d = 0.025; // 0.8 x 0.8 x 1.0 m box
    const double h_dry = -5.0; // very dry initial state -> sharp wetting front when the top wets up
    const double h_top = -0.05; // near-saturated Dirichlet top -> strong gradient
    const int nsteps = 8;
    // Sweep dt: small dt is easy (Picard fine); large dt makes the front jump per step -> Picard stalls,
    // which is the regime where the lagged-K nonlinear multigrid (NGS_FAS) should win over plain Picard.
    const double dts[] = {600.0, 1800.0, 3600.0, 7200.0, 14400.0};

    struct Result {
        const char *name;
        uint outer;
        uint cg;
        double secs;
        double maxdh_vs_picard;
        double massrel;
        bool ok;
    };

    auto run = [&](const char *name, NM method, double dt, std::vector<double> *hout) -> Result {
        SoilSolver3D s(Nx, Ny, Nz, d, d, d, soil);
        auto p = s.getParams();
        p.nonlinear_method = method;
        p.use_multigrid = true;
        p.K_weighting = 1; // arithmetic (good for dry-soil infiltration; same on every method here)
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), h_dry));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, h_top);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s.resetMassBalance();
        uint outer = 0, cg = 0;
        bool ok = true;
        auto t0 = std::chrono::steady_clock::now();
        for (int n = 0; n < nsteps; n++) {
            uint cgi = 0;
            uint it = s.solve(dt, &cgi);
            if (it == 0)
                ok = false;
            outer += it;
            cg += cgi;
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double massrel = std::fabs(s.getMassBalanceResidual()) / (std::fabs(s.getCumulativeInflux()) + 1e-30);
        if (hout)
            *hout = s.getPressureHeads();
        return Result{name, outer, cg, secs, 0.0, massrel, ok};
    };
    auto maxdiff = [&](const std::vector<double> &a, const std::vector<double> &b) {
        double m = 0;
        for (size_t c = 0; c < a.size(); c++)
            m = std::max(m, std::fabs(a[c] - b[c]));
        return m;
    };

    std::printf("\n=== A/B nonlinear-solver benchmark  (%ux%ux%u = %u cells, %d steps, dry %.1fm -> top %.2fm, arithmetic-K) ===\n", Nx, Ny, Nz, Nx * Ny * Nz, nsteps, h_dry, h_top);
    std::printf("%-7s  %-12s %8s %10s %10s %14s %12s  %s\n", "dt[s]", "method", "outer", "totalCG", "wall[s]", "|dh-Picard|", "mass_rel", "conv");
    bool ms_ok_all = true;
    for (double dt: dts) {
        std::vector<double> hp, hn, hs;
        Result rp = run("PICARD", NM::PICARD, dt, &hp);
        Result rn = run("NGS_FAS", NM::NGS_FAS, dt, &hn);
        Result rs = run("M_SCHEME", NM::M_SCHEME, dt, &hs); // faithful Kirchhoff M-scheme (homogeneous loam here)
        rn.maxdh_vs_picard = maxdiff(hp, hn);
        rs.maxdh_vs_picard = maxdiff(hp, hs);
        for (const Result *r: {&rp, &rn, &rs})
            std::printf("%-7.0f  %-12s %8u %10u %10.3f %14.2e %12.2e  %s\n", dt, r->name, r->outer, r->cg, r->secs, r->maxdh_vs_picard, r->massrel, r->ok ? "ok" : "FAILED");
        std::printf("\n");
        ms_ok_all = ms_ok_all && rs.ok;
    }
    std::printf("  (NGS_FAS 'outer'=FAS sweeps, M_SCHEME 'totalCG'=inner-PCG iters; |dh-Picard| for NGS/M_SCHEME is the\n");
    std::printf("   Kirchhoff/lagged-K discretization difference, not an error. Wall time is the apples-to-apples speed metric.)\n\n");

    DOCTEST_CHECK(ms_ok_all); // M_SCHEME converges across the dt sweep (it does not match Picard pointwise: different discretization)
}

DOCTEST_TEST_CASE("SoilSolver3D - N-scaling: NGS-FAS vs Picard" * doctest::skip()) {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - N-scaling: NGS-FAS vs Picard'"))
        return;
    // Grid-size scaling at fixed domain + fixed dt: does the nonlinear-multigrid NGS-FAS stay grid-INDEPENDENT
    // (flat iteration count / time-per-cell as N grows) while Picard's cost climbs? Fixed 1 m cube refined
    // n x n x n (d = 1/n); use_gpu so PICARD->solveGPU and NGS_FAS->solveNGSFASGPU run on the device when built
    // with CUDA (CPU fallback otherwise). Run with: ./soiltransport_tests --no-skip -tc="*N-scaling*"
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties soil = getSoilPropertiesFromTexture("loam");
    const int ns[] = {16, 24, 32, 48, 64, 96, 128, 160};
    const double L = 1.0, dt = 600.0;
    const int nsteps = 6;
    const double h_dry = -3.0, h_top = -0.05;
    auto run = [&](int n, NM method, uint &outer, uint &cg, double &secs, double &massrel, bool &ok) {
        const double d = L / n;
        SoilSolver3D s((uint) n, (uint) n, (uint) n, d, d, d, soil);
        auto p = s.getParams();
        p.nonlinear_method = method;
        p.use_gpu = true;
        p.use_multigrid = true;
        p.K_weighting = 1;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), h_dry));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, h_top);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s.resetMassBalance();
        outer = 0;
        cg = 0;
        ok = true;
        auto t0 = std::chrono::steady_clock::now();
        for (int step = 0; step < nsteps; step++) {
            uint cgi = 0;
            uint it = s.solve(dt, &cgi);
            if (it == 0)
                ok = false;
            outer += it;
            cg += cgi;
        }
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        massrel = std::fabs(s.getMassBalanceResidual()) / (std::fabs(s.getCumulativeInflux()) + 1e-30); // instantaneous, quadrature-free
    };
    {
        uint o, c;
        double s_, mr;
        bool ok;
        run(8, NM::PICARD, o, c, s_, mr, ok);
        run(8, NM::NGS_FAS, o, c, s_, mr, ok);
    } // warm up CUDA (discard)
    FILE *csv = std::fopen("nscaling.csv", "w");
    if (csv)
        std::fprintf(csv, "n,cells,method,outer,totalCG,wall_s,us_per_cell_step,mass_rel,converged\n");
    std::printf("\n=== N-scaling: NGS-FAS vs Picard  (1 m cube, d=1/n, %d steps x dt=%.0fs, dry %.1fm -> top %.2fm) ===\n", nsteps, dt, h_dry, h_top);
    std::printf("%5s %10s  %-10s %8s %10s %10s %12s %11s  %s\n", "n", "cells", "method", "outer", "totalCG", "wall[s]", "us/cell/stp", "mass_rel", "conv");
    for (int n: ns) {
        const double cells = (double) n * n * n;
        for (NM m: {NM::PICARD, NM::NGS_FAS}) {
            uint outer = 0, cg = 0;
            double secs = 0, mr = 0;
            bool ok = true;
            run(n, m, outer, cg, secs, mr, ok);
            const double us_per = secs * 1e6 / (cells * nsteps);
            const char *nm = (m == NM::PICARD ? "PICARD" : "NGS_FAS");
            std::printf("%5d %10.0f  %-10s %8u %10u %10.3f %12.3f %11.2e  %s\n", n, cells, nm, outer, cg, secs, us_per, mr, ok ? "ok" : "FAILED");
            if (csv)
                std::fprintf(csv, "%d,%.0f,%s,%u,%u,%.6f,%.6f,%.6e,%d\n", n, cells, nm, outer, cg, secs, us_per, mr, ok ? 1 : 0);
            std::fflush(stdout);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    if (csv)
        std::fclose(csv);
    std::printf("  (PICARD 'outer'=Picard iters, NGS_FAS 'outer'=FAS sweeps. Grid-independent -> flat us/cell/stp. CSV: nscaling.csv)\n\n");
    DOCTEST_CHECK(true);
}

DOCTEST_TEST_CASE("SoilSolver3D - N-scaling adaptive: NGS-FAS vs Picard" * doctest::skip()) {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - N-scaling adaptive: NGS-FAS vs Picard'"))
        return;
    // Same simple infiltration benchmark, but both methods advance a fixed physical window with the
    // adaptive Delta-theta stepper (solveAdaptive). Compares wall time, substeps the controller takes,
    // and conservation as the grid refines. Run with: ./soiltransport_tests --no-skip -tc="*N-scaling adaptive*"
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties soil = getSoilPropertiesFromTexture("loam");
    const int ns[] = {16, 24, 32, 48, 64, 96, 128};
    const double L = 1.0, target_time = 7200.0; // advance 2 h -- forces multi-step adaptive behavior (window > dt_max)
    const double h_dry = -3.0, h_top = -0.05;
    auto run = [&](int n, NM method, uint &substeps, double &secs, double &massrel, bool &ok) {
        const double d = L / n;
        SoilSolver3D s((uint) n, (uint) n, (uint) n, d, d, d, soil);
        auto p = s.getParams();
        p.nonlinear_method = method;
        p.use_gpu = true;
        p.use_multigrid = true;
        p.K_weighting = 1;
        p.dt_target_rel_change = 0.1;
        p.dt_min = 60.0;
        p.dt_max = 3600.0; // adaptive controller (bounded floor)
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), h_dry));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, h_top);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s.resetMassBalance();
        double dt = 3600.0;
        auto t0 = std::chrono::steady_clock::now();
        s.solveAdaptive(target_time, dt, &substeps);
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        massrel = std::fabs(s.getMassBalanceResidual()) / (std::fabs(s.getCumulativeInflux()) + 1e-30);
        ok = true;
    };
    {
        uint o;
        double s_, mr;
        bool ok;
        run(8, NM::PICARD, o, s_, mr, ok);
        run(8, NM::NGS_FAS, o, s_, mr, ok);
    } // warm up
    FILE *csv = std::fopen("nscaling_adaptive.csv", "w");
    if (csv)
        std::fprintf(csv, "n,cells,method,substeps,wall_s,us_per_cell_substep,mass_rel\n");
    std::printf("\n=== N-scaling ADAPTIVE: NGS-FAS vs Picard  (1 m cube d=1/n, advance %.0f s with Delta-theta stepper) ===\n", target_time);
    std::printf("%5s %10s  %-10s %9s %10s %14s %11s\n", "n", "cells", "method", "substeps", "wall[s]", "us/cell/sstp", "mass_rel");
    for (int n: ns) {
        const double cells = (double) n * n * n;
        for (NM m: {NM::PICARD, NM::NGS_FAS}) {
            uint substeps = 0;
            double secs = 0, mr = 0;
            bool ok = true;
            run(n, m, substeps, secs, mr, ok);
            const char *nm = (m == NM::PICARD ? "PICARD" : "NGS_FAS");
            const double us_per = (substeps > 0) ? secs * 1e6 / (cells * substeps) : 0.0;
            std::printf("%5d %10.0f  %-10s %9u %10.3f %14.3f %11.2e\n", n, cells, nm, substeps, secs, us_per, mr);
            if (csv)
                std::fprintf(csv, "%d,%.0f,%s,%u,%.6f,%.6f,%.6e\n", n, cells, nm, substeps, secs, us_per, mr);
            std::fflush(stdout);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    if (csv)
        std::fclose(csv);
    std::printf("  (substeps = adaptive Delta-theta steps to cross the window; finer grid -> more substeps. CSV: nscaling_adaptive.csv)\n\n");
    DOCTEST_CHECK(true);
}

DOCTEST_TEST_CASE("SoilSolver3D - dt-scaling: NGS-FAS vs Picard" * doctest::skip()) {
    if (!gpuTestsCanRun("Test case 'SoilSolver3D - dt-scaling: NGS-FAS vs Picard'"))
        return;
    // Timestep scaling at fixed grid (64^3): NGS-FAS stays cheap+robust as dt grows while Picard's iteration
    // count climbs / may fail; and the residual exposes NGS-FAS's O(dt) lagged-K mass error growing with dt.
    // Run with: ./soiltransport_tests --no-skip -tc="*dt-scaling*"
    using NM = SoilSolver3D::NonlinearMethod;
    SoilProperties soil = getSoilPropertiesFromTexture("loam");
    const int n = 64;
    const double L = 1.0, d = L / n;
    const double dts[] = {150, 300, 600, 1200, 2400, 4800, 9600, 19200};
    const int nsteps = 6;
    const double h_dry = -3.0, h_top = -0.05;
    auto run = [&](double dt, NM method, uint &outer, uint &cg, double &secs, double &massrel, bool &ok) {
        SoilSolver3D s((uint) n, (uint) n, (uint) n, d, d, d, soil);
        auto p = s.getParams();
        p.nonlinear_method = method;
        p.use_gpu = true;
        p.use_multigrid = true;
        p.K_weighting = 1;
        s.setParams(p);
        s.setInitialCondition(std::vector<double>(s.numCells(), h_dry));
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, h_top);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        s.resetMassBalance();
        outer = 0;
        cg = 0;
        ok = true;
        auto t0 = std::chrono::steady_clock::now();
        for (int step = 0; step < nsteps; step++) {
            uint cgi = 0;
            uint it = s.solve(dt, &cgi);
            if (it == 0)
                ok = false;
            outer += it;
            cg += cgi;
        }
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        massrel = std::fabs(s.getMassBalanceResidual()) / (std::fabs(s.getCumulativeInflux()) + 1e-30);
    };
    {
        uint o, c;
        double s_, mr;
        bool ok;
        run(600, NM::PICARD, o, c, s_, mr, ok);
        run(600, NM::NGS_FAS, o, c, s_, mr, ok);
    } // warm up CUDA
    FILE *csv = std::fopen("dtscaling.csv", "w");
    if (csv)
        std::fprintf(csv, "dt,method,outer,totalCG,wall_s,mass_rel,converged\n");
    std::printf("\n=== dt-scaling: NGS-FAS vs Picard  (%d^3 = %d cells, %d steps, dry %.1fm -> top %.2fm) ===\n", n, n * n * n, nsteps, h_dry, h_top);
    std::printf("%8s  %-10s %8s %10s %10s %11s  %s\n", "dt[s]", "method", "outer", "totalCG", "wall[s]", "mass_rel", "conv");
    for (double dt: dts) {
        for (NM m: {NM::PICARD, NM::NGS_FAS}) {
            uint outer = 0, cg = 0;
            double secs = 0, mr = 0;
            bool ok = true;
            run(dt, m, outer, cg, secs, mr, ok);
            const char *nm = (m == NM::PICARD ? "PICARD" : "NGS_FAS");
            std::printf("%8.0f  %-10s %8u %10u %10.3f %11.2e  %s\n", dt, nm, outer, cg, secs, mr, ok ? "ok" : "FAILED");
            if (csv)
                std::fprintf(csv, "%.0f,%s,%u,%u,%.6f,%.6e,%d\n", dt, nm, outer, cg, secs, mr, ok ? 1 : 0);
            std::fflush(stdout);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    if (csv)
        std::fclose(csv);
    std::printf("  (NGS_FAS mass_rel grows ~O(dt): the lagged-K semi-implicit error. CSV: dtscaling.csv)\n\n");
    DOCTEST_CHECK(true);
}

DOCTEST_TEST_CASE("SoilSolver3D - single-column cross-check vs SoilColumn (1D)") {
    // A 1x1xNz 3D solve must reproduce the independent 1D SoilColumn solution
    // (different discretization: Thomas/float column vs CG/double 3D solver).
    // Conventions differ vertically: column layer 0 = top; 3D k = 0 = bottom.
    SoilProperties sl = getSoilPropertiesFromTexture("sandy_loam");
    uint Nz = 40;
    double dz = 0.025, q = 0.3 * sl.K_s, T = 3600.0, dt = 20.0;

    SoilColumn col(Nz, (float) dz, sl);
    SolverParameters sp;
    sp.K_weighting = KWeighting::ARITHMETIC; // same face conductivity as the 3D solver below
    sp.max_picard_iterations = 50;
    sp.picard_tolerance = 1e-6f;
    col.setSolverParameters(sp);
    col.setInitialCondition(-3.0f);
    col.setTopBoundary(BCType::NEUMANN, (float) q);
    col.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);

    SoilSolver3D s(1, 1, Nz, 0.025, 0.025, dz, sl);
    {
        auto p = s.getParams();
        p.bdf2 = false;
        p.picard_tol = 1e-6;
        p.K_weighting = 1;
        s.setParams(p);
    }
    s.setInitialCondition(-3.0);
    s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
    s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE);

    for (double t = 0; t < T - 1e-9; t += dt) {
        double h = std::min(dt, T - t);
        col.solve((float) h);
        DOCTEST_REQUIRE(s.solve(h) > 0);
    }
    double maxd = 0.0;
    for (uint L = 0; L < Nz; L++)
        maxd = std::max(maxd, std::fabs((double) col.getPressureHead(L) - s.getPressureHead(0, 0, Nz - 1 - L)));
    DOCTEST_CHECK(maxd < 3e-2); // two independent solvers agree to ~cm over a sharp front
}

DOCTEST_TEST_CASE("SoilTransportModel - full 3D solver end-to-end") {
    // Exercise the SOLVER_3D path through the public front-end API.
    Context context;
    SoilTransportModel swt(&context);
    swt.disableMessages();
    swt.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(0.2f, 0.2f, 0.5f));
    swt.setGridResolution(4, 4, 20);
    swt.setSolverType(SolverType::SOLVER_3D);
    swt.setSoilPropertiesFromTexture("loam");
    swt.setInitialCondition(-2.0f);
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    swt.setTopBoundaryFlux(0.3f * loam.K_s); // infiltration at the surface
    swt.setBottomBoundaryFreeDrainage();

    float s0 = swt.getTotalWaterStorage();
    for (int n = 0; n < 60; n++)
        swt.run(60.0f);
    float s1 = swt.getTotalWaterStorage();
    DOCTEST_CHECK(s1 > s0); // infiltration added water to the domain

    // Surface (z near 0) must be wetter than the bottom (z near -extent.z).
    float th_top = swt.getWaterContent(make_vec3(0.1f, 0.1f, -0.02f));
    float th_bot = swt.getWaterContent(make_vec3(0.1f, 0.1f, -0.48f));
    DOCTEST_CHECK(th_top > th_bot);
}

// ============================================================================
// 1D Richards-equation benchmark validation cases
//
// These mirror the standalone benchmarks in
// samples/soiltransport_benchmarks/benchmark_tests.cpp, but as automated
// doctest cases (no CSV / stdout side effects). The four heaviest cases are
// placed in the "slow" test suite so they can be excluded for quick runs:
//     ./soiltransport_tests -tse=slow      (skip slow benchmarks)
//     ./soiltransport_tests -ts=slow       (run only slow benchmarks)
// ============================================================================

// Zadeh (2007) experimental cumulative infiltration: {time_days, infiltration_cm}.
// Shared by the fixed-dt (#13) and adaptive (#16) Zadeh benchmarks.
static std::vector<std::pair<float, float>> zadehReferenceData() {
    return {{0.0038f, 0.049f}, {0.0128f, 0.220f}, {0.0103f, 0.416f}, {0.0154f, 0.563f}, {0.0218f, 0.734f}, {0.0295f, 1.052f}, {0.0398f, 1.273f}, {0.0526f, 1.420f}, {0.0590f, 1.591f}, {0.0718f, 1.885f}, {0.0833f, 2.080f}, {0.0974f, 2.325f},
            {0.1115f, 2.423f}, {0.1243f, 2.790f}, {0.1409f, 2.913f}, {0.1563f, 2.962f}, {0.1704f, 3.304f}, {0.1883f, 3.500f}, {0.2075f, 3.818f}, {0.2306f, 4.063f}, {0.2523f, 4.357f}, {0.2753f, 4.455f}, {0.2894f, 4.552f}, {0.3163f, 4.822f},
            {0.3368f, 5.189f}, {0.3611f, 5.336f}, {0.3905f, 5.483f}, {0.4097f, 5.752f}, {0.4379f, 5.825f}, {0.4622f, 5.948f}, {0.4903f, 6.315f}, {0.5185f, 6.486f}, {0.5441f, 6.829f}, {0.5697f, 6.902f}, {0.6029f, 7.024f}, {0.6311f, 7.196f},
            {0.6618f, 7.490f}, {0.6874f, 7.710f}, {0.7181f, 8.028f}, {0.7437f, 8.322f}, {0.7796f, 8.542f}, {0.8103f, 9.252f}, {0.8410f, 9.276f}, {0.8691f, 9.448f}, {0.9037f, 9.497f}, {0.9318f, 9.668f}, {0.9612f, 9.717f}, {0.9971f, 10.647f}};
}

DOCTEST_TEST_CASE("Benchmark 1 - Mass balance (zero-flux boundaries)") {
    // Zero flux at top and bottom: water must be conserved.
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    uint nz = 50;
    float dz = 0.02f, dt = 60.0f, total_time = 3600.0f;

    SoilColumn column(nz, dz, props);
    column.setInitialCondition(-1.0f);
    column.setTopBoundary(BCType::NEUMANN, 0.0f);
    column.setBottomBoundary(BCType::NEUMANN, 0.0f);

    float initial_storage = column.getTotalWaterStorage();
    for (float time = 0.0f; time < total_time; time += dt) {
        column.solve(dt);
    }
    float relative_error = std::fabs(column.getTotalWaterStorage() - initial_storage) / initial_storage;
    DOCTEST_CHECK(relative_error < 0.001f);
}

DOCTEST_TEST_CASE("Benchmark 2 - Infiltration into dry soil") {
    SoilProperties props = getSoilPropertiesFromTexture("sandy_loam");
    uint nz = 100;
    float dz = 0.01f, dt = 10.0f, total_time = 3600.0f;

    SoilColumn column(nz, dz, props);
    column.setInitialCondition(-2.0f);
    column.setTopBoundary(BCType::DIRICHLET, -0.1f);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);

    for (float time = 0.0f; time < total_time; time += dt) {
        column.solve(dt);
    }
    float top_theta = column.getWaterContent(0);
    float bottom_theta = column.getWaterContent(nz - 1);
    DOCTEST_CHECK(top_theta > bottom_theta);
    DOCTEST_CHECK(top_theta > 0.2f);
}

DOCTEST_TEST_CASE("Benchmark 3 - Gravity drainage") {
    SoilProperties props = getSoilPropertiesFromTexture("sandy_loam");
    uint nz = 50;
    float dz = 0.02f, dt = 60.0f, total_time = 7200.0f;

    SoilColumn column(nz, dz, props);
    column.setInitialCondition(-0.3f);
    column.setTopBoundary(BCType::NEUMANN, 0.0f);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);

    float initial_storage = column.getTotalWaterStorage();
    for (float time = 0.0f; time < total_time; time += dt) {
        column.solve(dt);
    }
    float drainage = initial_storage - column.getTotalWaterStorage();
    DOCTEST_CHECK(drainage > 0.0f);
}

DOCTEST_TEST_CASE("Benchmark 4 - Soil-type infiltration ordering") {
    // Under identical ponding, a sandy soil admits more water than a clay soil
    // (higher saturated conductivity).
    uint nz = 100;
    float dz = 0.01f, dt = 10.0f, total_time = 3600.0f;

    auto infiltration = [&](const std::string &soil) {
        SoilProperties props = getSoilPropertiesFromTexture(soil);
        SoilColumn column(nz, dz, props);
        column.setInitialCondition(-2.0f);
        column.setTopBoundary(BCType::DIRICHLET, -0.1f);
        column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
        float initial = column.getTotalWaterStorage();
        for (float time = 0.0f; time < total_time; time += dt) {
            column.solve(dt);
        }
        return column.getTotalWaterStorage() - initial;
    };

    float infil_sand = infiltration("sand");
    float infil_clay = infiltration("clay");
    DOCTEST_CHECK(infil_sand > 0.0f);
    DOCTEST_CHECK(infil_sand > infil_clay);
}

DOCTEST_TEST_CASE("Benchmark 5 - Grid convergence") {
    // Successive refinement should converge toward the fine-grid solution.
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    float column_depth = 1.0f, total_time = 1800.0f;

    auto finalStorage = [&](uint nz) {
        float dz = column_depth / static_cast<float>(nz);
        float dt = std::min(60.0f, 0.5f * dz * dz / props.K_s);
        SoilColumn column(nz, dz, props);
        column.setInitialCondition(-2.0f);
        column.setTopBoundary(BCType::DIRICHLET, -0.1f);
        column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
        for (float time = 0.0f; time < total_time; time += dt) {
            column.solve(dt);
        }
        return column.getTotalWaterStorage();
    };

    // Error against a fine-grid reference must shrink with refinement (first-order in dz).
    const float ref = finalStorage(400);
    const float e20 = std::fabs(finalStorage(20) - ref), e50 = std::fabs(finalStorage(50) - ref), e100 = std::fabs(finalStorage(100) - ref);
    DOCTEST_CHECK(e50 < e20);
    DOCTEST_CHECK(e100 < e50);
}

DOCTEST_TEST_CASE("Benchmark 6 - Philip sorptivity (verification)") {
    // VERIFICATION against an exact analytical result: the Philip sorptivity, obtained
    // by solving the Boltzmann similarity ODE for the same van Genuchten soil.
    //
    // Early-time ponded infiltration obeys I(t) = S*sqrt(t) + A2*t, so
    //   I(t)/sqrt(t) = S + A2*sqrt(t)
    // is linear in sqrt(t): a least-squares fit recovers the numerical sorptivity as the
    // intercept (gravity is carried by the slope A2). Arithmetic interblock conductivity
    // is used because the geometric mean under-predicts flux across the sharp wetting front.
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    float h_init = -3.0f;
    float S_exact = sorptivity(props, h_init); // exact Boltzmann sorptivity
    DOCTEST_CHECK(S_exact > 0.0f);

    uint nz = 300;
    float dz = 0.0005f, dt = 0.5f; // fine grid resolves the thin early-time front
    SoilColumn column(nz, dz, props);
    column.setInitialCondition(h_init);
    column.setTopBoundary(BCType::DIRICHLET, 0.0f);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    SolverParameters params;
    params.max_picard_iterations = 100;
    params.picard_tolerance = 1e-7f;
    params.K_weighting = KWeighting::ARITHMETIC; // arithmetic mean
    column.setSolverParameters(params);

    const std::vector<float> sample_times = {100.0f, 150.0f, 200.0f, 300.0f, 400.0f};
    float initial_storage = column.getTotalWaterStorage();
    std::vector<float> xs, ys; // x = sqrt(t), y = I/sqrt(t)
    float time = 0.0f;
    size_t next = 0;
    while (next < sample_times.size()) {
        if (time >= sample_times[next] - 0.5f * dt) {
            float I = column.getTotalWaterStorage() - initial_storage;
            xs.push_back(std::sqrt(sample_times[next]));
            ys.push_back(I / std::sqrt(sample_times[next]));
            next++;
            continue;
        }
        column.solve(dt);
        time += dt;
    }

    // Linear regression y = S_num + slope*x; intercept is the numerical sorptivity.
    float npts = static_cast<float>(xs.size());
    float sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < xs.size(); i++) {
        sx += xs[i];
        sy += ys[i];
        sxx += xs[i] * xs[i];
        sxy += xs[i] * ys[i];
    }
    float slope = (npts * sxy - sx * sy) / (npts * sxx - sx * sx);
    float S_num = (sy - slope * sx) / npts;

    DOCTEST_CHECK(S_num == doctest::Approx(S_exact).epsilon(0.05)); // numerical -> exact within 5%
}

DOCTEST_TEST_CASE("Benchmark 7 - Steady-state unit gradient") {
    SoilProperties props = getSoilPropertiesFromTexture("sandy_loam");
    uint nz = 50;
    float dz = 0.02f, dt = 60.0f, total_time = 86400.0f;

    SoilColumn column(nz, dz, props);
    float h_uniform = -1.0f;
    column.setInitialCondition(h_uniform);
    column.setTopBoundary(BCType::DIRICHLET, h_uniform);
    column.setBottomBoundary(BCType::DIRICHLET, h_uniform);

    for (float time = 0.0f; time < total_time; time += dt) {
        column.solve(dt);
    }
    float max_deviation = 0.0f;
    for (uint k = 0; k < nz; k++) {
        max_deviation = std::max(max_deviation, std::fabs(column.getPressureHead(k) - h_uniform));
    }
    DOCTEST_CHECK(max_deviation < 0.01f); // uniform within 1 cm
}

DOCTEST_TEST_CASE("Benchmark 17 - Gardner steady infiltration (exact verification)") {
    // The Gardner exponential model K(h) = K_s*exp(alpha*h) linearizes Richards' equation,
    // so steady infiltration at flux q toward a water table (h=0) has the closed-form profile
    //   Phi(z) = [(K_s - q) e^{-alpha z} + q]/alpha,  h(z) = ln(alpha*Phi/K_s)/alpha,
    // with z the height above the water table. This is a true verification: the numerical
    // steady state should match the exact solution to discretization error.
    SoilProperties soil = SoilProperties::gardner(0.06f, 0.40f, 2.0f, 1e-5f);
    float Ks = soil.K_s, alpha = soil.alpha;
    float q = 0.5f * Ks;
    float L = 0.5f;
    uint nz = 50;
    float dz = L / static_cast<float>(nz);
    auto h_exact = [&](float z) { return std::log(((Ks - q) * std::exp(-alpha * z) + q) / Ks) / alpha; };

    SoilColumn column(nz, dz, soil);
    column.setInitialCondition(0.0f);
    column.setTopBoundary(BCType::NEUMANN, q);
    column.setBottomBoundary(BCType::DIRICHLET, 0.0f);
    SolverParameters params;
    params.max_picard_iterations = 100;
    params.picard_tolerance = 1e-9f;
    params.K_weighting = KWeighting::ARITHMETIC;
    column.setSolverParameters(params);

    float dt = 60.0f;
    std::vector<float> h_prev(nz);
    for (uint s = 0; s < 100000; s++) {
        for (uint k = 0; k < nz; k++)
            h_prev[k] = column.getPressureHead(k);
        column.solve(dt);
        float dmax = 0.0f;
        for (uint k = 0; k < nz; k++)
            dmax = std::max(dmax, std::fabs(column.getPressureHead(k) - h_prev[k]));
        if (dmax < 1e-9f && s > 100)
            break;
    }

    float max_err = 0.0f;
    for (uint k = 0; k < nz; k++) {
        float z = L - column.getLayerDepth(k);
        max_err = std::max(max_err, std::fabs(column.getPressureHead(k) - h_exact(z)));
    }
    DOCTEST_CHECK(max_err < 1e-4f); // numerical steady state == exact, to discretization error
}

DOCTEST_TEST_CASE("Benchmark 9 - Celia/Polmann infiltration") {
    SoilProperties soil;
    soil.theta_r = 0.102f;
    soil.theta_s = 0.368f;
    soil.alpha = 3.35f;
    soil.n = 2.0f;
    soil.K_s = 9.22e-5f;

    uint nz = 40;
    float dz = 0.01f, dt = 1.0f, total_time = 360.0f;
    float h_init = -0.615f, h_top = -0.207f, h_bottom = -0.615f;

    SoilColumn column(nz, dz, soil);
    column.setInitialCondition(h_init);
    column.setTopBoundary(BCType::DIRICHLET, h_top);
    column.setBottomBoundary(BCType::DIRICHLET, h_bottom);
    SolverParameters params;
    params.max_picard_iterations = 50;
    params.picard_tolerance = 1e-6f;
    column.setSolverParameters(params);

    // Approximate reference heads (depth_cm from top, h_ref_cm) for a sanity bound.
    const std::vector<std::pair<float, float>> reference_data = {{0.0f, -20.7f}, {4.0f, -21.5f}, {8.0f, -24.0f}, {12.0f, -30.0f}, {16.0f, -40.0f}, {20.0f, -52.0f}, {24.0f, -58.0f}, {28.0f, -60.5f}, {32.0f, -61.2f}, {36.0f, -61.4f}, {40.0f, -61.5f}};

    float initial_storage = column.getTotalWaterStorage();
    uint total_steps = 0, converged_steps = 0;
    for (float time = 0.0f; time < total_time; time += dt) {
        if (column.solve(dt) > 0)
            converged_steps++;
        total_steps++;
    }
    float final_storage = column.getTotalWaterStorage();

    float sum_sq_error = 0.0f;
    int n_compare = 0;
    for (const auto &ref_point: reference_data) {
        float num_h_cm = 0.0f;
        for (uint k = 0; k < nz; k++) {
            if (std::fabs(column.getLayerDepth(k) * 100.0f - ref_point.first) < dz * 100.0f) {
                num_h_cm = column.getPressureHead(k) * 100.0f;
                break;
            }
        }
        float error = num_h_cm - ref_point.second;
        sum_sq_error += error * error;
        n_compare++;
    }
    float rmse = std::sqrt(sum_sq_error / n_compare);

    DOCTEST_CHECK(rmse < 100.0f); // loose: reference values are approximate
    DOCTEST_CHECK(converged_steps >= static_cast<uint>(total_steps * 0.99f));
    DOCTEST_CHECK(final_storage > initial_storage); // infiltration occurred
}

DOCTEST_TEST_CASE("Benchmark 10 - Warrick ponded infiltration (dt stability)" * doctest::test_suite("slow")) {
    // Yolo light clay, ponded surface. The implicit solver should give a
    // dt-independent profile and remain convergent across timestep sizes.
    SoilProperties soil;
    soil.theta_r = 0.124f;
    soil.theta_s = 0.495f;
    soil.alpha = 1.5f;
    soil.n = 2.0f;
    soil.K_s = 1.203e-7f;

    uint nz = 24;
    float dz = 0.05f, total_time = 360000.0f, h_init = -2.025f, h_top = 0.0f;

    std::vector<std::vector<float>> profiles;
    for (float dt: {1.0f, 300.0f, 900.0f}) {
        SoilColumn column(nz, dz, soil);
        column.setInitialCondition(h_init);
        column.setTopBoundary(BCType::DIRICHLET, h_top);
        column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
        SolverParameters params;
        params.max_picard_iterations = 50;
        params.picard_tolerance = 1e-6f;
        column.setSolverParameters(params);

        uint converged = 0, total = 0;
        for (float time = 0.0f; time < total_time; time += dt) {
            if (column.solve(dt) > 0)
                converged++;
            total++;
        }
        DOCTEST_CHECK(converged >= static_cast<uint>(total * 0.95f));
        DOCTEST_CHECK(column.getWaterContent(0) > 0.4f); // surface near saturation (theta_s=0.495)

        std::vector<float> prof;
        for (uint k = 0; k < nz; k++)
            prof.push_back(column.getPressureHead(k));
        profiles.push_back(prof);
    }

    float max_diff = 0.0f;
    for (uint k = 0; k < nz; k++) {
        max_diff = std::max(max_diff, std::fabs(profiles[0][k] - profiles[2][k]));
    }
    DOCTEST_CHECK(max_diff < 0.5f); // dt=1s vs dt=900s agree within 0.5 m head
}

DOCTEST_TEST_CASE("Benchmark 11 - Miller multi-soil infiltration (adaptive)") {
    struct MillerSoil {
        const char *name;
        SoilProperties props;
    };
    std::vector<MillerSoil> soils = {{"sand", {0.093f, 0.301f, 547.0f, 4.264f, 5.83e-7f}}, {"loam", {0.078f, 0.43f, 360.0f, 1.56f, 2.89e-8f}}, {"clay_loam", {0.095f, 0.41f, 190.0f, 1.31f, 7.18e-9f}}};

    uint nz = 80;
    float dz = 0.00125f, total_time = 15552.0f, h_top = 0.001f, h_bottom = 0.0f, h_init = -0.05f;

    for (const auto &s: soils) {
        SoilColumn column(nz, dz, s.props);
        column.setInitialCondition(h_init);
        column.setTopBoundary(BCType::DIRICHLET, h_top);
        column.setBottomBoundary(BCType::DIRICHLET, h_bottom);
        SolverParameters params;
        params.max_picard_iterations = 100;
        params.picard_tolerance = 1e-6f;
        params.adaptive_dt = true;
        params.dt_min = 0.01f;
        params.dt_max = 120.0f;
        params.dt_target_rel_change = 0.015f;
        column.setSolverParameters(params);

        float dt = 1.0f;
        uint steps = 0;
        column.solveAdaptive(total_time, dt, &steps);
        auto stats = column.getAdaptiveSolveStats();
        uint attempts = stats.total_steps + stats.failed_steps;
        float converged_ratio = (attempts > 0) ? static_cast<float>(stats.total_steps) / attempts : 1.0f;
        DOCTEST_CHECK(converged_ratio >= 0.9f);
    }
}

DOCTEST_TEST_CASE("Benchmark 12 - Ireson HYDRUS validation") {
    SoilProperties soil;
    soil.theta_r = 0.131f;
    soil.theta_s = 0.396f;
    soil.alpha = 0.423f;
    soil.n = 2.06f;
    soil.K_s = 5.74e-7f;

    struct Ref {
        float day, h0, t0, h5, t5, h10, t10;
    };
    std::vector<Ref> hydrus_ref = {{1.0f, -3.78f, 0.2685f, -3.61f, 0.2731f, -3.59f, 0.2736f},  {10.0f, -4.09f, 0.2601f, -3.78f, 0.2677f, -3.68f, 0.2709f}, {20.0f, -4.31f, 0.2548f, -3.99f, 0.2624f, -3.83f, 0.2661f},
                                   {30.0f, -4.03f, 0.2615f, -4.02f, 0.2616f, -3.99f, 0.2624f}, {40.0f, -4.10f, 0.2597f, -3.81f, 0.2667f, -3.79f, 0.2674f}, {50.0f, -4.03f, 0.2613f, -3.99f, 0.2623f, -3.91f, 0.2641f},
                                   {60.0f, -3.89f, 0.2644f, -3.70f, 0.2703f, -3.73f, 0.2693f}, {70.0f, -4.10f, 0.2597f, -3.81f, 0.2669f, -3.74f, 0.2689f}, {80.0f, -3.18f, 0.2861f, -3.59f, 0.2734f, -3.77f, 0.2680f},
                                   {90.0f, -3.95f, 0.2629f, -3.73f, 0.2695f, -3.70f, 0.2702f}};

    // openRE forcing: daily precipitation [mm/day] for days 1..90 (no evaporation),
    // top = precipitation flux, bottom = free drainage (github.com/amireson/openRE).
    const std::vector<float> precip_mm_day = {0, 0, 0, 1,       1, 0,       1,       0, 0, 0, 0, 0, 1, 0, 0, 0,       1, 0, 0,       0,      0, 0, 0, 0, 0, 0,       0, 5.0071f, 1.0014f, 0, 10, 0, 2, 1,       0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              1, 0, 0, 3.0568f, 0, 8.3274f, 4.0903f, 0, 0, 0, 2, 1, 1, 0, 0, 1.0416f, 0, 0, 4.0972f, 1.001f, 0, 0, 0, 0, 0, 1.0044f, 0, 0,       3.094f,  0, 0,  3, 1, 6.7048f, 2.2349f, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

    uint nz = 30;
    float dz = 0.05f, h_init = -3.59f, dt = 3600.0f, total_time = 90.0f * 86400.0f;
    uint L0 = 0, L5 = 10, L10 = 20;

    SoilColumn column(nz, dz, soil);
    column.setInitialCondition(h_init);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    SolverParameters params;
    params.max_picard_iterations = 50;
    params.picard_tolerance = 1e-5f;
    column.setSolverParameters(params);

    std::vector<float> h0 = {column.getPressureHead(L0)}, t0 = {column.getWaterContent(L0)};
    std::vector<float> h5 = {column.getPressureHead(L5)}, t5 = {column.getWaterContent(L5)};
    std::vector<float> h10 = {column.getPressureHead(L10)}, t10 = {column.getWaterContent(L10)};

    uint converged = 0, total = 0;
    float time = 0.0f;
    while (time < total_time) {
        int day = std::min(static_cast<int>(time / 86400.0f), static_cast<int>(precip_mm_day.size()) - 1);
        column.setTopBoundary(BCType::NEUMANN, precip_mm_day[day] / 1000.0f / 86400.0f);
        if (column.solve(dt) > 0)
            converged++;
        total++;
        time += dt;
        if (static_cast<int>(time) % 86400 == 0) {
            h0.push_back(column.getPressureHead(L0));
            t0.push_back(column.getWaterContent(L0));
            h5.push_back(column.getPressureHead(L5));
            t5.push_back(column.getWaterContent(L5));
            h10.push_back(column.getPressureHead(L10));
            t10.push_back(column.getWaterContent(L10));
        }
    }

    float ssh = 0.0f, sst = 0.0f;
    int n = 0;
    for (const auto &r: hydrus_ref) {
        int d = static_cast<int>(r.day);
        if (d >= static_cast<int>(h0.size()))
            continue;
        ssh += (h0[d] - r.h0) * (h0[d] - r.h0) + (h5[d] - r.h5) * (h5[d] - r.h5) + (h10[d] - r.h10) * (h10[d] - r.h10);
        sst += (t0[d] - r.t0) * (t0[d] - r.t0) + (t5[d] - r.t5) * (t5[d] - r.t5) + (t10[d] - r.t10) * (t10[d] - r.t10);
        n += 3;
    }
    float rmse_h = std::sqrt(ssh / n), rmse_theta = std::sqrt(sst / n);
    DOCTEST_CHECK(rmse_h < 1.0f);
    DOCTEST_CHECK(rmse_theta < 0.05f);
    DOCTEST_CHECK(converged >= static_cast<uint>(total * 0.95f));
}

DOCTEST_TEST_CASE("Benchmark 13 - Zadeh 2007 infiltration (fixed dt)" * doctest::test_suite("slow")) {
    SoilProperties soil;
    soil.theta_r = 0.062f;
    soil.theta_s = 0.46f;
    soil.alpha = 3.7f;
    soil.n = 1.67f;
    soil.K_s = 9.3663e-7f;

    uint nz = 100;
    float dz = 0.0035f, h_init = -200.0f, dt = 1.0f, total_time = 86400.0f;

    SoilColumn column(nz, dz, soil);
    column.setInitialCondition(h_init);
    column.setTopBoundary(BCType::DIRICHLET, 0.0f);
    column.setBottomBoundary(BCType::NEUMANN, 0.0f);
    SolverParameters params;
    params.max_picard_iterations = 200;
    params.picard_tolerance = 1e-6f;
    params.picard_omega = 0.7f;
    params.K_weighting = KWeighting::ARITHMETIC;
    column.setSolverParameters(params);

    float cumulative = 0.0f, prev = column.getTotalWaterStorage();
    for (float time = 0.0f; time <= total_time; time += dt) {
        column.solve(dt);
        float cur = column.getTotalWaterStorage();
        cumulative += cur - prev;
        prev = cur;
    }
    // Reference: final experimental cumulative infiltration at t=1 day (last data point).
    float ref_final_infil = zadehReferenceData().back().second;
    float final_infil_cm = cumulative * 100.0f;
    float error_pct = std::fabs(final_infil_cm - ref_final_infil) / ref_final_infil * 100.0f;
    DOCTEST_CHECK(error_pct < 1.0f);
}

DOCTEST_TEST_CASE("Benchmark 14 - Abeele 1984 free drainage" * doctest::test_suite("slow")) {
    SoilProperties soil;
    soil.theta_r = 0.0f;
    soil.theta_s = 0.33f;
    soil.alpha = 1.43f;
    soil.n = 1.506f;
    soil.K_s = 2.8889e-6f;

    uint nz = 100;
    float dz = 0.06f, dt = 10.0f, total_time = 100.0f * 86400.0f;
    float h_init = -1e-7f;

    // HYDRUS reference at t=100 days: {theta, depth_m} and {h_m, depth_m}.
    const std::vector<std::pair<float, float>> ref_theta = {{0.152f, 0.0641f}, {0.156f, 0.214f}, {0.158f, 0.385f}, {0.162f, 0.556f}, {0.165f, 0.734f}, {0.167f, 0.924f}, {0.170f, 1.10f}, {0.172f, 1.28f}, {0.175f, 1.45f}, {0.177f, 1.63f},
                                                            {0.179f, 1.83f},   {0.181f, 2.00f},  {0.184f, 2.20f},  {0.185f, 2.42f},  {0.187f, 2.58f},  {0.189f, 2.75f},  {0.191f, 2.97f}, {0.193f, 3.17f}, {0.196f, 3.36f}, {0.197f, 3.53f},
                                                            {0.199f, 3.69f},   {0.203f, 3.92f},  {0.207f, 4.11f},  {0.210f, 4.25f},  {0.216f, 4.43f},  {0.221f, 4.59f},  {0.227f, 4.74f}, {0.235f, 4.88f}, {0.243f, 5.03f}, {0.253f, 5.18f},
                                                            {0.264f, 5.31f},   {0.275f, 5.43f},  {0.285f, 5.54f},  {0.296f, 5.64f},  {0.307f, 5.74f},  {0.317f, 5.85f},  {0.328f, 5.95f}};
    const std::vector<std::pair<float, float>> ref_h = {{-3.00f, 0.0746f}, {-2.87f, 0.214f}, {-2.76f, 0.351f}, {-2.66f, 0.498f}, {-2.57f, 0.644f}, {-2.49f, 0.780f}, {-2.43f, 0.920f}, {-2.36f, 1.05f},  {-2.29f, 1.19f},  {-2.23f, 1.35f},
                                                        {-2.17f, 1.52f},   {-2.13f, 1.65f},  {-2.07f, 1.83f},  {-2.02f, 1.98f},  {-1.98f, 2.15f},  {-1.93f, 2.33f},  {-1.88f, 2.50f},  {-1.84f, 2.72f},  {-1.78f, 2.90f},  {-1.76f, 3.03f},
                                                        {-1.72f, 3.21f},   {-1.66f, 3.40f},  {-1.62f, 3.56f},  {-1.57f, 3.74f},  {-1.52f, 3.90f},  {-1.47f, 4.07f},  {-1.40f, 4.22f},  {-1.30f, 4.42f},  {-1.22f, 4.57f},  {-1.13f, 4.71f},
                                                        {-1.03f, 4.86f},   {-0.927f, 5.01f}, {-0.806f, 5.15f}, {-0.693f, 5.27f}, {-0.564f, 5.42f}, {-0.421f, 5.58f}, {-0.285f, 5.72f}, {-0.141f, 5.87f}, {-0.0275f, 5.97f}};

    SoilColumn column(nz, dz, soil);
    column.setInitialCondition(h_init);
    column.setTopBoundary(BCType::NEUMANN, 0.0f);
    column.setBottomBoundary(BCType::DIRICHLET, 0.0f);
    SolverParameters params;
    params.max_picard_iterations = 50;
    params.picard_tolerance = 1e-5f;
    column.setSolverParameters(params);

    uint converged = 0, total = 0;
    for (float time = 0.0f; time < total_time; time += dt) {
        if (column.solve(dt) > 0)
            converged++;
        total++;
    }
    float convergence_pct = 100.0f * converged / total;

    float theta_err = 0.0f;
    for (const auto &p: ref_theta) {
        uint layer = std::min(static_cast<uint>(p.second / dz), nz - 1);
        theta_err += std::fabs(column.getWaterContent(layer) - p.first);
    }
    float theta_mae = theta_err / ref_theta.size();

    float h_err = 0.0f;
    for (const auto &p: ref_h) {
        uint layer = std::min(static_cast<uint>(p.second / dz), nz - 1);
        h_err += std::fabs(column.getPressureHead(layer) - p.first);
    }
    float h_mae = h_err / ref_h.size();

    DOCTEST_CHECK(convergence_pct >= 90.0f);
    DOCTEST_CHECK(theta_mae <= 0.05f);
    DOCTEST_CHECK(h_mae <= 0.5f);
}

DOCTEST_TEST_CASE("Benchmark 15 - Lehmann-Ackerer heterogeneous column" * doctest::test_suite("slow")) {
    SoilProperties sand_soil;
    sand_soil.theta_r = 0.0286f;
    sand_soil.theta_s = 0.366f;
    sand_soil.alpha = 2.8f;
    sand_soil.n = 2.239f;
    sand_soil.K_s = 6.26e-5f;

    SoilProperties clay_soil;
    clay_soil.theta_r = 0.106f;
    clay_soil.theta_s = 0.469f;
    clay_soil.alpha = 1.04f;
    clay_soil.n = 1.395f;
    clay_soil.K_s = 1.51e-6f;

    uint nz = 200;
    float dz = 1.8f / static_cast<float>(nz);
    uint layers_per_zone = nz / 3;

    SoilColumn column(nz, dz, sand_soil);
    column.setLayerSoilProperties(0, layers_per_zone, sand_soil);
    column.setLayerSoilProperties(layers_per_zone, 2 * layers_per_zone, clay_soil);
    column.setLayerSoilProperties(2 * layers_per_zone, nz, sand_soil);
    column.setInitialCondition(-1.0f);
    column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);

    // Daily surface ET flux (cm/day); index 0 = day 0 (unused), days 1-20 used.
    const std::vector<float> et_cm_per_day = {0.0f, 0.0f, 1.003f, 0.6015f, 0.7986f, 0.5043f, 0.1049f, 0.3046f, 0.6041f, 1.201f, 1.201f, -0.09727f, -0.04608f, -0.09727f, -0.1459f, -0.04863f, -0.09471f, -0.09471f, 0.5043f, -0.04863f, -0.06655f};
    const std::vector<std::pair<float, float>> ref_theta_profile = {{0.07536f, 0.003342f}, {0.08123f, 0.1074f}, {0.08595f, 0.2397f}, {0.08839f, 0.4295f}, {0.09078f, 0.5350f}, {0.3813f, 0.6299f},
                                                                    {0.3843f, 0.7171f},    {0.3831f, 0.8127f},  {0.3843f, 0.9027f},  {0.3856f, 1.008f},   {0.3862f, 1.097f},   {0.09762f, 1.252f},
                                                                    {0.1000f, 1.387f},     {0.1001f, 1.491f},   {0.1007f, 1.583f},   {0.1008f, 1.712f},   {0.08833f, 0.3409f}};

    float dt = 20.0f;
    uint n_days = 20;
    float total_time = static_cast<float>(n_days) * 86400.0f;
    float seconds_per_day = 86400.0f;

    SolverParameters params;
    params.max_picard_iterations = 100;
    params.picard_tolerance = 1e-5f;
    params.picard_omega = 0.8f;
    params.K_weighting = KWeighting::ARITHMETIC;
    column.setSolverParameters(params);

    uint converged = 0, total = 0;
    for (float time = 0.0f; time < total_time; time += dt) {
        int day_num = static_cast<int>(time / seconds_per_day);
        int et_index = std::min(day_num + 1, static_cast<int>(et_cm_per_day.size()) - 1);
        float flux_m_s = et_cm_per_day[et_index] / 100.0f / 86400.0f;
        column.setTopBoundary(BCType::NEUMANN, flux_m_s);
        if (column.solve(dt) > 0)
            converged++;
        total++;
    }
    float convergence_pct = 100.0f * converged / total;

    float total_error = 0.0f;
    for (const auto &p: ref_theta_profile) {
        uint layer = std::min(static_cast<uint>(p.second / dz), nz - 1);
        total_error += std::fabs(column.getWaterContent(layer) - p.first);
    }
    float mean_error_pct = (total_error / ref_theta_profile.size()) * 100.0f;

    float theta_sand_top = column.getWaterContent(layers_per_zone / 2);
    float theta_clay = column.getWaterContent(3 * layers_per_zone / 2);
    float theta_sand_bot = column.getWaterContent(5 * layers_per_zone / 2);

    DOCTEST_CHECK(convergence_pct >= 90.0f);
    DOCTEST_CHECK(theta_clay > theta_sand_top); // clay retains more water than sand
    DOCTEST_CHECK(theta_clay > theta_sand_bot);
    DOCTEST_CHECK(mean_error_pct <= 5.0f);
}

DOCTEST_TEST_CASE("Benchmark 16 - Adaptive timestepping (Zadeh problem)") {
    // Same dry-soil infiltration as benchmark 13, solved adaptively. Verify the
    // controller is both faster than the fixed dt=1s run and accurate.
    const std::vector<std::pair<float, float>> ref_data = zadehReferenceData();

    SoilProperties soil;
    soil.theta_r = 0.062f;
    soil.theta_s = 0.46f;
    soil.alpha = 3.7f;
    soil.n = 1.67f;
    soil.K_s = 9.3663e-7f;

    uint nz = 100;
    float dz = 0.0035f, h_init = -200.0f, total_time = 86400.0f;

    SoilColumn column(nz, dz, soil);
    column.setInitialCondition(h_init);
    column.setTopBoundary(BCType::DIRICHLET, 0.0f);
    column.setBottomBoundary(BCType::NEUMANN, 0.0f);
    SolverParameters params;
    params.max_picard_iterations = 200;
    params.picard_tolerance = 1e-6f;
    params.picard_omega = 0.7f;
    params.K_weighting = KWeighting::ARITHMETIC;
    params.adaptive_dt = true;
    params.dt_min = 0.001f;
    params.dt_max = 30.0f;
    params.dt_target_rel_change = 0.03f;
    column.setSolverParameters(params);

    std::vector<float> times_days, infil_cm;
    float cumulative = 0.0f;
    float time = 0.0f, dt = 0.01f;
    float output_interval = 0.01f * 86400.0f;
    uint total_steps = 0;

    while (time < total_time) {
        times_days.push_back(time / 86400.0f);
        infil_cm.push_back(cumulative * 100.0f);

        float chunk = std::min(output_interval, total_time - time);
        float prev_storage = column.getTotalWaterStorage();
        float advanced = column.solveAdaptive(chunk, dt);
        total_steps += column.getAdaptiveSolveStats().total_steps;
        cumulative += column.getTotalWaterStorage() - prev_storage;
        time += advanced;
        if (advanced < chunk * 0.99f)
            break; // solver stalled
    }
    times_days.push_back(time / 86400.0f);
    infil_cm.push_back(cumulative * 100.0f);

    float speedup = 86400.0f / static_cast<float>(total_steps);

    float sum_sq_error = 0.0f;
    int n_compare = 0;
    for (const auto &ref: ref_data) {
        size_t idx = 0;
        float min_diff = 1e10f;
        for (size_t i = 0; i < times_days.size(); i++) {
            float diff = std::fabs(times_days[i] - ref.first);
            if (diff < min_diff) {
                min_diff = diff;
                idx = i;
            }
        }
        if (min_diff > 0.05f)
            continue;
        float error = infil_cm[idx] - ref.second;
        sum_sq_error += error * error;
        n_compare++;
    }
    float rmse = (n_compare > 0) ? std::sqrt(sum_sq_error / n_compare) : 0.0f;

    DOCTEST_CHECK(speedup >= 2.0f);
    DOCTEST_CHECK(rmse < 1.0f);
}

// ============================================================================
// SoilGrid - accessors, bounds checking, and spatial queries
// ============================================================================

DOCTEST_TEST_CASE("SoilGrid - accessors and spacing") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(4, 3, 12, 0.5f, 0.25f, 0.1f, props);

    DOCTEST_CHECK(grid.getNx() == 4);
    DOCTEST_CHECK(grid.getNy() == 3);
    DOCTEST_CHECK(grid.getNz() == 12);
    DOCTEST_CHECK(grid.getNumColumns() == 12);
    DOCTEST_CHECK(grid.getDx() == doctest::Approx(0.5f));
    DOCTEST_CHECK(grid.getDy() == doctest::Approx(0.25f));
    DOCTEST_CHECK(grid.getDz() == doctest::Approx(0.1f));
}

DOCTEST_TEST_CASE("SoilGrid - constructor rejects zero dimensions") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    DOCTEST_CHECK_THROWS_WITH_AS(SoilGrid(0, 2, 10, 1.0f, 1.0f, 0.1f, props), doctest::Contains("ERROR (SoilGrid::SoilGrid)"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(SoilGrid(2, 0, 10, 1.0f, 1.0f, 0.1f, props), doctest::Contains("ERROR (SoilGrid::SoilGrid)"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(SoilGrid(2, 2, 0, 1.0f, 1.0f, 0.1f, props), doctest::Contains("ERROR (SoilGrid::SoilGrid)"), std::runtime_error);
}

DOCTEST_TEST_CASE("SoilGrid - column access bounds") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(2, 2, 8, 1.0f, 1.0f, 0.1f, props);

    DOCTEST_SUBCASE("valid indices") {
        DOCTEST_CHECK_NOTHROW(grid.getColumn(0u, 0u));
        DOCTEST_CHECK_NOTHROW(grid.getColumn(1u, 1u));
        DOCTEST_CHECK_NOTHROW(grid.getColumn(3u)); // linear index, last column
    }
    DOCTEST_SUBCASE("out-of-range throws") {
        // Per-axis bounds are enforced: an overflowed i must not wrap into the next row.
        DOCTEST_CHECK_THROWS_WITH_AS(grid.getColumn(2u, 0u), doctest::Contains("column index out of range"), std::runtime_error); // i == nx
        DOCTEST_CHECK_THROWS_WITH_AS(grid.getColumn(0u, 2u), doctest::Contains("column index out of range"), std::runtime_error); // j == ny
        DOCTEST_CHECK_THROWS_WITH_AS(grid.getColumn(1u, 3u), doctest::Contains("column index out of range"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(grid.getColumn(4u), doctest::Contains("column index out of range"), std::runtime_error); // linear overload
    }
}

DOCTEST_TEST_CASE("SoilGrid - initial condition and water storage") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(2, 2, 10, 1.0f, 1.0f, 0.1f, props);

    DOCTEST_SUBCASE("uniform scalar initial condition") {
        grid.setInitialCondition(-0.5f);
        std::vector<float> heads = grid.getAllPressureHeads();
        DOCTEST_CHECK(heads.size() == 2u * 2u * 10u);
        for (float h: heads)
            DOCTEST_CHECK(h == doctest::Approx(-0.5f));

        float storage = grid.getTotalWaterStorage();
        DOCTEST_CHECK(storage > 0.0f);
        DOCTEST_CHECK(std::isfinite(storage));
    }
    DOCTEST_SUBCASE("profile size mismatch throws") {
        std::vector<float> wrong_size(5, -1.0f);
        DOCTEST_CHECK_THROWS_WITH_AS(grid.setInitialCondition(wrong_size), doctest::Contains("ERROR (SoilGrid::setInitialCondition)"), std::runtime_error);
    }
    DOCTEST_SUBCASE("drier soil holds less water") {
        grid.setInitialCondition(-0.2f);
        float wet = grid.getTotalWaterStorage();
        grid.setInitialCondition(-5.0f);
        float dry = grid.getTotalWaterStorage();
        DOCTEST_CHECK(dry < wet);
    }
}

DOCTEST_TEST_CASE("SoilGrid - spatial column lookup") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(4, 4, 10, 0.5f, 0.5f, 0.1f, props);
    grid.setOrigin(-1.0f, -1.0f, 0.0f); // domain spans [-1, 1] x [-1, 1]
    grid.setInitialCondition(-0.5f);

    DOCTEST_SUBCASE("findColumn inside domain") {
        uint i = 99, j = 99;
        bool found = grid.findColumn(-0.75f, -0.75f, i, j);
        DOCTEST_CHECK(found);
        DOCTEST_CHECK(i == 0);
        DOCTEST_CHECK(j == 0);

        found = grid.findColumn(0.75f, 0.75f, i, j);
        DOCTEST_CHECK(found);
        DOCTEST_CHECK(i == 3);
        DOCTEST_CHECK(j == 3);
    }
    DOCTEST_SUBCASE("findColumn outside domain returns false") {
        uint i = 99, j = 99;
        DOCTEST_CHECK(!grid.findColumn(-5.0f, 0.0f, i, j));
        DOCTEST_CHECK(!grid.findColumn(0.0f, 5.0f, i, j));
    }
    DOCTEST_SUBCASE("getColumnPosition returns cell center") {
        float x = 0.0f, y = 0.0f;
        grid.getColumnPosition(0, 0, x, y);
        DOCTEST_CHECK(x == doctest::Approx(-0.75f)); // origin + half-cell
        DOCTEST_CHECK(y == doctest::Approx(-0.75f));
    }
    DOCTEST_SUBCASE("getPressureHeadAtPosition: inside reads the cell, outside raises an error") {
        float h_in = grid.getPressureHeadAtPosition(0.0f, 0.0f, -0.5f);
        DOCTEST_CHECK(h_in == doctest::Approx(-0.5f));
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(grid.getPressureHeadAtPosition(10.0f, 0.0f, -0.5f)), doctest::Contains("outside the grid"), std::runtime_error);
    }
}

DOCTEST_TEST_CASE("SoilGrid - sources and transpiration bounds") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(2, 2, 10, 1.0f, 1.0f, 0.1f, props);

    DOCTEST_SUBCASE("valid source and transpiration") {
        DOCTEST_CHECK_NOTHROW(grid.addSource(0, 0, 0, 1e-6f));
        DOCTEST_CHECK_NOTHROW(grid.setTranspiration(1, 1, 1e-7f));
        DOCTEST_CHECK_NOTHROW(grid.clearSources());
    }
    DOCTEST_SUBCASE("out-of-range column throws") {
        DOCTEST_CHECK_THROWS_WITH_AS(grid.addSource(5, 0, 0, 1e-6f), doctest::Contains("column index out of range"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(grid.setTranspiration(0, 9, 1e-7f), doctest::Contains("column index out of range"), std::runtime_error);
    }
}

// ============================================================================
// SoilTransportModel - sensors, sources, and high-level API
// ============================================================================

DOCTEST_TEST_CASE("SoilTransportModel - GPU flag defaults off on CPU build") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
#ifndef HELIOS_CUDA_AVAILABLE
    DOCTEST_CHECK(model.isGPUEnabled() == false);
#endif
}

DOCTEST_TEST_CASE("SoilTransportModel - sensors record query and bounds") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(2, 2, 10);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.5f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFreeDrainage();

    uint s0 = model.addSensor(make_vec3(0.5f, 0.5f, -0.25f), SoilMeasurement::WATER_CONTENT);
    uint s1 = model.addSensor(make_vec3(0.5f, 0.5f, -0.75f), SoilMeasurement::PRESSURE_HEAD);
    DOCTEST_CHECK(model.getSensorCount() == 2);

    DOCTEST_SUBCASE("recording appends samples and times") {
        model.recordSensors();
        model.run(60.0f);
        model.recordSensors();

        DOCTEST_CHECK(model.getSensorData(s0).size() == 2);
        DOCTEST_CHECK(model.getSensorTimes(s0).size() == 2);
        DOCTEST_CHECK(model.getSensorQuantity(s0) == SoilMeasurement::WATER_CONTENT);
        DOCTEST_CHECK(model.getSensorQuantity(s1) == SoilMeasurement::PRESSURE_HEAD);

        vec3 p = model.getSensorPosition(s1);
        DOCTEST_CHECK(p.z == doctest::Approx(-0.75f));

        // water-content sensor reads a physical theta; head sensor reads near IC
        DOCTEST_CHECK(model.getSensorData(s0).front() > 0.0f);
        DOCTEST_CHECK(model.getSensorData(s0).front() < 1.0f);
        DOCTEST_CHECK(model.getSensorData(s1).front() == doctest::Approx(-0.5f).epsilon(0.2));

        model.clearSensorData();
        DOCTEST_CHECK(model.getSensorData(s0).empty());
    }
    DOCTEST_SUBCASE("invalid sensor id throws") {
        helios::capture_cerr cerr_capture;
        DOCTEST_CHECK_THROWS(model.getSensorData(99));
        DOCTEST_CHECK_THROWS(model.getSensorTimes(99));
        DOCTEST_CHECK_THROWS(model.getSensorPosition(99));
        DOCTEST_CHECK_THROWS(model.getSensorQuantity(99));
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - sensor CSV round-trips to disk") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(2, 2, 8);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.5f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFreeDrainage();
    model.addSensor(make_vec3(0.5f, 0.5f, -0.25f), SoilMeasurement::WATER_CONTENT);
    model.recordSensors();
    model.run(60.0f);
    model.recordSensors();

    DOCTEST_SUBCASE("writes a readable file") {
        const std::string fn = "swt_sensor_test_output.csv";
        std::remove(fn.c_str());
        DOCTEST_CHECK_NOTHROW(model.writeSensorData(fn));

        std::ifstream f(fn);
        DOCTEST_CHECK(f.good());
        std::string header;
        std::getline(f, header);
        DOCTEST_CHECK(header.find("time") != std::string::npos);
        f.close();
        std::remove(fn.c_str());
    }
    DOCTEST_SUBCASE("unwritable path throws") {
        helios::capture_cerr cerr_capture;
        DOCTEST_CHECK_THROWS(model.writeSensorData("/nonexistent_dir_xyz/out.csv"));
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - surface emitters") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(-0.5f, -0.5f, 0.0f), make_vec3(1, 1, 1));
    model.setGridResolution(4, 4, 10);
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-1.0f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFreeDrainage();

    DOCTEST_SUBCASE("emitter inside domain adds water over a run") {
        model.addSurfaceEmitter(0.0f, 0.0f, 1e-6f);
        float before = model.getTotalWaterStorage();
        model.run(300.0f);
        float after = model.getTotalWaterStorage();
        DOCTEST_CHECK(after > before); // emitter injected water
    }
    DOCTEST_SUBCASE("emitter outside domain raises an error") {
        DOCTEST_CHECK_THROWS_WITH_AS(model.addSurfaceEmitter(5.0f, 0.0f, 1e-6f), doctest::Contains("outside the soil domain"), std::runtime_error);
    }
    DOCTEST_SUBCASE("clearSurfaceEmitters is safe and removes injection") {
        model.addSurfaceEmitter(0.0f, 0.0f, 1e-6f);
        DOCTEST_CHECK_NOTHROW(model.clearSurfaceEmitters());
        float before = model.getTotalWaterStorage();
        model.run(300.0f);
        float after = model.getTotalWaterStorage();
        DOCTEST_CHECK(after == doctest::Approx(before).epsilon(1e-3)); // no source, sealed box
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - timestep and storage diagnostics") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(2, 2, 10);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.5f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFreeDrainage();

    SolverParameters sp; // adaptive stepping populates the persistent grid timestep
    sp.adaptive_dt = true;
    sp.dt_max = 120.0f;
    model.setSolverParameters(sp);

    float storage0 = model.getTotalWaterStorage();
    DOCTEST_CHECK(storage0 > 0.0f);
    model.run(120.0f);
    DOCTEST_CHECK(model.getTimestep() > 0.0f); // adaptive controller picked a step
    DOCTEST_CHECK(std::isfinite(model.getTimestep()));
    // Sealed top + free drainage: storage should not increase.
    DOCTEST_CHECK(model.getTotalWaterStorage() <= doctest::Approx(storage0).epsilon(1e-3));
}

// ============================================================================
// RootSoilCoupling - root distributions and uptake
// ============================================================================

DOCTEST_TEST_CASE("RootSoilCoupling - getters for an unknown plant raise errors") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(3, 3, 10, 0.5f, 0.5f, 0.1f, props);
    grid.setInitialCondition(-0.5f);
    RootSoilCoupling coupling(&grid);

    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(coupling.getSoilWaterPotential(42)), doctest::Contains("plant 42 has no root distribution"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(coupling.getCumulativeRootUptake(42)), doctest::Contains("plant 42 has no root distribution"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(coupling.getRootUptakeFlux(42)), doctest::Contains("plant 42 has no root distribution"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(coupling.computeRootUptake(42, 1e-7f), doctest::Contains("plant 42 has no root distribution"), std::runtime_error);
    DOCTEST_CHECK_NOTHROW(coupling.setRootRadialConductivity(1e-8f));
    DOCTEST_CHECK_NOTHROW(coupling.removePlant(42)); // no-op for unknown plant

    // A root zone entirely outside the domain (the grid spans [0, 1.5] in x and y)
    RootDistributionParams far_away;
    far_away.plant_x = 50.0f;
    far_away.plant_y = 50.0f;
    DOCTEST_CHECK_THROWS_WITH_AS(coupling.addPlant(5, far_away), doctest::Contains("no roots inside the soil domain"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(coupling.getSoilWaterPotential(5)), doctest::Contains("plant 5 has no root distribution"), std::runtime_error);
}

DOCTEST_TEST_CASE("RootSoilCoupling - add plant populates root-weighted potential") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(3, 3, 12, 0.5f, 0.5f, 0.1f, props);
    grid.setOrigin(-0.75f, -0.75f, 0.0f);
    grid.setInitialCondition(-0.5f); // uniform head everywhere
    RootSoilCoupling coupling(&grid);

    DOCTEST_SUBCASE("Vrugt-style distribution: weighted potential equals the uniform IC") {
        RootDistributionParams params;
        params.plant_x = 0.0f;
        params.plant_y = 0.0f;
        coupling.addPlant(7, params);
        // With a uniform head field, the root-weighted soil-water potential must
        // equal that head exactly (it is a convex combination of identical values).
        grid.setInitialCondition(-0.3f);
        float psi = coupling.getSoilWaterPotential(7);
        DOCTEST_CHECK(psi == doctest::Approx(-0.3f));
    }
    DOCTEST_SUBCASE("Weibull depth-only distribution") {
        WeibullRootParameters wp; // radial=false (1D depth profile)
        coupling.setWeibullDistribution(8, wp);
        DOCTEST_CHECK(coupling.getSoilWaterPotential(8) == doctest::Approx(-0.5f));
    }
    DOCTEST_SUBCASE("Weibull 3D distribution is accepted") {
        WeibullRootParameters wp;
        wp.radial = true;
        wp.r_max = 0.5f;
        coupling.setWeibullDistribution(9, wp);
        DOCTEST_CHECK(coupling.getSoilWaterPotential(9) == doctest::Approx(-0.5f));
        DOCTEST_CHECK_NOTHROW(coupling.computeAllRootUptake());
    }
}

DOCTEST_TEST_CASE("RootSoilCoupling - uptake removes water from the soil") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");

    // Two identical grids: one with root uptake, one without. Solve both for the
    // same interval and confirm the rooted grid loses more water.
    auto storage_after_solve = [&](bool with_roots) {
        SoilGrid grid(2, 2, 12, 0.5f, 0.5f, 0.1f, props);
        grid.setInitialCondition(-0.5f);
        grid.setTopBoundary(BCType::NEUMANN, 0.0f); // sealed top
        grid.setBottomBoundary(BCType::NEUMANN, 0.0f); // sealed bottom -> closed box
        RootSoilCoupling coupling(&grid);
        if (with_roots) {
            RootDistributionParams params;
            coupling.addPlant(1, params);
            coupling.computeRootUptake(1, 1e-6f); // transpiration sink
        }
        for (int step = 0; step < 5; step++)
            grid.solve(60.0f);
        return grid.getTotalWaterStorage();
    };

    float sealed = storage_after_solve(false);
    float rooted = storage_after_solve(true);
    DOCTEST_CHECK(rooted < sealed); // roots extracted water
    DOCTEST_CHECK(sealed == doctest::Approx(storage_after_solve(false))); // closed box conserves
}

// ============================================================================
// SoilGrid - hydrostatic IC, per-column properties, quasi-3D coupling
// ============================================================================

DOCTEST_TEST_CASE("SoilGrid - hydrostatic initial condition wets with depth") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(1, 1, 20, 1.0f, 1.0f, 0.1f, props); // 2 m deep column
    grid.setInitialConditionHydrostatic(1.0f); // water table 1 m down

    float theta_top = grid.getWaterContent(0, 0, 0); // surface layer
    float theta_bottom = grid.getWaterContent(0, 0, 19); // deepest layer
    DOCTEST_CHECK(theta_bottom > theta_top); // wetter at depth
    // Head should increase (less negative) monotonically downward in hydrostatic equilibrium.
    DOCTEST_CHECK(grid.getPressureHead(0, 0, 19) > grid.getPressureHead(0, 0, 0));
}

DOCTEST_TEST_CASE("SoilGrid - water content is consistent with pressure head") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(2, 2, 8, 0.5f, 0.5f, 0.1f, props);
    grid.setInitialCondition(-0.4f);
    // theta computed from the cell head must match the constitutive curve.
    float h = grid.getPressureHead(1, 1, 3);
    float theta = grid.getWaterContent(1, 1, 3);
    DOCTEST_CHECK(theta == doctest::Approx(h_to_theta(h, props)));
    DOCTEST_CHECK(theta > props.theta_r);
    DOCTEST_CHECK(theta < props.theta_s);
}

DOCTEST_TEST_CASE("SoilGrid - per-column soil properties take effect") {
    SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilProperties sand = getSoilPropertiesFromTexture("sand");
    SoilGrid grid(2, 1, 10, 0.5f, 0.5f, 0.1f, loam);
    grid.setSoilProperties(1, 0, sand); // column (1,0) becomes sand
    grid.setInitialCondition(-1.0f); // same head, different texture

    // At the same tension, sand (lower air-entry) holds much less water than loam.
    float theta_loam = grid.getWaterContent(0, 0, 5);
    float theta_sand = grid.getWaterContent(1, 0, 5);
    DOCTEST_CHECK(theta_sand < theta_loam);
}

DOCTEST_TEST_CASE("SoilGrid - quasi-3D horizontal coupling spreads water laterally") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");

    auto run_row = [&](bool couple) {
        SoilGrid grid(3, 1, 10, 0.2f, 0.2f, 0.1f, props);
        grid.setInitialCondition(-2.0f); // dry row
        grid.getColumn(1u, 0u).setInitialCondition(-0.1f); // wet center column
        grid.setTopBoundary(BCType::NEUMANN, 0.0f);
        grid.setBottomBoundary(BCType::NEUMANN, 0.0f); // closed box
        if (couple) {
            grid.enableHorizontalCoupling(1);
            grid.setSplittingType(SplittingType::SEQUENTIAL);
            for (int s = 0; s < 20; s++)
                grid.solveQuasi3D(120.0f);
        } else {
            grid.disableHorizontalCoupling();
            for (int s = 0; s < 20; s++)
                grid.solve(120.0f); // independent columns
        }
        return std::make_pair(grid.getColumn(0u, 0u).getTotalWaterStorage(), // dry neighbor
                              grid.getColumn(1u, 0u).getTotalWaterStorage()); // wet center
    };

    auto coupled = run_row(true);
    auto uncoupled = run_row(false);

    // With coupling the dry neighbor gains water and the wet center loses it.
    DOCTEST_CHECK(coupled.first > uncoupled.first); // neighbor wetter when coupled
    DOCTEST_CHECK(coupled.second < uncoupled.second); // center drier when coupled
    // Without coupling the columns are independent: the dry neighbor stays ~dry.
    DOCTEST_CHECK(uncoupled.first == doctest::Approx(run_row(false).first));
}

DOCTEST_TEST_CASE("SoilGrid - uniform boundary setters reach every column") {
    SoilProperties props = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(2, 2, 10, 0.5f, 0.5f, 0.1f, props);
    grid.setInitialCondition(-1.0f);
    grid.setTopBoundary(BCType::NEUMANN, 1e-6f); // infiltration on all columns
    grid.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
    float before = grid.getTotalWaterStorage();
    for (int s = 0; s < 10; s++)
        grid.solve(60.0f);
    DOCTEST_CHECK(grid.getTotalWaterStorage() > before); // every column infiltrated
}

// ============================================================================
// SoilTransportModel - 3D mass balance, boundary modes, roots, evaporation
// ============================================================================

DOCTEST_TEST_CASE("SoilTransportModel - adaptive steps conserve the water of an emitter switched on and off") {
    // A closed loam block: an emitter runs for 6 h, then is switched off. With adaptive steps, the water stored must rise by
    // exactly what the emitter delivered: the step sizes change at every switch, which a variable-step BDF2 integrator does
    // not conserve (it lost ~1.6% of the water here).
    for (bool gpu: solverBackends()) {
        CAPTURE(gpu);
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        model.defineSoilDomain(make_vec3(-0.6f, -0.6f, 0.f), make_vec3(1.2f, 1.2f, 0.8f));
        model.setGridResolution(24, 24, 16);
        model.setSolverType(SolverType::SOLVER_3D);
        model.setSoilPropertiesFromTexture("loam");
        model.setInitialCondition(-3.0f);
        model.setTopBoundaryFlux(0.f);
        model.setBottomBoundaryFlux(0.f);
        SolverParameters params;
        params.adaptive_dt = true;
        params.dt_max = 1800.f;
        params.dt_min = 1.f;
        model.setSolverParameters(params);
        if (gpu)
            model.enableGPU();
        const double W0 = model.getTotalWaterStorage();
        const float q = 2.0e-3f / 3600.0f, dt = 1800.f;
        double water_in = 0.0;
        for (int n = 0; n < 48; n++) {
            const float t = n * dt / 3600.f;
            const bool on = t >= 2.f && t < 8.f;
            model.clearSurfaceEmitters();
            if (on)
                model.addSurfaceEmitter(0.f, 0.f, q, 0.08f);
            model.run(dt);
            water_in += on ? q * dt : 0.0;
        }
        DOCTEST_CHECK((model.getTotalWaterStorage() - W0) == doctest::Approx(water_in).epsilon(1e-5));
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - 3D infiltration conserves mass") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(4, 4, 20);
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-1.0f);
    model.setTopBoundaryFlux(2e-6f); // steady infiltration
    model.setBottomBoundaryFreeDrainage();

    SolverParameters sp; // adaptive stepping exercises solveAdaptive + the iteration counter
    sp.adaptive_dt = true;
    sp.dt_max = 60.0f;
    model.setSolverParameters(sp);

    const double storage0 = model.getTotalWaterStorage();
    for (int s = 0; s < 10; s++)
        model.run(60.0f);
    const double storage1 = model.getTotalWaterStorage();

    const double influx = model.getCumulativeBoundaryInflux();
    DOCTEST_CHECK(influx > 0.0); // water entered through the top
    // Backward Euler conserves water to the solver tolerance: the storage gain equals the net boundary influx.
    DOCTEST_CHECK((storage1 - storage0) == doctest::Approx(influx).epsilon(1e-6));
    DOCTEST_CHECK(std::abs(model.getMassBalanceResidual()) < 1e-6 * influx);
    DOCTEST_CHECK(model.getSolverIterationCount() > 0); // Picard iterations were counted
}

DOCTEST_TEST_CASE("SoilTransportModel - Dirichlet top pressure drives infiltration") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(3, 3, 15);
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-1.0f);
    model.setTopBoundaryPressure(0.0f); // saturated (ponded) surface
    model.setBottomBoundaryFreeDrainage();

    float wc_surface_before = model.getWaterContent(make_vec3(0.5f, 0.5f, -0.05f));
    for (int s = 0; s < 10; s++)
        model.run(60.0f);
    float wc_surface_after = model.getWaterContent(make_vec3(0.5f, 0.5f, -0.05f));
    DOCTEST_CHECK(wc_surface_after > wc_surface_before); // surface wetted toward saturation
    DOCTEST_CHECK(model.getCumulativeBoundaryInflux() > 0.0f);
}

DOCTEST_TEST_CASE("SoilTransportModel - hydrostatic IC wets with depth") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 2));
    model.setGridResolution(2, 2, 20);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialConditionHydrostatic(1.5f);

    float wc_shallow = model.getWaterContent(make_vec3(0.5f, 0.5f, -0.1f));
    float wc_deep = model.getWaterContent(make_vec3(0.5f, 0.5f, -1.8f));
    DOCTEST_CHECK(wc_deep > wc_shallow);
}

DOCTEST_TEST_CASE("SoilTransportModel - quasi-3D mode runs with horizontal coupling") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setColumnGrid(3, 3);
    model.setColumnResolution(12);
    model.setSolverType(SolverType::SOLVER_QUASI_3D);
    model.enableHorizontalCoupling(2);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.8f);
    model.setTopBoundaryFlux(1e-6f);
    model.setBottomBoundaryFreeDrainage();

    float before = model.getTotalWaterStorage();
    for (int s = 0; s < 5; s++)
        model.run(60.0f);
    DOCTEST_CHECK(model.getTotalWaterStorage() > before);
}

DOCTEST_TEST_CASE("SoilTransportModel - root transpiration removes stored water") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(-0.5f, -0.5f, 0.0f), make_vec3(1, 1, 1.5f));
    model.setGridResolution(3, 3, 15);
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.5f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFreeDrainage();

    RootDistributionParams roots;
    model.setRootDistribution(1, roots);
    model.setRootRadialConductivity(1e-8f);
    model.enableRootUptake();
    model.setTranspiration(1, 5e-7f);

    // The root-weighted soil-water potential reflects the (uniform) initial head.
    DOCTEST_CHECK(model.getSoilWaterPotential(1) == doctest::Approx(-0.5f).epsilon(0.1));

    float before = model.getTotalWaterStorage();
    for (int s = 0; s < 10; s++)
        model.run(60.0f);
    DOCTEST_CHECK(model.getTotalWaterStorage() < before); // transpiration drew water down
}

DOCTEST_TEST_CASE("SoilTransportModel - surface evaporation requires the 3D solver") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(3, 3, 12);
    model.setSoilPropertiesFromTexture("loam");

    DOCTEST_SUBCASE("column-grid mode raises an error") {
        model.setSolverType(SolverType::SOLVER_1D);
        model.setInitialCondition(-0.3f);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.setSurfaceEvaporation(1e-7f)), doctest::Contains("only supported by the full 3D solver"), std::runtime_error);
    }
    DOCTEST_SUBCASE("3D mode returns a positive rate and dries the surface") {
        model.setSolverType(SolverType::SOLVER_3D);
        model.setInitialCondition(-0.3f);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryFreeDrainage();
        float rate = model.setSurfaceEvaporation(5e-7f);
        DOCTEST_CHECK(rate > 0.0f); // actual evaporation flux
        float before = model.getTotalWaterStorage();
        for (int s = 0; s < 10; s++)
            model.run(60.0f);
        DOCTEST_CHECK(model.getTotalWaterStorage() < before);
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - GPU toggle on a CPU build raises an error") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(2, 2, 8);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.5f);

    model.disableGPU();
    DOCTEST_CHECK(model.isGPUEnabled() == false);
#ifndef HELIOS_CUDA_AVAILABLE
    DOCTEST_CHECK_THROWS_WITH_AS(model.enableGPU(), doctest::Contains("compiled without CUDA"), std::runtime_error);
    DOCTEST_CHECK(model.isGPUEnabled() == false);
#endif
}

DOCTEST_TEST_CASE("SoilTransportModel - visualization tags cutaway cross-section faces") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(4, 4, 4);
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.5f);
    model.initializeDomain();

    auto countCut = [&](const std::vector<uint> &UUIDs) {
        DOCTEST_REQUIRE(!UUIDs.empty());
        for (uint UUID: UUIDs)
            DOCTEST_REQUIRE(context.doesPrimitiveDataExist(UUID, "soil_cut_face"));
        return context.filterPrimitivesByData(UUIDs, "soil_cut_face", 1, "==").size();
    };

    DOCTEST_SUBCASE("full domain has no cut faces") {
        std::vector<uint> UUIDs = model.addSoilVisualization(SoilColorMode::WATER_CONTENT);
        DOCTEST_CHECK(countCut(UUIDs) == 0);
    }
    DOCTEST_SUBCASE("corner cutaway exposes only the notch walls") {
        // Remove the x in [0.5,1], y in [0,0.5] quarter column (full depth)
        std::vector<uint> UUIDs = model.addSoilVisualization(SoilColorMode::WATER_CONTENT, make_vec3(0.5f, 0, 0), make_vec3(1, 0.5f, 1), SoilRegion::REMOVE);
        std::vector<uint> cut = context.filterPrimitivesByData(UUIDs, "soil_cut_face", 1, "==");
        DOCTEST_CHECK(countCut(UUIDs) > 0);
        DOCTEST_CHECK(cut.size() < UUIDs.size());
        for (uint UUID: cut) {
            vec3 c = make_vec3(0, 0, 0);
            for (const vec3 &v: context.getPrimitiveVertices(UUID))
                c = c + v * 0.25f;
            // Every cut face lies on an interior plane (x = 0.5 or y = 0.5), never on the domain boundary
            const bool on_x = std::fabs(c.x - 0.5f) < 1e-4f, on_y = std::fabs(c.y - 0.5f) < 1e-4f;
            DOCTEST_CHECK((on_x || on_y));
        }
    }
    DOCTEST_SUBCASE("cross-section slice is a cut face") {
        std::vector<uint> UUIDs = model.addSoilVisualization(SoilColorMode::PRESSURE_HEAD, make_vec3(0.5f, 0, 0), make_vec3(0.5f, 1, 1), SoilRegion::KEEP);
        DOCTEST_CHECK(countCut(UUIDs) == UUIDs.size());
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - saturated bottom boundary wets the column from below") {
    for (SolverType type: {SolverType::SOLVER_1D, SolverType::SOLVER_3D}) {
        DOCTEST_CAPTURE(static_cast<int>(type));
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
        model.setGridResolution(2, 2, 10);
        model.setSolverType(type);
        model.setSoilPropertiesFromTexture("loam");
        model.setInitialCondition(-1.0f);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryPressure(0.0f); // water table at the bottom face

        const vec3 deep = make_vec3(0.5f, 0.5f, -0.95f);
        const float h_before = model.getPressureHead(deep);
        const float storage_before = model.getTotalWaterStorage();
        for (int s = 0; s < 12; s++)
            model.run(3600.0f);

        DOCTEST_CHECK(model.getTotalWaterStorage() > storage_before); // water enters through the bottom
        DOCTEST_CHECK(model.getPressureHead(deep) > h_before); // bottom cell moves toward h = 0
        DOCTEST_CHECK(model.getPressureHead(deep) <= 1e-3f);
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - Weibull root distribution drives transpiration uptake") {
    for (bool radial: {false, true}) {
        DOCTEST_CAPTURE(radial);
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        model.defineSoilDomain(make_vec3(-0.5f, -0.5f, 0.0f), make_vec3(1, 1, 1.0f));
        model.setGridResolution(4, 4, 10);
        model.setSolverType(SolverType::SOLVER_3D);
        model.setSoilPropertiesFromTexture("loam");
        model.setInitialCondition(-0.5f);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryFlux(0.0f); // sealed box: only the roots remove water

        WeibullRootParameters roots;
        roots.z_max = 0.8f;
        roots.radial = radial;
        roots.r_max = 0.5f;
        model.setRootDistributionWeibull(1, roots);
        model.setRootRadialConductivity(1e-8f);
        model.enableRootUptake();
        model.setTranspiration(1, 2e-7f);

        DOCTEST_CHECK(model.getSoilWaterPotential(1) == doctest::Approx(-0.5f).epsilon(0.1));

        const float storage_before = model.getTotalWaterStorage();
        for (int s = 0; s < 10; s++)
            model.run(600.0f);
        const float removed = storage_before - model.getTotalWaterStorage();

        DOCTEST_CHECK(removed > 0.0f);
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) > 0.0f);
        // Sealed box: storage loss = uptake. getCumulativeRootUptake is a depth [m] over the 1 m^2 footprint.
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) * 1.0f == doctest::Approx(removed).scale(0).epsilon(0.05));
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) == doctest::Approx(2e-7f * 6000.0f).scale(0).epsilon(1e-4)); // T * total time
        // The root-zone potential must follow the 3D solution as the roots dry the soil (it read stale column heads before)
        DOCTEST_CHECK(model.getSoilWaterPotential(1) < -0.5f);
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - two plants with overlapping roots both take up water") {
    // Sealed box, two plants rooted in the same cells: the soil must lose the water both plants report taking up.
    for (SolverType solver: {SolverType::SOLVER_3D, SolverType::SOLVER_1D}) {
        DOCTEST_CAPTURE(static_cast<int>(solver));
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        model.defineSoilDomain(make_vec3(-0.5f, -0.5f, 0.0f), make_vec3(1, 1, 1.0f));
        model.setGridResolution(4, 4, 10);
        model.setSolverType(solver);
        model.setSoilPropertiesFromTexture("loam");
        model.setInitialCondition(-0.5f);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryFlux(0.0f); // sealed box: only the roots remove water

        WeibullRootParameters roots;
        roots.z_max = 0.8f;
        roots.radial = true;
        roots.r_max = 0.5f;
        model.setRootDistributionWeibull(1, roots);
        model.setRootDistributionWeibull(2, roots);
        model.setRootRadialConductivity(1e-8f);
        model.enableRootUptake();
        model.setTranspiration(1, 2e-7f);
        model.setTranspiration(2, 3e-7f);

        const float storage_before = model.getTotalWaterStorage();
        for (int s = 0; s < 10; s++)
            model.run(600.0f);
        const float removed = storage_before - model.getTotalWaterStorage();

        const float uptake_1 = model.getCumulativeRootUptake(1);
        const float uptake_2 = model.getCumulativeRootUptake(2);
        DOCTEST_CHECK(uptake_1 == doctest::Approx(2e-7f * 6000.0f).scale(0).epsilon(1e-4));
        DOCTEST_CHECK(uptake_2 == doctest::Approx(3e-7f * 6000.0f).scale(0).epsilon(1e-4));
        // getCumulativeRootUptake is a depth [m] over the 1 m^2 footprint
        DOCTEST_CHECK((uptake_1 + uptake_2) == doctest::Approx(removed).scale(0).epsilon(0.02));
    }
}

// ============================================================================
// SoilTransportModel - state is kept, and failures raise errors instead of returning placeholder values
// ============================================================================

namespace {
    //! A small model of loam on a domain whose top surface is at z = 0 and whose x-y footprint is [0,1] x [0,1]
    void setupSmallModel(SoilTransportModel &model, SolverType solver, uint nx = 2, uint ny = 2, uint nz = 10) {
        model.disableMessages();
        model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
        model.setGridResolution(nx, ny, nz);
        model.setSolverType(solver);
        model.setSoilPropertiesFromTexture("loam");
    }
} // namespace

DOCTEST_TEST_CASE("SoilTransportModel - changing the soil texture keeps the initial and boundary conditions") {
    for (SolverType solver: {SolverType::SOLVER_1D, SolverType::SOLVER_3D}) {
        DOCTEST_CAPTURE(static_cast<int>(solver));
        Context context;
        SoilTransportModel model(&context);
        setupSmallModel(model, solver);
        model.setInitialCondition(-2.0f);
        const float inflow = 1e-6f; // [m/s]
        model.setTopBoundaryFlux(inflow);
        model.setBottomBoundaryFlux(0.0f); // sealed bottom: storage gains exactly the inflow
        model.setSoilPropertiesFromTexture("sandy_loam");

        const vec3 point = make_vec3(0.25f, 0.25f, -0.55f);
        DOCTEST_CHECK(model.getPressureHead(point) == doctest::Approx(-2.0f));
        const float before = model.getTotalWaterStorage();
        model.run(3600.0f);
        DOCTEST_CHECK(model.getTotalWaterStorage() - before == doctest::Approx(inflow * 3600.0f).epsilon(0.01));
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - grid and solver changes after setup raise an error") {
    Context context;
    SoilTransportModel model(&context);
    setupSmallModel(model, SolverType::SOLVER_1D);
    model.setInitialCondition(-1.0f); // builds the domain

    DOCTEST_CHECK_THROWS_WITH_AS(model.setGridResolution(4, 4, 10), doctest::Contains("already been set up"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.setColumnResolution(12), doctest::Contains("already been set up"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.setColumnGrid(3, 3), doctest::Contains("already been set up"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(2, 2, 1)), doctest::Contains("already been set up"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.setSolverType(SolverType::SOLVER_3D), doctest::Contains("already been set up"), std::runtime_error);
    // Switching between the column solvers does not change the grid
    DOCTEST_CHECK_NOTHROW(model.setSolverType(SolverType::SOLVER_QUASI_3D));
    DOCTEST_CHECK(model.getPressureHead(make_vec3(0.25f, 0.25f, -0.55f)) == doctest::Approx(-1.0f));
}

DOCTEST_TEST_CASE("SoilTransportModel - invalid queries raise errors instead of returning placeholder values") {
    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(getSoilPropertiesFromTexture("chalk")), doctest::Contains("Unknown soil texture"), std::runtime_error);

    for (SolverType solver: {SolverType::SOLVER_1D, SolverType::SOLVER_3D}) {
        DOCTEST_CAPTURE(static_cast<int>(solver));
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        DOCTEST_CHECK_THROWS_WITH_AS(model.setSoilPropertiesFromTexture("chalk"), doctest::Contains("Unknown soil texture"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getTotalWaterStorage()), doctest::Contains("not been set up"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getWaterContent(make_vec3(0.5f, 0.5f, -0.5f))), doctest::Contains("not been set up"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.addSoilVisualization(SoilColorMode::WATER_CONTENT)), doctest::Contains("not been set up"), std::runtime_error);

        setupSmallModel(model, solver);
        model.setInitialCondition(-1.0f);
        const vec3 outside = make_vec3(0.5f, 0.5f, -1.5f); // below the 1 m deep domain
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getWaterContent(outside)), doctest::Contains("outside the soil domain"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getPressureHead(outside)), doctest::Contains("outside the soil domain"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.addSensor(outside, SoilMeasurement::WATER_CONTENT)), doctest::Contains("outside the soil domain"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(model.addSurfaceEmitter(2.0f, 0.5f, 1e-6f, 0.1f), doctest::Contains("outside the soil domain"), std::runtime_error);

        // Root coupling: unknown plants and plants without roots
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getSoilWaterPotential(7)), doctest::Contains("plant 7"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getCumulativeRootUptake(7)), doctest::Contains("plant 7"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getRootUptakeFlux(7)), doctest::Contains("plant 7"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(model.setTranspiration(7, 1e-7f), doctest::Contains("plant 7"), std::runtime_error);

        if (solver == SolverType::SOLVER_1D) {
            DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getCumulativeBoundaryInflux()), doctest::Contains("SOLVER_3D"), std::runtime_error);
            DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.getMassBalanceResidual()), doctest::Contains("SOLVER_3D"), std::runtime_error);
            DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(model.setSurfaceEvaporation(1e-8f)), doctest::Contains("SOLVER_3D"), std::runtime_error);
        }
    }

#ifndef HELIOS_CUDA_AVAILABLE
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    DOCTEST_CHECK(SoilTransportModel::isGPUAvailable() == false);
    DOCTEST_CHECK_THROWS_WITH_AS(model.enableGPU(), doctest::Contains("CUDA"), std::runtime_error);
#endif
}

DOCTEST_TEST_CASE("SoilTransportModel - a step that cannot be completed raises an error and leaves the state unchanged") {
    // Evaporation far beyond what dry loam can deliver: the step has no solution at any sub-step length.
    for (SolverType solver: {SolverType::SOLVER_1D, SolverType::SOLVER_3D}) {
        for (bool adaptive: {false, true}) {
            DOCTEST_CAPTURE(static_cast<int>(solver));
            DOCTEST_CAPTURE(adaptive);
            Context context;
            SoilTransportModel model(&context);
            setupSmallModel(model, solver, 1, 1, 10);
            SolverParameters params;
            params.adaptive_dt = adaptive;
            params.dt_min = 600.0f;
            params.dt_max = 86400.0f;
            model.setSolverParameters(params);
            model.setInitialCondition(-100.0f);
            model.setTopBoundaryFlux(-1e-4f);
            model.setBottomBoundaryFlux(0.0f);
            const uint sensor = model.addSensor(make_vec3(0.5f, 0.5f, -0.05f), SoilMeasurement::PRESSURE_HEAD);

            const float storage_before = model.getTotalWaterStorage();
            DOCTEST_CHECK_THROWS_WITH_AS(model.run(86400.0f), doctest::Contains("could not"), std::runtime_error);
            DOCTEST_CHECK(model.getTotalWaterStorage() == doctest::Approx(storage_before).epsilon(1e-6));
            DOCTEST_CHECK(model.getPressureHead(make_vec3(0.5f, 0.5f, -0.05f)) == doctest::Approx(-100.0f).epsilon(1e-6));
            model.recordSensors();
            DOCTEST_CHECK(model.getSensorTimes(sensor).back() == doctest::Approx(0.0f)); // the clock did not advance
        }
    }
}

DOCTEST_TEST_CASE("SoilSolver3D and SoilColumn - a failed Picard step leaves the state unchanged") {
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    {
        SoilSolver3D s(1, 1, 10, 1.0, 1.0, 0.1, loam);
        SoilSolver3D::Params p = s.getParams();
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::PICARD;
        p.picard_max = 3;
        s.setParams(p);
        s.setInitialCondition(-100.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, -1e-4);
        s.setBoundary(Face3D::ZMIN, BCType::NEUMANN, 0.0);
        const std::vector<double> before = s.getPressureHeads();
        DOCTEST_CHECK(s.solve(86400.0) == 0);
        DOCTEST_CHECK(s.getPressureHeads() == before);
    }
    {
        SoilColumn column(10, 0.1f, loam);
        SolverParameters cp;
        cp.nonlinear_method = NonlinearMethod::PICARD;
        cp.max_picard_iterations = 3;
        column.setSolverParameters(cp);
        column.setInitialCondition(-100.0f);
        column.setTopBoundary(BCType::NEUMANN, -1e-4f);
        column.setBottomBoundary(BCType::NEUMANN, 0.0f);
        const std::vector<float> before = column.getPressureHeads();
        DOCTEST_CHECK(column.solve(86400.0f) == 0);
        DOCTEST_CHECK(column.getPressureHeads() == before);
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - points on the domain boundary read the boundary cells") {
    for (SolverType solver: {SolverType::SOLVER_1D, SolverType::SOLVER_3D}) {
        DOCTEST_CAPTURE(static_cast<int>(solver));
        Context context;
        SoilTransportModel model(&context);
        setupSmallModel(model, solver);
        model.setInitialCondition(-1.0f);
        const float theta = h_to_theta(-1.0f, getSoilPropertiesFromTexture("loam"));
        // The top surface, the far x-y corner, and the bottom of the domain
        for (const vec3 &point: {make_vec3(0.25f, 0.25f, 0.0f), make_vec3(1.0f, 1.0f, -0.5f), make_vec3(0.25f, 0.25f, -1.0f)}) {
            DOCTEST_CAPTURE(point);
            DOCTEST_CHECK(model.getWaterContent(point) == doctest::Approx(theta));
            DOCTEST_CHECK(model.getPressureHead(point) == doctest::Approx(-1.0f));
        }
        const uint surface_sensor = model.addSensor(make_vec3(0.25f, 0.25f, 0.0f), SoilMeasurement::WATER_CONTENT);
        model.recordSensors();
        DOCTEST_CHECK(model.getSensorData(surface_sensor).back() == doctest::Approx(theta));
    }
}

DOCTEST_TEST_CASE("SoilTransportModel - water content and evaporation use each cell's own soil") {
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    const SoilProperties sand = getSoilPropertiesFromTexture("sand");
    for (SolverType solver: {SolverType::SOLVER_1D, SolverType::SOLVER_3D}) {
        DOCTEST_CAPTURE(static_cast<int>(solver));
        Context context;
        SoilTransportModel model(&context);
        setupSmallModel(model, solver, 2, 1, 10);
        model.setInitialCondition(-1.0f);
        model.setSoilProperties(1, 0, sand); // the column at x in [0.5, 1]

        DOCTEST_CHECK(model.getWaterContent(make_vec3(0.25f, 0.5f, -0.5f)) == doctest::Approx(h_to_theta(-1.0f, loam)));
        DOCTEST_CHECK(model.getWaterContent(make_vec3(0.75f, 0.5f, -0.5f)) == doctest::Approx(h_to_theta(-1.0f, sand)));

        if (solver == SolverType::SOLVER_3D) {
            const float potential = 5e-8f; // [m/s]
            const float expected = 0.5f * potential * (surfaceEvaporationFactor(h_to_theta(-1.0f, loam), loam) + surfaceEvaporationFactor(h_to_theta(-1.0f, sand), sand));
            DOCTEST_CHECK(model.setSurfaceEvaporation(potential) == doctest::Approx(expected));
        }
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - M-scheme follows a change of soil after its first step") {
    // M_SCHEME tabulates its Kirchhoff transform for the (homogeneous) soil; a solver whose soil was changed must give
    // the same step as one built with the new soil.
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    const SoilProperties sand = getSoilPropertiesFromTexture("sand");
    auto configure = [](SoilSolver3D &s) {
        SoilSolver3D::Params p = s.getParams();
        p.nonlinear_method = SoilSolver3D::NonlinearMethod::M_SCHEME;
        s.setParams(p);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 1e-6);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE);
    };
    SoilSolver3D changed(2, 2, 10, 0.1, 0.1, 0.1, loam);
    configure(changed);
    changed.setInitialCondition(-1.0);
    DOCTEST_REQUIRE(changed.solve(600.0) > 0);
    for (uint k = 0; k < 10; k++)
        for (uint j = 0; j < 2; j++)
            for (uint i = 0; i < 2; i++)
                changed.setSoilProperties(i, j, k, sand);
    changed.setInitialCondition(-1.0);
    changed.resetMassBalance();
    DOCTEST_REQUIRE(changed.solve(600.0) > 0);

    SoilSolver3D fresh(2, 2, 10, 0.1, 0.1, 0.1, sand);
    configure(fresh);
    fresh.setInitialCondition(-1.0);
    DOCTEST_REQUIRE(fresh.solve(600.0) > 0);

    double max_difference = 0.0;
    for (size_t c = 0; c < fresh.getPressureHeads().size(); c++)
        max_difference = std::max(max_difference, std::fabs(changed.getPressureHeads()[c] - fresh.getPressureHeads()[c]));
    DOCTEST_CHECK(max_difference < 1e-6);
}

// ============================================================================
// Input validation, CPU/GPU consistency and precision
// ============================================================================

DOCTEST_TEST_CASE("SoilColumn, SoilGrid and SoilSolver3D - invalid sizes and timesteps raise errors") {
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    DOCTEST_CHECK_THROWS_WITH_AS(SoilColumn(1, 0.1f, loam), doctest::Contains("at least 2 layers"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(SoilColumn(std::vector<float>{0.1f}, loam), doctest::Contains("at least 2 layers"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(SoilColumn(10, 0.0f, loam), doctest::Contains("thickness"), std::runtime_error);

    SoilColumn column(10, 0.1f, loam);
    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(column.solve(0.0f)), doctest::Contains("positive"), std::runtime_error);

    SoilGrid grid(2, 2, 10, 0.5f, 0.5f, 0.1f, loam);
    DOCTEST_CHECK_THROWS_WITH_AS(grid.enableHorizontalCoupling(0), doctest::Contains("at least 1"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(grid.solve(-1.0f)), doctest::Contains("positive"), std::runtime_error);

    SoilSolver3D solver(2, 2, 10, 0.5, 0.5, 0.1, loam);
    DOCTEST_CHECK_THROWS_WITH_AS(static_cast<void>(solver.solve(0.0)), doctest::Contains("positive"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(SoilSolver3D(2, 2, 10, 0.5, 0.0, 0.1, loam), doctest::Contains("spacing"), std::runtime_error);

    // The solver owns GPU memory through a raw pointer, so it must not be copied
    DOCTEST_CHECK(!std::is_copy_constructible<SoilSolver3D>::value);
    DOCTEST_CHECK(!std::is_copy_assignable<SoilSolver3D>::value);
}

DOCTEST_TEST_CASE("Soil properties - non-physical parameters raise errors") {
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    std::vector<std::pair<std::string, SoilProperties>> invalid;
    auto with = [&](const std::string &name, auto change) {
        SoilProperties p = loam;
        change(p);
        invalid.emplace_back(name, p);
    };
    with("theta_s <= theta_r", [](SoilProperties &p) { p.theta_s = p.theta_r; });
    with("theta_r < 0", [](SoilProperties &p) { p.theta_r = -0.01f; });
    with("theta_s > 1", [](SoilProperties &p) { p.theta_s = 1.2f; });
    with("alpha <= 0", [](SoilProperties &p) { p.alpha = 0.0f; });
    with("n <= 1", [](SoilProperties &p) { p.n = 1.0f; });
    with("K_s <= 0", [](SoilProperties &p) { p.K_s = 0.0f; });
    with("kr_spline_start", [](SoilProperties &p) { p.kr_spline_start = 0.0f; });
    with("h_e", [](SoilProperties &p) {
        p.model = SoilModel::MODIFIED_VAN_GENUCHTEN;
        p.h_e = 0.0f;
    });
    for (const auto &entry: invalid) {
        const std::string &name = entry.first;
        const SoilProperties &props = entry.second;
        DOCTEST_CAPTURE(name);
        DOCTEST_CHECK_THROWS_WITH_AS(SoilColumn(10, 0.1f, props), doctest::Contains("soil properties"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(SoilSolver3D(2, 2, 10, 0.5, 0.5, 0.1, props), doctest::Contains("soil properties"), std::runtime_error);
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        DOCTEST_CHECK_THROWS_WITH_AS(model.setSoilProperties(props), doctest::Contains("soil properties"), std::runtime_error);
    }
    // Gardner and linear soils do not use n
    DOCTEST_CHECK_NOTHROW(SoilColumn(10, 0.1f, SoilProperties::gardner(0.05f, 0.4f, 2.0f, 1e-6f)));
    DOCTEST_CHECK_NOTHROW(SoilColumn(10, 0.1f, SoilProperties::linear(0.05f, 0.4f, 0.5f, 1e-6f)));
}

DOCTEST_TEST_CASE("SoilTransportModel - invalid domain, emitter and solver settings raise errors") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    DOCTEST_CHECK_THROWS_WITH_AS(model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 0, 1)), doctest::Contains("extent"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.setGridResolution(0, 2, 10), doctest::Contains("at least 1"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.setColumnResolution(0), doctest::Contains("at least 1"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.enableHorizontalCoupling(0), doctest::Contains("at least 1"), std::runtime_error);
    SolverParameters bad_interval;
    bad_interval.horizontal_coupling_interval = 0;
    DOCTEST_CHECK_THROWS_WITH_AS(model.setSolverParameters(bad_interval), doctest::Contains("horizontal_coupling_interval"), std::runtime_error);

    setupSmallModel(model, SolverType::SOLVER_3D, 1, 1, 10);
    model.setInitialCondition(-1.0f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFlux(0.0f);
    DOCTEST_CHECK_THROWS_WITH_AS(model.addSurfaceEmitter(0.5f, 0.5f, 1e-6f, -0.1f), doctest::Contains("radius"), std::runtime_error);
    DOCTEST_CHECK_THROWS_WITH_AS(model.addSurfaceEmitter(0.5f, 0.5f, std::numeric_limits<float>::quiet_NaN(), 0.1f), doctest::Contains("flow rate"), std::runtime_error);

    // A rejected parameter set must not be partly applied: adaptive stepping with dt_max = 60 s would take 10 sub-steps
    SolverParameters rejected;
    rejected.adaptive_dt = true;
    rejected.dt_max = 60.0f;
    rejected.enable_gravity = false; // not supported by SOLVER_3D
    DOCTEST_CHECK_THROWS_WITH_AS(model.setSolverParameters(rejected), doctest::Contains("enable_gravity"), std::runtime_error);
    model.run(600.0f);
    DOCTEST_CHECK(model.getLastSubstepCount() == 1);
}

DOCTEST_TEST_CASE("SoilSolver3D - a fixed-head boundary uses the cells' conductivity floor") {
    // Uniform head equal to the fixed head at the top, so every face carries the gravity flux K. With the conductivity
    // floor K_min applied to the boundary as to the cells, the top cell receives what it passes on and stays unchanged;
    // without it the top face carries less and the top cell drains into the cells below.
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    for (auto method: {SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC, SoilSolver3D::NonlinearMethod::PICARD}) {
        DOCTEST_CAPTURE(static_cast<int>(method));
        SoilSolver3D s(1, 1, 10, 1.0, 1.0, 0.1, loam);
        SoilSolver3D::Params p = s.getParams();
        p.nonlinear_method = method;
        p.K_min = 1e-6;
        s.setParams(p);
        s.setInitialCondition(-5.0);
        s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -5.0);
        s.setBoundary(Face3D::ZMIN, BCType::NEUMANN, 0.0);
        const double theta_top = s.getWaterContent(0, 0, 9), theta_bottom = s.getWaterContent(0, 0, 0);
        DOCTEST_REQUIRE(s.solve(60.0) > 0);
        const double gain_bottom = s.getWaterContent(0, 0, 0) - theta_bottom;
        DOCTEST_REQUIRE(gain_bottom > 0.0);
        DOCTEST_CHECK(std::fabs(s.getWaterContent(0, 0, 9) - theta_top) < 0.01 * gain_bottom);
    }
}

DOCTEST_TEST_CASE("SoilSolver3D - settings the GPU cannot run raise errors") {
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilSolver3D s(2, 2, 10, 0.5, 0.5, 0.1, loam);
    SoilSolver3D::Params p = s.getParams();
    p.use_gpu = true;
    // M_SCHEME on the GPU is invalid on any hardware, so it is reported whether or not a GPU is present
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::M_SCHEME;
    DOCTEST_CHECK_THROWS_WITH_AS(s.setParams(p), doctest::Contains("M_SCHEME"), std::runtime_error);
    p.nonlinear_method = SoilSolver3D::NonlinearMethod::NEWTON_ANALYTIC;
#ifndef HELIOS_CUDA_AVAILABLE
    DOCTEST_CHECK_THROWS_WITH_AS(s.setParams(p), doctest::Contains("without CUDA"), std::runtime_error);
#else
    if (!SoilSolver3D::isGPUAvailable()) {
        DOCTEST_CHECK_THROWS_WITH_AS(s.setParams(p), doctest::Contains("no usable CUDA device"), std::runtime_error);
    }
#endif
}

DOCTEST_TEST_CASE("SoilTransportModel - the GPU is used when available") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    DOCTEST_CHECK(model.isGPUEnabled() == SoilTransportModel::isGPUAvailable());
    DOCTEST_CHECK(SoilTransportModel::isGPUAvailable() == SoilSolver3D::isGPUAvailable());
    model.disableGPU();
    DOCTEST_CHECK(model.isGPUEnabled() == false);
    if (SoilTransportModel::isGPUAvailable()) {
        model.enableGPU();
        DOCTEST_CHECK(model.isGPUEnabled());
    } else {
        DOCTEST_CHECK_THROWS_AS(model.enableGPU(), std::runtime_error);
    }
}

DOCTEST_TEST_CASE("RootSoilCoupling - a radial Weibull zone with shape < 1 centred on a column") {
    // The radial density of a Weibull distribution with shape < 1 is infinite at r = 0, where the plant stands.
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(-0.75f, -0.75f, 0.0f), make_vec3(1.5f, 1.5f, 1.0f));
    model.setGridResolution(3, 3, 10); // the plant at (0, 0) is at the centre of the middle column
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-0.5f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFlux(0.0f);
    WeibullRootParameters roots;
    roots.radial = true;
    roots.shape_r = 0.5f;
    roots.r_max = 1.0f;
    model.setRootDistributionWeibull(1, roots);
    model.setTranspiration(1, 2e-7f);

    DOCTEST_CHECK(std::isfinite(model.getSoilWaterPotential(1)));
    const double before = model.getTotalWaterStorage();
    model.run(3600.0f);
    const double removed = before - model.getTotalWaterStorage();
    DOCTEST_CHECK(removed == doctest::Approx(2e-7 * 3600.0 * 1.5 * 1.5).epsilon(1e-3)); // uptake flux x time x footprint
}

// ============================================================================
// Second-review fixes
// ============================================================================

DOCTEST_TEST_CASE("SoilTransportModel - the automatic GPU choice runs the CPU-only M_SCHEME on the CPU") {
    Context context;
    SoilTransportModel model(&context);
    setupSmallModel(model, SolverType::SOLVER_3D, 2, 2, 10);
    SolverParameters params;
    params.nonlinear_method = NonlinearMethod::M_SCHEME;
    DOCTEST_CHECK_NOTHROW(model.setSolverParameters(params));
    DOCTEST_CHECK(model.isGPUEnabled() == false);
    model.setInitialCondition(-1.0f);
    model.setTopBoundaryFlux(1e-6f);
    model.setBottomBoundaryFreeDrainage();
    DOCTEST_CHECK_NOTHROW(model.run(600.0f));
    // Asking for the GPU explicitly with a CPU-only method is an error (on a machine without a GPU, for that reason)
    DOCTEST_CHECK_THROWS_AS(model.enableGPU(), std::runtime_error);
    DOCTEST_CHECK(model.isGPUEnabled() == false);
}

DOCTEST_TEST_CASE("RootSoilCoupling - a rejected root distribution keeps the plant's previous one") {
    for (SolverType solver: {SolverType::SOLVER_1D, SolverType::SOLVER_3D}) {
        DOCTEST_CAPTURE(static_cast<int>(solver));
        Context context;
        SoilTransportModel model(&context);
        setupSmallModel(model, solver, 2, 2, 10);
        model.setInitialCondition(-0.5f);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryFlux(0.0f);
        RootDistributionParams roots;
        roots.plant_x = 0.5f;
        roots.plant_y = 0.5f;
        model.setRootDistribution(1, roots);
        model.setTranspiration(1, 2e-7f);

        RootDistributionParams outside = roots;
        outside.plant_x = 50.0f;
        DOCTEST_CHECK_THROWS_WITH_AS(model.setRootDistribution(1, outside), doctest::Contains("no roots inside the soil domain"), std::runtime_error);

        const double before = model.getTotalWaterStorage();
        model.run(3600.0f);
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) == doctest::Approx(2e-7 * 3600.0).epsilon(1e-4));
        DOCTEST_CHECK(before - model.getTotalWaterStorage() == doctest::Approx(2e-7 * 3600.0).epsilon(0.01)); // 1 m^2 footprint
    }
}

DOCTEST_TEST_CASE("RootSoilCoupling - moving or removing a plant's roots updates the soil sinks at once") {
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    SoilGrid grid(2, 1, 10, 0.5f, 0.5f, 0.1f, loam); // columns at x in [0, 0.5] and [0.5, 1]
    grid.setInitialCondition(-0.5f);
    grid.setTopBoundary(BCType::NEUMANN, 0.0f);
    grid.setBottomBoundary(BCType::NEUMANN, 0.0f);
    RootSoilCoupling coupling(&grid);
    RootDistributionParams roots;
    roots.plant_x = 0.25f;
    roots.plant_y = 0.25f;
    roots.r_max = 0.2f; // within the first column
    coupling.addPlant(1, roots);
    coupling.computeRootUptake(1, 1e-7f);
    DOCTEST_CHECK(grid.getColumn(0, 0).getTranspiration() > 0.0f);
    DOCTEST_CHECK(grid.getColumn(1, 0).getTranspiration() == 0.0f);

    roots.plant_x = 0.75f; // move the roots to the second column
    coupling.updatePlantRoots(1, roots);
    DOCTEST_CHECK(grid.getColumn(0, 0).getTranspiration() == 0.0f);
    DOCTEST_CHECK(grid.getColumn(1, 0).getTranspiration() > 0.0f);

    coupling.removePlant(1);
    DOCTEST_CHECK(grid.getColumn(0, 0).getTranspiration() == 0.0f);
    DOCTEST_CHECK(grid.getColumn(1, 0).getTranspiration() == 0.0f);
}

DOCTEST_TEST_CASE("SoilTransportModel - an error inside a step leaves the state and the clock as they were") {
    // Quasi-3D raises an error when a cell saturates, after the vertical solve has already moved the columns
    Context context;
    SoilTransportModel model(&context);
    setupSmallModel(model, SolverType::SOLVER_QUASI_3D, 2, 2, 10);
    model.enableHorizontalCoupling(1);
    model.setInitialCondition(-0.3f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryPressure(0.5f); // a water table above the bottom: the bottom cells saturate in the first step
    const uint sensor = model.addSensor(make_vec3(0.25f, 0.25f, -0.95f), SoilMeasurement::PRESSURE_HEAD);
    const double storage_before = model.getTotalWaterStorage();
    DOCTEST_CHECK_THROWS_WITH_AS(model.run(600.0f), doctest::Contains("saturation"), std::runtime_error);
    DOCTEST_CHECK(model.getTotalWaterStorage() == doctest::Approx(storage_before).epsilon(1e-9));
    DOCTEST_CHECK(model.getPressureHead(make_vec3(0.25f, 0.25f, -0.95f)) == doctest::Approx(-0.3f));
    model.recordSensors();
    DOCTEST_CHECK(model.getSensorTimes(sensor).back() == doctest::Approx(0.0f));
}

DOCTEST_TEST_CASE("SoilColumn - Picard and Newton apply the conductivity floor alike") {
    // With K_min > 0 every face carries at least K_min, including the fixed-head top face; the two methods must agree.
    const SoilProperties loam = getSoilPropertiesFromTexture("loam");
    std::vector<double> bottom_gain;
    for (NonlinearMethod method: {NonlinearMethod::NEWTON_ANALYTIC, NonlinearMethod::PICARD}) {
        SoilColumn column(10, 0.1f, loam);
        SolverParameters params;
        params.nonlinear_method = method;
        params.K_min = 1e-6f;
        column.setSolverParameters(params);
        column.setInitialCondition(-5.0f);
        column.setTopBoundary(BCType::DIRICHLET, -5.0f);
        column.setBottomBoundary(BCType::NEUMANN, 0.0f);
        const double theta_top = column.getWaterContent(0), theta_bottom = column.getWaterContent(9);
        DOCTEST_REQUIRE(column.solve(60.0f) > 0);
        bottom_gain.push_back(column.getWaterContent(9) - theta_bottom);
        DOCTEST_CHECK(std::fabs(column.getWaterContent(0) - theta_top) < 0.01 * bottom_gain.back()); // the top cell passes on what it receives
    }
    DOCTEST_CHECK(bottom_gain[1] == doctest::Approx(bottom_gain[0]).epsilon(0.02));
}

DOCTEST_TEST_CASE("RootSoilCoupling - a radial Weibull zone whose columns lie far out in its tail") {
    // Every column is 1.3 m or more from the plant, 26 radial scales out, where the Weibull tail is about 1e-11 but not 0
    Context context;
    SoilTransportModel model(&context);
    setupSmallModel(model, SolverType::SOLVER_3D, 2, 2, 10);
    model.setInitialCondition(-0.5f);
    model.setTopBoundaryFlux(0.0f);
    model.setBottomBoundaryFlux(0.0f);
    WeibullRootParameters roots;
    roots.radial = true;
    roots.plant_x = -1.0f;
    roots.plant_y = 0.5f;
    roots.shape_r = 1.0f;
    roots.scale_r = 0.05f;
    roots.r_max = 3.0f;
    DOCTEST_REQUIRE_NOTHROW(model.setRootDistributionWeibull(1, roots));
    model.setTranspiration(1, 2e-7f);
    const double before = model.getTotalWaterStorage();
    model.run(3600.0f);
    DOCTEST_CHECK(before - model.getTotalWaterStorage() == doctest::Approx(2e-7 * 3600.0).epsilon(1e-3));
}

DOCTEST_TEST_CASE("SoilTransportModel - explicit root primitives are not implemented") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(-0.5f, -0.5f, 0.0f), make_vec3(1, 1, 1.0f));
    model.setGridResolution(3, 3, 8);
    model.setSoilPropertiesFromTexture("loam");

    std::vector<uint> root_UUIDs = context.addTube(8, {make_vec3(0, 0, 0), make_vec3(0, 0, -0.5f)}, {0.01f, 0.005f});
    std::string message;
    try {
        model.setRootPrimitives(1, root_UUIDs);
    } catch (const std::runtime_error &error) {
        message = error.what();
    }
    DOCTEST_CHECK(message.find("not implemented") != std::string::npos);
}

DOCTEST_TEST_CASE("SoilTransportModel - updateSoilVisualization re-samples the current field") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(3, 3, 6);
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-1.0f);

    std::vector<uint> UUIDs = model.addSoilVisualization(SoilColorMode::WATER_CONTENT);
    DOCTEST_REQUIRE(!UUIDs.empty());
    float theta_before;
    context.getPrimitiveData(UUIDs.front(), "water_content", theta_before);

    model.setInitialCondition(-0.05f); // much wetter state, same geometry
    model.updateSoilVisualization(UUIDs, SoilColorMode::WATER_CONTENT);

    float theta_after;
    context.getPrimitiveData(UUIDs.front(), "water_content", theta_after);
    vec3 sample_pos;
    context.getPrimitiveData(UUIDs.front(), "soil_sample_pos", sample_pos);
    DOCTEST_CHECK(theta_after > theta_before);
    DOCTEST_CHECK(theta_after == doctest::Approx(model.getWaterContent(sample_pos)));

    // Pressure-head mode writes its own data label on the same faces
    model.updateSoilVisualization(UUIDs, SoilColorMode::PRESSURE_HEAD);
    float h_face;
    context.getPrimitiveData(UUIDs.front(), "pressure_head", h_face);
    DOCTEST_CHECK(h_face == doctest::Approx(-0.05f).epsilon(0.01));
}

DOCTEST_TEST_CASE("SoilTransportModel - disableMessages silences messages, enableMessages restores them") {
    std::string silenced, restored;
    {
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        helios::capture_cout cout_capture;
        model.initializeDomain();
        silenced = cout_capture.get_captured_output();
    }
    {
        Context context;
        SoilTransportModel model(&context);
        model.disableMessages();
        model.enableMessages();
        helios::capture_cout cout_capture;
        model.initializeDomain();
        restored = cout_capture.get_captured_output();
    }
    DOCTEST_CHECK(silenced.empty());
    DOCTEST_CHECK(restored.find("Initialized domain") != std::string::npos);
}

DOCTEST_TEST_CASE("SoilTransportModel - solver diagnostics counters and resets") {
    Context context;
    SoilTransportModel model(&context);
    model.disableMessages();
    model.defineSoilDomain(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    model.setGridResolution(4, 4, 8);
    model.setSolverType(SolverType::SOLVER_3D);
    model.setSoilPropertiesFromTexture("loam");
    model.setInitialCondition(-1.0f);
    model.setTopBoundaryFlux(1e-6f); // infiltration: a few nonlinear iterations per step
    model.setBottomBoundaryFreeDrainage();

    SolverParameters sp;
    sp.adaptive_dt = true;
    sp.dt_max = 300.0f;
    sp.nonlinear_method = NonlinearMethod::PICARD_NEWTON; // the GPU Picard/Newton counters are kept by this path
    model.setSolverParameters(sp);

    model.resetSolverIterationCount();
    model.resetSolverStats();
    model.run(1800.0f);

    DOCTEST_CHECK(model.getLastSubstepCount() >= 6); // 1800 s at dt_max = 300 s
    DOCTEST_CHECK(model.getSolverIterationCount() > 0);
    if (model.isGPUEnabled()) {
        DOCTEST_CHECK(model.solverPicardOuter() + model.solverNewtonOuter() > 0);
    }

    model.resetSolverIterationCount();
    model.resetSolverStats();
    DOCTEST_CHECK(model.getSolverIterationCount() == 0);
    DOCTEST_CHECK(model.solverPicardOuter() == 0);
    DOCTEST_CHECK(model.solverNewtonOuter() == 0);
    DOCTEST_CHECK(model.solverPicardGmres() == 0);
    DOCTEST_CHECK(model.solverNewtonGmres() == 0);
}

DOCTEST_TEST_CASE("SoilTransportModel - root water stress limits uptake in dry soil") {
    // Sealed 3D box with a 1D Weibull root zone, uniform initial head. Default stress parameters:
    // alpha = 1 above -4 m, 0 below -150 m; compensation up to omega_c = 0.5.
    auto makeModel = [](Context &context, SoilTransportModel &model, float h_init) {
        model.disableMessages();
        model.defineSoilDomain(make_vec3(-0.5f, -0.5f, 0.0f), make_vec3(1, 1, 1.0f));
        model.setGridResolution(3, 3, 8);
        model.setSolverType(SolverType::SOLVER_3D);
        model.setSoilPropertiesFromTexture("loam");
        model.setInitialCondition(h_init);
        model.setTopBoundaryFlux(0.0f);
        model.setBottomBoundaryFlux(0.0f);
        WeibullRootParameters roots;
        roots.z_max = 0.8f;
        model.setRootDistributionWeibull(1, roots);
        model.enableRootUptake();
        model.setTranspiration(1, 2e-7f);
    };
    const float T = 2e-7f;

    DOCTEST_SUBCASE("moist soil supplies the full demand") {
        Context context;
        SoilTransportModel model(&context);
        makeModel(context, model, -0.5f);
        DOCTEST_CHECK(model.getRootUptakeFlux(1) == doctest::Approx(T));
    }
    DOCTEST_SUBCASE("soil below wilting point supplies nothing") {
        Context context;
        SoilTransportModel model(&context);
        makeModel(context, model, -200.0f);
        const float storage_before = model.getTotalWaterStorage();
        for (int s = 0; s < 6; s++)
            model.run(600.0f);
        DOCTEST_CHECK(model.getRootUptakeFlux(1) == doctest::Approx(0.0f));
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) == doctest::Approx(0.0f));
        DOCTEST_CHECK(model.getTotalWaterStorage() == doctest::Approx(storage_before).epsilon(1e-4));
    }
    DOCTEST_SUBCASE("partly stressed soil: compensation then proportional reduction") {
        // h = -100 m: alpha = 50/146 = 0.342 < omega_c = 0.5, so uptake = T * 0.342 / 0.5
        Context context;
        SoilTransportModel model(&context);
        makeModel(context, model, -100.0f);
        const float expected = T * (50.0f / 146.0f) / 0.5f;
        DOCTEST_CHECK(model.getRootUptakeFlux(1) == doctest::Approx(expected).epsilon(1e-3));
        const float storage_before = model.getTotalWaterStorage();
        for (int s = 0; s < 6; s++)
            model.run(600.0f);
        const float removed = storage_before - model.getTotalWaterStorage();
        // Uptake starts at the stressed rate and falls further as the extraction dries the soil
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) < expected * 3600.0f);
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) > 0.5f * expected * 3600.0f);
        DOCTEST_CHECK(model.getCumulativeRootUptake(1) * 1.0f == doctest::Approx(removed).scale(0).epsilon(0.05)); // 1 m^2 footprint
    }
    DOCTEST_SUBCASE("omega_c = 1 disables compensation") {
        Context context;
        SoilTransportModel model(&context);
        makeModel(context, model, -100.0f);
        model.setRootWaterStressParameters(-4.0f, -150.0f, 1.0f);
        model.setTranspiration(1, T);
        DOCTEST_CHECK(model.getRootUptakeFlux(1) == doctest::Approx(T * 50.0f / 146.0f).epsilon(1e-3));
    }
    DOCTEST_SUBCASE("invalid parameters are rejected") {
        Context context;
        SoilTransportModel model(&context);
        makeModel(context, model, -0.5f);
        DOCTEST_CHECK_THROWS_WITH_AS(model.setRootWaterStressParameters(-150.0f, -4.0f, 0.5f), doctest::Contains("ERROR (RootSoilCoupling::setWaterStressParameters)"), std::runtime_error);
        DOCTEST_CHECK_THROWS_WITH_AS(model.setRootWaterStressParameters(-4.0f, -150.0f, 0.0f), doctest::Contains("ERROR (RootSoilCoupling::setWaterStressParameters)"), std::runtime_error);
    }
}

int SoilTransportModel::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
