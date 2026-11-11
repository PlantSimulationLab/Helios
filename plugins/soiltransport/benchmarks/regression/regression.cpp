/** \file "regression.cpp" SoilTransport regression benchmarks: accuracy, solver work and run time of a fixed set of
    cases, written as one JSON line per case for run_regression.py.

    The same source compiles against every generation of the plug-in: features a generation lacks are detected at
    compile time and their cases report "unsupported". Each case runs with the defaults of the generation it is built
    against (what a user of that version got), unless the case is about a specific setting.

    Usage:  soil_regression list [quick|full]       case names of a tier
            soil_regression run &lt;case&gt;              run one case, print its JSON line
            soil_regression reference &lt;out.f32&gt;     write the 3D drip reference (fine grid, short steps; slow)

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilColumn.h"
#include "SoilSolver3D.h"
#include "SoilUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


// ============================================================================
// Compile-time feature detection (so one source builds against every generation)
// ============================================================================

namespace {
    template<class, class = void>
    struct HasRadial : std::false_type {};
    template<class T>
    struct HasRadial<T, std::void_t<decltype(std::declval<T &>().setRadialGeometry(1.0f))>> : std::true_type {};
    template<class, class = void>
    struct HasAtmospheric : std::false_type {};
    template<class T>
    struct HasAtmospheric<T, std::void_t<decltype(std::declval<T &>().setTopBoundaryAtmospheric(0.0f, 0.0f)), decltype(std::declval<T &>().getTopFlux())>> : std::true_type {};
    template<class, class = void>
    struct HasKrSpline : std::false_type {};
    template<class T>
    struct HasKrSpline<T, std::void_t<decltype(std::declval<T &>().kr_spline_start)>> : std::true_type {};
    template<class, class = void>
    struct HasGravitySwitch : std::false_type {};
    template<class T>
    struct HasGravitySwitch<T, std::void_t<decltype(std::declval<T &>().enable_gravity)>> : std::true_type {};
    template<class, class = void>
    struct HasCellGeometry : std::false_type {};
    template<class T>
    struct HasCellGeometry<T, std::void_t<decltype(std::declval<T &>().setCellGeometry(std::vector<double>(), std::vector<double>()))>> : std::true_type {};
    template<class, class = void>
    struct HasNewtonAnalytic : std::false_type {};
    template<class T>
    struct HasNewtonAnalytic<T, std::void_t<decltype(T::NonlinearMethod::NEWTON_ANALYTIC)>> : std::true_type {};
    template<class, class = void>
    struct HasFixedHead : std::false_type {};
    template<class T>
    struct HasFixedHead<T, std::void_t<decltype(std::declval<T &>().setFixedHead(0u, 0u, 0u, 0.0)), decltype(std::declval<T &>().setSpecificStorage(std::vector<double>()))>> : std::true_type {};
    template<class, class = void>
    struct HasSubstepSwitch : std::false_type {};
    template<class T>
    struct HasSubstepSwitch<T, std::void_t<decltype(std::declval<T &>().substep_on_failure)>> : std::true_type {};
    template<class, class = void>
    struct HasResetHistory : std::false_type {};
    template<class T>
    struct HasResetHistory<T, std::void_t<decltype(std::declval<T &>().resetHistory())>> : std::true_type {};

#ifdef HELIOS_CUDA_AVAILABLE
    constexpr bool kGpuBuild = true;
#else
    constexpr bool kGpuBuild = false;
#endif

    // ============================================================================
    // Results
    // ============================================================================

    struct Result {
        std::string status = "ok", note;
        double time_s = 0.0; // solve time only (no setup or post-processing)
        std::vector<std::pair<std::string, double>> errors; // lower is better (vs an exact or reference solution)
        std::vector<std::pair<std::string, double>> values; // results without an exact reference: must not drift
        std::vector<std::pair<std::string, double>> work; // steps, iterations (lower is better)
    };
    Result unsupported(const std::string &why) {
        Result r;
        r.status = "unsupported";
        r.note = why;
        return r;
    }
    Result failure(const std::string &why, double time_s = 0.0) {
        Result r;
        r.status = "failed";
        r.note = why;
        r.time_s = time_s;
        return r;
    }

    std::string jsonEscape(const std::string &s) {
        std::string o;
        for (char c: s)
            o += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
        return o;
    }
    void printResult(const std::string &name, Result r) {
        for (auto *group: {&r.errors, &r.values, &r.work})
            for (auto &m: *group)
                if (!std::isfinite(m.second) && r.status == "ok") {
                    r.status = "failed";
                    r.note = "non-finite " + m.first;
                }
        std::printf("{\"case\": \"%s\", \"status\": \"%s\", \"note\": \"%s\", \"time_s\": %.6g", name.c_str(), r.status.c_str(), jsonEscape(r.note).c_str(), r.time_s);
        const char *names[3] = {"errors", "values", "work"};
        const std::vector<std::pair<std::string, double>> *groups[3] = {&r.errors, &r.values, &r.work};
        for (int g = 0; g < 3; g++) {
            std::printf(", \"%s\": {", names[g]);
            for (size_t i = 0; i < groups[g]->size(); i++)
                std::printf("%s\"%s\": %.10g", i ? ", " : "", (*groups[g])[i].first.c_str(), std::isfinite((*groups[g])[i].second) ? (*groups[g])[i].second : 1e300);
            std::printf("}");
        }
        std::printf("}\n");
    }

    //! Accumulates solve time
    struct SolveClock {
        double seconds = 0.0;
        template<class F>
        auto operator()(F &&f) {
            const auto t0 = std::chrono::steady_clock::now();
            auto r = f();
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return r;
        }
    };
} // namespace

// ============================================================================
// Analytical solutions (ported from DuMux 3.10; see samples/soiltransport_benchmarks)
// ============================================================================

namespace {
    struct DumuxSoil {
        const char *name;
        double Swr, porosity, alpha_Pa, n, permeability;
        double tr() const { return Swr * porosity; }
        double ts() const { return porosity; }
        double alpha_cm() const { return alpha_Pa * 1000.0 * 9.81 / 100.0; } // [1/cm]
        double Ks_cmd() const { return permeability * 1000.0 * 9.81 / 1e-3 * 100.0 * 86400.0; } // [cm/day]
        SoilProperties props() const {
            return SoilProperties((float) tr(), (float) ts(), (float) (alpha_cm() * 100.0), (float) n, (float) (Ks_cmd() / 100.0 / 86400.0));
        }
        // van Genuchten-Mualem (L = 0.5, no regularization) in head units [cm] and water content
        double m() const { return 1.0 - 1.0 / n; }
        double thetaOfHead(double h_cm) const {
            return h_cm >= 0.0 ? ts() : tr() + (ts() - tr()) * std::pow(1.0 + std::pow(-alpha_cm() * h_cm, n), -m());
        }
        double conductivity(double theta) const { // [cm/day]
            const double Se = std::min(std::max((theta - tr()) / (ts() - tr()), 0.0), 1.0);
            const double b = 1.0 - std::pow(1.0 - std::pow(Se, 1.0 / m()), m());
            return Ks_cmd() * std::sqrt(Se) * b * b;
        }
        double diffusivity(double theta) const { // K dh/dtheta [cm^2/day]
            const double Se = std::min(std::max((theta - tr()) / (ts() - tr()), 1e-300), 1.0 - 1e-15);
            const double x = std::pow(std::pow(Se, -1.0 / m()) - 1.0, 1.0 / n); // alpha |h|
            const double dSe_dh = m() * n * alpha_cm() * std::pow(x, n - 1.0) * std::pow(1.0 + std::pow(x, n), -m() - 1.0);
            return conductivity(theta) / ((ts() - tr()) * dSe_dh);
        }
    };

    //! Adaptive Simpson quadrature of f over [a, b]
    template<class F>
    double integrate(const F &f, double a, double b, double tol = 1e-10, int depth = 60) {
        std::function<double(double, double, double, double, double, double, int)> rec = [&](double l, double r, double fl, double fm, double fr, double whole, int d) {
            const double c = 0.5 * (l + r), lm = 0.5 * (l + c), rm = 0.5 * (c + r);
            const double flm = f(lm), frm = f(rm);
            const double left = (c - l) / 6.0 * (fl + 4.0 * flm + fm), right = (r - c) / 6.0 * (fm + 4.0 * frm + fr);
            if (d <= 0 || std::fabs(left + right - whole) <= 15.0 * tol)
                return left + right + (left + right - whole) / 15.0;
            return rec(l, c, fl, flm, fm, left, d - 1) + rec(c, r, fm, frm, fr, right, d - 1);
        };
        const double fa = f(a), fb = f(b), fm = f(0.5 * (a + b));
        return rec(a, b, fa, fm, fb, (b - a) / 6.0 * (fa + 4.0 * fm + fb), depth);
    }

    //! M2.1: water content vs Delta eta (Vanderborght et al. 2005, eq. 4abc)
    struct InfiltrationAnalytical {
        DumuxSoil s;
        double thi, ths, Ki, Ks, thref, x_ref, t_ref;
        std::vector<double> theta, deta; // tabulated curve
        InfiltrationAnalytical(const DumuxSoil &soil, double h_init_cm, double theta_surface, double ref_depth_cm, double ref_time_d) : s(soil), ths(theta_surface), x_ref(ref_depth_cm), t_ref(ref_time_d) {
            thi = s.thetaOfHead(h_init_cm);
            Ki = s.conductivity(thi);
            Ks = s.conductivity(ths);
            thref = 0.5 * (ths + thi);
            const int N = 2000;
            for (int i = 0; i < N; i++) {
                const double th = (ths - 1e-8) + (thi + 1e-8 - (ths - 1e-8)) * i / (N - 1.0);
                auto integrand = [&](double tau) { return s.diffusivity(tau) / ((Ks - Ki) * (tau - thi) - (s.conductivity(tau) - Ki) * (ths - thi)); };
                theta.push_back(th);
                deta.push_back((ths - thi) * integrate(integrand, th, thref, 1e-12));
            }
        }
        double eta(double x_cm, double t_d) const {
            return std::fabs(x_cm) - (Ks - Ki) / (ths - thi) * t_d;
        }
        double deltaEta(double x_cm, double t_d) const {
            return eta(x_cm, t_d) - eta(x_ref, t_ref);
        }
        //! Analytical water content at a given Delta eta (outside the tabulated range: the end values)
        double thetaAt(double de) const {
            if (de <= deta.front())
                return theta.front();
            if (de >= deta.back())
                return theta.back();
            for (size_t i = 1; i < deta.size(); i++)
                if (de <= deta[i])
                    return theta[i - 1] + (theta[i] - theta[i - 1]) * (de - deta[i - 1]) / (deta[i] - deta[i - 1]);
            return theta.back();
        }
    };

    //! M2.2: actual evaporation rate (Vanderborght et al. 2005, eq. 5abcd)
    struct EvaporationAnalytical {
        double E, S, tpot, tprime; // potential rate [mm/day], desorptivity [cm/day^0.5], times [day]
        EvaporationAnalytical(const DumuxSoil &s, double h_init_cm, double rate_mm_d) : E(rate_mm_d) {
            const double thi = s.thetaOfHead(h_init_cm), thsur = s.thetaOfHead(-10000.0);
            auto th = [&](double te) { return te * (thi - thsur) + thsur; };
            const double dI = integrate([&](double te) { return s.diffusivity(th(te)); }, 0.0, 1.0, 1e-10);
            const double dTI = integrate([&](double te) { return te * s.diffusivity(th(te)); }, 0.0, 1.0, 1e-10);
            const double beta = dTI * dTI / (dI * dI);
            const double dTF = integrate([&](double te) { const double w = 1.0 - beta * te; return w * w * s.diffusivity(th(te)); }, 0.0, 1.0, 1e-10);
            const double a = dTF / dI, bracket = 1.0 - a / ((1.0 - beta) * (1.0 - beta));
            const double mu = 3.0 * beta * (1.0 + std::sqrt(1.0 - 14.0 / 9.0 * bracket)) / (2.0 * (beta - 1.0) * bracket);
            S = (thi - thsur) * std::sqrt(4.0 / mu * dI);
            tpot = S * S / (2.0 * (E * 0.1) * (E * 0.1));
            tprime = 0.5 * tpot;
        }
        double rate(double t_d) const {
            return t_d < tpot ? E : 0.5 * S / std::sqrt(tprime + t_d - tpot) * 10.0;
        }
    };

    const DumuxSoil kVdSand{"sand", 0.10465116279, 0.43, 1.52905199e-3, 3.0, 1.1798241e-11};
    const DumuxSoil kVdLoam{"loam", 0.18604651162, 0.43, 4.07747197e-4, 1.6, 5.8991203e-13};
    const DumuxSoil kVdClay{"clay", 0.25, 0.40, 1.01936799e-4, 1.1, 1.1798241e-13};
} // namespace

namespace {
    struct RootAnnulus {
        static constexpr double ri = 0.02e-2, ro = 0.6e-2, uptake_flux = 0.1 * 0.01 / 86400.0; // [m], [m], [m/s] at r = ri
        DumuxSoil soil;
        std::vector<double> hn, Pn; // cumulative Kirchhoff table, log-spaced in -h
        explicit RootAnnulus(const DumuxSoil &s) : soil(s) {
            const int NT = 4000;
            hn.resize(NT + 1);
            Pn.assign(NT + 1, 0.0);
            for (int j = 0; j <= NT; j++)
                hn[j] = -std::exp(std::log(200.0) + (std::log(1e-3) - std::log(200.0)) * j / NT); // -200 .. -1e-3 m
            for (int j = 1; j <= NT; j++)
                Pn[j] = Pn[j - 1] + gauss(hn[j - 1], hn[j]);
        }
        double K(double h) const { // [m/s], independent of the plug-in's constitutive code
            return soil.conductivity(soil.thetaOfHead(h * 100.0)) / 100.0 / 86400.0;
        }
        double gauss(double a, double b) const { // 20-point Gauss-Legendre
            static const double x[10] = {0.0765265211334973, 0.2277858511416451, 0.3737060887154195, 0.5108670019508271, 0.6360536807265150, 0.7463319064601508, 0.8391169718222188, 0.9122344282513259, 0.9639719272779138, 0.9931285991850949};
            static const double w[10] = {0.1527533871307258, 0.1491729864726037, 0.1420961093183820, 0.1316886384491766, 0.1181945319615184, 0.1019301198172404, 0.0832767415767048, 0.0626720483341091, 0.0406014298003869, 0.0176140071391521};
            const double c = 0.5 * (a + b), d = 0.5 * (b - a);
            double sum = 0.0;
            for (int j = 0; j < 10; j++)
                sum += w[j] * (K(c - d * x[j]) + K(c + d * x[j]));
            return sum * d;
        }
        double Phi(double h) const {
            const int j = std::max(0, std::min((int) hn.size() - 2, (int) (std::upper_bound(hn.begin(), hn.end(), h) - hn.begin()) - 1));
            return Pn[j] + gauss(hn[j], h);
        }
        double headOfPhi(double P) const { // bisection (Phi increases with h)
            double a = hn.front(), b = hn.back();
            for (int it = 0; it < 200 && b - a > 1e-13 * std::max(1.0, std::fabs(a)); it++) {
                const double m = 0.5 * (a + b);
                (Phi(m) < P ? a : b) = m;
            }
            return 0.5 * (a + b);
        }
        double source() const { // uniform source balancing the uptake [1/s]
            return 2.0 * ri * uptake_flux / (ro * ro - ri * ri);
        }
        double steadyHead(double r, double h_root) const {
            return headOfPhi(Phi(h_root) - 0.5 * source() * (0.5 * (r * r - ri * ri) - ro * ro * std::log(r / ri)));
        }
        //! Face radii of DuMux's graded grid, each cell halved `refinements` times
        static std::vector<double> grid(int refinements) {
            const int N = 300;
            const double g = 0.95, h = (1.0 - g) / (1.0 - std::pow(g, N));
            std::vector<double> pos(1, 0.0);
            for (int i = 0; i < N - 1; i++)
                pos.push_back(pos.back() + h * std::pow(g, N - i - 1));
            pos.push_back(1.0);
            for (double &x: pos)
                x = ri + x * (ro - ri);
            for (int r = 0; r < refinements; r++) {
                std::vector<double> fine;
                for (size_t k = 0; k + 1 < pos.size(); k++) {
                    fine.push_back(pos[k]);
                    fine.push_back(0.5 * (pos[k] + pos[k + 1]));
                }
                fine.push_back(pos.back());
                pos = fine;
            }
            return pos;
        }
        static std::vector<float> thicknesses(const std::vector<double> &pos) {
            std::vector<float> dz(pos.size() - 1);
            for (size_t k = 0; k < dz.size(); k++)
                dz[k] = (float) (pos[k + 1] - pos[k]);
            return dz;
        }
    };
} // namespace

// ============================================================================
// 1D column cases
// ============================================================================

namespace {
    //! Closed column, no fluxes, 1 h of 60 s steps: storage must not drift
    template<class Col = SoilColumn>
    Result colMassBalance() {
        Col column(50, 0.02f, getSoilPropertiesFromTexture("loam"));
        column.setInitialCondition(-1.0f);
        column.setTopBoundary(BCType::NEUMANN, 0.0f);
        column.setBottomBoundary(BCType::NEUMANN, 0.0f);
        const double s0 = column.getTotalWaterStorage();
        SolveClock clock;
        double iters = 0.0;
        for (int n = 0; n < 60; n++)
            iters += clock([&] { return column.solve(60.0f); });
        Result r;
        r.time_s = clock.seconds;
        r.errors = {{"storage_drift_rel", std::fabs(column.getTotalWaterStorage() - s0) / s0}};
        r.work = {{"iterations", iters}};
        return r;
    }

    //! Steady infiltration into a Gardner soil above a water table: exact solution
    template<class Col = SoilColumn>
    Result colGardnerSteady() {
        const SoilProperties soil = SoilProperties::gardner(0.06f, 0.40f, 2.0f, 1e-5f);
        const double Ks = soil.K_s, alpha = soil.alpha, q = 0.5 * Ks, L = 0.5;
        const uint nz = 50;
        Col column(nz, (float) (L / nz), soil);
        column.setInitialCondition(0.0f);
        column.setTopBoundary(BCType::NEUMANN, (float) q);
        column.setBottomBoundary(BCType::DIRICHLET, 0.0f);
        SolverParameters sp;
        sp.max_picard_iterations = 100;
        sp.picard_tolerance = 1e-9f;
        sp.K_weighting = KWeighting::ARITHMETIC;
        column.setSolverParameters(sp);
        SolveClock clock;
        int steps = 0;
        for (; steps < 5000; steps++) {
            std::vector<float> prev(column.getPressureHeads().begin(), column.getPressureHeads().end());
            if (clock([&] { return column.solve(600.0f); }) == 0)
                return failure("step failed", clock.seconds);
            double dmax = 0.0;
            for (uint k = 0; k < nz; k++)
                dmax = std::max(dmax, (double) std::fabs(column.getPressureHead(k) - prev[k]));
            if (dmax < 1e-7 && steps > 10)
                break;
        }
        double err = 0.0;
        for (uint k = 0; k < nz; k++) {
            const double z = L - column.getLayerDepth(k);
            err = std::max(err, std::fabs(column.getPressureHead(k) - std::log(((Ks - q) * std::exp(-alpha * z) + q) / Ks) / alpha));
        }
        Result r;
        r.time_s = clock.seconds;
        r.errors = {{"max_dh_exact_m", err}};
        r.work = {{"steps", (double) steps}};
        return r;
    }

    //! Vanderborght et al. (2005) M2.1 infiltration into dry soil: travelling-wave analytical solution (DuMux benchmark).
    //! Fixed boundaries available in every generation: sand takes the full 1000 mm/day (never ponds); loam and clay
    //! pond at once (h = 0). spline: the k_r spline above S_e = 0.9999 (as DuMux's clay input), where available.
    template<class Col = SoilColumn, class P = SoilProperties>
    Result colInfiltration(const DumuxSoil &soil, double theta_surface, double ref_depth_cm, double ref_time_d, double dt_max, double t_end_d, bool spline) {
        P props = soil.props();
        if (spline) {
            if constexpr (HasKrSpline<P>::value)
                props.kr_spline_start = 0.9999f;
            else
                return unsupported("no k_r spline");
        }
        const InfiltrationAnalytical ana(soil, -400.0, theta_surface, ref_depth_cm, ref_time_d);
        const uint nz = 300;
        const float dz = 2.0f / nz;
        Col column(nz, dz, props);
        column.setInitialCondition(-4.0f);
        if (std::string(soil.name) == "sand")
            column.setTopBoundary(BCType::NEUMANN, 1.0f / 86400.0f);
        else
            column.setTopBoundary(BCType::DIRICHLET, 0.0f);
        column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
        SolverParameters sp;
        sp.dt_max = (float) dt_max;
        sp.dt_min = 1e-3f;
        column.setSolverParameters(sp);
        SolveClock clock;
        float dt = 1.0f;
        uint steps = 0;
        const float target = (float) (t_end_d * 86400.0);
        const float advanced = clock([&] { return column.solveAdaptive(target, dt, &steps); });
        if (advanced < target * 0.9999f)
            return failure("stopped at t = " + std::to_string(advanced) + " s", clock.seconds);
        double se = 0.0;
        int cnt = 0;
        for (uint k = 0; k < nz; k++) {
            const double de = ana.deltaEta((k + 0.5) * dz * 100.0, t_end_d);
            if (de > -10.0 && de < 5.0) {
                const double e = column.getWaterContent(k) - ana.thetaAt(de);
                se += e * e;
                cnt++;
            }
        }
        Result r;
        r.time_s = clock.seconds;
        r.errors = {{"rmse_theta_front", std::sqrt(se / std::max(cnt, 1))}};
        r.work = {{"steps", (double) steps}};
        return r;
    }

    //! Vanderborght et al. (2005) M2.2 evaporation: desorptivity analytical rate (DuMux benchmark); needs the
    //! head-limited (atmospheric) top boundary
    template<class Col = SoilColumn, class SP = SolverParameters>
    Result colEvaporation(const DumuxSoil &soil, double h_init_cm, double rate_mm_d, double t_end_d, double dt_max) {
        if constexpr (!HasAtmospheric<Col>::value || !HasGravitySwitch<SP>::value)
            return unsupported("no atmospheric top boundary");
        else {
            const EvaporationAnalytical ana(soil, h_init_cm, rate_mm_d);
            const uint nz = 1000;
            std::vector<float> dz(nz);
            const double g = 0.99, h0 = (1.0 - g) / (1.0 - std::pow(g, (double) nz));
            for (uint i = 0; i < nz; i++)
                dz[nz - 1 - i] = (float) (h0 * std::pow(g, (double) i));
            Col column(dz, soil.props());
            column.setInitialCondition((float) (h_init_cm / 100.0));
            column.setTopBoundaryAtmospheric((float) (-rate_mm_d / 1000.0 / 86400.0), -100.0f);
            column.setBottomBoundary(BCType::NEUMANN, 0.0f);
            SP sp;
            sp.enable_gravity = false;
            sp.dt_max = (float) dt_max;
            sp.dt_min = 1e-3f;
            column.setSolverParameters(sp);
            SolveClock clock;
            const int nout = 400;
            float dt = 10.0f, t = 0.0f;
            double se = 0.0, cum_h = 0.0, cum_a = 0.0;
            uint steps = 0;
            for (int i = 1; i <= nout; i++) {
                const double td = t_end_d * i / nout;
                uint st = 0;
                t += clock([&] { return column.solveAdaptive((float) (td * 86400.0) - t, dt, &st); });
                steps += st;
                const double rh = -column.getTopFlux() * 1000.0 * 86400.0, ra = ana.rate(td);
                se += (rh - ra) * (rh - ra);
                cum_h += rh * t_end_d / nout;
                cum_a += ra * t_end_d / nout;
            }
            if (t < t_end_d * 86400.0 * 0.9999)
                return failure("stopped early", clock.seconds);
            Result r;
            r.time_s = clock.seconds;
            r.errors = {{"rmse_rate_rel", std::sqrt(se / nout) / rate_mm_d}, {"cumulative_rel", std::fabs(cum_h - cum_a) / cum_a}};
            r.work = {{"steps", (double) steps}};
            return r;
        }
    }

    //! Radial flow to a root (DuMux annulus): steady state on DuMux's graded grid and two refinements
    template<class Col = SoilColumn>
    Result colAnnulusSteady(const DumuxSoil &soil) {
        if constexpr (!HasRadial<Col>::value)
            return unsupported("no radial geometry");
        else {
            const RootAnnulus ann(soil);
            SolveClock clock;
            double l2[3], iters = 0.0;
            for (int ref = 0; ref < 3; ref++) {
                const std::vector<double> pos = RootAnnulus::grid(ref);
                const uint N = pos.size() - 1;
                Col column(RootAnnulus::thicknesses(pos), soil.props());
                column.setRadialGeometry((float) RootAnnulus::ri);
                SolverParameters sp;
                sp.picard_tolerance = 1e-9f;
                column.setSolverParameters(sp);
                column.setInitialCondition(-1.0f);
                column.setTopBoundary(BCType::DIRICHLET, -150.0f);
                column.setBottomBoundary(BCType::NEUMANN, 0.0f);
                for (uint k = 0; k < N; k++)
                    column.setSource(k, (float) (ann.source() * (pos[k + 1] * pos[k + 1] - pos[k] * pos[k]) / (2.0 * RootAnnulus::ri)));
                for (int s = 0; s < 4; s++) {
                    const uint it = clock([&] { return column.solve(1e10f); });
                    if (it == 0)
                        return failure("steady solve failed", clock.seconds);
                    iters += it;
                }
                double e2 = 0.0;
                for (uint k = 0; k < N; k++) {
                    const double e = (column.getPressureHead(k) - ann.steadyHead(0.5 * (pos[k] + pos[k + 1]), -150.0)) * 9810.0;
                    e2 += (pos[k + 1] - pos[k]) * e * e;
                }
                l2[ref] = std::sqrt(e2);
            }
            Result r;
            r.time_s = clock.seconds;
            r.errors = {{"l2_pa_300", l2[0]}, {"l2_pa_1200", l2[2]}, {"order_deficit", std::max(0.0, 2.0 - std::log(l2[2] / l2[1]) / std::log(0.5))}};
            r.work = {{"iterations", iters}};
            return r;
        }
    }

    //! Shared driver: a fixed demand until the head at layer/cell 0 reaches -150 m; steps of up to dt_max, a step that
    //! fails or overshoots by more than 0.1% is halved. Returns {time, water balance error, steps, retries}.
    struct DryingRun {
        double t = 0.0, balance = 0.0, steps = 0.0, retries = 0.0;
        bool ok = false;
    };
    //! These drivers shorten failed steps themselves: switch off solve()'s own sub-stepping where it exists
    template<class Col, class SP = SolverParameters>
    void ownRetries(Col &column) {
        if constexpr (HasSubstepSwitch<SP>::value) {
            SP sp;
            sp.substep_on_failure = false;
            column.setSolverParameters(sp);
        }
    }
    template<class Params>
    void ownRetries3D(Params &p) {
        if constexpr (HasSubstepSwitch<Params>::value)
            p.substep_on_failure = false;
    }

    template<class Solve, class Head, class Save, class Restore, class Storage>
    DryingRun dryToWilting(double demand, double area, double dt_max, Solve solve, Head head, Save save, Restore restore, Storage storage, SolveClock &clock) {
        DryingRun d;
        const double s0 = storage(), h_wilt = -150.0;
        double dt = dt_max;
        for (int tries = 0; tries < 5000; tries++) {
            save();
            const bool ok = clock([&] { return solve(dt); });
            if (!ok || head() < 1.001 * h_wilt) {
                restore();
                dt *= 0.5;
                d.retries++;
                if (dt < 1e-3)
                    return d;
                continue;
            }
            d.t += dt;
            d.steps++;
            if (head() <= h_wilt) {
                d.ok = true;
                break;
            }
            dt = std::min(2.0 * dt, dt_max);
        }
        d.balance = (s0 - storage() - demand * area * d.t) / (demand * area * d.t);
        return d;
    }

    //! Root uptake from the DuMux annulus until wilting, vs the steady-rate approximation
    template<class Col = SoilColumn>
    Result colAnnulusWilting(const DumuxSoil &soil) {
        if constexpr (!HasRadial<Col>::value)
            return unsupported("no radial geometry");
        else {
            const RootAnnulus ann(soil);
            const double ri = RootAnnulus::ri, ro = RootAnnulus::ro, q = RootAnnulus::uptake_flux;
            const std::vector<double> pos = RootAnnulus::grid(0);
            Col column(RootAnnulus::thicknesses(pos), soil.props());
            column.setRadialGeometry((float) ri);
            ownRetries(column);
            column.setInitialCondition(-1.0f);
            column.setTopBoundary(BCType::NEUMANN, (float) -q);
            column.setBottomBoundary(BCType::NEUMANN, 0.0f);
            std::vector<float> saved;
            SolveClock clock;
            const DryingRun d = dryToWilting(
                    q, 1.0, 43200.0, [&](double dt) { return column.solve((float) dt) > 0; }, [&] { return (double) column.getPressureHead(0); },
                    [&] { saved.assign(column.getPressureHeads().begin(), column.getPressureHeads().end()); }, [&] { column.setInitialCondition(saved); }, [&] { return (double) column.getTotalWaterStorage(); }, clock);
            if (!d.ok)
                return failure("did not reach the wilting point", clock.seconds);
            const double hs = column.getPressureHead(0);
            double water = 0.0;
            const int M = 2000;
            for (int m = 0; m < M; m++) {
                const double a = ri * std::pow(ro / ri, (double) m / M), b = ri * std::pow(ro / ri, (double) (m + 1) / M), c = 0.5 * (a + b);
                auto f = [&](double r) { return 2.0 * M_PI * r * soil.thetaOfHead(100.0 * ann.steadyHead(r, hs)); };
                water += (b - a) / 6.0 * (f(a) + 4.0 * f(c) + f(b));
            }
            const double t_pred = (soil.thetaOfHead(-100.0) * M_PI * (ro * ro - ri * ri) - water) / (q * 2.0 * M_PI * ri);
            Result r;
            r.time_s = clock.seconds;
            r.errors = {{"time_vs_steady_rate_rel", std::fabs(d.t - t_pred) / t_pred}, {"water_balance_rel", std::fabs(d.balance)}};
            r.values = {{"time_to_wilting_d", d.t / 86400.0}};
            r.work = {{"steps", d.steps}, {"retries", d.retries}};
            return r;
        }
    }

    //! Evaporation demand (5 mm/day) from a loam column until the surface reaches -150 m: the column runs out of
    //! deliverable water, so the last steps have no solution and must fail cleanly
    template<class Col = SoilColumn>
    Result colDrying() {
        Col column(60, 0.005f, getSoilPropertiesFromTexture("loam"));
        const double E = 5e-3 / 86400.0;
        ownRetries(column);
        column.setInitialCondition(-1.0f);
        column.setTopBoundary(BCType::NEUMANN, (float) -E);
        column.setBottomBoundary(BCType::NEUMANN, 0.0f);
        std::vector<float> saved;
        SolveClock clock;
        const DryingRun d = dryToWilting(
                E, 1.0, 21600.0, [&](double dt) { return column.solve((float) dt) > 0; }, [&] { return (double) column.getPressureHead(0); },
                [&] { saved.assign(column.getPressureHeads().begin(), column.getPressureHeads().end()); }, [&] { column.setInitialCondition(saved); }, [&] { return (double) column.getTotalWaterStorage(); }, clock);
        if (!d.ok)
            return failure("did not reach the wilting point", clock.seconds);
        Result r;
        r.time_s = clock.seconds;
        r.errors = {{"water_balance_rel", std::fabs(d.balance)}};
        r.values = {{"time_to_wilting_s", d.t}};
        r.work = {{"steps", d.steps}, {"retries", d.retries}};
        return r;
    }
} // namespace

// ============================================================================
// 3D solver cases
// ============================================================================

namespace {
    template<class S3>
    void setGpu(S3 &s, bool gpu) {
        auto p = s.getParams();
        p.use_gpu = gpu;
        s.setParams(p);
    }

    //! The same infiltration in the column and in a 1 x 1 x n 3D solver (backward Euler, arithmetic weighting):
    //! the discretizations are identical, so the answers must agree
    template<class S3 = SoilSolver3D>
    Result consistency3D(bool gpu) {
        if (gpu && !kGpuBuild)
            return unsupported("no CUDA");
        const uint nz = 50;
        const double dz = 0.02, q = 5e-7;
        const SoilProperties loam = getSoilPropertiesFromTexture("loam");
        SoilColumn column(nz, (float) dz, loam);
        SolverParameters cp;
        cp.K_weighting = KWeighting::ARITHMETIC;
        column.setSolverParameters(cp);
        column.setInitialCondition(-1.5f);
        column.setTopBoundary(BCType::NEUMANN, (float) q);
        column.setBottomBoundary(BCType::FREE_DRAINAGE, 0.0f);
        S3 s(1, 1, nz, 1.0, 1.0, dz, loam);
        s.setInitialCondition(-1.5);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, q);
        s.setBoundary(Face3D::ZMIN, BCType::FREE_DRAINAGE, 0.0);
        auto p = s.getParams();
        p.K_weighting = 1;
        p.bdf2 = false;
        p.use_gpu = gpu;
        s.setParams(p);
        SolveClock clock;
        double iters = 0.0;
        for (int n = 0; n < 48; n++) {
            if (column.solve(1800.0f) == 0)
                return failure("column step failed");
            const uint it = clock([&] { return s.solve(1800.0); });
            if (it == 0)
                return failure("3D step failed", clock.seconds);
            iters += it;
        }
        double err = 0.0;
        for (uint k = 0; k < nz; k++)
            err = std::max(err, std::fabs((double) column.getWaterContent(k) - s.getWaterContent(0, 0, nz - 1 - k)));
        Result r;
        r.time_s = clock.seconds;
        r.errors = {{"max_dtheta_vs_column", err}};
        r.work = {{"iterations", iters}};
        return r;
    }

    //! The column's drying case in a 1 x 1 x 60 3D solver (backward Euler); time to wilting must match the column's
    template<class S3 = SoilSolver3D>
    Result drying3D(bool gpu) {
        if (gpu && !kGpuBuild)
            return unsupported("no CUDA");
        const uint nz = 60;
        const double dz = 0.005, E = 5e-3 / 86400.0;
        const SoilProperties loam = getSoilPropertiesFromTexture("loam");
        SoilColumn column(nz, (float) dz, loam);
        ownRetries(column);
        column.setInitialCondition(-1.0f);
        column.setTopBoundary(BCType::NEUMANN, (float) -E);
        column.setBottomBoundary(BCType::NEUMANN, 0.0f);
        std::vector<float> hc;
        SolveClock col_clock, clock;
        const DryingRun dc = dryToWilting(
                E, 1.0, 21600.0, [&](double dt) { return column.solve((float) dt) > 0; }, [&] { return (double) column.getPressureHead(0); },
                [&] { hc.assign(column.getPressureHeads().begin(), column.getPressureHeads().end()); }, [&] { column.setInitialCondition(hc); }, [&] { return (double) column.getTotalWaterStorage(); }, col_clock);
        S3 s(1, 1, nz, 1.0, 1.0, dz, loam);
        s.setInitialCondition(-1.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, -E);
        s.setBoundary(Face3D::ZMIN, BCType::NEUMANN, 0.0);
        auto p = s.getParams();
        p.bdf2 = false;
        p.use_gpu = gpu;
        ownRetries3D(p);
        s.setParams(p);
        std::vector<double> h3;
        const DryingRun d = dryToWilting(
                E, 1.0, 21600.0, [&](double dt) { return s.solve(dt) > 0; }, [&] { return s.getPressureHead(0, 0, nz - 1); }, [&] { h3 = s.getPressureHeads(); }, [&] { s.setInitialCondition(h3); },
                [&] { return s.getTotalWaterStorage(); }, clock);
        if (!d.ok)
            return failure("did not reach the wilting point", clock.seconds);
        Result r;
        r.time_s = clock.seconds;
        r.errors = {{"water_balance_rel", std::fabs(d.balance)}, {"time_vs_column_rel", dc.ok ? std::fabs(d.t - dc.t) / dc.t : 1.0}};
        r.values = {{"time_to_wilting_s", d.t}};
        r.work = {{"steps", d.steps}, {"retries", d.retries}};
        return r;
    }

    //! Cells ~1e-10 m thick next to a -150 m held head (the DuMux annulus grid, vertical): column and 3D agree
    template<class S3 = SoilSolver3D>
    Result thinCells3D(bool gpu) {
        if constexpr (!HasCellGeometry<S3>::value || !HasNewtonAnalytic<S3>::value)
            return unsupported("no per-cell geometry");
        else {
            if (gpu && !kGpuBuild)
                return unsupported("no CUDA");
            const int N = 300;
            const double gr = 0.95, L = 5.8e-3, height = (1.0 - gr) / (1.0 - std::pow(gr, N)), q = 1.157e-8;
            std::vector<float> dz(N);
            for (int i = 0; i < N; i++)
                dz[i] = (float) (L * height * std::pow(gr, N - i - 1));
            const SoilProperties clay(0.1f, 0.4f, 1.0f, 1.1f, 1.1574e-6f);
            SoilColumn column(dz, clay);
            column.setInitialCondition(-100.0f);
            column.setTopBoundary(BCType::DIRICHLET, -150.0f);
            column.setBottomBoundary(BCType::NEUMANN, (float) -q);
            for (int n = 0; n < 4; n++)
                if (column.solve(1e9f) == 0)
                    return failure("column steady solve failed");
            S3 s(1, 1, N, 1.0, 1.0, L / N, clay);
            std::vector<double> zc(N), thk(N);
            double z = 0.0;
            for (int k = 0; k < N; k++) {
                thk[k] = dz[N - 1 - k];
                zc[k] = z + 0.5 * thk[k];
                z += thk[k];
            }
            s.setCellGeometry(zc, thk);
            s.setInitialCondition(-100.0);
            s.setBoundary(Face3D::ZMAX, BCType::DIRICHLET, -150.0);
            s.setBoundary(Face3D::ZMIN, BCType::NEUMANN, q);
            auto p = s.getParams();
            p.nonlinear_method = S3::NonlinearMethod::NEWTON_ANALYTIC;
            p.bdf2 = false;
            p.use_gpu = gpu;
            s.setParams(p);
            SolveClock clock;
            double iters = 0.0;
            for (int n = 0; n < 4; n++) {
                const uint it = clock([&] { return s.solve(1e9); });
                if (it == 0)
                    return failure("3D steady solve failed", clock.seconds);
                iters += it;
            }
            double err = 0.0;
            for (int k = 0; k < N; k++)
                err = std::max(err, std::fabs((double) column.getPressureHead(k) - s.getPressureHead(0, 0, N - 1 - k)));
            Result r;
            r.time_s = clock.seconds;
            r.errors = {{"max_dh_vs_column_m", err}};
            r.work = {{"iterations", iters}};
            return r;
        }
    }

    //! 3D drip irrigation: 2 x 2 x 1 m, loam over sandy loam, water table at the base, four 2 L/h emitters over
    //! 0.1 m x 0.1 m footprints (aligned with cell faces at 5, 2.5 and 1.25 cm) for 12 h, then 36 h of redistribution,
    //! fixed steps. Water content averaged onto the 5 cm grid (20 x 40 x 40, x fastest, bottom layer first).
    template<class S3 = SoilSolver3D>
    Result drip3D(double d, double dt, bool gpu, int weighting, bool bdf2, std::vector<double> *theta5) {
        if (gpu && !kGpuBuild)
            return unsupported("no CUDA");
        const double Lx = 2.0, Ly = 2.0, Lz = 1.0, day = 86400.0, q_emitter = 2.0e-3 / 3600.0;
        const uint nx = (uint) std::lround(Lx / d), ny = (uint) std::lround(Ly / d), nz = (uint) std::lround(Lz / d);
        const SoilProperties loam(0.078f, 0.43f, 3.6f, 1.56f, (float) (0.2496 / day));
        const SoilProperties sloam(0.065f, 0.41f, 7.5f, 1.89f, (float) (1.061 / day));
        S3 s(nx, ny, nz, d, d, d, sloam);
        std::vector<double> h0(nx * ny * nz);
        for (uint k = 0; k < nz; k++) {
            const double zc = (k + 0.5) * d;
            for (uint j = 0; j < ny; j++)
                for (uint i = 0; i < nx; i++) {
                    if (Lz - zc < 0.3)
                        s.setSoilProperties(i, j, k, loam);
                    h0[i + nx * (j + ny * k)] = -zc;
                }
        }
        s.setInitialCondition(h0);
        s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, 0.0);
        std::vector<uint> foot;
        for (uint i = 0; i < nx; i++)
            if (std::fabs((i + 0.5) * d - 0.5) < 0.05 || std::fabs((i + 0.5) * d - 1.5) < 0.05)
                foot.push_back(i);
        const double per_cell = q_emitter / ((foot.size() / 2) * (foot.size() / 2)) / (d * d * d);
        for (uint j: foot)
            for (uint i: foot)
                s.setSource(i, j, nz - 1, per_cell);
        auto p = s.getParams();
        p.use_gpu = gpu;
        if (weighting >= 0)
            p.K_weighting = weighting;
        if (bdf2)
            p.bdf2 = true;
        else
            p.bdf2 = false;
        s.setParams(p);
        SolveClock clock;
        double iters = 0.0, fails = 0.0;
        for (int period = 0; period < 2; period++) {
            const double T = (period == 0 ? 0.5 : 1.5) * day;
            if (period == 1) {
                s.clearSources();
                if constexpr (HasResetHistory<S3>::value)
                    s.resetHistory();
            }
            for (int n = 0; n < (int) std::lround(T / dt); n++) {
                const uint it = clock([&] { return s.solve(dt); });
                if (it == 0)
                    fails++;
                iters += it;
            }
        }
        const uint f = (uint) std::lround(0.05 / d), cx = nx / f, cy = ny / f, cz = nz / f;
        theta5->assign((size_t) cx * cy * cz, 0.0);
        for (uint K = 0; K < cz; K++)
            for (uint J = 0; J < cy; J++)
                for (uint I = 0; I < cx; I++) {
                    double sum = 0.0;
                    for (uint k = K * f; k < (K + 1) * f; k++)
                        for (uint j = J * f; j < (J + 1) * f; j++)
                            for (uint i = I * f; i < (I + 1) * f; i++)
                                sum += s.getWaterContent(i, j, k);
                    (*theta5)[I + cx * (J + cy * K)] = sum / (f * f * f);
                }
        Result r;
        r.time_s = clock.seconds;
        const double injected = 4.0 * q_emitter * 0.5 * day;
        r.errors = {{"mass_residual_rel", std::fabs(s.getMassBalanceResidual()) / injected}};
        r.work = {{"nonlinear_iterations", iters}, {"failed_steps", fails}};
        return r;
    }

    std::string referenceDir() {
        if (const char *e = std::getenv("SOIL_REGRESSION_REFERENCE"))
            return e;
        return "reference";
    }

    Result dripCase(double d, bool gpu) {
        std::vector<double> th;
        Result r = drip3D(d, 900.0, gpu, -1, SoilSolver3D::Params().bdf2, &th); // the generation's default integrator (BDF2 up to 25d7628, backward Euler since)
        if (r.status != "ok")
            return r;
        std::ifstream f(referenceDir() + "/drip3d_theta5cm.f32", std::ios::binary);
        std::vector<float> ref(th.size());
        f.read(reinterpret_cast<char *>(ref.data()), (std::streamsize) (ref.size() * sizeof(float)));
        if (!f)
            return failure("missing reference " + referenceDir() + "/drip3d_theta5cm.f32");
        double se = 0.0, mx = 0.0;
        for (size_t c = 0; c < th.size(); c++) {
            const double e = th[c] - ref[c];
            se += e * e;
            mx = std::max(mx, std::fabs(e));
        }
        r.errors.insert(r.errors.begin(), {{"rmse_theta_vs_reference", std::sqrt(se / th.size())}, {"max_theta_vs_reference", mx}});
        return r;
    }

    // ---- MODFLOW 6 comparison cases (Vázquez-Gasty et al. 2026, Tests 2 and 3), external data ----

    std::vector<double> loadBin(const std::string &path, size_t n, bool &ok) {
        std::vector<double> v(n);
        std::ifstream f(path, std::ios::binary);
        f.read(reinterpret_cast<char *>(v.data()), (std::streamsize) (n * sizeof(double)));
        ok = ok && (bool) f;
        return v;
    }

    //! Vázquez-Gasty et al. (2026) Test 1: a 30.48 m layered column (999 cells, 13 soils from MODFLOW 6's input), recharge
    //! 9.44834e-4 m/day on top, water table at the base, from hydraulic head 0 (MODFLOW's start) to steady state at
    //! 365,250 days, compared with MODFLOW 6's heads. mode 0: MODFLOW 6's schedule (100 fixed steps from its start,
    //! hydraulic head 0), closure (largest head change below 1 mm) and full Newton steps once ||F|| < 4e-4, upstream
    //! k_r. Modes 1 and 2: adaptive steps (dt_target_rel_change = 1) from hydrostatic equilibrium with the water table
    //! at the base, upstream k_r (1) or the generation's default weighting (2). Mode 3: the generation's defaults with
    //! adaptive steps from MODFLOW 6's saturated, incompressible start (a known failure).
    template<class S3 = SoilSolver3D>
    Result mf6Test1(int mode) {
        const bool mf6_steps = mode == 0;
        if constexpr (!HasNewtonAnalytic<S3>::value)
            return unsupported("no NEWTON_ANALYTIC");
        else {
            const uint Nz = 999;
            const double depth = 30.48, dz = depth / Nz, day = 86400.0;
            struct Band {
                uint start;
                double Ks, phi, alpha, beta, Sr;
            };
            std::vector<Band> bands;
            std::vector<double> ref;
            {
                std::ifstream f(referenceDir() + "/test1_layers.txt");
                Band b;
                while (f >> b.start >> b.Ks >> b.phi >> b.alpha >> b.beta >> b.Sr)
                    bands.push_back(b);
                std::ifstream g(referenceDir() + "/test1_mf6_psi.txt");
                double v;
                while (g >> v)
                    ref.push_back(v);
            }
            if (bands.empty() || ref.size() != Nz)
                return failure("missing Test 1 data in " + referenceDir());
            S3 s(1, 1, Nz, 1.0, 1.0, dz, SoilProperties());
            std::vector<double> h0(Nz);
            for (uint n = 0; n < Nz; n++) { // n: MODFLOW node from the top
                const Band *b = &bands[0];
                for (const auto &bb: bands)
                    if (bb.start <= n)
                        b = &bb;
                const uint k = Nz - 1 - n;
                s.setSoilProperties(0, 0, k, SoilProperties((float) (b->Sr * b->phi), (float) b->phi, (float) b->alpha, (float) b->beta, (float) (b->Ks / day)));
                h0[k] = (mode == 1 || mode == 2) ? -depth + (n + 0.5) * dz : (n + 0.5) * dz; // hydrostatic, or head 0
            }
            s.setInitialCondition(h0);
            s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 9.448340e-4 / day);
            s.setBoundary(Face3D::ZMIN, BCType::DIRICHLET, 0.0);
            auto p = s.getParams();
            p.nonlinear_method = S3::NonlinearMethod::NEWTON_ANALYTIC;
            p.bdf2 = false;
            p.dt_max = 3652.5 * day;
            p.dt_min = 1e-3;
            if (mf6_steps) {
                p.K_weighting = 4;
                p.newton_dvclose = 1e-3;
                p.newton_backtrack_limit = 4e-4;
            }
            if (mode == 1)
                p.K_weighting = 4;
            if (mode == 1 || mode == 2)
                p.dt_target_rel_change = 1.0;
            s.setParams(p);
            const double t_end = 365250.0 * day;
            SolveClock clock;
            double steps = 0.0;
            if (mf6_steps) {
                for (int c = 0; c < 100; c++) {
                    if (clock([&] { return s.solve(t_end / 100); }) == 0)
                        return failure("step " + std::to_string(c + 1) + " failed", clock.seconds);
                    steps++;
                }
            } else {
                double dt = 60.0, t = 0.0;
                for (int c = 0; c < 100; c++) {
                    uint st = 0;
                    t += clock([&] { return s.solveAdaptive(t_end / 100, dt, &st); });
                    steps += st;
                    if (t < (c + 1) * t_end / 100 * (1.0 - 1e-9))
                        return failure("stopped at " + std::to_string(t / day) + " days", clock.seconds);
                }
            }
            double err = 0.0;
            for (uint n = 0; n < Nz; n++)
                err = std::max(err, std::fabs(s.getPressureHead(0, 0, Nz - 1 - n) - ref[n]));
            Result r;
            r.time_s = clock.seconds;
            r.errors = {{"max_dpsi_vs_mf6_m", err}};
            r.work = {{"steps", steps}};
            return r;
        }
    }

    //! A 106 x 25 x 25 terrain-following hillslope, a year of daily recharge, specified-head cells and specific
    //! storage, upstream-k_r weighting and backward Euler as in the published comparison. Needs the case data in
    //! $SOIL_REGRESSION_MF6 (test2/, test3/ from mf6_richards, plus mf6_final_psi.bin: MF6's final pressure heads).
    template<class S3 = SoilSolver3D>
    Result mf6Case(const std::string &which, bool gpu) {
        if constexpr (!HasCellGeometry<S3>::value || !HasFixedHead<S3>::value || !HasNewtonAnalytic<S3>::value)
            return unsupported("no per-cell geometry / specified heads / specific storage");
        else {
            if (gpu && !kGpuBuild)
                return unsupported("no CUDA");
            const char *root = std::getenv("SOIL_REGRESSION_MF6");
            if (!root)
                return unsupported("set SOIL_REGRESSION_MF6 to the MODFLOW 6 case data");
            const std::string dir = std::string(root) + "/" + which + "/";
            const uint nx = 106, ny = 25, nz = 25, N = nx * ny * nz;
            const double day = 86400.0;
            bool ok = true;
            auto zc = loadBin(dir + "zc.bin", N, ok), thk = loadBin(dir + "thk.bin", N, ok), Ks = loadBin(dir + "Ks.bin", N, ok), ts = loadBin(dir + "theta_s.bin", N, ok);
            auto al = loadBin(dir + "alpha.bin", N, ok), nn = loadBin(dir + "n.bin", N, ok), sr = loadBin(dir + "sr.bin", N, ok), psi0 = loadBin(dir + "psi0.bin", N, ok), fixed = loadBin(dir + "fixed_psi.bin", N, ok);
            auto ssv = loadBin(dir + "ss.bin", N, ok), rch = loadBin(dir + "rch.bin", (size_t) 365 * nx * ny, ok), mf6 = loadBin(dir + "mf6_final_psi.bin", N, ok);
            if (!ok)
                return failure("cannot read the case data in " + dir);
            S3 s(nx, ny, nz, 5.0, 5.0, 1.0, SoilProperties());
            s.setCellGeometry(zc, thk);
            for (uint k = 0; k < nz; k++)
                for (uint j = 0; j < ny; j++)
                    for (uint i = 0; i < nx; i++) {
                        const uint c = i + nx * (j + ny * k);
                        s.setSoilProperties(i, j, k, SoilProperties((float) (sr[c] * ts[c]), (float) ts[c], (float) al[c], (float) nn[c], (float) (Ks[c] / day)));
                    }
            s.setInitialCondition(psi0);
            s.setSpecificStorage(ssv);
            for (uint k = 0; k < nz; k++)
                for (uint j = 0; j < ny; j++)
                    for (uint i = 0; i < nx; i++)
                        if (!std::isnan(fixed[i + nx * (j + ny * k)]))
                            s.setFixedHead(i, j, k, fixed[i + nx * (j + ny * k)]);
            auto p = s.getParams();
            p.nonlinear_method = S3::NonlinearMethod::NEWTON_ANALYTIC;
            p.K_weighting = 4;
            p.bdf2 = false;
            p.dt_max = day;
            p.use_gpu = gpu;
            s.setParams(p);
            SolveClock clock;
            double dt = 3600.0, steps = 0.0, rch_total = 0.0;
            for (int d = 0; d < 365; d++) {
                s.clearSources();
                for (uint j = 0; j < ny; j++)
                    for (uint i = 0; i < nx; i++) {
                        const double q = rch[(size_t) d * nx * ny + i + nx * j] / day;
                        const uint c = i + nx * (j + ny * (nz - 1));
                        if (q != 0.0)
                            s.setSource(i, j, nz - 1, q / thk[c]);
                        if (std::isnan(fixed[c]))
                            rch_total += q * 25.0 * day;
                    }
                uint st = 0;
                const double adv = clock([&] { return s.solveAdaptive(day, dt, &st); });
                steps += st;
                if (adv < day * (1.0 - 1e-9))
                    return failure("stopped on day " + std::to_string(d + 1), clock.seconds);
            }
            // MF6 node order: layer from the top, row from max y, column
            double mx = 0.0, sum = 0.0;
            for (uint l = 0; l < nz; l++)
                for (uint r = 0; r < ny; r++)
                    for (uint c = 0; c < nx; c++) {
                        const double e = std::fabs(s.getPressureHead(c, ny - 1 - r, nz - 1 - l) - mf6[c + nx * (r + ny * l)]);
                        mx = std::max(mx, e);
                        sum += e;
                    }
            Result r;
            r.time_s = clock.seconds;
            r.errors = {{"max_dpsi_vs_mf6_m", mx}, {"mean_dpsi_vs_mf6_m", sum / N}, {"mass_residual_rel", std::fabs(s.getMassBalanceResidual()) / rch_total}};
            r.work = {{"steps", steps}};
            return r;
        }
    }
} // namespace

// ============================================================================
// Registry
// ============================================================================

namespace {
    // An emitter switched on and off under adaptive stepping, with the generation's default integrator: a closed loam block,
    // 6 h of emitter, then 18 h without. The water stored must rise by exactly what the emitter delivered; the step sizes
    // change at every switch, which variable-step BDF2 does not conserve.
    template<class S3 = SoilSolver3D>
    Result emitterSwitchAdaptive(bool gpu) {
        if (gpu && !kGpuBuild)
            return unsupported("no CUDA");
        const double d = 0.05, q = 2.0e-3 / 3600.0;
        const uint nx = 24, ny = 24, nz = 16;
        const SoilProperties loam = getSoilPropertiesFromTexture("loam");
        S3 s(nx, ny, nz, d, d, d, loam);
        s.setInitialCondition(std::vector<double>(s.numCells(), -3.0));
        s.setBoundary(Face3D::ZMIN, BCType::NEUMANN, 0.0);
        s.setBoundary(Face3D::ZMAX, BCType::NEUMANN, 0.0);
        auto p = s.getParams();
        p.use_gpu = gpu;
        p.dt_max = 1800.0;
        p.dt_min = 1.0;
        s.setParams(p);
        std::vector<std::pair<uint, uint>> cells;
        for (uint j = 0; j < ny; j++)
            for (uint i = 0; i < nx; i++)
                if (std::hypot((i + 0.5) * d - 0.6, (j + 0.5) * d - 0.6) <= 0.08)
                    cells.push_back({i, j});
        const double W0 = s.getTotalWaterStorage();
        SolveClock clock;
        double dt = 60.0, water_in = 0.0;
        uint steps = 0;
        for (int n = 0; n < 48; n++) {
            const double t = n * 0.5;
            const bool on = t >= 2.0 && t < 8.0;
            s.clearSources();
            if (on)
                for (auto c: cells)
                    s.setSource(c.first, c.second, nz - 1, q / cells.size() / (d * d * d));
            uint taken = 0;
            double advanced = 0.0;
            clock([&] {
                advanced = s.solveAdaptive(1800.0, dt, &taken);
                return taken;
            });
            if (advanced < 1800.0 * (1.0 - 1e-9))
                return failure("adaptive step failed", clock.seconds);
            steps += taken;
            water_in += on ? q * 1800.0 : 0.0;
        }
        Result r;
        r.time_s = clock.seconds;
        r.errors = {{"water_storage_error_rel", std::fabs(s.getTotalWaterStorage() - W0 - water_in) / water_in}};
        r.work = {{"steps", (double) steps}};
        return r;
    }

    struct Case {
        std::string name, tier;
        std::function<Result()> run;
    };
    std::vector<Case> registry() {
        return {
                {"col_mass_balance", "quick", [] { return colMassBalance(); }},
                {"col_gardner_steady", "quick", [] { return colGardnerSteady(); }},
                {"col_infiltration_sand", "quick", [] { return colInfiltration(kVdSand, 0.2824, -42.78, 0.1, 25.0, 0.3, false); }},
                {"col_infiltration_loam", "quick", [] { return colInfiltration(kVdLoam, 0.43, -40.03, 0.2, 200.0, 1.0, false); }},
                {"col_infiltration_clay", "full", [] { return colInfiltration(kVdClay, 0.40, -24.41, 0.1, 40.0, 0.5, false); }},
                {"col_infiltration_clay_spline", "quick", [] { return colInfiltration(kVdClay, 0.40, -24.41, 0.1, 40.0, 0.5, true); }},
                {"col_evaporation_sand", "quick", [] { return colEvaporation(kVdSand, -40.0, 1.0, 1.0, 360.0); }},
                {"col_evaporation_loam1", "quick", [] { return colEvaporation(kVdLoam, -200.0, 1.0, 10.0, 7200.0); }},
                {"col_evaporation_loam2", "quick", [] { return colEvaporation(kVdLoam, -200.0, 3.0, 2.0, 7200.0); }},
                {"col_evaporation_clay", "quick", [] { return colEvaporation(kVdClay, -200.0, 3.0, 6.0, 1200.0); }},
                {"col_annulus_steady_clay", "quick", [] { return colAnnulusSteady(kVdClay); }},
                {"col_annulus_steady_loam", "quick", [] { return colAnnulusSteady(kVdLoam); }},
                {"col_annulus_wilting_clay", "quick", [] { return colAnnulusWilting(kVdClay); }},
                {"col_annulus_wilting_loam", "quick", [] { return colAnnulusWilting(kVdLoam); }},
                {"col_drying", "quick", [] { return colDrying(); }},
                {"3d_consistency_cpu", "quick", [] { return consistency3D(false); }},
                {"3d_consistency_gpu", "quick", [] { return consistency3D(true); }},
                {"3d_drying_cpu", "quick", [] { return drying3D(false); }},
                {"3d_drying_gpu", "quick", [] { return drying3D(true); }},
                {"3d_thin_cells_cpu", "quick", [] { return thinCells3D(false); }},
                {"3d_thin_cells_gpu", "quick", [] { return thinCells3D(true); }},
                {"3d_drip_cpu", "quick", [] { return dripCase(0.05, false); }},
                {"3d_drip_gpu", "quick", [] { return dripCase(0.05, true); }},
                {"3d_drip_fine_cpu", "full", [] { return dripCase(0.025, false); }},
                {"3d_drip_fine_gpu", "full", [] { return dripCase(0.025, true); }},
                {"3d_emitter_switch_adaptive_cpu", "quick", [] { return emitterSwitchAdaptive(false); }},
                {"3d_emitter_switch_adaptive_gpu", "quick", [] { return emitterSwitchAdaptive(true); }},
                {"mf6_test1_mf6_steps", "quick", [] { return mf6Test1(0); }},
                {"mf6_test1_adaptive_ukr", "quick", [] { return mf6Test1(1); }},
                {"mf6_test1_adaptive", "quick", [] { return mf6Test1(2); }},
                {"mf6_test1_adaptive_mf6_start", "full", [] { return mf6Test1(3); }},
                {"mf6_test2_cpu", "full", [] { return mf6Case("test2", false); }},
                {"mf6_test2_gpu", "full", [] { return mf6Case("test2", true); }},
                {"mf6_test3_cpu", "full", [] { return mf6Case("test3", false); }},
                {"mf6_test3_gpu", "full", [] { return mf6Case("test3", true); }},
        };
    }
} // namespace

int main(int argc, char *argv[]) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "list") {
        const std::string tier = argc > 2 ? argv[2] : "full";
        for (const Case &c: registry())
            if (tier == "full" || c.tier == tier)
                std::printf("%s %s\n", c.name.c_str(), c.tier.c_str());
        return 0;
    }
    if (cmd == "run" && argc > 2) {
        for (const Case &c: registry())
            if (c.name == argv[2]) {
                printResult(c.name, c.run());
                return 0;
            }
        std::fprintf(stderr, "unknown case %s\n", argv[2]);
        return 2;
    }
    if (cmd == "reference" && argc > 2) {
        // Grid- and time-converged reference for the drip cases: 1.25 cm cells, 112.5 s steps, BDF2, averaged over
        // arithmetic and upstream-k_r weighting (their difference bounds the remaining discretization error)
        std::vector<double> a, b;
        const bool gpu = kGpuBuild;
        Result ra = drip3D(0.0125, 112.5, gpu, 1, true, &a);
        Result rb = drip3D(0.0125, 112.5, gpu, 4, true, &b);
        if (ra.status != "ok" || rb.status != "ok") {
            std::fprintf(stderr, "reference runs failed\n");
            return 1;
        }
        std::vector<float> ref(a.size());
        double spread = 0.0;
        for (size_t c = 0; c < a.size(); c++) {
            ref[c] = (float) (0.5 * (a[c] + b[c]));
            spread = std::max(spread, std::fabs(a[c] - b[c]));
        }
        std::ofstream f(argv[2], std::ios::binary);
        f.write(reinterpret_cast<const char *>(ref.data()), (std::streamsize) (ref.size() * sizeof(float)));
        std::printf("reference written to %s (%zu values); max |arithmetic - upstream k_r| = %.3e; solve %.1f + %.1f s\n", argv[2], ref.size(), spread, ra.time_s, rb.time_s);
        return 0;
    }
    std::fprintf(stderr, "usage: %s list [quick|full] | run <case> | reference <out.f32>\n", argv[0]);
    return 2;
}
