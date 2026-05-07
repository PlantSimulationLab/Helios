/** \file "FluspectB.cpp" Implementation of Fluspect-B leaf fluorescence model. */

#include "FluspectB.h"
#include "Context.h"
#include "global.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace helios {

    // ------------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------------
    namespace {

        // Exponential integral E_1(x) for x > 0. Port of the Chebyshev / NAG S13AAF
        // approximation used in LeafOptics::transmittance() (plugins/leafoptics/src/
        // LeafOptics.cpp), rearranged to return E_1 directly rather than the
        // (1 - k)·exp(-k) + k²·E_1(k) composite.
        //
        // Accuracy is approximately 1e-12 relative for x in (0, 85).
        double expint(double x) {
            if (x <= 0.0) {
                // E_1 diverges at 0 and is undefined for x < 0. Caller guards
                // non-positive Kall values before invoking this.
                return std::numeric_limits<double>::infinity();
            }
            if (x < 4.0) {
                double xx = 0.5 * x - 1.0;
                double yy;
                // Chebyshev polynomial (ported verbatim from LeafOptics).
                yy = (((((((((((((((-3.60311230482612224e-13L * xx + 3.46348526554087424e-12L) * xx - 2.99627399604128973e-11L) * xx + 2.57747807106988589e-10L) * xx - 2.09330568435488303e-9L) * xx + 1.59501329936987818e-8L) *
                                  xx -
                                  1.13717900285428895e-7L) *
                                 xx +
                                 7.55292885309152956e-7L) *
                                xx -
                                4.64980751480619431e-6L) *
                               xx +
                               2.63830365675408129e-5L) *
                              xx -
                              1.37089870978830576e-4L) *
                             xx +
                             6.47686503728103400e-4L) *
                            xx -
                            2.76060141343627983e-3L) *
                           xx +
                           1.05306034687449505e-2L) *
                          xx -
                          3.57191348753631956e-2L) *
                         xx +
                         1.07774527938978692e-1L) *
                        xx -
                        2.96997075145080963e-1L;
                yy = (yy * xx + 8.64664716763387311e-1L) * xx + 7.42047691268006429e-1L;
                return yy - std::log(x);
            }
            if (x < 85.0) {
                double xx = 14.5 / (x + 3.25) - 1.0;
                double yy;
                yy = (((((((((((((((-1.62806570868460749e-12L * xx - 8.95400579318284288e-13L) * xx - 4.08352702838151578e-12L) * xx - 1.45132988248537498e-11L) * xx - 8.35086918940757852e-11L) * xx - 2.13638678953766289e-10L) * xx -
                                  1.10302431467069770e-9L) *
                                 xx -
                                 3.67128915633455484e-9L) *
                                xx -
                                1.66980544304104726e-8L) *
                               xx -
                               6.11774386401295125e-8L) *
                              xx -
                              2.70306163610271497e-7L) *
                             xx -
                             1.05565006992891261e-6L) *
                            xx -
                            4.72090467203711484e-6L) *
                           xx -
                           1.95076375089955937e-5L) *
                          xx -
                          9.16450482931221453e-5L) *
                         xx -
                         4.05892130452128677e-4L) *
                        xx -
                        2.14213055000334718e-3L;
                yy = ((yy * xx - 1.06374875116569657e-2L) * xx - 8.50699154984571871e-2L) * xx + 9.23755307807784058e-1L;
                return std::exp(-x) * yy / x;
            }
            return 0.0;
        }

        // Angular-average transmittivity of a dielectric interface with refractive
        // index `nr`, averaged from normal to `alpha` degrees. Port of calctav.m
        // (Stern 1964, Allen 1973; used internally by Fluspect/PROSPECT).
        double calctav(double alpha_deg, double nr) {
            const double rd = M_PI / 180.0;
            const double n2 = nr * nr;
            const double np = n2 + 1.0;
            const double nm = n2 - 1.0;
            const double a = (nr + 1.0) * (nr + 1.0) / 2.0;
            const double k = -nm * nm / 4.0;
            const double sa = std::sin(alpha_deg * rd);
            double b1 = 0.0;
            if (alpha_deg != 90.0) {
                b1 = std::sqrt((sa * sa - np / 2.0) * (sa * sa - np / 2.0) + k);
            }
            const double b2 = sa * sa - np / 2.0;
            const double b = b1 - b2;
            const double b3 = b * b * b;
            const double a3 = a * a * a;
            const double ts = (k * k / (6.0 * b3) + k / b - b / 2.0) - (k * k / (6.0 * a3) + k / a - a / 2.0);
            const double tp1 = -2.0 * n2 * (b - a) / (np * np);
            const double tp2 = -2.0 * n2 * np * std::log(b / a) / (nm * nm);
            const double tp3 = n2 * (1.0 / b - 1.0 / a) / 2.0;
            const double tp4 = 16.0 * n2 * n2 * (n2 * n2 + 1.0) * std::log((2.0 * np * b - nm * nm) / (2.0 * np * a - nm * nm)) / (np * np * np * nm * nm);
            const double tp5 = 16.0 * n2 * n2 * n2 * (1.0 / (2.0 * np * b - nm * nm) - 1.0 / (2.0 * np * a - nm * nm)) / (np * np * np);
            const double tp = tp1 + tp2 + tp3 + tp4 + tp5;
            return (ts + tp) / (2.0 * sa * sa);
        }

        // Linear interpolation of `y(x_src)` onto `x_dst`. Requires monotonically
        // increasing x_src. Out-of-range x_dst values clamp to endpoints.
        std::vector<double> interp1(const std::vector<float> &x_src, const std::vector<double> &y_src, const std::vector<float> &x_dst) {
            std::vector<double> y_dst(x_dst.size());
            size_t j = 0;
            for (size_t i = 0; i < x_dst.size(); ++i) {
                const double x = x_dst[i];
                if (x <= x_src.front()) {
                    y_dst[i] = y_src.front();
                    continue;
                }
                if (x >= x_src.back()) {
                    y_dst[i] = y_src.back();
                    continue;
                }
                while (j + 1 < x_src.size() && x_src[j + 1] < x) {
                    ++j;
                }
                const double x0 = x_src[j];
                const double x1 = x_src[j + 1];
                const double y0 = y_src[j];
                const double y1 = y_src[j + 1];
                y_dst[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0);
            }
            return y_dst;
        }

        // Load one globaldata_vec2 spectrum from the given XML file into a vector of
        // (wavelength, value) pairs. Throws on missing/invalid entries.
        void loadCoefficientSeries(Context &ctx, const std::string &xml_path, const std::string &label, std::vector<float> &wavelengths_nm, std::vector<float> &values) {
            if (!ctx.doesGlobalDataExist(label.c_str())) {
                ctx.loadXML(xml_path.c_str(), true);
            }
            if (!ctx.doesGlobalDataExist(label.c_str())) {
                helios_runtime_error("ERROR (FluspectB::loadFluspectOptipar): Coefficient '" + label + "' not found in " + xml_path);
            }
            if (ctx.getGlobalDataType(label.c_str()) != HELIOS_TYPE_VEC2) {
                helios_runtime_error("ERROR (FluspectB::loadFluspectOptipar): Coefficient '" + label + "' is not of type HELIOS_TYPE_VEC2.");
            }
            std::vector<vec2> spectrum;
            ctx.getGlobalData(label.c_str(), spectrum);
            if (spectrum.empty()) {
                helios_runtime_error("ERROR (FluspectB::loadFluspectOptipar): Coefficient '" + label + "' is empty.");
            }
            const bool reserve_wavelengths = wavelengths_nm.empty();
            if (reserve_wavelengths) {
                wavelengths_nm.reserve(spectrum.size());
            } else if (wavelengths_nm.size() != spectrum.size()) {
                helios_runtime_error("ERROR (FluspectB::loadFluspectOptipar): Coefficient '" + label + "' has " + std::to_string(spectrum.size()) + " samples but previous coefficients have " + std::to_string(wavelengths_nm.size()) +
                                     " - all must share the same grid.");
            }
            values.clear();
            values.reserve(spectrum.size());
            for (size_t i = 0; i < spectrum.size(); ++i) {
                if (reserve_wavelengths) {
                    wavelengths_nm.push_back(spectrum[i].x);
                } else if (std::abs(wavelengths_nm[i] - spectrum[i].x) > 1e-3f) {
                    helios_runtime_error("ERROR (FluspectB::loadFluspectOptipar): Coefficient '" + label + "' wavelength mismatch at index " + std::to_string(i));
                }
                values.push_back(spectrum[i].y);
            }
        }

    } // namespace

    // ------------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------------

    void loadFluspectOptipar(const std::string &xml_path, FluspectOptipar &optipar) {
        // Temporary Context just to load the XML globaldata. Using a local Context
        // keeps the Optipar data self-contained and independent of the user's scene.
        Context tmp_ctx;

        optipar.wavelengths_nm.clear();
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_nr", optipar.wavelengths_nm, optipar.nr);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Kab", optipar.wavelengths_nm, optipar.Kab);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Kca", optipar.wavelengths_nm, optipar.Kca);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_KcaV", optipar.wavelengths_nm, optipar.KcaV);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_KcaZ", optipar.wavelengths_nm, optipar.KcaZ);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Kdm", optipar.wavelengths_nm, optipar.Kdm);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Kw", optipar.wavelengths_nm, optipar.Kw);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Ks", optipar.wavelengths_nm, optipar.Ks);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Kant", optipar.wavelengths_nm, optipar.Kant);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Kp", optipar.wavelengths_nm, optipar.Kp);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_Kcbc", optipar.wavelengths_nm, optipar.Kcbc);
        loadCoefficientSeries(tmp_ctx, xml_path, "fluspect_optipar_phi", optipar.wavelengths_nm, optipar.phi);
    }

    FluspectKernel computeFluspectKernel(const FluspectBiochemistry &biochem, const FluspectOptipar &optipar, float excitation_step_nm) {
        if (excitation_step_nm <= 0.f) {
            helios_runtime_error("ERROR (computeFluspectKernel): excitation_step_nm must be > 0.");
        }
        const size_t nw = optipar.wavelengths_nm.size();
        if (nw < 2) {
            helios_runtime_error("ERROR (computeFluspectKernel): Optipar wavelength grid is empty or degenerate.");
        }

        // ---- Choose Kca (V2Z linear combination of KcaV and KcaZ) ----
        // MATLAB: if V2Z == -999 use legacy optipar.Kca, else (1-V2Z)*KcaV + V2Z*KcaZ.
        // We always use the V2Z combination (there's no -999 sentinel in our API).
        std::vector<double> Kca(nw);
        for (size_t i = 0; i < nw; ++i) {
            Kca[i] = (1.0 - biochem.V2Z) * optipar.KcaV[i] + biochem.V2Z * optipar.KcaZ[i];
        }

        // ---- PROSPECT calculations ----
        // Kall = (Cab*Kab + Cca*Kca + ...) / N
        std::vector<double> Kall(nw);
        for (size_t i = 0; i < nw; ++i) {
            Kall[i] = (biochem.Cab * optipar.Kab[i] + biochem.Cca * Kca[i] + biochem.Cdm * optipar.Kdm[i] + biochem.Cw * optipar.Kw[i] + biochem.Cs * optipar.Ks[i] + biochem.Cant * optipar.Kant[i] + biochem.Cp * optipar.Kp[i] +
                      biochem.Cbc * optipar.Kcbc[i]) /
                     biochem.N;
        }

        // tau = (1-Kall)*exp(-Kall) + Kall^2 * E_1(Kall) for Kall > 0, else 1.
        // kChlrel = Cab*Kab / (Kall * N) for Kall > 0, else 0.
        std::vector<double> tau(nw, 1.0);
        std::vector<double> kChlrel(nw, 0.0);
        for (size_t i = 0; i < nw; ++i) {
            const double K = Kall[i];
            if (K > 0.0) {
                tau[i] = (1.0 - K) * std::exp(-K) + K * K * expint(K);
                kChlrel[i] = biochem.Cab * optipar.Kab[i] / (K * biochem.N);
            }
        }

        // Interface transmittivities: talf=tav(59), t12=tav(90), t21=t12/n^2.
        std::vector<double> talf(nw), ralf(nw), t12(nw), r12(nw), t21(nw), r21(nw);
        for (size_t i = 0; i < nw; ++i) {
            talf[i] = calctav(59.0, optipar.nr[i]);
            ralf[i] = 1.0 - talf[i];
            t12[i] = calctav(90.0, optipar.nr[i]);
            r12[i] = 1.0 - t12[i];
            t21[i] = t12[i] / (optipar.nr[i] * optipar.nr[i]);
            r21[i] = 1.0 - t21[i];
        }

        // Top-surface and bottom-surface layer transmittance/reflectance.
        // Stokes solution for N-layer stack.
        std::vector<double> refl(nw), tran(nw);
        std::vector<double> r_mes(nw), t_mes(nw);  // mesophyll r/t after interface removal
        for (size_t i = 0; i < nw; ++i) {
            const double denom0 = 1.0 - r21[i] * r21[i] * tau[i] * tau[i];
            const double Ta = talf[i] * tau[i] * t21[i] / denom0;
            const double Ra = ralf[i] + r21[i] * tau[i] * Ta;

            const double t_bot = t12[i] * tau[i] * t21[i] / denom0;
            const double r_bot = r12[i] + r21[i] * tau[i] * t_bot;

            double r = r_bot;
            double t = t_bot;

            double D = std::sqrt((1.0 + r + t) * (1.0 + r - t) * (1.0 - r + t) * (1.0 - r - t));
            const double rq = r * r;
            const double tq = t * t;
            double a = (1.0 + rq - tq + D) / (2.0 * r);
            double b = (1.0 - rq + tq + D) / (2.0 * t);

            const double bNm1 = std::pow(b, biochem.N - 1.0);
            const double bN2 = bNm1 * bNm1;
            const double a2 = a * a;
            double denom1 = a2 * bN2 - 1.0;
            double Rsub = a * (bN2 - 1.0) / denom1;
            double Tsub = bNm1 * (a2 - 1.0) / denom1;
            if (r + t >= 1.0) {
                // Zero-absorption limit
                Tsub = t / (t + (1.0 - t) * (biochem.N - 1.0));
                Rsub = 1.0 - Tsub;
            }

            const double denom2 = 1.0 - Rsub * r;
            tran[i] = Ta * Tsub / denom2;
            refl[i] = Ra + Ta * Rsub * t / denom2;
        }

        // ---- Remove leaf-air interfaces to get mesophyll-only rho, tau. ----
        // Rb = (refl - ralf) / (talf*t21 + (refl - ralf)*r21)
        // Z  = tran*(1 - Rb*r21) / (talf*t21)
        // rho = (Rb - r21*Z^2) / (1 - (r21*Z)^2)
        // tau_m = (1 - Rb*r21) / (1 - (r21*Z)^2) * Z
        std::vector<double> rho_m(nw), tau_m(nw), Rb_full(nw);
        for (size_t i = 0; i < nw; ++i) {
            const double Rb = (refl[i] - ralf[i]) / (talf[i] * t21[i] + (refl[i] - ralf[i]) * r21[i]);
            const double Z = tran[i] * (1.0 - Rb * r21[i]) / (talf[i] * t21[i]);
            double rho_val = (Rb - r21[i] * Z * Z) / (1.0 - (r21[i] * Z) * (r21[i] * Z));
            double tau_val = (1.0 - Rb * r21[i]) / (1.0 - (r21[i] * Z) * (r21[i] * Z)) * Z;
            if (rho_val < 0.0) rho_val = 0.0;  // Avoid negative r (MATLAB max(rho,0))
            rho_m[i] = rho_val;
            tau_m[i] = tau_val;
            r_mes[i] = rho_val;
            t_mes[i] = tau_val;
            // Rb used later with interfaces reintroduced: Rb = rho + tau^2*r21/(1-rho*r21)
            Rb_full[i] = rho_val + tau_val * tau_val * r21[i] / (1.0 - rho_val * r21[i]);
        }

        // ---- Kubelka-Munk s and k for mesophyll layer ----
        std::vector<double> s(nw), k_km(nw), kChl(nw);
        for (size_t i = 0; i < nw; ++i) {
            const double r = r_mes[i];
            const double t = t_mes[i];
            double a = 1.0;
            double b = 1.0;
            if (r + t < 1.0) {
                const double D = std::sqrt((1.0 + r + t) * (1.0 + r - t) * (1.0 - r + t) * (1.0 - r - t));
                a = (1.0 + r * r - t * t + D) / (2.0 * r);
                b = (1.0 - r * r + t * t + D) / (2.0 * t);
            }
            double s_val = r / t;
            double k_val = std::log(b);
            if (a > 1.0 && std::isfinite(a)) {
                s_val = 2.0 * a / (a * a - 1.0) * std::log(b);
                k_val = (a - 1.0) / (a + 1.0) * std::log(b);
            }
            s[i] = s_val;
            k_km[i] = k_val;
            kChl[i] = kChlrel[i] * k_val;
        }

        // ---- Prepare output kernel struct ----
        FluspectKernel out;
        out.wavelengths_nm = optipar.wavelengths_nm;
        out.refl.assign(refl.begin(), refl.end());
        out.tran.assign(tran.begin(), tran.end());

        // Skip fluorescence matrices if fqe == 0.
        if (biochem.fqe <= 0.f) {
            return out;
        }

        // ---- Excitation and emission grids ----
        // wle = 400 : excitation_step_nm : 750 (inclusive)
        const double int_step = excitation_step_nm;
        std::vector<float> wle;
        for (double w = 400.0; w <= 750.0 + 1e-9; w += int_step) {
            wle.push_back(static_cast<float>(w));
        }
        // wlf = 640:4:850 — MATLAB yields 640, 644, ..., 848 (53 entries).
        std::vector<float> wlf;
        for (double w = 640.0; w <= 850.0 - 1e-9; w += 4.0) {
            wlf.push_back(static_cast<float>(w));
        }
        const size_t n_wle = wle.size();
        const size_t n_wlf = wlf.size();

        out.wle = wle;
        out.wlf = wlf;

        // Interpolate per-wavelength quantities onto excitation grid
        auto k_iwle = interp1(optipar.wavelengths_nm, k_km, wle);
        auto s_iwle = interp1(optipar.wavelengths_nm, s, wle);
        auto kChl_iwle = interp1(optipar.wavelengths_nm, kChl, wle);
        auto r21_iwle = interp1(optipar.wavelengths_nm, r21, wle);
        auto rho_iwle = interp1(optipar.wavelengths_nm, rho_m, wle);
        auto tau_iwle = interp1(optipar.wavelengths_nm, tau_m, wle);
        auto talf_iwle = interp1(optipar.wavelengths_nm, talf, wle);
        auto Rb_iwle = interp1(optipar.wavelengths_nm, Rb_full, wle);

        // Locate wlf wavelengths in the full optipar grid (equivalent to MATLAB's
        // intersect(wlp, wlf)). We require exact matches within 0.5 nm since both
        // grids are on integer nm.
        std::vector<size_t> Iwlf(n_wlf);
        size_t guess = 0;
        for (size_t i = 0; i < n_wlf; ++i) {
            const float target = wlf[i];
            bool found = false;
            for (size_t j = guess; j < optipar.wavelengths_nm.size(); ++j) {
                if (std::abs(optipar.wavelengths_nm[j] - target) < 0.5f) {
                    Iwlf[i] = j;
                    guess = j;
                    found = true;
                    break;
                }
            }
            if (!found) {
                helios_runtime_error("ERROR (computeFluspectKernel): emission wavelength " + std::to_string(target) + " nm not present in Optipar grid.");
            }
        }

        // Values at wlf indices
        std::vector<double> k_wlf(n_wlf), s_wlf(n_wlf), phi_wlf(n_wlf), r21_wlf(n_wlf), t21_wlf(n_wlf), tau_wlf(n_wlf), rho_wlf(n_wlf), Rb_wlf(n_wlf);
        for (size_t i = 0; i < n_wlf; ++i) {
            const size_t j = Iwlf[i];
            k_wlf[i] = k_km[j];
            s_wlf[i] = s[j];
            phi_wlf[i] = optipar.phi[j];
            r21_wlf[i] = r21[j];
            t21_wlf[i] = t21[j];
            tau_wlf[i] = tau_m[j];
            rho_wlf[i] = rho_m[j];
            Rb_wlf[i] = Rb_full[j];
        }

        // ---- Doubling initialisation ----
        // ndub = 15 doublings, eps = 2^(-ndub) is the initial layer-thickness scale.
        const int ndub = 15;
        const double eps_step = std::ldexp(1.0, -ndub);  // 2^-15

        std::vector<double> te(n_wle), re(n_wle);
        for (size_t j = 0; j < n_wle; ++j) {
            te[j] = 1.0 - (k_iwle[j] + s_iwle[j]) * eps_step;
            re[j] = s_iwle[j] * eps_step;
        }
        std::vector<double> tf(n_wlf), rf(n_wlf);
        for (size_t i = 0; i < n_wlf; ++i) {
            tf[i] = 1.0 - (k_wlf[i] + s_wlf[i]) * eps_step;
            rf[i] = s_wlf[i] * eps_step;
        }

        // sigmoid[wlf_idx][wle_idx] = 1 / (1 + exp(-wlf/10) * exp(wle/10))
        // Damps emission at wavelengths shorter than the excitation (Stokes shift).
        std::vector<std::vector<double>> sigmoid(n_wlf, std::vector<double>(n_wle));
        for (size_t i = 0; i < n_wlf; ++i) {
            const double em_factor = std::exp(-static_cast<double>(wlf[i]) / 10.0);
            for (size_t j = 0; j < n_wle; ++j) {
                const double ex_factor = std::exp(static_cast<double>(wle[j]) / 10.0);
                sigmoid[i][j] = 1.0 / (1.0 + em_factor * ex_factor);
            }
        }

        // Initial Mf, Mb matrices (both start equal):
        //   M[i][j] = int_step * fqe * 0.5 * phi(wlf_i) * eps * kChl(wle_j) * sigmoid[i][j]
        std::vector<std::vector<double>> Mf(n_wlf, std::vector<double>(n_wle));
        std::vector<std::vector<double>> Mb(n_wlf, std::vector<double>(n_wle));
        const double init_scale = int_step * biochem.fqe * 0.5 * eps_step;
        for (size_t i = 0; i < n_wlf; ++i) {
            for (size_t j = 0; j < n_wle; ++j) {
                const double v = init_scale * phi_wlf[i] * kChl_iwle[j] * sigmoid[i][j];
                Mf[i][j] = v;
                Mb[i][j] = v;
            }
        }

        // ---- Doubling loop ----
        std::vector<std::vector<double>> Mfn(n_wlf, std::vector<double>(n_wle));
        std::vector<std::vector<double>> Mbn(n_wlf, std::vector<double>(n_wle));
        for (int iter = 0; iter < ndub; ++iter) {
            // Excitation-side updates (column per wle_j)
            std::vector<double> xe(n_wle), ten(n_wle), ren(n_wle);
            for (size_t j = 0; j < n_wle; ++j) {
                xe[j] = te[j] / (1.0 - re[j] * re[j]);
                ten[j] = te[j] * xe[j];
                ren[j] = re[j] * (1.0 + ten[j]);
            }
            // Emission-side updates (row per wlf_i)
            std::vector<double> xf(n_wlf), tfn(n_wlf), rfn(n_wlf);
            for (size_t i = 0; i < n_wlf; ++i) {
                xf[i] = tf[i] / (1.0 - rf[i] * rf[i]);
                tfn[i] = tf[i] * xf[i];
                rfn[i] = rf[i] * (1.0 + tfn[i]);
            }

            // Apply: Mfn = Mf*A11 + Mb*A12; Mbn = Mb*A21 + Mf*A22 (element-wise)
            // A11 = xf[i] + xe[j]
            // A12 = xf[i]*xe[j] * (rf[i] + re[j])
            // A21 = 1 + xf[i]*xe[j] * (1 + rf[i]*re[j])
            // A22 = xf[i]*rf[i] + xe[j]*re[j]
            for (size_t i = 0; i < n_wlf; ++i) {
                for (size_t j = 0; j < n_wle; ++j) {
                    const double A11 = xf[i] + xe[j];
                    const double A12 = xf[i] * xe[j] * (rf[i] + re[j]);
                    const double A21 = 1.0 + xf[i] * xe[j] * (1.0 + rf[i] * re[j]);
                    const double A22 = xf[i] * rf[i] + xe[j] * re[j];
                    Mfn[i][j] = Mf[i][j] * A11 + Mb[i][j] * A12;
                    Mbn[i][j] = Mb[i][j] * A21 + Mf[i][j] * A22;
                }
            }

            te = ten;
            re = ren;
            tf = tfn;
            rf = rfn;
            Mf.swap(Mfn);
            Mb.swap(Mbn);
        }

        // ---- Re-add leaf-air interfaces to Mf/Mb ----
        // Rb_iwle = Rb_full interpolated to wle grid
        // Xe[i][j] = talf_iwle[j] / (1 - r21_iwle[j] * Rb_iwle[j])
        // Xf[i][j] = t21_wlf[i] / (1 - r21_wlf[i] * Rb_wlf[i])
        // Ye[i][j] = tau_iwle[j] * r21_iwle[j] / (1 - rho_iwle[j] * r21_iwle[j])
        // Yf[i][j] = tau_wlf[i] * r21_wlf[i] / (1 - rho_wlf[i] * r21_wlf[i])
        // A = Xe * (1 + Ye*Yf) * Xf
        // B = Xe * (Ye + Yf) * Xf
        // Mb_out = A*Mb_init + B*Mf_init
        // Mf_out = A*Mf_init + B*Mb_init
        std::vector<double> xe_vec(n_wle), ye_vec(n_wle);
        for (size_t j = 0; j < n_wle; ++j) {
            xe_vec[j] = talf_iwle[j] / (1.0 - r21_iwle[j] * Rb_iwle[j]);
            ye_vec[j] = tau_iwle[j] * r21_iwle[j] / (1.0 - rho_iwle[j] * r21_iwle[j]);
        }
        std::vector<double> xf_vec(n_wlf), yf_vec(n_wlf);
        for (size_t i = 0; i < n_wlf; ++i) {
            xf_vec[i] = t21_wlf[i] / (1.0 - r21_wlf[i] * Rb_wlf[i]);
            yf_vec[i] = tau_wlf[i] * r21_wlf[i] / (1.0 - rho_wlf[i] * r21_wlf[i]);
        }

        out.Mf.assign(n_wlf, std::vector<float>(n_wle));
        out.Mb.assign(n_wlf, std::vector<float>(n_wle));
        for (size_t i = 0; i < n_wlf; ++i) {
            for (size_t j = 0; j < n_wle; ++j) {
                const double A = xe_vec[j] * (1.0 + ye_vec[j] * yf_vec[i]) * xf_vec[i];
                const double B = xe_vec[j] * (ye_vec[j] + yf_vec[i]) * xf_vec[i];
                const double g = Mb[i][j];  // post-doubling Mb
                const double f = Mf[i][j];  // post-doubling Mf
                const double gn = A * g + B * f;
                const double fn = A * f + B * g;
                out.Mb[i][j] = static_cast<float>(gn);
                out.Mf[i][j] = static_cast<float>(fn);
            }
        }

        return out;
    }

} // namespace helios
