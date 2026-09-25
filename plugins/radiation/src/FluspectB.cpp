/** \file "FluspectB.cpp" Leaf optics and chlorophyll-fluorescence kernels following Fluspect-B/CX (Vilfan et al. 2016, 2018).

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

        // --------------------------------------------------------------------
        // Leaf radiative transfer building blocks
        // --------------------------------------------------------------------
        //
        // References:
        //   [JB90]  Jacquemoud & Baret (1990) PROSPECT: A model of leaf optical properties spectra.
        //           Remote Sensing of Environment 34:75-91.
        //   [V16]   Vilfan et al. (2016) Fluspect-B: A model for leaf fluorescence, reflectance and
        //           transmittance spectra. Remote Sensing of Environment 186:596-615.
        //   [V18]   Vilfan et al. (2018) Extending Fluspect to simulate xanthophyll driven leaf reflectance
        //           dynamics. Remote Sensing of Environment 211:345-356.
        //   [DLMF]  NIST Digital Library of Mathematical Functions, chapter 6.

        //! Half-angle (degrees) of the cone of incident directions on the leaf surface ([V16] Sec. 2.1).
        constexpr double incidence_cone_half_angle_deg = 59.0;
        //! Number of doubling steps; the initial sub-layer holds a fraction 2^-15 of the optical thickness ([V16] Sec. 2.4).
        constexpr int number_of_doublings = 15;
        //! Width (nm) of the sigmoid suppressing anti-Stokes emission ([V16] Eq. 28).
        constexpr double anti_stokes_sigmoid_width_nm = 10.0;
        //! Emission grid: 640-848 nm at 4 nm spacing (documented contract in FluspectB.h).
        constexpr double emission_min_nm = 640.0;
        constexpr double emission_step_nm = 4.0;
        constexpr int emission_count = 53;
        //! Excitation grid bounds (nm).
        constexpr double excitation_min_nm = 400.0;
        constexpr double excitation_max_nm = 750.0;

        //! Exponential integral E1(x), x > 0. Returns E1(x) for x <= 1 (power series, [DLMF] 6.6.2), and the scaled value
        //! exp(x) E1(x) for x > 1 (continued fraction [DLMF] 6.9.1, even contraction, modified Lentz evaluation).
        double exponentialIntegralE1(double x, bool &scaled_by_exp) {
            constexpr double euler_gamma = 0.57721566490153286060651209;
            if (x <= 1.0) {
                scaled_by_exp = false;
                double series_sum = 0.0;
                double power_over_factorial = 1.0; // (-x)^k / k!
                for (int k = 1; k < 100; ++k) {
                    power_over_factorial *= -x / static_cast<double>(k);
                    const double term = power_over_factorial / static_cast<double>(k);
                    series_sum += term;
                    if (std::abs(term) < 1e-17 * std::abs(series_sum)) {
                        break;
                    }
                }
                return -euler_gamma - std::log(x) - series_sum;
            }
            // exp(x) E1(x) = 1 / (x + 1 - 1^2 / (x + 3 - 2^2 / (x + 5 - ...)))
            scaled_by_exp = true;
            constexpr double tiny = 1e-300;
            double denominator_term = x + 1.0;
            double lentz_c = 1.0 / tiny;
            double lentz_d = 1.0 / denominator_term;
            double fraction = lentz_d;
            for (int i = 1; i < 10000; ++i) {
                const double numerator_term = -static_cast<double>(i) * static_cast<double>(i);
                denominator_term += 2.0;
                lentz_d = 1.0 / (numerator_term * lentz_d + denominator_term);
                lentz_c = denominator_term + numerator_term / lentz_c;
                const double delta = lentz_c * lentz_d;
                fraction *= delta;
                if (std::abs(delta - 1.0) < 1e-16) {
                    return fraction;
                }
            }
            helios_runtime_error("ERROR (computeFluspectKernel): continued fraction for the exponential integral E1(" + std::to_string(x) + ") failed to converge.");
            return 0.0;
        }

        //! Diffuse transmission of a compact elementary layer, theta(k) = (1 - k) e^-k + k^2 E1(k) ([JB90] Eq. 11).
        double elementaryLayerTransmission(double k) {
            if (k <= 0.0) {
                return 1.0;
            }
            bool scaled_by_exp = false;
            const double e1_value = exponentialIntegralE1(k, scaled_by_exp);
            if (!scaled_by_exp) {
                return (1.0 - k) * std::exp(-k) + k * k * e1_value;
            }
            return std::exp(-k) * ((1.0 - k) + k * k * e1_value);
        }

        struct GaussLegendreRule {
            std::vector<double> nodes;
            std::vector<double> weights;
        };

        //! Gauss-Legendre nodes/weights on [-1, 1] by Newton iteration on the Legendre three-term recurrence.
        GaussLegendreRule makeGaussLegendreRule(int order) {
            GaussLegendreRule rule;
            rule.nodes.resize(order);
            rule.weights.resize(order);
            for (int i = 0; i < order; ++i) {
                double x = std::cos(M_PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(order) + 0.5));
                double derivative = 1.0;
                for (int iteration = 0; iteration < 100; ++iteration) {
                    double p_current = 1.0;
                    double p_previous = 0.0;
                    for (int degree = 1; degree <= order; ++degree) {
                        const double p_older = p_previous;
                        p_previous = p_current;
                        p_current = ((2.0 * degree - 1.0) * x * p_previous - (degree - 1.0) * p_older) / static_cast<double>(degree);
                    }
                    derivative = static_cast<double>(order) * (x * p_current - p_previous) / (x * x - 1.0);
                    const double step = p_current / derivative;
                    x -= step;
                    if (std::abs(step) < 1e-15) {
                        break;
                    }
                }
                rule.nodes[i] = x;
                rule.weights[i] = 2.0 / ((1.0 - x * x) * derivative * derivative);
            }
            return rule;
        }

        //! Fresnel transmissivity for unpolarized light entering a medium of refractive index n from air at angle theta.
        double fresnelTransmissivity(double incidence_angle, double refractive_index) {
            const double cos_incidence = std::cos(incidence_angle);
            const double sin_incidence = std::sin(incidence_angle);
            const double n_squared = refractive_index * refractive_index;
            const double n_cos_transmitted = std::sqrt(n_squared - sin_incidence * sin_incidence);
            const double r_perpendicular = (cos_incidence - n_cos_transmitted) / (cos_incidence + n_cos_transmitted);
            const double r_parallel = (n_squared * cos_incidence - n_cos_transmitted) / (n_squared * cos_incidence + n_cos_transmitted);
            return 1.0 - 0.5 * (r_perpendicular * r_perpendicular + r_parallel * r_parallel);
        }

        //! Interface transmissivity t_av(alpha, n) averaged over an isotropic cone of half-angle alpha ([JB90] Eqs. 1-2):
        //! t_av = 2 / sin^2(alpha) * integral_0^alpha t(theta) sin(theta) cos(theta) dtheta (32-point Gauss-Legendre).
        double averageInterfaceTransmissivity(double alpha_degrees, double refractive_index) {
            static const GaussLegendreRule rule = makeGaussLegendreRule(32);
            if (refractive_index < 1.0) {
                helios_runtime_error("ERROR (computeFluspectKernel): refractive index " + std::to_string(refractive_index) + " is below 1; check the Optipar 'nr' coefficients.");
            }
            const double alpha = alpha_degrees * M_PI / 180.0;
            const double half_alpha = 0.5 * alpha;
            double integral = 0.0;
            for (size_t i = 0; i < rule.nodes.size(); ++i) {
                const double theta = half_alpha * (rule.nodes[i] + 1.0);
                integral += rule.weights[i] * half_alpha * fresnelTransmissivity(theta, refractive_index) * std::sin(theta) * std::cos(theta);
            }
            const double sin_alpha = std::sin(alpha);
            return 2.0 * integral / (sin_alpha * sin_alpha);
        }

        //! Stokes solution for a stack of m identical diffusing layers ([JB90] Eq. 9), m >= 0 and possibly non-integer.
        void stokesLayerStack(double layer_reflectance, double layer_transmittance, double number_of_layers, double &stack_reflectance, double &stack_transmittance) {
            if (number_of_layers <= 0.0) {
                stack_reflectance = 0.0;
                stack_transmittance = 1.0;
                return;
            }
            const double layer_absorptance = 1.0 - layer_reflectance - layer_transmittance;
            if (layer_absorptance <= 1e-10) {
                // Non-absorbing limit (a, b -> 1): T_m = tau / (tau + m rho), R_m = 1 - T_m.
                stack_transmittance = layer_transmittance / (layer_transmittance + number_of_layers * layer_reflectance);
                stack_reflectance = 1.0 - stack_transmittance;
                return;
            }
            const double rho2 = layer_reflectance * layer_reflectance;
            const double tau2 = layer_transmittance * layer_transmittance;
            const double delta = std::sqrt((1.0 + layer_reflectance + layer_transmittance) * (1.0 + layer_reflectance - layer_transmittance) * (1.0 - layer_reflectance + layer_transmittance) * layer_absorptance);
            const double a = (1.0 + rho2 - tau2 + delta) / (2.0 * layer_reflectance);
            const double b = (1.0 - rho2 + tau2 + delta) / (2.0 * layer_transmittance);
            // Numerator and denominator divided by b^m so that b >> 1 cannot overflow.
            const double b_inverse_power = std::exp(-number_of_layers * std::log(b));
            const double b_inverse_power2 = b_inverse_power * b_inverse_power;
            const double denominator = a - b_inverse_power2 / a;
            stack_reflectance = (1.0 - b_inverse_power2) / denominator;
            stack_transmittance = (a - 1.0 / a) * b_inverse_power / denominator;
        }

        //! Optipar coefficients evaluated at one wavelength.
        struct SpectralCoefficients {
            double refractive_index = 0;
            double total_absorption = 0; //!< sum_i K_i C_i
            double chlorophyll_absorption = 0; //!< K_ab C_ab
            double emission_spectrum = 0; //!< Phi (nm^-1)
        };

        //! Combine Optipar coefficients at grid index i with the leaf biochemistry ([V18] Eqs. 1-2).
        SpectralCoefficients coefficientsAtIndex(const FluspectBiochemistry &biochem, const FluspectOptipar &optipar, size_t i) {
            // Carotenoid absorption is a linear mixture of the violaxanthin- and zeaxanthin-state spectra ([V18] Eq. 2).
            const double carotenoid_sac = (1.0 - biochem.V2Z) * optipar.KcaV[i] + biochem.V2Z * static_cast<double>(optipar.KcaZ[i]);
            SpectralCoefficients c;
            c.refractive_index = optipar.nr[i];
            c.chlorophyll_absorption = static_cast<double>(biochem.Cab) * optipar.Kab[i];
            c.total_absorption = c.chlorophyll_absorption + static_cast<double>(biochem.Cca) * carotenoid_sac + static_cast<double>(biochem.Cw) * optipar.Kw[i] + static_cast<double>(biochem.Cdm) * optipar.Kdm[i] +
                                 static_cast<double>(biochem.Cs) * optipar.Ks[i] + static_cast<double>(biochem.Cant) * optipar.Kant[i] + static_cast<double>(biochem.Cp) * optipar.Kp[i] + static_cast<double>(biochem.Cbc) * optipar.Kcbc[i];
            c.emission_spectrum = optipar.phi[i];
            return c;
        }

        //! Linearly interpolate the combined coefficients to an arbitrary wavelength inside the Optipar grid.
        SpectralCoefficients coefficientsAtWavelength(const FluspectBiochemistry &biochem, const FluspectOptipar &optipar, double wavelength_nm) {
            const std::vector<float> &grid = optipar.wavelengths_nm;
            if (wavelength_nm < grid.front() - 1e-4 || wavelength_nm > grid.back() + 1e-4) {
                helios_runtime_error("ERROR (computeFluspectKernel): wavelength " + std::to_string(wavelength_nm) + " nm lies outside the Optipar grid [" + std::to_string(grid.front()) + ", " + std::to_string(grid.back()) + "] nm.");
            }
            const auto upper = std::lower_bound(grid.begin(), grid.end(), static_cast<float>(wavelength_nm));
            size_t upper_index = static_cast<size_t>(upper - grid.begin());
            if (upper_index < grid.size() && std::abs(grid[upper_index] - wavelength_nm) < 1e-4) {
                return coefficientsAtIndex(biochem, optipar, upper_index);
            }
            if (upper_index > 0 && std::abs(grid[upper_index - 1] - wavelength_nm) < 1e-4) {
                return coefficientsAtIndex(biochem, optipar, upper_index - 1);
            }
            upper_index = std::min(std::max<size_t>(upper_index, 1), grid.size() - 1);
            const SpectralCoefficients lower_c = coefficientsAtIndex(biochem, optipar, upper_index - 1);
            const SpectralCoefficients upper_c = coefficientsAtIndex(biochem, optipar, upper_index);
            const double w = (wavelength_nm - grid[upper_index - 1]) / (grid[upper_index] - grid[upper_index - 1]);
            SpectralCoefficients c;
            c.refractive_index = (1.0 - w) * lower_c.refractive_index + w * upper_c.refractive_index;
            c.total_absorption = (1.0 - w) * lower_c.total_absorption + w * upper_c.total_absorption;
            c.chlorophyll_absorption = (1.0 - w) * lower_c.chlorophyll_absorption + w * upper_c.chlorophyll_absorption;
            c.emission_spectrum = (1.0 - w) * lower_c.emission_spectrum + w * upper_c.emission_spectrum;
            return c;
        }

        //! Whole-leaf and mesophyll-layer optics at one wavelength.
        struct LeafLayerOptics {
            double leaf_reflectance = 0; //!< R_t, whole leaf, cone incidence
            double leaf_transmittance = 0; //!< T_t
            double t_alpha = 0; //!< interface transmissivity for external cone incidence
            double t21 = 0; //!< interface transmissivity leaf -> air (diffuse)
            double r21 = 0; //!< interface reflectivity leaf -> air (diffuse)
            double background_reflectance = 0; //!< R_b: mesophyll + bottom interface seen from inside the top interface
            double rho = 0; //!< mesophyll reflectance (interfaces removed)
            double tau = 0; //!< mesophyll transmittance
            double km_absorption = 0; //!< Kubelka-Munk absorption coefficient k of the mesophyll
            double km_scattering = 0; //!< Kubelka-Munk backscattering coefficient s
            double chlorophyll_fraction = 0; //!< share of total absorption due to chlorophyll
        };

        //! PROSPECT whole-leaf optics ([JB90] Eqs. 1-2, 7-9, 11). The elementary-layer absorption is k = sum_i K_i C_i / N.
        void computeLeafOptics(const SpectralCoefficients &c, double structure_N, LeafLayerOptics &optics) {
            const double layer_absorption = c.total_absorption / structure_N;
            const double theta = elementaryLayerTransmission(layer_absorption);
            const double n2 = c.refractive_index * c.refractive_index;
            const double t_alpha = averageInterfaceTransmissivity(incidence_cone_half_angle_deg, c.refractive_index);
            const double t_90 = averageInterfaceTransmissivity(90.0, c.refractive_index);

            const double theta2 = theta * theta;
            const double plate_denominator = n2 * n2 - theta2 * (n2 - t_90) * (n2 - t_90);
            const double rho_alpha = (1.0 - t_alpha) + t_90 * t_alpha * theta2 * (n2 - t_90) / plate_denominator;
            const double tau_alpha = t_90 * t_alpha * theta * n2 / plate_denominator;
            const double rho_90 = (1.0 - t_90) + t_90 * t_90 * theta2 * (n2 - t_90) / plate_denominator;
            const double tau_90 = t_90 * t_90 * theta * n2 / plate_denominator;

            double stack_reflectance = 0.0;
            double stack_transmittance = 1.0;
            stokesLayerStack(rho_90, tau_90, structure_N - 1.0, stack_reflectance, stack_transmittance);
            const double interreflection = 1.0 - rho_90 * stack_reflectance;
            optics.leaf_reflectance = rho_alpha + tau_alpha * tau_90 * stack_reflectance / interreflection;
            optics.leaf_transmittance = tau_alpha * stack_transmittance / interreflection;

            optics.t_alpha = t_alpha;
            // Diffuse light leaving the leaf interior: t21 = t_av(90, n) / n^2 (reciprocity; implied by [JB90] Eqs. 1-2).
            optics.t21 = t_90 / n2;
            optics.r21 = 1.0 - optics.t21;
            optics.chlorophyll_fraction = c.total_absorption > 0.0 ? c.chlorophyll_absorption / c.total_absorption : 0.0;
        }

        //! Remove the leaf-air interfaces to obtain mesophyll rho/tau ([V16] Eqs. 2, 6, 10) and then the Kubelka-Munk
        //! coefficients k and s ([V16] Eqs. 24-26 / Appendix B). Needed only at excitation and emission wavelengths.
        void computeMesophyllOptics(double wavelength_nm, LeafLayerOptics &optics) {
            const double r_alpha = 1.0 - optics.t_alpha;
            const double reflectance_excess = optics.leaf_reflectance - r_alpha;
            optics.background_reflectance = reflectance_excess / (optics.t_alpha * optics.t21 + reflectance_excess * optics.r21); // Eq. 2
            const double Z = optics.leaf_transmittance * (1.0 - optics.background_reflectance * optics.r21) / (optics.t_alpha * optics.t21); // Eq. 6
            const double r21Z = optics.r21 * Z;
            optics.tau = (1.0 - optics.background_reflectance * optics.r21) / (1.0 - r21Z * r21Z) * Z; // Eq. 10
            optics.rho = (optics.background_reflectance - optics.r21 * Z * Z) / (1.0 - r21Z * r21Z); // Eq. 10

            const double rho = optics.rho;
            const double tau = optics.tau;
            if (!std::isfinite(rho) || !std::isfinite(tau) || tau <= 0.0 || rho < 0.0) {
                helios_runtime_error("ERROR (computeFluspectKernel): mesophyll reflectance/transmittance at " + std::to_string(wavelength_nm) + " nm are rho=" + std::to_string(rho) + ", tau=" + std::to_string(tau) +
                                     "; the leaf is too strongly absorbing to derive Kubelka-Munk coefficients. Check the leaf biochemistry inputs (pigment contents, N).");
            }
            const double absorptance = 1.0 - rho - tau;
            if (rho == 0.0) {
                // Purely absorbing layer: s = 0 and tau = exp(-k).
                optics.km_scattering = 0.0;
                optics.km_absorption = -std::log(tau);
            } else if (absorptance <= 1e-12) {
                // Conservative scattering: k = 0, rho = s / (1 + s), tau = 1 / (1 + s).
                optics.km_absorption = 0.0;
                optics.km_scattering = rho / tau;
            } else {
                const double D = std::sqrt((1.0 + rho + tau) * (1.0 + rho - tau) * (1.0 - rho + tau) * absorptance); // Eq. 26
                const double a = (1.0 + rho * rho - tau * tau + D) / (2.0 * rho); // Eq. 25
                const double b = (1.0 - rho * rho + tau * tau + D) / (2.0 * tau);
                const double log_b = std::log(b);
                optics.km_scattering = 2.0 * a / (a * a - 1.0) * log_b; // Eq. 24
                optics.km_absorption = (a - 1.0) / (a + 1.0) * log_b;
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
        // ---- Input validation --------------------------------------------------------------------------------------
        if (!(excitation_step_nm > 0.f) || !std::isfinite(excitation_step_nm)) {
            helios_runtime_error("ERROR (computeFluspectKernel): excitation_step_nm must be a finite value > 0, but " + std::to_string(excitation_step_nm) + " was given.");
        }
        if (!(biochem.N >= 1.f)) {
            helios_runtime_error("ERROR (computeFluspectKernel): leaf structure parameter N was " + std::to_string(biochem.N) + ", but must be >= 1 (it counts elementary layers).");
        }
        const float contents[] = {biochem.Cab, biochem.Cca, biochem.Cw, biochem.Cdm, biochem.Cs, biochem.Cant, biochem.Cp, biochem.Cbc, biochem.V2Z};
        const char *content_names[] = {"Cab", "Cca", "Cw", "Cdm", "Cs", "Cant", "Cp", "Cbc", "V2Z"};
        for (size_t i = 0; i < 9; ++i) {
            if (!(contents[i] >= 0.f) || !std::isfinite(contents[i])) {
                helios_runtime_error("ERROR (computeFluspectKernel): leaf biochemistry input " + std::string(content_names[i]) + " was " + std::to_string(contents[i]) + ", but must be finite and non-negative.");
            }
        }
        const size_t grid_size = optipar.wavelengths_nm.size();
        const std::vector<const std::vector<float> *> coefficient_arrays = {&optipar.nr, &optipar.Kab, &optipar.KcaV, &optipar.KcaZ, &optipar.Kdm, &optipar.Kw, &optipar.Ks, &optipar.Kant, &optipar.Kp, &optipar.Kcbc, &optipar.phi};
        for (const auto *array: coefficient_arrays) {
            if (array->size() != grid_size) {
                helios_runtime_error("ERROR (computeFluspectKernel): Optipar coefficient arrays do not all match the wavelength grid size (" + std::to_string(grid_size) + "). Load them with loadFluspectOptipar().");
            }
        }
        const double emission_max_nm = emission_min_nm + emission_step_nm * (emission_count - 1);
        if (grid_size < 2 || optipar.wavelengths_nm.front() > excitation_min_nm || optipar.wavelengths_nm.back() < emission_max_nm) {
            helios_runtime_error("ERROR (computeFluspectKernel): the Optipar wavelength grid must span at least " + std::to_string(excitation_min_nm) + "-" + std::to_string(emission_max_nm) + " nm.");
        }

        FluspectKernel kernel;
        const double structure_N = biochem.N;

        // ---- (a) Leaf reflectance and transmittance on the full Optipar grid ([JB90]) ---------------------------------
        kernel.wavelengths_nm = optipar.wavelengths_nm;
        kernel.refl.resize(grid_size);
        kernel.tran.resize(grid_size);
        for (size_t i = 0; i < grid_size; ++i) {
            LeafLayerOptics optics;
            computeLeafOptics(coefficientsAtIndex(biochem, optipar, i), structure_N, optics);
            kernel.refl[i] = static_cast<float>(optics.leaf_reflectance);
            kernel.tran[i] = static_cast<float>(optics.leaf_transmittance);
        }

        // ---- Excitation and emission grids ------------------------------------------------------------------------
        const int excitation_count = static_cast<int>(std::floor((excitation_max_nm - excitation_min_nm) / excitation_step_nm + 1e-6)) + 1;
        kernel.wle.resize(excitation_count);
        for (int j = 0; j < excitation_count; ++j) {
            kernel.wle[j] = static_cast<float>(excitation_min_nm + j * static_cast<double>(excitation_step_nm));
        }
        kernel.wlf.resize(emission_count);
        for (int i = 0; i < emission_count; ++i) {
            kernel.wlf[i] = static_cast<float>(emission_min_nm + i * emission_step_nm);
        }

        if (biochem.fqe <= 0.f) {
            return kernel; // no fluorescence: Mf/Mb intentionally left empty (see header contract)
        }

        // ---- Mesophyll optics and Kubelka-Munk coefficients at excitation (e) and emission (f) wavelengths -----------
        std::vector<LeafLayerOptics> excitation_optics(excitation_count);
        std::vector<LeafLayerOptics> emission_optics(emission_count);
        std::vector<double> emission_spectrum(emission_count);
        for (int j = 0; j < excitation_count; ++j) {
            computeLeafOptics(coefficientsAtWavelength(biochem, optipar, kernel.wle[j]), structure_N, excitation_optics[j]);
            computeMesophyllOptics(kernel.wle[j], excitation_optics[j]);
        }
        for (int i = 0; i < emission_count; ++i) {
            const SpectralCoefficients c = coefficientsAtWavelength(biochem, optipar, kernel.wlf[i]);
            emission_spectrum[i] = c.emission_spectrum;
            computeLeafOptics(c, structure_N, emission_optics[i]);
            computeMesophyllOptics(kernel.wlf[i], emission_optics[i]);
        }

        // ---- Doubling algorithm with fluorescence ([V16] Sec. 2.4, Eqs. 27-36; Appendix C) -------------------------
        const double epsilon = std::ldexp(1.0, -number_of_doublings);
        std::vector<double> r_e(excitation_count), t_e(excitation_count), r_f(emission_count), t_f(emission_count);
        for (int j = 0; j < excitation_count; ++j) {
            r_e[j] = excitation_optics[j].km_scattering * epsilon; // Eq. 33
            t_e[j] = 1.0 - (excitation_optics[j].km_absorption + excitation_optics[j].km_scattering) * epsilon;
        }
        for (int i = 0; i < emission_count; ++i) {
            r_f[i] = emission_optics[i].km_scattering * epsilon;
            t_f[i] = 1.0 - (emission_optics[i].km_absorption + emission_optics[i].km_scattering) * epsilon;
        }
        for (int j = 0; j < excitation_count; ++j) {
            if (t_e[j] <= 0.0) {
                helios_runtime_error("ERROR (computeFluspectKernel): the leaf mesophyll at " + std::to_string(kernel.wle[j]) + " nm is too optically thick for the doubling algorithm. Check the leaf biochemistry inputs.");
            }
        }

        // Fluorescence source per unit optical thickness ([V16] Eq. 27):
        //   phi(e, f) = 0.5 * eta * Phi(lambda_f) * k_Chl(lambda_e) * sigma(lambda_e, lambda_f),
        // where k_Chl is the chlorophyll share of the Kubelka-Munk absorption coefficient and sigma is the anti-Stokes
        // sigmoid of Eq. 28. The source is additionally multiplied by the excitation grid spacing, so that each kernel
        // column acts as the quadrature weight of its excitation bin (this convention is not stated in [V16]; it is the
        // one that reproduces the Fluspect reference kernels shipped in plugins/radiation/tests/reference/).
        std::vector<std::vector<double>> forward(emission_count, std::vector<double>(excitation_count));
        std::vector<std::vector<double>> backward(emission_count, std::vector<double>(excitation_count));
        for (int i = 0; i < emission_count; ++i) {
            for (int j = 0; j < excitation_count; ++j) {
                const double chlorophyll_absorption = excitation_optics[j].km_absorption * excitation_optics[j].chlorophyll_fraction;
                const double anti_stokes = 1.0 / (1.0 + std::exp((static_cast<double>(kernel.wle[j]) - kernel.wlf[i]) / anti_stokes_sigmoid_width_nm));
                const double source = 0.5 * biochem.fqe * emission_spectrum[i] * chlorophyll_absorption * anti_stokes * excitation_step_nm;
                forward[i][j] = source * epsilon; // Eq. 33: f = g = phi * epsilon
                backward[i][j] = source * epsilon;
            }
        }

        std::vector<double> x_e(excitation_count), x_f(emission_count);
        for (int step = 0; step < number_of_doublings; ++step) {
            for (int j = 0; j < excitation_count; ++j) {
                x_e[j] = t_e[j] / (1.0 - r_e[j] * r_e[j]); // Eq. 34
            }
            for (int i = 0; i < emission_count; ++i) {
                x_f[i] = t_f[i] / (1.0 - r_f[i] * r_f[i]);
            }
            for (int i = 0; i < emission_count; ++i) {
                for (int j = 0; j < excitation_count; ++j) {
                    const double f_old = forward[i][j];
                    const double g_old = backward[i][j];
                    const double xx = x_e[j] * x_f[i];
                    forward[i][j] = f_old * (x_e[j] + x_f[i]) + g_old * xx * (r_e[j] + r_f[i]); // Eq. 35
                    backward[i][j] = g_old * (1.0 + xx * (1.0 + r_e[j] * r_f[i])) + f_old * (x_e[j] * r_e[j] + x_f[i] * r_f[i]); // Eq. 36
                }
            }
            for (int j = 0; j < excitation_count; ++j) {
                const double t_new = t_e[j] * x_e[j];
                r_e[j] = r_e[j] * (1.0 + t_new);
                t_e[j] = t_new;
            }
            for (int i = 0; i < emission_count; ++i) {
                const double t_new = t_f[i] * x_f[i];
                r_f[i] = r_f[i] * (1.0 + t_new);
                t_f[i] = t_new;
            }
        }

        // ---- Add the leaf-air interfaces ([V16] Eqs. 13-17; Appendix A) ----------------------------------------------
        // X_e uses t_alpha because the excitation enters from outside the leaf through the top interface (Eq. A.11);
        // X_f uses t21 because the fluorescence leaves through the interface from the inside (Eq. A.40).
        std::vector<double> X_e(excitation_count), Y_e(excitation_count), X_f(emission_count), Y_f(emission_count);
        for (int j = 0; j < excitation_count; ++j) {
            const LeafLayerOptics &o = excitation_optics[j];
            X_e[j] = o.t_alpha / (1.0 - o.r21 * o.background_reflectance);
            Y_e[j] = o.tau * o.r21 / (1.0 - o.rho * o.r21);
        }
        for (int i = 0; i < emission_count; ++i) {
            const LeafLayerOptics &o = emission_optics[i];
            X_f[i] = o.t21 / (1.0 - o.r21 * o.background_reflectance);
            Y_f[i] = o.tau * o.r21 / (1.0 - o.rho * o.r21);
        }

        kernel.Mf.assign(emission_count, std::vector<float>(excitation_count));
        kernel.Mb.assign(emission_count, std::vector<float>(excitation_count));
        for (int i = 0; i < emission_count; ++i) {
            for (int j = 0; j < excitation_count; ++j) {
                const double cross = Y_e[j] + Y_f[i];
                const double direct = 1.0 + Y_e[j] * Y_f[i];
                const double f = forward[i][j];
                const double g = backward[i][j];
                kernel.Mf[i][j] = static_cast<float>(X_e[j] * (cross * g + direct * f) * X_f[i]); // Eq. 15
                kernel.Mb[i][j] = static_cast<float>(X_e[j] * (direct * g + cross * f) * X_f[i]); // Eq. 16
            }
        }
        return kernel;
    }

} // namespace helios
