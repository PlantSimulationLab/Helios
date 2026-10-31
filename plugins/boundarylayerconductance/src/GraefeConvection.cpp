/** \file "GraefeConvection.cpp" Leaf boundary-layer conductance model of Graefe et al. (2022), with the mixed-convection extensions.

    Copyright (C) 2026 Jan Graefe (original Python/MATLAB implementation)
    Copyright (C) 2026 Brian Bailey (C++ port)

    C++ port of the Python/MATLAB package "leaf-boundary-layer" v1.1.0 by Jan Graefe, Leibniz Institute of Vegetable and Ornamental Crops (IGZ),
    https://github.com/Code4Plants/Leaf-Boundary-Layer, distributed under LGPL-2.1-or-later. The free-convection model is that of Graefe, Grosch and Bitterlich (2022);
    the forced and mixed convection, laminar-turbulent transition, angle-of-attack and scalar-conductance extensions are those of the package.
    Please cite both:
      Graefe, J., Grosch, R., Bitterlich, M. (2022). The boundary layer conductance of inclined elliptical leaves under free convection. Agricultural and
      Forest Meteorology 317, 108884. https://doi.org/10.1016/j.agrformet.2022.108884
      Graefe, J. (2026). leaf-conductance: boundary-layer conductance of leaves under free, forced and mixed convection (Python and MATLAB), version 1.1.0.

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

#include "GraefeConvection.h"

#include <algorithm>
#include <cmath>

namespace graefe {

    namespace {

        // Table 3 of Graefe et al. (2022)
        constexpr double P1 = 6.0;
        constexpr double P2 = 1.42;
        constexpr double P3 = 0.513;
        constexpr double P4 = 0.4;
        constexpr double P5 = 0.202;
        constexpr double P6 = 0.492;

        constexpr double PRANDTL = 0.71; // Prandtl number of air (Table 1)
        constexpr double GRAVITY = 9.81; // m/s^2
        constexpr double CP_AIR = 1004.0; // specific heat of air (J/kg/K)
        constexpr double R_DRY_AIR = 287.05; // gas constant of dry air (J/kg/K)
        constexpr double R_UNIVERSAL = 8.314462618; // universal gas constant (J/mol/K)
        constexpr double N_INCLINED = 0.75; // exponent n of Eq. 4 / Eq. 9 for free convection (3 times the Rayleigh exponent 0.25)
        constexpr double N_FORCED = 0.5; // exponent n of Eq. 4 for laminar forced convection (Nu ~ Re^0.5)
        constexpr double C_FORCED_LAMINAR = 0.664; // laminar flat plate (Pohlhausen)

        // laminar-turbulent transition of the free convection
        constexpr double C_FREE_TURBULENT = 0.14; // NuF = 0.14 Ra^(1/3)
        constexpr double M_FREE_TURBULENT = 6.0; // blending exponent of the laminar and turbulent free convection

        // laminar-turbulent transition of the forced convection (Lienhard 2020)
        constexpr double A_LAMINAR_LOCAL = 0.332; // local laminar Nu_x = 0.332 Re_x^0.5 Pr^(1/3)
        constexpr double C_TURBULENT_LOCAL = 0.0296; // local turbulent Nu_x = 0.0296 Re_x^0.8 Pr^0.6
        constexpr double RE_ONSET_SMOOTH = 5.0e5; // onset for a disturbance-free stream
        constexpr double RE_ONSET_MIN = 2.0e3; // lower bound of the onset at very high turbulence

        // angle-of-attack model fitted to 815 thin-plate/leaf measurements
        constexpr double F_EDGE = 1.161; // finite-plate enhancement
        constexpr double A_WINDWARD = 0.226;
        constexpr double A_REATTACHMENT = 0.574;
        constexpr double BETA_SEPARATION_DEG = 31.2;
        constexpr double C_WAKE = 0.833;

        // diffusivity exponents of the transport modes, g_s = g_heat (D_s/kappa)^k
        constexpr double K_CONDUCTION = 1.0;
        constexpr double K_FREE_LAMINAR = 0.75;
        constexpr double K_FREE_TURBULENT = 2.0 / 3.0;
        constexpr double K_FORCED_LAMINAR = 2.0 / 3.0;
        constexpr double K_FORCED_TURBULENT = 0.4;
        constexpr double D_H2O_0 = 2.12e-5; // m^2/s at 273.15 K and 101325 Pa (Massman 1998)
        constexpr double D_CO2_0 = 1.39e-5;

        constexpr double PI = 3.14159265358979323846;

        double dot(const Vec3 &a, const Vec3 &b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        double norm(const Vec3 &a) {
            return std::sqrt(dot(a, a));
        }

        // a - (a.n) n
        Vec3 projectOntoPlane(const Vec3 &a, const Vec3 &n) {
            double an = dot(a, n);
            return {a.x - an * n.x, a.y - an * n.y, a.z - an * n.z};
        }

        // Angle (rad, 0..pi/2) between the in-plane direction 'direction' and the short axis
        double angleToShortAxis(const Vec3 &direction, double direction_norm, const Vec3 &short_axis) {
            return std::acos(std::min(1.0, std::fabs(dot(direction, short_axis)) / direction_norm));
        }

        // Characteristic length for the orientation-averaged chord length pi S / P (Eq. 5)
        double orientationAveragedLength(const LeafGeometry &geometry, double n) {
            double D0 = 4.0 * geometry.area / geometry.perimeter;
            double beta = std::tgamma(0.5) * std::tgamma(0.5 * n + 1.0) / std::tgamma(0.5 * n + 1.5);
            return D0 * std::pow(2.0 / PI * beta, 1.0 / (n - 1.0));
        }

    } // namespace

    AirProperties airProperties(double temperature, double pressure) {
        AirProperties air{};
        air.density = pressure / (R_DRY_AIR * temperature);
        double dynamic_viscosity = 1.458e-6 * std::pow(temperature, 1.5) / (temperature + 110.4); // Sutherland
        air.thermal_conductivity = 0.02624 * std::pow(temperature / 300.0, 0.8646);
        air.kinematic_viscosity = dynamic_viscosity / air.density;
        air.thermal_diffusivity = air.thermal_conductivity / (air.density * CP_AIR);
        return air;
    }

    double dissectionIndex(double length, double width) {
        return 1.112 * std::pow((length + width) / std::sqrt(length * width), 0.783) - 0.913;
    }

    double meanChordLength(double length, double width, double phi) {
        double s = std::sin(phi);
        double c = std::cos(phi);
        return PI * length * width / (4.0 * std::sqrt((s * width) * (s * width) + (c * length) * (c * length)));
    }

    double characteristicLength(double length, double width, double phi, double n) {
        if (std::fabs(n - 1.0) < 1e-12) {
            return meanChordLength(length, width, phi);
        }
        // chords parallel to the streamline vary as D(t) = D0 sqrt(1 - (t/t_max)^2), so Eq. 4 gives Lc^(n-1) = D0^(n-1) (2/pi) B(1/2, n/2 + 1)
        double D0 = 4.0 / PI * meanChordLength(length, width, phi);
        double beta = std::tgamma(0.5) * std::tgamma(0.5 * n + 1.0) / std::tgamma(0.5 * n + 1.5);
        return D0 * std::pow(2.0 / PI * beta, 1.0 / (n - 1.0));
    }

    LeafGeometry leafGeometry(double length, double width, const Vec3 &normal_up, const Vec3 &short_axis) {
        LeafGeometry geometry{};
        geometry.length = length;
        geometry.width = width;
        geometry.area = 0.25 * PI * length * width;
        geometry.aspect_ratio = length / width;
        geometry.dissection_index = dissectionIndex(length, width);
        geometry.perimeter = 2.0 * geometry.dissection_index * std::sqrt(PI * geometry.area);
        geometry.length_horizontal = geometry.area / geometry.perimeter;
        geometry.theta = std::acos(std::max(-1.0, std::min(1.0, normal_up.z))); // Eq. 11

        // steepest-ascent direction: projection of the vertical onto the leaf plane (Appendix B); undefined for a horizontal leaf
        Vec3 steepest_ascent = projectOntoPlane({0, 0, 1}, normal_up);
        double steepest_ascent_norm = norm(steepest_ascent);
        geometry.phi = (steepest_ascent_norm < 1e-12) ? 0.5 * PI : angleToShortAxis(steepest_ascent, steepest_ascent_norm, short_axis);

        geometry.mean_chord_length = meanChordLength(length, width, geometry.phi);
        geometry.length_inclined = (0.04511 * N_INCLINED + 1.006) * geometry.mean_chord_length;
        return geometry;
    }

    FreeConvectionResult freeConvection(double leaf_temperature, double air_temperature, double pressure, const LeafGeometry &geometry, bool free_transition) {
        FreeConvectionResult result{};
        result.film_temperature = 0.5 * (leaf_temperature + air_temperature);
        result.air = airProperties(result.film_temperature, pressure);

        result.dT = leaf_temperature - air_temperature;
        if (std::fabs(result.dT) < 1e-9) {
            result.dT = 1e-9;
        }

        double nu = result.air.kinematic_viscosity;
        result.Ra = GRAVITY * std::fabs(result.dT) * PRANDTL / (air_temperature * nu * nu); // Eq. 3g

        double sin_theta = std::sin(geometry.theta);
        double cos_theta = std::cos(geometry.theta);
        double Ra_025 = std::pow(result.Ra, 0.25);
        double nu_inclined_U = P3 * std::pow(geometry.length_inclined, -0.25) * std::pow(sin_theta, P4) * Ra_025; // Eq. 3b
        double nu_inclined_L = P3 * std::pow(geometry.length_inclined, -0.25) * std::pow(sin_theta, 0.25) * Ra_025; // Eq. 3c
        double nu_horizontal_U = P5 * std::pow(geometry.length_horizontal, -0.25) * std::pow(cos_theta, 0.25) * Ra_025; // Eq. 3d
        double nu_horizontal_L = P6 * std::pow(geometry.length_horizontal, -0.4) * std::pow(cos_theta, 0.2) * std::pow(result.Ra, 0.2); // Eq. 3e
        result.Nuc = 5.3 * std::sin(0.0338 * geometry.aspect_ratio + 0.607) / std::sqrt(geometry.area); // Eq. 3f

        // for a cooled leaf the right-hand sides of the upper and lower side are exchanged
        bool heated = result.dT >= 0.0;
        if (!heated) {
            std::swap(nu_inclined_U, nu_inclined_L);
            std::swap(nu_horizontal_U, nu_horizontal_L);
        }

        auto blend = [&](double nu_inclined, double nu_horizontal) { // Eq. 3a
            return std::pow(std::pow(std::pow(nu_inclined, P1) + std::pow(nu_horizontal, P1), P2 / P1) + std::pow(result.Nuc, P2), 1.0 / P2);
        };
        result.NuF_laminar_U = blend(nu_inclined_U, nu_horizontal_U);
        result.NuF_laminar_L = blend(nu_inclined_L, nu_horizontal_L);
        result.NuF_U = result.NuF_laminar_U;
        result.NuF_L = result.NuF_laminar_L;
        result.turbulent_share_U = 0.0;
        result.turbulent_share_L = 0.0;

        if (free_transition) {
            // turbulent free convection on the buoyantly unstable side (upper side of a heated leaf, lower side of a cooled leaf, both sides of a vertical leaf); the stable side stays laminar
            double NuF_turbulent = C_FREE_TURBULENT * std::cbrt(result.Ra);
            bool vertical = std::fabs(sin_theta - 1.0) < 1e-9;
            bool unstable_U = heated || vertical;
            bool unstable_L = !heated || vertical;
            auto transition = [&](double NuF_laminar, double &NuF, double &turbulent_share) {
                double sum = std::pow(NuF_laminar, M_FREE_TURBULENT) + std::pow(NuF_turbulent, M_FREE_TURBULENT);
                NuF = std::pow(sum, 1.0 / M_FREE_TURBULENT);
                turbulent_share = std::pow(NuF_turbulent, M_FREE_TURBULENT) / sum;
            };
            if (unstable_U) {
                transition(result.NuF_laminar_U, result.NuF_U, result.turbulent_share_U);
            }
            if (unstable_L) {
                transition(result.NuF_laminar_L, result.NuF_L, result.turbulent_share_L);
            }
        }

        return result;
    }

    double transitionOnsetReynolds(double turbulence_intensity_percent) {
        // Mayle (1991) Re_theta,t = 400 TI^-5/8 with the Blasius momentum thickness, Re_l = 3.63e5 TI^-1.25
        double TI = std::max(turbulence_intensity_percent, 1e-3);
        return std::clamp(3.63e5 * std::pow(TI, -1.25), RE_ONSET_MIN, RE_ONSET_SMOOTH);
    }

    void forcedNusseltParts(double reynolds, double turbulence_intensity_percent, double &laminar, double &transitional, double &turbulent) {
        double Re_l = transitionOnsetReynolds(turbulence_intensity_percent);
        double c = 0.9922 * std::log10(Re_l) - 3.013;
        // end of transition Re_u from continuity of the local Nusselt number: a_l Re_l^0.5 Pr^(1/3) (Re_u/Re_l)^c = 0.0296 Re_u^0.8 Pr^0.6
        double Re_u = std::pow(std::pow(Re_l, 0.5 - c) * A_LAMINAR_LOCAL / C_TURBULENT_LOCAL * std::pow(PRANDTL, 1.0 / 3.0 - 0.6), 1.0 / (0.8 - c));
        Re_u = std::max(Re_u, Re_l * 1.0001);

        laminar = 2.0 * A_LAMINAR_LOCAL * std::cbrt(PRANDTL) * std::sqrt(std::min(reynolds, Re_l));
        transitional = 0.0;
        if (reynolds > Re_l) {
            double nu_local_onset = A_LAMINAR_LOCAL * std::sqrt(Re_l) * std::cbrt(PRANDTL);
            double re_mid = std::clamp(reynolds, Re_l, Re_u);
            transitional = nu_local_onset / c * (std::pow(re_mid / Re_l, c) - 1.0);
        }
        turbulent = 0.0;
        if (reynolds > Re_u) {
            turbulent = 0.037 * std::pow(PRANDTL, 0.6) * std::max(std::pow(reynolds, 0.8) - std::pow(Re_u, 0.8), 0.0);
        }
    }

    void angleOfAttackFactors(double attack_angle_deg, double &factor_windward, double &factor_leeward) {
        double beta = attack_angle_deg * PI / 180.0;
        factor_windward = 1.0 + A_WINDWARD * std::sin(beta); // accelerated outer flow
        double w = std::exp(-std::pow(attack_angle_deg / BETA_SEPARATION_DEG, 2)); // reattaching separation bubble -> wake
        factor_leeward = w * (1.0 + A_REATTACHMENT * std::sin(2.0 * beta)) + (1.0 - w) * C_WAKE;
    }

    void windStreamlineAngle(const Vec3 &normal_up, const Vec3 &short_axis, const Vec3 &wind_direction, double &phi_wind, double &in_plane_fraction) {
        Vec3 wind_in_plane = projectOntoPlane(wind_direction, normal_up);
        in_plane_fraction = norm(wind_in_plane);
        if (in_plane_fraction < 1e-9) {
            phi_wind = 0.5 * PI;
            in_plane_fraction = 0.0;
            return;
        }
        phi_wind = angleToShortAxis(wind_in_plane, in_plane_fraction, short_axis);
    }

    ConductanceResult leafConductances(double leaf_temperature, double air_temperature, double wind_speed, double pressure, const LeafGeometry &geometry, const Vec3 &normal_up, const Vec3 &short_axis, const Vec3 *wind_direction,
                                       const ModelOptions &options) {
        ConductanceResult result{};
        result.free = freeConvection(leaf_temperature, air_temperature, pressure, geometry, options.free_transition);
        const double kappa = result.free.air.thermal_diffusivity;
        const double nu = result.free.air.kinematic_viscosity;

        // streamline length of the forced convection along the wind projected onto the leaf plane (flowline concept); orientation-averaged chord length if the direction is unknown or normal to the leaf
        result.streamline_length = orientationAveragedLength(geometry, N_FORCED);
        if (wind_direction != nullptr) {
            double phi_wind, in_plane_fraction;
            windStreamlineAngle(normal_up, short_axis, *wind_direction, phi_wind, in_plane_fraction);
            if (in_plane_fraction >= 1e-6) {
                result.streamline_length = characteristicLength(geometry.length, geometry.width, phi_wind, N_FORCED);
            }
        }

        result.reynolds = std::max(wind_speed * result.streamline_length / nu, 0.0);
        double laminar, transitional, turbulent;
        double NuF_forced;
        if (options.forced_transition) {
            forcedNusseltParts(result.reynolds, options.turbulence_intensity_percent, laminar, transitional, turbulent);
            double total = laminar + transitional + turbulent;
            NuF_forced = total / result.streamline_length;
            result.forced_turbulent_share = (total > 0.0) ? turbulent / total : 0.0;
        } else {
            NuF_forced = C_FORCED_LAMINAR * std::sqrt(result.reynolds) * std::cbrt(PRANDTL) / result.streamline_length;
            result.forced_turbulent_share = 0.0;
        }

        // angle of attack: finite-plate enhancement and windward/leeward factors
        result.NuF_forced_U = NuF_forced;
        result.NuF_forced_L = NuF_forced;
        result.attack_angle_deg = 0.0;
        result.windward_upper = true;
        if (options.angle_model) {
            if (wind_direction != nullptr) {
                double wind_normal = dot(*wind_direction, normal_up);
                result.attack_angle_deg = std::asin(std::min(1.0, std::fabs(wind_normal))) * 180.0 / PI;
                result.windward_upper = wind_normal < 0.0; // wind moving against the upward normal hits the upper side
            }
            double factor_windward, factor_leeward;
            angleOfAttackFactors(result.attack_angle_deg, factor_windward, factor_leeward);
            result.NuF_forced_U = F_EDGE * NuF_forced * (result.windward_upper ? factor_windward : factor_leeward);
            result.NuF_forced_L = F_EDGE * NuF_forced * (result.windward_upper ? factor_leeward : factor_windward);
        }

        // mixed convection by cubic blending, and the scalar conductances with mode-dependent diffusivity exponents (generalization of Eq. 17)
        double diffusivity_scale = std::pow(result.free.film_temperature / 273.15, 1.75) * 101325.0 / pressure;
        double D_H2O = D_H2O_0 * diffusivity_scale;
        double D_CO2 = D_CO2_0 * diffusivity_scale;

        auto side = [&](double NuF_free, double NuF_free_laminar, double free_turbulent_share, double NuF_forced_side, double &gb, double &gv, double &gc, double &k, double &forced_share) {
            double free_cubed = NuF_free * NuF_free * NuF_free;
            double forced_cubed = NuF_forced_side * NuF_forced_side * NuF_forced_side;
            gb = kappa * std::cbrt(free_cubed + forced_cubed);
            forced_share = forced_cubed / (free_cubed + forced_cubed);

            double conduction_share = std::clamp(result.free.Nuc / std::max(NuF_free_laminar, result.free.Nuc), 0.0, 1.0);
            double k_free_laminar = K_FREE_LAMINAR + (K_CONDUCTION - K_FREE_LAMINAR) * conduction_share;
            double k_free = (1.0 - free_turbulent_share) * k_free_laminar + free_turbulent_share * K_FREE_TURBULENT;
            double k_forced = (1.0 - result.forced_turbulent_share) * K_FORCED_LAMINAR + result.forced_turbulent_share * K_FORCED_TURBULENT;
            k = (1.0 - forced_share) * k_free + forced_share * k_forced;

            gv = gb * std::pow(D_H2O / kappa, k);
            gc = gb * std::pow(D_CO2 / kappa, k);
        };
        side(result.free.NuF_U, result.free.NuF_laminar_U, result.free.turbulent_share_U, result.NuF_forced_U, result.gb_U, result.gv_U, result.gc_U, result.k_U, result.forced_share_U);
        side(result.free.NuF_L, result.free.NuF_laminar_L, result.free.turbulent_share_L, result.NuF_forced_L, result.gb_L, result.gv_L, result.gc_L, result.k_L, result.forced_share_L);

        result.molar_density = pressure / (R_UNIVERSAL * result.free.film_temperature);
        return result;
    }

    void orientationFromAngles(double alpha_x_deg, double alpha_y_deg, double leaf_azimuth_deg, Vec3 &normal_up, Vec3 &short_axis) {
        double ax = alpha_x_deg * PI / 180.0;
        double ay = alpha_y_deg * PI / 180.0;
        double cx = std::cos(ax), sx = std::sin(ax);
        double cy = std::cos(ay), sy = std::sin(ay);
        // leaf frame (x = short axis, y = long axis, z = up) after the rotations Rx(alpha_x) Ry(alpha_y) (Appendix B)
        Vec3 normal_leaf{-sy, -sx * cy, cx * cy};
        Vec3 short_leaf{cy, -sx * sy, cx * sy};
        // the unrotated leaf axes in the (east, north, up) frame
        double A = leaf_azimuth_deg * PI / 180.0;
        Vec3 x_world{-std::cos(A), std::sin(A), 0};
        Vec3 y_world{-std::sin(A), -std::cos(A), 0};
        auto toWorld = [&](const Vec3 &v) { return Vec3{v.x * x_world.x + v.y * y_world.x, v.x * x_world.y + v.y * y_world.y, v.z}; };
        normal_up = toWorld(normal_leaf);
        short_axis = toWorld(short_leaf);
    }

    Vec3 windVectorFromDirection(double wind_from_deg) {
        double toward = (wind_from_deg + 180.0) * PI / 180.0;
        return {std::sin(toward), std::cos(toward), 0};
    }

} // namespace graefe
