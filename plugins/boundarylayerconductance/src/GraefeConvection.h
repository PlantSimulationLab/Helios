/** \file "GraefeConvection.h" Internal header for the leaf boundary-layer conductance model of Graefe et al. (2022), with the mixed-convection extensions.

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

#ifndef GRAEFE_CONVECTION_H
#define GRAEFE_CONVECTION_H

//! Leaf boundary-layer conductance model of Graefe, Grosch & Bitterlich (2022) with the forced/mixed-convection extensions of the reference implementation
/**
 * Equation numbers refer to Graefe et al. (2022). The leaf is a flat ellipse of length L and width W (L >= W). Its upper side "U" is the side whose normal points upward, and its lower side "L" is the other one.
 * Orientation is given by vectors in a world frame with z pointing up: the upward-pointing unit normal and the in-plane unit vector of the short (width) axis. Wind is a horizontal unit vector in the direction
 * the air moves toward. All quantities are SI and in double precision. Conductances are in m/s; multiply by the molar density of air to get mol/m^2/s.
 */
namespace graefe {

    //! Minimal double-precision 3D vector
    struct Vec3 {
        double x = 0; //!< x component (east)
        double y = 0; //!< y component (north)
        double z = 0; //!< z component (up)
    };

    //! Dry-air properties at a given temperature and pressure
    struct AirProperties {
        double density; //!< Air density (kg/m^3)
        double kinematic_viscosity; //!< Kinematic viscosity (m^2/s), Sutherland's law
        double thermal_diffusivity; //!< Thermal diffusivity (m^2/s)
        double thermal_conductivity; //!< Thermal conductivity (W/m/K)
    };

    //! Geometric descriptors of an elliptical leaf (Section 2.2)
    struct LeafGeometry {
        double length; //!< Leaf length along the long axis (m)
        double width; //!< Leaf width along the short axis (m)
        double area; //!< One-sided area S = pi/4 L W (m^2)
        double aspect_ratio; //!< L/W
        double dissection_index; //!< Dissection index DI (Eq. 8)
        double perimeter; //!< Perimeter implied by DI (m)
        double length_horizontal; //!< Characteristic length of a horizontal leaf Lh = S/P (Eq. 7, m)
        double theta; //!< Angle between the upward leaf normal and the vertical (rad, Eq. 11)
        double phi; //!< Angle between the steepest-ascent direction and the short axis (rad, Eq. 10)
        double mean_chord_length; //!< Mean chord length along the steepest-ascent direction LCo(phi) (m, Eq. 9)
        double length_inclined; //!< Characteristic length of an inclined leaf Li (m, Eq. 9)
    };

    //! Switches for the extensions of the general model. The defaults are those of the general model (leaf_conductances() in the reference implementation).
    struct ModelOptions {
        bool forced_transition = true; //!< Laminar-turbulent transition of the forced convection (Lienhard 2020); otherwise the laminar flat-plate law with coefficient 0.664
        bool free_transition = true; //!< Laminar-turbulent transition of the free convection on the buoyantly unstable side
        bool angle_model = true; //!< Finite-plate enhancement and windward/leeward angle-of-attack factors of the forced convection
        double turbulence_intensity_percent = 1.0; //!< Free-stream turbulence intensity (%), which sets the onset of transition
    };

    //! Free-convection result for both leaf sides
    struct FreeConvectionResult {
        double NuF_U; //!< Flux-proportional Nusselt number of the upper side (1/m)
        double NuF_L; //!< Flux-proportional Nusselt number of the lower side (1/m)
        double NuF_laminar_U; //!< Laminar part of NuF_U (equal to NuF_U without the free-convection transition)
        double NuF_laminar_L; //!< Laminar part of NuF_L
        double turbulent_share_U; //!< Turbulent share of the free convection on the upper side
        double turbulent_share_L; //!< Turbulent share of the free convection on the lower side
        double Nuc; //!< Conduction limit (1/m, Eq. 3f)
        double Ra; //!< Modified Rayleigh number without length scale (1/m^3, Eq. 3g)
        double dT; //!< Leaf minus air temperature difference used for buoyancy (K)
        double film_temperature; //!< (TL + TA)/2 (K)
        AirProperties air; //!< Air properties at the film temperature
    };

    //! Result of the general model for both leaf sides
    struct ConductanceResult {
        FreeConvectionResult free; //!< Free-convection part
        double streamline_length; //!< Characteristic length of the forced convection Lf (m)
        double reynolds; //!< Reynolds number on the streamline length
        double attack_angle_deg; //!< Angle of attack of the wind on the leaf plane (deg, 0 = parallel)
        bool windward_upper; //!< True if the wind hits the upper side first
        double NuF_forced_U; //!< Forced-convection flux-proportional Nusselt number of the upper side (1/m)
        double NuF_forced_L; //!< Forced-convection flux-proportional Nusselt number of the lower side (1/m)
        double gb_U; //!< Boundary-layer conductance to heat of the upper side (m/s)
        double gb_L; //!< Boundary-layer conductance to heat of the lower side (m/s)
        double gv_U; //!< Boundary-layer conductance to water vapor of the upper side (m/s)
        double gv_L; //!< Boundary-layer conductance to water vapor of the lower side (m/s)
        double gc_U; //!< Boundary-layer conductance to CO2 of the upper side (m/s)
        double gc_L; //!< Boundary-layer conductance to CO2 of the lower side (m/s)
        double k_U; //!< Effective diffusivity exponent of the upper side
        double k_L; //!< Effective diffusivity exponent of the lower side
        double forced_share_U; //!< Forced share of the mixed convection on the upper side
        double forced_share_L; //!< Forced share of the mixed convection on the lower side
        double forced_turbulent_share; //!< Turbulent share of the forced convection
        double molar_density; //!< Molar density of air at the film temperature (mol/m^3), to convert conductances to mol/m^2/s
    };

    //! Dry-air density, kinematic viscosity, thermal diffusivity and thermal conductivity
    /**
     * \param[in] temperature Air temperature (K).
     * \param[in] pressure Air pressure (Pa).
     */
    AirProperties airProperties(double temperature, double pressure);

    //! Dissection index of an ellipse (Eq. 8)
    double dissectionIndex(double length, double width);

    //! Mean chord length LCo along a direction at angle phi to the short axis (Eq. 9 / A5)
    double meanChordLength(double length, double width, double phi);

    //! Exact flux-weighted characteristic length of Eq. 4 for an ellipse, for a heat transfer coefficient proportional to length^(n-1)
    double characteristicLength(double length, double width, double phi, double n);

    //! Geometric descriptors of an elliptical leaf
    /**
     * \param[in] length Leaf length (m), at least the width.
     * \param[in] width Leaf width (m).
     * \param[in] normal_up Unit leaf normal with a non-negative z component.
     * \param[in] short_axis Unit vector of the short leaf axis, in the leaf plane.
     */
    LeafGeometry leafGeometry(double length, double width, const Vec3 &normal_up, const Vec3 &short_axis);

    //! Free convection of both leaf sides (Eq. 1, 3a-3g), optionally with the laminar-turbulent transition on the buoyantly unstable side
    /**
     * \param[in] leaf_temperature Leaf temperature (K).
     * \param[in] air_temperature Air temperature (K).
     * \param[in] pressure Air pressure (Pa).
     * \param[in] geometry Leaf geometry.
     * \param[in] free_transition Apply the laminar-turbulent transition of the free convection.
     */
    FreeConvectionResult freeConvection(double leaf_temperature, double air_temperature, double pressure, const LeafGeometry &geometry, bool free_transition);

    //! Onset Reynolds number of the laminar-turbulent transition as a function of the turbulence intensity (%), after Mayle (1991)
    double transitionOnsetReynolds(double turbulence_intensity_percent);

    //! Laminar, transitional and turbulent contributions to the average flat-plate Nusselt number of Lienhard (2020), isothermal plate
    void forcedNusseltParts(double reynolds, double turbulence_intensity_percent, double &laminar, double &transitional, double &turbulent);

    //! Multipliers of the parallel-flow Nusselt number for the windward and leeward sides at an angle of attack (deg)
    void angleOfAttackFactors(double attack_angle_deg, double &factor_windward, double &factor_leeward);

    //! Angle between the wind streamlines projected onto the leaf plane and the short axis (rad), and the fraction of the wind speed lying in the leaf plane
    void windStreamlineAngle(const Vec3 &normal_up, const Vec3 &short_axis, const Vec3 &wind_direction, double &phi_wind, double &in_plane_fraction);

    //! General model: heat, water vapor and CO2 boundary-layer conductances of both leaf sides
    /**
     * \param[in] leaf_temperature Leaf temperature (K).
     * \param[in] air_temperature Air temperature (K).
     * \param[in] wind_speed Wind speed (m/s).
     * \param[in] pressure Air pressure (Pa).
     * \param[in] geometry Leaf geometry from leafGeometry().
     * \param[in] normal_up The normal passed to leafGeometry().
     * \param[in] short_axis The short axis passed to leafGeometry().
     * \param[in] wind_direction Horizontal unit vector the wind blows toward, or nullptr if unknown (orientation-averaged streamline length, zero angle of attack).
     * \param[in] options Model switches.
     */
    ConductanceResult leafConductances(double leaf_temperature, double air_temperature, double wind_speed, double pressure, const LeafGeometry &geometry, const Vec3 &normal_up, const Vec3 &short_axis, const Vec3 *wind_direction,
                                       const ModelOptions &options);

    //! Orientation vectors for the angle convention of the reference implementation (used to check the port against its reference values)
    /**
     * The leaf (short axis along x, long axis along y) is rotated about y by alpha_y and then about x by alpha_x, and the leaf frame is placed in a world frame of (east, north, up) so that the upper side faces
     * the compass azimuth leaf_azimuth_deg.
     * \param[in] alpha_x_deg Rotation about the short axis (inclination from horizontal, deg).
     * \param[in] alpha_y_deg Rotation about the long axis (deg).
     * \param[in] leaf_azimuth_deg Compass direction the upper side faces (deg, 180 = south).
     * \param[out] normal_up Leaf normal.
     * \param[out] short_axis Short leaf axis.
     */
    void orientationFromAngles(double alpha_x_deg, double alpha_y_deg, double leaf_azimuth_deg, Vec3 &normal_up, Vec3 &short_axis);

    //! Horizontal unit vector the wind blows toward, in the (east, north, up) frame, for a meteorological wind direction (deg, the direction the wind comes from)
    Vec3 windVectorFromDirection(double wind_from_deg);

} // namespace graefe

#endif
