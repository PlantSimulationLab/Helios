/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef HELIOS_PLANT_ARCHITECTURE_INTERNAL_H
#define HELIOS_PLANT_ARCHITECTURE_INTERNAL_H

/** \file "PlantArchitectureInternal.h" Declarations shared between the plant architecture library model builders and organ prototype functions. Not part of the public API. */

#include "PlantArchitecture.h"

//! Look up a build parameter, falling back to a default if it was not given
/**
 * Used by the library plant builders and by flower/fruit prototype functions to read the build parameters passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
 * \param[in] build_parameters Map of parameter names to values
 * \param[in] parameter_name Name of the parameter to retrieve
 * \param[in] default_value Default value to use if parameter not specified
 * \param[in] min_value Minimum valid value for the parameter
 * \param[in] max_value Maximum valid value for the parameter
 * \param[in] parameter_description Description of the parameter for error messages
 * \return The parameter value from the map if specified, otherwise the default value
 * \throws helios_runtime_error if the parameter value is outside the valid range
 */
[[nodiscard]] float getBuildParameterValue(const std::map<std::string, float> &build_parameters, const std::string &parameter_name, float default_value, float min_value, float max_value, const std::string &parameter_description);

//! Look up an integer-valued build parameter, falling back to a default if it was not given
/**
 * As \ref getBuildParameterValue(), and in addition the value must be a whole number.
 * \param[in] build_parameters Map of parameter names to values
 * \param[in] parameter_name Name of the parameter to retrieve
 * \param[in] default_value Default value to use if parameter not specified
 * \param[in] min_value Minimum valid value for the parameter
 * \param[in] max_value Maximum valid value for the parameter
 * \param[in] parameter_description Description of the parameter for error messages
 * \return The parameter value from the map if specified, otherwise the default value
 * \throws helios_runtime_error if the parameter value is outside the valid range or is not a whole number
 */
[[nodiscard]] int getBuildParameterIntegerValue(const std::map<std::string, float> &build_parameters, const std::string &parameter_name, int default_value, int min_value, int max_value, const std::string &parameter_description);

//! Look up a declared build parameter, falling back to a default if it was not given
/**
 * As \ref getBuildParameterValue(), with the name, range and description taken from the declaration the model registers, so that the range enforced here is the one a distribution is checked against.
 * \param[in] build_parameters Map of parameter names to values
 * \param[in] declaration Declaration of the parameter, which must not be a whole-number parameter
 * \param[in] default_value Default value to use if parameter not specified
 * \return The parameter value from the map if specified, otherwise the default value
 */
[[nodiscard]] float getBuildParameterValue(const std::map<std::string, float> &build_parameters, const BuildParameterDeclaration &declaration, float default_value);

//! Look up a declared whole-number build parameter, falling back to a default if it was not given
/**
 * As \ref getBuildParameterIntegerValue(), with the name, range and description taken from the declaration the model registers.
 * \param[in] build_parameters Map of parameter names to values
 * \param[in] declaration Declaration of the parameter, which must be a whole-number parameter
 * \param[in] default_value Default value to use if parameter not specified
 * \return The parameter value from the map if specified, otherwise the default value
 */
[[nodiscard]] int getBuildParameterIntegerValue(const std::map<std::string, float> &build_parameters, const BuildParameterDeclaration &declaration, int default_value);

//! Build parameters with every value given as a constant distribution
/**
 * \param[in] build_parameters Map of parameter names to values.
 * \return Map of the same names to \ref BuildParameterDistribution::constant() of each value.
 */
[[nodiscard]] std::map<std::string, BuildParameterDistribution> constantBuildParameterDistributions(const std::map<std::string, float> &build_parameters);

//! Grape cluster parameters that can be set through the build parameters of a library grapevine
struct GrapeClusterParameters {
    //! Berry diameter in meters
    float berry_diameter = 0.012f;
    //! Mean number of berries in a cluster. The count of each cluster varies by up to 17% either side of it.
    float cluster_berry_count = 90.f;

    //! Cluster parameters set from the build parameters "berry_diameter" and "cluster_berry_count", keeping the defaults for any not given
    /**
     * \throws helios_runtime_error if a value is outside its valid range.
     */
    [[nodiscard]] static GrapeClusterParameters fromBuildParameters(const std::map<std::string, float> &build_parameters);
    //! Declarations of all build parameter keys read by \ref fromBuildParameters()
    [[nodiscard]] static std::vector<BuildParameterDeclaration> buildParameterDeclarations();
};

//! Morphological parameters of a procedurally generated sorghum panicle
/**
 * A panicle is a central rachis bearing whorls of primary branches. Each primary branch carries short racemes along its distal, fertile portion, and each raceme carries a row of grains, each partly enclosed by a
 * pair of glumes. Every grain is resolved as its own low-polygon mesh; nothing is represented by a texture. The parameters span the range of panicle types seen across grain sorghum genotypes: compact
 * cylindrical heads with short, erect branches, semi-compact heads, and loose open heads whose long, spreading branches expose the rachis.
 *
 * Lengths given as a fraction are fractions of the panicle length, so the shape is preserved when the length is changed. Grain dimensions are absolute.
 *
 * The defaults describe a semi-compact panicle with tan grain. \ref compact() and \ref loose() return the two ends of the range.
 *
 * The primitive count is set mainly by the grain count and the number of facets per grain. Each grain costs 4 x grain_subdivisions triangles, plus 4 more when it has glumes.
 */
struct SorghumPanicleParameters {

    // ---- Overall form ---- //

    //! Length of the panicle in meters, from its lowest branch to its tip
    float length = 0.25f;
    //! Coefficient of variation of the panicle length between panicles. Grain count varies in proportion to length.
    float length_variation = 0.08f;
    //! Fraction of the panicle length spanned by the rachis. The uppermost primary branches fill the remainder.
    float rachis_length_fraction = 0.85f;
    //! Total bend of the rachis from base to tip in degrees. Zero gives a straight, erect panicle.
    float rachis_curvature = 0.f;
    //! How strongly the grains are drawn into a continuous layer over the head, from 0 for a loose head to 1 for a closed one
    /**
     * At zero each grain stays where its raceme places it, so the head is only as dense as its branching. Larger values draw the grains into a layer a few grains deep just inside the envelope of
     * the branches, pulling in grains that stand out beyond it and moving out grains from the inner parts of the branches, until they bear on one another; at one the head is a closed surface of
     * grain with the branches hidden beneath it. A real head holds its grain this way rather than solid through, so the outline of the head is still set by the branch lengths and angles.
     */
    float compactness = 0.8f;
    //! Radius of the rachis at its base in meters. The rachis tapers to one third of this at its tip.
    /**
     * The rachis continues the peduncle, so its base radius should equal the peduncle radius where the two meet. The default matches the peduncle of the library sorghum plant at maturity, which is sized
     * from the culm tip and comes to 4.5-4.8 mm.
     */
    float rachis_radius = 0.0046f;

    // ---- Primary branches ---- //

    //! Number of primary branches on the rachis
    int primary_branch_count = 70;
    //! Length of the basal primary branches as a fraction of the panicle length
    float branch_length_base = 0.2f;
    //! Length of the primary branches at the middle of the rachis as a fraction of the panicle length
    float branch_length_middle = 0.21f;
    //! Length of the apical primary branches as a fraction of the panicle length
    float branch_length_tip = 0.12f;
    //! Coefficient of variation of primary branch length
    float branch_length_variation = 0.2f;
    //! Insertion angle of the basal primary branches in degrees from the rachis axis
    float branch_angle_base = 45.f;
    //! Insertion angle of the apical primary branches in degrees from the rachis axis
    float branch_angle_tip = 25.f;
    //! Angle in degrees through which each primary branch turns back toward the rachis direction between its base and its tip
    float branch_upcurve = 15.f;
    //! Fraction of each primary branch, measured from its base, that bears no racemes
    float branch_bare_fraction = 0.15f;
    //! Fraction of primary branches that splay out from the head, giving the ragged outline common in the field
    float splayed_branch_fraction = 0.04f;
    //! Insertion angle in degrees from the rachis axis of a splayed primary branch
    float splayed_branch_angle = 65.f;
    //! Radius of a primary branch at its base in meters
    float branch_radius = 0.0007f;

    // ---- Racemes ---- //

    //! Mean number of grains on each raceme
    int grains_per_raceme = 5;
    //! Length of a raceme as a fraction of the length of the primary branch bearing it. A raceme is never shorter than its grains need.
    float raceme_length_fraction = 0.2f;
    //! Angle in degrees between a raceme and the primary branch bearing it
    float raceme_angle = 35.f;

    // ---- Grain ---- //

    //! Mean number of grains on a panicle of the nominal length
    int grain_count = 2000;
    //! Grain length in meters along its long axis
    float grain_length = 0.0045f;
    //! Grain width in meters
    float grain_width = 0.0040f;
    //! Grain thickness in meters, across its flattened faces
    float grain_thickness = 0.0034f;
    //! Coefficient of variation of grain size
    float grain_size_variation = 0.08f;
    //! Mean grain color
    helios::RGBcolor grain_color = helios::make_RGBcolor(0.78f, 0.62f, 0.42f);
    //! Coefficient of variation of grain brightness between grains
    float grain_color_variation = 0.08f;
    //! Number of facets around a grain. Each grain costs 4 x grain_subdivisions triangles.
    uint grain_subdivisions = 4;

    // ---- Glumes ---- //

    //! Glume length as a fraction of grain length. Zero omits the glumes; values near one enclose the grain.
    float glume_length_fraction = 0.55f;
    //! Angle in degrees by which the glumes gape open from the grain
    float glume_spread = 5.f;
    //! Mean glume color
    helios::RGBcolor glume_color = helios::make_RGBcolor(0.45f, 0.30f, 0.18f);

    // ---- Branches ---- //

    //! Color of the rachis, primary branches and racemes. The library sorghum gives its peduncle this color so the peduncle and rachis meet without a change in color.
    helios::RGBcolor branch_color = helios::make_RGBcolor(0.42f, 0.45f, 0.28f);
    //! Number of radial subdivisions of the primary branch and raceme tubes
    uint branch_radial_subdivisions = 3;
    //! Number of radial subdivisions of the rachis tube. The default matches the library sorghum peduncle, so the two tubes meet with the same cross-section.
    uint rachis_radial_subdivisions = 10;

    //! Compact cylindrical panicle with short, erect, densely packed branches
    static SorghumPanicleParameters compact();
    //! Loose, open panicle with long, spreading branches and a visible rachis
    static SorghumPanicleParameters loose();

    //! Panicle parameters set from the build parameters of a library sorghum plant
    /**
     * The key "panicle_type" (0 to 1) is applied first: 0 gives \ref compact(), 0.5 the defaults, and 1 \ref loose(), with values between blending linearly between neighbouring presets. Each other key
     * then overrides one field, and fields not given keep their value. Keys are listed in the plant library documentation (see \ref PlantArchBuildParams).
     * \param[in] build_parameters Map of build parameter names to values, as passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
     * \return Panicle parameters with the given values applied.
     * \throws helios_runtime_error if a value is outside its valid range.
     */
    [[nodiscard]] static SorghumPanicleParameters fromBuildParameters(const std::map<std::string, float> &build_parameters);

    //! Declarations of all build parameter keys read by \ref fromBuildParameters()
    [[nodiscard]] static std::vector<BuildParameterDeclaration> buildParameterDeclarations();
};

//! Build a sorghum panicle with resolved grains
/**
 * The panicle is built along +x from its attachment at the origin, in meters. Random variation between calls is drawn from the Context random generator.
 * \param[in] context_ptr Pointer to the Helios Context.
 * \param[in] parameters Morphology of the panicle.
 * \return Object ID of the panicle polymesh. All of its primitives carry the object_label "fruit".
 */
uint buildSorghumPanicle(helios::Context *context_ptr, const SorghumPanicleParameters &parameters);

//! Build parameter "tiller_count" of the library sorghum
inline constexpr BuildParameterDeclaration sorghum_tiller_count{"tiller_count", 0.f, 6.f, true, "number of tillers of a sorghum plant"};

//! Build parameter "tiller_count" of the library wheat
inline constexpr BuildParameterDeclaration wheat_tiller_count{"tiller_count", 0.f, 4.f, true, "number of tillers of a wheat plant"};

//! Number of tillers of a library sorghum plant, read from its build parameter "tiller_count" (default 0, range 0-6)
/**
 * \param[in] build_parameters Map of build parameter names to values, as passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
 * \return Number of main-stem leaf axils that produce a tiller.
 */
[[nodiscard]] int sorghumTillerCount(const std::map<std::string, float> &build_parameters);

//! Number of tillers of a library wheat plant, read from its build parameter "tiller_count" (default 0, range 0-4)
/**
 * \param[in] build_parameters Map of build parameter names to values, as passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
 * \return Number of main-stem leaf axils that produce a tiller.
 */
[[nodiscard]] int wheatTillerCount(const std::map<std::string, float> &build_parameters);

#endif // HELIOS_PLANT_ARCHITECTURE_INTERNAL_H
