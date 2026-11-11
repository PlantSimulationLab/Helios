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

//! Morphological parameters of a procedurally generated maize tassel
/**
 * A tassel is a central axis that is bare where it bears the primary branches, in its lower part, and continues above them as the central spike, which carries spikelet pairs in many ranks. Each primary branch
 * is bare at its base and carries spikelet pairs in two rows beyond that. A spikelet is a closed low-polygon hull standing for its two glumes, and the florets that have reached anthesis hang their three
 * anthers below it on hair-thin filaments. Nothing is represented by a texture.
 *
 * Tassel form differs strongly and heritably between genotypes, and the three kinds of plant described by the defaults, \ref fewBranched() and \ref manyBranched() are points on a continuum rather than
 * recognized classes: no published classification of tassel types was found. The quantities that were measured are noted on the fields. Those noted as read from photographs have no published measurement.
 *
 * Lengths of the axis and the branches are absolute, and the number of spikelets follows from them through the spikelet densities. The primitive count is set mainly by the number of spikelets: each costs 4 x
 * spikelet_subdivisions triangles, and each spikelet with anthers costs 21 more for its three anthers and filaments.
 */
struct MaizeTasselParameters {

    // ---- Overall form ---- //

    //! Length of the tassel in meters from its lowest primary branch to its tip, excluding the peduncle
    /** The default is the sum of the branched and central spike lengths of 20 commercial hybrids, 12.11 and 26.60 cm (Silveira et al. 2021, Australian Journal of Crop Science, doi:10.21475/ajcs.21.15.08.p3228). Inbred means run 21-54 cm (Gage et al. 2017, Plant Methods, doi:10.1186/s13007-017-0172-8). */
    float length = 0.387f;
    //! Coefficient of variation of the tassel length between tassels. The branch lengths vary in proportion to it. Not measured; plant-to-plant variation within a genotype.
    float length_variation = 0.06f;
    //! Fraction of the tassel length spanned by the central spike, the part of the axis above the highest branch
    /** 26.60 of 38.71 cm in the hybrids of Silveira et al. (2021), and 0.66-0.70 in the populations of Gage et al. (2018, Genetics 210:1125). Ignored when there are no primary branches, when all of the axis is central spike. */
    float central_spike_fraction = 0.69f;
    //! Total bend of the axis from base to tip in degrees, in a plane of random orientation. Read from photographs; the tortuosity of Gage et al. (2017) confirms a usually slight bend.
    float axis_curvature = 8.f;
    //! Radius of the axis at its base in meters. It continues the peduncle, so the default matches the peduncle of the library maize plant at maturity, which is sized from the culm tip.
    float axis_radius = 0.00675f;
    //! Radius of the central spike at its tip in meters
    float axis_tip_radius = 0.0008f;

    // ---- Primary branches ---- //

    //! Mean number of primary branches
    /** Means of 11.41, 9.91 and 9.16 by year in the hybrids of Silveira et al. (2021). Inbred means run from 2.75 to 25, and landraces are reported up to about 58 (Wu et al. 2016, Plant Biotechnology Journal, doi:10.1111/pbi.12519). */
    int primary_branch_count = 10;
    //! Coefficient of variation of the number of primary branches between tassels. The inbred B73 is 10.67 +/- 2.47 (Gallavotti et al. 2011, The Plant Cell, doi:10.1105/tpc.111.084590).
    float primary_branch_count_variation = 0.15f;
    //! Number of secondary branches, each borne near the base of one of the lowest primary branches. Means of 2.08-2.59 in the hybrids of Silveira et al. (2021).
    int secondary_branch_count = 2;
    //! Length of the longest primary branches in meters
    /** With the default profile below this gives a mean of 16.5 cm, matching 15.65 and 16.79 cm in B73 and A619 (Gallavotti et al. 2011); the longest match the 20 cm modal length of 180 scanned tassels (Zhang et al. 2023, Plant Methods 19:76). */
    float branch_length = 0.20f;
    //! Length of the uppermost primary branch as a fraction of the longest. The branches below 30% of the way up the branched zone are the longest. The profile is read from photographs.
    float branch_length_tip_fraction = 0.5f;
    //! Coefficient of variation of primary branch length
    float branch_length_variation = 0.1f;
    //! Insertion angle of the lowest primary branch in degrees from the axis
    /** Between the mean of 32.3 degrees in 571 doubled haploid lines, whose genotype means range over 10.8-59.6 (Zhu et al. 2023, Plants, doi:10.3390/plants12030680), and the 44 of the inbred PSA104 (US patent 6646188). */
    float branch_angle_base = 38.f;
    //! Insertion angle of the uppermost primary branch in degrees from the axis. The decline toward the top is read from photographs.
    float branch_angle_tip = 20.f;
    //! Angle in degrees through which the lowest primary branch arches downward between its base and its tip. Read from photographs; no measurement was found.
    float branch_droop_base = 30.f;
    //! Angle in degrees through which the uppermost primary branch arches downward between its base and its tip. Read from photographs; no measurement was found.
    float branch_droop_tip = 5.f;
    //! Fraction of each branch, measured from its base, that bears no spikelets. Read from photographs.
    float branch_bare_fraction = 0.12f;
    //! Radius of a primary branch at its base in meters. The "radial size" of a B73 branch is 1.96 +/- 0.36 mm (Gallavotti et al. 2011).
    float branch_radius = 0.00095f;

    // ---- Spikelets ---- //

    //! Spikelets per centimeter of central spike. About 14.1 in B73 (Bommert et al. 2005, Development, doi:10.1242/dev.01671).
    float spike_spikelet_density = 14.1f;
    //! Spikelets per centimeter of primary branch. About 5.2 in B73 (Bommert et al. 2005).
    float branch_spikelet_density = 5.2f;
    //! Spikelet length in meters. Lower glumes average 10.2 mm, with variety means of 6.7-13.9 mm (Alava 1952, Annals of the Missouri Botanical Garden).
    float spikelet_length = 0.0102f;
    //! Spikelet width in meters. Read from photographs.
    float spikelet_width = 0.0032f;
    //! Spikelet thickness in meters. Read from photographs.
    float spikelet_thickness = 0.0026f;
    //! Coefficient of variation of spikelet size
    float spikelet_size_variation = 0.08f;
    //! Angle in degrees between a spikelet and the axis or branch bearing it
    float spikelet_angle = 16.f;
    //! Number of facets around a spikelet. Each spikelet costs 4 x spikelet_subdivisions triangles.
    uint spikelet_subdivisions = 4;
    //! Color of the body of the glumes. Read from photographs; patents give values such as Munsell 5GY 5/4 to 2.5GY 8/4 with varying anthocyanin.
    helios::RGBcolor glume_color = helios::make_RGBcolor(0.40f, 0.46f, 0.24f);
    //! Color of the tips of the glumes, where anthocyanin and senescence show first. Read from photographs.
    helios::RGBcolor glume_tip_color = helios::make_RGBcolor(0.50f, 0.46f, 0.27f);
    //! Coefficient of variation of glume brightness between spikelets
    float glume_color_variation = 0.1f;

    // ---- Anthers ---- //

    //! Fraction of spikelets with a floret at anthesis, each hanging 3 anthers. Zero gives a tassel before or long after pollen shed. Anthesis is not modeled as it proceeds in time.
    float anther_fraction = 0.5f;
    //! Anther length in meters. About 5 mm in W23 (Wang et al. 2010, The Plant Journal); 5.2-6.1 mm in a control inbred.
    float anther_length = 0.0055f;
    //! Anther width in meters. Read from photographs.
    float anther_width = 0.0013f;
    //! Length of the filament on which an anther hangs in meters. Read from photographs; no measurement was found.
    float filament_length = 0.004f;
    //! Mean anther color. Read from photographs. Patents give values such as Munsell 2.5GY 8/8 (yellow) and 5RP 5/4 (pale purple).
    helios::RGBcolor anther_color = helios::make_RGBcolor(0.76f, 0.60f, 0.30f);
    //! Coefficient of variation of anther brightness
    float anther_color_variation = 0.1f;

    // ---- Tubes ---- //

    //! Color of the axis and branches. The library maize gives its peduncle this color at its top, blending down to the culm color at its base.
    helios::RGBcolor branch_color = helios::make_RGBcolor(0.34f, 0.42f, 0.20f);
    //! Number of radial subdivisions of the branch tubes
    uint branch_radial_subdivisions = 3;
    //! Number of radial subdivisions of the axis tube. The default matches the library maize peduncle, so the two tubes meet with the same cross-section.
    uint axis_radial_subdivisions = 10;

    //! Few-branched tassel dominated by its central spike, with erect, stiff branches
    /** Anchored on the 6.1 branches of the inbred A619 (Gallavotti et al. 2011) and the 21.5 degree branch angle of the inbred Chang7-2 (Zhu et al. 2023). Many-branched types have shorter central spikes (Silveira et al. 2021). */
    static MaizeTasselParameters fewBranched();
    //! Many-branched tassel with long, spreading branches
    /** Anchored on the 53.9 degree branch angle of the inbred PHBA6 (Zhu et al. 2023) and the upper range of inbred branch number (Wu et al. 2016). */
    static MaizeTasselParameters manyBranched();

    //! Tassel parameters set from the build parameters of a library maize plant
    /**
     * The key "tassel_type" (0 to 1) is applied first: 0 gives \ref fewBranched(), 0.5 the defaults, and 1 \ref manyBranched(), with values between blending linearly between neighbouring presets. Each other key
     * then overrides one field, and fields not given keep their value. Keys are listed in the plant library documentation (see \ref MaizeTassel).
     * \param[in] build_parameters Map of build parameter names to values, as passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
     * \return Tassel parameters with the given values applied.
     * \throws helios_runtime_error if a value is outside its valid range.
     */
    [[nodiscard]] static MaizeTasselParameters fromBuildParameters(const std::map<std::string, float> &build_parameters);

    //! Declarations of all build parameter keys read by \ref fromBuildParameters()
    [[nodiscard]] static std::vector<BuildParameterDeclaration> buildParameterDeclarations();
};

//! Build a maize tassel with resolved spikelets and anthers
/**
 * The tassel is built along +x from its attachment at the origin, in meters. Branches arch and anthers hang toward -x, so the tassel is meant to be used standing upright. Random variation between calls is drawn
 * from the Context random generator.
 * \param[in] context_ptr Pointer to the Helios Context.
 * \param[in] parameters Morphology of the tassel.
 * \return Object ID of the tassel polymesh. All of its primitives carry the object_label "fruit".
 */
uint buildMaizeTassel(helios::Context *context_ptr, const MaizeTasselParameters &parameters);

//! Parameters of a procedurally generated rice panicle
/**
 * The panicle is a rachis bearing primary branches, whose lower parts carry short secondary branches and whose upper parts carry spikelets directly. Every spikelet is its own small mesh. The form of the panicle
 * differs between its three stages (see \ref RicePanicleStage): it stands erect and green at heading, opens at anthesis, and nods under the weight of its grain as it ripens.
 */
struct RicePanicleParameters {

    // ---- Overall form ---- //

    //! Length of the panicle in meters, from the panicle base to its tip. Nipponbare 23.3 cm (SPO1 study, 2023) and IR64 25.6 cm (Xu et al. 2011, Annals of Botany 107:817).
    float length = 0.24f;
    //! Coefficient of variation of the panicle length between panicles. Branch lengths vary in proportion to it.
    float length_variation = 0.06f;
    //! Angle in degrees through which the rachis turns from its base to its tip when ripe. Zero is erect, 90 has the tip horizontal on an upright culm. The bend is concentrated toward the tip. A panicle is called curved when it bends more than 45 degrees at the neck and erect when under 30 (Yang et al. 2024, Plants 13:2035); the default is read from photographs.
    float rachis_droop = 75.f;
    //! Radius of the rachis at its base in meters. It continues the peduncle.
    float rachis_radius = 0.0011f;
    //! Radius of the rachis at its tip in meters
    float rachis_tip_radius = 0.0004f;

    // ---- Primary branches ---- //

    //! Mean number of primary branches. About 10.7 in isogenic lines (Rice 2022, doi via PMC8901826); 8-10 rachis nodes usually bear one branch each (Yoshida 1981).
    int primary_branch_count = 10;
    //! Coefficient of variation of the number of primary branches between panicles
    float primary_branch_count_variation = 0.12f;
    //! Length of the lowest primary branches in meters
    float branch_length_base = 0.115f;
    //! Length of the uppermost primary branch in meters
    float branch_length_tip = 0.045f;
    //! Coefficient of variation of primary branch length
    float branch_length_variation = 0.12f;
    //! Insertion angle in degrees of the lowest primary branch from the rachis, on a ripe panicle
    float branch_angle_base = 24.f;
    //! Insertion angle in degrees of the uppermost primary branch from the rachis, on a ripe panicle
    float branch_angle_tip = 12.f;
    //! How far a primary branch turns toward hanging when ripe, from 0 (keeps its insertion direction) to 1 (its tip hangs plumb)
    float branch_hang = 0.85f;
    //! Fraction of each primary branch, measured from its base, that bears nothing
    float branch_bare_fraction = 0.07f;
    //! Radius of a primary branch at its base in meters
    float branch_radius = 0.00035f;

    // ---- Secondary branches ---- //

    //! Mean number of secondary branches on the longest primary branches. A shorter primary branch carries proportionally fewer. Reported totals of 11-25 per panicle on about 11 primary branches (Xu et al. 2016, Breeding Science 66:659) give about 1-2 per primary branch on average.
    float secondary_branches_per_primary = 2.5f;
    //! Fraction of a primary branch, measured from its base, over which its secondary branches arise
    float secondary_zone_fraction = 0.5f;
    //! Length of a secondary branch in meters
    float secondary_branch_length = 0.017f;
    //! Number of spikelets on a secondary branch
    int spikelets_per_secondary = 3;

    // ---- Spikelets ---- //

    //! Distance along a primary branch between successive spikelets in meters. They alternate between the two sides of the branch.
    float spikelet_spacing = 0.0055f;
    //! Spikelet length in meters. 7.61 mm in IR747B2-6 (Yoshida 1981).
    float spikelet_length = 0.0076f;
    //! Spikelet width in meters. 2.46-3.48 mm across the varieties tabulated by Yoshida (1981).
    float spikelet_width = 0.0028f;
    //! Spikelet thickness in meters. 1.81-2.19 mm across the varieties tabulated by Yoshida (1981).
    float spikelet_thickness = 0.002f;
    //! Coefficient of variation of spikelet size
    float spikelet_size_variation = 0.05f;
    //! Angle in degrees between a spikelet and the branch bearing it
    float spikelet_angle = 12.f;
    //! Length of the pedicel in meters
    float pedicel_length = 0.0015f;
    //! Number of facets around a spikelet. Each spikelet costs 6 x spikelet_subdivisions + 2 triangles.
    uint spikelet_subdivisions = 6;
    //! Ripeness of the panicle from 0 (green hulls) to 1 (fully ripe). The top of a panicle ripens first, so a value below 1 leaves its base greener.
    float ripeness = 0.85f;
    //! Hull color of an unripe spikelet
    helios::RGBcolor hull_green_color = helios::make_RGBcolor(0.42f, 0.54f, 0.22f);
    //! Hull color of a ripe spikelet
    helios::RGBcolor hull_ripe_color = helios::make_RGBcolor(0.72f, 0.60f, 0.36f);
    //! Coefficient of variation of hull brightness between spikelets
    float hull_color_variation = 0.08f;

    // ---- Anthers ---- //

    //! Fraction of spikelets open on a panicle at anthesis, each with its six anthers hanging from it. Each lasts only a few hours, and a panicle opens from its top downward over 7-10 days (Yoshida 1981).
    float anther_fraction = 0.3f;
    //! Anther length in meters. 1.96 and 2.53 mm in two parental lines (PMC11053959).
    float anther_length = 0.0022f;
    //! Anther width in meters
    float anther_width = 0.0005f;
    //! Length of the filament on which an anther hangs in meters
    float filament_length = 0.004f;
    //! Anther color
    helios::RGBcolor anther_color = helios::make_RGBcolor(0.85f, 0.78f, 0.42f);
    //! Coefficient of variation of anther brightness
    float anther_color_variation = 0.08f;

    // ---- Tubes ---- //

    //! Color of the rachis and branches. The library rice gives its peduncle this color at its top, blending down to the culm color at its base.
    helios::RGBcolor branch_color = helios::make_RGBcolor(0.42f, 0.50f, 0.22f);
    //! Number of radial subdivisions of the branch tubes
    uint branch_radial_subdivisions = 3;
    //! Number of radial subdivisions of the rachis tube. The default matches the library rice peduncle, so the two tubes meet with the same cross-section.
    uint rachis_radial_subdivisions = 8;

    //! Short, dense panicle that stays erect as it ripens
    static RicePanicleParameters erect();
    //! Long, open panicle whose branches hang freely when ripe
    static RicePanicleParameters lax();

    //! Panicle parameters set from the build parameters of a library rice plant
    /**
     * The key "panicle_type" (0 to 1) is applied first: 0 gives \ref erect(), 0.5 the defaults, and 1 \ref lax(), with values between blending linearly between neighbouring presets. Each further key then
     * overrides one field, and fields not given keep their value. Keys are listed in the plant library documentation (see \ref RicePanicle).
     * \param[in] build_parameters Map of build parameter names to values, as passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
     * \return Panicle parameters with the given values applied.
     * \throws helios_runtime_error if a value is outside its valid range.
     */
    [[nodiscard]] static RicePanicleParameters fromBuildParameters(const std::map<std::string, float> &build_parameters);

    //! Declarations of all build parameter keys read by \ref fromBuildParameters()
    [[nodiscard]] static std::vector<BuildParameterDeclaration> buildParameterDeclarations();
};

//! The stages in which a rice panicle differs in form and color
enum class RicePanicleStage {
    //! Just emerged from the flag leaf sheath: erect, green, with branches held close to the rachis and no anthers
    heading,
    //! Upright and green with branches spread, and anthers hanging from the open spikelets
    anthesis,
    //! Nodding under the weight of its grain, with its hulls colored by \ref RicePanicleParameters::ripeness
    ripening
};

//! Build a rice panicle with resolved spikelets
/**
 * The panicle is built along +x from its attachment at the origin, in meters. Its rachis bends toward -z, which the plant model keeps on the side the culm leans toward, and its branches and anthers hang toward -x,
 * so the panicle is meant to be used on a near-upright culm. Random variation between calls is drawn from the Context random generator.
 * \param[in] context_ptr Pointer to the Helios Context.
 * \param[in] parameters Morphology of the panicle.
 * \param[in] stage Stage of the panicle.
 * \return Object ID of the panicle polymesh. All of its primitives carry the object_label "fruit".
 */
uint buildRicePanicle(helios::Context *context_ptr, const RicePanicleParameters &parameters, RicePanicleStage stage);

//! Build parameter "tiller_count" of the library sorghum
inline constexpr BuildParameterDeclaration sorghum_tiller_count{"tiller_count", 0.f, 6.f, true, "number of tillers of a sorghum plant"};

//! Build parameter "tiller_count" of the library wheat
inline constexpr BuildParameterDeclaration wheat_tiller_count{"tiller_count", 0.f, 4.f, true, "number of tillers of a wheat plant"};

//! Build parameter "tiller_count" of the library rice
inline constexpr BuildParameterDeclaration rice_tiller_count{"tiller_count", 0.f, 18.f, true, "number of tillers of a rice plant"};

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

//! Number of tillers of a library rice plant, read from its build parameter "tiller_count" (default 12, range 0-18)
/**
 * \param[in] build_parameters Map of build parameter names to values, as passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
 * \return Number of primary and secondary tillers.
 */
[[nodiscard]] int riceTillerCount(const std::map<std::string, float> &build_parameters);

//! Fruit-branch contact geometry and push-out solver, shared between PlantArchitecture::resolveFruitBranchCollisions() and its self-tests (see FruitCollision.cpp)
namespace fruit_contact {

    //! One internode segment, a capsule whose radius varies linearly along it
    struct BranchSegment {
        helios::vec3 start;
        helios::vec3 end;
        float start_radius = 0;
        float end_radius = 0;
    };

    //! World-space copy of one fruit's contact proxy (see FruitContactProxy), moved while a push-out is worked out
    struct WorldProxy {
        helios::vec3 bounding_center;
        float bounding_radius = 0;
        std::vector<helios::vec3> sphere_centers;
        std::vector<float> sphere_radii;
        //! Three vertices per triangle; empty when the fruit is tested through its spheres only
        std::vector<helios::vec3> triangle_vertices;
    };

    //! A contact between a fruit and a branch
    struct Contact {
        //! Depth of the contact; positive when a fruit is inside a branch, or within the clearance of one
        float penetration = 0;
        //! Point of the fruit proxy in contact, which is where a hinge's lever acts
        helios::vec3 point;
        //! Unit direction in which that point must move to clear the contact
        helios::vec3 direction;
        //! Fruit of the bud the contact belongs to
        size_t fruit_index = 0;
    };

    //! Deepest contact between the fruit of a bud and a set of branch segments; its penetration is zero when there is none
    [[nodiscard]] Contact deepestContact(const std::vector<WorldProxy> &fruit, const std::vector<BranchSegment> &segments, float clearance);

    //! How the push-out may turn the fruit: whether a rotation is within its budget, and applying it
    /** apply_swing turns every fruit about the bud's node (the pivot) and must update the fruit proxies and attachment points it was given; apply_tilt turns one fruit about its attachment point. */
    struct HingeControls {
        std::function<bool(const helios::vec3 &axis, float angle)> swing_allowed;
        std::function<void(const helios::vec3 &axis, float angle)> apply_swing;
        std::function<bool(size_t fruit_index, const helios::vec3 &axis, float angle)> tilt_allowed;
        std::function<void(size_t fruit_index, const helios::vec3 &axis, float angle)> apply_tilt;
    };

    //! Move the fruit of a bud out of the branches, a few degrees at a time, at whichever hinge clears a contact with the smaller rotation
    /**
     * \param[in] fruit Fruit proxies; read here and moved by the callbacks
     * \param[in] fruit_bases Fruit attachment points, the tilt pivots; read here and moved by apply_swing
     * \param[in] pivot The bud's node, the swing pivot
     * \param[in] segments Branch segments to clear
     * \param[in] clearance Gap (meters) to leave between fruit and branch surfaces
     * \param[in] controls Budget checks and the rotations themselves
     * \return True if the fruit end clear of every segment; false if the bud is left wedged at its least-penetrating pose
     */
    [[nodiscard]] bool pushOut(const std::vector<WorldProxy> &fruit, const std::vector<helios::vec3> &fruit_bases, const helios::vec3 &pivot, const std::vector<BranchSegment> &segments, float clearance, const HingeControls &controls);

} // namespace fruit_contact

#endif // HELIOS_PLANT_ARCHITECTURE_INTERNAL_H
