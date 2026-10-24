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

#ifndef HELIOS_ASSETS_H
#define HELIOS_ASSETS_H

/** \file "Assets.h" Declarations of the organ prototype and phytomer functions used by the plant architecture library models. */

//! Deflect a leaf blade lattice under its own weight
/**
 * Treats the blade as a cantilever clamped at its base and loaded by its own distributed weight. The bending moment at each station along the midrib is the weight of everything distal to it acting through
 * its horizontal lever arm; curvature is proportional to that moment and is integrated along the arc to give the deflected centreline, which the rest of the lattice is then carried along with.
 *
 * The deflection is computed from the undeformed rest lattice every time rather than accumulated onto the previous result, so a leaf that stops growing stops drooping instead of creeping downward.
 * Because the moment grows with the cube of blade length while the stiffness does not, a leaf that doubles in length droops roughly eight times as far - growth-driven droop falls out of the mechanics
 * and needs no separate age term.
 * \param[in] rest_vertices Undeformed lattice vertices, row-major over (Nx+1) x (Ny+1) with the midrib running along the local x-axis.
 * \param[in] Nx Number of blade subdivisions along the leaf length.
 * \param[in] Ny Number of blade subdivisions across the leaf width.
 * \param[in] scale Current length scale of the leaf, as applied to the unit-length rest lattice.
 * \param[in] mature_scale Length scale the leaf reaches when fully grown. This is the reference length the dimensionless flexibility is measured against, so it must be the leaf's final size rather than its
 * current one - normalizing by the current size would divide out the growth-driven droop entirely.
 * \param[in] flexibility Dimensionless droopiness of the fully-grown leaf. Zero returns the rest lattice unchanged, which is the path taken by every species that does not droop. Values around 20 give a
 * mature grass blade that arcs over to roughly a third of its length, and the same value means the same shape regardless of how long the leaf actually is.
 * \param[in] taper How much more compliant the blade is at its tip than at its base, as a ratio of the two. A blade of uniform stiffness bends hardest where the bending moment is largest, which is at the
 * clamped base, and is left nearly straight over its outer half; a real grass leaf does the opposite, staying straight near the base and curving hardest toward the tip. The difference is the midrib, whose
 * thickness falls several-fold from base to tip and takes the blade's bending stiffness down with it faster than the moment itself decays. One leaves the stiffness uniform along the blade.
 * \return Deflected lattice vertices, parallel to \p rest_vertices.
 */
std::vector<helios::vec3> deformLeafLattice(const std::vector<helios::vec3> &rest_vertices, uint Nx, uint Ny, float scale, float mature_scale, float flexibility, float taper = 1.f);

//! Number of vertices a petiolule adds to a leaf prototype built by the generic leaf prototype function
/**
 * The petiolule's vertices are appended after the blade lattice in the prototype's mesh, so this is what separates the two: the lattice is every vertex but the last petioluleVertexCount().
 * \return Number of petiolule vertices, which does not depend on the petiolule's length or radius.
 */
uint petioluleVertexCount();

//! Bend a petiole centerline as a tapered cantilever loaded by the leaflets attached along it
/**
 * The centerline is clamped at its first node. Each load acts at its arclength along the centerline, and the rotation of segment j is
 *
 *     theta_j = 2 * compliance / (reference_load * L^2) * (r_base / r_j)^4 * integral over segment j of M(s) ds,
 *
 * where L is the total centerline length, r_j the mean radius of the segment (with r_j / r_base floored at 0.5), and M(s) the bending moment at s: the sum over loads beyond s of the load times its horizontal
 * lever arm along the segment's heading, clamped to be non-negative. The cumulative rotation of the segments before j - the curvature accumulated up to segment j's own base - lowers segment j's elevation
 * within its own vertical plane, so its horizontal heading is kept and it never passes hanging straight down. The first segment is therefore left at its rest direction, as a cantilever built in at its
 * support has no slope change there: the centerline leaves its base at the angle it was given and the bend appears beyond that as curvature accumulating along the length. The lever arms are those of the
 * bent shape, found by an under-relaxed fixed-point iteration from the rest shape. With this normalization a straight, horizontal, untapered centerline whose whole reference load acts at its tip turns
 * through \p compliance radians from base to tip in the small-deflection limit, independently of its length; the last segment's own share of that turn falls beyond it, so its direction falls short of the
 * total by a fraction equal to the square of one segment's share of the length.
 * \param[in] rest_offsets Undeformed centerline nodes, as offsets from the base; at least two, with no coincident neighbours.
 * \param[in] radii Radius at each node, parallel to \p rest_offsets. Only ratios to the first radius are used; a non-positive first radius leaves the stiffness uniform.
 * \param[in] load_arclength_fractions Position of each load as a fraction of the centerline's arclength, in [0, 1].
 * \param[in] load_weights Weight of each load, in the same arbitrary units as \p reference_load; non-negative.
 * \param[in] reference_load Load that \p compliance is normalized against (the petiole's full-grown leaflet load). A non-positive value returns the rest shape.
 * \param[in] compliance Dimensionless bending compliance; non-negative. Zero returns the rest shape.
 * \param[out] equilibrium_residual If not null, receives the largest difference in radians between a segment's elevation and the elevation the moments of the returned shape call for; zero when the rest shape is returned.
 * \return Bent centerline nodes, as offsets from the base, parallel to \p rest_offsets.
 */
std::vector<helios::vec3> bendPetioleCenterline(const std::vector<helios::vec3> &rest_offsets, const std::vector<float> &radii, const std::vector<float> &load_arclength_fractions, const std::vector<float> &load_weights,
                                                float reference_load, float compliance, float *equilibrium_residual = nullptr);

uint GenericLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);

uint GeneralSphericalFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);

uint AlmondFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint AlmondFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void AlmondPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void AlmondPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);
//! Phytomer callback for an almond model that defines a "spur" shoot type: grows spurs from the lateral buds that did not grow a long shoot.
void AlmondSpurPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint AppleFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint AppleFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void ApplePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void ApplePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint AsparagusLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
void AsparagusPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint BeanLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint BeanFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint BeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void BeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint BindweedFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);

uint BougainvilleaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);

uint CapsicumFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
void CapsicumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint CheeseweedLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint CowpeaFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint CowpeaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void CowpeaPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

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
    //! Names of all build parameter keys read by \ref fromBuildParameters()
    [[nodiscard]] static std::vector<std::string> buildParameterNames();
};

//! Grape cluster fruit prototype, built in meters so the fruit_prototype_scale that goes with it is 1
/**
 * The cluster is set by \ref GrapeClusterParameters read from the plant's build parameters. Its size follows from its berries, growing in proportion to berry diameter and with the cube root of berry count.
 */
uint GrapevineFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
// uint GrapevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void GrapevinePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void GrapevinePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint MaizeTasselPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint MaizeEarPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
void MaizePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint OliveLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint OliveFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint OliveFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void OlivePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void OlivePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint PistachioFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint PistachioFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void PistachioPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void PistachioPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint PuncturevineFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);

uint RedbudFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
uint RedbudFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
void RedbudPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void RedbudPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint RiceSpikePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
void RicePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

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

    //! Names of all build parameter keys read by \ref fromBuildParameters()
    [[nodiscard]] static std::vector<std::string> buildParameterNames();
};

//! Build a sorghum panicle with resolved grains
/**
 * The panicle is built along +x from its attachment at the origin, in meters. Random variation between calls is drawn from the Context random generator.
 * \param[in] context_ptr Pointer to the Helios Context.
 * \param[in] parameters Morphology of the panicle.
 * \return Object ID of the panicle polymesh. All of its primitives carry the object_label "fruit".
 */
uint buildSorghumPanicle(helios::Context *context_ptr, const SorghumPanicleParameters &parameters);

//! Sorghum panicle fruit prototype built from \ref SorghumPanicleParameters set by the plant's build parameters (see \ref SorghumPanicleParameters::fromBuildParameters())
/**
 * The geometry is in meters, so the fruit_prototype_scale that goes with it is 1. The subdivisions argument is not used; tessellation is set by \ref SorghumPanicleParameters.
 */
uint SorghumPaniclePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);

//! Number of tillers of a library sorghum plant, read from its build parameter "tiller_count" (default 0, range 0-6)
/**
 * \param[in] build_parameters Map of build parameter names to values, as passed to \ref PlantArchitecture::buildPlantInstanceFromLibrary().
 * \return Number of main-stem leaf axils that produce a tiller.
 */
[[nodiscard]] int sorghumTillerCount(const std::map<std::string, float> &build_parameters);

void SorghumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint SoybeanFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint SoybeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void SoybeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint StrawberryFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint StrawberryFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void StrawberryPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint TomatoFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint TomatoFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void TomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

void CherryTomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void CherryTomatoPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint WalnutFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint WalnutFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void WalnutPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void WalnutPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint WheatSpikePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
void WheatPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);


#endif // HELIOS_ASSETS_H
