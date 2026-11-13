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
 * \param[in] gravity Direction gravity acts in, expressed in the lattice's own frame; it need not be a unit vector but must be finite and not zero. The default, the frame's -z, treats "down" as
 * perpendicular to the direction the blade leaves its base, which is the behaviour every species was calibrated with. Passing the true direction of gravity lets a blade that leaves its base steeply - a
 * grass leaf emerging from the whorl - arch over past horizontal and hang, which the default cannot produce: it stops the bend once the blade points along the frame's -z, 90 degrees from its base
 * direction. The blade bends only in its own plane, so the component of gravity across the blade does not bend it, and each station stops turning once it points along gravity.
 * \return Deflected lattice vertices, parallel to \p rest_vertices.
 */
std::vector<helios::vec3> deformLeafLattice(const std::vector<helios::vec3> &rest_vertices, uint Nx, uint Ny, float scale, float mature_scale, float flexibility, float taper = 1.f,
                                            const helios::vec3 &gravity = helios::make_vec3(0.f, 0.f, -1.f));

//! Build the vertex lattice of a generic leaf blade in the prototype's own frame
/**
 * This is the geometry \ref GenericLeafPrototype() builds, and the leaf reshaping setters rebuild a leaf from it. The blade runs along +x from its base, at unit length, with its upper (adaxial) side toward
 * +z. The shape terms are composed in this order:
 *  1. the cross-section: the midrib fold, the lateral curvature and the roll together give the tangent angle of the cross-section as a function of arc length from the midrib, and each lattice cell is laid
 *     along that curve at its own width, so the cross-section bends without stretching;
 *  2. the petiole roll near the blade base;
 *  3. the longitudinal curvature, which bends the blade along its length;
 *  4. the waves, displaced along the bent blade's normal;
 *  5. the translation by \ref LeafShapeValues::lattice_offset.
 * The deflection under the blade's own weight is applied later, to the leaf in the canopy (see \ref deformLeafLattice()).
 * \param[in] shape Sampled blade shape and lattice dimensions.
 * \return Lattice vertices, row-major over (subdivisions_y + 1) rows across the blade by (subdivisions_x + 1) along it, with row subdivisions_y / 2 on the midrib.
 */
std::vector<helios::vec3> buildLeafBladeLattice(const LeafShapeValues &shape);

//! Number of vertices a petiolule adds to a leaf prototype built by the generic leaf prototype function
/**
 * The petiolule's vertices are appended after the blade lattice in the prototype's mesh, so this is what separates the two: the lattice is every vertex but the last petioluleVertexCount().
 * \return Number of petiolule vertices, which does not depend on the petiolule's length or radius.
 */
uint petioluleVertexCount();

//! Number of vertices in each ring of a leaf's petiolule
/**
 * The petiolule's vertices are appended ring by ring from the leaflet's base: the ring at its base, the ring where the blade begins, the rings of the join under the blade, and the join's tip.
 */
uint petioluleRadialSubdivisions();

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

//! Build a leaf blade procedurally from the parameters of a leaf prototype
/**
 * The default LeafPrototype::prototype_function. Builds the blade lattice with buildLeafBladeLattice(), textured and with fully transparent facets dropped, and a petiolule if the prototype asks for one.
 * \param[in] context_ptr Pointer to the Helios context.
 * \param[in] prototype_parameters Leaf prototype parameters; the shape it built is recorded on it, and its random shape parameters are resampled.
 * \param[in] compound_leaf_index Index of the leaflet within the compound leaf (0 for the terminal leaflet).
 * \return Object ID of the leaf.
 */
uint GenericLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);

uint GeneralSphericalFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);

uint AlmondFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint AlmondFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void AlmondPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void AlmondPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);
//! Phytomer callback for an almond model that defines a "spur" shoot type: grows spurs from the lateral buds that did not grow a long shoot.
void AlmondSpurPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

//! Apple fruit prototype. Its primitives carry, as defaults for the radiation model, the primitive data "reflectivity_spectrum" (apple_fruit_reflectivity_0000), "specular_roughness" (0.15) and "specular_n" (1.5).
uint AppleFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint AppleFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void ApplePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void ApplePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);
//! Phytomer creation function of the "apple_fruitingwall" model: the apple rules, with the basal node of every lateral blind.
void AppleFruitingWallPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
//! Phytomer callback of the "apple_fruitingwall" model: the apple rules, and the tall-spindle training of the wall, applied once per time step from the leader.
void AppleFruitingWallPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

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

//! Grape cluster fruit prototype, built in meters so the fruit_prototype_scale that goes with it is 1
/**
 * The cluster is set by the plant's build parameters "berry_diameter" and "cluster_berry_count" (see \ref PlantArchBuildParams). Its size follows from its berries, growing in proportion to berry diameter and with the cube root of berry count.
 */
uint GrapevineFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
// uint GrapevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void GrapevinePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void GrapevinePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

//! Maize tassel fruit prototype with resolved spikelets and anthers, shaped by the plant's tassel build parameters (see \ref MaizeTassel)
/**
 * The geometry is in meters, so the fruit_prototype_scale that goes with it is 1. The subdivisions argument is not used.
 */
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

uint RicePaniclePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint RiceFloweringPaniclePrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void RicePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

//! Sorghum panicle fruit prototype with resolved grains, shaped by the plant's panicle build parameters (see \ref PlantArchBuildParams)
/**
 * The geometry is in meters, so the fruit_prototype_scale that goes with it is 1. The subdivisions argument is not used.
 */
uint SorghumPaniclePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);

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


uint CucumberFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters);
uint CucumberFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters);
void CucumberMainPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

//! Attraction points along the vertical string a high-wire cucumber is trained up, from just above string_base to the top of the string.
std::vector<helios::vec3> CucumberStringAttractionPoints(const helios::vec3 &string_base);

//! Trains and harvests a high-wire cucumber: picks each fruit once it is full size, and once the head reaches the wire, lowers and leans the stem, laying the older stem along the row (+y) just above the floor so that the head stays at the wire.
void CucumberMainPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);
#endif // HELIOS_ASSETS_H
