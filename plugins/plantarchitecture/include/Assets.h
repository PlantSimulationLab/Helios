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

uint GeneralSphericalFruitPrototype(helios::Context *context_ptr, uint subdivisions);

uint AlmondFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint AlmondFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void AlmondPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void AlmondPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint AppleFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint AppleFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void ApplePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void ApplePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint AsparagusLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
void AsparagusPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint BeanLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint BeanFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint BeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void BeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint BindweedFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);

uint BougainvilleaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);

uint CapsicumFruitPrototype(helios::Context *context_ptr, uint subdivisions);
void CapsicumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint CheeseweedLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint CowpeaFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint CowpeaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void CowpeaPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint GrapevineFruitPrototype(helios::Context *context_ptr, uint subdivisions);
// uint GrapevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void GrapevinePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
// void GrapevinePhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint MaizeTasselPrototype(helios::Context *context_ptr, uint subdivisions);
uint MaizeEarPrototype(helios::Context *context_ptr, uint subdivisions);
void MaizePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint OliveLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint OliveFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint OliveFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void OlivePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void OlivePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint PistachioFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint PistachioFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void PistachioPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void PistachioPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint PuncturevineFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);

uint RedbudFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
uint RedbudFruitPrototype(helios::Context *context_ptr, uint subdivisions);
void RedbudPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void RedbudPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint RiceSpikePrototype(helios::Context *context_ptr, uint subdivisions);
void RicePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint SorghumPaniclePrototype(helios::Context *context_ptr, uint subdivisions);
void SorghumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint SoybeanFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint SoybeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void SoybeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint StrawberryFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint StrawberryFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void StrawberryPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint TomatoFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint TomatoFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void TomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

void CherryTomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void CherryTomatoPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint WalnutFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint WalnutFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void WalnutPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void WalnutPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint WheatSpikePrototype(helios::Context *context_ptr, uint subdivisions);
void WheatPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);


#endif // HELIOS_ASSETS_H
