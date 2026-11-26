/** \file "PlantPhysics.h" Adds plants from the plant architecture plug-in to the physics engine. Only available when plantarchitecture is in the project's list of plug-ins.

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

#ifndef HELIOS_PLANT_PHYSICS_H
#define HELIOS_PLANT_PHYSICS_H

#include "PhysicsEngine.h"
#include "PlantArchitecture.h"

//! Settings of how a plant is converted to an articulation
struct PlantPhysicsParameters {
    //! Add the joint armature computed by PlantArchitecture::getPlantArticulation(). It is needed for integrators that apply joint springs explicitly. With the discrete integrator the simulation is stable without it, but the iterative constraint solvers (PGS and CG) then need several times more iterations, because stiff joints on very light links (such as the end of a fruit stem, which carries only the fruit held to it) leave their problem badly conditioned; Newton is unaffected.
    bool apply_joint_armature = false;
    //! Maximum bending angle of each joint in degrees. Zero leaves the joints unlimited.
    float max_joint_bend_degrees = 60.f;
    //! Give leaves a thin box collision shape fitted to the leaf blade. When false, leaves carry mass but nothing collides with them.
    bool leaf_collisions = true;
    //! Thickness in meters of the leaf collision shape
    float leaf_collision_thickness = 0.002f;
    //! Collision shape of each leaf blade: "box" or "ellipsoid" fitted to the blade, or "triangles" for a thin prism on every triangle of the blade. The ellipsoid follows the outline of a rounded blade more closely than the box, whose corners extend past it; only "triangles" follows a folded or curved blade, at a higher cost.
    std::string leaf_collision_shape = "box";
    //! Weld the internode segments of the plant to each other, so that only petioles, peduncles, leaves, fruit, and flowers move. This makes large plants several times faster to simulate.
    bool rigid_internodes = false;
    //! Weld petioles and leaves to what they grow from, so that they follow the wood without moving themselves. Leaves still add weight to the wood, but a plant in leaf can be simulated about as fast as one without leaves, for when they matter to what a camera sees rather than to the mechanics.
    bool rigid_leaves = false;
    //! Indices of internode links in PlantArchitecture::getPlantArticulation() that stay flexible when rigid_internodes is enabled. Making only the wood near where the plant is touched flexible keeps most of the speed of rigid internodes.
    std::vector<uint> flexible_links;
    //! Indices of the links in PlantArchitecture::getPlantArticulation() that can later be released with PhysicsEngine::detachLink(), such as branches that may be cut or fruit that may be picked. Each one adds six degrees of freedom and a constraint to the model, and is rigidly held to its parent until released.
    std::vector<uint> detachable_links;
    //! Indices of links whose joint stiffness is multiplied by the matching entry of stiffness_scales, together with their joint damping and armature, which are proportional to it. Each must be a link that keeps its joint (not welded to its parent).
    /**
     * With rigid_internodes, a flexible internode carries the welded internodes beyond it as one rigid piece, so its joint alone stands for the bending of all of them. Its stiffness from getPlantArticulation() is that of the one internode; the
     * springs of the welded internodes in series are softer. Giving a flexible link that carries n welded internodes the scale 1 / (1 + n), as for n + 1 equal springs in series, keeps the branch about as compliant as when every internode
     * bends, with fewer joints. Summing the welded joints' own compliances (1 / K_series = sum of 1 / K) over-softens it: the thin internodes nearest a load, which bend little under it, then dominate.
     */
    std::vector<uint> stiffness_scaled_links;
    //! Positive factor for each link in stiffness_scaled_links
    std::vector<float> stiffness_scales;
    //! Modulus of rupture of the wood in Pa, past which an internode's joint breaks: each internode with a joint is given the rupture moment of a solid circular section, modulus_of_rupture * pi * r^3 / 4
    //! (PhysicsLink::rupture_moment). Green apple wood is about 45-60 MPa. Zero, the default, makes wood that does not break. Only internodes that keep their joint can break; with rigid_internodes, a thin welded internode
    //! carried by a thicker flexible one cannot.
    float modulus_of_rupture = 0.f;
};

//! Add a plant to a physics simulation, and make its Context geometry follow the simulated links. Must be called before PhysicsEngine::initialize().
/**
 * Every internode, petiole, and peduncle segment becomes a capsule link and every leaf, fruit, and flower a mass link, each connected to its parent by an elastic ball joint.
 * Link indices in the physics engine are the same as those of PlantArchitecture::getPlantArticulation().
 * \param[in] engine Physics simulation to add the plant to.
 * \param[in] context Context holding the plant geometry.
 * \param[in] plantarchitecture PlantArchitecture model that built the plant.
 * \param[in] plantID ID of the PlantArchitecture plant instance.
 * \param[in] plant_physics Mechanical properties used to build the plant articulation.
 * \param[in] params Settings of how the plant is converted to an articulation.
 * \return ID of the articulation in the physics engine.
 */
uint addPlantToPhysicsEngine(PhysicsEngine &engine, const helios::Context &context, const PlantArchitecture &plantarchitecture, uint plantID, const USDExportParameters &plant_physics = USDExportParameters(),
                             const PlantPhysicsParameters &params = PlantPhysicsParameters());

//! Get the label that addPlantToPhysicsEngine() gives the links of an organ type (PhysicsLink::label), by which ContactSensor reports contact with them
/**
 * \param[in] type Organ type of a plant articulation link.
 * \return "wood" for internodes, and "petiole", "peduncle", "leaf", "fruit", or "flower" for the others.
 */
std::string plantLinkLabel(PlantArticulationLinkType type);

//! Find the links of a plant of one organ type whose rest position is within a box, such as the fruit a robot can reach
/**
 * \param[in] articulation Articulation of the plant from PlantArchitecture::getPlantArticulation().
 * \param[in] type Organ type of the links to find.
 * \param[in] box_min Lowest corner of the box. Use minus infinity along an axis that is not bounded below.
 * \param[in] box_max Highest corner of the box. Use plus infinity along an axis that is not bounded above.
 * \return Indices of the links in the articulation whose rest position is in the box, its faces included, in increasing order.
 */
std::vector<uint> findPlantLinksInBox(const PlantArticulation &articulation, PlantArticulationLinkType type, const helios::vec3 &box_min, const helios::vec3 &box_max);

//! Find the internode that bears a link: its nearest ancestor that is an internode
/**
 * \param[in] articulation Articulation of the plant from PlantArchitecture::getPlantArticulation().
 * \param[in] link_index Index of the link.
 * \return Index of the internode. An error is raised if the link has no internode among its ancestors.
 */
uint findBearingInternode(const PlantArticulation &articulation, uint link_index);

//! Find the internodes that carry a link, nearest first, such as the wood to leave flexible behind a fruit (PlantPhysicsParameters::flexible_links)
/**
 * The internode at the root, which is fixed to the world, is never included.
 * \param[in] articulation Articulation of the plant from PlantArchitecture::getPlantArticulation().
 * \param[in] link_index Index of the link.
 * \param[in] count Most internodes to return.
 * \return Indices of up to count internodes among the link's ancestors, nearest the link first.
 */
std::vector<uint> findSupportingInternodes(const PlantArticulation &articulation, uint link_index, uint count);

//! Where the nearest organ of a plant is from a point, found by NearestPlantOrgan
struct NearestPlantOrganResult {
    //! Index of the nearest link in PlantArchitecture::getPlantArticulation()
    uint link_index = 0;
    //! From the point to the nearest point on the link's axis, in meters. Its length less the link's radius is the distance to the link's surface.
    helios::vec3 axis_offset;
    //! From the point to the link's surface in meters, negative when the point is inside the link
    float surface_distance = 0.f;
};

//! Finds the organ of a plant nearest to a point, among the organs of one type, with the plant as it stands at rest
/**
 * The organs are where they stand in the rest pose of the articulation, not where the simulation has moved them. This stands in for a map of the plant that perception would have built before a robot starts work.
 * Only organs with a capsule shape can be searched: internodes, petioles, and peduncles.
 */
class NearestPlantOrgan {
public:
    //! Constructor
    /**
     * \param[in] articulation Articulation of the plant from PlantArchitecture::getPlantArticulation().
     * \param[in] type Organ type to search: ARTICULATION_LINK_INTERNODE, ARTICULATION_LINK_PETIOLE, or ARTICULATION_LINK_PEDUNCLE.
     */
    NearestPlantOrgan(const PlantArticulation &articulation, PlantArticulationLinkType type);

    //! Find the organ whose surface is nearest to a point
    /**
     * \param[in] point Point in world coordinates.
     */
    [[nodiscard]] NearestPlantOrganResult find(const helios::vec3 &point) const;

    //! Find the organ whose surface is nearest to a point, with the organs where they are now in a simulation
    /**
     * The search of the rest shape is faster but does not follow wood that bends; this one moves every organ with its link.
     * \param[in] point Point in world coordinates.
     * \param[in] engine Physics simulation the plant was added to with addPlantToPhysicsEngine().
     * \param[in] articulationID ID of the plant's articulation, returned by addPlantToPhysicsEngine().
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] NearestPlantOrganResult find(const helios::vec3 &point, const PhysicsEngine &engine, uint articulationID, uint instance = 0) const;

private:
    struct Segment {
        uint link_index;
        helios::vec3 start;
        helios::vec3 end;
        float radius;
    };
    std::vector<Segment> segments;
    size_t link_count = 0;

    [[nodiscard]] static NearestPlantOrganResult nearestOnSegment(const helios::vec3 &point, const helios::vec3 &start, const helios::vec3 &end, float radius, uint link_index);
};

#endif
