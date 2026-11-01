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
    //! Add the joint armature computed by PlantArchitecture::getPlantArticulation(). It is needed for integrators that apply joint springs explicitly.
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

#endif
