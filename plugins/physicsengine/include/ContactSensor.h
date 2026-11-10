/** \file "ContactSensor.h" Contact force on part of a simulation, split by what it touches.

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

#ifndef HELIOS_CONTACT_SENSOR_H
#define HELIOS_CONTACT_SENSOR_H

#include "PhysicsEngine.h"

//! Measures the contact force on a body of a loaded model and the bodies below it, split by the partner each contact is with
/**
 * The partners are named:
 * - the label of each articulation link (PhysicsLink::label), such as "wood", "fruit", or "leaf" for a plant added with addPlantToPhysicsEngine(), or "articulation" for links without a label
 * - "target" for one link chosen per instance with setTargetLink(), such as the fruit a robot is picking, whose contacts are then not counted under its label
 * - "ground" for the ground plane
 * - "other" for every other body, such as kinematic boxes and other loaded models
 *
 * Contacts between two sensed bodies are not counted. Forces are those on the sensed bodies, in the world frame, among the contacts of the last step.
 * Create the sensor after PhysicsEngine::initialize(). Different instances can be read and changed at the same time from different threads.
 */
class ContactSensor {
public:
    //! Constructor
    /**
     * \param[in] engine Initialized physics simulation.
     * \param[in] body_name Name of a body of a loaded model, including the prefix given to PhysicsEngine::addModelFile(). It and every body below it are sensed: pass the model's root body to sense a whole robot.
     */
    ContactSensor(const PhysicsEngine *engine, const std::string &body_name);

    //! Allocate per-instance state for every simulation instance that exists now. Call it after creating instances, before using them from several threads.
    void prepareInstances();

    //! Get the number of simulation instances the sensor has been prepared for, by its constructor or prepareInstances()
    [[nodiscard]] uint getPreparedInstanceCount() const;

    //! Report contact with one articulation link as the partner "target", rather than under the link's label
    /**
     * \param[in] articulationID ID of the articulation returned by PhysicsEngine::addArticulation().
     * \param[in] link_index Index of the link within the articulation.
     * \param[in] instance ID of the simulation instance.
     */
    void setTargetLink(uint articulationID, uint link_index, uint instance = 0);

    //! Stop reporting any link as the partner "target"
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void clearTargetLink(uint instance = 0);

    //! Get the names of the partners, in the order of getForces()
    [[nodiscard]] const std::vector<std::string> &getPartnerNames() const;

    //! Get the contact force on the sensed bodies from each partner, in the order of getPartnerNames()
    /**
     * \param[in] instance ID of the simulation instance.
     * \return Force in N from each partner, in the world frame.
     */
    [[nodiscard]] std::vector<helios::vec3> getForces(uint instance = 0) const;

    //! Get the contact force on the sensed bodies from one partner
    /**
     * \param[in] partner Name of the partner, one of getPartnerNames().
     * \param[in] instance ID of the simulation instance.
     * \return Force in N, in the world frame.
     */
    [[nodiscard]] helios::vec3 getForce(const std::string &partner, uint instance = 0) const;

    //! Get the total contact force on the sensed bodies from every partner except those excluded
    /**
     * \param[in] excluded_partners Names of partners whose contacts are left out.
     * \param[in] instance ID of the simulation instance.
     * \return Force in N, in the world frame.
     */
    [[nodiscard]] helios::vec3 getTotalForce(const std::vector<std::string> &excluded_partners = {}, uint instance = 0) const;

    //! Check whether a sensed body penetrates any partner other than those excluded, among the contacts MuJoCo last found
    /**
     * Useful for checking that a pose placed directly touches nothing, since Robot::setJointPositions() finds the contacts of the pose it places.
     * \param[in] excluded_partners Names of partners whose contacts are left out.
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] bool isPenetrating(const std::vector<std::string> &excluded_partners = {}, uint instance = 0) const;

private:
    [[nodiscard]] size_t findPartner(const std::string &partner, const std::string &method) const;
    [[nodiscard]] std::vector<char> excludedPartnerMask(const std::vector<std::string> &excluded_partners, const std::string &method) const;
    [[nodiscard]] int partnerOfBody(int body_ID, uint instance) const;
    void requireInstance(uint instance, const std::string &method) const;

    // Calls a function for every contact of the last step between a sensed body and a partner, with the contact index, the partner index, and the sign that turns MuJoCo's force of geom 0 on geom 1 into the force on the sensed body
    template<typename Function>
    void forEachSensedContact(uint instance, Function function) const;

    const PhysicsEngine *engine;
    std::vector<char> body_is_sensed; // indexed by MuJoCo body ID
    std::vector<int> body_partner; // indexed by MuJoCo body ID: the partner of an unsensed body, before any target link
    std::vector<std::string> partner_names;
    size_t target_partner = 0;
    std::vector<int> target_body_IDs; // per instance, -1 when there is no target
};

#endif
