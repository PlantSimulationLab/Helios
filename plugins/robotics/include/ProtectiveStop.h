/** \file "ProtectiveStop.h" Stops a robot that presses on its surroundings too hard.

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

#ifndef HELIOS_PROTECTIVE_STOP_H
#define HELIOS_PROTECTIVE_STOP_H

#include "ContactSensor.h"
#include "RobotComponent.h"

//! Trips when the contact force on a robot passes a limit, as a robot's protective stop does
/**
 * The force is the total measured by a ContactSensor on the robot, leaving out partners the robot is meant to touch, such as the "target" it is picking.
 * Add it to a ControlLoop, or call update() after every physics step yourself, so that a force peak between control steps is not missed. Once tripped, the stop stays tripped until the instance is reset.
 * It reports the stop and does not move the robot: the task decides what a stop means, such as ending the episode.
 * Different instances can be used at the same time from different threads.
 */
class ProtectiveStop : public RobotComponent {
public:
    //! Constructor
    /**
     * \param[in] sensor Contact sensor on the robot.
     * \param[in] force_limit Contact force in N above which the stop trips.
     * \param[in] ignored_partners Names of the sensor's partners whose contacts do not count, such as "target" and "ground".
     */
    ProtectiveStop(const ContactSensor *sensor, float force_limit, const std::vector<std::string> &ignored_partners = {"target"});

    //! Allocate per-instance state for every simulation instance that exists now. Call it after creating instances, before using them from several threads.
    void prepareInstances() override;

    //! Clear the stop of an instance
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance = 0) override;

    //! Start measuring the peak force of a control step. Called by ControlLoop.
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void beginControlStep(uint instance) override;

    //! Measure the force after a physics step (update()). Called by ControlLoop.
    /**
     * \param[in] instance ID of the simulation instance.
     * \return Whether the stop tripped in this step.
     */
    bool afterPhysicsStep(uint instance) override;

    //! Get the largest force in N measured by update() since the control step began (beginControlStep())
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getPeakForce(uint instance = 0) const;

    //! Measure the contact force after a physics step, and trip the stop if it passes the limit
    /**
     * \param[in] instance ID of the simulation instance.
     * \return The magnitude of the contact force in N.
     */
    float update(uint instance = 0);

    //! Measure the magnitude of the contact force that counts toward the stop, in N, without tripping it
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float measureForce(uint instance = 0) const;

    //! Check whether the stop has tripped since the instance was last reset
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] bool isStopped(uint instance = 0) const;

    //! Get the contact force in N above which the stop trips
    [[nodiscard]] float getForceLimit() const;

private:
    void requireInstance(uint instance, const std::string &method) const;

    const ContactSensor *sensor;
    float force_limit;
    std::vector<std::string> ignored_partners;
    std::vector<char> stopped; // per instance
    std::vector<float> peak_forces; // per instance, since the control step began
};

#endif
