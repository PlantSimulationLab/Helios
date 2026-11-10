/** \file "RobotComponent.h" Interfaces of the parts of a robot task that run inside the control loop.

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

#ifndef HELIOS_ROBOT_COMPONENT_H
#define HELIOS_ROBOT_COMPONENT_H

#include "global.h"

//! A part of a robot task that keeps state per simulation instance and runs inside a ControlLoop, such as an end effector or a safety check
/**
 * ControlLoop calls the hooks in this order for each control step of an instance: beginControlStep(), then for every physics step beforePhysicsStep(), the step itself, and afterPhysicsStep().
 * Every hook acts on one instance, and different instances may be run at the same time from different threads, so a hook must only touch the state of its own instance.
 */
class RobotComponent {
public:
    virtual ~RobotComponent() = default;

    //! Allocate per-instance state for every simulation instance that exists now
    virtual void prepareInstances() = 0;

    //! Return an instance's state to its start, as at the start of an episode
    /**
     * \param[in] instance ID of the simulation instance.
     */
    virtual void resetInstance(uint instance) = 0;

    //! Called at the start of each control step
    /**
     * \param[in] instance ID of the simulation instance.
     */
    virtual void beginControlStep(uint instance) {
    }

    //! Called before each physics step, such as to apply forces
    /**
     * \param[in] instance ID of the simulation instance.
     * \param[in] dt Physics time step in seconds.
     */
    virtual void beforePhysicsStep(uint instance, float dt) {
    }

    //! Called after each physics step, such as to check a limit
    /**
     * \param[in] instance ID of the simulation instance.
     * \return Whether something happened in this step that should end the control step early, such as a seal breaking or a protective stop. Report each event once, in the step it happens.
     */
    virtual bool afterPhysicsStep(uint instance) {
        return false;
    }
};

//! A RobotComponent that a controller commands through a block of numbers each control step, such as a velocity command or a grip command
class RobotAction : public RobotComponent {
public:
    //! Get how many numbers the action takes each control step
    [[nodiscard]] virtual uint getActionSize() const = 0;

    //! Command the action for one control step
    /**
     * \param[in] action getActionSize() numbers.
     * \param[in] dt Duration of the control step in seconds.
     * \param[in] instance ID of the simulation instance.
     */
    virtual void applyAction(const float *action, float dt, uint instance) = 0;
};

#endif
