/** \file "ControlLoop.h" Runs the physics steps of each control step of a robot task, with the components that act inside them.

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

#ifndef HELIOS_CONTROL_LOOP_H
#define HELIOS_CONTROL_LOOP_H

#include "PhysicsEngine.h"
#include "RobotComponent.h"

//! What happened in one control step, returned by ControlLoop::step()
struct ControlStepResult {
    //! Physics steps simulated, fewer than ControlLoop::getPhysicsStepsPerControl() if an event ended the control step early
    uint physics_steps = 0;
    //! Whether an event ended the control step early: a link released by load (PhysicsEngine::setDetachForce()), or a component reporting one from RobotComponent::afterPhysicsStep()
    bool event = false;
};

//! Runs the physics steps of a control step for one simulation instance, calling the components that act within them
/**
 * A robot task is commanded at a control rate, such as 25 Hz, while the physics runs many steps in between. Whatever must act or be checked at every physics step, such as the grip of a VacuumCup or a ProtectiveStop, is a RobotComponent added here, so that a controller only acts once per control step.
 * A control step stops at the first physics step in which an event happens, so that the state when it ends is the state at the event.
 * The loop does not own its components. Different instances can be stepped at the same time from different threads.
 */
class ControlLoop {
public:
    //! Constructor
    /**
     * \param[in] engine Physics simulation to step.
     * \param[in] physics_steps_per_control Physics steps in each control step.
     */
    ControlLoop(PhysicsEngine *engine, uint physics_steps_per_control);

    //! Add a component, whose hooks are called in the order components were added
    /**
     * \param[in] component Component to call. It must outlive the loop.
     */
    void addComponent(RobotComponent *component);

    //! Get the number of physics steps in each control step
    [[nodiscard]] uint getPhysicsStepsPerControl() const;

    //! Get the duration of a control step in seconds
    [[nodiscard]] float getControlPeriod() const;

    //! Allocate per-instance state of every component for every simulation instance that exists now. Call it after creating instances, before using them from several threads.
    void prepareInstances();

    //! Return a simulation instance to its initial state (PhysicsEngine::resetInstance()), and every component's state for it to its start
    /**
     * This resets the simulation, so robots return to the pose the model was built in, or to PhysicsEngine::setResetPose(). Place them afterward, such as with Robot::setJointPositions().
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance);

    //! Run one control step of an instance
    /**
     * \param[in] instance ID of the simulation instance.
     * \return How many physics steps were run, and whether an event ended the control step early.
     */
    ControlStepResult step(uint instance);

private:
    PhysicsEngine *engine;
    uint physics_steps_per_control;
    std::vector<RobotComponent *> components;
};

#endif
