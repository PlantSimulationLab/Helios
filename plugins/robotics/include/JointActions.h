/** \file "JointActions.h" Joint position and velocity control of a robot, as actions of a robot task.

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

#ifndef HELIOS_JOINT_ACTIONS_H
#define HELIOS_JOINT_ACTIONS_H

#include "Robot.h"
#include "RobotComponent.h"

//! Sets the servo targets of some joints of a robot from commands in [-1, 1], scaled to a range of positions
/**
 * A command of -1 is the bottom of a joint's range and 1 the top. It takes one number per joint, in the order given to the constructor.
 * It keeps no state of its own. Construct it after Robot::initialize().
 */
class JointPositionAction : public RobotAction {
public:
    //! Constructor
    /**
     * \param[in] robot Initialized robot.
     * \param[in] joint_names Names of servo-driven joints of the robot.
     * \param[in] ranges Lowest (x) and highest (y) position of each joint that the commands span: degrees for hinges, meters for slides. If empty, the ranges of the joints in the model file (Robot::getJointRange()), which must then all be limited.
     */
    JointPositionAction(Robot *robot, const std::vector<std::string> &joint_names, const std::vector<helios::vec2> &ranges = {});

    //! Does nothing: the action keeps no state
    void prepareInstances() override;

    //! Does nothing: the action keeps no state
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance) override;

    //! Get the number of numbers in the action: one per joint
    [[nodiscard]] uint getActionSize() const override;

    //! Set the joint targets from commands
    /**
     * \param[in] action Command of each joint, clamped to [-1, 1].
     * \param[in] dt Duration of the control step in seconds, which is not used.
     * \param[in] instance ID of the simulation instance.
     */
    void applyAction(const float *action, float dt, uint instance) override;

private:
    Robot *robot;
    std::vector<std::string> joint_names;
    std::vector<helios::vec2> ranges;
};

//! Moves the servo targets of some joints of a robot at commanded velocities, from commands in [-1, 1] scaled by a maximum speed
/**
 * Each control step moves the target of each joint by its command times its maximum speed times the control period, keeps it within the joint's range in the model file, and lets it lead the joint's position by at most a set distance, so that it cannot run away while the joint is blocked.
 * It takes one number per joint, in the order given to the constructor. The targets start from the joints' positions at the first control step after the instance is reset.
 * Construct it after Robot::initialize(). Different instances can be driven at the same time from different threads.
 */
class JointVelocityAction : public RobotAction {
public:
    //! Constructor
    /**
     * \param[in] robot Initialized robot.
     * \param[in] joint_names Names of servo-driven joints of the robot.
     * \param[in] max_speeds Speed of each joint's target for a command of 1: degrees per second for hinges, meters per second for slides.
     * \param[in] max_leads Farthest each joint's target may lead its position: degrees for hinges, meters for slides.
     */
    JointVelocityAction(Robot *robot, const std::vector<std::string> &joint_names, const std::vector<float> &max_speeds, const std::vector<float> &max_leads);

    //! Allocate per-instance state for every simulation instance that exists now. Call it after creating instances, before using them from several threads.
    void prepareInstances() override;

    //! Start the targets of an instance again from the joints' positions at the next control step
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance) override;

    //! Get the number of numbers in the action: one per joint
    [[nodiscard]] uint getActionSize() const override;

    //! Move the joint targets for one control step
    /**
     * \param[in] action Velocity command of each joint, clamped to [-1, 1].
     * \param[in] dt Duration of the control step in seconds.
     * \param[in] instance ID of the simulation instance.
     */
    void applyAction(const float *action, float dt, uint instance) override;

    //! Get the current target of each joint, in the order given to the constructor
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] std::vector<float> getTargets(uint instance) const;

private:
    void requireInstance(uint instance, const std::string &method) const;
    [[nodiscard]] std::vector<float> currentPositions(uint instance) const;

    Robot *robot;
    std::vector<std::string> joint_names;
    std::vector<size_t> joint_indices; // in Robot::getJointNames()
    std::vector<float> max_speeds;
    std::vector<float> max_leads;
    std::vector<helios::vec2> ranges;
    std::vector<std::vector<float>> targets; // per instance
    std::vector<char> needs_start; // per instance
};

#endif
