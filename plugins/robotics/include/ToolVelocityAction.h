/** \file "ToolVelocityAction.h" Cartesian velocity control of a robot's tool point.

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

#ifndef HELIOS_TOOL_VELOCITY_ACTION_H
#define HELIOS_TOOL_VELOCITY_ACTION_H

#include "Robot.h"
#include "RobotComponent.h"

//! Moves a robot's tool point at a commanded Cartesian velocity, through inverse kinematics and the joint servos
/**
 * Each control step moves a target point at the commanded velocity, solves inverse kinematics for it (Robot::solveInverseKinematics()), and sets the joint targets.
 * The target may lead the tool by at most a set distance, so that it cannot run away while the arm is blocked or slower than commanded, and the arm does not lunge once it is free.
 * A target the arm cannot reach leaves the joint targets as they were.
 *
 * The tool axis is held along a fixed direction (by default the one it had when the instance was last reset), or points horizontally away from a vertical line through a pivot, as for an arm on a turret that turns to face its work.
 * As a RobotAction it takes three numbers, the command along x, y, and z.
 * Construct it after Robot::initialize(). Different instances can be driven at the same time from different threads.
 */
class ToolVelocityAction : public RobotAction {
public:
    //! Constructor
    /**
     * \param[in] robot Initialized robot with a tool (Robot::setTool()).
     * \param[in] max_speed Speed of the tool target in m/s for a command of 1 along an axis.
     * \param[in] max_lead Farthest the target may lead the tool point, in meters.
     */
    ToolVelocityAction(Robot *robot, float max_speed, float max_lead);

    //! Allocate per-instance state for every simulation instance that exists now. Call it after creating instances, before using them from several threads.
    void prepareInstances() override;

    //! Keep joints at their current positions when solving, such as a roll joint that would otherwise spin the tool about its axis
    /**
     * \param[in] joint_names Names of servo-driven joints of the robot.
     */
    void setLockedJoints(const std::vector<std::string> &joint_names);

    //! Hold the tool axis along a fixed world direction
    /**
     * \param[in] axis Direction of the tool axis in world coordinates.
     */
    void setToolAxis(const helios::vec3 &axis);

    //! Point the tool axis horizontally away from a vertical line through a pivot, such as the turret an arm turns on
    /**
     * The pivot is fixed in the robot's frame, optionally carried along a slide joint of the robot, such as the rail a turret rides on. While the target is nearer the line than min_distance horizontally, the joint targets are left as they were, since the direction is then undefined.
     * \param[in] pivot Pivot in the robot's frame (relative to the position and yaw given to the Robot constructor) with every slide joint at zero.
     * \param[in] slide_joint Name of a servo-driven slide joint that carries the pivot along its axis, or an empty string if the pivot is fixed.
     * \param[in] min_distance Smallest horizontal distance in meters from the pivot's line at which the target is solved for.
     */
    void setToolAxisAwayFromPivot(const helios::vec3 &pivot, const std::string &slide_joint = "", float min_distance = 0.05f);

    //! Start an instance's target again from wherever the tool is at the next apply(), and, if no other axis was set, hold the axis the tool has then
    /**
     * Call it whenever the robot is placed, such as after Robot::resetInstance() or Robot::setJointPositions(). ControlLoop::resetInstance() calls it.
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance = 0) override;

    //! Get the number of numbers in the action: three, the velocity command along x, y, and z
    [[nodiscard]] uint getActionSize() const override;

    //! Apply a velocity command given as three numbers (apply())
    /**
     * \param[in] action Command along x, y, and z.
     * \param[in] dt Duration of the control step in seconds.
     * \param[in] instance ID of the simulation instance.
     */
    void applyAction(const float *action, float dt, uint instance) override;

    //! Move the target for one control step and set the joint targets that reach it
    /**
     * \param[in] command Velocity command along the world x, y, and z axes, each clamped to [-1, 1] and scaled by the maximum speed.
     * \param[in] dt Duration of the control step in seconds.
     * \param[in] instance ID of the simulation instance.
     */
    void apply(const helios::vec3 &command, float dt, uint instance = 0);

    //! Get the current target of the tool point
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getTarget(uint instance = 0) const;

private:
    enum class AxisMode { reset_axis, fixed, away_from_pivot };

    void requireInstance(uint instance, const std::string &method) const;
    [[nodiscard]] size_t jointIndex(const std::string &joint_name, const std::string &method) const;

    Robot *robot;
    float max_speed;
    float max_lead;
    std::vector<std::string> locked_joints;

    AxisMode axis_mode = AxisMode::reset_axis;
    helios::vec3 fixed_axis;
    helios::vec3 pivot;
    int pivot_slide_joint_index = -1; // in Robot::getJointNames()
    int pivot_slide_joint_ID = -1; // MuJoCo joint ID
    float pivot_min_distance = 0.05f;

    void startFromTool(uint instance);

    std::vector<helios::vec3> targets; // per instance
    std::vector<helios::vec3> reset_axes; // per instance
    std::vector<char> needs_start; // per instance: the target starts from the tool at the next apply()
};

#endif
