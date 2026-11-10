/** \file "ToolVelocityAction.cpp" Cartesian velocity control of a robot's tool point.

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

#include "ToolVelocityAction.h"

#include <mujoco/mujoco.h>

using namespace helios;

ToolVelocityAction::ToolVelocityAction(Robot *robot, float max_speed, float max_lead) : robot(robot), max_speed(max_speed), max_lead(max_lead) {
    if (robot == nullptr) {
        helios_runtime_error("ERROR (ToolVelocityAction): The Robot pointer must not be null.");
    }
    if (!(max_speed > 0.f) || !(max_lead > 0.f)) {
        helios_runtime_error("ERROR (ToolVelocityAction): The maximum speed and lead must be positive, but were " + std::to_string(max_speed) + " m/s and " + std::to_string(max_lead) + " m.");
    }
    // Fails here, rather than at the first step, if the robot is not initialized or has no tool
    static_cast<void>(robot->getToolPosition(0));
    prepareInstances();
}

void ToolVelocityAction::prepareInstances() {
    const uint instance_count = robot->getPhysicsEngine()->getInstanceCount();
    targets.resize(std::max(targets.size(), static_cast<size_t>(instance_count)));
    reset_axes.resize(targets.size());
    needs_start.resize(targets.size(), 1);
}

void ToolVelocityAction::requireInstance(uint instance, const std::string &method) const {
    if (instance >= targets.size()) {
        if (instance < robot->getPhysicsEngine()->getInstanceCount()) {
            helios_runtime_error("ERROR (ToolVelocityAction::" + method + "): Instance " + std::to_string(instance) + " was created after the action. Call prepareInstances() after creating instances.");
        }
        helios_runtime_error("ERROR (ToolVelocityAction::" + method + "): Instance " + std::to_string(instance) + " does not exist.");
    }
}

size_t ToolVelocityAction::jointIndex(const std::string &joint_name, const std::string &method) const {
    const std::vector<std::string> names = robot->getJointNames();
    auto found = std::find(names.begin(), names.end(), joint_name);
    if (found == names.end()) {
        helios_runtime_error("ERROR (ToolVelocityAction::" + method + "): Joint " + joint_name + " does not have a servo. Only joints given to Robot::addJointServo() can be used here.");
    }
    return static_cast<size_t>(found - names.begin());
}

void ToolVelocityAction::setLockedJoints(const std::vector<std::string> &joint_names) {
    for (const std::string &joint_name: joint_names) {
        static_cast<void>(jointIndex(joint_name, "setLockedJoints"));
    }
    locked_joints = joint_names;
}

void ToolVelocityAction::setToolAxis(const vec3 &axis) {
    if (axis.magnitude() < 1e-6f) {
        helios_runtime_error("ERROR (ToolVelocityAction::setToolAxis): The tool axis must not be a zero vector.");
    }
    axis_mode = AxisMode::fixed;
    fixed_axis = axis;
    fixed_axis.normalize();
}

void ToolVelocityAction::setToolAxisAwayFromPivot(const vec3 &pivot_in_robot_frame, const std::string &slide_joint, float min_distance) {
    if (!(min_distance > 0.f)) {
        helios_runtime_error("ERROR (ToolVelocityAction::setToolAxisAwayFromPivot): The minimum distance must be positive, but was " + std::to_string(min_distance) + " m.");
    }
    pivot_slide_joint_index = -1;
    pivot_slide_joint_ID = -1;
    if (!slide_joint.empty()) {
        pivot_slide_joint_index = static_cast<int>(jointIndex(slide_joint, "setToolAxisAwayFromPivot"));
        const mjModel *model = robot->getPhysicsEngine()->getMuJoCoModel();
        pivot_slide_joint_ID = mj_name2id(model, mjOBJ_JOINT, (robot->getName() + "/" + slide_joint).c_str());
        if (pivot_slide_joint_ID < 0 || model->jnt_type[pivot_slide_joint_ID] != mjJNT_SLIDE) {
            helios_runtime_error("ERROR (ToolVelocityAction::setToolAxisAwayFromPivot): Joint " + slide_joint + " is not a slide joint.");
        }
    }
    // The pivot in the world frame with the slide joint at zero
    const float yaw = deg2rad(robot->getBaseYaw());
    const float cos_yaw = std::cos(yaw);
    const float sin_yaw = std::sin(yaw);
    pivot = robot->getBasePosition() + make_vec3(cos_yaw * pivot_in_robot_frame.x - sin_yaw * pivot_in_robot_frame.y, sin_yaw * pivot_in_robot_frame.x + cos_yaw * pivot_in_robot_frame.y, pivot_in_robot_frame.z);
    pivot_min_distance = min_distance;
    axis_mode = AxisMode::away_from_pivot;
}

void ToolVelocityAction::resetInstance(uint instance) {
    requireInstance(instance, "resetInstance");
    needs_start[instance] = 1;
}

void ToolVelocityAction::startFromTool(uint instance) {
    targets[instance] = robot->getToolPosition(instance);
    reset_axes[instance] = robot->getToolAxis(instance);
    needs_start[instance] = 0;
}

vec3 ToolVelocityAction::getTarget(uint instance) const {
    requireInstance(instance, "getTarget");
    return needs_start[instance] ? robot->getToolPosition(instance) : targets[instance];
}

uint ToolVelocityAction::getActionSize() const {
    return 3;
}

void ToolVelocityAction::applyAction(const float *action, float dt, uint instance) {
    apply(make_vec3(action[0], action[1], action[2]), dt, instance);
}

void ToolVelocityAction::apply(const vec3 &command, float dt, uint instance) {
    requireInstance(instance, "apply");
    if (needs_start[instance]) {
        startFromTool(instance);
    }
    const vec3 velocity = max_speed * make_vec3(std::clamp(command.x, -1.f, 1.f), std::clamp(command.y, -1.f, 1.f), std::clamp(command.z, -1.f, 1.f));
    const vec3 tool = robot->getToolPosition(instance);
    vec3 &target = targets[instance];
    target = target + dt * velocity;
    vec3 lead = target - tool;
    if (lead.magnitude() > max_lead) {
        lead = lead / lead.magnitude() * max_lead;
        target = tool + lead;
    }

    vec3 axis;
    if (axis_mode == AxisMode::away_from_pivot) {
        vec3 pivot_now = pivot;
        if (pivot_slide_joint_index >= 0) {
            const float slide = robot->getJointPositions(instance)[pivot_slide_joint_index];
            const mjtNum *slide_axis = robot->getPhysicsEngine()->getMuJoCoData(instance)->xaxis + 3 * pivot_slide_joint_ID;
            pivot_now = pivot_now + slide * make_vec3(float(slide_axis[0]), float(slide_axis[1]), float(slide_axis[2]));
        }
        axis = target - make_vec3(pivot_now.x, pivot_now.y, target.z);
        axis.z = 0.f;
        if (axis.magnitude() <= pivot_min_distance) {
            return;
        }
        axis.normalize();
    } else {
        axis = axis_mode == AxisMode::fixed ? fixed_axis : reset_axes[instance];
    }

    const RobotInverseKinematicsResult pose = robot->solveInverseKinematics(target, axis, locked_joints, instance);
    if (pose.converged) {
        robot->setJointTargets(pose.joint_positions, instance);
    }
}
