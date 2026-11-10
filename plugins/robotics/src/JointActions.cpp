/** \file "JointActions.cpp" Joint position and velocity control of a robot, as actions of a robot task.

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

#include "JointActions.h"

using namespace helios;

namespace {

    // Index of each joint in the robot's servo-driven joints, failing for a joint without a servo
    std::vector<size_t> findJoints(const Robot &robot, const std::vector<std::string> &joint_names, const std::string &caller) {
        if (joint_names.empty()) {
            helios_runtime_error("ERROR (" + caller + "): At least one joint must be given.");
        }
        const std::vector<std::string> names = robot.getJointNames();
        std::vector<size_t> indices;
        for (const std::string &joint_name: joint_names) {
            auto found = std::find(names.begin(), names.end(), joint_name);
            if (found == names.end()) {
                helios_runtime_error("ERROR (" + caller + "): Joint " + joint_name + " does not have a servo. Only joints given to Robot::addJointServo() can be used.");
            }
            if (std::find(indices.begin(), indices.end(), size_t(found - names.begin())) != indices.end()) {
                helios_runtime_error("ERROR (" + caller + "): Joint " + joint_name + " is given more than once.");
            }
            indices.push_back(static_cast<size_t>(found - names.begin()));
        }
        return indices;
    }

} // namespace

JointPositionAction::JointPositionAction(Robot *robot, const std::vector<std::string> &joint_names, const std::vector<vec2> &ranges) : robot(robot), joint_names(joint_names), ranges(ranges) {
    if (robot == nullptr) {
        helios_runtime_error("ERROR (JointPositionAction): The Robot pointer must not be null.");
    }
    static_cast<void>(findJoints(*robot, joint_names, "JointPositionAction"));
    if (this->ranges.empty()) {
        for (const std::string &joint_name: joint_names) {
            const vec2 range = robot->getJointRange(joint_name);
            if (!std::isfinite(range.x) || !std::isfinite(range.y)) {
                helios_runtime_error("ERROR (JointPositionAction): Joint " + joint_name + " is not limited in the model file, so give the range of positions the commands span.");
            }
            this->ranges.push_back(range);
        }
    }
    if (this->ranges.size() != joint_names.size()) {
        helios_runtime_error("ERROR (JointPositionAction): " + std::to_string(this->ranges.size()) + " ranges were given for " + std::to_string(joint_names.size()) + " joints.");
    }
    for (size_t j = 0; j < this->ranges.size(); j++) {
        if (!(this->ranges[j].y > this->ranges[j].x)) {
            helios_runtime_error("ERROR (JointPositionAction): The range of joint " + joint_names[j] + " must have its highest position above its lowest.");
        }
    }
}

void JointPositionAction::prepareInstances() {
}

void JointPositionAction::resetInstance(uint instance) {
}

uint JointPositionAction::getActionSize() const {
    return static_cast<uint>(joint_names.size());
}

void JointPositionAction::applyAction(const float *action, float dt, uint instance) {
    for (size_t j = 0; j < joint_names.size(); j++) {
        const float command = std::clamp(action[j], -1.f, 1.f);
        robot->setJointTarget(joint_names[j], ranges[j].x + 0.5f * (command + 1.f) * (ranges[j].y - ranges[j].x), instance);
    }
}

JointVelocityAction::JointVelocityAction(Robot *robot, const std::vector<std::string> &joint_names, const std::vector<float> &max_speeds, const std::vector<float> &max_leads) :
    robot(robot), joint_names(joint_names), max_speeds(max_speeds), max_leads(max_leads) {
    if (robot == nullptr) {
        helios_runtime_error("ERROR (JointVelocityAction): The Robot pointer must not be null.");
    }
    joint_indices = findJoints(*robot, joint_names, "JointVelocityAction");
    if (max_speeds.size() != joint_names.size() || max_leads.size() != joint_names.size()) {
        helios_runtime_error("ERROR (JointVelocityAction): Give one maximum speed and one maximum lead for each of the " + std::to_string(joint_names.size()) + " joints.");
    }
    for (size_t j = 0; j < joint_names.size(); j++) {
        if (!(max_speeds[j] > 0.f) || !(max_leads[j] > 0.f)) {
            helios_runtime_error("ERROR (JointVelocityAction): The maximum speed and lead of joint " + joint_names[j] + " must be positive.");
        }
        ranges.push_back(robot->getJointRange(joint_names[j]));
    }
    prepareInstances();
}

void JointVelocityAction::prepareInstances() {
    const size_t instance_count = std::max(targets.size(), static_cast<size_t>(robot->getPhysicsEngine()->getInstanceCount()));
    targets.resize(instance_count, std::vector<float>(joint_names.size(), 0.f));
    needs_start.resize(instance_count, 1);
}

void JointVelocityAction::requireInstance(uint instance, const std::string &method) const {
    if (instance >= targets.size()) {
        helios_runtime_error("ERROR (JointVelocityAction::" + method + "): Instance " + std::to_string(instance) + " does not exist, or was created after the action. Call prepareInstances() after creating instances.");
    }
}

std::vector<float> JointVelocityAction::currentPositions(uint instance) const {
    const std::vector<float> all = robot->getJointPositions(instance);
    std::vector<float> positions;
    for (size_t index: joint_indices) {
        positions.push_back(all[index]);
    }
    return positions;
}

void JointVelocityAction::resetInstance(uint instance) {
    requireInstance(instance, "resetInstance");
    needs_start[instance] = 1;
}

uint JointVelocityAction::getActionSize() const {
    return static_cast<uint>(joint_names.size());
}

void JointVelocityAction::applyAction(const float *action, float dt, uint instance) {
    requireInstance(instance, "applyAction");
    const std::vector<float> positions = currentPositions(instance);
    std::vector<float> &target = targets[instance];
    if (needs_start[instance]) {
        target = positions;
        needs_start[instance] = 0;
    }
    for (size_t j = 0; j < joint_names.size(); j++) {
        target[j] += std::clamp(action[j], -1.f, 1.f) * max_speeds[j] * dt;
        target[j] = std::clamp(target[j], ranges[j].x, ranges[j].y);
        target[j] = std::clamp(target[j], positions[j] - max_leads[j], positions[j] + max_leads[j]);
        robot->setJointTarget(joint_names[j], target[j], instance);
    }
}

std::vector<float> JointVelocityAction::getTargets(uint instance) const {
    requireInstance(instance, "getTargets");
    return needs_start[instance] ? currentPositions(instance) : targets[instance];
}
