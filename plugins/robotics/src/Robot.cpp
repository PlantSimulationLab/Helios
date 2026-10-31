/** \file "Robot.cpp" Robots in a PhysicsEngine simulation.

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

#include "Robot.h"

#include <mujoco/mujoco.h>

using namespace helios;

Robot::Robot(PhysicsEngine *engine, const std::string &model_file, const std::string &name, const vec3 &position, float yaw_degrees) : engine(engine), prefix(name + "/") {
    if (engine == nullptr) {
        helios_runtime_error("ERROR (Robot): The PhysicsEngine pointer must not be null.");
    }
    if (name.empty()) {
        helios_runtime_error("ERROR (Robot): The robot must have a name.");
    }
    engine->addModelFile(model_file, prefix, position, yaw_degrees, true);
}

Robot::~Robot() {
    if (kinematics_data != nullptr) {
        mj_deleteData(kinematics_data);
    }
}

void Robot::requireInitialized(const std::string &method) const {
    if (!initialized) {
        helios_runtime_error("ERROR (Robot::" + method + "): The robot has not been initialized. Call PhysicsEngine::initialize() and then Robot::initialize() first.");
    }
}

void Robot::requireBeforeSimulationInitialized(const std::string &method) const {
    if (initialized) {
        helios_runtime_error("ERROR (Robot::" + method + "): This must be done before the robot is initialized.");
    }
}

void Robot::addJointServo(const std::string &joint_name, float stiffness, float damping, float force_limit, float armature) {
    requireBeforeSimulationInitialized("addJointServo");
    for (const JointRecord &joint: joints) {
        if (joint.name == joint_name) {
            helios_runtime_error("ERROR (Robot::addJointServo): Joint " + joint_name + " already has a servo.");
        }
    }
    engine->addPositionServo(prefix + joint_name, stiffness, damping, force_limit, armature);
    JointRecord joint;
    joint.name = joint_name;
    joints.push_back(joint);
}

void Robot::setTool(const std::string &body_name, const vec3 &point, const vec3 &axis) {
    requireBeforeSimulationInitialized("setTool");
    if (!tool_body_name.empty()) {
        helios_runtime_error("ERROR (Robot::setTool): The tool has already been set to body " + tool_body_name + ".");
    }
    if (axis.magnitude() < 1e-6f) {
        helios_runtime_error("ERROR (Robot::setTool): The tool axis must not be a zero vector.");
    }
    tool_body_name = body_name;
    tool_point = point;
    tool_axis = axis;
    tool_axis.normalize();
}

void Robot::initialize() {
    if (initialized) {
        helios_runtime_error("ERROR (Robot::initialize): The robot was already initialized.");
    }
    const mjModel *model = engine->getMuJoCoModel();

    for (JointRecord &joint: joints) {
        const std::string joint_name = prefix + joint.name;
        joint.joint_ID = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
        joint.actuator_ID = mj_name2id(model, mjOBJ_ACTUATOR, (joint_name + "_servo").c_str());
        if (joint.joint_ID < 0 || joint.actuator_ID < 0) {
            helios_runtime_error("ERROR (Robot::initialize): Joint " + joint.name + " or its servo is missing from the compiled model.");
        }
        const int joint_type = model->jnt_type[joint.joint_ID];
        if (joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE) {
            helios_runtime_error("ERROR (Robot::initialize): Joint " + joint.name + " is not a hinge or slide, so it cannot be driven by a position servo.");
        }
        joint.is_hinge = joint_type == mjJNT_HINGE;
        joint.qpos_address = model->jnt_qposadr[joint.joint_ID];
        joint.dof_address = model->jnt_dofadr[joint.joint_ID];
    }

    if (!tool_body_name.empty()) {
        tool_body_ID = mj_name2id(model, mjOBJ_BODY, (prefix + tool_body_name).c_str());
        if (tool_body_ID < 0) {
            helios_runtime_error("ERROR (Robot::initialize): Tool body " + tool_body_name + " is missing from the compiled model.");
        }
        // A body's parent always has a lower ID, so one pass in order finds all descendants
        body_is_on_tool.assign(model->nbody, 0);
        body_is_on_tool[tool_body_ID] = 1;
        for (int body_ID = tool_body_ID + 1; body_ID < model->nbody; body_ID++) {
            body_is_on_tool[body_ID] = body_is_on_tool[model->body_parentid[body_ID]];
        }
    }

    kinematics_data = mj_makeData(model);
    if (kinematics_data == nullptr) {
        helios_runtime_error("ERROR (Robot::initialize): MuJoCo could not allocate the data used for inverse kinematics.");
    }
    initialized = true;
}

void Robot::setHomePose(const std::vector<float> &positions) {
    requireInitialized("setHomePose");
    if (positions.size() != joints.size()) {
        helios_runtime_error("ERROR (Robot::setHomePose): " + std::to_string(positions.size()) + " positions were given for " + std::to_string(joints.size()) + " joints.");
    }
    home_pose = positions;
    for (uint instance = 0; instance < engine->getInstanceCount(); instance++) {
        resetInstance(instance);
    }
}

void Robot::resetInstance(uint instance) {
    requireInitialized("resetInstance");
    engine->resetInstance(instance);
    if (home_pose.empty()) {
        return;
    }
    // Set in the state rather than in the model's reference pose: MuJoCo measures a hinge's rotation from that reference, so changing it would change the reported angle without moving the arm
    mjData *data = engine->getMuJoCoData(instance);
    for (size_t j = 0; j < joints.size(); j++) {
        data->qpos[joints[j].qpos_address] = toModelUnits(joints[j], home_pose[j]);
    }
    setJointTargets(home_pose, instance);
    mj_forward(engine->getMuJoCoModel(), data);
}

float Robot::toUserUnits(const JointRecord &joint, double value) const {
    return joint.is_hinge ? rad2deg(float(value)) : float(value);
}

double Robot::toModelUnits(const JointRecord &joint, float value) const {
    return joint.is_hinge ? double(deg2rad(value)) : double(value);
}

size_t Robot::findJoint(const std::string &joint_name, const std::string &method) const {
    for (size_t j = 0; j < joints.size(); j++) {
        if (joints[j].name == joint_name) {
            return j;
        }
    }
    helios_runtime_error("ERROR (Robot::" + method + "): Joint " + joint_name + " does not have a servo. Only joints given to addJointServo() can be used here.");
    return 0;
}

std::vector<std::string> Robot::getJointNames() const {
    std::vector<std::string> names;
    names.reserve(joints.size());
    for (const JointRecord &joint: joints) {
        names.push_back(joint.name);
    }
    return names;
}

void Robot::setJointTargets(const std::vector<float> &targets, uint instance) {
    requireInitialized("setJointTargets");
    if (targets.size() != joints.size()) {
        helios_runtime_error("ERROR (Robot::setJointTargets): " + std::to_string(targets.size()) + " targets were given for " + std::to_string(joints.size()) + " joints.");
    }
    mjData *data = engine->getMuJoCoData(instance);
    for (size_t j = 0; j < joints.size(); j++) {
        data->ctrl[joints[j].actuator_ID] = toModelUnits(joints[j], targets[j]);
    }
}

void Robot::setJointTarget(const std::string &joint_name, float target, uint instance) {
    requireInitialized("setJointTarget");
    const JointRecord &joint = joints[findJoint(joint_name, "setJointTarget")];
    engine->getMuJoCoData(instance)->ctrl[joint.actuator_ID] = toModelUnits(joint, target);
}

std::vector<float> Robot::getJointPositions(uint instance) const {
    requireInitialized("getJointPositions");
    const mjData *data = engine->getMuJoCoData(instance);
    std::vector<float> positions(joints.size());
    for (size_t j = 0; j < joints.size(); j++) {
        positions[j] = toUserUnits(joints[j], data->qpos[joints[j].qpos_address]);
    }
    return positions;
}

vec3 Robot::getToolPosition(uint instance) const {
    requireInitialized("getToolPosition");
    if (tool_body_ID < 0) {
        helios_runtime_error("ERROR (Robot::getToolPosition): The robot has no tool. Call setTool() before PhysicsEngine::initialize().");
    }
    const mjData *data = engine->getMuJoCoData(instance);
    const mjtNum point[3] = {tool_point.x, tool_point.y, tool_point.z};
    mjtNum rotated[3];
    mju_mulMatVec3(rotated, data->xmat + 9 * tool_body_ID, point);
    const mjtNum *origin = data->xpos + 3 * tool_body_ID;
    return make_vec3(float(origin[0] + rotated[0]), float(origin[1] + rotated[1]), float(origin[2] + rotated[2]));
}

vec3 Robot::getToolAxis(uint instance) const {
    requireInitialized("getToolAxis");
    if (tool_body_ID < 0) {
        helios_runtime_error("ERROR (Robot::getToolAxis): The robot has no tool. Call setTool() before PhysicsEngine::initialize().");
    }
    const mjtNum axis[3] = {tool_axis.x, tool_axis.y, tool_axis.z};
    mjtNum rotated[3];
    mju_mulMatVec3(rotated, engine->getMuJoCoData(instance)->xmat + 9 * tool_body_ID, axis);
    return make_vec3(float(rotated[0]), float(rotated[1]), float(rotated[2]));
}

vec3 Robot::getToolForce(uint instance) const {
    requireInitialized("getToolForce");
    if (tool_body_ID < 0) {
        helios_runtime_error("ERROR (Robot::getToolForce): The robot has no tool. Call setTool() before PhysicsEngine::initialize().");
    }
    const mjModel *model = engine->getMuJoCoModel();
    const mjData *data = engine->getMuJoCoData(instance);
    mjtNum total[3] = {0, 0, 0};
    for (int c = 0; c < data->ncon; c++) {
        const mjContact &contact = data->contact[c];
        if (contact.efc_address < 0) {
            continue;
        }
        const bool geom_0_on_tool = body_is_on_tool[model->geom_bodyid[contact.geom[0]]];
        const bool geom_1_on_tool = body_is_on_tool[model->geom_bodyid[contact.geom[1]]];
        if (geom_0_on_tool == geom_1_on_tool) {
            continue;
        }
        // MuJoCo gives the force that geom 0 exerts on geom 1, in the contact frame
        mjtNum contact_force[6];
        mj_contactForce(model, data, c, contact_force);
        mjtNum world_force[3];
        mju_mulMatTVec3(world_force, contact.frame, contact_force);
        const mjtNum sign = geom_1_on_tool ? 1. : -1.;
        for (int k = 0; k < 3; k++) {
            total[k] += sign * world_force[k];
        }
    }
    return make_vec3(float(total[0]), float(total[1]), float(total[2]));
}

RobotInverseKinematicsResult Robot::solveInverseKinematics(const vec3 &target_position, const vec3 &target_axis, const std::vector<std::string> &locked_joints, uint instance) const {
    requireInitialized("solveInverseKinematics");
    if (tool_body_ID < 0) {
        helios_runtime_error("ERROR (Robot::solveInverseKinematics): The robot has no tool. Call setTool() before PhysicsEngine::initialize().");
    }
    if (target_axis.magnitude() < 1e-6f) {
        helios_runtime_error("ERROR (Robot::solveInverseKinematics): The target axis must not be a zero vector.");
    }
    vec3 target_direction = target_axis;
    target_direction.normalize();

    std::vector<bool> locked(joints.size(), false);
    for (const std::string &name: locked_joints) {
        locked[findJoint(name, "solveInverseKinematics")] = true;
    }

    const mjModel *model = engine->getMuJoCoModel();
    const mjData *simulation_data = engine->getMuJoCoData(instance);
    mjData *data = kinematics_data;
    mju_copy(data->qpos, simulation_data->qpos, model->nq);

    constexpr int max_iterations = 200;
    constexpr double position_tolerance = 1e-4; // m
    constexpr double axis_tolerance = 1e-3; // sine of the angle
    constexpr double damping = 0.01; // regularizes the solve near singular poses
    constexpr double max_step = 0.2; // largest change of one joint per iteration, in radians or meters
    constexpr double stall_improvement = 1e-5; // an iteration reducing the error by less than this has stalled
    constexpr double singularity_nudge = 0.1; // change of each hinge when stalled, in radians
    constexpr int max_nudges = 5;

    const int dof_count = static_cast<int>(model->nv);
    std::vector<mjtNum> jacobian_position(3 * dof_count);
    std::vector<mjtNum> jacobian_rotation(3 * dof_count);
    const mjtNum point_in_body[3] = {tool_point.x, tool_point.y, tool_point.z};
    const mjtNum axis_in_body[3] = {tool_axis.x, tool_axis.y, tool_axis.z};
    const mjtNum target[3] = {target_position.x, target_position.y, target_position.z};
    const mjtNum direction[3] = {target_direction.x, target_direction.y, target_direction.z};

    RobotInverseKinematicsResult result;
    const size_t joint_count = joints.size();
    std::vector<mjtNum> jacobian(6 * joint_count);
    std::vector<mjtNum> steps(joint_count);
    std::vector<mjtNum> accepted_positions(joint_count);
    for (size_t j = 0; j < joint_count; j++) {
        accepted_positions[j] = data->qpos[joints[j].qpos_address];
    }
    double accepted_error = std::numeric_limits<double>::max();
    double line_search_scale = 1.;
    double stalled_error = std::numeric_limits<double>::max();
    int nudges = 0;

    auto setJoints = [&](const std::vector<mjtNum> &positions) {
        for (size_t j = 0; j < joint_count; j++) {
            data->qpos[joints[j].qpos_address] = positions[j];
        }
    };
    auto clipToRange = [&](size_t j, mjtNum position) {
        if (model->jnt_limited[joints[j].joint_ID]) {
            return mju_clip(position, model->jnt_range[2 * joints[j].joint_ID], model->jnt_range[2 * joints[j].joint_ID + 1]);
        }
        return position;
    };

    for (int iteration = 0; iteration <= max_iterations; iteration++) {
        mj_kinematics(model, data);
        mj_comPos(model, data);

        mjtNum point_world[3], axis_world[3];
        mju_mulMatVec3(point_world, data->xmat + 9 * tool_body_ID, point_in_body);
        mju_addTo3(point_world, data->xpos + 3 * tool_body_ID);
        mju_mulMatVec3(axis_world, data->xmat + 9 * tool_body_ID, axis_in_body);

        // Six-element error: position, then the rotation that would turn the tool axis onto the target axis
        mjtNum error[6];
        mju_sub3(error, target, point_world);
        mju_cross(error + 3, axis_world, direction);
        const bool axis_opposite = mju_dot3(axis_world, direction) < 0.;
        const double total_error = mju_norm(error, 6) + (axis_opposite ? 1. : 0.);

        if (mju_norm3(error) < position_tolerance && mju_norm3(error + 3) < axis_tolerance && !axis_opposite) {
            result.converged = true;
            result.position_error = float(mju_norm3(error));
            result.axis_error_degrees = rad2deg(float(std::atan2(mju_norm3(error + 3), mju_dot3(axis_world, direction))));
            accepted_error = total_error;
            for (size_t j = 0; j < joint_count; j++) {
                accepted_positions[j] = data->qpos[joints[j].qpos_address];
            }
            break;
        }
        if (iteration == max_iterations) {
            break;
        }

        // A step that made the error worse is taken back and tried at half the length
        if (total_error >= accepted_error && line_search_scale > 1e-3) {
            line_search_scale *= 0.5;
            for (size_t j = 0; j < joint_count; j++) {
                data->qpos[joints[j].qpos_address] = clipToRange(j, accepted_positions[j] + line_search_scale * steps[j]);
            }
            continue;
        }

        if (total_error < accepted_error) {
            accepted_error = total_error;
            result.position_error = float(mju_norm3(error));
            result.axis_error_degrees = rad2deg(float(std::atan2(mju_norm3(error + 3), mju_dot3(axis_world, direction))));
            for (size_t j = 0; j < joint_count; j++) {
                accepted_positions[j] = data->qpos[joints[j].qpos_address];
            }
        } else {
            // No length of this step helped. Carry on from the best pose found.
            setJoints(accepted_positions);
            mj_kinematics(model, data);
            mj_comPos(model, data);
        }
        line_search_scale = 1.;

        // At a singular pose the Jacobian has no component along the remaining error, so the solve makes no progress however
        // far the target is. Moving the hinges changes the pose, so that it can. Joints alternate in sign to bend rather than swing a straight chain.
        if (stalled_error - accepted_error < stall_improvement && nudges < max_nudges) {
            nudges++;
            stalled_error = std::numeric_limits<double>::max();
            accepted_error = std::numeric_limits<double>::max();
            for (size_t j = 0; j < joint_count; j++) {
                if (!locked[j] && joints[j].is_hinge) {
                    data->qpos[joints[j].qpos_address] = clipToRange(j, accepted_positions[j] + (j % 2 == 0 ? singularity_nudge : -singularity_nudge));
                }
            }
            continue;
        }
        stalled_error = accepted_error;

        mju_mulMatVec3(point_world, data->xmat + 9 * tool_body_ID, point_in_body);
        mju_addTo3(point_world, data->xpos + 3 * tool_body_ID);
        mju_mulMatVec3(axis_world, data->xmat + 9 * tool_body_ID, axis_in_body);
        mju_sub3(error, target, point_world);
        mju_cross(error + 3, axis_world, direction);

        mj_jac(model, data, jacobian_position.data(), jacobian_rotation.data(), point_world, tool_body_ID);
        for (size_t j = 0; j < joint_count; j++) {
            const int dof = joints[j].dof_address;
            for (int r = 0; r < 3; r++) {
                jacobian[r * joint_count + j] = locked[j] ? 0. : jacobian_position[r * dof_count + dof];
                jacobian[(3 + r) * joint_count + j] = locked[j] ? 0. : jacobian_rotation[r * dof_count + dof];
            }
        }

        // Damped least squares: dq = J^T (J J^T + damping^2 I)^-1 error
        mjtNum normal_matrix[36] = {0};
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                for (size_t j = 0; j < joint_count; j++) {
                    normal_matrix[6 * r + c] += jacobian[r * joint_count + j] * jacobian[c * joint_count + j];
                }
            }
            normal_matrix[6 * r + r] += damping * damping;
        }
        mjtNum weights[6];
        mju_cholFactor(normal_matrix, 6, 0.);
        mju_cholSolve(weights, normal_matrix, error, 6);

        // The whole step is scaled to the limit on the largest joint change, since limiting joints separately would change its direction
        mjtNum largest_step = 0;
        for (size_t j = 0; j < joint_count; j++) {
            steps[j] = 0;
            for (int r = 0; r < 6; r++) {
                steps[j] += jacobian[r * joint_count + j] * weights[r];
            }
            largest_step = std::max(largest_step, mju_abs(steps[j]));
        }
        const mjtNum step_scale = largest_step > max_step ? max_step / largest_step : 1.;
        for (size_t j = 0; j < joint_count; j++) {
            steps[j] *= step_scale;
            data->qpos[joints[j].qpos_address] = clipToRange(j, accepted_positions[j] + steps[j]);
        }
    }

    // The last pose tried may be worse than the best one found
    if (!result.converged) {
        setJoints(accepted_positions);
    }

    result.joint_positions.resize(joint_count);
    for (size_t j = 0; j < joint_count; j++) {
        result.joint_positions[j] = toUserUnits(joints[j], data->qpos[joints[j].qpos_address]);
    }
    return result;
}
