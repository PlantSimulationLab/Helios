/** \file "VacuumCup.cpp" A vacuum cup end effector that seals on round objects such as fruit.

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

#include "VacuumCup.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {
    vec3 toVec3(const mjtNum *p) {
        return make_vec3(float(p[0]), float(p[1]), float(p[2]));
    }
} // namespace

VacuumCup::VacuumCup(Robot *robot, const std::string &body_name, const vec3 &face_center, const vec3 &face_normal, const VacuumCupParameters &parameters) : robot(robot), face_center(face_center), face_normal(face_normal), parameters(parameters) {
    if (robot == nullptr) {
        helios_runtime_error("ERROR (VacuumCup): The Robot pointer must not be null.");
    }
    engine = robot->getPhysicsEngine();
    const mjModel *model = engine->getMuJoCoModel();
    if (model == nullptr) {
        helios_runtime_error("ERROR (VacuumCup): The simulation has not been initialized. Create the cup after PhysicsEngine::initialize().");
    }
    body_ID = mj_name2id(model, mjOBJ_BODY, (robot->getName() + "/" + body_name).c_str());
    if (body_ID < 0) {
        helios_runtime_error("ERROR (VacuumCup): Body " + body_name + " is not in the model of robot " + robot->getName() + ".");
    }
    if (face_normal.magnitude() < 1e-6f) {
        helios_runtime_error("ERROR (VacuumCup): The face normal must not be a zero vector.");
    }
    this->face_normal.normalize();
    const VacuumCupParameters &p = parameters;
    if (!(p.lip_radius > 0.f) || !(p.lip_thickness > 0.f) || !(p.seal_reach >= 0.f) || !(p.stiffness > 0.f) || !(p.damping >= 0.f) || !(p.ramp_seconds >= 0.f) || !(p.pull_limit > 0.f) || !(p.shear_limit > 0.f) || !(p.moment_limit > 0.f)) {
        helios_runtime_error("ERROR (VacuumCup): The lip radius and thickness, stiffness, and seal limits must be positive, and the seal reach, damping, and ramp time must not be negative.");
    }
    prepareInstances();
}

void VacuumCup::prepareInstances() {
    instances.resize(std::max(instances.size(), static_cast<size_t>(engine->getInstanceCount())));
}

const VacuumCup::InstanceState &VacuumCup::state(uint instance, const std::string &method) const {
    if (instance >= instances.size()) {
        if (instance < engine->getInstanceCount()) {
            helios_runtime_error("ERROR (VacuumCup::" + method + "): Instance " + std::to_string(instance) + " was created after the cup. Call prepareInstances() after creating instances.");
        }
        helios_runtime_error("ERROR (VacuumCup::" + method + "): Instance " + std::to_string(instance) + " does not exist.");
    }
    return instances[instance];
}

VacuumCup::InstanceState &VacuumCup::state(uint instance, const std::string &method) {
    return const_cast<InstanceState &>(static_cast<const VacuumCup *>(this)->state(instance, method));
}

const VacuumCup::InstanceState &VacuumCup::targetState(uint instance, const std::string &method) const {
    const InstanceState &cup = state(instance, method);
    if (cup.target_body_ID < 0) {
        helios_runtime_error("ERROR (VacuumCup::" + method + "): Instance " + std::to_string(instance) + " has no target. Call setTarget() first.");
    }
    return cup;
}

void VacuumCup::clearAppliedForces(const InstanceState &cup, uint instance) {
    mjData *data = engine->getMuJoCoData(instance);
    mju_zero(data->xfrc_applied + 6 * body_ID, 6);
    if (cup.target_body_ID >= 0) {
        mju_zero(data->xfrc_applied + 6 * cup.target_body_ID, 6);
    }
}

void VacuumCup::resetInstance(uint instance) {
    InstanceState &cup = state(instance, "resetInstance");
    clearAppliedForces(cup, instance);
    const int target_body_ID = cup.target_body_ID;
    const float target_radius = cup.target_radius;
    cup = InstanceState();
    cup.target_body_ID = target_body_ID;
    cup.target_radius = target_radius;
}

void VacuumCup::setTarget(uint articulationID, uint link_index, uint instance) {
    InstanceState &cup = state(instance, "setTarget");
    const int target_body_ID = engine->getLinkBodyID(articulationID, link_index);
    const mjModel *model = engine->getMuJoCoModel();
    if (model->body_geomnum[target_body_ID] < 1 || model->geom_type[model->body_geomadr[target_body_ID]] != mjGEOM_SPHERE) {
        helios_runtime_error("ERROR (VacuumCup::setTarget): Link " + std::to_string(link_index) + " of articulation " + std::to_string(articulationID) + " does not have a sphere as its first collision shape, so the cup cannot seal on it.");
    }
    release(instance);
    cup.target_body_ID = target_body_ID;
    cup.target_radius = float(model->geom_size[3 * model->body_geomadr[target_body_ID]]);
}

vec3 VacuumCup::getFacePosition(uint instance) const {
    static_cast<void>(state(instance, "getFacePosition"));
    const mjData *data = engine->getMuJoCoData(instance);
    const mjtNum point[3] = {face_center.x, face_center.y, face_center.z};
    mjtNum rotated[3];
    mju_mulMatVec3(rotated, data->xmat + 9 * body_ID, point);
    const mjtNum *origin = data->xpos + 3 * body_ID;
    return make_vec3(float(origin[0] + rotated[0]), float(origin[1] + rotated[1]), float(origin[2] + rotated[2]));
}

vec3 VacuumCup::getAxis(uint instance) const {
    static_cast<void>(state(instance, "getAxis"));
    const mjtNum normal[3] = {face_normal.x, face_normal.y, face_normal.z};
    mjtNum rotated[3];
    mju_mulMatVec3(rotated, engine->getMuJoCoData(instance)->xmat + 9 * body_ID, normal);
    return toVec3(rotated);
}

vec3 VacuumCup::getTargetPosition(uint instance) const {
    const InstanceState &cup = targetState(instance, "getTargetPosition");
    return toVec3(engine->getMuJoCoData(instance)->xipos + 3 * cup.target_body_ID);
}

vec3 VacuumCup::getOffAxis(uint instance) const {
    const vec3 axis = getAxis(instance);
    const vec3 to_target = getTargetPosition(instance) - getFacePosition(instance);
    return to_target - (to_target * axis) * axis;
}

float VacuumCup::getSealGap(uint instance) const {
    const InstanceState &cup = targetState(instance, "getSealGap");
    const vec3 to_target = getTargetPosition(instance) - getFacePosition(instance);
    const float along_axis = to_target * getAxis(instance);
    const float off_axis = getOffAxis(instance).magnitude();
    // The farthest point of the lip is on the side away from the target's center
    const float lip_to_axis = parameters.lip_radius + off_axis;
    return std::sqrt(along_axis * along_axis + lip_to_axis * lip_to_axis) - cup.target_radius - parameters.lip_thickness;
}

float VacuumCup::getSeatDistance(uint instance) const {
    const InstanceState &cup = targetState(instance, "getSeatDistance");
    const float reach = cup.target_radius + parameters.lip_thickness;
    return std::sqrt(std::max(0.f, reach * reach - parameters.lip_radius * parameters.lip_radius));
}

bool VacuumCup::grip(uint instance) {
    InstanceState &cup = state(instance, "grip");
    if (cup.gripped) {
        return true;
    }
    if (getSealGap(instance) >= parameters.seal_reach) {
        return false;
    }
    // Hold the target seated in the mouth, with its center on the cup's axis and its surface on the whole lip, and its orientation as it is now
    const mjData *data = engine->getMuJoCoData(instance);
    cup.grip_seat_distance = getSeatDistance(instance);
    cup.grip_center[0] = face_center.x + cup.grip_seat_distance * face_normal.x;
    cup.grip_center[1] = face_center.y + cup.grip_seat_distance * face_normal.y;
    cup.grip_center[2] = face_center.z + cup.grip_seat_distance * face_normal.z;
    cup.grip_normal[0] = face_normal.x;
    cup.grip_normal[1] = face_normal.y;
    cup.grip_normal[2] = face_normal.z;
    mjtNum cup_inverse[4];
    mju_negQuat(cup_inverse, data->xquat + 4 * body_ID);
    mju_mulQuat(cup.grip_orientation, cup_inverse, data->xquat + 4 * cup.target_body_ID);
    cup.gripped = true;
    cup.seal_broken = false;
    cup.grip_seconds = 0.f;
    return true;
}

void VacuumCup::release(uint instance) {
    InstanceState &cup = state(instance, "release");
    clearAppliedForces(cup, instance);
    cup.gripped = false;
}

void VacuumCup::beforePhysicsStep(uint instance, float dt) {
    applyForces(dt, instance);
}

bool VacuumCup::afterPhysicsStep(uint instance) {
    InstanceState &cup = state(instance, "afterPhysicsStep");
    if (!cup.gripped || !isSealOverloaded(instance)) {
        return false;
    }
    release(instance);
    cup.seal_broken = true;
    return true;
}

bool VacuumCup::isSealBroken(uint instance) const {
    return state(instance, "isSealBroken").seal_broken;
}

void VacuumCup::applyForces(float dt, uint instance) {
    InstanceState &cup = state(instance, "applyForces");
    clearAppliedForces(cup, instance);
    cup.pull = 0.f;
    cup.shear = 0.f;
    cup.moment = 0.f;
    cup.force_on_target = make_vec3(0, 0, 0);
    if (!cup.gripped) {
        return;
    }
    const mjModel *model = engine->getMuJoCoModel();
    mjData *data = engine->getMuJoCoData(instance);
    const int target_body = cup.target_body_ID;
    cup.grip_seconds += dt;
    const float ramp = parameters.ramp_seconds > 0.f ? std::min(1.f, cup.grip_seconds / parameters.ramp_seconds) : 1.f;

    // Where the target should be, and how fast that point moves with the cup
    mjtNum offset[3], normal_world[3];
    mju_mulMatVec3(offset, data->xmat + 9 * body_ID, cup.grip_center);
    mju_mulMatVec3(normal_world, data->xmat + 9 * body_ID, cup.grip_normal);
    const vec3 held = toVec3(data->xpos + 3 * body_ID) + toVec3(offset);
    const vec3 normal = toVec3(normal_world);
    mjtNum velocity_target[6], velocity_cup[6];
    mj_objectVelocity(model, data, mjOBJ_BODY, target_body, velocity_target, 0);
    mj_objectVelocity(model, data, mjOBJ_BODY, body_ID, velocity_cup, 0);
    const vec3 cup_center = toVec3(data->xipos + 3 * body_ID);
    const vec3 cup_spin = toVec3(velocity_cup);
    const vec3 held_velocity = toVec3(velocity_cup + 3) + cross(cup_spin, held - cup_center);
    const vec3 target = toVec3(data->xipos + 3 * target_body);
    const vec3 on_target = ramp * (parameters.stiffness * (held - target) + parameters.damping * (held_velocity - toVec3(velocity_target + 3)));

    // The orientation the target should have, and the rotation from where it is to there. The rotational springs have the time constants of the linear ones for this target's inertia.
    mjtNum held_orientation[4], target_inverse[4], error[4], error_rotation[3];
    mju_mulQuat(held_orientation, data->xquat + 4 * body_ID, cup.grip_orientation);
    mju_negQuat(target_inverse, data->xquat + 4 * target_body);
    mju_mulQuat(error, held_orientation, target_inverse);
    mju_quat2Vel(error_rotation, error, 1.0);
    const float mass = float(model->body_mass[target_body]);
    const float inertia = float(model->body_inertia[3 * target_body] + model->body_inertia[3 * target_body + 1] + model->body_inertia[3 * target_body + 2]) / 3.f;
    const vec3 torque_on_target = ramp * (inertia / mass) * (parameters.stiffness * toVec3(error_rotation) + parameters.damping * (cup_spin - toVec3(velocity_target)));

    // The load on the seal: pulling the target toward the cup is a pull out of it, the rest is across the mouth, and the moment is about the mouth's center
    const vec3 along_face = on_target - (on_target * normal) * normal;
    const vec3 face = held - cup.grip_seat_distance * normal;
    cup.pull = std::max(0.f, -(on_target * normal));
    cup.shear = along_face.magnitude();
    cup.moment = (torque_on_target + cross(held - face, on_target)).magnitude();
    cup.force_on_target = on_target;

    const float target_load[6] = {on_target.x, on_target.y, on_target.z, torque_on_target.x, torque_on_target.y, torque_on_target.z};
    // The reaction acts where the target is held rather than at the cup body's center of mass, so it also carries a torque
    const vec3 on_cup = -1.f * on_target;
    const vec3 torque_on_cup = cross(held - cup_center, on_cup) - torque_on_target;
    const float cup_load[6] = {on_cup.x, on_cup.y, on_cup.z, torque_on_cup.x, torque_on_cup.y, torque_on_cup.z};
    for (int k = 0; k < 6; k++) {
        data->xfrc_applied[6 * target_body + k] = target_load[k];
        data->xfrc_applied[6 * body_ID + k] = cup_load[k];
    }
}

bool VacuumCup::isGripped(uint instance) const {
    return state(instance, "isGripped").gripped;
}

float VacuumCup::getSealLoad(uint instance) const {
    const InstanceState &cup = state(instance, "getSealLoad");
    return std::max({cup.pull / parameters.pull_limit, cup.shear / parameters.shear_limit, cup.moment / parameters.moment_limit});
}

bool VacuumCup::isSealOverloaded(uint instance) const {
    return getSealLoad(instance) > 1.f;
}

float VacuumCup::getPullForce(uint instance) const {
    return state(instance, "getPullForce").pull;
}

vec3 VacuumCup::getForceOnTarget(uint instance) const {
    return state(instance, "getForceOnTarget").force_on_target;
}

VacuumCupGripAction::VacuumCupGripAction(VacuumCup *cup, bool release_unless_positive) : cup(cup), release_unless_positive(release_unless_positive) {
    if (cup == nullptr) {
        helios_runtime_error("ERROR (VacuumCupGripAction): The VacuumCup pointer must not be null.");
    }
}

void VacuumCupGripAction::prepareInstances() {
}

void VacuumCupGripAction::resetInstance(uint instance) {
}

uint VacuumCupGripAction::getActionSize() const {
    return 1;
}

void VacuumCupGripAction::applyAction(const float *action, float dt, uint instance) {
    if (action[0] > 0.f) {
        static_cast<void>(cup->grip(instance));
    } else if (release_unless_positive && cup->isGripped(instance)) {
        cup->release(instance);
    }
}
