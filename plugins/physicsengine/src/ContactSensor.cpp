/** \file "ContactSensor.cpp" Contact force on part of a simulation, split by what it touches.

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

#include "ContactSensor.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {
    const int ground_body_ID = 0; // the ground plane is a geom of MuJoCo's world body
} // namespace

ContactSensor::ContactSensor(const PhysicsEngine *engine, const std::string &body_name) : engine(engine) {
    if (engine == nullptr) {
        helios_runtime_error("ERROR (ContactSensor): The PhysicsEngine pointer must not be null.");
    }
    const mjModel *model = engine->getMuJoCoModel();
    if (model == nullptr) {
        helios_runtime_error("ERROR (ContactSensor): The simulation has not been initialized. Create the sensor after PhysicsEngine::initialize().");
    }
    const int sensed_root = mj_name2id(model, mjOBJ_BODY, body_name.c_str());
    if (sensed_root < 0) {
        helios_runtime_error("ERROR (ContactSensor): Body " + body_name + " is not in the model. Body names include the prefix given to PhysicsEngine::addModelFile().");
    }

    // A body's parent always has a lower ID, so one pass in order finds all descendants
    body_is_sensed.assign(model->nbody, 0);
    body_is_sensed[sensed_root] = 1;
    for (int body_ID = sensed_root + 1; body_ID < model->nbody; body_ID++) {
        body_is_sensed[body_ID] = body_is_sensed[model->body_parentid[body_ID]];
    }

    auto partnerIndex = [&](const std::string &name) {
        auto existing = std::find(partner_names.begin(), partner_names.end(), name);
        if (existing != partner_names.end()) {
            return static_cast<int>(existing - partner_names.begin());
        }
        partner_names.push_back(name);
        return static_cast<int>(partner_names.size() - 1);
    };

    const int other = partnerIndex("other");
    body_partner.assign(model->nbody, other);
    body_partner[ground_body_ID] = partnerIndex("ground");
    target_partner = static_cast<size_t>(partnerIndex("target"));
    for (uint articulationID = 0; articulationID < engine->getArticulationCount(); articulationID++) {
        for (uint link = 0; link < engine->getArticulationLinkCount(articulationID); link++) {
            const std::string label = engine->getLinkLabel(articulationID, link);
            body_partner[engine->getLinkBodyID(articulationID, link)] = partnerIndex(label.empty() ? "articulation" : label);
        }
    }
    if (body_is_sensed[ground_body_ID]) {
        helios_runtime_error("ERROR (ContactSensor): Body " + body_name + " is the world body. Sense a body of a loaded model.");
    }
    for (int body_ID = 0; body_ID < model->nbody; body_ID++) {
        if (body_is_sensed[body_ID] && body_partner[body_ID] != other) {
            helios_runtime_error("ERROR (ContactSensor): Body " + body_name + " has articulation links below it. Sense a body of a loaded model.");
        }
    }
    prepareInstances();
}

void ContactSensor::prepareInstances() {
    target_body_IDs.resize(engine->getInstanceCount(), -1);
}

uint ContactSensor::getPreparedInstanceCount() const {
    return static_cast<uint>(target_body_IDs.size());
}

void ContactSensor::requireInstance(uint instance, const std::string &method) const {
    if (instance >= target_body_IDs.size()) {
        if (instance < engine->getInstanceCount()) {
            helios_runtime_error("ERROR (ContactSensor::" + method + "): Instance " + std::to_string(instance) + " was created after the sensor. Call prepareInstances() after creating instances.");
        }
        helios_runtime_error("ERROR (ContactSensor::" + method + "): Instance " + std::to_string(instance) + " does not exist.");
    }
}

void ContactSensor::setTargetLink(uint articulationID, uint link_index, uint instance) {
    requireInstance(instance, "setTargetLink");
    target_body_IDs[instance] = engine->getLinkBodyID(articulationID, link_index);
}

void ContactSensor::clearTargetLink(uint instance) {
    requireInstance(instance, "clearTargetLink");
    target_body_IDs[instance] = -1;
}

const std::vector<std::string> &ContactSensor::getPartnerNames() const {
    return partner_names;
}

size_t ContactSensor::findPartner(const std::string &partner, const std::string &method) const {
    auto found = std::find(partner_names.begin(), partner_names.end(), partner);
    if (found == partner_names.end()) {
        std::string names;
        for (const std::string &name: partner_names) {
            names += (names.empty() ? "" : ", ") + name;
        }
        helios_runtime_error("ERROR (ContactSensor::" + method + "): Unknown partner \"" + partner + "\". The partners of this sensor are: " + names + ".");
    }
    return static_cast<size_t>(found - partner_names.begin());
}

std::vector<char> ContactSensor::excludedPartnerMask(const std::vector<std::string> &excluded_partners, const std::string &method) const {
    std::vector<char> excluded(partner_names.size(), 0);
    for (const std::string &partner: excluded_partners) {
        excluded[findPartner(partner, method)] = 1;
    }
    return excluded;
}

int ContactSensor::partnerOfBody(int body_ID, uint instance) const {
    return body_ID == target_body_IDs[instance] ? static_cast<int>(target_partner) : body_partner[body_ID];
}

template<typename Function>
void ContactSensor::forEachSensedContact(uint instance, Function function) const {
    const mjModel *model = engine->getMuJoCoModel();
    const mjData *data = engine->getMuJoCoData(instance);
    for (int c = 0; c < data->ncon; c++) {
        const mjContact &contact = data->contact[c];
        const int body_0 = model->geom_bodyid[contact.geom[0]];
        const int body_1 = model->geom_bodyid[contact.geom[1]];
        if (body_is_sensed[body_0] == body_is_sensed[body_1]) {
            continue;
        }
        // MuJoCo gives the force that geom 0 exerts on geom 1
        if (body_is_sensed[body_1]) {
            function(c, partnerOfBody(body_0, instance), 1.);
        } else {
            function(c, partnerOfBody(body_1, instance), -1.);
        }
    }
}

std::vector<vec3> ContactSensor::getForces(uint instance) const {
    requireInstance(instance, "getForces");
    const mjModel *model = engine->getMuJoCoModel();
    const mjData *data = engine->getMuJoCoData(instance);
    std::vector<mjtNum> totals(3 * partner_names.size(), 0.);
    forEachSensedContact(instance, [&](int c, int partner, mjtNum sign) {
        if (data->contact[c].efc_address < 0) {
            return;
        }
        mjtNum contact_force[6];
        mj_contactForce(model, data, c, contact_force);
        mjtNum world_force[3];
        mju_mulMatTVec3(world_force, data->contact[c].frame, contact_force);
        for (int k = 0; k < 3; k++) {
            totals[3 * partner + k] += sign * world_force[k];
        }
    });
    std::vector<vec3> forces(partner_names.size());
    for (size_t p = 0; p < partner_names.size(); p++) {
        forces[p] = make_vec3(float(totals[3 * p]), float(totals[3 * p + 1]), float(totals[3 * p + 2]));
    }
    return forces;
}

vec3 ContactSensor::getForce(const std::string &partner, uint instance) const {
    const size_t index = findPartner(partner, "getForce");
    return getForces(instance)[index];
}

vec3 ContactSensor::getTotalForce(const std::vector<std::string> &excluded_partners, uint instance) const {
    requireInstance(instance, "getTotalForce");
    const std::vector<char> excluded = excludedPartnerMask(excluded_partners, "getTotalForce");
    const mjModel *model = engine->getMuJoCoModel();
    const mjData *data = engine->getMuJoCoData(instance);
    mjtNum total[3] = {0, 0, 0};
    forEachSensedContact(instance, [&](int c, int partner, mjtNum sign) {
        if (excluded[partner] || data->contact[c].efc_address < 0) {
            return;
        }
        mjtNum contact_force[6];
        mj_contactForce(model, data, c, contact_force);
        mjtNum world_force[3];
        mju_mulMatTVec3(world_force, data->contact[c].frame, contact_force);
        for (int k = 0; k < 3; k++) {
            total[k] += sign * world_force[k];
        }
    });
    return make_vec3(float(total[0]), float(total[1]), float(total[2]));
}

bool ContactSensor::isPenetrating(const std::vector<std::string> &excluded_partners, uint instance) const {
    requireInstance(instance, "isPenetrating");
    const std::vector<char> excluded = excludedPartnerMask(excluded_partners, "isPenetrating");
    const mjData *data = engine->getMuJoCoData(instance);
    bool penetrating = false;
    forEachSensedContact(instance, [&](int c, int partner, mjtNum) {
        if (!excluded[partner] && data->contact[c].dist < 0) {
            penetrating = true;
        }
    });
    return penetrating;
}
