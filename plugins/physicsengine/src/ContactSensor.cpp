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
    const int world_body_ID = 0; // MuJoCo's world body, which carries the ground plane and any links welded to the world
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

    // Partners are told apart by geom, since one body can carry several links (PhysicsEngineParameters::merge_welded_links), and the world body carries the ground as well
    const int other = partnerIndex("other");
    const int ground = partnerIndex("ground");
    target_partner = static_cast<size_t>(partnerIndex("target"));
    link_key_base.resize(engine->getArticulationCount());
    uint key_base = 0;
    for (uint articulationID = 0; articulationID < engine->getArticulationCount(); articulationID++) {
        link_key_base[articulationID] = key_base;
        key_base += engine->getArticulationLinkCount(articulationID);
        // Every label is a partner, including those of links without collision shapes, so the partners do not depend on the shapes
        for (uint link = 0; link < engine->getArticulationLinkCount(articulationID); link++) {
            const std::string label = engine->getLinkLabel(articulationID, link);
            static_cast<void>(partnerIndex(label.empty() ? "articulation" : label));
        }
    }
    geom_partner.assign(model->ngeom, other);
    geom_link_key.assign(model->ngeom, -1);
    for (int geom_ID = 0; geom_ID < model->ngeom; geom_ID++) {
        uint articulationID = 0;
        uint link = 0;
        if (engine->getGeomLink(geom_ID, articulationID, link)) {
            const std::string label = engine->getLinkLabel(articulationID, link);
            geom_partner[geom_ID] = partnerIndex(label.empty() ? "articulation" : label);
            geom_link_key[geom_ID] = static_cast<int>(link_key_base[articulationID] + link);
            if (body_is_sensed[model->geom_bodyid[geom_ID]]) {
                helios_runtime_error("ERROR (ContactSensor): Body " + body_name + " has articulation links below it. Sense a body of a loaded model.");
            }
        } else if (model->geom_bodyid[geom_ID] == world_body_ID) {
            geom_partner[geom_ID] = ground;
        }
    }
    if (body_is_sensed[world_body_ID]) {
        helios_runtime_error("ERROR (ContactSensor): Body " + body_name + " is the world body. Sense a body of a loaded model.");
    }
    prepareInstances();
}

void ContactSensor::prepareInstances() {
    target_link_keys.resize(engine->getInstanceCount(), -1);
}

uint ContactSensor::getPreparedInstanceCount() const {
    return static_cast<uint>(target_link_keys.size());
}

void ContactSensor::requireInstance(uint instance, const std::string &method) const {
    if (instance >= target_link_keys.size()) {
        if (instance < engine->getInstanceCount()) {
            helios_runtime_error("ERROR (ContactSensor::" + method + "): Instance " + std::to_string(instance) + " was created after the sensor. Call prepareInstances() after creating instances.");
        }
        helios_runtime_error("ERROR (ContactSensor::" + method + "): Instance " + std::to_string(instance) + " does not exist.");
    }
}

void ContactSensor::setTargetLink(uint articulationID, uint link_index, uint instance) {
    requireInstance(instance, "setTargetLink");
    if (articulationID >= link_key_base.size() || link_index >= engine->getArticulationLinkCount(articulationID)) {
        helios_runtime_error("ERROR (ContactSensor::setTargetLink): Link " + std::to_string(link_index) + " of articulation " + std::to_string(articulationID) + " does not exist.");
    }
    target_link_keys[instance] = static_cast<int>(link_key_base[articulationID] + link_index);
}

void ContactSensor::clearTargetLink(uint instance) {
    requireInstance(instance, "clearTargetLink");
    target_link_keys[instance] = -1;
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

int ContactSensor::partnerOfGeom(int geom_ID, uint instance) const {
    const int key = geom_link_key[geom_ID];
    return key >= 0 && key == target_link_keys[instance] ? static_cast<int>(target_partner) : geom_partner[geom_ID];
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
            function(c, partnerOfGeom(contact.geom[0], instance), 1.);
        } else {
            function(c, partnerOfGeom(contact.geom[1], instance), -1.);
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
