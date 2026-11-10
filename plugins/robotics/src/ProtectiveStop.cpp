/** \file "ProtectiveStop.cpp" Stops a robot that presses on its surroundings too hard.

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

#include "ProtectiveStop.h"

using namespace helios;

ProtectiveStop::ProtectiveStop(const ContactSensor *sensor, float force_limit, const std::vector<std::string> &ignored_partners) : sensor(sensor), force_limit(force_limit), ignored_partners(ignored_partners) {
    if (sensor == nullptr) {
        helios_runtime_error("ERROR (ProtectiveStop): The ContactSensor pointer must not be null.");
    }
    if (!(force_limit > 0.f)) {
        helios_runtime_error("ERROR (ProtectiveStop): The force limit must be positive, but was " + std::to_string(force_limit) + " N.");
    }
    const std::vector<std::string> &partners = sensor->getPartnerNames();
    for (const std::string &partner: ignored_partners) {
        if (std::find(partners.begin(), partners.end(), partner) == partners.end()) {
            helios_runtime_error("ERROR (ProtectiveStop): The sensor has no partner \"" + partner + "\" to ignore.");
        }
    }
    prepareInstances();
}

void ProtectiveStop::prepareInstances() {
    const size_t instance_count = std::max(stopped.size(), static_cast<size_t>(sensor->getPreparedInstanceCount()));
    stopped.resize(instance_count, 0);
    peak_forces.resize(instance_count, 0.f);
}

void ProtectiveStop::requireInstance(uint instance, const std::string &method) const {
    if (instance >= stopped.size()) {
        helios_runtime_error("ERROR (ProtectiveStop::" + method + "): Instance " + std::to_string(instance) + " does not exist, or was created after the stop. Call prepareInstances() on the sensor and then on the stop after creating instances.");
    }
}

void ProtectiveStop::resetInstance(uint instance) {
    requireInstance(instance, "resetInstance");
    stopped[instance] = 0;
    peak_forces[instance] = 0.f;
}

void ProtectiveStop::beginControlStep(uint instance) {
    requireInstance(instance, "beginControlStep");
    peak_forces[instance] = 0.f;
}

bool ProtectiveStop::afterPhysicsStep(uint instance) {
    const bool was_stopped = isStopped(instance);
    update(instance);
    return !was_stopped && stopped[instance];
}

float ProtectiveStop::getPeakForce(uint instance) const {
    requireInstance(instance, "getPeakForce");
    return peak_forces[instance];
}

float ProtectiveStop::measureForce(uint instance) const {
    requireInstance(instance, "measureForce");
    return sensor->getTotalForce(ignored_partners, instance).magnitude();
}

float ProtectiveStop::update(uint instance) {
    const float force = measureForce(instance);
    if (force > force_limit) {
        stopped[instance] = 1;
    }
    peak_forces[instance] = std::max(peak_forces[instance], force);
    return force;
}

bool ProtectiveStop::isStopped(uint instance) const {
    requireInstance(instance, "isStopped");
    return stopped[instance] != 0;
}

float ProtectiveStop::getForceLimit() const {
    return force_limit;
}
