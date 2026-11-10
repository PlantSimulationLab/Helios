/** \file "ControlLoop.cpp" Runs the physics steps of each control step of a robot task, with the components that act inside them.

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

#include "ControlLoop.h"

using namespace helios;

ControlLoop::ControlLoop(PhysicsEngine *engine, uint physics_steps_per_control) : engine(engine), physics_steps_per_control(physics_steps_per_control) {
    if (engine == nullptr) {
        helios_runtime_error("ERROR (ControlLoop): The PhysicsEngine pointer must not be null.");
    }
    if (physics_steps_per_control == 0) {
        helios_runtime_error("ERROR (ControlLoop): A control step must have at least one physics step.");
    }
}

void ControlLoop::addComponent(RobotComponent *component) {
    if (component == nullptr) {
        helios_runtime_error("ERROR (ControlLoop::addComponent): The component pointer must not be null.");
    }
    if (std::find(components.begin(), components.end(), component) != components.end()) {
        helios_runtime_error("ERROR (ControlLoop::addComponent): The component was already added.");
    }
    components.push_back(component);
}

uint ControlLoop::getPhysicsStepsPerControl() const {
    return physics_steps_per_control;
}

float ControlLoop::getControlPeriod() const {
    return float(physics_steps_per_control) * engine->getTimestep();
}

void ControlLoop::prepareInstances() {
    for (RobotComponent *component: components) {
        component->prepareInstances();
    }
}

void ControlLoop::resetInstance(uint instance) {
    engine->resetInstance(instance);
    for (RobotComponent *component: components) {
        component->resetInstance(instance);
    }
}

ControlStepResult ControlLoop::step(uint instance) {
    const float dt = engine->getTimestep();
    for (RobotComponent *component: components) {
        component->beginControlStep(instance);
    }
    ControlStepResult result;
    while (result.physics_steps < physics_steps_per_control && !result.event) {
        for (RobotComponent *component: components) {
            component->beforePhysicsStep(instance, dt);
        }
        const uint detachments_before = engine->getLoadDetachmentCount(instance);
        engine->step(1, instance);
        result.physics_steps++;
        result.event = engine->getLoadDetachmentCount(instance) != detachments_before;
        // Every component sees every step, even one in which another reported an event
        for (RobotComponent *component: components) {
            if (component->afterPhysicsStep(instance)) {
                result.event = true;
            }
        }
    }
    return result;
}
