/** \file "RobotCamera.cpp" A camera that renders robot task scenes with the Visualizer.

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

#include "RobotCamera.h"

#include "Visualizer.h"

#include <mujoco/mujoco.h>

using namespace helios;

RobotCamera::RobotCamera(PhysicsEngine *engine, Context *context, uint width, uint height, const RobotCameraParameters &parameters) : engine(engine), context(context), width(width), height(height) {
    if (engine == nullptr || context == nullptr) {
        helios_runtime_error("ERROR (RobotCamera): The PhysicsEngine and Context pointers must not be null.");
    }
    if (engine->getMuJoCoModel() == nullptr) {
        helios_runtime_error("ERROR (RobotCamera): The simulation has not been initialized. Create the camera after PhysicsEngine::initialize().");
    }
    if (width == 0 || height == 0) {
        helios_runtime_error("ERROR (RobotCamera): The image must be at least one pixel wide and high.");
    }
    visualizer = std::make_unique<Visualizer>(width, height, parameters.antialiasing_samples, false, true);
    visualizer->disableMessages();
    visualizer->hideWatermark();
    visualizer->setBackgroundColor(parameters.background);
    visualizer->setLightingModel(parameters.shadows ? Visualizer::LIGHTING_PHONG_SHADOWED : Visualizer::LIGHTING_PHONG);
    visualizer->setLightDirection(parameters.light_direction);
    visualizer->setCameraFieldOfView(parameters.field_of_view);
    // The gizmo is rebuilt whenever the camera moves, which is every frame for a camera on a robot
    visualizer->hideNavigationGizmo();
    visualizer->buildContextGeometry(context);
}

RobotCamera::~RobotCamera() = default;

void RobotCamera::setPose(const vec3 &eye_position, const vec3 &look_at_point) {
    eye = eye_position;
    look_at = look_at_point;
    mount_body_ID = -1;
}

void RobotCamera::mountOnBody(const std::string &body_name, const vec3 &eye_in_body, const vec3 &look_at_in_body) {
    const int body_ID = mj_name2id(engine->getMuJoCoModel(), mjOBJ_BODY, body_name.c_str());
    if (body_ID < 0) {
        helios_runtime_error("ERROR (RobotCamera::mountOnBody): Body " + body_name + " is not in the model. Body names include the prefix of their model, such as the name of a robot.");
    }
    eye = eye_in_body;
    look_at = look_at_in_body;
    mount_body_ID = body_ID;
}

void RobotCamera::setFieldOfView(float degrees) {
    if (!(degrees > 0.f && degrees < 180.f)) {
        helios_runtime_error("ERROR (RobotCamera::setFieldOfView): The field of view must be between 0 and 180 degrees, but was " + std::to_string(degrees) + ".");
    }
    visualizer->setCameraFieldOfView(degrees);
}

void RobotCamera::getPose(uint instance, vec3 &eye_world, vec3 &look_at_world) const {
    if (mount_body_ID < 0) {
        eye_world = eye;
        look_at_world = look_at;
        return;
    }
    const mjData *data = engine->getMuJoCoData(instance);
    auto toWorld = [&](const vec3 &point) {
        const mjtNum local[3] = {point.x, point.y, point.z};
        mjtNum rotated[3];
        mju_mulMatVec3(rotated, data->xmat + 9 * mount_body_ID, local);
        const mjtNum *origin = data->xpos + 3 * mount_body_ID;
        return make_vec3(float(origin[0] + rotated[0]), float(origin[1] + rotated[1]), float(origin[2] + rotated[2]));
    };
    eye_world = toWorld(eye);
    look_at_world = toWorld(look_at);
}

void RobotCamera::prepare(uint instance) {
    engine->updateContextGeometry(instance);
    vec3 eye_world, look_at_world;
    getPose(instance, eye_world, look_at_world);
    visualizer->setCameraPosition(eye_world, look_at_world);
    visualizer->prepareContextGeometry();
    // Flags cleared after each frame report only what moves between frames, so the Visualizer does not check every primitive
    context->markGeometryClean();
}

void RobotCamera::draw() {
    visualizer->plotUpdate(true);
}

void RobotCamera::render(uint instance) {
    prepare(instance);
    draw();
}

uint RobotCamera::getWidth() const {
    return width;
}

uint RobotCamera::getHeight() const {
    return height;
}

void RobotCamera::getColorImage(unsigned char *rgb) const {
    std::vector<uint> pixels;
    uint frame_width = 0;
    uint frame_height = 0;
    visualizer->getWindowPixelsRGB(pixels, frame_width, frame_height);
    if (frame_width != width || frame_height != height) {
        helios_runtime_error("ERROR (RobotCamera::getColorImage): The frame is " + std::to_string(frame_width) + "x" + std::to_string(frame_height) + " pixels rather than the " + std::to_string(width) + "x" + std::to_string(height) + " the camera was made with.");
    }
    // OpenGL gives the bottom row first
    for (uint row = 0; row < height; row++) {
        const uint *source = pixels.data() + size_t(height - 1 - row) * size_t(width) * 3;
        unsigned char *destination = rgb + size_t(row) * size_t(width) * 3;
        for (uint k = 0; k < width * 3; k++) {
            destination[k] = static_cast<unsigned char>(source[k]);
        }
    }
}

void RobotCamera::getDepthImage(float *depth) {
    std::vector<float> pixels;
    uint frame_width = 0;
    uint frame_height = 0;
    visualizer->getDepthImage(pixels, frame_width, frame_height);
    std::copy(pixels.begin(), pixels.end(), depth);
}

void RobotCamera::getPrimitiveImage(uint *primitive_UUIDs) {
    std::vector<uint> pixels;
    uint frame_width = 0;
    uint frame_height = 0;
    visualizer->getPixelUUIDs(pixels, frame_width, frame_height);
    std::copy(pixels.begin(), pixels.end(), primitive_UUIDs);
}

void RobotCamera::getObjectImage(uint *objIDs) {
    std::vector<uint> pixels;
    uint frame_width = 0;
    uint frame_height = 0;
    visualizer->getPixelUUIDs(pixels, frame_width, frame_height);
    for (size_t k = 0; k < pixels.size(); k++) {
        objIDs[k] = pixels[k] == 0 || !context->doesPrimitiveExist(pixels[k] - 1) ? 0 : context->getPrimitiveParentObjectID(pixels[k] - 1);
    }
}

void RobotCamera::saveImage(const std::string &filename, const std::string &format) {
    visualizer->printWindow(filename.c_str(), format);
}
