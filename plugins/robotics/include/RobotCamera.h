/** \file "RobotCamera.h" A camera that renders robot task scenes with the Visualizer. Only available when visualizer is in the project's list of plug-ins.

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

#ifndef HELIOS_ROBOT_CAMERA_H
#define HELIOS_ROBOT_CAMERA_H

#include "PhysicsEngine.h"

class Visualizer;

//! How a RobotCamera draws
struct RobotCameraParameters {
    //! Vertical field of view in degrees
    float field_of_view = 70.f;
    //! Samples per pixel for anti-aliasing. One gives none, and is fastest.
    int antialiasing_samples = 1;
    //! Color of the sky behind the scene
    helios::RGBcolor background = helios::make_RGBcolor(0.6f, 0.75f, 0.95f);
    //! Direction toward the light
    helios::vec3 light_direction = helios::make_vec3(0.3f, -0.5f, 0.8f);
    //! Cast shadows, which look better and draw more slowly
    bool shadows = false;
};

//! Renders the Context geometry of a simulation, as it stands in one instance, from a camera that is fixed or mounted on a body, such as a robot's wrist
/**
 * Each image gives color, depth, and which Context primitive and object is seen in each pixel, so that a task can use images as observations or measure what a camera sees, such as how much of a fruit is visible.
 * One camera draws one Context, and so the instances of one PhysicsEngine one at a time: each frame moves the geometry to the instance (PhysicsEngine::updateContextGeometry()) and draws it.
 *
 * OpenGL is tied to a thread: construct the camera, and call draw() and the image getters, on the thread that will draw. prepare() uses no OpenGL, so the cameras of different scenes can prepare in parallel, such as with
 * ParallelRobotSimulations::forEachIndex(), and then draw one after another. After each frame the Context's geometry flags are cleared, so that the next frame only redraws what moved; use only one camera per Context,
 * since each would otherwise miss motion the other drew.
 * Images are row-major with the first row at the top.
 */
class RobotCamera {
public:
    //! Constructor. Builds the camera's copy of the Context geometry, so add all geometry first.
    /**
     * \param[in] engine Initialized physics simulation whose Context geometry is drawn.
     * \param[in] context Context holding the geometry, the one given to the PhysicsEngine.
     * \param[in] width Width of the images in pixels.
     * \param[in] height Height of the images in pixels.
     * \param[in] parameters How the camera draws.
     */
    RobotCamera(PhysicsEngine *engine, helios::Context *context, uint width, uint height, const RobotCameraParameters &parameters = RobotCameraParameters());

    //! Destructor
    ~RobotCamera();

    RobotCamera(const RobotCamera &) = delete;
    RobotCamera &operator=(const RobotCamera &) = delete;

    //! Place the camera in the world, looking at a point. It stays there until placed or mounted again.
    /**
     * \param[in] eye Position of the camera.
     * \param[in] look_at Point the camera looks at.
     */
    void setPose(const helios::vec3 &eye, const helios::vec3 &look_at);

    //! Mount the camera on a body, so that it moves with the body in each instance
    /**
     * \param[in] body_name Name of the body in the compiled model, such as "robot/wrist" for a body of a robot named "robot".
     * \param[in] eye Position of the camera in the body's frame.
     * \param[in] look_at Point the camera looks at, in the body's frame.
     */
    void mountOnBody(const std::string &body_name, const helios::vec3 &eye, const helios::vec3 &look_at);

    //! Set the vertical field of view in degrees
    /**
     * \param[in] degrees Field of view.
     */
    void setFieldOfView(float degrees);

    //! Get where the camera is and what it looks at in an instance, in the world
    /**
     * \param[in] instance ID of the simulation instance, which matters for a mounted camera.
     * \param[out] eye Position of the camera.
     * \param[out] look_at Point the camera looks at.
     */
    void getPose(uint instance, helios::vec3 &eye, helios::vec3 &look_at) const;

    //! Move the geometry to an instance and place the camera for it, without drawing. Uses no OpenGL.
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void prepare(uint instance);

    //! Draw the frame prepared by prepare(). Call on the thread that constructed the camera.
    void draw();

    //! Prepare and draw a frame of an instance
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void render(uint instance);

    //! Get the width of the images in pixels
    [[nodiscard]] uint getWidth() const;

    //! Get the height of the images in pixels
    [[nodiscard]] uint getHeight() const;

    //! Get the color image of the last frame drawn
    /**
     * \param[out] rgb getHeight() rows of getWidth() pixels of 3 bytes, red, green, blue, the first row at the top.
     */
    void getColorImage(unsigned char *rgb) const;

    //! Get the distance from the camera to the surface seen in each pixel of the last frame, along the view direction
    /**
     * \param[out] depth getHeight() rows of getWidth() distances in meters, the first row at the top, and -1 where nothing is seen.
     */
    void getDepthImage(float *depth);

    //! Get the Context primitive seen in each pixel of the last frame
    /**
     * \param[out] primitive_UUIDs getHeight() rows of getWidth() values, the first row at the top: the UUID of the primitive plus one, or 0 where no primitive is seen.
     */
    void getPrimitiveImage(uint *primitive_UUIDs);

    //! Get the Context object seen in each pixel of the last frame, such as which fruit
    /**
     * \param[out] objIDs getHeight() rows of getWidth() values, the first row at the top: the ID of the object the primitive seen belongs to, or 0 where none is seen or the primitive belongs to no object.
     */
    void getObjectImage(uint *objIDs);

    //! Write the last frame to an image file, for looking at it
    /**
     * \param[in] filename Path of the file.
     * \param[in] format "jpeg" or "png".
     */
    void saveImage(const std::string &filename, const std::string &format = "jpeg");

private:
    PhysicsEngine *engine;
    helios::Context *context;
    std::unique_ptr<Visualizer> visualizer;
    uint width;
    uint height;
    helios::vec3 eye;
    helios::vec3 look_at;
    int mount_body_ID = -1;
};

#endif
