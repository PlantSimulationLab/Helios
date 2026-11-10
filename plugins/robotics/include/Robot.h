/** \file "Robot.h" Primary header file for the robotics plug-in.

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

#ifndef HELIOS_ROBOT_H
#define HELIOS_ROBOT_H

#include "PhysicsEngine.h"

//! Result of Robot::solveInverseKinematics()
struct RobotInverseKinematicsResult {
    //! Whether the tool reached the target within tolerance
    bool converged = false;
    //! Joint positions that place the tool closest to the target, in the order of Robot::getJointNames(): degrees for hinges, meters for slides
    std::vector<float> joint_positions;
    //! Remaining distance from the tool to the target position, in meters
    float position_error = 0.f;
    //! Remaining angle between the tool axis and the target axis, in degrees
    float axis_error_degrees = 0.f;
};

//! A robot loaded from an MJCF or URDF file into a PhysicsEngine simulation, with servo-driven joints and a tool
/**
 * Gravity on the robot is compensated, as a robot controller does, so servos hold poses without steady error.
 * Use in this order: construct the robot, call addJointServo() for each driven joint and setTool(), call PhysicsEngine::initialize(), then Robot::initialize().
 * Joint positions are in degrees for hinges and meters for slides. Every method taking an instance acts on that simulation instance, so one robot description drives all parallel instances.
 */
class Robot {
public:
    //! Constructor. Loads the robot model into the simulation.
    /**
     * \param[in] engine Physics simulation to add the robot to. Must not have been initialized.
     * \param[in] model_file Path of the MJCF or URDF file.
     * \param[in] name Name of the robot, unique within the simulation.
     * \param[in] position Position in the world of the robot's origin.
     * \param[in] yaw_degrees Rotation of the robot about the world z-axis in degrees.
     */
    Robot(PhysicsEngine *engine, const std::string &model_file, const std::string &name, const helios::vec3 &position = helios::nullorigin, float yaw_degrees = 0.f);

    //! Destructor
    ~Robot();

    Robot(const Robot &) = delete;
    Robot &operator=(const Robot &) = delete;

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Drive a joint with a position servo. Must be called before PhysicsEngine::initialize().
    /**
     * \param[in] joint_name Name of the joint in the model file.
     * \param[in] stiffness Proportional gain, in N*m/rad for a hinge or N/m for a slide.
     * \param[in] damping Derivative gain, in N*m*s/rad for a hinge or N*s/m for a slide.
     * \param[in] force_limit Largest torque or force the servo can apply. Zero leaves it unlimited.
     * \param[in] armature Inertia of the motor reflected through its gearing, in kg*m^2 for a hinge or kg for a slide. See PhysicsEngine::addPositionServo(), including the limit it sets on damping for stability.
     */
    void addJointServo(const std::string &joint_name, float stiffness, float damping, float force_limit = 0.f, float armature = 0.f);

    //! Define the tool. Must be called before PhysicsEngine::initialize().
    /**
     * \param[in] body_name Name in the model file of the body holding the tool. getToolForce() measures contact with this body and its descendants.
     * \param[in] point Tool point in the body frame, which solveInverseKinematics() places on the target.
     * \param[in] axis Tool axis in the body frame, which solveInverseKinematics() aligns with the target axis.
     */
    void setTool(const std::string &body_name, const helios::vec3 &point, const helios::vec3 &axis);

    //! Connect to the compiled model. Must be called after PhysicsEngine::initialize(), and before any of the methods below.
    void initialize();

    //! Allocate what inverse kinematics needs for every simulation instance that exists now. solveInverseKinematics() does this itself as needed, but that is not safe while other threads are solving.
    void prepareInstances();

    //! Set the pose the robot starts from and returns to with resetInstance(), which is also where the servos hold it. Resets every instance that exists now. An instance created later starts at the zero pose until it is reset.
    /**
     * \param[in] positions Position of each joint in the order of getJointNames(): degrees for hinges, meters for slides.
     */
    void setHomePose(const std::vector<float> &positions);

    //! Return a simulation instance to its initial state, with the robot at its home pose and the servos holding it there
    /**
     * Use this rather than PhysicsEngine::resetInstance(), which also returns the servo targets to zero.
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance = 0);

    //! Get the names of the servo-driven joints, in the order used for joint positions and targets
    [[nodiscard]] std::vector<std::string> getJointNames() const;

    //! Get the range of positions of a servo-driven joint given in the model file
    /**
     * \param[in] joint_name Name of the joint in the model file.
     * \return Lowest (x) and highest (y) position: degrees for a hinge, meters for a slide. Minus and plus infinity if the joint is not limited.
     */
    [[nodiscard]] helios::vec2 getJointRange(const std::string &joint_name) const;

    //! Get the name of the robot given to the constructor. Its bodies and joints are named "<name>/<name in the model file>" in the simulation.
    [[nodiscard]] std::string getName() const;

    //! Get the world position of the robot's origin given to the constructor
    [[nodiscard]] helios::vec3 getBasePosition() const;

    //! Get the rotation of the robot about the world z-axis given to the constructor, in degrees
    [[nodiscard]] float getBaseYaw() const;

    //! Get the physics simulation the robot is in
    [[nodiscard]] PhysicsEngine *getPhysicsEngine() const;

    //! Set the target position of every servo-driven joint
    /**
     * \param[in] targets Target of each joint in the order of getJointNames(): degrees for hinges, meters for slides.
     * \param[in] instance ID of the simulation instance.
     */
    void setJointTargets(const std::vector<float> &targets, uint instance = 0);

    //! Set the target position of one servo-driven joint
    /**
     * \param[in] joint_name Name of the joint in the model file.
     * \param[in] target Target position: degrees for a hinge, meters for a slide.
     * \param[in] instance ID of the simulation instance.
     */
    void setJointTarget(const std::string &joint_name, float target, uint instance = 0);

    //! Place every servo-driven joint directly, with the servos holding it there, without simulating the motion
    /**
     * \param[in] positions Position of each joint in the order of getJointNames(): degrees for hinges, meters for slides.
     * \param[in] instance ID of the simulation instance.
     */
    void setJointPositions(const std::vector<float> &positions, uint instance = 0);

    //! Get the current position of every servo-driven joint: degrees for hinges, meters for slides
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] std::vector<float> getJointPositions(uint instance = 0) const;

    //! Get the current world position of the tool point
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getToolPosition(uint instance = 0) const;

    //! Get the current world direction of the tool axis
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getToolAxis(uint instance = 0) const;

    //! Get the total contact force on the tool and its descendants among the contacts of the last step, in the world frame
    /**
     * This is what an ideal wrist force sensor would read once the tool's weight and inertia are removed, so it is zero when nothing touches the tool.
     * \param[in] instance ID of the simulation instance.
     * \return Force on the tool in N.
     */
    [[nodiscard]] helios::vec3 getToolForce(uint instance = 0) const;

    //! Find the joint positions that place the tool on a target, starting from the current joint positions
    /**
     * Different instances can solve at the same time from different threads, after a call to prepareInstances().
     * Uses damped least squares on MuJoCo's Jacobians, respecting joint limits, and nudges the joints when it stalls at a singular pose such as a fully straight arm. Rotation about the tool axis is left free. This does not move the robot or check for collisions: pass the result to setJointTargets().
     * \param[in] target_position World position for the tool point.
     * \param[in] target_axis World direction for the tool axis.
     * \param[in] locked_joints Names of joints that must stay at their current positions.
     * \param[in] instance ID of the simulation instance whose joint positions are the starting point.
     * \return The solution. Check RobotInverseKinematicsResult::converged, since the target may be out of reach.
     */
    [[nodiscard]] RobotInverseKinematicsResult solveInverseKinematics(const helios::vec3 &target_position, const helios::vec3 &target_axis, const std::vector<std::string> &locked_joints = {}, uint instance = 0) const;

private:
    struct JointRecord {
        std::string name;
        int joint_ID = -1;
        int qpos_address = -1;
        int dof_address = -1;
        int actuator_ID = -1;
        bool is_hinge = true;
    };

    void requireInitialized(const std::string &method) const;
    void requireBeforeSimulationInitialized(const std::string &method) const;
    [[nodiscard]] size_t findJoint(const std::string &joint_name, const std::string &method) const;
    [[nodiscard]] float toUserUnits(const JointRecord &joint, double value) const;
    [[nodiscard]] double toModelUnits(const JointRecord &joint, float value) const;

    PhysicsEngine *engine;
    std::string name;
    std::string prefix;
    helios::vec3 base_position;
    float base_yaw_degrees = 0.f;

    std::vector<JointRecord> joints;
    std::vector<float> home_pose; // empty until setHomePose() is called

    std::string tool_body_name;
    helios::vec3 tool_point;
    helios::vec3 tool_axis = helios::make_vec3(0, 0, 1);
    int tool_body_ID = -1;
    std::vector<char> body_is_on_tool; // indexed by MuJoCo body ID: the tool body and its descendants

    std::vector<struct mjData_ *> kinematics_data; // separate state for inverse kinematics per instance, so that solving never disturbs the simulation and instances can solve at the same time
    bool initialized = false;
};

#endif
