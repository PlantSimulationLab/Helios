/** \file "VacuumCup.h" A vacuum cup end effector that seals on round objects such as fruit.

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

#ifndef HELIOS_VACUUM_CUP_H
#define HELIOS_VACUUM_CUP_H

#include "Robot.h"
#include "RobotComponent.h"

//! Properties of a vacuum cup
struct VacuumCupParameters {
    //! Distance in meters from the cup's axis to the middle of its lip
    float lip_radius = 0.025f;
    //! Radius in meters of the lip's cross section
    float lip_thickness = 0.004f;
    //! The cup seals when every point of its lip is within this distance in meters of the object's surface, about how far a compliant lip reaches to seal
    float seal_reach = 0.005f;
    //! Stiffness in N/m of the springs that hold a sealed object seated in the cup
    float stiffness = 2000.f;
    //! Damping in N*s/m of the springs that hold a sealed object seated in the cup
    float damping = 20.f;
    //! Time in seconds over which the grip builds up after the cup seals, as a vacuum does
    float ramp_seconds = 0.1f;
    //! Pull in N straight out of the cup's mouth above which the seal breaks
    float pull_limit = 40.f;
    //! Load in N along the cup's mouth above which the seal breaks
    float shear_limit = 20.f;
    //! Moment in N*m about the center of the cup's mouth above which the seal breaks, from twisting and from loads that peel the lip
    float moment_limit = 1.f;
};

//! An ideal vacuum cup on a robot body, which seals on a spherical target such as a fruit and holds it seated in its mouth
/**
 * The cup is a bowl whose mouth is a round lip. It seals only when the whole lip is on the target, which means the target sits in the mouth with its center near the cup's axis.
 * Springs then hold the target's center on the axis, where its surface touches the whole lip, and its orientation as it was when the seal formed, both relative to the cup, reacting on the robot.
 * The seal breaks when the target pulls out of the cup, slides across the mouth, or twists or peels the lip, too hard (VacuumCupParameters).
 *
 * The target of each instance is an articulation link whose first collision shape is a sphere, such as a fruit added by addPlantToPhysicsEngine().
 * Add it to a ControlLoop, which applies the grip before every physics step and releases the target when the seal breaks; or call applyForces() before every physics step yourself.
 * Construct it after Robot::initialize(). Different instances can be used at the same time from different threads.
 */
class VacuumCup : public RobotComponent {
public:
    //! Constructor
    /**
     * \param[in] robot Initialized robot carrying the cup.
     * \param[in] body_name Name in the robot's model file of the body carrying the cup.
     * \param[in] face_center Center of the cup's mouth in the body's frame.
     * \param[in] face_normal Direction the cup's mouth faces in the body's frame, which is also the cup's axis.
     * \param[in] parameters Properties of the cup.
     */
    VacuumCup(Robot *robot, const std::string &body_name, const helios::vec3 &face_center, const helios::vec3 &face_normal, const VacuumCupParameters &parameters = VacuumCupParameters());

    //! Allocate per-instance state for every simulation instance that exists now. Call it after creating instances, before using them from several threads.
    void prepareInstances() override;

    //! Release any grip and clear the loads on the seal, as when an instance is reset. The target is kept.
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance = 0) override;

    //! Apply the grip for a physics step (applyForces()). Called by ControlLoop.
    /**
     * \param[in] instance ID of the simulation instance.
     * \param[in] dt Physics time step in seconds.
     */
    void beforePhysicsStep(uint instance, float dt) override;

    //! Release the target if the load on the seal in the step was more than it holds. Called by ControlLoop.
    /**
     * \param[in] instance ID of the simulation instance.
     * \return Whether the seal broke in this step.
     */
    bool afterPhysicsStep(uint instance) override;

    //! Check whether the seal broke under load, and the target was released, since the instance was reset or the cup last sealed
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] bool isSealBroken(uint instance = 0) const;

    //! Set the object the cup can seal on, releasing any grip
    /**
     * \param[in] articulationID ID of the articulation returned by PhysicsEngine::addArticulation().
     * \param[in] link_index Index of a link whose first collision shape is a sphere.
     * \param[in] instance ID of the simulation instance.
     */
    void setTarget(uint articulationID, uint link_index, uint instance = 0);

    //! Seal on the target if the whole lip is within reach of it (VacuumCupParameters::seal_reach). Does nothing if the cup is already gripping.
    /**
     * \param[in] instance ID of the simulation instance.
     * \return Whether the cup is gripping after the call.
     */
    bool grip(uint instance = 0);

    //! Release the target, removing the grip's forces
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void release(uint instance = 0);

    //! Apply the grip's force and torque to the target and, in reaction, to the robot. Call it before every physics step, gripping or not, since it also clears the forces of the last step.
    /**
     * \param[in] dt Physics time step in seconds, over which the grip builds up after sealing.
     * \param[in] instance ID of the simulation instance.
     */
    void applyForces(float dt, uint instance = 0);

    //! Check whether the cup is gripping its target
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] bool isGripped(uint instance = 0) const;

    //! Get the load on the seal in the last physics step as a fraction of what breaks it: the largest of the pull, the load along the mouth, and the moment, each over its limit
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getSealLoad(uint instance = 0) const;

    //! Check whether the load on the seal in the last physics step was more than it can hold. The cup keeps gripping until release() is called.
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] bool isSealOverloaded(uint instance = 0) const;

    //! Get the pull in N on the seal straight out of the cup's mouth in the last physics step
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getPullForce(uint instance = 0) const;

    //! Get the force in N of the grip on the target in the last physics step, in the world frame. Zero if the cup was not gripping in that step.
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getForceOnTarget(uint instance = 0) const;

    //! Get the world position of the center of the cup's mouth
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getFacePosition(uint instance = 0) const;

    //! Get the world direction of the cup's axis, the way its mouth faces
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getAxis(uint instance = 0) const;

    //! Get the world position of the target's center of mass
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getTargetPosition(uint instance = 0) const;

    //! Get the offset from the cup's axis to the target's center, at right angles to the axis: how far, and which way, the cup is out of line with the target
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] helios::vec3 getOffAxis(uint instance = 0) const;

    //! Get how far the point of the lip farthest from the target is from the target's surface, in meters
    /**
     * The cup can seal when this is within VacuumCupParameters::seal_reach. It is small only when the cup is in line with the target; far from it, it is about the distance to its surface.
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getSealGap(uint instance = 0) const;

    //! Get how far the center of the target, seated in the cup, is from the center of the mouth: its surface then touches the whole lip
    /**
     * \param[in] instance ID of the simulation instance.
     */
    [[nodiscard]] float getSeatDistance(uint instance = 0) const;

private:
    struct InstanceState {
        int target_body_ID = -1;
        float target_radius = 0.f;
        bool gripped = false;
        bool seal_broken = false;
        // Where the cup holds the target, in the cup body's frame: the target's center, the cup's axis, and the target's orientation; and how far the target's center is from the center of the mouth
        double grip_center[3] = {0, 0, 0};
        double grip_normal[3] = {0, 0, 0};
        double grip_orientation[4] = {1, 0, 0, 0};
        float grip_seat_distance = 0.f;
        float grip_seconds = 0.f;
        float pull = 0.f; // N straight out of the mouth in the last physics step
        float shear = 0.f; // N along the mouth in the last physics step
        float moment = 0.f; // N m about the mouth's center in the last physics step
        helios::vec3 force_on_target;
    };

    [[nodiscard]] const InstanceState &state(uint instance, const std::string &method) const;
    [[nodiscard]] InstanceState &state(uint instance, const std::string &method);
    [[nodiscard]] const InstanceState &targetState(uint instance, const std::string &method) const;
    void clearAppliedForces(const InstanceState &cup, uint instance);

    Robot *robot;
    PhysicsEngine *engine;
    int body_ID = -1;
    helios::vec3 face_center;
    helios::vec3 face_normal;
    VacuumCupParameters parameters;
    std::vector<InstanceState> instances;
};

//! A grip command for a VacuumCup, taking one number each control step
/**
 * A positive number makes the cup seal on its target if it can (VacuumCup::grip()). Otherwise nothing happens, unless the action was made to release on a command that is not positive.
 * It keeps no state of its own, so it need not be added to the ControlLoop that runs the cup.
 */
class VacuumCupGripAction : public RobotAction {
public:
    //! Constructor
    /**
     * \param[in] cup Cup to command.
     * \param[in] release_unless_positive Release the target on a command that is not positive. Otherwise the cup keeps its grip until its seal breaks or it is reset.
     */
    explicit VacuumCupGripAction(VacuumCup *cup, bool release_unless_positive = false);

    //! Does nothing: the action keeps no state
    void prepareInstances() override;

    //! Does nothing: the action keeps no state
    /**
     * \param[in] instance ID of the simulation instance.
     */
    void resetInstance(uint instance) override;

    //! Get the number of numbers in the action: one, the grip command
    [[nodiscard]] uint getActionSize() const override;

    //! Grip if the command is positive, and otherwise release if the action was made to
    /**
     * \param[in] action The grip command.
     * \param[in] dt Duration of the control step in seconds, which is not used.
     * \param[in] instance ID of the simulation instance.
     */
    void applyAction(const float *action, float dt, uint instance) override;

private:
    VacuumCup *cup;
    bool release_unless_positive;
};

#endif
