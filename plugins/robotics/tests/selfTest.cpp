/** \file "selfTest.cpp" Self-test routines for the robotics plug-in

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

#include "JointActions.h"
#include "ParallelRobotSimulations.h"
#include "ProtectiveStop.h"
#ifdef HELIOS_ROBOTICS_CAMERA
#include "RobotCamera.h"
#include "Visualizer.h"
#endif
#include "Robot.h"
#include "ToolVelocityAction.h"
#include "VacuumCup.h"
#ifdef HELIOS_ROBOTICS_PLANT_TESTS
#include "PlantPhysics.h"
#endif
#include <mujoco/mujoco.h>
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

int Robot::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

namespace {

    // A planar arm that swings in the x-z plane about y: hinge at the origin, 0.4 m upper link, 0.3 m forearm with a 0.05 m tool
    // body. At zero joint angles it lies along +x. A slide along y moves the whole arm.
    const char *planar_arm_urdf = R"(<?xml version="1.0"?>
<robot name="planar_arm">
  <mujoco><compiler fusestatic="false" discardvisual="true"/></mujoco>
  <link name="base"><inertial><mass value="5"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="carriage"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="upper"><inertial><origin xyz="0.2 0 0"/><mass value="1"/><inertia ixx="0.001" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="forearm"><inertial><origin xyz="0.15 0 0"/><mass value="0.5"/><inertia ixx="0.001" iyy="0.005" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="tool">
    <inertial><origin xyz="0.05 0 0"/><mass value="0.2"/><inertia ixx="0.0001" iyy="0.0005" izz="0.0005" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0.05 0 0"/><geometry><sphere radius="0.04"/></geometry></collision>
  </link>
  <joint name="slide" type="prismatic"><parent link="base"/><child link="carriage"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="500" velocity="1"/></joint>
  <joint name="shoulder" type="revolute"><parent link="carriage"/><child link="upper"/><axis xyz="0 -1 0"/><limit lower="-1.5" upper="1.5" effort="100" velocity="2"/></joint>
  <joint name="elbow" type="revolute"><origin xyz="0.4 0 0"/><parent link="upper"/><child link="forearm"/><axis xyz="0 -1 0"/><limit lower="-2.5" upper="2.5" effort="100" velocity="2"/></joint>
  <joint name="wrist" type="revolute"><origin xyz="0.3 0 0"/><parent link="forearm"/><child link="tool"/><axis xyz="0 -1 0"/><limit lower="-2" upper="2" effort="100" velocity="2"/></joint>
</robot>
)";

    std::string writeArmFile() {
        std::string filename = "robotics_test_planar_arm.urdf";
        std::ofstream file(filename);
        file << planar_arm_urdf;
        return filename;
    }

    void addArmServos(Robot &robot) {
        robot.addJointServo("slide", 5000.f, 500.f, 0.f, 5.f);
        robot.addJointServo("shoulder", 500.f, 50.f, 0.f, 0.5f);
        robot.addJointServo("elbow", 500.f, 50.f, 0.f, 0.5f);
        robot.addJointServo("wrist", 100.f, 10.f, 0.f, 0.1f);
        robot.setTool("tool", make_vec3(0.1f, 0, 0), make_vec3(1, 0, 0));
    }

    // A rod hanging from the world by an elastic joint, like a branch: a tiny stem link fixed to the world and a long rod below it
    PhysicsArticulation makeBranch(const vec3 &base, float length, float joint_stiffness) {
        PhysicsArticulation branch;
        branch.name = "branch";
        PhysicsLink root;
        root.name = "root";
        root.position = base;
        root.mass = 0.01f;
        root.inertia_diagonal = make_vec3(1e-6f, 1e-6f, 1e-6f);
        branch.links.push_back(root);
        PhysicsLink rod;
        rod.name = "rod";
        rod.parent_index = 0;
        rod.position = base + make_vec3(0, 0, 0.5f * length);
        rod.mass = 0.05f;
        rod.inertia_diagonal = make_vec3(1e-4f, 1e-4f, 1e-5f);
        rod.joint_type = PHYSICS_JOINT_BALL;
        rod.anchor_in_parent = make_vec3(0, 0, 0);
        rod.stiffness = joint_stiffness;
        rod.damping = 0.1f * joint_stiffness;
        PhysicsCollisionShape capsule;
        capsule.type = PHYSICS_SHAPE_CAPSULE;
        capsule.size = make_vec3(0.005f, 0.5f * length, 0);
        rod.shapes.push_back(capsule);
        branch.links.push_back(rod);
        return branch;
    }

} // namespace

DOCTEST_TEST_CASE("Robot - forward kinematics matches the arm's geometry") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    const vec3 base = make_vec3(1, 2, 0.5f);
    Robot robot(&engine, arm_file, "arm", base);
    addArmServos(robot);
    engine.initialize();
    robot.initialize();

    DOCTEST_CHECK(robot.getJointNames() == std::vector<std::string>{"slide", "shoulder", "elbow", "wrist"});

    // All joints at zero: the arm lies along +x
    DOCTEST_CHECK((robot.getToolPosition() - (base + make_vec3(0.8f, 0, 0))).magnitude() < 1e-5f);
    DOCTEST_CHECK((robot.getToolAxis() - make_vec3(1, 0, 0)).magnitude() < 1e-5f);

    // Rotated about z, the same arm lies along +y
    {
        Context rotated_context;
        PhysicsEngine rotated_engine(&rotated_context);
        rotated_engine.disableMessages();
        Robot rotated(&rotated_engine, arm_file, "arm", base, 90.f);
        addArmServos(rotated);
        rotated_engine.initialize();
        rotated.initialize();
        DOCTEST_CHECK((rotated.getToolPosition() - (base + make_vec3(0, 0.8f, 0))).magnitude() < 1e-5f);
    }
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("Robot - inverse kinematics reaches a target from the analytic two-link solution") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();

    // With the tool pointing along +x the wrist is 0.1 m behind the tool point, and the two links solve a triangle
    const vec3 target = make_vec3(0.55f, 0.1f, 0.25f);
    RobotInverseKinematicsResult result = robot.solveInverseKinematics(target, make_vec3(1, 0, 0));
    DOCTEST_REQUIRE(result.converged);
    DOCTEST_CHECK(result.position_error < 1e-3f);
    DOCTEST_CHECK(result.axis_error_degrees < 0.1f);

    const float wrist_x = target.x - 0.1f;
    const float wrist_z = target.z;
    const float elbow_cosine = (wrist_x * wrist_x + wrist_z * wrist_z - 0.4f * 0.4f - 0.3f * 0.3f) / (2.f * 0.4f * 0.3f);
    DOCTEST_REQUIRE(std::abs(elbow_cosine) < 1.f);
    const float expected_elbow = rad2deg(std::acos(elbow_cosine));
    DOCTEST_CHECK(result.joint_positions[0] == doctest::Approx(0.1f).epsilon(0.01));
    // Either elbow branch is a valid solution
    DOCTEST_CHECK(std::abs(result.joint_positions[2]) == doctest::Approx(expected_elbow).epsilon(0.01));

    // Solving does not disturb the simulation
    DOCTEST_CHECK(robot.getJointPositions()[1] == doctest::Approx(0.f));

    // A locked joint stays put, here the slide, so a target off the arm's plane cannot be reached
    RobotInverseKinematicsResult locked = robot.solveInverseKinematics(target, make_vec3(1, 0, 0), {"slide"});
    DOCTEST_CHECK(!locked.converged);
    DOCTEST_CHECK(locked.joint_positions[0] == doctest::Approx(0.f));
    DOCTEST_CHECK(locked.position_error > 0.09f);

    // A target beyond the arm's reach is reported, not silently approximated
    RobotInverseKinematicsResult too_far = robot.solveInverseKinematics(make_vec3(3, 0, 0), make_vec3(1, 0, 0));
    DOCTEST_CHECK(!too_far.converged);
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("Robot - servos drive the arm to a solved pose") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();

    const vec3 target = make_vec3(0.5f, -0.2f, 0.3f);
    RobotInverseKinematicsResult result = robot.solveInverseKinematics(target, make_vec3(1, 0, 0));
    DOCTEST_REQUIRE(result.converged);
    robot.setJointTargets(result.joint_positions);
    engine.step(1500);

    INFO("tool " << robot.getToolPosition().x << " " << robot.getToolPosition().y << " " << robot.getToolPosition().z);
    DOCTEST_CHECK((robot.getToolPosition() - target).magnitude() < 5e-3f);

    // Parallel instances are driven independently
    uint other = engine.createInstance();
    robot.setJointTarget("slide", 0.3f, other);
    engine.step(1000, other);
    DOCTEST_CHECK(robot.getJointPositions(other)[0] == doctest::Approx(0.3f).epsilon(0.02));
    DOCTEST_CHECK(robot.getJointPositions(0)[0] == doctest::Approx(-0.2f).epsilon(0.02));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("Robot - tool force is the force that deflects a branch, matching the joint spring") {
    std::string arm_file = writeArmFile();

    // A vertical rod 0.4 m long on a stiff elastic joint, off to the side so that the arm clears it at its straight initial pose
    const float rod_length = 0.4f;
    const float joint_stiffness = 2.f; // N*m/rad
    const float push_height = 0.25f; // above the rod's joint
    const float rod_y = 0.3f;
    const vec3 rod_base = make_vec3(0.695f, rod_y, -push_height);

    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint branchID = engine.addArticulation(makeBranch(rod_base, rod_length, joint_stiffness));
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    PhysicsEngineParameters params;
    params.gravity = false;
    engine.initialize(params);
    robot.initialize();

    // Bring the tool to just short of the rod. Tool point at x = 0.6, so the front of the 0.04 m sphere 0.05 m behind it is at 0.59.
    RobotInverseKinematicsResult start = robot.solveInverseKinematics(make_vec3(0.6f, 0, 0), make_vec3(1, 0, 0), {"slide"});
    DOCTEST_REQUIRE(start.converged);
    robot.setJointTargets(start.joint_positions);
    engine.step(1000);

    // Slide across to the rod's y
    robot.setJointTarget("slide", rod_y);
    engine.step(1500);
    const float tool_y = robot.getToolPosition().y;
    DOCTEST_REQUIRE(std::abs(tool_y - rod_y) < 2e-3f);

    // Free of the rod: no load
    DOCTEST_REQUIRE(engine.getContactCount() == 0);
    DOCTEST_CHECK(robot.getToolForce().magnitude() < 0.01f);

    // Drive the tool slowly along +x and hold. First contact is at about x = 0.7; this goes 6 cm past it.
    const uint ramp_steps = 2000;
    for (uint s = 0; s < ramp_steps; s++) {
        const vec3 tool_target = make_vec3(0.6f + 0.16f * float(s + 1) / float(ramp_steps), tool_y, 0);
        RobotInverseKinematicsResult ik = robot.solveInverseKinematics(tool_target, make_vec3(1, 0, 0), {"slide"});
        DOCTEST_REQUIRE(ik.converged);
        robot.setJointTargets(ik.joint_positions);
        engine.step();
    }
    engine.step(1000);

    const float displacement = engine.getMaxLinkDisplacement();
    DOCTEST_REQUIRE(engine.getContactCount() > 0);
    const vec3 sensor_force = robot.getToolForce();
    const vec3 contact_on_branch = engine.getExternalContactForce(branchID);

    // Newton's third law: the tool and the branch feel equal and opposite forces, with the branch pushed along +x
    INFO("sensor " << sensor_force.x << " " << sensor_force.y << " " << sensor_force.z << " contact " << contact_on_branch.x << " " << contact_on_branch.y << " " << contact_on_branch.z);
    DOCTEST_CHECK(contact_on_branch.x > 0.f);
    DOCTEST_CHECK((sensor_force + contact_on_branch).magnitude() < 1e-3f * contact_on_branch.magnitude());

    // The rod's rest-pose center moves by angle * length / 2. At rest the contact torque balances the joint spring: F * h = K * angle, for contact height h above the joint.
    const float angle = displacement / (0.5f * rod_length);
    const float expected_force = joint_stiffness * angle / push_height;
    INFO("angle " << rad2deg(angle) << " deg, expected force " << expected_force << " N");
    DOCTEST_CHECK(angle > deg2rad(1.f));
    DOCTEST_CHECK(contact_on_branch.x == doctest::Approx(expected_force).epsilon(0.15));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("Robot - home pose holds, and reset returns to it") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();

    const std::vector<float> home = {0.2f, 30.f, -60.f, 30.f};
    robot.setHomePose(home);

    // The arm is really there, not just reporting those angles: upper arm 30 degrees up, forearm 30 degrees down, tool level
    const vec3 expected_tool = make_vec3(0.4f * cosf(deg2rad(30.f)) + 0.3f * cosf(deg2rad(30.f)) + 0.1f, 0.2f, 0.4f * sinf(deg2rad(30.f)) - 0.3f * sinf(deg2rad(30.f)));
    DOCTEST_CHECK((robot.getToolPosition() - expected_tool).magnitude() < 1e-4f);
    DOCTEST_CHECK((robot.getToolAxis() - make_vec3(1, 0, 0)).magnitude() < 1e-4f);

    uint other = engine.createInstance();
    DOCTEST_CHECK(robot.getJointPositions(0)[1] == doctest::Approx(30.f));
    robot.resetInstance(other);
    DOCTEST_CHECK(robot.getJointPositions(other)[1] == doctest::Approx(30.f));

    // The servos hold it there
    engine.stepInstances(500, 2);
    DOCTEST_CHECK(robot.getJointPositions(0)[0] == doctest::Approx(home[0]).epsilon(0.01));
    DOCTEST_CHECK(robot.getJointPositions(0)[2] == doctest::Approx(home[2]).epsilon(0.01));
    DOCTEST_CHECK(robot.getJointPositions(other)[2] == doctest::Approx(home[2]).epsilon(0.01));

    // Move away and come back
    robot.setJointTarget("elbow", 0.f);
    engine.step(500);
    DOCTEST_CHECK(std::abs(robot.getJointPositions()[2] - home[2]) > 10.f);
    robot.resetInstance(0);
    DOCTEST_CHECK(robot.getJointPositions()[2] == doctest::Approx(home[2]));
    engine.step(100);
    DOCTEST_CHECK(robot.getJointPositions()[2] == doctest::Approx(home[2]).epsilon(0.01));
    std::filesystem::remove(arm_file);
}

#ifdef HELIOS_ROBOTICS_PLANT_TESTS

DOCTEST_TEST_CASE("Robot - arm deflects a plant stem, senses the force, and the stem is cut") {
    std::string arm_file = writeArmFile();

    Context context;
    context.seedRandomGenerator(3);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 25);

    // The internode segment nearest two thirds of the way up the main stem, which is where the arm pushes and cuts
    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);
    float top = 0.f;
    for (const PlantArticulationLink &link: articulation.links) {
        if (link.type == ARTICULATION_LINK_INTERNODE) {
            top = std::max(top, link.position.z);
        }
    }
    // The tool approaches along +x from 20 cm in front of the stem, so the stem segment chosen is the one nearest two
    // thirds of the way up whose approach is clear of the rest of the plant. The plant grown from a seed differs between
    // platforms, and on some the nearest segment has another shoot's petioles in front of it.
    std::vector<uint> candidates;
    for (uint i = 0; i < articulation.links.size(); i++) {
        if (articulation.links[i].type == ARTICULATION_LINK_INTERNODE && articulation.links[i].parent_index >= 0) {
            candidates.push_back(i);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&](uint a, uint b) { return std::abs(articulation.links[a].position.z - 0.66f * top) < std::abs(articulation.links[b].position.z - 0.66f * top); });
    auto approachIsClear = [&](uint target) {
        const vec3 &stem = articulation.links[target].position;
        for (uint i = 0; i < articulation.links.size(); i++) {
            const vec3 &p = articulation.links[i].position;
            const bool adjacent = i == target || int(i) == articulation.links[target].parent_index || articulation.links[i].parent_index == int(target);
            if (!adjacent && p.x > stem.x - 0.25f && p.x < stem.x - 0.02f && std::abs(p.y - stem.y) < 0.03f && std::abs(p.z - stem.z) < 0.03f) {
                return false;
            }
        }
        return true;
    };
    const auto clear_candidate = std::find_if(candidates.begin(), candidates.end(), approachIsClear);
    DOCTEST_REQUIRE(clear_candidate != candidates.end());
    const uint cut_link = *clear_candidate;
    const vec3 stem_point = articulation.links[cut_link].position;

    PlantPhysicsParameters plant_parameters;
    plant_parameters.leaf_collisions = false;
    plant_parameters.detachable_links = {cut_link};

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint plantBody = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), plant_parameters);

    // The arm stands to the side of the plant, so that it starts clear of it and slides across
    const float side_offset = 0.35f;
    const vec3 arm_base = make_vec3(stem_point.x - 0.7f, stem_point.y - side_offset, stem_point.z);
    Robot robot(&engine, arm_file, "arm", arm_base);
    addArmServos(robot);
    PhysicsEngineParameters params;
    params.ground_collisions = true;
    engine.initialize(params);
    robot.initialize();

    // Let the plant settle under its own weight with the arm bent back, then slide in front of the stem
    RobotInverseKinematicsResult start = robot.solveInverseKinematics(arm_base + make_vec3(0.5f, 0, 0), make_vec3(1, 0, 0), {"slide"});
    DOCTEST_REQUIRE(start.converged);
    robot.setJointTargets(start.joint_positions);
    engine.step(1000);
    const vec3 settled = engine.getLinkPosition(plantBody, cut_link);
    robot.setJointTarget("slide", settled.y - arm_base.y);
    engine.step(1500);

    // Not touching anything yet
    DOCTEST_CHECK(robot.getToolForce().magnitude() == doctest::Approx(0.f));

    // Push the tool slowly through where the stem stands
    const float tool_y = robot.getToolPosition().y;
    const float tool_z = robot.getToolPosition().z;
    const float start_x = robot.getToolPosition().x;
    const float end_x = settled.x + 0.05f; // the front of the tool sphere is 1 cm behind the tool point
    const uint ramp_steps = 2500;
    float max_force = 0.f;
    float max_contact_force = 0.f;
    float max_deflection = 0.f;
    for (uint s = 0; s < ramp_steps; s++) {
        const vec3 tool_target = make_vec3(start_x + (end_x - start_x) * float(s + 1) / float(ramp_steps), tool_y, tool_z);
        // The slide is left free, since the contact flexes its servo slightly and a locked joint could then not reach the target
        RobotInverseKinematicsResult ik = robot.solveInverseKinematics(tool_target, make_vec3(1, 0, 0));
        DOCTEST_REQUIRE(ik.converged);
        robot.setJointTargets(ik.joint_positions);
        engine.step();
        max_force = std::max(max_force, robot.getToolForce().magnitude());
        max_contact_force = std::max(max_contact_force, engine.getExternalContactForce(plantBody).magnitude());
        max_deflection = std::max(max_deflection, (engine.getLinkPosition(plantBody, cut_link) - settled).magnitude());
    }

    // The arm moved the stem, and felt it
    INFO("deflection " << 1000.f * max_deflection << " mm, sensed force " << max_force << " N, contact force " << max_contact_force << " N");
    DOCTEST_CHECK(max_deflection > 0.005f);
    DOCTEST_CHECK(max_force > 0.01f);
    DOCTEST_CHECK(max_contact_force > 0.01f);

    // Cut: the upper part of the plant falls to the ground and the rest stays standing
    uint lower_link = static_cast<uint>(articulation.links[cut_link].parent_index);
    const vec3 lower_before = engine.getLinkPosition(plantBody, lower_link);
    engine.detachLink(plantBody, cut_link);
    engine.splitTubesAtLink(plantBody, cut_link);
    engine.step(1000);
    engine.updateContextGeometry();

    DOCTEST_CHECK(engine.isLinkDetached(plantBody, cut_link));
    DOCTEST_CHECK(engine.getLinkPosition(plantBody, cut_link).z < 0.5f * settled.z);
    // The stump springs back a little as the load comes off, so it stays near where it was
    DOCTEST_CHECK(std::abs(engine.getLinkPosition(plantBody, lower_link).z - lower_before.z) < 0.25f * settled.z);
    std::filesystem::remove(arm_file);
}

#endif // HELIOS_ROBOTICS_PLANT_TESTS

DOCTEST_TEST_CASE("Robot - instances solve inverse kinematics at the same time, and joints can be placed directly") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();
    const uint instance_count = 4;
    while (engine.getInstanceCount() < instance_count) {
        engine.createInstance();
    }
    robot.prepareInstances();

    // Placed directly: really there, and held there
    robot.setJointPositions({0.1f, 30.f, -60.f, 30.f}, 2);
    const vec3 expected_tool = make_vec3(0.7f * cosf(deg2rad(30.f)) + 0.1f, 0.1f, 0.1f * sinf(deg2rad(30.f)));
    DOCTEST_CHECK((robot.getToolPosition(2) - expected_tool).magnitude() < 1e-4f);
    engine.step(200, 2);
    DOCTEST_CHECK((robot.getToolPosition(2) - expected_tool).magnitude() < 2e-3f);

    // Each instance reaches its own target from its own thread, many times over, as one thread alone does
    const uint iterations = 200;
    std::vector<RobotInverseKinematicsResult> threaded(instance_count);
    std::vector<std::thread> threads;
    for (uint i = 0; i < instance_count; i++) {
        threads.emplace_back([&, i]() {
            for (uint n = 0; n < iterations; n++) {
                threaded[i] = robot.solveInverseKinematics(make_vec3(0.5f + 0.02f * float(i), 0.05f * float(i), 0.2f), make_vec3(1, 0, 0), {}, i);
            }
        });
    }
    for (std::thread &thread: threads) {
        thread.join();
    }
    for (uint i = 0; i < instance_count; i++) {
        const RobotInverseKinematicsResult alone = robot.solveInverseKinematics(make_vec3(0.5f + 0.02f * float(i), 0.05f * float(i), 0.2f), make_vec3(1, 0, 0), {}, i);
        DOCTEST_CHECK(threaded[i].converged);
        for (size_t j = 0; j < alone.joint_positions.size(); j++) {
            DOCTEST_CHECK(threaded[i].joint_positions[j] == alone.joint_positions[j]);
        }
    }
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("Robot - misuse is reported") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();

    DOCTEST_CHECK_THROWS(Robot(nullptr, arm_file, "arm"));
    DOCTEST_CHECK_THROWS(Robot(&engine, arm_file, ""));
    DOCTEST_CHECK_THROWS(Robot(&engine, "does_not_exist.urdf", "missing"));

    Robot robot(&engine, arm_file, "arm");
    robot.addJointServo("shoulder", 100.f, 10.f);
    DOCTEST_CHECK_THROWS(robot.addJointServo("shoulder", 100.f, 10.f));
    DOCTEST_CHECK_THROWS(robot.setTool("tool", nullorigin, make_vec3(0, 0, 0)));

    DOCTEST_CHECK_THROWS(static_cast<void>(robot.getJointPositions()));
    DOCTEST_CHECK_THROWS(static_cast<void>(robot.getToolPosition()));
    DOCTEST_CHECK_THROWS(robot.setJointTargets({0.f}));
    DOCTEST_CHECK_THROWS(static_cast<void>(robot.solveInverseKinematics(nullorigin, make_vec3(1, 0, 0))));

    engine.initialize();
    robot.initialize();

    DOCTEST_CHECK_THROWS(robot.initialize());
    DOCTEST_CHECK_THROWS(robot.addJointServo("elbow", 100.f, 10.f));
    DOCTEST_CHECK_THROWS(robot.setJointTargets({0.f, 0.f}));
    DOCTEST_CHECK_THROWS(robot.setHomePose({0.f, 0.f}));
    DOCTEST_CHECK_THROWS(robot.setJointTarget("elbow", 0.f));
    // No tool was set
    DOCTEST_CHECK_THROWS(static_cast<void>(robot.getToolPosition()));
    DOCTEST_CHECK_THROWS(static_cast<void>(robot.getToolForce()));
    DOCTEST_CHECK_THROWS(static_cast<void>(robot.solveInverseKinematics(nullorigin, make_vec3(1, 0, 0))));

    // A servo on a joint that is not in the model is reported when the model is built
    {
        Context other_context;
        PhysicsEngine other_engine(&other_context);
        other_engine.disableMessages();
        Robot other(&other_engine, arm_file, "arm");
        other.addJointServo("not_a_joint", 100.f, 10.f);
        DOCTEST_CHECK_THROWS(other_engine.initialize());
    }
    std::filesystem::remove(arm_file);
}

namespace {

    // Place a robot arm with its tool point at a position and the tool axis along +x
    void placeTool(Robot &robot, const vec3 &tool_point, uint instance = 0) {
        RobotInverseKinematicsResult pose = robot.solveInverseKinematics(tool_point, make_vec3(1, 0, 0), {"slide"}, instance);
        DOCTEST_REQUIRE(pose.converged);
        robot.setJointPositions(pose.joint_positions, instance);
    }

    // A sphere like a fruit, held to a fixed root above it by a joint that can be released
    PhysicsArticulation makeFruit(const vec3 &center, float radius) {
        PhysicsArticulation fruit;
        fruit.name = "fruit";
        PhysicsLink root;
        root.name = "root";
        root.position = center + make_vec3(0, 0, 0.05f);
        root.mass = 0.01f;
        root.inertia_diagonal = make_vec3(1e-6f, 1e-6f, 1e-6f);
        fruit.links.push_back(root);
        PhysicsLink body;
        body.name = "body";
        body.label = "fruit";
        body.parent_index = 0;
        body.position = center;
        body.mass = 0.15f;
        body.inertia_diagonal = make_vec3(5.4e-5f, 5.4e-5f, 5.4e-5f);
        body.joint_type = PHYSICS_JOINT_DETACHABLE;
        body.anchor_in_parent = make_vec3(0, 0, -0.05f);
        PhysicsCollisionShape sphere;
        sphere.type = PHYSICS_SHAPE_SPHERE;
        sphere.size.x = radius;
        body.shapes.push_back(sphere);
        fruit.links.push_back(body);
        return fruit;
    }

} // namespace

DOCTEST_TEST_CASE("ToolVelocityAction - the tool follows the commanded velocity, with the target held near the tool") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();
    while (engine.getInstanceCount() < 3) {
        engine.createInstance();
    }
    robot.prepareInstances();

    const float max_speed = 0.1f;
    const float dt = 0.04f;
    const uint substeps = 20; // 2 ms physics steps
    const vec3 start = make_vec3(0.55f, 0, 0);

    // By default the tool keeps the axis it had when reset
    {
        ToolVelocityAction action(&robot, max_speed, 0.05f);
        action.setLockedJoints({"slide"});
        placeTool(robot, start);
        action.resetInstance();
        DOCTEST_CHECK((action.getTarget() - start).magnitude() < 1e-4f);
        // A command beyond 1 is clamped, so this moves up at the maximum speed for one second
        for (int step = 0; step < 25; step++) {
            action.apply(make_vec3(0, 0, 5.f), dt);
            engine.step(substeps);
        }
        DOCTEST_CHECK(action.getTarget().z == doctest::Approx(0.1f).epsilon(1e-3));
        engine.step(250);
        DOCTEST_CHECK((robot.getToolPosition() - (start + make_vec3(0, 0, 0.1f))).magnitude() < 0.01f);
        DOCTEST_CHECK(robot.getToolAxis().x > 0.999f);

        // Without the arm moving, the target runs ahead only as far as the lead allows
        // 20 steps would move it 8 cm
        for (int step = 0; step < 20; step++) {
            action.apply(make_vec3(-1, 0, 0), dt);
        }
        DOCTEST_CHECK((action.getTarget() - robot.getToolPosition()).magnitude() == doctest::Approx(0.05f).epsilon(1e-3));
    }

    // Pointing away from a pivot carried by the slide, the arm follows a sideways command by sliding, with the tool still along +x. With the pivot fixed, the
    // tool would have to turn, which this arm cannot do, so it stays where it is.
    for (const bool carried: {true, false}) {
        const uint instance = carried ? 1 : 2;
        ToolVelocityAction action(&robot, max_speed, 0.05f);
        action.setToolAxisAwayFromPivot(make_vec3(0, 0, 0), carried ? "slide" : "");
        placeTool(robot, start, instance);
        action.resetInstance(instance);
        for (int step = 0; step < 25; step++) {
            action.apply(make_vec3(0, 1, 0), dt, instance);
            engine.step(substeps, instance);
        }
        engine.step(250, instance);
        INFO("pivot carried by the slide: " << carried);
        if (carried) {
            DOCTEST_CHECK(robot.getToolPosition(instance).y == doctest::Approx(0.1f).epsilon(0.1));
            DOCTEST_CHECK(robot.getToolAxis(instance).x > 0.999f);
        } else {
            DOCTEST_CHECK(std::abs(robot.getToolPosition(instance).y) < 0.005f);
        }
    }

    // Too near the pivot's line, the direction is undefined and the joint targets are left alone
    {
        ToolVelocityAction action(&robot, max_speed, 0.05f);
        action.setToolAxisAwayFromPivot(start + make_vec3(0.01f, 0, 0), "", 0.05f);
        placeTool(robot, start);
        action.resetInstance();
        const std::vector<float> before = robot.getJointPositions();
        action.apply(make_vec3(0, 0, 1), dt);
        engine.step(substeps);
        const std::vector<float> after = robot.getJointPositions();
        for (size_t j = 0; j < before.size(); j++) {
            DOCTEST_CHECK(after[j] == doctest::Approx(before[j]).epsilon(1e-3));
        }
    }

    // A fixed axis the arm cannot reach leaves the arm where it was
    {
        ToolVelocityAction action(&robot, max_speed, 0.05f);
        action.setToolAxis(make_vec3(0, 1, 0));
        placeTool(robot, start);
        action.resetInstance();
        for (int step = 0; step < 5; step++) {
            action.apply(make_vec3(0, 0, 1), dt);
            engine.step(substeps);
        }
        DOCTEST_CHECK((robot.getToolPosition() - start).magnitude() < 0.005f);
    }

    ToolVelocityAction action(&robot, max_speed, 0.05f);
    DOCTEST_CHECK_THROWS(action.setLockedJoints({"missing"}));
    DOCTEST_CHECK_THROWS(action.setToolAxis(make_vec3(0, 0, 0)));
    DOCTEST_CHECK_THROWS(action.setToolAxisAwayFromPivot(nullorigin, "shoulder"));
    DOCTEST_CHECK_THROWS(action.setToolAxisAwayFromPivot(nullorigin, "", 0.f));
    DOCTEST_CHECK_THROWS(action.apply(make_vec3(0, 0, 1), dt, 7));
    DOCTEST_CHECK_THROWS(ToolVelocityAction(nullptr, max_speed, 0.05f));
    DOCTEST_CHECK_THROWS(ToolVelocityAction(&robot, 0.f, 0.05f));
    DOCTEST_CHECK_THROWS(ToolVelocityAction(&robot, max_speed, -1.f));
    const uint late_instance = engine.createInstance();
    DOCTEST_CHECK_THROWS(action.resetInstance(late_instance));
    robot.prepareInstances();
    action.prepareInstances();
    DOCTEST_CHECK_NOTHROW(action.resetInstance(late_instance));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("VacuumCup - seals only when lined up, holds a freed fruit, and breaks its seal when pulled too hard") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();

    const vec3 face = make_vec3(0.55f, 0, 0);
    const float fruit_radius = 0.03f;
    VacuumCupParameters parameters;
    const float reach = fruit_radius + parameters.lip_thickness;
    const float seat = std::sqrt(reach * reach - parameters.lip_radius * parameters.lip_radius);
    const vec3 seated_center = face + make_vec3(seat, 0, 0);
    uint fruitID = engine.addArticulation(makeFruit(seated_center, fruit_radius));
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();

    // The cup's mouth is at the tool point
    VacuumCup cup(&robot, "tool", make_vec3(0.1f, 0, 0), make_vec3(1, 0, 0), parameters);
    DOCTEST_CHECK_THROWS(static_cast<void>(cup.getSealGap()));
    DOCTEST_CHECK_THROWS(cup.setTarget(fruitID, 0)); // the root has no sphere
    cup.setTarget(fruitID, 1);
    DOCTEST_CHECK(cup.getSeatDistance() == doctest::Approx(seat));

    // Short of the fruit, and out of line with it, the cup does not seal
    placeTool(robot, face - make_vec3(0.03f, 0, 0));
    // From the far side of the lip to the fruit's surface
    const float along = seat + 0.03f;
    DOCTEST_CHECK(cup.getSealGap() == doctest::Approx(std::sqrt(along * along + parameters.lip_radius * parameters.lip_radius) - reach).epsilon(0.01));
    DOCTEST_CHECK_FALSE(cup.grip());
    placeTool(robot, face + make_vec3(0, 0, 0.01f));
    DOCTEST_CHECK(cup.getOffAxis().z == doctest::Approx(-0.01f).epsilon(0.01));
    DOCTEST_CHECK(cup.getSealGap() > parameters.seal_reach);
    DOCTEST_CHECK_FALSE(cup.grip());
    DOCTEST_CHECK_FALSE(cup.isGripped());

    // Lined up with the fruit seated in the mouth, it seals
    placeTool(robot, face);
    DOCTEST_CHECK((cup.getFacePosition() - face).magnitude() < 1e-4f);
    DOCTEST_CHECK(cup.getAxis().x == doctest::Approx(1.f));
    DOCTEST_CHECK(std::abs(cup.getSealGap()) < 1e-3f);
    DOCTEST_CHECK(cup.getOffAxis().magnitude() < 1e-4f);
    DOCTEST_CHECK(cup.grip());
    DOCTEST_CHECK(cup.isGripped());

    // Pulled straight back while the fruit is still attached, the pull passes the seal's limit
    const RobotInverseKinematicsResult back = robot.solveInverseKinematics(face - make_vec3(0.04f, 0, 0), make_vec3(1, 0, 0), {"slide"});
    DOCTEST_REQUIRE(back.converged);
    robot.setJointTargets(back.joint_positions);
    bool overloaded = false;
    float peak_pull = 0.f;
    for (int s = 0; s < 1000 && !overloaded; s++) {
        cup.applyForces(engine.getTimestep());
        engine.step();
        peak_pull = std::max(peak_pull, cup.getPullForce());
        overloaded = cup.isSealOverloaded();
    }
    INFO("peak pull " << peak_pull << " N");
    DOCTEST_CHECK(overloaded);
    DOCTEST_CHECK(cup.getSealLoad() > 1.f);
    DOCTEST_CHECK(peak_pull > 0.5f * parameters.pull_limit);
    DOCTEST_CHECK(cup.isGripped()); // until released
    cup.release();
    DOCTEST_CHECK_FALSE(cup.isGripped());
    DOCTEST_CHECK_FALSE(cup.isSealBroken()); // released by hand, not by the load
    // The force of the last step stays readable until the next one, in which the cup no longer grips
    DOCTEST_CHECK(cup.getForceOnTarget().magnitude() > 0.f);
    cup.applyForces(engine.getTimestep());
    DOCTEST_CHECK(cup.getForceOnTarget().magnitude() == doctest::Approx(0.f));

    // Gripped again and freed from its stem, the fruit is carried with the cup, its weight held across the mouth
    placeTool(robot, face);
    engine.step(50);
    DOCTEST_REQUIRE(cup.grip());
    engine.detachLink(fruitID, 1);
    const RobotInverseKinematicsResult lifted = robot.solveInverseKinematics(face + make_vec3(0, 0, 0.05f), make_vec3(1, 0, 0), {"slide"});
    DOCTEST_REQUIRE(lifted.converged);
    robot.setJointTargets(lifted.joint_positions);
    for (int s = 0; s < 1500; s++) {
        cup.applyForces(engine.getTimestep());
        engine.step();
        DOCTEST_REQUIRE_FALSE(cup.isSealOverloaded());
    }
    const vec3 held = cup.getFacePosition() + seat * cup.getAxis();
    DOCTEST_CHECK((cup.getTargetPosition() - held).magnitude() < 0.002f);
    DOCTEST_CHECK(cup.getTargetPosition().z > seated_center.z + 0.04f);
    const float weight = 0.15f * 9.81f;
    DOCTEST_CHECK(cup.getForceOnTarget().z == doctest::Approx(weight).epsilon(0.05));
    DOCTEST_CHECK(cup.getSealLoad() == doctest::Approx(weight / parameters.shear_limit).epsilon(0.1));

    // Reset releases the grip but keeps the target
    cup.resetInstance();
    DOCTEST_CHECK_FALSE(cup.isGripped());
    DOCTEST_CHECK(cup.getSealLoad() == doctest::Approx(0.f));
    DOCTEST_CHECK_NOTHROW(static_cast<void>(cup.getSealGap()));

    DOCTEST_CHECK_THROWS(VacuumCup(&robot, "missing", make_vec3(0.1f, 0, 0), make_vec3(1, 0, 0)));
    DOCTEST_CHECK_THROWS(VacuumCup(&robot, "tool", make_vec3(0.1f, 0, 0), make_vec3(0, 0, 0)));
    VacuumCupParameters bad;
    bad.pull_limit = 0.f;
    DOCTEST_CHECK_THROWS(VacuumCup(&robot, "tool", make_vec3(0.1f, 0, 0), make_vec3(1, 0, 0), bad));
    DOCTEST_CHECK_THROWS(static_cast<void>(cup.isGripped(3)));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("ProtectiveStop - trips when the arm presses on wood, and not on its target") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();

    // A stiff upright rod, labeled as wood, just ahead of the tool sphere
    const vec3 start = make_vec3(0.55f, 0, 0);
    PhysicsArticulation rod = makeBranch(make_vec3(0.58f, 0, -0.3f), 0.6f, 20.f);
    rod.links[1].label = "wood";
    uint rodID = engine.addArticulation(rod);
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();
    engine.createInstance();
    robot.prepareInstances();

    ContactSensor sensor(&engine, "arm/base");
    const float limit = 5.f;
    ProtectiveStop stop(&sensor, limit);
    DOCTEST_CHECK(stop.getForceLimit() == limit);

    // The same push in both instances, with the rod as the target in instance 1, where the stop ignores it
    sensor.setTargetLink(rodID, 1, 1);
    for (uint instance = 0; instance < 2; instance++) {
        placeTool(robot, start, instance);
        engine.step(10, instance);
        DOCTEST_CHECK(stop.update(instance) == doctest::Approx(0.f));
        DOCTEST_CHECK_FALSE(stop.isStopped(instance));
        float peak = 0.f;
        for (int s = 0; s < 1000; s++) {
            const RobotInverseKinematicsResult pose = robot.solveInverseKinematics(start + make_vec3(0.1f * float(s + 1) / 1000.f, 0, 0), make_vec3(1, 0, 0), {"slide"}, instance);
            DOCTEST_REQUIRE(pose.converged);
            robot.setJointTargets(pose.joint_positions, instance);
            engine.step(1, instance);
            peak = std::max(peak, stop.update(instance));
        }
        INFO("instance " << instance << ", peak force " << peak << " N, force on the arm " << sensor.getTotalForce({}, instance).magnitude() << " N");
        if (instance == 0) {
            DOCTEST_CHECK(peak > limit);
            DOCTEST_CHECK(stop.isStopped(0));
        } else {
            DOCTEST_CHECK(peak == doctest::Approx(0.f));
            DOCTEST_CHECK_FALSE(stop.isStopped(1));
            DOCTEST_CHECK(sensor.getForce("target", 1).magnitude() > limit);
        }
    }

    // The stop stays tripped until reset
    DOCTEST_CHECK(stop.measureForce(0) > 0.f);
    stop.resetInstance(0);
    DOCTEST_CHECK_FALSE(stop.isStopped(0));

    DOCTEST_CHECK_THROWS(ProtectiveStop(nullptr, limit));
    DOCTEST_CHECK_THROWS(ProtectiveStop(&sensor, 0.f));
    DOCTEST_CHECK_THROWS(ProtectiveStop(&sensor, limit, {"leaf"}));
    DOCTEST_CHECK_THROWS(static_cast<void>(stop.isStopped(2)));
    std::filesystem::remove(arm_file);
}

namespace {

    // A component that records the hooks called on it, and reports an event after a chosen number of physics steps since it was reset
    struct RecordingComponent : public RobotComponent {
        std::vector<std::string> calls;
        uint resets = 0;
        uint steps_since_reset = 0;
        uint event_after_steps = 0; // zero for never

        void prepareInstances() override {
            calls.push_back("prepare");
        }
        void resetInstance(uint instance) override {
            resets++;
            steps_since_reset = 0;
        }
        void beginControlStep(uint instance) override {
            calls.push_back("begin");
        }
        void beforePhysicsStep(uint instance, float dt) override {
            calls.push_back("before");
        }
        bool afterPhysicsStep(uint instance) override {
            calls.push_back("after");
            steps_since_reset++;
            return steps_since_reset == event_after_steps;
        }
    };

    // A planar arm in its own simulation, with a velocity action on its tool and a control loop, for testing ParallelRobotSimulations
    struct ArmScene {
        Context context;
        PhysicsEngine engine;
        std::unique_ptr<Robot> robot;
        std::unique_ptr<ToolVelocityAction> action;
        std::unique_ptr<ControlLoop> loop;

        ArmScene(const std::string &arm_file, const vec3 &base, uint instance_count) : engine(&context) {
            engine.disableMessages();
            robot = std::make_unique<Robot>(&engine, arm_file, "arm", base);
            addArmServos(*robot);
            engine.initialize();
            robot->initialize();
            while (engine.getInstanceCount() < instance_count) {
                engine.createInstance();
            }
            robot->prepareInstances();
            action = std::make_unique<ToolVelocityAction>(robot.get(), 0.1f, 0.05f);
            action->setLockedJoints({"slide"});
            loop = std::make_unique<ControlLoop>(&engine, 20);
            loop->addComponent(action.get());
            for (uint instance = 0; instance < instance_count; instance++) {
                placeTool(*robot, base + make_vec3(0.55f, 0, 0), instance);
            }
        }
    };

} // namespace

DOCTEST_TEST_CASE("ControlLoop - runs the physics steps of a control step with every component, and stops at an event") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    PhysicsArticulation branch = makeBranch(make_vec3(0, 0, 0), 0.4f, 1.f);
    branch.links[1].joint_type = PHYSICS_JOINT_DETACHABLE;
    uint branchID = engine.addArticulation(branch);
    engine.initialize();

    ControlLoop loop(&engine, 5);
    DOCTEST_CHECK(loop.getPhysicsStepsPerControl() == 5);
    DOCTEST_CHECK(loop.getControlPeriod() == doctest::Approx(5.f * engine.getTimestep()));
    RecordingComponent first;
    RecordingComponent second;
    loop.addComponent(&first);
    loop.addComponent(&second);
    loop.prepareInstances();
    DOCTEST_CHECK(first.calls == std::vector<std::string>{"prepare"});

    // A whole control step, with the hooks in order
    first.calls.clear();
    ControlStepResult result = loop.step(0);
    DOCTEST_CHECK(result.physics_steps == 5);
    DOCTEST_CHECK_FALSE(result.event);
    std::vector<std::string> expected = {"begin"};
    for (int s = 0; s < 5; s++) {
        expected.push_back("before");
        expected.push_back("after");
    }
    DOCTEST_CHECK(first.calls == expected);
    DOCTEST_CHECK(engine.getSimulationTime() == doctest::Approx(5.f * engine.getTimestep()));

    // An event from the second component ends the next control step after its second physics step, and the first still sees that step
    second.event_after_steps = 7;
    first.calls.clear();
    result = loop.step(0);
    DOCTEST_CHECK(result.physics_steps == 2);
    DOCTEST_CHECK(result.event);
    DOCTEST_CHECK(std::count(first.calls.begin(), first.calls.end(), "after") == 2);

    // Reset returns the simulation and every component to the start
    loop.resetInstance(0);
    DOCTEST_CHECK(engine.getSimulationTime() == doctest::Approx(0.f));
    DOCTEST_CHECK(first.resets == 1);
    DOCTEST_CHECK(second.resets == 1);

    // A link released by load is an event too
    second.event_after_steps = 0;
    engine.setDetachForce(branchID, 1, 1e-4f);
    result = loop.step(0);
    DOCTEST_CHECK(result.event);
    DOCTEST_CHECK(result.physics_steps == 1);
    DOCTEST_CHECK(engine.isLinkDetached(branchID, 1));

    DOCTEST_CHECK_THROWS(loop.addComponent(&first));
    DOCTEST_CHECK_THROWS(loop.addComponent(nullptr));
    DOCTEST_CHECK_THROWS(ControlLoop(&engine, 0));
    DOCTEST_CHECK_THROWS(ControlLoop(nullptr, 5));
}

DOCTEST_TEST_CASE("ControlLoop - a vacuum cup in the loop lets go when its seal breaks, and a grip action commands it") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    const vec3 face = make_vec3(0.55f, 0, 0);
    VacuumCupParameters parameters;
    const float reach = 0.03f + parameters.lip_thickness;
    const float seat = std::sqrt(reach * reach - parameters.lip_radius * parameters.lip_radius);
    uint fruitID = engine.addArticulation(makeFruit(face + make_vec3(seat, 0, 0), 0.03f));
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();

    VacuumCup cup(&robot, "tool", make_vec3(0.1f, 0, 0), make_vec3(1, 0, 0), parameters);
    cup.setTarget(fruitID, 1);
    ControlLoop loop(&engine, 20);
    loop.addComponent(&cup);
    VacuumCupGripAction grip(&cup);
    DOCTEST_CHECK(grip.getActionSize() == 1);

    // A command that is not positive does nothing; a positive one seals the lined-up cup
    placeTool(robot, face);
    const float no[1] = {-1.f};
    const float yes[1] = {1.f};
    grip.applyAction(no, loop.getControlPeriod(), 0);
    DOCTEST_CHECK_FALSE(cup.isGripped());
    grip.applyAction(yes, loop.getControlPeriod(), 0);
    DOCTEST_CHECK(cup.isGripped());

    // Pulled back hard while the fruit is attached, the seal breaks: the control step ends there, and the cup has let go
    const RobotInverseKinematicsResult back = robot.solveInverseKinematics(face - make_vec3(0.04f, 0, 0), make_vec3(1, 0, 0), {"slide"});
    DOCTEST_REQUIRE(back.converged);
    robot.setJointTargets(back.joint_positions);
    ControlStepResult result;
    int control_steps = 0;
    while (!result.event && control_steps < 50) {
        result = loop.step(0);
        control_steps++;
    }
    DOCTEST_CHECK(result.event);
    DOCTEST_CHECK_FALSE(cup.isGripped());
    DOCTEST_CHECK(cup.isSealBroken());
    DOCTEST_CHECK(cup.getPullForce() > parameters.pull_limit * 0.5f);

    // Sealing again clears the break; an action that releases unless positive lets go on a negative command
    loop.resetInstance(0);
    DOCTEST_CHECK_FALSE(cup.isSealBroken());
    placeTool(robot, face);
    VacuumCupGripAction grip_or_release(&cup, true);
    grip_or_release.applyAction(yes, loop.getControlPeriod(), 0);
    DOCTEST_CHECK(cup.isGripped());
    static_cast<void>(loop.step(0));
    grip_or_release.applyAction(no, loop.getControlPeriod(), 0);
    DOCTEST_CHECK_FALSE(cup.isGripped());
    DOCTEST_CHECK_FALSE(cup.isSealBroken());

    DOCTEST_CHECK_THROWS(VacuumCupGripAction(nullptr));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("ProtectiveStop - in a control loop, it ends the control step when it trips and keeps the peak force of the step") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    const vec3 start = make_vec3(0.55f, 0, 0);
    PhysicsArticulation rod = makeBranch(make_vec3(0.58f, 0, -0.3f), 0.6f, 20.f);
    rod.links[1].label = "wood";
    engine.addArticulation(rod);
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();

    ContactSensor sensor(&engine, "arm/base");
    ProtectiveStop stop(&sensor, 5.f);
    ToolVelocityAction move(&robot, 0.1f, 0.05f);
    move.setLockedJoints({"slide"});
    ControlLoop loop(&engine, 20);
    loop.addComponent(&stop);
    loop.addComponent(&move);
    placeTool(robot, start);

    // Pushing into the rod, every control step but the last runs all its physics steps
    const float forward[3] = {1.f, 0.f, 0.f};
    ControlStepResult result;
    float last_peak = 0.f;
    int control_steps = 0;
    while (!result.event && control_steps < 100) {
        move.applyAction(forward, loop.getControlPeriod(), 0);
        result = loop.step(0);
        last_peak = stop.getPeakForce();
        control_steps++;
        if (!result.event) {
            DOCTEST_CHECK(result.physics_steps == 20);
        }
    }
    DOCTEST_CHECK(result.event);
    DOCTEST_CHECK(stop.isStopped());
    DOCTEST_CHECK(last_peak > 5.f);

    // Once tripped it reports no new event, so later control steps run in full
    move.applyAction(forward, loop.getControlPeriod(), 0);
    result = loop.step(0);
    DOCTEST_CHECK_FALSE(result.event);
    DOCTEST_CHECK(result.physics_steps == 20);

    loop.resetInstance(0);
    DOCTEST_CHECK_FALSE(stop.isStopped());
    DOCTEST_CHECK(stop.getPeakForce() == doctest::Approx(0.f));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("ParallelRobotSimulations - steps every copy of every variant in parallel exactly as one at a time") {
    std::string arm_file = writeArmFile();
    const std::vector<vec3> bases = {make_vec3(0, 0, 0), make_vec3(0, 1, 0)};
    const uint instances = 2;

    std::vector<std::unique_ptr<ArmScene>> parallel_scenes;
    std::vector<std::unique_ptr<ArmScene>> serial_scenes;
    ParallelRobotSimulations simulations(3);
    DOCTEST_CHECK(simulations.getThreadCount() == 3);
    for (const vec3 &base: bases) {
        parallel_scenes.push_back(std::make_unique<ArmScene>(arm_file, base, instances));
        serial_scenes.push_back(std::make_unique<ArmScene>(arm_file, base, instances));
        ArmScene &scene = *parallel_scenes.back();
        simulations.addVariant(&scene.engine, scene.loop.get(), {scene.action.get()});
    }
    DOCTEST_REQUIRE(simulations.getCopyCount() == 4);
    DOCTEST_CHECK(simulations.getVariantCount() == 2);
    DOCTEST_CHECK(simulations.getActionSize() == 3);
    DOCTEST_CHECK(simulations.getVariant(3) == 1);
    DOCTEST_CHECK(simulations.getInstance(3) == 1);
    DOCTEST_CHECK(simulations.getInstance(2) == 0);
    DOCTEST_CHECK(simulations.getPhysicsEngine(1) == &parallel_scenes[1]->engine);
    DOCTEST_CHECK(simulations.getControlLoop(0) == parallel_scenes[0]->loop.get());

    // Each copy moves its own way
    const std::vector<float> actions = {0, 0, 1, 0, 0, -1, 1, 0, 1, -1, 0, 0};
    std::vector<ControlStepResult> results(4);
    for (int step = 0; step < 10; step++) {
        simulations.step(actions.data(), results.data());
        for (uint copy = 0; copy < 4; copy++) {
            ArmScene &scene = *serial_scenes[copy / instances];
            scene.action->applyAction(actions.data() + 3 * copy, scene.loop->getControlPeriod(), copy % instances);
            static_cast<void>(scene.loop->step(copy % instances));
        }
    }
    for (uint copy = 0; copy < 4; copy++) {
        const vec3 parallel_tool = parallel_scenes[copy / instances]->robot->getToolPosition(copy % instances);
        const vec3 serial_tool = serial_scenes[copy / instances]->robot->getToolPosition(copy % instances);
        DOCTEST_CHECK(parallel_tool.x == serial_tool.x);
        DOCTEST_CHECK(parallel_tool.z == serial_tool.z);
        DOCTEST_CHECK(results[copy].physics_steps == 20);
    }
    DOCTEST_CHECK(parallel_scenes[0]->robot->getToolPosition(0).z > 0.01f);
    DOCTEST_CHECK(parallel_scenes[0]->robot->getToolPosition(1).z < -0.01f);

    // Rows of numbers, gathered over the copies in order
    std::vector<float> heights(4);
    simulations.gather(heights.data(), 1, [&](uint copy, uint variant, uint instance, float *row) { row[0] = parallel_scenes[variant]->robot->getToolPosition(instance).z; });
    for (uint copy = 0; copy < 4; copy++) {
        DOCTEST_CHECK(heights[copy] == parallel_scenes[copy / instances]->robot->getToolPosition(copy % instances).z);
    }

    // Only the copies asked for are reset
    simulations.reset({1, 2});
    DOCTEST_CHECK(parallel_scenes[0]->engine.getSimulationTime(1) == doctest::Approx(0.f));
    DOCTEST_CHECK(parallel_scenes[1]->engine.getSimulationTime(0) == doctest::Approx(0.f));
    DOCTEST_CHECK(parallel_scenes[0]->engine.getSimulationTime(0) > 0.f);
    DOCTEST_CHECK(parallel_scenes[1]->engine.getSimulationTime(1) > 0.f);

    // An error for one copy is raised once every copy has been handled; nesting a parallel run is an error
    std::atomic<int> handled{0};
    DOCTEST_CHECK_THROWS(simulations.forEachCopy([&](uint copy, uint, uint) {
        handled++;
        if (copy == 2) {
            helios_runtime_error("ERROR (test): copy 2 fails");
        }
    }));
    DOCTEST_CHECK(handled == 4);
    DOCTEST_CHECK_THROWS(simulations.forEachCopy([&](uint, uint, uint) { simulations.forEachIndex(1, [](size_t) {}); }));
    // The threads still work after an error
    handled = 0;
    simulations.forEachIndex(10, [&](size_t) { handled++; });
    DOCTEST_CHECK(handled == 10);

    // Every variant must take the same actions
    ArmScene other(arm_file, make_vec3(0, 2, 0), 1);
    DOCTEST_CHECK_THROWS(simulations.addVariant(&other.engine, other.loop.get(), {}));
    DOCTEST_CHECK_THROWS(simulations.addVariant(nullptr, other.loop.get(), {other.action.get()}));
    DOCTEST_CHECK_THROWS(static_cast<void>(simulations.getVariant(4)));
    DOCTEST_CHECK_THROWS(simulations.resetCopy(4));
    DOCTEST_CHECK_THROWS(simulations.step(nullptr));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("JointPositionAction and JointVelocityAction - commands in [-1, 1] set and move joint targets") {
    std::string arm_file = writeArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    Robot robot(&engine, arm_file, "arm");
    addArmServos(robot);
    engine.initialize();
    robot.initialize();
    engine.createInstance();
    robot.prepareInstances();
    const mjModel *model = engine.getMuJoCoModel();
    const int shoulder_servo = mj_name2id(model, mjOBJ_ACTUATOR, "arm/shoulder_servo");
    const int slide_servo = mj_name2id(model, mjOBJ_ACTUATOR, "arm/slide_servo");
    DOCTEST_REQUIRE(shoulder_servo >= 0);
    DOCTEST_REQUIRE(slide_servo >= 0);

    // The ranges in the model file: the shoulder turns 1.5 rad either way, the slide moves 1 m either way
    const vec2 shoulder_range = robot.getJointRange("shoulder");
    DOCTEST_CHECK(shoulder_range.x == doctest::Approx(-rad2deg(1.5f)));
    DOCTEST_CHECK(shoulder_range.y == doctest::Approx(rad2deg(1.5f)));
    DOCTEST_CHECK(robot.getJointRange("slide").y == doctest::Approx(1.f));
    DOCTEST_CHECK_THROWS(static_cast<void>(robot.getJointRange("missing")));

    // Positions: -1 is the bottom of the range, 1 the top, and commands beyond are clamped
    JointPositionAction position(&robot, {"shoulder", "slide"});
    DOCTEST_CHECK(position.getActionSize() == 2);
    const float middle[2] = {0.f, 0.5f};
    position.applyAction(middle, 0.04f, 1);
    DOCTEST_CHECK(engine.getMuJoCoData(1)->ctrl[shoulder_servo] == doctest::Approx(0.0));
    DOCTEST_CHECK(engine.getMuJoCoData(1)->ctrl[slide_servo] == doctest::Approx(0.5));
    const float beyond[2] = {3.f, -3.f};
    position.applyAction(beyond, 0.04f, 1);
    DOCTEST_CHECK(engine.getMuJoCoData(1)->ctrl[shoulder_servo] == doctest::Approx(1.5).epsilon(1e-4));
    DOCTEST_CHECK(engine.getMuJoCoData(1)->ctrl[slide_servo] == doctest::Approx(-1.0));
    DOCTEST_CHECK(engine.getMuJoCoData(0)->ctrl[slide_servo] == doctest::Approx(0.0)); // only its own instance
    JointPositionAction narrow(&robot, {"slide"}, {make_vec2(0.f, 0.2f)});
    const float low[1] = {-1.f};
    narrow.applyAction(low, 0.04f, 0);
    DOCTEST_CHECK(engine.getMuJoCoData(0)->ctrl[slide_servo] == doctest::Approx(0.0));

    // Velocities: the slide's target moves at its maximum speed and the slide follows
    JointVelocityAction velocity(&robot, {"slide"}, {0.1f}, {0.05f});
    DOCTEST_CHECK(velocity.getActionSize() == 1);
    robot.setJointPositions({0.f, 0.f, 0.f, 0.f}, 0);
    velocity.resetInstance(0);
    DOCTEST_CHECK(velocity.getTargets(0)[0] == doctest::Approx(0.f));
    const float forward[1] = {1.f};
    for (int step = 0; step < 10; step++) {
        velocity.applyAction(forward, 0.04f, 0);
        engine.step(20, 0);
    }
    DOCTEST_CHECK(velocity.getTargets(0)[0] == doctest::Approx(0.04f).epsilon(0.01));
    engine.step(200, 0);
    DOCTEST_CHECK(robot.getJointPositions(0)[0] == doctest::Approx(0.04f).epsilon(0.05));

    // Without the slide moving, the target leads it by at most the lead, and never passes the end of the range
    for (int step = 0; step < 50; step++) {
        velocity.applyAction(forward, 0.04f, 0);
    }
    DOCTEST_CHECK(velocity.getTargets(0)[0] - robot.getJointPositions(0)[0] == doctest::Approx(0.05f).epsilon(1e-3));
    robot.setJointPositions({0.99f, 0.f, 0.f, 0.f}, 0);
    velocity.resetInstance(0);
    for (int step = 0; step < 10; step++) {
        velocity.applyAction(forward, 0.04f, 0);
    }
    DOCTEST_CHECK(velocity.getTargets(0)[0] == doctest::Approx(1.f));

    DOCTEST_CHECK_THROWS(JointPositionAction(&robot, {}));
    DOCTEST_CHECK_THROWS(JointPositionAction(&robot, {"missing"}));
    DOCTEST_CHECK_THROWS(JointPositionAction(&robot, {"slide", "slide"}));
    DOCTEST_CHECK_THROWS(JointPositionAction(&robot, {"slide"}, {make_vec2(0.f, 0.2f), make_vec2(0.f, 0.2f)}));
    DOCTEST_CHECK_THROWS(JointPositionAction(&robot, {"slide"}, {make_vec2(0.2f, 0.f)}));
    DOCTEST_CHECK_THROWS(JointPositionAction(nullptr, {"slide"}));
    DOCTEST_CHECK_THROWS(JointVelocityAction(&robot, {"slide"}, {0.f}, {0.05f}));
    DOCTEST_CHECK_THROWS(JointVelocityAction(&robot, {"slide"}, {0.1f}, {}));
    DOCTEST_CHECK_THROWS(static_cast<void>(velocity.getTargets(5)));
    std::filesystem::remove(arm_file);
}

#ifdef HELIOS_ROBOTICS_CAMERA

namespace {

    // Whether an offscreen OpenGL context can be created here, which drawing needs; machines without a GL driver cannot
    bool headlessContextAvailable() {
        static const bool available = []() {
            try {
                Visualizer probe(16, 16, 0, true, true);
                probe.disableMessages();
            } catch (...) {
                return false;
            }
            return true;
        }();
        return available;
    }

} // namespace

DOCTEST_TEST_CASE("RobotCamera - draws each instance's scene in color, depth, and the primitive and object seen in each pixel") {
    if (!headlessContextAvailable()) {
        DOCTEST_MESSAGE("Skipped: no OpenGL context can be created on this machine");
        return;
    }
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    const vec3 box_center = make_vec3(1, 0, 0.5f);
    const RGBcolor box_color = make_RGBcolor(0.9f, 0.1f, 0.1f);
    const uint boxID = engine.addKinematicBox(box_center, make_vec3(0.4f, 0.4f, 0.4f), box_color);
    PhysicsArticulation pendulum;
    pendulum.name = "pendulum";
    PhysicsLink pivot;
    pivot.name = "pivot";
    pivot.position = make_vec3(-1, 0, 1);
    pivot.mass = 1.f;
    pivot.inertia_diagonal = make_vec3(1e-3f, 1e-3f, 1e-3f);
    pendulum.links.push_back(pivot);
    PhysicsLink bob = pivot;
    bob.name = "bob";
    bob.parent_index = 0;
    bob.position = make_vec3(-0.8f, 0, 0.6f);
    bob.joint_type = PHYSICS_JOINT_BALL;
    pendulum.links.push_back(bob);
    engine.addArticulation(pendulum);
    engine.initialize();
    const uint other = engine.createInstance();
    const std::vector<uint> objects = context.getAllObjectIDs();
    DOCTEST_REQUIRE(objects.size() == 1);
    const uint box_object = objects.front();

    // In the other instance the box is moved out of view
    engine.setKinematicBoxPosition(boxID, box_center + make_vec3(0, 5, 0), make_vec3(0, 0, 0), other);
    engine.step(1, other);

    const uint width = 32;
    const uint height = 24;
    RobotCamera camera(&engine, &context, width, height);
    DOCTEST_CHECK(camera.getWidth() == width);
    DOCTEST_CHECK(camera.getHeight() == height);
    const vec3 eye = make_vec3(0, 0, 0.5f);
    camera.setPose(eye, box_center);
    camera.setFieldOfView(30.f);

    const size_t pixels = size_t(width) * height;
    const size_t center = size_t(height / 2) * width + width / 2;
    std::vector<unsigned char> rgb(3 * pixels);
    std::vector<float> depth(pixels);
    std::vector<uint> primitives(pixels);
    std::vector<uint> object_image(pixels);

    // The box fills the middle of the view, 0.8 m away at its near face
    camera.render(0);
    camera.getColorImage(rgb.data());
    camera.getDepthImage(depth.data());
    camera.getPrimitiveImage(primitives.data());
    camera.getObjectImage(object_image.data());
    DOCTEST_CHECK(rgb[3 * center] > 2 * rgb[3 * center + 1]); // red
    DOCTEST_CHECK(depth[center] == doctest::Approx(0.8f).epsilon(0.02));
    DOCTEST_REQUIRE(primitives[center] > 0);
    DOCTEST_CHECK(context.getPrimitiveParentObjectID(primitives[center] - 1) == box_object);
    DOCTEST_CHECK(object_image[center] == box_object);
    // A corner shows only the sky
    DOCTEST_CHECK(depth[0] == doctest::Approx(-1.f));
    DOCTEST_CHECK(object_image[0] == 0);

    // The other instance has no box in view
    camera.render(other);
    camera.getObjectImage(object_image.data());
    DOCTEST_CHECK(object_image[center] == 0);

    // Mounted on the bob, the camera moves with it in each instance
    camera.mountOnBody("pendulum_bob", make_vec3(0, 0, 0.1f), make_vec3(1, 0, 0.1f));
    engine.step(200, other);
    vec3 eye_0, look_0, eye_1, look_1;
    camera.getPose(0, eye_0, look_0);
    camera.getPose(other, eye_1, look_1);
    DOCTEST_CHECK((eye_0 - (bob.position + make_vec3(0, 0, 0.1f))).magnitude() < 1e-4f);
    DOCTEST_CHECK((eye_1 - eye_0).magnitude() > 0.01f);
    DOCTEST_CHECK_NOTHROW(camera.render(other));

    const std::string image_file = "robotics_test_camera.jpeg";
    camera.saveImage(image_file);
    DOCTEST_CHECK(std::filesystem::exists(image_file));
    std::filesystem::remove(image_file);

    DOCTEST_CHECK_THROWS(camera.mountOnBody("missing", nullorigin, make_vec3(1, 0, 0)));
    DOCTEST_CHECK_THROWS(camera.setFieldOfView(0.f));
    DOCTEST_CHECK_THROWS(RobotCamera(&engine, &context, 0, 10));
    DOCTEST_CHECK_THROWS(RobotCamera(nullptr, &context, 10, 10));
}

#endif // HELIOS_ROBOTICS_CAMERA
