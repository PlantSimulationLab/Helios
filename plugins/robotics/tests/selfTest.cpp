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

#include "Robot.h"
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
    uint cut_link = 0;
    float nearest = std::numeric_limits<float>::max();
    for (uint i = 0; i < articulation.links.size(); i++) {
        const PlantArticulationLink &link = articulation.links[i];
        const float distance = std::abs(link.position.z - 0.66f * top);
        if (link.type == ARTICULATION_LINK_INTERNODE && link.parent_index >= 0 && distance < nearest) {
            nearest = distance;
            cut_link = i;
        }
    }
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
