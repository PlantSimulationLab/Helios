/** \file "selfTest.cpp" Self-test routines for the physics engine plug-in

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
#include "PhysicsEngine.h"
#ifdef HELIOS_PHYSICS_ENGINE_PLANTS
#include "PlantPhysics.h"
#endif
#include <mujoco/mujoco.h>
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

int PhysicsEngine::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

namespace {

    // A mass hanging from a frictionless ball joint at the given height, displaced from vertical along x
    PhysicsArticulation makePendulum(const std::string &name, float pivot_height, float length, float start_angle_radians) {
        PhysicsArticulation pendulum;
        pendulum.name = name;

        PhysicsLink pivot;
        pivot.name = "pivot";
        pivot.position = make_vec3(0, 0, pivot_height);
        pivot.mass = 1.f;
        pivot.inertia_diagonal = make_vec3(1e-3f, 1e-3f, 1e-3f);
        pendulum.links.push_back(pivot);

        PhysicsLink bob;
        bob.name = "bob";
        bob.parent_index = 0;
        bob.position = make_vec3(length * std::sin(start_angle_radians), 0, pivot_height - length * std::cos(start_angle_radians));
        bob.mass = 1.f;
        bob.inertia_diagonal = make_vec3(1e-6f, 1e-6f, 1e-6f);
        bob.joint_type = PHYSICS_JOINT_BALL;
        PhysicsCollisionShape sphere;
        sphere.type = PHYSICS_SHAPE_SPHERE;
        sphere.size.x = 0.02f;
        bob.shapes.push_back(sphere);
        pendulum.links.push_back(bob);

        return pendulum;
    }

    std::vector<vec3> getAllVertices(const Context &context, const std::vector<uint> &objIDs) {
        std::vector<vec3> vertices;
        for (uint UUID: context.getObjectPrimitiveUUIDs(objIDs)) {
            for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                vertices.push_back(vertex);
            }
        }
        return vertices;
    }

#ifdef HELIOS_PHYSICS_ENGINE_PLANTS
    // A young bean plant: a few hundred links with internodes, petioles, and leaves
    uint buildTestPlant(Context &context, PlantArchitecture &plantarchitecture) {
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary("bean");
        return plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 25);
    }
#endif

} // namespace

DOCTEST_TEST_CASE("PhysicsEngine - pendulum swings with the period from theory") {
    Context context;
    const float length = 0.5f;
    const float start_angle = 10.f * float(M_PI) / 180.f;

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint pendulumID = engine.addArticulation(makePendulum("pendulum", 1.f, length, start_angle));
    PhysicsEngineParameters params;
    params.timestep = 0.0005f;
    params.contact_time_constant = 0.002f;
    engine.initialize(params);

    // The bob starts at one extreme of the swing along x, and has reached the other half a period later
    float previous_x = engine.getLinkPosition(pendulumID, 1).x;
    float half_period = 0.f;
    float min_x = previous_x;
    for (uint s = 0; s < 10000 && half_period == 0.f; s++) {
        engine.step();
        float x = engine.getLinkPosition(pendulumID, 1).x;
        if (x > previous_x && previous_x < 0.f) {
            half_period = engine.getSimulationTime() - params.timestep;
            min_x = previous_x;
        }
        previous_x = x;
    }

    // Large-amplitude correction to first order: T = 2*pi*sqrt(L/g)*(1 + theta^2/16)
    float expected_half_period = float(M_PI) * std::sqrt(length / 9.81f) * (1.f + start_angle * start_angle / 16.f);
    INFO("half period " << half_period << " s, expected " << expected_half_period << " s");
    DOCTEST_CHECK(std::abs(half_period - expected_half_period) < 0.01f * expected_half_period);
    // No joint damping, so the swing reaches the same amplitude on the other side
    DOCTEST_CHECK(std::abs(min_x + length * std::sin(start_angle)) < 0.02f * length * std::sin(start_angle));
}

DOCTEST_TEST_CASE("PhysicsEngine - kinematic box pushes a link it hits, and misses one out of reach") {
    auto maxDisplacement = [](float box_y) {
        Context context;
        PhysicsEngine engine(&context);
        engine.disableMessages();
        // The pendulum hangs straight down at x = 0, so only a box passing through it can move it
        engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.f));
        vec3 box_start = make_vec3(-0.3f, box_y, 0.5f);
        uint boxID = engine.addKinematicBox(box_start, make_vec3(0.1f, 0.1f, 0.1f));
        engine.initialize();

        const vec3 velocity = make_vec3(0.5f, 0, 0);
        float max_displacement = 0.f;
        const uint steps = 500;
        for (uint s = 0; s < steps; s++) {
            engine.setKinematicBoxPosition(boxID, box_start + float(s) * engine.getTimestep() * velocity, velocity);
            engine.step();
            max_displacement = std::max(max_displacement, engine.getMaxLinkDisplacement());
        }
        return max_displacement;
    };

    DOCTEST_CHECK(maxDisplacement(0.f) > 0.05f);
    DOCTEST_CHECK(maxDisplacement(1.f) < 1e-4f);
}

DOCTEST_TEST_CASE("PhysicsEngine - instances are independent and step in parallel as they do serially") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint pendulumID = engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.3f));
    engine.initialize();
    DOCTEST_CHECK(engine.getInstanceCount() == 1);

    const uint instance_count = 4;
    while (engine.getInstanceCount() < instance_count) {
        engine.createInstance();
    }
    DOCTEST_CHECK(engine.getInstanceCount() == instance_count);

    const uint steps = 300;

    // Only instance 1 moves
    engine.step(steps, 1);
    DOCTEST_CHECK(engine.getSimulationTime(0) == doctest::Approx(0.f));
    DOCTEST_CHECK(engine.getSimulationTime(1) == doctest::Approx(float(steps) * engine.getTimestep()));
    vec3 serial_position = engine.getLinkPosition(pendulumID, 1, 1);
    DOCTEST_CHECK((serial_position - engine.getLinkPosition(pendulumID, 1, 0)).magnitude() > 0.01f);

    // Back at the initial state, every instance reproduces exactly what the serial run gave
    engine.resetInstance(1);
    DOCTEST_CHECK(engine.getSimulationTime(1) == doctest::Approx(0.f));
    DOCTEST_CHECK((engine.getLinkPosition(pendulumID, 1, 1) - engine.getLinkPosition(pendulumID, 1, 0)).magnitude() == doctest::Approx(0.f));

    engine.stepInstances(steps, 2);
    for (uint i = 0; i < instance_count; i++) {
        vec3 position = engine.getLinkPosition(pendulumID, 1, i);
        DOCTEST_CHECK(position.x == serial_position.x);
        DOCTEST_CHECK(position.y == serial_position.y);
        DOCTEST_CHECK(position.z == serial_position.z);
    }
}

DOCTEST_TEST_CASE("PhysicsEngine - kinematic box state is per instance, and Context follows the requested instance") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.f));
    vec3 box_center = make_vec3(2.f, 0, 1.f);
    uint boxID = engine.addKinematicBox(box_center, make_vec3(0.1f, 0.1f, 0.1f));
    engine.initialize();
    uint moved_instance = engine.createInstance();

    const vec3 velocity = make_vec3(0, 0, 0.1f);
    engine.setKinematicBoxPosition(boxID, box_center, velocity, moved_instance);
    engine.stepInstances(100, 2);

    float box_top_rest = 0.f;
    {
        std::vector<uint> box_objIDs = context.getAllObjectIDs();
        DOCTEST_REQUIRE(box_objIDs.size() == 1);
        engine.updateContextGeometry(0);
        for (const vec3 &vertex: getAllVertices(context, box_objIDs)) {
            box_top_rest = std::max(box_top_rest, vertex.z);
        }
        engine.updateContextGeometry(moved_instance);
        float box_top_moved = 0.f;
        for (const vec3 &vertex: getAllVertices(context, box_objIDs)) {
            box_top_moved = std::max(box_top_moved, vertex.z);
        }
        DOCTEST_CHECK(box_top_moved - box_top_rest == doctest::Approx(100.f * engine.getTimestep() * velocity.z).epsilon(0.01));
    }
}

DOCTEST_TEST_CASE("PhysicsEngine - copying to the Context only touches geometry that moved") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();

    // A link fixed to the world, whose sphere never moves, and a swinging pendulum, whose sphere does
    PhysicsArticulation pendulum = makePendulum("pendulum", 1.f, 0.5f, 0.3f);
    uint pendulumID = engine.addArticulation(pendulum);
    uint fixed_sphere = context.addSphereObject(6, pendulum.links[0].position, 0.05f);
    uint moving_sphere = context.addSphereObject(6, pendulum.links[1].position, 0.05f);
    uint tube = context.addTubeObject(6, {pendulum.links[0].position, pendulum.links[0].position + make_vec3(0, 0, 0.2f)}, {0.01f, 0.01f});
    engine.bindObjectToLink(fixed_sphere, pendulumID, 0);
    engine.bindObjectToLink(moving_sphere, pendulumID, 1);
    engine.bindTubeNodesToLinks(tube, pendulumID, {0, 0});
    engine.initialize();

    const std::vector<uint> fixed_UUIDs = context.getObjectPrimitiveUUIDs(fixed_sphere);
    const std::vector<uint> moving_UUIDs = context.getObjectPrimitiveUUIDs(moving_sphere);
    const std::vector<uint> tube_UUIDs = context.getObjectPrimitiveUUIDs(tube);

    engine.step(100);
    engine.updateContextGeometry();
    context.markGeometryClean();
    engine.step(100);
    engine.updateContextGeometry();

    // A renderer can find what moved from the dirty flags, instead of checking everything
    DOCTEST_CHECK(context.isPrimitiveDirty(moving_UUIDs.front()));
    DOCTEST_CHECK(!context.isPrimitiveDirty(fixed_UUIDs.front()));
    DOCTEST_CHECK(!context.isPrimitiveDirty(tube_UUIDs.front()));

    // Nothing has stepped, so a second copy has nothing to touch
    context.markGeometryClean();
    engine.updateContextGeometry();
    DOCTEST_CHECK(!context.isPrimitiveDirty(moving_UUIDs.front()));

    // Showing another instance moves the geometry back
    uint other = engine.createInstance();
    engine.updateContextGeometry(other);
    DOCTEST_CHECK(context.isPrimitiveDirty(moving_UUIDs.front()));
    vec3 mean = make_vec3(0, 0, 0);
    std::vector<vec3> vertices = getAllVertices(context, {moving_sphere});
    for (const vec3 &vertex: vertices) {
        mean = mean + vertex;
    }
    mean = mean / float(vertices.size());
    DOCTEST_CHECK((mean - pendulum.links[1].position).magnitude() < 1e-3f);
}

DOCTEST_TEST_CASE("PhysicsEngine - body poses can be recorded, and shown again to move the Context geometry") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    PhysicsArticulation pendulum = makePendulum("pendulum", 1.f, 0.5f, 0.3f);
    uint pendulumID = engine.addArticulation(pendulum);
    uint moving_sphere = context.addSphereObject(6, pendulum.links[1].position, 0.05f);
    engine.bindObjectToLink(moving_sphere, pendulumID, 1);
    engine.initialize();

    // One body per link, each once even though the bob also carries the sphere
    DOCTEST_REQUIRE(engine.getDisplayBodyCount() == 2);
    const std::vector<std::string> names = engine.getDisplayBodyNames();
    DOCTEST_CHECK(names == std::vector<std::string>{"pendulum_pivot", "pendulum_bob"});

    // A recorded pose is where the body was, and showing it again puts the geometry back there
    engine.step(100);
    std::vector<float> recorded(7 * 2);
    engine.getDisplayBodyPoses(recorded.data());
    const vec3 recorded_bob = make_vec3(recorded[7], recorded[8], recorded[9]);
    DOCTEST_CHECK((recorded_bob - engine.getLinkPosition(pendulumID, 1)).magnitude() < 1e-6f);
    DOCTEST_CHECK(recorded[10] * recorded[10] + recorded[11] * recorded[11] + recorded[12] * recorded[12] + recorded[13] * recorded[13] == doctest::Approx(1.f));
    engine.step(200);
    engine.updateContextGeometry();

    auto sphereCenter = [&]() {
        vec3 mean = make_vec3(0, 0, 0);
        const std::vector<vec3> vertices = getAllVertices(context, {moving_sphere});
        for (const vec3 &vertex: vertices) {
            mean = mean + vertex;
        }
        return mean / float(vertices.size());
    };
    DOCTEST_REQUIRE((sphereCenter() - recorded_bob).magnitude() > 0.01f);
    engine.showDisplayBodyPoses(recorded.data());
    DOCTEST_CHECK((sphereCenter() - recorded_bob).magnitude() < 1e-3f);

    // A pose from elsewhere, such as another simulator: the bob moved a meter along x, with an unnormalized orientation
    std::vector<float> shifted = recorded;
    shifted[7] += 1.f;
    for (int k = 10; k < 14; k++) {
        shifted[k] *= 2.f;
    }
    engine.showDisplayBodyPoses(shifted.data());
    DOCTEST_CHECK((sphereCenter() - (recorded_bob + make_vec3(1, 0, 0))).magnitude() < 1e-3f);

    std::vector<float> degenerate = recorded;
    std::fill(degenerate.begin() + 10, degenerate.begin() + 14, 0.f);
    DOCTEST_CHECK_THROWS(engine.showDisplayBodyPoses(degenerate.data()));
    DOCTEST_CHECK_THROWS(engine.getDisplayBodyPoses(recorded.data(), 3));
}

DOCTEST_TEST_CASE("PhysicsEngine - a tolerance ignores small motion, and slow drift still arrives") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint pendulumID = engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.2f));
    uint sphere = context.addSphereObject(6, make_vec3(0.1f, 0, 0.5f), 0.05f);
    engine.bindObjectToLink(sphere, pendulumID, 1);
    engine.setContextUpdateTolerance(0.01f);
    DOCTEST_CHECK_THROWS(engine.setContextUpdateTolerance(-1.f));
    engine.initialize();
    const uint UUID = context.getObjectPrimitiveUUIDs(sphere).front();

    engine.updateContextGeometry();
    context.markGeometryClean();

    // One step moves the bob by much less than a centimeter
    engine.step();
    engine.updateContextGeometry();
    DOCTEST_CHECK(!context.isPrimitiveDirty(UUID));

    // Keep stepping: each step is under the tolerance, but what they add up to is not, so the geometry has to move at some point
    bool moved = false;
    for (int s = 0; s < 300 && !moved; s++) {
        engine.step();
        engine.updateContextGeometry();
        moved = context.isPrimitiveDirty(UUID);
    }
    DOCTEST_CHECK(moved);
}

DOCTEST_TEST_CASE("PhysicsEngine - model and state are available directly") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint pendulumID = engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.3f));
    engine.initialize();
    engine.step(50);

    const mjModel *model = engine.getMuJoCoModel();
    const mjData *data = engine.getMuJoCoData();
    int body_ID = mj_name2id(model, mjOBJ_BODY, "pendulum_bob");
    DOCTEST_REQUIRE(body_ID >= 0);
    vec3 position = engine.getLinkPosition(pendulumID, 1);
    DOCTEST_CHECK(float(data->xpos[3 * body_ID]) == position.x);
    DOCTEST_CHECK(float(data->xpos[3 * body_ID + 2]) == position.z);
    DOCTEST_CHECK(float(data->time) == engine.getSimulationTime());
}

DOCTEST_TEST_CASE("PhysicsEngine - shapes and joints of every type compile") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();

    PhysicsArticulation articulation = makePendulum("shapes", 1.f, 0.5f, 0.f);
    PhysicsLink &bob = articulation.links[1];
    bob.stiffness = 1.f;
    bob.damping = 0.1f;
    bob.armature = 1e-4f;
    bob.max_rotation = 1.f;
    PhysicsCollisionShape capsule;
    capsule.type = PHYSICS_SHAPE_CAPSULE;
    capsule.size = make_vec3(0.01f, 0.05f, 0.f);
    bob.shapes.push_back(capsule);
    PhysicsCollisionShape box;
    box.type = PHYSICS_SHAPE_BOX;
    box.size = make_vec3(0.01f, 0.02f, 0.03f);
    box.position = make_vec3(0.1f, 0, 0);
    box.orientation[0] = std::sqrt(0.5);
    box.orientation[3] = std::sqrt(0.5);
    bob.shapes.push_back(box);
    PhysicsCollisionShape ellipsoid;
    ellipsoid.type = PHYSICS_SHAPE_ELLIPSOID;
    ellipsoid.size = make_vec3(0.01f, 0.02f, 0.03f);
    bob.shapes.push_back(ellipsoid);
    PhysicsCollisionShape mesh;
    mesh.type = PHYSICS_SHAPE_CONVEX_MESH;
    mesh.mesh_vertices = {make_vec3(0, 0, 0), make_vec3(0.1f, 0, 0), make_vec3(0, 0.1f, 0), make_vec3(0, 0, 0.1f)};
    bob.shapes.push_back(mesh);

    PhysicsLink welded;
    welded.name = "welded";
    welded.parent_index = 1;
    welded.position = bob.position + make_vec3(0.1f, 0, 0);
    welded.mass = 0.1f;
    welded.inertia_diagonal = make_vec3(1e-5f, 1e-5f, 1e-5f);
    articulation.links.push_back(welded);

    engine.addArticulation(articulation);
    DOCTEST_CHECK_NOTHROW(engine.initialize());
    DOCTEST_CHECK(engine.getLinkCount() == 3);

    const mjModel *model = engine.getMuJoCoModel();
    DOCTEST_CHECK(model->njnt == 1);
    DOCTEST_CHECK(model->nmesh == 1);
    DOCTEST_CHECK_NOTHROW(engine.step(10));
}

DOCTEST_TEST_CASE("PhysicsEngine - misuse is reported") {
    Context context;

    PhysicsEngine engine(&context);
    engine.disableMessages();

    // Nothing to simulate yet
    DOCTEST_CHECK_THROWS(engine.initialize());
    DOCTEST_CHECK_THROWS(engine.step());
    DOCTEST_CHECK_THROWS(engine.stepInstances());
    DOCTEST_CHECK_THROWS(engine.createInstance());
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getMuJoCoModel()));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getMuJoCoData()));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getLinkOverlapCount()));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getMaxLinkOverlapDepth()));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getMaxLinkSpeed()));
    DOCTEST_CHECK_THROWS(engine.addKinematicBox(nullorigin, make_vec3(0.1f, 0.f, 0.1f)));

    // Invalid articulations
    DOCTEST_CHECK_THROWS(engine.addArticulation(PhysicsArticulation()));
    PhysicsArticulation no_links;
    no_links.name = "no_links";
    DOCTEST_CHECK_THROWS(engine.addArticulation(no_links));
    PhysicsArticulation child_first = makePendulum("child_first", 1.f, 0.5f, 0.f);
    std::swap(child_first.links[0], child_first.links[1]);
    DOCTEST_CHECK_THROWS(engine.addArticulation(child_first));
    PhysicsArticulation bad_exclusion = makePendulum("bad_exclusion", 1.f, 0.5f, 0.f);
    bad_exclusion.contact_exclusions.emplace_back(0, 2);
    DOCTEST_CHECK_THROWS(engine.addArticulation(bad_exclusion));

    uint pendulumID = engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.f));
    DOCTEST_CHECK_THROWS(engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.f)));

    // Invalid bindings
    uint sphere_objID = context.addSphereObject(10, make_vec3(0, 0, 0), 0.1f);
    DOCTEST_CHECK_THROWS(engine.bindObjectToLink(sphere_objID + 100, pendulumID, 0));
    DOCTEST_CHECK_THROWS(engine.bindObjectToLink(sphere_objID, pendulumID, 2));
    DOCTEST_CHECK_THROWS(engine.bindObjectToLink(sphere_objID, pendulumID + 1, 0));
    DOCTEST_CHECK_THROWS(engine.bindObjectToBody(sphere_objID + 100, "pendulum_bob"));
    DOCTEST_CHECK_THROWS(engine.addModelFile("does_not_exist.urdf", "x/"));
    DOCTEST_CHECK_THROWS(engine.addPositionServo("j", 0.f, 1.f));
    DOCTEST_CHECK_THROWS(engine.bindTubeNodesToLinks(sphere_objID, pendulumID, {0}));
    uint tube_objID = context.addTubeObject(4, {make_vec3(0, 0, 0), make_vec3(0, 0, 1)}, {0.1f, 0.1f});
    DOCTEST_CHECK_THROWS(engine.bindTubeNodesToLinks(tube_objID, pendulumID, {0}));
    DOCTEST_CHECK_THROWS(engine.bindTubeNodesToLinks(tube_objID, pendulumID, {0, 5}));

    // Invalid settings
    PhysicsEngineParameters params;
    params.contact_time_constant = 1.5f * params.timestep;
    DOCTEST_CHECK_THROWS(engine.initialize(params));
    params = PhysicsEngineParameters();
    params.articulation_collision_margin = -0.001f;
    DOCTEST_CHECK_THROWS(engine.initialize(params));
    params = PhysicsEngineParameters();
    params.timestep = 0.f;
    DOCTEST_CHECK_THROWS(engine.initialize(params));

    engine.initialize();

    DOCTEST_CHECK_THROWS(engine.initialize());
    DOCTEST_CHECK_THROWS(engine.addArticulation(makePendulum("late", 1.f, 0.5f, 0.f)));
    DOCTEST_CHECK_THROWS(engine.addKinematicBox(nullorigin, make_vec3(0.1f, 0.1f, 0.1f)));
    DOCTEST_CHECK_THROWS(engine.bindObjectToLink(sphere_objID, pendulumID, 0));
    DOCTEST_CHECK_THROWS(engine.setKinematicBoxPosition(0, nullorigin, nullorigin));
    DOCTEST_CHECK_THROWS(engine.step(1, 1));
    DOCTEST_CHECK_THROWS(engine.resetInstance(1));
    DOCTEST_CHECK_THROWS(engine.updateContextGeometry(1));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getMuJoCoData(1)));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getLinkPosition(pendulumID + 1, 0)));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getLinkPosition(pendulumID, engine.getLinkCount())));
}

DOCTEST_TEST_CASE("PhysicsEngine - detachable link is held until released, then falls freely") {
    Context context;
    PhysicsArticulation branch = makePendulum("branch", 1.f, 0.5f, 0.f);
    branch.links[1].joint_type = PHYSICS_JOINT_DETACHABLE;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint branchID = engine.addArticulation(branch);
    engine.initialize();
    uint other_instance = engine.createInstance();

    vec3 rest = engine.getLinkPosition(branchID, 1);
    engine.stepInstances(500, 2);

    // Held rigidly against gravity in every instance
    DOCTEST_CHECK(!engine.isLinkDetached(branchID, 1));
    DOCTEST_CHECK((engine.getLinkPosition(branchID, 1) - rest).magnitude() < 5e-3f);
    DOCTEST_CHECK((engine.getLinkPosition(branchID, 1, other_instance) - rest).magnitude() < 5e-3f);

    // Released in one instance only: that one falls as a free body, and the other stays put
    vec3 before_fall = engine.getLinkPosition(branchID, 1);
    engine.detachLink(branchID, 1);
    DOCTEST_CHECK(engine.isLinkDetached(branchID, 1));
    DOCTEST_CHECK(!engine.isLinkDetached(branchID, 1, other_instance));
    const uint fall_steps = 150;
    engine.stepInstances(fall_steps, 2);
    const float fall_time = float(fall_steps) * engine.getTimestep();
    const float fallen = before_fall.z - engine.getLinkPosition(branchID, 1).z;
    DOCTEST_CHECK(fallen == doctest::Approx(0.5f * 9.81f * fall_time * fall_time).epsilon(0.05));
    DOCTEST_CHECK((engine.getLinkPosition(branchID, 1, other_instance) - rest).magnitude() < 5e-3f);

    // Resetting restores the joint
    engine.resetInstance(0);
    DOCTEST_CHECK(!engine.isLinkDetached(branchID, 1));
    DOCTEST_CHECK((engine.getLinkPosition(branchID, 1) - rest).magnitude() < 1e-4f);

    // Only detachable links can be released
    DOCTEST_CHECK_THROWS(engine.detachLink(branchID, 0));
    DOCTEST_CHECK_THROWS(static_cast<void>(engine.isLinkDetached(branchID, 0)));
}

DOCTEST_TEST_CASE("PhysicsEngine - the load on a detachable link is what holds it") {
    Context context;
    PhysicsArticulation branch = makePendulum("branch", 1.f, 0.5f, 0.f);
    branch.links[1].joint_type = PHYSICS_JOINT_DETACHABLE;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint branchID = engine.addArticulation(branch);
    engine.initialize();
    engine.step(500);

    // Holding up a 1 kg link
    DOCTEST_CHECK(engine.getDetachableLinkLoad(branchID, 1) == doctest::Approx(9.81f).epsilon(0.02));

    // Pulling down on it with another 20 N adds to the load
    mjData *data = engine.getMuJoCoData();
    const int body = mj_name2id(engine.getMuJoCoModel(), mjOBJ_BODY, "branch_bob");
    DOCTEST_REQUIRE(body >= 0);
    data->xfrc_applied[6 * body + 2] = -20.;
    engine.step(500);
    DOCTEST_CHECK(engine.getDetachableLinkLoad(branchID, 1) == doctest::Approx(29.81f).epsilon(0.02));

    // Nothing holds it once released
    data->xfrc_applied[6 * body + 2] = 0.;
    engine.detachLink(branchID, 1);
    engine.step();
    DOCTEST_CHECK(engine.getDetachableLinkLoad(branchID, 1) == doctest::Approx(0.f));

    DOCTEST_CHECK_THROWS(static_cast<void>(engine.getDetachableLinkLoad(branchID, 0)));
}

DOCTEST_TEST_CASE("PhysicsEngine - a link with a detach force is released by itself when the load passes it, in its own instance only") {
    Context context;
    PhysicsArticulation branch = makePendulum("branch", 1.f, 0.5f, 0.f);
    branch.links[1].joint_type = PHYSICS_JOINT_DETACHABLE;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint branchID = engine.addArticulation(branch);
    engine.initialize();
    engine.createInstance();
    const int body = mj_name2id(engine.getMuJoCoModel(), mjOBJ_BODY, "branch_bob");

    // Holding up its 1 kg, the link stays below a 15 N limit; pulled down with 10 N more, it passes it
    engine.setDetachForce(branchID, 1, 15.f, 0);
    engine.setDetachForce(branchID, 1, 50.f, 1);
    for (uint instance = 0; instance < 2; instance++) {
        engine.step(300, instance);
        DOCTEST_CHECK_FALSE(engine.isLinkDetached(branchID, 1, instance));
        engine.getMuJoCoData(instance)->xfrc_applied[6 * body + 2] = -10.;
        engine.step(300, instance);
    }
    DOCTEST_CHECK(engine.isLinkDetached(branchID, 1, 0));
    DOCTEST_CHECK(engine.getLoadDetachmentCount(0) == 1);
    DOCTEST_CHECK_FALSE(engine.isLinkDetached(branchID, 1, 1));
    DOCTEST_CHECK(engine.getLoadDetachmentCount(1) == 0);

    // The load at release can be read until the next step
    {
        Context fresh_context;
        PhysicsEngine fresh(&fresh_context);
        fresh.disableMessages();
        uint freshID = fresh.addArticulation(branch);
        fresh.initialize();
        fresh.step(300);
        fresh.setDetachForce(freshID, 1, 15.f);
        fresh.getMuJoCoData()->xfrc_applied[6 * body + 2] = -10.;
        uint steps = 0;
        while (!fresh.isLinkDetached(freshID, 1) && steps < 1000) {
            fresh.step();
            steps++;
        }
        DOCTEST_REQUIRE(fresh.isLinkDetached(freshID, 1));
        DOCTEST_CHECK(fresh.getDetachableLinkLoad(freshID, 1) > 15.f);
    }

    // Reset attaches the link again, clears the count, and keeps the force
    engine.resetInstance(0);
    DOCTEST_CHECK_FALSE(engine.isLinkDetached(branchID, 1, 0));
    DOCTEST_CHECK(engine.getLoadDetachmentCount(0) == 0);
    engine.getMuJoCoData(0)->xfrc_applied[6 * body + 2] = -10.;
    engine.step(300, 0);
    DOCTEST_CHECK(engine.isLinkDetached(branchID, 1, 0));

    // A cleared force releases nothing
    engine.resetInstance(0);
    engine.clearDetachForces(0);
    engine.getMuJoCoData(0)->xfrc_applied[6 * body + 2] = -10.;
    engine.step(300, 0);
    DOCTEST_CHECK_FALSE(engine.isLinkDetached(branchID, 1, 0));

    DOCTEST_CHECK_THROWS(engine.setDetachForce(branchID, 0, 10.f));
    DOCTEST_CHECK_THROWS(engine.setDetachForce(branchID, 1, 0.f));
    DOCTEST_CHECK_THROWS(engine.setDetachForce(branchID, 1, 10.f, 5));
}

DOCTEST_TEST_CASE("PhysicsEngine - instances reset to a pose set from a settled instance") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint pendulumID = engine.addArticulation(makePendulum("pendulum", 1.f, 0.5f, 0.6f));
    engine.initialize();
    const vec3 built = engine.getLinkPosition(pendulumID, 1);

    // Damped by nothing, so stop it where it is by keeping only its pose
    engine.step(100);
    // Body positions are found at the start of each step, so bring them up to the joint positions the step ended with
    mj_forward(engine.getMuJoCoModel(), engine.getMuJoCoData());
    const vec3 moved = engine.getLinkPosition(pendulumID, 1);
    DOCTEST_REQUIRE((moved - built).magnitude() > 0.01f);
    engine.setResetPose(0);

    const uint later = engine.createInstance();
    DOCTEST_CHECK((engine.getLinkPosition(pendulumID, 1, later) - moved).magnitude() < 1e-6f);
    engine.step(50, 0);
    engine.resetInstance(0);
    DOCTEST_CHECK((engine.getLinkPosition(pendulumID, 1, 0) - moved).magnitude() < 1e-6f);
    DOCTEST_CHECK(engine.getMaxLinkSpeed(0) == doctest::Approx(0.f));

    // Both reset instances then move the same way
    engine.resetInstance(later);
    engine.step(50, 0);
    engine.step(50, later);
    DOCTEST_CHECK((engine.getLinkPosition(pendulumID, 1, 0) - engine.getLinkPosition(pendulumID, 1, later)).magnitude() < 1e-7f);

    engine.clearResetPose();
    engine.resetInstance(0);
    DOCTEST_CHECK((engine.getLinkPosition(pendulumID, 1, 0) - built).magnitude() < 1e-6f);
    DOCTEST_CHECK_THROWS(engine.setResetPose(9));
}

DOCTEST_TEST_CASE("PhysicsEngine - a detached link comes to rest on the ground when ground collisions are enabled") {
    auto finalHeight = [](bool ground_collisions) {
        Context context;
        PhysicsArticulation branch = makePendulum("branch", 1.f, 0.5f, 0.f);
        branch.links[1].joint_type = PHYSICS_JOINT_DETACHABLE;
        PhysicsEngine engine(&context);
        engine.disableMessages();
        uint branchID = engine.addArticulation(branch);
        PhysicsEngineParameters params;
        params.ground_collisions = ground_collisions;
        engine.initialize(params);
        engine.detachLink(branchID, 1);
        engine.step(1000);
        return engine.getLinkPosition(branchID, 1).z;
    };
    // The bob's collision sphere has a radius of 2 cm
    DOCTEST_CHECK(finalHeight(true) == doctest::Approx(0.02f).epsilon(0.1));
    DOCTEST_CHECK(finalHeight(false) < -1.f);
}

DOCTEST_TEST_CASE("PhysicsEngine - links that overlap as built can be kept overlapped while every other pair still collides") {
    // Pendulums a and b hang with their bobs 2 cm apart, so that the 2 cm spheres overlap. Pendulum c starts swung out to the side and swings
    // through where a's bob hangs, about 0.35 s later.
    struct Outcome {
        uint excluded_pairs;
        uint overlaps_at_start;
        float a_moved_before_c_arrives;
        float a_moved_by_end;
    };
    auto run = [](bool keep_initial_link_overlaps) {
        Context context;
        PhysicsEngine engine(&context);
        engine.disableMessages();
        uint a = engine.addArticulation(makePendulum("a", 1.f, 0.5f, 0.f));
        PhysicsArticulation b = makePendulum("b", 1.f, 0.5f, 0.f);
        for (PhysicsLink &link: b.links) {
            link.position.x += 0.02f;
        }
        engine.addArticulation(b);
        PhysicsArticulation c = makePendulum("c", 1.f, 0.5f, -1.f);
        for (PhysicsLink &link: c.links) {
            link.position.x -= 0.1f;
        }
        engine.addArticulation(c);
        PhysicsEngineParameters params;
        params.articulation_collisions = true;
        params.keep_initial_link_overlaps = keep_initial_link_overlaps;
        engine.initialize(params);
        Outcome outcome;
        outcome.excluded_pairs = engine.getExcludedInitialOverlapCount();
        outcome.overlaps_at_start = engine.getLinkOverlapCount();
        const vec3 a_start = engine.getLinkPosition(a, 1);
        engine.step(50);
        outcome.a_moved_before_c_arrives = (engine.getLinkPosition(a, 1) - a_start).magnitude();
        engine.step(450);
        outcome.a_moved_by_end = (engine.getLinkPosition(a, 1) - a_start).magnitude();
        return outcome;
    };

    // By default the overlapping bobs push each other apart
    const Outcome pushed = run(false);
    DOCTEST_CHECK(pushed.excluded_pairs == 0);
    DOCTEST_CHECK(pushed.overlaps_at_start == 1);
    DOCTEST_CHECK(pushed.a_moved_before_c_arrives > 0.001f);

    // Kept overlapped, a and b stay as they hang, until c strikes a, which still collides with it
    const Outcome kept = run(true);
    DOCTEST_CHECK(kept.excluded_pairs == 1);
    DOCTEST_CHECK(kept.overlaps_at_start == 0);
    DOCTEST_CHECK(kept.a_moved_before_c_arrives < 1e-4f);
    DOCTEST_CHECK(kept.a_moved_by_end > 0.01f);
}

static void checkTubeSplit(const std::vector<uint> &node_link_indices) {
    Context context;

    // A chain of four links up the z-axis, the third one detachable, and one tube with a node at the base of each link and one at the top
    PhysicsArticulation chain;
    chain.name = "chain";
    const float segment = 0.2f;
    for (int i = 0; i < 4; i++) {
        PhysicsLink link;
        link.name = "s" + std::to_string(i);
        link.parent_index = i - 1;
        link.position = make_vec3(0, 0, (float(i) + 0.5f) * segment);
        link.mass = 0.1f;
        link.inertia_diagonal = make_vec3(1e-4f, 1e-4f, 1e-4f);
        link.anchor_in_parent = make_vec3(0, 0, 0.5f * segment);
        chain.links.push_back(link);
    }
    chain.links[2].joint_type = PHYSICS_JOINT_DETACHABLE;

    std::vector<vec3> nodes;
    for (int i = 0; i <= 4; i++) {
        nodes.push_back(make_vec3(0, 0, float(i) * segment));
    }
    uint tube = context.addTubeObject(6, nodes, std::vector<float>(5, 0.01f));

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint chainID = engine.addArticulation(chain);
    engine.bindTubeNodesToLinks(tube, chainID, node_link_indices);
    engine.initialize();

    const size_t objects_before = context.getAllObjectIDs().size();
    engine.detachLink(chainID, 2);
    engine.splitTubesAtLink(chainID, 2);
    DOCTEST_CHECK(context.getAllObjectIDs().size() == objects_before + 1);

    engine.step(150);
    engine.updateContextGeometry();

    // The original tube keeps the lower links and stays where it was; the new one holds the upper links and has fallen with them
    std::vector<vec3> lower = context.getTubeObjectNodes(tube);
    DOCTEST_REQUIRE(lower.size() == 3);
    DOCTEST_CHECK(lower.back().z == doctest::Approx(2.f * segment).epsilon(0.02));

    uint upper_tube = 0;
    for (uint objID: context.getAllObjectIDs()) {
        if (objID != tube) {
            upper_tube = objID;
        }
    }
    std::vector<vec3> upper = context.getTubeObjectNodes(upper_tube);
    DOCTEST_REQUIRE(upper.size() == 3);
    const float fall = 0.5f * 9.81f * std::pow(150.f * engine.getTimestep(), 2.f);
    DOCTEST_CHECK(upper.front().z == doctest::Approx(2.f * segment - fall).epsilon(0.05));
    // The upper piece fell as one rigid body, so its segments keep their length rather than stretching
    DOCTEST_CHECK((upper[1] - upper[0]).magnitude() == doctest::Approx(segment).epsilon(0.01));
}

DOCTEST_TEST_CASE("PhysicsEngine - a tube split at a detached link parts cleanly") {
    // A node at the junction of two links can be bound to either, and both must part the tube at the same node
    DOCTEST_SUBCASE("junction nodes bound to the lower link") {
        checkTubeSplit({0, 0, 1, 2, 3});
    }
    DOCTEST_SUBCASE("junction nodes bound to the upper link") {
        checkTubeSplit({0, 1, 2, 3, 3});
    }
}

namespace {

    // A hinged arm along +x: base link, then an upper arm on a hinge about y with a sphere collision shape at its tip
    const char *hinged_arm_urdf = R"(<?xml version="1.0"?>
<robot name="hinged_arm">
  <mujoco><compiler fusestatic="false" discardvisual="true"/></mujoco>
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="arm">
    <inertial><origin xyz="0.5 0 0"/><mass value="2"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0.5 0 0"/><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 -1 0"/><limit lower="-3" upper="3" effort="1000" velocity="5"/></joint>
</robot>
)";

    std::string writeHingedArmFile() {
        std::string filename = "physicsengine_test_hinged_arm.urdf";
        std::ofstream file(filename);
        file << hinged_arm_urdf;
        return filename;
    }

} // namespace

DOCTEST_TEST_CASE("PhysicsEngine - a URDF model loads, is placed, is driven by a servo, and carries Context geometry") {
    std::string arm_file = writeHingedArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    const vec3 base = make_vec3(1, 1, 1);
    engine.addModelFile(arm_file, "a/", base, 90.f);
    engine.addPositionServo("a/hinge", 500.f, 50.f);
    const vec3 tip_rest = base + make_vec3(0, 0.5f, 0);
    uint sphere = context.addSphereObject(10, tip_rest, 0.05f);
    engine.bindObjectToBody(sphere, "a/arm");
    engine.initialize();

    const mjModel *model = engine.getMuJoCoModel();
    mjData *data = engine.getMuJoCoData();
    DOCTEST_REQUIRE(mj_name2id(model, mjOBJ_JOINT, "a/hinge") >= 0);
    const int actuator = mj_name2id(model, mjOBJ_ACTUATOR, "a/hinge_servo");
    DOCTEST_REQUIRE(actuator >= 0);

    // Placed with its yaw: the arm's +x points along world +y
    const int arm_body = mj_name2id(model, mjOBJ_BODY, "a/arm");
    DOCTEST_REQUIRE(arm_body >= 0);
    DOCTEST_CHECK(float(data->xpos[3 * arm_body]) == doctest::Approx(base.x).epsilon(1e-4));
    DOCTEST_CHECK(float(data->xpos[3 * arm_body + 1]) == doctest::Approx(base.y).epsilon(1e-4));

    // The hinge axis is -y in the model, so positive angles lift the tip
    const double target = 0.5;
    data->ctrl[actuator] = target;
    engine.step(2000);
    const int joint = mj_name2id(model, mjOBJ_JOINT, "a/hinge");
    DOCTEST_CHECK(float(data->qpos[model->jnt_qposadr[joint]]) == doctest::Approx(float(target)).epsilon(0.02));

    // The bound object rides on the body. The sphere's center is at the tip, which has risen along the arc.
    engine.updateContextGeometry();
    vec3 mean = make_vec3(0, 0, 0);
    std::vector<vec3> vertices = getAllVertices(context, {sphere});
    for (const vec3 &vertex: vertices) {
        mean = mean + vertex;
    }
    mean = mean / float(vertices.size());
    DOCTEST_CHECK(mean.z - base.z == doctest::Approx(0.5f * std::sin(float(target))).epsilon(0.05));
    DOCTEST_CHECK(mean.y - base.y == doctest::Approx(0.5f * std::cos(float(target))).epsilon(0.05));

    // The whole model including the loaded file can be written out, and MuJoCo can load it back
    engine.writeMJCF("physicsengine_test_written.xml");
    char error[1024] = "";
    mjModel *reloaded = mj_loadXML("physicsengine_test_written.xml", nullptr, error, sizeof(error));
    INFO(error);
    DOCTEST_REQUIRE(reloaded != nullptr);
    DOCTEST_CHECK(reloaded->nu == model->nu);
    mj_deleteModel(reloaded);
    std::filesystem::remove("physicsengine_test_written.xml");
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("PhysicsEngine - a model with thousands of colliding shapes has the memory to step") {
    // MuJoCo's collision search needs memory that grows with the square of the number of shapes, and it ends the program when it runs out
    std::string arm_file = writeHingedArmFile();
    Context context;
    PhysicsArticulation forest;
    forest.name = "forest";
    PhysicsLink root;
    root.name = "root";
    root.position = make_vec3(0, 0, 5);
    root.mass = 1.f;
    root.inertia_diagonal = make_vec3(1e-3f, 1e-3f, 1e-3f);
    forest.links.push_back(root);
    const int shape_count = 3000;
    for (int i = 0; i < shape_count; i++) {
        PhysicsLink link;
        link.name = "s" + std::to_string(i);
        link.parent_index = 0;
        link.position = make_vec3(float(i % 60), float(i / 60), 5);
        link.mass = 0.01f;
        link.inertia_diagonal = make_vec3(1e-5f, 1e-5f, 1e-5f);
        PhysicsCollisionShape sphere;
        sphere.type = PHYSICS_SHAPE_SPHERE;
        sphere.size.x = 0.05f;
        link.shapes.push_back(sphere);
        forest.links.push_back(link);
    }
    PhysicsEngine engine(&context);
    engine.disableMessages();
    engine.addArticulation(forest);
    engine.addModelFile(arm_file, "a/");
    engine.initialize();
    DOCTEST_CHECK_NOTHROW(engine.step(5));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("PhysicsEngine - force/torque sensor reads the load on a body") {
    std::string arm_file = writeHingedArmFile();
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    engine.addModelFile(arm_file, "a/");
    engine.addPositionServo("a/hinge", 500.f, 50.f);
    engine.addForceTorqueSensor("wrist", "a/arm");
    engine.initialize();
    engine.step(1000);

    const mjModel *model = engine.getMuJoCoModel();
    const mjData *data = engine.getMuJoCoData();
    const int sensor = mj_name2id(model, mjOBJ_SENSOR, "wrist_force");
    DOCTEST_REQUIRE(sensor >= 0);
    const mjtNum *force = data->sensordata + model->sensor_adr[sensor];

    // At rest, holding up a 2 kg arm
    DOCTEST_CHECK(float(mju_norm3(force)) == doctest::Approx(2.f * 9.81f).epsilon(0.02));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("PhysicsEngine - a loaded model pushes articulations, and the contact force matches the joint spring") {
    std::string arm_file = writeHingedArmFile();
    Context context;

    // A horizontal rod along y on an elastic joint at its end, just above the arc of the arm's tip at x = 0.5
    const float rod_half_length = 0.3f;
    const float joint_stiffness = 2.f; // N*m/rad
    const float rod_height = 0.12f;
    PhysicsArticulation rod;
    rod.name = "rod";
    PhysicsLink root;
    root.name = "root";
    root.position = make_vec3(0.5f, -rod_half_length, rod_height);
    root.mass = 0.01f;
    root.inertia_diagonal = make_vec3(1e-6f, 1e-6f, 1e-6f);
    rod.links.push_back(root);
    PhysicsLink shaft;
    shaft.name = "shaft";
    shaft.parent_index = 0;
    shaft.position = make_vec3(0.5f, 0, rod_height);
    // Local z, the capsule axis, along world y
    shaft.qw = std::sqrt(0.5f);
    shaft.qx = -std::sqrt(0.5f);
    shaft.mass = 0.05f;
    shaft.inertia_diagonal = make_vec3(1e-3f, 1e-3f, 1e-5f);
    shaft.joint_type = PHYSICS_JOINT_BALL;
    shaft.stiffness = joint_stiffness;
    shaft.damping = 0.5f;
    PhysicsCollisionShape capsule;
    capsule.type = PHYSICS_SHAPE_CAPSULE;
    capsule.size = make_vec3(0.005f, rod_half_length, 0);
    shaft.shapes.push_back(capsule);
    rod.links.push_back(shaft);

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint rodID = engine.addArticulation(rod);
    engine.addModelFile(arm_file, "a/");
    engine.addPositionServo("a/hinge", 500.f, 50.f);
    PhysicsEngineParameters params;
    params.gravity = false;
    engine.initialize(params);

    mjData *data = engine.getMuJoCoData();
    const int actuator = mj_name2id(engine.getMuJoCoModel(), mjOBJ_ACTUATOR, "a/hinge_servo");
    DOCTEST_REQUIRE(actuator >= 0);

    // Clear of the rod at the start
    engine.step(10);
    DOCTEST_REQUIRE(engine.getContactCount() == 0);
    DOCTEST_CHECK(engine.getExternalContactForce(rodID).magnitude() == doctest::Approx(0.f));

    // Lift the tip into the rod slowly, and hold
    const uint ramp_steps = 1500;
    for (uint s = 0; s < ramp_steps; s++) {
        data->ctrl[actuator] = 0.3 * double(s + 1) / double(ramp_steps);
        engine.step();
    }
    engine.step(1000);
    DOCTEST_REQUIRE(engine.getContactCount() > 0);

    // The rod is pushed upward, and at rest the contact torque about its joint balances the spring: F * L = K * angle
    const vec3 force = engine.getExternalContactForce(rodID);
    const float angle = engine.getMaxLinkDisplacement() / rod_half_length;
    const float expected_force = joint_stiffness * angle / rod_half_length;
    INFO("force " << force.x << " " << force.y << " " << force.z << " N, angle " << rad2deg(angle) << " deg, expected " << expected_force << " N");
    DOCTEST_CHECK(angle > deg2rad(1.f));
    DOCTEST_CHECK(force.z > 0.f);
    DOCTEST_CHECK(force.z == doctest::Approx(expected_force).epsilon(0.15));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("ContactSensor - splits the force on a loaded model by the label of what it touches, with a target per instance") {
    std::string arm_file = writeHingedArmFile();
    Context context;

    // A horizontal rod labeled as wood just above the arc of the arm's tip, as in the test above
    const float rod_height = 0.12f;
    PhysicsArticulation rod;
    rod.name = "rod";
    PhysicsLink root;
    root.name = "root";
    root.position = make_vec3(0.5f, -0.3f, rod_height);
    root.mass = 0.01f;
    root.inertia_diagonal = make_vec3(1e-6f, 1e-6f, 1e-6f);
    rod.links.push_back(root);
    PhysicsLink shaft;
    shaft.name = "shaft";
    shaft.label = "wood";
    shaft.parent_index = 0;
    shaft.position = make_vec3(0.5f, 0, rod_height);
    shaft.qw = std::sqrt(0.5f);
    shaft.qx = -std::sqrt(0.5f);
    shaft.mass = 0.05f;
    shaft.inertia_diagonal = make_vec3(1e-3f, 1e-3f, 1e-5f);
    shaft.joint_type = PHYSICS_JOINT_BALL;
    shaft.stiffness = 2.f;
    shaft.damping = 0.5f;
    PhysicsCollisionShape capsule;
    capsule.type = PHYSICS_SHAPE_CAPSULE;
    capsule.size = make_vec3(0.005f, 0.3f, 0);
    shaft.shapes.push_back(capsule);
    rod.links.push_back(shaft);

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint rodID = engine.addArticulation(rod);
    engine.addModelFile(arm_file, "a/");
    engine.addPositionServo("a/hinge", 500.f, 50.f);
    PhysicsEngineParameters params;
    params.gravity = false;
    engine.initialize(params);
    engine.createInstance();

    DOCTEST_CHECK(engine.getArticulationCount() == 1);
    DOCTEST_CHECK(engine.getArticulationLinkCount(rodID) == 2);
    DOCTEST_CHECK(engine.getLinkLabel(rodID, 1) == "wood");
    DOCTEST_CHECK(engine.getLinkLabel(rodID, 0).empty());
    DOCTEST_CHECK(engine.getLinkBodyID(rodID, 1) == mj_name2id(engine.getMuJoCoModel(), mjOBJ_BODY, "rod_shaft"));

    ContactSensor sensor(&engine, "a/base");
    const std::vector<std::string> &partners = sensor.getPartnerNames();
    for (const std::string &expected: {"wood", "articulation", "target", "ground", "other"}) {
        DOCTEST_CHECK(std::find(partners.begin(), partners.end(), expected) != partners.end());
    }
    DOCTEST_CHECK(sensor.getPreparedInstanceCount() == 2);

    // Lift the tip into the rod in both instances, with the rod as the target in instance 1 only
    sensor.setTargetLink(rodID, 1, 1);
    const int actuator = mj_name2id(engine.getMuJoCoModel(), mjOBJ_ACTUATOR, "a/hinge_servo");
    for (uint instance = 0; instance < 2; instance++) {
        engine.step(10, instance);
        DOCTEST_CHECK(sensor.getTotalForce({}, instance).magnitude() == doctest::Approx(0.f));
        DOCTEST_CHECK_FALSE(sensor.isPenetrating({}, instance));
        for (uint s = 0; s < 1500; s++) {
            engine.getMuJoCoData(instance)->ctrl[actuator] = 0.3 * double(s + 1) / 1500.0;
            engine.step(1, instance);
        }
        engine.step(1000, instance);
    }

    // The arm feels the reaction to the force on the rod, and only under the rod's label
    const vec3 on_rod = engine.getExternalContactForce(rodID, 0);
    DOCTEST_REQUIRE(on_rod.z > 0.01f);
    const vec3 wood = sensor.getForce("wood", 0);
    DOCTEST_CHECK((wood + on_rod).magnitude() < 1e-4f * on_rod.magnitude());
    DOCTEST_CHECK((sensor.getTotalForce({}, 0) - wood).magnitude() < 1e-6f);
    DOCTEST_CHECK(sensor.getTotalForce({"wood"}, 0).magnitude() == doctest::Approx(0.f));
    DOCTEST_CHECK(sensor.getForce("target", 0).magnitude() == doctest::Approx(0.f));
    const std::vector<vec3> forces = sensor.getForces(0);
    DOCTEST_REQUIRE(forces.size() == partners.size());
    const size_t wood_index = std::find(partners.begin(), partners.end(), "wood") - partners.begin();
    DOCTEST_CHECK((forces[wood_index] - wood).magnitude() < 1e-6f);

    // In instance 1 the same contact is with the target
    DOCTEST_CHECK(sensor.getForce("wood", 1).magnitude() == doctest::Approx(0.f));
    DOCTEST_CHECK((sensor.getForce("target", 1) - wood).magnitude() < 1e-3f * wood.magnitude());
    sensor.clearTargetLink(1);
    DOCTEST_CHECK((sensor.getForce("wood", 1) - wood).magnitude() < 1e-3f * wood.magnitude());

    // The tip rests slightly inside the rod, held there by the contact
    DOCTEST_CHECK(sensor.isPenetrating({}, 0));
    DOCTEST_CHECK_FALSE(sensor.isPenetrating({"wood"}, 0));

    // Sensing the arm body alone sees the same contact, since the tip is on it
    ContactSensor arm_sensor(&engine, "a/arm");
    DOCTEST_CHECK((arm_sensor.getForce("wood", 0) - wood).magnitude() < 1e-6f);

    // Instances created after the sensor need prepareInstances()
    const uint late_instance = engine.createInstance();
    DOCTEST_CHECK_THROWS(sensor.setTargetLink(rodID, 1, late_instance));
    sensor.prepareInstances();
    DOCTEST_CHECK_NOTHROW(sensor.setTargetLink(rodID, 1, late_instance));

    DOCTEST_CHECK_THROWS(static_cast<void>(sensor.getForce("leaf", 0)));
    DOCTEST_CHECK_THROWS(static_cast<void>(sensor.getTotalForce({"leaf"}, 0)));
    DOCTEST_CHECK_THROWS(static_cast<void>(sensor.getForces(5)));
    DOCTEST_CHECK_THROWS(ContactSensor(&engine, "a/missing"));
    DOCTEST_CHECK_THROWS(ContactSensor(&engine, "world"));
    std::filesystem::remove(arm_file);
}

DOCTEST_TEST_CASE("ContactSensor - cannot be created before the simulation is initialized") {
    Context context;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    DOCTEST_CHECK_THROWS(ContactSensor(&engine, "a/base"));
    DOCTEST_CHECK_THROWS(ContactSensor(nullptr, "a/base"));
}

#ifdef HELIOS_PHYSICS_ENGINE_PLANTS

DOCTEST_TEST_CASE("PlantPhysics - model compiles with one body per articulation link") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);

    PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);
    DOCTEST_REQUIRE(articulation.links.size() > 10);

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint articulationID = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID);
    DOCTEST_CHECK_NOTHROW(engine.initialize());

    DOCTEST_CHECK(engine.getLinkCount() == articulation.links.size());
    DOCTEST_CHECK(engine.getSimulationTime() == doctest::Approx(0.f));

    // Before any step, every link is at its articulation rest position
    for (uint i = 0; i < articulation.links.size(); i++) {
        vec3 position = engine.getLinkPosition(articulationID, i);
        DOCTEST_CHECK((position - articulation.links[i].position).magnitude() < 1e-5f);
    }
}

DOCTEST_TEST_CASE("PlantPhysics - plant settles under gravity without diverging") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);

    vec3 lower, upper;
    context.getObjectBoundingBox(plantarchitecture.getAllPlantObjectIDs(plantID), lower, upper);
    float height = upper.z - lower.z;

    PhysicsEngine engine(&context);
    engine.disableMessages();
    addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID);
    engine.initialize();

    uint steps = static_cast<uint>(std::round(1.f / engine.getTimestep()));
    DOCTEST_CHECK_NOTHROW(engine.step(steps));
    float sag_after_1s = engine.getMaxLinkDisplacement();
    DOCTEST_CHECK_NOTHROW(engine.step(steps));
    float sag_after_2s = engine.getMaxLinkDisplacement();

    // Gravity bends the plant a little, and it comes to rest rather than drifting or collapsing
    DOCTEST_CHECK(sag_after_1s > 0.f);
    DOCTEST_CHECK(sag_after_2s < 0.25f * height);
    DOCTEST_CHECK(std::abs(sag_after_2s - sag_after_1s) < 0.1f * sag_after_1s + 1e-3f);
}

DOCTEST_TEST_CASE("PlantPhysics - kinematic box sweeping through the plant displaces it") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);

    vec3 lower, upper;
    context.getObjectBoundingBox(plantarchitecture.getAllPlantObjectIDs(plantID), lower, upper);
    vec3 center = 0.5f * (lower + upper);
    float height = upper.z - lower.z;
    float box_size = 0.2f * height;
    vec3 box_start = make_vec3(lower.x - box_size, center.y, lower.z + 0.75f * height);
    vec3 box_end = make_vec3(upper.x + box_size, center.y, box_start.z);

    // Gravity-only reference run, so that the sweep is judged against the sag it would have anyway
    float sag_only = 0.f;
    {
        Context reference_context;
        PlantArchitecture reference_plantarchitecture(&reference_context);
        uint reference_plantID = buildTestPlant(reference_context, reference_plantarchitecture);
        PhysicsEngine reference(&reference_context);
        reference.disableMessages();
        addPlantToPhysicsEngine(reference, reference_context, reference_plantarchitecture, reference_plantID);
        reference.initialize();
        uint steps = static_cast<uint>(std::round(1.5f / reference.getTimestep()));
        for (uint s = 0; s < steps; s++) {
            reference.step();
            sag_only = std::max(sag_only, reference.getMaxLinkDisplacement());
        }
    }

    PhysicsEngine engine(&context);
    engine.disableMessages();
    addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID);
    uint boxID = engine.addKinematicBox(box_start, make_vec3(box_size, box_size, box_size));
    engine.initialize();

    const float travel_time = 1.5f;
    uint steps = static_cast<uint>(std::round(travel_time / engine.getTimestep()));
    uint max_contacts = 0;
    float max_displacement = 0.f;
    for (uint s = 0; s < steps; s++) {
        float fraction = float(s + 1) / float(steps);
        engine.setKinematicBoxPosition(boxID, box_start + fraction * (box_end - box_start), (box_end - box_start) / travel_time);
        DOCTEST_REQUIRE_NOTHROW(engine.step());
        max_contacts = std::max(max_contacts, engine.getContactCount());
        max_displacement = std::max(max_displacement, engine.getMaxLinkDisplacement());
    }

    DOCTEST_CHECK(max_contacts > 0);
    DOCTEST_CHECK(max_displacement > 5.f * sag_only);
    DOCTEST_CHECK(max_displacement > 0.05f * height);
}

DOCTEST_TEST_CASE("PlantPhysics - giving the box velocity and stiff contacts keeps branches out of it") {

    struct SweepPenetration {
        float mean = 0.f; // mean over the steps with contact of the deepest penetration
        float max = 0.f;
        uint contact_steps = 0;
    };

    // Sweep a box through an identical plant each time, so that the two contact settings are compared on the same geometry
    auto sweep = [](float contact_time_constant, bool give_box_velocity) {
        Context context;
        context.seedRandomGenerator(11);
        PlantArchitecture plantarchitecture(&context);
        uint plantID = buildTestPlant(context, plantarchitecture);

        vec3 lower, upper;
        context.getObjectBoundingBox(plantarchitecture.getAllPlantObjectIDs(plantID), lower, upper);
        vec3 center = 0.5f * (lower + upper);
        float height = upper.z - lower.z;
        float box_size = 0.2f * height;
        vec3 box_start = make_vec3(lower.x - box_size, center.y, lower.z + 0.75f * height);
        vec3 box_end = make_vec3(upper.x + box_size, center.y, box_start.z);

        // Leaves are thin plates whose contact depth against a box is poorly defined, so only stems and petioles collide here
        PlantPhysicsParameters plant_params;
        plant_params.leaf_collisions = false;
        PhysicsEngineParameters params;
        params.contact_time_constant = contact_time_constant;

        PhysicsEngine engine(&context);
        engine.disableMessages();
        addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), plant_params);
        uint boxID = engine.addKinematicBox(box_start, make_vec3(box_size, box_size, box_size));
        engine.initialize(params);

        const float speed = 0.5f;
        const float travel_time = (box_end.x - box_start.x) / speed;
        uint steps = static_cast<uint>(std::round(travel_time / engine.getTimestep()));
        SweepPenetration result;
        for (uint s = 0; s < steps; s++) {
            float fraction = float(s) / float(steps);
            engine.setKinematicBoxPosition(boxID, box_start + fraction * (box_end - box_start), give_box_velocity ? make_vec3(speed, 0, 0) : make_vec3(0, 0, 0));
            engine.step();
            if (engine.getContactCount() > 0) {
                float penetration = engine.getMaxKinematicBoxPenetration();
                result.mean += penetration;
                result.max = std::max(result.max, penetration);
                result.contact_steps++;
            }
        }
        if (result.contact_steps > 0) {
            result.mean /= float(result.contact_steps);
        }
        return result;
    };

    SweepPenetration recommended = sweep(PhysicsEngineParameters().contact_time_constant, true);
    SweepPenetration soft_teleported = sweep(0.02f, false);

    DOCTEST_REQUIRE(recommended.contact_steps > 10);
    DOCTEST_REQUIRE(soft_teleported.contact_steps > 10);
    DOCTEST_CHECK(recommended.mean < 0.5f * soft_teleported.mean);
    DOCTEST_CHECK(recommended.max < soft_teleported.max);
    DOCTEST_CHECK(recommended.mean < 0.003f);
}

DOCTEST_TEST_CASE("PlantPhysics - Context geometry follows the simulated links") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);
    std::vector<uint> plant_objIDs = plantarchitecture.getAllPlantObjectIDs(plantID);

    vec3 lower, upper;
    context.getObjectBoundingBox(plant_objIDs, lower, upper);
    vec3 center = 0.5f * (lower + upper);
    float height = upper.z - lower.z;
    vec3 box_center = make_vec3(lower.x - height, center.y, center.z);

    std::vector<vec3> rest_vertices = getAllVertices(context, plant_objIDs);

    PhysicsEngine engine(&context);
    engine.disableMessages();
    addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID);
    uint boxID = engine.addKinematicBox(box_center, make_vec3(0.1f, 0.1f, 0.1f));
    engine.initialize();

    // At the rest pose the write-back leaves the plant where it was
    engine.updateContextGeometry();
    std::vector<vec3> vertices = getAllVertices(context, plant_objIDs);
    DOCTEST_REQUIRE(vertices.size() == rest_vertices.size());
    float max_change = 0.f;
    for (size_t i = 0; i < vertices.size(); i++) {
        max_change = std::max(max_change, (vertices[i] - rest_vertices[i]).magnitude());
    }
    DOCTEST_CHECK(max_change < 1e-4f);

    // Moving the box, which is out of reach of the plant, moves its Context box object by the same amount
    std::vector<uint> box_objIDs;
    for (uint objID: context.getAllObjectIDs()) {
        if (std::find(plant_objIDs.begin(), plant_objIDs.end(), objID) == plant_objIDs.end()) {
            box_objIDs.push_back(objID);
        }
    }
    DOCTEST_REQUIRE(box_objIDs.size() == 1);
    vec3 box_rest_center = context.getObjectCenter(box_objIDs.front());
    vec3 shift = make_vec3(0.f, 0.f, 0.05f);
    engine.setKinematicBoxPosition(boxID, box_center + shift, make_vec3(0, 0, 0));
    engine.step(static_cast<uint>(std::round(0.5f / engine.getTimestep())));
    engine.updateContextGeometry();
    vec3 box_moved = make_vec3(0, 0, 0);
    for (const vec3 &vertex: getAllVertices(context, box_objIDs)) {
        box_moved = box_moved + vertex;
    }
    box_moved = box_moved / float(getAllVertices(context, box_objIDs).size()) - box_rest_center;
    DOCTEST_CHECK((box_moved - shift).magnitude() < 1e-4f);

    // After sagging under gravity the plant geometry has moved by about the largest link displacement, and the plant base has not moved
    vertices = getAllVertices(context, plant_objIDs);
    max_change = 0.f;
    for (size_t i = 0; i < vertices.size(); i++) {
        max_change = std::max(max_change, (vertices[i] - rest_vertices[i]).magnitude());
    }
    float max_link_displacement = engine.getMaxLinkDisplacement();
    DOCTEST_CHECK(max_change > 0.5f * max_link_displacement);
    DOCTEST_CHECK(max_change < 2.f * max_link_displacement + 0.01f);

    uint base_tube = plantarchitecture.getPlantInternodeObjectIDs(plantID).front();
    vec3 base_node = context.getTubeObjectNodes(base_tube).front();
    DOCTEST_CHECK(base_node.magnitude() < 1e-4f);
}

DOCTEST_TEST_CASE("PlantPhysics - plant self-collisions push overlapping organs apart") {
    // Two plants 3 cm apart, whose leaves and stems interpenetrate as built
    Context context;
    context.seedRandomGenerator(5);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("bean");
    std::vector<uint> plantIDs = {plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 25), plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0.03f, 0, 0), 25)};

    PlantPhysicsParameters plant_params;
    plant_params.leaf_collision_shape = "ellipsoid";
    PhysicsEngineParameters params;
    params.gravity = false;

    {
        PhysicsEngine engine(&context);
        engine.disableMessages();
        for (uint plantID: plantIDs) {
            addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), plant_params);
        }
        engine.initialize(params);
        DOCTEST_CHECK(engine.getLinkOverlapCount() == 0);
        DOCTEST_CHECK(engine.getMaxLinkOverlapDepth() == doctest::Approx(0.f));
    }

    params.articulation_collisions = true;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    for (uint plantID: plantIDs) {
        addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), plant_params);
    }
    engine.initialize(params);
    const float tolerance = 0.0005f;
    uint overlaps_before = engine.getLinkOverlapCount(tolerance);
    float depth_before = engine.getMaxLinkOverlapDepth();
    DOCTEST_REQUIRE(overlaps_before >= 5);
    DOCTEST_CHECK(engine.getMaxLinkSpeed() == doctest::Approx(0.f));

    engine.step(500);
    uint overlaps_after = engine.getLinkOverlapCount(tolerance);
    INFO("overlaps " << overlaps_before << " -> " << overlaps_after << ", deepest " << 1000.f * depth_before << " -> " << 1000.f * engine.getMaxLinkOverlapDepth() << " mm");
    DOCTEST_CHECK(overlaps_after * 3 <= overlaps_before);
    DOCTEST_CHECK(engine.getMaxLinkOverlapDepth() < depth_before);
    DOCTEST_CHECK(engine.getMaxLinkDisplacement() > 0.001f);
}

DOCTEST_TEST_CASE("PlantPhysics - per-triangle leaf shapes push leaves out of stems") {
    // A young apple tree, whose leaves have few triangles and whose blades cross the shoot above them as built
    Context context;
    context.seedRandomGenerator(7);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 165);
    plantarchitecture.advanceTime(plantID, 5);

    PlantPhysicsParameters plant_params;
    plant_params.rigid_internodes = true;
    plant_params.leaf_collision_shape = "triangles";
    PhysicsEngineParameters params;
    params.gravity = false;
    params.articulation_collisions = true;
    PhysicsEngine engine(&context);
    engine.disableMessages();
    addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), plant_params);
    engine.initialize(params);

    const float tolerance = 0.0005f;
    uint overlaps_before = engine.getLinkOverlapCount(tolerance);
    DOCTEST_REQUIRE(overlaps_before >= 1);
    engine.step(200);
    INFO("overlaps " << overlaps_before << " -> " << engine.getLinkOverlapCount(tolerance));
    DOCTEST_CHECK(engine.getLinkOverlapCount(tolerance) < overlaps_before);
}

DOCTEST_TEST_CASE("PlantPhysics - rigid internodes hold the stems still") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);
    PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);

    // Largest motion of the internode links after sagging under gravity
    auto internodeSag = [&](bool rigid_internodes) {
        PlantPhysicsParameters plant_params;
        plant_params.rigid_internodes = rigid_internodes;
        PhysicsEngine engine(&context);
        engine.disableMessages();
        uint articulationID = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), plant_params);
        engine.initialize();
        engine.step(250);
        float max_motion = 0.f;
        for (uint i = 0; i < articulation.links.size(); i++) {
            if (articulation.links[i].type == ARTICULATION_LINK_INTERNODE) {
                max_motion = std::max(max_motion, (engine.getLinkPosition(articulationID, i) - articulation.links[i].position).magnitude());
            }
        }
        return max_motion;
    };

    float flexible_sag = internodeSag(false);
    float rigid_sag = internodeSag(true);
    INFO("internode sag: flexible " << 1000.f * flexible_sag << " mm, rigid " << 1000.f * rigid_sag << " mm");
    DOCTEST_CHECK(flexible_sag > 2e-4f);
    DOCTEST_CHECK(rigid_sag < 1e-5f);
}

DOCTEST_TEST_CASE("PlantPhysics - listed internodes stay flexible when the others are rigid") {
    Context context;
    context.seedRandomGenerator(3);
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);
    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);

    // The top internode segment, and its parent, which is a stem joint with something to bend
    uint top_link = 0;
    float top = -1.f;
    for (uint i = 0; i < articulation.links.size(); i++) {
        const PlantArticulationLink &link = articulation.links[i];
        if (link.type == ARTICULATION_LINK_INTERNODE && link.parent_index >= 0 && link.position.z > top) {
            top = link.position.z;
            top_link = i;
        }
    }

    auto modelSize = [&](const std::vector<uint> &flexible_links) {
        PlantPhysicsParameters params;
        params.rigid_internodes = true;
        params.leaf_collisions = false;
        params.flexible_links = flexible_links;
        PhysicsEngine engine(&context);
        engine.disableMessages();
        uint articulationID = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), params);
        engine.initialize();
        engine.step(250);
        return std::make_pair(static_cast<int>(engine.getMuJoCoModel()->nv), (engine.getLinkPosition(articulationID, top_link) - articulation.links[top_link].position).magnitude());
    };

    const auto [rigid_dof, rigid_sag] = modelSize({});
    const auto [flexible_dof, flexible_sag] = modelSize({top_link});

    // One ball joint more, and that link now moves when it did not before
    DOCTEST_CHECK(flexible_dof == rigid_dof + 3);
    DOCTEST_CHECK(flexible_sag > rigid_sag);

    PlantPhysicsParameters params;
    params.flexible_links = {static_cast<uint>(articulation.links.size())};
    PhysicsEngine engine(&context);
    engine.disableMessages();
    DOCTEST_CHECK_THROWS(addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), params));
}

DOCTEST_TEST_CASE("PlantPhysics - rigid leaves have no joints of their own, and follow the wood") {
    Context context;
    context.seedRandomGenerator(3);
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);
    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);

    // The relative position of each leaf and its parent, in the rest pose and after sagging
    auto simulate = [&](bool rigid_leaves) {
        PlantPhysicsParameters params;
        params.leaf_collisions = false;
        params.rigid_leaves = rigid_leaves;
        PhysicsEngine engine(&context);
        engine.disableMessages();
        uint articulationID = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), params);
        engine.initialize();
        engine.step(250);
        float largest_leaf_motion_in_world = 0.f;
        float largest_change_from_parent = 0.f;
        for (uint i = 0; i < articulation.links.size(); i++) {
            const PlantArticulationLink &link = articulation.links[i];
            if (link.type != ARTICULATION_LINK_LEAF) {
                continue;
            }
            const vec3 position = engine.getLinkPosition(articulationID, i);
            const vec3 parent_position = engine.getLinkPosition(articulationID, static_cast<uint>(link.parent_index));
            largest_leaf_motion_in_world = std::max(largest_leaf_motion_in_world, (position - link.position).magnitude());
            const float rest_distance = (link.position - articulation.links[link.parent_index].position).magnitude();
            largest_change_from_parent = std::max(largest_change_from_parent, std::abs((position - parent_position).magnitude() - rest_distance));
        }
        return std::make_tuple(static_cast<int>(engine.getMuJoCoModel()->nv), largest_leaf_motion_in_world, largest_change_from_parent);
    };

    const auto [flexible_dof, flexible_motion, unused_change] = simulate(false);
    static_cast<void>(unused_change);
    const auto [rigid_dof, rigid_motion, rigid_change] = simulate(true);

    int leaf_and_petiole_joints = 0;
    for (const PlantArticulationLink &link: articulation.links) {
        if (link.type == ARTICULATION_LINK_LEAF || link.type == ARTICULATION_LINK_PETIOLE) {
            leaf_and_petiole_joints++;
        }
    }
    DOCTEST_REQUIRE(leaf_and_petiole_joints > 0);
    // A ball joint has three degrees of freedom
    DOCTEST_CHECK(flexible_dof - rigid_dof == 3 * leaf_and_petiole_joints);
    // The stem is still flexible, so the leaves still move with it, but each keeps its place on what it grows from
    DOCTEST_CHECK(rigid_motion > 1e-4f);
    DOCTEST_CHECK(rigid_change < 1e-5f);
    DOCTEST_CHECK(flexible_motion > 0.f);
}

DOCTEST_TEST_CASE("PlantPhysics - links are labeled by organ, and a contact sensor reports them by label") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);
    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);

    DOCTEST_CHECK(plantLinkLabel(ARTICULATION_LINK_INTERNODE) == "wood");
    DOCTEST_CHECK(plantLinkLabel(ARTICULATION_LINK_FRUIT) == "fruit");
    DOCTEST_CHECK(plantLinkLabel(ARTICULATION_LINK_LEAF) == "leaf");

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint articulationID = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID);
    engine.addModelFile(writeHingedArmFile(), "a/", make_vec3(2, 2, 0));
    engine.initialize();
    std::filesystem::remove("physicsengine_test_hinged_arm.urdf");

    std::set<std::string> labels;
    for (uint i = 0; i < articulation.links.size(); i++) {
        DOCTEST_CHECK(engine.getLinkLabel(articulationID, i) == plantLinkLabel(articulation.links[i].type));
        labels.insert(engine.getLinkLabel(articulationID, i));
    }
    DOCTEST_CHECK(labels.count("wood") == 1);
    DOCTEST_CHECK(labels.count("leaf") == 1);

    ContactSensor sensor(&engine, "a/base");
    const std::vector<std::string> &partners = sensor.getPartnerNames();
    for (const std::string &label: labels) {
        DOCTEST_CHECK(std::find(partners.begin(), partners.end(), label) != partners.end());
    }
    // Every link has a label, so none is reported as an unlabeled articulation
    DOCTEST_CHECK(std::find(partners.begin(), partners.end(), "articulation") == partners.end());
}

DOCTEST_TEST_CASE("PlantPhysics - the nearest organ to a point is found among organs of one type at rest") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);
    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);
    NearestPlantOrgan nearest_wood(articulation, ARTICULATION_LINK_INTERNODE);

    // On the axis of an internode, the point is inside that internode by its radius
    uint probe_link = 0;
    for (uint i = 0; i < articulation.links.size(); i++) {
        if (articulation.links[i].type == ARTICULATION_LINK_INTERNODE && articulation.links[i].radius > articulation.links[probe_link].radius) {
            probe_link = i;
        }
    }
    const PlantArticulationLink &probe = articulation.links[probe_link];
    const NearestPlantOrganResult inside = nearest_wood.find(probe.position);
    DOCTEST_CHECK(inside.surface_distance == doctest::Approx(-probe.radius).epsilon(1e-3));
    DOCTEST_CHECK(inside.axis_offset.magnitude() < 1e-5f);
    DOCTEST_CHECK(articulation.links[inside.link_index].type == ARTICULATION_LINK_INTERNODE);

    // Away from the plant, the nearest internode is the one a search of every internode finds, and the offset reaches its axis
    const vec3 point = make_vec3(0.3f, -0.2f, 0.25f);
    const NearestPlantOrganResult found = nearest_wood.find(point);
    float best = std::numeric_limits<float>::max();
    for (const PlantArticulationLink &link: articulation.links) {
        if (link.type != ARTICULATION_LINK_INTERNODE) {
            continue;
        }
        const vec3 axis = make_vec3(2.f * (link.qx * link.qz + link.qw * link.qy), 2.f * (link.qy * link.qz - link.qw * link.qx), 1.f - 2.f * (link.qx * link.qx + link.qy * link.qy));
        const vec3 start = link.position - link.half_length * axis;
        const float t = std::clamp((point - start) * axis, 0.f, 2.f * link.half_length);
        best = std::min(best, (start + t * axis - point).magnitude() - link.radius);
    }
    DOCTEST_CHECK(found.surface_distance == doctest::Approx(best).epsilon(1e-4));
    DOCTEST_CHECK(found.surface_distance > 0.f);
    DOCTEST_CHECK(found.axis_offset.magnitude() - articulation.links[found.link_index].radius == doctest::Approx(found.surface_distance).epsilon(1e-4));

    DOCTEST_CHECK_THROWS(NearestPlantOrgan(articulation, ARTICULATION_LINK_LEAF));
    PlantArticulation no_wood;
    DOCTEST_CHECK_THROWS(NearestPlantOrgan(no_wood, ARTICULATION_LINK_INTERNODE));
}

DOCTEST_TEST_CASE("PlantPhysics - organs are found in a box, and the internodes that carry an organ are found toward the root") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);
    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID);
    const float unbounded = std::numeric_limits<float>::infinity();

    // An unbounded box holds every leaf; one bounded in height only those in that band
    const std::vector<uint> leaves = findPlantLinksInBox(articulation, ARTICULATION_LINK_LEAF, make_vec3(-unbounded, -unbounded, -unbounded), make_vec3(unbounded, unbounded, unbounded));
    const size_t leaf_count = std::count_if(articulation.links.begin(), articulation.links.end(), [](const PlantArticulationLink &link) { return link.type == ARTICULATION_LINK_LEAF; });
    DOCTEST_REQUIRE(leaves.size() == leaf_count);
    DOCTEST_REQUIRE(leaf_count > 2);
    float top = -unbounded;
    for (uint leaf: leaves) {
        top = std::max(top, articulation.links[leaf].position.z);
    }
    const std::vector<uint> high_leaves = findPlantLinksInBox(articulation, ARTICULATION_LINK_LEAF, make_vec3(-unbounded, -unbounded, 0.5f * top), make_vec3(unbounded, unbounded, unbounded));
    DOCTEST_CHECK_FALSE(high_leaves.empty());
    DOCTEST_CHECK(high_leaves.size() < leaves.size());
    for (uint leaf: high_leaves) {
        DOCTEST_CHECK(articulation.links[leaf].position.z >= 0.5f * top);
    }
    DOCTEST_CHECK(std::is_sorted(high_leaves.begin(), high_leaves.end()));

    // The internodes carrying the highest leaf: each the ancestor of the one before, none at the root, the first the one bearing the leaf
    const uint highest = *std::max_element(leaves.begin(), leaves.end(), [&](uint a, uint b) { return articulation.links[a].position.z < articulation.links[b].position.z; });
    const uint bearing = findBearingInternode(articulation, highest);
    DOCTEST_CHECK(articulation.links[bearing].type == ARTICULATION_LINK_INTERNODE);
    const std::vector<uint> supporting = findSupportingInternodes(articulation, highest, 3);
    DOCTEST_REQUIRE(supporting.size() == 3);
    DOCTEST_CHECK(supporting.front() == bearing);
    auto isAncestor = [&](uint ancestor, uint link) {
        for (int parent = articulation.links[link].parent_index; parent >= 0; parent = articulation.links[parent].parent_index) {
            if (static_cast<uint>(parent) == ancestor) {
                return true;
            }
        }
        return false;
    };
    for (size_t k = 0; k < supporting.size(); k++) {
        DOCTEST_CHECK(articulation.links[supporting[k]].type == ARTICULATION_LINK_INTERNODE);
        DOCTEST_CHECK(articulation.links[supporting[k]].parent_index >= 0);
        if (k > 0) {
            DOCTEST_CHECK(isAncestor(supporting[k], supporting[k - 1]));
        }
    }
    // Asking for more than there are returns all of them but the root
    const std::vector<uint> all_supporting = findSupportingInternodes(articulation, highest, 100000);
    for (uint link: all_supporting) {
        DOCTEST_CHECK(articulation.links[link].parent_index >= 0);
    }
    DOCTEST_CHECK(all_supporting.size() >= supporting.size());

    // The root has nothing below it
    uint root = 0;
    while (articulation.links[root].parent_index >= 0) {
        root++;
    }
    DOCTEST_CHECK(findSupportingInternodes(articulation, root, 5).empty());
    DOCTEST_CHECK_THROWS(static_cast<void>(findBearingInternode(articulation, root)));
    DOCTEST_CHECK_THROWS(static_cast<void>(findBearingInternode(articulation, static_cast<uint>(articulation.links.size()))));
    DOCTEST_CHECK_THROWS(static_cast<void>(findSupportingInternodes(articulation, static_cast<uint>(articulation.links.size()), 1)));
}

DOCTEST_TEST_CASE("PlantPhysics - misuse is reported") {
    Context context;
    PlantArchitecture plantarchitecture(&context);
    uint plantID = buildTestPlant(context, plantarchitecture);

    PhysicsEngine engine(&context);
    engine.disableMessages();

    PlantPhysicsParameters plant_params;
    plant_params.leaf_collision_shape = "mesh";
    DOCTEST_CHECK_THROWS(addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, USDExportParameters(), plant_params));
    DOCTEST_CHECK_THROWS(addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID + 100));

    addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID);
    DOCTEST_CHECK_THROWS(addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID));
    engine.initialize();
    DOCTEST_CHECK_THROWS(addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID));
}

#endif // HELIOS_PHYSICS_ENGINE_PLANTS
