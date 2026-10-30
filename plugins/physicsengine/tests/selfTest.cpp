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
