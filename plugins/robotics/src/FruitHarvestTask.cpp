/** \file "FruitHarvestTask.cpp" A reinforcement learning task: a mobile arm with a vacuum cup picks fruit from trees, in many parallel simulations.

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

#include "FruitHarvestTask.h"

#include "ParallelRobotSimulations.h"
#include "ProtectiveStop.h"
#include "ToolVelocityAction.h"

#ifdef HELIOS_ROBOTICS_CAMERA
#include "RobotCamera.h"
#endif
#ifdef HELIOS_ROBOTICS_RADIATION
#include "LeafOptics.h"
#include "RadiationModel.h"
#endif

#include <mujoco/mujoco.h>

using namespace helios;

namespace {

    // Position servos on the robot's joints. The force limits are the effort limits of the joints in harvest_arm.urdf.
    struct JointServo {
        const char *joint;
        float stiffness; // N/m or N m/rad
        float damping; // N s/m or N m s/rad
        float force_limit; // N or N m
        float armature; // kg or kg m^2
    };
    const JointServo joint_servos[] = {{"drive", 20000.f, 5000.f, 2000.f, 20.f}, {"yaw", 2000.f, 400.f, 300.f, 2.f}, {"shoulder", 2000.f, 400.f, 300.f, 2.f},
                                       {"elbow", 1000.f, 200.f, 200.f, 1.f}, {"wrist", 200.f, 30.f, 60.f, 0.2f}, {"roll", 30.f, 5.f, 30.f, 0.05f}};
    const size_t robot_joint_count = 6;

    // The center of the cup's mouth, in the tool's frame of harvest_arm.urdf; the cup's axis is the tool's x axis
    const vec3 cup_face = make_vec3(0.32f, 0, 0);
    // The turret turns about a vertical axis this far ahead of the platform along x
    const vec3 turret_offset = make_vec3(0.15f, 0, 0);

    // Smaller motion of a body does not move its geometry for the camera. It is about 0.1 pixel at 128x128 for something 30 cm from the camera.
    const float context_update_tolerance = 0.0005f; // m

    // A start near the fruit, for training only (setNearStartFraction()): the cup this far short of where the fruit would be seated, along the line from the
    // turret, and up to near_start_off_axis off that line, sideways and up or down
    const float near_start_min_standoff = 0.02f; // m
    const float near_start_max_standoff = 0.12f; // m
    const float near_start_off_axis = 0.03f; // m
    const int near_start_attempts = 5; // placements tried, each kept only if the arm reaches it without touching anything; otherwise the start pose

    // Labels for measuring what the camera sees. A fruit is labeled by its object, offset past the others.
    const uint organ_other = 0;
    const uint organ_leaf = 1;
    const uint organ_wood = 2;
    const uint organ_fruit_base = 3;

    const float wrist_camera_field_of_view = 70.f;
    const float movie_camera_field_of_view = 40.f;
    // Toward the sun, for both the visualizer's light and the radiation cameras' sun
    const vec3 movie_light_direction = make_vec3(-0.35f, -0.45f, 0.82f);

    // Radiation cameras for movies: rays per primitive for the radiation reaching each primitive, which smoothing interpolates across it (the sun's is traced
    // again for each pixel sample), and bounces of scattered light
    const size_t movie_direct_rays = 10;
    const size_t movie_diffuse_rays = 100;
    const uint movie_scattering_depth = 2;
    const char *const movie_ground_spectrum = "soil_reflectivity_0000";
    // The ground for radiation cameras: an adaptive tile of the visualizer's ground size, finest in the alley between the robot and the row
    const float movie_ground_size = 20.f;
    const float movie_ground_refinement_target_x = -0.8f;
    const float movie_ground_subpatch_size_min = 0.1f;
    const float movie_ground_subpatch_size_max = 2.f;
    // Rows repeated beside the drawn one by a periodic boundary are this far apart, a usual spacing for tall spindle; the periodic cell across the rows is
    // centred off the row, into the alley the robot stands in, so that the robot (whose platform reaches 1.83 m from the row) lies in one cell
    const float movie_row_spacing = 3.5f;
    const float movie_row_cell_center_x = -0.25f;

    // The rest of the row, for movies: the wall's default tree spacing, and a tall-spindle trellis (Hoying, Robinson & Fargione, "The Tall Spindle
    // Planting System"): high-tensile wires along the row, a conduit stake tied to each leader, and a wooden line post every ten trees.
    const float row_tree_spacing = 0.9f;
    const std::vector<float> trellis_wire_heights = {0.75f, 1.5f, 2.25f, 3.f};
    const float trellis_wire_radius = 0.002f; // 4 mm across, a little thicker than the 2.5 mm (12.5 gauge) wire, so that it shows at movie resolution
    const float stake_height = 3.f;
    const float stake_radius = 0.01f;
    const vec3 stake_offset = make_vec3(0.03f, 0, 0); // behind the leader, as the robot sees it
    const float post_height = 3.3f;
    const float post_radius = 0.06f;
    const int trees_per_post = 10;

    // Fruit at heights the arm can reach
    std::vector<uint> findReachableFruit(const PlantArticulation &articulation, const FruitHarvestTaskParameters &parameters) {
        const float unbounded = std::numeric_limits<float>::infinity();
        return findPlantLinksInBox(articulation, ARTICULATION_LINK_FRUIT, make_vec3(-unbounded, -unbounded, parameters.min_fruit_height), make_vec3(unbounded, unbounded, parameters.max_fruit_height));
    }

    void validateParameters(const FruitHarvestTaskParameters &parameters) {
        const std::string where = "ERROR (FruitHarvestTask): ";
        if (parameters.plant_model.empty()) {
            helios_runtime_error(where + "No plant model was given.");
        }
        if (!(parameters.plant_age > 0.f)) {
            helios_runtime_error(where + "The plant age must be positive, but was " + std::to_string(parameters.plant_age) + " days.");
        }
        if (!(parameters.min_fruit_height < parameters.max_fruit_height)) {
            helios_runtime_error(where + "The lowest height of a fruit that can be picked (" + std::to_string(parameters.min_fruit_height) + " m) must be below the highest (" + std::to_string(parameters.max_fruit_height) + " m).");
        }
        if (parameters.candidates_per_tree == 0) {
            helios_runtime_error(where + "Each tree needs at least one candidate fruit.");
        }
        if (!(parameters.wood_elastic_modulus > 0.f && parameters.wood_density > 0.f && parameters.fruit_mass > 0.f)) {
            helios_runtime_error(where + "The wood's elastic modulus and density and the fruit's mass must be positive.");
        }
        if (parameters.start_pose.size() != robot_joint_count) {
            helios_runtime_error(where + "The start pose must give the robot's " + std::to_string(robot_joint_count) + " joints (drive, yaw, shoulder, elbow, wrist, roll), but gives " + std::to_string(parameters.start_pose.size()) + ".");
        }
        if (!(parameters.protective_stop_force > 0.f)) {
            helios_runtime_error(where + "The protective stop force must be positive, but was " + std::to_string(parameters.protective_stop_force) + " N.");
        }
        if (!(parameters.min_detach_force > 0.f && parameters.min_detach_force <= parameters.max_detach_force)) {
            helios_runtime_error(where + "The forces to detach a fruit must be positive, with the smallest (" + std::to_string(parameters.min_detach_force) + " N) no larger than the largest (" + std::to_string(parameters.max_detach_force) + " N).");
        }
        if (parameters.physics_steps_per_control == 0 || parameters.max_control_steps == 0) {
            helios_runtime_error(where + "The physics steps per control step and the control steps per episode must be positive.");
        }
        if (!(parameters.max_tool_speed > 0.f && parameters.max_target_lead > 0.f)) {
            helios_runtime_error(where + "The tool's speed limit and the lead of its target must be positive.");
        }
        if (!(parameters.start_offset >= 0.f)) {
            helios_runtime_error(where + "The start offset along the row must not be negative, but was " + std::to_string(parameters.start_offset) + " m.");
        }
        if (!(parameters.memory_arena_megabytes > 0.f)) {
            helios_runtime_error(where + "The memory of each simulation must be positive, but was " + std::to_string(parameters.memory_arena_megabytes) + " MB.");
        }
    }

    // A camera on the wrist, behind and above the cup, looking along the tool past the cup
    void wristCameraPose(const Robot &robot, uint instance, vec3 &eye, vec3 &look_at) {
        const vec3 cup = robot.getToolPosition(instance);
        const vec3 axis = robot.getToolAxis(instance);
        eye = cup - 0.3f * axis + make_vec3(0, 0, 0.08f);
        look_at = cup + 0.3f * axis;
    }

    // The action of a control step after a success, for showing: the cup backs straight away from the tree, and grips no more
    const float follow_through_action[FruitHarvestTask::action_size] = {-1.f, 0.f, 0.f, 0.f};

} // namespace

struct FruitHarvestTask::Tree {

    struct Candidate {
        uint link;
        uint bearing_wood_link; // the nearest internode toward the trunk
        uint objID;
        vec3 settled_position;
        vec3 settled_wood_position;
    };

    const FruitHarvestTaskParameters &parameters;
    Context context;
    PlantArchitecture plantarchitecture;
    PhysicsEngine engine;
    std::unique_ptr<Robot> robot;
    uint plant = 0;
    int tool_body_ID = -1;
    std::vector<Candidate> candidates;
    std::unique_ptr<NearestPlantOrgan> nearest_wood; // where the wood stands at rest, which stands in for what perception would report
    // What the robot uses: contact by what it touches, the protective stop on it, the cup, and the cup's velocity control
    std::unique_ptr<ContactSensor> contact;
    std::unique_ptr<ProtectiveStop> protective_stop;
    std::unique_ptr<VacuumCup> cup;
    std::unique_ptr<ToolVelocityAction> tool_action;
    std::unique_ptr<VacuumCupGripAction> grip_action;
    std::unique_ptr<ControlLoop> control_loop; // runs the cup and the protective stop at every physics step
    std::unordered_map<uint, uint> organ_of_primitive; // for measuring what the camera sees: which kind of plant organ each primitive belongs to, or which fruit

    // For showing a simulation from outside (movies): the links of the plant, parents first, with the MuJoCo body of each (-1 if it has none) and its
    // rest pose (x, y, z, qw, qx, qy, qz), which is also the rest pose of its body
    std::vector<std::string> link_names;
    std::vector<int> link_parents;
    std::vector<int> link_bodies;
    std::vector<std::array<mjtNum, 7>> link_rest_poses;
    // For movies drawn by radiation cameras: the reflectivity spectrum of each object of the robot and the scenery, which the visualizer draws in its color,
    // and the ground
    std::vector<std::pair<uint, std::string>> painted_objects;
    uint ground_UUID = 0;
    // The objects of the robot, which a periodic row of trees leaves out of its repeats, and the half-length of the row of drawn trees and trellis along y
    // (negative before addMovieScenery())
    std::vector<uint> robot_objects;
    float movie_row_half_length = -1.f;
#ifdef HELIOS_ROBOTICS_CAMERA
    std::unique_ptr<RobotCamera> camera;
    std::unique_ptr<RobotCamera> movie_camera;
#endif
#ifdef HELIOS_ROBOTICS_RADIATION
    std::unique_ptr<RadiationModel> radiation;
#endif

    [[nodiscard]] bool hasCamera() const {
#ifdef HELIOS_ROBOTICS_CAMERA
        if (camera || movie_camera) {
            return true;
        }
#endif
#ifdef HELIOS_ROBOTICS_RADIATION
        if (radiation) {
            return true;
        }
#endif
        return false;
    }

    uint paint(uint objID, const std::string &reflectivity_spectrum) {
        painted_objects.emplace_back(objID, reflectivity_spectrum);
        return objID;
    }

    uint paintRobot(uint objID, const std::string &reflectivity_spectrum) {
        robot_objects.push_back(objID);
        return paint(objID, reflectivity_spectrum);
    }

    Tree(const FruitHarvestTaskParameters &task_parameters, int seed, uint copy_count) : parameters(task_parameters), plantarchitecture(&context), engine(&context) {
        context.seedRandomGenerator(seed);
        plantarchitecture.disableMessages();
        engine.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary(parameters.plant_model);
        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), parameters.plant_age);
        if (!parameters.leaves) {
            plantarchitecture.removePlantLeaves(plantID);
        }

        const USDExportParameters plant_physics = FruitHarvestTask::plantArticulationParameters(parameters);
        const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID, plant_physics);

        nearest_wood = std::make_unique<NearestPlantOrgan>(articulation, ARTICULATION_LINK_INTERNODE);

        std::vector<uint> reachable;
        try {
            reachable = FruitHarvestTask::chooseCandidateFruit(articulation, parameters);
        } catch (const std::exception &) {
            helios_runtime_error("ERROR (FruitHarvestTask): The tree with seed " + std::to_string(seed) + " has no fruit within reach of the robot, between " + std::to_string(parameters.min_fruit_height) + " and " +
                                 std::to_string(parameters.max_fruit_height) + " m high. Choose another seed (see FruitHarvestTask::treeHasReachableFruit()).");
        }

        for (const PlantArticulationLink &link: articulation.links) {
            if (link.source_objID == 0 || !context.doesObjectExist(link.source_objID)) {
                continue;
            }
            const uint organ = link.type == ARTICULATION_LINK_FRUIT ? link.source_objID + organ_fruit_base : (link.type == ARTICULATION_LINK_LEAF ? organ_leaf : (link.type == ARTICULATION_LINK_INTERNODE ? organ_wood : organ_other));
            for (uint UUID: context.getObjectPrimitiveUUIDs(link.source_objID)) {
                organ_of_primitive[UUID] = organ;
            }
        }

        PlantPhysicsParameters plant_parameters;
        plant_parameters.rigid_internodes = true;
        plant_parameters.leaf_collisions = false;
        plant_parameters.rigid_leaves = true;
        plant_parameters.detachable_links = reachable;
        plant_parameters.flexible_links = FruitHarvestTask::chooseFlexibleInternodes(articulation, reachable, parameters);
        plant = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, plant_physics, plant_parameters);

        const vec3 robot_base = parameters.robot_base;
        robot = std::make_unique<Robot>(&engine, resolvePluginAsset("robotics", "assets/harvest_arm.urdf").string(), "robot", robot_base);
        for (const JointServo &servo: joint_servos) {
            robot->addJointServo(servo.joint, servo.stiffness, servo.damping, servo.force_limit, servo.armature);
        }
        robot->setTool("tool", cup_face, make_vec3(1, 0, 0));

        // What the camera can see of the robot, in its zero pose, in which the tool lies along +x from the wrist
        const vec3 wrist = robot_base + make_vec3(0.15f + 0.65f + 0.6f, 0, 0.75f);
        // The neck, and the cup: its bottom and its lip as in harvest_arm.urdf, and its wall between them, which does not collide
        const RGBcolor rubber = make_RGBcolor(0.15f, 0.15f, 0.15f);
        const vec3 mouth = wrist + cup_face;
        const float lip_radius = parameters.cup.lip_radius;
        engine.bindObjectToBody(paintRobot(context.addConeObject(16, wrist, wrist + make_vec3(0.28f, 0, 0), 0.03f, 0.018f, make_RGBcolor(0.9f, 0.4f, 0.08f)), "ColorReference_Calibrite_07"), "robot/tool");
        engine.bindObjectToBody(paintRobot(context.addSphereObject(16, wrist + make_vec3(0.29f, 0, 0), 0.012f, rubber), "ColorReference_Calibrite_24"), "robot/tool");
        engine.bindObjectToBody(paintRobot(context.addConeObject(24, wrist + make_vec3(0.28f, 0, 0), mouth, 0.016f, lip_radius, rubber), "ColorReference_Calibrite_24"), "robot/tool");
        std::vector<vec3> lip_nodes;
        for (int k = 0; k <= 32; k++) {
            const float angle = 2.f * float(M_PI) * float(k) / 32.f;
            lip_nodes.push_back(mouth + lip_radius * make_vec3(0, std::cos(angle), std::sin(angle)));
        }
        engine.bindObjectToBody(paintRobot(context.addTubeObject(8, lip_nodes, std::vector<float>(lip_nodes.size(), parameters.cup.lip_thickness), std::vector<RGBcolor>(lip_nodes.size(), rubber)), "ColorReference_Calibrite_24"), "robot/tool");
        if (parameters.show_arm) {
            // The rest of the arm, as the collision shapes of harvest_arm.urdf in its zero pose, for showing the robot from outside. The wrist camera
            // does not need it, and the observations do not depend on it.
            const RGBcolor arm_color = make_RGBcolor(0.95f, 0.6f, 0.12f);
            const RGBcolor joint_color = make_RGBcolor(0.25f, 0.25f, 0.28f);
            const vec3 shoulder = robot_base + make_vec3(0.15f, 0, 0.75f);
            const vec3 elbow = shoulder + make_vec3(0.65f, 0, 0);
            engine.bindObjectToBody(paintRobot(context.addBoxObject(robot_base + make_vec3(0, 0, 0.22f), make_vec3(0.55f, 0.8f, 0.2f), make_int3(1, 1, 1), joint_color), "ColorReference_Calibrite_23"), "robot/platform");
            engine.bindObjectToBody(paintRobot(context.addConeObject(24, robot_base + make_vec3(0.15f, 0, 0.25f), shoulder, 0.075f, 0.075f, arm_color), "ColorReference_Calibrite_12"), "robot/turret");
            engine.bindObjectToBody(paintRobot(context.addSphereObject(16, shoulder, 0.07f, joint_color), "ColorReference_Calibrite_23"), "robot/upper_arm");
            engine.bindObjectToBody(paintRobot(context.addConeObject(24, shoulder, elbow, 0.05f, 0.05f, arm_color), "ColorReference_Calibrite_12"), "robot/upper_arm");
            engine.bindObjectToBody(paintRobot(context.addSphereObject(16, elbow, 0.055f, joint_color), "ColorReference_Calibrite_23"), "robot/forearm");
            engine.bindObjectToBody(paintRobot(context.addConeObject(24, elbow, wrist, 0.038f, 0.038f, arm_color), "ColorReference_Calibrite_12"), "robot/forearm");
            engine.bindObjectToBody(paintRobot(context.addSphereObject(16, wrist, 0.042f, joint_color), "ColorReference_Calibrite_23"), "robot/wrist_link");
        }
        ground_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(20.f, 20.f), make_SphericalCoord(0, 0), make_RGBcolor(0.42f, 0.33f, 0.24f));

        PhysicsEngineParameters engine_parameters;
        engine_parameters.ground_collisions = true;
        // The tree's parts collide with each other, so that the arm cannot push a fruit into wood or another fruit. Those that the tree was grown with
        // overlapping stay as they are: pushing them apart would bend the wood against the fruit and load their stems before the arm touches them.
        engine_parameters.articulation_collisions = true;
        engine_parameters.keep_initial_link_overlaps = true;
        engine_parameters.memory_arena_megabytes = parameters.memory_arena_megabytes;
        engine.initialize(engine_parameters);
        robot->initialize();

        while (engine.getInstanceCount() < copy_count) {
            engine.createInstance();
        }
        robot->prepareInstances();

        const mjModel *model = engine.getMuJoCoModel();
        tool_body_ID = mj_name2id(model, mjOBJ_BODY, "robot/tool");

        // The contact sensor covers the whole robot. The fruit being picked is its "target", which the cup is meant to touch, so the stop leaves it out.
        contact = std::make_unique<ContactSensor>(&engine, mj_id2name(model, mjOBJ_BODY, model->body_rootid[tool_body_ID]));
        protective_stop = std::make_unique<ProtectiveStop>(contact.get(), parameters.protective_stop_force, std::vector<std::string>{"target"});
        cup = std::make_unique<VacuumCup>(robot.get(), "tool", cup_face, make_vec3(1, 0, 0), parameters.cup);
        tool_action = std::make_unique<ToolVelocityAction>(robot.get(), parameters.max_tool_speed, parameters.max_target_lead);
        // The cup is level and points away from the turret, which turns about a vertical axis ahead of the platform as it drives along the row
        tool_action->setToolAxisAwayFromPivot(turret_offset, "drive");
        tool_action->setLockedJoints({"roll"});
        grip_action = std::make_unique<VacuumCupGripAction>(cup.get());
        control_loop = std::make_unique<ControlLoop>(&engine, parameters.physics_steps_per_control);
        control_loop->addComponent(cup.get());
        control_loop->addComponent(protective_stop.get());
        control_loop->addComponent(tool_action.get());

        for (const PlantArticulationLink &link: articulation.links) {
            link_names.push_back(link.name);
            link_parents.push_back(link.parent_index);
            link_bodies.push_back(mj_name2id(model, mjOBJ_BODY, ("p" + std::to_string(plantID) + "_" + link.name).c_str()));
            link_rest_poses.push_back({link.position.x, link.position.y, link.position.z, link.qw, link.qx, link.qy, link.qz});
            mju_normalize4(link_rest_poses.back().data() + 3);
        }

        // Let the fruit and flexible wood settle, and start every episode from there
        robot->setJointPositions(parameters.start_pose, 0);
        engine.step(parameters.settling_physics_steps, 0);
        engine.setResetPose(0);

        for (uint link: reachable) {
            Candidate candidate;
            candidate.link = link;
            candidate.objID = articulation.links[link].source_objID;
            candidate.bearing_wood_link = findBearingInternode(articulation, link);
            candidate.settled_wood_position = engine.getLinkPosition(plant, candidate.bearing_wood_link, 0);
            // Fails here, rather than in an episode, if the cup cannot seal on this fruit
            cup->setTarget(plant, link, 0);
            candidate.settled_position = engine.getLinkPosition(plant, link, 0);
            candidates.push_back(candidate);
        }
    }

    // The world pose (x, y, z, qw, qx, qy, qz) of every link of the plant, NaN for a link with no body, and the robot's joint positions (m, degrees)
    void captureDisplayState(uint instance, std::vector<float> &link_poses, std::vector<float> &robot_joints) const {
        const mjData *data = engine.getMuJoCoData(instance);
        link_poses.assign(7 * link_names.size(), std::numeric_limits<float>::quiet_NaN());
        for (size_t i = 0; i < link_names.size(); i++) {
            const int body = link_bodies[i];
            if (body < 0) {
                continue;
            }
            for (int k = 0; k < 3; k++) {
                link_poses[7 * i + k] = float(data->xpos[3 * body + k]);
            }
            for (int k = 0; k < 4; k++) {
                link_poses[7 * i + 3 + k] = float(data->xquat[4 * body + k]);
            }
        }
        robot_joints = robot->getJointPositions(instance);
    }

    // Puts a simulation in a pose to be shown, which need not be one this simulation reached: it may come from another simulator. Every link of the plant takes
    // the given world pose, or, where that is NaN, follows its parent as it does at rest. The robot takes the given joint positions. This leaves the simulation
    // in the pose shown, with no velocity.
    void showPose(uint instance, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, const std::string &method) {
        const size_t link_count = link_names.size();
        if (link_poses.size() != 7 * link_count) {
            helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): " + std::to_string(link_poses.size()) + " link pose values were given, but the tree has " + std::to_string(link_count) + " links of 7 values each.");
        }
        if (robot_joints.size() != robot_joint_count) {
            helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): " + std::to_string(robot_joints.size()) + " robot joint positions were given, but the robot has " + std::to_string(robot_joint_count) + " joints.");
        }
        mjData *data = engine.getMuJoCoData(instance);
        robot->setJointPositions(robot_joints, instance);

        std::vector<std::array<mjtNum, 7>> world(link_count);
        for (size_t i = 0; i < link_count; i++) {
            const int parent = link_parents[i];
            const float *given = link_poses.data() + 7 * i;
            if (std::isfinite(given[0])) {
                for (int k = 0; k < 7; k++) {
                    world[i][k] = given[k];
                }
                mju_normalize4(world[i].data() + 3);
            } else if (parent >= 0) {
                if (size_t(parent) >= i) {
                    helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): Link " + link_names[i] + " is listed before its parent.");
                }
                // Carried by the parent: pose = parent * parent_rest^-1 * rest
                const std::array<mjtNum, 7> &rest = link_rest_poses[i];
                const std::array<mjtNum, 7> &parent_rest = link_rest_poses[parent];
                mjtNum parent_rest_conjugate[4], relative_quat[4], offset[3], relative_position[3], moved[3];
                mju_negQuat(parent_rest_conjugate, parent_rest.data() + 3);
                mju_mulQuat(relative_quat, parent_rest_conjugate, rest.data() + 3);
                mju_sub3(offset, rest.data(), parent_rest.data());
                mju_rotVecQuat(relative_position, offset, parent_rest_conjugate);
                mju_rotVecQuat(moved, relative_position, world[parent].data() + 3);
                mju_add3(world[i].data(), world[parent].data(), moved);
                mju_mulQuat(world[i].data() + 3, world[parent].data() + 3, relative_quat);
            } else {
                world[i] = link_rest_poses[i];
            }
            const int body = link_bodies[i];
            if (body >= 0) {
                mju_copy3(data->xpos + 3 * body, world[i].data());
                mju_copy4(data->xquat + 4 * body, world[i].data() + 3);
                mju_quat2Mat(data->xmat + 9 * body, world[i].data() + 3);
            }
        }
    }
};

struct FruitHarvestTask::Copy {
    Tree *tree = nullptr;
    uint instance = 0;
    std::mt19937 random;
    float near_start_fraction = 0.f; // share of episodes started with the cup near the fruit
    const Tree::Candidate *target = nullptr;
    float detach_force = 0.f;
    float previous_load = 0.f;
    float previous_distance = 0.f;
    uint control_step = 0;
    vec3 pull_direction;
    FruitHarvestEpisodeRecord current{};
    FruitHarvestEpisodeRecord last{};
    bool finished_an_episode = false;
    // How the simulation looked when its last episode ended, for showing it (see Tree::captureDisplayState())
    std::vector<float> last_link_poses;
    std::vector<float> last_robot_joints;
    // For showing: control steps a success goes on for, with the cup carrying the freed fruit away (see setFollowThrough())
    uint follow_through_steps = 0;
    uint follow_through_remaining = 0; // nonzero while a success goes on
    bool gripped_before_step = false; // whether the cup was gripping when the control step started
    FruitHarvestEpisodeRecord detached_record{}; // the record of an episode that goes on, as it was when the fruit detached
};

struct FruitHarvestTask::Simulations {
    explicit Simulations(uint thread_count) : parallel(thread_count) {
    }
    ParallelRobotSimulations parallel;
    std::vector<float> actions; // each copy's actions for the control step being run
};

FruitHarvestTask::FruitHarvestTask(const std::vector<int> &tree_seeds, uint copies_per_tree, uint thread_count, uint seed, const FruitHarvestTaskParameters &task_parameters) : parameters(task_parameters) {
    validateParameters(parameters);
    if (tree_seeds.empty()) {
        helios_runtime_error("ERROR (FruitHarvestTask): No tree seeds were given.");
    }
    if (copies_per_tree == 0) {
        helios_runtime_error("ERROR (FruitHarvestTask): Each tree needs at least one copy.");
    }
    simulations = std::make_unique<Simulations>(thread_count);
    for (size_t t = 0; t < tree_seeds.size(); t++) {
        trees.push_back(std::make_unique<Tree>(parameters, tree_seeds[t], copies_per_tree));
        Tree &tree = *trees.back();
        simulations->parallel.addVariant(&tree.engine, tree.control_loop.get(), {tree.tool_action.get(), tree.grip_action.get()});
        for (uint i = 0; i < copies_per_tree; i++) {
            auto copy = std::make_unique<Copy>();
            copy->tree = &tree;
            copy->instance = i;
            copy->random.seed(static_cast<unsigned>(seed + 1000 * t + i));
            copies.push_back(std::move(copy));
        }
    }
}

FruitHarvestTask::~FruitHarvestTask() {
    // The worker threads step the trees, so they stop first
    simulations.reset();
}

uint FruitHarvestTask::getCopyCount() const {
    return static_cast<uint>(copies.size());
}

uint FruitHarvestTask::getObservationSize() {
    return observation_size;
}

uint FruitHarvestTask::getActionSize() {
    return action_size;
}

uint FruitHarvestTask::getMaxControlSteps() const {
    return parameters.max_control_steps;
}

const FruitHarvestTaskParameters &FruitHarvestTask::getParameters() const {
    return parameters;
}

FruitHarvestTask::Copy &FruitHarvestTask::requireCopy(uint copy, const std::string &method) {
    if (copy >= copies.size()) {
        helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): Copy " + std::to_string(copy) + " does not exist; the task has " + std::to_string(copies.size()) + " copies.");
    }
    return *copies[copy];
}

const FruitHarvestTask::Copy &FruitHarvestTask::requireCopy(uint copy, const std::string &method) const {
    if (copy >= copies.size()) {
        helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): Copy " + std::to_string(copy) + " does not exist; the task has " + std::to_string(copies.size()) + " copies.");
    }
    return *copies[copy];
}

std::vector<uint> FruitHarvestTask::chooseCandidateFruit(const PlantArticulation &articulation, const FruitHarvestTaskParameters &task_parameters) {
    std::vector<uint> reachable = findReachableFruit(articulation, task_parameters);
    // Nearest the robot first, since those are the ones not hidden behind the wall
    std::sort(reachable.begin(), reachable.end(), [&](uint a, uint b) { return articulation.links[a].position.x < articulation.links[b].position.x; });
    if (reachable.size() > task_parameters.candidates_per_tree) {
        reachable.resize(task_parameters.candidates_per_tree);
    }
    if (reachable.empty()) {
        helios_runtime_error("ERROR (FruitHarvestTask::chooseCandidateFruit): The tree has no fruit within reach of the robot, between " + std::to_string(task_parameters.min_fruit_height) + " and " + std::to_string(task_parameters.max_fruit_height) + " m high.");
    }
    return reachable;
}

std::vector<uint> FruitHarvestTask::chooseFlexibleInternodes(const PlantArticulation &articulation, const std::vector<uint> &candidates, const FruitHarvestTaskParameters &task_parameters) {
    std::vector<uint> flexible;
    for (uint fruit: candidates) {
        const std::vector<uint> supporting = findSupportingInternodes(articulation, fruit, task_parameters.flexible_internodes_per_candidate);
        flexible.insert(flexible.end(), supporting.begin(), supporting.end());
    }
    return flexible;
}

std::vector<FruitHarvestJointServo> FruitHarvestTask::getJointServos() {
    std::vector<FruitHarvestJointServo> servos;
    for (const JointServo &servo: joint_servos) {
        servos.push_back({servo.joint, servo.stiffness, servo.damping, servo.force_limit, servo.armature});
    }
    return servos;
}

vec3 FruitHarvestTask::getCupFace() {
    return cup_face;
}

USDExportParameters FruitHarvestTask::plantArticulationParameters(const FruitHarvestTaskParameters &task_parameters) {
    USDExportParameters plant_physics;
    plant_physics.elastic_modulus = task_parameters.wood_elastic_modulus;
    plant_physics.wood_density = task_parameters.wood_density;
    plant_physics.fruit_mass = task_parameters.fruit_mass;
    return plant_physics;
}

bool FruitHarvestTask::treeHasReachableFruit(int tree_seed, const FruitHarvestTaskParameters &task_parameters) {
    validateParameters(task_parameters);
    Context context;
    context.seedRandomGenerator(tree_seed);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary(task_parameters.plant_model);
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), task_parameters.plant_age);
    // The fruit's positions do not depend on the physical properties
    USDExportParameters plant_physics;
    return !findReachableFruit(plantarchitecture.getPlantArticulation(plantID, plant_physics), task_parameters).empty();
}

void FruitHarvestTask::setNearStartFraction(float fraction) {
    if (!(fraction >= 0.f && fraction <= 1.f)) {
        helios_runtime_error("ERROR (FruitHarvestTask::setNearStartFraction): The fraction must be from 0 to 1, but was " + std::to_string(fraction) + ".");
    }
    for (const auto &copy: copies) {
        copy->near_start_fraction = fraction;
    }
}

void FruitHarvestTask::setFollowThrough(uint control_steps) {
    for (const auto &copy: copies) {
        copy->follow_through_steps = control_steps;
    }
}

FruitHarvestEpisodeRecord FruitHarvestTask::getLastEpisode(uint copy) const {
    const Copy &which = requireCopy(copy, "getLastEpisode");
    if (!which.finished_an_episode) {
        helios_runtime_error("ERROR (FruitHarvestTask::getLastEpisode): Copy " + std::to_string(copy) + " has not finished an episode yet.");
    }
    return which.last;
}

vec3 FruitHarvestTask::getTargetFruitPosition(uint copy) const {
    const Copy &which = requireCopy(copy, "getTargetFruitPosition");
    if (which.target == nullptr) {
        helios_runtime_error("ERROR (FruitHarvestTask::getTargetFruitPosition): Copy " + std::to_string(copy) + " has no fruit to pick until the task is reset.");
    }
    return which.target->settled_position;
}

void FruitHarvestTask::writeObservation(const Copy &copy, float *observation) const {
    const Tree &tree = *copy.tree;
    const Robot &robot = *tree.robot;
    const mjModel *model = tree.engine.getMuJoCoModel();
    const mjData *data = tree.engine.getMuJoCoData(copy.instance);
    const VacuumCup &cup = *tree.cup;
    const vec3 tool = robot.getToolPosition(copy.instance);
    const vec3 fruit = cup.getTargetPosition(copy.instance);
    const vec3 relative = fruit - tool;
    const std::vector<float> joints = robot.getJointPositions(copy.instance);
    mjtNum velocity[6];
    mj_objectVelocity(model, data, mjOBJ_BODY, tree.tool_body_ID, velocity, 0);
    // The contact force of the robot on the tree, other than on the fruit it is picking, which is what the protective stop acts on
    const float contact = tree.protective_stop->measureForce(copy.instance);
    const vec3 wood = tree.nearest_wood->find(tool).axis_offset;

    int i = 0;
    observation[i++] = relative.x;
    observation[i++] = relative.y;
    observation[i++] = relative.z;
    observation[i++] = cup.getSealGap(copy.instance);
    observation[i++] = tool.x - parameters.robot_base.x;
    observation[i++] = tool.y - joints[0];
    observation[i++] = tool.z;
    observation[i++] = float(velocity[3]);
    observation[i++] = float(velocity[4]);
    observation[i++] = float(velocity[5]);
    observation[i++] = joints[0];
    for (size_t j = 1; j < robot_joint_count; j++) {
        observation[i++] = deg2rad(joints[j]);
    }
    observation[i++] = cup.isGripped(copy.instance) ? 1.f : 0.f;
    observation[i++] = cup.getSealLoad(copy.instance); // how near the seal is to breaking
    observation[i++] = contact / 10.f;
    observation[i++] = float(copy.control_step) / float(parameters.max_control_steps);
    observation[i++] = wood.x;
    observation[i++] = wood.y;
    observation[i++] = wood.z;
    const vec3 off_axis = cup.getOffAxis(copy.instance);
    observation[i++] = off_axis.x;
    observation[i++] = off_axis.y;
    observation[i++] = off_axis.z;
}

// Place the arm with the cup a few centimeters short of the fruit and somewhat out of line with it, pointing away from the turret, so that training practices
// lining the cup up and sealing it. Returns false, leaving the arm as it was, when no placement tried is reachable without touching anything.
bool FruitHarvestTask::placeCupNearFruit(Copy &copy, const std::vector<float> &start_pose) {
    Tree &tree = *copy.tree;
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    const vec3 fruit = tree.cup->getTargetPosition(copy.instance);
    const vec3 turret = parameters.robot_base + turret_offset;
    for (int attempt = 0; attempt < near_start_attempts; attempt++) {
        vec3 line = fruit - make_vec3(turret.x, start_pose[0], fruit.z);
        line.normalize();
        const vec3 sideways = make_vec3(-line.y, line.x, 0.f);
        const float standoff = tree.cup->getSeatDistance(copy.instance) + near_start_min_standoff + unit(copy.random) * (near_start_max_standoff - near_start_min_standoff);
        const vec3 target = fruit - standoff * line + near_start_off_axis * ((2.f * unit(copy.random) - 1.f) * sideways + (2.f * unit(copy.random) - 1.f) * make_vec3(0, 0, 1));
        vec3 pointing = target - make_vec3(turret.x, start_pose[0], target.z);
        pointing.normalize();
        const RobotInverseKinematicsResult pose = tree.robot->solveInverseKinematics(target, pointing, {"roll"}, copy.instance);
        if (pose.converged) {
            tree.robot->setJointPositions(pose.joint_positions, copy.instance);
            // Whether any part of the robot touches anything other than itself and the ground
            if ((tree.robot->getToolPosition(copy.instance) - target).magnitude() < 0.01f && !tree.contact->isPenetrating({"ground"}, copy.instance)) {
                return true;
            }
            tree.robot->setJointPositions(start_pose, copy.instance);
        }
    }
    return false;
}

void FruitHarvestTask::resetCopy(Copy &copy) {
    Tree &tree = *copy.tree;
    PhysicsEngine &engine = tree.engine;
    // Back to the settled tree, with the cup, the stop, and the cup's velocity control reset
    tree.control_loop->resetInstance(copy.instance);

    std::uniform_int_distribution<size_t> pick_fruit(0, tree.candidates.size() - 1);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    copy.target = &tree.candidates[pick_fruit(copy.random)];
    copy.detach_force = parameters.min_detach_force + unit(copy.random) * (parameters.max_detach_force - parameters.min_detach_force);
    // The stem fails when the load on it passes the force for this fruit
    engine.clearDetachForces(copy.instance);
    engine.setDetachForce(tree.plant, copy.target->link, copy.detach_force, copy.instance);
    tree.cup->setTarget(tree.plant, copy.target->link, copy.instance);
    tree.contact->setTargetLink(tree.plant, copy.target->link, copy.instance);

    std::vector<float> start_pose = parameters.start_pose;
    start_pose[0] = copy.target->settled_position.y + (2.f * unit(copy.random) - 1.f) * parameters.start_offset;
    tree.robot->setJointPositions(start_pose, copy.instance);
    if (copy.near_start_fraction > 0.f && unit(copy.random) < copy.near_start_fraction) {
        placeCupNearFruit(copy, start_pose);
    }

    copy.follow_through_remaining = 0;
    copy.previous_load = 0.f;
    copy.control_step = 0;
    copy.previous_distance = tree.cup->getSealGap(copy.instance);
    copy.pull_direction = make_vec3(0, 0, 0);
    copy.current = FruitHarvestEpisodeRecord{};
    copy.current.detach_force = copy.detach_force;
    copy.current.steps_to_grip = -1;
}

void FruitHarvestTask::reset(std::vector<float> &observations) {
    observations.resize(copies.size() * observation_size);
    simulations->parallel.forEachCopy([&](uint i, uint, uint) {
        resetCopy(*copies[i]);
        writeObservation(*copies[i], observations.data() + size_t(i) * observation_size);
    });
}

// After a control step following a success, with the cup backing straight away from the tree whatever the action, carrying the fruit for as long as the grip
// holds. The episode is a success either way, and ends when the steps run out, with the record it had when the fruit detached.
void FruitHarvestTask::finishFollowThroughStep(Copy &copy, float *observation, float &reward, int &outcome, float *terminal_observation) {
    reward = 0.f;
    outcome = FRUIT_HARVEST_RUNNING;
    copy.follow_through_remaining--;
    if (copy.follow_through_remaining == 0) {
        outcome = FRUIT_HARVEST_SUCCESS;
        copy.last = copy.detached_record;
        copy.finished_an_episode = true;
        copy.tree->captureDisplayState(copy.instance, copy.last_link_poses, copy.last_robot_joints);
        writeObservation(copy, terminal_observation);
        resetCopy(copy);
    }
    writeObservation(copy, observation);
}

// After a control step, which moved the cup's target and sealed it if asked and it could (the actions), then simulated until the control step ended, or the
// fruit detached, the seal broke, or the protective stop tripped (the control loop)
void FruitHarvestTask::finishStep(Copy &copy, float *observation, float &reward, int &outcome, float *terminal_observation) {
    if (copy.follow_through_remaining > 0) {
        finishFollowThroughStep(copy, observation, reward, outcome, terminal_observation);
        return;
    }
    Tree &tree = *copy.tree;
    PhysicsEngine &engine = tree.engine;
    VacuumCup &cup = *tree.cup;

    // A seal that broke in the step that ended it still held the fruit until then
    const bool holding = cup.isGripped(copy.instance) || cup.isSealBroken(copy.instance);
    const bool gripped_now = !copy.gripped_before_step && holding;
    if (gripped_now) {
        copy.current.steps_to_grip = static_cast<int>(copy.control_step);
    }
    // Keep the direction the cup last pulled the fruit in, for the episode's record
    const vec3 on_fruit = cup.getForceOnTarget(copy.instance);
    if (on_fruit.magnitude() > 1e-6f) {
        copy.pull_direction = on_fruit / on_fruit.magnitude();
    }
    const float load = engine.getDetachableLinkLoad(tree.plant, copy.target->link, copy.instance);
    const float peak_contact = tree.protective_stop->getPeakForce(copy.instance);
    outcome = FRUIT_HARVEST_RUNNING;
    if (engine.isLinkDetached(tree.plant, copy.target->link, copy.instance)) {
        outcome = holding ? FRUIT_HARVEST_SUCCESS : FRUIT_HARVEST_KNOCKED_OFF;
    } else if (cup.isSealBroken(copy.instance)) {
        outcome = FRUIT_HARVEST_SLIPPED;
    } else if (tree.protective_stop->isStopped(copy.instance)) {
        outcome = FRUIT_HARVEST_COLLISION;
    }
    copy.control_step++;
    if (outcome == FRUIT_HARVEST_RUNNING && copy.control_step >= parameters.max_control_steps) {
        outcome = FRUIT_HARVEST_TIMEOUT;
    }

    // Reward: approach the fruit, grip it, load the stem, and detach it, without pushing on the tree
    const float distance = cup.getSealGap(copy.instance);
    reward = -0.005f;
    if (!holding) {
        reward += 5.f * (copy.previous_distance - distance);
    }
    if (gripped_now) {
        reward += 1.f;
    }
    if (holding) {
        reward += 3.f * (load - copy.previous_load) / parameters.max_detach_force;
    }
    reward -= 0.02f * std::min(peak_contact, parameters.protective_stop_force);
    if (outcome == FRUIT_HARVEST_SUCCESS) {
        reward += 10.f;
    } else if (outcome == FRUIT_HARVEST_SLIPPED || outcome == FRUIT_HARVEST_KNOCKED_OFF || outcome == FRUIT_HARVEST_COLLISION) {
        reward -= 2.f;
    }
    copy.previous_distance = distance;
    copy.previous_load = load;

    mjtNum tool_velocity[6];
    mj_objectVelocity(engine.getMuJoCoModel(), engine.getMuJoCoData(copy.instance), mjOBJ_BODY, tree.tool_body_ID, tool_velocity, 0);
    copy.current.peak_contact_force = std::max(copy.current.peak_contact_force, peak_contact);
    copy.current.peak_tool_speed = std::max(copy.current.peak_tool_speed, float(mju_norm3(tool_velocity + 3)));
    copy.current.peak_wood_motion = std::max(copy.current.peak_wood_motion, (engine.getLinkPosition(tree.plant, copy.target->bearing_wood_link, copy.instance) - copy.target->settled_wood_position).magnitude());
    copy.current.steps = static_cast<int>(copy.control_step);
    copy.current.final_stem_load = load;
    copy.current.final_grip_force = cup.getPullForce(copy.instance);
    copy.current.pull_direction = copy.pull_direction;

    if (outcome == FRUIT_HARVEST_SUCCESS && copy.follow_through_steps > 0) {
        // The episode goes on for showing; its reward and record are those of the success
        copy.detached_record = copy.current;
        copy.follow_through_remaining = copy.follow_through_steps;
        outcome = FRUIT_HARVEST_RUNNING;
    }
    if (outcome != FRUIT_HARVEST_RUNNING) {
        copy.last = copy.current;
        copy.finished_an_episode = true;
        tree.captureDisplayState(copy.instance, copy.last_link_poses, copy.last_robot_joints);
        writeObservation(copy, terminal_observation);
        resetCopy(copy);
    }
    writeObservation(copy, observation);
}

void FruitHarvestTask::step(const std::vector<float> &actions, std::vector<float> &observations, std::vector<float> &rewards, std::vector<int> &outcomes, std::vector<float> &terminal_observations) {
    const size_t count = copies.size();
    if (actions.size() != count * action_size) {
        helios_runtime_error("ERROR (FruitHarvestTask::step): " + std::to_string(actions.size()) + " action values were given, but the task has " + std::to_string(count) + " copies of " + std::to_string(action_size) + " actions each.");
    }
    for (const auto &copy: copies) {
        if (copy->target == nullptr) {
            helios_runtime_error("ERROR (FruitHarvestTask::step): Call reset() before the first step.");
        }
    }
    observations.resize(count * observation_size);
    rewards.resize(count);
    outcomes.resize(count);
    terminal_observations.resize(count * observation_size);
    simulations->actions.resize(count * action_size);
    for (size_t i = 0; i < count; i++) {
        Copy &copy = *copies[i];
        const float *action = copy.follow_through_remaining > 0 ? follow_through_action : actions.data() + i * action_size;
        std::copy(action, action + action_size, simulations->actions.begin() + i * action_size);
        copy.gripped_before_step = copy.tree->cup->isGripped(copy.instance);
    }
    simulations->parallel.step(simulations->actions.data());
    simulations->parallel.forEachCopy([&](uint i, uint, uint) {
        finishStep(*copies[i], observations.data() + size_t(i) * observation_size, rewards[i], outcomes[i], terminal_observations.data() + size_t(i) * observation_size);
    });
}

// ---- Showing ---- //

uint FruitHarvestTask::getPlantLinkCount(uint copy) const {
    return static_cast<uint>(requireCopy(copy, "getPlantLinkCount").tree->link_names.size());
}

std::vector<std::string> FruitHarvestTask::getPlantLinkNames(uint copy) const {
    return requireCopy(copy, "getPlantLinkNames").tree->link_names;
}

std::vector<float> FruitHarvestTask::getRestLinkPoses(uint copy) const {
    const Tree &tree = *requireCopy(copy, "getRestLinkPoses").tree;
    std::vector<float> link_poses(7 * tree.link_rest_poses.size());
    for (size_t i = 0; i < tree.link_rest_poses.size(); i++) {
        for (int k = 0; k < 7; k++) {
            link_poses[7 * i + k] = float(tree.link_rest_poses[i][k]);
        }
    }
    return link_poses;
}

void FruitHarvestTask::getDisplayState(uint copy, bool last_episode, std::vector<float> &link_poses, std::vector<float> &robot_joints) const {
    const Copy &which = requireCopy(copy, "getDisplayState");
    if (last_episode) {
        if (!which.finished_an_episode) {
            helios_runtime_error("ERROR (FruitHarvestTask::getDisplayState): Copy " + std::to_string(copy) + " has not finished an episode yet.");
        }
        link_poses = which.last_link_poses;
        robot_joints = which.last_robot_joints;
    } else {
        which.tree->captureDisplayState(which.instance, link_poses, robot_joints);
    }
}

void FruitHarvestTask::addMovieScenery(uint neighbors_each_side, bool trellis) {
    for (const std::unique_ptr<Tree> &tree: trees) {
        if (tree->hasCamera()) {
            helios_runtime_error("ERROR (FruitHarvestTask::addMovieScenery): Add the scenery before enabling a camera, which takes its geometry from the Context when it is created.");
        }
        // None of this is bound to a body, so the physics engine leaves it out: it is drawn, and nothing collides with it
        for (uint k = 1; k <= neighbors_each_side; k++) {
            for (const float side: {-1.f, 1.f}) {
                const uint neighborID = tree->plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, side * float(k) * row_tree_spacing, 0), parameters.plant_age);
                if (!parameters.leaves) {
                    tree->plantarchitecture.removePlantLeaves(neighborID);
                }
            }
        }
        const int neighbors = static_cast<int>(neighbors_each_side);
        const float row_end = (float(neighbors) + 0.5f) * row_tree_spacing;
        tree->movie_row_half_length = row_end;
        if (!trellis) {
            continue;
        }
        const RGBcolor steel = make_RGBcolor(0.62f, 0.63f, 0.65f);
        const RGBcolor wood = make_RGBcolor(0.42f, 0.33f, 0.24f);
        for (const float height: trellis_wire_heights) {
            tree->paint(tree->context.addConeObject(6, make_vec3(0, -row_end, height), make_vec3(0, row_end, height), trellis_wire_radius, trellis_wire_radius, steel), "ColorReference_Calibrite_21");
        }
        for (int k = -neighbors; k <= neighbors; k++) {
            const vec3 base = make_vec3(0, float(k) * row_tree_spacing, 0) + stake_offset;
            tree->paint(tree->context.addConeObject(12, base, base + make_vec3(0, 0, stake_height), stake_radius, stake_radius, steel), "ColorReference_Calibrite_21");
        }
        // Line posts stand between two trees; one between the second and third trees up the row (+y) from the simulated one, where a camera in the alley sees it
        for (int k = -neighbors; k < neighbors; k++) {
            if ((k - 2) % trees_per_post == 0) {
                const vec3 base = make_vec3(0, (float(k) + 0.5f) * row_tree_spacing, 0);
                tree->paint(tree->context.addConeObject(16, base, base + make_vec3(0, 0, post_height), post_radius, post_radius, wood), "ColorReference_Calibrite_01");
            }
        }
    }
}

// ---- Cameras drawn by the visualizer ---- //

#ifdef HELIOS_ROBOTICS_CAMERA

namespace {
    // The part of rendering that needs no OpenGL, so that trees can do it in parallel
    void prepareWristFrame(RobotCamera &camera, const Robot &robot, uint instance) {
        vec3 eye, look_at;
        wristCameraPose(robot, instance, eye, look_at);
        camera.setPose(eye, look_at);
        camera.prepare(instance);
    }
} // namespace

void FruitHarvestTask::enableWristCamera(uint width, uint height) {
    if (width == 0 || height == 0) {
        helios_runtime_error("ERROR (FruitHarvestTask::enableWristCamera): The image size must be positive.");
    }
    for (const std::unique_ptr<Tree> &tree: trees) {
        if (tree->hasCamera()) {
            helios_runtime_error("ERROR (FruitHarvestTask::enableWristCamera): A camera is already enabled. Each camera clears the Context's geometry flags after each frame, so another would miss motion it drew.");
        }
    }
    for (const std::unique_ptr<Tree> &tree: trees) {
        RobotCameraParameters wrist;
        wrist.field_of_view = wrist_camera_field_of_view;
        wrist.antialiasing_samples = 1;
        wrist.background = make_RGBcolor(0.6f, 0.75f, 0.95f);
        wrist.light_direction = make_vec3(0.3f, -0.5f, 0.8f);
        wrist.shadows = false;
        tree->camera = std::make_unique<RobotCamera>(&tree->engine, &tree->context, width, height, wrist);
        tree->engine.setContextUpdateTolerance(context_update_tolerance);
    }
}

void FruitHarvestTask::renderWristCameras(std::vector<unsigned char> &images) {
    for (const std::unique_ptr<Tree> &tree: trees) {
        if (!tree->camera) {
            helios_runtime_error("ERROR (FruitHarvestTask::renderWristCameras): Call enableWristCamera() first.");
        }
    }
    const size_t image_bytes = size_t(trees.front()->camera->getWidth()) * size_t(trees.front()->camera->getHeight()) * 3;
    images.resize(copies.size() * image_bytes);
    // Copies of a tree share its camera and so its framebuffer, so one copy of each tree is rendered at a time. The copies of a tree are next to each other.
    const size_t tree_count = trees.size();
    const size_t copies_per_tree = copies.size() / tree_count;
    for (size_t instance = 0; instance < copies_per_tree; instance++) {
        simulations->parallel.forEachIndex(tree_count, [&](size_t t) { prepareWristFrame(*trees[t]->camera, *trees[t]->robot, static_cast<uint>(instance)); });
        // Reading the images back only after all are drawn lets the GPU work through them while the next is being drawn, rather than waiting for each one
        for (size_t t = 0; t < tree_count; t++) {
            trees[t]->camera->draw();
        }
        for (size_t t = 0; t < tree_count; t++) {
            trees[t]->camera->getColorImage(images.data() + (t * copies_per_tree + instance) * image_bytes);
        }
    }
}

void FruitHarvestTask::saveWristCameraImage(uint copy, const std::string &filename) {
    Copy &which = requireCopy(copy, "saveWristCameraImage");
    Tree &tree = *which.tree;
    if (!tree.camera) {
        helios_runtime_error("ERROR (FruitHarvestTask::saveWristCameraImage): Call enableWristCamera() first.");
    }
    prepareWristFrame(*tree.camera, *tree.robot, which.instance);
    tree.camera->draw();
    const std::string extension = std::filesystem::path(filename).extension().string();
    tree.camera->saveImage(filename, (extension == ".png" || extension == ".PNG") ? "png" : "jpeg");
}

FruitHarvestCameraView FruitHarvestTask::measureWristCameraView(uint copy) {
    Copy &which = requireCopy(copy, "measureWristCameraView");
    Tree &tree = *which.tree;
    if (!tree.camera) {
        helios_runtime_error("ERROR (FruitHarvestTask::measureWristCameraView): Call enableWristCamera() first.");
    }
    if (which.target == nullptr) {
        helios_runtime_error("ERROR (FruitHarvestTask::measureWristCameraView): Call reset() first, so that the copy has a fruit to pick.");
    }
    prepareWristFrame(*tree.camera, *tree.robot, which.instance);
    tree.camera->draw();
    const size_t pixel_count = size_t(tree.camera->getWidth()) * size_t(tree.camera->getHeight());
    std::vector<uint> pixel_UUIDs(pixel_count);
    std::vector<float> depth(pixel_count);
    tree.camera->getPrimitiveImage(pixel_UUIDs.data());
    tree.camera->getDepthImage(depth.data());
    FruitHarvestCameraView view;
    const uint target_label = which.target->objID + organ_fruit_base;
    for (size_t k = 0; k < pixel_UUIDs.size(); k++) {
        if (pixel_UUIDs[k] == 0) {
            continue;
        }
        const auto organ = tree.organ_of_primitive.find(pixel_UUIDs[k] - 1);
        if (organ == tree.organ_of_primitive.end()) {
            continue;
        }
        if (organ->second == target_label) {
            view.target_fruit += 1.f;
        } else if (organ->second >= organ_fruit_base) {
            view.other_fruit += 1.f;
        } else if (organ->second == organ_leaf) {
            view.leaves += 1.f;
            if (depth[k] < 0.15f) {
                view.near_leaves += 1.f;
            }
        } else if (organ->second == organ_wood) {
            view.wood += 1.f;
        }
    }
    const float total = float(pixel_UUIDs.size());
    view.target_fruit /= total;
    view.other_fruit /= total;
    view.leaves /= total;
    view.near_leaves /= total;
    view.wood /= total;
    return view;
}

void FruitHarvestTask::enableMovieCamera(uint width, uint height) {
    if (width == 0 || height == 0) {
        helios_runtime_error("ERROR (FruitHarvestTask::enableMovieCamera): The image size must be positive.");
    }
    for (const std::unique_ptr<Tree> &tree: trees) {
        if (tree->hasCamera()) {
            helios_runtime_error("ERROR (FruitHarvestTask::enableMovieCamera): A camera is already enabled. Each camera clears the Context's geometry flags after each frame, so another would miss motion it drew.");
        }
    }
    for (const std::unique_ptr<Tree> &tree: trees) {
        RobotCameraParameters movie;
        movie.field_of_view = movie_camera_field_of_view;
        movie.antialiasing_samples = 8;
        movie.background = make_RGBcolor(0.72f, 0.83f, 0.95f);
        movie.light_direction = movie_light_direction;
        movie.shadows = true;
        tree->movie_camera = std::make_unique<RobotCamera>(&tree->engine, &tree->context, width, height, movie);
        tree->engine.setContextUpdateTolerance(1e-5f);
    }
}

void FruitHarvestTask::renderMovieFrameOf(Copy &copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, const vec3 &eye, const vec3 &look_at, bool wrist_view, std::vector<unsigned char> &image) {
    Tree &tree = *copy.tree;
    const std::string method = wrist_view ? "renderWristFrame" : "renderMovieFrame";
    if (!tree.movie_camera) {
        helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): Call enableMovieCamera() first.");
    }
    tree.showPose(copy.instance, link_poses, robot_joints, method);
    if (wrist_view) {
        vec3 wrist_eye, wrist_look_at;
        wristCameraPose(*tree.robot, copy.instance, wrist_eye, wrist_look_at);
        tree.movie_camera->setFieldOfView(wrist_camera_field_of_view);
        tree.movie_camera->setPose(wrist_eye, wrist_look_at);
    } else {
        tree.movie_camera->setFieldOfView(movie_camera_field_of_view);
        tree.movie_camera->setPose(eye, look_at);
    }
    // The geometry follows the poses just written into the simulation's state
    tree.movie_camera->render(copy.instance);
    image.resize(size_t(tree.movie_camera->getWidth()) * size_t(tree.movie_camera->getHeight()) * 3);
    tree.movie_camera->getColorImage(image.data());
}

#else

namespace {
    [[noreturn]] void requireVisualizer(const std::string &method) {
        helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): The cameras draw with the visualizer, which is not in this project's plug-ins. Add \"visualizer\" to the project's PLUGINS and rebuild.");
        throw std::logic_error("unreachable");
    }
} // namespace

void FruitHarvestTask::enableWristCamera(uint, uint) {
    requireVisualizer("enableWristCamera");
}

void FruitHarvestTask::renderWristCameras(std::vector<unsigned char> &) {
    requireVisualizer("renderWristCameras");
}

void FruitHarvestTask::saveWristCameraImage(uint, const std::string &) {
    requireVisualizer("saveWristCameraImage");
}

FruitHarvestCameraView FruitHarvestTask::measureWristCameraView(uint) {
    requireVisualizer("measureWristCameraView");
}

void FruitHarvestTask::enableMovieCamera(uint, uint) {
    requireVisualizer("enableMovieCamera");
}

void FruitHarvestTask::renderMovieFrameOf(Copy &, const std::vector<float> &, const std::vector<float> &, const vec3 &, const vec3 &, bool wrist_view, std::vector<unsigned char> &) {
    requireVisualizer(wrist_view ? "renderWristFrame" : "renderMovieFrame");
}

#endif

void FruitHarvestTask::renderMovieFrame(uint copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, const vec3 &eye, const vec3 &look_at, std::vector<unsigned char> &image) {
    renderMovieFrameOf(requireCopy(copy, "renderMovieFrame"), link_poses, robot_joints, eye, look_at, false, image);
}

void FruitHarvestTask::renderWristFrame(uint copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, std::vector<unsigned char> &image) {
    renderMovieFrameOf(requireCopy(copy, "renderWristFrame"), link_poses, robot_joints, make_vec3(0, 0, 0), make_vec3(0, 0, 0), true, image);
}

// ---- Movies drawn by radiation cameras ---- //

#ifdef HELIOS_ROBOTICS_RADIATION

void FruitHarvestTask::enableRadiationMovieCameras(uint width, uint height, uint wrist_width, uint wrist_height, uint antialiasing_samples, uint periodic_repeats, uint row_repeats) {
    if (width == 0 || height == 0 || wrist_width == 0 || wrist_height == 0 || antialiasing_samples == 0) {
        helios_runtime_error("ERROR (FruitHarvestTask::enableRadiationMovieCameras): The image sizes and the number of antialiasing samples must be positive.");
    }
    for (const std::unique_ptr<Tree> &tree: trees) {
        if (tree->hasCamera()) {
            helios_runtime_error("ERROR (FruitHarvestTask::enableRadiationMovieCameras): A camera is already enabled.");
        }
        if ((periodic_repeats > 0 || row_repeats > 0) && tree->movie_row_half_length <= 0.f) {
            helios_runtime_error("ERROR (FruitHarvestTask::enableRadiationMovieCameras): A periodic row repeats the drawn row of trees; call addMovieScenery() first.");
        }
    }
    for (const std::unique_ptr<Tree> &tree: trees) {
        Context &context = tree->context;
        // The robot and the scenery: color references and soil, chosen to look like the colors the visualizer draws them in
        context.loadXML("plugins/radiation/spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml", true);
        for (const auto &[objID, spectrum]: tree->painted_objects) {
            context.setPrimitiveData(context.getObjectPrimitiveUUIDs(objID), "reflectivity_spectrum", spectrum);
        }
        // The leaves, with LeafOptics' default properties; the bark and fruit take the spectra of the classes of their textures (applySpectralClassMaps() below)
        const std::vector<uint> leaves = tree->plantarchitecture.getAllLeafUUIDs();
        if (!leaves.empty()) {
            LeafOptics leafoptics(&context);
            leafoptics.disableMessages();
            leafoptics.run(leaves, LeafOpticsProperties(), "apple");
        }

        // The ground as an adaptive tile, so that it is smoothed and takes the sun's shadows per pixel like the rest (a lone patch is not smoothed, and shows
        // one value for the whole ground), with sub-patches finest in the alley beside the row, to resolve the sky the canopy hides there. With a periodic
        // boundary along the row, the ground spans the row's length along y exactly, as the trellis wires do, so that the repeats of the row meet without a
        // gap; with repeated rows, it spans one row spacing across the rows.
        context.deletePrimitive(tree->ground_UUID);
        const bool periodic_along = periodic_repeats > 0;
        const bool periodic_across = row_repeats > 0;
        const float row_half_length = tree->movie_row_half_length;
        const vec2 cell_x = periodic_across ? make_vec2(movie_row_cell_center_x - 0.5f * movie_row_spacing, movie_row_cell_center_x + 0.5f * movie_row_spacing) : make_vec2(-0.5f * movie_ground_size, 0.5f * movie_ground_size);
        const vec2 cell_y = periodic_along ? make_vec2(-row_half_length, row_half_length) : make_vec2(-0.5f * movie_ground_size, 0.5f * movie_ground_size);
        AdaptiveTileRefinement ground_refinement;
        ground_refinement.target = make_vec2(movie_ground_refinement_target_x, 0);
        ground_refinement.subpatch_size_min = movie_ground_subpatch_size_min;
        // The largest sub-patch of an adaptive tile may be at most half its smaller side, which a periodic cell can be narrow enough to limit
        ground_refinement.subpatch_size_max = std::min(movie_ground_subpatch_size_max, 0.5f * std::min(cell_x.y - cell_x.x, cell_y.y - cell_y.x));
        const uint ground = context.addAdaptiveTileObject(make_vec3(0.5f * (cell_x.x + cell_x.y), 0.5f * (cell_y.x + cell_y.y), 0), make_vec2(cell_x.y - cell_x.x, cell_y.y - cell_y.x), make_SphericalCoord(0, 0), ground_refinement,
                                                          make_RGBcolor(0.42f, 0.33f, 0.24f));
        tree->ground_UUID = 0;
        context.setPrimitiveData(context.getObjectPrimitiveUUIDs(ground), "reflectivity_spectrum", movie_ground_spectrum);

        std::vector<uint> robot_UUIDs;
        if (periodic_along || periodic_across) {
            // The period in each periodic direction is the extent of the geometry that repeats, so it must be the ground's: canopy that hangs past an edge of
            // the cell (the end trees' along the row) is moved by one period to the opposite edge, where it is the overhang of the next repeat, and whatever
            // still crosses an edge (an object moved or not that straddles it) is cut off there. The simulated tree and the robot lie well inside the cell.
            const vec2 period = make_vec2(cell_x.y - cell_x.x, cell_y.y - cell_y.x);
            const std::set<uint> robot_object_set(tree->robot_objects.begin(), tree->robot_objects.end());
            size_t wrapped_objects = 0;
            for (uint objID: context.getAllObjectIDs()) {
                if (objID == ground || robot_object_set.count(objID) != 0) {
                    continue;
                }
                vec3 min_corner, max_corner;
                context.getObjectBoundingBox(objID, min_corner, max_corner);
                const vec3 center = 0.5f * (min_corner + max_corner);
                vec3 shift = make_vec3(0, 0, 0);
                if (periodic_across && (center.x < cell_x.x || center.x > cell_x.y)) {
                    shift.x = center.x > cell_x.y ? -period.x : period.x;
                }
                if (periodic_along && (center.y < cell_y.x || center.y > cell_y.y)) {
                    shift.y = center.y > cell_y.y ? -period.y : period.y;
                }
                if (shift.x != 0.f || shift.y != 0.f) {
                    context.translateObject(objID, shift);
                    wrapped_objects++;
                }
            }
            for (uint objID: tree->robot_objects) {
                const std::vector<uint> UUIDs = context.getObjectPrimitiveUUIDs(objID);
                robot_UUIDs.insert(robot_UUIDs.end(), UUIDs.begin(), UUIDs.end());
            }
            const std::set<uint> robot_UUID_set(robot_UUIDs.begin(), robot_UUIDs.end());
            const float tolerance = 1e-4f;
            std::vector<uint> straddling;
            for (uint UUID: context.getAllUUIDs()) {
                if (robot_UUID_set.count(UUID) != 0) {
                    continue;
                }
                for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                    if ((periodic_across && (vertex.x < cell_x.x - tolerance || vertex.x > cell_x.y + tolerance)) || (periodic_along && (vertex.y < cell_y.x - tolerance || vertex.y > cell_y.y + tolerance))) {
                        straddling.push_back(UUID);
                        break;
                    }
                }
            }
            context.deletePrimitive(straddling);
            std::cout << "periodic cell " << period.x << " m across by " << period.y << " m along the row, repeated " << row_repeats << " times across and " << periodic_repeats << " times along each way: moved " << wrapped_objects
                      << " overhanging objects to the opposite edge, cut " << straddling.size() << " primitives crossing an edge" << std::endl;
        }

        tree->radiation = std::make_unique<RadiationModel>(&context);
        RadiationModel &radiation = *tree->radiation;
        radiation.disableMessages();
        if (periodic_along || periodic_across) {
            const std::string boundary = std::string(periodic_across ? "x" : "") + (periodic_along ? "y" : "");
            radiation.enforcePeriodicBoundary(boundary, robot_UUIDs, make_int2(static_cast<int>(row_repeats), static_cast<int>(periodic_repeats)));
        }
        const std::vector<std::string> bands = {"red", "green", "blue"};
        for (const std::string &band: bands) {
            radiation.addRadiationBand(band);
            radiation.disableEmission(band);
            radiation.setScatteringDepth(band, movie_scattering_depth);
            radiation.setDirectRayCount(band, movie_direct_rays);
            radiation.setDiffuseRayCount(band, movie_diffuse_rays);
        }
        // The sun from the same direction as the visualizer's light, and the sky
        const uint sun = radiation.addSunSphereRadiationSource(movie_light_direction);
        radiation.setSourceSpectrum(sun, "solar_spectrum_direct_ASTMG173");
        radiation.setDiffuseSpectrum("solar_spectrum_diffuse_ASTMG173");
        radiation.applySpectralClassMaps();
        radiation.enableCameraFluxSmoothing();

        for (const auto &[label, image_width, image_height, vertical_field_of_view]: {std::tuple<const char *, uint, uint, float>{"movie", width, height, movie_camera_field_of_view}, {"wrist", wrist_width, wrist_height, wrist_camera_field_of_view}}) {
            CameraProperties camera;
            camera.camera_resolution = make_int2(static_cast<int>(image_width), static_cast<int>(image_height));
            // The visualizer's field of view is vertical; a radiation camera's is horizontal
            camera.HFOV = 2.f * std::atan(std::tan(0.5f * deg2rad(vertical_field_of_view)) * float(image_width) / float(image_height)) * 180.f / float(M_PI);
            camera.lens_diameter = 0.f;
            camera.model = "Nikon D700";
            radiation.addRadiationCamera(label, bands, make_vec3(-3, 0, 1), make_vec3(0, 0, 1), camera, antialiasing_samples);
            radiation.setCameraSpectralResponseFromLibrary(label, "Nikon_D700");
        }
    }
}

void FruitHarvestTask::renderRadiationMovieFrame(uint copy, const std::vector<float> &link_poses, const std::vector<float> &robot_joints, const vec3 &eye, const vec3 &look_at, const std::string &directory, std::string &movie_image_file,
                                                 std::string &wrist_image_file) {
    Copy &which = requireCopy(copy, "renderRadiationMovieFrame");
    Tree &tree = *which.tree;
    if (!tree.radiation) {
        helios_runtime_error("ERROR (FruitHarvestTask::renderRadiationMovieFrame): Call enableRadiationMovieCameras() first.");
    }
    tree.showPose(which.instance, link_poses, robot_joints, "renderRadiationMovieFrame");
    tree.engine.updateContextGeometry(which.instance);
    RadiationModel &radiation = *tree.radiation;
    radiation.setCameraPosition("movie", eye);
    radiation.setCameraLookat("movie", look_at);
    vec3 wrist_eye, wrist_look_at;
    wristCameraPose(*tree.robot, which.instance, wrist_eye, wrist_look_at);
    radiation.setCameraPosition("wrist", wrist_eye);
    radiation.setCameraLookat("wrist", wrist_look_at);
    const std::vector<std::string> bands = {"red", "green", "blue"};
    radiation.updateGeometry();
    radiation.runBand(bands);
    for (const auto &[label, file_name]: {std::pair<const char *, std::string *>{"movie", &movie_image_file}, {"wrist", &wrist_image_file}}) {
        const std::string written = radiation.writeCameraImage(label, bands, "RGB", directory + "/");
        if (written.empty()) {
            helios_runtime_error("ERROR (FruitHarvestTask::renderRadiationMovieFrame): The " + std::string(label) + " camera's image could not be written to " + directory + ".");
        }
        *file_name = written;
    }
}

#else

namespace {
    [[noreturn]] void requireRadiation(const std::string &method) {
        helios_runtime_error("ERROR (FruitHarvestTask::" + method + "): The radiation movie cameras need the radiation and leafoptics plug-ins, which are not both in this project's plug-ins. Add them to the project's PLUGINS and rebuild.");
        throw std::logic_error("unreachable");
    }
} // namespace

void FruitHarvestTask::enableRadiationMovieCameras(uint, uint, uint, uint, uint, uint, uint) {
    requireRadiation("enableRadiationMovieCameras");
}

void FruitHarvestTask::renderRadiationMovieFrame(uint, const std::vector<float> &, const std::vector<float> &, const vec3 &, const vec3 &, const std::string &, std::string &, std::string &) {
    requireRadiation("renderRadiationMovieFrame");
}

#endif
