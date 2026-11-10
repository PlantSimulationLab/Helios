// Apple harvesting environment. See harvest_env.h.
//
// Each tree has its own compiled model, since no two trees have the same structure, and several simulation instances of it.
// Only the wood bearing the candidate fruit is flexible, which keeps almost all of the speed of rigid wood.

#include "harvest_env.h"

#include "ParallelRobotSimulations.h"
#include "PlantPhysics.h"
#include "ProtectiveStop.h"
#include "RobotCamera.h"
#include "ToolVelocityAction.h"
#include "VacuumCup.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {

    // Control
    const uint physics_steps_per_control = 20; // 25 Hz at a 2 ms physics step
    const float max_tool_speed = 0.3f; // m/s
    const float max_target_lead = 0.08f; // m the commanded tool position can lead the actual one, so the target cannot wind up while the arm is blocked
    const int max_control_steps = 250; // 10 s

    // Gripper: an ideal vacuum cup, a bowl whose mouth is a round lip. It seals only when the whole lip is on the fruit, which means the fruit sits in the
    // mouth with its center on the cup's axis, and then holds the fruit seated there: springs hold the fruit's center on the axis, where its surface
    // touches the whole lip, and its orientation as it was when the seal formed, both relative to the cup, reacting on the arm. The seal breaks when the
    // fruit pulls out of the cup, slides across the mouth, or twists or peels the lip, too hard. The limits on sliding and peeling were chosen for this
    // task: suction cups hold less sideways than straight out, and least of all against peeling.
    const float cup_lip_radius = 0.025f; // m from the cup's axis to the middle of its lip: a 5 cm cup
    const float cup_lip_thickness = 0.004f; // m, the radius of the lip's cross section, which harvest_arm.urdf gives as a ring of spheres
    const float grip_reach = 0.005f; // m: every point of the lip within this of the fruit's surface, about how far a compliant lip reaches to seal
    const float grip_stiffness = 2000.f; // N/m
    const float grip_damping = 20.f; // N*s/m
    const float grip_ramp_seconds = 0.1f; // the grip builds up over this time, as a vacuum does
    const float grip_limit = 40.f; // N of pull straight out of the face above which the cup loses its seal
    const float grip_shear_limit = 20.f; // N along the face: friction of 0.5 on the face, pressed on by the vacuum
    const float grip_moment_limit = 1.f; // N m about the face's center, from twisting and from loads that peel the cup: about the full pull at the rim of a 5 cm cup

    // Force to detach a fruit, drawn for each episode. OrchardBench uses 14 to 23 N, and 28.3 N detached 95% of apples in a straight pull in a Michigan State thesis.
    const float min_detach_force = 14.f; // N
    const float max_detach_force = 28.f; // N

    const std::string plant_model = "apple_fruitingwall";
    const float plant_age = 650.f; // days

    // Fruit that can be targeted must be within reach of the arm
    const float min_fruit_height = 0.8f; // m
    const float max_fruit_height = 1.75f; // m
    const size_t candidates_per_tree = 8;
    const int flexible_internodes_per_candidate = 8;

    // How long the fruit and flexible wood settle under gravity, once, at 2 ms per step. Episodes start from the pose reached, with no velocity, so a shorter
    // settling leaves the tree out of balance and it swings for much of each episode. That also costs time to render, since all the wood that swings has to be redrawn.
    const uint settling_physics_steps = 10000; // 20 s

    const vec3 robot_base = make_vec3(-1.05f, 0, 0);
    const vec3 cup_face = make_vec3(0.32f, 0, 0); // the center of the cup's mouth, in the tool's frame; the cup's axis is the tool's x axis
    const std::vector<float> tucked_pose = {0.f, 0.f, 110.f, -150.f, 40.f, 0.f}; // drive, yaw, shoulder, elbow, wrist, roll
    // A robot stops itself when it presses on something this hard, as a protective stop does. Chosen for this task, not taken from a particular robot.
    const float protective_stop_force = 50.f; // N

    // Position servos on the robot's joints. The force limits are the effort limits of the joints in the robot's model.
    struct JointServo {
        const char *joint;
        float stiffness; // N/m or N m/rad
        float damping; // N s/m or N m s/rad
        float force_limit; // N or N m
        float armature; // kg or kg m^2
    };
    const JointServo joint_servos[] = {{"drive", 20000.f, 5000.f, 2000.f, 20.f}, {"yaw", 2000.f, 400.f, 300.f, 2.f}, {"shoulder", 2000.f, 400.f, 300.f, 2.f},
                                       {"elbow", 1000.f, 200.f, 200.f, 1.f}, {"wrist", 200.f, 30.f, 60.f, 0.2f}, {"roll", 30.f, 5.f, 30.f, 0.05f}};

    // Smaller motion of a body does not move its geometry for the camera. It is about 0.1 pixel at 128x128 for something 30 cm from the camera.
    const float context_update_tolerance = 0.0005f; // m

    const float start_offset = 0.2f; // m along the row that the robot starts from the fruit, at most
    // A start near the fruit, for training only (harvest_env_set_near_start_fraction()): the cup this far short of where the fruit would be seated, along
    // the line from the turret, and up to near_start_off_axis off that line, sideways and up or down
    const float near_start_min_standoff = 0.02f; // m
    const float near_start_max_standoff = 0.12f; // m
    const float near_start_off_axis = 0.03f; // m
    const int near_start_attempts = 5; // placements tried, each kept only if the arm reaches it without touching anything; otherwise the tucked start

    const int observation_size = 26;
    const int action_size = 4;

    struct Candidate {
        uint link;
        uint bearing_wood_link; // the nearest internode toward the trunk
        uint objID;
        vec3 settled_position;
        vec3 settled_wood_position;
    };

    // Labels for measuring what the camera sees. A fruit is labeled by its object, offset past the others.
    const uint organ_other = 0;
    const uint organ_leaf = 1;
    const uint organ_wood = 2;
    const uint organ_fruit_base = 3;

    VacuumCupParameters cupParameters() {
        VacuumCupParameters cup;
        cup.lip_radius = cup_lip_radius;
        cup.lip_thickness = cup_lip_thickness;
        cup.seal_reach = grip_reach;
        cup.stiffness = grip_stiffness;
        cup.damping = grip_damping;
        cup.ramp_seconds = grip_ramp_seconds;
        cup.pull_limit = grip_limit;
        cup.shear_limit = grip_shear_limit;
        cup.moment_limit = grip_moment_limit;
        return cup;
    }

    USDExportParameters plantPhysics() {
        USDExportParameters plant_physics;
        plant_physics.elastic_modulus = 5e9f;
        plant_physics.wood_density = 750.f;
        plant_physics.fruit_mass = 0.15f;
        return plant_physics;
    }

    // Fruit at heights the arm can reach
    std::vector<uint> findReachableFruit(const PlantArticulation &articulation) {
        const float unbounded = std::numeric_limits<float>::infinity();
        return findPlantLinksInBox(articulation, ARTICULATION_LINK_FRUIT, make_vec3(-unbounded, -unbounded, min_fruit_height), make_vec3(unbounded, unbounded, max_fruit_height));
    }

    // Reachable fruit, nearest the robot first, since those are the ones not hidden behind the wall
    std::vector<uint> chooseCandidateFruit(const PlantArticulation &articulation, int seed) {
        std::vector<uint> reachable = findReachableFruit(articulation);
        std::sort(reachable.begin(), reachable.end(), [&](uint a, uint b) { return articulation.links[a].position.x < articulation.links[b].position.x; });
        if (reachable.size() > candidates_per_tree) {
            reachable.resize(candidates_per_tree);
        }
        if (reachable.empty()) {
            helios_runtime_error("ERROR (harvest_env): Tree with seed " + std::to_string(seed) + " has no fruit within reach of the robot.");
        }
        return reachable;
    }

    // The internodes behind each candidate fruit, which stay flexible while the rest of the wood is rigid
    std::vector<uint> chooseFlexibleInternodes(const PlantArticulation &articulation, const std::vector<uint> &candidates) {
        std::vector<uint> flexible;
        for (uint fruit: candidates) {
            const std::vector<uint> supporting = findSupportingInternodes(articulation, fruit, flexible_internodes_per_candidate);
            flexible.insert(flexible.end(), supporting.begin(), supporting.end());
        }
        return flexible;
    }

    struct Tree {
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
        std::unique_ptr<RobotCamera> camera;
        std::unordered_map<uint, uint> organ_of_primitive; // for measuring what the camera sees: which kind of plant organ each primitive belongs to, or which fruit

        // For showing a simulation from outside (movies): the links of the plant, parents first, with the MuJoCo body of each (-1 if it has none) and its
        // rest pose (x, y, z, qw, qx, qy, qz), which is also the rest pose of its body
        std::vector<std::string> link_names;
        std::vector<int> link_parents;
        std::vector<int> link_bodies;
        std::vector<std::array<mjtNum, 7>> link_rest_poses;
        std::unique_ptr<RobotCamera> movie_camera;

        Tree(int seed, int instance_count, bool leaves, bool show_arm = false) : plantarchitecture(&context), engine(&context) {
            context.seedRandomGenerator(seed);
            plantarchitecture.disableMessages();
            engine.disableMessages();
            plantarchitecture.loadPlantModelFromLibrary(plant_model);
            uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), plant_age);
            if (!leaves) {
                plantarchitecture.removePlantLeaves(plantID);
            }

            const USDExportParameters plant_physics = plantPhysics();
            const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID, plant_physics);

            nearest_wood = std::make_unique<NearestPlantOrgan>(articulation, ARTICULATION_LINK_INTERNODE);

            const std::vector<uint> reachable = chooseCandidateFruit(articulation, seed);

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
            plant_parameters.flexible_links = chooseFlexibleInternodes(articulation, reachable);
            plant = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, plant_physics, plant_parameters);

            robot = std::make_unique<Robot>(&engine, "../harvest_arm.urdf", "robot", robot_base);
            // The force limits are the effort limits of the joints in the robot's model
            for (const JointServo &servo: joint_servos) {
                robot->addJointServo(servo.joint, servo.stiffness, servo.damping, servo.force_limit, servo.armature);
            }
            robot->setTool("tool", cup_face, make_vec3(1, 0, 0));

            // What the camera can see of the robot, in its zero pose, in which the tool lies along +x from the wrist
            const vec3 wrist = robot_base + make_vec3(0.15f + 0.65f + 0.6f, 0, 0.75f);
            // The neck, and the cup: its bottom and its lip as in harvest_arm.urdf, and its wall between them, which does not collide
            const RGBcolor rubber = make_RGBcolor(0.15f, 0.15f, 0.15f);
            const vec3 mouth = wrist + cup_face;
            engine.bindObjectToBody(context.addConeObject(16, wrist, wrist + make_vec3(0.28f, 0, 0), 0.03f, 0.018f, make_RGBcolor(0.9f, 0.4f, 0.08f)), "robot/tool");
            engine.bindObjectToBody(context.addSphereObject(16, wrist + make_vec3(0.29f, 0, 0), 0.012f, rubber), "robot/tool");
            engine.bindObjectToBody(context.addConeObject(24, wrist + make_vec3(0.28f, 0, 0), mouth, 0.016f, cup_lip_radius, rubber), "robot/tool");
            std::vector<vec3> lip_nodes;
            for (int k = 0; k <= 32; k++) {
                const float angle = 2.f * float(M_PI) * float(k) / 32.f;
                lip_nodes.push_back(mouth + cup_lip_radius * make_vec3(0, std::cos(angle), std::sin(angle)));
            }
            engine.bindObjectToBody(context.addTubeObject(8, lip_nodes, std::vector<float>(lip_nodes.size(), cup_lip_thickness), std::vector<RGBcolor>(lip_nodes.size(), rubber)), "robot/tool");
            if (show_arm) {
                // The rest of the arm, as the collision shapes of harvest_arm.urdf in its zero pose, for showing the robot from outside. The wrist camera
                // does not need it, and the observations do not depend on it.
                const RGBcolor arm_color = make_RGBcolor(0.95f, 0.6f, 0.12f);
                const RGBcolor joint_color = make_RGBcolor(0.25f, 0.25f, 0.28f);
                const vec3 shoulder = robot_base + make_vec3(0.15f, 0, 0.75f);
                const vec3 elbow = shoulder + make_vec3(0.65f, 0, 0);
                engine.bindObjectToBody(context.addBoxObject(robot_base + make_vec3(0, 0, 0.22f), make_vec3(0.55f, 0.8f, 0.2f), make_int3(1, 1, 1), joint_color), "robot/platform");
                engine.bindObjectToBody(context.addConeObject(24, robot_base + make_vec3(0.15f, 0, 0.25f), shoulder, 0.075f, 0.075f, arm_color), "robot/turret");
                engine.bindObjectToBody(context.addSphereObject(16, shoulder, 0.07f, joint_color), "robot/upper_arm");
                engine.bindObjectToBody(context.addConeObject(24, shoulder, elbow, 0.05f, 0.05f, arm_color), "robot/upper_arm");
                engine.bindObjectToBody(context.addSphereObject(16, elbow, 0.055f, joint_color), "robot/forearm");
                engine.bindObjectToBody(context.addConeObject(24, elbow, wrist, 0.038f, 0.038f, arm_color), "robot/forearm");
                engine.bindObjectToBody(context.addSphereObject(16, wrist, 0.042f, joint_color), "robot/wrist_link");
            }
            context.addPatch(make_vec3(0, 0, 0), make_vec2(20.f, 20.f), make_SphericalCoord(0, 0), make_RGBcolor(0.42f, 0.33f, 0.24f));

            PhysicsEngineParameters parameters;
            parameters.ground_collisions = true;
            // The tree's parts collide with each other, so that the arm cannot push a fruit into wood or another fruit. Those that the tree was grown with
            // overlapping stay as they are: pushing them apart would bend the wood against the fruit and load their stems before the arm touches them.
            parameters.articulation_collisions = true;
            parameters.keep_initial_link_overlaps = true;
            engine.initialize(parameters);
            robot->initialize();

            while (engine.getInstanceCount() < static_cast<uint>(instance_count)) {
                engine.createInstance();
            }
            robot->prepareInstances();

            const mjModel *model = engine.getMuJoCoModel();
            tool_body_ID = mj_name2id(model, mjOBJ_BODY, "robot/tool");

            // The contact sensor covers the whole robot. The fruit being picked is its "target", which the cup is meant to touch, so the stop leaves it out.
            contact = std::make_unique<ContactSensor>(&engine, mj_id2name(model, mjOBJ_BODY, model->body_rootid[tool_body_ID]));
            protective_stop = std::make_unique<ProtectiveStop>(contact.get(), protective_stop_force, std::vector<std::string>{"target"});
            cup = std::make_unique<VacuumCup>(robot.get(), "tool", cup_face, make_vec3(1, 0, 0), cupParameters());
            tool_action = std::make_unique<ToolVelocityAction>(robot.get(), max_tool_speed, max_target_lead);
            // The cup is level and points away from the turret, which turns about a vertical axis 0.15 m ahead of the platform as it drives along the row
            tool_action->setToolAxisAwayFromPivot(make_vec3(0.15f, 0, 0), "drive");
            tool_action->setLockedJoints({"roll"});
            grip_action = std::make_unique<VacuumCupGripAction>(cup.get());
            control_loop = std::make_unique<ControlLoop>(&engine, physics_steps_per_control);
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
            robot->setJointPositions(tucked_pose, 0);
            engine.step(settling_physics_steps, 0);
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
    };

    struct Slot {
        Tree *tree = nullptr;
        uint instance = 0;
        std::mt19937 random;
        float near_start_fraction = 0.f; // share of episodes started with the cup near the fruit
        const Candidate *target = nullptr;
        float detach_force = 0.f;
        float previous_load = 0.f;
        float previous_distance = 0.f;
        int control_step = 0;
        vec3 pull_direction;
        HarvestEpisodeRecord current{};
        HarvestEpisodeRecord last{};
        // How the simulation looked when its last episode ended, for showing it (see captureDisplayState())
        std::vector<float> last_link_poses;
        std::vector<float> last_robot_joints;
        // For showing: control steps a success goes on for, with the cup carrying the freed fruit away (see harvest_env_set_follow_through())
        int follow_through_steps = 0;
        int follow_through_remaining = 0; // nonzero while a success goes on
        bool gripped_before_step = false; // whether the cup was gripping when the control step started
        HarvestEpisodeRecord detached_record{}; // the record of an episode that goes on, as it was when the fruit detached
    };

    // The contact force of the robot on the tree, other than on the fruit it is picking: the robot is meant to touch that fruit, and pushing it too hard
    // knocks it off. This is what the protective stop acts on.
    float contactOnTree(const Slot &slot) {
        return slot.tree->protective_stop->measureForce(slot.instance);
    }

    // A camera on the wrist, behind and above the cup, looking along the tool past the cup
    void wristCameraPose(const Robot &robot, uint instance, vec3 &eye, vec3 &look_at) {
        const vec3 cup = robot.getToolPosition(instance);
        const vec3 axis = robot.getToolAxis(instance);
        eye = cup - 0.3f * axis + make_vec3(0, 0, 0.08f);
        look_at = cup + 0.3f * axis;
    }

    const float wrist_camera_field_of_view = 70.f;
    const float movie_camera_field_of_view = 40.f;

    // The part of rendering that needs no OpenGL, so that trees can do it in parallel
    void prepareSlotFrame(Slot &slot) {
        vec3 eye, look_at;
        wristCameraPose(*slot.tree->robot, slot.instance, eye, look_at);
        slot.tree->camera->setPose(eye, look_at);
        slot.tree->camera->prepare(slot.instance);
    }

    void renderSlot(Slot &slot) {
        prepareSlotFrame(slot);
        slot.tree->camera->draw();
    }

    // The world pose (x, y, z, qw, qx, qy, qz) of every link of the plant, NaN for a link with no body, and the robot's joint positions (m, degrees)
    void captureDisplayState(const Slot &slot, std::vector<float> &link_poses, std::vector<float> &robot_joints) {
        const Tree &tree = *slot.tree;
        const mjData *data = tree.engine.getMuJoCoData(slot.instance);
        link_poses.assign(7 * tree.link_names.size(), std::numeric_limits<float>::quiet_NaN());
        for (size_t i = 0; i < tree.link_names.size(); i++) {
            const int body = tree.link_bodies[i];
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
        robot_joints = tree.robot->getJointPositions(slot.instance);
    }

    // Show a simulation in a given pose, which need not be one this simulation reached: it may come from another simulator. Every link of the plant takes
    // the given world pose, or, where that is NaN, follows its parent as it does at rest. The robot takes the given joint positions. This leaves the
    // simulation in the pose shown, with no velocity, so an environment used for this should only be used for showing. With wrist_view, eye and look_at
    // are ignored and the camera is where the wrist camera would be (pointCamera()), with its field of view.
    void renderMovieFrame(Slot &slot, const float *link_poses, const float *robot_joints, const vec3 &eye, const vec3 &look_at, bool wrist_view, unsigned char *image) {
        Tree &tree = *slot.tree;
        if (!tree.movie_camera) {
            helios_runtime_error("ERROR (harvest_env_render_movie_frame): Call harvest_env_enable_movie_camera() first.");
        }
        mjData *data = tree.engine.getMuJoCoData(slot.instance);
        tree.robot->setJointPositions(std::vector<float>(robot_joints, robot_joints + tree.robot->getJointPositions(slot.instance).size()), slot.instance);

        const size_t link_count = tree.link_names.size();
        std::vector<std::array<mjtNum, 7>> world(link_count);
        for (size_t i = 0; i < link_count; i++) {
            const int parent = tree.link_parents[i];
            const float *given = link_poses + 7 * i;
            if (std::isfinite(given[0])) {
                for (int k = 0; k < 7; k++) {
                    world[i][k] = given[k];
                }
                mju_normalize4(world[i].data() + 3);
            } else if (parent >= 0) {
                if (size_t(parent) >= i) {
                    helios_runtime_error("ERROR (harvest_env_render_movie_frame): Link " + tree.link_names[i] + " is listed before its parent.");
                }
                // Carried by the parent: pose = parent * parent_rest^-1 * rest
                const std::array<mjtNum, 7> &rest = tree.link_rest_poses[i];
                const std::array<mjtNum, 7> &parent_rest = tree.link_rest_poses[parent];
                mjtNum parent_rest_conjugate[4], relative_quat[4], offset[3], relative_position[3], moved[3];
                mju_negQuat(parent_rest_conjugate, parent_rest.data() + 3);
                mju_mulQuat(relative_quat, parent_rest_conjugate, rest.data() + 3);
                mju_sub3(offset, rest.data(), parent_rest.data());
                mju_rotVecQuat(relative_position, offset, parent_rest_conjugate);
                mju_rotVecQuat(moved, relative_position, world[parent].data() + 3);
                mju_add3(world[i].data(), world[parent].data(), moved);
                mju_mulQuat(world[i].data() + 3, world[parent].data() + 3, relative_quat);
            } else {
                world[i] = tree.link_rest_poses[i];
            }
            const int body = tree.link_bodies[i];
            if (body >= 0) {
                mju_copy3(data->xpos + 3 * body, world[i].data());
                mju_copy4(data->xquat + 4 * body, world[i].data() + 3);
                mju_quat2Mat(data->xmat + 9 * body, world[i].data() + 3);
            }
        }

        if (wrist_view) {
            vec3 wrist_eye, wrist_look_at;
            wristCameraPose(*tree.robot, slot.instance, wrist_eye, wrist_look_at);
            tree.movie_camera->setFieldOfView(wrist_camera_field_of_view);
            tree.movie_camera->setPose(wrist_eye, wrist_look_at);
        } else {
            tree.movie_camera->setFieldOfView(movie_camera_field_of_view);
            tree.movie_camera->setPose(eye, look_at);
        }
        // The geometry follows the poses just written into the simulation's state
        tree.movie_camera->render(slot.instance);
        tree.movie_camera->getColorImage(image);
    }

    void writeObservation(const Slot &slot, float *observation) {
        Robot &robot = *slot.tree->robot;
        const mjModel *model = slot.tree->engine.getMuJoCoModel();
        const mjData *data = slot.tree->engine.getMuJoCoData(slot.instance);
        const VacuumCup &cup = *slot.tree->cup;
        const vec3 tool = robot.getToolPosition(slot.instance);
        const vec3 fruit = cup.getTargetPosition(slot.instance);
        const vec3 relative = fruit - tool;
        const std::vector<float> joints = robot.getJointPositions(slot.instance);
        mjtNum velocity[6];
        mj_objectVelocity(model, data, mjOBJ_BODY, slot.tree->tool_body_ID, velocity, 0);
        const float contact = contactOnTree(slot);
        const vec3 wood = slot.tree->nearest_wood->find(tool).axis_offset;

        int i = 0;
        observation[i++] = relative.x;
        observation[i++] = relative.y;
        observation[i++] = relative.z;
        observation[i++] = cup.getSealGap(slot.instance);
        observation[i++] = tool.x - robot_base.x;
        observation[i++] = tool.y - joints[0];
        observation[i++] = tool.z;
        observation[i++] = float(velocity[3]);
        observation[i++] = float(velocity[4]);
        observation[i++] = float(velocity[5]);
        observation[i++] = joints[0];
        for (size_t j = 1; j < 6; j++) {
            observation[i++] = deg2rad(joints[j]);
        }
        observation[i++] = cup.isGripped(slot.instance) ? 1.f : 0.f;
        observation[i++] = cup.getSealLoad(slot.instance); // how near the seal is to breaking
        observation[i++] = contact / 10.f;
        observation[i++] = float(slot.control_step) / float(max_control_steps);
        observation[i++] = wood.x;
        observation[i++] = wood.y;
        observation[i++] = wood.z;
        const vec3 off_axis = cup.getOffAxis(slot.instance);
        observation[i++] = off_axis.x;
        observation[i++] = off_axis.y;
        observation[i++] = off_axis.z;
    }

    // Whether any part of the robot touches anything other than itself and the ground
    bool robotTouches(const Slot &slot) {
        return slot.tree->contact->isPenetrating({"ground"}, slot.instance);
    }

    // Place the arm with the cup a few centimeters short of the fruit and somewhat out of line with it, pointing away from the turret, so that training
    // practices lining the cup up and sealing it. Returns false, leaving the arm as it was, when no placement tried is reachable without touching anything.
    bool placeCupNearFruit(Slot &slot, const std::vector<float> &start_pose) {
        Tree &tree = *slot.tree;
        std::uniform_real_distribution<float> unit(0.f, 1.f);
        const vec3 fruit = tree.cup->getTargetPosition(slot.instance);
        for (int attempt = 0; attempt < near_start_attempts; attempt++) {
            vec3 line = fruit - make_vec3(robot_base.x + 0.15f, start_pose[0], fruit.z);
            line.normalize();
            const vec3 sideways = make_vec3(-line.y, line.x, 0.f);
            const float standoff = tree.cup->getSeatDistance(slot.instance) + near_start_min_standoff + unit(slot.random) * (near_start_max_standoff - near_start_min_standoff);
            const vec3 target = fruit - standoff * line + near_start_off_axis * ((2.f * unit(slot.random) - 1.f) * sideways + (2.f * unit(slot.random) - 1.f) * make_vec3(0, 0, 1));
            vec3 pointing = target - make_vec3(robot_base.x + 0.15f, start_pose[0], target.z);
            pointing.normalize();
            const RobotInverseKinematicsResult pose = tree.robot->solveInverseKinematics(target, pointing, {"roll"}, slot.instance);
            if (pose.converged) {
                tree.robot->setJointPositions(pose.joint_positions, slot.instance);
                if ((tree.robot->getToolPosition(slot.instance) - target).magnitude() < 0.01f && !robotTouches(slot)) {
                    return true;
                }
                tree.robot->setJointPositions(start_pose, slot.instance);
            }
        }
        return false;
    }

    void resetSlot(Slot &slot) {
        Tree &tree = *slot.tree;
        PhysicsEngine &engine = tree.engine;
        // Back to the settled tree, with the cup, the stop, and the cup's velocity control reset
        tree.control_loop->resetInstance(slot.instance);

        std::uniform_int_distribution<size_t> pick_fruit(0, tree.candidates.size() - 1);
        std::uniform_real_distribution<float> unit(0.f, 1.f);
        slot.target = &tree.candidates[pick_fruit(slot.random)];
        slot.detach_force = min_detach_force + unit(slot.random) * (max_detach_force - min_detach_force);
        // The stem fails when the load on it passes the force for this fruit
        engine.clearDetachForces(slot.instance);
        engine.setDetachForce(tree.plant, slot.target->link, slot.detach_force, slot.instance);
        tree.cup->setTarget(tree.plant, slot.target->link, slot.instance);
        tree.contact->setTargetLink(tree.plant, slot.target->link, slot.instance);

        std::vector<float> start_pose = tucked_pose;
        start_pose[0] = slot.target->settled_position.y + (2.f * unit(slot.random) - 1.f) * start_offset;
        tree.robot->setJointPositions(start_pose, slot.instance);
        if (slot.near_start_fraction > 0.f && unit(slot.random) < slot.near_start_fraction) {
            placeCupNearFruit(slot, start_pose);
        }

        slot.follow_through_remaining = 0;
        slot.previous_load = 0.f;
        slot.control_step = 0;
        slot.previous_distance = tree.cup->getSealGap(slot.instance);
        slot.pull_direction = make_vec3(0, 0, 0);
        slot.current = HarvestEpisodeRecord{};
        slot.current.detach_force = slot.detach_force;
        slot.current.steps_to_grip = -1;
    }

    // Keep the direction the cup last pulled the fruit in, for the episode's record
    void recordPullDirection(Slot &slot) {
        const vec3 on_fruit = slot.tree->cup->getForceOnTarget(slot.instance);
        if (on_fruit.magnitude() > 1e-6f) {
            slot.pull_direction = on_fruit / on_fruit.magnitude();
        }
    }

    // After a control step following a success, with the cup backing straight away from the tree whatever the action, carrying the fruit for as long as the
    // grip holds (see followThroughAction()). The episode is a success either way, and ends when the steps run out, with the record it had when the fruit detached.
    void finishFollowThroughStep(Slot &slot, float *observation, float &reward, int &outcome, float *terminal_observation) {
        reward = 0.f;
        outcome = HARVEST_RUNNING;
        slot.follow_through_remaining--;
        if (slot.follow_through_remaining == 0) {
            outcome = HARVEST_SUCCESS;
            slot.last = slot.detached_record;
            captureDisplayState(slot, slot.last_link_poses, slot.last_robot_joints);
            writeObservation(slot, terminal_observation);
            resetSlot(slot);
        }
        writeObservation(slot, observation);
    }

    // The action of a control step after a success, for showing: the cup backs straight away from the tree, and grips no more
    const float follow_through_action[action_size] = {-1.f, 0.f, 0.f, 0.f};

    // After a control step, which moved the cup's target and sealed it if asked and it could (the actions), then simulated until the control step ended, or
    // the fruit detached, the seal broke, or the protective stop tripped (the control loop)
    void finishStep(Slot &slot, float *observation, float &reward, int &outcome, float *terminal_observation) {
        if (slot.follow_through_remaining > 0) {
            finishFollowThroughStep(slot, observation, reward, outcome, terminal_observation);
            return;
        }
        Tree &tree = *slot.tree;
        PhysicsEngine &engine = tree.engine;
        VacuumCup &cup = *tree.cup;

        // A seal that broke in the step that ended it still held the fruit until then
        const bool holding = cup.isGripped(slot.instance) || cup.isSealBroken(slot.instance);
        const bool gripped_now = !slot.gripped_before_step && holding;
        if (gripped_now) {
            slot.current.steps_to_grip = slot.control_step;
        }
        recordPullDirection(slot);
        const float load = engine.getDetachableLinkLoad(tree.plant, slot.target->link, slot.instance);
        const float peak_contact = tree.protective_stop->getPeakForce(slot.instance);
        outcome = HARVEST_RUNNING;
        if (engine.isLinkDetached(tree.plant, slot.target->link, slot.instance)) {
            outcome = holding ? HARVEST_SUCCESS : HARVEST_KNOCKED_OFF;
        } else if (cup.isSealBroken(slot.instance)) {
            outcome = HARVEST_SLIPPED;
        } else if (tree.protective_stop->isStopped(slot.instance)) {
            outcome = HARVEST_COLLISION;
        }
        slot.control_step++;
        if (outcome == HARVEST_RUNNING && slot.control_step >= max_control_steps) {
            outcome = HARVEST_TIMEOUT;
        }

        // Reward: approach the fruit, grip it, load the stem, and detach it, without pushing on the tree
        const float distance = cup.getSealGap(slot.instance);
        reward = -0.005f;
        if (!holding) {
            reward += 5.f * (slot.previous_distance - distance);
        }
        if (gripped_now) {
            reward += 1.f;
        }
        if (holding) {
            reward += 3.f * (load - slot.previous_load) / max_detach_force;
        }
        reward -= 0.02f * std::min(peak_contact, protective_stop_force);
        if (outcome == HARVEST_SUCCESS) {
            reward += 10.f;
        } else if (outcome == HARVEST_SLIPPED || outcome == HARVEST_KNOCKED_OFF || outcome == HARVEST_COLLISION) {
            reward -= 2.f;
        }
        slot.previous_distance = distance;
        slot.previous_load = load;

        mjtNum tool_velocity[6];
        mj_objectVelocity(engine.getMuJoCoModel(), engine.getMuJoCoData(slot.instance), mjOBJ_BODY, tree.tool_body_ID, tool_velocity, 0);
        slot.current.peak_contact_force = std::max(slot.current.peak_contact_force, peak_contact);
        slot.current.peak_tool_speed = std::max(slot.current.peak_tool_speed, float(mju_norm3(tool_velocity + 3)));
        slot.current.peak_wood_motion = std::max(slot.current.peak_wood_motion, (engine.getLinkPosition(tree.plant, slot.target->bearing_wood_link, slot.instance) - slot.target->settled_wood_position).magnitude());
        slot.current.steps = slot.control_step;
        slot.current.final_stem_load = load;
        slot.current.final_grip_force = cup.getPullForce(slot.instance);
        slot.current.pull_x = slot.pull_direction.x;
        slot.current.pull_y = slot.pull_direction.y;
        slot.current.pull_z = slot.pull_direction.z;

        if (outcome == HARVEST_SUCCESS && slot.follow_through_steps > 0) {
            // The episode goes on for showing; its reward and record are those of the success
            slot.detached_record = slot.current;
            slot.follow_through_remaining = slot.follow_through_steps;
            outcome = HARVEST_RUNNING;
        }
        if (outcome != HARVEST_RUNNING) {
            slot.last = slot.current;
            captureDisplayState(slot, slot.last_link_poses, slot.last_robot_joints);
            writeObservation(slot, terminal_observation);
            resetSlot(slot);
        }
        writeObservation(slot, observation);
    }

    struct Environment {
        std::vector<std::unique_ptr<Tree>> trees;
        std::vector<Slot> slots; // in the order of the copies of simulations: the instances of each tree, tree by tree
        std::unique_ptr<ParallelRobotSimulations> simulations; // destroyed before the trees it steps
        std::vector<float> actions; // each copy's actions for the control step being run
    };

} // namespace

extern "C" {

void *harvest_env_create(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves) {
    return harvest_env_create_for_movie(tree_seeds, tree_count, instances_per_tree, thread_count, seed, leaves, 0);
}

void *harvest_env_create_for_movie(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves, int show_arm) {
    try {
        auto environment = std::make_unique<Environment>();
        environment->simulations = std::make_unique<ParallelRobotSimulations>(static_cast<uint>(std::max(1, thread_count)));
        for (int t = 0; t < tree_count; t++) {
            environment->trees.push_back(std::make_unique<Tree>(tree_seeds[t], instances_per_tree, leaves != 0, show_arm != 0));
            Tree &tree = *environment->trees.back();
            environment->simulations->addVariant(&tree.engine, tree.control_loop.get(), {tree.tool_action.get(), tree.grip_action.get()});
            for (int i = 0; i < instances_per_tree; i++) {
                Slot slot;
                slot.tree = environment->trees.back().get();
                slot.instance = static_cast<uint>(i);
                slot.random.seed(static_cast<unsigned>(seed + 1000 * t + i));
                environment->slots.push_back(std::move(slot));
            }
        }
        return environment.release();
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return nullptr;
    }
}

void harvest_env_destroy(void *env) {
    delete static_cast<Environment *>(env);
}

int harvest_env_tree_has_reachable_fruit(int tree_seed) {
    Context context;
    context.seedRandomGenerator(tree_seed);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary(plant_model);
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), plant_age);
    USDExportParameters plant_physics;
    return findReachableFruit(plantarchitecture.getPlantArticulation(plantID, plant_physics)).empty() ? 0 : 1;
}

int harvest_env_export_tree(int tree_seed, const char *directory) {
    try {
        Context context;
        context.seedRandomGenerator(tree_seed);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary(plant_model);
        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), plant_age);
        plantarchitecture.removePlantLeaves(plantID);

        USDExportParameters plant_physics = plantPhysics();
        // PhysX needs this much armature to hold a tree up (doc/isaac_comparison_report.md); MuJoCo does not use it
        plant_physics.armature_stability_ratio = 1.f;
        plant_physics.physics_steps_per_second = 500.f;
        const std::filesystem::path base = std::filesystem::path(directory) / ("tree_" + std::to_string(tree_seed));
        plantarchitecture.writePlantStructureUSD(plantID, base.string() + ".usda", plant_physics);

        const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID, plant_physics);
        const std::vector<uint> candidates = chooseCandidateFruit(articulation, tree_seed);
        const std::vector<uint> flexible = chooseFlexibleInternodes(articulation, candidates);
        std::vector<char> is_flexible(articulation.links.size(), 0), is_candidate(articulation.links.size(), 0);
        for (uint link: flexible) {
            is_flexible[link] = 1;
        }
        for (uint link: candidates) {
            is_candidate[link] = 1;
        }

        std::ofstream json(base.string() + ".json");
        json << std::setprecision(9);
        json << "{\n\"seed\": " << tree_seed << ", \"plant_model\": \"" << plant_model << "\", \"plant_age\": " << plant_age << ",\n";
        // The task, as harvest_env.cpp sets it up
        json << "\"task\": {\"physics_timestep\": 0.002, \"physics_steps_per_control\": " << physics_steps_per_control << ", \"max_tool_speed\": " << max_tool_speed << ", \"max_target_lead\": " << max_target_lead << ", \"max_control_steps\": " << max_control_steps
             << ", \"grip_reach\": " << grip_reach << ", \"cup_lip_radius\": " << cup_lip_radius << ", \"cup_lip_thickness\": " << cup_lip_thickness << ", \"grip_stiffness\": " << grip_stiffness << ", \"grip_damping\": " << grip_damping << ", \"grip_ramp_seconds\": " << grip_ramp_seconds << ", \"grip_limit\": " << grip_limit
             << ", \"min_detach_force\": " << min_detach_force << ", \"max_detach_force\": " << max_detach_force << ", \"settling_physics_steps\": " << settling_physics_steps << ", \"protective_stop_force\": " << protective_stop_force
             << ", \"start_offset\": " << start_offset << ", \"robot_base\": [" << robot_base.x << ", " << robot_base.y << ", " << robot_base.z << "], \"cup_face\": [" << cup_face.x << ", " << cup_face.y << ", " << cup_face.z << "], \"tucked_pose\": [";
        for (size_t j = 0; j < tucked_pose.size(); j++) {
            json << (j ? ", " : "") << tucked_pose[j];
        }
        // The servos on the robot: stiffness, damping, force limit, armature, in N/m or N m/rad
        json << "],\n  \"servos\": {";
        bool first_servo = true;
        for (const JointServo &servo: joint_servos) {
            json << (first_servo ? "" : ", ") << "\"" << servo.joint << "\": [" << servo.stiffness << ", " << servo.damping << ", " << servo.force_limit << ", " << servo.armature << "]";
            first_servo = false;
        }
        json << "}},\n";

        // How each link's joint is simulated by addPlantToPhysicsEngine() with the parameters of Tree: "sprung" (a ball joint), "locked" (welded, or held by
        // the detachable weld of a candidate fruit), or "root". Leaves are removed, so rigid_leaves does not matter.
        json << "\"links\": [\n";
        for (size_t i = 0; i < articulation.links.size(); i++) {
            const PlantArticulationLink &link = articulation.links[i];
            std::string mode = "sprung";
            if (link.parent_index < 0) {
                mode = "root";
            } else if ((link.type == ARTICULATION_LINK_INTERNODE && !is_flexible[i]) || is_candidate[i] || link.type == ARTICULATION_LINK_PETIOLE || link.type == ARTICULATION_LINK_LEAF) {
                mode = "locked";
            }
            json << "  {\"name\": \"" << link.name << "\", \"type\": " << link.type << ", \"parent\": " << link.parent_index << ", \"joint\": \"" << mode << "\", \"mass\": " << link.mass << ", \"inertia\": [" << link.inertia_diagonal.x << ", " << link.inertia_diagonal.y << ", "
                 << link.inertia_diagonal.z << "], \"position\": [" << link.position.x << ", " << link.position.y << ", " << link.position.z << "]";
            if (link.type == ARTICULATION_LINK_FRUIT && link.source_objID != 0 && context.doesObjectExist(link.source_objID)) {
                // The sphere addPlantToPhysicsEngine() gives a fruit: the mean half extent of its bounding box in the link frame, at the box center
                vec3 lower = make_vec3(1e30f, 1e30f, 1e30f), upper = make_vec3(-1e30f, -1e30f, -1e30f);
                for (uint UUID: context.getObjectPrimitiveUUIDs(link.source_objID)) {
                    for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                        // Rotate into the link frame by the conjugate of its orientation: v + 2w(q x v) + 2q x (q x v), with q negated
                        const vec3 q = make_vec3(-link.qx, -link.qy, -link.qz);
                        const vec3 v = vertex - link.position;
                        const vec3 t = 2.f * cross(q, v);
                        const vec3 local = v + link.qw * t + cross(q, t);
                        lower = make_vec3(std::min(lower.x, local.x), std::min(lower.y, local.y), std::min(lower.z, local.z));
                        upper = make_vec3(std::max(upper.x, local.x), std::max(upper.y, local.y), std::max(upper.z, local.z));
                    }
                }
                const vec3 extent = upper - lower;
                const vec3 center = 0.5f * (lower + upper);
                json << ", \"sphere_radius\": " << (extent.x + extent.y + extent.z) / 6.f << ", \"sphere_center\": [" << center.x << ", " << center.y << ", " << center.z << "]";
            }
            json << "}" << (i + 1 < articulation.links.size() ? "," : "") << "\n";
        }
        json << "],\n\"candidates\": [";
        for (size_t c = 0; c < candidates.size(); c++) {
            const uint wood = findBearingInternode(articulation, candidates[c]);
            json << (c ? ", " : "") << "{\"link\": \"" << articulation.links[candidates[c]].name << "\", \"bearing_wood\": \"" << articulation.links[wood].name << "\"}";
        }
        // The wood where it stands at rest, which stands in for what perception would report
        json << "],\n\"wood\": [\n";
        bool first = true;
        for (const PlantArticulationLink &link: articulation.links) {
            if (link.type == ARTICULATION_LINK_INTERNODE) {
                const vec3 axis = make_vec3(2.f * (link.qx * link.qz + link.qw * link.qy), 2.f * (link.qy * link.qz - link.qw * link.qx), 1.f - 2.f * (link.qx * link.qx + link.qy * link.qy));
                const vec3 start = link.position - link.half_length * axis;
                const vec3 end = link.position + link.half_length * axis;
                json << (first ? "  " : ", ") << "[" << start.x << ", " << start.y << ", " << start.z << ", " << end.x << ", " << end.y << ", " << end.z << ", " << link.radius << "]";
                first = false;
            }
        }
        json << "\n]\n}\n";
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_count(const void *env) {
    return static_cast<int>(static_cast<const Environment *>(env)->slots.size());
}

int harvest_env_observation_size(void) {
    return observation_size;
}

int harvest_env_action_size(void) {
    return action_size;
}

int harvest_env_max_steps(void) {
    return max_control_steps;
}

int harvest_env_set_near_start_fraction(void *env, float fraction) {
    Environment &environment = *static_cast<Environment *>(env);
    try {
        if (!(fraction >= 0.f && fraction <= 1.f)) {
            helios_runtime_error("ERROR (harvest_env_set_near_start_fraction): The fraction must be from 0 to 1, but was " + std::to_string(fraction) + ".");
        }
        for (Slot &slot: environment.slots) {
            slot.near_start_fraction = fraction;
        }
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_enable_camera(void *env, int width, int height) {
    Environment &environment = *static_cast<Environment *>(env);
    try {
        for (const std::unique_ptr<Tree> &tree: environment.trees) {
            RobotCameraParameters wrist;
            wrist.field_of_view = wrist_camera_field_of_view;
            wrist.antialiasing_samples = 1;
            wrist.background = make_RGBcolor(0.6f, 0.75f, 0.95f);
            wrist.light_direction = make_vec3(0.3f, -0.5f, 0.8f);
            wrist.shadows = false;
            tree->camera = std::make_unique<RobotCamera>(&tree->engine, &tree->context, static_cast<uint>(width), static_cast<uint>(height), wrist);
            tree->engine.setContextUpdateTolerance(context_update_tolerance);
        }
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

void harvest_env_render(void *env, unsigned char *images) {
    Environment &environment = *static_cast<Environment *>(env);
    // Instances of a tree share its visualizer and so its framebuffer, so one instance of each tree is rendered at a time. The slots of a tree are next to each other.
    const size_t tree_count = environment.trees.size();
    const size_t instances_per_tree = environment.slots.size() / tree_count;
    for (size_t instance = 0; instance < instances_per_tree; instance++) {
        environment.simulations->forEachIndex(tree_count, [&](size_t tree) { prepareSlotFrame(environment.slots[tree * instances_per_tree + instance]); });
        // Reading the images back only after all are drawn lets the GPU work through them while the next is being drawn, rather than waiting for each one
        for (size_t tree = 0; tree < tree_count; tree++) {
            environment.slots[tree * instances_per_tree + instance].tree->camera->draw();
        }
        for (size_t tree = 0; tree < tree_count; tree++) {
            const size_t i = tree * instances_per_tree + instance;
            const RobotCamera &camera = *environment.slots[i].tree->camera;
            camera.getColorImage(images + i * size_t(camera.getWidth()) * size_t(camera.getHeight()) * 3);
        }
    }
}

void harvest_env_save_camera_image(void *env, int index, const char *filename) {
    Slot &slot = static_cast<Environment *>(env)->slots.at(index);
    renderSlot(slot);
    slot.tree->camera->saveImage(filename, "jpeg");
}

void harvest_env_measure_camera_view(void *env, int index, HarvestCameraView *view) {
    Slot &slot = static_cast<Environment *>(env)->slots.at(index);
    Tree &tree = *slot.tree;
    renderSlot(slot);
    const size_t pixel_count = size_t(tree.camera->getWidth()) * size_t(tree.camera->getHeight());
    std::vector<uint> pixel_UUIDs(pixel_count);
    std::vector<float> depth(pixel_count);
    tree.camera->getPrimitiveImage(pixel_UUIDs.data());
    tree.camera->getDepthImage(depth.data());
    *view = HarvestCameraView{};
    const uint target_label = slot.target->objID + organ_fruit_base;
    for (size_t k = 0; k < pixel_UUIDs.size(); k++) {
        if (pixel_UUIDs[k] == 0) {
            continue;
        }
        const auto organ = tree.organ_of_primitive.find(pixel_UUIDs[k] - 1);
        if (organ == tree.organ_of_primitive.end()) {
            continue;
        }
        if (organ->second == target_label) {
            view->target_fruit += 1.f;
        } else if (organ->second >= organ_fruit_base) {
            view->other_fruit += 1.f;
        } else if (organ->second == organ_leaf) {
            view->leaves += 1.f;
            if (depth[k] < 0.15f) {
                view->near_leaves += 1.f;
            }
        } else if (organ->second == organ_wood) {
            view->wood += 1.f;
        }
    }
    const float total = float(pixel_UUIDs.size());
    view->target_fruit /= total;
    view->other_fruit /= total;
    view->leaves /= total;
    view->near_leaves /= total;
    view->wood /= total;
}

int harvest_env_link_count(const void *env, int index) {
    return static_cast<int>(static_cast<const Environment *>(env)->slots.at(index).tree->link_names.size());
}

const char *harvest_env_link_name(const void *env, int index, int link) {
    return static_cast<const Environment *>(env)->slots.at(index).tree->link_names.at(link).c_str();
}

void harvest_env_rest_link_poses(const void *env, int index, float *link_poses) {
    const Tree &tree = *static_cast<const Environment *>(env)->slots.at(index).tree;
    for (size_t i = 0; i < tree.link_rest_poses.size(); i++) {
        for (int k = 0; k < 7; k++) {
            link_poses[7 * i + k] = float(tree.link_rest_poses[i][k]);
        }
    }
}

int harvest_env_display_state(const void *env, int index, int last_episode, float *link_poses, float *robot_joints) {
    try {
        const Slot &slot = static_cast<const Environment *>(env)->slots.at(index);
        std::vector<float> poses;
        std::vector<float> joints;
        if (last_episode) {
            if (slot.last_link_poses.empty()) {
                helios_runtime_error("ERROR (harvest_env_display_state): Simulation " + std::to_string(index) + " has not finished an episode yet.");
            }
            poses = slot.last_link_poses;
            joints = slot.last_robot_joints;
        } else {
            captureDisplayState(slot, poses, joints);
        }
        std::copy(poses.begin(), poses.end(), link_poses);
        std::copy(joints.begin(), joints.end(), robot_joints);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

void harvest_env_target_fruit(const void *env, int index, float *position) {
    const Slot &slot = static_cast<const Environment *>(env)->slots.at(index);
    position[0] = slot.target->settled_position.x;
    position[1] = slot.target->settled_position.y;
    position[2] = slot.target->settled_position.z;
}

int harvest_env_enable_movie_camera(void *env, int width, int height) {
    Environment &environment = *static_cast<Environment *>(env);
    try {
        for (const std::unique_ptr<Tree> &tree: environment.trees) {
            if (tree->camera) {
                helios_runtime_error("ERROR (harvest_env_enable_movie_camera): The wrist camera is enabled. Both cameras clear the Context's geometry flags after each frame, so each would miss motion the other drew.");
            }
            RobotCameraParameters movie;
            movie.field_of_view = movie_camera_field_of_view;
            movie.antialiasing_samples = 8;
            movie.background = make_RGBcolor(0.72f, 0.83f, 0.95f);
            movie.light_direction = make_vec3(-0.35f, -0.45f, 0.82f);
            movie.shadows = true;
            tree->movie_camera = std::make_unique<RobotCamera>(&tree->engine, &tree->context, static_cast<uint>(width), static_cast<uint>(height), movie);
            tree->engine.setContextUpdateTolerance(1e-5f);
        }
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_render_movie_frame(void *env, int index, const float *link_poses, const float *robot_joints, const float *eye, const float *look_at, unsigned char *image) {
    try {
        renderMovieFrame(static_cast<Environment *>(env)->slots.at(index), link_poses, robot_joints, make_vec3(eye[0], eye[1], eye[2]), make_vec3(look_at[0], look_at[1], look_at[2]), false, image);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_render_wrist_frame(void *env, int index, const float *link_poses, const float *robot_joints, unsigned char *image) {
    try {
        renderMovieFrame(static_cast<Environment *>(env)->slots.at(index), link_poses, robot_joints, make_vec3(0, 0, 0), make_vec3(0, 0, 0), true, image);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

void harvest_env_set_follow_through(void *env, int control_steps) {
    for (Slot &slot: static_cast<Environment *>(env)->slots) {
        slot.follow_through_steps = std::max(0, control_steps);
    }
}

void harvest_env_last_episode(const void *env, int index, HarvestEpisodeRecord *record) {
    *record = static_cast<const Environment *>(env)->slots.at(index).last;
}

void harvest_env_reset(void *env, float *observations) {
    Environment &environment = *static_cast<Environment *>(env);
    environment.simulations->forEachCopy([&](uint i, uint, uint) {
        resetSlot(environment.slots[i]);
        writeObservation(environment.slots[i], observations + size_t(i) * observation_size);
    });
}

void harvest_env_step(void *env, const float *actions, float *observations, float *rewards, int *outcomes, float *terminal_observations) {
    Environment &environment = *static_cast<Environment *>(env);
    const size_t count = environment.slots.size();
    environment.actions.resize(count * action_size);
    for (size_t i = 0; i < count; i++) {
        Slot &slot = environment.slots[i];
        const float *action = slot.follow_through_remaining > 0 ? follow_through_action : actions + i * action_size;
        std::copy(action, action + action_size, environment.actions.begin() + i * action_size);
        slot.gripped_before_step = slot.tree->cup->isGripped(slot.instance);
    }
    environment.simulations->step(environment.actions.data());
    environment.simulations->forEachCopy([&](uint i, uint, uint) { finishStep(environment.slots[i], observations + size_t(i) * observation_size, rewards[i], outcomes[i], terminal_observations + size_t(i) * observation_size); });
}

} // extern "C"
