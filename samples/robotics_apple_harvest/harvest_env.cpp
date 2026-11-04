// Apple harvesting environment. See harvest_env.h.
//
// Each tree has its own compiled model, since no two trees have the same structure, and several simulation instances of it.
// Only the wood bearing the candidate fruit is flexible, which keeps almost all of the speed of rigid wood.

#include "harvest_env.h"

#include "PlantPhysics.h"
#include "Robot.h"
#include "Visualizer.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {

    // Control
    const uint physics_steps_per_control = 20; // 25 Hz at a 2 ms physics step
    const float max_tool_speed = 0.3f; // m/s
    const float max_target_lead = 0.08f; // m the commanded tool position can lead the actual one, so the target cannot wind up while the arm is blocked
    const int max_control_steps = 250; // 10 s

    // Gripper: an ideal vacuum cup. It grips a fruit its face touches, where it touches it, and then holds the fruit on its face: springs hold the fruit's
    // center where it was when the seal formed (moved onto the face) and its orientation as it was then, both relative to the cup, reacting on the arm.
    // The seal breaks when the fruit pulls away from the face, slides along it, or twists or peels it, too hard. The limits on sliding and peeling were
    // chosen for this task: suction cups hold less sideways than straight out, and least of all against peeling.
    const float grip_reach = 0.005f; // m between the cup's face and the fruit's surface, about how far a compliant lip reaches to seal on contact
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
    const vec3 cup_face = make_vec3(0.32f, 0, 0); // in the tool's frame
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

    const int observation_size = 23;
    const int action_size = 4;

    struct Candidate {
        uint link;
        uint bearing_wood_link; // the nearest internode toward the trunk
        uint objID;
        int body_ID;
        float radius;
        vec3 settled_position;
        vec3 settled_wood_position;
    };

    // Labels for measuring what the camera sees. A fruit is labeled by its object, offset past the others.
    const uint organ_other = 0;
    const uint organ_leaf = 1;
    const uint organ_wood = 2;
    const uint organ_fruit_base = 3;

    struct WoodSegment {
        vec3 start;
        vec3 end;
        float radius;
    };

    USDExportParameters plantPhysics() {
        USDExportParameters plant_physics;
        plant_physics.elastic_modulus = 5e9f;
        plant_physics.wood_density = 750.f;
        plant_physics.fruit_mass = 0.15f;
        return plant_physics;
    }

    // Reachable fruit, nearest the robot first, since those are the ones not hidden behind the wall
    std::vector<uint> chooseCandidateFruit(const PlantArticulation &articulation, int seed) {
        std::vector<uint> reachable;
        for (uint i = 0; i < articulation.links.size(); i++) {
            const PlantArticulationLink &link = articulation.links[i];
            if (link.type == ARTICULATION_LINK_FRUIT && link.position.z >= min_fruit_height && link.position.z <= max_fruit_height) {
                reachable.push_back(i);
            }
        }
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
            int internodes = 0;
            for (int link = articulation.links[fruit].parent_index; link >= 0 && internodes < flexible_internodes_per_candidate; link = articulation.links[link].parent_index) {
                if (articulation.links[link].type == ARTICULATION_LINK_INTERNODE && articulation.links[link].parent_index >= 0) {
                    flexible.push_back(static_cast<uint>(link));
                    internodes++;
                }
            }
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
        std::vector<mjtNum> settled_qpos;
        std::vector<WoodSegment> wood; // where the wood stands at rest, which stands in for what perception would report
        std::unique_ptr<Visualizer> camera;
        std::unordered_map<uint, uint> organ_of_primitive; // for measuring what the camera sees: which kind of plant organ each primitive belongs to, or which fruit

        // For showing a simulation from outside (movies): the links of the plant, parents first, with the MuJoCo body of each (-1 if it has none) and its
        // rest pose (x, y, z, qw, qx, qy, qz), which is also the rest pose of its body
        std::vector<std::string> link_names;
        std::vector<int> link_parents;
        std::vector<int> link_bodies;
        std::vector<std::array<mjtNum, 7>> link_rest_poses;
        std::unique_ptr<Visualizer> movie_camera;

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

            for (const PlantArticulationLink &link: articulation.links) {
                if (link.type == ARTICULATION_LINK_INTERNODE) {
                    // The local +z axis of the link is the capsule axis, the third column of its rotation matrix
                    const vec3 axis = make_vec3(2.f * (link.qx * link.qz + link.qw * link.qy), 2.f * (link.qy * link.qz - link.qw * link.qx), 1.f - 2.f * (link.qx * link.qx + link.qy * link.qy));
                    wood.push_back({link.position - link.half_length * axis, link.position + link.half_length * axis, link.radius});
                }
            }

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
            engine.bindObjectToBody(context.addConeObject(16, wrist, wrist + make_vec3(0.28f, 0, 0), 0.03f, 0.018f, make_RGBcolor(0.9f, 0.4f, 0.08f)), "robot/tool");
            engine.bindObjectToBody(context.addSphereObject(16, wrist + make_vec3(0.3f, 0, 0), 0.02f, make_RGBcolor(0.15f, 0.15f, 0.15f)), "robot/tool");
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
            engine.initialize(parameters);
            robot->initialize();

            while (engine.getInstanceCount() < static_cast<uint>(instance_count)) {
                engine.createInstance();
            }
            robot->prepareInstances();

            const mjModel *model = engine.getMuJoCoModel();
            tool_body_ID = mj_name2id(model, mjOBJ_BODY, "robot/tool");

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
            const mjData *data = engine.getMuJoCoData(0);
            settled_qpos.assign(data->qpos, data->qpos + model->nq);

            for (uint link: reachable) {
                Candidate candidate;
                candidate.link = link;
                candidate.objID = articulation.links[link].source_objID;
                int wood = articulation.links[link].parent_index;
                while (wood >= 0 && articulation.links[wood].type != ARTICULATION_LINK_INTERNODE) {
                    wood = articulation.links[wood].parent_index;
                }
                if (wood < 0) {
                    helios_runtime_error("ERROR (harvest_env): Fruit " + articulation.links[link].name + " is not borne on an internode.");
                }
                candidate.bearing_wood_link = static_cast<uint>(wood);
                candidate.settled_wood_position = engine.getLinkPosition(plant, candidate.bearing_wood_link, 0);
                candidate.body_ID = mj_name2id(model, mjOBJ_BODY, ("p" + std::to_string(plantID) + "_" + articulation.links[link].name).c_str());
                if (candidate.body_ID < 0 || model->body_geomnum[candidate.body_ID] < 1) {
                    helios_runtime_error("ERROR (harvest_env): Fruit " + articulation.links[link].name + " has no collision shape in the compiled model.");
                }
                candidate.radius = float(model->geom_size[3 * model->body_geomadr[candidate.body_ID]]);
                candidate.settled_position = engine.getLinkPosition(plant, link, 0);
                candidates.push_back(candidate);
            }
        }
    };

    struct Slot {
        Tree *tree = nullptr;
        uint instance = 0;
        std::mt19937 random;
        const Candidate *target = nullptr;
        float detach_force = 0.f;
        bool gripped = false;
        // Where the cup holds the fruit, in the tool body's frame: the fruit's center, the direction from the cup's face to it, and its orientation
        mjtNum grip_center[3] = {0, 0, 0};
        mjtNum grip_normal[3] = {0, 0, 0};
        mjtNum grip_orientation[4] = {1, 0, 0, 0};
        float grip_shear = 0.f; // N, the load along the face in the last physics step
        float grip_moment = 0.f; // N m, the load's moment about the face's center in the last physics step
        float grip_seconds = 0.f;
        float grip_force = 0.f;
        float previous_load = 0.f;
        float previous_distance = 0.f;
        vec3 tool_target;
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
        HarvestEpisodeRecord detached_record{}; // the record of an episode that goes on, as it was when the fruit detached
    };

    vec3 toVec3(const mjtNum *p) {
        return make_vec3(float(p[0]), float(p[1]), float(p[2]));
    }

    float surfaceDistance(const Slot &slot) {
        const mjData *data = slot.tree->engine.getMuJoCoData(slot.instance);
        return (toVec3(data->xipos + 3 * slot.target->body_ID) - slot.tree->robot->getToolPosition(slot.instance)).magnitude() - slot.target->radius;
    }

    // The load on the cup's seal in the last physics step, as a fraction of what breaks it: the largest of the pull, the load along the face and the moment,
    // each over its limit
    float sealLoad(const Slot &slot) {
        return std::max({slot.grip_force / grip_limit, slot.grip_shear / grip_shear_limit, slot.grip_moment / grip_moment_limit});
    }

    bool sealBroken(const Slot &slot) {
        return sealLoad(slot) > 1.f;
    }

    // From a point to the nearest surface of the wood
    vec3 towardNearestWood(const Tree &tree, const vec3 &point) {
        vec3 nearest_offset = make_vec3(0, 0, 0);
        float nearest_distance = std::numeric_limits<float>::max();
        for (const WoodSegment &segment: tree.wood) {
            const vec3 along = segment.end - segment.start;
            const float length_squared = along * along;
            const float t = length_squared > 0.f ? std::clamp((point - segment.start) * along / length_squared, 0.f, 1.f) : 0.f;
            const vec3 offset = segment.start + t * along - point;
            const float distance = offset.magnitude() - segment.radius;
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest_offset = offset;
            }
        }
        return nearest_offset;
    }

    // A camera on the wrist, behind and above the cup, looking along the tool past the cup
    void wristCameraPose(const Robot &robot, uint instance, vec3 &eye, vec3 &look_at) {
        const vec3 cup = robot.getToolPosition(instance);
        const vec3 axis = robot.getToolAxis(instance);
        eye = cup - 0.3f * axis + make_vec3(0, 0, 0.08f);
        look_at = cup + 0.3f * axis;
    }

    void pointCamera(Slot &slot) {
        vec3 eye, look_at;
        wristCameraPose(*slot.tree->robot, slot.instance, eye, look_at);
        slot.tree->camera->setCameraPosition(eye, look_at);
    }

    const float wrist_camera_field_of_view = 70.f;
    const float movie_camera_field_of_view = 40.f;

    // The part of rendering that needs no OpenGL, so that trees can do it in parallel
    void prepareSlotFrame(Slot &slot) {
        Tree &tree = *slot.tree;
        tree.engine.updateContextGeometry(slot.instance);
        pointCamera(slot);
        tree.camera->prepareContextGeometry();
        // Flags cleared after each frame report only what moves between frames, so the visualizer does not check every primitive
        tree.context.markGeometryClean();
    }

    void renderSlot(Slot &slot) {
        prepareSlotFrame(slot);
        slot.tree->camera->plotUpdate(true);
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

        tree.engine.updateContextGeometry(slot.instance);
        tree.movie_camera->prepareContextGeometry();
        tree.context.markGeometryClean();
        if (wrist_view) {
            vec3 wrist_eye, wrist_look_at;
            wristCameraPose(*tree.robot, slot.instance, wrist_eye, wrist_look_at);
            tree.movie_camera->setCameraFieldOfView(wrist_camera_field_of_view);
            tree.movie_camera->setCameraPosition(wrist_eye, wrist_look_at);
        } else {
            tree.movie_camera->setCameraFieldOfView(movie_camera_field_of_view);
            tree.movie_camera->setCameraPosition(eye, look_at);
        }
        tree.movie_camera->plotUpdate(true);
        std::vector<uint> pixels;
        uint width = 0;
        uint height = 0;
        tree.movie_camera->getWindowPixelsRGB(pixels, width, height);
        // OpenGL gives the bottom row first
        for (uint row = 0; row < height; row++) {
            const uint *source = pixels.data() + size_t(height - 1 - row) * size_t(width) * 3;
            unsigned char *destination = image + size_t(row) * size_t(width) * 3;
            for (uint k = 0; k < width * 3; k++) {
                destination[k] = static_cast<unsigned char>(source[k]);
            }
        }
    }

    void writeObservation(const Slot &slot, float *observation) {
        Robot &robot = *slot.tree->robot;
        const mjModel *model = slot.tree->engine.getMuJoCoModel();
        const mjData *data = slot.tree->engine.getMuJoCoData(slot.instance);
        const vec3 tool = robot.getToolPosition(slot.instance);
        const vec3 fruit = toVec3(data->xipos + 3 * slot.target->body_ID);
        const vec3 relative = fruit - tool;
        const std::vector<float> joints = robot.getJointPositions(slot.instance);
        mjtNum velocity[6];
        mj_objectVelocity(model, data, mjOBJ_BODY, slot.tree->tool_body_ID, velocity, 0);
        const float contact = slot.tree->engine.getExternalContactForce(slot.tree->plant, slot.instance).magnitude();
        const vec3 wood = towardNearestWood(*slot.tree, tool);

        int i = 0;
        observation[i++] = relative.x;
        observation[i++] = relative.y;
        observation[i++] = relative.z;
        observation[i++] = surfaceDistance(slot);
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
        observation[i++] = slot.gripped ? 1.f : 0.f;
        observation[i++] = sealLoad(slot); // how near the seal is to breaking
        observation[i++] = contact / 10.f;
        observation[i++] = float(slot.control_step) / float(max_control_steps);
        observation[i++] = wood.x;
        observation[i++] = wood.y;
        observation[i++] = wood.z;
    }

    void resetSlot(Slot &slot) {
        Tree &tree = *slot.tree;
        PhysicsEngine &engine = tree.engine;
        engine.resetInstance(slot.instance);

        std::uniform_int_distribution<size_t> pick_fruit(0, tree.candidates.size() - 1);
        std::uniform_real_distribution<float> unit(0.f, 1.f);
        slot.target = &tree.candidates[pick_fruit(slot.random)];
        slot.detach_force = min_detach_force + unit(slot.random) * (max_detach_force - min_detach_force);

        mjData *data = engine.getMuJoCoData(slot.instance);
        mju_copy(data->qpos, tree.settled_qpos.data(), engine.getMuJoCoModel()->nq);
        std::vector<float> start_pose = tucked_pose;
        start_pose[0] = slot.target->settled_position.y + (2.f * unit(slot.random) - 1.f) * start_offset;
        tree.robot->setJointPositions(start_pose, slot.instance);

        slot.gripped = false;
        slot.grip_seconds = 0.f;
        slot.grip_force = 0.f;
        slot.follow_through_remaining = 0;
        slot.previous_load = 0.f;
        slot.control_step = 0;
        slot.tool_target = tree.robot->getToolPosition(slot.instance);
        slot.previous_distance = surfaceDistance(slot);
        slot.pull_direction = make_vec3(0, 0, 0);
        slot.current = HarvestEpisodeRecord{};
        slot.current.detach_force = slot.detach_force;
        slot.current.steps_to_grip = -1;
    }

    // Seal the cup on the fruit where they touch: the fruit is to be held with its surface on the face, along the line from the face to its center
    void startGrip(Slot &slot) {
        Tree &tree = *slot.tree;
        const mjData *data = tree.engine.getMuJoCoData(slot.instance);
        const int fruit_body = slot.target->body_ID;
        const int tool_body = tree.tool_body_ID;
        const vec3 face = tree.robot->getToolPosition(slot.instance);
        vec3 normal = toVec3(data->xipos + 3 * fruit_body) - face;
        normal.normalize();
        const vec3 center = face + slot.target->radius * normal;
        const mjtNum center_from_tool[3] = {center.x - data->xpos[3 * tool_body], center.y - data->xpos[3 * tool_body + 1], center.z - data->xpos[3 * tool_body + 2]};
        const mjtNum normal_world[3] = {normal.x, normal.y, normal.z};
        mju_mulMatTVec3(slot.grip_center, data->xmat + 9 * tool_body, center_from_tool);
        mju_mulMatTVec3(slot.grip_normal, data->xmat + 9 * tool_body, normal_world);
        mjtNum tool_inverse[4];
        mju_negQuat(tool_inverse, data->xquat + 4 * tool_body);
        mju_mulQuat(slot.grip_orientation, tool_inverse, data->xquat + 4 * fruit_body);
        slot.gripped = true;
        slot.grip_seconds = 0.f;
    }

    // The force and torque of the grip on the fruit, applied to the fruit and, in reaction, to the arm. The rotational springs have the time constants of the
    // linear ones for this fruit's inertia.
    void applyGrip(Slot &slot, float dt) {
        Tree &tree = *slot.tree;
        const mjModel *model = tree.engine.getMuJoCoModel();
        mjData *data = tree.engine.getMuJoCoData(slot.instance);
        const int fruit_body = slot.target->body_ID;
        const int tool_body = tree.tool_body_ID;
        mju_zero(data->xfrc_applied + 6 * fruit_body, 6);
        mju_zero(data->xfrc_applied + 6 * tool_body, 6);
        slot.grip_force = 0.f;
        slot.grip_shear = 0.f;
        slot.grip_moment = 0.f;
        if (!slot.gripped) {
            return;
        }
        slot.grip_seconds += dt;
        const float ramp = std::min(1.f, slot.grip_seconds / grip_ramp_seconds);

        // Where the fruit should be, and how fast that point moves with the tool
        mjtNum offset[3], normal_world[3];
        mju_mulMatVec3(offset, data->xmat + 9 * tool_body, slot.grip_center);
        mju_mulMatVec3(normal_world, data->xmat + 9 * tool_body, slot.grip_normal);
        const vec3 held = toVec3(data->xpos + 3 * tool_body) + toVec3(offset);
        const vec3 normal = toVec3(normal_world);
        mjtNum velocity_fruit[6], velocity_tool[6];
        mj_objectVelocity(model, data, mjOBJ_BODY, fruit_body, velocity_fruit, 0);
        mj_objectVelocity(model, data, mjOBJ_BODY, tool_body, velocity_tool, 0);
        const vec3 tool_center = toVec3(data->xipos + 3 * tool_body);
        const vec3 tool_spin = toVec3(velocity_tool);
        const vec3 held_velocity = toVec3(velocity_tool + 3) + cross(tool_spin, held - tool_center);
        const vec3 fruit = toVec3(data->xipos + 3 * fruit_body);
        const vec3 on_fruit = ramp * (grip_stiffness * (held - fruit) + grip_damping * (held_velocity - toVec3(velocity_fruit + 3)));

        // The orientation the fruit should have, and the rotation from where it is to there
        mjtNum held_orientation[4], fruit_inverse[4], error[4], error_rotation[3];
        mju_mulQuat(held_orientation, data->xquat + 4 * tool_body, slot.grip_orientation);
        mju_negQuat(fruit_inverse, data->xquat + 4 * fruit_body);
        mju_mulQuat(error, held_orientation, fruit_inverse);
        mju_quat2Vel(error_rotation, error, 1.0);
        const float mass = float(model->body_mass[fruit_body]);
        const float inertia = float(model->body_inertia[3 * fruit_body] + model->body_inertia[3 * fruit_body + 1] + model->body_inertia[3 * fruit_body + 2]) / 3.f;
        const vec3 torque_on_fruit = ramp * (inertia / mass) * (grip_stiffness * toVec3(error_rotation) + grip_damping * (tool_spin - toVec3(velocity_fruit)));

        // The load on the seal: pulling the fruit toward the face is a pull out of it, the rest is along it, and the moment is about the face's center
        const float pull = std::max(0.f, -(on_fruit * normal));
        const vec3 along_face = on_fruit - (on_fruit * normal) * normal;
        const vec3 face_center = held - slot.target->radius * normal;
        slot.grip_force = pull;
        slot.grip_shear = along_face.magnitude();
        slot.grip_moment = (torque_on_fruit + cross(held - face_center, on_fruit)).magnitude();
        if (on_fruit.magnitude() > 1e-6f) {
            slot.pull_direction = on_fruit / on_fruit.magnitude();
        }

        const float fruit_load[6] = {on_fruit.x, on_fruit.y, on_fruit.z, torque_on_fruit.x, torque_on_fruit.y, torque_on_fruit.z};
        // The reaction acts where the fruit is held rather than at the tool's center of mass, so it also carries a torque
        const vec3 on_tool = -1.f * on_fruit;
        const vec3 torque_on_tool = cross(held - tool_center, on_tool) - torque_on_fruit;
        const float tool_load[6] = {on_tool.x, on_tool.y, on_tool.z, torque_on_tool.x, torque_on_tool.y, torque_on_tool.z};
        for (int k = 0; k < 6; k++) {
            data->xfrc_applied[6 * fruit_body + k] = fruit_load[k];
            data->xfrc_applied[6 * tool_body + k] = tool_load[k];
        }
    }

    // A control step after the fruit detached, with the cup backing straight away from the tree whatever the action, carrying the fruit for as long as the
    // grip holds. The episode is a success either way, and ends when the steps run out, with the record it had when the fruit detached.
    void stepFollowThrough(Slot &slot, float *observation, float &reward, int &outcome, float *terminal_observation) {
        Tree &tree = *slot.tree;
        Robot &robot = *tree.robot;
        const float dt = float(physics_steps_per_control) * tree.engine.getTimestep();
        const vec3 tool = robot.getToolPosition(slot.instance);
        slot.tool_target = slot.tool_target + dt * max_tool_speed * make_vec3(-1, 0, 0);
        vec3 lead = slot.tool_target - tool;
        if (lead.magnitude() > max_target_lead) {
            slot.tool_target = tool + lead / lead.magnitude() * max_target_lead;
        }
        const std::vector<float> joints = robot.getJointPositions(slot.instance);
        vec3 pointing = slot.tool_target - make_vec3(robot_base.x + 0.15f, joints[0], slot.tool_target.z);
        pointing.z = 0.f;
        if (pointing.magnitude() > 0.05f) {
            pointing.normalize();
            RobotInverseKinematicsResult pose = robot.solveInverseKinematics(slot.tool_target, pointing, {"roll"}, slot.instance);
            if (pose.converged) {
                robot.setJointTargets(pose.joint_positions, slot.instance);
            }
        }
        for (uint s = 0; s < physics_steps_per_control; s++) {
            applyGrip(slot, tree.engine.getTimestep());
            tree.engine.step(1, slot.instance);
            if (sealBroken(slot)) {
                slot.gripped = false; // the fruit falls
            }
        }
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

    void stepSlot(Slot &slot, const float *action, float *observation, float &reward, int &outcome, float *terminal_observation) {
        if (slot.follow_through_remaining > 0) {
            stepFollowThrough(slot, observation, reward, outcome, terminal_observation);
            return;
        }
        Tree &tree = *slot.tree;
        PhysicsEngine &engine = tree.engine;
        Robot &robot = *tree.robot;
        const float dt = float(physics_steps_per_control) * engine.getTimestep();

        // Move the tool target, and solve for the arm with the cup pointing away from the robot's turret
        const vec3 velocity = max_tool_speed * make_vec3(std::clamp(action[0], -1.f, 1.f), std::clamp(action[1], -1.f, 1.f), std::clamp(action[2], -1.f, 1.f));
        const vec3 tool = robot.getToolPosition(slot.instance);
        slot.tool_target = slot.tool_target + dt * velocity;
        vec3 lead = slot.tool_target - tool;
        if (lead.magnitude() > max_target_lead) {
            lead = lead / lead.magnitude() * max_target_lead;
            slot.tool_target = tool + lead;
        }
        const std::vector<float> joints = robot.getJointPositions(slot.instance);
        vec3 pointing = slot.tool_target - make_vec3(robot_base.x + 0.15f, joints[0], slot.tool_target.z);
        pointing.z = 0.f;
        if (pointing.magnitude() > 0.05f) {
            pointing.normalize();
            RobotInverseKinematicsResult pose = robot.solveInverseKinematics(slot.tool_target, pointing, {"roll"}, slot.instance);
            // An unreachable target leaves the arm where it was heading
            if (pose.converged) {
                robot.setJointTargets(pose.joint_positions, slot.instance);
            }
        }

        const float distance_before_grip = surfaceDistance(slot);
        const bool gripped_now = !slot.gripped && action[3] > 0.f && distance_before_grip < grip_reach;
        if (gripped_now) {
            startGrip(slot);
            slot.current.steps_to_grip = slot.control_step;
        }

        // Simulate, with the stem failing when the load on it passes the force for this fruit
        outcome = HARVEST_RUNNING;
        float load = 0.f;
        float physics_dt = engine.getTimestep();
        float peak_contact = 0.f;
        for (uint s = 0; s < physics_steps_per_control && outcome == HARVEST_RUNNING; s++) {
            applyGrip(slot, physics_dt);
            engine.step(1, slot.instance);
            load = engine.getDetachableLinkLoad(tree.plant, slot.target->link, slot.instance);
            if (load > slot.detach_force) {
                engine.detachLink(tree.plant, slot.target->link, slot.instance);
                outcome = slot.gripped ? HARVEST_SUCCESS : HARVEST_KNOCKED_OFF;
            } else if (sealBroken(slot)) {
                outcome = HARVEST_SLIPPED;
            } else if (engine.getExternalContactForce(tree.plant, slot.instance).magnitude() > protective_stop_force) {
                outcome = HARVEST_COLLISION;
            }
            peak_contact = std::max(peak_contact, engine.getExternalContactForce(tree.plant, slot.instance).magnitude());
        }
        slot.control_step++;
        if (outcome == HARVEST_RUNNING && slot.control_step >= max_control_steps) {
            outcome = HARVEST_TIMEOUT;
        }

        // Reward: approach the fruit, grip it, load the stem, and detach it, without pushing on the tree
        const float distance = surfaceDistance(slot);
        reward = -0.005f;
        if (!slot.gripped) {
            reward += 5.f * (slot.previous_distance - distance);
        }
        if (gripped_now) {
            reward += 1.f;
        }
        if (slot.gripped) {
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
        slot.current.final_grip_force = slot.grip_force;
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
        std::vector<Slot> slots;
        int thread_count = 1;
    };

    // Run a function for every index up to a count, each on one thread at a time
    template<typename Function>
    void forEachIndex(const Environment &environment, size_t count, Function function) {
        std::atomic<size_t> next{0};
        std::exception_ptr first_failure;
        std::mutex failure_mutex;
        auto worker = [&]() {
            for (size_t i = next++; i < count; i = next++) {
                try {
                    function(i);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if (!first_failure) {
                        first_failure = std::current_exception();
                    }
                }
            }
        };
        std::vector<std::thread> threads;
        for (int t = 1; t < environment.thread_count; t++) {
            threads.emplace_back(worker);
        }
        worker();
        for (std::thread &thread: threads) {
            thread.join();
        }
        if (first_failure) {
            std::rethrow_exception(first_failure);
        }
    }

    template<typename Function>
    void forEachSlot(Environment &environment, Function function) {
        forEachIndex(environment, environment.slots.size(), function);
    }

} // namespace

extern "C" {

void *harvest_env_create(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves) {
    return harvest_env_create_for_movie(tree_seeds, tree_count, instances_per_tree, thread_count, seed, leaves, 0);
}

void *harvest_env_create_for_movie(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves, int show_arm) {
    try {
        auto environment = std::make_unique<Environment>();
        environment->thread_count = std::max(1, thread_count);
        for (int t = 0; t < tree_count; t++) {
            environment->trees.push_back(std::make_unique<Tree>(tree_seeds[t], instances_per_tree, leaves != 0, show_arm != 0));
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
    for (const PlantArticulationLink &link: plantarchitecture.getPlantArticulation(plantID, plant_physics).links) {
        if (link.type == ARTICULATION_LINK_FRUIT && link.position.z >= min_fruit_height && link.position.z <= max_fruit_height) {
            return 1;
        }
    }
    return 0;
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
             << ", \"grip_reach\": " << grip_reach << ", \"grip_stiffness\": " << grip_stiffness << ", \"grip_damping\": " << grip_damping << ", \"grip_ramp_seconds\": " << grip_ramp_seconds << ", \"grip_limit\": " << grip_limit
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
            int wood = articulation.links[candidates[c]].parent_index;
            while (wood >= 0 && articulation.links[wood].type != ARTICULATION_LINK_INTERNODE) {
                wood = articulation.links[wood].parent_index;
            }
            if (wood < 0) {
                helios_runtime_error("ERROR (harvest_env_export_tree): Fruit " + articulation.links[candidates[c]].name + " is not borne on an internode.");
            }
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

int harvest_env_enable_camera(void *env, int width, int height) {
    Environment &environment = *static_cast<Environment *>(env);
    try {
        for (const std::unique_ptr<Tree> &tree: environment.trees) {
            tree->camera = std::make_unique<Visualizer>(static_cast<uint>(width), static_cast<uint>(height), 1, false, true);
            tree->camera->disableMessages();
            tree->camera->hideWatermark();
            tree->camera->setBackgroundColor(make_RGBcolor(0.6f, 0.75f, 0.95f));
            tree->camera->setLightingModel(Visualizer::LIGHTING_PHONG);
            tree->camera->setLightDirection(make_vec3(0.3f, -0.5f, 0.8f));
            tree->camera->setCameraFieldOfView(wrist_camera_field_of_view);
            tree->camera->hideNavigationGizmo(); // the gizmo is rebuilt whenever the camera moves, which is every frame on a wrist
            tree->camera->buildContextGeometry(&tree->context);
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
    std::vector<uint> pixels;
    // Instances of a tree share its visualizer and so its framebuffer, so one instance of each tree is rendered at a time. The slots of a tree are next to each other.
    const size_t tree_count = environment.trees.size();
    const size_t instances_per_tree = environment.slots.size() / tree_count;
    for (size_t instance = 0; instance < instances_per_tree; instance++) {
        forEachIndex(environment, tree_count, [&](size_t tree) { prepareSlotFrame(environment.slots[tree * instances_per_tree + instance]); });
        // Reading the images back only after all are drawn lets the GPU work through them while the next is being drawn, rather than waiting for each one
        for (size_t tree = 0; tree < tree_count; tree++) {
            environment.slots[tree * instances_per_tree + instance].tree->camera->plotUpdate(true);
        }
        for (size_t tree = 0; tree < tree_count; tree++) {
            const size_t i = tree * instances_per_tree + instance;
            uint width = 0;
            uint height = 0;
            environment.slots[i].tree->camera->getWindowPixelsRGB(pixels, width, height);
            // OpenGL gives the bottom row first
            unsigned char *image = images + i * size_t(width) * size_t(height) * 3;
            for (uint row = 0; row < height; row++) {
                const uint *source = pixels.data() + size_t(height - 1 - row) * size_t(width) * 3;
                unsigned char *destination = image + size_t(row) * size_t(width) * 3;
                for (uint k = 0; k < width * 3; k++) {
                    destination[k] = static_cast<unsigned char>(source[k]);
                }
            }
        }
    }
}

void harvest_env_save_camera_image(void *env, int index, const char *filename) {
    Slot &slot = static_cast<Environment *>(env)->slots.at(index);
    renderSlot(slot);
    slot.tree->camera->printWindow(filename, "jpeg");
}

void harvest_env_measure_camera_view(void *env, int index, HarvestCameraView *view) {
    Slot &slot = static_cast<Environment *>(env)->slots.at(index);
    Tree &tree = *slot.tree;
    renderSlot(slot);
    std::vector<uint> pixel_UUIDs;
    std::vector<float> depth;
    uint width = 0;
    uint height = 0;
    tree.camera->getPixelUUIDs(pixel_UUIDs, width, height);
    tree.camera->getDepthImage(depth, width, height);
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
            tree->movie_camera = std::make_unique<Visualizer>(static_cast<uint>(width), static_cast<uint>(height), 8, false, true);
            tree->movie_camera->disableMessages();
            tree->movie_camera->hideWatermark();
            tree->movie_camera->setBackgroundColor(make_RGBcolor(0.72f, 0.83f, 0.95f));
            tree->movie_camera->setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
            tree->movie_camera->setLightDirection(make_vec3(-0.35f, -0.45f, 0.82f));
            tree->movie_camera->setCameraFieldOfView(movie_camera_field_of_view);
            tree->movie_camera->hideNavigationGizmo();
            tree->movie_camera->buildContextGeometry(&tree->context);
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
    forEachSlot(environment, [&](size_t i) {
        resetSlot(environment.slots[i]);
        writeObservation(environment.slots[i], observations + i * observation_size);
    });
}

void harvest_env_step(void *env, const float *actions, float *observations, float *rewards, int *outcomes, float *terminal_observations) {
    Environment &environment = *static_cast<Environment *>(env);
    forEachSlot(environment, [&](size_t i) { stepSlot(environment.slots[i], actions + i * action_size, observations + i * observation_size, rewards[i], outcomes[i], terminal_observations + i * observation_size); });
}

} // extern "C"
