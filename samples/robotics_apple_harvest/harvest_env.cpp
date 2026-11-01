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

    // Gripper: an ideal vacuum cup. It grips a fruit within reach of its face, and then acts as a spring that can only pull, reacting on the arm.
    const float grip_reach = 0.03f; // m between the cup's face and the fruit's surface
    const float grip_stiffness = 2000.f; // N/m
    const float grip_damping = 20.f; // N*s/m
    const float grip_ramp_seconds = 0.1f; // the grip builds up over this time, as a vacuum does
    const float grip_limit = 40.f; // N above which the cup loses its seal

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

        Tree(int seed, int instance_count, bool leaves) : plantarchitecture(&context), engine(&context) {
            context.seedRandomGenerator(seed);
            plantarchitecture.disableMessages();
            engine.disableMessages();
            plantarchitecture.loadPlantModelFromLibrary(plant_model);
            uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), plant_age);
            if (!leaves) {
                plantarchitecture.removePlantLeaves(plantID);
            }

            USDExportParameters plant_physics;
            plant_physics.elastic_modulus = 5e9f;
            plant_physics.wood_density = 750.f;
            plant_physics.fruit_mass = 0.15f;
            const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID, plant_physics);

            for (const PlantArticulationLink &link: articulation.links) {
                if (link.type == ARTICULATION_LINK_INTERNODE) {
                    // The local +z axis of the link is the capsule axis, the third column of its rotation matrix
                    const vec3 axis = make_vec3(2.f * (link.qx * link.qz + link.qw * link.qy), 2.f * (link.qy * link.qz - link.qw * link.qx), 1.f - 2.f * (link.qx * link.qx + link.qy * link.qy));
                    wood.push_back({link.position - link.half_length * axis, link.position + link.half_length * axis, link.radius});
                }
            }

            // Reachable fruit, nearest the robot first, since those are the ones not hidden behind the wall
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
            for (uint fruit: reachable) {
                int internodes = 0;
                for (int link = articulation.links[fruit].parent_index; link >= 0 && internodes < flexible_internodes_per_candidate; link = articulation.links[link].parent_index) {
                    if (articulation.links[link].type == ARTICULATION_LINK_INTERNODE && articulation.links[link].parent_index >= 0) {
                        plant_parameters.flexible_links.push_back(static_cast<uint>(link));
                        internodes++;
                    }
                }
            }
            plant = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, plant_physics, plant_parameters);

            robot = std::make_unique<Robot>(&engine, "../harvest_arm.urdf", "robot", robot_base);
            // The force limits are the effort limits of the joints in the robot's model
            robot->addJointServo("drive", 20000.f, 5000.f, 2000.f, 20.f);
            robot->addJointServo("yaw", 2000.f, 400.f, 300.f, 2.f);
            robot->addJointServo("shoulder", 2000.f, 400.f, 300.f, 2.f);
            robot->addJointServo("elbow", 1000.f, 200.f, 200.f, 1.f);
            robot->addJointServo("wrist", 200.f, 30.f, 60.f, 0.2f);
            robot->addJointServo("roll", 30.f, 5.f, 30.f, 0.05f);
            robot->setTool("tool", cup_face, make_vec3(1, 0, 0));

            // What the camera can see of the robot, in its zero pose, in which the tool lies along +x from the wrist
            const vec3 wrist = robot_base + make_vec3(0.15f + 0.65f + 0.6f, 0, 0.75f);
            engine.bindObjectToBody(context.addConeObject(16, wrist, wrist + make_vec3(0.28f, 0, 0), 0.03f, 0.018f, make_RGBcolor(0.9f, 0.4f, 0.08f)), "robot/tool");
            engine.bindObjectToBody(context.addSphereObject(16, wrist + make_vec3(0.3f, 0, 0), 0.02f, make_RGBcolor(0.15f, 0.15f, 0.15f)), "robot/tool");
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
        float grip_rest_length = 0.f;
        float grip_seconds = 0.f;
        float grip_force = 0.f;
        float previous_load = 0.f;
        float previous_distance = 0.f;
        vec3 tool_target;
        int control_step = 0;
        vec3 pull_direction;
        HarvestEpisodeRecord current{};
        HarvestEpisodeRecord last{};
    };

    vec3 toVec3(const mjtNum *p) {
        return make_vec3(float(p[0]), float(p[1]), float(p[2]));
    }

    float surfaceDistance(const Slot &slot) {
        const mjData *data = slot.tree->engine.getMuJoCoData(slot.instance);
        return (toVec3(data->xipos + 3 * slot.target->body_ID) - slot.tree->robot->getToolPosition(slot.instance)).magnitude() - slot.target->radius;
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
    void pointCamera(Slot &slot) {
        Robot &robot = *slot.tree->robot;
        const vec3 cup = robot.getToolPosition(slot.instance);
        const vec3 axis = robot.getToolAxis(slot.instance);
        slot.tree->camera->setCameraPosition(cup - 0.3f * axis + make_vec3(0, 0, 0.08f), cup + 0.3f * axis);
    }

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
        observation[i++] = slot.grip_force / grip_limit;
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
        slot.previous_load = 0.f;
        slot.control_step = 0;
        slot.tool_target = tree.robot->getToolPosition(slot.instance);
        slot.previous_distance = surfaceDistance(slot);
        slot.pull_direction = make_vec3(0, 0, 0);
        slot.current = HarvestEpisodeRecord{};
        slot.current.detach_force = slot.detach_force;
        slot.current.steps_to_grip = -1;
    }

    // The force of the grip on the fruit, applied to the fruit and, in reaction, to the arm at the cup's face
    void applyGrip(Slot &slot, float dt) {
        Tree &tree = *slot.tree;
        mjData *data = tree.engine.getMuJoCoData(slot.instance);
        const int fruit_body = slot.target->body_ID;
        const int tool_body = tree.tool_body_ID;
        mju_zero(data->xfrc_applied + 6 * fruit_body, 6);
        mju_zero(data->xfrc_applied + 6 * tool_body, 6);
        slot.grip_force = 0.f;
        if (!slot.gripped) {
            return;
        }
        slot.grip_seconds += dt;
        const float ramp = std::min(1.f, slot.grip_seconds / grip_ramp_seconds);
        const vec3 tool = tree.robot->getToolPosition(slot.instance);
        const vec3 fruit = toVec3(data->xipos + 3 * fruit_body);
        vec3 toward_tool = tool - fruit;
        const float distance = toward_tool.magnitude();
        if (distance < 1e-6f) {
            return;
        }
        toward_tool = toward_tool / distance;
        slot.pull_direction = toward_tool;
        const float stretch = distance - slot.grip_rest_length;
        if (stretch <= 0.f) {
            return;
        }
        mjtNum velocity_fruit[6], velocity_tool[6];
        mj_objectVelocity(tree.engine.getMuJoCoModel(), data, mjOBJ_BODY, fruit_body, velocity_fruit, 0);
        mj_objectVelocity(tree.engine.getMuJoCoModel(), data, mjOBJ_BODY, tool_body, velocity_tool, 0);
        const vec3 separating_velocity = toVec3(velocity_tool + 3) - toVec3(velocity_fruit + 3);
        const float force = std::max(0.f, ramp * (grip_stiffness * stretch + grip_damping * (separating_velocity * toward_tool)));
        slot.grip_force = force;

        const vec3 on_fruit = force * toward_tool;
        data->xfrc_applied[6 * fruit_body] = on_fruit.x;
        data->xfrc_applied[6 * fruit_body + 1] = on_fruit.y;
        data->xfrc_applied[6 * fruit_body + 2] = on_fruit.z;

        // The reaction acts at the cup's face rather than the tool's center of mass, so it also carries a torque
        const vec3 arm = tool - toVec3(data->xipos + 3 * tool_body);
        const vec3 on_tool = -1.f * on_fruit;
        const vec3 torque = cross(arm, on_tool);
        data->xfrc_applied[6 * tool_body] = on_tool.x;
        data->xfrc_applied[6 * tool_body + 1] = on_tool.y;
        data->xfrc_applied[6 * tool_body + 2] = on_tool.z;
        data->xfrc_applied[6 * tool_body + 3] = torque.x;
        data->xfrc_applied[6 * tool_body + 4] = torque.y;
        data->xfrc_applied[6 * tool_body + 5] = torque.z;
    }

    void stepSlot(Slot &slot, const float *action, float *observation, float &reward, int &outcome, float *terminal_observation) {
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
            slot.gripped = true;
            slot.grip_rest_length = distance_before_grip + slot.target->radius;
            slot.grip_seconds = 0.f;
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
            } else if (slot.grip_force > grip_limit) {
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

        if (outcome != HARVEST_RUNNING) {
            slot.last = slot.current;
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
    try {
        auto environment = std::make_unique<Environment>();
        environment->thread_count = std::max(1, thread_count);
        for (int t = 0; t < tree_count; t++) {
            environment->trees.push_back(std::make_unique<Tree>(tree_seeds[t], instances_per_tree, leaves != 0));
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
            tree->camera->setCameraFieldOfView(70.f);
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
