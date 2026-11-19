// Apple harvesting environment. See harvest_env.h.
//
// A C interface to FruitHarvestTask of the robotics plug-in, which holds the task, for driving it from Python (harvest_env.py) through ctypes. It also
// writes trees for the IsaacSim comparison (isaac/), which is particular to this sample.

#include "harvest_env.h"

#include "FruitHarvestTask.h"

using namespace helios;

namespace {

    struct Environment {
        std::unique_ptr<FruitHarvestTask> task;
        // Buffers for the arrays the task fills, copied into the caller's
        std::vector<float> observations;
        std::vector<float> rewards;
        std::vector<int> outcomes;
        std::vector<float> terminal_observations;
        std::vector<float> actions;
        // The names of the links of each copy's tree, kept for harvest_env_link_name(), which hands out pointers to them
        std::map<int, std::vector<std::string>> link_names;
    };

    FruitHarvestTask &taskOf(void *env) {
        return *static_cast<Environment *>(env)->task;
    }

    const FruitHarvestTask &taskOf(const void *env) {
        return *static_cast<const Environment *>(env)->task;
    }

    // The settings of this sample: those of the task's defaults, which are the apple harvest
    FruitHarvestTaskParameters sampleParameters(bool leaves, bool show_arm) {
        FruitHarvestTaskParameters parameters;
        parameters.leaves = leaves;
        parameters.show_arm = show_arm;
        return parameters;
    }

    std::vector<float> asVector(const float *values, size_t count) {
        return std::vector<float>(values, values + count);
    }

} // namespace

extern "C" {

void *harvest_env_create(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves) {
    return harvest_env_create_for_movie(tree_seeds, tree_count, instances_per_tree, thread_count, seed, leaves, 0);
}

void *harvest_env_create_for_movie(const int *tree_seeds, int tree_count, int instances_per_tree, int thread_count, int seed, int leaves, int show_arm) {
    try {
        if (tree_count <= 0 || instances_per_tree <= 0) {
            helios_runtime_error("ERROR (harvest_env_create): The numbers of trees and of instances of each must be positive.");
        }
        auto environment = std::make_unique<Environment>();
        environment->task = std::make_unique<FruitHarvestTask>(std::vector<int>(tree_seeds, tree_seeds + tree_count), static_cast<uint>(instances_per_tree), static_cast<uint>(std::max(1, thread_count)), static_cast<uint>(seed),
                                                               sampleParameters(leaves != 0, show_arm != 0));
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
    return FruitHarvestTask::treeHasReachableFruit(tree_seed, sampleParameters(false, false)) ? 1 : 0;
}

int harvest_env_export_tree(int tree_seed, const char *directory) {
    try {
        const FruitHarvestTaskParameters task = sampleParameters(false, false);
        Context context;
        context.seedRandomGenerator(tree_seed);
        PlantArchitecture plantarchitecture(&context);
        plantarchitecture.disableMessages();
        plantarchitecture.loadPlantModelFromLibrary(task.plant_model);
        uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), task.plant_age);
        plantarchitecture.removePlantLeaves(plantID);

        USDExportParameters plant_physics = FruitHarvestTask::plantArticulationParameters(task);
        // PhysX needs this much armature to hold a tree up (doc/isaac_comparison_report.md); MuJoCo does not use it
        plant_physics.armature_stability_ratio = 1.f;
        plant_physics.physics_steps_per_second = 500.f;
        const std::filesystem::path base = std::filesystem::path(directory) / ("tree_" + std::to_string(tree_seed));
        plantarchitecture.writePlantStructureUSD(plantID, base.string() + ".usda", plant_physics);

        const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID, plant_physics);
        const std::vector<uint> candidates = FruitHarvestTask::chooseCandidateFruit(articulation, task);
        const std::vector<uint> flexible = FruitHarvestTask::chooseFlexibleInternodes(articulation, candidates, task);
        std::vector<char> is_flexible(articulation.links.size(), 0), is_candidate(articulation.links.size(), 0);
        for (uint link: flexible) {
            is_flexible[link] = 1;
        }
        for (uint link: candidates) {
            is_candidate[link] = 1;
        }

        const vec3 robot_base = task.robot_base;
        const vec3 cup_face = FruitHarvestTask::getCupFace();
        std::ofstream json(base.string() + ".json");
        json << std::setprecision(9);
        json << "{\n\"seed\": " << tree_seed << ", \"plant_model\": \"" << task.plant_model << "\", \"plant_age\": " << task.plant_age << ",\n";
        // The task, as FruitHarvestTask sets it up
        json << "\"task\": {\"physics_timestep\": 0.002, \"physics_steps_per_control\": " << task.physics_steps_per_control << ", \"max_tool_speed\": " << task.max_tool_speed << ", \"max_target_lead\": " << task.max_target_lead
             << ", \"max_control_steps\": " << task.max_control_steps << ", \"grip_reach\": " << task.cup.seal_reach << ", \"cup_lip_radius\": " << task.cup.lip_radius << ", \"cup_lip_thickness\": " << task.cup.lip_thickness
             << ", \"grip_stiffness\": " << task.cup.stiffness << ", \"grip_damping\": " << task.cup.damping << ", \"grip_ramp_seconds\": " << task.cup.ramp_seconds << ", \"grip_limit\": " << task.cup.pull_limit
             << ", \"min_detach_force\": " << task.min_detach_force << ", \"max_detach_force\": " << task.max_detach_force << ", \"settling_physics_steps\": " << task.settling_physics_steps << ", \"protective_stop_force\": " << task.protective_stop_force
             << ", \"start_offset\": " << task.start_offset << ", \"robot_base\": [" << robot_base.x << ", " << robot_base.y << ", " << robot_base.z << "], \"cup_face\": [" << cup_face.x << ", " << cup_face.y << ", " << cup_face.z << "], \"tucked_pose\": [";
        for (size_t j = 0; j < task.start_pose.size(); j++) {
            json << (j ? ", " : "") << task.start_pose[j];
        }
        // The servos on the robot: stiffness, damping, force limit, armature, in N/m or N m/rad
        json << "],\n  \"servos\": {";
        bool first_servo = true;
        for (const FruitHarvestJointServo &servo: FruitHarvestTask::getJointServos()) {
            json << (first_servo ? "" : ", ") << "\"" << servo.joint << "\": [" << servo.stiffness << ", " << servo.damping << ", " << servo.force_limit << ", " << servo.armature << "]";
            first_servo = false;
        }
        json << "}},\n";

        // How each link's joint is simulated by addPlantToPhysicsEngine() with the parameters FruitHarvestTask gives it: "sprung" (a ball joint), "locked"
        // (welded, or held by the detachable weld of a candidate fruit), or "root". Leaves are removed, so rigid_leaves does not matter.
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
    return static_cast<int>(taskOf(env).getCopyCount());
}

int harvest_env_observation_size(void) {
    return static_cast<int>(FruitHarvestTask::getObservationSize());
}

int harvest_env_action_size(void) {
    return static_cast<int>(FruitHarvestTask::getActionSize());
}

int harvest_env_max_steps(void) {
    return static_cast<int>(FruitHarvestTaskParameters().max_control_steps);
}

int harvest_env_set_near_start_fraction(void *env, float fraction) {
    try {
        taskOf(env).setNearStartFraction(fraction);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_enable_camera(void *env, int width, int height) {
    try {
        if (width <= 0 || height <= 0) {
            helios_runtime_error("ERROR (harvest_env_enable_camera): The image size must be positive.");
        }
        taskOf(env).enableWristCamera(static_cast<uint>(width), static_cast<uint>(height));
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

void harvest_env_render(void *env, unsigned char *images) {
    std::vector<unsigned char> rendered;
    taskOf(env).renderWristCameras(rendered);
    std::copy(rendered.begin(), rendered.end(), images);
}

void harvest_env_save_camera_image(void *env, int index, const char *filename) {
    taskOf(env).saveWristCameraImage(static_cast<uint>(index), filename);
}

void harvest_env_measure_camera_view(void *env, int index, HarvestCameraView *view) {
    const FruitHarvestCameraView measured = taskOf(env).measureWristCameraView(static_cast<uint>(index));
    view->target_fruit = measured.target_fruit;
    view->other_fruit = measured.other_fruit;
    view->leaves = measured.leaves;
    view->near_leaves = measured.near_leaves;
    view->wood = measured.wood;
}

int harvest_env_link_count(const void *env, int index) {
    return static_cast<int>(taskOf(env).getPlantLinkCount(static_cast<uint>(index)));
}

const char *harvest_env_link_name(const void *env, int index, int link) {
    // The names are kept so that the pointer stays valid while the environment lives
    Environment &environment = *static_cast<Environment *>(const_cast<void *>(env));
    auto names = environment.link_names.find(index);
    if (names == environment.link_names.end()) {
        names = environment.link_names.emplace(index, environment.task->getPlantLinkNames(static_cast<uint>(index))).first;
    }
    return names->second.at(link).c_str();
}

void harvest_env_rest_link_poses(const void *env, int index, float *link_poses) {
    const std::vector<float> poses = taskOf(env).getRestLinkPoses(static_cast<uint>(index));
    std::copy(poses.begin(), poses.end(), link_poses);
}

int harvest_env_display_state(const void *env, int index, int last_episode, float *link_poses, float *robot_joints) {
    try {
        std::vector<float> poses;
        std::vector<float> joints;
        taskOf(env).getDisplayState(static_cast<uint>(index), last_episode != 0, poses, joints);
        std::copy(poses.begin(), poses.end(), link_poses);
        std::copy(joints.begin(), joints.end(), robot_joints);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

void harvest_env_target_fruit(const void *env, int index, float *position) {
    const vec3 fruit = taskOf(env).getTargetFruitPosition(static_cast<uint>(index));
    position[0] = fruit.x;
    position[1] = fruit.y;
    position[2] = fruit.z;
}

int harvest_env_enable_movie_camera(void *env, int width, int height) {
    try {
        if (width <= 0 || height <= 0) {
            helios_runtime_error("ERROR (harvest_env_enable_movie_camera): The image size must be positive.");
        }
        taskOf(env).enableMovieCamera(static_cast<uint>(width), static_cast<uint>(height));
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_add_movie_scenery(void *env, int neighbors_each_side, int trellis) {
    try {
        if (neighbors_each_side < 0) {
            helios_runtime_error("ERROR (harvest_env_add_movie_scenery): The number of neighboring trees on each side must not be negative.");
        }
        taskOf(env).addMovieScenery(static_cast<uint>(neighbors_each_side), trellis != 0);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_enable_radiation_movie_cameras(void *env, int width, int height, int wrist_width, int wrist_height, int antialiasing_samples, int periodic_repeats, int row_repeats) {
    try {
        if (width <= 0 || height <= 0 || wrist_width <= 0 || wrist_height <= 0 || antialiasing_samples <= 0) {
            helios_runtime_error("ERROR (harvest_env_enable_radiation_movie_cameras): The image sizes and the number of antialiasing samples must be positive.");
        }
        if (periodic_repeats < 0 || row_repeats < 0) {
            helios_runtime_error("ERROR (harvest_env_enable_radiation_movie_cameras): The numbers of periodic repeats along the row and of rows must not be negative.");
        }
        taskOf(env).enableRadiationMovieCameras(static_cast<uint>(width), static_cast<uint>(height), static_cast<uint>(wrist_width), static_cast<uint>(wrist_height), static_cast<uint>(antialiasing_samples), static_cast<uint>(periodic_repeats),
                                                static_cast<uint>(row_repeats));
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_render_radiation_movie_frame(void *env, int index, const float *link_poses, const float *robot_joints, const float *eye, const float *look_at, const char *directory, char *movie_image_file, char *wrist_image_file, int file_name_capacity) {
    try {
        FruitHarvestTask &task = taskOf(env);
        const uint copy = static_cast<uint>(index);
        std::string movie_written, wrist_written;
        task.renderRadiationMovieFrame(copy, asVector(link_poses, 7 * size_t(task.getPlantLinkCount(copy))), asVector(robot_joints, 6), make_vec3(eye[0], eye[1], eye[2]), make_vec3(look_at[0], look_at[1], look_at[2]), directory, movie_written,
                                       wrist_written);
        for (const auto &[written, file_name]: {std::pair<const std::string *, char *>{&movie_written, movie_image_file}, {&wrist_written, wrist_image_file}}) {
            if (written->size() + 1 > size_t(file_name_capacity)) {
                helios_runtime_error("ERROR (harvest_env_render_radiation_movie_frame): The name of the image written, " + *written + ", is longer than the room given for it.");
            }
            std::snprintf(file_name, size_t(file_name_capacity), "%s", written->c_str());
        }
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_render_movie_frame(void *env, int index, const float *link_poses, const float *robot_joints, const float *eye, const float *look_at, unsigned char *image) {
    try {
        FruitHarvestTask &task = taskOf(env);
        const uint copy = static_cast<uint>(index);
        std::vector<unsigned char> rendered;
        task.renderMovieFrame(copy, asVector(link_poses, 7 * size_t(task.getPlantLinkCount(copy))), asVector(robot_joints, 6), make_vec3(eye[0], eye[1], eye[2]), make_vec3(look_at[0], look_at[1], look_at[2]), rendered);
        std::copy(rendered.begin(), rendered.end(), image);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

int harvest_env_render_wrist_frame(void *env, int index, const float *link_poses, const float *robot_joints, unsigned char *image) {
    try {
        FruitHarvestTask &task = taskOf(env);
        const uint copy = static_cast<uint>(index);
        std::vector<unsigned char> rendered;
        task.renderWristFrame(copy, asVector(link_poses, 7 * size_t(task.getPlantLinkCount(copy))), asVector(robot_joints, 6), rendered);
        std::copy(rendered.begin(), rendered.end(), image);
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 0;
    }
}

void harvest_env_set_follow_through(void *env, int control_steps) {
    taskOf(env).setFollowThrough(static_cast<uint>(std::max(0, control_steps)));
}

void harvest_env_last_episode(const void *env, int index, HarvestEpisodeRecord *record) {
    const FruitHarvestEpisodeRecord last = taskOf(env).getLastEpisode(static_cast<uint>(index));
    record->detach_force = last.detach_force;
    record->final_stem_load = last.final_stem_load;
    record->final_grip_force = last.final_grip_force;
    record->pull_x = last.pull_direction.x;
    record->pull_y = last.pull_direction.y;
    record->pull_z = last.pull_direction.z;
    record->peak_contact_force = last.peak_contact_force;
    record->peak_wood_motion = last.peak_wood_motion;
    record->peak_tool_speed = last.peak_tool_speed;
    record->steps_to_grip = last.steps_to_grip;
    record->steps = last.steps;
}

void harvest_env_reset(void *env, float *observations) {
    Environment &environment = *static_cast<Environment *>(env);
    environment.task->reset(environment.observations);
    std::copy(environment.observations.begin(), environment.observations.end(), observations);
}

void harvest_env_step(void *env, const float *actions, float *observations, float *rewards, int *outcomes, float *terminal_observations) {
    Environment &environment = *static_cast<Environment *>(env);
    FruitHarvestTask &task = *environment.task;
    const size_t count = task.getCopyCount();
    const size_t observation_size = FruitHarvestTask::getObservationSize();
    environment.actions.assign(actions, actions + count * FruitHarvestTask::getActionSize());
    // Rows of copies that go on are left as the caller had them
    environment.terminal_observations.assign(terminal_observations, terminal_observations + count * observation_size);
    task.step(environment.actions, environment.observations, environment.rewards, environment.outcomes, environment.terminal_observations);
    std::copy(environment.observations.begin(), environment.observations.end(), observations);
    std::copy(environment.rewards.begin(), environment.rewards.end(), rewards);
    std::copy(environment.outcomes.begin(), environment.outcomes.end(), outcomes);
    std::copy(environment.terminal_observations.begin(), environment.terminal_observations.end(), terminal_observations);
}

} // extern "C"
