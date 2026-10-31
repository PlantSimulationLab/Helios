// A mobile robot arm drives along a dormant grapevine row, pushes its open pruner jaws into a shoot until the shoot bends
// against them, reads the force on its tool, and cuts the shoot, which falls to the ground. Everything is simulated: the
// arm follows its servos, the shoot responds to the arm, and the arm feels the shoot.
//
// Usage: ./robotics_pruning [--norender]
// Run from the build directory, so that the robot model is found at ../pruning_arm.urdf. Frames are written to ./frames.

#include "PlantPhysics.h"
#include "Robot.h"
#include "Visualizer.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {

    const RGBcolor robot_orange = make_RGBcolor(0.9f, 0.4f, 0.08f);
    const RGBcolor joint_gray = make_RGBcolor(0.25f, 0.25f, 0.27f);
    const RGBcolor platform_green = make_RGBcolor(0.2f, 0.4f, 0.25f);
    const RGBcolor blade_steel = make_RGBcolor(0.75f, 0.77f, 0.8f);

    // Geometry of the robot in its zero pose, in which the arm is straight along +x, with each piece following its body
    void addRobotGeometry(Context &context, PhysicsEngine &engine, const vec3 &base) {
        const vec3 turret = base + make_vec3(0.15f, 0, 0);
        const vec3 shoulder = turret + make_vec3(0, 0, 0.75f);
        const vec3 elbow = shoulder + make_vec3(0.65f, 0, 0);
        const vec3 wrist = elbow + make_vec3(0.6f, 0, 0);
        const vec3 pivot = wrist + make_vec3(0.22f, 0, 0);
        const vec3 joint_axis = make_vec3(0, 1, 0);

        auto addCone = [&](const vec3 &start, const vec3 &end, float start_radius, float end_radius, const RGBcolor &color, const std::string &body) { engine.bindObjectToBody(context.addConeObject(20, start, end, start_radius, end_radius, color), "robot/" + body); };

        engine.bindObjectToBody(context.addBoxObject(base + make_vec3(0, 0, 0.22f), make_vec3(0.55f, 0.8f, 0.2f), make_int3(1, 1, 1), platform_green), "robot/platform");
        for (float dx: {-0.33f, 0.33f}) {
            for (float dy: {-0.28f, 0.28f}) {
                const vec3 hub = base + make_vec3(dx, dy, 0.12f);
                addCone(hub - make_vec3(0.05f, 0, 0), hub + make_vec3(0.05f, 0, 0), 0.12f, 0.12f, make_RGBcolor(0.08f, 0.08f, 0.08f), "platform");
            }
        }
        auto addCap = [&](const vec3 &center, float radius, const RGBcolor &color, const std::string &body) { engine.bindObjectToBody(context.addSphereObject(16, center, radius, color), "robot/" + body); };
        addCap(shoulder, 0.075f, joint_gray, "upper_arm");
        addCap(elbow, 0.058f, joint_gray, "forearm");
        addCap(wrist, 0.04f, joint_gray, "wrist_link");
        addCone(turret + make_vec3(0, 0, 0.32f), shoulder - make_vec3(0, 0, 0.06f), 0.09f, 0.075f, joint_gray, "turret");
        addCone(shoulder - 0.1f * joint_axis, shoulder + 0.1f * joint_axis, 0.07f, 0.07f, joint_gray, "upper_arm");
        addCone(shoulder, elbow, 0.055f, 0.045f, robot_orange, "upper_arm");
        addCone(elbow - 0.08f * joint_axis, elbow + 0.08f * joint_axis, 0.055f, 0.055f, joint_gray, "forearm");
        addCone(elbow, wrist, 0.042f, 0.032f, robot_orange, "forearm");
        addCone(wrist - 0.055f * joint_axis, wrist + 0.055f * joint_axis, 0.035f, 0.035f, joint_gray, "wrist_link");
        addCone(wrist, wrist + make_vec3(0.11f, 0, 0), 0.036f, 0.036f, joint_gray, "tool");
        addCone(wrist + make_vec3(0.11f, 0, 0), pivot, 0.024f, 0.012f, robot_orange, "tool");

        // Fixed jaw on the -y side of the throat, and the blade, which is open in the zero pose
        engine.bindObjectToBody(context.addBoxObject(pivot + make_vec3(0.05f, -0.016f, 0), make_vec3(0.1f, 0.008f, 0.016f), make_int3(1, 1, 1), joint_gray), "robot/tool");
        uint blade = context.addBoxObject(make_vec3(0, 0, 0), make_vec3(0.1f, 0.008f, 0.016f), make_int3(1, 1, 1), blade_steel);
        context.rotateObject(blade, 0.5f, "z");
        context.translateObject(blade, pivot + make_vec3(0.0439f, 0.024f, 0));
        engine.bindObjectToBody(blade, "robot/blade");
    }

    // The internode segment most exposed to the robot: near-vertical, not too thick, within the arm's reach, nearest to it, and with enough of the shoot beyond it to be worth cutting
    uint findTargetLink(const PlantArticulation &articulation) {
        const float min_length_removed = 0.4f; // m

        // Parents precede their children, so walking backward accumulates the length of everything beyond each link
        std::vector<float> length_beyond(articulation.links.size(), 0.f);
        for (size_t i = articulation.links.size(); i-- > 0;) {
            const PlantArticulationLink &link = articulation.links[i];
            if (link.parent_index >= 0) {
                length_beyond[link.parent_index] += length_beyond[i] + 2.f * link.half_length;
            }
        }

        uint target = 0;
        float nearest_x = std::numeric_limits<float>::max();
        for (uint i = 0; i < articulation.links.size(); i++) {
            const PlantArticulationLink &link = articulation.links[i];
            if (link.type != ARTICULATION_LINK_INTERNODE || link.parent_index < 0 || link.radius < 0.002f || link.radius > 0.008f) {
                continue;
            }
            // The local +z axis of the link is the capsule axis; its world z component is 1 - 2(qx^2 + qy^2)
            const float axis_z = 1.f - 2.f * (link.qx * link.qx + link.qy * link.qy);
            if (axis_z < 0.9f || link.position.z < 1.15f || link.position.z > 1.5f || link.position.x > 0.f || length_beyond[i] < min_length_removed) {
                continue;
            }
            if (link.position.x < nearest_x) {
                nearest_x = link.position.x;
                target = i;
            }
        }
        if (target == 0) {
            helios_runtime_error("ERROR (robotics_pruning): No shoot segment within reach of the robot was found.");
        }
        return target;
    }

} // namespace

int main(int argc, char **argv) {

    const bool render = !(argc > 1 && std::string(argv[1]) == "--norender");
    if (render) {
        std::filesystem::create_directories("frames");
    }

    Context context;
    context.seedRandomGenerator(7);

    // Dormant vine, with leaves and fruit removed as for winter pruning
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary("grapevine_VSP");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 300.f);
    plantarchitecture.removePlantLeaves(plantID);
    plantarchitecture.harvestPlant(plantID);

    USDExportParameters plant_physics;
    plant_physics.elastic_modulus = 5e9f;
    plant_physics.wood_density = 750.f;

    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID, plant_physics);
    const uint target_link = findTargetLink(articulation);
    const vec3 target_rest = articulation.links[target_link].position;
    std::cout << articulation.links.size() << " links. Target " << articulation.links[target_link].name << " (radius " << 1000.f * articulation.links[target_link].radius << " mm) at " << target_rest.x << " " << target_rest.y << " " << target_rest.z << std::endl;

    PlantPhysicsParameters plant_parameters;
    plant_parameters.detachable_links = {target_link};

    PhysicsEngine engine(&context);
    engine.disableMessages();
    uint plant = addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, plant_physics, plant_parameters);

    // Ground large enough that its edges stay out of view
    context.addPatch(make_vec3(0, 0, 0), make_vec2(20.f, 20.f), make_SphericalCoord(0, 0), make_RGBcolor(0.42f, 0.33f, 0.24f));

    const vec3 robot_base = make_vec3(target_rest.x - 1.05f, 0, 0);
    Robot robot(&engine, "../pruning_arm.urdf", "robot", robot_base);
    addRobotGeometry(context, engine, robot_base);
    robot.addJointServo("drive", 20000.f, 5000.f, 0.f, 20.f);
    robot.addJointServo("yaw", 2000.f, 400.f, 0.f, 2.f);
    robot.addJointServo("shoulder", 2000.f, 400.f, 0.f, 2.f);
    robot.addJointServo("elbow", 1000.f, 200.f, 0.f, 1.f);
    robot.addJointServo("wrist", 200.f, 30.f, 0.f, 0.2f);
    robot.addJointServo("roll", 30.f, 5.f, 0.f, 0.05f);
    robot.addJointServo("jaw", 10.f, 1.5f, 0.f, 0.01f);
    robot.setTool("tool", make_vec3(0.22f, 0, 0), make_vec3(1, 0, 0));

    PhysicsEngineParameters parameters;
    parameters.ground_collisions = true;
    engine.initialize(parameters);
    robot.initialize();
    // Tucked well back from the row, so that the arm clears the vine while the platform drives
    robot.setHomePose({0.f, 0.f, 110.f, -150.f, 40.f, 0.f, 0.f});
    engine.writeMJCF("robotics_pruning_model.xml");

    std::unique_ptr<Visualizer> visualizer;
    if (render) {
        visualizer = std::make_unique<Visualizer>(1280, 720, 8, false, true);
        visualizer->disableMessages();
        visualizer->hideWatermark();
        visualizer->setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
        visualizer->setLightDirection(make_vec3(0.3f, -0.5f, 0.8f));
        visualizer->setBackgroundSkyTexture();
    }

    const uint physics_steps_per_control = 10; // 50 Hz control at a 2 ms physics step
    const uint control_steps_per_frame = 2; // 25 frames per second
    uint frame = 0;
    uint control_step = 0;
    vec3 camera_target = target_rest + make_vec3(-0.3f, 0, -0.55f);
    vec3 camera_position = target_rest + make_vec3(-1.1f, -2.1f, -0.1f);

    auto advance = [&]() {
        engine.step(physics_steps_per_control);
        control_step++;
        if (render && control_step % control_steps_per_frame == 0) {
            engine.updateContextGeometry();
            visualizer->setCameraPosition(camera_position, camera_target);
            visualizer->plotUpdate(true);
            char filename[64];
            std::snprintf(filename, sizeof(filename), "frames/frame_%04u.jpeg", frame++);
            visualizer->printWindow(filename, "jpeg");
        }
    };
    if (render) {
        engine.updateContextGeometry();
        visualizer->buildContextGeometry(&context);
    }
    auto controlSteps = [&](float seconds) { return static_cast<uint>(std::round(seconds / (float(physics_steps_per_control) * engine.getTimestep()))); };

    // ---- Let the vine settle under its own weight ---- //
    const uint settle_steps = controlSteps(3.f);
    for (uint s = 0; s < settle_steps; s++) {
        advance();
    }
    const vec3 settled = engine.getLinkPosition(plant, target_link);
    std::cout << "Settled: target moved " << 1000.f * (settled - target_rest).magnitude() << " mm under gravity, largest link speed now " << engine.getMaxLinkSpeed() << " m/s" << std::endl;

    // ---- Drive along the row until the shoot is straight ahead ---- //
    const std::vector<float> home = robot.getJointPositions();
    const uint drive_steps = controlSteps(1.5f);
    float max_force_while_driving = 0.f;
    for (uint s = 0; s < drive_steps; s++) {
        robot.setJointTarget("drive", (settled.y - robot_base.y) * float(s + 1) / float(drive_steps));
        advance();
        max_force_while_driving = std::max(max_force_while_driving, engine.getExternalContactForce(plant).magnitude());
    }
    if (max_force_while_driving > 0.f) {
        helios_runtime_error("ERROR (robotics_pruning): The robot hit the vine while driving along the row, with a force of " + std::to_string(max_force_while_driving) + " N. Tuck the home pose further back.");
    }
    for (uint s = 0; s < controlSteps(0.5f); s++) {
        advance();
    }

    // ---- Swing the arm to a point in front of the shoot, in joint space ---- //
    const vec3 forward = make_vec3(1, 0, 0);
    const float approach_distance = 0.18f;
    const vec3 approach_point = settled - approach_distance * forward;
    RobotInverseKinematicsResult approach_pose = robot.solveInverseKinematics(approach_point, forward);
    if (!approach_pose.converged) {
        helios_runtime_error("ERROR (robotics_pruning): The robot cannot reach the approach point. Remaining error " + std::to_string(approach_pose.position_error) + " m.");
    }
    const std::vector<float> swing_start = robot.getJointPositions();
    const uint swing_steps = controlSteps(2.f);
    for (uint s = 0; s < swing_steps; s++) {
        const float t = float(s + 1) / float(swing_steps);
        const float smooth = t * t * (3.f - 2.f * t);
        std::vector<float> targets = swing_start;
        for (size_t j = 0; j < targets.size(); j++) {
            targets[j] = swing_start[j] + smooth * (approach_pose.joint_positions[j] - swing_start[j]);
        }
        robot.setJointTargets(targets);
        advance();
    }
    for (uint s = 0; s < controlSteps(0.5f); s++) {
        advance();
    }
    std::cout << "At approach point, tool error " << 1000.f * (robot.getToolPosition() - approach_point).magnitude() << " mm" << std::endl;

    // ---- Push the open jaws straight into the shoot and past it, aiming at where the shoot is now ---- //
    const vec3 before_push = engine.getLinkPosition(plant, target_link);
    const vec3 push_start = robot.getToolPosition();
    const vec3 push_end = before_push + 0.04f * forward; // the pivot passes the shoot's position by this much
    const float push_speed = 0.1f; // m/s
    const uint push_steps = controlSteps((push_end - push_start).magnitude() / push_speed);
    float max_force = 0.f;
    float contact_force_on_plant_at_max = 0.f;
    float max_deflection = 0.f;
    bool touched = false;
    float deflection_at_first_touch = 0.f;
    for (uint s = 0; s < push_steps + controlSteps(0.5f); s++) {
        const float t = std::min(1.f, float(s + 1) / float(push_steps));
        const vec3 tool_target = push_start + t * (push_end - push_start);
        RobotInverseKinematicsResult ik = robot.solveInverseKinematics(tool_target, forward);
        if (!ik.converged) {
            helios_runtime_error("ERROR (robotics_pruning): The robot cannot follow the push path. Remaining error " + std::to_string(ik.position_error) + " m.");
        }
        robot.setJointTargets(ik.joint_positions);
        advance();
        const float force = robot.getToolForce().magnitude();
        const float deflection = (engine.getLinkPosition(plant, target_link) - before_push).magnitude();
        if (force > 0.f && !touched) {
            touched = true;
            deflection_at_first_touch = deflection;
        }
        if (force > max_force) {
            max_force = force;
            contact_force_on_plant_at_max = engine.getExternalContactForce(plant).magnitude();
        }
        max_deflection = std::max(max_deflection, deflection);
    }
    const float force_before_cut = robot.getToolForce().magnitude();
    const float deflection_before_cut = (engine.getLinkPosition(plant, target_link) - before_push).magnitude();
    if (!touched) {
        helios_runtime_error("ERROR (robotics_pruning): The jaws never touched the shoot.");
    }

    // ---- Close the blade and cut when it has closed on the shoot ---- //
    const uint close_steps = controlSteps(0.3f);
    const float cut_at_angle = -15.f; // degrees of jaw closure at which the blade has reached the shoot
    bool cut = false;
    double cut_time = 0;
    const size_t jaw_index = 6;
    for (uint s = 0; s < close_steps; s++) {
        robot.setJointTarget("jaw", rad2deg(-0.5f) * float(s + 1) / float(close_steps));
        if (!cut && robot.getJointPositions()[jaw_index] < cut_at_angle) {
            engine.detachLink(plant, target_link);
            engine.splitTubesAtLink(plant, target_link);
            cut = true;
            cut_time = engine.getSimulationTime();
        }
        advance();
    }
    if (!cut) {
        helios_runtime_error("ERROR (robotics_pruning): The blade did not close on the shoot.");
    }
    for (uint s = 0; s < controlSteps(0.1f); s++) {
        advance();
    }
    const float deflection_of_stump = (engine.getLinkPosition(plant, static_cast<uint>(articulation.links[target_link].parent_index)) - before_push).magnitude();
    const float force_after_cut = robot.getToolForce().magnitude();

    // ---- Open, back away, and let the cut piece land ---- //
    std::cout << "Stump is " << 1000.f * deflection_of_stump << " mm from the shoot's position before the push" << std::endl;
    const std::vector<float> at_cut = robot.getJointPositions();
    const uint retreat_steps = controlSteps(1.5f);
    for (uint s = 0; s < retreat_steps; s++) {
        const float t = float(s + 1) / float(retreat_steps);
        const float smooth = t * t * (3.f - 2.f * t);
        std::vector<float> targets = at_cut;
        for (size_t j = 0; j < targets.size(); j++) {
            targets[j] = at_cut[j] + smooth * (swing_start[j] - at_cut[j]);
        }
        targets[0] = at_cut[0];
        targets[jaw_index] = at_cut[jaw_index] + t * (0.f - at_cut[jaw_index]);
        robot.setJointTargets(targets);
        advance();
    }
    for (uint s = 0; s < controlSteps(1.f); s++) {
        advance();
    }

    const vec3 piece_final = engine.getLinkPosition(plant, target_link);
    const vec3 stump_final = engine.getLinkPosition(plant, static_cast<uint>(articulation.links[target_link].parent_index));

    std::cout << std::endl << "Physical robot gate" << std::endl;
    std::cout << "  Arm deflects a branch: " << 1000.f * deflection_at_first_touch << " mm of drift before first touch, then " << 1000.f * max_deflection << " mm largest deflection, " << 1000.f * deflection_before_cut << " mm held before the cut" << std::endl;
    std::cout << "  Sensor reads the force: largest " << max_force << " N on the tool, with " << contact_force_on_plant_at_max << " N on the plant at that moment; " << force_before_cut << " N before the cut and " << force_after_cut << " N just after it" << std::endl;
    std::cout << "  Cut completes mid-run: at t = " << cut_time << " s the cut link is released; it comes to rest at z = " << piece_final.z << " m, and the stump stays at z = " << stump_final.z << " m" << std::endl;
    std::cout << "  " << frame << " frames, " << engine.getSimulationTime() << " s simulated" << std::endl;

    return 0;
}
