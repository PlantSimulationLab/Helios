// Sweep a kinematic box through the crown of an apple tree with the physics engine plug-in, and render the deflected
// canopy with the Visualizer. The box moves through the crown and back, as in the IsaacSim test of the USD export.

#include "PlantPhysics.h"
#include "Visualizer.h"

using namespace helios;

int main(int argc, char **argv) {

    const float plant_age = argc > 1 ? std::stof(argv[1]) : 275.f; // days
    const bool render_frames = argc > 2 ? std::string(argv[2]) != "norender" : true;

    PhysicsEngineParameters simulation_parameters;
    if (argc > 3) {
        simulation_parameters.integrator = argv[3];
    }
    PlantPhysicsParameters plant_parameters;
    if (argc > 4) {
        plant_parameters.apply_joint_armature = std::string(argv[4]) == "armature";
    }

    const float box_height_fraction = 0.75f; // box path height as a fraction of tree height
    const float box_size_fraction = 0.1f; // box edge length as a fraction of tree height
    const float box_speed = 0.5f; // m/s
    const float settle_time = 1.f; // s to let the tree sag under gravity before the box moves
    const float frame_rate = 30.f; // rendered frames per simulated second

    Context context;
    context.seedRandomGenerator(7);

    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.loadPlantModelFromLibrary("apple");
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), plant_age);

    USDExportParameters plant_physics;
    plant_physics.elastic_modulus = 5e9f;
    plant_physics.wood_density = 750.f;
    plant_physics.fruit_mass = 0.15f;

    vec3 lower, upper;
    context.getObjectBoundingBox(plantarchitecture.getAllPlantObjectIDs(plantID), lower, upper);
    vec3 center = 0.5f * (lower + upper);
    float height = upper.z - lower.z;

    float box_size = box_size_fraction * height;
    vec3 box_start = make_vec3(lower.x - box_size, center.y, lower.z + box_height_fraction * height);
    vec3 box_end = make_vec3(upper.x + box_size, center.y, box_start.z);
    float travel_time = (box_end.x - box_start.x) / box_speed;

    PhysicsEngine simulation(&context);
    addPlantToPhysicsEngine(simulation, context, plantarchitecture, plantID, plant_physics, plant_parameters);
    uint boxID = simulation.addKinematicBox(box_start, make_vec3(box_size, box_size, box_size));

    auto timer_start = std::chrono::steady_clock::now();
    simulation.initialize(simulation_parameters);
    double compile_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - timer_start).count();
    simulation.writeMJCF("apple_mujoco.xml");

    std::map<int, int> link_type_count;
    for (const auto &link: plantarchitecture.getPlantArticulation(plantID, plant_physics).links) {
        link_type_count[link.type]++;
    }
    std::cout << "Links by type (internode, petiole, peduncle, leaf, fruit, flower):";
    for (int type = ARTICULATION_LINK_INTERNODE; type <= ARTICULATION_LINK_FLOWER; type++) {
        std::cout << " " << link_type_count[type];
    }
    std::cout << std::endl;

    std::cout << "Tree height " << height << " m, " << simulation.getLinkCount() << " links; " << box_size << " m box at z = " << box_start.z << " m, x = " << box_start.x << " -> " << box_end.x << " m, " << 2.f * travel_time
              << " s per out-and-back pass. Model built in " << compile_seconds << " s." << std::endl;

    std::unique_ptr<Visualizer> visualizer;
    if (render_frames) {
        std::filesystem::create_directories("frames");
        visualizer = std::make_unique<Visualizer>(1280, 960, 8, false, true);
        visualizer->setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
        visualizer->setLightDirection(sphere2cart(make_SphericalCoord(1.f, deg2rad(50.f), deg2rad(135.f))));
        visualizer->setBackgroundColor(make_RGBcolor(0.85f, 0.9f, 1.f));
        visualizer->hideWatermark();
        visualizer->disableMessages();
        vec3 crown_center = make_vec3(center.x, center.y, lower.z + 0.65f * height);
        visualizer->setCameraPosition(crown_center + make_vec3(0.35f * height, -1.1f * height, 0.15f * height), crown_center);
        visualizer->buildContextGeometry(&context);
    }

    const float total_time = settle_time + 2.f * travel_time;
    const uint steps_per_frame = std::max(1u, static_cast<uint>(std::round(1.f / (frame_rate * simulation.getTimestep()))));
    uint frame = 0;
    uint max_contacts = 0;
    float max_displacement = 0.f;
    float max_penetration = 0.f;
    double physics_seconds = 0;

    while (simulation.getSimulationTime() < total_time) {

        timer_start = std::chrono::steady_clock::now();
        for (uint s = 0; s < steps_per_frame; s++) {
            float t = simulation.getSimulationTime() - settle_time;
            float fraction = t <= 0.f ? 0.f : (t < travel_time ? t / travel_time : std::max(0.f, 2.f - t / travel_time));
            vec3 velocity = make_vec3(0, 0, 0);
            if (t > 0.f && t < travel_time) {
                velocity = (box_end - box_start) / travel_time;
            } else if (t >= travel_time && t < 2.f * travel_time) {
                velocity = (box_start - box_end) / travel_time;
            }
            simulation.setKinematicBoxPosition(boxID, box_start + fraction * (box_end - box_start), velocity);
            simulation.step();
            max_contacts = std::max(max_contacts, simulation.getContactCount());
            max_penetration = std::max(max_penetration, simulation.getMaxKinematicBoxPenetration());
        }
        physics_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - timer_start).count();

        float displacement = simulation.getMaxLinkDisplacement();
        max_displacement = std::max(max_displacement, displacement);
        if (frame % 15 == 0) {
            std::cout << "t = " << simulation.getSimulationTime() << " s: " << simulation.getContactCount() << " contacts, max link displacement " << displacement << " m" << std::endl;
        }

        if (render_frames) {
            simulation.updateContextGeometry();
            visualizer->plotUpdate(true);
            char filename[64];
            std::snprintf(filename, sizeof(filename), "frames/frame_%04u.jpeg", frame);
            visualizer->printWindow(filename, "jpeg");
        }
        frame++;
    }

    std::cout << "Simulated " << simulation.getSimulationTime() << " s in " << physics_seconds << " s of physics wall time (" << physics_seconds / simulation.getSimulationTime() << "x real time). Peak contacts " << max_contacts
              << ", peak link displacement " << max_displacement << " m, deepest penetration into the box " << 1000.f * max_penetration << " mm." << std::endl;

    return 0;
}
