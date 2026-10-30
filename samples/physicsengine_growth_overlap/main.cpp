// Feasibility test of using MuJoCo to keep the organs of a growing plant from overlapping each other. The plant is
// grown in fixed increments, and after each one its organs are made to collide with each other in MuJoCo and the plant
// is relaxed until it comes to rest. The time this takes is compared with the time spent growing the plant.
//
// Usage: ./physicsengine_growth_overlap [model=apple] [final_age=275] [start_age=165] [increment_days=5] [box|ellipsoid] [rigid|flexible]
// Setting start_age equal to final_age builds the plant at that age and relaxes it once.
//
// The relaxed poses are not written back into the PlantArchitecture model, so each relaxation starts again from the
// plant as it grew and has to resolve every overlap in it, not only those formed since the last increment. The
// relaxation times are therefore an upper bound on what a coupled growth model would spend.

#include "PlantPhysics.h"

using namespace helios;

namespace {
    double secondsSince(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
} // namespace

int main(int argc, char **argv) {

    const std::string plant_model = argc > 1 ? argv[1] : "apple";
    const float final_age = argc > 2 ? std::stof(argv[2]) : 275.f; // days
    const float start_age = argc > 3 ? std::stof(argv[3]) : 165.f; // days; the library apple is dormant until day 165
    const float growth_increment = argc > 4 ? std::stof(argv[4]) : 5.f; // days between relaxations
    const std::string leaf_shape = argc > 5 ? argv[5] : "ellipsoid";
    const bool rigid_internodes = argc > 6 ? std::string(argv[6]) != "flexible" : true;

    const float overlap_tolerance = 0.0002f; // m; shallower contacts are not counted as overlaps
    const float rest_speed = 0.002f; // m/s; the relaxation stops once no link moves faster than this, whether or not overlaps remain
    const uint max_relaxation_steps = 1000;
    const uint steps_between_checks = 10;

    PhysicsEngineParameters simulation_parameters;
    simulation_parameters.articulation_collisions = true;
    PlantPhysicsParameters plant_parameters;
    plant_parameters.leaf_collision_shape = leaf_shape;
    plant_parameters.rigid_internodes = rigid_internodes;
    // The plant as built already includes its sag under its own weight, which is the rest pose of the joints
    simulation_parameters.gravity = false;

    USDExportParameters plant_physics;
    plant_physics.elastic_modulus = 5e9f;
    plant_physics.wood_density = 750.f;
    plant_physics.fruit_mass = 0.15f;

    Context context;
    context.seedRandomGenerator(7);
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary(plant_model);

    auto timer = std::chrono::steady_clock::now();
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), start_age);
    double growth_seconds = secondsSince(timer);
    double total_growth_seconds = growth_seconds;
    double total_compile_seconds = 0;
    double total_relax_seconds = 0;

    std::cout << plant_model << " grown from day " << start_age << " to " << final_age << " in " << growth_increment << "-day increments, " << leaf_shape << " leaves, " << (rigid_internodes ? "rigid" : "flexible") << " internodes" << std::endl;
    std::cout << "  age  links  grow(s)  build(s)  relax(s)  steps  overlaps before (max mm)  after (max mm)  max move (mm)" << std::endl;

    for (float age = start_age; age <= final_age + 0.5f * growth_increment; age += growth_increment) {
        if (age > start_age) {
            timer = std::chrono::steady_clock::now();
            plantarchitecture.advanceTime(plantID, growth_increment);
            growth_seconds = secondsSince(timer);
            total_growth_seconds += growth_seconds;
        }

        timer = std::chrono::steady_clock::now();
        PhysicsEngine simulation(&context);
        simulation.disableMessages();
        addPlantToPhysicsEngine(simulation, context, plantarchitecture, plantID, plant_physics, plant_parameters);
        simulation.initialize(simulation_parameters);
        double compile_seconds = secondsSince(timer);
        total_compile_seconds += compile_seconds;

        simulation.writeMJCF("growth_overlap_model.xml"); // the model of the last increment is kept for inspection
        uint overlaps_before = simulation.getLinkOverlapCount(overlap_tolerance);
        float depth_before = simulation.getMaxLinkOverlapDepth();

        timer = std::chrono::steady_clock::now();
        uint steps = 0;
        while (steps < max_relaxation_steps) {
            simulation.step(steps_between_checks);
            steps += steps_between_checks;
            if (simulation.getMaxLinkSpeed() < rest_speed) {
                break;
            }
        }
        double relax_seconds = secondsSince(timer);
        total_relax_seconds += relax_seconds;

        std::printf("%5.0f  %5u  %7.3f  %8.3f  %8.3f  %5u  %8u (%6.1f)  %8u (%6.1f)  %8.1f\n", age, simulation.getLinkCount(), growth_seconds, compile_seconds, relax_seconds, steps, overlaps_before, 1000.f * depth_before,
                    simulation.getLinkOverlapCount(overlap_tolerance), 1000.f * simulation.getMaxLinkOverlapDepth(), 1000.f * simulation.getMaxLinkDisplacement());
        std::fflush(stdout);
    }

    std::cout << "Total: growth " << total_growth_seconds << " s, MuJoCo model building " << total_compile_seconds << " s, relaxation " << total_relax_seconds << " s ("
              << (total_compile_seconds + total_relax_seconds) / total_growth_seconds << "x the growth time)." << std::endl;

    return 0;
}
