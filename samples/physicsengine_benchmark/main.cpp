// Measures what a simulated robot-plant step costs: physics at several plant fidelities, the overhead of the physics
// engine plug-in over calling MuJoCo directly, parallel instances, and copying the state to the Context and rendering it.
//
// Usage: ./physicsengine_benchmark [model=apple] [age=275] [physics_steps=500]

#include "PlantPhysics.h"
#include "RadiationModel.h"
#include "Visualizer.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {

    double secondsSince(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    struct Scene {
        Context context;
        PlantArchitecture plantarchitecture;
        uint plantID;
        uint boxID = 0;
        vec3 box_start;
        vec3 box_velocity = make_vec3(0.5f, 0, 0);
        vec3 center;
        float height = 0;
        PhysicsEngine engine;
        double build_seconds = 0;
        double initialize_seconds = 0;

        Scene(const std::string &model, float age, bool rigid_internodes, bool leaf_collisions, bool articulation_collisions) : plantarchitecture(&context), engine(&context) {
            context.seedRandomGenerator(7);
            plantarchitecture.disableMessages();
            engine.disableMessages();
            plantarchitecture.loadPlantModelFromLibrary(model);

            auto timer = std::chrono::steady_clock::now();
            plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), age);
            build_seconds = secondsSince(timer);

            vec3 lower, upper;
            context.getObjectBoundingBox(plantarchitecture.getAllPlantObjectIDs(plantID), lower, upper);
            center = 0.5f * (lower + upper);
            height = upper.z - lower.z;

            USDExportParameters plant_physics;
            plant_physics.elastic_modulus = 5e9f;
            plant_physics.wood_density = 750.f;
            plant_physics.fruit_mass = 0.15f;

            PlantPhysicsParameters plant_parameters;
            plant_parameters.rigid_internodes = rigid_internodes;
            plant_parameters.leaf_collisions = leaf_collisions;
            plant_parameters.leaf_collision_shape = "ellipsoid";

            PhysicsEngineParameters parameters;
            parameters.articulation_collisions = articulation_collisions;

            timer = std::chrono::steady_clock::now();
            addPlantToPhysicsEngine(engine, context, plantarchitecture, plantID, plant_physics, plant_parameters);
            float box_size = 0.1f * height;
            box_start = make_vec3(lower.x - box_size, center.y, lower.z + 0.75f * height);
            boxID = engine.addKinematicBox(box_start, make_vec3(box_size, box_size, box_size));
            engine.initialize(parameters);
            initialize_seconds = secondsSince(timer);
        }

        // Advance the box through the crown at constant speed
        void stepWithBox(uint step_count, uint instance = 0) {
            for (uint s = 0; s < step_count; s++) {
                engine.setKinematicBoxPosition(boxID, box_start + engine.getSimulationTime(instance) * box_velocity, box_velocity, instance);
                engine.step(1, instance);
            }
        }
    };

} // namespace

int main(int argc, char **argv) {

    const std::string model = argc > 1 ? argv[1] : "apple";
    const float age = argc > 2 ? std::stof(argv[2]) : 275.f;
    const uint physics_steps = argc > 3 ? static_cast<uint>(std::stoi(argv[3])) : 500;

    std::cout << "Plant: " << model << ", " << age << " days. Physics time step 2 ms, box sweeping through the crown at 0.5 m/s." << std::endl << std::endl;

    // ---- Physics cost per step at several fidelities ---- //

    struct Fidelity {
        std::string name;
        bool rigid_internodes, leaf_collisions, articulation_collisions;
    };
    const std::vector<Fidelity> fidelities = {
            {"flexible, leaves collide", false, true, false},
            {"flexible, leaves do not collide", false, false, false},
            {"rigid internodes, leaves collide", true, true, false},
            {"rigid internodes, leaves do not collide", true, false, false},
            {"rigid internodes, organs collide with each other", true, true, true},
    };

    std::printf("%-50s %6s %6s %9s %9s %9s %9s\n", "Physics fidelity", "links", "dof", "init(s)", "ms/step", "x realtime", "contacts");
    for (const Fidelity &fidelity: fidelities) {
        Scene scene(model, age, fidelity.rigid_internodes, fidelity.leaf_collisions, fidelity.articulation_collisions);
        const uint settle_steps = static_cast<uint>(0.5f * (scene.center.x - scene.box_start.x) / (scene.box_velocity.x * scene.engine.getTimestep()));
        scene.stepWithBox(settle_steps); // bring the box into the crown, so that the measured steps have contacts
        auto timer = std::chrono::steady_clock::now();
        uint max_contacts = 0;
        for (uint s = 0; s < physics_steps; s++) {
            scene.stepWithBox(1);
            max_contacts = std::max(max_contacts, scene.engine.getContactCount());
        }
        double step_seconds = secondsSince(timer) / double(physics_steps);
        std::printf("%-50s %6u %6d %9.3f %9.3f %9.2f %9u\n", fidelity.name.c_str(), scene.engine.getLinkCount(), static_cast<int>(scene.engine.getMuJoCoModel()->nv), scene.initialize_seconds, 1000. * step_seconds, step_seconds / scene.engine.getTimestep(), max_contacts);
        std::fflush(stdout);
    }

    // ---- Overhead of the plug-in over calling MuJoCo directly, on the same model and trajectory ---- //

    {
        Scene scene(model, age, true, true, false);
        uint plugin_instance = 0;
        uint raw_instance = scene.engine.createInstance();
        const mjModel *mj_model = scene.engine.getMuJoCoModel();
        mjData *mj_data = scene.engine.getMuJoCoData(raw_instance);

        const uint block_steps = 100;
        const uint block_count = 10;
        double plugin_seconds = 0;
        double raw_seconds = 0;
        for (uint block = 0; block < block_count; block++) {
            auto timer = std::chrono::steady_clock::now();
            for (uint s = 0; s < block_steps; s++) {
                scene.engine.step(1, plugin_instance);
            }
            plugin_seconds += secondsSince(timer);

            timer = std::chrono::steady_clock::now();
            for (uint s = 0; s < block_steps; s++) {
                mj_step(mj_model, mj_data);
            }
            raw_seconds += secondsSince(timer);
        }
        std::printf("\nPlug-in step %.4f ms, direct mj_step %.4f ms: %+.2f%% overhead\n", 1000. * plugin_seconds / (block_count * block_steps), 1000. * raw_seconds / (block_count * block_steps), 100. * (plugin_seconds / raw_seconds - 1.));
        std::fflush(stdout);
    }

    // ---- Parallel instances of one model ---- //

    {
        std::cout << std::endl << "Parallel instances (rigid internodes, leaves collide, no box motion)" << std::endl;
        std::printf("%10s %12s %12s\n", "instances", "steps/s", "speedup");
        double single_rate = 0;
        const uint hardware_threads = std::max(1u, std::thread::hardware_concurrency());
        for (uint instance_count: {1u, 2u, 4u, 8u}) {
            if (instance_count > hardware_threads) {
                break;
            }
            Scene scene(model, age, true, true, false);
            while (scene.engine.getInstanceCount() < instance_count) {
                scene.engine.createInstance();
            }
            const uint steps = 200;
            auto timer = std::chrono::steady_clock::now();
            scene.engine.stepInstances(steps, instance_count);
            double rate = double(steps * instance_count) / secondsSince(timer);
            if (instance_count == 1) {
                single_rate = rate;
            }
            std::printf("%10u %12.0f %12.2f\n", instance_count, rate, rate / single_rate);
            std::fflush(stdout);
        }
    }

    // ---- Copying the state to the Context, and rendering it ---- //

    {
        Scene scene(model, age, true, true, false);
        const uint primitive_count = static_cast<uint>(scene.context.getPrimitiveCount());
        std::cout << std::endl << "Rendering " << primitive_count << " primitives; plant built in " << scene.build_seconds << " s" << std::endl;

        const uint steps_per_frame = 16; // about 30 frames per simulated second
        const uint frames = 5;
        vec3 camera_position = scene.center + make_vec3(0.35f * scene.height, -1.1f * scene.height, 0.15f * scene.height);

        double sync_seconds = 0;
        for (uint f = 0; f < frames; f++) {
            scene.stepWithBox(steps_per_frame);
            auto timer = std::chrono::steady_clock::now();
            scene.engine.updateContextGeometry();
            sync_seconds += secondsSince(timer);
        }
        std::printf("%-46s %10.2f ms/frame\n", "Copy state to Context", 1000. * sync_seconds / frames);
        std::fflush(stdout);

        struct RenderSetup {
            std::string name;
            uint width, height;
            int antialiasing_samples;
            Visualizer::LightingModel lighting;
        };
        const std::vector<RenderSetup> render_setups = {
                {"Visualizer, 1280x720, 4x AA, shadows", 1280, 720, 4, Visualizer::LIGHTING_PHONG_SHADOWED},
                {"Visualizer, 640x480, 4x AA, shadows", 640, 480, 4, Visualizer::LIGHTING_PHONG_SHADOWED},
                {"Visualizer, 640x480, 4x AA, no shadows", 640, 480, 4, Visualizer::LIGHTING_PHONG},
                {"Visualizer, 640x480, no AA, no shadows", 640, 480, 1, Visualizer::LIGHTING_PHONG},
        };
        for (const RenderSetup &setup: render_setups) {
            Visualizer visualizer(setup.width, setup.height, setup.antialiasing_samples, false, true);
            visualizer.disableMessages();
            visualizer.setCameraPosition(camera_position, scene.center);
            visualizer.setLightingModel(setup.lighting);
            visualizer.buildContextGeometry(&scene.context);
            visualizer.plotUpdate(true); // the first frame allocates everything, and is not counted
            double render_seconds = 0;
            for (uint f = 0; f < frames; f++) {
                scene.stepWithBox(steps_per_frame);
                scene.engine.updateContextGeometry();
                auto timer = std::chrono::steady_clock::now();
                visualizer.plotUpdate(true);
                render_seconds += secondsSince(timer);
            }
            std::printf("%-46s %10.2f ms/frame\n", setup.name.c_str(), 1000. * render_seconds / frames);
            std::fflush(stdout);
        }

        {
            RadiationModel radiation(&scene.context);
            radiation.disableMessages();
            radiation.addRadiationBand("visible");
            radiation.disableEmission("visible");
            radiation.setDirectRayCount("visible", 100);
            radiation.setDiffuseRayCount("visible", 100);
            radiation.setScatteringDepth("visible", 1);
            uint sun = radiation.addCollimatedRadiationSource(make_vec3(0.3f, -0.5f, 0.8f));
            radiation.setSourceFlux(sun, "visible", 500.f);
            radiation.setDiffuseRadiationFlux("visible", 100.f);
            CameraProperties camera;
            camera.camera_resolution = make_int2(640, 480);
            camera.HFOV = 45.f;
            camera.lens_diameter = 0.f;
            radiation.addRadiationCamera("camera", {"visible"}, camera_position, scene.center, camera, 8);

            auto timer = std::chrono::steady_clock::now();
            radiation.updateGeometry();
            radiation.runBand("visible"); // the first frame builds everything, and is not counted
            double first_seconds = secondsSince(timer);

            double geometry_seconds = 0;
            double trace_seconds = 0;
            for (uint f = 0; f < frames; f++) {
                scene.stepWithBox(steps_per_frame);
                scene.engine.updateContextGeometry();
                timer = std::chrono::steady_clock::now();
                radiation.updateGeometry();
                geometry_seconds += secondsSince(timer);
                timer = std::chrono::steady_clock::now();
                radiation.runBand("visible");
                trace_seconds += secondsSince(timer);
            }
            std::printf("%-46s %10.2f ms/frame\n", "Radiation geometry update (BVH rebuild)", 1000. * geometry_seconds / frames);
            std::printf("%-46s %10.2f ms/frame\n", "Radiation band + camera, 640x480, 8 spp", 1000. * trace_seconds / frames);
            std::printf("%-46s %10.2f ms\n", "Radiation first frame", 1000. * first_seconds);
            std::fflush(stdout);
        }
    }

    return 0;
}
