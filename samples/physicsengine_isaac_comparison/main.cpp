// Helios (MuJoCo) side of a comparison with IsaacSim (PhysX) on the same plant.
//
// Grows a plant, writes it to USD for IsaacSim with PlantArchitecture::writePlantStructureUSD(), and runs the same box
// sweep as samples/physicsengine_box_sweep in the physics engine plug-in. The positions of every link over time are written
// in the same format as isaac_box_sweep.py writes them, so that compare.py can compare the two link by link.
//
// Usage: physicsengine_isaac_comparison [key=value ...]
//   plant=apple       plant model from the library
//   age=275           plant age in days
//   seed=7            random seed passed to Context::seedRandomGenerator()
//   dt=0.002          physics time step in s; also sets USDExportParameters::physics_steps_per_second = 1/dt
//   armature_ratio=8  USDExportParameters::armature_stability_ratio, for both the USD export and the MuJoCo armature
//   armature=0        1 to add the joint armature that the USD export always adds (PlantPhysicsParameters::apply_joint_armature)
//   integrator=discrete
//   record=0.02       interval in s between recorded samples
//   out=.             output directory; it must exist
//   usd=1             0 to skip writing the USD file
//   simulate=1        0 to only write the USD file and the link table
//   settle=1          s of settling under gravity before the box moves
//   duration=0        stop after this many simulated s (0 runs the whole sweep)
//   timing=0          1 to read nothing back while stepping and write no trajectory, so that the wall time is for physics alone

#include "PlantPhysics.h"

using namespace helios;

namespace {

    std::map<std::string, std::string> parseArguments(int argc, char **argv) {
        std::map<std::string, std::string> arguments = {{"plant", "apple"}, {"age", "275"}, {"seed", "7"}, {"dt", "0.002"}, {"armature", "0"}, {"armature_ratio", "8"}, {"integrator", "discrete"}, {"record", "0.02"}, {"out", "."}, {"usd", "1"}, {"simulate", "1"}, {"timing", "0"}, {"settle", "1"}, {"duration", "0"}};
        for (int i = 1; i < argc; i++) {
            std::string argument = argv[i];
            size_t equals = argument.find('=');
            if (equals == std::string::npos || arguments.find(argument.substr(0, equals)) == arguments.end()) {
                helios_runtime_error("ERROR (physicsengine_isaac_comparison): Unknown argument \"" + argument + "\". Arguments are key=value, with keys plant, age, seed, dt, armature, armature_ratio, integrator, record, out, usd, simulate, timing, settle, duration.");
            }
            arguments[argument.substr(0, equals)] = argument.substr(equals + 1);
        }
        return arguments;
    }

    std::string shootOf(const std::string &link_name) {
        return link_name.substr(0, link_name.find('_'));
    }

    // Branch order of each link: 0 on the main stem, increasing by one at each change of shoot along the path from the base
    std::vector<int> computeBranchOrders(const std::vector<PlantArticulationLink> &links) {
        std::vector<int> orders(links.size(), -1);
        for (size_t i = 0; i < links.size(); i++) {
            // links are listed parents first
            const int parent = links[i].parent_index;
            if (parent < 0) {
                orders[i] = 0;
            } else {
                orders[i] = orders.at(parent) + (shootOf(links[i].name) != shootOf(links[parent].name) ? 1 : 0);
            }
        }
        return orders;
    }

} // namespace

int main(int argc, char **argv) {

    std::map<std::string, std::string> arguments = parseArguments(argc, argv);
    const std::string plant_model = arguments["plant"];
    const float plant_age = std::stof(arguments["age"]);
    const uint seed = static_cast<uint>(std::stoul(arguments["seed"]));
    const float timestep = std::stof(arguments["dt"]);
    const bool use_armature = arguments["armature"] == "1";
    const bool timing_only = arguments["timing"] == "1";
    const float record_interval = std::stof(arguments["record"]);
    const std::filesystem::path output_directory = arguments["out"];
    if (!std::filesystem::is_directory(output_directory)) {
        helios_runtime_error("ERROR (physicsengine_isaac_comparison): Output directory " + output_directory.string() + " does not exist.");
    }

    // The same sweep as samples/physicsengine_box_sweep
    const float box_height_fraction = 0.75f;
    const float box_size_fraction = 0.1f;
    const float box_speed = 0.5f;
    const float settle_time = std::stof(arguments["settle"]);

    Context context;
    context.seedRandomGenerator(seed);

    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary(plant_model);
    uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), plant_age);

    USDExportParameters plant_physics;
    plant_physics.elastic_modulus = 5e9f;
    plant_physics.wood_density = 750.f;
    plant_physics.fruit_mass = 0.15f;
    plant_physics.physics_steps_per_second = 1.f / timestep;
    plant_physics.armature_stability_ratio = std::stof(arguments["armature_ratio"]);
    const std::string ratio_tag = arguments["armature_ratio"] == "8" ? "" : "_ratio" + arguments["armature_ratio"];

    const std::string run_name = plant_model + "_dt" + arguments["dt"] + (arguments["settle"] == "1" ? "" : "_settle" + arguments["settle"]) + (use_armature ? "_armature" + ratio_tag : "_noarmature");
    if (arguments["usd"] == "1") {
        const std::string usd_file = (output_directory / (plant_model + "_steps" + std::to_string(int(std::round(1.f / timestep))) + ratio_tag + ".usda")).string();
        plantarchitecture.writePlantStructureUSD(plantID, usd_file, plant_physics);
        std::cout << "Wrote " << usd_file << std::endl;
    }

    const PlantArticulation articulation = plantarchitecture.getPlantArticulation(plantID, plant_physics);
    const std::vector<int> branch_orders = computeBranchOrders(articulation.links);
    {
        std::vector<int> joint_of_child(articulation.links.size(), -1);
        for (size_t j = 0; j < articulation.joints.size(); j++) {
            joint_of_child.at(articulation.joints[j].child_index) = static_cast<int>(j);
        }
        std::ofstream link_table(output_directory / (plant_model + "_links.csv"));
        link_table << "index,name,type,parent,branch_order,rest_x,rest_y,rest_z,mass,radius,half_length,stiffness,damping,armature,inertia_x,inertia_y,inertia_z\n";
        link_table << std::setprecision(9);
        for (size_t i = 0; i < articulation.links.size(); i++) {
            const PlantArticulationLink &link = articulation.links[i];
            const int joint = joint_of_child[i];
            link_table << i << "," << link.name << "," << link.type << "," << link.parent_index << "," << branch_orders[i] << "," << link.position.x << "," << link.position.y << "," << link.position.z << "," << link.mass << "," << link.radius << ","
                       << link.half_length << "," << (joint >= 0 ? articulation.joints[joint].stiffness : 0.f) << "," << (joint >= 0 ? articulation.joints[joint].damping : 0.f) << "," << (joint >= 0 ? articulation.joints[joint].armature : 0.f) << ","
                       << link.inertia_diagonal.x << "," << link.inertia_diagonal.y << "," << link.inertia_diagonal.z << "\n";
        }
    }

    vec3 lower, upper;
    context.getObjectBoundingBox(plantarchitecture.getAllPlantObjectIDs(plantID), lower, upper);
    const vec3 center = 0.5f * (lower + upper);
    const float height = upper.z - lower.z;
    const float box_size = box_size_fraction * height;
    const vec3 box_start = make_vec3(lower.x - box_size, center.y, lower.z + box_height_fraction * height);
    const vec3 box_end = make_vec3(upper.x + box_size, center.y, box_start.z);
    const float travel_time = (box_end.x - box_start.x) / box_speed;
    const float sweep_end_time = settle_time + 2.f * travel_time;
    const float total_time = std::stof(arguments["duration"]) > 0.f ? std::stof(arguments["duration"]) : sweep_end_time;

    // The sweep as written for isaac_box_sweep.py, so that both simulators move the box identically
    {
        std::ofstream sweep(output_directory / (plant_model + "_sweep.json"));
        sweep << std::setprecision(9);
        sweep << "{\n  \"plant\": \"" << plant_model << "\", \"age\": " << plant_age << ", \"seed\": " << seed << ",\n";
        sweep << "  \"height\": " << height << ", \"box_size\": " << box_size << ",\n";
        sweep << "  \"box_start\": [" << box_start.x << ", " << box_start.y << ", " << box_start.z << "],\n";
        sweep << "  \"box_end\": [" << box_end.x << ", " << box_end.y << ", " << box_end.z << "],\n";
        sweep << "  \"settle_time\": " << settle_time << ", \"travel_time\": " << travel_time << ", \"total_time\": " << sweep_end_time << ",\n";
        sweep << "  \"link_count\": " << articulation.links.size() << "\n}\n";
    }
    std::cout << plant_model << " at " << plant_age << " days: height " << height << " m, " << articulation.links.size() << " links, " << box_size << " m box at z = " << box_start.z << " m, x = " << box_start.x << " -> " << box_end.x
              << " m, " << total_time << " s in total." << std::endl;

    if (arguments["simulate"] != "1") {
        return 0;
    }

    PlantPhysicsParameters plant_parameters;
    plant_parameters.apply_joint_armature = use_armature;
    PhysicsEngineParameters simulation_parameters;
    simulation_parameters.timestep = timestep;
    simulation_parameters.integrator = arguments["integrator"];
    // MuJoCo cannot resolve a contact faster than two time steps
    simulation_parameters.contact_time_constant = std::max(simulation_parameters.contact_time_constant, 2.f * timestep);

    PhysicsEngine simulation(&context);
    simulation.disableMessages();
    const uint plantBody = addPlantToPhysicsEngine(simulation, context, plantarchitecture, plantID, plant_physics, plant_parameters);
    const uint boxID = simulation.addKinematicBox(box_start, make_vec3(box_size, box_size, box_size));
    simulation.initialize(simulation_parameters);
    const uint link_count = simulation.getLinkCount();

    const std::string trajectory_file = (output_directory / ("mujoco_" + run_name + ".csv")).string();
    std::ofstream trajectory;
    if (!timing_only) {
        trajectory.open(trajectory_file);
        trajectory << "time,box_x,box_y,box_z,force_x,force_y,force_z,peak_force,peak_penetration,contacts";
        for (const PlantArticulationLink &link: articulation.links) {
            trajectory << "," << link.name << "_x," << link.name << "_y," << link.name << "_z";
        }
        trajectory << "\n" << std::setprecision(7);
    }

    auto boxPosition = [&](float t_since_settle) {
        const float fraction = t_since_settle <= 0.f ? 0.f : (t_since_settle < travel_time ? t_since_settle / travel_time : std::max(0.f, 2.f - t_since_settle / travel_time));
        return box_start + fraction * (box_end - box_start);
    };
    auto boxVelocity = [&](float t_since_settle) {
        if (t_since_settle > 0.f && t_since_settle < travel_time) {
            return (box_end - box_start) / travel_time;
        } else if (t_since_settle >= travel_time && t_since_settle < 2.f * travel_time) {
            return (box_start - box_end) / travel_time;
        }
        return make_vec3(0, 0, 0);
    };

    const uint steps_per_record = std::max(1u, static_cast<uint>(std::round(record_interval / timestep)));
    const uint total_steps = static_cast<uint>(std::ceil(total_time / timestep - 1e-6f));
    float peak_force = 0.f;
    float peak_penetration = 0.f;
    uint peak_contacts = 0;
    double physics_seconds = 0;
    std::string divergence;
    vec3 box = box_start; // where the box was during the last step

    auto writeRecord = [&](double time) {
        const vec3 force = simulation.getExternalContactForce(plantBody);
        trajectory << time << "," << box.x << "," << box.y << "," << box.z << "," << force.x << "," << force.y << "," << force.z << "," << peak_force << "," << peak_penetration << "," << peak_contacts;
        for (uint i = 0; i < link_count; i++) {
            const vec3 p = simulation.getLinkPosition(plantBody, i);
            trajectory << "," << p.x << "," << p.y << "," << p.z;
        }
        trajectory << "\n";
        peak_force = 0.f;
        peak_penetration = 0.f;
        peak_contacts = 0;
    };

    if (!timing_only) {
        writeRecord(0.0);
    }
    uint step = 0;
    const auto timer_start = std::chrono::steady_clock::now();
    try {
        while (step < total_steps) {
            for (uint s = 0; s < steps_per_record && step < total_steps; s++, step++) {
                const float t = float(step) * timestep - settle_time;
                box = boxPosition(t);
                simulation.setKinematicBoxPosition(boxID, box, boxVelocity(t));
                simulation.step();
                if (!timing_only) {
                    peak_force = std::max(peak_force, simulation.getExternalContactForce(plantBody).magnitude());
                    peak_penetration = std::max(peak_penetration, simulation.getMaxKinematicBoxPenetration());
                    peak_contacts = std::max(peak_contacts, simulation.getContactCount());
                }
            }
            if (!timing_only) {
                writeRecord(double(step) * timestep);
            }
        }
    } catch (const std::exception &error) {
        // Reported in the summary below and in the run's JSON file, and the program exits with an error
        divergence = error.what();
    }
    physics_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - timer_start).count();
    if (!timing_only) {
        trajectory.close();
    }

    const double simulated_seconds = double(step) * timestep;
    std::ofstream summary(output_directory / ("mujoco_" + run_name + (timing_only ? "_timing" : "") + ".json"));
    summary << std::setprecision(9);
    summary << "{\n  \"engine\": \"mujoco\", \"plant\": \"" << plant_model << "\", \"dt\": " << timestep << ", \"armature\": " << (use_armature ? "true" : "false") << ", \"armature_stability_ratio\": " << plant_physics.armature_stability_ratio << ", \"integrator\": \"" << simulation_parameters.integrator << "\",\n";
    summary << "  \"contact_time_constant\": " << simulation_parameters.contact_time_constant << ", \"links\": " << link_count << ", \"steps\": " << step << ",\n";
    summary << "  \"simulated_seconds\": " << simulated_seconds << ", \"timing_run\": " << (timing_only ? "true" : "false") << ", \"physics_wall_seconds\": " << physics_seconds << ",\n";
    summary << "  \"diverged\": " << (divergence.empty() ? "false" : "true") << ", \"divergence_message\": \"" << (divergence.empty() ? "" : "see console output") << "\"\n}\n";

    if (!divergence.empty()) {
        std::cerr << "Simulation diverged at t = " << simulated_seconds << " s: " << divergence << std::endl;
        return 1;
    }
    std::cout << "Simulated " << simulated_seconds << " s in " << physics_seconds << " s of physics wall time (" << physics_seconds / simulated_seconds << " s per simulated s)." << (timing_only ? "" : " Wrote " + trajectory_file) << std::endl;
    return 0;
}
