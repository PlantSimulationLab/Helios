/** \file "render_plant_library.cpp" Renders every plant model of the plant architecture library to an image, for visual assessment. Built and run by utilities/render_plant_library.sh.

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

#include "PlantArchitecture.h"
#include "Visualizer.h"

using namespace helios;

//! What a plant looks like at one age, as far as choosing an age to render it at is concerned
struct PlantState {
    float age = 0;
    float leaf_area = 0;
    size_t flower_count = 0;
    size_t fruit_count = 0;
};

//! Grow a library plant in equal steps, recording its state after each one
/**
 * The same species, seed, step and number of steps always grow the same plant, which is what lets the plant be grown once to find the age to render it at and then again to that age.
 * \param[in] context Context to build the plant in, which must be empty.
 * \param[in] plantarchitecture Plant architecture model using that Context.
 * \param[in] species_name Name of the library plant model.
 * \param[in] random_seed Seed of the Context's random number generator.
 * \param[in] step_days Length of each growth step, in days.
 * \param[in] step_count Number of steps to grow the plant for.
 * \param[out] states State of the plant after each step.
 * \return ID of the plant.
 */
uint growLibraryPlant(Context &context, PlantArchitecture &plantarchitecture, const std::string &species_name, uint random_seed, float step_days, uint step_count, std::vector<PlantState> &states) {
    context.seedRandomGenerator(random_seed);
    plantarchitecture.disableMessages();
    plantarchitecture.loadPlantModelFromLibrary(species_name);
    const uint plantID = plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0);

    states.clear();
    for (uint step = 0; step < step_count; step++) {
        plantarchitecture.advanceTime(plantID, step_days);
        PlantState state;
        state.age = plantarchitecture.getPlantAge(plantID);
        state.leaf_area = plantarchitecture.sumPlantLeafArea(plantID);
        state.flower_count = plantarchitecture.getPlantFlowerObjectIDs(plantID).size();
        state.fruit_count = plantarchitecture.getPlantFruitObjectIDs(plantID).size();
        states.push_back(state);
    }
    return plantID;
}

//! Number of growth steps after which a plant is best looked at: the last time it is in full leaf and, if it ever bears them, carrying fruit or else flowers
/**
 * A plant's maximum age is often a poor age to look at it. A deciduous tree or vine is usually dormant then, since a whole number of years after planting falls in winter, and an annual may have shed its
 * leaves. "In full leaf" is taken as at least three quarters of the largest leaf area the plant ever has. Among those ages, the ones with fruit are preferred, then the ones with flowers, and the latest
 * of what is left is chosen, which is the plant at its largest.
 * \param[in] states State of the plant after each growth step.
 * \return Number of steps to grow the plant for.
 */
uint chooseStepToRender(const std::vector<PlantState> &states) {
    float largest_leaf_area = 0;
    for (const PlantState &state: states) {
        largest_leaf_area = std::max(largest_leaf_area, state.leaf_area);
    }

    // A plant that never has a leaf (a leafless model) is looked at when it is oldest
    std::vector<uint> candidate_steps;
    for (uint step = 0; step < states.size(); step++) {
        if (states.at(step).leaf_area >= 0.75f * largest_leaf_area) {
            candidate_steps.push_back(step);
        }
    }

    auto keepOnly = [&states, &candidate_steps](const std::function<bool(const PlantState &)> &wanted) {
        std::vector<uint> kept_steps;
        for (uint step: candidate_steps) {
            if (wanted(states.at(step))) {
                kept_steps.push_back(step);
            }
        }
        // A preference that nothing satisfies is dropped rather than leaving nothing to choose from
        if (!kept_steps.empty()) {
            candidate_steps = kept_steps;
            return true;
        }
        return false;
    };
    if (!keepOnly([](const PlantState &state) { return state.fruit_count > 0; })) {
        keepOnly([](const PlantState &state) { return state.flower_count > 0; });
    }

    return candidate_steps.back() + 1;
}

// Usage: render_plant_library <output directory> <image width> <image height> <fraction of maximum age> <random seed> <render at final age: 0 or 1> <also render top view: 0 or 1> [species ...]
// With no species given, every plant model in the library is rendered.
int main(int argc, char **argv) {

    if (argc < 8) {
        helios_runtime_error("ERROR (render_plant_library): Expected at least 7 arguments: <output directory> <image width> <image height> <fraction of maximum age> <random seed> <render at final age: 0 or 1> <also render top view: 0 or 1> "
                             "[species ...]. Run it through utilities/render_plant_library.sh.");
    }

    const std::string output_directory = argv[1];
    const int image_width = std::stoi(argv[2]);
    const int image_height = std::stoi(argv[3]);
    const float age_fraction = std::stof(argv[4]);
    const int random_seed = std::stoi(argv[5]);
    const bool render_at_final_age = std::string(argv[6]) == "1";
    const bool render_top_view = std::string(argv[7]) == "1";
    if (image_width <= 0 || image_height <= 0) {
        helios_runtime_error("ERROR (render_plant_library): The image width and height must be positive, but " + std::string(argv[2]) + " x " + std::string(argv[3]) + " was given.");
    }
    if (!(age_fraction > 0.f) || age_fraction > 1.f) {
        helios_runtime_error("ERROR (render_plant_library): The fraction of maximum age must be greater than 0 and no more than 1, but " + std::string(argv[4]) + " was given.");
    }
    if (random_seed < 0) {
        helios_runtime_error("ERROR (render_plant_library): The random seed must not be negative, but " + std::string(argv[5]) + " was given.");
    }

    // Species to render: those named on the command line, or the whole library
    std::vector<std::string> species_names;
    {
        Context library_context;
        PlantArchitecture library(&library_context);
        const std::vector<std::string> available_names = library.getAvailablePlantModels();
        for (int argument = 8; argument < argc; argument++) {
            const std::string requested_name = argv[argument];
            if (std::find(available_names.begin(), available_names.end(), requested_name) == available_names.end()) {
                std::string available_list;
                for (const std::string &name: available_names) {
                    available_list += " " + name;
                }
                helios_runtime_error("ERROR (render_plant_library): There is no plant model named '" + requested_name + "' in the library. The available models are:" + available_list);
            }
            species_names.push_back(requested_name);
        }
        if (species_names.empty()) {
            species_names = available_names;
        }
    }

    // View direction of the main image: from the side and a little above
    const float camera_elevation = deg2rad(20.f);
    const float camera_azimuth = deg2rad(35.f);
    const float camera_field_of_view_degrees = 30.f;

    std::cout << "Rendering " << species_names.size() << " plant model(s) to " << output_directory << std::endl;
    std::cout << "species, maximum age (days), age rendered (days), height (m), trunk radius (m), leaves, leaf area (m^2), flowers, fruit, primitives, time to grow (s)" << std::endl;

    for (const std::string &species_name: species_names) {

        Timer timer;
        timer.tic();

        // The length of the plant's life is read from a plant of its own, so that nothing carries over to the plants grown below
        float life_span_days;
        {
            Context age_context;
            PlantArchitecture age_plantarchitecture(&age_context);
            age_plantarchitecture.disableMessages();
            age_plantarchitecture.loadPlantModelFromLibrary(species_name);
            life_span_days = age_fraction * age_plantarchitecture.getPlantMaxAge(age_plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), 0));
        }

        // Steps short enough to catch a flowering or fruiting period, and at least 40 of them for a plant that lives only a few weeks
        const float step_days = std::min(5.f, life_span_days / 40.f);
        const uint life_step_count = static_cast<uint>(std::ceil(life_span_days / step_days));

        // First the plant's whole life, to find the age to look at it at, unless it is to be looked at at the end of its life
        uint render_step_count = life_step_count;
        if (!render_at_final_age) {
            Context survey_context;
            PlantArchitecture survey_plantarchitecture(&survey_context);
            std::vector<PlantState> life_states;
            growLibraryPlant(survey_context, survey_plantarchitecture, species_name, static_cast<uint>(random_seed), step_days, life_step_count, life_states);
            render_step_count = chooseStepToRender(life_states);
        }

        // Then the same plant again, grown to that age
        Context context;
        PlantArchitecture plantarchitecture(&context);
        std::vector<PlantState> states;
        const uint plantID = growLibraryPlant(context, plantarchitecture, species_name, static_cast<uint>(random_seed), step_days, render_step_count, states);
        const PlantState &rendered_state = states.back();

        const double growth_seconds = timer.toc("mute");

        if (context.getPrimitiveCount() == 0) {
            helios_runtime_error("ERROR (render_plant_library): Plant model '" + species_name + "' has no geometry at an age of " + std::to_string(rendered_state.age) + " days, so there is nothing to render.");
        }

        // Radius of the base of the plant's first shoot, which for a tree is its trunk
        float trunk_radius = 0;
        const std::vector<uint> internode_objIDs = plantarchitecture.getPlantInternodeObjectIDs(plantID);
        if (!internode_objIDs.empty() && context.doesObjectExist(internode_objIDs.front())) {
            trunk_radius = context.getTubeObjectNodeRadii(internode_objIDs.front()).front();
        }

        std::cout << species_name << ", " << life_span_days << ", " << rendered_state.age << ", " << plantarchitecture.getPlantHeight(plantID) << ", " << trunk_radius << ", " << plantarchitecture.getPlantLeafCount(plantID) << ", "
                  << rendered_state.leaf_area << ", " << rendered_state.flower_count << ", " << rendered_state.fruit_count << ", " << context.getPrimitiveCount() << ", " << growth_seconds << std::endl;

        // Frame the plant: look at the center of its bounding box from a distance set by the size of the box
        vec2 x_bounds, y_bounds, z_bounds;
        context.getDomainBoundingBox(x_bounds, y_bounds, z_bounds);
        const vec3 plant_center = make_vec3(0.5f * (x_bounds.x + x_bounds.y), 0.5f * (y_bounds.x + y_bounds.y), 0.5f * (z_bounds.x + z_bounds.y));
        const float bounding_radius = 0.5f * make_vec3(x_bounds.y - x_bounds.x, y_bounds.y - y_bounds.x, z_bounds.y - z_bounds.x).magnitude();
        // The sphere enclosing the bounding box is a loose fit to a plant, which does not reach the corners of its box, so the camera is brought in to 90% of the distance at which that sphere would just fit
        const float camera_distance = 0.9f * bounding_radius / std::sin(0.5f * deg2rad(camera_field_of_view_degrees));
        const vec3 camera_position = plant_center + camera_distance * make_vec3(std::cos(camera_elevation) * std::cos(camera_azimuth), std::cos(camera_elevation) * std::sin(camera_azimuth), std::sin(camera_elevation));

        // A ground for the plant to stand on and cast its shadow onto, added after the plant was framed so that it does not change the framing
        context.addPatch(make_vec3(plant_center.x, plant_center.y, z_bounds.x), make_vec2(8.f * bounding_radius, 8.f * bounding_radius), nullrotation, make_RGBcolor(0.75f, 0.72f, 0.68f));

        Visualizer visualizer(static_cast<uint>(image_width), static_cast<uint>(image_height), 8, false, true);
        visualizer.disableMessages();
        visualizer.hideWatermark();
        visualizer.setBackgroundColor(RGB::white);
        visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
        visualizer.setLightDirection(make_vec3(0.6f, -0.4f, 1.f));
        visualizer.setCameraFieldOfView(camera_field_of_view_degrees);
        visualizer.setCameraPosition(camera_position, plant_center);
        visualizer.buildContextGeometry(&context);

        const std::string image_file = output_directory + "/" + species_name + ".jpg";
        visualizer.printWindow(image_file.c_str());

        if (render_top_view) {
            // Straight down onto the plant, from far enough away to fit its horizontal extent. The camera is set a hair off the vertical, which a camera looking exactly along it cannot be oriented about.
            const float horizontal_radius = 0.5f * make_vec2(x_bounds.y - x_bounds.x, y_bounds.y - y_bounds.x).magnitude();
            const float top_view_distance = 1.05f * horizontal_radius / std::sin(0.5f * deg2rad(camera_field_of_view_degrees)) + 0.5f * (z_bounds.y - z_bounds.x);
            visualizer.setCameraPosition(plant_center + make_vec3(0.001f * top_view_distance, 0, top_view_distance), plant_center);
            const std::string top_image_file = output_directory + "/" + species_name + "_top.jpg";
            visualizer.printWindow(top_image_file.c_str());
        }
    }

    std::cout << "Done." << std::endl;
    return 0;
}
