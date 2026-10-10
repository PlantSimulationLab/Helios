/** \file "PlantLibrary.cpp" Contains routines for loading and building plant models from a library of predefined plant types.

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

#include "CollisionDetection.h"
#include "PlantArchitecture.h"

using namespace helios;

//! Smooth one-dimensional wander that cannot drift: a sum of sinusoids of given amplitude, wavelength and phase.
/**
 * Used to vary the path of trained wood. Trained wood is held in place -- a trunk by its stake, a cane by the wire it is tied to -- so it departs from its trained line by a bounded amount and keeps returning to it.
 * A random walk does neither, since its excursion grows with the length of the path. A sum of sinusoids never strays further than the sum of its amplitudes however long the path is, which states the constraint
 * directly, and the wavelengths set how quickly it is allowed to turn.
 */
struct BoundedWander {

    //! Add a sinusoidal component.
    /**
     * \param[in] amplitude Largest departure this component contributes, in meters.
     * \param[in] wavelength Distance along the path over which this component repeats, in meters.
     * \param[in] phase Phase of this component at the start of the path, in radians.
     */
    void addComponent(float amplitude, float wavelength, float phase) {
        amplitudes.push_back(amplitude);
        wavelengths.push_back(wavelength);
        phases.push_back(phase);
    }

    //! Departure from the trained line at a given distance along the path.
    /**
     * \param[in] distance Distance along the path, in meters.
     * \return Departure in meters, whose magnitude never exceeds the sum of the component amplitudes.
     */
    [[nodiscard]] float evaluate(float distance) const {
        float departure = 0.f;
        for (size_t component = 0; component < amplitudes.size(); component++) {
            departure += amplitudes.at(component) * std::sin(2.f * PI_F * distance / wavelengths.at(component) + phases.at(component));
        }
        return departure;
    }

private:
    std::vector<float> amplitudes;
    std::vector<float> wavelengths;
    std::vector<float> phases;
};

//! Node path of an almond trunk: a lean plus a slow bounded sway, as seen in orchard trunks.
/**
 * A trunk is built node by node from this path rather than extruded from a single base rotation, which could only ever give a
 * straight, uniformly tilted tube. Measured on the Phytograph QSMs of the 16 Nickels block-1 almonds, from the base to the
 * scaffold fork, trunks lean 1.3-15 degrees (median about 6) and bow away from their base-to-fork chord by 3-15% of its length
 * (median about 6%). The lean is drawn directly; the bow comes from a BoundedWander in each horizontal direction, which keeps
 * the departure bounded however tall the trunk.
 *
 * \param[in] context_ptr Context whose random generator is used.
 * \param[in] base_position Position of the base of the trunk.
 * \param[in] trunk_nodes Number of trunk internodes.
 * \param[in] internode_length Vertical spacing of the trunk nodes (m).
 * \param[in] radius Radius assigned to every node (m); girth growth thickens the trunk from here.
 * \param[out] node_positions trunk_nodes + 1 node positions, base first.
 * \param[out] node_radii One radius per node.
 */
static void almondTrunkPath(helios::Context *context_ptr, const vec3 &base_position, uint trunk_nodes, float internode_length, float radius, std::vector<vec3> &node_positions, std::vector<float> &node_radii) {
    const float trunk_height = float(trunk_nodes) * internode_length;

    const float lean = deg2rad(context_ptr->randu(1.5f, 11.f));
    const float lean_azimuth = context_ptr->randu(0.f, 2.f * PI_F);

    // A slow bend of about half a cycle over the trunk, and a slight kink on top of it. The sway's net displacement between
    // base and top is removed, so it bows the trunk without tilting it: left in, a slow sway adds to the lean, and the
    // model's lean distribution grew a tail well past the 15 degrees of the most-leaning reference trunk.
    BoundedWander sway_x;
    BoundedWander sway_y;
    for (BoundedWander *sway: {&sway_x, &sway_y}) {
        sway->addComponent(context_ptr->randu(0.01f, 0.075f), trunk_height * context_ptr->randu(1.5f, 3.f), context_ptr->randu(0.f, 2.f * PI_F));
        sway->addComponent(context_ptr->randu(0.002f, 0.008f), context_ptr->randu(0.3f, 0.6f), context_ptr->randu(0.f, 2.f * PI_F));
    }

    node_positions.clear();
    node_radii.clear();
    node_positions.reserve(trunk_nodes + 1);
    node_radii.reserve(trunk_nodes + 1);
    for (uint node = 0; node <= trunk_nodes; node++) {
        const float height = float(node) * internode_length;
        const float lean_offset = std::tan(lean) * height;
        const float along = height / trunk_height;
        const float bow_x = sway_x.evaluate(height) - sway_x.evaluate(0.f) - along * (sway_x.evaluate(trunk_height) - sway_x.evaluate(0.f));
        const float bow_y = sway_y.evaluate(height) - sway_y.evaluate(0.f) - along * (sway_y.evaluate(trunk_height) - sway_y.evaluate(0.f));
        const float offset_x = lean_offset * std::cos(lean_azimuth) + bow_x;
        const float offset_y = lean_offset * std::sin(lean_azimuth) + bow_y;
        node_positions.push_back(base_position + make_vec3(offset_x, offset_y, height));
        node_radii.push_back(radius);
    }
}

//! Close the open end of a prescribed woody tube over with a dome.
/**
 * A shoot built from node positions is an open-ended tube. Three more nodes are appended past its last one, along the direction
 * it was heading in, whose radius falls away as a quarter circle.
 *
 * \param[in,out] node_positions Node positions of the shoot, base first; must hold at least two.
 * \param[in,out] node_radii One radius per node.
 */
static void appendWoodDome(std::vector<vec3> &node_positions, std::vector<float> &node_radii) {
    vec3 end_direction = node_positions.back() - node_positions.at(node_positions.size() - 2);
    end_direction.normalize();
    const vec3 end_position = node_positions.back();
    const float end_radius = node_radii.back();
    for (const float dome_angle_degrees: {30.f, 60.f, 85.f}) {
        const float dome_angle = deg2rad(dome_angle_degrees);
        node_positions.push_back(end_position + end_direction * (end_radius * std::sin(dome_angle)));
        node_radii.push_back(end_radius * std::cos(dome_angle));
    }
}

//! Node path and girth of a trained grapevine trunk, from the ground to the head.
/**
 * A trunk is trained up a stake, so it is close to straight: a slow sway that gets through about half a cycle over its height, a
 * slight kink or two on top of that, and a lean of a few centimetres along the row, where nothing holds it. Its girth flares into
 * the root collar, tapers slightly up the trunk, swells at the head and carries shallow lumps from old pruning wounds. The head is
 * closed over with a dome.
 *
 * \param[in] context_ptr Context whose random generator is used.
 * \param[in] base_position Position of the base of the trunk.
 * \param[in] head_height Height of the head above the base (m).
 * \param[in] nominal_radius Radius partway up the trunk (m), varied from vine to vine by 15% either way.
 * \param[out] node_positions Node positions, base first: the nodes up to the head, then those of the dome.
 * \param[out] node_radii One radius per node.
 * \return Number of trunk internodes below the dome; the node at the head is the tip of the last of them.
 */
static uint grapevineTrunkPath(helios::Context *context_ptr, const vec3 &base_position, float head_height, float nominal_radius, std::vector<vec3> &node_positions, std::vector<float> &node_radii) {
    // The trunk bears no shoots, so its nodes are only there to carry its shape and are spaced far more
    // closely than buds would be. The floor keeps enough of them to seat the wood that leaves the head on a
    // very short trunk.
    const uint trunk_nodes = std::max(uint(6), uint(std::ceil(head_height / 0.04f)));
    const float trunk_node_spacing = head_height / float(trunk_nodes);

    // Sway of the trunk axis across (x) and along (y) the row. A slow bend any tighter than this reads as
    // a snake.
    BoundedWander trunk_sway_x;
    BoundedWander trunk_sway_y;
    for (BoundedWander *trunk_sway: {&trunk_sway_x, &trunk_sway_y}) {
        trunk_sway->addComponent(context_ptr->randu(0.015f, 0.04f), context_ptr->randu(1.2f, 2.2f), context_ptr->randu(0.f, 2.f * PI_F));
        trunk_sway->addComponent(context_ptr->randu(0.002f, 0.006f), context_ptr->randu(0.3f, 0.6f), context_ptr->randu(0.f, 2.f * PI_F));
    }
    // The vine is planted where it is planted, so the sway is measured from the base. Across the row the
    // head is tied in close to the trellis, and the lean is whatever gets it there; along the row nothing
    // holds it, and trunks commonly lean a few centimetres toward one neighbour.
    const float head_offset_x = context_ptr->randu(-0.015f, 0.015f);
    const float trunk_lean_x = (head_offset_x - (trunk_sway_x.evaluate(head_height) - trunk_sway_x.evaluate(0.f))) / head_height;
    const float trunk_lean_y = context_ptr->randu(-0.06f, 0.06f);

    const float trunk_radius = nominal_radius * context_ptr->randu(0.85f, 1.15f);
    const float head_swelling = context_ptr->randu(0.3f, 0.6f);
    BoundedWander trunk_lumps;
    trunk_lumps.addComponent(context_ptr->randu(0.03f, 0.06f), context_ptr->randu(0.12f, 0.3f), context_ptr->randu(0.f, 2.f * PI_F));

    // The dome over the head follows the trunk's own lean and sway, with a radius falling away as a quarter circle
    const std::vector<float> dome_angles_degrees = {30.f, 60.f, 85.f};

    node_positions.clear();
    node_radii.clear();
    node_positions.reserve(trunk_nodes + 1 + dome_angles_degrees.size());
    node_radii.reserve(trunk_nodes + 1 + dome_angles_degrees.size());
    for (uint node = 0; node <= trunk_nodes + dome_angles_degrees.size(); node++) {
        float height = float(node) * trunk_node_spacing;
        float dome_radius_fraction = 1.f;
        if (node > trunk_nodes) {
            const float dome_angle = deg2rad(dome_angles_degrees.at(node - trunk_nodes - 1));
            height = head_height + node_radii.at(trunk_nodes) * std::sin(dome_angle);
            dome_radius_fraction = std::cos(dome_angle);
        }
        const float girth_height = std::min(height, head_height); // the dome scales the radius at the top of the head
        const float root_flare = 0.35f * std::exp(-girth_height / 0.05f);
        const float taper = -0.18f * girth_height / head_height;
        const float head_distance = (girth_height - (head_height - 0.02f)) / 0.06f;
        const float head = head_swelling * std::exp(-head_distance * head_distance);
        const float radius = trunk_radius * (1.f + root_flare + taper + head) * (1.f + trunk_lumps.evaluate(girth_height));

        const float offset_x = trunk_lean_x * height + trunk_sway_x.evaluate(height) - trunk_sway_x.evaluate(0.f);
        const float offset_y = trunk_lean_y * height + trunk_sway_y.evaluate(height) - trunk_sway_y.evaluate(0.f);
        node_positions.push_back(base_position + make_vec3(offset_x, offset_y, height));
        node_radii.push_back(radius * dome_radius_fraction);
    }

    return trunk_nodes;
}

float PlantArchitecture::getParameterValue(const std::map<std::string, float> &build_parameters, const std::string &parameter_name, float default_value, float min_value, float max_value, const std::string &parameter_description) const {

    // Check if parameter is specified in the map
    auto it = build_parameters.find(parameter_name);
    if (it == build_parameters.end()) {
        // Parameter not specified, use default
        return default_value;
    }

    float value = it->second;

    // Validate parameter is within valid range
    if (value < min_value || value > max_value) {
        helios_runtime_error("ERROR (PlantArchitecture::getParameterValue): build parameter '" + parameter_name + "' (" + parameter_description + ") has value " + std::to_string(value) + " which is outside the valid range [" +
                             std::to_string(min_value) + ", " + std::to_string(max_value) + "].");
    }

    return value;
}

void PlantArchitecture::initializePlantModelRegistrations() {
    // Register all available plant models
    registerPlantModel("almond", [this]() { initializeAlmondTreeShoots(); }, [this](const helios::vec3 &pos) { return buildAlmondTree(pos); }, "tree");

    registerPlantModel("almond_aldrich", [this]() { initializeAlmondTreeAldrichShoots(); }, [this](const helios::vec3 &pos) { return buildAlmondTreeAldrich(pos); }, "tree");

    registerPlantModel("almond_independence", [this]() { initializeAlmondTreeIndependenceShoots(); }, [this](const helios::vec3 &pos) { return buildAlmondTreeIndependence(pos); }, "tree");

    registerPlantModel("almond_wood_colony", [this]() { initializeAlmondTreeWoodColonyShoots(); }, [this](const helios::vec3 &pos) { return buildAlmondTreeWoodColony(pos); }, "tree");

    registerPlantModel("apple", [this]() { initializeAppleTreeShoots(); }, [this](const helios::vec3 &pos) { return buildAppleTree(pos); }, "tree");

    registerPlantModel("apple_fruitingwall", [this]() { initializeAppleFruitingWallShoots(); }, [this](const helios::vec3 &pos) { return buildAppleFruitingWall(pos); }, "tree");

    registerPlantModel("asparagus", [this]() { initializeAsparagusShoots(); }, [this](const helios::vec3 &pos) { return buildAsparagusPlant(pos); });

    registerPlantModel("bindweed", [this]() { initializeBindweedShoots(); }, [this](const helios::vec3 &pos) { return buildBindweedPlant(pos); }, "weed");

    registerPlantModel("bean", [this]() { initializeBeanShoots(); }, [this](const helios::vec3 &pos) { return buildBeanPlant(pos); });

    registerPlantModel("bougainvillea", [this]() { initializeBougainvilleaShoots(); }, [this](const helios::vec3 &pos) { return buildBougainvilleaPlant(pos); });

    registerPlantModel("capsicum", [this]() { initializeCapsicumShoots(); }, [this](const helios::vec3 &pos) { return buildCapsicumPlant(pos); });

    registerPlantModel("cheeseweed", [this]() { initializeCheeseweedShoots(); }, [this](const helios::vec3 &pos) { return buildCheeseweedPlant(pos); }, "weed");

    registerPlantModel("cowpea", [this]() { initializeCowpeaShoots(); }, [this](const helios::vec3 &pos) { return buildCowpeaPlant(pos); }, "herbaceous", helios::make_vec2(1.398f, 1.574f));

    registerPlantModel("grapevine_VSP", [this]() { initializeGrapevineVSPShoots(); }, [this](const helios::vec3 &pos) { return buildGrapevineVSP(pos); });

    registerPlantModel("grapevine_Wye", [this]() { initializeGrapevineWyeShoots(); }, [this](const helios::vec3 &pos) { return buildGrapevineWye(pos); });

    registerPlantModel("groundcherryweed", [this]() { initializeGroundCherryWeedShoots(); }, [this](const helios::vec3 &pos) { return buildGroundCherryWeedPlant(pos); }, "weed");

    registerPlantModel("maize", [this]() { initializeMaizeShoots(); }, [this](const helios::vec3 &pos) { return buildMaizePlant(pos); });

    registerPlantModel("olive", [this]() { initializeOliveTreeShoots(); }, [this](const helios::vec3 &pos) { return buildOliveTree(pos); }, "tree");

    registerPlantModel("pistachio", [this]() { initializePistachioTreeShoots(); }, [this](const helios::vec3 &pos) { return buildPistachioTree(pos); }, "tree", helios::make_vec2(1.32f, 1.85f));

    registerPlantModel("puncturevine", [this]() { initializePuncturevineShoots(); }, [this](const helios::vec3 &pos) { return buildPuncturevinePlant(pos); }, "weed");

    registerPlantModel("easternredbud", [this]() { initializeEasternRedbudShoots(); }, [this](const helios::vec3 &pos) { return buildEasternRedbudPlant(pos); }, "tree", helios::make_vec2(1.00f, 2.20f));

    registerPlantModel("rice", [this]() { initializeRiceShoots(); }, [this](const helios::vec3 &pos) { return buildRicePlant(pos); });

    registerPlantModel("butterlettuce", [this]() { initializeButterLettuceShoots(); }, [this](const helios::vec3 &pos) { return buildButterLettucePlant(pos); });

    registerPlantModel("sorghum", [this]() { initializeSorghumShoots(); }, [this](const helios::vec3 &pos) { return buildSorghumPlant(pos); });

    registerPlantModel("soybean", [this]() { initializeSoybeanShoots(); }, [this](const helios::vec3 &pos) { return buildSoybeanPlant(pos); });

    registerPlantModel("strawberry", [this]() { initializeStrawberryShoots(); }, [this](const helios::vec3 &pos) { return buildStrawberryPlant(pos); });

    registerPlantModel("sugarbeet", [this]() { initializeSugarbeetShoots(); }, [this](const helios::vec3 &pos) { return buildSugarbeetPlant(pos); });

    registerPlantModel("tomato", [this]() { initializeTomatoShoots(); }, [this](const helios::vec3 &pos) { return buildTomatoPlant(pos); });

    registerPlantModel("cherrytomato", [this]() { initializeCherryTomatoShoots(); }, [this](const helios::vec3 &pos) { return buildCherryTomatoPlant(pos); });

    registerPlantModel("walnut", [this]() { initializeWalnutTreeShoots(); }, [this](const helios::vec3 &pos) { return buildWalnutTree(pos); }, "tree");

    registerPlantModel("wheat", [this]() { initializeWheatShoots(); }, [this](const helios::vec3 &pos) { return buildWheatPlant(pos); });
}

void PlantArchitecture::loadPlantModelFromLibrary(const std::string &plant_label) {
    if (skipStructuralOperationInCallback("loadPlantModelFromLibrary")) {
        return;
    }

    current_plant_model = plant_label;
    initializeDefaultShoots(plant_label);
}

std::vector<std::string> PlantArchitecture::getAvailablePlantModels() const {
    std::vector<std::string> models;
    models.reserve(shoot_initializers.size());
    for (const auto &pair: shoot_initializers) {
        models.push_back(pair.first);
    }
    return models;
}

uint PlantArchitecture::buildPlantInstanceFromLibrary(const helios::vec3 &base_position, float age, const std::map<std::string, float> &build_parameters) {
    rejectStructuralOperationInCallback("buildPlantInstanceFromLibrary");
    StructuralOperationScope structural_operation_scope(this);

    if (current_plant_model.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): current plant model has not been initialized from library. You must call loadPlantModelFromLibrary() first.");
    }

    // Find the plant builder function
    auto builder_it = plant_builders.find(current_plant_model);
    if (builder_it == plant_builders.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): plant label of " + current_plant_model + " does not exist in the library.");
    }

    // Set current build parameters so builder functions can access them
    current_build_parameters = build_parameters;

    // Call the builder function
    uint plantID = builder_it->second(base_position);

    // Clear build parameters after use
    current_build_parameters.clear();

    plant_instances.at(plantID).plant_name = current_plant_model;

    // Rename organ materials that were created with the temporary "custom" plant name during the
    // build (geometry is constructed before plant_name is set, so renameAutoMaterial() in
    // PlantArchitecture.cpp tags them "custom_<organ>"). Rewrite the "custom_" prefix to the model
    // name so materials get descriptive labels like "bean_trifoliate_leaf".
    if (current_plant_model != "custom") {
        std::vector<std::string> material_labels = context_ptr->listMaterials();
        std::string old_prefix = "custom_";
        for (const auto &label: material_labels) {
            if (label.substr(0, old_prefix.size()) == old_prefix) {
                std::string new_label = current_plant_model + "_" + label.substr(old_prefix.size());
                if (!context_ptr->doesMaterialExist(new_label)) {
                    context_ptr->renameMaterial(label, new_label);
                }
            }
        }
    }

    // Update plant_name object data if enabled (geometry was built with temporary "custom" name)
    if (output_object_data.at("plant_name")) {
        std::vector<uint> plant_primitives = getAllPlantObjectIDs(plantID);
        for (uint objID: plant_primitives) {
            if (context_ptr->doesObjectDataExist(objID, "plant_name")) {
                context_ptr->setObjectData(objID, "plant_name", current_plant_model);
            }
        }
    }

    // Set plant_type object data if enabled
    if (output_object_data.at("plant_type")) {
        std::string plant_type = "herbaceous"; // Default type
        auto type_it = plant_type_map.find(current_plant_model);
        if (type_it != plant_type_map.end()) {
            plant_type = type_it->second;
        }
        std::vector<uint> plant_primitives = getAllPlantObjectIDs(plantID);
        context_ptr->setObjectData(plant_primitives, "plant_type", plant_type);
    }

    // Register plant with per-tree BVH collision detection if enabled
    if (collision_detection_enabled && collision_detection_ptr != nullptr && collision_detection_ptr->isTreeBasedBVHEnabled()) {
        std::vector<uint> plant_primitives = getPlantCollisionRelevantObjectIDs(plantID);
        if (!plant_primitives.empty()) {
            collision_detection_ptr->registerTree(plantID, plant_primitives);
        }
    }

    // Empirical leaf inclination distribution declared for this species by registerPlantModel(), if any.
    //
    // Applied here, before the growth below, rather than after it. advanceTime() is where a plant built at an
    // age grows every leaf it has, so steering switched on afterward would find the plant already fully
    // leaved and have nothing left to act on -- tracking only steers leaves that are still expanding. The
    // feature would then be a no-op for buildPlantInstanceFromLibrary(position, age), which is how plants are
    // usually built.
    //
    // A model that declares no distribution is absent from the map and takes no part in this, so it consumes
    // no random draws and grows exactly as it did before the map existed.
    auto leaf_inclination_it = plant_leaf_inclination_distribution_map.find(current_plant_model);
    if (leaf_inclination_it != plant_leaf_inclination_distribution_map.end()) {
        enablePlantLeafElevationAngleDistributionTracking(plantID, leaf_inclination_it->second.x, leaf_inclination_it->second.y, library_leaf_angle_lambda_degrees);
    }

    if (age > 0) {
        advanceTime(plantID, age);
    }

    return plantID;
}

ShootParameters PlantArchitecture::getCurrentShootParameters(const std::string &shoot_type_label) {

    if (shoot_types.find(shoot_type_label) == shoot_types.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getCurrentShootParameters): shoot type label of " + shoot_type_label + " does not exist in the current shoot parameters.");
    }

    return shoot_types.at(shoot_type_label);
}

std::map<std::string, ShootParameters> PlantArchitecture::getCurrentShootParameters() {
    if (shoot_types.empty()) {
        std::cerr
                << "WARNING (PlantArchitecture::getCurrentShootParameters): No plant models have been loaded. You need to first load a plant model from the library (see loadPlantModelFromLibrary()) or manually add shoot parameters (see updateCurrentShootParameters())."
                << std::endl;
    }
    return shoot_types;
}

std::map<std::string, PhytomerParameters> PlantArchitecture::getCurrentPhytomerParameters() {
    if (shoot_types.empty()) {
        std::cerr
                << "WARNING (PlantArchitecture::getCurrentPhytomerParameters): No plant models have been loaded. You need to first load a plant model from the library (see loadPlantModelFromLibrary()) or manually add shoot parameters (see updateCurrentShootParameters())."
                << std::endl;
    }
    std::map<std::string, PhytomerParameters> phytomer_parameters;
    for (const auto &type: shoot_types) {
        phytomer_parameters[type.first] = type.second.phytomer_parameters;
    }
    return phytomer_parameters;
}

std::vector<std::string> PlantArchitecture::listShootTypeLabels() const {
    if (shoot_types.empty()) {
        helios_runtime_error(
                "ERROR (PlantArchitecture::listShootTypeLabels): No plant model has been loaded. You must call loadPlantModelFromLibrary() first, or use listShootTypeLabels(plant_model_name) to query a specific model, or use listShootTypeLabels(plantID) to query a plant instance.");
    }

    std::vector<std::string> labels;
    labels.reserve(shoot_types.size());

    for (const auto &pair: shoot_types) {
        labels.push_back(pair.first);
    }

    return labels;
}

std::vector<std::string> PlantArchitecture::listShootTypeLabels(const std::string &plant_model_name) {
    // Validate model exists before modifying state
    auto init_it = shoot_initializers.find(plant_model_name);
    if (init_it == shoot_initializers.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::listShootTypeLabels): plant model '" + plant_model_name + "' does not exist in the library. Use getAvailablePlantModels() to see available plant models.");
    }

    // Save current state
    std::string saved_plant_model = current_plant_model;
    std::map<std::string, ShootParameters> saved_shoot_types = shoot_types;

    // Load target plant model to extract shoot types
    try {
        initializeDefaultShoots(plant_model_name);

        // Extract shoot type labels
        std::vector<std::string> labels;
        labels.reserve(shoot_types.size());
        for (const auto &pair: shoot_types) {
            labels.push_back(pair.first);
        }

        // Restore original state
        current_plant_model = saved_plant_model;
        shoot_types = saved_shoot_types;

        return labels;
    } catch (...) {
        // Restore state even on exception
        current_plant_model = saved_plant_model;
        shoot_types = saved_shoot_types;
        throw;
    }
}

void PlantArchitecture::updateCurrentShootParameters(const std::string &shoot_type_label, const ShootParameters &params) {
    shoot_types[shoot_type_label] = params;
}

void PlantArchitecture::updateCurrentShootParameters(const std::map<std::string, ShootParameters> &params) {
    if (skipStructuralOperationInCallback("updateCurrentShootParameters")) {
        return;
    }
    shoot_types = params;
}

void PlantArchitecture::initializeDefaultShoots(const std::string &plant_label) {

    // Clear existing shoot types to prevent contamination between plant models
    shoot_types.clear();

    // Find the shoot initializer function
    auto init_it = shoot_initializers.find(plant_label);
    if (init_it == shoot_initializers.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::loadPlantModelFromLibrary): plant label of " + plant_label + " does not exist in the library.");
    }

    // Call the initializer function
    init_it->second();
}

void PlantArchitecture::registerPlantModel(const std::string &name, std::function<void()> shoot_init, std::function<uint(const helios::vec3 &)> plant_build, const std::string &plant_type, const helios::vec2 &leaf_inclination_Beta) {
    shoot_initializers[name] = shoot_init;
    plant_builders[name] = plant_build;
    plant_type_map[name] = plant_type;

    // A model either declares a measured leaf inclination distribution or it does not. Half of one is always a
    // mistake -- a value typed into the wrong component, or a parameter left behind -- and silently storing it
    // would either throw later from deep inside the build with a message pointing at the wrong place, or, for a
    // zero mu, be rejected as a bad Beta distribution by a caller who never wrote one.
    const bool declares_distribution = leaf_inclination_Beta.x > 0.f && leaf_inclination_Beta.y > 0.f;
    const bool declares_nothing = leaf_inclination_Beta.x == 0.f && leaf_inclination_Beta.y == 0.f;
    if (!declares_distribution && !declares_nothing) {
        helios_runtime_error("ERROR (PlantArchitecture::registerPlantModel): Plant model '" + name + "' declares an incomplete leaf inclination distribution of (" + std::to_string(leaf_inclination_Beta.x) + ", " +
                             std::to_string(leaf_inclination_Beta.y) + "). Both Beta parameters must be positive to declare a distribution, or both zero to declare none.");
    }
    if (declares_distribution) {
        plant_leaf_inclination_distribution_map[name] = leaf_inclination_Beta;
    }
}

helios::vec2 PlantArchitecture::getPlantModelLeafInclinationDistribution(const std::string &plant_model_name) const {
    if (plant_builders.find(plant_model_name) == plant_builders.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantModelLeafInclinationDistribution): plant label of " + plant_model_name + " does not exist in the library.");
    }

    auto distribution_it = plant_leaf_inclination_distribution_map.find(plant_model_name);
    if (distribution_it == plant_leaf_inclination_distribution_map.end()) {
        return make_vec2(0.f, 0.f);
    }
    return distribution_it->second;
}

void PlantArchitecture::setPlantModelLeafInclinationDistribution(const std::string &plant_model_name, float Beta_mu_inclination, float Beta_nu_inclination) {
    if (plant_builders.find(plant_model_name) == plant_builders.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantModelLeafInclinationDistribution): plant label of " + plant_model_name + " does not exist in the library.");
    }

    // Same all-or-nothing rule as registerPlantModel(): (0,0) clears the distribution, two positive values set
    // one, and anything else is a half-filled entry.
    const bool sets_distribution = Beta_mu_inclination > 0.f && Beta_nu_inclination > 0.f;
    const bool clears_distribution = Beta_mu_inclination == 0.f && Beta_nu_inclination == 0.f;
    if (!sets_distribution && !clears_distribution) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantModelLeafInclinationDistribution): Beta distribution parameters of (" + std::to_string(Beta_mu_inclination) + ", " + std::to_string(Beta_nu_inclination) +
                             ") are not a valid leaf inclination distribution. Both parameters must be positive to set a distribution, or both zero to clear it.");
    }

    if (clears_distribution) {
        plant_leaf_inclination_distribution_map.erase(plant_model_name);
    } else {
        plant_leaf_inclination_distribution_map[plant_model_name] = make_vec2(Beta_mu_inclination, Beta_nu_inclination);
    }
}

bool PlantArchitecture::doesPlantModelDeclareLeafInclinationDistribution(const std::string &plant_model_name) const {
    if (plant_builders.find(plant_model_name) == plant_builders.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::doesPlantModelDeclareLeafInclinationDistribution): plant label of " + plant_model_name + " does not exist in the library.");
    }

    return plant_leaf_inclination_distribution_map.find(plant_model_name) != plant_leaf_inclination_distribution_map.end();
}

void PlantArchitecture::initializeAlmondTreeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "AlmondLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.33f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature = 0.05;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 1;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_almond(context_ptr->getRandomGenerator());

    // Node-scale roughness: the small, discrete kink applied at each phytomer base. This is the fine-texture scale, distinct from
    // the larger-scale wander produced by the tortuosity random walk along the internode.
    //
    // The distribution is symmetric about zero rather than about a positive value. The kink direction is already scattered around
    // the stem by the phyllotactic angle, so a signed magnitude lets successive kinks partially cancel: this yields the same net
    // wander as a positive-mean distribution of twice the width, but spends roughly half the total bending to get there, leaving
    // the directed gravitropic response rather than accumulated node kinks to set the overall branch trajectory. A constant pitch
    // traces a near-perfect helix (the kinks cancel almost exactly) and reads as a smooth systematic bend, not as texture.
    //
    // Note this is a phenomenological roughness term. A negative pitch has no specific botanical reading on its own; the
    // mechanistic direction of nodal deflection is carried by the phyllotactic rotation of the axis it is applied about.
    phytomer_parameters_almond.internode.pitch.uniformDistribution(-7, 7);
    phytomer_parameters_almond.internode.phyllotactic_angle.uniformDistribution(120, 160);
    phytomer_parameters_almond.internode.radius_initial = 0.002;
    phytomer_parameters_almond.internode.length_segments = 1;
    phytomer_parameters_almond.internode.image_texture = "AlmondBark.jpg";
    phytomer_parameters_almond.internode.max_floral_buds_per_petiole = 1; //

    phytomer_parameters_almond.petiole.petioles_per_internode = 1;
    phytomer_parameters_almond.petiole.pitch.uniformDistribution(-145, -90);
    phytomer_parameters_almond.petiole.taper = 0.1;
    phytomer_parameters_almond.petiole.curvature = 0;
    phytomer_parameters_almond.petiole.length = 0.04;
    phytomer_parameters_almond.petiole.radius = 0.0005;
    phytomer_parameters_almond.petiole.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;
    phytomer_parameters_almond.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_almond.leaf.leaves_per_petiole = 1;
    phytomer_parameters_almond.leaf.roll.uniformDistribution(-10, 10);
    phytomer_parameters_almond.leaf.prototype_scale = 0.12;
    phytomer_parameters_almond.leaf.prototype = leaf_prototype;

    phytomer_parameters_almond.peduncle.length = 0.002;
    phytomer_parameters_almond.peduncle.radius = 0.0005;
    phytomer_parameters_almond.peduncle.pitch = 80;
    phytomer_parameters_almond.peduncle.roll = 90;
    phytomer_parameters_almond.peduncle.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;

    phytomer_parameters_almond.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_almond.inflorescence.pitch = 0;
    phytomer_parameters_almond.inflorescence.roll = 0;
    phytomer_parameters_almond.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
    phytomer_parameters_almond.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.fruit_prototype_function = AlmondFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.005;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 24;
    // The trunk node count is derived in buildAlmondTree() as trunk_height / 0.03, and trunk_height is
    // accepted over 0.1-3.0 m. The cap must therefore cover the whole of that range, otherwise any
    // trunk_height above 0.6 m throws out of addBaseStemShoot(). The trunk apical meristem is killed
    // immediately after the plant is built, so this only sizes the guard; it does not permit extra growth.
    shoot_parameters_trunk.max_nodes = 101;
    shoot_parameters_trunk.girth_area_factor = 0.85f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 10;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.internode.radial_subdivisions = 5;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = AlmondSpurPhytomerCallbackFunction;
    // Calibrated against a Phytograph QSM reconstruction of a seven-year-old leaf-off almond (see
    // projects/QSMCalibration). Capping the lifetime node count well below the old value of 60 stops a
    // proleptic shoot from extending indefinitely across seasons; the reference tree carries its wood as
    // many short branches rather than a few long ones, and the uncapped shoot produced branches roughly
    // 1.5x too long.
    shoot_parameters_proleptic.max_nodes.uniformDistribution(20, 55);
    shoot_parameters_proleptic.max_nodes_per_season = 15;
    shoot_parameters_proleptic.phyllochron_min = 1;
    shoot_parameters_proleptic.elongation_rate_max = 0.3;
    shoot_parameters_proleptic.girth_area_factor = 1.8f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.05;
    // Scaled by 0.85 when the whole-plant leaf-area-index and shadow-grid bud-break terms were removed from the model: at
    // almond's canopy density those two had been multiplying every bud-break probability by roughly this much. Matched on
    // Nonpareil, 0.85 on the proleptic and scaffold maxima restores total woody length with crown shape unchanged.
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.31;
    // The decay is linear in nodes from the tip: p = max(p_min, p_max - decay * nodes_from_tip). At the
    // old rate of 0.15 the probability fell from 0.5 to the 0.1 floor within three nodes, so effectively
    // the whole shoot broke buds at the minimum and the tree ramified at about half the observed rate.
    // Spreading the decay over roughly eight nodes brings forks per metre into line with the QSM tree.
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    // Sets how fast a lateral reorients toward vertical, in degrees per metre of arc. At the old value of
    // 250 a branch that emerged near-horizontal was back to near-vertical within 10-25 cm, so laterals in
    // the lower crown ran upward rather than outward: they piled on woody length and branch count without
    // contributing any crown width, which read as a narrow-based, flare-topped (cone/mushroom) canopy.
    //
    // Measured against the QSM reference, order-2 branches emerge at a sensible angle in both trees
    // (71 deg from vertical in the model, 64 deg in the reference) but the model then turned 24 deg along
    // its length against the reference's 7 deg. Halving the response brings crown radius from 1.8 to 2.0 m
    // (reference 2.5 m) and the upper/lower crown-radius ratio from 1.24 to 1.06 (reference 1.14).
    //
    // Note this is the proleptic value only. The scaffold and sylleptic types are copy-constructed from
    // these parameters but both assign their own gravitropic_curvature below, so neither inherits it.
    // Gravitropic curvatures raised 20% across all shoot types from renders, which showed branches needing a little more upward
    // curvature throughout the crown.
    shoot_parameters_proleptic.gravitropic_curvature = 180;
    shoot_parameters_proleptic.tortuosity = 37.5;
    shoot_parameters_proleptic.tortuosity_persistence_length = 0.3;
    shoot_parameters_proleptic.insertion_angle_tip = 45;
    shoot_parameters_proleptic.insertion_angle_decay_rate = 15;
    shoot_parameters_proleptic.internode_length_max = 0.04;
    shoot_parameters_proleptic.internode_length_min = 0.002;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.4;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 3;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic", "sylleptic"}, {1.0, 0.});

    // Sylleptic shoots
    ShootParameters shoot_parameters_sylleptic = shoot_parameters_proleptic;
    //    shoot_parameters_sylleptic.phytomer_parameters.internode.color = RGB::red;
    shoot_parameters_sylleptic.phytomer_parameters.internode.image_texture = "";
    shoot_parameters_sylleptic.phytomer_parameters.leaf.prototype_scale = 0.14;
    shoot_parameters_sylleptic.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20);
    shoot_parameters_sylleptic.insertion_angle_tip = 0;
    shoot_parameters_sylleptic.insertion_angle_decay_rate = 0;
    shoot_parameters_sylleptic.phyllochron_min = 1;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_min = 0.0;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_max = 0.7;
    shoot_parameters_sylleptic.gravitropic_curvature = 720;
    shoot_parameters_sylleptic.internode_length_max = 0.02;
    shoot_parameters_sylleptic.tortuosity = 75;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true;
    shoot_parameters_sylleptic.defineChildShootTypes({"proleptic"}, {1.0});
    // Sylleptic shoots are copied from the proleptic parameters above, so they would otherwise inherit the
    // QSM-calibrated node cap and bud-break decay. Those were validated for proleptic shoots only, so the
    // previous values are restored here rather than being changed as an invisible side effect of the
    // calibration. (The almond model gives sylleptic a child-type probability of zero, so this is a
    // correctness measure and does not alter the generated tree.)
    shoot_parameters_sylleptic.max_nodes = 60;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_decay_rate = 0.15;

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    //    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 105;
    shoot_parameters_scaffold.gravitropic_curvature = 276;
    // shoot_parameters_scaffold.internode_length_max = 0.04;
    shoot_parameters_scaffold.tortuosity = 10;
    shoot_parameters_scaffold.tortuosity_persistence_length = 2.5;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});
    // Likewise for the scaffolds: the calibration applies to proleptic shoots. The scaffolds already pin
    // their own max_nodes above, so only the bud-break decay needs restoring.
    // Positive: acrotonic, with bud-break vigour rising toward the apex. Acrotony is "the increase in vigor
    // (length, diameter, number of leaves) of vegetative branches from the bottom to the top position of the
    // parent growth unit" and is the pattern that governs tree development, whereas basitony -- stronger growth
    // at the base, which a negative rate encodes -- characterises shrubs (Costes et al., Front. Plant Sci.
    // 5:666, 2014; Wilson, Am. J. Bot. 87:601, 2000). The previous -0.01 concentrated scaffold recruitment at
    // the trunk junction, which is where the long structural branches that distorted the crown were emerging.
    shoot_parameters_scaffold.vegetative_bud_break_probability_decay_rate = 0.02;
    shoot_parameters_scaffold.vegetative_bud_break_probability_max = 0.72;
    shoot_parameters_scaffold.vegetative_bud_break_probability_min = 0.10;
    // Girth is area = girth_area_factor * downstream_leaf_area, and downstream leaf area partitions exactly
    // across a fork, so with a single factor the pipe model conserves cross-sectional area through a
    // junction. Inheriting the proleptic value of 6 while the trunk uses 10 broke that at the one junction
    // where it matters most: measured at the trunk/scaffold fork, the summed scaffold area came to exactly
    // 0.600 of the trunk's -- the 6/10 parameter ratio -- discarding 40% of the cross-section, while every
    // other junction in the tree conserved area to within the reference's own scatter.
    //
    // The QSM reference slightly more than conserves at its head (summed scaffold area / trunk area = 1.39).
    //
    // Refit after the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(): girth is now
    // proportional to CUMULATIVE downstream leaf area, so the factor is much smaller than the 30 used before. Against
    // tree_13 at 2555 days this gives trunk base diameter 1.03x, largest limb 1.00x and woody volume 1.02x the reference,
    // and limb taper now follows sqrt(downstream leaf area) instead of stepping up by wood age. Trimmed from 4.5 when the
    // leaf-area-index and shadow-grid bud-break terms were removed, since the canopy then carries about 15% more leaf area.
    // See projects/QSMCalibration.
    shoot_parameters_scaffold.girth_area_factor = 0.85f;

    // Spurs. Most of an almond's lateral vegetative buds grow a short shoot carrying a rosette of leaves, and these spurs carry most of
    // the foliage of a mature tree. A lateral that happens to break far below its parent's tip also grows short (see
    // AlmondPhytomerCreationFunction()), but how many do is set by the bud-break probabilities, which also set the number of long shoots:
    // at the values the wood is calibrated with, the tree carried about 250 spurs and a leaf area index well under half that measured in
    // orchards. Spurs of this type are released by AlmondPhytomerCallbackFunction(), from buds that did not grow a long shoot. Four
    // leaves of about 18 cm2 give a spur about 70 cm2, within the 40 to over 90 cm2 measured on Nonpareil spurs.
    ShootParameters shoot_parameters_spur = shoot_parameters_proleptic;
    shoot_parameters_spur.max_nodes = 40;
    shoot_parameters_spur.max_nodes_per_season = 4;
    shoot_parameters_spur.internode_length_max = 0.003;
    shoot_parameters_spur.internode_length_min = 0.003;
    shoot_parameters_spur.internode_length_decay_rate = 0;
    // A spur does not branch.
    shoot_parameters_spur.vegetative_bud_break_probability_max = 0;
    shoot_parameters_spur.vegetative_bud_break_probability_min = 0;
    shoot_parameters_spur.vegetative_bud_break_probability_decay_rate = 0;
    shoot_parameters_spur.gravitropic_curvature = 0;
    shoot_parameters_spur.defineChildShootTypes({"spur"}, {1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);
    defineShootType("spur", shoot_parameters_spur);
}

uint PlantArchitecture::buildAlmondTree(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize almond tree shoots
        initializeAlmondTreeShoots();
    }

    // Get training system parameters (with defaults matching original hard-coded values)
    // Left at 0.6 m deliberately. The QSM reference tree carries its structural limbs from about 0.9-1.0 m,
    // and raising this to match does fix that one measurement -- but it wrecks the crown silhouette, taking
    // the upper/lower crown radius ratio from about 1.1 (the reference value) to 1.9, i.e. a cone. The real
    // tree is wide by 1 m of height because its limbs spread hard straight off the trunk, whereas these
    // scaffolds emerge at 40 degrees and curve upward, so they need vertical distance to spread. Raising the
    // trunk simply empties the lower crown. Reproducing the real architecture needs the scaffold geometry
    // changed, not the trunk raised. See projects/QSMCalibration.
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 0.78f, 0.1f, 3.f, "total trunk height in meters");
    auto num_scaffolds = uint(getParameterValue(current_build_parameters, "num_scaffolds", 4.f, 2.f, 8.f, "number of scaffold branches"));
    auto scaffold_angle = getParameterValue(current_build_parameters, "scaffold_angle", 52.f, 20.f, 70.f, "scaffold branch angle in degrees");

    // Calculate trunk nodes based on desired height and internode length
    float trunk_internode_length = 0.03f; // Default internode length for almond
    uint trunk_nodes = uint(trunk_height / trunk_internode_length);
    if (trunk_nodes < 1)
        trunk_nodes = 1;

    // Fixed training parameters (not user-customizable)
    float trunk_radius = 0.015f;
    float scaffold_radius = 0.007f;
    float scaffold_length = 0.06f;

    uint plantID = addPlantInstance(base_position, 0);

    //    enableEpicormicChildShoots(plantID,"sylleptic",0.001);

    // The trunk is built node by node so it can lean and bow like an orchard trunk; see almondTrunkPath().
    std::vector<vec3> trunk_node_positions;
    std::vector<float> trunk_node_radii;
    almondTrunkPath(context_ptr, base_position, trunk_nodes, trunk_internode_length, trunk_radius, trunk_node_positions, trunk_node_radii);
    uint uID_trunk = addShootFromNodePositions(plantID, -1, 0, trunk_node_positions, trunk_node_radii, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0.01, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // Hard-coded scaffold node range (not user-customizable)
    uint scaffold_nodes_min = 5;
    uint scaffold_nodes_max = 5;

    for (int i = 0; i < num_scaffolds; i++) {
        float pitch = deg2rad(scaffold_angle) + context_ptr->randu(-0.1f, 0.1f); // Small randomness around specified angle
        uint scaffold_nodes = context_ptr->randu(int(scaffold_nodes_min), int(scaffold_nodes_max));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, scaffold_nodes, make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(num_scaffolds) * 2 * M_PI, 0), scaffold_radius,
                                       scaffold_length, 1.f, 1.f, 0.5, "scaffold", 0);
        // The bud at a scaffold's base sits in the trunk/scaffold crotch. Left alone it grows into a long limb straight out of
        // the junction: on Nonpareil it carries the largest subtree on the scaffold, 1.4 m of wood on average against 0.2-0.7 m
        // for the laterals just above it. Orchard training clears the crotch, so the bud is killed here.
        plant_instances.at(plantID).shoot_tree.at(uID_shoot)->phytomers.front()->setVegetativeBudState(BUD_DEAD);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 90, -1, 3, 7, 20, 275);
    plant_instances.at(plantID).max_age = 3000;

    return plantID;
}

void PlantArchitecture::initializeAlmondTreeAldrichShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "AlmondLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.33f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature = 0.05;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 1;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_almond(context_ptr->getRandomGenerator());

    phytomer_parameters_almond.internode.pitch = 3;
    phytomer_parameters_almond.internode.phyllotactic_angle.uniformDistribution(120, 160);
    phytomer_parameters_almond.internode.radius_initial = 0.002;
    phytomer_parameters_almond.internode.length_segments = 1;
    phytomer_parameters_almond.internode.image_texture = "AlmondBark.jpg";
    phytomer_parameters_almond.internode.max_floral_buds_per_petiole = 1; //

    phytomer_parameters_almond.petiole.petioles_per_internode = 1;
    phytomer_parameters_almond.petiole.pitch.uniformDistribution(-145, -90);
    phytomer_parameters_almond.petiole.taper = 0.1;
    phytomer_parameters_almond.petiole.curvature = 0;
    phytomer_parameters_almond.petiole.length = 0.04;
    phytomer_parameters_almond.petiole.radius = 0.0005;
    phytomer_parameters_almond.petiole.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;
    phytomer_parameters_almond.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_almond.leaf.leaves_per_petiole = 1;
    phytomer_parameters_almond.leaf.roll.uniformDistribution(-10, 10);
    phytomer_parameters_almond.leaf.prototype_scale = 0.12;
    phytomer_parameters_almond.leaf.prototype = leaf_prototype;

    phytomer_parameters_almond.peduncle.length = 0.002;
    phytomer_parameters_almond.peduncle.radius = 0.0005;
    phytomer_parameters_almond.peduncle.pitch = 80;
    phytomer_parameters_almond.peduncle.roll = 90;
    phytomer_parameters_almond.peduncle.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;

    phytomer_parameters_almond.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_almond.inflorescence.pitch = 0;
    phytomer_parameters_almond.inflorescence.roll = 0;
    phytomer_parameters_almond.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
    phytomer_parameters_almond.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.fruit_prototype_function = AlmondFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.005;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 24;
    // The trunk is built with a fixed node count in buildAlmondTreeAldrich(), and addBaseStemShoot() throws
    // if that count exceeds this cap. Sized to cover the whole plausible trunk-height range rather than
    // just the current value, as for the almond model. The trunk apical meristem is killed immediately
    // after the plant is built, so this only sizes the guard and does not permit extra growth.
    shoot_parameters_trunk.max_nodes = 101;
    // Girth is area = girth_area_factor * downstream_leaf_area, and downstream leaf area partitions exactly
    // across a fork, so a single factor conserves cross-sectional area through a junction. The trunk and
    // scaffold must carry the SAME value or the junction discards area. 14 brings the modelled trunk to
    // 0.22 m against the reference's 0.24 m.
    // Raised alongside the reduced bud break: girth is driven by downstream leaf area, so thinning the
    // branching removes the leaf area that drives it and the wood thins with it unless the factor rises.
    // 55. The structural wood was measurably too thin -- trunk 0.68x and largest limb 0.73-0.79x of the
    // references -- which is visible directly in renders. This brings the trunk to 0.23 m against their
    // 0.24 m and the largest limb to 0.14 m against 0.13 m.
    //
    // Refit when the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(). Girth is now proportional
    // to cumulative downstream leaf area, which runs well above the old decayed value, so the factor is far smaller; the
    // history above is in the old units. With the acrotonic scaffold decay and 230 curvature this puts trunk base diameter at
    // 0.241 m and the largest limb at 0.140 m against tree_3's 0.244 m and 0.141 m. The scaffold carries the same value so the
    // trunk/scaffold fork still conserves area.
    shoot_parameters_trunk.girth_area_factor = 5.8f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 10;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.internode.radial_subdivisions = 5;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = AlmondPhytomerCallbackFunction;
    // Calibrated against five Phytograph QSM reconstructions of seven-year-old Aldrich almonds (Nickels Block 1,
    // 2022-01-24 survey; variety mean height 6.61 m, crown radius 2.20 m, woody volume 151 L). The stock
    // values produced a tree roughly half the reference height and an eighth of its volume.
    // EXPERIMENTAL (25 is the committed value): raised to extend the interior branching chain.
    // Interior shoots stop at mid-crown because they run out of nodes, not because they are shed or
    // shaded: 76-89% of mid-crown interior shoots sit at their lifetime node cap, and since the
    // bud-break decay concentrates recruitment within about two nodes of the apex, a capped shoot has
    // no high-probability buds left and recruits below replacement (0.77-0.99 children against the 1.0
    // a lineage needs). The chain then halves every few generations, leaving 0.4 m of wood in the top
    // interior against the references' 13.9-21.1 m.
    // Drawn per shoot rather than fixed. A constant node ceiling gives every proleptic shoot the same
    // length budget, so the model cannot produce the short-spur population that dominates a real almond:
    // measured against three Aldrich references, 20-54% of their shoots are under 0.3 m in every height
    // band, against 3-18% here, and their median shoot is 0.28-0.67 m against this model's 0.48-1.20 m.
    // That deficit is what makes the upper canopy read as lacking fine detail, and what makes the few long
    // low branches conspicuous -- the references have equally long low branches (up to 6.5-9.3 m), but
    // surround them with spurs. Other species in this library already sample max_nodes per shoot.
    shoot_parameters_proleptic.max_nodes.uniformDistribution(20, 55);
    shoot_parameters_proleptic.max_nodes_per_season = 15;
    shoot_parameters_proleptic.phyllochron_min = 1;
    shoot_parameters_proleptic.elongation_rate_max = 0.3;
    // Branch cross-section per unit downstream leaf area. At 24 every branch came out the same middling
    // thickness -- 98% of them between 1 and 3 cm, with nothing below 1 cm -- which renders as an
    // undifferentiated tangle with only the scaffolds standing out. 16 restores a spread (30% below 1 cm)
    // and scores best on the objective across a 24/16/12/10/8 scan.
    //
    // Do not push this lower to chase the QSM's branch-diameter distribution. That distribution is clipped
    // by the reconstruction, not by the tree: only 0.5% of reference cylinders fall below the 3.5 mm
    // detection radius the harness prunes at, and its 5th percentile is 3.8 mm. Matching its apparent 57-62%
    // "thin" fraction requires driving half the model's wood below detectability -- pruned cylinders fall
    // from 13738 to 7535 at a factor of 8 -- which improves the statistic by hiding branches rather than by
    // making the architecture right.
    //
    // The numbers above are in the units of the old girth law, refit when the per-phytomer 365/age decay was removed from
    // incrementPhytomerInternodeGirth(). The warning stands in the new units: going from 12 to 7 cuts woody volume only from
    // 257 L to 211 L (reference 165 L) while detected order-4 branches fall from 69 to 33 (reference 119), so the remaining
    // volume excess is not worth chasing with this factor.
    shoot_parameters_proleptic.girth_area_factor = 12.f;
    // Reduced to thin the branching. Total woody length was 2.3x the reference before this change, with
    // only 32% of it in orders 1-2 against the references' 49%.
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.05;
    // Trimmed to 0.32 to offset the scaffold bud-break increase below, which would otherwise push total
    // woody length well past the reference.
    // Lowered from 0.38. At 0.38 the tree carried 981 branches over 733 m of wood against the references'
    // 509-545 branches and 404-455 m -- roughly 80% too much wood, which reads as a solid mass rather than a
    // crown with gaps in it. At 0.22 the branch count (562), total wood length (408 m) and order-2 count
    // (174.5 against a reference 176) all land on the references, and the objective improves from 10.3 to 6.6.
    // Raised from 0.22 to compensate the low-crown shedding term (see branch_shedding_low_crown_probability
    // in PlantArchitecture.cpp), which removes interior wood from the lower crown. Pruning without
    // compensation subtracts wood rather than redistributing it -- total woody length fell to 267 m against
    // the references' 404 m -- so the generative side has to be raised alongside it. Paired with a shed
    // probability of 0.35 this gives the reference's vertical profile at the reference's total wood.
    // EXPERIMENTAL (0.28 is the committed value): cut to pay for the raised node cap above. Longer
    // shoots carry more buds, so without this the tree reaches 854 branches and 824 m of wood against
    // the references' ~509 and ~404 m. At 0.15 the upper-crown density lands on the references
    // (10/34/63 branches within 0.5/1.0/1.5 m of the top, against 13/33/62 and 11/34/70) at 524
    // branches -- redistribution rather than inflation.
    // EXPERIMENTAL (0.15 is the committed value): raised to pay for the spur gradient above. Turning basal
    // children into spurs removes most of their wood, halving total woody length to 199 m against the
    // references' 404-455; more shoots offsets each being shorter.
    // Scaled by 0.85 when the whole-plant leaf-area-index and shadow-grid bud-break terms were removed from the model: at
    // almond's canopy density those two had been multiplying every bud-break probability by roughly this much. Matched on
    // Nonpareil, 0.85 on the proleptic and scaffold maxima restores total woody length with crown shape unchanged.
    // Aldrich needed a further cut (0.24 -> 0.21) to bring total woody length back to its previous fit.
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.21;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    // Reduced from 450: at that value the crown was strongly top-heavy (upper/lower crown radius ratio
    // 1.75 against the reference's 0.98). Aldrich is widest low and tapers upward.
    // Reduced from 280. The model was relying on the gravitropic response to hold the crown upright, which
    // bent every branch to the same attitude regardless of how it emerged -- measured length-weighted
    // cylinder zenith was ~38 deg at EVERY order against the references' 44-56 deg -- and produced the long
    // curved "whippy" strands that distinguish the generated crowns from the real ones. The habit is set by
    // the insertion angle instead, which is what the references show: branches leave at a wide angle and
    // then run comparatively straight.
    // 150 rather than 90: the weaker value was paired with a 60 deg insertion angle and, on its own, lets
    // the crown spread. The straightness argument for reducing it from 280 still holds -- see the notes in
    // doc/calibration/almond_variety_calibration.md -- but it cannot be pushed this low until the branching
    // density defect is fixed, since the model currently needs long branches to fill its crown.
    // 190 rather than 150. The reduction from the stock 450 was to stop the model relying on the
    // gravitropic response to hold the crown upright, which bent every branch to the same attitude and read
    // as long curved strands; but taken too far the branches run dead straight and the crown becomes
    // inverse-conical, since nothing curves the wood back in as it extends. Paired with the tortuosity
    // below and with the scaffold values, this sits between the two failure modes.
    // EXPERIMENTAL (270 is the committed value): reduced for crown shape. A weaker upward response lets
    // laterals run outward instead of turning up, which both widens the crown and lowers its widest point --
    // the two geometric defects that have persisted longest. Scanned 270/220/180/150/120/90: the objective
    // turns at 150 (5.55 against 6.29 at 220 and 7.95 at 270), crown flare reaches 1.23 (best recorded) and
    // crown radius 0.90 of reference against 0.83 at 270. Below 150 the crown keeps widening but the
    // objective degrades again, so this is an interior optimum rather than a floor.
    // Gravitropic curvatures raised 20% across all shoot types from renders, which showed branches needing a little more upward
    // curvature throughout the crown.
    shoot_parameters_proleptic.gravitropic_curvature = 180;
    // 1.5 rather than 2.6. Tortuosity is a NOISE parameter, so raising it widens the run-to-run spread as
    // well as the wander: at 2.6 the ensemble coefficient of variation on total woody length was 19.4%,
    // which showed up as some trees coming out sparse and others as a dense tangle. Lowering it, with the
    // scaffold changes above, brings that to 7.8%.
    shoot_parameters_proleptic.tortuosity = 37.5;
    shoot_parameters_proleptic.tortuosity_persistence_length = 0.3;
    // Widened from 5-30 deg: the references carry their laterals at a median branching angle of 57-66 deg,
    // and the narrow distribution contributed to the clustered, near-axial branching habit.
    // Held at 45 deg. Measured against the references the insertion angle is already correct at this value
    // (branch_angle_order1 62 deg model against 57 deg reference), and widening it further to 60 deg flared
    // the crown badly without improving the metric it was meant to serve.
    shoot_parameters_proleptic.insertion_angle_tip = 45;
    shoot_parameters_proleptic.insertion_angle_decay_rate = 5;
    shoot_parameters_proleptic.internode_length_max = 0.04;
    shoot_parameters_proleptic.internode_length_min = 0.002;
    // EXPERIMENTAL (0.002 is the committed value). Internode length falls with a child's distance below its
    // parent's apex, so distal children are long and basal ones are spurs -- the gradient a real almond has.
    // At 0.002 the decay needed 19 nodes to reach the floor, but a parent only has that many nodes after
    // several seasons, by which time its apex has moved far above; measured, only 1% of shoots were born
    // anywhere near the floor. At 0.004 the floor is reached in ~10 nodes, which is inside the range parents
    // actually occupy when bearing children, and the short-shoot fraction rises from 8-29% to 25-67% across
    // height bands against reference values of 27-54%.
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.4;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 3;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic", "sylleptic"}, {1.0, 0.});

    // Sylleptic shoots
    ShootParameters shoot_parameters_sylleptic = shoot_parameters_proleptic;
    //    shoot_parameters_sylleptic.phytomer_parameters.internode.color = RGB::red;
    shoot_parameters_sylleptic.phytomer_parameters.internode.image_texture = "";
    shoot_parameters_sylleptic.phytomer_parameters.leaf.prototype_scale = 0.12;
    shoot_parameters_sylleptic.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20);
    shoot_parameters_sylleptic.insertion_angle_tip = 0;
    shoot_parameters_sylleptic.insertion_angle_decay_rate = 0;
    shoot_parameters_sylleptic.phyllochron_min = 1;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_min = 0.0;
    // NOTE: this line assigned to shoot_parameters_PROLEPTIC while sitting in the middle of the sylleptic
    // block, silently overwriting the proleptic value set above. Almost certainly a copy-paste slip. It is
    // corrected to sylleptic here; the proleptic value is set in the proleptic block.
    shoot_parameters_sylleptic.vegetative_bud_break_probability_max = 0.6;
    shoot_parameters_sylleptic.gravitropic_curvature = 720;
    shoot_parameters_sylleptic.internode_length_max = 0.02;
    shoot_parameters_sylleptic.tortuosity = 75;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true;
    shoot_parameters_sylleptic.defineChildShootTypes({"proleptic"}, {1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    //    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    // Extended from 20 nodes x 0.02 m (0.4 m) to 45 x 0.04 m (1.8 m). The references are characterised by
    // heavy limbs that stay thick and identifiable for 2-3 m above the head; at the old values a scaffold
    // handed off to proleptic children within half a metre, so the tree had no visible structural skeleton
    // and read as a uniform bush. Measured length of wood at or above 4 cm diameter: references 23-24 m
    // reaching to 3 m, model 12 m reaching to 2 m before this change.
    // 90 nodes x 0.04 m gives scaffolds about 5.4 m long, reaching 87-92% of tree height. At the previous
    // 45 they ran 2.7 m and stopped at 43% -- the tree looked reasonable below that point and turned into a
    // bush above it, because the whole upper crown was then built from fine proleptic wood. The reference
    // scaffolds run 6.1 m to 91% of height (tree 3) and 3.6 m to 55% (tree 7).
    //
    // Paired with the reduced proleptic max_nodes below: lengthening the scaffolds alone simply made the
    // tree taller (8.8 m against a 6.5 m reference), because the scaffolds extended past the crown instead
    // of running up through it.
    // EXPERIMENTAL (90 is the committed value): raised for crown height. The scaffolds are the tree's
    // vertical carriers, so lengthening them lifts the laterals they already bear rather than creating
    // new ones -- height rises while total wood FALLS (volume 1.70 -> 1.63 of reference). Gravitropic
    // curvature was tried first and does not buy height at all: at 330 it gained nothing, and at 400 it
    // reached the target only by narrowing the crown to 0.78 of the reference radius.
    shoot_parameters_scaffold.max_nodes = 105;
    // Reduced from 200. The scaffolds are launched at 40-65 deg from vertical in buildAlmondTreeAldrich(),
    // but at 200 deg/m the gravitropic response bent them back to a measured base zenith of 41 deg against
    // the references' 64 deg, which read as a narrow steep broom instead of an open vase. At 60 deg/m the
    // limbs hold the angle they emerge at (measured 59 deg, branching angle 55 deg against 57 deg).
    // Raised with the proleptic value. Acting on the scaffolds matters more than acting on the proleptic
    // shoots for crown shape: curvature and wander applied to the proleptic shoots alone barely moved the
    // upper/lower crown radius ratio (1.845 -> 1.849), while applying them to the scaffolds as well brought
    // it to 1.735 and the objective from 16.6 to 13.5. The scaffolds carry the crown's outline.
    // 45, well below the proleptic value. The references' scaffolds radiate outward essentially straight,
    // forming a cone; a strong gravitropic response instead curves them upward and loses that structure.
    // 170. The scaffolds are now 90 nodes (about 5.4 m) rather than 45, and at that length a weak
    // gravitropic response lets them run essentially straight out of the crown: measured scaffold tip
    // distance from the trunk axis was 2.56 m against the references' 1.43-1.88 m, giving a cone-shaped
    // tree with an empty middle. The references' scaffolds reach only about a third of the horizontal
    // distance a straight branch of their length and emergence angle would, so they emerge wide and then
    // curve strongly upward. A short scaffold can afford to be straight; a long one cannot.
    // Raised to 230 to match the Nonpareil set, where renders showed the scaffolds wanting slightly more upward curvature.
    shoot_parameters_scaffold.gravitropic_curvature = 276;
    shoot_parameters_scaffold.internode_length_max = 0.04;
    // 1.6. Gravitropism is a deterministic force, so a strong curvature with weak tortuosity bends every
    // scaffold along the same uniform arc, which reads as an unnaturally smooth sweeping curve over a 5.4 m
    // limb. Tortuosity is the only term breaking that regularity. The proleptic shoots run a
    // curvature-to-tortuosity ratio near 180; the scaffolds were at 425, more than twice as smooth per unit
    // of bend. Raising tortuosity rather than lowering curvature keeps the inward curve that the scaffold
    // tip distance requires.
    // Raised from 0.4 from renders, once the 20% stronger gravitropic response left the Aldrich scaffolds looking too smooth.
    shoot_parameters_scaffold.tortuosity = 12.5;
    // Correlation length of the tortuosity random walk, and the reason the wiggle reads as flailing rather
    // than as natural jitter. The walk is an Ornstein-Uhlenbeck process that mean-reverts over this
    // distance, so it is the RATIO of shoot length to persistence length that sets the character: at the
    // 0.5 m default a 5.4 m scaffold wanders through eleven independent excursions and loses its trajectory
    // entirely. At 2.5 m it makes about two, which reads as a limb holding its general path with some
    // wiggle. This model never set the parameter at all and was inheriting the default; the almond model
    // sets it explicitly, and this one now does too.
    shoot_parameters_scaffold.tortuosity_persistence_length = 2.5;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});
    // Match the trunk, so cross-sectional area is conserved through the trunk/scaffold junction. Inheriting
    // the proleptic value instead makes the junction discard area in proportion to the two factors' ratio.
    // Raised from the inherited proleptic value so the scaffolds themselves carry laterals. Without this
    // the lower crown is bare scaffold wood, since almost everything else is borne higher up.
    shoot_parameters_scaffold.vegetative_bud_break_probability_max = 0.72;
    // A scaffold in the references bears laterals along its whole length, not just near its base: the QSM
    // origin counts on order-1 axes run 27/45/32/48/19 across the 0-1,1-2,...,4-5 m height bands. The default
    // decay rate of -0.5 per node drives the break probability to its floor within two nodes (about 8 cm) of
    // the scaffold base, which leaves the lower crown empty and forces all the remaining branch mass to arrive
    // through proleptic-on-proleptic cascades, piling it into a narrow band in the middle of the crown. Decaying
    // roughly a hundred times more slowly spreads the laterals over the full scaffold.
    //
    // Later made acrotonic (+0.02), as in the Nonpareil set: bud-break vigour rising toward the apex is the pattern that
    // governs tree development, whereas basitony characterises shrubs (Costes et al., Front. Plant Sci. 5:666, 2014; Wilson,
    // Am. J. Bot. 87:601, 2000). The basitonic -0.01 concentrated scaffold recruitment at the trunk junction, which in
    // Nonpareil renders is where long structural branches emerged and distorted the crown.
    shoot_parameters_scaffold.vegetative_bud_break_probability_decay_rate = 0.02;
    shoot_parameters_scaffold.vegetative_bud_break_probability_min = 0.10;
    shoot_parameters_scaffold.girth_area_factor = 5.8f;

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);
}

uint PlantArchitecture::buildAlmondTreeAldrich(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize almond tree shoots
        initializeAlmondTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    //    enableEpicormicChildShoots(plantID,"sylleptic",0.001);

    // 26 nodes x 0.03 m puts the head at about 0.78 m, matching the mean height of the major scaffolds in
    // the Aldrich QSM references. At the previous 19 nodes the head sat at 0.57 m and the whole crown rode
    // correspondingly low.
    // The trunk is built node by node so it can lean and bow like an orchard trunk; see almondTrunkPath().
    std::vector<vec3> trunk_node_positions;
    std::vector<float> trunk_node_radii;
    almondTrunkPath(context_ptr, base_position, 26, 0.03f, 0.015f, trunk_node_positions, trunk_node_radii);
    uint uID_trunk = addShootFromNodePositions(plantID, -1, 0, trunk_node_positions, trunk_node_radii, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0.01, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 4; // context_ptr->randu(4,5);

    for (int i = 0; i < Nscaffolds; i++) {
        // Emergence angle from vertical. The references carry their scaffolds at a median base zenith of
        // 64 deg; at the previous 5-35 deg the limbs rose almost vertically out of the head, which read as
        // a narrow steep broom rather than an open vase and left the crown too dense near the axis.
        // Emergence angle from vertical. Narrowed from 40-65 deg: the wider range carried the scaffolds --
        // and everything borne on them -- too far out, giving a crown radius of 2.76 m against the
        // references' 2.43 m. At 30-50 deg the radius lands at 2.53 m and the crown's widest point drops
        // from 0.83 to 0.80 of tree height.
        //
        // The range still sits inside what the references show: their major scaffolds emerge between 35 and
        // 77 deg with a median near 50, so this samples the lower half of the observed spread rather than
        // anything the trees do not do.
        // Emergence angle from vertical, narrowed from the original 40-65 deg. The references' major
        // scaffolds emerge between 35 and 77 deg with a median near 50, so this samples the lower part of
        // the observed spread; matching their mean carried the scaffolds, and everything borne on them, too
        // far out.
        //
        // Note that narrowing this range does NOT reduce the run-to-run variation in crown width: tightening
        // it from 20 deg wide to 7 deg leaves the ensemble coefficient of variation on crown radius at
        // 9-12% throughout. That variability comes from which buds break on the scaffolds and how far the
        // resulting laterals run, not from where the scaffolds point.
        // Emergence angle from vertical, measured on the references as 55-58 deg mean with a standard
        // deviation near 15. The narrower range used previously was compensating for scaffolds that ran
        // almost straight; with the gravitropic response restored they emerge wide and curve back in, which
        // is what the references do.
        // Emergence angle from vertical, paired with the scaffold gravitropic curvature below. The references
        // show a scaffold leaving at a wide angle, making a fairly quick upward bend, and then running
        // comparatively straight. The model cannot switch its gravitropic response off partway along a
        // shoot, but it does not need to: the response is already self-limiting, since its magnitude scales
        // with (1 - cos(zenith))/2 and so falls to zero as the shoot approaches vertical. What it cannot do
        // is bend sharply and then stop, so a weaker response paired with a smaller emergence angle is the
        // closest this formulation gets to the reference behaviour.
        //
        // The range is set from the QSM order-1 axes filtered to a base radius of 3 cm or more, which
        // isolates the three or four true scaffolds of a headed tree: zenith mean 49-57 deg with a standard
        // deviation of 11-14 deg. Filtering matters. All order-1 axes taken together give 41-45 branches per
        // tree with a mean of 62 deg and a standard deviation of 26-28 deg, because the reconstruction
        // labels many ordinary upper laterals as order 1; calibrating to that unfiltered spread produces
        // scaffolds that occasionally emerge near-horizontal and throw the crown badly out of shape.
        float pitch = context_ptr->randu(deg2rad(46), deg2rad(58));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, context_ptr->randu(7, 9), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007, 0.06,
                                       1.f, 1.f, 0.5, "scaffold", 0);
        // The bud at a scaffold's base sits in the trunk/scaffold crotch. Left alone it grows into a long limb straight out of
        // the junction: on Nonpareil it carries the largest subtree on the scaffold, 1.4 m of wood on average against 0.2-0.7 m
        // for the laterals just above it. Orchard training clears the crotch, so the bud is killed here.
        plant_instances.at(plantID).shoot_tree.at(uID_shoot)->phytomers.front()->setVegetativeBudState(BUD_DEAD);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 90, -1, 3, 7, 20, 275);
    // Seven-year QSM references are the calibration target for this variety, and growth stops entirely at
    // max_age, so a 5-year cap left the modelled tree at roughly a third of the reference height. Matches
    // the value used by the almond (Nonpareil) model.
    plant_instances.at(plantID).max_age = 3000;

    return plantID;
}

void PlantArchitecture::initializeAlmondTreeIndependenceShoots() {

    // Independence almond, fit against the four Independence Phytograph QSM reconstructions (trees 34-37 of the Nickels block-1
    // scan). Starts from the calibrated Nonpareil set; see projects/QSMCalibration for the fit.


    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "AlmondLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.33f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature = 0.05;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 1;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_almond(context_ptr->getRandomGenerator());

    // Node-scale roughness: the small, discrete kink applied at each phytomer base. This is the fine-texture scale, distinct from
    // the larger-scale wander produced by the tortuosity random walk along the internode.
    //
    // The distribution is symmetric about zero rather than about a positive value. The kink direction is already scattered around
    // the stem by the phyllotactic angle, so a signed magnitude lets successive kinks partially cancel: this yields the same net
    // wander as a positive-mean distribution of twice the width, but spends roughly half the total bending to get there, leaving
    // the directed gravitropic response rather than accumulated node kinks to set the overall branch trajectory. A constant pitch
    // traces a near-perfect helix (the kinks cancel almost exactly) and reads as a smooth systematic bend, not as texture.
    //
    // Note this is a phenomenological roughness term. A negative pitch has no specific botanical reading on its own; the
    // mechanistic direction of nodal deflection is carried by the phyllotactic rotation of the axis it is applied about.
    phytomer_parameters_almond.internode.pitch.uniformDistribution(-7, 7);
    phytomer_parameters_almond.internode.phyllotactic_angle.uniformDistribution(120, 160);
    phytomer_parameters_almond.internode.radius_initial = 0.002;
    phytomer_parameters_almond.internode.length_segments = 1;
    phytomer_parameters_almond.internode.image_texture = "AlmondBark.jpg";
    phytomer_parameters_almond.internode.max_floral_buds_per_petiole = 1; //

    phytomer_parameters_almond.petiole.petioles_per_internode = 1;
    phytomer_parameters_almond.petiole.pitch.uniformDistribution(-145, -90);
    phytomer_parameters_almond.petiole.taper = 0.1;
    phytomer_parameters_almond.petiole.curvature = 0;
    phytomer_parameters_almond.petiole.length = 0.04;
    phytomer_parameters_almond.petiole.radius = 0.0005;
    phytomer_parameters_almond.petiole.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;
    phytomer_parameters_almond.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_almond.leaf.leaves_per_petiole = 1;
    phytomer_parameters_almond.leaf.roll.uniformDistribution(-10, 10);
    phytomer_parameters_almond.leaf.prototype_scale = 0.12;
    phytomer_parameters_almond.leaf.prototype = leaf_prototype;

    phytomer_parameters_almond.peduncle.length = 0.002;
    phytomer_parameters_almond.peduncle.radius = 0.0005;
    phytomer_parameters_almond.peduncle.pitch = 80;
    phytomer_parameters_almond.peduncle.roll = 90;
    phytomer_parameters_almond.peduncle.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;

    phytomer_parameters_almond.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_almond.inflorescence.pitch = 0;
    phytomer_parameters_almond.inflorescence.roll = 0;
    phytomer_parameters_almond.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
    phytomer_parameters_almond.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.fruit_prototype_function = AlmondFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.005;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 24;
    // The trunk node count is derived in buildAlmondTree() as trunk_height / 0.03, and trunk_height is
    // accepted over 0.1-3.0 m. The cap must therefore cover the whole of that range, otherwise any
    // trunk_height above 0.6 m throws out of addBaseStemShoot(). The trunk apical meristem is killed
    // immediately after the plant is built, so this only sizes the guard; it does not permit extra growth.
    shoot_parameters_trunk.max_nodes = 101;
    // Raised from the Nonpareil 4.5 to recover the trunk diameter and woody volume lost with the thinner branching.
    shoot_parameters_trunk.girth_area_factor = 5.7f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 10;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.internode.radial_subdivisions = 5;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = AlmondPhytomerCallbackFunction;
    // Calibrated against a Phytograph QSM reconstruction of a seven-year-old leaf-off almond (see
    // projects/QSMCalibration). Capping the lifetime node count well below the old value of 60 stops a
    // proleptic shoot from extending indefinitely across seasons; the reference tree carries its wood as
    // many short branches rather than a few long ones, and the uncapped shoot produced branches roughly
    // 1.5x too long.
    shoot_parameters_proleptic.max_nodes.uniformDistribution(20, 55);
    shoot_parameters_proleptic.max_nodes_per_season = 15;
    shoot_parameters_proleptic.phyllochron_min = 1;
    shoot_parameters_proleptic.elongation_rate_max = 0.3;
    shoot_parameters_proleptic.girth_area_factor = 4.5f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.05;
    // Scaled by 0.85 when the whole-plant leaf-area-index and shadow-grid bud-break terms were removed from the model: at
    // almond's canopy density those two had been multiplying every bud-break probability by roughly this much. Matched on
    // Nonpareil, 0.85 on the proleptic and scaffold maxima restores total woody length with crown shape unchanged.
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.31;
    // The decay is linear in nodes from the tip: p = max(p_min, p_max - decay * nodes_from_tip). At the
    // old rate of 0.15 the probability fell from 0.5 to the 0.1 floor within three nodes, so effectively
    // the whole shoot broke buds at the minimum and the tree ramified at about half the observed rate.
    // Spreading the decay over roughly eight nodes brings forks per metre into line with the QSM tree.
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    // Sets how fast a lateral reorients toward vertical, in degrees per metre of arc. At the old value of
    // 250 a branch that emerged near-horizontal was back to near-vertical within 10-25 cm, so laterals in
    // the lower crown ran upward rather than outward: they piled on woody length and branch count without
    // contributing any crown width, which read as a narrow-based, flare-topped (cone/mushroom) canopy.
    //
    // Measured against the QSM reference, order-2 branches emerge at a sensible angle in both trees
    // (71 deg from vertical in the model, 64 deg in the reference) but the model then turned 24 deg along
    // its length against the reference's 7 deg. Halving the response brings crown radius from 1.8 to 2.0 m
    // (reference 2.5 m) and the upper/lower crown-radius ratio from 1.24 to 1.06 (reference 1.14).
    //
    // Note this is the proleptic value only. The scaffold and sylleptic types are copy-constructed from
    // these parameters but both assign their own gravitropic_curvature below, so neither inherits it.
    // Gravitropic curvatures raised 20% across all shoot types from renders, which showed branches needing a little more upward
    // curvature throughout the crown.
    shoot_parameters_proleptic.gravitropic_curvature = 180;
    shoot_parameters_proleptic.tortuosity = 37.5;
    shoot_parameters_proleptic.tortuosity_persistence_length = 0.3;
    shoot_parameters_proleptic.insertion_angle_tip = 45;
    shoot_parameters_proleptic.insertion_angle_decay_rate = 15;
    shoot_parameters_proleptic.internode_length_max = 0.04;
    shoot_parameters_proleptic.internode_length_min = 0.002;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.4;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 3;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic", "sylleptic"}, {1.0, 0.});

    // Sylleptic shoots
    ShootParameters shoot_parameters_sylleptic = shoot_parameters_proleptic;
    //    shoot_parameters_sylleptic.phytomer_parameters.internode.color = RGB::red;
    shoot_parameters_sylleptic.phytomer_parameters.internode.image_texture = "";
    shoot_parameters_sylleptic.phytomer_parameters.leaf.prototype_scale = 0.14;
    shoot_parameters_sylleptic.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20);
    shoot_parameters_sylleptic.insertion_angle_tip = 0;
    shoot_parameters_sylleptic.insertion_angle_decay_rate = 0;
    shoot_parameters_sylleptic.phyllochron_min = 1;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_min = 0.0;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_max = 0.7;
    shoot_parameters_sylleptic.gravitropic_curvature = 720;
    shoot_parameters_sylleptic.internode_length_max = 0.02;
    shoot_parameters_sylleptic.tortuosity = 75;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true;
    shoot_parameters_sylleptic.defineChildShootTypes({"proleptic"}, {1.0});
    // Sylleptic shoots are copied from the proleptic parameters above, so they would otherwise inherit the
    // QSM-calibrated node cap and bud-break decay. Those were validated for proleptic shoots only, so the
    // previous values are restored here rather than being changed as an invisible side effect of the
    // calibration. (The almond model gives sylleptic a child-type probability of zero, so this is a
    // correctness measure and does not alter the generated tree.)
    shoot_parameters_sylleptic.max_nodes = 60;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_decay_rate = 0.15;

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    //    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    // Shortened from the Nonpareil 105 from renders, where Independence read as too tall. The scaffolds finish exactly at this
    // cap, so it sets their length directly.
    shoot_parameters_scaffold.max_nodes = 90;
    shoot_parameters_scaffold.gravitropic_curvature = 276;
    // shoot_parameters_scaffold.internode_length_max = 0.04;
    shoot_parameters_scaffold.tortuosity = 10;
    shoot_parameters_scaffold.tortuosity_persistence_length = 2.5;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});
    // Likewise for the scaffolds: the calibration applies to proleptic shoots. The scaffolds already pin
    // their own max_nodes above, so only the bud-break decay needs restoring.
    // Positive: acrotonic, with bud-break vigour rising toward the apex. Acrotony is "the increase in vigor
    // (length, diameter, number of leaves) of vegetative branches from the bottom to the top position of the
    // parent growth unit" and is the pattern that governs tree development, whereas basitony -- stronger growth
    // at the base, which a negative rate encodes -- characterises shrubs (Costes et al., Front. Plant Sci.
    // 5:666, 2014; Wilson, Am. J. Bot. 87:601, 2000). The previous -0.01 concentrated scaffold recruitment at
    // the trunk junction, which is where the long structural branches that distorted the crown were emerging.
    shoot_parameters_scaffold.vegetative_bud_break_probability_decay_rate = 0.02;
    // Independence carries about a fifth fewer branches than Nonpareil in the QSMs (659 against 812 per tree) inside a
    // similar envelope. Lowering scaffold recruitment brings order-1 and order-2 counts to the four-tree means (38 and 197)
    // and takes the crown from a vase (upper/lower radius 1.46) to the references' 0.67-0.99.
    shoot_parameters_scaffold.vegetative_bud_break_probability_max = 0.55;
    shoot_parameters_scaffold.vegetative_bud_break_probability_min = 0.10;
    // Girth is area = girth_area_factor * downstream_leaf_area, and downstream leaf area partitions exactly
    // across a fork, so with a single factor the pipe model conserves cross-sectional area through a
    // junction. Inheriting the proleptic value of 6 while the trunk uses 10 broke that at the one junction
    // where it matters most: measured at the trunk/scaffold fork, the summed scaffold area came to exactly
    // 0.600 of the trunk's -- the 6/10 parameter ratio -- discarding 40% of the cross-section, while every
    // other junction in the tree conserved area to within the reference's own scatter.
    //
    // The QSM reference slightly more than conserves at its head (summed scaffold area / trunk area = 1.39).
    //
    // Refit after the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(): girth is now
    // proportional to CUMULATIVE downstream leaf area, so the factor is much smaller than the 30 used before. Against
    // tree_13 at 2555 days this gives trunk base diameter 1.04x, largest limb 0.96x and woody volume 0.98x the reference,
    // and limb taper now follows sqrt(downstream leaf area) instead of stepping up by wood age. See projects/QSMCalibration.
    shoot_parameters_scaffold.girth_area_factor = 5.7f;

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);
}

uint PlantArchitecture::buildAlmondTreeIndependence(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize Independence almond tree shoots
        initializeAlmondTreeIndependenceShoots();
    }

    // Get training system parameters (with defaults matching original hard-coded values)
    // Left at 0.6 m deliberately. The QSM reference tree carries its structural limbs from about 0.9-1.0 m,
    // and raising this to match does fix that one measurement -- but it wrecks the crown silhouette, taking
    // the upper/lower crown radius ratio from about 1.1 (the reference value) to 1.9, i.e. a cone. The real
    // tree is wide by 1 m of height because its limbs spread hard straight off the trunk, whereas these
    // scaffolds emerge at 40 degrees and curve upward, so they need vertical distance to spread. Raising the
    // trunk simply empties the lower crown. Reproducing the real architecture needs the scaffold geometry
    // changed, not the trunk raised. See projects/QSMCalibration.
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 0.78f, 0.1f, 3.f, "total trunk height in meters");
    auto num_scaffolds = uint(getParameterValue(current_build_parameters, "num_scaffolds", 4.f, 2.f, 8.f, "number of scaffold branches"));
    auto scaffold_angle = getParameterValue(current_build_parameters, "scaffold_angle", 52.f, 20.f, 70.f, "scaffold branch angle in degrees");

    // Calculate trunk nodes based on desired height and internode length
    float trunk_internode_length = 0.03f; // Default internode length for almond
    uint trunk_nodes = uint(trunk_height / trunk_internode_length);
    if (trunk_nodes < 1)
        trunk_nodes = 1;

    // Fixed training parameters (not user-customizable)
    float trunk_radius = 0.015f;
    float scaffold_radius = 0.007f;
    float scaffold_length = 0.06f;

    uint plantID = addPlantInstance(base_position, 0);

    //    enableEpicormicChildShoots(plantID,"sylleptic",0.001);

    // The trunk is built node by node so it can lean and bow like an orchard trunk; see almondTrunkPath().
    std::vector<vec3> trunk_node_positions;
    std::vector<float> trunk_node_radii;
    almondTrunkPath(context_ptr, base_position, trunk_nodes, trunk_internode_length, trunk_radius, trunk_node_positions, trunk_node_radii);
    uint uID_trunk = addShootFromNodePositions(plantID, -1, 0, trunk_node_positions, trunk_node_radii, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0.01, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // Hard-coded scaffold node range (not user-customizable)
    uint scaffold_nodes_min = 5;
    uint scaffold_nodes_max = 5;

    for (int i = 0; i < num_scaffolds; i++) {
        float pitch = deg2rad(scaffold_angle) + context_ptr->randu(-0.1f, 0.1f); // Small randomness around specified angle
        uint scaffold_nodes = context_ptr->randu(int(scaffold_nodes_min), int(scaffold_nodes_max));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, scaffold_nodes, make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(num_scaffolds) * 2 * M_PI, 0), scaffold_radius,
                                       scaffold_length, 1.f, 1.f, 0.5, "scaffold", 0);
        // The bud at a scaffold's base sits in the trunk/scaffold crotch. Left alone it grows into a long limb straight out of
        // the junction: on Nonpareil it carries the largest subtree on the scaffold, 1.4 m of wood on average against 0.2-0.7 m
        // for the laterals just above it. Orchard training clears the crotch, so the bud is killed here.
        plant_instances.at(plantID).shoot_tree.at(uID_shoot)->phytomers.front()->setVegetativeBudState(BUD_DEAD);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 90, -1, 3, 7, 20, 275);
    plant_instances.at(plantID).max_age = 3000;

    return plantID;
}


void PlantArchitecture::initializeAlmondTreeWoodColonyShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "AlmondLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.33f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature = 0.05;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 1;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_almond(context_ptr->getRandomGenerator());

    phytomer_parameters_almond.internode.pitch = 3;
    phytomer_parameters_almond.internode.phyllotactic_angle.uniformDistribution(120, 160);
    phytomer_parameters_almond.internode.radius_initial = 0.002;
    phytomer_parameters_almond.internode.length_segments = 1;
    phytomer_parameters_almond.internode.image_texture = "AlmondBark.jpg";
    phytomer_parameters_almond.internode.max_floral_buds_per_petiole = 1; //

    phytomer_parameters_almond.petiole.petioles_per_internode = 1;
    phytomer_parameters_almond.petiole.pitch.uniformDistribution(-145, -90);
    phytomer_parameters_almond.petiole.taper = 0.1;
    phytomer_parameters_almond.petiole.curvature = 0;
    phytomer_parameters_almond.petiole.length = 0.04;
    phytomer_parameters_almond.petiole.radius = 0.0005;
    phytomer_parameters_almond.petiole.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;
    phytomer_parameters_almond.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_almond.leaf.leaves_per_petiole = 1;
    phytomer_parameters_almond.leaf.roll.uniformDistribution(-10, 10);
    phytomer_parameters_almond.leaf.prototype_scale = 0.12;
    phytomer_parameters_almond.leaf.prototype = leaf_prototype;

    phytomer_parameters_almond.peduncle.length = 0.002;
    phytomer_parameters_almond.peduncle.radius = 0.0005;
    phytomer_parameters_almond.peduncle.pitch = 80;
    phytomer_parameters_almond.peduncle.roll = 90;
    phytomer_parameters_almond.peduncle.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;

    phytomer_parameters_almond.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_almond.inflorescence.pitch = 0;
    phytomer_parameters_almond.inflorescence.roll = 0;
    phytomer_parameters_almond.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
    phytomer_parameters_almond.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_almond.inflorescence.fruit_prototype_function = AlmondFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.005;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 24;
    shoot_parameters_trunk.max_nodes = 15;
    // Rescaled when the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(), to keep base diameters at
    // max_age where they were: girth is now proportional to cumulative downstream leaf area, which runs well above the old value.
    shoot_parameters_trunk.girth_area_factor = 2.3f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 30.8;
    shoot_parameters_trunk.internode_length_max = 0.0325;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.internode.radial_subdivisions = 5;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = AlmondPhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes = 20;
    shoot_parameters_proleptic.max_nodes_per_season = 15;
    shoot_parameters_proleptic.phyllochron_min = 1;
    shoot_parameters_proleptic.elongation_rate_max = 0.3;
    shoot_parameters_proleptic.girth_area_factor = 5.9f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.5;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 150;
    shoot_parameters_proleptic.tortuosity = 150;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(30, 75);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 17.5;
    shoot_parameters_proleptic.internode_length_max = 0.02;
    shoot_parameters_proleptic.internode_length_min = 0.002;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.002;
    shoot_parameters_proleptic.fruit_set_probability = 0.4;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 3;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic", "sylleptic"}, {1.0, 0.});

    // Sylleptic shoots
    ShootParameters shoot_parameters_sylleptic = shoot_parameters_proleptic;
    //    shoot_parameters_sylleptic.phytomer_parameters.internode.color = RGB::red;
    shoot_parameters_sylleptic.phytomer_parameters.internode.image_texture = "";
    shoot_parameters_sylleptic.phytomer_parameters.leaf.prototype_scale = 0.12;
    shoot_parameters_sylleptic.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20);
    shoot_parameters_sylleptic.insertion_angle_tip = 0;
    shoot_parameters_sylleptic.insertion_angle_decay_rate = 0;
    shoot_parameters_sylleptic.phyllochron_min = 1;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_min = 0.0;
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.6;
    shoot_parameters_sylleptic.gravitropic_curvature = 600;
    shoot_parameters_sylleptic.internode_length_max = 0.02;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true;
    shoot_parameters_sylleptic.defineChildShootTypes({"proleptic"}, {1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    // Scaffolds are older wood than proleptic shoots, so they need their own factor to hold their base diameter.
    shoot_parameters_scaffold.girth_area_factor = 1.95f;
    //    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 15;
    shoot_parameters_scaffold.gravitropic_curvature = 100;
    shoot_parameters_scaffold.internode_length_max = 0.02;
    shoot_parameters_scaffold.tortuosity = 50;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);
}

uint PlantArchitecture::buildAlmondTreeWoodColony(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize almond tree shoots
        initializeAlmondTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    //    enableEpicormicChildShoots(plantID,"sylleptic",0.001);

    // The trunk is built node by node so it can lean and bow like an orchard trunk; see almondTrunkPath().
    std::vector<vec3> trunk_node_positions;
    std::vector<float> trunk_node_radii;
    almondTrunkPath(context_ptr, base_position, 14, 0.03f, 0.015f, trunk_node_positions, trunk_node_radii);
    uint uID_trunk = addShootFromNodePositions(plantID, -1, 0, trunk_node_positions, trunk_node_radii, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0.01, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 5; // context_ptr->randu(4,5);

    for (int i = 0; i < Nscaffolds; i++) {
        float pitch = context_ptr->randu(deg2rad(45), deg2rad(60));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, context_ptr->randu(7, 9), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007, 0.06,
                                       1.f, 1.f, 0.5, "scaffold", 0);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 90, -1, 3, 7, 20, 275);
    plant_instances.at(plantID).max_age = 1825;

    return plantID;
}

void PlantArchitecture::initializeAppleTreeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "AppleLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.6f;
    leaf_prototype.midrib_fold_fraction = 0.4f;
    leaf_prototype.longitudinal_curvature = -0.3f;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 3;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_apple(context_ptr->getRandomGenerator());

    phytomer_parameters_apple.internode.pitch = 0;
    phytomer_parameters_apple.internode.phyllotactic_angle.uniformDistribution(130, 145);
    phytomer_parameters_apple.internode.radius_initial = 0.004;
    phytomer_parameters_apple.internode.length_segments = 1;
    phytomer_parameters_apple.internode.image_texture = "AppleBark.jpg";
    phytomer_parameters_apple.internode.max_floral_buds_per_petiole = 1;

    phytomer_parameters_apple.petiole.petioles_per_internode = 1;
    phytomer_parameters_apple.petiole.pitch.uniformDistribution(-40, -25);
    phytomer_parameters_apple.petiole.taper = 0.1;
    phytomer_parameters_apple.petiole.curvature = 0;
    phytomer_parameters_apple.petiole.length = 0.04;
    phytomer_parameters_apple.petiole.radius = 0.00075;
    phytomer_parameters_apple.petiole.length_segments = 1;
    phytomer_parameters_apple.petiole.radial_subdivisions = 3;
    phytomer_parameters_apple.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_apple.leaf.leaves_per_petiole = 1;
    phytomer_parameters_apple.leaf.prototype_scale = 0.12;
    phytomer_parameters_apple.leaf.prototype = leaf_prototype;

    phytomer_parameters_apple.peduncle.length = 0.04;
    phytomer_parameters_apple.peduncle.radius = 0.001;
    phytomer_parameters_apple.peduncle.pitch = 90;
    phytomer_parameters_apple.peduncle.roll = 90;
    phytomer_parameters_apple.peduncle.length_segments = 1;

    phytomer_parameters_apple.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_apple.inflorescence.pitch = 0;
    phytomer_parameters_apple.inflorescence.roll = 0;
    phytomer_parameters_apple.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_apple.inflorescence.flower_prototype_function = AppleFlowerPrototype;
    phytomer_parameters_apple.inflorescence.fruit_prototype_scale = 0.1;
    phytomer_parameters_apple.inflorescence.fruit_prototype_function = AppleFruitPrototype;
    phytomer_parameters_apple.inflorescence.fruit_gravity_factor_fraction = 0.5;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_apple;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.01;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 24;
    shoot_parameters_trunk.max_nodes = 20;
    // Rescaled when the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(), to keep base diameters at
    // max_age where they were: girth is now proportional to cumulative downstream leaf area, which runs well above the old value.
    shoot_parameters_trunk.girth_area_factor = 1.4f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 20;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"proleptic"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_apple;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = ApplePhytomerCreationFunction;
    shoot_parameters_proleptic.max_nodes = 40;
    shoot_parameters_proleptic.max_nodes_per_season = 20;
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.15;
    shoot_parameters_proleptic.girth_area_factor = 4.8f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.4;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(450, 500);
    shoot_parameters_proleptic.tortuosity = 75;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(30, 40);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 20;
    shoot_parameters_proleptic.internode_length_max = 0.04;
    shoot_parameters_proleptic.internode_length_min = 0.01;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.3;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 1;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"}, {1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);
}

uint PlantArchitecture::buildAppleTree(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize apple tree shoots
        initializeAppleTreeShoots();
    }

    // Get training system parameters (with defaults matching original hard-coded values)
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 0.8f, 0.1f, 3.f, "total trunk height in meters");
    auto num_scaffolds = uint(getParameterValue(current_build_parameters, "num_scaffolds", 4.f, 2.f, 8.f, "number of scaffold branches"));
    auto scaffold_angle = getParameterValue(current_build_parameters, "scaffold_angle", 40.f, 20.f, 70.f, "scaffold branch angle in degrees");

    // Calculate trunk nodes based on desired height and internode length
    float trunk_internode_length = 0.04f; // Default internode length for apple
    uint trunk_nodes = uint(trunk_height / trunk_internode_length);
    if (trunk_nodes < 1)
        trunk_nodes = 1;

    // Fixed training parameters (not user-customizable)
    float trunk_radius = 0.015f;
    float scaffold_radius = 0.005f;
    float scaffold_length = 0.04f;

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), trunk_radius, trunk_internode_length, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // Hard-coded scaffold node range (not user-customizable)
    uint scaffold_nodes_min = 7;
    uint scaffold_nodes_max = 9;

    for (int i = 0; i < num_scaffolds; i++) {
        float pitch = deg2rad(scaffold_angle) + context_ptr->randu(-0.1f, 0.1f); // Small randomness around specified angle
        uint scaffold_nodes = context_ptr->randu(int(scaffold_nodes_min), int(scaffold_nodes_max));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, scaffold_nodes, make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(num_scaffolds) * 2 * M_PI, 0), scaffold_radius,
                                       scaffold_length, 1.f, 1.f, 0.5, "proleptic", 0);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200);
    plant_instances.at(plantID).max_age = 1460;

    return plantID;
}

void PlantArchitecture::initializeAppleFruitingWallShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "AppleLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.6f;
    leaf_prototype.midrib_fold_fraction = 0.4f;
    leaf_prototype.longitudinal_curvature = -0.3f;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 3;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_apple(context_ptr->getRandomGenerator());

    phytomer_parameters_apple.internode.pitch = 0;
    phytomer_parameters_apple.internode.phyllotactic_angle.uniformDistribution(130, 145);
    phytomer_parameters_apple.internode.radius_initial = 0.004;
    phytomer_parameters_apple.internode.length_segments = 1;
    phytomer_parameters_apple.internode.image_texture = "AppleBark.jpg";
    phytomer_parameters_apple.internode.max_floral_buds_per_petiole = 1;

    phytomer_parameters_apple.petiole.petioles_per_internode = 1;
    phytomer_parameters_apple.petiole.pitch.uniformDistribution(-40, -25);
    phytomer_parameters_apple.petiole.taper = 0.1;
    phytomer_parameters_apple.petiole.curvature = 0;
    phytomer_parameters_apple.petiole.length = 0.04;
    phytomer_parameters_apple.petiole.radius = 0.00075;
    phytomer_parameters_apple.petiole.length_segments = 1;
    phytomer_parameters_apple.petiole.radial_subdivisions = 3;
    phytomer_parameters_apple.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_apple.leaf.leaves_per_petiole = 1;
    phytomer_parameters_apple.leaf.prototype_scale = 0.12;
    phytomer_parameters_apple.leaf.prototype = leaf_prototype;

    phytomer_parameters_apple.peduncle.length = 0.04;
    phytomer_parameters_apple.peduncle.radius = 0.001;
    phytomer_parameters_apple.peduncle.pitch = 90;
    phytomer_parameters_apple.peduncle.roll = 90;
    phytomer_parameters_apple.peduncle.length_segments = 1;

    phytomer_parameters_apple.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_apple.inflorescence.pitch = 0;
    phytomer_parameters_apple.inflorescence.roll = 0;
    phytomer_parameters_apple.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_apple.inflorescence.flower_prototype_function = AppleFlowerPrototype;
    phytomer_parameters_apple.inflorescence.fruit_prototype_scale = 0.1;
    phytomer_parameters_apple.inflorescence.fruit_prototype_function = AppleFruitPrototype;
    phytomer_parameters_apple.inflorescence.fruit_gravity_factor_fraction = 0.5;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_apple;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(175, 185);
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.01;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 24;
    shoot_parameters_trunk.max_nodes = 30;
    // Rescaled when the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(), to keep base diameters at
    // max_age where they were: girth is now proportional to cumulative downstream leaf area, which runs well above the old value.
    shoot_parameters_trunk.girth_area_factor = 3.9f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 10;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"lateral"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_apple;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = ApplePhytomerCreationFunction;
    shoot_parameters_proleptic.max_nodes = 20;
    shoot_parameters_proleptic.max_nodes_per_season = 20;
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.15;
    shoot_parameters_proleptic.girth_area_factor = 6.8f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.4;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(550, 700);
    shoot_parameters_proleptic.tortuosity = 75;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(70, 110);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 20;
    shoot_parameters_proleptic.internode_length_max = 0.04;
    shoot_parameters_proleptic.internode_length_min = 0.01;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.3;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 1;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"}, {1.0});

    // lateral shoots
    ShootParameters shoot_parameters_lateral = shoot_parameters_proleptic;
    // Laterals are older wood than proleptic shoots, so they need their own factor to hold their base diameter.
    shoot_parameters_lateral.girth_area_factor = 5.5f;
    shoot_parameters_lateral.gravitropic_curvature = 0;
    shoot_parameters_lateral.phytomer_parameters.internode.phyllotactic_angle = 360;

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("lateral", shoot_parameters_lateral);
}

uint PlantArchitecture::buildAppleFruitingWall(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize apple tree shoots
        initializeAppleFruitingWallShoots();
    }

    std::vector<std::vector<vec3>> trellis_points;

    std::vector<float> wire_heights{0.75, 1.6, 2.25, 2.8};
    float tree_spacing = 1.2;

    trellis_points.reserve(wire_heights.size());
    for (int i = 0; i < wire_heights.size(); i++) {
        trellis_points.push_back(linspace(make_vec3(0, -0.5f * tree_spacing, wire_heights.at(i)), make_vec3(0, 0.5f * tree_spacing, wire_heights.at(i)), 8));
        // context_ptr->addTube( 4, {base_position+trellis_points.back().at(0), base_position+trellis_points.back().back()}, {0.005, 0.005}, {RGB::silver, RGB::silver});
    }

    for (int j = 0; j < trellis_points.size(); j++) {
        for (int i = 0; i < trellis_points[j].size(); i++) {
            trellis_points.at(j).at(i) += base_position;
        }
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_leader = addBaseStemShoot(plantID, std::ceil(wire_heights.back() / 0.1), make_AxisRotation(context_ptr->randu(0, 0.05 * M_PI), 0, 0), shoot_types.at("trunk").phytomer_parameters.internode.radius_initial.val(), 0.09, 1, 1, 0.1, "trunk");
    plant_instances.at(plantID).shoot_tree.at(uID_leader)->meristem_is_alive = false;
    removeShootLeaves(plantID, uID_leader);
    removeShootVegetativeBuds(plantID, uID_leader);
    removeShootFloralBuds(plantID, uID_leader);

    for (int i = 0; i < 8; i++) {
        float z;
        if (i == 0) {
            z = wire_heights.front() - 0.1;
        } else {
            z = context_ptr->randu(wire_heights.front() - 0.1, wire_heights.at(wire_heights.size() - 1));
        }
        int node = std::round(z / 0.1) - 1;
        addChildShoot(plantID, uID_leader, node, 8, make_AxisRotation(context_ptr->randu(float(0.45f * M_PI), 0.52f * M_PI), context_ptr->randu(0, 1) * M_PI, M_PI), 0.005, 0.05, 1, 1, 0.5, "lateral");
    }

    // Set plant-specific attraction points for this grapevine's trellis system
    setPlantAttractionPoints(plantID, flatten(trellis_points), 45.f, 1.f, 0.8);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200);
    plant_instances.at(plantID).max_age = 730;

    return plantID;
}

void PlantArchitecture::initializeAsparagusShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 1;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(127.5, 147.5);
    phytomer_parameters.internode.radius_initial = 0.00025;
    phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters.internode.color = RGB::forestgreen;
    phytomer_parameters.internode.length_segments = 2;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch = 90;
    phytomer_parameters.petiole.radius = 0.0001;
    phytomer_parameters.petiole.length = 0.0005;
    phytomer_parameters.petiole.taper = 0.5;
    phytomer_parameters.petiole.curvature = 0;
    phytomer_parameters.petiole.color = phytomer_parameters.internode.color;
    phytomer_parameters.petiole.length_segments = 1;
    phytomer_parameters.petiole.radial_subdivisions = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 5;
    phytomer_parameters.leaf.pitch.normalDistribution(-5, 30);
    phytomer_parameters.leaf.yaw = 30;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.leaflet_offset = 0;
    phytomer_parameters.leaf.leaflet_scale = 0.9;
    phytomer_parameters.leaf.prototype.prototype_function = AsparagusLeafPrototype;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.018, 0.02);
    //    phytomer_parameters.leaf.subdivisions = 6;

    phytomer_parameters.peduncle.length = 0.17;
    phytomer_parameters.peduncle.radius = 0.0015;
    phytomer_parameters.peduncle.pitch.uniformDistribution(0, 30);
    phytomer_parameters.peduncle.roll = 90;
    phytomer_parameters.peduncle.curvature.uniformDistribution(50, 250);
    phytomer_parameters.peduncle.length_segments = 6;
    phytomer_parameters.peduncle.radial_subdivisions = 6;

    phytomer_parameters.inflorescence.flowers_per_peduncle.uniformDistribution(1, 3);
    phytomer_parameters.inflorescence.flower_offset = 0.;
    phytomer_parameters.inflorescence.pitch.uniformDistribution(50, 70);
    phytomer_parameters.inflorescence.roll.uniformDistribution(-20, 20);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.015;
    phytomer_parameters.inflorescence.fruit_prototype_scale.uniformDistribution(0.02, 0.025);
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0., 0.5);

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.phytomer_parameters.phytomer_creation_function = AsparagusPhytomerCreationFunction;

    shoot_parameters.max_nodes = 20;
    shoot_parameters.insertion_angle_tip.uniformDistribution(40, 70);
    //    shoot_parameters.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters.internode_length_max = 0.015;
    //    shoot_parameters.child_internode_length_min = 0.0; (default)
    //    shoot_parameters.child_internode_length_decay_rate = 0; (default)
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature = -200;

    shoot_parameters.phyllochron_min = 1;
    shoot_parameters.elongation_rate_max = 0.15;
    //    shoot_parameters.girth_growth_rate = 0.00005;
    shoot_parameters.girth_area_factor = 30;
    shoot_parameters.vegetative_bud_break_time = 5;
    shoot_parameters.vegetative_bud_break_probability_min = 0.25;
    //    shoot_parameters.max_terminal_floral_buds = 0; (default)
    //    shoot_parameters.flower_bud_break_probability.uniformDistribution(0.1, 0.2);
    shoot_parameters.fruit_set_probability = 0.;
    //    shoot_parameters.flowers_require_dormancy = false; (default)
    //    shoot_parameters.growth_requires_dormancy = false; (default)
    //    shoot_parameters.determinate_shoot_growth = true; (default)

    shoot_parameters.defineChildShootTypes({"main"}, {1.0});

    defineShootType("main", shoot_parameters);
}

uint PlantArchitecture::buildAsparagusPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize asparagus plant shoots
        initializeAsparagusShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(0, 0.f, 0.f), 0.0003, 0.015, 1, 0.1, 0.1, "main");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, 5, 100);

    plant_instances.at(plantID).max_age = 20;

    return plantID;
}

void PlantArchitecture::initializeBindweedShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_bindweed(context_ptr->getRandomGenerator());

    phytomer_parameters_bindweed.internode.pitch.uniformDistribution(0, 15);
    phytomer_parameters_bindweed.internode.phyllotactic_angle = 180.f;
    phytomer_parameters_bindweed.internode.radius_initial = 0.0012;
    phytomer_parameters_bindweed.internode.color = make_RGBcolor(0.3, 0.38, 0.21);
    phytomer_parameters_bindweed.internode.length_segments = 1;

    phytomer_parameters_bindweed.petiole.petioles_per_internode = 1;
    phytomer_parameters_bindweed.petiole.pitch.uniformDistribution(80, 100);
    phytomer_parameters_bindweed.petiole.radius = 0.001;
    phytomer_parameters_bindweed.petiole.length = 0.006;
    phytomer_parameters_bindweed.petiole.taper = 0;
    phytomer_parameters_bindweed.petiole.curvature = 0;
    phytomer_parameters_bindweed.petiole.color = phytomer_parameters_bindweed.internode.color;
    phytomer_parameters_bindweed.petiole.length_segments = 1;

    phytomer_parameters_bindweed.leaf.leaves_per_petiole = 1;
    phytomer_parameters_bindweed.leaf.pitch.uniformDistribution(5, 30);
    phytomer_parameters_bindweed.leaf.yaw = 0;
    phytomer_parameters_bindweed.leaf.roll = 0;
    phytomer_parameters_bindweed.leaf.prototype_scale = 0.05;
    phytomer_parameters_bindweed.leaf.prototype.OBJ_model_file = "BindweedLeaf.obj";

    phytomer_parameters_bindweed.peduncle.length = 0.01;
    phytomer_parameters_bindweed.peduncle.radius = 0.0005;
    phytomer_parameters_bindweed.peduncle.color = phytomer_parameters_bindweed.internode.color;

    phytomer_parameters_bindweed.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_bindweed.inflorescence.pitch = -90.f;
    phytomer_parameters_bindweed.inflorescence.flower_prototype_function = BindweedFlowerPrototype;
    phytomer_parameters_bindweed.inflorescence.flower_prototype_scale = 0.045;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_primary(context_ptr->getRandomGenerator());
    shoot_parameters_primary.phytomer_parameters = phytomer_parameters_bindweed;
    shoot_parameters_primary.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_primary.vegetative_bud_break_probability_decay_rate = -1.;
    shoot_parameters_primary.vegetative_bud_break_time = 10;
    shoot_parameters_primary.base_roll = 90;
    shoot_parameters_primary.phyllochron_min = 1;
    shoot_parameters_primary.elongation_rate_max = 0.25;
    shoot_parameters_primary.girth_area_factor = 0;
    shoot_parameters_primary.internode_length_max = 0.03;
    shoot_parameters_primary.internode_length_decay_rate = 0;
    shoot_parameters_primary.insertion_angle_tip.uniformDistribution(70, 80);
    shoot_parameters_primary.flowers_require_dormancy = false;
    shoot_parameters_primary.growth_requires_dormancy = false;
    shoot_parameters_primary.flower_bud_break_probability = 0.2;
    shoot_parameters_primary.determinate_shoot_growth = false;
    shoot_parameters_primary.max_nodes = 15;
    shoot_parameters_primary.gravitropic_curvature = 30;
    shoot_parameters_primary.tortuosity = 0;
    shoot_parameters_primary.defineChildShootTypes({"secondary_bindweed"}, {1.f});

    ShootParameters shoot_parameters_base = shoot_parameters_primary;
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_bindweed;
    shoot_parameters_base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5 - 10, 137.5 + 10);
    shoot_parameters_base.phytomer_parameters.petiole.petioles_per_internode = 0;
    shoot_parameters_base.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_base.phytomer_parameters.petiole.pitch = 0;
    shoot_parameters_base.vegetative_bud_break_probability_min = 1.0;
    shoot_parameters_base.vegetative_bud_break_time = 2;
    shoot_parameters_base.phyllochron_min = 2;
    shoot_parameters_base.elongation_rate_max = 0.15;
    shoot_parameters_base.girth_area_factor = 0.f;
    shoot_parameters_base.gravitropic_curvature = 0;
    shoot_parameters_base.internode_length_max = 0.01;
    shoot_parameters_base.internode_length_decay_rate = 0;
    shoot_parameters_base.insertion_angle_tip = 95;
    shoot_parameters_base.insertion_angle_decay_rate = 0;
    shoot_parameters_base.flowers_require_dormancy = false;
    shoot_parameters_base.growth_requires_dormancy = false;
    shoot_parameters_base.flower_bud_break_probability = 0.0;
    shoot_parameters_base.max_nodes.uniformDistribution(3, 5);
    shoot_parameters_base.defineChildShootTypes({"primary_bindweed"}, {1.f});

    ShootParameters shoot_parameters_children = shoot_parameters_primary;
    shoot_parameters_children.base_roll = 0;
    shoot_parameters_children.insertion_angle_tip.uniformDistribution(50, 80);

    defineShootType("base_bindweed", shoot_parameters_base);
    defineShootType("primary_bindweed", shoot_parameters_primary);
    defineShootType("secondary_bindweed", shoot_parameters_children);
}

uint PlantArchitecture::buildBindweedPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize bindweed plant shoots
        initializeBindweedShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0.f, 0.f), 0.001, 0.001, 1, 1, 0, "base_bindweed");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 14, -1, -1, 1000);

    plant_instances.at(plantID).max_age = 50;

    return plantID;
}

void PlantArchitecture::initializeBeanShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype_trifoliate(context_ptr->getRandomGenerator());
    leaf_prototype_trifoliate.leaf_texture_file[0] = "BeanLeaf_tip.png";
    // Leaflet index -1 is the leaflet to the right of the petiole when the leaf is seen from above looking toward its tip, and +1 the one to its left, so the image named for each side goes with the
    // index of the opposite sign. Assigned the other way round, the broad half of each lateral blade faces in toward the terminal leaflet rather than out toward the petiole base.
    leaf_prototype_trifoliate.leaf_texture_file[-1] = "BeanLeaf_right_centered.png";
    leaf_prototype_trifoliate.leaf_texture_file[1] = "BeanLeaf_left_centered.png";
    leaf_prototype_trifoliate.leaf_aspect_ratio = 1.f;
    leaf_prototype_trifoliate.midrib_fold_fraction = 0.2;
    leaf_prototype_trifoliate.longitudinal_curvature.uniformDistribution(-0.3f, -0.2f);
    leaf_prototype_trifoliate.lateral_curvature = -1.f;
    leaf_prototype_trifoliate.subdivisions = 6;
    leaf_prototype_trifoliate.unique_prototypes = 5;
    leaf_prototype_trifoliate.build_petiolule = true;
    // Petiolule length as a fraction of blade length: about 2 mm, within the 1.5-2.5 mm given by Flora of Pakistan for Phaseolus vulgaris. Its radius follows the petiole.
    leaf_prototype_trifoliate.petiolule_length = {{0, 0.02f}};

    LeafPrototype leaf_prototype_unifoliate = leaf_prototype_trifoliate;
    leaf_prototype_unifoliate.leaf_texture_file.clear();
    leaf_prototype_unifoliate.leaf_texture_file[0] = "BeanLeaf_unifoliate_centered.png";
    leaf_prototype_unifoliate.unique_prototypes = 2;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());

    phytomer_parameters_trifoliate.internode.pitch = 20;
    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters_trifoliate.internode.radius_initial = 0.001;
    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.2, 0.25, 0.05);
    phytomer_parameters_trifoliate.internode.length_segments = 2;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(20, 50);
    phytomer_parameters_trifoliate.petiole.radius = 0.0015;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.1, 0.14);
    phytomer_parameters_trifoliate.petiole.taper = 0.;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-100, 200);
    phytomer_parameters_trifoliate.petiole.color = make_RGBcolor(0.28, 0.35, 0.07);
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.normalDistribution(0, 20);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll = -15;
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.3;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.09, 0.11);
    phytomer_parameters_trifoliate.leaf.prototype = leaf_prototype_trifoliate;

    phytomer_parameters_trifoliate.peduncle.length = 0.04;
    phytomer_parameters_trifoliate.peduncle.radius = 0.00075;
    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 40);
    phytomer_parameters_trifoliate.peduncle.roll = 90;
    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(-500, 500);
    phytomer_parameters_trifoliate.peduncle.color = phytomer_parameters_trifoliate.petiole.color;
    phytomer_parameters_trifoliate.peduncle.length_segments = 1;
    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.inflorescence.flowers_per_peduncle.uniformDistribution(1, 4);
    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.2;
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50, 70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20, 20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = BeanFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.15, 0.2);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = BeanFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8, 1.0);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.0001;
    phytomer_parameters_unifoliate.petiole.radius = 0.0004;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(50, 70);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.04;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype = leaf_prototype_unifoliate;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 25;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40, 60);
    //    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.025;
    //    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
    //    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters_trifoliate.gravitropic_curvature = 200;

    shoot_parameters_trifoliate.phyllochron_min = 2;
    shoot_parameters_trifoliate.elongation_rate_max = 0.1;
    shoot_parameters_trifoliate.girth_area_factor = 1.5f;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 15;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_decay_rate = -0.4;
    //    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.3, 0.4);
    shoot_parameters_trifoliate.fruit_set_probability = 0.4;
    //    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
    //    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
    //    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)

    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"}, {1.0});


    ShootParameters shoot_parameters_unifoliate = shoot_parameters_trifoliate;
    shoot_parameters_unifoliate.phytomer_parameters = phytomer_parameters_unifoliate;
    shoot_parameters_unifoliate.phytomer_parameters.phytomer_creation_function = nullptr;
    shoot_parameters_unifoliate.max_nodes = 1;
    shoot_parameters_unifoliate.girth_area_factor = 1.f;
    shoot_parameters_unifoliate.vegetative_bud_break_probability_min = 1.0;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 50;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 8;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"}, {1.0});

    defineShootType("unifoliate", shoot_parameters_unifoliate);
    defineShootType("trifoliate", shoot_parameters_trifoliate);
}

uint PlantArchitecture::buildBeanPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize bean plant shoots
        initializeBeanShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.03, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").phytomer_parameters.internode.radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeBougainvilleaShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "CapsicumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.8f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.15, -0.05f);
    leaf_prototype.lateral_curvature = -0.25f;
    leaf_prototype.wave_period = 0.35f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.subdivisions = 7;
    leaf_prototype.unique_prototypes = 5;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(160, 190);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.213, 0.270, 0.056);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(-60, -40);
    phytomer_parameters.petiole.radius = 0.0001;
    phytomer_parameters.petiole.length = 0.0001;
    phytomer_parameters.petiole.taper = 1;
    phytomer_parameters.petiole.curvature = 0;
    phytomer_parameters.petiole.color = phytomer_parameters.internode.color;
    phytomer_parameters.petiole.length_segments = 1;

    phytomer_parameters.leaf.leaves_per_petiole = 1;
    phytomer_parameters.leaf.pitch = 0;
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.08, 0.12);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.15;
    phytomer_parameters.peduncle.radius = 0.0015;
    phytomer_parameters.peduncle.pitch.uniformDistribution(70, 90);
    phytomer_parameters.peduncle.roll.uniformDistribution(0, 180);
    phytomer_parameters.peduncle.curvature.uniformDistribution(-200, 200);
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 3;
    phytomer_parameters.peduncle.radial_subdivisions = 6;

    phytomer_parameters.inflorescence.flowers_per_peduncle.uniformDistribution(4, 6);
    phytomer_parameters.inflorescence.pitch = 80;
    phytomer_parameters.inflorescence.roll.uniformDistribution(-180, 180);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.05;
    phytomer_parameters.inflorescence.flower_prototype_function = BougainvilleaFlowerPrototype;
    phytomer_parameters.inflorescence.unique_prototypes = 1;
    phytomer_parameters.inflorescence.flower_offset = 0.12;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;

    shoot_parameters.max_nodes = 60;
    shoot_parameters.insertion_angle_tip = 25;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.04;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature = 500;
    shoot_parameters.tortuosity = 75;

    shoot_parameters.phyllochron_min = 2.75;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 1.f;
    shoot_parameters.vegetative_bud_break_time = 30;
    shoot_parameters.vegetative_bud_break_probability_min = 0.075;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = 0.8;
    shoot_parameters.flower_bud_break_probability = 0.5;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = false;

    shoot_parameters.defineChildShootTypes({"secondary"}, {1.0});

    defineShootType("mainstem", shoot_parameters);

    ShootParameters shoot_parameters_secondary = shoot_parameters;
    shoot_parameters_secondary.max_nodes = 15;

    defineShootType("secondary", shoot_parameters_secondary);
}

uint PlantArchitecture::buildBougainvilleaPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize bougainvillea plant shoots
        initializeBougainvilleaShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 3, base_rotation, 0.002, shoot_types.at("mainstem").internode_length_max.val(), 0.01, 0.01, 0, "mainstem");

    removeShootVegetativeBuds(plantID, uID_stem);
    removeShootFloralBuds(plantID, uID_stem);
    removeShootLeaves(plantID, uID_stem);

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 75, 1000, 1000, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeCapsicumShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "CapsicumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.45f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.15, -0.05f);
    leaf_prototype.lateral_curvature = -0.15f;
    leaf_prototype.wave_period = 0.35f;
    leaf_prototype.wave_amplitude = 0.0f;
    leaf_prototype.subdivisions = 5;
    leaf_prototype.unique_prototypes = 5;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5 - 10, 137.5 + 10);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.213, 0.270, 0.056);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(-60, -40);
    phytomer_parameters.petiole.radius = 0.0001;
    phytomer_parameters.petiole.length = 0.0001;
    phytomer_parameters.petiole.taper = 1;
    phytomer_parameters.petiole.curvature = 0;
    phytomer_parameters.petiole.color = phytomer_parameters.internode.color;
    phytomer_parameters.petiole.length_segments = 1;

    phytomer_parameters.leaf.leaves_per_petiole = 1;
    phytomer_parameters.leaf.pitch = 0;
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12, 0.15);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.01;
    phytomer_parameters.peduncle.radius = 0.001;
    phytomer_parameters.peduncle.pitch.uniformDistribution(10, 30);
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -700;
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 3;
    phytomer_parameters.peduncle.radial_subdivisions = 6;

    phytomer_parameters.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters.inflorescence.pitch = 20;
    phytomer_parameters.inflorescence.roll.uniformDistribution(-30, 30);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.025;
    phytomer_parameters.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale.uniformDistribution(0.12, 0.16);
    phytomer_parameters.inflorescence.fruit_prototype_function = CapsicumFruitPrototype;
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction = 0.9;
    phytomer_parameters.inflorescence.unique_prototypes = 10;

    PhytomerParameters phytomer_parameters_secondary = phytomer_parameters;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.phytomer_parameters.phytomer_creation_function = CapsicumPhytomerCreationFunction;

    shoot_parameters.max_nodes = 30;
    shoot_parameters.insertion_angle_tip = 35;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.04;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature = 300;
    shoot_parameters.tortuosity = 75;

    shoot_parameters.phyllochron_min = 3;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 2.f;
    shoot_parameters.vegetative_bud_break_time = 30;
    shoot_parameters.vegetative_bud_break_probability_min = 0.4;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = 0;
    shoot_parameters.flower_bud_break_probability = 0.5;
    shoot_parameters.fruit_set_probability = 0.05;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = true;

    shoot_parameters.defineChildShootTypes({"secondary"}, {1.0});

    defineShootType("mainstem", shoot_parameters);

    ShootParameters shoot_parameters_secondary = shoot_parameters;
    shoot_parameters_secondary.phytomer_parameters = phytomer_parameters_secondary;
    shoot_parameters_secondary.max_nodes = 7;
    shoot_parameters_secondary.phyllochron_min = 6;
    shoot_parameters_secondary.vegetative_bud_break_probability_min = 0.2;

    defineShootType("secondary", shoot_parameters_secondary);
}

uint PlantArchitecture::buildCapsicumPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize capsicum plant shoots
        initializeCapsicumShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.002, shoot_types.at("mainstem").internode_length_max.val(), 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 75, 14, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeCheeseweedShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.OBJ_model_file = "CheeseweedLeaf.obj";

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_cheeseweed(context_ptr->getRandomGenerator());

    phytomer_parameters_cheeseweed.internode.pitch = 0;
    phytomer_parameters_cheeseweed.internode.phyllotactic_angle.uniformDistribution(127.5f, 147.5);
    phytomer_parameters_cheeseweed.internode.radius_initial = 0.0005;
    phytomer_parameters_cheeseweed.internode.color = make_RGBcolor(0.60, 0.65, 0.40);
    phytomer_parameters_cheeseweed.internode.length_segments = 1;

    phytomer_parameters_cheeseweed.petiole.petioles_per_internode = 1;
    phytomer_parameters_cheeseweed.petiole.pitch.uniformDistribution(45, 75);
    phytomer_parameters_cheeseweed.petiole.radius = 0.0005;
    phytomer_parameters_cheeseweed.petiole.length.uniformDistribution(0.02, 0.06);
    phytomer_parameters_cheeseweed.petiole.taper = 0;
    phytomer_parameters_cheeseweed.petiole.curvature = -300;
    phytomer_parameters_cheeseweed.petiole.length_segments = 5;
    phytomer_parameters_cheeseweed.petiole.color = phytomer_parameters_cheeseweed.internode.color;

    phytomer_parameters_cheeseweed.leaf.leaves_per_petiole = 1;
    phytomer_parameters_cheeseweed.leaf.pitch.uniformDistribution(-30, 0);
    phytomer_parameters_cheeseweed.leaf.yaw = 0;
    phytomer_parameters_cheeseweed.leaf.roll = 0;
    phytomer_parameters_cheeseweed.leaf.prototype_scale = 0.035;
    phytomer_parameters_cheeseweed.leaf.prototype = leaf_prototype;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_base(context_ptr->getRandomGenerator());
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_cheeseweed;
    shoot_parameters_base.vegetative_bud_break_probability_min = 0.2;
    shoot_parameters_base.vegetative_bud_break_time = 6;
    shoot_parameters_base.phyllochron_min = 2;
    shoot_parameters_base.elongation_rate_max = 0.1;
    shoot_parameters_base.girth_area_factor = 10.f;
    shoot_parameters_base.gravitropic_curvature = 0;
    shoot_parameters_base.internode_length_max = 0.0015;
    shoot_parameters_base.internode_length_decay_rate = 0;
    shoot_parameters_base.flowers_require_dormancy = false;
    shoot_parameters_base.growth_requires_dormancy = false;
    shoot_parameters_base.flower_bud_break_probability = 0.;
    shoot_parameters_base.max_nodes = 8;

    defineShootType("base", shoot_parameters_base);
}

uint PlantArchitecture::buildCheeseweedPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize cheeseweed plant shoots
        initializeCheeseweedShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(0, 0.f, 0.f), 0.0001, 0.0025, 0.1, 0.1, 0, "base");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000);

    plant_instances.at(plantID).max_age = 40;

    return plantID;
}

void PlantArchitecture::initializeCowpeaShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype_trifoliate(context_ptr->getRandomGenerator());
    leaf_prototype_trifoliate.leaf_texture_file[0] = "CowpeaLeaf_tip_centered.png";
    // Leaflet index -1 is the leaflet to the right of the petiole when the leaf is seen from above looking toward its tip, and +1 the one to its left, so the image named for each side goes with the
    // index of the opposite sign. Assigned the other way round, the broad half of each lateral blade faces in toward the terminal leaflet rather than out toward the petiole base.
    leaf_prototype_trifoliate.leaf_texture_file[-1] = "CowpeaLeaf_right_centered.png";
    leaf_prototype_trifoliate.leaf_texture_file[1] = "CowpeaLeaf_left_centered.png";
    leaf_prototype_trifoliate.leaf_aspect_ratio = 0.7f;
    leaf_prototype_trifoliate.midrib_fold_fraction = 0.2;
    leaf_prototype_trifoliate.longitudinal_curvature.uniformDistribution(-0.3f, -0.1f);
    leaf_prototype_trifoliate.lateral_curvature = -0.4f;
    leaf_prototype_trifoliate.subdivisions = 6;
    leaf_prototype_trifoliate.unique_prototypes = 5;
    leaf_prototype_trifoliate.build_petiolule = true;
    // Petiolule length as a fraction of blade length: about 3.5 mm, within the 2-5 mm given by Flora of China for Vigna unguiculata. Its radius follows the petiole.
    leaf_prototype_trifoliate.petiolule_length = {{0, 0.038f}};

    LeafPrototype leaf_prototype_unifoliate = leaf_prototype_trifoliate;
    leaf_prototype_unifoliate.leaf_texture_file.clear();
    // The embryonic (unifoliate) leaves have an image of their own, drawn from the terminal leaflet's: symmetric about its midrib, as a cowpea's first pair of leaves nearly is, and rounded at the base
    // where the terminal leaflet is cut off nearly square.
    leaf_prototype_unifoliate.leaf_texture_file[0] = "CowpeaLeaf_unifoliate_centered.png";

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());

    phytomer_parameters_trifoliate.internode.pitch = 20;
    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters_trifoliate.internode.radius_initial = 0.0015;
    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.15, 0.2, 0.1);
    phytomer_parameters_trifoliate.internode.length_segments = 2;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(45, 60);
    phytomer_parameters_trifoliate.petiole.radius = 0.0018;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.06, 0.08);
    phytomer_parameters_trifoliate.petiole.taper = 0.25;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-200, -50);
    phytomer_parameters_trifoliate.petiole.color = make_RGBcolor(0.2, 0.25, 0.06);
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.normalDistribution(45, 20);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll = -15;
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.4;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.08, 0.105);
    phytomer_parameters_trifoliate.leaf.prototype = leaf_prototype_trifoliate;

    phytomer_parameters_trifoliate.peduncle.length.uniformDistribution(0.3, 0.4);
    phytomer_parameters_trifoliate.peduncle.radius = 0.00225;
    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 30);
    phytomer_parameters_trifoliate.peduncle.roll = 90;
    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(125, 200);
    phytomer_parameters_trifoliate.peduncle.color = make_RGBcolor(0.17, 0.213, 0.051);
    phytomer_parameters_trifoliate.peduncle.length_segments = 6;
    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.inflorescence.flowers_per_peduncle.uniformDistribution(1, 3);
    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.05;
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(40, 60);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20, 20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = CowpeaFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.09, 0.1);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = CowpeaFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.5, 0.7);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.0001;
    phytomer_parameters_unifoliate.petiole.radius = 0.0004;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(60, 80);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.02;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype = leaf_prototype_unifoliate;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = CowpeaPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 13;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40, 60);
    //    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.04;
    //    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
    //    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters_trifoliate.gravitropic_curvature = 200;

    shoot_parameters_trifoliate.phyllochron_min = 2;
    shoot_parameters_trifoliate.elongation_rate_max = 0.1;
    shoot_parameters_trifoliate.girth_area_factor = 1.5f;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 10;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_min = 0.2;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_decay_rate = -0.4;
    //    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.3, 0.4);
    shoot_parameters_trifoliate.fruit_set_probability = 0.4;
    //    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
    //    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
    //    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)

    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"}, {1.0});


    ShootParameters shoot_parameters_unifoliate = shoot_parameters_trifoliate;
    shoot_parameters_unifoliate.phytomer_parameters = phytomer_parameters_unifoliate;
    shoot_parameters_unifoliate.max_nodes = 1;
    shoot_parameters_unifoliate.girth_area_factor = 1.f;
    shoot_parameters_unifoliate.vegetative_bud_break_probability_min = 1;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 40;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 8;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"}, {1.0});

    defineShootType("unifoliate", shoot_parameters_unifoliate);
    defineShootType("trifoliate", shoot_parameters_trifoliate);
}

uint PlantArchitecture::buildCowpeaPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize cowpea plant shoots
        initializeCowpeaShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.03, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").phytomer_parameters.internode.radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeGrapevineVSPShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "GrapeLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.4, 0.4);
    leaf_prototype.lateral_curvature = 0;
    leaf_prototype.wave_period = 0.3f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.subdivisions = 5;
    leaf_prototype.unique_prototypes = 10;
    leaf_prototype.leaf_offset = make_vec3(-0.3, 0, 0);

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_grapevine(context_ptr->getRandomGenerator());

    phytomer_parameters_grapevine.internode.pitch = 20;
    phytomer_parameters_grapevine.internode.phyllotactic_angle.uniformDistribution(170, 190);
    phytomer_parameters_grapevine.internode.radius_initial = 0.003;
    phytomer_parameters_grapevine.internode.color = make_RGBcolor(0.23, 0.13, 0.062);
    phytomer_parameters_grapevine.internode.length_segments = 1;
    phytomer_parameters_grapevine.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_grapevine.internode.max_vegetative_buds_per_petiole = 1;

    phytomer_parameters_grapevine.petiole.petioles_per_internode = 1;
    // Grapevine petioles are green, often flushed red; the previous (0.13, 0.066, 0.03) rendered nearly black.
    phytomer_parameters_grapevine.petiole.color = make_RGBcolor(0.35, 0.38, 0.15);
    phytomer_parameters_grapevine.petiole.pitch.uniformDistribution(45, 70);
    phytomer_parameters_grapevine.petiole.radius = 0.0025;
    phytomer_parameters_grapevine.petiole.length = 0.08;
    phytomer_parameters_grapevine.petiole.taper = 0;
    phytomer_parameters_grapevine.petiole.curvature = 0;
    phytomer_parameters_grapevine.petiole.length_segments = 1;

    phytomer_parameters_grapevine.leaf.leaves_per_petiole = 1;
    phytomer_parameters_grapevine.leaf.pitch.uniformDistribution(-110, -80);
    phytomer_parameters_grapevine.leaf.yaw.uniformDistribution(-20, 20);
    phytomer_parameters_grapevine.leaf.roll.uniformDistribution(-5, 5);
    // Sized to a mean blade area of ~124 cm^2, the primary-leaf area measured by destructive whole-vine
    // defoliation on VSP Pinot noir in the Willamette Valley (121-127 cm^2, Navarrete 2015) and
    // independently on single-canopy Chenin blanc (123 cm^2, Kliewer & Dokoozlian 2005). Blade area
    // scales as the square of this parameter.
    phytomer_parameters_grapevine.leaf.prototype_scale = 0.14;
    phytomer_parameters_grapevine.leaf.prototype = leaf_prototype;

    phytomer_parameters_grapevine.peduncle.length = 0.08;
    phytomer_parameters_grapevine.peduncle.pitch.uniformDistribution(50, 90);
    // A grapevine cluster is borne on the opposite side of the node from the leaf, not in the leaf axil.
    phytomer_parameters_grapevine.peduncle.yaw = 180;
    phytomer_parameters_grapevine.peduncle.color = make_RGBcolor(0.32, 0.05, 0.13);

    phytomer_parameters_grapevine.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_grapevine.inflorescence.pitch = 0;
    //    phytomer_parameters_grapevine.inflorescence.flower_prototype_function = GrapevineFlowerPrototype;
    phytomer_parameters_grapevine.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_function = GrapevineFruitPrototype;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_gravity_factor_fraction = 0.7;

    phytomer_parameters_grapevine.phytomer_creation_function = GrapevinePhytomerCreationFunction;
    // Pulls the fruit-zone leaves at fruit set; see GrapevinePhytomerCallbackFunction().
    phytomer_parameters_grapevine.phytomer_callback_function = GrapevinePhytomerCallbackFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_main(context_ptr->getRandomGenerator());
    shoot_parameters_main.phytomer_parameters = phytomer_parameters_grapevine;
    // Flat along the shoot, rather than weighted toward the apex. A positive decay rate makes the break
    // probability depend on how many nodes the shoot has when the bud is sampled, and buds are sampled as
    // each phytomer is appended -- so with the maximum left at its default of 1 the formula sits on a
    // knife edge, giving either every bud or almost none depending on a node index that is not yet final.
    // A flat probability states the lateral density directly instead: about one bud in nine breaking over
    // a 20-node shoot puts the laterals near the 27-35% of vine leaf area measured on VSP Pinot noir
    // (Navarrete 2015).
    // About one bud in three pushes a summer lateral, which is what puts 27-35% of the vine's leaf area
    // on laterals -- the fraction measured by destructive whole-vine defoliation on VSP Pinot noir
    // (Navarrete 2015) and independently on VSP Cabernet (Kliewer & Dokoozlian 2005, 1050 of 3330 cm2
    // per shoot). Laterals are confined to the basal nodes by GrapevinePhytomerCreationFunction().
    shoot_parameters_main.vegetative_bud_break_probability_min = 0.30;
    shoot_parameters_main.vegetative_bud_break_probability_max = 0.30;
    shoot_parameters_main.vegetative_bud_break_probability_decay_rate = 0.;
    shoot_parameters_main.vegetative_bud_break_time = 30;
    shoot_parameters_main.phyllochron_min.uniformDistribution(1.75, 2.25);
    shoot_parameters_main.elongation_rate_max = 0.15;
    shoot_parameters_main.girth_area_factor = 1.f;
    // TEMPORARY DIAGNOSTIC -- revert before committing. Zeroed so the shoots show the frame they
    // emerge in, with nothing bending them afterwards. Real values: gravitropic_curvature 400, tortuosity 15.
    shoot_parameters_main.gravitropic_curvature = 300;
    shoot_parameters_main.tortuosity = 226;
    // 5.3 cm is the internode length measured on VSP shoots (Kliewer & Dokoozlian 2005, Table 3,
    // "Vertical" row), which also reports 24 nodes on an unhedged 130 cm shoot.
    shoot_parameters_main.internode_length_max.uniformDistribution(0.048, 0.058);
    shoot_parameters_main.internode_length_decay_rate = 0;
    // TEMPORARY DIAGNOSTIC -- revert before committing. Real value: insertion_angle_tip 45.
    // 90 degrees from the parent axis: on a horizontal cordon that is straight up. The spread is the
    // shoot-to-shoot variation in how upright a bud pushes.
    shoot_parameters_main.insertion_angle_tip.uniformDistribution(55, 125);
    shoot_parameters_main.insertion_angle_decay_rate = 0;
    shoot_parameters_main.flowers_require_dormancy = false;
    shoot_parameters_main.growth_requires_dormancy = false;
    shoot_parameters_main.determinate_shoot_growth = false;
    shoot_parameters_main.max_terminal_floral_buds = 0;
    shoot_parameters_main.flower_bud_break_probability = 0.5;
    // Left unset this is zero, and the vine bore no clusters at all. A shoot has three candidate cluster nodes (see
    // GrapevinePhytomerCreationFunction()), and with no flower stage in this model a bud at each becomes a cluster with probability
    // flower_bud_break_probability * fruit_set_probability, so this gives 1.5 clusters per shoot on average, within the one to three a
    // fruitful shoot carries.
    shoot_parameters_main.fruit_set_probability = 1.0;
    // Commercial VSP is hedged about 15 cm above the top wire, so a shoot does not run to the 24 nodes it
    // would reach unhedged. There is no hedging operation in the model, so the cap stands in for one: with
    // 20 nodes the mean shoot tip sits about 0.8 m above the cordon and the tallest shoots reach 1.8-2.0 m,
    // matching the ~0.9-1.0 m canopy and ~1.9 m hedged height of commercial VSP. The count also has to
    // leave room for the summer laterals, which carry a further third of the vine's leaf area on top of
    // what the primary axis bears.
    shoot_parameters_main.max_nodes = 20;
    // Roll spins the shoot about its own axis, which sets the azimuth its two-ranked leaf plane faces.
    // Grapevine phyllotaxy is near 180 degrees, so each shoot carries its leaves in one flat plane; with
    // a single fixed roll every shoot on the vine presented that plane the same way and the leaves lined
    // up in ranks across the row. Drawing a roll per shoot spreads the planes around.
    shoot_parameters_main.base_roll.uniformDistribution(90 - 15, 90 + 15);
    // Yaw turns the insertion away from the vertical plane of the row, so a spread about zero fans the
    // shoots to either side of the wire instead of stacking them all in one plane.
    shoot_parameters_main.base_yaw.uniformDistribution(-30, 30);

    ShootParameters shoot_parameters_cane = shoot_parameters_main;
    //    shoot_parameters_cane.phytomer_parameters.internode.image_texture = "GrapeBark.jpg";
    shoot_parameters_cane.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_cane.phytomer_parameters.internode.radial_subdivisions = 15;
    shoot_parameters_cane.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_cane.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_cane.insertion_angle_tip.uniformDistribution(60, 120);
    shoot_parameters_cane.girth_area_factor = 0.7f;
    // A cordon is a permanent woody arm laid down at training, not a shoot that elongates: it must stop
    // at the length the builder lays out. max_nodes is what stops it, so buildGrapevineVSP() overwrites
    // this with the node count it actually built before growing the plant. The value here only applies
    // if the shoot type is used without that step.
    shoot_parameters_cane.max_nodes = 25;
    shoot_parameters_cane.tortuosity = 0;
    // A cordon is tied down along the fruiting wire when the vine is trained, so it does not curve up
    // the way a free shoot does. The upward curvature this carried arced the arm about 35 degrees above
    // horizontal over its length, lifting the shoot positions with it and smearing the canopy vertically
    // instead of leaving them in a row along the wire.
    shoot_parameters_cane.gravitropic_curvature = 0;
    shoot_parameters_cane.vegetative_bud_break_probability_min = 0.9;
    shoot_parameters_cane.defineChildShootTypes({"grapevine_shoot"}, {1.f});

    ShootParameters shoot_parameters_trunk = shoot_parameters_main;
    shoot_parameters_trunk.phytomer_parameters.internode.image_texture = "GrapeBark.jpg";
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    // Buds on alternate sides from node to node, as on any grapevine shoot. buildGrapevineVSP() relies on
    // this to take the two canes off opposite sides of the head.
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 180;
    // Radius partway up the trunk. buildGrapevineVSP() varies it from vine to vine and shapes the flare at
    // the ground and the swelling of the head around it. 8 cm across is a mature VSP trunk.
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.04;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 25;
    shoot_parameters_trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_trunk.phyllochron_min = 2.5;
    shoot_parameters_trunk.insertion_angle_tip = 90;
    shoot_parameters_trunk.girth_area_factor = 0;
    // Room for the closely spaced nodes buildGrapevineVSP() lays the shape of the trunk out with, at the
    // tallest trunk it accepts.
    shoot_parameters_trunk.max_nodes = 40;
    shoot_parameters_trunk.tortuosity = 0;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.defineChildShootTypes({"grapevine_shoot"}, {1.f});

    // A lateral (secondary) shoot is not a small copy of the shoot that bore it. On VSP vines the
    // laterals carry blades about a third the area of the primary leaves -- 39-40 cm^2 against 121-127
    // -- and contribute 27-35% of the vine's total leaf area (Navarrete 2015, destructive whole-vine
    // defoliation, Willamette Valley Pinot noir). Without a type of its own a lateral inherits the type
    // of its parent (see PlantArchitecture.cpp, sampleChildShootType(): an empty child_shoot_type_labels
    // reuses the parent's label), so every lateral grew full-size primary leaves on a full-length shoot,
    // and since laterals hold over half the leaves on a mature vine that alone put total leaf area
    // several times over what a VSP canopy carries.
    ShootParameters shoot_parameters_lateral = shoot_parameters_main;
    shoot_parameters_lateral.phytomer_parameters.leaf.prototype_scale = 0.079; // ~39 cm^2 blades
    // The petiole is scaled with the blade it carries. Left at the primary leaf's length and radius it was longer than the lateral blade.
    shoot_parameters_lateral.phytomer_parameters.petiole.length = 0.08 * 0.079 / 0.14;
    shoot_parameters_lateral.phytomer_parameters.petiole.radius = 0.0025 * 0.079 / 0.14;
    shoot_parameters_lateral.phytomer_parameters.internode.max_floral_buds_per_petiole = 0; // laterals do not fruit
    // Laterals are tucked into the same curtain as the shoot bearing them, so they emerge close to the
    // parent axis rather than at the wide angles a primary leaves the cordon at. Left at the primary's
    // spread they pushed out across the row and thickened the canopy.
    shoot_parameters_lateral.insertion_angle_tip.uniformDistribution(10, 30);
    shoot_parameters_lateral.base_yaw.uniformDistribution(-15, 15);
    shoot_parameters_lateral.max_nodes = 9;
    shoot_parameters_lateral.vegetative_bud_break_probability_min = 0; // no laterals on laterals
    shoot_parameters_lateral.vegetative_bud_break_probability_max = 0;
    shoot_parameters_lateral.defineChildShootTypes({"grapevine_lateral"}, {1.f});

    shoot_parameters_main.defineChildShootTypes({"grapevine_lateral"}, {1.f});

    // The two cordons of a bilateral vine run in opposite directions along the row, and a shoot's
    // insertion angle is measured from the axis of the cordon bearing it. The same insertion angle
    // therefore swings the shoots of one arm up and the shoots of the other down -- on this model the
    // second arm's shoots grew straight through the ground. Rolling the insertion 180 degrees about the
    // cordon axis mirrors it back, so the two arms are reflections of each other rather than rotations.
    ShootParameters shoot_parameters_main_mirrored = shoot_parameters_main;
    // Yaw, not roll. Yaw turns the petiole rotation axis about the parent axis, and it is that axis the
    // insertion pitch then swings around, so half a turn of yaw sends the pitch the other way. Roll spins
    // the shoot about its own axis once it is already pointing somewhere, which leaves its direction alone.
    shoot_parameters_main_mirrored.base_yaw.uniformDistribution(180.f - 30.f, 180.f + 30.f);
    shoot_parameters_main_mirrored.defineChildShootTypes({"grapevine_lateral"}, {1.f});

    defineShootType("grapevine_trunk", shoot_parameters_trunk);
    defineShootType("grapevine_cane", shoot_parameters_cane);
    defineShootType("grapevine_shoot", shoot_parameters_main);
    defineShootType("grapevine_shoot_mirrored", shoot_parameters_main_mirrored);
    defineShootType("grapevine_lateral", shoot_parameters_lateral);
}

uint PlantArchitecture::buildGrapevineVSP(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize grapevine plant shoots
        initializeGrapevineVSPShoots();
    }

    // Get training system parameters
    auto vine_spacing = getParameterValue(current_build_parameters, "vine_spacing", 2.4f, 0.5f, 5.f, "plant-to-plant spacing in meters");
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 0.8f, 0.05f, 1.f, "height of the fruiting wire in meters");

    // Calculate cane nodes to span to neighboring plant. One node is one bud, and so one shoot position:
    // spur-pruned cordons are laid down at roughly a bud every 8-12 cm, giving the 10-16 shoots per metre
    // of cordon that commercial VSP is thinned to (Kliewer & Dokoozlian 2005; OSU Extension EM 9071). At
    // the 0.15 m this used previously a 2.4 m vine carried only about 5.6 shoots per metre, less than half
    // the commercial density, and the canopy could not close into a wall no matter how large its leaves.
    float cane_internode_length = 0.10f;
    float cane_total_length = vine_spacing / 2.f; // Cane extends from center to next plant
    uint cane_nodes = uint(cane_total_length / cane_internode_length);
    if (cane_nodes < 1)
        cane_nodes = 1;

    // Catch wires. Vertical shoot positioning is the practice of tucking shoots between pairs of wires
    // running above the fruiting wire, which is what makes the canopy a narrow vertical curtain rather
    // than a sprawl; without them the shoots of this model spread roughly 1.7 m across the row where a
    // VSP curtain is 0.3-0.6 m thick, and the same leaf area smeared over that envelope reads as a thin,
    // porous canopy. The wires are represented as attraction points, the same mechanism buildGrapevineWye()
    // uses. They are not drawn -- only their guiding effect on shoot direction is modelled.
    const float catch_wire_half_separation = 0.075f; // pairs sit +/- this across the row, giving a ~0.15 m curtain
    const uint catch_wire_pairs = 4;
    const float catch_wire_lowest = 0.35f; // height of the first wire pair above the cordon
    const float catch_wire_spacing = 0.20f; // vertical spacing between wire pairs
    // A wire is represented by a run of points, and a shoot is only steered by one that falls inside its
    // forward view cone, so the points have to be closer together than the distance that cone looks ahead
    // or a shoot climbing between two wires sees nothing to follow. Roughly one point every 6 cm.
    const uint catch_wire_samples = std::max(uint(2), uint(vine_spacing / 0.06f));

    std::vector<std::vector<vec3>> trellis_points;
    trellis_points.reserve(2 * catch_wire_pairs);
    for (uint wire = 0; wire < catch_wire_pairs; wire++) {
        const float wire_height = trunk_height + catch_wire_lowest + float(wire) * catch_wire_spacing;
        // The cordon runs along y, so the wires do too, spanning the full vine spacing.
        for (const float side: {-catch_wire_half_separation, catch_wire_half_separation}) {
            trellis_points.push_back(linspace(make_vec3(side, -0.5f * vine_spacing, wire_height), make_vec3(side, 0.5f * vine_spacing, wire_height), catch_wire_samples));
        }
    }
    for (auto &wire_points: trellis_points) {
        for (vec3 &point: wire_points) {
            point += base_position;
        }
    }

    uint plantID = addPlantInstance(base_position, 0);

    // Catch wires: guide the shoots into a narrow vertical curtain (see the trellis_points block above).
    setPlantAttractionPoints(plantID, flatten(trellis_points), 70.f, 0.35f, 0.15f);

    // The trunk and the two cordons are trained wood: a grower ties them to the fruiting wire, so their
    // shape is imposed rather than grown. They are therefore prescribed node by node instead of being
    // extrapolated from a base rotation. Deriving them from a pitch angle meant their final shape was the
    // end of a chain of interacting parameters -- base pitch, per-phytomer internode pitch, tortuosity and
    // gravitropic curvature -- and tuning any one of them against the others produced a cordon that swept
    // up out of the horizontal into a "Y" rather than running level along the wire. Prescribing the path
    // states the trained shape directly, and the growth model leaves it alone: prescribed phytomers are
    // built fully elongated and are not re-curved or re-scaled by advanceTime().
    //
    // Every vine is drawn differently, but none of the variation is a free random walk. A trunk is held by
    // its stake and a cane by the wire it is tied to, so each departs from its trained line by a bounded
    // amount and keeps coming back to it -- see BoundedWander.
    const float fruiting_wire_height = base_position.z + trunk_height;

    // ---- Trunk ---- //

    // The head is the knob of old wood at the top of the trunk that the canes are renewed from each winter.
    // Cane pruning keeps it a little below the fruiting wire so that the canes can be bent up and over onto
    // the wire; a head at the wire leaves nowhere for them to go but up.
    const float head_height = trunk_height * (1.f - context_ptr->randu(0.08f, 0.16f));

    // The path and girth of the trunk are shared with the Wye vine; see grapevineTrunkPath().
    std::vector<vec3> trunk_node_positions;
    std::vector<float> trunk_node_radii;
    const uint trunk_nodes = grapevineTrunkPath(context_ptr, base_position, head_height, shoot_types.at("grapevine_trunk").phytomer_parameters.internode.radius_initial.val(), trunk_node_positions, trunk_node_radii);
    uint uID_stem = addShootFromNodePositions(plantID, -1, 0, trunk_node_positions, trunk_node_radii, "grapevine_trunk");

    // ---- Canes ---- //

    // One cane to each side along the row, which runs along y. The two leave the head from adjacent nodes,
    // each from the one whose bud faces its own side: a child shoot is seated on the surface of its parent
    // on the side of the petiole, so a cane attached to the wrong node would start on the far side of the
    // head and run back through it.
    const auto &trunk_phytomers = plant_instances.at(plantID).shoot_tree.at(uID_stem)->phytomers;
    const uint head_phytomer = trunk_nodes - 1; // its tip is the node at head_height, just below the dome
    const bool head_bud_faces_negative_y = trunk_phytomers.at(head_phytomer)->getPetioleAxisVector(0.f, 0).y < 0.f;

    uint uID_cane_L = 0;
    uint uID_cane_R = 0;
    for (const float direction: {-1.f, 1.f}) {
        const uint attachment_phytomer = ((direction < 0.f) == head_bud_faces_negative_y) ? head_phytomer : head_phytomer - 1;
        const vec3 attachment_node = trunk_node_positions.at(attachment_phytomer + 1);

        // The cane leaves the head climbing, and is bent over onto the wire a short way out.
        const float rise = fruiting_wire_height - attachment_node.z;
        const float bend_length = rise * context_ptr->randu(1.8f, 3.f);

        // Along the wire: sag between the ties and a slower drift across the wire as the cane is wrapped
        // around it, each with a tighter wobble on top.
        BoundedWander cane_wander_x;
        cane_wander_x.addComponent(context_ptr->randu(0.008f, 0.018f), context_ptr->randu(0.4f, 0.8f), context_ptr->randu(0.f, 2.f * PI_F));
        cane_wander_x.addComponent(context_ptr->randu(0.003f, 0.006f), context_ptr->randu(0.15f, 0.3f), context_ptr->randu(0.f, 2.f * PI_F));
        BoundedWander cane_wander_z;
        cane_wander_z.addComponent(context_ptr->randu(0.006f, 0.015f), context_ptr->randu(0.5f, 0.9f), context_ptr->randu(0.f, 2.f * PI_F));
        cane_wander_z.addComponent(context_ptr->randu(0.003f, 0.006f), context_ptr->randu(0.18f, 0.3f), context_ptr->randu(0.f, 2.f * PI_F));

        // The last tie is short of the tip, which is left to droop or spring up a little.
        const float free_tip_length = 0.2f;
        const float free_tip_deflection = context_ptr->randu(-0.03f, 0.02f);

        // A cane does not quite reach the vine next door.
        const float cane_length = cane_total_length * context_ptr->randu(0.93f, 1.f);

        // The first few internodes of a cane are markedly shorter than the rest, and all of them vary. They
        // are scaled together afterwards so that the cane comes out at its length whatever was drawn.
        std::vector<float> cane_internode_lengths(cane_nodes);
        float unscaled_cane_length = 0.f;
        for (uint internode = 0; internode < cane_nodes; internode++) {
            const float basal_shortening = std::min(1.f, 0.4f + 0.2f * float(internode));
            cane_internode_lengths.at(internode) = basal_shortening * context_ptr->randu(0.8f, 1.2f);
            unscaled_cane_length += cane_internode_lengths.at(internode);
        }
        for (float &internode_length: cane_internode_lengths) {
            internode_length *= cane_length / unscaled_cane_length;
        }

        // Position of the cane at a given distance along the row from the node it is attached to
        auto cane_position = [&](float row_distance) {
            const float bend_fraction = std::min(row_distance / bend_length, 1.f);
            const float on_wire = bend_fraction * bend_fraction * (3.f - 2.f * bend_fraction); // 0 at the head, 1 once tied to the wire
            const float tip_fraction = std::max(0.f, (row_distance - (cane_length - free_tip_length)) / free_tip_length);
            const float x = (1.f - on_wire) * (attachment_node.x - base_position.x) + on_wire * cane_wander_x.evaluate(row_distance);
            const float z = -rise * (1.f - bend_fraction) * (1.f - bend_fraction) + on_wire * cane_wander_z.evaluate(row_distance) + free_tip_deflection * tip_fraction * tip_fraction;
            return make_vec3(base_position.x + x, attachment_node.y + direction * row_distance, fruiting_wire_height + z);
        };

        std::vector<vec3> cane_node_positions;
        std::vector<float> cane_node_radii;
        cane_node_positions.reserve(cane_nodes + 1);
        cane_node_radii.reserve(cane_nodes + 1);
        const float cane_radius = 0.0055f * context_ptr->randu(0.9f, 1.15f);
        float row_distance = 0.f;
        float path_distance = 0.f;
        for (uint node = 0; node <= cane_nodes; node++) {
            vec3 node_position = cane_position(row_distance);
            if (node > 0) {
                // A grapevine cane zig-zags slightly from node to node.
                const float zigzag = ((node % 2 == 0) ? 1.f : -1.f) * context_ptr->randu(0.001f, 0.004f);
                node_position.x += zigzag;
            }
            cane_node_positions.push_back(node_position);
            // Tapering to the tip, and thickened where it comes off the older wood of the head.
            const float tip_taper = 1.f - 0.4f * float(node) / float(cane_nodes);
            const float base_thickening = 1.f + 0.8f * std::exp(-path_distance / 0.06f);
            cane_node_radii.push_back(cane_radius * tip_taper * base_thickening);

            // Step along the row by whatever advances one internode along the cane, which is less than an
            // internode where the cane is climbing.
            if (node < cane_nodes) {
                const float slope_step = 0.005f;
                const float path_length_per_row_distance = (cane_position(row_distance + slope_step) - cane_position(row_distance)).magnitude() / slope_step;
                row_distance += cane_internode_lengths.at(node) / path_length_per_row_distance;
                path_distance += cane_internode_lengths.at(node);
            }
        }
        // Built as a cane, but grown as one too: the growth type is what decides the type of the shoots
        // its buds produce, and those must be grapevine_shoot.
        const uint uID_cane = addShootFromNodePositions(plantID, int(uID_stem), attachment_phytomer, cane_node_positions, cane_node_radii, "grapevine_cane", "grapevine_cane");
        if (direction < 0.f) {
            uID_cane_L = uID_cane;
        } else {
            uID_cane_R = uID_cane;
        }
    }

    // A cordon is permanent woody structure -- it carries the buds, it does not keep extending -- but the
    // model elongates any shoot whose meristem is alive up to its type's max_nodes, and a cane left free
    // to do that grows on past the end of its allotted span into the neighbouring vine. Killing the apical
    // meristem leaves the lateral buds, and so the shoots, untouched.
    plant_instances.at(plantID).shoot_tree.at(uID_cane_L)->meristem_is_alive = false;
    plant_instances.at(plantID).shoot_tree.at(uID_cane_R)->meristem_is_alive = false;

    // Wake the trained wood and its buds. Two things have to be undone here, both consequences of building
    // this structure from prescribed node positions rather than with appendShoot().
    //
    // First, every bud on a shoot built by addShootFromNodePositions() comes back dead, where the same
    // cordon built by appendShoot() comes back with nearly all of them active: buds are sampled for break
    // as each phytomer is appended, against a node index that is not yet final on the prescribed path, and
    // the apex-weighted probability this shoot type uses then floors every one of them. A cordon whose buds
    // are all dead bears no shoots at all. They are set dormant rather than sampled, which is the right
    // behaviour for trained wood in any case: the grower decides how many shoots a cordon carries.
    //
    // Second, a prescribed shoot is created dormant, so without clearing the flag the cordons would sit
    // until the plant's dormancy-break threshold -- 165 days for this model -- and the shoots they bear
    // would be born after it. That costs those shoots the makeDormant()/breakDormancy() pass at the season
    // boundary, which is where a shoot's own buds are re-sampled against its final node count, and it is
    // what collapsed the vine's lateral shoots from roughly thirty to two.
    for (const uint uID_trained: {uID_stem, uID_cane_L, uID_cane_R}) {
        const auto &trained_shoot = plant_instances.at(plantID).shoot_tree.at(uID_trained);
        trained_shoot->isdormant = false;
        // Only the cordons bear shoots. A bud that pushes on the trunk is a sucker, and a grower strips
        // those off; left active they added a third again as many shoots as the cordons carried, growing
        // out of the trunk below the fruiting wire where a VSP canopy has no foliage at all.
        const bool bears_shoots = (uID_trained != uID_stem);
        for (auto &phytomer: trained_shoot->phytomers) {
            phytomer->isdormant = false;
            for (auto &petiole: phytomer->axillary_vegetative_buds) {
                for (auto &vegetative_bud: petiole) {
                    // Active, not dormant: with the shoot's dormancy flag cleared above, advanceTime()
                    // never calls breakDormancy() on it, so a bud left dormant here would never be woken.
                    phytomer->setVegetativeBudState(bears_shoots ? BUD_ACTIVE : BUD_DEAD, vegetative_bud);
                    // A shoot is inserted toward the side of the cane its petiole is on, so where the petiole
                    // hangs below the cane the insertion angle sends the shoot downward and it has to be
                    // turned half way round -- see the note on grapevine_shoot_mirrored where the two types
                    // are defined. Which way a cane's petioles face is inherited from the trunk node it
                    // comes off and the direction it runs in, so it is read from the petiole itself rather
                    // than assumed from which of the two canes this is.
                    const bool petiole_hangs_below_cane = phytomer->getPetioleAxisVector(0.f, 0).z < 0.f;
                    vegetative_bud.shoot_type_label = petiole_hangs_below_cane ? "grapevine_shoot_mirrored" : "grapevine_shoot";
                }
            }
        }
    }

    // The trunk is trained wood too: it is taken up to the fruiting wire and stopped there, not left to
    // keep extending to its type's node cap. Without this it grew on past the cordons, carrying the head
    // of the vine a metre above the wire they are tied to.
    plant_instances.at(plantID).shoot_tree.at(uID_stem)->meristem_is_alive = false;

    //    makePlantDormant(plantID);

    removeShootLeaves(plantID, uID_stem);
    removeShootLeaves(plantID, uID_cane_L);
    removeShootLeaves(plantID, uID_cane_R);

    setPlantPhenologicalThresholds(plantID, 165, -1, -1, 45, 45, 200);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeGrapevineWyeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "GrapeLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.4, 0.4);
    leaf_prototype.lateral_curvature = 0;
    leaf_prototype.wave_period = 0.3f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.subdivisions = 5;
    leaf_prototype.unique_prototypes = 10;
    leaf_prototype.leaf_offset = make_vec3(-0.3, 0, 0);

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_grapevine(context_ptr->getRandomGenerator());

    phytomer_parameters_grapevine.internode.pitch = 20;
    phytomer_parameters_grapevine.internode.phyllotactic_angle.uniformDistribution(160, 200);
    phytomer_parameters_grapevine.internode.radius_initial = 0.003;
    phytomer_parameters_grapevine.internode.color = make_RGBcolor(0.23, 0.13, 0.062);
    phytomer_parameters_grapevine.internode.length_segments = 1;
    phytomer_parameters_grapevine.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_grapevine.internode.max_vegetative_buds_per_petiole = 1;

    phytomer_parameters_grapevine.petiole.petioles_per_internode = 1;
    // Grapevine petioles are green, often flushed red; matches the VSP model.
    phytomer_parameters_grapevine.petiole.color = make_RGBcolor(0.35, 0.38, 0.15);
    phytomer_parameters_grapevine.petiole.pitch.uniformDistribution(45, 70);
    phytomer_parameters_grapevine.petiole.radius = 0.0025;
    phytomer_parameters_grapevine.petiole.length = 0.08;
    phytomer_parameters_grapevine.petiole.taper = 0;
    phytomer_parameters_grapevine.petiole.curvature = 0;
    phytomer_parameters_grapevine.petiole.length_segments = 1;

    phytomer_parameters_grapevine.leaf.leaves_per_petiole = 1;
    phytomer_parameters_grapevine.leaf.pitch.uniformDistribution(-110, -80);
    phytomer_parameters_grapevine.leaf.yaw.uniformDistribution(-20, 20);
    phytomer_parameters_grapevine.leaf.roll.uniformDistribution(-5, 5);
    // Sized to a mean primary blade area of ~100 cm^2. A divided canopy carries more shoots per vine than
    // a single curtain, and smaller leaves on them: on the same Chenin blanc vines Kliewer & Dokoozlian
    // (2005) measured 74 cm^2 on GDC and 102 cm^2 on lyre against 123 cm^2 on a single canopy, and their
    // divided-canopy Cabernet Sauvignon shoots carried 1810-2380 cm^2 of primary leaf over 21-25 nodes.
    // Blade area scales as the square of this parameter (0.14 gives the ~124 cm^2 of the VSP vine).
    phytomer_parameters_grapevine.leaf.prototype_scale = 0.126;
    phytomer_parameters_grapevine.leaf.prototype = leaf_prototype;

    phytomer_parameters_grapevine.peduncle.length = 0.04;
    phytomer_parameters_grapevine.peduncle.radius = 0.005;
    phytomer_parameters_grapevine.peduncle.color = make_RGBcolor(0.13, 0.125, 0.03);
    phytomer_parameters_grapevine.peduncle.pitch.uniformDistribution(80, 100);
    // A grapevine cluster is borne on the opposite side of the node from the leaf, not in the leaf axil.
    phytomer_parameters_grapevine.peduncle.yaw = 180;

    phytomer_parameters_grapevine.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_grapevine.inflorescence.pitch = 0;
    //    phytomer_parameters_grapevine.inflorescence.flower_prototype_function = GrapevineFlowerPrototype;
    phytomer_parameters_grapevine.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_function = GrapevineFruitPrototype;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_gravity_factor_fraction = 0.9;

    phytomer_parameters_grapevine.phytomer_creation_function = GrapevinePhytomerCreationFunction;
    // Pulls the fruit-zone leaves at fruit set; see GrapevinePhytomerCallbackFunction().
    phytomer_parameters_grapevine.phytomer_callback_function = GrapevinePhytomerCallbackFunction;

    // ---- Shoot Parameters ---- //

    // The fruiting shoot is that of the VSP vine, where the values are discussed and referenced; see
    // initializeGrapevineVSPShoots(). What differs is what the trellis does to it. The values specific to a
    // divided canopy are from the quadrilateral-cordon, GDC and lyre vines of Kliewer & Dokoozlian (2005, Am.
    // J. Enol. Vitic. 56:170-181).
    ShootParameters shoot_parameters_main(context_ptr->getRandomGenerator());
    shoot_parameters_main.phytomer_parameters = phytomer_parameters_grapevine;
    // About one bud in four pushes a summer lateral, flat along the shoot. Laterals are confined to the
    // basal nodes by GrapevinePhytomerCreationFunction(). On divided canopies they carry 17-29% of the leaf
    // area of a shoot, a little less than on a single curtain.
    shoot_parameters_main.vegetative_bud_break_probability_min = 0.25;
    shoot_parameters_main.vegetative_bud_break_probability_max = 0.25;
    shoot_parameters_main.vegetative_bud_break_probability_decay_rate = 0.;
    shoot_parameters_main.vegetative_bud_break_time = 30;
    shoot_parameters_main.phyllochron_min.uniformDistribution(1.75, 2.25);
    shoot_parameters_main.elongation_rate_max = 0.15;
    shoot_parameters_main.girth_area_factor = 1.f;
    shoot_parameters_main.gravitropic_curvature = 300;
    shoot_parameters_main.tortuosity = 226;
    // Internodes of 4.9-5.3 cm were measured on the shoots of divided canopies.
    shoot_parameters_main.internode_length_max.uniformDistribution(0.048, 0.058);
    shoot_parameters_main.internode_length_decay_rate = 0;
    // Measured from the axis of the cordon, toward its tip, so 90 degrees leaves the cordon square on. At
    // the 45 degrees this had before, every shoot set off leaning toward the end of its cordon; the four
    // cordons run away from the trunk, so all of the shoots leaned away from the middle of the vine and
    // left a gap in the canopy above it.
    shoot_parameters_main.insertion_angle_tip.uniformDistribution(55, 125);
    shoot_parameters_main.insertion_angle_decay_rate = 0;
    shoot_parameters_main.flowers_require_dormancy = false;
    shoot_parameters_main.growth_requires_dormancy = false;
    shoot_parameters_main.determinate_shoot_growth = false;
    shoot_parameters_main.max_terminal_floral_buds = 0;
    shoot_parameters_main.flower_bud_break_probability = 0.5;
    // About 1.5 clusters per shoot on average, within the one to three a fruitful shoot carries; see initializeGrapevineVSPShoots().
    shoot_parameters_main.fruit_set_probability = 1.0;
    // The shoots of a divided canopy are not hedged to a narrow wall as VSP shoots are, and were measured
    // at 21-25 nodes and 103-136 cm.
    shoot_parameters_main.max_nodes.uniformDistribution(21, 25);
    shoot_parameters_main.base_roll.uniformDistribution(90 - 15, 90 + 15);
    // Yaw turns the insertion about the cordon. The shoots are not tucked between paired wires as on a VSP
    // trellis, so they fan more widely over the gable.
    shoot_parameters_main.base_yaw.uniformDistribution(-40, 40);

    // A cordon is a permanent woody arm tied down along its wire. buildGrapevineWye() lays it out node by
    // node, one node to a shoot position, stops it at the length it was laid out to and sets every one of
    // its buds, so nothing here bends it or decides how many shoots it carries.
    ShootParameters shoot_parameters_cordon = shoot_parameters_main;
    shoot_parameters_cordon.phytomer_parameters.internode.image_texture = "GrapeBark.jpg";
    shoot_parameters_cordon.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_cordon.phytomer_parameters.internode.radial_subdivisions = 15;
    shoot_parameters_cordon.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    // Every bud on the same side of the cordon, which buildGrapevineWye() arranges to be the upper one
    shoot_parameters_cordon.phytomer_parameters.internode.phyllotactic_angle = 0;
    // The radii the cordon is laid out with are those of mature wood, and are kept
    shoot_parameters_cordon.girth_area_factor = 0;
    // Room for the nodes of a cordon at the widest vine spacing buildGrapevineWye() accepts
    shoot_parameters_cordon.max_nodes = 40;
    shoot_parameters_cordon.tortuosity = 0;
    shoot_parameters_cordon.gravitropic_curvature = 0;
    shoot_parameters_cordon.base_yaw = 0;
    shoot_parameters_cordon.defineChildShootTypes({"grapevine_shoot"}, {1.f});

    // The trunk, and the two arms of the Y that leave its head
    ShootParameters shoot_parameters_trunk = shoot_parameters_main;
    shoot_parameters_trunk.phytomer_parameters.internode.image_texture = "GrapeBark.jpg";
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    // No turn from node to node. The side of a cordon its buds are on is handed down from the trunk
    // through the arm, and has to come out the same whatever number of nodes each was laid out with.
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    // Radius partway up the trunk; buildGrapevineWye() varies it from vine to vine and shapes the flare at
    // the ground and the swelling of the head around it. Trunks of mature vines are commonly described as
    // 7-9 cm across.
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.04;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 25;
    shoot_parameters_trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_trunk.phyllochron_min = 2.5;
    shoot_parameters_trunk.insertion_angle_tip = 90;
    shoot_parameters_trunk.girth_area_factor = 0;
    // Room for the closely spaced nodes buildGrapevineWye() lays the shape of the trunk out with, at the
    // tallest trunk it accepts.
    shoot_parameters_trunk.max_nodes = 60;
    shoot_parameters_trunk.tortuosity = 0;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.defineChildShootTypes({"grapevine_shoot"}, {1.f});

    // A summer lateral carries blades about a third the area of the primary leaves, and does not fruit;
    // see initializeGrapevineVSPShoots(). Without a type of its own a lateral takes the type of the shoot
    // bearing it, and grows as a second full-length fruiting shoot.
    ShootParameters shoot_parameters_lateral = shoot_parameters_main;
    shoot_parameters_lateral.phytomer_parameters.leaf.prototype_scale = 0.079; // ~39 cm^2 blades
    shoot_parameters_lateral.phytomer_parameters.petiole.length = 0.08 * 0.079 / 0.14;
    shoot_parameters_lateral.phytomer_parameters.petiole.radius = 0.0025 * 0.079 / 0.14;
    shoot_parameters_lateral.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_lateral.insertion_angle_tip.uniformDistribution(10, 30);
    shoot_parameters_lateral.base_yaw.uniformDistribution(-15, 15);
    shoot_parameters_lateral.max_nodes = 9;
    shoot_parameters_lateral.vegetative_bud_break_probability_min = 0; // no laterals on laterals
    shoot_parameters_lateral.vegetative_bud_break_probability_max = 0;
    shoot_parameters_lateral.defineChildShootTypes({"grapevine_lateral"}, {1.f});

    shoot_parameters_main.defineChildShootTypes({"grapevine_lateral"}, {1.f});

    // The fruiting shoot of a bud on the underside of its cordon: the same shoot, turned half way round
    // the cordon so that it sets off upward. See the note on this type in initializeGrapevineVSPShoots().
    ShootParameters shoot_parameters_main_mirrored = shoot_parameters_main;
    shoot_parameters_main_mirrored.base_yaw.uniformDistribution(180.f - 40.f, 180.f + 40.f);
    shoot_parameters_main_mirrored.defineChildShootTypes({"grapevine_lateral"}, {1.f});

    defineShootType("grapevine_trunk", shoot_parameters_trunk);
    defineShootType("grapevine_cordon", shoot_parameters_cordon);
    defineShootType("grapevine_shoot", shoot_parameters_main);
    defineShootType("grapevine_shoot_mirrored", shoot_parameters_main_mirrored);
    defineShootType("grapevine_lateral", shoot_parameters_lateral);
}

uint PlantArchitecture::buildGrapevineWye(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize grapevine plant shoots
        initializeGrapevineWyeShoots();
    }

    // The vine modelled is a quadrilateral-cordon wine grape on a Wye (open gable) trellis. The row runs
    // along y, as it does for the VSP vine. A trunk comes up to a head below the cordon wires and divides
    // into two arms, one to each side of the row -- the Y. Each arm reaches a cordon wire and carries two
    // cordons along it, one each way, so the vine has four. Above each cordon wire the gable wires step up
    // and outward, and the shoots are carried up over them into the two curtains of a divided canopy.
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 1.4f, 0.5f, 2.f, "height of the cordon wires in meters");
    auto cordon_spacing = getParameterValue(current_build_parameters, "cordon_spacing", 0.6f, 0.2f, 2.f, "distance between the two cordon wires across the row in meters");
    auto vine_spacing = getParameterValue(current_build_parameters, "vine_spacing", 1.8f, 0.5f, 5.f, "plant-to-plant spacing in meters");
    auto catch_wire_height = getParameterValue(current_build_parameters, "catch_wire_height", trunk_height + 0.6f, 0.5f, 4.f, "height of the top gable wires in meters");
    if (catch_wire_height <= trunk_height) {
        helios_runtime_error("ERROR (PlantArchitecture::buildGrapevineWye): build parameter 'catch_wire_height' (" + std::to_string(catch_wire_height) + " m) must be greater than 'trunk_height' (" + std::to_string(trunk_height) +
                             " m). The gable wires that carry the canopy rise from the cordon wires, whose height is 'trunk_height'.");
    }

    // One node of a cordon is one bud, and so one shoot position. Spur-pruned cordons carry a two-bud spur
    // about every 15 cm, and the divided canopies of Kliewer & Dokoozlian (2005) were measured at 12-13
    // shoots per metre of canopy and 41-50 shoots per vine.
    const float cordon_internode_length = 0.08f;
    const float cordon_total_length = 0.5f * vine_spacing; // each cordon covers half of the vine's length of row
    const uint cordon_nodes = std::max(uint(1), uint(cordon_total_length / cordon_internode_length));

    // Trellis wires, represented as attraction points; they are not drawn. The lowest pair are the cordon
    // wires. Above each, the gable wires step up and outward to the top wire at catch_wire_height, and
    // steer the shoots up the two faces of the gable rather than leaving them to sprawl.
    const float gable_rise = catch_wire_height - trunk_height;
    const float gable_spread_per_rise = 0.6f; // each face leans outward about 30 degrees from vertical
    const uint gable_wire_levels = std::max(uint(1), uint(std::round(gable_rise / 0.2f)));
    // A shoot is only steered by a point inside its forward view cone, so the points along a wire have to
    // be closer together than the distance that cone looks ahead. Roughly one point every 6 cm.
    const uint wire_samples = std::max(uint(2), uint(vine_spacing / 0.06f));

    std::vector<std::vector<vec3>> trellis_points;
    trellis_points.reserve(2 * (gable_wire_levels + 1));
    for (uint level = 0; level <= gable_wire_levels; level++) {
        const float height_above_cordon = gable_rise * float(level) / float(gable_wire_levels);
        const float wire_offset = 0.5f * cordon_spacing + gable_spread_per_rise * height_above_cordon;
        for (const float side: {-1.f, 1.f}) {
            trellis_points.push_back(linspace(make_vec3(side * wire_offset, -0.5f * vine_spacing, trunk_height + height_above_cordon), make_vec3(side * wire_offset, 0.5f * vine_spacing, trunk_height + height_above_cordon), wire_samples));
        }
    }
    for (auto &wire_points: trellis_points) {
        for (vec3 &point: wire_points) {
            point += base_position;
        }
    }

    uint plantID = addPlantInstance(base_position, 0);

    setPlantAttractionPoints(plantID, flatten(trellis_points), 70.f, 0.35f, 0.15f);

    // The trunk, the arms and the cordons are trained wood, whose shape is imposed by the trellis rather
    // than grown. They are prescribed node by node, as for the VSP vine: see buildGrapevineVSP().
    const float cordon_wire_height = base_position.z + trunk_height;

    // ---- Trunk ---- //

    // The arms rise from the head to the cordon wires at about 45 degrees, so the head sits about half the
    // cordon spacing below them. On a short trunk under widely spaced wires it is kept in the upper part of
    // the trunk, and the arms leave it at a shallower angle.
    const float arm_rise = std::min(0.5f * cordon_spacing, 0.35f * trunk_height) * context_ptr->randu(0.9f, 1.1f);
    const float head_height = trunk_height - arm_rise;

    std::vector<vec3> trunk_node_positions;
    std::vector<float> trunk_node_radii;
    const uint trunk_nodes = grapevineTrunkPath(context_ptr, base_position, head_height, shoot_types.at("grapevine_trunk").phytomer_parameters.internode.radius_initial.val(), trunk_node_positions, trunk_node_radii);

    // The arms are as thick as the head they leave, and fork out of the inside of it. A child shoot is
    // seated on the surface of its parent at the node it is attached to, which for wood this thick leaves
    // an open-ended tube standing off the trunk with a gap showing between the two. The trunk is therefore
    // given one last node of negligible radius, doubled back down its own axis from the top of the dome to
    // just below the head. It is hidden inside the head, and an arm attached to it starts on the axis of
    // the trunk with its base buried.
    const float head_radius = trunk_node_radii.at(trunk_nodes);
    trunk_node_positions.push_back(trunk_node_positions.at(trunk_nodes) - make_vec3(0, 0, 0.4f * head_radius));
    trunk_node_radii.push_back(0.002f);
    const uint uID_stem = addShootFromNodePositions(plantID, -1, 0, trunk_node_positions, trunk_node_radii, "grapevine_trunk");

    // ---- Arms and cordons ---- //

    std::vector<uint> uID_arms;
    std::vector<uint> uID_cordons;
    for (const float side: {-1.f, 1.f}) {

        // -- Arm: from the head, out across the row and up to the cordon wire on this side -- //

        // Both arms fork from the node hidden inside the head; see the note where the trunk is built
        const uint arm_attachment_phytomer = uint(trunk_node_positions.size()) - 2;
        const vec3 arm_base = trunk_node_positions.back();
        const vec3 arm_top = make_vec3(base_position.x + side * 0.5f * cordon_spacing, arm_base.y + context_ptr->randu(-0.02f, 0.02f), cordon_wire_height);
        const float arm_length = (arm_top - arm_base).magnitude();
        const uint arm_nodes = std::max(uint(4), uint(std::ceil(arm_length / 0.04f)));

        // An arm is short and stiff, so it only bows a little out of the straight line between its ends
        const float arm_bow = context_ptr->randu(0.05f, 0.15f) * arm_length;
        // Arms of mature vines are roughly half the girth of the trunk they leave (no measured source)
        const float arm_radius = 0.55f * head_radius * context_ptr->randu(0.9f, 1.1f);

        std::vector<vec3> arm_node_positions;
        std::vector<float> arm_node_radii;
        arm_node_positions.reserve(arm_nodes + 4);
        arm_node_radii.reserve(arm_nodes + 4);
        for (uint node = 0; node <= arm_nodes; node++) {
            const float along = float(node) / float(arm_nodes);
            // Bowed downward and outward, so that the arm leaves the head nearer the horizontal and comes
            // up to its wire nearer the vertical
            vec3 node_position = arm_base + (arm_top - arm_base) * along;
            node_position.z -= arm_bow * std::sin(PI_F * along);
            node_position.x += side * 0.5f * arm_bow * std::sin(PI_F * along);
            arm_node_positions.push_back(node_position);
            arm_node_radii.push_back(arm_radius * (1.f + 0.5f * std::exp(-along * arm_length / 0.05f) - 0.2f * along));
        }
        appendWoodDome(arm_node_positions, arm_node_radii);
        // The cordons fork out of the inside of the top of the arm, from a hidden node like that of the trunk
        vec3 arm_top_direction = arm_node_positions.at(arm_nodes) - arm_node_positions.at(arm_nodes - 1);
        arm_top_direction.normalize();
        arm_node_positions.push_back(arm_node_positions.at(arm_nodes) - arm_top_direction * (0.4f * arm_node_radii.at(arm_nodes)));
        arm_node_radii.push_back(0.002f);

        const uint uID_arm = addShootFromNodePositions(plantID, int(uID_stem), arm_attachment_phytomer, arm_node_positions, arm_node_radii, "grapevine_trunk");
        uID_arms.push_back(uID_arm);

        // -- Cordons: one each way along the wire from the top of the arm -- //

        for (const float direction: {-1.f, 1.f}) {
            const uint attachment_phytomer = uint(arm_node_positions.size()) - 2;
            // Read back from the arm as built, which is moved to where it is seated on the trunk
            const vec3 attachment_node = plant_instances.at(plantID).shoot_tree.at(uID_arm)->shoot_internode_vertices.at(attachment_phytomer).back();

            // The cordon leaves the arm a little off the wire, and is brought onto it a short way out
            const float rise = cordon_wire_height - attachment_node.z;
            const float inset = side * (base_position.x + side * 0.5f * cordon_spacing - attachment_node.x);
            const float bend_length = std::min(0.25f, std::max(0.08f, std::max(std::fabs(rise), std::fabs(inset)) * context_ptr->randu(2.f, 3.f)));

            // Along the wire: sag between the ties and a slower drift across the wire, each with a tighter wobble on top
            BoundedWander cordon_wander_x;
            cordon_wander_x.addComponent(context_ptr->randu(0.008f, 0.018f), context_ptr->randu(0.4f, 0.8f), context_ptr->randu(0.f, 2.f * PI_F));
            cordon_wander_x.addComponent(context_ptr->randu(0.003f, 0.006f), context_ptr->randu(0.15f, 0.3f), context_ptr->randu(0.f, 2.f * PI_F));
            BoundedWander cordon_wander_z;
            cordon_wander_z.addComponent(context_ptr->randu(0.006f, 0.015f), context_ptr->randu(0.5f, 0.9f), context_ptr->randu(0.f, 2.f * PI_F));
            cordon_wander_z.addComponent(context_ptr->randu(0.003f, 0.006f), context_ptr->randu(0.18f, 0.3f), context_ptr->randu(0.f, 2.f * PI_F));

            // A cordon does not quite reach the vine next door. The arm may sit a few centimetres along the
            // row from where the vine is planted, and the cordon is cut to end where its half of the row does.
            const float cordon_reach = cordon_total_length * context_ptr->randu(0.93f, 1.f) - direction * (attachment_node.y - base_position.y);
            const float cordon_length = std::max(cordon_internode_length, cordon_reach);

            // Spur positions are not evenly spaced, and are scaled together so the cordon comes out at its length
            std::vector<float> cordon_internode_lengths(cordon_nodes);
            float unscaled_cordon_length = 0.f;
            for (uint internode = 0; internode < cordon_nodes; internode++) {
                cordon_internode_lengths.at(internode) = context_ptr->randu(0.8f, 1.2f);
                unscaled_cordon_length += cordon_internode_lengths.at(internode);
            }
            for (float &internode_length: cordon_internode_lengths) {
                internode_length *= cordon_length / unscaled_cordon_length;
            }

            // Position of the cordon at a given distance along the row from the node it is attached to
            auto cordon_position = [&](float row_distance) {
                const float bend_fraction = std::min(row_distance / bend_length, 1.f);
                const float on_wire = bend_fraction * bend_fraction * (3.f - 2.f * bend_fraction); // 0 at the arm, 1 once tied to the wire
                const float x = -side * inset * (1.f - on_wire) + on_wire * cordon_wander_x.evaluate(row_distance);
                const float z = -rise * (1.f - on_wire) + on_wire * cordon_wander_z.evaluate(row_distance);
                return make_vec3(base_position.x + side * 0.5f * cordon_spacing + x, attachment_node.y + direction * row_distance, cordon_wire_height + z);
            };

            std::vector<vec3> cordon_node_positions;
            std::vector<float> cordon_node_radii;
            cordon_node_positions.reserve(cordon_nodes + 1);
            cordon_node_radii.reserve(cordon_nodes + 1);
            // An arm divides into its two cordons, so each carries about half its cross-section, which is
            // 0.7 of its radius. A cordon is permanent wood as old as the arm, and tapers only slightly to
            // its tip; where it leaves the arm it flares out to nearly the girth of the arm. (Cordons of
            // mature vines are commonly described as 3-5 cm across; no measured source.)
            const float arm_top_radius = arm_node_radii.at(arm_nodes);
            const float cordon_radius = 0.75f * arm_top_radius * context_ptr->randu(0.92f, 1.08f);
            const float junction_flare = std::max(0.f, 0.95f * arm_top_radius / cordon_radius - 1.f);
            float row_distance = 0.f;
            float path_distance = 0.f;
            for (uint node = 0; node <= cordon_nodes; node++) {
                cordon_node_positions.push_back(cordon_position(row_distance));
                const float tip_taper = 1.f - 0.2f * float(node) / float(cordon_nodes);
                cordon_node_radii.push_back(cordon_radius * tip_taper * (1.f + junction_flare * std::exp(-path_distance / 0.05f)));

                // Step along the row by whatever advances one internode along the cordon, which is less
                // than an internode where the cordon is still coming onto the wire.
                if (node < cordon_nodes) {
                    const float slope_step = 0.005f;
                    const float path_length_per_row_distance = (cordon_position(row_distance + slope_step) - cordon_position(row_distance)).magnitude() / slope_step;
                    row_distance += cordon_internode_lengths.at(node) / path_length_per_row_distance;
                    path_distance += cordon_internode_lengths.at(node);
                }
            }
            uID_cordons.push_back(addShootFromNodePositions(plantID, int(uID_arm), attachment_phytomer, cordon_node_positions, cordon_node_radii, "grapevine_cordon"));
        }
    }

    // Wake the trained wood, set its buds and stop it extending. A prescribed shoot is created dormant and
    // with its buds dead, and one left with a live meristem grows on past the length it was laid out to;
    // see the notes on each of these in buildGrapevineVSP().
    std::vector<uint> uID_trained = {uID_stem};
    uID_trained.insert(uID_trained.end(), uID_arms.begin(), uID_arms.end());
    uID_trained.insert(uID_trained.end(), uID_cordons.begin(), uID_cordons.end());
    for (const uint uID: uID_trained) {
        const auto &trained_shoot = plant_instances.at(plantID).shoot_tree.at(uID);
        trained_shoot->isdormant = false;
        trained_shoot->meristem_is_alive = false;
        // Only the cordons bear shoots. A bud that pushes on the trunk or an arm is a sucker, and is stripped off.
        const bool bears_shoots = trained_shoot->shoot_type_label == "grapevine_cordon";
        for (auto &phytomer: trained_shoot->phytomers) {
            phytomer->isdormant = false;
            for (auto &petiole: phytomer->axillary_vegetative_buds) {
                for (auto &vegetative_bud: petiole) {
                    phytomer->setVegetativeBudState(bears_shoots ? BUD_ACTIVE : BUD_DEAD, vegetative_bud);
                    // A shoot is inserted toward the side of the cordon its bud is on, so a bud on the
                    // underside is given the type that is turned half way round to set off upward.
                    const bool bud_is_below_cordon = phytomer->getPetioleAxisVector(0.f, 0).z < 0.f;
                    vegetative_bud.shoot_type_label = bud_is_below_cordon ? "grapevine_shoot_mirrored" : "grapevine_shoot";
                }
            }
        }
        removeShootLeaves(plantID, uID);
    }

    setPlantPhenologicalThresholds(plantID, 165, -1, -1, 45, 45, 200);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeGroundCherryWeedShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "GroundCherryLeaf.png";
    leaf_prototype.leaf_aspect_ratio.uniformDistribution(0.3, 0.5);
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature = 0.1f;
    ;
    leaf_prototype.lateral_curvature = -0.3f;
    leaf_prototype.wave_period = 0.35f;
    leaf_prototype.wave_amplitude = 0.08f;
    leaf_prototype.subdivisions = 6;
    leaf_prototype.unique_prototypes = 5;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 5;
    phytomer_parameters.internode.phyllotactic_angle = 137.5;
    phytomer_parameters.internode.radius_initial = 0.0005;
    phytomer_parameters.internode.color = make_RGBcolor(0.266, 0.3375, 0.0700);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45, 60);
    phytomer_parameters.petiole.radius = 0.0005;
    phytomer_parameters.petiole.length = 0.025;
    phytomer_parameters.petiole.taper = 0.15;
    phytomer_parameters.petiole.curvature.uniformDistribution(-150, -50);
    phytomer_parameters.petiole.color = phytomer_parameters.internode.color;
    phytomer_parameters.petiole.length_segments = 2;

    phytomer_parameters.leaf.leaves_per_petiole = 1;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.06, 0.08);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.01;
    phytomer_parameters.peduncle.radius = 0.001;
    phytomer_parameters.peduncle.pitch = 20;
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -700;
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 2;
    phytomer_parameters.peduncle.radial_subdivisions = 6;

    phytomer_parameters.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters.inflorescence.pitch = 0;
    phytomer_parameters.inflorescence.roll.uniformDistribution(-30, 30);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.01;
    phytomer_parameters.inflorescence.flower_prototype_function = BindweedFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.06;
    // phytomer_parameters.inflorescence.fruit_prototype_function = GroundCherryFruitPrototype;
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction = 0.75;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    // shoot_parameters.phytomer_parameters.phytomer_creation_function = TomatoPhytomerCreationFunction;

    shoot_parameters.max_nodes = 26;
    shoot_parameters.insertion_angle_tip = 50;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.015;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature = 700;
    shoot_parameters.tortuosity = 200;

    shoot_parameters.phyllochron_min = 1;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 2.f;
    shoot_parameters.vegetative_bud_break_time = 10;
    shoot_parameters.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = -0.5;
    shoot_parameters.flower_bud_break_probability = 0.25;
    shoot_parameters.fruit_set_probability = 0.5;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = false;

    shoot_parameters.defineChildShootTypes({"mainstem"}, {1.0});

    defineShootType("mainstem", shoot_parameters);
}

uint PlantArchitecture::buildGroundCherryWeedPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize ground cherry plant shoots
        initializeGroundCherryWeedShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.0025, 0.018, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 20, -1, 20, 30, 1000);

    plant_instances.at(plantID).max_age = 80;

    return plantID;
}

void PlantArchitecture::initializeMaizeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.15f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.07, -0.04);
    leaf_prototype.longitudinal_curvature_exponent = 3.f;
    leaf_prototype.flexibility_taper = 150.f;
    leaf_prototype.flexibility_aging = 11.f;
    leaf_prototype.flexibility_aging_max = 6.f;
    leaf_prototype.lateral_curvature = -0.3f;
    leaf_prototype.petiole_roll = 0.f;
    leaf_prototype.wave_period = 0.1f;
    leaf_prototype.wave_amplitude = 0.06f;
    leaf_prototype.flexibility.uniformDistribution(1.5, 2.3);
    leaf_prototype.subdivisions = 50;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_maize(context_ptr->getRandomGenerator());

    phytomer_parameters_maize.internode.pitch = 0;
    phytomer_parameters_maize.internode.phyllotactic_angle.uniformDistribution(170, 190);
    phytomer_parameters_maize.internode.radius_initial = 0.0075;
    phytomer_parameters_maize.internode.color = make_RGBcolor(0.126, 0.182, 0.084);
    phytomer_parameters_maize.internode.length_segments = 2;
    phytomer_parameters_maize.internode.radial_subdivisions = 10;
    phytomer_parameters_maize.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_maize.internode.max_vegetative_buds_per_petiole = 0;

    phytomer_parameters_maize.petiole.petioles_per_internode = 1;
    phytomer_parameters_maize.petiole.pitch.uniformDistribution(-30, -16);
    phytomer_parameters_maize.petiole.radius = 0.0;
    phytomer_parameters_maize.petiole.length = 0.01;
    phytomer_parameters_maize.petiole.taper = 0;
    phytomer_parameters_maize.petiole.curvature = 0;
    phytomer_parameters_maize.petiole.length_segments = 1;

    phytomer_parameters_maize.leaf.leaves_per_petiole = 1;
    phytomer_parameters_maize.leaf.pitch = 0;
    phytomer_parameters_maize.leaf.yaw = 0;
    phytomer_parameters_maize.leaf.roll = 0;
    phytomer_parameters_maize.leaf.prototype_scale = 0.9;
    phytomer_parameters_maize.leaf.prototype = leaf_prototype;

    phytomer_parameters_maize.peduncle.length = 0.14f;
    phytomer_parameters_maize.peduncle.radius = 0.004;
    phytomer_parameters_maize.peduncle.curvature = 0;
    phytomer_parameters_maize.peduncle.color = phytomer_parameters_maize.internode.color;
    phytomer_parameters_maize.peduncle.radial_subdivisions = 6;
    phytomer_parameters_maize.peduncle.length_segments = 2;

    // The tassel is assembled from one copy of MaizeTassel.obj per primary branch, distributed along the terminal peduncle. Modern commercial hybrids carry 10-12 primary branches: Gage et al. (2018,
    // Genetics 210:1125) measured 12.0 in the 1930s BSSSC0 population against about 10.2 in ex-PVP lines from the 1980s-90s, the reduction in branch number being one of the clearer architectural signatures
    // of hybrid breeding. Eleven sits in the middle of that.
    phytomer_parameters_maize.inflorescence.flowers_per_peduncle = 11;
    phytomer_parameters_maize.inflorescence.pitch.uniformDistribution(0, 30);
    phytomer_parameters_maize.inflorescence.roll = 0;

    // Spreads the branches over the upper 90% of the peduncle, which is the branch zone. The bound that matters is not clampOffset() -- its denominator is derived for compound leaves, where the index is
    // divided by petioles_per_internode, and is roughly twice too loose for an inflorescence, where that divisor is 1. The real constraint is that the lowest branch attaches at
    // frac = 1 - (flowers_per_peduncle - 1.5) * flower_offset, which must stay positive or interpolateTube() trips its assert in a debug build and silently returns the peduncle base in a release one. At
    // eleven branches that caps the offset at 0.105; 0.095 leaves the lowest branch at frac 0.098.
    phytomer_parameters_maize.inflorescence.flower_offset = 0.095;

    // Primary branch length near 20 cm, the modal value over 180 three-dimensionally scanned tassels (Zhang et al. 2023, Plant Methods 19:76). The prototype is 1.027 units along its own axis, so the scale
    // is the target length divided by that.
    phytomer_parameters_maize.inflorescence.fruit_prototype_scale = 0.195;
    phytomer_parameters_maize.inflorescence.fruit_prototype_function = MaizeTasselPrototype;

    // The tassel finishes elongating at VT and does not grow afterward: "the tassel is at maximum size" at that stage (Abendroth et al. 2011, Iowa State PMR 1009), and VT "begins approximately 2-3 days
    // before silk emergence" (Ritchie, Hanway & Benson, How a Corn Plant Develops, Iowa State Special Report 48, p.11). Without a period of its own the tassel inherits dd_to_fruit_maturity, which is the
    // ear's 58-day R1-to-R6 grain fill, leaving it at roughly 40% of full size at silking and still expanding at physiological maturity. Six days covers the exsertion of an already-elongated tassel from the
    // whorl, which is what the growth ramp is standing in for here.
    phytomer_parameters_maize.inflorescence.inflorescence_maturity_period = 6;

    phytomer_parameters_maize.phytomer_creation_function = MaizePhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_maize;
    shoot_parameters_mainstem.vegetative_bud_break_probability_min = 0.5;
    shoot_parameters_mainstem.flower_bud_break_probability = 1;
    shoot_parameters_mainstem.phyllochron_min = 3;
    shoot_parameters_mainstem.elongation_rate_max = 0.1;
    shoot_parameters_mainstem.girth_area_factor = 6.f;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-250, 0);
    shoot_parameters_mainstem.internode_length_max = 0.1;
    shoot_parameters_mainstem.tortuosity = 10;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 1.0;
    shoot_parameters_mainstem.fruit_set_probability = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"}, {1.0});
    shoot_parameters_mainstem.max_nodes = 20;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem", shoot_parameters_mainstem);
}

uint PlantArchitecture::buildMaizePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize maize plant shoots
        initializeMaizeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.035f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.003,
                                     shoot_types.at("mainstem").internode_length_max.val(), 0.01, 0.01, 0.2, "mainstem");

    breakPlantDormancy(plantID);

    // Grain fill (R1 silking -> R6 physiological maturity) runs 55-65 days in the field
    // (Ritchie et al. 1993; Abendroth et al. 2011), not the 10 days previously used here.
    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 58, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeOliveTreeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.prototype_function = OliveLeafPrototype;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_olive(context_ptr->getRandomGenerator());

    phytomer_parameters_olive.internode.pitch = 0;
    phytomer_parameters_olive.internode.phyllotactic_angle.uniformDistribution(80, 100);
    phytomer_parameters_olive.internode.radius_initial = 0.002;
    phytomer_parameters_olive.internode.length_segments = 1;
    phytomer_parameters_olive.internode.image_texture = "OliveBark.jpg";
    phytomer_parameters_olive.internode.max_floral_buds_per_petiole = 3;

    phytomer_parameters_olive.petiole.petioles_per_internode = 2;
    phytomer_parameters_olive.petiole.pitch.uniformDistribution(-40, -20);
    phytomer_parameters_olive.petiole.taper = 0.1;
    phytomer_parameters_olive.petiole.curvature = 0;
    phytomer_parameters_olive.petiole.length = 0.01;
    phytomer_parameters_olive.petiole.radius = 0.0005;
    phytomer_parameters_olive.petiole.length_segments = 1;
    phytomer_parameters_olive.petiole.radial_subdivisions = 3;
    phytomer_parameters_olive.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_olive.leaf.leaves_per_petiole = 1;
    phytomer_parameters_olive.leaf.prototype_scale = 0.06;
    phytomer_parameters_olive.leaf.prototype = leaf_prototype;

    phytomer_parameters_olive.peduncle.length = 0.065;
    phytomer_parameters_olive.peduncle.radius = 0.001;
    phytomer_parameters_olive.peduncle.pitch = 60;
    phytomer_parameters_olive.peduncle.roll = 0;
    phytomer_parameters_olive.peduncle.length_segments = 1;
    phytomer_parameters_olive.peduncle.color = make_RGBcolor(0.7, 0.72, 0.7);

    phytomer_parameters_olive.inflorescence.flowers_per_peduncle = 10;
    phytomer_parameters_olive.inflorescence.flower_offset = 0.13;
    phytomer_parameters_olive.inflorescence.pitch.uniformDistribution(80, 100);
    phytomer_parameters_olive.inflorescence.roll.uniformDistribution(0, 360);
    phytomer_parameters_olive.inflorescence.flower_prototype_scale = 0.01;
    //    phytomer_parameters_olive.inflorescence.flower_prototype_function = OliveFlowerPrototype;
    phytomer_parameters_olive.inflorescence.fruit_prototype_scale = 0.025;
    phytomer_parameters_olive.inflorescence.fruit_prototype_function = OliveFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_olive;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.015;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 20;
    shoot_parameters_trunk.max_nodes = 20;
    // Rescaled when the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(), to keep base diameters at
    // max_age where they were: girth is now proportional to cumulative downstream leaf area, which runs well above the old value.
    shoot_parameters_trunk.girth_area_factor = 0.66f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 20;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_olive;
    //    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = OlivePhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = OlivePhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes.uniformDistribution(16, 24);
    shoot_parameters_proleptic.max_nodes_per_season.uniformDistribution(8, 12);
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.25;
    shoot_parameters_proleptic.girth_area_factor = 3.4f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.025;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 1.0;
    shoot_parameters_proleptic.vegetative_bud_break_time = 30;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(550, 650);
    shoot_parameters_proleptic.tortuosity = 100;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(35, 40);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 2;
    shoot_parameters_proleptic.internode_length_max = 0.05;
    shoot_parameters_proleptic.internode_length_min = 0.03;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.25;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.25;
    // Olive inflorescences are borne in the leaf axils of the previous year's growth; the terminal bud is vegetative.
    shoot_parameters_proleptic.max_terminal_floral_buds = 0;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"}, {1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    // Scaffolds are older wood than proleptic shoots, so they need their own factor to hold their base diameter.
    shoot_parameters_scaffold.girth_area_factor = 1.1f;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 30;
    shoot_parameters_scaffold.max_nodes_per_season = 10;
    shoot_parameters_scaffold.gravitropic_curvature = 700;
    shoot_parameters_scaffold.internode_length_max = 0.04;
    shoot_parameters_scaffold.tortuosity = 75;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
}

uint PlantArchitecture::buildOliveTree(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize olive tree shoots
        initializeOliveTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, 19, make_AxisRotation(context_ptr->randu(0.f, 0.025f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)),
                                      shoot_types.at("trunk").phytomer_parameters.internode.radius_initial.val(), 0.01, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 4; // context_ptr->randu(4,5);

    for (int i = 0; i < Nscaffolds; i++) {
        float pitch = context_ptr->randu(deg2rad(30), deg2rad(35));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, context_ptr->randu(5, 7), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007,
                                       shoot_types.at("scaffold").internode_length_max.val(), 1.f, 1.f, 0.5, "scaffold", 0);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200, 600, true);
    plant_instances.at(plantID).max_age = 1825;

    return plantID;
}

void PlantArchitecture::initializePistachioTreeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "PistachioLeaf.png";
    // Pistachio leaflets are broadly ovate.
    leaf_prototype.leaf_aspect_ratio = 0.65f;
    leaf_prototype.midrib_fold_fraction = 0.;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.4, 0.4);
    leaf_prototype.lateral_curvature = 0.;
    leaf_prototype.wave_period = 0.3f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.subdivisions = 3;
    leaf_prototype.unique_prototypes = 5;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_pistachio(context_ptr->getRandomGenerator());

    phytomer_parameters_pistachio.internode.pitch = 0;
    // Pistachio leaves are alternate and spirally arranged, at close to the golden angle; 160-200 deg made the branching nearly
    // two-ranked, with every shoot's laterals in one plane.
    phytomer_parameters_pistachio.internode.phyllotactic_angle.uniformDistribution(132.5, 142.5);
    phytomer_parameters_pistachio.internode.radius_initial = 0.002;
    phytomer_parameters_pistachio.internode.length_segments = 1;
    phytomer_parameters_pistachio.internode.image_texture = "OliveBark.jpg";
    phytomer_parameters_pistachio.internode.max_floral_buds_per_petiole = 3;

    phytomer_parameters_pistachio.petiole.petioles_per_internode = 1;
    phytomer_parameters_pistachio.petiole.pitch.uniformDistribution(-60, -45);
    phytomer_parameters_pistachio.petiole.taper = 0.1;
    phytomer_parameters_pistachio.petiole.curvature.uniformDistribution(-800, 800);
    phytomer_parameters_pistachio.petiole.length = 0.075;
    phytomer_parameters_pistachio.petiole.radius = 0.001;
    phytomer_parameters_pistachio.petiole.length_segments = 1;
    phytomer_parameters_pistachio.petiole.radial_subdivisions = 3;
    phytomer_parameters_pistachio.petiole.color = make_RGBcolor(0.6, 0.6, 0.4);

    phytomer_parameters_pistachio.leaf.leaves_per_petiole = 3;
    // The leaf is mostly trifoliate, with a terminal leaflet 8-13 cm long and 4-8 cm wide and side leaflets 5-8 cm long, which gives a
    // leaf of about 100 cm2. At the previous 8 cm terminal leaflet the leaf was 64 cm2.
    phytomer_parameters_pistachio.leaf.prototype_scale = 0.105;
    phytomer_parameters_pistachio.leaf.leaflet_offset = 0.3;
    phytomer_parameters_pistachio.leaf.leaflet_scale = 0.7;
    phytomer_parameters_pistachio.leaf.pitch.uniformDistribution(-20, 20);
    phytomer_parameters_pistachio.leaf.roll.uniformDistribution(-20, 20);
    phytomer_parameters_pistachio.leaf.prototype = leaf_prototype;

    phytomer_parameters_pistachio.peduncle.length = 0.1;
    phytomer_parameters_pistachio.peduncle.radius = 0.001;
    phytomer_parameters_pistachio.peduncle.pitch = 60;
    phytomer_parameters_pistachio.peduncle.roll = 0;
    phytomer_parameters_pistachio.peduncle.length_segments = 1;
    phytomer_parameters_pistachio.peduncle.curvature.uniformDistribution(500, 900);
    phytomer_parameters_pistachio.peduncle.color = make_RGBcolor(0.7, 0.72, 0.7);

    phytomer_parameters_pistachio.inflorescence.flowers_per_peduncle = 16;
    phytomer_parameters_pistachio.inflorescence.flower_offset = 0.08;
    phytomer_parameters_pistachio.inflorescence.pitch.uniformDistribution(50, 70);
    phytomer_parameters_pistachio.inflorescence.roll.uniformDistribution(0, 360);
    phytomer_parameters_pistachio.inflorescence.flower_prototype_scale = 0.025;
    // phytomer_parameters_pistachio.inflorescence.flower_prototype_function = PistachioFlowerPrototype;
    phytomer_parameters_pistachio.inflorescence.fruit_prototype_scale = 0.025;
    phytomer_parameters_pistachio.inflorescence.fruit_prototype_function = PistachioFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_pistachio;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 180;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.015;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 20;
    shoot_parameters_trunk.max_nodes = 20;
    // Calibrated against QSM reconstructions of ten 5th-leaf orchard trees (projects/QSMcalibration). Girth follows downstream leaf area,
    // and the calibrated crown carries far fewer shoots than before, so the factor is correspondingly larger; this gives a trunk close
    // to the 11.5 cm measured.
    shoot_parameters_trunk.girth_area_factor = 1.4f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 15;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"proleptic"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_pistachio;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = PistachioPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = PistachioPhytomerCallbackFunction;
    // A random tilt per internode (about every 6 cm) reads as a zig-zag along every branch; the measured orchard trees' branches bend
    // smoothly. Reduced from +/-15 deg (projects/QSMcalibration).
    shoot_parameters_proleptic.phytomer_parameters.internode.pitch.uniformDistribution(-4, 4);
    // A new internode starts at this radius and girth builds from the leaf area it supports, so the last few internodes of a shoot stay
    // at this value; at 2 mm, against the thick bases the calibrated girth gives, every shoot came to a sharp point.
    shoot_parameters_proleptic.phytomer_parameters.internode.radius_initial = 0.004;
    shoot_parameters_proleptic.max_nodes = 20;
    shoot_parameters_proleptic.max_nodes_per_season.uniformDistribution(8, 10);
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.25;
    // Calibrated against QSM reconstructions of ten 5th-leaf orchard trees (projects/QSMcalibration). The measured trees carry few, short
    // laterals, about 39 m of wood in total, and almost no branches growing through one another. Buds break with the same
    // probability all along the shoot. The long laterals are still concentrated near the tip, because internode length falls away with
    // distance below it (internode_length_decay_rate): a lateral that breaks further down grows as a spur, which does not branch or cross
    // the crown (see PistachioPhytomerCreationFunction()). The girth factor is sized to the leaf area the spurs and long shoots carry.
    shoot_parameters_proleptic.girth_area_factor = 2.2f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.38;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.38;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.5;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    // Calibrated against QSM reconstructions of ten 5th-leaf orchard trees (projects/QSMcalibration). At 350 every shoot turned up toward
    // vertical, where shoots from different parts of the crown converged and grew through one another; 250 keeps them apart while holding
    // the crown to the measured width once the trained forks carry two vigorous limbs each. The random wander is kept small for the same
    // reason.
    shoot_parameters_proleptic.gravitropic_curvature = 300;
    shoot_parameters_proleptic.tortuosity = 30;
    // Measured on the orchard trees (projects/QSMcalibration; chord over a child's first 0.15 m against the parent's chord about the
    // attachment): children at a parent's tip leave at a median 49 deg, and children below it at a nearly constant 73-75 deg. The model's
    // angle rises linearly by insertion_angle_decay_rate per node below the tip, so 12 deg/node on 6 cm internodes would reach about 74 deg two
    // nodes down; 18 deg/node reaches it one node down, closer to the measured step, and fills out the crown.
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(40, 50);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 18;
    // Each new shoot starts its phyllotaxy at an independent orientation around its parent. With a fixed roll every generation's buds sat at
    // the same angles relative to their parent, so the branch that continued a limb tended to stay in the plane of the one before it and
    // limb systems flattened into fans (a fifth of branching chains near-coplanar, against 5% in the measured trees).
    shoot_parameters_proleptic.base_roll.uniformDistribution(0, 360);
    shoot_parameters_proleptic.internode_length_max = 0.06;
    shoot_parameters_proleptic.internode_length_min = 0.004;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.06;
    shoot_parameters_proleptic.fruit_set_probability = 0.2;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.35;
    // Pistachio bears its inflorescence buds laterally, in the leaf axils of one-year-old wood; the terminal bud is vegetative.
    shoot_parameters_proleptic.max_terminal_floral_buds = 0;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;

    // Main scaffolds. Identical to proleptic shoots in how they grow, but a type of their own so that isStructuralShoot() treats them
    // as structure: built as "proleptic", shade-driven branch shedding could remove whole scaffolds with everything they carried.
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    // Scaffolds clothe themselves in laterals along their whole length (projects/QSMcalibration). With the proleptic values, a scaffold's
    // laterals came almost only from the few buds near its tip, so whether each scaffold carried any crown was a draw: many trees had one
    // or two bare scaffolds and the crown on the other side.
    shoot_parameters_scaffold.vegetative_bud_break_probability_max = 1.0;
    shoot_parameters_scaffold.vegetative_bud_break_probability_min = 0.2;
    shoot_parameters_scaffold.vegetative_bud_break_probability_decay_rate = 0.7;
    // The tip buds always break so that a scaffold forks where it ends.
    shoot_parameters_scaffold.max_nodes = 30;
    // The scaffolds keep a weaker upward response than the proleptic shoots copied above; the calibration was made with them at 150.
    shoot_parameters_scaffold.gravitropic_curvature = 180;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});

    // Spurs. A lateral vegetative bud that does not grow a long shoot grows a short one instead: a few leaves in a rosette on a
    // centimetre of wood, which extends by as much again each year. These carry most of a pistachio's foliage, clothing the limbs behind
    // the current season's extension growth; without them the model bore leaves only on that extension growth, about a fifth of the leaf
    // area measured by terrestrial LiDAR in the orchard the wood is calibrated against (projects/QSMcalibration). They are released by
    // PistachioPhytomerCallbackFunction() rather than by the bud-break probabilities, which set how many long shoots there are.
    ShootParameters shoot_parameters_spur = shoot_parameters_proleptic;
    shoot_parameters_spur.phytomer_parameters.internode.radius_initial = 0.002;
    shoot_parameters_spur.max_nodes = 40;
    shoot_parameters_spur.max_nodes_per_season = 4;
    shoot_parameters_spur.internode_length_max = 0.004;
    shoot_parameters_spur.internode_length_min = 0.004;
    shoot_parameters_spur.internode_length_decay_rate = 0;
    // A spur does not branch.
    shoot_parameters_spur.vegetative_bud_break_probability_max = 0;
    shoot_parameters_spur.vegetative_bud_break_probability_min = 0;
    shoot_parameters_spur.vegetative_bud_break_probability_decay_rate = 0;
    shoot_parameters_spur.gravitropic_curvature = 0;
    shoot_parameters_spur.defineChildShootTypes({"spur"}, {1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("spur", shoot_parameters_spur);
}

uint PlantArchitecture::buildPistachioTree(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize pistachio tree shoots
        initializePistachioTreeShoots();
    }

    // Get training system parameters (with defaults matching original hard-coded values)
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 1.0f, 0.1f, 3.f, "total trunk height in meters");
    auto num_scaffolds = uint(getParameterValue(current_build_parameters, "num_scaffolds", 4.f, 2.f, 8.f, "number of scaffold branches"));
    // 57 deg: the measured orchard scaffolds leave the trunk at a median 56-57 deg from vertical.
    auto scaffold_angle = getParameterValue(current_build_parameters, "scaffold_angle", 57.f, 20.f, 70.f, "scaffold branch angle in degrees");

    // Calculate trunk nodes based on desired height and internode length
    float trunk_internode_length = 0.05f; // Default internode length for pistachio
    uint trunk_nodes = uint(trunk_height / trunk_internode_length);
    if (trunk_nodes < 1)
        trunk_nodes = 1;

    // Fixed training parameters (not user-customizable). Each scaffold starts as a single node whose own buds are dead: it stands in for
    // the bud below the heading cut from which the scaffold grows, so in its first season only its tip extends, and its laterals are on
    // that season's growth (as in a tree headed at planting). Built with several nodes of pre-existing wood, the scaffolds pushed laterals
    // from that wood in the first spring as well, a season earlier than a real scaffold can. Its internode length is that of the
    // proleptic shoots it would otherwise be.
    float scaffold_radius = 0.007f;
    float scaffold_length = 0.06f;
    uint scaffold_nodes = 1;

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)),
                                      shoot_types.at("trunk").phytomer_parameters.internode.radius_initial.val(), trunk_internode_length, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    float pitch = deg2rad(scaffold_angle);
    // Scaffolds are attached in opposite pairs: pair m sits m nodes below the top of the trunk and is rotated by m * pi / num_pairs
    // in azimuth, so the pairs share the circle evenly. With an odd count the last pair has one member. For 2-4 scaffolds this
    // is the original cardinal-direction layout (0 and pi on the top node, pi/2 and 3pi/2 on the node below).
    uint num_pairs = (num_scaffolds + 1) / 2;
    uint trunk_node_count = getShootNodeCount(plantID, uID_trunk);
    if (num_pairs > trunk_node_count) {
        helios_runtime_error("ERROR (PlantArchitecture::buildPistachioTree): " + std::to_string(num_scaffolds) + " scaffolds need " + std::to_string(num_pairs) + " trunk nodes to attach to, but a trunk_height of " + std::to_string(trunk_height) +
                             " m gives only " + std::to_string(trunk_node_count) + ". Increase trunk_height or reduce num_scaffolds.");
    }
    for (uint scaffold = 0; scaffold < num_scaffolds; scaffold++) {
        uint pair = scaffold / 2;
        float azimuth = float(pair) * float(M_PI) / float(num_pairs) + float(scaffold % 2) * float(M_PI);
        // The roll sets which side of the scaffold its phyllotactic sequence, and so its first laterals, start on; drawn at random like
        // the proleptic shoots' base_roll so that the scaffolds do not all put their laterals on the same side.
        uint scaffoldID = addChildShoot(plantID, uID_trunk, trunk_node_count - 1 - pair, scaffold_nodes, make_AxisRotation(pitch + context_ptr->randu(-0.15f, 0.15f), azimuth, context_ptr->randu(0.f, 2.f * float(M_PI))), scaffold_radius,
                                         scaffold_length, 1.f, 1.f, 0.5, "scaffold", 0);
        for (const auto &phytomer: plant_instances.at(plantID).shoot_tree.at(scaffoldID)->phytomers) {
            phytomer->setVegetativeBudState(BUD_DEAD);
            phytomer->setFloralBudState(BUD_DEAD);
        }
    }


    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, -1, 7, 20, 200);
    // Five seasons, matching the 5th-leaf orchard trees the model is calibrated against (projects/QSMcalibration).
    plant_instances.at(plantID).max_age = 1825;

    return plantID;
}

void PlantArchitecture::initializePuncturevineShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "PuncturevineLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.4f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature = -0.1f;
    leaf_prototype.lateral_curvature = 0.4f;
    leaf_prototype.subdivisions = 1;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_puncturevine(context_ptr->getRandomGenerator());

    phytomer_parameters_puncturevine.internode.pitch.uniformDistribution(0, 15);
    phytomer_parameters_puncturevine.internode.phyllotactic_angle = 180.f;
    phytomer_parameters_puncturevine.internode.radius_initial = 0.001;
    phytomer_parameters_puncturevine.internode.color = make_RGBcolor(0.28, 0.18, 0.13);
    phytomer_parameters_puncturevine.internode.length_segments = 1;

    // Puncturevine leaves are opposite.
    phytomer_parameters_puncturevine.petiole.petioles_per_internode = 2;
    phytomer_parameters_puncturevine.petiole.pitch.uniformDistribution(60, 80);
    phytomer_parameters_puncturevine.petiole.radius = 0.0005;
    phytomer_parameters_puncturevine.petiole.length = 0.03;
    phytomer_parameters_puncturevine.petiole.taper = 0;
    phytomer_parameters_puncturevine.petiole.curvature = 0;
    phytomer_parameters_puncturevine.petiole.color = phytomer_parameters_puncturevine.internode.color;
    phytomer_parameters_puncturevine.petiole.length_segments = 1;

    // The leaves are even-pinnate, with 6-16 leaflets in pairs and no terminal leaflet. An even count places the leaflets in pairs.
    phytomer_parameters_puncturevine.leaf.leaves_per_petiole = 10;
    phytomer_parameters_puncturevine.leaf.pitch.uniformDistribution(0, 40);
    phytomer_parameters_puncturevine.leaf.yaw = 30;
    phytomer_parameters_puncturevine.leaf.roll.uniformDistribution(-5, 5);
    phytomer_parameters_puncturevine.leaf.prototype_scale = 0.012;
    phytomer_parameters_puncturevine.leaf.leaflet_offset = 0.18;
    phytomer_parameters_puncturevine.leaf.leaflet_scale = 1;
    phytomer_parameters_puncturevine.leaf.prototype = leaf_prototype;

    phytomer_parameters_puncturevine.peduncle.length = 0.001;
    phytomer_parameters_puncturevine.peduncle.color = phytomer_parameters_puncturevine.internode.color;

    phytomer_parameters_puncturevine.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_puncturevine.inflorescence.pitch = -90.f;
    phytomer_parameters_puncturevine.inflorescence.flower_prototype_function = PuncturevineFlowerPrototype;
    phytomer_parameters_puncturevine.inflorescence.flower_prototype_scale = 0.01;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_primary(context_ptr->getRandomGenerator());
    shoot_parameters_primary.phytomer_parameters = phytomer_parameters_puncturevine;
    shoot_parameters_primary.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_primary.vegetative_bud_break_probability_decay_rate = 1.f;
    shoot_parameters_primary.vegetative_bud_break_time = 10;
    shoot_parameters_primary.base_roll = 90;
    shoot_parameters_primary.phyllochron_min = 1;
    shoot_parameters_primary.elongation_rate_max = 0.2;
    shoot_parameters_primary.girth_area_factor = 0.f;
    shoot_parameters_primary.internode_length_max = 0.02;
    shoot_parameters_primary.internode_length_decay_rate = 0;
    shoot_parameters_primary.insertion_angle_tip.uniformDistribution(75, 85);
    shoot_parameters_primary.insertion_angle_decay_rate = 0;
    shoot_parameters_primary.flowers_require_dormancy = false;
    shoot_parameters_primary.growth_requires_dormancy = false;
    shoot_parameters_primary.flower_bud_break_probability = 0.2;
    shoot_parameters_primary.determinate_shoot_growth = false;
    shoot_parameters_primary.max_nodes = 15;
    shoot_parameters_primary.gravitropic_curvature = 25;
    shoot_parameters_primary.tortuosity = 0;
    shoot_parameters_primary.defineChildShootTypes({"secondary_puncturevine"}, {1.f});

    ShootParameters shoot_parameters_base = shoot_parameters_primary;
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_puncturevine;
    shoot_parameters_base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5 - 10, 137.5 + 10);
    shoot_parameters_base.phytomer_parameters.petiole.petioles_per_internode = 0;
    shoot_parameters_base.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_base.phytomer_parameters.petiole.pitch = 0;
    shoot_parameters_base.vegetative_bud_break_probability_min = 1;
    shoot_parameters_base.vegetative_bud_break_time = 2;
    shoot_parameters_base.phyllochron_min = 2;
    shoot_parameters_base.elongation_rate_max = 0.15;
    shoot_parameters_base.gravitropic_curvature = 0;
    shoot_parameters_base.internode_length_max = 0.01;
    shoot_parameters_base.internode_length_decay_rate = 0;
    shoot_parameters_base.insertion_angle_tip = 90;
    shoot_parameters_base.insertion_angle_decay_rate = 0;
    shoot_parameters_base.flowers_require_dormancy = false;
    shoot_parameters_base.growth_requires_dormancy = false;
    shoot_parameters_base.flower_bud_break_probability = 0.0;
    shoot_parameters_base.max_nodes.uniformDistribution(3, 5);
    shoot_parameters_base.defineChildShootTypes({"primary_puncturevine"}, {1.f});

    ShootParameters shoot_parameters_children = shoot_parameters_primary;
    shoot_parameters_children.base_roll = 0;

    defineShootType("base_puncturevine", shoot_parameters_base);
    defineShootType("primary_puncturevine", shoot_parameters_primary);
    defineShootType("secondary_puncturevine", shoot_parameters_children);
}

uint PlantArchitecture::buildPuncturevinePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize puncturevine plant shoots
        initializePuncturevineShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0.f, 0.f), 0.001, 0.001, 1, 1, 0, "base_puncturevine");

    breakPlantDormancy(plantID);

    plant_instances.at(plantID).max_age = 45;

    setPlantPhenologicalThresholds(plantID, 0, -1, 14, -1, -1, 1000);

    return plantID;
}

void PlantArchitecture::initializeEasternRedbudShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "RedbudLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature = -0.15f;
    leaf_prototype.lateral_curvature = -0.1f;
    leaf_prototype.wave_period = 0.3f;
    leaf_prototype.wave_amplitude = 0.025f;
    leaf_prototype.subdivisions = 5;
    leaf_prototype.unique_prototypes = 5;
    leaf_prototype.leaf_offset = make_vec3(-0.3, 0, 0);

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_redbud(context_ptr->getRandomGenerator());

    phytomer_parameters_redbud.internode.pitch = 15;
    phytomer_parameters_redbud.internode.phyllotactic_angle.uniformDistribution(170, 190);
    phytomer_parameters_redbud.internode.radius_initial = 0.0015;
    phytomer_parameters_redbud.internode.image_texture = "WesternRedbudBark.jpg";
    phytomer_parameters_redbud.internode.color.scale(0.3);
    phytomer_parameters_redbud.internode.length_segments = 1;
    phytomer_parameters_redbud.internode.max_floral_buds_per_petiole = 5;

    phytomer_parameters_redbud.petiole.petioles_per_internode = 1;
    phytomer_parameters_redbud.petiole.color = make_RGBcolor(0.65, 0.52, 0.39);
    phytomer_parameters_redbud.petiole.pitch.uniformDistribution(20, 40);
    phytomer_parameters_redbud.petiole.radius = 0.002;
    phytomer_parameters_redbud.petiole.length = 0.075;
    phytomer_parameters_redbud.petiole.taper = 0;
    phytomer_parameters_redbud.petiole.curvature = 0;
    phytomer_parameters_redbud.petiole.length_segments = 1;

    phytomer_parameters_redbud.leaf.leaves_per_petiole = 1;
    phytomer_parameters_redbud.leaf.pitch.uniformDistribution(-110, -80);
    phytomer_parameters_redbud.leaf.yaw = 0;
    phytomer_parameters_redbud.leaf.roll.uniformDistribution(-5, 5);
    phytomer_parameters_redbud.leaf.prototype_scale = 0.1;
    phytomer_parameters_redbud.leaf.prototype = leaf_prototype;

    phytomer_parameters_redbud.peduncle.length = 0.02;
    phytomer_parameters_redbud.peduncle.pitch.uniformDistribution(50, 90);
    phytomer_parameters_redbud.peduncle.color = make_RGBcolor(0.32, 0.05, 0.13);

    phytomer_parameters_redbud.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_redbud.inflorescence.pitch = 0;
    phytomer_parameters_redbud.inflorescence.flower_prototype_function = RedbudFlowerPrototype;
    phytomer_parameters_redbud.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_redbud.inflorescence.fruit_prototype_function = RedbudFruitPrototype;
    phytomer_parameters_redbud.inflorescence.fruit_prototype_scale = 0.1;
    phytomer_parameters_redbud.inflorescence.fruit_gravity_factor_fraction = 0.7;

    phytomer_parameters_redbud.phytomer_creation_function = RedbudPhytomerCreationFunction;
    phytomer_parameters_redbud.phytomer_callback_function = RedbudPhytomerCallbackFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_main(context_ptr->getRandomGenerator());
    shoot_parameters_main.phytomer_parameters = phytomer_parameters_redbud;
    shoot_parameters_main.vegetative_bud_break_probability_min = 1.0;
    shoot_parameters_main.vegetative_bud_break_time = 2;
    shoot_parameters_main.phyllochron_min = 2;
    shoot_parameters_main.elongation_rate_max = 0.1;
    // Rescaled when the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(), to keep base diameters at
    // max_age where they were: girth is now proportional to cumulative downstream leaf area, which runs well above the old value.
    shoot_parameters_main.girth_area_factor = 3.0f;
    shoot_parameters_main.gravitropic_curvature = 300;
    shoot_parameters_main.tortuosity = 125;
    shoot_parameters_main.internode_length_max = 0.04;
    shoot_parameters_main.internode_length_decay_rate = 0.005;
    shoot_parameters_main.internode_length_min = 0.01;
    shoot_parameters_main.insertion_angle_tip = 75;
    shoot_parameters_main.insertion_angle_decay_rate = 10;
    shoot_parameters_main.flowers_require_dormancy = true;
    shoot_parameters_main.growth_requires_dormancy = true;
    shoot_parameters_main.determinate_shoot_growth = false;
    shoot_parameters_main.max_terminal_floral_buds = 0;
    shoot_parameters_main.flower_bud_break_probability = 0.8;
    shoot_parameters_main.fruit_set_probability = 0.3;
    shoot_parameters_main.max_nodes = 25;
    shoot_parameters_main.max_nodes_per_season = 10;
    shoot_parameters_main.base_roll = 90;

    ShootParameters shoot_parameters_trunk = shoot_parameters_main;
    // The trunk is older wood than the shoots, so it needs its own factor to hold its base diameter.
    shoot_parameters_trunk.girth_area_factor = 1.13f;
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 15;
    shoot_parameters_trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_trunk.insertion_angle_tip = 60;
    shoot_parameters_trunk.max_nodes = 75;
    shoot_parameters_trunk.max_nodes_per_season = 10;
    shoot_parameters_trunk.tortuosity = 37.5;
    shoot_parameters_trunk.defineChildShootTypes({"eastern_redbud_shoot"}, {1.f});

    defineShootType("eastern_redbud_trunk", shoot_parameters_trunk);
    defineShootType("eastern_redbud_shoot", shoot_parameters_main);
}

uint PlantArchitecture::buildEasternRedbudPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize redbud plant shoots
        initializeEasternRedbudShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 16, make_AxisRotation(context_ptr->randu(0, 0.1 * M_PI), context_ptr->randu(0, 2 * M_PI), context_ptr->randu(0, 2 * M_PI)), 0.0075, 0.05, 1, 1, 0.4, "eastern_redbud_trunk");

    makePlantDormant(plantID);
    breakPlantDormancy(plantID);

    // leave four vegetative buds on the trunk and remove the rest
    for (auto &phytomer: this->plant_instances.at(plantID).shoot_tree.at(uID_stem)->phytomers) {
        if (phytomer->shoot_index.x < 12) {
            for (auto &petiole: phytomer->axillary_vegetative_buds) {
                for (auto &vbud: petiole) {
                    phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                }
            }
        }
    }

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200);

    plant_instances.at(plantID).max_age = 1460;

    return plantID;
}

void PlantArchitecture::initializeRiceShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.06f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.2, 0);
    leaf_prototype.lateral_curvature = -0.3;
    leaf_prototype.wave_period = 0.1f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.subdivisions = 20;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_rice(context_ptr->getRandomGenerator());

    phytomer_parameters_rice.internode.pitch = 0;
    phytomer_parameters_rice.internode.phyllotactic_angle.uniformDistribution(67, 77);
    phytomer_parameters_rice.internode.radius_initial = 0.001;
    phytomer_parameters_rice.internode.color = make_RGBcolor(0.27, 0.31, 0.16);
    phytomer_parameters_rice.internode.length_segments = 1;
    phytomer_parameters_rice.internode.radial_subdivisions = 6;
    phytomer_parameters_rice.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_rice.internode.max_vegetative_buds_per_petiole = 0;

    phytomer_parameters_rice.petiole.petioles_per_internode = 1;
    phytomer_parameters_rice.petiole.pitch.uniformDistribution(-40, 0);
    phytomer_parameters_rice.petiole.radius = 0.0;
    phytomer_parameters_rice.petiole.length = 0.01;
    phytomer_parameters_rice.petiole.taper = 0;
    phytomer_parameters_rice.petiole.curvature = 0;
    phytomer_parameters_rice.petiole.length_segments = 1;

    phytomer_parameters_rice.leaf.leaves_per_petiole = 1;
    phytomer_parameters_rice.leaf.pitch = 0;
    phytomer_parameters_rice.leaf.yaw = 0;
    phytomer_parameters_rice.leaf.roll = 0;
    phytomer_parameters_rice.leaf.prototype_scale = 0.15;
    phytomer_parameters_rice.leaf.prototype = leaf_prototype;

    phytomer_parameters_rice.peduncle.pitch = 0;
    phytomer_parameters_rice.peduncle.length.uniformDistribution(0.14, 0.18);
    phytomer_parameters_rice.peduncle.radius = 0.0005;
    phytomer_parameters_rice.peduncle.color = phytomer_parameters_rice.internode.color;
    phytomer_parameters_rice.peduncle.curvature.uniformDistribution(-800, -50);
    phytomer_parameters_rice.peduncle.radial_subdivisions = 6;
    phytomer_parameters_rice.peduncle.length_segments = 8;

    phytomer_parameters_rice.inflorescence.flowers_per_peduncle = 60;
    phytomer_parameters_rice.inflorescence.pitch.uniformDistribution(20, 25);
    phytomer_parameters_rice.inflorescence.roll = 0;
    phytomer_parameters_rice.inflorescence.fruit_prototype_scale = 0.008;
    phytomer_parameters_rice.inflorescence.flower_offset = 0.012;
    phytomer_parameters_rice.inflorescence.fruit_prototype_function = RiceSpikePrototype;

    //    phytomer_parameters_rice.phytomer_creation_function = RicePhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_rice;
    shoot_parameters_mainstem.vegetative_bud_break_probability_min = 0;
    shoot_parameters_mainstem.flower_bud_break_probability = 1;
    shoot_parameters_mainstem.phyllochron_min = 2;
    shoot_parameters_mainstem.elongation_rate_max = 0.1;
    shoot_parameters_mainstem.girth_area_factor = 5.f;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-1000, -400);
    shoot_parameters_mainstem.internode_length_max = 0.0075;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.fruit_set_probability = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"}, {1.0});
    shoot_parameters_mainstem.max_nodes = 30;
    shoot_parameters_mainstem.max_terminal_floral_buds = 5;

    defineShootType("mainstem", shoot_parameters_mainstem);
}

uint PlantArchitecture::buildRicePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize rice plant shoots
        initializeRiceShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.1f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.001, 0.0075, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 10, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeButterLettuceShoots() {

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "RomaineLettuceLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.85f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.2, 0.05);
    leaf_prototype.lateral_curvature = -0.4f;
    leaf_prototype.wave_period.uniformDistribution(0.15, 0.25);
    leaf_prototype.wave_amplitude.uniformDistribution(0.05, 0.1);
    leaf_prototype.subdivisions = 30;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 137.5;
    phytomer_parameters.internode.radius_initial = 0.02;
    phytomer_parameters.internode.color = make_RGBcolor(0.402, 0.423, 0.413);
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.internode.radial_subdivisions = 10;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(0, 30);
    phytomer_parameters.petiole.radius = 0.001;
    phytomer_parameters.petiole.length = 0.001;
    phytomer_parameters.petiole.length_segments = 1;
    phytomer_parameters.petiole.radial_subdivisions = 3;
    phytomer_parameters.petiole.color = RGB::red;

    phytomer_parameters.leaf.leaves_per_petiole = 1;
    phytomer_parameters.leaf.pitch = 10;
    phytomer_parameters.leaf.yaw = 0;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.15, 0.25);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.phytomer_creation_function = ButterLettucePhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters;
    shoot_parameters_mainstem.vegetative_bud_break_probability_min = 0;
    shoot_parameters_mainstem.phyllochron_min = 2;
    shoot_parameters_mainstem.elongation_rate_max = 0.15;
    shoot_parameters_mainstem.girth_area_factor = 0.f;
    shoot_parameters_mainstem.gravitropic_curvature = 10;
    shoot_parameters_mainstem.internode_length_max = 0.001;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 0.0;
    shoot_parameters_mainstem.max_nodes = 25;

    defineShootType("mainstem", shoot_parameters_mainstem);
}

uint PlantArchitecture::buildButterLettucePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize lettuce plant shoots
        initializeButterLettuceShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(context_ptr->randu(0.f, 0.03f * M_PI), 0.f, context_ptr->randu(0.f, 2.f * M_PI)), 0.005, 0.001, 1, 1, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeSorghumShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.15f;
    leaf_prototype.midrib_fold_fraction = 0.45f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.07, -0.04);
    leaf_prototype.longitudinal_curvature_exponent = 3.f;
    leaf_prototype.flexibility_taper = 150.f;
    leaf_prototype.flexibility_aging = 11.f;
    leaf_prototype.flexibility_aging_max = 6.f;
    leaf_prototype.lateral_curvature = -0.3f;
    leaf_prototype.petiole_roll = 0.f;
    leaf_prototype.wave_period = 0.1f;
    leaf_prototype.wave_amplitude = 0.06f;
    leaf_prototype.flexibility.uniformDistribution(1.2, 1.8);
    leaf_prototype.subdivisions = 50;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_sorghum(context_ptr->getRandomGenerator());

    phytomer_parameters_sorghum.internode.pitch = 0;
    phytomer_parameters_sorghum.internode.phyllotactic_angle.uniformDistribution(155, 205);
    phytomer_parameters_sorghum.internode.radius_initial = 0.003;
    phytomer_parameters_sorghum.internode.color = make_RGBcolor(0.09, 0.13, 0.06);
    phytomer_parameters_sorghum.internode.length_segments = 2;
    phytomer_parameters_sorghum.internode.radial_subdivisions = 10;
    phytomer_parameters_sorghum.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_sorghum.internode.max_vegetative_buds_per_petiole = 0;

    phytomer_parameters_sorghum.petiole.petioles_per_internode = 1;
    phytomer_parameters_sorghum.petiole.pitch.uniformDistribution(-25, -12);
    phytomer_parameters_sorghum.petiole.radius = 0.0;
    phytomer_parameters_sorghum.petiole.length = 0.002;
    phytomer_parameters_sorghum.petiole.taper = 0;
    phytomer_parameters_sorghum.petiole.curvature = 0;
    phytomer_parameters_sorghum.petiole.length_segments = 1;

    phytomer_parameters_sorghum.leaf.leaves_per_petiole = 1;
    phytomer_parameters_sorghum.leaf.pitch = 0;
    phytomer_parameters_sorghum.leaf.yaw = 0;
    phytomer_parameters_sorghum.leaf.roll.uniformDistribution(-12, 12);
    phytomer_parameters_sorghum.leaf.prototype_scale = 0.65;
    phytomer_parameters_sorghum.leaf.prototype = leaf_prototype;

    phytomer_parameters_sorghum.peduncle.length = 0.3;
    phytomer_parameters_sorghum.peduncle.radius = 0.008;
    phytomer_parameters_sorghum.peduncle.color = phytomer_parameters_sorghum.internode.color;
    phytomer_parameters_sorghum.peduncle.radial_subdivisions = 10;

    phytomer_parameters_sorghum.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_sorghum.inflorescence.pitch = 0;
    phytomer_parameters_sorghum.inflorescence.roll = 0;
    phytomer_parameters_sorghum.inflorescence.fruit_prototype_scale = 0.18;
    phytomer_parameters_sorghum.inflorescence.fruit_prototype_function = SorghumPaniclePrototype;

    phytomer_parameters_sorghum.phytomer_creation_function = SorghumPhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_sorghum;
    shoot_parameters_mainstem.vegetative_bud_break_probability_min = 0;
    shoot_parameters_mainstem.flower_bud_break_probability = 1;
    shoot_parameters_mainstem.phyllochron_min = 3.5;
    shoot_parameters_mainstem.elongation_rate_max = 0.05;
    shoot_parameters_mainstem.girth_area_factor = 5.f;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-700, -200);
    shoot_parameters_mainstem.internode_length_max = 0.075;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 1.0;
    shoot_parameters_mainstem.fruit_set_probability = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"}, {1.0});
    shoot_parameters_mainstem.max_nodes = 16;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem", shoot_parameters_mainstem);
}

uint PlantArchitecture::buildSorghumPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize sorghum plant shoots
        initializeSorghumShoots();
    }

    uint plantID = addPlantInstance(base_position - make_vec3(0, 0, 0.025), 0);

    // The base stem is built upright. A tilt here is not confined to leaning the plant over: the phyllotactic rotation of each successive phytomer is applied about its own internode axis, so tilting the
    // base makes those axes non-parallel and the two ranks of a distichous grass converge instead of staying opposite. The previous 0.075*pi tilt cost roughly 18 degrees of rank separation.
    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(0.f, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.003,
                                     shoot_types.at("mainstem").internode_length_max.val(), 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    // Grain fill (GS6 half-bloom -> GS9 physiological maturity) takes about 35 days, within a
    // reported 25-45 day range (Vanderlip, Kansas State S-3; Gerik et al., Texas A&M B-6137).
    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 35, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeSoybeanShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "SoybeanLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(0.1, 0.2);
    leaf_prototype.lateral_curvature = 0.45;
    leaf_prototype.subdivisions = 8;
    leaf_prototype.unique_prototypes = 5;
    leaf_prototype.build_petiolule = true;
    // Petiolule length as a fraction of blade length: about 3 mm, within the 1.5-4 mm given by Flora of China for Glycine max. Its radius follows the petiole.
    leaf_prototype.petiolule_length = {{0, 0.025f}};

    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());

    phytomer_parameters_trifoliate.internode.pitch = 20;
    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters_trifoliate.internode.radius_initial = 0.002;
    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.2, 0.25, 0.05);
    phytomer_parameters_trifoliate.internode.length_segments = 5;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(15, 40);
    phytomer_parameters_trifoliate.petiole.radius = 0.002;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.12, 0.16);
    phytomer_parameters_trifoliate.petiole.taper = 0.25;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-250, 50);
    phytomer_parameters_trifoliate.petiole.color = phytomer_parameters_trifoliate.internode.color;
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.uniformDistribution(-30, 10);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll.uniformDistribution(-25, 5);
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.5;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.1, 0.14);
    phytomer_parameters_trifoliate.leaf.prototype = leaf_prototype;

    phytomer_parameters_trifoliate.peduncle.length = 0.01;
    phytomer_parameters_trifoliate.peduncle.radius = 0.0005;
    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 40);
    phytomer_parameters_trifoliate.peduncle.roll = 90;
    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(-500, 500);
    phytomer_parameters_trifoliate.peduncle.color = phytomer_parameters_trifoliate.internode.color;
    phytomer_parameters_trifoliate.peduncle.length_segments = 1;
    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.inflorescence.flowers_per_peduncle.uniformDistribution(1, 4);
    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.2;
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50, 70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20, 20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.015;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = SoybeanFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.1, 0.12);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = SoybeanFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8, 1.0);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.01;
    phytomer_parameters_unifoliate.petiole.radius = 0.001;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(60, 80);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.02;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype = leaf_prototype;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 25;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(20, 30);
    //    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.035;
    //    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
    //    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters_trifoliate.gravitropic_curvature = 400;

    shoot_parameters_trifoliate.phyllochron_min = 2;
    shoot_parameters_trifoliate.elongation_rate_max = 0.1;
    shoot_parameters_trifoliate.girth_area_factor = 2.f;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 15;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_min = 0.05;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_decay_rate = 0.6;
    //    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.8, 1.0);
    shoot_parameters_trifoliate.fruit_set_probability = 0.4;
    //    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
    //    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
    //    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)

    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"}, {1.0});


    ShootParameters shoot_parameters_unifoliate = shoot_parameters_trifoliate;
    shoot_parameters_unifoliate.phytomer_parameters = phytomer_parameters_unifoliate;
    shoot_parameters_unifoliate.max_nodes = 1;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 0;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 8;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"}, {1.0});

    defineShootType("unifoliate", shoot_parameters_unifoliate);
    defineShootType("trifoliate", shoot_parameters_trifoliate);
}

uint PlantArchitecture::buildSoybeanPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize bean plant shoots
        initializeSoybeanShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.04, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").phytomer_parameters.internode.radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeStrawberryShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "StrawberryLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature = 0.15f;
    leaf_prototype.lateral_curvature = -0.35f;
    leaf_prototype.wave_period = 0.4f;
    leaf_prototype.wave_amplitude = 0.03f;
    leaf_prototype.subdivisions = 6;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    // Strawberry leaves are arranged on the crown in a 2/5 spiral, 144 deg between successive leaves, so that every sixth leaf stands above the first.
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(144 - 10, 144 + 10);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.15, 0.2, 0.1);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(10, 45);
    phytomer_parameters.petiole.radius = 0.0025;
    phytomer_parameters.petiole.length.uniformDistribution(0.15, 0.35);
    phytomer_parameters.petiole.taper = 0.5;
    phytomer_parameters.petiole.curvature.uniformDistribution(-150, -50);
    phytomer_parameters.petiole.color = make_RGBcolor(0.18, 0.23, 0.1);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 3;
    phytomer_parameters.leaf.pitch.uniformDistribution(-35, 0);
    phytomer_parameters.leaf.yaw = -30;
    phytomer_parameters.leaf.roll = -30;
    phytomer_parameters.leaf.leaflet_offset = 0.01;
    phytomer_parameters.leaf.leaflet_scale = 1.0;
    phytomer_parameters.leaf.prototype_scale = 0.12;
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length.uniformDistribution(0.16, 0.2);
    phytomer_parameters.peduncle.radius = 0.0018;
    phytomer_parameters.peduncle.pitch.uniformDistribution(35, 55);
    phytomer_parameters.peduncle.roll = 90;
    phytomer_parameters.peduncle.curvature = -150;
    phytomer_parameters.peduncle.length_segments = 5;
    phytomer_parameters.peduncle.radial_subdivisions = 6;
    phytomer_parameters.peduncle.color = phytomer_parameters.petiole.color;

    phytomer_parameters.inflorescence.flowers_per_peduncle = 1; //.uniformDistribution(1, 2);
    phytomer_parameters.inflorescence.flower_offset = 0.2;
    phytomer_parameters.inflorescence.pitch = 70;
    phytomer_parameters.inflorescence.roll = 90;
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters.inflorescence.flower_prototype_function = StrawberryFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.085;
    phytomer_parameters.inflorescence.fruit_prototype_function = StrawberryFruitPrototype;
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction = 0.65;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;

    shoot_parameters.max_nodes = 15;
    shoot_parameters.insertion_angle_tip = 40;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.015;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature.uniformDistribution(-10, 0);
    shoot_parameters.tortuosity = 0;

    shoot_parameters.phyllochron_min = 2;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 2.f;
    shoot_parameters.vegetative_bud_break_time = 15;
    shoot_parameters.vegetative_bud_break_probability_min = 0.;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = -0.5;
    shoot_parameters.flower_bud_break_probability = 1;
    shoot_parameters.fruit_set_probability = 0.3;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = true;

    shoot_parameters.defineChildShootTypes({"mainstem"}, {1.0});

    defineShootType("mainstem", shoot_parameters);
}

uint PlantArchitecture::buildStrawberryPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize strawberry plant shoots
        initializeStrawberryShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.001, 0.004, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000);

    plant_instances.at(plantID).max_age = 120;

    return plantID;
}

void PlantArchitecture::initializeSugarbeetShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "SugarbeetLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.4f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature = -0.2f;
    leaf_prototype.lateral_curvature = -0.4f;
    leaf_prototype.petiole_roll = 0.75f;
    leaf_prototype.wave_period.uniformDistribution(0.08f, 0.15f);
    leaf_prototype.wave_amplitude.uniformDistribution(0.02, 0.04);
    leaf_prototype.subdivisions = 20;
    ;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_sugarbeet(context_ptr->getRandomGenerator());

    phytomer_parameters_sugarbeet.internode.pitch = 0;
    phytomer_parameters_sugarbeet.internode.phyllotactic_angle = 137.5;
    phytomer_parameters_sugarbeet.internode.radius_initial = 0.005;
    phytomer_parameters_sugarbeet.internode.color = make_RGBcolor(0.44, 0.58, 0.19);
    phytomer_parameters_sugarbeet.internode.length_segments = 1;
    phytomer_parameters_sugarbeet.internode.max_vegetative_buds_per_petiole = 0;
    phytomer_parameters_sugarbeet.internode.max_floral_buds_per_petiole = 0;

    phytomer_parameters_sugarbeet.petiole.petioles_per_internode = 1;
    phytomer_parameters_sugarbeet.petiole.pitch.uniformDistribution(0, 40);
    phytomer_parameters_sugarbeet.petiole.radius = 0.005;
    phytomer_parameters_sugarbeet.petiole.length.uniformDistribution(0.15, 0.2);
    phytomer_parameters_sugarbeet.petiole.taper = 0.6;
    phytomer_parameters_sugarbeet.petiole.curvature.uniformDistribution(-300, 100);
    phytomer_parameters_sugarbeet.petiole.color = phytomer_parameters_sugarbeet.internode.color;
    phytomer_parameters_sugarbeet.petiole.length_segments = 8;

    phytomer_parameters_sugarbeet.leaf.leaves_per_petiole = 1;
    phytomer_parameters_sugarbeet.leaf.pitch.uniformDistribution(-10, 0);
    phytomer_parameters_sugarbeet.leaf.yaw.uniformDistribution(-5, 5);
    phytomer_parameters_sugarbeet.leaf.roll.uniformDistribution(-15, 15);
    phytomer_parameters_sugarbeet.leaf.prototype_scale.uniformDistribution(0.15, 0.25);
    phytomer_parameters_sugarbeet.leaf.prototype = leaf_prototype;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_sugarbeet;
    shoot_parameters_mainstem.vegetative_bud_break_probability_min = 0;
    shoot_parameters_mainstem.phyllochron_min = 2;
    shoot_parameters_mainstem.elongation_rate_max = 0.1;
    shoot_parameters_mainstem.girth_area_factor = 20.f;
    shoot_parameters_mainstem.gravitropic_curvature = 10;
    shoot_parameters_mainstem.internode_length_max = 0.001;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 0.0;
    shoot_parameters_mainstem.max_nodes = 30;

    defineShootType("mainstem", shoot_parameters_mainstem);
}

uint PlantArchitecture::buildSugarbeetPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize sugarbeet plant shoots
        initializeSugarbeetShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(context_ptr->randu(0.f, 0.01f * M_PI), 0.f * context_ptr->randu(0.f, 2.f * M_PI), 0.25f * M_PI), 0.005, 0.001, 1, 1, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeTomatoShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    // The blade-only texture drops the stalk painted onto TomatoLeaf_centered.png, keeping 0.9031 of its width, so the aspect ratio here and the leaflet scale
    // below are divided and multiplied by that fraction to leave the blade the same size. The stalk is built as geometry instead.
    leaf_prototype.leaf_texture_file[0] = "TomatoLeaf_blade.png";
    leaf_prototype.leaf_aspect_ratio = 0.5f / 0.9031f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.45, -0.2f);
    leaf_prototype.lateral_curvature = -0.3f;
    leaf_prototype.wave_period = 0.35f;
    leaf_prototype.wave_amplitude = 0.08f;
    leaf_prototype.subdivisions = 6;
    leaf_prototype.unique_prototypes = 5;
    leaf_prototype.build_petiolule = true;
    // Petiolule length as a fraction of blade length: about 9-14 mm on the terminal leaflet and 3-10 mm on the laterals, within the 5-15 mm (terminal) and
    // 3-20 mm (lateral) given by Flora of North America for Solanum lycopersicum. Its radius follows the petiole.
    leaf_prototype.petiolule_length = {{0, 0.085f}};

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    // Tomato leaves are alternate and spirally arranged, at close to the golden angle; the previous 140-220 deg was centred on a two-ranked arrangement.
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5 - 40, 137.5 + 40);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.2495, 0.3162, 0.0657);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45, 60);
    phytomer_parameters.petiole.radius = 0.002;
    phytomer_parameters.petiole.length = 0.2;
    // The rachis thins to 30% of its base radius at the tip, so the leaflet stalks, which take the rachis radius where they attach, are about 1.2-2.3 mm thick.
    phytomer_parameters.petiole.taper = 0.7;
    phytomer_parameters.petiole.curvature.uniformDistribution(-150, -50);
    phytomer_parameters.petiole.color = make_RGBcolor(0.3, 0.37, 0.0657);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 7;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.leaflet_offset = 0.15;
    phytomer_parameters.leaf.leaflet_scale = 0.7;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12f * 0.9031f, 0.18f * 0.9031f);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.18;
    phytomer_parameters.peduncle.radius = 0.0015;
    phytomer_parameters.peduncle.pitch = 20;
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -900;
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 5;
    phytomer_parameters.peduncle.radial_subdivisions = 8;

    phytomer_parameters.inflorescence.flowers_per_peduncle = 6;
    phytomer_parameters.inflorescence.flower_offset = 0.15;
    phytomer_parameters.inflorescence.pitch = 80;
    phytomer_parameters.inflorescence.roll = 180;
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.05;
    phytomer_parameters.inflorescence.flower_prototype_function = TomatoFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.15;
    phytomer_parameters.inflorescence.fruit_prototype_function = TomatoFruitPrototype;
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction = 0.;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.phytomer_parameters.phytomer_creation_function = TomatoPhytomerCreationFunction;

    shoot_parameters.max_nodes = 16;
    shoot_parameters.insertion_angle_tip = 30;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.04;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature = 150;
    shoot_parameters.tortuosity = 75;

    shoot_parameters.phyllochron_min = 2;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 2.f;
    shoot_parameters.vegetative_bud_break_time = 30;
    shoot_parameters.vegetative_bud_break_probability_min = 0.25;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = -0.25;
    shoot_parameters.flower_bud_break_probability = 0.2;
    shoot_parameters.fruit_set_probability = 0.8;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = true;

    shoot_parameters.defineChildShootTypes({"mainstem"}, {1.0});

    defineShootType("mainstem", shoot_parameters);
}

uint PlantArchitecture::buildTomatoPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize tomato plant shoots
        initializeTomatoShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.002, 0.04, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}

void PlantArchitecture::initializeCherryTomatoShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    // See initializeTomatoShoots(): the blade-only texture keeps 0.9031 of the original's width, so the aspect ratio and leaflet scale are compensated for it.
    leaf_prototype.leaf_texture_file[0] = "TomatoLeaf_blade.png";
    leaf_prototype.leaf_aspect_ratio = 0.6f / 0.9031f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.3, -0.15f);
    leaf_prototype.lateral_curvature = -0.8f;
    leaf_prototype.wave_period = 0.35f;
    leaf_prototype.wave_amplitude = 0.08f;
    leaf_prototype.subdivisions = 7;
    leaf_prototype.unique_prototypes = 5;
    leaf_prototype.build_petiolule = true;
    // Petiolule length as a fraction of blade length: about 8-11 mm on the terminal leaflet and 5-10 mm on the laterals, within the ranges Flora of North America
    // gives for Solanum lycopersicum (see initializeTomatoShoots()). Its radius follows the petiole.
    leaf_prototype.petiolule_length = {{0, 0.07f}};

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 5;
    // Spiral, at close to the golden angle; see initializeTomatoShoots().
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5 - 40, 137.5 + 40);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.2495, 0.3162, 0.0657);
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.internode.radial_subdivisions = 14;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45, 60);
    phytomer_parameters.petiole.radius = 0.0025;
    phytomer_parameters.petiole.length = 0.25;
    // The rachis thins to 30% of its base radius at the tip; see initializeTomatoShoots().
    phytomer_parameters.petiole.taper = 0.7;
    phytomer_parameters.petiole.curvature.uniformDistribution(-250, 0);
    phytomer_parameters.petiole.color = make_RGBcolor(0.32, 0.37, 0.12);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 9;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll.uniformDistribution(-20, 20);
    phytomer_parameters.leaf.leaflet_offset = 0.22;
    phytomer_parameters.leaf.leaflet_scale = 0.9;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12f * 0.9031f, 0.17f * 0.9031f);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.2;
    phytomer_parameters.peduncle.radius = 0.0015;
    phytomer_parameters.peduncle.pitch = 20;
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -1000;
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 5;
    phytomer_parameters.peduncle.radial_subdivisions = 8;

    phytomer_parameters.inflorescence.flowers_per_peduncle = 6;
    phytomer_parameters.inflorescence.flower_offset = 0.15;
    phytomer_parameters.inflorescence.pitch = 80;
    phytomer_parameters.inflorescence.roll = 180;
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.05;
    phytomer_parameters.inflorescence.flower_prototype_function = TomatoFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.1;
    phytomer_parameters.inflorescence.fruit_prototype_function = TomatoFruitPrototype;
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction = 0.2;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.phytomer_parameters.phytomer_creation_function = CherryTomatoPhytomerCreationFunction;
    shoot_parameters.phytomer_parameters.phytomer_callback_function = CherryTomatoPhytomerCallbackFunction;

    shoot_parameters.max_nodes = 100;
    shoot_parameters.insertion_angle_tip = 30;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.04;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature = 800;
    shoot_parameters.tortuosity = 37.5;

    shoot_parameters.phyllochron_min = 4;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 2.f;
    shoot_parameters.vegetative_bud_break_time = 40;
    shoot_parameters.vegetative_bud_break_probability_min = 0.2;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = 0.;
    shoot_parameters.flower_bud_break_probability = 0.5;
    shoot_parameters.fruit_set_probability = 0.9;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = false;

    shoot_parameters.defineChildShootTypes({"mainstem"}, {1.0});

    defineShootType("mainstem", shoot_parameters);
}

uint PlantArchitecture::buildCherryTomatoPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize cherry tomato plant shoots
        initializeCherryTomatoShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.002, 0.06, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 50, 10, 5, 30, 1000);

    plant_instances.at(plantID).max_age = 175;

    return plantID;
}

void PlantArchitecture::initializeWalnutTreeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "WalnutLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.5f;
    leaf_prototype.midrib_fold_fraction = 0.15f;
    leaf_prototype.longitudinal_curvature = -0.2f;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.wave_period.uniformDistribution(0.08, 0.15);
    leaf_prototype.wave_amplitude.uniformDistribution(0.02, 0.04);
    leaf_prototype.subdivisions = 3;
    leaf_prototype.unique_prototypes = 5;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_walnut(context_ptr->getRandomGenerator());

    phytomer_parameters_walnut.internode.pitch = 0;
    // Walnut leaves are alternate and spirally arranged, at close to the golden angle; the previous 160-200 deg made the branching nearly two-ranked.
    phytomer_parameters_walnut.internode.phyllotactic_angle.uniformDistribution(137.5 - 20, 137.5 + 20);
    phytomer_parameters_walnut.internode.radius_initial = 0.004;
    phytomer_parameters_walnut.internode.length_segments = 1;
    phytomer_parameters_walnut.internode.image_texture = "AppleBark.jpg";
    phytomer_parameters_walnut.internode.max_floral_buds_per_petiole = 3;

    phytomer_parameters_walnut.petiole.petioles_per_internode = 1;
    phytomer_parameters_walnut.petiole.pitch.uniformDistribution(-80, -70);
    phytomer_parameters_walnut.petiole.taper = 0.2;
    phytomer_parameters_walnut.petiole.curvature.uniformDistribution(-1000, 0);
    phytomer_parameters_walnut.petiole.length = 0.15;
    phytomer_parameters_walnut.petiole.radius = 0.0015;
    phytomer_parameters_walnut.petiole.length_segments = 5;
    phytomer_parameters_walnut.petiole.radial_subdivisions = 3;
    phytomer_parameters_walnut.petiole.color = make_RGBcolor(0.61, 0.5, 0.24);

    phytomer_parameters_walnut.leaf.leaves_per_petiole = 5;
    phytomer_parameters_walnut.leaf.pitch.uniformDistribution(-40, 0);
    phytomer_parameters_walnut.leaf.prototype_scale = 0.14;
    phytomer_parameters_walnut.leaf.leaflet_scale = 0.7;
    phytomer_parameters_walnut.leaf.leaflet_offset = 0.35;
    phytomer_parameters_walnut.leaf.prototype = leaf_prototype;

    phytomer_parameters_walnut.peduncle.length = 0.02;
    phytomer_parameters_walnut.peduncle.radius = 0.0005;
    phytomer_parameters_walnut.peduncle.pitch = 90;
    phytomer_parameters_walnut.peduncle.roll = 90;
    phytomer_parameters_walnut.peduncle.length_segments = 1;

    phytomer_parameters_walnut.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_walnut.inflorescence.pitch = 0;
    phytomer_parameters_walnut.inflorescence.roll = 0;
    phytomer_parameters_walnut.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_walnut.inflorescence.flower_prototype_function = WalnutFlowerPrototype;
    phytomer_parameters_walnut.inflorescence.fruit_prototype_scale = 0.075;
    phytomer_parameters_walnut.inflorescence.fruit_prototype_function = WalnutFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_walnut;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.01;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 24;
    shoot_parameters_trunk.max_nodes = 20;
    // Rescaled when the per-phytomer 365/age decay was removed from incrementPhytomerInternodeGirth(), to keep base diameters at
    // max_age where they were: girth is now proportional to cumulative downstream leaf area, which runs well above the old value.
    shoot_parameters_trunk.girth_area_factor = 1.4f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 20;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_walnut;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = WalnutPhytomerCreationFunction;
    // shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = WalnutPhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes = 24;
    shoot_parameters_proleptic.max_nodes_per_season = 12;
    shoot_parameters_proleptic.phyllochron_min = 2.;
    shoot_parameters_proleptic.elongation_rate_max = 0.15;
    shoot_parameters_proleptic.girth_area_factor = 6.7f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.05;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.7;
    shoot_parameters_proleptic.vegetative_bud_break_time = 3;
    shoot_parameters_proleptic.gravitropic_curvature = 300;
    shoot_parameters_proleptic.tortuosity = 50;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(20, 25);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 15;
    shoot_parameters_proleptic.internode_length_max = 0.08;
    shoot_parameters_proleptic.internode_length_min = 0.01;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.006;
    shoot_parameters_proleptic.fruit_set_probability = 0.3;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 4;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"}, {1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    // Scaffolds are older wood than proleptic shoots, so they need their own factor to hold their base diameter.
    shoot_parameters_scaffold.girth_area_factor = 2.6f;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 30;
    shoot_parameters_scaffold.gravitropic_curvature = 300;
    shoot_parameters_scaffold.internode_length_max = 0.06;
    shoot_parameters_scaffold.tortuosity = 66.7;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
}

uint PlantArchitecture::buildWalnutTree(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize walnut tree shoots
        initializeWalnutTreeShoots();
    }

    // Get training system parameters (with defaults matching original hard-coded values)
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 0.8f, 0.1f, 3.f, "total trunk height in meters");
    auto num_scaffolds = uint(getParameterValue(current_build_parameters, "num_scaffolds", 4.f, 2.f, 8.f, "number of scaffold branches"));
    auto scaffold_angle = getParameterValue(current_build_parameters, "scaffold_angle", 40.f, 25.f, 60.f, "scaffold branch angle in degrees");

    // Calculate trunk nodes based on desired height and internode length
    float trunk_internode_length = 0.04f; // Default internode length for walnut
    uint trunk_nodes = uint(trunk_height / trunk_internode_length);
    if (trunk_nodes < 1)
        trunk_nodes = 1;

    // Fixed training parameters (not user-customizable)
    float scaffold_radius = 0.007f;
    float scaffold_length = 0.06f;

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), shoot_types.at("trunk").phytomer_parameters.internode.radius_initial.val(),
                                      trunk_internode_length, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot(plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for (const auto &phytomer: phytomers) {
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // Hard-coded scaffold node range (not user-customizable)
    uint scaffold_nodes_min = 5;
    uint scaffold_nodes_max = 7;

    for (int i = 0; i < num_scaffolds; i++) {
        float pitch = deg2rad(scaffold_angle) + context_ptr->randu(-0.1f, 0.1f); // Small randomness around specified angle
        uint scaffold_nodes = context_ptr->randu(int(scaffold_nodes_min), int(scaffold_nodes_max));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, scaffold_nodes, make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(num_scaffolds) * 2 * M_PI, 0), scaffold_radius,
                                       scaffold_length, 1.f, 1.f, 0.5, "scaffold", 0);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200);

    plant_instances.at(plantID).max_age = 1460;

    return plantID;
}

void PlantArchitecture::initializeWheatShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.1f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.07, -0.04);
    leaf_prototype.longitudinal_curvature_exponent = 3.f;
    leaf_prototype.flexibility_taper = 150.f;
    leaf_prototype.flexibility_aging = 11.f;
    leaf_prototype.flexibility_aging_max = 5.f;
    leaf_prototype.lateral_curvature = -0.3;
    leaf_prototype.petiole_roll = 0.f;
    leaf_prototype.wave_period = 0.1f;
    leaf_prototype.wave_amplitude = 0.04f;
    leaf_prototype.flexibility.uniformDistribution(0.8, 1.2);
    leaf_prototype.subdivisions = 20;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_wheat(context_ptr->getRandomGenerator());

    phytomer_parameters_wheat.internode.pitch = 0;
    // Wheat is distichous, as grasses are: successive leaves stand on opposite sides of the culm.
    phytomer_parameters_wheat.internode.phyllotactic_angle.uniformDistribution(175, 185);
    phytomer_parameters_wheat.internode.radius_initial = 0.001;
    phytomer_parameters_wheat.internode.color = make_RGBcolor(0.27, 0.31, 0.16);
    phytomer_parameters_wheat.internode.length_segments = 1;
    phytomer_parameters_wheat.internode.radial_subdivisions = 6;
    phytomer_parameters_wheat.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_wheat.internode.max_vegetative_buds_per_petiole = 0;

    phytomer_parameters_wheat.petiole.petioles_per_internode = 1;
    phytomer_parameters_wheat.petiole.pitch.uniformDistribution(-40, -20);
    phytomer_parameters_wheat.petiole.radius = 0.0;
    phytomer_parameters_wheat.petiole.length = 0.005;
    phytomer_parameters_wheat.petiole.taper = 0;
    phytomer_parameters_wheat.petiole.curvature = 0;
    phytomer_parameters_wheat.petiole.length_segments = 1;

    phytomer_parameters_wheat.leaf.leaves_per_petiole = 1;
    phytomer_parameters_wheat.leaf.pitch = 0;
    phytomer_parameters_wheat.leaf.yaw = 0;
    phytomer_parameters_wheat.leaf.roll = 0;
    phytomer_parameters_wheat.leaf.prototype_scale = 0.36;
    phytomer_parameters_wheat.leaf.prototype = leaf_prototype;

    phytomer_parameters_wheat.peduncle.length = 0.1;
    phytomer_parameters_wheat.peduncle.radius = 0.002;
    phytomer_parameters_wheat.peduncle.color = phytomer_parameters_wheat.internode.color;
    phytomer_parameters_wheat.peduncle.curvature = 0;
    phytomer_parameters_wheat.peduncle.radial_subdivisions = 6;

    phytomer_parameters_wheat.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_wheat.inflorescence.pitch = 0;
    phytomer_parameters_wheat.inflorescence.roll = 0;
    phytomer_parameters_wheat.inflorescence.fruit_prototype_scale = 0.1;
    phytomer_parameters_wheat.inflorescence.fruit_prototype_function = WheatSpikePrototype;

    phytomer_parameters_wheat.phytomer_creation_function = WheatPhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_wheat;
    shoot_parameters_mainstem.vegetative_bud_break_probability_min = 0;
    shoot_parameters_mainstem.flower_bud_break_probability = 1;
    shoot_parameters_mainstem.phyllochron_min = 2;
    shoot_parameters_mainstem.elongation_rate_max = 0.1;
    shoot_parameters_mainstem.girth_area_factor = 6.f;
    shoot_parameters_mainstem.internode_length_max = 0.11;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-500, -200);
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.fruit_set_probability = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"}, {1.0});
    shoot_parameters_mainstem.max_nodes = 8;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem", shoot_parameters_mainstem);
}

uint PlantArchitecture::buildWheatPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize wheat plant shoots
        initializeWheatShoots();
    }

    uint plantID = addPlantInstance(base_position - make_vec3(0, 0, 0.025), 0);

    // The internode length is taken from the shoot type rather than fixed here, so the eight-node culm reaches the height of a real wheat plant; passing a literal overrode it and left the plant knee-high.
    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.001,
                                     shoot_types.at("mainstem").internode_length_max.val(), 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 10, 1000);

    plant_instances.at(plantID).max_age = 365;

    return plantID;
}
