/** \file "PlantLibrary.cpp" Contains routines for loading and building plant models from a library of predefined plant types.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "CollisionDetection.h"
#include "PlantArchitecture.h"

using namespace helios;

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

    registerPlantModel("almond_wood_colony", [this]() { initializeAlmondTreeWoodColonyShoots(); }, [this](const helios::vec3 &pos) { return buildAlmondTreeWoodColony(pos); }, "tree");

    registerPlantModel("apple", [this]() { initializeAppleTreeShoots(); }, [this](const helios::vec3 &pos) { return buildAppleTree(pos); }, "tree");

    registerPlantModel("apple_fruitingwall", [this]() { initializeAppleFruitingWallShoots(); }, [this](const helios::vec3 &pos) { return buildAppleFruitingWall(pos); }, "tree");

    registerPlantModel("asparagus", [this]() { initializeAsparagusShoots(); }, [this](const helios::vec3 &pos) { return buildAsparagusPlant(pos); });

    registerPlantModel("bindweed", [this]() { initializeBindweedShoots(); }, [this](const helios::vec3 &pos) { return buildBindweedPlant(pos); }, "weed");

    registerPlantModel("bean", [this]() { initializeBeanShoots(); }, [this](const helios::vec3 &pos) { return buildBeanPlant(pos); });

    registerPlantModel("bougainvillea", [this]() { initializeBougainvilleaShoots(); }, [this](const helios::vec3 &pos) { return buildBougainvilleaPlant(pos); });

    registerPlantModel("capsicum", [this]() { initializeCapsicumShoots(); }, [this](const helios::vec3 &pos) { return buildCapsicumPlant(pos); });

    registerPlantModel("cheeseweed", [this]() { initializeCheeseweedShoots(); }, [this](const helios::vec3 &pos) { return buildCheeseweedPlant(pos); }, "weed");

    registerPlantModel("cowpea", [this]() { initializeCowpeaShoots(); }, [this](const helios::vec3 &pos) { return buildCowpeaPlant(pos); });

    registerPlantModel("grapevine_VSP", [this]() { initializeGrapevineVSPShoots(); }, [this](const helios::vec3 &pos) { return buildGrapevineVSP(pos); });

    registerPlantModel("grapevine_Wye", [this]() { initializeGrapevineWyeShoots(); }, [this](const helios::vec3 &pos) { return buildGrapevineWye(pos); });

    registerPlantModel("groundcherryweed", [this]() { initializeGroundCherryWeedShoots(); }, [this](const helios::vec3 &pos) { return buildGroundCherryWeedPlant(pos); }, "weed");

    registerPlantModel("maize", [this]() { initializeMaizeShoots(); }, [this](const helios::vec3 &pos) { return buildMaizePlant(pos); });

    registerPlantModel("olive", [this]() { initializeOliveTreeShoots(); }, [this](const helios::vec3 &pos) { return buildOliveTree(pos); }, "tree");

    registerPlantModel("pistachio", [this]() { initializePistachioTreeShoots(); }, [this](const helios::vec3 &pos) { return buildPistachioTree(pos); }, "tree");

    registerPlantModel("puncturevine", [this]() { initializePuncturevineShoots(); }, [this](const helios::vec3 &pos) { return buildPuncturevinePlant(pos); }, "weed");

    registerPlantModel("easternredbud", [this]() { initializeEasternRedbudShoots(); }, [this](const helios::vec3 &pos) { return buildEasternRedbudPlant(pos); }, "tree");

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

void PlantArchitecture::registerPlantModel(const std::string &name, std::function<void()> shoot_init, std::function<uint(const helios::vec3 &)> plant_build, const std::string &plant_type) {
    shoot_initializers[name] = shoot_init;
    plant_builders[name] = plant_build;
    plant_type_map[name] = plant_type;
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
    shoot_parameters_trunk.max_nodes = 20;
    shoot_parameters_trunk.girth_area_factor = 10.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 1;
    shoot_parameters_trunk.internode_length_max = 0.04;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2);
    shoot_parameters_proleptic.phytomer_parameters.internode.radial_subdivisions = 5;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = AlmondPhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes = 60;
    shoot_parameters_proleptic.max_nodes_per_season = 20;
    shoot_parameters_proleptic.phyllochron_min = 1;
    shoot_parameters_proleptic.elongation_rate_max = 0.3;
    shoot_parameters_proleptic.girth_area_factor = 6.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.5;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 250;
    shoot_parameters_proleptic.tortuosity = 3.5;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(25, 30);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 15;
    shoot_parameters_proleptic.internode_length_max = 0.03;
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
    shoot_parameters_sylleptic.phytomer_parameters.leaf.prototype_scale = 0.14;
    shoot_parameters_sylleptic.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20);
    shoot_parameters_sylleptic.insertion_angle_tip = 0;
    shoot_parameters_sylleptic.insertion_angle_decay_rate = 0;
    shoot_parameters_sylleptic.phyllochron_min = 1;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_min = 0.0;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_max = 0.7;
    shoot_parameters_sylleptic.gravitropic_curvature = 600;
    shoot_parameters_sylleptic.internode_length_max = 0.02;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true;
    shoot_parameters_sylleptic.defineChildShootTypes({"proleptic"}, {1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    //    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 40;
    shoot_parameters_scaffold.gravitropic_curvature = 150;
    // shoot_parameters_scaffold.internode_length_max = 0.02;
    shoot_parameters_scaffold.tortuosity = 1.;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);
}

uint PlantArchitecture::buildAlmondTree(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize almond tree shoots
        initializeAlmondTreeShoots();
    }

    // Get training system parameters (with defaults matching original hard-coded values)
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 0.6f, 0.1f, 3.f, "total trunk height in meters");
    auto num_scaffolds = uint(getParameterValue(current_build_parameters, "num_scaffolds", 4.f, 2.f, 8.f, "number of scaffold branches"));
    auto scaffold_angle = getParameterValue(current_build_parameters, "scaffold_angle", 40.f, 20.f, 70.f, "scaffold branch angle in degrees");

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

    uint uID_trunk = addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), trunk_radius, trunk_internode_length, 1.f, 1.f, 0, "trunk");
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
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 90, -1, 3, 7, 20, 275);
    plant_instances.at(plantID).max_age = 1825;

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
    shoot_parameters_trunk.max_nodes = 20;
    shoot_parameters_trunk.girth_area_factor = 8.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = .5;
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
    shoot_parameters_proleptic.max_nodes = 25;
    shoot_parameters_proleptic.max_nodes_per_season = 15;
    shoot_parameters_proleptic.phyllochron_min = 1;
    shoot_parameters_proleptic.elongation_rate_max = 0.3;
    shoot_parameters_proleptic.girth_area_factor = 8.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.5;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 450;
    shoot_parameters_proleptic.tortuosity = 2.5;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(5, 30);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 5;
    shoot_parameters_proleptic.internode_length_max = 0.025;
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
    //    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 20;
    shoot_parameters_scaffold.gravitropic_curvature = 200;
    shoot_parameters_scaffold.internode_length_max = 0.02;
    shoot_parameters_scaffold.tortuosity = 0.5;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"}, {1.0});

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

    uint uID_trunk = addBaseStemShoot(plantID, 19, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), 0.015, 0.03, 1.f, 1.f, 0, "trunk");
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
        float pitch = context_ptr->randu(deg2rad(5), deg2rad(35));
        uint uID_shoot = addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - i - 1, context_ptr->randu(7, 9), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007, 0.06,
                                       1.f, 1.f, 0.5, "scaffold", 0);
    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 90, -1, 3, 7, 20, 275);
    plant_instances.at(plantID).max_age = 1825;

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
    shoot_parameters_trunk.girth_area_factor = 10.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 1;
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
    shoot_parameters_proleptic.girth_area_factor = 8.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.5;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 150;
    shoot_parameters_proleptic.tortuosity = 3;
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
    //    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 15;
    shoot_parameters_scaffold.gravitropic_curvature = 100;
    shoot_parameters_scaffold.internode_length_max = 0.02;
    shoot_parameters_scaffold.tortuosity = 1.;
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

    uint uID_trunk = addBaseStemShoot(plantID, 14, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), 0.015, 0.03, 1.f, 1.f, 0, "trunk");
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
    shoot_parameters_trunk.girth_area_factor = 5.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 1;
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
    shoot_parameters_proleptic.girth_area_factor = 7.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.4;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(450, 500);
    shoot_parameters_proleptic.tortuosity = 3;
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
    shoot_parameters_trunk.girth_area_factor = 5.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 0.5;
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
    shoot_parameters_proleptic.girth_area_factor = 7.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.4;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(550, 700);
    shoot_parameters_proleptic.tortuosity = 3;
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
    leaf_prototype_trifoliate.leaf_texture_file[-1] = "BeanLeaf_left_centered.png";
    leaf_prototype_trifoliate.leaf_texture_file[1] = "BeanLeaf_right_centered.png";
    leaf_prototype_trifoliate.leaf_aspect_ratio = 1.f;
    leaf_prototype_trifoliate.midrib_fold_fraction = 0.2;
    leaf_prototype_trifoliate.longitudinal_curvature.uniformDistribution(-0.3f, -0.2f);
    leaf_prototype_trifoliate.lateral_curvature = -1.f;
    leaf_prototype_trifoliate.subdivisions = 6;
    leaf_prototype_trifoliate.unique_prototypes = 5;
    leaf_prototype_trifoliate.build_petiolule = true;

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
    shoot_parameters.tortuosity = 3;

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
    shoot_parameters.tortuosity = 3;

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
    leaf_prototype_trifoliate.leaf_texture_file[-1] = "CowpeaLeaf_left_centered.png";
    leaf_prototype_trifoliate.leaf_texture_file[1] = "CowpeaLeaf_right_centered.png";
    leaf_prototype_trifoliate.leaf_aspect_ratio = 0.7f;
    leaf_prototype_trifoliate.midrib_fold_fraction = 0.2;
    leaf_prototype_trifoliate.longitudinal_curvature.uniformDistribution(-0.3f, -0.1f);
    leaf_prototype_trifoliate.lateral_curvature = -0.4f;
    leaf_prototype_trifoliate.subdivisions = 6;
    leaf_prototype_trifoliate.unique_prototypes = 5;
    leaf_prototype_trifoliate.build_petiolule = true;

    LeafPrototype leaf_prototype_unifoliate = leaf_prototype_trifoliate;
    leaf_prototype_unifoliate.leaf_texture_file.clear();
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
    phytomer_parameters_grapevine.petiole.color = make_RGBcolor(0.13, 0.066, 0.03);
    phytomer_parameters_grapevine.petiole.pitch.uniformDistribution(45, 70);
    phytomer_parameters_grapevine.petiole.radius = 0.0025;
    phytomer_parameters_grapevine.petiole.length = 0.1;
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
    phytomer_parameters_grapevine.peduncle.color = make_RGBcolor(0.32, 0.05, 0.13);

    phytomer_parameters_grapevine.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_grapevine.inflorescence.pitch = 0;
    //    phytomer_parameters_grapevine.inflorescence.flower_prototype_function = GrapevineFlowerPrototype;
    phytomer_parameters_grapevine.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_function = GrapevineFruitPrototype;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_gravity_factor_fraction = 0.7;

    phytomer_parameters_grapevine.phytomer_creation_function = GrapevinePhytomerCreationFunction;
    //    phytomer_parameters_grapevine.phytomer_callback_function = GrapevinePhytomerCallbackFunction;

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
    shoot_parameters_main.tortuosity = 12;
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
    shoot_parameters_main.max_nodes = 17;
    // Commercial VSP is hedged at the top wire, roughly 1 m above the cordon, so a shoot does not run to
    // the 24 nodes it would reach unhedged. There is no hedging operation in the model, so the cap stands
    // in for one. The count also has to leave room for the summer laterals, which carry a further third
    // of the vine's leaf area on top of what the primary axis bears.
    shoot_parameters_main.max_nodes = 17;
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

    // The trunk bears no shoots, so its nodes are only there to carry its shape and are spaced far more
    // closely than buds would be. The floor keeps enough of them to seat both canes on a very short trunk.
    const uint trunk_nodes = std::max(uint(6), uint(std::ceil(head_height / 0.04f)));
    const float trunk_node_spacing = head_height / float(trunk_nodes);

    // Sway of the trunk axis across (x) and along (y) the row: a slow bend that gets through about half a
    // cycle over the height of the trunk, and a slight kink or two on top of it. A trunk is trained up a
    // stake, so it is close to straight; a slow bend any tighter than this reads as a snake.
    BoundedWander trunk_sway_x;
    BoundedWander trunk_sway_y;
    for (BoundedWander *trunk_sway: {&trunk_sway_x, &trunk_sway_y}) {
        trunk_sway->addComponent(context_ptr->randu(0.015f, 0.04f), context_ptr->randu(1.2f, 2.2f), context_ptr->randu(0.f, 2.f * PI_F));
        trunk_sway->addComponent(context_ptr->randu(0.002f, 0.006f), context_ptr->randu(0.3f, 0.6f), context_ptr->randu(0.f, 2.f * PI_F));
    }
    // The vine is planted where it is planted, so the sway is measured from the base. Across the row the
    // head is tied in close to the wire, and the lean is whatever gets it there; along the row nothing
    // holds it, and trunks commonly lean a few centimetres toward one neighbour.
    const float head_offset_x = context_ptr->randu(-0.015f, 0.015f);
    const float trunk_lean_x = (head_offset_x - (trunk_sway_x.evaluate(head_height) - trunk_sway_x.evaluate(0.f))) / head_height;
    const float trunk_lean_y = context_ptr->randu(-0.06f, 0.06f);

    // Girth: a flare into the root collar, a slight taper up the trunk, a swollen head, and shallow lumps
    // from old pruning wounds.
    const float trunk_radius = shoot_types.at("grapevine_trunk").phytomer_parameters.internode.radius_initial.val() * context_ptr->randu(0.85f, 1.15f);
    const float head_swelling = context_ptr->randu(0.3f, 0.6f);
    BoundedWander trunk_lumps;
    trunk_lumps.addComponent(context_ptr->randu(0.03f, 0.06f), context_ptr->randu(0.12f, 0.3f), context_ptr->randu(0.f, 2.f * PI_F));

    // The tube is open-ended, so the head is closed over with a dome of three more nodes whose radius
    // falls away as a quarter circle. They are appended after the loop below.
    const std::vector<float> dome_angles_degrees = {30.f, 60.f, 85.f};

    std::vector<vec3> trunk_node_positions;
    std::vector<float> trunk_node_radii;
    trunk_node_positions.reserve(trunk_nodes + 1 + dome_angles_degrees.size());
    trunk_node_radii.reserve(trunk_nodes + 1 + dome_angles_degrees.size());
    for (uint node = 0; node <= trunk_nodes + dome_angles_degrees.size(); node++) {
        float height = float(node) * trunk_node_spacing;
        float dome_radius_fraction = 1.f;
        if (node > trunk_nodes) {
            const float dome_angle = deg2rad(dome_angles_degrees.at(node - trunk_nodes - 1));
            height = head_height + trunk_node_radii.at(trunk_nodes) * std::sin(dome_angle);
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
        trunk_node_positions.push_back(base_position + make_vec3(offset_x, offset_y, height));
        trunk_node_radii.push_back(radius * dome_radius_fraction);
    }
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
    phytomer_parameters_grapevine.petiole.color = make_RGBcolor(0.13, 0.125, 0.03);
    phytomer_parameters_grapevine.petiole.pitch.uniformDistribution(45, 70);
    phytomer_parameters_grapevine.petiole.radius = 0.0025;
    phytomer_parameters_grapevine.petiole.length = 0.1;
    phytomer_parameters_grapevine.petiole.taper = 0;
    phytomer_parameters_grapevine.petiole.curvature = 0;
    phytomer_parameters_grapevine.petiole.length_segments = 1;

    phytomer_parameters_grapevine.leaf.leaves_per_petiole = 1;
    phytomer_parameters_grapevine.leaf.pitch.uniformDistribution(-110, -80);
    phytomer_parameters_grapevine.leaf.yaw.uniformDistribution(-20, 20);
    phytomer_parameters_grapevine.leaf.roll.uniformDistribution(-5, 5);
    // Individual blade area is a property of the vine rather than of the trellis it is trained to, so
    // this matches the VSP model: a mean of ~124 cm^2 (Navarrete 2015; Kliewer & Dokoozlian 2005). The
    // Wye shoot architecture is left as it was -- its lateral regime differs from VSP and no
    // divided-canopy calibration was available when this was set.
    phytomer_parameters_grapevine.leaf.prototype_scale = 0.14;
    phytomer_parameters_grapevine.leaf.prototype = leaf_prototype;

    phytomer_parameters_grapevine.peduncle.length = 0.04;
    phytomer_parameters_grapevine.peduncle.radius = 0.005;
    phytomer_parameters_grapevine.peduncle.color = make_RGBcolor(0.13, 0.125, 0.03);
    phytomer_parameters_grapevine.peduncle.pitch.uniformDistribution(80, 100);

    phytomer_parameters_grapevine.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters_grapevine.inflorescence.pitch = 0;
    //    phytomer_parameters_grapevine.inflorescence.flower_prototype_function = GrapevineFlowerPrototype;
    phytomer_parameters_grapevine.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_function = GrapevineFruitPrototype;
    phytomer_parameters_grapevine.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters_grapevine.inflorescence.fruit_gravity_factor_fraction = 0.9;

    phytomer_parameters_grapevine.phytomer_creation_function = GrapevinePhytomerCreationFunction;
    //    phytomer_parameters_grapevine.phytomer_callback_function = GrapevinePhytomerCallbackFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_main(context_ptr->getRandomGenerator());
    shoot_parameters_main.phytomer_parameters = phytomer_parameters_grapevine;
    shoot_parameters_main.vegetative_bud_break_probability_min = 0.025;
    shoot_parameters_main.vegetative_bud_break_probability_decay_rate = -1.;
    shoot_parameters_main.vegetative_bud_break_time = 30;
    shoot_parameters_main.phyllochron_min.uniformDistribution(2.5, 3.5);
    shoot_parameters_main.elongation_rate_max = 0.15;
    shoot_parameters_main.girth_area_factor = 0.8f;
    shoot_parameters_main.gravitropic_curvature.uniformDistribution(0, 100);
    shoot_parameters_main.tortuosity = 10;
    shoot_parameters_main.internode_length_max.uniformDistribution(0.06, 0.08);
    shoot_parameters_main.internode_length_decay_rate = 0;
    shoot_parameters_main.insertion_angle_tip = 45;
    shoot_parameters_main.insertion_angle_decay_rate = 0;
    shoot_parameters_main.flowers_require_dormancy = false;
    shoot_parameters_main.growth_requires_dormancy = false;
    shoot_parameters_main.determinate_shoot_growth = false;
    shoot_parameters_main.max_terminal_floral_buds = 0;
    shoot_parameters_main.flower_bud_break_probability = 0.5;
    shoot_parameters_main.fruit_set_probability = 0.2;
    shoot_parameters_main.max_nodes.uniformDistribution(14, 18);
    shoot_parameters_main.base_roll.uniformDistribution(90 - 25, 90 + 25);
    shoot_parameters_main.base_yaw.uniformDistribution(-50, 50);

    ShootParameters shoot_parameters_cordon = shoot_parameters_main;
    shoot_parameters_cordon.phytomer_parameters.internode.image_texture = "GrapeBark.jpg";
    shoot_parameters_cordon.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_cordon.phytomer_parameters.internode.radial_subdivisions = 15;
    shoot_parameters_cordon.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_cordon.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_cordon.insertion_angle_tip.uniformDistribution(60, 120);
    shoot_parameters_cordon.girth_area_factor = 3.5f;
    shoot_parameters_cordon.max_nodes = 8;
    shoot_parameters_cordon.tortuosity = 1;
    shoot_parameters_cordon.gravitropic_curvature = 0;
    shoot_parameters_cordon.vegetative_bud_break_probability_min = 0.9;
    shoot_parameters_cordon.base_yaw = 0;
    shoot_parameters_cordon.defineChildShootTypes({"grapevine_shoot"}, {1.f});

    ShootParameters shoot_parameters_trunk = shoot_parameters_main;
    shoot_parameters_trunk.phytomer_parameters.internode.image_texture = "GrapeBark.jpg";
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.05;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 25;
    shoot_parameters_trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_trunk.phyllochron_min = 2.5;
    shoot_parameters_trunk.insertion_angle_tip = 90;
    shoot_parameters_trunk.girth_area_factor = 0;
    shoot_parameters_trunk.max_nodes = 18;
    shoot_parameters_trunk.tortuosity = 2;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.defineChildShootTypes({"grapevine_shoot"}, {1.f});

    defineShootType("grapevine_trunk", shoot_parameters_trunk);
    defineShootType("grapevine_cordon", shoot_parameters_cordon);
    defineShootType("grapevine_shoot", shoot_parameters_main);
}

uint PlantArchitecture::buildGrapevineWye(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize grapevine plant shoots
        initializeGrapevineWyeShoots();
    }

    // Get training system parameters
    auto trunk_height_total = getParameterValue(current_build_parameters, "trunk_height", 0.165f, 0.05f, 1.f, "total trunk height in meters");
    auto cordon_spacing = getParameterValue(current_build_parameters, "cordon_spacing", 0.6f, 0.2f, 2.f, "spacing between cordon rows in meters");
    auto vine_spacing = getParameterValue(current_build_parameters, "vine_spacing", 1.8f, 0.5f, 5.f, "plant-to-plant spacing in meters");
    auto catch_wire_height = getParameterValue(current_build_parameters, "catch_wire_height", 2.1f, 0.5f, 4.f, "absolute height of catch wires in meters");

    // Calculate trunk nodes based on desired height
    float trunk_internode_length = 0.165f / 8.f; // Original was 8 nodes * 0.165m total
    uint trunk_nodes = uint(trunk_height_total / trunk_internode_length);
    if (trunk_nodes < 1)
        trunk_nodes = 1;

    // Calculate trellis head height from catch wire height (catch wires above fruiting wires)
    float head_height = catch_wire_height - 0.35f; // Offset to match original geometry

    // Fixed training parameters (not user-customizable)
    uint upright_nodes = 3;
    float upright_pitch_min = 42.f;
    float upright_pitch_max = 48.f;
    float upright_radius = 0.03f;
    float upright_length = 0.14f;
    uint cordon_nodes = 8;
    float cordon_radius = 0.02f;
    float cordon_length = 0.11f;
    float catch_wire_offset_1 = 0.15f;
    float catch_wire_offset_2 = 0.35f;

    std::vector<std::vector<vec3>> trellis_points;

    // fruiting wires
    trellis_points.push_back(linspace(make_vec3(-0.5f * vine_spacing, -0.5f * cordon_spacing, head_height), make_vec3(0.5f * vine_spacing, -0.5f * cordon_spacing, head_height), 8));
    trellis_points.push_back(linspace(make_vec3(-0.5f * vine_spacing, 0.5f * cordon_spacing, head_height), make_vec3(0.5f * vine_spacing, 0.5f * cordon_spacing, head_height), 8));

    // first catch wires (these don't exist in a real Wye trellis, but are needed to keep the vines from falling through the wires)
    trellis_points.push_back(linspace(make_vec3(-0.5f * vine_spacing, -0.5f * cordon_spacing - catch_wire_offset_1, catch_wire_height - catch_wire_offset_2),
                                      make_vec3(0.5f * vine_spacing, -0.5f * cordon_spacing - catch_wire_offset_1, catch_wire_height - catch_wire_offset_2), 8));
    trellis_points.push_back(linspace(make_vec3(-0.5f * vine_spacing, 0.5f * cordon_spacing + catch_wire_offset_1, catch_wire_height - catch_wire_offset_2),
                                      make_vec3(0.5f * vine_spacing, 0.5f * cordon_spacing + catch_wire_offset_1, catch_wire_height - catch_wire_offset_2), 8));

    // second catch wires (at specified height)
    trellis_points.push_back(linspace(make_vec3(-0.5f * vine_spacing, -0.5f * cordon_spacing - catch_wire_offset_2, catch_wire_height), make_vec3(0.5f * vine_spacing, -0.5f * cordon_spacing - catch_wire_offset_2, catch_wire_height), 8));
    trellis_points.push_back(linspace(make_vec3(-0.5f * vine_spacing, 0.5f * cordon_spacing + catch_wire_offset_2, catch_wire_height), make_vec3(0.5f * vine_spacing, 0.5f * cordon_spacing + catch_wire_offset_2, catch_wire_height), 8));

    for (int j = 0; j < trellis_points.size(); j++) {
        for (int i = 0; i < trellis_points[j].size(); i++) {
            trellis_points.at(j).at(i) += base_position;
        }
    }

    uint plantID = addPlantInstance(base_position, 0);

    // Set plant-specific attraction points for this grapevine's trellis system
    setPlantAttractionPoints(plantID, flatten(trellis_points), 45.f, 0.5f, 0.5);

    uint uID_stem = addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(0., 0, 0), shoot_types.at("grapevine_trunk").phytomer_parameters.internode.radius_initial.val(), trunk_internode_length, 1, 1, 0.1, "grapevine_trunk");

    uint uID_upright_L = appendShoot(plantID, uID_stem, upright_nodes, make_AxisRotation(deg2rad(context_ptr->randu(upright_pitch_min, upright_pitch_max)), 0, M_PI), upright_radius, upright_length, 1, 1, 0.2, "grapevine_trunk");
    uint uID_upright_R = appendShoot(plantID, uID_stem, upright_nodes, make_AxisRotation(deg2rad(context_ptr->randu(upright_pitch_min, upright_pitch_max)), M_PI, M_PI), upright_radius, upright_length, 1, 1, 0.2, "grapevine_trunk");

    uint uID_cordon_L1 = appendShoot(plantID, uID_upright_L, cordon_nodes, make_AxisRotation(deg2rad(-90), 0.5 * M_PI, -0.2), cordon_radius, cordon_length, 1, 1, 0.5, "grapevine_cordon");
    uint uID_cordon_L2 = appendShoot(plantID, uID_upright_L, cordon_nodes, make_AxisRotation(deg2rad(-90), -0.5 * M_PI, 0.2), cordon_radius, cordon_length, 1, 1, 0.5, "grapevine_cordon");

    uint uID_cordon_R1 = appendShoot(plantID, uID_upright_R, cordon_nodes, make_AxisRotation(deg2rad(-90), 0.5 * M_PI, 0.2), cordon_radius, cordon_length, 1, 1, 0.5, "grapevine_cordon");
    uint uID_cordon_R2 = appendShoot(plantID, uID_upright_R, cordon_nodes, make_AxisRotation(deg2rad(-90), -0.5 * M_PI, -0.2), cordon_radius, cordon_length, 1, 1, 0.5, "grapevine_cordon");

    removeShootLeaves(plantID, uID_stem);
    removeShootLeaves(plantID, uID_upright_L);
    removeShootLeaves(plantID, uID_upright_R);
    removeShootLeaves(plantID, uID_cordon_L1);
    removeShootLeaves(plantID, uID_cordon_L2);
    removeShootLeaves(plantID, uID_cordon_R1);
    removeShootLeaves(plantID, uID_cordon_R2);

    removeShootVegetativeBuds(plantID, uID_upright_L);
    removeShootVegetativeBuds(plantID, uID_upright_R);

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
    shoot_parameters.tortuosity = 3;

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
    shoot_parameters_mainstem.tortuosity = 0.5f;
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
    shoot_parameters_trunk.girth_area_factor = 3.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 1;
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
    shoot_parameters_proleptic.girth_area_factor = 5.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.025;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 1.0;
    shoot_parameters_proleptic.vegetative_bud_break_time = 30;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(550, 650);
    shoot_parameters_proleptic.tortuosity = 5;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(35, 40);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 2;
    shoot_parameters_proleptic.internode_length_max = 0.05;
    shoot_parameters_proleptic.internode_length_min = 0.03;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.25;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.25;
    shoot_parameters_proleptic.max_terminal_floral_buds = 4;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"}, {1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 30;
    shoot_parameters_scaffold.max_nodes_per_season = 10;
    shoot_parameters_scaffold.gravitropic_curvature = 700;
    shoot_parameters_scaffold.internode_length_max = 0.04;
    shoot_parameters_scaffold.tortuosity = 3;
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
    leaf_prototype.leaf_aspect_ratio = 0.6f;
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
    phytomer_parameters_pistachio.internode.phyllotactic_angle.uniformDistribution(160, 200);
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
    phytomer_parameters_pistachio.leaf.prototype_scale = 0.08;
    phytomer_parameters_pistachio.leaf.leaflet_offset = 0.3;
    phytomer_parameters_pistachio.leaf.leaflet_scale = 0.75;
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
    shoot_parameters_trunk.girth_area_factor = 5.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 0.75;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"proleptic"}, {1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_pistachio;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = PistachioPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = PistachioPhytomerCallbackFunction;
    shoot_parameters_proleptic.phytomer_parameters.internode.pitch.uniformDistribution(-15, 15);
    shoot_parameters_proleptic.max_nodes.uniformDistribution(16, 24);
    shoot_parameters_proleptic.max_nodes_per_season.uniformDistribution(8, 10);
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.25;
    shoot_parameters_proleptic.girth_area_factor = 7.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.7;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 350;
    shoot_parameters_proleptic.tortuosity = 2;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution(45, 55);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 10;
    shoot_parameters_proleptic.internode_length_max = 0.06;
    shoot_parameters_proleptic.internode_length_min = 0.02;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.06;
    shoot_parameters_proleptic.fruit_set_probability = 0.2;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.35;
    shoot_parameters_proleptic.max_terminal_floral_buds = 2;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);
}

uint PlantArchitecture::buildPistachioTree(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        // automatically initialize pistachio tree shoots
        initializePistachioTreeShoots();
    }

    // Get training system parameters (with defaults matching original hard-coded values)
    auto trunk_height = getParameterValue(current_build_parameters, "trunk_height", 1.0f, 0.1f, 3.f, "total trunk height in meters");
    auto num_scaffolds = uint(getParameterValue(current_build_parameters, "num_scaffolds", 4.f, 2.f, 8.f, "number of scaffold branches"));
    auto scaffold_angle = getParameterValue(current_build_parameters, "scaffold_angle", 50.f, 20.f, 70.f, "scaffold branch angle in degrees");

    // Calculate trunk nodes based on desired height and internode length
    float trunk_internode_length = 0.05f; // Default internode length for pistachio
    uint trunk_nodes = uint(trunk_height / trunk_internode_length);
    if (trunk_nodes < 1)
        trunk_nodes = 1;

    // Fixed training parameters (not user-customizable)
    float scaffold_radius = 0.007f;
    float scaffold_length = 0.03f;
    uint scaffold_nodes = 5;

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
    // Four-scaffold training system in cardinal directions
    if (num_scaffolds >= 1) {
        addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - 1, scaffold_nodes, make_AxisRotation(pitch + context_ptr->randu(-0.15f, 0.15f), 0, 0.5 * M_PI + context_ptr->randu(-0.2f, 0.2f)), scaffold_radius, scaffold_length, 1.f,
                      1.f, 0.5, "proleptic", 0);
    }
    if (num_scaffolds >= 2) {
        addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - 1, scaffold_nodes, make_AxisRotation(pitch + context_ptr->randu(-0.15f, 0.15f), M_PI, 0.5 * M_PI + context_ptr->randu(-0.2f, 0.2f)), scaffold_radius, scaffold_length,
                      1.f, 1.f, 0.5, "proleptic", 0);
    }
    if (num_scaffolds >= 3) {
        addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - 2, scaffold_nodes, make_AxisRotation(pitch + context_ptr->randu(-0.15f, 0.15f), 0.5 * M_PI, 0.5 * M_PI + context_ptr->randu(-0.2f, 0.2f)), scaffold_radius,
                      scaffold_length, 1.f, 1.f, 0.5, "proleptic", 0);
    }
    if (num_scaffolds >= 4) {
        addChildShoot(plantID, uID_trunk, getShootNodeCount(plantID, uID_trunk) - 2, scaffold_nodes, make_AxisRotation(pitch + context_ptr->randu(-0.15f, 0.15f), 1.5 * M_PI, 0.5 * M_PI + context_ptr->randu(-0.2f, 0.2f)), scaffold_radius,
                      scaffold_length, 1.f, 1.f, 0.5, "proleptic", 0);
    }


    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, -1, 7, 20, 200);
    plant_instances.at(plantID).max_age = 1460;

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

    phytomer_parameters_puncturevine.petiole.petioles_per_internode = 1;
    phytomer_parameters_puncturevine.petiole.pitch.uniformDistribution(60, 80);
    phytomer_parameters_puncturevine.petiole.radius = 0.0005;
    phytomer_parameters_puncturevine.petiole.length = 0.03;
    phytomer_parameters_puncturevine.petiole.taper = 0;
    phytomer_parameters_puncturevine.petiole.curvature = 0;
    phytomer_parameters_puncturevine.petiole.color = phytomer_parameters_puncturevine.internode.color;
    phytomer_parameters_puncturevine.petiole.length_segments = 1;

    phytomer_parameters_puncturevine.leaf.leaves_per_petiole = 11;
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
    shoot_parameters_main.girth_area_factor = 4.f;
    shoot_parameters_main.gravitropic_curvature = 300;
    shoot_parameters_main.tortuosity = 5;
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
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 15;
    shoot_parameters_trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_trunk.insertion_angle_tip = 60;
    shoot_parameters_trunk.max_nodes = 75;
    shoot_parameters_trunk.max_nodes_per_season = 10;
    shoot_parameters_trunk.tortuosity = 1.5;
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
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(80, 100);
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
    leaf_prototype.leaf_texture_file[0] = "TomatoLeaf_centered.png";
    leaf_prototype.leaf_aspect_ratio = 0.5f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.45, -0.2f);
    leaf_prototype.lateral_curvature = -0.3f;
    leaf_prototype.wave_period = 0.35f;
    leaf_prototype.wave_amplitude = 0.08f;
    leaf_prototype.subdivisions = 6;
    leaf_prototype.unique_prototypes = 5;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(140, 220);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.2495, 0.3162, 0.0657);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45, 60);
    phytomer_parameters.petiole.radius = 0.002;
    phytomer_parameters.petiole.length = 0.2;
    phytomer_parameters.petiole.taper = 0.25;
    phytomer_parameters.petiole.curvature.uniformDistribution(-150, -50);
    phytomer_parameters.petiole.color = make_RGBcolor(0.3, 0.37, 0.0657);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 7;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.leaflet_offset = 0.15;
    phytomer_parameters.leaf.leaflet_scale = 0.7;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12, 0.18);
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
    shoot_parameters.tortuosity = 3;

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
    leaf_prototype.leaf_texture_file[0] = "CherryTomatoLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.6f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.3, -0.15f);
    leaf_prototype.lateral_curvature = -0.8f;
    leaf_prototype.wave_period = 0.35f;
    leaf_prototype.wave_amplitude = 0.08f;
    leaf_prototype.subdivisions = 7;
    leaf_prototype.unique_prototypes = 5;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 5;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(14, 220);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.2495, 0.3162, 0.0657);
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.internode.radial_subdivisions = 14;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45, 60);
    phytomer_parameters.petiole.radius = 0.0025;
    phytomer_parameters.petiole.length = 0.25;
    phytomer_parameters.petiole.taper = 0.25;
    phytomer_parameters.petiole.curvature.uniformDistribution(-250, 0);
    phytomer_parameters.petiole.color = make_RGBcolor(0.32, 0.37, 0.12);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 9;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll.uniformDistribution(-20, 20);
    phytomer_parameters.leaf.leaflet_offset = 0.22;
    phytomer_parameters.leaf.leaflet_scale = 0.9;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12, 0.17);
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
    shoot_parameters.tortuosity = 1.5;

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
    phytomer_parameters_walnut.internode.phyllotactic_angle.uniformDistribution(160, 200);
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
    shoot_parameters_trunk.girth_area_factor = 5.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 1;
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
    shoot_parameters_proleptic.girth_area_factor = 9.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.05;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.7;
    shoot_parameters_proleptic.vegetative_bud_break_time = 3;
    shoot_parameters_proleptic.gravitropic_curvature = 300;
    shoot_parameters_proleptic.tortuosity = 4;
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
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 30;
    shoot_parameters_scaffold.gravitropic_curvature = 300;
    shoot_parameters_scaffold.internode_length_max = 0.06;
    shoot_parameters_scaffold.tortuosity = 4;
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
    phytomer_parameters_wheat.internode.phyllotactic_angle.uniformDistribution(67, 77);
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
