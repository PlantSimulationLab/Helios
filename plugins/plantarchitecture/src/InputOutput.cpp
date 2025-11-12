/** \file "InputOutput.cpp" Routines for reading and writing plant geometry in the plant architecture plug-in.

Copyright (C) 2016-2025 Brian Bailey

This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "PlantArchitecture.h"

using namespace helios;

std::string PlantArchitecture::makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> &shoot_tree) const {

    std::string outstring = current_string;

    if (shoot->parent_shoot_ID != -1) {
        outstring += "[";
    }

    outstring += "{" + std::to_string(rad2deg(shoot->base_rotation.pitch)) + "," + std::to_string(rad2deg(shoot->base_rotation.yaw)) + "," + std::to_string(rad2deg(shoot->base_rotation.roll)) + "," +
                 std::to_string(shoot->shoot_parameters.gravitropic_curvature.val() /*\todo */) + "," + shoot->shoot_type_label + "}";

    uint node_number = 0;
    for (auto &phytomer: shoot->phytomers) {

        float length = phytomer->getInternodeLength();
        float radius = phytomer->getInternodeRadius();

        outstring += "Internode(" + std::to_string(length) + "," + std::to_string(radius) + "," + std::to_string(rad2deg(phytomer->internode_pitch)) + "," + std::to_string(rad2deg(phytomer->internode_phyllotactic_angle)) + ")";

        for (uint petiole = 0; petiole < phytomer->petiole_length.size(); petiole++) {

            outstring += "Petiole(" + std::to_string(phytomer->petiole_length.at(petiole)) + "," + std::to_string(phytomer->petiole_radii.at(petiole).front()) + "," + std::to_string(rad2deg(phytomer->petiole_pitch.at(petiole))) + ")";

            //\todo If leaf is compound, just using rotation for the first leaf for now rather than adding multiple 'Leaf()' strings for each leaflet.
            outstring += "Leaf(" + std::to_string(phytomer->leaf_size_max.at(petiole).front() * phytomer->current_leaf_scale_factor.at(petiole)) + "," + std::to_string(rad2deg(phytomer->leaf_rotation.at(petiole).front().pitch)) + "," +
                         std::to_string(rad2deg(phytomer->leaf_rotation.at(petiole).front().yaw)) + "," + std::to_string(rad2deg(phytomer->leaf_rotation.at(petiole).front().roll)) + ")";

            if (shoot->childIDs.find(node_number) != shoot->childIDs.end()) {
                for (int childID: shoot->childIDs.at(node_number)) {
                    outstring = makeShootString(outstring, shoot_tree.at(childID), shoot_tree);
                }
            }
        }

        node_number++;
    }

    if (shoot->parent_shoot_ID != -1) {
        outstring += "]";
    }

    return outstring;
}

std::string PlantArchitecture::getPlantString(uint plantID) const {

    auto plant_shoot_tree = &plant_instances.at(plantID).shoot_tree;

    std::string out_string;

    for (auto &shoot: *plant_shoot_tree) {
        out_string = makeShootString(out_string, shoot, *plant_shoot_tree);
    }

    return out_string;
}

void PlantArchitecture::parseShootArgument(const std::string &shoot_argument, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters, AxisRotation &base_rotation, std::string &phytomer_label) {

    // shoot argument order {}:
    //  1. shoot base pitch/insertion angle (degrees)
    //  2. shoot base yaw angle (degrees)
    //  3. shoot roll angle (degrees)
    //  4. gravitropic curvature (degrees/meter)
    //  5. [optional] phytomer parameters string

    size_t pos_shoot_start = 0;

    std::string s_argument = shoot_argument;

    pos_shoot_start = s_argument.find(',');
    if (pos_shoot_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float insertion_angle = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    base_rotation.pitch = deg2rad(insertion_angle);

    pos_shoot_start = s_argument.find(',');
    if (pos_shoot_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float shoot_yaw = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    base_rotation.yaw = deg2rad(shoot_yaw);

    pos_shoot_start = s_argument.find(',');
    if (pos_shoot_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float shoot_roll = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    base_rotation.roll = deg2rad(shoot_roll);

    pos_shoot_start = s_argument.find(',');
    if (pos_shoot_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float shoot_curvature = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    shoot_parameters.gravitropic_curvature = shoot_curvature;

    if (pos_shoot_start != std::string::npos) { // shoot type argument was given
        pos_shoot_start = s_argument.find(',');
        phytomer_label = s_argument.substr(0, pos_shoot_start);
        s_argument.erase(0, pos_shoot_start + 1);
        if (phytomer_parameters.find(phytomer_label) == phytomer_parameters.end()) {
            helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Phytomer parameters with label " + phytomer_label + " was not provided to PlantArchitecture::generatePlantFromString().");
        }
        shoot_parameters.phytomer_parameters = phytomer_parameters.at(phytomer_label);
    } else { // shoot type argument not given - use first phytomer parameters in the map
        phytomer_label = phytomer_parameters.begin()->first;
        shoot_parameters.phytomer_parameters = phytomer_parameters.begin()->second;
    }
}

void PlantArchitecture::parseInternodeArgument(const std::string &internode_argument, float &internode_radius, float &internode_length, PhytomerParameters &phytomer_parameters) {

    // internode argument order Internode():
    //  1. internode length (m)
    //  2. internode radius (m)
    //  3. internode pitch (degrees)
    //  4. phyllotactic angle (degrees)

    size_t pos_inode_start = 0;

    std::string inode_argument = internode_argument;

    pos_inode_start = inode_argument.find(',');
    if (pos_inode_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    internode_length = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);

    pos_inode_start = inode_argument.find(',');
    if (pos_inode_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    internode_radius = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);

    pos_inode_start = inode_argument.find(',');
    if (pos_inode_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    float internode_pitch = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);
    phytomer_parameters.internode.pitch = internode_pitch;

    pos_inode_start = inode_argument.find(',');
    if (pos_inode_start != std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    float internode_phyllotaxis = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);
    phytomer_parameters.internode.phyllotactic_angle = internode_phyllotaxis;
}

void PlantArchitecture::parsePetioleArgument(const std::string &petiole_argument, PhytomerParameters &phytomer_parameters) {

    // petiole argument order Petiole():
    //  1. petiole length (m)
    //  2. petiole radius (m)
    //  3. petiole pitch (degrees)

    if (petiole_argument.empty()) {
        phytomer_parameters.petiole.length = 0;
        return;
    }

    size_t pos_petiole_start = 0;

    std::string pet_argument = petiole_argument;

    pos_petiole_start = pet_argument.find(',');
    if (pos_petiole_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Petiole()' does not have the correct number of values given.");
    }
    float petiole_length = std::stof(pet_argument.substr(0, pos_petiole_start));
    pet_argument.erase(0, pos_petiole_start + 1);
    phytomer_parameters.petiole.length = petiole_length;

    pos_petiole_start = pet_argument.find(',');
    if (pos_petiole_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Petiole()' does not have the correct number of values given.");
    }
    float petiole_radius = std::stof(pet_argument.substr(0, pos_petiole_start));
    pet_argument.erase(0, pos_petiole_start + 1);
    phytomer_parameters.petiole.radius = petiole_radius;

    pos_petiole_start = pet_argument.find(',');
    if (pos_petiole_start != std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Petiole()' does not have the correct number of values given.");
    }
    float petiole_pitch = std::stof(pet_argument.substr(0, pos_petiole_start));
    pet_argument.erase(0, pos_petiole_start + 1);
    phytomer_parameters.petiole.pitch = petiole_pitch;
}

void PlantArchitecture::parseLeafArgument(const std::string &leaf_argument, PhytomerParameters &phytomer_parameters) {

    // leaf argument order Leaf():
    //  1. leaf scale factor
    //  2. leaf pitch (degrees)
    //  3. leaf yaw (degrees)
    //  4. leaf roll (degrees)

    if (leaf_argument.empty()) {
        phytomer_parameters.leaf.prototype_scale = 0;
        return;
    }

    size_t pos_leaf_start = 0;

    std::string l_argument = leaf_argument;

    pos_leaf_start = l_argument.find(',');
    if (pos_leaf_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Leaf()' does not have the correct number of values given.");
    }
    float leaf_scale = std::stof(l_argument.substr(0, pos_leaf_start));
    l_argument.erase(0, pos_leaf_start + 1);
    phytomer_parameters.leaf.prototype_scale = leaf_scale;

    pos_leaf_start = l_argument.find(',');
    if (pos_leaf_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Leaf()' does not have the correct number of values given.");
    }
    float leaf_pitch = std::stof(l_argument.substr(0, pos_leaf_start));
    l_argument.erase(0, pos_leaf_start + 1);
    phytomer_parameters.leaf.pitch = leaf_pitch;

    pos_leaf_start = l_argument.find(',');
    if (pos_leaf_start == std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Leaf()' does not have the correct number of values given.");
    }
    float leaf_yaw = std::stof(l_argument.substr(0, pos_leaf_start));
    l_argument.erase(0, pos_leaf_start + 1);
    phytomer_parameters.leaf.yaw = leaf_yaw;

    pos_leaf_start = l_argument.find(',');
    if (pos_leaf_start != std::string::npos) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Leaf()' does not have the correct number of values given.");
    }
    float leaf_roll = std::stof(l_argument.substr(0, pos_leaf_start));
    l_argument.erase(0, pos_leaf_start + 1);
    phytomer_parameters.leaf.roll = leaf_roll;
}

size_t findShootClosingBracket(const std::string &lstring) {

    size_t pos_close = std::string::npos;
    size_t pos_open = lstring.find_first_of('[', 0);
    if (pos_open == std::string::npos) {
        return pos_close;
    }

    size_t pos = pos_open;
    int count = 1;
    while (count > 0) {
        pos++;
        if (lstring[pos] == '[') {
            count++;
        } else if (lstring[pos] == ']') {
            count--;
        }
        if (pos == lstring.length()) {
            return pos_close;
        }
    }
    if (count == 0) {
        pos_close = pos;
    }
    return pos_close; // Return the position of the closing bracket
}

void PlantArchitecture::parseStringShoot(const std::string &LString_shoot, uint plantID, int parentID, uint parent_node, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters) {

    std::string lstring_tobeparsed = LString_shoot;

    size_t pos_inode_start = 0;
    std::string inode_delimiter = "Internode(";
    std::string petiole_delimiter = "Petiole(";
    std::string leaf_delimiter = "Leaf(";
    bool base_shoot = true;
    uint baseID;
    AxisRotation shoot_base_rotation;

    // parse shoot arguments
    if (LString_shoot.front() != '{') {
        helios_runtime_error("ERROR (PlantArchitecture::parseStringShoot): Shoot string is not formatted correctly. All shoots should start with a curly bracket containing two arguments {X,Y}.");
    }
    size_t pos_shoot_end = lstring_tobeparsed.find('}');
    std::string shoot_argument = lstring_tobeparsed.substr(1, pos_shoot_end - 1);
    std::string phytomer_label;
    parseShootArgument(shoot_argument, phytomer_parameters, shoot_parameters, shoot_base_rotation, phytomer_label);
    lstring_tobeparsed.erase(0, pos_shoot_end + 1);

    uint shoot_node_count = 0;
    while ((pos_inode_start = lstring_tobeparsed.find(inode_delimiter)) != std::string::npos) {

        if (pos_inode_start != 0) {
            helios_runtime_error("ERROR (PlantArchitecture::parseStringShoot): Shoot string is not formatted correctly.");
        }

        size_t pos_inode_end = lstring_tobeparsed.find(')');
        std::string inode_argument = lstring_tobeparsed.substr(pos_inode_start + inode_delimiter.length(), pos_inode_end - pos_inode_start - inode_delimiter.length());
        float internode_radius = 0;
        float internode_length = 0;
        parseInternodeArgument(inode_argument, internode_radius, internode_length, shoot_parameters.phytomer_parameters);
        lstring_tobeparsed.erase(0, pos_inode_end + 1);

        size_t pos_petiole_start = lstring_tobeparsed.find(petiole_delimiter);
        size_t pos_petiole_end = lstring_tobeparsed.find(')');
        std::string petiole_argument;
        if (pos_petiole_start == 0) {
            petiole_argument = lstring_tobeparsed.substr(pos_petiole_start + petiole_delimiter.length(), pos_petiole_end - pos_petiole_start - petiole_delimiter.length());
        } else {
            petiole_argument = "";
        }
        parsePetioleArgument(petiole_argument, shoot_parameters.phytomer_parameters);
        if (pos_petiole_start == 0) {
            lstring_tobeparsed.erase(0, pos_petiole_end + 1);
        }

        size_t pos_leaf_start = lstring_tobeparsed.find(leaf_delimiter);
        size_t pos_leaf_end = lstring_tobeparsed.find(')');
        std::string leaf_argument;
        if (pos_leaf_start == 0) {
            leaf_argument = lstring_tobeparsed.substr(pos_leaf_start + leaf_delimiter.length(), pos_leaf_end - pos_leaf_start - leaf_delimiter.length());
        } else {
            leaf_argument = "";
        }
        parseLeafArgument(leaf_argument, shoot_parameters.phytomer_parameters);
        if (pos_leaf_start == 0) {
            lstring_tobeparsed.erase(0, pos_leaf_end + 1);
        }

        // override phytomer creation function
        shoot_parameters.phytomer_parameters.phytomer_creation_function = nullptr;

        if (base_shoot) { // this is the first phytomer of the shoot
            //            defineShootType("shoot_"+phytomer_label, shoot_parameters);
            defineShootType(phytomer_label, shoot_parameters); //*testing*

            if (parentID < 0) { // this is the first shoot of the plant
                //                baseID = addBaseStemShoot(plantID, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, "shoot_" + phytomer_label);
                baseID = addBaseStemShoot(plantID, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, phytomer_label); //*testing*
            } else { // this is a child of an existing shoot
                //                baseID = addChildShoot(plantID, parentID, parent_node, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, "shoot_" + phytomer_label, 0);
                baseID = addChildShoot(plantID, parentID, parent_node, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, phytomer_label, 0); //*testing*
            }

            base_shoot = false;
        } else {
            appendPhytomerToShoot(plantID, baseID, shoot_parameters.phytomer_parameters, internode_radius, internode_length, 1, 1);
        }

        while (!lstring_tobeparsed.empty() && lstring_tobeparsed.substr(0, 1) == "[") {
            size_t pos_shoot_bracket_end = findShootClosingBracket(lstring_tobeparsed);
            if (pos_shoot_bracket_end == std::string::npos) {
                helios_runtime_error("ERROR (PlantArchitecture::parseStringShoot): Shoot string is not formatted correctly. Shoots must be closed with a ']'.");
            }
            std::string lstring_child = lstring_tobeparsed.substr(1, pos_shoot_bracket_end - 1);
            parseStringShoot(lstring_child, plantID, (int) baseID, shoot_node_count, phytomer_parameters, shoot_parameters);
            lstring_tobeparsed.erase(0, pos_shoot_bracket_end + 1);
        }

        shoot_node_count++;
    }
}

uint PlantArchitecture::generatePlantFromString(const std::string &generation_string, const PhytomerParameters &phytomer_parameters) {
    std::map<std::string, PhytomerParameters> phytomer_parameters_map;
    phytomer_parameters_map["default"] = phytomer_parameters;
    return generatePlantFromString(generation_string, phytomer_parameters_map);
}

uint PlantArchitecture::generatePlantFromString(const std::string &generation_string, const std::map<std::string, PhytomerParameters> &phytomer_parameters) {

    // check that first characters are 'Internode'
    if (generation_string.front() != '{') {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): First character of string must be '{'");
    }

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.max_nodes = 200;
    shoot_parameters.vegetative_bud_break_probability_min = 0;
    shoot_parameters.vegetative_bud_break_time = 0;
    shoot_parameters.phyllochron_min = 0;

    // assign default phytomer parameters. This can be changed later if the optional phytomer parameters label is provided in the shoot argument '{}'.
    if (phytomer_parameters.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Phytomer parameters must be provided.");
    }

    uint plantID;

    plantID = addPlantInstance(nullorigin, 0);

    size_t pos_first_child_shoot = generation_string.find('[');

    if (pos_first_child_shoot == std::string::npos) {
        pos_first_child_shoot = generation_string.length();
    }

    parseStringShoot(generation_string, plantID, -1, 0, phytomer_parameters, shoot_parameters);


    return plantID;
}

void PlantArchitecture::comparePlantGeometry(uint plantID_original, uint plantID_loaded, const std::string &output_prefix) const {

    if (plant_instances.find(plantID_original) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::comparePlantGeometry): Original plant ID " + std::to_string(plantID_original) + " does not exist.");
    }
    if (plant_instances.find(plantID_loaded) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::comparePlantGeometry): Loaded plant ID " + std::to_string(plantID_loaded) + " does not exist.");
    }

    const auto &plant_orig = plant_instances.at(plantID_original);
    const auto &plant_load = plant_instances.at(plantID_loaded);

    // Statistics tracking
    float max_internode_position_delta = 0.f;
    float max_internode_radius_delta = 0.f;
    float max_petiole_position_delta = 0.f;
    float max_petiole_radius_delta = 0.f;
    float max_leaf_position_delta = 0.f;
    float max_leaf_rotation_delta = 0.f;
    float max_peduncle_position_delta = 0.f;
    float max_peduncle_radius_delta = 0.f;
    float max_flower_position_delta = 0.f;
    float sum_internode_position_delta = 0.f;
    float sum_internode_radius_delta = 0.f;
    float sum_petiole_position_delta = 0.f;
    float sum_petiole_radius_delta = 0.f;
    float sum_leaf_position_delta = 0.f;
    float sum_leaf_rotation_delta = 0.f;
    float sum_peduncle_position_delta = 0.f;
    float sum_peduncle_radius_delta = 0.f;
    float sum_flower_position_delta = 0.f;
    uint internode_vertex_count = 0;
    uint internode_radius_count = 0;
    uint petiole_vertex_count = 0;
    uint petiole_radius_count = 0;
    uint leaf_count = 0;
    uint peduncle_vertex_count = 0;
    uint peduncle_radius_count = 0;
    uint flower_count = 0;

    // Open CSV files
    std::ofstream internode_csv(output_prefix + "_internodes.csv");
    std::ofstream petiole_csv(output_prefix + "_petioles.csv");

    if (!internode_csv.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::comparePlantGeometry): Could not open file " + output_prefix + "_internodes.csv for writing.");
    }
    if (!petiole_csv.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::comparePlantGeometry): Could not open file " + output_prefix + "_petioles.csv for writing.");
    }

    std::ofstream leaf_csv(output_prefix + "_leaves.csv");
    if (!leaf_csv.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::comparePlantGeometry): Could not open file " + output_prefix + "_leaves.csv for writing.");
    }

    std::ofstream peduncle_csv(output_prefix + "_peduncles.csv");
    if (!peduncle_csv.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::comparePlantGeometry): Could not open file " + output_prefix + "_peduncles.csv for writing.");
    }

    std::ofstream flower_csv(output_prefix + "_flowers.csv");
    if (!flower_csv.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::comparePlantGeometry): Could not open file " + output_prefix + "_flowers.csv for writing.");
    }

    // Write CSV headers
    internode_csv << "shoot_id,phytomer_idx,segment_idx,orig_x,orig_y,orig_z,load_x,load_y,load_z,position_delta,orig_radius,load_radius,radius_delta" << std::endl;
    petiole_csv << "shoot_id,phytomer_idx,petiole_idx,segment_idx,orig_x,orig_y,orig_z,load_x,load_y,load_z,position_delta,orig_radius,load_radius,radius_delta" << std::endl;
    leaf_csv << "shoot_id,phytomer_idx,petiole_idx,leaf_idx,orig_x,orig_y,orig_z,load_x,load_y,load_z,position_delta,orig_pitch,orig_yaw,orig_roll,load_pitch,load_yaw,load_roll,rotation_delta" << std::endl;
    peduncle_csv << "shoot_id,phytomer_idx,petiole_idx,bud_idx,segment_idx,orig_x,orig_y,orig_z,load_x,load_y,load_z,position_delta,orig_radius,load_radius,radius_delta" << std::endl;
    flower_csv << "shoot_id,phytomer_idx,petiole_idx,bud_idx,flower_idx,orig_x,orig_y,orig_z,load_x,load_y,load_z,position_delta" << std::endl;

    // Compare internodes
    size_t min_shoot_count = std::min(plant_orig.shoot_tree.size(), plant_load.shoot_tree.size());
    for (size_t shoot_idx = 0; shoot_idx < min_shoot_count; shoot_idx++) {
        const auto &shoot_orig = plant_orig.shoot_tree[shoot_idx];
        const auto &shoot_load = plant_load.shoot_tree[shoot_idx];

        size_t min_phytomer_count = std::min(shoot_orig->phytomers.size(), shoot_load->phytomers.size());
        for (size_t phyt_idx = 0; phyt_idx < min_phytomer_count; phyt_idx++) {

            // Compare internode vertices and radii
            const auto &verts_orig = shoot_orig->shoot_internode_vertices[phyt_idx];
            const auto &verts_load = shoot_load->shoot_internode_vertices[phyt_idx];
            const auto &radii_orig = shoot_orig->shoot_internode_radii[phyt_idx];
            const auto &radii_load = shoot_load->shoot_internode_radii[phyt_idx];

            size_t min_vert_count = std::min(verts_orig.size(), verts_load.size());
            for (size_t seg_idx = 0; seg_idx < min_vert_count; seg_idx++) {
                vec3 pos_delta_vec = verts_orig[seg_idx] - verts_load[seg_idx];
                float pos_delta = pos_delta_vec.magnitude();
                max_internode_position_delta = std::max(max_internode_position_delta, pos_delta);
                sum_internode_position_delta += pos_delta;
                internode_vertex_count++;

                internode_csv << shoot_orig->ID << "," << phyt_idx << "," << seg_idx << "," << verts_orig[seg_idx].x << "," << verts_orig[seg_idx].y << "," << verts_orig[seg_idx].z << "," << verts_load[seg_idx].x << "," << verts_load[seg_idx].y
                              << "," << verts_load[seg_idx].z << "," << pos_delta << ",";

                if (seg_idx < radii_orig.size() && seg_idx < radii_load.size()) {
                    float rad_delta = std::abs(radii_orig[seg_idx] - radii_load[seg_idx]);
                    max_internode_radius_delta = std::max(max_internode_radius_delta, rad_delta);
                    sum_internode_radius_delta += rad_delta;
                    internode_radius_count++;
                    internode_csv << radii_orig[seg_idx] << "," << radii_load[seg_idx] << "," << rad_delta;
                } else {
                    internode_csv << "NA,NA,NA";
                }
                internode_csv << std::endl;
            }

            // Compare petiole vertices and radii
            const auto &phyt_orig = shoot_orig->phytomers[phyt_idx];
            const auto &phyt_load = shoot_load->phytomers[phyt_idx];

            size_t min_petiole_count = std::min(phyt_orig->petiole_vertices.size(), phyt_load->petiole_vertices.size());
            for (size_t pet_idx = 0; pet_idx < min_petiole_count; pet_idx++) {
                const auto &pet_verts_orig = phyt_orig->petiole_vertices[pet_idx];
                const auto &pet_verts_load = phyt_load->petiole_vertices[pet_idx];
                const auto &pet_radii_orig = phyt_orig->petiole_radii[pet_idx];
                const auto &pet_radii_load = phyt_load->petiole_radii[pet_idx];

                size_t min_pet_vert_count = std::min(pet_verts_orig.size(), pet_verts_load.size());
                for (size_t seg_idx = 0; seg_idx < min_pet_vert_count; seg_idx++) {
                    vec3 pos_delta_vec = pet_verts_orig[seg_idx] - pet_verts_load[seg_idx];
                    float pos_delta = pos_delta_vec.magnitude();
                    max_petiole_position_delta = std::max(max_petiole_position_delta, pos_delta);
                    sum_petiole_position_delta += pos_delta;
                    petiole_vertex_count++;

                    petiole_csv << shoot_orig->ID << "," << phyt_idx << "," << pet_idx << "," << seg_idx << "," << pet_verts_orig[seg_idx].x << "," << pet_verts_orig[seg_idx].y << "," << pet_verts_orig[seg_idx].z << "," << pet_verts_load[seg_idx].x
                                << "," << pet_verts_load[seg_idx].y << "," << pet_verts_load[seg_idx].z << "," << pos_delta << ",";

                    if (seg_idx < pet_radii_orig.size() && seg_idx < pet_radii_load.size()) {
                        float rad_delta = std::abs(pet_radii_orig[seg_idx] - pet_radii_load[seg_idx]);
                        max_petiole_radius_delta = std::max(max_petiole_radius_delta, rad_delta);
                        sum_petiole_radius_delta += rad_delta;
                        petiole_radius_count++;
                        petiole_csv << pet_radii_orig[seg_idx] << "," << pet_radii_load[seg_idx] << "," << rad_delta;
                    } else {
                        petiole_csv << "NA,NA,NA";
                    }
                    petiole_csv << std::endl;
                }
            }
        }
    }

    // Compare leaves
    for (size_t shoot_idx = 0; shoot_idx < min_shoot_count; shoot_idx++) {
        const auto &shoot_orig = plant_orig.shoot_tree[shoot_idx];
        const auto &shoot_load = plant_load.shoot_tree[shoot_idx];

        size_t min_phytomer_count = std::min(shoot_orig->phytomers.size(), shoot_load->phytomers.size());
        for (size_t phyt_idx = 0; phyt_idx < min_phytomer_count; phyt_idx++) {
            const auto &phyt_orig = shoot_orig->phytomers[phyt_idx];
            const auto &phyt_load = shoot_load->phytomers[phyt_idx];

            size_t min_petiole_count = std::min(phyt_orig->leaf_bases.size(), phyt_load->leaf_bases.size());
            for (size_t pet_idx = 0; pet_idx < min_petiole_count; pet_idx++) {
                const auto &leaf_bases_orig = phyt_orig->leaf_bases[pet_idx];
                const auto &leaf_bases_load = phyt_load->leaf_bases[pet_idx];
                const auto &leaf_objIDs_orig = phyt_orig->leaf_objIDs[pet_idx];
                const auto &leaf_objIDs_load = phyt_load->leaf_objIDs[pet_idx];

                size_t min_leaf_count = std::min(leaf_bases_orig.size(), leaf_bases_load.size());
                for (size_t leaf_idx = 0; leaf_idx < min_leaf_count; leaf_idx++) {
                    vec3 pos_delta_vec = leaf_bases_orig[leaf_idx] - leaf_bases_load[leaf_idx];
                    float pos_delta = pos_delta_vec.magnitude();
                    max_leaf_position_delta = std::max(max_leaf_position_delta, pos_delta);
                    sum_leaf_position_delta += pos_delta;
                    leaf_count++;

                    // Compare actual Context object transforms, not stored rotation values
                    // (for compound leaves, stored values don't match actual geometry)
                    float rot_delta = 0.f;
                    float orig_pitch = 0.f, orig_yaw = 0.f, orig_roll = 0.f;
                    float load_pitch = 0.f, load_yaw = 0.f, load_roll = 0.f;

                    if (leaf_idx < leaf_objIDs_orig.size() && leaf_idx < leaf_objIDs_load.size()) {
                        // Get actual primitive transforms from Context
                        uint objID_orig = leaf_objIDs_orig[leaf_idx];
                        uint objID_load = leaf_objIDs_load[leaf_idx];

                        // Get object centroids to compare actual position
                        vec3 centroid_orig = context_ptr->getObjectCenter(objID_orig);
                        vec3 centroid_load = context_ptr->getObjectCenter(objID_load);

                        // Calculate average primitive normals to estimate orientation
                        // (this is a proxy for rotation since Context doesn't store explicit object rotations)
                        std::vector<uint> prims_orig = context_ptr->getObjectPrimitiveUUIDs(objID_orig);
                        std::vector<uint> prims_load = context_ptr->getObjectPrimitiveUUIDs(objID_load);

                        if (!prims_orig.empty() && !prims_load.empty()) {
                            vec3 avg_normal_orig = make_vec3(0, 0, 0);
                            for (uint UUID: prims_orig) {
                                avg_normal_orig = avg_normal_orig + context_ptr->getPrimitiveNormal(UUID);
                            }
                            avg_normal_orig = avg_normal_orig / float(prims_orig.size());
                            avg_normal_orig.normalize();

                            vec3 avg_normal_load = make_vec3(0, 0, 0);
                            for (uint UUID: prims_load) {
                                avg_normal_load = avg_normal_load + context_ptr->getPrimitiveNormal(UUID);
                            }
                            avg_normal_load = avg_normal_load / float(prims_load.size());
                            avg_normal_load.normalize();

                            // Calculate angle between average normals as rotation metric
                            float dot = avg_normal_orig.x * avg_normal_load.x + avg_normal_orig.y * avg_normal_load.y + avg_normal_orig.z * avg_normal_load.z;
                            dot = std::max(-1.f, std::min(1.f, dot)); // clamp to [-1, 1]
                            rot_delta = std::acos(dot); // angle between normals in radians

                            max_leaf_rotation_delta = std::max(max_leaf_rotation_delta, rot_delta);
                            sum_leaf_rotation_delta += rot_delta;

                            // For CSV output, use stored rotation values (with caveat that they don't match geometry for compound leaves)
                            const auto &leaf_rot_orig = phyt_orig->leaf_rotation[pet_idx];
                            const auto &leaf_rot_load = phyt_load->leaf_rotation[pet_idx];
                            if (leaf_idx < leaf_rot_orig.size()) {
                                orig_pitch = leaf_rot_orig[leaf_idx].pitch;
                                orig_yaw = leaf_rot_orig[leaf_idx].yaw;
                                orig_roll = leaf_rot_orig[leaf_idx].roll;
                            }
                            if (leaf_idx < leaf_rot_load.size()) {
                                load_pitch = leaf_rot_load[leaf_idx].pitch;
                                load_yaw = leaf_rot_load[leaf_idx].yaw;
                                load_roll = leaf_rot_load[leaf_idx].roll;
                            }
                        }
                    }

                    leaf_csv << shoot_orig->ID << "," << phyt_idx << "," << pet_idx << "," << leaf_idx << "," << leaf_bases_orig[leaf_idx].x << "," << leaf_bases_orig[leaf_idx].y << "," << leaf_bases_orig[leaf_idx].z << ","
                             << leaf_bases_load[leaf_idx].x << "," << leaf_bases_load[leaf_idx].y << "," << leaf_bases_load[leaf_idx].z << "," << pos_delta << "," << rad2deg(orig_pitch) << "," << rad2deg(orig_yaw) << "," << rad2deg(orig_roll) << ","
                             << rad2deg(load_pitch) << "," << rad2deg(load_yaw) << "," << rad2deg(load_roll) << "," << rad2deg(rot_delta) << std::endl;
                }
            }
        }
    }

    // Compare peduncle and flower/fruit geometry
    for (size_t shoot_idx = 0; shoot_idx < min_shoot_count; shoot_idx++) {
        const auto &shoot_orig = plant_orig.shoot_tree[shoot_idx];
        const auto &shoot_load = plant_load.shoot_tree[shoot_idx];

        size_t min_phytomer_count = std::min(shoot_orig->phytomers.size(), shoot_load->phytomers.size());
        for (size_t phyt_idx = 0; phyt_idx < min_phytomer_count; phyt_idx++) {
            const auto &phyt_orig = shoot_orig->phytomers[phyt_idx];
            const auto &phyt_load = shoot_load->phytomers[phyt_idx];

            // Compare peduncles
            size_t min_petiole_count = std::min(phyt_orig->peduncle_vertices.size(), phyt_load->peduncle_vertices.size());
            for (size_t pet_idx = 0; pet_idx < min_petiole_count; pet_idx++) {
                size_t min_bud_count = std::min(phyt_orig->peduncle_vertices[pet_idx].size(), phyt_load->peduncle_vertices[pet_idx].size());

                for (size_t bud_idx = 0; bud_idx < min_bud_count; bud_idx++) {
                    const auto &ped_verts_orig = phyt_orig->peduncle_vertices[pet_idx][bud_idx];
                    const auto &ped_verts_load = phyt_load->peduncle_vertices[pet_idx][bud_idx];
                    const auto &ped_radii_orig = phyt_orig->peduncle_radii[pet_idx][bud_idx];
                    const auto &ped_radii_load = phyt_load->peduncle_radii[pet_idx][bud_idx];

                    size_t min_vert_count = std::min(ped_verts_orig.size(), ped_verts_load.size());
                    for (size_t seg_idx = 0; seg_idx < min_vert_count; seg_idx++) {
                        vec3 delta_vec = ped_verts_orig[seg_idx] - ped_verts_load[seg_idx];
                        float pos_delta = delta_vec.magnitude();
                        max_peduncle_position_delta = std::max(max_peduncle_position_delta, pos_delta);
                        sum_peduncle_position_delta += pos_delta;
                        peduncle_vertex_count++;

                        peduncle_csv << shoot_orig->ID << "," << phyt_idx << "," << pet_idx << "," << bud_idx << "," << seg_idx << "," << ped_verts_orig[seg_idx].x << "," << ped_verts_orig[seg_idx].y << "," << ped_verts_orig[seg_idx].z << ","
                                     << ped_verts_load[seg_idx].x << "," << ped_verts_load[seg_idx].y << "," << ped_verts_load[seg_idx].z << "," << pos_delta << ",";

                        if (seg_idx < ped_radii_orig.size() && seg_idx < ped_radii_load.size()) {
                            float rad_delta = std::abs(ped_radii_orig[seg_idx] - ped_radii_load[seg_idx]);
                            max_peduncle_radius_delta = std::max(max_peduncle_radius_delta, rad_delta);
                            sum_peduncle_radius_delta += rad_delta;
                            peduncle_radius_count++;
                            peduncle_csv << ped_radii_orig[seg_idx] << "," << ped_radii_load[seg_idx] << "," << rad_delta;
                        } else {
                            peduncle_csv << "NA,NA,NA";
                        }
                        peduncle_csv << std::endl;
                    }
                }
            }

            // Compare flower/fruit actual geometry positions using bounding boxes
            size_t min_floral_petiole_count = std::min(phyt_orig->floral_buds.size(), phyt_load->floral_buds.size());
            for (size_t pet_idx = 0; pet_idx < min_floral_petiole_count; pet_idx++) {
                size_t min_bud_count = std::min(phyt_orig->floral_buds[pet_idx].size(), phyt_load->floral_buds[pet_idx].size());

                for (size_t bud_idx = 0; bud_idx < min_bud_count; bud_idx++) {
                    const auto &fbud_orig = phyt_orig->floral_buds[pet_idx][bud_idx];
                    const auto &fbud_load = phyt_load->floral_buds[pet_idx][bud_idx];

                    // Compare actual flower/fruit geometry using bounding box centers
                    size_t min_flower_count = std::min(fbud_orig.inflorescence_objIDs.size(), fbud_load.inflorescence_objIDs.size());
                    for (size_t flower_idx = 0; flower_idx < min_flower_count; flower_idx++) {
                        // Get bounding box center for each flower object
                        vec3 bbox_orig_min, bbox_orig_max;
                        context_ptr->getObjectBoundingBox(fbud_orig.inflorescence_objIDs[flower_idx], bbox_orig_min, bbox_orig_max);
                        vec3 center_orig = 0.5f * (bbox_orig_min + bbox_orig_max);

                        vec3 bbox_load_min, bbox_load_max;
                        context_ptr->getObjectBoundingBox(fbud_load.inflorescence_objIDs[flower_idx], bbox_load_min, bbox_load_max);
                        vec3 center_load = 0.5f * (bbox_load_min + bbox_load_max);

                        vec3 delta_vec = center_orig - center_load;
                        float pos_delta = delta_vec.magnitude();
                        max_flower_position_delta = std::max(max_flower_position_delta, pos_delta);
                        sum_flower_position_delta += pos_delta;
                        flower_count++;

                        flower_csv << shoot_orig->ID << "," << phyt_idx << "," << pet_idx << "," << bud_idx << "," << flower_idx << "," << center_orig.x << "," << center_orig.y << "," << center_orig.z << "," << center_load.x << "," << center_load.y
                                   << "," << center_load.z << "," << pos_delta << std::endl;
                    }
                }
            }
        }
    }

    internode_csv.close();
    petiole_csv.close();
    leaf_csv.close();
    peduncle_csv.close();
    flower_csv.close();

    // Print summary statistics
    std::cout << "\n=== GEOMETRY COMPARISON SUMMARY ===" << std::endl;
    std::cout << "\nINTERNODES:" << std::endl;
    if (internode_vertex_count > 0) {
        std::cout << "  Vertices compared: " << internode_vertex_count << std::endl;
        std::cout << "  Max position delta: " << max_internode_position_delta << " m" << std::endl;
        std::cout << "  Avg position delta: " << (sum_internode_position_delta / internode_vertex_count) << " m" << std::endl;
    } else {
        std::cout << "  No vertices to compare" << std::endl;
    }
    if (internode_radius_count > 0) {
        std::cout << "  Radii compared: " << internode_radius_count << std::endl;
        std::cout << "  Max radius delta: " << max_internode_radius_delta << " m" << std::endl;
        std::cout << "  Avg radius delta: " << (sum_internode_radius_delta / internode_radius_count) << " m" << std::endl;
    } else {
        std::cout << "  No radii to compare" << std::endl;
    }

    std::cout << "\nPETIOLES:" << std::endl;
    if (petiole_vertex_count > 0) {
        std::cout << "  Vertices compared: " << petiole_vertex_count << std::endl;
        std::cout << "  Max position delta: " << max_petiole_position_delta << " m" << std::endl;
        std::cout << "  Avg position delta: " << (sum_petiole_position_delta / petiole_vertex_count) << " m" << std::endl;
    } else {
        std::cout << "  No vertices to compare" << std::endl;
    }
    if (petiole_radius_count > 0) {
        std::cout << "  Radii compared: " << petiole_radius_count << std::endl;
        std::cout << "  Max radius delta: " << max_petiole_radius_delta << " m" << std::endl;
        std::cout << "  Avg radius delta: " << (sum_petiole_radius_delta / petiole_radius_count) << " m" << std::endl;
    } else {
        std::cout << "  No radii to compare" << std::endl;
    }

    std::cout << "\nLEAVES:" << std::endl;
    if (leaf_count > 0) {
        std::cout << "  Leaves compared: " << leaf_count << std::endl;
        std::cout << "  Max position delta: " << max_leaf_position_delta << " m" << std::endl;
        std::cout << "  Avg position delta: " << (sum_leaf_position_delta / leaf_count) << " m" << std::endl;
        std::cout << "  Max rotation delta: " << rad2deg(max_leaf_rotation_delta) << " deg" << std::endl;
        std::cout << "  Avg rotation delta: " << rad2deg(sum_leaf_rotation_delta / leaf_count) << " deg" << std::endl;
    } else {
        std::cout << "  No leaves to compare" << std::endl;
    }

    std::cout << "\nPEDUNCLES:" << std::endl;
    if (peduncle_vertex_count > 0) {
        std::cout << "  Vertices compared: " << peduncle_vertex_count << std::endl;
        std::cout << "  Max position delta: " << max_peduncle_position_delta << " m" << std::endl;
        std::cout << "  Avg position delta: " << (sum_peduncle_position_delta / peduncle_vertex_count) << " m" << std::endl;
    } else {
        std::cout << "  No vertices to compare" << std::endl;
    }
    if (peduncle_radius_count > 0) {
        std::cout << "  Radii compared: " << peduncle_radius_count << std::endl;
        std::cout << "  Max radius delta: " << max_peduncle_radius_delta << " m" << std::endl;
        std::cout << "  Avg radius delta: " << (sum_peduncle_radius_delta / peduncle_radius_count) << " m" << std::endl;
    } else {
        std::cout << "  No radii to compare" << std::endl;
    }

    std::cout << "\nFLOWERS/FRUIT:" << std::endl;
    if (flower_count > 0) {
        std::cout << "  Flower/fruit positions compared: " << flower_count << std::endl;
        std::cout << "  Max position delta: " << max_flower_position_delta << " m" << std::endl;
        std::cout << "  Avg position delta: " << (sum_flower_position_delta / flower_count) << " m" << std::endl;
    } else {
        std::cout << "  No flowers/fruit to compare" << std::endl;
    }

    std::cout << "\n--- Original Plant Shoot Structure ---" << std::endl;
    for (size_t i = 0; i < plant_orig.shoot_tree.size(); i++) {
        std::cout << "  Shoot " << plant_orig.shoot_tree[i]->ID << ": " << plant_orig.shoot_tree[i]->phytomers.size() << " phytomers" << std::endl;
        for (size_t p = 0; p < plant_orig.shoot_tree[i]->phytomers.size(); p++) {
            size_t total_fbuds = 0;
            for (size_t pet = 0; pet < plant_orig.shoot_tree[i]->phytomers[p]->floral_buds.size(); pet++) {
                total_fbuds += plant_orig.shoot_tree[i]->phytomers[p]->floral_buds[pet].size();
            }
            if (total_fbuds > 0) {
                std::cout << "    Phytomer " << p << " has " << total_fbuds << " floral buds" << std::endl;
            }
        }
    }

    std::cout << "\n--- Loaded Plant Shoot Structure ---" << std::endl;
    for (size_t i = 0; i < plant_load.shoot_tree.size(); i++) {
        std::cout << "  Shoot " << plant_load.shoot_tree[i]->ID << ": " << plant_load.shoot_tree[i]->phytomers.size() << " phytomers" << std::endl;
        for (size_t p = 0; p < plant_load.shoot_tree[i]->phytomers.size(); p++) {
            size_t total_fbuds = 0;
            for (size_t pet = 0; pet < plant_load.shoot_tree[i]->phytomers[p]->floral_buds.size(); pet++) {
                total_fbuds += plant_load.shoot_tree[i]->phytomers[p]->floral_buds[pet].size();
            }
            if (total_fbuds > 0) {
                std::cout << "    Phytomer " << p << " has " << total_fbuds << " floral buds" << std::endl;
            }
        }
    }

    std::cout << "\nDetailed comparison written to:" << std::endl;
    std::cout << "  " << output_prefix << "_internodes.csv" << std::endl;
    std::cout << "  " << output_prefix << "_petioles.csv" << std::endl;
    std::cout << "  " << output_prefix << "_leaves.csv" << std::endl;
    std::cout << "  " << output_prefix << "_peduncles.csv" << std::endl;
    std::cout << "  " << output_prefix << "_flowers.csv" << std::endl;
    std::cout << "===================================\n" << std::endl;
}

void PlantArchitecture::writePlantStructureXML(uint plantID, const std::string &filename) const {

    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureXML): Plant ID " + std::to_string(plantID) + " does not exist.");
    }

    std::string output_file = filename;
    if (!validateOutputPath(output_file, {".xml", ".XML"})) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureXML): Could not open file " + filename + " for writing. Make sure the directory exists and is writable.");
    } else if (getFileName(output_file).empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureXML): The output file given was a directory. This argument should be the path to a file not to a directory.");
    }

    std::ofstream output_xml(filename);

    if (!output_xml.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureXML): Could not open file " + filename + " for writing. Make sure the directory exists and is writable.");
    }

    output_xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    output_xml << "<helios>" << std::endl;
    output_xml << "\t<plant_instance ID=\"" << plantID << "\">" << std::endl;

    output_xml << "\t\t<base_position> " << plant_instances.at(plantID).base_position.x << " " << plant_instances.at(plantID).base_position.y << " " << plant_instances.at(plantID).base_position.z << " </base_position>" << std::endl;
    output_xml << "\t\t<plant_age> " << plant_instances.at(plantID).current_age << " </plant_age>" << std::endl;

    for (auto &shoot: plant_instances.at(plantID).shoot_tree) {

        output_xml << "\t\t<shoot ID=\"" << shoot->ID << "\">" << std::endl;
        output_xml << "\t\t\t<shoot_type_label> " << shoot->shoot_type_label << " </shoot_type_label>" << std::endl;
        output_xml << "\t\t\t<parent_shoot_ID> " << shoot->parent_shoot_ID << " </parent_shoot_ID>" << std::endl;
        output_xml << "\t\t\t<parent_node_index> " << shoot->parent_node_index << " </parent_node_index>" << std::endl;
        output_xml << "\t\t\t<parent_petiole_index> " << shoot->parent_petiole_index << " </parent_petiole_index>" << std::endl;
        output_xml << "\t\t\t<base_rotation> " << rad2deg(shoot->base_rotation.pitch) << " " << rad2deg(shoot->base_rotation.yaw) << " " << rad2deg(shoot->base_rotation.roll) << " </base_rotation>" << std::endl;

        uint phytomer_index = 0;
        for (auto &phytomer: shoot->phytomers) {

            output_xml << "\t\t\t<phytomer>" << std::endl;
            output_xml << "\t\t\t\t<internode>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_length>" << phytomer->getInternodeLength() << "</internode_length>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_radius>" << phytomer->getInternodeRadius() << "</internode_radius>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_pitch>" << rad2deg(phytomer->internode_pitch) << "</internode_pitch>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_phyllotactic_angle>" << rad2deg(phytomer->internode_phyllotactic_angle) << "</internode_phyllotactic_angle>" << std::endl;

            // Save all internode tube vertices
            if (phytomer_index < shoot->shoot_internode_vertices.size()) {
                const auto &vertices = shoot->shoot_internode_vertices[phytomer_index];
                output_xml << "\t\t\t\t\t<internode_vertices>";
                for (size_t v = 0; v < vertices.size(); v++) {
                    output_xml << vertices[v].x << " " << vertices[v].y << " " << vertices[v].z;
                    if (v < vertices.size() - 1)
                        output_xml << ";";
                }
                output_xml << "</internode_vertices>" << std::endl;
            }

            // Save all internode tube radii
            if (phytomer_index < shoot->shoot_internode_radii.size()) {
                const auto &radii = shoot->shoot_internode_radii[phytomer_index];
                output_xml << "\t\t\t\t\t<internode_radii>";
                for (size_t r = 0; r < radii.size(); r++) {
                    output_xml << radii[r];
                    if (r < radii.size() - 1)
                        output_xml << ";";
                }
                output_xml << "</internode_radii>" << std::endl;
            }

            for (uint petiole = 0; petiole < phytomer->petiole_length.size(); petiole++) {

                output_xml << "\t\t\t\t\t<petiole>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_length>" << phytomer->petiole_length.at(petiole) << "</petiole_length>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_radius>" << phytomer->petiole_radii.at(petiole).front() << "</petiole_radius>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_pitch>" << rad2deg(phytomer->petiole_pitch.at(petiole)) << "</petiole_pitch>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_curvature>" << phytomer->petiole_curvature.at(petiole) << "</petiole_curvature>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_base_position>" << phytomer->petiole_vertices.at(petiole).front().x << " " << phytomer->petiole_vertices.at(petiole).front().y << " " << phytomer->petiole_vertices.at(petiole).front().z
                           << "</petiole_base_position>" << std::endl;
                output_xml << "\t\t\t\t\t\t<current_leaf_scale_factor>" << phytomer->current_leaf_scale_factor.at(petiole) << "</current_leaf_scale_factor>" << std::endl;

                // Save all petiole tube vertices
                const auto &pet_vertices = phytomer->petiole_vertices.at(petiole);
                output_xml << "\t\t\t\t\t\t<petiole_vertices>";
                for (size_t v = 0; v < pet_vertices.size(); v++) {
                    output_xml << pet_vertices[v].x << " " << pet_vertices[v].y << " " << pet_vertices[v].z;
                    if (v < pet_vertices.size() - 1)
                        output_xml << ";";
                }
                output_xml << "</petiole_vertices>" << std::endl;

                // Save all petiole tube radii
                const auto &pet_radii = phytomer->petiole_radii.at(petiole);
                output_xml << "\t\t\t\t\t\t<petiole_radii>";
                for (size_t r = 0; r < pet_radii.size(); r++) {
                    output_xml << pet_radii[r];
                    if (r < pet_radii.size() - 1)
                        output_xml << ";";
                }
                output_xml << "</petiole_radii>" << std::endl;

                if (phytomer->leaf_rotation.at(petiole).size() == 1) { // not compound leaf
                    output_xml << "\t\t\t\t\t\t<leaflet_scale>" << 1.0 << "</leaflet_scale>" << std::endl;
                } else {
                    float tip_ind = floor(float(phytomer->leaf_rotation.at(petiole).size() - 1) / 2.f);
                    output_xml << "\t\t\t\t\t\t<leaflet_scale>" << phytomer->leaf_size_max.at(petiole).at(int(tip_ind - 1)) / max(phytomer->leaf_size_max.at(petiole)) << "</leaflet_scale>" << std::endl;
                }

                for (uint leaf = 0; leaf < phytomer->leaf_rotation.at(petiole).size(); leaf++) {
                    output_xml << "\t\t\t\t\t\t<leaf>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_scale>" << phytomer->leaf_size_max.at(petiole).at(leaf) * phytomer->current_leaf_scale_factor.at(petiole) << "</leaf_scale>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_base>" << phytomer->leaf_bases.at(petiole).at(leaf).x << " " << phytomer->leaf_bases.at(petiole).at(leaf).y << " " << phytomer->leaf_bases.at(petiole).at(leaf).z << "</leaf_base>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_pitch>" << rad2deg(phytomer->leaf_rotation.at(petiole).at(leaf).pitch) << "</leaf_pitch>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_yaw>" << rad2deg(phytomer->leaf_rotation.at(petiole).at(leaf).yaw) << "</leaf_yaw>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_roll>" << rad2deg(phytomer->leaf_rotation.at(petiole).at(leaf).roll) << "</leaf_roll>" << std::endl;
                    output_xml << "\t\t\t\t\t\t</leaf>" << std::endl;
                }

                // Write floral buds
                if (petiole < phytomer->floral_buds.size()) {
                    for (uint bud = 0; bud < phytomer->floral_buds.at(petiole).size(); bud++) {
                        const FloralBud &fbud = phytomer->floral_buds.at(petiole).at(bud);

                        output_xml << "\t\t\t\t\t\t<floral_bud>" << std::endl;

                        // Write bud state and indices
                        output_xml << "\t\t\t\t\t\t\t<bud_state>" << static_cast<int>(fbud.state) << "</bud_state>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t<parent_index>" << fbud.parent_index << "</parent_index>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t<bud_index>" << fbud.bud_index << "</bud_index>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t<is_terminal>" << (fbud.isterminal ? 1 : 0) << "</is_terminal>" << std::endl;

                        // Write position and rotation
                        output_xml << "\t\t\t\t\t\t\t<base_position>" << fbud.base_position.x << " " << fbud.base_position.y << " " << fbud.base_position.z << "</base_position>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t<base_rotation>" << rad2deg(fbud.base_rotation.pitch) << " " << rad2deg(fbud.base_rotation.yaw) << " " << rad2deg(fbud.base_rotation.roll) << "</base_rotation>" << std::endl;

                        // Write fruit scale factor
                        output_xml << "\t\t\t\t\t\t\t<current_fruit_scale_factor>" << fbud.current_fruit_scale_factor << "</current_fruit_scale_factor>" << std::endl;

                        // Write peduncle parameters
                        // Note: Write the CURRENT VALUES from phytomer_parameters, which are the actual sampled values for this phytomer
                        // These get resampled each time so we need to save the actual values used
                        output_xml << "\t\t\t\t\t\t\t<peduncle>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t\t<length>" << phytomer->phytomer_parameters.peduncle.length.val() << "</length>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t\t<radius>" << phytomer->phytomer_parameters.peduncle.radius.val() << "</radius>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t\t<pitch>" << phytomer->phytomer_parameters.peduncle.pitch.val() << "</pitch>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t\t<roll>" << phytomer->phytomer_parameters.peduncle.roll.val() << "</roll>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t\t<curvature>" << phytomer->phytomer_parameters.peduncle.curvature.val() << "</curvature>" << std::endl;

                        // Write actual peduncle tube vertices and radii for exact geometry reconstruction (like petioles)
                        if (petiole < phytomer->peduncle_vertices.size() && bud < phytomer->peduncle_vertices.at(petiole).size()) {
                            const auto &ped_vertices = phytomer->peduncle_vertices.at(petiole).at(bud);
                            if (!ped_vertices.empty()) {
                                output_xml << "\t\t\t\t\t\t\t\t<peduncle_vertices>";
                                for (size_t v = 0; v < ped_vertices.size(); v++) {
                                    output_xml << ped_vertices[v].x << " " << ped_vertices[v].y << " " << ped_vertices[v].z;
                                    if (v < ped_vertices.size() - 1)
                                        output_xml << ";";
                                }
                                output_xml << "</peduncle_vertices>" << std::endl;
                            }
                        }

                        if (petiole < phytomer->peduncle_radii.size() && bud < phytomer->peduncle_radii.at(petiole).size()) {
                            const auto &ped_radii = phytomer->peduncle_radii.at(petiole).at(bud);
                            if (!ped_radii.empty()) {
                                output_xml << "\t\t\t\t\t\t\t\t<peduncle_radii>";
                                for (size_t r = 0; r < ped_radii.size(); r++) {
                                    output_xml << ped_radii[r];
                                    if (r < ped_radii.size() - 1)
                                        output_xml << ";";
                                }
                                output_xml << "</peduncle_radii>" << std::endl;
                            }
                        }

                        output_xml << "\t\t\t\t\t\t\t</peduncle>" << std::endl;

                        // Write inflorescence parameters
                        output_xml << "\t\t\t\t\t\t\t<inflorescence>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t\t<flower_offset>" << phytomer->phytomer_parameters.inflorescence.flower_offset.val() << "</flower_offset>" << std::endl;

                        // Write individual flower/fruit positions and rotations
                        for (uint i = 0; i < fbud.inflorescence_bases.size(); i++) {
                            output_xml << "\t\t\t\t\t\t\t\t<flower>" << std::endl;
                            // Save the base attachment point on the peduncle
                            output_xml << "\t\t\t\t\t\t\t\t\t<inflorescence_base>" << fbud.inflorescence_bases.at(i).x << " " << fbud.inflorescence_bases.at(i).y << " " << fbud.inflorescence_bases.at(i).z << "</inflorescence_base>" << std::endl;
                            // Save pitch, yaw, roll, and azimuth for this flower/fruit
                            if (i < fbud.inflorescence_rotation.size()) {
                                output_xml << "\t\t\t\t\t\t\t\t\t<flower_pitch>" << rad2deg(fbud.inflorescence_rotation.at(i).pitch) << "</flower_pitch>" << std::endl;
                                output_xml << "\t\t\t\t\t\t\t\t\t<flower_yaw>" << rad2deg(fbud.inflorescence_rotation.at(i).yaw) << "</flower_yaw>" << std::endl;
                                output_xml << "\t\t\t\t\t\t\t\t\t<flower_roll>" << rad2deg(fbud.inflorescence_rotation.at(i).roll) << "</flower_roll>" << std::endl;
                                output_xml << "\t\t\t\t\t\t\t\t\t<flower_azimuth>" << rad2deg(fbud.inflorescence_rotation.at(i).azimuth) << "</flower_azimuth>" << std::endl;
                            }
                            // Save individual base scale for this flower/fruit
                            if (i < fbud.inflorescence_base_scales.size()) {
                                output_xml << "\t\t\t\t\t\t\t\t\t<flower_base_scale>" << fbud.inflorescence_base_scales.at(i) << "</flower_base_scale>" << std::endl;
                            }
                            output_xml << "\t\t\t\t\t\t\t\t</flower>" << std::endl;
                        }

                        output_xml << "\t\t\t\t\t\t\t</inflorescence>" << std::endl;
                        output_xml << "\t\t\t\t\t\t</floral_bud>" << std::endl;
                    }
                }

                output_xml << "\t\t\t\t\t</petiole>" << std::endl;
            }
            output_xml << "\t\t\t\t</internode>" << std::endl;
            output_xml << "\t\t\t</phytomer>" << std::endl;

            phytomer_index++;
        }
        output_xml << "\t\t</shoot>" << std::endl;
    }
    output_xml << "\t</plant_instance>" << std::endl;
    output_xml << "</helios>" << std::endl;
    output_xml.close();
}

void PlantArchitecture::writeQSMCylinderFile(uint plantID, const std::string &filename) const {

    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::writeQSMCylinderFile): Plant ID " + std::to_string(plantID) + " does not exist.");
    }

    std::string output_file = filename;
    if (!validateOutputPath(output_file, {".txt", ".TXT"})) {
        helios_runtime_error("ERROR (PlantArchitecture::writeQSMCylinderFile): Could not open file " + filename + " for writing. Make sure the directory exists and is writable.");
    } else if (getFileName(output_file).empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::writeQSMCylinderFile): The output file given was a directory. This argument should be the path to a file not to a directory.");
    }

    std::ofstream output_qsm(filename);

    if (!output_qsm.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::writeQSMCylinderFile): Could not open file " + filename + " for writing. Make sure the directory exists and is writable.");
    }

    // Write header line
    output_qsm << "radius (m)\tlength (m)\tstart_point\taxis_direction\tparent\textension\tbranch\tbranch_order\tposition_in_branch\tmad\tSurfCov\tadded\tUnmodRadius (m)" << std::endl;

    const auto &plant = plant_instances.at(plantID);

    // Cylinder ID counter (row number = cylinder ID in TreeQSM format)
    uint cylinder_id = 1;

    // Maps to track relationships between shoots and cylinders
    std::map<int, std::vector<uint>> shoot_cylinder_ids; // shoot ID -> cylinder IDs
    std::map<int, uint> shoot_branch_id; // shoot ID -> branch ID
    std::map<int, uint> shoot_branch_order; // shoot ID -> branch order

    // Assign branch IDs and orders to shoots
    uint branch_id_counter = 1;
    for (const auto &shoot: plant.shoot_tree) {
        shoot_branch_id[shoot->ID] = branch_id_counter++;

        // Determine branch order based on parent
        if (shoot->parent_shoot_ID == -1) {
            // Base shoot is order 0 (trunk)
            shoot_branch_order[shoot->ID] = 0;
        } else {
            // Child shoot has order = parent order + 1
            shoot_branch_order[shoot->ID] = shoot_branch_order[shoot->parent_shoot_ID] + 1;
        }
    }

    // Process each shoot
    for (const auto &shoot: plant.shoot_tree) {

        // Get shoot properties
        uint branch_id = shoot_branch_id[shoot->ID];
        uint branch_order = shoot_branch_order[shoot->ID];
        uint position_in_branch = 1;
        // Track the last vertex position for vertex sharing between phytomers
        helios::vec3 last_vertex_position;
        bool has_last_vertex = false;

        // Process each phytomer in the shoot
        for (uint phytomer_idx = 0; phytomer_idx < shoot->phytomers.size(); phytomer_idx++) {

            const auto &vertices = shoot->shoot_internode_vertices[phytomer_idx];
            const auto &radii = shoot->shoot_internode_radii[phytomer_idx];

            // Handle vertex sharing for single-segment phytomer tubes
            if (vertices.size() == 1 && has_last_vertex) {
                // This phytomer has only one vertex - use the last vertex from previous phytomer as start
                helios::vec3 start = last_vertex_position;
                helios::vec3 current_end = vertices[0];
                float current_radius = radii[0];

                // Calculate initial length
                helios::vec3 axis = current_end - start;
                float length = axis.magnitude();

                axis = axis / length; // Normalize axis

                // Process this as a single cylinder
                // Determine parent cylinder ID
                uint parent_id = 0;
                if (cylinder_id > 1) {
                    parent_id = cylinder_id - 1; // Parent is previous cylinder
                }

                // Extension cylinder (next cylinder in same branch) - will be updated later if needed
                uint extension_id = 0;

                // Write cylinder data
                output_qsm << std::fixed << std::setprecision(4);
                output_qsm << current_radius << "\t";
                output_qsm << length << "\t";
                output_qsm << start.x << "\t" << start.y << "\t" << start.z << "\t";
                output_qsm << axis.x << "\t" << axis.y << "\t" << axis.z << "\t";
                output_qsm << parent_id << "\t";
                output_qsm << extension_id << "\t";
                output_qsm << branch_id << "\t";
                output_qsm << branch_order << "\t";
                output_qsm << position_in_branch << "\t";
                output_qsm << "0.0002" << "\t"; // mad (using default value)
                output_qsm << "1" << "\t"; // SurfCov (using default value)
                output_qsm << "0" << "\t"; // added flag
                output_qsm << current_radius << std::endl; // UnmodRadius

                // Store cylinder ID for this shoot
                shoot_cylinder_ids[shoot->ID].push_back(cylinder_id);

                cylinder_id++;
                position_in_branch++;

                // Update last vertex for next phytomer
                last_vertex_position = current_end;

            } else {
                // Normal processing for phytomers with multiple vertices
                for (int seg = 0; seg < vertices.size() - 1; seg++) {

                    // Start with the current segment
                    helios::vec3 start = vertices[seg];
                    helios::vec3 current_end = vertices[seg + 1];
                    float current_radius = radii[seg];

                    // Calculate initial length
                    helios::vec3 axis = current_end - start;
                    float length = axis.magnitude();

                    axis = axis / length; // Normalize axis

                    // Determine parent cylinder ID
                    uint parent_id = 0;
                    if (cylinder_id > 1) {
                        if (seg == 0 && phytomer_idx == 0 && shoot->parent_shoot_ID != -1) {
                            // First cylinder of child shoot - parent is last cylinder of connection point
                            // For simplicity, using previous cylinder as parent
                            parent_id = cylinder_id - 1;
                        } else if (seg == 0 && phytomer_idx > 0) {
                            // First segment of new phytomer - parent is last segment of previous phytomer
                            parent_id = cylinder_id - 1;
                        } else {
                            // Continuation within phytomer - parent is previous segment
                            parent_id = cylinder_id - 1;
                        }
                    }

                    // Extension cylinder (next cylinder in same branch) - will be updated later if needed
                    uint extension_id = 0;

                    // Write cylinder data
                    output_qsm << std::fixed << std::setprecision(4);
                    output_qsm << current_radius << "\t";
                    output_qsm << length << "\t";
                    output_qsm << start.x << "\t" << start.y << "\t" << start.z << "\t";
                    output_qsm << axis.x << "\t" << axis.y << "\t" << axis.z << "\t";
                    output_qsm << parent_id << "\t";
                    output_qsm << extension_id << "\t";
                    output_qsm << branch_id << "\t";
                    output_qsm << branch_order << "\t";
                    output_qsm << position_in_branch << "\t";
                    output_qsm << "0.0002" << "\t"; // mad (using default value)
                    output_qsm << "1" << "\t"; // SurfCov (using default value)
                    output_qsm << "0" << "\t"; // added flag
                    output_qsm << current_radius << std::endl; // UnmodRadius

                    // Store cylinder ID for this shoot
                    shoot_cylinder_ids[shoot->ID].push_back(cylinder_id);

                    cylinder_id++;
                    position_in_branch++;
                }

                // Update last vertex position for vertex sharing
                if (vertices.size() >= 2) {
                    last_vertex_position = vertices.back();
                    has_last_vertex = true;
                } else if (vertices.size() == 1) {
                    // This shouldn't happen in normal processing, but handle it for completeness
                    last_vertex_position = vertices[0];
                    has_last_vertex = true;
                }
            }
        }
    }

    output_qsm.close();
}

std::vector<uint> PlantArchitecture::readPlantStructureXML(const std::string &filename, bool quiet) {

    if (!quiet) {
        std::cout << "Loading plant architecture XML file: " << filename << "..." << std::flush;
    }

    std::string fn = filename;
    std::string ext = getFileExtension(filename);
    if (ext != ".xml" && ext != ".XML") {
        helios_runtime_error("failed.\n File " + fn + " is not XML format.");
    }

    std::vector<uint> plantIDs;

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    // Resolve file path using project-based resolution
    std::filesystem::path resolved_path = resolveProjectFile(filename);
    std::string resolved_filename = resolved_path.string();

    // load file
    pugi::xml_parse_result load_result = xmldoc.load_file(resolved_filename.c_str());

    // error checking
    if (!load_result) {
        helios_runtime_error("ERROR (Context::readPlantStructureXML): Could not parse " + std::string(filename) + ":\nError description: " + load_result.description());
    }

    pugi::xml_node helios = xmldoc.child("helios");

    pugi::xml_node node;
    std::string node_string;

    if (helios.empty()) {
        if (!quiet) {
            std::cout << "failed." << std::endl;
        }
        helios_runtime_error("ERROR (Context::readPlantStructureXML): XML file must have tag '<helios> ... </helios>' bounding all other tags.");
    }

    size_t phytomer_count = 0;

    std::map<int, int> shoot_ID_mapping;

    for (pugi::xml_node plant = helios.child("plant_instance"); plant; plant = plant.next_sibling("plant_instance")) {

        int plantID = std::stoi(plant.attribute("ID").value());

        // base position
        node_string = "base_position";
        vec3 base_position = parse_xml_tag_vec3(plant.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

        // plant age
        node_string = "plant_age";
        float plant_age = parse_xml_tag_float(plant.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

        plantID = addPlantInstance(base_position, plant_age);
        plantIDs.push_back(plantID);

        int current_shoot_ID;

        for (pugi::xml_node shoot = plant.child("shoot"); shoot; shoot = shoot.next_sibling("shoot")) {

            int shootID = std::stoi(shoot.attribute("ID").value());
            bool base_shoot = true;

            // shoot type
            node_string = "shoot_type_label";
            std::string shoot_type_label = parse_xml_tag_string(shoot.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

            // parent shoot ID
            node_string = "parent_shoot_ID";
            int parent_shoot_ID = parse_xml_tag_int(shoot.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

            // parent node index
            node_string = "parent_node_index";
            int parent_node_index = parse_xml_tag_int(shoot.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

            // parent petiole index
            node_string = "parent_petiole_index";
            int parent_petiole_index = parse_xml_tag_int(shoot.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

            // base rotation
            node_string = "base_rotation";
            vec3 base_rot = parse_xml_tag_vec3(shoot.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
            AxisRotation base_rotation(deg2rad(base_rot.x), deg2rad(base_rot.y), deg2rad(base_rot.z));

            for (pugi::xml_node phytomer = shoot.child("phytomer"); phytomer; phytomer = phytomer.next_sibling("phytomer")) {

                pugi::xml_node internode = phytomer.child("internode");

                // internode length
                node_string = "internode_length";
                float internode_length = parse_xml_tag_float(internode.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                // internode radius
                node_string = "internode_radius";
                float internode_radius = parse_xml_tag_float(internode.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                // internode pitch
                node_string = "internode_pitch";
                float internode_pitch = parse_xml_tag_float(internode.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                // internode phyllotactic angle
                node_string = "internode_phyllotactic_angle";
                float internode_phyllotactic_angle = parse_xml_tag_float(internode.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                // Read internode vertices if available (optional for backward compatibility)
                std::vector<vec3> internode_vertices;
                node_string = "internode_vertices";
                if (internode.child(node_string.c_str())) {
                    std::string vertices_str = internode.child_value(node_string.c_str());
                    std::istringstream verts_stream(vertices_str);
                    std::string vertex_str;
                    while (std::getline(verts_stream, vertex_str, ';')) {
                        std::istringstream vertex_coords(vertex_str);
                        float x, y, z;
                        if (vertex_coords >> x >> y >> z) {
                            internode_vertices.push_back(make_vec3(x, y, z));
                        }
                    }
                }

                // Read internode radii if available (optional for backward compatibility)
                std::vector<float> internode_radii;
                node_string = "internode_radii";
                if (internode.child(node_string.c_str())) {
                    std::string radii_str = internode.child_value(node_string.c_str());
                    std::istringstream radii_stream(radii_str);
                    std::string radius_str;
                    while (std::getline(radii_stream, radius_str, ';')) {
                        float radius = std::stof(radius_str);
                        internode_radii.push_back(radius);
                    }
                }

                float petiole_length;
                float petiole_radius;
                float petiole_pitch;
                float petiole_curvature;
                float current_leaf_scale_factor_value;
                float leaflet_scale;
                std::vector<float> petiole_lengths; // actual length of each petiole within internode
                std::vector<float> petiole_radii_values; // actual radius of each petiole within internode
                std::vector<float> petiole_pitches; // pitch of each petiole within internode
                std::vector<float> petiole_curvatures; // curvature of each petiole within internode
                std::vector<vec3> petiole_base_positions; // actual base position of each petiole within internode
                std::vector<float> current_leaf_scale_factors; // scale factor of each petiole within internode
                std::vector<std::vector<vec3>> petiole_all_vertices; // all vertices for each petiole (if saved in XML)
                std::vector<std::vector<float>> petiole_all_radii; // all radii for each petiole (if saved in XML)
                std::vector<std::vector<vec3>> saved_leaf_bases_all_petioles; // saved leaf attachment positions for each petiole (if saved in XML)
                std::vector<std::vector<float>> leaf_scale; // first index is petiole within internode; second index is leaf within petiole
                std::vector<std::vector<float>> leaf_pitch;
                std::vector<std::vector<float>> leaf_yaw;
                std::vector<std::vector<float>> leaf_roll;

                // Floral bud data structures
                struct FloralBudData {
                    int bud_state;
                    uint parent_index;
                    uint bud_index;
                    bool is_terminal;
                    vec3 base_position;
                    AxisRotation base_rotation;
                    float current_fruit_scale_factor;
                    std::vector<vec3> inflorescence_bases_saved; // saved attachment points on peduncle
                    std::vector<vec3> flower_positions; // saved flower/fruit center positions
                    std::vector<AxisRotation> flower_rotations; // pitch, yaw, roll for each flower/fruit
                    std::vector<float> flower_base_scales; // individual base scale for each flower/fruit
                    // Inflorescence parameters
                    float flower_offset = -1;
                    // Peduncle parameters (actual sampled values)
                    float peduncle_length = -1;
                    float peduncle_radius = -1;
                    float peduncle_pitch = 0;
                    float peduncle_roll = 0;
                    float peduncle_curvature = 0;
                    // Peduncle geometry (saved vertices and radii for deterministic reconstruction)
                    std::vector<vec3> peduncle_vertices_saved;
                    std::vector<float> peduncle_radii_saved;
                };
                std::vector<std::vector<FloralBudData>> floral_bud_data; // first index is petiole within internode; second index is bud within petiole
                for (pugi::xml_node petiole = internode.child("petiole"); petiole; petiole = petiole.next_sibling("petiole")) {

                    // petiole length
                    node_string = "petiole_length";
                    petiole_length = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // petiole radius
                    node_string = "petiole_radius";
                    petiole_radius = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // petiole pitch
                    node_string = "petiole_pitch";
                    petiole_pitch = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // petiole curvature
                    node_string = "petiole_curvature";
                    petiole_curvature = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // petiole base position (optional for backward compatibility)
                    vec3 petiole_base_pos = nullorigin;
                    node_string = "petiole_base_position";
                    if (petiole.child(node_string.c_str())) {
                        petiole_base_pos = parse_xml_tag_vec3(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                    }

                    // current leaf scale factor
                    node_string = "current_leaf_scale_factor";
                    current_leaf_scale_factor_value = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // leaflet scale factor
                    node_string = "leaflet_scale";
                    leaflet_scale = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // Read petiole vertices if available (optional for backward compatibility)
                    std::vector<vec3> pet_vertices;
                    node_string = "petiole_vertices";
                    if (petiole.child(node_string.c_str())) {
                        std::string vertices_str = petiole.child_value(node_string.c_str());
                        std::istringstream verts_stream(vertices_str);
                        std::string vertex_str;
                        while (std::getline(verts_stream, vertex_str, ';')) {
                            std::istringstream vertex_coords(vertex_str);
                            float x, y, z;
                            if (vertex_coords >> x >> y >> z) {
                                pet_vertices.push_back(make_vec3(x, y, z));
                            }
                        }
                    }

                    // Read petiole radii if available (optional for backward compatibility)
                    std::vector<float> pet_radii;
                    node_string = "petiole_radii";
                    if (petiole.child(node_string.c_str())) {
                        std::string radii_str = petiole.child_value(node_string.c_str());
                        std::istringstream radii_stream(radii_str);
                        std::string radius_str;
                        while (std::getline(radii_stream, radius_str, ';')) {
                            float radius = std::stof(radius_str);
                            pet_radii.push_back(radius);
                        }
                    }

                    // Store petiole properties in vectors
                    petiole_lengths.push_back(petiole_length);
                    petiole_radii_values.push_back(petiole_radius);
                    petiole_pitches.push_back(petiole_pitch);
                    petiole_curvatures.push_back(petiole_curvature);
                    current_leaf_scale_factors.push_back(current_leaf_scale_factor_value);
                    petiole_base_positions.push_back(petiole_base_pos);
                    petiole_all_vertices.push_back(pet_vertices);
                    petiole_all_radii.push_back(pet_radii);

                    leaf_scale.resize(leaf_scale.size() + 1);
                    leaf_pitch.resize(leaf_pitch.size() + 1);
                    leaf_yaw.resize(leaf_yaw.size() + 1);
                    leaf_roll.resize(leaf_roll.size() + 1);
                    floral_bud_data.resize(floral_bud_data.size() + 1); // Always resize to stay in sync with leaf vectors

                    std::vector<vec3> saved_leaf_bases; // Saved leaf attachment positions

                    for (pugi::xml_node leaf = petiole.child("leaf"); leaf; leaf = leaf.next_sibling("leaf")) {

                        // leaf scale factor
                        node_string = "leaf_scale";
                        leaf_scale.back().push_back(parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML"));

                        // leaf base position
                        node_string = "leaf_base";
                        if (leaf.child(node_string.c_str())) {
                            std::string base_str = leaf.child_value(node_string.c_str());
                            std::istringstream base_stream(base_str);
                            float x, y, z;
                            if (base_stream >> x >> y >> z) {
                                saved_leaf_bases.push_back(make_vec3(x, y, z));
                            }
                        }

                        // leaf pitch
                        node_string = "leaf_pitch";
                        leaf_pitch.back().push_back(parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML"));

                        // leaf yaw
                        node_string = "leaf_yaw";
                        leaf_yaw.back().push_back(parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML"));

                        // leaf roll
                        node_string = "leaf_roll";
                        leaf_roll.back().push_back(parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML"));
                    }

                    // Store saved_leaf_bases for this petiole
                    saved_leaf_bases_all_petioles.push_back(saved_leaf_bases);

                    // Read floral buds (if present)
                    for (pugi::xml_node floral_bud = petiole.child("floral_bud"); floral_bud; floral_bud = floral_bud.next_sibling("floral_bud")) {

                        FloralBudData fbud_data;

                        // Read bud state
                        node_string = "bud_state";
                        fbud_data.bud_state = parse_xml_tag_int(floral_bud.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                        // Read parent index
                        node_string = "parent_index";
                        fbud_data.parent_index = parse_xml_tag_int(floral_bud.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                        // Read bud index
                        node_string = "bud_index";
                        fbud_data.bud_index = parse_xml_tag_int(floral_bud.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                        // Read is_terminal
                        node_string = "is_terminal";
                        int is_terminal = parse_xml_tag_int(floral_bud.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                        fbud_data.is_terminal = (is_terminal != 0);

                        // Read base position
                        node_string = "base_position";
                        fbud_data.base_position = parse_xml_tag_vec3(floral_bud.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                        // Read base rotation
                        node_string = "base_rotation";
                        vec3 base_rot = parse_xml_tag_vec3(floral_bud.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                        fbud_data.base_rotation = AxisRotation(deg2rad(base_rot.x), deg2rad(base_rot.y), deg2rad(base_rot.z));

                        // Read current fruit scale factor
                        node_string = "current_fruit_scale_factor";
                        fbud_data.current_fruit_scale_factor = parse_xml_tag_float(floral_bud.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                        // Read peduncle parameters (if present)
                        pugi::xml_node peduncle = floral_bud.child("peduncle");
                        if (peduncle) {
                            node_string = "length";
                            if (peduncle.child(node_string.c_str())) {
                                fbud_data.peduncle_length = parse_xml_tag_float(peduncle.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                            }
                            node_string = "radius";
                            if (peduncle.child(node_string.c_str())) {
                                fbud_data.peduncle_radius = parse_xml_tag_float(peduncle.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                            }
                            node_string = "pitch";
                            if (peduncle.child(node_string.c_str())) {
                                fbud_data.peduncle_pitch = parse_xml_tag_float(peduncle.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                            }
                            node_string = "roll";
                            if (peduncle.child(node_string.c_str())) {
                                fbud_data.peduncle_roll = parse_xml_tag_float(peduncle.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                            }
                            node_string = "curvature";
                            if (peduncle.child(node_string.c_str())) {
                                fbud_data.peduncle_curvature = parse_xml_tag_float(peduncle.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                            }

                            // Read peduncle vertices if available (for exact geometry reconstruction)
                            node_string = "peduncle_vertices";
                            if (peduncle.child(node_string.c_str())) {
                                std::string vertices_str = peduncle.child_value(node_string.c_str());
                                std::istringstream vertex_stream(vertices_str);
                                std::string vertex_str;
                                while (std::getline(vertex_stream, vertex_str, ';')) {
                                    std::istringstream vertex_coords(vertex_str);
                                    float x, y, z;
                                    if (vertex_coords >> x >> y >> z) {
                                        fbud_data.peduncle_vertices_saved.push_back(make_vec3(x, y, z));
                                    }
                                }
                            }

                            // Read peduncle radii if available (for exact geometry reconstruction)
                            node_string = "peduncle_radii";
                            if (peduncle.child(node_string.c_str())) {
                                std::string radii_str = peduncle.child_value(node_string.c_str());
                                std::istringstream radii_stream(radii_str);
                                std::string radius_str;
                                while (std::getline(radii_stream, radius_str, ';')) {
                                    float radius = std::stof(radius_str);
                                    fbud_data.peduncle_radii_saved.push_back(radius);
                                }
                            }
                        }

                        // Read inflorescence parameters and flower positions (if present)
                        pugi::xml_node inflorescence = floral_bud.child("inflorescence");
                        if (inflorescence) {
                            // Read inflorescence parameters
                            node_string = "flower_offset";
                            if (inflorescence.child(node_string.c_str())) {
                                fbud_data.flower_offset = parse_xml_tag_float(inflorescence.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                            }

                            // Read individual flower/fruit data (base position and rotations)
                            for (pugi::xml_node flower = inflorescence.child("flower"); flower; flower = flower.next_sibling("flower")) {
                                // Read base attachment point on peduncle
                                pugi::xml_node base_node = flower.child("inflorescence_base");
                                if (base_node) {
                                    vec3 base = parse_xml_tag_vec3(base_node, "inflorescence_base", "PlantArchitecture::readPlantStructureXML");
                                    fbud_data.inflorescence_bases_saved.push_back(base);
                                }

                                // Read pitch, yaw, roll, and azimuth
                                AxisRotation rotation;
                                pugi::xml_node pitch_node = flower.child("flower_pitch");
                                pugi::xml_node yaw_node = flower.child("flower_yaw");
                                pugi::xml_node roll_node = flower.child("flower_roll");
                                pugi::xml_node azimuth_node = flower.child("flower_azimuth");

                                if (pitch_node) {
                                    rotation.pitch = parse_xml_tag_float(pitch_node, "flower_pitch", "PlantArchitecture::readPlantStructureXML");
                                } else {
                                    rotation.pitch = 0;
                                }

                                if (yaw_node) {
                                    rotation.yaw = parse_xml_tag_float(yaw_node, "flower_yaw", "PlantArchitecture::readPlantStructureXML");
                                } else {
                                    rotation.yaw = 0;
                                }

                                if (roll_node) {
                                    rotation.roll = parse_xml_tag_float(roll_node, "flower_roll", "PlantArchitecture::readPlantStructureXML");
                                } else {
                                    rotation.roll = 0;
                                }

                                if (azimuth_node) {
                                    rotation.azimuth = parse_xml_tag_float(azimuth_node, "flower_azimuth", "PlantArchitecture::readPlantStructureXML");
                                } else {
                                    rotation.azimuth = 0;
                                }

                                fbud_data.flower_rotations.push_back(rotation);

                                // Read individual base scale for this flower/fruit
                                pugi::xml_node scale_node = flower.child("flower_base_scale");
                                if (scale_node) {
                                    float base_scale = parse_xml_tag_float(scale_node, "flower_base_scale", "PlantArchitecture::readPlantStructureXML");
                                    fbud_data.flower_base_scales.push_back(base_scale);
                                } else {
                                    // For backward compatibility with old XML files, use -1 to indicate not set
                                    fbud_data.flower_base_scales.push_back(-1.0f);
                                }
                            }
                        }

                        floral_bud_data.back().push_back(fbud_data);
                    }
                } // petioles

                if (shoot_types.find(shoot_type_label) == shoot_types.end()) {
                    helios_runtime_error("ERROR (PlantArchitecture::readPlantStructureXML): Shoot type " + shoot_type_label + " not found in shoot types.");
                }


                ShootParameters shoot_parameters = getCurrentShootParameters(shoot_type_label);

                shoot_parameters.phytomer_parameters.phytomer_creation_function = nullptr;

                shoot_parameters.phytomer_parameters.internode.pitch = internode_pitch;
                shoot_parameters.phytomer_parameters.internode.phyllotactic_angle = internode_phyllotactic_angle;

                shoot_parameters.phytomer_parameters.petiole.length = petiole_length;
                shoot_parameters.phytomer_parameters.petiole.radius = petiole_radius;
                shoot_parameters.phytomer_parameters.petiole.pitch = petiole_pitch;
                shoot_parameters.phytomer_parameters.petiole.curvature = petiole_curvature;

                shoot_parameters.phytomer_parameters.leaf.prototype_scale = 1.f; // leaf_scale.front().at(tip_ind);
                shoot_parameters.phytomer_parameters.leaf.pitch = 0;
                shoot_parameters.phytomer_parameters.leaf.yaw = 0;
                shoot_parameters.phytomer_parameters.leaf.roll = 0;
                shoot_parameters.phytomer_parameters.leaf.leaflet_scale = leaflet_scale;

                if (base_shoot) {

                    if (parent_shoot_ID < 0) { // this is the first shoot of the plant
                        current_shoot_ID = addBaseStemShoot(plantID, 1, base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, shoot_type_label);
                        shoot_ID_mapping[shootID] = current_shoot_ID;
                    } else { // this is a child of an existing shoot
                        current_shoot_ID = addChildShoot(plantID, shoot_ID_mapping.at(parent_shoot_ID), parent_node_index, 1, base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, shoot_type_label, parent_petiole_index);
                        shoot_ID_mapping[shootID] = current_shoot_ID;
                    }

                    base_shoot = false;
                } else {
                    appendPhytomerToShoot(plantID, current_shoot_ID, shoot_parameters.phytomer_parameters, internode_radius, internode_length, 1, 1);
                }

                // Get pointer to the newly created phytomer
                auto phytomer_ptr = plant_instances.at(plantID).shoot_tree.at(current_shoot_ID)->phytomers.back();

                // Get shoot pointer for internode geometry restoration
                auto shoot_ptr = plant_instances.at(plantID).shoot_tree.at(current_shoot_ID);
                uint phytomer_index_in_shoot = shoot_ptr->phytomers.size() - 1;

                // Restore internode vertices and radii from saved values if available
                if (!internode_vertices.empty() && !internode_radii.empty()) {
                    // Replace the reconstructed internode geometry data with saved geometry
                    // Note: We only update the data structures here, not the Context geometry
                    // All internodes in a shoot form a single continuous tube object
                    shoot_ptr->shoot_internode_vertices[phytomer_index_in_shoot] = internode_vertices;
                    shoot_ptr->shoot_internode_radii[phytomer_index_in_shoot] = internode_radii;
                }

                // Restore petiole properties and geometry from saved values
                for (size_t p = 0; p < petiole_lengths.size(); p++) {
                    if (p < phytomer_ptr->petiole_length.size()) {

                        // Check if we have complete saved geometry for this petiole
                        bool has_saved_geometry = (p < petiole_all_vertices.size() && !petiole_all_vertices[p].empty() && p < petiole_all_radii.size() && !petiole_all_radii[p].empty());

                        if (has_saved_geometry) {
                            // Calculate position correction based on petiole TIP (where leaves attach), not base
                            vec3 current_tip = phytomer_ptr->petiole_vertices[p].back(); // last vertex
                            vec3 saved_tip = petiole_all_vertices[p].back(); // last vertex from saved data
                            vec3 tip_correction = saved_tip - current_tip;

                            // Replace reconstructed geometry with saved geometry
                            phytomer_ptr->petiole_vertices[p] = petiole_all_vertices[p];
                            phytomer_ptr->petiole_radii[p] = petiole_all_radii[p];

                            // Rebuild the petiole Context geometry with saved geometry
                            context_ptr->deleteObject(phytomer_ptr->petiole_objIDs[p]);
                            std::vector<RGBcolor> petiole_colors(phytomer_ptr->petiole_radii[p].size(), phytomer_ptr->phytomer_parameters.petiole.color);
                            uint Ndiv_petiole_radius = std::max(uint(3), phytomer_ptr->phytomer_parameters.petiole.radial_subdivisions);
                            phytomer_ptr->petiole_objIDs[p] = makeTubeFromCones(Ndiv_petiole_radius, phytomer_ptr->petiole_vertices[p], phytomer_ptr->petiole_radii[p], petiole_colors, context_ptr);

                            // Restore primitive data labels
                            if (!phytomer_ptr->petiole_objIDs[p].empty()) {
                                context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(phytomer_ptr->petiole_objIDs[p]), "object_label", "petiole");
                            }

                            // Restore saved leaf base positions in data structure (leaves will be deleted and recreated with these positions)
                            // For compound leaves, lateral leaflets attach at different points along petiole
                            if (p < saved_leaf_bases_all_petioles.size() && !saved_leaf_bases_all_petioles[p].empty()) {
                                // Use saved leaf bases directly
                                for (size_t leaf = 0; leaf < phytomer_ptr->leaf_bases[p].size(); leaf++) {
                                    if (leaf < saved_leaf_bases_all_petioles[p].size()) {
                                        phytomer_ptr->leaf_bases[p][leaf] = saved_leaf_bases_all_petioles[p][leaf];
                                    }
                                }
                            } else {
                                // Fallback: update leaf bases by tip correction (for backward compatibility)
                                for (size_t leaf = 0; leaf < phytomer_ptr->leaf_bases[p].size(); leaf++) {
                                    phytomer_ptr->leaf_bases[p][leaf] += tip_correction;
                                }
                            }

                            // Translate all attached floral buds by tip correction
                            for (auto &fbud: phytomer_ptr->floral_buds[p]) {
                                fbud.base_position += tip_correction;
                                // Translate inflorescence geometry if it exists
                                for (uint inflorescence_objID: fbud.inflorescence_objIDs) {
                                    context_ptr->translateObject(inflorescence_objID, tip_correction);
                                }
                                // Translate peduncle geometry if it exists
                                for (uint peduncle_objID: fbud.peduncle_objIDs) {
                                    context_ptr->translateObject(peduncle_objID, tip_correction);
                                }
                            }
                        } else {
                            // Fallback: use old correction method for backward compatibility
                            // Scale petiole geometry to match the actual saved dimensions
                            phytomer_ptr->scalePetioleGeometry(p, petiole_lengths[p], petiole_radii_values[p]);

                            // Apply saved base position if available (to correct cumulative positioning errors)
                            if (p < petiole_base_positions.size() && petiole_base_positions[p] != nullorigin) {
                                vec3 current_base = phytomer_ptr->petiole_vertices[p][0];
                                vec3 position_correction = petiole_base_positions[p] - current_base;

                                // Translate all petiole vertices
                                for (auto &vertex: phytomer_ptr->petiole_vertices[p]) {
                                    vertex += position_correction;
                                }

                                // Rebuild the petiole Context geometry with corrected positions
                                context_ptr->deleteObject(phytomer_ptr->petiole_objIDs[p]);
                                std::vector<RGBcolor> petiole_colors(phytomer_ptr->petiole_radii[p].size(), phytomer_ptr->phytomer_parameters.petiole.color);
                                uint Ndiv_petiole_radius = std::max(uint(3), phytomer_ptr->phytomer_parameters.petiole.radial_subdivisions);
                                phytomer_ptr->petiole_objIDs[p] = makeTubeFromCones(Ndiv_petiole_radius, phytomer_ptr->petiole_vertices[p], phytomer_ptr->petiole_radii[p], petiole_colors, context_ptr);

                                // Restore primitive data labels
                                if (!phytomer_ptr->petiole_objIDs[p].empty()) {
                                    context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(phytomer_ptr->petiole_objIDs[p]), "object_label", "petiole");
                                }

                                // Restore saved leaf base positions in data structure (leaves will be deleted and recreated with these positions)
                                if (p < saved_leaf_bases_all_petioles.size() && !saved_leaf_bases_all_petioles[p].empty()) {
                                    // Use saved leaf bases directly
                                    for (size_t leaf = 0; leaf < phytomer_ptr->leaf_bases[p].size(); leaf++) {
                                        if (leaf < saved_leaf_bases_all_petioles[p].size()) {
                                            phytomer_ptr->leaf_bases[p][leaf] = saved_leaf_bases_all_petioles[p][leaf];
                                        }
                                    }
                                } else {
                                    // Fallback: update leaf bases by position correction (for backward compatibility)
                                    for (size_t leaf = 0; leaf < phytomer_ptr->leaf_bases[p].size(); leaf++) {
                                        phytomer_ptr->leaf_bases[p][leaf] += position_correction;
                                    }
                                }

                                // Translate all attached floral buds
                                for (auto &fbud: phytomer_ptr->floral_buds[p]) {
                                    fbud.base_position += position_correction;
                                    // Translate inflorescence geometry if it exists
                                    for (uint inflorescence_objID: fbud.inflorescence_objIDs) {
                                        context_ptr->translateObject(inflorescence_objID, position_correction);
                                    }
                                    // Translate peduncle geometry if it exists
                                    for (uint peduncle_objID: fbud.peduncle_objIDs) {
                                        context_ptr->translateObject(peduncle_objID, position_correction);
                                    }
                                }
                            }
                        }

                        // Restore other petiole properties
                        phytomer_ptr->petiole_pitch[p] = deg2rad(petiole_pitches[p]);
                        phytomer_ptr->petiole_curvature[p] = petiole_curvatures[p];
                        phytomer_ptr->current_leaf_scale_factor[p] = current_leaf_scale_factors[p];
                    }
                }

                // Delete and recreate leaves with correct petiole/internode axes
                // This is necessary because leaves were created with wrong axes before petiole geometry was restored

                // Step 1: Save object data from existing leaves
                std::vector<std::vector<std::map<std::string, int>>> saved_leaf_data_int;
                std::vector<std::vector<std::map<std::string, uint>>> saved_leaf_data_uint;
                std::vector<std::vector<std::map<std::string, float>>> saved_leaf_data_float;
                std::vector<std::vector<std::map<std::string, double>>> saved_leaf_data_double;
                std::vector<std::vector<std::map<std::string, vec2>>> saved_leaf_data_vec2;
                std::vector<std::vector<std::map<std::string, vec3>>> saved_leaf_data_vec3;
                std::vector<std::vector<std::map<std::string, vec4>>> saved_leaf_data_vec4;
                std::vector<std::vector<std::map<std::string, int2>>> saved_leaf_data_int2;
                std::vector<std::vector<std::map<std::string, int3>>> saved_leaf_data_int3;
                std::vector<std::vector<std::map<std::string, int4>>> saved_leaf_data_int4;
                std::vector<std::vector<std::map<std::string, std::string>>> saved_leaf_data_string;

                saved_leaf_data_int.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_uint.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_float.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_double.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_vec2.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_vec3.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_vec4.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_int2.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_int3.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_int4.resize(phytomer_ptr->leaf_objIDs.size());
                saved_leaf_data_string.resize(phytomer_ptr->leaf_objIDs.size());

                for (int petiole = 0; petiole < phytomer_ptr->leaf_objIDs.size(); petiole++) {
                    saved_leaf_data_int[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_uint[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_float[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_double[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_vec2[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_vec3[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_vec4[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_int2[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_int3[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_int4[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());
                    saved_leaf_data_string[petiole].resize(phytomer_ptr->leaf_objIDs[petiole].size());

                    for (int leaf = 0; leaf < phytomer_ptr->leaf_objIDs[petiole].size(); leaf++) {
                        uint objID = phytomer_ptr->leaf_objIDs[petiole][leaf];
                        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);
                        if (!UUIDs.empty()) {
                            // Get all primitive data labels
                            std::vector<std::string> data_int = context_ptr->listPrimitiveData(UUIDs.front());
                            for (const auto &label: data_int) {
                                HeliosDataType type = context_ptr->getPrimitiveDataType(label.c_str());
                                if (type == HELIOS_TYPE_INT) {
                                    int value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_int[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_UINT) {
                                    uint value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_uint[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_FLOAT) {
                                    float value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_float[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_DOUBLE) {
                                    double value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_double[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_VEC2) {
                                    vec2 value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_vec2[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_VEC3) {
                                    vec3 value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_vec3[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_VEC4) {
                                    vec4 value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_vec4[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_INT2) {
                                    int2 value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_int2[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_INT3) {
                                    int3 value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_int3[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_INT4) {
                                    int4 value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_int4[petiole][leaf][label] = value;
                                } else if (type == HELIOS_TYPE_STRING) {
                                    std::string value;
                                    context_ptr->getPrimitiveData(UUIDs.front(), label.c_str(), value);
                                    saved_leaf_data_string[petiole][leaf][label] = value;
                                }
                            }
                        }
                    }
                }

                // Step 2: Delete existing leaves and clear data structures
                for (int petiole = 0; petiole < phytomer_ptr->leaf_objIDs.size(); petiole++) {
                    for (uint objID: phytomer_ptr->leaf_objIDs[petiole]) {
                        context_ptr->deleteObject(objID);
                    }
                    phytomer_ptr->leaf_objIDs[petiole].clear();
                    phytomer_ptr->leaf_bases[petiole].clear();
                    phytomer_ptr->leaf_rotation[petiole].clear();
                }

                // Step 3: Recreate leaves using Phytomer constructor logic with corrected axes
                assert(leaf_scale.size() == leaf_pitch.size());
                float leaflet_offset_val = 0.f; // Will be set from saved data if available

                // Create unique leaf prototypes if they don't exist (matching Phytomer constructor)
                if (leaf_scale.size() > 0) {
                    // Find maximum leaves per petiole across all petioles
                    int max_leaves_per_petiole = 0;
                    for (size_t i = 0; i < leaf_scale.size(); i++) {
                        max_leaves_per_petiole = std::max(max_leaves_per_petiole, (int) leaf_scale[i].size());
                    }
                    int leaves_per_petiole = max_leaves_per_petiole;
                    assert(phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototype_identifier != 0);

                    if (phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototypes > 0 &&
                        this->unique_leaf_prototype_objIDs.find(phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototype_identifier) == this->unique_leaf_prototype_objIDs.end()) {
                        this->unique_leaf_prototype_objIDs[phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototype_identifier].resize(phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototypes);
                        for (int prototype = 0; prototype < phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototypes; prototype++) {
                            for (int leaf = 0; leaf < leaves_per_petiole; leaf++) {
                                float ind_from_tip = float(leaf) - float(leaves_per_petiole - 1) / 2.f;
                                uint objID_leaf = phytomer_ptr->phytomer_parameters.leaf.prototype.prototype_function(context_ptr, &phytomer_ptr->phytomer_parameters.leaf.prototype, ind_from_tip);
                                if (phytomer_ptr->phytomer_parameters.leaf.prototype.prototype_function == GenericLeafPrototype) {
                                    context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(objID_leaf), "object_label", "leaf");
                                }
                                this->unique_leaf_prototype_objIDs.at(phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).push_back(objID_leaf);
                                std::vector<uint> petiolule_UUIDs = context_ptr->filterPrimitivesByData(context_ptr->getObjectPrimitiveUUIDs(objID_leaf), "object_label", "petiolule");
                                context_ptr->setPrimitiveColor(petiolule_UUIDs, phytomer_ptr->phytomer_parameters.petiole.color);
                                context_ptr->hideObject(objID_leaf);
                            }
                        }
                    }
                }

                for (int petiole = 0; petiole < leaf_scale.size(); petiole++) {
                    int leaves_per_petiole = leaf_scale[petiole].size();

                    // Get CORRECT petiole axis (after geometry restoration)
                    vec3 petiole_tip_axis = phytomer_ptr->getPetioleAxisVector(1.f, petiole);
                    vec3 internode_axis = phytomer_ptr->getInternodeAxisVector(1.f);

                    // Resize leaf data structures
                    phytomer_ptr->leaf_objIDs[petiole].resize(leaves_per_petiole);
                    phytomer_ptr->leaf_bases[petiole].resize(leaves_per_petiole);
                    phytomer_ptr->leaf_rotation[petiole].resize(leaves_per_petiole);

                    for (int leaf = 0; leaf < leaves_per_petiole; leaf++) {
                        float ind_from_tip = float(leaf) - float(leaves_per_petiole - 1) / 2.f;

                        // Copy leaf from prototype (matching constructor logic)
                        uint objID_leaf;
                        if (phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototypes > 0) {
                            int prototype = context_ptr->randu(0, phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototypes - 1);
                            uint uid = phytomer_ptr->phytomer_parameters.leaf.prototype.unique_prototype_identifier;
                            assert(this->unique_leaf_prototype_objIDs.find(uid) != this->unique_leaf_prototype_objIDs.end());
                            assert(this->unique_leaf_prototype_objIDs.at(uid).size() > prototype);
                            assert(this->unique_leaf_prototype_objIDs.at(uid).at(prototype).size() > leaf);
                            objID_leaf = context_ptr->copyObject(this->unique_leaf_prototype_objIDs.at(uid).at(prototype).at(leaf));
                        } else {
                            objID_leaf = phytomer_ptr->phytomer_parameters.leaf.prototype.prototype_function(context_ptr, &phytomer_ptr->phytomer_parameters.leaf.prototype, ind_from_tip);
                        }

                        // Scale leaf
                        vec3 leaf_scale_vec = leaf_scale[petiole][leaf] * make_vec3(1, 1, 1);
                        context_ptr->scaleObject(objID_leaf, leaf_scale_vec);

                        // Calculate compound rotation (matching constructor logic)
                        float compound_rotation = 0;
                        if (leaves_per_petiole > 1) {
                            if (leaf == float(leaves_per_petiole - 1) / 2.f) {
                                compound_rotation = 0; // tip leaf
                            } else if (leaf < float(leaves_per_petiole - 1) / 2.f) {
                                compound_rotation = -0.5 * M_PI; // left lateral
                            } else {
                                compound_rotation = 0.5 * M_PI; // right lateral
                            }
                        }

                        // Apply saved rotations (matching constructor logic)
                        float saved_pitch = deg2rad(leaf_pitch[petiole][leaf]);
                        float saved_yaw = deg2rad(leaf_yaw[petiole][leaf]);
                        float saved_roll = deg2rad(leaf_roll[petiole][leaf]);

                        // Roll rotation
                        float roll_rot = 0;
                        if (leaves_per_petiole == 1) {
                            int sign = 1; // Simplified, constructor uses shoot_index
                            roll_rot = (acos_safe(internode_axis.z) - saved_roll) * sign;
                        } else if (ind_from_tip != 0) {
                            roll_rot = (asin_safe(petiole_tip_axis.z) + saved_roll) * compound_rotation / std::fabs(compound_rotation);
                        }
                        phytomer_ptr->leaf_rotation[petiole][leaf].roll = saved_roll;
                        context_ptr->rotateObject(objID_leaf, roll_rot, "x");

                        // Pitch rotation
                        phytomer_ptr->leaf_rotation[petiole][leaf].pitch = saved_pitch;
                        float pitch_rot = saved_pitch;
                        if (ind_from_tip == 0) {
                            pitch_rot += asin_safe(petiole_tip_axis.z);
                        }
                        context_ptr->rotateObject(objID_leaf, -pitch_rot, "y");

                        // Yaw rotation
                        if (ind_from_tip != 0) {
                            phytomer_ptr->leaf_rotation[petiole][leaf].yaw = saved_yaw;
                            context_ptr->rotateObject(objID_leaf, saved_yaw, "z");
                        } else {
                            phytomer_ptr->leaf_rotation[petiole][leaf].yaw = 0;
                        }

                        // Rotate to petiole azimuth
                        context_ptr->rotateObject(objID_leaf, -std::atan2(petiole_tip_axis.y, petiole_tip_axis.x) + compound_rotation, "z");

                        // Use saved leaf base position
                        vec3 leaf_base;
                        if (petiole < saved_leaf_bases_all_petioles.size() && leaf < saved_leaf_bases_all_petioles[petiole].size()) {
                            leaf_base = saved_leaf_bases_all_petioles[petiole][leaf];
                        } else {
                            // Fallback: calculate from petiole (shouldn't happen with saved data)
                            leaf_base = phytomer_ptr->petiole_vertices[petiole].back();
                        }
                        context_ptr->translateObject(objID_leaf, leaf_base);

                        phytomer_ptr->leaf_objIDs[petiole][leaf] = objID_leaf;
                        phytomer_ptr->leaf_bases[petiole][leaf] = leaf_base;
                    }
                }

                // Step 4: Restore saved object data
                for (int petiole = 0; petiole < phytomer_ptr->leaf_objIDs.size(); petiole++) {
                    for (int leaf = 0; leaf < phytomer_ptr->leaf_objIDs[petiole].size(); leaf++) {
                        uint objID = phytomer_ptr->leaf_objIDs[petiole][leaf];
                        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);

                        for (const auto &pair: saved_leaf_data_int[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_uint[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_float[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_double[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_vec2[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_vec3[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_vec4[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_int2[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_int3[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_int4[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                        for (const auto &pair: saved_leaf_data_string[petiole][leaf]) {
                            context_ptr->setPrimitiveData(UUIDs, pair.first.c_str(), pair.second);
                        }
                    }
                }

                // Apply floral bud data
                if (!floral_bud_data.empty() && floral_bud_data.size() <= phytomer_ptr->petiole_length.size()) {
                    // Ensure the floral_buds vector is properly sized
                    phytomer_ptr->floral_buds.resize(floral_bud_data.size());

                    for (size_t petiole = 0; petiole < floral_bud_data.size(); petiole++) {
                        phytomer_ptr->floral_buds.at(petiole).resize(floral_bud_data.at(petiole).size());

                        for (size_t bud = 0; bud < floral_bud_data.at(petiole).size(); bud++) {
                            const FloralBudData &fbud_data = floral_bud_data.at(petiole).at(bud);

                            // Create and populate FloralBud struct
                            FloralBud &fbud = phytomer_ptr->floral_buds.at(petiole).at(bud);
                            // Note: Do NOT set fbud.state here - setFloralBudState() will set it and create geometry
                            fbud.parent_index = fbud_data.parent_index;
                            fbud.bud_index = fbud_data.bud_index;
                            fbud.isterminal = fbud_data.is_terminal;
                            fbud.base_position = fbud_data.base_position;
                            fbud.base_rotation = fbud_data.base_rotation;
                            fbud.current_fruit_scale_factor = fbud_data.current_fruit_scale_factor;

                            // Set the floral bud state to trigger geometry creation
                            // This will call updateInflorescence() which uses prototype functions from shoot parameters
                            // Only rebuild geometry if the shoot has prototype functions defined
                            BudState desired_state = static_cast<BudState>(fbud_data.bud_state);

                            if (desired_state != BUD_DORMANT && desired_state != BUD_DEAD && desired_state != BUD_ACTIVE) {
                                // Apply saved peduncle parameters before creating geometry
                                if (fbud_data.peduncle_length >= 0) {
                                    phytomer_ptr->phytomer_parameters.peduncle.length = fbud_data.peduncle_length;
                                }
                                if (fbud_data.peduncle_radius >= 0) {
                                    phytomer_ptr->phytomer_parameters.peduncle.radius = fbud_data.peduncle_radius;
                                }
                                if (fbud_data.peduncle_pitch != 0) {
                                    phytomer_ptr->phytomer_parameters.peduncle.pitch = fbud_data.peduncle_pitch;
                                }
                                if (fbud_data.peduncle_roll != 0) {
                                    phytomer_ptr->phytomer_parameters.peduncle.roll = fbud_data.peduncle_roll;
                                }
                                if (fbud_data.peduncle_curvature != 0) {
                                    phytomer_ptr->phytomer_parameters.peduncle.curvature = fbud_data.peduncle_curvature;
                                }

                                // Apply saved inflorescence parameters before creating geometry
                                if (fbud_data.flower_offset >= 0) {
                                    phytomer_ptr->phytomer_parameters.inflorescence.flower_offset = fbud_data.flower_offset;
                                }

                                if ((desired_state == BUD_FLOWER_CLOSED || desired_state == BUD_FLOWER_OPEN) && phytomer_ptr->phytomer_parameters.inflorescence.flower_prototype_function != nullptr) {
                                    FloralBud &fbud = phytomer_ptr->floral_buds.at(petiole).at(bud);

                                    // Manually set up floral bud state without calling updateInflorescence
                                    context_ptr->deleteObject(fbud.inflorescence_objIDs);
                                    fbud.inflorescence_objIDs.clear();
                                    fbud.inflorescence_bases.clear();
                                    fbud.inflorescence_rotation.clear();

                                    if (phytomer_ptr->build_context_geometry_peduncle) {
                                        context_ptr->deleteObject(fbud.peduncle_objIDs);
                                        fbud.peduncle_objIDs.clear();
                                    }

                                    // Set the state without calling updateInflorescence
                                    fbud.state = desired_state;
                                    fbud.time_counter = 0;

                                    // Restore peduncle geometry from saved data
                                    if (!fbud_data.peduncle_vertices_saved.empty() && !fbud_data.peduncle_radii_saved.empty()) {
                                        if (petiole < phytomer_ptr->peduncle_vertices.size()) {
                                            if (phytomer_ptr->peduncle_vertices.at(petiole).size() <= bud) {
                                                phytomer_ptr->peduncle_vertices.at(petiole).resize(bud + 1);
                                                phytomer_ptr->peduncle_radii.at(petiole).resize(bud + 1);
                                            }
                                            phytomer_ptr->peduncle_vertices.at(petiole).at(bud) = fbud_data.peduncle_vertices_saved;
                                            phytomer_ptr->peduncle_radii.at(petiole).at(bud) = fbud_data.peduncle_radii_saved;
                                        }

                                        // Rebuild peduncle Context geometry with saved vertices/radii
                                        if (phytomer_ptr->build_context_geometry_peduncle) {
                                            std::vector<RGBcolor> colors(fbud_data.peduncle_vertices_saved.size(), phytomer_ptr->phytomer_parameters.peduncle.color);
                                            uint Ndiv = std::max(uint(3), phytomer_ptr->phytomer_parameters.peduncle.radial_subdivisions);

                                            fbud.peduncle_objIDs.push_back(context_ptr->addTubeObject(Ndiv, fbud_data.peduncle_vertices_saved, fbud_data.peduncle_radii_saved, colors));
                                            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(fbud.peduncle_objIDs.back()), "object_label", "peduncle");
                                        }
                                    }

                                    // Restore flower geometry using saved positions and rotations
                                    if (!fbud_data.inflorescence_bases_saved.empty() && !fbud_data.flower_rotations.empty() && fbud_data.inflorescence_bases_saved.size() == fbud_data.flower_rotations.size()) {

                                        for (size_t i = 0; i < fbud_data.inflorescence_bases_saved.size(); i++) {
                                            vec3 flower_base_saved = fbud_data.inflorescence_bases_saved.at(i);
                                            float saved_pitch = deg2rad(fbud_data.flower_rotations.at(i).pitch);
                                            float saved_yaw = deg2rad(fbud_data.flower_rotations.at(i).yaw);
                                            float saved_roll = deg2rad(fbud_data.flower_rotations.at(i).roll);
                                            float saved_azimuth = deg2rad(fbud_data.flower_rotations.at(i).azimuth);

                                            // Recalculate peduncle_axis from saved peduncle geometry and flower position
                                            int flowers_per_peduncle = fbud_data.inflorescence_bases_saved.size();
                                            int petioles_per_internode = phytomer_ptr->phytomer_parameters.petiole.petioles_per_internode;
                                            float flower_offset_val = fbud_data.flower_offset;
                                            if (flowers_per_peduncle > 2) {
                                                float denom = 0.5f * float(flowers_per_peduncle) - 1.f;
                                                if (flower_offset_val * denom > 1.f) {
                                                    flower_offset_val = 1.f / denom;
                                                }
                                            }

                                            float ind_from_tip = fabs(float(i) - float(flowers_per_peduncle - 1) / float(petioles_per_internode));
                                            float frac = 1.0f;
                                            if (flowers_per_peduncle > 1 && flower_offset_val > 0 && ind_from_tip != 0) {
                                                float offset = (ind_from_tip - 0.5f) * flower_offset_val * fbud_data.peduncle_length;
                                                if (fbud_data.peduncle_length > 0) {
                                                    frac = 1.f - offset / fbud_data.peduncle_length;
                                                }
                                            }
                                            vec3 recalculated_peduncle_axis = phytomer_ptr->getAxisVector(frac, fbud_data.peduncle_vertices_saved);

                                            // Use individual base scale if available, otherwise use default parameter
                                            float scale_factor;
                                            if (i < fbud_data.flower_base_scales.size() && fbud_data.flower_base_scales.at(i) >= 0) {
                                                scale_factor = fbud_data.flower_base_scales.at(i);
                                            } else {
                                                scale_factor = phytomer_ptr->phytomer_parameters.inflorescence.flower_prototype_scale.val();
                                            }

                                            // Determine if flower is open
                                            bool is_open_flower = (desired_state == BUD_FLOWER_OPEN);

                                            // Create flower geometry with saved rotations and recalculated peduncle axis
                                            phytomer_ptr->createInflorescenceGeometry(fbud, flower_base_saved, recalculated_peduncle_axis, saved_pitch, saved_roll, saved_azimuth, saved_yaw, scale_factor, is_open_flower);
                                        }
                                    }

                                } else if (desired_state == BUD_FRUITING && phytomer_ptr->phytomer_parameters.inflorescence.fruit_prototype_function != nullptr) {
                                    FloralBud &fbud = phytomer_ptr->floral_buds.at(petiole).at(bud);

                                    // Manually set up floral bud state without calling updateInflorescence
                                    context_ptr->deleteObject(fbud.inflorescence_objIDs);
                                    fbud.inflorescence_objIDs.clear();
                                    fbud.inflorescence_bases.clear();
                                    fbud.inflorescence_rotation.clear();

                                    if (phytomer_ptr->build_context_geometry_peduncle) {
                                        context_ptr->deleteObject(fbud.peduncle_objIDs);
                                        fbud.peduncle_objIDs.clear();
                                    }

                                    fbud.state = desired_state;
                                    fbud.time_counter = 0;

                                    // Restore peduncle geometry from saved data
                                    if (!fbud_data.peduncle_vertices_saved.empty() && !fbud_data.peduncle_radii_saved.empty()) {
                                        if (petiole < phytomer_ptr->peduncle_vertices.size()) {
                                            if (phytomer_ptr->peduncle_vertices.at(petiole).size() <= bud) {
                                                phytomer_ptr->peduncle_vertices.at(petiole).resize(bud + 1);
                                                phytomer_ptr->peduncle_radii.at(petiole).resize(bud + 1);
                                            }
                                            phytomer_ptr->peduncle_vertices.at(petiole).at(bud) = fbud_data.peduncle_vertices_saved;
                                            phytomer_ptr->peduncle_radii.at(petiole).at(bud) = fbud_data.peduncle_radii_saved;
                                        }

                                        // Rebuild peduncle Context geometry with saved vertices/radii
                                        if (phytomer_ptr->build_context_geometry_peduncle) {
                                            std::vector<RGBcolor> colors(fbud_data.peduncle_vertices_saved.size(), phytomer_ptr->phytomer_parameters.peduncle.color);
                                            uint Ndiv = std::max(uint(3), phytomer_ptr->phytomer_parameters.peduncle.radial_subdivisions);

                                            fbud.peduncle_objIDs.push_back(context_ptr->addTubeObject(Ndiv, fbud_data.peduncle_vertices_saved, fbud_data.peduncle_radii_saved, colors));
                                            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(fbud.peduncle_objIDs.back()), "object_label", "peduncle");
                                        }
                                    }

                                    // Restore fruit geometry using saved positions and rotations
                                    if (!fbud_data.inflorescence_bases_saved.empty() && !fbud_data.flower_rotations.empty() && fbud_data.inflorescence_bases_saved.size() == fbud_data.flower_rotations.size()) {

                                        for (size_t i = 0; i < fbud_data.inflorescence_bases_saved.size(); i++) {
                                            // Get saved values
                                            vec3 fruit_base_saved = fbud_data.inflorescence_bases_saved.at(i);
                                            float saved_pitch = deg2rad(fbud_data.flower_rotations.at(i).pitch);
                                            float saved_yaw = deg2rad(fbud_data.flower_rotations.at(i).yaw);
                                            float saved_roll = deg2rad(fbud_data.flower_rotations.at(i).roll);
                                            float saved_azimuth = deg2rad(fbud_data.flower_rotations.at(i).azimuth);

                                            // Recalculate peduncle_axis from saved peduncle geometry and fruit position
                                            int flowers_per_peduncle = fbud_data.inflorescence_bases_saved.size();
                                            int petioles_per_internode = phytomer_ptr->phytomer_parameters.petiole.petioles_per_internode;
                                            float flower_offset_val = fbud_data.flower_offset;
                                            if (flowers_per_peduncle > 2) {
                                                float denom = 0.5f * float(flowers_per_peduncle) - 1.f;
                                                if (flower_offset_val * denom > 1.f) {
                                                    flower_offset_val = 1.f / denom;
                                                }
                                            }

                                            float ind_from_tip = fabs(float(i) - float(flowers_per_peduncle - 1) / float(petioles_per_internode));
                                            float frac = 1.0f;
                                            if (flowers_per_peduncle > 1 && flower_offset_val > 0 && ind_from_tip != 0) {
                                                float offset = (ind_from_tip - 0.5f) * flower_offset_val * fbud_data.peduncle_length;
                                                if (fbud_data.peduncle_length > 0) {
                                                    frac = 1.f - offset / fbud_data.peduncle_length;
                                                }
                                            }
                                            vec3 recalculated_peduncle_axis = phytomer_ptr->getAxisVector(frac, fbud_data.peduncle_vertices_saved);

                                            // Use individual base scale if available, then apply growth scaling
                                            float base_fruit_scale;
                                            if (i < fbud_data.flower_base_scales.size() && fbud_data.flower_base_scales.at(i) >= 0) {
                                                base_fruit_scale = fbud_data.flower_base_scales.at(i);
                                            } else {
                                                base_fruit_scale = phytomer_ptr->phytomer_parameters.inflorescence.fruit_prototype_scale.val();
                                            }
                                            float scale_factor = base_fruit_scale * fbud_data.current_fruit_scale_factor;

                                            // Create fruit geometry with saved rotations and recalculated peduncle axis
                                            phytomer_ptr->createInflorescenceGeometry(fbud, fruit_base_saved, recalculated_peduncle_axis, saved_pitch, saved_roll, saved_azimuth, saved_yaw, scale_factor, false);
                                        }
                                    }
                                }
                            } else {
                                // For states that don't create geometry (DORMANT, DEAD, ACTIVE), just set the state
                                fbud.state = desired_state;
                            }
                        }
                    }
                }

                phytomer_count++;
            } // phytomers

        } // shoots

    } // plant instances

    if (!quiet) {
        std::cout << "done." << std::endl;
    }
    return plantIDs;
}
