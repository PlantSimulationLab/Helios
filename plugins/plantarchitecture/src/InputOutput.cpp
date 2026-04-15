/** \file "InputOutput.cpp" Routines for reading and writing plant geometry in the plant architecture plug-in.

Copyright (C) 2016-2026 Brian Bailey

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

static void renameAutoMaterial(helios::Context *context_ptr, uint objID, const std::string &desired_base_name) {
    std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);
    if (UUIDs.empty()) return;

    std::string current_label = context_ptr->getPrimitiveMaterialLabel(UUIDs.front());
    if (current_label.substr(0, 7) != "__auto_") return;

    if (!context_ptr->doesMaterialExist(desired_base_name)) {
        context_ptr->renameMaterial(current_label, desired_base_name);
    } else {
        uint existing_id = context_ptr->getMaterialIDFromLabel(desired_base_name);
        uint current_id = context_ptr->getMaterialIDFromLabel(current_label);
        if (existing_id == current_id) return;

        int suffix = 1;
        std::string candidate;
        do {
            candidate = desired_base_name + "_" + std::to_string(suffix++);
        } while (context_ptr->doesMaterialExist(candidate));
        context_ptr->renameMaterial(current_label, candidate);
    }
}

static void renameAutoMaterial(helios::Context *context_ptr, const std::vector<uint> &objIDs, const std::string &desired_base_name) {
    for (uint objID : objIDs) {
        renameAutoMaterial(context_ptr, objID, desired_base_name);
    }
}

// Helper function for calculating leaf base positions in compound leaves
static float clampOffset(int count_per_axis, float offset) {
    if (count_per_axis > 2) {
        float denom = 0.5f * float(count_per_axis) - 1.f;
        if (offset * denom > 1.f) {
            offset = 1.f / denom;
        }
    }
    return offset;
}

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

            // Additional parameters for reconstruction from parameters
            output_xml << "\t\t\t\t\t<internode_length_max>" << phytomer->internode_length_max << "</internode_length_max>" << std::endl;

            // Length segments should match perturbation count
            uint length_segments = phytomer->internode_curvature_perturbations.size();
            if (length_segments > 0) {
                output_xml << "\t\t\t\t\t<internode_length_segments>" << length_segments << "</internode_length_segments>" << std::endl;
            } else if (phytomer_index < shoot->shoot_internode_vertices.size()) {
                // Fallback: infer from vertex count (for phytomers with no perturbations)
                if (phytomer_index == 0) {
                    length_segments = shoot->shoot_internode_vertices[phytomer_index].size() - 1;
                } else {
                    length_segments = shoot->shoot_internode_vertices[phytomer_index].size();
                }
                output_xml << "\t\t\t\t\t<internode_length_segments>" << length_segments << "</internode_length_segments>" << std::endl;
            }

            // Radial subdivisions - use default from Context geometry (stored during tube creation)
            // For now, we'll read this from the tube object during reconstruction

            // Save curvature perturbations (semicolon-delimited)
            if (!phytomer->internode_curvature_perturbations.empty()) {
                output_xml << "\t\t\t\t\t<curvature_perturbations>";
                for (size_t i = 0; i < phytomer->internode_curvature_perturbations.size(); i++) {
                    output_xml << phytomer->internode_curvature_perturbations[i];
                    if (i < phytomer->internode_curvature_perturbations.size() - 1)
                        output_xml << ";";
                }
                output_xml << "</curvature_perturbations>" << std::endl;
            }

            // Save yaw perturbations (semicolon-delimited)
            if (!phytomer->internode_yaw_perturbations.empty()) {
                output_xml << "\t\t\t\t\t<yaw_perturbations>";
                for (size_t i = 0; i < phytomer->internode_yaw_perturbations.size(); i++) {
                    output_xml << phytomer->internode_yaw_perturbations[i];
                    if (i < phytomer->internode_yaw_perturbations.size() - 1)
                        output_xml << ";";
                }
                output_xml << "</yaw_perturbations>" << std::endl;
            }

            // Note: internode_vertices and internode_radii are no longer written to XML
            // Internodes are now reconstructed from parameters (length, pitch, phyllotactic_angle, length_max, segments)
            // and stochastic state (curvature_perturbations, yaw_perturbations) during XML reading

            // Removed in Phase 3 - internodes now reconstructed from parameters:
            // if (phytomer_index < shoot->shoot_internode_vertices.size()) {
            //     const auto &vertices = shoot->shoot_internode_vertices[phytomer_index];
            //     output_xml << "\t\t\t\t\t<internode_vertices>";
            //     for (size_t v = 0; v < vertices.size(); v++) {
            //         output_xml << vertices[v].x << " " << vertices[v].y << " " << vertices[v].z;
            //         if (v < vertices.size() - 1) output_xml << ";";
            //     }
            //     output_xml << "</internode_vertices>" << std::endl;
            // }
            // if (phytomer_index < shoot->shoot_internode_radii.size()) {
            //     const auto &radii = shoot->shoot_internode_radii[phytomer_index];
            //     output_xml << "\t\t\t\t\t<internode_radii>";
            //     for (size_t r = 0; r < radii.size(); r++) {
            //         output_xml << radii[r];
            //         if (r < radii.size() - 1) output_xml << ";";
            //     }
            //     output_xml << "</internode_radii>" << std::endl;
            // }

            for (uint petiole = 0; petiole < phytomer->petiole_length.size(); petiole++) {

                output_xml << "\t\t\t\t\t<petiole>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_length>" << phytomer->petiole_length.at(petiole) << "</petiole_length>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_radius>" << phytomer->petiole_radii.at(petiole).front() << "</petiole_radius>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_pitch>" << rad2deg(phytomer->petiole_pitch.at(petiole)) << "</petiole_pitch>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_curvature>" << phytomer->petiole_curvature.at(petiole) << "</petiole_curvature>" << std::endl;
                // Note: petiole_base_position is no longer written to XML - it is auto-calculated from the parent internode tip during XML reading
                output_xml << "\t\t\t\t\t\t<current_leaf_scale_factor>" << phytomer->current_leaf_scale_factor.at(petiole) << "</current_leaf_scale_factor>" << std::endl;

                // Bulk parameters for exact petiole reconstruction
                output_xml << "\t\t\t\t\t\t<petiole_taper>" << phytomer->petiole_taper.at(petiole) << "</petiole_taper>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_length_segments>" << phytomer->phytomer_parameters.petiole.length_segments << "</petiole_length_segments>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_radial_subdivisions>" << phytomer->phytomer_parameters.petiole.radial_subdivisions << "</petiole_radial_subdivisions>" << std::endl;

                if (phytomer->leaf_rotation.at(petiole).size() == 1) { // not compound leaf
                    output_xml << "\t\t\t\t\t\t<leaflet_scale>" << 1.0 << "</leaflet_scale>" << std::endl;
                } else {
                    float tip_ind = floor(float(phytomer->leaf_rotation.at(petiole).size() - 1) / 2.f);
                    output_xml << "\t\t\t\t\t\t<leaflet_scale>" << phytomer->leaf_size_max.at(petiole).at(int(tip_ind - 1)) / max(phytomer->leaf_size_max.at(petiole)) << "</leaflet_scale>" << std::endl;
                }
                output_xml << "\t\t\t\t\t\t<leaflet_offset>" << phytomer->phytomer_parameters.leaf.leaflet_offset.val() << "</leaflet_offset>" << std::endl;

                for (uint leaf = 0; leaf < phytomer->leaf_rotation.at(petiole).size(); leaf++) {
                    output_xml << "\t\t\t\t\t\t<leaf>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_scale>" << phytomer->leaf_size_max.at(petiole).at(leaf) * phytomer->current_leaf_scale_factor.at(petiole) << "</leaf_scale>" << std::endl;
                    // Note: leaf_base is no longer written to XML - it is auto-calculated from the petiole geometry and leaflet_offset during XML reading
                    // output_xml << "\t\t\t\t\t\t\t<leaf_base>" << phytomer->leaf_bases.at(petiole).at(leaf).x << " " << phytomer->leaf_bases.at(petiole).at(leaf).y << " " << phytomer->leaf_bases.at(petiole).at(leaf).z << "</leaf_base>" <<
                    // std::endl;
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

                        // Write fruit scale factor
                        output_xml << "\t\t\t\t\t\t\t<current_fruit_scale_factor>" << fbud.current_fruit_scale_factor << "</current_fruit_scale_factor>" << std::endl;

                        // Write peduncle parameters
                        // Note: Write the STORED VALUES that were actually used for this peduncle geometry
                        output_xml << "\t\t\t\t\t\t\t<peduncle>" << std::endl;
                        if (petiole < phytomer->peduncle_length.size() && bud < phytomer->peduncle_length.at(petiole).size()) {
                            output_xml << "\t\t\t\t\t\t\t\t<length>" << phytomer->peduncle_length.at(petiole).at(bud) << "</length>" << std::endl;
                            output_xml << "\t\t\t\t\t\t\t\t<radius>" << phytomer->peduncle_radius.at(petiole).at(bud) << "</radius>" << std::endl;
                            output_xml << "\t\t\t\t\t\t\t\t<pitch>" << phytomer->peduncle_pitch.at(petiole).at(bud) << "</pitch>" << std::endl;
                            output_xml << "\t\t\t\t\t\t\t\t<curvature>" << phytomer->peduncle_curvature.at(petiole).at(bud) << "</curvature>" << std::endl;
                        } else {
                            // Fallback to parameter values if stored values not available
                            output_xml << "\t\t\t\t\t\t\t\t<length>" << phytomer->phytomer_parameters.peduncle.length.val() << "</length>" << std::endl;
                            output_xml << "\t\t\t\t\t\t\t\t<radius>" << phytomer->phytomer_parameters.peduncle.radius.val() << "</radius>" << std::endl;
                            output_xml << "\t\t\t\t\t\t\t\t<pitch>" << phytomer->phytomer_parameters.peduncle.pitch.val() << "</pitch>" << std::endl;
                            output_xml << "\t\t\t\t\t\t\t\t<curvature>" << phytomer->phytomer_parameters.peduncle.curvature.val() << "</curvature>" << std::endl;
                        }
                        output_xml << "\t\t\t\t\t\t\t\t<roll>" << phytomer->phytomer_parameters.peduncle.roll.val() << "</roll>" << std::endl;

                        output_xml << "\t\t\t\t\t\t\t</peduncle>" << std::endl;

                        // Write inflorescence parameters
                        output_xml << "\t\t\t\t\t\t\t<inflorescence>" << std::endl;
                        output_xml << "\t\t\t\t\t\t\t\t<flower_offset>" << phytomer->phytomer_parameters.inflorescence.flower_offset.val() << "</flower_offset>" << std::endl;

                        // Write individual flower/fruit positions and rotations
                        for (uint i = 0; i < fbud.inflorescence_bases.size(); i++) {
                            output_xml << "\t\t\t\t\t\t\t\t<flower>" << std::endl;
                            // inflorescence_base is now auto-computed during XML reading - not saved to XML
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

                // Read new parameters for reconstruction from parameters
                // internode_length_max
                float internode_length_max = internode_length; // default to current length
                node_string = "internode_length_max";
                if (internode.child(node_string.c_str())) {
                    internode_length_max = parse_xml_tag_float(internode.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                }

                // internode_length_segments
                uint internode_length_segments = 1; // default
                node_string = "internode_length_segments";
                if (internode.child(node_string.c_str())) {
                    internode_length_segments = (uint) parse_xml_tag_int(internode.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                }

                // curvature_perturbations (semicolon-delimited)
                std::vector<float> curvature_perturbations;
                node_string = "curvature_perturbations";
                if (internode.child(node_string.c_str())) {
                    std::string perturbs_str = internode.child_value(node_string.c_str());
                    std::istringstream perturbs_stream(perturbs_str);
                    std::string pert_str;
                    while (std::getline(perturbs_stream, pert_str, ';')) {
                        curvature_perturbations.push_back(std::stof(pert_str));
                    }
                }

                // yaw_perturbations (semicolon-delimited)
                std::vector<float> yaw_perturbations;
                node_string = "yaw_perturbations";
                if (internode.child(node_string.c_str())) {
                    std::string yaw_perturbs_str = internode.child_value(node_string.c_str());
                    std::istringstream yaw_stream(yaw_perturbs_str);
                    std::string yaw_str;
                    while (std::getline(yaw_stream, yaw_str, ';')) {
                        yaw_perturbations.push_back(std::stof(yaw_str));
                    }
                }

                // Read internode vertices if available (optional for backward compatibility - values are ignored, geometry reconstructed from parameters)
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

                // Read internode radii if available (optional for backward compatibility - values are ignored, geometry reconstructed from parameters)
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
                float leaflet_offset;
                std::vector<float> petiole_lengths; // actual length of each petiole within internode
                std::vector<float> petiole_radii_values; // actual radius of each petiole within internode
                std::vector<float> petiole_pitches; // pitch of each petiole within internode
                std::vector<float> petiole_curvatures; // curvature of each petiole within internode
                std::vector<vec3> petiole_base_positions; // actual base position of each petiole within internode
                std::vector<float> current_leaf_scale_factors; // scale factor of each petiole within internode
                std::vector<float> petiole_tapers; // taper value for each petiole
                std::vector<uint> petiole_length_segments_all; // number of segments for each petiole
                std::vector<uint> petiole_radial_subdivisions_all; // radial subdivisions for each petiole
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

                    // petiole taper
                    node_string = "petiole_taper";
                    float petiole_taper = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // petiole length segments
                    node_string = "petiole_length_segments";
                    uint length_segments = (uint) parse_xml_tag_int(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // petiole radial subdivisions
                    node_string = "petiole_radial_subdivisions";
                    uint radial_subdivisions = (uint) parse_xml_tag_int(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // leaflet scale factor
                    node_string = "leaflet_scale";
                    leaflet_scale = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    // leaflet offset (optional for backward compatibility with old XML files)
                    node_string = "leaflet_offset";
                    if (petiole.child(node_string.c_str())) {
                        leaflet_offset = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");
                    } else {
                        leaflet_offset = 0.f; // Default for backward compatibility
                    }

                    // Store petiole properties in vectors
                    petiole_lengths.push_back(petiole_length);
                    petiole_radii_values.push_back(petiole_radius);
                    petiole_pitches.push_back(petiole_pitch);
                    petiole_curvatures.push_back(petiole_curvature);
                    current_leaf_scale_factors.push_back(current_leaf_scale_factor_value);
                    petiole_base_positions.push_back(petiole_base_pos);
                    petiole_tapers.push_back(petiole_taper);
                    petiole_length_segments_all.push_back(length_segments);
                    petiole_radial_subdivisions_all.push_back(radial_subdivisions);

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

                        // leaf base position (optional for backward compatibility - value is ignored, auto-calculated below)
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
                                // inflorescence_base (optional for backward compatibility - value is ignored, auto-calculated)
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
                shoot_parameters.phytomer_parameters.leaf.leaflet_offset = leaflet_offset;

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

                // Restore internode properties from saved values
                phytomer_ptr->internode_pitch = deg2rad(internode_pitch);
                phytomer_ptr->internode_phyllotactic_angle = deg2rad(internode_phyllotactic_angle);

                // Get shoot pointer for internode geometry restoration
                auto shoot_ptr = plant_instances.at(plantID).shoot_tree.at(current_shoot_ID);
                uint phytomer_index_in_shoot = shoot_ptr->phytomers.size() - 1;

                // Helper function to recompute internode orientation vectors from parent phytomer context
                auto recomputeInternodeOrientationVectors_local = [this, plantID](std::shared_ptr<Phytomer> phytomer_ptr, uint phytomer_index_in_shoot, float internode_pitch_rad, float internode_phyllotactic_angle_rad,
                                                                                  helios::vec3 &out_internode_axis_initial, helios::vec3 &out_petiole_rotation_axis, helios::vec3 &out_shoot_bending_axis) {
                    using helios::make_vec3;
                    using helios::rotatePointAboutLine;
                    using helios::nullorigin;
                    using helios::cross;

                    // Get parent axes
                    helios::vec3 parent_internode_axis = make_vec3(0, 0, 1);
                    helios::vec3 parent_petiole_axis = make_vec3(0, -1, 0);

                    if (phytomer_index_in_shoot > 0) {
                        auto prev_phytomer = phytomer_ptr->parent_shoot_ptr->phytomers.at(phytomer_index_in_shoot - 1);
                        parent_internode_axis = prev_phytomer->getInternodeAxisVector(1.0f);
                        parent_petiole_axis = prev_phytomer->getPetioleAxisVector(0.f, 0);
                    } else if (phytomer_ptr->parent_shoot_ptr->parent_shoot_ID >= 0) {
                        int parent_shoot_id = phytomer_ptr->parent_shoot_ptr->parent_shoot_ID;
                        uint parent_node_index = phytomer_ptr->parent_shoot_ptr->parent_node_index;
                        uint parent_petiole_index = phytomer_ptr->parent_shoot_ptr->parent_petiole_index;
                        auto &parent_shoot = plant_instances.at(plantID).shoot_tree.at(parent_shoot_id);
                        auto parent_phytomer = parent_shoot->phytomers.at(parent_node_index);
                        parent_internode_axis = parent_phytomer->getInternodeAxisVector(1.0f);
                        parent_petiole_axis = parent_phytomer->getPetioleAxisVector(0.f, parent_petiole_index);
                    }

                    helios::vec3 petiole_rotation_axis = cross(parent_internode_axis, parent_petiole_axis);
                    if (petiole_rotation_axis.magnitude() < 1e-6f) {
                        petiole_rotation_axis = make_vec3(1, 0, 0);
                    } else {
                        petiole_rotation_axis.normalize();
                    }

                    helios::vec3 internode_axis = parent_internode_axis;

                    if (phytomer_index_in_shoot == 0) {
                        AxisRotation shoot_base_rotation = phytomer_ptr->parent_shoot_ptr->base_rotation;
                        if (internode_pitch_rad != 0.f) {
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, 0.5f * internode_pitch_rad);
                        }
                        if (shoot_base_rotation.roll != 0.f) {
                            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll);
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll);
                        }
                        if (shoot_base_rotation.pitch != 0.f) {
                            helios::vec3 base_pitch_axis = -1.0f * cross(parent_internode_axis, parent_petiole_axis);
                            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
                        }
                        if (shoot_base_rotation.yaw != 0.f) {
                            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
                        }
                    } else {
                        if (internode_pitch_rad != 0.f) {
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, -1.25f * internode_pitch_rad);
                        }
                    }

                    helios::vec3 shoot_bending_axis = cross(internode_axis, make_vec3(0, 0, 1));
                    if (internode_axis == make_vec3(0, 0, 1)) {
                        shoot_bending_axis = make_vec3(0, 1, 0);
                    } else {
                        shoot_bending_axis.normalize();
                    }

                    out_internode_axis_initial = internode_axis;
                    out_petiole_rotation_axis = petiole_rotation_axis;
                    out_shoot_bending_axis = shoot_bending_axis;
                };

                // Reconstruct internode geometry from parameters
                if (internode_length_segments > 0) {
                    // Step 1: Compute base position
                    helios::vec3 internode_base;
                    if (phytomer_index_in_shoot == 0) {
                        // First phytomer in this shoot
                        if (shoot_ptr->parent_shoot_ID < 0) {
                            // Base shoot: use plant base position from plant instance
                            internode_base = plant_instances.at(plantID).base_position;
                        } else {
                            // Child shoot: get base from SAVED parent internode tip (petioles not reconstructed yet)
                            // Note: Cannot use parent petiole tip because petioles are reconstructed later in the loop
                            int parent_shoot_id = shoot_ptr->parent_shoot_ID;
                            uint parent_node_index = shoot_ptr->parent_node_index;
                            auto &parent_shoot = plant_instances.at(plantID).shoot_tree.at(parent_shoot_id);

                            // Use the saved internode tip from the parent phytomer
                            internode_base = parent_shoot->shoot_internode_vertices.at(parent_node_index).back();
                        }
                    } else {
                        // Subsequent phytomers: use previous phytomer's tip
                        internode_base = shoot_ptr->shoot_internode_vertices[phytomer_index_in_shoot - 1].back();
                    }

                    // Step 2: Recompute orientation vectors from parent context
                    helios::vec3 internode_axis_initial;
                    helios::vec3 petiole_rotation_axis;
                    helios::vec3 shoot_bending_axis;

                    recomputeInternodeOrientationVectors_local(phytomer_ptr, phytomer_index_in_shoot, deg2rad(internode_pitch), deg2rad(internode_phyllotactic_angle), internode_axis_initial, petiole_rotation_axis, shoot_bending_axis);

                    // Step 3: Reconstruct geometry segment-by-segment
                    std::vector<helios::vec3> reconstructed_vertices(internode_length_segments + 1);
                    std::vector<float> reconstructed_radii(internode_length_segments + 1);

                    reconstructed_vertices[0] = internode_base;
                    reconstructed_radii[0] = internode_radius;

                    float dr = internode_length / float(internode_length_segments);
                    float dr_max = internode_length_max / float(internode_length_segments);

                    helios::vec3 internode_axis = internode_axis_initial;

                    for (int i = 1; i <= internode_length_segments; i++) {
                        // Apply gravitropic curvature + SAVED perturbations
                        if (phytomer_index_in_shoot > 0 && !curvature_perturbations.empty()) {
                            // Compute curvature factor (lines 1633-1636 in PlantArchitecture.cpp)
                            float current_curvature_fact = 0.5f - internode_axis.z / 2.0f;
                            if (internode_axis.z < 0) {
                                current_curvature_fact *= 2.0f;
                            }

                            // Get gravitropic curvature from shoot
                            float gravitropic_curvature = shoot_ptr->gravitropic_curvature;

                            // Apply curvature with SAVED perturbation (matches line 1646-1647)
                            float curvature_angle = deg2rad(gravitropic_curvature * current_curvature_fact * dr_max + curvature_perturbations[i - 1]);
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, shoot_bending_axis, curvature_angle);

                            // Apply yaw perturbation if available (matches line 1651-1652)
                            if (!yaw_perturbations.empty() && (i - 1) < yaw_perturbations.size()) {
                                float yaw_angle = deg2rad(yaw_perturbations[i - 1]);
                                internode_axis = rotatePointAboutLine(internode_axis, nullorigin, make_vec3(0, 0, 1), yaw_angle);
                            }
                        }

                        // NOTE: Skip collision avoidance and attraction (environment-dependent, lines 1655-1693)

                        // Position next vertex
                        reconstructed_vertices[i] = reconstructed_vertices[i - 1] + dr * internode_axis;
                        reconstructed_radii[i] = internode_radius;
                    }

                    // Use reconstructed geometry
                    shoot_ptr->shoot_internode_vertices[phytomer_index_in_shoot] = reconstructed_vertices;
                    shoot_ptr->shoot_internode_radii[phytomer_index_in_shoot] = reconstructed_radii;
                } else if (!internode_vertices.empty() && !internode_radii.empty()) {
                    // Fallback: use saved geometry if no reconstruction parameters available (backward compatibility)
                    shoot_ptr->shoot_internode_vertices[phytomer_index_in_shoot] = internode_vertices;
                    shoot_ptr->shoot_internode_radii[phytomer_index_in_shoot] = internode_radii;
                }

                // Helper function to recompute petiole orientation vectors from parent phytomer context
                auto recomputePetioleOrientationVectors = [this, plantID](std::shared_ptr<Phytomer> phytomer_ptr, uint petiole_index, uint phytomer_index_in_shoot, float petiole_pitch_rad, float internode_phyllotactic_angle_rad,
                                                                          helios::vec3 &out_petiole_axis_initial, helios::vec3 &out_petiole_rotation_axis) {
                    using helios::make_vec3;
                    using helios::rotatePointAboutLine;
                    using helios::nullorigin;
                    using helios::cross;

                    // Step 1: Get parent axes for rotation axis computation
                    helios::vec3 parent_internode_axis = make_vec3(0, 0, 1); // default for base shoot
                    helios::vec3 parent_petiole_axis = make_vec3(0, -1, 0); // default for base shoot

                    // Get actual parent axes if available
                    if (phytomer_index_in_shoot > 0) {
                        // Previous phytomer in same shoot
                        auto prev_phytomer = phytomer_ptr->parent_shoot_ptr->phytomers.at(phytomer_index_in_shoot - 1);
                        parent_internode_axis = prev_phytomer->getInternodeAxisVector(1.0f);
                        parent_petiole_axis = prev_phytomer->getPetioleAxisVector(0.f, 0);
                    } else if (phytomer_ptr->parent_shoot_ptr->parent_shoot_ID >= 0) {
                        // First phytomer of child shoot - get from parent phytomer via shoot_tree
                        int parent_shoot_id = phytomer_ptr->parent_shoot_ptr->parent_shoot_ID;
                        uint parent_node_index = phytomer_ptr->parent_shoot_ptr->parent_node_index;
                        uint parent_petiole_index = phytomer_ptr->parent_shoot_ptr->parent_petiole_index;

                        auto &parent_shoot = plant_instances.at(plantID).shoot_tree.at(parent_shoot_id);
                        auto parent_phytomer = parent_shoot->phytomers.at(parent_node_index);
                        parent_internode_axis = parent_phytomer->getInternodeAxisVector(1.0f);
                        parent_petiole_axis = parent_phytomer->getPetioleAxisVector(0.f, parent_petiole_index);
                    }

                    // Step 2: Compute base petiole_rotation_axis from parent axes (matches line 1530)
                    helios::vec3 petiole_rotation_axis = cross(parent_internode_axis, parent_petiole_axis);
                    if (petiole_rotation_axis.magnitude() < 1e-6f) {
                        petiole_rotation_axis = make_vec3(1, 0, 0);
                    } else {
                        petiole_rotation_axis.normalize();
                    }

                    // Step 3: Compute internode axis with CORRECT order (matches lines 1528-1578)
                    helios::vec3 internode_axis = parent_internode_axis;
                    float internode_pitch_rad = phytomer_ptr->internode_pitch;


                    if (phytomer_index_in_shoot == 0) {
                        // First phytomer: apply rotations in CORRECT order

                        // 1. Internode pitch FIRST (line 1539) using UNMODIFIED petiole_rotation_axis
                        if (internode_pitch_rad != 0.f) {
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, 0.5f * internode_pitch_rad);
                        }

                        // 2. THEN apply shoot base rotations to BOTH axes (lines 1548-1564)
                        AxisRotation shoot_base_rotation = phytomer_ptr->parent_shoot_ptr->base_rotation;

                        if (shoot_base_rotation.roll != 0.f) {
                            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll);
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll);
                        }

                        if (shoot_base_rotation.pitch != 0.f) {
                            helios::vec3 base_pitch_axis = -1.0f * cross(parent_internode_axis, parent_petiole_axis);
                            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
                        }

                        if (shoot_base_rotation.yaw != 0.f) {
                            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
                        }
                    } else {
                        // Non-first phytomer: apply internode pitch (scaled by -1.25, line 1576)
                        if (internode_pitch_rad != 0.f) {
                            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, -1.25f * internode_pitch_rad);
                        }
                    }


                    // Step 5: Start with internode axis (matches line 1739)
                    helios::vec3 petiole_axis = internode_axis;

                    // Step 6: Apply petiole pitch rotation (matches line 1753)
                    petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, petiole_rotation_axis, std::abs(petiole_pitch_rad));

                    // Step 7: Apply phyllotactic rotation (for non-first phytomers, matches line 1758-1760)
                    if (phytomer_index_in_shoot != 0 && std::abs(internode_phyllotactic_angle_rad) > 0) {
                        petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, internode_axis, internode_phyllotactic_angle_rad);
                        petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis, internode_phyllotactic_angle_rad);
                    }

                    // Step 8: Apply bud rotation (for multi-petiole phytomers, matches line 1770-1773)
                    if (petiole_index > 0) {
                        uint petioles_per_internode = phytomer_ptr->phytomer_parameters.petiole.petioles_per_internode;
                        float budrot = float(petiole_index) * 2.0f * float(M_PI) / float(petioles_per_internode);
                        petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, internode_axis, budrot);
                        petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis, budrot);
                    }

                    // Return computed vectors
                    out_petiole_axis_initial = petiole_axis;
                    out_petiole_rotation_axis = petiole_rotation_axis;
                };

                // Lambda to recompute peduncle orientation vectors from parent context
                auto recomputePeduncleOrientationVectors = [this, plantID](std::shared_ptr<Phytomer> phytomer_ptr, uint petiole_index, uint phytomer_index_in_shoot, float peduncle_pitch_rad, float peduncle_roll_rad, const AxisRotation &base_rotation,
                                                                           vec3 &out_peduncle_axis_initial, vec3 &out_peduncle_rotation_axis) -> void {
                    // Step 1: Get parent internode axis at tip (fraction=1.0)
                    vec3 peduncle_axis = phytomer_ptr->getAxisVector(1.f, phytomer_ptr->getInternodeNodePositions());

                    // Step 2: Compute inflorescence_bending_axis from parent context
                    // This logic mirrors PlantArchitecture.cpp lines 976-1009 for parent_internode_axis
                    // and line 2041 for inflorescence_bending_axis
                    vec3 parent_internode_axis = make_vec3(0, 0, 1); // default for base shoot

                    // Get actual parent internode axis if available
                    if (phytomer_index_in_shoot > 0) {
                        // Previous phytomer in same shoot
                        auto prev_phytomer = phytomer_ptr->parent_shoot_ptr->phytomers.at(phytomer_index_in_shoot - 1);
                        parent_internode_axis = prev_phytomer->getInternodeAxisVector(1.0f);
                    } else if (phytomer_ptr->parent_shoot_ptr->parent_shoot_ID >= 0) {
                        // First phytomer of child shoot - get from parent phytomer via shoot_tree
                        int parent_shoot_id = phytomer_ptr->parent_shoot_ptr->parent_shoot_ID;
                        uint parent_node_index = phytomer_ptr->parent_shoot_ptr->parent_node_index;

                        auto &parent_shoot = plant_instances.at(plantID).shoot_tree.at(parent_shoot_id);
                        auto parent_phytomer = parent_shoot->phytomers.at(parent_node_index);
                        parent_internode_axis = parent_phytomer->getInternodeAxisVector(1.0f);
                    }

                    // Get current phytomer's petiole axis (NOT parent phytomer's petiole axis)
                    vec3 current_petiole_axis;
                    if (phytomer_ptr->petiole_vertices.empty()) {
                        current_petiole_axis = parent_internode_axis;
                    } else {
                        // Use actual curved petiole geometry at base (fraction=0.f) to match generation behavior
                        current_petiole_axis = phytomer_ptr->getPetioleAxisVector(0.f, petiole_index);
                    }

                    vec3 inflorescence_bending_axis_actual = cross(parent_internode_axis, current_petiole_axis);
                    if (inflorescence_bending_axis_actual.magnitude() < 0.001f) {
                        inflorescence_bending_axis_actual = make_vec3(1, 0, 0);
                    }
                    inflorescence_bending_axis_actual.normalize();

                    // Step 3: Apply peduncle pitch rotation
                    if (peduncle_pitch_rad != 0.f || base_rotation.pitch != 0.f) {
                        peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, inflorescence_bending_axis_actual, peduncle_pitch_rad + base_rotation.pitch);
                    }

                    // Step 4: Apply azimuthal alignment to parent petiole
                    vec3 internode_axis = phytomer_ptr->getAxisVector(1.f, phytomer_ptr->getInternodeNodePositions());
                    vec3 parent_petiole_base_axis = phytomer_ptr->petiole_vertices.empty() ? internode_axis : phytomer_ptr->getPetioleAxisVector(0.f, petiole_index);

                    float parent_petiole_azimuth = -std::atan2(parent_petiole_base_axis.y, parent_petiole_base_axis.x);
                    float current_peduncle_azimuth = -std::atan2(peduncle_axis.y, peduncle_axis.x);
                    float azimuthal_rotation = current_peduncle_azimuth - parent_petiole_azimuth;

                    peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, internode_axis, azimuthal_rotation);
                    inflorescence_bending_axis_actual = rotatePointAboutLine(inflorescence_bending_axis_actual, nullorigin, internode_axis, azimuthal_rotation);

                    // Step 5: Return computed vectors
                    out_peduncle_axis_initial = peduncle_axis;
                    out_peduncle_rotation_axis = inflorescence_bending_axis_actual;
                };

                // Restore petiole properties and reconstruct geometry from bulk parameters
                for (size_t p = 0; p < petiole_lengths.size(); p++) {
                    if (p < phytomer_ptr->petiole_length.size()) {

                        // Store bulk parameters in phytomer
                        phytomer_ptr->petiole_length.at(p) = petiole_lengths[p];
                        phytomer_ptr->petiole_pitch.at(p) = deg2rad(petiole_pitches[p]);
                        phytomer_ptr->petiole_curvature.at(p) = petiole_curvatures[p];
                        phytomer_ptr->petiole_taper.at(p) = petiole_tapers[p];
                        phytomer_ptr->current_leaf_scale_factor.at(p) = current_leaf_scale_factors[p];

                        // Update phytomer parameters for geometry construction
                        phytomer_ptr->phytomer_parameters.petiole.length_segments = petiole_length_segments_all[p];
                        phytomer_ptr->phytomer_parameters.petiole.radial_subdivisions = petiole_radial_subdivisions_all[p];

                        // Reconstruct petiole vertices and radii using exact construction algorithm
                        uint Ndiv_petiole_length = std::max(uint(1), petiole_length_segments_all[p]);
                        uint Ndiv_petiole_radius = std::max(uint(3), petiole_radial_subdivisions_all[p]);

                        phytomer_ptr->petiole_vertices.at(p).resize(Ndiv_petiole_length + 1);
                        phytomer_ptr->petiole_radii.at(p).resize(Ndiv_petiole_length + 1);

                        // Set base vertex and radius
                        // Auto-calculate petiole base from current phytomer's internode tip
                        phytomer_ptr->petiole_vertices.at(p).at(0) = shoot_ptr->shoot_internode_vertices[phytomer_index_in_shoot].back();
                        phytomer_ptr->petiole_radii.at(p).at(0) = petiole_radii_values[p];

                        // Compute segment length
                        float dr_petiole = petiole_lengths[p] / float(Ndiv_petiole_length);

                        // Recompute orientation vectors from parent context
                        vec3 recomputed_axis;
                        vec3 recomputed_rotation_axis;

                        recomputePetioleOrientationVectors(phytomer_ptr,
                                                           p, // petiole index
                                                           phytomer_index_in_shoot,
                                                           phytomer_ptr->petiole_pitch.at(p), // already in radians
                                                           phytomer_ptr->internode_phyllotactic_angle, // already in radians
                                                           recomputed_axis, recomputed_rotation_axis);

                        // Validate against saved vectors

                        // Use RECOMPUTED vectors for reconstruction (not saved ones)
                        vec3 petiole_axis_actual = recomputed_axis;
                        vec3 petiole_rotation_axis_actual = recomputed_rotation_axis;

                        // Create segments with curvature (matches construction algorithm at line 1830-1837)
                        for (int j = 1; j <= Ndiv_petiole_length; j++) {
                            // Apply curvature rotation
                            if (fabs(petiole_curvatures[p]) > 0) {
                                petiole_axis_actual = rotatePointAboutLine(petiole_axis_actual, nullorigin, petiole_rotation_axis_actual, -deg2rad(petiole_curvatures[p] * dr_petiole));
                            }

                            // Position next vertex
                            phytomer_ptr->petiole_vertices.at(p).at(j) = phytomer_ptr->petiole_vertices.at(p).at(j - 1) + dr_petiole * petiole_axis_actual;

                            // Apply taper to radius
                            phytomer_ptr->petiole_radii.at(p).at(j) = current_leaf_scale_factors[p] * petiole_radii_values[p] * (1.0f - petiole_tapers[p] / float(Ndiv_petiole_length) * float(j));
                        }

                        // Rebuild petiole Context geometry
                        context_ptr->deleteObject(phytomer_ptr->petiole_objIDs[p]);
                        std::vector<RGBcolor> petiole_colors(phytomer_ptr->petiole_radii[p].size(), phytomer_ptr->phytomer_parameters.petiole.color);
                        phytomer_ptr->petiole_objIDs[p] = makeTubeFromCones(Ndiv_petiole_radius, phytomer_ptr->petiole_vertices[p], phytomer_ptr->petiole_radii[p], petiole_colors, context_ptr);

                        // Set primitive data labels
                        if (!phytomer_ptr->petiole_objIDs[p].empty()) {
                            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(phytomer_ptr->petiole_objIDs[p]), "object_label", "petiole");
                            std::string petiole_material_name = plant_instances.at(plantID).plant_name + "_" + shoot_type_label + "_petiole";
                            renameAutoMaterial(context_ptr, phytomer_ptr->petiole_objIDs[p], petiole_material_name);
                        }

                        // Restore saved leaf base positions (already in saved_leaf_bases_all_petioles)
                        if (p < saved_leaf_bases_all_petioles.size() && !saved_leaf_bases_all_petioles[p].empty()) {
                            for (size_t leaf = 0; leaf < phytomer_ptr->leaf_bases[p].size(); leaf++) {
                                if (leaf < saved_leaf_bases_all_petioles[p].size()) {
                                    phytomer_ptr->leaf_bases[p][leaf] = saved_leaf_bases_all_petioles[p][leaf];
                                }
                            }
                        }
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
                                std::string leaf_material_name = plant_instances.at(plantID).plant_name + "_" + shoot_type_label + "_leaf";
                                renameAutoMaterial(context_ptr, objID_leaf, leaf_material_name);
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

                        // Auto-calculate leaf base from petiole geometry (handles compound leaves)
                        vec3 leaf_base = phytomer_ptr->petiole_vertices[petiole].back(); // Default: petiole tip

                        int leaves_per_petiole = phytomer_ptr->phytomer_parameters.leaf.leaves_per_petiole.val();
                        float leaflet_offset_val = clampOffset(leaves_per_petiole, phytomer_ptr->phytomer_parameters.leaf.leaflet_offset.val());

                        if (leaves_per_petiole > 1 && leaflet_offset_val > 0) {
                            // Compound leaf: calculate lateral leaflet positions along petiole
                            int ind_from_tip = leaf - int(floor(float(leaves_per_petiole - 1) / 2.f));
                            if (ind_from_tip != 0) { // Terminal leaflet stays at tip
                                float offset = (fabs(ind_from_tip) - 0.5f) * leaflet_offset_val * phytomer_ptr->phytomer_parameters.petiole.length.val();
                                leaf_base = PlantArchitecture::interpolateTube(phytomer_ptr->petiole_vertices[petiole], 1.f - offset / phytomer_ptr->phytomer_parameters.petiole.length.val());
                            }
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
                            fbud.current_fruit_scale_factor = fbud_data.current_fruit_scale_factor;

                            // Auto-calculate floral bud base_position and base_rotation from parent geometry
                            // Position: terminal buds attach at internode tip, axillary buds at petiole base
                            // Rotation: computed from bud_index and number of buds (matches growth algorithm)
                            if (fbud.isterminal) {
                                // Terminal buds attach at the tip of this phytomer's internode
                                fbud.base_position = phytomer_ptr->parent_shoot_ptr->shoot_internode_vertices.back().back();
                            } else {
                                // Axillary buds attach at the base of the parent petiole
                                if (petiole < phytomer_ptr->petiole_vertices.size()) {
                                    fbud.base_position = phytomer_ptr->petiole_vertices.at(petiole).front();
                                } else {
                                    helios_runtime_error("ERROR (PlantArchitecture::readPlantStructureXML): Floral bud parent_index " + std::to_string(fbud.parent_index) + " exceeds number of petioles.");
                                }
                            }

                            // Auto-calculate base_rotation from bud configuration
                            // Rotation depends on bud_index and total number of buds per petiole/shoot
                            uint Nbuds = floral_bud_data.at(petiole).size();
                            if (fbud.isterminal) {
                                // Terminal bud rotation formula (from PlantArchitecture.cpp:1243-1249)
                                float pitch_adjustment = (Nbuds > 1) ? deg2rad(30.f) : 0.f;
                                float yaw_adjustment = static_cast<float>(fbud.bud_index) * 2.f * M_PI / float(Nbuds);
                                fbud.base_rotation = make_AxisRotation(pitch_adjustment, yaw_adjustment, 0.f);
                            } else {
                                // Axillary bud rotation formula (from PlantArchitecture.cpp:1904-1906)
                                float pitch_adjustment = static_cast<float>(fbud.bud_index) * 0.1f * M_PI / float(Nbuds);
                                float yaw_adjustment = -0.25f * M_PI + static_cast<float>(fbud.bud_index) * 0.5f * M_PI / float(Nbuds);
                                fbud.base_rotation = make_AxisRotation(pitch_adjustment, yaw_adjustment, 0.f);
                            }

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

                                    // --- PEDUNCLE GEOMETRY RECONSTRUCTION ---
                                    if (fbud_data.peduncle_length > 0 && fbud_data.peduncle_radius > 0) {

                                        // Compute number of segments
                                        uint Ndiv_peduncle_length = std::max(uint(1), phytomer_ptr->phytomer_parameters.peduncle.length_segments);
                                        uint Ndiv_peduncle_radius = std::max(uint(3), phytomer_ptr->phytomer_parameters.peduncle.radial_subdivisions);

                                        // Initialize arrays
                                        std::vector<vec3> peduncle_vertices_computed(Ndiv_peduncle_length + 1);
                                        std::vector<float> peduncle_radii_computed(Ndiv_peduncle_length + 1);

                                        // Set base position and radius
                                        peduncle_vertices_computed.at(0) = fbud.base_position;
                                        peduncle_radii_computed.at(0) = fbud_data.peduncle_radius;

                                        float dr_peduncle = fbud_data.peduncle_length / float(Ndiv_peduncle_length);

                                        // Compute peduncle axis from parameters
                                        vec3 peduncle_axis_computed;
                                        vec3 peduncle_rotation_axis_computed;

                                        recomputePeduncleOrientationVectors(phytomer_ptr, fbud.parent_index, phytomer_index_in_shoot, deg2rad(fbud_data.peduncle_pitch), deg2rad(fbud_data.peduncle_roll), fbud.base_rotation, peduncle_axis_computed,
                                                                            peduncle_rotation_axis_computed);

                                        // Use computed value for reconstruction
                                        vec3 peduncle_axis_actual = peduncle_axis_computed;

                                        // Generate vertices with peduncle curvature algorithm
                                        for (int i = 1; i <= Ndiv_peduncle_length; i++) {
                                            if (fabs(fbud_data.peduncle_curvature) > 0) {
                                                float curvature_value = fbud_data.peduncle_curvature;

                                                // Horizontal bending axis perpendicular to peduncle direction
                                                vec3 horizontal_bending_axis = cross(peduncle_axis_actual, make_vec3(0, 0, 1));
                                                float axis_magnitude = horizontal_bending_axis.magnitude();

                                                if (axis_magnitude > 0.001f) {
                                                    horizontal_bending_axis = horizontal_bending_axis / axis_magnitude;

                                                    float theta_curvature = deg2rad(curvature_value * dr_peduncle);
                                                    float theta_from_target = (curvature_value > 0) ? std::acos(std::min(1.0f, std::max(-1.0f, peduncle_axis_actual.z))) : std::acos(std::min(1.0f, std::max(-1.0f, -peduncle_axis_actual.z)));

                                                    if (fabs(theta_curvature) >= theta_from_target) {
                                                        peduncle_axis_actual = (curvature_value > 0) ? make_vec3(0, 0, 1) : make_vec3(0, 0, -1);
                                                    } else {
                                                        peduncle_axis_actual = rotatePointAboutLine(peduncle_axis_actual, nullorigin, horizontal_bending_axis, theta_curvature);
                                                        peduncle_axis_actual.normalize();
                                                    }
                                                } else {
                                                    peduncle_axis_actual = (curvature_value > 0) ? make_vec3(0, 0, 1) : make_vec3(0, 0, -1);
                                                }
                                            }

                                            peduncle_vertices_computed.at(i) = peduncle_vertices_computed.at(i - 1) + dr_peduncle * peduncle_axis_actual;
                                            peduncle_radii_computed.at(i) = fbud_data.peduncle_radius;
                                        }

                                        // Store computed geometry
                                        if (petiole < phytomer_ptr->peduncle_vertices.size()) {
                                            if (phytomer_ptr->peduncle_vertices.at(petiole).size() <= bud) {
                                                phytomer_ptr->peduncle_vertices.at(petiole).resize(bud + 1);
                                                phytomer_ptr->peduncle_radii.at(petiole).resize(bud + 1);
                                            }
                                            phytomer_ptr->peduncle_vertices.at(petiole).at(bud) = peduncle_vertices_computed;
                                            phytomer_ptr->peduncle_radii.at(petiole).at(bud) = peduncle_radii_computed;
                                        }

                                        // Rebuild Context geometry with COMPUTED vertices/radii
                                        if (phytomer_ptr->build_context_geometry_peduncle) {
                                            std::vector<RGBcolor> colors(peduncle_vertices_computed.size(), phytomer_ptr->phytomer_parameters.peduncle.color);
                                            fbud.peduncle_objIDs.push_back(context_ptr->addTubeObject(Ndiv_peduncle_radius, peduncle_vertices_computed, peduncle_radii_computed, colors));
                                            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(fbud.peduncle_objIDs.back()), "object_label", "peduncle");
                                            std::string peduncle_material_name = plant_instances.at(plantID).plant_name + "_" + shoot_type_label + "_peduncle";
                                            renameAutoMaterial(context_ptr, fbud.peduncle_objIDs.back(), peduncle_material_name);
                                        }
                                    }

                                    // Restore flower geometry using saved rotations (base positions auto-computed)
                                    if (!fbud_data.flower_rotations.empty()) {

                                        // Ensure prototype maps are initialized before creating geometry
                                        ensureInflorescencePrototypesInitialized(phytomer_ptr->phytomer_parameters, plant_instances.at(plantID).plant_name);

                                        for (size_t i = 0; i < fbud_data.flower_rotations.size(); i++) {
                                            // Get saved base if available (for backward compatibility), otherwise will compute
                                            vec3 flower_base_saved = (i < fbud_data.inflorescence_bases_saved.size()) ? fbud_data.inflorescence_bases_saved.at(i) : make_vec3(0, 0, 0);
                                            float saved_pitch = deg2rad(fbud_data.flower_rotations.at(i).pitch);
                                            float saved_yaw = deg2rad(fbud_data.flower_rotations.at(i).yaw);
                                            float saved_roll = deg2rad(fbud_data.flower_rotations.at(i).roll);
                                            float saved_azimuth = deg2rad(fbud_data.flower_rotations.at(i).azimuth);

                                            // Compute flower base position from parameters
                                            int flowers_per_peduncle = fbud_data.flower_rotations.size();
                                            int petioles_per_internode = phytomer_ptr->phytomer_parameters.petiole.petioles_per_internode;

                                            // Clamp offset and compute position
                                            float flower_offset_clamped = clampOffset(flowers_per_peduncle, fbud_data.flower_offset);
                                            float ind_from_tip_computed = fabs(float(i) - float(flowers_per_peduncle - 1) / float(petioles_per_internode));

                                            // Default position: peduncle tip
                                            vec3 flower_base_computed = phytomer_ptr->peduncle_vertices.at(petiole).at(bud).back();

                                            // Compute position along peduncle if offset is non-zero
                                            if (flowers_per_peduncle > 1 && flower_offset_clamped > 0) {
                                                if (ind_from_tip_computed != 0) {
                                                    float offset_computed = (ind_from_tip_computed - 0.5f) * flower_offset_clamped * fbud_data.peduncle_length;
                                                    float frac_computed = 1.0f;
                                                    if (fbud_data.peduncle_length > 0) {
                                                        frac_computed = 1.f - offset_computed / fbud_data.peduncle_length;
                                                    }
                                                    flower_base_computed = interpolateTube(phytomer_ptr->peduncle_vertices.at(petiole).at(bud), frac_computed);
                                                }
                                            }

                                            // Recalculate peduncle_axis from saved peduncle geometry and flower position
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
                                            vec3 recalculated_peduncle_axis = phytomer_ptr->getAxisVector(frac, phytomer_ptr->peduncle_vertices.at(petiole).at(bud));

                                            // Use individual base scale if available, otherwise use default parameter
                                            float scale_factor;
                                            if (i < fbud_data.flower_base_scales.size() && fbud_data.flower_base_scales.at(i) >= 0) {
                                                scale_factor = fbud_data.flower_base_scales.at(i);
                                            } else {
                                                scale_factor = phytomer_ptr->phytomer_parameters.inflorescence.flower_prototype_scale.val();
                                            }

                                            // Determine if flower is open
                                            bool is_open_flower = (desired_state == BUD_FLOWER_OPEN);

                                            // Create flower geometry with computed base, saved rotations, and recalculated peduncle axis
                                            phytomer_ptr->createInflorescenceGeometry(fbud, flower_base_computed, recalculated_peduncle_axis, saved_pitch, saved_roll, saved_azimuth, saved_yaw, scale_factor, is_open_flower);
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

                                    // --- PEDUNCLE GEOMETRY RECONSTRUCTION ---
                                    if (fbud_data.peduncle_length > 0 && fbud_data.peduncle_radius > 0) {

                                        // Compute number of segments
                                        uint Ndiv_peduncle_length = std::max(uint(1), phytomer_ptr->phytomer_parameters.peduncle.length_segments);
                                        uint Ndiv_peduncle_radius = std::max(uint(3), phytomer_ptr->phytomer_parameters.peduncle.radial_subdivisions);

                                        // Initialize arrays
                                        std::vector<vec3> peduncle_vertices_computed(Ndiv_peduncle_length + 1);
                                        std::vector<float> peduncle_radii_computed(Ndiv_peduncle_length + 1);

                                        // Set base position and radius
                                        peduncle_vertices_computed.at(0) = fbud.base_position;
                                        peduncle_radii_computed.at(0) = fbud_data.peduncle_radius;

                                        float dr_peduncle = fbud_data.peduncle_length / float(Ndiv_peduncle_length);

                                        // Compute peduncle axis from parameters
                                        vec3 peduncle_axis_computed;
                                        vec3 peduncle_rotation_axis_computed;

                                        recomputePeduncleOrientationVectors(phytomer_ptr, fbud.parent_index, phytomer_index_in_shoot, deg2rad(fbud_data.peduncle_pitch), deg2rad(fbud_data.peduncle_roll), fbud.base_rotation, peduncle_axis_computed,
                                                                            peduncle_rotation_axis_computed);

                                        // Use computed value for reconstruction
                                        vec3 peduncle_axis_actual = peduncle_axis_computed;

                                        // Generate vertices with peduncle curvature algorithm
                                        for (int i = 1; i <= Ndiv_peduncle_length; i++) {
                                            if (fabs(fbud_data.peduncle_curvature) > 0) {
                                                float curvature_value = fbud_data.peduncle_curvature;

                                                // Horizontal bending axis perpendicular to peduncle direction
                                                vec3 horizontal_bending_axis = cross(peduncle_axis_actual, make_vec3(0, 0, 1));
                                                float axis_magnitude = horizontal_bending_axis.magnitude();

                                                if (axis_magnitude > 0.001f) {
                                                    horizontal_bending_axis = horizontal_bending_axis / axis_magnitude;

                                                    float theta_curvature = deg2rad(curvature_value * dr_peduncle);
                                                    float theta_from_target = (curvature_value > 0) ? std::acos(std::min(1.0f, std::max(-1.0f, peduncle_axis_actual.z))) : std::acos(std::min(1.0f, std::max(-1.0f, -peduncle_axis_actual.z)));

                                                    if (fabs(theta_curvature) >= theta_from_target) {
                                                        peduncle_axis_actual = (curvature_value > 0) ? make_vec3(0, 0, 1) : make_vec3(0, 0, -1);
                                                    } else {
                                                        peduncle_axis_actual = rotatePointAboutLine(peduncle_axis_actual, nullorigin, horizontal_bending_axis, theta_curvature);
                                                        peduncle_axis_actual.normalize();
                                                    }
                                                } else {
                                                    peduncle_axis_actual = (curvature_value > 0) ? make_vec3(0, 0, 1) : make_vec3(0, 0, -1);
                                                }
                                            }

                                            peduncle_vertices_computed.at(i) = peduncle_vertices_computed.at(i - 1) + dr_peduncle * peduncle_axis_actual;
                                            peduncle_radii_computed.at(i) = fbud_data.peduncle_radius;
                                        }

                                        // Store computed geometry
                                        if (petiole < phytomer_ptr->peduncle_vertices.size()) {
                                            if (phytomer_ptr->peduncle_vertices.at(petiole).size() <= bud) {
                                                phytomer_ptr->peduncle_vertices.at(petiole).resize(bud + 1);
                                                phytomer_ptr->peduncle_radii.at(petiole).resize(bud + 1);
                                            }
                                            phytomer_ptr->peduncle_vertices.at(petiole).at(bud) = peduncle_vertices_computed;
                                            phytomer_ptr->peduncle_radii.at(petiole).at(bud) = peduncle_radii_computed;
                                        }

                                        // Rebuild Context geometry with COMPUTED vertices/radii
                                        if (phytomer_ptr->build_context_geometry_peduncle) {
                                            std::vector<RGBcolor> colors(peduncle_vertices_computed.size(), phytomer_ptr->phytomer_parameters.peduncle.color);
                                            fbud.peduncle_objIDs.push_back(context_ptr->addTubeObject(Ndiv_peduncle_radius, peduncle_vertices_computed, peduncle_radii_computed, colors));
                                            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(fbud.peduncle_objIDs.back()), "object_label", "peduncle");
                                            std::string peduncle_material_name = plant_instances.at(plantID).plant_name + "_" + shoot_type_label + "_peduncle";
                                            renameAutoMaterial(context_ptr, fbud.peduncle_objIDs.back(), peduncle_material_name);
                                        }
                                    }

                                    // Restore fruit geometry using saved rotations (base positions auto-computed)
                                    if (!fbud_data.flower_rotations.empty()) {

                                        // Ensure prototype maps are initialized before creating geometry
                                        ensureInflorescencePrototypesInitialized(phytomer_ptr->phytomer_parameters, plant_instances.at(plantID).plant_name);

                                        for (size_t i = 0; i < fbud_data.flower_rotations.size(); i++) {
                                            // Get saved base if available (for backward compatibility), otherwise will compute
                                            vec3 fruit_base_saved = (i < fbud_data.inflorescence_bases_saved.size()) ? fbud_data.inflorescence_bases_saved.at(i) : make_vec3(0, 0, 0);
                                            float saved_pitch = deg2rad(fbud_data.flower_rotations.at(i).pitch);
                                            float saved_yaw = deg2rad(fbud_data.flower_rotations.at(i).yaw);
                                            float saved_roll = deg2rad(fbud_data.flower_rotations.at(i).roll);
                                            float saved_azimuth = deg2rad(fbud_data.flower_rotations.at(i).azimuth);

                                            // Compute fruit base position from parameters
                                            int flowers_per_peduncle = fbud_data.flower_rotations.size();
                                            int petioles_per_internode = phytomer_ptr->phytomer_parameters.petiole.petioles_per_internode;

                                            // Clamp offset and compute position
                                            float flower_offset_clamped = clampOffset(flowers_per_peduncle, fbud_data.flower_offset);
                                            float ind_from_tip_computed = fabs(float(i) - float(flowers_per_peduncle - 1) / float(petioles_per_internode));

                                            // Default position: peduncle tip
                                            vec3 fruit_base_computed = phytomer_ptr->peduncle_vertices.at(petiole).at(bud).back();

                                            // Compute position along peduncle if offset is non-zero
                                            if (flowers_per_peduncle > 1 && flower_offset_clamped > 0) {
                                                if (ind_from_tip_computed != 0) {
                                                    float offset_computed = (ind_from_tip_computed - 0.5f) * flower_offset_clamped * fbud_data.peduncle_length;
                                                    float frac_computed = 1.0f;
                                                    if (fbud_data.peduncle_length > 0) {
                                                        frac_computed = 1.f - offset_computed / fbud_data.peduncle_length;
                                                    }
                                                    fruit_base_computed = interpolateTube(phytomer_ptr->peduncle_vertices.at(petiole).at(bud), frac_computed);
                                                }
                                            }

                                            // Verification complete - computation matches saved values within tolerance
                                            // (Verification code removed after Phase 1 testing confirmed correctness)

                                            // Recalculate peduncle_axis from saved peduncle geometry and fruit position
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
                                            vec3 recalculated_peduncle_axis = phytomer_ptr->getAxisVector(frac, phytomer_ptr->peduncle_vertices.at(petiole).at(bud));

                                            // Use individual base scale if available, then apply growth scaling
                                            float base_fruit_scale;
                                            if (i < fbud_data.flower_base_scales.size() && fbud_data.flower_base_scales.at(i) >= 0) {
                                                base_fruit_scale = fbud_data.flower_base_scales.at(i);
                                            } else {
                                                base_fruit_scale = phytomer_ptr->phytomer_parameters.inflorescence.fruit_prototype_scale.val();
                                            }
                                            float scale_factor = base_fruit_scale * fbud_data.current_fruit_scale_factor;

                                            // Create fruit geometry with computed base, saved rotations, and recalculated peduncle axis
                                            phytomer_ptr->createInflorescenceGeometry(fbud, fruit_base_computed, recalculated_peduncle_axis, saved_pitch, saved_roll, saved_azimuth, saved_yaw, scale_factor, false);
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

// ---- USD Articulated Rigid Body Export for IsaacSim ---- //

namespace {

enum USDLinkType { USD_LINK_CAPSULE, USD_LINK_MESH };

struct USDMaterial {
    std::string name;          // Unique material name for USD prim
    std::string texture_file;  // Absolute path to texture image (empty if color-only)
    RGBcolor color;            // Diffuse color (used when no texture, or as fallback)
};

struct USDLink {
    std::string name;
    vec3 position;          // World-space centroid (capsule midpoint or mesh centroid)
    float qw, qx, qy, qz;  // Orientation quaternion (rotation from Z-axis to segment axis)
    float radius;
    float half_length;       // Half-length of cylinder shaft
    float mass;
    vec3 inertia_diagonal;   // (Ixx, Iyy, Izz)
    int parent_link_index;   // -1 for root
    USDLinkType link_type = USD_LINK_CAPSULE;
    int material_index = -1; // Index into ArticulationData::materials (-1 = default physics material)

    // Mesh data (only used when link_type == USD_LINK_MESH)
    std::vector<vec3> mesh_vertices;   // In local space (centroid at origin)
    std::vector<vec2> mesh_uvs;        // Per-vertex UV coordinates
    std::vector<int> mesh_face_vertex_counts;
    std::vector<int> mesh_face_vertex_indices;
};

struct USDJoint {
    std::string name;
    int parent_link_index;
    int child_link_index;
    vec3 local_pos_parent;   // Joint anchor in parent local frame
    vec3 local_pos_child;    // Joint anchor in child local frame
    float stiffness;
    float damping;
    bool is_fixed;           // true for world anchor joint
    // Orientation of joint frame in child body's local space.
    // Encodes relative rotation so that zero joint angle reproduces the rest pose.
    // localRot0 is always identity; localRot1 = inverse(child_quat) * parent_quat.
    float local_rot1_w = 1, local_rot1_x = 0, local_rot1_y = 0, local_rot1_z = 0;
};

float computeCapsuleVolume(float radius, float half_length) {
    float cylinder_vol = M_PI * radius * radius * 2.f * half_length;
    float sphere_vol = (4.f / 3.f) * M_PI * radius * radius * radius;
    return cylinder_vol + sphere_vol;
}

vec3 computeCapsuleInertia(float mass, float radius, float half_length) {
    float L = 2.f * half_length;
    float Ixx = mass * (3.f * radius * radius + L * L) / 12.f;
    float Iyy = Ixx;
    float Izz = mass * radius * radius / 2.f;
    return {Ixx, Iyy, Izz};
}

vec3 computeSphereInertia(float mass, float radius) {
    float I = 0.4f * mass * radius * radius;
    return {I, I, I};
}

float computeJointStiffness(float E, float radius, float segment_length) {
    float I = M_PI * radius * radius * radius * radius / 4.f;
    return E * I / segment_length;
}

float computeJointDamping(float stiffness, float rotational_inertia, float damping_ratio) {
    return damping_ratio * 2.f * std::sqrt(stiffness * rotational_inertia);
}

void axisToQuaternion(const vec3 &axis_dir, float &qw, float &qx, float &qy, float &qz) {
    vec3 z_axis(0, 0, 1);
    vec3 a = axis_dir;
    float mag = a.magnitude();
    if (mag < 1e-8f) {
        qw = 1; qx = 0; qy = 0; qz = 0;
        return;
    }
    a = a / mag;

    float dot = z_axis.x * a.x + z_axis.y * a.y + z_axis.z * a.z;

    if (dot > 0.9999f) {
        qw = 1; qx = 0; qy = 0; qz = 0;
        return;
    }
    if (dot < -0.9999f) {
        qw = 0; qx = 1; qy = 0; qz = 0;
        return;
    }

    vec3 cross_prod;
    cross_prod.x = z_axis.y * a.z - z_axis.z * a.y;
    cross_prod.y = z_axis.z * a.x - z_axis.x * a.z;
    cross_prod.z = z_axis.x * a.y - z_axis.y * a.x;

    qw = 1.f + dot;
    qx = cross_prod.x;
    qy = cross_prod.y;
    qz = cross_prod.z;

    float norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;
}

// Quaternion multiplication: result = a * b (Hamilton product)
void quatMultiply(float aw, float ax, float ay, float az,
                  float bw, float bx, float by, float bz,
                  float &rw, float &rx, float &ry, float &rz) {
    rw = aw * bw - ax * bx - ay * by - az * bz;
    rx = aw * bx + ax * bw + ay * bz - az * by;
    ry = aw * by - ax * bz + ay * bw + az * bx;
    rz = aw * bz + ax * by - ay * bx + az * bw;
}

// Compute localRot1 for a joint connecting parent_link to child_link.
// localRot1 = inverse(child_quat) * parent_quat
// This ensures zero joint angle reproduces the rest pose where both bodies
// are at their authored world orientations.
void computeJointLocalRot1(const USDLink &parent, const USDLink &child, USDJoint &joint) {
    // conjugate of child quaternion = inverse for unit quaternions
    float inv_cw = child.qw, inv_cx = -child.qx, inv_cy = -child.qy, inv_cz = -child.qz;
    quatMultiply(inv_cw, inv_cx, inv_cy, inv_cz,
                 parent.qw, parent.qx, parent.qy, parent.qz,
                 joint.local_rot1_w, joint.local_rot1_x, joint.local_rot1_y, joint.local_rot1_z);
}

struct ArticulationData {
    std::vector<USDLink> links;
    std::vector<USDJoint> joints;
    std::vector<USDMaterial> materials;
    std::map<std::string, int> material_cache; // key -> material index
};

} // anonymous namespace (temporarily close to define GrowthFrameSnapshot)

// GrowthFrameSnapshot: wraps ArticulationData for growth animation frame storage.
// Defined outside the anonymous namespace so shared_ptr<GrowthFrameSnapshot> in the header works.
struct GrowthFrameSnapshot {
    float time;  // plant age in days at this frame
    ArticulationData artic_data;
};

namespace {

// Look up or create a material entry for the given texture/color combination.
int getOrCreateMaterial(ArticulationData &data, const std::string &texture_file, const RGBcolor &color) {
    std::string key = texture_file + "|" + std::to_string(color.r) + "," + std::to_string(color.g) + "," + std::to_string(color.b);
    auto it = data.material_cache.find(key);
    if (it != data.material_cache.end()) {
        return it->second;
    }
    int idx = static_cast<int>(data.materials.size());
    USDMaterial mat;
    mat.name = "Material_" + std::to_string(idx);
    mat.texture_file = texture_file;
    mat.color = color;
    data.materials.push_back(mat);
    data.material_cache[key] = idx;
    return idx;
}

// Get or create a material from a context object by looking up the texture and color
// of its first primitive. Returns the material index.
int getMaterialFromContextObject(Context *context_ptr, uint objID, ArticulationData &data) {
    if (!context_ptr->doesObjectExist(objID)) {
        return -1;
    }
    std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);
    if (UUIDs.empty()) {
        return -1;
    }
    std::string texture_file = context_ptr->getPrimitiveTextureFile(UUIDs[0]);
    RGBcolor color = context_ptr->getPrimitiveColor(UUIDs[0]);
    return getOrCreateMaterial(data, texture_file, color);
}

// Apply inverse quaternion rotation to transform a world-space point into the Xform's local frame.
// The Xform has translate + orient, so local = inverseRotate(world - translate).
// Uses the vector rotation formula: v' = v + 2*w*(u x v) + 2*(u x (u x v))
// where q = (w, u) and we use the conjugate q* = (w, -u) for inverse rotation.
vec3 worldToLocal(const vec3 &world_point, const vec3 &xform_position, float qw, float qx, float qy, float qz) {
    vec3 v = world_point - xform_position;
    // Conjugate quaternion for inverse rotation: (qw, -qx, -qy, -qz)
    vec3 u(-qx, -qy, -qz);
    vec3 uv;
    uv.x = u.y * v.z - u.z * v.y;
    uv.y = u.z * v.x - u.x * v.z;
    uv.z = u.x * v.y - u.y * v.x;
    vec3 uuv;
    uuv.x = u.y * uv.z - u.z * uv.y;
    uuv.y = u.z * uv.x - u.x * uv.z;
    uuv.z = u.x * uv.y - u.y * uv.x;
    vec3 result;
    result.x = v.x + 2.f * (qw * uv.x + uuv.x);
    result.y = v.y + 2.f * (qw * uv.y + uuv.y);
    result.z = v.z + 2.f * (qw * uv.z + uuv.z);
    return result;
}

// Build a visual mesh for a tube segment by extracting the primitives belonging to
// that segment from the tube compound object. For a tube with N nodes and R radial
// subdivisions, segment i (between node i and node i+1) has 2*R triangles.
// The UUIDs are ordered: outer loop j=0..R-1, inner loop i=0..N-2, 2 triangles per (j,i).
void buildTubeSegmentVisualMesh(Context *context_ptr, uint tube_objID, int segment_index,
                                int total_segments, ArticulationData &data, USDLink &link) {

    if (!context_ptr->doesObjectExist(tube_objID)) {
        return;
    }

    uint radial_subdivs = context_ptr->getTubeObjectSubdivisionCount(tube_objID);
    std::vector<uint> all_UUIDs = context_ptr->getObjectPrimitiveUUIDs(tube_objID);

    // Collect UUIDs for this segment: for each radial strip j, grab 2 triangles at position (j, segment_index)
    std::vector<uint> seg_UUIDs;
    seg_UUIDs.reserve(2 * radial_subdivs);
    for (uint j = 0; j < radial_subdivs; j++) {
        int base = static_cast<int>(j) * 2 * total_segments + 2 * segment_index;
        if (base + 1 < static_cast<int>(all_UUIDs.size())) {
            seg_UUIDs.push_back(all_UUIDs[base]);
            seg_UUIDs.push_back(all_UUIDs[base + 1]);
        }
    }

    if (seg_UUIDs.empty()) {
        return;
    }

    // Extract mesh data from these primitives
    std::vector<vec3> world_vertices;
    std::vector<vec2> all_uvs;
    std::string texture_file;
    RGBcolor color = make_RGBcolor(0.5f, 0.5f, 0.5f);

    for (uint UUID : seg_UUIDs) {
        std::vector<vec3> prim_verts = context_ptr->getPrimitiveVertices(UUID);
        std::vector<vec2> prim_uvs = context_ptr->getPrimitiveTextureUV(UUID);
        int base_idx = static_cast<int>(world_vertices.size());
        link.mesh_face_vertex_counts.push_back(static_cast<int>(prim_verts.size()));
        for (int i = 0; i < static_cast<int>(prim_verts.size()); i++) {
            link.mesh_face_vertex_indices.push_back(base_idx + i);
            world_vertices.push_back(prim_verts[i]);
            if (i < static_cast<int>(prim_uvs.size())) {
                all_uvs.push_back(prim_uvs[i]);
            } else {
                all_uvs.push_back(make_vec2(0, 0));
            }
        }

        if (texture_file.empty()) {
            texture_file = context_ptr->getPrimitiveTextureFile(UUID);
            color = context_ptr->getPrimitiveColor(UUID);
        }
    }

    // Convert to local space: translate to origin then apply inverse rotation
    link.mesh_vertices.resize(world_vertices.size());
    for (size_t i = 0; i < world_vertices.size(); i++) {
        link.mesh_vertices[i] = worldToLocal(world_vertices[i], link.position, link.qw, link.qx, link.qy, link.qz);
    }
    link.mesh_uvs = all_uvs;

    // Assign material
    link.material_index = getOrCreateMaterial(data, texture_file, color);
}

// Build mesh data from a Helios context object. Vertices are stored in local space
// (translated so the centroid is at origin). Returns the world-space centroid.
// Also extracts UV coordinates and determines the material.
vec3 buildMeshFromContextObject(Context *context_ptr, uint objID, ArticulationData &data,
                                USDLink &link) {

    std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);

    // Collect all vertices, UVs, and faces
    std::vector<vec3> world_vertices;
    std::vector<vec2> all_uvs;
    std::string texture_file;
    RGBcolor color = make_RGBcolor(0.5f, 0.5f, 0.5f);

    for (uint UUID : UUIDs) {
        std::vector<vec3> prim_verts = context_ptr->getPrimitiveVertices(UUID);
        std::vector<vec2> prim_uvs = context_ptr->getPrimitiveTextureUV(UUID);
        int base_idx = static_cast<int>(world_vertices.size());
        link.mesh_face_vertex_counts.push_back(static_cast<int>(prim_verts.size()));
        for (int i = 0; i < static_cast<int>(prim_verts.size()); i++) {
            link.mesh_face_vertex_indices.push_back(base_idx + i);
            world_vertices.push_back(prim_verts[i]);
            if (i < static_cast<int>(prim_uvs.size())) {
                all_uvs.push_back(prim_uvs[i]);
            } else {
                all_uvs.push_back(make_vec2(0, 0));
            }
        }

        // Get texture and color from first primitive
        if (texture_file.empty()) {
            texture_file = context_ptr->getPrimitiveTextureFile(UUID);
            color = context_ptr->getPrimitiveColor(UUID);
        }
    }

    // Compute centroid
    vec3 centroid(0, 0, 0);
    if (!world_vertices.empty()) {
        for (const auto &v : world_vertices) {
            centroid = centroid + v;
        }
        centroid = centroid / static_cast<float>(world_vertices.size());
    }

    // Convert to local space (centroid at origin)
    link.mesh_vertices.resize(world_vertices.size());
    for (size_t i = 0; i < world_vertices.size(); i++) {
        link.mesh_vertices[i] = world_vertices[i] - centroid;
    }
    link.mesh_uvs = all_uvs;

    // Assign material
    link.material_index = getOrCreateMaterial(data, texture_file, color);

    return centroid;
}

// Compute approximate inertia for a mesh by treating it as a collection of point masses
vec3 computeMeshInertia(float mass, const std::vector<vec3> &local_vertices) {
    if (local_vertices.empty()) {
        return {0, 0, 0};
    }
    float mass_per_vertex = mass / static_cast<float>(local_vertices.size());
    float Ixx = 0, Iyy = 0, Izz = 0;
    for (const auto &v : local_vertices) {
        Ixx += mass_per_vertex * (v.y * v.y + v.z * v.z);
        Iyy += mass_per_vertex * (v.x * v.x + v.z * v.z);
        Izz += mass_per_vertex * (v.x * v.x + v.y * v.y);
    }
    return {Ixx, Iyy, Izz};
}

std::string sanitizePrimName(const std::string &name) {
    std::string result;
    result.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(c) || c == '_') {
            result += c;
        } else {
            result += '_';
        }
    }
    return result;
}

ArticulationData buildArticulationData(const PlantInstance &plant, const USDExportParameters &params, Context *context_ptr) {

    ArticulationData data;

    std::map<std::pair<int, int>, int> last_internode_link_index;
    std::map<int, int> shoot_last_link_index;
    std::map<std::pair<int, int>, int> shoot_node_link_index;

    for (const auto &shoot : plant.shoot_tree) {

        int prev_internode_link = -1;
        vec3 last_internode_vertex;
        bool has_last_internode_vertex = false;

        // Compute total tube nodes/segments for this shoot's internode tube (for visual mesh extraction)
        int total_tube_nodes = 0;
        for (uint pi = 0; pi < shoot->shoot_internode_vertices.size(); pi++) {
            total_tube_nodes += static_cast<int>(shoot->shoot_internode_vertices[pi].size());
        }
        int total_tube_segments = std::max(0, total_tube_nodes - 1);
        int flat_node_offset = 0; // running offset into the flattened node array

        int shoot_parent_link = -1;
        if (shoot->parent_shoot_ID >= 0) {
            // Try exact node match first
            auto key = std::make_pair(shoot->parent_shoot_ID, static_cast<int>(shoot->parent_node_index));
            auto it = shoot_node_link_index.find(key);
            if (it != shoot_node_link_index.end()) {
                shoot_parent_link = it->second;
            } else {
                // Exact node was filtered out — fall back to nearest upstream link on the parent shoot
                auto shoot_it = shoot_last_link_index.find(shoot->parent_shoot_ID);
                if (shoot_it != shoot_last_link_index.end()) {
                    shoot_parent_link = shoot_it->second;
                }
                // If parent shoot produced no links at all, shoot_parent_link stays -1
                // and this child shoot will become a new root (or be skipped if no segments survive)
            }
        }

        for (uint phytomer_idx = 0; phytomer_idx < shoot->phytomers.size(); phytomer_idx++) {

            const auto &vertices = shoot->shoot_internode_vertices[phytomer_idx];
            const auto &radii = shoot->shoot_internode_radii[phytomer_idx];
            const auto &phytomer = shoot->phytomers[phytomer_idx];

            // Build list of segment vertex pairs, handling single-vertex phytomers
            // by using the previous phytomer's last vertex as the start point.
            // Also track the flat tube segment index for visual mesh extraction.
            std::vector<std::pair<vec3, vec3>> segment_pairs;
            std::vector<float> segment_radii;
            std::vector<int> segment_tube_indices; // flat tube segment index for each segment

            if (vertices.size() == 1 && has_last_internode_vertex) {
                // Vertex-sharing: segment between previous phytomer's last node and this node
                // In flat tube, that's segment (flat_node_offset - 1)
                segment_pairs.push_back({last_internode_vertex, vertices[0]});
                segment_radii.push_back(radii[0]);
                segment_tube_indices.push_back(flat_node_offset - 1);
            } else {
                for (int seg = 0; seg < static_cast<int>(vertices.size()) - 1; seg++) {
                    segment_pairs.push_back({vertices[seg], vertices[seg + 1]});
                    segment_radii.push_back(radii[seg]);
                    segment_tube_indices.push_back(flat_node_offset + seg);
                }
            }

            flat_node_offset += static_cast<int>(vertices.size());

            // Update last vertex for next phytomer's vertex sharing
            if (!vertices.empty()) {
                last_internode_vertex = vertices.back();
                has_last_internode_vertex = true;
            }

            for (int seg = 0; seg < static_cast<int>(segment_pairs.size()); seg++) {
                vec3 start = segment_pairs[seg].first;
                vec3 end = segment_pairs[seg].second;
                vec3 axis = end - start;
                float length = axis.magnitude();

                if (length < params.min_segment_length) {
                    continue;
                }

                float seg_radius = segment_radii[seg];
                if (!(seg_radius > 0) || seg_radius > length) {
                    continue;
                }

                vec3 midpoint = (start + end) * 0.5f;
                vec3 axis_normalized = axis / length;
                float half_len = length / 2.f;

                USDLink link;
                link.name = "S" + std::to_string(shoot->ID) + "_P" + std::to_string(phytomer_idx) + "_Seg" + std::to_string(seg);
                link.position = midpoint;
                axisToQuaternion(axis_normalized, link.qw, link.qx, link.qy, link.qz);
                link.radius = seg_radius;
                link.half_length = half_len;
                link.mass = params.wood_density * computeCapsuleVolume(seg_radius, half_len);
                link.inertia_diagonal = computeCapsuleInertia(link.mass, seg_radius, half_len);

                // Extract visual mesh from tube geometry for this segment
                int tube_seg_idx = segment_tube_indices[seg];
                if (tube_seg_idx >= 0 && tube_seg_idx < total_tube_segments) {
                    buildTubeSegmentVisualMesh(context_ptr, shoot->internode_tube_objID,
                                               tube_seg_idx, total_tube_segments, data, link);
                }

                int parent_idx;
                if (prev_internode_link >= 0) {
                    parent_idx = prev_internode_link;
                } else if (shoot_parent_link >= 0) {
                    parent_idx = shoot_parent_link;
                } else {
                    parent_idx = -1;
                }
                link.parent_link_index = parent_idx;

                int this_link_idx = static_cast<int>(data.links.size());
                data.links.push_back(link);

                USDJoint joint;
                joint.child_link_index = this_link_idx;

                if (parent_idx < 0) {
                    joint.name = "WorldAnchor";
                    joint.parent_link_index = -1;
                    joint.local_pos_parent = start;
                    joint.local_pos_child = vec3(0, 0, -half_len);
                    joint.stiffness = 0;
                    joint.damping = 0;
                    joint.is_fixed = true;
                } else {
                    joint.name = "J_" + data.links[parent_idx].name + "_to_" + link.name;
                    joint.parent_link_index = parent_idx;
                    joint.local_pos_parent = vec3(0, 0, data.links[parent_idx].half_length);
                    joint.local_pos_child = vec3(0, 0, -half_len);

                    float avg_radius = (data.links[parent_idx].radius + seg_radius) / 2.f;
                    joint.stiffness = computeJointStiffness(params.elastic_modulus, avg_radius, length);
                    float child_Izz = link.inertia_diagonal.x;
                    joint.damping = computeJointDamping(joint.stiffness, child_Izz, params.damping_ratio);
                    joint.is_fixed = false;
                    computeJointLocalRot1(data.links[parent_idx], data.links[this_link_idx], joint);
                }
                data.joints.push_back(joint);

                prev_internode_link = this_link_idx;
            }

            // Always record the best available link for this phytomer node, even if
            // all its segments were filtered. This ensures child shoots attaching at this
            // node can find the nearest upstream link.
            if (prev_internode_link >= 0) {
                last_internode_link_index[{shoot->ID, static_cast<int>(phytomer_idx)}] = prev_internode_link;
                shoot_node_link_index[{shoot->ID, static_cast<int>(phytomer_idx)}] = prev_internode_link;
                shoot_last_link_index[shoot->ID] = prev_internode_link;
            }

            // ---- Petioles (only if leaves are present) ---- //

            int internode_attach_link = prev_internode_link;

            for (uint petiole_idx = 0; petiole_idx < phytomer->petiole_vertices.size(); petiole_idx++) {

                // Skip petioles that have no attached leaves
                bool has_leaves = false;
                if (petiole_idx < phytomer->leaf_objIDs.size()) {
                    for (uint li = 0; li < phytomer->leaf_objIDs[petiole_idx].size(); li++) {
                        if (phytomer->leaf_objIDs[petiole_idx][li] != 0) {
                            has_leaves = true;
                            break;
                        }
                    }
                }
                if (!has_leaves) {
                    continue;
                }

                const auto &pet_verts = phytomer->petiole_vertices[petiole_idx];
                const auto &pet_radii = phytomer->petiole_radii[petiole_idx];
                int prev_petiole_link = -1;

                for (int seg = 0; seg < static_cast<int>(pet_verts.size()) - 1; seg++) {
                    vec3 start = pet_verts[seg];
                    vec3 end = pet_verts[seg + 1];
                    vec3 axis = end - start;
                    float length = axis.magnitude();

                    if (length < params.min_segment_length) {
                        continue;
                    }

                    float seg_radius = pet_radii[seg];
                    if (!(seg_radius > 0) || seg_radius > length) {
                        continue;
                    }

                    vec3 midpoint = (start + end) * 0.5f;
                    vec3 axis_normalized = axis / length;
                    float half_len = length / 2.f;

                    USDLink link;
                    link.name = "S" + std::to_string(shoot->ID) + "_P" + std::to_string(phytomer_idx) + "_Pet" + std::to_string(petiole_idx) + "_Seg" + std::to_string(seg);
                    link.position = midpoint;
                    axisToQuaternion(axis_normalized, link.qw, link.qx, link.qy, link.qz);
                    link.radius = seg_radius;
                    link.half_length = half_len;
                    link.mass = params.wood_density * computeCapsuleVolume(seg_radius, half_len);
                    link.inertia_diagonal = computeCapsuleInertia(link.mass, seg_radius, half_len);

                    // Extract visual mesh from petiole object, transforming to link's local frame
                    if (petiole_idx < phytomer->petiole_objIDs.size() && seg < static_cast<int>(phytomer->petiole_objIDs[petiole_idx].size())) {
                        uint pet_objID = phytomer->petiole_objIDs[petiole_idx][seg];
                        if (context_ptr->doesObjectExist(pet_objID)) {
                            USDLink temp_link;
                            temp_link.position = link.position;
                            buildMeshFromContextObject(context_ptr, pet_objID, data, temp_link);
                            // Re-transform vertices into this link's rotated local frame
                            link.mesh_vertices.resize(temp_link.mesh_vertices.size());
                            for (size_t vi = 0; vi < temp_link.mesh_vertices.size(); vi++) {
                                vec3 world_pos = temp_link.mesh_vertices[vi] + temp_link.position; // back to world
                                link.mesh_vertices[vi] = worldToLocal(world_pos, link.position, link.qw, link.qx, link.qy, link.qz);
                            }
                            link.mesh_uvs = std::move(temp_link.mesh_uvs);
                            link.mesh_face_vertex_counts = std::move(temp_link.mesh_face_vertex_counts);
                            link.mesh_face_vertex_indices = std::move(temp_link.mesh_face_vertex_indices);
                            link.material_index = temp_link.material_index;
                        }
                    }

                    int parent_idx = (prev_petiole_link >= 0) ? prev_petiole_link : internode_attach_link;
                    // Skip if no valid parent link exists (e.g., all internode segments were filtered)
                    if (parent_idx < 0) {
                        continue;
                    }
                    link.parent_link_index = parent_idx;

                    int this_link_idx = static_cast<int>(data.links.size());
                    data.links.push_back(link);

                    USDJoint joint;
                    joint.name = "J_" + data.links[parent_idx].name + "_to_" + link.name;
                    joint.parent_link_index = parent_idx;
                    joint.child_link_index = this_link_idx;
                    joint.local_pos_parent = vec3(0, 0, data.links[parent_idx].half_length);
                    joint.local_pos_child = vec3(0, 0, -half_len);
                    float avg_radius = (data.links[parent_idx].radius + seg_radius) / 2.f;
                    joint.stiffness = computeJointStiffness(params.elastic_modulus, avg_radius, length);
                    float child_Izz = link.inertia_diagonal.x;
                    joint.damping = computeJointDamping(joint.stiffness, child_Izz, params.damping_ratio);
                    joint.is_fixed = false;
                    computeJointLocalRot1(data.links[parent_idx], data.links[this_link_idx], joint);
                    data.joints.push_back(joint);

                    prev_petiole_link = this_link_idx;
                }

                // ---- Leaves on this petiole ---- //

                if (petiole_idx < phytomer->leaf_bases.size()) {
                    for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_bases[petiole_idx].size(); leaf_idx++) {

                        if (petiole_idx >= phytomer->leaf_objIDs.size() || leaf_idx >= phytomer->leaf_objIDs[petiole_idx].size()) {
                            continue;
                        }
                        uint leaf_objID = phytomer->leaf_objIDs[petiole_idx][leaf_idx];
                        if (leaf_objID == 0) {
                            continue;
                        }

                        float leaf_area = phytomer->getLeafArea();
                        uint total_leaves = 0;
                        for (uint pi = 0; pi < phytomer->leaf_bases.size(); pi++) {
                            total_leaves += phytomer->leaf_bases[pi].size();
                        }
                        if (total_leaves > 0) {
                            leaf_area /= static_cast<float>(total_leaves);
                        }

                        float leaf_mass = params.leaf_mass_per_area * leaf_area;
                        if (leaf_mass < 1e-6f) {
                            continue;
                        }

                        USDLink link;
                        link.name = "S" + std::to_string(shoot->ID) + "_P" + std::to_string(phytomer_idx) + "_Pet" + std::to_string(petiole_idx) + "_Leaf" + std::to_string(leaf_idx);
                        link.link_type = USD_LINK_MESH;
                        link.qw = 1; link.qx = 0; link.qy = 0; link.qz = 0;
                        link.radius = 0;
                        link.half_length = 0;
                        link.mass = leaf_mass;

                        link.position = buildMeshFromContextObject(context_ptr, leaf_objID, data, link);
                        link.inertia_diagonal = computeMeshInertia(leaf_mass, link.mesh_vertices);

                        int parent_idx = (prev_petiole_link >= 0) ? prev_petiole_link : internode_attach_link;
                        // Skip if no valid parent link exists (e.g., all internode segments were filtered)
                        if (parent_idx < 0) {
                            continue;
                        }
                        link.parent_link_index = parent_idx;

                        int this_link_idx = static_cast<int>(data.links.size());
                        data.links.push_back(link);

                        USDJoint joint;
                        joint.name = "J_" + data.links[parent_idx].name + "_to_" + link.name;
                        joint.parent_link_index = parent_idx;
                        joint.child_link_index = this_link_idx;
                        joint.local_pos_parent = vec3(0, 0, data.links[parent_idx].half_length);
                        joint.local_pos_child = vec3(0, 0, 0);
                        joint.stiffness = params.organ_spring_stiffness;
                        joint.damping = params.organ_spring_damping;
                        joint.is_fixed = false;
                        computeJointLocalRot1(data.links[parent_idx], data.links[this_link_idx], joint);
                        data.joints.push_back(joint);
                    }
                }
            }

            // ---- Peduncles and inflorescences (only if flowers/fruit are present) ---- //

            for (uint petiole_idx = 0; petiole_idx < phytomer->floral_buds.size(); petiole_idx++) {
                for (uint bud_idx = 0; bud_idx < phytomer->floral_buds[petiole_idx].size(); bud_idx++) {

                    const auto &fbud = phytomer->floral_buds[petiole_idx][bud_idx];

                    // Skip buds that have no flowers or fruit
                    if (fbud.state != BUD_FRUITING && fbud.state != BUD_FLOWER_OPEN && fbud.state != BUD_FLOWER_CLOSED) {
                        continue;
                    }

                    int prev_peduncle_link = -1;
                    if (petiole_idx < phytomer->peduncle_vertices.size() && bud_idx < phytomer->peduncle_vertices[petiole_idx].size()) {

                        const auto &ped_verts = phytomer->peduncle_vertices[petiole_idx][bud_idx];
                        const auto &ped_radii = phytomer->peduncle_radii[petiole_idx][bud_idx];

                        for (int seg = 0; seg < static_cast<int>(ped_verts.size()) - 1; seg++) {
                            vec3 start = ped_verts[seg];
                            vec3 end = ped_verts[seg + 1];
                            vec3 axis = end - start;
                            float length = axis.magnitude();

                            if (length < params.min_segment_length) {
                                continue;
                            }

                            float seg_radius = ped_radii[seg];
                            if (!(seg_radius > 0) || seg_radius > length) {
                                continue;
                            }

                            vec3 midpoint = (start + end) * 0.5f;
                            vec3 axis_normalized = axis / length;
                            float half_len = length / 2.f;

                            USDLink link;
                            link.name = "S" + std::to_string(shoot->ID) + "_P" + std::to_string(phytomer_idx) + "_Ped" + std::to_string(petiole_idx) + "_B" + std::to_string(bud_idx) + "_Seg" + std::to_string(seg);
                            link.position = midpoint;
                            axisToQuaternion(axis_normalized, link.qw, link.qx, link.qy, link.qz);
                            link.radius = seg_radius;
                            link.half_length = half_len;
                            link.mass = params.wood_density * computeCapsuleVolume(seg_radius, half_len);
                            link.inertia_diagonal = computeCapsuleInertia(link.mass, seg_radius, half_len);

                            // Extract visual mesh from peduncle object, transforming to link's local frame
                            if (seg < static_cast<int>(fbud.peduncle_objIDs.size())) {
                                uint ped_objID = fbud.peduncle_objIDs[seg];
                                if (context_ptr->doesObjectExist(ped_objID)) {
                                    USDLink temp_link;
                                    temp_link.position = link.position;
                                    buildMeshFromContextObject(context_ptr, ped_objID, data, temp_link);
                                    link.mesh_vertices.resize(temp_link.mesh_vertices.size());
                                    for (size_t vi = 0; vi < temp_link.mesh_vertices.size(); vi++) {
                                        vec3 world_pos = temp_link.mesh_vertices[vi] + temp_link.position;
                                        link.mesh_vertices[vi] = worldToLocal(world_pos, link.position, link.qw, link.qx, link.qy, link.qz);
                                    }
                                    link.mesh_uvs = std::move(temp_link.mesh_uvs);
                                    link.mesh_face_vertex_counts = std::move(temp_link.mesh_face_vertex_counts);
                                    link.mesh_face_vertex_indices = std::move(temp_link.mesh_face_vertex_indices);
                                    link.material_index = temp_link.material_index;
                                }
                            }

                            int parent_idx = (prev_peduncle_link >= 0) ? prev_peduncle_link : internode_attach_link;
                            // Skip if no valid parent link exists (e.g., all internode segments were filtered)
                            if (parent_idx < 0) {
                                continue;
                            }
                            link.parent_link_index = parent_idx;

                            int this_link_idx = static_cast<int>(data.links.size());
                            data.links.push_back(link);

                            USDJoint joint;
                            joint.name = "J_" + data.links[parent_idx].name + "_to_" + link.name;
                            joint.parent_link_index = parent_idx;
                            joint.child_link_index = this_link_idx;
                            joint.local_pos_parent = vec3(0, 0, data.links[parent_idx].half_length);
                            joint.local_pos_child = vec3(0, 0, -half_len);
                            float avg_radius = (data.links[parent_idx].radius + seg_radius) / 2.f;
                            joint.stiffness = computeJointStiffness(params.elastic_modulus, avg_radius, length);
                            float child_Izz = link.inertia_diagonal.x;
                            joint.damping = computeJointDamping(joint.stiffness, child_Izz, params.damping_ratio);
                            joint.is_fixed = false;
                            computeJointLocalRot1(data.links[parent_idx], data.links[this_link_idx], joint);
                            data.joints.push_back(joint);

                            prev_peduncle_link = this_link_idx;
                        }
                    }

                    int organ_attach = (prev_peduncle_link >= 0) ? prev_peduncle_link : internode_attach_link;

                    for (uint inf_idx = 0; inf_idx < fbud.inflorescence_bases.size(); inf_idx++) {

                        if (inf_idx >= fbud.inflorescence_objIDs.size() || fbud.inflorescence_objIDs[inf_idx] == 0) {
                            continue;
                        }

                        float organ_mass;
                        std::string organ_label;
                        if (fbud.state == BUD_FRUITING) {
                            organ_mass = params.fruit_mass;
                            organ_label = "Fruit";
                        } else if (fbud.state == BUD_FLOWER_OPEN || fbud.state == BUD_FLOWER_CLOSED) {
                            organ_mass = params.flower_mass;
                            organ_label = "Flower";
                        } else {
                            continue;
                        }

                        // Skip if no valid parent link exists (e.g., all internode segments were filtered)
                        if (organ_attach < 0) {
                            continue;
                        }

                        uint organ_objID = fbud.inflorescence_objIDs[inf_idx];

                        USDLink link;
                        link.name = "S" + std::to_string(shoot->ID) + "_P" + std::to_string(phytomer_idx) + "_" + organ_label + std::to_string(inf_idx);
                        link.link_type = USD_LINK_MESH;
                        link.qw = 1; link.qx = 0; link.qy = 0; link.qz = 0;
                        link.radius = 0;
                        link.half_length = 0;
                        link.mass = organ_mass;

                        link.position = buildMeshFromContextObject(context_ptr, organ_objID, data, link);
                        link.inertia_diagonal = computeMeshInertia(organ_mass, link.mesh_vertices);
                        link.parent_link_index = organ_attach;

                        int this_link_idx = static_cast<int>(data.links.size());
                        data.links.push_back(link);

                        USDJoint joint;
                        joint.name = "J_" + data.links[organ_attach].name + "_to_" + link.name;
                        joint.parent_link_index = organ_attach;
                        joint.child_link_index = this_link_idx;
                        joint.local_pos_parent = vec3(0, 0, data.links[organ_attach].half_length);
                        joint.local_pos_child = vec3(0, 0, 0);
                        joint.stiffness = params.organ_spring_stiffness;
                        joint.damping = params.organ_spring_damping;
                        joint.is_fixed = false;
                        computeJointLocalRot1(data.links[organ_attach], data.links[this_link_idx], joint);
                        data.joints.push_back(joint);
                    }
                }
            }
        }
    }

    return data;
}

void writeUSDAHeader(std::ofstream &out, const std::string &default_prim) {
    out << "#usda 1.0\n";
    out << "(\n";
    out << "    defaultPrim = \"" << default_prim << "\"\n";
    out << "    metersPerUnit = 1.0\n";
    out << "    kilogramsPerUnit = 1.0\n";
    out << "    upAxis = \"Z\"\n";
    out << ")\n\n";
}

void writePhysicsScene(std::ofstream &out) {
    out << "def PhysicsScene \"PhysicsScene\"\n";
    out << "{\n";
    out << "    vector3f physics:gravityDirection = (0, 0, -1)\n";
    out << "    float physics:gravityMagnitude = 9.81\n";
    out << "}\n\n";
}

void writePhysicsMaterial(std::ofstream &out, const USDExportParameters &params,
                          const std::string &indent) {
    out << indent << "def Scope \"PhysicsMaterials\"\n";
    out << indent << "{\n";
    out << indent << "    def Material \"WoodMaterial\" (\n";
    out << indent << "        prepend apiSchemas = [\"PhysicsMaterialAPI\"]\n";
    out << indent << "    )\n";
    out << indent << "    {\n";
    out << indent << "        float physics:staticFriction = " << params.static_friction << "\n";
    out << indent << "        float physics:dynamicFriction = " << params.dynamic_friction << "\n";
    out << indent << "        float physics:restitution = " << params.restitution << "\n";
    out << indent << "    }\n";
    out << indent << "}\n\n";
}

// Compute smooth per-vertex normals in faceVarying layout (one entry per face-vertex).
// For each vertex, the normal is the area-weighted average of the normals of all faces
// that share that vertex — this produces smooth shading across curved surfaces like tubes.
// The result buffer has exactly sum(faceVertexCounts) entries, matching the faceVarying
// interpolation contract required by USD and Isaac Sim.
std::vector<vec3> computeFaceVaryingNormals(const std::vector<vec3> &verts,
                                             const std::vector<int> &face_vertex_counts,
                                             const std::vector<int> &face_vertex_indices) {

    // Accumulate area-weighted face normals per vertex
    std::vector<vec3> vertex_normals(verts.size(), make_vec3(0, 0, 0));

    size_t idx_offset = 0;
    for (int count : face_vertex_counts) {
        if (count >= 3 && idx_offset + 2 < face_vertex_indices.size()) {
            vec3 v0 = verts[face_vertex_indices[idx_offset + 0]];
            vec3 v1 = verts[face_vertex_indices[idx_offset + 1]];
            vec3 v2 = verts[face_vertex_indices[idx_offset + 2]];
            // Cross product is proportional to face area — area-weighting comes for free
            vec3 weighted_normal = cross(v1 - v0, v2 - v0);
            for (int k = 0; k < count; k++) {
                int vi = face_vertex_indices[idx_offset + k];
                if (vi >= 0 && vi < static_cast<int>(vertex_normals.size())) {
                    vertex_normals[vi] = vertex_normals[vi] + weighted_normal;
                }
            }
        }
        idx_offset += static_cast<size_t>(count);
    }

    // Normalize accumulated vertex normals
    for (auto &n : vertex_normals) {
        float len = n.magnitude();
        if (len > 1e-10f) {
            n = n / len;
        } else {
            n = make_vec3(0, 0, 1);  // degenerate vertex — emit up-vector
        }
    }

    // Build faceVarying buffer: one entry per face-vertex
    std::vector<vec3> normals;
    normals.reserve(face_vertex_indices.size());
    for (int vi : face_vertex_indices) {
        if (vi >= 0 && vi < static_cast<int>(vertex_normals.size())) {
            normals.push_back(vertex_normals[vi]);
        } else {
            normals.push_back(make_vec3(0, 0, 1));
        }
    }

    return normals;
}

// Build indexed primvar: deduplicate values and return (unique_values, indices).
// Reduces file size and satisfies the IndexedPrimvarChecker requirement.
template <typename T>
std::pair<std::vector<T>, std::vector<int>> buildIndexedPrimvar(const std::vector<T> &values) {
    std::vector<T> unique_vals;
    std::vector<int> indices;
    indices.reserve(values.size());

    // Simple linear search — meshes are small enough that this is fine
    for (const auto &v : values) {
        int found = -1;
        for (int i = 0; i < static_cast<int>(unique_vals.size()); i++) {
            if (unique_vals[i].x == v.x && unique_vals[i].y == v.y && unique_vals[i].z == v.z) {
                found = i; break;
            }
        }
        if (found < 0) {
            found = static_cast<int>(unique_vals.size());
            unique_vals.push_back(v);
        }
        indices.push_back(found);
    }
    return {unique_vals, indices};
}

// Specialisation for vec2 (UVs) — no z component
std::pair<std::vector<vec2>, std::vector<int>> buildIndexedPrimvarVec2(const std::vector<vec2> &values) {
    std::vector<vec2> unique_vals;
    std::vector<int> indices;
    indices.reserve(values.size());
    for (const auto &v : values) {
        int found = -1;
        for (int i = 0; i < static_cast<int>(unique_vals.size()); i++) {
            if (unique_vals[i].x == v.x && unique_vals[i].y == v.y) { found = i; break; }
        }
        if (found < 0) { found = static_cast<int>(unique_vals.size()); unique_vals.push_back(v); }
        indices.push_back(found);
    }
    return {unique_vals, indices};
}

void writeMeshGeometry(std::ofstream &out, const USDLink &link,
                       const std::vector<USDMaterial> &materials, const std::string &inner2,
                       const std::string &plant_prim) {
    // Points
    out << inner2 << "point3f[] points = [\n";
    for (size_t i = 0; i < link.mesh_vertices.size(); i++) {
        out << inner2 << "    (" << link.mesh_vertices[i].x << ", " << link.mesh_vertices[i].y << ", " << link.mesh_vertices[i].z << ")";
        if (i + 1 < link.mesh_vertices.size()) out << ",";
        out << "\n";
    }
    out << inner2 << "]\n";

    // Extent (bounding box) — required by ExtentsChecker
    if (!link.mesh_vertices.empty()) {
        float xmin = link.mesh_vertices[0].x, xmax = xmin;
        float ymin = link.mesh_vertices[0].y, ymax = ymin;
        float zmin = link.mesh_vertices[0].z, zmax = zmin;
        for (const auto &v : link.mesh_vertices) {
            xmin = std::min(xmin, v.x); xmax = std::max(xmax, v.x);
            ymin = std::min(ymin, v.y); ymax = std::max(ymax, v.y);
            zmin = std::min(zmin, v.z); zmax = std::max(zmax, v.z);
        }
        out << inner2 << "float3[] extent = [(" << xmin << ", " << ymin << ", " << zmin
            << "), (" << xmax << ", " << ymax << ", " << zmax << ")]\n";
    }

    out << inner2 << "int[] faceVertexCounts = [";
    for (size_t i = 0; i < link.mesh_face_vertex_counts.size(); i++) {
        if (i > 0) out << ", ";
        out << link.mesh_face_vertex_counts[i];
    }
    out << "]\n";

    out << inner2 << "int[] faceVertexIndices = [";
    for (size_t i = 0; i < link.mesh_face_vertex_indices.size(); i++) {
        if (i > 0) out << ", ";
        out << link.mesh_face_vertex_indices[i];
    }
    out << "]\n";

    // Normals — indexed faceVarying
    std::vector<vec3> normals = computeFaceVaryingNormals(link.mesh_vertices, link.mesh_face_vertex_counts, link.mesh_face_vertex_indices);
    if (!normals.empty()) {
        auto [unique_normals, normal_indices] = buildIndexedPrimvar(normals);
        out << inner2 << "normal3f[] primvars:normals = [";
        for (size_t i = 0; i < unique_normals.size(); i++) {
            if (i > 0) out << ", ";
            out << "(" << unique_normals[i].x << ", " << unique_normals[i].y << ", " << unique_normals[i].z << ")";
        }
        out << "] (\n";
        out << inner2 << "    interpolation = \"faceVarying\"\n";
        out << inner2 << ")\n";
        out << inner2 << "int[] primvars:normals:indices = [";
        for (size_t i = 0; i < normal_indices.size(); i++) {
            if (i > 0) out << ", ";
            out << normal_indices[i];
        }
        out << "]\n";
    }

    // UVs — indexed vertex
    if (!link.mesh_uvs.empty()) {
        auto [unique_uvs, uv_indices] = buildIndexedPrimvarVec2(link.mesh_uvs);
        out << inner2 << "texCoord2f[] primvars:st = [";
        for (size_t i = 0; i < unique_uvs.size(); i++) {
            if (i > 0) out << ", ";
            out << "(" << unique_uvs[i].x << ", " << unique_uvs[i].y << ")";
        }
        out << "] (\n";
        out << inner2 << "    interpolation = \"vertex\"\n";
        out << inner2 << ")\n";
        out << inner2 << "int[] primvars:st:indices = [";
        for (size_t i = 0; i < uv_indices.size(); i++) {
            if (i > 0) out << ", ";
            out << uv_indices[i];
        }
        out << "]\n";
    }

    if (link.material_index >= 0 && link.material_index < static_cast<int>(materials.size())) {
        out << inner2 << "rel material:binding = </" << plant_prim << "/Materials/" << materials[link.material_index].name << ">\n";
    }
}

void writeVisualMaterials(std::ofstream &out, const std::vector<USDMaterial> &materials,
                          const std::string &output_dir, const std::string &plant_prim,
                          const std::string &indent) {
    if (materials.empty()) {
        return;
    }

    std::string inner = indent + "    ";
    std::string inner2 = inner + "    ";
    std::string inner3 = inner2 + "    ";

    out << indent << "def Scope \"Materials\"\n";
    out << indent << "{\n";

    // Copy texture files into a "textures" subdirectory beside the output file
    // so the USDA is self-contained and portable to other machines.
    std::filesystem::path tex_dir = std::filesystem::path(output_dir) / "textures";
    bool textures_copied = false;

    for (const auto &mat : materials) {
        std::string tex_path;
        if (!mat.texture_file.empty()) {
            std::filesystem::path src(mat.texture_file);
            if (std::filesystem::exists(src)) {
                if (!textures_copied) {
                    std::filesystem::create_directories(tex_dir);
                    textures_copied = true;
                }
                std::filesystem::path dst = tex_dir / src.filename();
                // Only copy if source and destination differ
                if (!std::filesystem::exists(dst) || !std::filesystem::equivalent(src, dst)) {
                    std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
                }
                // USD relative path anchored to the file location
                tex_path = "./textures/" + src.filename().string();
            } else {
                // Source file doesn't exist — write the path as-is and hope the user fixes it
                tex_path = mat.texture_file;
            }
        }

        std::string mat_base = "/" + plant_prim + "/Materials/" + mat.name;

        out << inner << "def Material \"" << mat.name << "\"\n";
        out << inner << "{\n";

        out << inner2 << "token outputs:surface.connect = <" << mat_base << "/PreviewSurface.outputs:surface>\n";
        out << "\n";

        out << inner2 << "def Shader \"PreviewSurface\"\n";
        out << inner2 << "{\n";
        out << inner3 << "uniform token info:id = \"UsdPreviewSurface\"\n";
        out << inner3 << "color3f inputs:diffuseColor = (" << mat.color.r << ", " << mat.color.g << ", " << mat.color.b << ")\n";

        if (!tex_path.empty()) {
            out << inner3 << "color3f inputs:diffuseColor.connect = <" << mat_base << "/DiffuseTexture.outputs:rgb>\n";
            out << inner3 << "float inputs:opacity.connect = <" << mat_base << "/DiffuseTexture.outputs:a>\n";
            out << inner3 << "float inputs:opacityThreshold = 0.5\n";
        }

        out << inner3 << "float inputs:roughness = 0.8\n";
        out << inner3 << "float inputs:metallic = 0.0\n";
        out << inner3 << "token outputs:surface\n";
        out << inner2 << "}\n";

        if (!tex_path.empty()) {
            out << "\n";
            out << inner2 << "def Shader \"DiffuseTexture\"\n";
            out << inner2 << "{\n";
            out << inner3 << "uniform token info:id = \"UsdUVTexture\"\n";
            out << inner3 << "asset inputs:file = @" << tex_path << "@\n";
            out << inner3 << "float2 inputs:st.connect = <" << mat_base << "/TexCoordReader.outputs:result>\n";
            out << inner3 << "token inputs:wrapS = \"repeat\"\n";
            out << inner3 << "token inputs:wrapT = \"repeat\"\n";
            out << inner3 << "float3 outputs:rgb\n";
            out << inner3 << "float outputs:a\n";
            out << inner2 << "}\n";
            out << "\n";
            out << inner2 << "def Shader \"TexCoordReader\"\n";
            out << inner2 << "{\n";
            out << inner3 << "uniform token info:id = \"UsdPrimvarReader_float2\"\n";
            out << inner3 << "string inputs:varname = \"st\"\n";
            out << inner3 << "float2 outputs:result\n";
            out << inner2 << "}\n";
        }

        out << inner << "}\n";
    }

    out << indent << "}\n\n";
}

void writeLinkPrim(std::ofstream &out, const USDLink &link, const std::vector<USDMaterial> &materials,
                   const std::string &plant_prim, const std::string &indent) {
    out << indent << "def Xform \"" << link.name << "\" (\n";
    out << indent << "    prepend apiSchemas = [\"PhysicsRigidBodyAPI\", \"PhysicsMassAPI\"]\n";
    out << indent << ")\n";
    out << indent << "{\n";

    std::string inner = indent + "    ";
    std::string inner2 = inner + "    ";

    out << std::fixed << std::setprecision(6);
    out << inner << "point3f xformOp:translate = (" << link.position.x << ", " << link.position.y << ", " << link.position.z << ")\n";
    out << inner << "quatf xformOp:orient = (" << link.qw << ", " << link.qx << ", " << link.qy << ", " << link.qz << ")\n";
    out << inner << "uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\"]\n";

    // Enforce a minimum mass so PhysX doesn't get a zero-mass rigid body
    float written_mass = std::max(link.mass, 1e-6f);
    out << inner << "float physics:mass = " << written_mass << "\n";

    // Only author diagonalInertia + principalAxes when all components are large enough
    // to survive 6-decimal formatting. When omitted, PhysX auto-computes from the collision shape.
    float inertia_floor = 1e-6f;
    bool has_positive_inertia = link.inertia_diagonal.x > inertia_floor
                             && link.inertia_diagonal.y > inertia_floor
                             && link.inertia_diagonal.z > inertia_floor;
    if (has_positive_inertia) {
        out << inner << "float3 physics:diagonalInertia = (" << link.inertia_diagonal.x << ", " << link.inertia_diagonal.y << ", " << link.inertia_diagonal.z << ")\n";
        out << inner << "quatf physics:principalAxes = (1, 0, 0, 0)\n";
    }

    out << "\n";

    std::string phys_mat_path = "</" + plant_prim + "/PhysicsMaterials/WoodMaterial>";

    if (link.link_type == USD_LINK_MESH && !link.mesh_vertices.empty()) {
        // Visual mesh for rendering
        out << inner << "def Mesh \"Visual\" (\n";
        out << inner << "    prepend apiSchemas = [\"MaterialBindingAPI\"]\n";
        out << inner << ")\n";
        out << inner << "{\n";
        out << inner2 << "uniform token subdivisionScheme = \"none\"\n";
        out << inner2 << "bool doubleSided = 1\n";
        writeMeshGeometry(out, link, materials, inner2, plant_prim);
        out << inner << "}\n\n";

        // Collision mesh for physics (convex hull approximation, hidden from rendering)
        out << inner << "def Mesh \"Collision\" (\n";
        out << inner << "    prepend apiSchemas = [\"MaterialBindingAPI\", \"PhysicsCollisionAPI\", \"PhysxCollisionAPI\"]\n";
        out << inner << ")\n";
        out << inner << "{\n";
        out << inner2 << "uniform token subdivisionScheme = \"none\"\n";
        out << inner2 << "uniform token purpose = \"guide\"\n";

        out << inner2 << "point3f[] points = [\n";
        for (size_t i = 0; i < link.mesh_vertices.size(); i++) {
            out << inner2 << "    (" << link.mesh_vertices[i].x << ", " << link.mesh_vertices[i].y << ", " << link.mesh_vertices[i].z << ")";
            if (i + 1 < link.mesh_vertices.size()) out << ",";
            out << "\n";
        }
        out << inner2 << "]\n";

        // Extent for collision mesh
        {
            float xmin = link.mesh_vertices[0].x, xmax = xmin;
            float ymin = link.mesh_vertices[0].y, ymax = ymin;
            float zmin = link.mesh_vertices[0].z, zmax = zmin;
            for (const auto &v : link.mesh_vertices) {
                xmin = std::min(xmin, v.x); xmax = std::max(xmax, v.x);
                ymin = std::min(ymin, v.y); ymax = std::max(ymax, v.y);
                zmin = std::min(zmin, v.z); zmax = std::max(zmax, v.z);
            }
            out << inner2 << "float3[] extent = [(" << xmin << ", " << ymin << ", " << zmin
                << "), (" << xmax << ", " << ymax << ", " << zmax << ")]\n";
        }

        out << inner2 << "int[] faceVertexCounts = [";
        for (size_t i = 0; i < link.mesh_face_vertex_counts.size(); i++) {
            if (i > 0) out << ", ";
            out << link.mesh_face_vertex_counts[i];
        }
        out << "]\n";

        out << inner2 << "int[] faceVertexIndices = [";
        for (size_t i = 0; i < link.mesh_face_vertex_indices.size(); i++) {
            if (i > 0) out << ", ";
            out << link.mesh_face_vertex_indices[i];
        }
        out << "]\n";

        out << inner2 << "uniform token physics:approximation = \"convexHull\"\n";
        out << inner2 << "rel material:binding:physics = " << phys_mat_path << "\n";
        out << inner << "}\n";
    } else if (link.link_type == USD_LINK_CAPSULE) {
        // Visual mesh (if available from tube geometry)
        if (!link.mesh_vertices.empty()) {
            out << inner << "def Mesh \"Visual\" (\n";
            out << inner << "    prepend apiSchemas = [\"MaterialBindingAPI\"]\n";
            out << inner << ")\n";
            out << inner << "{\n";
            out << inner2 << "uniform token subdivisionScheme = \"none\"\n";
            out << inner2 << "bool doubleSided = 1\n";
            writeMeshGeometry(out, link, materials, inner2, plant_prim);
            out << inner << "}\n\n";
        }

        // Collision capsule (hidden from rendering)
        out << inner << "def Capsule \"Collision\" (\n";
        out << inner << "    prepend apiSchemas = [\"MaterialBindingAPI\", \"PhysicsCollisionAPI\"]\n";
        out << inner << ")\n";
        out << inner << "{\n";
        out << inner << "    uniform token purpose = \"guide\"\n";
        out << inner << "    double radius = " << link.radius << "\n";
        out << inner << "    double height = " << (2.f * link.half_length) << "\n";
        out << inner << "    uniform token axis = \"Z\"\n";
        out << inner << "    rel material:binding:physics = " << phys_mat_path << "\n";
        out << inner << "}\n";
    }

    out << indent << "}\n";
}

void writeFixedJoint(std::ofstream &out, const USDJoint &joint, const std::vector<USDLink> &links, const std::string &plant_prim, const std::string &indent) {
    out << indent << "def PhysicsFixedJoint \"" << joint.name << "\"\n";
    out << indent << "{\n";

    std::string inner = indent + "    ";
    out << std::fixed << std::setprecision(6);
    out << inner << "rel physics:body1 = </" << plant_prim << "/Links/" << links[joint.child_link_index].name << ">\n";
    out << inner << "point3f physics:localPos0 = (" << joint.local_pos_parent.x << ", " << joint.local_pos_parent.y << ", " << joint.local_pos_parent.z << ")\n";
    out << inner << "point3f physics:localPos1 = (" << joint.local_pos_child.x << ", " << joint.local_pos_child.y << ", " << joint.local_pos_child.z << ")\n";
    out << inner << "quatf physics:localRot0 = (1, 0, 0, 0)\n";
    out << inner << "quatf physics:localRot1 = (" << joint.local_rot1_w << ", " << joint.local_rot1_x << ", " << joint.local_rot1_y << ", " << joint.local_rot1_z << ")\n";
    out << inner << "bool physics:collisionEnabled = false\n";
    out << indent << "}\n";
}

void writeSphericalJoint(std::ofstream &out, const USDJoint &joint, const std::vector<USDLink> &links, const std::string &plant_prim, const std::string &indent) {
    out << indent << "def PhysicsSphericalJoint \"" << joint.name << "\" (\n";
    out << indent << "    prepend apiSchemas = [\"PhysicsLimitAPI:cone\", \"PhysicsDriveAPI:angular\"]\n";
    out << indent << ")\n";
    out << indent << "{\n";

    std::string inner = indent + "    ";
    out << std::fixed << std::setprecision(6);
    out << inner << "rel physics:body0 = </" << plant_prim << "/Links/" << links[joint.parent_link_index].name << ">\n";
    out << inner << "rel physics:body1 = </" << plant_prim << "/Links/" << links[joint.child_link_index].name << ">\n";
    out << inner << "point3f physics:localPos0 = (" << joint.local_pos_parent.x << ", " << joint.local_pos_parent.y << ", " << joint.local_pos_parent.z << ")\n";
    out << inner << "point3f physics:localPos1 = (" << joint.local_pos_child.x << ", " << joint.local_pos_child.y << ", " << joint.local_pos_child.z << ")\n";
    out << inner << "quatf physics:localRot0 = (1, 0, 0, 0)\n";
    out << inner << "quatf physics:localRot1 = (" << joint.local_rot1_w << ", " << joint.local_rot1_x << ", " << joint.local_rot1_y << ", " << joint.local_rot1_z << ")\n";
    out << inner << "bool physics:collisionEnabled = false\n";

    out << "\n";

    // Cone limit: constrain swing range
    out << inner << "float limit:cone:physics:yAngle = 60\n";
    out << inner << "float limit:cone:physics:zAngle = 60\n";

    out << "\n";

    // Angular drive: spring-damper to restore rest pose
    out << inner << "uniform token drive:angular:physics:type = \"force\"\n";
    out << inner << "float drive:angular:physics:stiffness = " << joint.stiffness << "\n";
    out << inner << "float drive:angular:physics:damping = " << joint.damping << "\n";
    out << inner << "float drive:angular:physics:targetPosition = 0\n";

    out << indent << "}\n";
}

} // anonymous namespace

void PlantArchitecture::writePlantStructureUSD(uint plantID, const std::string &filename, const USDExportParameters &params) const {

    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureUSD): Plant ID " + std::to_string(plantID) + " does not exist.");
    }

    const auto &plant = plant_instances.at(plantID);

    if (plant.shoot_tree.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureUSD): Plant has no shoots to export.");
    }

    std::string output_file = filename;
    if (!validateOutputPath(output_file, {".usda", ".USDA"})) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureUSD): Could not open file " + filename + " for writing. Make sure the directory exists and is writable.");
    } else if (getFileName(output_file).empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureUSD): The output file given was a directory. This argument should be the path to a file.");
    }

    std::ofstream out(output_file);
    if (!out.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureUSD): Could not open file " + output_file + " for writing.");
    }

    ArticulationData artic = buildArticulationData(plant, params, context_ptr);

    if (artic.links.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureUSD): No valid segments found in plant. All segments may be shorter than min_segment_length.");
    }

    std::string plant_prim = sanitizePrimName("Plant_" + std::to_string(plantID));

    writeUSDAHeader(out, plant_prim);
    writePhysicsScene(out);

    out << "def Xform \"" << plant_prim << "\" (\n";
    out << "    prepend apiSchemas = [\"PhysicsArticulationRootAPI\", \"PhysxArticulationAPI\"]\n";
    out << ")\n";
    out << "{\n";
    out << "    bool physxArticulation:enabledSelfCollisions = false\n";
    out << "    int physxArticulation:solverPositionIterationCount = " << params.solver_position_iterations << "\n";
    out << "\n";

    // Nest PhysicsMaterials and Materials inside the default prim so referencing it is self-contained
    writePhysicsMaterial(out, params, "    ");
    writeVisualMaterials(out, artic.materials, getFilePath(output_file), plant_prim, "    ");

    out << "    def Xform \"Links\"\n";
    out << "    {\n";
    for (const auto &link : artic.links) {
        writeLinkPrim(out, link, artic.materials, plant_prim, "        ");
        out << "\n";
    }
    out << "    }\n";
    out << "\n";

    out << "    def Xform \"Joints\"\n";
    out << "    {\n";
    for (const auto &joint : artic.joints) {
        if (joint.is_fixed) {
            writeFixedJoint(out, joint, artic.links, plant_prim, "        ");
        } else {
            writeSphericalJoint(out, joint, artic.links, plant_prim, "        ");
        }
        out << "\n";
    }
    out << "    }\n";

    out << "}\n";

    out.close();

    if (printmessages) {
        std::cout << "Wrote USD articulated plant to " << output_file << " (" << artic.links.size() << " links, " << artic.joints.size() << " joints)" << std::endl;
    }
}

// ============================================================================
// Growth Animation USD Export
// ============================================================================

void PlantArchitecture::registerGrowthFrame(uint plantID, float min_segment_length) {

    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::registerGrowthFrame): Plant ID " + std::to_string(plantID) + " does not exist.");
    }

    const auto &plant = plant_instances.at(plantID);

    if (plant.shoot_tree.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::registerGrowthFrame): Plant has no shoots to capture.");
    }

    USDExportParameters params;
    params.min_segment_length = min_segment_length;

    auto frame = std::make_shared<GrowthFrameSnapshot>();
    frame->time = plant.current_age;
    frame->artic_data = buildArticulationData(plant, params, context_ptr);

    growth_animation_storage.plant_frames[plantID].push_back(std::move(frame));

    if (printmessages) {
        std::cout << "Registered growth frame " << growth_animation_storage.plant_frames[plantID].size()
                  << " for plant " << plantID << " (age=" << plant.current_age << " days, "
                  << growth_animation_storage.plant_frames[plantID].back()->artic_data.links.size() << " links)" << std::endl;
    }
}

void PlantArchitecture::clearGrowthFrames(uint plantID) {
    growth_animation_storage.plant_frames.erase(plantID);
}

uint PlantArchitecture::getGrowthFrameCount(uint plantID) const {
    auto it = growth_animation_storage.plant_frames.find(plantID);
    if (it == growth_animation_storage.plant_frames.end()) {
        return 0;
    }
    return it->second.size();
}

void PlantArchitecture::writePlantGrowthUSD(uint plantID, const std::string &filename, float seconds_per_frame) const {

    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantGrowthUSD): Plant ID " + std::to_string(plantID) + " does not exist.");
    }

    auto frames_it = growth_animation_storage.plant_frames.find(plantID);
    if (frames_it == growth_animation_storage.plant_frames.end() || frames_it->second.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantGrowthUSD): No growth frames registered for plant " + std::to_string(plantID) + ". Call registerGrowthFrame() first.");
    }

    const auto &frames = frames_it->second;
    uint num_frames = frames.size();

    std::string output_file = filename;
    if (!validateOutputPath(output_file, {".usda", ".USDA"})) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantGrowthUSD): Could not open file " + filename + " for writing. Make sure the directory exists and is writable.");
    } else if (getFileName(output_file).empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantGrowthUSD): The output file given was a directory. This argument should be the path to a file.");
    }

    // --- Build superset topology ---
    // Collect all unique link names across all frames, and for each link keep the data from the last frame it appears in (canonical data).
    std::vector<std::string> link_order;  // ordered list of unique link names
    std::map<std::string, USDLink> canonical_links;  // link name -> canonical mesh/material data
    std::map<std::string, int> canonical_material_index;  // link name -> material index in merged material list

    // Merged material list across all frames
    std::vector<USDMaterial> all_materials;
    std::map<std::string, int> material_merge_cache;  // key -> index in all_materials

    // For each frame, track which links are present and their transforms
    // link_presence[link_name][frame_idx] = true if link exists at that frame
    std::map<std::string, std::vector<bool>> link_presence;
    // link_transforms[link_name][frame_idx] = (position, quaternion)
    struct LinkTransform {
        vec3 position;
        float qw, qx, qy, qz;
    };
    std::map<std::string, std::vector<LinkTransform>> link_transforms;

    for (uint f = 0; f < num_frames; f++) {
        const auto &artic = frames[f]->artic_data;

        // Merge materials from this frame
        std::map<int, int> frame_to_merged_mat;  // frame-local material index -> merged index
        for (int m = 0; m < (int)artic.materials.size(); m++) {
            const auto &mat = artic.materials[m];
            std::string key = mat.texture_file + "|" + std::to_string(mat.color.r) + "," + std::to_string(mat.color.g) + "," + std::to_string(mat.color.b);
            auto cache_it = material_merge_cache.find(key);
            if (cache_it != material_merge_cache.end()) {
                frame_to_merged_mat[m] = cache_it->second;
            } else {
                int idx = (int)all_materials.size();
                all_materials.push_back(mat);
                // Ensure unique material name
                all_materials.back().name = sanitizePrimName("Mat_" + std::to_string(idx));
                material_merge_cache[key] = idx;
                frame_to_merged_mat[m] = idx;
            }
        }

        for (const auto &link : artic.links) {
            // Initialize presence/transform vectors if first time seeing this link
            if (link_presence.find(link.name) == link_presence.end()) {
                link_order.push_back(link.name);
                link_presence[link.name].resize(num_frames, false);
                link_transforms[link.name].resize(num_frames);
            }

            link_presence[link.name][f] = true;
            link_transforms[link.name][f] = {link.position, link.qw, link.qx, link.qy, link.qz};

            // Update canonical data (last frame where link appears has most developed geometry)
            canonical_links[link.name] = link;
            if (link.material_index >= 0) {
                canonical_material_index[link.name] = frame_to_merged_mat[link.material_index];
            } else {
                canonical_material_index[link.name] = -1;
            }
        }
    }

    // --- Write USDA file ---
    std::ofstream out(output_file);
    if (!out.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantGrowthUSD): Could not open file " + output_file + " for writing.");
    }

    out << std::fixed;

    std::string plant_prim = sanitizePrimName("Plant_" + std::to_string(plantID));

    // Blender maps USD time codes 1:1 to Blender frames and uses framesPerSecond for playback.
    // To make each growth frame last `seconds_per_frame` seconds, we space time codes by
    // fps * seconds_per_frame. E.g., 6 frames at 1 sec/frame with 24fps ->
    // time codes 0, 24, 48, 72, 96, 120 -> 5 seconds of animation at 24fps playback.
    float fps = 24.0f;
    float time_code_spacing = fps * seconds_per_frame;
    std::vector<float> time_codes(num_frames);
    for (uint f = 0; f < num_frames; f++) {
        time_codes[f] = f * time_code_spacing;
    }

    // Header
    out << "#usda 1.0\n";
    out << "(\n";
    out << "    defaultPrim = \"" << plant_prim << "\"\n";
    out << "    metersPerUnit = 1.0\n";
    out << "    upAxis = \"Z\"\n";
    out << "    startTimeCode = 0\n";
    out << "    endTimeCode = " << time_codes.back() << "\n";
    out << "    framesPerSecond = " << fps << "\n";
    out << "    timeCodesPerSecond = " << fps << "\n";
    out << ")\n\n";

    // Plant root Xform (Materials nested inside so referencing the default prim is self-contained)
    out << "def Xform \"" << plant_prim << "\"\n";
    out << "{\n";

    if (!all_materials.empty()) {
        writeVisualMaterials(out, all_materials, getFilePath(output_file), plant_prim, "    ");
    }

    // Write each link as a time-sampled Xform
    for (const auto &link_name : link_order) {
        const auto &presence = link_presence[link_name];
        const auto &transforms = link_transforms[link_name];
        const auto &canonical = canonical_links[link_name];
        int mat_idx = canonical_material_index[link_name];

        // Find the first frame where this link appears — used as the default position
        // for frames before the link exists, so geometry doesn't cluster at the origin.
        LinkTransform first_known = {canonical.position, canonical.qw, canonical.qx, canonical.qy, canonical.qz};
        for (uint f = 0; f < num_frames; f++) {
            if (presence[f]) {
                first_known = transforms[f];
                break;
            }
        }

        std::string prim_name = sanitizePrimName(link_name);

        out << "    def Xform \"" << prim_name << "\"\n";
        out << "    {\n";

        // For each transform property, we use "step" keyframing: before each growth frame's
        // time code, we insert a "hold" keyframe 1 Blender frame earlier with the previous
        // frame's value. This prevents Blender from smoothly interpolating between growth
        // frames — all transitions are instantaneous.

        // Time-sampled translate
        out << "        point3f xformOp:translate.timeSamples = {\n";
        {
            bool first_entry = true;
            for (uint f = 0; f < num_frames; f++) {
                const auto &t = presence[f] ? transforms[f] : first_known;

                // Hold keyframe: 1 Blender frame before this time code, hold the previous value
                if (f > 0 && time_codes[f] > 0) {
                    const auto &prev_t = presence[f - 1] ? transforms[f - 1] : first_known;
                    if (!first_entry) out << ",\n";
                    out << "            " << (time_codes[f] - 1.0f) << ": (" << prev_t.position.x << ", " << prev_t.position.y << ", " << prev_t.position.z << ")";
                    first_entry = false;
                }

                if (!first_entry) out << ",\n";
                out << "            " << time_codes[f] << ": (" << t.position.x << ", " << t.position.y << ", " << t.position.z << ")";
                first_entry = false;
            }
            out << "\n        }\n";
        }

        // Time-sampled orient
        out << "        quatf xformOp:orient.timeSamples = {\n";
        {
            bool first_entry = true;
            for (uint f = 0; f < num_frames; f++) {
                const auto &t = presence[f] ? transforms[f] : first_known;

                if (f > 0 && time_codes[f] > 0) {
                    const auto &prev_t = presence[f - 1] ? transforms[f - 1] : first_known;
                    if (!first_entry) out << ",\n";
                    out << "            " << (time_codes[f] - 1.0f) << ": (" << prev_t.qw << ", " << prev_t.qx << ", " << prev_t.qy << ", " << prev_t.qz << ")";
                    first_entry = false;
                }

                if (!first_entry) out << ",\n";
                out << "            " << time_codes[f] << ": (" << t.qw << ", " << t.qx << ", " << t.qy << ", " << t.qz << ")";
                first_entry = false;
            }
            out << "\n        }\n";
        }

        // Time-sampled scale: (0,0,0) hides organs before they appear, (1,1,1) shows them.
        out << "        float3 xformOp:scale.timeSamples = {\n";
        {
            bool first_entry = true;
            for (uint f = 0; f < num_frames; f++) {
                bool is_visible = presence[f];
                bool was_visible = (f > 0) && presence[f - 1];

                // Hold previous scale 1 frame before transition
                if (f > 0 && time_codes[f] > 0) {
                    if (!first_entry) out << ",\n";
                    out << "            " << (time_codes[f] - 1.0f) << ": " << (was_visible ? "(1, 1, 1)" : "(0, 0, 0)");
                    first_entry = false;
                }

                if (!first_entry) out << ",\n";
                out << "            " << time_codes[f] << ": " << (is_visible ? "(1, 1, 1)" : "(0, 0, 0)");
                first_entry = false;
            }
            out << "\n        }\n";
        }

        out << "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\", \"xformOp:scale\"]\n";

        // Time-sampled visibility (kept for USD-compliant readers; scale-to-zero handles Blender)
        out << "        token visibility.timeSamples = {\n";
        for (uint f = 0; f < num_frames; f++) {
            out << "            " << time_codes[f] << ": \"" << (presence[f] ? "inherited" : "invisible") << "\"";
            out << (f < num_frames - 1 ? ",\n" : "\n");
        }
        out << "        }\n";

        // Static mesh from canonical frame
        bool has_mesh = false;
        if (canonical.link_type == USD_LINK_MESH && !canonical.mesh_vertices.empty()) {
            has_mesh = true;
        } else if (canonical.link_type == USD_LINK_CAPSULE && !canonical.mesh_vertices.empty()) {
            has_mesh = true;
        }

        // Helper: write visibility time samples on the child visual prim too, since Blender's
        // USD importer may not respect visibility inheritance from parent Xform.
        auto writeChildVisibility = [&]() {
            out << "            token visibility.timeSamples = {\n";
            for (uint f2 = 0; f2 < num_frames; f2++) {
                out << "                " << time_codes[f2] << ": \"" << (presence[f2] ? "inherited" : "invisible") << "\"";
                out << (f2 < num_frames - 1 ? ",\n" : "\n");
            }
            out << "            }\n";
        };

        if (has_mesh) {
            out << "\n";
            out << "        def Mesh \"Visual\" (\n";
            out << "            prepend apiSchemas = [\"MaterialBindingAPI\"]\n";
            out << "        )\n";
            out << "        {\n";

            writeChildVisibility();
            out << "            uniform token subdivisionScheme = \"none\"\n";
            out << "            bool doubleSided = 1\n";

            // Vertices
            out << "            point3f[] points = [";
            for (size_t v = 0; v < canonical.mesh_vertices.size(); v++) {
                if (v > 0) out << ", ";
                out << "(" << canonical.mesh_vertices[v].x << ", " << canonical.mesh_vertices[v].y << ", " << canonical.mesh_vertices[v].z << ")";
            }
            out << "]\n";

            // Extent
            if (!canonical.mesh_vertices.empty()) {
                float xmn = canonical.mesh_vertices[0].x, xmx = xmn;
                float ymn = canonical.mesh_vertices[0].y, ymx = ymn;
                float zmn = canonical.mesh_vertices[0].z, zmx = zmn;
                for (const auto &v : canonical.mesh_vertices) {
                    xmn = std::min(xmn, v.x); xmx = std::max(xmx, v.x);
                    ymn = std::min(ymn, v.y); ymx = std::max(ymx, v.y);
                    zmn = std::min(zmn, v.z); zmx = std::max(zmx, v.z);
                }
                out << "            float3[] extent = [(" << xmn << ", " << ymn << ", " << zmn
                    << "), (" << xmx << ", " << ymx << ", " << zmx << ")]\n";
            }

            // Face vertex counts
            out << "            int[] faceVertexCounts = [";
            for (size_t i = 0; i < canonical.mesh_face_vertex_counts.size(); i++) {
                if (i > 0) out << ", ";
                out << canonical.mesh_face_vertex_counts[i];
            }
            out << "]\n";

            // Face vertex indices
            out << "            int[] faceVertexIndices = [";
            for (size_t i = 0; i < canonical.mesh_face_vertex_indices.size(); i++) {
                if (i > 0) out << ", ";
                out << canonical.mesh_face_vertex_indices[i];
            }
            out << "]\n";

            // Normals — indexed faceVarying
            {
                std::vector<vec3> normals = computeFaceVaryingNormals(canonical.mesh_vertices, canonical.mesh_face_vertex_counts, canonical.mesh_face_vertex_indices);
                if (!normals.empty()) {
                    auto [unique_n, n_idx] = buildIndexedPrimvar(normals);
                    out << "            normal3f[] primvars:normals = [";
                    for (size_t i = 0; i < unique_n.size(); i++) {
                        if (i > 0) out << ", ";
                        out << "(" << unique_n[i].x << ", " << unique_n[i].y << ", " << unique_n[i].z << ")";
                    }
                    out << "] (\n                interpolation = \"faceVarying\"\n            )\n";
                    out << "            int[] primvars:normals:indices = [";
                    for (size_t i = 0; i < n_idx.size(); i++) { if (i > 0) out << ", "; out << n_idx[i]; }
                    out << "]\n";
                }
            }

            // UVs — indexed vertex
            if (!canonical.mesh_uvs.empty()) {
                auto [unique_uv, uv_idx] = buildIndexedPrimvarVec2(canonical.mesh_uvs);
                out << "            texCoord2f[] primvars:st = [";
                for (size_t i = 0; i < unique_uv.size(); i++) {
                    if (i > 0) out << ", ";
                    out << "(" << unique_uv[i].x << ", " << unique_uv[i].y << ")";
                }
                out << "] (\n                interpolation = \"vertex\"\n            )\n";
                out << "            int[] primvars:st:indices = [";
                for (size_t i = 0; i < uv_idx.size(); i++) { if (i > 0) out << ", "; out << uv_idx[i]; }
                out << "]\n";
            }

            // Material binding — path is now nested under the default prim
            if (mat_idx >= 0 && mat_idx < (int)all_materials.size()) {
                out << "            rel material:binding = </" << plant_prim << "/Materials/" << all_materials[mat_idx].name << ">\n";
            }

            out << "        }\n";
        } else if (canonical.link_type == USD_LINK_CAPSULE) {
            // Write a capsule shape for visual representation
            out << "\n";
            out << "        def Capsule \"Visual\"\n";
            out << "        {\n";
            writeChildVisibility();
            out << "            double radius = " << canonical.radius << "\n";
            out << "            double height = " << (2.0 * canonical.half_length) << "\n";
            out << "            uniform token axis = \"Z\"\n";
            if (mat_idx >= 0 && mat_idx < (int)all_materials.size()) {
                out << "            rel material:binding = </Materials/" << all_materials[mat_idx].name << ">\n";
            }
            out << "        }\n";
        }

        out << "    }\n\n";
    }

    out << "}\n";
    out.close();

    if (printmessages) {
        std::cout << "Wrote USD growth animation to " << output_file << " (" << link_order.size() << " prims, " << num_frames << " frames)" << std::endl;
    }
}
