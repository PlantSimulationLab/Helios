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
                            // Base shoot: use plant base (origin)
                            internode_base = make_vec3(0, 0, 0);
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
                                        }
                                    }

                                    // Restore flower geometry using saved rotations (base positions auto-computed)
                                    if (!fbud_data.flower_rotations.empty()) {

                                        // Ensure prototype maps are initialized before creating geometry
                                        ensureInflorescencePrototypesInitialized(phytomer_ptr->phytomer_parameters);

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
                                        }
                                    }

                                    // Restore fruit geometry using saved rotations (base positions auto-computed)
                                    if (!fbud_data.flower_rotations.empty()) {

                                        // Ensure prototype maps are initialized before creating geometry
                                        ensureInflorescencePrototypesInitialized(phytomer_ptr->phytomer_parameters);

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
