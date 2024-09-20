/** \file "InputOutput.cpp" Routines for reading and writing plant geometry in the plant architecture plug-in.

Copyright (C) 2016-2024 Brian Bailey

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

std::string PlantArchitecture::makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> & shoot_tree) const{

    std::string outstring = current_string;

    if(shoot->parent_shoot_ID != -1 ) {
        outstring += "[";
    }

    outstring += "{" + std::to_string(rad2deg(shoot->base_rotation.pitch)) + "," + std::to_string(rad2deg(shoot->base_rotation.yaw)) + "," + std::to_string(rad2deg(shoot->base_rotation.roll)) + "," + std::to_string(shoot->shoot_parameters.gravitropic_curvature.val() /*\todo */) + "," + shoot->shoot_type_label + "}";

    uint node_number = 0;
    for( auto &phytomer: shoot->phytomers ){

        float length = phytomer->getInternodeLength();
        float radius = phytomer->getInternodeRadius();

        outstring += "Internode(" + std::to_string(length) + "," + std::to_string(radius) + "," + std::to_string( rad2deg(phytomer->internode_pitch) ) + "," + std::to_string( rad2deg(phytomer->internode_phyllotactic_angle) ) + ")";

//        for( uint petiole=0; petiole<phytomer->petiole_length.size(); petiole++ ){
        uint petiole = 0;

            outstring += "Petiole(" + std::to_string( phytomer->petiole_length.at(petiole) ) + "," + std::to_string( phytomer->petiole_radii.at(petiole).front() ) + "," + std::to_string( rad2deg(phytomer->petiole_pitch) ) + ")";

            //\todo If leaf is compound, just using rotation for the first leaf for now rather than adding multiple 'Leaf()' strings for each leaflet.
            outstring += "Leaf(" + std::to_string(phytomer->leaf_size_max.at(petiole).front()*phytomer->current_leaf_scale_factor ) + "," + std::to_string( rad2deg(phytomer->leaf_rotation.at(petiole).front().pitch) ) + "," + std::to_string( rad2deg(phytomer->leaf_rotation.at(petiole).front().yaw) ) + "," + std::to_string( rad2deg(phytomer->leaf_rotation.at(petiole).front().roll) ) + ")";

            if( shoot->childIDs.find(node_number)!=shoot->childIDs.end() ){
                outstring = makeShootString(outstring, shoot_tree.at(shoot->childIDs.at(node_number)), shoot_tree );
            }

//        }

        node_number++;
    }

    if(shoot->parent_shoot_ID != -1 ) {
        outstring += "]";
    }

    return outstring;

}

std::string PlantArchitecture::getPlantString(uint plantID) const{

    auto plant_shoot_tree = &plant_instances.at(plantID).shoot_tree;

    std::string out_string;

    for( auto &shoot: *plant_shoot_tree ){
        out_string = makeShootString(out_string, shoot, *plant_shoot_tree );
    }

    return out_string;

}

void PlantArchitecture::parseShootArgument(const std::string &shoot_argument, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters, AxisRotation &base_rotation, std::string &phytomer_label) {

    //shoot argument order {}:
    // 1. shoot base pitch/insertion angle (degrees)
    // 2. shoot base yaw angle (degrees)
    // 3. shoot roll angle (degrees)
    // 4. gravitropic curvature (degrees/meter)
    // 5. [optional] phytomer parameters string

    size_t pos_shoot_start = 0;

    std::string s_argument = shoot_argument;

    pos_shoot_start = s_argument.find(',');
    if( pos_shoot_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float insertion_angle = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    base_rotation.pitch = deg2rad(insertion_angle);

    pos_shoot_start = s_argument.find(',');
    if( pos_shoot_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float shoot_yaw = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    base_rotation.yaw = deg2rad(shoot_yaw);

    pos_shoot_start = s_argument.find(',');
    if( pos_shoot_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float shoot_roll = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    base_rotation.roll = deg2rad(shoot_roll);

    pos_shoot_start = s_argument.find(',');
    if( pos_shoot_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Shoot brackets '{}' does not have the correct number of values given.");
    }
    float shoot_curvature = std::stof(s_argument.substr(0, pos_shoot_start));
    s_argument.erase(0, pos_shoot_start + 1);
    shoot_parameters.gravitropic_curvature = shoot_curvature;

    if( pos_shoot_start != std::string::npos ) { //shoot type argument was given
        pos_shoot_start = s_argument.find(',');
        phytomer_label = s_argument.substr(0, pos_shoot_start);
        s_argument.erase(0, pos_shoot_start + 1);
        if (phytomer_parameters.find(phytomer_label) == phytomer_parameters.end()) {
            helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Phytomer parameters with label " + phytomer_label + " was not provided to PlantArchitecture::generatePlantFromString().");
        }
        shoot_parameters.phytomer_parameters = phytomer_parameters.at(phytomer_label);
    }else{ //shoot type argument not given - use first phytomer parameters in the map
        phytomer_label = phytomer_parameters.begin()->first;
        shoot_parameters.phytomer_parameters = phytomer_parameters.begin()->second;
    }

}

void PlantArchitecture::parseInternodeArgument(const std::string &internode_argument, float &internode_radius, float &internode_length, PhytomerParameters &phytomer_parameters) {

    //internode argument order Internode():
    // 1. internode length (m)
    // 2. internode radius (m)
    // 3. internode pitch (degrees)
    // 4. phyllotactic angle (degrees)

    size_t pos_inode_start = 0;

    std::string inode_argument = internode_argument;

    pos_inode_start = inode_argument.find(',');
    if( pos_inode_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    internode_length = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);

    pos_inode_start = inode_argument.find(',');
    if( pos_inode_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    internode_radius = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);

    pos_inode_start = inode_argument.find(',');
    if( pos_inode_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    float internode_pitch = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);
    phytomer_parameters.internode.pitch = internode_pitch;

    pos_inode_start = inode_argument.find(',');
    if( pos_inode_start != std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    float internode_phyllotaxis = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);
    phytomer_parameters.internode.phyllotactic_angle = internode_phyllotaxis;

}

void PlantArchitecture::parsePetioleArgument(const std::string& petiole_argument, PhytomerParameters &phytomer_parameters ){

    //petiole argument order Petiole():
    // 1. petiole length (m)
    // 2. petiole radius (m)
    // 3. petiole pitch (degrees)

    if( petiole_argument.empty() ){
        phytomer_parameters.petiole.length = 0;
        return;
    }

    size_t pos_petiole_start = 0;

    std::string pet_argument = petiole_argument;

    pos_petiole_start = pet_argument.find(',');
    if( pos_petiole_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Petiole()' does not have the correct number of values given.");
    }
    float petiole_length = std::stof(pet_argument.substr(0, pos_petiole_start));
    pet_argument.erase(0, pos_petiole_start + 1);
    phytomer_parameters.petiole.length = petiole_length;

    pos_petiole_start = pet_argument.find(',');
    if( pos_petiole_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Petiole()' does not have the correct number of values given.");
    }
    float petiole_radius = std::stof(pet_argument.substr(0, pos_petiole_start));
    pet_argument.erase(0, pos_petiole_start + 1);
    phytomer_parameters.petiole.radius = petiole_radius;

    pos_petiole_start = pet_argument.find(',');
    if( pos_petiole_start != std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Petiole()' does not have the correct number of values given.");
    }
    float petiole_pitch = std::stof(pet_argument.substr(0, pos_petiole_start));
    pet_argument.erase(0, pos_petiole_start + 1);
    phytomer_parameters.petiole.pitch = petiole_pitch;

}

void PlantArchitecture::parseLeafArgument(const std::string& leaf_argument, PhytomerParameters &phytomer_parameters ){

    //leaf argument order Leaf():
    // 1. leaf scale factor
    // 2. leaf pitch (degrees)
    // 3. leaf yaw (degrees)
    // 4. leaf roll (degrees)

    if( leaf_argument.empty() ){
        phytomer_parameters.leaf.prototype_scale = 0;
        return;
    }

    size_t pos_leaf_start = 0;

    std::string l_argument = leaf_argument;

    pos_leaf_start = l_argument.find(',');
    if( pos_leaf_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Leaf()' does not have the correct number of values given.");
    }
    float leaf_scale = std::stof(l_argument.substr(0, pos_leaf_start));
    l_argument.erase(0, pos_leaf_start + 1);
    phytomer_parameters.leaf.prototype_scale = leaf_scale;

    pos_leaf_start = l_argument.find(',');
    if( pos_leaf_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Leaf()' does not have the correct number of values given.");
    }
    float leaf_pitch = std::stof(l_argument.substr(0, pos_leaf_start));
    l_argument.erase(0, pos_leaf_start + 1);
    phytomer_parameters.leaf.pitch = leaf_pitch;

    pos_leaf_start = l_argument.find(',');
    if( pos_leaf_start == std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Leaf()' does not have the correct number of values given.");
    }
    float leaf_yaw = std::stof(l_argument.substr(0, pos_leaf_start));
    l_argument.erase(0, pos_leaf_start + 1);
    phytomer_parameters.leaf.yaw = leaf_yaw;

    pos_leaf_start = l_argument.find(',');
    if( pos_leaf_start != std::string::npos ){
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

    //parse shoot arguments
    if( LString_shoot.front() != '{' ){
        helios_runtime_error("ERROR (PlantArchitecture::parseStringShoot): Shoot string is not formatted correctly. All shoots should start with a curly bracket containing two arguments {X,Y}.");
    }
    size_t pos_shoot_end = lstring_tobeparsed.find('}');
    std::string shoot_argument = lstring_tobeparsed.substr(1, pos_shoot_end - 1);
    std::string phytomer_label;
    parseShootArgument(shoot_argument, phytomer_parameters, shoot_parameters, shoot_base_rotation, phytomer_label);
    lstring_tobeparsed.erase(0, pos_shoot_end + 1);

    uint shoot_node_count = 0;
    while ((pos_inode_start = lstring_tobeparsed.find(inode_delimiter)) != std::string::npos) {

        if( pos_inode_start!=0 ){
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
        if( pos_petiole_start==0 ){
            petiole_argument = lstring_tobeparsed.substr(pos_petiole_start + petiole_delimiter.length(), pos_petiole_end - pos_petiole_start - petiole_delimiter.length());
        }else{
            petiole_argument = "";
        }
        parsePetioleArgument(petiole_argument, shoot_parameters.phytomer_parameters);
        if( pos_petiole_start==0 ) {
            lstring_tobeparsed.erase(0, pos_petiole_end + 1);
        }

        size_t pos_leaf_start = lstring_tobeparsed.find(leaf_delimiter);
        size_t pos_leaf_end = lstring_tobeparsed.find(')');
        std::string leaf_argument;
        if( pos_leaf_start==0 ){
            leaf_argument = lstring_tobeparsed.substr(pos_leaf_start + leaf_delimiter.length(), pos_leaf_end - pos_leaf_start - leaf_delimiter.length());
        }else{
            leaf_argument = "";
        }
        parseLeafArgument(leaf_argument, shoot_parameters.phytomer_parameters);
        if( pos_leaf_start==0 ) {
            lstring_tobeparsed.erase(0, pos_leaf_end + 1);
        }

        //override phytomer creation function
        shoot_parameters.phytomer_parameters.phytomer_creation_function = nullptr;

        if( base_shoot ){ //this is the first phytomer of the shoot
            defineShootType("shoot_"+phytomer_label, shoot_parameters);

            if( parentID<0 ) { //this is the first shoot of the plant
                baseID = addBaseStemShoot(plantID, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, "shoot_" + phytomer_label);
            }else{ //this is a child of an existing shoot
                baseID = addChildShoot(plantID, parentID, parent_node, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, "shoot_" + phytomer_label, 0);
            }

            base_shoot = false;
        }else{
            appendPhytomerToShoot(plantID, baseID, shoot_parameters.phytomer_parameters, internode_radius, internode_length, 1, 1);
        }

        while( !lstring_tobeparsed.empty() && lstring_tobeparsed.substr(0,1) == "[" ){
            size_t pos_shoot_bracket_end = findShootClosingBracket(lstring_tobeparsed);
            if( pos_shoot_bracket_end == std::string::npos ){
                helios_runtime_error("ERROR (PlantArchitecture::parseStringShoot): Shoot string is not formatted correctly. Shoots must be closed with a ']'.");
            }
            std::string lstring_child = lstring_tobeparsed.substr(1, pos_shoot_bracket_end-1 );
            parseStringShoot(lstring_child, plantID, (int) baseID, shoot_node_count, phytomer_parameters, shoot_parameters);
            lstring_tobeparsed.erase(0, pos_shoot_bracket_end + 1);
        }

        shoot_node_count++;
    }

}

uint PlantArchitecture::generatePlantFromString(const std::string &generation_string, const PhytomerParameters &phytomer_parameters) {
    std::map<std::string,PhytomerParameters> phytomer_parameters_map;
    phytomer_parameters_map["default"] = phytomer_parameters;
    return generatePlantFromString(generation_string, phytomer_parameters_map);
}

uint PlantArchitecture::generatePlantFromString(const std::string &generation_string, const std::map<std::string,PhytomerParameters> &phytomer_parameters) {

    //check that first characters are 'Internode'
    if(generation_string.front() != '{'){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): First character of string must be '{'");
    }

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.max_nodes = 200;
    shoot_parameters.vegetative_bud_break_probability = 0;
    shoot_parameters.vegetative_bud_break_time = 0;
    shoot_parameters.phyllochron = 0;

    //assign default phytomer parameters. This can be changed later if the optional phytomer parameters label is provided in the shoot argument '{}'.
    if( phytomer_parameters.empty() ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Phytomer parameters must be provided.");
    }

    uint plantID;

    plantID = addPlantInstance(nullorigin, 0);

    size_t pos_first_child_shoot = generation_string.find('[');

    if( pos_first_child_shoot==std::string::npos ) {
        pos_first_child_shoot = generation_string.length();
    }

    parseStringShoot(generation_string, plantID, -1, 0, phytomer_parameters, shoot_parameters);


    return plantID;

}

void PlantArchitecture::writePlantStructureXML(uint plantID, const std::string &filename) const{

    if( plant_instances.find(plantID)==plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureXML): Plant ID " + std::to_string(plantID) + " does not exist.");
    }

    //\todo Check the extension of 'filename' and add .xml if needed

    std::ofstream output_xml(filename);

    if( !output_xml.is_open() ){
        helios_runtime_error("ERROR (PlantArchitecture::writePlantStructureXML): Could not open file " + filename + " for writing. Make sure the directory exists and is writable.");
    }

    output_xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    output_xml << "<helios>" << std::endl;
    output_xml << "\t<plant_instance ID=\"" << plantID << "\">" << std::endl;

    output_xml << "\t\t<base_position> " << plant_instances.at(plantID).base_position.x << " " << plant_instances.at(plantID).base_position.y << " " << plant_instances.at(plantID).base_position.z << " </base_position>" << std::endl;
    output_xml << "\t\t<plant_age> " << plant_instances.at(plantID).current_age << " </plant_age>" << std::endl;

    for( auto& shoot : plant_instances.at(plantID).shoot_tree ) {

        output_xml << "\t\t<shoot ID=\"" << shoot->ID << "\">" << std::endl;
        output_xml << "\t\t\t<shoot_type_label> " << shoot->shoot_type_label << " </shoot_type_label>" << std::endl;
        output_xml << "\t\t\t<parent_shoot_ID> " << shoot->parent_shoot_ID << " </parent_shoot_ID>" << std::endl;
        output_xml << "\t\t\t<parent_node_index> " << shoot->parent_node_index << " </parent_node_index>" << std::endl;
        output_xml << "\t\t\t<parent_petiole_index> " << shoot->parent_petiole_index << " </parent_petiole_index>" << std::endl;
        output_xml << "\t\t\t<base_rotation> " << rad2deg(shoot->base_rotation.pitch) << " " << rad2deg(shoot->base_rotation.yaw) << " " << rad2deg(shoot->base_rotation.roll) << " </base_rotation>" << std::endl;

        for (auto &phytomer: shoot->phytomers) {

            output_xml << "\t\t\t<phytomer>" << std::endl;
            output_xml << "\t\t\t\t<internode>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_length>" << phytomer->getInternodeLength() << "</internode_length>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_radius>" << phytomer->getInternodeRadius() << "</internode_radius>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_pitch>" << rad2deg(phytomer->internode_pitch) << "</internode_pitch>" << std::endl;
            output_xml << "\t\t\t\t\t<internode_phyllotactic_angle>" << rad2deg(phytomer->internode_phyllotactic_angle) << "</internode_phyllotactic_angle>" << std::endl;

            for (uint petiole = 0; petiole < phytomer->petiole_length.size(); petiole++) {

                output_xml << "\t\t\t\t\t<petiole>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_length>" << phytomer->petiole_length.at(petiole) << "</petiole_length>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_radius>" << phytomer->petiole_radii.at(petiole).front() << "</petiole_radius>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_pitch>" << rad2deg(phytomer->petiole_pitch) << "</petiole_pitch>" << std::endl;
                output_xml << "\t\t\t\t\t\t<petiole_curvature>" << phytomer->petiole_curvature << "</petiole_curvature>" << std::endl;
                if( phytomer->leaf_rotation.at(petiole).size()==1 ){ //not compound leaf
                    output_xml << "\t\t\t\t\t\t<leaflet_scale>" << 1.0 << "</leaflet_scale>" << std::endl;
                }else {
                    float tip_ind = floor(float(phytomer->leaf_rotation.at(petiole).size() - 1) / 2.f);
                    output_xml << "\t\t\t\t\t\t<leaflet_scale>" << phytomer->leaf_size_max.at(petiole).at(int(tip_ind-1)) / max(phytomer->leaf_size_max.at(petiole)) << "</leaflet_scale>" << std::endl;
                }

                for( uint leaf=0; leaf < phytomer->leaf_rotation.at(petiole).size(); leaf++ ){
                    output_xml << "\t\t\t\t\t\t<leaf>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_scale>" << phytomer->leaf_size_max.at(petiole).at(leaf)*phytomer->current_leaf_scale_factor << "</leaf_scale>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_pitch>" << rad2deg(phytomer->leaf_rotation.at(petiole).at(leaf).pitch) << "</leaf_pitch>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_yaw>" << rad2deg(phytomer->leaf_rotation.at(petiole).at(leaf).yaw) << "</leaf_yaw>" << std::endl;
                    output_xml << "\t\t\t\t\t\t\t<leaf_roll>" << rad2deg(phytomer->leaf_rotation.at(petiole).at(leaf).roll) << "</leaf_roll>" << std::endl;
                    output_xml << "\t\t\t\t\t\t</leaf>" << std::endl;
                }

                output_xml << "\t\t\t\t\t</petiole>" << std::endl;
            }
            output_xml << "\t\t\t\t</internode>" << std::endl;
            output_xml << "\t\t\t</phytomer>" << std::endl;
        }
        output_xml << "\t\t</shoot>" << std::endl;
    }
    output_xml << "\t</plant_instance>" << std::endl;
    output_xml << "</helios>" << std::endl;
    output_xml.close();


}

std::vector<uint> PlantArchitecture::readPlantStructureXML( const std::string &filename, bool quiet){

    if( !quiet ) {
        std::cout << "Loading plant architecture XML file: " << filename << "..." << std::flush;
    }

    std::string fn = filename;
    std::string ext = getFileExtension(filename);
    if( ext != ".xml" && ext != ".XML" ) {
        helios_runtime_error("failed.\n File " + fn + " is not XML format.");
    }

    std::vector<uint> plantIDs;

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    //load file
    pugi::xml_parse_result load_result = xmldoc.load_file(filename.c_str());

    //error checking
    if (!load_result){
        helios_runtime_error("ERROR (Context::readPlantStructureXML): Could not parse " + std::string(filename) + ":\nError description: " + load_result.description() );
    }

    pugi::xml_node helios = xmldoc.child("helios");

    pugi::xml_node node;
    std::string node_string;

    if( helios.empty() ){
        if( !quiet ) {
            std::cout << "failed." << std::endl;
        }
        helios_runtime_error("ERROR (Context::readPlantStructureXML): XML file must have tag '<helios> ... </helios>' bounding all other tags.");
    }

    size_t phytomer_count = 0;

    std::map<int,int> shoot_ID_mapping;

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

                float petiole_length;
                float petiole_radius;
                float petiole_pitch;
                float petiole_curvature;
                float leaflet_scale;
                std::vector<std::vector<float>> leaf_scale; //first index is petiole within internode; second index is leaf within petiole
                std::vector<std::vector<float>> leaf_pitch;
                std::vector<std::vector<float>> leaf_yaw;
                std::vector<std::vector<float>> leaf_roll;
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

                    // leaflet scale factor
                    node_string = "leaflet_scale";
                    leaflet_scale = parse_xml_tag_float(petiole.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML");

                    leaf_scale.resize(leaf_scale.size() + 1);
                    leaf_pitch.resize(leaf_pitch.size() + 1);
                    leaf_yaw.resize(leaf_yaw.size() + 1);
                    leaf_roll.resize(leaf_roll.size() + 1);
                    for (pugi::xml_node leaf = petiole.child("leaf"); leaf; leaf = leaf.next_sibling("leaf")) {

                        // leaf scale factor
                        node_string = "leaf_scale";
                        leaf_scale.back().push_back( parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML") );

                        // leaf pitch
                        node_string = "leaf_pitch";
                        leaf_pitch.back().push_back( parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML") );

                        // leaf yaw
                        node_string = "leaf_yaw";
                        leaf_yaw.back().push_back( parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML") );

                        // leaf roll
                        node_string = "leaf_roll";
                        leaf_roll.back().push_back( parse_xml_tag_float(leaf.child(node_string.c_str()), node_string, "PlantArchitecture::readPlantStructureXML") );

                    }
                } //petioles

                if( shoot_types.find(shoot_type_label)==shoot_types.end() ){
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

                float tip_ind = floor(float(leaf_scale.front().size() - 1) / 2.f);
                shoot_parameters.phytomer_parameters.leaf.prototype_scale = leaf_scale.front().at(tip_ind);
                shoot_parameters.phytomer_parameters.leaf.pitch = 0;
                shoot_parameters.phytomer_parameters.leaf.yaw = 0;
                shoot_parameters.phytomer_parameters.leaf.roll = 0;
                shoot_parameters.phytomer_parameters.leaf.leaflet_scale = leaflet_scale;

                std::string shoot_label = "shoot_" + std::to_string(phytomer_count);
                defineShootType( shoot_label, shoot_parameters);

                if( base_shoot ){

                    if( parent_shoot_ID<0 ) { //this is the first shoot of the plant
                        current_shoot_ID = addBaseStemShoot(plantID, 1, base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, shoot_label);
                        shoot_ID_mapping[shootID] = current_shoot_ID;
                    }else{ //this is a child of an existing shoot
                        current_shoot_ID = addChildShoot(plantID, shoot_ID_mapping.at(parent_shoot_ID), parent_node_index, 1, base_rotation, internode_radius, internode_length, 1.f, 1.f, 0, shoot_label, parent_petiole_index);
                        shoot_ID_mapping[shootID] = current_shoot_ID;
                    }

                    base_shoot = false;
                }else{
                    appendPhytomerToShoot(plantID, current_shoot_ID, shoot_parameters.phytomer_parameters, internode_radius, internode_length, 1, 1);
                }

                //rotate leaves
                auto phytomer_ptr = plant_instances.at(plantID).shoot_tree.at(current_shoot_ID)->phytomers.back();
                for( int petiole = 0; petiole<phytomer_ptr->leaf_rotation.size(); petiole++ ){
                    for( int leaf=0; leaf<phytomer_ptr->leaf_rotation.at(petiole).size(); leaf++ ){
                        phytomer_ptr->rotateLeaf(petiole, leaf, make_AxisRotation(deg2rad(leaf_pitch.at(petiole).at(leaf)), deg2rad(leaf_yaw.at(petiole).at(leaf)), deg2rad(-leaf_roll.at(petiole).at(leaf))));
                    }
                }

                phytomer_count++;
        } //phytomers

        } //shoots

    } //plant instances

    if( !quiet ) {
        std::cout << "done." << std::endl;
    }
    return plantIDs;

}