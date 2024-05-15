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

        float length = phytomer->internode_length;
        float radius = phytomer->internode_radius_initial;

        outstring += "Internode(" + std::to_string(length) + "," + std::to_string(radius) + "," + std::to_string(phytomer->phytomer_parameters.internode.pitch.val() /*\todo */ ) + ")";

        outstring += "Petiole(" + std::to_string(phytomer->phytomer_parameters.petiole.length.val() /*\todo */ ) + "," + std::to_string(phytomer->phytomer_parameters.petiole.pitch.val() /*\todo */) + ")";

        outstring += "Leaf(" + std::to_string(phytomer->phytomer_parameters.leaf.prototype_scale.val() /*\todo */) + "," + std::to_string(phytomer->phytomer_parameters.leaf.pitch.val() /*\todo */) + "," + std::to_string(phytomer->phytomer_parameters.leaf.yaw.val() /*\todo */) + "," + std::to_string(phytomer->phytomer_parameters.leaf.roll.val() /*\todo */) + ")";

        if( shoot->childIDs.find(node_number)!=shoot->childIDs.end() ){
            outstring = makeShootString(outstring, shoot_tree.at(shoot->childIDs.at(node_number)), shoot_tree );
        }

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

    if( pos_shoot_start != std::string::npos ) {
        pos_shoot_start = s_argument.find(',');
        phytomer_label = s_argument.substr(0, pos_shoot_start);
        s_argument.erase(0, pos_shoot_start + 1);
        if (phytomer_parameters.find(phytomer_label) == phytomer_parameters.end()) {
            helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): Phytomer parameters with label " + phytomer_label + " was not provided to PlantArchitecture::generatePlantFromString().");
        }
        shoot_parameters.phytomer_parameters = phytomer_parameters.at(phytomer_label);
    }else{
        phytomer_label = phytomer_parameters.begin()->first;
        shoot_parameters.phytomer_parameters = phytomer_parameters.begin()->second;
    }

}

void PlantArchitecture::parseInternodeArgument(const std::string &internode_argument, float &internode_radius, float &internode_length, PhytomerParameters &phytomer_parameters) {

    //internode argument order Internode():
    // 1. internode length (m)
    // 2. internode radius (m)
    // 3. internode pitch (degrees)

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
    if( pos_inode_start != std::string::npos ){
        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): 'Internode()' does not have the correct number of values given.");
    }
    float internode_pitch = std::stof(inode_argument.substr(0, pos_inode_start));
    inode_argument.erase(0, pos_inode_start + 1);
    phytomer_parameters.internode.pitch = internode_pitch;

}

void PlantArchitecture::parsePetioleArgument(const std::string& petiole_argument, PhytomerParameters &phytomer_parameters ){

    //petiole argument order Petiole():
    // 1. petiole length (m)
    // 2. petiole pitch (degrees)

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

size_t findShootClosingBracket(const std::string &lstring ){


    size_t pos_open = lstring.find_first_of('[',1);
    size_t pos_close = lstring.find_first_of(']', 1);

    if( pos_open>pos_close ){
        return pos_close;
    }

    while( pos_open!=std::string::npos ){
        pos_open = lstring.find('[', pos_close+1);
        pos_close = lstring.find(']', pos_close+1);
    }

    return pos_close;

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
                baseID = addBaseStemShoot(plantID, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, "shoot_"+phytomer_label );
            }else{ //this is a child of an existing shoot
                baseID = addChildShoot(plantID, parentID, parent_node, 1, shoot_base_rotation, internode_radius, internode_length, 1.f, 1.f, "shoot_" + phytomer_label, 0);
            }

            base_shoot = false;
        }else{
            addPhytomerToShoot(plantID, baseID, shoot_parameters.phytomer_parameters, internode_radius, internode_length, 1, 1);
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
//    if( lsystems_string.substr(0,10)!="Internode(" ){
//        helios_runtime_error("ERROR (PlantArchitecture::generatePlantFromString): First characters of string must be 'Internode('");
//    }
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