/** \file "PlantLibrary.cpp" Contains routines for loading and building plant models from a library of predefined plant types.

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

void PlantArchitecture::loadPlantModelFromLibrary( const std::string &plant_label ){

    current_plant_model = plant_label;
    initializeDefaultShoots(plant_label);

}

uint PlantArchitecture::buildPlantInstanceFromLibrary( const helios::vec3 &base_position, float age ){

    if( current_plant_model.empty() ){
        helios_runtime_error("ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): current plant model has not been initialized from library. You must call loadPlantModelFromLibrary() first.");
    }

    uint plantID = 0;
    if( current_plant_model == "almond" ) {
        plantID = buildAlmondTree(base_position);
    }else if( current_plant_model == "apple" ) {
        plantID = buildAppleTree(base_position);
    }else if( current_plant_model == "asparagus" ) {
        plantID = buildAsparagusPlant(base_position);
    }else if( current_plant_model == "bindweed" ) {
        plantID = buildBindweedPlant(base_position);
    }else if( current_plant_model == "bean" ) {
        plantID = buildBeanPlant(base_position);
    }else if( current_plant_model == "capsicum" ) {
        plantID = buildCapsicumPlant(base_position);
    }else if( current_plant_model == "cheeseweed" ) {
        plantID = buildCheeseweedPlant(base_position);
    }else if( current_plant_model == "cowpea" ) {
        plantID = buildCowpeaPlant(base_position);
    }else if( current_plant_model == "grapevine_VSP" ) {
        plantID = buildGrapevineVSP(base_position);
    }else if( current_plant_model == "groundcherryweed" ) {
        plantID = buildGroundCherryWeedPlant(base_position);
    }else if( current_plant_model == "maize" ) {
        plantID = buildMaizePlant(base_position);
    }else if( current_plant_model == "olive" ) {
        plantID = buildOliveTree(base_position);
    }else if( current_plant_model == "pistachio" ) {
        plantID = buildPistachioTree(base_position);
    }else if( current_plant_model == "puncturevine" ) {
        plantID = buildPuncturevinePlant(base_position);
    }else if( current_plant_model == "easternredbud" ) {
        plantID = buildEasternRedbudPlant(base_position);
    }else if( current_plant_model == "rice" ) {
        plantID = buildRicePlant(base_position);
    }else if( current_plant_model == "butterlettuce" ) {
        plantID = buildButterLettucePlant(base_position);
    }else if( current_plant_model == "sorghum" ) {
        plantID = buildSorghumPlant(base_position);
    }else if( current_plant_model == "soybean" ) {
        plantID = buildSoybeanPlant(base_position);
    }else if( current_plant_model == "strawberry" ) {
        plantID = buildStrawberryPlant(base_position);
    }else if( current_plant_model == "sugarbeet" ) {
        plantID = buildSugarbeetPlant(base_position);
    }else if( current_plant_model == "tomato" ) {
        plantID = buildTomatoPlant(base_position);
    }else if( current_plant_model == "cherrytomato" ) {
        plantID = buildCherryTomatoPlant(base_position);
    }else if( current_plant_model == "walnut" ) {
        plantID = buildWalnutTree(base_position);
    }else if( current_plant_model == "wheat" ) {
        plantID = buildWheatPlant(base_position);
    }else{
        assert(true); //shouldn't be here
    }

    plant_instances.at(plantID).plant_name = current_plant_model;

    if( age>0 ){
        advanceTime( plantID, age);
    }

    return plantID;

}

ShootParameters PlantArchitecture::getCurrentShootParameters( const std::string &shoot_type_label ){

    if( shoot_types.find(shoot_type_label) == shoot_types.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getCurrentShootParameters): shoot type label of " + shoot_type_label + " does not exist in the current shoot parameters.");
    }

    return shoot_types.at(shoot_type_label);
}

std::map<std::string, ShootParameters> PlantArchitecture::getCurrentShootParameters(){
    if( shoot_types.empty() ){
        std::cerr << "WARNING (PlantArchitecture::getCurrentShootParameters): No plant models have been loaded. You need to first load a plant model from the library (see loadPlantModelFromLibrary()) or manually add shoot parameters (see updateCurrentShootParameters())." << std::endl;
    }
    return shoot_types;
}

std::map<std::string, PhytomerParameters> PlantArchitecture::getCurrentPhytomerParameters( ){
    if( shoot_types.empty() ){
        std::cerr << "WARNING (PlantArchitecture::getCurrentPhytomerParameters): No plant models have been loaded. You need to first load a plant model from the library (see loadPlantModelFromLibrary()) or manually add shoot parameters (see updateCurrentShootParameters())." << std::endl;
    }
    std::map<std::string, PhytomerParameters> phytomer_parameters;
    for( const auto & type : shoot_types ){
        phytomer_parameters[type.first] = type.second.phytomer_parameters;
    }
    return phytomer_parameters;
}

void PlantArchitecture::updateCurrentShootParameters( const std::string &shoot_type_label, const ShootParameters &params ){
    shoot_types[shoot_type_label] = params;
}

void PlantArchitecture::updateCurrentShootParameters( const std::map<std::string, ShootParameters> &params ){
    shoot_types = params;
}

void PlantArchitecture::initializeDefaultShoots( const std::string &plant_label ){

    if( plant_label == "almond" ) {
        initializeAlmondTreeShoots();
    }else if( plant_label == "apple" ) {
        initializeAppleTreeShoots();
    }else if( plant_label == "asparagus" ) {
        initializeAsparagusShoots();
    }else if( plant_label == "bindweed" ) {
        initializeBindweedShoots();
    }else if( plant_label == "bean" ) {
        initializeBeanShoots();
    }else if( plant_label == "capsicum" ) {
        initializeCapsicumShoots();
    }else if( plant_label == "cheeseweed" ) {
        initializeCheeseweedShoots();
    }else if( plant_label == "cowpea" ) {
        initializeCowpeaShoots();
    }else if( plant_label == "grapevine_VSP" ) {
        initializeGrapevineVSPShoots();
    }else if( plant_label == "groundcherryweed" ) {
        initializeGroundCherryWeedShoots();
    }else if( plant_label == "maize" ) {
        initializeMaizeShoots();
    }else if( plant_label == "olive" ) {
        initializeOliveTreeShoots();
    }else if( plant_label == "pistachio" ) {
        initializePistachioTreeShoots();
    }else if( plant_label == "puncturevine" ) {
        initializePuncturevineShoots();
    }else if( plant_label == "easternredbud" ) {
        initializeEasternRedbudShoots();
    }else if( plant_label == "rice" ) {
        initializeRiceShoots();
    }else if( plant_label == "butterlettuce" ) {
        initializeButterLettuceShoots();
    }else if( plant_label == "sorghum" ) {
        initializeSorghumShoots();
    }else if( plant_label == "soybean" ) {
        initializeSoybeanShoots();
    }else if( plant_label == "strawberry" ) {
        initializeStrawberryShoots();
    }else if( plant_label == "sugarbeet" ) {
        initializeSugarbeetShoots();
    }else if( plant_label == "tomato" ) {
        initializeTomatoShoots();
    }else if( plant_label == "cherrytomato" ) {
        initializeCherryTomatoShoots();
    }else if( plant_label == "walnut" ) {
        initializeWalnutTreeShoots();
    }else if( plant_label == "wheat" ) {
        initializeWheatShoots();
    }else{
        helios_runtime_error("ERROR (PlantArchitecture::loadPlantModelFromLibrary): plant label of " + plant_label + " does not exist in the library.");
    }

}

void PlantArchitecture::initializeAlmondTreeShoots(){

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/AlmondLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.33f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature = 0.05;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 1;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_almond(context_ptr->getRandomGenerator());

    phytomer_parameters_almond.internode.pitch = 3;
    phytomer_parameters_almond.internode.phyllotactic_angle.uniformDistribution( 120, 160 );
    phytomer_parameters_almond.internode.radius_initial = 0.002;
    phytomer_parameters_almond.internode.length_segments = 1;
    phytomer_parameters_almond.internode.image_texture = "plugins/plantarchitecture/assets/textures/AlmondBark.jpg";
    phytomer_parameters_almond.internode.max_floral_buds_per_petiole = 1; //

    phytomer_parameters_almond.petiole.petioles_per_internode = 1;
    phytomer_parameters_almond.petiole.pitch.uniformDistribution(-145,-90);
    phytomer_parameters_almond.petiole.taper = 0.1;
    phytomer_parameters_almond.petiole.curvature = 0;
    phytomer_parameters_almond.petiole.length = 0.04;
    phytomer_parameters_almond.petiole.radius = 0.0005;
    phytomer_parameters_almond.petiole.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;
    phytomer_parameters_almond.petiole.color = make_RGBcolor(0.61,0.5,0.24);

    phytomer_parameters_almond.leaf.leaves_per_petiole = 1;
    phytomer_parameters_almond.leaf.roll.uniformDistribution(-10,10);
    phytomer_parameters_almond.leaf.prototype_scale = 0.08;
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
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"},{1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3,0.2,0.2);
    shoot_parameters_proleptic.phytomer_parameters.internode.radial_subdivisions = 5;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = AlmondPhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes = 20;
    shoot_parameters_proleptic.max_nodes_per_season = 15;
    shoot_parameters_proleptic.phyllochron_min = 1;
    shoot_parameters_proleptic.elongation_rate_max = 0.3;
    shoot_parameters_proleptic.girth_area_factor = 8.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.6;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 200;
    shoot_parameters_proleptic.tortuosity = 3;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution( 25, 30);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 15;
    shoot_parameters_proleptic.internode_length_max = 0.02;
    shoot_parameters_proleptic.internode_length_min = 0.002;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.002;
    shoot_parameters_proleptic.fruit_set_probability = 0.4;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 3;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic","sylleptic"},{1.0,0.});

    // Sylleptic shoots
    ShootParameters shoot_parameters_sylleptic = shoot_parameters_proleptic;
//    shoot_parameters_sylleptic.phytomer_parameters.internode.color = RGB::red;
    shoot_parameters_sylleptic.phytomer_parameters.internode.image_texture = "";
    shoot_parameters_sylleptic.phytomer_parameters.leaf.prototype_scale = 0.12;
    shoot_parameters_sylleptic.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20);
    shoot_parameters_sylleptic.insertion_angle_tip = 0;
    shoot_parameters_sylleptic.insertion_angle_decay_rate = 0;
    shoot_parameters_sylleptic.phyllochron_min = 1;
    shoot_parameters_sylleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_sylleptic.gravitropic_curvature= 600;
    shoot_parameters_sylleptic.internode_length_max = 0.02;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true;
    shoot_parameters_sylleptic.defineChildShootTypes({"proleptic"},{1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
//    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 15;
    shoot_parameters_scaffold.gravitropic_curvature = 150;
    shoot_parameters_scaffold.internode_length_max = 0.02;
    shoot_parameters_scaffold.tortuosity = 1.;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"},{1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);

}

uint PlantArchitecture::buildAlmondTree(const helios::vec3 &base_position) {

    if( shoot_types.empty() ){
        //automatically initialize almond tree shoots
        initializeAlmondTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

//    enableEpicormicChildShoots(plantID,"sylleptic",0.001);

    uint uID_trunk = addBaseStemShoot(plantID, 19, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), 0.015, 0.03, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot( plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0.01, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 4;//context_ptr->randu(4,5);

    for( int i=0; i<Nscaffolds; i++ ) {
        float pitch = context_ptr->randu(deg2rad(35), deg2rad(45));
        uint uID_shoot = addChildShoot( plantID, uID_trunk, getShootNodeCount(plantID,uID_trunk)-i-1, context_ptr->randu(7, 9), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007, 0.06, 1.f, 1.f, 0.5, "scaffold", 0);

    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200, false);
    plant_instances.at(plantID).max_age = 1825;

    return plantID;

}

void PlantArchitecture::initializeAppleTreeShoots(){

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/AppleLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.6f;
    leaf_prototype.midrib_fold_fraction = 0.4f;
    leaf_prototype.longitudinal_curvature = -0.3f;
    leaf_prototype.lateral_curvature = 0.1f;
    leaf_prototype.subdivisions = 3;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_apple(context_ptr->getRandomGenerator());

    phytomer_parameters_apple.internode.pitch = 0;
    phytomer_parameters_apple.internode.phyllotactic_angle.uniformDistribution( 130, 145 );
    phytomer_parameters_apple.internode.radius_initial = 0.004;
    phytomer_parameters_apple.internode.length_segments = 1;
    phytomer_parameters_apple.internode.image_texture = "plugins/plantarchitecture/assets/textures/AppleBark.jpg";
    phytomer_parameters_apple.internode.max_floral_buds_per_petiole = 1;

    phytomer_parameters_apple.petiole.petioles_per_internode = 1;
    phytomer_parameters_apple.petiole.pitch.uniformDistribution(-40,-25);
    phytomer_parameters_apple.petiole.taper = 0.1;
    phytomer_parameters_apple.petiole.curvature = 0;
    phytomer_parameters_apple.petiole.length = 0.04;
    phytomer_parameters_apple.petiole.radius = 0.00075;
    phytomer_parameters_apple.petiole.length_segments = 1;
    phytomer_parameters_apple.petiole.radial_subdivisions = 3;
    phytomer_parameters_apple.petiole.color = make_RGBcolor(0.61,0.5,0.24);

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
    shoot_parameters_trunk.defineChildShootTypes({"proleptic"},{1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_apple;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3,0.2,0.2);
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = ApplePhytomerCreationFunction;
    shoot_parameters_proleptic.max_nodes = 40;
    shoot_parameters_proleptic.max_nodes_per_season = 20;
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.15;
    shoot_parameters_proleptic.girth_area_factor = 5.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.4;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(450,500);
    shoot_parameters_proleptic.tortuosity = 3;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution( 30, 40);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 20;
    shoot_parameters_proleptic.internode_length_max = 0.04;
    shoot_parameters_proleptic.internode_length_min = 0.01;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.4;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 1;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"},{1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);

}

uint PlantArchitecture::buildAppleTree(const helios::vec3 &base_position) {

    if( shoot_types.empty() ){
        //automatically initialize apple tree shoots
        initializeAppleTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, 19, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), 0.015, 0.04, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot( plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 4;//context_ptr->randu(4,5);

    for( int i=0; i<Nscaffolds; i++ ) {
        float pitch = context_ptr->randu(deg2rad(55), deg2rad(65));
        uint uID_shoot = addChildShoot( plantID, uID_trunk, getShootNodeCount(plantID,uID_trunk)-i-1, context_ptr->randu(7, 9), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.005, 0.04, 1.f, 1.f, 0.5, "proleptic", 0);

    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200, false);
    plant_instances.at(plantID).max_age = 1460;

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
        //automatically initialize asparagus plant shoots
        initializeAsparagusShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(0, 0.f, 0.f), 0.0003, 0.015, 1, 0.1, 0.1, "main");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, 5, 100, false);

    plant_instances.at(plantID).max_age = 20;

    return plantID;

}

void PlantArchitecture::initializeBindweedShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_bindweed(context_ptr->getRandomGenerator());

    phytomer_parameters_bindweed.internode.pitch.uniformDistribution(0,15);
    phytomer_parameters_bindweed.internode.phyllotactic_angle = 180.f;
    phytomer_parameters_bindweed.internode.radius_initial = 0.0012;
    phytomer_parameters_bindweed.internode.color = make_RGBcolor(0.3,0.38,0.21);
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
    phytomer_parameters_bindweed.leaf.roll = 90;
    phytomer_parameters_bindweed.leaf.prototype_scale = 0.05;
    phytomer_parameters_bindweed.leaf.prototype.OBJ_model_file = "plugins/plantarchitecture/assets/obj/BindweedLeaf.obj";

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
    shoot_parameters_primary.vegetative_bud_break_probability_min = 0.15;
    shoot_parameters_primary.vegetative_bud_break_probability_decay_rate = -1.;
    shoot_parameters_primary.vegetative_bud_break_time = 3;
    shoot_parameters_primary.base_roll = 90;
    shoot_parameters_primary.phyllochron_min = 1;
    shoot_parameters_primary.elongation_rate_max = 0.25;
    shoot_parameters_primary.girth_area_factor = 0;
    shoot_parameters_primary.internode_length_max = 0.03;
    shoot_parameters_primary.internode_length_decay_rate = 0;
    shoot_parameters_primary.insertion_angle_tip.uniformDistribution(50, 80);
    shoot_parameters_primary.flowers_require_dormancy = false;
    shoot_parameters_primary.growth_requires_dormancy = false;
    shoot_parameters_primary.flower_bud_break_probability = 0.2;
    shoot_parameters_primary.determinate_shoot_growth = false;
    shoot_parameters_primary.max_nodes = 15;
    shoot_parameters_primary.gravitropic_curvature = 40;
    shoot_parameters_primary.tortuosity = 0;
    shoot_parameters_primary.defineChildShootTypes({"secondary_bindweed"}, {1.f});

    ShootParameters shoot_parameters_base = shoot_parameters_primary;
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_bindweed;
    shoot_parameters_base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5-10,137.5+10);
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
    shoot_parameters_base.max_nodes.uniformDistribution(3,5);
    shoot_parameters_base.defineChildShootTypes({"primary_bindweed"},{1.f});

    ShootParameters shoot_parameters_children = shoot_parameters_primary;
    shoot_parameters_children.base_roll = 0;

    defineShootType("base_bindweed", shoot_parameters_base);
    defineShootType("primary_bindweed", shoot_parameters_primary);
    defineShootType("secondary_bindweed", shoot_parameters_children);

}

uint PlantArchitecture::buildBindweedPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize bindweed plant shoots
        initializeBindweedShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0.f, 0.f), 0.001, 0.001, 1, 1, 0, "base_bindweed");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 14, -1, -1, 1000, false);

    plant_instances.at(plantID).max_age = 50;

    return plantID;

}

void PlantArchitecture::initializeBeanShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype_trifoliate(context_ptr->getRandomGenerator());
    leaf_prototype_trifoliate.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/BeanLeaf_tip.png";
    leaf_prototype_trifoliate.leaf_texture_file[-1] = "plugins/plantarchitecture/assets/textures/BeanLeaf_left_centered.png";
    leaf_prototype_trifoliate.leaf_texture_file[1] = "plugins/plantarchitecture/assets/textures/BeanLeaf_right_centered.png";
    leaf_prototype_trifoliate.leaf_aspect_ratio = 1.f;
    leaf_prototype_trifoliate.midrib_fold_fraction = 0.2;
    leaf_prototype_trifoliate.longitudinal_curvature.uniformDistribution(-0.3f, -0.2f);
    leaf_prototype_trifoliate.lateral_curvature = -1.f;
    leaf_prototype_trifoliate.subdivisions = 6;
    leaf_prototype_trifoliate.unique_prototypes = 5;
    leaf_prototype_trifoliate.build_petiolule = true;

    LeafPrototype leaf_prototype_unifoliate = leaf_prototype_trifoliate;
    leaf_prototype_unifoliate.leaf_texture_file.clear();
    leaf_prototype_unifoliate.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/BeanLeaf_unifoliate_centered.png";
    leaf_prototype_unifoliate.unique_prototypes = 2;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());

    phytomer_parameters_trifoliate.internode.pitch = 20;
    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters_trifoliate.internode.radius_initial = 0.001;
    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.2,0.25,0.05);
    phytomer_parameters_trifoliate.internode.length_segments = 2;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(20,50);
    phytomer_parameters_trifoliate.petiole.radius = 0.0015;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.1,0.14);
    phytomer_parameters_trifoliate.petiole.taper = 0.;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-100,200);
    phytomer_parameters_trifoliate.petiole.color = make_RGBcolor(0.28,0.35,0.07);
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.normalDistribution(0, 20);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll = -15;
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.3;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.09,0.11);
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
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50,70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20,20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = BeanFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.15,0.2);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = BeanFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8,1.0);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.0001;
    phytomer_parameters_unifoliate.petiole.radius = 0.0004;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(50,70);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.04;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype = leaf_prototype_unifoliate;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 25;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40,60);
//    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.025;
//    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
//    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20,20);
    shoot_parameters_trifoliate.gravitropic_curvature = 200;

    shoot_parameters_trifoliate.phyllochron_min = 2;
    shoot_parameters_trifoliate.elongation_rate_max = 0.1;
    shoot_parameters_trifoliate.girth_area_factor = 1.5f;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 15;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_decay_rate = -0.4;
//    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.3,0.4);
    shoot_parameters_trifoliate.fruit_set_probability = 0.4;
//    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
//    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
//    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)

    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"},{1.0});


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
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"},{1.0});

    defineShootType("unifoliate",shoot_parameters_unifoliate);
    defineShootType("trifoliate",shoot_parameters_trifoliate);


}

uint PlantArchitecture::buildBeanPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize bean plant shoots
        initializeBeanShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.03, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").phytomer_parameters.internode.radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeCapsicumShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/CapsicumLeaf.png";
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
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5-10, 137.5+10);
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
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12,0.15);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.01;
    phytomer_parameters.peduncle.radius = 0.001;
    phytomer_parameters.peduncle.pitch.uniformDistribution(10,30);
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -700;
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 3;
    phytomer_parameters.peduncle.radial_subdivisions = 6;

    phytomer_parameters.inflorescence.flowers_per_peduncle = 1;
    phytomer_parameters.inflorescence.pitch = 20;
    phytomer_parameters.inflorescence.roll.uniformDistribution(-30,30);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.025;
    phytomer_parameters.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale.uniformDistribution(0.12,0.16);
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
    shoot_parameters.base_yaw.uniformDistribution(-20,20);
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

    shoot_parameters.defineChildShootTypes({"secondary"},{1.0});

    defineShootType("mainstem",shoot_parameters);

    ShootParameters shoot_parameters_secondary = shoot_parameters;
    shoot_parameters_secondary.phytomer_parameters = phytomer_parameters_secondary;
    shoot_parameters_secondary.max_nodes = 7;
    shoot_parameters_secondary.phyllochron_min = 6;
    shoot_parameters_secondary.vegetative_bud_break_probability_min = 0.2;

    defineShootType( "secondary", shoot_parameters_secondary);

}

uint PlantArchitecture::buildCapsicumPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize capsicum plant shoots
        initializeCapsicumShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.002, shoot_types.at("mainstem").internode_length_max.val(), 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 75, 14, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeCheeseweedShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.OBJ_model_file = "plugins/plantarchitecture/assets/obj/CheeseweedLeaf.obj";

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_cheeseweed(context_ptr->getRandomGenerator());

    phytomer_parameters_cheeseweed.internode.pitch = 0;
    phytomer_parameters_cheeseweed.internode.phyllotactic_angle.uniformDistribution( 127.5f, 147.5);
    phytomer_parameters_cheeseweed.internode.radius_initial = 0.0005;
    phytomer_parameters_cheeseweed.internode.color = make_RGBcolor(0.60, 0.65, 0.40);
    phytomer_parameters_cheeseweed.internode.length_segments = 1;

    phytomer_parameters_cheeseweed.petiole.petioles_per_internode = 1;
    phytomer_parameters_cheeseweed.petiole.pitch.uniformDistribution(45, 75);
    phytomer_parameters_cheeseweed.petiole.radius = 0.0005;
    phytomer_parameters_cheeseweed.petiole.length.uniformDistribution(0.02,0.06);
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
        //automatically initialize cheeseweed plant shoots
        initializeCheeseweedShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(0, 0.f, 0.f), 0.0001, 0.0025, 0.1, 0.1, 0, "base");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000, false);

    plant_instances.at(plantID).max_age = 40;

    return plantID;

}

void PlantArchitecture::initializeCowpeaShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype_trifoliate(context_ptr->getRandomGenerator());
    leaf_prototype_trifoliate.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_tip_centered.png";
    leaf_prototype_trifoliate.leaf_texture_file[-1] = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_left_centered.png";
    leaf_prototype_trifoliate.leaf_texture_file[1] = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_right_centered.png";
    leaf_prototype_trifoliate.leaf_aspect_ratio = 0.7f;
    leaf_prototype_trifoliate.midrib_fold_fraction = 0.2;
    leaf_prototype_trifoliate.longitudinal_curvature.uniformDistribution(-0.3f, -0.1f);
    leaf_prototype_trifoliate.lateral_curvature = -0.4f;
    leaf_prototype_trifoliate.subdivisions = 6;
    leaf_prototype_trifoliate.unique_prototypes = 5;
    leaf_prototype_trifoliate.build_petiolule = true;

    LeafPrototype leaf_prototype_unifoliate = leaf_prototype_trifoliate;
    leaf_prototype_unifoliate.leaf_texture_file.clear();
    leaf_prototype_unifoliate.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_unifoliate_centered.png";

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
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(45,60);
    phytomer_parameters_trifoliate.petiole.radius = 0.0018;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.06,0.08);
    phytomer_parameters_trifoliate.petiole.taper = 0.25;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-200,-50);
    phytomer_parameters_trifoliate.petiole.color = make_RGBcolor(0.2,0.25,0.06);
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.normalDistribution(45, 20);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll = -15;
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.4;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.09,0.12);
    phytomer_parameters_trifoliate.leaf.prototype = leaf_prototype_trifoliate;

    phytomer_parameters_trifoliate.peduncle.length.uniformDistribution(0.3,0.35);
    phytomer_parameters_trifoliate.peduncle.radius = 0.003;
    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 30);
    phytomer_parameters_trifoliate.peduncle.roll = 90;
    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(50, 250);
    phytomer_parameters_trifoliate.peduncle.color = make_RGBcolor(0.17,0.213,0.051);
    phytomer_parameters_trifoliate.peduncle.length_segments = 6;
    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.inflorescence.flowers_per_peduncle.uniformDistribution(1, 3);
    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.025;
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50,70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20,20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = CowpeaFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.11,0.13);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = CowpeaFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.5,0.7);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.0001;
    phytomer_parameters_unifoliate.petiole.radius = 0.0004;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(60,80);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.02;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype = leaf_prototype_unifoliate;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = CowpeaPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 20;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40,60);
//    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.025;
//    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
//    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20,20);
    shoot_parameters_trifoliate.gravitropic_curvature = 200;

    shoot_parameters_trifoliate.phyllochron_min = 2;
    shoot_parameters_trifoliate.elongation_rate_max = 0.1;
    shoot_parameters_trifoliate.girth_area_factor = 1.5f;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 10;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_min = 0.2;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_decay_rate = -0.4;
//    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.1,0.15);
    shoot_parameters_trifoliate.fruit_set_probability = 0.4;
//    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
//    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
//    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)

    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"},{1.0});


    ShootParameters shoot_parameters_unifoliate = shoot_parameters_trifoliate;
    shoot_parameters_unifoliate.phytomer_parameters = phytomer_parameters_unifoliate;
    shoot_parameters_unifoliate.max_nodes = 1;
    shoot_parameters_unifoliate.girth_area_factor = 1.f;
    shoot_parameters_unifoliate.vegetative_bud_break_probability_min = 1;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 40;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 8;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"},{1.0});

    defineShootType("unifoliate",shoot_parameters_unifoliate);
    defineShootType("trifoliate",shoot_parameters_trifoliate);

}

uint PlantArchitecture::buildCowpeaPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize cowpea plant shoots
        initializeCowpeaShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.03, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").phytomer_parameters.internode.radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeGrapevineVSPShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/GrapeLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.4,0.4);
    leaf_prototype.lateral_curvature = 0;
    leaf_prototype.wave_period = 0.3f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.subdivisions = 5;
    leaf_prototype.unique_prototypes = 10;
    leaf_prototype.leaf_offset = make_vec3(-0.3,0,0);

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_grapevine(context_ptr->getRandomGenerator());

    phytomer_parameters_grapevine.internode.pitch = 20;
    phytomer_parameters_grapevine.internode.phyllotactic_angle.uniformDistribution(160, 200);
    phytomer_parameters_grapevine.internode.radius_initial = 0.003;
    phytomer_parameters_grapevine.internode.color = make_RGBcolor(0.23,0.13,0.062);
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
    phytomer_parameters_grapevine.leaf.prototype_scale = 0.2;
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
    shoot_parameters_main.vegetative_bud_break_probability_min = 0.075;
    shoot_parameters_main.vegetative_bud_break_probability_decay_rate = 1.;
    shoot_parameters_main.vegetative_bud_break_time = 30;
    shoot_parameters_main.phyllochron_min.uniformDistribution(1.75,2.25);
    shoot_parameters_main.elongation_rate_max = 0.15;
    shoot_parameters_main.girth_area_factor = 1.f;
    shoot_parameters_main.gravitropic_curvature = 400;
    shoot_parameters_main.tortuosity = 15;
    shoot_parameters_main.internode_length_max.uniformDistribution(0.06,0.08);
    shoot_parameters_main.internode_length_decay_rate = 0;
    shoot_parameters_main.insertion_angle_tip = 45;
    shoot_parameters_main.insertion_angle_decay_rate = 0;
    shoot_parameters_main.flowers_require_dormancy = false;
    shoot_parameters_main.growth_requires_dormancy = false;
    shoot_parameters_main.determinate_shoot_growth = false;
    shoot_parameters_main.max_terminal_floral_buds = 0;
    shoot_parameters_main.flower_bud_break_probability = 0.5;
    shoot_parameters_main.fruit_set_probability = 0.2;
    shoot_parameters_main.max_nodes = 20;
    shoot_parameters_main.base_roll.uniformDistribution(90-25,90+25);
    shoot_parameters_main.base_yaw = 0;

    ShootParameters shoot_parameters_cane = shoot_parameters_main;
//    shoot_parameters_cane.phytomer_parameters.internode.image_texture = "plugins/plantarchitecture/assets/textures/GrapeBark.jpg";
    shoot_parameters_cane.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_cane.phytomer_parameters.internode.radial_subdivisions = 15;
    shoot_parameters_cane.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_cane.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_cane.insertion_angle_tip.uniformDistribution(60, 120);
    shoot_parameters_cane.girth_area_factor = 0.7f;
    shoot_parameters_cane.max_nodes = 9;
    shoot_parameters_cane.gravitropic_curvature.uniformDistribution(-20,20);
    shoot_parameters_cane.tortuosity = 1;
    shoot_parameters_cane.gravitropic_curvature = 10;
    shoot_parameters_cane.vegetative_bud_break_probability_min = 0.9;
    shoot_parameters_cane.defineChildShootTypes({"grapevine_shoot"},{1.f});

    ShootParameters shoot_parameters_trunk = shoot_parameters_main;
    shoot_parameters_trunk.phytomer_parameters.internode.image_texture = "plugins/plantarchitecture/assets/textures/GrapeBark.jpg";
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radius_initial = 0.05;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 25;
    shoot_parameters_trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_trunk.phyllochron_min = 2.5;
    shoot_parameters_trunk.insertion_angle_tip = 90;
    shoot_parameters_trunk.girth_area_factor = 0;
    shoot_parameters_trunk.max_nodes = 18;
    shoot_parameters_trunk.tortuosity = 0;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.defineChildShootTypes({"grapevine_shoot"},{1.f});

    defineShootType("grapevine_trunk", shoot_parameters_trunk);
    defineShootType("grapevine_cane", shoot_parameters_cane);
    defineShootType("grapevine_shoot", shoot_parameters_main);

}

uint PlantArchitecture::buildGrapevineVSP(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize grapevine plant shoots
        initializeGrapevineVSPShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 8, make_AxisRotation(context_ptr->randu(0,0.05*M_PI), 0, 0), shoot_types.at("grapevine_trunk").phytomer_parameters.internode.radius_initial.val(), 0.1, 1, 1, 0.1, "grapevine_trunk");

    uint uID_cane_L = appendShoot( plantID, uID_stem, 8, make_AxisRotation(context_ptr->randu(float(0.45f*M_PI),0.52f*M_PI),0,M_PI), 0.005, 0.15, 1, 1, 0.5, "grapevine_cane" );
    uint uID_cane_R = appendShoot( plantID, uID_stem, 8, make_AxisRotation(context_ptr->randu(float(0.45f*M_PI),0.52f*M_PI),M_PI,M_PI), 0.005, 0.15, 1, 1, 0.5, "grapevine_cane" );

//    makePlantDormant(plantID);

    removeShootLeaves( plantID, uID_stem );
    removeShootLeaves( plantID, uID_cane_L );
    removeShootLeaves( plantID, uID_cane_R );

    setPlantPhenologicalThresholds(plantID, 165, -1, -1, 45, 45, 200, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeGroundCherryWeedShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/GroundCherryLeaf.png";
    leaf_prototype.leaf_aspect_ratio.uniformDistribution(0.3,0.5);
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature = 0.1f;;
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
    phytomer_parameters.internode.color = make_RGBcolor(0.213, 0.270, 0.056);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45,60);
    phytomer_parameters.petiole.radius = 0.0005;
    phytomer_parameters.petiole.length = 0.025;
    phytomer_parameters.petiole.taper = 0.15;
    phytomer_parameters.petiole.curvature.uniformDistribution(-150,-50);
    phytomer_parameters.petiole.color = phytomer_parameters.internode.color;
    phytomer_parameters.petiole.length_segments = 2;

    phytomer_parameters.leaf.leaves_per_petiole = 1;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.06,0.08);
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
    phytomer_parameters.inflorescence.roll.uniformDistribution(-30,30);
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
    shoot_parameters.base_yaw.uniformDistribution(-20,20);
    shoot_parameters.gravitropic_curvature = 700;
    shoot_parameters.tortuosity = 3;

    shoot_parameters.phyllochron_min = 1;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 3.f;
    shoot_parameters.vegetative_bud_break_time = 7;
    shoot_parameters.vegetative_bud_break_probability_min = 0.2;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = -0.5;
    shoot_parameters.flower_bud_break_probability = 0.25;
    shoot_parameters.fruit_set_probability = 0.5;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = false;

    shoot_parameters.defineChildShootTypes({"mainstem"},{1.0});

    defineShootType("mainstem",shoot_parameters);

}

uint PlantArchitecture::buildGroundCherryWeedPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize ground cherry plant shoots
        initializeGroundCherryWeedShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.0025, 0.018, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 20, -1, 20, 30, 1000, false);

    plant_instances.at(plantID).max_age = 50;

    return plantID;

}

void PlantArchitecture::initializeMaizeShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.25f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.45,-0.3);
    leaf_prototype.lateral_curvature = -0.3f;;
    leaf_prototype.petiole_roll = 0.04f;
    leaf_prototype.wave_period = 0.1f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.leaf_buckle_length.uniformDistribution(0.4,0.6);
    leaf_prototype.leaf_buckle_angle.uniformDistribution(40,50);
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
    phytomer_parameters_maize.petiole.pitch.uniformDistribution(-40, -20);
    phytomer_parameters_maize.petiole.radius = 0.0;
    phytomer_parameters_maize.petiole.length = 0.05;
    phytomer_parameters_maize.petiole.taper = 0;
    phytomer_parameters_maize.petiole.curvature = 0;
    phytomer_parameters_maize.petiole.length_segments = 1;

    phytomer_parameters_maize.leaf.leaves_per_petiole = 1;
    phytomer_parameters_maize.leaf.pitch = 0;
    phytomer_parameters_maize.leaf.yaw = 0;
    phytomer_parameters_maize.leaf.roll = 0;
    phytomer_parameters_maize.leaf.prototype_scale = 0.6;
    phytomer_parameters_maize.leaf.prototype = leaf_prototype;

    phytomer_parameters_maize.peduncle.length = 0.14f;
    phytomer_parameters_maize.peduncle.radius = 0.004;
    phytomer_parameters_maize.peduncle.curvature = 0;
    phytomer_parameters_maize.peduncle.color = phytomer_parameters_maize.internode.color;
    phytomer_parameters_maize.peduncle.radial_subdivisions = 6;
    phytomer_parameters_maize.peduncle.length_segments = 2;

    phytomer_parameters_maize.inflorescence.flowers_per_peduncle = 7;
    phytomer_parameters_maize.inflorescence.pitch.uniformDistribution(0,30);
    phytomer_parameters_maize.inflorescence.roll = 0;
    phytomer_parameters_maize.inflorescence.flower_offset = 0.1;
    phytomer_parameters_maize.inflorescence.fruit_prototype_scale = 0.15;
    phytomer_parameters_maize.inflorescence.fruit_prototype_function = MaizeTasselPrototype;

    phytomer_parameters_maize.phytomer_creation_function = MaizePhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_maize;
    shoot_parameters_mainstem.vegetative_bud_break_probability_min = 0.5;
    shoot_parameters_mainstem.flower_bud_break_probability = 1;
    shoot_parameters_mainstem.phyllochron_min = 2;
    shoot_parameters_mainstem.elongation_rate_max = 0.1;
    shoot_parameters_mainstem.girth_area_factor = 6.f;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-500,0);
    shoot_parameters_mainstem.internode_length_max = 0.22;
    shoot_parameters_mainstem.tortuosity = 1.f;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 1.0;
    shoot_parameters_mainstem.fruit_set_probability  = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"},{1.0});
    shoot_parameters_mainstem.max_nodes = 17;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem",shoot_parameters_mainstem);

}

uint PlantArchitecture::buildMaizePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize maize plant shoots
        initializeMaizeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.035f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.003, 0.08, 0.01, 0.01, 0.2, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 10, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeOliveTreeShoots(){

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.prototype_function = OliveLeafPrototype;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_olive(context_ptr->getRandomGenerator());

    phytomer_parameters_olive.internode.pitch = 0;
    phytomer_parameters_olive.internode.phyllotactic_angle.uniformDistribution(80, 100 );
    phytomer_parameters_olive.internode.radius_initial = 0.002;
    phytomer_parameters_olive.internode.length_segments = 1;
    phytomer_parameters_olive.internode.image_texture = "plugins/plantarchitecture/assets/textures/OliveBark.jpg";
    phytomer_parameters_olive.internode.max_floral_buds_per_petiole = 3;

    phytomer_parameters_olive.petiole.petioles_per_internode = 2;
    phytomer_parameters_olive.petiole.pitch.uniformDistribution(-40,-20);
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
    phytomer_parameters_olive.inflorescence.pitch.uniformDistribution(80,100);
    phytomer_parameters_olive.inflorescence.roll.uniformDistribution(0,360);
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
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"},{1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_olive;
//    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = OlivePhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = OlivePhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes.uniformDistribution(16,24);
    shoot_parameters_proleptic.max_nodes_per_season.uniformDistribution(8,12);
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.25;
    shoot_parameters_proleptic.girth_area_factor = 5.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.025;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 1.0;
    shoot_parameters_proleptic.vegetative_bud_break_time = 30;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(550,650);
    shoot_parameters_proleptic.tortuosity = 5;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution( 35, 40);
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
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"},{1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 30;
    shoot_parameters_scaffold.max_nodes_per_season = 10;
    shoot_parameters_scaffold.gravitropic_curvature = 700;
    shoot_parameters_scaffold.internode_length_max = 0.04;
    shoot_parameters_scaffold.tortuosity = 3;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"},{1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);

}

uint PlantArchitecture::buildOliveTree(const helios::vec3 &base_position) {

    if( shoot_types.empty() ){
        //automatically initialize olive tree shoots
        initializeOliveTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, 19, make_AxisRotation(context_ptr->randu(0.f, 0.025f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), shoot_types.at("trunk").phytomer_parameters.internode.radius_initial.val(), 0.01, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot( plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 4;//context_ptr->randu(4,5);

    for( int i=0; i<Nscaffolds; i++ ) {
        float pitch = context_ptr->randu(deg2rad(30), deg2rad(35));
        uint uID_shoot = addChildShoot( plantID, uID_trunk, getShootNodeCount(plantID,uID_trunk)-i-1, context_ptr->randu(5, 7), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007, shoot_types.at("scaffold").internode_length_max.val(), 1.f, 1.f, 0.5, "scaffold", 0);

    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200, 600, true);
    plant_instances.at(plantID).max_age = 1825;

    return plantID;

}

void PlantArchitecture::initializePistachioTreeShoots(){

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/PistachioLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.6f;
    leaf_prototype.midrib_fold_fraction = 0.;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.4, 0.4 );
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
    phytomer_parameters_pistachio.internode.image_texture = "plugins/plantarchitecture/assets/textures/OliveBark.jpg";
    phytomer_parameters_pistachio.internode.max_floral_buds_per_petiole = 3;

    phytomer_parameters_pistachio.petiole.petioles_per_internode = 2;
    phytomer_parameters_pistachio.petiole.pitch.uniformDistribution(-60, -45);
    phytomer_parameters_pistachio.petiole.taper = 0.1;
    phytomer_parameters_pistachio.petiole.curvature.uniformDistribution(-800,800);
    phytomer_parameters_pistachio.petiole.length = 0.075;
    phytomer_parameters_pistachio.petiole.radius = 0.001;
    phytomer_parameters_pistachio.petiole.length_segments = 1;
    phytomer_parameters_pistachio.petiole.radial_subdivisions = 3;
    phytomer_parameters_pistachio.petiole.color = make_RGBcolor(0.6, 0.6, 0.4);

    phytomer_parameters_pistachio.leaf.leaves_per_petiole = 3;
    phytomer_parameters_pistachio.leaf.prototype_scale = 0.08;
    phytomer_parameters_pistachio.leaf.leaflet_offset = 0.3;
    phytomer_parameters_pistachio.leaf.leaflet_scale = 0.75;
    phytomer_parameters_pistachio.leaf.pitch.uniformDistribution(-20,20);
    phytomer_parameters_pistachio.leaf.roll.uniformDistribution(-20,20);
    phytomer_parameters_pistachio.leaf.prototype = leaf_prototype;

    phytomer_parameters_pistachio.peduncle.length = 0.1;
    phytomer_parameters_pistachio.peduncle.radius = 0.001;
    phytomer_parameters_pistachio.peduncle.pitch = 60;
    phytomer_parameters_pistachio.peduncle.roll = 0;
    phytomer_parameters_pistachio.peduncle.length_segments = 1;
    phytomer_parameters_pistachio.peduncle.curvature.uniformDistribution(500,900);
    phytomer_parameters_pistachio.peduncle.color = make_RGBcolor(0.7, 0.72, 0.7);

    phytomer_parameters_pistachio.inflorescence.flowers_per_peduncle = 16;
    phytomer_parameters_pistachio.inflorescence.flower_offset = 0.08;
    phytomer_parameters_pistachio.inflorescence.pitch.uniformDistribution(50, 70);
    phytomer_parameters_pistachio.inflorescence.roll.uniformDistribution(0, 360);
    phytomer_parameters_pistachio.inflorescence.flower_prototype_scale = 0.025;
//    phytomer_parameters_pistachio.inflorescence.flower_prototype_function = PistachioFlowerPrototype;
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
    shoot_parameters_trunk.girth_area_factor = 3.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 1;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"proleptic"},{1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_pistachio;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = PistachioPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = PistachioPhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes.uniformDistribution(16,24);
    shoot_parameters_proleptic.max_nodes_per_season.uniformDistribution(8,12);
    shoot_parameters_proleptic.phyllochron_min = 2.0;
    shoot_parameters_proleptic.elongation_rate_max = 0.25;
    shoot_parameters_proleptic.girth_area_factor = 8.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.025;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.7;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 500;
    shoot_parameters_proleptic.tortuosity = 10;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution( 45, 55);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 10;
    shoot_parameters_proleptic.internode_length_max = 0.06;
    shoot_parameters_proleptic.internode_length_min = 0.03;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.005;
    shoot_parameters_proleptic.fruit_set_probability = 0.35;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.35;
    shoot_parameters_proleptic.max_terminal_floral_buds = 4;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);

}

uint PlantArchitecture::buildPistachioTree(const helios::vec3 &base_position) {

    if( shoot_types.empty() ){
        //automatically initialize pistachio tree shoots
        initializePistachioTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, 19, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), shoot_types.at("trunk").phytomer_parameters.internode.radius_initial.val(), 0.05, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot( plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 4;//context_ptr->randu(4,5);

    for( int i=0; i<Nscaffolds; i++ ) {
        float pitch = context_ptr->randu(deg2rad(65), deg2rad(80));
        uint uID_shoot = addChildShoot( plantID, uID_trunk, getShootNodeCount(plantID,uID_trunk)-i-1, 1, make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.1f, 0.1f)) / float(Nscaffolds) * 2 * M_PI, 0.5*M_PI), 0.007, 0.06, 1.f, 1.f, 0.5, "proleptic", 0);

    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200, false);
    plant_instances.at(plantID).max_age = 1095;

    return plantID;

}

void PlantArchitecture::initializePuncturevineShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/PuncturevineLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.4f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature = -0.1f;
    leaf_prototype.lateral_curvature = 0.4f;
    leaf_prototype.subdivisions = 1;
    leaf_prototype.unique_prototypes = 1;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_puncturevine(context_ptr->getRandomGenerator());

    phytomer_parameters_puncturevine.internode.pitch.uniformDistribution(0,15);
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
    phytomer_parameters_puncturevine.leaf.pitch.uniformDistribution(0,40);
    phytomer_parameters_puncturevine.leaf.yaw = 30;
    phytomer_parameters_puncturevine.leaf.roll.uniformDistribution(-5,5);
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
    shoot_parameters_primary.vegetative_bud_break_time = 3;
    shoot_parameters_primary.base_roll = 90;
    shoot_parameters_primary.phyllochron_min = 1;
    shoot_parameters_primary.elongation_rate_max = 0.2;
    shoot_parameters_primary.girth_area_factor = 0.f;
    shoot_parameters_primary.internode_length_max = 0.02;
    shoot_parameters_primary.internode_length_decay_rate = 0;
    shoot_parameters_primary.insertion_angle_tip.uniformDistribution(50, 80);
    shoot_parameters_primary.flowers_require_dormancy = false;
    shoot_parameters_primary.growth_requires_dormancy = false;
    shoot_parameters_primary.flower_bud_break_probability = 0.2;
    shoot_parameters_primary.determinate_shoot_growth = false;
    shoot_parameters_primary.max_nodes = 15;
    shoot_parameters_primary.gravitropic_curvature = 50;
    shoot_parameters_primary.tortuosity = 0;
    shoot_parameters_primary.defineChildShootTypes({"secondary_puncturevine"}, {1.f});

    ShootParameters shoot_parameters_base = shoot_parameters_primary;
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_puncturevine;
    shoot_parameters_base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5-10,137.5+10);
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
    shoot_parameters_base.max_nodes.uniformDistribution(3,5);
    shoot_parameters_base.defineChildShootTypes({"primary_puncturevine"},{1.f});

    ShootParameters shoot_parameters_children = shoot_parameters_primary;
    shoot_parameters_children.base_roll = 0;

    defineShootType("base_puncturevine", shoot_parameters_base);
    defineShootType("primary_puncturevine", shoot_parameters_primary);
    defineShootType("secondary_puncturevine", shoot_parameters_children);

}

uint PlantArchitecture::buildPuncturevinePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize puncturevine plant shoots
        initializePuncturevineShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0.f, 0.f), 0.001, 0.001, 1, 1, 0, "base_puncturevine");

    breakPlantDormancy(plantID);

    plant_instances.at(plantID).max_age = 45;

    setPlantPhenologicalThresholds(plantID, 0, -1, 14, -1, -1, 1000, false);

    return plantID;

}

void PlantArchitecture::initializeEasternRedbudShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/RedbudLeaf.png";
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
    phytomer_parameters_redbud.internode.phyllotactic_angle.uniformDistribution(170,190);
    phytomer_parameters_redbud.internode.radius_initial = 0.0015;
    phytomer_parameters_redbud.internode.image_texture = "plugins/plantarchitecture/assets/textures/WesternRedbudBark.jpg";
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
    phytomer_parameters_redbud.peduncle.pitch.uniformDistribution(50,90);
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
    shoot_parameters_trunk.defineChildShootTypes({"eastern_redbud_shoot"},{1.f});

    defineShootType("eastern_redbud_trunk", shoot_parameters_trunk);
    defineShootType("eastern_redbud_shoot", shoot_parameters_main);

}

uint PlantArchitecture::buildEasternRedbudPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize redbud plant shoots
        initializeEasternRedbudShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 16, make_AxisRotation(context_ptr->randu(0,0.1*M_PI), context_ptr->randu(0,2*M_PI), context_ptr->randu(0,2*M_PI)), 0.0075, 0.05, 1, 1, 0.4, "eastern_redbud_trunk");

    makePlantDormant(plantID);
    breakPlantDormancy(plantID);

    //leave four vegetative buds on the trunk and remove the rest
    for (auto &phytomer: this->plant_instances.at(plantID).shoot_tree.at(uID_stem)->phytomers ) {
        if ( phytomer->shoot_index.x < 12 ) {
            for (auto &petiole: phytomer->axillary_vegetative_buds) {
                for (auto &vbud: petiole) {
                    phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                }
            }
        }
    }

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200, false);

    plant_instances.at(plantID).max_age = 1460;

    return plantID;

}

void PlantArchitecture::initializeRiceShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";
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
    phytomer_parameters_rice.peduncle.length.uniformDistribution(0.14,0.18);
    phytomer_parameters_rice.peduncle.radius = 0.0005;
    phytomer_parameters_rice.peduncle.color = phytomer_parameters_rice.internode.color;
    phytomer_parameters_rice.peduncle.curvature.uniformDistribution(-800,-50);
    phytomer_parameters_rice.peduncle.radial_subdivisions = 6;
    phytomer_parameters_rice.peduncle.length_segments = 8;

    phytomer_parameters_rice.inflorescence.flowers_per_peduncle = 60;
    phytomer_parameters_rice.inflorescence.pitch.uniformDistribution(20,25);
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
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-1000,-400);
    shoot_parameters_mainstem.internode_length_max = 0.0075;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.fruit_set_probability  = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"},{1.0});
    shoot_parameters_mainstem.max_nodes = 30;
    shoot_parameters_mainstem.max_terminal_floral_buds = 5;

    defineShootType("mainstem",shoot_parameters_mainstem);

}

uint PlantArchitecture::buildRicePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize rice plant shoots
        initializeRiceShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.1f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.001, 0.0075, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 10, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeButterLettuceShoots() {

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/RomaineLettuceLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.85f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.2, 0.05);
    leaf_prototype.lateral_curvature = -0.4f;
    leaf_prototype.wave_period.uniformDistribution(0.15, 0.25);
    leaf_prototype.wave_amplitude.uniformDistribution(0.05,0.1);
    leaf_prototype.subdivisions = 30;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 137.5;
    phytomer_parameters.internode.radius_initial = 0.02;
    phytomer_parameters.internode.color = make_RGBcolor(0.402,0.423,0.413);
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.internode.radial_subdivisions = 10;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(0,30);
    phytomer_parameters.petiole.radius = 0.001;
    phytomer_parameters.petiole.length = 0.001;
    phytomer_parameters.petiole.length_segments = 1;
    phytomer_parameters.petiole.radial_subdivisions = 3;
    phytomer_parameters.petiole.color = RGB::red;

    phytomer_parameters.leaf.leaves_per_petiole = 1;
    phytomer_parameters.leaf.pitch = 10;
    phytomer_parameters.leaf.yaw = 0;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.15,0.25);
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

    defineShootType("mainstem",shoot_parameters_mainstem);


}

uint PlantArchitecture::buildButterLettucePlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize lettuce plant shoots
        initializeButterLettuceShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(context_ptr->randu(0.f, 0.03f * M_PI), 0.f, context_ptr->randu(0.f, 2.f * M_PI)), 0.005, 0.001, 1, 1, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeSorghumShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.2f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.4, -0.2);
    leaf_prototype.lateral_curvature = -0.3f;
    leaf_prototype.petiole_roll = 0.04f;
    leaf_prototype.wave_period = 0.1f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.leaf_buckle_length.uniformDistribution(0.4,0.6);
    leaf_prototype.leaf_buckle_angle.uniformDistribution(45,55);
    leaf_prototype.subdivisions = 50;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_sorghum(context_ptr->getRandomGenerator());

    phytomer_parameters_sorghum.internode.pitch = 0;
    phytomer_parameters_sorghum.internode.phyllotactic_angle.uniformDistribution(170,190);
    phytomer_parameters_sorghum.internode.radius_initial = 0.003;
    phytomer_parameters_sorghum.internode.color = make_RGBcolor(0.09,0.13,0.06);
    phytomer_parameters_sorghum.internode.length_segments = 2;
    phytomer_parameters_sorghum.internode.radial_subdivisions = 10;
    phytomer_parameters_sorghum.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_sorghum.internode.max_vegetative_buds_per_petiole = 0;

    phytomer_parameters_sorghum.petiole.petioles_per_internode = 1;
    phytomer_parameters_sorghum.petiole.pitch.uniformDistribution(-40,-20);
    phytomer_parameters_sorghum.petiole.radius = 0.0;
    phytomer_parameters_sorghum.petiole.length = 0.05;
    phytomer_parameters_sorghum.petiole.taper = 0;
    phytomer_parameters_sorghum.petiole.curvature = 0;
    phytomer_parameters_sorghum.petiole.length_segments = 1;

    phytomer_parameters_sorghum.leaf.leaves_per_petiole = 1;
    phytomer_parameters_sorghum.leaf.pitch = 0;
    phytomer_parameters_sorghum.leaf.yaw = 0;
    phytomer_parameters_sorghum.leaf.roll = 0;
    phytomer_parameters_sorghum.leaf.prototype_scale = 0.6;
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
    shoot_parameters_mainstem.phyllochron_min = 2;
    shoot_parameters_mainstem.elongation_rate_max = 0.1;
    shoot_parameters_mainstem.girth_area_factor = 5.f;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-1000,-400);
    shoot_parameters_mainstem.internode_length_max = 0.26;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 1.0;
    shoot_parameters_mainstem.fruit_set_probability  = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"},{1.0});
    shoot_parameters_mainstem.max_nodes = 16;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem",shoot_parameters_mainstem);

}

uint PlantArchitecture::buildSorghumPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize sorghum plant shoots
        initializeSorghumShoots();
    }

    uint plantID = addPlantInstance(base_position - make_vec3(0,0,0.025), 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.075f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.003, 0.06, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 15, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeSoybeanShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/SoybeanLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(0.1,0.2);
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
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.2,0.25,0.05);
    phytomer_parameters_trifoliate.internode.length_segments = 5;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(15,40);
    phytomer_parameters_trifoliate.petiole.radius = 0.002;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.12,0.16);
    phytomer_parameters_trifoliate.petiole.taper = 0.25;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-250,50);
    phytomer_parameters_trifoliate.petiole.color = phytomer_parameters_trifoliate.internode.color;
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.uniformDistribution(-30, 10);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll.uniformDistribution(-25,5);
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.5;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.1,0.14);
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
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50,70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20,20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.015;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = SoybeanFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.1,0.12);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = SoybeanFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8,1.0);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.01;
    phytomer_parameters_unifoliate.petiole.radius = 0.001;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(60,80);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.02;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype = leaf_prototype;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 25;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(20,30);
//    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.035;
//    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
//    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20,20);
    shoot_parameters_trifoliate.gravitropic_curvature = 400;

    shoot_parameters_trifoliate.phyllochron_min = 2;
    shoot_parameters_trifoliate.elongation_rate_max = 0.1;
    shoot_parameters_trifoliate.girth_area_factor = 2.f;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 15;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_min = 0.05;
    shoot_parameters_trifoliate.vegetative_bud_break_probability_decay_rate = 0.6;
//    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.8,1.0);
    shoot_parameters_trifoliate.fruit_set_probability = 0.4;
//    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
//    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
//    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)

    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"},{1.0});


    ShootParameters shoot_parameters_unifoliate = shoot_parameters_trifoliate;
    shoot_parameters_unifoliate.phytomer_parameters = phytomer_parameters_unifoliate;
    shoot_parameters_unifoliate.max_nodes = 1;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 0;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 8;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"},{1.0});

    defineShootType("unifoliate",shoot_parameters_unifoliate);
    defineShootType("trifoliate",shoot_parameters_trifoliate);

}

uint PlantArchitecture::buildSoybeanPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize bean plant shoots
        initializeSoybeanShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.04, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").phytomer_parameters.internode.radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeStrawberryShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/StrawberryLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 1.f;
    leaf_prototype.midrib_fold_fraction = 0.2f;
    leaf_prototype.longitudinal_curvature = 0.15f;
    leaf_prototype.lateral_curvature = 0.4f;
    leaf_prototype.wave_period = 0.3f;
    leaf_prototype.wave_amplitude = 0.01f;
    leaf_prototype.subdivisions = 6;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(80,100);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.15, 0.2, 0.1);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(0,45);
    phytomer_parameters.petiole.radius = 0.0025;
    phytomer_parameters.petiole.length.uniformDistribution(0.15,0.25);
    phytomer_parameters.petiole.taper = 0.5;
    phytomer_parameters.petiole.curvature.uniformDistribution(-300,100);
    phytomer_parameters.petiole.color = make_RGBcolor(0.18, 0.23, 0.1);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 3;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30,10);
    phytomer_parameters.leaf.yaw = 20;
    phytomer_parameters.leaf.roll = -30;
    phytomer_parameters.leaf.leaflet_offset = 0.01;
    phytomer_parameters.leaf.leaflet_scale = 1.0;
    phytomer_parameters.leaf.prototype_scale = 0.1;
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.17;
    phytomer_parameters.peduncle.radius = 0.00075;
    phytomer_parameters.peduncle.pitch = 35;
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -200;
    phytomer_parameters.peduncle.length_segments = 5;
    phytomer_parameters.peduncle.radial_subdivisions = 6;
    phytomer_parameters.peduncle.color = phytomer_parameters.petiole.color;

    phytomer_parameters.inflorescence.flowers_per_peduncle.uniformDistribution(1, 3);
    phytomer_parameters.inflorescence.flower_offset = 0.2;
    phytomer_parameters.inflorescence.pitch = 70;
    phytomer_parameters.inflorescence.roll = 90;
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.04;
    phytomer_parameters.inflorescence.flower_prototype_function = StrawberryFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.06;
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
    shoot_parameters.base_yaw.uniformDistribution(-20,20);
    shoot_parameters.gravitropic_curvature.uniformDistribution(-10,0);
    shoot_parameters.tortuosity = 0;

    shoot_parameters.phyllochron_min = 2;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 2.f;
    shoot_parameters.vegetative_bud_break_time = 15;
    shoot_parameters.vegetative_bud_break_probability_min = 0.1;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = -0.4;
    shoot_parameters.flower_bud_break_probability = 1;
    shoot_parameters.fruit_set_probability = 0.5;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = true;

    shoot_parameters.defineChildShootTypes({"mainstem"},{1.0});

    defineShootType("mainstem",shoot_parameters);

}

uint PlantArchitecture::buildStrawberryPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize strawberry plant shoots
        initializeStrawberryShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.001, 0.004, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000, false);

    plant_instances.at(plantID).max_age = 120;

    return plantID;

}

void PlantArchitecture::initializeSugarbeetShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/SugarbeetLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.4f;
    leaf_prototype.midrib_fold_fraction = 0.1f;
    leaf_prototype.longitudinal_curvature = -0.2f;
    leaf_prototype.lateral_curvature = -0.4f;
    leaf_prototype.petiole_roll = 0.75f;
    leaf_prototype.wave_period.uniformDistribution( 0.08f, 0.15f);
    leaf_prototype.wave_amplitude.uniformDistribution(0.02, 0.04);
    leaf_prototype.subdivisions = 20;;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_sugarbeet(context_ptr->getRandomGenerator());

    phytomer_parameters_sugarbeet.internode.pitch = 0;
    phytomer_parameters_sugarbeet.internode.phyllotactic_angle = 137.5;
    phytomer_parameters_sugarbeet.internode.radius_initial = 0.005;
    phytomer_parameters_sugarbeet.internode.color = make_RGBcolor(0.44,0.58,0.19);
    phytomer_parameters_sugarbeet.internode.length_segments = 1;
    phytomer_parameters_sugarbeet.internode.max_vegetative_buds_per_petiole = 0;
    phytomer_parameters_sugarbeet.internode.max_floral_buds_per_petiole = 0;

    phytomer_parameters_sugarbeet.petiole.petioles_per_internode = 1;
    phytomer_parameters_sugarbeet.petiole.pitch.uniformDistribution(0,40);
    phytomer_parameters_sugarbeet.petiole.radius = 0.005;
    phytomer_parameters_sugarbeet.petiole.length.uniformDistribution(0.15,0.2);
    phytomer_parameters_sugarbeet.petiole.taper = 0.6;
    phytomer_parameters_sugarbeet.petiole.curvature.uniformDistribution(-300,100);
    phytomer_parameters_sugarbeet.petiole.color = phytomer_parameters_sugarbeet.internode.color;
    phytomer_parameters_sugarbeet.petiole.length_segments = 8;

    phytomer_parameters_sugarbeet.leaf.leaves_per_petiole = 1;
    phytomer_parameters_sugarbeet.leaf.pitch.uniformDistribution(-10,0);
    phytomer_parameters_sugarbeet.leaf.yaw.uniformDistribution(-5,5);
    phytomer_parameters_sugarbeet.leaf.roll.uniformDistribution(-15,15);
    phytomer_parameters_sugarbeet.leaf.prototype_scale.uniformDistribution(0.15,0.25);
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

    defineShootType("mainstem",shoot_parameters_mainstem);


}

uint PlantArchitecture::buildSugarbeetPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize sugarbeet plant shoots
        initializeSugarbeetShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(context_ptr->randu(0.f, 0.01f * M_PI), 0.f * context_ptr->randu(0.f, 2.f * M_PI), 0.25f * M_PI), 0.005, 0.001, 1, 1, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeTomatoShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/TomatoLeaf_centered.png";
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
    phytomer_parameters.internode.color = make_RGBcolor(0.217,0.275, 0.0571);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45,60);
    phytomer_parameters.petiole.radius = 0.002;
    phytomer_parameters.petiole.length = 0.2;
    phytomer_parameters.petiole.taper = 0.15;
    phytomer_parameters.petiole.curvature.uniformDistribution(-150,-50);
    phytomer_parameters.petiole.color = phytomer_parameters.internode.color;
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 7;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.leaflet_offset = 0.15;
    phytomer_parameters.leaf.leaflet_scale = 0.7;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12,0.18);
    phytomer_parameters.leaf.prototype = leaf_prototype;

    phytomer_parameters.peduncle.length = 0.16;
    phytomer_parameters.peduncle.radius = 0.0015;
    phytomer_parameters.peduncle.pitch = 20;
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -700;
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 5;
    phytomer_parameters.peduncle.radial_subdivisions = 8;

    phytomer_parameters.inflorescence.flowers_per_peduncle = 8;
    phytomer_parameters.inflorescence.flower_offset = 0.15;
    phytomer_parameters.inflorescence.pitch = 90;
    phytomer_parameters.inflorescence.roll.uniformDistribution(-30,30);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters.inflorescence.flower_prototype_function = TomatoFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.08;
    phytomer_parameters.inflorescence.fruit_prototype_function = TomatoFruitPrototype;
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction = 0.5;

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
    shoot_parameters.base_yaw.uniformDistribution(-20,20);
    shoot_parameters.gravitropic_curvature = 150;
    shoot_parameters.tortuosity = 3;

    shoot_parameters.phyllochron_min = 2;
    shoot_parameters.elongation_rate_max = 0.1;
    shoot_parameters.girth_area_factor = 2.5f;
    shoot_parameters.vegetative_bud_break_time = 30;
    shoot_parameters.vegetative_bud_break_probability_min = 0.25;
    shoot_parameters.vegetative_bud_break_probability_decay_rate = -0.25;
    shoot_parameters.flower_bud_break_probability = 0.2;
    shoot_parameters.fruit_set_probability = 0.8;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = true;

    shoot_parameters.defineChildShootTypes({"mainstem"},{1.0});

    defineShootType("mainstem",shoot_parameters);

}

uint PlantArchitecture::buildTomatoPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize tomato plant shoots
        initializeTomatoShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.002, 0.04, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}

void PlantArchitecture::initializeCherryTomatoShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/CherryTomatoLeaf.png";
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

    phytomer_parameters.internode.pitch = 10;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(14, 220);
    phytomer_parameters.internode.radius_initial = 0.001;
    phytomer_parameters.internode.color = make_RGBcolor(0.217,0.275, 0.0571);
    phytomer_parameters.internode.length_segments = 1;
    phytomer_parameters.internode.radial_subdivisions = 14;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45,60);
    phytomer_parameters.petiole.radius = 0.002;
    phytomer_parameters.petiole.length = 0.25;
    phytomer_parameters.petiole.taper = 0.25;
    phytomer_parameters.petiole.curvature.uniformDistribution(-250,0);
    phytomer_parameters.petiole.color = make_RGBcolor(0.32,0.37,0.12);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 9;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30, 5);
    phytomer_parameters.leaf.yaw = 10;
    phytomer_parameters.leaf.roll.uniformDistribution(-20,20);
    phytomer_parameters.leaf.leaflet_offset = 0.22;
    phytomer_parameters.leaf.leaflet_scale = 0.9;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.12,0.17);
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
    phytomer_parameters.inflorescence.roll.uniformDistribution(-10,10);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters.inflorescence.flower_prototype_function = TomatoFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.09;
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
    shoot_parameters.base_yaw.uniformDistribution(-20,20);
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

    shoot_parameters.defineChildShootTypes({"mainstem"},{1.0});

    defineShootType("mainstem",shoot_parameters);

}

uint PlantArchitecture::buildCherryTomatoPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize cherry tomato plant shoots
        initializeCherryTomatoShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.002, 0.06, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 50, 10, 5, 30, 1000, false);

    plant_instances.at(plantID).max_age = 175;

    return plantID;

}

void PlantArchitecture::initializeWalnutTreeShoots(){

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/WalnutLeaf.png";
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
    phytomer_parameters_walnut.internode.phyllotactic_angle.uniformDistribution( 160, 200 );
    phytomer_parameters_walnut.internode.radius_initial = 0.004;
    phytomer_parameters_walnut.internode.length_segments = 1;
    phytomer_parameters_walnut.internode.image_texture = "plugins/plantarchitecture/assets/textures/AppleBark.jpg";
    phytomer_parameters_walnut.internode.max_floral_buds_per_petiole = 3;

    phytomer_parameters_walnut.petiole.petioles_per_internode = 2;
    phytomer_parameters_walnut.petiole.pitch.uniformDistribution(-80,-70);
    phytomer_parameters_walnut.petiole.taper = 0.2;
    phytomer_parameters_walnut.petiole.curvature.uniformDistribution(-1000,0);
    phytomer_parameters_walnut.petiole.length = 0.15;
    phytomer_parameters_walnut.petiole.radius = 0.0015;
    phytomer_parameters_walnut.petiole.length_segments = 5;
    phytomer_parameters_walnut.petiole.radial_subdivisions = 3;
    phytomer_parameters_walnut.petiole.color = make_RGBcolor(0.61,0.5,0.24);

    phytomer_parameters_walnut.leaf.leaves_per_petiole = 5;
    phytomer_parameters_walnut.leaf.pitch.uniformDistribution(-40,0);
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
    shoot_parameters_trunk.girth_area_factor = 3.f;
    shoot_parameters_trunk.vegetative_bud_break_probability_min = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 1;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"},{1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_walnut;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3,0.2,0.2);
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = WalnutPhytomerCreationFunction;
    // shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = WalnutPhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes = 30;
    shoot_parameters_proleptic.max_nodes_per_season = 15;
    shoot_parameters_proleptic.phyllochron_min = 2.;
    shoot_parameters_proleptic.elongation_rate_max = 0.15;
    shoot_parameters_proleptic.girth_area_factor = 10.f;
    shoot_parameters_proleptic.vegetative_bud_break_probability_min = 0.05;
    shoot_parameters_proleptic.vegetative_bud_break_probability_decay_rate = 0.6;
    shoot_parameters_proleptic.vegetative_bud_break_time = 3;
    shoot_parameters_proleptic.gravitropic_curvature = 200;
    shoot_parameters_proleptic.tortuosity = 5;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution( 20, 25);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 15;
    shoot_parameters_proleptic.internode_length_max = 0.08;
    shoot_parameters_proleptic.internode_length_min = 0.01;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.006;
    shoot_parameters_proleptic.fruit_set_probability = 0.5;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 4;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.determinate_shoot_growth = false;
    shoot_parameters_proleptic.defineChildShootTypes({"proleptic"},{1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 30;
    shoot_parameters_scaffold.gravitropic_curvature = 300;
    shoot_parameters_scaffold.internode_length_max = 0.06;
    shoot_parameters_scaffold.tortuosity = 4;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"},{1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);

}

uint PlantArchitecture::buildWalnutTree(const helios::vec3 &base_position) {

    if( shoot_types.empty() ){
        //automatically initialize walnut tree shoots
        initializeWalnutTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, 0);

    uint uID_trunk = addBaseStemShoot(plantID, 19, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), shoot_types.at("trunk").phytomer_parameters.internode.radius_initial.val(), 0.04, 1.f, 1.f, 0, "trunk");
    appendPhytomerToShoot( plantID, uID_trunk, shoot_types.at("trunk").phytomer_parameters, 0, 0.01, 1, 1);

    plant_instances.at(plantID).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plantID).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    uint Nscaffolds = 4;//context_ptr->randu(4,5);

    for( int i=0; i<Nscaffolds; i++ ) {
//        float pitch = context_ptr->randu(deg2rad(25), deg2rad(35))+i*deg2rad(7.f);
        float pitch = context_ptr->randu(deg2rad(45), deg2rad(55));
        uint uID_shoot = addChildShoot( plantID, uID_trunk, getShootNodeCount(plantID,uID_trunk)-i-1, context_ptr->randu(7, 9), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007, 0.06, 1.f, 1.f, 0.5, "scaffold", 0);

    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200, false);

    plant_instances.at(plantID).max_age = 1095;

    return plantID;

}

void PlantArchitecture::initializeWheatShoots() {

    // ---- Leaf Prototype ---- //

    LeafPrototype leaf_prototype(context_ptr->getRandomGenerator());
    leaf_prototype.leaf_texture_file[0] = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";
    leaf_prototype.leaf_aspect_ratio = 0.1f;
    leaf_prototype.midrib_fold_fraction = 0.3f;
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.5, -0.1);
    leaf_prototype.lateral_curvature = -0.3;
    leaf_prototype.petiole_roll = 0.04f;
    leaf_prototype.wave_period = 0.1f;
    leaf_prototype.wave_amplitude = 0.1f;
    leaf_prototype.leaf_buckle_length.uniformDistribution(0.5,0.6);
    leaf_prototype.leaf_buckle_angle.uniformDistribution(25,35);
    leaf_prototype.subdivisions = 20;
    leaf_prototype.unique_prototypes = 10;

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_wheat(context_ptr->getRandomGenerator());

    phytomer_parameters_wheat.internode.pitch = 0;
    phytomer_parameters_wheat.internode.phyllotactic_angle.uniformDistribution(67,77);
    phytomer_parameters_wheat.internode.radius_initial = 0.001;
    phytomer_parameters_wheat.internode.color = make_RGBcolor(0.27,0.31,0.16);
    phytomer_parameters_wheat.internode.length_segments = 1;
    phytomer_parameters_wheat.internode.radial_subdivisions = 6;
    phytomer_parameters_wheat.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_wheat.internode.max_vegetative_buds_per_petiole = 0;

    phytomer_parameters_wheat.petiole.petioles_per_internode = 1;
    phytomer_parameters_wheat.petiole.pitch.uniformDistribution(-40,-20);
    phytomer_parameters_wheat.petiole.radius = 0.0;
    phytomer_parameters_wheat.petiole.length = 0.005;
    phytomer_parameters_wheat.petiole.taper = 0;
    phytomer_parameters_wheat.petiole.curvature = 0;
    phytomer_parameters_wheat.petiole.length_segments = 1;

    phytomer_parameters_wheat.leaf.leaves_per_petiole = 1;
    phytomer_parameters_wheat.leaf.pitch = 0;
    phytomer_parameters_wheat.leaf.yaw = 0;
    phytomer_parameters_wheat.leaf.roll = 0;
    phytomer_parameters_wheat.leaf.prototype_scale = 0.22;
    phytomer_parameters_wheat.leaf.prototype = leaf_prototype;

    phytomer_parameters_wheat.peduncle.length = 0.1;
    phytomer_parameters_wheat.peduncle.radius = 0.002;
    phytomer_parameters_wheat.peduncle.color = phytomer_parameters_wheat.internode.color;
    phytomer_parameters_wheat.peduncle.curvature = -100;
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
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-500,-200);
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.fruit_set_probability  = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"},{1.0});
    shoot_parameters_mainstem.max_nodes = 20;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem",shoot_parameters_mainstem);

}

uint PlantArchitecture::buildWheatPlant(const helios::vec3 &base_position) {

    if (shoot_types.empty()) {
        //automatically initialize wheat plant shoots
        initializeWheatShoots();
    }

    uint plantID = addPlantInstance(base_position - make_vec3(0,0,0.025), 0);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.001, 0.025, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 10, 1000, false);

    plant_instances.at(plantID).max_age = 365;

    return plantID;

}