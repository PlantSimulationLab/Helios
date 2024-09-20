/** \file "PlantLibrary.cpp" Contains routines for loading and building plant models from a library of predefined plant types.

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
        plantID = buildAlmondTree(base_position, age);
    }else if( current_plant_model == "asparagus" ) {
        plantID = buildAsparagusPlant(base_position, age);
    }else if( current_plant_model == "bindweed" ) {
        plantID = buildBindweedPlant(base_position, age);
    }else if( current_plant_model == "bean" ) {
        plantID = buildBeanPlant(base_position, age);
    }else if( current_plant_model == "cheeseweed" ) {
        plantID = buildCheeseweedPlant(base_position, age);
    }else if( current_plant_model == "cowpea" ) {
        plantID = buildCowpeaPlant(base_position, age);
    }else if( current_plant_model == "puncturevine" ) {
        plantID = buildPuncturevinePlant(base_position, age);
    }else if( current_plant_model == "easternredbud" ) {
        plantID = buildEasternRedbudPlant(base_position, age);
    }else if( current_plant_model == "romainelettuce" ) {
        plantID = buildRomaineLettucePlant(base_position, age);
    }else if( current_plant_model == "sorghum" ) {
        plantID = buildSorghumPlant(base_position, age);
    }else if( current_plant_model == "soybean" ) {
        plantID = buildSoybeanPlant(base_position, age);
    }else if( current_plant_model == "strawberry" ) {
        plantID = buildStrawberryPlant(base_position, age);
    }else if( current_plant_model == "sugarbeet" ) {
        plantID = buildSugarbeetPlant(base_position, age);
    }else if( current_plant_model == "tomato" ) {
        plantID = buildTomatoPlant(base_position, age);
    }else if( current_plant_model == "wheat" ) {
        plantID = buildWheatPlant(base_position, age);
    }else{
        assert(true); //shouldn't be here
    }

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

void PlantArchitecture::updateCurrentShootParameters( const std::string &shoot_type_label, const ShootParameters &params ){
    shoot_types[shoot_type_label] = params;
}

void PlantArchitecture::updateCurrentShootParameters( const std::map<std::string, ShootParameters> &params ){
    shoot_types = params;
}

void PlantArchitecture::initializeDefaultShoots( const std::string &plant_label ){

    if( plant_label == "almond" ) {
        initializeAlmondTreeShoots();
    }else if( plant_label == "asparagus" ) {
        initializeAsparagusShoots();
    }else if( plant_label == "bindweed" ) {
        initializeBindweedShoots();
    }else if( plant_label == "bean" ) {
        initializeBeanShoots();
    }else if( plant_label == "cheeseweed" ) {
        initializeCheeseweedShoots();
    }else if( plant_label == "cowpea" ) {
        initializeCowpeaShoots();
    }else if( plant_label == "puncturevine" ) {
        initializePuncturevineShoots();
    }else if( plant_label == "easternredbud" ) {
        initializeEasternRedbudShoots();
    }else if( plant_label == "romainelettuce" ) {
        initializeRomaineLettuceShoots();
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
    }else if( plant_label == "wheat" ) {
        initializeWheatShoots();
    }else{
        helios_runtime_error("ERROR (PlantArchitecture::loadPlantModelFromLibrary): plant label of " + plant_label + " does not exist in the library.");
    }

}

void PlantArchitecture::initializeAlmondTreeShoots(){

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_almond(context_ptr->getRandomGenerator());

    phytomer_parameters_almond.internode.pitch = 0;
    phytomer_parameters_almond.internode.phyllotactic_angle.uniformDistribution( 130, 145 );
    phytomer_parameters_almond.internode.length_segments = 1;
    phytomer_parameters_almond.internode.image_texture = "plugins/plantarchitecture/assets/textures/AlmondBark.jpg";
    phytomer_parameters_almond.internode.max_floral_buds_per_petiole = 3;

    phytomer_parameters_almond.petiole.petioles_per_internode = 1;
    phytomer_parameters_almond.petiole.pitch = -50;
    phytomer_parameters_almond.petiole.taper = 0.1;
    phytomer_parameters_almond.petiole.curvature = 0;
    phytomer_parameters_almond.petiole.length = 0.04;
    phytomer_parameters_almond.petiole.radius = 0.0005;
    phytomer_parameters_almond.petiole.length_segments = 1;
    phytomer_parameters_almond.petiole.radial_subdivisions = 3;
    phytomer_parameters_almond.petiole.color = make_RGBcolor(0.61,0.5,0.24);

    phytomer_parameters_almond.leaf.leaves_per_petiole = 1;
    phytomer_parameters_almond.leaf.prototype_function = AlmondLeafPrototype;
    phytomer_parameters_almond.leaf.prototype_scale = 0.09;
    phytomer_parameters_almond.leaf.unique_prototypes = 1;

    phytomer_parameters_almond.peduncle.length = 0.005;
    phytomer_parameters_almond.peduncle.radius = 0.0005;
    phytomer_parameters_almond.peduncle.pitch = 90;
    phytomer_parameters_almond.peduncle.roll = 90;
    phytomer_parameters_almond.peduncle.length_segments = 1;

    phytomer_parameters_almond.inflorescence.flowers_per_rachis = 1;
    phytomer_parameters_almond.inflorescence.pitch = 0;
    phytomer_parameters_almond.inflorescence.roll = 0;
    phytomer_parameters_almond.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters_almond.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
    phytomer_parameters_almond.inflorescence.fruit_prototype_scale = 0.05;
    phytomer_parameters_almond.inflorescence.fruit_prototype_function = AlmondFruitPrototype;

    // ---- Shoot Parameters ---- //

    // Trunk
    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_trunk.phytomer_parameters.internode.phyllotactic_angle = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 14;
    shoot_parameters_trunk.max_nodes = 20;
    shoot_parameters_trunk.girth_growth_rate = 0.00075;
    shoot_parameters_trunk.internode_radius_initial = 0.01;
    shoot_parameters_trunk.vegetative_bud_break_probability = 0;
    shoot_parameters_trunk.vegetative_bud_break_time = 0;
    shoot_parameters_trunk.tortuosity = 6;
    shoot_parameters_trunk.internode_length_max = 0.05;
    shoot_parameters_trunk.internode_length_decay_rate = 0;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"},{1});

    // Proleptic shoots
    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_almond;
    shoot_parameters_proleptic.phytomer_parameters.internode.color = make_RGBcolor(0.3,0.2,0.2);
    shoot_parameters_proleptic.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction;
    shoot_parameters_proleptic.phytomer_parameters.phytomer_callback_function = AlmondPhytomerCallbackFunction;
    shoot_parameters_proleptic.max_nodes = 40;
    shoot_parameters_proleptic.phyllochron = 0.9;
    shoot_parameters_proleptic.elongation_rate = 0.25;
    shoot_parameters_proleptic.girth_growth_rate = 0.00045;
    shoot_parameters_proleptic.vegetative_bud_break_probability = 0.15;
    shoot_parameters_proleptic.vegetative_bud_break_time = 0;
    shoot_parameters_proleptic.leaf_flush_count = 1;
    shoot_parameters_proleptic.gravitropic_curvature = 300;
    shoot_parameters_proleptic.tortuosity = 15;
    shoot_parameters_proleptic.internode_radius_initial = 0.004;
    shoot_parameters_proleptic.insertion_angle_tip.uniformDistribution( 20, 25);
    shoot_parameters_proleptic.insertion_angle_decay_rate = 15;
    shoot_parameters_proleptic.internode_length_max = 0.04;
    shoot_parameters_proleptic.internode_length_min = 0.003;
    shoot_parameters_proleptic.internode_length_decay_rate = 0.004;
    shoot_parameters_proleptic.fruit_set_probability = 0.5;
    shoot_parameters_proleptic.flower_bud_break_probability = 0.3;
    shoot_parameters_proleptic.max_terminal_floral_buds = 4;
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
    shoot_parameters_sylleptic.phyllochron = 0.9;
    shoot_parameters_sylleptic.vegetative_bud_break_probability = 0.1;
    shoot_parameters_sylleptic.gravitropic_curvature= 600;
    shoot_parameters_sylleptic.internode_length_max = 0.06;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true;
    shoot_parameters_sylleptic.defineChildShootTypes({"proleptic"},{1.0});

    // Main scaffolds
    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
//    shoot_parameters_scaffold.phytomer_parameters.internode.color = RGB::blue;
    shoot_parameters_scaffold.phytomer_parameters.internode.radial_subdivisions = 10;
    shoot_parameters_scaffold.max_nodes = 40;
    shoot_parameters_scaffold.gravitropic_curvature = 300;
    shoot_parameters_scaffold.internode_length_max = 0.04;
    shoot_parameters_scaffold.tortuosity = 7;
    shoot_parameters_scaffold.girth_growth_rate = 0.0005;
    shoot_parameters_scaffold.defineChildShootTypes({"proleptic"},{1.0});

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);

}

uint PlantArchitecture::buildAlmondTree(const helios::vec3 &base_position, float age) {

    if( shoot_types.empty() ){
        //automatically initialize almond tree shoots
        initializeAlmondTreeShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

//    enableEpicormicChildShoots(plantID,"sylleptic",0.001);

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
//        float pitch = context_ptr->randu(deg2rad(25), deg2rad(35))+i*deg2rad(7.f);
        float pitch = context_ptr->randu(deg2rad(45), deg2rad(55));
        uint uID_shoot = addChildShoot( plantID, uID_trunk, getShootNodeCount(plantID,uID_trunk)-i-1, context_ptr->randu(7, 9), make_AxisRotation(pitch, (float(i) + context_ptr->randu(-0.2f, 0.2f)) / float(Nscaffolds) * 2 * M_PI, 0), 0.007, 0.06, 1.f, 1.f, 0.5, "scaffold", 0);

//        plant_instances.at(plantID).shoot_tree.at(uID_shoot)->breakDormancy();

//        uint blind_nodes = context_ptr->randu(4,5);
//        for( int b=0; b<blind_nodes; b++){
//            if( b<plant_instances.at(plantID).shoot_tree.at(uID_shoot)->phytomers.size() ) {
//                plant_instances.at(plantID).shoot_tree.at(uID_shoot)->phytomers.at(b)->removeLeaf();
//                plant_instances.at(plantID).shoot_tree.at(uID_shoot)->phytomers.at(b)->setFloralBudState(BUD_DEAD);
//                plant_instances.at(plantID).shoot_tree.at(uID_shoot)->phytomers.at(b)->setVegetativeBudState(BUD_DEAD);
//            }
//        }

    }

    makePlantDormant(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 1, 3, 8, 20);

    return plantID;

}

void PlantArchitecture::initializeAsparagusShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 1;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(127.5, 147.5);
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
    phytomer_parameters.leaf.prototype_function = AsparagusLeafPrototype;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.018, 0.02);
//    phytomer_parameters.leaf.subdivisions = 6;

    phytomer_parameters.peduncle.length = 0.17;
    phytomer_parameters.peduncle.radius = 0.0015;
    phytomer_parameters.peduncle.pitch.uniformDistribution(0, 30);
    phytomer_parameters.peduncle.roll = 90;
    phytomer_parameters.peduncle.curvature.uniformDistribution(50, 250);
    phytomer_parameters.peduncle.length_segments = 6;
    phytomer_parameters.peduncle.radial_subdivisions = 6;

    phytomer_parameters.inflorescence.flowers_per_rachis.uniformDistribution(1, 3);
    phytomer_parameters.inflorescence.flower_offset = 0.;
    phytomer_parameters.inflorescence.flower_arrangement_pattern = "opposite";
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
    shoot_parameters.internode_radius_initial = 0.0001;
    shoot_parameters.internode_radius_max = 0.0003;
    shoot_parameters.insertion_angle_tip.uniformDistribution(40, 70);
//    shoot_parameters.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters.internode_length_max = 0.015;
//    shoot_parameters.child_internode_length_min = 0.0; (default)
//    shoot_parameters.child_internode_length_decay_rate = 0; (default)
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20, 20);
    shoot_parameters.gravitropic_curvature = -200;

    shoot_parameters.phyllochron = 1;
//    shoot_parameters.leaf_flush_count = 1; (default)
    shoot_parameters.elongation_rate = 0.15;
    shoot_parameters.girth_growth_rate = 0.00005;
    shoot_parameters.vegetative_bud_break_time = 5;
    shoot_parameters.vegetative_bud_break_probability = 0.25;
//    shoot_parameters.max_terminal_floral_buds = 0; (default)
//    shoot_parameters.flower_bud_break_probability.uniformDistribution(0.1, 0.2);
    shoot_parameters.fruit_set_probability = 0.;
//    shoot_parameters.flowers_require_dormancy = false; (default)
//    shoot_parameters.growth_requires_dormancy = false; (default)
//    shoot_parameters.determinate_shoot_growth = true; (default)

    shoot_parameters.defineChildShootTypes({"main"}, {1.0});

    defineShootType("main", shoot_parameters);

}

uint PlantArchitecture::buildAsparagusPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize asparagus plant shoots
        initializeAsparagusShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(0, 0.f, 0.f), 0.0003, 0.015, 1, 0.1, 0.1, "main");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeBindweedShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_bindweed(context_ptr->getRandomGenerator());

    phytomer_parameters_bindweed.internode.pitch.uniformDistribution(0,15);
    phytomer_parameters_bindweed.internode.phyllotactic_angle = 180.f;
    phytomer_parameters_bindweed.internode.color = make_RGBcolor(0.64, 0.71, 0.53);
    phytomer_parameters_bindweed.internode.length_segments = 1;

    phytomer_parameters_bindweed.petiole.petioles_per_internode = 1;
    phytomer_parameters_bindweed.petiole.pitch.uniformDistribution(80, 100);
    phytomer_parameters_bindweed.petiole.radius = 0.001;
    phytomer_parameters_bindweed.petiole.length = 0.005;
    phytomer_parameters_bindweed.petiole.taper = 0;
    phytomer_parameters_bindweed.petiole.curvature = 0;
    phytomer_parameters_bindweed.petiole.color = phytomer_parameters_bindweed.internode.color;
    phytomer_parameters_bindweed.petiole.length_segments = 1;

    phytomer_parameters_bindweed.leaf.leaves_per_petiole = 1;
    phytomer_parameters_bindweed.leaf.pitch.uniformDistribution(5, 30);
    phytomer_parameters_bindweed.leaf.yaw = 0;
    phytomer_parameters_bindweed.leaf.roll = 90;
    phytomer_parameters_bindweed.leaf.prototype_function = BindweedLeafPrototype;
    phytomer_parameters_bindweed.leaf.prototype_scale = 0.04;

    phytomer_parameters_bindweed.peduncle.length = 0.001;
    phytomer_parameters_bindweed.inflorescence.flowers_per_rachis = 1;
    phytomer_parameters_bindweed.inflorescence.pitch = -90.f;
    phytomer_parameters_bindweed.inflorescence.flower_prototype_function = BindweedFlowerPrototype;
    phytomer_parameters_bindweed.inflorescence.flower_prototype_scale = 0.03;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_primary(context_ptr->getRandomGenerator());
    shoot_parameters_primary.phytomer_parameters = phytomer_parameters_bindweed;
    shoot_parameters_primary.vegetative_bud_break_probability = 0.25;
    shoot_parameters_primary.vegetative_bud_break_time = 1;
    shoot_parameters_primary.base_roll = 90;
    shoot_parameters_primary.internode_radius_initial = 0.001;
    shoot_parameters_primary.phyllochron = 1;
    shoot_parameters_primary.elongation_rate = 0.25;
    shoot_parameters_primary.girth_growth_rate = 0.;
    shoot_parameters_primary.internode_length_max = 0.03;
    shoot_parameters_primary.internode_length_decay_rate = 0;
    shoot_parameters_primary.insertion_angle_tip.uniformDistribution(50, 80);
    shoot_parameters_primary.flowers_require_dormancy = false;
    shoot_parameters_primary.growth_requires_dormancy = false;
    shoot_parameters_primary.flower_bud_break_probability = 0.2;
    shoot_parameters_primary.determinate_shoot_growth = false;
    shoot_parameters_primary.max_nodes = 15;
    shoot_parameters_primary.gravitropic_curvature = 20;
    shoot_parameters_primary.tortuosity = 0;
    shoot_parameters_primary.defineChildShootTypes({"secondary_bindweed"}, {1.f});

    ShootParameters shoot_parameters_base = shoot_parameters_primary;
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_bindweed;
    shoot_parameters_base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5-10,137.5+10);
    shoot_parameters_base.phytomer_parameters.petiole.petioles_per_internode = 0;
    shoot_parameters_base.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_base.phytomer_parameters.petiole.pitch = 0;
    shoot_parameters_base.vegetative_bud_break_probability = 1.0;
    shoot_parameters_base.vegetative_bud_break_time = 1;
    shoot_parameters_base.internode_radius_initial = 0.01;
    shoot_parameters_base.phyllochron = 1;
    shoot_parameters_base.elongation_rate = 0.25;
    shoot_parameters_base.girth_growth_rate = 0.;
    shoot_parameters_base.gravitropic_curvature = 0;
    shoot_parameters_base.internode_length_max = 0.01;
    shoot_parameters_base.internode_length_decay_rate = 0;
    shoot_parameters_base.insertion_angle_tip = 95;
    shoot_parameters_base.insertion_angle_decay_rate = 0;
    shoot_parameters_base.internode_radius_initial = 0.001;
    shoot_parameters_base.flowers_require_dormancy = false;
    shoot_parameters_base.growth_requires_dormancy = false;
    shoot_parameters_base.flower_bud_break_probability = 0.0;
    shoot_parameters_base.max_nodes.uniformDistribution(3,5);
    shoot_parameters_base.defineChildShootTypes({"primary_bindweed"},{1.f});

    ShootParameters shoot_parameters_children = shoot_parameters_primary;
    shoot_parameters_children.base_roll = 90;

    defineShootType("base_bindweed", shoot_parameters_base);
    defineShootType("primary_bindweed", shoot_parameters_primary);
    defineShootType("secondary_bindweed", shoot_parameters_children);

}

uint PlantArchitecture::buildBindweedPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize bindweed plant shoots
        initializeBindweedShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0.f, 0.f), 0.001, 0.001, 1, 1, 0, "base_bindweed");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 7, 1000, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeBeanShoots() {

    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());

    phytomer_parameters_trifoliate.internode.pitch = 20;
    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.2,0.25,0.05);
    phytomer_parameters_trifoliate.internode.length_segments = 5;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(30,70);
    phytomer_parameters_trifoliate.petiole.radius = 0.003;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.12,0.16);
    phytomer_parameters_trifoliate.petiole.taper = 0.25;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-100,200);
    phytomer_parameters_trifoliate.petiole.color = make_RGBcolor(0.28,0.35,0.07);
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.normalDistribution(0, 10);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll = -15;
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.2;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_function = BeanLeafPrototype_trifoliate;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.11,0.13);
    phytomer_parameters_trifoliate.leaf.subdivisions = 10;
    phytomer_parameters_trifoliate.leaf.unique_prototypes = 5;

    phytomer_parameters_trifoliate.peduncle.length = 0.04;
    phytomer_parameters_trifoliate.peduncle.radius = 0.001;
    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 40);
    phytomer_parameters_trifoliate.peduncle.roll = 90;
    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(-500, 500);
    phytomer_parameters_trifoliate.peduncle.length_segments = 1;
    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.inflorescence.flowers_per_rachis.uniformDistribution(1, 4);
    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.2;
    phytomer_parameters_trifoliate.inflorescence.flower_arrangement_pattern = "opposite";
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50,70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20,20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.015;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = BeanFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.08,0.1);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = BeanFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8,1.0);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.001;
    phytomer_parameters_unifoliate.petiole.radius = 0.0004;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(60,80);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.02;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype_function = BeanLeafPrototype_unifoliate;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 25;
    shoot_parameters_trifoliate.internode_radius_initial = 0.0005;
    shoot_parameters_trifoliate.internode_radius_max = 0.003;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40,60);
//    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.03;
//    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
//    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20,20);
    shoot_parameters_trifoliate.gravitropic_curvature = 300;

    shoot_parameters_trifoliate.phyllochron = 1;
//    shoot_parameters_trifoliate.leaf_flush_count = 1; (default)
    shoot_parameters_trifoliate.elongation_rate = 0.1;
    shoot_parameters_trifoliate.girth_growth_rate = 0.0003;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 3;
    shoot_parameters_trifoliate.vegetative_bud_break_probability = 0.25;
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
    shoot_parameters_unifoliate.vegetative_bud_break_probability = 0;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 0;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 1;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"},{1.0});

    defineShootType("unifoliate",shoot_parameters_unifoliate);
    defineShootType("trifoliate",shoot_parameters_trifoliate);


}

uint PlantArchitecture::buildBeanPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize bean plant shoots
        initializeBeanShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.01, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").internode_radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 15, 3, 3, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeCheeseweedShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_cheeseweed(context_ptr->getRandomGenerator());

    phytomer_parameters_cheeseweed.internode.pitch = 0;
    phytomer_parameters_cheeseweed.internode.phyllotactic_angle.uniformDistribution( 127.5f, 147.5);
    phytomer_parameters_cheeseweed.internode.color = make_RGBcolor(0.60, 0.65, 0.40);
    phytomer_parameters_cheeseweed.internode.length_segments = 1;

    phytomer_parameters_cheeseweed.petiole.petioles_per_internode = 1;
    phytomer_parameters_cheeseweed.petiole.pitch.uniformDistribution(45, 75);
    phytomer_parameters_cheeseweed.petiole.radius = 0.00075;
    phytomer_parameters_cheeseweed.petiole.length.uniformDistribution(0.02,0.06);
    phytomer_parameters_cheeseweed.petiole.taper = 0;
    phytomer_parameters_cheeseweed.petiole.curvature = -300;
    phytomer_parameters_cheeseweed.petiole.length_segments = 5;
    phytomer_parameters_cheeseweed.petiole.color = phytomer_parameters_cheeseweed.internode.color;

    phytomer_parameters_cheeseweed.leaf.leaves_per_petiole = 1;
    phytomer_parameters_cheeseweed.leaf.pitch.uniformDistribution(-30, 0);
    phytomer_parameters_cheeseweed.leaf.yaw = 0;
    phytomer_parameters_cheeseweed.leaf.roll = 0;
    phytomer_parameters_cheeseweed.leaf.prototype_function = CheeseweedLeafPrototype;
    phytomer_parameters_cheeseweed.leaf.prototype_scale = 0.035;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_base(context_ptr->getRandomGenerator());
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_cheeseweed;
    shoot_parameters_base.vegetative_bud_break_probability = 0.2;
    shoot_parameters_base.vegetative_bud_break_time = 3;
    shoot_parameters_base.internode_radius_initial = 0.0005;
    shoot_parameters_base.internode_radius_max = 0.001;
    shoot_parameters_base.phyllochron = 1;
    shoot_parameters_base.elongation_rate = 0.2;
    shoot_parameters_base.girth_growth_rate = 0.0002;
    shoot_parameters_base.gravitropic_curvature = 0;
    shoot_parameters_base.internode_length_max = 0.0015;
    shoot_parameters_base.internode_length_decay_rate = 0;
    shoot_parameters_base.flowers_require_dormancy = false;
    shoot_parameters_base.growth_requires_dormancy = false;
    shoot_parameters_base.flower_bud_break_probability = 0.;
    shoot_parameters_base.max_nodes = 8;

    defineShootType("base", shoot_parameters_base);

}

uint PlantArchitecture::buildCheeseweedPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize cheeseweed plant shoots
        initializeCheeseweedShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(0, 0.f, 0.f), 0.0001, 0.0025, 0.1, 0.1, 0, "base");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 1000, 3, 1000, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeCowpeaShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());

    phytomer_parameters_trifoliate.internode.pitch = 20;
    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.15, 0.2, 0.1);
    phytomer_parameters_trifoliate.internode.length_segments = 2;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(45,60);
    phytomer_parameters_trifoliate.petiole.radius = 0.0015;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.08,0.1);
    phytomer_parameters_trifoliate.petiole.taper = 0.25;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-200,-50);
    phytomer_parameters_trifoliate.petiole.color = make_RGBcolor(0.17,0.25,0.07);
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.normalDistribution(0, 10);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll = -15;
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.4;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_function = CowpeaLeafPrototype_trifoliate;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.09,0.1);
    phytomer_parameters_trifoliate.leaf.subdivisions = 6;
    phytomer_parameters_trifoliate.leaf.unique_prototypes = 5;

    phytomer_parameters_trifoliate.peduncle.length = 0.17;
    phytomer_parameters_trifoliate.peduncle.radius = 0.0015;
    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 30);
    phytomer_parameters_trifoliate.peduncle.roll = 90;
    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(50, 250);
    phytomer_parameters_trifoliate.peduncle.color = make_RGBcolor(0.15, 0.25, 0.1);
    phytomer_parameters_trifoliate.peduncle.length_segments = 6;
    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.inflorescence.flowers_per_rachis = 3;//.uniformDistribution(1, 3);
    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.;
    phytomer_parameters_trifoliate.inflorescence.flower_arrangement_pattern = "opposite";
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50,70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20,20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.015;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = CowpeaFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.02,0.025);
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = CowpeaFruitPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.,0.5);

    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
    phytomer_parameters_unifoliate.internode.pitch = 0;
    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
    phytomer_parameters_unifoliate.petiole.length = 0.001;
    phytomer_parameters_unifoliate.petiole.radius = 0.0004;
    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(60,80);
    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.01;
    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
    phytomer_parameters_unifoliate.leaf.prototype_function = CowpeaLeafPrototype_unifoliate;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = CowpeaPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 20;
    shoot_parameters_trifoliate.internode_radius_initial = 0.00025;
    shoot_parameters_trifoliate.internode_radius_max = 0.0025;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40,70);
//    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.02;
//    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
//    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20,20);
    shoot_parameters_trifoliate.gravitropic_curvature = 200;

    shoot_parameters_trifoliate.phyllochron = 1;
//    shoot_parameters_trifoliate.leaf_flush_count = 1; (default)
    shoot_parameters_trifoliate.elongation_rate = 0.1;
    shoot_parameters_trifoliate.girth_growth_rate = 0.0003;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 3;
    shoot_parameters_trifoliate.vegetative_bud_break_probability = 0.5;
//    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.1,0.2);
    shoot_parameters_trifoliate.fruit_set_probability = 0.5;
//    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
//    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
//    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)

    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"},{1.0});


    ShootParameters shoot_parameters_unifoliate = shoot_parameters_trifoliate;
    shoot_parameters_unifoliate.phytomer_parameters = phytomer_parameters_unifoliate;
    shoot_parameters_unifoliate.max_nodes = 1;
    shoot_parameters_unifoliate.vegetative_bud_break_probability = 0;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 0;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 1;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"},{1.0});

    defineShootType("unifoliate",shoot_parameters_unifoliate);
    defineShootType("trifoliate",shoot_parameters_trifoliate);

}

uint PlantArchitecture::buildCowpeaPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize cowpea plant shoots
        initializeCowpeaShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.02, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").internode_radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 16, 3, 3, 5, 100);

    return plantID;

}

void PlantArchitecture::initializePuncturevineShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_puncturevine(context_ptr->getRandomGenerator());

    phytomer_parameters_puncturevine.internode.pitch.uniformDistribution(0,15);
    phytomer_parameters_puncturevine.internode.phyllotactic_angle = 180.f;
    phytomer_parameters_puncturevine.internode.color = make_RGBcolor(0.55, 0.52, 0.39);
    phytomer_parameters_puncturevine.internode.length_segments = 1;

    phytomer_parameters_puncturevine.petiole.petioles_per_internode = 1;
    phytomer_parameters_puncturevine.petiole.pitch.uniformDistribution(80, 100);
    phytomer_parameters_puncturevine.petiole.radius = 0.0003;
    phytomer_parameters_puncturevine.petiole.length = 0.025;
    phytomer_parameters_puncturevine.petiole.taper = 0;
    phytomer_parameters_puncturevine.petiole.curvature = 0;
    phytomer_parameters_puncturevine.petiole.color = phytomer_parameters_puncturevine.internode.color;
    phytomer_parameters_puncturevine.petiole.length_segments = 1;

    phytomer_parameters_puncturevine.leaf.leaves_per_petiole = 11;
    phytomer_parameters_puncturevine.leaf.pitch.uniformDistribution(-10,10);
    phytomer_parameters_puncturevine.leaf.yaw = 30;
    phytomer_parameters_puncturevine.leaf.roll.uniformDistribution(-5,5);
    phytomer_parameters_puncturevine.leaf.prototype_function = PuncturevineLeafPrototype;
    phytomer_parameters_puncturevine.leaf.prototype_scale = 0.008;
    phytomer_parameters_puncturevine.leaf.leaflet_offset = 0.15;
    phytomer_parameters_puncturevine.leaf.leaflet_scale = 1;
    phytomer_parameters_puncturevine.leaf.subdivisions = 3;

    phytomer_parameters_puncturevine.peduncle.length = 0.001;
    phytomer_parameters_puncturevine.inflorescence.flowers_per_rachis = 1;
    phytomer_parameters_puncturevine.inflorescence.pitch = -90.f;
    phytomer_parameters_puncturevine.inflorescence.flower_prototype_function = PuncturevineFlowerPrototype;
    phytomer_parameters_puncturevine.inflorescence.flower_prototype_scale = 0.006;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_primary(context_ptr->getRandomGenerator());
    shoot_parameters_primary.phytomer_parameters = phytomer_parameters_puncturevine;
    shoot_parameters_primary.vegetative_bud_break_probability = 0.3;
    shoot_parameters_primary.vegetative_bud_break_time = 3;
    shoot_parameters_primary.base_roll = 90;
    shoot_parameters_primary.internode_radius_initial = 0.001;
    shoot_parameters_primary.phyllochron = 1;
    shoot_parameters_primary.elongation_rate = 0.2;
    shoot_parameters_primary.girth_growth_rate = 0.;
    shoot_parameters_primary.internode_length_max = 0.02;
    shoot_parameters_primary.internode_length_decay_rate = 0;
    shoot_parameters_primary.insertion_angle_tip.uniformDistribution(50, 80);
    shoot_parameters_primary.flowers_require_dormancy = false;
    shoot_parameters_primary.growth_requires_dormancy = false;
    shoot_parameters_primary.flower_bud_break_probability = 0.2;
    shoot_parameters_primary.determinate_shoot_growth = false;
    shoot_parameters_primary.max_nodes = 15;
    shoot_parameters_primary.gravitropic_curvature = 20;
    shoot_parameters_primary.tortuosity = 0;
    shoot_parameters_primary.defineChildShootTypes({"secondary_puncturevine"}, {1.f});

    ShootParameters shoot_parameters_base = shoot_parameters_primary;
    shoot_parameters_base.phytomer_parameters = phytomer_parameters_puncturevine;
    shoot_parameters_base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5-10,137.5+10);
    shoot_parameters_base.phytomer_parameters.petiole.petioles_per_internode = 0;
    shoot_parameters_base.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_base.phytomer_parameters.petiole.pitch = 0;
    shoot_parameters_base.vegetative_bud_break_probability = 1;
    shoot_parameters_base.vegetative_bud_break_time = 1;
    shoot_parameters_base.internode_radius_initial = 0.001;
    shoot_parameters_base.phyllochron = 1;
    shoot_parameters_base.elongation_rate = 0.25;
    shoot_parameters_base.girth_growth_rate = 0.;
    shoot_parameters_base.gravitropic_curvature = 0;
    shoot_parameters_base.internode_length_max = 0.01;
    shoot_parameters_base.internode_length_decay_rate = 0;
    shoot_parameters_base.insertion_angle_tip = 90;
    shoot_parameters_base.insertion_angle_decay_rate = 0;
    shoot_parameters_base.internode_radius_initial = 0.0005;
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

uint PlantArchitecture::buildPuncturevinePlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize puncturevine plant shoots
        initializePuncturevineShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(0, 0.f, 0.f), 0.001, 0.001, 1, 1, 0, "base_puncturevine");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 7, 1000, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeEasternRedbudShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_redbud(context_ptr->getRandomGenerator());

    phytomer_parameters_redbud.internode.pitch = 15;
    phytomer_parameters_redbud.internode.phyllotactic_angle.uniformDistribution(170,190);
//    phytomer_parameters_redbud.internode.color = make_RGBcolor(0.55, 0.52, 0.39);
    phytomer_parameters_redbud.internode.image_texture = "plugins/plantarchitecture/assets/textures/WesternRedbudBark.jpg";
    phytomer_parameters_redbud.internode.color.scale(0.3);
    phytomer_parameters_redbud.internode.length_segments = 1;
    phytomer_parameters_redbud.internode.max_floral_buds_per_petiole = 5;

    phytomer_parameters_redbud.petiole.petioles_per_internode = 1;
    phytomer_parameters_redbud.petiole.color = make_RGBcolor(0.65, 0.52, 0.39);
    phytomer_parameters_redbud.petiole.pitch.uniformDistribution(20, 40);
    phytomer_parameters_redbud.petiole.radius = 0.0017;
    phytomer_parameters_redbud.petiole.length = 0.05;
    phytomer_parameters_redbud.petiole.taper = 0;
    phytomer_parameters_redbud.petiole.curvature = 0;
    phytomer_parameters_redbud.petiole.length_segments = 1;

    phytomer_parameters_redbud.leaf.leaves_per_petiole = 1;
    phytomer_parameters_redbud.leaf.pitch.uniformDistribution(-110, -80);
    phytomer_parameters_redbud.leaf.yaw = 0;
    phytomer_parameters_redbud.leaf.roll.uniformDistribution(-5, 5);
    phytomer_parameters_redbud.leaf.prototype_function = RedbudLeafPrototype;
    phytomer_parameters_redbud.leaf.prototype_scale = 0.1;
    phytomer_parameters_redbud.leaf.subdivisions = 5;

    phytomer_parameters_redbud.peduncle.length = 0.02;
    phytomer_parameters_redbud.peduncle.pitch.uniformDistribution(50,90);
    phytomer_parameters_redbud.peduncle.color = make_RGBcolor(0.32, 0.05, 0.13);

    phytomer_parameters_redbud.inflorescence.flowers_per_rachis = 1;
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
    shoot_parameters_main.vegetative_bud_break_probability = 1.0;
    shoot_parameters_main.vegetative_bud_break_time = 1;
    shoot_parameters_main.phyllochron = 1.5;
    shoot_parameters_main.elongation_rate = 0.17;
    shoot_parameters_main.girth_growth_rate = 0.00045;
    shoot_parameters_main.gravitropic_curvature = 300;
    shoot_parameters_main.tortuosity = 20;
    shoot_parameters_main.internode_length_max = 0.04;
    shoot_parameters_main.internode_length_decay_rate = 0.005;
    shoot_parameters_main.insertion_angle_tip = 75;
    shoot_parameters_main.insertion_angle_decay_rate = 10;
    shoot_parameters_main.internode_radius_initial = 0.0015;
    shoot_parameters_main.internode_radius_max = 0.02;
    shoot_parameters_main.flowers_require_dormancy = true;
    shoot_parameters_main.growth_requires_dormancy = true;
    shoot_parameters_main.determinate_shoot_growth = false;
    shoot_parameters_main.max_terminal_floral_buds = 0;
    shoot_parameters_main.flower_bud_break_probability = 0.8;
    shoot_parameters_main.fruit_set_probability = 0.3;
    shoot_parameters_main.max_nodes = 30;
    shoot_parameters_main.base_roll = 90;

    ShootParameters shoot_parameters_trunk = shoot_parameters_main;
    shoot_parameters_trunk.phytomer_parameters.internode.pitch = 0;
    shoot_parameters_trunk.phytomer_parameters.internode.radial_subdivisions = 15;
    shoot_parameters_trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0;
    shoot_parameters_trunk.internode_radius_max = 1;
    shoot_parameters_trunk.phyllochron = 1.25;
    shoot_parameters_trunk.insertion_angle_tip = 60;
    shoot_parameters_trunk.max_nodes = 45;
    shoot_parameters_trunk.tortuosity = 5;
    shoot_parameters_trunk.defineChildShootTypes({"eastern_redbud_shoot"},{1.f});

    defineShootType("eastern_redbud_trunk", shoot_parameters_trunk);
    defineShootType("eastern_redbud_shoot", shoot_parameters_main);

}

void PlantArchitecture::initializeRomaineLettuceShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 0;
    phytomer_parameters.internode.phyllotactic_angle = 137.5;
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
    phytomer_parameters.leaf.pitch = 0;
    phytomer_parameters.leaf.yaw = 0;
    phytomer_parameters.leaf.roll = 0;
    phytomer_parameters.leaf.prototype_function = RomaineLettuceLeafPrototype;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.15,0.25);
    phytomer_parameters.leaf.subdivisions = 30;

    phytomer_parameters.phytomer_creation_function = RomaineLettucePhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters;
    shoot_parameters_mainstem.vegetative_bud_break_probability = 0;
    shoot_parameters_mainstem.internode_radius_initial = 0.01;
    shoot_parameters_mainstem.phyllochron = 1;
    shoot_parameters_mainstem.elongation_rate = 0.15;
    shoot_parameters_mainstem.girth_growth_rate = 0;
    shoot_parameters_mainstem.gravitropic_curvature = 10;
    shoot_parameters_mainstem.internode_length_max = 0.001;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 0.0;
    shoot_parameters_mainstem.max_nodes = 25;

    defineShootType("mainstem",shoot_parameters_mainstem);


}

uint PlantArchitecture::buildRomaineLettucePlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize sugarbeet plant shoots
        initializeSugarbeetShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(context_ptr->randu(0.f, 0.03f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.25f * M_PI), 0.005, 0.001, 1, 1, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 1000, 10, 60, 5, 100);

    return plantID;

}

uint PlantArchitecture::buildEasternRedbudPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize redbud plant shoots
        initializeEasternRedbudShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    uint uID_stem = addBaseStemShoot(plantID, 16, make_AxisRotation(context_ptr->randu(0,0.1*M_PI), context_ptr->randu(0,2*M_PI), context_ptr->randu(0,2*M_PI)), 0.0075, 0.05, 1, 1, 0.4, "eastern_redbud_trunk");

    makePlantDormant(plantID);

    removeShootLeaves( plantID, uID_stem );

    setPlantPhenologicalThresholds(plantID, 0, -1, 1, 3, 3, 12);

    return plantID;

}

void PlantArchitecture::initializeSorghumShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_sorghum(context_ptr->getRandomGenerator());

    phytomer_parameters_sorghum.internode.pitch = 0;
    phytomer_parameters_sorghum.internode.phyllotactic_angle.uniformDistribution(170,190);
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
    phytomer_parameters_sorghum.leaf.prototype_function = SorghumLeafPrototype;
    phytomer_parameters_sorghum.leaf.prototype_scale = 0.6;
    phytomer_parameters_sorghum.leaf.subdivisions = 50;
    phytomer_parameters_sorghum.leaf.unique_prototypes = 10;

    phytomer_parameters_sorghum.peduncle.length = 0.3;
    phytomer_parameters_sorghum.peduncle.radius = 0.008;
    phytomer_parameters_sorghum.peduncle.color = phytomer_parameters_sorghum.internode.color;
    phytomer_parameters_sorghum.peduncle.radial_subdivisions = 10;

    phytomer_parameters_sorghum.inflorescence.flowers_per_rachis = 1;
    phytomer_parameters_sorghum.inflorescence.pitch = 0;
    phytomer_parameters_sorghum.inflorescence.roll = 0;
    phytomer_parameters_sorghum.inflorescence.fruit_prototype_scale = 0.16;
    phytomer_parameters_sorghum.inflorescence.fruit_prototype_function = SorghumPaniclePrototype;

    phytomer_parameters_sorghum.phytomer_creation_function = SorghumPhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_sorghum;
    shoot_parameters_mainstem.vegetative_bud_break_probability = 0;
    shoot_parameters_mainstem.flower_bud_break_probability = 1;
    shoot_parameters_mainstem.internode_radius_initial = 0.003;
    shoot_parameters_mainstem.phyllochron = 1;
    shoot_parameters_mainstem.elongation_rate = 0.15;
    shoot_parameters_mainstem.girth_growth_rate = 0.0015;
    shoot_parameters_mainstem.internode_radius_max = 0.015;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-1000,-400);
    shoot_parameters_mainstem.internode_length_max = 0.26;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 1.0;
    shoot_parameters_mainstem.fruit_set_probability  = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"},{1.0});
    shoot_parameters_mainstem.max_nodes = 15;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem",shoot_parameters_mainstem);

}

uint PlantArchitecture::buildSorghumPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize sorghum plant shoots
        initializeSorghumShoots();
    }

    uint plantID = addPlantInstance(base_position - make_vec3(0,0,0.025), age);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.075f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.003, 0.06, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 2, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeSoybeanShoots() {

    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());

    phytomer_parameters_trifoliate.internode.pitch = 20;
    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.2,0.25,0.05);
    phytomer_parameters_trifoliate.internode.length_segments = 5;

    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(25,50);
    phytomer_parameters_trifoliate.petiole.radius = 0.003;
    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.12,0.16);
    phytomer_parameters_trifoliate.petiole.taper = 0.25;
    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-250,450);
    phytomer_parameters_trifoliate.petiole.color = phytomer_parameters_trifoliate.internode.color;
    phytomer_parameters_trifoliate.petiole.length_segments = 5;
    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
    phytomer_parameters_trifoliate.leaf.pitch.uniformDistribution(-10, 20);
    phytomer_parameters_trifoliate.leaf.yaw = 10;
    phytomer_parameters_trifoliate.leaf.roll = -15;
    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.5;
    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
    phytomer_parameters_trifoliate.leaf.prototype_function = SoybeanLeafPrototype_trifoliate;
    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.12,0.16);
    phytomer_parameters_trifoliate.leaf.subdivisions = 8;
    phytomer_parameters_trifoliate.leaf.unique_prototypes = 5;

    phytomer_parameters_trifoliate.peduncle.length = 0.01;
    phytomer_parameters_trifoliate.peduncle.radius = 0.001;
    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 40);
    phytomer_parameters_trifoliate.peduncle.roll = 90;
    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(-500, 500);
    phytomer_parameters_trifoliate.peduncle.length_segments = 1;
    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;

    phytomer_parameters_trifoliate.inflorescence.flowers_per_rachis.uniformDistribution(1, 4);
    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.2;
    phytomer_parameters_trifoliate.inflorescence.flower_arrangement_pattern = "opposite";
    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50,70);
    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20,20);
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.015;
    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = SoybeanFlowerPrototype;
    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.08,0.1);
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
    phytomer_parameters_unifoliate.leaf.prototype_function = SoybeanLeafPrototype_unifoliate;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction;

    shoot_parameters_trifoliate.max_nodes = 20;
    shoot_parameters_trifoliate.internode_radius_initial = 0.001;
    shoot_parameters_trifoliate.internode_radius_max = 0.004;
    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40,60);
//    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
    shoot_parameters_trifoliate.internode_length_max = 0.035;
//    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
//    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
    shoot_parameters_trifoliate.base_roll = 90;
    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20,20);
    shoot_parameters_trifoliate.gravitropic_curvature = 200;

    shoot_parameters_trifoliate.phyllochron = 1;
//    shoot_parameters_trifoliate.leaf_flush_count = 1; (default)
    shoot_parameters_trifoliate.elongation_rate = 0.15;
    shoot_parameters_trifoliate.girth_growth_rate = 0.0003;
    shoot_parameters_trifoliate.vegetative_bud_break_time = 3;
    shoot_parameters_trifoliate.vegetative_bud_break_probability = 0.25;
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
    shoot_parameters_unifoliate.vegetative_bud_break_probability = 0;
    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
    shoot_parameters_unifoliate.insertion_angle_tip = 0;
    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
    shoot_parameters_unifoliate.vegetative_bud_break_time = 1;
    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"},{1.0});

    defineShootType("unifoliate",shoot_parameters_unifoliate);
    defineShootType("trifoliate",shoot_parameters_trifoliate);

//    PhytomerParameters phytomer_parameters_trifoliate(context_ptr->getRandomGenerator());
//
//    phytomer_parameters_trifoliate.internode.pitch = 20;
//    phytomer_parameters_trifoliate.internode.phyllotactic_angle.uniformDistribution(145, 215);
//    phytomer_parameters_trifoliate.internode.max_floral_buds_per_petiole = 1;
//    phytomer_parameters_trifoliate.internode.max_vegetative_buds_per_petiole = 1;
//    phytomer_parameters_trifoliate.internode.color = make_RGBcolor(0.2,0.25,0.05);
//    phytomer_parameters_trifoliate.internode.length_segments = 5;
//
//    phytomer_parameters_trifoliate.petiole.petioles_per_internode = 1;
//    phytomer_parameters_trifoliate.petiole.pitch.uniformDistribution(30,70);
//    phytomer_parameters_trifoliate.petiole.radius = 0.003;
//    phytomer_parameters_trifoliate.petiole.length.uniformDistribution(0.12,0.16);
//    phytomer_parameters_trifoliate.petiole.taper = 0.25;
//    phytomer_parameters_trifoliate.petiole.curvature.uniformDistribution(-100,200);
//    phytomer_parameters_trifoliate.petiole.color = make_RGBcolor(0.28,0.35,0.07);
//    phytomer_parameters_trifoliate.petiole.length_segments = 5;
//    phytomer_parameters_trifoliate.petiole.radial_subdivisions = 6;
//
//    phytomer_parameters_trifoliate.leaf.leaves_per_petiole = 3;
//    phytomer_parameters_trifoliate.leaf.pitch.normalDistribution(0, 10);
//    phytomer_parameters_trifoliate.leaf.yaw = 10;
//    phytomer_parameters_trifoliate.leaf.roll = -15;
//    phytomer_parameters_trifoliate.leaf.leaflet_offset = 0.2;
//    phytomer_parameters_trifoliate.leaf.leaflet_scale = 0.9;
//    phytomer_parameters_trifoliate.leaf.prototype_function = SoybeanLeafPrototype_trifoliate;
//    phytomer_parameters_trifoliate.leaf.prototype_scale.uniformDistribution(0.11,0.13);
//    phytomer_parameters_trifoliate.leaf.subdivisions = 10;
//    phytomer_parameters_trifoliate.leaf.unique_prototypes = 5;
//
//    phytomer_parameters_trifoliate.peduncle.length = 0.04;
//    phytomer_parameters_trifoliate.peduncle.radius = 0.001;
//    phytomer_parameters_trifoliate.peduncle.pitch.uniformDistribution(0, 40);
//    phytomer_parameters_trifoliate.peduncle.roll = 90;
//    phytomer_parameters_trifoliate.peduncle.curvature.uniformDistribution(-500, 500);
//    phytomer_parameters_trifoliate.peduncle.length_segments = 1;
//    phytomer_parameters_trifoliate.peduncle.radial_subdivisions = 6;
//
//    phytomer_parameters_trifoliate.inflorescence.flowers_per_rachis.uniformDistribution(1, 4);
//    phytomer_parameters_trifoliate.inflorescence.flower_offset = 0.2;
//    phytomer_parameters_trifoliate.inflorescence.flower_arrangement_pattern = "opposite";
//    phytomer_parameters_trifoliate.inflorescence.pitch.uniformDistribution(50,70);
//    phytomer_parameters_trifoliate.inflorescence.roll.uniformDistribution(-20,20);
//    phytomer_parameters_trifoliate.inflorescence.flower_prototype_scale = 0.015;
//    phytomer_parameters_trifoliate.inflorescence.flower_prototype_function = BeanFlowerPrototype;
//    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_scale.uniformDistribution(0.08,0.1);
//    phytomer_parameters_trifoliate.inflorescence.fruit_prototype_function = BeanFruitPrototype;
//    phytomer_parameters_trifoliate.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8,1.0);
//
//    PhytomerParameters phytomer_parameters_unifoliate = phytomer_parameters_trifoliate;
//    phytomer_parameters_unifoliate.internode.pitch = 0;
//    phytomer_parameters_unifoliate.internode.max_vegetative_buds_per_petiole = 0;
//    phytomer_parameters_unifoliate.internode.max_floral_buds_per_petiole = 0;
//    phytomer_parameters_unifoliate.petiole.petioles_per_internode = 2;
//    phytomer_parameters_unifoliate.petiole.length = 0.001;
//    phytomer_parameters_unifoliate.petiole.radius = 0.0004;
//    phytomer_parameters_unifoliate.petiole.pitch.uniformDistribution(60,80);
//    phytomer_parameters_unifoliate.leaf.leaves_per_petiole = 1;
//    phytomer_parameters_unifoliate.leaf.prototype_scale = 0.02;
//    phytomer_parameters_unifoliate.leaf.pitch.uniformDistribution(-10, 10);
//    phytomer_parameters_unifoliate.leaf.prototype_function = BeanLeafPrototype_unifoliate;
//
//    // ---- Shoot Parameters ---- //
//
//    ShootParameters shoot_parameters_trifoliate(context_ptr->getRandomGenerator());
//    shoot_parameters_trifoliate.phytomer_parameters = phytomer_parameters_trifoliate;
//    shoot_parameters_trifoliate.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction;
//
//    shoot_parameters_trifoliate.max_nodes = 25;
//    shoot_parameters_trifoliate.internode_radius_initial = 0.0005;
//    shoot_parameters_trifoliate.internode_radius_max = 0.003;
//    shoot_parameters_trifoliate.insertion_angle_tip.uniformDistribution(40,60);
////    shoot_parameters_trifoliate.child_insertion_angle_decay_rate = 0; (default)
//    shoot_parameters_trifoliate.internode_length_max = 0.03;
////    shoot_parameters_trifoliate.child_internode_length_min = 0.0; (default)
////    shoot_parameters_trifoliate.child_internode_length_decay_rate = 0; (default)
//    shoot_parameters_trifoliate.base_roll = 90;
//    shoot_parameters_trifoliate.base_yaw.uniformDistribution(-20,20);
//    shoot_parameters_trifoliate.gravitropic_curvature = 300;
//
//    shoot_parameters_trifoliate.phyllochron = 1;
////    shoot_parameters_trifoliate.leaf_flush_count = 1; (default)
//    shoot_parameters_trifoliate.elongation_rate = 0.1;
//    shoot_parameters_trifoliate.girth_growth_rate = 0.00025;
//    shoot_parameters_trifoliate.vegetative_bud_break_time = 3;
//    shoot_parameters_trifoliate.vegetative_bud_break_probability = 0.25;
////    shoot_parameters_trifoliate.max_terminal_floral_buds = 0; (default)
//    shoot_parameters_trifoliate.flower_bud_break_probability.uniformDistribution(0.8,1.0);
//    shoot_parameters_trifoliate.fruit_set_probability = 0.4;
////    shoot_parameters_trifoliate.flowers_require_dormancy = false; (default)
////    shoot_parameters_trifoliate.growth_requires_dormancy = false; (default)
////    shoot_parameters_trifoliate.determinate_shoot_growth = true; (default)
//
//    shoot_parameters_trifoliate.defineChildShootTypes({"trifoliate"},{1.0});
//
//
//    ShootParameters shoot_parameters_unifoliate = shoot_parameters_trifoliate;
//    shoot_parameters_unifoliate.phytomer_parameters = phytomer_parameters_unifoliate;
//    shoot_parameters_unifoliate.max_nodes = 1;
//    shoot_parameters_unifoliate.vegetative_bud_break_probability = 0;
//    shoot_parameters_unifoliate.flower_bud_break_probability = 0;
//    shoot_parameters_unifoliate.insertion_angle_tip = 0;
//    shoot_parameters_unifoliate.insertion_angle_decay_rate = 0;
//    shoot_parameters_unifoliate.vegetative_bud_break_time = 1;
//    shoot_parameters_unifoliate.defineChildShootTypes({"trifoliate"},{1.0});
//
//    defineShootType("unifoliate",shoot_parameters_unifoliate);
//    defineShootType("trifoliate",shoot_parameters_trifoliate);


}

uint PlantArchitecture::buildSoybeanPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize bean plant shoots
        initializeSoybeanShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    AxisRotation base_rotation = make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_unifoliate = addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.01, 0.01, 0.01, 0, "unifoliate");

    appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, 0.5f * M_PI), shoot_types.at("trifoliate").internode_radius_initial.val(), shoot_types.at("trifoliate").internode_length_max.val(), 0.1, 0.1, 0, "trifoliate");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 15, 3, 3, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeStrawberryShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(80,100);
    phytomer_parameters.internode.color = make_RGBcolor(0.38, 0.48, 0.1);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(10,45);
    phytomer_parameters.petiole.radius = 0.003;
    phytomer_parameters.petiole.length = 0.2;
    phytomer_parameters.petiole.taper = 0.5;
    phytomer_parameters.petiole.curvature.uniformDistribution(-200,100);
    phytomer_parameters.petiole.color = make_RGBcolor(0.24, 0.28, 0.08);
    phytomer_parameters.petiole.length_segments = 5;

    phytomer_parameters.leaf.leaves_per_petiole = 3;
    phytomer_parameters.leaf.pitch.uniformDistribution(-30,-15);
    phytomer_parameters.leaf.yaw = 20;
    phytomer_parameters.leaf.roll = -30;
    phytomer_parameters.leaf.leaflet_offset = 0.01;
    phytomer_parameters.leaf.leaflet_scale = 1.0;
    phytomer_parameters.leaf.prototype_function = StrawberryLeafPrototype;
    phytomer_parameters.leaf.prototype_scale = 0.1;
    phytomer_parameters.leaf.subdivisions = 6;
    phytomer_parameters.leaf.unique_prototypes = 5;

    phytomer_parameters.peduncle.length = 0.17;
    phytomer_parameters.peduncle.radius = 0.00075;
    phytomer_parameters.peduncle.pitch = 35;
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -200;
    phytomer_parameters.peduncle.length_segments = 5;
    phytomer_parameters.peduncle.radial_subdivisions = 6;
    phytomer_parameters.peduncle.color = phytomer_parameters.petiole.color;

    phytomer_parameters.inflorescence.flowers_per_rachis.uniformDistribution(1,3);
    phytomer_parameters.inflorescence.flower_offset = 0.2;
    phytomer_parameters.inflorescence.flower_arrangement_pattern = "alternate";
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
    shoot_parameters.internode_radius_initial = 0.001;
    shoot_parameters.internode_radius_max = 0.005;
    shoot_parameters.insertion_angle_tip = 40;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.01;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20,20);
    shoot_parameters.gravitropic_curvature.uniformDistribution(-10,0);
    shoot_parameters.tortuosity = 0;

    shoot_parameters.phyllochron = 1.;
    shoot_parameters.leaf_flush_count = 1;
    shoot_parameters.elongation_rate = 0.1;
    shoot_parameters.girth_growth_rate = 0.0003;
    shoot_parameters.vegetative_bud_break_time = 3;
    shoot_parameters.vegetative_bud_break_probability = 0.25;
    shoot_parameters.flower_bud_break_probability = 1;
    shoot_parameters.fruit_set_probability = 0.5;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = true;

    shoot_parameters.defineChildShootTypes({"mainstem"},{1.0});

    defineShootType("mainstem",shoot_parameters);

}

uint PlantArchitecture::buildStrawberryPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize strawberry plant shoots
        initializeStrawberryShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.001, 0.004, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 10, 3, 1, 100);

    return plantID;

}

void PlantArchitecture::initializeSugarbeetShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_sugarbeet(context_ptr->getRandomGenerator());

    phytomer_parameters_sugarbeet.internode.pitch = 0;
    phytomer_parameters_sugarbeet.internode.phyllotactic_angle = 137.5;
    phytomer_parameters_sugarbeet.internode.color = make_RGBcolor(0.44,0.58,0.19);
    phytomer_parameters_sugarbeet.internode.length_segments = 2;

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
    phytomer_parameters_sugarbeet.leaf.prototype_function = SugarbeetLeafPrototype;
    phytomer_parameters_sugarbeet.leaf.prototype_scale.uniformDistribution(0.15,0.25);
    phytomer_parameters_sugarbeet.leaf.subdivisions = 40;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_sugarbeet;
    shoot_parameters_mainstem.vegetative_bud_break_probability = 0;
    shoot_parameters_mainstem.internode_radius_initial = 0.005;
    shoot_parameters_mainstem.phyllochron = 1;
    shoot_parameters_mainstem.elongation_rate = 0.25;
    shoot_parameters_mainstem.girth_growth_rate = 0;
    shoot_parameters_mainstem.gravitropic_curvature = 10;
    shoot_parameters_mainstem.internode_length_max = 0.001;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 0.0;
    shoot_parameters_mainstem.max_nodes = 30;

    defineShootType("mainstem",shoot_parameters_mainstem);


}

uint PlantArchitecture::buildSugarbeetPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize sugarbeet plant shoots
        initializeSugarbeetShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    uint uID_stem = addBaseStemShoot(plantID, 3, make_AxisRotation(context_ptr->randu(0.f, 0.01f * M_PI), 0.f * context_ptr->randu(0.f, 2.f * M_PI), 0.25f * M_PI), 0.005, 0.001, 1, 1, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, 1000, 10, 60, 5, 100);

    return plantID;

}

void PlantArchitecture::initializeTomatoShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters(context_ptr->getRandomGenerator());

    phytomer_parameters.internode.pitch = 10;
    phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(145, 215);
    phytomer_parameters.internode.color = make_RGBcolor(0.213, 0.270, 0.056);
    phytomer_parameters.internode.length_segments = 1;

    phytomer_parameters.petiole.petioles_per_internode = 1;
    phytomer_parameters.petiole.pitch.uniformDistribution(45,60);
    phytomer_parameters.petiole.radius = 0.002;
    phytomer_parameters.petiole.length = 0.3;
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
    phytomer_parameters.leaf.prototype_function = TomatoLeafPrototype;
    phytomer_parameters.leaf.prototype_scale.uniformDistribution(0.1,0.14);
    phytomer_parameters.leaf.subdivisions = 6;
    phytomer_parameters.leaf.unique_prototypes = 5;

    phytomer_parameters.peduncle.length = 0.16;
    phytomer_parameters.peduncle.radius = 0.0015;
    phytomer_parameters.peduncle.pitch = 20;
    phytomer_parameters.peduncle.roll = 0;
    phytomer_parameters.peduncle.curvature = -700;
    phytomer_parameters.peduncle.color = phytomer_parameters.internode.color;
    phytomer_parameters.peduncle.length_segments = 5;
    phytomer_parameters.peduncle.radial_subdivisions = 8;

    phytomer_parameters.inflorescence.flowers_per_rachis = 8;
    phytomer_parameters.inflorescence.flower_offset = 0.15;
    phytomer_parameters.inflorescence.flower_arrangement_pattern = "opposite";
    phytomer_parameters.inflorescence.pitch = 90;
    phytomer_parameters.inflorescence.roll.uniformDistribution(-30,30);
    phytomer_parameters.inflorescence.flower_prototype_scale = 0.03;
    phytomer_parameters.inflorescence.flower_prototype_function = TomatoFlowerPrototype;
    phytomer_parameters.inflorescence.fruit_prototype_scale = 0.04;
    phytomer_parameters.inflorescence.fruit_prototype_function = TomatoFruitPrototype;
    phytomer_parameters.inflorescence.fruit_gravity_factor_fraction = 0.5;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters(context_ptr->getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.phytomer_parameters.phytomer_creation_function = TomatoPhytomerCreationFunction;

    shoot_parameters.max_nodes = 20;
    shoot_parameters.internode_radius_initial = 0.001;
    shoot_parameters.internode_radius_max = 0.009;
    shoot_parameters.insertion_angle_tip = 30;
    shoot_parameters.insertion_angle_decay_rate = 0;
    shoot_parameters.internode_length_max = 0.04;
    shoot_parameters.internode_length_min = 0.0;
    shoot_parameters.internode_length_decay_rate = 0;
    shoot_parameters.base_roll = 90;
    shoot_parameters.base_yaw.uniformDistribution(-20,20);
    shoot_parameters.gravitropic_curvature = -20;
    shoot_parameters.tortuosity = 0;

    shoot_parameters.phyllochron = 1;
    shoot_parameters.leaf_flush_count = 1;
    shoot_parameters.elongation_rate = 0.1;
    shoot_parameters.girth_growth_rate = 0.0002;
    shoot_parameters.vegetative_bud_break_time = 3;
    shoot_parameters.vegetative_bud_break_probability = 0.6;
    shoot_parameters.flower_bud_break_probability = 0.5;
    shoot_parameters.fruit_set_probability = 0.5;
    shoot_parameters.flowers_require_dormancy = false;
    shoot_parameters.growth_requires_dormancy = false;
    shoot_parameters.determinate_shoot_growth = true;

    shoot_parameters.defineChildShootTypes({"mainstem"},{1.0});

    defineShootType("mainstem",shoot_parameters);

}

uint PlantArchitecture::buildTomatoPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize tomato plant shoots
        initializeTomatoShoots();
    }

    uint plantID = addPlantInstance(base_position, age);

    AxisRotation base_rotation = make_AxisRotation(0, context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI));
    uint uID_stem = addBaseStemShoot(plantID, 1, base_rotation, 0.002, 0.04, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, 10, 3, 2, 100);

    return plantID;

}

void PlantArchitecture::initializeWheatShoots() {

    // ---- Phytomer Parameters ---- //

    PhytomerParameters phytomer_parameters_wheat(context_ptr->getRandomGenerator());

    phytomer_parameters_wheat.internode.pitch = 0;
    phytomer_parameters_wheat.internode.phyllotactic_angle.uniformDistribution(67,77);
    phytomer_parameters_wheat.internode.color = make_RGBcolor(0.27,0.31,0.16);
    phytomer_parameters_wheat.internode.length_segments = 1;
    phytomer_parameters_wheat.internode.radial_subdivisions = 6;
    phytomer_parameters_wheat.internode.max_floral_buds_per_petiole = 0;
    phytomer_parameters_wheat.internode.max_vegetative_buds_per_petiole = 0;

    phytomer_parameters_wheat.petiole.petioles_per_internode = 1;
    phytomer_parameters_wheat.petiole.pitch.uniformDistribution(-40,-20);
    phytomer_parameters_wheat.petiole.radius = 0.0;
    phytomer_parameters_wheat.petiole.length = 0.01;
    phytomer_parameters_wheat.petiole.taper = 0;
    phytomer_parameters_wheat.petiole.curvature = 0;
    phytomer_parameters_wheat.petiole.length_segments = 1;

    phytomer_parameters_wheat.leaf.leaves_per_petiole = 1;
    phytomer_parameters_wheat.leaf.pitch = 0;
    phytomer_parameters_wheat.leaf.yaw = 0;
    phytomer_parameters_wheat.leaf.roll = 0;
    phytomer_parameters_wheat.leaf.prototype_function = WheatLeafPrototype;
    phytomer_parameters_wheat.leaf.prototype_scale = 0.1;
    phytomer_parameters_wheat.leaf.subdivisions = 20;
    phytomer_parameters_wheat.leaf.unique_prototypes = 10;

    phytomer_parameters_wheat.peduncle.length = 0.07;
    phytomer_parameters_wheat.peduncle.radius = 0.002;
    phytomer_parameters_wheat.peduncle.color = phytomer_parameters_wheat.internode.color;
    phytomer_parameters_wheat.peduncle.curvature = -200;
    phytomer_parameters_wheat.peduncle.radial_subdivisions = 6;

    phytomer_parameters_wheat.inflorescence.flowers_per_rachis = 1;
    phytomer_parameters_wheat.inflorescence.pitch = 0;
    phytomer_parameters_wheat.inflorescence.roll = 0;
    phytomer_parameters_wheat.inflorescence.fruit_prototype_scale = 0.1;
    phytomer_parameters_wheat.inflorescence.fruit_prototype_function = WheatSpikePrototype;

    phytomer_parameters_wheat.phytomer_creation_function = WheatPhytomerCreationFunction;

    // ---- Shoot Parameters ---- //

    ShootParameters shoot_parameters_mainstem(context_ptr->getRandomGenerator());
    shoot_parameters_mainstem.phytomer_parameters = phytomer_parameters_wheat;
    shoot_parameters_mainstem.vegetative_bud_break_probability = 0;
    shoot_parameters_mainstem.flower_bud_break_probability = 1;
    shoot_parameters_mainstem.internode_radius_initial = 0.001;
    shoot_parameters_mainstem.phyllochron = 1;
    shoot_parameters_mainstem.elongation_rate = 0.15;
    shoot_parameters_mainstem.girth_growth_rate = 0.0015;
    shoot_parameters_mainstem.internode_radius_max = 0.0025;
    shoot_parameters_mainstem.gravitropic_curvature.uniformDistribution(-2000,-800);
    shoot_parameters_mainstem.internode_length_max = 0.025;
    shoot_parameters_mainstem.internode_length_decay_rate = 0;
    shoot_parameters_mainstem.flowers_require_dormancy = false;
    shoot_parameters_mainstem.growth_requires_dormancy = false;
    shoot_parameters_mainstem.determinate_shoot_growth = false;
    shoot_parameters_mainstem.flower_bud_break_probability = 1.0;
    shoot_parameters_mainstem.fruit_set_probability  = 1.0;
    shoot_parameters_mainstem.defineChildShootTypes({"mainstem"},{1.0});
    shoot_parameters_mainstem.max_nodes = 15;
    shoot_parameters_mainstem.max_terminal_floral_buds = 1;

    defineShootType("mainstem",shoot_parameters_mainstem);

}

uint PlantArchitecture::buildWheatPlant(const helios::vec3 &base_position, float age) {

    if (shoot_types.empty()) {
        //automatically initialize wheat plant shoots
        initializeWheatShoots();
    }

    uint plantID = addPlantInstance(base_position - make_vec3(0,0,0.025), age);

    uint uID_stem = addBaseStemShoot(plantID, 1, make_AxisRotation(context_ptr->randu(0.f, 0.1f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), context_ptr->randu(0.f, 2.f * M_PI)), 0.001, 0.025, 0.01, 0.01, 0, "mainstem");

    breakPlantDormancy(plantID);

    setPlantPhenologicalThresholds(plantID, 0, -1, -1, 2, 5, 100);

    return plantID;

}