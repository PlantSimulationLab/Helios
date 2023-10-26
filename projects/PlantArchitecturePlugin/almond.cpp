#include "PlantArchitecture.h"

using namespace helios;

void PlantArchitecture::addAlmondShoot() {

    PhytomerParameters phytomer_parameters_almond = getPhytomerParametersFromLibrary("almond");

    PhytomerParameters phytomer_parameters_spur = phytomer_parameters_almond;
    phytomer_parameters_spur.internode.length = 0.002;
    phytomer_parameters_spur.internode.radius = 0.001;
    phytomer_parameters_spur.internode.curvature = 0;
    phytomer_parameters_spur.internode.pitch = 0;
    phytomer_parameters_spur.leaf.prototype_scale = 0.05*make_vec3(1, 1,1);
    phytomer_parameters_spur.petiole.yaw.uniformDistribution( 0.4*M_PI, 0.6*M_PI );
    phytomer_parameters_spur.inflorescence.fruit_prototype_scale = 0.015*make_vec3(1, 1,1);
    phytomer_parameters_spur.inflorescence.fruit_pitch.uniformDistribution(-0.2*M_PI,0.*M_PI);
    phytomer_parameters_spur.inflorescence.fruit_roll = 0;
    phytomer_parameters_spur.inflorescence.fruit_per_inflorescence.uniformDistribution(1,2);

    PhytomerParameters phytomer_parameters_proleptic = phytomer_parameters_spur;
    phytomer_parameters_proleptic.internode.length = 0.02;
    phytomer_parameters_proleptic.internode.radius = 0.003;
    phytomer_parameters_proleptic.internode.curvature = 100;
    phytomer_parameters_proleptic.inflorescence.fruit_prototype_scale = 0.015*make_vec3(1, 1,1);

    PhytomerParameters phytomer_parameters_sylleptic = phytomer_parameters_proleptic;
    phytomer_parameters_sylleptic.internode.length = 0.04;

    PhytomerParameters phytomer_parameters_trunk = phytomer_parameters_almond;
    phytomer_parameters_trunk.internode.length = 0.05;
    //phytomer_parameters_trunk.internode.radius = 0.005;
    phytomer_parameters_trunk.leaf.prototype_scale = 0.075*make_vec3(1, 1,1);
    phytomer_parameters_trunk.inflorescence.flower_prototype_scale = 0.015*make_vec3(1, 1,1);

    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.max_nodes = 20;
    shoot_parameters_trunk.shoot_internode_taper = 0.3;
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_trunk;
    shoot_parameters_trunk.defineChildShootTypes({"spur","sylleptic","proleptic"},{0.5,0.2,0.3});
//    shoot_parameters_trunk.defineChildShootTypes({"sylleptic"},{1});


    ShootParameters shoot_parameters_spur(context_ptr->getRandomGenerator());
    shoot_parameters_spur.max_nodes = 15;
    shoot_parameters_spur.phyllochron = 1;
    shoot_parameters_spur.growth_rate = 0.004;
    shoot_parameters_spur.bud_break_probability = 1;
    shoot_parameters_spur.bud_time = 100000;
    shoot_parameters_spur.shoot_internode_taper = 0;
    shoot_parameters_spur.phytomer_parameters = phytomer_parameters_spur;
    shoot_parameters_spur.child_insertion_angle.uniformDistribution(deg2rad(35),deg2rad(45));
    shoot_parameters_spur.fruit_set_probability = 0.75;
    shoot_parameters_spur.flower_probability = 1;
    shoot_parameters_spur.flowers_require_dormancy = true;
    shoot_parameters_spur.growth_requires_dormancy = true;

    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic = shoot_parameters_spur;
    shoot_parameters_proleptic.max_nodes = 10;
    shoot_parameters_proleptic.phyllochron = 1;
    shoot_parameters_proleptic.growth_rate = 0.004;
    shoot_parameters_proleptic.bud_break_probability = 1;
    shoot_parameters_proleptic.bud_time = 0;
    shoot_parameters_proleptic.shoot_internode_taper = 0.5;
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_proleptic;
    shoot_parameters_proleptic.child_insertion_angle.uniformDistribution(deg2rad(35),deg2rad(45));
    shoot_parameters_proleptic.fruit_set_probability = 0.5;
    shoot_parameters_proleptic.flower_probability = 0.5;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.defineChildShootTypes({"spur","sylleptic","proleptic"},{0.5,0.2,0.3});

    ShootParameters shoot_parameters_sylleptic(context_ptr->getRandomGenerator());
    shoot_parameters_sylleptic = shoot_parameters_proleptic;
    shoot_parameters_sylleptic.phytomer_parameters = phytomer_parameters_sylleptic;
    shoot_parameters_sylleptic.bud_break_probability = 1;
    shoot_parameters_sylleptic.shoot_internode_taper = 0.5;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = false;


    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("spur", shoot_parameters_spur);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);

    //---- Make Initial Woody Structure ---- //

    int Nplants = 1;

    uint plant0 = addPlantInstance(nullorigin, 0);

    uint uID_trunk = addBaseShoot( plant0, 12, make_AxisRotation(0,0,0.*M_PI), "trunk" );

    auto phytomers = plant_instances.at(plant0).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->flower_bud_state = BUD_DEAD;
        phytomer->vegetative_bud_state = BUD_DEAD;
    }

    uint node_number = getShootNodeCount(plant0, uID_trunk);

    // child shoots
    for( uint node = 0; node<node_number-1; node++ ) {
        std::string new_shoot_type_label;
        if (sampleChildShootType(plant0, uID_trunk, new_shoot_type_label)) {
            std::cout << "Node " << node << " along main branch has type of " << new_shoot_type_label << std::endl;
            uint current_node_number;
            if( new_shoot_type_label=="sylleptic" ){
                current_node_number = 7;
            }else{
                current_node_number = 4;
            }
            uint childID = addChildShoot(plant0, uID_trunk, node, 4, make_AxisRotation(shoot_parameters_trunk.child_insertion_angle.val(), context_ptr->randu(0.f, 2.f * M_PI), -0. * M_PI), new_shoot_type_label);
            auto phytomers = plant_instances.at(plant0).shoot_tree.at(childID)->phytomers;
            for( uint p=0; p<phytomers.size(); p++ ){
                phytomers.at(p)->removeLeaf();
                if( p==phytomers.size()-1 || new_shoot_type_label=="sylleptic" ){
                    phytomers.at(p)->flower_bud_state = BUD_DORMANT;
                    phytomers.at(p)->vegetative_bud_state = BUD_DORMANT;
                }else{
                    phytomers.at(p)->flower_bud_state = BUD_DEAD;
                    phytomers.at(p)->vegetative_bud_state = BUD_DEAD;
                }
            }
            plant_instances.at(plant0).shoot_tree.at(childID)->makeDormant();
        }
    }

    //----- Initialize Shoot Carbon Pool ----- //

    float initial_assimilate_pool = 100; // mg SC/g DW
    float assimilate_flowering_threshold = 85; // mg SC/g DW

    for( auto &plant: plant_instances ){
        for( auto &shoot: plant.second.shoot_tree ){
            shoot->assimilate_pool = initial_assimilate_pool;
        }
    }

    setPlantPhenologicalThresholds(plant0, assimilate_flowering_threshold, 20, 3, 5, 0);


}