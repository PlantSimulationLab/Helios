#include "PlantArchitecture.h"

using namespace helios;

void PlantArchitecture::addAlmondShoot() {

    PhytomerParameters phytomer_parameters_almond = getPhytomerParametersFromLibrary("almond");

    PhytomerParameters phytomer_parameters_proleptic = phytomer_parameters_almond;
    phytomer_parameters_proleptic.internode.length = 0.02;
    phytomer_parameters_proleptic.internode.radius = 0.003;
    phytomer_parameters_proleptic.internode.curvature = 100;
    phytomer_parameters_proleptic.inflorescence.fruit_prototype_scale = 0.015*make_vec3(1, 1,1);
    phytomer_parameters_proleptic.internode.pitch = 0;
    phytomer_parameters_proleptic.leaf.prototype_scale = 0.05*make_vec3(1, 1,1);
    phytomer_parameters_proleptic.petiole.yaw.uniformDistribution( deg2rad(130), deg2rad(145) );
    phytomer_parameters_proleptic.inflorescence.fruit_prototype_scale = 0.015*make_vec3(1, 1,1);
    phytomer_parameters_proleptic.inflorescence.fruit_pitch.uniformDistribution(-0.2*M_PI,0.*M_PI);
    phytomer_parameters_proleptic.inflorescence.fruit_roll = 0.5*M_PI;
    phytomer_parameters_proleptic.inflorescence.fruit_per_inflorescence.uniformDistribution(1,2);


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
    shoot_parameters_trunk.bud_break_probability = 1;
    shoot_parameters_trunk.bud_time = 0;
    shoot_parameters_trunk.defineChildShootTypes({"sylleptic","proleptic"},{0.,1});

    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.max_nodes = 20;
    shoot_parameters_proleptic.phyllochron = 0.5;
    shoot_parameters_proleptic.growth_rate = 0.004;
    shoot_parameters_proleptic.bud_break_probability = 1;
    shoot_parameters_proleptic.bud_time = 0;
    shoot_parameters_proleptic.shoot_internode_taper = 0.5;
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_proleptic;
    shoot_parameters_proleptic.child_insertion_angle_tip = 0;
    shoot_parameters_proleptic.child_internode_length_max = 0.06;
    shoot_parameters_proleptic.child_internode_length_min = 0.002;
    shoot_parameters_proleptic.child_internode_length_decay_rate = 0.01;
    shoot_parameters_proleptic.fruit_set_probability = 0.5;
    shoot_parameters_proleptic.flower_probability = 0.5;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
//    shoot_parameters_proleptic.defineChildShootTypes({"sylleptic","proleptic"},{0.,1});

    ShootParameters shoot_parameters_sylleptic(context_ptr->getRandomGenerator());
    shoot_parameters_sylleptic = shoot_parameters_proleptic;
    shoot_parameters_sylleptic.phytomer_parameters = phytomer_parameters_sylleptic;
    shoot_parameters_sylleptic.bud_break_probability = 1;
    shoot_parameters_sylleptic.shoot_internode_taper = 0.5;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = false;


    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);

    //---- Make Initial Woody Structure ---- //

    int Nplants = 1;

    uint plant0 = addPlantInstance(nullorigin, 0);

    uint uID_trunk = addBaseShoot( plant0, 12, make_AxisRotation(0,0,0.*M_PI), "trunk" );

    plant_instances.at(plant0).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

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
            Shoot parent_shoot = *plant_instances.at(plant0).shoot_tree.at(uID_trunk);
            uint childID = addChildShoot(plant0, uID_trunk, node, 4, make_AxisRotation(shoot_parameters_trunk.child_insertion_angle_tip.val(), parent_shoot.phyllotactic_angle.val()*float(node)+context_ptr->randu(-0.1f, 0.1f), -0. * M_PI), new_shoot_type_label);
            auto phytomers = plant_instances.at(plant0).shoot_tree.at(childID)->phytomers;
            for( uint p=0; p<phytomers.size(); p++ ){
                phytomers.at(p)->removeLeaf();
                if( p!=phytomers.size()-1 ) {
                    phytomers.at(p)->flower_bud_state = BUD_DEAD;
                }
                phytomers.at(p)->vegetative_bud_state = BUD_DEAD;
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

    setPlantPhenologicalThresholds(plant0, assimilate_flowering_threshold, 20, 3, 5, 10);


}

void PlantArchitecture::addAlmondTree() {

    Context context;

    PhytomerParameters phytomer_parameters_almond = getPhytomerParametersFromLibrary("almond");

    PhytomerParameters phytomer_parameters_proleptic = phytomer_parameters_almond;
    phytomer_parameters_proleptic.internode.length = 0.02;
    phytomer_parameters_proleptic.internode.radius = 0.005;
    phytomer_parameters_proleptic.internode.curvature = 75;
    phytomer_parameters_proleptic.inflorescence.fruit_prototype_scale = 0.015*make_vec3(1, 1,1);
    phytomer_parameters_proleptic.internode.pitch = 0;
    phytomer_parameters_proleptic.leaf.prototype_scale = 0.05*make_vec3(1, 1,1);
    phytomer_parameters_proleptic.petiole.yaw.uniformDistribution( deg2rad(130), deg2rad(145) );
    phytomer_parameters_proleptic.inflorescence.fruit_prototype_scale = 0.015*make_vec3(1, 1,1);
    phytomer_parameters_proleptic.inflorescence.fruit_pitch.uniformDistribution(-0.2*M_PI,0.*M_PI);
    phytomer_parameters_proleptic.inflorescence.fruit_roll = 0.5*M_PI;
    phytomer_parameters_proleptic.inflorescence.fruit_per_inflorescence.uniformDistribution(1,2);


    PhytomerParameters phytomer_parameters_sylleptic = phytomer_parameters_proleptic;
    phytomer_parameters_sylleptic.internode.length = 0.04;

    PhytomerParameters phytomer_parameters_trunk = phytomer_parameters_almond;
    phytomer_parameters_trunk.internode.length = 0.05;
    phytomer_parameters_trunk.internode.radius = 0.008;
    phytomer_parameters_trunk.internode.petioles_per_internode = 0;

    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.max_nodes = 20;
    shoot_parameters_trunk.shoot_internode_taper = 0.3;
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_trunk;
    shoot_parameters_trunk.bud_break_probability = 1;
    shoot_parameters_trunk.bud_time = 0;
    shoot_parameters_trunk.defineChildShootTypes({"sylleptic","proleptic"},{0.,1});

    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.max_nodes = 30;
    shoot_parameters_proleptic.phyllochron = 0.75;
    shoot_parameters_proleptic.growth_rate = 0.004;
    shoot_parameters_proleptic.bud_break_probability = 0.5;
    shoot_parameters_proleptic.bud_time = 0;
    shoot_parameters_proleptic.shoot_internode_taper = 0.5;
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_proleptic;
    shoot_parameters_proleptic.child_insertion_angle_tip = deg2rad(30);
    shoot_parameters_proleptic.child_internode_length_max = 0.06;
    shoot_parameters_proleptic.child_internode_length_min = 0.002;
    shoot_parameters_proleptic.child_internode_length_decay_rate = 0.015;
    shoot_parameters_proleptic.fruit_set_probability = 0.5;
    shoot_parameters_proleptic.flower_probability = 0.5;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
//    shoot_parameters_proleptic.defineChildShootTypes({"sylleptic","proleptic"},{0.,1});

    ShootParameters shoot_parameters_sylleptic(context_ptr->getRandomGenerator());
    shoot_parameters_sylleptic = shoot_parameters_proleptic;
    shoot_parameters_sylleptic.phytomer_parameters = phytomer_parameters_sylleptic;
    shoot_parameters_sylleptic.bud_break_probability = 1;
    shoot_parameters_sylleptic.shoot_internode_taper = 0.5;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = false;


    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);

    //---- Make Initial Woody Structure ---- //

    int Nplants = 1;

    uint plant0 = addPlantInstance(nullorigin, 0);

    uint uID_trunk = addBaseShoot( plant0, 12, make_AxisRotation(context.randu(0,0.2*M_PI),context.randu(0,2*M_PI), 0.*M_PI), "trunk" );

    plant_instances.at(plant0).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plant0).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->flower_bud_state = BUD_DEAD;
        phytomer->vegetative_bud_state = BUD_DEAD;
    }

    uint Nscaffolds = context.randu(3,4);

    for( int i=0; i<Nscaffolds; i++ ) {
        uint uID_shoot = appendShoot(plant0, uID_trunk, context.randu(10, 15), make_AxisRotation(context.randu(deg2rad(20), deg2rad(60)), (float(i) + context.randu(-0.1f, 0.1f)) / float(Nscaffolds) * 2 * M_PI, 0), "proleptic");

        plant_instances.at(plant0).shoot_tree.at(uID_shoot)->makeDormant();

        uint blind_nodes = context.randu(3,5);
        for( int b=0; b<blind_nodes; b++){
            if( b<plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.size() ) {
                plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.at(b)->removeLeaf();
                plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.at(b)->flower_bud_state = BUD_DEAD;
                plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.at(b)->vegetative_bud_state = BUD_DEAD;
            }
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

    setPlantPhenologicalThresholds(plant0, assimilate_flowering_threshold, 100, 3, 5, 10);


}