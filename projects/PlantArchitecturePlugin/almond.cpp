#include "PlantArchitecture.h"

using namespace helios;

void PlantArchitecture::addAlmondShoot() {

    PhytomerParameters phytomer_parameters_almond = getPhytomerParametersFromLibrary("almond");

    PhytomerParameters phytomer_parameters_proleptic = phytomer_parameters_almond;
//    phytomer_parameters_proleptic.internode.length = 0.02;
//    phytomer_parameters_proleptic.internode.radius = 0.005;
    phytomer_parameters_proleptic.internode.pitch = 0;
    phytomer_parameters_proleptic.internode.color = make_RGBcolor(0.6,0.45,0.15);
    phytomer_parameters_proleptic.leaf.prototype_scale = 0.05;
    phytomer_parameters_proleptic.inflorescence.flower_prototype_scale = 0.01;
    phytomer_parameters_proleptic.inflorescence.fruit_prototype_scale = 0.01;
    phytomer_parameters_proleptic.inflorescence.fruit_pitch.uniformDistribution(-0.2*M_PI,0.*M_PI);
    phytomer_parameters_proleptic.inflorescence.fruit_roll = 0.5*M_PI;
    phytomer_parameters_proleptic.inflorescence.fruit_per_inflorescence.uniformDistribution(1,2);


    PhytomerParameters phytomer_parameters_sylleptic = phytomer_parameters_proleptic;

    PhytomerParameters phytomer_parameters_trunk = phytomer_parameters_almond;
    phytomer_parameters_trunk.internode.color = make_RGBcolor(0.6,0.45,0.15);

    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.max_nodes = 20;
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_trunk;
    shoot_parameters_trunk.girth_growth_rate = 1.025;
    shoot_parameters_trunk.bud_break_probability = 1;
    shoot_parameters_trunk.bud_time = 0;
    shoot_parameters_trunk.defineChildShootTypes({"sylleptic","proleptic"},{0.,1});

    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.max_nodes = 30;
    shoot_parameters_proleptic.phyllochron = 1;
    shoot_parameters_proleptic.phyllotactic_angle.uniformDistribution( deg2rad(130), deg2rad(145) );
    shoot_parameters_proleptic.elongation_rate = 0.03;
    shoot_parameters_proleptic.girth_growth_rate = 1.025;
    shoot_parameters_proleptic.bud_break_probability = 0.75;
    shoot_parameters_proleptic.bud_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature = 75;
    //shoot_parameters_proleptic.shoot_internode_taper = 0.5;
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_proleptic;
    shoot_parameters_proleptic.child_insertion_angle_tip = deg2rad(40);
    shoot_parameters_proleptic.child_internode_length_max = 0.06;
    shoot_parameters_proleptic.child_internode_length_min = 0.06;
    shoot_parameters_proleptic.child_internode_length_decay_rate = 0.015;
    shoot_parameters_proleptic.fruit_set_probability = 0.5;
    shoot_parameters_proleptic.flower_probability = 0.75;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.defineChildShootTypes({"sylleptic","proleptic"},{0.,1.});

    ShootParameters shoot_parameters_sylleptic(context_ptr->getRandomGenerator());
    shoot_parameters_sylleptic = shoot_parameters_proleptic;
    shoot_parameters_sylleptic.phytomer_parameters = phytomer_parameters_sylleptic;
    shoot_parameters_sylleptic.bud_break_probability = 1;
    //shoot_parameters_sylleptic.shoot_internode_taper = 0.5;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = false;


    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);

    //---- Make Initial Woody Structure ---- //

    int Nplants = 1;

    uint plant0 = addPlantInstance(nullorigin, 0);

    uint uID_trunk = addBaseStemShoot(plant0, 5, 0.3, make_AxisRotation(0, 0, 0. * M_PI), "trunk");

    //plant_instances.at(plant0).shoot_tree.at(uID_trunk)->meristem_is_alive = false;




//    auto phytomers = plant_instances.at(plant0).shoot_tree.at(uID_trunk)->phytomers;
//    for( const auto & phytomer : phytomers ){
//        //phytomer->removeLeaf();
//        phytomer->flower_bud_state = BUD_DEAD;
//        //phytomer->vegetative_bud_state = BUD_DEAD;
//        phytomer->vegetative_bud_state = BUD_ACTIVE;
//    }
//
//    uint node_number = getShootNodeCount(plant0, uID_trunk);
//
//    // child shoots
//    for( uint node = 0; node<node_number-1; node++ ) {
//        std::string new_shoot_type_label;
//        if (sampleChildShootType(plant0, uID_trunk, new_shoot_type_label)) {
//            std::cout << "Node " << node << " along main branch has type of " << new_shoot_type_label << std::endl;
//            uint current_node_number;
//            if( new_shoot_type_label=="sylleptic" ){
//                current_node_number = 7;
//            }else{
//                current_node_number = 4;
//            }
//            Shoot parent_shoot = *plant_instances.at(plant0).shoot_tree.at(uID_trunk);
//            uint childID = addChildShoot(plant0, uID_trunk, node, 4, new_shoot_type_label);
//            auto phytomers = plant_instances.at(plant0).shoot_tree.at(childID)->phytomers;
//            for( uint p=0; p<phytomers.size(); p++ ){
////                phytomers.at(p)->removeLeaf();
////                if( p!=phytomers.size()-1 ) {
////                    phytomers.at(p)->flower_bud_state = BUD_DEAD;
////                }
////                phytomers.at(p)->vegetative_bud_state = BUD_DEAD;
//                phytomers.at(p)->flower_bud_state = BUD_DEAD;
//                phytomers.at(p)->vegetative_bud_state = BUD_ACTIVE;
//            }
////            plant_instances.at(plant0).shoot_tree.at(childID)->makeDormant();
//        }
//    }




    breakPlantDormancy(plant0);

    setPlantPhenologicalThresholds(plant0, 0, 2, 3, 5, 20);


}

void PlantArchitecture::addAlmondTree() {

    PhytomerParameters phytomer_parameters_almond = getPhytomerParametersFromLibrary("almond");

    PhytomerParameters phytomer_parameters_proleptic = phytomer_parameters_almond;
//    phytomer_parameters_proleptic.internode.length = 0.02;
//    phytomer_parameters_proleptic.internode.radius = 0.006;
    phytomer_parameters_proleptic.internode.pitch = 0;
    phytomer_parameters_proleptic.internode.color = make_RGBcolor(0.6,0.45,0.15);
    phytomer_parameters_proleptic.leaf.prototype_scale = 0.015;
    phytomer_parameters_proleptic.inflorescence.flower_prototype_scale = 0.01;
    phytomer_parameters_proleptic.inflorescence.fruit_prototype_scale = 0.008;
    phytomer_parameters_proleptic.inflorescence.fruit_pitch.uniformDistribution(-0.2*M_PI,0.*M_PI);
    phytomer_parameters_proleptic.inflorescence.fruit_roll = 0.5*M_PI;
    phytomer_parameters_proleptic.inflorescence.fruit_per_inflorescence.uniformDistribution(1,2);

    phytomer_parameters_proleptic.internode.length_segments = 1;

    phytomer_parameters_proleptic.petiole.length = 0.001;

    PhytomerParameters phytomer_parameters_sylleptic = phytomer_parameters_proleptic;
//    phytomer_parameters_sylleptic.internode.length = 0.04;

    PhytomerParameters phytomer_parameters_trunk = phytomer_parameters_almond;
//    phytomer_parameters_trunk.internode.length = 0.05;
//    phytomer_parameters_trunk.internode.radius = 0.01;
    phytomer_parameters_trunk.internode.color = make_RGBcolor(0.6,0.45,0.15);

    ShootParameters shoot_parameters_trunk(context_ptr->getRandomGenerator());
    shoot_parameters_trunk.max_nodes = 20;
    shoot_parameters_trunk.girth_growth_rate = 1.02;
    shoot_parameters_trunk.phytomer_parameters = phytomer_parameters_trunk;
    shoot_parameters_trunk.internode_radius_initial = 0.005;
    shoot_parameters_trunk.bud_break_probability = 1;
    shoot_parameters_trunk.bud_time = 0;
    shoot_parameters_trunk.tortuosity = 1000;
    shoot_parameters_trunk.defineChildShootTypes({"scaffold"},{1});
    shoot_parameters_trunk.phyllochron = 100;

    ShootParameters shoot_parameters_proleptic(context_ptr->getRandomGenerator());
    shoot_parameters_proleptic.max_nodes = 36;
    shoot_parameters_proleptic.phyllochron.uniformDistribution(1,1.1);
    shoot_parameters_proleptic.phyllotactic_angle.uniformDistribution( deg2rad(130), deg2rad(145) );
    shoot_parameters_proleptic.elongation_rate = 0.04;
    shoot_parameters_proleptic.girth_growth_rate = 1.02;
    shoot_parameters_proleptic.bud_break_probability = 0.75;
    shoot_parameters_proleptic.bud_time = 0;
    shoot_parameters_proleptic.gravitropic_curvature.uniformDistribution(180,210);
    shoot_parameters_proleptic.tortuosity = 60;
    shoot_parameters_proleptic.internode_radius_initial = 0.00075;
    shoot_parameters_proleptic.phytomer_parameters = phytomer_parameters_proleptic;
    shoot_parameters_proleptic.child_insertion_angle_tip.uniformDistribution( deg2rad(35), deg2rad(45));
    shoot_parameters_proleptic.child_internode_length_max = 0.005;
    shoot_parameters_proleptic.child_internode_length_min = 0.0005;
    shoot_parameters_proleptic.child_internode_length_decay_rate = 0.0025;
    shoot_parameters_proleptic.fruit_set_probability = 0.5;
    shoot_parameters_proleptic.flower_probability = 0.75;
    shoot_parameters_proleptic.flowers_require_dormancy = true;
    shoot_parameters_proleptic.growth_requires_dormancy = true;
    shoot_parameters_proleptic.defineChildShootTypes({"sylleptic","proleptic"},{0.,1.});

    ShootParameters shoot_parameters_sylleptic = shoot_parameters_proleptic;
    shoot_parameters_sylleptic.phytomer_parameters.leaf.prototype_scale = 0.025;
    shoot_parameters_sylleptic.bud_break_probability = 1;
    shoot_parameters_sylleptic.gravitropic_curvature.uniformDistribution(250,300);
    shoot_parameters_sylleptic.child_internode_length_max = 0.01;
    shoot_parameters_sylleptic.flowers_require_dormancy = true;
    shoot_parameters_sylleptic.growth_requires_dormancy = true; //seems to not be working when false
    shoot_parameters_proleptic.defineChildShootTypes({"sylleptic"},{1.0});

    ShootParameters shoot_parameters_scaffold = shoot_parameters_proleptic;
    shoot_parameters_scaffold.gravitropic_curvature.uniformDistribution(50,70);
    shoot_parameters_scaffold.phyllochron = 0.9;
    shoot_parameters_scaffold.child_internode_length_max = 0.015;
    shoot_parameters_scaffold.tortuosity = 20;

    defineShootType("trunk", shoot_parameters_trunk);
    defineShootType("scaffold", shoot_parameters_scaffold);
    defineShootType("proleptic", shoot_parameters_proleptic);
    defineShootType("sylleptic", shoot_parameters_sylleptic);

    //---- Make Initial Woody Structure ---- //

    int Nplants = 1;

    uint plant0 = addPlantInstance(nullorigin, 0);

    uint uID_trunk = addBaseStemShoot(plant0, 3, 0.1, make_AxisRotation(context_ptr->randu(0.f, 0.05f * M_PI), context_ptr->randu(0.f, 2.f * M_PI), 0.f * M_PI), "trunk");

    plant_instances.at(plant0).shoot_tree.at(uID_trunk)->meristem_is_alive = false;

    auto phytomers = plant_instances.at(plant0).shoot_tree.at(uID_trunk)->phytomers;
    for( const auto & phytomer : phytomers ){
        phytomer->removeLeaf();
        phytomer->flower_bud_state = BUD_DEAD;
        phytomer->vegetative_bud_state = BUD_DEAD;
    }

    uint Nscaffolds = context_ptr->randu(5,5);

    for( int i=0; i<Nscaffolds; i++ ) {
        uint uID_shoot = appendShoot(plant0, uID_trunk, context_ptr->randu(6, 8), 0.02, make_AxisRotation(context_ptr->randu(deg2rad(35), deg2rad(45)), (float(i) + context_ptr->randu(-0.1f, 0.1f)) / float(Nscaffolds) * 2 * M_PI, 0), "scaffold");

        //plant_instances.at(plant0).shoot_tree.at(uID_shoot)->makeDormant();
        plant_instances.at(plant0).shoot_tree.at(uID_shoot)->breakDormancy();

            uint blind_nodes = context_ptr->randu(2,3);
            for( int b=0; b<blind_nodes; b++){
                if( b<plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.size() ) {
                    plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.at(b)->removeLeaf();
                    plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.at(b)->flower_bud_state = BUD_DEAD;
                    plant_instances.at(plant0).shoot_tree.at(uID_shoot)->phytomers.at(b)->vegetative_bud_state = BUD_DEAD;
                }
            }

    }

    setPlantPhenologicalThresholds(plant0, 1, 1, 3, 7, 12);

    //setShootOrigin(plant0, uID_trunk, make_vec3(0,0,0.5));

}