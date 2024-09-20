/** \file "CarbohydrateModel.cpp" Definitions related to carbohydrate model calculations in the plant architecture plug-in.

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


//Parameters
float C_molecular_wt = 12.01; //g C mol^-1
float cm2_m2 = 10000; //cm^2 m^-2


float SLA = 92; //ratio of leaf area to leaf dry mass
float leaf_carbon_percentage = .4444; //portion of the dry weight of the leaf made up by carbon


float total_flower_cost = 8.33e-3; //mol C flower^-1  (Bustan & Goldschmidt 2002)
float flower_production_cost = total_flower_cost*.69; //mol C flower^-1  (Bustan & Goldschmidt 2002)
float flower_growth_respiration = total_flower_cost*.31; //mol C flower^-1  (Bustan & Goldschmidt 2002)


float nut_density = 525 * 1000; //g m^-3
float percent_kernel = .27; //portion of the nut made up by the kernel
float percent_shell = .19;  //portion of the nut made up by the shell
float percent_hull = .54;  //portion of the nut made up by the hull
float kernel_carbon_percentage = .454; //portion of the kernel made up by carbon by dry weight
float shell_carbon_percentage = .470;  //portion of the shell made up by caron by dry weight
float hull_carbon_percentage = .494;  //portion of the hull made up by carbon by dry weight
float nut_carbon_percentage = percent_kernel*kernel_carbon_percentage + percent_shell*shell_carbon_percentage + percent_hull*hull_carbon_percentage;









float Phytomer::calculatePhytomerConstructionCosts(){

    //\todo make these values externally settable
    float leaf_construction_cost_base = 5.57; //mol C/m^2
    float internode_construction_cost_base = 39028; //mol C/m^3

    float phytomer_carbon_cost = 0.f; //mol C

    //leaves (cost per area basis)
    float leaf_area = 0;
    for( const auto &petiole : leaf_objIDs ) {
        for( uint leaf_objID : petiole ) {
            leaf_area += context_ptr->getObjectArea(leaf_objID);
        }
    }
    leaf_area /= current_leaf_scale_factor; //convert to fully-expanded area
    phytomer_carbon_cost += leaf_construction_cost_base*leaf_area;

    //internode (cost per volume basis)
    float internode_volume = 0;
    uint start_ind = shoot_index.x*phytomer_parameters.internode.length_segments;
    for( int segment = start_ind; segment<start_ind+phytomer_parameters.internode.length_segments; segment++ ){
        internode_volume += context_ptr->getTubeObjectSegmentVolume( parent_shoot_ptr->internode_tube_objID, segment );
    }
    internode_volume /= current_internode_scale_factor; //convert to fully-elongated volume
    phytomer_carbon_cost += internode_construction_cost_base*internode_volume;

    return phytomer_carbon_cost;

}

float Phytomer::calculateFlowerConstructionCosts(const FloralBud &fbud) {

    //\todo make these values externally settable
    float flower_construction_cost_base = 8.33e-3; //mol C/flower (Bustan & Goldschmidt 2002)

    float flower_carbon_cost = 0.f; //mol C

    for( uint flower_objID : fbud.inflorescence_objIDs ) {
        flower_carbon_cost += flower_construction_cost_base;
    }

    return flower_carbon_cost;
}

void PlantArchitecture::initializePlantCarbohydratePool(uint plantID, float carbohydrate_concentration_molC_m3 ){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantCarbohydratePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( carbohydrate_concentration_molC_m3 < 0 ){
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantCarbohydratePool): Carbohydrate concentration must be greater than or equal to zero.");
    }

    //loop over all shoots
    for( auto &shoot : plant_instances.at(plantID).shoot_tree ) {
        initializeShootCarbohydratePool(plantID, shoot->ID, carbohydrate_concentration_molC_m3);
    }

}

void PlantArchitecture::initializeShootCarbohydratePool(uint plantID, uint shootID, float carbohydrate_concentration_molC_m3 ){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::initializeShootCarbohydratePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::initializeShootCarbohydratePool): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if( carbohydrate_concentration_molC_m3 < 0 ){
        helios_runtime_error("ERROR (PlantArchitecture::initializeShootCarbohydratePool): Carbohydrate concentration must be greater than or equal to zero.");
    }

    //calculate shoot volume
    float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();

    //set carbon pool
    plant_instances.at(plantID).shoot_tree.at(shootID)->carbohydrate_pool_molC = shoot_volume * carbohydrate_concentration_molC_m3;

}

void PlantArchitecture::accumulateShootPhotosynthesis() {

    uint A_prim_data_missing = 0;

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){

            float net_photosynthesis = 0;

            for( auto &phytomer: shoot->phytomers ){

                for( auto &leaf_objID: flatten(phytomer->leaf_objIDs) ){
                    for( uint UUID : context_ptr->getObjectPrimitiveUUIDs(leaf_objID) ){
                        if( context_ptr->doesPrimitiveDataExist(UUID, "cumulative_net_photosynthesis") && context_ptr->getPrimitiveDataType(UUID,"cumulative_net_photosynthesis")==HELIOS_TYPE_FLOAT ){
                            float A;
                            context_ptr->getPrimitiveData(UUID,"cumulative_net_photosynthesis",A);
                            net_photosynthesis += A*context_ptr->getPrimitiveArea(UUID);
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", 0.f);
                        }else{
                            A_prim_data_missing++;
                        }
                    }
                }

            }

            shoot->carbohydrate_pool_molC += net_photosynthesis;

        }

    }

    if( A_prim_data_missing>0 ){
        std::cerr << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): " << A_prim_data_missing << " leaf primitives were missing net_photosynthesis primitive data. Did you run the photosynthesis model?" << std::endl;
    }

}

void PlantArchitecture::subtractShootMaintenanceCarbon(float dt ) {

    //\todo move to externally settable parameter
    float stem_maintainance_respiration_rate = 1.9458e-05; //mol C respired/mol C in pool/day

    float rho_w = 675000; //Almond wood density (g m^-3)
    float rho_cw = rho_w * .5 / 12.01; //Density of carbon in almond wood (mol C m^-3)


    for (auto &plant: plant_instances) {

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for (auto &shoot: *shoot_tree) {
            shoot->carbohydrate_pool_molC -= context_ptr->getTubeObjectVolume(shoot->internode_tube_objID) * rho_cw * stem_maintainance_respiration_rate * dt;
        }

    }

}

void PlantArchitecture::subtractShootGrowthCarbon(){

    //\todo move to externally settable parameter
    float rho_w = 675000; //Almond wood density (g m^-3)
    float rho_cw = rho_w * .5 / 12.01; //Density of carbon in almond wood (mol C m^-3)
    float growth_respiration_fraction = 0.28; //Accounting for the growth carbon lost to respiration (assumed 28%)


    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){
            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume();

            float shoot_growth_carbon_demand = rho_cw*( shoot_volume - shoot->old_shoot_volume) / (1-growth_respiration_fraction); //Structural carbon + growth respiration required to construct new shoot volume
            shoot->carbohydrate_pool_molC -= shoot_growth_carbon_demand; //Subtract construction carbon + growth respiration from the carbon pool
            shoot->old_shoot_volume = shoot_volume; //Set old volume to the current volume for the next timestep

        }

    }

}


float Phytomer::calculateFruitConstructionCosts(const FloralBud &fbud) {

    //\todo make these values externally settable
    float fruit_construction_cost_base = 29021; //mol C/m^3

    float fruit_carbon_cost = 0.f; //mol C

    //fruit (cost per fruit basis)
    for( uint fruit_objID : fbud.inflorescence_objIDs ) {
        float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID)/fbud.current_fruit_scale_factor; //mature fruit volume
        fruit_carbon_cost += fruit_construction_cost_base*(mature_volume)*(fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor);
    }

    return fruit_carbon_cost;
}

void PlantArchitecture::checkCarbonPool_abortbuds(){
    float carbohydrate_threshold = 0; //mol C/m3
    float day_threshold = 3;
    float storage_conductance = 0.5;

    //\todo make these values externally settable
    float fruit_construction_cost_base = 29021; //mol C/m^3

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;
        auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

        for( auto &shoot: *shoot_tree ){

            uint shootID = shoot->ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();
            float working_carb_pool = shoot->carbohydrate_pool_molC;
            uint parentID = shoot->parent_shoot_ID;


            for(auto &child : shoot->childIDs ){

                int node_number = child.first;
                int child_ID = child.second;

            }


            if(shoot->carbohydrate_pool_molC >= carbohydrate_threshold*shoot_volume){
                shoot_tree_ptr->at(parentID)->carbohydrate_pool_molC +=  (shoot->carbohydrate_pool_molC - carbohydrate_threshold*shoot_volume)*storage_conductance;
                shoot->carbohydrate_pool_molC -= (shoot->carbohydrate_pool_molC - carbohydrate_threshold*shoot_volume)*storage_conductance;
                shoot->days_with_negative_carbon_balance = 0;

            }else if(shoot->days_with_negative_carbon_balance <= day_threshold){
                shoot->days_with_negative_carbon_balance += 1;
            }else if(shoot->days_with_negative_carbon_balance > day_threshold) {

                auto phytomers = &shoot->phytomers;

                bool living_buds = true;

                while (living_buds) {

                    living_buds = false;

                    for (auto &phytomer: *phytomers) {
                        for (auto &petiole: phytomer->floral_buds) {
                            //all currently active lateral buds die at dormancy
                            for (auto &fbud: petiole) {
                                if (fbud.state != BUD_DORMANT && fbud.state != BUD_DEAD) {
                                    for (uint fruit_objID: fbud.inflorescence_objIDs) {
                                        float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID) /
                                                              fbud.current_fruit_scale_factor; //mature fruit volume
                                        working_carb_pool += fruit_construction_cost_base * (mature_volume) *
                                                             (fbud.current_fruit_scale_factor -
                                                              fbud.previous_fruit_scale_factor);
                                    }
                                    phytomer->setFloralBudState(BUD_DEAD, fbud);
                                    break;

                                }else{
                                    living_buds = true;
                                }
                            }
                            break;
                        }
                        if (working_carb_pool > carbohydrate_threshold) {
                            living_buds = false;
                            break;
                        }

                    }

                }


            }

        }

    }

}