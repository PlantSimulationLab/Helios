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


float rho_w = 540000; //Almond wood density (g m^-3) - Grossman 1993
float wood_carbon_percentage = .4559; //portion of the dry weight of the wood made up by carbon - Grossman 1993

float shoot_root_ratio = 3;


float SLA = 2.5e-2; //ratio of leaf area to leaf dry mass m^2 / g DW
float leaf_carbon_percentage = .444; //portion of the dry weight of the leaf made up by carbon - Penning de Vries et al. 1989


float total_flower_cost = 8.33e-4; //mol C flower^-1  (Bustan & Goldschmidt 2002)
float flower_production_cost = total_flower_cost*.69; //mol C flower^-1  (Bustan & Goldschmidt 2002)
float flower_growth_respiration = total_flower_cost*.31; //mol C flower^-1  (Bustan & Goldschmidt 2002)


float nut_density = 525000; //g m^-3
float percent_kernel = .27; //portion of the nut made up by the kernel
float percent_shell = .19;  //portion of the nut made up by the shell
float percent_hull = .54;  //portion of the nut made up by the hull
float kernel_carbon_percentage = .454; //portion of the kernel made up by carbon by dry weight
float shell_carbon_percentage = .470;  //portion of the shell made up by caron by dry weight
float hull_carbon_percentage = .494;  //portion of the hull made up by carbon by dry weight
float nut_carbon_percentage = percent_kernel*kernel_carbon_percentage + percent_shell*shell_carbon_percentage + percent_hull*hull_carbon_percentage; //overall portion of the nut made up by carbon by dry weight




float Phytomer::calculatePhytomerConstructionCosts(){

    //\todo make these values externally settable
    float leaf_construction_cost_base = leaf_carbon_percentage/(C_molecular_wt*SLA); //mol C/m^2
    float internode_construction_cost_base = rho_w * wood_carbon_percentage / C_molecular_wt; // (mol C /m^3)

    float phytomer_carbon_cost = 0.f; //mol C

    //leaves (cost per area basis)
    float leaf_area = 0;
    for( const auto &petiole : leaf_objIDs ) {
        for( uint leaf_objID : petiole ) {
            leaf_area += context_ptr->getObjectArea(leaf_objID);
        }
    }
    phytomer_carbon_cost += leaf_construction_cost_base*leaf_area;

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

void PlantArchitecture::initializeCarbohydratePool(float carbohydrate_concentration_molC_m3){

    for( auto &plant: plant_instances ) {

        auto shoot_tree = &plant.second.shoot_tree;

        for (auto &shoot: *shoot_tree) {
            //calculate shoot volume
            float shoot_volume = shoot->calculateShootInternodeVolume();
            //set carbon pool
            shoot->carbohydrate_pool_molC = shoot_volume * carbohydrate_concentration_molC_m3;

        }
    }
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

void PlantArchitecture::accumulateHourlyLeafPhotosynthesis() {

    for( auto &plant: plant_instances ){

        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){

            if(shoot->isdormant) {
                for( auto &phytomer: shoot->phytomers ){
                    for( auto &leaf_objID: flatten(phytomer->leaf_objIDs) ) {
                        for (uint UUID: context_ptr->getObjectPrimitiveUUIDs(leaf_objID)) {
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", 0);
                        }
                    }
                }
            }else{
                for( auto &phytomer: shoot->phytomers ){

                    for( auto &leaf_objID: flatten(phytomer->leaf_objIDs) ){
                        for( uint UUID : context_ptr->getObjectPrimitiveUUIDs(leaf_objID) ){
                            float lUUID_area = context_ptr->getPrimitiveArea(UUID);
                            float leaf_A;
                            context_ptr->getPrimitiveData(UUID,"net_photosynthesis", leaf_A);
                            //std::cout<< "leaf photosynthesis mol C: "<< leaf_A<<std::endl;

                            float new_hourly_photo = leaf_A * lUUID_area * 3600*1e-6; //hourly net photosynthesis (mol C) from umol CO2 m-2 sec-1
                            //std::cout<< "hourly photosynthesis mol C: "<< new_hourly_photo<<std::endl;
                            float current_net_photo;

                            context_ptr->getPrimitiveData(UUID,"cumulative_net_photosynthesis", current_net_photo);
                            current_net_photo += new_hourly_photo;
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", current_net_photo);
                            //std::cout<< "net photo mol C: "<< current_net_photo<<std::endl;

                        }
                    }

                }

            }

        }

    }


}

void PlantArchitecture::accumulateShootPhotosynthesis() {

    uint A_prim_data_missing = 0;

    for( auto &plant: plant_instances ){

        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){

            float net_photosynthesis = 0;

            for( auto &phytomer: shoot->phytomers ){

                for( auto &leaf_objID: flatten(phytomer->leaf_objIDs) ){
                    for( uint UUID : context_ptr->getObjectPrimitiveUUIDs(leaf_objID) ){
                        if( context_ptr->doesPrimitiveDataExist(UUID, "cumulative_net_photosynthesis") && context_ptr->getPrimitiveDataType(UUID,"cumulative_net_photosynthesis")==HELIOS_TYPE_FLOAT ){
                            float A;
                            context_ptr->getPrimitiveData(UUID,"cumulative_net_photosynthesis",A);
                            net_photosynthesis += A;
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", 0.f);
                        }else{
                            A_prim_data_missing++;
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", 0.f);
                        }
                    }
                }

            }

            shoot->carbohydrate_pool_molC += net_photosynthesis;
            //std::cout<< "Net photosynthesis"<< net_photosynthesis<<std::endl;

        }

    }

    if( A_prim_data_missing>0 ){
        std::cerr << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): " << A_prim_data_missing << " leaf primitives were missing net_photosynthesis primitive data. Did you run the photosynthesis model?" << std::endl;
    }

}

void PlantArchitecture::subtractShootMaintenanceCarbon(float dt ) {

    //\todo move to externally settable parameter
    float stem_maintainance_respiration_rate = 1.9458e-05; //mol C respired/mol C in pool/day
    float root_maintainance_respiration_rate = 1.9458e-05/shoot_root_ratio; //mol C respired/mol C in pool/day

    float rho_cw = rho_w * wood_carbon_percentage / C_molecular_wt; //Density of carbon in almond wood (mol C m^-3)


    for (auto &plant: plant_instances) {

        auto shoot_tree = &plant.second.shoot_tree;

        for (auto &shoot: *shoot_tree) {
            if( context_ptr->doesObjectExist(shoot->internode_tube_objID) ) {
                if(shoot->isdormant) {
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            stem_maintainance_respiration_rate *  .2 * dt; //remove shoot maintenance respiration
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            root_maintainance_respiration_rate * .2 * dt; //remove root maintenance respiration portion
                }else{
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            stem_maintainance_respiration_rate * dt; //remove shoot maintenance respiration
                    //std::cout << "shoot stem maintenance: " << context_ptr->getTubeObjectVolume(shoot->internode_tube_objID) * rho_cw * stem_maintainance_respiration_rate * dt << std::endl;
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            root_maintainance_respiration_rate * dt; //remove root maintenance respiration portion
                    //std::cout << "shoot root maintenance: " << context_ptr->getTubeObjectVolume(shoot->internode_tube_objID) * rho_cw * root_maintainance_respiration_rate * dt << std::endl;
                }
            }
        }

    }

}

void PlantArchitecture::subtractShootGrowthCarbon(){

    //\todo move to externally settable parameter
    float rho_cw = rho_w * wood_carbon_percentage / C_molecular_wt; //Density of carbon in almond wood (mol C m^-3)
    float growth_respiration_fraction = 0.28; //Accounting for the growth carbon lost to respiration (assumed 28%)


    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){
            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume();
            std::cout<<"Shoot volume: "<<shoot_volume;

            float shoot_growth_carbon_demand = rho_cw*( shoot_volume - shoot->old_shoot_volume)  * 0.1 * (1+growth_respiration_fraction); //Structural carbon + growth respiration required to construct new shoot volume - mol C / m^3 wood
            shoot->carbohydrate_pool_molC -= shoot_growth_carbon_demand; //Subtract construction carbon + growth respiration from the carbon pool
            //std::cout<<shoot->old_shoot_volume<<"  old shoot volume"<<std::endl;
            shoot->old_shoot_volume = shoot_volume; //Set old volume to the current volume for the next timestep

            std::cout<<"shoot growth demand: " << shoot_growth_carbon_demand<< std::endl;
            std::cout<<"shoot carbohydrate pool mol C: " << shoot->carbohydrate_pool_molC<< std::endl;

            context_ptr->setObjectData( shoot->internode_tube_objID, "carbohydrate_concentration", shoot->carbohydrate_pool_molC / shoot_volume );

            std::cout<<"Shoot Carbohydrate Concentration (mg/g dry weight): "<<(shoot->carbohydrate_pool_molC*12.01*1000) / (shoot_volume*rho_w)<<std::endl;
            std::cout<<"ID "<<shoot->ID<<std::endl;

        }

    }

}


float Phytomer::calculateFruitConstructionCosts(const FloralBud &fbud) {

    //\todo make these values externally settable
    float fruit_construction_cost_base = nut_density*nut_carbon_percentage/C_molecular_wt; //mol C/m^3

    float fruit_carbon_cost = 0.f; //mol C

    //fruit (cost per fruit basis)
    for( uint fruit_objID : fbud.inflorescence_objIDs ) {
        float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID)/fbud.current_fruit_scale_factor; //mature fruit volume
        fruit_carbon_cost += fruit_construction_cost_base*(mature_volume)*(fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor);
    }

    return fruit_carbon_cost;
}

void PlantArchitecture::checkCarbonPool_abortOrgans(){
    float carbohydrate_threshold = 10*rho_w/(1000*C_molecular_wt); //mol C/m3
    float bud_death_threshold = 6; //days
    float branch_death_threshold = 16; //days

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
            shoot->days_with_negative_carbon_balance += 1;

            if(shoot->carbohydrate_pool_molC > carbohydrate_threshold*shoot_volume){
                if(parentID <1000000){
                    shoot->days_with_negative_carbon_balance = 0;
                    goto shoot_balanced;
                }

            }else if(shoot->days_with_negative_carbon_balance > branch_death_threshold){
                pruneBranch(plantID, shootID, 0);
                goto shoot_balanced;
            }else if(shoot->days_with_negative_carbon_balance > bud_death_threshold) {
                auto phytomers = &shoot->phytomers;

                bool living_buds = true;

                while (living_buds) {

                    living_buds = false;

                    for (auto &phytomer: *phytomers) {
                        bool next_phytomer = false;


                        for (auto &petiole: phytomer->floral_buds) {
                            bool next_petiole = false;

                            if(next_phytomer){
                                break;
                            }else{
                                for (auto &fbud: petiole) {

                                    if (next_petiole){
                                        break;
                                    }else{
                                        if (fbud.state != BUD_DORMANT && fbud.state != BUD_DEAD) {
                                            for (uint fruit_objID: fbud.inflorescence_objIDs) {
                                                float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID) /
                                                                      fbud.current_fruit_scale_factor; //mature fruit volume
                                                working_carb_pool += fruit_construction_cost_base * (mature_volume) *
                                                                     (fbud.current_fruit_scale_factor -
                                                                      fbud.previous_fruit_scale_factor);
                                            }
                                            phytomer->setFloralBudState(BUD_DEAD, fbud); //Kill a floral bud to eliminate it as a future sink

                                            if (working_carb_pool > carbohydrate_threshold) {
                                                goto shoot_balanced;
                                            } //If the amount of carbon you've eliminated by aborting flower buds would have given you a positive carbon balance, move on to the next shoot

                                            living_buds = true; //There was at least one living bud, so stay in the loop until there aren't any more
                                            next_petiole = true; //As soon as you've eliminated one bud from a given petiole, move to the next one
                                            next_phytomer = true; //As soon as you've eliminated one bud from a given phytomer, move to the next one
                                        }

                                    }

                                }

                            }

                        }

                    }

                }

            }
            shoot_balanced:
            ; //empty statement after the label to avoid a compiler warning
        }

    }

}


void PlantArchitecture::checkCarbonPool_adjustPhyllochron(){
    float carbohydrate_threshold = 100*rho_w/(1000*C_molecular_wt); //mol C/m3
    float carbohydrate_threshold_low = 50*rho_w/(1000*C_molecular_wt); //mol C/m3

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;
        auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

        for( auto &shoot: *shoot_tree ){

            uint shootID = shoot->ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();
            float working_carb_pool = shoot->carbohydrate_pool_molC;
            uint parentID = shoot->parent_shoot_ID;

            if(shoot->carbohydrate_pool_molC > carbohydrate_threshold*shoot_volume){
                if(shoot->phyllochron_instantaneous > shoot->shoot_parameters.phyllochron_min.val() * shoot->phyllochron_recovery){
                    shoot->phyllochron_instantaneous = shoot->phyllochron_instantaneous / shoot->phyllochron_recovery;
                }else{
                    shoot->phyllochron_instantaneous = shoot->shoot_parameters.phyllochron_min.val();
                }

                if(shoot->elongation_rate_instantaneous < shoot->shoot_parameters.elongation_rate_max.val() * shoot->elongation_recovery){
                    shoot->elongation_rate_instantaneous = shoot->elongation_rate_instantaneous / shoot->elongation_recovery;
                }else{
                    shoot->elongation_rate_instantaneous = shoot->shoot_parameters.elongation_rate_max.val();
                }

            }else if (shoot->carbohydrate_pool_molC < carbohydrate_threshold_low*shoot_volume) {
                shoot->elongation_rate_instantaneous = shoot->shoot_parameters.elongation_rate_max.val() * .05;
                shoot->phyllochron_instantaneous = shoot->shoot_parameters.phyllochron_min.val() * 20;

            }else{
                if(shoot->phyllochron_instantaneous <= shoot->shoot_parameters.phyllochron_min.val() * 20){
                    shoot->phyllochron_instantaneous = shoot->phyllochron_instantaneous * shoot->phyllochron_increase;
                }
                if(shoot->elongation_rate_instantaneous >= shoot->shoot_parameters.elongation_rate_max.val()*0.05){
                    shoot->elongation_rate_instantaneous = shoot->elongation_rate_instantaneous * shoot->elongation_decay;
                }
            }

        }

    }

}

void PlantArchitecture::checkCarbonPool_transferCarbon(){
    float carbohydrate_threshold = 50*rho_w/(1000*C_molecular_wt); //mol C/m3
    float storage_conductance = 0.75;
    float return_conductance = 0.75;

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;
        auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

        for( auto &shoot: *shoot_tree ){

            uint shootID = shoot->ID;
            uint parentID = shoot->parent_shoot_ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();
            float shoot_carb_pool_molC = shoot->carbohydrate_pool_molC;
            float shoot_carb_conc = shoot_carb_pool_molC / shoot_volume;



            if(shoot_carb_pool_molC > carbohydrate_threshold*shoot_volume){
                if(parentID < 10000000 ){
                    float parent_shoot_volume = shoot_tree_ptr->at(parentID)->calculateShootInternodeVolume();
                    float parent_shoot_carb_pool_molC = shoot_tree_ptr->at(parentID)->carbohydrate_pool_molC;
                    float parent_shoot_carb_conc = parent_shoot_carb_pool_molC / parent_shoot_volume;

                    float transfer_mol_C = (shoot_carb_pool_molC - carbohydrate_threshold * shoot_volume) * storage_conductance * ((shoot_carb_conc - parent_shoot_carb_conc) / shoot_carb_conc);
                    shoot_tree_ptr->at(parentID)->carbohydrate_pool_molC +=
                            transfer_mol_C;
                    shoot->carbohydrate_pool_molC -=
                            transfer_mol_C;

                }

            }

        }

        for( auto &shoot: *shoot_tree ){
            uint shootID = shoot->ID;
            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();
            float shoot_carb_pool_molC = shoot->carbohydrate_pool_molC;
            float shoot_carb_conc = shoot_carb_pool_molC / shoot_volume;
            if(shoot->carbohydrate_pool_molC > carbohydrate_threshold*shoot_volume) {

                float totalChildVolume = shoot->sumChildVolume(0);
                //std::cout<<"child volume: "<<totalChildVolume<<std::endl;

                for (uint p = 0; p < shoot->phytomers.size(); p++) {
                    //call recursively for child shoots
                    if (shoot->childIDs.find(p) != shoot->childIDs.end()) {
                        for (int child_shoot_ID: shoot->childIDs.at(p)) {
                            float child_volume = plant_instances.at(plantID).shoot_tree.at(
                                    child_shoot_ID)->sumChildVolume(0)+plant_instances.at(plantID).shoot_tree.at(
                                    child_shoot_ID)->calculateShootInternodeVolume();
                            float child_ratio = child_volume / totalChildVolume;

                            float child_shoot_volume = plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->calculateShootInternodeVolume();
                            float child_shoot_carb_pool_molC = shoot_tree_ptr->at(child_shoot_ID)->carbohydrate_pool_molC;
                            float child_shoot_carb_conc = child_shoot_carb_pool_molC / child_shoot_volume;
                            //std::cout<<"child ratio: "<<child_ratio<<std::endl;

                            float transfer_mol_C = (shoot->carbohydrate_pool_molC - carbohydrate_threshold * shoot_volume)*return_conductance*child_ratio * ((shoot_carb_conc - child_shoot_carb_conc) / shoot_carb_conc);

                            shoot_tree_ptr->at(child_shoot_ID)->carbohydrate_pool_molC +=
                                    transfer_mol_C;
                            shoot->carbohydrate_pool_molC -=
                                    transfer_mol_C;
                            //std::cout<<"Carb to child: "<<(shoot->carbohydrate_pool_molC - carbohydrate_threshold*shoot_volume)*return_conductance*child_ratio<<std::endl;
                        }
                    }

                }
            }

        }

    }

}