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

void PlantArchitecture::setPlantCarbohydrateParameters(uint plantID, float rho_w, float wood_carbon_percentage,
    float shoot_root_ratio, float SLA, float leaf_carbon_percentage, float total_flower_cost,float fruit_density,
    float fruit_carbon_percentage, float stem_maintainance_respiration_rate, float root_maintainance_respiration_rate,
    float growth_respiration_fraction, float carbohydrate_abortion_threshold, float bud_death_threshold, float branch_death_threshold,
    float carbohydrate_phyllochron_threshold, float carbohydrate_phyllochron_threshold_low, float carbohydrate_transfer_threshold,
    float carbon_conductance){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantCarbohydrateParameters): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    //Stem Growth
    plant_instances.at(plantID).stem_density = rho_w;
    plant_instances.at(plantID).stem_carbon_percentage = wood_carbon_percentage;
    plant_instances.at(plantID).shoot_root_ratio = shoot_root_ratio;
    //Leaf Growth
    plant_instances.at(plantID).SLA = SLA;
    plant_instances.at(plantID).leaf_carbon_percentage = leaf_carbon_percentage;
    plant_instances.at(plantID).total_flower_cost = total_flower_cost;
    //Fruit Growth
    plant_instances.at(plantID).fruit_density = fruit_density;
    plant_instances.at(plantID).fruit_carbon_percentage = fruit_carbon_percentage;
    //Respiration
    plant_instances.at(plantID).stem_maintainance_respiration_rate = stem_maintainance_respiration_rate;
    plant_instances.at(plantID).root_maintainance_respiration_rate = root_maintainance_respiration_rate;
    plant_instances.at(plantID).growth_respiration_fraction = growth_respiration_fraction;
    //Abortion of Organs
    plant_instances.at(plantID).carbohydrate_abortion_threshold = carbohydrate_abortion_threshold;
    plant_instances.at(plantID).bud_death_threshold = bud_death_threshold;
    plant_instances.at(plantID).branch_death_threshold = branch_death_threshold;
    //Phyllochron Adjustment
    plant_instances.at(plantID).carbohydrate_phyllochron_threshold = carbohydrate_phyllochron_threshold;
    plant_instances.at(plantID).carbohydrate_phyllochron_threshold_low = carbohydrate_phyllochron_threshold_low;
    //Carbon Transfer
    plant_instances.at(plantID).carbohydrate_transfer_threshold = carbohydrate_transfer_threshold;
    plant_instances.at(plantID).carbon_conductance = carbon_conductance;
}



float Phytomer::calculatePhytomerConstructionCosts(){

    //\todo make these values externally settable
    float leaf_construction_cost_base = plantarchitecture_ptr->plant_instances.at(this->plantID).leaf_carbon_percentage / (C_molecular_wt * plantarchitecture_ptr->plant_instances.at(this->plantID).SLA); //mol C/m^2
    float internode_construction_cost_base = plantarchitecture_ptr->plant_instances.at(this->plantID).stem_density * plantarchitecture_ptr->plant_instances.at(this->plantID).stem_carbon_percentage / C_molecular_wt; // (mol C /m^3)

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

    float flower_carbon_cost = 0.f; //mol C

    for( uint flower_objID : fbud.inflorescence_objIDs ) {
        flower_carbon_cost += plantarchitecture_ptr->plant_instances.at(this->plantID).total_flower_cost;
    }
    std::cout<<"Flower carbon cost: "<<flower_carbon_cost<<std::endl;
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

                            float new_hourly_photo = leaf_A * lUUID_area * 3600*1e-6; //hourly net photosynthesis (mol C) from umol CO2 m-2 sec-1
                            //std::cout<< "hourly photosynthesis mol C: "<< new_hourly_photo<<std::endl;
                            float current_net_photo;

                            context_ptr->getPrimitiveData(UUID,"cumulative_net_photosynthesis", current_net_photo);
                            current_net_photo += new_hourly_photo;
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", current_net_photo);

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
            std::cout<< "Net photosynthesis"<< net_photosynthesis<<std::endl;
            //std::cout<<"shoot carbohydrate pool mol C/m3: " << shoot->carbohydrate_pool_molC / shoot->calculateShootInternodeVolume()<< std::endl;

        }

    }

    if( A_prim_data_missing>0 ){
        std::cerr << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): " << A_prim_data_missing << " leaf primitives were missing net_photosynthesis primitive data. Did you run the photosynthesis model?" << std::endl;
    }

}

void PlantArchitecture::subtractShootMaintenanceCarbon( float dt ) {

    for (auto &plant: plant_instances) {

        auto shoot_tree = &plant.second.shoot_tree;

        uint plantID = plant.first;
        float rho_cw = plant_instances.at(plantID).stem_density * plant_instances.at(plantID).stem_carbon_percentage / C_molecular_wt; //Density of carbon in almond wood (mol C m^-3)

        for (auto &shoot: *shoot_tree) {
            if( context_ptr->doesObjectExist(shoot->internode_tube_objID) ) {
                if(shoot->isdormant) {
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            plant_instances.at(plantID).stem_maintainance_respiration_rate *  .2 * dt; //remove shoot maintenance respiration
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            plant_instances.at(plantID).root_maintainance_respiration_rate * .2 * dt; //remove root maintenance respiration portion
                }else{
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            plant_instances.at(plantID).stem_maintainance_respiration_rate * dt; //remove shoot maintenance respiration
                    std::cout << "shoot stem maintenance: " << context_ptr->getTubeObjectVolume(shoot->internode_tube_objID) * rho_cw * plant_instances.at(plantID).stem_maintainance_respiration_rate * dt << std::endl;
                    shoot->carbohydrate_pool_molC -=
                            shoot->old_shoot_volume * rho_cw *
                            plant_instances.at(plantID).root_maintainance_respiration_rate * dt; //remove root maintenance respiration portion
                    std::cout << "shoot root maintenance: " << context_ptr->getTubeObjectVolume(shoot->internode_tube_objID) * rho_cw * plant_instances.at(plantID).root_maintainance_respiration_rate * dt << std::endl;
                }
            }
        }

    }

}

void PlantArchitecture::subtractShootGrowthCarbon(){

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        float rho_cw = plant_instances.at(plantID).stem_density * plant_instances.at(plantID).stem_carbon_percentage / C_molecular_wt; //Density of carbon in almond wood (mol C m^-3)

        for( auto &shoot: *shoot_tree ){
            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume();
            float shoot_growth_carbon_demand = rho_cw*( shoot_volume - shoot->old_shoot_volume) * 0.1 * (1+plant_instances.at(plantID).growth_respiration_fraction); //Structural carbon + growth respiration required to construct new shoot volume - mol C / m^3 wood
            shoot->carbohydrate_pool_molC -= shoot_growth_carbon_demand; //Subtract construction carbon + growth respiration from the carbon pool
            shoot->carbohydrate_pool_molC -= shoot_growth_carbon_demand / plant_instances.at(plantID).shoot_root_ratio; //Subtract construction carbon + growth respiration for the roots from the carbon pool


            std::cout<<"growth demand: " << shoot_growth_carbon_demand + shoot_growth_carbon_demand / plant_instances.at(plantID).shoot_root_ratio<< std::endl;
            std::cout<<"shoot carbohydrate pool mol C: " << shoot->carbohydrate_pool_molC<< std::endl;

            context_ptr->setObjectData( shoot->internode_tube_objID, "carbohydrate_concentration", shoot->carbohydrate_pool_molC / shoot_volume );

            std::cout<<"Shoot Carbohydrate Concentration (mg/g dry weight): "<<(shoot->carbohydrate_pool_molC*12.01*1000) / (shoot_volume*plant_instances.at(plantID).stem_density)<<std::endl;
            std::cout<<"ID "<<shoot->ID<<std::endl;

        }

    }

}


float Phytomer::calculateFruitConstructionCosts(const FloralBud &fbud) {

    //\todo make these values externally settable
    float fruit_construction_cost_base = plantarchitecture_ptr->plant_instances.at(this->plantID).fruit_density * plantarchitecture_ptr->plant_instances.at(this->plantID).fruit_carbon_percentage/C_molecular_wt; //mol C/m^3

    float fruit_carbon_cost = 0.f; //mol C

    //fruit (cost per fruit basis)
    for( uint fruit_objID : fbud.inflorescence_objIDs ) {
        float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID)/fbud.current_fruit_scale_factor; //mature fruit volume
        fruit_carbon_cost += fruit_construction_cost_base*(mature_volume)*(fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor);
    }
    std::cout<<"Fruit carbon cost: "<<fruit_carbon_cost<<std::endl;

    return fruit_carbon_cost;
}

void PlantArchitecture::checkCarbonPool_abortOrgans(float dt){

    //\todo make these values externally settable
    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        float fruit_construction_cost_base = plant_instances.at(plantID).fruit_density*plant_instances.at(plantID).fruit_carbon_percentage/C_molecular_wt; //mol C/m^3


        for( auto &shoot: *shoot_tree ){

            uint shootID = shoot->ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();
            float working_carb_pool = shoot->carbohydrate_pool_molC;
            if (dt > plant_instances.at(plantID).bud_death_threshold)
            {
                shoot->days_with_negative_carbon_balance += plant_instances.at(plantID).bud_death_threshold/2; //Make sure you have at least two timesteps before starting to abort buds
            }else
            {
                shoot->days_with_negative_carbon_balance += dt;
            }

            if(shoot->carbohydrate_pool_molC > plant_instances.at(plantID).carbohydrate_abortion_threshold*shoot_volume){
                shoot->days_with_negative_carbon_balance = 0;
                goto shoot_balanced;

            }else if(shoot->days_with_negative_carbon_balance > plant_instances.at(plantID).branch_death_threshold){
                pruneBranch(plantID, shootID, 0);
                goto shoot_balanced;
            }else if(shoot->days_with_negative_carbon_balance > plant_instances.at(plantID).bud_death_threshold) {
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

                                            if (working_carb_pool > plant_instances.at(plantID).carbohydrate_abortion_threshold) {
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

        }

        shoot_balanced:
        ; //empty statement after the label to avoid a compiler warning

    }

}


void PlantArchitecture::checkCarbonPool_adjustPhyllochron(float dt){

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){

            uint shootID = shoot->ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();

            if(shoot->carbohydrate_pool_molC > plant_instances.at(plantID).carbohydrate_phyllochron_threshold*shoot_volume){
                if(shoot->phyllochron_instantaneous > shoot->shoot_parameters.phyllochron_min.val() * shoot->phyllochron_recovery * dt){
                    shoot->phyllochron_instantaneous = shoot->phyllochron_instantaneous / (shoot->phyllochron_recovery*dt);
                }else{
                    shoot->phyllochron_instantaneous = shoot->shoot_parameters.phyllochron_min.val();
                }

            }else if (shoot->carbohydrate_pool_molC < plant_instances.at(plantID).carbohydrate_phyllochron_threshold_low*shoot_volume) {
                shoot->phyllochron_instantaneous = shoot->shoot_parameters.phyllochron_min.val() * 5;

            }else{
                if(shoot->phyllochron_instantaneous <= shoot->shoot_parameters.phyllochron_min.val() * 5){
                    shoot->phyllochron_instantaneous = shoot->phyllochron_instantaneous * shoot->phyllochron_increase;
                }
            }

            std::cout<<"Phyllochron: "<<shoot->phyllochron_instantaneous<<std::endl;

        }

    }

}

void PlantArchitecture::checkCarbonPool_transferCarbon(float dt){
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



            if(shoot_carb_pool_molC > plant_instances.at(plantID).carbohydrate_transfer_threshold*shoot_volume){
                float available_fraction_of_carb = (shoot->carbohydrate_pool_molC - plant_instances.at(plantID).carbohydrate_transfer_threshold * shoot_volume)/shoot->carbohydrate_pool_molC;
                if(parentID < 10000000 ){
                    float parent_shoot_volume = shoot_tree_ptr->at(parentID)->calculateShootInternodeVolume();
                    float parent_shoot_carb_pool_molC = shoot_tree_ptr->at(parentID)->carbohydrate_pool_molC;
                    float parent_shoot_carb_conc = parent_shoot_carb_pool_molC / parent_shoot_volume;

                    if (shoot_carb_conc > parent_shoot_carb_conc)
                    {
                        float delta_C = shoot_carb_conc - parent_shoot_carb_conc;
                        float transfer_mol_C = delta_C * available_fraction_of_carb * shoot_volume * plant_instances.at(plantID).carbon_conductance;
                        shoot_tree_ptr->at(parentID)->carbohydrate_pool_molC +=
                                transfer_mol_C;
                        shoot->carbohydrate_pool_molC -=
                                transfer_mol_C;

                        std::cout<< "Transferred Carbon " << transfer_mol_C << " shoot " << shootID << "to" << parentID << std::endl;
                        //std::cout << shoot_carb_conc << std::endl;
                    }

                }

            }

        }

        for( auto &shoot: *shoot_tree ){
            uint shootID = shoot->ID;
            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();
            float shoot_carb_pool_molC = shoot->carbohydrate_pool_molC;
            float shoot_carb_conc = shoot_carb_pool_molC / shoot_volume;
            if(shoot->carbohydrate_pool_molC > plant_instances.at(plantID).carbohydrate_transfer_threshold*shoot_volume) {

                float totalChildVolume = shoot->sumChildVolume(0);
                float available_fraction_of_carb = (shoot->carbohydrate_pool_molC - plant_instances.at(plantID).carbohydrate_transfer_threshold * shoot_volume) / shoot->carbohydrate_pool_molC;

                for (uint p = 0; p < shoot->phytomers.size(); p++) {
                    //call recursively for child shoots
                    if (shoot->childIDs.find(p) != shoot->childIDs.end()) {
                        for (int child_shoot_ID: shoot->childIDs.at(p)) {
                            float child_volume = plant_instances.at(plantID).shoot_tree.at(
                                    child_shoot_ID)->sumChildVolume(0)+plant_instances.at(plantID).shoot_tree.at(
                                    child_shoot_ID)->calculateShootInternodeVolume();
                            float child_ratio = child_volume / totalChildVolume;

                            float child_shoot_volume = plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->calculateShootInternodeVolume();

                            if (child_shoot_volume > 0){
                                float child_shoot_carb_pool_molC = shoot_tree_ptr->at(child_shoot_ID)->carbohydrate_pool_molC;
                                float child_shoot_carb_conc = child_shoot_carb_pool_molC / child_shoot_volume;
                                std::cout<<"child ratio: "<<child_ratio<<std::endl;
                                if (shoot_carb_conc > child_shoot_carb_conc){
                                    float delta_C = shoot_carb_conc - child_shoot_carb_conc;
                                    float transfer_mol_C = delta_C * available_fraction_of_carb * child_ratio * child_shoot_volume * plant_instances.at(plantID).carbon_conductance;
                                    shoot_tree_ptr->at(child_shoot_ID)->carbohydrate_pool_molC +=
                                        transfer_mol_C;
                                    shoot->carbohydrate_pool_molC -=
                                        transfer_mol_C;

                                    std::cout<<"Carb to child: "<<transfer_mol_C<< " shoot " << shootID<< "to" << child_shoot_ID<<std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}