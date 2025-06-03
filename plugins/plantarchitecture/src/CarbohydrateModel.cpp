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


/**
 Calculate the total carbon cost (mol C) required for the construction of a phytomer's total leaf area.
 Carbon construction cost is calculated per area basis using the leaf carbon percentage and specific leaf area (SLA)
 of the plant instance.

 @return The total carbon construction cost of the phytomer's leaf area (mol C).
 */
float Phytomer::calculatePhytomerConstructionCosts() const {

    float leaf_carbon_percentage = plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters.leaf_carbon_percentage;
    float SLA = plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters.SLA;

    float leaf_construction_cost_base = leaf_carbon_percentage/(C_molecular_wt*SLA); //mol C/m^2

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

/**
 Calculate the total carbon cost (mol C) of constructing the flowers for a given FloralBud object.
 Iterates over all flower object IDs stored in the FloralBud instance and accumulates the total construction cost.

 @param fbud References a FloralBud object that contains inflorescence object IDs.
 @return The total flower carbon construction cost (mol C).
 */
float Phytomer::calculateFlowerConstructionCosts(const FloralBud &fbud) const {

    float flower_carbon_cost = 0.f; //mol C

    for( uint flower_objID : fbud.inflorescence_objIDs ) {
        flower_carbon_cost += plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters.total_flower_cost;
    }

    return flower_carbon_cost;
}

/**
 Initialize the carbohydrate pool for all shoots in the plant architecture.
 Updates the carbohydrate concentration of each internode in the context.

 @param carbohydrate_concentration_molC_m3 The carbohydrate concentration of the shoot (mol C / m^3).
 */
void PlantArchitecture::initializeCarbohydratePool(float carbohydrate_concentration_molC_m3) const {

    for( auto &plant: plant_instances ) {

        auto shoot_tree = &plant.second.shoot_tree;

        for (auto &shoot: *shoot_tree) {
            //calculate shoot volume
            float shoot_volume = shoot->calculateShootInternodeVolume();
            //set carbon pool
            shoot->carbohydrate_pool_molC = shoot_volume * carbohydrate_concentration_molC_m3;
            context_ptr->setObjectData( shoot->internode_tube_objID, "carbohydrate_concentration", carbohydrate_concentration_molC_m3);

        }
    }
}

/**
Initialize the carbohydrate pool for a given plant by setting an initial carbohydrate concentration for each shoot
in the plant's shoot tree.

@param plantID The UUID of the plant
@param carbohydrate_concentration_molC_m3 The initial carbohydrate concentration to be set (mol C / m^3).
Must be greater than or equal to zero.

@throw helios_runtime_error If the plant with the given ID does not exist or if the provided carbohydrate concentration
is less than zero.
 */
void PlantArchitecture::initializePlantCarbohydratePool(uint plantID, float carbohydrate_concentration_molC_m3 ){

    //Make sure that the plant exists in the context
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

/**
Initialize the carbohydrate pool in a specific shoot of a plant.

@param plantID UUID of the plant to which the shoot belongs.
@param shootID UUID of the shoot for which the carbohydrate pool will be initialized.
@param carbohydrate_concentration_molC_m3 The concentration of carbohydrates (mol C / m^3) to be set for the shoot.
Must be greater than or equal to zero.

@throws std::runtime_error If the specified plant or shoot does not exist.
@throws std::runtime_error If the carbohydrate concentration is negative.
 */
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

/**
Calculate the hourly net photosynthesis (mol C) for each leaf in the context based on its area and photosynthesis rate.

Hourly net photosynthesis calculated as:
photosynthesisRate * leafArea * secondsInHour * unitConversionFactor
    -secondsInHour = 3600
    -unitConversionFactor = 1e-6 for conversion from micromoles CO2 m^-2 sec^-1 to mol C.
*/
void PlantArchitecture::accumulateHourlyLeafPhotosynthesis() const {

    for( const auto &[plantID, plant_instance]: plant_instances ){

        auto shoot_tree = &plant_instance.shoot_tree;

        for( auto &shoot: *shoot_tree ){
            //Set cumulative photosynthesis of dormant shoots equal to zero.
            if(shoot->isdormant) {
                for(const auto &phytomer: shoot->phytomers ){
                    for(const auto &leaf_objID: flatten(phytomer->leaf_objIDs) ) {
                        for (uint UUID: context_ptr->getObjectPrimitiveUUIDs(leaf_objID)) {
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", 0);
                        }
                    }
                }
            }else{
                for(const auto &phytomer: shoot->phytomers ){

                    for( auto &leaf_objID: flatten(phytomer->leaf_objIDs) ){
                        for( uint UUID : context_ptr->getObjectPrimitiveUUIDs(leaf_objID) ){
                            float lUUID_area = context_ptr->getPrimitiveArea(UUID);
                            float leaf_A = 0.f;
                            if ( context_ptr->doesPrimitiveDataExist(UUID, "net_photosynthesis") && context_ptr->getPrimitiveDataType(UUID, "net_photosynthesis") == HELIOS_TYPE_FLOAT ) {
                                context_ptr->getPrimitiveData(UUID,"net_photosynthesis", leaf_A);
                            }

                            float new_hourly_photo = leaf_A * lUUID_area * 3600.f* 1e-6f;; //hourly net photosynthesis (mol C) from umol CO2 m-2 sec-1
                            //std::cout<< "hourly photosynthesis mol C: "<< new_hourly_photo<<std::endl;
                            float current_net_photo = 0.f;
                            if ( context_ptr->doesPrimitiveDataExist(UUID, "cumulative_net_photosynthesis") && context_ptr->getPrimitiveDataType(UUID, "cumulative_net_photosynthesis") == HELIOS_TYPE_FLOAT ) {
                                context_ptr->getPrimitiveData(UUID,"cumulative_net_photosynthesis", current_net_photo);
                            }
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

/**
Accumulate net photosynthesis values for each shoot in the context and update the carbohydrate pool
for that shoot.
*/
void PlantArchitecture::accumulateShootPhotosynthesis() const {

    uint A_prim_data_missing = 0;

    for( const auto &[plantID, plant_instance]: plant_instances ){

        const auto shoot_tree = &plant_instance.shoot_tree;

        for(const auto &shoot: *shoot_tree ){

            float net_photosynthesis = 0;
            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume();

            if (shoot->isdormant)
            {
                for(const auto &phytomer: shoot->phytomers ){
                    for(const auto &leaf_objID: flatten(phytomer->leaf_objIDs) ) {
                        for (uint UUID: context_ptr->getObjectPrimitiveUUIDs(leaf_objID)) {
                            context_ptr->setPrimitiveData(UUID, "cumulative_net_photosynthesis", 0.f);
                        }
                    }
                }
            }else
            {
                for(const auto &phytomer: shoot->phytomers ){

                    for(const auto &leaf_objID: flatten(phytomer->leaf_objIDs) ){
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
            }
            if (net_photosynthesis >= 0.f)
            {
                shoot->carbohydrate_pool_molC += net_photosynthesis;
            }
            context_ptr->setObjectData( shoot->internode_tube_objID, "carbohydrate_concentration", shoot->carbohydrate_pool_molC / shoot_volume );
            //std::cout<< "Net photosynthesis"<< net_photosynthesis<< " Shoot: " << shoot->ID <<std::endl;

        }

    }

    if( A_prim_data_missing>0 ){
        std::cerr << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): " << A_prim_data_missing << " leaf primitives were missing net_photosynthesis primitive data. Did you run the photosynthesis model?" << std::endl;
    }

}

/**
Calculate the maintenance respiration needed for a given shoot, based on the volume of the shoot,
carbon density of wood, and respiration rates. Remove respiration carbon from the shoot.

@param dt Time step (days).
*/
void PlantArchitecture::subtractShootMaintenanceCarbon(float dt ) const {

    for (const auto &[plantID, plant_instance]: plant_instances) {

        auto shoot_tree = &plant_instance.shoot_tree;

        const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;

        float rho_cw = carbohydrate_params.stem_density * carbohydrate_params.stem_carbon_percentage / C_molecular_wt; //Density of carbon in almond wood (mol C m^-3)

        for (auto &shoot: *shoot_tree) {
            if( context_ptr->doesObjectExist(shoot->internode_tube_objID) ) {
                if(shoot->isdormant && shoot->old_shoot_volume >= 0.f) {
                    shoot->carbohydrate_pool_molC -= shoot->old_shoot_volume * rho_cw *
                            carbohydrate_params.stem_maintainance_respiration_rate *  0.2f * dt; //remove shoot maintenance respiration
                    shoot->carbohydrate_pool_molC -= shoot->old_shoot_volume * rho_cw *
                            carbohydrate_params.root_maintainance_respiration_rate / carbohydrate_params.shoot_root_ratio * 0.2f * dt; //remove root maintenance respiration portion
                }else if(shoot->old_shoot_volume >= 0.f){
                    shoot->carbohydrate_pool_molC -= shoot->old_shoot_volume * rho_cw *
                            carbohydrate_params.stem_maintainance_respiration_rate * dt; //remove shoot maintenance respiration
                    shoot->carbohydrate_pool_molC -= shoot->old_shoot_volume * rho_cw *
                            carbohydrate_params.root_maintainance_respiration_rate / carbohydrate_params.shoot_root_ratio * dt; //remove root maintenance respiration portion
                }
            }
        }

    }

}

/**
Compute growth carbon demand and adjust the carbon balance of a shoot based on its change in volume between timesteps.

1. Iterate through all plant instances and their shoot trees.
2. Calculate the mature and dynamic carbon densities for stem wood.
3. For each phytomer in a shoot:
    - Compute its volume and age.
    - Calculate the dynamic carbon demand for its growth using the dynamic density and volume difference between timesteps.
    - Reduce the carbohydrate pool of the shoot based on the carbon demand for both shoots and roots
    (based on shoot to root ratio).
    - Update the old volume of the phytomer for the next timestep.
*/
void PlantArchitecture::subtractShootGrowthCarbon(){

    for( auto &[plantID, plant_instance]: plant_instances ){

        const auto shoot_tree = &plant_instance.shoot_tree;

        const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;

        float rho_cw = carbohydrate_params.stem_density * carbohydrate_params.stem_carbon_percentage / C_molecular_wt; //Mature density of carbon in almond wood (mol C m^-3)

        for(const auto &shoot: *shoot_tree ){
            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume();
            uint parentID = shoot->parent_shoot_ID;


            for ( int p = 0; p < shoot->phytomers.size(); p++){
                    float phytomer_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->phytomers.at(p)->calculatePhytomerVolume(p);
                    float phytomer_age = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->phytomers.at(p)->age;
                    float density_dynamic = phytomer_age/carbohydrate_params.maturity_age;
                    //Clamp dynamic carbon density between minimum value and density at full maturity
                    float rho_cw_dynamic = rho_cw*std::clamp(density_dynamic, carbohydrate_params.initial_density_ratio, 1.f); //Carbon density of the stem for the given phytomer (mol C / m^3 wood)
                    float phytomer_growth_carbon_demand = 0.f;
                    if (plant_instances.at(plantID).shoot_tree.at(shoot->ID)->old_shoot_volume >= 0.f)
                    {
                        phytomer_growth_carbon_demand = rho_cw_dynamic*( phytomer_volume - plant_instances.at(plantID).shoot_tree.at(shoot->ID)->phytomers.at(p)->old_phytomer_volume); //Structural carbon - mol C / m^3 wood
                        shoot->carbohydrate_pool_molC -= phytomer_growth_carbon_demand; //Subtract construction carbon from the shoot's carbon pool
                        shoot->carbohydrate_pool_molC -= phytomer_growth_carbon_demand / carbohydrate_params.shoot_root_ratio; //Subtract construction carbon for the roots from the carbon pool
                    }

                    plant_instances.at(plantID).shoot_tree.at(shoot->ID)->phytomers.at(p)->old_phytomer_volume = phytomer_volume; //Update the old volume of the phytomer
            }

            //Update shoot's carbohydrate_concentration value (mol C / m^-3)
            if ( context_ptr->doesObjectExist(shoot->internode_tube_objID)  ) {
                context_ptr->setObjectData( shoot->internode_tube_objID, "carbohydrate_concentration", shoot->carbohydrate_pool_molC / shoot_volume );
            }

        }

    }

}


/**
Calculate the carbohydrate construction cost of fruits (mol C) by comparing volume change between timesteps.
*/
float Phytomer::calculateFruitConstructionCosts(const FloralBud &fbud) const {

    const CarbohydrateParameters &carbohydrate_params = plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters;

    float fruit_construction_cost_base = carbohydrate_params.fruit_density * carbohydrate_params.fruit_carbon_percentage/C_molecular_wt; //mol C/m^3

    float fruit_carbon_cost = 0.f; //mol C

    //fruit (cost per fruit basis)
    for( uint fruit_objID : fbud.inflorescence_objIDs ) {
        float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID)/fbud.current_fruit_scale_factor; //mature fruit volume
        fruit_carbon_cost += fruit_construction_cost_base*(mature_volume)*(fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor);
    }
    return fruit_carbon_cost;
}

/**
Check the carbon pool of individual shoots and abort organs or prune branches if carbohydrate levels fall
below specific thresholds in order to maintain sustainable growth and plant survival.

@param dt Time step (days)
*/
void PlantArchitecture::checkCarbonPool_abortOrgans(float dt){

    for( auto &[plantID, plant_instance]: plant_instances ){

        const auto shoot_tree = &plant_instance.shoot_tree;
        const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;

        float fruit_construction_cost_base = carbohydrate_params.fruit_density*carbohydrate_params.fruit_carbon_percentage/C_molecular_wt; //mol C/m^3

        for(const auto &shoot: *shoot_tree ){

            uint shootID = shoot->ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();
            //Establish a working carbon pool to see if you could stay above the carbon threshold by aborting fruits.
            float working_carb_pool = shoot->carbohydrate_pool_molC;
            if (dt > carbohydrate_params.bud_death_threshold_days)
            {
                shoot->days_with_negative_carbon_balance += dt/2; //Make sure you have at least two timesteps before starting to abort buds
            }else
            {
                shoot->days_with_negative_carbon_balance += dt;
            }

            //No need to mess with anything if you already have a carbon surplus
            if(shoot->carbohydrate_pool_molC > carbohydrate_params.carbohydrate_abortion_threshold*carbohydrate_params.stem_density * shoot_volume /C_molecular_wt) {
                shoot->days_with_negative_carbon_balance = 0;
                goto shoot_balanced;
            }
            //Prevent any shoots from reaching negative carbon values: instant death
            if (shoot->carbohydrate_pool_molC < 0.f)
            {
                pruneBranch(plantID, shootID, 0);
                goto shoot_balanced;
            }
            //Prune branches that can't stay above a sustainable threshold
            if (shoot->carbohydrate_pool_molC < carbohydrate_params.carbohydrate_pruning_threshold*carbohydrate_params.stem_density * shoot_volume /C_molecular_wt && shoot->days_with_negative_carbon_balance > carbohydrate_params.branch_death_threshold_days)
            {
                pruneBranch(plantID, shootID, 0);
                goto shoot_balanced;
            }
            //Keep track of how many days the shoot has been below the threshold
            if(shoot->days_with_negative_carbon_balance <= carbohydrate_params.bud_death_threshold_days) {
                continue;
            }

            //Loop over fruiting buds and abort them one at a time until you would be able to stay above the threshold
            const auto phytomers = &shoot->phytomers;

            bool living_buds = true;
            while (living_buds) {
                living_buds = false;

                for (const auto &phytomer: *phytomers) {
                    bool next_phytomer = false;

                    for (auto &petiole: phytomer->floral_buds) {
                        if (next_phytomer) {
                            break;
                        }

                        bool next_petiole = false;
                        for (auto &fbud: petiole) {
                            if (next_petiole) {
                                break;
                            }
                            if (fbud.state == BUD_DORMANT || fbud.state == BUD_DEAD) {
                                continue;
                            }

                            for (uint fruit_objID: fbud.inflorescence_objIDs) {
                                float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID) / fbud.current_fruit_scale_factor; //mature fruit volume
                                working_carb_pool += fruit_construction_cost_base * (mature_volume) * (fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor);
                            }
                            phytomer->setFloralBudState(BUD_DEAD, fbud);
                            //Kill a floral bud to eliminate it as a future sink

                            if (working_carb_pool > carbohydrate_params. carbohydrate_abortion_threshold*carbohydrate_params.stem_density*shoot_volume/C_molecular_wt ) {
                                goto shoot_balanced;
                            } //If the amount of carbon you've eliminated by aborting flower buds would have given you a positive carbon balance, move on to the next shoot

                            living_buds = true;
                            //There was at least one living bud, so stay in the loop until there aren't any more
                            next_petiole = true;
                            //As soon as you've eliminated one bud from a given petiole, move to the next one
                            next_phytomer = true;
                            //As soon as you've eliminated one bud from a given phytomer, move to the next one
                        }
                    }
                }
            }

        }

        shoot_balanced:
        ; //empty statement after the label to avoid a compiler warning

    }

}


/**
Adjust the phyllochron of plant shoots based on their carbon pool status.

@param dt Time step (days)
*/
void PlantArchitecture::checkCarbonPool_adjustPhyllochron(float dt){

    for( auto &[plantID, plant_instance]: plant_instances ){

        const auto shoot_tree = &plant_instance.shoot_tree;
        const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;

        for(const auto &shoot: *shoot_tree ){

            uint shootID = shoot->ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();

            if(shoot->carbohydrate_pool_molC > carbohydrate_params.carbohydrate_phyllochron_threshold*carbohydrate_params.stem_density * shoot_volume /C_molecular_wt){
                if(shoot->phyllochron_instantaneous > shoot->shoot_parameters.phyllochron_min.val() * shoot->phyllochron_recovery * dt){
                    shoot->phyllochron_instantaneous = shoot->phyllochron_instantaneous / (shoot->phyllochron_recovery*dt);
                }else{
                    shoot->phyllochron_instantaneous = shoot->shoot_parameters.phyllochron_min.val();
                }

            }else{
                if(shoot->phyllochron_instantaneous <= shoot->shoot_parameters.phyllochron_min.val() * 5.f){
                    shoot->phyllochron_instantaneous = shoot->phyllochron_instantaneous * shoot->phyllochron_increase;
                }
            }

        }

    }

}

/**
Transfer carbon between parent and child shoots based on carbohydrate pool gradients.
Updates the carbohydrate pools of plant shoots by transferring carbon up (from parent to child) and down
(from child to parent) based on concentration gradient and shoot volume.

@param dt Time step (days).
*/
void PlantArchitecture::checkCarbonPool_transferCarbon(float dt) {
    for( auto &[plantID, plant_instance]: plant_instances ){

        const auto shoot_tree = &plant_instance.shoot_tree;
        const auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;
        const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;

        //Transfer carbon from parent shoots to distal child shoots (gradient-driven flux)
        for(const auto &shoot_inner: *shoot_tree )
        {
            if (!shoot_inner) {
                continue;  // Skip null shoots
            }
            uint shootID_inner = shoot_inner->ID;
            float shoot_volume_inner = plant_instances.at(plantID).shoot_tree.at(shootID_inner)->calculateShootInternodeVolume();
            float shoot_carb_pool_molC_inner = shoot_inner->carbohydrate_pool_molC;
            float shoot_carb_conc_inner = shoot_carb_pool_molC_inner / shoot_volume_inner;
            if(shoot_inner->carbohydrate_pool_molC > carbohydrate_params.carbohydrate_transfer_threshold*shoot_volume_inner*carbohydrate_params.stem_density/C_molecular_wt) {

                float totalChildVolume = shoot_inner->sumChildVolume(0);
                if (totalChildVolume <= 0.0f) {
                    continue;
                }
                //Determine carbon pool (mol C) available for transfer from parent shoot.
                float available_fraction_of_carb = (shoot_inner->carbohydrate_pool_molC - carbohydrate_params.carbohydrate_transfer_threshold*shoot_volume_inner*carbohydrate_params.stem_density/C_molecular_wt) / shoot_inner->carbohydrate_pool_molC;

                for ( int p = 0; p < shoot_inner->phytomers.size(); p++) {
                    //call recursively for child shoots
                    if (shoot_inner->childIDs.find(p) != shoot_inner->childIDs.end()) {
                        for (int child_shoot_ID: shoot_inner->childIDs.at(p)) {
                            float child_volume = plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->sumChildVolume(0)+
                                plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->calculateShootInternodeVolume();
                            float child_ratio = child_volume / totalChildVolume;

                            float child_shoot_volume = plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->calculateShootInternodeVolume();

                            if (child_shoot_volume > 0){
                                float child_shoot_carb_pool_molC = shoot_tree_ptr->at(child_shoot_ID)->carbohydrate_pool_molC;
                                float child_shoot_carb_conc = child_shoot_carb_pool_molC / child_shoot_volume;
                                //Only tranfer carbon if the parent shoot has greater carbon concentration than the child shoot.
                                if (shoot_carb_conc_inner > child_shoot_carb_conc){
                                    float transfer_volume = shoot_volume_inner;
                                    if (child_shoot_volume < shoot_volume_inner){
                                        transfer_volume = child_shoot_volume;
                                    }
                                    float delta_C = shoot_carb_conc_inner - child_shoot_carb_conc;
                                    float transfer_mol_C_demand = delta_C * child_ratio * transfer_volume * carbohydrate_params.carbon_conductance_up * dt;

                                    float transfer_mol_C = std::clamp(transfer_mol_C_demand, 0.f, shoot_inner->carbohydrate_pool_molC*available_fraction_of_carb*child_ratio);

                                    //Mass-balance transfer of carbon between shoots
                                    plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->carbohydrate_pool_molC += transfer_mol_C;
                                    shoot_inner->carbohydrate_pool_molC -= transfer_mol_C;

                                }
                            }
                        }
                    }
                }
            }
        }

        //Transfer carbon from distal child shoots to parent shoots (gradient-driven flux)
        for(const auto &shoot: *shoot_tree )
        {
            if (!shoot) {
                continue;  // Skip null shoots
            }
            uint shootID = shoot->ID;
            uint parentID = shoot->parent_shoot_ID;

            float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shootID)->calculateShootInternodeVolume();

            if (shoot_volume <= 0.0f) {
                continue;
            }

            float shoot_carb_pool_molC = shoot->carbohydrate_pool_molC;
            float shoot_carb_conc = shoot_carb_pool_molC / shoot_volume;

            //Only transfer carbon if child shoot has greater carbon concentration than parent
            if(shoot_carb_pool_molC > carbohydrate_params.carbohydrate_transfer_threshold*shoot_volume*carbohydrate_params.stem_density/C_molecular_wt){
                float available_fraction_of_carb = (shoot->carbohydrate_pool_molC - carbohydrate_params.carbohydrate_transfer_threshold*shoot_volume*carbohydrate_params.stem_density/C_molecular_wt)/shoot->carbohydrate_pool_molC;
                if(parentID < 10000000 ){
                    float parent_shoot_volume = plant_instances.at(plantID).shoot_tree.at(parentID)->calculateShootInternodeVolume();

                    float parent_shoot_carb_pool_molC = plant_instances.at(plantID).shoot_tree.at(parentID)->carbohydrate_pool_molC;
                    float parent_shoot_carb_conc = parent_shoot_carb_pool_molC / parent_shoot_volume;

                    float delta_C = shoot_carb_conc - parent_shoot_carb_conc;
                    float transfer_mol_C_demand = delta_C * shoot_volume * carbohydrate_params.carbon_conductance_down * dt;
                    // Ensure only available carbon is transferred
                    float transfer_mol_C = std::clamp(transfer_mol_C_demand, 0.f, shoot->carbohydrate_pool_molC*available_fraction_of_carb);
                    // Mass balance transfer of C (mol) from child to parent shoot
                    plant_instances.at(plantID).shoot_tree.at(parentID)->carbohydrate_pool_molC += transfer_mol_C;
                    shoot->carbohydrate_pool_molC -= transfer_mol_C;

                }

            }
        }
    }
}


/**
 Update the internode girth of a specific phytomer within a shoot, based on distal leaf area.
 Limit radial growth rate if carbon concentration drops below a sustainable threshold.

@param plantID UUID of the plant
@param shootID UUID of the shoot
@param node_number The index of the node within the shoot whose internode girth being updated
@param dt Time step (days)
@param update_context_geometry A flag to determine whether to update the context geometry
*/
void PlantArchitecture::incrementPhytomerInternodeGirth_carb(uint plantID, uint shootID, uint node_number, float dt, bool update_context_geometry) {

    //Slow Radial growth of shoots if their carbon concentration falls below a sustainable threshold
    const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if( node_number>=shoot->current_node_number ){
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Cannot scale internode " + std::to_string(node_number) + " because there are only " + std::to_string(shoot->current_node_number) + " nodes in this shoot.");
    }

    auto phytomer = shoot->phytomers.at(node_number);

    // float leaf_area = phytomer->calculateDownstreamLeafArea();
    float leaf_area = phytomer->downstream_leaf_area;
    // std::cout << "leaf area: " << leaf_area_old << " " << leaf_area << std::endl;
    if ( context_ptr->doesObjectExist(shoot->internode_tube_objID) ) {
        context_ptr->setObjectData( shoot->internode_tube_objID, "leaf_area", leaf_area );
    }


        float internode_area = phytomer->parent_shoot_ptr->shoot_parameters.girth_area_factor.val() * leaf_area * 1e-4;
        phytomer->parent_shoot_ptr->shoot_parameters.girth_area_factor.resample();

        float phytomer_radius = sqrtf(internode_area / PI_F);

        float rho_cw = carbohydrate_params.stem_density * carbohydrate_params.stem_carbon_percentage / C_molecular_wt; //Density of carbon in almond wood (mol C m^-3)
        float max_shoot_volume = internode_area * shoot->calculateShootLength();
        float current_shoot_volume = shoot->calculateShootInternodeVolume();
        float max_carbon_demand = (max_shoot_volume - current_shoot_volume) * rho_cw; //(mol C)

            float threshold_carbon_pool = carbohydrate_params.carbohydrate_growth_threshold * current_shoot_volume * carbohydrate_params.stem_density / C_molecular_wt; //(mol C)

            auto &segment = shoot->shoot_internode_radii.at(node_number);
            for( float &radius : segment  ) {
                if( phytomer_radius > radius ) { //radius should only increase
                        float carbon_availability_ratio = std::clamp((shoot->carbohydrate_pool_molC - threshold_carbon_pool) / max_carbon_demand, .05f, 1.f);
                        radius = radius + carbon_availability_ratio * 0.5 * (phytomer_radius - radius);
                }


            if (update_context_geometry && context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                context_ptr->setTubeRadii(shoot->internode_tube_objID, flatten(shoot->shoot_internode_radii));
            }

    }
}