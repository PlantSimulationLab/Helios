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
    for( uint internode_objID : internode_objIDs ) {
        internode_volume += context_ptr->getConeObjectVolume(internode_objID);
    }
    internode_volume /= current_internode_scale_factor; //convert to fully-elongated volume
    phytomer_carbon_cost += internode_construction_cost_base*internode_volume;

    return phytomer_carbon_cost;

}

float Phytomer::calculateFlowerConstructionCosts(const FloralBud &fbud) {

    //\todo make these values externally settable
    float flower_construction_cost_base = 8.33e-3; //mol C/flower

    float flower_carbon_cost = 0.f; //mol C

    for( uint flower_objID : fbud.inflorescence_objIDs ) {
        flower_carbon_cost += flower_construction_cost_base;
    }

    return flower_carbon_cost;
}

float Phytomer::calculateFruitConstructionCosts(const FloralBud &fbud) {

    //\todo make these values externally settable
    float fruit_construction_cost_base = 29021; //mol C/m^3

    float fruit_carbon_cost = 0.f; //mol C

    //fruit (cost per fruit basis)
    for( uint fruit_objID : fbud.inflorescence_objIDs ) {
        float volume = context_ptr->getPolymeshObjectVolume(fruit_objID)/fbud.current_fruit_scale_factor; //mature fruit volume
        fruit_carbon_cost += fruit_construction_cost_base*fruit_carbon_cost;
    }

    return fruit_carbon_cost;
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

void PlantArchitecture::subtractShootMaintainenceCarbon(float dt ){

    //\todo move to externally settable parameter
    float stem_maintainance_respiration_rate = 1.9458e-05; //mol C respired/mol C in pool/day

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){

            shoot->carbohydrate_pool_molC -= shoot->carbohydrate_pool_molC * stem_maintainance_respiration_rate * dt;

        }

    }

}