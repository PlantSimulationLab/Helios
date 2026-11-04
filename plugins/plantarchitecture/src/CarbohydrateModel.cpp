/** \file "CarbohydrateModel.cpp" Definitions related to carbohydrate model calculations in the plant architecture plug-in.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#include "PlantArchitecture.h"

using namespace helios;

namespace {

    //! Mean radius of one internode of a shoot (m)
    float meanInternodeRadius(const Shoot &shoot, uint node_index) {
        const std::vector<float> &segment = shoot.shoot_internode_radii.at(node_index);
        if (segment.empty()) {
            return 0.f;
        }
        float sum = 0.f;
        for (float radius: segment) {
            sum += radius;
        }
        return sum / float(segment.size());
    }

    //! Check the storage-capacity parameters, which are divided by or used as fractions.
    void validateStorageParameters(const CarbohydrateParameters &params, const char *caller) {
        const std::string where = std::string("ERROR (PlantArchitecture::") + caller + "): CarbohydrateParameters::";
        if (!(params.storage_capacity > 0.f)) {
            helios_runtime_error(where + "storage_capacity must be positive (got " + std::to_string(params.storage_capacity) + " g C / g DW).");
        }
        if (!(params.storage_refill_rate > 0.f)) {
            helios_runtime_error(where + "storage_refill_rate must be positive (got " + std::to_string(params.storage_refill_rate) + " per day).");
        }
    }

    //! Dry mass of a plant's root system (g DW): a fixed fraction of its wood under CARBON_SUBMODEL_TREE, or the mass of the carbon grown by allocation under CARBON_SUBMODEL_GRASS
    float rootDryMass(const PlantInstance &plant_instance, float wood_mass_gDW) {
        const CarbohydrateParameters &params = plant_instance.carb_parameters;
        if (params.submodel == CARBON_SUBMODEL_GRASS) {
            return plant_instance.root_structural_carbon_molC * C_molecular_wt / params.stem_structural_carbon_percentage;
        }
        return wood_mass_gDW / params.shoot_root_ratio;
    }

    //! Move starch into sugar to cover a sugar deficit, as far as the starch allows.
    void hydrolyzeStarchToCoverSugar(float &sugar_molC, float &starch_molC) {
        if (sugar_molC < 0.f && starch_molC > 0.f) {
            const float released = std::min(starch_molC, -sugar_molC);
            starch_molC -= released;
            sugar_molC += released;
        }
    }

} // namespace

// -------------------- Organ construction costs -------------------- //

float Phytomer::calculatePhytomerConstructionCosts() const {
    const CarbohydrateParameters &carbohydrate_params = plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters;

    float leaf_area = 0;
    for (const auto &petiole: leaf_objIDs) {
        for (uint leaf_objID: petiole) {
            if (context_ptr->doesObjectExist(leaf_objID)) {
                leaf_area += plantarchitecture_ptr->getLeafBladeArea(leaf_objID);
            }
        }
    }

    if (!(carbohydrate_params.leaf_mass_fraction_at_expansion > 0.f && carbohydrate_params.leaf_mass_fraction_at_expansion <= 1.f) || !(carbohydrate_params.leaf_maturation_days > 0.f)) {
        helios_runtime_error("ERROR (Phytomer::calculatePhytomerConstructionCosts): CarbohydrateParameters::leaf_mass_fraction_at_expansion must be greater than 0 and at most 1, and leaf_maturation_days must be positive (got " +
                             std::to_string(carbohydrate_params.leaf_mass_fraction_at_expansion) + " and " + std::to_string(carbohydrate_params.leaf_maturation_days) + ").");
    }
    // A leaf reaches its full area before its full thickness: the rest of its dry mass per unit area is laid down as it ages
    const float maturity = std::clamp(age / carbohydrate_params.leaf_maturation_days, 0.f, 1.f);
    const float mass_fraction = carbohydrate_params.leaf_mass_fraction_at_expansion + (1.f - carbohydrate_params.leaf_mass_fraction_at_expansion) * maturity;

    return leaf_area / carbohydrate_params.SLA * mass_fraction * carbohydrate_params.leaf_carbon_percentage / C_molecular_wt; // mol C
}

float Phytomer::calculateInternodeStructuralCarbon(uint node_index) const {
    const CarbohydrateParameters &carbohydrate_params = plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters;

    const float volume = std::max(0.f, calculatePhytomerVolume(node_index));
    const float maturity = carbohydrate_params.maturity_age > 0.f ? std::clamp(age / carbohydrate_params.maturity_age, 0.f, 1.f) : 1.f;
    const float density_ratio = carbohydrate_params.initial_density_ratio + (1.f - carbohydrate_params.initial_density_ratio) * maturity;
    const float mature_carbon_density = carbohydrate_params.stem_density * carbohydrate_params.stem_structural_carbon_percentage / C_molecular_wt; // mol C / m^3

    return volume * mature_carbon_density * density_ratio;
}

float Phytomer::calculateFlowerConstructionCosts(const FloralBud &fbud) const {
    float flower_carbon_cost = 0.f; // mol C

    for (uint flower_objID: fbud.inflorescence_objIDs) {
        flower_carbon_cost += plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters.total_flower_cost;
    }

    return flower_carbon_cost;
}

// -------------------- Pool initialization and queries -------------------- //

void PlantArchitecture::markShootConstructionCharged(Shoot &shoot) const {
    const PlantInstance &plant_instance = plant_instances.at(shoot.plantID);
    for (uint p = 0; p < shoot.phytomers.size(); p++) {
        const auto &phytomer = shoot.phytomers.at(p);
        phytomer->internode_carbon_charged_molC = phytomer->calculateInternodeStructuralCarbon(p);
        phytomer->leaf_carbon_charged_molC = phytomer->calculatePhytomerConstructionCosts();
        if (plant_instance.carb_parameters.submodel == CARBON_SUBMODEL_TREE) {
            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    fbud.fruit_carbon_fraction_charged = phytomer->calculateFruitDryMatterFraction(fbud);
                    fbud.fruit_potential_fraction = fbud.fruit_carbon_fraction_charged;
                }
            }
        }
    }
    shoot.old_shoot_volume = shoot.calculateShootInternodeVolume();
}

void PlantArchitecture::initializeCarbohydratePool(float carbohydrate_concentration_molC_m3) {
    for (const auto &[plantID, plant_instance]: plant_instances) {
        initializePlantCarbohydratePool(plantID, carbohydrate_concentration_molC_m3);
    }
}

void PlantArchitecture::initializePlantCarbohydratePool(uint plantID, float carbohydrate_concentration_molC_m3) {

    // Make sure that the plant exists in the context
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantCarbohydratePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (carbohydrate_concentration_molC_m3 < 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantCarbohydratePool): Carbohydrate concentration must be greater than or equal to zero.");
    }

    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;

    float wood_volume = 0.f;
    float wood_structural_carbon = 0.f;
    for (const auto &shoot: plant_instance.shoot_tree) {
        initializeShootCarbohydratePool(plantID, shoot->ID, carbohydrate_concentration_molC_m3);
        wood_volume += shoot->calculateShootInternodeVolume();
        for (const auto &phytomer: shoot->phytomers) {
            wood_structural_carbon += phytomer->internode_carbon_charged_molC;
        }
    }

    // Roots are filled to the same fraction of their storage capacity as the wood, and start at the shoot:root ratio of the parameters.
    const float root_wood_equivalent_volume = wood_volume / carbohydrate_params.shoot_root_ratio;
    plant_instance.root_sugar_pool_molC = carbohydrate_concentration_molC_m3 * root_wood_equivalent_volume * carbohydrate_params.root_storage_ratio;
    plant_instance.root_starch_pool_molC = 0.f;
    plant_instance.root_structural_carbon_molC = wood_structural_carbon / carbohydrate_params.shoot_root_ratio;
    plant_instance.carbon_budget = PlantCarbonBudget();
}

void PlantArchitecture::initializeShootCarbohydratePool(uint plantID, uint shootID, float carbohydrate_concentration_molC_m3) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::initializeShootCarbohydratePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::initializeShootCarbohydratePool): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    } else if (carbohydrate_concentration_molC_m3 < 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::initializeShootCarbohydratePool): Carbohydrate concentration must be greater than or equal to zero.");
    }

    const std::shared_ptr<Shoot> &shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    const float shoot_volume = shoot->calculateShootInternodeVolume();
    shoot->sugar_pool_molC = shoot_volume * carbohydrate_concentration_molC_m3;
    shoot->starch_pool_molC = 0.f;
    shoot->total_carbohydrate_pool_molC = shoot->sugar_pool_molC;

    // The pool starts the carbon accounting: the structure standing now is taken as already built, so only growth from
    // here on is charged. Otherwise a plant grown before the model was enabled pays for all of its standing wood at once.
    markShootConstructionCharged(*shoot);

    // A pruned shoot, and any shoot built with internode context geometry disabled, has no tube object to carry the reported pool.
    if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        context_ptr->setObjectData(shoot->internode_tube_objID, "carbohydrate_concentration", carbohydrate_concentration_molC_m3);
        context_ptr->setObjectData(shoot->internode_tube_objID, "sugar_pool_molC", shoot->sugar_pool_molC);
    }
}

float PlantArchitecture::getPlantRootCarbohydratePool(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantRootCarbohydratePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    return plant_instance.root_sugar_pool_molC + plant_instance.root_starch_pool_molC;
}

float PlantArchitecture::getPlantTotalCarbohydratePool(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantTotalCarbohydratePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    float total = getPlantRootCarbohydratePool(plantID);
    for (const auto &shoot: plant_instances.at(plantID).shoot_tree) {
        if (!shoot->isPruned()) {
            total += shoot->sugar_pool_molC + shoot->starch_pool_molC;
        }
    }
    return total;
}

PlantCarbonBudget PlantArchitecture::getPlantCarbonBudget(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantCarbonBudget): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    return plant_instances.at(plantID).carbon_budget;
}

// -------------------- Sources -------------------- //

void PlantArchitecture::accumulateHourlyLeafPhotosynthesis() {
    // The running totals are held on the phytomers rather than as primitive data: this runs every simulated hour over
    // every leaf primitive, and reading and writing a Context data label per primitive dominated the cost of the model.
    for (auto &[plantID, plant_instance]: plant_instances) {
        plant_instance.photosynthesis_accumulations++;
        for (const auto &shoot: plant_instance.shoot_tree) {
            // A pruned shoot has no leaves; a dormant shoot's leaves do not contribute.
            if (shoot->isPruned() || shoot->isdormant) {
                continue;
            }
            for (const auto &phytomer: shoot->phytomers) {
                float hourly_umol_s = 0.f; // umol CO2 s^-1
                for (const auto &petiole: phytomer->leaf_objIDs) {
                    for (const uint leaf_objID: petiole) {
                        for (const uint UUID: context_ptr->getObjectPrimitiveUUIDs(leaf_objID)) {
                            if (!context_ptr->doesPrimitiveDataExist(UUID, "net_photosynthesis")) {
                                plant_instance.leaf_primitive_hours_missing_photosynthesis++;
                                continue;
                            }
                            float leaf_A;
                            context_ptr->getPrimitiveData(UUID, "net_photosynthesis", leaf_A);
                            hourly_umol_s += leaf_A * context_ptr->getPrimitiveArea(UUID);
                        }
                    }
                }
                phytomer->cumulative_net_photosynthesis_molC += hourly_umol_s * 3600.f * 1e-6f;
            }
        }
    }
}

void PlantArchitecture::accumulateShootPhotosynthesis(uint plantID) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;

    bool has_leaves = false;
    for (const auto &shoot: plant_instance.shoot_tree) {
        // A pruned shoot is left in the shoot tree as an empty shell with no phytomers, no volume
        // and no internode tube object. It takes no part in carbon accounting: including it divides
        // by a zero shoot volume, writes object data to a tube object that no longer exists, and
        // lets its stale carbon pool re-trigger the instant-death prune below on every timestep.
        if (shoot->isPruned()) {
            continue;
        }

        // Photosynthesis accumulated before a shoot went dormant is discarded, as is any on a dormant shoot.
        float net_photosynthesis = 0;
        for (const auto &phytomer: shoot->phytomers) {
            if (!shoot->isdormant) {
                net_photosynthesis += phytomer->cumulative_net_photosynthesis_molC;
                has_leaves = has_leaves || !phytomer->leaf_objIDs.empty();
            }
            phytomer->cumulative_net_photosynthesis_molC = 0.f;
        }

        // Net photosynthesis is credited whatever its sign: a shoot whose leaves respire more than they fix over the
        // period is a net carbon sink, and that loss must come out of the pools like any other. Photosynthate with no
        // sink is forgone later, in allocateSurplusCarbon().
        shoot->sugar_pool_molC += net_photosynthesis;
        shoot->season_net_photosynthesis_molC += net_photosynthesis;
        plant_instance.carbon_budget.net_photosynthesis += net_photosynthesis;

        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            const float shoot_volume = shoot->calculateShootInternodeVolume();
            float total_carbohydrate = shoot->sugar_pool_molC + shoot->starch_pool_molC;
            if (shoot_volume > 0.f) {
                context_ptr->setObjectData(shoot->internode_tube_objID, "carbohydrate_concentration", total_carbohydrate / shoot_volume);
            }
            context_ptr->setObjectData(shoot->internode_tube_objID, "sugar_pool_molC", shoot->sugar_pool_molC);
            context_ptr->setObjectData(shoot->internode_tube_objID, "total_carbohydrate_pool_molC", total_carbohydrate);
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_net_photosynthesis", net_photosynthesis);
        }
    }

    if (has_leaves && plant_instance.photosynthesis_accumulations == 0) {
        std::cerr << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): plant " << plantID
                  << " has leaves, but no photosynthesis was accumulated since the last time step. Run the photosynthesis model and call accumulateHourlyLeafPhotosynthesis() each hour." << std::endl;
    } else if (plant_instance.leaf_primitive_hours_missing_photosynthesis > 0) {
        std::cerr << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): " << plant_instance.leaf_primitive_hours_missing_photosynthesis
                  << " leaf primitive-hours had no net_photosynthesis primitive data and contributed nothing. Did you run the photosynthesis model?" << std::endl;
    }
    plant_instance.photosynthesis_accumulations = 0;
    plant_instance.leaf_primitive_hours_missing_photosynthesis = 0;
}

// -------------------- Sinks -------------------- //

void PlantArchitecture::subtractShootMaintenanceCarbon(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    const float temperature_factor = plant_instance.respiration_temperature_factor;

    float wood_maintenance_respiration_step = 0.f;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue; // a pruned shoot has no living tissue to respire
        }

        // Respiration per unit of structural carbon falls steeply with wood age and size: current-year shoots respire
        // ~30x more than trunk wood per gram (Grossman & DeJong 1994a).
        float respiration = 0.f;
        for (uint p = 0; p < shoot->phytomers.size(); p++) {
            const auto &phytomer = shoot->phytomers.at(p);
            float rate_20C;
            if (phytomer->age < 365.f) {
                rate_20C = carbohydrate_params.r_m_w_current_20;
            } else if (meanInternodeRadius(*shoot, p) < carbohydrate_params.branch_radius_threshold) {
                rate_20C = carbohydrate_params.r_m_w_branch_20;
            } else {
                rate_20C = carbohydrate_params.r_m_w_20;
            }
            respiration += phytomer->internode_carbon_charged_molC * rate_20C;
        }
        respiration *= temperature_factor * dt;

        shoot->sugar_pool_molC -= respiration;
        shoot->season_maintenance_respiration_molC += respiration;
        plant_instance.carbon_budget.wood_maintenance_respiration += respiration;
        wood_maintenance_respiration_step += respiration;
        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_respiration", respiration);
        }
    }

    const float root_structural_carbon = plantRootStructuralCarbon(plantID);
    const float root_respiration = root_structural_carbon * carbohydrate_params.r_m_r_20 * temperature_factor * dt;
    plant_instance.root_sugar_pool_molC -= root_respiration;
    plant_instance.carbon_budget.root_maintenance_respiration += root_respiration;
    plant_instance.step_maintenance_respiration_molC = wood_maintenance_respiration_step + root_respiration;
}

void PlantArchitecture::subtractShootGrowthCarbon(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    PlantCarbonBudget &budget = plant_instance.carbon_budget;

    double reserve_funded_spending = 0.0; // stem, leaf and root growth, with growth respiration, charged to the pools this sub-step (mol C)
    float total_wood_growth = 0.f;

    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue; // a pruned shoot has no phytomers left to grow
        }

        // Structural carbon added since the organ was last charged. Organs are only ever charged for growth: a leaf that
        // is shed or an internode whose carbon density is reset is not refunded.
        float wood_growth = 0.f;
        float leaf_growth = 0.f;
        for (uint p = 0; p < shoot->phytomers.size(); p++) {
            const auto &phytomer = shoot->phytomers.at(p);

            const float internode_carbon = phytomer->calculateInternodeStructuralCarbon(p);
            wood_growth += std::max(0.f, internode_carbon - phytomer->internode_carbon_charged_molC);
            phytomer->internode_carbon_charged_molC = internode_carbon;

            const float leaf_carbon = phytomer->calculatePhytomerConstructionCosts();
            leaf_growth += std::max(0.f, leaf_carbon - phytomer->leaf_carbon_charged_molC);
            phytomer->leaf_carbon_charged_molC = leaf_carbon;
        }

        const float growth_respiration = carbohydrate_params.growth_respiration_fraction * (wood_growth + leaf_growth);
        shoot->sugar_pool_molC -= wood_growth + leaf_growth + growth_respiration;
        shoot->season_growth_molC += wood_growth + leaf_growth + growth_respiration;

        budget.wood_growth += wood_growth;
        budget.leaf_growth += leaf_growth;
        budget.growth_respiration += growth_respiration;
        reserve_funded_spending += wood_growth + leaf_growth + growth_respiration;
        total_wood_growth += wood_growth;

        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_growth", wood_growth + leaf_growth);
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_growth_respiration", growth_respiration);
        }
    }

    // Root growth is the sub-model's rule: in a tree the roots keep pace with the wood; in a grass a stage-dependent share of the photosynthate goes below ground, high at emergence and
    // falling as the reproductive sinks take over. Either way it is paid from the root system's own pool.
    float root_growth = 0.f;
    switch (carbohydrate_params.submodel) {
        case CARBON_SUBMODEL_TREE:
            root_growth = total_wood_growth / carbohydrate_params.shoot_root_ratio;
            break;
        case CARBON_SUBMODEL_GRASS: {
            const CarbonGrassParameters &grass_params = carbohydrate_params.grass;
            float allocation = grass_params.root_allocation_early + (grass_params.root_allocation_late - grass_params.root_allocation_early) * grassMainStemDevelopmentFraction(plantID);
            for (const auto &shoot: plant_instance.shoot_tree) {
                // Root growth stops once the main stem has flowered
                if (!shoot->isPruned() && shoot->parent_shoot_ID < 0 && !shoot->phytomers.empty()) {
                    for (const auto &petiole: shoot->phytomers.back()->floral_buds) {
                        for (const auto &fbud: petiole) {
                            if (fbud.isterminal && fbud.state == BUD_FRUITING && fbud.time_counter >= grass_params.anthesis_days) {
                                allocation = 0.f;
                            }
                        }
                    }
                }
            }
            root_growth = allocation * std::max(0.f, plant_instance.photosynthesis_supply_rate_molC_day) * dt / (1.f + carbohydrate_params.growth_respiration_fraction);
            plant_instance.root_structural_carbon_molC += root_growth;
            break;
        }
    }
    const float root_growth_respiration = root_growth * carbohydrate_params.growth_respiration_fraction;
    plant_instance.root_sugar_pool_molC -= root_growth + root_growth_respiration;
    budget.root_growth += root_growth;
    budget.growth_respiration += root_growth_respiration;
    reserve_funded_spending += root_growth + root_growth_respiration;

    // The vegetative demand rate, averaged over the supply-averaging period, is the demand side of the grass sub-model's tiller supply/demand index
    if (dt > 0.f) {
        const float relaxation = 1.f - std::exp(-dt / carbohydrate_params.supply_averaging_days);
        plant_instance.vegetative_demand_rate_molC_day += relaxation * (float(reserve_funded_spending) / dt - plant_instance.vegetative_demand_rate_molC_day);
    }

    // The reproductive sink is the sub-model's: what it is, and whether it is built from reserves or from current photosynthate, differs between crops.
    switch (carbohydrate_params.submodel) {
        case CARBON_SUBMODEL_TREE:
            subtractFruitGrowthCarbon(plantID, dt, reserve_funded_spending);
            break;
        case CARBON_SUBMODEL_GRASS:
            subtractGrainGrowthCarbon(plantID, dt, reserve_funded_spending);
            break;
    }
}

// -------------------- Storage capacity and surplus -------------------- //

float PlantArchitecture::getPlantCarbohydrateStorageFill(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantCarbohydrateStorageFill): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    validateStorageParameters(carbohydrate_params, "getPlantCarbohydrateStorageFill");

    const float capacity_molC = carbohydrate_params.storage_capacity * plantStorageCapacityMass(plantID) / C_molecular_wt;
    return capacity_molC > 0.f ? getPlantTotalCarbohydratePool(plantID) / capacity_molC : 0.f;
}

float PlantArchitecture::plantStorageCapacityMass(uint plantID) const {
    // The same compartments as checkCarbonPool_redistributeCarbon(): living shoots' wood, and roots with root_storage_ratio times its capacity per unit mass
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    float wood_mass_gDW = 0.f;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (!shoot->isPruned()) {
            wood_mass_gDW += shoot->calculateShootInternodeVolume() * carbohydrate_params.stem_density;
        }
    }
    return wood_mass_gDW + rootDryMass(plant_instance, wood_mass_gDW) * carbohydrate_params.root_storage_ratio;
}

void PlantArchitecture::allocateSurplusCarbon(uint plantID, float dt, const PlantCarbonBudget &budget_before) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;

    // Radial growth laid down in this sub-step, and its potential, as recorded by incrementPhytomerInternodeGirth_carb()
    const float girth_growth_molC = plant_instance.step_girth_growth_molC;
    const float girth_potential_molC = plant_instance.step_girth_potential_molC;
    plant_instance.step_girth_growth_molC = 0.f;
    plant_instance.step_girth_potential_molC = 0.f;

    validateStorageParameters(carbohydrate_params, "allocateSurplusCarbon");
    if (!(carbohydrate_params.supply_averaging_days > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::allocateSurplusCarbon): CarbohydrateParameters::supply_averaging_days must be positive (got " + std::to_string(carbohydrate_params.supply_averaging_days) + ").");
    }
    PlantCarbonBudget &budget = plant_instance.carbon_budget;
    const float cost_per_wood_carbon = (1.f + carbohydrate_params.growth_respiration_fraction) * (1.f + 1.f / carbohydrate_params.shoot_root_ratio);

    // Radial growth competes with storage for photosynthate: its cost this sub-step is added back to the photosynthate left
    // over, and its share of that sets the cambium supply ratio for the next sub-steps.
    const float girth_cost_molC = girth_growth_molC * cost_per_wood_carbon;
    const float girth_potential_cost_molC = girth_potential_molC * cost_per_wood_carbon;
    auto recordGirthSupply = [&](float allotted_molC) {
        auto &history = plant_instance.cambium_supply_history;
        history.push_back({allotted_molC, girth_potential_cost_molC, dt});
        float history_days = 0.f;
        for (const auto &entry: history) {
            history_days += entry[2];
        }
        while (history.size() > 1 && history_days - history.front()[2] >= carbohydrate_params.supply_averaging_days) {
            history_days -= history.front()[2];
            history.erase(history.begin());
        }
        double allotted_sum = 0.0;
        double potential_sum = 0.0;
        for (const auto &entry: history) {
            allotted_sum += entry[0];
            potential_sum += entry[1];
        }
        plant_instance.cambium_supply_ratio = potential_sum > 0.0 ? float(std::min(1.0, allotted_sum / potential_sum)) : 1.f;
    };

    // Photosynthate left after everything charged in this sub-step (a deficit is met from the pools as usual), and with the
    // girth growth added back, what radial growth and storage compete for
    auto spent = [](const PlantCarbonBudget &b) {
        return b.wood_maintenance_respiration + b.root_maintenance_respiration + b.wood_growth + b.root_growth + b.leaf_growth + b.fruit_growth + b.flower_construction + b.growth_respiration + b.panicle_growth + b.grain_growth;
    };
    const float remainder_molC = plant_instance.photosynthesis_supply_rate_molC_day * dt - (spent(budget) - spent(budget_before));
    const float available_molC = remainder_molC + girth_cost_molC;
    if (available_molC <= 0.f) {
        recordGirthSupply(0.f);
        return;
    }

    // Storage demand, from the pool as it stood before this remainder was added to it
    const float storage_mass_gDW = plantStorageCapacityMass(plantID);
    if (storage_mass_gDW <= 0.f) {
        recordGirthSupply(0.f);
        return;
    }
    const float capacity_molC = carbohydrate_params.storage_capacity * storage_mass_gDW / C_molecular_wt;
    const float pool_before_molC = getPlantTotalCarbohydratePool(plantID) - remainder_molC;
    const float storage_demand_molC = std::max(0.f, capacity_molC - pool_before_molC) * (1.f - std::exp(-carbohydrate_params.storage_refill_rate * dt));

    // Each sink receives the photosynthate in proportion to its demand, up to that demand
    const float total_demand_molC = storage_demand_molC + girth_potential_cost_molC;
    float to_storage_molC = 0.f;
    float to_girth_molC = 0.f;
    if (total_demand_molC > 0.f) {
        to_storage_molC = std::min(storage_demand_molC, available_molC * storage_demand_molC / total_demand_molC);
        to_girth_molC = std::min(girth_potential_cost_molC, available_molC * girth_potential_cost_molC / total_demand_molC);
    }
    recordGirthSupply(to_girth_molC);

    // Photosynthate with no sink is forgone: withdrawn from each compartment in proportion to its carbohydrate, sugar first
    const float forgone_molC = available_molC - girth_cost_molC - to_storage_molC;
    if (forgone_molC <= 0.f) {
        return;
    }
    float compartment_total = std::max(0.f, plant_instance.root_sugar_pool_molC + plant_instance.root_starch_pool_molC);
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (!shoot->isPruned()) {
            compartment_total += std::max(0.f, shoot->sugar_pool_molC + shoot->starch_pool_molC);
        }
    }
    if (compartment_total <= 0.f) {
        return;
    }
    const float fraction = std::min(1.f, forgone_molC / compartment_total);
    auto withdraw = [fraction](float &sugar, float &starch) {
        float amount = std::max(0.f, sugar + starch) * fraction;
        const float from_sugar = std::min(amount, std::max(0.f, sugar));
        sugar -= from_sugar;
        starch -= amount - from_sugar;
        return amount;
    };
    float withdrawn_molC = withdraw(plant_instance.root_sugar_pool_molC, plant_instance.root_starch_pool_molC);
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (!shoot->isPruned()) {
            const float amount = withdraw(shoot->sugar_pool_molC, shoot->starch_pool_molC);
            shoot->season_net_photosynthesis_molC -= amount;
            withdrawn_molC += amount;
        }
    }
    budget.net_photosynthesis -= withdrawn_molC;
    budget.photosynthesis_downregulation += withdrawn_molC;
}

// -------------------- Transport and storage -------------------- //

void PlantArchitecture::checkCarbonPool_redistributeCarbon(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;

    if (carbohydrate_params.starch_temperature_warm <= carbohydrate_params.starch_temperature_cold) {
        helios_runtime_error("ERROR (PlantArchitecture::checkCarbonPool_redistributeCarbon): CarbohydrateParameters::starch_temperature_warm (" + std::to_string(carbohydrate_params.starch_temperature_warm) +
                             " C) must be greater than starch_temperature_cold (" + std::to_string(carbohydrate_params.starch_temperature_cold) + " C).");
    }

    // A storage compartment: a living shoot's wood, or the root system.
    struct Compartment {
        float *sugar_molC;
        float *starch_molC;
        float storage_capacity_gDW; //!< dry mass weighted by relative storage capacity
        float dry_mass_gDW; //!< dry mass, which sets how much sugar the tissue holds before storing starch
    };

    std::vector<Compartment> compartments;
    compartments.reserve(plant_instance.shoot_tree.size() + 1);
    float wood_mass_gDW = 0.f;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (!shoot || shoot->isPruned()) {
            continue;
        }
        const float wood_mass = shoot->calculateShootInternodeVolume() * carbohydrate_params.stem_density;
        if (wood_mass <= 0.f) {
            continue;
        }
        compartments.push_back({&shoot->sugar_pool_molC, &shoot->starch_pool_molC, wood_mass, wood_mass});
        wood_mass_gDW += wood_mass;
    }
    const float root_mass_gDW = rootDryMass(plant_instance, wood_mass_gDW);
    if (root_mass_gDW > 0.f) {
        compartments.push_back({&plant_instance.root_sugar_pool_molC, &plant_instance.root_starch_pool_molC, root_mass_gDW * carbohydrate_params.root_storage_ratio, root_mass_gDW});
    }
    if (compartments.empty()) {
        return;
    }

    // (1) Phloem transport: every compartment relaxes toward the plant's mean carbohydrate per unit of storage capacity.
    // The exchange sums to zero, so carbon is conserved.
    if (carbohydrate_params.carbon_mixing_rate > 0.f && compartments.size() > 1) {
        float total_carbohydrate = 0.f;
        float total_capacity = 0.f;
        for (const Compartment &compartment: compartments) {
            total_carbohydrate += *compartment.sugar_molC + *compartment.starch_molC;
            total_capacity += compartment.storage_capacity_gDW;
        }
        const float mean_fill = total_carbohydrate / total_capacity;
        const float relaxation = 1.f - std::exp(-carbohydrate_params.carbon_mixing_rate * dt);
        for (const Compartment &compartment: compartments) {
            const float carbohydrate = *compartment.sugar_molC + *compartment.starch_molC;
            *compartment.sugar_molC += relaxation * (compartment.storage_capacity_gDW * mean_fill - carbohydrate);
        }
    }

    // (2) A compartment that spent more than it held -- typically a shoot that has just been charged for new leaves and
    // wood -- imports like any active sink: it is brought up to the plant's mean fill, drawn from the other compartments
    // in proportion to what they hold. Only when the whole plant is exhausted does a deficit remain.
    float total_carbohydrate = 0.f;
    float total_capacity = 0.f;
    float surplus = 0.f;
    for (const Compartment &compartment: compartments) {
        const float carbohydrate = *compartment.sugar_molC + *compartment.starch_molC;
        total_carbohydrate += carbohydrate;
        total_capacity += compartment.storage_capacity_gDW;
        if (carbohydrate > 0.f) {
            surplus += carbohydrate;
        }
    }
    if (total_carbohydrate > 0.f) {
        const float mean_fill = total_carbohydrate / total_capacity;
        float import = 0.f;
        for (const Compartment &compartment: compartments) {
            const float carbohydrate = *compartment.sugar_molC + *compartment.starch_molC;
            if (carbohydrate < 0.f) {
                import += compartment.storage_capacity_gDW * mean_fill - carbohydrate;
            }
        }
        if (import > 0.f) {
            // The import cannot exceed what the positive compartments hold, since the plant as a whole is in surplus.
            const float fraction_given = import / surplus;
            for (const Compartment &compartment: compartments) {
                const float carbohydrate = *compartment.sugar_molC + *compartment.starch_molC;
                if (carbohydrate < 0.f) {
                    *compartment.sugar_molC += compartment.storage_capacity_gDW * mean_fill - carbohydrate;
                } else {
                    *compartment.sugar_molC -= carbohydrate * fraction_given;
                }
            }
        }
    }

    // (3) Storage capacity. Carbohydrate the plant can neither spend nor store leaves it below ground as root exudation and
    // respiration. Transport keeps the compartments at a common fill, so each gives up the same fraction.
    {
        validateStorageParameters(carbohydrate_params, "checkCarbonPool_redistributeCarbon");
        float carbohydrate = 0.f;
        float capacity_gDW = 0.f;
        for (const Compartment &compartment: compartments) {
            carbohydrate += *compartment.sugar_molC + *compartment.starch_molC;
            capacity_gDW += compartment.storage_capacity_gDW;
        }
        const float capacity_molC = carbohydrate_params.storage_capacity * capacity_gDW / C_molecular_wt;
        if (carbohydrate > capacity_molC) {
            const float retained = capacity_molC / carbohydrate;
            for (const Compartment &compartment: compartments) {
                *compartment.sugar_molC *= retained;
                *compartment.starch_molC *= retained;
            }
            plant_instance.carbon_budget.storage_overflow += carbohydrate - capacity_molC;
        }
    }

    // (4) Sugar-starch partitioning. Tissue holds carbohydrate as soluble sugar up to a saturation level and stores the
    // rest as starch; cold raises the sugar level, converting starch to sugar over winter.
    const float warmth = std::clamp((plant_instance.carbon_model_temperature_C - carbohydrate_params.starch_temperature_cold) / (carbohydrate_params.starch_temperature_warm - carbohydrate_params.starch_temperature_cold), 0.f, 1.f);
    const float sugar_saturation_gC_gDW = carbohydrate_params.sugar_saturation_cold + (carbohydrate_params.sugar_saturation_warm - carbohydrate_params.sugar_saturation_cold) * warmth;
    const float conversion = 1.f - std::exp(-carbohydrate_params.starch_conversion_rate * dt);
    for (const Compartment &compartment: compartments) {
        hydrolyzeStarchToCoverSugar(*compartment.sugar_molC, *compartment.starch_molC);
        const float carbohydrate = *compartment.sugar_molC + *compartment.starch_molC;
        if (carbohydrate <= 0.f) {
            continue;
        }
        const float sugar_target = std::min(carbohydrate, sugar_saturation_gC_gDW * compartment.dry_mass_gDW / C_molecular_wt);
        const float starch_change = conversion * ((carbohydrate - sugar_target) - *compartment.starch_molC);
        *compartment.starch_molC += starch_change;
        *compartment.sugar_molC -= starch_change;
    }

    for (const auto &shoot: plant_instance.shoot_tree) {
        if (!shoot || shoot->isPruned()) {
            continue;
        }
        shoot->total_carbohydrate_pool_molC = shoot->sugar_pool_molC + shoot->starch_pool_molC;
        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            const float shoot_volume = shoot->calculateShootInternodeVolume();
            if (shoot_volume > 0.f) {
                context_ptr->setObjectData(shoot->internode_tube_objID, "carbohydrate_concentration", shoot->total_carbohydrate_pool_molC / shoot_volume);
            }
            context_ptr->setObjectData(shoot->internode_tube_objID, "sugar_pool_molC", shoot->sugar_pool_molC);
            context_ptr->setObjectData(shoot->internode_tube_objID, "total_carbohydrate_pool_molC", shoot->total_carbohydrate_pool_molC);
        }
    }
}

// -------------------- Carbon-limited growth and abortion -------------------- //

void PlantArchitecture::checkCarbonPool_abortOrgans(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const auto shoot_tree = &plant_instance.shoot_tree;
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;

    for (const auto &shoot: *shoot_tree) {
        // A pruned shoot is left in the shoot tree as an empty shell with no phytomers, no volume
        // and no internode tube object. It takes no part in carbon accounting: including it divides
        // by a zero shoot volume, writes object data to a tube object that no longer exists, and
        // lets its stale carbon pool re-trigger the instant-death prune below on every timestep.
        if (shoot->isPruned()) {
            continue;
        }

        uint shootID = shoot->ID;

        shoot->total_carbohydrate_pool_molC = shoot->sugar_pool_molC + shoot->starch_pool_molC;

        // A shoot left with no carbohydrate at all -- which, after redistribution, means the whole plant is exhausted -- dies.
        if (shoot->total_carbohydrate_pool_molC <= 0.f) {
            pruneBranch(plantID, shootID, 0);
            continue;
        }

        float shoot_volume = shoot->calculateShootInternodeVolume();
        const float starvation_pool = carbohydrate_params.carbohydrate_growth_threshold * carbohydrate_params.stem_density * shoot_volume / C_molecular_wt;
        // Establish a working carbon pool to see if you could stay above the carbon threshold by aborting fruits.
        float working_carb_pool = shoot->total_carbohydrate_pool_molC;
        if (dt > carbohydrate_params.carbohydrate_starvation_days) {
            shoot->days_with_negative_carbon_balance += dt / 2; // Make sure you have at least two timesteps before starting to abort buds
        } else {
            shoot->days_with_negative_carbon_balance += dt;
        }

        // A shoot above the growth threshold is not starving
        if (shoot->total_carbohydrate_pool_molC > starvation_pool) {
            shoot->days_with_negative_carbon_balance = 0;
            continue;
        }
        // Keep track of how many days the shoot has been below the threshold
        if (shoot->days_with_negative_carbon_balance <= carbohydrate_params.carbohydrate_starvation_days) {
            continue;
        }

        // Loop over fruiting buds and abort them one at a time until you would be able to stay above the threshold
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

                        // Aborting a fruit or panicle saves the carbon it would still have needed to reach its potential.
                        working_carb_pool += remainingReproductiveCarbon(*phytomer, fbud);
                        phytomer->setFloralBudState(BUD_DEAD, fbud);
                        // Kill a floral bud to eliminate it as a future sink

                        if (working_carb_pool > starvation_pool) {
                            goto shoot_balanced;
                        } // If the amount of carbon you've eliminated by aborting flower buds would have given you a positive carbon balance, move on to the next shoot

                        living_buds = true;
                        // There was at least one living bud, so stay in the loop until there aren't any more
                        next_petiole = true;
                        // As soon as you've eliminated one bud from a given petiole, move to the next one
                        next_phytomer = true;
                        // As soon as you've eliminated one bud from a given phytomer, move to the next one
                    }
                }
            }
        }
    shoot_balanced:; // empty statement after the label to avoid a compiler warning
    }
}

void PlantArchitecture::checkCarbonPool_adjustPhyllochron(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;

    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue; // a pruned shoot has no meristem whose phyllochron could be adjusted
        }

        float shoot_volume = shoot->calculateShootInternodeVolume();
        if (shoot_volume <= 0.f) {
            continue;
        }

        float phyllochron_carbon_ratio = shoot->total_carbohydrate_pool_molC / (carbohydrate_params.carbohydrate_growth_threshold * carbohydrate_params.stem_density * shoot_volume / C_molecular_wt);

        shoot->phyllochron_instantaneous = std::fmax(shoot->shoot_parameters.phyllochron_min.val(), shoot->phyllochron_min / (phyllochron_carbon_ratio * dt));

        // Organ expansion slows in proportion to the carbohydrate available below the threshold.
        const float growth_fraction = std::clamp(phyllochron_carbon_ratio, 0.f, 1.f);
        shoot->elongation_rate_instantaneous = shoot->elongation_rate_potential * growth_fraction;
        shoot->leaf_expansion_rate_instantaneous = shoot->leaf_expansion_rate_potential * growth_fraction;
    }
}

bool Shoot::sampleVegetativeBudBreak_carb(uint node_index) {
    const CarbohydrateParameters &carbohydrate_params = plantarchitecture_ptr->plant_instances.at(plantID).carb_parameters;
    float shoot_volume = calculateShootInternodeVolume();

    if (node_index >= phytomers.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::sampleVegetativeBudBreak): Invalid node index. Node index must be less than the number of phytomers on the shoot.");
    }

    // Read from this shoot rather than from its type, matching Shoot::sampleVegetativeBudBreak(). The shoot's maximum
    // probability is reduced in proportion to its carbohydrate pool once the pool falls below the bud break threshold.
    float probability_min = shoot_parameters.vegetative_bud_break_probability_min.val();
    float probability_max = shoot_parameters.vegetative_bud_break_probability_max.val();
    float probability_decay = shoot_parameters.vegetative_bud_break_probability_decay_rate.val();

    // Read the pools directly: total_carbohydrate_pool_molC is only refreshed during the carbon balance, so it is stale
    // between initialization and the first time step.
    const float carbohydrate_pool = sugar_pool_molC + starch_pool_molC;
    const float threshold_pool = carbohydrate_params.carbohydrate_growth_threshold * shoot_volume * carbohydrate_params.stem_density / C_molecular_wt;
    if (carbohydrate_pool < threshold_pool) {
        probability_max *= std::max(0.f, carbohydrate_pool) / threshold_pool;
    }

    float bud_break_probability;
    if (!shoot_parameters.growth_requires_dormancy && probability_decay < 0.f) {
        bud_break_probability = probability_min;
    } else if (probability_decay > 0.f) { // probability maximum at apex
        bud_break_probability = std::fmax(probability_min, probability_max - probability_decay * float(this->current_node_number - node_index - 1));
    } else if (probability_decay < 0.f) { // probability maximum at base
        bud_break_probability = std::fmax(probability_min, probability_max - fabs(probability_decay) * float(node_index));
    } else {
        if (probability_decay == 0.f) {
            bud_break_probability = probability_min;
        } else {
            bud_break_probability = probability_max;
        }
    }

    const float vigour_factor = vigourBudBreakFactor();
    if (vigour_factor >= 0.f) {
        bud_break_probability *= vigour_factor;
    }

    // In a grass, tiller bud outgrowth also follows the plant's carbon supply relative to its vegetative demand
    if (carbohydrate_params.submodel == CARBON_SUBMODEL_GRASS) {
        bud_break_probability *= plantarchitecture_ptr->grassTillerBudBreakFactor(plantID);
    }

    bool bud_break = true;
    if (context_ptr->randu() > bud_break_probability) {
        bud_break = false;
    }

    return bud_break;
}

void PlantArchitecture::adjustPlantMaintenanceRespiration(float Ta) {
    for (auto &[plantID, plant_instance]: plant_instances) {
        const float Ta_C = Ta - 273.15f;
        plant_instance.respiration_temperature_factor = powf(plant_instance.carb_parameters.respiration_Q10, (Ta_C - 20.f) / 10.f);
        plant_instance.carbon_model_temperature_C = Ta_C;
    }
}

// -------------------- Sub-model selection -------------------- //

void PlantArchitecture::validateCarbonSubmodel(const CarbohydrateParameters &carb_parameters, const char *caller) {
    const std::string where = std::string("ERROR (PlantArchitecture::") + caller + "): ";
    switch (carb_parameters.submodel) {
        case CARBON_SUBMODEL_TREE:
            if (carb_parameters.wood.sapwood_area_per_leaf_area > 0.f && !(carb_parameters.wood.sapwood_functional_years > 0.f)) {
                helios_runtime_error(where + "CarbonWoodParameters::sapwood_functional_years must be positive when sapwood_area_per_leaf_area is set (got " + std::to_string(carb_parameters.wood.sapwood_functional_years) + ").");
            }
            return;
        case CARBON_SUBMODEL_GRASS: {
            const CarbonGrassParameters &grass = carb_parameters.grass;
            const std::string prefix = where + "CarbonGrassParameters::";
            if (grass.stem_elongation_node < 1) {
                helios_runtime_error(prefix + "stem_elongation_node must be at least 1 (got " + std::to_string(grass.stem_elongation_node) + ").");
            }
            if (!(grass.anthesis_days > 0.f) || !(grass.grain_fill_lag_days >= 0.f) || !(grass.grain_fill_days > 0.f)) {
                helios_runtime_error(prefix + "anthesis_days and grain_fill_days must be positive and grain_fill_lag_days must not be negative (got " + std::to_string(grass.anthesis_days) + ", " +
                                     std::to_string(grass.grain_fill_lag_days) + " and " + std::to_string(grass.grain_fill_days) + " days).");
            }
            if (!(grass.panicle_potential_carbon >= 0.f) || !(grass.panicle_carbon_percentage > 0.f) || !(grass.grains_per_gram_panicle >= 0.f) || !(grass.kernel_potential_carbon >= 0.f)) {
                helios_runtime_error(prefix + "panicle_potential_carbon, grains_per_gram_panicle and kernel_potential_carbon must not be negative and panicle_carbon_percentage must be positive (got " +
                                     std::to_string(grass.panicle_potential_carbon) + ", " + std::to_string(grass.grains_per_gram_panicle) + ", " + std::to_string(grass.kernel_potential_carbon) + " and " +
                                     std::to_string(grass.panicle_carbon_percentage) + ").");
            }
            if (!(grass.culm_reserve_floor >= carb_parameters.carbohydrate_growth_threshold)) {
                helios_runtime_error(prefix + "culm_reserve_floor must be at least CarbohydrateParameters::carbohydrate_growth_threshold, or grain filling would starve the culm (got " + std::to_string(grass.culm_reserve_floor) + " and " +
                                     std::to_string(carb_parameters.carbohydrate_growth_threshold) + " g C / g DW).");
            }
            if (!(grass.panicle_growth_respiration_fraction >= 0.f) || !(grass.grain_growth_respiration_fraction >= 0.f)) {
                helios_runtime_error(prefix + "panicle_growth_respiration_fraction and grain_growth_respiration_fraction must not be negative (got " + std::to_string(grass.panicle_growth_respiration_fraction) + " and " +
                                     std::to_string(grass.grain_growth_respiration_fraction) + ").");
            }
            if (!(grass.tiller_supply_demand_min >= 0.f) || !(grass.tiller_supply_demand_max > grass.tiller_supply_demand_min)) {
                helios_runtime_error(prefix + "tiller_supply_demand_min must not be negative and tiller_supply_demand_max must exceed it (got " + std::to_string(grass.tiller_supply_demand_min) + " and " +
                                     std::to_string(grass.tiller_supply_demand_max) + ").");
            }
            if (!(grass.root_allocation_early >= 0.f && grass.root_allocation_early < 1.f) || !(grass.root_allocation_late >= 0.f && grass.root_allocation_late < 1.f)) {
                helios_runtime_error(prefix + "root_allocation_early and root_allocation_late must be at least 0 and below 1 (got " + std::to_string(grass.root_allocation_early) + " and " +
                                     std::to_string(grass.root_allocation_late) + ").");
            }
            return;
        }
        default:
            helios_runtime_error(where + "CarbohydrateParameters::submodel has an unknown value (" + std::to_string(int(carb_parameters.submodel)) + ").");
    }
}

CarbohydrateParameters CarbohydrateParameters::forSubmodel(CarbonSubmodel submodel) {
    CarbohydrateParameters params;
    params.submodel = submodel;
    switch (submodel) {
        case CARBON_SUBMODEL_TREE:
            break;
        case CARBON_SUBMODEL_GRASS:
            // Grain sorghum. A culm is a pithy stem of low dry density that reaches its mature density within weeks of elongating; its leaves are thin; its tissues respire at herbaceous rates.
            params.stem_density = 200000.f; // ~0.2 g/cm^3 stalk dry bulk density (pith and rind); to be calibrated against stalk dry mass per length
            params.stem_structural_carbon_percentage = 0.42f;
            params.maturity_age = 20.f;
            params.initial_density_ratio = 0.3f;
            params.shoot_root_ratio = 4.f;
            params.leaf_carbon_percentage = 0.44f;
            params.SLA = 0.02f; // sorghum leaf mass per area 40-55 g/m^2
            params.leaf_mass_fraction_at_expansion = 1.f;
            params.total_flower_cost = 0.f; // the panicle is the sink; it bears no separate flowers
            // Herbaceous maintenance respiration (~0.012 g C per g C per day at 20 C for stems; roots somewhat lower). Every internode of an annual is younger than a year, so the current-year
            // rate is the one in use; the older-wood rates are set equal so that nothing changes should a culm outlive a year.
            params.r_m_w_current_20 = 0.012f;
            params.r_m_w_branch_20 = 0.012f;
            params.r_m_w_20 = 0.012f;
            params.r_m_r_20 = 0.01f;
            params.growth_respiration_fraction = 0.25f;
            // Stem water-soluble carbohydrate reaches 30-50% of dry mass in cereal culms around anthesis (~0.15 g C / g DW), far above the 5% of tree wood, and accumulates within weeks
            params.root_storage_ratio = 1.f;
            params.storage_capacity = 0.15f;
            params.storage_refill_rate = 0.1f;
            break;
    }
    return params;
}

// -------------------- Shared helpers of the sub-models -------------------- //

float PlantArchitecture::recordReproductiveSupply(uint plantID, float supply_molC, float demand_molC, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const float averaging_days = plant_instance.carb_parameters.supply_averaging_days;
    auto &history = plant_instance.reproductive_assimilate_history;
    history.push_back({supply_molC, demand_molC, dt});
    float history_days = 0.f;
    for (const auto &entry: history) {
        history_days += entry[2];
    }
    while (history.size() > 1 && history_days - history.front()[2] >= averaging_days) {
        history_days -= history.front()[2];
        history.erase(history.begin());
    }
    double supply_sum = 0.0;
    double demand_sum = 0.0;
    for (const auto &entry: history) {
        supply_sum += entry[0];
        demand_sum += entry[1];
    }
    plant_instance.reproductive_supply_ratio = demand_sum > 0.0 ? float(std::min(1.0, supply_sum / demand_sum)) : 1.f;
    return plant_instance.reproductive_supply_ratio;
}

float PlantArchitecture::plantRootStructuralCarbon(uint plantID) const {
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    switch (plant_instance.carb_parameters.submodel) {
        case CARBON_SUBMODEL_GRASS:
            return plant_instance.root_structural_carbon_molC;
        case CARBON_SUBMODEL_TREE:
        default: {
            float wood_structural_carbon = 0.f;
            for (const auto &shoot: plant_instance.shoot_tree) {
                if (shoot->isPruned()) {
                    continue;
                }
                for (const auto &phytomer: shoot->phytomers) {
                    wood_structural_carbon += phytomer->internode_carbon_charged_molC;
                }
            }
            return wood_structural_carbon / plant_instance.carb_parameters.shoot_root_ratio;
        }
    }
}

float PlantArchitecture::remainingReproductiveCarbon(const Phytomer &phytomer, const FloralBud &fbud) const {
    if (fbud.state != BUD_FRUITING) {
        return 0.f;
    }
    const CarbohydrateParameters &carbohydrate_params = plant_instances.at(phytomer.plantID).carb_parameters;
    switch (carbohydrate_params.submodel) {
        case CARBON_SUBMODEL_GRASS: {
            const CarbonGrassParameters &grass_params = carbohydrate_params.grass;
            const GrainSink &grain = fbud.grain;
            const float panicle_remaining = std::max(0.f, grass_params.panicle_potential_carbon - grain.panicle_carbon_molC) * (1.f + grass_params.panicle_growth_respiration_fraction);
            // Until the grain number is fixed, the grain a full panicle would set
            const float grain_number = grain.grain_number_set ? grain.grain_number : grass_params.grains_per_gram_panicle * grass_params.panicle_potential_carbon * C_molecular_wt / grass_params.panicle_carbon_percentage;
            const float grain_remaining = std::max(0.f, grain_number * grass_params.kernel_potential_carbon - grain.grain_carbon_molC) * (1.f + grass_params.grain_growth_respiration_fraction);
            return panicle_remaining + grain_remaining;
        }
        case CARBON_SUBMODEL_TREE:
        default:
            return phytomer.calculateMatureFruitCarbon(fbud) * (1.f - fbud.fruit_carbon_fraction_charged) * (1.f + carbohydrate_params.fruit.growth_respiration_fraction);
    }
}
