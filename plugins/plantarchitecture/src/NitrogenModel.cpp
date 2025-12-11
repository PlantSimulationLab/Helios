/** \file "NitrogenModel.cpp" Nitrogen model calculations for the PlantArchitecture plugin.

    Copyright (C) 2016-2025 Brian Bailey

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

// ==================== Enable/Disable Methods ==================== //

void PlantArchitecture::enableNitrogenModel() {
    nitrogen_model_enabled = true;
}

void PlantArchitecture::disableNitrogenModel() {
    nitrogen_model_enabled = false;
}

bool PlantArchitecture::isNitrogenModelEnabled() const {
    return nitrogen_model_enabled;
}

// ==================== Parameter Setters ==================== //

void PlantArchitecture::setPlantNitrogenParameters(uint plantID, const NitrogenParameters& params) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantNitrogenParameters): Plant ID " +
                           std::to_string(plantID) + " does not exist.");
    }
    plant_instances.at(plantID).nitrogen_parameters = params;
}

void PlantArchitecture::setPlantNitrogenParameters(const std::vector<uint>& plantIDs, const NitrogenParameters& params) {
    for (uint plantID : plantIDs) {
        setPlantNitrogenParameters(plantID, params);
    }
}

// ==================== Initialization Methods ==================== //

void PlantArchitecture::initializeNitrogenPools(float initial_leaf_N_concentration) {
    for (auto& [plantID, plant_instance] : plant_instances) {
        initializePlantNitrogenPools(plantID, initial_leaf_N_concentration);
    }
}

void PlantArchitecture::initializePlantNitrogenPools(uint plantID, float initial_leaf_N_concentration) {
    // Validate inputs
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantNitrogenPools): Plant with ID of " +
                           std::to_string(plantID) + " does not exist.");
    }
    if (initial_leaf_N_concentration < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantNitrogenPools): Initial leaf N concentration must be >= 0.");
    }

    PlantInstance& plant = plant_instances.at(plantID);
    const NitrogenParameters& N_params = plant.nitrogen_parameters;

    // Initialize plant-level pools to zero
    plant.root_nitrogen_pool_gN = 0;
    plant.available_nitrogen_pool_gN = 0;
    plant.cumulative_N_uptake_gN = 0;

    // Initialize per-leaf pools based on current leaf biomass
    for (auto& shoot : plant.shoot_tree) {
        // Clear existing leaf N pools for this shoot
        shoot->leaf_nitrogen_gN.clear();

        for (auto& phytomer : shoot->phytomers) {
            // Iterate through all leaves in this phytomer (2D vector structure)
            for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                    uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                    if (!context_ptr->doesObjectExist(leaf_objID)) {
                        continue;
                    }

                    // Get leaf area (scaled)
                    float leaf_area = context_ptr->getObjectArea(leaf_objID);

                    // Get current scale factor to calculate unscaled area
                    float scale_factor = phytomer->current_leaf_scale_factor.at(petiole_idx);
                    float unscaled_area = leaf_area / (scale_factor * scale_factor);

                    // Calculate leaf biomass (g DW)
                    float leaf_biomass_gDW = unscaled_area / N_params.SLA;

                    // Initialize leaf N pool
                    float initial_leaf_N = leaf_biomass_gDW * initial_leaf_N_concentration;
                    shoot->leaf_nitrogen_gN[leaf_objID] = initial_leaf_N;

                    // Add to plant total uptake tracking
                    plant.cumulative_N_uptake_gN += initial_leaf_N;
                }
            }
        }
    }
}

// ==================== Nitrogen Application ==================== //

void PlantArchitecture::addPlantNitrogen(uint plantID, float amount_gN) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addPlantNitrogen): Plant ID " +
                           std::to_string(plantID) + " does not exist.");
    }
    if (amount_gN < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::addPlantNitrogen): Nitrogen amount must be >= 0.");
    }

    PlantInstance& plant = plant_instances.at(plantID);
    const NitrogenParameters& N_params = plant.nitrogen_parameters;

    // Split between root and available pools
    float root_N = amount_gN * N_params.root_allocation_fraction;
    float available_N = amount_gN * (1.0f - N_params.root_allocation_fraction);

    plant.root_nitrogen_pool_gN += root_N;
    plant.available_nitrogen_pool_gN += available_N;
    plant.cumulative_N_uptake_gN += amount_gN;
}

void PlantArchitecture::addPlantNitrogen(const std::vector<uint>& plantIDs, float amount_gN) {
    for (uint plantID : plantIDs) {
        addPlantNitrogen(plantID, amount_gN);
    }
}

// ==================== Leaf Nitrogen Accumulation (Rate-Limited) ==================== //

void PlantArchitecture::accumulateLeafNitrogen(float dt) {
    for (auto& [plantID, plant] : plant_instances) {
        if (plant.available_nitrogen_pool_gN <= 0) {
            continue;  // No N available to distribute
        }

        const NitrogenParameters& N_params = plant.nitrogen_parameters;

        // Iterate through all shoots and their leaves
        for (auto& shoot : plant.shoot_tree) {
            for (auto& phytomer : shoot->phytomers) {
                // Iterate through all leaves in this phytomer (2D vector structure)
                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                        uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                        if (!context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }

                        // Get current leaf area (scaled)
                        float leaf_area = context_ptr->getObjectArea(leaf_objID);

                        // Get scale factor to calculate unscaled area
                        float scale_factor = phytomer->current_leaf_scale_factor.at(petiole_idx);
                        float unscaled_area = leaf_area / (scale_factor * scale_factor);

                        // Calculate leaf biomass (g DW)
                        float leaf_biomass_gDW = unscaled_area / N_params.SLA;

                        // Calculate target N for this leaf
                        float target_leaf_N = leaf_biomass_gDW * N_params.target_leaf_N_concentration;

                        // Get current leaf N
                        float current_leaf_N = 0;
                        if (shoot->leaf_nitrogen_gN.count(leaf_objID)) {
                            current_leaf_N = shoot->leaf_nitrogen_gN.at(leaf_objID);
                        } else {
                            // Initialize if not present
                            shoot->leaf_nitrogen_gN[leaf_objID] = 0;
                        }

                        // Calculate N demand
                        float N_demand = std::max(0.0f, target_leaf_N - current_leaf_N);
                        if (N_demand <= 0) {
                            continue;  // Leaf is at target
                        }

                        // Rate limiting: max N per day
                        float max_this_step = N_params.max_N_accumulation_rate * dt;

                        // Also limited by available N
                        float N_to_add = std::min({N_demand, max_this_step, plant.available_nitrogen_pool_gN});

                        // Update pools
                        shoot->leaf_nitrogen_gN[leaf_objID] += N_to_add;
                        plant.available_nitrogen_pool_gN -= N_to_add;

                        if (plant.available_nitrogen_pool_gN <= 0) {
                            goto next_plant;  // No more N available
                        }
                    }
                }
            }
        }

    next_plant:;
    }
}

// ==================== Nitrogen Remobilization (CRITICAL FEATURE) ==================== //

void PlantArchitecture::remobilizeNitrogen(float dt) {
    for (auto& [plantID, plant] : plant_instances) {
        const NitrogenParameters& N_params = plant.nitrogen_parameters;

        // Calculate current N stress to determine remobilization intensity
        float total_leaf_N = 0;
        float total_leaf_biomass = 0;

        for (auto& shoot : plant.shoot_tree) {
            for (auto& [leaf_objID, leaf_N] : shoot->leaf_nitrogen_gN) {
                if (!context_ptr->doesObjectExist(leaf_objID)) {
                    continue;
                }
                // Get leaf area (simplified: use current scaled area divided by SLA)
                float leaf_area = context_ptr->getObjectArea(leaf_objID);
                float leaf_biomass = leaf_area / N_params.SLA;
                total_leaf_N += leaf_N;
                total_leaf_biomass += leaf_biomass;
            }
        }

        if (total_leaf_biomass == 0) {
            continue;  // No leaves
        }

        float actual_N_concentration = total_leaf_N / total_leaf_biomass;
        float N_stress_factor = std::min(1.0f, actual_N_concentration / N_params.target_leaf_N_concentration);

        // Classify leaves by age
        std::vector<std::pair<uint, uint>> source_leaves;  // (shoot_idx, leaf_objID)
        std::vector<std::pair<uint, uint>> sink_leaves;    // (shoot_idx, leaf_objID)
        std::map<uint, float> source_N_available;          // leaf_objID → available N
        std::map<uint, float> sink_N_demand;               // leaf_objID → N demand
        float total_source_N_available = 0;
        float total_sink_demand = 0;

        uint shoot_idx = 0;
        for (auto& shoot : plant.shoot_tree) {
            for (auto& phytomer : shoot->phytomers) {
                // Get leaf lifespan from plant instance
                float leaf_lifespan = plant.max_leaf_lifespan;
                if (leaf_lifespan <= 0) {
                    leaf_lifespan = 1e6;  // Evergreen
                }

                float leaf_age = phytomer->age;
                float age_fraction = leaf_age / leaf_lifespan;

                // Iterate through all leaves in this phytomer
                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                        uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                        if (!context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }

                        if (!shoot->leaf_nitrogen_gN.count(leaf_objID)) {
                            continue;
                        }

                        // Classify by age
                        if (age_fraction >= N_params.remobilization_age_threshold) {
                            // OLD LEAF: Source for remobilization (>70% of lifespan)
                            float current_N = shoot->leaf_nitrogen_gN.at(leaf_objID);

                            // Calculate minimum N for this leaf
                            float leaf_area = context_ptr->getObjectArea(leaf_objID);
                            float scale_factor = phytomer->current_leaf_scale_factor.at(petiole_idx);
                            float unscaled_area = leaf_area / (scale_factor * scale_factor);
                            float leaf_biomass = unscaled_area / N_params.SLA;
                            float min_N = leaf_biomass * N_params.minimum_leaf_N_concentration;

                            // Available for remobilization (limited by efficiency and minimum)
                            float remobilizable_N = std::max(0.0f,
                                (current_N - min_N) * N_params.leaf_remobilization_efficiency);

                            // Accelerate under N stress (up to 1.3x faster)
                            if (N_stress_factor < 1.0f) {
                                remobilizable_N *= (1.0f + 0.3f * (1.0f - N_stress_factor));
                            }

                            if (remobilizable_N > 0) {
                                source_leaves.push_back({shoot_idx, leaf_objID});
                                source_N_available[leaf_objID] = remobilizable_N;
                                total_source_N_available += remobilizable_N;
                            }

                        } else if (age_fraction < 0.5f) {
                            // YOUNG LEAF: Sink for remobilized N (<50% of lifespan)
                            float leaf_area = context_ptr->getObjectArea(leaf_objID);
                            float scale_factor = phytomer->current_leaf_scale_factor.at(petiole_idx);
                            float unscaled_area = leaf_area / (scale_factor * scale_factor);
                            float leaf_biomass = unscaled_area / N_params.SLA;
                            float target_N = leaf_biomass * N_params.target_leaf_N_concentration;
                            float current_N = shoot->leaf_nitrogen_gN.at(leaf_objID);

                            float demand = std::max(0.0f, target_N - current_N);
                            if (demand > 0) {
                                sink_leaves.push_back({shoot_idx, leaf_objID});
                                sink_N_demand[leaf_objID] = demand;
                                total_sink_demand += demand;
                            }
                        }
                    }
                }
            }
            shoot_idx++;
        }

        // Remobilize N from old to young leaves
        if (total_source_N_available > 0 && total_sink_demand > 0) {
            float N_to_remobilize = std::min(total_source_N_available, total_sink_demand);

            // Remove from source leaves proportionally
            for (auto& [shoot_idx, source_objID] : source_leaves) {
                float leaf_contribution = source_N_available[source_objID] / total_source_N_available;
                float N_removed = N_to_remobilize * leaf_contribution;

                plant.shoot_tree.at(shoot_idx)->leaf_nitrogen_gN[source_objID] -= N_removed;
            }

            // Add to sink leaves proportionally
            for (auto& [shoot_idx, sink_objID] : sink_leaves) {
                float leaf_demand_fraction = sink_N_demand[sink_objID] / total_sink_demand;
                float N_added = N_to_remobilize * leaf_demand_fraction;

                plant.shoot_tree.at(shoot_idx)->leaf_nitrogen_gN[sink_objID] += N_added;
            }
        }
    }
}

// ==================== Nitrogen Stress Factor Calculation ==================== //

void PlantArchitecture::updateNitrogenStressFactor() {
    for (auto& [plantID, plant] : plant_instances) {
        const NitrogenParameters& N_params = plant.nitrogen_parameters;

        // Calculate average leaf N concentration across all leaves
        float total_leaf_N = 0;
        float total_leaf_biomass = 0;
        int num_leaves = 0;

        for (auto& shoot : plant.shoot_tree) {
            for (auto& phytomer : shoot->phytomers) {
                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                        uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                        if (!context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }

                        if (!shoot->leaf_nitrogen_gN.count(leaf_objID)) {
                            continue;
                        }

                        float leaf_N = shoot->leaf_nitrogen_gN.at(leaf_objID);
                        float leaf_area = context_ptr->getObjectArea(leaf_objID);
                        float scale_factor = phytomer->current_leaf_scale_factor.at(petiole_idx);
                        float unscaled_area = leaf_area / (scale_factor * scale_factor);
                        float leaf_biomass = unscaled_area / N_params.SLA;

                        total_leaf_N += leaf_N;
                        total_leaf_biomass += leaf_biomass;
                        num_leaves++;
                    }
                }
            }
        }

        if (num_leaves == 0) {
            continue;  // No leaves to evaluate
        }

        // Calculate actual leaf N concentration
        float actual_leaf_N_concentration = total_leaf_N / total_leaf_biomass;

        // Calculate stress factor: simple ratio
        float stress_factor = std::min(1.0f,
            actual_leaf_N_concentration / N_params.target_leaf_N_concentration);

        // Clamp to [0, 1]
        stress_factor = std::clamp(stress_factor, 0.0f, 1.0f);

        // Write SINGLE output to plant object data
        std::vector<uint> plant_objIDs = getAllPlantObjectIDs(plantID);
        if (!plant_objIDs.empty()) {
            context_ptr->setObjectData(plant_objIDs, "nitrogen_stress_factor", stress_factor);
        }
    }
}

// ==================== Fruit Nitrogen Removal ==================== //

void PlantArchitecture::removeFruitNitrogen() {
    for (auto& [plantID, plant] : plant_instances) {
        const NitrogenParameters& N_params = plant.nitrogen_parameters;

        // Iterate through shoots to find fruit
        for (auto& shoot : plant.shoot_tree) {
            for (auto& phytomer : shoot->phytomers) {
                // Check each floral bud
                for (auto& petiole : phytomer->floral_buds) {
                    for (auto& fbud : petiole) {
                        if (fbud.state != BUD_FRUITING) {
                            continue;  // Not a fruit
                        }

                        // Check if fruit grew this timestep
                        float scale_increment = fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor;
                        if (scale_increment <= 0) {
                            continue;  // No growth
                        }

                        // Calculate fruit biomass increment
                        float fruit_biomass_increment = 0;
                        for (uint fruit_objID : fbud.inflorescence_objIDs) {
                            if (!context_ptr->doesObjectExist(fruit_objID)) {
                                continue;
                            }

                            // Get mature fruit volume (unscaled)
                            float mature_volume = context_ptr->getPolymeshObjectVolume(fruit_objID) /
                                                 fbud.current_fruit_scale_factor;

                            // Calculate volume increment
                            float volume_increment = mature_volume * scale_increment;

                            // Convert to biomass using fruit density (from CarbohydrateParameters)
                            float biomass_increment = volume_increment * plant.carb_parameters.fruit_density;  // g DW
                            fruit_biomass_increment += biomass_increment;
                        }

                        if (fruit_biomass_increment > 0) {
                            // Calculate N demand for fruit growth
                            float fruit_N_demand = fruit_biomass_increment * N_params.fruit_N_concentration;

                            // Deduct from available pool (supply-limited)
                            float N_removed = std::min(fruit_N_demand, plant.available_nitrogen_pool_gN);
                            plant.available_nitrogen_pool_gN -= N_removed;

                            // Safety clamp
                            if (plant.available_nitrogen_pool_gN < 0) {
                                plant.available_nitrogen_pool_gN = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}
