/** \file "NitrogenModel.cpp" Nitrogen model calculations for the PlantArchitecture plugin.

    Copyright (C) 2016-2026 Brian Bailey

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

void PlantArchitecture::setPlantNitrogenParameters(uint plantID, const NitrogenParameters &params) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantNitrogenParameters): Plant ID " + std::to_string(plantID) + " does not exist.");
    }
    plant_instances.at(plantID).nitrogen_parameters = params;
}

void PlantArchitecture::setPlantNitrogenParameters(const std::vector<uint> &plantIDs, const NitrogenParameters &params) {
    for (uint plantID: plantIDs) {
        setPlantNitrogenParameters(plantID, params);
    }
}

// ==================== Initialization Methods ==================== //

void PlantArchitecture::initializeNitrogenPools(float initial_leaf_N_area) {
    for (auto &[plantID, plant_instance]: plant_instances) {
        initializePlantNitrogenPools(plantID, initial_leaf_N_area);
    }
}

void PlantArchitecture::initializePlantNitrogenPools(uint plantID, float initial_leaf_N_area) {
    // Validate inputs
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantNitrogenPools): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    if (initial_leaf_N_area < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::initializePlantNitrogenPools): Initial leaf N content per area must be >= 0.");
    }

    PlantInstance &plant = plant_instances.at(plantID);

    // Initialize plant-level pools to zero
    plant.root_nitrogen_pool_gN = 0;
    plant.available_nitrogen_pool_gN = 0;
    plant.cumulative_N_uptake_gN = 0;

    // Initialize per-leaf nitrogen content (area basis) based on current leaf areas
    for (auto &shoot: plant.shoot_tree) {
        // Clear existing leaf N pools for this shoot
        shoot->leaf_nitrogen_gN_m2.clear();

        for (auto &phytomer: shoot->phytomers) {
            // Iterate through all leaves in this phytomer (2D vector structure)
            for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                    uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                    if (!context_ptr->doesObjectExist(leaf_objID)) {
                        continue;
                    }

                    // Get current leaf area
                    float leaf_area = context_ptr->getObjectArea(leaf_objID);

                    // Initialize leaf N content per area (g N/m²)
                    shoot->leaf_nitrogen_gN_m2[leaf_objID] = initial_leaf_N_area;

                    // Track total N added (for cumulative uptake)
                    float total_N_added = initial_leaf_N_area * leaf_area;
                    plant.cumulative_N_uptake_gN += total_N_added;
                }
            }
        }
    }
}

// ==================== Nitrogen Application ==================== //

void PlantArchitecture::addPlantNitrogen(uint plantID, float amount_gN) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addPlantNitrogen): Plant ID " + std::to_string(plantID) + " does not exist.");
    }
    if (amount_gN < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::addPlantNitrogen): Nitrogen amount must be >= 0.");
    }

    PlantInstance &plant = plant_instances.at(plantID);
    const NitrogenParameters &N_params = plant.nitrogen_parameters;

    // Split between root and available pools
    float root_N = amount_gN * N_params.root_allocation_fraction;
    float available_N = amount_gN * (1.0f - N_params.root_allocation_fraction);

    plant.root_nitrogen_pool_gN += root_N;
    plant.available_nitrogen_pool_gN += available_N;
    plant.cumulative_N_uptake_gN += amount_gN;
}

void PlantArchitecture::addPlantNitrogen(const std::vector<uint> &plantIDs, float amount_gN) {
    for (uint plantID: plantIDs) {
        addPlantNitrogen(plantID, amount_gN);
    }
}

// ==================== Leaf Nitrogen Accumulation (Rate-Limited, Area Basis) ==================== //

void PlantArchitecture::accumulateLeafNitrogen(float dt) {
    for (auto &[plantID, plant]: plant_instances) {
        const NitrogenParameters &N_params = plant.nitrogen_parameters;
        bool has_available_nitrogen = (plant.available_nitrogen_pool_gN > 0);

        // Iterate through all shoots and their leaves
        for (auto &shoot: plant.shoot_tree) {
            for (auto &phytomer: shoot->phytomers) {
                // Iterate through all leaves in this phytomer (2D vector structure)
                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                        uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                        if (!context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }

                        // Get current leaf N content per area (g N/m²)
                        // Initialize in map first so all leaves get tracked (even when no N available)
                        float current_N_area = 0;
                        if (shoot->leaf_nitrogen_gN_m2.count(leaf_objID)) {
                            current_N_area = shoot->leaf_nitrogen_gN_m2.at(leaf_objID);
                        } else {
                            // Initialize if not present
                            shoot->leaf_nitrogen_gN_m2[leaf_objID] = 0;
                        }

                        // Skip accumulation if no nitrogen available
                        if (!has_available_nitrogen) {
                            continue;
                        }

                        // Get current leaf area
                        float leaf_area = context_ptr->getObjectArea(leaf_objID);
                        if (leaf_area <= 0) {
                            continue; // Skip nitrogen accumulation for leaves with no area (prevents division by zero)
                        }

                        // Calculate N content per area demand (g N/m²)
                        float N_area_demand = std::max(0.0f, N_params.target_leaf_N_area - current_N_area);
                        if (N_area_demand <= 0) {
                            continue; // Leaf is at target
                        }

                        // Rate limiting: max N per m² per day
                        float max_N_area_this_step = N_params.max_N_accumulation_rate * dt;

                        // Calculate total N demand for this leaf (g N)
                        float total_N_demand = std::min(N_area_demand, max_N_area_this_step) * leaf_area;

                        // Also limited by available N pool
                        float N_to_add = std::min(total_N_demand, plant.available_nitrogen_pool_gN);

                        // Update leaf N content per area (g N/m²)
                        float N_area_to_add = N_to_add / leaf_area;
                        shoot->leaf_nitrogen_gN_m2[leaf_objID] += N_area_to_add;

                        // Update available pool
                        plant.available_nitrogen_pool_gN -= N_to_add;

                        if (plant.available_nitrogen_pool_gN <= 0) {
                            has_available_nitrogen = false; // Continue iterating to track all leaves, but skip accumulation
                        }
                    }
                }
            }
        }
    }
}

// ==================== Nitrogen Remobilization (Area Basis) ==================== //

void PlantArchitecture::remobilizeNitrogen(float dt) {
    for (auto &[plantID, plant]: plant_instances) {
        const NitrogenParameters &N_params = plant.nitrogen_parameters;

        // Calculate current N stress to determine remobilization intensity
        float total_N_area = 0; // Sum of N_area across all leaves
        float total_area = 0; // Total leaf area
        int num_leaves = 0;

        for (auto &shoot: plant.shoot_tree) {
            for (auto &[leaf_objID, leaf_N_area]: shoot->leaf_nitrogen_gN_m2) {
                if (!context_ptr->doesObjectExist(leaf_objID)) {
                    continue;
                }
                float leaf_area = context_ptr->getObjectArea(leaf_objID);
                total_N_area += leaf_N_area * leaf_area; // g N
                total_area += leaf_area; // m²
                num_leaves++;
            }
        }

        if (num_leaves == 0 || total_area == 0) {
            continue; // No leaves
        }

        float avg_N_area = total_N_area / total_area; // Average g N/m²
        float N_stress_factor = std::min(1.0f, avg_N_area / N_params.target_leaf_N_area);

        // Classify leaves by age
        std::vector<std::pair<uint, uint>> source_leaves; // (shoot_idx, leaf_objID)
        std::vector<std::pair<uint, uint>> sink_leaves; // (shoot_idx, leaf_objID)
        std::map<uint, float> source_N_available; // leaf_objID → available N (g N)
        std::map<uint, float> sink_N_demand; // leaf_objID → N demand (g N)
        float total_source_N_available = 0;
        float total_sink_demand = 0;

        uint shoot_idx = 0;
        for (auto &shoot: plant.shoot_tree) {
            for (auto &phytomer: shoot->phytomers) {
                // Get leaf lifespan from plant instance
                float leaf_lifespan = plant.max_leaf_lifespan;
                if (leaf_lifespan <= 0) {
                    leaf_lifespan = 1e6; // Evergreen
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

                        if (!shoot->leaf_nitrogen_gN_m2.count(leaf_objID)) {
                            continue;
                        }

                        float leaf_area = context_ptr->getObjectArea(leaf_objID);
                        float current_N_area = shoot->leaf_nitrogen_gN_m2.at(leaf_objID); // g N/m²

                        // Classify by age
                        if (age_fraction >= N_params.remobilization_age_threshold) {
                            // OLD LEAF: Source for remobilization (>70% of lifespan)

                            // Calculate remobilizable N per area (g N/m²)
                            float remobilizable_N_area = std::max(0.0f, (current_N_area - N_params.minimum_leaf_N_area) * N_params.leaf_remobilization_efficiency);

                            // Accelerate under N stress (up to 1.3x faster)
                            if (N_stress_factor < 1.0f) {
                                remobilizable_N_area *= (1.0f + 0.3f * (1.0f - N_stress_factor));
                            }

                            // Convert to total N available (g N)
                            float remobilizable_N_total = remobilizable_N_area * leaf_area;

                            if (remobilizable_N_total > 0) {
                                source_leaves.push_back({shoot_idx, leaf_objID});
                                source_N_available[leaf_objID] = remobilizable_N_total;
                                total_source_N_available += remobilizable_N_total;
                            }

                        } else if (age_fraction < 0.5f) {
                            // YOUNG LEAF: Sink for remobilized N (<50% of lifespan)

                            // Calculate N demand per area (g N/m²)
                            float N_area_demand = std::max(0.0f, N_params.target_leaf_N_area - current_N_area);

                            // Convert to total N demand (g N)
                            float N_total_demand = N_area_demand * leaf_area;

                            if (N_total_demand > 0) {
                                sink_leaves.push_back({shoot_idx, leaf_objID});
                                sink_N_demand[leaf_objID] = N_total_demand;
                                total_sink_demand += N_total_demand;
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
            for (auto &[shoot_idx, source_objID]: source_leaves) {
                float leaf_contribution = source_N_available[source_objID] / total_source_N_available;
                float N_removed_total = N_to_remobilize * leaf_contribution; // g N

                // Convert to per-area and update
                float source_leaf_area = context_ptr->getObjectArea(source_objID);
                if (source_leaf_area <= 0) {
                    continue; // Skip leaves with no area (prevents division by zero)
                }
                float N_removed_area = N_removed_total / source_leaf_area; // g N/m²
                plant.shoot_tree.at(shoot_idx)->leaf_nitrogen_gN_m2[source_objID] -= N_removed_area;
            }

            // Add to sink leaves proportionally
            for (auto &[shoot_idx, sink_objID]: sink_leaves) {
                float leaf_demand_fraction = sink_N_demand[sink_objID] / total_sink_demand;
                float N_added_total = N_to_remobilize * leaf_demand_fraction; // g N

                // Convert to per-area and update
                float sink_leaf_area = context_ptr->getObjectArea(sink_objID);
                if (sink_leaf_area <= 0) {
                    continue; // Skip leaves with no area (prevents division by zero)
                }
                float N_added_area = N_added_total / sink_leaf_area; // g N/m²
                plant.shoot_tree.at(shoot_idx)->leaf_nitrogen_gN_m2[sink_objID] += N_added_area;
            }
        }
    }
}

// ==================== Nitrogen Stress Factor Calculation (Area Basis) ==================== //

void PlantArchitecture::updateNitrogenStressFactor() {
    for (auto &[plantID, plant]: plant_instances) {
        const NitrogenParameters &N_params = plant.nitrogen_parameters;

        // Calculate area-weighted average leaf N content per area across all leaves
        float total_N = 0; // Total nitrogen (g N)
        float total_area = 0; // Total leaf area (m²)
        int num_leaves = 0;

        for (auto &shoot: plant.shoot_tree) {
            for (auto &phytomer: shoot->phytomers) {
                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                        uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                        if (!context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }

                        if (!shoot->leaf_nitrogen_gN_m2.count(leaf_objID)) {
                            continue;
                        }

                        float leaf_N_area = shoot->leaf_nitrogen_gN_m2.at(leaf_objID); // g N/m²
                        float leaf_area = context_ptr->getObjectArea(leaf_objID); // m²

                        // Write leaf nitrogen per area to object data for visualization
                        context_ptr->setObjectData(leaf_objID, "leaf_nitrogen_gN_m2", leaf_N_area);

                        total_N += leaf_N_area * leaf_area; // g N
                        total_area += leaf_area; // m²
                        num_leaves++;
                    }
                }
            }
        }

        if (num_leaves == 0 || total_area == 0) {
            continue; // No leaves to evaluate
        }

        // Calculate average leaf N content per area (g N/m²)
        float avg_leaf_N_area = total_N / total_area;

        // Calculate stress factor: simple ratio
        float stress_factor = std::min(1.0f, avg_leaf_N_area / N_params.target_leaf_N_area);

        // Clamp to [0, 1]
        stress_factor = std::clamp(stress_factor, 0.0f, 1.0f);

        // Write SINGLE output to plant object data
        std::vector<uint> plant_objIDs = getAllPlantObjectIDs(plantID);
        if (!plant_objIDs.empty()) {
            context_ptr->setObjectData(plant_objIDs, "nitrogen_stress_factor", stress_factor);
        }
    }
}

// ==================== Fruit Nitrogen Removal (Area Basis) ==================== //

void PlantArchitecture::removeFruitNitrogen() {
    for (auto &[plantID, plant]: plant_instances) {
        const NitrogenParameters &N_params = plant.nitrogen_parameters;

        // ---- Pass A: aggregate per-plant fruit nitrogen demand ----
        float total_fruit_demand_gN = 0;
        for (auto &shoot: plant.shoot_tree) {
            for (auto &phytomer: shoot->phytomers) {
                for (auto &petiole: phytomer->floral_buds) {
                    for (auto &fbud: petiole) {
                        if (fbud.state != BUD_FRUITING) {
                            continue; // Not a fruit
                        }

                        // Check if fruit grew this timestep
                        float scale_increment = fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor;
                        if (scale_increment <= 0) {
                            continue; // No growth
                        }

                        // Calculate fruit area increment
                        float fruit_area_increment = 0;
                        for (uint fruit_objID: fbud.inflorescence_objIDs) {
                            if (!context_ptr->doesObjectExist(fruit_objID)) {
                                continue;
                            }
                            float current_fruit_area = context_ptr->getObjectArea(fruit_objID);
                            float mature_fruit_area = current_fruit_area / (fbud.current_fruit_scale_factor * fbud.current_fruit_scale_factor);
                            float area_increment = mature_fruit_area * (fbud.current_fruit_scale_factor * fbud.current_fruit_scale_factor - fbud.previous_fruit_scale_factor * fbud.previous_fruit_scale_factor);
                            fruit_area_increment += area_increment;
                        }

                        if (fruit_area_increment > 0) {
                            total_fruit_demand_gN += fruit_area_increment * N_params.fruit_N_area;
                        }
                    }
                }
            }
        }

        if (total_fruit_demand_gN <= 0) {
            continue; // No fruit growth this timestep
        }

        // ---- Deduct from available pool first (supply-limited) ----
        float pool_supply_gN = std::min(total_fruit_demand_gN, plant.available_nitrogen_pool_gN);
        plant.available_nitrogen_pool_gN -= pool_supply_gN;
        if (plant.available_nitrogen_pool_gN < 0) {
            plant.available_nitrogen_pool_gN = 0; // Safety clamp
        }
        float leaf_shortfall_gN = total_fruit_demand_gN - pool_supply_gN;
        if (leaf_shortfall_gN <= 0) {
            continue; // Pool fully covered demand
        }

        // ---- Pass B: classify leaves into old (priority sources) and young (fallback sources) ----
        // Translocation magnitude is governed by fruit demand, not stress signaling, so no stress-acceleration multiplier
        // is applied here (unlike remobilizeNitrogen). Per-leaf availability uses the same formula:
        //   max(0, (current_N_area - minimum_leaf_N_area) * leaf_remobilization_efficiency).
        std::vector<std::pair<uint, uint>> old_sources;   // (shoot_idx, leaf_objID)
        std::vector<std::pair<uint, uint>> young_sources; // (shoot_idx, leaf_objID)
        std::map<uint, float> source_N_available_gN;      // leaf_objID -> available N (g N)
        float total_old_available_gN = 0;
        float total_young_available_gN = 0;

        uint shoot_idx = 0;
        for (auto &shoot: plant.shoot_tree) {
            for (auto &phytomer: shoot->phytomers) {
                float leaf_lifespan = plant.max_leaf_lifespan;
                if (leaf_lifespan <= 0) {
                    leaf_lifespan = 1e6; // Evergreen
                }
                float age_fraction = phytomer->age / leaf_lifespan;
                bool is_old = (age_fraction >= N_params.remobilization_age_threshold);

                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                        uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);
                        if (!context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }
                        if (!shoot->leaf_nitrogen_gN_m2.count(leaf_objID)) {
                            continue;
                        }
                        float leaf_area = context_ptr->getObjectArea(leaf_objID);
                        if (leaf_area <= 0) {
                            continue;
                        }
                        float current_N_area = shoot->leaf_nitrogen_gN_m2.at(leaf_objID); // g N/m²
                        float available_N_area = std::max(0.0f, (current_N_area - N_params.minimum_leaf_N_area) * N_params.leaf_remobilization_efficiency);
                        float available_N_total = available_N_area * leaf_area;
                        if (available_N_total <= 0) {
                            continue;
                        }

                        source_N_available_gN[leaf_objID] = available_N_total;
                        if (is_old) {
                            old_sources.push_back({shoot_idx, leaf_objID});
                            total_old_available_gN += available_N_total;
                        } else {
                            young_sources.push_back({shoot_idx, leaf_objID});
                            total_young_available_gN += available_N_total;
                        }
                    }
                }
            }
            shoot_idx++;
        }

        // ---- Pass C: withdraw shortfall from old leaves first, then young leaves as fallback ----
        // Withdrawal is distributed proportionally to per-leaf available N (matches the remobilizeNitrogen pattern at lines 316-327).
        // `plant` is passed by reference rather than captured to avoid the C++20-extension warning for capturing structured bindings.
        PlantInstance &plant_ref = plant;
        auto withdraw_from_sources = [&plant_ref, &source_N_available_gN, &leaf_shortfall_gN, this](const std::vector<std::pair<uint, uint>> &sources, float total_available_gN) {
            if (leaf_shortfall_gN <= 0 || total_available_gN <= 0) {
                return;
            }
            float withdraw_total_gN = std::min(leaf_shortfall_gN, total_available_gN);
            for (auto &[idx, leaf_objID]: sources) {
                float leaf_share = source_N_available_gN.at(leaf_objID) / total_available_gN;
                float N_removed_total = withdraw_total_gN * leaf_share; // g N
                float leaf_area = context_ptr->getObjectArea(leaf_objID);
                if (leaf_area <= 0) {
                    continue;
                }
                float N_removed_area = N_removed_total / leaf_area; // g N/m²
                plant_ref.shoot_tree.at(idx)->leaf_nitrogen_gN_m2[leaf_objID] -= N_removed_area;
            }
            leaf_shortfall_gN -= withdraw_total_gN;
        };

        withdraw_from_sources(old_sources, total_old_available_gN);
        withdraw_from_sources(young_sources, total_young_available_gN);

        // Residual leaf_shortfall_gN (when even leaves cannot cover the demand) is silently absorbed,
        // preserving the preexisting fail-soft behavior of fruit nitrogen accounting.
    }
}
