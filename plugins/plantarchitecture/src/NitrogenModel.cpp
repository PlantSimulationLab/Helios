/** \file "NitrogenModel.cpp" Nitrogen model calculations for the PlantArchitecture plugin.

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

    //! Nitrogen one leaf wants this timestep (g N), with what is needed to turn it into a per-area change
    struct LeafNitrogenDemand {
        Shoot *shoot;
        uint leaf_objID;
        float area; //!< one-sided leaf area (m^2)
        float demand_gN;
    };

    //! Share the plant's available pool across a set of leaves in proportion to their demand
    /**
     * \return The fraction of the leaves' combined demand that the pool covered (1 when there was no demand).
     */
    float supplyLeafDemand(PlantInstance &plant, const std::vector<LeafNitrogenDemand> &leaves, float total_demand_gN) {
        if (total_demand_gN <= 0) {
            return 1.f;
        }
        const float supplied_gN = std::min(total_demand_gN, std::max(0.f, plant.available_nitrogen_pool_gN));
        const float fraction_supplied = supplied_gN / total_demand_gN;
        for (const LeafNitrogenDemand &leaf: leaves) {
            leaf.shoot->leaf_nitrogen_gN_m2[leaf.leaf_objID] += fraction_supplied * leaf.demand_gN / leaf.area;
        }
        plant.available_nitrogen_pool_gN -= supplied_gN;
        return fraction_supplied;
    }

    //! Nitrogen per area (g N/m^2) a senescing leaf is shed with: what it held above the minimum when senescence began, less the resorbed fraction
    float senescentLeafFinalNitrogen(const NitrogenParameters &N_params, float N_area_at_onset) {
        if (N_area_at_onset <= N_params.minimum_leaf_N_area) {
            return N_area_at_onset;
        }
        return N_params.minimum_leaf_N_area + (1.f - N_params.leaf_remobilization_efficiency) * (N_area_at_onset - N_params.minimum_leaf_N_area);
    }

    //! Whether the leaves on one petiole of a phytomer are still expanding toward full size
    bool leavesStillExpanding(const Phytomer &phytomer, uint petiole_index) {
        return petiole_index < phytomer.current_leaf_scale_factor.size() && phytomer.current_leaf_scale_factor.at(petiole_index) < 1.f;
    }

} // namespace

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
    plant.unmet_fruit_N_demand_gN = 0;

    // Initialize per-leaf nitrogen content (area basis) based on current leaf areas
    for (auto &shoot: plant.shoot_tree) {
        // Clear existing leaf N pools for this shoot
        shoot->leaf_nitrogen_gN_m2.clear();
        shoot->leaf_N_unmet_demand_gN.clear();
        shoot->leaf_N_area_basis_m2.clear();
        shoot->leaf_N_at_senescence_onset.clear();

        for (auto &phytomer: shoot->phytomers) {
            // Iterate through all leaves in this phytomer (2D vector structure)
            for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                for (uint leaf_idx = 0; leaf_idx < phytomer->leaf_objIDs.at(petiole_idx).size(); leaf_idx++) {
                    uint leaf_objID = phytomer->leaf_objIDs.at(petiole_idx).at(leaf_idx);

                    if (!context_ptr->doesObjectExist(leaf_objID)) {
                        continue;
                    }

                    // Get current leaf area
                    float leaf_area = getLeafBladeArea(leaf_objID);

                    // Initialize leaf N content per area (g N/m²)
                    shoot->leaf_nitrogen_gN_m2[leaf_objID] = initial_leaf_N_area;
                    shoot->leaf_N_area_basis_m2[leaf_objID] = leaf_area;

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

float PlantArchitecture::getPlantAvailableNitrogen(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantAvailableNitrogen): Plant ID " + std::to_string(plantID) + " does not exist.");
    }
    return plant_instances.at(plantID).available_nitrogen_pool_gN;
}

// ==================== Leaf Senescence ==================== //

void PlantArchitecture::senesceLeafNitrogen(float dt) {
    for (auto &[plantID, plant]: plant_instances) {
        const NitrogenParameters &N_params = plant.nitrogen_parameters;
        if (N_params.leaf_senescence_duration_fraction < 0 || N_params.stress_senescence_advance_fraction < 0 || N_params.leaf_senescence_duration_fraction + N_params.stress_senescence_advance_fraction > 1.f) {
            helios_runtime_error("ERROR (PlantArchitecture::senesceLeafNitrogen): NitrogenParameters::leaf_senescence_duration_fraction (" + std::to_string(N_params.leaf_senescence_duration_fraction) +
                                 ") and stress_senescence_advance_fraction (" + std::to_string(N_params.stress_senescence_advance_fraction) + ") must be non-negative and sum to at most 1 for plant " +
                                 std::to_string(plantID) + ".");
        }

        // A leaf does not keep its nitrogen until the day it is shed: it senesces over the last part of its life,
        // resorbing leaf_remobilization_efficiency of its nitrogen above the minimum and yellowing as it does, which
        // is why the oldest leaves of a plant are the palest. Nitrogen stress brings senescence forward, so a starved
        // plant sheds the nitrogen of its oldest leaves to its young ones sooner.
        const float lifespan = plant.max_leaf_lifespan;
        const float onset_fraction = 1.f - N_params.leaf_senescence_duration_fraction - N_params.stress_senescence_advance_fraction * (1.f - std::clamp(plant.nitrogen_stress_factor, 0.f, 1.f));
        const float onset_age = onset_fraction * lifespan;

        for (auto &shoot: plant.shoot_tree) {
            for (auto &phytomer: shoot->phytomers) {
                if (phytomer->age < onset_age) {
                    continue;
                }
                // Senescence ends when the leaf is shed at max_leaf_lifespan: the rest of what it resorbs is spread
                // evenly over the time it has left.
                const float time_remaining = lifespan - phytomer->age;
                const float fraction_this_step = (time_remaining > dt) ? dt / time_remaining : 1.f;
                for (const auto &petiole_leaves: phytomer->leaf_objIDs) {
                    for (uint leaf_objID: petiole_leaves) {
                        const auto leaf_N = shoot->leaf_nitrogen_gN_m2.find(leaf_objID);
                        const auto area_basis = shoot->leaf_N_area_basis_m2.find(leaf_objID);
                        if (leaf_N == shoot->leaf_nitrogen_gN_m2.end() || area_basis == shoot->leaf_N_area_basis_m2.end()) {
                            continue;
                        }
                        const float N_area_at_onset = shoot->leaf_N_at_senescence_onset.emplace(leaf_objID, leaf_N->second).first->second;
                        const float resorbed_N_area = std::max(0.f, leaf_N->second - senescentLeafFinalNitrogen(N_params, N_area_at_onset)) * fraction_this_step;
                        leaf_N->second -= resorbed_N_area;
                        plant.available_nitrogen_pool_gN += resorbed_N_area * area_basis->second;
                    }
                }
            }
        }
    }
}

// ==================== Leaf Nitrogen Accumulation (Rate-Limited, Area Basis) ==================== //

void PlantArchitecture::accumulateLeafNitrogen(float dt) {
    for (auto &[plantID, plant]: plant_instances) {
        const NitrogenParameters &N_params = plant.nitrogen_parameters;
        const float max_N_area_this_step = N_params.max_N_accumulation_rate * dt;

        // Newly acquired nitrogen goes to the leaves that are still expanding ("nitrogen taken up from roots is
        // allocated to new leaves", Hikosaka 2005), and a mature leaf below its target is topped up only from what
        // they leave over. Within each group the supply is shared in proportion to demand, as APSIM's relative
        // allocation does. Handing the pool out in storage order until it ran dry -- which is what this replaced --
        // gave a limited supply to the oldest leaves on the plant and left every new leaf with none: the reverse of
        // the pattern plants show, where nitrogen is mobile and a shortage yellows the oldest leaves first.
        std::vector<LeafNitrogenDemand> expanding_leaves;
        std::vector<LeafNitrogenDemand> mature_leaves;
        float expanding_demand_gN = 0;
        float mature_demand_gN = 0;

        for (auto &shoot: plant.shoot_tree) {
            shoot->leaf_N_unmet_demand_gN.clear();
            for (auto &phytomer: shoot->phytomers) {
                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    const bool expanding = leavesStillExpanding(*phytomer, petiole_idx);
                    for (uint leaf_objID: phytomer->leaf_objIDs.at(petiole_idx)) {
                        if (!context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }

                        // Every leaf is tracked, including a new one and one that receives nothing this step.
                        float &leaf_N_area = shoot->leaf_nitrogen_gN_m2[leaf_objID];

                        const float leaf_area = getLeafBladeArea(leaf_objID);
                        if (leaf_area <= 0) {
                            continue; // no area to hold nitrogen (prevents division by zero)
                        }

                        // Nitrogen is tracked per unit area but held as a mass, so a leaf that has grown since its
                        // nitrogen was last stated holds the same nitrogen spread over more area. Carrying the per-area
                        // value onto the new area instead created nitrogen from nothing -- most of the nitrogen an
                        // expanding leaf appeared to gain -- and let a starved canopy green up with no supply at all.
                        const auto area_basis = shoot->leaf_N_area_basis_m2.find(leaf_objID);
                        if (area_basis != shoot->leaf_N_area_basis_m2.end() && area_basis->second > 0) {
                            leaf_N_area *= area_basis->second / leaf_area;
                        }
                        shoot->leaf_N_area_basis_m2[leaf_objID] = leaf_area;
                        const float current_N_area = leaf_N_area;
                        if (shoot->leaf_N_at_senescence_onset.count(leaf_objID) > 0) {
                            continue; // a senescing leaf exports nitrogen; it does not take any up
                        }

                        const float N_area_demand = std::min(std::max(0.0f, N_params.target_leaf_N_area - current_N_area), max_N_area_this_step);
                        if (N_area_demand <= 0) {
                            continue;
                        }

                        const LeafNitrogenDemand demand{shoot.get(), leaf_objID, leaf_area, N_area_demand * leaf_area};
                        if (expanding) {
                            expanding_leaves.push_back(demand);
                            expanding_demand_gN += demand.demand_gN;
                        } else {
                            mature_leaves.push_back(demand);
                            mature_demand_gN += demand.demand_gN;
                        }
                    }
                }
            }
        }

        // What the pool cannot give the expanding leaves is recorded for remobilizeNitrogen() to supply.
        const float expanding_fraction_supplied = supplyLeafDemand(plant, expanding_leaves, expanding_demand_gN);
        if (expanding_fraction_supplied < 1.f) {
            for (const LeafNitrogenDemand &leaf: expanding_leaves) {
                leaf.shoot->leaf_N_unmet_demand_gN[leaf.leaf_objID] = (1.f - expanding_fraction_supplied) * leaf.demand_gN;
            }
        }
        supplyLeafDemand(plant, mature_leaves, mature_demand_gN);
    }
}

// ==================== Nitrogen Remobilization (Area Basis) ==================== //

void PlantArchitecture::remobilizeNitrogen(float dt) {
    for (auto &[plantID, plant]: plant_instances) {
        const NitrogenParameters &N_params = plant.nitrogen_parameters;

        // Remobilization answers a shortfall: what uptake could not give the expanding leaves this step, and fruit
        // growth the pool could not cover. A plant that is not short of nitrogen remobilizes nothing. It is not
        // gated on a leaf reaching a fixed fraction of its lifespan: under nitrogen limitation mature leaves export
        // to expanding ones during vegetative growth (Masclaux-Daubresse et al. 2010), and that fraction of a
        // lifespan most library plants leave effectively infinite would otherwise never be reached.
        std::vector<LeafNitrogenDemand> sinks;
        float leaf_unmet_gN = 0;
        for (auto &shoot: plant.shoot_tree) {
            for (const auto &[leaf_objID, unmet_gN]: shoot->leaf_N_unmet_demand_gN) {
                if (unmet_gN <= 0 || !context_ptr->doesObjectExist(leaf_objID)) {
                    continue;
                }
                const float leaf_area = getLeafBladeArea(leaf_objID);
                if (leaf_area <= 0) {
                    continue;
                }
                sinks.push_back({shoot.get(), leaf_objID, leaf_area, unmet_gN});
                leaf_unmet_gN += unmet_gN;
            }
        }

        const float fruit_unmet_gN = plant.unmet_fruit_N_demand_gN;
        plant.unmet_fruit_N_demand_gN = 0;
        const float total_unmet_gN = leaf_unmet_gN + fruit_unmet_gN;
        if (total_unmet_gN <= 0) {
            continue;
        }

        // Mature leaves are the sources, oldest first -- the leaves a shortage yellows first. Release is first-order
        // in time: each can give up at most leaf_remobilization_rate * dt of its nitrogen above the minimum this
        // step, so the amount moved follows elapsed time rather than the number of steps taken.
        struct LeafNitrogenSource {
            Shoot *shoot;
            uint leaf_objID;
            float area;
            float age;
            float available_gN;
        };
        const float release_fraction = std::clamp(N_params.leaf_remobilization_rate * dt, 0.f, 1.f);
        std::vector<LeafNitrogenSource> sources;
        for (auto &shoot: plant.shoot_tree) {
            for (auto &phytomer: shoot->phytomers) {
                for (uint petiole_idx = 0; petiole_idx < phytomer->leaf_objIDs.size(); petiole_idx++) {
                    if (leavesStillExpanding(*phytomer, petiole_idx)) {
                        continue;
                    }
                    for (uint leaf_objID: phytomer->leaf_objIDs.at(petiole_idx)) {
                        const auto leaf_N = shoot->leaf_nitrogen_gN_m2.find(leaf_objID);
                        if (leaf_N == shoot->leaf_nitrogen_gN_m2.end() || !context_ptr->doesObjectExist(leaf_objID)) {
                            continue;
                        }
                        const float leaf_area = getLeafBladeArea(leaf_objID);
                        const float available_gN = release_fraction * std::max(0.0f, leaf_N->second - N_params.minimum_leaf_N_area) * leaf_area;
                        if (leaf_area > 0 && available_gN > 0) {
                            sources.push_back({shoot.get(), leaf_objID, leaf_area, phytomer->age, available_gN});
                        }
                    }
                }
            }
        }
        std::stable_sort(sources.begin(), sources.end(), [](const LeafNitrogenSource &a, const LeafNitrogenSource &b) { return a.age > b.age; });

        float remaining_gN = total_unmet_gN;
        float collected_gN = 0;
        for (const LeafNitrogenSource &source: sources) {
            if (remaining_gN <= 0) {
                break;
            }
            const float withdrawn_gN = std::min(source.available_gN, remaining_gN);
            source.shoot->leaf_nitrogen_gN_m2[source.leaf_objID] -= withdrawn_gN / source.area;
            collected_gN += withdrawn_gN;
            remaining_gN -= withdrawn_gN;
        }
        if (collected_gN <= 0) {
            continue;
        }

        // Delivered in proportion to unmet demand, which was already capped at max_N_accumulation_rate * dt per leaf.
        // The fruit's share leaves the leaves with the fruit; fruit nitrogen is not tracked as a pool.
        const float delivered_fraction = collected_gN / total_unmet_gN;
        for (const LeafNitrogenDemand &sink: sinks) {
            sink.shoot->leaf_nitrogen_gN_m2[sink.leaf_objID] += delivered_fraction * sink.demand_gN / sink.area;
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
                        float leaf_area = getLeafBladeArea(leaf_objID); // m²

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

        plant.nitrogen_stress_factor = stress_factor;

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
        plant.unmet_fruit_N_demand_gN = 0;

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
        // Whatever the pool could not cover is drawn from mature leaves by remobilizeNitrogen(), which meets it
        // together with the unmet demand of expanding leaves and at the same first-order rate. Demand even the
        // leaves cannot meet is absorbed, as the pool's shortfall always has been.
        plant.unmet_fruit_N_demand_gN = total_fruit_demand_gN - pool_supply_gN;
    }
}

// ==================== Resorption Before Leaf Shedding ==================== //

void PlantArchitecture::resorbLeafNitrogen(PlantInstance &plant, Shoot &shoot, const Phytomer &phytomer) {
    // A leaf reaching the end of its life has, by then, resorbed leaf_remobilization_efficiency of the nitrogen it
    // held above minimum_leaf_N_area when it began to senesce (senesceLeafNitrogen()). Whatever of that remains --
    // all of it for a leaf shed before it began senescing, part of it when a long time step overshoots the lifespan --
    // is returned here, and the rest leaves with the leaf.
    const NitrogenParameters &N_params = plant.nitrogen_parameters;
    for (const auto &petiole_leaves: phytomer.leaf_objIDs) {
        for (uint leaf_objID: petiole_leaves) {
            const auto leaf_N = shoot.leaf_nitrogen_gN_m2.find(leaf_objID);
            if (leaf_N == shoot.leaf_nitrogen_gN_m2.end()) {
                continue;
            }
            const auto onset = shoot.leaf_N_at_senescence_onset.find(leaf_objID);
            const float N_area_at_onset = (onset != shoot.leaf_N_at_senescence_onset.end()) ? onset->second : leaf_N->second;
            // The area at which the leaf's nitrogen per area was last stated, not its current area: a leaf can
            // grow in the step it is shed, before that growth has diluted its nitrogen per area, and the current
            // area would count nitrogen the leaf never held.
            const auto area_basis = shoot.leaf_N_area_basis_m2.find(leaf_objID);
            if (area_basis != shoot.leaf_N_area_basis_m2.end()) {
                plant.available_nitrogen_pool_gN += std::max(0.f, leaf_N->second - senescentLeafFinalNitrogen(N_params, N_area_at_onset)) * std::max(0.f, area_basis->second);
            }
            shoot.leaf_nitrogen_gN_m2.erase(leaf_N);
            shoot.leaf_N_unmet_demand_gN.erase(leaf_objID);
            shoot.leaf_N_area_basis_m2.erase(leaf_objID);
            shoot.leaf_N_at_senescence_onset.erase(leaf_objID);
        }
    }
}
