/** \file "CarbohydrateModelGrass.cpp" The grass and grain sub-model (CARBON_SUBMODEL_GRASS) of the carbohydrate model: tiller outgrowth gated by carbon supply, tillers shed during stem
    elongation, a panicle built in a critical period that sets the grain number, and grain filled from current photosynthate and then from the culm reserve.

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

    //! The fruiting terminal bud of a culm -- its panicle -- or nullptr while the culm has not reached its flag leaf
    FloralBud *culmPanicle(const Shoot &shoot) {
        if (shoot.phytomers.empty()) {
            return nullptr;
        }
        // Shoot::addTerminalFloralBud() places the terminal bud on the last phytomer
        for (auto &petiole: shoot.phytomers.back()->floral_buds) {
            for (auto &fbud: petiole) {
                if (fbud.isterminal && fbud.state == BUD_FRUITING) {
                    return &fbud;
                }
            }
        }
        return nullptr;
    }

    //! The plant's main stem: its base shoot, the only shoot without a parent
    Shoot *mainStem(const PlantInstance &plant_instance) {
        for (const auto &shoot: plant_instance.shoot_tree) {
            if (shoot && !shoot->isPruned() && shoot->parent_shoot_ID < 0) {
                return shoot.get();
            }
        }
        return nullptr;
    }

} // namespace

// -------------------- Development -------------------- //

float PlantArchitecture::grassMainStemDevelopmentFraction(uint plantID) const {
    Shoot *main_stem = mainStem(plant_instances.at(plantID));
    if (main_stem == nullptr) {
        return 0.f;
    }
    const int node_cap = main_stem->getGrowthNodeCap();
    if (node_cap <= 0) {
        return 0.f;
    }
    return std::clamp(float(main_stem->current_node_number) / float(node_cap), 0.f, 1.f);
}

float PlantArchitecture::grassTillerBudBreakFactor(uint plantID) const {
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbonGrassParameters &grass_params = plant_instance.carb_parameters.grass;

    // Tillering ends when the elongating stem becomes the dominant sink
    if (plant_instance.grass_stem_elongation_started) {
        return 0.f;
    }
    // The index is undefined until the plant has recorded a vegetative demand
    const float demand = plant_instance.vegetative_demand_rate_molC_day;
    if (!(demand > 0.f)) {
        return 1.f;
    }
    const float index = std::max(0.f, plant_instance.photosynthesis_supply_rate_molC_day) / demand;
    return std::clamp((index - grass_params.tiller_supply_demand_min) / (grass_params.tiller_supply_demand_max - grass_params.tiller_supply_demand_min), 0.f, 1.f);
}

// -------------------- Panicle and grain -------------------- //

void PlantArchitecture::subtractGrainGrowthCarbon(uint plantID, float dt, double reserve_funded_spending_molC) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    const CarbonGrassParameters &grass_params = carbohydrate_params.grass;
    PlantCarbonBudget &budget = plant_instance.carbon_budget;

    const float critical_period_days = grass_params.anthesis_days + grass_params.grain_fill_lag_days;
    const float fill_end_days = critical_period_days + grass_params.grain_fill_days;

    // Each culm's panicle and what it could grow this sub-step. The panicle's clock is the days since it appeared with the flag leaf, which has already been advanced for this sub-step.
    struct Culm {
        Shoot *shoot;
        FloralBud *fbud;
        float panicle_potential; // structural carbon the panicle could lay down this sub-step (mol C)
        float grain_potential; // structural carbon the grain could lay down this sub-step (mol C)
    };
    std::vector<Culm> culms;
    double panicle_demand = 0.0; // with growth respiration (mol C)
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        FloralBud *fbud = culmPanicle(*shoot);
        if (fbud == nullptr) {
            continue;
        }
        const float t_end = fbud->time_counter;
        const float t_start = std::max(0.f, t_end - dt);
        float panicle_potential = 0.f;
        if (t_start < critical_period_days) {
            // Built at a constant potential rate over the critical period
            panicle_potential = grass_params.panicle_potential_carbon * (std::min(t_end, critical_period_days) - t_start) / critical_period_days;
        }
        culms.push_back({shoot.get(), fbud, panicle_potential, 0.f});
        panicle_demand += panicle_potential * (1.f + grass_params.panicle_growth_respiration_fraction);
    }

    // The panicle is built from current photosynthate: what is left after maintenance respiration and the reserve-funded stem, leaf and root growth of the sub-step, relative to the
    // panicles' potential growth, averaged over the supply-averaging period. Cereals are still laying down stem reserves at this stage, so a shortfall is not made up from them.
    const double photosynthate = plant_instance.photosynthesis_supply_rate_molC_day * dt;
    const double photosynthate_after_vegetative = std::max(0.0, photosynthate - plant_instance.step_maintenance_respiration_molC - reserve_funded_spending_molC);
    const float supply_ratio = recordReproductiveSupply(plantID, float(photosynthate_after_vegetative), float(panicle_demand), dt);

    double spent_on_panicle = 0.0;
    for (Culm &culm: culms) {
        if (culm.panicle_potential <= 0.f) {
            continue;
        }
        const float growth = culm.panicle_potential * supply_ratio;
        const float growth_respiration = growth * grass_params.panicle_growth_respiration_fraction;
        culm.shoot->sugar_pool_molC -= growth + growth_respiration;
        culm.shoot->season_growth_molC += growth + growth_respiration;
        culm.shoot->season_fruit_cost_molC += growth + growth_respiration;
        culm.fbud->grain.panicle_carbon_molC += growth;
        budget.panicle_growth += growth;
        budget.growth_respiration += growth_respiration;
        spent_on_panicle += growth + growth_respiration;
    }

    // At the end of the critical period the grain number is fixed from the panicle built: a shortfall in the window costs grain number, which later carbon cannot restore
    for (Culm &culm: culms) {
        GrainSink &grain = culm.fbud->grain;
        if (!grain.grain_number_set && culm.fbud->time_counter >= critical_period_days) {
            const float panicle_dry_mass_g = grain.panicle_carbon_molC * C_molecular_wt / grass_params.panicle_carbon_percentage;
            grain.grain_number = grass_params.grains_per_gram_panicle * panicle_dry_mass_g;
            grain.grain_number_set = true;
        }
    }

    // Grain fills at a constant potential rate over the filling period
    double grain_demand = 0.0; // with growth respiration (mol C)
    for (Culm &culm: culms) {
        const GrainSink &grain = culm.fbud->grain;
        if (!grain.grain_number_set || grain.grain_number <= 0.f) {
            continue;
        }
        const float t_end = std::clamp(culm.fbud->time_counter, critical_period_days, fill_end_days);
        const float t_start = std::clamp(culm.fbud->time_counter - dt, critical_period_days, fill_end_days);
        culm.grain_potential = grain.grain_number * grass_params.kernel_potential_carbon * (t_end - t_start) / grass_params.grain_fill_days;
        grain_demand += culm.grain_potential * (1.f + grass_params.grain_growth_respiration_fraction);
    }
    if (grain_demand <= 0.0) {
        return;
    }

    // Grain is paid from the photosynthate left after the panicle first, then from the carbohydrate the culms and roots hold above the reserve floor: the stem reserve is remobilized down to
    // that concentration, which sits above the one at which the culm is short of carbon so that filling does not starve it, and the share it contributes is an outcome of the balance
    // between supply and grain demand.
    float reserve_available = 0.f;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        const float floor_pool = grass_params.culm_reserve_floor * carbohydrate_params.stem_density * shoot->calculateShootInternodeVolume() / C_molecular_wt;
        reserve_available += std::max(0.f, shoot->sugar_pool_molC + shoot->starch_pool_molC - floor_pool);
    }
    const float root_dry_mass_g = plant_instance.root_structural_carbon_molC * C_molecular_wt / carbohydrate_params.stem_structural_carbon_percentage;
    const float root_floor_pool = grass_params.culm_reserve_floor * root_dry_mass_g / C_molecular_wt;
    reserve_available += std::max(0.f, plant_instance.root_sugar_pool_molC + plant_instance.root_starch_pool_molC - root_floor_pool);

    // The pools already hold the sub-step's photosynthate, so what they hold above the threshold bounds the grain growth; the split between photosynthate and reserve is the accounting of it.
    const double photosynthate_after_panicle = std::max(0.0, photosynthate_after_vegetative - spent_on_panicle);
    const double realized_with_respiration = std::min(grain_demand, double(reserve_available));
    const double from_photosynthate = std::min(realized_with_respiration, photosynthate_after_panicle);
    const double from_reserve = realized_with_respiration - from_photosynthate;
    const float realized_fraction = float(realized_with_respiration / grain_demand);

    for (Culm &culm: culms) {
        if (culm.grain_potential <= 0.f) {
            continue;
        }
        GrainSink &grain = culm.fbud->grain;
        const float growth = culm.grain_potential * realized_fraction;
        const float growth_respiration = growth * grass_params.grain_growth_respiration_fraction;
        culm.shoot->sugar_pool_molC -= growth + growth_respiration;
        culm.shoot->season_growth_molC += growth + growth_respiration;
        culm.shoot->season_fruit_cost_molC += growth + growth_respiration;
        grain.grain_carbon_molC += growth;
        grain.grain_potential_carbon_molC += culm.grain_potential;
        budget.grain_growth += growth;
        budget.growth_respiration += growth_respiration;
    }
    budget.reserve_remobilization += float(from_reserve);
}

// -------------------- Tillers -------------------- //

void PlantArchitecture::updateGrassTillers(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    const CarbonGrassParameters &grass_params = carbohydrate_params.grass;

    Shoot *main_stem = mainStem(plant_instance);
    if (main_stem == nullptr) {
        return;
    }

    // The onset of stem elongation: from here the elongating stem is the dominant sink. Every shoot's carbon records restart, so that a tiller is judged on what its leaves fixed and what
    // it spent from this point.
    if (!plant_instance.grass_stem_elongation_started && int(main_stem->current_node_number) >= grass_params.stem_elongation_node) {
        plant_instance.grass_stem_elongation_started = true;
        plant_instance.grass_stem_elongation_age = plant_instance.current_age;
        for (const auto &shoot: plant_instance.shoot_tree) {
            shoot->season_net_photosynthesis_molC = 0.f;
            shoot->season_maintenance_respiration_molC = 0.f;
            shoot->season_growth_molC = 0.f;
            shoot->season_fruit_cost_molC = 0.f;
        }
    }
    if (!plant_instance.grass_stem_elongation_started || !(grass_params.tiller_survival_carbon_ratio > 0.f)) {
        return;
    }
    // Tiller death runs from the onset of stem elongation to flowering of the main stem
    const FloralBud *main_panicle = culmPanicle(*main_stem);
    if (main_panicle != nullptr && main_panicle->time_counter >= grass_params.anthesis_days) {
        return;
    }
    if (plant_instance.current_age - plant_instance.grass_stem_elongation_age < carbohydrate_params.supply_averaging_days) {
        return;
    }

    // A tiller that has not yet reached its own flag leaf is shed when its leaves have not paid for its upkeep and growth since elongation began
    std::vector<uint> shed_list;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned() || shoot->parent_shoot_ID < 0 || shoot->phytomers.empty() || culmPanicle(*shoot) != nullptr) {
            continue;
        }
        const float spending = shoot->season_maintenance_respiration_molC + shoot->season_growth_molC;
        if (shoot->season_net_photosynthesis_molC < grass_params.tiller_survival_carbon_ratio * spending) {
            shed_list.push_back(shoot->ID);
        }
    }
    if (shed_list.empty()) {
        return;
    }
    const float carbohydrate_before = getPlantTotalCarbohydratePool(plantID);
    for (uint shootID: shed_list) {
        if (!plant_instance.shoot_tree.at(shootID)->phytomers.empty()) {
            pruneBranch(plantID, shootID, 0);
        }
    }
    plant_instance.carbon_budget.shed_branch_carbohydrate += carbohydrate_before - getPlantTotalCarbohydratePool(plantID);
    plant_instance.carbon_budget.tillers_shed += float(shed_list.size());
}

// -------------------- Queries -------------------- //

PlantGrainStatus PlantArchitecture::getPlantGrainStatus(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantGrainStatus): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    if (plant_instance.carb_parameters.submodel != CARBON_SUBMODEL_GRASS) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantGrainStatus): Plant with ID of " + std::to_string(plantID) + " does not run CARBON_SUBMODEL_GRASS, so it has no grain sink.");
    }

    PlantGrainStatus status;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        status.culm_carbohydrate_molC += shoot->sugar_pool_molC + shoot->starch_pool_molC;
        const FloralBud *fbud = culmPanicle(*shoot);
        if (fbud == nullptr) {
            continue;
        }
        status.culms += 1.f;
        status.grain_number += fbud->grain.grain_number;
        status.grain_carbon_molC += fbud->grain.grain_carbon_molC;
        status.grain_potential_carbon_molC += fbud->grain.grain_potential_carbon_molC;
        status.panicle_carbon_molC += fbud->grain.panicle_carbon_molC;
    }
    return status;
}
