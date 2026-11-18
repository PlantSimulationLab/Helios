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

    //! The fruiting terminal bud of a culm -- its panicle, which appears with the flag leaf -- or nullptr while the culm has not reached its flag leaf
    FloralBud *culmPanicle(const Shoot &shoot) {
        if (shoot.phytomers.empty()) {
            return nullptr;
        }
        // Shoot::addTerminalFloralBud() places the terminal bud on the last phytomer, when the shoot reaches its node cap
        for (auto &petiole: shoot.phytomers.back()->floral_buds) {
            for (auto &fbud: petiole) {
                if (fbud.isterminal && fbud.state == BUD_FRUITING) { // PlantArchitecture::isGrassPanicle()
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

bool PlantArchitecture::isGrassPanicle(const FloralBud &fbud) {
    return fbud.isterminal && fbud.state == BUD_FRUITING;
}

float PlantArchitecture::grassAnthesisAge(const PlantInstance &plant_instance) {
    return plant_instance.dd_to_fruit_set;
}

float PlantArchitecture::grassMaturityAge(const PlantInstance &plant_instance) {
    return plant_instance.dd_to_fruit_maturity;
}

float PlantArchitecture::grassCulmPanicleAge(const Shoot &shoot) {
    const FloralBud *fbud = culmPanicle(shoot);
    return fbud == nullptr ? -1.f : fbud->time_counter;
}

bool PlantArchitecture::isSenescedGrassCulm(const PlantInstance &plant_instance, const Shoot &shoot) {
    if (plant_instance.carb_parameters.submodel != CARBON_SUBMODEL_GRASS) {
        return false;
    }
    const float panicle_age = grassCulmPanicleAge(shoot);
    return panicle_age >= 0.f && panicle_age >= grassMaturityAge(plant_instance) + plant_instance.carb_parameters.grass.post_maturity_senescence_days;
}

float PlantArchitecture::grassMainStemDevelopmentFraction(uint plantID) const {
    Shoot *main_stem = mainStem(plant_instances.at(plantID));
    if (main_stem == nullptr) {
        // The main stem has died (pruned for starvation): the plant is past any vegetative development
        return 1.f;
    }
    const int node_cap = main_stem->getGrowthNodeCap();
    if (node_cap <= 0) {
        helios_runtime_error("ERROR (PlantArchitecture::grassMainStemDevelopmentFraction): the main stem of plant " + std::to_string(plantID) + " has a node cap of " + std::to_string(node_cap) +
                             ". The grass sub-model measures the main stem's development by its node count over its cap, so the main stem's shoot type needs a positive max_nodes.");
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

void PlantArchitecture::subtractGrainGrowthCarbon(uint plantID, float dt, double reserve_funded_spending_molC, double stem_spending_molC) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    const CarbonGrassParameters &grass_params = carbohydrate_params.grass;
    PlantCarbonBudget &budget = plant_instance.carbon_budget;

    // The panicle's stages are the plant's phenological thresholds, counted from the flag leaf: a culm's terminal bud bears its panicle from the moment the culm reaches its node cap
    // (PlantArchitecture::appendPhytomerToShoot()), so the time to fruit set is the panicle's age at anthesis and the time to fruit maturity its age at physiological maturity.
    const float critical_period_days = grassAnthesisAge(plant_instance) + grass_params.grain_fill_lag_days;
    const float fill_end_days = grassMaturityAge(plant_instance);
    if (!(plant_instance.dd_to_fruit_set >= 0.f) || !(fill_end_days > critical_period_days)) {
        helios_runtime_error("ERROR (PlantArchitecture::subtractGrainGrowthCarbon): plant " + std::to_string(plantID) + " (" + plant_instance.plant_name +
                             ") times its grain from its phenological thresholds, counted from the flag leaf: the time to fruit set is anthesis and the time to fruit maturity physiological maturity, which "
                             "must come more than CarbonGrassParameters::grain_fill_lag_days after anthesis (got fruit set " +
                             std::to_string(plant_instance.dd_to_fruit_set) + ", fruit maturity " + std::to_string(plant_instance.dd_to_fruit_maturity) + " and lag " + std::to_string(grass_params.grain_fill_lag_days) +
                             " days). Set them with setPlantPhenologicalThresholds().");
    }
    const float grain_fill_days = fill_end_days - critical_period_days;

    // Each culm's panicle and what it could grow this sub-step. The panicle's clock is the developmental days since it appeared with the flag leaf, which has already been advanced for this
    // sub-step: by the plant's developmental days before anthesis and by the grain-filling ones after it. The potential growth is accounted from where the last sub-step left it, and never
    // for more than a sub-step of the faster clock, so that a panicle the model first meets part-grown is not charged for its past.
    const float largest_clock_step = dt * std::max(plant_instance.development_rate, plant_instance.grain_development_rate);
    struct Culm {
        Shoot *shoot;
        FloralBud *fbud;
        float age_start; // panicle age at which this sub-step's accounting begins (developmental days)
        float age_end; // panicle age now
        float panicle_potential; // structural carbon the panicle could lay down this sub-step (mol C)
        float grain_potential; // structural carbon the grain could lay down this sub-step (mol C)
    };
    std::vector<Culm> culms;
    double panicle_demand = 0.0; // with growth respiration (mol C)
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        // The grass sub-model's only reproductive sink is the terminal panicle of a culm. Fruit borne in the leaf axils (a bean's pods, a maize ear) would otherwise grow uncharged.
        for (const auto &phytomer: shoot->phytomers) {
            for (const auto &petiole: phytomer->floral_buds) {
                for (const auto &fbud: petiole) {
                    if (fbud.isterminal && (fbud.state == BUD_FLOWER_CLOSED || fbud.state == BUD_FLOWER_OPEN)) {
                        helios_runtime_error("ERROR (PlantArchitecture::subtractGrainGrowthCarbon): plant " + std::to_string(plantID) + " (" + plant_instance.plant_name +
                                             ") bears flowers at the tip of a culm. CARBON_SUBMODEL_GRASS times the panicle from the flag leaf, which needs the culm's terminal bud to bear its panicle "
                                             "(fruit) from the moment the culm reaches its node cap: give the shoot type a fruit prototype and no flower prototype.");
                    }
                    if (!fbud.isterminal && fbud.state != BUD_DORMANT && fbud.state != BUD_DEAD) {
                        helios_runtime_error("ERROR (PlantArchitecture::subtractGrainGrowthCarbon): plant " + std::to_string(plantID) + " (" + plant_instance.plant_name +
                                             ") bears flowers or fruit in its leaf axils, which the grass sub-model does not represent: CARBON_SUBMODEL_GRASS charges only the terminal panicle of each culm. Use "
                                             "CARBON_SUBMODEL_TREE for this plant.");
                    }
                }
            }
        }
        FloralBud *fbud = culmPanicle(*shoot);
        if (fbud == nullptr) {
            continue;
        }
        const float t_end = fbud->time_counter;
        const float t_start = std::clamp(fbud->grain.accounted_age, std::max(0.f, t_end - largest_clock_step), t_end);
        fbud->grain.accounted_age = t_end;
        float panicle_potential = 0.f;
        if (t_start < critical_period_days) {
            // Built at a constant potential rate over the critical period
            panicle_potential = grass_params.panicle_potential_carbon * (std::min(t_end, critical_period_days) - t_start) / critical_period_days;
        }
        culms.push_back({shoot.get(), fbud, t_start, t_end, panicle_potential, 0.f});
        panicle_demand += panicle_potential * (1.f + grass_params.panicle_growth_respiration_fraction);
    }

    // The panicle is built from current photosynthate, which it shares with the elongating stem: of the photosynthate left after maintenance respiration and the sub-step's leaf and root
    // growth, the panicles receive the share their potential growth makes of the stems' growth and their own, and grow at that over their potential, averaged over the supply-averaging
    // period. Panicle growth in the critical period is plant growth times the fraction partitioned to the panicle (van Oosterom & Hammer 2008), and crop models give the developing head a
    // fixed share of the stem's dry matter (0.3 in APSIM-Sorghum); a panicle served only what the stem left received nothing under a carbon shortage. Cereals are still laying down stem
    // reserves at this stage, and grain number follows current assimilate rather than reserves, so a shortfall is not made up from them. The stem's own growth is charged in full, from
    // reserves where photosynthate falls short.
    const double photosynthate = plant_instance.photosynthesis_supply_rate_molC_day * dt;
    const double photosynthate_after_vegetative = std::max(0.0, photosynthate - plant_instance.step_maintenance_respiration_molC - reserve_funded_spending_molC);
    const double photosynthate_for_stem_and_panicle = std::max(0.0, photosynthate - plant_instance.step_maintenance_respiration_molC - (reserve_funded_spending_molC - stem_spending_molC));
    const double panicle_share = panicle_demand > 0.0 ? panicle_demand / (stem_spending_molC + panicle_demand) : 0.0;
    const float supply_ratio = recordReproductiveSupply(plantID, float(photosynthate_for_stem_and_panicle * panicle_share), float(panicle_demand), dt);

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
        const float t_end = std::clamp(culm.age_end, critical_period_days, fill_end_days);
        const float t_start = std::clamp(culm.age_start, critical_period_days, fill_end_days);
        culm.grain_potential = grain.grain_number * grass_params.kernel_potential_carbon * (t_end - t_start) / grain_fill_days;
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

    // The pools already hold the sub-step's photosynthate, so what they hold above the floor bounds the grain growth, and the split between photosynthate and reserve is the accounting of it.
    // A culm already below the floor does not refill itself before its grain: the grain still receives the sub-step's photosynthate, so the pool neither recovers nor falls further.
    const double photosynthate_after_panicle = std::max(0.0, photosynthate_after_vegetative - spent_on_panicle);
    const double realized_with_respiration = std::min(grain_demand, std::max(double(reserve_available), photosynthate_after_panicle));
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

void PlantArchitecture::updateGrassTillers(uint plantID) {
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
    if (main_panicle != nullptr && main_panicle->time_counter >= grassAnthesisAge(plant_instance)) {
        return;
    }

    // A tiller that has not yet reached its own flag leaf is shed when its leaves have not paid for its upkeep and growth since elongation began, or since it became established if that
    // was later: a young tiller is carried by the rest of the plant until it has tiller_establishment_leaves leaves, and its records restart then.
    std::vector<uint> shed_list;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned() || shoot->parent_shoot_ID < 0 || shoot->phytomers.empty() || culmPanicle(*shoot) != nullptr) {
            continue;
        }
        if (shoot->tiller_established_age < 0.f) {
            if (int(shoot->current_node_number) < grass_params.tiller_establishment_leaves) {
                continue;
            }
            shoot->tiller_established_age = plant_instance.current_age;
            if (plant_instance.current_age > plant_instance.grass_stem_elongation_age) {
                shoot->season_net_photosynthesis_molC = 0.f;
                shoot->season_maintenance_respiration_molC = 0.f;
                shoot->season_growth_molC = 0.f;
                shoot->season_fruit_cost_molC = 0.f;
            }
        }
        if (plant_instance.current_age - std::max(plant_instance.grass_stem_elongation_age, shoot->tiller_established_age) < carbohydrate_params.supply_averaging_days) {
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

// -------------------- Stem as the residual sink -------------------- //

float PlantArchitecture::growGrassCulmsFromSurplus(uint plantID, float surplus_molC) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    const float thickening_max = carbohydrate_params.grass.culm_thickening_max;
    if (!(thickening_max > 1.f) || !(surplus_molC > 0.f)) {
        return 0.f;
    }

    // Culms that have not yet reached anthesis
    const float anthesis_age = grassAnthesisAge(plant_instance);
    std::vector<Shoot *> growing_culms;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned() || shoot->phytomers.empty()) {
            continue;
        }
        const FloralBud *fbud = culmPanicle(*shoot);
        if (fbud != nullptr && fbud->time_counter >= anthesis_age) {
            continue;
        }
        growing_culms.push_back(shoot.get());
    }

    // An internode's structural carbon goes as its radius squared, so the surplus thickens every internode that still has room by one factor, each held at its largest radius. Internodes that
    // reach theirs leave part of the surplus unplaced, which a further pass offers to the rest.
    const float growth_cost = 1.f + carbohydrate_params.growth_respiration_fraction;
    float spent_molC = 0.f;
    for (int pass = 0; pass < 4 && surplus_molC - spent_molC > 1e-6f * surplus_molC; pass++) {
        double carbon_with_room = 0.0;
        for (const Shoot *shoot: growing_culms) {
            for (uint p = 0; p < shoot->phytomers.size(); p++) {
                const float radius_max = thickening_max * pipeModelInternodeRadius(*shoot, p);
                const auto &radii = shoot->shoot_internode_radii.at(p);
                if (std::any_of(radii.begin(), radii.end(), [radius_max](float radius) { return radius < radius_max; })) {
                    carbon_with_room += shoot->phytomers.at(p)->calculateInternodeStructuralCarbon(p);
                }
            }
        }
        if (!(carbon_with_room > 0.0)) {
            break;
        }
        const float radius_factor = std::sqrt(1.f + float(double(surplus_molC - spent_molC) / growth_cost / carbon_with_room));
        for (Shoot *shoot: growing_culms) {
            float stem_growth = 0.f;
            for (uint p = 0; p < shoot->phytomers.size(); p++) {
                const auto &phytomer = shoot->phytomers.at(p);
                const float radius_max = thickening_max * pipeModelInternodeRadius(*shoot, p);
                const float carbon_before = phytomer->calculateInternodeStructuralCarbon(p);
                for (float &radius: shoot->shoot_internode_radii.at(p)) {
                    if (radius < radius_max) {
                        radius = std::min(radius * radius_factor, radius_max);
                    }
                }
                const float growth = std::max(0.f, phytomer->calculateInternodeStructuralCarbon(p) - carbon_before);
                stem_growth += growth;
                // The internode's construction is charged here, so that the next sub-step's stem growth does not charge it again
                phytomer->internode_carbon_charged_molC += growth;
            }
            // Paid from the surplus, so it does not count against the culm in the tiller survival test
            const float growth_respiration = stem_growth * carbohydrate_params.growth_respiration_fraction;
            shoot->sugar_pool_molC -= stem_growth + growth_respiration;
            plant_instance.carbon_budget.wood_growth += stem_growth;
            plant_instance.carbon_budget.growth_respiration += growth_respiration;
            spent_molC += stem_growth + growth_respiration;
        }
    }
    return std::min(spent_molC, surplus_molC);
}

// -------------------- Senescence -------------------- //

void PlantArchitecture::senesceMatureGrassCulms(uint plantID) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbonGrassParameters &grass_params = plant_instance.carb_parameters.grass;
    const float maturity_age = grassMaturityAge(plant_instance);

    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        FloralBud *fbud = culmPanicle(*shoot);
        if (fbud == nullptr || fbud->time_counter < maturity_age) {
            continue;
        }
        // Leaves still on the culm, oldest (lowest) first
        std::vector<Phytomer *> leafy_phytomers;
        for (const auto &phytomer: shoot->phytomers) {
            if (phytomer->hasLeaf()) {
                leafy_phytomers.push_back(phytomer.get());
            }
        }
        GrainSink &grain = fbud->grain;
        if (grain.leaves_at_maturity < 0) {
            grain.leaves_at_maturity = int(leafy_phytomers.size());
        }
        // The culm keeps the share of its leaves that the senescence period has not yet reached
        const float elapsed_fraction = grass_params.post_maturity_senescence_days > 0.f ? std::clamp((fbud->time_counter - maturity_age) / grass_params.post_maturity_senescence_days, 0.f, 1.f) : 1.f;
        const int leaves_kept = int(std::round(float(grain.leaves_at_maturity) * (1.f - elapsed_fraction)));
        for (size_t i = 0; i + leaves_kept < leafy_phytomers.size(); i++) {
            if (nitrogen_model_enabled) {
                resorbLeafNitrogen(plant_instance, *shoot, *leafy_phytomers.at(i));
            }
            leafy_phytomers.at(i)->removeLeaf();
        }
    }
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
