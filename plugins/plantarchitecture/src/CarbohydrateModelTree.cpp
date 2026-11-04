/** \file "CarbohydrateModelTree.cpp" The dicot-tree sub-model (CARBON_SUBMODEL_TREE) of the carbohydrate model: a volumetric fruit sink set and then dropped for lack of carbon, a cambium
    that competes with storage for surplus photosynthate, and branches judged on their own carbon balance at the end of each season.

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

    //! Volume (m^3) of a fruit object, whatever primitive type the fruit prototype was built from.
    float fruitObjectVolume(const Context *context_ptr, uint objID) {
        switch (context_ptr->getObjectType(objID)) {
            case OBJECT_TYPE_POLYMESH:
                return context_ptr->getPolymeshObjectVolume(objID);
            case OBJECT_TYPE_SPHERE:
                return context_ptr->getSphereObjectVolume(objID);
            case OBJECT_TYPE_TUBE:
                return context_ptr->getTubeObjectVolume(objID);
            case OBJECT_TYPE_BOX:
                return context_ptr->getBoxObjectVolume(objID);
            case OBJECT_TYPE_CONE:
                return context_ptr->getConeObjectVolume(objID);
            default:
                helios_runtime_error("ERROR (PlantArchitecture carbohydrate model): Fruit object " + std::to_string(objID) +
                                     " is not a polymesh, sphere, tube, box or cone object, so its volume -- and therefore its carbon cost -- cannot be computed. Build the fruit prototype from one of these object types.");
        }
        return 0.f;
    }

    //! Fraction of a fruiting bud's development period (fruit set to maturity) that has elapsed. A plant whose fruit maturity threshold is disabled (not positive) never develops its fruit.
    float fruitDevelopmentFraction(const FloralBud &fbud, const PlantInstance &plant_instance) {
        const float dd_to_maturity = fbud.dd_to_inflorescence_maturity > 0 ? fbud.dd_to_inflorescence_maturity : plant_instance.dd_to_fruit_maturity;
        if (dd_to_maturity <= 0.f) {
            return 0.f;
        }
        return std::clamp(fbud.time_counter / dd_to_maturity, 0.f, 1.f);
    }

    //! Fraction of a mature fruit's dry matter in place after a given fraction of its development period: a logistic curve rescaled to run from 0 at fruit set to 1 at maturity.
    float fruitDryMatterFraction(float development_fraction, const CarbonFruitParameters &params) {
        if (!(params.dry_matter_width > 0.f) || !(params.dry_matter_midpoint > 0.f && params.dry_matter_midpoint < 1.f)) {
            helios_runtime_error("ERROR (PlantArchitecture carbohydrate model): CarbonFruitParameters::dry_matter_midpoint must lie strictly between 0 and 1 and dry_matter_width must be positive (got " +
                                 std::to_string(params.dry_matter_midpoint) + " and " + std::to_string(params.dry_matter_width) + ").");
        }
        if (!(params.late_fraction >= 0.f && params.late_fraction <= 1.f) || !(params.late_width > 0.f) || !(params.late_midpoint > 0.f && params.late_midpoint < 1.f)) {
            helios_runtime_error("ERROR (PlantArchitecture carbohydrate model): CarbonFruitParameters::late_fraction must be between 0 and 1, late_midpoint strictly between 0 and 1, and late_width positive (got " +
                                 std::to_string(params.late_fraction) + ", " + std::to_string(params.late_midpoint) + " and " + std::to_string(params.late_width) + ").");
        }
        const float x = std::clamp(development_fraction, 0.f, 1.f);
        // A logistic curve rescaled to run from 0 at fruit set to 1 at maturity
        auto rescaledLogistic = [x](float midpoint, float width) {
            auto logistic = [&](float position) { return 1.f / (1.f + std::exp(-(position - midpoint) / width)); };
            return (logistic(x) - logistic(0.f)) / (logistic(1.f) - logistic(0.f));
        };
        const float early = rescaledLogistic(params.dry_matter_midpoint, params.dry_matter_width);
        if (params.late_fraction == 0.f) {
            return early;
        }
        return (1.f - params.late_fraction) * early + params.late_fraction * rescaledLogistic(params.late_midpoint, params.late_width);
    }

} // namespace

// -------------------- Fruit carbon -------------------- //

float Phytomer::calculateMatureFruitCarbon(const FloralBud &fbud) const {
    if (fbud.current_fruit_scale_factor <= 0.f) {
        return 0.f;
    }
    const CarbonFruitParameters &fruit_params = plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters.fruit;

    // The fruit geometry is scaled uniformly, so its volume goes as the cube of the scale factor.
    float current_volume = 0.f;
    for (uint fruit_objID: fbud.inflorescence_objIDs) {
        if (context_ptr->doesObjectExist(fruit_objID)) {
            current_volume += fruitObjectVolume(context_ptr, fruit_objID);
        }
    }
    const float mature_volume = current_volume / powi(fbud.current_fruit_scale_factor, 3);

    const float fruit_carbon_density = fruit_params.density * fruit_params.carbon_percentage / C_molecular_wt; // mol C / m^3
    return fruit_carbon_density * mature_volume;
}

float Phytomer::calculateFruitDryMatterFraction(const FloralBud &fbud) const {
    if (fbud.state != BUD_FRUITING) {
        return 0.f;
    }
    const PlantInstance &plant_instance = plantarchitecture_ptr->plant_instances.at(this->plantID);
    return fruitDryMatterFraction(fruitDevelopmentFraction(fbud, plant_instance), plant_instance.carb_parameters.fruit);
}

void PlantArchitecture::subtractFruitGrowthCarbon(uint plantID, float dt, double reserve_funded_spending_molC) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    const CarbonFruitParameters &fruit_params = carbohydrate_params.fruit;
    PlantCarbonBudget &budget = plant_instance.carbon_budget;

    if (!(fruit_params.reserve_funded_days >= 0.f) || !(carbohydrate_params.supply_averaging_days > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::subtractFruitGrowthCarbon): CarbonFruitParameters::reserve_funded_days must not be negative and CarbohydrateParameters::supply_averaging_days must be positive (got " +
                             std::to_string(fruit_params.reserve_funded_days) + " and " + std::to_string(carbohydrate_params.supply_averaging_days) + ").");
    }

    // Fruit past its reserve-funded period is built from current photosynthate: its potential growth is collected here and
    // charged below, once the photosynthate left for it is known.
    struct PhotosynthateFruit {
        Shoot *shoot;
        FloralBud *fbud;
        float potential_carbon; // structural carbon of the fruit's potential growth this sub-step (mol C)
        float potential_fraction;
    };
    std::vector<PhotosynthateFruit> photosynthate_fruit;
    double reserve_funded_spending = reserve_funded_spending_molC; // vegetative and reserve-funded fruit growth, with roots and growth respiration (mol C)

    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue; // a pruned shoot has no phytomers left to grow
        }

        // Fruit is only ever charged for growth: an aborted or dropped fruit is not refunded.
        float fruit_growth = 0.f;
        for (const auto &phytomer: shoot->phytomers) {
            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    // Fruit carbon follows the fruit's dry-matter curve, which is laid down largely while the fruit is young, rather than its volume
                    if (fbud.state != BUD_FRUITING) {
                        continue;
                    }
                    const float dry_matter_fraction = phytomer->calculateFruitDryMatterFraction(fbud);
                    if (dry_matter_fraction > fbud.fruit_potential_fraction) {
                        const float potential_increment = dry_matter_fraction - fbud.fruit_potential_fraction;
                        const float potential_carbon = phytomer->calculateMatureFruitCarbon(fbud) * potential_increment;
                        if (fbud.time_counter <= fruit_params.reserve_funded_days) {
                            fruit_growth += potential_carbon;
                            fbud.fruit_carbon_fraction_charged += potential_increment;
                            fbud.fruit_potential_fraction = dry_matter_fraction;
                        } else {
                            photosynthate_fruit.push_back({shoot.get(), &fbud, potential_carbon, dry_matter_fraction});
                        }
                    }
                }
            }
        }

        if (fruit_growth <= 0.f) {
            continue;
        }
        const float growth_respiration = fruit_params.growth_respiration_fraction * fruit_growth;
        shoot->sugar_pool_molC -= fruit_growth + growth_respiration;
        shoot->season_growth_molC += fruit_growth + growth_respiration;
        shoot->season_fruit_cost_molC += fruit_growth + growth_respiration;
        budget.fruit_growth += fruit_growth;
        budget.growth_respiration += growth_respiration;
        reserve_funded_spending += fruit_growth + growth_respiration;

        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            // The shoot's vegetative growth was reported by subtractShootGrowthCarbon(); its fruit growth is added to the same totals
            float daily_growth = 0.f;
            float daily_growth_respiration = 0.f;
            if (context_ptr->doesObjectDataExist(shoot->internode_tube_objID, "daily_growth")) {
                context_ptr->getObjectData(shoot->internode_tube_objID, "daily_growth", daily_growth);
            }
            if (context_ptr->doesObjectDataExist(shoot->internode_tube_objID, "daily_growth_respiration")) {
                context_ptr->getObjectData(shoot->internode_tube_objID, "daily_growth_respiration", daily_growth_respiration);
            }
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_growth", daily_growth + fruit_growth);
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_growth_respiration", daily_growth_respiration + growth_respiration);
        }
    }

    // Photosynthate left for the fruit past its reserve-funded period after maintenance respiration, vegetative growth and
    // reserve-funded fruit, against that fruit's potential growth (both with growth respiration), averaged over the supply-averaging period
    double potential_demand = 0.0;
    for (const auto &fruit: photosynthate_fruit) {
        potential_demand += fruit.potential_carbon * (1.f + fruit_params.growth_respiration_fraction);
    }
    const double photosynthate = plant_instance.photosynthesis_supply_rate_molC_day * dt;
    // Girth growth, paid from photosynthate, ranks below fruit (allocateSurplusCarbon()), so it does not reduce the fruit's supply
    reserve_funded_spending -= plant_instance.step_girth_growth_molC * (1.f + carbohydrate_params.growth_respiration_fraction) * (1.f + 1.f / carbohydrate_params.shoot_root_ratio);
    const double fruit_supply = std::max(0.0, photosynthate - plant_instance.step_maintenance_respiration_molC - reserve_funded_spending);
    const float supply_ratio = recordReproductiveSupply(plantID, float(fruit_supply), float(potential_demand), dt);

    // Each fruit grows at its potential times the supply ratio; the potential it reached is recorded either way, so growth lost now is not made up later
    for (const auto &fruit: photosynthate_fruit) {
        const float fruit_growth = fruit.potential_carbon * supply_ratio;
        const float fruit_growth_respiration = fruit_growth * fruit_params.growth_respiration_fraction;
        fruit.shoot->sugar_pool_molC -= fruit_growth + fruit_growth_respiration;
        fruit.shoot->season_growth_molC += fruit_growth + fruit_growth_respiration;
        fruit.shoot->season_fruit_cost_molC += fruit_growth + fruit_growth_respiration;
        budget.fruit_growth += fruit_growth;
        budget.growth_respiration += fruit_growth_respiration;
        fruit.fbud->fruit_carbon_fraction_charged += supply_ratio * (fruit.potential_fraction - fruit.fbud->fruit_potential_fraction);
        fruit.fbud->fruit_potential_fraction = fruit.potential_fraction;
    }
}

// -------------------- Fruit drop -------------------- //

void PlantArchitecture::checkCarbonPool_dropFruit(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    const CarbonFruitParameters &fruit_params = carbohydrate_params.fruit;
    if (!(fruit_params.drop_window_end > fruit_params.reserve_funded_days)) {
        helios_runtime_error("ERROR (PlantArchitecture::checkCarbonPool_dropFruit): CarbonFruitParameters::drop_window_end must exceed reserve_funded_days (got " + std::to_string(fruit_params.drop_window_end) + " and " +
                             std::to_string(fruit_params.reserve_funded_days) + " days).");
    }
    if (!(carbohydrate_params.supply_averaging_days > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::checkCarbonPool_dropFruit): CarbohydrateParameters::supply_averaging_days must be positive (got " + std::to_string(carbohydrate_params.supply_averaging_days) + ").");
    }

    // Fruit past its reserve-funded period that can still drop
    struct YoungFruit {
        float days_since_set;
        Phytomer *phytomer;
        FloralBud *fbud;
    };
    std::vector<YoungFruit> young_fruit;
    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        for (const auto &phytomer: shoot->phytomers) {
            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    if (fbud.state == BUD_FRUITING && fbud.time_counter > fruit_params.reserve_funded_days && fbud.time_counter <= fruit_params.drop_window_end) {
                        young_fruit.push_back({fbud.time_counter, phytomer.get(), &fbud});
                    }
                }
            }
        }
    }

    // Fruit built from current photosynthate drops when it cannot grow at the rate it needs: a supply ratio below
    // abscission_growth_ratio drops the fraction of that fruit that would restore the ratio over the averaging period.
    if (!(fruit_params.abscission_growth_ratio > 0.f && fruit_params.abscission_growth_ratio <= 1.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::checkCarbonPool_dropFruit): CarbonFruitParameters::abscission_growth_ratio must be greater than 0 and at most 1 (got " + std::to_string(fruit_params.abscission_growth_ratio) +
                             ").");
    }
    const float supply_ratio = plant_instance.reproductive_supply_ratio;
    if (young_fruit.empty() || supply_ratio >= fruit_params.abscission_growth_ratio) {
        plant_instance.fruit_drop_debt = 0.f;
        return;
    }
    const float drop_fraction = std::min(1.f, (1.f - supply_ratio / fruit_params.abscission_growth_ratio) * dt / carbohydrate_params.supply_averaging_days);
    plant_instance.fruit_drop_debt += drop_fraction * float(young_fruit.size());
    const auto drop_count = std::min(size_t(plant_instance.fruit_drop_debt), young_fruit.size());
    plant_instance.fruit_drop_debt -= float(drop_count);
    std::sort(young_fruit.begin(), young_fruit.end(), [](const YoungFruit &a, const YoungFruit &b) { return a.days_since_set < b.days_since_set; });
    for (size_t i = 0; i < drop_count; i++) {
        young_fruit.at(i).phytomer->setFloralBudState(BUD_DEAD, *young_fruit.at(i).fbud);
    }
    plant_instance.carbon_budget.fruit_dropped += float(drop_count);
}

// -------------------- Branch mortality -------------------- //

void PlantArchitecture::shedCarbonStarvedBranches(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::shedCarbonStarvedBranches): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    PlantInstance &plant_instance = plant_instances.at(plantID);
    auto &shoot_tree = plant_instance.shoot_tree;
    const float carbon_ratio = plant_instance.carb_parameters.wood.branch_shedding_carbon_ratio;

    // Branch totals are built from the tips inward. A child shoot is always created after its parent and so has the larger
    // ID; visiting shoots in decreasing ID order therefore completes every branch before its parent is judged.
    std::vector<double> branch_photosynthesis(shoot_tree.size(), 0.0);
    std::vector<double> branch_spending(shoot_tree.size(), 0.0); // maintenance respiration plus construction
    std::vector<uint> shed_list;
    for (int shootID = int(shoot_tree.size()) - 1; shootID >= 0; shootID--) {
        const auto &shoot = shoot_tree.at(shootID);
        if (shoot->isPruned()) {
            continue;
        }
        if (shoot->parent_shoot_ID >= shootID) {
            helios_runtime_error("ERROR (PlantArchitecture::shedCarbonStarvedBranches): shoot " + std::to_string(shootID) + " has a parent with a larger ID (" + std::to_string(shoot->parent_shoot_ID) +
                                 "), so branches cannot be totalled from the tips inward.");
        }
        const ShootParameters &shoot_parameters = shoot->shoot_parameters;
        if (shoot_parameters.fruit_cost_share < 0.f || shoot_parameters.fruit_cost_share > 1.f) {
            helios_runtime_error("ERROR (PlantArchitecture::shedCarbonStarvedBranches): ShootParameters::fruit_cost_share of shoot type \"" + shoot->shoot_type_label + "\" must be between 0 and 1 (got " +
                                 std::to_string(shoot_parameters.fruit_cost_share) + ").");
        }

        // Only fruit_cost_share of the shoot's fruit is charged to the shoot; the bearing branch pays the rest, whether or not the shoot survives
        const double fruit_cost_passed_on = (1.0 - shoot_parameters.fruit_cost_share) * shoot->season_fruit_cost_molC;
        branch_photosynthesis.at(shootID) += shoot->season_net_photosynthesis_molC;
        branch_spending.at(shootID) += shoot->season_maintenance_respiration_molC + shoot->season_growth_molC - fruit_cost_passed_on;
        if (shoot->parent_shoot_ID >= 0) {
            branch_spending.at(shoot->parent_shoot_ID) += fruit_cost_passed_on;
        }

        // Carbon index of the branch, and of its vegetative growth alone
        auto logRatio = [](double income, double spending) {
            constexpr double smallest_ratio = 1e-6;
            if (spending <= 0.0) {
                return std::log(1.0 / smallest_ratio);
            }
            return std::log(std::max(income / spending, smallest_ratio));
        };
        const float carbon_index = float(logRatio(branch_photosynthesis.at(shootID), branch_spending.at(shootID)));
        const float vegetative_spending = shoot->season_maintenance_respiration_molC + shoot->season_growth_molC - shoot->season_fruit_cost_molC;
        const float vegetative_index = float(logRatio(shoot->season_net_photosynthesis_molC, vegetative_spending));

        // A shoot is judged only once it has been through a full season in leaf (its first dormancy boundary comes before
        // dormancy_cycles is incremented).
        const bool eligible = shoot->parent_shoot_ID >= 0 && shoot->dormancy_cycles >= 1 && !isStructuralShoot(plantID, uint(shootID));
        bool shed;
        if (shoot_parameters.carbon_survival_scale > 0.f) {
            const float survival_probability = 1.f / (1.f + std::exp(-(carbon_index - shoot_parameters.carbon_survival_midpoint) / shoot_parameters.carbon_survival_scale));
            shed = eligible && context_ptr->randu() >= survival_probability;
        } else {
            shed = eligible && branch_photosynthesis.at(shootID) < carbon_ratio * branch_spending.at(shootID);
        }
        if (shed) {
            shed_list.push_back(uint(shootID));
            continue; // a shed branch no longer counts toward its parent's balance
        }

        // A surviving shoot's chance of flowering in the next season follows its carbon
        if (shoot_parameters.return_bloom_scale > 0.f) {
            float flowering = shoot_parameters.return_bloom_max / (1.f + std::exp(-(vegetative_index - shoot_parameters.return_bloom_midpoint) / shoot_parameters.return_bloom_scale));
            if (shoot->season_fruit_cost_molC > 0.f) {
                flowering *= shoot_parameters.return_bloom_bearing_factor;
            }
            shoot->flowering_probability = std::clamp(flowering, 0.f, 1.f);
        }

        if (shoot->parent_shoot_ID >= 0) {
            branch_photosynthesis.at(shoot->parent_shoot_ID) += branch_photosynthesis.at(shootID);
            branch_spending.at(shoot->parent_shoot_ID) += branch_spending.at(shootID);
        }
    }

    // shed_list runs from the largest ID down, so a shed child is removed before a shed ancestor; a shoot already removed
    // with an ancestor's subtree has no phytomers left and is skipped.
    const float carbohydrate_before = getPlantTotalCarbohydratePool(plantID);
    for (uint shootID: shed_list) {
        if (!shoot_tree.at(shootID)->phytomers.empty()) {
            pruneBranch(plantID, shootID, 0);
        }
    }
    plant_instance.carbon_budget.shed_branch_carbohydrate += carbohydrate_before - getPlantTotalCarbohydratePool(plantID);

    for (const auto &shoot: shoot_tree) {
        shoot->season_net_photosynthesis_molC = 0.f;
        shoot->season_maintenance_respiration_molC = 0.f;
        shoot->season_growth_molC = 0.f;
        shoot->season_fruit_cost_molC = 0.f;
    }
}

// -------------------- Radial growth -------------------- //

void PlantArchitecture::incrementPhytomerInternodeGirth_carb(uint plantID, uint shootID, uint node_number, float dt, bool update_context_geometry, float shoot_length, float shoot_volume) {
    // Slow Radial growth of shoots if their carbon concentration falls below a sustainable threshold
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }

    const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;
    const CarbonWoodParameters &wood_params = carbohydrate_params.wood;
    auto shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if (node_number >= shoot->current_node_number) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Cannot scale internode " + std::to_string(node_number) + " because there are only " + std::to_string(shoot->current_node_number) + " nodes in this shoot.");
    }

    auto phytomer = shoot->phytomers.at(node_number);

    // float leaf_area = phytomer->calculateDownstreamLeafArea();
    float leaf_area = phytomer->downstream_leaf_area;
    if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        context_ptr->setObjectData(shoot->internode_tube_objID, "leaf_area", leaf_area);
    }

    float phytomer_age = phytomer->age;
    float girth_area_factor = shoot->shoot_parameters.girth_area_factor.val();
    if (phytomer_age > 365) {
        girth_area_factor = shoot->shoot_parameters.girth_area_factor.val() * 365 / phytomer_age;
    }


    float internode_area = girth_area_factor * leaf_area * 1e-4;

    // A terminal inflorescence is borne above every node on the shoot but contributes no leaf area, so the pipe model alone tapers the stem below it too far -- see the matching adjustment in
    // PlantArchitecture::incrementPhytomerInternodeGirth(). Only the rendered radius is widened here: internode_area continues to feed max_shoot_volume below, which sets the shoot's structural carbon
    // demand, and that demand should stay tied to the leaf area actually supplying it.
    const float supported_area = leaf_area + shoot->sumDownstreamInflorescenceArea(node_number);
    float phytomer_radius = sqrtf(girth_area_factor * supported_area * 1e-4 / PI_F);

    float rho_cw = carbohydrate_params.stem_density * carbohydrate_params.stem_structural_carbon_percentage / C_molecular_wt; // Density of carbon in wood (mol C m^-3)
    float max_shoot_volume = internode_area * shoot_length;
    float current_shoot_volume = shoot_volume;
    float max_carbon_demand = (max_shoot_volume - current_shoot_volume) * rho_cw; //(mol C)

    float threshold_carbon_pool = carbohydrate_params.carbohydrate_growth_threshold * current_shoot_volume * carbohydrate_params.stem_density / C_molecular_wt; //(mol C)
    const float carbohydrate_pool = shoot->sugar_pool_molC + shoot->starch_pool_molC;

    // Radial growth draws only on carbohydrate above the growth threshold, in proportion to how much of the girth
    // target's carbon demand that excess covers. Below the threshold the cambium does not grow; the girth target
    // persists, so the wood catches up once the pool recovers.
    float carbon_availability_ratio;
    if (carbohydrate_pool <= threshold_carbon_pool) {
        carbon_availability_ratio = 0.f;
    } else if (max_carbon_demand <= 0.f) {
        carbon_availability_ratio = 1.f;
    } else {
        carbon_availability_ratio = std::clamp((carbohydrate_pool - threshold_carbon_pool) / max_carbon_demand, 0.f, 1.f);
    }

    // The gap closes at radial_growth_rate, limited further by the photosynthate allotted to the cambium (cambium_supply_ratio);
    // the potential growth is recorded for allocateSurplusCarbon() to bid for.
    PlantInstance &plant_instance = plant_instances.at(plantID);
    if (!(wood_params.radial_growth_rate >= 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth_carb): CarbonWoodParameters::radial_growth_rate must not be negative (got " + std::to_string(wood_params.radial_growth_rate) + " per day).");
    }
    const float closing_fraction = 1.f - std::exp(-wood_params.radial_growth_rate * dt);
    const float supply_ratio = plant_instance.cambium_supply_ratio;
    const float structural_carbon_before = phytomer->calculateInternodeStructuralCarbon(node_number);

    auto &segment = shoot->shoot_internode_radii.at(node_number);
    float mean_radius_before = 0.f;
    for (float radius: segment) {
        mean_radius_before += radius;
    }
    mean_radius_before = segment.empty() ? 0.f : mean_radius_before / float(segment.size());

    // Sapwood ring: each season the internode owes a ring of area sapwood_area_per_leaf_area x (peak live leaf area above it) /
    // sapwood_functional_years, replacing the sapwood that turns to heartwood. The part not yet laid down closes at the same rate as
    // the pipe-model gap, added as a uniform area to every radius of the internode so that its taper is kept.
    float ring_area_remaining = 0.f;
    if (wood_params.sapwood_area_per_leaf_area > 0.f) {
        if (phytomer->ring_season_index != plant_instance.cambium_season_index) {
            phytomer->ring_season_index = plant_instance.cambium_season_index;
            phytomer->ring_season_start_area = PI_F * mean_radius_before * mean_radius_before;
            phytomer->ring_season_peak_leaf_area = 0.f;
        }
        phytomer->ring_season_peak_leaf_area = std::max(phytomer->ring_season_peak_leaf_area, phytomer->live_downstream_leaf_area);
        const float ring_area_owed = wood_params.sapwood_area_per_leaf_area * phytomer->ring_season_peak_leaf_area / wood_params.sapwood_functional_years;
        const float ring_area_laid = PI_F * mean_radius_before * mean_radius_before - phytomer->ring_season_start_area;
        ring_area_remaining = std::max(0.f, ring_area_owed - ring_area_laid);
    }
    // New radius for a fraction of this sub-step's potential growth: the larger of the pipe-model and sapwood-ring increments
    auto grownRadius = [&](float radius, float fraction) {
        float pipe_radius = radius;
        if (phytomer_radius > radius) { // radius should only increase
            pipe_radius = radius + fraction * closing_fraction * (phytomer_radius - radius);
        }
        const float ring_radius = std::sqrt(radius * radius + fraction * closing_fraction * ring_area_remaining / PI_F);
        return std::max(pipe_radius, ring_radius);
    };

    if (mean_radius_before > 0.f) {
        // Structural carbon goes as the square of the mean radius (Phytomer::calculateInternodeStructuralCarbon())
        float mean_radius_potential = 0.f;
        for (float radius: segment) {
            mean_radius_potential += grownRadius(radius, carbon_availability_ratio);
        }
        const float growth_ratio = mean_radius_potential / float(segment.size()) / mean_radius_before;
        plant_instance.step_girth_potential_molC += structural_carbon_before * (growth_ratio * growth_ratio - 1.f);
    }

    for (float &radius: segment) {
        radius = grownRadius(radius, carbon_availability_ratio * supply_ratio);

        if (update_context_geometry && context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            context_ptr->setTubeRadii(shoot->internode_tube_objID, flatten(shoot->shoot_internode_radii));
        }
    }

    plant_instance.step_girth_growth_molC += phytomer->calculateInternodeStructuralCarbon(node_number) - structural_carbon_before;
}
