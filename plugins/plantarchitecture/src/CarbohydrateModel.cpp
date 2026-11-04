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

    //! Fraction of a fruiting bud's development period (fruit set to maturity) that has elapsed. A plant whose fruit maturity threshold is disabled (not positive) never develops its fruit.
    float fruitDevelopmentFraction(const FloralBud &fbud, const PlantInstance &plant_instance) {
        const float dd_to_maturity = fbud.dd_to_inflorescence_maturity > 0 ? fbud.dd_to_inflorescence_maturity : plant_instance.dd_to_fruit_maturity;
        if (dd_to_maturity <= 0.f) {
            return 0.f;
        }
        return std::clamp(fbud.time_counter / dd_to_maturity, 0.f, 1.f);
    }

    //! Fraction of a mature fruit's dry matter in place after a given fraction of its development period: a logistic curve rescaled to run from 0 at fruit set to 1 at maturity.
    float fruitDryMatterFraction(float development_fraction, const CarbohydrateParameters &params) {
        if (!(params.fruit_dry_matter_width > 0.f) || !(params.fruit_dry_matter_midpoint > 0.f && params.fruit_dry_matter_midpoint < 1.f)) {
            helios_runtime_error("ERROR (PlantArchitecture carbohydrate model): CarbohydrateParameters::fruit_dry_matter_midpoint must lie strictly between 0 and 1 and fruit_dry_matter_width must be positive (got " +
                                 std::to_string(params.fruit_dry_matter_midpoint) + " and " + std::to_string(params.fruit_dry_matter_width) + ").");
        }
        if (!(params.fruit_late_fraction >= 0.f && params.fruit_late_fraction <= 1.f) || !(params.fruit_late_width > 0.f) || !(params.fruit_late_midpoint > 0.f && params.fruit_late_midpoint < 1.f)) {
            helios_runtime_error("ERROR (PlantArchitecture carbohydrate model): CarbohydrateParameters::fruit_late_fraction must be between 0 and 1, fruit_late_midpoint strictly between 0 and 1, and fruit_late_width positive (got " +
                                 std::to_string(params.fruit_late_fraction) + ", " + std::to_string(params.fruit_late_midpoint) + " and " + std::to_string(params.fruit_late_width) + ").");
        }
        const float x = std::clamp(development_fraction, 0.f, 1.f);
        // A logistic curve rescaled to run from 0 at fruit set to 1 at maturity
        auto rescaledLogistic = [x](float midpoint, float width) {
            auto logistic = [&](float position) { return 1.f / (1.f + std::exp(-(position - midpoint) / width)); };
            return (logistic(x) - logistic(0.f)) / (logistic(1.f) - logistic(0.f));
        };
        const float early = rescaledLogistic(params.fruit_dry_matter_midpoint, params.fruit_dry_matter_width);
        if (params.fruit_late_fraction == 0.f) {
            return early;
        }
        return (1.f - params.fruit_late_fraction) * early + params.fruit_late_fraction * rescaledLogistic(params.fruit_late_midpoint, params.fruit_late_width);
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

float Phytomer::calculateMatureFruitCarbon(const FloralBud &fbud) const {
    if (fbud.current_fruit_scale_factor <= 0.f) {
        return 0.f;
    }
    const CarbohydrateParameters &carbohydrate_params = plantarchitecture_ptr->plant_instances.at(this->plantID).carb_parameters;

    // The fruit geometry is scaled uniformly, so its volume goes as the cube of the scale factor.
    float current_volume = 0.f;
    for (uint fruit_objID: fbud.inflorescence_objIDs) {
        if (context_ptr->doesObjectExist(fruit_objID)) {
            current_volume += fruitObjectVolume(context_ptr, fruit_objID);
        }
    }
    const float mature_volume = current_volume / powi(fbud.current_fruit_scale_factor, 3);

    const float fruit_carbon_density = carbohydrate_params.fruit_density * carbohydrate_params.fruit_carbon_percentage / C_molecular_wt; // mol C / m^3
    return fruit_carbon_density * mature_volume;
}

// -------------------- Pool initialization and queries -------------------- //

void PlantArchitecture::markShootConstructionCharged(Shoot &shoot) const {
    const PlantInstance &plant_instance = plant_instances.at(shoot.plantID);
    for (uint p = 0; p < shoot.phytomers.size(); p++) {
        const auto &phytomer = shoot.phytomers.at(p);
        phytomer->internode_carbon_charged_molC = phytomer->calculateInternodeStructuralCarbon(p);
        phytomer->leaf_carbon_charged_molC = phytomer->calculatePhytomerConstructionCosts();
        for (auto &petiole: phytomer->floral_buds) {
            for (auto &fbud: petiole) {
                fbud.fruit_carbon_fraction_charged = (fbud.state == BUD_FRUITING) ? fruitDryMatterFraction(fruitDevelopmentFraction(fbud, plant_instance), plant_instance.carb_parameters) : 0.f;
                fbud.fruit_potential_fraction = fbud.fruit_carbon_fraction_charged;
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
    for (const auto &shoot: plant_instance.shoot_tree) {
        initializeShootCarbohydratePool(plantID, shoot->ID, carbohydrate_concentration_molC_m3);
        wood_volume += shoot->calculateShootInternodeVolume();
    }

    // Roots are filled to the same fraction of their storage capacity as the wood.
    const float root_wood_equivalent_volume = wood_volume / carbohydrate_params.shoot_root_ratio;
    plant_instance.root_sugar_pool_molC = carbohydrate_concentration_molC_m3 * root_wood_equivalent_volume * carbohydrate_params.root_storage_ratio;
    plant_instance.root_starch_pool_molC = 0.f;
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

    float wood_structural_carbon = 0.f;
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
            wood_structural_carbon += phytomer->internode_carbon_charged_molC;
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

    const float root_structural_carbon = wood_structural_carbon / carbohydrate_params.shoot_root_ratio;
    const float root_respiration = root_structural_carbon * carbohydrate_params.r_m_r_20 * temperature_factor * dt;
    plant_instance.root_sugar_pool_molC -= root_respiration;
    plant_instance.carbon_budget.root_maintenance_respiration += root_respiration;
    plant_instance.step_maintenance_respiration_molC = wood_maintenance_respiration_step + root_respiration;
}

void PlantArchitecture::subtractShootGrowthCarbon(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    PlantCarbonBudget &budget = plant_instance.carbon_budget;

    // Fruit past its reserve-funded period is built from current photosynthate: its potential growth is collected here and
    // charged below, once the photosynthate left for it is known.
    if (!(carbohydrate_params.fruit_reserve_funded_days >= 0.f) || !(carbohydrate_params.fruit_drop_averaging_days > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::subtractShootGrowthCarbon): CarbohydrateParameters::fruit_reserve_funded_days must not be negative and fruit_drop_averaging_days must be positive (got " +
                             std::to_string(carbohydrate_params.fruit_reserve_funded_days) + " and " + std::to_string(carbohydrate_params.fruit_drop_averaging_days) + ").");
    }
    struct PhotosynthateFruit {
        Shoot *shoot;
        FloralBud *fbud;
        float potential_carbon; // structural carbon of the fruit's potential growth this sub-step (mol C)
        float potential_fraction;
    };
    std::vector<PhotosynthateFruit> photosynthate_fruit;
    double reserve_funded_spending = 0.0; // vegetative and reserve-funded fruit growth, with roots and growth respiration (mol C)

    for (const auto &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned()) {
            continue; // a pruned shoot has no phytomers left to grow
        }

        // Structural carbon added since the organ was last charged. Organs are only ever charged for growth: a leaf that
        // is shed or an internode whose carbon density is reset is not refunded.
        float wood_growth = 0.f;
        float leaf_growth = 0.f;
        float fruit_growth = 0.f;
        for (uint p = 0; p < shoot->phytomers.size(); p++) {
            const auto &phytomer = shoot->phytomers.at(p);

            const float internode_carbon = phytomer->calculateInternodeStructuralCarbon(p);
            wood_growth += std::max(0.f, internode_carbon - phytomer->internode_carbon_charged_molC);
            phytomer->internode_carbon_charged_molC = internode_carbon;

            const float leaf_carbon = phytomer->calculatePhytomerConstructionCosts();
            leaf_growth += std::max(0.f, leaf_carbon - phytomer->leaf_carbon_charged_molC);
            phytomer->leaf_carbon_charged_molC = leaf_carbon;

            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    // Fruit carbon follows the fruit's dry-matter curve, which is laid down largely while the fruit is young, rather than its volume
                    if (fbud.state == BUD_FRUITING) {
                        const float dry_matter_fraction = fruitDryMatterFraction(fruitDevelopmentFraction(fbud, plant_instance), carbohydrate_params);
                        if (dry_matter_fraction > fbud.fruit_potential_fraction) {
                            const float potential_increment = dry_matter_fraction - fbud.fruit_potential_fraction;
                            const float potential_carbon = phytomer->calculateMatureFruitCarbon(fbud) * potential_increment;
                            if (fbud.time_counter <= carbohydrate_params.fruit_reserve_funded_days) {
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
        }

        const float growth_respiration = carbohydrate_params.growth_respiration_fraction * (wood_growth + leaf_growth) + carbohydrate_params.fruit_growth_respiration_fraction * fruit_growth;
        shoot->sugar_pool_molC -= wood_growth + leaf_growth + fruit_growth + growth_respiration;
        shoot->season_growth_molC += wood_growth + leaf_growth + fruit_growth + growth_respiration;
        shoot->season_fruit_cost_molC += fruit_growth * (1.f + carbohydrate_params.fruit_growth_respiration_fraction);

        // Roots grow in proportion to the wood, and are paid for from the root system's own pool.
        const float root_growth = wood_growth / carbohydrate_params.shoot_root_ratio;
        const float root_growth_respiration = root_growth * carbohydrate_params.growth_respiration_fraction;
        plant_instance.root_sugar_pool_molC -= root_growth + root_growth_respiration;

        budget.wood_growth += wood_growth;
        budget.leaf_growth += leaf_growth;
        budget.fruit_growth += fruit_growth;
        budget.root_growth += root_growth;
        budget.growth_respiration += growth_respiration + root_growth_respiration;
        reserve_funded_spending += wood_growth + leaf_growth + fruit_growth + growth_respiration + root_growth + root_growth_respiration;

        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_growth", wood_growth + leaf_growth + fruit_growth);
            context_ptr->setObjectData(shoot->internode_tube_objID, "daily_growth_respiration", growth_respiration);
        }
    }

    // Photosynthate left for the fruit past its reserve-funded period after maintenance respiration, vegetative growth and
    // reserve-funded fruit, against that fruit's potential growth (both with growth respiration), averaged over the fruit-drop period
    double potential_demand = 0.0;
    for (const auto &fruit: photosynthate_fruit) {
        potential_demand += fruit.potential_carbon * (1.f + carbohydrate_params.fruit_growth_respiration_fraction);
    }
    const double photosynthate = plant_instance.photosynthesis_supply_rate_molC_day * dt;
    // Girth growth, paid from photosynthate, ranks below fruit (allocateSurplusCarbon()), so it does not reduce the fruit's supply
    reserve_funded_spending -= plant_instance.step_girth_growth_molC * (1.f + carbohydrate_params.growth_respiration_fraction) * (1.f + 1.f / carbohydrate_params.shoot_root_ratio);
    const double fruit_supply = std::max(0.0, photosynthate - plant_instance.step_maintenance_respiration_molC - reserve_funded_spending);
    auto &history = plant_instance.fruit_assimilate_history;
    history.push_back({float(fruit_supply), float(potential_demand), dt});
    float history_days = 0.f;
    for (const auto &entry: history) {
        history_days += entry[2];
    }
    while (history.size() > 1 && history_days - history.front()[2] >= carbohydrate_params.fruit_drop_averaging_days) {
        history_days -= history.front()[2];
        history.erase(history.begin());
    }
    double supply_sum = 0.0;
    double demand_sum = 0.0;
    for (const auto &entry: history) {
        supply_sum += entry[0];
        demand_sum += entry[1];
    }
    plant_instance.fruit_growth_supply_ratio = demand_sum > 0.0 ? float(std::min(1.0, supply_sum / demand_sum)) : 1.f;

    // Each fruit grows at its potential times the supply ratio; the potential it reached is recorded either way, so growth lost now is not made up later
    const float supply_ratio = plant_instance.fruit_growth_supply_ratio;
    for (const auto &fruit: photosynthate_fruit) {
        const float fruit_growth = fruit.potential_carbon * supply_ratio;
        const float fruit_growth_respiration = fruit_growth * carbohydrate_params.fruit_growth_respiration_fraction;
        fruit.shoot->sugar_pool_molC -= fruit_growth + fruit_growth_respiration;
        fruit.shoot->season_growth_molC += fruit_growth + fruit_growth_respiration;
        fruit.shoot->season_fruit_cost_molC += fruit_growth + fruit_growth_respiration;
        budget.fruit_growth += fruit_growth;
        budget.growth_respiration += fruit_growth_respiration;
        fruit.fbud->fruit_carbon_fraction_charged += supply_ratio * (fruit.potential_fraction - fruit.fbud->fruit_potential_fraction);
        fruit.fbud->fruit_potential_fraction = fruit.potential_fraction;
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
    return wood_mass_gDW * (1.f + carbohydrate_params.root_storage_ratio / carbohydrate_params.shoot_root_ratio);
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
    if (!(carbohydrate_params.fruit_drop_averaging_days > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::allocateSurplusCarbon): CarbohydrateParameters::fruit_drop_averaging_days must be positive (got " + std::to_string(carbohydrate_params.fruit_drop_averaging_days) + ").");
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
        while (history.size() > 1 && history_days - history.front()[2] >= carbohydrate_params.fruit_drop_averaging_days) {
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
        return b.wood_maintenance_respiration + b.root_maintenance_respiration + b.wood_growth + b.root_growth + b.leaf_growth + b.fruit_growth + b.flower_construction + b.growth_respiration;
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

// -------------------- Fruit drop -------------------- //

void PlantArchitecture::checkCarbonPool_dropFruit(uint plantID, float dt) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const CarbohydrateParameters &carbohydrate_params = plant_instance.carb_parameters;
    if (!(carbohydrate_params.fruit_drop_window_end > carbohydrate_params.fruit_reserve_funded_days)) {
        helios_runtime_error("ERROR (PlantArchitecture::checkCarbonPool_dropFruit): CarbohydrateParameters::fruit_drop_window_end must exceed fruit_reserve_funded_days (got " +
                             std::to_string(carbohydrate_params.fruit_drop_window_end) + " and " + std::to_string(carbohydrate_params.fruit_reserve_funded_days) + " days).");
    }
    if (!(carbohydrate_params.fruit_drop_averaging_days > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::checkCarbonPool_dropFruit): CarbohydrateParameters::fruit_drop_averaging_days must be positive (got " + std::to_string(carbohydrate_params.fruit_drop_averaging_days) + ").");
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
                    if (fbud.state == BUD_FRUITING && fbud.time_counter > carbohydrate_params.fruit_reserve_funded_days && fbud.time_counter <= carbohydrate_params.fruit_drop_window_end) {
                        young_fruit.push_back({fbud.time_counter, phytomer.get(), &fbud});
                    }
                }
            }
        }
    }

    // Fruit built from current photosynthate drops when it cannot grow at the rate it needs: a supply ratio below
    // fruit_abscission_growth_ratio drops the fraction of that fruit that would restore the ratio over the averaging period.
    if (!(carbohydrate_params.fruit_abscission_growth_ratio > 0.f && carbohydrate_params.fruit_abscission_growth_ratio <= 1.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::checkCarbonPool_dropFruit): CarbohydrateParameters::fruit_abscission_growth_ratio must be greater than 0 and at most 1 (got " +
                             std::to_string(carbohydrate_params.fruit_abscission_growth_ratio) + ").");
    }
    const float supply_ratio = plant_instance.fruit_growth_supply_ratio;
    if (young_fruit.empty() || supply_ratio >= carbohydrate_params.fruit_abscission_growth_ratio) {
        plant_instance.fruit_drop_debt = 0.f;
        return;
    }
    const float drop_fraction = std::min(1.f, (1.f - supply_ratio / carbohydrate_params.fruit_abscission_growth_ratio) * dt / carbohydrate_params.fruit_drop_averaging_days);
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
    const float carbon_ratio = plant_instance.carb_parameters.branch_shedding_carbon_ratio;

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
    const float root_mass_gDW = wood_mass_gDW / carbohydrate_params.shoot_root_ratio;
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

                        // Aborting a fruit saves the carbon it would still have needed to reach full size.
                        if (fbud.state == BUD_FRUITING) {
                            working_carb_pool += phytomer->calculateMatureFruitCarbon(fbud) * (1.f - fbud.fruit_carbon_fraction_charged) * (1.f + carbohydrate_params.fruit_growth_respiration_fraction);
                        }
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

void PlantArchitecture::incrementPhytomerInternodeGirth_carb(uint plantID, uint shootID, uint node_number, float dt, bool update_context_geometry, float shoot_length, float shoot_volume) {
    // Slow Radial growth of shoots if their carbon concentration falls below a sustainable threshold
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }

    const CarbohydrateParameters &carbohydrate_params = plant_instances.at(plantID).carb_parameters;
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
    if (!(carbohydrate_params.radial_growth_rate >= 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth_carb): CarbohydrateParameters::radial_growth_rate must not be negative (got " + std::to_string(carbohydrate_params.radial_growth_rate) + " per day).");
    }
    const float closing_fraction = 1.f - std::exp(-carbohydrate_params.radial_growth_rate * dt);
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
    if (carbohydrate_params.sapwood_area_per_leaf_area > 0.f) {
        if (phytomer->ring_season_index != plant_instance.cambium_season_index) {
            phytomer->ring_season_index = plant_instance.cambium_season_index;
            phytomer->ring_season_start_area = PI_F * mean_radius_before * mean_radius_before;
            phytomer->ring_season_peak_leaf_area = 0.f;
        }
        phytomer->ring_season_peak_leaf_area = std::max(phytomer->ring_season_peak_leaf_area, phytomer->live_downstream_leaf_area);
        const float ring_area_owed = carbohydrate_params.sapwood_area_per_leaf_area * phytomer->ring_season_peak_leaf_area / carbohydrate_params.sapwood_functional_years;
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
