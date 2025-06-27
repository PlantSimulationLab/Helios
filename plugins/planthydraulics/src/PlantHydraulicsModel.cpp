/** \file "PlantHydraulicsModel.cpp" Primary source file for plant hydraulics plug-in.

Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "PlantHydraulicsModel.h"
#include "unordered_set"

using namespace std;
using namespace helios;

std::vector<float> linspace(double begin, double end, int n) {
    std::vector<float> result;
    if (n <= 0)
        return result;
    if (n == 1) {
        result.emplace_back(begin);
        return result;
    }

    double step = (end - begin) / (n - 1);
    for (int i = 0; i < n; i++) {
        result.emplace_back(begin + i * step);
    }
    return result;
}

float invertPVCurve(const HydraulicCapacitance &coeffs, float leaf_water_potential) {
    // simple inverted map of relative water content w to water potential psi
    std::vector<float> wis = linspace(0.02, 1.0, 50);
    std::vector<vec2> points(wis.size());
    for (int wi = 0; wi < 50; wi++) {
        points.at(wi).y = wis.at(wi);
        points.at(wi).x = computeWaterPotential(coeffs, wis.at(wi));
    }
    float leaf_relative_water_content = interp1(points, leaf_water_potential);

    return leaf_relative_water_content;
}


float computeOsmoticPotential(const HydraulicCapacitance &coeffs, float relative_water_content) {
    float pi_o = coeffs.osmotic_potential_at_full_turgor; // MPa
    float w = relative_water_content; // -
    float pi = pi_o / w;
    return pi;
}

float computeTurgorPressure(const HydraulicCapacitance &coeffs, float relative_water_content) {
    float pi_o = coeffs.osmotic_potential_at_full_turgor; // MPa
    float w_tlp = coeffs.relative_water_content_at_turgor_loss; // -
    float epsilon = coeffs.cell_wall_elasticity_exponent; // -
    float w = relative_water_content; // -
    float P = -pi_o * powf(max(0.f, (w - w_tlp) / (1.f - w_tlp)), epsilon);
    return P;
}

float computeWaterPotential(const HydraulicCapacitance &coeffs, float relative_water_content) {
    float pi_o = coeffs.osmotic_potential_at_full_turgor; // MPa
    float w_tlp = coeffs.relative_water_content_at_turgor_loss; // -
    float epsilon = coeffs.cell_wall_elasticity_exponent; // -
    float w = relative_water_content; // -
    float P = -pi_o * powf(max(0.f, (w - w_tlp) / (1.f - w_tlp)), epsilon);
    float pi = pi_o / w;
    float psi = P + pi;
    return psi;
}

float computeWaterPotential(float turgor_presure, float osmotic_potential) {
    return min(turgor_presure + osmotic_potential, 0.f);
}

float computeConductance(const HydraulicConductance &coeffs, float water_potential, float temperature) {
    float psi = water_potential; // MPa
    float T = temperature; // K
    float Ksat = coeffs.saturated_conductance; // mol/m2/s/MPa
    float a = coeffs.potential_at_half_saturated; // MPa
    float b = coeffs.sensitivity; // -
    float K;
    if (a == 0.f || b == 0.f) {
        K = Ksat;
    } else {
        K = Ksat / (1.f + pow(abs(psi / a), b));
    }

    if (coeffs.temperature_dependence) {
        K *= powf(T / 298.15f, 7.f);
    }
    return K;
}

float computeCapacitance(const HydraulicCapacitance &coeffs, float relative_water_content) {
    float C = coeffs.fixed_constant_capacitance;
    float Wsat = coeffs.saturated_specific_water_content;
    if (C > 0.f) {
        return C; // mol/m2/MPa
    }
    float w = relative_water_content; // -
    float h = 0.001;
    float f0 = computeWaterPotential(coeffs, w - h);
    float f1 = computeWaterPotential(coeffs, w);
    float f2 = computeWaterPotential(coeffs, w + h);
    float dpsidw = (f0 - 2.f * f1 + f2) / (h * h);
    C = Wsat / dpsidw; // mol/m2/MPa
    return C;
}

float computeCapacitance(const HydraulicCapacitance &coeffs) {
    float C = coeffs.fixed_constant_capacitance; // mol/m2/MPa
    if (C > 0.f) {
        return C;
    }

    helios_runtime_error("PlantHydraulicsModel::computeCapacitance Data fixed_constant_capacitance in HydraulicCapacitance not set for overloaded function computeCapacitance(). Set data value or provide relative_water_content and temperature to "
                         "computeCapacitance().");
    return 0; // never reaches here, but avoids compiler warning
}


PlantHydraulicsModel::PlantHydraulicsModel(helios::Context *a_context) {
    context = a_context;
}

void PlantHydraulicsModel::setModelCoefficients(const PlantHydraulicsModelCoefficients &modelcoefficients) {
    modelcoeffs = modelcoefficients;
    modelcoeffs_map.clear();
}

void PlantHydraulicsModel::setModelCoefficients(const PlantHydraulicsModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs) {
    for (uint UUID: UUIDs) {
        modelcoeffs_map[UUID] = modelcoefficients;
    }
}

void PlantHydraulicsModel::setModelCoefficientsFromLibrary(const std::string &species) {
    PlantHydraulicsModelCoefficients coeffs;
    coeffs = getModelCoefficientsFromLibrary(species);
    modelcoeffs = coeffs;
    modelcoeffs_map.clear();
}

void PlantHydraulicsModel::setModelCoefficientsFromLibrary(const std::string &species, const uint UUID) {
    PlantHydraulicsModelCoefficients coeffs;
    coeffs = getModelCoefficientsFromLibrary(species);
    modelcoeffs_map[UUID] = coeffs;
}

void PlantHydraulicsModel::setModelCoefficientsFromLibrary(const std::string &species, const std::vector<uint> &UUIDs) {
    PlantHydraulicsModelCoefficients coeffs;
    coeffs = getModelCoefficientsFromLibrary(species);
    for (uint UUID: UUIDs) {
        modelcoeffs_map[UUID] = coeffs;
    }
}


PlantHydraulicsModelCoefficients PlantHydraulicsModel::getModelCoefficientsFromLibrary(const std::string &species) {
    PlantHydraulicsModelCoefficients coeffs;
    bool defaultSpecies = false;
    const std::string &s = species;
    if (s == "Walnut" || s == "walnut") {
        coeffs.setLeafHydraulicCapacitance(-1.6386, 0.7683, 2.f);
    } else if (s == "PistachioFemale" || s == "pistachiofemale" || s == "pistachio_female" || s == "Pistachio_Female" || s == "Pistachio_female" || s == "pistachio" || s == "Pistachio") {
        coeffs.setLeafHydraulicCapacitance(-3.096, 0.7652, 2.f);
    } else if (s == "Elderberry" || s == "elderberry" || s == "blue_elderberry") {
        coeffs.setLeafHydraulicCapacitance(-2.011, 0.8135, 2.f);
    } else if (s == "Western_Redbud" || s == "western_redbud" || s == "Redbud" || s == "redbud") {
        coeffs.setLeafHydraulicCapacitance(-2.1963, 0.8872, 1.5);
    } else {
        std::cout << "WARNING (PlantHydraulicsModel::getModelCoefficientsFromLibrary): unknown species " << s << ". Returning default (Walnut)." << std::endl;
        defaultSpecies = true;
    }
    if (!defaultSpecies) {
        std::cout << "Returning Hydraulic Model Coefficients for " << s << std::endl;
    }
    return coeffs;
}


PlantHydraulicsModelCoefficients PlantHydraulicsModel::getModelCoefficients(uint UUID) {
    PlantHydraulicsModelCoefficients modelCoefficients = modelcoeffs_map[UUID];
    return modelCoefficients;
}

float PlantHydraulicsModel::getOrInitializePrimitiveData(uint UUID, const std::string &primitive_data_label, float default_value, bool message_flag) {
    float primitive_data;
    if (context->doesPrimitiveDataExist(UUID, primitive_data_label.c_str())) {
        context->getPrimitiveData(UUID, primitive_data_label.c_str(), primitive_data);
    } else {
        primitive_data = default_value;
        std::string message = "PlantHydraulicsModel::run: Primitive data " + primitive_data_label + " not found, using default value %f\n";
        if (message_flag)
            printf(message.c_str(), primitive_data);
    }
    return primitive_data;
}

int PlantHydraulicsModel::adjustTimestep(float time_step, float min_time_step, float max_time_step, float gradient, float gradient_upper_bound) {
    // Adpative Timestepper
    if (gradient > gradient_upper_bound) {
        time_step *= gradient_upper_bound / gradient; // Reduce timestep if gradient too large
    } else if (gradient < gradient_upper_bound * 0.1) {
        time_step *= 1.1; // Increase timestep if gradient is small
    }

    time_step = std::max(min_time_step, std::min(time_step, max_time_step));
    return int(time_step);
}

void PlantHydraulicsModel::updateRootAndStemWaterPotentialsOfPlant(const std::vector<uint> &UUIDs, int timespan, int timestep) {
    float canopy_transpiration, K_root, K_stem, K_leaf, C_root, C_stem, leaf_water_potential, leaf_transpiration, leaf_temperature, J11, J12, J21, J22, F1, F2, det, delta_psi, delta_psi_root, delta_psi_stem;

    int plantID = getPlantID(UUIDs);

    if (soilWaterPotentialsByPlantID.find(plantID) == soilWaterPotentialsByPlantID.end()) {
        soilWaterPotentialsByPlantID[plantID] = -0.001;
    }

    if (stemWaterPotentialsByPlantID.find(plantID) == stemWaterPotentialsByPlantID.end()) {
        stemWaterPotentialsByPlantID[plantID] = -0.001;
    }

    if (rootWaterPotentialsByPlantID.find(plantID) == rootWaterPotentialsByPlantID.end()) {
        rootWaterPotentialsByPlantID[plantID] = -0.001;
    }


    float soil_water_potential = soilWaterPotentialsByPlantID.at(plantID);
    float stem_water_potential = stemWaterPotentialsByPlantID.at(plantID);
    float root_water_potential = rootWaterPotentialsByPlantID.at(plantID);
    bool steadystate = true;
    if (timespan != 0) {
        steadystate = false;
    }
    float stem_temperature = getOrInitializePrimitiveData(UUIDs.front(), "air_temperature", 298.15f, false);
    float root_temperature = stem_temperature - 5.f;

    context->calculatePrimitiveDataAreaWeightedSum(UUIDs, "latent_flux", canopy_transpiration); // Wm2 -> W
    canopy_transpiration /= 44000.f; // W/m2 -> mol/m2/s

    // Steady State Solution is assumed if time step is not specified/equal to 0.f
    if (steadystate) {
        K_root = computeConductance(modelcoeffs.RootHydraulicConductance, root_water_potential, root_temperature);
        K_stem = computeConductance(modelcoeffs.StemHydraulicConductance, stem_water_potential, stem_temperature);
        root_water_potential = soil_water_potential - canopy_transpiration / K_root;
        stem_water_potential = root_water_potential - canopy_transpiration / K_stem;

        rootWaterPotentialsByPlantID.at(plantID) = root_water_potential;
        stemWaterPotentialsByPlantID.at(plantID) = stem_water_potential;
    } else {
        // Non-steady-state solution
        float dt = float(timestep);
        float dt_min = 10;
        float dt_max = 500.0;
        float delta_psi_max = 0.01;
        int max_iter = 100;
        float tol = 1e-5;
        for (int t = 0; t < int(timespan); t += timestep) {
            // Forward Euler
            //  Fin = K_root * (soil_water_potential - root_water_potential);
            //  Fout = K_stem * (root_water_potential - stem_water_potential);
            //  delta_psi_root = dt / C_root * (Fin - Fout);
            //  root_water_potential += delta_psi_root;
            //
            //  Fin = K_stem * (root_water_potential - stem_water_potential);
            //  Fout = canopy_transpiration;
            //  delta_psi_stem = dt / C_stem * (Fin - Fout);
            //  stem_water_potential += delta_psi_stem;

            // Backward Euler
            float root_water_potential_new = root_water_potential;
            float stem_water_potential_new = stem_water_potential;

            for (int iter = 0; iter < max_iter; ++iter) {
                K_root = computeConductance(modelcoeffs.RootHydraulicConductance, root_water_potential, root_temperature);
                C_root = computeCapacitance(modelcoeffs.RootHydraulicCapacitance);
                K_stem = computeConductance(modelcoeffs.StemHydraulicConductance, stem_water_potential, stem_temperature);
                C_stem = computeCapacitance(modelcoeffs.StemHydraulicCapacitance);

                // F  : C psi/dt = Fin - Fout = Kpsi/dx - Kpsi/dx
                // F1 : C_root dpsi_root/dt = K_root(psi_soil-psi_root) - K_stem(psi_root-psi_stem)
                // F2:  C_stem dpsi_stem/dt = K_stem(psi_root-psi_stem) - E

                // Residuals (F)
                F1 = C_root / dt * (root_water_potential_new - root_water_potential) - K_root * (soil_water_potential - root_water_potential_new) + K_stem * (root_water_potential_new - stem_water_potential_new);

                F2 = C_stem / dt * (stem_water_potential_new - stem_water_potential) - K_stem * (root_water_potential_new - stem_water_potential_new) + canopy_transpiration;

                // Jacobian (partial F / partial psi)
                J11 = C_root / dt + K_root + K_stem;
                J12 = -K_stem;
                J21 = -K_stem;
                J22 = C_stem / dt + K_stem;

                // delta psi = inv(J) * -F
                // inv(J) = 1/det(J) * [J22 -J12; -J21 J11]
                // delta psi = 1/det(J) * [J22 -J12; -J21 J11] * [-F1 -F2]
                det = J11 * J22 - J12 * J21;
                delta_psi_root = (-F1 * J22 + -F2 * -J12) / det;
                delta_psi_stem = (-F1 * -J21 + -F2 * J11) / det;

                root_water_potential_new += delta_psi_root;
                stem_water_potential_new += delta_psi_stem;

                if (fabs(delta_psi_root) < tol && fabs(delta_psi_stem) < tol) {
                    break;
                }
            }

            delta_psi = max(root_water_potential_new - root_water_potential, stem_water_potential_new - stem_water_potential);

            root_water_potential = root_water_potential_new;
            stem_water_potential = stem_water_potential_new;


            // Adpative Timestepper: Reduce timestep if change is large, increase timestep if change is small
            timestep = adjustTimestep(dt, dt_min, dt_max, delta_psi, delta_psi_max);
        }

        rootWaterPotentialsByPlantID.at(plantID) = root_water_potential;
        stemWaterPotentialsByPlantID.at(plantID) = stem_water_potential;
    }
}

void PlantHydraulicsModel::updateLeafWaterPotentialsOfPlant(const std::vector<uint> &UUIDs, int timespan, int timestep) {
    float K_leaf, leaf_water_potential, leaf_transpiration, leaf_relative_water_content, leaf_temperature, leaf_osmotic_content, leaf_area, C_leaf, leaf_water_potential_old, dt, Fin, Fout, stem_water_potential, leaf_turgor_pressure,
            leaf_osmotic_potential, delta_psi, root_water_potential;
    float default_leaf_osmotic_content = 0.2; // mol/m2
    bool steadystate = (timespan == 0);
    float dt_min = 10;
    float dt_max = 500.0;
    float delta_psi_max = 0.01;
    float lambdaw = 44000.f;

    if (steadystate) {
        for (uint i = 0; i < UUIDs.size(); i++) {
            PlantHydraulicsModelCoefficients coeffs;
            if (modelcoeffs_map.empty() || modelcoeffs_map.find(i) == modelcoeffs_map.end()) {
                coeffs = modelcoeffs;
            } else {
                coeffs = modelcoeffs_map.at(i);
            }

            stem_water_potential = stemWaterPotentialsByPlantID.at(getPlantID(UUIDs.at(i)));
            leaf_water_potential = getOrInitializePrimitiveData(UUIDs.at(i), "water_potential", -0.001, false);
            leaf_transpiration = getOrInitializePrimitiveData(UUIDs.at(i), "latent_flux", 0.f, false);
            leaf_temperature = getOrInitializePrimitiveData(UUIDs.at(i), "temperature", 300.f, false);
            leaf_transpiration /= lambdaw; // W/m2 -> mol/m2/s
            K_leaf = computeConductance(coeffs.LeafHydraulicConductance, leaf_water_potential, leaf_temperature);
            leaf_water_potential = stem_water_potential - leaf_transpiration / K_leaf;

            leaf_relative_water_content = invertPVCurve(coeffs.LeafHydraulicCapacitance, leaf_water_potential);
            leaf_relative_water_content = helios::clamp(leaf_relative_water_content, 0.02f, 1.f);
            leaf_turgor_pressure = computeTurgorPressure(coeffs.LeafHydraulicCapacitance, leaf_relative_water_content);
            leaf_osmotic_potential = leaf_water_potential - leaf_turgor_pressure;

            context->setPrimitiveData(UUIDs.at(i), "water_potential", leaf_water_potential);
            context->setPrimitiveData(UUIDs.at(i), "turgor_pressure", leaf_turgor_pressure);
            context->setPrimitiveData(UUIDs.at(i), "osmotic_potential", leaf_osmotic_potential);
            context->setPrimitiveData(UUIDs.at(i), "relative_water_content", leaf_relative_water_content);
        }
    } else {
        // Non-steady-state
        for (uint i = 0; i < UUIDs.size(); i++) {
            PlantHydraulicsModelCoefficients coeffs;
            if (modelcoeffs_map.empty() || modelcoeffs_map.find(i) == modelcoeffs_map.end()) {
                coeffs = modelcoeffs;
            } else {
                coeffs = modelcoeffs_map.at(i);
            }

            leaf_transpiration = getOrInitializePrimitiveData(UUIDs.at(i), "latent_flux", 0.f, false);
            leaf_temperature = getOrInitializePrimitiveData(UUIDs.at(i), "temperature", 300.f, false);
            leaf_osmotic_content = getOrInitializePrimitiveData(UUIDs.at(i), "osmotic_content", default_leaf_osmotic_content, false);
            leaf_relative_water_content = getOrInitializePrimitiveData(UUIDs.at(i), "relative_water_content", 1.0f, false);
            leaf_water_potential = getOrInitializePrimitiveData(UUIDs.at(i), "water_potential", -0.001, false);
            stem_water_potential = stemWaterPotentialsByPlantID.at(getPlantID(UUIDs.at(i)));

            leaf_transpiration /= 44000.f; // W/m2 to mol/m2/s
            dt = float(timestep);
            for (int t = 0; t < int(timespan); t += timestep) {
                leaf_water_potential_old = leaf_water_potential;
                K_leaf = computeConductance(coeffs.LeafHydraulicConductance, leaf_water_potential, leaf_temperature); // mol/m2/s/MPa
                C_leaf = computeCapacitance(coeffs.LeafHydraulicCapacitance, leaf_relative_water_content); // mol/m2/MPa
                Fin = K_leaf * (stem_water_potential - leaf_water_potential); // mol/m2/s
                Fout = leaf_transpiration; // mol/m2/s
                leaf_relative_water_content += dt * (Fin - Fout);
                // printf("leaf_water_potential: %f, C_leaf: %f, Fin: %f, Fout: %f\n",leaf_water_potential,C_leaf,Fin,Fout);
                //  leaf_relative_water_content += (Fin - Fout) * dt / (coeffs.LeafHydraulicCapacitance.saturated_specific_water_content);
                leaf_relative_water_content = helios::clamp(leaf_relative_water_content, 0.02f, 1.f);
                leaf_turgor_pressure = computeTurgorPressure(coeffs.LeafHydraulicCapacitance, leaf_relative_water_content);
                leaf_osmotic_potential = computeOsmoticPotential(coeffs.LeafHydraulicCapacitance, leaf_relative_water_content);
                leaf_water_potential = computeWaterPotential(leaf_turgor_pressure, leaf_osmotic_potential);
                // printf("pi: %f + P: %f = %f\n",leaf_osmotic_potential,leaf_turgor_pressure,leaf_water_potential);
                delta_psi = abs(leaf_water_potential - leaf_water_potential_old);

                timestep = adjustTimestep(dt, dt_min, dt_max, delta_psi, delta_psi_max);
            }
            leaf_relative_water_content = invertPVCurve(coeffs.LeafHydraulicCapacitance, leaf_water_potential);
            leaf_turgor_pressure = computeTurgorPressure(coeffs.LeafHydraulicCapacitance, leaf_relative_water_content);
            leaf_osmotic_potential = computeOsmoticPotential(coeffs.LeafHydraulicCapacitance, leaf_relative_water_content);
            leaf_water_potential = computeWaterPotential(leaf_turgor_pressure, leaf_osmotic_potential);


            context->setPrimitiveData(UUIDs.at(i), "turgor_pressure", leaf_turgor_pressure);
            context->setPrimitiveData(UUIDs.at(i), "osmotic_potential", leaf_osmotic_potential);
            context->setPrimitiveData(UUIDs.at(i), "relative_water_content", leaf_relative_water_content);
            context->setPrimitiveData(UUIDs.at(i), "water_potential", leaf_water_potential);
            if (OutputConductance) {
                context->setPrimitiveData(UUIDs.at(i), "hydraulic_conductance", K_leaf);
            }
            if (OutputCapacitance) {
                context->setPrimitiveData(UUIDs.at(i), "hydraulic_capacitance", C_leaf);
            }
        }
    }
}

void PlantHydraulicsModel::outputConductancePrimitiveData(bool toggle) {
    OutputConductance = toggle;
}

void PlantHydraulicsModel::outputCapacitancePrimitiveData(bool toggle) {
    OutputCapacitance = toggle;
}

void PlantHydraulicsModel::groupPrimitivesIntoPlantObject(const std::vector<uint> &UUIDs) {
    if (!UUIDs.empty()) {
        int plantID = context->addPolymeshObject(UUIDs);
        context->setObjectData(plantID, "plantID", plantID);
    }
}

int PlantHydraulicsModel::getPlantID(uint UUID) {
    int plantID;
    context->getObjectData(context->getPrimitiveParentObjectID(UUID), "plantID", plantID);
    return plantID;
}

int PlantHydraulicsModel::getPlantID(const std::vector<uint> &UUIDs) {
    if (UUIDs.empty()) {
        std::cout << "(PlantHydraulicsModel::getPlantID): UUIDs is empty. Returning -1. " << std::endl;
        return 0;
    }
    int plantIDfront;
    int plantIDback;
    int plantID;
    context->getObjectData(context->getPrimitiveParentObjectID(UUIDs.front()), "plantID", plantIDfront);
    context->getObjectData(context->getPrimitiveParentObjectID(UUIDs.back()), "plantID", plantIDback);
    if (plantIDfront == plantIDback) {
        plantID = plantIDfront;
        return plantID;
    }

    std::stringstream ss;
    ss << "(PlantHydraulicsModel::getPlantID): UUIDs do not share the same plantID. First primitive plantID: " << plantIDfront << ", last primitive plantID: " << plantIDback << ".";
    helios_runtime_error(ss.str());
    return 0; // This should never be reached, but avoids compiler warning
}

std::vector<int> PlantHydraulicsModel::getUniquePlantIDs(const std::vector<uint> &UUIDs) {
    std::vector<int> plantIDs;
    int plantID;
    std::vector<uint> plantObjectIDs;
    plantObjectIDs = context->getUniquePrimitiveParentObjectIDs(UUIDs);
    for (auto plantObjID: plantObjectIDs) {
        context->getObjectData(plantObjID, "plantID", plantID);
        if (std::find(plantIDs.begin(), plantIDs.end(), plantID) == plantIDs.end()) {
            plantIDs.emplace_back(plantID);
        }
    }

    return plantIDs;
}

float PlantHydraulicsModel::getStemWaterPotential(uint UUID) {
    uint plantID = getPlantID(UUID);
    return stemWaterPotentialsByPlantID.at(plantID);
}

float PlantHydraulicsModel::getStemWaterPotential(const std::vector<uint> &UUIDs) {
    uint plantID = getPlantID(UUIDs);
    return stemWaterPotentialsByPlantID.at(plantID);
}

float PlantHydraulicsModel::getStemWaterPotentialOfPlant(uint plantID) {
    return stemWaterPotentialsByPlantID.at(plantID);
}

float PlantHydraulicsModel::getRootWaterPotential(uint UUID) {
    uint plantID = getPlantID(UUID);
    return rootWaterPotentialsByPlantID.at(plantID);
}

float PlantHydraulicsModel::getRootWaterPotential(const std::vector<uint> &UUIDs) {
    uint plantID = getPlantID(UUIDs);
    return rootWaterPotentialsByPlantID.at(plantID);
}


float PlantHydraulicsModel::getRootWaterPotentialOfPlant(uint plantID) {
    return rootWaterPotentialsByPlantID.at(plantID);
}

float PlantHydraulicsModel::getSoilWaterPotential(uint UUID) {
    uint plantID = getPlantID(UUID);
    return soilWaterPotentialsByPlantID.at(plantID);
}

float PlantHydraulicsModel::getSoilWaterPotential(const std::vector<uint> &UUIDs) {
    uint plantID = getPlantID(UUIDs);
    return soilWaterPotentialsByPlantID.at(plantID);
}

float PlantHydraulicsModel::getSoilWaterPotentialOfPlant(uint plantID) {
    return soilWaterPotentialsByPlantID.at(plantID);
}

void PlantHydraulicsModel::setSoilWaterPotentialOfPlant(uint plantID, float soil_water_potential) {
    soilWaterPotentialsByPlantID[plantID] = soil_water_potential;
}

std::vector<uint> PlantHydraulicsModel::getPrimitivesWithoutPlantID(std::vector<uint> UUIDs) {
    auto objectIDs = context->getUniquePrimitiveParentObjectIDs(UUIDs);
    std::vector<uint> allPrimitivesWithPlantID;
    for (auto objectID: objectIDs) {
        if (context->doesObjectDataExist(objectID, "plantID")) {
            auto newleaves = context->getObjectPrimitiveUUIDs(objectID);
            allPrimitivesWithPlantID.insert(allPrimitivesWithPlantID.end(), newleaves.begin(), newleaves.end());
        }
    }

    std::unordered_set<uint> seen;
    seen.reserve(allPrimitivesWithPlantID.size());
    for (auto UUID: allPrimitivesWithPlantID) {
        seen.insert(UUID);
    }

    std::vector<uint> PrimitivesWithoutPlantID;
    PrimitivesWithoutPlantID.reserve(UUIDs.size());
    for (auto UUID: UUIDs) {
        if (seen.find(UUID) == seen.end()) {
            PrimitivesWithoutPlantID.emplace_back(UUID);
        }
    }
    return PrimitivesWithoutPlantID;
}

void PlantHydraulicsModel::run(int timespan, int timestep) {
    auto UUIDs = context->getAllUUIDs();

    groupPrimitivesIntoPlantObject(getPrimitivesWithoutPlantID(UUIDs));

    std::vector<uint> plantObjectIDs = context->getUniquePrimitiveParentObjectIDs(UUIDs);
    for (uint plantObjectID: plantObjectIDs) {
        int plantID;
        context->getObjectData(plantObjectID, "plantID", plantID);
        if (std::find(plantIDs.begin(), plantIDs.end(), plantID) == plantIDs.end()) {
            plantIDs.emplace_back(plantID);
        }
    }

    for (int plantID: plantIDs) {
        std::vector<uint> leavesOfPlant = context->getObjectPrimitiveUUIDs(context->filterObjectsByData(context->getUniquePrimitiveParentObjectIDs(UUIDs), std::string("plantID"), plantID, "=="));
        updateRootAndStemWaterPotentialsOfPlant(leavesOfPlant, timespan, timestep);
        updateLeafWaterPotentialsOfPlant(leavesOfPlant, timespan, timestep);
    }

    plantIDs.clear();
}

void PlantHydraulicsModel::run(const uint UUID, int timespan, int timestep) {
    std::vector<uint> leavesOfPlant = {UUID};
    groupPrimitivesIntoPlantObject(leavesOfPlant);
    plantIDs = context->getUniquePrimitiveParentObjectIDs({UUID});


    updateRootAndStemWaterPotentialsOfPlant(leavesOfPlant, timespan, timestep);
    updateLeafWaterPotentialsOfPlant(leavesOfPlant, timespan, timestep);

    plantIDs.clear();
}

void PlantHydraulicsModel::run(const std::vector<uint> &UUIDs, int timespan, int timestep) {
    if (UUIDs.empty()) {
        std::cout << "(PlantHydraulicsModel::run): UUIDs is empty. Exiting. " << std::endl;
        return;
    }
    groupPrimitivesIntoPlantObject(getPrimitivesWithoutPlantID(UUIDs));

    std::vector<uint> plantObjectIDs = context->getUniquePrimitiveParentObjectIDs(UUIDs);
    for (uint plantObjectID: plantObjectIDs) {
        int plantID;
        context->getObjectData(plantObjectID, "plantID", plantID);
        if (std::find(plantIDs.begin(), plantIDs.end(), plantID) == plantIDs.end()) {
            plantIDs.emplace_back(plantID);
        }
    }

    for (int plantID: plantIDs) {
        std::vector<uint> leavesOfPlant = context->getObjectPrimitiveUUIDs(context->filterObjectsByData(context->getUniquePrimitiveParentObjectIDs(UUIDs), std::string("plantID"), plantID, "=="));
        updateRootAndStemWaterPotentialsOfPlant(leavesOfPlant, timespan, timestep);
        updateLeafWaterPotentialsOfPlant(leavesOfPlant, timespan, timestep);
    }

    plantIDs.clear();
}

std::vector<std::vector<std::vector<float>>> PlantHydraulicsModel::meshgrid(const std::vector<float> &x, const std::vector<float> &y) {
    std::vector<std::vector<float>> xGrid;
    std::vector<std::vector<float>> yGrid;
    size_t nx = x.size();
    size_t ny = y.size();

    xGrid.resize(ny, std::vector<float>(nx));
    yGrid.resize(ny, std::vector<float>(nx));

    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            xGrid[i][j] = x[j];
            yGrid[i][j] = y[i];
        }
    }
    return {xGrid, yGrid};
}
