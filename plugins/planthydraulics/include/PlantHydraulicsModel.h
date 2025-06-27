/** \file "PlantHydraulicsModel.h" Primary header file for the Plant Hydraulics plug-in.

Copyright (C) 2016-2025 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#ifndef PLANTHYDRAULICS_MODEL
#define PLANTHYDRAULICS_MODEL

#include "Context.h"

struct HydraulicConductance {
    HydraulicConductance() {
        saturated_conductance = 0.5f;
        potential_at_half_saturated = 0.f;
        sensitivity = 0.f;
        temperature_dependence = false;
    }

    explicit HydraulicConductance(float saturated_conductance) {
        this->saturated_conductance = saturated_conductance;
        this->potential_at_half_saturated = 0.f;
        this->sensitivity = 0.f;
        this->temperature_dependence = false;
    }

    HydraulicConductance(float saturated_conductance, float potential_at_half_saturated) {
        this->saturated_conductance = saturated_conductance;
        this->potential_at_half_saturated = potential_at_half_saturated;
        this->sensitivity = 5.f;
        this->temperature_dependence = false;
    }

    HydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity) {
        this->saturated_conductance = saturated_conductance;
        this->potential_at_half_saturated = potential_at_half_saturated;
        this->sensitivity = sensitivity;
        this->temperature_dependence = false;
    }

    HydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity, bool temperature_dependence) {
        this->saturated_conductance = saturated_conductance;
        this->potential_at_half_saturated = potential_at_half_saturated;
        this->sensitivity = sensitivity;
        this->temperature_dependence = temperature_dependence;
    }

    void setTemperatureDependence(bool temperature_dependence) {
        this->temperature_dependence = temperature_dependence;
    }

    float saturated_conductance; // mol/m²/s/MPa
    float potential_at_half_saturated; // MPa
    float sensitivity; // unitless
    bool temperature_dependence; // 1 or true to toggle on temperature dependence
};

struct HydraulicCapacitance {
    HydraulicCapacitance() {
        this->osmotic_potential_at_full_turgor = -2.f; // MPa
        this->relative_water_content_at_turgor_loss = 0.8f; // unitless
        this->cell_wall_elasticity_exponent = 1.f; // unitless, affects the slope of turgor versus water content
        this->saturated_specific_water_content = 1.f; // mol/m²
    }

    explicit HydraulicCapacitance(float capacitance) {
        if (capacitance <= 0.f) {
            std::string message = "PlantHydraulicsModel::HydraulicCapacitance Capacitance given must be greater than zero. Given value was " + std::to_string(capacitance);
            helios::helios_runtime_error(message);
        }
        this->fixed_constant_capacitance = capacitance;
    }

    HydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss) {
        this->osmotic_potential_at_full_turgor = osmotic_potential_at_full_turgor; // MPa
        this->relative_water_content_at_turgor_loss = relative_water_content_at_turgor_loss; // unitless
        this->cell_wall_elasticity_exponent = 1.f; // unitless, affects the slope of turgor versus water content
        this->saturated_specific_water_content = 1.f; // mol/m²
    }

    HydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss, float cell_wall_elasticity_exponent) {
        this->osmotic_potential_at_full_turgor = osmotic_potential_at_full_turgor; // MPa
        this->relative_water_content_at_turgor_loss = relative_water_content_at_turgor_loss; // unitless
        this->cell_wall_elasticity_exponent = cell_wall_elasticity_exponent; // unitless, affects the slope of turgor versus water content
        this->saturated_specific_water_content = 1.f; // mol/m²
    }

    HydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss, float cell_wall_elasticity_exponent, float saturated_specific_water_content) {
        this->osmotic_potential_at_full_turgor = osmotic_potential_at_full_turgor; // MPa
        this->relative_water_content_at_turgor_loss = relative_water_content_at_turgor_loss; // unitless
        this->cell_wall_elasticity_exponent = cell_wall_elasticity_exponent; // unitless, affects the slope of turgor versus water content
        this->saturated_specific_water_content = saturated_specific_water_content; // mol/m²
    }

    float osmotic_potential_at_full_turgor = -2.f; // MPa
    float relative_water_content_at_turgor_loss = 0.8f; // unitless
    float cell_wall_elasticity_exponent = 1.f; // unitless, affects the slope of turgor versus water content
    float saturated_specific_water_content = 1.f; // mol/m²
    float fixed_constant_capacitance = -1.f; // mol/m²/MPa optional fixed, constant capacitance
};

//! Compute osmotic potential as a function of water content, osmotic content, temperature using parameterized pressure-volume relationship.
/**
 * \param[in] coeffs Pointer to hydraulic capacitance (pressure volume curve) model coefficient struct.
 * \param[in] relative_water_content Relative water content (-).
 * \return Water potential (MPa).
 */
float computeOsmoticPotential(const HydraulicCapacitance &coeffs, float relative_water_content);

//! Compute turgor pressure as a function of water content using parameterized pressure-volume relationship.
/**
 * \param[in] coeffs Pointer to hydraulic capacitance (pressure volume curve) model coefficient struct.
 * \param[in] relative_water_content Relative water content (-).
 * \return Water potential (MPa).
 */
float computeTurgorPressure(const HydraulicCapacitance &coeffs, float relative_water_content);

//! Compute water potential as a function of water content using parameterized pressure-volume relationship.
/**
 * \param[in] coeffs Pointer to hydraulic capacitance (pressure volume curve) model coefficient struct.
 * \param[in] relative_water_content Relative water content (-).
 * \return Water potential (MPa).
 */
float computeWaterPotential(const HydraulicCapacitance &coeffs, float relative_water_content);

//! Compute water potential as the sum of pressure potential and osmotic potential.
/**
 * \param[in] pressure_potential Pressure potential (+, MPa).
 * \param[in] osmotic_potential Osmotic potential (-, MPa).
 * \return Water potential (MPa).
 */
float computeWaterPotential(float pressure_potential, float osmotic_potential);


//! Compute hydraulic conductance as a function of water potential using parameterized pressure-volume relationship.
/**
 * \param[in] coeffs Pointer to hydraulic capacitance (pressure volume curve) model coefficient struct.
 * \param[in] water_potential Water potential (MPa).
 * \param[in] temperature Primitive temperature (K).
 * \return Hydraulic conductance (mol/m²/s/MPa).
 */
float computeConductance(const HydraulicConductance &coeffs, float water_potential, float temperature);

//! Compute hydraulic conductance as a function of water potential using parameterized pressure-volume relationship.
/**
 * \param[in] coeffs Pointer to hydraulic capacitance (pressure volume curve) model coefficient struct.
 * \param[in] relative_water_content Relative water content (-).
 * \return Hydraulic capacitance (mol/m²/MPa).
 */
float computeCapacitance(const HydraulicCapacitance &coeffs, float relative_water_content);

struct PlantHydraulicsModelCoefficients {
public:
    PlantHydraulicsModelCoefficients() = default;

    //! Set leaf hydraulic conductance as a constant value
    /**
     * \param[in] conductance Value of a constant leaf hydraulic conductance (mol/m²/s/MPa). Typical range: 0.001 to 1.
     */
    void setLeafHydraulicConductance(float conductance) {
        this->LeafHydraulicConductance = HydraulicConductance(conductance);
    }

    //! Set leaf hydraulic conductance as a function of leaf water potential
    /**
     * \param[in] saturated_conductance Value of leaf hydraulic conductance at saturation (leaf water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which conductance is half its saturated value (MPa).
     */
    void setLeafHydraulicConductance(float saturated_conductance, float potential_at_half_saturated) {
        this->LeafHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated);
    }

    //! Set leaf hydraulic conductance as a function of leaf water potential
    /**
     * \param[in] saturated_conductance Value of leaf hydraulic conductance at saturation (leaf water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which hydraulic conductance is half its saturated value (MPa).
     * \param[in] sensitivity Functional parameter controlling the sensitivity (maximum slope) of hydraulic conductance with respect to water potential.
     */
    void setLeafHydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity) {
        this->LeafHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated, sensitivity);
    }

    //! Set leaf hydraulic conductance as a function of leaf water potential
    /**
     * \param[in] saturated_conductance Value of leaf hydraulic conductance at saturation (leaf water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which hydraulic conductance is half its saturated value (MPa).
     * \param[in] sensitivity Functional parameter controlling the sensitivity (maximum slope) of hydraulic conductance with respect to water potential.
     * \param[in] temperature_dependence Toggle the temperature dependence of hydraulic conductance, i.e K(T) = K*(T/298.15)^7
     */
    void setLeafHydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity, bool temperature_dependence) {
        this->LeafHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated, sensitivity, temperature_dependence);
    }

    //! Toggle leaf hydraulic conductance's temperature dependence
    /**
     * \param[in] temperature_dependence Toggle the temperature dependence of hydraulic conductance, i.e K(T) = K*(T/298.15)^7
     */
    void setLeafHydraulicConductanceTemperatureDependence(bool temperature_dependence) {
        this->LeafHydraulicConductance.setTemperatureDependence(temperature_dependence);
    }

    //! Set stem hydraulic conductance as a constant value
    /**
     * \param[in] conductance Value of a constant stem hydraulic conductance (mol/m²/s/MPa). Typical range: 0.001 to 1.
     */
    void setStemHydraulicConductance(float conductance) {
        this->StemHydraulicConductance = HydraulicConductance(conductance);
    }

    //! Set stem hydraulic conductance as a function of stem water potential
    /**
     * \param[in] saturated_conductance Value of stem hydraulic conductance at saturation (stem water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which conductance is half its saturated value (MPa).
     */
    void setStemHydraulicConductance(float saturated_conductance, float potential_at_half_saturated) {
        this->StemHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated);
    }

    //! Set stem hydraulic conductance as a function of stem water potential
    /**
     * \param[in] saturated_conductance Value of stem hydraulic conductance at saturation (stem water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which hydraulic conductance is half its saturated value (MPa).
     * \param[in] sensitivity Functional parameter controlling the sensitivity (maximum slope) of hydraulic conductance with respect to water potential.
     */
    void setStemHydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity) {
        this->StemHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated, sensitivity);
    }

    //! Set stem hydraulic conductance as a function of stem water potential
    /**
     * \param[in] saturated_conductance Value of stem hydraulic conductance at saturation (stem water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which hydraulic conductance is half its saturated value (MPa).
     * \param[in] sensitivity Functional parameter controlling the sensitivity (maximum slope) of hydraulic conductance with respect to water potential.
     * \param[in] temperature_dependence Toggle the temperature dependence of hydraulic conductance, i.e K(T) = K*(T/298.15)^7
     */
    void setStemHydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity, bool temperature_dependence) {
        this->StemHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated, sensitivity, temperature_dependence);
    }

    //! Toggle stem hydraulic conductance's temperature dependence
    /**
     * \param[in] temperature_dependence Toggle the temperature dependence of hydraulic conductance, i.e K(T) = K*(T/298.15)^7
     */
    void setStemHydraulicConductanceTemperatureDependence(bool temperature_dependence) {
        this->RootHydraulicConductance.setTemperatureDependence(temperature_dependence);
    }

    //! Set root hydraulic conductance as a constant value
    /**
     * \param[in] conductance Value of root hydraulic conductance (mol/m²/s/MPa). Typical range: 0.001 to 1.
     */
    void setRootHydraulicConductance(float conductance) {
        this->RootHydraulicConductance = HydraulicConductance(conductance);
    }

    //! Set root hydraulic conductance as a function of root water potential
    /**
     * \param[in] saturated_conductance Value of root hydraulic conductance at saturation (root water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which hydraulic conductance is half its saturated value (MPa).
     */
    void setRootHydraulicConductance(float saturated_conductance, float potential_at_half_saturated) {
        this->RootHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated);
    }

    //! Set root hydraulic conductance as a function of root water potential
    /**
     * \param[in] saturated_conductance Value of root hydraulic conductance at saturation (root water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which hydraulic conductance is half its saturated value (MPa).
     * \param[in] sensitivity Functional parameter controlling the sensitivity (maximum slope) of hydraulic conductance with respect to water potential.
     */
    void setRootHydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity) {
        this->RootHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated, sensitivity);
    }

    //! Set root hydraulic conductance as a function of root water potential
    /**
     * \param[in] saturated_conductance Value of root hydraulic conductance at saturation (root water potential of zero) (mol/m²/s/MPa). Typical range: 0.001 to 1.
     * \param[in] potential_at_half_saturated Functional parameter controlling the water potential at which hydraulic conductance is half its saturated value (MPa).
     * \param[in] sensitivity Functional parameter controlling the sensitivity (maximum slope) of hydraulic conductance with respect to water potential.
     * \param[in] temperature_dependence Toggle the temperature dependence of hydraulic conductance, i.e K(T) = K*(T/298.15)^7
     */
    void setRootHydraulicConductance(float saturated_conductance, float potential_at_half_saturated, float sensitivity, bool temperature_dependence) {
        this->RootHydraulicConductance = HydraulicConductance(saturated_conductance, potential_at_half_saturated, sensitivity, temperature_dependence);
    }

    //! Toggle root hydraulic conductance's temperature dependence
    /**
     * \param[in] temperature_dependence Toggle the temperature dependence of hydraulic conductance, i.e K(T) = K*(T/298.15)^7
     */
    void setRootHydraulicConductanceTemperatureDependence(bool temperature_dependence) {
        this->RootHydraulicConductance.setTemperatureDependence(temperature_dependence);
    }

    //! Set leaf hydraulic capacitance as a constant value
    /**
     * \param[in] capacitance Value of a constant leaf hydraulic capacitance (mol/m²/MPa). Typical range 0.005-0.5.
     */
    void setLeafHydraulicCapacitance(float capacitance) {
        this->LeafHydraulicCapacitance = HydraulicCapacitance(capacitance);
    }

    //! Set leaf hydraulic capacitance as a function of water content with pressure-volume curve derived parameters.
    /**
     * \param[in] osmotic_potential_at_full_turgor Osmotic potential at full turgor (also, -1*maximum turgor pressure) (MPa)
     * \param[in] relative_water_content_at_turgor_loss Relative water content at the turgor loss point (unitless) Typical range 0.7-0.95.
     */
    void setLeafHydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss) {
        this->LeafHydraulicCapacitance = HydraulicCapacitance(osmotic_potential_at_full_turgor, relative_water_content_at_turgor_loss);
    }

    //! Set leaf hydraulic capacitance as a function of water content with pressure-volume curve derived parameters.
    /**
     * \param[in] osmotic_potential_at_full_turgor Osmotic potential at full turgor (also, -1*maximum turgor pressure) (MPa)
     * \param[in] relative_water_content_at_turgor_loss Relative water content at the turgor loss point (unitless) Typical range 0.7-0.95.
     * \param[in] cell_wall_elasticity_exponent Exponent of turgor pressure verse relative water content relationship (unitless) Typical range 1-2.
     */
    void setLeafHydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss, float cell_wall_elasticity_exponent) {
        this->LeafHydraulicCapacitance = HydraulicCapacitance(osmotic_potential_at_full_turgor, relative_water_content_at_turgor_loss, cell_wall_elasticity_exponent);
    }

    //! Set leaf hydraulic capacitance as a function of water content with pressure-volume curve derived parameters from Helios library.
    /**
     * \param[in] species String of species name to select from parameter library
     */
    void setLeafHydraulicCapacitanceFromLibrary(std::string species) {
        const std::string &s = species;
        float osmotic_potential_at_full_turgor;
        float relative_water_content_at_turgor_loss;
        float cell_wall_elasticity_exponent;

        if (s == "Walnut" || s == "walnut") {
            osmotic_potential_at_full_turgor = -1.6386;
            relative_water_content_at_turgor_loss = 0.7683;
            cell_wall_elasticity_exponent = 2.f;
        } else if (s == "PistachioFemale" || s == "pistachiofemale" || s == "pistachio_female" || s == "Pistachio_Female" || s == "Pistachio_female" || s == "pistachio" || s == "Pistachio") {
            osmotic_potential_at_full_turgor = 0.7652;
            relative_water_content_at_turgor_loss = 0.7683;
            cell_wall_elasticity_exponent = 2.f;
        } else if (s == "Elderberry" || s == "elderberry" || s == "blue_elderberry") {
            osmotic_potential_at_full_turgor = -2.011;
            relative_water_content_at_turgor_loss = 0.8135;
            cell_wall_elasticity_exponent = 2.f;
        } else if (s == "Western_Redbud" || s == "western_redbud" || s == "Redbud" || s == "redbud") {
            osmotic_potential_at_full_turgor = -2.1963;
            relative_water_content_at_turgor_loss = 0.8872;
            cell_wall_elasticity_exponent = 1.5f;
        } else {
            std::cout << "WARNING (PlantHydraulicsModel::getModelCoefficientsFromLibrary): unknown species " << s << ". Setting default (Walnut)." << std::endl;
            osmotic_potential_at_full_turgor = -2.1963;
            relative_water_content_at_turgor_loss = 0.8872;
            cell_wall_elasticity_exponent = 1.5f;
        }

        this->LeafHydraulicCapacitance = HydraulicCapacitance(osmotic_potential_at_full_turgor, relative_water_content_at_turgor_loss, cell_wall_elasticity_exponent);
    }

    //! Set stem hydraulic capacitance as a constant value
    /**
     * \param[in] capacitance Value of a constant leaf hydraulic capacitance (m3/MPa) Typical Range 0.005-2.
     */
    void setStemHydraulicCapacitance(float capacitance) {
        this->StemHydraulicCapacitance = HydraulicCapacitance(capacitance);
    }

    //! Set stem hydraulic capacitance as a function of water content with pressure-volume curve derived parameters.
    /**
     * \param[in] osmotic_potential_at_full_turgor Osmotic potential at full turgor (also, -1*maximum turgor pressure) (MPa)
     * \param[in] relative_water_content_at_turgor_loss Relative water content at the turgor loss point (unitless) Typical range 0.7-0.95.
     */
    void setStemHydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss) {
        this->StemHydraulicCapacitance = HydraulicCapacitance(osmotic_potential_at_full_turgor, relative_water_content_at_turgor_loss);
    }

    //! Set stem hydraulic capacitance as a function of water content with pressure-volume curve derived parameters.
    /**
     * \param[in] osmotic_potential_at_full_turgor Osmotic potential at full turgor (also, -1*maximum turgor pressure) (MPa)
     * \param[in] relative_water_content_at_turgor_loss Relative water content at the turgor loss point (unitless) Typical range 0.7-0.95.
     * \param[in] cell_wall_elasticity_exponent Exponent of turgor pressure verse relative water content relationship (unitless) Typical range 1-2.
     */
    void setStemHydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss, float cell_wall_elasticity_exponent) {
        this->StemHydraulicCapacitance = HydraulicCapacitance(osmotic_potential_at_full_turgor, relative_water_content_at_turgor_loss, cell_wall_elasticity_exponent);
    }

    //! Set root hydraulic capacitance as a constant value
    /**
     * \param[in] capacitance Value of a constant leaf hydraulic capacitance (mol/m²/MPa) Typical range 0.005-2.
     */
    void setRootHydraulicCapacitance(float capacitance) {
        this->RootHydraulicCapacitance = HydraulicCapacitance(capacitance);
    }

    //! Set root hydraulic capacitance as a function of water content with pressure-volume curve derived parameters.
    /**
     * \param[in] osmotic_potential_at_full_turgor Osmotic potential at full turgor (also, -1*maximum turgor pressure) (MPa)
     * \param[in] relative_water_content_at_turgor_loss Relative water content at the turgor loss point (unitless) Typical range 0.7-0.95.
     */
    void setRootHydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss) {
        this->RootHydraulicCapacitance = HydraulicCapacitance(osmotic_potential_at_full_turgor, relative_water_content_at_turgor_loss);
    }

    //! Set root hydraulic capacitance as a function of water content with pressure-volume curve derived parameters.
    /**
     * \param[in] osmotic_potential_at_full_turgor Osmotic potential at full turgor (also, -1*maximum turgor pressure) (MPa)
     * \param[in] relative_water_content_at_turgor_loss Relative water content at the turgor loss point (unitless) Typical range 0.7-0.95.
     * \param[in] cell_wall_elasticity_exponent Exponent of turgor pressure verse relative water content relationship (unitless) Typical range 1-2.
     */
    void setRootHydraulicCapacitance(float osmotic_potential_at_full_turgor, float relative_water_content_at_turgor_loss, float cell_wall_elasticity_exponent) {
        this->RootHydraulicCapacitance = HydraulicCapacitance(osmotic_potential_at_full_turgor, relative_water_content_at_turgor_loss, cell_wall_elasticity_exponent);
    }


private:
    HydraulicConductance LeafHydraulicConductance;
    HydraulicConductance StemHydraulicConductance;
    HydraulicConductance RootHydraulicConductance;

    HydraulicCapacitance LeafHydraulicCapacitance;
    HydraulicCapacitance StemHydraulicCapacitance;
    HydraulicCapacitance RootHydraulicCapacitance;

    friend class PlantHydraulicsModel;
};

//! PlantHydraulics model class
class PlantHydraulicsModel {
public:
    //! Constructor
    explicit PlantHydraulicsModel(helios::Context *context);

    //! Runs a self-test of the PlantHydraulicsModel, verifying its functionality.
    /**
     * The self-test executes a series of internal tests, validating critical methods and
     * functionalities of the PlantHydraulicsModel. Errors encountered during the process are counted and reported.
     *
     * \return The number of errors encountered during the self-test. Returns 0 if all tests pass successfully.
     */
    static int selfTest();

    //! Set the PlantHydraulics model coefficients for all primitives
    /**
     * \param[in] modelcoefficients Set of model coefficients.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const PlantHydraulicsModelCoefficients &modelcoefficients);

    //! Set the PlantHydraulics model coefficients for a subset of primitives based on their UUIDs
    /**
     * \param[in] modelcoefficients Set of model coefficients.
     * \param[in] UUIDs Universal unique identifiers for primitives to be set.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const PlantHydraulicsModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs);

    //! Set the PlantHydraulics model coefficients for a primitive based on its UUID and the name of the species from the library of species coefficients
    /**
     * \param[in] species String of species name in library of model coefficients.
     * \param[in] UUID Universal unique identifier for primitive to be set.
     */
    void setModelCoefficientsFromLibrary(const std::string &species, uint UUID);

    //! Set the PlantHydraulics model coefficients for all primitives
    /**
     * \param[in] species String of species name in library of model coefficients.
     */
    void setModelCoefficientsFromLibrary(const std::string &species);

    //! Set the PlantHydraulics model coefficients for a subset of primitives based on their UUIDs and the name of the species from the library of species coefficients
    /**
     * \param[in] species String of species name in library of model coefficients.
     * \param[in] UUIDs Universal unique identifiers for primitives to be set.
     */
    void setModelCoefficientsFromLibrary(const std::string &species, const std::vector<uint> &UUIDs);

    //! Get the PlantHydraulics model coefficients for a species from the library of species coefficients
    /**
     * \param[in] species String of species name in library of model coefficients.
     */
    PlantHydraulicsModelCoefficients getModelCoefficientsFromLibrary(const std::string &species);

    //! Get the PlantHydraulics model coefficients for a primitive based on its UUID
    /**
     * \param[in] UUID Universal unique identifier for primitive to be set.
     */
    PlantHydraulicsModelCoefficients getModelCoefficients(uint UUID);

    //! Check for existing primitive data, and set default value if it does not exist
    /**
     * \param[in] UUID Primitive UUID
     * \param[in] primitive_data_label String label of primitive data
     * \param[in] default_value Default value to be set if primitive data does not exist
     * \param[in] message_flag Toggle printed message warning that existing data does not exist and default value has been used
     * \return float primitive data
     * */
    float getOrInitializePrimitiveData(uint UUID, const std::string &primitive_data_label, float default_value, bool message_flag);

    //! Query the unique plantIDs of leaf primitives
    /**
     * \param[in] UUIDs List of leaf primitives who have plantID primitive data
     * \return std::vector<int> plantIDs List of unique plantIDs found in the input leaf primitive data
     * */
    std::vector<int> getUniquePlantIDs(const std::vector<uint> &UUIDs);

    //! Get the plantID primitive data of a leaf primitive
    /**
     * \param[in] UUID Leaf primitive
     * \return int plantID The primitive's plantID
     * */
    int getPlantID(uint UUID);

    //! Get the plantID primitive data of leaf primitives
    /**
     * \param[in] UUIDs Leaf primitives
     * \return int plantID The primitives shared plantID, if exists
     * */
    int getPlantID(const std::vector<uint> &UUIDs);

    //! Toggle the writing out of optional primitive data - hydraulic conductance
    /**
     * \param[in] toggle
     * */
    void outputConductancePrimitiveData(bool toggle);

    //! Toggle the writing out of optional primitive data - hydraulic capacitance
    /**
     * \param[in] toggle
     * */
    void outputCapacitancePrimitiveData(bool toggle);

    //! Group leaf primitives into a plant that share stem, root, and soil hydraulics
    /**
     * \param[in] UUIDs Leaf primitives
     * */
    void groupPrimitivesIntoPlantObject(const std::vector<uint> &UUIDs);

    //! Get the stem water potential of a plant given its plantID
    /**
     * \param[in] plantID The plant ID of a collection of primitives
     * \return float stem_water_potential The shared stem water potential of a plant
     * */
    float getStemWaterPotentialOfPlant(uint plantID);

    //! Get the stem water potential of a leaf primitive given its UUID
    /**
     * \param[in] UUID Leaf primitive UUID
     * \return float stem_water_potential The stem water potential of the plant of a leaf primitive
     * */
    float getStemWaterPotential(uint UUID);


    //! Get the stem water potential of the leaf primitives of one plant
    /**
     * \param[in] UUIDs Leaf primitive UUIDs of a plant
     * \return float stem_water_potential The shared stem water potential of a plant
     * */
    float getStemWaterPotential(const std::vector<uint> &UUIDs);

    //! Get the root water potential of a plant given its plantID
    /**
     * \param[in] plantID The plant ID of a collection of primitives
     * \return float root_water_potential The shared root water potential of a plant
     * */
    float getRootWaterPotentialOfPlant(uint plantID);

    //! Get the root water potential of a leaf primitive given its UUID
    /**
     * \param[in] UUID Leaf primitive UUID
     * \return float root_water_potential The root water potential of the plant of a leaf primitive
     * */
    float getRootWaterPotential(uint UUID);


    //! Get the root water potential of the leaf primitives of one plant
    /**
     * \param[in] UUIDs Leaf primitive UUIDs of a plant
     * \return float root_water_potential The shared root water potential of a plant
     * */
    float getRootWaterPotential(const std::vector<uint> &UUIDs);

    //! Get the soil water potential of a plant given its plantID
    /**
     * \param[in] plantID The plant ID of a collection of primitives
     * \return float soil_water_potential The shared soil water potential of a plant
     * */
    float getSoilWaterPotentialOfPlant(uint plantID);

    //! Get the soil water potential of a leaf primitive given its UUID
    /**
     * \param[in] UUID Leaf primitive UUID
     * \return float soil_water_potential The soil water potential of the plant of a leaf primitive
     * */
    float getSoilWaterPotential(uint UUID);


    //! Get the soil water potential of the leaf primitives of one plant
    /**
     * \param[in] UUIDs Leaf primitive UUIDs of a plant
     * \return float soil_water_potential The shared soil water potential of a plant
     * */
    float getSoilWaterPotential(const std::vector<uint> &UUIDs);

    //! Set the soil water potential of a plant given its plantID
    /**
     * \param[in] plantID The plant ID of a collection of primitives
     * \param[in] soil_water_potential The shared soil water potential of a plant
     * */
    void setSoilWaterPotentialOfPlant(uint plantID, float soil_water_potential);

    //! Get the primitives without plantID parent object data
    /**
     * \param[in] UUIDs Universal unique identifiers for primitives to checked for plantID parent object data
     * \return std::vector<uint> ungroupedPrimitives The UUIDs of primitives without plantID parent object data
     */
    std::vector<uint> getPrimitivesWithoutPlantID(std::vector<uint> UUIDs);

    //! Run the model for a primitive based on its UUID for a given timespan and timestep for all UUIDs with primitive data "plantID"
    /**
     * \param[in] timespan Timespan in seconds of simulation run
     * \param[in] timestep Timestep in seconds of simulation run
     * \note A timespan of "0" will enact steady-state simulation, while a timespan > 0 will run the non-steady-state simulation.
     * \note All UUIDs in context with primitive data "plantID" will be assumed to be a collection of a single plant's leaves, with one shared stem, root, and soil water potential
     */
    void run(int timespan, int timestep);

    //! Run the model for a primitive based on its UUID for a given timespan and timestep
    /**
     * \param[in] UUID Universal unique identifier for primitive to be set.
     * \param[in] timespan Timespan in seconds of simulation run
     * \param[in] timestep Timestep in seconds of simulation run
     * \note A timespan of "0" will enact steady-state simulation, while a timespan > 0 will run the non-steady-state simulation.
     * \note UUID is assumed to be a single leaf, with one stem, root, and soil water potential
     */
    void run(uint UUID, int timespan = 0, int timestep = 100);

    //! Run the model for a primitives based on their UUIDs for a given timespan and timestep
    /**
     * \param[in] UUIDs Universal unique identifiers for primitives to be run the model on.
     * \param[in] timespan Timespan in seconds of simulation run
     * \param[in] timestep Timestep in seconds of simulation run
     * \note A timespan of "0" will enact steady-state simulation, while a timespan > 0 will run the non-steady-state simulation
     * \note UUIDs with the same primitive data "plantID" will be assumed to be a collection of a single plant's leaves, with one shared stem, root, and soil water potential
     * \note All plantIDs within given UUIDs will have the model separately run on them
     */
    void run(const std::vector<uint> &UUIDs, int timespan = 0, int timestep = 100);


private:
    //! Copy of a pointer to the context
    helios::Context *context;
    PlantHydraulicsModelCoefficients modelcoeffs;
    std::map<uint, PlantHydraulicsModelCoefficients> modelcoeffs_map;
    std::map<uint, float> stemWaterPotentialsByPlantID;
    std::map<uint, float> rootWaterPotentialsByPlantID;
    std::map<uint, float> soilWaterPotentialsByPlantID;
    std::vector<uint> plantIDs;
    bool message_flag = false;

    bool OutputConductance = false;
    bool OutputCapacitance = false;

    void updateRootAndStemWaterPotentialsOfPlant(const std::vector<uint> &UUIDs, int timespan, int timestep);

    void updateLeafWaterPotentialsOfPlant(const std::vector<uint> &UUIDs, int timespan, int timestep);

    int adjustTimestep(float time_step, float min_time_step, float max_time_step, float gradient, float gradient_upper_bound);

    std::vector<std::vector<std::vector<float>>> meshgrid(const std::vector<float> &x, const std::vector<float> &y);
};

#endif
