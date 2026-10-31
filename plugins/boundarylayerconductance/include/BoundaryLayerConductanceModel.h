/** \file "BoundaryLayerConductanceModel.h" Primary header file for boundary layer conductance model.

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

#ifndef BOUNDARYLAYERCONDUCTANCEMODEL
#define BOUNDARYLAYERCONDUCTANCEMODEL

#include "Context.h"

//! Boundary-layer conductance model class
/**
 * This model computes boundary-layer conductance values for a number of objects. Each primitive gets primitive data "boundarylayer_conductance", the conductance to heat summed over its faces, and the per-face
 * conductances to heat, water vapor and CO2 "boundarylayer_conductance_front", "boundarylayer_conductance_back", "boundarylayer_conductance_vapor_front", "boundarylayer_conductance_vapor_back",
 * "boundarylayer_conductance_CO2_front" and "boundarylayer_conductance_CO2_back" (all mol/m^2/s). The front face is the one the primitive normal points out of. The "Graefe" and "InclinedPlate" models resolve differences
 * between the two faces; the other models split the summed conductance equally between them.
 */
class BLConductanceModel {
public:
    //! Constructor
    /**
     * \param[in] context Pointer to the Helios context
     */
    explicit BLConductanceModel(helios::Context *context);

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Enable standard output from this plug-in (default)
    void enableMessages();

    //! Disable standard output from this plug-in
    void disableMessages();

    //! Set the boundary-layer conductance model to be used for all primitives in the Context
    /**
     * \param[in] gH_model String corresponding to model to be run. One of "Pohlhausen", "InclinedPlate", "Sphere", "Ground", or "Graefe".
     */
    void setBoundaryLayerModel(const char *gH_model);

    //! Set the boundary-layer conductance model to be used for a set of primitives
    /**
     * \param[in] UUID  Unique universal identifier (UUID) for primitive the conductance model should be used.
     * \param[in] gH_model String corresponding to model to be run. One of "Pohlhausen", "InclinedPlate", "Sphere", "Ground", or "Graefe".
     */
    void setBoundaryLayerModel(uint UUID, const char *gH_model);

    //! Set the boundary-layer conductance model to be used for a set of primitives
    /**
     * \param[in] UUIDs  Unique universal identifiers (UUIDs) for primitives the conductance model should be used.
     * \param[in] gH_model String corresponding to model to be run. One of "Pohlhausen", "InclinedPlate", "Sphere", "Ground", or "Graefe".
     */
    void setBoundaryLayerModel(const std::vector<uint> &UUIDs, const char *gH_model);

    //! Run boundary-layer conductance calculations for all primitives in the Context
    void run();

    //! Run boundary-layer conductance calculations for a subset of primitives in the Context based on a vector of UUIDs
    void run(const std::vector<uint> &UUIDs);

private:
    //! Copy of a pointer to the context
    helios::Context *context;

    //! Default wind speed if it was not specified in the context
    float wind_speed_default;

    //! Default air temperature if it was not specified in the context
    float air_temperature_default;

    //! Default surface temperature if it was not specified in the context
    float surface_temperature_default;

    bool message_flag;

    //! Default turbulence intensity (%) for the "Graefe" model if primitive data "turbulence_intensity" was not specified, representative of the air inside and just above a plant canopy
    static constexpr float turbulence_intensity_default = 20.f;

    //! Default air pressure (Pa) for the "Graefe" model if primitive data "air_pressure" was not specified
    static constexpr float air_pressure_default = 101325.f;

    float calculateBoundaryLayerConductance(uint gH_model, float U, float L, char Nsides);

    //! Per-face boundary-layer conductances of a primitive (mol/m^2/s)
    struct FaceConductances {
        float heat_front; //!< Conductance to heat of the front face
        float heat_back; //!< Conductance to heat of the back face
        float vapor_front; //!< Conductance to water vapor of the front face
        float vapor_back; //!< Conductance to water vapor of the back face
        float CO2_front; //!< Conductance to CO2 of the front face
        float CO2_back; //!< Conductance to CO2 of the back face
    };

    //! Leaf shape used by the "Graefe" model, equivalent ellipse of a primitive or of its parent object
    struct LeafShape {
        double length; //!< Length of the equivalent ellipse (m)
        double width; //!< Width of the equivalent ellipse (m)
        helios::vec3 long_axis; //!< Unit vector along the long axis
    };

    //! Compute the per-face conductances of a primitive with the "InclinedPlate" model
    /**
     * \param[in] U Wind speed (m/s).
     * \param[in] L Characteristic length of the plate (m).
     * \param[in] TL Surface temperature (K).
     * \param[in] Ta Air temperature (K).
     * \param[in] normal Unit normal of the primitive.
     * \param[in] wind_direction Horizontal unit vector the wind blows toward, or nullptr if unknown.
     * \param[in] Nsides Number of faces exchanging heat.
     */
    FaceConductances calculateInclinedPlateConductances(float U, float L, float TL, float Ta, const helios::vec3 &normal, const helios::vec3 *wind_direction, char Nsides) const;

    //! Read the optional primitive data "wind_direction" of a primitive
    /**
     * \param[in] UUID Primitive UUID.
     * \param[out] wind_direction Horizontal unit vector the wind blows toward.
     * \return True if the primitive has a wind direction.
     */
    bool readWindDirection(uint UUID, helios::vec3 &wind_direction) const;

    //! Compute the per-face conductances of a primitive with the "Graefe" model
    /**
     * \param[in] UUID Primitive UUID.
     * \param[in] U Wind speed (m/s).
     * \param[in] TL Surface temperature (K).
     * \param[in] Ta Air temperature (K).
     * \param[in] Nsides Number of faces exchanging heat.
     * \param[in,out] leaf_shape_cache Leaf shapes of parent objects already computed in this run.
     */
    FaceConductances calculateGraefeConductances(uint UUID, float U, float TL, float Ta, char Nsides, std::map<uint, LeafShape> &leaf_shape_cache) const;

    //! Equivalent ellipse of a set of primitives, from the second moments of their area projected onto their mean plane
    /**
     * \param[in] UUIDs Primitives making up the leaf.
     * \param[in] area One-sided leaf area (m^2).
     * \param[in] UUID_for_messages Primitive being computed, for error messages.
     */
    LeafShape calculateLeafShape(const std::vector<uint> &UUIDs, float area, uint UUID_for_messages) const;

    std::map<uint, uint> boundarylayer_model;
};

#endif
