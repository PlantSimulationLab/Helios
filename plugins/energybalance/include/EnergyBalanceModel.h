/** \file "EnergyBalanceModel.h" Primary header file for energy balance model.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef ENERGYBALANCEMODEL
#define ENERGYBALANCEMODEL

#include "Context.h"

//! Energy balance model class
/** This model computes surface temperatures based on a local energy balance */
class EnergyBalanceModel{
public:

    //! Constructor
    /**
     * \param[in] "context" Pointer to the Helios context
    */
    EnergyBalanceModel( helios::Context* context );

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
    */
    int selfTest();

    //! Enable standard output from this plug-in (default)
    void enableMessages();

    //! Disable standard output from this plug-in
    void disableMessages();

    //! Function to run the energy balance model for all primitives in the Context
    void run() ;

    //! Function to run the dynamic /non steady state energy balance model for one timestep of length "dt" seconds
    /**
     * \param[in] "dt" Time step in seconds.
    */
    void run( float dt ) ;

    //! Function to run the energy balance model for a select set of primitives
    /**
     * \param[in] "UUIDs" Unique universal identifiers (UUIDs) for primitives that should be included in energy balance calculations. All other primitives will be skipped by the model.
    */
    void run( const std::vector<uint> &UUIDs );

    //! Function to run the energy balance model for a select set of primitives for one timestep of length "dt" seconds
    /**
     * \param[in] "UUIDs"  Unique universal identifiers (UUIDs) for primitives that should be included in energy balance calculations. All other primitives will be skipped by the model.
     * \param[in] "dt" Time step in seconds.
    */
    void run( const std::vector<uint> &UUIDs, float dt );

    //! Add the label of a radiation band in the RadiationModel plug-in that should be used in calculation of the absorbed all-wave radiation flux
    /**
     * \param[in] "band" Name of radiation band (e.g., PAR, NIR, LW, etc.)
     */
    void addRadiationBand( const char* band );

    //! Add optional output primitive data values to the Context
    /**
     * \param[in] "label" Name of primitive data (e.g., vapor_pressure_deficit)
    */
    void optionalOutputPrimitiveData( const char* label );

    //! Print a report detailing usage of default input values for all primitives in the Context
    void printDefaultValueReport() const;

    //! Print a report detailing usage of default input values based on a subset of primitive UUIDs
    /**
     * \params[in] "UUIDs" Universal unique identifiers for report
     */
    void printDefaultValueReport(const std::vector<uint> &UUIDs) const;

private:

    //! Copy of a pointer to the context
    helios::Context* context;

    //! Default surface temperature if it was not specified in the context
    float temperature_default;

    //! Default wind speed if it was not specified in the context
    float wind_speed_default;

    //! Default air temperature if it was not specified in the context
    float air_temperature_default;

    //! Default air relative humidity if it was not specified in the context
    float air_humidity_default;

    //! Default air pressure if it was not specified in the context
    float pressure_default;

    //! Default stomatal conductance if it was not specified in the context
    float gS_default;

    //! Default heat capacity if it was not specified in the context
    float heatcapacity_default;

    //! Default "other" flux if it was not specified in the context
    float Qother_default;

    //! Default surface humidity if it was not specified in the context
    float surface_humidity_default;

    bool message_flag;

    //! Names of radiation bands to be included in absorbed all-wave radiation flux
    std::vector<std::string> radiation_bands;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;

};

#endif
