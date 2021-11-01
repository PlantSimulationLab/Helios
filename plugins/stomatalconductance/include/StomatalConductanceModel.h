/** \file "StomatalConductance.h" Primary header file for stomatalconductance plug-in.
    \author Brian Bailey

    Copyright (C) 2016-2021  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef STOMATAL_CONDUCTANCE
#define STOMATAL_CONDUCTANCE

#include "Context.h"

//! Coefficients for simplified Buckley, Mott, & Farquhar stomatal conductance model
struct BMFcoefficients{

    BMFcoefficients(){
        Em = 7.0;
        k = 496;
        b = 6.1;
        i0 = 6.6;
    }

    float Em;
    float i0;
    float k;
    float b;

};


//! Coefficients for simplified Bailey stomatal conductance model
struct BBcoefficients{

    BBcoefficients(){
        pi_0 = 1.0;
        pi_m = 1.67;
        theta = 211.22;
        sigma = 0.4408;
        chi = 2.076;
    }

    float pi_0; //MPa
    float pi_m; //MPa
    float theta; //umol/m^2-s
    float sigma; //unitless
    float chi ; //mol/m^2-s-MPa

};

class StomatalConductanceModel{
public:

    //! Default constructor
    /** \param[in] "__context" Pointer to the helios context
     */
    explicit StomatalConductanceModel( helios::Context* m_context );

    //! Self-test (unit test) routine
    static int selfTest();

    //! Set the model coefficient values for all primitives
    /** \param[in] "coeffs" Model coefficient values */
    void setModelCoefficients( const BMFcoefficients &coeffs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs
    /** \param[in] "coeffs" Model coefficient values */
    void setModelCoefficients( const BMFcoefficients &coeffs, const std::vector<uint> &UUIDs );

    //! Set the model coefficient values for all primitives
    /** \param[in] "coeffs" Model coefficient values */
    void setModelCoefficients( const BBcoefficients &coeffs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs
    /** \param[in] "coeffs" Model coefficient values */
    void setModelCoefficients( const BBcoefficients &coeffs, const std::vector<uint> &UUIDs );

    //! Update the stomatal conductance for all primitives in the context
    void run();

    //! Update the stomatal conductance for a subset of primitives in the context
    void run( const std::vector<uint> &UUIDs );

private:

    //! Pointer to the Helios context
    helios::Context* context;

    std::string model;
    BMFcoefficients BMFcoeffs;
    BBcoefficients BBcoeffs;

    float i_default;
    float TL_default;
    float pressure_default;
    float air_temperature_default;
    float air_humidity_default;
    float xylem_potential_default;

    std::map<uint,BMFcoefficients> BMFmodel_coefficients;
    std::map<uint,BBcoefficients> BBmodel_coefficients;

    static float evaluate_BBmodel( float gs, std::vector<float> &variables, const void* parameters );

};

#endif
