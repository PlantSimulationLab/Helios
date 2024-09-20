/** \file "StomatalConductanceModel.h" Primary header file for stomatalconductance plug-in.

    Copyright (C) 2016-2024 Brian Bailey

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

//! Coefficients for original Ball, Woodrow, Berry (1987) stomatal conductance model
struct BWBcoefficients{

    BWBcoefficients(){
        gs0 = 0.0733;
        a1 = 9.422;
    }

    float gs0;  //mol/m^2-s
    float a1;  //unitless

};

//! Coefficients for Ball, Berry, Leuning (Leuning 1990, 1995) stomatal conductance model
struct BBLcoefficients{

    BBLcoefficients(){
        gs0 = 0.0743;
        a1 = 4.265;
        D0 = 14570.0;
    }

    float gs0;  //mol/m^2-s
    float a1;  //unitless
    float D0;  //mmol/mol

};

//! Coefficients for optimality-based Medlyn et al. (2011) stomatal conductance model
struct MOPTcoefficients{

  MOPTcoefficients(){
    gs0 = 0.0825;
    g1 = 2.637;
  }

  float gs0;  //mol/m^2/s
  float g1;  //(kPa)^0.5

};

//! Coefficients for simplified Buckley, Mott, & Farquhar stomatal conductance model
struct BMFcoefficients{

    BMFcoefficients(){
      Em = 258.25;
      i0 = 38.65;
      k = 232916.82;
      b = 609.67;
//      Em = 7.0;
//      k = 496;
//      b = 6.1;
//      i0 = 6.6;
    }

    float Em;  //mmol/m^2/s
    float i0;  //umol/m^2/s
    float k;   //umol/m^2/s mmol/mol
    float b;   //mmol/mol

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
    /**
     * \param[in] m_context Pointer to the helios context
     */
    explicit StomatalConductanceModel( helios::Context* m_context );

    //! Self-test (unit test) routine
    int selfTest();

    //! Set the model coefficient values for all primitives - Ball, Woodrow, Berry model
    /**
     * \param[in] coeffs Model coefficient values
     */
    void setModelCoefficients( const BWBcoefficients &coeffs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs - Ball, Woodrow, Berry model
    /**
     * \param[in] coeffs Model coefficient values
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setModelCoefficients( const BWBcoefficients &coeffs, const std::vector<uint> &UUIDs );

    //! Set the model coefficient values for all primitives - Ball-Berry-Leuning model
    /**
     * \param[in] coeffs Model coefficient values
     */
    void setModelCoefficients( const BBLcoefficients &coeffs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs - Ball-Berry-Leuning model
    /**
     * \param[in] coeffs Model coefficient values
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setModelCoefficients( const BBLcoefficients &coeffs, const std::vector<uint> &UUIDs );

    //! Set the model coefficient values for all primitives - Medlyn et al. optimality model
    /**
     * \param[in] coeffs Model coefficient values
     */
    void setModelCoefficients( const MOPTcoefficients &coeffs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs - Medlyn et al. optimality model
    /**
     * \param[in] coeffs Model coefficient values
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setModelCoefficients( const MOPTcoefficients &coeffs, const std::vector<uint> &UUIDs );

    //! Set the model coefficient values for all primitives - Buckley, Mott, Farquhar model
    /**
     * \param[in] coeffs Model coefficient values
     */
    void setModelCoefficients( const BMFcoefficients &coeffs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs - Buckley, Mott, Farquhar model
    /**
     * \param[in] coeffs Model coefficient values
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setModelCoefficients( const BMFcoefficients &coeffs, const std::vector<uint> &UUIDs );

    //! Set the model coefficient values for all primitives
    /**
     * \param[in] coeffs Model coefficient values
     */
    void setModelCoefficients( const BBcoefficients &coeffs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs
    /**
     * \param[in] coeffs Model coefficient values
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setModelCoefficients( const BBcoefficients &coeffs, const std::vector<uint> &UUIDs );

    //! Set the model coefficient values for a subset of primitives based on their UUIDs
    /**
     * \param[in] coeffs Vecotr of sets of model coefficient values
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setModelCoefficients(const std::vector<BMFcoefficients> &coeffs, const std::vector<uint> &UUIDs);

    //! Set the model coefficient values using one of the available species in the library
    /**
     * \param[in] species Name of species
     */
    void setBMFCoefficientsFromLibrary(const std::string &species);

    //! Set the model coefficient values using one of the available species in the library for a subset of primitives based on their UUIDs
    /**
     * \param[in] species Name of species
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setBMFCoefficientsFromLibrary(const std::string &species, const std::vector<uint> &UUIDs);

    //! Get the model coefficient values using one of the available species in the library
    /**
     * \param[in] species Name of species
     */
    BMFcoefficients getBMFCoefficientsFromLibrary(const std::string &species);

    //! Set time constants for dynamic stomatal opening and closing for all primitives
    /**
     * \param[in] tau_open Time constant (seconds) for dynamic stomatal opening
     * \param[in] tau_close Time constant (seconds) for dynamic stomatal closing
     */
    void setDynamicTimeConstants( float tau_open, float tau_close );

    //! Set time constants for dynamic stomatal opening and closing for a subset of primitives based on their UUIDs
    /**
     * \param[in] tau_open Time constant (seconds) for dynamic stomatal opening
     * \param[in] tau_close Time constant (seconds) for dynamic stomatal closing
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void setDynamicTimeConstants( float tau_open, float tau_close, const std::vector<uint> &UUIDs );

    //! Update the stomatal conductance for all primitives in the context
    void run();

    //! Update the stomatal conductance for a subset of primitives in the context
    /**
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void run( const std::vector<uint> &UUIDs );

    //! Update the stomatal conductance for all primitives in the context based on the dynamic stomatal model version
    /**
     * \param[in] dt Time step to advance stomatal conductance (seconds)
     */
    void run( float dt );

    //! Update the stomatal conductance for a subset of primitives in the context
    /**
     * \param[in] dt Time step to advance stomatal conductance (seconds)
     * \param[in] UUIDs Universal unique identifiers for primitives to set
     */
    void run( const std::vector<uint> &UUIDs, float dt );

    //! Add optional output primitive data values to the Context
    /**
        * \param[in] label Name of primitive data (e.g., Ci)
     */
    void optionalOutputPrimitiveData( const char* label );

    //! Print a report detailing usage of default input values for all primitives in the Context
    void printDefaultValueReport() const;

    //! Print a report detailing usage of default input values based on a subset of primitive UUIDs
    /**
     * \param[in] UUIDs Universal unique identifiers for report
     */
    void printDefaultValueReport(const std::vector<uint> &UUIDs) const;

private:

    //! Pointer to the Helios context
    helios::Context* context;

    std::string model;
    BWBcoefficients BWBcoeffs;
    BBLcoefficients BBLcoeffs;
    MOPTcoefficients MOPTcoeffs;
    BMFcoefficients BMFcoeffs;
    BBcoefficients BBcoeffs;

    float i_default;
    float TL_default;
    float pressure_default;
    float air_temperature_default;
    float air_humidity_default;
    float xylem_potential_default;
    float An_default;
    float Gamma_default;
    float air_CO2_default;
    float blconductance_default;
    float beta_default;

    std::map<uint,BWBcoefficients> BWBmodel_coefficients;
    std::map<uint,BBLcoefficients> BBLmodel_coefficients;
    std::map<uint,MOPTcoefficients> MOPTmodel_coefficients;
    std::map<uint,BMFcoefficients> BMFmodel_coefficients;
    std::map<uint,BBcoefficients> BBmodel_coefficients;

    std::map<uint,helios::vec2> dynamic_time_constants; // .x is tau_open, .y is tau_close

    static float evaluate_BWBmodel( float esurf, std::vector<float> &variables, const void* parameters );
    static float evaluate_BBLmodel( float esurf, std::vector<float> &variables, const void* parameters );
    static float evaluate_MOPTmodel( float esurf, std::vector<float> &variables, const void* parameters );
    static float evaluate_BMFmodel( float esurf, std::vector<float> &variables, const void* parameters );
    static float evaluate_BBmodel( float gs, std::vector<float> &variables, const void* parameters );


    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;

};

#endif
