/** \file "PhotosynthesisModel.h" Primary header file for photosynthesis plug-in.
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

#ifndef PHOTOSYNTHESIS_MODEL
#define PHOTOSYNTHESIS_MODEL

#include "Context.h"

struct EmpiricalModelCoefficients{

  EmpiricalModelCoefficients(){
    Tref = 298; //K
    Ci_ref = 290;  //umol CO2/mol air
    Asat = 18.18;  //umol/m^2-s
    theta = 62.03; //W/m^2
    Tmin = 290; //K
    Topt = 303; //K
    q = 0.344; //unitless
    R = 1.663e5; //umol-K^0.5/m^2-s
    ER = 3740; //1/K
    kC = 0.791; //unitless
  }

  //reference values
  float Tref;
  float Ci_ref;

  //light response coefficients
  float Asat;
  float theta;

  //assimilation temperature response coefficients
  float Tmin;
  float Topt;
  float q;

  //respiration temperature response coefficients
  float R;
  float ER;

  //CO2 response coefficients
  float kC;

};

struct FarquharModelCoefficients{

  FarquharModelCoefficients(){
    
    //parameters (at TL = 25C)
    Vcmax = 78.5; //umol/m^2/s
    Jmax = 150; //umol/m^2/s
    alpha = 0.45; //unitless
    Rd = 2.12; //umol/m^2/s

    O = 213.5; //ambient oxygen concentration (mmol/mol)

    //tempeature parameters
    c_Rd = 18.72;
    c_Vcmax = 26.35;
    c_Jmax = 17.57;
    c_Gamma = 19.02;
    c_Kc = 38.05;
    c_Ko = 20.30;

    dH_Rd = 46.39;
    dH_Vcmax = 65.33;
    dH_Jmax = 43.54;
    dH_Gamma = 37.83;
    dH_Kc = 79.43;
    dH_Ko = 36.38;

  }

  //parameters
  float Vcmax;
  float Jmax; 
  float Rd;
  float alpha;
  float O;
  
  //temperature parameters
  float c_Rd;
  float c_Vcmax;
  float c_Jmax;
  float c_Gamma;
  float c_Kc;
  float c_Ko;

  float dH_Rd;
  float dH_Vcmax;
  float dH_Jmax;
  float dH_Gamma;
  float dH_Kc;
  float dH_Ko;
  
};

class PhotosynthesisModel{
public:

  //! Default constructor
  /** \param[in] "__context" Pointer to the helios context
   */
  explicit PhotosynthesisModel( helios::Context* a_context );

  int selfTest();

  //! Sets photosynthesis to be calculated according to the empirical model
  void setModelType_Empirical();

  //! Sets photosynthesis to be calculated according to the Farquhar-von Caemmerer-Berry model
  void setModelType_Farquhar();

  //! Set the empricial model coefficients
  /** \note You must also call \ref setModelType_Empirical() in order to ensure that the emprical model is being used */
  void setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients );

  //! Set the Farquhar-von Caemmerer-Berry model coefficients
  /** \note You must also call \ref setModelType_Farquhar() in order to ensure that the emprical model is being used */
  void setModelCoefficients(const FarquharModelCoefficients &modelcoefficients );

  //! Run the model for all UUIDs in the Context
  void run();

  //! Run the model for a select sub-set of UUIDs
  void run(const std::vector<uint> &lUUIDs );

  //! Get the current model coefficients for the empirical model
  EmpiricalModelCoefficients getEmpiricalModelCoefficients();

  //! Get the current model coefficients for the Farquhar-von Caemmerer-Berry model
  FarquharModelCoefficients getFarquharModelCoefficients();

  //! Add optional output primitive data values to the Context
  /** \param[in] "label" Name of primitive data (e.g., Ci)
  */
  void optionalOutputPrimitiveData( const char* label );

private:

  //! Pointer to the Helios context
  helios::Context* context;

  EmpiricalModelCoefficients empiricalmodelcoeffs;
  FarquharModelCoefficients farquharmodelcoeffs;

  float evaluateEmpiricalModel( float i_PAR, float TL, float CO2, float gM );

  float evaluateFarquharModel( float i_PAR, float TL, float CO2, float gM, float& Ci, float& Gamma, int& limitation_state );

  float evaluateCi_Empirical( float Ci, float CO2, float fL, float Rd, float gM ) const;

  //float evaluateCi_Farquhar( const float Ci, const float CO2, const float i_PAR, const float TL, const float gM, float& A, int& limitation_state ) const;
  static float evaluateCi_Farquhar( float Ci, std::vector<float> &variables, const void *parameters );

  float i_PAR_default;
  float TL_default;
  float CO2_default;
  float gM_default;

  int model_flag;

  //! Names of additional primitive data to add to the Context
  std::vector<std::string> output_prim_data;
  
};

#endif
