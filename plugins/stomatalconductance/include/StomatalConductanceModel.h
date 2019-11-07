/** \file "StomatalConductance.h" Primary header file for stomatalconductance plug-in.
    \author Brian Bailey

    Copyright (C) 2018  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef __STOMATALCONDUCTANCE__
#define __STOMATALCONDUCTANCE__

#include "Context.h"

//! Coefficients for simplified Buckley, Mott, & Farquahr stomatal conductance model
struct BMFcoefficients{

BMFcoefficients(void){
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

class StomatalConductanceModel{
public:

  //! Default constructor
  /** \param[in] "__context" Pointer to the helios context
   */
  StomatalConductanceModel( helios::Context* __context );

  //! Self-test (unit test) routine
  int selfTest( void );

  //! Set the model coefficient values for all primitives
  /** \param[in] "coeffs" Model coefficient values */
  void setModelCoefficients( const BMFcoefficients coeffs );

  //! Set the model coefficient values for a subset of primitives based on their UUIDs
  /** \param[in] "coeffs" Model coefficient values */
  void setModelCoefficients( const BMFcoefficients coeffs, const std::vector<uint> UUIDs );

  //! Update the stomatal conductance for all primitives in the context
  void run( void );

  //! Update the stomatal conductance only for given primitives
  /** \param[in] "UUIDs" UUID's for which stomatal conductance will be calculated 
  */
  void run( const std::vector<uint>& UUIDs );

private:

  //! Pointer to the Helios context
  helios::Context* context;

  BMFcoefficients BMFcoeffs;

  float i_default;
  float TL_default;
  float pressure_default;
  float air_temperature_default;
  float air_humidity_default;
  float bl_conductance_default;

  std::map<uint,BMFcoefficients> model_coefficients;

  
  
};

#endif
