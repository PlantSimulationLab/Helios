/** \file "StomatalConductanceModel.cpp" Primary source file for stomatalconductance plug-in.
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

#include "StomatalConductanceModel.h"

using namespace std;
using namespace helios;

StomatalConductanceModel::StomatalConductanceModel( helios::Context* __context ){
  context = __context;

  //default values set here

  i_default = 0; //W/m^2
  
  TL_default = 300; // Kelvin

  air_temperature_default = 300;  // Kelvin

  air_humidity_default = 0.5; // %

  pressure_default = 101000; // Pa

  bl_conductance_default = 100; //mol/m^2-s
  
}

int StomatalConductanceModel::selfTest( void ){
  return 0;
}

void StomatalConductanceModel::setModelCoefficients( const BMFcoefficients coeffs ){
  BMFcoeffs = coeffs;
  model_coefficients.clear();
}

void StomatalConductanceModel::setModelCoefficients( const BMFcoefficients coeffs, const std::vector<uint> UUIDs ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    model_coefficients.at(UUIDs.at(p)) = coeffs;
  }
}

void StomatalConductanceModel::run( void ){
  run( context->getAllUUIDs() );
}

void StomatalConductanceModel::run( const std::vector<uint>& UUIDs ){

  size_t assumed_default_i = 0;
  size_t assumed_default_TL = 0;
  size_t assumed_default_p = 0;
  size_t assumed_default_Ta = 0;
  size_t assumed_default_rh = 0;
  size_t assumed_default_gH = 0;
  
  for( size_t j=0; j<UUIDs.size(); j++ ){

    size_t p = UUIDs.at(j);

    if( !context->doesPrimitiveExist(p) ){
      std::cout << "WARNING (StomatalConductance::run): primitive " << p << " does not exist in the Context." << std::endl;
      continue;
    }

    //PAR radiation flux
    float i = i_default;
    if( context->doesPrimitiveDataExist(p,"radiation_flux_PAR") ){
      context->getPrimitiveData(p,"radiation_flux_PAR",i); //W/m^2
      //i = i/0.3f; //umol/m^2-s
      i = i*4.57; //umol/m^2-s (ref https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf)
    }else{
      assumed_default_i++;
    }

    //surface temperature
    float TL = TL_default;
    if( context->doesPrimitiveDataExist(p,"temperature") ){
      context->getPrimitiveData(p,"temperature",TL); //Kelvin
    }else{
      assumed_default_TL++;
    }

    //air pressure
    float press = pressure_default;
    if( context->doesPrimitiveDataExist(p,"air_pressure") ){
      context->getPrimitiveData(p,"air_pressure",press); //Pa
    }else{
      assumed_default_p++;
    }

    //air temperature
    float Ta = air_temperature_default;
    if( context->doesPrimitiveDataExist(p,"air_temperature") ){
      context->getPrimitiveData(p,"air_temperature",Ta); //Kelvin
    }else{
      assumed_default_Ta++;
    }

    //air humidity
    float rh = air_humidity_default;
    if( context->doesPrimitiveDataExist(p,"air_humidity") ){
      context->getPrimitiveData(p,"air_humidity",rh);
    }else{
      assumed_default_rh++;
    }

    //boundary-layer conductance
    // float gH = bl_conductance_default;
    // if( context->doesPrimitiveDataExist(p,"boundarylayer_conductance") ){
    //   context->getPrimitiveData(p,"boundarylayer_conductance",gH);
    // }else{
    //   assumed_default_gH++;
    // }

    //model coefficients
    float Em;
    float i0;
    float k;
    float b;
    if( model_coefficients.find(p) == model_coefficients.end() ){
      Em = BMFcoeffs.Em;
      i0 = BMFcoeffs.i0;
      k = BMFcoeffs.k;
      b = BMFcoeffs.b;
    }else{
      BMFcoefficients coeffs = model_coefficients.at(p);
      Em = coeffs.Em;
      i0 = coeffs.i0;
      k = coeffs.k;
      b = coeffs.b;
    }

    //calculate VPD
    float esat = 611.f*exp(17.502f*(Ta-273.f)/((Ta-273.f)+240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in degC, and result is in Pascals
    float ea = rh*esat; // Definition of vapor pressure (see Campbell and Norman pp. 42 Eq. 3.11)
    float es = 611.f*exp(17.502f*(TL-273.f)/((TL-273.f)+240.97f));
    float D = max(0.f,(es-ea)/press*1000);

    float gs = Em*(i+i0)/(k+b*i+(i+i0)*D);

    context->setPrimitiveData(p,"moisture_conductance",HELIOS_TYPE_FLOAT,1,&gs);

  }

}
