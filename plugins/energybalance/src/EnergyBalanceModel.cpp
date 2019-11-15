/** \file "EnergyBalanceModel.cpp" Energy balance model plugin declarations. 
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

#include "EnergyBalanceModel.h"

using namespace helios;

EnergyBalanceModel::EnergyBalanceModel( helios::Context* __context ){

  //All default values set here

  temperature_default = 300; // Kelvin

  wind_speed_default = 1.f; // m/s

  air_temperature_default = 300;  // Kelvin

  air_humidity_default = 0.5; // %

  pressure_default = 101000; // Pa

  gS_default = 0.;  // mol/m^2-s

  Qother_default = 0; //W/m^2

  objectdensity_default=1; //kg/m^2

  dt_default=0; //sec

  message_flag = true; //print messages to screen by default

  //Copy pointer to the context
  context = __context; 

}

int EnergyBalanceModel::selfTest( void ){

  std::cout << "Running energy balance model self-test..." << std::flush;

  float err_tol = 1e-3;
  
  //---- Equilibrium test -----//
  //If we set the absorbed radiation flux to 5.67e-8*T^4 and have a zero moisture conductance, we should get a surface temperature of exactly T.
  
  Context context_test;

  std::vector<uint> UUIDs;
  
  UUIDs.push_back( context_test.addPatch( make_vec3(1,2,3), make_vec2(3,2) ) );
  UUIDs.push_back( context_test.addTriangle( make_vec3(4,5,6), make_vec3(5,5,6), make_vec3(5,6,6) ) );

  float Tref = 350;

  context_test.setPrimitiveData( UUIDs, "radiation_flux_LW", float(2.f*5.67e-8*pow(Tref,4)) );

  context_test.setPrimitiveData( UUIDs, "air_temperature", Tref );

  EnergyBalanceModel energymodeltest(&context_test);
  energymodeltest.disableMessages();

  energymodeltest.addRadiationBand("LW");

  energymodeltest.run();

  for( int p=0; p<UUIDs.size(); p++ ){
    float T;
    context_test.getPrimitiveData( UUIDs.at(p), "temperature", T );
    if( fabs(T-Tref)>err_tol ){
      std::cout << "failed equilibrium test." << std::endl;
      std::cout << T << std::endl;
      return 1;
    }
  }

  //---- Energy Budget Closure -----//
  //The energy balance terms output from the model should sum to 1
  
  Context context_2;

  std::vector<uint> UUIDs_2;
  
  UUIDs_2.push_back( context_2.addPatch( make_vec3(1,2,3), make_vec2(3,2) ) );
  UUIDs_2.push_back( context_2.addTriangle( make_vec3(4,5,6), make_vec3(5,5,6), make_vec3(5,6,6) ) );

  float T = 300;

  context_2.setPrimitiveData( UUIDs_2, "radiation_flux_LW", float(2.f*5.67e-8*pow(T,4)) );
  context_2.setPrimitiveData( UUIDs_2, "radiation_flux_SW", float(300.f) );

  context_2.setPrimitiveData( UUIDs_2, "air_temperature", T );

  EnergyBalanceModel energymodel_2(&context_2);
  energymodel_2.disableMessages();
  
  energymodel_2.addRadiationBand("LW");
  energymodel_2.addRadiationBand("SW");

  energymodel_2.run();

  for( int p=0; p<UUIDs_2.size(); p++ ){

    float sensible_flux, latent_flux, R, temperature;

    float Rin = 0;
    
    context_2.getPrimitiveData( UUIDs_2.at(p), "sensible_flux", sensible_flux );
    context_2.getPrimitiveData( UUIDs_2.at(p), "latent_flux", latent_flux );
    context_2.getPrimitiveData( UUIDs_2.at(p), "radiation_flux_LW", R );
    Rin += R;
    context_2.getPrimitiveData( UUIDs_2.at(p), "radiation_flux_SW", R );
    Rin += R;
    context_2.getPrimitiveData( UUIDs_2.at(p), "temperature", temperature );
    
    float Rout = 2.f*5.67e-8*pow(temperature,4);
    float resid = Rin - Rout - sensible_flux - latent_flux;
    if( fabs(resid)>err_tol ){
      std::cout << "failed energy budget closure test." << std::endl;
      return 1;
    }
  }

  //---- Temperature Solution Check -----//
  //Check that the temperature solution yields the correct value for a known configuration

  Context context_3;

  uint UUID_3;
  
  UUID_3 = context_3.addPatch( make_vec3(1,2,3), make_vec2(3,2) );

  T = 312.f;

  context_3.setPrimitiveData( UUID_3, "radiation_flux_LW", float(5.67e-8*pow(T,4)) );
  context_3.setPrimitiveData( UUID_3, "radiation_flux_SW", float(350.f) );
  context_3.setPrimitiveData( UUID_3, "wind_speed", float(1.244f) );
  context_3.setPrimitiveData( UUID_3, "moisture_conductance", float(0.05f) );
  context_3.setPrimitiveData( UUID_3, "air_humidity", float(0.4f) );
  context_3.setPrimitiveData( UUID_3, "air_pressure", float(956789) );
  context_3.setPrimitiveData( UUID_3, "other_surface_flux", float(150.f) );
  context_3.setPrimitiveData( UUID_3, "air_temperature", T );

  context_3.setPrimitiveData( UUID_3, "twosided_flag", uint(0) );

  EnergyBalanceModel energymodel_3(&context_3);
  energymodel_3.disableMessages();
  
  energymodel_3.addRadiationBand("LW");
  energymodel_3.addRadiationBand("SW");

  energymodel_3.run();

  float sensible_flux, latent_flux, R, temperature;

  float sensible_flux_exact = 48.8894;
  float latent_flux_exact = 21.0678;
  float temperature_exact = 329.3733;

  context_3.getPrimitiveData( UUID_3, "sensible_flux", sensible_flux );
  context_3.getPrimitiveData( UUID_3, "latent_flux", latent_flux );
  context_3.getPrimitiveData( UUID_3, "temperature", temperature );

  if( fabs(sensible_flux-sensible_flux_exact)/fabs(sensible_flux_exact)>err_tol || fabs(latent_flux-latent_flux_exact)/fabs(latent_flux_exact)>err_tol || fabs(temperature-temperature_exact)/fabs(temperature_exact)>err_tol ){
    std::cout << "failed temperature solver check #1." << std::endl;
    std::cout << temperature << std::endl;
    return 1;
  }

  //use object length rather than sqrt(area)

  context_3.setPrimitiveData( UUID_3, "object_length", float(0.374f) );

  energymodel_3.run();

  context_3.getPrimitiveData( UUID_3, "sensible_flux", sensible_flux );
  context_3.getPrimitiveData( UUID_3, "latent_flux", latent_flux );
  context_3.getPrimitiveData( UUID_3, "temperature", temperature );
  
  sensible_flux_exact = 89.3802;
  latent_flux_exact = 19.8921;
  temperature_exact = 324.4110;
      
  context_3.getPrimitiveData( UUID_3, "sensible_flux", sensible_flux );
  context_3.getPrimitiveData( UUID_3, "latent_flux", latent_flux );
  context_3.getPrimitiveData( UUID_3, "temperature", temperature );

  if( fabs(sensible_flux-sensible_flux_exact)/fabs(sensible_flux_exact)>err_tol || fabs(latent_flux-latent_flux_exact)/fabs(latent_flux_exact)>err_tol || fabs(temperature-temperature_exact)/fabs(temperature_exact)>err_tol ){
    std::cout << "failed temperature solver check #2." << std::endl;
    return 1;
  }
  
  //manually set boundary-layer conductance
  
  context_3.setPrimitiveData( UUID_3, "boundarylayer_conductance", float(0.134f) );

  energymodel_3.run();

  context_3.getPrimitiveData( UUID_3, "sensible_flux", sensible_flux );
  context_3.getPrimitiveData( UUID_3, "latent_flux", latent_flux );
  context_3.getPrimitiveData( UUID_3, "temperature", temperature );
  
  sensible_flux_exact = 61.7366f;
  latent_flux_exact = 21.2700f;
  temperature_exact = 327.7511f;
      
  context_3.getPrimitiveData( UUID_3, "sensible_flux", sensible_flux );
  context_3.getPrimitiveData( UUID_3, "latent_flux", latent_flux );
  context_3.getPrimitiveData( UUID_3, "temperature", temperature );

  if( fabs(sensible_flux-sensible_flux_exact)/fabs(sensible_flux_exact)>err_tol || fabs(latent_flux-latent_flux_exact)/fabs(latent_flux_exact)>err_tol || fabs(temperature-temperature_exact)/fabs(temperature_exact)>err_tol ){
    std::cout << "failed temperature solver check #3." << std::endl;
    return 1;
  }

  //---- Optional Primitive Data Output Check -----//

  Context context_4;

  uint UUID_4;
  
  UUID_4 = context_4.addPatch( make_vec3(1,2,3), make_vec2(3,2) );

  EnergyBalanceModel energymodel_4(&context_4);
  energymodel_4.disableMessages();
  
  energymodel_4.addRadiationBand("LW");

  energymodel_4.optionalOutputPrimitiveData( "boundarylayer_conductance_out" );
  energymodel_4.optionalOutputPrimitiveData( "vapor_pressure_deficit" );

  context_4.setPrimitiveData( UUID_4, "radiation_flux_LW", 0.f );
  
  energymodel_4.run();

  if( !context_4.doesPrimitiveDataExist( UUID_4, "vapor_pressure_deficit" ) || !context_4.doesPrimitiveDataExist( UUID_4, "boundarylayer_conductance_out" ) ){
    std::cout << "failed optional primitive data output check #4." << std::endl;
    return 1;
  }

  //---- Dynamic Model Check -----//

  Context context_5;

  float dt_5 = 1.f;      //sec
  float T_5 = 3600;    //sec
  float To_5 = 300.f;  // K
  float rho_5 = 0.5;   // kg/m^2

  float Rlow = 50.f;
  float Rhigh = 500.f;
  
  uint UUID_5 = context_5.addPatch( make_vec3(0,0,0), make_vec2(1,1) );

  context_5.setPrimitiveData( UUID_5, "radiation_flux_SW", Rlow );
  context_5.setPrimitiveData( UUID_5, "temperature", To_5 );
  context_5.setPrimitiveData( UUID_5, "object_density", rho_5 );
  context_5.setPrimitiveData( UUID_5, "twosided_flag", uint(0) );
  context_5.setPrimitiveData( UUID_5, "emissivity_SW", 0.f );

  EnergyBalanceModel energybalance_5(&context_5);
  energybalance_5.disableMessages();

  energybalance_5.addRadiationBand("SW");

  energybalance_5.optionalOutputPrimitiveData( "boundarylayer_conductance_out" );

  std::vector<float> temperature_dyn;

  int N = round(T_5/dt_5);
  for( int t=0; t<N; t++ ){

    if( t>0.5f*N ){
      context_5.setPrimitiveData( UUID_5, "radiation_flux_SW", Rhigh );
    }

    energybalance_5.run( dt_5 );

    float temp;
    context_5.getPrimitiveData( UUID_5, "temperature", temp );
    temperature_dyn.push_back(temp);
      
  }

  float gH_5;
  context_5.getPrimitiveData( UUID_5, "boundarylayer_conductance_out", gH_5 );

  float tau_5 = rho_5*4190.f/gH_5/29.25f;

  context_5.setPrimitiveData( UUID_5, "radiation_flux_SW", Rlow );
  energybalance_5.run();
  float Tlow;
  context_5.getPrimitiveData( UUID_5, "temperature", Tlow );

  context_5.setPrimitiveData( UUID_5, "radiation_flux_SW", Rhigh );
  energybalance_5.run();
  float Thigh;
  context_5.getPrimitiveData( UUID_5, "temperature", Thigh );

  float err=0;
  for( int t=round(0.5f*N); t<N; t++ ){

    float time = dt_5*(t-round(0.5f*N));

    float temperature_ref = Tlow+(Thigh-Tlow)*(1.f-exp(-time/tau_5));

    err += pow( temperature_ref-temperature_dyn.at(t), 2 );

  }

  err = sqrt(err/float(N));

  if( err>0.2f ){
    std::cout << "failed dynamic energy balance check #5." << std::endl;
    return 1;
  }
  
  
  std::cout << "passed." << std::endl;
  return 0;

}

void EnergyBalanceModel::enableMessages(void){
  message_flag = true;
}

void EnergyBalanceModel::disableMessages(void){
  message_flag = false;
}

void EnergyBalanceModel::addRadiationBand( const char* band ){
  radiation_bands.push_back(band);
}

void EnergyBalanceModel::optionalOutputPrimitiveData( const char* label ){

  if( strcmp(label,"boundarylayer_conductance_out")==0 || strcmp(label,"vapor_pressure_deficit")==0 ){
    output_prim_data.push_back( label );
  }else{
    std::cout << "WARNING (EnergyBalanceModel::optionalOutputPrimitiveData): unknown output primitive data " << label << std::endl;
  }
  
}
