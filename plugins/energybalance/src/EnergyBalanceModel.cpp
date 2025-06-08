/** \file "EnergyBalanceModel.cpp" Energy balance model plugin declarations.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "EnergyBalanceModel.h"

#include "global.h"

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

  heatcapacity_default = 0; //J/m^2-oC

  surface_humidity_default = 1; //(unitless)

  air_energy_balance_enabled = false;

  message_flag = true; //print messages to screen by default

  //Copy pointer to the context
  context = __context; 

}

int EnergyBalanceModel::selfTest(){

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

  float sensible_flux_exact = 48.7098;
  float latent_flux_exact = 21.7644;
  float temperature_exact = 329.309;

  context_3.getPrimitiveData( UUID_3, "sensible_flux", sensible_flux );
  context_3.getPrimitiveData( UUID_3, "latent_flux", latent_flux );
  context_3.getPrimitiveData( UUID_3, "temperature", temperature );

  if( fabs(sensible_flux-sensible_flux_exact)/fabs(sensible_flux_exact)>err_tol || fabs(latent_flux-latent_flux_exact)/fabs(latent_flux_exact)>err_tol || fabs(temperature-temperature_exact)/fabs(temperature_exact)>err_tol ){
    std::cout << "failed temperature solver check #1." << std::endl;
    return 1;
  }

  //use object length rather than sqrt(area)

  context_3.setPrimitiveData( UUID_3, "object_length", float(0.374f) );

  energymodel_3.run();

  context_3.getPrimitiveData( UUID_3, "sensible_flux", sensible_flux );
  context_3.getPrimitiveData( UUID_3, "latent_flux", latent_flux );
  context_3.getPrimitiveData( UUID_3, "temperature", temperature );
  
  sensible_flux_exact = 89.2217;
  latent_flux_exact = 20.2213;
  temperature_exact = 324.389;
      
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
  
  sensible_flux_exact = 61.5525f;
  latent_flux_exact = 21.8290f;
  temperature_exact = 327.704f;
      
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
  float cp_5 = 2000;   // kg/m^2

  float Rlow = 50.f;
  float Rhigh = 500.f;
  
  uint UUID_5 = context_5.addPatch( make_vec3(0,0,0), make_vec2(1,1) );

  context_5.setPrimitiveData( UUID_5, "radiation_flux_SW", Rlow );
  context_5.setPrimitiveData( UUID_5, "temperature", To_5 );
  context_5.setPrimitiveData( UUID_5, "heat_capacity", cp_5 );
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

  float tau_5 = cp_5/gH_5/cp_air_mol;

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

void EnergyBalanceModel::run(){
    run( context->getAllUUIDs() );
}

void EnergyBalanceModel::run( float dt ){
    run( context->getAllUUIDs(), dt );
}

void EnergyBalanceModel::run( const std::vector<uint> &UUIDs ){
    run( UUIDs, 0.f);
}


void EnergyBalanceModel::run( const std::vector<uint> &UUIDs, float dt ){

    if( message_flag ){
        std::cout << "Running energy balance model..." << std::flush;
    }

    // Check that some primitives exist in the context
    if( UUIDs.empty() ){
        std::cerr << "WARNING (EnergyBalanceModel::run): No primitives have been added to the context.  There is nothing to simulate. Exiting..." << std::endl;
        return;
    }

    evaluateSurfaceEnergyBalance( UUIDs, dt );

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}

void EnergyBalanceModel::evaluateAirEnergyBalance(float dt_sec, float time_advance_sec) {
    evaluateAirEnergyBalance( context->getAllUUIDs(), dt_sec, time_advance_sec );
}

void EnergyBalanceModel::evaluateAirEnergyBalance(const std::vector<uint> &UUIDs, float dt_sec, float time_advance_sec) {

    if (dt_sec <= 0) {
        std::cerr << "ERROR (EnergyBalanceModel::evaluateAirEnergyBalance): dt_sec must be greater than zero to run the air energy balance.  Skipping..." << std::endl;
        return;
    }

    float air_temperature_reference = air_temperature_default; // Default air temperature in Kelvin
    if ( context->doesGlobalDataExist("air_temperature_reference") && context->getGlobalDataType("air_temperature_reference") == helios::HELIOS_TYPE_FLOAT ) {
        context->getGlobalData("air_temperature_reference", air_temperature_reference);
    }

    float air_humidity_reference = air_humidity_default; // Default air relative humidity
    if ( context->doesGlobalDataExist("air_humidity_reference") && context->getGlobalDataType("air_humidity_reference") == helios::HELIOS_TYPE_FLOAT ) {
        context->getGlobalData("air_humidity_reference", air_humidity_reference);
    }

    float wind_speed_reference = wind_speed_default; // Default wind speed in m/s
    if ( context->doesGlobalDataExist("wind_speed_reference") && context->getGlobalDataType("wind_speed_reference") == helios::HELIOS_TYPE_FLOAT ) {
        context->getGlobalData("wind_speed_reference", wind_speed_reference);
    }

    // Variables to be set externally via global data

    float Patm;
    if ( context->doesGlobalDataExist("air_pressure") && context->getGlobalDataType("air_pressure") == helios::HELIOS_TYPE_FLOAT ) {
        context->getGlobalData("air_pressure", Patm);
    }else {
        Patm = pressure_default;
    }

    float air_temperature_average;
    if ( context->doesGlobalDataExist("air_temperature_average") && context->getGlobalDataType("air_temperature_average") == helios::HELIOS_TYPE_FLOAT ) {
        context->getGlobalData("air_temperature_average", air_temperature_average);
    }else {
        air_temperature_average = air_temperature_reference;
    }

    float air_moisture_average;
    if ( context->doesGlobalDataExist("air_moisture_average") && context->getGlobalDataType("air_moisture_average") == helios::HELIOS_TYPE_FLOAT ) {
        context->getGlobalData("air_moisture_average", air_moisture_average);
    }else {
        air_moisture_average = air_humidity_reference*esat_Pa(air_temperature_reference)/Patm;
    }

    // Get dimensions of canopy volume
    if ( canopy_dimensions == make_vec3(0,0,0) ) {
        vec2 xbounds, ybounds, zbounds;
        context->getDomainBoundingBox( UUIDs, xbounds, ybounds, zbounds);
        canopy_dimensions = make_vec3(xbounds.y - xbounds.x, ybounds.y - ybounds.x, zbounds.y - zbounds.x);
    }
    assert( canopy_dimensions.x>0 && canopy_dimensions.y>0 && canopy_dimensions.z>0 );
    if ( canopy_height_m == 0 ) {
        canopy_height_m = canopy_dimensions.z; // Set the canopy height if not already set
    }
    if ( reference_height_m == 0 ) {
        reference_height_m = canopy_height_m; // Set the reference height if not already set
    }

    std::cout << "Read in initial values. Ta = " << air_temperature_average << "; e = " << air_moisture_average << "; " << air_moisture_average/esat_Pa(air_temperature_reference)*Patm << std::endl;

    std::cout << "Canopy dimensions: " << canopy_dimensions.x << " x " << canopy_dimensions.y << " x " << canopy_height_m << std::endl;

    float displacement_height = 0.67f*canopy_height_m;
    float zo_m = 0.1f*canopy_height_m; // Roughness length for momentum transfer (m)
    float zo_h = 0.01f*canopy_height_m; // Roughness length for heat transfer (m)

    float time=0;
    float air_temperature_old = 0;
    float air_moisture_old = 0;
    while ( time<time_advance_sec ) {
        float dt_actual = dt_sec;
        if ( time + dt_actual > time_advance_sec ) {
            dt_actual = time_advance_sec - time; // Adjust the time step to not exceed the total time
        }

        // Update the surface energy balance
        this->run(UUIDs);

        // Calculate temperature source term
        float sensible_source_flux_W_m3;
        context->calculatePrimitiveDataAreaWeightedSum( UUIDs, "sensible_flux", sensible_source_flux_W_m3 ); //units: Watts
        sensible_source_flux_W_m3 /= canopy_dimensions.x*canopy_dimensions.y*canopy_height_m; // Convert to W/m^3

        // Calculate moisture source term
        float moisture_source_flux_W_m3;
        context->calculatePrimitiveDataAreaWeightedSum( UUIDs, "latent_flux", moisture_source_flux_W_m3 ); //units: Watts
        moisture_source_flux_W_m3 /= canopy_dimensions.x*canopy_dimensions.y*canopy_height_m; // Convert to W/m^3
        float moisture_source_flux_mol_s_m3 = moisture_source_flux_W_m3/lambda_mol; // Convert to mol/s/m^3

        float rho_air_mol_m3 = Patm / (R * air_temperature_average);; // Molar density of air (mol/m^3).

        // Calculate sensible flux upper boundary condition
        float ga = std::pow(von_Karman_constant,2)*wind_speed_reference/(std::log((reference_height_m-displacement_height)/zo_m)*std::log((reference_height_m-displacement_height)/zo_h))*rho_air_mol_m3; //aerodynamic conductance (mol/m^2/s).

        // Calculate moisture flux upper boundary condition
        float air_moisture_reference = esat_Pa(air_temperature_reference)/Patm*air_humidity_reference;

        // Advance air temperature equation
        // implicit denominator  1 + Δt/hc · g_a/n
        float denominator_temp = 1.f + (dt_actual * ga) / (rho_air_mol_m3 * canopy_height_m);
        // RHS numerator
        float numerator_temp = air_temperature_average + dt_actual / (rho_air_mol_m3 * cp_air_mol * canopy_height_m) * (sensible_source_flux_W_m3 * canopy_height_m + cp_air_mol * ga * air_temperature_reference);

        // update
        air_temperature_average = numerator_temp / denominator_temp;

        // Advance air moisture equation
        float air_moisture_average_old = air_moisture_average;
        // factor  f = [1-(1-ε)·x]² / ε   evaluated at the old state
        float f = std::pow(1.f - (1.f - 0.622f) * air_moisture_average_old, 2) / 0.622f;
        // implicit denominator  1 + Δt  f  ga / (ρ h_c)
        float denominator_moist = 1.f + dt_actual * f * ga / (rho_air_mol_m3 * canopy_height_m);
        // implicit numerator   x_old + Δt  f  (E_s + ga·x_ref / h_c) / (ρ)
        float numerator_moist = air_moisture_average_old + dt_actual * f * (moisture_source_flux_mol_s_m3 + ga * air_moisture_reference / canopy_height_m) / rho_air_mol_m3;
        air_moisture_average = numerator_moist / denominator_moist;

        // Cap relative humidity to 100%
        float esat = esat_Pa(air_temperature_average)/Patm;
        if ( air_moisture_average > esat ) {
            std::cerr << "WARNING (EnergyBalanceModel::evaluateAirEnergyBalance): Air moisture exceeds saturation. Capping to saturation value." << std::endl;
            air_moisture_average = 0.99f*esat;
        }

        // Check that timestep is not too large based on aerodynamic resistance
        if ( dt_actual > 0.5f*canopy_height_m*rho_air_mol_m3/ga ) {
            std::cerr << "WARNING (EnergyBalanceModel::evaluateAirEnergyBalance): Time step is too large.  The air energy balance may not converge properly." << std::endl;
        }

        // Set primitive data
        context->setPrimitiveData( UUIDs, "air_temperature", air_temperature_average );
        float air_humidity = air_moisture_average*Patm/esat_Pa(air_temperature_average);
        context->setPrimitiveData( UUIDs, "air_humidity", air_humidity );

        // Set global data
        context->setGlobalData("air_temperature_average", air_temperature_average);
        context->setGlobalData("air_humidity_average", air_humidity);
        context->setGlobalData("air_moisture_average", air_moisture_average);

        context->setGlobalData("aerodynamic_resistance", rho_air_mol_m3/ga);

        std::cout << "Computed air temperature: " << air_temperature_average << " " << air_temperature_old << std::endl;
        std::cout << "Computed air humidity: " << air_humidity << std::endl;
        std::cout << "Computed air moisture: " << air_moisture_average << " " << air_moisture_old << std::endl;

        if ( time>0 && std::abs(air_temperature_average-air_temperature_old)/air_temperature_old < 0.003f && std::abs(air_moisture_average-air_moisture_old)/air_moisture_old < 0.003f ) {
            // If the air temperature and humidity have not changed significantly, we can stop iterating
            std::cout << "Converged" << std::endl;
            break;
        }
        air_temperature_old = air_temperature_average;
        air_moisture_old = air_moisture_average;

        time += dt_sec;

    }

}

void EnergyBalanceModel::enableMessages(){
  message_flag = true;
}

void EnergyBalanceModel::disableMessages(){
  message_flag = false;
}

void EnergyBalanceModel::addRadiationBand( const char* band ){
    if( std::find(radiation_bands.begin(),radiation_bands.end(),band)==radiation_bands.end()) { //only add band if it doesn't exist
        radiation_bands.emplace_back(band);
    }
}

void EnergyBalanceModel::addRadiationBand( const std::vector<std::string> &bands ){
    for( auto &band : bands ){
        addRadiationBand(band.c_str());
    }
}

void EnergyBalanceModel::enableAirEnergyBalance() {
    enableAirEnergyBalance(0,0);
}

void EnergyBalanceModel::enableAirEnergyBalance( float canopy_height_m, float reference_height_m ) {

    if ( canopy_height_m<0 ) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableAirEnergyBalance): Canopy height must be greater than or equal to zero.");
    }else if ( canopy_height_m!=0 && reference_height_m<canopy_height_m ) {
        helios_runtime_error("ERROR (EnergyBalanceModel::enableAirEnergyBalance): Reference height must be greater than or equal to canopy height.");
    }

    air_energy_balance_enabled = true;
    this->canopy_height_m = canopy_height_m;
    this->reference_height_m = reference_height_m;
}

void EnergyBalanceModel::optionalOutputPrimitiveData( const char* label ){

  if( strcmp(label,"boundarylayer_conductance_out")==0 || strcmp(label,"vapor_pressure_deficit")==0 || strcmp(label,"storage_flux")==0 || strcmp(label,"net_radiation_flux")==0 ){
    output_prim_data.emplace_back( label );
  }else{
    std::cout << "WARNING (EnergyBalanceModel::optionalOutputPrimitiveData): unknown output primitive data " << label << " will be ignored." << std::endl;
  }
  
}

void EnergyBalanceModel::printDefaultValueReport() const{
    printDefaultValueReport(context->getAllUUIDs());
}

void EnergyBalanceModel::printDefaultValueReport(const std::vector<uint> &UUIDs) const {

    size_t assumed_default_TL = 0;
    size_t assumed_default_U = 0;
    size_t assumed_default_L = 0;
    size_t assumed_default_p = 0;
    size_t assumed_default_Ta = 0;
    size_t assumed_default_rh = 0;
    size_t assumed_default_gH = 0;
    size_t assumed_default_gs = 0;
    size_t assumed_default_Qother = 0;
    size_t assumed_default_heatcapacity = 0;
    size_t assumed_default_fs = 0;
    size_t twosided_0 = 0;
    size_t twosided_1 = 0;
    size_t Ne_1 = 0;
    size_t Ne_2 = 0;

    size_t Nprimitives = UUIDs.size();

    for (uint UUID: UUIDs) {

        //surface temperature (K)
        if (!context->doesPrimitiveDataExist(UUID, "temperature") ||
            context->getPrimitiveDataType(UUID, "temperature") != HELIOS_TYPE_FLOAT) {
            assumed_default_TL++;
        }

        //air pressure (Pa)
        if (!context->doesPrimitiveDataExist(UUID, "air_pressure") ||
            context->getPrimitiveDataType(UUID, "air_pressure") != HELIOS_TYPE_FLOAT) {
            assumed_default_p++;
        }

        //air temperature (K)
        if (!context->doesPrimitiveDataExist(UUID, "air_temperature") ||
            context->getPrimitiveDataType(UUID, "air_temperature") != HELIOS_TYPE_FLOAT) {
            assumed_default_Ta++;
        }

        //air humidity
        if (!context->doesPrimitiveDataExist(UUID, "air_humidity") ||
            context->getPrimitiveDataType(UUID, "air_humidity") != HELIOS_TYPE_FLOAT) {
            assumed_default_rh++;
        }

        //boundary-layer conductance to heat
        if (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") || context->getPrimitiveDataType(UUID, "boundarylayer_conductance") != HELIOS_TYPE_FLOAT){
             assumed_default_gH++;
        }

        //wind speed
        if (!context->doesPrimitiveDataExist(UUID, "wind_speed") || context->getPrimitiveDataType(UUID, "wind_speed") != HELIOS_TYPE_FLOAT){
            assumed_default_U++;
        }

        //object length
        if (!context->doesPrimitiveDataExist(UUID, "object_length") || context->getPrimitiveDataType(UUID, "object_length") != HELIOS_TYPE_FLOAT){
            assumed_default_L++;
        }

        //moisture conductance
        if (!context->doesPrimitiveDataExist(UUID, "moisture_conductance") || context->getPrimitiveDataType(UUID, "moisture_conductance") != HELIOS_TYPE_FLOAT){
            assumed_default_gs++;
        }

        //Heat capacity
        if (!context->doesPrimitiveDataExist(UUID, "heat_capacity") || context->getPrimitiveDataType(UUID, "heat_capacity") != HELIOS_TYPE_FLOAT) {
            assumed_default_heatcapacity++;
        }

        //"Other" heat fluxes
        if (!context->doesPrimitiveDataExist(UUID, "other_surface_flux") || context->getPrimitiveDataType(UUID, "other_surface_flux") != HELIOS_TYPE_FLOAT) {
            assumed_default_Qother++;
        }

        //two-sided flag
        if ( context->doesPrimitiveDataExist(UUID, "twosided_flag") && context->getPrimitiveDataType(UUID, "twosided_flag") == HELIOS_TYPE_UINT) {
            uint twosided;
            context->getPrimitiveData(UUID, "twosided_flag", twosided);
            if( twosided==0 ){
                twosided_0++;
            }else if( twosided==1 ){
                twosided_1++;
            }
        }else{
            twosided_1++;
        }

        //number of evaporating faces
        if ( context->doesPrimitiveDataExist(UUID, "evaporating_faces") && context->getPrimitiveDataType(UUID, "evaporating_faces") == HELIOS_TYPE_UINT) {
            uint Ne;
            context->getPrimitiveData(UUID, "evaporating_faces", Ne);
            if( Ne==1 ){
                Ne_1++;
            }else if( Ne==2 ){
                Ne_2++;
            }
        }else{
            Ne_1++;
        }

        //Surface humidity
        if (!context->doesPrimitiveDataExist(UUID, "surface_humidity") || context->getPrimitiveDataType(UUID, "surface_humidity") != HELIOS_TYPE_FLOAT) {
            assumed_default_fs++;
        }

    }

    std::cout << "--- Energy Balance Model Default Value Report ---" << std::endl;

    std::cout << "surface temperature (initial guess): " << assumed_default_TL << " of " << Nprimitives << " used default value of " << temperature_default << " because ""temperature"" primitive data did not exist" << std::endl;
    std::cout << "air pressure: " << assumed_default_p << " of " << Nprimitives << " used default value of " << pressure_default << " because ""air_pressure"" primitive data did not exist" << std::endl;
    std::cout << "air temperature: " << assumed_default_Ta << " of " << Nprimitives << " used default value of " << air_temperature_default << " because ""air_temperature"" primitive data did not exist" << std::endl;
    std::cout << "air humidity: " << assumed_default_rh << " of " << Nprimitives << " used default value of " << air_humidity_default << " because ""air_humidity"" primitive data did not exist" << std::endl;
    std::cout << "boundary-layer conductance: " << assumed_default_gH << " of " << Nprimitives << " calculated boundary-layer conductance from Polhausen equation because ""boundarylayer_conductance"" primitive data did not exist" << std::endl;
    if( assumed_default_gH>0 ){
        std::cout << "  - wind speed: " << assumed_default_U << " of " << assumed_default_gH << " using Polhausen equation used default value of " << wind_speed_default << " because ""wind_speed"" primitive data did not exist" << std::endl;
        std::cout << "  - object length: " << assumed_default_L << " of " << assumed_default_gH << " using Polhausen equation used the primitive/object length/area to calculate object length because ""object_length"" primitive data did not exist" << std::endl;
    }
    std::cout << "moisture conductance: " << assumed_default_gs << " of " << Nprimitives << " used default value of " << gS_default << " because ""moisture_conductance"" primitive data did not exist" << std::endl;
    std::cout << "surface humidity: " << assumed_default_fs << " of " << Nprimitives << " used default value of " << surface_humidity_default << " because ""surface_humidity"" primitive data did not exist" << std::endl;
    std::cout << "two-sided flag: " << twosided_0 << " of " << Nprimitives << " used two-sided flag=0; " << twosided_1 << " of " << Nprimitives << " used two-sided flag=1 (default)" << std::endl;
    std::cout << "evaporating faces: " << Ne_1 << " of " << Nprimitives << " used Ne = 1 (default); " << Ne_2 << " of " << Nprimitives << " used Ne = 2" << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;

}
