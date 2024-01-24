/** \file "StomatalConductanceModel.cpp" Primary source file for stomatalconductance plug-in.

    Copyright (C) 2016-2024 Brian Bailey

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

StomatalConductanceModel::StomatalConductanceModel( helios::Context* m_context ){
    context = m_context;

    //default values set here

    i_default = 0; //W/m^2

    TL_default = 300; // Kelvin

    air_temperature_default = 300;  // Kelvin

    air_humidity_default = 0.5; // %

    pressure_default = 101000; // Pa

    xylem_potential_default = -0.1; //MPa

    An_default = 0;   //umol/m^2/s

    Gamma_default = 100;   //umol/mol

    air_CO2_default = 400;  //umol/mol

    blconductance_default = 0.1;   //mol/m^2/s

    beta_default = 1.f;   //unitless

    model = "BMF"; //default model - Buckley, Mott, Farquhar

}

int StomatalConductanceModel::selfTest(){

    Context context_selftest;

  std::cout << "Running stomatal conductance model self-test..." << std::flush;

  float RMSE_max = 0.03;

  std::vector<float> An_ref{21.81, 22.71, 20.02, 22.60, 19.97, 17.32, 11.90, 6.87, 1.21, -1.49};
  std::vector<float> Tair_ref{20.69, 30.37, 39.19, 27.06, 27.12, 27.11, 27.08, 26.98, 26.87, 26.81};
  std::vector<float> TL_ref{21.00, 30.02, 38.01, 26.99, 27.00, 27.01, 27.01, 27.00, 27.00, 26.99};
  std::vector<float> Q_ref{2000, 2000, 2000, 2000, 1200, 800, 400, 200, 50, 0};
  float gbw_ref = 3.5;
  float Cs_ref = 400;
  float hs_ref = 0.55;
  float Patm_ref = 101300;
  std::vector<float> Gamma_ref = {43.7395, 70.2832, 105.5414, 60.0452, 60.0766, 60.1080, 60.1080, 60.0766, 60.0766, 60.0452};

  std::vector<float> gs_ref{0.3437, 0.3386, 0.3531, 0.3811, 0.3247, 0.2903, 0.2351, 0.1737, 0.0868, 0.0421};
  std::vector<float> gs_BWB(gs_ref.size());
  std::vector<float> gs_BBL(gs_ref.size());
  std::vector<float> gs_MOPT(gs_ref.size());
  std::vector<float> gs_BMF(gs_ref.size());

  uint UUID0 = context_selftest.addPatch();

  BWBcoefficients BWBcoeffs;
  BBLcoefficients BBLcoeffs;
  MOPTcoefficients MOPTcoeffs;
  BMFcoefficients BMFcoeffs;

  StomatalConductanceModel gsm(&context_selftest);

  float RMSE_BWB = 0.f;
  float RMSE_BBL = 0.f;
  float RMSE_MOPT = 0.f;
  float RMSE_BMF = 0.f;

  for( uint i=0; i<gs_ref.size(); i++ ) {

    context_selftest.setPrimitiveData( UUID0, "radiation_flux_PAR", Q_ref.at(i)/4.57f );
    context_selftest.setPrimitiveData( UUID0, "net_photosynthesis", An_ref.at(i) );
    context_selftest.setPrimitiveData( UUID0, "temperature", TL_ref.at(i)+273.f );
    context_selftest.setPrimitiveData( UUID0, "air_temperature", Tair_ref.at(i)+273.f );
    context_selftest.setPrimitiveData( UUID0, "air_CO2", Cs_ref );
    context_selftest.setPrimitiveData( UUID0, "air_humidity", hs_ref );
    context_selftest.setPrimitiveData( UUID0, "air_pressure", Patm_ref );
    context_selftest.setPrimitiveData( UUID0, "boundarylayer_conductance", gbw_ref);
    context_selftest.setPrimitiveData( UUID0, "Gamma_CO2", Gamma_ref.at(i));

    gsm.setModelCoefficients(BWBcoeffs);
    gsm.run();
    context_selftest.getPrimitiveData(UUID0,"moisture_conductance",gs_BWB.at(i));
    RMSE_BWB += pow(gs_BWB.at(i)-gs_ref.at(i),2)/float(gs_ref.size());

    gsm.setModelCoefficients(BBLcoeffs);
    gsm.run();
    context_selftest.getPrimitiveData(UUID0,"moisture_conductance",gs_BBL.at(i));
    RMSE_BBL += pow(gs_BBL.at(i)-gs_ref.at(i),2)/float(gs_ref.size());

    gsm.setModelCoefficients(MOPTcoeffs);
    gsm.run();
    context_selftest.getPrimitiveData(UUID0,"moisture_conductance",gs_MOPT.at(i));
    RMSE_MOPT += pow(gs_MOPT.at(i)-gs_ref.at(i),2)/float(gs_ref.size());

    gsm.setModelCoefficients(BMFcoeffs);
    gsm.run();
    context_selftest.getPrimitiveData(UUID0,"moisture_conductance",gs_BMF.at(i));
    RMSE_BMF += pow(gs_BMF.at(i)-gs_ref.at(i),2)/float(gs_ref.size());

  }

  if( sqrtf(RMSE_BWB) > RMSE_max || sqrtf(RMSE_BBL) > RMSE_max || sqrtf(RMSE_MOPT) > RMSE_max || sqrtf(RMSE_BMF) > RMSE_max ){
    std::cout << "failed." << std::endl;
    std::cout << sqrtf(RMSE_BWB) << " " << sqrtf(RMSE_BBL) << " " << sqrtf(RMSE_MOPT) << " " << sqrtf(RMSE_BMF) << std::endl;
        return 1;
  }

  std::cout << "passed." << std::endl;
  return 0;

}

void StomatalConductanceModel::setModelCoefficients(const BWBcoefficients &coeffs ){
  BWBcoeffs = coeffs;
  BWBmodel_coefficients.clear();
  model = "BWB";
}

void StomatalConductanceModel::setModelCoefficients(const BWBcoefficients &coeffs, const vector<uint> &UUIDs ){
  for( uint UUID : UUIDs){
    BWBmodel_coefficients[UUID] = coeffs;
  }
  model = "BWB";
}

void StomatalConductanceModel::setModelCoefficients(const BBLcoefficients &coeffs ){
  BBLcoeffs = coeffs;
  BBLmodel_coefficients.clear();
  model = "BBL";
}

void StomatalConductanceModel::setModelCoefficients(const BBLcoefficients &coeffs, const vector<uint> &UUIDs ){
  for( uint UUID : UUIDs){
    BBLmodel_coefficients[UUID] = coeffs;
  }
  model = "BBL";
}

void StomatalConductanceModel::setModelCoefficients(const MOPTcoefficients &coeffs ){
  MOPTcoeffs = coeffs;
  MOPTmodel_coefficients.clear();
  model = "MOPT";
}

void StomatalConductanceModel::setModelCoefficients(const MOPTcoefficients &coeffs, const vector<uint> &UUIDs ){
  for( uint UUID : UUIDs){
    MOPTmodel_coefficients[UUID] = coeffs;
  }
  model = "MOPT";
}

void StomatalConductanceModel::setModelCoefficients(const BMFcoefficients &coeffs ){
    BMFcoeffs = coeffs;
    BMFmodel_coefficients.clear();
    model = "BMF";
}

void StomatalConductanceModel::setModelCoefficients(const BMFcoefficients &coeffs, const vector<uint> &UUIDs ){
    for( uint UUID : UUIDs){
        BMFmodel_coefficients[UUID] = coeffs;
    }
    model = "BMF";
}

void StomatalConductanceModel::setModelCoefficients(const BBcoefficients &coeffs ){
    BBcoeffs = coeffs;
    BBmodel_coefficients.clear();
    model = "BB";
}

void StomatalConductanceModel::setModelCoefficients(const BBcoefficients &coeffs, const vector<uint> &UUIDs ){
    for( uint UUID : UUIDs){
        BBmodel_coefficients[UUID] = coeffs;
    }
    model = "BB";
}

void StomatalConductanceModel::setDynamicTimeConstants( float tau_open, float tau_close ){
    dynamic_time_constants.clear();
    dynamic_time_constants[0] = make_vec2(tau_open,tau_close);
}

void StomatalConductanceModel::setDynamicTimeConstants( float tau_open, float tau_close, const std::vector<uint> &UUIDs ){
    for( uint UUID : UUIDs){
        dynamic_time_constants[UUID] = make_vec2(tau_open,tau_close);
    }
}

void StomatalConductanceModel::run(){
    run( context->getAllUUIDs() );
}

void StomatalConductanceModel::run( const std::vector<uint>& UUIDs ){
    run( UUIDs, 0.f );
}

void StomatalConductanceModel::run( float dt ){
    run( context->getAllUUIDs(), dt );
}

void StomatalConductanceModel::run( const std::vector<uint>& UUIDs, float dt ){

    size_t assumed_default_An = 0;
    size_t assumed_default_Gamma = 0;

    bool warn_dt_too_large = false;
    bool warn_tau_unspecified = false;
    bool warn_old_gs_unspecified = false;

    for( uint UUID : UUIDs){

        if( !context->doesPrimitiveExist(UUID) ){
            std::cout << "WARNING (StomatalConductance::run): primitive " << UUID << " does not exist in the Context." << std::endl;
            continue;
        }

        //PAR radiation flux (W/m^2)
        float i = i_default;
        if( context->doesPrimitiveDataExist(UUID,"radiation_flux_PAR") && context->getPrimitiveDataType(UUID,"radiation_flux_PAR")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"radiation_flux_PAR",i); //W/m^2
            i = i*4.57f; //umol/m^2-s (ref https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf)
        }

        //surface temperature (K)
        float TL = TL_default;
        if( context->doesPrimitiveDataExist(UUID,"temperature") && context->getPrimitiveDataType(UUID,"temperature")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"temperature",TL); //Kelvin
            if( TL<250.f ) {
              std::cout << "WARNING (StomatalConductanceModel::run): Specified surface temperature value is very low - assuming default value instead. Did you accidentally specify temperature in Celcius instead of Kelvin?" << std::endl;
              TL = TL_default;
            }
        }

        //air pressure (Pa)
        float press = pressure_default;
        if( context->doesPrimitiveDataExist(UUID,"air_pressure") && context->getPrimitiveDataType(UUID,"air_pressure")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"air_pressure",press); //Pa
            if( press<50000 ) {
              std::cout << "WARNING (StomatalConductanceModel::run): Specified air pressure value is very low - assuming default value instead. Did you accidentally specify pressure in kPA instead of Pa?" << std::endl;
              press = pressure_default;
            }
        }

        //air temperature (K)
        float Ta = air_temperature_default;
        if( context->doesPrimitiveDataExist(UUID,"air_temperature") && context->getPrimitiveDataType(UUID,"air_temperature")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"air_temperature",Ta); //Kelvin
            if( Ta<250.f ) {
              std::cout << "WARNING (StomatalConductanceModel::run): Specified air temperature value is very low - assuming default value instead. Did you accidentally specify temperature in Celcius instead of Kelvin?" << std::endl;
              Ta = air_temperature_default;
            }
        }

        //boundary-layer conductance to heat/moisture (mol/m^2/s)
        float gbw = blconductance_default;
        if( context->doesPrimitiveDataExist(UUID,"boundarylayer_conductance") && context->getPrimitiveDataType(UUID,"boundarylayer_conductance")==HELIOS_TYPE_FLOAT ) {
            context->getPrimitiveData(UUID, "boundarylayer_conductance", gbw);
            gbw = gbw * 1.08;  //assume bl conductance to moisture is 1.08 of conductance to heat
        }else if( context->doesPrimitiveDataExist(UUID,"boundarylayer_conductance_out") && context->getPrimitiveDataType(UUID,"boundarylayer_conductance_out")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"boundarylayer_conductance_out",gbw);
            gbw = gbw*1.08;  //assume bl conductance to moisture is 1.08 of conductance to heat
        }
        if( gbw<0 ){
            gbw = 0;
            std::cout << "WARNING (StomatalConductanceModel::run): Boundary-layer conductance value provided was negative. Clipping to zero." << std::endl;
        }

        //beta soil moisture factor
        float beta = beta_default;
        if( context->doesPrimitiveDataExist(UUID,"beta_soil") && context->getPrimitiveDataType(UUID,"beta_soil")==HELIOS_TYPE_FLOAT ){
          context->getPrimitiveData(UUID,"beta_soil",beta);
        }

        //air humidity
        float rh = air_humidity_default;
        if( context->doesPrimitiveDataExist(UUID,"air_humidity") && context->getPrimitiveDataType(UUID,"air_humidity")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"air_humidity",rh);
            if( rh>1.f ) {
              std::cout << "WARNING (StomatalConductanceModel::run): Specified air humidity value is greater than 1 - clamping to 1. Did you accidentally specify in percent instead of a decimal?" << std::endl;
              rh = 1.f;
            }
        }

        //calculate VPD (mmol/mol) between sub-stomatal cavity and outside of boundary layer
        float esat = 611.f * exp(17.502f * (Ta - 273.f) / ((Ta - 273.f) + 240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in degC, and result is in Pascals
        float ea = rh * esat; // Definition of vapor pressure (see Campbell and Norman pp. 42 Eq. 3.11)
        float es = 611.f * exp(17.502f * (TL - 273.f) / ((TL - 273.f) + 240.97f));
        float D = max(0.f, (es - ea) / press * 1000.f); //mmol/mol


        //Compute CO2 concentration at leaf surface
        float An = An_default;
        float Gamma = Gamma_default;
        float Cs = air_CO2_default;
        if( model == "BWB" || model == "BBL" || model == "MOPT" ){

            //net photosynthesis
            if( context->doesPrimitiveDataExist(UUID,"net_photosynthesis") && context->getPrimitiveDataType(UUID,"net_photosynthesis")==HELIOS_TYPE_FLOAT ) {
                context->getPrimitiveData(UUID, "net_photosynthesis", An);
            }else{
                assumed_default_An++;
            }

            //CO2 compensation point - Gamma
            if( context->doesPrimitiveDataExist(UUID,"Gamma_CO2") && context->getPrimitiveDataType(UUID,"Gamma_CO2")==HELIOS_TYPE_FLOAT ) {
                context->getPrimitiveData(UUID, "Gamma_CO2", Gamma);
            }else{
                assumed_default_Gamma++;
            }

            //ambient air CO2
            float Ca = air_CO2_default;
            if( context->doesPrimitiveDataExist(UUID,"air_CO2") && context->getPrimitiveDataType(UUID,"air_CO2")==HELIOS_TYPE_FLOAT ){
                context->getPrimitiveData(UUID,"air_CO2",Ca);
            }

            float gbc = 0.75f*gbw; //bl conductance to CO2

            //An = gbc*(Ca-Cs)
            Cs = Ca - An/gbc;

        }

        float Psix = xylem_potential_default;
        if( model == "BB" ){
            //xylem moisture potential
            if( context->doesPrimitiveDataExist(UUID,"xylem_water_potential") && context->getPrimitiveDataType(UUID,"xylem_water_potential")==HELIOS_TYPE_FLOAT ){
                context->getPrimitiveData(UUID,"xylem_water_potential",Psix);
            }
        }


        float gs;
        if( model == "BWB" ){

          //model coefficients
          float gs0;
          float a1;
          BWBcoefficients coeffs;
          if ( BWBmodel_coefficients.empty() || BWBmodel_coefficients.find(UUID)==BWBmodel_coefficients.end() ) {
            coeffs = BWBcoeffs;
          } else {
            coeffs = BWBmodel_coefficients.at(UUID);
          }
          gs0 = coeffs.gs0;
          a1 = coeffs.a1;

          std::vector<float> variables{An,Cs,es,ea,gbw,beta};

          float esurf = fzero( evaluate_BWBmodel, variables, &coeffs, es );
          float hs = esurf/es;
          gs = gs0 + a1*An*beta*hs/Cs;

        }else if(model == "BBL" ) {

          //model coefficients
          float gs0;
          float a1;
          float D0;
          BBLcoefficients coeffs;
          if ( BBLmodel_coefficients.empty() || BBLmodel_coefficients.find(UUID)==BBLmodel_coefficients.end() ) {
            coeffs = BBLcoeffs;
          } else {
            coeffs = BBLmodel_coefficients.at(UUID);
          }
          gs0 = coeffs.gs0;
          a1 = coeffs.a1;
          D0 = coeffs.D0;

          std::vector<float> variables{An,Cs,Gamma,es,ea,gbw,press,beta};

          float esurf = fzero( evaluate_BBLmodel, variables, &coeffs, es ); //Pa
          float Ds = max(0.f, (es - esurf) / press * 1000.f); //mmol/mol
          gs = gs0 + a1*An*beta/(Cs-Gamma)/(1.f+Ds/D0);

        }else if(model == "MOPT" ) {

          //model coefficients
          float gs0;
          float g1;
          MOPTcoefficients coeffs;
          if ( MOPTmodel_coefficients.empty() || MOPTmodel_coefficients.find(UUID)==MOPTmodel_coefficients.end() ) {
            coeffs = MOPTcoeffs;
          } else {
            coeffs = MOPTmodel_coefficients.at(UUID);
          }
          gs0 = coeffs.gs0;
          g1 = coeffs.g1;

          std::vector<float> variables{An,Cs,es,ea,gbw,beta};

          float esurf = fzero( evaluate_MOPTmodel, variables, &coeffs, ea );
          float Ds = max(0.00001f, (es - esurf)/1000.f ); //kPa
          gs = gs0 + 1.6f*(1.f+g1*sqrtf(beta/Ds))*An/Cs;

        }else if(model == "BB" ) {

            //model coefficients
            BBcoefficients coeffs;
            if ( BBmodel_coefficients.empty() || BBmodel_coefficients.find(UUID)==BBmodel_coefficients.end() ) {
                coeffs = BBcoeffs;
            } else {
                coeffs = BBmodel_coefficients.at(UUID);
            }

            std::vector<float> variables{i,D,Psix};

            gs = fzero( evaluate_BBmodel, variables, &coeffs, 0.1f );

        }else{

            //model coefficients
            BMFcoefficients coeffs;
            if ( BMFmodel_coefficients.empty() || BMFmodel_coefficients.find(UUID)==BMFmodel_coefficients.end() ) {
                coeffs = BMFcoeffs;
            } else {
                coeffs = BMFmodel_coefficients.at(UUID);
            }
            float Em = coeffs.Em;
            float i0 = coeffs.i0;
            float k = coeffs.k;
            float b = coeffs.b;

            std::vector<float> variables{i,es,ea,gbw,press,beta};

            float esurf = fzero( evaluate_BMFmodel, variables, &coeffs, es );
            float Ds = max(0.f, (es - esurf) / press * 1000.f);

            gs = Em * beta * (i + i0) / (k + b * i + (i + i0) * Ds);

            for( const auto &data : output_prim_data ){
              if( data=="vapor_pressure_deficit" ) {
                context->setPrimitiveData(UUID, "vapor_pressure_deficit", Ds);
              }
            }

        }

        if( dt>0 ){ //run dynamic stomatal conductance model

            if( dynamic_time_constants.empty() ){
                warn_tau_unspecified = true;
            }

            if( !dynamic_time_constants.empty() && (dynamic_time_constants.find(UUID)!=dynamic_time_constants.end() || dynamic_time_constants.find(0)!=dynamic_time_constants.end() ) ) {

                if (context->doesPrimitiveDataExist(UUID, "moisture_conductance") && context->getPrimitiveDataType(UUID, "moisture_conductance") == HELIOS_TYPE_FLOAT) {

                    float gs_old;
                    context->getPrimitiveData(UUID, "moisture_conductance", gs_old);

                    float gs_ss = gs;

                    float tau_open;
                    float tau_close;
                    if (dynamic_time_constants.find(UUID) != dynamic_time_constants.end()) {
                        tau_open = dynamic_time_constants.at(UUID).x;
                        tau_close = dynamic_time_constants.at(UUID).y;
                    } else {
                        tau_open = dynamic_time_constants.at(0).x;
                        tau_close = dynamic_time_constants.at(0).y;
                    }

                    float tau;
                    if (gs_old > gs_ss) {
                        tau = tau_close;
                    } else {
                        tau = tau_open;
                    }

                    if (dt > tau) {
                        warn_dt_too_large = true;
                    }

                    gs = gs_old + (gs_ss - gs_old) * dt / tau;

                } else {
                    warn_old_gs_unspecified = true;
                }

            }

        }

        context->setPrimitiveData( UUID, "moisture_conductance", gs);

    }

    if( model == "BWB" && assumed_default_An>0 ){
      std::cout << "WARNING (StomatalConductanceModel::run): The Ball-Woodrow-Berry stomatal conductance model requires net photosynthesis, but primitive data ""net_photosynthesis"" could not be found for " << assumed_default_An << " primitives. Did you forget to run the photosynthesis model?" << std::endl;
    }else if( model == "BBL" && assumed_default_An>0 ){
      std::cout << "WARNING (StomatalConductanceModel::run): The Ball-Berry-Leuning stomatal conductance model requires net photosynthesis, but primitive data ""net_photosynthesis"" could not be found for " << assumed_default_An << " primitives. Did you forget to run the photosynthesis model?" << std::endl;
    }
    if( model == "BBL" && assumed_default_Gamma>0 ){
      std::cout << "WARNING (StomatalConductanceModel::run): The Ball-Berry-Leuning stomatal conductance model requires the CO2 compensation point ""Gamma"", but primitive data ""Gamma_CO2"" could not be found for " << assumed_default_An << " primitives. Did you forget to set optional output primitive data ""Gamma_CO2"" in the photosynthesis model?" << std::endl;
    }

    if( warn_dt_too_large ){
        std::cout << "WARNING (StomatalConductanceModel::run): The specified time step is larger than the dynamic stomatal conductance time constant. This may result in inaccurate stomatal conductance values." << std::endl;
    }
    if( warn_old_gs_unspecified ){
        std::cout << "WARNING (StomatalConductanceModel::run): The dynamic stomatal conductance model requires the previous stomatal conductance value, but primitive data ""moisture_conductance"" could not be found for one or more primitives. Dynamic model was not run for these primitives this time step." << std::endl;
    }
    if( warn_tau_unspecified ){
        std::cout << "WARNING (StomatalConductanceModel::run): The dynamic stomatal conductance model requires the time constants to be specified using the StomatalConductance::setDynamicTimeConstants() method, but these were not specified for one or more primitives. Dynamic model was not run for these primitives." << std::endl;
    }

}

float StomatalConductanceModel::evaluate_BWBmodel( float esurf, std::vector<float> &variables, const void* parameters ){

  // We want to find the vapor pressure at the surface, esurf, that balances the equation gs(esurf)*(es-esurf) = gbw*(esurf-ea). This function returns the residual of this equation.

  const auto* coeffs = reinterpret_cast<const BWBcoefficients*>(parameters);

  float gs0 = coeffs->gs0;
  float a1 = coeffs->a1;

  float An = variables[0];
  float Cs = variables[1];
  float es = variables[2];
  float ea = variables[3];
  float gbw = variables[4];
  float beta = variables[5];

  float hs = esurf/es;
  float gs = gs0+a1*An*beta*hs/Cs;

  return gs*(es-esurf) - gbw*(esurf-ea);

}

float StomatalConductanceModel::evaluate_BBLmodel( float esurf, std::vector<float> &variables, const void* parameters ){

  // We want to find the vapor pressure at the surface, esurf, that balances the equation gs(esurf)*(es-esurf) = gbw*(esurf-ea). This function returns the residual of this equation.

  const auto* coeffs = reinterpret_cast<const BBLcoefficients*>(parameters);

  float gs0 = coeffs->gs0;
  float a1 = coeffs->a1;
  float D0 = coeffs->D0;

  float An = variables[0];
  float Cs = variables[1];
  float Gamma = variables[2];
  float es = variables[3];
  float ea = variables[4];
  float gbw = variables[5];
  float press = variables[6];
  float beta = variables[7];

  float Ds = max(0.f, (es - esurf) / press * 1000.f); //mmol/mol
  float gs = gs0+a1*An*beta/(Cs-Gamma)/(1.f+Ds/D0);

  return gs*(es - esurf) - gbw*(esurf-ea);

}

float StomatalConductanceModel::evaluate_MOPTmodel( float esurf, std::vector<float> &variables, const void* parameters ){

  // We want to find the vapor pressure at the surface, esurf, that balances the equation gs(esurf)*(es-esurf) = gbw*(esurf-ea). This function returns the residual of this equation.

  const auto* coeffs = reinterpret_cast<const MOPTcoefficients*>(parameters);

  float gs0 = coeffs->gs0;
  float g1 = coeffs->g1;

  float An = variables[0];
  float Cs = variables[1];
  float es = variables[2];
  float ea = variables[3];
  float gbw = variables[4];
  float beta = variables[5];

  float Ds = max(0.f, (es - esurf)/1000.f );  //kPa
  float gs = gs0+1.6f*(1.f+g1*sqrtf(beta/Ds))*An/Cs;

  return gs*(es - esurf) - gbw*(esurf-ea);

}

float StomatalConductanceModel::evaluate_BMFmodel( float esurf, std::vector<float> &variables, const void* parameters ){

  // We want to find the vapor pressure at the surface, esurf, that balances the equation gs(esurf)*(es-esurf) = gbw*(esurf-ea). This function returns the residual of this equation.

  const auto* coeffs = reinterpret_cast<const BMFcoefficients*>(parameters);

  float Em = coeffs->Em;
  float i0 = coeffs->i0;
  float k = coeffs->k;
  float b = coeffs->b;

  float i = variables[0];
  float es = variables[1];
  float ea = variables[2];
  float gbw = variables[3];
  float press = variables[4];
  float beta = variables[5];

  float Ds = max(0.f, (es - esurf) / press * 1000.f);  //mmol/mol
  float gs = Em * beta * (i + i0) / (k + b * i + (i + i0) * Ds);

  return gs*(es - esurf) - gbw*(esurf-ea);

}

float StomatalConductanceModel::evaluate_BBmodel( float gs, std::vector<float> &variables, const void* parameters ){

    float fe = 0.5;
    float Rxe = 200.f;

    const auto* coeffs = reinterpret_cast<const BBcoefficients*>(parameters);

    float pi_0 = coeffs->pi_0;
    float pi_m = coeffs->pi_m;
    float theta = coeffs->theta;
    float sigma = coeffs->sigma;
    float chi = coeffs->chi;

    float i = variables[0];
    float D = variables[1]*1e-3F;
    float Psix = variables[2];

    float pig = pi_0+(pi_m-pi_0)*i/(i+theta)+sigma*(Psix-Rxe*fe*gs*D);
    float Psi_e = Psix-Rxe*fe*gs*D;

    float Pg = max(0.f,pig+Psi_e);
    float gsm = chi*Pg;

    return gs-gsm;

}

void StomatalConductanceModel::optionalOutputPrimitiveData( const char* label ){

  if( strcmp(label,"vapor_pressure_deficit")==0 ){
    output_prim_data.emplace_back(label );
  }else{
    std::cout << "WARNING (StomatalConductanceModel::optionalOutputPrimitiveData): unknown output primitive data " << label << std::endl;
  }

}

void StomatalConductanceModel::printDefaultValueReport() const{
    printDefaultValueReport(context->getAllUUIDs());
}

void StomatalConductanceModel::printDefaultValueReport(const std::vector<uint> &UUIDs) const {

    size_t assumed_default_i = 0;
    size_t assumed_default_TL = 0;
    size_t assumed_default_p = 0;
    size_t assumed_default_Ta = 0;
    size_t assumed_default_rh = 0;
    size_t assumed_default_Psix = 0;
    size_t assumed_default_An = 0;
    size_t assumed_default_Gamma = 0;
    size_t assumed_default_CO2 = 0;
    size_t assumed_default_gbw = 0;
    size_t assumed_default_beta = 0;

    size_t Nprimitives = UUIDs.size();

    for (uint UUID: UUIDs) {

        if (!context->doesPrimitiveDataExist(UUID, "radiation_flux_PAR") ||
            context->getPrimitiveDataType(UUID, "radiation_flux_PAR") != HELIOS_TYPE_FLOAT) {
            assumed_default_i++;
        }

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

        //boundary-layer conductance to heat/moisture (mol/m^2/s)
        if ((!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") || context->getPrimitiveDataType(UUID, "boundarylayer_conductance") != HELIOS_TYPE_FLOAT) &&
            (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out") || context->getPrimitiveDataType(UUID, "boundarylayer_conductance_out") != HELIOS_TYPE_FLOAT)) {
            assumed_default_gbw++;
        }

        //beta soil moisture factor
        if (!context->doesPrimitiveDataExist(UUID, "beta_soil") ||
            context->getPrimitiveDataType(UUID, "beta_soil") != HELIOS_TYPE_FLOAT) {
            assumed_default_beta++;
        }

        //Compute CO2 concentration at leaf surface
        if (model == "BWB" || model == "BBL" || model == "MOPT") {

            //net photosynthesis
            if (!context->doesPrimitiveDataExist(UUID, "net_photosynthesis") ||
                context->getPrimitiveDataType(UUID, "net_photosynthesis") != HELIOS_TYPE_FLOAT) {
                assumed_default_An++;
            }

            //CO2 compensation point - Gamma
            if (!context->doesPrimitiveDataExist(UUID, "Gamma_CO2") ||
                context->getPrimitiveDataType(UUID, "Gamma_CO2") != HELIOS_TYPE_FLOAT) {
                assumed_default_Gamma++;
            }

            //ambient air CO2
            if (!context->doesPrimitiveDataExist(UUID, "air_CO2") ||
                context->getPrimitiveDataType(UUID, "air_CO2") != HELIOS_TYPE_FLOAT) {
                assumed_default_CO2++;
            }
        }else if( model == "BB" ){
            //xylem moisture potential
            if (!context->doesPrimitiveDataExist(UUID, "xylem_water_potential") ||
                context->getPrimitiveDataType(UUID, "xylem_water_potential") != HELIOS_TYPE_FLOAT) {
                assumed_default_Psix++;
            }
        }


    }

    std::cout << "--- Stomatal Conductance Model Default Value Report ---" << std::endl;

    std::cout << "PAR flux: " << assumed_default_i << " of " << Nprimitives << " used default value of " << i_default << " because ""radiation_flux_PAR"" primitive data did not exist" << std::endl;
    std::cout << "surface temperature: " << assumed_default_TL << " of " << Nprimitives << " used default value of " << TL_default << " because ""temperature"" primitive data did not exist" << std::endl;
    std::cout << "air pressure: " << assumed_default_p << " of " << Nprimitives << " used default value of " << pressure_default << " because ""air_pressure"" primitive data did not exist" << std::endl;
    std::cout << "air temperature: " << assumed_default_Ta << " of " << Nprimitives << " used default value of " << air_temperature_default << " because ""air_temperature"" primitive data did not exist" << std::endl;
    std::cout << "air humidity: " << assumed_default_rh << " of " << Nprimitives << " used default value of " << air_humidity_default << " because ""air_humidity"" primitive data did not exist" << std::endl;
    std::cout << "boundary-layer conductance: " << assumed_default_gbw << " of " << Nprimitives << " used default value of " << blconductance_default << " because ""boundarylayer_conductance"" primitive data did not exist" << std::endl;
    std::cout << "soil moisture factor: " << assumed_default_beta << " of " << Nprimitives << " used default value of " << beta_default << " because ""soil_beta"" primitive data did not exist" << std::endl;

    if (model == "BWB" || model == "BBL" || model == "MOPT") {
        std::cout << "net photosynthesis: " << assumed_default_An << " of " << Nprimitives << " used default value of " << An_default << " because ""net_photosynthesis"" primitive data did not exist" << std::endl;
        std::cout << "Gamma: " << assumed_default_Gamma << " of " << Nprimitives << " used default value of " << Gamma_default << " because ""Gamma_CO2"" primitive data did not exist" << std::endl;
        std::cout << "air CO2: " << assumed_default_CO2 << " of " << Nprimitives << " used default value of " << air_CO2_default << " because ""air_CO2"" primitive data did not exist" << std::endl;
    }else if( model == "BB" ){
        std::cout << "xylem water potential: " << assumed_default_Psix << " of " << Nprimitives << " used default value of " << xylem_potential_default << " because ""xylem_water_potential"" primitive data did not exist" << std::endl;
    }

    std::cout << "------------------------------------------------------" << std::endl;

}