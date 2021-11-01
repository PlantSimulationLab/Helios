/** \file "StomatalConductanceModel.cpp" Primary source file for stomatalconductance plug-in.
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

}

int StomatalConductanceModel::selfTest(){
    return 0;
}

void StomatalConductanceModel::setModelCoefficients(const BMFcoefficients &coeffs ){
    BMFcoeffs = coeffs;
    BMFmodel_coefficients.clear();
    model = "BMF";
}

void StomatalConductanceModel::setModelCoefficients(const BMFcoefficients &coeffs, const vector<uint> &UUIDs ){
    for( uint UUID : UUIDs){
        BMFmodel_coefficients.at(UUID) = coeffs;
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
        BBmodel_coefficients.at(UUID) = coeffs;
    }
    model = "BB";
}

void StomatalConductanceModel::run(){
    run( context->getAllUUIDs() );
}

void StomatalConductanceModel::run( const std::vector<uint>& UUIDs ){

    size_t assumed_default_i = 0;
    size_t assumed_default_TL = 0;
    size_t assumed_default_p = 0;
    size_t assumed_default_Ta = 0;
    size_t assumed_default_rh = 0;
    size_t assumed_default_Psix = 0;

    for( uint UUID : UUIDs){

        if( !context->doesPrimitiveExist(UUID) ){
            std::cout << "WARNING (StomatalConductance::run): primitive " << UUID << " does not exist in the Context." << std::endl;
            continue;
        }

        //PAR radiation flux
        float i = i_default;
        if( context->doesPrimitiveDataExist(UUID,"radiation_flux_PAR") ){
            context->getPrimitiveData(UUID,"radiation_flux_PAR",i); //W/m^2
            i = i*4.57f; //umol/m^2-s (ref https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf)
        }else{
            assumed_default_i++;
        }

        //surface temperature
        float TL = TL_default;
        if( context->doesPrimitiveDataExist(UUID,"temperature") ){
            context->getPrimitiveData(UUID,"temperature",TL); //Kelvin
        }else{
            assumed_default_TL++;
        }

        //air pressure
        float press = pressure_default;
        if( context->doesPrimitiveDataExist(UUID,"air_pressure") ){
            context->getPrimitiveData(UUID,"air_pressure",press); //Pa
        }else{
            assumed_default_p++;
        }

        //air temperature
        float Ta = air_temperature_default;
        if( context->doesPrimitiveDataExist(UUID,"air_temperature") ){
            context->getPrimitiveData(UUID,"air_temperature",Ta); //Kelvin
        }else{
            assumed_default_Ta++;
        }

        //air humidity
        float rh = air_humidity_default;
        if( context->doesPrimitiveDataExist(UUID,"air_humidity") ){
            context->getPrimitiveData(UUID,"air_humidity",rh);
        }else{
            assumed_default_rh++;
        }

        //calculate VPD
        float esat = 611.f * exp(17.502f * (Ta - 273.f) / ((Ta - 273.f) + 240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in degC, and result is in Pascals
        float ea = rh * esat; // Definition of vapor pressure (see Campbell and Norman pp. 42 Eq. 3.11)
        float es = 611.f * exp(17.502f * (TL - 273.f) / ((TL - 273.f) + 240.97f));
        float D = max(0.f, (es - ea) / press * 1000.f);

        //xylem moisture potential
        float Psix = xylem_potential_default;
        if( context->doesPrimitiveDataExist(UUID,"xylem_water_potential") ){
            context->getPrimitiveData(UUID,"xylem_water_potential",Psix);
        }else{
            assumed_default_Psix++;
        }

        float gs;
        if( model == "BB" ) {

            //model coefficients
            BBcoefficients coeffs;
            if ( BBmodel_coefficients.empty() ) {
                coeffs = BBcoeffs;
            } else {
                coeffs = BBmodel_coefficients.at(UUID);
            }

            std::vector<float> variables{i,D,Psix};

            gs = fzero( evaluate_BBmodel, variables, &coeffs, 0.1f );

        }else{

            //model coefficients
            float Em;
            float i0;
            float k;
            float b;
            if ( BMFmodel_coefficients.empty() ) {
                Em = BMFcoeffs.Em;
                i0 = BMFcoeffs.i0;
                k = BMFcoeffs.k;
                b = BMFcoeffs.b;
            } else {
                BMFcoefficients coeffs = BMFmodel_coefficients.at(UUID);
                Em = coeffs.Em;
                i0 = coeffs.i0;
                k = coeffs.k;
                b = coeffs.b;
            }

            gs = Em * (i + i0) / (k + b * i + (i + i0) * D);

        }

        context->setPrimitiveData( UUID, "moisture_conductance", gs);

    }

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