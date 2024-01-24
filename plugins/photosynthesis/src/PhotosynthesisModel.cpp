/** \file "PhotosynthesisModel.cpp" Primary source file for photosynthesis plug-in.

Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "PhotosynthesisModel.h"

using namespace std;
using namespace helios;

PhotosynthesisModel::PhotosynthesisModel( helios::Context* a_context ){
    context = a_context;

    //default values set here
    model = "farquhar";

    i_PAR_default = 0;
    TL_default = 300;
    CO2_default = 390;
    gM_default = 0.25;
    gH_default = 1;


}

int PhotosynthesisModel::selfTest(){

    std::cout << "Running photosynthesis model self-test..." << std::flush;

    Context context_test;

    float errtol = 0.001f;

    uint UUID = context_test.addPatch( make_vec3(0,0,0), make_vec2(1,1) );

    PhotosynthesisModel photomodel(&context_test);

    std::vector<float> A;

    float Qin[9] = {0, 50, 100, 200, 400, 800, 1200, 1500, 2000};
    A.resize(9);
    std::vector<float> AQ_expected{-2.39479,8.30612,12.5873,16.2634,16.6826,16.6826,16.6826,16.6826,16.6826};

//Generate a light response curve using empirical model with default parmeters
    for( int i=0; i<9; i++ ){
        context_test.setPrimitiveData(UUID,"radiation_flux_PAR",Qin[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID,"net_photosynthesis",A[i]);
    }

//Generate a light response curve using Farquhar model

    FarquharModelCoefficients fcoeffs; //these are prior default model parameters, which is what was used for this test
    fcoeffs.Vcmax = 78.5;
    fcoeffs.Jmax = 150;
    fcoeffs.alpha = 0.45;
    fcoeffs.Rd = 2.12;
    fcoeffs.c_Jmax = 17.57;
    fcoeffs.dH_Jmax = 43.54;

    photomodel.setModelCoefficients(fcoeffs);

    for( int i=0; i<9; i++ ){
        context_test.setPrimitiveData(UUID,"radiation_flux_PAR",Qin[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID,"net_photosynthesis",A[i]);
        if( fabs(A.at(i)-AQ_expected.at(i))/fabs(AQ_expected.at(i))>errtol ){
            std::cout << "failed. Incorrect light response curve." << std::endl;
            return 1;
        }
    }

//Generate an A vs Ci curve using empirical model with default parameters

    float CO2[9] = {100, 200, 300, 400, 500, 600, 700, 800, 1000};
    A.resize(9);

    context_test.setPrimitiveData(UUID,"radiation_flux_PAR",Qin[8]);

    photomodel.setModelType_Empirical();
    for( int i=0; i<9; i++ ){
        context_test.setPrimitiveData(UUID,"air_CO2",CO2[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID,"net_photosynthesis",A[i]);
    }

//Generate an A vs Ci curve using Farquhar model with default parameters

    std::vector<float> ACi_expected{1.70787,7.29261,12.426,17.1353,21.4501,25.4004,28.5788,29.8179,31.5575};

    photomodel.setModelCoefficients(fcoeffs);
    for( int i=0; i<9; i++ ){
        context_test.setPrimitiveData(UUID,"air_CO2",CO2[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID,"net_photosynthesis",A[i]);
        if( fabs(A.at(i)-ACi_expected.at(i))/fabs(ACi_expected.at(i))>errtol ){
            std::cout << "failed. Incorrect CO2 response curve." << std::endl;
            return 1;
        }
    }

//Generate an A vs temperature curve using empirical model with default parameters

    float TL[7] = {270, 280, 290, 300, 310, 320, 330};
    A.resize(7);

    context_test.setPrimitiveData(UUID,"air_CO2",CO2[3]);

    photomodel.setModelType_Empirical();
    for( int i=0; i<7; i++ ){
        context_test.setPrimitiveData(UUID,"temperature",TL[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID,"net_photosynthesis",A[i]);
    }

//Generate an A vs temperature curve using Farquhar model with default parameters

    std::vector<float> AT_expected{3.8609,8.71169,14.3514,17.1353,16.0244,11.5661,3.91437};

    photomodel.setModelCoefficients(fcoeffs);
    for( int i=0; i<7; i++ ){
        context_test.setPrimitiveData(UUID,"temperature",TL[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID,"net_photosynthesis",A[i]);
        if( fabs(A.at(i)-AT_expected.at(i))/fabs(AT_expected.at(i))>errtol ){
            std::cout << "failed. Incorrect temperature response curve." << std::endl;
            return 1;
        }
    }

    std::cout << "passed." << std::endl;

    return 0;
}

void PhotosynthesisModel::setModelType_Empirical(){
    model="empirical";
}

void PhotosynthesisModel::setModelType_Farquhar(){
    model="farquhar";
}

void PhotosynthesisModel::setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients ){
    empiricalmodelcoeffs = modelcoefficients;
    empiricalmodel_coefficients.clear();
    model = "empirical";
}

void PhotosynthesisModel::setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs ){
    for( uint UUID : UUIDs){
        empiricalmodel_coefficients[UUID] = modelcoefficients;
    }
    model = "empirical";
}

void PhotosynthesisModel::setModelCoefficients(const FarquharModelCoefficients &modelcoefficients ){
    farquharmodelcoeffs = modelcoefficients;
    farquharmodel_coefficients.clear();
    model = "farquhar";
}

void PhotosynthesisModel::setModelCoefficients(const FarquharModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs ){
    for( uint UUID : UUIDs){
        farquharmodel_coefficients[UUID] = modelcoefficients;
    }
    model = "farquhar";
}

void PhotosynthesisModel::run(){
    run(context->getAllUUIDs());
}

void PhotosynthesisModel::run(const std::vector<uint> &lUUIDs ){

    for( uint UUID : lUUIDs){

        float i_PAR;
        if( context->doesPrimitiveDataExist(UUID,"radiation_flux_PAR") && context->getPrimitiveDataType(UUID,"radiation_flux_PAR")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"radiation_flux_PAR",i_PAR);
            i_PAR = i_PAR*4.57f; //umol/m^2-s (ref https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf)
            if( i_PAR<0 ){
                i_PAR = 0;
                std::cout << "WARNING (runPhotosynthesis): PAR flux value provided was negative.  Clipping to zero." << std::endl;
            }
        }else{
            i_PAR = i_PAR_default;
        }

        float TL;
        if( context->doesPrimitiveDataExist(UUID,"temperature") && context->getPrimitiveDataType(UUID,"temperature")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"temperature",TL);
            if( TL<200 ){
                std::cout << "WARNING (PhotosynthesisModel::run): Primitive temperature value was very low (" << TL << "K). Assuming. Are you using absolute temperature units?" << std::endl;
                TL = TL_default;
            }
        }else{
            TL = TL_default;
        }

        float CO2;
        if( context->doesPrimitiveDataExist(UUID,"air_CO2") && context->getPrimitiveDataType(UUID,"air_CO2")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"air_CO2",CO2);
            if( CO2<0 ){
                CO2 = 0;
                std::cout << "WARNING (PhotosynthesisModel::run): CO2 concentration value provided was negative. Clipping to zero." << std::endl;
            }
        }else{
            CO2 = CO2_default;
        }

        float gM;
        if( context->doesPrimitiveDataExist(UUID,"moisture_conductance") && context->getPrimitiveDataType(UUID,"moisture_conductance")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"moisture_conductance",gM);
            if( gM<0 ){
                gM = 0;
                std::cout << "WARNING (PhotosynthesisModel::run): Moisture conductance value provided was negative. Clipping to zero." << std::endl;
            }
        }else{
            gM = gM_default;
        }

        float gH;
        if( context->doesPrimitiveDataExist(UUID,"boundarylayer_conductance") && context->getPrimitiveDataType(UUID,"boundarylayer_conductance")==HELIOS_TYPE_FLOAT ) {
            context->getPrimitiveData(UUID, "boundarylayer_conductance", gH);
        }else if( context->doesPrimitiveDataExist(UUID,"boundarylayer_conductance_out") && context->getPrimitiveDataType(UUID,"boundarylayer_conductance_out")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(UUID,"boundarylayer_conductance_out",gH);
        }else{
            gH = gH_default;
        }
        if( gH<0 ){
            gH = 0;
            std::cout << "WARNING (PhotosynthesisModel::run): Boundary-layer conductance value provided was negative. Clipping to zero." << std::endl;
        }

        //combine stomatal (gM) and boundary-layer (gH) conductances
        gM = 1.08f*gH*gM/(1.08*gH+gM);

        float A, Ci, Gamma;
        int limitation_state;

        if( model=="farquhar" ){ //Farquhar-von Caemmerer-Berry Model

            FarquharModelCoefficients coeffs;
            if ( farquharmodel_coefficients.empty() || farquharmodel_coefficients.find(UUID)==farquharmodel_coefficients.end() ) {
                coeffs = farquharmodelcoeffs;
            } else {
                coeffs = farquharmodel_coefficients.at(UUID);
            }

            A = evaluateFarquharModel( coeffs, i_PAR, TL, CO2, gM, Ci, Gamma, limitation_state );

        }else{ //Empirical Model

            EmpiricalModelCoefficients coeffs;
            if ( empiricalmodel_coefficients.empty() || empiricalmodel_coefficients.find(UUID)==empiricalmodel_coefficients.end() ) {
                coeffs = empiricalmodelcoeffs;
            } else {
                coeffs = empiricalmodel_coefficients.at(UUID);
            }

            A = evaluateEmpiricalModel( coeffs, i_PAR, TL, CO2, gM );

        }

        if( A==0 ){
            std::cout << "WARNING (PhotosynthesisModel::run): Solution did not converge for primitive " << UUID << "." << std::endl;
        }

        context->setPrimitiveData(UUID,"net_photosynthesis",HELIOS_TYPE_FLOAT,1,&A);

        for( const auto &data : output_prim_data ){
            if( data=="Ci" && model=="farquhar" ){
                context->setPrimitiveData(UUID,"Ci",Ci);
            }else if( data=="limitation_state" && model=="farquhar" ){
                context->setPrimitiveData(UUID,"limitation_state",limitation_state);
            }else if( data=="Gamma_CO2" && model=="farquhar" ){
                context->setPrimitiveData(UUID,"Gamma_CO2",Gamma);
            }
        }

    }


}

float PhotosynthesisModel::evaluateCi_Empirical( const EmpiricalModelCoefficients &params, float Ci, float CO2, float fL, float Rd, float gM ) const{


//--- CO2 Response Function --- //

    float fC = params.kC*Ci/params.Ci_ref;


//--- Assimilation Rate --- //

    float A = params.Asat*fL*fC-Rd;

//--- Calculate error and update --- //

    float resid = 0.75f*gM*(CO2-Ci) - A;


    return resid;

}

float PhotosynthesisModel::evaluateEmpiricalModel( const EmpiricalModelCoefficients &params, float i_PAR, float TL, float CO2, float gM ){

//initial guess for intercellular CO2
    float Ci = CO2;

//--- Light Response Function --- //

    float fL = i_PAR/(params.theta+i_PAR);

    assert( fL>=0 && fL<=1 );

//--- Respiration Rate --- //

    float Rd = params.R*sqrt(TL-273.f)*exp(-params.ER/TL);

    float Ci_old = Ci;
    float Ci_old_old = 0.95f*Ci;

    float resid_old = evaluateCi_Empirical( params, Ci_old, CO2, fL, Rd, gM );
    float resid_old_old = evaluateCi_Empirical( params, Ci_old_old, CO2, fL, Rd, gM );

    float err = 10000, err_max = 0.01;
    int iter = 0, max_iter = 100;
    float resid;
    while( err>err_max && iter<max_iter ){

        if( resid_old==resid_old_old ){//this condition will cause NaN
            break;
        }

        Ci = fabs((Ci_old_old*resid_old-Ci_old*resid_old_old)/(resid_old-resid_old_old));

        resid = evaluateCi_Empirical( params, Ci, CO2, fL, Rd, gM );

        resid_old_old = resid_old;
        resid_old = resid;

        err = fabs(resid);

        Ci_old_old = Ci_old;
        Ci_old = Ci;

        iter++;

    }

    float A;
    if( err>err_max ){
        A = 0;
    }else{
        float fC = params.kC*Ci/params.Ci_ref;
        A = params.Asat*fL*fC-Rd;
    }

    return A;

}

float PhotosynthesisModel::evaluateCi_Farquhar( float Ci, std::vector<float> &variables, const void* parameters ){

    const FarquharModelCoefficients modelcoeffs = *reinterpret_cast<const FarquharModelCoefficients*>(parameters);

    float CO2 = variables[0];
    float i_PAR = variables[1];
    float TL = variables[2];
    float gM = variables[3];

//molar gas constant (kJ/K/mol)
    float R = 0.0083144598;

    float Rd = modelcoeffs.Rd*exp(modelcoeffs.c_Rd-modelcoeffs.dH_Rd/(R*TL));
    float Vcmax = modelcoeffs.Vcmax*exp(modelcoeffs.c_Vcmax-modelcoeffs.dH_Vcmax/(R*TL));
    float Jmax = modelcoeffs.Jmax*exp(modelcoeffs.c_Jmax-modelcoeffs.dH_Jmax/(R*TL));

    float Gamma_star = exp(modelcoeffs.c_Gamma-modelcoeffs.dH_Gamma/(R*TL));
    float Kc = exp(modelcoeffs.c_Kc-modelcoeffs.dH_Kc/(R*TL));
    float Ko = exp(modelcoeffs.c_Ko-modelcoeffs.dH_Ko/(R*TL));

    float Kco = Kc*(1.f+modelcoeffs.O/Ko);

    float Wc = Vcmax*Ci/(Ci+Kco);

    float J = Jmax*i_PAR*modelcoeffs.alpha/(i_PAR*modelcoeffs.alpha+Jmax);
    float Wj = J*Ci/(4.f*Ci+8.f*Gamma_star);

    float A = (1-Gamma_star/Ci)*fmin(Wc,Wj)-Rd;

    float limitation_state;
    if( Wj>Wc ){ //Rubisco limited
        limitation_state = 0;
    }else{ //Electron transport limited
        limitation_state = 1;
    }

//--- Calculate error and update --- //

    float resid = 0.75f*gM*(CO2-Ci) - A;

    variables[4] = A;
    variables[5] = limitation_state;

    float Gamma = (Gamma_star+Kco*Rd/Vcmax)/(1-Rd/Vcmax);  //Equation 39 of Farquhar et al. (1980)
    variables[6] = Gamma;

    return resid;

}

float PhotosynthesisModel::evaluateFarquharModel( const FarquharModelCoefficients &params, float i_PAR, float TL, float CO2, float gM, float& Ci, float& Gamma, int& limitation_state ){

    float A = 0;
    Ci = 100;

    std::vector<float> variables{CO2, i_PAR, TL, gM, A, float(limitation_state), Gamma};

    Ci = fzero( evaluateCi_Farquhar, variables, &farquharmodelcoeffs, Ci );

    A = variables[4];
    limitation_state = (int)variables[5];
    Gamma = variables[6];

    return A;

}

EmpiricalModelCoefficients PhotosynthesisModel::getEmpiricalModelCoefficients( uint UUID ){

    EmpiricalModelCoefficients coeffs;
    if ( empiricalmodel_coefficients.empty() || empiricalmodel_coefficients.find(UUID)==empiricalmodel_coefficients.end() ) {
        coeffs = empiricalmodelcoeffs;
    } else {
        coeffs = empiricalmodel_coefficients.at(UUID);
    }

    return coeffs;

}

FarquharModelCoefficients PhotosynthesisModel::getFarquharModelCoefficients( uint UUID ){

    FarquharModelCoefficients coeffs;
    if ( farquharmodel_coefficients.empty() || farquharmodel_coefficients.find(UUID)==farquharmodel_coefficients.end() ) {
        coeffs = farquharmodelcoeffs;
    } else {
        coeffs = farquharmodel_coefficients.at(UUID);
    }

    return coeffs;

}

void PhotosynthesisModel::optionalOutputPrimitiveData( const char* label ){

    if( strcmp(label,"Ci")==0 || strcmp(label,"limitation_state")==0 || strcmp(label,"Gamma_CO2")==0 ){
        output_prim_data.emplace_back(label );
    }else{
        std::cout << "WARNING (PhotosynthesisModel::optionalOutputPrimitiveData): unknown output primitive data " << label << std::endl;
    }

}

void PhotosynthesisModel::printDefaultValueReport() const{
    printDefaultValueReport(context->getAllUUIDs());
}

void PhotosynthesisModel::printDefaultValueReport(const std::vector<uint> &UUIDs) const {

//    i_PAR_default = 0;
//    TL_default = 300;
//    CO2_default = 390;
//    gM_default = 0.25;
//    gH_default = 1;

    size_t assumed_default_i = 0;
    size_t assumed_default_TL = 0;
    size_t assumed_default_CO2 = 0;
    size_t assumed_default_gM = 0;
    size_t assumed_default_gH = 0;

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

        //boundary-layer conductance to heat
        if ((!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") || context->getPrimitiveDataType(UUID, "boundarylayer_conductance") != HELIOS_TYPE_FLOAT) &&
            (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out") || context->getPrimitiveDataType(UUID, "boundarylayer_conductance_out") != HELIOS_TYPE_FLOAT)) {
            assumed_default_gH++;
        }

        //stomatal conductance
        if (!context->doesPrimitiveDataExist(UUID, "moisture_conductance") || context->getPrimitiveDataType(UUID, "moisture_conductance") != HELIOS_TYPE_FLOAT ) {
            assumed_default_gM++;
        }

        //ambient air CO2
        if (!context->doesPrimitiveDataExist(UUID, "air_CO2") ||
            context->getPrimitiveDataType(UUID, "air_CO2") != HELIOS_TYPE_FLOAT) {
            assumed_default_CO2++;
        }

    }

    std::cout << "--- Photosynthesis Model Default Value Report ---" << std::endl;

    std::cout << "PAR flux: " << assumed_default_i << " of " << Nprimitives << " used default value of " << i_PAR_default << " because ""radiation_flux_PAR"" primitive data did not exist" << std::endl;
    std::cout << "surface temperature: " << assumed_default_TL << " of " << Nprimitives << " used default value of " << TL_default << " because ""temperature"" primitive data did not exist" << std::endl;
    std::cout << "boundary-layer conductance: " << assumed_default_gH << " of " << Nprimitives << " used default value of " << gH_default << " because ""boundarylayer_conductance"" primitive data did not exist" << std::endl;
    std::cout << "moisture conductance: " << assumed_default_gM << " of " << Nprimitives << " used default value of " << gM_default << " because ""moisture_conductance"" primitive data did not exist" << std::endl;
    std::cout << "air CO2: " << assumed_default_CO2 << " of " << Nprimitives << " used default value of " << CO2_default << " because ""air_CO2"" primitive data did not exist" << std::endl;

    std::cout << "--------------------------------------------------" << std::endl;

}
