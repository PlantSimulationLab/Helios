/** \file "selfTest.cpp" Source file for radiation transport model unit tests.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RadiationModel.h"

using namespace helios;

int RadiationModel::selfTest(){

    std::cout << "Running radiation model self-test..." << std::endl;

    float error_threshold = 0.005;
    bool failure = false;

    int Nensemble = 500;

    float shortwave_rho, shortwave_tau, longwave_rho, eps;
    float shortwave_exact_0, shortwave_model_0, shortwave_error_0, shortwave_exact_1, shortwave_model_1, shortwave_error_1;
    float longwave_exact_0, longwave_model_0, longwave_error_0, longwave_exact_1, longwave_model_1, longwave_error_1;

    float a, b, c, X, X2, Y, Y2, F12;
    float R;
    float theta_s, dtheta;

    uint UUID0, UUID1, UUID2;
    uint ID, N;

    //----- Test #1: 90 degree common-edge squares ------//

    std::cout << "Test #1: 90 degree common-edge squares..." << std::flush;

    uint Ndiffuse_1 = 100000;
    uint Ndirect_1 = 5000;

    float Qs = 1000.f;

    shortwave_exact_0 = 0.7f*Qs;
    shortwave_exact_1 = 0.3f*0.2f*Qs;
    longwave_exact_0 = 0.f;//sigma*powf(300.f,4)*0.2f;
    longwave_exact_1 = sigma*powf(300.f,4)*0.2f;

    Context context_1;
    UUID0 = context_1.addPatch( make_vec3(0,0,0), make_vec2(1,1) );
    UUID1 = context_1.addPatch( make_vec3(0.5,0,0.5), make_vec2(1,1), make_SphericalCoord(0.5*M_PI,-0.5*M_PI) );

    uint ts_flag = 0;
    context_1.setPrimitiveData(UUID0,"twosided_flag",ts_flag);
    context_1.setPrimitiveData(UUID1,"twosided_flag",ts_flag);

    context_1.setPrimitiveData(0,"temperature",300.f);
    context_1.setPrimitiveData(1,"temperature",0.f);

    shortwave_rho = 0.3f;
    context_1.setPrimitiveData(0,"reflectivity_SW",HELIOS_TYPE_FLOAT,1,&shortwave_rho);

    RadiationModel radiationmodel_1(&context_1);
    radiationmodel_1.disableMessages();

    //Longwave band
    radiationmodel_1.addRadiationBand("LW");
    radiationmodel_1.setDirectRayCount("LW",Ndiffuse_1);
    radiationmodel_1.setDiffuseRayCount("LW",Ndiffuse_1);
    radiationmodel_1.setScatteringDepth("LW",0);

    //Shortwave band
    uint SunSource_1 = radiationmodel_1.addCollimatedRadiationSource( make_vec3(0,0,1) );
    radiationmodel_1.addRadiationBand("SW");
    radiationmodel_1.disableEmission("SW");
    radiationmodel_1.setDirectRayCount("SW",Ndirect_1);
    radiationmodel_1.setDiffuseRayCount("SW",Ndirect_1);
    radiationmodel_1.setScatteringDepth("SW",1);
    radiationmodel_1.setSourceFlux(SunSource_1,"SW",Qs);

    radiationmodel_1.updateGeometry();

    longwave_model_0 = 0.f;
    longwave_model_1 = 0.f;
    shortwave_model_0 = 0.f;
    shortwave_model_1 = 0.f;

    for( int r=0; r<Nensemble; r++ ){

        std::vector<std::string> bands{"LW","SW"};
        radiationmodel_1.runBand(bands);
//    radiationmodel_1.runBand("SW");

        //patch 0 emission
        context_1.getPrimitiveData(0,"radiation_flux_LW",R);
        longwave_model_0 += R/float(Nensemble);
        //patch 1 emission
        context_1.getPrimitiveData(1,"radiation_flux_LW",R);
        longwave_model_1 += R/float(Nensemble);

        //patch 0 shortwave
        context_1.getPrimitiveData(0,"radiation_flux_SW",R);
        shortwave_model_0 += R/float(Nensemble);
        //patch 1 shortwave
        context_1.getPrimitiveData(1,"radiation_flux_SW",R);
        shortwave_model_1 += R/float(Nensemble);

    }

    longwave_error_0 = fabsf(longwave_model_0-longwave_exact_0)/fabsf(longwave_exact_0);
    longwave_error_1 = fabsf(longwave_model_1-longwave_exact_1)/fabsf(longwave_exact_1);

    shortwave_error_0 = fabsf(shortwave_model_0-shortwave_exact_0)/fabsf(shortwave_exact_0);
    shortwave_error_1 = fabsf(shortwave_model_1-shortwave_exact_1)/fabsf(shortwave_exact_1);

    bool failure_0 = false;
    if( shortwave_error_0 > error_threshold || shortwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for shortwave radiation. Patch #0 error: " << shortwave_error_0 << ", Patch #1 error: " << shortwave_error_1 << std::endl;
        failure_0 = true;
        failure = true;
    }
    if( longwave_error_0 > error_threshold || longwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for radiation emission. Error: " << longwave_error_0 << " " << longwave_error_1 << std::endl;
        failure_0 = true;
        failure = true;
    }
    if( !failure_0 ){
        std::cout << "passed." << std::endl;
    }

    //----- Test #2: Parallel rectangles ------//

    std::cout << "Test #2: Black parallel rectangles..." << std::flush;

    uint Ndiffuse_2 = 50000;

    a = 1;
    b = 2;
    c = 0.5;

    X = a/c;
    Y = b/c;
    X2 = X*X;
    Y2 = Y*Y;

    F12 = 2.0f/float(M_PI*X*Y)*(logf(std::sqrt((1.f+X2)*(1.f+Y2)/(1.f+X2+Y2)))+X*std::sqrt(1.f+Y2)*atanf(X/std::sqrt(1.f+Y2))+Y*std::sqrt(1.f+X2)*atanf(Y/std::sqrt(1.f+X2))-X*atanf(X)-Y*atanf(Y));

    shortwave_exact_0 = (1.f-F12);
    shortwave_exact_1 = (1.f-F12);

    Context context_2;
    context_2.addPatch( make_vec3(0,0,0), make_vec2(a,b) );
    context_2.addPatch( make_vec3(0,0,c), make_vec2(a,b), make_SphericalCoord(M_PI,0.f) );

    uint flag = 0;
    context_2.setPrimitiveData(0,"twosided_flag",flag);
    context_2.setPrimitiveData(1,"twosided_flag",flag);

    RadiationModel radiationmodel_2(&context_2);
    radiationmodel_2.disableMessages();

    //Shortwave band
    radiationmodel_2.addRadiationBand("SW");
    radiationmodel_2.disableEmission("SW");
    radiationmodel_2.setDiffuseRayCount("SW",Ndiffuse_2);
    radiationmodel_2.setDiffuseRadiationFlux("SW",1.f);
    radiationmodel_2.setScatteringDepth("SW",0);

    radiationmodel_2.updateGeometry();

    shortwave_model_0 = 0.f;
    shortwave_model_1 = 0.f;

    for( int r=0; r<Nensemble; r++ ){

        radiationmodel_2.runBand("SW");

        context_2.getPrimitiveData(0,"radiation_flux_SW",R);
        shortwave_model_0 += R/float(Nensemble);
        context_2.getPrimitiveData(1,"radiation_flux_SW",R);
        shortwave_model_1 += R/float(Nensemble);

    }

    shortwave_error_0 = fabsf(shortwave_model_0-shortwave_exact_0)/fabsf(shortwave_exact_0);
    shortwave_error_1 = fabsf(shortwave_model_1-shortwave_exact_1)/fabsf(shortwave_exact_1);

    if( shortwave_error_0 > error_threshold || shortwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for shortwave radiation. Patch #0 error: " << shortwave_error_0 << ", Patch #1 error: " << shortwave_error_1 << std::endl;
        failure = true;
    }else{
        std::cout << "passed." << std::endl;
    }

    // ------ Test #3: Gray Parallel Rectangles ------- //

    std::cout << "Test #3: Gray parallel rectangles..." << std::flush;

    uint Ndiffuse_3 = 100000;

    uint Nscatter_3 = 5;

    longwave_rho = 0.4;
    eps = 0.6f;

    float T0 = 300.f;
    float T1 = 300.f;

    a = 1;
    b = 2;
    c = 0.5;

    X = a/c;
    Y = b/c;
    X2 = X*X;
    Y2 = Y*Y;

    F12 = 2.0f/float(M_PI*X*Y)*(logf(std::sqrt((1.f+X2)*(1.f+Y2)/(1.f+X2+Y2)))+X*std::sqrt(1.f+Y2)*atanf(X/std::sqrt(1.f+Y2))+Y*std::sqrt(1.f+X2)*atanf(Y/std::sqrt(1.f+X2))-X*atanf(X)-Y*atanf(Y));

    longwave_exact_0 = (eps*(1.f/eps-1.f)*F12*sigma*(powf(T1,4)-F12*powf(T0,4))+sigma*(powf(T0,4)-F12*powf(T1,4)))/(1.f/eps-(1.f/eps-1.f)*F12*eps*(1/eps-1)*F12)-eps*sigma*powf(T0,4);
    longwave_exact_1 =  fabsf(eps*((1/eps-1)*F12*(longwave_exact_0+eps*sigma*powf(T0,4))+sigma*(powf(T1,4)-F12*powf(T0,4)))-eps*sigma*powf(T1,4));
    longwave_exact_0 = fabsf(longwave_exact_0);

    Context context_3;
    context_3.addPatch( make_vec3(0,0,0), make_vec2(a,b) );
    context_3.addPatch( make_vec3(0,0,c), make_vec2(a,b), make_SphericalCoord(M_PI,0.f) );

    context_3.setPrimitiveData(0,"temperature",T0);
    context_3.setPrimitiveData(1,"temperature",T1);

    context_3.setPrimitiveData(0,"emissivity_LW",eps);
    context_3.setPrimitiveData(0,"reflectivity_LW",longwave_rho);
    context_3.setPrimitiveData(1,"emissivity_LW",eps);
    context_3.setPrimitiveData(1,"reflectivity_LW",longwave_rho);

    flag = 0;
    context_3.setPrimitiveData(0,"twosided_flag",flag);
    context_3.setPrimitiveData(1,"twosided_flag",flag);

    RadiationModel radiationmodel_3(&context_3);
    radiationmodel_3.disableMessages();

    //Longwave band
    radiationmodel_3.addRadiationBand("LW");
    radiationmodel_3.setDirectRayCount("LW",Ndiffuse_3);
    radiationmodel_3.setDiffuseRayCount("LW",Ndiffuse_3);
    radiationmodel_3.setDiffuseRadiationFlux("LW",0.f);
    radiationmodel_3.setScatteringDepth("LW",Nscatter_3);

    radiationmodel_3.updateGeometry();

    longwave_model_0 = 0.f;
    longwave_model_1 = 0.f;

    for( int r=0; r<Nensemble; r++ ){

        radiationmodel_3.runBand("LW");

        context_3.getPrimitiveData(0,"radiation_flux_LW",R);
        longwave_model_0 += R/float(Nensemble);
        context_3.getPrimitiveData(1,"radiation_flux_LW",R);
        longwave_model_1 += R/float(Nensemble);


    }

    longwave_error_0 = fabsf(longwave_exact_0-longwave_model_0)/fabsf(longwave_exact_0);
    longwave_error_1 = fabsf(longwave_exact_1-longwave_model_1)/fabsf(longwave_exact_1);

    if( longwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for radiation emission. Error: " << longwave_error_1 << std::endl;
        failure_0 = true;
        failure = true;
    }else{
        std::cout << "passed." << std::endl;
    }

    // ------ Test #4: Sphere source ------- //

    std::cout << "Test #4: Sphere source..." << std::flush;

    uint Ndirect_4 = 10000;

    float r = 0.5;
    float d = 0.75f;
    float l1 = 1.5f;
    float l2 = 2.f;

    float D1 = d/l1;
    float D2 = d/l2;

    F12 = 0.25f/float(M_PI)*atanf(sqrtf(1.f/(D1*D1+D2*D2+D1*D1*D2*D2)));

    shortwave_exact_0 = 4.0f*float(M_PI)*r*r*F12/(l1*l2);

    Context context_4;

    context_4.addPatch( make_vec3(0.5f*l1,0.5f*l2,0), make_vec2(l1,l2) );

    RadiationModel radiationmodel_4(&context_4);
    radiationmodel_4.disableMessages();

    uint Source_4 = radiationmodel_4.addSphereRadiationSource( make_vec3(0,0,d), r );

    //Shortwave band
    radiationmodel_4.addRadiationBand("SW");
    radiationmodel_4.disableEmission("SW");
    radiationmodel_4.setDirectRayCount("SW",Ndirect_4);
    radiationmodel_4.setSourceFlux(Source_4,"SW",1.f);
    radiationmodel_4.setScatteringDepth("SW",0);

    radiationmodel_4.updateGeometry();

    shortwave_model_0 = 0.f;

    for( int i=0; i<Nensemble; i++ ){

        radiationmodel_4.runBand("SW");

        context_4.getPrimitiveData(0,"radiation_flux_SW",R);
        shortwave_model_0 += R/float(Nensemble);

    }

    shortwave_error_0 = fabsf(shortwave_exact_0-shortwave_model_0)/fabsf(shortwave_exact_0);

    if( shortwave_error_0 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for shortwave radiation. Patch #0 error: " << shortwave_error_0 << std::endl;
        failure = true;
    }else{
        std::cout << "passed." << std::endl;
    }

    //----- Test #5: 90 degree common-edge sub-triangles ------//

    std::cout << "Test #5: 90 degree common-edge sub-triangles..." << std::flush;

    Qs = 1000.f;

    uint Ndiffuse_5 = 100000;
    uint Ndirect_5 = 5000;

    shortwave_exact_0 = 0.7f*Qs;
    shortwave_exact_1 = 0.3f*0.2f*Qs;
    longwave_exact_0 = 0.f;
    longwave_exact_1 = sigma*powf(300.f,4)*0.2f;

    Context context_5;

    context_5.addTriangle( make_vec3(-0.5,-0.5,0), make_vec3(0.5,-0.5,0), make_vec3(0.5,0.5,0) );
    context_5.addTriangle( make_vec3(-0.5,-0.5,0), make_vec3(0.5,0.5,0), make_vec3(-0.5,0.5,0) );

    context_5.addTriangle( make_vec3(0.5,0.5,0), make_vec3(0.5,-0.5,0), make_vec3(0.5,-0.5,1) );
    context_5.addTriangle( make_vec3(0.5,0.5,0), make_vec3(0.5,-0.5,1), make_vec3(0.5,0.5,1) );

    context_5.setPrimitiveData(0,"temperature",300.f);
    context_5.setPrimitiveData(1,"temperature",300.f);
    context_5.setPrimitiveData(2,"temperature",0.f);
    context_5.setPrimitiveData(3,"temperature",0.f);

    shortwave_rho = 0.3f;
    context_5.setPrimitiveData(0,"reflectivity_SW",shortwave_rho);
    context_5.setPrimitiveData(1,"reflectivity_SW",shortwave_rho);

    flag = 0;
    context_5.setPrimitiveData(0,"twosided_flag",flag);
    context_5.setPrimitiveData(1,"twosided_flag",flag);
    context_5.setPrimitiveData(2,"twosided_flag",flag);
    context_5.setPrimitiveData(3,"twosided_flag",flag);

    RadiationModel radiationmodel_5(&context_5);
    radiationmodel_5.disableMessages();

    //Longwave band
    radiationmodel_5.addRadiationBand("LW");
    radiationmodel_5.setDirectRayCount("LW",Ndiffuse_5);
    radiationmodel_5.setDiffuseRayCount("LW",Ndiffuse_5);
    radiationmodel_5.setScatteringDepth("LW",0);

    //Shortwave band
    uint SunSource_5 = radiationmodel_5.addCollimatedRadiationSource( make_vec3(0,0,1) );
    radiationmodel_5.addRadiationBand("SW");
    radiationmodel_5.disableEmission("SW");
    radiationmodel_5.setDirectRayCount("SW",Ndirect_5);
    radiationmodel_5.setDiffuseRayCount("SW",Ndirect_5);
    radiationmodel_5.setScatteringDepth("SW",1);
    radiationmodel_5.setSourceFlux(SunSource_5,"SW",Qs);

    radiationmodel_5.updateGeometry();

    longwave_model_0 = 0.f;
    longwave_model_1 = 0.f;
    shortwave_model_0 = 0.f;
    shortwave_model_1 = 0.f;

    for( int i=0; i<Nensemble; i++ ){

        std::vector<std::string> bands{"SW","LW"};
        radiationmodel_5.runBand(bands);

//    radiationmodel_5.runBand("LW");

        //patch 0 emission
        context_5.getPrimitiveData(0,"radiation_flux_LW",R);
        longwave_model_0 += 0.5f*R/float(Nensemble);
        context_5.getPrimitiveData(1,"radiation_flux_LW",R);
        longwave_model_0 += 0.5f*R/float(Nensemble);
        //patch 1 emission
        context_5.getPrimitiveData(2,"radiation_flux_LW",R);
        longwave_model_1 += 0.5f*R/float(Nensemble);
        context_5.getPrimitiveData(3,"radiation_flux_LW",R);
        longwave_model_1 += 0.5f*R/float(Nensemble);

//    radiationmodel_5.runBand("SW");

        //patch 0 shortwave
        context_5.getPrimitiveData(0,"radiation_flux_SW",R);
        shortwave_model_0 += 0.5f*R/float(Nensemble);
        context_5.getPrimitiveData(1,"radiation_flux_SW",R);
        shortwave_model_0 += 0.5f*R/float(Nensemble);
        //patch 1 shortwave
        context_5.getPrimitiveData(2,"radiation_flux_SW",R);
        shortwave_model_1 += 0.5f*R/float(Nensemble);
        context_5.getPrimitiveData(3,"radiation_flux_SW",R);
        shortwave_model_1 += 0.5f*R/float(Nensemble);

    }

    longwave_error_0 = fabsf(longwave_model_0-longwave_exact_0)/fabsf(longwave_exact_0);
    longwave_error_1 = fabsf(longwave_model_1-longwave_exact_1)/fabsf(longwave_exact_1);

    shortwave_error_0 = fabsf(shortwave_model_0-shortwave_exact_0)/fabsf(shortwave_exact_0);
    shortwave_error_1 = fabsf(shortwave_model_1-shortwave_exact_1)/fabsf(shortwave_exact_1);

    bool failure_5 = false;
    if( shortwave_error_0 > error_threshold || shortwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for shortwave radiation. Patch #0 error: " << shortwave_error_0 << ", Patch #1 error: " << shortwave_error_1 << std::endl;
        failure_5 = true;
        failure = true;
    }
    if( longwave_error_0 > error_threshold || longwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for radiation emission. Error: " << longwave_error_0 << " " << longwave_error_1 << std::endl;
        std::cout << longwave_exact_0 << std::endl;
        failure_5 = true;
        failure = true;
    }
    if( !failure_5 ){
        std::cout << "passed." << std::endl;
    }

    // ------ Test #6: Parallel Disks (Texture Masked Patches) ------- //

    std::cout << "Test #6: Parallel disks (Texture Masked Patches)..." << std::flush;

    uint Ndirect_6 = 100;
    uint Ndiffuse_6 = 500000;

    shortwave_rho = 0.3;

    float r1 = 1.f;
    float r2 = 0.5f;
    float h = 0.75f;

    float A1 = M_PI*r1*r1;
    float A2 = M_PI*r2*r2;

    float R1 = r1/h;
    float R2 = r2/h;

    X = 1.f+(1.f+R2*R2)/(R1*R1);
    F12 = 0.5f*(X-std::sqrt(X*X-4.f*powf(R2/R1,2)));

    shortwave_exact_0 = (A1-A2)/A1*(1.f-shortwave_rho);
    shortwave_exact_1 = (A1-A2)/A1*F12*A1/A2*shortwave_rho;
    longwave_exact_0 = sigma*powf(300.f,4)*F12;
    longwave_exact_1 = sigma*powf(300.f,4)*F12*A1/A2;

    Context context_6;

    context_6.addPatch( make_vec3(0,0,0), make_vec2(2.f*r1,2.f*r1), make_SphericalCoord(0,0), "plugins/radiation/disk.png" );
    context_6.addPatch( make_vec3(0,0,h), make_vec2(2.f*r2,2.f*r2), make_SphericalCoord(M_PI,0), "plugins/radiation/disk.png" );
    context_6.addPatch( make_vec3(0,0,h+0.01f), make_vec2(2.f*r2,2.f*r2), make_SphericalCoord(M_PI,0), "plugins/radiation/disk.png" );

    context_6.setPrimitiveData(0,"reflectivity_SW",shortwave_rho);

    context_6.setPrimitiveData(0,"temperature",300.f);
    context_6.setPrimitiveData(1,"temperature",300.f);

    flag = 0;
    context_6.setPrimitiveData(0,"twosided_flag",flag);
    context_6.setPrimitiveData(1,"twosided_flag",flag);
    context_6.setPrimitiveData(2,"twosided_flag",flag);

    RadiationModel radiationmodel_6(&context_6);
    radiationmodel_6.disableMessages();

    uint SunSource_6 = radiationmodel_6.addCollimatedRadiationSource( make_vec3(0,0,1) );

    //Shortwave band
    radiationmodel_6.addRadiationBand("SW");
    radiationmodel_6.disableEmission("SW");
    radiationmodel_6.setDirectRayCount("SW",Ndirect_6);
    radiationmodel_6.setDiffuseRayCount("SW",Ndiffuse_6);
    radiationmodel_6.setSourceFlux(SunSource_6,"SW",1.f);
    radiationmodel_6.setDiffuseRadiationFlux("SW",0);
    radiationmodel_6.setScatteringDepth("SW",1);

    //Longwave band
    radiationmodel_6.addRadiationBand("LW");
    radiationmodel_6.setDiffuseRayCount("LW",Ndiffuse_6);
    radiationmodel_6.setDiffuseRadiationFlux("LW",0.f);
    radiationmodel_6.setScatteringDepth("LW",0);

    radiationmodel_6.updateGeometry();

    shortwave_model_0 = 0;
    shortwave_model_1 = 0;
    longwave_model_0 = 0;
    longwave_model_1 = 0;

    for( uint i=0; i<Nensemble; i++ ){

        radiationmodel_6.runBand("SW");

        radiationmodel_6.runBand("LW");

        context_6.getPrimitiveData(0,"radiation_flux_SW",R);
        shortwave_model_0 += R/float(Nensemble);

        context_6.getPrimitiveData(1,"radiation_flux_SW",R);
        shortwave_model_1 += R/float(Nensemble);

        context_6.getPrimitiveData(0,"radiation_flux_LW",R);
        longwave_model_0 += R/float(Nensemble);

        context_6.getPrimitiveData(1,"radiation_flux_LW",R);
        longwave_model_1 += R/float(Nensemble);

    }

    shortwave_error_0 = fabsf(shortwave_exact_0-shortwave_model_0)/fabsf(shortwave_exact_0);
    shortwave_error_1 = fabsf(shortwave_exact_1-shortwave_model_1)/fabsf(shortwave_exact_1);
    longwave_error_0 = fabsf(longwave_exact_0-longwave_model_0)/fabsf(longwave_exact_0);
    longwave_error_1 = fabsf(longwave_exact_1-longwave_model_1)/fabsf(longwave_exact_1);

    bool failure_6 = false;
    if( shortwave_error_0 > error_threshold || shortwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for shortwave radiation. Patch #0 error: " << shortwave_error_0 << ", Patch #1 error: " << shortwave_error_1 << std::endl;
        failure_6 = true;
        failure = true;
    }
    if( longwave_error_0 > error_threshold || longwave_error_1 > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for radiation emission. Error: " << longwave_error_0 << " " << longwave_error_1 << std::endl;
        failure_6 = true;
        failure = true;
    }
    if( !failure_6 ){
        std::cout << "passed." << std::endl;
    }

    // ------ Test #7: Second Law (Equilibrium) Test ------- //

    std::cout << "Test #7: Second Law (Equilibrium) Test..." << std::flush;

    uint Ndiffuse_7 = 50000;

    float eps1_7 = 0.8f;
    float eps2_7 = 1.f;

    float T = 300.f;

    Context context_7;

    uint objID_7 = context_7.addBoxObject( make_vec3(0,0,0), make_vec3(10,10,10),make_int3(5,5,5), RGB::black, true );
    std::vector<uint> UUIDt = context_7.getObjectPointer(objID_7)->getPrimitiveUUIDs();

    flag = 0;
    context_7.setPrimitiveData( UUIDt, "twosided_flag", flag );
    context_7.setPrimitiveData( UUIDt, "emissivity_LW", eps1_7 );
    context_7.setPrimitiveData( UUIDt, "reflectivity_LW", 1.f-eps1_7 );

    context_7.setPrimitiveData(UUIDt,"temperature",T);

    RadiationModel radiationmodel_7(&context_7);
    radiationmodel_7.disableMessages();

    //Longwave band
    radiationmodel_7.addRadiationBand("LW");
    radiationmodel_7.setDiffuseRayCount("LW",Ndiffuse_7);
    radiationmodel_7.setDiffuseRadiationFlux("LW",0);
    radiationmodel_7.setScatteringDepth("LW",5);

    radiationmodel_7.updateGeometry();

    radiationmodel_7.runBand("LW");

    float flux_err = 0.f;
    for( int p=0; p<UUIDt.size(); p++ ){
        context_7.getPrimitiveData( UUIDt.at(p), "radiation_flux_LW", R );
        flux_err += fabsf( R - eps1_7*sigma*powf(300,4) )/(eps1_7*sigma*powf(300,4))/float(UUIDt.size());
    }

    bool failure_7 = false;
    if( flux_err > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for equilibrium emission with constant emissivity. error: " << flux_err << std::endl;
        failure_7 = true;
        failure = true;
    }


    for( uint p : UUIDt){
        float emissivity;
        if( context_7.randu()<0.5f ){
            emissivity = eps1_7;
        }else{
            emissivity = eps2_7;
        }
        context_7.setPrimitiveData( p, "emissivity_LW", emissivity );
        context_7.setPrimitiveData( p, "reflectivity_LW", 1.f-emissivity );
    }

    radiationmodel_7.updateGeometry();
    radiationmodel_7.runBand("LW");

    flux_err = 0.f;
    for( int p=0; p<UUIDt.size(); p++ ){
        context_7.getPrimitiveData( UUIDt.at(p), "radiation_flux_LW", R );
        float emissivity;
        context_7.getPrimitiveData( UUIDt.at(p), "emissivity_LW", emissivity );
        flux_err += fabsf( R - emissivity*sigma*powf(300,4) )/(emissivity*sigma*powf(300,4))/float(UUIDt.size());
    }

    if( flux_err > error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for equilibrium emission with constant emissivity. error: " << flux_err << std::endl;
        failure_7 = true;
        failure = true;
    }

    if( !failure_7 ){
        std::cout << "passed." << std::endl;
    }

    // ------ Test #8: Texture-Mapping Tests ------- //

    std::cout << "Test #8: Texture-mapping..." << std::flush;

    float F0, F1, F2;

    bool failure_8 = false;

    Context context_8;

    RadiationModel radiation(&context_8);

    uint source = radiation.addCollimatedRadiationSource( make_vec3(0,0,1) );

    radiation.addRadiationBand("SW");

    radiation.setDirectRayCount("SW",10000);
    radiation.disableEmission("SW");
    radiation.disableMessages();

    radiation.setSourceFlux( source, "SW", 1.f );

    vec2 sz( 4, 2 );

    vec3 p0( 3, 4, 2 );

    vec3 p1 = p0 + make_vec3(0,0,2.4);

    // 8a, texture-mapped ellipse patch above rectangle

    UUID0 = context_8.addPatch( p0, sz );
    UUID1 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png" );

    radiation.updateGeometry();

    radiation.runBand("SW");

    context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
    context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );

    if( fabs(F0-(1.f-0.25f*M_PI))>error_threshold || fabsf(F1-1.f)>error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for texture-mapped patch (a)." << std::endl;
        failure_8 = true;
        failure=true;
    }

    // 8b, texture-mapped (u,v) inscribed ellipse tile object above rectangle

    context_8.deletePrimitive(UUID1);

    uint objID_8 = context_8.addTileObject( p1, sz, make_SphericalCoord(0,0), make_int2(5,4), "lib/images/disk_texture.png" );
    std::vector<uint> UUIDs1 = context_8.getObjectPointer(objID_8)->getPrimitiveUUIDs();

    radiation.updateGeometry();

    radiation.runBand("SW");

    context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );

    F1=0;
    float A = 0;
    for( uint p : UUIDs1){

        float area = context_8.getPrimitiveArea(p);
        A+=area;

        float Rflux;
        context_8.getPrimitiveData( p, "radiation_flux_SW", Rflux );
        F1 += Rflux*area;

    }
    F1 = F1/A;

    for( uint p=0; p<UUIDs1.size(); p++ ){
        float R;
        context_8.getPrimitiveData( UUIDs1.at(p), "radiation_flux_SW", R );
        if( fabs(R-1.f)>error_threshold ){
            failure_8=true;
        }
    }

    if( fabs(F0-(1.f-0.25f*M_PI))>error_threshold || fabsf(F1-1.f)>error_threshold || failure_8 ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for texture-mapped patch (b)." << std::endl;
        failure_8 = true;
        failure=true;
    }

    context_8.deleteObject(objID_8);

    // 8c, texture-mapped (u,v) inscribed ellipse patch above rectangle

    UUID1 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png", make_vec2(0.5,0.5), make_vec2(0.5,0.5) );

    radiation.updateGeometry();

    radiation.runBand("SW");

    context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
    context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );

    if( fabsf(F0)>error_threshold || fabsf(F1-1.f)>error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for texture-mapped patch (c)." << std::endl;
        failure_8 = true;
        failure=true;
    }

    // 8d, texture-mapped (u,v) quarter ellipse patch above rectangle

    context_8.deletePrimitive(UUID1);

    UUID1 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png", make_vec2(0.5,0.5), make_vec2(1,1) );

    radiation.updateGeometry();

    radiation.runBand("SW");

    context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
    context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );

    if( fabs(F0-(1.f-0.25f*M_PI))>error_threshold || fabsf(F1-1.f)>error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for texture-mapped patch (d)." << std::endl;
        failure_8 = true;
        failure=true;
    }

    // 8e, texture-mapped (u,v) half ellipse triangle above rectangle

    context_8.deletePrimitive(UUID1);

    UUID1 = context_8.addTriangle( p1+make_vec3(-0.5f*sz.x,-0.5f*sz.y,0), p1+make_vec3(0.5f*sz.x,0.5f*sz.y,0.f), p1+make_vec3(-0.5f*sz.x,0.5f*sz.y,0.f), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,1), make_vec2(0,1) );

    radiation.updateGeometry();

    radiation.runBand("SW");

    context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
    context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );

    if( fabs(F0-0.5-0.5*(1.f-0.25f*M_PI))>error_threshold || fabsf(F1-1.f)>error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for texture-mapped triangle (e)." << std::endl;
        failure_8 = true;
        failure=true;
    }

    // 8f, texture-mapped (u,v) two ellipse triangles above ellipse patch

    context_8.deletePrimitive(UUID0);

    UUID0 = context_8.addPatch( p0, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png" );

    UUID2 = context_8.addTriangle( p1+make_vec3(-0.5f*sz.x,-0.5f*sz.y,0), p1+make_vec3(0.5f*sz.x,-0.5f*sz.y,0), p1+make_vec3(0.5f*sz.x,0.5f*sz.y,0), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,0), make_vec2(1,1) );

    radiation.updateGeometry();

    radiation.runBand("SW");

    context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
    context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );
    context_8.getPrimitiveData( UUID2, "radiation_flux_SW", F2 );

    if( fabsf(F0)>error_threshold || fabsf(F1-1.f)>error_threshold || fabsf(F2-1.f)>error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for texture-mapped triangle (f)." << std::endl;
        failure_8 = true;
        failure=true;
    }

    // 8g, texture-mapped (u,v) ellipse patch above two ellipse triangles

    context_8.deletePrimitive(UUID0);
    context_8.deletePrimitive(UUID1);
    context_8.deletePrimitive(UUID2);

    UUID0 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png" );

    UUID1 = context_8.addTriangle( p0+make_vec3(-0.5f*sz.x,-0.5f*sz.y,0), p0+make_vec3(0.5f*sz.x,0.5f*sz.y,0), p0+make_vec3(-0.5f*sz.x,0.5f*sz.y,0), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,1), make_vec2(0,1) );
    UUID2 = context_8.addTriangle( p0+make_vec3(-0.5f*sz.x,-0.5f*sz.y,0), p0+make_vec3(0.5f*sz.x,-0.5f*sz.y,0), p0+make_vec3(0.5f*sz.x,0.5f*sz.y,0), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,0), make_vec2(1,1) );

    radiation.updateGeometry();

    radiation.runBand("SW");

    context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
    context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );
    context_8.getPrimitiveData( UUID2, "radiation_flux_SW", F2 );

    if( fabsf(F1)>error_threshold || fabsf(F2)>error_threshold || fabsf(F0-1.f)>error_threshold ){
        std::cout << "failed." << std::endl;
        std::cerr << "Test failed for texture-mapped triangle (g)." << std::endl;
        failure_8 = true;
        failure=true;
    }

    if( !failure_8 ){
        std::cout << "passed." << std::endl;
    }

    // -------- Test #9: homogeneous "canopy" ---------- //

    std::cout << "Test #9: Homogeneous 'canopy' of patches..." << std::flush;

    bool failure_9 = false;

    uint Ndirect_9 = 1000;
    uint Ndiffuse_9 = 5000;

    float D_9 = 50;          //domain width
    float D_inc_9 = 40;      //domain size to include in calculations
    float LAI_9 = 2.0;       //canopy leaf area index
    float h_9 = 3;           //canopy height
    float w_leaf_9 = 0.075;    //leaf width

    int Nleaves = (int)lroundf(LAI_9*D_9*D_9/w_leaf_9/w_leaf_9);

    Context context_9;

    std::vector<uint> UUIDs_leaf, UUIDs_inc;

    for( int i=0; i<Nleaves; i++ ){

        vec3 position( (-0.5f+context_9.randu())*D_9, (-0.5f+context_9.randu())*D_9, 0.5f*w_leaf_9+context_9.randu()*h_9 );

        SphericalCoord rotation( 1.f, acos_safe(1.f-context_9.randu()), 2.f*float(M_PI)*context_9.randu() );

        uint UUID = context_9.addPatch( position, make_vec2(w_leaf_9,w_leaf_9), rotation );

        context_9.setPrimitiveData( UUID, "twosided_flag", uint(1) );

        if( fabsf(position.x)<=0.5*D_inc_9 && fabsf(position.y)<=0.5*D_inc_9 ){
            UUIDs_inc.push_back( UUID );
        }

    }

    std::vector<uint> UUIDs_ground = context_9.addTile( make_vec3(0,0,0), make_vec2(D_9,D_9), make_SphericalCoord(0,0), make_int2(100,100) );

    context_9.setPrimitiveData( UUIDs_ground, "twosided_flag", uint(0) );

    RadiationModel radiation_9(&context_9);
    radiation_9.disableMessages();

    radiation_9.addRadiationBand("direct");
    radiation_9.disableEmission("direct");
    radiation_9.setDirectRayCount("direct",Ndirect_9);
    theta_s = 0.2*M_PI;
    ID = radiation_9.addSunSphereRadiationSource( make_SphericalCoord(0.5f*float(M_PI)-theta_s,0.f) );
    radiation_9.setSourceFlux(ID,"direct",1.f/cosf(theta_s));

    radiation_9.addRadiationBand("diffuse");
    radiation_9.disableEmission("diffuse");
    radiation_9.setDiffuseRayCount("diffuse",Ndiffuse_9);
    radiation_9.setDiffuseRadiationFlux("diffuse",1.f);

    radiation_9.updateGeometry();

    radiation_9.runBand("direct");
    radiation_9.runBand("diffuse");

    float intercepted_leaf_direct = 0.f;
    float intercepted_leaf_diffuse = 0.f;
    for( uint i : UUIDs_inc){

        float area = context_9.getPrimitiveArea(i);

        float flux;

        context_9.getPrimitiveData( i, "radiation_flux_direct", flux );
        intercepted_leaf_direct += flux*area/D_inc_9/D_inc_9;

        context_9.getPrimitiveData( i, "radiation_flux_diffuse", flux );
        intercepted_leaf_diffuse += flux*area/D_inc_9/D_inc_9;

    }

    float intercepted_ground_direct = 0.f;
    float intercepted_ground_diffuse = 0.f;
    for( uint i : UUIDs_ground){

        float area = context_9.getPrimitiveArea(i);

        float flux_dir;
        context_9.getPrimitiveData( i, "radiation_flux_direct", flux_dir );

        float flux_diff;
        context_9.getPrimitiveData( i, "radiation_flux_diffuse", flux_diff );

        vec3 position = context_9.getPatchCenter(i);

        if( fabsf(position.x)<=0.5*D_inc_9 && fabsf(position.y)<=0.5*D_inc_9 ){
            intercepted_ground_direct += flux_dir*area/D_inc_9/D_inc_9;
            intercepted_ground_diffuse += flux_diff*area/D_inc_9/D_inc_9;
        }

    }

    intercepted_ground_direct = 1.f - intercepted_ground_direct;
    intercepted_ground_diffuse = 1.f - intercepted_ground_diffuse;

    N = 50;
    dtheta = 0.5f*float(M_PI)/float(N);

    float intercepted_theoretical_diffuse = 0.f;
    for( int i=0; i<N; i++ ){

        float theta = (float(i)+0.5f)*dtheta;
        intercepted_theoretical_diffuse += 2.f*(1.f-expf(-0.5f*LAI_9/cosf(theta)))*cosf(theta)*sinf(theta)*dtheta;

    }

    float intercepted_theoretical_direct = 1.f-expf(-0.5f*LAI_9/cosf(theta_s));

    if( fabsf(intercepted_ground_direct-intercepted_theoretical_direct)>2.f*error_threshold || fabsf(intercepted_leaf_direct-intercepted_theoretical_direct)>2.f*error_threshold ){
        std::cerr << "Test failed for direct radiation calculations." << std::endl;
        std::cout << intercepted_ground_direct << " " << intercepted_leaf_direct << " " << intercepted_theoretical_direct << std::endl;
        failure_9 = true;
        failure=true;
    }

    if( fabsf(intercepted_ground_diffuse-intercepted_theoretical_diffuse)>2.f*error_threshold || fabsf(intercepted_leaf_diffuse-intercepted_theoretical_diffuse)>2.f*error_threshold ){
        std::cerr << "Test failed for diffuse radiation calculations." << std::endl;
        std::cout << intercepted_ground_diffuse << " " << intercepted_leaf_diffuse << " " << intercepted_theoretical_diffuse << std::endl;
        failure_9 = true;
        failure=true;
    }

    if( failure_9 ){
        std::cout << "failed." << std::endl;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #10: gas-filled furnace ---------- //
    // Case from Chapter 6.2.1 of Mbiock and Weber (2000)
    // Gass-filled furnace of size 3x1x1 m. Attenuation coefficient is 0.1 1/m,
    // Temperature of the walls (black) are 1273K, temperature of medium is 1773K.
    // Small patches are used to create the participating medium.

    std::cout << "Test #10: Gas-filled furnace..." << std::flush;

    bool failure_10 = false;

    float Rref_10 = 33000.f;

    uint Ndiffuse_10 = 10000;

    float w_10 = 1.f;         //width of box (y-dir)
    float h_10 = 1.f;         //height of box (z-dir)
    float d_10 = 3.f;         //depth of box (x-dir)

    float Tw_10 = 1273.f;   //temperature of walls (K)
    float Tm_10 = 1773.f;   //temperature of mediusm (K)

    float kappa_10 = 0.1f;  //attenuation coefficient of medium (1/m)

    float eps_m_10 = 1.f;     //emissivity of medium

    float w_patch_10 = 0.01;

    int Npatches_10 = (int)lroundf(2.f*kappa_10*w_10*h_10*d_10/w_patch_10/w_patch_10);

    Context context_10;

    std::vector<uint> UUIDs_box = context_10.addBox( make_vec3(0,0,0), make_vec3(d_10,w_10,h_10), make_int3(round(d_10/w_patch_10), round(w_10/w_patch_10), round(h_10/w_patch_10)), RGB::green, true);

    context_10.setPrimitiveData( UUIDs_box, "temperature", Tw_10 );
    context_10.setPrimitiveData( UUIDs_box, "twosided_flag", uint(0) );

    std::vector<uint> UUIDs_patches;

    for( int i=0; i<Npatches_10; i++ ){

        float x = -0.5f*d_10+0.5f*w_patch_10 + (d_10-2*w_patch_10)*context_10.randu();
        float y = -0.5f*w_10+0.5f*w_patch_10 + (w_10-2*w_patch_10)*context_10.randu();
        float z = -0.5f*h_10+0.5f*w_patch_10 + (h_10-2*w_patch_10)*context_10.randu();

        float theta = acosf(1.f-context_10.randu());
        float phi = 2.f*float(M_PI)*context_10.randu();

        UUIDs_patches.push_back(  context_10.addPatch( make_vec3(x,y,z), make_vec2(w_patch_10,w_patch_10), make_SphericalCoord(theta,phi) ) );

    }
    context_10.setPrimitiveData( UUIDs_patches, "temperature", Tm_10 );
    context_10.setPrimitiveData( UUIDs_patches, "emissivity_LW", eps_m_10 );
    context_10.setPrimitiveData( UUIDs_patches, "reflectivity_LW", 1.f-eps_m_10 );

    RadiationModel radiation_10(&context_10);
    radiation_10.disableMessages();

    radiation_10.addRadiationBand("LW");
    radiation_10.setDiffuseRayCount( "LW", Ndiffuse_10 );
    radiation_10.setScatteringDepth( "LW", 0 );

    radiation_10.updateGeometry();

    radiation_10.runBand("LW");

    float R_wall = 0;
    float A_wall = 0.f;
    for( uint i : UUIDs_box){

        float area = context_10.getPrimitiveArea(i);

        float flux;
        context_10.getPrimitiveData( i, "radiation_flux_LW", flux );

        A_wall += area;
        R_wall += flux*area;

    }
    R_wall = R_wall/A_wall-sigma*powf(Tw_10,4);


    if( fabsf(R_wall-Rref_10)/Rref_10>error_threshold ){
        failure_10 = true;
        failure=true;
    }

    if( failure_10 ){
        std::cout << R_wall << " " << Rref_10 << std::endl;
        std::cout << "failed." << std::endl;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #11: purely scattering medium between two infinite plates ---------- //
    // Test case from Francoeur's class - not sure where he got it from
    // Two infinite horizontal plates are separated by distance h and filled with a purely scattering medium.

    std::cout << "Test #11: Purely scattering medium between infinite plates..." << std::flush;

    bool failure_11 = false;

    float W_11 = 10.f;      //width of entire slab in x and y directions
    float w_11 = 5.f;       //width of slab to be considered in claculations
    float h_11 = 1.f;       //height of slab

    float Tw1_11 = 300.f;   //temperature of upper wall (K)
    float Tw2_11 = 400.f;   //temperature of lower wall (K)

    float epsw1_11 = 0.8f;  //emissivity of upper wall
    float epsw2_11 = 0.5f;  //emissivity of lower wall

    float omega_11 = 1.f;   //single-scatter albedo

    float tauL_11 = 0.1f;   //optical depth of slab

    float Psi2_exact=0.427; //exact non-dimensional heat flux of lower plate

    float w_patch_11 = 0.05;  //width of medium patches


    float beta = tauL_11/h_11;  //attenuation coefficient

    int Nleaves_11 = (int)lroundf(2.f*beta*W_11*W_11*h_11/w_patch_11/w_patch_11);

    Context context_11;

    //top wall
    std::vector<uint> UUIDs_1 = context_11.addTile( make_vec3(0,0,0.5f*h_11), make_vec2(W_11,W_11), make_SphericalCoord(M_PI,0), make_int2(round(W_11/w_patch_11/5), round(W_11/w_patch_11/5) ));

    //bottom wall
    std::vector<uint> UUIDs_2 = context_11.addTile( make_vec3(0,0,-0.5f*h_11), make_vec2(W_11,W_11), make_SphericalCoord(0,0), make_int2(round(W_11/w_patch_11/5), round(W_11/w_patch_11/5) ));

    context_11.setPrimitiveData( UUIDs_1, "temperature", Tw1_11 );
    context_11.setPrimitiveData( UUIDs_2, "temperature", Tw2_11 );
    context_11.setPrimitiveData( UUIDs_1, "emissivity_LW", epsw1_11 );
    context_11.setPrimitiveData( UUIDs_2, "emissivity_LW", epsw2_11 );
    context_11.setPrimitiveData( UUIDs_1, "reflectivity_LW", 1.f-epsw1_11 );
    context_11.setPrimitiveData( UUIDs_2, "reflectivity_LW", 1.f-epsw2_11 );
    context_11.setPrimitiveData( UUIDs_1, "twosided_flag", uint(0) );
    context_11.setPrimitiveData( UUIDs_2, "twosided_flag", uint(0) );

    std::vector<uint> UUIDs_patches_11;

    for( int i=0; i<Nleaves_11; i++ ){

        float x = -0.5f*W_11+0.5f*w_patch_11 + (W_11-w_patch_11)*context_11.randu();
        float y = -0.5f*W_11+0.5f*w_patch_11 + (W_11-w_patch_11)*context_11.randu();
        float z = -0.5f*h_11+0.5f*w_patch_11 + (h_11-w_patch_11)*context_11.randu();

        float theta = acosf(1.f-context_11.randu());
        float phi = 2.f*float(M_PI)*context_11.randu();

        UUIDs_patches_11.push_back(  context_11.addPatch( make_vec3(x,y,z), make_vec2(w_patch_11,w_patch_11), make_SphericalCoord(theta,phi) ) );

    }
    context_11.setPrimitiveData( UUIDs_patches_11, "temperature", 0.f );
    context_11.setPrimitiveData( UUIDs_patches_11, "emissivity_LW", 1.f-omega_11 );
    context_11.setPrimitiveData( UUIDs_patches_11, "reflectivity_LW", omega_11 );

    RadiationModel radiation_11(&context_11);
    radiation_11.disableMessages();

    radiation_11.addRadiationBand("LW");
    radiation_11.setDiffuseRayCount( "LW", 10000 );
    radiation_11.setScatteringDepth( "LW", 4 );

    radiation_11.updateGeometry();

    radiation_11.runBand("LW");

    float R_wall2 = 0;
    float A_wall2 = 0.f;
    for( int i=0; i<UUIDs_1.size(); i++ ){

        vec3 position = context_11.getPatchCenter(UUIDs_1.at(i));

        if( fabsf(position.x)<0.5*w_11 && fabsf(position.y)<0.5*w_11 ){

            float area = context_11.getPrimitiveArea(UUIDs_1.at(i));

            float flux;
            context_11.getPrimitiveData( UUIDs_2.at(i), "radiation_flux_LW", flux );
            R_wall2 += flux*area;

            A_wall2 += area;

        }
    }
    R_wall2 = (R_wall2/A_wall2-epsw2_11*sigma*pow(Tw2_11,4))/(sigma*(pow(Tw1_11,4)-pow(Tw2_11,4)));

    if( fabsf(R_wall2-Psi2_exact)>10.f*error_threshold ){
        failure_11 = true;
        failure=true;
    }


    if( failure_11 ){
        std::cout << "failed." << std::endl;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #12: homogeneous "canopy" with periodic boundaries---------- //

    std::cout << "Test #12: Homogeneous 'canopy' of patches with periodic boundaries..." << std::flush;

    bool failure_12 = false;

    uint Ndirect_12 = 1000;
    uint Ndiffuse_12 = 5000;

    float D_12 = 20;          //domain width
    float LAI_12 = 2.0;       //canopy leaf area index
    float h_12 = 3;           //canopy height
    float w_leaf_12 = 0.05;    //leaf width

    int Nleaves_12 = round(LAI_12*D_12*D_12/w_leaf_12/w_leaf_12);

    Context context_12;

    std::vector<uint> UUIDs_leaf_12;

    for( int i=0; i<Nleaves_12; i++ ){

        vec3 position( (-0.5+context_12.randu())*D_12, (-0.5+context_12.randu())*D_12, 0.5*w_leaf_12+context_12.randu()*h_12 );

        SphericalCoord rotation( 1.f, acos(1.f-context_12.randu()), 2.f*M_PI*context_12.randu() );

        uint UUID = context_12.addPatch( position, make_vec2(w_leaf_12,w_leaf_12), rotation );

        context_12.setPrimitiveData( UUID, "twosided_flag", uint(1) );

        UUIDs_leaf_12.push_back( UUID );

    }

    std::vector<uint> UUIDs_ground_12 = context_12.addTile( make_vec3(0,0,0), make_vec2(D_12,D_12), make_SphericalCoord(0,0), make_int2(100,100) );

    context_12.setPrimitiveData( UUIDs_ground_12, "twosided_flag", uint(0) );

    RadiationModel radiation_12(&context_12);
    radiation_12.disableMessages();

    radiation_12.addRadiationBand("direct");
    radiation_12.disableEmission("direct");
    radiation_12.setDirectRayCount("direct",Ndirect_12);
    theta_s = 0.2*M_PI;
    ID = radiation_12.addCollimatedRadiationSource( make_SphericalCoord(0.5*M_PI-theta_s,0.f) );
    radiation_12.setSourceFlux(ID,"direct",1.f/cos(theta_s));

    radiation_12.addRadiationBand("diffuse");
    radiation_12.disableEmission("diffuse");
    radiation_12.setDiffuseRayCount("diffuse",Ndiffuse_12);
    radiation_12.setDiffuseRadiationFlux("diffuse",1.f);

    radiation_12.enforcePeriodicBoundary("xy");

    radiation_12.updateGeometry();

    radiation_12.runBand("direct");
    radiation_12.runBand("diffuse");

    float intercepted_leaf_direct_12 = 0.f;
    float intercepted_leaf_diffuse_12 = 0.f;
    for( int i=0; i<UUIDs_leaf_12.size(); i++ ){

        float area = context_12.getPrimitiveArea( UUIDs_leaf_12.at(i) );

        float flux;

        context_12.getPrimitiveData( UUIDs_leaf_12.at(i), "radiation_flux_direct", flux );
        intercepted_leaf_direct_12 += flux*area/D_12/D_12;

        context_12.getPrimitiveData( UUIDs_leaf_12.at(i), "radiation_flux_diffuse", flux );
        intercepted_leaf_diffuse_12 += flux*area/D_12/D_12;

    }

    float intercepted_ground_direct_12 = 0.f;
    float intercepted_ground_diffuse_12 = 0.f;
    for( int i=0; i<UUIDs_ground_12.size(); i++ ){

        float area = context_12.getPrimitiveArea(UUIDs_ground_12.at(i));

        float flux_dir;
        context_12.getPrimitiveData( UUIDs_ground_12.at(i), "radiation_flux_direct", flux_dir );

        float flux_diff;
        context_12.getPrimitiveData( UUIDs_ground_12.at(i), "radiation_flux_diffuse", flux_diff );

        vec3 position = context_12.getPatchCenter(UUIDs_ground_12.at(i));

        intercepted_ground_direct_12 += flux_dir*area/D_12/D_12;
        intercepted_ground_diffuse_12 += flux_diff*area/D_12/D_12;

    }

    intercepted_ground_direct_12 = 1.f - intercepted_ground_direct_12;
    intercepted_ground_diffuse_12 = 1.f - intercepted_ground_diffuse_12;

    N = 50;
    dtheta = 0.5*M_PI/float(N);

    float intercepted_theoretical_diffuse_12 = 0.f;
    for( int i=0; i<N; i++ ){

        float theta = (i+0.5f)*dtheta;
        intercepted_theoretical_diffuse_12 += 2.f*(1.f-exp(-0.5*LAI_12/cos(theta)))*cos(theta)*sin(theta)*dtheta;

    }

    float intercepted_theoretical_direct_12 = 1.f-exp(-0.5*LAI_12/cos(theta_s));

    if( fabsf(intercepted_ground_direct_12-intercepted_theoretical_direct_12)>2.f*error_threshold || fabsf(intercepted_leaf_direct_12-intercepted_theoretical_direct_12)>2.f*error_threshold ){
        std::cerr << "Test failed for direct radiation calculations." << std::endl;
        std::cout << intercepted_ground_direct_12 << " " << intercepted_leaf_direct_12 << " " << intercepted_theoretical_direct_12 << std::endl;
        failure_12 = true;
        failure=true;
    }

    if( fabsf(intercepted_ground_diffuse_12-intercepted_theoretical_diffuse_12)>2.f*error_threshold || fabsf(intercepted_leaf_diffuse_12-intercepted_theoretical_diffuse_12)>4.f*error_threshold ){
        std::cerr << "Test failed for diffuse radiation calculations." << std::endl;
        std::cout << intercepted_ground_diffuse_12 << " " << intercepted_leaf_diffuse_12 << " " << intercepted_theoretical_diffuse_12 << std::endl;
        failure_12 = true;
        failure=true;
    }

    if( failure_12 ){
        std::cout << "failed." << std::endl;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #13: homogeneous "canopy" of texture-masked tile objects with periodic boundaries---------- //

    std::cout << "Test #13: homogeneous 'canopy' of texture-masked tile objects with periodic boundaries..." << std::flush;

    bool failure_13 = false;

    uint Ndirect_13 = 1000;
    uint Ndiffuse_13 = 5000;

    float D_13 = 20;          //domain width
    float LAI_13 = 1.0;       //canopy leaf area index
    float h_13 = 3;           //canopy height
    float w_leaf_13 = 0.05;    //leaf width

    Context context_13;

    uint objID_ptype = context_13.addTileObject( make_vec3(0,0,0), make_vec2(w_leaf_13,w_leaf_13), make_SphericalCoord(0,0), make_int2(2,2), "plugins/radiation/disk.png" );
    std::vector<uint> UUIDs_ptype = context_13.getObjectPointer(objID_ptype)->getPrimitiveUUIDs();

    float A_leaf = 0;
    for( uint p=0; p<UUIDs_ptype.size(); p++ ){
        A_leaf += context_13.getPrimitiveArea(UUIDs_ptype.at(p));
    }

    int Nleaves_13 = round(LAI_13*D_13*D_13/A_leaf);

    std::vector<uint> UUIDs_leaf_13;

    for( int i=0; i<Nleaves_13; i++ ){

        vec3 position( (-0.5+context_13.randu())*D_13, (-0.5+context_13.randu())*D_13, 0.5*w_leaf_13+context_13.randu()*h_13 );

        SphericalCoord rotation( 1.f, acos(1.f-context_13.randu()), 2.f*M_PI*context_13.randu() );

        uint objID = context_13.copyObject( objID_ptype );

        context_13.getObjectPointer( objID )->rotate( -rotation.elevation, "y" );
        context_13.getObjectPointer( objID )->rotate( rotation.azimuth, "z" );

        context_13.getObjectPointer( objID )->translate( position );

        std::vector<uint> UUIDs = context_13.getObjectPointer( objID )->getPrimitiveUUIDs();

        UUIDs_leaf_13.insert( UUIDs_leaf_13.end(), UUIDs.begin(), UUIDs.end() );

    }

    context_13.deleteObject( objID_ptype );

    std::vector<uint> UUIDs_ground_13 = context_13.addTile( make_vec3(0,0,0), make_vec2(D_13,D_13), make_SphericalCoord(0,0), make_int2(100,100) );

    context_13.setPrimitiveData( UUIDs_ground_13, "twosided_flag", uint(0) );

    RadiationModel radiation_13(&context_13);
    radiation_13.disableMessages();

    radiation_13.addRadiationBand("direct");
    radiation_13.disableEmission("direct");
    radiation_13.setDirectRayCount("direct",Ndirect_13);
    theta_s = 0.2*M_PI;
    ID = radiation_13.addCollimatedRadiationSource( make_SphericalCoord(0.5*M_PI-theta_s,0.f) );
    radiation_13.setSourceFlux(ID,"direct",1.f/cos(theta_s));

    radiation_13.addRadiationBand("diffuse");
    radiation_13.disableEmission("diffuse");
    radiation_13.setDiffuseRayCount("diffuse",Ndiffuse_13);
    radiation_13.setDiffuseRadiationFlux("diffuse",1.f);

    radiation_13.enforcePeriodicBoundary("xy");

    radiation_13.updateGeometry();

    radiation_13.runBand("direct");
    radiation_13.runBand("diffuse");

    float intercepted_leaf_direct_13 = 0.f;
    float intercepted_leaf_diffuse_13 = 0.f;
    for( int i=0; i<UUIDs_leaf_13.size(); i++ ){

        float area = context_13.getPrimitiveArea(UUIDs_leaf_13.at(i));

        float flux;

        context_13.getPrimitiveData( UUIDs_leaf_13.at(i), "radiation_flux_direct", flux );
        intercepted_leaf_direct_13 += flux*area/D_13/D_13;

        context_13.getPrimitiveData( UUIDs_leaf_13.at(i), "radiation_flux_diffuse", flux );
        intercepted_leaf_diffuse_13 += flux*area/D_13/D_13;

    }

    float intercepted_ground_direct_13 = 0.f;
    float intercepted_ground_diffuse_13 = 0.f;
    for( int i=0; i<UUIDs_ground_13.size(); i++ ){

        float area = context_13.getPrimitiveArea(UUIDs_ground_13.at(i));

        float flux_dir;
        context_13.getPrimitiveData( UUIDs_ground_13.at(i), "radiation_flux_direct", flux_dir );

        float flux_diff;
        context_13.getPrimitiveData( UUIDs_ground_13.at(i), "radiation_flux_diffuse", flux_diff );

        vec3 position = context_13.getPatchCenter(UUIDs_ground_13.at(i));

        intercepted_ground_direct_13 += flux_dir*area/D_13/D_13;
        intercepted_ground_diffuse_13 += flux_diff*area/D_13/D_13;

    }

    intercepted_ground_direct_13 = 1.f - intercepted_ground_direct_13;
    intercepted_ground_diffuse_13 = 1.f - intercepted_ground_diffuse_13;

    N = 50;
    dtheta = 0.5*M_PI/float(N);

    float intercepted_theoretical_diffuse_13 = 0.f;
    for( int i=0; i<N; i++ ){

        float theta = (i+0.5f)*dtheta;
        intercepted_theoretical_diffuse_13 += 2.f*(1.f-exp(-0.5*LAI_13/cos(theta)))*cos(theta)*sin(theta)*dtheta;

    }

    float intercepted_theoretical_direct_13 = 1.f-exp(-0.5*LAI_13/cos(theta_s));

    if( fabsf(intercepted_ground_direct_13-intercepted_theoretical_direct_13)>2.f*error_threshold || fabsf(intercepted_leaf_direct_13-intercepted_theoretical_direct_13)>2.f*error_threshold ){
        std::cerr << "Test failed for direct radiation calculations." << std::endl;
        std::cout << intercepted_ground_direct_13 << " " << intercepted_leaf_direct_13 << " " << intercepted_theoretical_direct_13 << std::endl;
        failure_13 = true;
        failure=true;
    }

    if( fabsf(intercepted_ground_diffuse_13-intercepted_theoretical_diffuse_13)>2.f*error_threshold || fabsf(intercepted_leaf_diffuse_13-intercepted_theoretical_diffuse_13)>4.f*error_threshold ){
        std::cerr << "Test failed for diffuse radiation calculations." << std::endl;
        std::cout << intercepted_ground_diffuse_13 << " " << intercepted_leaf_diffuse_13 << " " << intercepted_theoretical_diffuse_13 << std::endl;
        failure_13 = true;
        failure=true;
    }

    if( failure_13 ){
        std::cout << "failed." << std::endl;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #14: anisotropoic diffuse radiation for a horizontal patch---------- //

    std::cout << "Test #14: Anisotropic diffuse radiation horizontal patch..." << std::flush;

    bool failure_14 = false;

    uint Ndiffuse_14 = 10000;

    Context context_14;

    std::vector<float> K_14;
    K_14.push_back( 0.f );
    K_14.push_back( 0.25f );
    K_14.push_back( 1.f );

    std::vector<float> thetas_14;
    thetas_14.push_back(0.f);
    thetas_14.push_back(0.25*M_PI);

    uint UUID_14 = context_14.addPatch();

    context_14.setPrimitiveData( UUID_14, "twosided_flag", uint(0) );

    RadiationModel radiation_14(&context_14);
    radiation_14.disableMessages();

    radiation_14.addRadiationBand( "diffuse" );

    radiation_14.disableEmission( "diffuse" );
    radiation_14.setDiffuseRayCount( "diffuse", Ndiffuse_14 );
    radiation_14.setDiffuseRadiationFlux( "diffuse", 1.f );

    radiation_14.updateGeometry();

    for( int t=0; t<thetas_14.size(); t++ ){
        for( int k=0; k<K_14.size(); k++ ){

            radiation_14.setDiffuseRadiationExtinctionCoeff( "diffuse", K_14.at(k), make_SphericalCoord( 0.5*M_PI-thetas_14.at(t), 0.f ) );

            radiation_14.runBand( "diffuse" );

            float Rdiff;
            context_14.getPrimitiveData( UUID_14, "radiation_flux_diffuse", Rdiff );

            if( fabsf(Rdiff-1.f)>2.f*error_threshold ){
                failure_14 = true;
            }

        }
    }

    if( failure_14 ){
        std::cout << "failed." << std::endl;
        failure=true;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #15: Disk radiation source above a circular element ---------- //

    std::cout << "Test #15: Disk radiation source above a circular element..." << std::flush;

    bool failure_15 = false;

    uint Ndirect_15 = 10000;

    float r1_15 = 0.2;  //disk source radius
    float r2_15 = 0.5;  //disk element radius
    float a_15 = 0.5;   //distance between radiation source and element

    Context context_15;
    RadiationModel radiation_15(&context_15);
    radiation_15.disableMessages();

    uint UUID_15 = context_15.addPatch( make_vec3(0,0,0), make_vec2(2*r2_15,2*r2_15), make_SphericalCoord(0.5*M_PI,0), "lib/images/disk_texture.png");

    uint ID_15 = radiation_15.addDiskRadiationSource( make_vec3(0, a_15, 0), r1_15, make_vec3(0.5*M_PI,0, 0) );

    radiation_15.addRadiationBand( "light" );
    radiation_15.disableEmission( "light" );
    radiation_15.setSourceFlux( ID_15, "light", 1.f );
    radiation_15.setDirectRayCount( "light", Ndirect_15 );

    radiation_15.updateGeometry();

    radiation_15.runBand( "light" );

    float F12_15;
    context_15.getPrimitiveData( UUID_15, "radiation_flux_light", F12_15 );

    float R1_15 = r1_15/a_15;
    float R2_15 = r2_15/a_15;
    float X_15 = 1.f + (1.f + R2_15*R2_15)/(R1_15*R1_15);
    float F12_exact_15 = 0.5f*(X_15-sqrtf(X_15*X_15-4.f*powf(R2_15/R1_15,2)));

    if( fabs(F12_15-F12_exact_15*r1_15*r1_15/r2_15/r2_15)>2.f*error_threshold){
        failure_15=true;
    }

    if( failure_15 ){
        std::cout << "failed." << std::endl;
        failure=true;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #16: Rectangular radiation source above a patch ---------- //

    std::cout << "Test #16: Rectangular radiation source above a patch..." << std::flush;

    bool failure_16 = false;

    uint Ndirect_16 = 10000;

    float a_16 = 1;   //width of patch/source
    float b_16 = 2;   //length of patch/source
    float c_16 = 0.5; //distance between source and patch

    Context context_16;
    RadiationModel radiation_16(&context_16);
    radiation_16.disableMessages();

    uint UUID_16 = context_16.addPatch( make_vec3(0,0,0), make_vec2(a_16,b_16), nullrotation);

    uint ID_16 = radiation_16.addRectangleRadiationSource( make_vec3(0,0, c_16), make_vec2(a_16,b_16), make_vec3(M_PI,0,0) );

    radiation_16.addRadiationBand( "light" );
    radiation_16.disableEmission( "light" );
    radiation_16.setSourceFlux( ID_16, "light", 1.f );
    radiation_16.setDirectRayCount( "light", Ndirect_16 );

    radiation_16.updateGeometry();

    radiation_16.runBand( "light" );

    float F12_16;
    context_16.getPrimitiveData( UUID_16, "radiation_flux_light", F12_16 );

    float X_16 = a_16/c_16;
    float Y_16 = b_16/c_16;
    float X2_16 = X_16*X_16;
    float Y2_16 = Y_16*Y_16;

    float F12_exact_16 = 2.0f/float(M_PI*X_16*Y_16)*(logf(std::sqrt((1.f+X2_16)*(1.f+Y2_16)/(1.f+X2_16+Y2_16)))+X_16*std::sqrt(1.f+Y2_16)*atanf(X_16/std::sqrt(1.f+Y2_16))+Y_16*std::sqrt(1.f+X2_16)*atanf(Y_16/std::sqrt(1.f+X2_16))-X_16*atanf(X_16)-Y_16*atanf(Y_16));

    if( fabs(F12_16-F12_exact_16)>2.f*error_threshold ){
        failure_16=true;
    }

    if( failure_16 ){
        std::cout << "failed." << std::endl;
        failure=true;
    }else{
        std::cout << "passed." << std::endl;
    }

    // -------- Test #17: Radiation camera test based on ROMC brfpp_sgl verification (experiment: HET51 RED) ---------- //

    std::cout << "Test #17: Radiation camera test based on ROMC brfpp_sgl verification..." << std::flush;

    bool failure_17 = false;

    Context context_17;
    float sunzenithd = 30;
    float reflectivityleaf = 0.02;  //NIR
    float transmissivityleaf = 0.01;
    std::string bandname = "RED";

    float viewazimuth = 0;
    float heightscene = 30.f;
    float rangescene = 100.f;
    std::vector<float> viewangles = {-75,0,36};
    float sunazimuth = 0;
    std::vector<float> referencevalues = {66.f,225.f,274.f};

    // add canopy to context
    std::vector<std::vector<float>> CSpositions = {{-24.8302, 11.6110, 15.6210}, {-38.3380, -9.06342, 17.6094}, {-5.26569, 18.9618, 17.2535},
                                                   {-27.4794, -32.0266, 15.9146}, {33.5709, -6.31039, 14.5332},	{11.9126, 8.32062, 12.1220},
                                                   {32.4756, -26.9023, 16.3684}}; //HET 51

    for (int w=-1; w<2; w++){
        vec3 movew = make_vec3(0,float(rangescene*w),0);
        for (auto & CSposition : CSpositions){
            vec3 transpos = movew+make_vec3(CSposition.at(0),CSposition.at(1),CSposition.at(2));
            CameraCalibration cameracalibration(&context_17);
            std::vector<uint> iCUUIDsn = cameracalibration.readROMCCanopy();
            context_17.translatePrimitive(iCUUIDsn, transpos);
            context_17.setPrimitiveData(iCUUIDsn, "twosided_flag", uint(1));
            context_17.setPrimitiveData(iCUUIDsn, "reflectivity_spectrum","leaf_reflectivity");
            context_17.setPrimitiveData(iCUUIDsn, "transmissivity_spectrum", "leaf_transmissivity");
        }
    }

    // set optical properties
    std::vector<helios::vec2> leafspectrarho(2200);
    std::vector<helios::vec2> leafspectratau(2200);
    std::vector<helios::vec2> sourceintensity(2200);
    for (int i=0; i<leafspectrarho.size(); i++){
        leafspectrarho.at(i).x = float(301+i);
        leafspectrarho.at(i).y=reflectivityleaf;
        leafspectratau.at(i).x = float(301+i);
        leafspectratau.at(i).y=transmissivityleaf;
        sourceintensity.at(i).x = float(301+i);
        sourceintensity.at(i).y=1;
    }
    context_17.setGlobalData("leaf_reflectivity",HELIOS_TYPE_VEC2,leafspectrarho.size(),&leafspectrarho[0]);
    context_17.setGlobalData("leaf_transmissivity",HELIOS_TYPE_VEC2,leafspectratau.size(),&leafspectratau[0]);
    context_17.setGlobalData("camera_response",HELIOS_TYPE_VEC2,sourceintensity.size(),&sourceintensity[0]);  // camera response is 1
    context_17.setGlobalData("source_intensity",HELIOS_TYPE_VEC2,sourceintensity.size(),&sourceintensity[0]); // source intensity is 1

    // Add sensors to receive radiation
    vec3 camera_lookat= make_vec3(0,0,heightscene);
    std::vector<std::string> cameralabels;
    RadiationModel radiation_17(&context_17);
    radiation_17.disableMessages();
    for (float viewangle:viewangles){
        // Set camera properties
        vec3 camerarotation = sphere2cart(make_SphericalCoord(deg2rad((90-viewangle)), deg2rad(viewazimuth)));
        vec3 camera_position = 100000*camerarotation+camera_lookat;
        CameraProperties cameraproperties;
        cameraproperties.camera_resolution=make_int2(200, int(std::abs(std::round(200*std::cos(deg2rad(viewangle))))));
        cameraproperties.focal_plane_distance = 100000;
        cameraproperties.lens_diameter = 0;
        cameraproperties.HFOV=0.02864786f*2.f;
        cameraproperties.FOV_aspect_ratio = 0.5f/float(std::abs(0.5*std::cos(deg2rad(viewangle))));

        std::string cameralabel = "ROMC"+std::to_string(viewangle);
        radiation_17.addRadiationCamera(cameralabel, {bandname}, camera_position, camera_lookat, cameraproperties,60); //overlap warning multiple cameras
        cameralabels.push_back(cameralabel);

    }
    radiation_17.addSunSphereRadiationSource(make_SphericalCoord(deg2rad(90-sunzenithd), deg2rad(sunazimuth)));
    radiation_17.setSourceSpectrum(0, "source_intensity");
    radiation_17.addRadiationBand(bandname, 500, 502);
    radiation_17.setDiffuseRayCount(bandname,20);
    radiation_17.disableEmission(bandname);
    radiation_17.setSourceFlux(0, bandname,5);  //try large source flux
    radiation_17.setScatteringDepth(bandname, 1);
    radiation_17.setDiffuseRadiationFlux(bandname,0);
    radiation_17.setDiffuseRadiationExtinctionCoeff(bandname, 0.f, make_vec3(-0.5, 0.5, 1) );

    for (const auto& cameralabel:cameralabels){
        radiation_17.setCameraSpectralResponse(cameralabel, bandname, "camera_response");
    }
    radiation_17.updateGeometry();
    radiation_17.runBand(bandname);

    float cameravalue;
    std::vector<float> camera_data;
    std::vector<uint> camera_UUID;

    for (int i=0; i<cameralabels.size();i++){
        std::string global_data_label = "camera_" + cameralabels.at(i) + "_" + bandname;  //_pixel_UUID
        std::string global_UUID = "camera_" + cameralabels.at(i) + "_pixel_UUID";
        context_17.getGlobalData(global_data_label.c_str(), camera_data);
        context_17.getGlobalData(global_UUID.c_str(), camera_UUID);
        float camera_all_data=0;
        for (int v=0; v<camera_data.size();v++){
            uint iUUID = camera_UUID.at(v)-1;
            if (camera_data.at(v)>0 && context_17.doesPrimitiveExist(iUUID)){
                camera_all_data += camera_data.at(v);
            }
        }
        cameravalue = std::abs(referencevalues.at(i)-camera_all_data);
        if( cameravalue > 1.5f){
            failure_17=true;
        }
    }

    if( failure_17 ){
        std::cout << "failed." << std::endl;
        failure=true;
    }else{
        std::cout << "passed." << std::endl;
    }

    // ------------- //

    if( failure ){
        std::cout << "!!!!!!! Some tests were not successfully passed. !!!!!!!" << std::endl;
        return 1;
    }else{
        std::cout << "All self-tests passed successfully." << std::endl;
        return 0;
    }


}

