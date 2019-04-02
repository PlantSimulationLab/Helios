/** \file "RadiationModel.cpp" Primary source file for radiation transport model.
    \author Brian Bailey

    Copyright (C) 2018  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RadiationModel.h"

using namespace helios;

RadiationModel::RadiationModel( helios::Context* __context ){

  context = __context;

  // All default values set here

  message_flag = true;
  
  directRayCount_default = 100;

  diffuseRayCount_default = 1000;

  diffuseFlux_default = 0.f;

  scatteringDepth_default = 0;

  minScatterEnergy_default = 0.1;

  rho_default = 0.f;
  tau_default = 0.f;
  eps_default = 1.f;

  kappa_default = 1.f;
  sigmas_default = 0.f;

  temperature_default = 300;

  initializeOptiX();

  return;

}

int RadiationModel::selfTest( void ){

  std::cout << "Running radiation model self-test..." << std::endl;

  float error_threshold = 0.004;
  bool failure = false;

  int Nensemble = 500;

  float shortwave_rho, shortwave_tau, longwave_rho, eps;
  float shortwave_exact_0, shortwave_model_0, shortwave_error_0, shortwave_exact_1, shortwave_model_1, shortwave_error_1;
  float longwave_exact_0, longwave_model_0, longwave_error_0, longwave_exact_1, longwave_model_1, longwave_error_1;

  float a, b, c, X, X2, Y, Y2, F12;
  float R;

  uint UUID0, UUID1, UUID2;
  
  //----- Test #1: 90 degree common-edge squares ------//

  std::cout << "Test #1: 90 degree common-edge squares..." << std::flush;

  uint Ndiffuse_1 = 100000;
  uint Ndirect_1 = 5000;

  float Qs = 1000.f;
  
  shortwave_exact_0 = 0.7f*Qs;
  shortwave_exact_1 = 0.3f*0.2f*Qs;
  longwave_exact_0 = 5.67e-8*pow(300.f,4)*0.2f;
  longwave_exact_1 = 5.67e-8*pow(300.f,4)*0.2f;
  
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

    radiationmodel_1.runBand("LW");

    //patch 0 emission
    context_1.getPrimitiveData(0,"radiation_flux_LW",R);
    longwave_model_0 += R/float(Nensemble);
    //patch 1 emission
    context_1.getPrimitiveData(1,"radiation_flux_LW",R);
    longwave_model_1 += R/float(Nensemble);

    radiationmodel_1.runBand("SW");

    //patch 0 shortwave
    context_1.getPrimitiveData(0,"radiation_flux_SW",R);
    shortwave_model_0 += R/float(Nensemble);
    //patch 1 shortwave
    context_1.getPrimitiveData(1,"radiation_flux_SW",R);
    shortwave_model_1 += R/float(Nensemble); 
    
  }

  longwave_error_0 = fabs(longwave_model_0-longwave_exact_0)/fabs(longwave_exact_0);
  longwave_error_1 = fabs(longwave_model_1-longwave_exact_1)/fabs(longwave_exact_1);

  shortwave_error_0 = fabs(shortwave_model_0-shortwave_exact_0)/fabs(shortwave_exact_0);
  shortwave_error_1 = fabs(shortwave_model_1-shortwave_exact_1)/fabs(shortwave_exact_1);

  bool failure_0 = false;
  if( shortwave_error_0 > error_threshold || shortwave_error_1 > error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for shortwave radiation. Patch #0 error: " << shortwave_error_0 << ", Patch #1 error: " << shortwave_error_1 << std::endl;
    failure_0 = true;
    failure = true;
  }
  if( longwave_error_1 > error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for radiation emission. Error: " << longwave_error_1 << std::endl;
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

  F12 = 2.f/(M_PI*X*Y)*(log(std::sqrt((1.f+X2)*(1.f+Y2)/(1.f+X2+Y2)))+X*std::sqrt(1.f+Y2)*atan(X/std::sqrt(1.f+Y2))+Y*std::sqrt(1.f+X2)*atan(Y/std::sqrt(1.f+X2))-X*atan(X)-Y*atan(Y));

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

  shortwave_error_0 = fabs(shortwave_model_0-shortwave_exact_0)/fabs(shortwave_exact_0);
  shortwave_error_1 = fabs(shortwave_model_1-shortwave_exact_1)/fabs(shortwave_exact_1);

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

  F12 = 2.f/(M_PI*X*Y)*(log(std::sqrt((1.f+X2)*(1.f+Y2)/(1.f+X2+Y2)))+X*std::sqrt(1.f+Y2)*atan(X/std::sqrt(1.f+Y2))+Y*std::sqrt(1.f+X2)*atan(Y/std::sqrt(1.f+X2))-X*atan(X)-Y*atan(Y));

  longwave_exact_0 = (eps*(1.f/eps-1.f)*F12*5.67e-8*(pow(T1,4)-F12*pow(T0,4))+5.67e-8*(pow(T0,4)-F12*pow(T1,4)))/(1.f/eps-(1.f/eps-1.f)*F12*eps*(1/eps-1)*F12)-eps*5.67e-8*pow(T0,4);
  longwave_exact_1 =  fabs(eps*((1/eps-1)*F12*(longwave_exact_0+eps*5.67e-8*pow(T0,4))+5.67e-8*(pow(T1,4)-F12*pow(T0,4)))-eps*5.67e-8*pow(T1,4));
  longwave_exact_0 = fabs(longwave_exact_0);
  
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

  longwave_error_0 = fabs(longwave_exact_0-longwave_model_0)/fabs(longwave_exact_0);
  longwave_error_1 = fabs(longwave_exact_1-longwave_model_1)/fabs(longwave_exact_1);

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

  F12 = 0.25f/M_PI*atan(sqrtf(1.f/(D1*D1+D2*D2+D1*D1*D2*D2)));

  shortwave_exact_0 = 4.f*M_PI*r*r*F12/(l1*l2);

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

  for( int r=0; r<Nensemble; r++ ){

    radiationmodel_4.runBand("SW");

    context_4.getPrimitiveData(0,"radiation_flux_SW",R);
    shortwave_model_0 += R/float(Nensemble);
      
  }

  shortwave_error_0 = fabs(shortwave_exact_0-shortwave_model_0)/fabs(shortwave_exact_0);
  
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
  longwave_exact_0 = 5.67e-8*pow(300.f,4)*0.2f;
  longwave_exact_1 = 5.67e-8*pow(300.f,4)*0.2f;
  
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

  for( int r=0; r<Nensemble; r++ ){

    radiationmodel_5.runBand("LW");

    //patch 0 emission
    context_5.getPrimitiveData(0,"radiation_flux_LW",R);
    longwave_model_0 += 0.5*R/float(Nensemble);
    context_5.getPrimitiveData(1,"radiation_flux_LW",R);
    longwave_model_0 += 0.5*R/float(Nensemble);
    //patch 1 emission
    context_5.getPrimitiveData(2,"radiation_flux_LW",R);
    longwave_model_1 += 0.5*R/float(Nensemble);
    context_5.getPrimitiveData(3,"radiation_flux_LW",R);
    longwave_model_1 += 0.5*R/float(Nensemble);

    radiationmodel_5.runBand("SW");

    //patch 0 shortwave
    context_5.getPrimitiveData(0,"radiation_flux_SW",R);
    shortwave_model_0 += 0.5*R/float(Nensemble);
    context_5.getPrimitiveData(1,"radiation_flux_SW",R);
    shortwave_model_0 += 0.5*R/float(Nensemble);
    //patch 1 shortwave
    context_5.getPrimitiveData(2,"radiation_flux_SW",R);
    shortwave_model_1 += 0.5*R/float(Nensemble);
    context_5.getPrimitiveData(3,"radiation_flux_SW",R);
    shortwave_model_1 += 0.5*R/float(Nensemble); 
    
  }

  longwave_error_0 = fabs(longwave_model_0-longwave_exact_0)/fabs(longwave_exact_0);
  longwave_error_1 = fabs(longwave_model_1-longwave_exact_1)/fabs(longwave_exact_1);

  shortwave_error_0 = fabs(shortwave_model_0-shortwave_exact_0)/fabs(shortwave_exact_0);
  shortwave_error_1 = fabs(shortwave_model_1-shortwave_exact_1)/fabs(shortwave_exact_1);

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

  // ------ Test #6: Parallel Disks (AlphaMasks) ------- //

  std::cout << "Test #6: Parallel disks (AlphaMasks)..." << std::flush;

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
  F12 = 0.5f*(X-std::sqrt(X*X-4.f*pow(R2/R1,2)));

  shortwave_exact_0 = (A1-A2)/A1*(1.f-shortwave_rho);
  shortwave_exact_1 = (A1-A2)/A1*F12*A1/A2*shortwave_rho;
  longwave_exact_0 = 5.67e-8*pow(300.f,4)*F12;
  longwave_exact_1 = 5.67e-8*pow(300.f,4)*F12*A1/A2;

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
  
  for( uint r=0; r<Nensemble; r++ ){

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

  shortwave_error_0 = fabs(shortwave_exact_0-shortwave_model_0)/fabs(shortwave_exact_0);
  shortwave_error_1 = fabs(shortwave_exact_1-shortwave_model_1)/fabs(shortwave_exact_1);
  longwave_error_0 = fabs(longwave_exact_0-longwave_model_0)/fabs(longwave_exact_0);
  longwave_error_1 = fabs(longwave_exact_1-longwave_model_1)/fabs(longwave_exact_1);

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

  std::vector<uint> UUIDt = context_7.addBox( make_vec3(0,0,0), make_vec3(10,10,10),make_int3(5,5,5), RGB::black, true );

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
    flux_err += fabs( R - eps1_7*5.67e-8*pow(300,4) )/(eps1_7*5.67e-8*pow(300,4))/float(UUIDt.size());
  }

  bool failure_7 = false;
  if( flux_err > error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for equilibrium emission with constant emissivity. error: " << flux_err << std::endl;
    failure_7 = true;
    failure = true;
  }
  

  for( int p=0; p<UUIDt.size(); p++ ){
    float eps;
    if( context_7.randu()<0.5f ){
      eps = eps1_7;
    }else{
      eps = eps2_7;
    }
    context_7.setPrimitiveData( UUIDt.at(p), "emissivity_LW", eps );
    context_7.setPrimitiveData( UUIDt.at(p), "reflectivity_LW", 1.f-eps );
  }

  radiationmodel_7.runBand("LW");

  flux_err = 0.f;
  for( int p=0; p<UUIDt.size(); p++ ){
    context_7.getPrimitiveData( UUIDt.at(p), "radiation_flux_LW", R );
    float eps;
    context_7.getPrimitiveData( UUIDt.at(p), "emissivity_LW", eps );
    flux_err += fabs( R - eps*5.67e-8*pow(300,4) )/(eps*5.67e-8*pow(300,4))/float(UUIDt.size());
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

  if( fabs(F0-(1.f-0.25f*M_PI))>error_threshold || fabs(F1-1.f)>error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for texture-mapped patch (a)." << std::endl;
    failure_8 = true;
    failure=true;
  }

  // 8b, texture-mapped (u,v) inscribed ellipse patch above rectangle

  context_8.deletePrimitive(UUID1);

  //UUID1 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png", make_vec2(0.25,0.75), make_vec2(0.75,0.75), make_vec2(0.75,0.25), make_vec2(0.25,0.25) );
  UUID1 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png", make_vec2(0.5,0.5), make_vec2(0.5,0.5) );

  radiation.updateGeometry();

  radiation.runBand("SW");

  context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
  context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );

  if( fabs(F0)>error_threshold || fabs(F1-1.f)>error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for texture-mapped patch (b)." << std::endl;
    failure_8 = true;
    failure=true;
  }

  // 8c, texture-mapped (u,v) quarter ellipse patch above rectangle

  context_8.deletePrimitive(UUID1);

  UUID1 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png", make_vec2(0.5,0.5), make_vec2(1,1) );

  radiation.updateGeometry();

  radiation.runBand("SW");

  context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
  context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );

  if( fabs(F0-(1.f-0.25f*M_PI))>error_threshold || fabs(F1-1.f)>error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for texture-mapped patch (c)." << std::endl;
    failure_8 = true;
    failure=true;
  }

  // 8d, texture-mapped (u,v) half ellipse triangle above rectangle

  context_8.deletePrimitive(UUID1);

  UUID1 = context_8.addTriangle( p1+make_vec3(-0.5*sz.x,-0.5*sz.y,0), p1+make_vec3(0.5*sz.x,0.5*sz.y,0), p1+make_vec3(-0.5*sz.x,0.5*sz.y,0), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,1), make_vec2(0,1) ); 

  radiation.updateGeometry();

  radiation.runBand("SW");

  context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
  context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );

  if( fabs(F0-0.5-0.5*(1.f-0.25f*M_PI))>error_threshold || fabs(F1-1.f)>error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for texture-mapped triangle (d)." << std::endl;
    failure_8 = true;
    failure=true;
  }

  // 8e, texture-mapped (u,v) two ellipse triangles above ellipse patch

  context_8.deletePrimitive(UUID0);

  UUID0 = context_8.addPatch( p0, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png" );
  
  UUID2 = context_8.addTriangle( p1+make_vec3(-0.5*sz.x,-0.5*sz.y,0), p1+make_vec3(0.5*sz.x,-0.5*sz.y,0), p1+make_vec3(0.5*sz.x,0.5*sz.y,0), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,0), make_vec2(1,1) );

  radiation.updateGeometry();

  radiation.runBand("SW");

  context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
  context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );
  context_8.getPrimitiveData( UUID2, "radiation_flux_SW", F2 );

  if( fabs(F0)>error_threshold || fabs(F1-1.f)>error_threshold || fabs(F2-1.f)>error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for texture-mapped triangle (e)." << std::endl;
    failure_8 = true;
    failure=true;
  }
  
  // 8f, texture-mapped (u,v) ellipse patch above two ellipse triangles

  context_8.deletePrimitive(UUID0);
  context_8.deletePrimitive(UUID1);
  context_8.deletePrimitive(UUID2);

  UUID0 = context_8.addPatch( p1, sz, make_SphericalCoord(0,0), "lib/images/disk_texture.png" );

  UUID1 = context_8.addTriangle( p0+make_vec3(-0.5*sz.x,-0.5*sz.y,0), p0+make_vec3(0.5*sz.x,0.5*sz.y,0), p0+make_vec3(-0.5*sz.x,0.5*sz.y,0), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,1), make_vec2(0,1) ); 
  UUID2 = context_8.addTriangle( p0+make_vec3(-0.5*sz.x,-0.5*sz.y,0), p0+make_vec3(0.5*sz.x,-0.5*sz.y,0), p0+make_vec3(0.5*sz.x,0.5*sz.y,0), "lib/images/disk_texture.png", make_vec2(0,0), make_vec2(1,0), make_vec2(1,1) );

  radiation.updateGeometry();

  radiation.runBand("SW");

  context_8.getPrimitiveData( UUID0, "radiation_flux_SW", F0 );
  context_8.getPrimitiveData( UUID1, "radiation_flux_SW", F1 );
  context_8.getPrimitiveData( UUID2, "radiation_flux_SW", F2 );

  if( fabs(F1)>error_threshold || fabs(F2)>error_threshold || fabs(F0-1.f)>error_threshold ){
    std::cout << "failed." << std::endl;
    std::cerr << "Test failed for texture-mapped triangle (f)." << std::endl;
    failure_8 = true;
    failure=true;
  }

  if( !failure_8 ){
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

void RadiationModel::disableMessages(void){
  message_flag=false;
}

void RadiationModel::enableMessages(void){
  message_flag=true;
}

void RadiationModel::setDirectRayCount( const char* label, size_t N ){
  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (setDirectRayCount): Cannot set ray count for band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint band = band_names.at(label);
  directRayCount.at(band) = N;
}

void RadiationModel::setDiffuseRayCount(  const char* label, size_t N ){
  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (setDiffuseRayCount): Cannot set ray count for band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint band = band_names.at(label);
  diffuseRayCount.at(band) = N;
}

void RadiationModel::setDiffuseRadiationFlux( const char* label, float flux ){
  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (setDiffuseRadiationFlux): Cannot set flux value for band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint band = band_names.at(label);
  diffuseFlux.at(band) = flux;
}

uint RadiationModel::addRadiationBand( const char* label ){

  emission_flag.push_back(true);//by default, emission is run

  uint band = emission_flag.size()-1; //band index

  band_names[label] = band;

  isbandpropertyinitialized.push_back(false);

  //Set OptiX buffer 'Nbands'

  RT_CHECK_ERROR( rtVariableSet1ui( Nbands_RTvariable, band+1 ) );

  //Initialize all radiation source fluxes to zero
  std::vector<float> fluxes;
  fluxes.resize( source_positions.size(), 0.f );
  source_fluxes[label] = fluxes;

  //Initialize direct/diffuse fluxes using default values
  directRayCount.push_back( directRayCount_default );
  diffuseRayCount.push_back( diffuseRayCount_default );
  diffuseFlux.push_back( diffuseFlux_default );
  scatteringDepth.push_back( scatteringDepth_default );
  minScatterEnergy.push_back( minScatterEnergy_default );

  return band;

}

void RadiationModel::disableEmission( const char* label ){

  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (disableEmission): Cannot disable emission for band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  
  uint band = band_names[label];

  emission_flag.at(band) = false;

}

uint RadiationModel::addCollimatedRadiationSource( void ){

  return addCollimatedRadiationSource( make_vec3(0,0,1) );
  
}

uint RadiationModel::addCollimatedRadiationSource( const helios::SphericalCoord direction ){
  return addCollimatedRadiationSource( sphere2cart(direction) );
}

uint RadiationModel::addCollimatedRadiationSource( const helios::vec3 direction ){

  if( direction.magnitude()==0 ){
    std::cerr << "ERROR (addCollimatedRadiationSource): Invalid collimated source direction. Direction vector should not have length of zero." << std::endl;
    exit(EXIT_FAILURE);
  }

  source_positions.push_back( direction );

  source_position_scaling_factors.push_back( 1.f );

  source_widths.push_back( 0 );

  source_types.push_back( RADIATION_SOURCE_TYPE_COLLIMATED );

  source_flux_scaling_factors.push_back( 1.f );

  uint ID = source_positions.size()-1;

  //Initialize source flux to zero for all bands
  for( std::map<std::string,std::vector<float> >::iterator it=source_fluxes.begin(); it!=source_fluxes.end(); ++it ){
    if( it->second.size()<=ID ){
      it->second.resize(ID+1, 0.f);
    }
  }
  
  return ID;
  
}

uint RadiationModel::addSphereRadiationSource( const helios::vec3 position, const float radius ){

  source_positions.push_back( position );

  source_position_scaling_factors.push_back( 1.f );

  source_widths.push_back( 2.f*fabs(radius) );

  source_types.push_back( RADIATION_SOURCE_TYPE_SPHERE );

  source_flux_scaling_factors.push_back( 1.f );

  uint ID = source_positions.size()-1;

  //Initialize source flux to zero for all bands
  for( std::map<std::string,std::vector<float> >::iterator it=source_fluxes.begin(); it!=source_fluxes.end(); ++it ){
    if( it->second.size()<=ID ){
      it->second.resize(ID+1, 0.f);
    }
  }
  
  return ID;
  
}

uint RadiationModel::addSunSphereRadiationSource( const helios::SphericalCoord sun_direction ){
  return addSunSphereRadiationSource( sphere2cart(sun_direction) );
}

uint RadiationModel::addSunSphereRadiationSource( const helios::vec3 sun_direction ){

  source_positions.push_back( 150e9*sun_direction/sun_direction.magnitude() );

  source_position_scaling_factors.push_back( 150e9 );

  source_widths.push_back( 2.f*695.5e6 );

  source_types.push_back( RADIATION_SOURCE_TYPE_SPHERE );

  source_flux_scaling_factors.push_back( 5.67e-8*pow(5700,4)/1288.437 );

  uint ID = source_positions.size()-1;

  //Initialize source flux to zero for all bands
  for( std::map<std::string,std::vector<float> >::iterator it=source_fluxes.begin(); it!=source_fluxes.end(); ++it ){
    if( it->second.size()<=ID ){
      it->second.resize(ID+1, 0.f);
    }
  }
  
  return ID;
  
}

void RadiationModel::setSourceFlux( const uint ID, const char* band_label, const float flux ){

  if( band_names.find(band_label) == band_names.end() ){
    std::cerr << "ERROR (setSourceFlux): Cannot add radiation source for band " << band_label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }else if( ID>=source_positions.size() ){
    std::cerr << "ERROR (setSourceFlux): Source ID out of bounds. Only " << source_positions.size()-1 << " radiation sources." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<float> fluxes = source_fluxes.at(band_label);

  assert( source_positions.size() == fluxes.size() );
  
  source_fluxes.at(band_label).at(ID) = flux*source_flux_scaling_factors.at(ID);


}

void RadiationModel::setSourcePosition( const uint ID, const helios::vec3 position ){

  if( ID>=source_positions.size() ){
    std::cerr << "ERROR (setSourcePosition): Source ID out of bounds. Only " << source_positions.size()-1 << " radiation sources." << std::endl;
    exit(EXIT_FAILURE);
  }

  source_positions.at(ID) = position*source_position_scaling_factors.at(ID);

  if( source_position_scaling_factors.at(ID)!= 0 ){
    source_positions.at(ID) = source_positions.at(ID)/position.magnitude();
  }
  
}

void RadiationModel::setScatteringDepth( const char* label, uint depth ){

  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (setScatteringDepth): Cannot set scattering depth for band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint band = band_names.at(label);
  scatteringDepth.at(band) = depth;
  
}

void RadiationModel::setMinScatterEnergy( const char* label, uint energy ){

  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (setMinScatterEnergy): Cannot set minimum scattering energy for band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint band = band_names.at(label);
  minScatterEnergy.at(band) = energy;
  
}


void RadiationModel::initializeOptiX( void ){
    
  /* Context */
  RT_CHECK_ERROR( rtContextCreate( &OptiX_Context ) );
  RT_CHECK_ERROR( rtContextSetPrintEnabled( OptiX_Context, 1 ) );

  RT_CHECK_ERROR( rtContextSetRayTypeCount( OptiX_Context, 5 ) );
  //ray types are:
  // 0: direct_ray_type
  // 1: diffuse_ray_type
  // 2: direct_ray_type_MCRT
  // 3: diffuse_ray_type_MCRT
  // 4: emission_ray_type_MCRT

  RT_CHECK_ERROR( rtContextSetEntryPointCount( OptiX_Context, 5 ) );
  //ray entery points are
  // 0: direct_raygen
  // 1: diffuse_raygen
  // 2: direct_raygen_MCRT
  // 3: diffuse_raygen_MCRT
  // 4: emission_raygen_MCRT

  /* Ray Types */
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "direct_ray_type", &direct_ray_type_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( direct_ray_type_RTvariable, RAYTYPE_DIRECT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "diffuse_ray_type", &diffuse_ray_type_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( diffuse_ray_type_RTvariable, RAYTYPE_DIFFUSE ) );

  //MCRT
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "direct_ray_type_MCRT", &direct_ray_type_MCRT_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( direct_ray_type_MCRT_RTvariable, RAYTYPE_DIRECT_MCRT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "diffuse_ray_type_MCRT", &diffuse_ray_type_MCRT_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( diffuse_ray_type_MCRT_RTvariable, RAYTYPE_DIFFUSE_MCRT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "emission_ray_type_MCRT", &emission_ray_type_MCRT_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( emission_ray_type_MCRT_RTvariable, RAYTYPE_EMISSION_MCRT ) );
  
  /* Ray Generation Program */

  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "direct_raygen", &direct_raygen ) );
  RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_DIRECT, direct_raygen ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "diffuse_raygen", &diffuse_raygen ) );
  RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_DIFFUSE, diffuse_raygen ) );

  //MCRT
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "direct_raygen_MCRT", &direct_raygen_MCRT ) );
  RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_DIRECT_MCRT, direct_raygen_MCRT ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "diffuse_raygen_MCRT", &diffuse_raygen_MCRT ) );
  RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_DIFFUSE_MCRT, diffuse_raygen_MCRT ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "emission_raygen_MCRT", &emission_raygen_MCRT ) );
  RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_EMISSION_MCRT, emission_raygen_MCRT ) );

  /* Declare Buffers and Variables */

  //primitive reflectivity buffer
  addBuffer( "rho", rho_RTbuffer, rho_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );
  //primitive transmissivity buffer
  addBuffer( "tau", tau_RTbuffer, tau_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );
  //primitive emissivity buffer
  addBuffer( "eps", eps_RTbuffer, eps_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

  //primitive transformation matrix buffer
  addBuffer( "transform_matrix", transform_matrix_RTbuffer, transform_matrix_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 2 );

  //primitive type buffer
  addBuffer( "primitive_type", primitive_type_RTbuffer, primitive_type_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );

  //primitive area buffer
  addBuffer( "primitive_area", primitive_area_RTbuffer, primitive_area_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

  //primitive UUID buffers
  addBuffer( "patch_UUID", patch_UUID_RTbuffer, patch_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
  addBuffer( "triangle_UUID", triangle_UUID_RTbuffer, triangle_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
  addBuffer( "disk_UUID", disk_UUID_RTbuffer, disk_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
  addBuffer( "alphamask_UUID", alphamask_UUID_RTbuffer, alphamask_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
  addBuffer( "voxel_UUID", voxel_UUID_RTbuffer, voxel_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );

  //primitive two-sided flag buffer
  addBuffer( "twosided_flag", twosided_flag_RTbuffer, twosided_flag_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_BYTE, 1 );

  //patch buffers
  addBuffer( "patch_vertices", patch_vertices_RTbuffer, patch_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );
  
  //triangle buffers
  addBuffer( "triangle_vertices", triangle_vertices_RTbuffer, triangle_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

  //disk buffers
  addBuffer( "disk_centers", disk_centers_RTbuffer, disk_centers_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );
  addBuffer( "disk_radii", disk_radii_RTbuffer, disk_radii_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );
  addBuffer( "disk_normals", disk_normals_RTbuffer, disk_normals_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );

  //alphamask buffers
  addBuffer( "alphamask_vertices", alphamask_vertices_RTbuffer, alphamask_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

  //voxel buffers
  addBuffer( "voxel_vertices", voxel_vertices_RTbuffer, voxel_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

  //radiation energy rate data buffers
  // - in -
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT_OUTPUT, &radiation_in_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( radiation_in_RTbuffer, RT_FORMAT_FLOAT ) );
  //RT_CHECK_ERROR( rtBufferSetElementSize( radiation_in_RTbuffer, sizeof(double) ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "radiation_in", &radiation_in_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( radiation_in_RTvariable, radiation_in_RTbuffer ) );
  zeroBuffer1D(  radiation_in_RTbuffer, 1 );
  // - out,top -
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT_OUTPUT, &radiation_out_top_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( radiation_out_top_RTbuffer, RT_FORMAT_FLOAT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "radiation_out_top", &radiation_out_top_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( radiation_out_top_RTvariable, radiation_out_top_RTbuffer ) );
  zeroBuffer1D(  radiation_out_top_RTbuffer, 1 );
  // - out,bottom -
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT_OUTPUT, &radiation_out_bottom_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( radiation_out_bottom_RTbuffer, RT_FORMAT_FLOAT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "radiation_out_bottom", &radiation_out_bottom_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( radiation_out_bottom_RTvariable, radiation_out_bottom_RTbuffer ) );
  zeroBuffer1D(  radiation_out_bottom_RTbuffer, 1 );

  //primitive scattering buffers
  // - top -
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT_OUTPUT, &scatter_buff_top_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( scatter_buff_top_RTbuffer, RT_FORMAT_FLOAT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "scatter_buff_top", &scatter_buff_top_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( scatter_buff_top_RTvariable, scatter_buff_top_RTbuffer ) );
  zeroBuffer1D( scatter_buff_top_RTbuffer, 1 );
  // - bottom -
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT_OUTPUT, &scatter_buff_bottom_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( scatter_buff_bottom_RTbuffer, RT_FORMAT_FLOAT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "scatter_buff_bottom", &scatter_buff_bottom_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( scatter_buff_bottom_RTvariable, scatter_buff_bottom_RTbuffer ) );
  zeroBuffer1D( scatter_buff_bottom_RTbuffer, 1 );

  //Energy absorbed by "sky"
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT_OUTPUT, &Rsky_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( Rsky_RTbuffer, RT_FORMAT_FLOAT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "Rsky", &Rsky_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( Rsky_RTvariable, Rsky_RTbuffer ) );
  zeroBuffer1D( Rsky_RTbuffer, 1 );

  //number of external radiation sources
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "Nsources", &Nsources_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( Nsources_RTvariable,0 ) );

  //External radiation source positions
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &source_positions_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( source_positions_RTbuffer, RT_FORMAT_FLOAT3 ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "source_positions", &source_positions_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( source_positions_RTvariable, source_positions_RTbuffer ) );
  zeroBuffer1D( source_positions_RTbuffer, 1 );

  //External radiation source widths
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &source_widths_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( source_widths_RTbuffer, RT_FORMAT_FLOAT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "source_widths", &source_widths_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( source_widths_RTvariable, source_widths_RTbuffer ) );
  zeroBuffer1D( source_widths_RTbuffer, 1 );

   //External radiation source types
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &source_types_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( source_types_RTbuffer, RT_FORMAT_UNSIGNED_INT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "source_types", &source_types_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( source_types_RTvariable, source_types_RTbuffer ) );
  zeroBuffer1D( source_types_RTbuffer, 1 );

  //External radiation source fluxes
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &source_fluxes_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( source_fluxes_RTbuffer, RT_FORMAT_FLOAT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "source_fluxes", &source_fluxes_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( source_fluxes_RTvariable, source_fluxes_RTbuffer ) );
  zeroBuffer1D( source_fluxes_RTbuffer, 1 );

  //number of radiation bands
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "Nbands", &Nbands_RTvariable ) );

  //Flux of diffuse radiation
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "diffuseFlux", &diffuseFlux_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1f( diffuseFlux_RTvariable, 0.f ));

  //Bounding sphere radius and center
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "bound_sphere_radius", &bound_sphere_radius_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1f( bound_sphere_radius_RTvariable, 0.f ));
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "bound_sphere_center", &bound_sphere_center_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet3f( bound_sphere_center_RTvariable, 0.f, 0.f, 0.f ));

  //Texture mask data
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &maskdata_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( maskdata_RTbuffer, RT_FORMAT_BYTE ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "maskdata", &maskdata_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( maskdata_RTvariable, maskdata_RTbuffer ) );
  std::vector<std::vector<std::vector<bool> > > dummydata;
  initializeBuffer3D( maskdata_RTbuffer, dummydata );

  //Texture mask size
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &masksize_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( masksize_RTbuffer, RT_FORMAT_INT2 ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "masksize", &masksize_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( masksize_RTvariable, masksize_RTbuffer ) );
  zeroBuffer1D( masksize_RTbuffer, 1 );

  //Texture mask ID
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &maskID_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( maskID_RTbuffer, RT_FORMAT_INT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "maskID", &maskID_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( maskID_RTvariable, maskID_RTbuffer ) );
  zeroBuffer1D( maskID_RTbuffer, 1 );

  //Texture u,v data
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &uvdata_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( uvdata_RTbuffer, RT_FORMAT_FLOAT2 ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "uvdata", &uvdata_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( uvdata_RTvariable, uvdata_RTbuffer ) );
  zeroBuffer2D( uvdata_RTbuffer, optix::make_int2(1,1) );

  //Texture u,v ID
  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &uvID_RTbuffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( uvID_RTbuffer, RT_FORMAT_INT ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "uvID", &uvID_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSetObject( uvID_RTvariable, uvID_RTbuffer ) );
  zeroBuffer1D( uvID_RTbuffer, 1 );

  /* Hit Programs */
  RTprogram closest_hit_direct;
  RTprogram closest_hit_diffuse;
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "closest_hit_direct", &closest_hit_direct ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "closest_hit_diffuse", &closest_hit_diffuse ) );
  RTprogram closest_hit_MCRT;
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit_MCRT.cu.ptx", "closest_hit_MCRT", &closest_hit_MCRT ) );
    
  /* Initialize Patch Geometry */

  RTprogram  patch_intersection_program;
  RTprogram  patch_bounding_box_program;

  RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &patch ) );

  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "rectangle_bounds", &patch_bounding_box_program ) );
  RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( patch, patch_bounding_box_program ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "rectangle_intersect", &patch_intersection_program ) );
  RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( patch, patch_intersection_program ) );

  /* Create Patch Material */
  
  RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &patch_material ) );
  
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_DIRECT, closest_hit_direct ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
  //MCRT
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_DIRECT_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_DIFFUSE_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_EMISSION_MCRT, closest_hit_MCRT ) );

  /* Initialize Triangle Geometry */

  RTprogram  triangle_intersection_program;
  RTprogram  triangle_bounding_box_program;

  RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &triangle ) );

  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "triangle_bounds", &triangle_bounding_box_program ) );
  RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( triangle, triangle_bounding_box_program ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "triangle_intersect", &triangle_intersection_program ) );
  RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( triangle, triangle_intersection_program ) );

  /* Create Triangle Material */
  
  RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &triangle_material ) );
  
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_DIRECT, closest_hit_direct ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
  //MCRT
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_DIRECT_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_DIFFUSE_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_EMISSION_MCRT, closest_hit_MCRT ) );

  /* Initialize Disk Geometry */

  RTprogram  disk_intersection_program;
  RTprogram  disk_bounding_box_program;

  RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &disk ) );

  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "disk_bounds", &disk_bounding_box_program ) );
  RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( disk, disk_bounding_box_program ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "disk_intersect", &disk_intersection_program ) );
  RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( disk, disk_intersection_program ) );

  /* Create Disk Material */
  
  RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &disk_material ) );
  
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_DIRECT, closest_hit_direct ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
  //MCRT
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_DIRECT_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_DIFFUSE_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_EMISSION_MCRT, closest_hit_MCRT ) );

  /* Initialize AlphaMask Geometry */

  RTprogram  alphamask_intersection_program;
  RTprogram  alphamask_bounding_box_program;

  RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &alphamask ) );

  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "alphamask_bounds", &alphamask_bounding_box_program ) );
  RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( alphamask, alphamask_bounding_box_program ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "alphamask_intersect", &alphamask_intersection_program ) );
  RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( alphamask, alphamask_intersection_program ) );

  /* Create AlphaMask Material */
  
  RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &alphamask_material ) );
  
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( alphamask_material, RAYTYPE_DIRECT, closest_hit_direct ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( alphamask_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
  //MCRT
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( alphamask_material, RAYTYPE_DIRECT_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( alphamask_material, RAYTYPE_DIFFUSE_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( alphamask_material, RAYTYPE_EMISSION_MCRT, closest_hit_MCRT ) );

  /* Initialize Voxel Geometry */

  RTprogram  voxel_intersection_program;
  RTprogram  voxel_bounding_box_program;

  RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &voxel ) );

  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "voxel_bounds", &voxel_bounding_box_program ) );
  RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( voxel, voxel_bounding_box_program ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "voxel_intersect", &voxel_intersection_program ) );
  RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( voxel, voxel_intersection_program ) );

  /* Create Voxel Material */
  
  RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &voxel_material ) );
  
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_DIRECT, closest_hit_direct ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
  //MCRT
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_DIRECT_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_DIFFUSE_MCRT, closest_hit_MCRT ) );
  RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_EMISSION_MCRT, closest_hit_MCRT ) );

  /* Miss Program */
  RTprogram miss_program_direct;
  RTprogram miss_program_diffuse;
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "miss_direct", &miss_program_direct ) );
  RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_DIRECT, miss_program_direct ) );
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "miss_diffuse", &miss_program_diffuse ) );  
  RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_DIFFUSE, miss_program_diffuse ) );
  //MCRT
  RTprogram miss_program_MCRT;
  RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit_MCRT.cu.ptx", "miss_MCRT", &miss_program_MCRT ) );
  RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_DIRECT_MCRT, miss_program_MCRT ) );
  RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_DIFFUSE_MCRT, miss_program_MCRT ) );
  RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_EMISSION_MCRT, miss_program_MCRT ) );

  /* Create OptiX Geometry Structures */

  RTtransform transform;

  RTgeometrygroup geometry_group;

  RTgeometryinstance patch_instance;
  RTgeometryinstance triangle_instance;
  RTgeometryinstance disk_instance;
  RTgeometryinstance alphamask_instance;
  RTgeometryinstance voxel_instance;

  /* Create top level group and associated (dummy) acceleration */
  RT_CHECK_ERROR( rtGroupCreate( OptiX_Context, &top_level_group ) );
  RT_CHECK_ERROR( rtGroupSetChildCount( top_level_group, 1 ) );
   
  RT_CHECK_ERROR( rtAccelerationCreate( OptiX_Context, &top_level_acceleration ) );
  RT_CHECK_ERROR( rtAccelerationSetBuilder(top_level_acceleration,"NoAccel") );
  RT_CHECK_ERROR( rtAccelerationSetTraverser(top_level_acceleration,"NoAccel") );
  RT_CHECK_ERROR( rtGroupSetAcceleration( top_level_group, top_level_acceleration) );
   
  /* mark acceleration as dirty */
  RT_CHECK_ERROR( rtAccelerationMarkDirty( top_level_acceleration ) );

  /* Create transform node */
  RT_CHECK_ERROR( rtTransformCreate( OptiX_Context, &transform ) );
  float m[16];
  m[0] = 1.f; m[1] = 0; m[2] = 0; m[3] = 0;
  m[4] = 0.f; m[5] = 1.f; m[6] = 0; m[7] = 0;
  m[8] = 0.f; m[9] = 0; m[10] = 1.f; m[11] = 0;
  m[12] = 0.f; m[13] = 0; m[14] = 0; m[15] = 1.f;
  RT_CHECK_ERROR( rtTransformSetMatrix( transform, 0, m, 0 ) );
  RT_CHECK_ERROR( rtGroupSetChild( top_level_group, 0, transform ) );
  
  /* Create geometry group and associated acceleration*/
  RT_CHECK_ERROR( rtGeometryGroupCreate( OptiX_Context, &geometry_group ) );
  RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometry_group, 5 ) );
  RT_CHECK_ERROR( rtTransformSetChild( transform, geometry_group ) );

  //create acceleration object for group and specify some build hints
  RT_CHECK_ERROR( rtAccelerationCreate(OptiX_Context,&geometry_acceleration) );
  RT_CHECK_ERROR( rtAccelerationSetBuilder(geometry_acceleration,"Trbvh") );
  RT_CHECK_ERROR( rtAccelerationSetTraverser(geometry_acceleration,"Bvh") );
  RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometry_group, geometry_acceleration) );
  RT_CHECK_ERROR( rtAccelerationMarkDirty( geometry_acceleration ) );

  /* Create geometry instances */
  //patches
  RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &patch_instance ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( patch_instance, patch ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( patch_instance, 1 ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( patch_instance, 0, patch_material ) );
  RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 0, patch_instance ) );
  //triangles
  RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &triangle_instance ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( triangle_instance, triangle ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( triangle_instance, 1 ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( triangle_instance, 0, triangle_material ) );
  RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 1, triangle_instance ) );
  //disks
  RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &disk_instance ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( disk_instance, disk ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( disk_instance, 1 ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( disk_instance, 0, disk_material ) );
  RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 2, disk_instance ) );
  //alphamasks
  RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &alphamask_instance ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( alphamask_instance, alphamask ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( alphamask_instance, 1 ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( alphamask_instance, 0, alphamask_material ) );
  RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 3, alphamask_instance ) );
  //voxels
  RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &voxel_instance ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( voxel_instance, voxel ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( voxel_instance, 1 ) );
  RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( voxel_instance, 0, voxel_material ) );
  RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 4, voxel_instance ) );

  /* Set the top_object variable */
  //NOTE: Not sure exactly where this has to be set
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "top_object", &top_object ) );
  RT_CHECK_ERROR( rtVariableSetObject( top_object, top_level_group ) );

  //random number seed
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "random_seed", &random_seed_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( random_seed_RTvariable, std::chrono::system_clock::now().time_since_epoch().count() ) );

  //maximum scattering depth
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "max_scatters", &max_scatters_RTvariable ) );
  RT_CHECK_ERROR( rtVariableSet1ui( max_scatters_RTvariable, 0 ) );
  
  // RTsize device_memory;
  // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

  // device_memory *= 1e-6; 
  // if( device_memory < 1000 ){
  //   printf("available device memory at end of OptiX initialization: %6.3f MB\n",device_memory);
  // }else{
  //   printf("available device memory at end of OptiX initialization: %6.3f GB\n",device_memory*1e-3);
  // }
  
}

void RadiationModel::updateGeometry( void ){
  updateGeometry( context->getAllUUIDs() );
}

void RadiationModel::updateGeometry( const std::vector<uint> UUIDs ){

  if( message_flag ){
    std::cout << "Updating geometry in radiation transport model..." << std::flush;
  }

  context_UUIDs = UUIDs;

  for( size_t u=0; u<context_UUIDs.size(); u++ ){
    if( !context->doesPrimitiveExist( context_UUIDs.at(u) ) ){
      context_UUIDs.erase( context_UUIDs.begin()+u );
    }
  }

  //--- Make Bounding Patches ---//

  //determine domain bounding sphere
  float sphere_radius;
  vec3 sphere_center;
  context->getDomainBoundingSphere( sphere_center, sphere_radius );
  
  rtVariableSet1f( bound_sphere_radius_RTvariable, sphere_radius );
  rtVariableSet3f( bound_sphere_center_RTvariable, sphere_center.x, sphere_center.y, sphere_center.z );

  //--- Populate Primitive Geometry Buffers ---//

  size_t Nprimitives = context_UUIDs.size(); //Number of primitives

  uint Nbands = emission_flag.size(); //Number of spectral bands

  //transformation matrix buffer
  std::vector<std::vector<float> > m_global;
  m_global.resize(Nprimitives);

  //primitive type buffer
  std::vector<uint> ptype_global;
  ptype_global.resize(Nprimitives);

  //primitive area buffer
  std::vector<float> area_global;
  area_global.resize(Nprimitives);

  //primitive UUID buffers
  std::vector<uint> patch_UUID;
  std::vector<uint> triangle_UUID;
  std::vector<uint> disk_UUID;
  std::vector<uint> alphamask_UUID;
  std::vector<uint> voxel_UUID;

  //twosided flag buffer
  std::vector<bool> twosided_flag_global;
  twosided_flag_global.resize(Nprimitives);//initialize to be two-sided
  for( size_t i=0; i<Nprimitives; i++ ){
    twosided_flag_global.at(i) = true;
  }
  
  //primitive geometry specification buffers
  std::vector<std::vector<optix::float3> > patch_vertices;
  std::vector<std::vector<optix::float3> > triangle_vertices;
  std::vector<std::vector<optix::float3> > disk_vertices;
  std::vector<std::vector<optix::float3> > alphamask_vertices;
  std::vector<std::vector<optix::float3> > voxel_vertices;

  std::size_t patch_count = 0;
  std::size_t triangle_count = 0;
  std::size_t disk_count = 0;
  std::size_t alphamask_count = 0;
  std::size_t voxel_count = 0;
  for( std::size_t u=0; u<Nprimitives; u++ ){
    uint p = context_UUIDs.at(u);
    
    helios::Primitive* prim = context->getPrimitivePointer(p);

    //transformation matrix
    float m[16];
    prim->getTransformationMatrix(m);

    m_global.at(u).resize(16);
    for( uint i=0; i<16; i++ ){
      m_global.at(u).at(i) = m[i];
    }

    //primitive type
    helios::PrimitiveType type = prim->getType();
    ptype_global.at(u) =  type;
      
    //primitve area
    area_global.at(u) = prim->getArea();

    //primitive twosided flag
    if( prim->doesPrimitiveDataExist("twosided_flag") ){
      uint flag;
      prim->getPrimitiveData("twosided_flag",flag);
      if( flag ){
    	twosided_flag_global.at(u) = true;
      }else{
  	twosided_flag_global.at(u) = false;
      }
    }

    //primitive geometry
    if( type == helios::PRIMITIVE_TYPE_PATCH ){
      std::vector<vec3> vertices = prim->getVertices();
      std::vector<optix::float3> v;
      v.push_back( optix::make_float3(vertices.at(0).x,vertices.at(0).y,vertices.at(0).z) );
      v.push_back( optix::make_float3(vertices.at(1).x,vertices.at(1).y,vertices.at(1).z) );
      v.push_back( optix::make_float3(vertices.at(2).x,vertices.at(2).y,vertices.at(2).z) );
      v.push_back( optix::make_float3(vertices.at(3).x,vertices.at(3).y,vertices.at(3).z) );
      patch_vertices.push_back(v);
      patch_UUID.push_back(u);
      patch_count ++;
    }else if( type == helios::PRIMITIVE_TYPE_TRIANGLE ){
      std::vector<vec3> vertices = prim->getVertices();
      std::vector<optix::float3> v;
      v.push_back( optix::make_float3(vertices.at(0).x,vertices.at(0).y,vertices.at(0).z) );
      v.push_back( optix::make_float3(vertices.at(1).x,vertices.at(1).y,vertices.at(1).z) );
      v.push_back( optix::make_float3(vertices.at(2).x,vertices.at(2).y,vertices.at(2).z) );
      triangle_vertices.push_back(v);
      triangle_UUID.push_back(u);
      triangle_count ++;
    }else if( type == helios::PRIMITIVE_TYPE_VOXEL ){
      helios::vec3 center = context->getVoxelPointer(p)->getCenter();
      helios::vec3 size = context->getVoxelPointer(p)->getSize(); 
      std::vector<optix::float3> v;
      v.push_back( optix::make_float3(center.x-0.5*size.x, center.y-0.5*size.y, center.z-0.5*size.z ) );
      v.push_back( optix::make_float3(center.x+0.5*size.x, center.y+0.5*size.y, center.z+0.5*size.z ) );
      voxel_vertices.push_back(v);
      voxel_UUID.push_back(u);
      voxel_count ++;
    }

  }

  initializeBuffer2Df( transform_matrix_RTbuffer, m_global );
  initializeBuffer1Dui( primitive_type_RTbuffer, ptype_global );
  initializeBuffer1Df( primitive_area_RTbuffer, area_global );
  initializeBuffer1Dbool( twosided_flag_RTbuffer, twosided_flag_global );
  initializeBuffer2Dfloat3( patch_vertices_RTbuffer, patch_vertices );
  initializeBuffer2Dfloat3( triangle_vertices_RTbuffer, triangle_vertices );
  initializeBuffer2Dfloat3( alphamask_vertices_RTbuffer, alphamask_vertices );
  initializeBuffer2Dfloat3( voxel_vertices_RTbuffer, voxel_vertices );
  
  initializeBuffer1Dui( patch_UUID_RTbuffer, patch_UUID );
  initializeBuffer1Dui( triangle_UUID_RTbuffer, triangle_UUID );
  initializeBuffer1Dui( disk_UUID_RTbuffer, disk_UUID );
  initializeBuffer1Dui( alphamask_UUID_RTbuffer, alphamask_UUID );
  initializeBuffer1Dui( voxel_UUID_RTbuffer, voxel_UUID );

  //Texture mask data
  std::vector<std::vector<std::vector<bool> > > maskdata;
  std::map<std::string,uint> maskname;
  std::vector<optix::int2> masksize;
  std::vector<int> maskID;
  std::vector<std::vector<optix::float2> > uvdata;
  std::vector<int> uvID;
  maskID.resize(context->getPrimitiveCount());
  uvID.resize(context->getPrimitiveCount());

  for( size_t u=0; u<Nprimitives; u++ ){
    uint p = context_UUIDs.at(u);
    Primitive* prim = context->getPrimitivePointer(p);
    std::string maskfile = prim->getTextureFile();
    
    if( context->getPrimitivePointer(p)->getType()==PRIMITIVE_TYPE_VOXEL || maskfile.size()==0 || !prim->getTexture()->hasTransparencyChannel() ){ //does not have texture transparency
      
      maskID.at(u) = -1;
      uvID.at(u) = -1;
      
    }else{
      
      // texture mask data //
      
      //Check if this mask has already been added
      if( maskname.find(maskfile) != maskname.end() ){ //has already been added
  	uint ID = maskname.at(maskfile);
  	maskID.at(u) = ID;
      }else{ //mask has not been added
  	uint ID = maskdata.size();
  	maskID.at(u) = ID;
  	maskname[maskfile] = maskdata.size();
  	maskdata.push_back( *context->getPrimitivePointer(p)->getTexture()->getTransparencyData() );
  	uint sy = maskdata.back().size();
  	uint sx = maskdata.back().front().size();
  	masksize.push_back( optix::make_int2(sx,sy) );
      }

      // uv coordinates //
      std::vector<vec2> uv = context->getPrimitivePointer(p)->getTextureUV();
      if( uv.size()!=0 ){ //has custom (u,v) coordinates
  	std::vector<optix::float2> uvf2;
  	uvf2.resize(4);
	//first index if uvf2 is the minimum (u,v) coordinate, second index is the size of the (u,v) rectangle in x- and y-directions.

  	for( int i=0; i<uv.size(); i++ ){
  	  uvf2.at(i) = optix::make_float2(uv.at(i).x, uv.at(i).y);
  	}
  	if( uv.size()==3 ){
  	  uvf2.at(3) = optix::make_float2(0,0);
  	}
  	uvdata.push_back( uvf2 );
  	uvID.at(u) = uvdata.size()-1;
      }else{ //DOES NOT have custom (u,v) coordinates
  	uvID.at(u) = -1;
      }
      
    }
  }

  initializeBuffer3D( maskdata_RTbuffer, maskdata );
  initializeBuffer1Dint2( masksize_RTbuffer, masksize );
  initializeBuffer1Di( maskID_RTbuffer, maskID );

  initializeBuffer2Dfloat2( uvdata_RTbuffer, uvdata );
  initializeBuffer1Di( uvID_RTbuffer, uvID );

  RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( patch, patch_count ) );
  RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( triangle, triangle_count ) );
  RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( disk, disk_count ) );
  RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( alphamask, alphamask_count ) );
  RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( voxel, voxel_count ) );

  RT_CHECK_ERROR( rtAccelerationMarkDirty( geometry_acceleration ) );
  
  /* Set the top_object variable */
  //NOTE: not sure if this has to be set again or not..
  RT_CHECK_ERROR( rtVariableSetObject( top_object, top_level_group ) );

  RTsize device_memory;
  RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

  device_memory *= 1e-6;

  if( device_memory<500 ){
    std::cout << "WARNING (RadiationModel): device memory is very low (" << device_memory << " MB)" << std::endl;
  }

  /* Validate/Compile OptiX Context */
  RT_CHECK_ERROR( rtContextValidate( OptiX_Context ) );
  RT_CHECK_ERROR( rtContextCompile( OptiX_Context ) );

  // device_memory;
  // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

  // device_memory *= 1e-6; 
  // if( device_memory < 1000 ){
  //   printf("available device memory at end of OptiX context compile: %6.3f MB\n",device_memory);
  // }else{
  //   printf("available device memory at end of OptiX context compile: %6.3f GB\n",device_memory*1e-3);
  // }
  
  isgeometryinitialized = true;

  // device_memory;
  // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

  // device_memory *= 1e-6; 
  // if( device_memory < 1000 ){
  //   printf("available device memory before acceleration build: %6.3f MB\n",device_memory);
  // }else{
  //   printf("available device memory before acceleration build: %6.3f GB\n",device_memory*1e-3);
  // }

  optix::int3 launch_dim_dummy = optix::make_int3( 1, 1, 1 );
  RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIRECT , launch_dim_dummy.x, launch_dim_dummy.y, launch_dim_dummy.z ) );

  RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

  // device_memory;
  // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

  // device_memory *= 1e-6; 
  // if( device_memory < 1000 ){
  //   printf("available device memory at end of acceleration build: %6.3f MB\n",device_memory);
  // }else{
  //   printf("available device memory at end of acceleration build: %6.3f GB\n",device_memory*1e-3);
  // }

  if( message_flag ){
    std::cout << "done." << std::endl;
  }
    
}

void RadiationModel::updateRadiativeProperties( const char* label ){

  const uint Nbands = emission_flag.size();//number of radiative bands

  uint band = band_names.at(label);
  
  std::vector<float> rho, tau, eps;
  
  char prop[20];

  size_t Nprimitives = context_UUIDs.size();

  rho.resize(Nprimitives);
  tau.resize(Nprimitives);
  eps.resize(Nprimitives);

  for( size_t u=0; u<Nprimitives; u++ ){

    uint p = context_UUIDs.at(u);

    helios::Primitive* prim = context->getPrimitivePointer(p);

    helios::PrimitiveType type = prim->getType();
    uint UUID = prim->getUUID();

    isbandpropertyinitialized.at(band) = true;
    
    if( type==helios::PRIMITIVE_TYPE_VOXEL ){

      // NOTE: This is a little confusing - for volumes of participating media, we're going to use the "rho" variable to store the absorption coefficient and use the "tau" variable to store the scattering coefficient.  This is to save on memory so we don't have to define separate arrays.

      // Absorption coefficient

      sprintf(prop,"attenuation_coefficient_%s",label);
    
      if( prim->doesPrimitiveDataExist(prop) ){
	prim->getPrimitiveData(prop,rho[u]);
      }else{
	rho[u] = kappa_default;
	prim->setPrimitiveData(prop,helios::HELIOS_TYPE_FLOAT,1,&kappa_default);  
      }

      if( rho[u]<0 ){
	rho[u] = 0.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): absorption coefficient cannot be less than 0.  Clamping to 0 for band " << band << "." << std::endl;
	}
      }else if( rho[u]>1.f ){
	rho[u] = 1.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): absorption coefficient cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::endl;
	}
      }

      // Scattering coefficient

      sprintf(prop,"scattering_coefficient_%s",label);

      if( prim->doesPrimitiveDataExist(prop) ){
	prim->getPrimitiveData(prop,tau[u]);
      }else{
	tau[u] = sigmas_default;
	prim->setPrimitiveData(prop,helios::HELIOS_TYPE_FLOAT,1,&sigmas_default);
      }
      
      if( tau[u]<0 ){
	tau[u] = 0.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): scattering coefficient cannot be less than 0.  Clamping to 0 for band " << band << "." << std::endl;
	}
      }else if( tau[u]>1.f ){
	tau[u] = 1.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): scattering coefficient cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::endl;
	}
      }

    }else{ //other than voxels

      // Reflectivity 

      sprintf(prop,"reflectivity_%s",label);

      if( prim->doesPrimitiveDataExist(prop) ){
	prim->getPrimitiveData(prop,rho[u]);
      }else{
	rho[u] = rho_default;
	prim->setPrimitiveData(prop,helios::HELIOS_TYPE_FLOAT,1,&rho_default);  
      }

      if( rho[u]<0 ){
	rho[u] = 0.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): reflectivity cannot be less than 0.  Clamping to 0 for band " << band << "." << std::endl;
	}
      }else if( rho[u]>1.f ){
	rho[u] = 1.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): reflectivity cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::endl;
	}
      }
      
      // Transmissivity
      
      sprintf(prop,"transmissivity_%s",label);
      
      if( prim->doesPrimitiveDataExist(prop) ){
	prim->getPrimitiveData(prop,tau[u]);
      }else{
	tau[u] = tau_default;
	prim->setPrimitiveData(prop,helios::HELIOS_TYPE_FLOAT,1,&tau_default);
      }

      if( tau[u]<0 ){
	tau[u] = 0.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): transmissivity cannot be less than 0.  Clamping to 0 for band " << band << "." << std::endl;
	}
      }else if( tau[u]>1.f ){
	tau[u] = 1.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): transmissivity cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::endl;
	}
      }

      // Emissivity
      
      sprintf(prop,"emissivity_%s",label);

      if( prim->doesPrimitiveDataExist(prop) ){
	prim->getPrimitiveData(prop,eps[u]);
      }else{
	eps[u] = eps_default;
	prim->setPrimitiveData(prop,helios::HELIOS_TYPE_FLOAT,1,&eps_default);
      }
    
      if( eps[u]<0 ){
	eps[u] = 0.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): emissivity cannot be less than 0.  Clamping to 0 for band " << band << "." << std::endl;
	}
      }else if( eps[u]>1.f ){
	eps[u] = 1.f;
	if( message_flag ){
	  std::cout << "WARNING (RadiationModel): emissivity cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::endl;
	}
      }
      
      if( emission_flag.at(band) ){ //emission enabled
	if( eps[u]+tau[u]+rho[u]!=1.f && eps[u]>0.f ){
	  std::cerr << "ERROR (RadiationModel): emissivity, transmissivity, and reflectivity must sum to 1 to ensure energy conservation. Band " << label << ", Primitive #" << p << ": eps=" << eps[u] << ", tau=" << tau[u] << ", rho=" << rho[u] << std::endl;
	  exit(EXIT_FAILURE);
	}
      }else if( tau[u]+rho[u]>1.f ){
      	std::cerr << "ERROR (RadiationModel): transmissivity and reflectivity cannot sum to greater than 1 ensure energy conservation. Band " << label << ", Primitive #" << p << ": tau=" << tau[u] << ", rho=" << rho[u] << std::endl;
	exit(EXIT_FAILURE);
      }
 
    }
      

  }

  initializeBuffer1Df( rho_RTbuffer, rho );
  initializeBuffer1Df( tau_RTbuffer, tau );
  initializeBuffer1Df( eps_RTbuffer, eps );

}

void RadiationModel::runBand( const char* label ){

  //----- VERIFICATIONS -----//

  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (runBand): Cannot run band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint band = band_names.at(label);

  //Check to make sure some geometry was added to the context
  if( context->getPrimitiveCount()==0 ){
    std::cerr << "WARNING (runSimulation): No geometry was added to the context. There is nothing to simulate...exiting." << std::endl;
    return;
  }

  //Check to make sure geometry was built in OptiX
  if( !isgeometryinitialized ){
    std::cerr << "ERROR (runBand): Geometry must be built before running the simulation. Please call updateGeometry() before runBand(). " << std::endl;
    exit(EXIT_FAILURE);
  }

  //Check to make sure a band was actually added
  uint Nbands = emission_flag.size();
  if( band>=Nbands ){
    std::cerr << "ERROR (runBand): Cannot run simulation for radiation band #" << band << " because there are only " << Nbands << " bands (see addRadiationBand)." << std::endl;
    exit(EXIT_FAILURE);
  }

  //Set the random number seed
  RT_CHECK_ERROR( rtVariableSet1ui( random_seed_RTvariable, std::chrono::system_clock::now().time_since_epoch().count() ) );

  //Number of external radiation sources for this band
  uint Nsources = source_fluxes.at(label).size();
  RT_CHECK_ERROR( rtVariableSet1ui( Nsources_RTvariable, Nsources ) );
  
  updateRadiativeProperties(label);

  size_t Nprimitives = context_UUIDs.size();

  bool to_be_scattered;

  // Zero buffers
  zeroBuffer1D( radiation_in_RTbuffer, Nprimitives );
  zeroBuffer1D( scatter_buff_top_RTbuffer, Nprimitives );
  zeroBuffer1D( scatter_buff_bottom_RTbuffer, Nprimitives );
  //zeroBuffer1D( Rsky_RTbuffer, 1 );
  zeroBuffer1D( Rsky_RTbuffer, Nprimitives );

  std::vector<float> TBS_top, TBS_bottom;
  TBS_top.resize(context->getPrimitiveCount(),0.f);
  TBS_bottom = TBS_top;

  // -- Direct launch --//

  optix::int3 launch_dim_dir;

  if( Nsources>0 ){

    std::vector<float> fluxes = source_fluxes.at(label);
    float f = 0;
    for( uint i=0; i<Nsources; i++ ){
      f+=fluxes.at(i);
    }

    if( f>0 ){ //at least one nonzero flux this band
    
      //update radiation source buffers
      initializeBuffer1Df( source_fluxes_RTbuffer, fluxes );
      std::vector<optix::float3> positions;
      std::vector<float> widths;
      std::vector<uint> types;
      for( uint i=0; i<Nsources; i++ ){
    	positions.push_back( optix::make_float3( source_positions.at(i).x, source_positions.at(i).y, source_positions.at(i).z ) );
	widths.push_back( source_widths.at(i) );
    	types.push_back( source_types.at(i) );
      }
      initializeBuffer1Dfloat3( source_positions_RTbuffer, positions );
      initializeBuffer1Df( source_widths_RTbuffer, widths );
      initializeBuffer1Dui( source_types_RTbuffer, types );
      
      // -- Ray Trace -- //
      
      // Compute direct launch dimension
      size_t n = ceil(sqrt(double(directRayCount[band])));
      launch_dim_dir = optix::make_int3( round(n), round(n), Nprimitives );
      
      assert( launch_dim_dir.x>0 && launch_dim_dir.y>0 );
      
      if( message_flag ){
    	std::cout << "Performing primary direct radiation ray trace for band " << label << "..." << std::flush;
      }
      RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIRECT , launch_dim_dir.x, launch_dim_dir.y, launch_dim_dir.z ) );
      if( message_flag ){
    	std::cout << "done." << std::endl;
      }
      
      if( scatteringDepth.at(band)==0 ){
    	to_be_scattered = false;
	//deposit rest of energy
	TBS_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
	TBS_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );
      }else{
    	to_be_scattered = true;
      }

    }
  }
  
  // --- Diffuse/Emission launch ---- //

  if( emission_flag[band] || to_be_scattered || diffuseFlux.at(band)>0.f ){

    if( emission_flag[band] || diffuseFlux.at(band)>0.f ){
    
      /* Set Diffuse Flux Variable */
      RT_CHECK_ERROR( rtVariableSet1f( diffuseFlux_RTvariable, diffuseFlux.at(band) ));
      
      std::vector<float> flux_top, flux_bottom;
      flux_top.resize(Nprimitives,0.f);
      flux_bottom.resize(Nprimitives,0.f);
      
      if( to_be_scattered ){
  	flux_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
  	flux_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );
  	zeroBuffer1D( scatter_buff_top_RTbuffer, Nprimitives );
  	zeroBuffer1D( scatter_buff_bottom_RTbuffer, Nprimitives );
      }
      
      if( emission_flag[band] ){
  	// Update primitive outgoing emission
  	float eps, temperature;
  	char prop[30];
  	sprintf(prop,"emissivity_%s",label);
  	for( size_t u=0; u<Nprimitives; u++ ){
	  uint p = context_UUIDs.at(u);
  	  context->getPrimitiveData(p,prop,eps);
	  if( context->doesPrimitiveDataExist(p,"temperature") ){
	    context->getPrimitiveData(p,"temperature",temperature);
	    if( temperature<=0 ){
	      temperature = temperature_default;
	    }
	  }else{
	    temperature = temperature_default;
	  }
  	  flux_top.at(u) += 5.67e-8*eps*pow(temperature,4);
  	  if( !context->doesPrimitiveDataExist(p,"twosided_flag") ){
	    flux_bottom.at(u) += flux_top.at(u);
	  }else{
  	    uint flag;
  	    context->getPrimitiveData(p,"twosided_flag",flag);	    
  	    if( flag ){
  	      flux_bottom.at(u) += flux_top.at(u);
  	    }
  	  }
  	}
      }
      
      initializeBuffer1Df( radiation_out_top_RTbuffer, flux_top );
      initializeBuffer1Df( radiation_out_bottom_RTbuffer, flux_bottom );
      
      // Compute diffuse launch dimension
      size_t n = ceil(sqrt(double(diffuseRayCount[band])));
      optix::int3 launch_dim_diff = optix::make_int3( round(n), round(n), Nprimitives );
      assert( launch_dim_diff.x>0 && launch_dim_diff.y>0 );
      
      if( message_flag ){
  	std::cout << "Performing primary diffuse radiation ray trace for band " << label << " ..." << std::flush;
      }
      RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIFFUSE , launch_dim_diff.x, launch_dim_diff.y, launch_dim_diff.z ) );
      if( message_flag ){
  	std::cout << "done." << std::endl;
      }
    }
    
    if( scatteringDepth.at(band)>0 ){

      RT_CHECK_ERROR( rtVariableSet1f( diffuseFlux_RTvariable, 0.f ));

      size_t n = ceil(sqrt(double(diffuseRayCount[band])));
      optix::int3 launch_dim_diff = optix::make_int3( round(n), round(n), Nprimitives );
      uint s;
      for( s=0; s<scatteringDepth.at(band); s++ ){
    	if( message_flag ){
    	  std::cout << "Performing scattering ray trace (iteration " << s+1 << " of " << scatteringDepth.at(band) << ")..." << std::flush;
    	}

	TBS_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
	TBS_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );
	float TBS_max = 0;
	for( size_t u=0; u<Nprimitives; u++ ){
	  size_t p = context_UUIDs.at(u);
	  if( TBS_top.at(u)+TBS_bottom.at(u)>TBS_max ){
	    TBS_max = TBS_top.at(u)+TBS_bottom.at(u);
	  }
	}
	// if( TBS_max<=minScatterEnergy.at(band) ){
	//   std::cout << "Terminating at " << s << " scatters: max to-be-scattered energy " << TBS_max << std::endl;
	//   break;
	// }
	
    	copyBuffer1D( scatter_buff_top_RTbuffer, radiation_out_top_RTbuffer );
    	zeroBuffer1D( scatter_buff_top_RTbuffer, Nprimitives );
    	copyBuffer1D( scatter_buff_bottom_RTbuffer, radiation_out_bottom_RTbuffer );
    	zeroBuffer1D( scatter_buff_bottom_RTbuffer, Nprimitives );
    	RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIFFUSE , launch_dim_diff.x, launch_dim_diff.y, launch_dim_diff.z ) );
    	if( message_flag ){
    	  std::cout << "\r                                                                            \r" << std::flush;
    	}
      }

      //deposit anything that is left to make sure we satsify conservation of energy
      if( s==scatteringDepth.at(band) ){
	TBS_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
	TBS_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );
      }
      
      if( message_flag ){
    	std::cout << "Performing scattering ray trace for band " << label << " ...done." << std::endl;
      }

    }else{//deposit anything that is left to make sure we satsify conservation of energy
      TBS_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
      TBS_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );
    }   
  }

  // Set variables in geometric objects
    
  std::vector<float> radiation_flux_data;
  radiation_flux_data=getOptiXbufferData( radiation_in_RTbuffer );
  
  char prop[30];
  sprintf(prop,"radiation_flux_%s",label);
  for( size_t u=0; u<Nprimitives; u++ ){
    size_t p = context_UUIDs.at(u);
    float R = radiation_flux_data.at(u)+TBS_top.at(u)+TBS_bottom.at(u);
    context->setPrimitiveData(p,prop,R);
    if( radiation_flux_data.at(u)!=radiation_flux_data.at(u) ){
      std::cout << "NaN here " << p << std::endl;
    }
  }

}

void RadiationModel::runBand_MCRT( const char* label ){

  //----- VERIFICATIONS -----//

  if( band_names.find(label) == band_names.end() ){
    std::cerr << "ERROR (runBand): Cannot run band " << label << " because it is not a valid band." << std::endl;
    exit(EXIT_FAILURE);
  }
  uint band = band_names.at(label);

  //Check to make sure some geometry was added to the context
  if( context->getPrimitiveCount()==0 ){
    std::cerr << "ERROR (runSimulation): No geometry was added to the context. There is nothing to simulate...exiting." << std::endl;
    exit(EXIT_FAILURE);
  }

  //Check to make sure geometry was built in OptiX
  if( !isgeometryinitialized ){
    std::cerr << "ERROR (runBand): Geometry must be built before running the simulation. Please call updateGeometry() before runBand(). " << std::endl;
    exit(EXIT_FAILURE);
  }

  //Check to make sure a band was actually added
  uint Nbands = emission_flag.size();
  if( band>=Nbands ){
    std::cerr << "ERROR (runBand): Cannot run simulation for radiation band #" << band << " because there are only " << Nbands << " bands (see addRadiationBand)." << std::endl;
    exit(EXIT_FAILURE);
  }

  //Set the random number seed
  RT_CHECK_ERROR( rtVariableSet1ui( random_seed_RTvariable, std::chrono::system_clock::now().time_since_epoch().count() ) );

  //Set the scattering depth
  RT_CHECK_ERROR( rtVariableSet1ui( max_scatters_RTvariable, scatteringDepth.at(band) ) );
  
  //Number of external radiation sources for this band
  uint Nsources = source_fluxes.at(label).size();
  RT_CHECK_ERROR( rtVariableSet1ui( Nsources_RTvariable, Nsources ) );
    
  updateRadiativeProperties(label);

  size_t Nprimitives = context_UUIDs.size();

  // Zero buffers
  zeroBuffer1D( radiation_in_RTbuffer, Nprimitives );
  //zeroBuffer1D( Rsky_RTbuffer, 1 );
  zeroBuffer1D( Rsky_RTbuffer, Nprimitives );
  
  // -- Direct launch --//

  if( Nsources>0 ){

    std::vector<float> fluxes = source_fluxes.at(label);
    float f = 0;
    for( uint i=0; i<Nsources; i++ ){
      f+=fluxes.at(i);
    }

    if( f>0 ){ //at least one nonzero flux this band
    
      //update radiation source buffers
      initializeBuffer1Df( source_fluxes_RTbuffer, fluxes );
      std::vector<optix::float3> positions;
      std::vector<float> widths;
      std::vector<uint> types;
      for( uint i=0; i<Nsources; i++ ){
    	positions.push_back( optix::make_float3( source_positions.at(i).x, source_positions.at(i).y, source_positions.at(i).z ) );
	widths.push_back( source_widths.at(i) );
    	types.push_back( source_types.at(i) );
      }
      initializeBuffer1Dfloat3( source_positions_RTbuffer, positions );
      initializeBuffer1Df( source_widths_RTbuffer, widths );
      initializeBuffer1Dui( source_types_RTbuffer, types );
      
      // -- Ray Trace -- //
      
      // Compute direct launch dimension
      size_t n = ceil(sqrt(double(directRayCount[band])));
      optix::int2 launch_dim_dir = optix::make_int2( round(n), round(n) );
      
      assert( launch_dim_dir.x>0 && launch_dim_dir.y>0 );
      
      if( message_flag ){
    	std::cout << "Performing direct radiation ray trace for band " << label << "..." << std::flush;
      }
      RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIRECT_MCRT , launch_dim_dir.x, launch_dim_dir.y, 1) );
      if( message_flag ){
    	std::cout << "done." << std::endl;
      }

    }
  }
  
  // --- Diffuse launch ---- //

  if( diffuseFlux.at(band)>0.f ){

    /* Set Diffuse Flux Variable */
    RT_CHECK_ERROR( rtVariableSet1f( diffuseFlux_RTvariable, diffuseFlux.at(band) ));
      
    // Compute diffuse launch dimension
    size_t n = ceil(sqrt(double(diffuseRayCount[band])));
    optix::int2 launch_dim_diff = optix::make_int2( round(n), round(n) );
    assert( launch_dim_diff.x>0 && launch_dim_diff.y>0 );
      
    if( message_flag ){
      std::cout << "Performing  diffuse radiation ray trace for band " << label << " ..." << std::flush;
    }
    RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIFFUSE_MCRT , launch_dim_diff.x, launch_dim_diff.y, 1 ) );
    if( message_flag ){
      std::cout << "done." << std::endl;
    }

  }

  // --- Emission launch ---- //

  if( emission_flag[band] ){

    size_t Nprimitives = context_UUIDs.size();

    std::vector<float> flux_top, flux_bottom;
    flux_top.resize(Nprimitives,0.f);
    flux_bottom.resize(Nprimitives,0.f);
    
    // Update primitive outgoing emission
    float eps, temperature;
    char prop[30];
    sprintf(prop,"emissivity_%s",label);
    for( size_t u=0; u<Nprimitives; u++ ){
      uint p = context_UUIDs.at(u);
      context->getPrimitiveData(p,prop,eps);
      if( context->doesPrimitiveDataExist(p,"temperature") ){
	context->getPrimitiveData(p,"temperature",temperature);
	if( temperature<=0 ){
	  temperature = temperature_default;
	}
      }else{
	temperature = temperature_default;
      }
      flux_top.at(u) += 5.67e-8*eps*pow(temperature,4);
      if( !context->doesPrimitiveDataExist(p,"twosided_flag") ){
	flux_bottom.at(u) += flux_top.at(u);
      }else{
	uint flag;
	context->getPrimitiveData(p,"twosided_flag",flag);
	if( flag ){
	  flux_bottom.at(u) += flux_top.at(u);
	}
      }
    }

    initializeBuffer1Df( radiation_out_top_RTbuffer, flux_top );
    initializeBuffer1Df( radiation_out_bottom_RTbuffer, flux_bottom );
      
    // Compute emission launch dimension
    size_t n = ceil(sqrt(double(directRayCount[band])));//TODO: fix this - should not be based on 'directRayCount'
    optix::int3 launch_dim_emiss = optix::make_int3( round(n), round(n), Nprimitives );
    assert( launch_dim_emiss.x>0 && launch_dim_emiss.y>0 );
      
    if( message_flag ){
      std::cout << "Performing radiation emission ray trace for band " << label << " ..." << std::flush;
    }
    RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_EMISSION_MCRT , launch_dim_emiss.x, launch_dim_emiss.y, launch_dim_emiss.z ) );
    if( message_flag ){
      std::cout << "done." << std::endl;
    }
  
  }

  // Set variables in geometric objects
    
  std::vector<float> radiation_flux_data;
  radiation_flux_data=getOptiXbufferData( radiation_in_RTbuffer );

  char prop[30];
  sprintf(prop,"radiation_flux_%s",label);
  for( size_t u=0; u<context_UUIDs.size(); u++ ){
    size_t p = context_UUIDs.at(u);
    float R = radiation_flux_data.at(u);
    context->setPrimitiveData(p,prop,R);
    if( radiation_flux_data.at(u)!=radiation_flux_data.at(u) ){
      std::cout << "NaN here " << p << std::endl;
    }
  }

  if( message_flag ){
    std::cout << "|--- Finished radiation transport ----!" << std::endl;
  }
    
}

float RadiationModel::getSkyEnergy( void ){

  std::vector<float> Rsky_SW;
  Rsky_SW=getOptiXbufferData( Rsky_RTbuffer );
  float Rsky=0.f;
  for( size_t i=0; i<Rsky_SW.size(); i++ ){
    Rsky += Rsky_SW.at(i);
  }
  return Rsky;
  
}

std::vector<float> RadiationModel::getTotalAbsorbedFlux( void ){

  std::vector<float> total_flux;
  total_flux.resize(context->getPrimitiveCount(),0.f);

  for( std::map<std::string,uint>::iterator iter = band_names.begin(); iter != band_names.end(); iter++ ){
    
    std::string label = iter->first;

    for( size_t u=0; u<context_UUIDs.size(); u++ ){

      uint p = context_UUIDs.at(u);
      
      char str[50];
      printf(str,"radiation_flux_%s",label.c_str());

      float R;
      context->getPrimitiveData( p, str, R );
      total_flux.at(u) += R;

    }

  }

  return total_flux;
  
}

std::vector<float> RadiationModel::getOptiXbufferData( RTbuffer buffer ){

  void* _data_;
  RT_CHECK_ERROR( rtBufferMap( buffer, &_data_ ) );
  float* data_ptr = (float*)_data_;

  unsigned long int size;
  RT_CHECK_ERROR( rtBufferGetSize1D( buffer, &size ) );

  std::vector<float> data_vec;
  data_vec.resize(size);
  for( int i=0; i<size; i++ ){
    data_vec.at(i)=data_ptr[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

  return data_vec;

}

std::vector<double> RadiationModel::getOptiXbufferData_d( RTbuffer buffer ){

  void* _data_;
  RT_CHECK_ERROR( rtBufferMap( buffer, &_data_ ) );
  double* data_ptr = (double*)_data_;

  unsigned long int size;
  RT_CHECK_ERROR( rtBufferGetSize1D( buffer, &size ) );

  std::vector<double> data_vec;
  data_vec.resize(size);
  for( int i=0; i<size; i++ ){
    data_vec.at(i)=data_ptr[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

  return data_vec;

}

void RadiationModel::addBuffer( const char* name, RTbuffer& buffer, RTvariable& variable, RTbuffertype type, RTformat format, size_t dimension ){

  RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, type, &buffer ) );
  RT_CHECK_ERROR( rtBufferSetFormat( buffer, format ) );
  RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, name, &variable ) );
  RT_CHECK_ERROR( rtVariableSetObject( variable, buffer ) );
  if( dimension==1 ){
    zeroBuffer1D( buffer, 1 );
  }else if( dimension==2 ){
    zeroBuffer2D( buffer, optix::make_int2(1,1) );
  }else{
    std::cerr << "ERROR (addBuffer): invalid buffer dimension of " << dimension << ", must be 1 or 2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void RadiationModel::zeroBuffer1D( RTbuffer &buffer, const size_t bsize ){

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format==RT_FORMAT_USER ){//Note: for now, assume user format means it's a double
  
    std::vector<double> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=0.f;
    }
    
    initializeBuffer1Dd( buffer, array );

  }else if( format==RT_FORMAT_FLOAT ){
      
    std::vector<float> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=0.f;
    }
    
    initializeBuffer1Df( buffer, array );

  }else if( format==RT_FORMAT_FLOAT2 ){
  
    std::vector<optix::float2> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=optix::make_float2(0,0);
    }
    
    initializeBuffer1Dfloat2( buffer, array );

  }else if( format==RT_FORMAT_FLOAT3 ){
  
    std::vector<optix::float3> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=optix::make_float3(0,0,0);
    }
    
    initializeBuffer1Dfloat3( buffer, array );

  }else if( format==RT_FORMAT_INT ){
  
    std::vector<int> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=0;
    }
    
    initializeBuffer1Di( buffer, array );

  }else if( format==RT_FORMAT_INT2 ){
  
    std::vector<optix::int2> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=optix::make_int2(0,0);
    }
    
    initializeBuffer1Dint2( buffer, array );

  }else if( format==RT_FORMAT_INT3 ){
  
    std::vector<optix::int3> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=optix::make_int3(0,0,0);
    }
    
    initializeBuffer1Dint3( buffer, array );

  }else if( format==RT_FORMAT_UNSIGNED_INT ){
  
    std::vector<uint> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=0;
    }
    
    initializeBuffer1Dui( buffer, array );

  }else if( format==RT_FORMAT_BYTE ){
  
    std::vector<bool> array;
    array.resize(bsize);
    for( int i=0; i<bsize; i++ ){
      array.at(i)=false;
    }
    
    initializeBuffer1Dbool( buffer, array );
  }else{
    std::cerr << "ERROR (zeroBuffer1D): Buffer type not supported." << std::endl;
    exit(EXIT_FAILURE);
  }
    
}

void RadiationModel::copyBuffer1D( RTbuffer &buffer, RTbuffer &buffer_copy ){

  /* \todo Add support for all data types (currently only works for float and float3)*/

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  //get buffer size
  RTsize bsize;
  rtBufferGetSize1D( buffer, &bsize );

  rtBufferSetSize1D( buffer_copy, bsize );

  if( format==RT_FORMAT_FLOAT ){

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );
    float* data = (float*)ptr;
    
    void* ptr_copy;
    RT_CHECK_ERROR( rtBufferMap( buffer_copy, &ptr_copy ) );
    float* data_copy = (float*)ptr_copy;

    for( size_t i = 0; i <bsize; i++ ) {
      data_copy[i] = data[i];
    }

  }else if( format==RT_FORMAT_FLOAT3 ){
  
    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );
    optix::float3* data = (optix::float3*)ptr;

    void* ptr_copy;
    RT_CHECK_ERROR( rtBufferMap( buffer_copy, &ptr_copy ) );
    optix::float3* data_copy = (optix::float3*)ptr_copy;

    for( size_t i = 0; i <bsize; i++ ) {
      data_copy[i] = data[i];
    }

  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );
  RT_CHECK_ERROR( rtBufferUnmap( buffer_copy ) );
  
}

void RadiationModel::initializeBuffer1Dd( RTbuffer &buffer, std::vector<double> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_USER ){
    std::cerr << "ERROR (initializeBuffer1Dd): Buffer must have type double." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  double* data = (double*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i] = array[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Df( RTbuffer &buffer, std::vector<float> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_FLOAT ){
    std::cerr << "ERROR (initializeBuffer1Df): Buffer must have type float." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  float* data = (float*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i] = array[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dfloat2( RTbuffer &buffer, std::vector<optix::float2> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_FLOAT2 ){
    std::cerr << "ERROR (initializeBuffer1Dfloat2): Buffer must have type float2." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  optix::float2* data = (optix::float2*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
      data[i].x = array[i].x;
      data[i].y = array[i].y;
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dfloat3( RTbuffer &buffer, std::vector<optix::float3> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_FLOAT3 ){
    std::cerr << "ERROR (initializeBuffer1Dfloat3): Buffer must have type float3." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  optix::float3* data = (optix::float3*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i].x = array[i].x;
    data[i].y = array[i].y;
    data[i].z = array[i].z;
  }
  
  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dfloat4( RTbuffer &buffer, std::vector<optix::float4> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_FLOAT4 ){
    std::cerr << "ERROR (initializeBuffer1Dfloat4): Buffer must have type float4." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  optix::float4* data = (optix::float4*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i].x = array[i].x;
    data[i].y = array[i].y;
    data[i].z = array[i].z;
    data[i].w = array[i].w;
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Di( RTbuffer &buffer, std::vector<int> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_INT ){
    std::cerr << "ERROR (initializeBuffer1Di): Buffer must have type int." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  int* data = (int*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i] = array[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dui( RTbuffer &buffer, std::vector<uint> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_UNSIGNED_INT ){
    std::cerr << "ERROR (initializeBuffer1Dui): Buffer must have type unsigned int." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  uint* data = (uint*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i] = array[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dint2( RTbuffer &buffer, std::vector<optix::int2> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_INT2 ){
    std::cerr << "ERROR (initializeBuffer1Dint2): Buffer must have type int2." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  optix::int2* data = (optix::int2*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i] = array[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dint3( RTbuffer &buffer, std::vector<optix::int3> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_INT3 ){
    std::cerr << "ERROR (initializeBuffer1Dint3): Buffer must have type int3." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  optix::int3* data = (optix::int3*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i] = array[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dbool( RTbuffer &buffer, std::vector<bool> array ){

  size_t bsize = array.size();

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format!=RT_FORMAT_BYTE ){
    std::cerr << "ERROR (initializeBuffer1Dbool): Buffer must have type bool." << std::endl;
    exit(EXIT_FAILURE);
  }

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  bool* data = (bool*)ptr;

  for( size_t i = 0; i <bsize; i++ ) {
    data[i] = array[i];
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::zeroBuffer2D( RTbuffer &buffer, const optix::int2 bsize ){

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );
  
  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  if( format==RT_FORMAT_USER ){//Note: for now we'll assume this means it's a double
    std::vector<std::vector<double> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i)=0.f;
      }
    }
    initializeBuffer2Dd( buffer, array );
  }else if( format==RT_FORMAT_FLOAT ){
    std::vector<std::vector<float> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i)=0.f;
      }
    }
    initializeBuffer2Df( buffer, array );
  }else if( format==RT_FORMAT_FLOAT2 ){
    std::vector<std::vector<optix::float2> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i).x=0.f;
	array.at(j).at(i).y=0.f;
      }
    }
    initializeBuffer2Dfloat2( buffer, array );
  }else if( format==RT_FORMAT_FLOAT3 ){
    std::vector<std::vector<optix::float3> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i).x=0.f;
	array.at(j).at(i).y=0.f;
	array.at(j).at(i).z=0.f;
      }
    }
    initializeBuffer2Dfloat3( buffer, array );
  }else if( format==RT_FORMAT_FLOAT4 ){
    std::vector<std::vector<optix::float4> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i).x=0.f;
	array.at(j).at(i).y=0.f;
	array.at(j).at(i).z=0.f;
	array.at(j).at(i).w=0.f;
      }
    }
    initializeBuffer2Dfloat4( buffer, array );
  }else if( format==RT_FORMAT_INT ){
    std::vector<std::vector<int> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i)=0;
      }
    }
    initializeBuffer2Di( buffer, array );
  }else if( format==RT_FORMAT_UNSIGNED_INT ){
    std::vector<std::vector<uint> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i)=0;
      }
    }
    initializeBuffer2Dui( buffer, array );
  }else if( format==RT_FORMAT_INT2 ){
    std::vector<std::vector<optix::int2> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i).x=0;
	array.at(j).at(i).y=0;
      }
    }
    initializeBuffer2Dint2( buffer, array );
  }else if( format==RT_FORMAT_INT3 ){
    std::vector<std::vector<optix::int3> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i).x=0;
	array.at(j).at(i).y=0;
	array.at(j).at(i).z=0;
      }
    }
    initializeBuffer2Dint3( buffer, array );
  }else if( format==RT_FORMAT_BYTE ){
    std::vector<std::vector<bool> > array;
    array.resize(bsize.y);
    for( int j=0; j<bsize.y; j++ ){
      array.at(j).resize(bsize.x);
      for( int i=0; i<bsize.x; i++ ){
	array.at(j).at(i)=false;
      }
    }
    initializeBuffer2Dbool( buffer, array );
  }else{
    std::cerr << "ERROR (zeroBuffer2D): unknown buffer format." << std::endl;
    exit(EXIT_FAILURE);
  }	

}

void RadiationModel::initializeBuffer2Dd( RTbuffer &buffer, std::vector<std::vector<double> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_USER ){
    double* data = (double*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x] = array[j][i];
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dd): Buffer does not have format 'RT_FORMAT_USER'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Df( RTbuffer &buffer, std::vector<std::vector<float> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_FLOAT ){
    float* data = (float*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x] = array[j][i];
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Df): Buffer does not have format 'RT_FORMAT_FLOAT'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dfloat2( RTbuffer &buffer, std::vector<std::vector<optix::float2> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_FLOAT2 ){
    optix::float2* data = (optix::float2*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x].x = array[j][i].x;
	data[i+j*bsize.x].y = array[j][i].y;
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dfloat2): Buffer does not have format 'RT_FORMAT_FLOAT2'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dfloat3( RTbuffer &buffer, std::vector<std::vector<optix::float3> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_FLOAT3 ){
    optix::float3* data = (optix::float3*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x].x = array.at(j).at(i).x;
	data[i+j*bsize.x].y = array.at(j).at(i).y;
	data[i+j*bsize.x].z = array.at(j).at(i).z;
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dfloat3): Buffer does not have format 'RT_FORMAT_FLOAT3'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dfloat4( RTbuffer &buffer, std::vector<std::vector<optix::float4> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_FLOAT4 ){
    optix::float4* data = (optix::float4*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x].x = array[j][i].x;
	data[i+j*bsize.x].y = array[j][i].y;
	data[i+j*bsize.x].z = array[j][i].z;
	data[i+j*bsize.x].w = array[j][i].w;
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dfloat4): Buffer does not have format 'RT_FORMAT_FLOAT4'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Di( RTbuffer &buffer, std::vector<std::vector<int> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_INT ){
    int* data = (int*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x] = array[j][i];
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Di): Buffer does not have format 'RT_FORMAT_INT'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dui( RTbuffer &buffer, std::vector<std::vector<uint> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_UNSIGNED_INT ){
    uint* data = (uint*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x] = array[j][i];
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dui): Buffer does not have format 'RT_FORMAT_UNSIGNED_INT'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dint2( RTbuffer &buffer, std::vector<std::vector<optix::int2> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_INT2 ){
    optix::int2* data = (optix::int2*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x].x = array[j][i].x;
	data[i+j*bsize.x].y = array[j][i].y;
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dint2): Buffer does not have format 'RT_FORMAT_INT2'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dint3( RTbuffer &buffer, std::vector<std::vector<optix::int3> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_INT3 ){
    optix::int3* data = (optix::int3*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x].x = array[j][i].x;
	data[i+j*bsize.x].y = array[j][i].y;
	data[i+j*bsize.x].z = array[j][i].z;
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dint3): Buffer does not have format 'RT_FORMAT_INT3'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dbool( RTbuffer &buffer, std::vector<std::vector<bool> > array ){

  optix::int2 bsize;
  bsize.y = array.size();
  if( bsize.y==0 ){
    bsize.x = 0;
  }else{
    bsize.x = array.front().size();
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_BYTE ){
    bool* data = (bool*)ptr;
    for( size_t j = 0; j<bsize.y; j++ ) {
      for( size_t i = 0; i<bsize.x; i++ ) {
	data[i+j*bsize.x] = array[j][i];
      }
    }
  }else{
    std::cerr << "ERROR (initializeBuffer2Dbool): Buffer does not have format 'RT_FORMAT_BYTE'." << std::endl;
    exit(EXIT_FAILURE);
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

template <typename anytype>
void RadiationModel::initializeBuffer3D( RTbuffer &buffer, std::vector<std::vector<std::vector<anytype> > > array ){

  optix::int3 bsize;
  bsize.z = array.size();
  if( bsize.z==0 ){
    bsize.y=0;
    bsize.x=0;
  }else{
    bsize.y=array.front().size();
    if( bsize.y==0 ){
      bsize.x=0;
    }else{
      bsize.x = array.front().front().size();
    }
  }

  //set buffer size
  RT_CHECK_ERROR( rtBufferSetSize3D( buffer, bsize.x, bsize.y, bsize.z ) );

  //get buffer format
  RTformat format;
  RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

  //zero out buffer
  void* ptr;
  RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

  if( format==RT_FORMAT_FLOAT ){
    float* data = (float*)ptr;
    for( size_t k = 0; k<bsize.z; k++ ) {
      for( size_t j = 0; j<bsize.y; j++ ) {
	for( size_t i = 0; i<bsize.x; i++ ) {
	  data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
	}
      }
    }
  }else if( format==RT_FORMAT_INT ){
    int* data = (int*)ptr;
    for( size_t k = 0; k<bsize.z; k++ ) {
      for( size_t j = 0; j<bsize.y; j++ ) {
	for( size_t i = 0; i<bsize.x; i++ ) {
	  data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
	}
      }
    }
  }else if( format==RT_FORMAT_UNSIGNED_INT ){
    uint* data = (uint*)ptr;
    for( size_t k = 0; k<bsize.z; k++ ) {
      for( size_t j = 0; j<bsize.y; j++ ) {
	for( size_t i = 0; i<bsize.x; i++ ) {
	  data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
	}
      }
    }
  }else if( format==RT_FORMAT_BYTE ){
    bool* data = (bool*)ptr;
    for( size_t k = 0; k<bsize.z; k++ ) {
      for( size_t j = 0; j<bsize.y; j++ ) {
	for( size_t i = 0; i<bsize.x; i++ ) {
	  data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
	}
      }
    }
  }else{
    std::cerr<< "ERROR (initializeBuffer3D): unsupported buffer format." << std::endl;
    
  }

  RT_CHECK_ERROR( rtBufferUnmap( buffer ) );
  

}

float RadiationModel::calculateGtheta( helios::Context* context, const helios::vec3 view_direction ){

  vec3 dir = view_direction;
  dir.normalize();

  float Gtheta = 0;
  float total_area = 0;
  for( std::size_t u=0; u<context_UUIDs.size(); u++ ){
    std::size_t p = context_UUIDs.at(u);
    
    helios::Primitive* prim = context->getPrimitivePointer(p);

    vec3 normal = prim->getNormal();
    float area = prim->getArea();

    Gtheta += fabs(normal*dir)*area;

    total_area += area;
    
  }

  return Gtheta/total_area;
  
}

void RadiationModel::finalize(void){
  RT_CHECK_ERROR( rtContextDestroy( OptiX_Context ) );
}

void sutilHandleError(RTcontext context, RTresult code, const char* file, int line)
{
  const char* message;
  char s[2048];
  rtContextGetErrorString(context, code, &message);
  sprintf(s, "%s\n(%s:%d)", message, file, line);
  sutilReportError( s );
  exit(1);
}

void sutilReportError(const char* message)
{
  fprintf( stderr, "OptiX Error: %s\n", message );
#if defined(_WIN32) && defined(RELEASE_PUBLIC)
  {
    char s[2048];
    sprintf( s, "OptiX Error: %s", message );
    MessageBox( 0, s, "OptiX Error", MB_OK|MB_ICONWARNING|MB_SYSTEMMODAL );
  }
#endif
}
