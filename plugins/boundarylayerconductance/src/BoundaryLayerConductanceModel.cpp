/** \file "BoundaryLayerConductanceModel.cpp" Boundary-layer conductance  model plugin declarations.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "BoundaryLayerConductanceModel.h"

using namespace helios;

BLConductanceModel::BLConductanceModel( helios::Context* __context ){

  wind_speed_default = 1.f;

  air_temperature_default = 293.f;

  surface_temperature_default = 303.f;

  message_flag = true; //print messages to screen by default

  //Copy pointer to the context
  context = __context; 

}

int BLConductanceModel::selfTest(){

  std::cout << "Running boundary-layer conductance model self-test..." << std::flush;

  float err_tol = 1e-3;
  
  Context context_1;

  float TL = 300;
  float Ta = 290;
  float U = 0.6;

  std::vector<uint> UUID_1;

  UUID_1.push_back( context_1.addPatch( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0.2*M_PI,0) ) );
  UUID_1.push_back( context_1.addPatch( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0.2*M_PI,0) ) );
  UUID_1.push_back( context_1.addPatch( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0.2*M_PI,0) ) );
  UUID_1.push_back( context_1.addPatch( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0.2*M_PI,0) ) );

  context_1.setPrimitiveData( UUID_1, "air_temperature", Ta );
  context_1.setPrimitiveData( UUID_1, "temperature", TL );
  context_1.setPrimitiveData( UUID_1, "wind_speed", U );

  BLConductanceModel blc_1(&context_1);
  blc_1.disableMessages();

  blc_1.setBoundaryLayerModel( UUID_1.at(0), "Polhausen" );
  blc_1.setBoundaryLayerModel( UUID_1.at(1), "InclinedPlate" );
  blc_1.setBoundaryLayerModel( UUID_1.at(2), "Sphere" );
  blc_1.setBoundaryLayerModel( UUID_1.at(3), "Ground" );

  blc_1.run();

  std::vector<float> gH;
  gH.resize(UUID_1.size());

  context_1.getPrimitiveData( UUID_1.at(0), "boundarylayer_conductance", gH.at(0) );
  context_1.getPrimitiveData( UUID_1.at(1), "boundarylayer_conductance", gH.at(1) );
  context_1.getPrimitiveData( UUID_1.at(2), "boundarylayer_conductance", gH.at(2) );
  context_1.getPrimitiveData( UUID_1.at(3), "boundarylayer_conductance", gH.at(3) );

  if( fabs(gH.at(0)-0.20914112f)/gH.at(0)>err_tol || fabs(gH.at(1)-0.133763f)/gH.at(1)>err_tol || fabs(gH.at(2)-0.087149f)/gH.at(2)>err_tol || fabs(gH.at(3)-0.465472f)/gH.at(3)>err_tol ){
    std::cout << "failed boundary-layer conductance model check #1." << std::endl;
    return 1;
  }
  
  std::cout << "passed." << std::endl;
  return 0;

}

void BLConductanceModel::enableMessages(void){
  message_flag = true;
}

void BLConductanceModel::disableMessages(void){
  message_flag = false;
}

void BLConductanceModel::setBoundaryLayerModel( const char* gH_model ){
  std::vector<uint> UUIDs = context->getAllUUIDs();
  setBoundaryLayerModel(UUIDs,gH_model);
}

void BLConductanceModel::setBoundaryLayerModel( const uint UUID, const char* gH_model ){
  std::vector<uint> UUIDs{UUID};
  setBoundaryLayerModel(UUIDs,gH_model);
}

void BLConductanceModel::setBoundaryLayerModel(const std::vector<uint> &UUIDs, const char* gH_model ){

  uint model = 0;
  
  if( strcmp(gH_model,"Polhausen")==0 ){
    model = 0;
  }else if( strcmp(gH_model,"InclinedPlate")==0 ){
    model = 1;
  }else if( strcmp(gH_model,"Sphere")==0 ){
    model = 2;
  }else if( strcmp(gH_model,"Ground")==0 ){
    model = 3;
  }else{
    std::cout << "WARNING (EnergyBalanceModel::setBoundaryLayerModel): Boundary-layer conductance model """ << gH_model << """ is unknown. Skipping this function call.." << std::endl;
    return;
  }

  for( uint UUID : UUIDs ){
    boundarylayer_model[UUID] = model;
  }

}

void BLConductanceModel::run(){

  run( context->getAllUUIDs() );
  
}

void BLConductanceModel::run(const std::vector<uint> &UUIDs ){

  for( uint UUID : UUIDs ){

    float U;
    if( context->doesPrimitiveDataExist( UUID, "wind_speed" ) ){
      context->getPrimitiveData( UUID, "wind_speed", U );
    }else{
      U = wind_speed_default;
    }

    float Ta;
    if( context->doesPrimitiveDataExist( UUID, "air_temperature" ) ){
      context->getPrimitiveData( UUID, "air_temperature", Ta );
    }else{
      Ta = air_temperature_default;
    }

    float T;
    if( context->doesPrimitiveDataExist( UUID, "temperature" ) ){
      context->getPrimitiveData( UUID, "temperature", T );
    }else{
      T = surface_temperature_default;
    }

    float L;
    if( context->doesPrimitiveDataExist( UUID, "object_length") ) {
      context->getPrimitiveData(UUID, "object_length", L);
      if (L == 0) {
        L = sqrt(context->getPrimitiveArea(UUID));
      }
    }else if( context->getPrimitiveParentObjectID(UUID)>0 ){
      uint objID = context->getPrimitiveParentObjectID(UUID);
      L = sqrt(context->getObjectArea(objID));
    }else{
      L = sqrt(context->getPrimitiveArea(UUID));
    }

    //Number of primitive faces
    char Nsides = 2; //default is 2
    if( context->doesPrimitiveDataExist(UUID,"twosided_flag") && context->getPrimitiveDataType(UUID,"twosided_flag")==HELIOS_TYPE_UINT ){
      uint flag;
      context->getPrimitiveData(UUID,"twosided_flag",flag);
      if( flag==0 ){
        Nsides=1;
      }
    }

    vec3 norm = context->getPrimitiveNormal(UUID);
    float inclination = cart2sphere( norm ).zenith;

    float gH = calculateBoundaryLayerConductance( boundarylayer_model[UUID], U, L, Nsides, inclination, T, Ta );

    context->setPrimitiveData( UUID, "boundarylayer_conductance", gH );

  }

}

float BLConductanceModel::calculateBoundaryLayerConductance( uint gH_model, float U, float L, char Nsides, float inclination, float TL, float Ta ){

  
  assert( gH_model<4 );

  float gH;

  if( L==0 ){
    return 0;
  }

  if( gH_model==0 ){ //Polhausen equation
    //This comes from the correlation by Polhausen - see Eq. XX of Campbell and Norman (1998). It assumes a flat plate parallel to the direction of the flow, which extends infinitely in the cross-stream direction and "L" in the streamwise direction. It also assumes that the air is at standard temperature and pressure, and flow is laminar, forced convection.

    gH = 0.135f*sqrt(U/L)*float(Nsides);

  }else if( gH_model==1 ){ //Inclined Plate

    float Pr=0.7f;     //air Prandtl number

    float nu=1.568e-5; //air viscosity (m^2/sec)
    float alpha=nu/Pr; //air diffusivity (m^2/sec)

    float Re=U*L/nu;

    float Gr=9.81*fabs(TL-Ta)*pow(L,3)/((Ta)*nu*nu);
    
    float F1=0.399*pow(Pr,1.f/3.f)*pow(1.f+pow(0.0468/Pr,2.f/3.f),-0.25);
    float F2=0.75*pow(Pr,0.5)*pow(2.5*(1.f+2.f*pow(Pr,0.5)+2.f*Pr),-0.25);

    //direction of free convection
    float free_direction;
    if(TL>=Ta){
      free_direction=1;
    }else{
      free_direction=-1;
    }

    //free_direction=1;

    if( inclination<75.f*M_PI/180.f || inclination>105.f*M_PI/180.f){
        
      gH=41.4*alpha/L*2.f*F1*sqrtf(Re)*pow(1.f+free_direction*pow(2.f*F2*pow(Gr*fabs(cos(inclination))/(Re*Re),0.25)/(3.f*F1),3),1.f/3.f);
        
    }else{
        
      float C=0.07*sqrtf(fabs(cos(inclination)));
      float F3=pow(Pr,0.5f)*pow(0.25+1.6*pow(Pr,0.5),-1)*pow(Pr/5.f,1.f/5.f+C);
        
      gH=41.4*alpha/L*2.f*F1*sqrtf(Re)*pow(1.f+free_direction*pow((F3*pow(Gr*pow(Re,-5.f/2.f),1.f/5.f)*pow(Gr,C))/(6.f*(1.f/5.f+C)*F1),3),1.f/3.f);

    }

  }else if( gH_model==2 ){ //Sphere

      //Laminar flow around a sphere (L is sphere diameter). From Eq. 4 of Smart and Sinclair (1976).

      float Pr=0.7f;     //air Prandtl number
      float nu=1.568e-5; //air viscosity (m^2/sec)
      float ka = 0.024;  //air thermal conductivity (W/m-K)
      float cp = 29.25; //air heat capacity (J/mol-K)

      float Re=U*L/nu;

      gH = (ka/cp)/L*(2.0 + 0.6*sqrtf(Re)*pow(Pr,1.f/3.f));  

  }else if( gH_model==3 ){ //Ground
      //From Eq. A17 of Kustas and Norman (1999). For the soil-air interface.

      gH = 0.004 + 0.012*U; //units in (m^3 air)/(m^2-sec.)

      //assuming standard temperature and pressure, (m^3 air)*(1.2041 kg/m^3)/(0.02897 kg/mol) = 41.56 (mol air)/(m^2-sec.)

      gH = gH*41.56;

  }
    
  return gH;

}
