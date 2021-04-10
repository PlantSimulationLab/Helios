/** \file "EnergyBalanceModel.cu" Energy balance model plugin declarations (CUDA kernels). 
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

#include <cuda_runtime.h>
#include "EnergyBalanceModel.h"

using namespace helios;
using namespace std;

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ float evaluateEnergyBalance( float T, float R, float Qother, float eps, float Ta, float ea, float pressure, float gH, float gS, uint Nsides, float heatcapacity, float dt, float Tprev ){

  //Outgoing emission flux
  float Rout = float(Nsides)*eps*5.67e-8*pow(T,4);
    
  //Sensible heat flux
  float cp = 29.25; //Molar specific heat of air. Units: J/mol
  float QH = cp*gH*(T-Ta); // (see Campbell and Norman Eq. 6.8)

  //Latent heat flux
  float es = 611.f*exp(17.502f*(T-273.f)/((T-273.f)+240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in Kelvin, and result is in Pascals
  float gM = 0.97*gH*gS/(0.97*gH+gS); //resistors in series
  if( gH==0 && gS==0 ){//if somehow both go to zero, can get NaN
    gM = 0;
  }
  float lambda = 44000; //Latent heat of vaporization for water. Units: J/mol
  float QL = gM*lambda*(es-ea)/pressure;

  //Storage heat flux

  float storage = 0.f;
  if (dt>0){
    storage=heatcapacity*(T-Tprev)/dt;
  }

  //Residual
  return R-Rout-QH-QL-Qother-storage;

}

__global__ void solveEnergyBalance( uint Nprimitives, float* To, float* R, float* Qother, float* eps, float* Ta, float* ea, float* pressure, float* gH, float* gS, uint* Nsides, float* TL, float* heatcapacity, float dt ){

  uint p = blockIdx.x*blockDim.x+threadIdx.x;

  if( p>=Nprimitives ){
    return;
  }

  float T;

  float err_max = 0.0001;
  uint max_iter = 100;

  float T_old_old = To[p];

  float T_old = T_old_old;
  T_old_old = 400.f;

  float resid_old = evaluateEnergyBalance(T_old,R[p],Qother[p],eps[p],Ta[p],ea[p],pressure[p],gH[p],gS[p],Nsides[p],heatcapacity[p],dt,To[p]);
  float resid_old_old = evaluateEnergyBalance(T_old_old,R[p],Qother[p],eps[p],Ta[p],ea[p],pressure[p],gH[p],gS[p],Nsides[p],heatcapacity[p],dt,To[p]);
  
  float resid = 100;
  float err = resid;
  uint iter = 0;
  while( err>err_max && iter<max_iter ){

    if( resid_old==resid_old_old ){//this condition will cause NaN
      err=0;
      break;
    }

    T = fabs((T_old_old*resid_old-T_old*resid_old_old)/(resid_old-resid_old_old));

    resid = evaluateEnergyBalance(T,R[p],Qother[p],eps[p],Ta[p],ea[p],pressure[p],gH[p],gS[p],Nsides[p],heatcapacity[p],dt,To[p]);

    resid_old_old = resid_old;
    resid_old = resid;

    //err = fabs(resid);
    //err = fabs(resid_old-resid_old_old)/fabs(resid_old_old);
    err = fabs(T_old-T_old_old)/fabs(T_old_old);
    
    T_old_old = T_old;
    T_old = T;

    iter++;

  }

  if( err>err_max ){
    printf("WARNING (solveEnergyBalance): Energy balance did not converge.\n");
  }

  TL[p] = T;

}

void EnergyBalanceModel::run( void ){
  run( context->getAllUUIDs() );
}

void EnergyBalanceModel::run( const float dt ){
  run( context->getAllUUIDs(), dt );
}

void EnergyBalanceModel::run( std::vector<uint> UUIDs ){
  run( UUIDs, 0.f);
}


void EnergyBalanceModel::run( std::vector<uint> UUIDs, const float dt ){
 
  if( message_flag ){
    std::cout << "Running energy balance model..." << std::flush;
  }
    
  // Check that some primitives exist in the context

  uint Nprimitives = UUIDs.size();

  if( Nprimitives==0 ){
    std::cerr << "ERROR (EnergyBalanceModel): No primitives have been added to the context.  There is nothing to simulate. Exiting..." << std::endl;
    return;
  }

  //---- Sum up to get total absorbed radiation across all bands ----//

  // Look through all flux primitive data in the context and sum them up in vector Rn.  Each element of Rn corresponds to a primitive.

  if( radiation_bands.size()==0 ){
    std::cerr << "ERROR (EnergyBalanceModel): No radiation bands were found." << std::endl;
    exit(EXIT_FAILURE);
  }
    
  std::vector<float> Rn;
  Rn.resize(Nprimitives);

  std::vector<float> emissivity;
  emissivity.resize(Nprimitives);
  for( size_t u=0; u<Nprimitives; u++ ){
    emissivity.at(u) = 1.f;
  }

  for( int b=0; b<radiation_bands.size(); b++ ){
    for( size_t u=0; u<Nprimitives; u++ ){
      size_t p = UUIDs.at(u);
      
      char str[50];
      sprintf(str,"radiation_flux_%s",radiation_bands.at(b).c_str());
      if( !context->doesPrimitiveDataExist(p,str) ){
	std::cerr << "ERROR (EnergyBalanceModel): No radiation was found in the context for band " << radiation_bands.at(b) << ". Did you run the radiation model for this band?" << std::endl;
	exit(EXIT_FAILURE);
      }
      float R;
      context->getPrimitiveData(p,str,R);
      Rn.at(u) += R;
      
      sprintf(str,"emissivity_%s",radiation_bands.at(b).c_str());
      if( context->doesPrimitiveDataExist(p,str) ){
	context->getPrimitiveData(p,str,emissivity.at(u));
      }
      
    }
  }

  //---- Set up temperature solution ----//

  //To,R,Qother,eps,U,L,Ta,ea,pressure,gS,Nsides
  
  float* To = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_To;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_To, Nprimitives*sizeof(float)) );

  float* R = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_R;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_R, Nprimitives*sizeof(float)) );
  
  float* Qother = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_Qother;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_Qother, Nprimitives*sizeof(float)) );
  
  float* eps = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_eps;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_eps, Nprimitives*sizeof(float)) );
  
  float* Ta = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_Ta;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_Ta, Nprimitives*sizeof(float)) );
  
  float* ea = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_ea;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_ea, Nprimitives*sizeof(float)) );
  
  float* pressure = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_pressure;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_pressure, Nprimitives*sizeof(float)) );

  float* gH = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_gH;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_gH, Nprimitives*sizeof(float)) );
  
  float* gS = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_gS;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_gS, Nprimitives*sizeof(float)) );
  
  uint* Nsides = (uint*)malloc( Nprimitives*sizeof(uint) );
  uint* d_Nsides;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_Nsides, Nprimitives*sizeof(uint)) );

  float* heatcapacity = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_heatcapacity;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_heatcapacity, Nprimitives*sizeof(float)) );
  
  for( uint u=0; u<Nprimitives; u++ ){
    size_t p = UUIDs.at(u);

    helios::Primitive* prim = context->getPrimitivePointer(p);   

    //Initial guess for surface temperature
    if( prim->doesPrimitiveDataExist("temperature") ){
      prim->getPrimitiveData("temperature",To[u]);
    }else{
      To[u] = temperature_default;
    }
    if( To[u]==0 ){//can't have To equal to 0
      To[u] = 300;
    }

    //Net absorbed radiation
    R[u] = Rn.at(u);

    //Emissivity
    eps[u] = emissivity.at(u);

    //Air temperature
    if( prim->doesPrimitiveDataExist("air_temperature") ){
      prim->getPrimitiveData("air_temperature",Ta[u]);
    }else{
      Ta[u] = air_temperature_default;
    }

    //Air relative humidity
    float hr;
    if( prim->doesPrimitiveDataExist("air_humidity") ){
      prim->getPrimitiveData("air_humidity",hr);
    }else{
      hr = air_humidity_default;
    }

    //Air vapor pressure
    float esat = 611.f*exp(17.502f*(Ta[u]-273.f)/((Ta[u]-273.f)+240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in degC, and result is in Pascals
    ea[u] = hr*esat; // Definition of vapor pressure (see Campbell and Norman pp. 42 Eq. 3.11)

    //Air pressure
    if( prim->doesPrimitiveDataExist("air_pressure") ){
      prim->getPrimitiveData("air_pressure",pressure[u]);
    }else{
      pressure[u] = pressure_default;
    }

    //Boundary-layer conductance to heat
    if( prim->doesPrimitiveDataExist("boundarylayer_conductance") ){
      prim->getPrimitiveData("boundarylayer_conductance",gH[u]);
    }else{

      //Wind speed
      float U;
      if( prim->doesPrimitiveDataExist("wind_speed") ){
	prim->getPrimitiveData("wind_speed",U);
      }else{
	U = wind_speed_default;
      }

      //Characteristic size of primitive
      float L;
      if( prim->doesPrimitiveDataExist("object_length") ){
	prim->getPrimitiveData("object_length",L);
	if( L==0 ){
	  L = sqrt(prim->getArea());
	}
      }else{
	L = sqrt(prim->getArea());
      }
      
      gH[u]=0.135f*sqrt(U/L);
    }
 
    //Moisture conductance
    if( prim->doesPrimitiveDataExist("moisture_conductance") ){
      prim->getPrimitiveData("moisture_conductance",gS[u]);
    }else{
      gS[u] = gS_default;
    }
      
    //Other fluxes
    if( prim->doesPrimitiveDataExist("other_surface_flux") ){
      prim->getPrimitiveData("other_surface_flux",Qother[u]);
    }else{
      Qother[u] = Qother_default;
    }
    
    //Number of sides emitting radiation
    Nsides[u] = 2; //default is 2
    if( prim->doesPrimitiveDataExist("twosided_flag") ){
      uint flag;
      prim->getPrimitiveData("twosided_flag",flag);
      if( flag==0 ){
    	Nsides[u]=1;
      }
    }

    //Object heat capacity
    if( prim->doesPrimitiveDataExist("heat_capacity") ){
      prim->getPrimitiveData("heat_capacity",heatcapacity[u]);
    }else{
      heatcapacity[u] = heatcapacity_default;
    }

  }

  //To,R,Qother,eps,U,L,Ta,ea,pressure,gS,Nsides
  CUDA_CHECK_ERROR( cudaMemcpy(d_To, To, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_R, R, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_Qother, Qother, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_eps, eps, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_Ta, Ta, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_ea, ea, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_pressure, pressure, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_gH, gH, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_gS, gS, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_Nsides, Nsides, Nprimitives*sizeof(uint), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_heatcapacity, heatcapacity, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );

  float* T = (float*)malloc( Nprimitives*sizeof(float) );
  float* d_T;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_T, Nprimitives*sizeof(float)) );

  //launch kernel
  dim3 dimBlock( 64, 1 );
  dim3 dimGrid( ceil(Nprimitives/64.f) );
  
  solveEnergyBalance <<< dimGrid, dimBlock >>>(Nprimitives,d_To,d_R,d_Qother,d_eps,d_Ta,d_ea,d_pressure,d_gH,d_gS,d_Nsides,d_T,d_heatcapacity,dt);
      
  CUDA_CHECK_ERROR( cudaPeekAtLastError() );
  CUDA_CHECK_ERROR( cudaDeviceSynchronize() );

  CUDA_CHECK_ERROR( cudaMemcpy(T, d_T, Nprimitives*sizeof(float), cudaMemcpyDeviceToHost) );

  for( uint u=0; u<Nprimitives; u++ ){
    size_t p = UUIDs.at(u);

    helios::Primitive* prim = context->getPrimitivePointer(p);  
  
    if( T[u]!=T[u] ){
      T[u] = temperature_default;
    }
    
    prim->setPrimitiveData("temperature",T[u]);

    float QH = 29.25*gH[u]*(T[u]-Ta[u]);
    prim->setPrimitiveData("sensible_flux",QH);

    float es = 611.f*exp(17.502f*(T[u]-273.f)/((T[u]-273.f)+240.97f));
    float gM = 0.97*gH[u]*gS[u]/(0.97*gH[u]+gS[u]);
    float QL = 44000*gM*(es-ea[u])/pressure[u];
    prim->setPrimitiveData("latent_flux",QL);

    float storage=0.f;
    if ( dt>0){
      storage=heatcapacity[u]*(T[u]-To[u])/dt;
    }
    prim->setPrimitiveData("storage_flux", storage);

    for( int i=0; i<output_prim_data.size(); i++ ){
      if( output_prim_data.at(i).compare("boundarylayer_conductance_out")==0 ){
	prim->setPrimitiveData("boundarylayer_conductance_out",gH[u]);
      }else if( output_prim_data.at(i).compare("vapor_pressure_deficit")==0 ){
	float vpd = (es-ea[u])/pressure[u];
	prim->setPrimitiveData("vapor_pressure_deficit",vpd);
      }
    }

  }

  free( To );
  free( R );
  free( Qother );
  free( eps );
  free( Ta );
  free( ea );
  free( pressure );
  free( gH );
  free( gS );
  free( Nsides );
  free( T );
  free( heatcapacity );

  CUDA_CHECK_ERROR( cudaFree(d_To) );
  CUDA_CHECK_ERROR( cudaFree(d_R) );
  CUDA_CHECK_ERROR( cudaFree(d_Qother) );
  CUDA_CHECK_ERROR( cudaFree(d_eps) );
  CUDA_CHECK_ERROR( cudaFree(d_Ta) );
  CUDA_CHECK_ERROR( cudaFree(d_ea) );
  CUDA_CHECK_ERROR( cudaFree(d_pressure) );
  CUDA_CHECK_ERROR( cudaFree(d_gH) );
  CUDA_CHECK_ERROR( cudaFree(d_gS) );
  CUDA_CHECK_ERROR( cudaFree(d_Nsides) );
  CUDA_CHECK_ERROR( cudaFree(d_T) );
  CUDA_CHECK_ERROR( cudaFree(d_heatcapacity) );

  if( message_flag ){
    std::cout << "done." << std::endl;
  }
    
}


