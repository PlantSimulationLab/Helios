/** \file "EnergyBalanceModel.cu" Energy balance model plugin declarations (CUDA kernels).

    Copyright (C) 2016-2024 Brian Bailey

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

__device__ float evaluateEnergyBalance( float T, float R, float Qother, float eps, float Ta, float ea, float pressure, float gH, float gS, uint Nsides, char Ntranspire, float heatcapacity, float surfacehumidity, float dt, float Tprev ){

    //Outgoing emission flux
    float Rout = float(Nsides)*eps*5.67e-8F*T*T*T*T;

    //Sensible heat flux
    float cp = 29.25f; //Molar specific heat of air. Units: J/mol
    float QH = cp*gH*(T-Ta); // (see Campbell and Norman Eq. 6.8)

    //Latent heat flux
    float es = 611.f*exp(17.502f*(T-273.f)/((T-273.f)+240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in Kelvin, and result is in Pascals
    float gM = float(Ntranspire)*1.08f*(gH/float(Nsides))*gS/(1.08f*(gH/float(Nsides))+gS);
    if( gH==0 && gS==0 ){//if somehow both go to zero, can get NaN
        gM = 0;
    }
    float lambda = 44000.f; //Latent heat of vaporization for water. Units: J/mol
    float QL = gM*lambda*(es-ea*surfacehumidity)/pressure;

    //Storage heat flux
    float storage = 0.f;
    if (dt>0){
        storage=heatcapacity*(T-Tprev)/dt;
    }

    //Residual
    return R-Rout-QH-QL-Qother-storage;

}

__global__ void solveEnergyBalance( uint Nprimitives, float* To, float* R, float* Qother, float* eps, float* Ta, float* ea, float* pressure, float* gH, float* gS, uint* Nsides, char* Ntranspire, float* TL, float* heatcapacity, float* surfacehumidity, float dt ){

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

    float resid_old = evaluateEnergyBalance(T_old,R[p],Qother[p],eps[p],Ta[p],ea[p],pressure[p],gH[p],gS[p],Nsides[p],Ntranspire[p],heatcapacity[p],surfacehumidity[p],dt,To[p]);
    float resid_old_old = evaluateEnergyBalance(T_old_old,R[p],Qother[p],eps[p],Ta[p],ea[p],pressure[p],gH[p],gS[p],Nsides[p],Ntranspire[p],heatcapacity[p],surfacehumidity[p],dt,To[p]);

    float resid = 100;
    float err = resid;
    uint iter = 0;
    while( err>err_max && iter<max_iter ){

        if( resid_old==resid_old_old ){//this condition will cause NaN
            err=0;
            break;
        }

        T = fabs((T_old_old*resid_old-T_old*resid_old_old)/(resid_old-resid_old_old));

        resid = evaluateEnergyBalance(T,R[p],Qother[p],eps[p],Ta[p],ea[p],pressure[p],gH[p],gS[p],Nsides[p],Ntranspire[p],heatcapacity[p],surfacehumidity[p],dt,To[p]);

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
        printf("WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge.\n");
    }

    TL[p] = T;

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

    uint Nprimitives = UUIDs.size();

    if( Nprimitives==0 ){
        std::cerr << "WARNING (EnergyBalanceModel::run): No primitives have been added to the context.  There is nothing to simulate. Exiting..." << std::endl;
        return;
    }

    //---- Sum up to get total absorbed radiation across all bands ----//

    // Look through all flux primitive data in the context and sum them up in vector Rn.  Each element of Rn corresponds to a primitive.

    if( radiation_bands.size()==0 ){
        helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation bands were found.");
    }

    std::vector<float> Rn;
    Rn.resize(Nprimitives,0);

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
            if( !context->doesPrimitiveDataExist(p,str) ) {
                helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation was found in the context for band " + std::string(radiation_bands.at(b)) + ". Did you run the radiation model for this band?");
            }else if( context->getPrimitiveDataType(p,str)!=HELIOS_TYPE_FLOAT ){
                helios_runtime_error("ERROR (EnergyBalanceModel::run): Radiation primitive data for band " + std::string(radiation_bands.at(b)) + " does not have the correct type of ''float'");
            }
            float R;
            context->getPrimitiveData(p,str,R);
            Rn.at(u) += R;

            sprintf(str,"emissivity_%s",radiation_bands.at(b).c_str());
            if( context->doesPrimitiveDataExist(p,str) && context->getPrimitiveDataType(p,str)==HELIOS_TYPE_FLOAT ){
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

    char* Ntranspire = (char*)malloc( Nprimitives*sizeof(char) );
    char* d_Ntranspire;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_Ntranspire, Nprimitives*sizeof(char)) );

    float* heatcapacity = (float*)malloc( Nprimitives*sizeof(float) );
    float* d_heatcapacity;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_heatcapacity, Nprimitives*sizeof(float)) );

    float* surfacehumidity = (float*)malloc( Nprimitives*sizeof(float) );
    float* d_surfacehumidity;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_surfacehumidity, Nprimitives*sizeof(float)) );

    bool calculated_blconductance_used = false;
    bool primitive_length_used = false;

    for( uint u=0; u<Nprimitives; u++ ){
        size_t p = UUIDs.at(u);

        //Initial guess for surface temperature
        if( context->doesPrimitiveDataExist(p,"temperature") && context->getPrimitiveDataType(p,"temperature")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"temperature",To[u]);
        }else{
            To[u] = temperature_default;
        }
        if( To[u]==0 ){//can't have To equal to 0
            To[u] = 300;
        }

        //Air temperature
        if( context->doesPrimitiveDataExist(p,"air_temperature") && context->getPrimitiveDataType(p,"air_temperature")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"air_temperature",Ta[u]);
            if( message_flag && Ta[u]<250.f ){
              std::cout << "WARNING (EnergyBalanceModel::run): Value of " << Ta[u] << " given in 'air_temperature' primitive data is very small. Values should be given in units of Kelvin. Assuming default value of " << air_temperature_default << std::endl;
              Ta[u] = air_temperature_default;
            }
        }else{
            Ta[u] = air_temperature_default;
        }

        //Air relative humidity
        float hr;
        if( context->doesPrimitiveDataExist(p,"air_humidity") && context->getPrimitiveDataType(p,"air_humidity")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"air_humidity",hr);
            if( hr>1.f ){
                if( message_flag ){
                    std::cout << "WARNING (EnergyBalanceModel::run): Value of " << hr << " given in 'air_humidity' primitive data is large than 1. Values should be given as fractional values between 0 and 1. Assuming default value of " << air_humidity_default << std::endl;
                }
                hr = air_humidity_default;
            }else if( hr<0.f ){
                if( message_flag ) {
                    std::cout << "WARNING (EnergyBalanceModel::run): Value of " << hr << " given in 'air_humidity' primitive data is less than 0. Values should be given as fractional values between 0 and 1. Assuming default value of " << air_humidity_default << std::endl;
                }
                hr = air_humidity_default;
            }
        }else{
            hr = air_humidity_default;
        }

        //Air vapor pressure
        float esat = 611.f*exp(17.502f*(Ta[u]-273.f)/((Ta[u]-273.f)+240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in degC, and result is in Pascals
        ea[u] = hr*esat; // Definition of vapor pressure (see Campbell and Norman pp. 42 Eq. 3.11)

        //Air pressure
        if( context->doesPrimitiveDataExist(p,"air_pressure") && context->getPrimitiveDataType(p,"air_pressure")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"air_pressure",pressure[u]);
            if( pressure[u]<10000.f ){
                if( message_flag ) {
                    std::cout << "WARNING (EnergyBalanceModel::run): Value of " << pressure[u] << " given in 'air_pressure' primitive data is very small. Values should be given in units of Pascals. Assuming default value of " << pressure_default << std::endl;
                }
              pressure[u] = pressure_default;
            }
        }else{
            pressure[u] = pressure_default;
        }

        //Number of sides emitting radiation
        Nsides[u] = 2; //default is 2
        if( context->doesPrimitiveDataExist(p,"twosided_flag") && context->getPrimitiveDataType(p,"twosided_flag")==HELIOS_TYPE_UINT ){
          uint flag;
          context->getPrimitiveData(p,"twosided_flag",flag);
          if( flag==0 ){
            Nsides[u]=1;
          }
        }

        //Number of evaporating/transpiring faces
        if( Nsides[u]==2 && context->doesPrimitiveDataExist(p,"evaporating_faces") && context->getPrimitiveDataType(p,"evaporating_faces")==HELIOS_TYPE_UINT ){
          uint flag;
          context->getPrimitiveData(p,"evaporating_faces",flag);
          if( flag==1 || flag==2 ){
            Ntranspire[u]=char(flag);
          }else {
            Ntranspire[u] = 1;
          }
        }else{
          Ntranspire[u] = 1;
        }

        //Boundary-layer conductance to heat
        if( context->doesPrimitiveDataExist(p,"boundarylayer_conductance") && context->getPrimitiveDataType(p,"boundarylayer_conductance")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"boundarylayer_conductance",gH[u]);
        }else{

            //Wind speed
            float U;
            if( context->doesPrimitiveDataExist(p,"wind_speed") && context->getPrimitiveDataType(p,"wind_speed")==HELIOS_TYPE_FLOAT ){
                context->getPrimitiveData(p,"wind_speed",U);
            }else{
                U = wind_speed_default;
            }

            //Characteristic size of primitive
            float L;
            if( context->doesPrimitiveDataExist(p,"object_length") && context->getPrimitiveDataType(p,"object_length")==HELIOS_TYPE_FLOAT ){
                context->getPrimitiveData(p,"object_length",L);
                if( L==0 ){
                    L = sqrt(context->getPrimitiveArea(p));
                    primitive_length_used = true;
                }
            }else if( context->getPrimitiveParentObjectID(p)>0 ){
              uint objID = context->getPrimitiveParentObjectID(p);
              L = sqrt(context->getObjectArea(objID));
            }else{
                L = sqrt(context->getPrimitiveArea(p));
                primitive_length_used = true;
            }

            gH[u]=0.135f*sqrt(U/L)*float(Nsides[u]);

            calculated_blconductance_used = true;
        }

        //Moisture conductance
        if( context->doesPrimitiveDataExist(p,"moisture_conductance") && context->getPrimitiveDataType(p,"moisture_conductance")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"moisture_conductance",gS[u]);
        }else{
            gS[u] = gS_default;
        }

        //Other fluxes
        if( context->doesPrimitiveDataExist(p,"other_surface_flux") && context->getPrimitiveDataType(p,"other_surface_flux")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"other_surface_flux",Qother[u]);
        }else{
            Qother[u] = Qother_default;
        }

        //Object heat capacity
        if( context->doesPrimitiveDataExist(p,"heat_capacity") && context->getPrimitiveDataType(p,"heat_capacity")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"heat_capacity",heatcapacity[u]);
        }else{
            heatcapacity[u] = heatcapacity_default;
        }

        //Surface humidity
        if( context->doesPrimitiveDataExist(p,"surface_humidity") && context->getPrimitiveDataType(p,"surface_humidity")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"surface_humidity",surfacehumidity[u]);
        }else{
            surfacehumidity[u] = surface_humidity_default;
        }

        //Emissivity
        eps[u] = emissivity.at(u);

        //Net absorbed radiation
        R[u] = Rn.at(u);

    }

    //if we used the calculated boundary-layer conductance, enable output primitive data "boundarylayer_conductance_out" so that it can be used by other plug-ins
    if( calculated_blconductance_used ){
        auto it = find( output_prim_data.begin(), output_prim_data.end(), "boundarylayer_conductance_out" );
        if( it == output_prim_data.end() ){
            output_prim_data.emplace_back( "boundarylayer_conductance_out" );
        }
    }

    //if the length of a primitive that is not a member of an object was used, issue a warning
    if( message_flag && primitive_length_used ){
        std::cout << "WARNING (EnergyBalanceModel::run): The length of a primitive that is not a member of a compound object was used to calculate the boundary-layer conductance. This often results in incorrect values because the length should be that of the object (e.g., leaf, stem) not the primitive. Make sure this is what you intended." << std::endl;
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
    CUDA_CHECK_ERROR( cudaMemcpy(d_Ntranspire, Ntranspire, Nprimitives*sizeof(char), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_heatcapacity, heatcapacity, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_surfacehumidity, surfacehumidity, Nprimitives*sizeof(float), cudaMemcpyHostToDevice) );

    float* T = (float*)malloc( Nprimitives*sizeof(float) );
    float* d_T;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_T, Nprimitives*sizeof(float)) );

    //launch kernel
    dim3 dimBlock( 64, 1 );
    dim3 dimGrid( ceil(Nprimitives/64.f) );

    solveEnergyBalance <<< dimGrid, dimBlock >>>(Nprimitives,d_To,d_R,d_Qother,d_eps,d_Ta,d_ea,d_pressure,d_gH,d_gS,d_Nsides,d_Ntranspire,d_T,d_heatcapacity,d_surfacehumidity,dt);

    CUDA_CHECK_ERROR( cudaPeekAtLastError() );
    CUDA_CHECK_ERROR( cudaDeviceSynchronize() );

    CUDA_CHECK_ERROR( cudaMemcpy(T, d_T, Nprimitives*sizeof(float), cudaMemcpyDeviceToHost) );

    for( uint u=0; u<Nprimitives; u++ ){
        size_t p = UUIDs.at(u);

        if( T[u]!=T[u] ){
            T[u] = temperature_default;
        }

        context->setPrimitiveData(p,"temperature",T[u]);

        float QH = 29.25*gH[u]*(T[u]-Ta[u]);
        context->setPrimitiveData(p,"sensible_flux",QH);

        float es = 611.f*exp(17.502f*(T[u]-273.f)/((T[u]-273.f)+240.97f));
        float gM = float(Ntranspire[u])*1.08f*(gH[u]/float(Nsides[u]))*gS[u]/(1.08f*(gH[u]/float(Nsides[u]))+gS[u]);
        float QL = 44000*gM*(es-ea[u])/pressure[u];
        context->setPrimitiveData(p,"latent_flux",QL);

        float storage=0.f;
        if ( dt>0){
            storage=heatcapacity[u]*(T[u]-To[u])/dt;
        }
        context->setPrimitiveData(p,"storage_flux", storage);

        for( int i=0; i<output_prim_data.size(); i++ ){
            if( output_prim_data.at(i) == "boundarylayer_conductance_out" ){
                context->setPrimitiveData(p,"boundarylayer_conductance_out",gH[u]);
            }else if( output_prim_data.at(i) == "vapor_pressure_deficit" ){
                float vpd = (es-ea[u])/pressure[u];
                context->setPrimitiveData(p,"vapor_pressure_deficit",vpd);
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
    free( Ntranspire );
    free( heatcapacity );
    free( surfacehumidity );
    free( T );


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
    CUDA_CHECK_ERROR( cudaFree(d_Ntranspire) );
    CUDA_CHECK_ERROR( cudaFree(d_heatcapacity) );
    CUDA_CHECK_ERROR( cudaFree(d_surfacehumidity) );
    CUDA_CHECK_ERROR( cudaFree(d_T) );

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}


