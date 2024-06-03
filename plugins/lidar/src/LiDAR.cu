/** \file "LiDAR.cu" CUDA-related LiDAR plug-in functions

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "LiDAR.cuh"
#include "LiDAR.h"

using namespace LIDAR_CUDA;

__host__ __device__ float2 LIDAR_CUDA::operator+(const float2 &a, const float2 &b) {

    return make_float2(a.x+b.x, a.y+b.y);

}

__host__ __device__ float3 LIDAR_CUDA::operator+(const float3 &a, const float3 &b) {

    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__host__ __device__ float2 LIDAR_CUDA::operator-(const float2 &a, const float2 &b) {

    return make_float2(a.x-b.x, a.y-b.y);

}

__host__ __device__ float3 LIDAR_CUDA::operator-(const float3 &a, const float3 &b) {

    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}

__host__ __device__ float LIDAR_CUDA::operator*(const float2 &a, const float2 &b) {

    return a.x*b.x + a.y*b.y;

}

__host__ __device__ float LIDAR_CUDA::operator*(const float3 &a, const float3 &b) {

    return a.x*b.x + a.y*b.y + a.z*b.z;

}

__host__ __device__ float3 LIDAR_CUDA::operator*(const float3 &a, const float& b) {
    float3 v = a;
    v.x *= b;
    v.y *= b;
    v.z *= b;
    return v;
}

__host__ __device__ float3 LIDAR_CUDA::operator/(const float3 &a, const float& b) {
    float3 v = a;
    v.x /= b;
    v.y /= b;
    v.z /= b;
    return v;
}

__host__ __device__ float3 LIDAR_CUDA::cross(const float3 &a, const float3 &b){
    float3 c;
    c.x=a.y*b.z - b.y*a.z;
    c.y=b.x*a.z - a.x*b.z;
    c.z=a.x*b.y - b.x*a.y;
    return c;
}

__host__ __device__ float LIDAR_CUDA::magnitude(const float3 &a){
    return sqrt( a.x*a.x + a.y*a.y + a.z*a.z );
}

__host__ __device__ float3 LIDAR_CUDA::normalize(const float3 &a){
    return a/magnitude(a);
}

float2 LIDAR_CUDA::vec2tofloat2( helios::vec2 v2 ){
    float2 f2;
    f2.x=v2.x;
    f2.y=v2.y;
    return f2;
}

helios::vec2 LIDAR_CUDA::float2tovec2( float2 f2 ){
    helios::vec2 v2;
    v2.x=f2.x;
    v2.y=f2.y;
    return v2;
}

float3 LIDAR_CUDA::vec3tofloat3( helios::vec3 v3 ){
    float3 f3;
    f3.x=v3.x;
    f3.y=v3.y;
    f3.z=v3.z;
    return f3;
}

helios::vec3 LIDAR_CUDA::float3tovec3( float3 f3 ){
    helios::vec3 v3;
    v3.x=f3.x;
    v3.y=f3.y;
    v3.z=f3.z;
    return v3;
}

__device__ float3 LIDAR_CUDA::d_rotatePoint(const float3 &position, const float &theta, const float &phi) {

    float Ry[3][3], Rz[3][3];

    float st = sinf(theta);
    float ct = cosf(theta);

    float sp = sinf(phi);
    float cp = cosf(phi);

    // Setup the rotation matrix, this matrix is based off of the rotation matrix used in glRotatef.
    Ry[0][0] = ct;
    Ry[0][1] = 0.f;
    Ry[0][2] = st;
    Ry[1][0] = 0.f;
    Ry[1][1] = 1.f;
    Ry[1][2] = 0.f;
    Ry[2][0] = -st;
    Ry[2][1] = 0.f;
    Ry[2][2] = ct;

    Rz[0][0] = cp;
    Rz[0][1] = -sp;
    Rz[0][2] = 0.f;
    Rz[1][0] = sp;
    Rz[1][1] = cp;
    Rz[1][2] = 0.f;
    Rz[2][0] = 0.f;
    Rz[2][1] = 0.f;
    Rz[2][2] = 1.f;

    // Multiply Ry*Rz

    float rotMat[3][3] = {0.f};

    for( int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                rotMat[i][j]=rotMat[i][j]+Rz[i][k]*Ry[k][j];
            }
        }
    }

    // Multiply the rotation matrix with the position vector.
    float3 tmp;
    tmp.x = rotMat[0][0] * position.x + rotMat[0][1] * position.y + rotMat[0][2] * position.z;
    tmp.y = rotMat[1][0] * position.x + rotMat[1][1] * position.y + rotMat[1][2] * position.z;
    tmp.z = rotMat[2][0] * position.x + rotMat[2][1] * position.y + rotMat[2][2] * position.z;

    return tmp;

}

__device__ float3 LIDAR_CUDA::d_sphere2cart( float radius, float elevation, float azimuth ){

    float3 Cartesian;

    Cartesian.x = radius*cos(elevation)*sin(azimuth);
    Cartesian.y = radius*cos(elevation)*cos(azimuth);
    Cartesian.z = radius*sin(elevation);

    return Cartesian;

}

__host__ __device__ float LIDAR_CUDA::atan2_2pi( const float& y, const float& x){

    float v=0;

    if(x>0.f){
        v=atan(y/x);
    }
    if(y>=0.f && x<0.f){
        v=M_PI+atan(y/x);
    }
    if(y<0.f && x<0.f){
        v=-M_PI+atan(y/x);
    }
    if(y>0.f && x==0.f){
        v=0.5f*M_PI;
    }
    if(y<0.f && x==0.f){
        v=-0.5f*M_PI;
    }
    if(v<0.f){
        v=v+2.f*M_PI;
    }
    return v;

}

__host__ __device__ float LIDAR_CUDA::acos_safe( float x ){
    if (x < -1.0) x = -1.0 ;
    else if (x > 1.0) x = 1.0 ;
    return acosf(x) ;
}

__host__ int LIDAR_CUDA::randu( int imin, int imax  ){

    float ru = float(rand()) / float(RAND_MAX + 1.);

    if( imin==imax || imin>imax ){
        return imin;
    }else{
        return imin + round(float(imax-imin)*ru);
    }

}

__device__ float2 LIDAR_CUDA::d_sampleDisk( float radius, uint seed ){

    // use Box-Muller tranform approach
    // the input radius is the place where the intensity decreases to 1/e^2 (~0.1353) of the intensity at the beam center

    // sample from uniform random distribution [0,1] for radial distance and angle
    float U1 = rnd(seed);
    float U2 = rnd(seed);

    // calculate the standard deviation of a normal distribution where  intensity decreases to 1/e^2 at the input radius
    float sd = sqrt(0.25*radius*radius);

    // formula from https://www.alanzucconi.com/2015/09/16/how-to-sample-from-a-gaussian-distribution/
    float R = sqrt(-2.f*logf(U1))*sd;
    float angle = 2.f*M_PI*U2;

    float2 sp;
    //find x,y point on unit disk
    sp.x = R*cosf(angle);
    sp.y = R*sinf(angle);
    return sp;

}

bool LIDAR_CUDA::sortcol0( const std::vector<float>& v0, const std::vector<float>& v1 ){
    return v0.at(0)<v1.at(0);
}

// bool LIDAR_CUDA::sortcol1( const std::vector<float>& v0, const std::vector<float>& v1 ){
//   return v0.at(1)<v1.at(1);
// }

// bool LIDAR_CUDA::sortcol0( const std::vector<double>& v0, const std::vector<double>& v1 ){
//   return v0.at(0)<v1.at(0);
// }

bool LIDAR_CUDA::sortcol1( const std::vector<double>& v0, const std::vector<double>& v1 ){
    return v0.at(1)<v1.at(1);
}

void LiDARcloud::calculateHitGridCellGPU() {

    if( printmessages ){
        std::cout << "Grouping hit points by grid cell..." << std::flush;
    }

    // ---- Hit Points ---- //

    float3* d_hit_xyz;

    uint Nscans = getScanCount();

    std::size_t total_hits = getHitCount();

    const uint N = total_hits;

    float3* hit_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_xyz,N*sizeof(float3)) ); //allocate device memory

    //copy scan data into the host buffer
    for( std::size_t r=0; r< getHitCount(); r++ ){
        hit_xyz[r] = vec3tofloat3(getHitXYZ(r));
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_xyz, hit_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );

    // ---- Grid Cells ---- //

    //! Grid cell centers on device (GPU) memory
    float3* d_grid_center;

    //! Grid cell global anchor on device (GPU) memory
    float3* d_grid_anchor;

    //! Grid sizes on device (GPU) memory
    float3* d_grid_size;

    //! Grid rotations on device (GPU) memory
    float* d_grid_rotation;

    const uint Ncells = getGridCellCount();

    float3* center = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_center,Ncells*sizeof(float3)) ); //allocate device memory

    float3* anchor = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_anchor,Ncells*sizeof(float3)) ); //allocate device memory

    float3* size = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_size,Ncells*sizeof(float3)) ); //allocate device memory

    float* rotation = (float*)malloc(Ncells * sizeof(float)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_rotation,Ncells*sizeof(float)) ); //allocate device memory

    //copy grid data into the host buffer
    for( int c=0; c<Ncells; c++ ){
        center[c] = vec3tofloat3(getCellCenter(c));
        anchor[c] = vec3tofloat3(getCellGlobalAnchor(c));
        size[c] = vec3tofloat3(getCellSize(c));
        rotation[c] = getCellRotation(c);
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_center, center, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_anchor, anchor, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_size, size, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_rotation, rotation, Ncells*sizeof(float), cudaMemcpyHostToDevice) );

    free(hit_xyz);
    free(center);
    free(anchor);
    free(size);
    free(rotation);

    // Result buffer
    int* hit_vol = (int*)malloc(total_hits * sizeof(int));
    int* d_hit_vol;
    CUDA_CHECK_ERROR( cudaMalloc(&d_hit_vol,total_hits*sizeof(int)) );

    dim3 dimBlock( 64, 1, 1 );
    dim3 dimGrid( ceil(total_hits/64.f), 1, 1 );
    if( total_hits==0 ){
        std::cout << "WARNING (calculateHitGridcellGPU): There are no hits currently in the point cloud. Skipping grid cell binning calculation." << std::endl;
    }else{
        insideVolume <<< dimGrid, dimBlock >>>( total_hits, d_hit_xyz, getGridCellCount(), d_grid_size, d_grid_center, d_grid_anchor, d_grid_rotation, d_hit_vol );
    }

    CUDA_CHECK_ERROR( cudaPeekAtLastError() );
    CUDA_CHECK_ERROR( cudaDeviceSynchronize() );

    CUDA_CHECK_ERROR( cudaMemcpy(hit_vol, d_hit_vol, total_hits*sizeof(int), cudaMemcpyDeviceToHost) );

    for( std::size_t r=0; r< getHitCount(); r++ ){
        setHitGridCell( r, hit_vol[r] );
    }

    free(hit_vol);

    CUDA_CHECK_ERROR( cudaFree(d_hit_vol) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_xyz) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_center) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_anchor) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_size) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_rotation) );

    hitgridcellcomputed = true;

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

void LiDARcloud::sourcesInsideGridCellGPU() {

    if( printmessages ){
        std::cout << "Checking that scan origins are not inside grid cells..." << std::flush;
    }

    // ---- Hit Points ---- //

    float3* d_hit_xyz;

    uint Nscans = getScanCount();

    std::size_t total_hits = Nscans;

    const uint N = total_hits;

    float3* hit_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_xyz,N*sizeof(float3)) ); //allocate device memory

    //copy scan data into the host buffer
    for( std::size_t s=0; s<Nscans; s++ ){
        hit_xyz[s] = vec3tofloat3(getScanOrigin(s));
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_xyz, hit_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );

    // ---- Grid Cells ---- //

    //! Grid cell centers on device (GPU) memory
    float3* d_grid_center;

    //! Grid cell global anchor on device (GPU) memory
    float3* d_grid_anchor;

    //! Grid sizes on device (GPU) memory
    float3* d_grid_size;

    //! Grid rotations on device (GPU) memory
    float* d_grid_rotation;

    const uint Ncells = getGridCellCount();

    float3* center = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_center,Ncells*sizeof(float3)) ); //allocate device memory

    float3* anchor = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_anchor,Ncells*sizeof(float3)) ); //allocate device memory

    float3* size = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_size,Ncells*sizeof(float3)) ); //allocate device memory

    float* rotation = (float*)malloc(Ncells * sizeof(float)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_rotation,Ncells*sizeof(float)) ); //allocate device memory

    //copy grid data into the host buffer
    for( int c=0; c<Ncells; c++ ){
        center[c] = vec3tofloat3(getCellCenter(c));
        anchor[c] = vec3tofloat3(getCellGlobalAnchor(c));
        size[c] = vec3tofloat3(getCellSize(c));
        rotation[c] = getCellRotation(c);
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_center, center, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_anchor, anchor, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_size, size, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_rotation, rotation, Ncells*sizeof(float), cudaMemcpyHostToDevice) );

    free(hit_xyz);
    free(center);
    free(anchor);
    free(size);
    free(rotation);

    // Result buffer
    int* hit_vol = (int*)malloc(total_hits * sizeof(int));
    int* d_hit_vol;
    CUDA_CHECK_ERROR( cudaMalloc(&d_hit_vol,total_hits*sizeof(int)) );

    dim3 dimBlock( 64, 1 );
    dim3 dimGrid( ceil(total_hits/64.f) );
    insideVolume <<< dimGrid, dimBlock >>>( total_hits, d_hit_xyz, getGridCellCount(), d_grid_size, d_grid_center, d_grid_anchor, d_grid_rotation, d_hit_vol );

    CUDA_CHECK_ERROR( cudaPeekAtLastError() );
    CUDA_CHECK_ERROR( cudaDeviceSynchronize() );

    CUDA_CHECK_ERROR( cudaMemcpy(hit_vol, d_hit_vol, total_hits*sizeof(int), cudaMemcpyDeviceToHost) );

    for( std::size_t s=0; s<Nscans; s++ ){
        if( hit_vol[s]>=0 ){
            std::cout << "Scan #" << s << " is inside gricell " << hit_vol[s] << std::endl;
        }
    }

    free(hit_vol);

    CUDA_CHECK_ERROR( cudaFree(d_hit_vol) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_xyz) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_center) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_anchor) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_size) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_rotation) );

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

std::vector<helios::vec3> LiDARcloud::gapfillMisses(){
    std::vector<helios::vec3> xyz_filled;
    for( uint scanID=0; scanID<getScanCount(); scanID++ ){
        std::vector<helios::vec3> filled_this_scan = gapfillMisses(scanID,false,false);
        xyz_filled.insert(xyz_filled.end(), filled_this_scan.begin(), filled_this_scan.end());
    }
    return xyz_filled;
}

std::vector<helios::vec3> LiDARcloud::gapfillMisses(uint scanID){
    return gapfillMisses(scanID, false, false);
}

std::vector<helios::vec3> LiDARcloud::gapfillMisses(uint scanID, const bool gapfill_grid_only, const bool add_flags){

    if( printmessages ) {
        std::cout << "Gap filling complete misses in scan " << scanID << "..." << std::flush;
    }

    float gap_distance = 20000;


    helios::vec3 origin = getScanOrigin(scanID);
    std::vector<helios::vec3> xyz_filled;

    // Populating a hit table for each scan:
    // Column 0 - hit index; Column 1 - timestamp; Column 2 - ray zenith; Column 3 - ray azimuth
    std::vector<std::vector<double> > hit_table;
    for( size_t r=0; r< getHitCount(); r++ ){
        if( getHitScanID(r)== scanID){

            if(add_flags)
            {
                // gapfillMisses_code = 0: original points
                setHitData(r, "gapfillMisses_code", 0.0);
            }


            helios::SphericalCoord raydir = getHitRaydir(r);

            if( !doesHitDataExist(r,"timestamp") ){
                std::cerr << "ERROR (LiDARcloud::gapfillMisses): timestamp value is missing for hit " << r << ". Cannot gapfill. Skipping..." << std::endl;
                return xyz_filled;
            }

            double timestamp = getHitData(r,"timestamp");
            std::vector<double> data;
            data.resize(4);
            data.at(0) = float(r);
            data.at(1) = timestamp;
            data.at(2) = raydir.zenith;
            data.at(3) = raydir.azimuth;
            hit_table.push_back( data );

        }
    }

    // sorting, initial dt and dtheta calculations, and determining minimum target index in the scan

    //sort the hit table by column 1 (timestamp)
    std::sort( hit_table.begin(), hit_table.end(), LIDAR_CUDA::sortcol1 );

    int min_tindex = 1;
    for( size_t r=0; r<hit_table.size()-1; r++ ){

        //this is to figure out if target indexing uses 0 or 1 offset
        if( min_tindex==1 && doesHitDataExist(hit_table.at(r).at(0),"target_index") && doesHitDataExist(hit_table.at(r).at(0),"target_count") ){
            if( getHitData(hit_table.at(r).at(0),"target_index")==0 ){
                min_tindex=0;
            }
        }

    }

    // getting rid of points with target index greater than the minimum

    int ndup_target = 0;
    // create new array without duplicate timestamps
    std::vector<std::vector<double> > hit_table_semiclean;
    for( size_t r=0; r<hit_table.size()-1; r++ ){

        //only consider first hits
        if( doesHitDataExist(hit_table.at(r).at(0),"target_index") && doesHitDataExist(hit_table.at(r).at(0),"target_count") ){
            if( getHitData(hit_table.at(r).at(0),"target_index")>min_tindex ){
                ndup_target ++;
                continue;
            }
        }

        hit_table_semiclean.push_back(hit_table.at(r));

    }

    //  re-calculating dt

    std::vector<double> dt_semiclean;
    dt_semiclean.resize(hit_table_semiclean.size());
    for( size_t r=0; r<hit_table_semiclean.size()-1; r++ ){

        dt_semiclean.at(r) = hit_table_semiclean.at(r+1).at(1)-hit_table_semiclean.at(r).at(1);
        // set the hit index of the new array
        hit_table_semiclean.at(r).at(0) = r;

    }

    //  checking for duplicate timestamps in the remaining data

    int ndup = 0;
    // create new array without duplicate timestamps
    std::vector<std::vector<double> > hit_table_clean;
    for( size_t r=0; r<hit_table_semiclean.size()-1; r++ ){

        // if there are still rows with duplicate timestamps, it probably means there is no "target_index" column, but multiple hits per timestamp are still included
        // proceed using this assumption, just get rid of the rows where dt = 0 for simplicity (last hits probably are what remain). 
        if( dt_semiclean.at(r)==0){
            ndup ++;
            continue;
        }

        hit_table_clean.push_back(hit_table_semiclean.at(r));
    }

    // recalculate dt and dtheta with only one hit per beam
    // and calculate the minimum dt value  
    std::vector<double> dt_clean;
    std::vector<float> dtheta_clean;
    dt_clean.resize(hit_table_clean.size());
    dtheta_clean.resize(hit_table_clean.size());

    double dt_clean_min = 1e6;
    for( size_t r=0; r<hit_table_clean.size()-1; r++ ){

        dt_clean.at(r) = hit_table_clean.at(r+1).at(1)-hit_table_clean.at(r).at(1);
        dtheta_clean.at(r) = hit_table_clean.at(r+1).at(2)-hit_table_clean.at(r).at(2);
        // set the hit index of the new array
        hit_table_clean.at(r).at(0) = r;

        if( dt_clean.at(r)<dt_clean_min){
            dt_clean_min = dt_clean.at(r);
        }

    }

    // configuration of 2D map
    // reconfigure hit table into 2D (theta,phi) map
    std::vector<std::vector<std::vector<double> > > hit_table2D;

    int column = 0;
    hit_table2D.resize(1);
    for( size_t r=0; r<hit_table_clean.size()-1; r++ ){

        hit_table2D.at(column).push_back( hit_table_clean.at(r) );
        //for small scans (like the rectangle test case, this needs to change to < 0 or some smaller angle (that is larger than noise))
        // if( dtheta_clean.at(r) < 0 ){
        // for normal scans, this threshold allows for 10 degrees drops in theta within a given sweep as noise. This can be adjusted as appropriate. 
        if( dtheta_clean.at(r) < -0.1745329f ){
            column++;
            hit_table2D.resize(column+1);
        }

    }

    // calculate average dt and dtheta for subsequent points

    //calculate average dt
    float dt_avg = 0;
    int dt_sum = 0;

    //calculate the average dtheta to use for extrapolation
    float dtheta_avg = 0;
    int dtheta_sum = 0;

    for( int j=0; j<hit_table2D.size(); j++ ){
        for( int i=0; i<hit_table2D.at(j).size(); i++ ){
            int r = int(hit_table2D.at(j).at(i).at(0));
            if( dt_clean.at(r)>=dt_clean_min && dt_clean.at(r)<1.5*dt_clean_min ){
                dt_avg += dt_clean.at(r);
                dt_sum ++;

                //calculate the average dtheta to use for extrapolation
                dtheta_avg += dtheta_clean.at(r);
                dtheta_sum ++;

            }
        }
    }

    dt_avg = dt_avg/float(dt_sum);
    //Calculate the average dtheta to use for extrapolation
    dtheta_avg = dtheta_avg/float(dtheta_sum);

//    std::cout << "dt_avg = " << dt_avg << std::endl;
//    std::cout << "dtheta_avg = " << dtheta_avg << std::endl;
//    std::cout << "hit_table2D.size() = "<< hit_table2D.size() << std::endl;

    //identify gaps and fill
    for( int j=0; j<hit_table2D.size(); j++ ){

        if(hit_table2D.at(j).size() > 0)
        {
            for( int i=0; i<hit_table2D.at(j).size()-1; i++ ){

                double dt = hit_table2D.at(j).at(i+1).at(1)-hit_table2D.at(j).at(i).at(1);

                if( dt>1.5f*dt_clean_min ){ //missing hit(s)

                    //calculate number of missing hits
                    int Ngap = round(dt/dt_avg)-1;

                    //fill missing points
                    for( int k=1; k<=Ngap; k++ ){

                        float timestep = hit_table2D.at(j).at(i).at(1) + dt_avg*float(k);

                        //interpolate theta and phi
                        float theta = hit_table2D.at(j).at(i).at(2) + (hit_table2D.at(j).at(i+1).at(2)-hit_table2D.at(j).at(i).at(2))*float(k)/float(Ngap+1);
                        float phi = hit_table2D.at(j).at(i).at(3) + (hit_table2D.at(j).at(i+1).at(3)-hit_table2D.at(j).at(i).at(3))*float(k)/float(Ngap+1);
                        if( phi>2.f*M_PI ){
                            phi = phi-2.f*M_PI;
                        }

                        //calculate the (x,y,z) position of the filled hit point
                        helios::SphericalCoord spherical(gap_distance,0.5*M_PI-theta,phi);
                        helios::vec3 xyz = origin+helios::sphere2cart(spherical);

                        xyz_filled.push_back(xyz);

                        //ERK
                        std::map<std::string, double> data;
                        data.insert(std::pair<std::string, double>("timestamp", timestep));
                        data.insert(std::pair<std::string, double>("target_index", min_tindex));
                        data.insert(std::pair<std::string, double>("nRaysHit", 500));
                        if(add_flags)
                        {
                            // gapfillMisses_code = 1: gapfilled points
                            data.insert(std::pair<std::string, double>("gapfillMisses_code", 1.0));
                        }
                        addHitPoint(scanID, xyz, spherical, data);

                    }
                }
            }
        }

    }
    uint npointsfilled = xyz_filled.size();

    helios::vec2 theta_range;
    theta_range = getScanRangeTheta(scanID); // use ranges from xml file to start

    if(gapfill_grid_only == true)
    {
        //instead of extrapolating to the angle ranges given in the xml file, we can extrapolate to the angle range of the voxel grid to save time. 
        // to do this we loop through the vertices of the voxel grid.
        std::vector<helios::vec3> grid_vertices;
        helios::vec3 boxmin, boxmax;
        getGridBoundingBox(boxmin, boxmax); // axis aligned bounding box of all grid cells
        grid_vertices.push_back(boxmin);
        grid_vertices.push_back(boxmax);
        grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmin.y, boxmax.z));
        grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmax.y, boxmin.z));
        grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmax.y, boxmin.z));
        grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmax.y, boxmax.z));
        grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmin.y, boxmin.z));
        grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmin.y, boxmax.z));

        float max_theta = 0;
        float min_theta = M_PI;
        float max_phi = 0;
        float min_phi = 2*M_PI;
        for(uint gg=0;gg<grid_vertices.size();gg++)
        {
            helios::vec3 direction_cart = grid_vertices.at(gg)-getScanOrigin(scanID);
            helios::SphericalCoord sc = cart2sphere(direction_cart);
            if(sc.azimuth < min_phi)
            {
                min_phi = sc.azimuth;
            }

            if(sc.azimuth > max_phi)
            {
                max_phi = sc.azimuth;
            }

            if(sc.zenith < min_theta)
            {
                min_theta = sc.zenith;
            }

            if(sc.zenith > max_theta)
            {
                max_theta = sc.zenith;
            }
        }


        // if the min or max theta is outside of the values provided in xml, use the xml values
        if(min_theta < theta_range.x)
        {
            min_theta = theta_range.x;
        }

        if(max_theta > theta_range.y)
        {
            max_theta = theta_range.y;
        }

        theta_range = helios::make_vec2(min_theta, max_theta);
    }

//    std::cout << "theta_range for extrapolation: " << theta_range.x*(180.0/M_PI) << ", " << theta_range.y*(180.0/M_PI) << std::endl;

    // extrapolate missing points
    for( int j=0; j<hit_table2D.size(); j++ ){

        if(hit_table2D.at(j).size() > 0)
        {

            //upward edge points
            if( hit_table2D.at(j).front().at(2)>theta_range.x ){

                float dtheta = dtheta_avg;
                float theta = hit_table2D.at(j).at(0).at(2) - dtheta;
                //just use the last value of phi in the sweep
                float phi =  hit_table2D.at(j).at(0).at(3);
                float timestep = hit_table2D.at(j).at(0).at(1) - dt_avg;
                if( dtheta==0 ){
                    continue;
                }

                while( theta>theta_range.x ){

                    helios::SphericalCoord spherical(gap_distance,0.5*M_PI-theta,phi);
                    helios::vec3 xyz = origin+helios::sphere2cart(spherical);

                    xyz_filled.push_back(xyz);

                    std::map<std::string, double> data;
                    data.insert(std::pair<std::string, double>("timestamp", timestep));
                    data.insert(std::pair<std::string, double>("target_index", min_tindex));
                    data.insert(std::pair<std::string, double>("nRaysHit", 500));
                    if(add_flags)
                    {
                        // gapfillMisses_code = 3: upward edge points
                        data.insert(std::pair<std::string, double>("gapfillMisses_code", 3.0));

                    }

                    addHitPoint(scanID, xyz, spherical, data);


                    theta = theta - dtheta;
                    timestep = timestep - dt_avg;


                }
            }

            //downward edge points
            if( hit_table2D.at(j).back().at(2)<theta_range.y ){

                int sz = hit_table2D.at(j).size();
                // same concept as above for downward edge points
                float dtheta = dtheta_avg;
                float theta = hit_table2D.at(j).at(sz-1).at(2) + dtheta;
                float phi = hit_table2D.at(j).at(sz-1).at(3);
                float timestep = hit_table2D.at(j).at(sz-1).at(1) + dt_avg;
                while( theta<theta_range.y ){

                    helios::SphericalCoord spherical(gap_distance,0.5*M_PI-theta,phi);
                    helios::vec3 xyz = origin+helios::sphere2cart(spherical);

                    xyz_filled.push_back(xyz);

                    std::map<std::string, double> data;
                    data.insert(std::pair<std::string, double>("timestamp", timestep));
                    data.insert(std::pair<std::string, double>("target_index", min_tindex));
                    data.insert(std::pair<std::string, double>("nRaysHit", 500));
                    if(add_flags)
                    {
                        // gapfillMisses_code = 2: downward edge points
                        data.insert(std::pair<std::string, double>("gapfillMisses_code", 2.0));
                    }

                    addHitPoint(scanID, xyz, spherical, data);

                    theta = theta + dtheta;
                    timestep = timestep + dt_avg;

                }
            }
        }

    }

    uint npointsextrapolated = xyz_filled.size() - npointsfilled;

    if( printmessages ) {
        std::cout << "filled " << xyz_filled.size() << " points." << std::endl;
    }
    return xyz_filled;

}


void LiDARcloud::calculateLeafAreaGPU() {
    calculateLeafAreaGPU( 1 );
}

void LiDARcloud::calculateLeafAreaGPU( int min_voxel_hits){

    if( printmessages ){
        std::cout << "Calculating leaf area..." << std::endl;
    }

    if( !triangulationcomputed ){
        helios::helios_runtime_error("ERROR (LiDARcloud::calculateLeafAreaGPU): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints().");
    }

    if( !hitgridcellcomputed ){
        calculateHitGridCellGPU();
    }

    const uint Nscans = getScanCount();
    const uint Ncells = getGridCellCount();

    //variable aggregates over all scans where we just keep tacking hits on the end for all scans
    std::vector<std::vector<float> > dr_agg; //dr is path length through grid cell
    dr_agg.resize(Ncells);
    std::vector<float> G_agg; //G is dot product between ray direction and triangle normal that was hit (only relevant for hits)
    G_agg.resize(Ncells,0);
    std::vector<float> hit_before_agg; //hit_before corresponds to scan points that hit something before encountering a particular grid cell
    hit_before_agg.resize(Ncells,0);
    std::vector<float> hit_after_agg; //hit_after corresponds to scan points that hit something after encountering a particular grid cell (including something inside that cell)
    hit_after_agg.resize(Ncells,0);
    std::vector<float> hit_inside_agg; //hit_inside corresponds to scan points that hit something within a particular grid cell.
    hit_inside_agg.resize(Ncells,0);

    //average G(theta)
    std::vector<float> Gtheta_bar;
    Gtheta_bar.resize(Ncells,0.f);

    // We are going to perform all calculations on a scan-by-scan basis: loop through each scan
    for( uint s=0; s<Nscans; s++ ){

        //----------- Calculate dr and number of rays passed through voxels -------------- //

        const int Nt = getScanSizeTheta(s);
        const int Np = getScanSizePhi(s);
        const size_t Nmisses = Nt*Np;

        const float3 origin = vec3tofloat3(getScanOrigin(s));

        // ----- BOUNDING BOX INTERSECTION ----- //
        // First, we are going to perform a preliminary ray trace to filter out rays that do not intersect any grid volumes.  This will speed up calculations overall. Consider all rays in scan to start.

        float3* scan_xyz = (float3*)malloc( Nmisses*sizeof(float3) );
        float* scan_weight = (float*)malloc( Nmisses*sizeof(float) );
        float3 xyz;

        //populate misses
        for( int j=0; j<Np; j++ ){
            for( int i=0; i<Nt; i++ ){
                float3 direction = vec3tofloat3(sphere2cart(scans.at(s).rc2direction(i,j)));
                xyz = origin + direction*10000.f;

                scan_xyz[j*Nt+i] = xyz;

                scan_weight[j*Nt+i] = 1.f;
            }
        }

        float3* d_scan_xyz;
        CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nmisses*sizeof(float3)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nmisses*sizeof(float3), cudaMemcpyHostToDevice) );

        float* d_scan_weight;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_scan_weight, Nmisses*sizeof(float)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_weight, scan_weight, Nmisses*sizeof(float), cudaMemcpyHostToDevice) );

        //set up an axis-aligned bounding box that encompasses all grid cells
        uint* boundingbox_hit = (uint*)malloc( Nmisses * sizeof(uint));
        uint* d_boundingbox_hit;
        CUDA_CHECK_ERROR( cudaMalloc((uint**)&d_boundingbox_hit, Nmisses*sizeof(uint)) );
        CUDA_CHECK_ERROR( cudaMemset( d_boundingbox_hit, 0, Nmisses*sizeof(uint)) ); //initialize to zero, set equal to 1 if the ray is found to intersect bounding box

        helios::vec3 gboxmin, gboxmax;
        getGridBoundingBox( gboxmin, gboxmax );

        float3 bbcenter = vec3tofloat3( gboxmin + 0.5*(gboxmax-gboxmin) );
        float3 bbsize = vec3tofloat3( gboxmax-gboxmin );

        //perform the ray-trace and mark all rays that passed through the bounding box
        uint3 dimBlock = make_uint3( 512, 1, 1 );
        uint3 dimGrid = make_uint3( ceil(float(Nmisses)/float(dimBlock.x)), 1, 1 );
        intersectBoundingBox <<< dimGrid, dimBlock >>>( Nmisses, origin, d_scan_xyz, bbcenter, bbsize, d_boundingbox_hit );

        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

        //copy results back to host
        CUDA_CHECK_ERROR( cudaMemcpy( boundingbox_hit, d_boundingbox_hit, Nmisses*sizeof(uint), cudaMemcpyDeviceToHost));

        //counting up how many points passed through bounding box
        size_t Nmissesbb = 0;
        for( size_t i=0; i<Nmisses; i++ ){
            if( boundingbox_hit[i]==1 ){
                Nmissesbb ++;
            }
        }

        if( Nmissesbb==0 ){
            std::cerr << "ERROR (calculateLeafAreaGPU): No scan rays passed through grid cells.  Check that scan configuration is properly specified." << std::endl;
            std::cerr << "Terminating leaf area calculation." << std::endl;
            for( uint c=0; c<Ncells; c++ ){
                setCellLeafArea(0,c);
            }
            return;
        }

        CUDA_CHECK_ERROR( cudaFree(d_boundingbox_hit) );
        CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );

        // Reconfigure scan_xyz with points that do not hit the bounding box eliminated
        float3* missesbb_xyz = (float3*)malloc( Nmissesbb*sizeof(float3) );
        size_t miss_count=0;
        for( size_t i=0; i<Nmisses; i++ ){
            if( boundingbox_hit[i]==1 ){
                missesbb_xyz[miss_count] = scan_xyz[i];
                miss_count++;
            }
        }
        CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nmissesbb*sizeof(float3)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, missesbb_xyz, Nmissesbb*sizeof(float3), cudaMemcpyHostToDevice) );
        free( missesbb_xyz );
        free( boundingbox_hit );

        //Result buffer (path lengths through volume)
        float* dr = (float*)malloc( Nmissesbb * sizeof(float));
        float* d_dr;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr, Nmissesbb*sizeof(float)) );

        float* hit_before = (float*)malloc( sizeof(float));
        float* hit_after = (float*)malloc( sizeof(float));
        float* d_hit_before;
        float* d_hit_after;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_before, sizeof(float)) );
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_after, sizeof(float)) );

        // Perform ray-volume intersection tests to determine volumes that rays passed through
        for( uint c=0; c<Ncells; c++ ){

            //load the attributes of the grid cell
            float3 center = vec3tofloat3(getCellCenter(c));
            float3 anchor = vec3tofloat3(getCellGlobalAnchor(c));
            float3 size = vec3tofloat3(getCellSize(c));
            float rotation = getCellRotation(c);

            CUDA_CHECK_ERROR( cudaMemset( d_dr, 0.f, Nmissesbb*sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_before, 0.f, sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_after, 0.f, sizeof(float)) );

            dimBlock = make_uint3( 512, 1, 1 );
            dimGrid = make_uint3( ceil(float(Nmissesbb)/dimBlock.x), 1, 1  );
            intersectGridcell <<< dimGrid, dimBlock >>>( Nmissesbb, origin, d_scan_xyz, d_scan_weight, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after );

            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

            //copy results back to host
            CUDA_CHECK_ERROR( cudaMemcpy( dr, d_dr, Nmissesbb*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_after, d_hit_after, sizeof(float), cudaMemcpyDeviceToHost));

            for( size_t r=0; r<Nmissesbb; r++ ){
                if( dr[r]>0.f ){
                    dr_agg.at(c).push_back(dr[r]);
                }
            }
            hit_after_agg.at(c) += *hit_after;

        }

        // Perform ray-volume intersection to determine rays that intersected a leaf BEFORE passing through the volume

        std::vector<helios::vec3> this_scan_xyz;
        std::vector<float> this_scan_weight;

        for( size_t r=0; r< getHitCount(); r++ ){
            if( getHitScanID(r)==s ){
                this_scan_xyz.push_back( getHitXYZ(r) );

                if( doesHitDataExist(r,"target_count") ){
                    this_scan_weight.push_back( 1.f/float(getHitData(r,"target_count")) );
                }else{
                    this_scan_weight.push_back(1.f);
                }

            }
        }

        size_t Nhits = this_scan_xyz.size();

        free( scan_xyz );
        scan_xyz = (float3*)malloc( Nhits*sizeof(float3) );
        free( scan_weight );
        scan_weight = (float*)malloc( Nhits*sizeof(float) );

        for( size_t r=0; r<Nhits; r++ ){
            scan_xyz[r] = vec3tofloat3(this_scan_xyz.at(r));
            scan_weight[r] = this_scan_weight.at(r);
        }

        CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
        CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nhits*sizeof(float3)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

        CUDA_CHECK_ERROR( cudaFree(d_scan_weight) );
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_scan_weight, Nhits*sizeof(float)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_weight, scan_weight, Nhits*sizeof(float), cudaMemcpyHostToDevice) );

        CUDA_CHECK_ERROR( cudaFree(d_dr) );
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr, Nhits*sizeof(float)) );

        for( uint c=0; c<Ncells; c++ ){

            //load the attributes of the grid cell
            float3 center = vec3tofloat3(getCellCenter(c));
            float3 anchor = vec3tofloat3(getCellGlobalAnchor(c));
            float3 size = vec3tofloat3(getCellSize(c));
            float rotation = getCellRotation(c);

            CUDA_CHECK_ERROR( cudaMemset( d_dr, 0, Nhits*sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_before, 0, sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_after, 0, sizeof(float)) );

            dimBlock = make_uint3( min(size_t(512),Nhits), 1, 1 );
            dimGrid = make_uint3( ceil(float(Nhits)/float(dimBlock.x) ), 1, 1  );

            if( dimBlock.x==0 && dimGrid.x==0 ){
                continue;
            }

            intersectGridcell <<< dimGrid, dimBlock >>>( Nhits, origin, d_scan_xyz, d_scan_weight, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after );

            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

            //copy results back to host
            CUDA_CHECK_ERROR( cudaMemcpy( hit_before, d_hit_before, sizeof(float), cudaMemcpyDeviceToHost));

            hit_before_agg.at(c) += *hit_before;

        }

        free( scan_xyz );
        free( scan_weight );
        free( dr );
        free( hit_before );
        free( hit_after );
        CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
        CUDA_CHECK_ERROR( cudaFree(d_scan_weight) );
        CUDA_CHECK_ERROR( cudaFree(d_dr) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_before) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_after) );


    }//end scan loop

    //----------- Calculate number of hits in voxels -------------- //

    //figure out hits for all scans
    for( size_t r=0; r< getHitCount(); r++ ){
        if( getHitGridCell(r)>=0 ){
            helios::vec3 direction = getHitXYZ(r)-getScanOrigin(getHitScanID(r));
            direction.normalize();
            hit_inside_agg.at(getHitGridCell(r)) += sin(acos_safe(direction.z));
        }
    }

    //---------------------- Calculate G(theta) --------------------------//

    std::vector<float> Gtheta;
    Gtheta.resize(Ncells,0.f);

    const size_t Ntri = getTriangleCount();

    std::vector<float> denom_sum;
    denom_sum.resize(Ncells,0.f);
    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells,0);

    for( size_t t=0; t<Ntri; t++ ){

        Triangulation tri = getTriangle(t);

        int cell = tri.gridcell;

        if( cell>=0 && cell<Ncells ){ //triangle is inside a grid cell

            helios::vec3 t0 = tri.vertex0;
            helios::vec3 t1 = tri.vertex1;
            helios::vec3 t2 = tri.vertex2;

            helios::vec3 v0 = t1-t0;
            helios::vec3 v1 = t2-t0;
            helios::vec3 v2 = t2-t1;

            float L0 = v0.magnitude();
            float L1 = v1.magnitude();
            float L2 = v2.magnitude();

            float S = 0.5f*(L0+L1+L2);
            float area = sqrt( S*(S-L0)*(S-L1)*(S-L2) );

            helios::vec3 normal = cross( v0, v2 );
            normal.normalize();

            helios::vec3 raydir = t0-getScanOrigin( tri.scanID );
            raydir.normalize();

            float theta = fabs(acos_safe(raydir.z));

            if( area==area ){ //in rare cases you can get area=NaN

                Gtheta.at(cell) += fabs(normal*raydir)*area*fabs(sin(theta));

                denom_sum.at(cell) += fabs(sin(theta))*area;
                cell_tri_count.at(cell) += 1;

            }

        }
    }
    for( uint v=0; v<Ncells; v++ ){
        if( cell_tri_count[v]>0 ){
            Gtheta[v] = Gtheta[v]/denom_sum[v];
            Gtheta_bar[v] += Gtheta[v]/float(Nscans);
        }
    }


    //------------------ Perform inversion to get LAD --------------------//

    if( printmessages ){
        std::cout << "Inverting to find LAD..." << std::flush;
    }

    float etol = 5e-5;
    uint maxiter = 100;

    float error, eold, aold, tmp;
    for( uint v=0; v<Ncells; v++ ){

        if( Gtheta[v]==0 || Gtheta[v]!=Gtheta[v] ){
            // if( printmessages ){
            // 	std::cout << "G(theta) value bad for cell " << v << ": " << Gtheta[v] << std::endl;
            // }
            setCellLeafArea(0,v);
            continue;
        }else if( hit_after_agg[v]-hit_before_agg[v]<0 ){
            if( printmessages ){
                std::cout << "Negative number of rays reaching cell " << v << ": " << hit_after_agg[v] << " " << hit_before_agg[v] << std::endl;
            }
            setCellLeafArea(0,v);
            continue;
        }else if( hit_inside_agg[v]< min_voxel_hits){
            if( printmessages ){
                std::cout << "Not enough hits in voxel: " << hit_inside_agg[v] << " < " << min_voxel_hits << std::endl;
            }
            setCellLeafArea(0,v);
            continue;
        }

        float P = 1.f-float(hit_inside_agg[v])/float(hit_after_agg[v]-hit_before_agg[v]);

        //assert(P>0);

        //initial guesses
        float a = 0.1f;
        float h = 0.01f;

        float mean = 0.f;
        uint count=0;
        for( int j=0; j<dr_agg.at(v).size(); j++ ){
            mean += exp(-a*dr_agg.at(v).at(j)*Gtheta[v]);
            count++;
        }
        mean /= float(dr_agg.at(v).size());
        error = fabs(mean-P)/P;

        tmp = a;

        a = a + h;

        uint iter = 0;
        while( error>etol && iter<maxiter){

            aold = tmp;
            eold = error;

            float mean = 0.f;
            uint count = 0;
            for( int j=0; j<dr_agg.at(v).size(); j++ ){
                mean += exp(-a*dr_agg.at(v).at(j)*Gtheta[v]);
                count ++;
            }
            assert( count!=0 );
            mean /= float(count);
            error = fabs(mean-P)/P;

            tmp = a;

            if( error==eold ){
                break;
            }

            a = fabs( (aold*error-a*eold)/(error-eold) );

            iter++;

        }

        float dr_bar = 0;
        for( uint i=0; i<dr_agg.at(v).size(); i++ ){
            dr_bar += dr_agg.at(v).at(i);
        }
        dr_bar /= float(dr_agg.at(v).size());

        if( iter>=maxiter-1 || a!=a || a>100 ){
            if( printmessages ){
                std::cout << "WARNING: LAD inversion failed for volume #" << v << ". Using average dr formulation." << std::endl;
            }

            a = (1.f-P)/(dr_bar*Gtheta[v]);

        }

        if( a>5 ){
            a = fmin((1.f-P)/dr_bar/Gtheta[v],-log(P)/dr_bar/Gtheta[v]);
        }

        helios::vec3 gridsize = getCellSize(v);
        setCellLeafArea(a*gridsize.x*gridsize.y*gridsize.z,v);

        setCellGtheta( Gtheta[v], v );

        if( printmessages ){
            std::cout << "Vol #" << v << " mean dr: " << dr_bar << std::endl;
            std::cout << "Vol #" << v << " mean G(theta): " << Gtheta[v] << std::endl;
            //std::cout << "Vol #" << v << " intersections: " << hit_inside_agg[v] << " " << hit_after_agg[v] << " " << hit_before_agg[v] << std::endl;

            std::cout << "Vol #" << v << " LAD: " << a << std::endl;
            //std::cout << "Vol #" << v << " LAD: " << a << std::endl;
            //   cout << "Vol #" << v << " LAD: " << (1.f-P)/(dr_bar[v]*Gthetav]) << " [point quadrat]" << endl;
            //std::cout << "Vol #" << v << " Leaf Area: " << getCellLeafArea(v)*100.f*100.f << " [P = mean(exp(-a*dr*Gtheta))]" << std::endl;
            //std::cout << "Vol #" << v << " Leaf Area: " << -log(P)/(dr_bar*Gtheta[v])*gridsize.x*gridsize.y*gridsize.z*10000.f << " [a = -ln(P)/(dr*Gtheta)]" << std::endl;
            // A_sum += LAD[v]*gridsize.x*gridsize.y*gridsize.z*100.f*100.f;
        }

    }

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

void LiDARcloud::calculateLeafAreaGPU_testing( int min_voxel_hits){

    if( printmessages ){
        std::cout << "Calculating leaf area..." << std::endl;
    }

    if( !triangulationcomputed ){
        helios::helios_runtime_error("ERROR (LiDARcloud::calculateLeafAreaGPU): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints().");
    }

    if( !hitgridcellcomputed ){
        calculateHitGridCellGPU();
    }

    const uint Nscans = getScanCount();
    const uint Ncells = getGridCellCount();

    //variable aggregates over all scans where we just keep tacking hits on the end for all scans
    std::vector<std::vector<float> > dr_agg; //dr is path length through grid cell
    dr_agg.resize(Ncells);
    std::vector<float> G_agg; //G is dot product between ray direction and triangle normal that was hit (only relevant for hits)
    G_agg.resize(Ncells,0);
    std::vector<float> hit_before_agg; //hit_before corresponds to scan points that hit something before encountering a particular grid cell
    hit_before_agg.resize(Ncells,0);
    std::vector<float> hit_after_agg; //hit_after corresponds to scan points that hit something after encountering a particular grid cell (including something inside that cell)
    hit_after_agg.resize(Ncells,0);
    std::vector<float> hit_inside_agg; //hit_inside corresponds to scan points that hit something within a particular grid cell.
    hit_inside_agg.resize(Ncells,0);

    //average G(theta)
    std::vector<float> Gtheta_bar;
    Gtheta_bar.resize(Ncells,0.f);

    // We are going to perform all calculations on a scan-by-scan basis: loop through each scan
    for( uint s=0; s<Nscans; s++ ){

        // Perform ray-volume intersection to determine rays that intersected a leaf BEFORE passing through the volume

        std::vector<helios::vec3> this_scan_xyz;

        for( size_t r=0; r< getHitCount(); r++ ){
            if( getHitScanID(r)==s ){
                this_scan_xyz.push_back( getHitXYZ(r) );
            }
        }

        std::vector<helios::vec3> xyz_filled = gapfillMisses(s);
        this_scan_xyz.insert( this_scan_xyz.end(), xyz_filled.begin(), xyz_filled.end() );

        size_t Nhits = this_scan_xyz.size();

        const float3 origin = vec3tofloat3(getScanOrigin(s));

        float3* scan_xyz = (float3*)malloc( Nhits*sizeof(float3) );

        float* scan_weight = (float*)malloc( Nhits*sizeof(float) );

        for( size_t r=0; r<Nhits; r++ ){
            scan_xyz[r] = vec3tofloat3(this_scan_xyz.at(r));
            scan_weight[r] = 1.f;
        }

        float* hit_before = (float*)malloc( sizeof(float));
        float* hit_after = (float*)malloc( sizeof(float));
        float* d_hit_before;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_before, sizeof(float)) );
        float* d_hit_after;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_after, sizeof(float)) );

        float3* d_scan_xyz;
        CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nhits*sizeof(float3)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

        float* d_scan_weight;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_scan_weight, Nhits*sizeof(float)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_weight, scan_weight, Nhits*sizeof(float), cudaMemcpyHostToDevice) );

        float* dr = (float*)malloc( Nhits*sizeof(float));
        float* d_dr;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr, Nhits*sizeof(float)) );

        for( uint c=0; c<Ncells; c++ ){

            //load the attributes of the grid cell
            float3 center = vec3tofloat3(getCellCenter(c));
            float3 anchor = vec3tofloat3(getCellGlobalAnchor(c));
            float3 size = vec3tofloat3(getCellSize(c));
            float rotation = getCellRotation(c);

            CUDA_CHECK_ERROR( cudaMemset( d_dr, 0, Nhits*sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_before, 0, sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_after, 0, sizeof(float)) );

            uint3 dimBlock = make_uint3( min(size_t(512),Nhits), 1, 1 );
            uint3 dimGrid = make_uint3( ceil(float(Nhits)/dimBlock.x), 1, 1  );

            if( dimBlock.x==0 && dimGrid.x==0 ){
                continue;
            }

            intersectGridcell <<< dimGrid, dimBlock >>>( Nhits, origin, d_scan_xyz, d_scan_weight, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after );

            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

            //copy results back to host
            CUDA_CHECK_ERROR( cudaMemcpy( hit_before, d_hit_before, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_after, d_hit_after, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( dr, d_dr, Nhits*sizeof(float), cudaMemcpyDeviceToHost));

            hit_before_agg.at(c) += *hit_before;
            hit_after_agg.at(c) += *hit_after;

            for( size_t r=0; r<Nhits; r++ ){
                if( dr[r]>0.f ){
                    dr_agg.at(c).push_back(dr[r]);
                }
            }

        }

        free( scan_xyz );
        free( scan_weight );
        free( dr );
        free( hit_before );
        free( hit_after );
        CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
        CUDA_CHECK_ERROR( cudaFree(d_scan_weight) );
        CUDA_CHECK_ERROR( cudaFree(d_dr) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_before) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_after) );


    }//end scan loop

    //----------- Calculate number of hits in voxels -------------- //

    //figure out hits for all scans
    for( size_t r=0; r< getHitCount(); r++ ){
        if( getHitGridCell(r)>=0 ){
            helios::vec3 direction = getHitXYZ(r)-getScanOrigin(getHitScanID(r));
            direction.normalize();
            hit_inside_agg.at(getHitGridCell(r)) += sin(acos_safe(direction.z));
        }
    }

    //---------------------- Calculate G(theta) --------------------------//

    std::vector<float> Gtheta;
    Gtheta.resize(Ncells,0.f);

    const size_t Ntri = getTriangleCount();

    std::vector<float> area_sum;
    area_sum.resize(Ncells,0.f);
    std::vector<float> sin_sum;
    sin_sum.resize(Ncells,0.f);
    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells,0);

    for( size_t t=0; t<Ntri; t++ ){

        Triangulation tri = getTriangle(t);

        int cell = tri.gridcell;

        if( cell>=0 && cell<Ncells ){ //triangle is inside a grid cell

            helios::vec3 t0 = tri.vertex0;
            helios::vec3 t1 = tri.vertex1;
            helios::vec3 t2 = tri.vertex2;

            helios::vec3 v0 = t1-t0;
            helios::vec3 v1 = t2-t0;
            helios::vec3 v2 = t2-t1;

            float L0 = v0.magnitude();
            float L1 = v1.magnitude();
            float L2 = v2.magnitude();

            float S = 0.5f*(L0+L1+L2);
            float area = sqrt( S*(S-L0)*(S-L1)*(S-L2) );

            helios::vec3 normal = cross( v0, v2 );
            normal.normalize();

            helios::vec3 raydir = t0-getScanOrigin( tri.scanID );
            raydir.normalize();

            float theta = fabs(acos_safe(raydir.z));

            if( area==area ){ //in rare cases you can get area=NaN

                Gtheta.at(cell) += fabs(normal*raydir)*area*fabs(sin(theta));

                area_sum.at(cell) += area;
                sin_sum.at(cell) += fabs(sin(theta));
                cell_tri_count.at(cell) += 1;

            }

        }
    }
    for( uint v=0; v<Ncells; v++ ){
        if( cell_tri_count[v]>0 ){
            Gtheta[v] *= float(cell_tri_count[v])/(area_sum[v]*sin_sum[v]);
            Gtheta_bar[v] += Gtheta[v]/float(Nscans);
        }
    }


    //------------------ Perform inversion to get LAD --------------------//

    if( printmessages ){
        std::cout << "Inverting to find LAD..." << std::flush;
    }

    float etol = 5e-5;
    uint maxiter = 100;

    float error, eold, aold, tmp;
    for( uint v=0; v<Ncells; v++ ){

        if( Gtheta[v]==0 || Gtheta[v]!=Gtheta[v] ){
            // if( printmessages ){
            // 	std::cout << "G(theta) value bad for cell " << v << ": " << Gtheta[v] << std::endl;
            // }
            setCellLeafArea(0,v);
            continue;
        }else if( hit_inside_agg[v]< min_voxel_hits){
            if( printmessages ){
                std::cout << "Not enough hits in voxel: " << hit_inside_agg[v] << " < " << min_voxel_hits << std::endl;
            }
            setCellLeafArea(0,v);
            continue;
        }

        float P = 1.f-float(hit_inside_agg[v])/float(hit_after_agg[v]);

        //assert(P>0);

        //initial guesses
        float a = 0.1f;
        float h = 0.01f;

        float mean = 0.f;
        uint count=0;
        for( int j=0; j<dr_agg.at(v).size(); j++ ){
            mean += exp(-a*dr_agg.at(v).at(j)*Gtheta[v]);
            count++;
        }
        mean /= float(dr_agg.at(v).size());
        error = fabs(mean-P)/P;

        tmp = a;

        a = a + h;

        uint iter = 0;
        while( error>etol && iter<maxiter){

            aold = tmp;
            eold = error;

            float mean = 0.f;
            uint count = 0;
            for( int j=0; j<dr_agg.at(v).size(); j++ ){
                mean += exp(-a*dr_agg.at(v).at(j)*Gtheta[v]);
                count ++;
            }
            assert( count!=0 );
            mean /= float(count);
            error = fabs(mean-P)/P;

            tmp = a;

            if( error==eold ){
                break;
            }

            a = fabs( (aold*error-a*eold)/(error-eold) );

            iter++;

        }

        float dr_bar = 0;
        for( uint i=0; i<dr_agg.at(v).size(); i++ ){
            dr_bar += dr_agg.at(v).at(i);
        }
        dr_bar /= float(dr_agg.at(v).size());

        if( iter>=maxiter-1 || a!=a || a>100 ){
            if( printmessages ){
                std::cout << "WARNING: LAD inversion failed for volume #" << v << ". Using average dr formulation." << std::endl;
            }

            a = (1.f-P)/(dr_bar*Gtheta[v]);

        }

        if( a>5 ){
            a = fmin((1.f-P)/dr_bar/Gtheta[v],-log(P)/dr_bar/Gtheta[v]);
        }

        helios::vec3 gridsize = getCellSize(v);
        setCellLeafArea(a*gridsize.x*gridsize.y*gridsize.z,v);

        setCellGtheta( Gtheta[v], v );

        if( printmessages ){
            std::cout << "Vol #" << v << " mean dr: " << dr_bar << std::endl;
            std::cout << "Vol #" << v << " mean G(theta): " << Gtheta[v] << std::endl;
            //std::cout << "Vol #" << v << " intersections: " << hit_inside_agg[v] << " " << hit_after_agg[v] << " " << hit_before_agg[v] << std::endl;

            std::cout << "Vol #" << v << " LAD: " << a << std::endl;
            //std::cout << "Vol #" << v << " LAD: " << a << std::endl;
            //   cout << "Vol #" << v << " LAD: " << (1.f-P)/(dr_bar[v]*Gthetav]) << " [point quadrat]" << endl;
            //std::cout << "Vol #" << v << " Leaf Area: " << getCellLeafArea(v)*100.f*100.f << " [P = mean(exp(-a*dr*Gtheta))]" << std::endl;
            //std::cout << "Vol #" << v << " Leaf Area: " << -log(P)/(dr_bar*Gtheta[v])*gridsize.x*gridsize.y*gridsize.z*10000.f << " [a = -ln(P)/(dr*Gtheta)]" << std::endl;
            // A_sum += LAD[v]*gridsize.x*gridsize.y*gridsize.z*100.f*100.f;
        }

    }

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

void LiDARcloud::calculateLeafAreaGPU_synthetic( helios::Context* context, bool beamoutput, bool fillAnalytic  ){

    // calculates LAD using several different methods investigated in Kent & Bailey (2023)
    // writes all voxel level variables to a file
    // optionally writes detailed information about each beam
    // unlike previous calculateLeafAreaGPU versions, this one does not set the cell leaf area variable and does not filter based on minVoxelHits argument

    if( printmessages ){
        std::cout << "Calculating leaf area..." << std::endl;
    }

    if( !triangulationcomputed ){
        helios::helios_runtime_error("ERROR (LiDARcloud::calculateLeafAreaGPU_synthetic): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints().");
    }

    if( !hitgridcellcomputed ){
        calculateHitGridCellGPU();
    }

    const uint Nscans = getScanCount();
    const uint Ncells = getGridCellCount();

    //variable aggregates over all scans where we just keep tacking hits on the end for all scans
    std::vector<std::vector<float> > dr_agg; //dr is path length through grid cell
    dr_agg.resize(Ncells);
    std::vector<float> G_agg; //G is dot product between ray direction and triangle normal that was hit (only relevant for hits)
    G_agg.resize(Ncells,0);

    //average G(theta)
    std::vector<float> Gtheta_bar;
    Gtheta_bar.resize(Ncells,0.f);

    // nested vectors to save P calculation terms
    // cell, scan, variable, value
    std::vector<std::vector<float>>  P_first_numerator_array(Ncells);
    std::vector<std::vector<float>>  P_first_denominator_array(Ncells);

    std::vector<std::vector<float>>  P_equal_numerator_array(Ncells);
    std::vector<std::vector<float>>  P_equal_denominator_array(Ncells);

    std::vector<std::vector<float>>  P_ideal_numerator_array(Ncells);
    std::vector<std::vector<float>>  P_ideal_denominator_array(Ncells);

    std::vector<std::vector<float>>  P_intensity_numerator_array(Ncells);
    std::vector<std::vector<float>>  P_intensity_denominator_array(Ncells);

    std::vector<std::vector<float>>  P_exact_numerator_array(Ncells);
    std::vector<std::vector<float>>  P_exact_denominator_array(Ncells);

    std::vector<std::vector<uint>>  voxel_beam_count_array(Ncells);


    std::vector<std::vector<float>>  dr_array(Ncells);


    // r is the index of all hitpoints (all scans)
    // it is saved for the current scan in this_scan_index
    // i is the index of this_scan_index
    // so when accessing the things using getHitData(), getHitXYZ(), etc,
    // for only one of the scans (this_scan),
    // need to use getHitData(this_scan_index[i]) not getHitData(i)

    ////////////// We are going to perform all calculations on a scan-by-scan basis: loop through each scan
    for( uint s=0; s<Nscans; s++ ){

        if( printmessages ){
            std::cout << "------------------------SCAN = " << s << std::endl;
        }

        //only work with hitpoints assoicated with current scan
        std::vector<helios::vec3> this_scan_xyz;
        std::vector<uint> this_scan_index;
        for( size_t r=0; r< getHitCount(); r++ ){
            if( getHitScanID(r)==s ){
                this_scan_xyz.push_back( getHitXYZ(r) );
                this_scan_index.push_back(r);
            }
        }
        // size of the array of all the hits in this scan
        size_t Nhits = this_scan_xyz.size();

        if( printmessages ){
            std::cout << "Number of hitpoints associated with this scan = Nhits = " << Nhits << std::endl;
        }

        // count the number of beams (i.e. don't count multiple hits per beam)
        float previous_time = -1.f;
        uint Nbeams = 0;
        for(uint i=0; i < Nhits; i++)
        {
            float current_time = getHitData(this_scan_index[i], "timestamp");
            if(current_time != previous_time){
                Nbeams ++;
                previous_time = current_time;
            }
        }

        if( printmessages ){
            std::cout << "Number of beams associated with this scan = Nbeams = " << Nbeams << std::endl;
        }

        // create a vector each element of which represents a unique beam and is another vector of the hit point indices of this_scan_xyz that is associated with this beam
        float previous_beam = getHitData(this_scan_index[0], "timestamp");
        uint beam_ID = 0;
        std::vector<std::vector<uint>> beam_array(Nbeams);
        for(uint i=0; i < Nhits; i++)
        {
            float current_beam = getHitData(this_scan_index[i], "timestamp");

            if(current_beam == previous_beam)
            {
                beam_array.at(beam_ID).push_back(i); //ERK
            }else{
                beam_ID ++;
                beam_array.at(beam_ID).push_back(i); //ERK
                previous_beam = current_beam;
            }

        }

        // set up GPU things for current scan
        const float3 origin = vec3tofloat3(getScanOrigin(s));
        float3* scan_xyz = (float3*)malloc( Nhits*sizeof(float3) );
        float* scan_weight = (float*)malloc( Nhits*sizeof(float) );

        for( size_t i=0; i<Nhits; i++ ){
            scan_xyz[i] = vec3tofloat3(this_scan_xyz.at(i));
            scan_weight[i] = 1.f;
        }

        float* hit_before = (float*)malloc( sizeof(float));
        float* hit_after = (float*)malloc( sizeof(float));
        float* d_hit_before;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_before, sizeof(float)) );
        float* d_hit_after;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_after, sizeof(float)) );

        float3* d_scan_xyz;
        CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nhits*sizeof(float3)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

        float* d_scan_weight;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_scan_weight, Nhits*sizeof(float)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_weight, scan_weight, Nhits*sizeof(float), cudaMemcpyHostToDevice) );

        float* dr = (float*)malloc( Nhits*sizeof(float));
        float* d_dr;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr, Nhits*sizeof(float)) );

        uint* hit_location = (uint*)malloc( Nhits*sizeof(uint));
        uint* d_hit_location;
        CUDA_CHECK_ERROR( cudaMalloc((uint**)&d_hit_location, Nhits*sizeof(uint)) );


        /////////////////////////////////////////////////////////////////////
        // now loop through each voxel
        for( uint c=0; c<Ncells; c++ ){

            if( printmessages ){
                std::cout << "----CELL = " << c << std::endl;
            }

            std::ofstream file_beam;
            if( beamoutput){
                // set up header of file that outputs one row for each beam in the current scan that interacts with the current voxel

                file_beam.open("../beamoutput/beam_data_s_" + std::to_string(s) + "_c_" + std::to_string(c) + ".txt");
                file_beam << "scan, cell, beam, R_before, R_inside, R_after, R_miss, E_before, E_inside, E_after, E_miss, sin_theta, dr, last_dr, I_before, I_inside, I_after, I_miss"  << std::endl;
            }

            //load the attributes of the grid cell
            float3 center = vec3tofloat3(getCellCenter(c));
            float3 anchor = vec3tofloat3(getCellGlobalAnchor(c));
            float3 size = vec3tofloat3(getCellSize(c));
            float rotation = getCellRotation(c);

            CUDA_CHECK_ERROR( cudaMemset( d_hit_location, 0, Nhits*sizeof(uint)) );
            CUDA_CHECK_ERROR( cudaMemset( d_dr, 0, Nhits*sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_before, 0, sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_after, 0, sizeof(float)) );

            uint3 dimBlock = make_uint3( min(size_t(512),Nhits), 1, 1 );
            uint3 dimGrid = make_uint3( ceil(float(Nhits)/dimBlock.x), 1, 1  );

            if( dimBlock.x==0 && dimGrid.x==0 ){
                continue;
            }

            float scanner_range = 1000.0;
            intersectGridcell_synthetic <<< dimGrid, dimBlock >>>( Nhits, origin, d_scan_xyz, d_scan_weight, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after, d_hit_location, scanner_range );

            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

            //copy results back to host
            CUDA_CHECK_ERROR( cudaMemcpy( hit_before, d_hit_before, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_after, d_hit_after, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( dr, d_dr, Nhits*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_location, d_hit_location, Nhits*sizeof(uint), cudaMemcpyDeviceToHost));

            float P_first_numerator = 0;
            float P_first_denominator = 0;
            float P_equal_numerator = 0;
            float P_equal_denominator = 0;
            float P_intensity_numerator = 0;
            float P_intensity_denominator = 0;
            float P_ideal_numerator = 0;
            float P_ideal_denominator = 0;
            float P_exact_numerator = 0;
            float P_exact_denominator = 0;
            uint voxel_beam_count = 0;

            if( printmessages ){
                std::cout << "Nbeams = " << Nbeams << std::endl;
            }
            ////////////////////////// loop through each beam associated with the current scan
            for(int k = 0; k < Nbeams; k++){

                float R_before = 0;
                float R_inside = 0;
                float R_after = 0;
                float R_miss = 0;

                float I_before = 0;
                float I_inside = 0;
                float I_after = 0;
                float I_miss = 0;

                float E_before = 0;
                float E_inside = 0;
                float E_after = 0;
                float E_miss = 0;

                float sin_theta;
                float W = 0;
                float drr = 0;
                float last_drr = 0;

                ////////// loop through the hitpoints in the current beam to get number of rays in each location
                for(int j = 0; j < beam_array.at(k).size(); j++){

                    // pull out the index of the current scan's current beam's current hit (which is used to access the overall hit index r through this_scan_index[i])
                    uint i = beam_array.at(k).at(j);

                    helios::vec3 direction = getHitXYZ(this_scan_index[i])-getScanOrigin(getHitScanID(this_scan_index[i]));
                    direction.normalize();
                    sin_theta = sin(acos_safe(direction.z));

                    last_drr = dr[i];
                    drr += dr[i];

                    if(hit_location[i] == 1){
                        R_before = R_before +  getHitData(this_scan_index[i], "nRaysHit");
                        I_before = I_before + getHitData(this_scan_index[i], "intensity");
                        E_before ++;
                    }else if(hit_location[i] == 2){
                        R_inside = R_inside +  getHitData(this_scan_index[i], "nRaysHit");
                        I_inside = I_inside + getHitData(this_scan_index[i], "intensity");
                        E_inside ++;
                    }else if(hit_location[i] == 3){
                        R_after = R_after +  getHitData(this_scan_index[i], "nRaysHit");
                        I_after = I_after + getHitData(this_scan_index[i], "intensity");
                        E_after ++;
                        if(getHitData(this_scan_index[i], "target_index") == 0){ // if this is the first hitpoint for this beam,
                            W = 1.0;
                        }
                    }else if(hit_location[i] == 4){
                        R_miss = R_miss +  getHitData(this_scan_index[i], "nRaysHit");
                        I_miss = I_miss + getHitData(this_scan_index[i], "intensity");
                        E_miss ++;
                    } // or this hitpoint / beam did not intersect the voxel and should not be added to the total number of beams for this voxel

                } // end of loop through each hit in the current beam

                // calculate the average dr across all hitpoints for the beam (they should all be the same anyway... but just in case)
                float drrx = drr / float(beam_array.at(k).size());

                // if the path length is greater than 0 (it passes throught the current voxel), save it as an element of
                if( drrx > 0.f )
                {
                    dr_array.at(c).push_back(drrx);
                }

                if( beamoutput ){
                    // output info about the current beam to file
                    file_beam << s << "," << c << "," << k << "," << R_before << "," << R_inside << "," << R_after << "," << R_miss << "," << E_before << "," << E_inside << "," << E_after << "," << E_miss << "," << sin_theta << "," << drrx << "," << last_drr << "," << I_before << "," << I_inside << "," << I_after << "," << I_miss << std::endl;
                }

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // calculate terms for different P methods

                // P_exact
                if(R_inside != 0 || R_after != 0 || R_miss != 0){ // only count this beam for P_exact if some energy made it to the voxel
                    P_exact_numerator += ((R_after + R_miss) / (R_before + R_inside + R_after + R_miss));
                    P_exact_denominator += ((R_inside + R_after + R_miss) / (R_before + R_inside + R_after + R_miss));
                    voxel_beam_count++;
                }

                // P_ideal
                if(R_inside != 0 || R_after != 0){ // only count this beam for P_ideal if some energy hit inside or after the voxel (not including misses)
                    P_ideal_numerator += (R_after / (R_before + R_inside + R_after));
                    P_ideal_denominator += ((R_inside + R_after) / (R_before + R_inside + R_after));
                }else if(R_inside == 0 && R_after == 0 && R_before == 0 && R_miss != 0){ // also count this beam if all the energy missed (but still went through the voxel)
                    P_ideal_numerator += 1;
                    P_ideal_denominator += 1;
                }

                // P_intensity
                if(I_inside != 0 || I_after != 0){ // only count this beam for P_intensity if some energy hit inside or after the voxel (not including misses)
                    P_intensity_numerator += (I_after / (I_before + I_inside + I_after));
                    P_intensity_denominator += ((I_inside + I_after) / (I_before + I_inside + I_after));
                }else if(I_inside == 0 && I_after == 0 && I_before == 0 && E_miss != 0){ // also count this beam if all the energy missed (but still went through the voxel)
                    P_intensity_numerator += 1;
                    P_intensity_denominator += 1;
                }

                // P_equal
                if(E_inside != 0 || E_after != 0){ // only count for P_equal if some hit points were inside or after the voxel
                    P_equal_numerator += (E_after / (E_before + E_inside + E_after));
                    P_equal_denominator += ((E_inside + E_after) / (E_before + E_inside + E_after));
                }else if(E_inside == 0 && E_after == 0 && E_before == 0 && E_miss != 0){ // also count this beam if there is only a "hitpoint" that missed (far after the voxel)
                    P_equal_numerator += 1;
                    P_equal_denominator += 1;
                }

                // P_first
                if(E_before == 0 && (E_inside != 0 || E_after != 0)){ // only count for P_first if some hit points were inside or after the voxel
                    P_first_numerator += W;
                    P_first_denominator += 1;
                }else if(E_inside == 0 && E_after == 0 && E_before == 0 && E_miss != 0){ // also count this beam if there is only a "hitpoint" that missed (far after the voxel)
                    P_first_numerator += 1;
                    P_first_denominator += 1;
                }


            }// end of loop through all beams for the current cell

            // save results for the current cell to the arrays

            P_first_numerator_array.at(c).push_back(P_first_numerator);
            P_first_denominator_array.at(c).push_back(P_first_denominator);

            P_equal_numerator_array.at(c).push_back(P_equal_numerator);
            P_equal_denominator_array.at(c).push_back(P_equal_denominator);

            P_intensity_numerator_array.at(c).push_back(P_intensity_numerator);
            P_intensity_denominator_array.at(c).push_back(P_intensity_denominator);

            P_ideal_numerator_array.at(c).push_back(P_ideal_numerator);
            P_ideal_denominator_array.at(c).push_back(P_ideal_denominator);

            P_exact_numerator_array.at(c).push_back(P_exact_numerator);
            P_exact_denominator_array.at(c).push_back(P_exact_denominator);

            voxel_beam_count_array.at(c).push_back(voxel_beam_count);


            for( size_t i=0; i<Nhits; i++ ){
                if( dr[i]>0.f ){
                    dr_agg.at(c).push_back(dr[i]);
                }
            }

            file_beam.close();
        } // end of loop through all cells

        free( scan_xyz );
        free( scan_weight );
        free( dr );
        free( hit_before );
        free( hit_after );
        CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
        CUDA_CHECK_ERROR( cudaFree(d_scan_weight) );
        CUDA_CHECK_ERROR( cudaFree(d_dr) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_before) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_after) );

    }//end scan loop

    std::vector<float> P_first(Ncells);
    std::vector<float> P_equal(Ncells);
    std::vector<float> P_intensity(Ncells);
    std::vector<float> P_ideal(Ncells);
    std::vector<float> P_exact(Ncells);
    std::vector<uint> voxel_beam_count_tot(Ncells);
    std::vector<float> dr_bar(Ncells);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // now aggregate over all scans
    for( uint c=0; c<Ncells; c++ ){

        float P_first_numerator_array_agg = 0;
        float P_first_denominator_array_agg = 0;

        float P_equal_numerator_array_agg = 0;
        float P_equal_denominator_array_agg = 0;

        float P_intensity_numerator_array_agg= 0;
        float P_intensity_denominator_array_agg= 0;

        float P_ideal_numerator_array_agg= 0;
        float P_ideal_denominator_array_agg= 0;

        float P_exact_numerator_array_agg= 0;
        float P_exact_denominator_array_agg= 0;

        uint voxel_beam_count_array_agg=0;


        for( uint s=0; s<Nscans; s++ ){

            P_first_numerator_array_agg += P_first_numerator_array.at(c).at(s);
            P_first_denominator_array_agg += P_first_denominator_array.at(c).at(s);

            P_equal_numerator_array_agg += P_equal_numerator_array.at(c).at(s);
            P_equal_denominator_array_agg += P_equal_denominator_array.at(c).at(s);

            P_intensity_numerator_array_agg += P_intensity_numerator_array.at(c).at(s);
            P_intensity_denominator_array_agg += P_intensity_denominator_array.at(c).at(s);

            P_ideal_numerator_array_agg += P_ideal_numerator_array.at(c).at(s);
            P_ideal_denominator_array_agg += P_ideal_denominator_array.at(c).at(s);

            P_exact_numerator_array_agg += P_exact_numerator_array.at(c).at(s);
            P_exact_denominator_array_agg += P_exact_denominator_array.at(c).at(s);
            voxel_beam_count_array_agg += voxel_beam_count_array.at(c).at(s);
        }


        P_first[c] = P_first_numerator_array_agg / P_first_denominator_array_agg ;
        P_equal[c] = P_equal_numerator_array_agg / P_equal_denominator_array_agg ;
        P_intensity[c] = P_intensity_numerator_array_agg / P_intensity_denominator_array_agg ;
        P_ideal[c] = P_ideal_numerator_array_agg / P_ideal_denominator_array_agg ;
        P_exact[c] = P_exact_numerator_array_agg / P_exact_denominator_array_agg ;
        voxel_beam_count_tot[c] = voxel_beam_count_array_agg;
        if( printmessages ){
            std::cout << "Cell " << c << ", voxel_beam_count = " << voxel_beam_count_tot[c]  << std::endl;
            std::cout << "Cell " << c << ", P_first = " << P_first[c] << std::endl;
            std::cout << "Cell " << c << ", P_equal = " << P_equal[c] << std::endl;
            std::cout << "Cell " << c << ", P_intensity = " << P_intensity[c] << std::endl;
            std::cout << "Cell " << c << ", P_ideal = " << P_ideal[c] << std::endl;
            std::cout << "Cell " << c << ", P_exact = " << P_exact[c] << std::endl;
        }

        float mean_dr = 0;
        for(int j = 0; j < dr_array.at(c).size(); j++)
        {
            mean_dr += dr_array.at(c).at(j);
        }
        dr_bar[c] = mean_dr/ float(dr_array.at(c).size());
        if( printmessages ){
            std::cout << "Cell " << c << ", dr_bar = " <<  dr_bar[c] << std::endl;
        }
    } //end of loop through cells to aggregate all scans


    //---------------------- Calculate G(theta) from triangulation --------------------------//

    std::vector<float> Gtheta;
    Gtheta.resize(Ncells,0.f);

    const size_t Ntri = getTriangleCount();

    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells,0);

    std::vector<float> area_sin_sum;
    area_sin_sum.resize(Ncells,0.f);

    for( size_t t=0; t<Ntri; t++ ){

        Triangulation tri = getTriangle(t);

        int cell = tri.gridcell;

        if( cell>=0 && cell<Ncells ){ //triangle is inside a grid cell

            helios::vec3 t0 = tri.vertex0;
            helios::vec3 t1 = tri.vertex1;
            helios::vec3 t2 = tri.vertex2;

            helios::vec3 v0 = t1-t0;
            helios::vec3 v1 = t2-t0;
            helios::vec3 v2 = t2-t1;

            float L0 = v0.magnitude();
            float L1 = v1.magnitude();
            float L2 = v2.magnitude();

            float S = 0.5f*(L0+L1+L2);
            float area = sqrt( S*(S-L0)*(S-L1)*(S-L2) );

            helios::vec3 normal = cross( v0, v2 );
            normal.normalize();

            helios::vec3 raydir = t0-getScanOrigin( tri.scanID );
            raydir.normalize();

            float theta = fabs(acos_safe(raydir.z));

            if( area==area ){ //in rare cases you can get area=NaN

                Gtheta.at(cell) += fabs(normal*raydir)*area*fabs(sin(theta));

                area_sin_sum.at(cell) += area*fabs(sin(theta));
                cell_tri_count.at(cell) += 1;

            }

        }
    }

    for( uint c=0; c<Ncells; c++ ){
        if( cell_tri_count[c]>0 ){
            //Gtheta[c] *= float(cell_tri_count[c])/(area_sum[c]*sin_sum[c]);
            //Gtheta[c] *= float(cell_tri_count[c])/(area_sin_sum[c]);
            Gtheta[c] *= 1.0/(area_sin_sum[c]);

            Gtheta_bar[c] += Gtheta[c]/float(Nscans);
        }
    }


    //------------------Calculate "Reference" P and also mean and variance of dr --------------------//
    // this is the value of P calculated using the reference values of LAD and Gtheta (from calculateSyntheticLeafArea, calculateSyntheticGtheta),
    // and the path length from all beams for a given cell

    // first get the reference LA and Gtheta using modified functions that save output to arrays
    std::vector<float> LA_ref = calculateSyntheticLeafArea(context);
    std::vector<float> Gtheta_ref = calculateSyntheticGtheta(context);

    std::vector<float> P_ref(Ncells);
    P_ref.resize(Ncells,0.f);

    std::vector<float> dr_bar_ref(Ncells);
    dr_bar_ref.resize(Ncells,0.f);

    std::vector<float> dr_var_ref(Ncells);
    dr_var_ref.resize(Ncells,0.f);

    for(uint c=0; c<Ncells; c++)
    {
        helios::vec3 gridsize = getCellSize(c);
        float LAD_ref = LA_ref.at(c) / (gridsize.x*gridsize.y*gridsize.z);
        float G_ref = Gtheta_ref.at(c);

        float dr_bar_tmp = 0;
        float sum_term = 0;
        for(uint j=0; j<dr_array.at(c).size(); j++)
        {
            dr_bar_tmp += dr_array.at(c).at(j);
            sum_term += exp(-1.0*LAD_ref*G_ref*dr_array.at(c).at(j));
        }
        sum_term /= float(dr_array.at(c).size());
        dr_bar_tmp /= float(dr_array.at(c).size());

        dr_bar_ref[c] = dr_bar_tmp;
        P_ref[c] = sum_term;
    }

    for(uint c=0; c<Ncells; c++)
    {
        float dr_var = 0;
        for(uint j=0; j<dr_array.at(c).size(); j++)
        {
            dr_var += pow(dr_array.at(c).at(j) - dr_bar_ref[c], 2.0);
        }
        dr_var /= float(dr_array.at(c).size());
        dr_var_ref[c] = dr_var;
    }

    if( printmessages ){
        std::cout << "finished P_ref and dr calculations" << std::endl;
    }

    //------------------ Perform inversions to get LAD using the different methods for P --------------------//

    if( printmessages ){
        std::cout << "Inverting to find LAD..." << std::flush;
    }

    // perform LAD inversion using reference Gtheta
    std::vector<float> LAD_refcheck_Gref = LAD_inversion(P_ref, Gtheta_ref, dr_array, fillAnalytic);
    std::vector<float> LAD_first_Gref = LAD_inversion(P_first, Gtheta_ref, dr_array, fillAnalytic);
    std::vector<float> LAD_equal_Gref = LAD_inversion(P_equal, Gtheta_ref, dr_array, fillAnalytic);
    std::vector<float> LAD_intensity_Gref = LAD_inversion(P_intensity, Gtheta_ref, dr_array, fillAnalytic);
    std::vector<float> LAD_ideal_Gref = LAD_inversion(P_ideal, Gtheta_ref, dr_array, fillAnalytic);
    std::vector<float> LAD_exact_Gref = LAD_inversion(P_exact, Gtheta_ref, dr_array, fillAnalytic );

    // perform LAD inversion using triangulation-estimated Gtheta
    std::vector<float> LAD_refcheck = LAD_inversion(P_ref, Gtheta, dr_array, fillAnalytic);
    std::vector<float> LAD_first = LAD_inversion(P_first, Gtheta, dr_array, fillAnalytic);
    std::vector<float> LAD_equal = LAD_inversion(P_equal, Gtheta, dr_array, fillAnalytic);
    std::vector<float> LAD_intensity = LAD_inversion(P_intensity, Gtheta, dr_array, fillAnalytic);
    std::vector<float> LAD_ideal = LAD_inversion(P_ideal, Gtheta, dr_array, fillAnalytic);
    std::vector<float> LAD_exact = LAD_inversion(P_exact, Gtheta, dr_array, fillAnalytic);

    if( printmessages ){
        std::cout << "finished LAD inversions" << std::endl;
    }

    for(uint c=0; c<Ncells; c++)
    {
        helios::vec3 grid_size = getCellSize(c);
        helios::vec3 grid_center = getCellCenter(c);

        if( printmessages ){
            std::cout << "Cell " << c << ", voxel_beam_count = " << voxel_beam_count_tot[c]  << std::endl;
            std::cout << "Cell " << c << ", P_first = " << P_first[c] << std::endl;
            std::cout << "Cell " << c << ", P_equal = " << P_equal[c] << std::endl;
            std::cout << "Cell " << c << ", P_intensity = " << P_intensity[c] << std::endl;
            std::cout << "Cell " << c << ", P_exact = " << P_exact[c] << std::endl;
            std::cout << "Cell " << c << ", P_ref = " << P_ref[c] << std::endl;

            std::cout << " " << std::endl;
            std::cout << "Cell " << c << ", Gref = " << Gtheta_ref[c] << std::endl;
            std::cout << "Cell " << c << ", Gtri = " << Gtheta[c] << std::endl;
            std::cout << " " << std::endl;
        }


        if( printmessages ){
            std::cout << "Cell " << c << ", LAD_first = " << LAD_first[c] << std::endl;
            std::cout << "Cell " << c << ", LAD_equal = " << LAD_equal[c] << std::endl;
            std::cout << "Cell " << c << ", LAD_intensity = " << LAD_intensity[c] << std::endl;
            std::cout << "Cell " << c << ", LAD_exact = " << LAD_exact[c] << std::endl;

            std::cout << " " << std::endl;

            std::cout << "Cell " << c << ", LAD_first_Gref = " << LAD_first_Gref[c] << std::endl;
            std::cout << "Cell " << c << ", LAD_equal_Gref = " << LAD_equal_Gref[c] << std::endl;
            std::cout << "Cell " << c << ", LAD_intensity_Gref = " << LAD_intensity_Gref[c] << std::endl;
            std::cout << "Cell " << c << ", LAD_exact_Gref = " << LAD_exact_Gref[c] << std::endl;

            std::cout << "Cell " << c << ", LAD_ref = " <<LA_ref[c]/(grid_size.x*grid_size.y*grid_size.z) << std::endl;

        }
    }



    // output the voxel level variables
    std::ofstream file_output;
    file_output.open("../voxeloutput/voxeloutput.txt");
    file_output << "cell, grid_center_x, grid_center_y, grid_center_z, grid_size_x, grid_size_y, grid_size_z, Nbeams, LAD_ref, G_ref, dr_bar, dr_var, P_ref, G, P_first, P_equal, P_intensity, P_ideal, P_exact, LADGref_refcheck, LADGref_first, LADGref_equal, LADGref_intensity, LADGref_ideal,LADGref_exact, LAD_refcheck, LAD_first, LAD_equal, LAD_intensity, LAD_ideal, LAD_exact"  << std::endl;

    for(uint c=0; c<Ncells; c++)
    {
        helios::vec3 grid_size = getCellSize(c);
        helios::vec3 grid_center = getCellCenter(c);

        file_output << c << "," << grid_center.x << "," << grid_center.y << "," << grid_center.z << "," << grid_size.x << "," << grid_size.y << "," << grid_size.z << "," << voxel_beam_count_tot.at(c) << "," << LA_ref[c]/(grid_size.x*grid_size.y*grid_size.z) << "," << Gtheta_ref[c] << "," <<
                    dr_bar_ref[c] << "," << dr_var_ref[c] << "," << P_ref[c] << "," << Gtheta[c] << "," << P_first[c] << "," << P_equal[c] << ","  << P_intensity[c] << "," << P_ideal[c] << "," << P_exact[c] << "," << LAD_refcheck_Gref[c] << "," << LAD_first_Gref[c] << "," << LAD_equal_Gref[c] << "," <<
                    LAD_intensity_Gref[c] << "," << LAD_ideal_Gref[c] << "," << LAD_exact_Gref[c] << "," << LAD_refcheck[c] << "," << LAD_first[c] << "," << LAD_equal[c] << "," << LAD_intensity[c] << "," << LAD_ideal[c] << "," << LAD_exact[c] << std::endl;
    }
    file_output.close();

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

std::vector<float> LiDARcloud::LAD_inversion(std::vector<float> &P, std::vector<float> &Gtheta,
                                             std::vector<std::vector<float>> &dr_array, bool fillAnalytic){

    float etol = 5e-5;
    uint maxiter = 100;

    int Ncells = getGridCellCount();

    float error, eold, aold, tmp;

    std::vector<float> LAD(Ncells);
    LAD.resize(Ncells,0.f);

    for( uint v=0; v<Ncells; v++ ){

        //initial guesses
        float a = 0.1f;
        float h = 0.01f;

        float mean = 0.f;
        uint count=0;
        //loop through dr value for each beam
        for( int j=0; j<dr_array.at(v).size(); j++ ){
            mean += exp(-a*dr_array.at(v).at(j)*Gtheta[v]);
            count++;
        }
        mean /= float(count);
        error = fabs(mean-P[v])/P[v];

        tmp = a;
        a = a + h;

        uint iter = 0;
        while( error>etol && iter<maxiter){

            aold = tmp;
            eold = error;

            float mean = 0.f;
            uint count = 0;
            for( int j=0; j<dr_array.at(v).size(); j++ ){
                mean += exp(-a*dr_array.at(v).at(j)*Gtheta[v]);
                count ++;
            }
            assert( count!=0 );
            mean /= float(count);
            error = fabs(mean-P[v])/P[v];

            tmp = a;

            if( error==eold ){
                break;
            }

            a = fabs( (aold*error-a*eold)/(error-eold) );
            iter++;
            //  std::cout << "iter = " << iter << ", a = " << a << std::endl;
        }

        // check if the inversion failed
        if( iter>=maxiter-1 || a!=a || a>100 ){

            if(fillAnalytic)
            {

                if( printmessages ){
                    std::cout << "WARNING: LAD inversion failed for volume #" << v << ". Using average dr formulation." << std::endl;
                }

                // calculate the mean dr for the cell
                float dr_bar = 0;
                for(uint j=0; j<dr_array.at(v).size(); j++)
                {
                    dr_bar += dr_array.at(v).at(j);
                }
                dr_bar /= float(dr_array.at(v).size());

                //do the analytic solution
                a =   -log(P.at(v))/(dr_bar*Gtheta.at(v));

            }else{

                if( printmessages ){
                    std::cout << "WARNING: LAD inversion failed for volume #" << v << ". setting LAD to 999"  << std::endl;
                }

                a = 999.0;
            }
        }

        LAD[v] = a;
    }

    return LAD;
}


__global__ void LIDAR_CUDA::insideVolume( const uint Nhits, const float3* d_hit_xyz, const uint Ngridcells, const float3* d_grid_size, const float3* d_grid_center, const float3* d_grid_anchor, const float* d_grid_rotation, int* d_hit_vol ){

    uint t = blockIdx.x*blockDim.x+threadIdx.x;

    if( t>=Nhits ){
        return;
    }

    d_hit_vol[t] = -1;

    float3 hit_xyz = d_hit_xyz[t];

    for( int i=0; i<Ngridcells; i++ ){

        float3 center = d_grid_center[i];
        float3 anchor = d_grid_anchor[i];
        float3 size = d_grid_size[i];
        float rotation = d_grid_rotation[i];

        float3 origin = make_float3(0,0,0);

        float3 hit_xyz_rot = d_rotatePoint(hit_xyz-anchor,0,-rotation) + anchor;

        float3 direction = normalize(hit_xyz_rot-origin);

        float ox = origin.x; float oy = origin.y; float oz = origin.z;
        float dx = direction.x; float dy = direction.y; float dz = direction.z;

        float x0 = center.x - 0.5f*size.x; float x1 = center.x + 0.5f*size.x;
        float y0 = center.y - 0.5f*size.y; float y1 = center.y + 0.5f*size.y;
        float z0 = center.z - 0.5f*size.z; float z1 = center.z + 0.5f*size.z;

        float tx_min, ty_min, tz_min;
        float tx_max, ty_max, tz_max;

        float a = 1.0 / dx;
        if (a >= 0) {
            tx_min = (x0 - ox) * a;
            tx_max = (x1 - ox) * a;
        }
        else {
            tx_min = (x1 - ox) * a;
            tx_max = (x0 - ox) * a;
        }

        float b = 1.0 / dy;
        if (b >= 0) {
            ty_min = (y0 - oy) * b;
            ty_max = (y1 - oy) * b;
        }
        else {
            ty_min = (y1 - oy) * b;
            ty_max = (y0 - oy) * b;
        }

        float c = 1.0 / dz;
        if (c >= 0) {
            tz_min = (z0 - oz) * c;
            tz_max = (z1 - oz) * c;
        }
        else {
            tz_min = (z1 - oz) * c;
            tz_max = (z0 - oz) * c;
        }

        float t0, t1;

        // find largest entering t value

        if (tx_min > ty_min)
            t0 = tx_min;
        else
            t0 = ty_min;

        if (tz_min > t0)
            t0 = tz_min;

        // find smallest exiting t value

        if (tx_max < ty_max)
            t1 = tx_max;
        else
            t1 = ty_max;

        if (tz_max < t1)
            t1 = tz_max;

        if (t0 < t1 && t1 > 1e-6){ //Ray passed through box
            float T = magnitude(hit_xyz_rot-origin);
            if( T>=t0 && T<=t1 ){ //Ray endpoint is inside box
                d_hit_vol[t] = i;
                return;
            }
        }

    }

}

__global__ void LIDAR_CUDA::intersectBoundingBox( const size_t scanSize, const float3 origin, const float3* d_scan_xyz, const float3 bbcenter, const float3 bbsize, uint* d_boundingbox_hit ){

    size_t i = blockIdx.x*blockDim.x+threadIdx.x;

    if( i>=scanSize ){
        return;
    }

    float3 scan_xyz = d_scan_xyz[ i ];

    float3 direction = normalize( scan_xyz - origin );

    float ox = origin.x; float oy = origin.y; float oz = origin.z;
    float dx = direction.x; float dy = direction.y; float dz = direction.z;

    float x0 = bbcenter.x - 0.5f*bbsize.x; float x1 = bbcenter.x + 0.5f*bbsize.x;
    float y0 = bbcenter.y - 0.5f*bbsize.y; float y1 = bbcenter.y + 0.5f*bbsize.y;
    float z0 = bbcenter.z - 0.5f*bbsize.z; float z1 = bbcenter.z + 0.5f*bbsize.z;

    //first check if we are inside the bounding box

    if( ox>=x0 && ox<=x1 && oy>=y0 && oy<=y1 && oz>=z0 && oz<=z1 ){
        d_boundingbox_hit[ i ] = 1;
        return;
    }

    //if not inside, intersect bounding box

    float tx_min, ty_min, tz_min;
    float tx_max, ty_max, tz_max;

    float a = 1.0 / dx;
    if (a >= 0) {
        tx_min = (x0 - ox) * a;
        tx_max = (x1 - ox) * a;
    }
    else {
        tx_min = (x1 - ox) * a;
        tx_max = (x0 - ox) * a;
    }

    float b = 1.0 / dy;
    if (b >= 0) {
        ty_min = (y0 - oy) * b;
        ty_max = (y1 - oy) * b;
    }
    else {
        ty_min = (y1 - oy) * b;
        ty_max = (y0 - oy) * b;
    }

    float c = 1.0 / dz;
    if (c >= 0) {
        tz_min = (z0 - oz) * c;
        tz_max = (z1 - oz) * c;
    }
    else {
        tz_min = (z1 - oz) * c;
        tz_max = (z0 - oz) * c;
    }

    float t0, t1;

    // find largest entering t value

    if (tx_min > ty_min)
        t0 = tx_min;
    else
        t0 = ty_min;

    if (tz_min > t0)
        t0 = tz_min;

    // find smallest exiting t value

    if (tx_max < ty_max)
        t1 = tx_max;
    else
        t1 = ty_max;

    if (tz_max < t1)
        t1 = tz_max;

    if (t0 < t1 && t1 > 1e-6){ //Ray passed through box
        d_boundingbox_hit[ i ] = 1;
    }

}

__global__ void LIDAR_CUDA::intersectGridcell( const size_t Nhitsbb, const float3 origin, float3* d_scan_xyz, float* d_scan_weight, const float3 center, const float3 anchor, const float3 size, const float rotation, float* d_dr, float* hit_before, float* hit_after ){

    size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

    if( idx>=Nhitsbb ){
        return;
    }

    float3 scan_xyz = d_scan_xyz[ idx ];

    //Inverse rotate the ray

    float3 origin_rot = d_rotatePoint(origin-anchor,0,-rotation) + anchor;
    float3 scan_xyz_rot = d_rotatePoint(scan_xyz-anchor,0,-rotation) + anchor;

    float3 direction = normalize(scan_xyz_rot-origin_rot);

    float ox = origin_rot.x; float oy = origin_rot.y; float oz = origin_rot.z;
    float dx = direction.x; float dy = direction.y; float dz = direction.z;

    float x0 = center.x - 0.5f*size.x; float x1 = center.x + 0.5f*size.x;
    float y0 = center.y - 0.5f*size.y; float y1 = center.y + 0.5f*size.y;
    float z0 = center.z - 0.5f*size.z; float z1 = center.z + 0.5f*size.z;

    float tx_min, ty_min, tz_min;
    float tx_max, ty_max, tz_max;

    float a = 1.0 / dx;
    if (a >= 0) {
        tx_min = (x0 - ox) * a;
        tx_max = (x1 - ox) * a;
    }
    else {
        tx_min = (x1 - ox) * a;
        tx_max = (x0 - ox) * a;
    }

    float b = 1.0 / dy;
    if (b >= 0) {
        ty_min = (y0 - oy) * b;
        ty_max = (y1 - oy) * b;
    }
    else {
        ty_min = (y1 - oy) * b;
        ty_max = (y0 - oy) * b;
    }

    float c = 1.0 / dz;
    if (c >= 0) {
        tz_min = (z0 - oz) * c;
        tz_max = (z1 - oz) * c;
    }
    else {
        tz_min = (z1 - oz) * c;
        tz_max = (z0 - oz) * c;
    }

    float t0, t1;

    // find largest entering t value

    if (tx_min > ty_min)
        t0 = tx_min;
    else
        t0 = ty_min;

    if (tz_min > t0)
        t0 = tz_min;

    // find smallest exiting t value

    if (tx_max < ty_max)
        t1 = tx_max;
    else
        t1 = ty_max;

    if (tz_max < t1)
        t1 = tz_max;

    if (t0 < t1 && t1 > 1e-6){ //Ray passed through box
        float t = magnitude( scan_xyz_rot - origin_rot );  //t-distance to hit point

        if( t>=t0 ){ //hit lies within or beyond the volume

            atomicAdd( hit_after, sin(acos_safe(dz))*d_scan_weight[idx] );

            d_dr[ idx ] = fabs(t1-t0);

        }else if( t<t0 ){ //hit lies before the volume

            atomicAdd( hit_before, sin(acos_safe(dz))*d_scan_weight[idx] );

        }

    }

}

__global__ void LIDAR_CUDA::intersectGridcell_synthetic( const size_t Nhitsbb, const float3 origin, float3* d_scan_xyz, float* d_scan_weight, const float3 center, const float3 anchor, const float3 size, const float rotation, float* d_dr, float* hit_before, float* hit_after, uint* d_hit_location, const float scanner_range ){

    size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

    if( idx>=Nhitsbb ){
        return;
    }

    float3 scan_xyz = d_scan_xyz[ idx ];

    //Inverse rotate the ray

    float3 origin_rot = d_rotatePoint(origin-anchor,0,-rotation) + anchor;
    float3 scan_xyz_rot = d_rotatePoint(scan_xyz-anchor,0,-rotation) + anchor;

    float3 direction = normalize(scan_xyz_rot-origin_rot);

    float ox = origin_rot.x; float oy = origin_rot.y; float oz = origin_rot.z;
    float dx = direction.x; float dy = direction.y; float dz = direction.z;

    float x0 = center.x - 0.5f*size.x; float x1 = center.x + 0.5f*size.x;
    float y0 = center.y - 0.5f*size.y; float y1 = center.y + 0.5f*size.y;
    float z0 = center.z - 0.5f*size.z; float z1 = center.z + 0.5f*size.z;

    float tx_min, ty_min, tz_min;
    float tx_max, ty_max, tz_max;

    float a = 1.0 / dx;
    if (a >= 0) {
        tx_min = (x0 - ox) * a;
        tx_max = (x1 - ox) * a;
    }
    else {
        tx_min = (x1 - ox) * a;
        tx_max = (x0 - ox) * a;
    }

    float b = 1.0 / dy;
    if (b >= 0) {
        ty_min = (y0 - oy) * b;
        ty_max = (y1 - oy) * b;
    }
    else {
        ty_min = (y1 - oy) * b;
        ty_max = (y0 - oy) * b;
    }

    float c = 1.0 / dz;
    if (c >= 0) {
        tz_min = (z0 - oz) * c;
        tz_max = (z1 - oz) * c;
    }
    else {
        tz_min = (z1 - oz) * c;
        tz_max = (z0 - oz) * c;
    }

    float t0, t1;

    // find largest entering t value

    if (tx_min > ty_min)
        t0 = tx_min;
    else
        t0 = ty_min;

    if (tz_min > t0)
        t0 = tz_min;

    // find smallest exiting t value

    if (tx_max < ty_max)
        t1 = tx_max;
    else
        t1 = ty_max;

    if (tz_max < t1)
        t1 = tz_max;

    if (t0 < t1 && t1 > 1e-6){ //Ray passed through box
        float t = magnitude( scan_xyz_rot - origin_rot );  //t-distance to hit point

        if( t>=t0 && t<= t1 ){ //hit lies within the volume
            atomicAdd( hit_after, sin(acos_safe(dz))*d_scan_weight[idx] );
            d_hit_location[ idx ] = 2;
            d_dr[ idx ] = fabs(t1-t0);
        }else if( t > t1 && t < scanner_range){ //hit lies after the volume (but not a miss)
            d_hit_location[ idx ] = 3;
            d_dr[ idx ] = fabs(t1-t0);
        }else if( t > t1 && t >= scanner_range){ //hit lies after volume but out of range of scanner (miss)
            d_hit_location[ idx ] = 4;
            d_dr[ idx ] = fabs(t1-t0);
        }else if( t<t0 ){ //hit lies before the volume
            d_hit_location[ idx ] = 1;
            atomicAdd( hit_before, sin(acos_safe(dz))*d_scan_weight[idx] );

        }

    }

}

void LiDARcloud::trunkReconstruction(const helios::vec3 &box_center,
                                     const helios::vec3 &box_size, float Lmax, float max_aspect_ratio ){

    if( printmessages ){
        std::cout << "Performing trunk reconstruction..." << std::flush;
    }

    // ---- Determine which points are in trunk volume ---- //

    float3* d_hit_xyz;

    const uint N = getHitCount();

    float3* hit_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_xyz,N*sizeof(float3)) ); //allocate device memory

    //copy scan data into the host buffer
    for( std::size_t r=0; r<N; r++ ){
        hit_xyz[r] = vec3tofloat3(getHitXYZ(r));
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_xyz, hit_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );

    const uint Ncells = getGridCellCount();

    float3 grid_center = vec3tofloat3(box_center);
    float3 grid_size = vec3tofloat3(box_size);
    float grid_rotation = 0;

    float3* d_grid_center;
    float3* d_grid_size;
    float* d_grid_rotation;

    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_center,sizeof(float3)) );
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_size,sizeof(float3)) );
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_rotation,sizeof(float)) );

    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_center, &grid_center, sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_size, &grid_size, sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_rotation, &grid_rotation, sizeof(float), cudaMemcpyHostToDevice) );

    free(hit_xyz);

    // Result buffer
    int* hit_vol = (int*)malloc( N * sizeof(int));
    int* d_hit_vol;
    CUDA_CHECK_ERROR( cudaMalloc(&d_hit_vol,N*sizeof(int)) );

    dim3 dimBlock( 64, 1 );
    dim3 dimGrid( ceil(N/64.f) );
    insideVolume <<< dimGrid, dimBlock >>>( N, d_hit_xyz, 1, d_grid_size, d_grid_center, d_grid_center, d_grid_rotation, d_hit_vol );

    CUDA_CHECK_ERROR( cudaPeekAtLastError() );
    CUDA_CHECK_ERROR( cudaDeviceSynchronize() );

    CUDA_CHECK_ERROR( cudaMemcpy(hit_vol, d_hit_vol, N*sizeof(int), cudaMemcpyDeviceToHost) );

    CUDA_CHECK_ERROR( cudaFree(d_hit_vol) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_xyz) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_center) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_size) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_rotation) );

    // ------ Triangulate trunk points -------- //

    for( uint s=0; s< getScanCount(); s++ ){

        std::vector<int> Delaunay_inds;

        std::vector<Shx> pts;
        Shx pt;

        int count = 0;
        for( int r=0; r<N; r++ ){

            if( getHitScanID(r)==s && hit_vol[r]>=0 ){

                helios::SphericalCoord direction = getHitRaydir(r);

                //pt.id = count;
                pt.id = r;
                pt.r = direction.zenith;
                pt.c = direction.azimuth;

                pts.push_back(pt);

                Delaunay_inds.push_back(r);

                count++;

            }

        }

        std::vector<int> dupes;
        int nx = de_duplicate( pts, dupes);

        std::vector<Triad> triads;

        s_hull_pro( pts, triads );

        std::ofstream file;
        file.open("nodes.txt");

        std::ofstream filea;
        filea.open("aspect.txt");

        std::ofstream filev;
        filev.open("vertices.txt");
        for( int r=0; r<N; r++ ){
            helios::vec3 pos = getHitXYZ(r);
            filev << pos.x << " " << pos.y << " " << pos.z << std::endl;
        }
        filev.close();

        std::ofstream filed;
        filed.open("angles.txt");
        for( int r=0; r<N; r++ ){
            if( getHitScanID(r)==s && hit_vol[r]>=0 ){
                helios::SphericalCoord direction = getHitRaydir(r);
                filed << direction.zenith << " " << direction.azimuth << std::endl;
            }
        }
        filed.close();

        std::vector<Triangulation> trunk_triangles;
        for( int t=0; t<triads.size(); t++ ){

            // int ID0 = Delaunay_inds.at(triads.at(t).a);
            // int ID1 = Delaunay_inds.at(triads.at(t).b);
            // int ID2 = Delaunay_inds.at(triads.at(t).c);
            int ID0 = triads.at(t).a;
            int ID1 = triads.at(t).b;
            int ID2 = triads.at(t).c;

            helios::vec3 vertex0 = getHitXYZ( ID0 );
            helios::vec3 vertex1 = getHitXYZ( ID1 );
            helios::vec3 vertex2 = getHitXYZ( ID2 );

            helios::vec3 v;
            v=vertex0-vertex1;
            float L0 = v.magnitude();
            v=vertex0-vertex2;
            float L1 = v.magnitude();
            v=vertex1-vertex2;
            float L2 = v.magnitude();

            float aspect_ratio = fmax(fmax(L0,L1),L2)/fmin(fmin(L0,L1),L2);

            if( L0>Lmax || L1>Lmax || L2>Lmax || aspect_ratio>max_aspect_ratio ){
                continue;
            }

            helios::RGBcolor color = helios::make_RGBcolor(0,0,0);
            color.r = (hits.at(ID0).color.r + hits.at(ID1).color.r + hits.at(ID2).color.r )/3.f;
            color.g = (hits.at(ID0).color.g + hits.at(ID1).color.g + hits.at(ID2).color.g )/3.f;
            color.b = (hits.at(ID0).color.b + hits.at(ID1).color.b + hits.at(ID2).color.b )/3.f;

            Triangulation tri( s, vertex0, vertex1, vertex2, ID0, ID1, ID2, color, 0 );

            trunk_triangles.push_back(tri);

            file << ID0 << " " << ID1 << " " << ID2 << std::endl;
            filea << aspect_ratio << std::endl;

        }

        file.close();
        filea.close();

        // ------ Flood fill -------- //

        int current_group = 0;

        std::vector<std::vector<int> > nodes;
        nodes.resize(getHitCount());

        size_t Ntri = trunk_triangles.size();

        for( size_t t=0; t<Ntri; t++ ){

            Triangulation tri = trunk_triangles.at(t);

            nodes.at( tri.ID0 ).push_back(t);
            nodes.at( tri.ID1 ).push_back(t);
            nodes.at( tri.ID2 ).push_back(t);

        }

        std::vector<int> fill_flag;
        fill_flag.resize(Ntri,-1);

        for( size_t t=0; t<Ntri; t++ ){//looping through all triangles

            if( fill_flag.at(t)<0 ){

                floodfill( t, trunk_triangles, fill_flag, nodes, current_group, 0, 1e6 );

                current_group ++;

            }

        }

        std::vector<std::vector<Triangulation> > trunk_groups;

        for( size_t t=0; t<Ntri; t++ ){//looping through all triangles

            if( fill_flag.at(t)>=0 ){
                int fill_group = fill_flag.at(t);

                if( fill_group>=trunk_groups.size() ){
                    trunk_groups.resize( fill_group+1 );
                }

                trunk_groups.at(fill_group).push_back(trunk_triangles.at(t));

            }

        }

        //calculate the area of fill groups

        float max_area = 0.f;
        int max_ind = 0;
        for( size_t i=0; i<trunk_groups.size(); i++ ){

            float group_area = 0;
            for( size_t t=0; t<trunk_groups.at(i).size(); t++ ){
                float area = trunk_groups.at(i).at(t).area;
                if( area==area ){
                    group_area += area;
                }
            }

            if( group_area>max_area ){
                max_area = group_area;
                max_ind = i;
            }

        }

        reconstructed_trunk_triangles.push_back(trunk_groups.at(max_ind));

    }

    free(hit_vol);

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

void LiDARcloud::syntheticScan( helios::Context* context ){
    syntheticScan( context, 1, 0, false, false );
}

void LiDARcloud::syntheticScan( helios::Context* context, bool scan_grid_only, bool record_misses ){
    syntheticScan( context, 1, 0, scan_grid_only, record_misses );
}

void LiDARcloud::syntheticScan( helios::Context* context, int rays_per_pulse, float pulse_distance_threshold ){
    syntheticScan( context, rays_per_pulse, pulse_distance_threshold, false, false );
}

void LiDARcloud::syntheticScan( helios::Context* context, int rays_per_pulse, float pulse_distance_threshold, bool scan_grid_only, bool record_misses ){

    int Npulse;
    if( rays_per_pulse<1 ){
        Npulse=1;
    }else{
        Npulse=rays_per_pulse;
    }

    if( printmessages ){
        if( Npulse>1 ){
            std::cout << "Performing full-waveform synthetic LiDAR scan..." << std::endl;
        }else{
            std::cout << "Performing discrete return synthetic LiDAR scan..." << std::endl;
        }
    }

    if(getScanCount() ==0 ){
        std::cout << "WARNING (syntheticScan): No scans added to the point cloud. Exiting.." << std::endl;
        return;
    }

    float miss_distance = 1001.0;  //arbitrary distance from scanner for 'miss' points

    float3 bb_center;
    float3 bb_size;

    if(scan_grid_only == false){

        //Determine bounding box for Context geometry
        helios::vec2 xbounds, ybounds, zbounds;
        context->getDomainBoundingBox(xbounds,ybounds,zbounds);
        bb_center = make_float3(xbounds.x+0.5*(xbounds.y-xbounds.x),ybounds.x+0.5*(ybounds.y-ybounds.x),zbounds.x+0.5*(zbounds.y-zbounds.x));
        bb_size = make_float3(xbounds.y-xbounds.x,ybounds.y-ybounds.x,zbounds.y-zbounds.x);

    }else{

        // Determine bounding box for voxels instead of whole domain
        helios::vec3 boxmin, boxmax;
        getGridBoundingBox(boxmin, boxmax);
        bb_center = make_float3(boxmin.x + 0.5*(boxmax.x-boxmin.x),boxmin.y + 0.5*(boxmax.y-boxmin.y),boxmin.z + 0.5*(boxmax.z-boxmin.z) );
        bb_size = make_float3(boxmax.x-boxmin.x, boxmax.y-boxmin.y, boxmax.z-boxmin.z );

    }

    //get geometry information and copy to GPU

    size_t c=0;

    std::map<std::string,int> textures;
    std::map<std::string,int2> texture_size;
    std::map<std::string,std::vector<std::vector<bool> > > texture_data;
    int tID = 0;

    std::vector<uint> UUIDs_all = context->getAllUUIDs();

    std::vector<uint> ID_mapping;

    //----- PATCHES ----- //

    //figure out how many patches
    size_t Npatches = 0;
    for( int p=0; p<UUIDs_all.size(); p++ ){
        if( context->getPrimitiveType(UUIDs_all.at(p)) == helios::PRIMITIVE_TYPE_PATCH ){
            Npatches++;
        }
    }

    ID_mapping.resize(Npatches);

    float3* patch_vertex = (float3*)malloc(4*Npatches * sizeof(float3)); //allocate host memory
    int* patch_textureID = (int*)malloc(Npatches * sizeof(int)); //allocate host memory
    float2* patch_uv = (float2*)malloc(2*Npatches * sizeof(float2)); //allocate host memory

    c=0;
    for( int p=0; p<UUIDs_all.size(); p++ ){
        uint UUID = UUIDs_all.at(p);
        if( context->getPrimitiveType(UUID) == helios::PRIMITIVE_TYPE_PATCH ){
            std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUID);
            patch_vertex[4*c] = vec3tofloat3(verts.at(0));
            patch_vertex[4*c+1] = vec3tofloat3(verts.at(1));
            patch_vertex[4*c+2] = vec3tofloat3(verts.at(2));
            patch_vertex[4*c+3] = vec3tofloat3(verts.at(3));

            ID_mapping.at(c) = UUIDs_all.at(p);

            if( !context->getPrimitiveTextureFile(UUID).empty() && context->primitiveTextureHasTransparencyChannel(UUID) ){
                std::string tex = context->getPrimitiveTextureFile(UUID);
                std::map<std::string,int>::iterator it = textures.find(tex);
                if( it != textures.end() ){ //texture already exits
                    patch_textureID[c] = textures.at(tex);
                }else{ //new texture
                    patch_textureID[c] = tID;
                    textures[tex] = tID;
                    helios::int2 tsize = context->getPrimitiveTextureSize(UUID);
                    texture_size[tex] = make_int2(tsize.x,tsize.y);
                    texture_data[tex] = *context->getPrimitiveTextureTransparencyData(UUID);
                    tID++;
                }

                std::vector<helios::vec2> uv = context->getPrimitiveTextureUV(UUID);
                if( uv.size()==4 ){//custom uv coordinates
                    patch_uv[2*c] = vec2tofloat2(uv.at(1));
                    patch_uv[2*c+1] = vec2tofloat2(uv.at(3));
                }else{//default uv coordinates
                    patch_uv[2*c] = make_float2(0,0);
                    patch_uv[2*c+1] = make_float2(1,1);
                }

            }else{
                patch_textureID[c]=-1;
            }

            c++;
        }
    }

    float3* d_patch_vertex;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex,4*Npatches*sizeof(float3)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex, patch_vertex, 4*Npatches*sizeof(float3), cudaMemcpyHostToDevice) );
    int* d_patch_textureID;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_textureID,Npatches*sizeof(int)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_patch_textureID, patch_textureID, Npatches*sizeof(int), cudaMemcpyHostToDevice) );
    float2* d_patch_uv;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_uv, 2*Npatches*sizeof(float2)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_patch_uv, patch_uv, 2*Npatches*sizeof(float2), cudaMemcpyHostToDevice) );

    //----- TRIANGLES ----- //

    //figure out how many triangles
    size_t Ntriangles = 0;
    for( int p=0; p<UUIDs_all.size(); p++ ){
        if( context->getPrimitiveType(UUIDs_all.at(p)) == helios::PRIMITIVE_TYPE_TRIANGLE ){
            Ntriangles++;
        }
    }

    ID_mapping.resize(Npatches+Ntriangles);

    float3* tri_vertex = (float3*)malloc(3*Ntriangles * sizeof(float3)); //allocate host memory
    int* tri_textureID = (int*)malloc(Ntriangles * sizeof(int)); //allocate host memory
    float2* tri_uv = (float2*)malloc(3*Ntriangles * sizeof(float2)); //allocate host memory

    c=0;
    for( int p=0; p<UUIDs_all.size(); p++ ){
        uint UUID = UUIDs_all.at(p);
        if( context->getPrimitiveType(UUID) == helios::PRIMITIVE_TYPE_TRIANGLE ){
            std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUID);
            tri_vertex[3*c] = vec3tofloat3(verts.at(0));
            tri_vertex[3*c+1] = vec3tofloat3(verts.at(1));
            tri_vertex[3*c+2] = vec3tofloat3(verts.at(2));

            ID_mapping.at(Npatches+c) = UUIDs_all.at(p);

            if( !context->getPrimitiveTextureFile(UUID).empty() && context->primitiveTextureHasTransparencyChannel(UUID) ){
                std::string tex = context->getPrimitiveTextureFile(UUID);
                std::map<std::string,int>::iterator it = textures.find(tex);
                if( it != textures.end() ){ //texture already exits
                    tri_textureID[c] = textures.at(tex);
                }else{ //new texture
                    tri_textureID[c] = tID;
                    textures[tex] = tID;
                    helios::int2 tsize = context->getPrimitiveTextureSize(UUID);
                    texture_size[tex] = make_int2(tsize.x,tsize.y);
                    texture_data[tex] = *context->getPrimitiveTextureTransparencyData(UUID);
                    tID++;
                }

                std::vector<helios::vec2> uv = context->getPrimitiveTextureUV(UUID);
                assert( uv.size()==3 );
                tri_uv[3*c] = vec2tofloat2(uv.at(0));
                tri_uv[3*c+1] = vec2tofloat2(uv.at(1));
                tri_uv[3*c+2] = vec2tofloat2(uv.at(2));

            }else{
                tri_textureID[c]=-1;
            }

            c++;
        }
    }

    float3* d_tri_vertex;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_vertex,3*Ntriangles*sizeof(float3)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_tri_vertex, tri_vertex, 3*Ntriangles*sizeof(float3), cudaMemcpyHostToDevice) );
    int* d_tri_textureID;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_textureID, Ntriangles*sizeof(int)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_tri_textureID, tri_textureID, Ntriangles*sizeof(int), cudaMemcpyHostToDevice) );
    float2* d_tri_uv;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_uv,3*Ntriangles*sizeof(float2)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_tri_uv, tri_uv, 3*Ntriangles*sizeof(float2), cudaMemcpyHostToDevice) );

    //transfer texture data to GPU
    const int Ntextures = textures.size();

    int2 masksize_max = make_int2(0,0);
    for( std::map<std::string,int2>::iterator it=texture_size.begin(); it!=texture_size.end(); ++it ){
        if( it->second.x>masksize_max.x ){
            masksize_max.x=it->second.x;
        }
        if( it->second.y>masksize_max.y ){
            masksize_max.y=it->second.y;
        }
    }

    bool* maskdata = (bool*)malloc(Ntextures*masksize_max.x*masksize_max.y * sizeof(bool)); //allocate host memory
    int2* masksize = (int2*)malloc(Ntextures * sizeof(int2)); //allocate host memory

    for( std::map<std::string,int2>::iterator it=texture_size.begin(); it!=texture_size.end(); ++it ){
        std::string texture_file = it->first;

        int ID = textures.at(texture_file);

        masksize[ID] = it->second;

        int ind=0;
        for( int j=0; j<masksize_max.y; j++ ){
            for( int i=0; i<masksize_max.x; i++ ){

                if( i<texture_size.at(texture_file).x && j<texture_size.at(texture_file).y ){
                    maskdata[ID*masksize_max.x*masksize_max.y+ind] = texture_data.at(texture_file).at(j).at(i);
                }else{
                    maskdata[ID*masksize_max.x*masksize_max.y+ind] = false;
                }
                ind++;
            }
        }
    }

    bool* d_maskdata;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_maskdata, Ntextures*masksize_max.x*masksize_max.y * sizeof(bool)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_maskdata, maskdata, Ntextures*masksize_max.x*masksize_max.y * sizeof(bool), cudaMemcpyHostToDevice) );
    int2* d_masksize;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_masksize, Ntextures * sizeof(int2)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_masksize, masksize, Ntextures * sizeof(int2), cudaMemcpyHostToDevice) );

    for( int s=0; s< getScanCount(); s++ ){

        float3 scan_origin = vec3tofloat3(getScanOrigin(s));

        int Ntheta = getScanSizeTheta(s);
        int Nphi = getScanSizePhi(s);

        helios::vec2 thetarange = getScanRangeTheta(s);
        float thetamin = thetarange.x;
        float thetamax = thetarange.y;
        helios::vec2 phirange = getScanRangePhi(s);
        float phimin = phirange.x;
        float phimax = phirange.y;

        std::vector<std::string> column_format = getScanColumnFormat(s);

        std::vector<helios::vec3> raydir;
        raydir.resize(Ntheta*Nphi);

        for (uint j=0; j<Nphi; j++ ){
            float phi = phimin+float(j)*(phimax-phimin)/float(Nphi);
            for (uint i=0; i<Ntheta; i++ ){
                float theta_z = thetamin+float(i)*(thetamax-thetamin)/float(Ntheta);
                float theta_elev = 0.5f*M_PI-theta_z;
                helios::vec3 dir = sphere2cart(helios::make_SphericalCoord(1.f,theta_elev,phi));
                raydir.at(Ntheta*j+i) = dir;
            }
        }

        size_t N = Ntheta*Nphi;

        float3* d_hit_xyz;
        CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_xyz,N*sizeof(float3)) ); //allocate device memory

        //copy scan data into the host buffer
        float3* hit_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
        for( std::size_t r=0; r<N; r++ ){
            hit_xyz[r] = scan_origin+vec3tofloat3(raydir.at(r)*10000.f);
        }

        //copy from host to device memory
        CUDA_CHECK_ERROR( cudaMemcpy(d_hit_xyz, hit_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );

        uint* bb_hit = (uint*)malloc(N * sizeof(uint)); //allocate host memory
        uint* d_bb_hit;
        CUDA_CHECK_ERROR( cudaMalloc((void**)&d_bb_hit,N*sizeof(uint)) ); //allocate device memory
        CUDA_CHECK_ERROR( cudaMemset( d_bb_hit, 0, N*sizeof(uint)) ); //initialize to zero, set equal to 1 if the ray is found to intersect bounding box

        //Launch kernel to determine which rays intersect bounding box
        uint3 dimBlock = make_uint3( 512, 1, 1 );
        uint3 dimGrid = make_uint3( ceil(float(N)/float(dimBlock.x)), 1, 1 );
        intersectBoundingBox<<< dimGrid, dimBlock >>>( N, scan_origin, d_hit_xyz, bb_center, bb_size, d_bb_hit );

        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

        //copy hit flag back to host
        CUDA_CHECK_ERROR( cudaMemcpy(bb_hit, d_bb_hit, N*sizeof(uint), cudaMemcpyDeviceToHost) );

        CUDA_CHECK_ERROR( cudaFree(d_hit_xyz) );
        CUDA_CHECK_ERROR( cudaFree(d_bb_hit) );

        //determine how many rays hit the bounding box
        N = 0;
        float hit_out = 0;
        for( int i=0; i<Ntheta*Nphi; i++ ){
            if( bb_hit[i]==1 ){
                N++;
                helios::SphericalCoord dir = cart2sphere(raydir[i]);
                hit_out += sin(dir.zenith);
            }
        }

        if( N==0 ){
            std::cout << "WARNING: Synthetic rays did not hit any primitives." << std::endl;
            return;
        }

        //make a new array of ray directions for rays that hit bounding box
        float3* direction = (float3*)malloc(N * sizeof(float3)); //allocate host memory

        std::vector<helios::int2> pulse_scangrid_ij(N);

        int count=0;
        for( int i=0; i<Ntheta*Nphi; i++ ){
            if( bb_hit[i]==1 ){

                direction[count] = vec3tofloat3(raydir.at(i));

                int jj = floor(i/Ntheta);
                int ii = i-jj*Ntheta;
                pulse_scangrid_ij[count] = helios::make_int2(ii,jj);

                count++;
            }else if( record_misses && !scan_grid_only ){

                std::map<std::string,double> data;
                data["target_index"] = 0;
                data["target_count"] = 1;
                data["deviation"] = 0;
                data["timestamp"] = i;
                data["intensity"] = 0;
                data["distance"] = miss_distance;
                data["nRaysHit"] = rays_per_pulse;

                helios::vec3 miss_xyz = getScanOrigin(s) + miss_distance*raydir.at(i);

                addHitPoint( s, miss_xyz, helios::cart2sphere(raydir.at(i)), helios::RGB::black, data );

            }
        }
        free(bb_hit);

        float3* d_raydir;
        CUDA_CHECK_ERROR( cudaMalloc((void**)&d_raydir,N*sizeof(float3)) ); //allocate device memory
        CUDA_CHECK_ERROR( cudaMemcpy(d_raydir, direction, N*sizeof(float3), cudaMemcpyHostToDevice) );

        //Distance to intersection
        float* d_hit_t;
        CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_t,N*Npulse*sizeof(float)) ); //allocate device memory
        float* hit_t = (float*)malloc(N*Npulse * sizeof(float)); //allocate host memory
        for( int i=0; i<N*Npulse; i++ ){
            hit_t[i] = miss_distance;
        }
        CUDA_CHECK_ERROR( cudaMemcpy(d_hit_t, hit_t, N*Npulse*sizeof(float), cudaMemcpyHostToDevice) );

        //Dot product of primitive normal and ray direction (for calculating intensity)
        float* d_hit_fnorm;
        CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_fnorm,N*Npulse*sizeof(float)) ); //allocate device memory
        float* hit_fnorm = (float*)malloc(N*Npulse * sizeof(float)); //allocate host memory
        for( int i=0; i<N*Npulse; i++ ){
            hit_fnorm[i] = 1e6;
        }
        CUDA_CHECK_ERROR( cudaMemcpy(d_hit_fnorm, hit_fnorm, N*Npulse*sizeof(float), cudaMemcpyHostToDevice) );

        //ID of intersected primitive
        int* d_hit_ID;
        CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_ID,N*Npulse*sizeof(int)) ); //allocate device memory
        int* hit_ID = (int*)malloc(N*Npulse * sizeof(int)); //allocate host memory
        for( int i=0; i<N*Npulse; i++ ){
            hit_ID[i] = 999999999;
        }
        CUDA_CHECK_ERROR( cudaMemcpy(d_hit_ID, hit_ID, N*Npulse*sizeof(float), cudaMemcpyHostToDevice) );

        float exit_diameter = getScanBeamExitDiameter(s);
        float beam_divergence = getScanBeamDivergence(s);

        if( Npulse>1 ){
            dimBlock = make_uint3( 128, 4, 1 );
        }else{
            dimBlock = make_uint3( 512, 1, 1 );
        }
        dimGrid = make_uint3( ceil(float(N)/float(dimBlock.x)), ceil(float(Npulse)/float(dimBlock.y)), 1 );

        //---- patch kernel ----//
        intersectPatches<<< dimGrid, dimBlock >>>( N, Npulse, scan_origin, d_raydir, exit_diameter, beam_divergence, Npatches, d_patch_vertex, d_patch_textureID, Ntextures, d_masksize, masksize_max, d_maskdata, d_patch_uv, d_hit_t, d_hit_fnorm, d_hit_ID );

        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

        //---- triangle kernel ----//
        intersectTriangles<<< dimGrid, dimBlock >>>( N, Npulse, scan_origin, d_raydir, exit_diameter, beam_divergence, Ntriangles, Npatches, d_tri_vertex, d_tri_textureID, Ntextures, d_masksize, masksize_max, d_maskdata, d_tri_uv, d_hit_t, d_hit_fnorm, d_hit_ID );

        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

        //copy back
        CUDA_CHECK_ERROR( cudaMemcpy(hit_t, d_hit_t, N*Npulse*sizeof(float), cudaMemcpyDeviceToHost) );
        CUDA_CHECK_ERROR( cudaMemcpy(hit_fnorm, d_hit_fnorm, N*Npulse*sizeof(float), cudaMemcpyDeviceToHost) );
        CUDA_CHECK_ERROR( cudaMemcpy(hit_ID, d_hit_ID, N*Npulse*sizeof(int), cudaMemcpyDeviceToHost) );

        size_t Nhits = 0;
        //looping over beams
        for( size_t r=0; r<N; r++ ){

            std::vector<std::vector<float> > t_pulse;
            std::vector<std::vector<float> > t_hit;

            //looping over rays in each beam
            for( size_t p=0; p<Npulse; p++ ){

                float t = hit_t[r*Npulse+p];  //distance to hit (misses t=1001.0)
                float i = hit_fnorm[r*Npulse+p]; //dot product between beam direction and primitive normal
                float ID = float(hit_ID[r*Npulse+p]);   //ID of intersected primitive

                if( record_misses || (!record_misses && t<miss_distance ) ){
                    std::vector<float> v{t,i,ID};
                    t_pulse.push_back(v);
                }

            }

            if( t_pulse.size()==1 ){ //this is discrete-return data, or we only had one hit for this pulse

                float distance = t_pulse.front().at(0);
                float intensity = t_pulse.front().at(1);
                if( distance>=0.98f*miss_distance ){
                    intensity=0.0;
                }
                float nPulseHit = 1;
                float IDmap = t_pulse.front().at(2);

                std::vector<float> v{distance,intensity,nPulseHit,IDmap};
                t_hit.push_back( v );

            }else if( t_pulse.size()>1 ){ //more than one hit for this pulse

                std::sort( t_pulse.begin(), t_pulse.end(), LIDAR_CUDA::sortcol0 );

                float t0=t_pulse.at(0).at(0);
                float d=t_pulse.at(0).at(0);
                float f=t_pulse.at(0).at(1);
                int count=1;

                //loop over rays in each beam and group into hit points
                for( size_t hit=1; hit<=t_pulse.size(); hit++ ){

                    // if the end has been reached, output the last hitpoint
                    if( hit == t_pulse.size()){

                        float distance = d/float(count);
                        float intensity = f/float(Npulse);
                        if( distance>=0.98f*miss_distance ){
                            intensity=0;
                        }
                        float nPulseHit = float(count);
                        float IDmap = t_pulse.at(hit-1).at(2);

                        //test for full misses and set those to have intensity = 1
                        if(nPulseHit == Npulse & distance >= 0.98f*miss_distance)
                        {
                            std::vector<float> v{distance, 1.0, nPulseHit, IDmap}; //included the ray count here
                            //Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back( v );
                        }else{
                            std::vector<float> v{distance, intensity, nPulseHit, IDmap}; //included the ray count here
                            //Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back( v );
                        }

                        // else if the current ray is more than the pulse threshold distance from t0,  it is part of the next hitpoint so output the previous hitpoint and reset
                    }else if( t_pulse.at(hit).at(0)-t0>pulse_distance_threshold ){

                        float distance = d/float(count);
                        float intensity = f/float(Npulse);
                        if( distance>=0.98f*miss_distance ){
                            intensity=0;
                        }
                        float nPulseHit = float(count);
                        float IDmap = t_pulse.at(hit-1).at(2);

                        //test for full misses and set those to have intensity = 1
                        if(nPulseHit == Npulse & distance >= 0.98f*miss_distance)
                        {
                            std::vector<float> v{distance, 1.0, nPulseHit, IDmap}; //included the ray count here
                            //Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back( v );
                        }else{
                            std::vector<float> v{distance, intensity, nPulseHit, IDmap}; //included the ray count here
                            //Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back( v );
                        }

                        count=1;
                        d=t_pulse.at(hit).at(0);
                        t0=t_pulse.at(hit).at(0);
                        f=t_pulse.at(hit).at(1);

                        // or else the current ray is less than pulse threshold and is part of current hitpoint; add it to the current hit point and continue on
                    }else{

                        count++;
                        d+=t_pulse.at(hit).at(0);
                        f+=t_pulse.at(hit).at(1);

                    }

                }

            }

            float average=0;
            for( size_t hit=0; hit<t_hit.size(); hit++ ){
                average+=t_hit.at(hit).at(0)/float(t_hit.size());
            }

            for( size_t hit=0; hit<t_hit.size(); hit++ ){

                std::map<std::string,double> data;
                data["target_index"] = hit;
                data["target_count"] = t_hit.size();
                data["deviation"] = fabs(t_hit.at(hit).at(0)-average);
                data["timestamp"] = pulse_scangrid_ij.at(r).y*Ntheta+pulse_scangrid_ij.at(r).x;
                data["intensity"] = t_hit.at(hit).at(1);
                data["distance"] = t_hit.at(hit).at(0);
                data["nRaysHit"] = t_hit.at(hit).at(2);

                float UUID = t_hit.at(hit).at(3);
                if( UUID>=0 && UUID<ID_mapping.size() ){
                    UUID = ID_mapping.at(int(t_hit.at(hit).at(3)));
                }

                helios::RGBcolor color = helios::RGB::red;

                if( UUID>=0 && context->doesPrimitiveExist(uint(UUID)) ){

                    color = context->getPrimitiveColor(uint(UUID));

                    if( context->doesPrimitiveDataExist(uint(UUID),"object_label") && context->getPrimitiveDataType(uint(UUID),"object_label")==helios::HELIOS_TYPE_INT ){
                        int label;
                        context->getPrimitiveData(uint(UUID),"object_label",label);
                        data["object_label"] = double(label);
                    }

                    if( context->doesPrimitiveDataExist(uint(UUID),"reflectivity_lidar") && context->getPrimitiveDataType(uint(UUID),"reflectivity_lidar")==helios::HELIOS_TYPE_FLOAT ){
                        float rho;
                        context->getPrimitiveData(uint(UUID),"reflectivity_lidar",rho);
                        data.at("intensity")*=rho;
                    }

                }

                helios::vec3 dir = helios::make_vec3(direction[r].x,direction[r].y,direction[r].z);
                helios::vec3 origin = helios::make_vec3(scan_origin.x,scan_origin.y,scan_origin.z);
                helios::vec3 p = origin+dir*t_hit.at(hit).at(0);
                addHitPoint( s, p, helios::cart2sphere(dir), color, data );

                Nhits++;
            }

        }

        CUDA_CHECK_ERROR( cudaFree(d_hit_t) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_fnorm) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_ID) );
        CUDA_CHECK_ERROR( cudaFree(d_raydir) );
        free(hit_xyz);
        free(direction);
        free(hit_t);
        free(hit_fnorm);
        free(hit_ID);

        if( printmessages ){
            std::cout << "Created synthetic scan #" << s << " with " << Nhits << " hit points." << std::endl;
        }

    }

    CUDA_CHECK_ERROR( cudaFree(d_patch_vertex) );
    CUDA_CHECK_ERROR( cudaFree(d_patch_textureID) );
    CUDA_CHECK_ERROR( cudaFree(d_patch_uv) );
    CUDA_CHECK_ERROR( cudaFree(d_tri_vertex) );
    CUDA_CHECK_ERROR( cudaFree(d_tri_textureID) );
    CUDA_CHECK_ERROR( cudaFree(d_tri_uv) );
    CUDA_CHECK_ERROR( cudaFree(d_maskdata) );
    CUDA_CHECK_ERROR( cudaFree(d_masksize) );
    free(patch_vertex);
    free(patch_textureID);
    free(patch_uv);
    free(tri_vertex);
    free(tri_textureID);
    free(tri_uv);
    free(maskdata);
    free(masksize);

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

std::vector<float> LiDARcloud::calculateSyntheticLeafArea( helios::Context* context ){

    float3* d_prim_xyz;

    std::vector<uint> UUIDs_all = context->getAllUUIDs();

    const uint N = UUIDs_all.size();

    float3* prim_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_prim_xyz,N*sizeof(float3)) ); //allocate device memory

    //copy scan data into the host buffer
    for( std::size_t p=0; p<N; p++ ){
        std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUIDs_all.at(p));
        prim_xyz[p] = vec3tofloat3( verts.at(0) );
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_prim_xyz, prim_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );

    // ---- Grid Cells ---- //

    //! Grid cell centers on device (GPU) memory
    float3* d_grid_center;

    //! Grid cell global anchor on device (GPU) memory
    float3* d_grid_anchor;

    //! Grid sizes on device (GPU) memory
    float3* d_grid_size;

    //! Grid rotations on device (GPU) memory
    float* d_grid_rotation;

    const uint Ncells = getGridCellCount();

    float3* center = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_center,Ncells*sizeof(float3)) ); //allocate device memory

    float3* anchor = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_anchor,Ncells*sizeof(float3)) ); //allocate device memory

    float3* size = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_size,Ncells*sizeof(float3)) ); //allocate device memory

    float* rotation = (float*)malloc(Ncells * sizeof(float)); //allocate host memory
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_rotation,Ncells*sizeof(float)) ); //allocate device memory

    //copy grid data into the host buffer
    for( int c=0; c<Ncells; c++ ){
        center[c] = vec3tofloat3(getCellCenter(c));
        anchor[c] = vec3tofloat3(getCellGlobalAnchor(c));
        size[c] = vec3tofloat3(getCellSize(c));
        rotation[c] = getCellRotation(c);
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_center, center, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_anchor, anchor, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_size, size, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_grid_rotation, rotation, Ncells*sizeof(float), cudaMemcpyHostToDevice) );

    free(prim_xyz);
    free(center);
    free(anchor);
    free(size);
    free(rotation);

    // Result buffer
    int* prim_vol = (int*)malloc(N * sizeof(int));
    int* d_prim_vol;
    CUDA_CHECK_ERROR( cudaMalloc(&d_prim_vol,N*sizeof(int)) );

    dim3 dimBlock( 64, 1 );
    dim3 dimGrid( ceil(N/64.f) );
    insideVolume <<< dimGrid, dimBlock >>>( N, d_prim_xyz, getGridCellCount(), d_grid_size, d_grid_center, d_grid_anchor, d_grid_rotation, d_prim_vol );

    CUDA_CHECK_ERROR( cudaPeekAtLastError() );
    CUDA_CHECK_ERROR( cudaDeviceSynchronize() );

    CUDA_CHECK_ERROR( cudaMemcpy(prim_vol, d_prim_vol, N*sizeof(int), cudaMemcpyDeviceToHost) );

    std::vector<float> total_area;
    total_area.resize(Ncells,0.f);
    for( std::size_t p=0; p<N; p++ ){
        if( prim_vol[p]>=0 ){
            uint gridcell = prim_vol[p];
            total_area.at(gridcell) += context->getPrimitiveArea(UUIDs_all.at(p));
            context->setPrimitiveData(UUIDs_all.at(p),"gridCell",helios::HELIOS_TYPE_UINT,1,&gridcell);
        }
    }

    std::vector<float> output_LeafArea;
    output_LeafArea.resize(Ncells,0.f);

    for( int v=0; v<Ncells; v++ ){
        output_LeafArea.at(v) =  total_area.at(v);
        if( context->doesPrimitiveDataExist(UUIDs_all.at(v), "gridCell") ){
            context->setPrimitiveData(UUIDs_all.at(v),"synthetic_leaf_area",total_area.at(v));
        }
    }

    free(prim_vol);

    CUDA_CHECK_ERROR( cudaFree(d_prim_vol) );
    CUDA_CHECK_ERROR( cudaFree(d_prim_xyz) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_center) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_anchor) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_size) );
    CUDA_CHECK_ERROR( cudaFree(d_grid_rotation) );

    return output_LeafArea;

}

__global__ void LIDAR_CUDA::intersectPatches( const size_t Npulse, const int rays_per_pulse, const float3 origin, float3* d_raydir, const float exit_diameter, const float beam_divergence,const int Npatches, float3* d_patch_vertex, int* d_patch_textureID, const int Ntextures, int2* d_masksize, int2 masksize_max, bool* d_maskdata, float2* d_patch_uv, float* d_hit_t, float* d_hit_fnorm, int* d_hit_ID ){

    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    int subpulse = blockIdx.y*blockDim.y+threadIdx.y;

    if( idx>=Npulse || subpulse>=rays_per_pulse ){
        return;
    }

    int i_global = idx*rays_per_pulse+subpulse;

    uint seed = tea<16>(subpulse,i_global);

    float kEpsilon = 1e-6;

    float2 disk_pt = d_sampleDisk( 0.5*exit_diameter, seed );

    float3 raydir = d_sphere2cart( 1.f, 0.5*M_PI-0.5*beam_divergence*rnd(seed), 2.f*M_PI*rnd(seed) );

    float zenith = acos_safe( d_raydir[idx].z/sqrt(d_raydir[idx].x*d_raydir[idx].x+d_raydir[idx].y*d_raydir[idx].y+d_raydir[idx].z*d_raydir[idx].z) );
    float azimuth = atan2_2pi( d_raydir[idx].x, d_raydir[idx].y );
    raydir = d_rotatePoint( raydir, zenith, 0.5*M_PI-azimuth );

    float3 disk_pt3 = d_rotatePoint( make_float3(disk_pt.x,disk_pt.y,0), zenith, azimuth );

    float tmin = d_hit_t[i_global];
    float nmin = d_hit_fnorm[i_global];
    for( int p=0; p<Npatches; p++ ){

        float3 v0 = d_patch_vertex[4*p];
        float3 v1 = d_patch_vertex[4*p+1];
        float3 v2 = d_patch_vertex[4*p+2];
        float3 v3 = d_patch_vertex[4*p+3];

        float3 anchor = v0;
        float3 normal = normalize( cross( v1-v0, v2-v0 ) );

        float3 a = v1-v0;
        float3 b = v3-v0;

        float t = ((anchor - origin - disk_pt3)*normal) / (raydir*normal);

        if( t==t && t>-kEpsilon && t<tmin ){

            float3 point = origin + disk_pt3 + raydir * t;
            float3 d = point - anchor;

            float ddota = d*a;

            if (ddota > -kEpsilon && ddota < a*a ){

                float ddotb = d*b;

                if (ddotb > -kEpsilon && ddotb < b*b ){

                    if( d_patch_textureID[p]<0 ){//no texture mask

                        tmin = t;
                        nmin = fabs(normal*raydir);
                        d_hit_ID[i_global] = p;

                    }else{ //has texture mask

                        float amag = magnitude(a);
                        float bmag = magnitude(b);
                        float2 uv = make_float2( 1.f-ddota/amag/amag, ddotb/bmag/bmag );
                        int ID = d_patch_textureID[p];
                        int2 sz = d_masksize[ID];

                        float2 uvmin = d_patch_uv[2*p];
                        float2 duv;
                        duv.x = d_patch_uv[ 2*p+1 ].x - d_patch_uv[ 2*p ].x;
                        duv.y = d_patch_uv[ 2*p+1 ].y - d_patch_uv[ 2*p ].y;
                        int2 ind = make_int2( roundf(float(sz.x-1)*(uvmin.x+fabs(uv.x)*duv.x)), roundf(float(sz.y-1)*(uvmin.y+fabs(uv.y)*duv.y)) );
                        assert( ind.x>=0 && ind.x<sz.x && ind.y>=0 && ind.y<sz.y );

                        if( d_maskdata[ ID*masksize_max.x*masksize_max.y + ind.y*masksize_max.x + ind.x ] ){

                            tmin = t;
                            nmin = fabs(normal*raydir);
                            d_hit_ID[i_global] = p;

                        }

                    }

                }

            }
        }
    }

    d_hit_t[i_global] = tmin;

    d_hit_fnorm[i_global] = nmin;

}

__global__ void LIDAR_CUDA::intersectTriangles( const size_t Npulse, const int rays_per_pulse, const float3 origin, float3* d_raydir, const float exit_diameter, const float beam_divergence, const int Ntriangles, const int Npatches, float3* d_tri_vertex, int* d_tri_textureID, const int Ntextures, int2* d_masksize, const int2 masksize_max, bool* d_maskdata, float2* d_tri_uv, float* d_hit_t, float* d_hit_fnorm, int* d_hit_ID ){

    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    int subpulse = blockIdx.y*blockDim.y+threadIdx.y;

    if( idx>=Npulse || subpulse>=rays_per_pulse ){
        return;
    }

    int i_global = idx*rays_per_pulse+subpulse;

    uint seed = tea<16>(subpulse,i_global);

    float kEpsilon = 1e-5;

    float2 disk_pt = d_sampleDisk( 0.5*exit_diameter, seed );

    float3 raydir = d_sphere2cart( 1.f, 0.5*M_PI-0.5*beam_divergence*rnd(seed), 2.f*M_PI*rnd(seed) );

    float zenith = acos_safe( d_raydir[idx].z/sqrt(d_raydir[idx].x*d_raydir[idx].x+d_raydir[idx].y*d_raydir[idx].y+d_raydir[idx].z*d_raydir[idx].z) );
    float azimuth = atan2_2pi( d_raydir[idx].x, d_raydir[idx].y );
    raydir = d_rotatePoint( raydir, zenith, 0.5*M_PI-azimuth );

    float3 disk_pt3 = d_rotatePoint( make_float3(disk_pt.x,disk_pt.y,0), zenith, azimuth );

    float tmin = d_hit_t[i_global];
    float nmin = d_hit_fnorm[i_global];
    for( int tri=0; tri<Ntriangles; tri++ ){

        float3 v0 = d_tri_vertex[3*tri];
        float3 v1 = d_tri_vertex[3*tri+1];
        float3 v2 = d_tri_vertex[3*tri+2];

        float a = v0.x - v1.x, b = v0.x - v2.x, c = raydir.x, d = v0.x - origin.x - disk_pt3.x;
        float e = v0.y - v1.y, f = v0.y - v2.y, g = raydir.y, h = v0.y - origin.y - disk_pt3.y;
        float i = v0.z - v1.z, j = v0.z - v2.z, k = raydir.z, l = v0.z - origin.z - disk_pt3.z;

        float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
        float q = g * i - e * k, s = e * j - f * i;

        float inv_denom  = 1.f / (a * m + b * q + c * s);

        float e1 = d * m - b * n - c * p;
        float beta = e1 * inv_denom;

        if (beta >= -kEpsilon ){

            float r = r = e * l - h * i;
            float e2 = a * n + d * q + c * r;
            float gamma = e2 * inv_denom;

            if (gamma > -kEpsilon && beta + gamma < 1.f+kEpsilon ){

                float e3 = a * p - b * r + d * s;
                float t = e3 * inv_denom;

                if( t>kEpsilon && t<tmin ){

                    if( d_tri_textureID[tri]<0 ){//no texture mask
                        tmin = t;
                        float3 normal = normalize( cross( v1-v0, v2-v0 ) );
                        nmin = fabs(normal*raydir);
                        d_hit_ID[i_global] = Npatches+tri;
                    }else{//has texture mask

                        int ID = d_tri_textureID[tri];
                        int2 sz = d_masksize[ID];

                        float2 uv0 = d_tri_uv[tri*3];
                        float2 uv1 = d_tri_uv[tri*3+1];
                        float2 uv2 = d_tri_uv[tri*3+2];

                        float2 uv;
                        uv.x = (uv0.x + beta*(uv1.x-uv0.x) + gamma*(uv2.x-uv0.x));
                        uv.y = (uv0.y + beta*(uv1.y-uv0.y) + gamma*(uv2.y-uv0.y));

                        uint2 ind = make_uint2( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)) );
                        assert( ind.x<sz.x && ind.y<sz.y );

                        if( d_maskdata[ ID*masksize_max.x*masksize_max.y + ind.y*masksize_max.x + ind.x ] ){
                            tmin = t;
                            float3 normal = normalize( cross( v1-v0, v2-v0 ) );
                            nmin = fabs(normal*raydir);
                            d_hit_ID[i_global] = Npatches+tri;
                        }


                    }
                }

            }
        }
    }

    d_hit_t[i_global] = tmin;

    d_hit_fnorm[i_global] = nmin;

}



void LiDARcloud::calculateLeafAreaGPU_equal_weighting(bool beamoutput, bool fillAnalytic  ){

    // calculates LAD using the equal weighting method
    // writes all voxel level variables to a file
    // unlike previous calculateLeafAreaGPU versions, this one does not set the cell leaf area variable and does not filter based on minVoxelHits argument

    if( printmessages ){
        std::cout << "Calculating leaf area..." << std::endl;
    }

    if( !triangulationcomputed ){
        helios::helios_runtime_error("ERROR (LiDARcloud::calculateLeafAreaGPU): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints().");
    }

    if( !hitgridcellcomputed ){
        calculateHitGridCellGPU();
    }

    const uint Nscans = getScanCount();
    const uint Ncells = getGridCellCount();

    //variable aggregates over all scans where we just keep tacking hits on the end for all scans
    std::vector<std::vector<float> > dr_agg; //dr is path length through grid cell
    dr_agg.resize(Ncells);
    std::vector<float> G_agg; //G is dot product between ray direction and triangle normal that was hit (only relevant for hits)
    G_agg.resize(Ncells,0);
    std::vector<float> hit_before_agg; //hit_before corresponds to scan points that hit something before encountering a particular grid cell
    hit_before_agg.resize(Ncells,0);
    std::vector<float> hit_after_agg; //hit_after corresponds to scan points that hit something after encountering a particular grid cell (including something inside that cell)
    hit_after_agg.resize(Ncells,0);
    std::vector<float> hit_inside_agg; //hit_inside corresponds to scan points that hit something within a particular grid cell.
    hit_inside_agg.resize(Ncells,0);

    //average G(theta)
    std::vector<float> Gtheta_bar;
    Gtheta_bar.resize(Ncells,0.f);

    // nested vectors to save P calculation terms
    // cell, scan, variable, value
    std::vector<std::vector<float>>  P_equal_numerator_array(Ncells);
    std::vector<std::vector<float>>  P_equal_denominator_array(Ncells);

    std::vector<std::vector<uint>>  voxel_beam_count_array(Ncells);
    std::vector<std::vector<float>>  dr_array(Ncells);


    // r is the index of all hitpoints (all scans)
    // it is saved for the current scan in this_scan_index
    // i is the index of this_scan_index
    // so when accessing the things using getHitData(), getHitXYZ(), etc,
    // for only one of the scans (this_scan),
    // need to use getHitData(this_scan_index[i]) not getHitData(i)

    ////////////// We are going to perform all calculations on a scan-by-scan basis: loop through each scan
    for( uint s=0; s<Nscans; s++ ){

        if( printmessages ){
            std::cout << "------------------------SCAN = " << s << std::endl;
        }

        //only work with hitpoints associated with current scan
        std::vector<helios::vec3> this_scan_xyz;
        std::vector<uint> this_scan_index;
        for( size_t r=0; r< getHitCount(); r++ ){
            if( getHitScanID(r)==s ){
                this_scan_xyz.push_back( getHitXYZ(r) );
                this_scan_index.push_back(r);
            }
        }
        // size of the array of all the hits in this scan
        size_t Nhits = this_scan_xyz.size();

        if( printmessages ){
            std::cout << "Number of hitpoints associated with this scan = Nhits = " << Nhits << std::endl;
        }

        // count the number of beams (i.e. don't count multiple hits per beam)
        float previous_time = -1.f;
        uint Nbeams = 0;
        for(uint i=0; i < Nhits; i++)
        {
            float current_time = getHitData(this_scan_index[i], "timestamp");
            if(current_time != previous_time){
                Nbeams ++;
                previous_time = current_time;
            }
        }

        if( printmessages ){
            std::cout << "Number of beams associated with this scan = Nbeams = " << Nbeams << std::endl;
        }

        // create a vector each element of which represents a unique beam and is another vector of the hit point indices of this_scan_xyz that is associated with this beam
        float previous_beam = getHitData(this_scan_index[0], "timestamp");
        uint beam_ID = 0;
        std::vector<std::vector<uint>> beam_array(Nbeams);
        for(uint i=0; i < Nhits; i++)
        {
            float current_beam = getHitData(this_scan_index[i], "timestamp");

            if(current_beam == previous_beam)
            {
                beam_array.at(beam_ID).push_back(i); //ERK
            }else{
                beam_ID ++;
                beam_array.at(beam_ID).push_back(i); //ERK
                previous_beam = current_beam;
            }

        }

        // set up GPU things for current scan
        const float3 origin = vec3tofloat3(getScanOrigin(s));
        float3* scan_xyz = (float3*)malloc( Nhits*sizeof(float3) );
        float* scan_weight = (float*)malloc( Nhits*sizeof(float) );

        for( size_t i=0; i<Nhits; i++ ){
            scan_xyz[i] = vec3tofloat3(this_scan_xyz.at(i));
            scan_weight[i] = 1.f;
        }

        float* hit_before = (float*)malloc( sizeof(float));
        float* hit_after = (float*)malloc( sizeof(float));
        float* d_hit_before;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_before, sizeof(float)) );
        float* d_hit_after;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_after, sizeof(float)) );

        float3* d_scan_xyz;
        CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nhits*sizeof(float3)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

        float* d_scan_weight;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_scan_weight, Nhits*sizeof(float)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_weight, scan_weight, Nhits*sizeof(float), cudaMemcpyHostToDevice) );

        float* dr = (float*)malloc( Nhits*sizeof(float));
        float* d_dr;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr, Nhits*sizeof(float)) );

        uint* hit_location = (uint*)malloc( Nhits*sizeof(uint));
        uint* d_hit_location;
        CUDA_CHECK_ERROR( cudaMalloc((uint**)&d_hit_location, Nhits*sizeof(uint)) );


        /////////////////////////////////////////////////////////////////////
        // now loop through each voxel
        for( uint c=0; c<Ncells; c++ ){

            if( printmessages ){
                std::cout << "----CELL = " << c << std::endl;
            }

            std::ofstream file_beam;
            if( beamoutput){
                // set up header of file that outputs one row for each beam in the current scan that interacts with the current voxel

                file_beam.open("../beamoutput/beam_data_s_" + std::to_string(s) + "_c_" + std::to_string(c) + ".txt");
                file_beam << "scan, cell, beam, E_before, E_inside, E_after, E_miss, sin_theta, dr, last_dr"  << std::endl;
            }

            //load the attributes of the grid cell
            float3 center = vec3tofloat3(getCellCenter(c));
            float3 anchor = vec3tofloat3(getCellGlobalAnchor(c));
            float3 size = vec3tofloat3(getCellSize(c));
            float rotation = getCellRotation(c);

            CUDA_CHECK_ERROR( cudaMemset( d_hit_location, 0, Nhits*sizeof(uint)) );
            CUDA_CHECK_ERROR( cudaMemset( d_dr, 0, Nhits*sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_before, 0, sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_after, 0, sizeof(float)) );

            uint3 dimBlock = make_uint3( min(size_t(512),Nhits), 1, 1 );
            uint3 dimGrid = make_uint3( ceil(float(Nhits)/dimBlock.x), 1, 1  );

            if( dimBlock.x==0 && dimGrid.x==0 ){
                continue;
            }

            float scanner_range = 5000.0;
            intersectGridcell_synthetic <<< dimGrid, dimBlock >>>( Nhits, origin, d_scan_xyz, d_scan_weight, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after, d_hit_location, scanner_range );

            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

            //copy results back to host
            CUDA_CHECK_ERROR( cudaMemcpy( hit_before, d_hit_before, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_after, d_hit_after, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( dr, d_dr, Nhits*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_location, d_hit_location, Nhits*sizeof(uint), cudaMemcpyDeviceToHost));

            float P_equal_numerator = 0;
            float P_equal_denominator = 0;
            uint voxel_beam_count = 0;

            if( printmessages ){
                std::cout << "Nbeams = " << Nbeams << std::endl;
            }
            ////////////////////////// loop through each beam associated with the current scan
            for(int k = 0; k < Nbeams; k++){

                float E_before = 0;
                float E_inside = 0;
                float E_after = 0;
                float E_miss = 0;

                float sin_theta;
                // float W = 0;
                float drr = 0;
                float last_drr = 0;

                ////////// loop through the hitpoints in the current beam to get number of rays in each location
                for(int j = 0; j < beam_array.at(k).size(); j++){

                    // pull out the index of the current scan's current beam's current hit (which is used to access the overall hit index r through this_scan_index[i])
                    uint i = beam_array.at(k).at(j);

                    helios::vec3 direction = getHitXYZ(this_scan_index[i])-getScanOrigin(getHitScanID(this_scan_index[i]));
                    direction.normalize();
                    sin_theta = sin(acos_safe(direction.z));

                    last_drr = dr[i];
                    drr += dr[i];

                    if(hit_location[i] == 1){
                        E_before ++;
                    }else if(hit_location[i] == 2){
                        E_inside ++;
                    }else if(hit_location[i] == 3){
                        E_after ++;
                        //   if(getHitData(this_scan_index[i], "target_index") == 0){ // if this is the first hitpoint for this beam,
                        //      W = 1.0;
                        //    }
                    }else if(hit_location[i] == 4){
                        E_miss ++;
                    } // or this hitpoint / beam did not intersect the voxel and should not be added to the total number of beams for this voxel

                } // end of loop through each hit in the current beam

                // calculate the average dr across all hitpoints for the beam (they should all be the same anyway... but just in case)
                float drrx = drr / float(beam_array.at(k).size());

                // if the path length is greater than 0 (it passes throught the current voxel), save it as an element of
                if( drrx > 0.f )
                {
                    dr_array.at(c).push_back(drrx);
                }

                if( beamoutput ){
                    // output info about the current beam to file
                    file_beam << s << "," << c << "," << k << "," << E_before << "," << E_inside << "," << E_after << "," << E_miss << "," << sin_theta << "," << drrx << "," << last_drr << std::endl;
                }

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // calculate terms equal weighting P method

                // P_equal
                if(E_inside != 0 || E_after != 0){ // only count for P_equal if some hit points were inside or after the voxel
                    P_equal_numerator += (E_after / (E_before + E_inside + E_after))*sin_theta;
                    P_equal_denominator += ((E_inside + E_after) / (E_before + E_inside + E_after))*sin_theta;
                    voxel_beam_count++;
                }else if(E_inside == 0 && E_after == 0 && E_before == 0 && E_miss != 0){ // also count this beam if there is only a "hitpoint" that missed (far after the voxel)
                    P_equal_numerator += 1*sin_theta;
                    P_equal_denominator += 1*sin_theta;
                    voxel_beam_count++;
                }


            }// end of loop through all beams for the current cell

            // save results for the current cell to the arrays
            P_equal_numerator_array.at(c).push_back(P_equal_numerator);
            P_equal_denominator_array.at(c).push_back(P_equal_denominator);
            voxel_beam_count_array.at(c).push_back(voxel_beam_count);
            hit_before_agg.at(c) += *hit_before;
            hit_after_agg.at(c) += *hit_after;

            for( size_t i=0; i<Nhits; i++ ){
                if( dr[i]>0.f ){
                    dr_agg.at(c).push_back(dr[i]);
                }
            }

            file_beam.close();
        } // end of loop through all cells

        free( scan_xyz );
        free( scan_weight );
        free( dr );
        free( hit_before );
        free( hit_after );
        CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
        CUDA_CHECK_ERROR( cudaFree(d_scan_weight) );
        CUDA_CHECK_ERROR( cudaFree(d_dr) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_before) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_after) );

    }//end scan loop

    std::vector<float> P_equal(Ncells);
    std::vector<uint> voxel_beam_count_tot(Ncells);
    std::vector<float> dr_bar(Ncells);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // now aggregate over all scans
    for( uint c=0; c<Ncells; c++ ){

        float P_equal_numerator_array_agg = 0;
        float P_equal_denominator_array_agg = 0;
        uint voxel_beam_count_array_agg=0;

        for( uint s=0; s<Nscans; s++ ){

            P_equal_numerator_array_agg += P_equal_numerator_array.at(c).at(s);
            P_equal_denominator_array_agg += P_equal_denominator_array.at(c).at(s);
            voxel_beam_count_array_agg += voxel_beam_count_array.at(c).at(s);
        }

        P_equal[c] = P_equal_numerator_array_agg / P_equal_denominator_array_agg ;
        voxel_beam_count_tot[c] = voxel_beam_count_array_agg;

        if( printmessages ){
            std::cout << "Cell " << c << ", voxel_beam_count = " << voxel_beam_count_tot[c]  << std::endl;
            std::cout << "Cell " << c << ", P_equal = " << P_equal[c] << std::endl;
        }

        float mean_dr = 0;
        for(int j = 0; j < dr_array.at(c).size(); j++)
        {
            mean_dr += dr_array.at(c).at(j);
        }
        dr_bar[c] = mean_dr/ float(dr_array.at(c).size());
        if( printmessages ){
            std::cout << "Cell " << c << ", dr_bar = " <<  dr_bar[c] << std::endl;
        }
    } //end of loop through cells to aggregate all scans


    /// old code in original calculateLeafAreaGPU_testing() ... do we need to keep this??
    //----------- Calculate number of hits in voxels -------------- //

    //figure out hits for all scans
    for( size_t r=0; r< getHitCount(); r++ ){
        if( getHitGridCell(r)>=0 ){
            helios::vec3 direction = getHitXYZ(r)-getScanOrigin(getHitScanID(r));
            direction.normalize();
            hit_inside_agg.at(getHitGridCell(r)) += sin(acos_safe(direction.z));
        }
    }

    //---------------------- Calculate G(theta) from triangulation --------------------------//

    std::vector<float> Gtheta;
    Gtheta.resize(Ncells,0.f);

    const size_t Ntri = getTriangleCount();

    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells,0);

    std::vector<float> area_sin_sum;
    area_sin_sum.resize(Ncells,0.f);

    for( size_t t=0; t<Ntri; t++ ){

        Triangulation tri = getTriangle(t);
        int cell = tri.gridcell;

        if( cell>=0 && cell<Ncells ){ //triangle is inside a grid cell

            helios::vec3 t0 = tri.vertex0;
            helios::vec3 t1 = tri.vertex1;
            helios::vec3 t2 = tri.vertex2;

            helios::vec3 v0 = t1-t0;
            helios::vec3 v1 = t2-t0;
            helios::vec3 v2 = t2-t1;

            float L0 = v0.magnitude();
            float L1 = v1.magnitude();
            float L2 = v2.magnitude();

            float S = 0.5f*(L0+L1+L2);
            float area = sqrt( S*(S-L0)*(S-L1)*(S-L2) );

            helios::vec3 normal = cross( v0, v2 );
            normal.normalize();

            helios::vec3 raydir = t0-getScanOrigin( tri.scanID );
            raydir.normalize();

            float theta = fabs(acos_safe(raydir.z));

            if( area==area ){ //in rare cases you can get area=NaN

                Gtheta.at(cell) += fabs(normal*raydir)*area*fabs(sin(theta));
                area_sin_sum.at(cell) += area*fabs(sin(theta));
                cell_tri_count.at(cell) += 1;

            }

        }
    }

    for( uint c=0; c<Ncells; c++ ){
        if( cell_tri_count[c]>0 ){
            //Gtheta[c] *= float(cell_tri_count[c])/(area_sum[c]*sin_sum[c]);
            //Gtheta[c] *= float(cell_tri_count[c])/(area_sin_sum[c]);
            Gtheta[c] *= 1.0/(area_sin_sum[c]);

            Gtheta_bar[c] += Gtheta[c]/float(Nscans);
            std::cout << "Cell " << c << " Gtheta_bar = " <<  Gtheta_bar[c] << std::endl;
        }
    }


    //------------------ Perform inversions to get LAD using the different methods for P --------------------//

    if( printmessages ){
        std::cout << "Inverting to find LAD..." << std::flush;
    }

    // perform LAD inversion using triangulation-estimated Gtheta
    std::vector<float> LAD_equal = LAD_inversion(P_equal, Gtheta, dr_array, fillAnalytic);

    if( printmessages ){
        std::cout << "finished LAD inversions" << std::endl;
    }

    // output the voxel level variables
    std::ofstream file_output;
    file_output.open("../voxeloutput/voxeloutput.txt");
    file_output << "cell, grid_center_x, grid_center_y, grid_center_z, grid_size_x, grid_size_y, grid_size_z, Nbeams, dr_bar, G, P_equal, LAD_equal"  << std::endl;

    for(uint c=0; c<Ncells; c++)
    {
        helios::vec3 grid_size = getCellSize(c);
        helios::vec3 grid_center = getCellCenter(c);

        setCellLeafArea(LAD_equal[c]*grid_size.x*grid_size.y*grid_size.z,c);

        setCellGtheta( Gtheta[c], c );

        file_output << c << "," << grid_center.x << "," << grid_center.y << "," << grid_center.z << "," << grid_size.x << "," << grid_size.y << "," << grid_size.z << "," << voxel_beam_count_tot.at(c) << "," << dr_bar[c] << "," << Gtheta[c] << "," << P_equal[c] << "," << LAD_equal[c] << std::endl;
    }
    file_output.close();

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}


void LiDARcloud::calculateLeafAreaGPU_equal_weighting(bool beamoutput, bool fillAnalytic, std::vector<float> constant_G){

    // calculates LAD using the equal weighting method 
    // writes all voxel level variables to a file
    // unlike previous calculateLeafAreaGPU versions, this one does not set the cell leaf area variable and does not filter based on minVoxelHits argument 

    if( printmessages ){
        std::cout << "Calculating leaf area..." << std::endl;
    }

    if( !triangulationcomputed ){
        helios::helios_runtime_error("ERROR (LiDARcloud::calculateLeafAreaGPU): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints().");
    }

    if( !hitgridcellcomputed ){
        calculateHitGridCellGPU();
    }

    const uint Nscans = getScanCount();
    const uint Ncells = getGridCellCount();

    //variable aggregates over all scans where we just keep tacking hits on the end for all scans
    std::vector<std::vector<float> > dr_agg; //dr is path length through grid cell
    dr_agg.resize(Ncells);
    std::vector<float> G_agg; //G is dot product between ray direction and triangle normal that was hit (only relevant for hits)
    G_agg.resize(Ncells,0);
    std::vector<float> hit_before_agg; //hit_before corresponds to scan points that hit something before encountering a particular grid cell
    hit_before_agg.resize(Ncells,0);
    std::vector<float> hit_after_agg; //hit_after corresponds to scan points that hit something after encountering a particular grid cell (including something inside that cell)
    hit_after_agg.resize(Ncells,0);
    std::vector<float> hit_inside_agg; //hit_inside corresponds to scan points that hit something within a particular grid cell.
    hit_inside_agg.resize(Ncells,0);

    //average G(theta)
    std::vector<float> Gtheta_bar;
    Gtheta_bar.resize(Ncells,0.f);

    // nested vectors to save P calculation terms
    // cell, scan, variable, value
    std::vector<std::vector<float>>  P_equal_numerator_array(Ncells);
    std::vector<std::vector<float>>  P_equal_denominator_array(Ncells);

    std::vector<std::vector<uint>>  voxel_beam_count_array(Ncells);
    std::vector<std::vector<float>>  dr_array(Ncells);


    // r is the index of all hitpoints (all scans)
    // it is saved for the current scan in this_scan_index
    // i is the index of this_scan_index
    // so when accessing the things using getHitData(), getHitXYZ(), etc, 
    // for only one of the scans (this_scan), 
    // need to use getHitData(this_scan_index[i]) not getHitData(i)

    ////////////// We are going to perform all calculations on a scan-by-scan basis: loop through each scan
    for( uint s=0; s<Nscans; s++ ){

        if( printmessages ){
            std::cout << "------------------------SCAN = " << s << std::endl;
        }

        //only work with hitpoints associated with current scan
        std::vector<helios::vec3> this_scan_xyz;
        std::vector<uint> this_scan_index;
        for( size_t r=0; r< getHitCount(); r++ ){
            if( getHitScanID(r)==s ){
                this_scan_xyz.push_back( getHitXYZ(r) );
                this_scan_index.push_back(r);
            }
        }
        // size of the array of all the hits in this scan
        size_t Nhits = this_scan_xyz.size();

        if( printmessages ){
            std::cout << "Number of hitpoints associated with this scan = Nhits = " << Nhits << std::endl;
        }

        // count the number of beams (i.e. don't count multiple hits per beam)
        float previous_time = -1.f;
        uint Nbeams = 0;
        for(uint i=0; i < Nhits; i++)
        {
            float current_time = getHitData(this_scan_index[i], "timestamp");
            if(current_time != previous_time){
                Nbeams ++;
                previous_time = current_time;
            }
        }

        if( printmessages ){
            std::cout << "Number of beams associated with this scan = Nbeams = " << Nbeams << std::endl;
        }

        // create a vector each element of which represents a unique beam and is another vector of the hit point indices of this_scan_xyz that is associated with this beam
        float previous_beam = getHitData(this_scan_index[0], "timestamp");
        uint beam_ID = 0;
        std::vector<std::vector<uint>> beam_array(Nbeams);
        for(uint i=0; i < Nhits; i++)
        {
            float current_beam = getHitData(this_scan_index[i], "timestamp");

            if(current_beam == previous_beam)
            {
                beam_array.at(beam_ID).push_back(i); //ERK 
            }else{
                beam_ID ++;
                beam_array.at(beam_ID).push_back(i); //ERK
                previous_beam = current_beam;
            }

        }

        // set up GPU things for current scan
        const float3 origin = vec3tofloat3(getScanOrigin(s));
        float3* scan_xyz = (float3*)malloc( Nhits*sizeof(float3) );
        float* scan_weight = (float*)malloc( Nhits*sizeof(float) );

        for( size_t i=0; i<Nhits; i++ ){
            scan_xyz[i] = vec3tofloat3(this_scan_xyz.at(i));
            scan_weight[i] = 1.f;
        }

        float* hit_before = (float*)malloc( sizeof(float));
        float* hit_after = (float*)malloc( sizeof(float));
        float* d_hit_before;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_before, sizeof(float)) );
        float* d_hit_after;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_after, sizeof(float)) );

        float3* d_scan_xyz;
        CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nhits*sizeof(float3)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

        float* d_scan_weight;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_scan_weight, Nhits*sizeof(float)) );
        CUDA_CHECK_ERROR( cudaMemcpy(d_scan_weight, scan_weight, Nhits*sizeof(float), cudaMemcpyHostToDevice) );

        float* dr = (float*)malloc( Nhits*sizeof(float));
        float* d_dr;
        CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr, Nhits*sizeof(float)) );

        uint* hit_location = (uint*)malloc( Nhits*sizeof(uint));
        uint* d_hit_location;
        CUDA_CHECK_ERROR( cudaMalloc((uint**)&d_hit_location, Nhits*sizeof(uint)) );


        /////////////////////////////////////////////////////////////////////
        // now loop through each voxel
        for( uint c=0; c<Ncells; c++ ){

            if( printmessages ){
                std::cout << "----CELL = " << c << std::endl;
            }

            std::ofstream file_beam;
            if( beamoutput){
                // set up header of file that outputs one row for each beam in the current scan that interacts with the current voxel

                file_beam.open("../beamoutput/beam_data_s_" + std::to_string(s) + "_c_" + std::to_string(c) + ".txt");
                file_beam << "scan, cell, beam, E_before, E_inside, E_after, E_miss, sin_theta, dr, last_dr"  << std::endl;
            }

            //load the attributes of the grid cell
            float3 center = vec3tofloat3(getCellCenter(c));
            float3 anchor = vec3tofloat3(getCellGlobalAnchor(c));
            float3 size = vec3tofloat3(getCellSize(c));
            float rotation = getCellRotation(c);

            CUDA_CHECK_ERROR( cudaMemset( d_hit_location, 0, Nhits*sizeof(uint)) );
            CUDA_CHECK_ERROR( cudaMemset( d_dr, 0, Nhits*sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_before, 0, sizeof(float)) );
            CUDA_CHECK_ERROR( cudaMemset( d_hit_after, 0, sizeof(float)) );

            uint3 dimBlock = make_uint3( min(size_t(512),Nhits), 1, 1 );
            uint3 dimGrid = make_uint3( ceil(float(Nhits)/dimBlock.x), 1, 1  );

            if( dimBlock.x==0 && dimGrid.x==0 ){
                continue;
            }

            float scanner_range = 5000.0;
            intersectGridcell_synthetic <<< dimGrid, dimBlock >>>( Nhits, origin, d_scan_xyz, d_scan_weight, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after, d_hit_location, scanner_range );

            cudaDeviceSynchronize();
            CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

            //copy results back to host
            CUDA_CHECK_ERROR( cudaMemcpy( hit_before, d_hit_before, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_after, d_hit_after, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( dr, d_dr, Nhits*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR( cudaMemcpy( hit_location, d_hit_location, Nhits*sizeof(uint), cudaMemcpyDeviceToHost));

            float P_equal_numerator = 0;
            float P_equal_denominator = 0;
            uint voxel_beam_count = 0;

            if( printmessages ){
                std::cout << "Nbeams = " << Nbeams << std::endl;
            }
            ////////////////////////// loop through each beam associated with the current scan
            for(int k = 0; k < Nbeams; k++){

                float E_before = 0;
                float E_inside = 0;
                float E_after = 0;
                float E_miss = 0;

                float sin_theta;
                // float W = 0;
                float drr = 0;
                float last_drr = 0;

                ////////// loop through the hitpoints in the current beam to get number of rays in each location
                for(int j = 0; j < beam_array.at(k).size(); j++){

                    // pull out the index of the current scan's current beam's current hit (which is used to access the overall hit index r through this_scan_index[i])
                    uint i = beam_array.at(k).at(j);

                    helios::vec3 direction = getHitXYZ(this_scan_index[i])-getScanOrigin(getHitScanID(this_scan_index[i]));
                    direction.normalize();
                    sin_theta = sin(acos_safe(direction.z));

                    last_drr = dr[i];
                    drr += dr[i];

                    if(hit_location[i] == 1){
                        E_before ++;
                    }else if(hit_location[i] == 2){
                        E_inside ++;
                    }else if(hit_location[i] == 3){
                        E_after ++;
                        //   if(getHitData(this_scan_index[i], "target_index") == 0){ // if this is the first hitpoint for this beam,
                        //      W = 1.0;
                        //    }
                    }else if(hit_location[i] == 4){
                        E_miss ++;
                    } // or this hitpoint / beam did not intersect the voxel and should not be added to the total number of beams for this voxel

                } // end of loop through each hit in the current beam

                // calculate the average dr across all hitpoints for the beam (they should all be the same anyway... but just in case)
                float drrx = drr / float(beam_array.at(k).size());

                // if the path length is greater than 0 (it passes throught the current voxel), save it as an element of
                if( drrx > 0.f )
                {
                    dr_array.at(c).push_back(drrx);
                }

                if( beamoutput ){
                    // output info about the current beam to file
                    file_beam << s << "," << c << "," << k << "," << E_before << "," << E_inside << "," << E_after << "," << E_miss << "," << sin_theta << "," << drrx << "," << last_drr << std::endl;
                }

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // calculate terms equal weighting P method

                // P_equal
                if(E_inside != 0 || E_after != 0){ // only count for P_equal if some hit points were inside or after the voxel
                    P_equal_numerator += (E_after / (E_before + E_inside + E_after))*sin_theta;
                    P_equal_denominator += ((E_inside + E_after) / (E_before + E_inside + E_after))*sin_theta;
                    voxel_beam_count++;
                }else if(E_inside == 0 && E_after == 0 && E_before == 0 && E_miss != 0){ // also count this beam if there is only a "hitpoint" that missed (far after the voxel)
                    P_equal_numerator += 1*sin_theta;
                    P_equal_denominator += 1*sin_theta;
                    voxel_beam_count++;
                }


            }// end of loop through all beams for the current cell

            // save results for the current cell to the arrays
            P_equal_numerator_array.at(c).push_back(P_equal_numerator);
            P_equal_denominator_array.at(c).push_back(P_equal_denominator);
            voxel_beam_count_array.at(c).push_back(voxel_beam_count);
            hit_before_agg.at(c) += *hit_before;
            hit_after_agg.at(c) += *hit_after;

            for( size_t i=0; i<Nhits; i++ ){
                if( dr[i]>0.f ){
                    dr_agg.at(c).push_back(dr[i]);
                }
            }

            file_beam.close();
        } // end of loop through all cells

        free( scan_xyz );
        free( scan_weight );
        free( dr );
        free( hit_before );
        free( hit_after );
        CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
        CUDA_CHECK_ERROR( cudaFree(d_scan_weight) );
        CUDA_CHECK_ERROR( cudaFree(d_dr) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_before) );
        CUDA_CHECK_ERROR( cudaFree(d_hit_after) );

    }//end scan loop

    std::vector<float> P_equal(Ncells);
    std::vector<uint> voxel_beam_count_tot(Ncells);
    std::vector<float> dr_bar(Ncells);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // now aggregate over all scans
    for( uint c=0; c<Ncells; c++ ){

        float P_equal_numerator_array_agg = 0;
        float P_equal_denominator_array_agg = 0;
        uint voxel_beam_count_array_agg=0;

        for( uint s=0; s<Nscans; s++ ){

            P_equal_numerator_array_agg += P_equal_numerator_array.at(c).at(s);
            P_equal_denominator_array_agg += P_equal_denominator_array.at(c).at(s);
            voxel_beam_count_array_agg += voxel_beam_count_array.at(c).at(s);
        }

        P_equal[c] = P_equal_numerator_array_agg / P_equal_denominator_array_agg ;
        voxel_beam_count_tot[c] = voxel_beam_count_array_agg;

        if( printmessages ){
            std::cout << "Cell " << c << ", voxel_beam_count = " << voxel_beam_count_tot[c]  << std::endl;
            std::cout << "Cell " << c << ", P_equal = " << P_equal[c] << std::endl;
        }

        float mean_dr = 0;
        for(int j = 0; j < dr_array.at(c).size(); j++)
        {
            mean_dr += dr_array.at(c).at(j);
        }
        dr_bar[c] = mean_dr/ float(dr_array.at(c).size());
        if( printmessages ){
            std::cout << "Cell " << c << ", dr_bar = " <<  dr_bar[c] << std::endl;
        }
    } //end of loop through cells to aggregate all scans


    /// old code in original calculateLeafAreaGPU_testing() ... do we need to keep this??
    //----------- Calculate number of hits in voxels -------------- //

    //figure out hits for all scans
    for( size_t r=0; r< getHitCount(); r++ ){
        if( getHitGridCell(r)>=0 ){
            helios::vec3 direction = getHitXYZ(r)-getScanOrigin(getHitScanID(r));
            direction.normalize();
            hit_inside_agg.at(getHitGridCell(r)) += sin(acos_safe(direction.z));
        }
    }

    //---------------------- Calculate G(theta) from triangulation --------------------------//

    std::vector<float> Gtheta;
    Gtheta.resize(Ncells,0.f);

    const size_t Ntri = getTriangleCount();

    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells,0);

    std::vector<float> area_sin_sum;
    area_sin_sum.resize(Ncells,0.f);

    for( size_t t=0; t<Ntri; t++ ){

        Triangulation tri = getTriangle(t);
        int cell = tri.gridcell;

        if( cell>=0 && cell<Ncells ){ //triangle is inside a grid cell

            helios::vec3 t0 = tri.vertex0;
            helios::vec3 t1 = tri.vertex1;
            helios::vec3 t2 = tri.vertex2;

            helios::vec3 v0 = t1-t0;
            helios::vec3 v1 = t2-t0;
            helios::vec3 v2 = t2-t1;

            float L0 = v0.magnitude();
            float L1 = v1.magnitude();
            float L2 = v2.magnitude();

            float S = 0.5f*(L0+L1+L2);
            float area = sqrt( S*(S-L0)*(S-L1)*(S-L2) );

            helios::vec3 normal = cross( v0, v2 );
            normal.normalize();

            helios::vec3 raydir = t0-getScanOrigin( tri.scanID );
            raydir.normalize();

            float theta = fabs(acos_safe(raydir.z));

            if( area==area ){ //in rare cases you can get area=NaN

                Gtheta.at(cell) += fabs(normal*raydir)*area*fabs(sin(theta));
                area_sin_sum.at(cell) += area*fabs(sin(theta));
                cell_tri_count.at(cell) += 1;

            }

        }
    }

    for( uint c=0; c<Ncells; c++ ){
        if( cell_tri_count[c]>0 ){
            //Gtheta[c] *= float(cell_tri_count[c])/(area_sum[c]*sin_sum[c]);
            //Gtheta[c] *= float(cell_tri_count[c])/(area_sin_sum[c]);
            Gtheta[c] *= 1.0/(area_sin_sum[c]);

            Gtheta_bar[c] += Gtheta[c]/float(Nscans);
            std::cout << "Cell " << c << " Gtheta_bar = " <<  Gtheta_bar[c] << std::endl;
        }
    }


    //------------------ Perform inversions to get LAD using the different methods for P --------------------//

    if( printmessages ){
        std::cout << "Inverting to find LAD..." << std::flush;
    }

    // perform LAD inversion using triangulation-estimated Gtheta
    std::vector<float> LAD_equal = LAD_inversion(P_equal, Gtheta, dr_array, fillAnalytic);

    std::cout << "LAD inversion using triangulation complete" << std::endl;



    //ERK
    // std::vector<float> constant_G{0.6, 0.55, 0.5, 0.45, 0.4, 0.35};
    std::cout << "constant_G.size() = " << constant_G.size() << std::endl;

    std::vector<std::vector<float> > constant_G_LAD_equal;
    constant_G_LAD_equal.resize(constant_G.size());
    for(uint i=0;i<constant_G.size();i++)
    {
        std::vector<float> constant_G_vector(P_equal.size(), constant_G.at(i));
        // perform LAD inversion using triangulation-estimated Gtheta
        std::vector<float> LAD_equal_cg = LAD_inversion(P_equal, constant_G_vector, dr_array, fillAnalytic);
        //   std::cout << "LAD_equal_cg.size() = " << LAD_equal_cg.size() << std::endl; 
        constant_G_LAD_equal.at(i) = LAD_equal_cg;
    }

    // std::cout << "constant_G_LAD_equal.size() = " << constant_G_LAD_equal.size() << std::endl;
    //  std::cout << "constant_G_LAD_equal.at(0).size() = " << constant_G_LAD_equal.at(0).size() << std::endl;


    if( printmessages ){
        std::cout << "finished LAD inversions" << std::endl;
    }



    // output the voxel level variables
    std::ofstream file_output;
    file_output.open("../voxeloutput/voxeloutput.txt");
    file_output << "cell, grid_center_x, grid_center_y, grid_center_z, grid_size_x, grid_size_y, grid_size_z, Nbeams, dr_bar, G, P_equal, LAD_equal, ";
    for(uint i=0;i<constant_G.size();i++)
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << constant_G.at(i);
        std::string s = stream.str();
        std::string clab = "LAD_equal_constant_G_" + s;

        file_output <<  clab << ", ";
    }
    file_output << std::endl;


    std::cout << "printed labels " << std::endl;

    std::cout << "Ncells =  " << Ncells << std::endl;

    for(uint c=0; c<Ncells; c++)
    {
        helios::vec3 grid_size = getCellSize(c);
        helios::vec3 grid_center = getCellCenter(c);

        setCellLeafArea(LAD_equal[c]*grid_size.x*grid_size.y*grid_size.z,c);

        setCellGtheta( Gtheta[c], c );

        file_output << c << "," << grid_center.x << "," << grid_center.y << "," << grid_center.z << "," << grid_size.x << "," << grid_size.y << "," << grid_size.z << "," << voxel_beam_count_tot.at(c) << "," << dr_bar[c] << "," << Gtheta[c] << "," << P_equal[c] << "," << LAD_equal[c];

        for(uint i=0;i<constant_G.size();i++)
        {
            // std::cout << constant_G_LAD_equal.at(i).size() << std::endl;
            file_output << ", " << constant_G_LAD_equal.at(i).at(c);
        }
        file_output << std::endl;

    }
    file_output.close();

    if( printmessages ){
        std::cout << "done." << std::endl;
    }

}

std::vector<uint> LiDARcloud::peakFinder(std::vector<float> signal){

    for(uint i=0;i<signal.size();i++)
    {
        signal.at(i) = fabs(signal.at(i));
    }


    std::vector<uint> peakIndices;
    int peakIndex = -1;
    float peakValue = -1.0;
    float baseline = 0.0;
    // for(uint i=0;i<signal.size();i++)
    // {
    //   baseline = baseline + signal.at(i);
    // }
    // baseline = baseline/float(signal.size());
    // // baseline = 0.11;

    // std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
    // std::cout << "signal" << std::endl;
    //

    for(int i=0;i<signal.size();i++)
    {

        if(signal.at(i) > baseline)
        {
            // std::cout << ", signal.at(i) > baseline";

            if(peakValue < 0.0 | signal.at(i) > peakValue)
            {
                peakIndex = i;
                peakValue = signal.at(i);

                // std::cout << ", new peakIndex = " << i << ", new peakValue = " << peakValue << std::endl;

            }else{
                // std::cout << std::endl;
            }

        }else if(signal.at(i) <= baseline & peakIndex > -1){

            // std::cout << ", signal.at(i) <= baseline & peakIndex > -1";


            peakIndices.push_back(peakIndex);
            peakIndex = -1;
            peakValue = -1.0;

            // std::cout << ", peakIndex added = " << peakIndex << std::endl;


        }


    }
    if(peakIndex > -1)
    {
        peakIndices.push_back(peakIndex);
    }

    return peakIndices;
}




