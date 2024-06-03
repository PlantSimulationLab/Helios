/** \file "AerialLiDAR.cu" CUDA-related Aerial LiDAR plug-in functions

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "AerialLiDAR.cuh"
#include "AerialLiDAR.h"

using namespace AERIALLIDAR_CUDA;

__host__ __device__ float2 AERIALLIDAR_CUDA::operator+(const float2 &a, const float2 &b) {

  return make_float2(a.x+b.x, a.y+b.y);

}

__host__ __device__ float3 AERIALLIDAR_CUDA::operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__host__ __device__ float2 AERIALLIDAR_CUDA::operator-(const float2 &a, const float2 &b) {

  return make_float2(a.x-b.x, a.y-b.y);

}

__host__ __device__ float3 AERIALLIDAR_CUDA::operator-(const float3 &a, const float3 &b) {

  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}

// __host__ __device__ float3 operator-(const float3 &a){
  
//   return make_float3(-a.x,-a.y,-a.z);

// }

__host__ __device__ float AERIALLIDAR_CUDA::operator*(const float2 &a, const float2 &b) {

  return a.x*b.x + a.y*b.y;

}

__host__ __device__ float AERIALLIDAR_CUDA::operator*(const float3 &a, const float3 &b) {

  return a.x*b.x + a.y*b.y + a.z*b.z;

}

__host__ __device__ float3 AERIALLIDAR_CUDA::operator*(const float3 &a, const float& b) {
  float3 v = a;
  v.x *= b;
  v.y *= b;
  v.z *= b;
  return v;
}

__host__ __device__ float3 AERIALLIDAR_CUDA::operator/(const float3 &a, const float& b) {
  float3 v = a;
  v.x /= b;
  v.y /= b;
  v.z /= b;
  return v;
}

__host__ __device__ float3 AERIALLIDAR_CUDA::cross(const float3 &a, const float3 &b){
  float3 c;
  c.x=a.y*b.z - b.y*a.z;
  c.y=b.x*a.z - a.x*b.z;
  c.z=a.x*b.y - b.x*a.y;
  return c;
}

__host__ __device__ float AERIALLIDAR_CUDA::magnitude(const float3 &a){
  return sqrt( a.x*a.x + a.y*a.y + a.z*a.z );
}

__host__ __device__ float3 AERIALLIDAR_CUDA::normalize(const float3 &a){
  return a/magnitude(a);
}

float2 AERIALLIDAR_CUDA::vec2tofloat2( helios::vec2 v2 ){
  float2 f2;
  f2.x=v2.x;
  f2.y=v2.y;
  return f2;
}

helios::vec2 AERIALLIDAR_CUDA::float2tovec2( float2 f2 ){
  helios::vec2 v2;
  v2.x=f2.x;
  v2.y=f2.y;
  return v2;
}

float3 AERIALLIDAR_CUDA::vec3tofloat3( helios::vec3 v3 ){
  float3 f3;
  f3.x=v3.x;
  f3.y=v3.y;
  f3.z=v3.z;
  return f3;
}

helios::vec3 AERIALLIDAR_CUDA::float3tovec3( float3 f3 ){
  helios::vec3 v3;
  v3.x=f3.x;
  v3.y=f3.y;
  v3.z=f3.z;
  return v3;
}

__device__ float3 AERIALLIDAR_CUDA::d_rotatePoint(const float3 &position, const float &theta, const float &phi) {

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

__device__ float3 AERIALLIDAR_CUDA::d_sphere2cart( float radius, float elevation, float azimuth ){

  float3 Cartesian;

  Cartesian.x = radius*cos(elevation)*sin(azimuth);
  Cartesian.y = radius*cos(elevation)*cos(azimuth);
  Cartesian.z = radius*sin(elevation);

  return Cartesian;
  
}

__host__ __device__ float AERIALLIDAR_CUDA::atan2_2pi( const float& y, const float& x){

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

__host__ __device__ float AERIALLIDAR_CUDA::acos_safe( float x ){
  if (x < -1.0) x = -1.0 ;
  else if (x > 1.0) x = 1.0 ;
  return acosf(x) ;
}

__host__ int AERIALLIDAR_CUDA::randu( int imin, int imax  ){

  float ru = float(rand()) / float(RAND_MAX + 1.); 

  if( imin==imax || imin>imax ){
    return imin;
  }else{
    return imin + round(float(imax-imin)*ru);
  }
    
}

__device__ float2 AERIALLIDAR_CUDA::d_sampleDisk( float radius, uint seed ){

  // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

  float2 sp;

  float Rx = rnd(seed);
  float Ry = rnd(seed);
  
  //first map sample point to rectangle [-1,1] [-1,1]
  sp.x = -1.f + 2.f*Rx;
  sp.y = -1.f + 2.f*Ry;

  float r, p;
  if( sp.x>-sp.y) {
    if( sp.x > sp.y ){
      r = sp.x;
      p = sp.y/sp.x;
    }else{
      r = sp.y;
      p = 2.f-sp.x/sp.y;
    }
  }else{
    if( sp.x < sp.y ){
      r = -sp.x;
      p = 4.f + sp.y/sp.x;
    }else{
      r = -sp.y;
      if( sp.y!=0.f ){ //avoid division by zero at origin
	p = 6.f - sp.x/sp.y;
      }else{
	p = 0.f;
      } 
    }
  }
  p*=0.25f*M_PI;
  
  //find x,y point on unit disk
  sp.x = radius*r*cosf(p);
  sp.y = radius*r*sinf(p);

  return sp;
  
}

bool AERIALLIDAR_CUDA::sortcol0( const std::vector<float>& v0, const std::vector<float>& v1 ){
  return v0.at(0)<v1.at(0);
}

bool AERIALLIDAR_CUDA::sortcol1( const std::vector<float>& v0, const std::vector<float>& v1 ){
  return v0.at(1)<v1.at(1);
}

void AerialLiDARcloud::calculateLeafAreaGPU( const float Gtheta ){
  calculateLeafAreaGPU( Gtheta, 1 );
}

void AerialLiDARcloud::calculateLeafAreaGPU( const float Gtheta, const int minVoxelHits ){

  if( printmessages ){
    std::cout << "Calculating leaf area..." << std::endl;
  }
   
  if( !hitgridcellcomputed ){
    calculateHitGridCell();
  }
  
  const uint Nscans = getScanCount();

  helios::int3 N3 = getGridResolution();
  const uint Ncells = N3.x*N3.y*N3.z;

  //variable aggregates over all scans where we just keep tacking hits on the end for all scans
  std::vector<float> dr_agg; //dr is path length through grid cell
  dr_agg.resize(Ncells,0);
  std::vector<float> dr_hit_agg; //dr_hit is path length between grid cell intersection and hit point
  dr_hit_agg.resize(Ncells,0);
  std::vector<float> hit_denom_agg; //hit_denom corresponds to total number of scan points that reached a given grid cell
  hit_denom_agg.resize(Ncells,0);
  std::vector<float> hit_inside_agg; //hit_inside corresponds to scan points that hit something within a particular grid cell.
  hit_inside_agg.resize(Ncells,0);
  
  // We are going to perform all calculations on a scan-by-scan basis: loop through each scan
  for( uint s=0; s<Nscans; s++ ){

    // Perform ray-volume intersection tests to determine P

    std::vector<helios::vec3> this_scan_xyz, this_scan_raydir;

    for( size_t r=0; r<getHitCount(); r++ ){
      if( getHitScanID(r)==s ){

	helios::vec3 xyz=getHitXYZ(r);
	helios::vec3 raydir=sphere2cart(getHitRaydir(r));

	//if it is a ground point, extend it downward to way below the grid
	if( doesHitDataExist( r, "ground_flag" ) && getHitData( r, "ground_flag" )==1 ){
	  xyz = xyz + 1000.f*raydir;
	}
	
	this_scan_xyz.push_back( xyz );
	this_scan_raydir.push_back( raydir );
	//this_scan_raydir.push_back( helios::make_vec3(0,0,-1) );//assuming rays are vertical
      }
    }
    //std::cout << "NOTE: assuming all rays are vertical here." << std::endl;

    size_t Nhits = this_scan_xyz.size();

    float3* scan_xyz = (float3*)malloc( Nhits*sizeof(float3) );
    float3* scan_raydir = (float3*)malloc( Nhits*sizeof(float3) );
   
    for( size_t r=0; r<Nhits; r++ ){
      scan_xyz[r] = vec3tofloat3(this_scan_xyz.at(r));
      scan_raydir[r] = vec3tofloat3(this_scan_raydir.at(r));
    }

    float3* d_scan_xyz;
    CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nhits*sizeof(float3)) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

    float3* d_scan_raydir;
    CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_raydir, Nhits*sizeof(float3)) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_scan_raydir, scan_raydir, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

    float* d_dr;
    CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr, Ncells*sizeof(float)) );
    float* dr = (float*)malloc( Ncells*sizeof(float) );

    float* d_dr_hit;
    CUDA_CHECK_ERROR( cudaMalloc((float**)&d_dr_hit, Ncells*sizeof(float)) );
    float* dr_hit = (float*)malloc( Ncells * sizeof(float));
    
    float* d_hit_denom;
    CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_denom, Ncells*sizeof(float)) );
    float* hit_denom = (float*)malloc( Ncells*sizeof(float) );

    float* d_hit_inside;
    CUDA_CHECK_ERROR( cudaMalloc((float**)&d_hit_inside, Ncells*sizeof(float)) );
    float* hit_inside = (float*)malloc( Ncells*sizeof(float) );

    float3* d_cell_center;
    CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_cell_center, Ncells*sizeof(float3)) );
    float3* cell_center = (float3*)malloc( Ncells*sizeof(float3) );

    float3* d_cell_anchor;
    CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_cell_anchor, Ncells*sizeof(float3)) );
    float3* cell_anchor = (float3*)malloc( Ncells*sizeof(float3) );

    float3* d_cell_size;
    CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_cell_size, Ncells*sizeof(float3)) );
    float3* cell_size = (float3*)malloc( Ncells*sizeof(float3) );

    float* d_cell_rotation;
    CUDA_CHECK_ERROR( cudaMalloc((float**)&d_cell_rotation, Ncells*sizeof(float)) );
    float* cell_rotation = (float*)malloc( Ncells*sizeof(float) );

    for( uint c=0; c<Ncells; c++ ){
      
      //load the attributes of the grid cell
      cell_center[c] = vec3tofloat3(getCellCenter( gridindex2ijk(c) ));
      cell_anchor[c] = vec3tofloat3(getGridCenter() );
      cell_size[c] = vec3tofloat3(getCellSize() );
      cell_rotation[c] = getGridRotation();

    }

    CUDA_CHECK_ERROR( cudaMemcpy(d_cell_center, cell_center, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_cell_anchor, cell_anchor, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_cell_size, cell_size, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_cell_rotation, cell_rotation, Ncells*sizeof(float), cudaMemcpyHostToDevice) );

    CUDA_CHECK_ERROR( cudaMemset( d_dr, 0, Ncells*sizeof(float)) );
    CUDA_CHECK_ERROR( cudaMemset( d_dr_hit, 0, Ncells*sizeof(float)) );
    CUDA_CHECK_ERROR( cudaMemset( d_hit_denom, 0, Ncells*sizeof(float)) );
    CUDA_CHECK_ERROR( cudaMemset( d_hit_inside, 0, Ncells*sizeof(float)) );

    uint3 dimBlock = make_uint3( min(size_t(32),Nhits), min(uint(32),Ncells), 1 );
    uint3 dimGrid = make_uint3( ceil(float(Nhits)/dimBlock.x), ceil(float(Ncells)/dimBlock.y), 1  );

    if( dimBlock.x==0 || dimGrid.x==0 || dimBlock.y==0 || dimGrid.y==0 ){
      continue;
    }

    if( dimGrid.y>65535 ){
      std::cerr << "ERROR: Maximum supported number of gridcells is " << 65535*32 << "." << std::endl;
      return;
    }

    intersectGridcell <<< dimGrid, dimBlock >>>( Nhits, d_scan_xyz, d_scan_raydir, Ncells, d_cell_center, d_cell_anchor, d_cell_size, d_cell_rotation, d_dr, d_dr_hit, d_hit_denom, d_hit_inside );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here
    
    //copy results back to host
    CUDA_CHECK_ERROR( cudaMemcpy( dr, d_dr, Ncells*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR( cudaMemcpy( dr_hit, d_dr_hit, Ncells*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR( cudaMemcpy( hit_denom, d_hit_denom, Ncells*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR( cudaMemcpy( hit_inside, d_hit_inside, Ncells*sizeof(float), cudaMemcpyDeviceToHost));

    for( uint c=0; c<Ncells; c++ ){
      
	hit_denom_agg.at(c) += hit_denom[c];
	hit_inside_agg.at(c) += hit_inside[c];

	dr_agg.at(c) += dr[c];
	
	dr_hit_agg.at(c) += dr_hit[c];

    }
    
    free( scan_xyz );
    free( scan_raydir );
    free( dr );
    free( dr_hit );
    free( hit_denom );
    free( hit_inside );
    CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
    CUDA_CHECK_ERROR( cudaFree(d_scan_raydir) );
    CUDA_CHECK_ERROR( cudaFree(d_dr) );
    CUDA_CHECK_ERROR( cudaFree(d_dr_hit) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_denom) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_inside) );

 
  }//end scan loop

  //------------------ Perform inversion to get LAD --------------------//

  if( printmessages ){
    std::cout << "Inverting to find LAD..." << std::flush;
  }
     
  // float etol = 5e-5;
  // uint maxiter = 100;
    
  // float error, eold, aold, tmp;
  // for( uint v=0; v<Ncells; v++ ){
      
  //   if( Gtheta[v]==0 || Gtheta[v]!=Gtheta[v] ){
  //     // if( printmessages ){
  //     // 	std::cout << "G(theta) value bad for cell " << v << ": " << Gtheta[v] << std::endl;
  //     // }
  //     setCellLeafArea(0,v);
  //     continue;
  //   }else if( hit_after_agg[v]-hit_before_agg[v]<0 ){
  //     if( printmessages ){
  // 	std::cout << "Negative number of rays reaching cell " << v << ": " << hit_after_agg[v] << " " << hit_before_agg[v] << std::endl;
  //     }
  //     setCellLeafArea(0,v);
  //     continue;
  //   }else if( hit_inside_agg[v]<minVoxelHits ){
  //     if( printmessages ){
  // 	std::cout << "Not enough hits in voxel: " << hit_inside_agg[v] << " < " << minVoxelHits << std::endl;
  //     }
  //     setCellLeafArea(0,v);
  //     continue;
  //   }
      
  //   float P = 1.f-float(hit_inside_agg[v])/float(hit_after_agg[v]-hit_before_agg[v]);

  //    //assert(P>0);
      
  //   //initial guesses
  //   float a = 0.1f;
  //   float h = 0.01f;
    
  //   float mean = 0.f;
  //   uint count=0;
  //   for( int j=0; j<dr_agg.at(v).size(); j++ ){
  //     mean += exp(-a*dr_agg.at(v).at(j)*Gtheta[v]);
  //     count++;
  //     }
  //   mean /= float(dr_agg.at(v).size());      
  //   error = fabs(mean-P)/P;
    
  //   tmp = a;
    
  //   a = a + h;
    
  //   uint iter = 0;
  //   while( error>etol && iter<maxiter){
      
  //     aold = tmp;
  //     eold = error;
      
  //     float mean = 0.f;
  //     uint count = 0;
  //     for( int j=0; j<dr_agg.at(v).size(); j++ ){
  // 	mean += exp(-a*dr_agg.at(v).at(j)*Gtheta[v]);
  // 	count ++;
  //     }
  //     assert( count!=0 );
  //     mean /= float(count);      
  //     error = fabs(mean-P)/P;
      
  //     tmp = a;
      
  //     if( error==eold ){
  // 	break;
  //     }
      
  //     a = fabs( (aold*error-a*eold)/(error-eold) );
      
  //     iter++;
      
  //   }
    
  //   float dr_bar = 0;
  //   for( uint i=0; i<dr_agg.at(v).size(); i++ ){
  //     dr_bar += dr_agg.at(v).at(i);
  //   }
  //   dr_bar /= float(dr_agg.at(v).size());
    
  //   if( iter>=maxiter-1 || a!=a || a>100 ){
  //     if( printmessages ){
  // 	std::cout << "WARNING: LAD inversion failed for volume #" << v << ". Using average dr formulation." << std::endl;
  //     }
      
  //     a = (1.f-P)/(dr_bar*Gtheta[v]);
      
  //   }

  //   if( a>5 ){
  //     a = fmin((1.f-P)/dr_bar/Gtheta[v],-log(P)/dr_bar/Gtheta[v]);
  //   }

  //   helios::vec3 gridsize = getCellSize(v);
  //   setCellLeafArea(a*gridsize.x*gridsize.y*gridsize.z,v);

  //   setCellGtheta( Gtheta[v], v );

  //   if( printmessages ){
  //     std::cout << "Vol #" << v << " mean dr: " << dr_bar << std::endl;
  //     std::cout << "Vol #" << v << " mean G(theta): " << Gtheta[v] << std::endl;
  //     //std::cout << "Vol #" << v << " intersections: " << hit_inside_agg[v] << " " << hit_after_agg[v] << " " << hit_before_agg[v] << std::endl;
      
  //     std::cout << "Vol #" << v << " LAD: " << a << std::endl;
  //     //std::cout << "Vol #" << v << " LAD: " << a << std::endl;
  //     //   cout << "Vol #" << v << " LAD: " << (1.f-P)/(dr_bar[v]*Gthetav]) << " [point quadrat]" << endl;
  //     //std::cout << "Vol #" << v << " Leaf Area: " << getCellLeafArea(v)*100.f*100.f << " [P = mean(exp(-a*dr*Gtheta))]" << std::endl;
  //     //std::cout << "Vol #" << v << " Leaf Area: " << -log(P)/(dr_bar*Gtheta[v])*gridsize.x*gridsize.y*gridsize.z*10000.f << " [a = -ln(P)/(dr*Gtheta)]" << std::endl;
  //     // A_sum += LAD[v]*gridsize.x*gridsize.y*gridsize.z*100.f*100.f;
  //   }
      
  // }

  for( uint v=0; v<Ncells; v++ ){

    dr_agg.at(v) /= float(hit_denom_agg.at(v));
    dr_hit_agg.at(v) /= float(hit_inside_agg.at(v));

    float dr_bar = getCellSize().z;
    
    float P = 1.f-float(hit_inside_agg[v])/float(hit_denom_agg[v]);

    float a;
    if( hit_inside_agg[v]<minVoxelHits ){
      a = 0.f;
    // }else if( P<0.1 ){

    //   dr_bar = dr_hit_agg.at(v);
      
    //   a = 1.f/dr_bar/Gtheta;
    }else{
      a = -log(P)/dr_bar/Gtheta;
    }

    helios::vec3 gridsize = getCellSize();
    setCellLeafArea(a*gridsize.x*gridsize.y*gridsize.z, gridindex2ijk(v) );

    setCellRbar( dr_hit_agg.at(v), gridindex2ijk(v) );

    setCellTransmissionProbability( hit_denom_agg[v], hit_denom_agg[v]-hit_inside_agg[v],gridindex2ijk(v) );


    if( printmessages ){
      // std::cout << "Vol #" << v << " LAD: " << a << std::endl;
      //std::cout << "Vol #" << v << " LAI: " << a*gridsize.z << std::endl;
      //std::cout << "Vol #" << v << " dr: " << dr_bar << std::endl;
      // std::cout << "Vol #" << v << " P: " << P << std::endl;
    }
    
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }
    
}

void AerialLiDARcloud::syntheticScan( helios::Context* context, const char* xml_file ){
  syntheticScan( context, xml_file, 1, 0 );
}

void AerialLiDARcloud::syntheticScan( helios::Context* context, const char* xml_file, const int rays_per_pulse, const float pulse_distance_threshold ){

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

  if( printmessages ){
    disableMessages();
    loadXML(xml_file);
    enableMessages();
  }else{
    loadXML(xml_file);
  }
  
  AerialLiDARcloud synthscan;

  if( !printmessages ){
    synthscan.disableMessages();
  }
    
  //Load the synthetic scan metedata
  synthscan.loadXML(xml_file);

  //Determine bounding box for Context geometry
  helios::vec2 xbounds, ybounds, zbounds;
  context->getDomainBoundingBox(xbounds,ybounds,zbounds);

  if( xbounds.x==xbounds.y ){
    xbounds.y = xbounds.x+1.f;
  }
  if( ybounds.x==ybounds.y ){
    ybounds.y = ybounds.x+1.f;
  }
  if( zbounds.x==zbounds.y ){
    zbounds.y = zbounds.x+1.f;
  }

  float3 bb_center = make_float3(xbounds.x+0.5*(xbounds.y-xbounds.x),ybounds.x+0.5*(ybounds.y-ybounds.x),zbounds.x+0.5*(zbounds.y-zbounds.x));
  float3 bb_size = make_float3(xbounds.y-xbounds.x,ybounds.y-ybounds.x,zbounds.y-zbounds.x);

  //get geometry information and copy to GPU

  size_t c=0;
  
  std::map<std::string,int> textures;
  std::map<std::string,int2> texture_size;
  std::map<std::string,std::vector<std::vector<bool> > > texture_data;
  int tID = 0;

  std::vector<uint> UUIDs_all = context->getAllUUIDs();
  
  //----- PATCHES ----- //

  //figure out how many patches
  size_t Npatches = 0;
  for( int p=0; p<UUIDs_all.size(); p++ ){
    if( context->getPrimitiveType(UUIDs_all.at(p)) == helios::PRIMITIVE_TYPE_PATCH ){
      Npatches++;
    }
  }

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
              if( uv.size()==4 ){//cusom uv coordinates
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
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_textureID,Ntriangles*sizeof(int)) ); //allocate device memory
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

  for( int s=0; s<synthscan.getScanCount(); s++ ){

    float coneangle = synthscan.getScanConeAngle(s);

    helios::vec3 center = synthscan.getScanCenter(s);

    helios::vec2 extent = synthscan.getScanExtent(s);

    float scandensity = synthscan.getScanDensity(s);

    size_t N = scandensity*extent.x*extent.y;

    if( extent.x<=0 || extent.y<=0 ){
      std::cerr << "ERROR (syntheticScan): The scan extent specified in the metadata file " << xml_file << " must be greater than zero." << std::endl;
      exit(EXIT_FAILURE);
    }else if( scandensity<=0 ){
      std::cerr << "ERROR (syntheticScan): The scan density specified in the metadata file " << xml_file << " must be greater than zero." << std::endl;
      exit(EXIT_FAILURE);
    }

    assert( N>0 );

    std::vector<helios::vec3> rayorigin;
    rayorigin.resize(N);
    
    std::vector<helios::vec3> raydir;
    raydir.resize(N);
  
    for (size_t i=0; i<N; i++ ){

      float x = center.x - 0.5*extent.x + context->randu()*extent.x;
      float y = center.y - 0.5*extent.y + context->randu()*extent.y;
      float z = center.z;

      rayorigin.at(i) = helios::make_vec3(x,y,z);

      float theta = -0.5f*M_PI+0.5f*coneangle*context->randu();
      float phi = 2.f*M_PI*context->randu();

      raydir.at(i) = helios::sphere2cart(helios::make_SphericalCoord(theta,phi));

    }

    float3* d_hit_xyz;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_xyz,N*sizeof(float3)) ); //allocate device memory

    float3* d_hit_origin;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_origin,N*sizeof(float3)) ); //allocate device memory

    //copy scan data into the host buffer
    float3* hit_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
    float3* hit_origin = (float3*)malloc(N * sizeof(float3)); //allocate host memory
    for( std::size_t r=0; r<N; r++ ){
      hit_xyz[r] = vec3tofloat3(rayorigin.at(r)+raydir.at(r)*10000.f);
      hit_origin[r] = vec3tofloat3(rayorigin.at(r));
    }

    //copy from host to device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_xyz, hit_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_origin, hit_origin, N*sizeof(float3), cudaMemcpyHostToDevice) );

    uint* bb_hit = (uint*)malloc(N * sizeof(uint)); //allocate host memory
    uint* d_bb_hit;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_bb_hit,N*sizeof(uint)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemset( d_bb_hit, 0, N*sizeof(uint)) ); //initialize to zero, set equal to 1 if the ray is found to intersect bounding box
    
    //Launch kernel to determine which rays intersect bounding box
    uint3 dimBlock = make_uint3( 512, 1, 1 );
    uint3 dimGrid = make_uint3( ceil(float(N)/float(dimBlock.x)), 1, 1 );
    intersectBoundingBox<<< dimGrid, dimBlock >>>( N, d_hit_origin, d_hit_xyz, bb_center, bb_size, d_bb_hit );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

    //copy hit flag back to host
    CUDA_CHECK_ERROR( cudaMemcpy(bb_hit, d_bb_hit, N*sizeof(uint), cudaMemcpyDeviceToHost) );

    CUDA_CHECK_ERROR( cudaFree(d_hit_xyz) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_origin) );
    CUDA_CHECK_ERROR( cudaFree(d_bb_hit) );

    //determine how many rays hit the bounding box
    uint Nbb = 0;
    for( int i=0; i<N; i++ ){
      if( bb_hit[i]==1 ){
  	Nbb++;
      }
    }
    N = Nbb;

    if( N==0 ){
      std::cout << "WARNING: Synthetic rays did not hit any primitives for scan " << s << "." << std::endl;
      continue;
    }

    //make a new array of ray directions for rays that hit bounding box
    float3* direction = (float3*)malloc(N * sizeof(float3)); //allocate host memory
    float3* origin = (float3*)malloc(N * sizeof(float3)); //allocate host memory

    int count=0;
    for( int i=0; i<N; i++ ){
      if( bb_hit[i]==1 ){
  	direction[count] = vec3tofloat3(raydir.at(i));
  	//origin[count] = vec3tofloat3(rayorigin.at(i)-raydir.at(i)*1e5);
	origin[count] = vec3tofloat3(rayorigin.at(i));
  	count++;
      }
    }    
    free(bb_hit);

    float3* d_raydir;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_raydir,N*sizeof(float3)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_raydir, direction, N*sizeof(float3), cudaMemcpyHostToDevice) );

    float3* d_origin;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_origin,N*sizeof(float3)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_origin, origin, N*sizeof(float3), cudaMemcpyHostToDevice) );


    //Distance to intersection
    float* d_hit_t;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_t,N*Npulse*sizeof(float)) ); //allocate device memory
    float* hit_t = (float*)malloc(N * Npulse * sizeof(float)); //allocate host memory
    for( int i=0; i<N*Npulse; i++ ){
      hit_t[i] = 1e6;
    }
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_t, hit_t, N*Npulse*sizeof(float), cudaMemcpyHostToDevice) );

    float* d_hit_fnorm;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_fnorm,N*Npulse*sizeof(float)) ); //allocate device memory
    float* hit_fnorm = (float*)malloc(N*Npulse * sizeof(float)); //allocate host memory
    for( int i=0; i<N*Npulse; i++ ){
      hit_fnorm[i] = 1e6;
    }
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_fnorm, hit_fnorm, N*Npulse*sizeof(float), cudaMemcpyHostToDevice) );

    float exit_diameter = synthscan.getScanBeamExitDiameter(s);
    float beam_divergence = synthscan.getScanBeamDivergence(s);

    if( Npulse>1 ){
        dimBlock = make_uint3( 128, 4, 1 );
    }else{
        dimBlock = make_uint3( 512, 1, 1 );
    }
    dimGrid = make_uint3( ceil(float(N)/float(dimBlock.x)), ceil(float(Npulse)/float(dimBlock.y)), 1 );

    //---- patch kernel ----//
    intersectPatches<<< dimGrid, dimBlock >>>( N, Npulse, d_origin, d_raydir, exit_diameter, beam_divergence, Npatches, d_patch_vertex, d_patch_textureID, Ntextures, d_masksize, masksize_max, d_maskdata, d_patch_uv, d_hit_t, d_hit_fnorm );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here
      
    //---- triangle kernel ----//
    intersectTriangles<<< dimGrid, dimBlock >>>( N, Npulse, d_origin, d_raydir, exit_diameter, beam_divergence, Ntriangles, d_tri_vertex, d_tri_textureID, Ntextures, d_masksize, masksize_max, d_maskdata, d_tri_uv, d_hit_t, d_hit_fnorm );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here
    
    //copy back
    CUDA_CHECK_ERROR( cudaMemcpy(hit_t, d_hit_t, N*Npulse*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK_ERROR( cudaMemcpy(hit_fnorm, d_hit_fnorm, N*Npulse*sizeof(float), cudaMemcpyDeviceToHost) );

    size_t Nhits = 0;
    for( size_t r=0; r<N; r++ ){

      std::vector<std::vector<float> > t_pulse;
      std::vector<std::vector<float> > t_hit;
      
      for( size_t p=0; p<Npulse; p++ ){
	
    	float t = hit_t[r*Npulse+p];
	float i = hit_fnorm[r*Npulse+p];

    	if( t<1e6 ){
	  std::vector<float> v{t,i};
    	  t_pulse.push_back(v);
    	}

      }

      if( t_pulse.size()==1 ){

    	t_hit.push_back(t_pulse.front());

      }else if( t_pulse.size()>1 ){
	
    	std::sort( t_pulse.begin(), t_pulse.end(), sortcol0 );

	float t0=t_pulse.at(0).at(0);
    	float d=t_pulse.at(0).at(0);
	float f=t_pulse.at(0).at(1);
    	int count=1;
    	for( size_t hit=1; hit<t_pulse.size(); hit++ ){

    	  if( t_pulse.at(hit).at(0)-t0>pulse_distance_threshold || hit==t_pulse.size()-1 ){

	    std::vector<float> v{d/float(count), f/float(Npulse)};
    	    t_hit.push_back( v );
	    
    	    Nhits++;
    	    count=1;
    	    d=t_pulse.at(hit).at(0);
    	    t0=t_pulse.at(hit).at(0);
	    f=t_pulse.at(hit).at(1);
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
	
    	std::map<std::string,float> data;
    	data["target_index"] = hit;
    	data["target_count"] = t_hit.size();
	data["deviation"] = fabs(t_hit.at(hit).at(0)-average);
	data["timestamp"] = r;
	data["intensity"] = t_hit.at(hit).at(1);
	
    	helios::vec3 dir = helios::make_vec3(direction[r].x,direction[r].y,direction[r].z);
    	helios::vec3 o = helios::make_vec3(origin[r].x,origin[r].y,origin[r].z);
    	helios::vec3 p = o+dir*t_hit.at(hit).at(0);
    	addHitPoint( s, p, helios::cart2sphere(dir), helios::RGB::red, data );

	Nhits++;
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
    CUDA_CHECK_ERROR( cudaFree(d_hit_t) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_fnorm) );
    CUDA_CHECK_ERROR( cudaFree(d_raydir) );
    CUDA_CHECK_ERROR( cudaFree(d_origin) );
    free(patch_vertex);
    free(patch_textureID);
    free(patch_uv);
    free(tri_vertex);
    free(tri_textureID);
    free(tri_uv);
    free(maskdata);
    free(masksize);
    free(hit_xyz);
    free(direction);
    free(origin);
    free(hit_t);
    free(hit_fnorm);

    if( printmessages ){
      std::cout << "Created synthetic scan #" << s << " with " << Nhits << " hit points." << std::endl;
    }
    
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }
  
}

void AerialLiDARcloud::calculateSyntheticLeafArea( helios::Context* context ){

  // float3* d_prim_xyz;

  // const uint N = context->getPrimitiveCount();
  
  // float3* prim_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_prim_xyz,N*sizeof(float3)) ); //allocate device memory

  // //copy scan data into the host buffer
  // for( std::size_t p=0; p<N; p++ ){
  //   std::vector<helios::vec3> verts = context->getPrimitivePointer(p)->getVertices();
  //   prim_xyz[p] = vec3tofloat3( verts.at(0) );
  // }

  // //copy from host to device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_prim_xyz, prim_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );

  // // ---- Grid Cells ---- //

  // //! Grid cell centers on device (GPU) memory
  // float3* d_grid_center;

  // //! Grid cell global anchor on device (GPU) memory
  // float3* d_grid_anchor;

  // //! Grid sizes on device (GPU) memory
  // float3* d_grid_size;

  // //! Grid rotations on device (GPU) memory
  // float* d_grid_rotation;

  // const uint Ncells = getGridCellCount();

  // float3* center = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_center,Ncells*sizeof(float3)) ); //allocate device memory

  // float3* anchor = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_anchor,Ncells*sizeof(float3)) ); //allocate device memory

  // float3* size = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_size,Ncells*sizeof(float3)) ); //allocate device memory

  // float* rotation = (float*)malloc(Ncells * sizeof(float)); //allocate host memory
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_rotation,Ncells*sizeof(float)) ); //allocate device memory
  
  // //copy grid data into the host buffer
  // for( int c=0; c<Ncells; c++ ){
  //   center[c] = vec3tofloat3(getCellCenter(c));
  //   anchor[c] = vec3tofloat3(getCellGlobalAnchor(c));
  //   size[c] = vec3tofloat3(getCellSize(c));
  //   rotation[c] = getCellRotation(c);
  // }

  // //copy from host to device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_grid_center, center, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
  // CUDA_CHECK_ERROR( cudaMemcpy(d_grid_anchor, anchor, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
  // CUDA_CHECK_ERROR( cudaMemcpy(d_grid_size, size, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
  // CUDA_CHECK_ERROR( cudaMemcpy(d_grid_rotation, rotation, Ncells*sizeof(float), cudaMemcpyHostToDevice) );

  // free(prim_xyz);
  // free(center);
  // free(anchor);
  // free(size);
  // free(rotation);

  // // Result buffer
  // int* prim_vol = (int*)malloc(N * sizeof(int));
  // int* d_prim_vol;
  // CUDA_CHECK_ERROR( cudaMalloc(&d_prim_vol,N*sizeof(int)) );

  // dim3 dimBlock( 64, 1 );
  // dim3 dimGrid( ceil(N/64.f) );
  // insideVolume <<< dimGrid, dimBlock >>>( N, d_prim_xyz, getGridCellCount(), d_grid_size, d_grid_center, d_grid_anchor, d_grid_rotation, d_prim_vol );

  // CUDA_CHECK_ERROR( cudaPeekAtLastError() );
  // CUDA_CHECK_ERROR( cudaDeviceSynchronize() );
  
  // CUDA_CHECK_ERROR( cudaMemcpy(prim_vol, d_prim_vol, N*sizeof(int), cudaMemcpyDeviceToHost) );

  // std::vector<float> total_area;
  // total_area.resize(Ncells,0.f);
  // for( std::size_t p=0; p<N; p++ ){
  //   if( prim_vol[p]>=0 ){
  //     uint gridcell = prim_vol[p];
  //     total_area.at(gridcell) += context->getPrimitivePointer(p)->getArea();
  //     context->setPrimitiveData(p,"gridCell",helios::HELIOS_TYPE_UINT,1,&gridcell);
  //   }
  // }

  // std::ofstream file;

  // file.open("../output/synthetic_leaf_areas.txt");
  
  // for( int v=0; v<Ncells; v++ ){
  //   file << total_area.at(v) << std::endl;
  // }

  // file.close();
  
  // free(prim_vol);

  // CUDA_CHECK_ERROR( cudaFree(d_prim_vol) );
  // CUDA_CHECK_ERROR( cudaFree(d_prim_xyz) );
  // CUDA_CHECK_ERROR( cudaFree(d_grid_center) );
  // CUDA_CHECK_ERROR( cudaFree(d_grid_anchor) );
  // CUDA_CHECK_ERROR( cudaFree(d_grid_size) );
  // CUDA_CHECK_ERROR( cudaFree(d_grid_rotation) );  

}

__global__ void AERIALLIDAR_CUDA::intersectBoundingBox( const size_t scanSize, float3* d_hit_origin, const float3* d_scan_xyz, const float3 bbcenter, const float3 bbsize, uint* d_boundingbox_hit ){
  
  size_t i = blockIdx.x*blockDim.x+threadIdx.x;

  if( i>=scanSize ){
    return;
  }

  float3 scan_xyz = d_scan_xyz[ i ];

  float3 origin = d_hit_origin[i];

  float3 direction = normalize(scan_xyz-origin);

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

__global__ void AERIALLIDAR_CUDA::intersectGridcell( const size_t Nhitsbb, float3* d_scan_xyz, float3* d_scan_raydir, const size_t Ncells, float3* center, float3* anchor, float3* size, float* rotation, float* d_dr, float* d_dr_hit, float* hit_denom, float* hit_inside ){
  
  size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

  if( idx>=Nhitsbb ){
    return;
  }

  size_t cell = blockIdx.y*blockDim.y+threadIdx.y;

  if( cell>=Ncells ){
    return;
  }

  float3 scan_xyz = d_scan_xyz[ idx ];
  float3 direction = d_scan_raydir[ idx ];

  //Inverse rotate the ray

  float3 direction_rot = d_rotatePoint(direction,0,-rotation[cell]);
  float3 scan_xyz_rot = d_rotatePoint(scan_xyz-anchor[cell],0,-rotation[cell]) + anchor[cell];

  float3 origin = scan_xyz_rot - direction_rot*1e5;
  
  float ox = origin.x; float oy = origin.y; float oz = origin.z;
  float dx = direction_rot.x; float dy = direction_rot.y; float dz = direction_rot.z;
  
  float x0 = center[cell].x - 0.5f*size[cell].x; float x1 = center[cell].x + 0.5f*size[cell].x;
  float y0 = center[cell].y - 0.5f*size[cell].y; float y1 = center[cell].y + 0.5f*size[cell].y;
  float z0 = center[cell].z - 0.5f*size[cell].z; float z1 = center[cell].z + 0.5f*size[cell].z;
  
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
    float t = magnitude( scan_xyz_rot - origin );  //t-distance to hit point
    
    if( t>=t0 ){ //hit lies within or beyond the volume

      atomicAdd( &hit_denom[cell], 1.f );

      atomicAdd( &d_dr[ cell ], fabs(t1-t0) );

    }
	
    if( t>=t0 && t<t1 ){ //hit lies inside the volume

      atomicAdd( &hit_inside[cell], 1.f );

      atomicAdd( &d_dr_hit[ cell ], fabs(t-t0) );
            
    }
    
  }

}

__global__ void AERIALLIDAR_CUDA::intersectPatches( const size_t Npulse, const int rays_per_pulse, float3* d_origin, float3* d_raydir, const float exit_diameter, const float beam_divergence, const int Npatches, float3* d_patch_vertex, int* d_patch_textureID, const int Ntextures, int2* d_masksize, int2 masksize_max, bool* d_maskdata, float2* d_patch_uv, float* d_hit_t, float* d_hit_fnorm ){

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  int subpulse = blockIdx.y*blockDim.y+threadIdx.y;

  if( idx>=Npulse || subpulse>=rays_per_pulse ){
    return;
  }

  int i_global = idx*rays_per_pulse+subpulse;

  uint seed = tea<16>(subpulse,i_global);

  float2 disk_pt = d_sampleDisk( 0.5*exit_diameter, seed );
  
  float3 raydir = d_sphere2cart( 1.f, 0.5*M_PI-0.5*beam_divergence*rnd(seed), 2.f*M_PI*rnd(seed) );

  float zenith = acos_safe( d_raydir[idx].z/sqrt(d_raydir[idx].x*d_raydir[idx].x+d_raydir[idx].y*d_raydir[idx].y+d_raydir[idx].z*d_raydir[idx].z) );
  float azimuth = atan2_2pi( d_raydir[idx].x, d_raydir[idx].y );
  raydir = d_rotatePoint( raydir, zenith, 0.5*M_PI-azimuth );

  float3 disk_pt3 = d_rotatePoint( make_float3(disk_pt.x,disk_pt.y,0), zenith, azimuth );

  float3 origin = d_origin[idx] + disk_pt3;

  float tmin = d_hit_t[i_global];
  float nmin=d_hit_fnorm[i_global];
  for( int p=0; p<Npatches; p++ ){

    float3 v0 = d_patch_vertex[4*p];
    float3 v1 = d_patch_vertex[4*p+1];
    float3 v2 = d_patch_vertex[4*p+2];
    float3 v3 = d_patch_vertex[4*p+3];

    float3 anchor = v0;
    float3 normal = normalize( cross( v1-v0, v2-v0 ) );

    float3 a = v1-v0;
    float3 b = v3-v0;
    
    float t = ((anchor - origin)*normal) / (raydir*normal); 
    
    if( t==t && t>1e-8 && t<tmin ){
			
      float3 point = origin + raydir * t;
      float3 d = point - anchor;
      
      float ddota = d*a;
      
      if (ddota > 0.0 && ddota < a*a ){
	
	float ddotb = d*b;
	
	if (ddotb > 0.0 && ddotb < b*b ){

	  if( d_patch_textureID[p]<0 ){//no texture mask
	    
	    tmin = t;
	    nmin = fabs(normal*raydir);

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
	    
	    }
	    
	  }
	    
	}	
	
      }
    }
  }

  d_hit_t[i_global] = tmin;

  d_hit_fnorm[i_global] = nmin;

}

__global__ void AERIALLIDAR_CUDA::intersectTriangles( const size_t Npulse, const int rays_per_pulse, float3* d_origin, float3* d_raydir, const float exit_diameter, const float beam_divergence, const int Ntriangles, float3* d_tri_vertex, int* d_tri_textureID, const int Ntextures, int2* d_masksize, const int2 masksize_max, bool* d_maskdata, float2* d_tri_uv, float* d_hit_t, float* d_hit_fnorm ){

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  int subpulse = blockIdx.y*blockDim.y+threadIdx.y;

  if( idx>=Npulse || subpulse>=rays_per_pulse ){
    return;
  }

  int i_global = idx*rays_per_pulse+subpulse;

  uint seed = tea<16>(subpulse,i_global);

  float2 disk_pt = d_sampleDisk( 0.5*exit_diameter, seed );
  
  float3 raydir = d_sphere2cart( 1.f, 0.5*M_PI-0.5*beam_divergence*rnd(seed), 2.f*M_PI*rnd(seed) );

  float zenith = acos_safe( d_raydir[idx].z/sqrt(d_raydir[idx].x*d_raydir[idx].x+d_raydir[idx].y*d_raydir[idx].y+d_raydir[idx].z*d_raydir[idx].z) );
  float azimuth = atan2_2pi( d_raydir[idx].x, d_raydir[idx].y );
  raydir = d_rotatePoint( raydir, zenith, 0.5*M_PI-azimuth );

  float3 disk_pt3 = d_rotatePoint( make_float3(disk_pt.x,disk_pt.y,0), zenith, azimuth );

  float3 origin = d_origin[idx] + disk_pt3;

  float tmin = d_hit_t[i_global];
  float nmin = d_hit_fnorm[i_global];
  for( int tri=0; tri<Ntriangles; tri++ ){

    float3 v0 = d_tri_vertex[3*tri];
    float3 v1 = d_tri_vertex[3*tri+1];
    float3 v2 = d_tri_vertex[3*tri+2];

    float a = v0.x - v1.x, b = v0.x - v2.x, c = raydir.x, d = v0.x - origin.x; 
    float e = v0.y - v1.y, f = v0.y - v2.y, g = raydir.y, h = v0.y - origin.y;
    float i = v0.z - v1.z, j = v0.z - v2.z, k = raydir.z, l = v0.z - origin.z;
		
    float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
    float q = g * i - e * k, s = e * j - f * i;
  
    float inv_denom  = 1.f / (a * m + b * q + c * s);
  
    float e1 = d * m - b * n - c * p;
    float beta = e1 * inv_denom;
  
    if (beta > 0.0){
      
      float r = r = e * l - h * i;
      float e2 = a * n + d * q + c * r;
      float gamma = e2 * inv_denom;
      
      if (gamma > 0.0 && beta + gamma < 1.0 ){
	
	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;

	if( t>1e-6 && t<tmin ){

	  if( d_tri_textureID[tri]<0 ){//no texture mask	  
	    tmin = t;
	    float3 normal = normalize( cross( v1-v0, v2-v0 ) );
	    nmin = fabs(normal*raydir);
	  }else{//has texture mask

	    int ID = d_tri_textureID[tri];
	    int2 sz = d_masksize[ID];
	    
	    float2 uv0 = d_tri_uv[tri*3];
	    float2 uv1 = d_tri_uv[tri*3+1];
	    float2 uv2 = d_tri_uv[tri*3+2];

	    float2 uv;
	    uv.x = 1.f - (uv0.x + beta*(uv1.x-uv0.x) + gamma*(uv2.x-uv0.x));
	    uv.y = (uv0.y + beta*(uv1.y-uv0.y) + gamma*(uv2.y-uv0.y));

	    uint2 ind = make_uint2( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)) );
	    assert( ind.x<sz.x && ind.y<sz.y );

	    if( d_maskdata[ ID*masksize_max.x*masksize_max.y + ind.y*masksize_max.x + ind.x ] ){
	      tmin = t;
	      float3 normal = normalize( cross( v1-v0, v2-v0 ) );
	      nmin = fabs(normal*raydir);
	    }
	    
	    
	  }
	}	
      
      }
    }
  }

  d_hit_t[i_global] = tmin;

  d_hit_fnorm[i_global] = nmin;
  
}
