/** \file "VoxelIntersection.cu" CUDA-related functions for voxel-intersection plug-in

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/
    
#include "VoxelIntersection.h"

using namespace helios;

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float3 inline vec3tofloat3( helios::vec3 v3 ){
  float3 f3;
  f3.x=v3.x;
  f3.y=v3.y;
  f3.z=v3.z;
  return f3;
}

helios::vec3 inline float3tovec3( float3 f3 ){
  helios::vec3 v3;
  v3.x=f3.x;
  v3.y=f3.y;
  v3.z=f3.z;
  return v3;
}

__device__ float3 d_rotatePoint_vi(const float3 &position, const float &theta, const float &phi) {

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

__global__ void insideVolume_vi( const uint Nhits, const float3* d_hit_xyz, const uint Ngridcells, const float3* d_grid_size, const float3* d_grid_center, const float* d_grid_rotation, int* d_hit_vol ){
  
  uint t = blockIdx.x*blockDim.x+threadIdx.x;

  if( t>=Nhits ){
    return;
  }

  d_hit_vol[t] = -1;

  float3 hit_xyz = d_hit_xyz[t];

  for( int i=0; i<Ngridcells; i++ ){

    float3 center = d_grid_center[i];
    float3 size = d_grid_size[i];
    float rotation = d_grid_rotation[i];

    float3 origin = make_float3(0,0,0); 

    float3 xyz = hit_xyz;
    xyz.x -= center.x;
    xyz.y -= center.y;
    xyz.z -= center.z;
    float3 hit_xyz_rot = d_rotatePoint_vi(xyz,0,-rotation);
    hit_xyz_rot.x += center.x;
    hit_xyz_rot.y += center.y;
    hit_xyz_rot.z += center.z;

    float3 direction = hit_xyz_rot;
    direction.x -= origin.x;
    direction.y -= origin.y;
    direction.z -= origin.z;

    float mag = sqrt( direction.x*direction.x + direction.y*direction.y + direction.z*direction.z );
    direction.x = direction.x/mag;
    direction.y = direction.y/mag;
    direction.z = direction.z/mag;

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
      float T = mag;
      if( T>=t0 && T<=t1 ){ //Ray endpoint is inside box
	d_hit_vol[t] = i;
	return;
      }
    }
  
  }

}

void VoxelIntersection::disableMessages( void ){
  printmessages=false;
}

void VoxelIntersection::enableMessages( void ){
  printmessages=true;
}

void VoxelIntersection::calculatePrimitiveVoxelIntersection( void ){
  calculatePrimitiveVoxelIntersection( context->getAllUUIDs() );
}

void VoxelIntersection::calculatePrimitiveVoxelIntersection( std::vector<uint> UUIDs ){

  if( printmessages ){
    std::cout << "Calculating primitive-voxel intersections..." << std::flush;
  }
    
  std::vector<uint> UUIDs_voxels;
  std::vector<uint> UUIDs_prims;

  UUIDs_voxels.resize(UUIDs.size());
  UUIDs_prims.resize(UUIDs.size());

  size_t Nvoxels = 0;
  size_t Nprims = 0;

  //separate out UUIDs of voxels from planar primitives
  for( size_t u = 0; u<UUIDs.size(); u++ ){
    size_t p = UUIDs[u];

    PrimitiveType type = context->getPrimitiveType(p);

    if( type == PRIMITIVE_TYPE_VOXEL ){
      UUIDs_voxels.at(Nvoxels) = p;
      Nvoxels++;
    }else{
      UUIDs_prims.at(Nprims) = p;
      Nprims++;
    }
  }

  if( Nvoxels==0 ){
    if( printmessages ){
      std::cout << "done. ";
    }
    std::cout << "WARNING: no voxels found in Context, nothing to intersect." << std::endl;
  }else if( Nprims==0 ){
    if( printmessages ){
      std::cout << "done. ";
    }
    std::cout << "WARNING: no planar primitives found in Context, nothing to intersect." << std::endl;
  }

  UUIDs_voxels.resize(Nvoxels);
  UUIDs_prims.resize(Nprims);

  std::map<uint,std::vector<uint> > vint;

  // ---- Determine Primitive positions and copy to GPU ---- //

  const uint N = Nprims;

  float3* d_hit_xyz;
  float3* hit_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_xyz,N*sizeof(float3)) ); //allocate device memory

  //copy scan data into the host buffer
  for( std::size_t r=0; r<N; r++ ){
    hit_xyz[r] = vec3tofloat3( context->getPrimitiveVertices(UUIDs_prims.at(r)).front() );
  }

  //copy from host to device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_hit_xyz, hit_xyz, N*sizeof(float3), cudaMemcpyHostToDevice) );

  // ---- Voxels ---- //

  float3* d_grid_center;
  float3* d_grid_size;
  float* d_grid_rotation;

  const uint Ncells = Nvoxels;

  float3* center = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_center,Ncells*sizeof(float3)) ); //allocate device memory

  float3* size = (float3*)malloc(Ncells * sizeof(float3)); //allocate host memory
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_size,Ncells*sizeof(float3)) ); //allocate device memory

  float* rotation = (float*)malloc(Ncells * sizeof(float)); //allocate host memory
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_grid_rotation,Ncells*sizeof(float)) ); //allocate device memory

  //copy grid data into the host buffer
  for( int c=0; c<Ncells; c++ ){
    center[c] = vec3tofloat3( context->getVoxelCenter(UUIDs_voxels.at(c)) );
    size[c] = vec3tofloat3( context->getVoxelSize(UUIDs_voxels.at(c)) );
    //rotation[c] = voxel->getRotation();
    rotation[c] = 0.f;
    context->setPrimitiveData( UUIDs_voxels.at(c), "inside_UUIDs", HELIOS_TYPE_UINT, 0, NULL );
  }

  //copy from host to device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_grid_center, center, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_grid_size, size, Ncells*sizeof(float3), cudaMemcpyHostToDevice) );
  CUDA_CHECK_ERROR( cudaMemcpy(d_grid_rotation, rotation, Ncells*sizeof(float), cudaMemcpyHostToDevice) );

  free(hit_xyz);
  free(center);
  free(size);
  free(rotation);

  // Result buffer
  int* hit_vol = (int*)malloc( N * sizeof(int));
  int* d_hit_vol;
  CUDA_CHECK_ERROR( cudaMalloc(&d_hit_vol,N*sizeof(int)) );

  dim3 dimBlock( 64, 1 );
  dim3 dimGrid( ceil(N/64.f) );
  insideVolume_vi <<< dimGrid, dimBlock >>>( N, d_hit_xyz, Ncells, d_grid_size, d_grid_center, d_grid_rotation, d_hit_vol );

  CUDA_CHECK_ERROR( cudaPeekAtLastError() );
  CUDA_CHECK_ERROR( cudaDeviceSynchronize() );
  
  CUDA_CHECK_ERROR( cudaMemcpy(hit_vol, d_hit_vol, N*sizeof(int), cudaMemcpyDeviceToHost) );

  for( std::size_t r=0; r<N; r++ ){
    if( hit_vol[r]>=0 ){
      vint[UUIDs_voxels.at(hit_vol[r])].push_back(UUIDs_prims.at(r));
    }
  }

  for( std::map<uint,std::vector<uint> >::iterator it = vint.begin(); it!=vint.end(); ++it ){
    uint UUID = it->first;
    size_t s = vint.at(UUID).size();
    context->setPrimitiveData( UUID, "inside_UUIDs", HELIOS_TYPE_UINT, s, &vint.at(UUID)[0] );
  }

  free(hit_vol);

  CUDA_CHECK_ERROR( cudaFree(d_hit_vol) );
  CUDA_CHECK_ERROR( cudaFree(d_hit_xyz) );
  CUDA_CHECK_ERROR( cudaFree(d_grid_center) );
  CUDA_CHECK_ERROR( cudaFree(d_grid_size) );

  
  if( printmessages ){
    std::cout << "done." << std::endl;
  }
    
}
