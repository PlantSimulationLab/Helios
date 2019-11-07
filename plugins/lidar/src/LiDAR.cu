/** \file "LiDAR.cu" CUDA-related LiDAR plug-in functions 
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

#include "LiDAR.cuh"
#include "LiDAR.h"
#include "AerialLiDAR.h"

__host__ __device__ float2 operator+(const float2 &a, const float2 &b) {

  return make_float2(a.x+b.x, a.y+b.y);

}

__host__ __device__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__host__ __device__ float2 operator-(const float2 &a, const float2 &b) {

  return make_float2(a.x-b.x, a.y-b.y);

}

__host__ __device__ float3 operator-(const float3 &a, const float3 &b) {

  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}

// __host__ __device__ float3 operator-(const float3 &a){
  
//   return make_float3(-a.x,-a.y,-a.z);

// }

__host__ __device__ float operator*(const float2 &a, const float2 &b) {

  return a.x*b.x + a.y*b.y;

}

__host__ __device__ float operator*(const float3 &a, const float3 &b) {

  return a.x*b.x + a.y*b.y + a.z*b.z;

}

__host__ __device__ float3 operator*(const float3 &a, const float& b) {
  float3 v = a;
  v.x *= b;
  v.y *= b;
  v.z *= b;
  return v;
}

__host__ __device__ float3 operator/(const float3 &a, const float& b) {
  float3 v = a;
  v.x /= b;
  v.y /= b;
  v.z /= b;
  return v;
}

__device__ float3 d_rotatePoint(const float3 &position, const float &theta, const float &phi) {

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

__host__ __device__ float acos_safe( float x ){
  if (x < -1.0) x = -1.0 ;
  else if (x > 1.0) x = 1.0 ;
  return acosf(x) ;
}

__host__ int randu( int imin, int imax  ){

  float ru = float(rand()) / float(RAND_MAX + 1.); 

  if( imin==imax || imin>imax ){
    return imin;
  }else{
    return imin + round(float(imax-imin)*ru);
  }
    
}

void LiDARcloud::calculateHitGridCellGPU( void ){

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
  for( std::size_t r=0; r<getHitCount(); r++ ){
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

  dim3 dimBlock( 64, 1 );
  dim3 dimGrid( ceil(total_hits/64.f) );
  insideVolume <<< dimGrid, dimBlock >>>( total_hits, d_hit_xyz, getGridCellCount(), d_grid_size, d_grid_center, d_grid_anchor, d_grid_rotation, d_hit_vol );

  CUDA_CHECK_ERROR( cudaPeekAtLastError() );
  CUDA_CHECK_ERROR( cudaDeviceSynchronize() );
  
  CUDA_CHECK_ERROR( cudaMemcpy(hit_vol, d_hit_vol, total_hits*sizeof(int), cudaMemcpyDeviceToHost) );

  for( std::size_t r=0; r<getHitCount(); r++ ){
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

void LiDARcloud::sourcesInsideGridCellGPU( void ){

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

void LiDARcloud::calculateLeafAreaGPU( void ){
  calculateLeafAreaGPU( 1 );
}

void LiDARcloud::calculateLeafAreaGPU( const int minVoxelHits ){

  if( printmessages ){
    std::cout << "Calculating leaf area..." << std::endl;
  }
    
  if( !triangulationcomputed ){
    std::cerr << "ERROR (calculateLeafAreaGPU): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( !hitgridcellcomputed ){
    calculateHitGridCellGPU();
  }
  
  const uint Nscans = getScanCount();
  const uint Ncells = getGridCellCount();

  //variable aggregates over all scans where we just keep tacking hits on the end for all scans
  std::vector<std::vector<float> > dr_agg; //dr is path length through grid cell
  dr_agg.resize(Ncells);
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
    float3 xyz;

    //populate misses
    for( int j=0; j<Np; j++ ){
      for( int i=0; i<Nt; i++ ){
    	float3 direction = vec3tofloat3(sphere2cart(scans.at(s).rc2direction(i,j)));
    	xyz = origin + direction*10000.f;
  	
    	scan_xyz[j*Nt+i] = xyz;
      }
    }
    
    float3* d_scan_xyz;
    CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nmisses*sizeof(float3)) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nmisses*sizeof(float3), cudaMemcpyHostToDevice) );

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
      intersectGridcell <<< dimGrid, dimBlock >>>( Nmissesbb, origin, d_scan_xyz, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after );

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

    for( size_t r=0; r<getHitCount(); r++ ){
      if( getHitScanID(r)==s ){
	this_scan_xyz.push_back( getHitXYZ(r) );
      }
    }

    size_t Nhits = this_scan_xyz.size();

    free( scan_xyz );
    scan_xyz = (float3*)malloc( Nhits*sizeof(float3) );
   
    for( size_t r=0; r<Nhits; r++ ){
      scan_xyz[r] = vec3tofloat3(this_scan_xyz.at(r));
    }

    CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
    CUDA_CHECK_ERROR( cudaMalloc((float3**)&d_scan_xyz, Nhits*sizeof(float3)) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_scan_xyz, scan_xyz, Nhits*sizeof(float3), cudaMemcpyHostToDevice) );

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
      dimGrid = make_uint3( ceil(float(Nhits)/dimBlock.x), 1, 1  );
      
      if( dimBlock.x==0 && dimGrid.x==0 ){
	continue;
      }
      
      intersectGridcell <<< dimGrid, dimBlock >>>( Nhits, origin, d_scan_xyz, center, anchor, size, rotation, d_dr, d_hit_before, d_hit_after );

      cudaDeviceSynchronize();
      CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here

      //copy results back to host
      CUDA_CHECK_ERROR( cudaMemcpy( hit_before, d_hit_before, sizeof(float), cudaMemcpyDeviceToHost));
      
      hit_before_agg.at(c) += *hit_before;

    }
    
    free( scan_xyz );
    free( dr );
    free( hit_before );
    free( hit_after );
    CUDA_CHECK_ERROR( cudaFree(d_scan_xyz) );
    CUDA_CHECK_ERROR( cudaFree(d_dr) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_before) );
    CUDA_CHECK_ERROR( cudaFree(d_hit_after) );

 
  }//end scan loop

  //----------- Calculate number of hits in voxels -------------- //

  //figure out hits for all scans
  for( size_t r=0; r<getHitCount(); r++ ){
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
    }else if( hit_after_agg[v]-hit_before_agg[v]<0 ){
      if( printmessages ){
	std::cout << "Negative number of rays reaching cell " << v << ": " << hit_after_agg[v] << " " << hit_before_agg[v] << std::endl;
      }
      setCellLeafArea(0,v);
      continue;
    }else if( hit_inside_agg[v]<minVoxelHits ){
      if( printmessages ){
	std::cout << "Not enough hits in voxel: " << hit_inside_agg[v] << " < " << minVoxelHits << std::endl;
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

__global__ void insideVolume( const uint Nhits, const float3* d_hit_xyz, const uint Ngridcells, const float3* d_grid_size, const float3* d_grid_center, const float3* d_grid_anchor, const float* d_grid_rotation, int* d_hit_vol ){
  
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

__global__ void intersectBoundingBox( const size_t scanSize, const float3 origin, const float3* d_scan_xyz, const float3 bbcenter, const float3 bbsize, uint* d_boundingbox_hit ){
  
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

__global__ void intersectGridcell( const size_t Nhitsbb, const float3 origin, float3* d_scan_xyz, const float3 center, const float3 anchor, const float3 size, const float rotation, float* d_dr, float* hit_before, float* hit_after ){
  
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

      atomicAdd( hit_after, sin(acos_safe(dz)) );

      d_dr[ idx ] = fabs(t1-t0);
	
    }else if( t<t0 ){ //hit lies before the volume

      atomicAdd( hit_before, sin(acos_safe(dz)) );
            
    }
    
  }

}

void LiDARcloud::trunkReconstruction( const helios::vec3 box_center, const helios::vec3 box_size, const float Lmax, const float max_aspect_ratio ){

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

  for( uint s=0; s<getScanCount(); s++ ){
    
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


void LiDARcloud::syntheticScan( helios::Context* context, const char* xml_file ){

  if( printmessages ){
    std::cout << "Performing synthetic LiDAR scan..." << std::endl;
  }

  if( printmessages ){
    disableMessages();
    loadXML(xml_file);
    enableMessages();
  }else{
    loadXML(xml_file);
  }
  
  LiDARcloud synthscan;

  if( !printmessages ){
    synthscan.disableMessages();
  }
    
  //Load the synthetic scan metedata
  synthscan.loadXML(xml_file);

  //Determine bounding box for Context geometry
  helios::vec2 xbounds, ybounds, zbounds;
  context->getDomainBoundingBox(xbounds,ybounds,zbounds);

  float3 bb_center = make_float3(xbounds.x+0.5*(xbounds.y-xbounds.x),ybounds.x+0.5*(ybounds.y-ybounds.x),zbounds.x+0.5*(zbounds.y-zbounds.x));
  float3 bb_size = make_float3(xbounds.y-xbounds.x,ybounds.y-ybounds.x,zbounds.y-zbounds.x);

  //get geometry information and copy to GPU

  size_t c=0;

  //----- PATCHES ----- //

  //figure out how many patches
  size_t Npatches = 0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_PATCH ){
      Npatches++;
    }
  }

  float3* patch_vertex0 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory
  float3* patch_vertex1 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory
  float3* patch_vertex2 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory
  float3* patch_vertex3 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory

  c=0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_PATCH ){
      std::vector<helios::vec3> verts = prim->getVertices();
      patch_vertex0[c] = vec3tofloat3(verts.at(0));
      patch_vertex1[c] = vec3tofloat3(verts.at(1));
      patch_vertex2[c] = vec3tofloat3(verts.at(2));
      patch_vertex3[c] = vec3tofloat3(verts.at(3));
      c++;
    }
  }
  
  float3* d_patch_vertex0;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex0,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex0, patch_vertex0, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_patch_vertex1;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex1,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex1, patch_vertex1, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_patch_vertex2;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex2,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex2, patch_vertex2, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_patch_vertex3;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex3,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex3, patch_vertex3, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );

  /** \todo Need to check if patch has a transparency channel, and if so get and store it. */

  //----- TRIANGLES ----- //

  //figure out how many triangles
  size_t Ntriangles = 0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_TRIANGLE ){
      Ntriangles++;
    }
  }

  float3* tri_vertex0 = (float3*)malloc(Ntriangles * sizeof(float3)); //allocate host memory
  float3* tri_vertex1 = (float3*)malloc(Ntriangles * sizeof(float3)); //allocate host memory
  float3* tri_vertex2 = (float3*)malloc(Ntriangles * sizeof(float3)); //allocate host memory

  c=0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_TRIANGLE ){
      std::vector<helios::vec3> verts = prim->getVertices();
      tri_vertex0[c] = vec3tofloat3(verts.at(0));
      tri_vertex1[c] = vec3tofloat3(verts.at(1));
      tri_vertex2[c] = vec3tofloat3(verts.at(2));
      c++;
    }
  }
  
  float3* d_tri_vertex0;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_vertex0,Ntriangles*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_tri_vertex0, tri_vertex0, Ntriangles*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_tri_vertex1;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_vertex1,Ntriangles*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_tri_vertex1, tri_vertex1, Ntriangles*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_tri_vertex2;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_vertex2,Ntriangles*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_tri_vertex2, tri_vertex2, Ntriangles*sizeof(float3), cudaMemcpyHostToDevice) );

  // //----- ALPHAMASKS ----- //

  // //figure out how many alpha masks
  // size_t Namasks = 0;
  // for( int p=0; p<context->getPrimitiveCount(); p++ ){
  //   helios::Primitive* prim = context->getPrimitivePointer(p);
  //   if( prim->getType() == helios::PRIMITIVE_TYPE_ALPHAMASK ){
  //     Namasks++;
  //   }
  // }

  // float3* amask_vertex0 = (float3*)malloc(Namasks * sizeof(float3)); //allocate host memory
  // float3* amask_vertex1 = (float3*)malloc(Namasks * sizeof(float3)); //allocate host memory
  // float3* amask_vertex2 = (float3*)malloc(Namasks * sizeof(float3)); //allocate host memory
  // float3* amask_vertex3 = (float3*)malloc(Namasks * sizeof(float3)); //allocate host memory

  // std::vector<std::vector<bool> > maskdata;

  // c=0;
  // for( int p=0; p<context->getPrimitiveCount(); p++ ){
  //   helios::Primitive* prim = context->getPrimitivePointer(p);
  //   if( prim->getType() == helios::PRIMITIVE_TYPE_ALPHAMASK ){
  //     //vertices
  //     std::vector<helios::vec3> verts = prim->getVertices();
  //     amask_vertex0[c] = vec3tofloat3(verts.at(0));
  //     amask_vertex1[c] = vec3tofloat3(verts.at(1));
  //     amask_vertex2[c] = vec3tofloat3(verts.at(2));
  //     amask_vertex3[c] = vec3tofloat3(verts.at(3));
  //     //mask data
  //     if( c==0 ){
  // 	maskdata = context->getAlphaMaskPointer(p)->getMaskData();
  //     }
      
  //     c++;
  //   }
  // }
  
  // float3* d_amask_vertex0;
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_amask_vertex0,Namasks*sizeof(float3)) ); //allocate device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_amask_vertex0, amask_vertex0, Namasks*sizeof(float3), cudaMemcpyHostToDevice) );
  // float3* d_amask_vertex1;
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_amask_vertex1,Namasks*sizeof(float3)) ); //allocate device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_amask_vertex1, amask_vertex1, Namasks*sizeof(float3), cudaMemcpyHostToDevice) );
  // float3* d_amask_vertex2;
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_amask_vertex2,Namasks*sizeof(float3)) ); //allocate device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_amask_vertex2, amask_vertex2, Namasks*sizeof(float3), cudaMemcpyHostToDevice) );
  // float3* d_amask_vertex3;
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_amask_vertex3,Namasks*sizeof(float3)) ); //allocate device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_amask_vertex3, amask_vertex3, Namasks*sizeof(float3), cudaMemcpyHostToDevice) );

  // int2 masksize;
  // if( Namasks==0 ){
  //   masksize = make_int2(0,0);
  // }else{
  //   masksize = make_int2( maskdata.size(), maskdata.at(0).size() );
  // }
  // bool* amask_maskdata = (bool*)malloc( masksize.x*masksize.y * sizeof(bool));
  // for( int j=0; j<masksize.y; j++ ){
  //   for( int i=0; i<masksize.x; i++ ){
  //     amask_maskdata[i+j*masksize.x] = maskdata.at(i).at(j);
  //   }
  // }
  // bool* d_amask_maskdata;
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_amask_maskdata,masksize.x*masksize.y*sizeof(bool)) ); //allocate device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_amask_maskdata, amask_maskdata, masksize.x*masksize.y*sizeof(bool), cudaMemcpyHostToDevice) );
  // int* amask_masksize = (int*)malloc( 2 * sizeof(int));
  // amask_masksize[0] = masksize.x;
  // amask_masksize[1] = masksize.y;
  // int* d_amask_masksize;
  // CUDA_CHECK_ERROR( cudaMalloc((void**)&d_amask_masksize,2*sizeof(int)) ); //allocate device memory
  // CUDA_CHECK_ERROR( cudaMemcpy(d_amask_masksize, amask_masksize, 2*sizeof(int), cudaMemcpyHostToDevice) );

  for( int s=0; s<synthscan.getScanCount(); s++ ){

    float3 scan_origin = vec3tofloat3(synthscan.getScanOrigin(s));

    int Ntheta = synthscan.getScanSizeTheta(s);
    int Nphi = synthscan.getScanSizePhi(s);

    helios::vec2 thetarange = synthscan.getScanRangeTheta(s);
    float thetamin = thetarange.x;
    float thetamax = thetarange.y;
    helios::vec2 phirange = synthscan.getScanRangePhi(s);
    float phimin = phirange.x;
    float phimax = phirange.y;
    
    std::vector<helios::vec3> raydir;
    raydir.resize(Ntheta*Nphi);
  
    for (uint j=0; j<Nphi; j++ ){
      float phi = phimax+float(j)*(phimax-phimin)/float(Nphi);
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
    }

    //make a new array of ray directions for rays that hit bounding box
    float3* direction = (float3*)malloc(N * sizeof(float3)); //allocate host memory

    int count=0;
    for( int i=0; i<Ntheta*Nphi; i++ ){
      if( bb_hit[i]==1 ){
	direction[count] = vec3tofloat3(raydir.at(i));
	count++;
      }
    }    
    free(bb_hit);

    float3* d_raydir;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_raydir,N*sizeof(float3)) ); //allocate device memory
    CUDA_CHECK_ERROR( cudaMemcpy(d_raydir, direction, N*sizeof(float3), cudaMemcpyHostToDevice) );

    //Distance to intersection
    float* d_hit_t;
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_t,N*sizeof(float)) ); //allocate device memory
    float* hit_t = (float*)malloc(N * sizeof(float)); //allocate host memory
    for( int i=0; i<N; i++ ){
      hit_t[i] = 1e6;
    }
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_t, hit_t, N*sizeof(float), cudaMemcpyHostToDevice) );

    dimGrid = make_uint3( ceil(float(N)/float(dimBlock.x)), 1, 1 );

    //---- patch kernel ----//
    intersectPatches<<< dimGrid, dimBlock >>>( N, scan_origin, d_raydir, Npatches, d_patch_vertex0, d_patch_vertex1, d_patch_vertex2, d_patch_vertex3, d_hit_t );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here
      
    //---- triangle kernel ----//
    intersectTriangles<<< dimGrid, dimBlock >>>( N, scan_origin, d_raydir, Ntriangles, d_tri_vertex0, d_tri_vertex1, d_tri_vertex2, d_hit_t );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here
    
    //copy back
    CUDA_CHECK_ERROR( cudaMemcpy(hit_t, d_hit_t, N*sizeof(float), cudaMemcpyDeviceToHost) );

    float hit_in = 0;
    size_t Nhits = 0;
    for( size_t r=0; r<N; r++ ){
      float t = hit_t[r];
      if( t<1e6 ){
    	helios::vec3 dir = helios::make_vec3(direction[r].x,direction[r].y,direction[r].z);
    	helios::vec3 origin = helios::make_vec3(scan_origin.x,scan_origin.y,scan_origin.z);
    	helios::vec3 p = origin+dir*t;
    	addHitPoint( s, p, helios::cart2sphere(dir), helios::RGB::red );
    	Nhits++;
	helios::SphericalCoord dir_s = cart2sphere(dir);
	hit_in += sin(dir_s.zenith);
      }
    }

    CUDA_CHECK_ERROR( cudaFree(d_hit_t) );
    CUDA_CHECK_ERROR( cudaFree(d_raydir) );
    free(hit_xyz);
    free(direction);
    free(hit_t);

    if( printmessages ){
      std::cout << "Created synthetic scan #" << s << " with " << Nhits << " hit points." << std::endl;
    }
    
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }
  
}

void LiDARcloud::calculateSyntheticLeafArea( helios::Context* context ){

  float3* d_prim_xyz;

  const uint N = context->getPrimitiveCount();
  
  float3* prim_xyz = (float3*)malloc(N * sizeof(float3)); //allocate host memory
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_prim_xyz,N*sizeof(float3)) ); //allocate device memory

  //copy scan data into the host buffer
  for( std::size_t p=0; p<N; p++ ){
    std::vector<helios::vec3> verts = context->getPrimitivePointer(p)->getVertices();
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
      total_area.at(gridcell) += context->getPrimitivePointer(p)->getArea();
      context->setPrimitiveData(p,"gridCell",helios::HELIOS_TYPE_UINT,1,&gridcell);
    }
  }

  std::ofstream file;

  file.open("../output/synthetic_leaf_areas.txt");
  
  for( int v=0; v<Ncells; v++ ){
    file << total_area.at(v) << std::endl;
  }

  file.close();
  
  free(prim_vol);

  CUDA_CHECK_ERROR( cudaFree(d_prim_vol) );
  CUDA_CHECK_ERROR( cudaFree(d_prim_xyz) );
  CUDA_CHECK_ERROR( cudaFree(d_grid_center) );
  CUDA_CHECK_ERROR( cudaFree(d_grid_anchor) );
  CUDA_CHECK_ERROR( cudaFree(d_grid_size) );
  CUDA_CHECK_ERROR( cudaFree(d_grid_rotation) );  

}

__global__ void intersectPatches( const size_t N, const float3 origin, float3* d_raydir, const int Npatches, float3* d_patch_vertex0, float3* d_patch_vertex1, float3* d_patch_vertex2, float3* d_patch_vertex3, float* d_hit_t ){

  size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

  if( idx>=N ){
    return;
  }

  float3 raydir = d_raydir[idx];

  float tmin = d_hit_t[idx];
  for( int p=0; p<Npatches; p++ ){

    float3 v0 = d_patch_vertex0[p];
    float3 v1 = d_patch_vertex1[p];
    float3 v2 = d_patch_vertex2[p];
    float3 v3 = d_patch_vertex3[p];

    float3 anchor = v0;
    float3 normal = normalize( cross( v1-v0, v2-v0 ) );

    float3 a = v1-v0;
    float3 b = v3-v0;
    
    float t = ((anchor - origin)*normal) / (raydir*normal); 
    
    if( t==t && t>1e-8 && t<tmin ){
			
      float3 p = origin + raydir * t;
      float3 d = p - anchor;
      
      float ddota = d*a;
      
      if (ddota > 0.0 && ddota < a*a ){
	
	float ddotb = d*b;
	
	if (ddotb > 0.0 && ddotb < b*b ){
	  tmin = t;
	}	
	
      }
    }
  }

  d_hit_t[idx] = tmin;
  
}

__global__ void intersectTriangles( const size_t N, const float3 origin, float3* d_raydir, const int Ntriangles, float3* d_tri_vertex0, float3* d_tri_vertex1, float3* d_tri_vertex2, float* d_hit_t ){

  size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

  if( idx>=N ){
    return;
  }

  float3 raydir = d_raydir[idx];

  float tmin = d_hit_t[idx];
  for( int p=0; p<Ntriangles; p++ ){

    float3 v0 = d_tri_vertex0[p];
    float3 v1 = d_tri_vertex1[p];
    float3 v2 = d_tri_vertex2[p];

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
	  tmin = t;
	}	
      
      }
    }
  }

  d_hit_t[idx] = tmin;
  
}

// __global__ void intersectAlphaMasks( const size_t N, const float3 origin, float3* d_raydir, const int Namasks, float3* d_patch_vertex0, float3* d_patch_vertex1, float3* d_patch_vertex2, float3* d_patch_vertex3, bool* d_amask_maskdata, int* d_amask_masksize, float* d_hit_t ){

//   size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

//   if( idx>=N ){
//     return;
//   }

//   float3 raydir = d_raydir[idx];

//   float tmin = d_hit_t[idx];
//   for( int p=0; p<Namasks; p++ ){

//     float3 v0 = d_patch_vertex0[p];
//     float3 v1 = d_patch_vertex1[p];
//     float3 v2 = d_patch_vertex2[p];
//     float3 v3 = d_patch_vertex3[p];

//     float3 anchor = v0;
//     float3 normal = normalize( cross( v1-v0, v2-v0 ) );

//     float3 a = v1-v0;
//     float3 b = v3-v0;
    
//     float t = ((anchor - origin)*normal) / (raydir*normal); 
    
//     if( t==t && t>1e-8 && t<tmin ){
			
//       float3 p = origin + raydir * t;
//       float3 d = p - anchor;
      
//       float ddota = d*a;
      
//       if (ddota > 0.0 && ddota < a*a ){
	
// 	float ddotb = d*b;
	
// 	if (ddotb > 0.0 && ddotb < b*b ){

// 	  //use mask
// 	  float amag = sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
// 	  float bmag = sqrt(b.x*b.x+b.y*b.y+b.z*b.z);
// 	  float2 uv = make_float2( ddota/amag/amag, 1.f-ddotb/bmag/bmag );
// 	  int2 sz;
// 	  sz.x = d_amask_masksize[0];
// 	  sz.y = d_amask_masksize[1];
// 	  int2 ind = make_int2( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)) );
// 	  int i = ind.x + ind.y*sz.x;

// 	  if( d_amask_maskdata[i] ){
// 	    tmin = t;
// 	  }
// 	}	
	
//       }
//     }
//   }

//   d_hit_t[idx] = tmin;
  
// }

//*************************************************************//
//              AERIAL LIDAR DEFINITIONS                      //
//************************************************************//

void AerialLiDARcloud::calculateLeafAreaGPU( const float Gtheta ){
  calculateLeafAreaGPU( Gtheta, 1 );
}

void AerialLiDARcloud::calculateLeafAreaGPU( const float Gtheta, const int minVoxelHits ){

  if( printmessages ){
    std::cout << "Calculating leaf area..." << std::endl;
  }
   
  if( !hitgridcellcomputed ){
    calculateHitGridCellGPU();
  }
  
  const uint Nscans = getScanCount();
  const uint Ncells = getGridCellCount();

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
      cell_center[c] = vec3tofloat3(getCellCenter(c));
      cell_anchor[c] = vec3tofloat3(getCellGlobalAnchor(c));
      cell_size[c] = vec3tofloat3(getCellSize(c));
      cell_rotation[c] = getCellRotation(c);

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

    float dr_bar = getCellSize(v).z;
    
    float P = 1.f-float(hit_inside_agg[v])/float(hit_denom_agg[v]);

    float a;
    if( hit_inside_agg[v]<minVoxelHits ){
      a = 0.f;
    }else if( P<0.1 ){

      dr_bar = dr_hit_agg.at(v);
      
      a = 1.f/dr_bar/Gtheta;
    }else{
      a = -log(P)/dr_bar/Gtheta;
    }
      
    helios::vec3 gridsize = getCellSize(v);
    setCellLeafArea(a*gridsize.x*gridsize.y*gridsize.z,v);

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

helios::vec4 AerialLiDARcloud::RANSAC( const int maxIter, const float threshDist, const float inlierRatio, const std::vector<helios::vec3>& hits, std::vector<bool>& inliers ){

  int N = hits.size();

  //trying to find a model in the form of Ax+By+Cz+D=0
  //bestModel.x = A, bestModel.y = B, bestModel.z = C, bestModel.w = D 
  helios::vec4 bestModel;
  float bestError = 1e20;
  
  for( int iter=0; iter<maxIter; iter++ ){

    helios::vec4 maybeModel;
	
    //generate 3 random points
    int i0 = randu(0,N-1);
    int i1 = randu(0,N-1);
    int i2 = randu(0,N-1);

    //get x,y,z coordinates for these points
    helios::vec3 x0 = hits.at(i0);
    helios::vec3 x1 = hits.at(i1);
    helios::vec3 x2 = hits.at(i2);
    
    //determine 'maybeModel' for these three points
    helios::vec3 n = cross( x2-x0, x1-x0 );
    maybeModel.x = n.x;
    maybeModel.y = n.y;
    maybeModel.z = n.z;
    
    maybeModel.w = -(n*x2);

    //loop over all other points
    float errsum = 0.f;
    int Ninlier = 0;
    for( int p=0; p<N; p++ ){
      
      if( p!=i0 && p!=i1 && p!=i2 ){

	helios::vec3 point = hits.at(p);
	    
	float zplane = -(maybeModel.x*point.x+maybeModel.y*point.y+maybeModel.w)/maybeModel.z;
	float zerr = fabs( zplane - point.z ); //NOTE: since we know the plane will be somewhat close to horizontal, we're only using the vertical distance to calculate the error of fit.

	if( zerr<=threshDist ){
	  errsum += zerr;
	  Ninlier += 1;
	}
	
      }

    }

    //check if this model is better than the current 'best' model
    if( errsum/float(Ninlier)<bestError && float(Ninlier)/float(N)>inlierRatio ){

      //update the 'best' model
      bestModel = maybeModel;
      bestError = errsum/float(Ninlier);
      
    }

  }//end iterations loop (iter)

  //separate inliers and outliers based on 'best' model
  inliers.resize(N,0);
  for( int p=0; p<N; p++ ){

    helios::vec3 point = hits.at(p);
	    
    float zplane = -(bestModel.x*point.x+bestModel.y*point.y+bestModel.w)/bestModel.z;
    float zerr = fabs( zplane - point.z );
    
    if( zerr<=threshDist ){
      inliers.at(p) = true;
    }
      
  }
  
  return bestModel;
  
}

void AerialLiDARcloud::generateHeightModel( const int maxIter, const float threshDist_ground, const float inlierRatio_ground, const float threshDist_vegetation, const float inlierRatio_vegetation ){

  if( printmessages ){
    std::cout << "Calculating ground and vegetation heights..." << std::endl;
  }
   
  if( !hitgridcellcomputed ){
    calculateHitGridCellGPU();
  }

  std::srand(time(NULL));

  const uint Nscans = getScanCount();
  const uint Ncells = getGridCellCount();
  size_t Nhits = getHitCount();

  if( Ncells==0 ){
    std::cerr << "ERROR (generateHeightModel): There are no grid cells in the point cloud." << std::endl;
    exit(EXIT_FAILURE);
  }else if( Nhits==0 ){
    std::cerr << "ERROR (generateHeightModel): There are no hits in the point cloud." << std::endl;
    exit(EXIT_FAILURE);
  }

  //aggregate all points in a given x,y column

  std::vector<std::vector<std::vector<int> > > first_hits_ind; //first index are hit indices, second index is grid cell in x-dir, third index is grid cell in y-dir
  std::vector<std::vector<std::vector<int> > > last_hits_ind; //first index are hit indices, second index is grid cell in x-dir, third index is grid cell in y-dir
  std::vector<std::vector<std::vector<float> > > max_height_xy;
  
  helios::int3 gsize3 = getGlobalGridCount(0);

  first_hits_ind.resize(gsize3.y);
  last_hits_ind.resize(gsize3.y);
  for( int j=0; j<gsize3.y; j++ ){
    first_hits_ind.at(j).resize(gsize3.x);
    last_hits_ind.at(j).resize(gsize3.x);
  }

  max_height_xy.resize(3);
  for( int k=0; k<max_height_xy.size(); k++ ){
    max_height_xy.at(k).resize(gsize3.y);
    for( int j=0; j<gsize3.y; j++ ){
      max_height_xy.at(k).at(j).resize(gsize3.x);
      for( int i=0; i<gsize3.x; i++ ){
	max_height_xy.at(k).at(j).at(i) = -9999999;
      }
    }
  }

  //do an initial pass to see whether "target_index" starts at 0 or 1
  float target_offset = 0;
  for( size_t r=0; r<Nhits; r++ ){

    if( doesHitDataExist(r,"target_index") ){
      float target_index = getHitData(r,"target_index");
      if( target_index==0 ){
	target_offset=1;
	break;
      }
    }

  }

  for( size_t r=0; r<Nhits; r++ ){

    setHitData( r, "ground_flag", 0 );

    if( !doesHitDataExist(r,"target_index") || !doesHitDataExist(r,"target_count") ){
	continue;
    }

    int cell = getHitGridCell(r);

    if( cell>=0 ){

      float target_index = getHitData(r,"target_index")+target_offset;
      float target_count = getHitData(r,"target_count");
      
      helios::int3 index3 = getGlobalGridIndex(cell);

      float height = getHitXYZ(r).z;

      for( int k=max_height_xy.size()-1; k>=0; k-- ){
	if( height>max_height_xy.at(k).at(index3.y).at(index3.x) ){//&& target_index==1 ){
	  max_height_xy.at(k).at(index3.y).at(index3.x) = height;
	  break;
	}
      }

      if( target_index==target_count ){
	last_hits_ind.at(index3.y).at(index3.x).push_back( r );
      }else if( target_index==1 ){
	first_hits_ind.at(index3.y).at(index3.x).push_back( r );
      }

    }

  }

  //apply RANSAC to first and last hits for each x,y colum to determine ground and vegetation height

  std::vector<std::vector<float> > ground_height_xy, canopy_height_xy;

  ground_height_xy.resize(gsize3.y);
  canopy_height_xy.resize(gsize3.y);

  for( int j=0; j<gsize3.y; j++ ){
    
    ground_height_xy.at(j).resize(gsize3.x);
    canopy_height_xy.at(j).resize(gsize3.x);
    
    for( int i=0; i<gsize3.x; i++ ){

      float canopy_height = nanf("");
      float ground_height = nanf("");

      ground_height_xy.at(j).at(i) = ground_height;
      canopy_height_xy.at(j).at(i) = canopy_height;

      if( last_hits_ind.at(j).at(i).size()==0 ){ //no hits in this column
	continue;
      }

      int cell = getHitGridCell( last_hits_ind.at(j).at(i).front() );

      helios::vec3 center = getCellCenter( cell ); 

      //compute ground height
      if( last_hits_ind.at(j).at(i).size()>3 ){
	
	std::vector<helios::vec3> last_hits_xyz;
	last_hits_xyz.resize( last_hits_ind.at(j).at(i).size() );
	for( int r=0; r<last_hits_xyz.size(); r++ ){
	  last_hits_xyz.at(r) = getHitXYZ( last_hits_ind.at(j).at(i).at(r) );
	}   
	
	//ground planes
	std::vector<bool> ground_inliers;
	helios::vec4 groundModel = RANSAC( maxIter, threshDist_ground, inlierRatio_ground, last_hits_xyz, ground_inliers );
	
	for( int r=0; r<ground_inliers.size(); r++ ){
	  if( ground_inliers.at(r) ){
	    setHitData( last_hits_ind.at(j).at(i).at(r), "ground_flag", float(1) ); 
	  }
	}

	ground_height = -(groundModel.x*center.x+groundModel.y*center.y+groundModel.w)/groundModel.z;

      }

      //compute canopy height
      if( first_hits_ind.at(j).at(i).size()>=3 ){

	std::vector<helios::vec3> first_hits_xyz;
	first_hits_xyz.resize( first_hits_ind.at(j).at(i).size() );
	for( int r=0; r<first_hits_xyz.size(); r++ ){
	  first_hits_xyz.at(r) = getHitXYZ( first_hits_ind.at(j).at(i).at(r) );
	}

	//canopy top planes
	std::vector<bool> canopy_inliers;
	helios::vec4 canopyModel = RANSAC( maxIter, threshDist_vegetation, inlierRatio_vegetation, first_hits_xyz, canopy_inliers );

	canopy_height = -(canopyModel.x*center.x+canopyModel.y*center.y+canopyModel.w)/canopyModel.z;

	if( canopy_height<ground_height ){
	  canopy_height = ground_height;
	}else if( canopy_height>max_height_xy.front().at(j).at(i) ){
	  canopy_height = max_height_xy.front().at(j).at(i);
	}
	  
      }

      ground_height_xy.at(j).at(i) = ground_height;
      canopy_height_xy.at(j).at(i) = canopy_height;
      
    }//end i (x,y) loop  
  }//end j (x,y) loop

  //fill any holes in the ground height model
  std::vector<std::vector<float> > ground_height_filled;
  ground_height_filled.resize(gsize3.y);
  for( int j=0; j<gsize3.y; j++ ){
    ground_height_filled.at(j).insert(ground_height_filled.at(j).begin(), ground_height_xy.at(j).begin(), ground_height_xy.at(j).end() );
  }
  
  for( int j=1; j<gsize3.y-1; j++ ){
    for( int i=1; i<gsize3.x-1; i++ ){

      if( ground_height_xy.at(j).at(i)!=ground_height_xy.at(j).at(i) ){ //NaN

	int count=0;
	float mean=0;
	for( int jj=-1; jj<2; jj+=2 ){
	  for( int ii=-1; ii<2; ii+=2 ){
	    if( ground_height_xy.at(j+jj).at(i+ii)==ground_height_xy.at(j+jj).at(i+ii) ){
	      mean += ground_height_xy.at(j+jj).at(i+ii);
	      count ++;
	    }
	  }
	}

	if( count>0 ){
	  ground_height_filled.at(j).at(i) = mean/float(count);
	}
	  
      }
      
    }
  }

  //remove outliers in the ground height model
  int stencil_half_size = 4;
  float std_fact = 0.5;

  for( int j=0; j<gsize3.y; j++ ){
    for( int i=0; i<gsize3.x; i++ ){
      ground_height_xy.at(j).at(i) = ground_height_filled.at(j).at(i);
    }
  }
  
  for( int j=stencil_half_size; j<gsize3.y-stencil_half_size; j++ ){
    for( int i=stencil_half_size; i<gsize3.x-stencil_half_size; i++ ){

      if( ground_height_filled.at(j).at(i)!=ground_height_filled.at(j).at(i) ){
	continue;
      }

      int count=0;
      float mean=0;
      float var=0;
      for( int jj=-stencil_half_size; jj<=stencil_half_size; jj++ ){
	for( int ii=-stencil_half_size; ii<=stencil_half_size; ii++ ){
	  if( ii==0 || jj==0 ){
	    continue;
	  }

	  if( ground_height_filled.at(j+jj).at(i+ii)==ground_height_filled.at(j+jj).at(i+ii) ){
	    mean += ground_height_filled.at(j+jj).at(i+ii);
	    var += pow( ground_height_filled.at(j+jj).at(i+ii)-ground_height_filled.at(j).at(i), 2);
	    count ++;
	  }
	}
      }

      float std = sqrt(var/float(count));
      mean = mean/float(count);

      if( fabs(ground_height_filled.at(j).at(i)-mean)>std_fact*std ){
            
	float mean_new=0;
	count = 0;
	for( int jj=-stencil_half_size; jj<=stencil_half_size; jj++ ){
	  for( int ii=-stencil_half_size; ii<=stencil_half_size; ii++ ){
	    if( ii==0 || jj==0 ){
	      continue;
	    }
	    if( fabs(ground_height_filled.at(j+jj).at(i+ii)-mean)<std_fact*std ){
	      mean_new += ground_height_filled.at(j+jj).at(i+ii);
	      count ++;
	    }
	  }
	}
        
	if( count>0 ){
	  ground_height_xy.at(j).at(i) = mean_new/float(count);
	}
      }
    }
  }

  //set the values for all grid cells
  for( int k=0; k<gsize3.z; k++ ){
    for( int j=0; j<gsize3.y; j++ ){
      for( int i=0; i<gsize3.x; i++ ){

	uint cell = getCellGlobalIndex( helios::make_int3(i,j,k) );

	setCellGroundHeight( ground_height_xy.at(j).at(i), cell );
	setCellVegetationHeight( canopy_height_xy.at(j).at(i), cell );
	if( max_height_xy.front().at(j).at(i)==-9999999 ){
	  setCellMaximumHitHeight( nanf(""), cell );
	}else{
	  setCellMaximumHitHeight( max_height_xy.front().at(j).at(i), cell );
	}
	
      }
    }
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }

}

void AerialLiDARcloud::alignGridToGround( void ){

  size_t Nc = getGridCellCount();

  if( Nc==0 ){
    std::cout << "WARNING (alignGridToGround): No grid cells added to the point cloud." << std::endl;
    return;
  }
  
  helios::int3 gsize3 = getGlobalGridCount(0);

  float zbase = getCellGlobalAnchor(0).z-0.5*getGlobalGridExtent(0).z;

  int shift_count = 0;
  for( int k=0; k<gsize3.z; k++ ){
    for( int j=0; j<gsize3.y; j++ ){
      for( int i=0; i<gsize3.x; i++ ){

	uint index = getCellGlobalIndex( helios::make_int3(i,j,k) );

	helios::vec3 center = getCellCenter(index);

	if( doesHitDataExist( index, "ground_flag" ) ){

	  float ground_height = getCellGroundHeight( index );

	  setCellCenter( index, helios::make_vec3(center.x,center.y,center.z-zbase+ground_height) );

	  shift_count++;

	}
	
      }
    }
  }

  if( shift_count==0 ){
    std::cout << "WARNING (alignGridToGround): No grid cells were aligned to the ground because ground heights were not set. You probably forgot to run the generateHeightModel() function first." << std::endl;
  }

  hitgridcellcomputed=false;
  
  
}

void AerialLiDARcloud::syntheticScan( helios::Context* context, const char* xml_file ){

  if( printmessages ){
    std::cout << "Performing synthetic LiDAR scan..." << std::endl;
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

  //----- PATCHES ----- //

  //figure out how many patches
  size_t Npatches = 0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_PATCH ){
      Npatches++;
    }
  }

  float3* patch_vertex0 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory
  float3* patch_vertex1 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory
  float3* patch_vertex2 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory
  float3* patch_vertex3 = (float3*)malloc(Npatches * sizeof(float3)); //allocate host memory

  c=0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_PATCH ){
      std::vector<helios::vec3> verts = prim->getVertices();
      patch_vertex0[c] = vec3tofloat3(verts.at(0));
      patch_vertex1[c] = vec3tofloat3(verts.at(1));
      patch_vertex2[c] = vec3tofloat3(verts.at(2));
      patch_vertex3[c] = vec3tofloat3(verts.at(3));
      c++;
    }
  }
  
  float3* d_patch_vertex0;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex0,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex0, patch_vertex0, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_patch_vertex1;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex1,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex1, patch_vertex1, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_patch_vertex2;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex2,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex2, patch_vertex2, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_patch_vertex3;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_patch_vertex3,Npatches*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_patch_vertex3, patch_vertex3, Npatches*sizeof(float3), cudaMemcpyHostToDevice) );

  /** \todo Need to check if patch has a transparency channel, and if so get and store it. */

  //----- TRIANGLES ----- //

  //figure out how many triangles
  size_t Ntriangles = 0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_TRIANGLE ){
      Ntriangles++;
    }
  }

  float3* tri_vertex0 = (float3*)malloc(Ntriangles * sizeof(float3)); //allocate host memory
  float3* tri_vertex1 = (float3*)malloc(Ntriangles * sizeof(float3)); //allocate host memory
  float3* tri_vertex2 = (float3*)malloc(Ntriangles * sizeof(float3)); //allocate host memory

  c=0;
  for( int p=0; p<context->getPrimitiveCount(); p++ ){
    helios::Primitive* prim = context->getPrimitivePointer(p);
    if( prim->getType() == helios::PRIMITIVE_TYPE_TRIANGLE ){
      std::vector<helios::vec3> verts = prim->getVertices();
      tri_vertex0[c] = vec3tofloat3(verts.at(0));
      tri_vertex1[c] = vec3tofloat3(verts.at(1));
      tri_vertex2[c] = vec3tofloat3(verts.at(2));
      c++;
    }
  }
  
  float3* d_tri_vertex0;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_vertex0,Ntriangles*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_tri_vertex0, tri_vertex0, Ntriangles*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_tri_vertex1;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_vertex1,Ntriangles*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_tri_vertex1, tri_vertex1, Ntriangles*sizeof(float3), cudaMemcpyHostToDevice) );
  float3* d_tri_vertex2;
  CUDA_CHECK_ERROR( cudaMalloc((void**)&d_tri_vertex2,Ntriangles*sizeof(float3)) ); //allocate device memory
  CUDA_CHECK_ERROR( cudaMemcpy(d_tri_vertex2, tri_vertex2, Ntriangles*sizeof(float3), cudaMemcpyHostToDevice) );


  for( int s=0; s<synthscan.getScanCount(); s++ ){

    float coneangle = synthscan.getScanConeAngle(s);

    helios::vec3 center = synthscan.getScanCenter(s);

    helios::vec2 extent = synthscan.getScanExtent(s);

    float scandensity = synthscan.getScanDensity(s);

    size_t N = scandensity*extent.x*extent.y;

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
	origin[count] = vec3tofloat3(rayorigin.at(i)-raydir.at(i)*1e5);
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
    CUDA_CHECK_ERROR( cudaMalloc((void**)&d_hit_t,N*sizeof(float)) ); //allocate device memory
    float* hit_t = (float*)malloc(N * sizeof(float)); //allocate host memory
    for( int i=0; i<N; i++ ){
      hit_t[i] = 1e6;
    }
    CUDA_CHECK_ERROR( cudaMemcpy(d_hit_t, hit_t, N*sizeof(float), cudaMemcpyHostToDevice) );

    dimGrid = make_uint3( ceil(float(N)/float(dimBlock.x)), 1, 1 );

    //---- patch kernel ----//
    intersectPatches<<< dimGrid, dimBlock >>>( N, d_origin, d_raydir, Npatches, d_patch_vertex0, d_patch_vertex1, d_patch_vertex2, d_patch_vertex3, d_hit_t );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here
      
    //---- triangle kernel ----//
    intersectTriangles<<< dimGrid, dimBlock >>>( N, d_origin, d_raydir, Ntriangles, d_tri_vertex0, d_tri_vertex1, d_tri_vertex2, d_hit_t );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR( cudaPeekAtLastError() ); //if there was an error inside the kernel, it will show up here
    
    //copy back
    CUDA_CHECK_ERROR( cudaMemcpy(hit_t, d_hit_t, N*sizeof(float), cudaMemcpyDeviceToHost) );

    size_t Nhits = 0;
    for( size_t r=0; r<N; r++ ){
      float t = hit_t[r];
      if( t<1e6 ){
	helios::vec3 dir = helios::make_vec3(direction[r].x,direction[r].y,direction[r].z);
    	helios::vec3 hit_origin = helios::make_vec3(origin[r].x,origin[r].y,origin[r].z);
    	helios::vec3 p = hit_origin+dir*t;
    	addHitPoint( s, p, helios::cart2sphere(dir), helios::RGB::red );
    	Nhits++;
      }
    }

    CUDA_CHECK_ERROR( cudaFree(d_hit_t) );
    CUDA_CHECK_ERROR( cudaFree(d_raydir) );
    CUDA_CHECK_ERROR( cudaFree(d_origin) );
    free(hit_xyz);
    free(direction);
    free(origin);
    free(hit_t);

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

void AerialLiDARcloud::calculateHitGridCellGPU( void ){

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
  for( std::size_t r=0; r<getHitCount(); r++ ){
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

  dim3 dimBlock( 64, 1 );
  dim3 dimGrid( ceil(total_hits/64.f) );
  insideVolume <<< dimGrid, dimBlock >>>( total_hits, d_hit_xyz, getGridCellCount(), d_grid_size, d_grid_center, d_grid_anchor, d_grid_rotation, d_hit_vol );

  CUDA_CHECK_ERROR( cudaPeekAtLastError() );
  CUDA_CHECK_ERROR( cudaDeviceSynchronize() );
  
  CUDA_CHECK_ERROR( cudaMemcpy(hit_vol, d_hit_vol, total_hits*sizeof(int), cudaMemcpyDeviceToHost) );

  for( std::size_t r=0; r<getHitCount(); r++ ){
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

__global__ void intersectGridcell( const size_t Nhitsbb, float3* d_scan_xyz, float3* d_scan_raydir, const size_t Ncells, float3* center, float3* anchor, float3* size, float* rotation, float* d_dr, float* d_dr_hit, float* hit_denom, float* hit_inside ){
  
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

__global__ void intersectBoundingBox( const size_t scanSize, float3* d_hit_origin, const float3* d_scan_xyz, const float3 bbcenter, const float3 bbsize, uint* d_boundingbox_hit ){
  
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

__global__ void intersectPatches( const size_t N, float3* d_origin, float3* d_raydir, const int Npatches, float3* d_patch_vertex0, float3* d_patch_vertex1, float3* d_patch_vertex2, float3* d_patch_vertex3, float* d_hit_t ){

  size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

  if( idx>=N ){
    return;
  }

  float3 raydir = d_raydir[idx];

  float3 origin = d_origin[idx];

  float tmin = d_hit_t[idx];
  for( int p=0; p<Npatches; p++ ){

    float3 v0 = d_patch_vertex0[p];
    float3 v1 = d_patch_vertex1[p];
    float3 v2 = d_patch_vertex2[p];
    float3 v3 = d_patch_vertex3[p];

    float3 anchor = v0;
    float3 normal = normalize( cross( v1-v0, v2-v0 ) );

    float3 a = v1-v0;
    float3 b = v3-v0;
    
    float t = ((anchor - origin)*normal) / (raydir*normal); 
    
    if( t==t && t>1e-8 && t<tmin ){
			
      float3 p = origin + raydir * t;
      float3 d = p - anchor;
      
      float ddota = d*a;
      
      if (ddota > 0.0 && ddota < a*a ){
	
	float ddotb = d*b;
	
	if (ddotb > 0.0 && ddotb < b*b ){
	  tmin = t;
	}	
	
      }
    }
  }

  d_hit_t[idx] = tmin;
  
}

__global__ void intersectTriangles( const size_t N, float3* d_origin, float3* d_raydir, const int Ntriangles, float3* d_tri_vertex0, float3* d_tri_vertex1, float3* d_tri_vertex2, float* d_hit_t ){

  size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

  if( idx>=N ){
    return;
  }

  float3 raydir = d_raydir[idx];

  float3 origin = d_origin[idx];

  float tmin = d_hit_t[idx];
  for( int p=0; p<Ntriangles; p++ ){

    float3 v0 = d_tri_vertex0[p];
    float3 v1 = d_tri_vertex1[p];
    float3 v2 = d_tri_vertex2[p];

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
	  tmin = t;
	}	
      
      }
    }
  }

  d_hit_t[idx] = tmin;
  
}
