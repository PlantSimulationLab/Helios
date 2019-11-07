/** \file "LiDAR.cuh" Header file for CUDA-related LiDAR functions.
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
#include "helios_vector_types.h"

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//! Device function to determine the grid cell in which a hit point resides.
/** \param[in] "Nhits" Total number of hit points
    \param[in] "d_hit_xyz" (x,y,z) coordinates of hit point - resides in GPU memory and has size Nhits
    \param[in] "Ngridcells" Total number of grid cells
    \param[in] "d_grid_size" Size of each grid cell in x-, y-, and z-directions - resides in GPU memory and has size Ngricells
    \param[in] "d_grid_center" (x,y,z) coordinates of each grid cell center point - resides in GPU memory and has size Ngricells
    \param[in] "d_grid_anchor" If grid cell is a subcell of a larger grid that was rotated about its center, d_grid_anchor contains the (x,y,z) coordinates of the points about which the cell was rotated.  If the above conditions are not true, d_grid_anchor is equal to d_grid_center. - resides in GPU memory and has size Ngricells
    \param[in] "d_grid_rotation" angle in radians of grid cell azimuthal rotation - resides in GPU memory and has size Ngricells
    \param[out] "d_hit_vol" Index of grid cell that hit point resides in.  If the hit point does not reside in any grid cells, d_hit_vol[] = -1. - resides in GPU memory and has size Ngricells
*/
__global__ void insideVolume( const uint Nhits, const float3* d_hit_xyz, const uint Ngridcells, const float3* d_grid_size, const float3* d_grid_center, const float3* d_grid_anchor, const float* d_grid_rotation, int* d_hit_vol );

//! Device function to intersect all rays with an axis-aligned bounding box
__global__ void intersectBoundingBox( const size_t scanSize, const float3 origin, const float3* d_hit_xyz, const float3 bbcenter, const float3 bbsize, uint* d_boundingbox_hit );

__global__ void intersectGridcell( const size_t Nhitsbb, const float3 origin, float3* d_scan_xyz, const float3 center, const float3 anchor, const float3 size, const float rotation, float* d_dr, float* hit_before, float* hit_after );


__global__ void intersectPatches( const size_t N, const float3 origin, float3* d_raydir, const int Npatches, float3* d_patch_vertex0, float3* d_patch_vertex1, float3* d_patch_vertex2, float3* d_patch_vertex3, float* d_hit_t );

__global__ void intersectTriangles( const size_t N, const float3 origin, float3* d_raydir, const int Ntriangles, float3* d_tri_vertex0, float3* d_tri_vertex1, float3* d_tri_vertex2, float* d_hit_t );

__global__ void intersectDisks( const size_t N, const float3 origin, float3* d_raydir, const int Ndisks, float3* d_disk_center, float* d_disk_radius, float3* d_disk_normal, float* d_hit_t );

__global__ void intersectAlphaMasks( const size_t N, const float3 origin, float3* d_raydir, const int Namasks, float3* d_patch_vertex0, float3* d_patch_vertex1, float3* d_patch_vertex2, float3* d_patch_vertex3, bool* d_amask_maskdata, int* d_amask_masksize, float* d_hit_t );

/** Function to rotate a 3D vector given spherical angles elevation and azimuth ON THE GPU. */
/** \param[in] "position" 3D coordinate of point to be rotated. 
  \param[in] "theta" elevation angle of rotation. 
  \param[in] "phi" azimuthal angle of rotation. 
*/
__device__ float3 d_rotatePoint(const float3 &position, const float &theta, const float &phi);

//! CUDA device function to add float2 vectors
/** 
    \param a float2 vector
    \param b float2 vector
    \return sum
*/
__host__ __device__ float2 operator+(const float2 &a, const float2 &b);

//! CUDA device function to add float3 vectors
/** 
    \param a float3 vector
    \param b float3 vector
    \return sum
*/
__host__ __device__ float3 operator+(const float3 &a, const float3 &b);

//! CUDA device function to subtract float2 vectors
/** 
    \param a float2 vector
    \param b float2 vector
    \return difference
*/
__host__ __device__ float2 operator-(const float2 &a, const float2 &b);

float2 operator-(const float2 &a, const float2 &b);

//! CUDA device function to subtract float3 vectors
/** 
    \param a float3 vector
    \param b float3 vector
    \return difference
*/
__host__ __device__ float3 operator-(const float3 &a, const float3 &b);

// //! CUDA device function to reverse float3 vector (multiply by -1)
// /** 
//     \param a float3 vector
//     \return -a
// */
// __host__ __device__ float3 operator-(const float3 &a);

//! CUDA device function to dot float2 vectors
/** 
    \param a float2 vector
    \param b float2 vector
    \return dot product
*/
__host__ __device__ float operator*(const float2 &a, const float2 &b);

//! CUDA device function to dot float3 vectors
/** 
    \param a float3 vector
    \param b float3 vector
    \return dot product
*/
__host__ __device__ float operator*(const float3 &a, const float3 &b);

//! CUDA device function to multiply float3 by a float on the right
/** 
    \param a float3 vector
    \param b float scalar
    \return product
*/
__host__ __device__ float3 operator*(const float3 &a, const float& b);

//! CUDA device function to divide float3 by a float
/** 
    \param a float3 vector
    \param b float scalar
    \return quotient
*/
__host__ __device__ float3 operator/(const float3 &a, const float& b);

//! CUDA device function to calculate cross product of float3 vectors
/** 
    \param a float3 vector
    \param b float3 vector
    \return cross product
*/
__host__ __device__ inline float3 cross(const float3 &a, const float3 &b){
  float3 c;
  c.x=a.y*b.z - b.y*a.z;
  c.y=b.x*a.z - a.x*b.z;
  c.z=a.x*b.y - b.x*a.y;
  return c;
}

__host__ __device__ inline float magnitude(const float3 &a){
  return sqrt( a.x*a.x + a.y*a.y + a.z*a.z );
}

__host__ __device__ inline float3 normalize(const float3 &a){
  return a/magnitude(a);
}

float3 inline vec3tofloat3( helios::vec3 v3 ){
  float3 f3;
  f3.x=v3.x;
  f3.y=v3.y;
  f3.z=v3.z;
  return f3;
}

helios::vec3 inline flota3tovec3( float3 f3 ){
  helios::vec3 v3;
  v3.x=f3.x;
  v3.y=f3.y;
  v3.z=f3.z;
  return v3;
}

//*************************************************************//
//              AERIAL LIDAR DEFINITIONS                      //
//************************************************************//

__global__ void intersectBoundingBox( const size_t scanSize, float3* d_hit_origin, const float3* d_scan_xyz, const float3 bbcenter, const float3 bbsize, uint* d_boundingbox_hit );

__global__ void intersectGridcell( const size_t Nhitsbb, float3* d_scan_xyz, float3* d_scan_raydir, const size_t Ncells, float3* center, float3* anchor, float3* size, float* rotation, float* d_dr, float* d_dr_hit, float* hit_denom, float* hit_inside );

__global__ void intersectPatches( const size_t N, float3* origin, float3* d_raydir, const int Npatches, float3* d_patch_vertex0, float3* d_patch_vertex1, float3* d_patch_vertex2, float3* d_patch_vertex3, float* d_hit_t );

__global__ void intersectTriangles( const size_t N, float3* origin, float3* d_raydir, const int Ntriangles, float3* d_tri_vertex0, float3* d_tri_vertex1, float3* d_tri_vertex2, float* d_hit_t );
