/** \file "LiDAR.cuh" Header file for CUDA-related LiDAR functions.

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
#include "helios_vector_types.h"
#include "random.h"

namespace LIDAR_CUDA {

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
__host__ __device__ float3 cross(const float3 &a, const float3 &b);

__host__ __device__ float magnitude(const float3 &a);

__host__ __device__ float3 normalize(const float3 &a);

float2 vec2tofloat2( helios::vec2 v2 );

helios::vec2 float2tovec2( float2 f2 );

float3 vec3tofloat3( helios::vec3 v3 );

helios::vec3 float3tovec3( float3 f3 );

__device__ float3 d_sphere2cart( float radius, float elevation, float azimuth );

__host__ __device__ float atan2_2pi( const float& y, const float& x);
  
__host__ __device__ float acos_safe( float x );
  
__host__ int randu( int imin, int imax  );

__device__ float2 d_sampleDisk( float radius, uint seed );

__host__ bool sortcol0( const std::vector<float>& v0, const std::vector<float>& v1 );

  //__host__ bool sortcol1( const std::vector<float>& v0, const std::vector<float>& v1 );

  //__host__ bool sortcol0( const std::vector<double>& v0, const std::vector<double>& v1 );

__host__ bool sortcol1( const std::vector<double>& v0, const std::vector<double>& v1 );

__global__ void insideVolume( const uint Nhits, const float3* d_hit_xyz, const uint Ngridcells, const float3* d_grid_size, const float3* d_grid_center, const float3* d_grid_anchor, const float* d_grid_rotation, int* d_hit_vol );

//! Device function to intersect all rays with an axis-aligned bounding box
__global__ void intersectBoundingBox( const size_t scanSize, const float3 origin, const float3* d_hit_xyz, const float3 bbcenter, const float3 bbsize, uint* d_boundingbox_hit );

 __global__ void intersectGridcell( const size_t Nhitsbb, const float3 origin, float3* d_scan_xyz, float* d_scan_weight, const float3 center, const float3 anchor, const float3 size, const float rotation, float* d_dr, float* hit_before, float* hit_after );

__global__ void intersectGridcell_synthetic( const size_t Nhitsbb, const float3 origin, float3* d_scan_xyz, float* d_scan_weight, const float3 center, const float3 anchor, const float3 size, const float rotation, float* d_dr, float* hit_before, float* hit_after, uint* hit_location, const float scanner_range);

  __global__ void intersectPatches( const size_t N, const int rays_per_pulse, const float3 origin, float3* d_raydir, const float exit_diameter, const float beam_divergence, const int Npatches, float3* d_patch_vertex, int* d_patch_textureID, const int Ntextures, int2* d_masksize, const int2 masksize_max, bool* d_maskdata, float2* d_patch_uv, float* d_hit_t, float* d_hit_fnorm, int* d_hit_ID );

  __global__ void intersectTriangles( const size_t Npulse, const int rays_per_pulse, const float3 origin, float3* d_raydir, const float exit_diameter, const float beam_divergence, const int Ntriangles, const int Npatches, float3* d_tri_vertex, int* d_tri_textureID, const int Ntextures, int2* d_masksize, const int2 masksize_max, bool* d_maskdata, float2* d_tri_uv, float* d_hit_t, float* d_hit_fnorm, int* d_hit_ID );

}
