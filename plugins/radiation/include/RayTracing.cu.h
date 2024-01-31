/** \file "RayTracing.cu.h" This file contains definitions and helper functions for CUDA/OptiX routines

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int uint;

void queryGPUMemory( void );

/** OptiX ray payload - contains data for each ray traversal. */
struct PerRayData
{

  //! "strength" or amount of energy associated with the ray. 
  double strength;  
  //! UUID of primitive from which ray originated
  uint origin_UUID;
  //! Face of primitive from which ray was launched (true=top, false=bottom)
  bool face;
  //! Area of primitive (note for voxels, this is area density)
  float area;
  //! Seed for curand random number generator
  uint seed;
  //! Number of periodic boundary intersections for ray
  unsigned char periodic_depth;
  //! Numerical identifier for radiation source corresponding to each ray
  /**
   * \note The data type limits to a maximum of 256 radiation sources
   */
  unsigned char source_ID;

};

static __device__ void init_state( PerRayData* prd );

__device__ __inline__ void atomicFloatAdd( float *address, float val ) {
  #if __CUDA_ARCH__ >= 200
      atomicAdd( address, val );
  #else
      float old = val;
      while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
  #endif
} // end atomic float add

#if __CUDA_ARCH__ < 600
__device__ __inline__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// -- random number generation -- //

template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for( unsigned int n = 0; n < N; n++ )
    {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }

    return v0;
}

static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
    return ((float) lcg(prev) / (float) 0x01000000);
}

/** Function to rotate a 3D vector given spherical angles elevation and azimuth ON THE GPU. */
__device__ float3 d_rotatePoint(const float3 &position, const float &theta, const float &phi) {

  /** @param[in] "position" 3D coordinate of point to be rotated. */
  /** @param[in] "theta" elevation angle of rotation. */
  /** @param[in] "phi" azimuthal angle of rotation. */

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

__device__ void d_makeTransformMatrix( optix::float3 rotation, float (&T)[16] ){

    T[0] = cosf(rotation.z)*cosf(rotation.y);
    T[1] = cosf(rotation.z)*sinf(rotation.y)*sinf(rotation.x)-sinf(rotation.z)*cosf(rotation.x);
    T[2] = cosf(rotation.z)*sinf(rotation.y)*cosf(rotation.x)+sinf(rotation.z)*sinf(rotation.x);
    T[3] = 0.f;
    T[4] = sinf(rotation.z)*cosf(rotation.y);
    T[5] = sinf(rotation.z)*sinf(rotation.y)*sinf(rotation.x)+cosf(rotation.z)*cosf(rotation.x);
    T[6] = sinf(rotation.z)*sinf(rotation.y)*cosf(rotation.x)-cosf(rotation.z)*sinf(rotation.x);
    T[7] = 0.f;
    T[8] = -sinf(rotation.y);
    T[9] = cosf(rotation.y)*sinf(rotation.x);
    T[10] = cosf(rotation.y)*cosf(rotation.x);
    T[11] = 0.f;
    T[12] = 0.f;
    T[13] = 0.f;
    T[14] = 0.f;
    T[15] = 1.f;

}

__device__ void d_invertMatrix( const float* m, float* minv ){

  float inv[16], det;

  inv[0] = m[5]  * m[10] * m[15] - 
    m[5]  * m[11] * m[14] - 
    m[9]  * m[6]  * m[15] + 
    m[9]  * m[7]  * m[14] +
    m[13] * m[6]  * m[11] - 
    m[13] * m[7]  * m[10];

  inv[4] = -m[4]  * m[10] * m[15] + 
    m[4]  * m[11] * m[14] + 
    m[8]  * m[6]  * m[15] - 
    m[8]  * m[7]  * m[14] - 
    m[12] * m[6]  * m[11] + 
    m[12] * m[7]  * m[10];

  inv[8] = m[4]  * m[9] * m[15] - 
    m[4]  * m[11] * m[13] - 
    m[8]  * m[5] * m[15] + 
    m[8]  * m[7] * m[13] + 
    m[12] * m[5] * m[11] - 
    m[12] * m[7] * m[9];
  
  inv[12] = -m[4]  * m[9] * m[14] + 
    m[4]  * m[10] * m[13] +
    m[8]  * m[5] * m[14] - 
    m[8]  * m[6] * m[13] - 
    m[12] * m[5] * m[10] + 
    m[12] * m[6] * m[9];
  
  inv[1] = -m[1]  * m[10] * m[15] + 
    m[1]  * m[11] * m[14] + 
    m[9]  * m[2] * m[15] - 
    m[9]  * m[3] * m[14] - 
    m[13] * m[2] * m[11] + 
    m[13] * m[3] * m[10];
  
  inv[5] = m[0]  * m[10] * m[15] - 
    m[0]  * m[11] * m[14] - 
    m[8]  * m[2] * m[15] + 
    m[8]  * m[3] * m[14] + 
    m[12] * m[2] * m[11] - 
    m[12] * m[3] * m[10];
  
  inv[9] = -m[0]  * m[9] * m[15] + 
    m[0]  * m[11] * m[13] + 
    m[8]  * m[1] * m[15] - 
    m[8]  * m[3] * m[13] - 
    m[12] * m[1] * m[11] + 
    m[12] * m[3] * m[9];
  
  inv[13] = m[0]  * m[9] * m[14] - 
    m[0]  * m[10] * m[13] - 
    m[8]  * m[1] * m[14] + 
    m[8]  * m[2] * m[13] + 
    m[12] * m[1] * m[10] - 
    m[12] * m[2] * m[9];
  
  inv[2] = m[1]  * m[6] * m[15] - 
    m[1]  * m[7] * m[14] - 
    m[5]  * m[2] * m[15] + 
    m[5]  * m[3] * m[14] + 
    m[13] * m[2] * m[7] - 
    m[13] * m[3] * m[6];
  
  inv[6] = -m[0]  * m[6] * m[15] + 
    m[0]  * m[7] * m[14] + 
    m[4]  * m[2] * m[15] - 
    m[4]  * m[3] * m[14] - 
    m[12] * m[2] * m[7] + 
    m[12] * m[3] * m[6];
  
  inv[10] = m[0]  * m[5] * m[15] - 
    m[0]  * m[7] * m[13] - 
    m[4]  * m[1] * m[15] + 
    m[4]  * m[3] * m[13] + 
    m[12] * m[1] * m[7] - 
    m[12] * m[3] * m[5];
  
  inv[14] = -m[0]  * m[5] * m[14] + 
    m[0]  * m[6] * m[13] + 
    m[4]  * m[1] * m[14] - 
    m[4]  * m[2] * m[13] - 
    m[12] * m[1] * m[6] + 
    m[12] * m[2] * m[5];
  
  inv[3] = -m[1] * m[6] * m[11] + 
    m[1] * m[7] * m[10] + 
    m[5] * m[2] * m[11] - 
    m[5] * m[3] * m[10] - 
    m[9] * m[2] * m[7] + 
    m[9] * m[3] * m[6];
  
  inv[7] = m[0] * m[6] * m[11] - 
    m[0] * m[7] * m[10] - 
    m[4] * m[2] * m[11] + 
    m[4] * m[3] * m[10] + 
    m[8] * m[2] * m[7] - 
    m[8] * m[3] * m[6];
  
  inv[11] = -m[0] * m[5] * m[11] + 
    m[0] * m[7] * m[9] + 
    m[4] * m[1] * m[11] - 
    m[4] * m[3] * m[9] - 
    m[8] * m[1] * m[7] + 
    m[8] * m[3] * m[5];
  
  inv[15] = m[0] * m[5] * m[10] - 
    m[0] * m[6] * m[9] - 
    m[4] * m[1] * m[10] + 
    m[4] * m[2] * m[9] + 
    m[8] * m[1] * m[6] - 
    m[8] * m[2] * m[5];
  
  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  
  //if (det == 0)
    //return false;
  
  det = 1.0 / det;
  
  for (int i = 0; i < 16; i++)
    minv[i] = inv[i] * det;

  
  
}


//! Function to transform a 3D point based on current affine transformation matrix on the GPU
__device__ void d_transformPoint( const float (&transform_matrix)[16], float3 &v ){

  float3 V;

  V.x = transform_matrix[0] * v.x + transform_matrix[1] * v.y + transform_matrix[2] * v.z + transform_matrix[3];
  V.y = transform_matrix[4] * v.x + transform_matrix[5] * v.y + transform_matrix[6] * v.z + transform_matrix[7];
  V.z = transform_matrix[8] * v.x + transform_matrix[9] * v.y + transform_matrix[10] * v.z + transform_matrix[11];

  v=V;

}

//! Function to calculate the magnitude of float3 vector on the GPU
__device__ float d_magnitude( const float3 v ){
  return sqrtf( v.x*v.x+v.y*v.y+v.z*v.z );
}

//! Function to calculate the surface area of a primitive
__device__ float d_calculatePrimitiveArea( const uint primitive_type, const float (&transform_matrix)[16] ){

  if( primitive_type == 0 || primitive_type == 3 ){ //Patch or AlphaMask

    float3 s0 = make_float3(0,0,0);
    float3 s1 = make_float3(1,1,0);
    float3 s2 = make_float3(0,1,0);
    d_transformPoint(transform_matrix,s0);
    d_transformPoint(transform_matrix,s1);
    d_transformPoint(transform_matrix,s2);
    return d_magnitude(s2-s0)*d_magnitude(s2-s1);

  }else if( primitive_type == 1 ){ //Triangle

    float3 v0 = make_float3(0,0,0);
    d_transformPoint(transform_matrix,v0);
    float3 v1 = make_float3(0,1,0);
    d_transformPoint(transform_matrix,v1);
    float3 v2 = make_float3(1,1,0);
    d_transformPoint(transform_matrix,v2);
    float3 A(v1-v0);
    float3 B(v2-v0);
    float3 C(v2-v1);
    float a = d_magnitude(A);
    float b = d_magnitude(B);
    float c = d_magnitude(C);
    float s = 0.5f*( a+b+c );
    return sqrtf(s*(s-a)*(s-b)*(s-c));

  }else if( primitive_type == 2 ){ //Disk

    float r0 = transform_matrix[0];
    float r1 = transform_matrix[5];
    return M_PI*r0*r1;

  }

  return 0;
  
}

__device__ void d_matMult33( float* ml, float* mr, float* m ){

  for( int i=0; i<9; i++ ){
    m[i] = 0.f;
  }

  for( int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      for(int k=0;k<3;k++){
	m[i*3+j]=m[i*3+j]+ml[i*3+k]*mr[k*3+j];
      }
    }
  }
  
}

__device__ void d_matMult44( float* ml, float* mr, float* m ){

  for( int i=0; i<16; i++ ){
    m[i] = 0.f;
  }

  for( int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
	m[i*4+j]=m[i*4+j]+ml[i*4+k]*mr[k*4+j];
      }
    }
  }

}

__device__ void d_sampleDisk( uint& seed, optix::float3& sample ){

    // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

    optix::float3 sp;

    float Rx = rnd(seed);
    float Ry = rnd(seed);

    //first map sample point to rectangle [-1,1] [-1,1]
    sp.x = -1.f + 2.f*Rx;
    sp.y = -1.f + 2.f*Ry;

    float r,p;

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
    sp.x = r*cos(p);
    sp.y = r*sin(p);
    sp.z = 0.f;

    sample = sp;

}

__device__ void d_sampleSquare( uint& seed, optix::float3& sample ){

    optix::float3 sp;

    float Rx = rnd(seed);
    float Ry = rnd(seed);

    sp.x = -0.5f + Rx;
    sp.y = -0.5f + Ry;
    sp.z = 0.f;

    sample = sp;

}

static __host__ __device__ __inline__ float acos_safe( float x )
{
  if (x < -1.0) x = -1.0 ;
  else if (x > 1.0) x = 1.0 ;
  return acos(x) ;
}

static __host__ __device__ __inline__ float asin_safe( float x )
{
  if (x < -1.0) x = -1.0 ;
  else if (x > 1.0) x = 1.0 ;
  return asin(x) ;
}
