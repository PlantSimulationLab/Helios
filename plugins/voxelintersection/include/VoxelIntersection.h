/** \file "VoxelIntersection.h" Primary header file for Primitive-Voxel Intersection plug-in.
    \author Brian Bailey

    Copyright (C) 2018  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef __VOXELINTERSECTION__
#define __VOXELINTERSECTION__

#include "Context.h"

#include <cuda_runtime.h>

class VoxelIntersection{
 public:

  //! Default constructor
  /** \param[in] "context" Pointer to Helios context
   */
  VoxelIntersection( helios::Context* context );

  //! Self-test fuction (unit tests)
  int selfTest( void );
  
  //! Calculate primitives that reside in all voxels for every primitive in the Context
  void calculatePrimitiveVoxelIntersection( void );

  //! Calculate primitives that reside in all voxels for a subset of primitives in the Context
  /** \param[in] "UUIDs" Unique universal identifiers for primitives (non-voxels) to include in intersection calculation
   */
  void calculatePrimitiveVoxelIntersection( std::vector<uint> UUIDs );

 private:

  helios::Context* context;

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
  
};

#endif
