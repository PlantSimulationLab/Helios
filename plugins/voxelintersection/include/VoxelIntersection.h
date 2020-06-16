/** \file "VoxelIntersection.h" Primary header file for Primitive-Voxel Intersection plug-in.
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

#ifndef __VOXELINTERSECTION__
#define __VOXELINTERSECTION__

#include "Context.h"

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

  //! Disable output messages (enabled by default)
  void disableMessages( void );

  //! Re-enable output messages
  void enableMessages( void );

 private:

  helios::Context* context;

  bool printmessages;
  
};

#endif
