/** \file "VoxelIntersection.h" Primary header file for Primitive-Voxel Intersection plug-in.

    Copyright (C) 2016-2024 Brian Bailey

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

  //! Calculate the intersection point of two lines
  /** \param[in] "line1_point" point on the first line
      \param[in] "line1_direction" direction vector of first line
     \param[in] "line2_point" point on the second line
     \param[in] "line2_direction" direction vector of second line
    \note this function assumes that the lines intersect - behavior for lines that don't intersect currently untested
    \return coordinate of the intersection point between the two lines
   */
  helios::vec3 linesIntersection(helios::vec3 line1_point, helios::vec3 line1_direction, helios::vec3 line2_point, helios::vec3 line2_direction);
  
  //! Slice a primitive (patch or triangle) into multiple triangles using a defined plane (often a voxel face)
  /** \param[in] "UUID" Unique universal identifier (UUID) of primitive to be sliced
   \param[in] "voxel_face_vertices" a vector of at least 3 points on the plane that will be used to do the slicing
   \return the number of new triangles created by slicing the original primitive
   */
  std::vector<uint> slicePrimitive(uint UUID, std::vector<helios::vec3> voxel_face_vertices);
  
  //! Slice a primitives based on a defined grid of voxels
  /** \param[in] "UUIDs" vector of Unique universal identifiers (UUIDs) of primitives to be sliced
   \param[in] "grid_center" coordinates of the center of the grid
   \param[in] "grid_size" size of the grid in the x, y, z dimenions
   \param[in] "grid_divisions" number of grid divisions in the x, y, and z dimensions
   \return vector of new planar primitives representing the original surfaces provided in UUIDs
   */
   std::vector<uint> slicePrimitivesUsingGrid(std::vector<uint> UUIDs, helios::vec3 grid_center, helios::vec3 grid_size, helios::int3 grid_divisions);

   //! get 2D vector storing the UUIDs within each grid cell
   /** 
    \return 2D vector of UUIDs within grid cells with grid cell ID as the first index and primitive ID as second index
    \note vector is one element longer than the number of grid cells, with the last element storing UUIDs that didn't fall into any grid cell
    */
   std::vector<std::vector<uint>> getCellPrimitives();
   
   //! compare two floating point numbers to within an absolute and relative error tolerance
   /** \param[in] "a" first floating point number to be compared
    \param[in] "b" second floating point number to be compared
    \param[in] "absTol" absolute tolerance for comparison
    \param[in] "relTol" relative tolerance for comparison
    \return boolean true if absolute difference between a and b is within absolute or relative tolerances
    */
    bool approxSame(float a, float b, float absTol, float relTol);

    //! compare two vec3's to within an absolute and relative error tolerance
    /** \param[in] "a" first vec3 to be compared
     \param[in] "b" second vec3  to be compared
     \param[in] "absTol" absolute tolerance for comparison
     \return boolean true if absolute difference between each element of a and b is within the absolute tolerance
     */
    bool approxSame(helios::vec3 a, helios::vec3 b, float absTol);  
    
    //! find the uv coordinate of a slice point
    /** \param[in] "p1" xyz coordinate of first vertex of the primitive edge that has the slice point on it
     \param[in] "uv1" uv coordinate of first vertex of the primitive edge that has the slice point on it
     \param[in] "p2" xyz coordinate of second vertex of the primitive edge that has the slice point on it
     \param[in] "uv2" uv coordinate of second vertex of the primitive edge that has the slice point on it
     \param[in] "ps" xyz coordinate of the slice point
     \return vec2 uvs the uv coordinate of the slice point
     */
    helios::vec2 interpolate_texture_UV_to_slice_point(helios::vec3 p1, helios::vec2 uv1, helios::vec3 p2, helios::vec2 uv2, helios::vec3 ps);

 private:

  helios::Context* context;

  bool printmessages;

  std::vector<std::vector<uint>> cell_primitives;
  
};

#endif
