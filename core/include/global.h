/** \file "global.h" Header file for all global function/object definitions. 
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

#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif

#ifndef __GLOBAL__
#define __GLOBAL__

#include <stdlib.h>
//#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <exception>
#include <assert.h>
#include <cmath>
#include <memory>
#include <time.h>
#include <map>
#include <algorithm>
#include <ctime>
#include <random>
#include <chrono>
#include <thread>

typedef unsigned int uint;

#include "helios_vector_types.h"

// *** Groups *** //

//! Miscellaneous helper functions
/** \defgroup functions Functions */

//! Basic geometric objects to construct the domain
/** \defgroup primitives Geometric Primitives */

//! Geometric objects comprised of multiple primitives
/** \defgroup compoundobjects Compound Objects */

namespace helios{

//--------------------- HELPER FUNCTIONS -----------------------------------//

//! Construct a rotation matrix to perform rotation about the x-, y-, or z-axis.
/** 4x4 Affine rotation matrix 
    \param[in] "rotation" Rotation angle about axis in radians.
    \param[in] "axis" Axis about which to rotate (one of 'x', 'y', or 'z').
    \param[out] "transform" Transformation matrix in a 1D array
    \ingroup functions
*/
void makeRotationMatrix( const float rotation, const char* axis, float (&transform)[16] );

//! Construct a rotation matrix to perform rotation about an arbitrary axis.
/** 4x4 Affine rotation matrix 
    \param[in] "rotation" Rotation angle about axis in radians.
    \param[in] "axis" Vector describing axis about which to perform rotation.
    \param[out] "transform" Transformation matrix in a 1D array
    \ingroup functions
*/
void makeRotationMatrix( const float rotation, const helios::vec3, float (&transform)[16] );

//! Construct translation matrix
/** 4x4 Affine translation matrix 
    \param[in] "translation" Distance to translate in x, y, and z directions.
    \param[out] "transform" Transformation matrix in a 1D array
    \ingroup functions
*/
void makeTranslationMatrix( const helios::vec3 translation, float (&transform)[16] );

 //! Construct matrix to scale
/** 4x4 Affine scaling matrix 
    \param[in] "scale" Scaling factor to apply in x, y, and z directions.
    \param[out] "transform" Transformation matrix in a 1D array
    \ingroup functions
*/
void makeScaleMatrix( const helios::vec3 scale, float (&T)[16] );

//! Multiply 4x4 matrices: T=ML*MR 
void matmult( const float ML[16], const float MR[16], float (&T)[16] );

//! Multiply 4x4 transformation matrix by 3-element vector: T=M*v 
void vecmult( const float M[16], const float v[3], float (&result)[3] );

//! Multiply 4x4 transformation matrix by 3-element vector: T=M*v 
 void vecmult( const float M[16], const helios::vec3 v3, helios::vec3& result );

/* //! Construct a transformation matrix */
/* /\** 4x4 Affine transformation matrix  */
/*     \param[in] "translation" Translation distance in x,y,z directions  */
/*     \param[in] "scale" Scaling factor in x,y,z directions */
/*     \param[out] "transform" Transformation matrix in a 1D array */
/*     \ingroup functions */
/* *\/ */
/* void makeTransformationMatrix( const vec3 translation, const vec3 scale, float (&transform)[16] ); */

 //! Construct an identity matrix
 void makeIdentityMatrix( float (&T)[16] );

//! Convert degrees to radians
/** \param deg angle in degrees
    \return angle in radians
    \sa rad2deg()
    \ingroup functions
*/
 float deg2rad( const float& deg );

//! Convert radians to degrees
/** \param rad angle in radians
    \return angle in degrees
    \sa deg2rad()
    \ingroup functions
*/
 float rad2deg( const float& rad );

//! Four quadrant arc tangent between 0 and \f$2\pi\f$
/** \note atan2_2pi(0,1) = 0, atan2_2pi(1,0) = \f$0.5\pi\f$, etc.
    \ingroup functions */
 float atan2_2pi( const float& y, const float& x);

 //! Convert Cartesian coordinates to spherical coordinates
/** \param[in] "Cartesian" (x,y,z) Cartesian coordinates
    \returns SphericalCoord vector.
    \sa \ref sphere2cart(), \ref SphericalCoord
*/
 SphericalCoord cart2sphere( const vec3& Cartesian );

//! Convert Spherical coordinates to Cartesian coordinates
/** \param[in] "Spherical" (radius,elevation,azimuth) \ref SphericalCoord vector
    \returns vec3 vector of Cartesian coordinates
    \sa \ref cart2sphere()
*/
 vec3 sphere2cart( const SphericalCoord& Spherical );

//! Convert a space-delimited string into a vec2 vector
/** \ingroup functions */
vec2 string2vec2( const char* str );

//! Convert a space-delimited string into a vec3 vector
/** \ingroup functions */
vec3 string2vec3( const char* str );

//! Convert a space-delimited string into an int2 vector
/** \ingroup functions */
int2 string2int2( const char* str );

 //! Convert a space-delimited string into an int3 vector
/** \ingroup functions */
int3 string2int3( const char* str );

//! Convert a space-delimited string into an RGBAcolor vector
/** \ingroup functions */
RGBAcolor string2RGBcolor( const char* str );

//! Remove whitespace from character array
/** \ingroup functions */
const char* deblank(const char* input);

//! Clamp value to be within some specified bounds
/** \param[in] "value" Value to be clamped
    \param[in] "min" Lower bound
    \param[in] "max" Upper bound
    \ingroup functions */
template <typename anytype>
anytype clamp( anytype value, anytype min, anytype max );

//! Mean value of a vector of floats
/** \param[in] "vect" C++ vector of floats 
 \ingroup functions
*/
float mean( std::vector<float> vect );

//! Minimum value of a vector of floats
/** \param[in] "vect" C++ vector of floats 
 \ingroup functions
*/
float min( std::vector<float> vect );

//! Minimum value of a vector of ints
/** \param[in] "vect" C++ vector of ints 
 \ingroup functions
*/
int min( std::vector<int> vect );

//! Minimum value of a vector of vec3's
/** \param[in] "vect" C++ vector of vec3's 
 \ingroup functions
*/
vec3 min( std::vector<vec3> vect );

//! Maximum value of a vector of floats
/** \param[in] "vect" C++ vector of floats
\ingroup functions
*/
float max( std::vector<float> vect );

//! Maximum value of a vector of ints
/** \param[in] "vect" C++ vector of ints
\ingroup functions
*/
int max( std::vector<int> vect );

//! Maximum value of a vector of vec3's
/** \param[in] "vect" C++ vector of vec3's
\ingroup functions
*/
vec3 max( std::vector<vec3> vect );

//! Resize 2D C++ vector
/** \param[in] "vec" C++ vector
    \param[in] "Nx" Size of vector in x-direction, i.e., vec.at(0).size()
    \param[in] "Ny" Size of vector in y-direction, i.e., vec.size()
    \ingroup functions
*/
template <typename anytype>
void resize_vector( std::vector<std::vector<anytype> > &vec, uint Nx, uint Ny );

//! Resize 3D C++ vector
/** \param[in] "vec" C++ vector
    \param[in] "Nx" Size of vector in x-direction, i.e., vec[0][0].size()
    \param[in] "Ny" Size of vector in y-direction, i.e., vec[0].size()
    \param[in] "Nz" Size of vector in y-direction, i.e., vec.size()
    \ingroup functions
*/
template <typename anytype>
void resize_vector( std::vector<std::vector<std::vector<anytype> > > &vec, int Nx, int Ny, int Nz );

//! Resize 4D C++ vector
/** \param[in] "vec" C++ vector
    \param[in] "Nx" Size of vector in x-direction, i.e., vec[0][0][0].size()
    \param[in] "Ny" Size of vector in y-direction, i.e., vec[0][0].size()
    \param[in] "Nz" Size of vector in y-direction, i.e., vec[0].size()
    \param[in] "Nw" Size of vector in y-direction, i.e., vec.size()
    \ingroup functions
*/
template <typename anytype>
void resize_vector( std::vector<std::vector<std::vector<std::vector<anytype> > > > &vec, int Nx, int Ny, int Nz, int Nw );

//! Function to rotate a 3D vector given spherical angles elevation and azimuth
/** \param[in] "position" 3D coordinate of point to be rotated.
    \param[in] "rotation" Spherical rotation angles (elevation,azimuth)
    \ingroup functions
*/
vec3 rotatePoint(const vec3 position, const SphericalCoord rotation );

//! Function to rotate a 3D vector given spherical angles elevation and azimuth
/** \param[in] "position" 3D coordinate of point to be rotated.
    \param[in] "theta" elevation angle of rotation.
    \param[in] "phi" azimuthal angle of rotation. 
    \ingroup functions
*/
vec3 rotatePoint(const vec3 position, const float theta, const float phi);

//! Rotate a 3D vector about an arbitrary line
/** \param[in] "point" 3D coordinate of point to be rotated
    \param[in] "line_base" Point through which line passes
    \param[in] "line_direction" Unit vector pointing in the direction of the line
    \param[in] "theta" Rotation angle in radians
*/
 vec3 rotatePointAboutLine(const vec3 point, const vec3 line_base, const vec3 line_direction, const float theta);

//! Calculate the area of a triangle given its three vertices
/** \param[in] "v0" (x,y,z) coordinate of first vertex
    \param[in] "v1" (x,y,z) coordinate of second vertex
    \param[in] "v2" (x,y,z) coordinate of third vertex
    \return One-sided surface area of triangle
    \ingroup functions
*/
float calculateTriangleArea( const vec3 v0, const vec3 v1, const vec3 v2 );

//! Convert Julian day to calendar day (day,month,year)
/** 
    \param[in] "Julian_day" Julian day of year (1-366)
    \return \ref int3, with \ref int3::x = day, \ref int3::y = month, \ref int3::z = year
    \sa \ref JulianDay()
    \ingroup sun
*/
Date CalendarDay( int Julian_day, int year );

//! Convert calendar day (day,month,year) to Julian day
/**
   \param[in] "day" Day of month (1-31)
   \param[in] "month" Month of year (1-12)
   \param[in] "year" Year in YYYY format
   \return Julian day of year
   \sa \ref CalendarDay()
   \ingroup sun
*/
int JulianDay( int day, int month, int year );

//! Convert calendar day (day,month,year) to Julian day
/**
   \param[in] "date" Date vector
   \return Julian day of year
   \sa \ref CalendarDay()
   \ingroup functions
   \ingroup sun
*/
int JulianDay( Date date );

//! Random number from a uniform distribution between 0 and 1
/** \ingroup functions */
float randu( void );

//! arccosine function to handle cases when round-off errors cause an argument <-1 or >1, and thus regular acos() returns NaN
float acos_safe( float x );

//! arcsine function to handle cases when round-off errors cause an argument <-1 or >1, and thus regular asin() returns NaN
float asin_safe( float x );

//!Determine if two line segments intersect. The lines segments are defined by vertices (p1,q1) and (p2,q2).
bool lineIntersection( const helios::vec2 p1, const helios::vec2 q1, const helios::vec2 p2, const helios::vec2 q2);

//!Determine whether point lines within a polygon
bool pointInPolygon( const helios::vec2 point, const std::vector<helios::vec2> polygon_verts );

//! MATLAB-style timer.  Call tic() to start timer, call \ref toc() to stop timer and print duration
struct Timer{
public:

  Timer(void){
    running=false;
  }

  //! Start timer
  void tic(void){
    timer_start = std::chrono::high_resolution_clock::now();
    running = true;
  };

  //! Stop timer and print elapsed time
  double toc(void){
    return toc("");
  }
  
  //! Stop timer and print elapsed time and a user-defined message
  double toc( const char* message ){
    if( !running ){
      std::cerr << "ERROR (Timer): You must call `tic' before calling `toc'. Ignoring call to `toc'..." << std::endl;
      return 0;
    }

    auto timer_end = std::chrono::high_resolution_clock::now();;
    double duration = std::chrono::duration<double>(timer_end-timer_start).count();
    std::cout << "Elapsed time is " << duration << " seconds: " << message << std::endl;
    return duration;
  }

private:

  bool running;
  std::chrono::high_resolution_clock::time_point timer_start;

};

//! Wait/sleep for a specified amount of time
/** \param[in] "seconds" Number of seconds to wait
    \ingroup functions
*/
void wait( float seconds );

 //! Check whether PNG image file has an alpha/transparency channel
 /** \param[in] "filename" Name of the PNG image file
  */
 bool PNGHasAlpha( const char* filename );
 
 //! Function to read the alpha channel from a PNG image
 /** \param[in] "filename" Name of the PNG image file
       \return 2D mask, where false denotes no material
 */
 std::vector<std::vector<bool> > readPNGAlpha( const char* filename );

 
}

#endif
