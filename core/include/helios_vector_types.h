/** \file "helios_vector_types.h" Declarations for vector types
    \author Brian Bailey

    Copyright (C) 2016-2021  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_VECTOR_TYPES
#define HELIOS_VECTOR_TYPES

#include <vector>
#include <iostream>
#include <cstring>
#include <cassert>

//! Vector data structures
/** \defgroup vectors Vectors */

namespace helios{

//! Vector of two elements of type 'int'. 
/** \ingroup vectors */
struct int2{
public:

  //! First element in vector
  int x;
  //! Second element in vector
  int y;

  //! Default constructor
  //* initializes to zero */
  int2(){
    x=0;
    y=0;
  }
  //! Construct int2 from a vector of ints
  explicit int2( std::vector<int> v ){
    if( v.size()!=2){
      throw(std::runtime_error("ERROR: vector for initialization of int2 must have length 2."));
    }
    x=v.at(0);
    y=v.at(1);
  }
  //! Construct int2 from an array of ints
  explicit int2( const int v[2] ){
    x=v[0];
    y=v[1];
  }
  //! Construct int2 from two ints
  int2( int v0, int v1 ){
    x=v0;
    y=v1;
  }

  //! Add two int2 vectors
  int2 operator+(const int2& a) const;
  //! Subtract two int2 vectors
  int2 operator-(const int2& a) const;

};

//! Make an int2 vector from two ints
/* \param[in] "X" First component of integer vector
   \param[in] "Y" First component of integer vector
   \ingroup vectors
*/
inline int2 make_int2( int X, int Y ){
  int2 v(X,Y);
  return v;
}
 
//! Make an int2 vector from an array of ints
/* \param[in] "X" 2D array if integers
   \ingroup vectors
*/
inline int2 make_int2( int X[2] ){
  int2 v(X[0],X[1]);
  return v;
}

inline int2 int2::operator+(const int2& a) const{
  int2 c;
  c.x=a.x+x;
  c.y=a.y+y;
  return  c;
}

inline int2 int2::operator-(const int2& a) const{
  int2 c;
  c.x=x-a.x;
  c.y=y-a.y;
  return  c;
}

//! Vector of three elements of type 'int'. 
/** \ingroup vectors */
struct int3{
public:

  //! First element in vector
  int x;
  //! Second element in vector
  int y;
  //! Third element in vector
  int z;

  //! Default constructor
  //* initializes to zero */
  int3(){
    x=0;
    y=0;
    z=0;
  }
  //! Construct int3 from a vector of ints
  explicit int3( std::vector<int> v ){
    if( v.size()!=3){
        throw(std::runtime_error("ERROR: vector for initialization of int3 must have length 3."));
    }
    x=v.at(0);
    y=v.at(1);
    z=v.at(2);
  }
  //! Construct int3 from an array of ints
  explicit int3( const int v[3] ){
    x=v[0];
    y=v[1];
    z=v[2];
  }
  //! Construct int3 from three ints
  int3( int v0, int v1, int v2 ){
    x=v0;
    y=v1;
    z=v2;
  }

  //! Add two int3 vectors
  int3 operator+(const int3& a) const;
  //! Subtract two int3 vectors
  int3 operator-(const int3& a) const;

};

//! Make an int3 vector from three ints
inline int3 make_int3( int X, int Y, int Z ){
  int3 v(X,Y,Z);
  return v;
}

//! Make an int2 vector from an array of ints
inline int3 make_int3( int X[3] ){
  int3 v(X[0],X[1],X[2]);
  return v;
}

inline int3 int3::operator+(const int3& a) const{
  int3 c;
  c.x=a.x+x;
  c.y=a.y+y;
  c.z=a.z+z;
  return  c;
}

inline int3 int3::operator-(const int3& a) const{
  int3 c;
  c.x=x-a.x;
  c.y=y-a.y;
  c.z=z-a.z;
  return  c;
}

//! Vector of four elements of type 'int'. 
/** \ingroup vectors */
struct int4{
public:

  //! First element in vector
  int x;
  //! Second element in vector
  int y;
  //! Third element in vector
  int z;
  //! Fourth element in vector
  int w;

  //! Default constructor
  //* initializes to zero */
  int4(){
    x=0;
    y=0;
    z=0;
    w=0;
  }
  //! Construct int3 from a vector of ints
  explicit int4( std::vector<int> v ){
    if( v.size()!=4){
        throw(std::runtime_error("ERROR: vector for initialization of int4 must have length 4."));
    }
    x=v.at(0);
    y=v.at(1);
    z=v.at(2);
    w=v.at(3);
  }
  //! Construct int4 from an array of ints
  explicit int4( const int v[3] ){
    x=v[0];
    y=v[1];
    z=v[2];
    w=v[3];
  }
  //! Construct int4 from four ints
  int4( int v0, int v1, int v2, int v3 ){
    x=v0;
    y=v1;
    z=v2;
    w=v3;
  }

  //! Add two int4 vectors
  int4 operator+(const int4& a) const;
  //! Subtract two int4 vectors
  int4 operator-(const int4& a) const;

};

//! Make an int4 vector from three ints
 inline int4 make_int4( int X, int Y, int Z, int W ){
   int4 v(X,Y,Z,W);
   return v;
 }

 //! Make an int4 vector from an array of ints
inline int4 make_int4( int X[4] ){
  int4 v(X[0],X[1],X[2],X[3]);
  return v;
}

inline int4 int4::operator+(const int4& a) const{
  int4 c;
  c.x=a.x+x;
  c.y=a.y+y;
  c.z=a.z+z;
  c.w=a.w+w;
  return  c;
}

inline int4 int4::operator-(const int4& a) const{
  int4 c;
  c.x=x-a.x;
  c.y=y-a.y;
  c.z=z-a.z;
  c.w=w-a.w;
  return  c;
}

//! Vector of two elements of type 'float'. 
/** \ingroup vectors */
struct vec2{
public:

  //! First element in vector
  float x;
  //! Second element in vector
  float y;

  //! Normalize vector components such that the magnitue is unity.
  void normalize(){
    float mag = sqrt( x*x + y*y );
    x/=mag;
    y/=mag;
  }

  //! Compute the vector magnitude 
  /**  \return magnitude of vector: \f$\sqrt{x^2+y^2}\f$ */
  float magnitude() const{
    return sqrt( x*x + y*y );
  }

  //! Default constructor 
  /** initializes to zero */
  vec2(){
    x=0;
    y=0;
  }
  //! Initialize vec2 using a vector of floats
  explicit vec2( std::vector<float> v ){
    if( v.size()!=2){
        throw(std::runtime_error("ERROR: vector for initialization of vec2 must have length 2."));
    }
    x=v.at(0);
    y=v.at(1);
  }						
  //! Initialize vec2 using an array of floats
  explicit vec2( const float v[2] ){
    x=v[0];
    y=v[1];
  }
  //! Initialize vec2 using two floats
  vec2( float v0, float v1 ){
    x=v0;
    y=v1;
  }

  //! Dot (scalar) product of two vec2 vectors
  float operator*(const vec2& a) const;
  //! Sum of two vec2 vectors
  vec2 operator+(const vec2& a) const;
  //! Difference of two vec2 vectors
  vec2 operator-(const vec2& a) const;
  //! Multiply each element by scalar (scalar is multiplied on right: vec2*a)
  vec2 operator*( float a) const;
  //! Divide each element by scalar (scalar is divided on right: vec2/a)
  vec2 operator/( float a) const;
  //!Add a scalar to each element (scalar is added on right: vec2+a)
  vec2 operator+( float a) const;
  //!Subtract a scalar from each element (scalar is subtracted on right: vec2-a)
  vec2 operator-( float a) const;
  //! check for equality of two vec2 vectors
  bool operator==(const vec2& a) const;

};

//! Multiply each element by scalar (scalar is multiplied on left: a*vec2)
vec2 operator*( float a, const vec2& v);
//! Add a scalar to each element (scalar is added on left: a+vec2)
vec2 operator+( float a, const vec2& v);
//! Subtract a scalar from each element (scalar is subtracted on left: a-vec2)
vec2 operator-( float a, const vec2& v);

//! Make a vec2 from two floats
/** \param[in] "X" First element of vector
    \param[in] "Y" Second element of vector
    \ingroup vectors
*/
inline vec2 make_vec2( float X, float Y ){
  return {X,Y};
}

inline float vec2::operator*(const vec2& a) const{
  return  a.x*x+a.y*y;
}

inline vec2 vec2::operator+(const vec2& a) const{
  vec2 c;
  c.x=a.x+x;
  c.y=a.y+y;
  return  c;
}

inline vec2 vec2::operator+(const float a) const {
  return {a+x, a+y};
}

inline vec2 operator+(const float a, const vec2& v){
  return {a+v.x, a+v.y};
}

inline vec2 vec2::operator-(const vec2& a) const{
  vec2 c;
  c.x=x-a.x;
  c.y=y-a.y;
  return  c;
}

inline vec2 vec2::operator-(const float a) const {
  return {x-a, y-a};
}

inline vec2 operator-(const float a, const vec2& v) {
  return {a-v.x, a-v.y};
}

inline vec2 vec2::operator*(const float a) const{
  vec2 c;
  c.x=x*a;
  c.y=y*a;
  return c;
}

inline vec2 operator*(const float a, const vec2& v) {
  return {a * v.x, a * v.y};
}

inline vec2 vec2::operator/(const float a) const{
  vec2 c;
  c.x=x/a;
  c.y=y/a;
  return c;
}

inline bool vec2::operator==(const vec2& a) const{
  bool c = (x == a.x && y == a.y);
  return c;
}

//! Vector of three elements of type 'float'. 
/** \ingroup vectors */
struct vec3{
public:

  //! First element in vector
  float x;
  //! Second element in vector
  float y;
  //! Third element in vector
  float z;

  //! Normalize vector components such that the magnitue is unity.
  void normalize(){
    float mag = sqrt( x*x + y*y + z*z );
    x/=mag;
    y/=mag;
    z/=mag;
  }

  //! Compute the vector magnitude 
  //*   \return magnitude of vector: \f$\sqrt{x^2+y^2+z^2}\f$ */
  float magnitude() const{
    return sqrt( x*x + y*y + z*z );
  }

  //! Default constructor 
  /** initializes to zero */
  vec3(){
    x=y=z=0;
  }
  //! Initialize vec3 using a vector of floats
  explicit vec3( std::vector<float> v ){
    if( v.size()!=3){
        throw(std::runtime_error("ERROR: vector for initialization of vec3 must have length 3."));
    }
    x=v.at(0);
    y=v.at(1);
    z=v.at(2);
  }
  //! Initialize vec3 using an array of floats
  explicit vec3( const float v[3] ){
    x=v[0];
    y=v[1];
    z=v[2];
  }
  //! Initialize vec3 using three floats
  vec3( float v0, float v1, float v2 ){
    x=v0;
    y=v1;
    z=v2;
  }

  //! Dot (scalar) product of two vec3 vectors
  float operator*(const vec3& a) const;
  //! Sum of two vec3 vectors
  vec3 operator+(const vec3& a) const;
  //! Difference of two vec2 vectors
  vec3 operator-(const vec3& a) const;
  //! Multiply each element by scalar (scalar is multiplied on right: vec3*a)
  vec3 operator*( float a) const;
  //! Divide each element by scalar (scalar is divided on right: vec3/a)
  vec3 operator/( float a) const;
  //!Add a scalar to each element (scalar is added on right: vec3+a)
  vec3 operator+( float a) const;
  //!Subtract a scalar from each element (scalar is subtracted on right: vec3-a)
  vec3 operator-( float a) const;

  //! check for equality of two vec3 vectors
  bool operator==(const vec3& a) const;

};

//! Multiply each element by scalar (scalar is multiplied on left: a*vec3)
vec3 operator*( float a, const vec3& v);
//! Add a scalar to each element (scalar is added on left: a+vec3)
vec3 operator+( float a, const vec3& v);
//! Subtract a scalar from each element (scalar is subtracted on left: a-vec3)
vec3 operator-( float a, const vec3& v);

//! Make a vec3 from three floats
inline vec3 make_vec3( float X, float Y, float Z ){
  return {X,Y,Z};
}

//! Make a vec3 from an array of floats
inline vec3 make_vec3( float X[3] ){
  return {X[0],X[1],X[2]};
}

inline vec3 cross(const vec3 &a, const vec3 &b ){
  vec3 c;
  c.x=a.y*b.z - b.y*a.z;
  c.y=b.x*a.z - a.x*b.z;
  c.z=a.x*b.y - b.x*a.y;
  return c;
}

inline float vec3::operator*(const vec3& a) const{
  return  a.x*x+a.y*y+a.z*z;
}

inline vec3 vec3::operator+(const vec3& a) const{
  vec3 c;
  c.x=a.x+x;
  c.y=a.y+y;
  c.z=a.z+z;
  return  c;
}

inline vec3 vec3::operator+(const float a) const{
  return { x+a, y+a, z+a };
}

inline vec3 operator+(const float a, const vec3& v) {
  return {a+v.x, a+v.y, a+v.z};
}

inline vec3 vec3::operator-(const vec3& a) const{
  vec3 c;
  c.x=x-a.x;
  c.y=y-a.y;
  c.z=z-a.z;
  return  c;
}

inline vec3 vec3::operator-(const float a) const{
  return { x-a, y-a, z-a };
}

inline vec3 operator-(const float a, const vec3& v) {
  return {a-v.x, a-v.y, a-v.z};
}

inline vec3 vec3::operator*(const float a) const{
  vec3 c;
  c.x=x*a;
  c.y=y*a;
  c.z=z*a;
  return  c;
}

inline vec3 operator*(const float a, const vec3& v) {
  return {a * v.x, a * v.y, a * v.z};
}

inline vec3 vec3::operator/(const float a) const{
  vec3 c;
  c.x=x/a;
  c.y=y/a;
  c.z=z/a;
  return c;
}

inline bool vec3::operator==(const vec3& a) const{
  bool c = (x == a.x && y == a.y && z == a.z);
  return c;
}

//! Vector of four elements of type 'float'. 
/** \ingroup vectors */
struct vec4{
public:

  //! First element in vector
  float x;
  //! Second element in vector
  float y;
  //! Third element in vector
  float z;
  //! Fourth element in vector
  float w;

  //! Normalize vector components such that the magnitue is unity.
  void normalize(){
    float mag = sqrt( x*x + y*y + z*z + w*w );
    x/=mag;
    y/=mag;
    z/=mag;
    w/=mag;
  }

  //! Compute the vector magnitude 
  //*   \return magnitude of vector: \f$\sqrt{x^2+y^2+z^2+w^2}\f$ */
  float magnitude() const{
    return sqrt( x*x + y*y + z*z + w*w );
  }

  //! Default constructor 
  /** initializes to zero */
  vec4(){
    x=y=z=w=0;
  }
  //! Initialize vec4 using a vector of floats
  explicit vec4( std::vector<float> v ){
    if( v.size()!=4){
        throw(std::runtime_error("ERROR: vector for initialization of vec4 must have length 4."));
    }
    x=v.at(0);
    y=v.at(1);
    z=v.at(2);
    w=v.at(3);
  }
  //! Initialize vec3 using an array of floats
  explicit vec4( const float v[4] ){
    x=v[0];
    y=v[1];
    z=v[2];
    w=v[3];
  }
  //! Initialize vec4 using four floats
  vec4( float v0, float v1, float v2, float v3 ){
    x=v0;
    y=v1;
    z=v2;
    w=v3;
  }

  //! Dot (scalar) product of two vec4 vectors
  float operator*(const vec4& a) const;
  //! Sum of two vec4 vectors
  vec4 operator+(const vec4& a) const;
  //! Difference of two vec4 vectors
  vec4 operator-(const vec4& a) const;
  //! Multiply each element by scalar (scalar is multiplied on right: vec4*a)
  vec4 operator*( float a) const;
  //! Divide each element by scalar (scalar is divided on right: vec4/a)
  vec4 operator/( float a) const;
  //!Add a scalar to each element (scalar is added on right: vec4+a)
  vec4 operator+( float a) const;
  //!Subtract a scalar from each element (scalar is subtracted on right: vec4-a)
  vec4 operator-( float a) const;
  //! check for equality of two vec4 vectors
  bool operator==(const vec4& a) const;

};

//! Multiply each element by scalar (scalar is multiplied on left: a*vec4)
vec4 operator*( float a, const vec4& v);
//! Add a scalar to each element (scalar is added on left: a+vec4)
vec4 operator+( float a, const vec4& v);
//! Subtract a scalar from each element (scalar is subtracted on left: a-vec4)
vec4 operator-( float a, const vec4& v);

//! Make a vec4 from three floats
 inline vec4 make_vec4( float X, float Y, float Z, float W ){
  return {X,Y,Z,W};
}

//! Make a vec4 from an array of floats
inline vec4 make_vec4( float X[4] ){
  return {X[0],X[1],X[2],X[3]};
}

inline float vec4::operator*(const vec4& a) const{
  return  a.x*x+a.y*y+a.z*z+a.w*w;
}

inline vec4 vec4::operator+(const vec4& a) const{
  vec4 c;
  c.x=a.x+x;
  c.y=a.y+y;
  c.z=a.z+z;
  c.w=a.w+w;
  return  c;
}

inline vec4 vec4::operator+(const float a) const{
  return { x+a, y+a, z+a, w+a };
}

inline vec4 operator+(const float a, const vec4& v) {
  return {a+v.x, a+v.y, a+v.z, a+v.w};
}

inline vec4 vec4::operator-(const vec4& a) const{
  vec4 c;
  c.x=x-a.x;
  c.y=y-a.y;
  c.z=z-a.z;
  c.w=w-a.w;
  return  c;
}

inline vec4 vec4::operator-(const float a) const{
  return { x-a, y-a, z-a, w-a };
}

inline vec4 operator-(const float a, const vec4& v) {
  return {a-v.x, a-v.y, a-v.z, a-v.w};
}

inline vec4 vec4::operator*(const float a) const{
  vec4 c;
  c.x=x*a;
  c.y=y*a;
  c.z=z*a;
  c.w=w*a;
  return  c;
}

inline vec4 operator*(const float a, const vec4& v) {
  return {a * v.x, a * v.y, a * v.z, a * v.w};
}

inline vec4 vec4::operator/(const float a) const{
  vec4 c;
  c.x=x/a;
  c.y=y/a;
  c.z=z/a;
  c.w=w/a;
  return c;
}

inline bool vec4::operator==(const vec4& a) const{
  bool c = (x == a.x && y == a.y && z == a.z && w == a.w);
  return c;
}

//! R-G-B color vector
/**  \ingroup vectors */
struct RGBcolor{
public:

  //! Red color component.
  /*! Takes a value between 0 and 1. */
  float r;
  //! Green color component.  
    /*! Takes a value between 0 and 1. */
  float g;
  //! Blue color component.
  /*! Takes a value between 0 and 1. */
  float b;

  //! Defalut constructor - initializes color to black. 
  RGBcolor(){
    r=0;
    g=0;
    b=0;
  }
  //! Constructor given three floats denoting R-G-B components.  \note If arguments are outside of the range 0 to 1, values are clamped.
  RGBcolor( float r_, float g_, float b_ ){
    /** @param [in] "r_" Red color component */
    /** @param [in] "g_" Green color component */
    /** @param [in] "b_" Blue color component */
    r=clamp(r_);
    g=clamp(g_);
    b=clamp(b_);
  }
  //! Constructor given an array of three floats denoting R-G-B components.  \note If arguments are outside of the range 0 to 1, values are clamped.
  explicit RGBcolor( const float C[3] ){
    /** @param [in] "C" Array (3-elements) of R-G-B color components */
    r=clamp(C[0]);
    g=clamp(C[1]);
    b=clamp(C[2]);
  }
  //! Constructor given a vector of three floats denoting R-G-B components.  \note If arguments are outside of the range 0 to 1, values are clamped.
  explicit RGBcolor( std::vector<float> C ){
    /** @param [in] "C" Vector (3-elements) of R-G-B color components */
    if( C.size()!=3 ){
        throw(std::runtime_error("ERROR: cannot initialize RGBcolor using supplied vector - size should be 3."));
    }
    r=clamp(C.at(0));
    g=clamp(C.at(1));
    b=clamp(C.at(2));
  }
  //! Constructor given a vec3 denoting R-G-B components.  \note If arguments are outside of the range 0 to 1, values are clamped.
  explicit RGBcolor(const vec3 &C ){
    /** @param [in] "C" vec3 of R-G-B color components */
    r=clamp(C.x);
    g=clamp(C.y);
    b=clamp(C.z);
  }

  //! Scale RGBcolor by some factor.
  void scale( float scl ){
    //* @param [in] "scl" Positive scaling factor */
    if( scl<0 ){
        throw(std::runtime_error("ERROR: cannot scale RGBcolor by negative factor."));
    }
    r*=scl;
    g*=scl;
    b*=scl;
  }

private:
  //! Clamp float to range of 0 to 1.
  /** \return Clamped value */
  static float clamp( float value ){
    /** @param [in] "value" Value to be clamped. */
    if( value<0.f ){
      return 0.f;
    }else if( value>1.f ){
      return 1.f;
    }else{
      return value;
    }
  }

};

//! Make an RGBcolor vector
/** \param[in] "r" Red component
    \param[in] "g" Green component
    \param[in] "b" Blue component
    \ingroup vectors
*/
inline RGBcolor make_RGBcolor( float r, float g, float b ){
  return {r,g,b};
}

//! Blend two RGB colors together
/** \param[in] "color0" First color to be blended.
    \param[in] "color1" Second color to be bleded.
    \param[out] "weight" Weighting factor (0<weight<1). A weight of 0 returns color0, weight of 1 returns color1.
*/
RGBcolor blend( RGBcolor color0, RGBcolor color1, float weight );

namespace RGB{

  extern RGBcolor red;
  extern RGBcolor blue;
  extern RGBcolor green;
  extern RGBcolor cyan;
  extern RGBcolor magenta;
  extern RGBcolor yellow;
  extern RGBcolor orange;
  extern RGBcolor violet;
  extern RGBcolor black;
  extern RGBcolor white;
  extern RGBcolor lime;
  extern RGBcolor silver;
  extern RGBcolor gray;
  extern RGBcolor navy;
  extern RGBcolor brown;
  extern RGBcolor khaki;
  extern RGBcolor greenyellow;
  extern RGBcolor forestgreen;
  extern RGBcolor yellowgreen;
  extern RGBcolor goldenrod;

}

//! R-G-B-A color vector
/**  \ingroup vectors */
struct RGBAcolor{
public:

  //! Red color component.
  /*! Takes a value between 0 and 1. */
  float r;
  //! Green color component.  
    /*! Takes a value between 0 and 1. */
  float g;
  //! Blue color component.
  /*! Takes a value between 0 and 1. */
  float b;
  //! Alpha (transparency) component.
  /*! Takes a value between 0 and 1. */
  float a;

  //! Defalut constructor - initializes color to black. 
  RGBAcolor(){
    r=0;
    g=0;
    b=0;
    a=0;
  }
  //! Constructor given three floats denoting R-G-B components.  \note If arguments are outside of the range 0 to 1, values are clamped.
  RGBAcolor( float r_, float g_, float b_, float a_ ){
    /** @param [in] "r_" Red color component */
    /** @param [in] "g_" Green color component */
    /** @param [in] "b_" Blue color component */
    /** @param [in] "a_" Alpha (transparency) component */
    r=clamp(r_);
    g=clamp(g_);
    b=clamp(b_);
    a=clamp(a_);
  }
  //! Constructor given an array of three floats denoting R-G-B-A components.  \note If arguments are outside of the range 0 to 1, values are clamped.
  explicit RGBAcolor( const float C[4] ){
    /** @param [in] "C" Array (4-elements) of R-G-B-A color components */
    r=clamp(C[0]);
    g=clamp(C[1]);
    b=clamp(C[2]);
    a=clamp(C[3]);
  }
  //! Constructor given a vector of three floats denoting R-G-B-A components.  \note If arguments are outside of the range 0 to 1, values are clamped.
  explicit RGBAcolor( std::vector<float> C ){
    /** @param [in] "C" Vector (4-elements) of R-G-B-A color components */
    if( C.size()!=4 ){
        throw(std::runtime_error("ERROR: cannot initialize RGBAcolor using supplied vector - size should be 4."));
    }
    r=clamp(C.at(0));
    g=clamp(C.at(1));
    b=clamp(C.at(2));
    a=clamp(C.at(3));
  }

  //! Scale RGBAcolor by some factor.
  void scale( float scl ){
    //* @param [in] "scl" Positive scaling factor */
    if( scl<0 ){
        throw(std::runtime_error("ERROR: cannot scale RGBAcolor by negative factor."));
    }
    r*=scl;
    g*=scl;
    b*=scl;
    a*=scl;
  }

private:
  //! Clamp float to range of 0 to 1.
  /** \return Clamped value */
  static float clamp( float value ){
    /** @param [in] "value" Value to be clamped. */
    if( value<0.f ){
      return 0.f;
    }else if( value>1.f ){
      return 1.f;
    }else{
      return value;
    }
  }

};

//! Make an RGBAcolor vector
/** \param[in] "r" Red component
    \param[in] "g" Green component
    \param[in] "b" Blue component
    \param[in] "a" Alpha (transparency) component
    \ingroup vectors
*/
inline RGBAcolor make_RGBAcolor( float r, float g, float b, float a ){
  return {r,g,b,a};
}

//! Make an RGBAcolor vector (add alpha channel to RGBcolor)
/** \param[in] "color" RGBcolor
    \param[in] "a" Alpha (transparency) component
    \ingroup vectors
*/
inline RGBAcolor make_RGBAcolor( RGBcolor color, float a ){
  return {color.r,color.g,color.b,a};
}

//! Blend two RGBA colors together
/** \param[in] "color0" First color to be blended.
    \param[in] "color1" Second color to be bleded.
    \param[out] "weight" Weighting factor (0<weight<1). A weight of 0 returns color0, weight of 1 returns color1.
*/
RGBAcolor blend(const RGBAcolor &color0, const RGBAcolor &color1, float weight );

//! Date vector
/** \sa \ref make_Date()
    \ingroup vectors 
*/
struct Date{
public:

  //! Day of month
  int day;
  //! Month of year
  int month;
  //! Year in YYYY format
  int year;

  //! Default constructor
  Date(){
    day = 1;
    month = 1;
    year = 2000;
  }

  //! Day/Month/Year constructor
  /** \param[in] "day" Day of month (1-31)
      \param[in] "month" Day of year (1-12)
      \param[in] "year" Year in YYYY format
  */
  Date( int a_day, int a_month, int a_year ){
    
    if( a_day<1 || a_day>31 ){
        throw(std::runtime_error("ERROR (Date constructor): Day of month is out of range (day of " + std::to_string(a_day) + " was given)."));
    }else if( a_month<1 || a_month>12){
        throw(std::runtime_error("ERROR (Date constructor): Month of year is out of range (month of " + std::to_string(a_month) + " was given)."));
    }else if( a_year<1000 ){
        throw(std::runtime_error( "ERROR (Date constructor): Year should be specified in YYYY format."));
    }
    
    day = a_day;
    month = a_month;
    year = a_year;
    
  }

  //! Convert to Julian day
  int JulianDay() const;

};

//! Make a \ref Date vector
/** \param[in] "day" Day of month (1-31)
    \param[in] "month" Day of year (1-12)
    \param[in] "year" Year in YYYY format
    \ingroup vectors
*/
inline Date make_Date( int day, int month, int year ){
  if( day<1 || day>31 ){
      throw(std::runtime_error("ERROR (make_Date): Day of month is out of range (day of " + std::to_string(day) + " was given)."));
  }else if( month<1 || month>12){
      throw(std::runtime_error("ERROR (make_Date): Month of year is out of range (month of " + std::to_string(month) + " was given)."));
  }else if( year<1000 ){
      throw(std::runtime_error("ERROR (make_Date): Year should be specified in YYYY format."));
  }

  Date date;
  date.day = day;
  date.month = month;
  date.year = year;
  return date;
  
}

//! Convert a Julian day to a calendar \ref Date vector.
/** \param[in] "JulianDay" Julian day of year (0-366)
    \param[in] "year" Year in YYYY format
    \ingroup vectors
*/
inline Date Julian2Calendar( int JulianDay, int year ){

  int day, month; 
  
  int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
  int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int* skips;

  if( JulianDay<1 || JulianDay>366 ){
      throw(std::runtime_error("ERROR (Julian2Calendar): Julian day of " + std::to_string(JulianDay) + " is out of range (should be >0 and <367)."));
  }
  
  if( (year-2000)%4 == 0 ){  //leap year
    skips=skips_leap;
  }else{                 //non-leap year
    skips=skips_nonleap;
  }

  int i;
  for( i=1; i<=12; i++ ){
    if( i==12 ){
      month = 12;
    }else if( JulianDay>skips[i-1] && JulianDay<=skips[i] ){
      month = i;
      break;
    }
  }
  assert( month>0 && month<13 );

  day = JulianDay-skips[month-1];

  assert(day>0 && day<32 );
  
  return make_Date(day,month,year);
  
}

//! Convert a calendar \ref Date vector to Julian day.
/** \param[in] "Date" Date vector
    \ingroup vectors
*/
inline int Calendar2Julian( Date date ){

  int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 214, 244, 274, 305, 335};
  int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int* skips;
  
  if( (date.year-2000)%4 == 0 ){  //leap year
    skips=skips_leap;
  }else{                 //non-leap year
    skips=skips_nonleap;
  }

  int JD = skips[date.month-1]+date.day;
  assert( JD>0 && JD<366 );
  
  return JD;
  
}

//! Make a \ref Date vector
/** \param[in] "JulianDay" Julian day of year (0-366)
    \param[in] "year" Year in YYYY format
    \ingroup vectors
*/
inline Date make_Date( int JulianDay, int year ){
  if( JulianDay<1 || JulianDay>366 ){
      throw(std::runtime_error("ERROR (make_Date): Julian day is out of range (Julian day of " + std::to_string(JulianDay) + " was given)."));
  }else if( year<1000 ){
      throw(std::runtime_error("ERROR (make_Date): Year should be specified in YYYY format."));
  }

  return Julian2Calendar(JulianDay,year);
  
}


//! Time vector
/** \sa \ref make_Time()
    \ingroup vectors 
*/
struct Time{

  //! Second of minute
  int second;
  //! Minute of hour
  int minute;
  //! Hour of day
  int hour;

  //! Default constructor
  Time(){
    second = 0;
    minute = 0;
    hour = 0;
  }

  //! second/minute/hour constructor
  /** \param[in] "hour" Hour of day (0-23)
      \param[in] "minute" Minute of hour (0-59)
  */
  Time( int a_hour, int a_minute ){

    if( a_minute<0 || a_minute>59 ){
        throw(std::runtime_error("ERROR (Time constructor): Minute out of range (0-59)."));
    }else if( a_hour<0 || a_hour>23 ){
        throw(std::runtime_error("ERROR (Time constructor): Hour out of range (0-23)."));
    }

    second = 0;
    minute = a_minute;
    hour = a_hour;
    
  }

  //! minute/hour constructor
  /** \param[in] "hour" Hour of day (0-23)
      \param[in] "minute" Minute of hour (0-59)
      \param[in] "second" Second of minute (0-59)
  */
  Time( int a_hour, int a_minute, int a_second ){

    if( a_second<0 || a_second>59 ){
        throw(std::runtime_error("ERROR (Time constructor): Second out of range (0-59)."));
    }else if( a_minute<0 || a_minute>59 ){
        throw(std::runtime_error("ERROR (Time constructor): Minute out of range (0-59)."));
    }else if( a_hour<0 || a_hour>23 ){
        throw(std::runtime_error("ERROR (Time constructor): Hour out of range (0-23)."));
    }

    second = a_second;
    minute = a_minute;
    hour = a_hour;
    
  }

};

//! Make a \ref Time vector
/** \param[in] "hour" Hour of day (0-23)
    \param[in] "minute" Minute of hour (0-59)
    \ingroup vectors
*/
inline Time make_Time( int hour, int minute ){
  if( minute<0 || minute>59 ){
      throw(std::runtime_error("ERROR (make_Time): Minute of " + std::to_string(minute) + " out of range (0-59)."));
  }else if( hour<0 || hour>23 ){
      throw(std::runtime_error("ERROR (make_Time): Hour of " + std::to_string(hour) + " out of range (0-23)."));
  }

  return {hour,minute,0};
  
}

//! Make a \ref Time vector
/** \param[in] "hour" Hour of day (0-23)
    \param[in] "minute" Minute of hour (0-59)
    \param[in] "second" Second of minute (0-59)
    \ingroup vectors
*/
inline Time make_Time( int hour, int minute, int second ){
  if( second<0 || second>59 ){
      throw(std::runtime_error("ERROR (make_Time): Second of " + std::to_string(second) + " out of range (0-59)."));
  }else if( minute<0 || minute>59 ){
      throw(std::runtime_error("ERROR (make_Time): Minute of " + std::to_string(minute) + " out of range (0-59)."));
  }else if( hour<0 || hour>23 ){
      throw(std::runtime_error("ERROR (make_Time): Hour of " + std::to_string(hour) + " out of range (0-23)."));
  }

  return {hour,minute,second};
  
}

//! Vector of spherical coordinates (elevation,azimuth)
/** \ingroup vectors */
struct SphericalCoord{
public:

  //! Radius
  float radius;
  //! Elevation angle (radians)
  float elevation;
   //! Zenithal angle (radians)
  float zenith;
  //! Azimuthal angle (radians)
  float azimuth;

  //! Default constructor 
  /** initializes to vertical unit vector */
  SphericalCoord(){
    radius=1;
    elevation=0.5*float(M_PI);
    zenith=0.f;
    azimuth=0;
  }
  //! Initialize SphericalCoord by specifying radius, elevation, and azimuth
  /** \param[in] "radius" Radius
      \param[in] "elevation" Elevation angle (radians)
      \param[in] "azimuth" Azimuthal angle (radians)
  */
  SphericalCoord( float a_radius, float a_elevation, float a_azimuth ){
    radius=a_radius;
    elevation=a_elevation;
    zenith=0.5f*float(M_PI)-elevation;
    azimuth=a_azimuth;
  }

};

//! Make a SphericalCoord by specifying elevation and azimuth
/** \param[in] "elevation" Elevation angle (radians)
    \param[in] "azimuth" Azimuthal angle (radians)
  */
inline SphericalCoord make_SphericalCoord( float a_elevation, float a_azimuth ){
  return {1,a_elevation,a_azimuth};
}

//! Make a SphericalCoord by specifying radius, elevation, and azimuth
/** \param[in] "radius" Radius
    \param[in] "elevation" Elevation angle (radians)
    \param[in] "azimuth" Azimuthal angle (radians)
  */
 inline SphericalCoord make_SphericalCoord( float a_radius, float a_elevation, float a_azimuth ){
  return {a_radius,a_elevation,a_azimuth};
}


}

#endif
