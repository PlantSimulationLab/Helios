/** \file "global.h" Header file for all global function/object definitions.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef HELIOS_GLOBAL
#define HELIOS_GLOBAL

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <exception>
#include <cassert>
#include <cmath>
#include <memory>
#include <ctime>
#include <map>
#include <algorithm>
#include <ctime>
#include <random>
#include <chrono>
#include <thread>
#include <iomanip>

typedef unsigned int uint;

#include "helios_vector_types.h"

//pugi XML parser
#include "pugixml.hpp"

// *** Groups *** //

//! Miscellaneous helper functions
/**
 * \defgroup functions Utility functions
 */

//! Basic geometric objects to construct the domain
/**
 * \defgroup primitives Geometric Primitives
 */

//! Geometric objects comprised of multiple primitives
/**
 * \defgroup compoundobjects Compound Objects
 */

namespace helios{

    //! Function to throw a runtime error
    /**
     * \param[in] error_message Message to be issued when error is thrown.
     */
    void helios_runtime_error( const std::string &error_message );

//--------------------- HELPER FUNCTIONS -----------------------------------//

//! Construct a rotation matrix to perform rotation about the x-, y-, or z-axis.
/** 4x4 Affine rotation matrix 
    * \param[in] rotation Rotation angle about axis in radians.
    * \param[in] axis Axis about which to rotate (one of 'x', 'y', or 'z').
    * \param[out] transform Transformation matrix in a 1D array
    * \ingroup functions
*/
    void makeRotationMatrix( float rotation, const char* axis, float (&transform)[16] );

//! Construct a rotation matrix to perform rotation about an arbitrary axis passing through the origin.
/** 4x4 Affine rotation matrix 
    * \param[in] rotation Rotation angle about axis in radians.
    * \param[in] axis Vector describing axis about which to perform rotation.
    * \param[out] transform Transformation matrix in a 1D array
    * \ingroup functions
*/
    void makeRotationMatrix( float rotation, const helios::vec3& axis, float (&transform)[16] );

//! Construct a rotation matrix to perform rotation about an arbitrary line (not necessarily passing through the origin).
/** 4x4 Affine rotation matrix
 * \param[in] rotation Rotation angle about axis in radians.
 * \param[in] origin Cartesian coordinate of the base/origin of rotation axis.
 * \param[in] axis Vector describing axis about which to perform rotation.
 * \param[out] transform Transformation matrix in a 1D array
 * \ingroup functions
*/
    void makeRotationMatrix( float rotation, const helios::vec3& origin, const helios::vec3& axis, float (&transform)[16] );

//! Construct translation matrix
/** 4x4 Affine translation matrix 
    * \param[in] translation Distance to translate in x, y, and z directions.
    * \param[out] transform Transformation matrix in a 1D array
    * \ingroup functions
*/
    void makeTranslationMatrix( const helios::vec3& translation, float (&transform)[16] );

    //! Construct matrix to scale
/** 4x4 Affine scaling matrix 
    * \param[in] scale Scaling factor to apply in x, y, and z directions.
    * \param[out] transform Transformation matrix in a 1D array
    * \ingroup functions
*/
    void makeScaleMatrix( const helios::vec3& scale, float (&T)[16] );

//! Multiply 4x4 matrices: T=ML*MR 
    void matmult( const float ML[16], const float MR[16], float (&T)[16] );

//! Multiply 4x4 transformation matrix by 3-element vector: T=M*v 
    void vecmult( const float M[16], const float v[3], float (&result)[3] );

//! Multiply 4x4 transformation matrix by 3-element vector: T=M*v 
    void vecmult( const float M[16], const helios::vec3& v3, helios::vec3& result );

    //! Construct an identity matrix
    void makeIdentityMatrix( float (&T)[16] );

//! Convert degrees to radians
/**
 * \param deg angle in degrees
 * \return angle in radians
 * \sa rad2deg()
 * \ingroup functions
*/
    float deg2rad( float deg );

//! Convert radians to degrees
/**
 * \param rad angle in radians
 * \return angle in degrees
 * \sa deg2rad()
 * \ingroup functions
*/
    float rad2deg( float rad );

//! Four quadrant arc tangent between 0 and 2*pi
/**
 * \note atan2_2pi(0,1) = 0, atan2_2pi(1,0) = pi/2, etc.
 * \ingroup functions
 */
    float atan2_2pi(float y, float x);

    //! Convert Cartesian coordinates to spherical coordinates
/**
 * \param[in] Cartesian (x,y,z) Cartesian coordinates
 * \returns SphericalCoord vector.
 * \sa sphere2cart(), SphericalCoord
 * \ingroup functions
*/
    SphericalCoord cart2sphere( const vec3& Cartesian );

//! Convert Spherical coordinates to Cartesian coordinates
/**
 * \param[in] Spherical (radius,elevation,azimuth) SphericalCoord vector
 * \returns vec3 vector of Cartesian coordinates
 * \sa cart2sphere()
 * \ingroup functions
*/
    vec3 sphere2cart( const SphericalCoord& Spherical );

//! Convert a space-delimited string into a vec2 vector
/**
 * \ingroup functions
 */
    vec2 string2vec2( const char* str );

//! Convert a space-delimited string into a vec3 vector
/**
 * \ingroup functions
 */
    vec3 string2vec3( const char* str );

//! Convert a space-delimited string into a vec4 vector
/**
 * \ingroup functions
 */
    vec4 string2vec4( const char* str );

//! Convert a space-delimited string into an int2 vector
/**
 * \ingroup functions
 */
    int2 string2int2( const char* str );

//! Convert a space-delimited string into an int3 vector
/**
 * \ingroup functions
 */
    int3 string2int3( const char* str );

//! Convert a space-delimited string into an int4 vector
/**
 * \ingroup functions
 */
    int4 string2int4( const char* str );

//! Convert a string into a float with error checking
/**
 * \param[in] input_string String to be converted to numerical value
 * \param[out] converted_float Output numerical value converted from input string
 * \return True if conversion was successful, false if unsuccessful
 */
    bool parse_float( const std::string &input_string, float &converted_float );

//! Convert a string into a double with error checking
/**
 * \param[in] input_string String to be converted to numerical value
 * \param[out] converted_double Output numerical value converted from input string
 * \return True if conversion was successful, false if unsuccessful
 */
    bool parse_double( const std::string &input_string, double &converted_double );

//! Convert a string into an integer with error checking
/**
 * \param[in] input_string String to be converted to numerical value
 * \param[out] converted_int Output numerical value converted from input string
 * \return True if conversion was successful, false if unsuccessful
 */
    bool parse_int( const std::string &input_string, int &converted_int );

//! Convert a string into an unsigned integer with error checking
/**
 * \param[in] input_string String to be converted to numerical value
 * \param[out] converted_uint Output numerical value converted from input string
 * \return True if conversion was successful, false if unsuccessful
 */
    bool parse_uint( const std::string &input_string, uint &converted_uint );

//! Convert a space-delimited string into an RGBcolor vector
/**
 * \ingroup functions
 */
    RGBAcolor string2RGBcolor( const char* str );

//! Remove all whitespace from character array
/**
 * \ingroup functions
 */
    std::string deblank(const char* input);

//! Remove all whitespace from a string
/**
 * \ingroup functions
 */
    std::string deblank(const std::string &input);

//! Remove leading and trailing whitespace from a string
/**
 * \ingroup functions
 */
    std::string trim_whitespace(const std::string &input);

//! Separate string by delimiter and store into a vector
/**
 * \param[in] inputstring String to be separated
 * \param[in] delimiter Delimiter character for separation
 * \return Vector of substrings. If inputstring does not contain delimiter, empty vector is returned
 * \ingroup functions
 */
    std::vector<std::string> separate_string_by_delimiter( const std::string &inputstring, const std::string &delimiter );

//! Clamp value to be within some specified bounds
/**
 * \param[in] value Value to be clamped
 * \param[in] min Lower bound
 * \param[in] max Upper bound
 * \ingroup functions */
    template <typename anytype>
    anytype clamp( anytype value, anytype min, anytype max );

//! Sum of a vector of floats
/**
 * \param[in] vect C++ vector of floats
 * \ingroup functions
*/
    float sum( const std::vector<float>& vect );

//! Mean value of a vector of floats
/**
 * \param[in] vect C++ vector of floats
 * \ingroup functions
*/
    float mean( const std::vector<float>& vect );

//! Minimum value of a vector of floats
/**
 * \param[in] vect C++ vector of floats
 * \ingroup functions
*/
    float min( const std::vector<float>& vect );

//! Minimum value of a vector of ints
/**
 * \param[in] vect C++ vector of ints
 * \ingroup functions
*/
    int min( const std::vector<int>& vect );

//! Minimum value of a vector of vec3's
/**
 * \param[in] vect C++ vector of vec3's
 * \ingroup functions
*/
    vec3 min( const std::vector<vec3>& vect );

//! Maximum value of a vector of floats
/**
 * \param[in] vect C++ vector of floats
 * \ingroup functions
*/
    float max( const std::vector<float>& vect );

//! Maximum value of a vector of ints
/**
 * \param[in] vect C++ vector of ints
 * \ingroup functions
*/
    int max( const std::vector<int>& vect );

//! Maximum value of a vector of vec3's
/**
 * \param[in] vect C++ vector of vec3's
 * \ingroup functions
*/
    vec3 max( const std::vector<vec3>& vect );

//! Standard deviation of a vector of floats
/**
 * \param[in] vect C++ vector of floats
 * \ingroup functions
 */
    float stdev( const std::vector<float> &vect );

//! Median of a vector of floats
/**
 * \param[in] vect C++ vector of floats
 * \ingroup functions
 */
    float median( std::vector<float> vect );

//! Resize 2D C++ vector
/**
 * \param[in] vec C++ vector
 * \param[in] Nx Size of vector in x-direction, i.e., vec.at(0).size()
 * \param[in] Ny Size of vector in y-direction, i.e., vec.size()
 * \ingroup functions
*/
    template <typename anytype>
    void resize_vector( std::vector<std::vector<anytype> > &vec, uint Nx, uint Ny );

//! Resize 3D C++ vector
/**
 * \param[in] vec C++ vector
 * \param[in] Nx Size of vector in x-direction, i.e., vec[0][0].size()
 * \param[in] Ny Size of vector in y-direction, i.e., vec[0].size()
 * \param[in] Nz Size of vector in y-direction, i.e., vec.size()
 * \ingroup functions
*/
    template <typename anytype>
    void resize_vector( std::vector<std::vector<std::vector<anytype> > > &vec, uint Nx, uint Ny, uint Nz );

//! Resize 4D C++ vector
/**
 * \param[in] vec C++ vector
 * \param[in] Nx Size of vector in x-direction, i.e., vec[0][0][0].size()
 * \param[in] Ny Size of vector in y-direction, i.e., vec[0][0].size()
 * \param[in] Nz Size of vector in y-direction, i.e., vec[0].size()
 * \param[in] Nw Size of vector in y-direction, i.e., vec.size()
 * \ingroup functions
*/
    template <typename anytype>
    void resize_vector( std::vector<std::vector<std::vector<std::vector<anytype> > > > &vec, uint Nx, uint Ny, uint Nz, uint Nw );

//! Blend two RGB colors together
/**
 * \param[in] color0 First color to blend
 * \param[in] color1 Second color to blend
 * \param[in] weight Weighting to apply to each color during blending. E.g., weight=0 will produce color0, weight=0.5 will be an average of color0 and color1, weight=1 will produce color1.
 */
    RGBcolor blend(const RGBcolor &color0, const RGBcolor &color1, float weight );

//! Blend two RGBA colors together
/**
 * \param[in] color0 First color to blend
 * \param[in] color1 Second color to blend
 * \param[in] weight Weighting to apply to each color during blending. E.g., weight=0 will produce color0, weight=0.5 will be an average of color0 and color1, weight=1 will produce color1.
 */
    RGBAcolor blend(const RGBAcolor &color0, const RGBAcolor &color1, float weight );

//! Function to rotate a 3D vector given spherical angles elevation and azimuth
/**
 * \param[in] position 3D coordinate of point to be rotated.
 * \param[in] rotation Spherical rotation angles (elevation,azimuth)
 * \ingroup functions
*/
    vec3 rotatePoint(const vec3& position, const SphericalCoord& rotation );

//! Function to rotate a 3D vector given spherical angles elevation and azimuth
/**
 * \param[in] position 3D coordinate of point to be rotated.
 * \param[in] theta elevation angle of rotation.
 * \param[in] phi azimuthal angle of rotation.
 * \ingroup functions
*/
    vec3 rotatePoint(const vec3& position, float theta, float phi);

//! Rotate a 3D vector about an arbitrary line
/**
 * \param[in] point 3D coordinate of point to be rotated
 * \param[in] line_base Point through which line passes
 * \param[in] line_direction Unit vector pointing in the direction of the line
 * \param[in] theta Rotation angle in radians
*/
    vec3 rotatePointAboutLine(const vec3& point, const vec3& line_base, const vec3& line_direction, float theta);

//! Calculate the area of a triangle given its three vertices
/**
 * \param[in] v0 (x,y,z) coordinate of first vertex
 * \param[in] v1 (x,y,z) coordinate of second vertex
 * \param[in] v2 (x,y,z) coordinate of third vertex
 * \return One-sided surface area of triangle
 * \ingroup functions
*/
    float calculateTriangleArea( const vec3& v0, const vec3& v1, const vec3& v2 );

//! Convert Julian day to calendar day (day,month,year)
/**
 * \param[in] Julian_day Julian day of year (1-366)
 * \param[in] year Year in YYYY format
 * \return int3, with int3::x = day, int3::y = month, int3::z = year
 * \sa JulianDay()
 * \ingroup sun
*/
    Date CalendarDay( int Julian_day, int year );

//! Convert calendar day (day,month,year) to Julian day
/**
 * \param[in] day Day of month (1-31)
 * \param[in] month Month of year (1-12)
 * \param[in] year Year in YYYY format
 * \return Julian day of year
 * \sa CalendarDay()
 * \ingroup sun
*/
    int JulianDay( int day, int month, int year );

//! Convert calendar day (day,month,year) to Julian day
/**
 * \param[in] date Date vector
 * \return Julian day of year
 * \sa CalendarDay()
 * \ingroup functions
 * \ingroup sun
*/
    int JulianDay( const Date& date );

//! Random number from a uniform distribution between 0 and 1
/** \ingroup functions */
    float randu();

//! Random integer from a uniform distribution between imin and imax
/**
 * \ingroup functions
 */
    int randu( int imin, int imax  );

//! arccosine function to handle cases when round-off errors cause an argument <-1 or >1, and thus regular acos() returns NaN
/**
 * \ingroup functions
 */
    float acos_safe( float x );

//! arcsine function to handle cases when round-off errors cause an argument <-1 or >1, and thus regular asin() returns NaN
/**
 * \ingroup functions
 */
    float asin_safe( float x );

//!Determine if two line segments intersect. The lines segments are defined by vertices (p1,q1) and (p2,q2)
/**
 * \ingroup functions
 */
    bool lineIntersection(const vec2 &p1, const vec2 &q1, const vec2 &p2, const vec2 &q2);

//!Determine whether point lines within a polygon
/**
 * \ingroup functions
 */
    bool pointInPolygon(const vec2 &point, const std::vector<vec2> &polygon_verts );

//! MATLAB-style timer.  Call tic() to start timer, call toc() to stop timer and print duration
/**
 * \ingroup functions
 */
    struct Timer{
    public:

        Timer(){
            running=false;
        }

        //! Start timer
        void tic(){
            timer_start = std::chrono::high_resolution_clock::now();
            running = true;
        };

        //! Stop timer and print elapsed time
        double toc() const{
            return toc("");
        }

        //! Stop timer and print elapsed time and a user-defined message
        /**
         * \note the timer print message can be turned off by passing the message argument  "mute"
        */
        double toc( const char* message ) const {
            if (!running) {
                std::cerr << "ERROR (Timer): You must call `tic' before calling `toc'. Ignoring call to `toc'..." << std::endl;
                return 0;
            }

            auto timer_end = std::chrono::high_resolution_clock::now();;
            double duration = std::chrono::duration<double>(timer_end - timer_start).count();
            if (strcmp(message, "mute")!=0) {
                std::cout << "Elapsed time is " << duration << " seconds: " << message << std::endl;
            }
            return duration;
        }

    private:

        bool running;
        std::chrono::high_resolution_clock::time_point timer_start;

    };

//! Wait/sleep for a specified amount of time
/**
 * \param[in] seconds Number of seconds to wait
 * \ingroup functions
*/
    void wait( float seconds );

    //! Check whether PNG image file has an alpha/transparency channel
    /**
     * \param[in] filename Name of the PNG image file
     */
    bool PNGHasAlpha( const char* filename );

    //! Function to read the alpha channel from a PNG image
    /**
     * \param[in] filename Name of the PNG image file
     * \return 2D mask, where false denotes no material
    */
    std::vector<std::vector<bool> > readPNGAlpha(const std::string &filename );

    //! Function to read a PNG image file into pixel data array
    /**
     * \param[in] filename Name of the PNG image file
     * \param[out] width Image width in pixels
     * \param[out] height Image height in pixels
     * \param[out] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
     */
    void readPNG( const std::string &filename, uint &width, uint &height, std::vector<helios::RGBAcolor> &pixel_data);

    //! Function to write a PNG image based on pixel data
    /**
    * \param[in] filename Name of the PNG image file
    * \param[in] width Image width in pixels
    * \param[in] height Image height in pixels
    * \param[in] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
    */
    void writePNG( const std::string &filename, uint width, uint height, const std::vector<helios::RGBAcolor> &pixel_data );


    //! Function to read a JPEG image file into pixel data array
    /**
        * \param[in] filename Name of the JPEG image file
        * \param[out] width Image width in pixels
        * \param[out] height Image height in pixels
        * \param[out] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
    */
    void readJPEG( const std::string &filename, uint &width, uint &height, std::vector<helios::RGBcolor> &pixel_data);

    //! Function to read a JPEG image file into pixel data array
    /**
        * \param[in] filename Name of the JPEG image file
    */
    helios::int2 getImageResolutionJPEG( const std::string &filename );

    //! Function to write a JPEG image based on pixel data
    /**
        * \param[in] filename Name of the JPEG image file
        * \param[in] width Image width in pixels
        * \param[in] height Image height in pixels
        * \param[in] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
    */
    void writeJPEG( const std::string &filename, uint width, uint height, const std::vector<helios::RGBcolor> &pixel_data );

    //! Function to flatten a 2D int vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<int> flatten( const std::vector<std::vector<int> > &vec );

    //! Function to flatten a 2D uint &vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<uint> flatten( const std::vector<std::vector<uint> > &vec );

    //! Function to flatten a 2D float vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<float> flatten( const std::vector<std::vector<float> > &vec );

    //! Function to flatten a 2D double vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<double> flatten( const std::vector<std::vector<double> > &vec );

    //! Function to flatten a 2D vec2 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec2> flatten( const std::vector<std::vector<helios::vec2> > &vec );

    //! Function to flatten a 2D vec3 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec3> flatten( const std::vector<std::vector<helios::vec3> > &vec );

    //! Function to flatten a 2D vec4 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec4> flatten( const std::vector<std::vector<helios::vec4> > &vec );

    //! Function to flatten a 2D int2 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int2> flatten( const std::vector<std::vector<helios::int2> > &vec );

    //! Function to flatten a 2D int3 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int3> flatten( const std::vector<std::vector<helios::int3> > &vec );

    //! Function to flatten a 2D int4 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int4> flatten( const std::vector<std::vector<helios::int4> > &vec );

    //! Function to flatten a 2D string vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<std::string> flatten( const std::vector<std::vector<std::string> > &vec );

    //! Function to flatten a 3D int vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<int> flatten( const std::vector<std::vector<std::vector<int> > > &vec );

    //! Function to flatten a 3D uint vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<uint> flatten( const std::vector<std::vector<std::vector<uint> > > &vec );

    //! Function to flatten a 3D float vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<float> flatten( const std::vector<std::vector<std::vector<float> > > &vec );

    //! Function to flatten a 3D double vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<double> flatten( const std::vector<std::vector<std::vector<double> > > &vec );

    //! Function to flatten a 3D vec2 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec2> flatten( const std::vector<std::vector<std::vector<helios::vec2> > > &vec );

    //! Function to flatten a 3D vec3 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec3> flatten( const std::vector<std::vector<std::vector<helios::vec3> > > &vec );

    //! Function to flatten a 3D vec4 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec4> flatten( const std::vector<std::vector<std::vector<helios::vec4> > > &vec );

    //! Function to flatten a 3D int2 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int2> flatten( const std::vector<std::vector<std::vector<helios::int2> > > &vec );

    //! Function to flatten a 3D int3 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int3> flatten( const std::vector<std::vector<std::vector<helios::int3> > > &vec );

    //! Function to flatten a 3D int4 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int4> flatten( const std::vector<std::vector<std::vector<helios::int4> > > &vec );

    //! Function to flatten a 3D string vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<std::string> flatten( const std::vector<std::vector<std::vector<std::string> > > &vec );


    //! Function to flatten a 4D int vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<int> flatten( const std::vector<std::vector<std::vector<std::vector<int> > > > &vec );

    //! Function to flatten a 4D uint vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<uint> flatten( const std::vector<std::vector<std::vector<std::vector<uint> > > > &vec );

    //! Function to flatten a 4D float vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<float> flatten( const std::vector<std::vector<std::vector<std::vector<float> > > > &vec );

    //! Function to flatten a 4D double vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<double> flatten( const std::vector<std::vector<std::vector<std::vector<double> > > > &vec );

    //! Function to flatten a 4D vec2 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec2> flatten( const std::vector<std::vector<std::vector<std::vector<helios::vec2> > > > &vec );

    //! Function to flatten a 4D vec3 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec3> flatten( const std::vector<std::vector<std::vector<std::vector<helios::vec3> > > > &vec );

    //! Function to flatten a 4D vec4 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::vec4> flatten( const std::vector<std::vector<std::vector<std::vector<helios::vec4> > > > &vec );

    //! Function to flatten a 4D int2 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int2> flatten( const std::vector<std::vector<std::vector<std::vector<helios::int2> > > > &vec );

    //! Function to flatten a 4D int3 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int3> flatten( const std::vector<std::vector<std::vector<std::vector<helios::int3> > > > &vec );

    //! Function to flatten a 4D int4 vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<helios::int4> flatten( const std::vector<std::vector<std::vector<std::vector<helios::int4> > > > &vec );

    //! Function to flatten a 4D string vector into a 1D vector
    /**
    * \ingroup functions
    */
    std::vector<std::string> flatten( const std::vector<std::vector<std::vector<std::vector<std::string> > > > &vec );


    //! Function to perform cubic Hermite spline interpolation
    /**
     * \param[in] u Interpolation point as a fraction of the distance between the start and end points (must be between 0 and 1).
     * \param[in] x_start (x,y,z) Cartesian coordinate of spline starting point.
     * \param[in] tan_start Vector tangent to spline at starting point.
     * \param[in] x_end (x,y,z) Cartesian coordinate of spline ending point.
     * \param[in] tan_end Vector tangent to spline at ending point.
     * \return Interpolated (x,y,z) Cartesian point.
      * \ingroup functions
    */
    helios::vec3 spline_interp3(float u, const vec3 &x_start, const vec3 &tan_start, const vec3 &x_end, const vec3 &tan_end );

    //! Function to load and convert a field in a pugi XML node into a float
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    float XMLloadfloat( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into an int
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    int XMLloadint( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into a string
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    std::string XMLloadstring( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into a vec2
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::vec2 XMLloadvec2( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into a vec3
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::vec3 XMLloadvec3( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into a vec4
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::vec4 XMLloadvec4( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into a vec2
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::int2 XMLloadint2( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into a int3
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::int3 XMLloadint3( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into a int4
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::int4 XMLloadint4( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into an RGB color vector
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::RGBcolor XMLloadrgb( pugi::xml_node node, const char* field );

    //! Function to load and convert a field in a pugi XML node into an RGBA color vector
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    helios::RGBAcolor XMLloadrgba( pugi::xml_node node, const char* field );

    //! Use Newton-Raphson method to find the zero of a function
    /**
     * \param[in] function Function to be evaluated. The function should take as its first argument the value at which the function should be evaluated, as second argument any function arguments.
     * \param[in] variables Vector of function arguments
     * \param[in] parameters Pointer to any additional parameters needed by the function
     * \param[in] init_guess Initial guess for the zero of the function.
     * \param[in] err_tol Maximum allowable relative error in solution.
     * \param[in] max_iterations Maximum number of iterations to allow before exiting solver.
     */
    float fzero(float(*function)(float value, std::vector<float> &variables, const void *parameters), std::vector<float> &variables, const void *parameters, float init_guess, float err_tol = 0.0001f, int max_iterations = 100 );

//! Function to perform linear interpolation based on a vector of discrete (x,y) values
/**
  * \param[in] points Vector of (x,y) pairs. x values must be monotonically increasing and not duplicated.
  * \param[in] x x value at which to interpolate. If x is less than the first x-value in points or greater than the last x-value in points, interp1 will respectively return the first or last y value.
  * \return Value of y interpolated at location x.
 */
    float interp1( const std::vector<helios::vec2> &points, float x );

    //! Function to calculate the distance between two points
    /**
      * \param[in] p1 first point (vec3)
      * \param[in] p2 second point (vec3)
      * \return distance between p1 and p2 in three dimensions
     */
    float point_distance( const helios::vec3 &p1 , const helios::vec3 &p2);

//! Parse a file string to get the extension
/**
  * Example: /path/to/file/filename.ext -> extension=".ext"
  * \param[in] filepath String containing file path and name
  * \return File extension contained in input string
  * \note If the string does not contain an extension, an empty string is returned
 */
    std::string getFileExtension( const std::string &filepath );

//! Parse a file string to get the filename without extension
/**
  * Example: /path/to/file/filename.ext -> stem="filename"
  * \param[in] filepath String containing file path and name
  * \return Filename without extension contained in input string
  * \note If the string does not contain a file name, an empty string is returned
 */
    std::string getFileStem( const std::string &filepath );

//! Parse a file string to get the filename including extension
/**
  * Example: /path/to/file/filename.ext -> filename="filename.ext"
  * \param[in] filepath String containing file path and name
  * \return Filename including extension contained in input string
  * \note If the string does not contain a file name, an empty string is returned
 */
    std::string getFileName( const std::string &filepath );

//! Parse a file string to get the path (i.e., portion of the string before the file name).
/**
  * Example: /path/to/file/filename.ext -> path="/path/to/file/"
  * \param[in] filepath String containing file path and name
  * \param[in] trailingslash If true, a trailing slash is added to the path
  * \return File path contained in input string
  * \note If the string does not contain a path, an empty string is returned
 */
    std::string getFilePath( const std::string &filepath, bool trailingslash = true );

//! Read values contained in a text file into a one-dimensional vector of floats
/**
  * \param[in] filepath Path to text file
  * \return One-dimensional vector of floats contained in text file
 */
    std::vector<float> importVectorFromFile(const std::string &filepath);

//! Default null SphericalCoord that applies no rotation
    extern SphericalCoord nullrotation;
//! Default null vec3 that gives the origin (0,0,0)
    extern vec3 nullorigin;

}

#endif
