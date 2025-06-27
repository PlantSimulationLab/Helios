/** \file "global.h" Header file for all global function/object definitions.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_GLOBAL
#define HELIOS_GLOBAL

//! Macro for marking functions as deprecated.
#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif

//! Pi constant.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//! Pi constant as a float.
constexpr float PI_F = 3.14159265358979323846f;

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <stdexcept>
#include <exception>
#include <cassert>
#include <cmath>
#include <memory>
#include <ctime>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <ctime>
#include <random>
#include <chrono>
#include <thread>
#include <iomanip>
#include <filesystem>

#ifdef USE_OPENMP
#include <omp.h>
#endif

//! Unsigned integer type.
typedef unsigned int uint;

//! Static cast for converting between types.
template<typename To, typename From>
constexpr To scast(From&& v) noexcept
{
    return static_cast<To>(std::forward<From>(v));
}


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

namespace helios {

    //! Function to throw a runtime error
    /**
     * \param[in] error_message Message to be issued when error is thrown.
     */
    void helios_runtime_error(const std::string &error_message);

    //--------------------- HELPER FUNCTIONS -----------------------------------//

    //! Construct a rotation matrix to perform rotation about the x-, y-, or z-axis.
    /** 4x4 Affine rotation matrix
        * \param[in] rotation Rotation angle about axis in radians.
        * \param[in] axis Axis about which to rotate (one of 'x', 'y', or 'z').
        * \param[out] transform Transformation matrix in a 1D array
        * \ingroup functions
    */
    void makeRotationMatrix(float rotation, const char *axis, float (&transform)[16]);

    //! Construct a rotation matrix to perform rotation about an arbitrary axis passing through the origin.
    /** 4x4 Affine rotation matrix
        * \param[in] rotation Rotation angle about axis in radians.
        * \param[in] axis Vector describing axis about which to perform rotation.
        * \param[out] transform Transformation matrix in a 1D array
        * \ingroup functions
    */
    void makeRotationMatrix(float rotation, const helios::vec3 &axis, float (&transform)[16]);

    //! Construct a rotation matrix to perform rotation about an arbitrary line (not necessarily passing through the origin).
    /** 4x4 Affine rotation matrix
     * \param[in] rotation Rotation angle about axis in radians.
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis.
     * \param[in] axis Vector describing axis about which to perform rotation.
     * \param[out] transform Transformation matrix in a 1D array
     * \ingroup functions
    */
    void makeRotationMatrix(float rotation, const helios::vec3 &origin, const helios::vec3 &axis, float (&transform)[16]);

    //! Construct translation matrix
    /** 4x4 Affine translation matrix
        * \param[in] translation Distance to translate in x, y, and z directions.
        * \param[out] transform Transformation matrix in a 1D array
        * \ingroup functions
    */
    void makeTranslationMatrix(const helios::vec3 &translation, float (&transform)[16]);

    //! Construct matrix to scale about the origin
    /** 4x4 Affine scaling matrix
        * \param[in] scale Scaling factor to apply in x, y, and z directions.
        * \param[out] transform Transformation matrix in a 1D array
        * \ingroup functions
    */
    void makeScaleMatrix(const helios::vec3 &scale, float (&transform)[16]);

    //! Construct matrix to scale about arbitrary point in space
    /** 4x4 Affine scaling matrix
        * \param[in] scale Scaling factor to apply in x, y, and z directions.
        * \param[in] point Cartesian coordinates of point about which to scale
        * \param[out] transform Transformation matrix in a 1D array
        * \ingroup functions
    */
    void makeScaleMatrix(const helios::vec3 &scale, const helios::vec3 &point, float (&transform)[16]);

    //! Multiply 4x4 matrices: T=ML*MR
    /**
    * Performs matrix multiplication between two 4x4 matrices (ML and MR) and
    * stores the result in the output matrix T. This function implements the standard
    * matrix multiplication algorithm where each element of the result is the dot product
    * of the corresponding row of ML and column of MR.
    *
    * \param[in] ML The left 4x4 matrix in the multiplication
    * \param[in] MR The right 4x4 matrix in the multiplication
    * \param[out] T Reference to a 4x4 matrix where the result will be stored
    */
    void matmult(const float ML[16], const float MR[16], float (&T)[16]);

    //! Multiply 4x4 transformation matrix by 3-element vector: T=M*v
    /**
     * \param[in] M A 4x4 matrix represented as a 1D array of 16 elements in row-major order
     * \param[in] v A 3D vector represented as an array of 3 elements
     * \param[out] result The resulting 3D vector after matrix-vector multiplication
     */
    void vecmult(const float M[16], const float v[3], float (&result)[3]);

    //! Multiply 4x4 transformation matrix by 3D vector: T=M*v
    /**
     * \param[in] M A 4x4 matrix represented as a float array of size 16
     * \param[in] v3 Input 3D vector to be multiplied with the matrix
     * \param[out] result Output vector containing the result of the multiplication
     */
    void vecmult(const float M[16], const helios::vec3 &v3, helios::vec3 &result);

    //! Construct an identity matrix
    /**
     * \param[in,out] T A reference to a 16-element array representing a flat 4x4 matrix. The array will be filled with the identity matrix values.
     */
    void makeIdentityMatrix(float (&T)[16]);

    //! Convert degrees to radians
    /**
     * \param deg angle in degrees
     * \return angle in radians
     * \sa rad2deg()
     * \ingroup functions
    */
    [[nodiscard ]] float deg2rad(float deg);

    //! Convert radians to degrees
    /**
     * \param rad angle in radians
     * \return angle in degrees
     * \sa deg2rad()
     * \ingroup functions
    */
    [[nodiscard ]] float rad2deg(float rad);

    //! Four quadrant arc tangent between 0 and 2*pi
    /**
     * \note atan2_2pi(0,1) = 0, atan2_2pi(1,0) = pi/2, etc.
     * \ingroup functions
     */
    [[nodiscard ]] float atan2_2pi(float y, float x);

    //! Convert Cartesian coordinates to spherical coordinates
    /**
     * \param[in] Cartesian (x,y,z) Cartesian coordinates
     * \returns SphericalCoord vector.
     * \sa sphere2cart(), SphericalCoord
     * \ingroup functions
    */
    [[nodiscard ]] SphericalCoord cart2sphere(const vec3 &Cartesian);

    //! Convert Spherical coordinates to Cartesian coordinates
    /**
     * \param[in] Spherical (radius,elevation,azimuth) SphericalCoord vector
     * \returns vec3 vector of Cartesian coordinates
     * \sa cart2sphere()
     * \ingroup functions
    */
    [[nodiscard ]] vec3 sphere2cart(const SphericalCoord &Spherical);

    //! Convert a space-delimited string into a vec2 vector
    /**
     * \ingroup functions
     */
    [[nodiscard ]] vec2 string2vec2(const char *str);

    //! Convert a space-delimited string into a vec3 vector
    /**
     * \ingroup functions
     */
    [[nodiscard ]] vec3 string2vec3(const char *str);

    //! Convert a space-delimited string into a vec4 vector
    /**
     * \ingroup functions
     */
    [[nodiscard ]] vec4 string2vec4(const char *str);

    //! Convert a space-delimited string into an int2 vector
    /**
     * \ingroup functions
     */
    [[nodiscard ]] int2 string2int2(const char *str);

    //! Convert a space-delimited string into an int3 vector
    /**
     * \ingroup functions
     */
    [[nodiscard ]] int3 string2int3(const char *str);

    //! Convert a space-delimited string into an int4 vector
    /**
     * \ingroup functions
     */
    [[nodiscard ]] int4 string2int4(const char *str);

    //! Convert a string into a float with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_float Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_float(const std::string &input_string, float &converted_float);

    //! Convert a string into a double with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_double Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_double(const std::string &input_string, double &converted_double);

    //! Convert a string into an integer with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_int Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_int(const std::string &input_string, int &converted_int);

    //! Convert a string into an int2 with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_int2 Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_int2(const std::string &input_string, int2 &converted_int2);

    //! Convert a string into an int3 with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_int3 Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_int3(const std::string &input_string, int3 &converted_int3);

    //! Convert a string into an unsigned integer with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_uint Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_uint(const std::string &input_string, uint &converted_uint);

    //! Convert a string into a vec2 with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_vec2 Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_vec2(const std::string &input_string, vec2 &converted_vec2);

    //! Convert a string into a vec3 with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_vec3 Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_vec3(const std::string &input_string, vec3 &converted_vec3);

    //! Convert a string into an RGBcolor with error checking
    /**
     * \param[in] input_string String to be converted to numerical value
     * \param[out] converted_rgb Output numerical value converted from input string
     * \return True if conversion was successful, false if unsuccessful
     */
    bool parse_RGBcolor(const std::string &input_string, RGBcolor &converted_rgb);

    //! Opens and parses an XML file into a pugi::xml_document object
    /**
    * \param[in] xml_file Path to the XML file to be opened and parsed
    * \param[out] xmldoc Reference to a pugi::xml_document object that will hold the parsed XML data
    * \param[out] error_string Reference to a string that will be populated with error information if the function fails
    * \return true if the XML file was successfully opened and parsed, false otherwise (in case of failure, check error_string for details)
    */
    bool open_xml_file(const std::string &xml_file, pugi::xml_document &xmldoc, std::string &error_string);

    //! Parse an XML tag containing an integer value
    /**
     * \param[in] node XML node containing the tag
     * \param[in] tag Name of the tag to be parsed
     * \param[in] calling_function Name of the function calling this function (for error reporting). e.g., Context::loadXML
     * \return Integer value of the tag
     */
    [[nodiscard]] int parse_xml_tag_int(const pugi::xml_node &node, const std::string &tag, const std::string &calling_function);

    //! Parse an XML tag containing a float value
    /**
     * \param[in] node XML node containing the tag
     * \param[in] tag Name of the tag to be parsed
     * \param[in] calling_function Name of the function calling this function (for error reporting). e.g., Context::loadXML
     * \return Float value of the tag
     */
    [[nodiscard]] float parse_xml_tag_float(const pugi::xml_node &node, const std::string &tag,
                                            const std::string &calling_function);

    //! Parse an XML tag containing a vec2 value (i.e., two space delimited floats)
    /**
     * \param[in] node XML node containing the tag
     * \param[in] tag Name of the tag to be parsed
     * \param[in] calling_function Name of the function calling this function (for error reporting). e.g., Context::loadXML
     * \return vec2 value of the tag
     */
    [[nodiscard]] vec2 parse_xml_tag_vec2(const pugi::xml_node &node, const std::string &tag,
                                          const std::string &calling_function);

    //! Parse an XML tag containing a vec3 value (i.e., three space delimited floats)
    /**
     * \param[in] node XML node containing the tag
     * \param[in] tag Name of the tag to be parsed
     * \param[in] calling_function Name of the function calling this function (for error reporting). e.g., Context::loadXML
     * \return vec3 value of the tag
     */
    [[nodiscard]] vec3 parse_xml_tag_vec3(const pugi::xml_node &node, const std::string &tag,
                                          const std::string &calling_function);

    //! Parse an XML tag containing a string
    /**
     * \param[in] node XML node containing the tag
     * \param[in] tag Name of the tag to be parsed
     * \param[in] calling_function Name of the function calling this function (for error reporting). e.g., Context::loadXML
     * \return String value of the tag
     */
    [[nodiscard]] std::string parse_xml_tag_string(const pugi::xml_node &node, const std::string &tag,
                                                   const std::string &calling_function);

    //! Convert a space-delimited string into an RGBcolor vector
    /**
     * \ingroup functions
     */
    [[nodiscard]] RGBAcolor string2RGBcolor(const char *str);

    //! Remove all whitespace from character array
    /**
     * \ingroup functions
     */
    [[nodiscard]] std::string deblank(const char *input);

    //! Remove all whitespace from a string
    /**
     * \ingroup functions
     */
    [[nodiscard]] std::string deblank(const std::string &input);

    //! Remove leading and trailing whitespace from a string
    /**
     * \ingroup functions
     */
    [[nodiscard]] std::string trim_whitespace(const std::string &input);

    //! Separate string by delimiter and store into a vector
    /**
     * \param[in] inputstring String to be separated
     * \param[in] delimiter Delimiter character for separation
     * \return Vector of substrings. If inputstring does not contain delimiter, empty vector is returned
     * \ingroup functions
     */
    [[nodiscard]] std::vector<std::string> separate_string_by_delimiter(const std::string &inputstring,
                                                                        const std::string &delimiter);

    //! Clamp value to be within some specified bounds
    /**
     * \param[in] value Value to be clamped
     * \param[in] min Lower bound
     * \param[in] max Upper bound
     * \ingroup functions */
    template<typename anytype>
    [[nodiscard]] anytype clamp(anytype value, anytype min, anytype max);

    //! Sum of a vector of floats
    /**
     * \param[in] vect C++ vector of floats
     * \ingroup functions
    */
    [[nodiscard]] float sum(const std::vector<float> &vect);

    //! Mean value of a vector of floats
    /**
     * \param[in] vect C++ vector of floats
     * \ingroup functions
    */
    [[nodiscard]] float mean(const std::vector<float> &vect);

    //! Minimum value of a vector of floats
    /**
     * \param[in] vect C++ vector of floats
     * \ingroup functions
    */
    [[nodiscard]] float min(const std::vector<float> &vect);

    //! Minimum value of a vector of ints
    /**
     * \param[in] vect C++ vector of ints
     * \ingroup functions
    */
    [[nodiscard]] int min(const std::vector<int> &vect);

    //! Minimum value of a vector of vec3's
    /**
     * \param[in] vect C++ vector of vec3's
     * \ingroup functions
    */
    [[nodiscard]] vec3 min(const std::vector<vec3> &vect);

    //! Maximum value of a vector of floats
    /**
     * \param[in] vect C++ vector of floats
     * \ingroup functions
    */
    [[nodiscard]] float max(const std::vector<float> &vect);

    //! Maximum value of a vector of ints
    /**
     * \param[in] vect C++ vector of ints
     * \ingroup functions
    */
    [[nodiscard]] int max(const std::vector<int> &vect);

    //! Maximum value of a vector of vec3's
    /**
     * \param[in] vect C++ vector of vec3's
     * \ingroup functions
    */
    [[nodiscard]] vec3 max(const std::vector<vec3> &vect);

    //! Standard deviation of a vector of floats
    /**
     * \param[in] vect C++ vector of floats
     * \ingroup functions
     */
    [[nodiscard]] float stdev(const std::vector<float> &vect);

    //! Median of a vector of floats
    /**
     * \param[in] vect C++ vector of floats
     * \ingroup functions
     */
    [[nodiscard]] float median(std::vector<float> vect);

    //! Resize 2D C++ vector
    /**
     * \param[in] vec C++ vector
     * \param[in] Nx Size of vector in x-direction, i.e., vec.at(0).size()
     * \param[in] Ny Size of vector in y-direction, i.e., vec.size()
     * \ingroup functions
    */
    template<typename anytype>
    void resize_vector(std::vector<std::vector<anytype> > &vec, uint Nx, uint Ny);

    //! Resize 3D C++ vector
    /**
     * \param[in] vec C++ vector
     * \param[in] Nx Size of vector in x-direction, i.e., vec[0][0].size()
     * \param[in] Ny Size of vector in y-direction, i.e., vec[0].size()
     * \param[in] Nz Size of vector in y-direction, i.e., vec.size()
     * \ingroup functions
    */
    template<typename anytype>
    void resize_vector(std::vector<std::vector<std::vector<anytype> > > &vec, uint Nx, uint Ny, uint Nz);

    //! Resize 4D C++ vector
    /**
     * \param[in] vec C++ vector
     * \param[in] Nx Size of vector in x-direction, i.e., vec[0][0][0].size()
     * \param[in] Ny Size of vector in y-direction, i.e., vec[0][0].size()
     * \param[in] Nz Size of vector in y-direction, i.e., vec[0].size()
     * \param[in] Nw Size of vector in y-direction, i.e., vec.size()
     * \ingroup functions
    */
    template<typename anytype>
    void resize_vector(std::vector<std::vector<std::vector<std::vector<anytype> > > > &vec, uint Nx, uint Ny, uint Nz, uint Nw);

    //! Blend two RGB colors together
    /**
     * \param[in] color0 First color to blend
     * \param[in] color1 Second color to blend
     * \param[in] weight Weighting to apply to each color during blending. E.g., weight=0 will produce color0, weight=0.5 will be an average of color0 and color1, weight=1 will produce color1.
     */
    [[nodiscard]] RGBcolor blend(const RGBcolor &color0, const RGBcolor &color1, float weight);

    //! Blend two RGBA colors together
    /**
     * \param[in] color0 First color to blend
     * \param[in] color1 Second color to blend
     * \param[in] weight Weighting to apply to each color during blending. E.g., weight=0 will produce color0, weight=0.5 will be an average of color0 and color1, weight=1 will produce color1.
     */
    [[nodiscard]] RGBAcolor blend(const RGBAcolor &color0, const RGBAcolor &color1, float weight);

    //! Function to rotate a 3D vector given spherical angles elevation and azimuth
    /**
     * \param[in] position 3D coordinate of point to be rotated.
     * \param[in] rotation Spherical rotation angles (elevation,azimuth)
     * \ingroup functions
    */
    [[nodiscard]] vec3 rotatePoint(const vec3 &position, const SphericalCoord &rotation);

    //! Function to rotate a 3D vector given spherical angles elevation and azimuth
    /**
     * \param[in] position 3D coordinate of point to be rotated.
     * \param[in] theta elevation angle of rotation.
     * \param[in] phi azimuthal angle of rotation.
     * \ingroup functions
    */
    [[nodiscard]] vec3 rotatePoint(const vec3 &position, float theta, float phi);

    //! Rotate a 3D vector about an arbitrary line
    /**
     * \param[in] point 3D coordinate of point to be rotated
     * \param[in] line_base Point through which line passes
     * \param[in] line_direction Unit vector pointing in the direction of the line
     * \param[in] theta Rotation angle in radians
    */
    [[nodiscard]] vec3 rotatePointAboutLine(const vec3 &point, const vec3 &line_base, const vec3 &line_direction,
                                            float theta);

    //! Calculate the area of a triangle given its three vertices
    /**
     * \param[in] v0 (x,y,z) coordinate of first vertex
     * \param[in] v1 (x,y,z) coordinate of second vertex
     * \param[in] v2 (x,y,z) coordinate of third vertex
     * \return One-sided surface area of triangle
     * \ingroup functions
    */
    [[nodiscard]] float calculateTriangleArea(const vec3 &v0, const vec3 &v1, const vec3 &v2);

    //! Convert Julian day to calendar day (day,month,year)
    /**
     * \param[in] Julian_day Julian day of year (1-366)
     * \param[in] year Year in YYYY format
     * \return int3, with int3::x = day, int3::y = month, int3::z = year
     * \sa JulianDay()
    */
    [[nodiscard]] Date CalendarDay(int Julian_day, int year);

    //! Convert calendar day (day,month,year) to Julian day
    /**
     * \param[in] day Day of month (1-31)
     * \param[in] month Month of year (1-12)
     * \param[in] year Year in YYYY format
     * \return Julian day of year
     * \sa CalendarDay()
    */
    [[nodiscard]] int JulianDay(int day, int month, int year);

    //! Convert calendar day (day,month,year) to Julian day
    /**
     * \param[in] date Date vector
     * \return Julian day of year
     * \sa CalendarDay()
     * \ingroup functions
    */
    [[nodiscard]] int JulianDay(const Date &date);

    //! Random number from a uniform distribution between 0 and 1
    /** \ingroup functions */
    [[nodiscard]] float randu();

    //! Random integer from a uniform distribution between imin and imax
    /**
     * \ingroup functions
     */
    [[nodiscard]] int randu(int imin, int imax);

    //! arccosine function to handle cases when round-off errors cause an argument <-1 or >1, and thus regular acos() returns NaN
    /**
     * \ingroup functions
     */
    [[nodiscard]] float acos_safe(float x);

    //! arcsine function to handle cases when round-off errors cause an argument <-1 or >1, and thus regular asin() returns NaN
    /**
     * \ingroup functions
     */
    [[nodiscard]] float asin_safe(float x);

    //! Exponentiation function for integer powers
    /**
     * \param[in] base Base to be raised to a power
     * \param[in] exp Exponent to which the base is raised (must be a non-negative integer)
     */
    template<typename T>
    T powi(T base, std::size_t exp);

    //!Determine if two line segments intersect. The lines segments are defined by vertices (p1,q1) and (p2,q2)
    /**
     * \ingroup functions
     */
    [[nodiscard]] bool lineIntersection(const vec2 &p1, const vec2 &q1, const vec2 &p2, const vec2 &q2);

    //!Determine whether point lines within a polygon
    /**
     * \ingroup functions
     */
    [[nodiscard]] bool pointInPolygon(const vec2 &point, const std::vector<vec2> &polygon_verts);

    //! MATLAB-style timer.  Call tic() to start timer, call toc() to stop timer and print duration
    /**
     * \ingroup functions
     */
    struct Timer {
    public:
        Timer() {
            running = false;
        }

        //! Start timer
        void tic() {
            timer_start = std::chrono::high_resolution_clock::now();
            running = true;
        };

        //! Stop timer and print elapsed time
        double toc() const {
            return toc("");
        }

        //! Stop timer and print elapsed time and a user-defined message
        /**
         * \param[in] message Message to print with elapsed time
         * \note the timer print message can be turned off by passing the message argument  "mute"
        */
        double toc(const char *message) const {
            if (!running) {
                std::cerr << "ERROR (Timer): You must call `tic' before calling `toc'. Ignoring call to `toc'..." <<
                        std::endl;
                return 0;
            }

            auto timer_end = std::chrono::high_resolution_clock::now();;
            double duration = std::chrono::duration<double>(timer_end - timer_start).count();
            if (strcmp(message, "mute") != 0) {
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
    void wait(float seconds);

    //! Check whether PNG image file has an alpha/transparency channel
    /**
     * \param[in] filename Name of the PNG image file
     */
    [[nodiscard]] bool PNGHasAlpha(const char *filename);

    //! Function to read the alpha channel from a PNG image
    /**
     * \param[in] filename Name of the PNG image file
     * \return 2D mask, where false denotes no material
    */
    [[nodiscard]] std::vector<std::vector<bool> > readPNGAlpha(const std::string &filename);

    //! Function to read a PNG image file into pixel data array
    /**
     * \param[in] filename Name of the PNG image file
     * \param[out] width Image width in pixels
     * \param[out] height Image height in pixels
     * \param[out] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
     */
    void readPNG(const std::string &filename, uint &width, uint &height, std::vector<helios::RGBAcolor> &pixel_data);

    //! Function to write a PNG image based on pixel data
    /**
    * \param[in] filename Name of the PNG image file
    * \param[in] width Image width in pixels
    * \param[in] height Image height in pixels
    * \param[in] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
    */
    void writePNG(const std::string &filename, uint width, uint height, const std::vector<helios::RGBAcolor> &pixel_data);

    //! Function to write a PNG image based on pixel data
    /**
    * \param[in] filename Name of the PNG image file
    * \param[in] width Image width in pixels
    * \param[in] height Image height in pixels
    * \param[in] pixel_data pixel_data Flat 1D array of pixel color data (index at pixel_data[(row*width+column)*width*height+channel]).
        * \note The length of pixel_data must be width*height*4 (or if it is width*height*3, the last channel is assumed opaque).
    */
    void writePNG(const std::string &filename, uint width, uint height, const std::vector<unsigned char> &pixel_data);

    //! Function to read a JPEG image file into pixel data array
    /**
        * \param[in] filename Name of the JPEG image file
        * \param[out] width Image width in pixels
        * \param[out] height Image height in pixels
        * \param[out] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
    */
    void readJPEG(const std::string &filename, uint &width, uint &height, std::vector<helios::RGBcolor> &pixel_data);

    //! Function to read a JPEG image file into pixel data array
    /**
        * \param[in] filename Name of the JPEG image file
    */
    [[nodiscard]] helios::int2 getImageResolutionJPEG(const std::string &filename);

    //! Function to write a JPEG image based on pixel data
    /**
        * \param[in] filename Name of the JPEG image file
        * \param[in] width Image width in pixels
        * \param[in] height Image height in pixels
        * \param[in] pixel_data Colors at each pixel (index at pixel_data[row*width+column])
    */
    void writeJPEG(const std::string &filename, uint width, uint height, const std::vector<helios::RGBcolor> &pixel_data);

    //! Function to write a JPEG image based on pixel data
    /**
        * \param[in] filename Name of the JPEG image file
        * \param[in] width Image width in pixels
        * \param[in] height Image height in pixels
        * \param[in] pixel_data Flat 1D array of pixel color data (index at pixel_data[(row*width+column)*width*height+channel]).
        * \note The length of pixel_data must be width*height*3 (or if it is width*height*4, the last channel is ignored).
    */
    void writeJPEG(const std::string &filename, uint width, uint height, const std::vector<unsigned char> &pixel_data);

    //! Template function to flatten a 2D vector into a 1D vector
    /**
    * \ingroup functions
    * @tparam T Type of elements in the vector
    */
    template <typename T>
    [[nodiscard]] std::vector<T> flatten(const std::vector<std::vector<T>>& vec);

    //! Template function to flatten a 3D vector into a 1D vector
    /**
    * \ingroup functions
    * @tparam T Type of elements in the vector
    */
    template <typename T>
    [[nodiscard]] std::vector<T> flatten(const std::vector<std::vector<std::vector<T>>>& vec);

    //! Template function to flatten a 4D vector into a 1D vector
    /**
    * \ingroup functions
    * @tparam T Type of elements in the vector
    */
    template <typename T>
    [[nodiscard]] std::vector<T> flatten(const std::vector<std::vector<std::vector<std::vector<T>>>>& vec);


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
    [[nodiscard]] helios::vec3 spline_interp3(float u, const vec3 &x_start, const vec3 &tan_start, const vec3 &x_end,
                                const vec3 &tan_end);

    //! Function to load and convert a field in a pugi XML node into a float
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] float XMLloadfloat(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into an int
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] int XMLloadint(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into a string
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] std::string XMLloadstring(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into a vec2
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] helios::vec2 XMLloadvec2(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into a vec3
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] helios::vec3 XMLloadvec3(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into a vec4
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] helios::vec4 XMLloadvec4(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into a vec2
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] helios::int2 XMLloadint2(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into a int3
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] helios::int3 XMLloadint3(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into a int4
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] helios::int4 XMLloadint4(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into an RGB color vector
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
    [[nodiscard]] helios::RGBcolor XMLloadrgb(pugi::xml_node node, const char *field);

    //! Function to load and convert a field in a pugi XML node into an RGBA color vector
    /**
     * \param[in] node Pugi XML node
     * \param[in] field String corresponding to field value to be read
    */
   [[nodiscard]]  helios::RGBAcolor XMLloadrgba(pugi::xml_node node, const char *field);

    //! Use Newton-Raphson method to find the zero of a function
    /**
     * \param[in] function Function to be evaluated. The function should take as its first argument the value at which the function should be evaluated, as second argument any function arguments.
     * \param[in] variables Vector of function arguments
     * \param[in] parameters Pointer to any additional parameters needed by the function
     * \param[in] init_guess Initial guess for the zero of the function.
     * \param[in] err_tol [optional] Maximum allowable relative error in solution.
     * \param[in] max_iterations [optional] Maximum number of iterations to allow before exiting solver.
     */
    [[nodiscard]] float fzero(float (*function)(float value, std::vector<float> &variables, const void *parameters),
                std::vector<float> &variables, const void *parameters, float init_guess, float err_tol = 0.0001f,
                int max_iterations = 100);

    //! Function to perform linear interpolation based on a vector of discrete (x,y) values
    /**
      * \param[in] points Vector of (x,y) pairs. x values must be monotonically increasing and not duplicated.
      * \param[in] x x value at which to interpolate. If x is less than the first x-value in points or greater than the last x-value in points, interp1 will respectively return the first or last y value.
      * \return Value of y interpolated at location x.
     */
    [[nodiscard]] float interp1(const std::vector<helios::vec2> &points, float x);

    //! Function to calculate the distance between two points
    /**
      * \param[in] p1 first point (vec3)
      * \param[in] p2 second point (vec3)
      * \return distance between p1 and p2 in three dimensions
     */
    [[nodiscard]] float point_distance(const helios::vec3 &p1, const helios::vec3 &p2);

    //! Parse a file string to get the extension
    /**
      * Example: /path/to/file/filename.ext -> extension=".ext"
      * \param[in] filepath String containing file path and name
      * \return File extension contained in input string
      * \note If the string does not contain an extension, an empty string is returned
     */
    [[nodiscard]] std::string getFileExtension(const std::string &filepath);

    //! Parse a file string to get the filename without extension
    /**
      * Example: /path/to/file/filename.ext -> stem="filename"
      * \param[in] filepath String containing file path and name
      * \return Filename without extension contained in input string
      * \note If the string does not contain a file name, an empty string is returned
     */
    [[nodiscard]] std::string getFileStem(const std::string &filepath);

    //! Parse a file string to get the filename including extension
    /**
      * Example: /path/to/file/filename.ext -> filename="filename.ext"
      * \param[in] filepath String containing file path and name
      * \return Filename including extension contained in input string
      * \note If the string does not contain a file name, an empty string is returned
     */
    [[nodiscard]] std::string getFileName(const std::string &filepath);

    //! Parse a file string to get the path (i.e., portion of the string before the file name).
    /**
      * Example: /path/to/file/filename.ext -> path="/path/to/file/"
      * \param[in] filepath String containing file path and name
      * \param[in] trailingslash If true, a trailing slash is added to the path
      * \return File path contained in input string
      * \note If the string does not contain a path, an empty string is returned
     */
    [[nodiscard]] std::string getFilePath(const std::string &filepath, bool trailingslash = true);

    //! Check whether output file and/or directory is valid. Add a trailing slash if it is a directory.
    /**
     * \param[inout] output_directory String containing the output directory or file name
     * \param[in] allowable_file_extensions Vector of allowable file extensions. If the output_directory is a file, it must have one of these extensions.
     * \return True if directory/file was valid, false otherwise
     */
    [[nodiscard]] bool validateOutputPath(std::string &output_directory, const std::vector<std::string> &allowable_file_extensions = {});

    //! Read values contained in a text file into a one-dimensional vector of floats
    /**
      * \param[in] filepath Path to text file
      * \return One-dimensional vector of floats contained in text file
     */
    [[nodiscard]] std::vector<float> importVectorFromFile(const std::string &filepath);

    //! Sample from Beta(μ,ν) distribution
    /**
        *  Sample θ_L ∈ [0, π/2] from beta distribution given u∼Uniform[0,1]
        *  \param[in] mu First parameter of beta distribution
        *  \param[in] nu Second parameter of beta distribution
        *  \param[in] generator Random number generator
    */
    [[nodiscard]] float sample_Beta_distribution(float mu, float nu, std::minstd_rand0 *generator);

    /**
     * \brief Samples an azimuth angle for an ellipsoidal distribution.
     *
     * \param[in] e Eccentricity of the ellipsoidal distribution.
     * \param[in] phi0_degrees Azimuthal rotation of ellipse (degrees).
     * \param[in] generator Random number generator.
     * \return The sampled azimuth angle.
     */
    [[nodiscard]] float sample_ellipsoidal_azimuth(float e, float phi0_degrees, std::minstd_rand0 *generator);

    inline std::vector<float>& operator+=(std::vector<float>& lhs, const std::vector<float>& rhs) {
        // Make sure vectors have the same size
        if (lhs.size() != rhs.size()) {
            throw std::invalid_argument("Vector sizes must match for element-wise addition");
        }

        // Perform element-by-element addition
        for (size_t i = 0; i < lhs.size(); ++i) {
            lhs[i] += rhs[i];
        }

        return lhs;
    }

    inline std::vector<float> operator+(const std::vector<float>& vector1, const std::vector<float>& vector2)
    {
        if (vector1.size() != vector2.size()) {
            throw std::invalid_argument("Vector sizes must match for element-wise addition");
        }

        std::vector<float> result(vector1.size());
        for (std::size_t i = 0; i < vector1.size(); ++i) {
            result[i] = vector1[i] + vector2[i];
        }
        return result;
    }

    /**
     * \class PixelUVKey
     * \brief Represents a unique key for a pixel using its UV coordinates.
     *
     * Combines the U and V coordinates of a pixel into a unique identifier, which is then used for a hash table lookup to see if these (u,v)'s have been used before for a given texture.
     */
    struct PixelUVKey {
        std::vector<int> coords;            // {x0,y0, x1,y1, …}
        bool operator==(PixelUVKey const &o) const noexcept {
            return coords == o.coords;
        }
    };

    /**
     * \brief A hash function object for PixelUVKey.
     *
     * Provides a way to generate unique hash values for PixelUVKey objects.
     */
    struct PixelUVKeyHash {
        size_t operator()(PixelUVKey const &k) const noexcept {
            uint64_t h = 146527;  // arbitrary seed
            for(int v: k.coords) {
                // mix in v
                h ^= uint64_t(v) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            }
            return size_t(h);
        }
    };

    //! Default null SphericalCoord that applies no rotation
    extern SphericalCoord nullrotation;
    //! Default null vec3 that gives the origin (0,0,0)
    extern vec3 nullorigin;
}

#endif
