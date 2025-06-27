/** \file "helios_vector_types.h" Declarations for vector types

    Copyright (C) 2016-2025 Brian Bailey

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int uint;

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
// #include <stdexcept>
#include <cmath>
#include <iomanip>
#include <limits>

//! Vector data structures
/**
 * \defgroup vectors Vectors
 */

namespace helios {

    //! Vector of two elements of type 'int'.
    /**
     * \ingroup vectors
     */
    struct int2 {

        //! First element in vector
        int x;
        //! Second element in vector
        int y;

        //! Default constructor
        /* initializes to zero */
        constexpr int2() : x(0), y(0) {
        }

        //! Construct int2 from a vector of ints
        explicit int2(const std::vector<int> &v) {
            if (v.size() != 2) {
                throw(std::runtime_error("ERROR: vector for initialization of int2 must have length 2."));
            }
            x = v[0];
            y = v[1];
        }
        //! Construct int2 from an array of ints
        constexpr explicit int2(const int v[2]) : x(v[0]), y(v[1]) {
        }

        //! Construct int2 from two ints
        constexpr int2(int v0, int v1) : x(v0), y(v1) {
        }


        //! Add two int2 vectors
        constexpr int2 operator+(const int2 &a) const noexcept;
        //! Increment int2 vector
        inline int2 &operator+=(const int2 &a) noexcept;
        //! Decrement int2 vector
        inline int2 &operator-=(const int2 &a) noexcept;
        //! Subtract two int2 vectors
        constexpr int2 operator-(const int2 &a) const noexcept;
        //! Equality for all elements
        constexpr bool operator==(const int2 &a) const noexcept;
        //! Inequality for all elements
        constexpr bool operator!=(const int2 &a) const noexcept;
        //! Multiply each element by -1
        constexpr int2 operator-() const noexcept;

        //! Write int2 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::int2 &vec) {
            return os << "helios::int2<" << vec.x << ", " << vec.y << ">";
        }
    };

    //! Make an int2 vector from two ints
    /**
     * \param[in] x First component of integer vector
     * \param[in] y First component of integer vector
     * \ingroup vectors
     */
    constexpr int2 make_int2(int x, int y) {
        return {x, y};
    }

    //! Make an int2 vector from an array of ints
    /**
     * \param[in] X 2D array if integers
     * \ingroup vectors
     */
    inline int2 make_int2(int X[2]) {
        return {X[0], X[1]};
    }

    constexpr int2 int2::operator+(const int2 &a) const noexcept {
        return {a.x + x, a.y + y};
    }

    inline int2 &int2::operator+=(const int2 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        return *this;
    }

    inline int2 &int2::operator-=(const int2 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        return *this;
    }

    constexpr int2 int2::operator-(const int2 &a) const noexcept {
        return {x - a.x, y - a.y};
    }

    constexpr bool int2::operator==(const int2 &a) const noexcept {
        return a.x == x && a.y == y;
    }

    constexpr bool int2::operator!=(const int2 &a) const noexcept {
        return a.x != x || a.y != y;
    }

    constexpr int2 int2::operator-() const noexcept {
        return {-x, -y};
    }

    //! Vector of three elements of type 'int'.
    /** \ingroup vectors */
    struct int3 {

        //! First element in vector
        int x;
        //! Second element in vector
        int y;
        //! Third element in vector
        int z;

        //! Default constructor
        /* initializes to zero */
        constexpr int3() : x(0), y(0), z(0) {
        }

        //! Construct int3 from a vector of ints
        explicit int3(const std::vector<int> &v) {
            if (v.size() != 3) {
                throw(std::runtime_error("ERROR: vector for initialization of int3 must have length 3."));
            }
            x = v[0];
            y = v[1];
            z = v[2];
        }
        //! Construct int3 from an array of ints
        constexpr explicit int3(const int v[3]) : x(v[0]), y(v[1]), z(v[2]) {
        }

        //! Construct int3 from three ints
        constexpr int3(int v0, int v1, int v2) : x(v0), y(v1), z(v2) {
        }

        //! Add two int3 vectors
        constexpr int3 operator+(const int3 &a) const noexcept;
        //! Increment int3 vector
        inline int3 &operator+=(const int3 &a) noexcept;
        //! Decrement int3 vector
        inline int3 &operator-=(const int3 &a) noexcept;
        //! Subtract two int3 vectors
        constexpr int3 operator-(const int3 &a) const noexcept;
        //! Equality for all elements
        constexpr bool operator==(const int3 &a) const noexcept;
        //! Inequality for all elements
        constexpr bool operator!=(const int3 &a) const noexcept;
        //! Multiply each element by -1
        constexpr int3 operator-() const noexcept;

        //! Write int3 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::int3 &vec) {
            return os << "helios::int3<" << vec.x << ", " << vec.y << ", " << vec.z << ">";
        }
    };

    //! Make an int3 vector from three ints
    constexpr int3 make_int3(int X, int Y, int Z) {
        return {X, Y, Z};
    }

    //! Make an int2 vector from an array of ints
    constexpr int3 make_int3(int X[3]) {
        return {X[0], X[1], X[2]};
    }

    constexpr int3 int3::operator+(const int3 &a) const noexcept {
        return {a.x + x, a.y + y, a.z + z};
    }

    inline int3 &int3::operator+=(const int3 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        return *this;
    }

    inline int3 &int3::operator-=(const int3 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        this->z -= a.z;
        return *this;
    }

    constexpr int3 int3::operator-(const int3 &a) const noexcept {
        return {x - a.x, y - a.y, z - a.z};
    }

    constexpr bool int3::operator==(const int3 &a) const noexcept {
        return a.x == x && a.y == y && a.z == z;
    }

    constexpr bool int3::operator!=(const int3 &a) const noexcept {
        return a.x != x || a.y != y || a.z != z;
    }

    constexpr int3 int3::operator-() const noexcept {
        return {-x, -y, -z};
    }

    //! Vector of four elements of type 'int'.
    /** \ingroup vectors */
    struct int4 {


        //! First element in vector
        int x;
        //! Second element in vector
        int y;
        //! Third element in vector
        int z;
        //! Fourth element in vector
        int w;

        //! Default constructor
        /* initializes to zero */
        constexpr int4() : x(0), y(0), z(0), w(0) {
        }

        //! Construct int3 from a vector of ints
        explicit int4(const std::vector<int> &v) {
            if (v.size() != 4) {
                throw(std::runtime_error("ERROR: vector for initialization of int4 must have length 4."));
            }
            x = v[0];
            y = v[1];
            z = v[2];
            w = v[3];
        }

        //! Construct int4 from an array of ints
        explicit constexpr int4(const int v[4]) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {
        }

        //! Construct int4 from four ints
        constexpr int4(int v0, int v1, int v2, int v3) : x(v0), y(v1), z(v2), w(v3) {
        }

        //! Add two int4 vectors
        constexpr int4 operator+(const int4 &a) const noexcept;
        //! Increment int4 vector
        inline int4 &operator+=(const int4 &a) noexcept;
        //! Decrement int4 vector
        inline int4 &operator-=(const int4 &a) noexcept;
        //! Subtract two int4 vectors
        constexpr int4 operator-(const int4 &a) const noexcept;
        //! Equality for all elements
        constexpr bool operator==(const int4 &a) const noexcept;
        //! Inequality for all elements
        constexpr bool operator!=(const int4 &a) const noexcept;
        //! Multiply each element by -1
        constexpr int4 operator-() const noexcept;

        //! Write int4 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::int4 &vec) {
            return os << "helios::int4<" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ">";
        }
    };

    //! Make an int4 vector from three ints
    constexpr int4 make_int4(int x, int y, int z, int w) {
        return {x, y, z, w};
    }

    //! Make an int4 vector from an array of ints
    constexpr int4 make_int4(const int X[4]) {
        return {X[0], X[1], X[2], X[3]};
    }

    constexpr int4 int4::operator+(const int4 &a) const noexcept {
        return {a.x + x, a.y + y, a.z + z, a.w + w};
    }

    inline int4 &int4::operator+=(const int4 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        this->w += a.w;
        return *this;
    }

    inline int4 &int4::operator-=(const int4 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        this->z -= a.z;
        this->w -= a.w;
        return *this;
    }

    constexpr int4 int4::operator-(const int4 &a) const noexcept {
        return {a.x - x, a.y - y, a.z - z, a.w - w};
    }

    constexpr bool int4::operator==(const int4 &a) const noexcept {
        return a.x == x && a.y == y && a.z == z && a.w == w;
    }

    constexpr bool int4::operator!=(const int4 &a) const noexcept {
        return a.x != x || a.y != y || a.z != z || a.w != w;
    }

    constexpr int4 int4::operator-() const noexcept {
        return {-x, -y, -z, -w};
    }

    //! Vector of two elements of type 'unsigned int'.
    /**
     * \ingroup vectors
     */
    struct uint2 {


        //! First element in vector
        unsigned int x;
        //! Second element in vector
        unsigned int y;

        //! Default constructor
        /* initializes to zero */
        constexpr uint2() : x(0u), y(0u) {
        }

        //! Construct uint2 from a vector of unsigned ints
        explicit uint2(const std::vector<unsigned int> &v) {
            if (v.size() != 2) {
                throw(std::runtime_error("ERROR: vector for initialization of uint2 must have length 2."));
            }
            x = v[0];
            y = v[1];
        }
        //! Construct uint2 from an array of unsigned ints
        constexpr explicit uint2(const unsigned int v[2]) : x(v[0]), y(v[1]) {
        }

        //! Construct uint2 from two unsigned ints
        constexpr uint2(unsigned int v0, unsigned int v1) : x(v0), y(v1) {
        }


        //! Add two uint2 vectors
        constexpr uint2 operator+(const uint2 &a) const noexcept;
        //! Increment uint2 vector
        inline uint2 &operator+=(const uint2 &a) noexcept;
        //! Decrement uint2 vector
        inline uint2 &operator-=(const uint2 &a) noexcept;
        //! Subtract two uint2 vectors
        constexpr uint2 operator-(const uint2 &a) const noexcept;
        //! Equality for all elements
        constexpr bool operator==(const uint2 &a) const noexcept;
        //! Inequality for all elements
        constexpr bool operator!=(const uint2 &a) const noexcept;
        //! Multiply each element by zero minus itself (wraps)
        constexpr uint2 operator-() const noexcept;

        //! Write uint2 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::uint2 &vec) {
            return os << "helios::uint2<" << vec.x << ", " << vec.y << ">";
        }
    };

    //! Make a uint2 vector from two unsigned ints
    /**
     * \param[in] x First component of unsigned integer vector
     * \param[in] y Second component of unsigned integer vector
     * \ingroup vectors
     */
    constexpr uint2 make_uint2(unsigned int x, unsigned int y) {
        return {x, y};
    }

    //! Make a uint2 vector from an array of unsigned ints
    /**
     * \param[in] X 2D array of unsigned integers
     * \ingroup vectors
     */
    constexpr uint2 make_uint2(unsigned int X[2]) {
        return {X[0], X[1]};
    }

    constexpr uint2 uint2::operator+(const uint2 &a) const noexcept {
        return {a.x + x, a.y + y};
    }

    inline uint2 &uint2::operator+=(const uint2 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        return *this;
    }

    inline uint2 &uint2::operator-=(const uint2 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        return *this;
    }

    constexpr uint2 uint2::operator-(const uint2 &a) const noexcept {
        return {x - a.x, y - a.y};
    }

    constexpr bool uint2::operator==(const uint2 &a) const noexcept {
        return a.x == x && a.y == y;
    }

    constexpr bool uint2::operator!=(const uint2 &a) const noexcept {
        return a.x != x || a.y != y;
    }

    constexpr uint2 uint2::operator-() const noexcept {
        return {-x, -y};
    }


    //! Vector of three elements of type 'unsigned int'.
    /**
     * \ingroup vectors
     */
    struct uint3 {


        //! First element in vector
        unsigned int x;
        //! Second element in vector
        unsigned int y;
        //! Third element in vector
        unsigned int z;

        //! Default constructor
        /**
         * initializes to zero
         */
        constexpr uint3() : x(0u), y(0u), z(0u) {
        }

        //! Construct uint3 from a vector of unsigned ints
        explicit uint3(const std::vector<unsigned int> &v) {
            if (v.size() != 3) {
                throw(std::runtime_error("ERROR: vector for initialization of uint3 must have length 3."));
            }
            x = v[0];
            y = v[1];
            z = v[2];
        }
        //! Construct uint3 from an array of unsigned ints
        constexpr explicit uint3(const unsigned int v[3]) : x(v[0]), y(v[1]), z(v[2]) {
        }

        //! Construct uint3 from three unsigned ints
        constexpr uint3(unsigned int v0, unsigned int v1, unsigned int v2) : x(v0), y(v1), z(v2) {
        }

        //! Add two uint3 vectors
        constexpr uint3 operator+(const uint3 &a) const noexcept;
        //! Increment uint3 vector
        inline uint3 &operator+=(const uint3 &a) noexcept;
        //! Decrement uint3 vector
        inline uint3 &operator-=(const uint3 &a) noexcept;
        //! Subtract two uint3 vectors
        constexpr uint3 operator-(const uint3 &a) const noexcept;
        //! Equality for all elements
        constexpr bool operator==(const uint3 &a) const noexcept;
        //! Inequality for all elements
        constexpr bool operator!=(const uint3 &a) const noexcept;
        //! Multiply each element by zero minus itself (wraps)
        constexpr uint3 operator-() const noexcept;

        //! Write uint3 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::uint3 &vec) {
            return os << "helios::uint3<" << vec.x << ", " << vec.y << ", " << vec.z << ">";
        }
    };

    //! Make a uint3 vector from three unsigned ints
    /**
     * \param[in] X First component of unsigned integer vector
     * \param[in] Y Second component of unsigned integer vector
     * \param[in] Z Third component of unsigned integer vector
     * \ingroup vectors
     */
    constexpr uint3 make_uint3(unsigned int X, unsigned int Y, unsigned int Z) {
        return {X, Y, Z};
    }

    //! Make a uint3 vector from an array of unsigned ints
    /**
     * \param[in] X 3D array of unsigned integers
     * \ingroup vectors
     */
    constexpr uint3 make_uint3(unsigned int X[3]) {
        return {X[0], X[1], X[2]};
    }

    constexpr uint3 uint3::operator+(const uint3 &a) const noexcept {
        return {a.x + x, a.y + y, a.z + z};
    }

    inline uint3 &uint3::operator+=(const uint3 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        return *this;
    }

    inline uint3 &uint3::operator-=(const uint3 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        this->z -= a.z;
        return *this;
    }

    constexpr uint3 uint3::operator-(const uint3 &a) const noexcept {
        return {x - a.x, y - a.y, z - a.z};
    }

    constexpr bool uint3::operator==(const uint3 &a) const noexcept {
        return a.x == x && a.y == y && a.z == z;
    }

    constexpr bool uint3::operator!=(const uint3 &a) const noexcept {
        return a.x != x || a.y != y || a.z != z;
    }

    constexpr uint3 uint3::operator-() const noexcept {
        return {-x, -y, -z};
    }


    //! Vector of four elements of type 'unsigned int'.
    /**
     * \ingroup vectors
     */
    struct uint4 {

        //! First element in vector
        unsigned int x;
        //! Second element in vector
        unsigned int y;
        //! Third element in vector
        unsigned int z;
        //! Fourth element in vector
        unsigned int w;

        //! Default constructor
        /**
         * initializes to zero
         */
        constexpr uint4() : x(0u), y(0u), z(0u), w(0u) {
        }

        //! Construct uint4 from a vector of unsigned ints
        explicit uint4(const std::vector<unsigned int> &v) {
            if (v.size() != 4) {
                throw(std::runtime_error("ERROR: vector for initialization of uint4 must have length 4."));
            }
            x = v[0];
            y = v[1];
            z = v[2];
            w = v[3];
        }

        //! Construct uint4 from an array of unsigned ints
        constexpr explicit uint4(const unsigned int v[4]) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {
        }

        //! Construct uint4 from four unsigned ints
        constexpr uint4(unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3) : x(v0), y(v1), z(v2), w(v3) {
        }

        //! Add two uint4 vectors
        constexpr uint4 operator+(const uint4 &a) const noexcept;
        //! Increment uint4 vector
        inline uint4 &operator+=(const uint4 &a) noexcept;
        //! Decrement uint4 vector
        inline uint4 &operator-=(const uint4 &a) noexcept;
        //! Subtract two uint4 vectors
        constexpr uint4 operator-(const uint4 &a) const noexcept;
        //! Equality for all elements
        constexpr bool operator==(const uint4 &a) const noexcept;
        //! Inequality for all elements
        constexpr bool operator!=(const uint4 &a) const noexcept;
        //! Multiply each element by zero minus itself (wraps)
        constexpr uint4 operator-() const noexcept;

        //! Write uint4 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::uint4 &vec) {
            return os << "helios::uint4<" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ">";
        }
    };

    //! Make a uint4 vector from four unsigned ints
    /**
     * \param[in] x First component of unsigned integer vector
     * \param[in] y Second component of unsigned integer vector
     * \param[in] z Third component of unsigned integer vector
     * \param[in] w Fourth component of unsigned integer vector
     * \ingroup vectors
     */
    constexpr uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
        return {x, y, z, w};
    }

    //! Make a uint4 vector from an array of unsigned ints
    /**
     * \param[in] X 4D array of unsigned integers
     * \ingroup vectors
     */
    constexpr uint4 make_uint4(const unsigned int X[4]) {
        return {X[0], X[1], X[2], X[3]};
    }

    constexpr uint4 uint4::operator+(const uint4 &a) const noexcept {
        return {a.x + x, a.y + y, a.z + z, a.w + w};
    }

    inline uint4 &uint4::operator+=(const uint4 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        this->w += a.w;
        return *this;
    }

    inline uint4 &uint4::operator-=(const uint4 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        this->z -= a.z;
        this->w -= a.w;
        return *this;
    }

    constexpr uint4 uint4::operator-(const uint4 &a) const noexcept {
        return {x - a.x, y - a.y, z - a.z, w - a.w};
    }

    constexpr bool uint4::operator==(const uint4 &a) const noexcept {
        return a.x == x && a.y == y && a.z == z && a.w == w;
    }

    constexpr bool uint4::operator!=(const uint4 &a) const noexcept {
        return a.x != x || a.y != y || a.z != z || a.w != w;
    }

    constexpr uint4 uint4::operator-() const noexcept {
        return {static_cast<unsigned int>(0u - x), static_cast<unsigned int>(0u - y), static_cast<unsigned int>(0u - z), static_cast<unsigned int>(0u - w)};
    }

    //! Vector of two elements of type 'float'.
    /**
     * \ingroup vectors
     */
    struct vec2 {


        //! First element in vector
        float x;
        //! Second element in vector
        float y;

        //! Normalize vector components such that the magnitude is unity.
        vec2 normalize() {
            const float m2 = x * x + y * y;
            if (m2 <= 0)
                return *this;
            const float inv = 1.0f / sqrtf(m2);
            this->x *= inv;
            this->y *= inv;
            return *this;
        }

        //! Compute the vector magnitude
        /**
         * \return magnitude of vector
         */
        [[nodiscard]] float magnitude() const {
            return sqrtf(x * x + y * y);
        }

        //! Default constructor
        /**
         * initializes to zero
         */
        constexpr vec2() : x(0), y(0) {
        }

        //! Initialize vec2 using a vector of floats
        explicit vec2(const std::vector<float> &v) {
            if (v.size() != 2) {
                throw(std::runtime_error("ERROR: vector for initialization of vec2 must have length 2."));
            }
            x = v[0];
            y = v[1];
        }
        //! Initialize vec2 using an array of floats
        explicit constexpr vec2(const float v[2]) : x(v[0]), y(v[1]) {
        }

        //! Initialize vec2 using two floats
        constexpr vec2(float v0, float v1) : x(v0), y(v1) {
        }

        //! Dot (scalar) product of two vec2 vectors
        constexpr float operator*(const vec2 &a) const noexcept;
        //! Sum of two vec2 vectors
        constexpr vec2 operator+(const vec2 &a) const noexcept;
        //! Increment vec2 vector
        inline vec2 &operator+=(const vec2 &a) noexcept;
        //! Decrement vec2 vector
        inline vec2 &operator-=(const vec2 &a) noexcept;
        //! Difference of two vec2 vectors noexcept
        constexpr vec2 operator-(const vec2 &a) const noexcept;
        //! Multiply each element by scalar (scalar is multiplied on right: vec2*a)
        constexpr vec2 operator*(float a) const noexcept;
        //! Divide each element by scalar (scalar is divided on right: vec2/a)
        inline vec2 operator/(float a) const noexcept;
        //! Add a scalar to each element (scalar is added on right: vec2+a)
        constexpr vec2 operator+(float a) const noexcept;
        //! Subtract a scalar from each element (scalar is subtracted on right: vec2-a)
        constexpr vec2 operator-(float a) const noexcept;
        //! check for equality of two vec2 vectors
        constexpr bool operator==(const vec2 &a) const noexcept;
        //! check for inequality of two vec2 vectors
        constexpr bool operator!=(const vec2 &a) const noexcept;
        //! Multiply each element by -1
        constexpr vec2 operator-() const noexcept;

        //! Write vec2 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::vec2 &vec) {
            return os << "helios::vec2<" << vec.x << ", " << vec.y << ">";
        }
    };

    //! Multiply each element by scalar (scalar is multiplied on left: a*vec2)
    constexpr vec2 operator*(float a, const vec2 &v) noexcept;
    //! Add a scalar to each element (scalar is added on left: a+vec2)
    constexpr vec2 operator+(float a, const vec2 &v) noexcept;
    //! Subtract a scalar from each element (scalar is subtracted on left: a-vec2)
    constexpr vec2 operator-(float a, const vec2 &v) noexcept;

    //! Make a vec2 from two floats
    /**
     * \param[in] x First element of vector
     * \param[in] y Second element of vector
     * \ingroup vectors
     */
    constexpr vec2 make_vec2(float x, float y) {
        return {x, y};
    }

    inline vec2 normalize(const vec2 &v) {
        const float m2 = v.x * v.x + v.y * v.y;
        const float inv = 1.0f / sqrtf(m2);
        return {v.x * inv, v.y * inv};
    }

    constexpr float vec2::operator*(const vec2 &a) const noexcept {
        return a.x * x + a.y * y;
    }

    constexpr vec2 vec2::operator+(const vec2 &a) const noexcept {
        return {a.x + x, a.y + y};
    }

    inline vec2 &vec2::operator+=(const vec2 &a) noexcept {
        x += a.x;
        y += a.y;
        return *this;
    }

    inline vec2 &vec2::operator-=(const vec2 &a) noexcept {
        x -= a.x;
        y -= a.y;
        return *this;
    }

    constexpr vec2 vec2::operator+(const float a) const noexcept {
        return {a + x, a + y};
    }

    constexpr vec2 operator+(const float a, const vec2 &v) noexcept {
        return {a + v.x, a + v.y};
    }

    constexpr vec2 vec2::operator-(const vec2 &a) const noexcept {
        return {x - a.x, y - a.y};
    }

    constexpr vec2 vec2::operator-(const float a) const noexcept {
        return {x - a, y - a};
    }

    constexpr vec2 operator-(const float a, const vec2 &v) noexcept {
        return {a - v.x, a - v.y};
    }

    constexpr vec2 vec2::operator*(const float a) const noexcept {
        return {x * a, y * a};
    }

    constexpr vec2 operator*(const float a, const vec2 &v) noexcept {
        return {a * v.x, a * v.y};
    }

    inline vec2 vec2::operator/(const float a) const noexcept {
        const float inv_a = (a != 0.f ? 1.f / a : std::copysign(std::numeric_limits<float>::infinity(), a));
        return {x * inv_a, y * inv_a};
    }

    constexpr bool vec2::operator==(const vec2 &a) const noexcept {
        return x == a.x && y == a.y;
    }

    constexpr bool vec2::operator!=(const vec2 &a) const noexcept {
        return x != a.x || y != a.y;
    }

    constexpr vec2 vec2::operator-() const noexcept {
        return {-x, -y};
    }

    //! Vector of three elements of type 'float'.
    /**
     * \ingroup vectors
     */
    struct vec3 {

        //! First element in vector
        float x;
        //! Second element in vector
        float y;
        //! Third element in vector
        float z;

        //! Normalize vector components such that the magnitude is unity.
        vec3 normalize() {
            const float m2 = x * x + y * y + z * z;
            if (m2 <= 0)
                return *this;
            const float inv = 1.0f / sqrtf(m2);
            this->x *= inv;
            this->y *= inv;
            this->z *= inv;
            return *this;
        }

        //! Compute the vector magnitude
        /**
         * \return magnitude of vector
         */
        [[nodiscard]] float magnitude() const {
            return sqrtf(x * x + y * y + z * z);
        }

        //! Default constructor
        /**
         * initializes to zero
         */
        constexpr vec3() : x(0), y(0), z(0) {
        }

        //! Initialize vec3 using a vector of floats
        explicit vec3(const std::vector<float> &v) {
            if (v.size() != 3) {
                throw(std::runtime_error("ERROR: vector for initialization of vec3 must have length 3."));
            }
            x = v[0];
            y = v[1];
            z = v[2];
        }
        //! Initialize vec3 using an array of floats
        explicit constexpr vec3(const float v[3]) : x(v[0]), y(v[1]), z(v[2]) {
        }

        //! Initialize vec3 using three floats
        constexpr vec3(float v0, float v1, float v2) : x(v0), y(v1), z(v2) {
        }

        //! Dot (scalar) product of two vec3 vectors
        constexpr float operator*(const vec3 &a) const noexcept;
        //! Sum of two vec3 vectors
        constexpr vec3 operator+(const vec3 &a) const noexcept;
        //! Increment vec3 vector
        inline vec3 &operator+=(const vec3 &a) noexcept;
        //! Decrement vec3 vector
        inline vec3 &operator-=(const vec3 &a) noexcept;
        //! Difference of two vec3 vectors
        constexpr vec3 operator-(const vec3 &a) const noexcept;
        //! Multiply each element by scalar (scalar is multiplied on right: vec3*a)
        constexpr vec3 operator*(float a) const noexcept;
        //! Divide each element by scalar (scalar is divided on right: vec3/a)
        inline vec3 operator/(float a) const noexcept;
        //! Add a scalar to each element (scalar is added on right: vec3+a)
        constexpr vec3 operator+(float a) const noexcept;
        //! Subtract a scalar from each element (scalar is subtracted on right: vec3-a)
        constexpr vec3 operator-(float a) const noexcept;
        //! check for equality of two vec3 vectors
        constexpr bool operator==(const vec3 &a) const noexcept;
        //! check for inequality of two vec3 vectors
        constexpr bool operator!=(const vec3 &a) const noexcept;
        //! Multiply vector by -1
        constexpr vec3 operator-() const noexcept;

        //! Write vec3 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::vec3 &vec) {
            return os << "helios::vec3<" << vec.x << ", " << vec.y << ", " << vec.z << ">";
        }
    };

    //! Multiply each element by scalar (scalar is multiplied on left: a*vec3)
    constexpr vec3 operator*(float a, const vec3 &v) noexcept;
    //! Add a scalar to each element (scalar is added on left: a+vec3)
    constexpr vec3 operator+(float a, const vec3 &v) noexcept;
    //! Subtract a scalar from each element (scalar is subtracted on left: a-vec3)
    constexpr vec3 operator-(float a, const vec3 &v) noexcept;

    //! Make a vec3 from three floats
    constexpr vec3 make_vec3(float x, float y, float z) {
        return {x, y, z};
    }

    //! Make a vec3 from an array of floats
    constexpr vec3 make_vec3(float X[3]) {
        return {X[0], X[1], X[2]};
    }

    //! Cross product of two vec3 vectors
    constexpr vec3 cross(const vec3 &a, const vec3 &b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    inline vec3 normalize(const vec3 &v) noexcept {
        const float m2 = v.x * v.x + v.y * v.y + v.z * v.z;
        const float inv = 1.0f / sqrtf(m2);
        return {v.x * inv, v.y * inv, v.z * inv};
    }

    constexpr float vec3::operator*(const vec3 &a) const noexcept {
        return a.x * x + a.y * y + a.z * z;
    }

    constexpr vec3 vec3::operator+(const vec3 &a) const noexcept {
        return {a.x + x, a.y + y, a.z + z};
    }

    inline vec3 &vec3::operator+=(const vec3 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        return *this;
    }

    inline vec3 &vec3::operator-=(const vec3 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        this->z -= a.z;
        return *this;
    }

    constexpr vec3 vec3::operator+(float a) const noexcept {
        return {x + a, y + a, z + a};
    }

    constexpr vec3 operator+(float a, const vec3 &v) noexcept {
        return {a + v.x, a + v.y, a + v.z};
    }

    constexpr vec3 vec3::operator-(const vec3 &a) const noexcept {
        return {x - a.x, y - a.y, z - a.z};
    }

    constexpr vec3 vec3::operator-(float a) const noexcept {
        return {x - a, y - a, z - a};
    }

    constexpr vec3 operator-(float a, const vec3 &v) noexcept {
        return {a - v.x, a - v.y, a - v.z};
    }

    constexpr vec3 vec3::operator*(float a) const noexcept {
        return {x * a, y * a, z * a};
    }

    constexpr vec3 operator*(float a, const vec3 &v) noexcept {
        return {a * v.x, a * v.y, a * v.z};
    }

    inline vec3 vec3::operator/(const float a) const noexcept {
        const float inv_a = (a != 0.f ? 1.f / a : std::copysign(std::numeric_limits<float>::infinity(), a));
        return {x * inv_a, y * inv_a, z * inv_a};
    }

    constexpr bool vec3::operator==(const vec3 &a) const noexcept {
        return x == a.x && y == a.y && z == a.z;
    }

    constexpr bool vec3::operator!=(const vec3 &a) const noexcept {
        return x != a.x || y != a.y || z != a.z;
    }

    constexpr vec3 vec3::operator-() const noexcept {
        return {-x, -y, -z};
    }

    //! Vector of four elements of type 'float'.
    /**
     * \ingroup vectors
     */
    struct vec4 {


        //! First element in vector
        float x;
        //! Second element in vector
        float y;
        //! Third element in vector
        float z;
        //! Fourth element in vector
        float w;

        //! Normalize vector components such that the magnitude is unity.
        vec4 normalize() {
            const float m2 = x * x + y * y + z * z + w * w;
            if (m2 <= 0)
                return *this;
            const float inv = 1.0f / sqrtf(m2);
            this->x *= inv;
            this->y *= inv;
            this->z *= inv;
            this->w *= inv;
            return *this;
        }

        //! Compute the vector magnitude
        /**
         * \return magnitude of vector.
         */
        [[nodiscard]] float magnitude() const {
            return sqrt(x * x + y * y + z * z + w * w);
        }

        //! Default constructor
        /**
         * initializes to zero
         */
        constexpr vec4() : x(0), y(0), z(0), w(0) {
        }

        //! Initialize vec4 using a vector of floats
        explicit vec4(const std::vector<float> &v) {
            if (v.size() != 4) {
                throw(std::runtime_error("ERROR: vector for initialization of vec4 must have length 4."));
            }
            x = v[0];
            y = v[1];
            z = v[2];
            w = v[3];
        }
        //! Initialize vec3 using an array of floats
        explicit constexpr vec4(const float v[4]) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {
        }

        //! Initialize vec4 using four floats
        constexpr vec4(float v0, float v1, float v2, float v3) : x(v0), y(v1), z(v2), w(v3) {
        }

        //! Dot (scalar) product of two vec4 vectors
        constexpr float operator*(const vec4 &a) const noexcept;
        //! Sum of two vec4 vectors
        constexpr vec4 operator+(const vec4 &a) const noexcept;
        //! Increment vec4 vector
        inline vec4 &operator+=(const vec4 &a) noexcept;
        //! Decrement vec4 vector
        inline vec4 &operator-=(const vec4 &a) noexcept;
        //! Difference of two vec4 vectors
        constexpr vec4 operator-(const vec4 &a) const noexcept;
        //! Multiply each element by scalar (scalar is multiplied on right: vec4*a)
        constexpr vec4 operator*(float a) const noexcept;
        //! Divide each element by scalar (scalar is divided on right: vec4/a)
        inline vec4 operator/(float a) const noexcept;
        //! Add a scalar to each element (scalar is added on right: vec4+a)
        constexpr vec4 operator+(float a) const noexcept;
        //! Subtract a scalar from each element (scalar is subtracted on right: vec4-a)
        constexpr vec4 operator-(float a) const noexcept;
        //! check for equality of two vec4 vectors
        constexpr bool operator==(const vec4 &a) const noexcept;
        //! check for equality of two vec4 vectors
        constexpr bool operator!=(const vec4 &a) const noexcept;
        //! Multiply each element by -1
        constexpr vec4 operator-() const noexcept;

        //! Write vec4 to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::vec4 &vec) {
            return os << "helios::vec4<" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ">";
        }
    };

    //! Multiply each element by scalar (scalar is multiplied on left: a*vec4)
    constexpr vec4 operator*(float a, const vec4 &v) noexcept;
    //! Add a scalar to each element (scalar is added on left: a+vec4)
    constexpr vec4 operator+(float a, const vec4 &v) noexcept;
    //! Subtract a scalar from each element (scalar is subtracted on left: a-vec4)
    constexpr vec4 operator-(float a, const vec4 &v) noexcept;

    //! Make a vec4 from three floats
    constexpr vec4 make_vec4(float x, float y, float z, float w) {
        return {x, y, z, w};
    }

    //! Make a vec4 from an array of floats
    constexpr vec4 make_vec4(float X[4]) {
        return {X[0], X[1], X[2], X[3]};
    }

    inline vec4 normalize(const vec4 &v) {
        const float m2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        const float inv = 1.0f / sqrtf(m2);
        return {v.x * inv, v.y * inv, v.z * inv, v.w * inv};
    }

    constexpr float vec4::operator*(const vec4 &a) const noexcept {
        return a.x * x + a.y * y + a.z * z + a.w * w;
    }

    constexpr vec4 vec4::operator+(const vec4 &a) const noexcept {
        return {a.x + x, a.y + y, a.z + z, a.w + w};
    }

    inline vec4 &vec4::operator+=(const vec4 &a) noexcept {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        this->w += a.w;
        return *this;
    }

    inline vec4 &vec4::operator-=(const vec4 &a) noexcept {
        this->x -= a.x;
        this->y -= a.y;
        this->z -= a.z;
        this->w -= a.w;
        return *this;
    }

    constexpr vec4 vec4::operator+(float a) const noexcept {
        return {x + a, y + a, z + a, w + a};
    }

    constexpr vec4 operator+(float a, const vec4 &v) noexcept {
        return {a + v.x, a + v.y, a + v.z, a + v.w};
    }

    constexpr vec4 vec4::operator-(const vec4 &a) const noexcept {
        return {x - a.x, y - a.y, z - a.z, w - a.w};
    }

    constexpr vec4 vec4::operator-(float a) const noexcept {
        return {x - a, y - a, z - a, w - a};
    }

    constexpr vec4 operator-(float a, const vec4 &v) noexcept {
        return {a - v.x, a - v.y, a - v.z, a - v.w};
    }

    constexpr vec4 vec4::operator*(float a) const noexcept {
        return {x * a, y * a, z * a, w * a};
    }

    constexpr vec4 operator*(float a, const vec4 &v) noexcept {
        return {a * v.x, a * v.y, a * v.z, a * v.w};
    }

    inline vec4 vec4::operator/(const float a) const noexcept {
        const float inv_a = (a != 0.f ? 1.f / a : std::copysign(std::numeric_limits<float>::infinity(), a));
        return {x * inv_a, y * inv_a, z * inv_a, w * inv_a};
    }

    constexpr bool vec4::operator==(const vec4 &a) const noexcept {
        return x == a.x && y == a.y && z == a.z && w == a.w;
    }

    constexpr bool vec4::operator!=(const vec4 &a) const noexcept {
        return x != a.x || y != a.y || z != a.z || w != a.w;
    }

    constexpr vec4 vec4::operator-() const noexcept {
        return {-x, -y, -z, -w};
    }

    //! R-G-B color vector
    /**
     * \ingroup vectors
     */
    struct RGBcolor {


        //! Red color component.
        /** Takes a value between 0 and 1. */
        float r;
        //! Green color component.
        /** Takes a value between 0 and 1. */
        float g;
        //! Blue color component.
        /** Takes a value between 0 and 1. */
        float b;

        //! Default constructor - initializes color to black.
        constexpr RGBcolor() : r(0), g(0), b(0) {
        }

        //! Constructor given three floats denoting R-G-B components.
        /**
         * \param[in] red Red color component
         * \param[in] green Green color component
         * \param[in] blue Blue color component
         * \note If arguments are outside of the range 0 to 1, values are clamped.
         */
        constexpr RGBcolor(float red, float green, float blue) : r(RGBcolor::clamp(red)), g(RGBcolor::clamp(green)), b(RGBcolor::clamp(blue)) {}

        //! Constructor given an array of three floats denoting R-G-B components.
        /**
         * \param[in] C Array (3-elements) of R-G-B color components
         * \note If arguments are outside of the range 0 to 1, values are clamped.
         */
        explicit constexpr RGBcolor(const float C[3]) : r(RGBcolor::clamp(C[0])), g(RGBcolor::clamp(C[1])), b(RGBcolor::clamp(C[2])) {}

        //! Constructor given a vector of three floats denoting R-G-B components.
        /**
         * \param[in] C Vector (3-elements) of R-G-B color components
         * \note If arguments are outside of the range 0 to 1, values are clamped.
         */
        explicit RGBcolor(const std::vector<float> &C) {
            if (C.size() != 3) {
                throw(std::runtime_error("ERROR: cannot initialize RGBcolor using supplied vector - size should be 3."));
            }
            r = RGBcolor::clamp(C[0]);
            g = RGBcolor::clamp(C[1]);
            b = RGBcolor::clamp(C[2]);
        }
        //! Constructor given a vec3 denoting R-G-B components.
        /**
         * \param[in] C vec3 of R-G-B color components
         * \note If arguments are outside of the range 0 to 1, values are clamped.
         */
        explicit constexpr RGBcolor(const vec3 &C) : r(RGBcolor::clamp(C.x)), g(RGBcolor::clamp(C.y)), b(RGBcolor::clamp(C.z)) {
        }

        //! Scale RGBcolor by some factor.
        /**
         * \param[in] scale_factor Positive scaling factor
         */
        void scale(float scale_factor) {
            if (scale_factor < 0) {
                throw(std::runtime_error("ERROR (RGBcolor::scale): cannot scale RGBcolor by negative factor."));
            }
            r *= scale_factor;
            g *= scale_factor;
            b *= scale_factor;
        }

        //! write RGBcolor to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::RGBcolor &c) {
            return os << "helios::RGBcolor<" << c.r << ", " << c.g << ", " << c.b << ">";
        }

        //! Add two RGB colors together color-by-color
        constexpr RGBcolor operator+(const RGBcolor &c) const noexcept;
        //! Subtract two RGB colors color-by-color
        constexpr RGBcolor operator-(const RGBcolor &c) const noexcept;
        //! check for equality of two RGB colors
        constexpr bool operator==(const RGBcolor &c) const noexcept;
        //! check for inequality of two RGB colors
        constexpr bool operator!=(const RGBcolor &c) const noexcept;

    private:
        //! Clamp float to range of 0 to 1.
        /**
         * \param[in] value Value to be clamped.
         * \return Clamped value
         */
        static constexpr float clamp(float value) {
            return (value < 0.f) ? 0.f : (value > 1.f) ? 1.f : value;
        }
    };

    //! Make an RGBcolor vector
    /**
     * \param[in] red Red component
     * \param[in] green Green component
     * \param[in] blue Blue component
     * \ingroup vectors
     */
    inline RGBcolor make_RGBcolor(float red, float green, float blue) {
        return {red, green, blue};
    }

    //! Blend two RGB colors together
    /**
     * \param[in] color0 First color to be blended.
     * \param[in] color1 Second color to be blended.
     * \param[in] weight Weighting factor (0<weight<1). A weight of 0 returns color0, weight of 1 returns color1.
     */
    RGBcolor blend(const RGBcolor &color0, const RGBcolor &color1, float weight);

    constexpr RGBcolor RGBcolor::operator+(const RGBcolor &c) const noexcept {
        return {clamp(r + c.r), clamp(g + c.g), clamp(b + c.b)};
    }

    constexpr RGBcolor RGBcolor::operator-(const RGBcolor &c) const noexcept {
        return {clamp(r - c.r), clamp(g - c.g), clamp(b - c.b)};
    }

    constexpr bool RGBcolor::operator==(const RGBcolor &c) const noexcept {
        return c.r == r && c.g == g && c.b == b;
    }

    constexpr bool RGBcolor::operator!=(const RGBcolor &c) const noexcept {
        return c.r != r || c.g != g || c.b != b;
    }

    /// @private
    namespace RGB {

        /// @private
        extern RGBcolor red;
        /// @private
        extern RGBcolor blue;
        /// @private
        extern RGBcolor green;
        /// @private
        extern RGBcolor cyan;
        /// @private
        extern RGBcolor magenta;
        /// @private
        extern RGBcolor yellow;
        /// @private
        extern RGBcolor orange;
        /// @private
        extern RGBcolor violet;
        /// @private
        extern RGBcolor black;
        /// @private
        extern RGBcolor white;
        /// @private
        extern RGBcolor lime;
        /// @private
        extern RGBcolor silver;
        /// @private
        extern RGBcolor gray;
        /// @private
        extern RGBcolor navy;
        /// @private
        extern RGBcolor brown;
        /// @private
        extern RGBcolor khaki;
        /// @private
        extern RGBcolor greenyellow;
        /// @private
        extern RGBcolor forestgreen;
        /// @private
        extern RGBcolor yellowgreen;
        /// @private
        extern RGBcolor goldenrod;

    } // namespace RGB

    //! R-G-B-A color vector
    /**
     * \ingroup vectors
     */
    struct RGBAcolor {

        //! Red color component.
        /** Takes a value between 0 and 1. */
        float r;
        //! Green color component.
        /** Takes a value between 0 and 1. */
        float g;
        //! Blue color component.
        /** Takes a value between 0 and 1. */
        float b;
        //! Alpha (transparency) component.
        /** Takes a value between 0 and 1. */
        float a;

        //! Default constructor - initializes color to black.
        constexpr RGBAcolor() : r(0), g(0), b(0), a(1) {}

        //! Constructor given three floats denoting R-G-B components.
        /**
         * \param[in] red Red color component
         * \param[in] green Green color component
         * \param [in] blue Blue color component
         * \param[in] alpha Alpha (transparency) component
         * \note If arguments are outside of the range 0 to 1, values are clamped.
         */
        constexpr RGBAcolor(float red, float green, float blue, float alpha) : r(RGBAcolor::clamp(red)), g(RGBAcolor::clamp(green)), b(RGBAcolor::clamp(blue)), a(RGBAcolor::clamp(alpha)) {}

        //! Constructor given an array of three floats denoting R-G-B-A components.
        /**
         * \note If arguments are outside of the range 0 to 1, values are clamped.
         * \param[in] C Array (4-elements) of R-G-B-A color components
         */
        explicit constexpr RGBAcolor(const float C[4]) : r(RGBAcolor::clamp(C[0])), g(RGBAcolor::clamp(C[1])), b(RGBAcolor::clamp(C[2])), a(RGBAcolor::clamp(C[3])) {}

        //! Constructor given a vector of three floats denoting R-G-B-A components.
        /**
         * \param[in] C Vector (4-elements) of R-G-B-A color components
         * \note If arguments are outside of the range 0 to 1, values are clamped.
         */
        explicit RGBAcolor(const std::vector<float> &C) {
            if (C.size() != 4) {
                throw(std::runtime_error("ERROR: cannot initialize RGBAcolor using supplied vector - size should be 4."));
            }
            r = RGBAcolor::clamp(C[0]);
            g = RGBAcolor::clamp(C[1]);
            b = RGBAcolor::clamp(C[2]);
            a = RGBAcolor::clamp(C[3]);
        }

        //! Scale RGBAcolor by some factor.
        /**
         * \param[in] scale_factor Positive scaling factor
         */
        void scale(float scale_factor) {
            if (scale_factor < 0) {
                throw(std::runtime_error("ERROR (RGBAcolor::scale): cannot scale RGBAcolor by negative factor."));
            }
            r *= scale_factor;
            g *= scale_factor;
            b *= scale_factor;
            a *= scale_factor;
        }

        //! write RGBAcolor to output stream
        friend std::ostream &operator<<(std::ostream &os, const helios::RGBAcolor &c) {
            return os << "helios::RGBAcolor<" << c.r << ", " << c.g << ", " << c.b << ", " << c.a << ">";
        }

        //! Add two RGBA colors together color-by-color
        constexpr RGBAcolor operator+(const RGBAcolor &c) const noexcept;
        //! Subtract two RGBA colors color-by-color
        constexpr RGBAcolor operator-(const RGBAcolor &c) const noexcept;
        //! check for equality of two RGBA colors
        constexpr bool operator==(const helios::RGBAcolor &c) const noexcept;
        //! check for inequality of two RGBA colors
        constexpr bool operator!=(const helios::RGBAcolor &c) const noexcept;

    private:
        //! Clamp float to range of 0 to 1.
        /**
         * \param[in] value Value to be clamped.
         * \return Clamped value
         */
        static constexpr float clamp(float value) {
            return (value < 0.f) ? 0.f : (value > 1.f) ? 1.f : value;
        }
    };

    //! Make an RGBAcolor vector
    /**
     * \param[in] r Red component
     * \param[in] g Green component
     * \param[in] b Blue component
     * \param[in] a Alpha (transparency) component
     * \ingroup vectors
     */
    inline RGBAcolor make_RGBAcolor(float r, float g, float b, float a) {
        return {r, g, b, a};
    }

    //! Make an RGBAcolor vector (add alpha channel to RGBcolor)
    /**
     * \param[in] color RGBcolor
     * \param[in] a Alpha (transparency) component
     * \ingroup vectors
     */
    inline RGBAcolor make_RGBAcolor(RGBcolor color, float a) {
        return {color.r, color.g, color.b, a};
    }

    //! Blend two RGBA colors together
    /**
     * \param[in] color0 First color to be blended.
     * \param[in] color1 Second color to be blended.
     * \param[in] weight Weighting factor (0<weight<1). A weight of 0 returns color0, weight of 1 returns color1.
     */
    RGBAcolor blend(const helios::RGBAcolor &color0, const helios::RGBAcolor &color1, float weight);

    constexpr bool RGBAcolor::operator==(const RGBAcolor &c) const noexcept {
        return c.r == r && c.g == g && c.b == b && c.a == a;
    }

    constexpr bool RGBAcolor::operator!=(const RGBAcolor &c) const noexcept {
        return c.r != r || c.g != g || c.b != b || c.a != a;
    }

    constexpr RGBAcolor RGBAcolor::operator+(const RGBAcolor &c) const noexcept {
        return {clamp(r + c.r), clamp(g + c.g), clamp(b + c.b), clamp(a + c.a)};
    }

    constexpr RGBAcolor RGBAcolor::operator-(const RGBAcolor &c) const noexcept {
        return {clamp(r - c.r), clamp(g - c.g), clamp(b - c.b), clamp(a - c.a)};
    }

    /// @private
    namespace RGBA {

        /// @private
        extern RGBAcolor red;
        /// @private
        extern RGBAcolor blue;
        /// @private
        extern RGBAcolor green;
        /// @private
        extern RGBAcolor cyan;
        /// @private
        extern RGBAcolor magenta;
        /// @private
        extern RGBAcolor yellow;
        /// @private
        extern RGBAcolor orange;
        /// @private
        extern RGBAcolor violet;
        /// @private
        extern RGBAcolor black;
        /// @private
        extern RGBAcolor white;
        /// @private
        extern RGBAcolor lime;
        /// @private
        extern RGBAcolor silver;
        /// @private
        extern RGBAcolor gray;
        /// @private
        extern RGBAcolor navy;
        /// @private
        extern RGBAcolor brown;
        /// @private
        extern RGBAcolor khaki;
        /// @private
        extern RGBAcolor greenyellow;
        /// @private
        extern RGBAcolor forestgreen;
        /// @private
        extern RGBAcolor yellowgreen;
        /// @private
        extern RGBAcolor goldenrod;

    } // namespace RGBA

    //! Date vector
    /**
     * \sa make_Date()
     * \ingroup vectors
     */
    struct Date {


        //! Day of month
        int day;
        //! Month of year
        int month;
        //! Year in YYYY format
        int year;

        //! Default constructor
        constexpr Date() : day(1), month(1), year(2000) {
        }

        //! Day/Month/Year constructor
        /**
         * \param[in] day Day of month (1-31)
         * \param[in] month Day of year (1-12)
         * \param[in] year Year in YYYY format
         */
        Date(int day, int month, int year) : day(day), month(month), year(year) {

            if (day < 1 || day > 31) {
                throw(std::runtime_error("ERROR (Date constructor): Day of month is out of range (day of " + std::to_string(day) + " was given)."));
            } else if (month < 1 || month > 12) {
                throw(std::runtime_error("ERROR (Date constructor): Month of year is out of range (month of " + std::to_string(month) + " was given)."));
            } else if (year < 1000) {
                throw(std::runtime_error("ERROR (Date constructor): Year should be specified in YYYY format."));
            }
        }

        //! Convert to Julian day
        /**
         * \return Julian day of the year
         */
        [[nodiscard]] int JulianDay() const;

        //! Increment Date vector by one day
        void incrementDay();

        //! Check whether it is a leap year
        /**
         * \return True if it is a leap year, false if not
         */
        [[nodiscard]] bool isLeapYear() const;

        //! check for equality of two dates
        constexpr bool operator==(const helios::Date &c) const noexcept;
        //! check for inequality of two Dates
        constexpr bool operator!=(const helios::Date &c) const noexcept;

        //! Write Date to output stream
        friend std::ostream &operator<<(std::ostream &os, helios::Date const &d) {
            return os << d.year << "-" << std::setfill('0') << std::setw(2) << d.month << "-" << std::setfill('0') << std::setw(2) << d.day;
        }
    };

    //! Make a Date vector
    /**
     * \param[in] day Day of month (1-31)
     * \param[in] month Day of year (1-12)
     * \param[in] year Year in YYYY format
     * \ingroup vectors
     */
    inline Date make_Date(int day, int month, int year) {
        if (day < 1 || day > 31) {
            throw(std::runtime_error("ERROR (make_Date): Day of month is out of range (day of " + std::to_string(day) + " was given)."));
        } else if (month < 1 || month > 12) {
            throw(std::runtime_error("ERROR (make_Date): Month of year is out of range (month of " + std::to_string(month) + " was given)."));
        } else if (year < 1000) {
            throw(std::runtime_error("ERROR (make_Date): Year should be specified in YYYY format."));
        }

        return {day, month, year};
    }

    constexpr bool Date::operator==(const Date &c) const noexcept {
        return c.day == day && c.month == month && c.year == year;
    }

    constexpr bool Date::operator!=(const Date &c) const noexcept {
        return c.day != day || c.month != month || c.year != year;
    }

    //! Convert a Julian day to a calendar \ref Date vector.
    /**
     * \param[in] JulianDay Julian day of year (0-366)
     * \param[in] year Year in YYYY format
     * \ingroup vectors
     */
    inline Date Julian2Calendar(int JulianDay, int year) {

        int month = -1;

        int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
        int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
        const int *skips = nullptr;

        if (JulianDay < 1 || JulianDay > 366) {
            throw(std::runtime_error("ERROR (Julian2Calendar): Julian day of " + std::to_string(JulianDay) + " is out of range (should be >0 and <367)."));
        }

        if ((year - 2000) % 4 == 0) { // leap year
            skips = skips_leap;
        } else { // non-leap year
            skips = skips_nonleap;
        }

        for (int i = 1; i <= 12; i++) {
            if (i == 12) {
                month = 12;
            } else if (JulianDay > skips[i - 1] && JulianDay <= skips[i]) {
                month = i;
                break;
            }
        }
        assert(month > 0 && month < 13);

        int day = JulianDay - skips[month - 1];

        assert(day > 0 && day < 32);

        return {day, month, year};
    }

    //! Convert a calendar Date vector to Julian day.
    /**
     * \param[in] date Date vector
     * \ingroup vectors
     */
    inline int Calendar2Julian(Date date) {

        int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 214, 244, 274, 305, 335};
        int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
        int *skips = nullptr;

        if ((date.year - 2000) % 4 == 0) { // leap year
            skips = skips_leap;
        } else { // non-leap year
            skips = skips_nonleap;
        }

        int JD = skips[date.month - 1] + date.day;
        assert(JD > 0 && JD < 366);

        return JD;
    }

    //! Make a Date vector
    /**
     * \param[in] JulianDay Julian day of year (0-366)
     * \param[in] year Year in YYYY format
     * \ingroup vectors
     */
    inline Date make_Date(int JulianDay, int year) {
        if (JulianDay < 1 || JulianDay > 366) {
            throw(std::runtime_error("ERROR (make_Date): Julian day is out of range (Julian day of " + std::to_string(JulianDay) + " was given)."));
        } else if (year < 1000) {
            throw(std::runtime_error("ERROR (make_Date): Year should be specified in YYYY format."));
        }

        return Julian2Calendar(JulianDay, year);
    }


    //! Time vector
    /**
     * \sa make_Time()
     * \ingroup vectors
     */
    struct Time {

        //! Second of minute
        int second;
        //! Minute of hour
        int minute;
        //! Hour of day
        int hour;

        //! Default constructor
        constexpr Time() : second(0), minute(0), hour(0) {
        }

        //! second/minute/hour constructor
        /**
         * \param[in] hour Hour of day (0-23)
         * \param[in] minute Minute of hour (0-59)
         */
        Time(int hour, int minute) : second(0), minute(minute), hour(hour) {

            if (minute < 0 || minute > 59) {
                throw(std::runtime_error("ERROR (Time constructor): Minute out of range (0-59)."));
            } else if (hour < 0 || hour > 23) {
                throw(std::runtime_error("ERROR (Time constructor): Hour out of range (0-23)."));
            }
        }

        //! minute/hour constructor
        /**
         * \param[in] hour Hour of day (0-23)
         * \param[in] minute Minute of hour (0-59)
         * \param[in] second Second of minute (0-59)
         */
        Time(int hour, int minute, int second) : second(second), minute(minute), hour(hour) {

            if (second < 0 || second > 59) {
                throw(std::runtime_error("ERROR (Time constructor): Second out of range (0-59)."));
            } else if (minute < 0 || minute > 59) {
                throw(std::runtime_error("ERROR (Time constructor): Minute out of range (0-59)."));
            } else if (hour < 0 || hour > 23) {
                throw(std::runtime_error("ERROR (Time constructor): Hour out of range (0-23)."));
            }
        }

        //! check for equality of two times
        constexpr bool operator==(const helios::Time &c) const noexcept;
        //! check for inequality of two times
        constexpr bool operator!=(const helios::Time &c) const noexcept;

        //! Write Time to output stream
        friend std::ostream &operator<<(std::ostream &os, helios::Time const &t) {
            return os << t.hour << ":" << std::setfill('0') << std::setw(2) << t.minute << ":" << std::setfill('0') << std::setw(2) << t.second;
        }
    };

    //! Make a Time vector
    /**
     * \param[in] hour Hour of day (0-23)
     * \param[in] minute Minute of hour (0-59)
     * \ingroup vectors
     */
    inline Time make_Time(int hour, int minute) {
        if (minute < 0 || minute > 59) {
            throw(std::runtime_error("ERROR (make_Time): Minute of " + std::to_string(minute) + " out of range (0-59)."));
        } else if (hour < 0 || hour > 23) {
            throw(std::runtime_error("ERROR (make_Time): Hour of " + std::to_string(hour) + " out of range (0-23)."));
        }

        return {hour, minute, 0};
    }

    //! Make a Time vector
    /**
     * \param[in] hour Hour of day (0-23)
     * \param[in] minute Minute of hour (0-59)
     * \param[in] second Second of minute (0-59)
     * \ingroup vectors
     */
    inline Time make_Time(int hour, int minute, int second) {
        if (second < 0 || second > 59) {
            throw(std::runtime_error("ERROR (make_Time): Second of " + std::to_string(second) + " out of range (0-59)."));
        } else if (minute < 0 || minute > 59) {
            throw(std::runtime_error("ERROR (make_Time): Minute of " + std::to_string(minute) + " out of range (0-59)."));
        } else if (hour < 0 || hour > 23) {
            throw(std::runtime_error("ERROR (make_Time): Hour of " + std::to_string(hour) + " out of range (0-23)."));
        }

        return {hour, minute, second};
    }

    constexpr bool Time::operator==(const Time &c) const noexcept {
        return c.hour == hour && c.minute == minute && c.second == second;
    }

    constexpr bool Time::operator!=(const Time &c) const noexcept {
        return c.hour != hour || c.minute != minute || c.second != second;
    }

    //! Location vector
    /**
     * \ingroup vectors
     */
    struct Location {

        //! Latitude in degrees (+northern hemisphere, -southern hemisphere)
        float latitude_deg;
        //! Longitude in degrees (+western hemisphere, -eastern hemisphere)
        float longitude_deg;
        //! Offset from UTC in hours (+moving West)
        float UTC_offset;

        //! Default constructor
        Location() {
            latitude_deg = 38.55f;
            longitude_deg = 121.76f;
            UTC_offset = 8;
        }

        //! latitude/longitude/UTC constructor
        /**
         * \param[in] latitude_deg Latitude in degrees (+northern hemisphere, -southern hemisphere)
         * \param[in] longitude_deg Longitude in degrees (+western hemisphere, -eastern hemisphere)
         * \param[in] UTC_offset Offset from UTC in hours (+moving West)
         */
        Location(float latitude_deg, float longitude_deg, float UTC_offset) {

            this->latitude_deg = latitude_deg;
            this->longitude_deg = longitude_deg;
            this->UTC_offset = UTC_offset;
        }

        //! check for equality of two locations
        bool operator==(const helios::Location &c) const noexcept;
        //! check for inequality of two locations
        bool operator!=(const helios::Location &c) const noexcept;

        //! Write Location to output stream
        friend std::ostream &operator<<(std::ostream &os, helios::Location const &t) {
            return os << "<" << t.latitude_deg << "," << t.longitude_deg << "," << t.UTC_offset << ">";
        }
    };

    //! Make a Location vector
    /**
     * \param[in] latitude_deg Latitude in degrees (+northern hemisphere, -southern hemisphere)
     * \param[in] longitude_deg Longitude in degrees (+western hemisphere, -eastern hemisphere)
     * \param[in] UTC_offset Offset from UTC in hours (+moving West)
     * \ingroup vectors
     */
    inline Location make_Location(float latitude_deg, float longitude_deg, float UTC_offset) {
        return {latitude_deg, longitude_deg, UTC_offset};
    }

    inline bool Location::operator==(const Location &c) const noexcept {
        return c.latitude_deg == latitude_deg && c.longitude_deg == longitude_deg && c.UTC_offset == UTC_offset;
    }

    inline bool Location::operator!=(const Location &c) const noexcept {
        return c.latitude_deg != latitude_deg || c.longitude_deg != longitude_deg || c.UTC_offset != UTC_offset;
    }

    //! Vector of spherical coordinates (elevation,azimuth)
    /**
     * \ingroup vectors
     */
    struct SphericalCoord {
    private:
        float elevation_private;
        float zenith_private;

    public:
        //! Radius
        float radius;
        //! Elevation angle (radians)
        /*
         * \note Elevation angle is read-only, since it is linked to the zenith angle.
         */
        const float &elevation;
        //! Zenithal angle (radians)
        /*
         * \note Zenith angle is read-only, since it is linked to the elevation angle.
         */
        const float &zenith;
        //! Azimuthal angle (radians)
        float azimuth;

        //! Default constructor
        /** initializes to vertical unit vector */
        SphericalCoord() : elevation_private(0.5f * static_cast<float>(M_PI)), zenith_private(0.f), elevation(elevation_private), zenith(zenith_private) {
            radius = 1;
            azimuth = 0;
        }
        //! Initialize SphericalCoord by specifying radius, elevation, and azimuth
        /**
         * \param[in] radius Radius
         * \param[in] elevation_radians Elevation angle (radians)
         * \param[in] azimuth_radians Azimuthal angle (radians)
         */
        SphericalCoord(float radius, float elevation_radians, float azimuth_radians) :
            elevation_private(elevation_radians), zenith_private(0.5f * static_cast<float>(M_PI) - elevation_radians), radius(radius), elevation(elevation_private), zenith(zenith_private), azimuth(azimuth_radians) {
        }

        //! Copy constructor
        SphericalCoord(const SphericalCoord &c) : elevation_private(c.elevation_private), zenith_private(c.zenith_private), radius(c.radius), elevation(c.elevation_private), zenith(c.zenith_private), azimuth(c.azimuth) {
        }

        //! Assignment operator
        SphericalCoord &operator=(const SphericalCoord &c) noexcept {
            if (this != &c) {
                elevation_private = c.elevation_private;
                zenith_private = c.zenith_private;
                radius = c.radius;
                azimuth = c.azimuth;
            }
            return *this;
        }

        //! check for equality of two spherical coordinates
        bool operator==(const helios::SphericalCoord &c) const noexcept;
        //! check for inequality of two spherical coordinates
        bool operator!=(const helios::SphericalCoord &c) const noexcept;

        //! Write SphericalCoord to output stream
        friend std::ostream &operator<<(std::ostream &os, helios::SphericalCoord const &coord) {
            return os << "helios::SphericalCoord<" << coord.radius << ", " << coord.elevation << ", " << coord.azimuth << ">";
        }
    };

    //! Make a SphericalCoord by specifying elevation and azimuth
    /**
     * \param[in] elevation_radians Elevation angle (radians)
     * \param[in] azimuth_radians Azimuthal angle (radians)
     */
    inline SphericalCoord make_SphericalCoord(float elevation_radians, float azimuth_radians) {
        return {1, elevation_radians, azimuth_radians};
    }

    //! Make a SphericalCoord by specifying radius, elevation, and azimuth
    /**
     * \param[in] radius Radius
     * \param[in] elevation_radians Elevation angle (radians)
     * \param[in] azimuth_radians Azimuthal angle (radians)
     */
    inline SphericalCoord make_SphericalCoord(float radius, float elevation_radians, float azimuth_radians) {
        return {radius, elevation_radians, azimuth_radians};
    }

    inline bool SphericalCoord::operator==(const SphericalCoord &c) const noexcept {
        return c.radius == radius && c.zenith == zenith && c.elevation == elevation && c.azimuth == azimuth;
    }

    inline bool SphericalCoord::operator!=(const SphericalCoord &c) const noexcept {
        return c.radius != radius || c.zenith != zenith || c.elevation != elevation || c.azimuth != azimuth;
    }


} // namespace helios

#endif
