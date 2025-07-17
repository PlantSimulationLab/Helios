/** \file "selfTest.cpp" Context selfTest() function.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <set>
#include "Context.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

using namespace helios;

double errtol = 1e-6;

int Context::selfTest() {

    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) { // important - query flags (and --exit) rely on this
        return res;             // propagate the result of the tests
    }

    return res;
}

// =================================================================================
// Suite 1: Core Utilities and Data Types
//
// Tests for fundamental data structures (vectors, colors, date/time),
// coordinate systems, and their associated operations.
// =================================================================================
TEST_SUITE("Core Utilities and Data Types") {

    TEST_CASE("Vector and Color Types (helios_vector_types.h)") {
        // This test case contains the full, unaltered "helios_vector_types coverage" test.
        SUBCASE("int2") {
            int2 v0(1, 2);
            int2 v1(3, 4);
            DOCTEST_CHECK(v0.x == 1);
            DOCTEST_CHECK(v0.y == 2);

            // Constructors
            int2 v_vec_c(std::vector<int>{1, 2});
            DOCTEST_CHECK(v_vec_c == v0);
            DOCTEST_CHECK_THROWS(int2({1}));
            int arr[2] = {1, 2};
            int2 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Operators
            DOCTEST_CHECK(v0 + v1 == int2(4, 6));
            v0 += v1;
            DOCTEST_CHECK(v0 == int2(4, 6));
            DOCTEST_CHECK(v1 - v0 == int2(-1, -2));
            v0 -= v1;
            DOCTEST_CHECK(v0 == int2(1, 2));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == int2(-1, -2));

            // make_int2
            int arr2[2] = {5, 6};
            DOCTEST_CHECK(make_int2(arr2) == int2(5, 6));

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::int2<1, 2>");
        }

        SUBCASE("int3") {
            int3 v0(1, 2, 3);
            int3 v1(4, 5, 6);
            DOCTEST_CHECK(v0.x == 1);

            // Constructors
            int3 v_vec_c(std::vector<int>{1, 2, 3});
            DOCTEST_CHECK(v_vec_c == v0);
            DOCTEST_CHECK_THROWS(int3({1}));
            int arr[3] = {1, 2, 3};
            int3 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Operators
            DOCTEST_CHECK(v0 + v1 == int3(5, 7, 9));
            v0 += v1;
            DOCTEST_CHECK(v0 == int3(5, 7, 9));
            DOCTEST_CHECK(v1 - v0 == int3(-1, -2, -3));
            v0 -= v1;
            DOCTEST_CHECK(v0 == int3(1, 2, 3));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == int3(-1, -2, -3));

            // make_int3
            DOCTEST_CHECK(make_int3(1, 2, 3) == v0);
            int arr2[3] = {1, 2, 3};
            DOCTEST_CHECK(make_int3(arr2) == v0);

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::int3<1, 2, 3>");
        }

        SUBCASE("int4") {
            int4 v0(1, 2, 3, 4);
            int4 v1(5, 6, 7, 8);
            DOCTEST_CHECK(v0.w == 4);

            // Constructors
            int4 v_def_c;
            DOCTEST_CHECK(v_def_c.x == 0);
            int4 v_vec_c(std::vector<int>{1, 2, 3, 4});
            DOCTEST_CHECK(v_vec_c == v0);
            DOCTEST_CHECK_THROWS(int4({1}));
            int arr[4] = {1, 2, 3, 4};
            int4 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Operators
            DOCTEST_CHECK(v0 + v1 == int4(6, 8, 10, 12));
            v0 += v1;
            DOCTEST_CHECK(v0 == int4(6, 8, 10, 12));
            DOCTEST_CHECK(v1 - v0 == int4(-1, -2, -3, -4));
            v0 -= v1;
            DOCTEST_CHECK(v0 == int4(1, 2, 3, 4));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == int4(-1, -2, -3, -4));

            // make_int4
            DOCTEST_CHECK(make_int4(1, 2, 3, 4) == v0);
            int arr2[4] = {1, 2, 3, 4};
            DOCTEST_CHECK(make_int4(arr2) == v0);

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::int4<1, 2, 3, 4>");
        }

        SUBCASE("uint2") {
            uint2 v0(1, 2);
            uint2 v1(3, 4);
            DOCTEST_CHECK(v0.x == 1);

            // Constructors
            uint2 v_def_c;
            DOCTEST_CHECK(v_def_c.x == 0);
            uint2 v_vec_c(std::vector<uint>{1, 2});
            DOCTEST_CHECK(v_vec_c == v0);
            DOCTEST_CHECK_THROWS(uint2({1}));
            unsigned int arr[2] = {1, 2};
            uint2 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Operators
            DOCTEST_CHECK(v0 + v1 == uint2(4, 6));
            v0 += v1;
            DOCTEST_CHECK(v0 == uint2(4, 6));
            DOCTEST_CHECK(v0 - v1 == uint2(1, 2));
            v0 -= v1;
            DOCTEST_CHECK(v0 == uint2(1, 2));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == uint2((unsigned int) -1, (unsigned int) -2));

            // make_uint2
            DOCTEST_CHECK(make_uint2(1, 2) == v0);
            unsigned int arr2[2] = {1, 2};
            DOCTEST_CHECK(make_uint2(arr2) == v0);

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::uint2<1, 2>");
        }

        SUBCASE("uint3") {
            uint3 v0(1, 2, 3);
            uint3 v1(4, 5, 6);
            DOCTEST_CHECK(v0.x == 1);

            // Constructors
            uint3 v_def_c;
            DOCTEST_CHECK(v_def_c.x == 0);
            uint3 v_vec_c(std::vector<uint>{1, 2, 3});
            DOCTEST_CHECK(v_vec_c == v0);
            DOCTEST_CHECK_THROWS(uint3({1}));
            unsigned int arr[3] = {1, 2, 3};
            uint3 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Operators
            DOCTEST_CHECK(v0 + v1 == uint3(5, 7, 9));
            v0 += v1;
            DOCTEST_CHECK(v0 == uint3(5, 7, 9));
            DOCTEST_CHECK(v0 - v1 == uint3(1, 2, 3));
            v0 -= v1;
            DOCTEST_CHECK(v0 == uint3(1, 2, 3));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == uint3((unsigned int) -1, (unsigned int) -2, (unsigned int) -3));

            // make_uint3
            DOCTEST_CHECK(make_uint3(1, 2, 3) == v0);
            unsigned int arr2[3] = {1, 2, 3};
            DOCTEST_CHECK(make_uint3(arr2) == v0);

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::uint3<1, 2, 3>");
        }

        SUBCASE("uint4") {
            uint4 v0(1, 2, 3, 4);
            uint4 v1(5, 6, 7, 8);
            DOCTEST_CHECK(v0.w == 4);

            // Constructors
            uint4 v_def_c;
            DOCTEST_CHECK(v_def_c.x == 0);
            uint4 v_vec_c(std::vector<uint>{1, 2, 3, 4});
            DOCTEST_CHECK(v_vec_c == v0);
            DOCTEST_CHECK_THROWS(uint4({1}));
            unsigned int arr[4] = {1, 2, 3, 4};
            uint4 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Operators
            DOCTEST_CHECK(v0 + v1 == uint4(6, 8, 10, 12));
            v0 += v1;
            DOCTEST_CHECK(v0 == uint4(6, 8, 10, 12));
            DOCTEST_CHECK(v0 - v1 == uint4(1, 2, 3, 4));
            v0 -= v1;
            DOCTEST_CHECK(v0 == uint4(1, 2, 3, 4));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == uint4((unsigned int) -1, (unsigned int) -2, (unsigned int) -3, (unsigned int) -4));

            // make_uint4
            DOCTEST_CHECK(make_uint4(1, 2, 3, 4) == v0);
            unsigned int arr2[4] = {1, 2, 3, 4};
            DOCTEST_CHECK(make_uint4(arr2) == v0);

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::uint4<1, 2, 3, 4>");
        }

        SUBCASE("vec2") {
            vec2 v0(1.f, 2.f);
            vec2 v1(3.f, 4.f);

            // Constructors
            DOCTEST_CHECK_THROWS(vec2({1.f}));
            float arr[] = {1.f, 2.f};
            vec2 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Methods
            DOCTEST_CHECK(vec2(3.f, 4.f).magnitude() == doctest::Approx(5.f));
            DOCTEST_CHECK(normalize(vec2(3.f, 4.f)) == vec2(0.6f, 0.8f));
            vec2 v_norm = vec2(3.f, 4.f);
            v_norm.normalize();
            DOCTEST_CHECK(v_norm == vec2(0.6f, 0.8f));
            vec2 zero_vec(0.f, 0.f);
            zero_vec.normalize();
            DOCTEST_CHECK(zero_vec == vec2(0.f, 0.f));


            // Operators
            DOCTEST_CHECK(v0 * v1 == doctest::Approx(11.f));
            v0 += v1;
            DOCTEST_CHECK(v0 == vec2(4.f, 6.f));
            v0 -= v1;
            DOCTEST_CHECK(v0 == vec2(1.f, 2.f));
            v0 *= 2.f;
            DOCTEST_CHECK(v0 == vec2(2.f, 4.f));
            v0 /= 2.f;
            DOCTEST_CHECK(v0 == vec2(1.f, 2.f));
            DOCTEST_CHECK(v0 + 1.f == vec2(2.f, 3.f));
            DOCTEST_CHECK(1.f + v0 == vec2(2.f, 3.f));
            DOCTEST_CHECK(v0 - 1.f == vec2(0.f, 1.f));
            DOCTEST_CHECK(1.f - v0 == vec2(0.f, -1.f));
            DOCTEST_CHECK(v0 * 2.f == vec2(2.f, 4.f));
            DOCTEST_CHECK(v0 / 2.f == vec2(0.5f, 1.f));
            DOCTEST_CHECK(v0 / 0.f == vec2(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == vec2(-1.f, -2.f));

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::vec2<1, 2>");
        }

        SUBCASE("vec3") {
            vec3 v0(1.f, 2.f, 3.f);
            vec3 v1(4.f, 5.f, 6.f);

            // Constructors
            DOCTEST_CHECK_THROWS(vec3({1.f}));
            float arr[] = {1.f, 2.f, 3.f};
            vec3 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Methods
            vec3 zero_vec(0.f, 0.f, 0.f);
            zero_vec.normalize();
            DOCTEST_CHECK(zero_vec == vec3(0.f, 0.f, 0.f));

            // Operators
            v0 += v1;
            DOCTEST_CHECK(v0 == vec3(5.f, 7.f, 9.f));
            v0 -= v1;
            DOCTEST_CHECK(v0 == vec3(1.f, 2.f, 3.f));
            v0 *= 2.f;
            DOCTEST_CHECK(v0 == vec3(2.f, 4.f, 6.f));
            v0 /= 2.f;
            DOCTEST_CHECK(v0 == vec3(1.f, 2.f, 3.f));
            DOCTEST_CHECK(v0 + 1.f == vec3(2.f, 3.f, 4.f));
            DOCTEST_CHECK(1.f + v0 == vec3(2.f, 3.f, 4.f));
            DOCTEST_CHECK(v0 - 1.f == vec3(0.f, 1.f, 2.f));
            DOCTEST_CHECK(1.f - v0 == vec3(0.f, -1.f, -2.f));
            DOCTEST_CHECK(v0 * 2.f == vec3(2.f, 4.f, 6.f));
            DOCTEST_CHECK(v0 != v1);

            // make_vec3
            float arr2[] = {1.f, 2.f, 3.f};
            DOCTEST_CHECK(make_vec3(arr2) == v0);

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::vec3<1, 2, 3>");
        }

        SUBCASE("vec4") {
            vec4 v0(1.f, 2.f, 3.f, 4.f);
            vec4 v1(5.f, 6.f, 7.f, 8.f);

            // Constructors
            vec4 v_def_c;
            DOCTEST_CHECK(v_def_c.x == 0.f);
            DOCTEST_CHECK_THROWS(vec4({1.f}));
            float arr[] = {1.f, 2.f, 3.f, 4.f};
            vec4 v_arr_c(arr);
            DOCTEST_CHECK(v_arr_c == v0);

            // Methods
            DOCTEST_CHECK(vec4(1, 2, 3, 4).magnitude() == doctest::Approx(sqrt(30.f)));
            vec4 v_norm(1, 2, 3, 4);
            v_norm.normalize();
            float mag = sqrt(30.f);
            DOCTEST_CHECK(v_norm.x == doctest::Approx(1.f / mag));
            vec4 zero_vec;
            zero_vec.normalize();
            DOCTEST_CHECK(zero_vec == vec4(0, 0, 0, 0));
            DOCTEST_CHECK(normalize(vec4(1, 2, 3, 4)) == v_norm);

            // Operators
            DOCTEST_CHECK(v0 * v1 == doctest::Approx(70.f));
            DOCTEST_CHECK(v0 + v1 == vec4(6.f, 8.f, 10.f, 12.f));
            v0 += v1;
            DOCTEST_CHECK(v0 == vec4(6.f, 8.f, 10.f, 12.f));
            DOCTEST_CHECK(v0 - v1 == vec4(1.f, 2.f, 3.f, 4.f));
            v0 -= v1;
            DOCTEST_CHECK(v0 == vec4(1.f, 2.f, 3.f, 4.f));
            v0 *= 2.f;
            DOCTEST_CHECK(v0 == vec4(2.f, 4.f, 6.f, 8.f));
            v0 /= 2.f;
            DOCTEST_CHECK(v0 == vec4(1.f, 2.f, 3.f, 4.f));
            DOCTEST_CHECK(v0 + 1.f == vec4(2.f, 3.f, 4.f, 5.f));
            DOCTEST_CHECK(1.f + v0 == vec4(2.f, 3.f, 4.f, 5.f));
            DOCTEST_CHECK(v0 - 1.f == vec4(0.f, 1.f, 2.f, 3.f));
            DOCTEST_CHECK(1.f - v0 == vec4(0.f, -1.f, -2.f, -3.f));
            DOCTEST_CHECK(v0 * 2.f == vec4(2.f, 4.f, 6.f, 8.f));
            DOCTEST_CHECK(2.f * v0 == vec4(2.f, 4.f, 6.f, 8.f));
            DOCTEST_CHECK(v0 / 2.f == vec4(0.5f, 1.f, 1.5f, 2.f));
            DOCTEST_CHECK(v0 / 0.f == vec4(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()));
            DOCTEST_CHECK(v0 != v1);
            DOCTEST_CHECK(-v0 == vec4(-1.f, -2.f, -3.f, -4.f));

            // make_vec4
            DOCTEST_CHECK(make_vec4(1.f, 2.f, 3.f, 4.f) == v0);
            float arr2[] = {1.f, 2.f, 3.f, 4.f};
            DOCTEST_CHECK(make_vec4(arr2) == v0);

            // Stream operator
            std::stringstream ss;
            ss << v0;
            DOCTEST_CHECK(ss.str() == "helios::vec4<1, 2, 3, 4>");
        }

        SUBCASE("RGBcolor") {
            RGBcolor c0(0.1f, 0.2f, 0.3f);
            RGBcolor c1(0.4f, 0.5f, 0.6f);

            // Constructors
            float arr[] = {0.1f, 0.2f, 0.3f};
            RGBcolor c_arr_c(arr);
            DOCTEST_CHECK(c_arr_c == c0);
            std::vector<float> vec = {0.1f, 0.2f, 0.3f};
            RGBcolor c_vec_c(vec);
            DOCTEST_CHECK(c_vec_c == c0);
            DOCTEST_CHECK_THROWS(RGBcolor({0.1f}));
            vec3 v(0.1f, 0.2f, 0.3f);
            RGBcolor c_v3_c(v);
            DOCTEST_CHECK(c_v3_c == c0);

            // Methods
            c0.scale(2.f);
            DOCTEST_CHECK(c0.r == doctest::Approx(0.2f));
            DOCTEST_CHECK_THROWS(c0.scale(-1.f));

            // Operators
            c0 = RGBcolor(0.1f, 0.2f, 0.3f);
            DOCTEST_CHECK(c0 + c1 == RGBcolor(0.5f, 0.7f, 0.9f));
            DOCTEST_CHECK(c1 - c0 == RGBcolor(0.3f, 0.3f, 0.3f));
            DOCTEST_CHECK(c0 != c1);

            // Stream operator
            std::stringstream ss;
            ss << c0;
            DOCTEST_CHECK(ss.str() == "helios::RGBcolor<0.1, 0.2, 0.3>");
        }

        SUBCASE("RGBAcolor") {
            RGBAcolor c0(0.1f, 0.2f, 0.3f, 0.4f);
            RGBAcolor c1(0.5f, 0.6f, 0.7f, 0.8f);

            // Constructors
            float arr[] = {0.1f, 0.2f, 0.3f, 0.4f};
            RGBAcolor c_arr_c(arr);
            DOCTEST_CHECK(c_arr_c == c0);
            std::vector<float> vec = {0.1f, 0.2f, 0.3f, 0.4f};
            RGBAcolor c_vec_c(vec);
            DOCTEST_CHECK(c_vec_c == c0);
            DOCTEST_CHECK_THROWS(RGBAcolor({0.1f}));

            // Methods
            c0.scale(2.f);
            DOCTEST_CHECK(c0.r == doctest::Approx(0.2f));
            DOCTEST_CHECK_THROWS(c0.scale(-1.f));

            // Operators
            c0 = RGBAcolor(0.1f, 0.2f, 0.3f, 0.4f);
            DOCTEST_CHECK(c0 + c1 == RGBAcolor(0.6f, 0.8f, 1.0f, 1.0f));
            DOCTEST_CHECK(c1 - c0 == RGBAcolor(0.4f, 0.4f, 0.4f, 0.4f));
            DOCTEST_CHECK(c0 != c1);

            // Stream operator
            std::stringstream ss;
            ss << c0;
            DOCTEST_CHECK(ss.str() == "helios::RGBAcolor<0.1, 0.2, 0.3, 0.4>");
        }
    }

    TEST_CASE("Date and Time Logic") {
        SUBCASE("Date struct and helpers") {
            Date d(10, 1, 2000);
            // Constructors
            DOCTEST_CHECK_THROWS(Date(32, 1, 2000));
            DOCTEST_CHECK_THROWS(Date(1, 13, 2000));
            DOCTEST_CHECK_THROWS(Date(1, 1, 999));

            // Operators
            DOCTEST_CHECK(d == Date(10, 1, 2000));
            DOCTEST_CHECK(d != Date(11, 1, 2000));

            // make_Date
            DOCTEST_CHECK_THROWS(make_Date(32, 1, 2000));
            DOCTEST_CHECK_THROWS(make_Date(1, 13, 2000));
            DOCTEST_CHECK_THROWS(make_Date(1, 1, 999));
            DOCTEST_CHECK_THROWS(make_Date(0, 2000));
            DOCTEST_CHECK_THROWS(make_Date(367, 2000));
            DOCTEST_CHECK_THROWS(make_Date(1, 999));
            DOCTEST_CHECK(make_Date(10, 2000) == d);

            // Julian2Calendar
            DOCTEST_CHECK_THROWS(Julian2Calendar(0, 2000));
            DOCTEST_CHECK_THROWS(Julian2Calendar(367, 2000));
            DOCTEST_CHECK(Julian2Calendar(366, 2000).day == 31);
            DOCTEST_CHECK(Julian2Calendar(366, 2000).month == 12);

            // Calendar2Julian
            DOCTEST_CHECK(Calendar2Julian(d) == 10);
            DOCTEST_CHECK(Calendar2Julian(Date(31, 12, 2001)) == 365);

            // Stream operator
            std::stringstream ss;
            ss << d;
            DOCTEST_CHECK(ss.str() == "2000-01-10");
        }

        SUBCASE("Time struct and helpers") {
            Time t(1, 2, 3);
            // Constructors
            DOCTEST_CHECK_THROWS(Time(24, 1));
            DOCTEST_CHECK_THROWS(Time(1, 60));
            DOCTEST_CHECK_THROWS(Time(1, 1, 60));
            DOCTEST_CHECK_THROWS(Time(1, 60, 1));
            DOCTEST_CHECK_THROWS(Time(24, 1, 1));

            // Operators
            DOCTEST_CHECK(t == Time(1, 2, 3));
            DOCTEST_CHECK(t != Time(1, 2, 4));

            // make_Time
            DOCTEST_CHECK(make_Time(1, 2) == Time(1, 2, 0));
            DOCTEST_CHECK_THROWS(make_Time(24, 1));
            DOCTEST_CHECK_THROWS(make_Time(1, 60));
            DOCTEST_CHECK_THROWS(make_Time(1, 1, 60));
            DOCTEST_CHECK_THROWS(make_Time(1, 60, 1));
            DOCTEST_CHECK_THROWS(make_Time(24, 1, 1));

            // Stream operator
            std::stringstream ss;
            ss << t;
            DOCTEST_CHECK(ss.str() == "1:02:03");
        }

        SUBCASE("Location struct") {
            Location l(1.f, 2.f, 3.f);

            // Operators
            DOCTEST_CHECK(l == Location(1.f, 2.f, 3.f));
            DOCTEST_CHECK(l != Location(1.f, 2.f, 4.f));

            // Stream operator
            std::stringstream ss;
            ss << l;
            DOCTEST_CHECK(ss.str() == "<1,2,3>");
        }

        SUBCASE("Julian day conversion") {
            int year = 2000;
            Date d = Julian2Calendar(10, year);
            DOCTEST_CHECK(d.year == year);
            DOCTEST_CHECK(d.month == 1);
            DOCTEST_CHECK(d.day == 10);
            d = Julian2Calendar(230, year);
            DOCTEST_CHECK(d.month == 8);
            DOCTEST_CHECK(d.day == 17);
            year = 2001;
            d = Julian2Calendar(230, year);
            DOCTEST_CHECK(d.month == 8);
            DOCTEST_CHECK(d.day == 18);
        }

        SUBCASE("Date methods") {
            SUBCASE("incrementDay") {
                Date d(31, 12, 2020);
                d.incrementDay();
                DOCTEST_CHECK(d.day == 1);
                DOCTEST_CHECK(d.month == 1);
                DOCTEST_CHECK(d.year == 2021);

                Date d2(28, 2, 2021);
                d2.incrementDay();
                DOCTEST_CHECK(d2.day == 1);
                DOCTEST_CHECK(d2.month == 3);
                DOCTEST_CHECK(d2.year == 2021);

                Date d3(28, 2, 2020);
                d3.incrementDay();
                DOCTEST_CHECK(d3.day == 29);
                DOCTEST_CHECK(d3.month == 2);
                DOCTEST_CHECK(d3.year == 2020);
            }
            SUBCASE("isLeapYear") {
                Date d1(1, 1, 2020);
                DOCTEST_CHECK(d1.isLeapYear());
                Date d2(1, 1, 2021);
                DOCTEST_CHECK(!d2.isLeapYear());
                Date d3(1, 1, 2000);
                DOCTEST_CHECK(d3.isLeapYear());
                Date d4(1, 1, 2100);
                DOCTEST_CHECK(!d4.isLeapYear());
            }
        }
    }

    TEST_CASE("Coordinate System Conversions") {
        SUBCASE("Spherical coordinate struct and conversions") {
            SphericalCoord sc(1.f, 0.5f, 1.5f);

            // Constructors
#ifdef HELIOS_DEBUG
            DOCTEST_CHECK_THROWS(make_SphericalCoord(-1.f, 0.f, 0.f));
#endif

            // Operators
            DOCTEST_CHECK(sc == SphericalCoord(1.f, 0.5f, 1.5f));

            // Methods
            vec3 cart = sphere2cart(sc);
            DOCTEST_CHECK(cart.x == doctest::Approx(1.f * cos(0.5f) * sin(1.5f)));
            DOCTEST_CHECK(cart.y == doctest::Approx(1.f * cos(0.5f) * cos(1.5f)));
            DOCTEST_CHECK(cart.z == doctest::Approx(1.f * sin(0.5f)));

            // Stream operator
            std::stringstream ss;
            ss << sc;
            DOCTEST_CHECK(ss.str() == "helios::SphericalCoord<1, 0.5, 1.5>");
        }
        SUBCASE("Cartesian to Spherical and back") {
            SphericalCoord sph = make_SphericalCoord(1.f, 0.25 * PI_F, 1.5 * PI_F);
            vec3 cart = sphere2cart(sph);
            SphericalCoord sph2 = cart2sphere(cart);
            DOCTEST_CHECK(sph.radius == doctest::Approx(sph2.radius).epsilon(errtol));
            DOCTEST_CHECK(sph.elevation == doctest::Approx(sph2.elevation).epsilon(errtol));
            DOCTEST_CHECK(sph.zenith == doctest::Approx(sph2.zenith).epsilon(errtol));
            DOCTEST_CHECK(sph.azimuth == doctest::Approx(sph2.azimuth).epsilon(errtol));
        }
    }
}

// =================================================================================
// Suite 2: Global Helper Functions (global.h)
//
// Tests for standalone utility functions, typically found in a global header,
// that provide common functionality like math, string parsing, and file handling.
// =================================================================================
TEST_SUITE("Global Helper Functions (global.h)") {

    TEST_CASE("Mathematical and Geometric Helpers") {
        SUBCASE("global.h utilities") {
            SUBCASE("deg2rad and rad2deg") {
                DOCTEST_CHECK(deg2rad(180.f) == doctest::Approx(PI_F));
                DOCTEST_CHECK(rad2deg(PI_F) == doctest::Approx(180.f));
            }

            SUBCASE("clamp") {
                DOCTEST_CHECK(clamp(5, 0, 10) == 5);
                DOCTEST_CHECK(clamp(-5, 0, 10) == 0);
                DOCTEST_CHECK(clamp(15, 0, 10) == 10);
                DOCTEST_CHECK(clamp(5.5f, 0.f, 10.f) == doctest::Approx(5.5f));
            }

            SUBCASE("Safe trigonometry") {
                DOCTEST_CHECK(acos_safe(1.000001f) == doctest::Approx(0.f));
                DOCTEST_CHECK(acos_safe(-1.000001f) == doctest::Approx(PI_F));
                DOCTEST_CHECK(asin_safe(1.000001f) == doctest::Approx(PI_F / 2.f));
                DOCTEST_CHECK(asin_safe(-1.000001f) == doctest::Approx(-PI_F / 2.f));
            }

            SUBCASE("Integer power") {
                DOCTEST_CHECK(powi(2, 10) == 1024);
                DOCTEST_CHECK(powi(3.f, 3) == doctest::Approx(27.f));
            }

            SUBCASE("calculateTriangleArea") {
                vec3 v0(0, 0, 0), v1(1, 0, 0), v2(0, 1, 0);
                DOCTEST_CHECK(calculateTriangleArea(v0, v1, v2) == doctest::Approx(0.5f));
            }

            SUBCASE("blend colors") {
                RGBcolor c1(1, 0, 0), c2(0, 1, 0);
                RGBcolor blended = blend(c1, c2, 0.5f);
                DOCTEST_CHECK(blended.r == doctest::Approx(0.5f));
                DOCTEST_CHECK(blended.g == doctest::Approx(0.5f));
                DOCTEST_CHECK(blended.b == doctest::Approx(0.f));

                RGBAcolor ca1(1, 0, 0, 0), ca2(0, 1, 0, 1);
                RGBAcolor blended_a = blend(ca1, ca2, 0.5f);
                DOCTEST_CHECK(blended_a.r == doctest::Approx(0.5f));
                DOCTEST_CHECK(blended_a.g == doctest::Approx(0.5f));
                DOCTEST_CHECK(blended_a.b == doctest::Approx(0.f));
                DOCTEST_CHECK(blended_a.a == doctest::Approx(0.5f));
            }
        }
        SUBCASE("global.h geometric functions") {
            SUBCASE("atan2_2pi") {
                DOCTEST_CHECK(atan2_2pi(0, 1) == doctest::Approx(0));
                DOCTEST_CHECK(atan2_2pi(1, 0) == doctest::Approx(PI_F / 2.0));
                DOCTEST_CHECK(atan2_2pi(0, -1) == doctest::Approx(PI_F));
                DOCTEST_CHECK(atan2_2pi(-1, 0) == doctest::Approx(3.0 * PI_F / 2.0));
                DOCTEST_CHECK(atan2_2pi(-1, -1) > PI_F);
            }

            SUBCASE("rotatePoint") {
                vec3 p(1, 0, 0);
                vec3 rotated = rotatePoint(p, PI_F / 2.f, PI_F / 2.f); // elevation, azimuth
                DOCTEST_CHECK(rotated.x == doctest::Approx(0.f));
                DOCTEST_CHECK(rotated.y == doctest::Approx(0.f));
                DOCTEST_CHECK(rotated.z == doctest::Approx(-1.f));
            }

            SUBCASE("rotatePointAboutLine") {
                vec3 point(1, 0, 0);
                vec3 line_base(0, 0, 0);
                vec3 line_dir(0, 0, 1);
                vec3 rotated = rotatePointAboutLine(point, line_base, line_dir, PI_F / 2.f);
                DOCTEST_CHECK(rotated.x == doctest::Approx(0.f));
                DOCTEST_CHECK(rotated.y == doctest::Approx(1.f));
                DOCTEST_CHECK(rotated.z == doctest::Approx(0.f));
            }

            SUBCASE("lineIntersection") {
                vec2 p1(0, 0), q1(2, 2);
                vec2 p2(0, 2), q2(2, 0);
                DOCTEST_CHECK(lineIntersection(p1, q1, p2, q2));

                vec2 p3(0, 0), q3(1, 1);
                vec2 p4(2, 2), q4(3, 3);
                DOCTEST_CHECK(!lineIntersection(p3, q3, p4, q4));
            }

            SUBCASE("pointInPolygon") {
                std::vector<vec2> square = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
                DOCTEST_CHECK(pointInPolygon(vec2(0.5, 0.5), square));
                DOCTEST_CHECK(!pointInPolygon(vec2(1.5, 0.5), square));
                DOCTEST_CHECK(pointInPolygon(vec2(0, 0), square)); // On edge
            }
        }
        SUBCASE("Spline Interpolation") {
            vec3 p0 = make_vec3(0, 0, 0);
            vec3 t0 = make_vec3(3, 0, 0);
            vec3 p1 = make_vec3(1, 1.5, 0.4);
            vec3 t1 = make_vec3(0, 1, 0);
            vec3 xi = spline_interp3(0.6f, p0, t0, p1, t1);
            vec3 ref = make_vec3(0.9360f, 0.8280f, 0.2592f);
            DOCTEST_CHECK(xi.x == doctest::Approx(ref.x).epsilon(errtol));
            DOCTEST_CHECK(xi.y == doctest::Approx(ref.y).epsilon(errtol));
            DOCTEST_CHECK(xi.z == doctest::Approx(ref.z).epsilon(errtol));
        }
        SUBCASE("Angle conversion helpers") {
            CHECK(deg2rad(180.f) == doctest::Approx(PI_F).epsilon(errtol));
            CHECK(rad2deg(PI_F) == doctest::Approx(180.f).epsilon(errtol));

            CHECK(atan2_2pi(0.f, 1.f) == doctest::Approx(0.f).epsilon(errtol));
            CHECK(atan2_2pi(1.f, 0.f) == doctest::Approx(0.5f * PI_F * 2.f / 2.f).epsilon(errtol));
            CHECK(atan2_2pi(0.f, -1.f) == doctest::Approx(PI_F).epsilon(errtol));
            CHECK(atan2_2pi(-1.f, 0.f) == doctest::Approx(1.5f * PI_F).epsilon(errtol));
        }

        SUBCASE("clamp template for common types") {
            CHECK(clamp(5, 1, 4) == 4);
            CHECK(clamp(-1, 0, 10) == 0);
            CHECK(clamp(3.5f, 0.f, 10.f) == doctest::Approx(3.5f).epsilon(errtol));
            CHECK(clamp(12.0, -5.0, 11.0) == doctest::Approx(11.0).epsilon(errtol));
        }
    }

    TEST_CASE("Matrix and Transformation Helpers") {
        float M[16], T[16], R[16];

        SUBCASE("makeIdentityMatrix") {
            makeIdentityMatrix(T);
            DOCTEST_CHECK(T[0] == 1.f);
            DOCTEST_CHECK(T[1] == 0.f);
            DOCTEST_CHECK(T[2] == 0.f);
            DOCTEST_CHECK(T[3] == 0.f);
            DOCTEST_CHECK(T[4] == 0.f);
            DOCTEST_CHECK(T[5] == 1.f);
            DOCTEST_CHECK(T[6] == 0.f);
            DOCTEST_CHECK(T[7] == 0.f);
            DOCTEST_CHECK(T[8] == 0.f);
            DOCTEST_CHECK(T[9] == 0.f);
            DOCTEST_CHECK(T[10] == 1.f);
            DOCTEST_CHECK(T[11] == 0.f);
            DOCTEST_CHECK(T[12] == 0.f);
            DOCTEST_CHECK(T[13] == 0.f);
            DOCTEST_CHECK(T[14] == 0.f);
            DOCTEST_CHECK(T[15] == 1.f);
        }

        SUBCASE("makeRotationMatrix") {
            // About x-axis
            makeRotationMatrix(PI_F / 2.f, "x", T);
            DOCTEST_CHECK(T[5] == doctest::Approx(0.f));
            DOCTEST_CHECK(T[6] == doctest::Approx(-1.f));
            DOCTEST_CHECK(T[9] == doctest::Approx(1.f));
            DOCTEST_CHECK(T[10] == doctest::Approx(0.f));

            // About y-axis
            makeRotationMatrix(PI_F / 2.f, "y", T);
            DOCTEST_CHECK(T[0] == doctest::Approx(0.f));
            DOCTEST_CHECK(T[2] == doctest::Approx(1.f));
            DOCTEST_CHECK(T[8] == doctest::Approx(-1.f));
            DOCTEST_CHECK(T[10] == doctest::Approx(0.f));

            // About z-axis
            makeRotationMatrix(PI_F / 2.f, "z", T);
            DOCTEST_CHECK(T[0] == doctest::Approx(0.f));
            DOCTEST_CHECK(T[1] == doctest::Approx(-1.f));
            DOCTEST_CHECK(T[4] == doctest::Approx(1.f));
            DOCTEST_CHECK(T[5] == doctest::Approx(0.f));

            // About arbitrary axis
            vec3 axis(1, 0, 0);
            makeRotationMatrix(PI_F / 2.f, axis, T);
            DOCTEST_CHECK(T[5] == doctest::Approx(0.f));
            DOCTEST_CHECK(T[6] == doctest::Approx(-1.f));
            DOCTEST_CHECK(T[9] == doctest::Approx(1.f));
            DOCTEST_CHECK(T[10] == doctest::Approx(0.f));
        }

        SUBCASE("makeTranslationMatrix") {
            makeTranslationMatrix(vec3(1, 2, 3), T);
            DOCTEST_CHECK(T[3] == 1.f);
            DOCTEST_CHECK(T[7] == 2.f);
            DOCTEST_CHECK(T[11] == 3.f);
        }

        SUBCASE("makeScaleMatrix") {
            // About origin
            makeScaleMatrix(vec3(2, 3, 4), T);
            DOCTEST_CHECK(T[0] == 2.f);
            DOCTEST_CHECK(T[5] == 3.f);
            DOCTEST_CHECK(T[10] == 4.f);

            // About point
            makeScaleMatrix(vec3(2, 2, 2), vec3(1, 1, 1), T);
            vec3 v(2, 2, 2);
            vec3 res;
            vecmult(T, v, res);
            DOCTEST_CHECK(res.x == doctest::Approx(3.f));
            DOCTEST_CHECK(res.y == doctest::Approx(3.f));
            DOCTEST_CHECK(res.z == doctest::Approx(3.f));
        }

        SUBCASE("matmult") {
            makeRotationMatrix(PI_F / 2.f, "x", M);
            makeRotationMatrix(-PI_F / 2.f, "x", T);
            matmult(M, T, R); // Should be identity
            DOCTEST_CHECK(R[0] == doctest::Approx(1.f));
            DOCTEST_CHECK(R[5] == doctest::Approx(1.f));
            DOCTEST_CHECK(R[10] == doctest::Approx(1.f));
            DOCTEST_CHECK(R[15] == doctest::Approx(1.f));
        }

        SUBCASE("vecmult") {
            makeRotationMatrix(PI_F / 2.f, "z", M);
            vec3 v(1, 0, 0);
            vec3 res;
            vecmult(M, v, res);
            DOCTEST_CHECK(res.x == doctest::Approx(0.f));
            DOCTEST_CHECK(res.y == doctest::Approx(1.f));
            DOCTEST_CHECK(res.z == doctest::Approx(0.f));

            float v_arr[3] = {1, 0, 0};
            float res_arr[3];
            vecmult(M, v_arr, res_arr);
            DOCTEST_CHECK(res_arr[0] == doctest::Approx(0.f));
            DOCTEST_CHECK(res_arr[1] == doctest::Approx(1.f));
            DOCTEST_CHECK(res_arr[2] == doctest::Approx(0.f));
        }

        SUBCASE("makeIdentityMatrix / matmult / vecmult") {
            float I[16] = {};
            makeIdentityMatrix(I);
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    CHECK(I[4 * r + c] == doctest::Approx(r == c ? 1.f : 0.f).epsilon(errtol));
                }
            }

            float T[16];
            makeTranslationMatrix(make_vec3(1.f, 2.f, 3.f), T);
            float R[16];
            matmult(I, T, R); // I * T == T
            for (int i = 0; i < 16; ++i) {
                CHECK(R[i] == doctest::Approx(T[i]).epsilon(errtol));
            }

            float v[3] = {4.f, 5.f, 6.f}, out[3] = {};
            vecmult(I, v, out);
            CHECK(out[0] == doctest::Approx(4.f).epsilon(errtol));
            CHECK(out[1] == doctest::Approx(5.f).epsilon(errtol));
            CHECK(out[2] == doctest::Approx(6.f).epsilon(errtol));
        }

        SUBCASE("rotatePoint & rotatePointAboutLine") {
            vec3 p = make_vec3(1, 0, 0);
            // Rotate 90° about z -> y axis
            vec3 r = rotatePoint(p, 0.f, 0.5f * PI_F);
            CHECK(r.x == doctest::Approx(0.f).epsilon(errtol));
            CHECK(r.y == doctest::Approx(1.f).epsilon(errtol));

            // Rotate about arbitrary line (x-axis) back to original
            vec3 q = rotatePointAboutLine(r, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -0.5f * PI_F);
            CHECK(q.x == doctest::Approx(0.f).epsilon(errtol));
            CHECK(q.y == doctest::Approx(0.f).epsilon(errtol));
        }
    }

    TEST_CASE("String, File Path, and Parsing Utilities") {
        SUBCASE("String Manipulation") {
            DOCTEST_CHECK(deblank("  hello world  ") == "helloworld");
            DOCTEST_CHECK(trim_whitespace("  hello world  ") == "hello world");
        }
        SUBCASE("String Delimiting") {
            std::vector<std::string> result = separate_string_by_delimiter("a,b,c", ",");
            DOCTEST_CHECK(result.size() == 3);
            if (result.size() == 3) {
                DOCTEST_CHECK(result[0] == "a");
                DOCTEST_CHECK(result[1] == "b");
                DOCTEST_CHECK(result[2] == "c");
            }
        }
        SUBCASE("String to Vector Conversions") {
            DOCTEST_CHECK(string2vec2("1.5 2.5") == vec2(1.5f, 2.5f));
            DOCTEST_CHECK(string2vec3("1.5 2.5 3.5") == vec3(1.5f, 2.5f, 3.5f));
            DOCTEST_CHECK(string2vec4("1.5 2.5 3.5 4.5") == vec4(1.5f, 2.5f, 3.5f, 4.5f));
            DOCTEST_CHECK(string2int2("1 2") == int2(1, 2));
            DOCTEST_CHECK(string2int3("1 2 3") == int3(1, 2, 3));
            DOCTEST_CHECK(string2int4("1 2 3 4") == int4(1, 2, 3, 4));
        }
        SUBCASE("String to Color Conversion") {
            RGBAcolor color = string2RGBcolor("0.1 0.2 0.3 0.4");
            DOCTEST_CHECK(color.r == doctest::Approx(0.1f));
            DOCTEST_CHECK(color.g == doctest::Approx(0.2f));
            DOCTEST_CHECK(color.b == doctest::Approx(0.3f));
            DOCTEST_CHECK(color.a == doctest::Approx(0.4f));

            color = string2RGBcolor("0.5 0.6 0.7");
            DOCTEST_CHECK(color.r == doctest::Approx(0.5f));
            DOCTEST_CHECK(color.g == doctest::Approx(0.6f));
            DOCTEST_CHECK(color.b == doctest::Approx(0.7f));
            DOCTEST_CHECK(color.a == doctest::Approx(1.0f));
        }
        SUBCASE("File Path Parsing") {
            std::string filepath = "/path/to/file/filename.ext";
            DOCTEST_CHECK(getFileExtension(filepath) == ".ext");
            DOCTEST_CHECK(getFileName(filepath) == "filename.ext");
            DOCTEST_CHECK(getFileStem(filepath) == "filename");
    #ifndef _WIN32
            DOCTEST_CHECK(getFilePath(filepath, true) == "/path/to/file/");
            DOCTEST_CHECK(getFilePath(filepath, false) == "/path/to/file");
    #endif

            std::string filepath_noext = "/path/to/file/filename";
            DOCTEST_CHECK(getFileExtension(filepath_noext).empty());
            DOCTEST_CHECK(getFileName(filepath_noext) == "filename");
            DOCTEST_CHECK(getFileStem(filepath_noext) == "filename");

            std::string filepath_nodir = "filename.ext";
            DOCTEST_CHECK(getFileExtension(filepath_nodir) == ".ext");
            DOCTEST_CHECK(getFileName(filepath_nodir) == "filename.ext");
            DOCTEST_CHECK(getFileStem(filepath_nodir) == "filename");
            DOCTEST_CHECK(getFilePath(filepath_nodir, true).empty());
        }
        SUBCASE("Primitive Type Parsing") {
            float f;
            DOCTEST_CHECK(parse_float("1.23", f));
            DOCTEST_CHECK(f == doctest::Approx(1.23f));
            DOCTEST_CHECK(!parse_float("abc", f));

            double d;
            DOCTEST_CHECK(parse_double("1.23", d));
            DOCTEST_CHECK(d == doctest::Approx(1.23));
            DOCTEST_CHECK(!parse_double("abc", d));

            int i;
            DOCTEST_CHECK(parse_int("123", i));
            DOCTEST_CHECK(i == 123);
            DOCTEST_CHECK(!parse_int("abc", i));
            DOCTEST_CHECK(!parse_int("1.23", i));

            unsigned int u;
            DOCTEST_CHECK(parse_uint("123", u));
            DOCTEST_CHECK(u == 123u);
            DOCTEST_CHECK(!parse_uint("-123", u));
        }
        SUBCASE("Compound Type Parsing") {
            int2 i2;
            DOCTEST_CHECK(parse_int2("1 2", i2));
            DOCTEST_CHECK(i2 == int2(1, 2));
            DOCTEST_CHECK(!parse_int2("1", i2));

            int3 i3;
            DOCTEST_CHECK(parse_int3("1 2 3", i3));
            DOCTEST_CHECK(i3 == int3(1, 2, 3));
            DOCTEST_CHECK(!parse_int3("1 2", i3));

            vec2 v2;
            DOCTEST_CHECK(parse_vec2("1.1 2.2", v2));
            DOCTEST_CHECK(v2.x == doctest::Approx(1.1f));
            DOCTEST_CHECK(v2.y == doctest::Approx(2.2f));
            DOCTEST_CHECK(!parse_vec2("1.1", v2));

            vec3 v3;
            DOCTEST_CHECK(parse_vec3("1.1 2.2 3.3", v3));
            DOCTEST_CHECK(v3.x == doctest::Approx(1.1f));
            DOCTEST_CHECK(!parse_vec3("1.1 2.2", v3));

            RGBcolor rgb;
            DOCTEST_CHECK(parse_RGBcolor("0.1 0.2 0.3", rgb));
            DOCTEST_CHECK(rgb.r == doctest::Approx(0.1f));
            DOCTEST_CHECK(!parse_RGBcolor("0.1 0.2", rgb));
        }
    }

    TEST_CASE("Vector Statistics and Manipulation") {
        SUBCASE("Vector Statistics") {
            std::vector<float> v = {1.f, 2.f, 3.f, 4.f, 5.f};
            DOCTEST_CHECK(sum(v) == doctest::Approx(15.f));
            DOCTEST_CHECK(mean(v) == doctest::Approx(3.f));
            DOCTEST_CHECK(min(v) == doctest::Approx(1.f));
            DOCTEST_CHECK(max(v) == doctest::Approx(5.f));
            DOCTEST_CHECK(stdev(v) == doctest::Approx(sqrtf(2.f)));
            DOCTEST_CHECK(median(v) == doctest::Approx(3.f));
            std::vector<float> v2 = {1.f, 2.f, 3.f, 4.f};
            DOCTEST_CHECK(median(v2) == doctest::Approx(2.5f));
        }
        SUBCASE("sum / mean / min / max / stdev / median") {
            std::vector<float> vf{1.f, 2.f, 3.f, 4.f, 5.f};
            CHECK(sum(vf) == doctest::Approx(15.f).epsilon(errtol));
            CHECK(mean(vf) == doctest::Approx(3.f).epsilon(errtol));
            CHECK(min(vf) == doctest::Approx(1.f).epsilon(errtol));
            CHECK(max(vf) == doctest::Approx(5.f).epsilon(errtol));
            CHECK(median(vf) == doctest::Approx(3.f).epsilon(errtol));
            CHECK(stdev(vf) == doctest::Approx(std::sqrt(2.f)).epsilon(1e-4));

            std::vector<int> vi{9, 4, -3, 10, 2};
            CHECK(min(vi) == -3);
            CHECK(max(vi) == 10);

            std::vector<vec3> vv{make_vec3(2, 3, 4), make_vec3(-1, 7, 0)};
            CHECK(min(vv) == make_vec3(-1, 3, 0));
            CHECK(max(vv) == make_vec3(2, 7, 4));
        }
        SUBCASE("Vector Manipulation") {
            SUBCASE("resize_vector") {
                std::vector<std::vector<int>> vec2d;
                resize_vector(vec2d, 2, 3);
                DOCTEST_CHECK(vec2d.size() == 3);
                DOCTEST_CHECK(vec2d[0].size() == 2);

                std::vector<std::vector<std::vector<int>>> vec3d;
                resize_vector(vec3d, 2, 3, 4);
                DOCTEST_CHECK(vec3d.size() == 4);
                DOCTEST_CHECK(vec3d[0].size() == 3);
                DOCTEST_CHECK(vec3d[0][0].size() == 2);

                std::vector<std::vector<std::vector<std::vector<int>>>> vec4d;
                resize_vector(vec4d, 2, 3, 4, 5);
                DOCTEST_CHECK(vec4d.size() == 5);
                DOCTEST_CHECK(vec4d[0].size() == 4);
                DOCTEST_CHECK(vec4d[0][0].size() == 3);
                DOCTEST_CHECK(vec4d[0][0][0].size() == 2);
            }

            SUBCASE("flatten") {
                std::vector<std::vector<int>> vec2d = {{1, 2}, {3, 4}};
                std::vector<int> flat = flatten(vec2d);
                DOCTEST_CHECK(flat.size() == 4);
                DOCTEST_CHECK(flat[3] == 4);

                std::vector<std::vector<std::vector<int>>> vec3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
                flat = flatten(vec3d);
                DOCTEST_CHECK(flat.size() == 8);
                DOCTEST_CHECK(flat[7] == 8);

                std::vector<std::vector<std::vector<std::vector<int>>>> vec4d = {{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}};
                flat = flatten(vec4d);
                DOCTEST_CHECK(flat.size() == 16);
                DOCTEST_CHECK(flat[15] == 16);
            }

            SUBCASE("vector operators") {
                std::vector<float> v1 = {1, 2, 3};
                std::vector<float> v2 = {4, 5, 6};
                std::vector<float> v_sum = v1 + v2;
                DOCTEST_CHECK(v_sum[0] == 5.f);
                DOCTEST_CHECK(v_sum[2] == 9.f);

                v1 += v2;
                DOCTEST_CHECK(v1[0] == 5.f);
                DOCTEST_CHECK(v1[2] == 9.f);

                std::vector<float> v3 = {1};
                DOCTEST_CHECK_THROWS(v1 + v3);
            }
        }
    }

    TEST_CASE("Miscellaneous Utilities") {
        SUBCASE("1D Interpolation") {
            std::vector<vec2> points = {{0, 0}, {1, 1}, {2, 0}};
            float res = interp1(points, 0.5f);
            DOCTEST_CHECK(res == doctest::Approx(0.5f));
            res = interp1(points, 1.5f);
            DOCTEST_CHECK(res == doctest::Approx(0.5f));
            res = interp1(points, -1.f);
            DOCTEST_CHECK(res == doctest::Approx(0.f));
            res = interp1(points, 3.f);
            DOCTEST_CHECK(res == doctest::Approx(0.f));
        }
        SUBCASE("Timer") {
            Timer t;
            t.tic();
            // Not a real test of time, just that it runs.
            double elapsed = t.toc("mute");
            DOCTEST_CHECK(elapsed >= 0);
        }
        SUBCASE("Custom Error") {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_THROWS_AS(helios_runtime_error("test error"), std::runtime_error);
        }
        SUBCASE("Random Number Generation") {
            float r = randu();
            DOCTEST_CHECK(r >= 0.f);
            DOCTEST_CHECK(r <= 1.f);

            int ri = randu(1, 10);
            DOCTEST_CHECK(ri >= 1);
            DOCTEST_CHECK(ri <= 10);
        }
        SUBCASE("acos_safe & asin_safe clamp inputs") {
            CHECK(acos_safe(1.5f) == doctest::Approx(0.f).epsilon(errtol));
            CHECK(asin_safe(-2.f) == doctest::Approx(-0.5f * PI_F).epsilon(errtol));
        }

        SUBCASE("randu range checks") {
            for (int i = 0; i < 100; ++i) {
                float r = randu();
                CHECK(r >= 0.f);
                CHECK(r <= 1.f);

                int ri = randu(5, 10);
                CHECK(ri >= 5);
                CHECK(ri <= 10);
                CHECK(randu(3, 3) == 3);
            }
        }
    }
}


// =================================================================================
// Suite 3: XML Parsing Utilities
//
// Tests for functions related to loading data from XML files and nodes.
// =================================================================================
TEST_SUITE("XML Parsing Utilities") {

    TEST_CASE("XMLoad functions") {
        pugi::xml_document doc;
        pugi::xml_node node = doc.append_child("node");
        node.append_child("float").text().set("1.23");
        node.append_child("int").text().set("123");
        node.append_child("string").text().set("hello");
        node.append_child("vec2").text().set("1.1 2.2");
        node.append_child("vec3").text().set("1.1 2.2 3.3");
        node.append_child("vec4").text().set("1.1 2.2 3.3 4.4");
        node.append_child("int2").text().set("1 2");
        node.append_child("int3").text().set("1 2 3");
        node.append_child("int4").text().set("1 2 3 4");
        node.append_child("rgb").text().set("0.1 0.2 0.3");
        node.append_child("rgba").text().set("0.1 0.2 0.3 0.4");

        SUBCASE("XMLloadfloat") {
            float f = XMLloadfloat(node, "float");
            DOCTEST_CHECK(f == doctest::Approx(1.23f));
        }
        SUBCASE("XMLloadint") {
            int i = XMLloadint(node, "int");
            DOCTEST_CHECK(i == 123);
        }
        SUBCASE("XMLloadstring") {
            std::string s = XMLloadstring(node, "string");
            DOCTEST_CHECK(s == "hello");
        }
        SUBCASE("XMLloadvec2") {
            vec2 v = XMLloadvec2(node, "vec2");
            DOCTEST_CHECK(v == vec2(1.1f, 2.2f));
        }
        SUBCASE("XMLloadvec3") {
            vec3 v = XMLloadvec3(node, "vec3");
            DOCTEST_CHECK(v == vec3(1.1f, 2.2f, 3.3f));
        }
        SUBCASE("XMLloadvec4") {
            vec4 v = XMLloadvec4(node, "vec4");
            DOCTEST_CHECK(v == vec4(1.1f, 2.2f, 3.3f, 4.4f));
        }
        SUBCASE("XMLloadint2") {
            int2 i = XMLloadint2(node, "int2");
            DOCTEST_CHECK(i == int2(1, 2));
        }
        SUBCASE("XMLloadint3") {
            int3 i = XMLloadint3(node, "int3");
            DOCTEST_CHECK(i == int3(1, 2, 3));
        }
        SUBCASE("XMLloadint4") {
            int4 i = XMLloadint4(node, "int4");
            DOCTEST_CHECK(i == int4(1, 2, 3, 4));
        }
    }
}


// =================================================================================
// Suite 4: Numerical Methods
//
// Tests for numerical algorithms such as root-finding.
// =================================================================================
TEST_SUITE("Numerical Methods") {

    static float quadratic(float x, std::vector<float>&, const void*) {
        return x * x - 4.0f;
    }
    static float linear(float x, std::vector<float>&, const void*) {
        return 3.5f - x;
    }
    static float flat(float, std::vector<float>&, const void*) {
        return 2.0f;
    }
    static float cubic(float x, std::vector<float>&, const void*) {
        return (x - 1.0f) * (x + 2.0f) * (x - 4.0f);
    }
    static float near_singular(float x, std::vector<float>&, const void*) {
        return (x - 1e-3f) * (x - 1e-3f);
    }
    TEST_CASE("fzero") {
        SUBCASE("fzero finds positive quadratic root") {
            std::vector<float> v;
            float root = helios::fzero(quadratic, v, nullptr, 1.0f, 1e-5f, 50);
            DOCTEST_CHECK(root == doctest::Approx(2.0f).epsilon(errtol));
        }

        SUBCASE("fzero finds root far from initial guess") {
            std::vector<float> v;
            float root = helios::fzero(linear, v, nullptr, -10.0f, 1e-6f, 50);
            DOCTEST_CHECK( root == doctest::Approx(3.5f).epsilon(errtol));
        }

        SUBCASE("fzero handles function without zero") {
            std::vector<float> v;
            capture_cerr cerr_buffer;
            float root = helios::fzero(flat, v, nullptr, 0.0f, 1e-6f, 10);
            DOCTEST_CHECK(std::isfinite(root));
            DOCTEST_CHECK(cerr_buffer.has_output());
        }

        SUBCASE("fzero returns exact root at initial guess") {
            std::vector<float> v;
            float root = helios::fzero(quadratic, v, nullptr, 2.0f, 1e-6f, 5);
            DOCTEST_CHECK(root == doctest::Approx(2.0f).epsilon(errtol));
        }

        SUBCASE("fzero finds a cubic root") {
            std::vector<float> v;
            float root = helios::fzero(cubic, v, nullptr, 3.5f, 1e-5f, 80);
            DOCTEST_CHECK(root == doctest::Approx(4.0f).epsilon(errtol) );
        }

        SUBCASE("fzero copes with near-singular derivative") {
            std::vector<float> v;
            float root = helios::fzero(near_singular, v, nullptr, 0.01f, 1e-4f, 50);
            DOCTEST_CHECK(std::fabs(near_singular(root, v, nullptr)) < 1e-4f);
        }
    }
}

// =================================================================================
// Suite 5: Context Class
//
// Tests for the main Context class, which manages the scene, primitives,
// objects, data, and simulation state.
// =================================================================================
TEST_SUITE("Context Class") {

    TEST_CASE("Core Context State and Configuration") {
        SUBCASE("Constructor and basic setup") {
            Context ctx;
            DOCTEST_CHECK(ctx.getPrimitiveCount() == 0);
            DOCTEST_CHECK(ctx.getObjectCount() == 0);
            DOCTEST_CHECK(!ctx.isGeometryDirty());

            Date d = ctx.getDate();
            DOCTEST_CHECK(d.day == 1);
            DOCTEST_CHECK(d.month == 6);
            DOCTEST_CHECK(d.year == 2000);

            Time t = ctx.getTime();
            DOCTEST_CHECK(t.hour == 12);
            DOCTEST_CHECK(t.minute == 0);
            DOCTEST_CHECK(t.second == 0);

            Location l = ctx.getLocation();
            DOCTEST_CHECK(l.latitude_deg == doctest::Approx(38.55));
            DOCTEST_CHECK(l.longitude_deg == doctest::Approx(121.76));
            DOCTEST_CHECK(l.UTC_offset == doctest::Approx(8));
        }

        SUBCASE("Random number generator") {
            Context ctx;
            ctx.seedRandomGenerator(12345);
            std::minstd_rand0* gen1 = ctx.getRandomGenerator();
            float rand1 = (*gen1)();

            ctx.seedRandomGenerator(12345);
            std::minstd_rand0* gen2 = ctx.getRandomGenerator();
            float rand2 = (*gen2)();

            DOCTEST_CHECK(rand1 == rand2);

            float r_uniform = ctx.randu();
            DOCTEST_CHECK(r_uniform >= 0.f);
            DOCTEST_CHECK(r_uniform <= 1.f);

            float r_norm = ctx.randn();
            // Hard to test for normality, but let's check it's a number
            DOCTEST_CHECK(!std::isnan(r_norm));
        }

        SUBCASE("Texture utility methods") {
            Context ctx;
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, "lib/images/solid.jpg"));
            DOCTEST_CHECK_THROWS(ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, "lib/images/missing.png"));
            DOCTEST_CHECK_THROWS(ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, "lib/images/invalid.txt"));

            Texture tex("lib/images/solid.jpg");
            DOCTEST_CHECK(tex.getTextureFile() == "lib/images/solid.jpg");
            int2 res = tex.getImageResolution();
            DOCTEST_CHECK(res.x == 5);
            DOCTEST_CHECK(res.y == 5);
            DOCTEST_CHECK(!tex.hasTransparencyChannel());
            const auto* alpha = tex.getTransparencyData();
            DOCTEST_CHECK(alpha->empty());
            std::vector<vec2> uv{{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}};
            float sf = tex.getSolidFraction(uv);
            DOCTEST_CHECK(sf == doctest::Approx(1.f));
        }

        SUBCASE("Geometry dirty flags") {
            Context ctx;
            uint p = ctx.addPatch();
            DOCTEST_CHECK(ctx.isGeometryDirty());
            DOCTEST_CHECK(ctx.isPrimitiveDirty(p));

            ctx.markGeometryClean();
            DOCTEST_CHECK(!ctx.isGeometryDirty());
            DOCTEST_CHECK(!ctx.isPrimitiveDirty(p));

            ctx.markPrimitiveDirty(p);
            DOCTEST_CHECK(ctx.isGeometryDirty());
            DOCTEST_CHECK(ctx.isPrimitiveDirty(p));

            ctx.markPrimitiveClean(p);
            DOCTEST_CHECK(!ctx.isGeometryDirty());
            DOCTEST_CHECK(!ctx.isPrimitiveDirty(p));

            ctx.markGeometryDirty();
            DOCTEST_CHECK(ctx.isGeometryDirty());
        }

        SUBCASE("Date and Time Manipulation") {
            Context ctx;
            ctx.setDate(15, 7, 2025);
            Date d = ctx.getDate();
            DOCTEST_CHECK(d.day == 15);
            DOCTEST_CHECK(d.month == 7);
            DOCTEST_CHECK(d.year == 2025);
            DOCTEST_CHECK(strcmp(ctx.getMonthString(), "JUL") == 0);
            DOCTEST_CHECK(ctx.getJulianDate() == 196);

            ctx.setTime(45, 30, 10);
            Time t = ctx.getTime();
            DOCTEST_CHECK(t.hour == 10);
            DOCTEST_CHECK(t.minute == 30);
            DOCTEST_CHECK(t.second == 45);

            capture_cerr cerr_buffer;
            DOCTEST_CHECK_THROWS(ctx.setDate(32, 1, 2025));
            DOCTEST_CHECK_THROWS(ctx.setTime(60, 0, 0));
        }

        SUBCASE("Location Manipulation") {
            Context ctx;
            Location loc = {40.7128, -74.0060, 10.0};
            ctx.setLocation(loc);
            Location l = ctx.getLocation();
            DOCTEST_CHECK(l.latitude_deg == doctest::Approx(40.7128));
            DOCTEST_CHECK(l.longitude_deg == doctest::Approx(-74.0060));
            DOCTEST_CHECK(l.UTC_offset == doctest::Approx(10.0));
        }

        SUBCASE("primitive orientation and transforms") {
            Context ctx;
            uint id = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
            ctx.markGeometryClean();

            vec3 n = ctx.getPrimitiveNormal(id);
            DOCTEST_CHECK(n == vec3(0.f, 0.f, 1.f));

            ctx.setPrimitiveElevation(id, make_vec3(0, 0, 0), 0.f);
            n = ctx.getPrimitiveNormal(id);
            DOCTEST_CHECK(n.x == doctest::Approx(0.f).epsilon(errtol));
            DOCTEST_CHECK(n.y == doctest::Approx(1.f).epsilon(errtol));
            DOCTEST_CHECK(n.z == doctest::Approx(0.f).epsilon(errtol));

            ctx.setPrimitiveAzimuth(id, make_vec3(0, 0, 0), 0.5f * PI_F);
            n = ctx.getPrimitiveNormal(id);
            DOCTEST_CHECK(n.x == doctest::Approx(1.f).epsilon(errtol));
            DOCTEST_CHECK(n.y == doctest::Approx(0.f).epsilon(errtol));

            ctx.setPrimitiveNormal(id, make_vec3(0, 0, 0), make_vec3(0, 0, 1));
            n = ctx.getPrimitiveNormal(id);
            DOCTEST_CHECK(n.z == doctest::Approx(1.f).epsilon(errtol));

            float M[16];
            makeTranslationMatrix(make_vec3(1.f, 2.f, 3.f), M);
            ctx.setPrimitiveTransformationMatrix(id, M);
            float out[16];
            ctx.getPrimitiveTransformationMatrix(id, out);
            for (int i = 0; i < 16; ++i) {
                DOCTEST_CHECK(out[i] == doctest::Approx(M[i]));
            }
            DOCTEST_CHECK(ctx.isPrimitiveDirty(id));
        }
    }

    TEST_CASE("Primitive Management: Creation, Properties, and Operations") {
        SUBCASE("addPatch") {
            vec3 center, center_r;
            vec2 size, size_r;
            std::vector<vec3> vertices, vertices_r;
            SphericalCoord rotation, rotation_r;
            vec3 normal, normal_r;
            RGBcolor color, color_r;
            uint UUID;
            std::vector<uint> UUIDs;
            PrimitiveType type;
            float area_r;
            uint objID;

            Context context_test;

            // uint addPatch( const vec3& center, const vec2& size );
            center = make_vec3(1, 2, 3);
            size = make_vec2(1, 2);
            vertices.resize(4);
            vertices.at(0) = center + make_vec3(-0.5f * size.x, -0.5f * size.y, 0.f);
            vertices.at(1) = center + make_vec3(0.5f * size.x, -0.5f * size.y, 0.f);
            vertices.at(2) = center + make_vec3(0.5f * size.x, 0.5f * size.y, 0.f);
            vertices.at(3) = center + make_vec3(-0.5f * size.x, 0.5f * size.y, 0.f);

            DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size));
            DOCTEST_CHECK_NOTHROW(type = context_test.getPrimitiveType(UUID));
            DOCTEST_CHECK_NOTHROW(center_r = context_test.getPatchCenter(UUID));
            DOCTEST_CHECK_NOTHROW(size_r = context_test.getPatchSize(UUID));
            DOCTEST_CHECK_NOTHROW(normal_r = context_test.getPrimitiveNormal(UUID));
            DOCTEST_CHECK_NOTHROW(vertices_r = context_test.getPrimitiveVertices(UUID));
            DOCTEST_CHECK_NOTHROW(area_r = context_test.getPrimitiveArea(UUID));
            DOCTEST_CHECK_NOTHROW(color_r = context_test.getPrimitiveColor(UUID));

            DOCTEST_CHECK(type == PRIMITIVE_TYPE_PATCH);
            DOCTEST_CHECK(center_r.x == center.x);
            DOCTEST_CHECK(center_r.y == center.y);
            DOCTEST_CHECK(center_r.z == center.z);
            DOCTEST_CHECK(size_r.x == size.x);
            DOCTEST_CHECK(size_r.y == size.y);
            DOCTEST_CHECK(normal_r.x == 0.f);
            DOCTEST_CHECK(normal_r.y == 0.f);
            DOCTEST_CHECK(normal_r.z == 1.f);
            DOCTEST_CHECK(vertices_r.size() == 4);
            DOCTEST_CHECK(vertices_r.at(0).x == vertices.at(0).x);
            DOCTEST_CHECK(vertices_r.at(0).y == vertices.at(0).y);
            DOCTEST_CHECK(vertices_r.at(0).z == vertices.at(0).z);
            DOCTEST_CHECK(vertices_r.at(1).x == vertices.at(1).x);
            DOCTEST_CHECK(vertices_r.at(1).y == vertices.at(1).y);
            DOCTEST_CHECK(vertices_r.at(1).z == vertices.at(1).z);
            DOCTEST_CHECK(vertices_r.at(2).x == vertices.at(2).x);
            DOCTEST_CHECK(vertices_r.at(2).y == vertices.at(2).y);
            DOCTEST_CHECK(vertices_r.at(2).z == vertices.at(2).z);
            DOCTEST_CHECK(vertices_r.at(3).x == vertices.at(3).x);
            DOCTEST_CHECK(vertices_r.at(3).y == vertices.at(3).y);
            DOCTEST_CHECK(vertices_r.at(3).z == vertices.at(3).z);
            CHECK(area_r == doctest::Approx(size.x * size.y).epsilon(errtol));
            DOCTEST_CHECK(color_r.r == 0.f);
            DOCTEST_CHECK(color_r.g == 0.f);
            DOCTEST_CHECK(color_r.b == 0.f);
            DOCTEST_CHECK(context_test.getPrimitiveTextureFile(UUID).empty());
        }
        SUBCASE("rotated patch") {
            Context context_test;

            vec3 center = make_vec3(1, 2, 3);
            vec2 size = make_vec2(1, 2);
            SphericalCoord rotation = make_SphericalCoord(1.f, 0.15f * PI_F, 0.5f * PI_F);
            rotation.azimuth = 0.5f * PI_F;

            uint UUID;
            DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size, rotation));

            vec3 normal_r;
            DOCTEST_CHECK_NOTHROW(normal_r = context_test.getPrimitiveNormal(UUID));

            SphericalCoord rotation_r;
            DOCTEST_CHECK_NOTHROW(rotation_r = make_SphericalCoord(0.5f * PI_F - asinf(normal_r.z), atan2f(normal_r.x, normal_r.y)));

            DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUID));

            DOCTEST_CHECK(rotation_r.elevation == doctest::Approx(rotation.elevation).epsilon(errtol));
            DOCTEST_CHECK(rotation_r.azimuth == doctest::Approx(rotation.azimuth).epsilon(errtol));
        }
        SUBCASE("addTriangle") {
            Context context_test;

            vec3 v0, v0_r;
            vec3 v1, v1_r;
            vec3 v2, v2_r;
            uint UUID;

            // uint addTriangle( const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color );
            v0 = make_vec3(1, 2, 3);
            v1 = make_vec3(2, 4, 6);
            v2 = make_vec3(3, 6, 5);
            std::vector<vec3> vertices{v0, v1, v2};
            RGBcolor color = RGB::red;

            DOCTEST_CHECK_NOTHROW(UUID = context_test.addTriangle(v0, v1, v2, color));
            DOCTEST_CHECK(context_test.getPrimitiveType(UUID) == PRIMITIVE_TYPE_TRIANGLE);

            vec3 normal = normalize(cross(v1 - v0, v2 - v1));
            vec3 normal_r = context_test.getPrimitiveNormal(UUID);
            DOCTEST_CHECK(normal_r.x == doctest::Approx(normal.x).epsilon(errtol));
            DOCTEST_CHECK(normal_r.y == doctest::Approx(normal.y).epsilon(errtol));
            DOCTEST_CHECK(normal_r.z == doctest::Approx(normal.z).epsilon(errtol));

            std::vector<vec3> vertices_r;
            DOCTEST_CHECK_NOTHROW(vertices_r = context_test.getPrimitiveVertices(UUID));
            DOCTEST_CHECK(vertices_r.size() == 3);
            DOCTEST_CHECK(vertices_r.at(0).x == v0.x);
            DOCTEST_CHECK(vertices_r.at(0).y == v0.y);
            DOCTEST_CHECK(vertices_r.at(0).z == v0.z);

            RGBcolor color_r;
            DOCTEST_CHECK_NOTHROW(color_r = context_test.getPrimitiveColor(UUID));
            DOCTEST_CHECK(color_r.r == color.r);
            DOCTEST_CHECK(color_r.g == color.g);
            DOCTEST_CHECK(color_r.b == color.b);
            DOCTEST_CHECK(context_test.getPrimitiveTextureFile(UUID).empty());

            float a = (v1 - v0).magnitude();
            float b = (v2 - v0).magnitude();
            float c = (v2 - v1).magnitude();
            float s = 0.5f * (a + b + c);
            float area = sqrtf(s * (s - a) * (s - b) * (s - c));
            float area_r;
            DOCTEST_CHECK_NOTHROW(area_r = context_test.getPrimitiveArea(UUID));
            DOCTEST_CHECK(area_r == doctest::Approx(area).epsilon(errtol));
        }
        SUBCASE("copyPrimitive (patch)") {
            Context context_test;
            uint UUID, UUID_cpy;

            std::vector<float> cpdata{5.2f, 2.5f, 3.1f};

            vec3 center = make_vec3(1, 2, 3);
            vec2 size = make_vec2(1, 2);

            DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size));

            DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "somedata", cpdata));

            DOCTEST_CHECK_NOTHROW(UUID_cpy = context_test.copyPrimitive(UUID));

            vec3 center_cpy;
            DOCTEST_CHECK_NOTHROW(center_cpy = context_test.getPatchCenter(UUID_cpy));
            vec2 size_cpy;
            DOCTEST_CHECK_NOTHROW(size_cpy = context_test.getPatchSize(UUID_cpy));

            DOCTEST_CHECK(UUID_cpy == 1);
            DOCTEST_CHECK(center_cpy.x == center.x);
            DOCTEST_CHECK(center_cpy.y == center.y);
            DOCTEST_CHECK(center_cpy.z == center.z);
            DOCTEST_CHECK(size_cpy.x == size.x);
            DOCTEST_CHECK(size_cpy.y == size.y);

            std::vector<float> cpdata_copy;
            context_test.getPrimitiveData(UUID_cpy, "somedata", cpdata_copy);

            DOCTEST_CHECK(cpdata.size() == cpdata_copy.size());
            for (uint i = 0; i < cpdata.size(); i++) {
                DOCTEST_CHECK(cpdata.at(i) == cpdata_copy.at(i));
            }

            // translate the copied patch
            vec3 shift = make_vec3(5.f, 4.f, 3.f);
            DOCTEST_CHECK_NOTHROW(context_test.translatePrimitive(UUID_cpy, shift));
            DOCTEST_CHECK_NOTHROW(center_cpy = context_test.getPatchCenter(UUID_cpy));
            vec3 center_r;
            DOCTEST_CHECK_NOTHROW(center_r = context_test.getPatchCenter(UUID));

            DOCTEST_CHECK(center_cpy.x == doctest::Approx(center.x + shift.x).epsilon(errtol));
            DOCTEST_CHECK(center_cpy.y == doctest::Approx(center.y + shift.y).epsilon(errtol));
            DOCTEST_CHECK(center_cpy.z == doctest::Approx(center.z + shift.z).epsilon(errtol));
            DOCTEST_CHECK(center_r.x == center.x);
            DOCTEST_CHECK(center_r.y == center.y);
            DOCTEST_CHECK(center_r.z == center.z);
        }
        SUBCASE("copyPrimitive (triangle)") {
            Context context_test;

            vec3 v0 = make_vec3(0, 0, 0);
            vec3 v1 = make_vec3(1, 0, 0);
            vec3 v2 = make_vec3(0, 1, 0);
            uint UUID, UUID_cpy;

            DOCTEST_CHECK_NOTHROW(UUID = context_test.addTriangle(v0, v1, v2, RGB::blue));
            DOCTEST_CHECK_NOTHROW(UUID_cpy = context_test.copyPrimitive(UUID));

            std::vector<vec3> verts_org, verts_cpy;
            DOCTEST_CHECK_NOTHROW(verts_org = context_test.getPrimitiveVertices(UUID));
            DOCTEST_CHECK_NOTHROW(verts_cpy = context_test.getPrimitiveVertices(UUID_cpy));
            DOCTEST_CHECK(verts_org == verts_cpy);

            vec3 shift = make_vec3(5.f, 4.f, 3.f);
            DOCTEST_CHECK_NOTHROW(context_test.translatePrimitive(UUID_cpy, shift));
            DOCTEST_CHECK_NOTHROW(verts_cpy = context_test.getPrimitiveVertices(UUID_cpy));
            DOCTEST_CHECK(verts_cpy.at(0) == verts_org.at(0) + shift);
            DOCTEST_CHECK(verts_cpy.at(1) == verts_org.at(1) + shift);
            DOCTEST_CHECK(verts_cpy.at(2) == verts_org.at(2) + shift);

            DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUID));
            DOCTEST_CHECK(!context_test.doesPrimitiveExist(UUID));
        }
        SUBCASE("deletePrimitive") {
            Context context_test;
            uint UUID;
            vec3 center = make_vec3(1, 2, 3);
            vec2 size = make_vec2(1, 2);

            DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size));

            DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUID));

            uint primitive_count;
            DOCTEST_CHECK_NOTHROW(primitive_count = context_test.getPrimitiveCount(UUID));
            DOCTEST_CHECK(primitive_count == 0);
            DOCTEST_CHECK(!context_test.doesPrimitiveExist(UUID));
        }
        SUBCASE("primitive bounding box") {
            Context context_test;
            std::vector<uint> UUIDs;
            UUIDs.push_back(context_test.addPatch(make_vec3(-1, 0, 0), make_vec2(0.5, 0.5)));
            UUIDs.push_back(context_test.addPatch(make_vec3(1, 0, 0), make_vec2(0.5, 0.5)));

            vec3 bmin, bmax;
            DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveBoundingBox(UUIDs, bmin, bmax));
            DOCTEST_CHECK(bmin == make_vec3(-1.25f, -0.25f, 0.f));
            DOCTEST_CHECK(bmax == make_vec3(1.25f, 0.25f, 0.f));
        }
        SUBCASE("primitive scale and data") {
            Context context_test;
            vec2 sz_0 = make_vec2(0.5f, 3.f);
            float area0 = sz_0.x * sz_0.y;
            float scale = 2.6f;
            uint UUID = context_test.addPatch(make_vec3(0, 0, 0), sz_0);
            context_test.scalePrimitive(UUID, make_vec3(scale, scale, scale));
            float area1 = context_test.getPrimitiveArea(UUID);
            DOCTEST_CHECK(area1 == doctest::Approx(scale * scale * area0).epsilon(1e-5));

            float data = 5.f;
            context_test.setPrimitiveData(UUID, "some_data", data);
            DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUID, "some_data"));
            float data_r;
            context_test.getPrimitiveData(UUID, "some_data", data_r);
            DOCTEST_CHECK(data_r == data);

            std::vector<float> vec = {0, 1, 2, 3, 4};
            context_test.setPrimitiveData(UUID, "vec_data", vec);
            std::vector<float> vec_r;
            context_test.getPrimitiveData(UUID, "vec_data", vec_r);
            DOCTEST_CHECK(vec_r == vec);

            std::vector<uint> UUIDs_filter;
            std::vector<uint> UUIDs_multi;
            for (uint i = 0; i < 4; i++) {
                UUIDs_multi.push_back(context_test.addPatch());
            }
            context_test.setPrimitiveData(UUIDs_multi[0], "val", 4.f);
            context_test.setPrimitiveData(UUIDs_multi[0], "str", "cat");
            context_test.setPrimitiveData(UUIDs_multi[1], "val", 3.f);
            context_test.setPrimitiveData(UUIDs_multi[1], "str", "cat");
            context_test.setPrimitiveData(UUIDs_multi[2], "val", 2.f);
            context_test.setPrimitiveData(UUIDs_multi[2], "str", "dog");
            context_test.setPrimitiveData(UUIDs_multi[3], "val", 1.f);
            context_test.setPrimitiveData(UUIDs_multi[3], "str", "dog");

            UUIDs_filter = context_test.filterPrimitivesByData(UUIDs_multi, "val", 2.f, "<=");
            DOCTEST_CHECK(UUIDs_filter.size() == 2);
            DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[2]) != UUIDs_filter.end());
            DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[3]) != UUIDs_filter.end());

            UUIDs_filter = context_test.filterPrimitivesByData(UUIDs_multi, "str", "cat");
            DOCTEST_CHECK(UUIDs_filter.size() == 2);
            DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[0]) != UUIDs_filter.end());
            DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[1]) != UUIDs_filter.end());
        }
        SUBCASE("texture uv and solid fraction") {
            Context context_test;

            vec2 sizep = make_vec2(2, 3);
            const char *texture = "lib/images/disk_texture.png";
            vec2 uv0 = make_vec2(0, 0);
            vec2 uv1 = make_vec2(1, 0);
            vec2 uv2 = make_vec2(1, 1);
            vec2 uv3 = make_vec2(0, 1);
            uint UUIDp = context_test.addPatch(make_vec3(2, 3, 4), sizep, nullrotation, texture, 0.5f * (uv0 + uv2), uv2 - uv0);
            DOCTEST_CHECK(!context_test.getPrimitiveTextureFile(UUIDp).empty());
            float Ap = context_test.getPrimitiveArea(UUIDp);
            DOCTEST_CHECK(Ap == doctest::Approx(0.25f * PI_F * sizep.x * sizep.y).epsilon(0.01));
            std::vector<vec2> uv = context_test.getPrimitiveTextureUV(UUIDp);
            DOCTEST_CHECK(uv.size() == 4);
            DOCTEST_CHECK(uv.at(0) == uv0);
            DOCTEST_CHECK(uv.at(1) == uv1);
            DOCTEST_CHECK(uv.at(2) == uv2);
            DOCTEST_CHECK(uv.at(3) == uv3);

            uint UUIDt = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), "lib/images/diamond_texture.png", make_vec2(0, 0), make_vec2(1, 0), make_vec2(1, 1));
            float solid_fraction = context_test.getPrimitiveSolidFraction(UUIDt);
            DOCTEST_CHECK(solid_fraction == doctest::Approx(0.5f).epsilon(errtol));
        }
    }

    TEST_CASE("Object Management") {
        SUBCASE("addBoxObject") {
            Context context_test;

            vec3 center = make_vec3(1, 2, 3);
            vec3 size = make_vec3(3, 2, 1);
            int3 subdiv(1, 1, 1);

            uint objID;
            DOCTEST_CHECK_NOTHROW(objID = context_test.addBoxObject(center, size, subdiv));
            std::vector<uint> UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();

            DOCTEST_CHECK(UUIDs.size() == 6);
            vec3 normal_r = context_test.getPrimitiveNormal(UUIDs.at(0));
            DOCTEST_CHECK(doctest::Approx(normal_r.magnitude()).epsilon(errtol) == 1.f);
            normal_r = context_test.getPrimitiveNormal(UUIDs.at(2));
            DOCTEST_CHECK(doctest::Approx(normal_r.magnitude()).epsilon(errtol) == 1.f);

            vec2 size_r = context_test.getPatchSize(UUIDs.at(0));
            DOCTEST_CHECK(size_r.x == doctest::Approx(size.x).epsilon(errtol));
            DOCTEST_CHECK(size_r.y == doctest::Approx(size.z).epsilon(errtol));

            size_r = context_test.getPatchSize(UUIDs.at(2));
            DOCTEST_CHECK(size_r.x == doctest::Approx(size.y).epsilon(errtol));
            DOCTEST_CHECK(size_r.y == doctest::Approx(size.z).epsilon(errtol));

            float volume = context_test.getBoxObjectVolume(objID);
            DOCTEST_CHECK(volume == doctest::Approx(size.x * size.y * size.z).epsilon(errtol));
        }
        SUBCASE("addTileObject rotated") {
            Context context_test;

            vec3 center = make_vec3(1, 2, 3);
            vec2 size = make_vec2(3, 2);
            int2 subdiv(3, 3);
            SphericalCoord rotation = make_SphericalCoord(0.25f * PI_F, 1.4f * PI_F);
            uint objID = context_test.addTileObject(center, size, rotation, subdiv);

            std::vector<uint> UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();
            for (uint UUIDp: UUIDs) {
                vec3 n = context_test.getPrimitiveNormal(UUIDp);
                SphericalCoord rot = cart2sphere(n);
                DOCTEST_CHECK(rot.zenith == doctest::Approx(rotation.zenith).epsilon(errtol));
                DOCTEST_CHECK(rot.azimuth == doctest::Approx(rotation.azimuth).epsilon(errtol));
            }
        }
        SUBCASE("textured tile area") {
            Context context_test;

            vec3 center = make_vec3(1, 2, 3);
            vec2 size = make_vec2(3, 2);
            int2 subdiv = make_int2(5, 5);
            SphericalCoord rotation = make_SphericalCoord(0.1f * PI_F, 2.4f * PI_F);

            uint objID = context_test.addTileObject(center, size, rotation, subdiv, "lib/images/disk_texture.png");
            std::vector<uint> UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();
            float area_sum = 0.f;
            for (uint UUID: UUIDs) {
                area_sum += context_test.getPrimitiveArea(UUID);
            }
            float area_exact = 0.25f * PI_F * size.x * size.y;
            DOCTEST_CHECK(area_sum == doctest::Approx(area_exact).epsilon(5e-3));
        }
        SUBCASE("cone object transforms") {
            Context context_test;
            float r0 = 0.5f, r1 = 1.f, len = 2.f;
            vec3 node0 = make_vec3(0, 0, 0);
            vec3 node1 = make_vec3(0, 0, len);
            uint cone = context_test.addConeObject(50, node0, node1, r0, r1);
            context_test.getConeObjectPointer(cone)->translate(make_vec3(1, 1, 1));
            std::vector<vec3> nodes = context_test.getConeObjectPointer(cone)->getNodeCoordinates();
            DOCTEST_CHECK(nodes.at(0) == make_vec3(1, 1, 1));
            DOCTEST_CHECK(nodes.at(1) == make_vec3(1, 1, 1 + len));
            vec3 axis = cross(make_vec3(0, 0, 1), make_vec3(1, 0, 0));
            float ang = acos_safe(make_vec3(1, 0, 0) * make_vec3(0, 0, 1));
            context_test.getConeObjectPointer(cone)->translate(-nodes.at(0));
            context_test.getConeObjectPointer(cone)->rotate(ang, axis);
            context_test.getConeObjectPointer(cone)->translate(nodes.at(0));
            nodes = context_test.getConeObjectPointer(cone)->getNodeCoordinates();
            DOCTEST_CHECK(nodes.at(1).x == doctest::Approx(nodes.at(0).x + len).epsilon(errtol));
            context_test.getConeObjectPointer(cone)->scaleLength(2.0);
            nodes = context_test.getConeObjectPointer(cone)->getNodeCoordinates();
            DOCTEST_CHECK(nodes.at(1).x == doctest::Approx(nodes.at(0).x + 2 * len).epsilon(errtol));
            context_test.getConeObjectPointer(cone)->scaleGirth(2.0);
            std::vector<float> radii = context_test.getConeObjectPointer(cone)->getNodeRadii();
            DOCTEST_CHECK(radii.at(0) == doctest::Approx(2 * r0).epsilon(errtol));
            DOCTEST_CHECK(radii.at(1) == doctest::Approx(2 * r1).epsilon(errtol));
        }
    }

    TEST_CASE("Data Management") {
        SUBCASE("global and object data") {
            Context context_test;
            float gdata = 5.f;
            context_test.setGlobalData("some_data", gdata);
            float gdata_r;
            DOCTEST_CHECK(context_test.doesGlobalDataExist("some_data"));
            context_test.getGlobalData("some_data", gdata_r);
            DOCTEST_CHECK(gdata_r == gdata);

            std::vector<float> gvec{0, 1, 2, 3, 4};
            context_test.setGlobalData("vec", gvec);
            std::vector<float> gvec_r;
            context_test.getGlobalData("vec", gvec_r);
            DOCTEST_CHECK(gvec_r == gvec);

            uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(3, 1), nullrotation, make_int2(3, 3));
            float objdata = 7.f;
            context_test.setObjectData(objID, "obj", objdata);
            float objdata_r;
            context_test.getObjectData(objID, "obj", objdata_r);
            DOCTEST_CHECK(objdata_r == objdata);
        }
        SUBCASE("timeseries") {
            Context ctx;
            Date date = make_Date(12, 3, 2010);
            ctx.setDate(date);
            Time time0 = make_Time(13, 15, 39);
            ctx.setTime(time0);
            Time time1 = make_Time(time0.hour, 49, 14);
            ctx.addTimeseriesData("ts", 302.3f, date, time0);
            ctx.addTimeseriesData("ts", 305.3f, date, time1);
            ctx.setCurrentTimeseriesPoint("ts", 0);
            DOCTEST_CHECK(ctx.getTimeseriesLength("ts") == 2);
            DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 0) == doctest::Approx(302.3f));
            DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 1) == doctest::Approx(305.3f));
            float val = ctx.queryTimeseriesData("ts", date, time1);
            DOCTEST_CHECK(val == doctest::Approx(305.3f));
        }
        SUBCASE("Context_data coverage") {
            SUBCASE("Primitive data") {
                capture_cerr cerr_buffer;

                Context ctx;
                uint p = ctx.addPatch();
                ctx.setPrimitiveData(p, "test_int", 5);
                ctx.setPrimitiveData(p, "test_float", 3.14f);

                // getPrimitiveDataType
                DOCTEST_CHECK(ctx.getPrimitiveDataType(p, "test_int") == HELIOS_TYPE_INT);
                DOCTEST_CHECK(ctx.getPrimitiveDataType(p, "test_float") == HELIOS_TYPE_FLOAT);
#ifdef HELIOS_DEBUG
                DOCTEST_CHECK_THROWS_AS(ctx.getPrimitiveDataType(p, "non_existent"), std::runtime_error);
                DOCTEST_CHECK(cerr_buffer.has_output());
#endif

                // getPrimitiveDataSize
                DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "test_int") == 1);
                std::vector<float> float_vec = {1.f, 2.f, 3.f};
                ctx.setPrimitiveData(p, "test_float_vec", float_vec);
                DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "test_float_vec") == 3);
#ifdef HELIOS_DEBUG
                DOCTEST_CHECK_THROWS_AS(ctx.getPrimitiveDataSize(p, "non_existent"), std::runtime_error);
#endif

                // clearPrimitiveData
                ctx.clearPrimitiveData(p, "test_int");
                DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "test_int"));
                ctx.clearPrimitiveData(p, "non_existent"); // should not throw

                // listPrimitiveData
                std::vector<std::string> data_labels = ctx.listPrimitiveData(p);
                DOCTEST_CHECK(std::find(data_labels.begin(), data_labels.end(), "test_float") != data_labels.end());
                DOCTEST_CHECK(std::find(data_labels.begin(), data_labels.end(), "test_float_vec") != data_labels.end());
            }

            SUBCASE("Context primitive data management") {
                Context ctx;
                uint p1 = ctx.addPatch();
                uint p2 = ctx.addPatch();
                ctx.setPrimitiveData(p1, "my_data", 10);

                // copyPrimitiveData
                ctx.copyPrimitiveData(p1, p2);
                DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p2, "my_data"));
                int val;
                ctx.getPrimitiveData(p2, "my_data", val);
                DOCTEST_CHECK(val == 10);

                // renamePrimitiveData
                ctx.renamePrimitiveData(p1, "my_data", "new_data_name");
                DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p1, "my_data"));
                DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p1, "new_data_name"));

                // duplicatePrimitiveData
                ctx.duplicatePrimitiveData(p2, "my_data", "my_data_copy");
                DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p2, "my_data_copy"));
                ctx.getPrimitiveData(p2, "my_data_copy", val);
                DOCTEST_CHECK(val == 10);

                // duplicatePrimitiveData (all primitives)
                ctx.setPrimitiveData(p1, "global_copy_test", 5.5f);
                ctx.duplicatePrimitiveData("global_copy_test", "global_copy_test_new");
                DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p1, "global_copy_test_new"));
                DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p2, "global_copy_test_new")); // p2 doesn't have original

            }

            SUBCASE("Context primitive data calculations") {
                Context ctx;
                std::vector<uint> uuids;
                for (int i = 0; i < 5; ++i) {
                    uint p = ctx.addPatch(make_vec3(0,0,0), make_vec2(1,1));
                    ctx.setPrimitiveData(p, "float_val", (float)i);
                    ctx.setPrimitiveData(p, "double_val", (double)i);
                    ctx.setPrimitiveData(p, "vec2_val", make_vec2((float)i, (float)i));
                    uuids.push_back(p);
                }

                // calculatePrimitiveDataMean
                float float_mean;
                ctx.calculatePrimitiveDataMean(uuids, "float_val", float_mean);
                DOCTEST_CHECK(float_mean == doctest::Approx(2.0f));
                double double_mean;
                ctx.calculatePrimitiveDataMean(uuids, "double_val", double_mean);
                DOCTEST_CHECK(double_mean == doctest::Approx(2.0));
                vec2 vec2_mean;
                ctx.calculatePrimitiveDataMean(uuids, "vec2_val", vec2_mean);
                DOCTEST_CHECK(vec2_mean.x == doctest::Approx(2.0f));

                // calculatePrimitiveDataAreaWeightedMean
                float awt_mean_f;
                ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "float_val", awt_mean_f);
                DOCTEST_CHECK(awt_mean_f == doctest::Approx(2.0f)); // Area is 1 for all

                // calculatePrimitiveDataSum
                float float_sum;
                ctx.calculatePrimitiveDataSum(uuids, "float_val", float_sum);
                DOCTEST_CHECK(float_sum == doctest::Approx(10.0f));

                // calculatePrimitiveDataAreaWeightedSum
                float awt_sum_f;
                ctx.calculatePrimitiveDataAreaWeightedSum(uuids, "float_val", awt_sum_f);
                DOCTEST_CHECK(awt_sum_f == doctest::Approx(10.0f));

                // scalePrimitiveData
                ctx.scalePrimitiveData(uuids, "float_val", 2.0f);
                ctx.getPrimitiveData(uuids[2], "float_val", float_mean);
                DOCTEST_CHECK(float_mean == doctest::Approx(4.0f));
                ctx.scalePrimitiveData("double_val", 0.5f);
                ctx.getPrimitiveData(uuids[4], "double_val", double_mean);
                DOCTEST_CHECK(double_mean == doctest::Approx(2.0));

                // incrementPrimitiveData
                ctx.setPrimitiveData(uuids, "int_val", 10);
                ctx.incrementPrimitiveData(uuids, "int_val", 5);
                int int_val;
                ctx.getPrimitiveData(uuids[0], "int_val", int_val);
                DOCTEST_CHECK(int_val == 15);
                capture_cerr cerr_buffer;
                ctx.incrementPrimitiveData(uuids, "float_val", 1); // Wrong type, should warn
                DOCTEST_CHECK(cerr_buffer.has_output());

            }

            SUBCASE("Context primitive data aggregation and filtering") {
                Context ctx;
                std::vector<uint> uuids;
                for (int i = 0; i < 3; ++i) {
                    uint p = ctx.addPatch();
                    ctx.setPrimitiveData(p, "d1", (float)i);
                    ctx.setPrimitiveData(p, "d2", (float)i * 2.0f);
                    ctx.setPrimitiveData(p, "d3", (float)i * 3.0f);
                    ctx.setPrimitiveData(p, "filter_me", i);
                    uuids.push_back(p);
                }

                // aggregatePrimitiveDataSum
                std::vector<std::string> labels = {"d1", "d2", "d3"};
                ctx.aggregatePrimitiveDataSum(uuids, labels, "sum_data");
                float sum_val;
                ctx.getPrimitiveData(uuids[1], "sum_data", sum_val);
                DOCTEST_CHECK(sum_val == doctest::Approx(1.f + 2.f + 3.f));

                // aggregatePrimitiveDataProduct
                ctx.aggregatePrimitiveDataProduct(uuids, labels, "prod_data");
                float prod_val;
                ctx.getPrimitiveData(uuids[2], "prod_data", prod_val);
                DOCTEST_CHECK(prod_val == doctest::Approx(2.f * 4.f * 6.f));

                // filterPrimitivesByData
                std::vector<uint> filtered = ctx.filterPrimitivesByData(uuids, "filter_me", 1, ">=");
                DOCTEST_CHECK(filtered.size() == 2);
                filtered = ctx.filterPrimitivesByData(uuids, "filter_me", 1, "==");
                DOCTEST_CHECK(filtered.size() == 1);
                DOCTEST_CHECK(filtered[0] == uuids[1]);
                DOCTEST_CHECK_THROWS_AS(filtered = ctx.filterPrimitivesByData(uuids, "filter_me", 1, "!!"), std::runtime_error);
            }

            SUBCASE("Object data") {
                Context ctx;
                uint o = ctx.addTileObject( nullorigin, make_vec2(1,1), nullrotation, make_int2(2,2));
                ctx.setObjectData(o, "test_int", 5);
                ctx.setObjectData(o, "test_float", 3.14f);

                // getObjectDataType
                DOCTEST_CHECK(ctx.getObjectDataType(o, "test_int") == HELIOS_TYPE_INT);
#ifdef HELIOS_DEBUG
                DOCTEST_CHECK_THROWS_AS(ctx.getObjectDataType(o, "non_existent"), std::runtime_error);
#endif

                // getObjectDataSize
                DOCTEST_CHECK(ctx.getObjectDataSize(o, "test_int") == 1);

                // clearObjectData
                ctx.clearObjectData(o, "test_int");
                DOCTEST_CHECK(!ctx.doesObjectDataExist(o, "test_int"));

                // listObjectData
                std::vector<std::string> data_labels = ctx.listObjectData(o);
                DOCTEST_CHECK(std::find(data_labels.begin(), data_labels.end(), "test_float") != data_labels.end());
            }

            SUBCASE("Context object data management") {
                Context ctx;
                uint o1 = ctx.addTileObject(nullorigin, make_vec2(1,1), nullrotation, make_int2(2,2));
                uint o2 = ctx.addTileObject(nullorigin, make_vec2(1,1), nullrotation, make_int2(2,2));
                ctx.setObjectData(o1, "my_data", 10);

                // copyObjectData
                ctx.copyObjectData(o1, o2);
                DOCTEST_CHECK(ctx.doesObjectDataExist(o2, "my_data"));

                // renameObjectData
                ctx.renameObjectData(o1, "my_data", "new_name");
                DOCTEST_CHECK(!ctx.doesObjectDataExist(o1, "my_data"));
                DOCTEST_CHECK(ctx.doesObjectDataExist(o1, "new_name"));

                // duplicateObjectData
                ctx.duplicateObjectData(o2, "my_data", "my_data_copy");
                DOCTEST_CHECK(ctx.doesObjectDataExist(o2, "my_data_copy"));
            }

            SUBCASE("Global data") {
                Context ctx;
                ctx.setGlobalData("g_int", 5);
                ctx.setGlobalData("g_float", 3.14f);

                // getGlobalDataType/Size/Exists
                DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int"));
                DOCTEST_CHECK(ctx.getGlobalDataType("g_int") == HELIOS_TYPE_INT);
                DOCTEST_CHECK(ctx.getGlobalDataSize("g_int") == 1);

                // rename/duplicate/clear
                ctx.duplicateGlobalData("g_int", "g_int_copy");
                DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int_copy"));
                ctx.renameGlobalData("g_int", "g_int_new");
                DOCTEST_CHECK(!ctx.doesGlobalDataExist("g_int"));
                DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int_new"));
                ctx.clearGlobalData("g_int_new");
                DOCTEST_CHECK(!ctx.doesGlobalDataExist("g_int_new"));

                // listGlobalData
                std::vector<std::string> g_labels = ctx.listGlobalData();
                DOCTEST_CHECK(g_labels.size() > 0);

                // incrementGlobalData
                ctx.setGlobalData("inc_me", 10);
                ctx.incrementGlobalData("inc_me", 5);
                int val;
                ctx.getGlobalData("inc_me", val);
                DOCTEST_CHECK(val == 15);
                capture_cerr cerr_buffer;
                ctx.incrementGlobalData("g_float", 1); // Wrong type
                DOCTEST_CHECK(cerr_buffer.has_output());
            }
        }
    }

    TEST_CASE("Error Handling") {
        SUBCASE("Context error handling") {
            Context context_test;
            uint tri = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::green);
            capture_cerr cerr_buffer;
            vec3 center;
#ifdef HELIOS_DEBUG
            DOCTEST_CHECK_THROWS_AS(center = context_test.getPatchCenter(tri), std::runtime_error);
#endif

            uint vox = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
            std::vector<uint> vlist{vox};
            DOCTEST_CHECK_THROWS_AS(context_test.rotatePrimitive(vlist, PI_F / 4.f, "a"), std::runtime_error);
        }
    }
}


// =================================================================================
// Suite 6: Additional Coverage and Edge Cases
//
// This suite contains tests that provide additional coverage or test specific
// edge cases that don't fit neatly into the other categories. All tests are
// preserved from the original file.
// =================================================================================
TEST_SUITE("Additional Coverage and Edge Cases") {

    TEST_CASE("global.h additional coverage") {
        SUBCASE("string to vector conversions with invalid input") {
            capture_cerr cerr_buffer;
            vec2 result_vec2;
            vec3 result_vec3;
            vec4 result_vec4;
            int2 result_int2;
            int3 result_int3;
            int4 result_int4;
            DOCTEST_CHECK_THROWS(result_vec2 = string2vec2("1.5"));
            DOCTEST_CHECK_THROWS(result_vec3 = string2vec3("1.5 2.5"));
            DOCTEST_CHECK_THROWS(result_vec4 = string2vec4("1.5 2.5 3.5"));
            DOCTEST_CHECK_THROWS(result_int2 = string2int2("1"));
            DOCTEST_CHECK_THROWS(result_int3 = string2int3("1 2"));
            DOCTEST_CHECK_THROWS(result_int4 = string2int4("1 2 3"));
            DOCTEST_CHECK_THROWS(result_vec2 = string2vec2("1.5 abc"));
        }

        SUBCASE("separate_string_by_delimiter edge cases") {
            std::vector<std::string> result;
            DOCTEST_CHECK_NOTHROW(result = separate_string_by_delimiter("a,b,c", ";"));
            DOCTEST_CHECK(result.size() == 1);
            DOCTEST_CHECK(result[0] == "a,b,c");

            DOCTEST_CHECK_NOTHROW(result = separate_string_by_delimiter("a|b|c", "|"));
            DOCTEST_CHECK(result.size() == 3);
            DOCTEST_CHECK(result[0] == "a");
            DOCTEST_CHECK(result[1] == "b");
            DOCTEST_CHECK(result[2] == "c");

            DOCTEST_CHECK_NOTHROW(result = separate_string_by_delimiter("", ","));
            DOCTEST_CHECK(result.size()==1);

            capture_cerr cerr_buffer;
            DOCTEST_CHECK_THROWS(result = separate_string_by_delimiter("a,b,c", ""));

            DOCTEST_CHECK_NOTHROW(result = separate_string_by_delimiter(",", ","));
            DOCTEST_CHECK(result.size() == 2);
            if(result.size() == 2) {
                DOCTEST_CHECK(result[0] == "");
                DOCTEST_CHECK(result[1] == "");
            }
        }

        SUBCASE("File path parsing edge cases") {
            DOCTEST_CHECK( getFileExtension(".bashrc").empty() );
            DOCTEST_CHECK(getFileName(".bashrc") == ".bashrc");
            DOCTEST_CHECK(getFileStem(".bashrc") == ".bashrc");
    #ifndef _WIN32
            DOCTEST_CHECK(getFilePath("/path/to/file/", true) == "/path/to/file/");
            DOCTEST_CHECK(getFilePath("/path/to/file/", false) == "/path/to/file");
    #endif
            DOCTEST_CHECK(getFilePath("..", true).empty());
            DOCTEST_CHECK(getFileName("..") == "..");
            DOCTEST_CHECK(getFileStem("..") == "..");
        }

        SUBCASE("interp1 edge cases") {
            capture_cerr cerr_buffer;
            float result;
            std::vector<vec2> empty_points;
            DOCTEST_CHECK_THROWS(result = interp1(empty_points, 0.5f));

            std::vector<vec2> single_point = {{1, 5}};
            result = interp1(single_point, 0.5f);
            DOCTEST_CHECK(result == doctest::Approx(5.f));
            result = interp1(single_point, 2.f);
            DOCTEST_CHECK(result == doctest::Approx(5.f));
        }

        SUBCASE("XMLoad* for missing nodes") {
            pugi::xml_document doc;
            pugi::xml_node node = doc.append_child("node");
            // The XMLoad functions throw when the node is not found, so we check for that.
            capture_cerr cerr_buffer;
            DOCTEST_CHECK(XMLloadfloat(node, "non_existent")==99999);
            DOCTEST_CHECK(XMLloadint(node, "non_existent")==99999);
            DOCTEST_CHECK(XMLloadstring(node, "non_existent")=="99999");
            DOCTEST_CHECK(XMLloadvec2(node, "non_existent")==vec2(99999,99999));
            DOCTEST_CHECK(XMLloadvec3(node, "non_existent")==vec3(99999,99999,99999));
            DOCTEST_CHECK(XMLloadvec4(node, "non_existent")==vec4(99999,99999,99999,99999));
            DOCTEST_CHECK(XMLloadint2(node, "non_existent")==int2(99999,99999));
            DOCTEST_CHECK(XMLloadint3(node, "non_existent")==int3(99999,99999,99999));
            DOCTEST_CHECK(XMLloadint4(node, "non_existent")==int4(99999,99999,99999,99999));
            DOCTEST_CHECK(XMLloadrgb(node, "non_existent")==RGBcolor(99999,99999,99999));
            DOCTEST_CHECK(XMLloadrgba(node, "non_existent")==RGBAcolor(99999,99999,99999,99999));
        }

        SUBCASE("string2RGBcolor with invalid input") {
            capture_cerr cerr_buffer;
            RGBAcolor result;
            DOCTEST_CHECK_THROWS(result = string2RGBcolor("0.1 0.2"));
            DOCTEST_CHECK_THROWS(result = string2RGBcolor("0.1 0.2 0.3 0.4 0.5"));
            DOCTEST_CHECK_THROWS(result = string2RGBcolor("a b c"));
        }

        SUBCASE("lineIntersection with collinear and overlapping segments") {
            // Collinear, non-overlapping
            vec2 p1(0, 0), q1(1, 1);
            vec2 p2(2, 2), q2(3, 3);
            DOCTEST_CHECK(!lineIntersection(p1, q1, p2, q2));

            // Collinear, overlapping
            vec2 p3(0, 0), q3(2, 2);
            vec2 p4(1, 1), q4(3, 3);
            DOCTEST_CHECK(lineIntersection(p3, q3, p4, q4));

            // Collinear, one contains another
            vec2 p5(0,0), q5(3,3);
            vec2 p6(1,1), q6(2,2);
            DOCTEST_CHECK(lineIntersection(p5, q5, p6, q6));

            // Collinear, share an endpoint
            vec2 p7(0,0), q7(1,1);
            vec2 p8(1,1), q8(2,2);
            DOCTEST_CHECK(lineIntersection(p7, q7, p8, q8));
        }

        SUBCASE("pointInPolygon with concave polygon") {
            // Concave polygon (U-shaped)
            std::vector<vec2> concave_poly = {{0, 0}, {5, 0}, {5, 5}, {3, 3}, {2, 5}, {0, 5}};
            DOCTEST_CHECK(pointInPolygon(vec2(1, 1), concave_poly)); // Inside
            DOCTEST_CHECK(!pointInPolygon(vec2(4, 4), concave_poly)); // In concave part, but outside
            DOCTEST_CHECK(pointInPolygon(vec2(2.5, 4), concave_poly)); // Inside
        }

        SUBCASE("calculateTriangleArea with degenerate triangle") {
            vec3 v0(0, 0, 0), v1(1, 1, 1), v2(2, 2, 2); // Collinear points
            DOCTEST_CHECK(calculateTriangleArea(v0, v1, v2) == doctest::Approx(0.f));
        }

        SUBCASE("Vector statistics with single element") {
            std::vector<float> v = {5.f};
            DOCTEST_CHECK(stdev(v) == doctest::Approx(0.f));
        }

        SUBCASE("deblank and trim_whitespace edge cases") {
            DOCTEST_CHECK(deblank("") == "");
            DOCTEST_CHECK(deblank("   ") == "");
            DOCTEST_CHECK(trim_whitespace("") == "");
            DOCTEST_CHECK(trim_whitespace("   ") == "");
        }

        SUBCASE("parse functions with whitespace") {
            int i;
            DOCTEST_CHECK(parse_int("  123  ", i));
            DOCTEST_CHECK(i == 123);
            float f;
            DOCTEST_CHECK(parse_float("  1.23  ", f));
            DOCTEST_CHECK(f == doctest::Approx(1.23f));
        }
    }

    TEST_CASE("global.h additional coverage 2") {
        SUBCASE("spline_interp3 edge cases") {
            vec3 p0 = make_vec3(0, 0, 0);
            vec3 t0 = make_vec3(1, 0, 0);
            vec3 p1 = make_vec3(1, 1, 1);
            vec3 t1 = make_vec3(0, 1, 0);

            vec3 res_start = spline_interp3(0.0f, p0, t0, p1, t1);
            DOCTEST_CHECK(res_start.x == doctest::Approx(p0.x));
            DOCTEST_CHECK(res_start.y == doctest::Approx(p0.y));
            DOCTEST_CHECK(res_start.z == doctest::Approx(p0.z));

            vec3 res_end = spline_interp3(1.0f, p0, t0, p1, t1);
            DOCTEST_CHECK(res_end.x == doctest::Approx(p1.x));
            DOCTEST_CHECK(res_end.y == doctest::Approx(p1.y));
            DOCTEST_CHECK(res_end.z == doctest::Approx(p1.z));

            // Test clamping
            capture_cerr cerr_buffer;
            vec3 res_low = spline_interp3(-0.5f, p0, t0, p1, t1);
            DOCTEST_CHECK(res_low.x == doctest::Approx(p0.x));
            vec3 res_high = spline_interp3(1.5f, p0, t0, p1, t1);
            DOCTEST_CHECK(res_high.x == doctest::Approx(p1.x));
        }

        SUBCASE("XMLoad* for missing fields") {
            pugi::xml_document doc;
            pugi::xml_node node = doc.append_child("node");
            DOCTEST_CHECK(XMLloadfloat(node, "non_existent") == 99999);
            DOCTEST_CHECK(XMLloadint(node, "non_existent") == 99999);
            DOCTEST_CHECK(XMLloadstring(node, "non_existent") == "99999");
            DOCTEST_CHECK(XMLloadvec2(node, "non_existent") == make_vec2(99999, 99999));
            DOCTEST_CHECK(XMLloadvec3(node, "non_existent") == make_vec3(99999, 99999, 99999));
        }

        SUBCASE("parse_xml_tag_*") {
            pugi::xml_document doc;
            pugi::xml_node node = doc.append_child("tag");
            node.text().set(" 123 ");
            int i = parse_xml_tag_int(node, "tag", "test");
            DOCTEST_CHECK(i == 123);
            node.text().set(" 1.23 ");
            float f = parse_xml_tag_float(node, "tag", "test");
            DOCTEST_CHECK(f == doctest::Approx(1.23f));
            node.text().set(" 1.1 2.2 ");
            vec2 v2 = parse_xml_tag_vec2(node, "tag", "test");
            DOCTEST_CHECK(v2 == vec2(1.1f, 2.2f));
            node.text().set(" 1.1 2.2 3.3 ");
            vec3 v3 = parse_xml_tag_vec3(node, "tag", "test");
            DOCTEST_CHECK(v3 == vec3(1.1f, 2.2f, 3.3f));
            node.text().set(" hello ");
            std::string s = parse_xml_tag_string(node, "tag", "test");
            DOCTEST_CHECK(s == "hello");

            capture_cerr cerr_buffer;
            node.text().set("abc");
            int result_int;
            DOCTEST_CHECK_THROWS(result_int=parse_xml_tag_int(node, "tag", "test"));
            float result_float;
            DOCTEST_CHECK_THROWS(result_float=parse_xml_tag_float(node, "tag", "test"));
            vec2 result_vec2;
            DOCTEST_CHECK_THROWS(result_vec2=parse_xml_tag_vec2(node, "tag", "test"));
            vec3 result_vec3;
            DOCTEST_CHECK_THROWS(result_vec3=parse_xml_tag_vec3(node, "tag", "test"));
        }

        SUBCASE("open_xml_file") {
            pugi::xml_document doc;
            std::string error_string;
            DOCTEST_CHECK(!open_xml_file("non_existent.xml", doc, error_string));
            DOCTEST_CHECK(!error_string.empty());

            // Create a dummy invalid xml file
            std::ofstream outfile("invalid.xml");
            outfile << "<helios><data>blah</helios>";
            outfile.close();
            DOCTEST_CHECK(!open_xml_file("invalid.xml", doc, error_string));

            // Create a dummy valid xml file
            outfile.open("valid.xml");
            outfile << "<helios><data>blah</data></helios>";
            outfile.close();
            DOCTEST_CHECK(open_xml_file("valid.xml", doc, error_string));
            std::remove("invalid.xml");
            std::remove("valid.xml");
        }

        SUBCASE("Date::JulianDay") {
            Date d1(1, 1, 2021);
            int jd1 = d1.JulianDay();
            DOCTEST_CHECK(jd1 == 1);
            Date d2(31, 12, 2021);
            int jd2 = d2.JulianDay();
            DOCTEST_CHECK(jd2 == 365);
            Date d3(29, 2, 2020);
            int jd3 = d3.JulianDay();
            DOCTEST_CHECK(jd3 == 60);
        }

        SUBCASE("randu(int, int)") {
            int r = randu(5, 5);
            DOCTEST_CHECK(r == 5);
            for(int i=0; i<100; ++i) {
                r = randu(1, 100);
                DOCTEST_CHECK(r >= 1);
                DOCTEST_CHECK(r <= 100);
            }
        }

        SUBCASE("deblank and trim_whitespace complex") {
            std::string s1 = deblank("  hello   world  ");
            DOCTEST_CHECK(s1 == "helloworld");
            std::string s2 = trim_whitespace(" \t\r\n hello world \t\r\n ");
            DOCTEST_CHECK(s2 == "hello world");
        }

        SUBCASE("makeRotationMatrix invalid axis") {
            capture_cerr cerr_buffer;
            float T[16];
            DOCTEST_CHECK_THROWS(makeRotationMatrix(0.5f, "w", T));
        }

        SUBCASE("atan2_2pi edge cases") {
            float res = atan2_2pi(0.f, 0.f);
            DOCTEST_CHECK(res == doctest::Approx(0.f));
        }

        SUBCASE("cart2sphere and sphere2cart") {
            vec3 p1(1, 1, 1);
            SphericalCoord s1 = cart2sphere(p1);
            vec3 p2 = sphere2cart(s1);
            DOCTEST_CHECK(p2.x == doctest::Approx(p1.x));
            DOCTEST_CHECK(p2.y == doctest::Approx(p1.y));
            DOCTEST_CHECK(p2.z == doctest::Approx(p1.z));

            vec3 p_pole(0, 0, 5);
            SphericalCoord s_pole = cart2sphere(p_pole);
            DOCTEST_CHECK(s_pole.radius == doctest::Approx(5.f));
            DOCTEST_CHECK(s_pole.elevation == doctest::Approx(PI_F / 2.f));
        }

        SUBCASE("parse_* invalid input") {
            capture_cerr cerr_buffer;
            int i;
            DOCTEST_CHECK(!parse_int("1.5", i));
            float f;
            DOCTEST_CHECK(!parse_float("abc", f));
            double d;
            DOCTEST_CHECK(!parse_double("abc", d));
            unsigned int u;
            DOCTEST_CHECK(!parse_uint("-1", u));
            int2 i2;
            DOCTEST_CHECK(!parse_int2("1 abc", i2));
            int3 i3;
            DOCTEST_CHECK(!parse_int3("1 2 abc", i3));
            vec2 v2;
            DOCTEST_CHECK(!parse_vec2("1.1 abc", v2));
            vec3 v3;
            DOCTEST_CHECK(!parse_vec3("1.1 2.2 abc", v3));
            RGBcolor c;
            DOCTEST_CHECK(!parse_RGBcolor("0.1 0.2 abc", c));
            DOCTEST_CHECK(!parse_RGBcolor("0.1 0.2 1.1", c)); // out of range
        }

        SUBCASE("lineIntersection parallel") {
            vec2 p1(0, 0), q1(1, 0);
            vec2 p2(0, 1), q2(1, 1);
            DOCTEST_CHECK(!lineIntersection(p1, q1, p2, q2));
        }

        SUBCASE("JulianDay and CalendarDay") {
            int jd1 = JulianDay(1, 1, 2021);
            DOCTEST_CHECK(jd1 == 1);
            int jd2 = JulianDay(31, 12, 2021);
            DOCTEST_CHECK(jd2 == 365);
            int jd3 = JulianDay(29, 2, 2020);
            DOCTEST_CHECK(jd3 == 60);

            Date d = CalendarDay(60, 2020);
            DOCTEST_CHECK(d.day == 29);
            DOCTEST_CHECK(d.month == 2);
            DOCTEST_CHECK(d.year == 2020);

            capture_cerr cerr_buffer;
            DOCTEST_CHECK_THROWS(d=CalendarDay(366, 2021));
            DOCTEST_CHECK_THROWS(d=CalendarDay(0, 2021));
            DOCTEST_CHECK_THROWS(jd1=JulianDay(32, 1, 2021));
            DOCTEST_CHECK_THROWS(jd2=JulianDay(1, 13, 2021));
        }
    }

    TEST_CASE("Context_data.cpp coverage gaps") {
        SUBCASE("Primitive::getPrimitiveDataSize all types") {
            Context ctx;
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "uint", (uint)1);
            ctx.setPrimitiveData(p, "double", 1.0);
            ctx.setPrimitiveData(p, "vec2", make_vec2(1,1));
            ctx.setPrimitiveData(p, "vec3", make_vec3(1,1,1));
            ctx.setPrimitiveData(p, "vec4", make_vec4(1,1,1,1));
            ctx.setPrimitiveData(p, "int2", make_int2(1,1));
            ctx.setPrimitiveData(p, "int3", make_int3(1,1,1));
            ctx.setPrimitiveData(p, "int4", make_int4(1,1,1,1));
            ctx.setPrimitiveData(p, "string", "hello");

            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "uint") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "double") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec2") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec3") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec4") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "int2") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "int3") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "int4") == 1);
            DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "string") == 1);
        }

        SUBCASE("Primitive::clearPrimitiveData all types") {
            Context ctx;
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "uint", (uint)1);
            ctx.setPrimitiveData(p, "double", 1.0);
            ctx.setPrimitiveData(p, "vec2", make_vec2(1,1));
            ctx.setPrimitiveData(p, "vec3", make_vec3(1,1,1));
            ctx.setPrimitiveData(p, "vec4", make_vec4(1,1,1,1));
            ctx.setPrimitiveData(p, "int2", make_int2(1,1));
            ctx.setPrimitiveData(p, "int3", make_int3(1,1,1));
            ctx.setPrimitiveData(p, "int4", make_int4(1,1,1,1));
            ctx.setPrimitiveData(p, "string", "hello");

            ctx.clearPrimitiveData(p, "uint");
            ctx.clearPrimitiveData(p, "double");
            ctx.clearPrimitiveData(p, "vec2");
            ctx.clearPrimitiveData(p, "vec3");
            ctx.clearPrimitiveData(p, "vec4");
            ctx.clearPrimitiveData(p, "int2");
            ctx.clearPrimitiveData(p, "int3");
            ctx.clearPrimitiveData(p, "int4");
            ctx.clearPrimitiveData(p, "string");

            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "uint"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "double"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "vec2"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "vec3"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "vec4"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "int2"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "int3"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "int4"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "string"));
        }

        SUBCASE("Context data functions with invalid UUID") {
            Context ctx;
            capture_cerr cerr_buffer;
            uint bad_uuid = 999;
            DOCTEST_CHECK_THROWS(ctx.getPrimitiveDataType(bad_uuid, "test"));
            DOCTEST_CHECK_THROWS(ctx.getPrimitiveDataSize(bad_uuid, "test"));
            DOCTEST_CHECK_THROWS(ctx.doesPrimitiveDataExist(bad_uuid, "test"));
            DOCTEST_CHECK_THROWS(ctx.clearPrimitiveData(bad_uuid, "test"));
            DOCTEST_CHECK_THROWS(ctx.copyPrimitiveData(0, bad_uuid));
            DOCTEST_CHECK_THROWS(ctx.copyPrimitiveData(bad_uuid, 0));
            DOCTEST_CHECK_THROWS(ctx.renamePrimitiveData(bad_uuid, "old", "new"));
            DOCTEST_CHECK_THROWS(ctx.duplicatePrimitiveData(bad_uuid, "old", "new"));
        }

        SUBCASE("Context::clearPrimitiveData for multiple UUIDs") {
            Context ctx;
            uint p1 = ctx.addPatch();
            uint p2 = ctx.addPatch();
            ctx.setPrimitiveData(p1, "data", 1);
            ctx.setPrimitiveData(p2, "data", 2);
            std::vector<uint> uuids = {p1, p2};
            ctx.clearPrimitiveData(uuids, "data");
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p1, "data"));
            DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p2, "data"));
        }

        SUBCASE("Context::duplicatePrimitiveData all types") {
            Context ctx;
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "uint", (uint)1);
            ctx.setPrimitiveData(p, "double", 1.0);
            ctx.setPrimitiveData(p, "vec2", make_vec2(1,1));
            ctx.setPrimitiveData(p, "vec3", make_vec3(1,1,1));
            ctx.setPrimitiveData(p, "vec4", make_vec4(1,1,1,1));
            ctx.setPrimitiveData(p, "int2", make_int2(1,1));
            ctx.setPrimitiveData(p, "int3", make_int3(1,1,1));
            ctx.setPrimitiveData(p, "int4", make_int4(1,1,1,1));
            ctx.setPrimitiveData(p, "string", "hello");

            ctx.duplicatePrimitiveData(p, "uint", "uint_copy");
            ctx.duplicatePrimitiveData(p, "double", "double_copy");
            ctx.duplicatePrimitiveData(p, "vec2", "vec2_copy");
            ctx.duplicatePrimitiveData(p, "vec3", "vec3_copy");
            ctx.duplicatePrimitiveData(p, "vec4", "vec4_copy");
            ctx.duplicatePrimitiveData(p, "int2", "int2_copy");
            ctx.duplicatePrimitiveData(p, "int3", "int3_copy");
            ctx.duplicatePrimitiveData(p, "int4", "int4_copy");
            ctx.duplicatePrimitiveData(p, "string", "string_copy");

            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "uint_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "double_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "vec2_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "vec3_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "vec4_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "int2_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "int3_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "int4_copy"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "string_copy"));
        }

        SUBCASE("Context::calculatePrimitiveDataMean empty list") {
            Context ctx;
            std::vector<uint> uuids;
            float float_mean;
            capture_cerr cerr_buffer;
            ctx.calculatePrimitiveDataMean(uuids, "non_existent", float_mean);
            DOCTEST_CHECK(float_mean == 0.f);
            DOCTEST_CHECK(cerr_buffer.has_output());
        }

        SUBCASE("Context::calculatePrimitiveDataMean vec3/vec4") {
            Context ctx;
            uint p1 = ctx.addPatch();
            ctx.setPrimitiveData(p1, "v3", make_vec3(1,2,3));
            ctx.setPrimitiveData(p1, "v4", make_vec4(1,2,3,4));
            std::vector<uint> uuids = {p1};
            vec3 v3_mean;
            vec4 v4_mean;
            ctx.calculatePrimitiveDataMean(uuids, "v3", v3_mean);
            ctx.calculatePrimitiveDataMean(uuids, "v4", v4_mean);
            DOCTEST_CHECK(v3_mean == make_vec3(1,2,3));
            DOCTEST_CHECK(v4_mean == make_vec4(1,2,3,4));
        }

        SUBCASE("Context::calculatePrimitiveDataAreaWeightedMean all types") {
            Context ctx;
            uint p = ctx.addPatch(make_vec3(0,0,0), make_vec2(2,2)); // Area = 4
            ctx.setPrimitiveData(p, "double", 2.5);
            ctx.setPrimitiveData(p, "vec2", make_vec2(1,2));
            ctx.setPrimitiveData(p, "vec3", make_vec3(1,2,3));
            ctx.setPrimitiveData(p, "vec4", make_vec4(1,2,3,4));
            std::vector<uint> uuids = {p};

            double d_mean;
            ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "double", d_mean);
            DOCTEST_CHECK(d_mean == doctest::Approx(2.5));

            vec2 v2_mean;
            ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec2", v2_mean);
            DOCTEST_CHECK(v2_mean == make_vec2(1,2));

            vec3 v3_mean;
            ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec3", v3_mean);
            DOCTEST_CHECK(v3_mean == make_vec3(1,2,3));

            vec4 v4_mean;
            ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec4", v4_mean);
            DOCTEST_CHECK(v4_mean == make_vec4(1,2,3,4));
        }

        SUBCASE("Context::calculatePrimitiveDataSum all types") {
            Context ctx;
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "double", 2.5);
            ctx.setPrimitiveData(p, "vec2", make_vec2(1,2));
            ctx.setPrimitiveData(p, "vec3", make_vec3(1,2,3));
            ctx.setPrimitiveData(p, "vec4", make_vec4(1,2,3,4));
            std::vector<uint> uuids = {p};

            double d_sum;
            ctx.calculatePrimitiveDataSum(uuids, "double", d_sum);
            DOCTEST_CHECK(d_sum == doctest::Approx(2.5));

            vec2 v2_sum;
            ctx.calculatePrimitiveDataSum(uuids, "vec2", v2_sum);
            DOCTEST_CHECK(v2_sum == make_vec2(1,2));

            vec3 v3_sum;
            ctx.calculatePrimitiveDataSum(uuids, "vec3", v3_sum);
            DOCTEST_CHECK(v3_sum == make_vec3(1,2,3));

            vec4 v4_sum;
            ctx.calculatePrimitiveDataSum(uuids, "vec4", v4_sum);
            DOCTEST_CHECK(v4_sum == make_vec4(1,2,3,4));
        }

        SUBCASE("Context::incrementPrimitiveData all types") {
            Context ctx;
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "uint", (uint)1);
            ctx.setPrimitiveData(p, "float", 1.f);
            ctx.setPrimitiveData(p, "double", 1.0);
            std::vector<uint> uuids = {p};

            ctx.incrementPrimitiveData(uuids, "uint", (uint)2);
            uint u_val;
            ctx.getPrimitiveData(p, "uint", u_val);
            DOCTEST_CHECK(u_val == 3);

            ctx.incrementPrimitiveData(uuids, "float", 2.f);
            float f_val;
            ctx.getPrimitiveData(p, "float", f_val);
            DOCTEST_CHECK(f_val == doctest::Approx(3.f));

            ctx.incrementPrimitiveData(uuids, "double", 2.0);
            double d_val;
            ctx.getPrimitiveData(p, "double", d_val);
            DOCTEST_CHECK(d_val == doctest::Approx(3.0));
        }

        SUBCASE("Context::filterPrimitivesByData all types") {
            Context ctx;
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "double", 1.0);
            ctx.setPrimitiveData(p, "uint", (uint)10);
            std::vector<uint> uuids = {p};

            auto filtered_d = ctx.filterPrimitivesByData(uuids, "double", 0.5, ">");
            DOCTEST_CHECK(filtered_d.size() == 1);

            auto filtered_u = ctx.filterPrimitivesByData(uuids, "uint", (uint)5, ">");
            DOCTEST_CHECK(filtered_u.size() == 1);
        }

        SUBCASE("Object data error handling") {
            Context ctx;
            capture_cerr cerr_buffer;
            uint bad_oid = 999;
            DOCTEST_CHECK_THROWS(ctx.getObjectDataType(bad_oid, "test"));
            DOCTEST_CHECK_THROWS(ctx.getObjectDataSize(bad_oid, "test"));
            DOCTEST_CHECK_THROWS(ctx.doesObjectDataExist(bad_oid, "test"));
            DOCTEST_CHECK_THROWS(ctx.clearObjectData(bad_oid, "test"));
            DOCTEST_CHECK_THROWS(ctx.copyObjectData(0, bad_oid));
            DOCTEST_CHECK_THROWS(ctx.copyObjectData(bad_oid, 0));
            DOCTEST_CHECK_THROWS(ctx.renameObjectData(bad_oid, "old", "new"));
            DOCTEST_CHECK_THROWS(ctx.duplicateObjectData(bad_oid, "old", "new"));
        }

        SUBCASE("Global data error handling and all types") {
            Context ctx;
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_THROWS(ctx.getGlobalDataType("non_existent"));
            DOCTEST_CHECK_THROWS(ctx.getGlobalDataSize("non_existent"));
            DOCTEST_CHECK_THROWS(ctx.renameGlobalData("non_existent", "new"));
            DOCTEST_CHECK_THROWS(ctx.duplicateGlobalData("non_existent", "new"));
            DOCTEST_CHECK_THROWS(ctx.incrementGlobalData("non_existent", 1));

            ctx.setGlobalData("g_uint", std::vector<uint>{1});
            ctx.setGlobalData("g_float", std::vector<float>{1.f});
            ctx.setGlobalData("g_double", std::vector<double>{1.0});
            ctx.setGlobalData("g_vec2", std::vector<vec2>{make_vec2(1,1)});
            ctx.setGlobalData("g_vec3", std::vector<vec3>{make_vec3(1,1,1)});
            ctx.setGlobalData("g_vec4", std::vector<vec4>{make_vec4(1,1,1,1)});
            ctx.setGlobalData("g_int2", std::vector<int2>{make_int2(1,1)});
            ctx.setGlobalData("g_int3", std::vector<int3>{make_int3(1,1,1)});
            ctx.setGlobalData("g_int4", std::vector<int4>{make_int4(1,1,1,1)});
            ctx.setGlobalData("g_string", std::vector<std::string>{"hello"});

            ctx.duplicateGlobalData("g_uint", "g_uint_copy");
            ctx.duplicateGlobalData("g_float", "g_float_copy");
            ctx.duplicateGlobalData("g_double", "g_double_copy");
            ctx.duplicateGlobalData("g_vec2", "g_vec2_copy");
            ctx.duplicateGlobalData("g_vec3", "g_vec3_copy");
            ctx.duplicateGlobalData("g_vec4", "g_vec4_copy");
            ctx.duplicateGlobalData("g_int2", "g_int2_copy");
            ctx.duplicateGlobalData("g_int3", "g_int3_copy");
            ctx.duplicateGlobalData("g_int4", "g_int4_copy");
            ctx.duplicateGlobalData("g_string", "g_string_copy");

            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_uint_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_float_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_double_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec2_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec3_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec4_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int2_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int3_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int4_copy"));
            DOCTEST_CHECK(ctx.doesGlobalDataExist("g_string_copy"));

            ctx.incrementGlobalData("g_uint", (uint)1);
            ctx.incrementGlobalData("g_float", 1.f);
            ctx.incrementGlobalData("g_double", 1.0);

            std::vector<uint> g_uint;
            ctx.getGlobalData("g_uint", g_uint);
            DOCTEST_CHECK(g_uint[0] == 2);
        }
    }
}