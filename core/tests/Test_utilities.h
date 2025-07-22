#pragma once
// =================================================================================
// Suite 1: Core Utilities and Data Types
//
// Tests for fundamental data structures (vectors, colors, date/time),
// coordinate systems, and their associated operations.
// =================================================================================
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

    SUBCASE("Date::JulianDay edge cases") {
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
        DOCTEST_CHECK_THROWS(d = CalendarDay(366, 2021));
        DOCTEST_CHECK_THROWS(d = CalendarDay(0, 2021));
        DOCTEST_CHECK_THROWS(jd1 = JulianDay(32, 1, 2021));
        DOCTEST_CHECK_THROWS(jd2 = JulianDay(1, 13, 2021));
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
}
