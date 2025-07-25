#pragma once
// =================================================================================
// Suite 2: Global Helper Functions (global.h)
//
// Tests for standalone utility functions, typically found in a global header,
// that provide common functionality like math, string parsing, and file handling.
// =================================================================================
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

        SUBCASE("calculateTriangleArea with degenerate triangle") {
            vec3 v0(0, 0, 0), v1(1, 1, 1), v2(2, 2, 2); // Collinear points
            DOCTEST_CHECK(calculateTriangleArea(v0, v1, v2) == doctest::Approx(0.f));
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

        SUBCASE("atan2_2pi edge cases") {
            float res = atan2_2pi(0.f, 0.f);
            DOCTEST_CHECK(res == doctest::Approx(0.f));
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
            vec2 p5(0, 0), q5(3, 3);
            vec2 p6(1, 1), q6(2, 2);
            DOCTEST_CHECK(lineIntersection(p5, q5, p6, q6));

            // Collinear, share an endpoint
            vec2 p7(0, 0), q7(1, 1);
            vec2 p8(1, 1), q8(2, 2);
            DOCTEST_CHECK(lineIntersection(p7, q7, p8, q8));
        }
        SUBCASE("lineIntersection parallel") {
            vec2 p1(0, 0), q1(1, 0);
            vec2 p2(0, 1), q2(1, 1);
            DOCTEST_CHECK(!lineIntersection(p1, q1, p2, q2));
        }
        SUBCASE("pointInPolygon") {
            std::vector<vec2> square = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            DOCTEST_CHECK(pointInPolygon(vec2(0.5, 0.5), square));
            DOCTEST_CHECK(!pointInPolygon(vec2(1.5, 0.5), square));
            DOCTEST_CHECK(pointInPolygon(vec2(0, 0), square)); // On edge
        }
        SUBCASE("pointInPolygon with concave polygon") {
            // Concave polygon (U-shaped)
            std::vector<vec2> concave_poly = {{0, 0}, {5, 0}, {5, 5}, {3, 3}, {2, 5}, {0, 5}};
            DOCTEST_CHECK(pointInPolygon(vec2(1, 1), concave_poly)); // Inside
            DOCTEST_CHECK(!pointInPolygon(vec2(4, 4), concave_poly)); // In concave part, but outside
            DOCTEST_CHECK(pointInPolygon(vec2(2.5, 4), concave_poly)); // Inside
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

    SUBCASE("makeRotationMatrix invalid axis") {
        capture_cerr cerr_buffer;
        float T[16];
        DOCTEST_CHECK_THROWS(makeRotationMatrix(0.5f, "w", T));
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
        DOCTEST_CHECK(result.size() == 1);

        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(result = separate_string_by_delimiter("a,b,c", ""));

        DOCTEST_CHECK_NOTHROW(result = separate_string_by_delimiter(",", ","));
        DOCTEST_CHECK(result.size() == 2);
        if (result.size() == 2) {
            DOCTEST_CHECK(result[0] == "");
            DOCTEST_CHECK(result[1] == "");
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
    SUBCASE("string2RGBcolor with invalid input") {
        capture_cerr cerr_buffer;
        RGBAcolor result;
        DOCTEST_CHECK_THROWS(result = string2RGBcolor("0.1 0.2"));
        DOCTEST_CHECK_THROWS(result = string2RGBcolor("0.1 0.2 0.3 0.4 0.5"));
        DOCTEST_CHECK_THROWS(result = string2RGBcolor("a b c"));
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
    SUBCASE("File path parsing edge cases") {
        DOCTEST_CHECK(getFileExtension(".bashrc").empty());
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
    SUBCASE("parse functions with whitespace") {
        int i;
        DOCTEST_CHECK(parse_int("  123  ", i));
        DOCTEST_CHECK(i == 123);
        float f;
        DOCTEST_CHECK(parse_float("  1.23  ", f));
        DOCTEST_CHECK(f == doctest::Approx(1.23f));
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
    SUBCASE("Vector statistics with single element") {
        std::vector<float> v = {5.f};
        DOCTEST_CHECK(stdev(v) == doctest::Approx(0.f));
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
    SUBCASE("randu(int, int)") {
        int r = randu(5, 5);
        DOCTEST_CHECK(r == 5);
        for (int i = 0; i < 100; ++i) {
            r = randu(1, 100);
            DOCTEST_CHECK(r >= 1);
            DOCTEST_CHECK(r <= 100);
        }
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

static float quadratic(float x, std::vector<float> &, const void *) {
    return x * x - 4.0f;
}
static float linear(float x, std::vector<float> &, const void *) {
    return 3.5f - x;
}
static float flat(float, std::vector<float> &, const void *) {
    return 2.0f;
}
static float cubic(float x, std::vector<float> &, const void *) {
    return (x - 1.0f) * (x + 2.0f) * (x - 4.0f);
}
static float near_singular(float x, std::vector<float> &, const void *) {
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
        DOCTEST_CHECK(root == doctest::Approx(3.5f).epsilon(errtol));
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
        DOCTEST_CHECK(root == doctest::Approx(4.0f).epsilon(errtol));
    }

    SUBCASE("fzero copes with near-singular derivative") {
        std::vector<float> v;
        float root = helios::fzero(near_singular, v, nullptr, 0.01f, 1e-4f, 50);
        DOCTEST_CHECK(std::fabs(near_singular(root, v, nullptr)) < 1e-4f);
    }
}
