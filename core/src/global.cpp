/** \file "global.cpp" global declarations.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "global.h"

//PNG Libraries (reading and writing PNG images)
//! PNG debug level.
#define PNG_DEBUG 3
//! Macro to skip setjmp check.
#define PNG_SKIP_SETJMP_CHECK 1
#include "png.h"

//JPEG Libraries (reading and writing JPEG images)
extern "C" {
#include "jpeglib.h"
}

using namespace helios;

void helios::helios_runtime_error(const std::string &error_message) {
#ifdef HELIOS_DEBUG
    std::cerr << error_message << std::endl;
#endif
    throw(std::runtime_error(error_message));
}

RGBcolor RGB::red = make_RGBcolor(1.f, 0.f, 0.f);
RGBcolor RGB::blue = make_RGBcolor(0.f, 0.f, 1.f);
RGBcolor RGB::green = make_RGBcolor(0.f, 0.6f, 0.f);
RGBcolor RGB::cyan = make_RGBcolor(0.f, 1.f, 1.f);
RGBcolor RGB::magenta = make_RGBcolor(1.f, 0.f, 1.f);
RGBcolor RGB::yellow = make_RGBcolor(1.f, 1.f, 0.f);
RGBcolor RGB::orange = make_RGBcolor(1.f, 0.5f, 0.f);
RGBcolor RGB::violet = make_RGBcolor(0.5f, 0.f, 0.5f);
RGBcolor RGB::black = make_RGBcolor(0.f, 0.f, 0.f);
RGBcolor RGB::white = make_RGBcolor(1.f, 1.f, 1.f);
RGBcolor RGB::lime = make_RGBcolor(0.f, 1.f, 0.f);
RGBcolor RGB::silver = make_RGBcolor(0.75f, 0.75f, 0.75f);
RGBcolor RGB::gray = make_RGBcolor(0.5f, 0.5f, 0.5f);
RGBcolor RGB::navy = make_RGBcolor(0.f, 0.f, 0.5f);
RGBcolor RGB::brown = make_RGBcolor(0.55f, 0.27f, 0.075);
RGBcolor RGB::khaki = make_RGBcolor(0.94f, 0.92f, 0.55f);
RGBcolor RGB::greenyellow = make_RGBcolor(0.678f, 1.f, 0.184f);
RGBcolor RGB::forestgreen = make_RGBcolor(0.133f, 0.545f, 0.133f);
RGBcolor RGB::yellowgreen = make_RGBcolor(0.6, 0.8, 0.2);
RGBcolor RGB::goldenrod = make_RGBcolor(0.855, 0.647, 0.126);

RGBAcolor RGBA::red = make_RGBAcolor(RGB::red, 1.f);
RGBAcolor RGBA::blue = make_RGBAcolor(RGB::blue, 1.f);
RGBAcolor RGBA::green = make_RGBAcolor(RGB::green, 1.f);
RGBAcolor RGBA::cyan = make_RGBAcolor(RGB::cyan, 1.f);
RGBAcolor RGBA::magenta = make_RGBAcolor(RGB::magenta, 1.f);
RGBAcolor RGBA::yellow = make_RGBAcolor(RGB::yellow, 1.f);
RGBAcolor RGBA::orange = make_RGBAcolor(RGB::orange, 1.f);
RGBAcolor RGBA::violet = make_RGBAcolor(RGB::violet, 1.f);
RGBAcolor RGBA::black = make_RGBAcolor(RGB::black, 1.f);
RGBAcolor RGBA::white = make_RGBAcolor(RGB::white, 1.f);
RGBAcolor RGBA::lime = make_RGBAcolor(RGB::lime, 1.f);
RGBAcolor RGBA::silver = make_RGBAcolor(RGB::silver, 1.f);
RGBAcolor RGBA::gray = make_RGBAcolor(RGB::gray, 1.f);
RGBAcolor RGBA::navy = make_RGBAcolor(RGB::navy, 1.f);
RGBAcolor RGBA::brown = make_RGBAcolor(RGB::brown, 1.f);
RGBAcolor RGBA::khaki = make_RGBAcolor(RGB::khaki, 1.f);
RGBAcolor RGBA::greenyellow = make_RGBAcolor(RGB::greenyellow, 1.f);
RGBAcolor RGBA::forestgreen = make_RGBAcolor(RGB::forestgreen, 1.f);
RGBAcolor RGBA::yellowgreen = make_RGBAcolor(RGB::yellowgreen, 1.f);
RGBAcolor RGBA::goldenrod = make_RGBAcolor(RGB::goldenrod, 1.f);

SphericalCoord helios::nullrotation = make_SphericalCoord(0, 0);
vec3 helios::nullorigin = make_vec3(0, 0, 0);

RGBcolor helios::blend(const RGBcolor &color0, const RGBcolor &color1, float weight) {
    RGBcolor color_out;
    weight = clamp(weight, 0.f, 1.f);
    color_out.r = weight * color1.r + (1.f - weight) * color0.r;
    color_out.g = weight * color1.g + (1.f - weight) * color0.g;
    color_out.b = weight * color1.b + (1.f - weight) * color0.b;
    return color_out;
}

RGBAcolor helios::blend(const RGBAcolor &color0, const RGBAcolor &color1, float weight) {
    RGBAcolor color_out;
    weight = clamp(weight, 0.f, 1.f);
    color_out.r = weight * color1.r + (1.f - weight) * color0.r;
    color_out.g = weight * color1.g + (1.f - weight) * color0.g;
    color_out.b = weight * color1.b + (1.f - weight) * color0.b;
    color_out.a = weight * color1.a + (1.f - weight) * color0.a;
    return color_out;
}

vec3 helios::rotatePoint(const vec3 &position, const SphericalCoord &rotation) {
    return rotatePoint(position, rotation.elevation, rotation.azimuth);
}

vec3 helios::rotatePoint(const vec3 &position, float theta, float phi) {
    if (theta == 0.f && phi == 0.f) {
        return position;
    }

    float Ry[3][3], Rz[3][3];

    const float st = sin(theta);
    const float ct = cos(theta);

    const float sp = sin(phi);
    const float cp = cos(phi);

    // Setup the rotation matrix, this matrix is based off of the rotation matrix used in glRotatef.
    Ry[0][0] = ct;
    Ry[0][1] = 0.f;
    Ry[0][2] = st;
    Ry[1][0] = 0.f;
    Ry[1][1] = 1.f;
    Ry[1][2] = 0.f;
    Ry[2][0] = -st;
    Ry[2][1] = 0.f;
    Ry[2][2] = ct;

    Rz[0][0] = cp;
    Rz[0][1] = -sp;
    Rz[0][2] = 0.f;
    Rz[1][0] = sp;
    Rz[1][1] = cp;
    Rz[1][2] = 0.f;
    Rz[2][0] = 0.f;
    Rz[2][1] = 0.f;
    Rz[2][2] = 1.f;

    // Multiply Ry*Rz

    float rotMat[3][3] = {0.f};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                rotMat[i][j] = rotMat[i][j] + Rz[i][k] * Ry[k][j];
            }
        }
    }

    // Multiply the rotation matrix with the position vector.
    vec3 tmp;
    tmp.x = rotMat[0][0] * position.x + rotMat[0][1] * position.y + rotMat[0][2] * position.z;
    tmp.y = rotMat[1][0] * position.x + rotMat[1][1] * position.y + rotMat[1][2] * position.z;
    tmp.z = rotMat[2][0] * position.x + rotMat[2][1] * position.y + rotMat[2][2] * position.z;

    return tmp;
}

vec3 helios::rotatePointAboutLine(const vec3 &point, const vec3 &line_base, const vec3 &line_direction, float theta) {
    if (theta == 0.f) {
        return point;
    }

    //for reference this was taken from http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/

    vec3 position;

    vec3 tmp = line_direction;
    float mag = tmp.magnitude();
    if (mag < 1e-6f) {
        return point;
    }
    tmp = tmp / mag;
    const float u = tmp.x;
    const float v = tmp.y;
    const float w = tmp.z;

    const float a = line_base.x;
    const float b = line_base.y;
    const float c = line_base.z;

    const float x = point.x;
    const float y = point.y;
    const float z = point.z;

    const float st = sin(theta);
    const float ct = cos(theta);

    position.x = (a * (v * v + w * w) - u * (b * v + c * w - u * x - v * y - w * z)) * (1 - ct) + x * ct + (-c * v + b * w - w * y + v * z) * st;
    position.y = (b * (u * u + w * w) - v * (a * u + c * w - u * x - v * y - w * z)) * (1 - ct) + y * ct + (c * u - a * w + w * x - u * z) * st;
    position.z = (c * (u * u + v * v) - w * (a * u + b * v - u * x - v * y - w * z)) * (1 - ct) + z * ct + (-b * u + a * v - v * x + u * y) * st;

    return position;
}

float helios::calculateTriangleArea(const vec3 &v0, const vec3 &v1, const vec3 &v2) {
    const float a = (v1 - v0).magnitude();
    const float b = (v2 - v0).magnitude();
    const float c = (v2 - v1).magnitude();

    const float s = 0.5f * (a + b + c);
    return sqrtf(s * (s - a) * (s - b) * (s - c));
}

int helios::Date::JulianDay() const {
    int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
    int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    int *skips;

    if (isLeapYear()) { //leap year
        skips = skips_leap;
    } else { //non-leap year
        skips = skips_nonleap;
    }

    return skips[month - 1] + day;
}

void helios::Date::incrementDay() {
    // Compute “Julian day of year” for *this
    const int jd = Calendar2Julian(*this);

    // 2) Sanity-check jd
    bool leap = isLeapYear();
    const int maxJD = leap ? 366 : 365;
    if (jd < 1 || jd > maxJD) {
        helios_runtime_error("ERROR (incrementDay): current date out of range (JD=" + std::to_string(jd) + ")");
    }

    // Advance
    if (jd < maxJD) {
        // still inside this year
        const Date next = Julian2Calendar(jd + 1, year);
        day = next.day;
        month = next.month;
        // year unchanged
    } else {
        // rollover to Jan 1 of next year
        year += 1;
        month = 1;
        day = 1;
    }
}

bool helios::Date::isLeapYear() const {
    if ((year - 2000) % 4 == 0) { //leap year
        return true;
    } else { //non-leap year
        return false;
    }
}

float helios::randu() {
    return float(rand()) / float(RAND_MAX + 1.);
}

int helios::randu(int imin, int imax) {
    float ru = randu();

    if (imin == imax || imin > imax) {
        return imin;
    } else {
        return imin + (int) lround(float(imax - imin) * ru);
    }
}

float helios::acos_safe(float x) {
    if (x < -1.0) x = -1.0;
    else if (x > 1.0) x = 1.0;
    return acosf(x);
}

float helios::asin_safe(float x) {
    if (x < -1.0) x = -1.0;
    else if (x > 1.0) x = 1.0;
    return asinf(x);
}

template<typename T>
T helios::powi(T base, std::size_t exp) {
    T result = static_cast<T>(1);
    while (exp > 0) {
        // If the low bit is set, multiply result by current base
        if (exp & 1) {
            result *= base;
        }
        // Square the base for the next bit
        base *= base;
        // Shift off the processed bit
        exp >>= 1;
    }
    return result;
}

template float helios::powi(float base, std::size_t exp);
template double helios::powi(double base, std::size_t exp);
template int helios::powi(int base, std::size_t exp);
template uint helios::powi(uint base, std::size_t exp);
template char helios::powi(char base, std::size_t exp);
template size_t helios::powi(size_t base, std::size_t exp);

bool helios::lineIntersection(const vec2 &p1, const vec2 &q1, const vec2 &p2, const vec2 &q2) {
    float ax = q1.x - p1.x; // direction of line a
    float ay = q1.y - p1.y; // ax and ay as above

    float bx = p2.x - q2.x; // direction of line b, reversed
    float by = p2.y - q2.y; // really -by and -by as above

    float dx = p2.x - p1.x; // right-hand side
    float dy = p2.y - p1.y;

    float det = ax * by - ay * bx;

    if (det == 0) return false;

    float r = (dx * by - dy * bx) / det;
    float s = (ax * dy - ay * dx) / det;

    return !(r < 0 || r > 1 || s < 0 || s > 1);
}

bool helios::pointInPolygon(const vec2 &point, const std::vector<vec2> &polygon_verts) {
    std::vector<vec2> pverts = polygon_verts;
    pverts.push_back(polygon_verts.front());

    vec2 p1 = point;
    vec2 q1 = make_vec2(100000, 0); //todo: calculate a bounding box

    int Nintersect = 0;
    for (size_t i = 0; i < pverts.size() - 1; i++) {
        vec2 p2 = pverts.at(i);
        vec2 q2 = pverts.at(i + 1);

        if (lineIntersection(p1, q1, p2, q2)) {
            Nintersect++;
        }
    }

    if (Nintersect != 0 && Nintersect % 2 == 1) {
        return true;
    } else {
        return false;
    }
}

void helios::wait(float seconds) {
    int msec = (int) lround(seconds * 1000.f);
    std::this_thread::sleep_for(std::chrono::milliseconds(msec));
}

void helios::makeRotationMatrix(const float rotation, const char *axis, float (&T)[16]) {
    float sx = sin(rotation);
    float cx = cos(rotation);

    if (strcmp(axis, "x") == 0) {
        T[0] = 1.f; //(0,0)
        T[1] = 0.f; //(0,1)
        T[2] = 0.f; //(0,2)
        T[3] = 0.f; //(0,3)
        T[4] = 0.f; //(1,0)
        T[5] = cx; //(1,1)
        T[6] = -sx; //(1,2)
        T[7] = 0.f; //(1,3)
        T[8] = 0.f; //(2,0)
        T[9] = sx; //(2,1)
        T[10] = cx; //(2,2)
        T[11] = 0.f; //(2,3)
    } else if (strcmp(axis, "y") == 0) {
        T[0] = cx; //(0,0)
        T[1] = 0.f; //(0,1)
        T[2] = sx; //(0,2)
        T[3] = 0.f; //(0,3)
        T[4] = 0.f; //(1,0)
        T[5] = 1.f; //(1,1)
        T[6] = 0.f; //(1,2)
        T[7] = 0.f; //(1,3)
        T[8] = -sx; //(2,0)
        T[9] = 0.f; //(2,1)
        T[10] = cx; //(2,2)
        T[11] = 0.f; //(2,3)
    } else if (strcmp(axis, "z") == 0) {
        T[0] = cx; //(0,0)
        T[1] = -sx; //(0,1)
        T[2] = 0.f; //(0,2)
        T[3] = 0.f; //(0,3)
        T[4] = sx; //(1,0)
        T[5] = cx; //(1,1)
        T[6] = 0.f; //(1,2)
        T[7] = 0.f; //(1,3)
        T[8] = 0.f; //(2,0)
        T[9] = 0.f; //(2,1)
        T[10] = 1.f; //(2,2)
        T[11] = 0.f; //(2,3)
    } else {
        helios_runtime_error("ERROR (makeRotationMatrix): Rotation axis should be one of x, y, or z.");
    }
    T[12] = T[13] = T[14] = 0.f;
    T[15] = 1.f;
}

void helios::makeRotationMatrix(float rotation, const helios::vec3 &axis, float (&T)[16]) {
    vec3 u = axis;
    u.normalize();

    float sx = sin(rotation);
    float cx = cos(rotation);

    T[0] = cx + u.x * u.x * (1.f - cx); //(0,0)
    T[1] = u.x * u.y * (1.f - cx) - u.z * sx; //(0,1)
    T[2] = u.x * u.z * (1.f - cx) + u.y * sx; //(0,2)
    T[3] = 0.f; //(0,3)
    T[4] = u.y * u.x * (1.f - cx) + u.z * sx; //(1,0)
    T[5] = cx + u.y * u.y * (1.f - cx); //(1,1)
    T[6] = u.y * u.z * (1.f - cx) - u.x * sx; //(1,2)
    T[7] = 0.f; //(1,3)
    T[8] = u.z * u.x * (1.f - cx) - u.y * sx; //(2,0)
    T[9] = u.z * u.y * (1.f - cx) + u.x * sx; //(2,1)
    T[10] = cx + u.z * u.z * (1.f - cx); //(2,2)
    T[11] = 0.f; //(2,3)

    T[12] = T[13] = T[14] = 0.f;
    T[15] = 1.f;
}

void helios::makeRotationMatrix(float rotation, const helios::vec3 &origin, const helios::vec3 &axis, float (&T)[16]) {
    //Construct inverse translation matrix to translate back to the origin
    float Ttrans[16];
    makeIdentityMatrix(Ttrans);

    Ttrans[3] = -origin.x; //(0,3)
    Ttrans[7] = -origin.y; //(1,3)
    Ttrans[11] = -origin.z; //(2,3)

    //Construct rotation matrix
    vec3 u = axis;
    u.normalize();

    float sx = sin(rotation);
    float cx = cos(rotation);

    float Trot[16];
    makeIdentityMatrix(Trot);

    Trot[0] = cx + u.x * u.x * (1.f - cx); //(0,0)
    Trot[1] = u.x * u.y * (1.f - cx) - u.z * sx; //(0,1)
    Trot[2] = u.x * u.z * (1.f - cx) + u.y * sx; //(0,2)
    Trot[3] = 0.f; //(0,3)
    Trot[4] = u.y * u.x * (1.f - cx) + u.z * sx; //(1,0)
    Trot[5] = cx + u.y * u.y * (1.f - cx); //(1,1)
    Trot[6] = u.y * u.z * (1.f - cx) - u.x * sx; //(1,2)
    Trot[7] = 0.f; //(1,3)
    Trot[8] = u.z * u.x * (1.f - cx) - u.y * sx; //(2,0)
    Trot[9] = u.z * u.y * (1.f - cx) + u.x * sx; //(2,1)
    Trot[10] = cx + u.z * u.z * (1.f - cx); //(2,2)
    Trot[11] = 0.f; //(2,3)

    //Multiply first two matrices and store in 'T'
    matmult(Trot, Ttrans, T);

    //Construct transformation matrix to translate back to 'origin'
    Ttrans[3] = origin.x; //(0,3)
    Ttrans[7] = origin.y; //(1,3)
    Ttrans[11] = origin.z; //(2,3)

    matmult(Ttrans, T, T);
}

void helios::makeTranslationMatrix(const helios::vec3 &translation, float (&T)[16]) {
    T[0] = 1.f; //(0,0)
    T[1] = 0.f; //(0,1)
    T[2] = 0.f; //(0,2)
    T[3] = translation.x; //(0,3)
    T[4] = 0.f; //(1,0)
    T[5] = 1.f; //(1,1)
    T[6] = 0.f; //(1,2)
    T[7] = translation.y; //(1,3)
    T[8] = 0.f; //(2,0)
    T[9] = 0.f; //(2,1)
    T[10] = 1.f; //(2,2)
    T[11] = translation.z; //(2,3)
    T[12] = 0.f; //(3,0)
    T[13] = 0.f; //(3,1)
    T[14] = 0.f; //(3,2)
    T[15] = 1.f; //(3,3)
}

void helios::makeScaleMatrix(const helios::vec3 &scale, float (&transform)[16]) {
    transform[0] = scale.x; //(0,0)
    transform[1] = 0.f; //(0,1)
    transform[2] = 0.f; //(0,2)
    transform[3] = 0.f; //(0,3)
    transform[4] = 0.f; //(1,0)
    transform[5] = scale.y; //(1,1)
    transform[6] = 0.f; //(1,2)
    transform[7] = 0.f; //(1,3)
    transform[8] = 0.f; //(2,0)
    transform[9] = 0.f; //(2,1)
    transform[10] = scale.z; //(2,2)
    transform[11] = 0.f; //(2,3)
    transform[12] = 0.f; //(3,0)
    transform[13] = 0.f; //(3,1)
    transform[14] = 0.f; //(3,2)
    transform[15] = 1.f; //(3,3)
}

void helios::makeScaleMatrix(const helios::vec3 &scale, const helios::vec3 &point, float (&transform)[16]) {
    transform[0] = scale.x; //(0,0)
    transform[1] = 0.f; //(0,1)
    transform[2] = 0.f; //(0,2)
    transform[3] = point.x * (1 - scale.x); //(0,3)
    transform[4] = 0.f; //(1,0)
    transform[5] = scale.y; //(1,1)
    transform[6] = 0.f; //(1,2)
    transform[7] = point.y * (1 - scale.y); //(1,3)
    transform[8] = 0.f; //(2,0)
    transform[9] = 0.f; //(2,1)
    transform[10] = scale.z; //(2,2)
    transform[11] = point.z * (1 - scale.z); //(2,3)
    transform[12] = 0.f; //(3,0)
    transform[13] = 0.f; //(3,1)
    transform[14] = 0.f; //(3,2)
    transform[15] = 1.f; //(3,3)
}

void helios::matmult(const float ML[16], const float MR[16], float (&T)[16]) {
    float M[16] = {0.f};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                M[4 * i + j] = M[4 * i + j] + ML[4 * i + k] * MR[4 * k + j];
            }
        }
    }

    for (int i = 0; i < 16; i++) {
        T[i] = M[i];
    }
}

void helios::vecmult(const float M[16], const helios::vec3 &v3, helios::vec3 &result) {
    float v[4] = {v3.x, v3.y, v3.z, 1.f};

    float V[4] = {0.f};

    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 4; ++k) {
            V[i] += M[4 * i + k] * v[k];
        }
    }

    result.x = V[0];
    result.y = V[1];
    result.z = V[2];
}

void helios::vecmult(const float M[16], const float v[3], float (&result)[3]) {
    float V[4] = {0.f};
    float v4[4] = {v[0], v[1], v[2], 1.f};

    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            V[j] = V[j] + v4[k] * M[k + 4 * j];
        }
    }

    for (int i = 0; i < 3; i++) {
        result[i] = V[i];
    }
}

void helios::makeIdentityMatrix(float (&T)[16]) {
    /* [0,0] */
    T[0] = 1.f;
    /* [0,1] */
    T[1] = 0.f;
    /* [0,2] */
    T[2] = 0.f;
    /* [0,3] */
    T[3] = 0.f;
    /* [1,0] */
    T[4] = 0.f;
    /* [1,1] */
    T[5] = 1.f;
    /* [1,2] */
    T[6] = 0.f;
    /* [1,3] */
    T[7] = 0.f;
    /* [2,0] */
    T[8] = 0.f;
    /* [2,1] */
    T[9] = 0.f;
    /* [2,2] */
    T[10] = 1.f;
    /* [2,3] */
    T[11] = 0.f;
    /* [3,0] */
    T[12] = 0.f;
    /* [3,1] */
    T[13] = 0.f;
    /* [3,2] */
    T[14] = 0.f;
    /* [3,3] */
    T[15] = 1.f;
}

float helios::deg2rad(float deg) {
    return deg * float(M_PI) / 180.f;
}

float helios::rad2deg(float rad) {
    return rad * 180.f / float(M_PI);
}

float helios::atan2_2pi(float y, float x) {
    float v = 0;

    if (x > 0.f) {
        v = atanf(y / x);
    }
    if (y >= 0.f && x < 0.f) {
        v = float(M_PI) + atanf(y / x);
    }
    if (y < 0.f && x < 0.f) {
        v = -float(M_PI) + atanf(y / x);
    }
    if (y > 0.f && x == 0.f) {
        v = 0.5f * float(M_PI);
    }
    if (y < 0.f && x == 0.f) {
        v = -0.5f * float(M_PI);
    }
    if (v < 0.f) {
        v = v + 2.f * float(M_PI);
    }
    return v;
}

SphericalCoord helios::cart2sphere(const vec3 &Cartesian) {
    float radius = sqrtf(Cartesian.x * Cartesian.x + Cartesian.y * Cartesian.y + Cartesian.z * Cartesian.z);
    return {radius, asin_safe(Cartesian.z / radius), atan2_2pi(Cartesian.x, Cartesian.y)};
}

vec3 helios::sphere2cart(const SphericalCoord &Spherical) {
    return {Spherical.radius * cosf(Spherical.elevation) * sinf(Spherical.azimuth), Spherical.radius * cosf(Spherical.elevation) * cosf(Spherical.azimuth), Spherical.radius * sinf(Spherical.elevation)};
}

vec2 helios::string2vec2(const char *str) {
    float o[2] = {99999, 99999};
    std::string tmp;

    std::istringstream stream(str);
    int c = 0;
    while (stream >> tmp) {
        if (c == 2) {
            break;
        }
        if (!parse_float(tmp, o[c])) {
            o[c] = 99999;
        }
        c++;
    }
    return make_vec2(o[0], o[1]);
}

vec3 helios::string2vec3(const char *str) {
    float o[3] = {99999, 99999, 99999};
    std::string tmp;

    std::istringstream stream(str);
    int c = 0;
    while (stream >> tmp) {
        if (c == 3) {
            break;
        }
        if (!parse_float(tmp, o[c])) {
            o[c] = 99999;
        }
        c++;
    }
    return make_vec3(o[0], o[1], o[2]);
}

vec4 helios::string2vec4(const char *str) {
    float o[4] = {99999, 99999, 99999, 99999};
    std::string tmp;

    std::istringstream stream(str);
    int c = 0;
    while (stream >> tmp) {
        if (c == 4) {
            break;
        }
        if (!parse_float(tmp, o[c])) {
            o[c] = 99999;
        }
        c++;
    }
    return make_vec4(o[0], o[1], o[2], o[3]);
}

int2 helios::string2int2(const char *str) {
    int o[2] = {99999, 99999};
    std::string tmp;

    std::istringstream stream(str);
    int c = 0;
    while (stream >> tmp) {
        if (c == 2) {
            break;
        }
        if (!parse_int(tmp, o[c])) {
            o[c] = 99999;
        }
        c++;
    }
    return make_int2(o[0], o[1]);
}

int3 helios::string2int3(const char *str) {
    int o[3] = {99999, 99999, 99999};
    std::string tmp;

    std::istringstream stream(str);
    int c = 0;
    while (stream >> tmp) {
        if (c == 3) {
            break;
        }
        if (!parse_int(tmp, o[c])) {
            o[c] = 99999;
        }
        c++;
    }
    return make_int3(o[0], o[1], o[2]);
}

int4 helios::string2int4(const char *str) {
    int o[4] = {99999, 99999, 99999, 99999};
    std::string tmp;

    std::istringstream stream(str);
    int c = 0;
    while (stream >> tmp) {
        if (c == 4) {
            break;
        }
        if (!parse_int(tmp, o[c])) {
            o[c] = 99999;
        }
        c++;
    }
    return make_int4(o[0], o[1], o[2], o[3]);
}

RGBAcolor helios::string2RGBcolor(const char *str) {
    float o[4] = {0, 0, 0, 1};
    std::string tmp;

    std::istringstream stream(str);
    int c = 0;
    while (stream >> tmp) {
        if (c == 4) {
            break;
        }
        if (!parse_float(tmp, o[c])) {
            o[c] = 0;
        }
        c++;
    }
    return make_RGBAcolor(o[0], o[1], o[2], o[3]);
}

bool helios::parse_float(const std::string &input_string, float &converted_float) {
    try {
        size_t read = 0;
        std::string str = trim_whitespace(input_string);
        double converted_double = std::stod(str, &read);
        converted_float = (float) converted_double;
        if (str.size() != read)
            return false;
    } catch (std::invalid_argument &e) {
        return false;
    }
    return true;
}

bool helios::parse_double(const std::string &input_string, double &converted_double) {
    try {
        size_t read = 0;
        std::string str = trim_whitespace(input_string);
        converted_double = std::stod(str, &read);
        if (str.size() != read)
            return false;
    } catch (std::invalid_argument &e) {
        return false;
    }
    return true;
}

bool helios::parse_int(const std::string &input_string, int &converted_int) {
    try {
        size_t read = 0;
        std::string str = trim_whitespace(input_string);
        converted_int = std::stoi(str, &read);
        if (str.size() != read)
            return false;
    } catch (std::invalid_argument &e) {
        return false;
    }
    return true;
}

bool helios::parse_int2(const std::string &input_string, int2 &converted_int2) {
    std::istringstream vecstream(input_string);
    std::vector<std::string> tmp_s(2);
    vecstream >> tmp_s[0];
    vecstream >> tmp_s[1];
    int2 tmp;
    if (!parse_int(tmp_s[0], tmp.x) || !parse_int(tmp_s[1], tmp.y)) {
        return false;
    } else {
        converted_int2 = tmp;
    }
    return true;
}

bool helios::parse_int3(const std::string &input_string, int3 &converted_int3) {
    std::istringstream vecstream(input_string);
    std::vector<std::string> tmp_s(3);
    vecstream >> tmp_s[0];
    vecstream >> tmp_s[1];
    vecstream >> tmp_s[2];
    int3 tmp;
    if (!parse_int(tmp_s[0], tmp.x) || !parse_int(tmp_s[1], tmp.y) || !parse_int(tmp_s[2], tmp.z)) {
        return false;
    } else {
        converted_int3 = tmp;
    }
    return true;
}

bool helios::parse_uint(const std::string &input_string, uint &converted_uint) {
    try {
        size_t read = 0;
        std::string str = trim_whitespace(input_string);
        int converted_int = std::stoi(str, &read);
        if (str.size() != read || converted_int < 0) {
            return false;
        } else {
            converted_uint = (uint) converted_int;
        }
    } catch (std::invalid_argument &e) {
        return false;
    }
    return true;
}

bool helios::parse_vec2(const std::string &input_string, vec2 &converted_vec2) {
    std::istringstream vecstream(input_string);
    std::vector<std::string> tmp_s(2);
    vecstream >> tmp_s[0];
    vecstream >> tmp_s[1];
    vec2 tmp;
    if (!parse_float(tmp_s[0], tmp.x) || !parse_float(tmp_s[1], tmp.y)) {
        return false;
    } else {
        converted_vec2 = tmp;
    }
    return true;
}

bool helios::parse_vec3(const std::string &input_string, vec3 &converted_vec3) {
    std::istringstream vecstream(input_string);
    std::vector<std::string> tmp_s(3);
    vecstream >> tmp_s[0];
    vecstream >> tmp_s[1];
    vecstream >> tmp_s[2];
    vec3 tmp;
    if (!parse_float(tmp_s[0], tmp.x) || !parse_float(tmp_s[1], tmp.y) || !parse_float(tmp_s[2], tmp.z)) {
        return false;
    } else {
        converted_vec3 = tmp;
    }
    return true;
}

bool helios::parse_RGBcolor(const std::string &input_string, RGBcolor &converted_rgb) {
    std::istringstream vecstream(input_string);
    std::vector<std::string> tmp_s(3);
    vecstream >> tmp_s[0];
    vecstream >> tmp_s[1];
    vecstream >> tmp_s[2];
    RGBcolor tmp;
    if (!parse_float(tmp_s[0], tmp.r) || !parse_float(tmp_s[1], tmp.g) || !parse_float(tmp_s[2], tmp.b)) {
        return false;
    } else {
        if (tmp.r < 0 || tmp.g < 0 || tmp.b < 0 || tmp.r > 1.f || tmp.g > 1.f || tmp.b > 1.f) {
            return false;
        }
        converted_rgb = tmp;
    }
    return true;
}

bool helios::open_xml_file(const std::string &xml_file, pugi::xml_document &xmldoc, std::string &error_string) {
    const std::string &fn = xml_file;
    std::string ext = getFileExtension(xml_file);
    if (ext != ".xml" && ext != ".XML") {
        error_string = "XML file " + fn + " is not XML format.";
        return false;
    }

    //load file
    pugi::xml_parse_result load_result = xmldoc.load_file(xml_file.c_str());

    //error checking
    if (!load_result) {
        error_string = "XML file " + xml_file + " parsed with errors: " + load_result.description();
        return false;
    }

    pugi::xml_node helios = xmldoc.child("helios");

    if (helios.empty()) {
        error_string = "XML file " + xml_file + " does not have tag '<helios> ... </helios>' bounding all other tags.";
        return false;
    }

    return true;
}

int helios::parse_xml_tag_int(const pugi::xml_node &node, const std::string &tag, const std::string &calling_function) {
    std::string value_string = node.child_value();
    if (value_string.empty()) {
        return 0;
    }
    int value;
    if (!parse_int(value_string, value)) {
        helios_runtime_error("ERROR (" + calling_function + "): Could not parse tag '" + tag + "' integer value.");
    }
    return value;
}

float helios::parse_xml_tag_float(const pugi::xml_node &node, const std::string &tag, const std::string &calling_function) {
    std::string value_string = node.child_value();
    if (value_string.empty()) {
        return 0;
    }
    float value;
    if (!parse_float(value_string, value)) {
        helios_runtime_error("ERROR (" + calling_function + "): Could not parse tag '" + tag + "' float value.");
    }
    return value;
}

vec2 helios::parse_xml_tag_vec2(const pugi::xml_node &node, const std::string &tag, const std::string &calling_function) {
    std::string value_string = node.child_value();
    if (value_string.empty()) {
        return {0, 0};
    }
    vec2 value;
    if (!parse_vec2(value_string, value)) {
        helios_runtime_error("ERROR (" + calling_function + "): Could not parse tag '" + tag + "' vec2 value.");
    }
    return value;
}

vec3 helios::parse_xml_tag_vec3(const pugi::xml_node &node, const std::string &tag, const std::string &calling_function) {
    std::string value_string = node.child_value();
    if (value_string.empty()) {
        return {0, 0, 0};
    }
    vec3 value;
    if (!parse_vec3(value_string, value)) {
        helios_runtime_error("ERROR (" + calling_function + "): Could not parse tag '" + tag + "' vec3 value.");
    }
    return value;
}

std::string helios::parse_xml_tag_string(const pugi::xml_node &node, const std::string &tag, const std::string &calling_function) {
    return deblank(node.child_value());
}

std::string helios::deblank(const char *input) {
    std::string out;
    out.reserve(std::strlen(input));
    for (const char *p = input; *p; ++p) {
        if (*p != ' ') {
            out.push_back(*p);
        }
    }
    return out;
}

std::string helios::deblank(const std::string &input) {
    return deblank(input.c_str());
}

std::string helios::trim_whitespace(const std::string &input) {
    static const std::string WHITESPACE = " \n\r\t\f\v";

    // Find first non-whitespace character
    size_t start = input.find_first_not_of(WHITESPACE);
    if (start == std::string::npos) {
        return ""; // String is all whitespace
    }

    // Find last non-whitespace character
    size_t end = input.find_last_not_of(WHITESPACE);

    // Return the trimmed substring
    return input.substr(start, end - start + 1);
}

std::vector<std::string> helios::separate_string_by_delimiter(const std::string &inputstring, const std::string &delimiter) {
    std::vector<std::string> separated_string;

    size_t pos = 0;
    size_t found;
    while ((found = inputstring.find(delimiter, pos)) != std::string::npos) {
        separated_string.push_back(trim_whitespace(inputstring.substr(pos, found - pos)));
        pos = found + delimiter.size();
    }

    // add the remaining part (including case of no delimiter found)
    separated_string.push_back(trim_whitespace(inputstring.substr(pos)));

    return separated_string;
}

template<typename anytype>
anytype helios::clamp(anytype value, anytype min, anytype max) {
    if (value < min) {
        value = min;
    } else if (value > max) {
        value = max;
    }
    return value;
}

template float helios::clamp<float>(float, float, float);

template double helios::clamp<double>(double, double, double);

template int helios::clamp<int>(int, int, int);

template uint helios::clamp<uint>(uint, uint, uint);

template unsigned char helios::clamp<unsigned char>(unsigned char, unsigned char, unsigned char);

float helios::sum(const std::vector<float> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (sum): Vector is empty.");
    }

    float m = 0;
    for (float i: vect) {
        m += i;
    }

    return m;
}

float helios::mean(const std::vector<float> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (mean): Vector is empty.");
    }

    float m = 0;
    for (float i: vect) {
        m += i;
    }
    m /= float(vect.size());

    return m;
}

float helios::min(const std::vector<float> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (min): Vector is empty.");
    }

    return *std::min_element(vect.begin(), vect.end());
}

int helios::min(const std::vector<int> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (min): Vector is empty.");
    }

    return *std::min_element(vect.begin(), vect.end());
}

vec3 helios::min(const std::vector<vec3> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (min): Vector is empty.");
    }

    vec3 vmin = vect.at(0);

    for (int i = 1; i < vect.size(); i++) {
        if (vect.at(i).x < vmin.x) {
            vmin.x = vect.at(i).x;
        }
        if (vect.at(i).y < vmin.y) {
            vmin.y = vect.at(i).y;
        }
        if (vect.at(i).z < vmin.z) {
            vmin.z = vect.at(i).z;
        }
    }

    return vmin;
}

float helios::max(const std::vector<float> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (max): Vector is empty.");
    }

    return *std::max_element(vect.begin(), vect.end());
}

int helios::max(const std::vector<int> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (max): Vector is empty.");
    }

    return *std::max_element(vect.begin(), vect.end());
}

vec3 helios::max(const std::vector<vec3> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (max): Vector is empty.");
    }

    vec3 vmax = vect.at(0);

    for (int i = 1; i < vect.size(); i++) {
        if (vect.at(i).x > vmax.x) {
            vmax.x = vect.at(i).x;
        }
        if (vect.at(i).y > vmax.y) {
            vmax.y = vect.at(i).y;
        }
        if (vect.at(i).z > vmax.z) {
            vmax.z = vect.at(i).z;
        }
    }

    return vmax;
}

float helios::stdev(const std::vector<float> &vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (stdev): Vector is empty.");
    }

    size_t size = vect.size();

    float m = 0;
    for (float i: vect) {
        m += i;
    }
    m /= float(size);

    float stdev = 0;
    for (float i: vect) {
        stdev += powf(i - m, 2.0);
    }

    return sqrtf(stdev / float(size));
}

float helios::median(std::vector<float> vect) {
    if (vect.empty()) {
        helios_runtime_error("ERROR (median): Vector is empty.");
    }

    size_t size = vect.size();

    sort(vect.begin(), vect.end());

    size_t middle_index = size / 2;

    float median;
    if (size % 2 == 0) {
        median = (vect.at(middle_index) + vect.at(middle_index - 1)) / 2.f;
    } else {
        median = vect.at(middle_index);
    }
    return median;
}

template<typename anytype>
void helios::resize_vector(std::vector<std::vector<anytype> > &vec, uint Nx, uint Ny) {
    vec.resize(Ny);
    for (int j = 0; j < Ny; j++) {
        vec.at(j).resize(Nx);
    }
}

template void helios::resize_vector<float>(std::vector<std::vector<float> > &, uint, uint);

template void helios::resize_vector<double>(std::vector<std::vector<double> > &, uint, uint);

template void helios::resize_vector<int>(std::vector<std::vector<int> > &, uint, uint);

template void helios::resize_vector<uint>(std::vector<std::vector<uint> > &, uint, uint);

template void helios::resize_vector<std::string>(std::vector<std::vector<std::string> > &, uint, uint);

template void helios::resize_vector<helios::vec2>(std::vector<std::vector<helios::vec2> > &, uint, uint);

template void helios::resize_vector<helios::vec3>(std::vector<std::vector<helios::vec3> > &, uint, uint);

template void helios::resize_vector<helios::vec4>(std::vector<std::vector<helios::vec4> > &, uint, uint);

template void helios::resize_vector<helios::int2>(std::vector<std::vector<helios::int2> > &, uint, uint);

template void helios::resize_vector<helios::int3>(std::vector<std::vector<helios::int3> > &, uint, uint);

template void helios::resize_vector<helios::int4>(std::vector<std::vector<helios::int4> > &, uint, uint);

template<typename anytype>
void helios::resize_vector(std::vector<std::vector<std::vector<anytype> > > &vec, uint Nx, uint Ny, uint Nz) {
    vec.resize(Nz);
    for (int k = 0; k < Nz; k++) {
        vec.at(k).resize(Ny);
        for (int j = 0; j < Ny; j++) {
            vec.at(k).at(j).resize(Nx);
        }
    }
}

template void helios::resize_vector<float>(std::vector<std::vector<std::vector<float> > > &, uint, uint, uint);

template void helios::resize_vector<double>(std::vector<std::vector<std::vector<double> > > &, uint, uint, uint);

template void helios::resize_vector<int>(std::vector<std::vector<std::vector<int> > > &, uint, uint, uint);

template void helios::resize_vector<uint>(std::vector<std::vector<std::vector<uint> > > &, uint, uint, uint);

template void helios::resize_vector<std::string>(std::vector<std::vector<std::vector<std::string> > > &, uint, uint, uint);

template void helios::resize_vector<helios::vec2>(std::vector<std::vector<std::vector<helios::vec2> > > &, uint, uint, uint);

template void helios::resize_vector<helios::vec3>(std::vector<std::vector<std::vector<helios::vec3> > > &, uint, uint, uint);

template void helios::resize_vector<helios::vec4>(std::vector<std::vector<std::vector<helios::vec4> > > &, uint, uint, uint);

template void helios::resize_vector<helios::int2>(std::vector<std::vector<std::vector<helios::int2> > > &, uint, uint, uint);

template void helios::resize_vector<helios::int3>(std::vector<std::vector<std::vector<helios::int3> > > &, uint, uint, uint);

template void helios::resize_vector<helios::int4>(std::vector<std::vector<std::vector<helios::int4> > > &, uint, uint, uint);

template<typename anytype>
void helios::resize_vector(std::vector<std::vector<std::vector<std::vector<anytype> > > > &vec, uint Nx, uint Ny, uint Nz, uint Nw) {
    vec.resize(Nw);
    for (int w = 0; w < Nw; w++) {
        vec.at(w).resize(Nz);
        for (int k = 0; k < Nz; k++) {
            vec.at(w).at(k).resize(Ny);
            for (int j = 0; j < Ny; j++) {
                vec.at(w).at(k).at(j).resize(Nx);
            }
        }
    }
}

template void helios::resize_vector<float>(std::vector<std::vector<std::vector<std::vector<float> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<double>(std::vector<std::vector<std::vector<std::vector<double> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<int>(std::vector<std::vector<std::vector<std::vector<int> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<uint>(std::vector<std::vector<std::vector<std::vector<uint> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<std::string>(std::vector<std::vector<std::vector<std::vector<std::string> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<helios::vec2>(std::vector<std::vector<std::vector<std::vector<helios::vec2> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<helios::vec3>(std::vector<std::vector<std::vector<std::vector<helios::vec3> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<helios::vec4>(std::vector<std::vector<std::vector<std::vector<helios::vec4> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<helios::int2>(std::vector<std::vector<std::vector<std::vector<helios::int2> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<helios::int3>(std::vector<std::vector<std::vector<std::vector<helios::int3> > > > &, uint, uint, uint, uint);

template void helios::resize_vector<helios::int4>(std::vector<std::vector<std::vector<std::vector<helios::int4> > > > &, uint, uint, uint, uint);

Date helios::CalendarDay(int Julian_day, int year) {
    // -----------------------------  input checks  ----------------------------
    if (Julian_day < 1 || Julian_day > 366)
        helios_runtime_error("ERROR (CalendarDay): Julian day out of range [1–366].");

    if (year < 1000)
        helios_runtime_error("ERROR (CalendarDay): Year must be given in YYYY format.");

    const bool leap =
            (year % 4 == 0 && year % 100 != 0) || // divisible by 4 but not by 100
            (year % 400 == 0); // or divisible by 400

    if (!leap && Julian_day == 366)
        helios_runtime_error("ERROR (CalendarDay): Day 366 occurs only in leap years.");

    // -------------------  month lengths for the chosen year  -----------------
    // Index 0 = January, …, 11 = December
    int month_lengths[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (leap) // adjust February
        month_lengths[1] = 29;

    // ---------------------------  computation  ------------------------------
    int d_remaining = Julian_day; // days still to account for
    int month = 1; // 1‑based calendar month

    // subtract complete months until the remainder lies in the current month
    for (int i = 0; i < 12; ++i) {
        if (d_remaining > month_lengths[i]) {
            d_remaining -= month_lengths[i];
            ++month;
        } else {
            break;
        }
    }

    // d_remaining is now the calendar day of the computed month
    return make_Date(d_remaining, month, year);
}


int helios::JulianDay(int day, int month, int year) {
    return JulianDay(make_Date(day, month, year));
}

int helios::JulianDay(const Date &date) {
    int day = date.day;
    int month = date.month;
    int year = date.year;

    // Validate inputs
    if (month < 1 || month > 12) {
        helios_runtime_error("ERROR (JulianDay): Month of year is out of range (month of " + std::to_string(month) + " was given).");
    }

    // Get the correct number of days for the month (accounting for leap year in February)
    int daysInMonth[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    // Correct leap year calculation
    if (bool isLeapYear = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)) {
        daysInMonth[2] = 29;
    }

    if (day < 1 || day > daysInMonth[month]) {
        helios_runtime_error("ERROR (JulianDay): Day of month is out of range (day of " + std::to_string(day) +
                             " was given for month " + std::to_string(month) + ").");
    }

    if (year < 1000) {
        helios_runtime_error("ERROR (JulianDay): Year should be specified in YYYY format.");
    }

    // Calculate day of year
    int dayOfYear = day;
    for (int m = 1; m < month; m++) {
        dayOfYear += daysInMonth[m];
    }

    return dayOfYear;
}

bool helios::PNGHasAlpha(const char *filename) {
    if (!filename) {
        helios_runtime_error("ERROR (PNGHasAlpha): Null filename provided.");
    }

    std::string fn(filename);
    auto dot_pos = fn.find_last_of('.');
    if (dot_pos == std::string::npos) {
        helios_runtime_error("ERROR (PNGHasAlpha): File " + fn + " has no extension.");
    }
    std::string ext = fn.substr(dot_pos + 1);
    if (ext != "png" && ext != "PNG") {
        helios_runtime_error("ERROR (PNGHasAlpha): File " + fn + " is not PNG format.");
    }

    // 3) Open file with RAII
    auto fileCloser = [](FILE *f) { if (f) std::fclose(f); };
    std::unique_ptr<FILE, decltype(fileCloser)> fp(
        std::fopen(fn.c_str(), "rb"), fileCloser);
    if (!fp) {
        helios_runtime_error(
            "ERROR (PNGHasAlpha): File " + fn +
            " could not be opened for reading.");
    }

    // 4) Read & validate PNG signature
    unsigned char header[8];
    if (std::fread(header, 1, 8, fp.get()) != 8 ||
        png_sig_cmp(header, 0, 8)) {
        helios_runtime_error(
            "ERROR (PNGHasAlpha): File " + fn + " is not a valid PNG file.");
    }

    png_structp png_ptr = nullptr;
    png_infop info_ptr = nullptr;

    try {
        // 5) Create libpng read & info structs
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                         nullptr, nullptr, nullptr);
        if (!png_ptr) {
            throw std::runtime_error("png_create_read_struct failed.");
        }
        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
            png_destroy_read_struct(&png_ptr, nullptr, nullptr);
            throw std::runtime_error("png_create_info_struct failed.");
        }

        // 6) Error handling via setjmp
        if (setjmp(png_jmpbuf(png_ptr))) {
            throw std::runtime_error("Error during PNG initialization.");
        }

        // 7) Initialize IO & read info
        png_init_io(png_ptr, fp.get());
        png_set_sig_bytes(png_ptr, 8);
        png_read_info(png_ptr, info_ptr);

        // 8) Inspect color type and tRNS chunk
        png_byte color_type = png_get_color_type(png_ptr, info_ptr);
        bool has_tRNS = png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0;

        // 9) Determine alpha presence
        bool has_alpha =
                ((color_type & PNG_COLOR_MASK_ALPHA) != 0) ||
                has_tRNS;

        // 10) Clean up libpng structs
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

        return has_alpha;
    } catch (const std::exception &e) {
        // Ensure libpng structs are freed on error
        if (png_ptr) {
            if (info_ptr)
                png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
            else
                png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        }
        helios_runtime_error(std::string("ERROR (PNGHasAlpha): ") + e.what());
    }

    // Should never reach here
    return false;
}

std::vector<std::vector<bool> > helios::readPNGAlpha(const std::string &filename) {
    const std::string &fn = filename;
    auto dot = fn.find_last_of('.');
    if (dot == std::string::npos) {
        helios_runtime_error("ERROR (readPNGAlpha): File " + fn + " has no extension.");
    }
    std::string ext = fn.substr(dot + 1);
    if (ext != "png" && ext != "PNG") {
        helios_runtime_error("ERROR (readPNGAlpha): File " + fn + " is not PNG format.");
    }

    int y;

    std::vector<std::vector<bool> > mask;

    png_structp png_ptr;
    png_infop info_ptr;

    char header[8]; // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        helios_runtime_error("ERROR (readPNGAlpha): File " + std::string(filename) + " could not be opened for reading.");
    }
    size_t head = fread(header, 1, 8, fp);
    // if (png_sig_cmp(header, 0, 8)){
    //   std::cerr << "ERROR (read_png_alpha): File " << filename << " is not recognized as a PNG file." << std::endl;
    //   exit(EXIT_FAILURE);
    // }

    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!png_ptr) {
        helios_runtime_error("ERROR (readPNGAlpha): png_create_read_struct failed.");
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        helios_runtime_error("ERROR (readPNGAlpha): png_create_info_struct failed.");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        helios_runtime_error("ERROR (readPNGAlpha): init_io failed.");
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    uint width = png_get_image_width(png_ptr, info_ptr);
    uint height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0 || png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0;

    mask.resize(height);
    for (uint i = 0; i < height; i++) {
        mask.at(i).resize(width);
    }

    if (!has_alpha) {
        for (uint j = 0; j < height; ++j) {
            std::fill(mask.at(j).begin(), mask.at(j).end(), true);
        }
        fclose(fp);
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        return mask;
    }

    //  number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr))) {
        helios_runtime_error("ERROR (readPNGAlpha): read_image failed.");
    }

    auto *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (y = 0; y < height; y++)
        row_pointers[y] = (png_byte *) malloc(png_get_rowbytes(png_ptr, info_ptr));

    png_read_image(png_ptr, row_pointers);

    fclose(fp);

    for (uint j = 0; j < height; j++) {
        png_byte *row = row_pointers[j];
        for (int i = 0; i < width; i++) {
            png_byte *ba = &(row[i * 4]);
            float alpha = ba[3];
            if (alpha < 250) {
                mask.at(j).at(i) = false;
            } else {
                mask.at(j).at(i) = true;
            }
        }
    }

    for (y = 0; y < height; y++)
        png_free(png_ptr, row_pointers[y]);
    png_free(png_ptr, row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);


    return mask;
}

void helios::readPNG(const std::string &filename, uint &width, uint &height, std::vector<helios::RGBAcolor> &texture) {
    // 1) Safe extension check
    auto ext_pos = filename.find_last_of('.');
    if (ext_pos == std::string::npos) {
        helios_runtime_error("ERROR (readPNG): File " + filename + " has no extension.");
    }
    std::string ext = filename.substr(ext_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext != "png") {
        helios_runtime_error("ERROR (readPNG): File " + filename + " is not PNG format.");
    }

    png_structp png_ptr = nullptr;
    png_infop info_ptr = nullptr;

    try {
        //
        // 2) RAII for FILE*
        //
        auto fileDeleter = [](FILE *f) { if (f) fclose(f); };
        std::unique_ptr<FILE, decltype(fileDeleter)> fp(
            fopen(filename.c_str(), "rb"), fileDeleter);
        if (!fp) {
            throw std::runtime_error("File " + filename + " could not be opened.");
        }

        // 3) Read & validate PNG signature
        unsigned char header[8];
        if (fread(header, 1, 8, fp.get()) != 8) {
            throw std::runtime_error("Failed to read PNG header from " + filename);
        }
        if (png_sig_cmp(header, 0, 8)) {
            throw std::runtime_error("File " + filename + " is not a valid PNG.");
        }

        // 4) Create libpng structs
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png_ptr) {
            throw std::runtime_error("Failed to create PNG read struct.");
        }
        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
            png_destroy_read_struct(&png_ptr, nullptr, nullptr);
            throw std::runtime_error("Failed to create PNG info struct.");
        }

        // 5) libpng error handling
        if (setjmp(png_jmpbuf(png_ptr))) {
            throw std::runtime_error("Error during PNG initialization.");
        }

        // 6) Set up IO & read basic info
        png_init_io(png_ptr, fp.get());
        png_set_sig_bytes(png_ptr, 8);
        png_read_info(png_ptr, info_ptr);

        // 7) Transformations → strip 16-bit, expand palette/gray, add alpha
        png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
        png_byte color_type = png_get_color_type(png_ptr, info_ptr);

        if (bit_depth == 16) {
            png_set_strip_16(png_ptr);
        }
        if (color_type == PNG_COLOR_TYPE_PALETTE) {
            png_set_palette_to_rgb(png_ptr);
        }
        if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
            png_set_expand_gray_1_2_4_to_8(png_ptr);
        }
        if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
            png_set_tRNS_to_alpha(png_ptr);
        }
        // Ensure we have RGBA
        if (color_type == PNG_COLOR_TYPE_RGB ||
            color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_PALETTE) {
            png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
        }
        if (color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
            png_set_gray_to_rgb(png_ptr);
        }

        // 8) Handle interlacing
        png_set_interlace_handling(png_ptr);

        // 9) Apply transforms & re-fetch info
        png_read_update_info(png_ptr, info_ptr);

        // 10) Get & validate dimensions
        size_t w = png_get_image_width(png_ptr, info_ptr);
        size_t h = png_get_image_height(png_ptr, info_ptr);
        // Prevent overflow when resizing vectors
        constexpr size_t max_pixels = (std::numeric_limits<size_t>::max)() / sizeof(helios::RGBAcolor);
        if (w == 0 || h == 0 || w > max_pixels / h) {
            throw std::runtime_error(
                "Invalid image dimensions: " +
                std::to_string(w) + "×" + std::to_string(h));
        }
        width = scast<uint>(w);
        height = scast<uint>(h);

        // 11) Prepare row pointers
        size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
        if (rowbytes < width * 4) {
            throw std::runtime_error(
                "Unexpected row size: " + std::to_string(rowbytes));
        }
        std::vector<std::vector<png_byte> > row_data(
            height, std::vector<png_byte>(rowbytes));
        std::vector<png_bytep> row_pointers(height);
        for (uint y = 0; y < height; ++y) {
            row_pointers[y] = row_data[y].data();
        }

        // 12) Read the image
        if (setjmp(png_jmpbuf(png_ptr))) {
            throw std::runtime_error("Error during PNG read.");
        }
        png_read_image(png_ptr, row_pointers.data());

        // 13) Convert into normalized RGBAcolor
        texture.resize(scast<size_t>(width) * height);
        for (uint y = 0; y < height; ++y) {
            png_bytep row = row_pointers[y];
            for (uint x = 0; x < width; ++x) {
                png_bytep px = row + x * 4;
                auto &c = texture[y * width + x];
                c.r = px[0] / 255.0f;
                c.g = px[1] / 255.0f;
                c.b = px[2] / 255.0f;
                c.a = px[3] / 255.0f;
            }
        }
    } catch (const std::exception &e) {
        // Clean up libpng structs on error
        if (png_ptr) {
            if (info_ptr) png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
            else png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        }
        helios_runtime_error("ERROR (readPNG): " + std::string(e.what()));
    }

    // Normal cleanup
    if (png_ptr) {
        if (info_ptr) png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        else png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    }
}


void helios::writePNG(const std::string &filename, uint width, uint height, const std::vector<helios::RGBAcolor> &pixel_data) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        helios_runtime_error("ERROR (writePNG): failed to open image file.");
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        helios_runtime_error("ERROR (writePNG): failed to create PNG write structure.");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        helios_runtime_error("ERROR (writePNG): failed to create PNG info structure.");
    }

    if (setjmp(png_jmpbuf(png))) {
        helios_runtime_error("ERROR (writePNG): init_io failed.");
    }

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
        png,
        info,
        width,
        height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    std::vector<unsigned char *> row_pointers;
    row_pointers.resize(height);

    std::vector<std::vector<unsigned char> > data;
    data.resize(height);

    for (uint row = 0; row < height; row++) {
        data.at(row).resize(4 * width);
        for (uint col = 0; col < width; col++) {
            data.at(row).at(4 * col) = (unsigned char) round(clamp(pixel_data.at(row * width + col).r, 0.f, 1.f) * 255.f);
            data.at(row).at(4 * col + 1) = (unsigned char) round(clamp(pixel_data.at(row * width + col).g, 0.f, 1.f) * 255.f);
            data.at(row).at(4 * col + 2) = (unsigned char) round(clamp(pixel_data.at(row * width + col).b, 0.f, 1.f) * 255.f);
            data.at(row).at(4 * col + 3) = (unsigned char) round(clamp(pixel_data.at(row * width + col).a, 0.f, 1.f) * 255.f);
        }
        row_pointers.at(row) = &data.at(row).at(0);
    }

    png_write_image(png, &row_pointers.at(0));
    png_write_end(png, nullptr);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}

void helios::writePNG(const std::string &filename, uint width, uint height, const std::vector<unsigned char> &pixel_data) {

    // Convert pixel_data array into RGBcolor vector

    size_t pixels = width * height;

    std::vector<RGBAcolor> rgb_data;
    rgb_data.resize(pixels);

    size_t channels = pixel_data.size() / pixels;

    if (channels < 3) {
        helios_runtime_error("ERROR (writePNG): Pixel data must have at least 3 color channels");
    }

    // Convert pixel data into RGBA values
    for (size_t i = 0; i < pixels; i++) {
        rgb_data[i].r = float(pixel_data[i]) / 255.0f;
        rgb_data[i].g = float(pixel_data[i + pixels]) / 255.0f;
        rgb_data[i].b = float(pixel_data[i + 2 * pixels]) / 255.0f;
        rgb_data[i].a = channels > 3 ? float(pixel_data[i + 3 * pixels]) / 255.0f : 1.0f;
    }

    // Call RGB version of writePNG
    writePNG(filename, width, height, rgb_data);
}


//! Error manager for JPEG library.
struct jpg_error_mgr {
    jpeg_error_mgr pub; /* "public" fields */

    jmp_buf setjmp_buffer; /* for return to caller */
};

//! Pointer to JPEG error manager.
typedef jpg_error_mgr *jpg_error_ptr;

//! Error exit function for JPEG library.
METHODDEF(void) jpg_error_exit(j_common_ptr cinfo) {
    auto myerr = (jpg_error_ptr) cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

void helios::readJPEG(const std::string &filename, uint &width, uint &height, std::vector<helios::RGBcolor> &pixel_data) {
    auto file_extension = getFileExtension(filename);
    if (file_extension != ".jpg" && file_extension != ".JPG" && file_extension != ".jpeg" && file_extension != ".JPEG") {
        helios_runtime_error("ERROR (Context::readJPEG): File " + filename + " is not JPEG format.");
    }

    jpeg_decompress_struct cinfo{};

    jpg_error_mgr jerr{};
    FILE *infile; /* source file */
    JSAMPARRAY buffer; /*output row buffer */
    int row_stride;

    if ((infile = fopen(filename.c_str(), "rb")) == nullptr) {
        helios_runtime_error("ERROR (Context::readJPEG): File " + filename + " could not be opened. Check that the file exists and that you have permission to read it.");
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpg_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return;
    }

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);

    (void) jpeg_read_header(&cinfo, TRUE);

    (void) jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    width = cinfo.output_width;
    height = cinfo.output_height;

    if (cinfo.output_components != 3) {
        helios_runtime_error("ERROR (Context::readJPEG): Image file does not have RGB components.");
    } else if (width == 0 || height == 0) {
        helios_runtime_error("ERROR (Context::readJPEG): Image file is empty.");
    }

    pixel_data.resize(width * height);

    JSAMPLE *ba;
    int row = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);

        ba = buffer[0];

        for (int col = 0; col < row_stride; col += 3) {
            pixel_data.at(row * width + col / 3.f) = make_RGBcolor(ba[row] / 255.f, ba[row + 1] / 255.f, ba[row + 2] / 255.f);
        }

        row++;
    }

    (void) jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);

    fclose(infile);
}

helios::int2 helios::getImageResolutionJPEG(const std::string &filename) {
    auto file_extension = getFileExtension(filename);
    if (file_extension != ".jpg" && file_extension != ".JPG" && file_extension != ".jpeg" && file_extension != ".JPEG") {
        helios_runtime_error("ERROR (Context::getImageResolutionJPEG): File " + filename + " is not JPEG format.");
    }

    jpeg_decompress_struct cinfo{};

    jpg_error_mgr jerr{};
    FILE *infile; /* source file */

    if ((infile = fopen(filename.c_str(), "rb")) == nullptr) {
        helios_runtime_error("ERROR (Context::getImageResolutionJPEG): File " + filename + " could not be opened. Check that the file exists and that you have permission to read it.");
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpg_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return {0, 0};
    }

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);

    (void) jpeg_read_header(&cinfo, TRUE);
    (void) jpeg_start_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);

    return make_int2(cinfo.output_width, cinfo.output_height);
}

void helios::writeJPEG(const std::string &a_filename, uint width, uint height, const std::vector<helios::RGBcolor> &pixel_data) {
    struct my_error_mgr {
        jpeg_error_mgr pub; /* "public" fields */

        jmp_buf setjmp_buffer; /* for return to caller */
    };

    std::string filename = a_filename;
    auto file_extension = getFileExtension(filename);
    if (file_extension != ".jpg" && file_extension != ".JPG" && file_extension != ".jpeg" && file_extension != ".JPEG") {
        filename.append(".jpeg");
    }

    if (pixel_data.size() != width * height) {
        helios_runtime_error("ERROR (Context::writeJPEG): Pixel data does not have size of width*height.");
    }

    const uint bsize = 3 * width * height;
    std::vector<unsigned char> screen_shot_trans(bsize);

    size_t ii = 0;
    for (size_t i = 0; i < width * height; i++) {
        screen_shot_trans.at(ii) = (unsigned char) round(clamp(pixel_data.at(i).r, 0.f, 1.f) * 255);
        screen_shot_trans.at(ii + 1) = (unsigned char) round(clamp(pixel_data.at(i).g, 0.f, 1.f) * 255);
        screen_shot_trans.at(ii + 2) = (unsigned char) round(clamp(pixel_data.at(i).b, 0.f, 1.f) * 255);
        ii += 3;
    }

    struct jpeg_compress_struct cinfo{};

    struct jpeg_error_mgr jerr{};

    cinfo.err = jpeg_std_error(&jerr);

    /* More stuff */
    FILE *outfile; /* target file */
    JSAMPROW row_pointer; /* pointer to JSAMPLE row[s] */
    int row_stride;

    /* Now we can initialize the JPEG compression object. */
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename.c_str(), "wb")) == nullptr) {
        helios_runtime_error("ERROR (Context::writeJPEG): File " + filename + " could not be opened. Check that the file path is correct you have permission to write to it.");
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width; /* image width and height, in pixels */
    cinfo.image_height = height;
    cinfo.input_components = 3; /* # of color components per pixel */
    cinfo.in_color_space = JCS_RGB; /* colorspace of input image */

    jpeg_set_defaults(&cinfo);

    jpeg_set_quality(&cinfo, 100, TRUE /* limit to baseline-JPEG values */);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width * 3; /* JSAMPLEs per row in image_buffer */

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW) &screen_shot_trans[(cinfo.image_height - cinfo.next_scanline - 1) * row_stride];
        (void) jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    /* After finish_compress, we can close the output file. */
    fclose(outfile);

    jpeg_destroy_compress(&cinfo);
}

void helios::writeJPEG(const std::string &a_filename, uint width, uint height, const std::vector<unsigned char> &pixel_data) {

    // Convert pixel_data array into RGBcolor vector

    size_t pixels = width * height;

    std::vector<RGBcolor> rgb_data;
    rgb_data.resize(pixels);

    size_t channels = pixel_data.size() / pixels;

    if (channels < 3) {
        helios_runtime_error("ERROR (writeJPEG): Pixel data must have at least 3 color channels");
    }

    // Convert pixel data into RGB values
    for (size_t i = 0; i < pixels; i++) {
        rgb_data[i].r = scast<float>(pixel_data[i]) / 255.0f;
        rgb_data[i].g = scast<float>(pixel_data[i + pixels]) / 255.0f;
        rgb_data[i].b = scast<float>(pixel_data[i + 2 * pixels]) / 255.0f;
    }

    // Call RGB version of writeJPEG
    writeJPEG(a_filename, width, height, rgb_data);
}

// Implementation of template function to flatten a 2D vector into a 1D vector
template<typename T>
std::vector<T> helios::flatten(const std::vector<std::vector<T> > &vec) {
    std::vector<T> result;
    for (const auto &row: vec) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

template std::vector<int> helios::flatten(const std::vector<std::vector<int> > &vec);

template std::vector<uint> helios::flatten(const std::vector<std::vector<uint> > &vec);

template std::vector<float> helios::flatten(const std::vector<std::vector<float> > &vec);

template std::vector<double> helios::flatten(const std::vector<std::vector<double> > &vec);

template std::vector<helios::vec2> helios::flatten(const std::vector<std::vector<helios::vec2> > &vec);

template std::vector<helios::vec3> helios::flatten(const std::vector<std::vector<helios::vec3> > &vec);

template std::vector<helios::vec4> helios::flatten(const std::vector<std::vector<helios::vec4> > &vec);

template std::vector<helios::int2> helios::flatten(const std::vector<std::vector<helios::int2> > &vec);

template std::vector<helios::int3> helios::flatten(const std::vector<std::vector<helios::int3> > &vec);

template std::vector<helios::int4> helios::flatten(const std::vector<std::vector<helios::int4> > &vec);

template std::vector<std::string> helios::flatten(const std::vector<std::vector<std::string> > &vec);


// Implementation of template function to flatten a 3D vector into a 1D vector
template<typename T>
std::vector<T> helios::flatten(const std::vector<std::vector<std::vector<T> > > &vec) {
    std::vector<T> result;
    for (const auto &matrix: vec) {
        for (const auto &row: matrix) {
            result.insert(result.end(), row.begin(), row.end());
        }
    }
    return result;
}

template std::vector<int> helios::flatten(const std::vector<std::vector<std::vector<int> > > &vec);

template std::vector<uint> helios::flatten(const std::vector<std::vector<std::vector<uint> > > &vec);

template std::vector<float> helios::flatten(const std::vector<std::vector<std::vector<float> > > &vec);

template std::vector<double> helios::flatten(const std::vector<std::vector<std::vector<double> > > &vec);

template std::vector<helios::vec2> helios::flatten(const std::vector<std::vector<std::vector<helios::vec2> > > &vec);

template std::vector<helios::vec3> helios::flatten(const std::vector<std::vector<std::vector<helios::vec3> > > &vec);

template std::vector<helios::vec4> helios::flatten(const std::vector<std::vector<std::vector<helios::vec4> > > &vec);

template std::vector<helios::int2> helios::flatten(const std::vector<std::vector<std::vector<helios::int2> > > &vec);

template std::vector<helios::int3> helios::flatten(const std::vector<std::vector<std::vector<helios::int3> > > &vec);

template std::vector<helios::int4> helios::flatten(const std::vector<std::vector<std::vector<helios::int4> > > &vec);

template std::vector<std::string> helios::flatten(const std::vector<std::vector<std::vector<std::string> > > &vec);


// Implementation of template function to flatten a 4D vector into a 1D vector
template<typename T>
std::vector<T> helios::flatten(const std::vector<std::vector<std::vector<std::vector<T> > > > &vec) {
    std::vector<T> result;
    for (const auto &tensor: vec) {
        for (const auto &matrix: tensor) {
            for (const auto &row: matrix) {
                result.insert(result.end(), row.begin(), row.end());
            }
        }
    }
    return result;
}

template std::vector<int> helios::flatten(const std::vector<std::vector<std::vector<std::vector<int> > > > &vec);

template std::vector<uint> helios::flatten(const std::vector<std::vector<std::vector<std::vector<uint> > > > &vec);

template std::vector<float> helios::flatten(const std::vector<std::vector<std::vector<std::vector<float> > > > &vec);

template std::vector<double> helios::flatten(const std::vector<std::vector<std::vector<std::vector<double> > > > &vec);

template std::vector<helios::vec2> helios::flatten(const std::vector<std::vector<std::vector<std::vector<helios::vec2> > > > &vec);

template std::vector<helios::vec3> helios::flatten(const std::vector<std::vector<std::vector<std::vector<helios::vec3> > > > &vec);

template std::vector<helios::vec4> helios::flatten(const std::vector<std::vector<std::vector<std::vector<helios::vec4> > > > &vec);

template std::vector<helios::int2> helios::flatten(const std::vector<std::vector<std::vector<std::vector<helios::int2> > > > &vec);

template std::vector<helios::int3> helios::flatten(const std::vector<std::vector<std::vector<std::vector<helios::int3> > > > &vec);

template std::vector<helios::int4> helios::flatten(const std::vector<std::vector<std::vector<std::vector<helios::int4> > > > &vec);

template std::vector<std::string> helios::flatten(const std::vector<std::vector<std::vector<std::vector<std::string> > > > &vec);

helios::vec3 helios::spline_interp3(float u, const vec3 &x_start, const vec3 &tan_start, const vec3 &x_end, const vec3 &tan_end) {
    //Perform interpolation between two 3D points using Cubic Hermite Spline

    if (u < 0 || u > 1.f) {
        std::cout << "WARNING (spline_interp3): Clamping query point 'u' to the interval (0,1)" << std::endl;
        u = clamp(u, 0.f, 1.f);
    }

    //Basis matrix
    float B[16] = {2.f, -2.f, 1.f, 1.f, -3.f, 3.f, -2.f, -1.f, 0, 0, 1.f, 0, 1.f, 0, 0, 0};

    //Control matrix
    const float C[12] = {x_start.x, x_start.y, x_start.z, x_end.x, x_end.y, x_end.z, tan_start.x, tan_start.y, tan_start.z, tan_end.x, tan_end.y, tan_end.z};

    //Parameter vector
    const float P[4] = {u * u * u, u * u, u, 1.f};

    float R[12] = {0.f};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                R[3 * i + j] = R[3 * i + j] + B[4 * i + k] * C[3 * k + j];
            }
        }
    }

    float xq[3] = {0.f};

    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 4; k++) {
            xq[j] = xq[j] + P[k] * R[3 * k + j];
        }
    }

    return make_vec3(xq[0], xq[1], xq[2]);
}

float helios::XMLloadfloat(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    float value;
    if (strlen(field_str) == 0) {
        value = 99999;
    } else {
        if (!parse_float(field_str, value)) {
            value = 99999;
        }
    }

    return value;
}

int helios::XMLloadint(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    int value;
    if (strlen(field_str) == 0) {
        value = 99999;
    } else {
        if (!parse_int(field_str, value)) {
            value = 99999;
        }
    }

    return value;
}

std::string helios::XMLloadstring(const pugi::xml_node node, const char *field) {
    const std::string field_str = deblank(node.child_value(field));

    std::string value;
    if (field_str.empty()) {
        value = "99999";
    } else {
        value = field_str; //note: pugi loads xml data as a character.  need to separate it into int
    }

    return value;
}

helios::vec2 helios::XMLloadvec2(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::vec2 value;
    if (strlen(field_str) == 0) {
        value = make_vec2(99999, 99999);
    } else {
        value = string2vec2(field_str); //note: pugi loads xml data as a character.  need to separate it into 2 floats
    }

    return value;
}

helios::vec3 helios::XMLloadvec3(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::vec3 value;
    if (strlen(field_str) == 0) {
        value = make_vec3(99999, 99999, 99999);
    } else {
        value = string2vec3(field_str); //note: pugi loads xml data as a character.  need to separate it into 3 floats
    }

    return value;
}

helios::vec4 helios::XMLloadvec4(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::vec4 value;
    if (strlen(field_str) == 0) {
        value = make_vec4(99999, 99999, 99999, 99999);
    } else {
        value = string2vec4(field_str); //note: pugi loads xml data as a character.  need to separate it into 4 floats
    }

    return value;
}

helios::int2 helios::XMLloadint2(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::int2 value;
    if (strlen(field_str) == 0) {
        value = make_int2(99999, 99999);
    } else {
        value = string2int2(field_str); //note: pugi loads xml data as a character.  need to separate it into 2 ints
    }

    return value;
}

helios::int3 helios::XMLloadint3(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::int3 value;
    if (strlen(field_str) == 0) {
        value = make_int3(99999, 99999, 99999);
    } else {
        value = string2int3(field_str); //note: pugi loads xml data as a character.  need to separate it into 3 ints
    }

    return value;
}

helios::int4 helios::XMLloadint4(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::int4 value;
    if (strlen(field_str) == 0) {
        value = make_int4(99999, 99999, 99999, 99999);
    } else {
        value = string2int4(field_str); //note: pugi loads xml data as a character.  need to separate it into 4 ints
    }

    return value;
}

helios::RGBcolor helios::XMLloadrgb(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::RGBAcolor value;
    if (strlen(field_str) == 0) {
        value = make_RGBAcolor(1, 1, 1, 0);
    } else {
        value = string2RGBcolor(field_str); //note: pugi loads xml data as a character.  need to separate it into 3 floats
    }

    return make_RGBcolor(value.r, value.g, value.b);
}

helios::RGBAcolor helios::XMLloadrgba(const pugi::xml_node node, const char *field) {
    const char *field_str = node.child_value(field);

    helios::RGBAcolor value;
    if (strlen(field_str) == 0) {
        value = make_RGBAcolor(1, 1, 1, 0);
    } else {
        value = string2RGBcolor(field_str); //note: pugi loads xml data as a character.  need to separate it into 3 floats
    }

    return value;
}

float helios::fzero(float (*function)(float value, std::vector<float> &variables, const void *parameters), std::vector<float> &variables, const void *parameters, float init_guess, float err_tol, int max_iterations) {
    float T = init_guess;

    float T_old_old = 1.1f * init_guess;

    float T_old = init_guess;

    float resid_old = function(T_old, variables, parameters);
    float resid_old_old = function(T_old_old, variables, parameters);

    float resid = 100;
    float err = resid;
    uint iter = 0;
    while (err > err_tol && iter < max_iterations) {
        if (resid_old == resid_old_old) { //this condition will cause NaN
            err = 0;
            break;
        }

        T = fabs((T_old_old * resid_old - T_old * resid_old_old) / (resid_old - resid_old_old));

        resid = function(T, variables, parameters);

        resid_old_old = resid_old;
        resid_old = resid;

        err = fabs(T_old - T_old_old) / fabs(T_old_old);

        T_old_old = T_old;
        T_old = T;

        iter++;
    }

    if (err > err_tol) {
        printf("WARNING: fzero solution did not converge.\n");
    }

    return T;
}

float helios::interp1(const std::vector<helios::vec2> &points, float x) {
    //Ensure that no 2 adjacent x values are equal, and that x values are monotonically increasing
    constexpr float EPSILON{1.0E-5};
    for (std::size_t i = 1; i < points.size(); ++i) {
        float deltaX{std::abs(points[i].x - points[i - 1].x)};
        if (deltaX < EPSILON) {
            if (deltaX > -EPSILON) {
                helios_runtime_error("ERROR (interp1): Adjacent X points cannot be equal.");
            } else {
                helios_runtime_error("ERROR (interp1): X points must increase monotonically.");
            }
        }
    }

    //Define a lambda that returns true if the x value
    //of a point pair is < the caller's x value
    auto lessThan = [](const vec2 &point, float x) {
        return point.x < x;
    };

    //Find the first table entry whose value is >= caller's x value
    const auto iter = std::lower_bound(points.cbegin(), points.cend(), x, lessThan);

    //If the caller's X value is greater than the largest
    //X value in the table, we can't interpolate.
    if (iter == points.cend()) {
        return (points.cend() - 1)->y;
    }

    //If the caller's X value is less than the smallest X value in the table,
    //we can't interpolate.
    if (iter == points.cbegin() && x <= points.cbegin()->x) {
        return points.cbegin()->y;
    }

    //We can interpolate!
    float upperX{iter->x};
    float upperY{iter->y};
    float lowerX{(iter - 1)->x};
    float lowerY{(iter - 1)->y};

    float deltaY{upperY - lowerY};
    float deltaX{upperX - lowerX};

    return lowerY + ((x - lowerX) / deltaX) * deltaY;
}

float helios::point_distance(const helios::vec3 &p1, const helios::vec3 &p2) {
    return (p1 - p2).magnitude();
}

std::string helios::getFileExtension(const std::string &filepath) {
    std::filesystem::path output_path_fs = filepath;
    return output_path_fs.extension().string();
}

std::string helios::getFileStem(const std::string &filepath) {
    std::filesystem::path output_path_fs = filepath;
    return output_path_fs.stem().string();
}

std::string helios::getFileName(const std::string &filepath) {
    std::filesystem::path output_path_fs = filepath;
    return output_path_fs.filename().string();
}

std::string helios::getFilePath(const std::string &filepath, bool trailingslash) {
    std::filesystem::path output_path_fs = filepath;
    std::string output_path = output_path_fs.parent_path().string();
    if (trailingslash) {
        if (output_path.find_last_of('/') != output_path.length() - 1) {
            output_path += "/";
        }
    }

    return output_path;
}

bool helios::validateOutputPath(std::string &output_path, const std::vector<std::string> &allowable_file_extensions) {
    if (output_path.empty()) { //path was empty
        return false;
    }

    std::filesystem::path output_path_fs = output_path;

    std::string output_file = output_path_fs.filename().string();
    std::string output_file_ext = output_path_fs.extension().string();
    std::string output_dir = output_path_fs.parent_path().string();

    if (output_file.empty()) { //path was a directory without a file

        // Make sure directory has a trailing slash
        if (output_dir.find_last_of('/') != output_dir.length() - 1) {
            output_path += "/";
        }
    }

    // Create the output directory if it does not exist
    if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
        if (!std::filesystem::create_directory(output_dir)) {
            return false;
        }
    }

    if (!output_file.empty() && !allowable_file_extensions.empty()) {
        //validate file extension
        bool valid_extension = false;
        for (const auto &ext: allowable_file_extensions) {
            if (output_file_ext == ext) {
                valid_extension = true;
                break;
            }
        }
        if (!valid_extension) {
            return false;
        }
    }

    return true;
}

std::vector<float> helios::importVectorFromFile(const std::string &filepath) {
    std::ifstream stream(filepath.c_str());

    if (!stream.is_open()) {
        helios_runtime_error("ERROR (helios::importVectorFromFile): File " + filepath + " could not be opened for reading. Check that it exists and that you have permission to read it.");
    }

    std::istream_iterator<float> start(stream), end;
    std::vector<float> vec(start, end);
    return vec;
}

float helios::sample_Beta_distribution(float mu, float nu, std::minstd_rand0 *generator) {
    // 1) draw two independent Gamma variates:
    //    X ~ Gamma(α=ν, 1),  Y ~ Gamma(β=μ, 1)
    std::gamma_distribution<float> dist_nu(nu, 1.0);
    std::gamma_distribution<float> dist_mu(mu, 1.0);

    float X = dist_nu(*generator);
    float Y = dist_mu(*generator);

    // 2) form the Beta = X/(X+Y)
    float b = X / (X + Y);

    // 3) rescale to θ_L = (π/2)*b
    return 0.5f * PI_F * b;
}

// Complete elliptic integral of the first kind via the arithmetic–geometric mean (AGM)
float compute_elliptic_integral_first_kind(float e) {
    // K(e) = π / (2 * AGM(1, sqrt(1 - e^2)))
    float a = 1.0f;
    float b = std::sqrt(1.0f - e * e);
    for (int iter = 0; iter < 10; ++iter) {
        float an = 0.5f * (a + b);
        float bn = std::sqrt(a * b);
        a = an;
        b = bn;
    }
    return PI_F / (2.0f * a);
}

// Ellipsoidal PDF for leaf azimuth distribution
// phi: sample angle [0,2π), e: eccentricity, phi0: rotation offset, K_e: precomputed ellip. integral
float evaluate_ellipsoidal_azimuth_PDF(float phi, float e, float phi0, float K_e) {
    float d = phi - phi0;
    float c2 = (1.f - e * e) * std::cos(d) * std::cos(d) + std::sin(d) * std::sin(d);
    return 1.f / (4.f * K_e * std::sqrt(c2));
}

// Sample phi from ellipsoidal distribution via rejection sampling
float helios::sample_ellipsoidal_azimuth(
    float e,
    float phi0_degrees,
    std::minstd_rand0 *generator
) {
    // sanity‐check
    if (e < 0.f || e > 1.f) {
        helios_runtime_error("ERROR (helios::sample_ellipsoidal_azimuth): Eccentricity must be in [0,1].");
    }

    // convert rotation offset to radians
    float phi0 = deg2rad(phi0_degrees);

    // ellipse semiaxes: a=1, b = sqrt(1 - e^2)
    float a = 1.f;
    float b = std::sqrt(1.f - e * e);

    // sample the ellipse parameter t uniformly in [0,2π)
    std::uniform_real_distribution<float> distT(0.f, 2.f * PI_F);
    float t = distT(*generator);

    // point on the ellipse boundary
    float x = a * std::cos(t);
    float y = b * std::sin(t);

    // compute its polar angle
    float phi = std::atan2(y, x) + phi0;

    // wrap into [0,2π)
    if (phi < 0.f)
        phi += 2.f * PI_F;
    else if (phi >= 2.f * PI_F)
        phi -= 2.f * PI_F;

    return phi;
}

// float helios::sample_ellipsoidal_azimuth(
//     float e,
//     float phi0_degrees,
//     std::minstd_rand0 *generator
// ) {
//     // 1) sanity‐check
//     if (e < 0.f || e > 1.f) {
//         helios_runtime_error(
//             "ERROR (helios::sample_ellipsoidal_azimuth): "
//             "eccentricity must be in [0,1]."
//         );
//     }
//
//     // 2) trivial uniform case
//     std::uniform_real_distribution<float> distPhi(0.f, 2.f * PI_F);
//     if (e == 0.f) {
//         return distPhi(*generator);
//     }
//
//     // 3) precompute rotation offset
//     float phi0 = deg2rad(phi0_degrees);
//
//     // 4) rejection sampling: envelope = uniform φ, accept with ratio = (1–e²)/denominator
//     std::uniform_real_distribution<float> dist01(0.f, 1.f);
//     while (true) {
//         float phi = distPhi(*generator);
//         float d   = phi - phi0;
//         // wrap to [–π,π) for numerical stability
//         if      (d < -PI_F) d += 2.f*PI_F;
//         else if (d >=  PI_F) d -= 2.f*PI_F;
//
//         // denominator = (1–e²)·cos²d + sin²d
//         float c = std::cos(d), s = std::sin(d);
//         float denom = (1.f - e*e)*c*c + s*s;
//
//         // acceptance ratio ∈ (0,1]
//         float ratio = (1.f - e*e) / denom;
//
//         if (dist01(*generator) <= ratio) {
//             // wrap phi back into [0,2π)
//             if      (phi < 0.f)        phi += 2.f*PI_F;
//             else if (phi >= 2.f*PI_F)  phi -= 2.f*PI_F;
//             return phi;
//         }
//         // otherwise retry
//     }
// }
