/** \file "NormalMapMath.h" Normal-map math shared by the host and the CUDA ray-tracing backends.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#ifndef HELIOS_RADIATION_NORMAL_MAP_MATH_H
#define HELIOS_RADIATION_NORMAL_MAP_MATH_H

// This header is the specification of how the radiation model applies normal maps. The CUDA backends (OptiX 6.5 and 8.1) include it
// directly, the host uses it for checks and tests, and shaders/common/normal_map.glsl is a line-for-line port for the Vulkan backend:
// change both together.
//
// Packed buffer (built by packNormalMaps() in RayTracingTypes.cpp), 32-bit words:
//   w[0] P   number of primitives, or 0 when no primitive has a normal map (the devices then skip everything here)
//   w[1] M   number of maps
//   w[2] TB  word index where the texels begin
//   w[3]     format version (1)
//   w[4 ..]  map index of each primitive as 16 bits, two per word with the lower half first; 0xFFFF for none
//   then 3 words per map: width, height, and the first word of its texels relative to TB
//   texels   two per word with the lower half first; texel = uint8(x) | uint8(y) << 8 with x, y signed 8-bit normalized (value / 127)
//            and z = sqrt(1 - x^2 - y^2), rows from the top of the image (the convention of masks and class maps)
//
// Tangent space (OpenGL / glTF / Blender): +x along increasing u, +y along increasing v (up in the image), +z along the facet normal.
// The bottom face of a primitive uses the negated shading normal (a thin sheet: a ridge on the top face is a groove on the bottom).
//
// Canonical primitives and their edges (c0 = (T[0],T[4],T[8]) and c1 = (T[1],T[5],T[9]) are the first two columns of the transform):
//   patch:    canonical square [-0.5,0.5]^2, corners 0..3 at (-,-), (+,-), (+,+), (-,+); E1 = c0, E2 = c1; texture coordinates
//             uv(s) = uv0 + (s.x+0.5) a + (s.y+0.5) b with a = uv1-uv0, b = uv2-uv1 (the corners of the whole image without uv)
//   triangle: canonical corners (0,0,0), (0,1,0), (1,1,0); E1 = c1, E2 = c0 + c1; uv = uv0 + beta a + gamma b with a = uv1-uv0,
//             b = uv2-uv0, where beta and gamma are the weights of corners 1 and 2
//   The facet (top) normal is normalize(cross(E1, E2)), as the ray-generation programs compute it.

#include <cstdint>

#if defined(__CUDACC__)
#define NM_FUNC __host__ __device__ __forceinline__
#else
#include <cmath>
#define NM_FUNC inline
#endif

namespace helios_nm {

    //! Minimal 3-vector shared by host and device code
    struct Vec3 {
        float x, y, z;
    };

    NM_FUNC Vec3 make(float x, float y, float z) {
        Vec3 v;
        v.x = x;
        v.y = y;
        v.z = z;
        return v;
    }
    NM_FUNC Vec3 add(const Vec3 &a, const Vec3 &b) {
        return make(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    NM_FUNC Vec3 sub(const Vec3 &a, const Vec3 &b) {
        return make(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    NM_FUNC Vec3 scale(const Vec3 &a, float s) {
        return make(a.x * s, a.y * s, a.z * s);
    }
    NM_FUNC float dot(const Vec3 &a, const Vec3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    NM_FUNC Vec3 cross(const Vec3 &a, const Vec3 &b) {
        return make(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    NM_FUNC Vec3 normalize(const Vec3 &a) {
        const float length = sqrtf(dot(a, a));
        return length > 0.f ? scale(a, 1.f / length) : a;
    }

    //! Value of the map index meaning "no normal map"
    constexpr uint32_t NO_MAP = 0xFFFFu;

    //! Map index of the primitive at position pos, or NO_MAP. Words is any type with operator[](uint32_t) returning a 32-bit word.
    template<typename Words>
    NM_FUNC uint32_t mapIndex(const Words &words, uint32_t pos) {
        if (words[0] == 0u) {
            return NO_MAP;
        }
        const uint32_t word = words[4u + pos / 2u];
        return (pos & 1u) ? (word >> 16) : (word & 0xFFFFu);
    }

    //! Unit tangent-space normal of a packed 16-bit texel
    NM_FUNC Vec3 decodeTexel(uint32_t texel) {
        const int xi = int(int8_t(uint8_t(texel & 0xFFu)));
        const int yi = int(int8_t(uint8_t((texel >> 8) & 0xFFu)));
        const float x = float(xi) / 127.f;
        const float y = float(yi) / 127.f;
        const float zz = 1.f - x * x - y * y;
        return normalize(make(x, y, zz > 0.f ? sqrtf(zz) : 0.f));
    }

    //! Normal of texel (column, row from the top) of map m
    template<typename Words>
    NM_FUNC Vec3 texel(const Words &words, uint32_t m, int column, int row) {
        const uint32_t P = words[0];
        const uint32_t header = 4u + (P + 1u) / 2u + 3u * m;
        const uint32_t width = words[header];
        const uint32_t index = uint32_t(row) * width + uint32_t(column);
        const uint32_t word = words[words[2] + words[header + 2u] + index / 2u];
        return decodeTexel((index & 1u) ? (word >> 16) : (word & 0xFFFFu));
    }

    //! Normal of map m at texture coordinate (u,v): the texel at column floor((W-1)u) and row floor((H-1)(1-v)), clamped to the image
    template<typename Words>
    NM_FUNC Vec3 fetchNearest(const Words &words, uint32_t m, float u, float v) {
        const uint32_t header = 4u + (words[0] + 1u) / 2u + 3u * m;
        const int width = int(words[header]);
        const int height = int(words[header + 1u]);
        int column = int(floorf(float(width - 1) * u));
        int row = int(floorf(float(height - 1) * (1.f - v)));
        column = column < 0 ? 0 : (column > width - 1 ? width - 1 : column);
        row = row < 0 ? 0 : (row > height - 1 ? height - 1 : row);
        return texel(words, m, column, row);
    }

    //! Normal of map m bilinearly interpolated at (u,v): texel centres at ((column+0.5)/(W-1), 1-(row+0.5)/(H-1)), clamped to the edge, renormalized
    template<typename Words>
    NM_FUNC Vec3 fetchBilinear(const Words &words, uint32_t m, float u, float v) {
        const uint32_t header = 4u + (words[0] + 1u) / 2u + 3u * m;
        const int width = int(words[header]);
        const int height = int(words[header + 1u]);
        const float fx = float(width - 1) * u - 0.5f;
        const float fy = float(height - 1) * (1.f - v) - 0.5f;
        const int x0 = int(floorf(fx));
        const int y0 = int(floorf(fy));
        const float wx = fx - float(x0);
        const float wy = fy - float(y0);
        const int c0 = x0 < 0 ? 0 : (x0 > width - 1 ? width - 1 : x0);
        const int c1 = x0 + 1 < 0 ? 0 : (x0 + 1 > width - 1 ? width - 1 : x0 + 1);
        const int r0 = y0 < 0 ? 0 : (y0 > height - 1 ? height - 1 : y0);
        const int r1 = y0 + 1 < 0 ? 0 : (y0 + 1 > height - 1 ? height - 1 : y0 + 1);
        const Vec3 top = add(scale(texel(words, m, c0, r0), 1.f - wx), scale(texel(words, m, c1, r0), wx));
        const Vec3 bottom = add(scale(texel(words, m, c0, r1), 1.f - wx), scale(texel(words, m, c1, r1), wx));
        return normalize(add(scale(top, 1.f - wy), scale(bottom, wy)));
    }

    //! Edges E1, E2 of a primitive's canonical shape in world space (see the header comment), from its row-major transform
    NM_FUNC void primitiveEdges(const float *T, uint32_t primitive_type, Vec3 &E1, Vec3 &E2) {
        const Vec3 c0 = make(T[0], T[4], T[8]);
        const Vec3 c1 = make(T[1], T[5], T[9]);
        if (primitive_type == 1u) {
            E1 = c1;
            E2 = add(c0, c1);
        } else {
            E1 = c0;
            E2 = c1;
        }
    }

    //! Columns a, b of the texture-coordinate Jacobian of a primitive (see the header comment); has_uv false gives the whole image for a patch
    NM_FUNC void uvJacobian(uint32_t primitive_type, bool has_uv, float u0, float v0, float u1, float v1, float u2, float v2, float &ax, float &ay, float &bx, float &by) {
        if (!has_uv) {
            if (primitive_type == 1u) {
                // A triangle without (u,v) samples the image at (beta + gamma, gamma), as the masks do
                ax = 1.f;
                ay = 0.f;
                bx = 1.f;
                by = 1.f;
            } else {
                ax = 1.f;
                ay = 0.f;
                bx = 0.f;
                by = 1.f;
            }
            return;
        }
        ax = u1 - u0;
        ay = v1 - v0;
        if (primitive_type == 1u) {
            bx = u2 - u0;
            by = v2 - v0;
        } else {
            bx = u2 - u1;
            by = v2 - v1;
        }
    }

    //! Texture coordinate (u,v), with v up the image, of the point at canonical coordinates (sx, sy) of a primitive (see the header comment): for a patch (sx, sy) in [-0.5,0.5]^2,
    //! for a triangle sx = gamma and sy = beta + gamma. ax..by are the columns of uvJacobian(), and (u0, v0) the texture coordinate of corner 0 ((0,0) without uv).
    NM_FUNC void texcoord(uint32_t primitive_type, float u0, float v0, float ax, float ay, float bx, float by, float sx, float sy, float &u, float &v) {
        float s1, s2;
        if (primitive_type == 1u) {
            s1 = sy - sx; // beta
            s2 = sx; // gamma
        } else {
            s1 = sx + 0.5f;
            s2 = sy + 0.5f;
        }
        u = u0 + s1 * ax + s2 * bx;
        v = v0 + s1 * ay + s2 * by;
    }

    //! Orthonormal tangent frame of a primitive about a base normal N (unit, on the top side of the facet)
    struct Frame {
        Vec3 t; //!< Unit tangent along increasing u
        Vec3 b; //!< Unit bitangent along increasing v
        Vec3 n; //!< The base normal
        bool valid; //!< False if the texture coordinates are degenerate, in which case the shading normal is the base normal
    };

    //! Tangent frame from the primitive's edges and texture-coordinate Jacobian
    NM_FUNC Frame buildFrame(const Vec3 &E1, const Vec3 &E2, float ax, float ay, float bx, float by, const Vec3 &N) {
        Frame frame;
        frame.n = N;
        frame.t = N;
        frame.b = N;
        frame.valid = false;
        const float det = ax * by - ay * bx;
        const float extent = (ax * ax + ay * ay) > (bx * bx + by * by) ? (ax * ax + ay * ay) : (bx * bx + by * by);
        if (!(fabsf(det) > 1e-12f * extent) || extent == 0.f) {
            return frame;
        }
        // dP/du = (b.y E1 - a.y E2) / det
        const Vec3 dPdu = scale(sub(scale(E1, by), scale(E2, ay)), 1.f / det);
        const Vec3 tangent = sub(dPdu, scale(N, dot(N, dPdu)));
        const float length2 = dot(tangent, tangent);
        if (!(length2 > 1e-20f)) {
            return frame;
        }
        frame.t = scale(tangent, 1.f / sqrtf(length2));
        // Mirrored texture coordinates (det < 0) put the v direction on the other side of the tangent
        frame.b = scale(cross(N, frame.t), det > 0.f ? 1.f : -1.f);
        frame.valid = true;
        return frame;
    }

    //! Shading normal on the top face: the tangent-space normal n_ts carried into the frame
    NM_FUNC Vec3 perturb(const Frame &frame, const Vec3 &n_ts) {
        if (!frame.valid) {
            return frame.n;
        }
        return normalize(add(add(scale(frame.t, n_ts.x), scale(frame.b, n_ts.y)), scale(frame.n, n_ts.z)));
    }

    //! Shadow-terminator softening of Chiang, Li and Burley (2019), for a face with geometric normal n_f and shading normal m_f lit from L
    /** Both normals are oriented to the face; returns 0 when either cosine is not positive. */
    NM_FUNC float terminatorSoftening(const Vec3 &n_f, const Vec3 &m_f, const Vec3 &L) {
        const float mu_m = dot(m_f, L);
        const float mu_g = dot(n_f, L);
        const float c = dot(n_f, m_f);
        if (mu_m <= 0.f || mu_g <= 0.f || c <= 1e-6f) {
            return 0.f;
        }
        float G = mu_g / (mu_m * c);
        G = G < 1.f ? G : 1.f;
        return G * (1.f + G * (1.f - G));
    }

} // namespace helios_nm

#endif // HELIOS_RADIATION_NORMAL_MAP_MATH_H
