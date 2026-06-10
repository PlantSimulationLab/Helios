/** \file "triangulation_cdt.h" Adapter exposing the CDT library through an
    s_hull_pro-compatible interface for the Helios LiDAR plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_LIDAR_TRIANGULATION_CDT_H
#define HELIOS_LIDAR_TRIANGULATION_CDT_H

#include "CDT.h"

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal point/triangle types and de-duplication.
//
// These were historically provided by the s_hull_pro library (now removed).
// LiDAR.cpp only uses Shx.{id,r,c}, Triad.{a,b,c}, and de_duplicate(), so the
// minimal definitions are kept here alongside the CDT adapter. The names and
// field semantics are preserved so the call sites in LiDAR.cpp are unchanged.
// ---------------------------------------------------------------------------

/// 2D input point: id is the caller's index, (r,c) are the 2D coordinates
/// (zenith, azimuth) to triangulate.
struct Shx {
    int id = -1;
    float r = 0.f, c = 0.f;
};

/// Triangle as three vertex indices (a,b,c) into the input point vector.
struct Triad {
    int a = 0, b = 0, c = 0;
};

/// Helper used only by de_duplicate() to sort points by (r,c).
struct Dupex {
    int id;
    float r, c;
};

inline bool operator<(const Dupex &a, const Dupex &b) {
    // Epsilon-based comparison for cross-platform-consistent sort order.
    const float epsilon = 1e-9f;
    if (std::abs(a.r - b.r) < epsilon)
        return a.c < b.c;
    return a.r < b.r;
}

/// Remove duplicate points (identical r and c) from \p pts in place.
/// \param[in,out] pts  Points; duplicates are erased.
/// \param[out]    outx Indices (into the original \p pts) that were removed.
/// \return Number of removed points.
inline int de_duplicate(std::vector<Shx> &pts, std::vector<int> &outx) {

    int nump = (int) pts.size();
    std::vector<Dupex> dpx;
    Dupex d;
    for (int k = 0; k < nump; k++) {
        d.r = pts[k].r;
        d.c = pts[k].c;
        d.id = k;
        dpx.push_back(d);
    }

    std::sort(dpx.begin(), dpx.end());

    for (int k = 0; k < nump - 1; k++) {
        if (dpx[k].r == dpx[k + 1].r && dpx[k].c == dpx[k + 1].c) {
            outx.push_back(dpx[k + 1].id);
        }
    }

    if (outx.size() == 0)
        return (0);

    std::sort(outx.begin(), outx.end());

    int nx = (int) outx.size();
    for (int k = nx - 1; k >= 0; k--) {
        pts.erase(pts.begin() + outx[k]);
    }

    return (nx);
}

/**
 * \brief Compute a 2D Delaunay triangulation of \p pts using the CDT library.
 *
 * The i-th point in \p pts maps to input index i, and the returned Triad
 * vertex indices (a/b/c) are indices back into \p pts, so the
 * Delaunay_inds.at(triad.a) mapping in LiDAR.cpp is straightforward.
 *
 * This function does NOT de-duplicate internally -- de-duplication is performed
 * by the caller (de_duplicate()) prior to this call, preserving the
 * pts[i] <-> caller-index correspondence. CDT throws on duplicate/degenerate
 * input; such throws are caught here and reported as failure (return 0) so the
 * caller can skip the offending scan.
 *
 * \param[in]  pts    Points to triangulate. Shx.r is used as the x-coordinate
 *                    (zenith) and Shx.c as the y-coordinate (azimuth).
 * \param[out] triads Resulting triangles (a/b/c vertex indices into \p pts).
 * \return 1 on success, 0 on failure.
 */
inline int triangulate_CDT(const std::vector<Shx> &pts, std::vector<Triad> &triads) {

    triads.clear();

    // CDT requires at least 3 points to form a triangle.
    if (pts.size() < 3) {
        return 0;
    }

    try {
        CDT::Triangulation<float> cdt;

        std::vector<CDT::V2d<float>> verts;
        verts.reserve(pts.size());
        for (const Shx &p: pts) {
            // s_hull uses (r=zenith, c=azimuth) as the 2D coordinates.
            verts.push_back(CDT::V2d<float>(p.r, p.c));
        }

        cdt.insertVertices(verts);
        cdt.eraseSuperTriangle();

        // After eraseSuperTriangle(), CDT subtracts the super-triangle vertex
        // offset, so triangle vertex indices reference the original input
        // order (index i == pts[i]).
        triads.reserve(cdt.triangles.size());
        for (const CDT::Triangle &t: cdt.triangles) {
            Triad tri;
            tri.a = static_cast<int>(t.vertices[0]);
            tri.b = static_cast<int>(t.vertices[1]);
            tri.c = static_cast<int>(t.vertices[2]);
            triads.push_back(tri);
        }

    } catch (...) {
        // CDT throws on duplicate/degenerate input. Report failure (fail-fast)
        // and let the caller's retry/recovery logic handle it.
        triads.clear();
        return 0;
    }

    return 1;
}

#endif // HELIOS_LIDAR_TRIANGULATION_CDT_H
