/** \file "CollisionGeometry.h" Exact closest-point and distance queries between segments and triangles, shared by the collision queries and by other plug-ins.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.0 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/

#ifndef HELIOS_COLLISION_GEOMETRY_H
#define HELIOS_COLLISION_GEOMETRY_H

#include "global.h"

namespace helios {
    namespace collision {

        //! Closest points between two segments, returning the squared distance between them
        /**
         * Minimizes the distance between a point of segment p1-q1 and a point of segment p2-q2. A zero-length segment is treated as a point, so this also gives point-to-segment and point-to-point
         * distances. For (nearly) parallel segments one of the equally close pairs is returned.
         * \param[in] p1 Start of the first segment
         * \param[in] q1 End of the first segment
         * \param[in] p2 Start of the second segment
         * \param[in] q2 End of the second segment
         * \param[out] closest1 Point of the first segment nearest the second
         * \param[out] closest2 Point of the second segment nearest the first
         * \return Squared distance between \a closest1 and \a closest2
         */
        float closestPointsOnSegments(const helios::vec3 &p1, const helios::vec3 &q1, const helios::vec3 &p2, const helios::vec3 &q2, helios::vec3 &closest1, helios::vec3 &closest2);

        //! Closest point of a triangle to a point
        /**
         * Found from the Voronoi region of the triangle (vertex, edge or face) that the point projects into. A degenerate (zero-area) triangle is handled through its edges.
         * \param[in] p Query point
         * \param[in] a First vertex of the triangle
         * \param[in] b Second vertex of the triangle
         * \param[in] c Third vertex of the triangle
         * \return Point of the triangle nearest \a p
         */
        helios::vec3 closestPointOnTriangle(const helios::vec3 &p, const helios::vec3 &a, const helios::vec3 &b, const helios::vec3 &c);

        //! Closest points between a segment and a triangle, returning the squared distance between them
        /**
         * If the segment crosses the triangle the distance is zero and both points are the crossing. Otherwise the closest pair joins a segment endpoint to the triangle, or the segment to one of the
         * triangle's edges. The gap between a capsule of radius r about the segment and the triangle is sqrt of the returned value minus r.
         * \param[in] p Start of the segment
         * \param[in] q End of the segment
         * \param[in] a First vertex of the triangle
         * \param[in] b Second vertex of the triangle
         * \param[in] c Third vertex of the triangle
         * \param[out] on_segment Point of the segment nearest the triangle
         * \param[out] on_triangle Point of the triangle nearest the segment
         * \return Squared distance between \a on_segment and \a on_triangle
         */
        float closestPointsSegmentTriangle(const helios::vec3 &p, const helios::vec3 &q, const helios::vec3 &a, const helios::vec3 &b, const helios::vec3 &c, helios::vec3 &on_segment, helios::vec3 &on_triangle);

    } // namespace collision
} // namespace helios

#endif
