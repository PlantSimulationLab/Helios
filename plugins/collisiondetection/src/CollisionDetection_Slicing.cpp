/** \file "CollisionDetection_Slicing.cpp" Source file for primitive slicing and voxel intersection operations.

    Copyright (C) 2016-2026 Brian Bailey, Eric Kent

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

#include "CollisionDetection.h"

using namespace helios;

// -------- GEOMETRIC UTILITY FUNCTIONS --------

bool CollisionDetection::approxSame(float a, float b, float absTol, float relTol) const {
    return fabs(a - b) <= absTol || fabs(a - b) <= relTol * (std::max(fabs(a), fabs(b)));
}

bool CollisionDetection::approxSame(const helios::vec3 &a, const helios::vec3 &b, float absTol) const {
    return fabs(a.x - b.x) <= absTol && fabs(a.y - b.y) <= absTol && fabs(a.z - b.z) <= absTol;
}

helios::vec2 CollisionDetection::interpolate_texture_UV_to_slice_point(const helios::vec3 &p1, const helios::vec2 &uv1, const helios::vec3 &p2, const helios::vec2 &uv2, const helios::vec3 &ps) const {
    // uv coordinate that will be output
    helios::vec2 uvs;

    float Dxyz = sqrtf(powf(p2.x - p1.x, 2.0) + powf(p2.y - p1.y, 2.0) + powf(p2.z - p1.z, 2.0)); // distance between edge vertex xyz coordinates
    float Dxyzs = sqrtf(powf(ps.x - p1.x, 2.0) + powf(ps.y - p1.y, 2.0) + powf(ps.z - p1.z, 2.0)); // distance between slice point and first vertex xyz coordinates

    float absTol = pow(10, -6);

    float F = (Dxyzs / Dxyz);
    if (F > 1.0) {
        helios_runtime_error("ERROR (CollisionDetection::interpolate_texture_UV_to_slice_point): slice point is not between the two end points.");
    } else if (approxSame(p1, ps, absTol)) {
        // then the slice point is the same as the first vertex
        uvs = make_vec2(uv1.x, uv1.y);
        return uvs;
    } else if (approxSame(p2, ps, absTol)) {
        // then the slice point is the same as the second vertex
        uvs = make_vec2(uv2.x, uv2.y);
        return uvs;
    }

    // The slice point lies on the edge, so its UV coordinate lies the same fraction F of the way from uv1 to uv2
    uvs = uv1 + F * (uv2 - uv1);

    return uvs;
}

// -------- PRIMITIVE SLICING FUNCTIONS --------

std::vector<uint> CollisionDetection::slicePrimitive(uint UUID, const std::vector<helios::vec3> &voxel_face_vertices, helios::WarningAggregator &warnings) {

    // vector of UUIDs that will be output
    std::vector<uint> resulting_UUIDs;

    if (voxel_face_vertices.size() < 3) {
        helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): voxel_face_vertices must contain at least three points.");
    }

    helios::vec3 face_normal = cross(voxel_face_vertices.at(1) - voxel_face_vertices.at(0), voxel_face_vertices.at(2) - voxel_face_vertices.at(1));
    face_normal.normalize();

    std::vector<helios::vec3> primitive_vertices = this->context->getPrimitiveVertices(UUID);
    helios::vec3 primitive_normal = this->context->getPrimitiveNormal(UUID);
    primitive_normal.normalize();

    helios::RGBAcolor primitive_color = this->context->getPrimitiveColorRGBA(UUID);

    std::string texa;
    const char *tex;
    texa = this->context->getPrimitiveTextureFile(UUID);
    tex = texa.c_str();
    bool primitiveHasTexture = !texa.empty();

    // get the area of the original primitive for comparison with the area of the sliced primitives later
    float original_area = this->context->getPrimitiveArea(UUID);

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // get the points where each edge of the primitive crosses the plane of the voxel face

    // Each edge is intersected with the face plane directly from the signed distances of its end points to the plane. A
    // vertex lying on the plane has a signed distance of exactly zero for an axis-aligned face, so it is returned exactly
    // rather than as the nearly-equal result of a line-line intersection, which at coordinates far from the origin can
    // differ from the vertex by more than the vertex-matching tolerance below.
    const float face_offset = face_normal * voxel_face_vertices.at(0);

    std::vector<helios::vec3> slice_points; // points where an edge crosses or touches the face plane
    std::vector<uint> slice_points_edge_ID; // edge i runs from vertex i to vertex i+1 (wrapping around)
    uint vertex_index = 0; // index for cases where one slice point is on a vertex (used for patch cases only)

    const uint vertex_count = primitive_vertices.size();
    for (uint i = 0; i < vertex_count; i++) {
        const helios::vec3 &edge_start = primitive_vertices.at(i);
        const helios::vec3 &edge_end = primitive_vertices.at((i + 1) % vertex_count);
        const float start_distance = face_normal * edge_start - face_offset;
        const float end_distance = face_normal * edge_end - face_offset;

        if (start_distance == 0.f && end_distance == 0.f) {
            continue; // the edge lies in the face plane; its end points are reported by the two neighbouring edges
        } else if (start_distance == 0.f) {
            slice_points.push_back(edge_start);
        } else if (end_distance == 0.f) {
            slice_points.push_back(edge_end);
        } else if ((start_distance < 0.f) != (end_distance < 0.f)) {
            const float t = start_distance / (start_distance - end_distance);
            slice_points.push_back(edge_start + t * (edge_end - edge_start));
        } else {
            continue; // both end points on the same side of the plane
        }
        slice_points_edge_ID.push_back(i);
    }

    // can be 0, 1, 2, 3, or 4 (0 and 2 are most common)
    uint initial_slice_points_size = slice_points.size();
    // std::cout << "initial_slice_points_size = " << initial_slice_points_size << std::endl;

    float absTol = pow(10, -6);
    float relTol = pow(10, -20);


    // the primitive did not intersect with the voxel face
    if (initial_slice_points_size == 0) {
        resulting_UUIDs.push_back(UUID);
        return resulting_UUIDs;

    } else if (initial_slice_points_size == 1) {
        // the primitive intersected with the face at a single point (a corner) - no slicing needed
        resulting_UUIDs.push_back(UUID);
        if (this->printmessages) {
            std::cout << "the primitive intersected with the face at a single point (a corner) - no slicing needed" << std::endl;
        }
        return resulting_UUIDs;
    } else if (initial_slice_points_size == 2) {

        // This is the usual case
        // just check to see if the two slice points are approximately at two vertices for edge cases here

        // the primitive intersected with the face along an edge - no need to slice
        if (slice_points_edge_ID.at(0) == slice_points_edge_ID.at(1)) {
            resulting_UUIDs.push_back(UUID);
            if (this->printmessages) {
                std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
            }
            return resulting_UUIDs;
        }

        if (primitive_vertices.size() == 4) {
            if ((approxSame(slice_points.at(0), primitive_vertices.at(0), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(1), absTol)) ||
                (approxSame(slice_points.at(0), primitive_vertices.at(1), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(0), absTol)) ||
                (approxSame(slice_points.at(0), primitive_vertices.at(1), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(2), absTol)) ||
                (approxSame(slice_points.at(0), primitive_vertices.at(2), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(1), absTol)) ||
                (approxSame(slice_points.at(0), primitive_vertices.at(2), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(3), absTol)) ||
                (approxSame(slice_points.at(0), primitive_vertices.at(3), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(2), absTol)) ||
                (approxSame(slice_points.at(0), primitive_vertices.at(3), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(0), absTol)) ||
                (approxSame(slice_points.at(0), primitive_vertices.at(0), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(3), absTol))) {
                if (this->printmessages) {
                    std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
                }
                resulting_UUIDs.push_back(UUID);
                return resulting_UUIDs;
            }

        } else if (primitive_vertices.size() == 3) {

            if ((approxSame(slice_points.at(0), primitive_vertices.at(0), absTol) || approxSame(slice_points.at(0), primitive_vertices.at(1), absTol) || approxSame(slice_points.at(0), primitive_vertices.at(2), absTol)) &&
                (approxSame(slice_points.at(1), primitive_vertices.at(0), absTol) || approxSame(slice_points.at(1), primitive_vertices.at(1), absTol) || approxSame(slice_points.at(1), primitive_vertices.at(2), absTol))) {
                resulting_UUIDs.push_back(UUID);
                if (this->printmessages) {
                    std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
                }
                return resulting_UUIDs;
            }
        }


        // now that edge cases are taken care of,
        // for each slice point, if it is approximately the same as a vertex, set it to that vertex
        for (uint j = 0; j < primitive_vertices.size(); j++) {
            for (uint i = 0; i < slice_points.size(); i++) {
                // distance between slice point and primitive vertex
                float Dxyza = sqrtf(powf(primitive_vertices.at(j).x - slice_points.at(i).x, 2.0) + powf(primitive_vertices.at(j).y - slice_points.at(i).y, 2.0) + powf(primitive_vertices.at(j).z - slice_points.at(i).z, 2.0));
                if (approxSame(Dxyza, float(0.0), absTol, relTol)) {
                    slice_points.at(i) = primitive_vertices.at(j);
                }
            }
        }


    } else if (initial_slice_points_size == 3) {

        // if there are 3 slice points, this probably means that two of the points are very close to each other,
        //  at or approximately at one of the primitive's vertices
        //  in this case, if the primitive is a triangle, then it should be sliced into two triangles, not the usual three
        //  in case the primitive is a patch, then it should be sliced into 3 triangles if this occurs at only one vertex

        vec3 non_vertex_slice_point;
        uint non_vertex_slice_edge_ID = 0;
        vec3 vertex_slice_point;
        bool vertex_found = false;
        bool non_vertex_found = false;
        std::set<uint> matched_vertices;

        for (uint bb = 0; bb < slice_points.size(); bb++) {
            bool this_point_vert_test = false;
            for (uint cc = 0; cc < primitive_vertices.size(); cc++) {
                bool vert_test = approxSame(slice_points.at(bb), primitive_vertices.at(cc), absTol);
                if (vert_test) {
                    this_point_vert_test = true;
                    vertex_slice_point = primitive_vertices.at(cc);
                    vertex_index = cc;
                    vertex_found = true;
                    matched_vertices.insert(cc);
                }
            }

            if (this_point_vert_test == false) {
                non_vertex_slice_point = slice_points.at(bb);
                non_vertex_slice_edge_ID = slice_points_edge_ID.at(bb);
                non_vertex_found = true;
            }
        }

        // Every slice point is (within tolerance) a vertex: the plane passes through one vertex and crosses the opposite
        // edge right next to one of its end points, as happens for the thin slivers earlier cuts leave behind. The plane
        // then runs along an edge, so there is nothing to cut - the same outcome as when it touches only vertices above.
        // Two diagonal corners of a patch would need the diagonal split instead, which is not handled here.
        if (vertex_found && !non_vertex_found && matched_vertices.size() == 2) {
            const uint first_vertex = *matched_vertices.begin();
            const uint second_vertex = *matched_vertices.rbegin();
            const bool vertices_share_an_edge = primitive_vertices.size() == 3 || second_vertex - first_vertex == 1 || (first_vertex == 0 && second_vertex == primitive_vertices.size() - 1);
            if (vertices_share_an_edge) {
                resulting_UUIDs.push_back(UUID);
                return resulting_UUIDs;
            }
        }
        if (!vertex_found || !non_vertex_found) {
            std::ostringstream geometry;
            geometry << std::setprecision(9) << "vertices";
            for (const vec3 &vertex: primitive_vertices) {
                geometry << " (" << vertex.x << ", " << vertex.y << ", " << vertex.z << ")";
            }
            geometry << ", face plane n = (" << face_normal.x << ", " << face_normal.y << ", " << face_normal.z << "), n.p = " << face_offset << ", slice points";
            for (const vec3 &point: slice_points) {
                geometry << " (" << point.x << ", " << point.y << ", " << point.z << ")";
            }
            helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): primitive " + std::to_string(UUID) + " crosses the voxel face at 3 points, but they are not one vertex and one edge point as expected: " + geometry.str());
        }
        slice_points.resize(2);
        slice_points.at(0) = non_vertex_slice_point;
        slice_points_edge_ID.at(0) = non_vertex_slice_edge_ID;
        slice_points.at(1) = vertex_slice_point;

        // std::cout << "slice_points.at(0) = " << slice_points.at(0) << std::endl;
        // std::cout << "slice_points.at(1) = " << slice_points.at(1) << std::endl;
        // std::cout << "slice_points_edge_ID.at(0) = " << slice_points_edge_ID.at(0) << std::endl;
        // std::cout << "vertex_index = " << vertex_index << std::endl;

    } else if (initial_slice_points_size == 4) {
        // if the voxel face splits a patch diagonally, then only 2 triangles should be produced instead of the usual four
        bool vertex_found = false;
        for (uint bb = 0; bb < slice_points.size(); bb++) {
            for (uint cc = 0; cc < primitive_vertices.size(); cc++) {
                if (approxSame(slice_points.at(bb), primitive_vertices.at(cc), absTol)) {
                    vertex_index = cc;
                    vertex_found = true;
                }
            }
        }
        if (!vertex_found) {
            helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): primitive " + std::to_string(UUID) + " crosses the voxel face at 4 points, none of which is a vertex.");
        }
        slice_points.resize(2);
    } else {
        helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): more than 5 slice points detected - invalid geometry.");
    }

    // determine which side of the plane vertex 0 is on and use that to determine the sign of the buffer to add to the face coordinate
    //  the buffer ensures that the vertices will be categorized into grid cells correctly
    //  note that some of these checks are based on the assumption of a axis aligned grid - would need to be re-worked if implementing rotated grid

    helios::vec3 face_coordinate = make_vec3(fabs(face_normal.x) * voxel_face_vertices.at(0).x, fabs(face_normal.y) * voxel_face_vertices.at(0).y, fabs(face_normal.z) * voxel_face_vertices.at(0).z);
    // float buffer_value = powf(float(10), float(-6));
    float buffer_value = powf(float(10), float(-5));
    helios::vec3 buffer = make_vec3(0, 0, 0);
    if (fabs(face_normal.x) > 0.5) {
        if (primitive_vertices.at(0).x < face_coordinate.x) {
            buffer = make_vec3(float(-1) * buffer_value, 0, 0);
        } else if (primitive_vertices.at(0).x > face_coordinate.x) {
            buffer = make_vec3(buffer_value, 0, 0);
        } else {
            if (this->printmessages) {
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }

    } else if (fabs(face_normal.y) > 0.5) {
        if (primitive_vertices.at(0).y < face_coordinate.y) {
            buffer = make_vec3(0, float(-1) * buffer_value, 0);
        } else if (primitive_vertices.at(0).y > face_coordinate.y) {
            buffer = make_vec3(0, buffer_value, 0);
        } else {
            if (this->printmessages) {
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }

    } else if (fabs(face_normal.z) > 0.5) {
        if (primitive_vertices.at(0).z < face_coordinate.z) {
            buffer = make_vec3(0, 0, float(-1) * buffer_value);
        } else if (primitive_vertices.at(0).z > face_coordinate.z) {
            buffer = make_vec3(0, 0, buffer_value);
        } else {
            if (this->printmessages) {
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }
    }

    // UUIDs for triangles to be created below
    uint t0;
    uint t1;
    uint t2;
    uint t3;

    // if a resulting triangle area is below this value, delete it
    float minArea = pow(10, -13);

    // use this diagnostic code to locate where a particular triangle is being created
    // (uncomment the print out far below)
    uint diag_1 = 0;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // if the primitive isn't texture masked
    if (primitiveHasTexture == false) {

        if (primitive_vertices.size() == 3) {
            // split into three triangles (usual case)
            if (initial_slice_points_size == 2) {
                if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 1;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 2;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 3;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 4;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 5;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color); //
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color); //
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 6;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color); //
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color); //
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
                if (this->context->getPrimitiveArea(t2) < minArea) {
                    this->context->deletePrimitive(t2);
                } else {
                    resulting_UUIDs.push_back(t2);
                }


            } else if (initial_slice_points_size == 3) {
                // split into two triangles instead of three since a vertex falls on the slicing face

                if (slice_points_edge_ID.at(0) == 0) {
                    diag_1 = 7;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 1) {
                    diag_1 = 8;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 2) {
                    diag_1 = 9;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
            }

        } else if (primitive_vertices.size() == 4) {

            // split into four triangles (usual case)
            if (initial_slice_points_size == 2) {
                // cases where intersection points are on opposite sides
                if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 10;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);


                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 11;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);

                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 3)) {
                    diag_1 = 12;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);

                } else if ((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 13;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);

                    // cases where intersection points are on adjacent sides
                } else if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 3)) {
                    diag_1 = 14;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 15;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);

                } else if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 16;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 17;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 18;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 19;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 20;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 3)) {
                    diag_1 = 21;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
                if (this->context->getPrimitiveArea(t2) < minArea) {
                    this->context->deletePrimitive(t2);
                } else {
                    resulting_UUIDs.push_back(t2);
                }
                if (this->context->getPrimitiveArea(t3) < minArea) {
                    this->context->deletePrimitive(t3);
                } else {
                    resulting_UUIDs.push_back(t3);
                }

            } else if (initial_slice_points_size == 3) {
                // split into three triangles instead of four since one vertex falls on the slicing face

                if (slice_points_edge_ID.at(0) == 0 && vertex_index == 2) {
                    diag_1 = 22;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 0 && vertex_index == 3) {
                    diag_1 = 23;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 1 && vertex_index == 3) {
                    diag_1 = 24;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 1 && vertex_index == 0) {
                    diag_1 = 25;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 2 && vertex_index == 1) {
                    diag_1 = 26;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 2 && vertex_index == 0) {
                    diag_1 = 27;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 3 && vertex_index == 2) {
                    diag_1 = 28;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                } else if (slice_points_edge_ID.at(0) == 3 && vertex_index == 1) {
                    diag_1 = 29;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
                if (this->context->getPrimitiveArea(t2) < minArea) {
                    this->context->deletePrimitive(t2);
                } else {
                    resulting_UUIDs.push_back(t2);
                }

            } else if (initial_slice_points_size == 4) {
                // split into two triangles instead of four since both vertices fall on the slicing face
                if (vertex_index == 0 || vertex_index == 2) {
                    diag_1 = 30;
                    t0 = this->context->addTriangle(primitive_vertices.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t1 = this->context->addTriangle(primitive_vertices.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, primitive_color);

                } else if (vertex_index == 1 || vertex_index == 3) {
                    diag_1 = 31;
                    t0 = this->context->addTriangle(primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    t1 = this->context->addTriangle(primitive_vertices.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
            }
        }

    } else if (primitiveHasTexture) {

        // get uv coordinates of the vertices
        std::vector<helios::vec2> v_uv = this->context->getPrimitiveTextureUV(UUID);

        // get uv coordinates of the intersection points
        std::vector<helios::vec2> ip_uv;
        ip_uv.resize(2);

        if (primitive_vertices.size() == 3) {

            // split into three triangles (usual case)
            if (initial_slice_points_size == 2) {
                for (uint i = 0; i < slice_points.size(); i++) {
                    // vectors to hold point coordinates and uv coordinates for the points on the current point's edge
                    helios::vec3 point_0;
                    helios::vec3 point_1;
                    helios::vec2 point_0uv;
                    helios::vec2 point_1uv;
                    helios::vec2 point_uv;

                    if (slice_points_edge_ID.at(i) == 0) {
                        point_0 = primitive_vertices.at(0);
                        point_1 = primitive_vertices.at(1);
                        point_0uv = v_uv.at(0);
                        point_1uv = v_uv.at(1);
                    } else if (slice_points_edge_ID.at(i) == 1) {
                        point_0 = primitive_vertices.at(1);
                        point_1 = primitive_vertices.at(2);
                        point_0uv = v_uv.at(1);
                        point_1uv = v_uv.at(2);
                    } else if (slice_points_edge_ID.at(i) == 2) {
                        point_0 = primitive_vertices.at(2);
                        point_1 = primitive_vertices.at(0);
                        point_0uv = v_uv.at(2);
                        point_1uv = v_uv.at(0);
                    }

                    ip_uv.at(i) = interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(i));

                    if (ip_uv.at(0).x < 0 || ip_uv.at(0).x > 1 || ip_uv.at(0).y < 0 || ip_uv.at(0).y > 1) {
                        helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): texture UV coordinates for UUID " + std::to_string(UUID) + " are out of valid range [0,1].");
                    }
                }

                if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 101;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(1), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(2));
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(0));
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 102;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1), v_uv.at(2), v_uv.at(0));
                } else if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 103;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(0));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 104;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(2));
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1), v_uv.at(2));
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 105;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(0));
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1));
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 106;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(2));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0), v_uv.at(1));
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
                if (this->context->getPrimitiveArea(t2) < minArea) {
                    this->context->deletePrimitive(t2);
                } else {
                    resulting_UUIDs.push_back(t2);
                }

                // split into two triangles instead of three since a vertex falls on the slicing face
                // the non-vertex slice point is slice_points.at(0) and the vertex slice point is slice_points.at(1)
            } else if (initial_slice_points_size == 3) {

                // std::cout << "initial_slice_points_size = " << initial_slice_points_size << std::endl;

                // vectors to hold point coordinates and uv coordinates for the points on the current point's edge for interpolation
                helios::vec3 point_0;
                helios::vec3 point_1;
                helios::vec2 point_0uv;
                helios::vec2 point_1uv;
                helios::vec2 point_uv;

                if (slice_points_edge_ID.at(0) == 0) {
                    point_0 = primitive_vertices.at(0);
                    point_1 = primitive_vertices.at(1);
                    point_0uv = v_uv.at(0);
                    point_1uv = v_uv.at(1);
                    ip_uv.at(1) = v_uv.at(2); // this sets the uv coordinate for the vertex slice point

                } else if (slice_points_edge_ID.at(0) == 1) {
                    point_0 = primitive_vertices.at(1);
                    point_1 = primitive_vertices.at(2);
                    point_0uv = v_uv.at(1);
                    point_1uv = v_uv.at(2);
                    ip_uv.at(1) = v_uv.at(0); // this sets the uv coordinate for the vertex slice point
                } else if (slice_points_edge_ID.at(0) == 2) {
                    point_0 = primitive_vertices.at(2);
                    point_1 = primitive_vertices.at(0);
                    point_0uv = v_uv.at(2);
                    point_1uv = v_uv.at(0);
                    ip_uv.at(1) = v_uv.at(1); // this sets the uv coordinate for the vertex slice point
                }

                // UV for non-vertex slice point
                ip_uv.at(0) = interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(0));

                if (ip_uv.at(0).x < 0 || ip_uv.at(0).x > 1 || ip_uv.at(0).y < 0 || ip_uv.at(0).y > 1) {
                    helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): texture UV coordinates for UUID " + std::to_string(UUID) + " are out of valid range [0,1].");
                }

                if (slice_points_edge_ID.at(0) == 0) {
                    diag_1 = 107;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(0));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(1), ip_uv.at(1));
                } else if (slice_points_edge_ID.at(0) == 1) {
                    diag_1 = 108;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(1));
                } else if (slice_points_edge_ID.at(0) == 2) {
                    diag_1 = 109;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(2));
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
            }


        } else if (primitive_vertices.size() == 4) {

            // it seems patches that are not explicitly set up with texture UV coordinates just don't have them
            // so set the default here
            if (v_uv.size() == 0) {
                std::vector<helios::vec2> uv{make_vec2(0, 0), make_vec2(1, 0), make_vec2(1, 1), make_vec2(0, 1)};
                v_uv = uv;
            }

            // split into four triangles (usual case)
            if (initial_slice_points_size == 2) {
                // for each intersection point, choose the patch vertices on the corresponding edge
                for (uint i = 0; i < 2; i++) {
                    helios::vec3 point_0;
                    helios::vec3 point_1;
                    helios::vec2 point_uv;
                    helios::vec2 point_0uv;
                    helios::vec2 point_1uv;

                    if (slice_points_edge_ID.at(i) == 0) {
                        point_0 = primitive_vertices.at(0);
                        point_1 = primitive_vertices.at(1);
                        point_0uv = v_uv.at(0);
                        point_1uv = v_uv.at(1);
                    } else if (slice_points_edge_ID.at(i) == 1) {
                        point_0 = primitive_vertices.at(1);
                        point_1 = primitive_vertices.at(2);
                        point_0uv = v_uv.at(1);
                        point_1uv = v_uv.at(2);
                    } else if (slice_points_edge_ID.at(i) == 2) {
                        point_0 = primitive_vertices.at(2);
                        point_1 = primitive_vertices.at(3);
                        point_0uv = v_uv.at(2);
                        point_1uv = v_uv.at(3);
                    } else if (slice_points_edge_ID.at(i) == 3) {
                        point_0 = primitive_vertices.at(3);
                        point_1 = primitive_vertices.at(0);
                        point_0uv = v_uv.at(3);
                        point_1uv = v_uv.at(0);
                    }

                    ip_uv.at(i) = interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(i));

                    if (ip_uv.at(i).x < 0 || ip_uv.at(i).x > 1 || ip_uv.at(i).y < 0 || ip_uv.at(i).y > 1) {
                        helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): texture UV coordinates for UUID " + std::to_string(UUID) + " are out of valid range [0,1].");
                    }
                }

                // cases where intersection points are on opposite sides
                if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 110;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(0));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1), v_uv.at(3), v_uv.at(0));
                    t3 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 111;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(2));
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(0));
                    t3 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1), v_uv.at(2));
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 3)) {
                    diag_1 = 112;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(3), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0), v_uv.at(1));
                    t3 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(3));
                } else if ((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 113;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(1), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(3));
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1));
                    t3 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(1), v_uv.at(2), v_uv.at(3));
                    // cases where intersection points are on adjacent sides
                } else if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 3)) {
                    diag_1 = 114;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(0));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                    t3 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(1), v_uv.at(2), v_uv.at(3));
                } else if ((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 115;
                    t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(2));
                    t2 = this->context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1), v_uv.at(2));
                    t3 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(3));
                } else if ((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 116;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(1), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(3));
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, tex, ip_uv.at(1), v_uv.at(2), v_uv.at(3));
                    t3 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(0));
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)) {
                    diag_1 = 117;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(3), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(3));
                    t3 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1), v_uv.at(3), v_uv.at(0));
                } else if ((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 118;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(0));
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1), v_uv.at(03), v_uv.at(0));
                    t3 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1));
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)) {
                    diag_1 = 119;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(2));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(0));
                    t3 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0), v_uv.at(1));
                } else if ((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 2)) {
                    diag_1 = 120;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(3));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(1), ip_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1));
                    t3 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(1), v_uv.at(1), v_uv.at(2));
                } else if ((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 3)) {
                    diag_1 = 121;
                    t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(3), ip_uv.at(1));
                    t1 = this->context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1), v_uv.at(1));
                    t2 = this->context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0), v_uv.at(1));
                    t3 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
                if (this->context->getPrimitiveArea(t2) < minArea) {
                    this->context->deletePrimitive(t2);
                } else {
                    resulting_UUIDs.push_back(t2);
                }
                if (this->context->getPrimitiveArea(t3) < minArea) {
                    this->context->deletePrimitive(t3);
                } else {
                    resulting_UUIDs.push_back(t3);
                }


            } else if (initial_slice_points_size == 3) {

                // for the first intersection point (index 0), choose the endpoints of the edge to interpolate UV between
                // for this case where the other intersection point is at a primitive vertex, that vertex UV will be used
                for (uint i = 0; i < 1; i++) {
                    helios::vec3 point_0;
                    helios::vec3 point_1;
                    helios::vec2 point_uv;
                    helios::vec2 point_0uv;
                    helios::vec2 point_1uv;

                    if (slice_points_edge_ID.at(i) == 0) {
                        point_0 = primitive_vertices.at(0);
                        point_1 = primitive_vertices.at(1);
                        point_0uv = v_uv.at(0);
                        point_1uv = v_uv.at(1);
                    } else if (slice_points_edge_ID.at(i) == 1) {
                        point_0 = primitive_vertices.at(1);
                        point_1 = primitive_vertices.at(2);
                        point_0uv = v_uv.at(1);
                        point_1uv = v_uv.at(2);
                    } else if (slice_points_edge_ID.at(i) == 2) {
                        point_0 = primitive_vertices.at(2);
                        point_1 = primitive_vertices.at(3);
                        point_0uv = v_uv.at(2);
                        point_1uv = v_uv.at(3);
                    } else if (slice_points_edge_ID.at(i) == 3) {
                        point_0 = primitive_vertices.at(3);
                        point_1 = primitive_vertices.at(0);
                        point_0uv = v_uv.at(3);
                        point_1uv = v_uv.at(0);
                    }

                    // std::cout << "point_0 = " << point_0 << std::endl;
                    // std::cout << "point_0uv = " << point_0uv << std::endl;
                    // std::cout << "point_1 = " << point_1 << std::endl;
                    // std::cout << "point_1uv = " << point_1uv << std::endl;
                    // std::cout << "i = " <<  i << std::endl;
                    // std::cout << "slice_points.at(i) = " << slice_points.at(i) << std::endl;
                    // std::cout << "slice_points.size() = " << slice_points.size() << std::endl;
                    // std::cout << "slice_points_edge_ID.at(i) = " << slice_points_edge_ID.at(i) << std::endl;

                    ip_uv.at(i) = interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(i));

                    if (ip_uv.at(i).x < 0 || ip_uv.at(i).x > 1 || ip_uv.at(i).y < 0 || ip_uv.at(i).y > 1) {
                        helios_runtime_error("ERROR (CollisionDetection::slicePrimitive): texture UV coordinates for UUID " + std::to_string(UUID) + " are out of valid range [0,1].");
                    }


                    if (slice_points_edge_ID.at(0) == 0 && vertex_index == 2) {
                        diag_1 = 122;
                        t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(0));
                        t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                        t2 = this->context->addTriangle(primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, v_uv.at(2), v_uv.at(3), v_uv.at(0));
                    } else if (slice_points_edge_ID.at(0) == 0 && vertex_index == 3) {
                        diag_1 = 123;
                        t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(0));
                        t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(3));
                        t2 = this->context->addTriangle(primitive_vertices.at(3) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, v_uv.at(3), v_uv.at(1), v_uv.at(2));
                    } else if (slice_points_edge_ID.at(0) == 1 && vertex_index == 3) {
                        diag_1 = 124;
                        t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(1));
                        t1 = this->context->addTriangle(primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, v_uv.at(3), v_uv.at(0), v_uv.at(1));
                        t2 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(3));
                    } else if (slice_points_edge_ID.at(0) == 1 && vertex_index == 0) {
                        diag_1 = 125;
                        t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1));
                        t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(0) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(0));
                        t2 = this->context->addTriangle(primitive_vertices.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, v_uv.at(0), v_uv.at(2), v_uv.at(3));
                    } else if (slice_points_edge_ID.at(0) == 2 && vertex_index == 1) {
                        diag_1 = 126;
                        t0 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                        t1 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(1));
                        t2 = this->context->addTriangle(primitive_vertices.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, v_uv.at(1), v_uv.at(3), v_uv.at(0));
                    } else if (slice_points_edge_ID.at(0) == 2 && vertex_index == 0) {
                        diag_1 = 127;
                        t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(0));
                        t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(0) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(2));
                        t2 = this->context->addTriangle(primitive_vertices.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, v_uv.at(0), v_uv.at(1), v_uv.at(2));
                    } else if (slice_points_edge_ID.at(0) == 3 && vertex_index == 2) {
                        diag_1 = 128;
                        t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(2));
                        t1 = this->context->addTriangle(primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, v_uv.at(2), v_uv.at(0), v_uv.at(1));
                        t2 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(3));
                    } else if (slice_points_edge_ID.at(0) == 3 && vertex_index == 1) {
                        diag_1 = 129;
                        t0 = this->context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1));
                        t1 = this->context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(3));
                        t2 = this->context->addTriangle(primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, v_uv.at(1), v_uv.at(2), v_uv.at(3));
                    }

                    // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                    if (this->context->getPrimitiveArea(t0) < minArea) {
                        this->context->deletePrimitive(t0);
                    } else {
                        resulting_UUIDs.push_back(t0);
                    }
                    if (this->context->getPrimitiveArea(t1) < minArea) {
                        this->context->deletePrimitive(t1);
                    } else {
                        resulting_UUIDs.push_back(t1);
                    }
                    if (this->context->getPrimitiveArea(t2) < minArea) {
                        this->context->deletePrimitive(t2);
                    } else {
                        resulting_UUIDs.push_back(t2);
                    }
                }

            } else if (initial_slice_points_size == 4) {

                if (vertex_index == 0 || vertex_index == 2) {
                    diag_1 = 130;
                    t0 = this->context->addTriangle(primitive_vertices.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, v_uv.at(0), v_uv.at(1), v_uv.at(2));
                    t1 = this->context->addTriangle(primitive_vertices.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, tex, v_uv.at(0), v_uv.at(2), v_uv.at(3));

                } else if (vertex_index == 1 || vertex_index == 3) {
                    diag_1 = 131;
                    t0 = this->context->addTriangle(primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, v_uv.at(1), v_uv.at(2), v_uv.at(3));
                    t1 = this->context->addTriangle(primitive_vertices.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, v_uv.at(1), v_uv.at(3), v_uv.at(0));
                }

                // delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if (this->context->getPrimitiveArea(t0) < minArea) {
                    this->context->deletePrimitive(t0);
                } else {
                    resulting_UUIDs.push_back(t0);
                }
                if (this->context->getPrimitiveArea(t1) < minArea) {
                    this->context->deletePrimitive(t1);
                } else {
                    resulting_UUIDs.push_back(t1);
                }
            }
        }
    }

    // print this out to find where a certain triangle is created
    // std::cout << "diag_1 = " << diag_1 << std::endl;

    // copy over primitive data to the new triangles
    for (uint i = 0; i < resulting_UUIDs.size(); i++) {
        this->context->copyPrimitiveData(UUID, resulting_UUIDs.at(i));
        uint parentID = this->context->getPrimitiveParentObjectID(UUID);
        if (parentID > 0 && (this->context->getObjectType(parentID) == helios::OBJECT_TYPE_TILE || this->context->getObjectType(parentID) == helios::OBJECT_TYPE_ADAPTIVE_TILE)) {
            this->context->setPrimitiveParentObjectID(resulting_UUIDs.at(i), 0);
        } else {
            this->context->setPrimitiveParentObjectID(resulting_UUIDs.at(i), parentID);
        }
        if (this->context->isPrimitiveTextureColorOverridden(UUID)) {
            this->context->overridePrimitiveTextureColor(resulting_UUIDs.at(i));
        }
    }

    // compare original and resulting primitive areas to make sure they approximately match
    float resulting_area = this->context->sumPrimitiveSurfaceArea(resulting_UUIDs);
    float pdiff_area = (resulting_area - original_area) / original_area * 100.0;
    float pdiff_area_abs = fabs(pdiff_area);
    if (pdiff_area_abs > 1) {
        warnings.addWarning("slice_area_mismatch", "Sum of slice areas does not equal area of original primitive (UUID = " + std::to_string(UUID) + ", original area = " + std::to_string(original_area) +
                                                           ", resulting area = " + std::to_string(resulting_area) + ", percent difference = " + std::to_string(pdiff_area) + "%)");
    }

    // compare original and resulting primitive normals to make sure they match
    absTol = 0.5;
    relTol = 0.4;
    for (uint aa = 0; aa < resulting_UUIDs.size(); aa++) {
        helios::vec3 this_normal = this->context->getPrimitiveNormal(resulting_UUIDs.at(aa));
        this_normal.normalize();
        if (!approxSame(primitive_normal.x, this_normal.x, absTol, relTol) || !approxSame(primitive_normal.y, this_normal.y, absTol, relTol) || !approxSame(primitive_normal.z, this_normal.z, absTol, relTol)) {
            warnings.addWarning("slice_normal_mismatch", "UUID " + std::to_string(resulting_UUIDs.at(aa)) + " normal (" + std::to_string(this_normal.x) + ", " + std::to_string(this_normal.y) + ", " + std::to_string(this_normal.z) +
                                                                 ") does not match original normal (" + std::to_string(primitive_normal.x) + ", " + std::to_string(primitive_normal.y) + ", " + std::to_string(primitive_normal.z) + ")");
        }
    }

    // delete the original primitive
    this->context->deletePrimitive(UUID);

    return resulting_UUIDs;
}

std::vector<uint> CollisionDetection::slicePrimitivesUsingGrid(const std::vector<uint> &UUIDs, const helios::vec3 &grid_center, const helios::vec3 &grid_size, const helios::int3 &grid_divisions) {

    // Create warning aggregator
    helios::WarningAggregator warnings;
    warnings.setEnabled(this->printmessages);

    // set up the grid
    std::vector<std::vector<helios::vec3>> grid_face_vertices;
    helios::vec3 grid_min = make_vec3(grid_center.x - grid_size.x * 0.5, grid_center.y - grid_size.y * 0.5, grid_center.z - grid_size.z * 0.5);
    helios::vec3 grid_max = make_vec3(grid_center.x + grid_size.x * 0.5, grid_center.y + grid_size.y * 0.5, grid_center.z + grid_size.z * 0.5);
    helios::vec3 grid_spacing = make_vec3(grid_size.x / grid_divisions.x, grid_size.y / grid_divisions.y, grid_size.z / grid_divisions.z);

    // faces in the y-z plane (change x)
    for (uint k = 0; k < (grid_divisions.x + 1); k++) {
        std::vector<helios::vec3> this_face_vertices;
        this_face_vertices.push_back(make_vec3(grid_min.x + k * grid_spacing.x, grid_min.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_min.x + k * grid_spacing.x, grid_min.y, grid_max.z));
        this_face_vertices.push_back(make_vec3(grid_min.x + k * grid_spacing.x, grid_max.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_min.x + k * grid_spacing.x, grid_max.y, grid_max.z));
        grid_face_vertices.push_back(this_face_vertices);
    }

    // faces in the x-z plane (change y)
    for (uint k = 0; k < (grid_divisions.y + 1); k++) {
        std::vector<helios::vec3> this_face_vertices;
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_min.y + k * grid_spacing.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_min.y + k * grid_spacing.y, grid_max.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_min.y + k * grid_spacing.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_min.y + k * grid_spacing.y, grid_max.z));
        grid_face_vertices.push_back(this_face_vertices);
    }

    // faces in the x-z plane (change y)
    for (uint k = 0; k < (grid_divisions.z + 1); k++) {
        std::vector<helios::vec3> this_face_vertices;
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_min.y, grid_min.z + k * grid_spacing.z));
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_max.y, grid_min.z + k * grid_spacing.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_min.y, grid_min.z + k * grid_spacing.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_max.y, grid_min.z + k * grid_spacing.z));
        grid_face_vertices.push_back(this_face_vertices);
    }

    if (this->printmessages) {
        std::cout << UUIDs.size() << " input primitives" << std::endl;
        std::cout << grid_face_vertices.size() << " grid faces used for slicing" << std::endl;
        std::cout << grid_divisions.x * grid_divisions.y * grid_divisions.z << " total grid cells" << std::endl;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // do an initial classification of primitives into grid cells based on if all their vertices fall into a given voxel

    this->grid_cells.clear();
    this->grid_cells.resize(grid_divisions.x);
    for (uint i = 0; i < grid_divisions.x; i++) {
        this->grid_cells[i].resize(grid_divisions.y);
        for (uint j = 0; j < grid_divisions.y; j++) {
            this->grid_cells[i][j].resize(grid_divisions.z);
        }
    }

    // initially set all UUIDs as outside any voxel
    this->context->setPrimitiveData(UUIDs, "cell_ID", int(-1));

    // vectors for UUIDs that do and do not need to be sliced
    std::vector<uint> UUIDs_to_slice;
    std::vector<uint> UUIDs_no_slice;

    // Only cells overlapping a primitive's vertex bounding box can contain any of its vertices, so the classification
    // loops visit just those cells (in the same k-j-i order as a full scan, so the first matching cell is unchanged).
    // The range is padded by one cell on each side so vertices on, or within tolerance of, a cell face are still tested.
    auto overlappingCellRange = [&](const std::vector<helios::vec3> &verts, helios::int3 &first_cell, helios::int3 &last_cell) {
        helios::vec3 verts_min = verts.front();
        helios::vec3 verts_max = verts.front();
        for (const helios::vec3 &vert: verts) {
            verts_min = make_vec3(std::min(verts_min.x, vert.x), std::min(verts_min.y, vert.y), std::min(verts_min.z, vert.z));
            verts_max = make_vec3(std::max(verts_max.x, vert.x), std::max(verts_max.y, vert.y), std::max(verts_max.z, vert.z));
        }
        first_cell.x = std::max(0, int(std::floor((verts_min.x - grid_min.x) / grid_spacing.x)) - 1);
        first_cell.y = std::max(0, int(std::floor((verts_min.y - grid_min.y) / grid_spacing.y)) - 1);
        first_cell.z = std::max(0, int(std::floor((verts_min.z - grid_min.z) / grid_spacing.z)) - 1);
        last_cell.x = std::min(grid_divisions.x - 1, int(std::floor((verts_max.x - grid_min.x) / grid_spacing.x)) + 1);
        last_cell.y = std::min(grid_divisions.y - 1, int(std::floor((verts_max.y - grid_min.y) / grid_spacing.y)) + 1);
        last_cell.z = std::min(grid_divisions.z - 1, int(std::floor((verts_max.z - grid_min.z) / grid_spacing.z)) + 1);
    };

    auto start = std::chrono::high_resolution_clock::now();
    for (uint p = 0; p < UUIDs.size(); p++) {
        bool flag = false;
        std::vector<helios::vec3> verts = this->context->getPrimitiveVertices(UUIDs.at(p));
        helios::int3 first_cell, last_cell;
        overlappingCellRange(verts, first_cell, last_cell);
        for (int k = first_cell.z; k <= last_cell.z; k++) {
            for (int j = first_cell.y; j <= last_cell.y; j++) {
                for (int i = first_cell.x; i <= last_cell.x; i++) {
                    helios::vec3 cell_min = make_vec3(grid_min.x + float(i) * grid_spacing.x, grid_min.y + float(j) * grid_spacing.y, grid_min.z + float(k) * grid_spacing.z);
                    helios::vec3 cell_max = make_vec3(grid_min.x + float(i) * grid_spacing.x + grid_spacing.x, grid_min.y + float(j) * grid_spacing.y + grid_spacing.y, grid_min.z + float(k) * grid_spacing.z + grid_spacing.z);

                    uint v_in = 0;
                    for (uint v = 0; v < verts.size(); v++) {

                        bool test2_x = (verts.at(v).x >= cell_min.x) && (verts.at(v).x <= cell_max.x);
                        bool test2_y = (verts.at(v).y >= cell_min.y) && (verts.at(v).y <= cell_max.y);
                        bool test2_z = (verts.at(v).z >= cell_min.z) && (verts.at(v).z <= cell_max.z);

                        if (test2_x && test2_y && test2_z) {
                            v_in++;
                        }
                    }

                    if (v_in == verts.size()) {
                        // the UUID doesn't need to be sliced since its vertices all are within a cell
                        int cell_ID = i * grid_divisions.y * grid_divisions.z + j * grid_divisions.z + k;
                        this->context->setPrimitiveData(UUIDs.at(p), "cell_ID", cell_ID);
                        this->grid_cells[i][j][k].push_back(UUIDs.at(p));
                        UUIDs_no_slice.push_back(UUIDs.at(p));
                        flag = true;
                        break;
                    } else if (v_in != 0) {
                        // some verticies in and some out: UUID needs to be sliced
                        UUIDs_to_slice.push_back(UUIDs.at(p));
                        flag = true;
                        break;
                    }
                }
                if (flag == true) {
                    break;
                }
            }
            if (flag == true) {
                break;
            }
        }

        // if all vertices fell outside of all grid cells, add it to be sliced just in case (corner cases)
        if (flag == false) {
            UUIDs_to_slice.push_back(UUIDs.at(p));
        }
    }

    if (this->printmessages) {

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << duration.count() << " seconds to do initial grid cell classification" << std::endl;
        std::cout << UUIDs_no_slice.size() << " input primitives (" << float(UUIDs_no_slice.size()) / float(UUIDs.size()) * 100 << "%) not sliced" << std::endl;
        std::cout << UUIDs_to_slice.size() << " input primitives (" << float(UUIDs_to_slice.size()) / float(UUIDs.size()) * 100 << "%) being sliced" << std::endl;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // do the slicing

    std::vector<uint> primitives_to_remove;
    std::vector<uint> primitives_to_add;

    auto start2 = std::chrono::high_resolution_clock::now();
    // Vertices of the primitives still to be sliced, read from the Context once per primitive rather than once per face
    std::unordered_map<uint, std::vector<helios::vec3>> pending_vertices;
    for (uint UUID: UUIDs_to_slice) {
        pending_vertices[UUID] = this->context->getPrimitiveVertices(UUID);
    }

    // loop through each voxel face
    for (uint i = 0; i < grid_face_vertices.size(); i++) {
        // A primitive whose vertices all lie strictly on one side of the face plane cannot be cut by it, and
        // slicePrimitive() would return it unchanged; skip it. Uses the same signed distance as slicePrimitive().
        const std::vector<helios::vec3> &face = grid_face_vertices.at(i);
        helios::vec3 face_normal = cross(face.at(1) - face.at(0), face.at(2) - face.at(1));
        face_normal.normalize();
        const float face_offset = face_normal * face.at(0);

        for (uint j = 0; j < UUIDs_to_slice.size(); j++) {
            const std::vector<helios::vec3> &vertices = pending_vertices.at(UUIDs_to_slice.at(j));
            bool all_above = true;
            bool all_below = true;
            for (const helios::vec3 &vertex: vertices) {
                const float signed_distance = face_normal * vertex - face_offset;
                all_above = all_above && signed_distance > 0.f;
                all_below = all_below && signed_distance < 0.f;
            }
            if (all_above || all_below) {
                continue;
            }

            // slice
            std::vector<uint> resulting_UUIDs;
            resulting_UUIDs = slicePrimitive(UUIDs_to_slice.at(j), grid_face_vertices.at(i), warnings);

            // update the UUIDs_to_slice vector so it doesn't include deleted primitives (the originals that were split)
            bool exists = this->context->doesPrimitiveExist(UUIDs_to_slice.at(j));
            if (!exists) {
                primitives_to_remove.push_back(j);
                primitives_to_add.insert(primitives_to_add.end(), resulting_UUIDs.begin(), resulting_UUIDs.end());
                pending_vertices.erase(UUIDs_to_slice.at(j));
                for (uint piece: resulting_UUIDs) {
                    pending_vertices[piece] = this->context->getPrimitiveVertices(piece);
                }
            }
        }

        for (int k = primitives_to_remove.size() - 1; k >= 0; k--) {
            UUIDs_to_slice.erase(UUIDs_to_slice.begin() + primitives_to_remove.at(k));
        }
        primitives_to_remove.clear();

        UUIDs_to_slice.insert(UUIDs_to_slice.end(), primitives_to_add.begin(), primitives_to_add.end());
        primitives_to_add.clear();
    }

    if (this->printmessages) {

        auto stop2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::seconds>(stop2 - start2);
        std::cout << duration2.count() << " seconds to do slicing" << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // now classify the sliced primitives into grid cells
    // save the cell_ID as primitive data for the triangle
    // save the primitive UUID to a vector of UUIDs that are in a given cell and save that

    auto start3 = std::chrono::high_resolution_clock::now();

    for (uint p = 0; p < UUIDs_to_slice.size(); p++) {
        // std::cout << "UUIDs_to_slice.at(p) = " << UUIDs_to_slice.at(p) << std::endl;
        bool flag = false;
        std::vector<helios::vec3> verts = this->context->getPrimitiveVertices(UUIDs_to_slice.at(p));
        helios::int3 first_cell, last_cell;
        overlappingCellRange(verts, first_cell, last_cell);

        for (int k = first_cell.z; k <= last_cell.z; k++) {
            for (int j = first_cell.y; j <= last_cell.y; j++) {
                for (int i = first_cell.x; i <= last_cell.x; i++) {

                    helios::vec3 cell_min = make_vec3(grid_min.x + i * grid_spacing.x, grid_min.y + j * grid_spacing.y, grid_min.z + k * grid_spacing.z);
                    helios::vec3 cell_max = make_vec3(grid_min.x + i * grid_spacing.x + grid_spacing.x, grid_min.y + j * grid_spacing.y + grid_spacing.y, grid_min.z + k * grid_spacing.z + grid_spacing.z);

                    uint v_in = 0;
                    for (uint v = 0; v < verts.size(); v++) {

                        float absTol = pow(10, -6);
                        float relTol = pow(10, -20);
                        bool test2_x = (verts.at(v).x > cell_min.x || approxSame(verts.at(v).x, cell_min.x, absTol, relTol)) && (verts.at(v).x < cell_max.x || approxSame(verts.at(v).x, cell_max.x, absTol, relTol));
                        bool test2_y = (verts.at(v).y > cell_min.y || approxSame(verts.at(v).y, cell_min.y, absTol, relTol)) && (verts.at(v).y < cell_max.y || approxSame(verts.at(v).y, cell_max.y, absTol, relTol));
                        bool test2_z = (verts.at(v).z > cell_min.z || approxSame(verts.at(v).z, cell_min.z, absTol, relTol)) && (verts.at(v).z < cell_max.z || approxSame(verts.at(v).z, cell_max.z, absTol, relTol));

                        if (test2_x && test2_y && test2_z) {
                            v_in++;
                        }
                    }

                    if (v_in == verts.size()) {
                        int cell_ID = i * grid_divisions.y * grid_divisions.z + j * grid_divisions.z + k;
                        this->context->setPrimitiveData(UUIDs_to_slice.at(p), "cell_ID", cell_ID);
                        this->grid_cells[i][j][k].push_back(UUIDs_to_slice.at(p));
                        flag = true;
                        break;
                    }
                }
                if (flag == true) {
                    break;
                }
            }
            if (flag == true) {
                break;
            }
        }

        if (flag == false) {
            // Primitive doesn't fit in any cell - this is an error condition that should be reported
            if (this->printmessages) {
                warnings.addWarning("slice_does_not_fit_grid_cell", "Primitive " + std::to_string(UUIDs_to_slice.at(p)) + " does not fit in any grid cell after slicing");
            }
        }
    }

    if (this->printmessages) {
        auto stop3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::seconds>(stop3 - start3);
        std::cout << duration3.count() << " seconds to do second classification" << std::endl;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Join the unsliced and sliced primitive UUIDs back into a single vector
    std::vector<uint> UUIDs_out = UUIDs_no_slice;
    UUIDs_out.insert(UUIDs_out.end(), UUIDs_to_slice.begin(), UUIDs_to_slice.end());

    if (this->printmessages) {
        std::cout << UUIDs_to_slice.size() << " primitives created from slicing" << std::endl;
        std::cout << UUIDs_out.size() << " total output primitives" << std::endl;
    }

    // Report aggregated warnings
    warnings.report(std::cerr);

    return UUIDs_out;
}

// -------- VOXEL-PRIMITIVE INTERSECTION (OpenMP) --------

void CollisionDetection::calculatePrimitiveVoxelIntersection(const std::vector<uint> &UUIDs) {
    if (this->printmessages) {
        std::cout << "Calculating primitive-voxel intersections..." << std::flush;
    }

    // Separate voxels from planar primitives
    std::vector<uint> voxel_uuids;
    std::vector<uint> primitive_uuids;

    std::vector<uint> uuids_to_process = UUIDs.empty() ? this->context->getAllUUIDs() : UUIDs;

    for (uint uuid: uuids_to_process) {
        if (this->context->getPrimitiveType(uuid) == PRIMITIVE_TYPE_VOXEL) {
            voxel_uuids.push_back(uuid);
        } else {
            primitive_uuids.push_back(uuid);
        }
    }

    // Each call reports where the primitives are now, so results from an earlier call are replaced, not appended to
    for (uint voxel_uuid: voxel_uuids) {
        if (this->context->doesPrimitiveDataExist(voxel_uuid, "inside_UUIDs")) {
            this->context->clearPrimitiveData(voxel_uuid, "inside_UUIDs");
        }
    }

    if (voxel_uuids.empty() || primitive_uuids.empty()) {
        if (this->printmessages) {
            std::cout << "done. WARNING: ";
            if (voxel_uuids.empty())
                std::cout << "no voxels found";
            if (primitive_uuids.empty())
                std::cout << "no planar primitives found";
            std::cout << std::endl;
        }
        return;
    }

    // Get primitive centers (geometric centroid)
    std::vector<vec3> primitive_centers(primitive_uuids.size());
    for (size_t i = 0; i < primitive_uuids.size(); i++) {
        auto vertices = this->context->getPrimitiveVertices(primitive_uuids[i]);
        if (vertices.empty()) {
            helios_runtime_error("ERROR (CollisionDetection::calculatePrimitiveVoxelIntersection): Primitive " + std::to_string(primitive_uuids[i]) + " has no vertices");
        }
        vec3 centroid = make_vec3(0, 0, 0);
        for (const auto &v: vertices) {
            centroid = centroid + v;
        }
        primitive_centers[i] = centroid / static_cast<float>(vertices.size());
    }

    // Get voxel bounds
    // NOTE: Rotated voxels are NOT supported - voxels must be axis-aligned
    std::vector<vec3> voxel_mins(voxel_uuids.size());
    std::vector<vec3> voxel_maxs(voxel_uuids.size());
    for (size_t i = 0; i < voxel_uuids.size(); i++) {
        const vec3 voxel_center = this->context->getVoxelCenter(voxel_uuids[i]);
        const vec3 voxel_size = this->context->getVoxelSize(voxel_uuids[i]);
        voxel_mins[i] = voxel_center - voxel_size * 0.5f;
        voxel_maxs[i] = voxel_center + voxel_size * 0.5f;
    }

    // A primitive is inside a voxel if its centroid lies within the voxel's box (faces included). A centroid on a face
    // shared by two voxels goes to the first of them in voxel order. The assignment runs in parallel with each thread
    // writing only its own primitives' slots; the per-voxel lists are then built serially so their order is deterministic.
    constexpr int not_inside_any_voxel = -1;
    std::vector<int> containing_voxel(primitive_uuids.size(), not_inside_any_voxel);
#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < static_cast<int>(primitive_uuids.size()); p++) {
        const vec3 &center = primitive_centers[p];
        for (size_t v = 0; v < voxel_uuids.size(); v++) {
            const vec3 &voxel_min = voxel_mins[v];
            const vec3 &voxel_max = voxel_maxs[v];
            if (center.x >= voxel_min.x && center.x <= voxel_max.x && center.y >= voxel_min.y && center.y <= voxel_max.y && center.z >= voxel_min.z && center.z <= voxel_max.z) {
                containing_voxel[p] = static_cast<int>(v);
                break;
            }
        }
    }

    std::vector<std::vector<uint>> inside_lists(voxel_uuids.size());
    for (size_t p = 0; p < primitive_uuids.size(); p++) {
        if (containing_voxel[p] != not_inside_any_voxel) {
            inside_lists[containing_voxel[p]].push_back(primitive_uuids[p]);
        }
    }
    for (size_t v = 0; v < voxel_uuids.size(); v++) {
        if (!inside_lists[v].empty()) {
            std::sort(inside_lists[v].begin(), inside_lists[v].end());
            this->context->setPrimitiveData(voxel_uuids[v], "inside_UUIDs", inside_lists[v]);
        }
    }

    if (this->printmessages) {
        std::cout << "done." << std::endl;
    }
}
