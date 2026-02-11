/** \file "CollisionDetection_Slicing.cpp" Source file for primitive slicing and voxel intersection operations.

    Copyright (C) 2016-2026 Brian Bailey, Eric Kent

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "CollisionDetection.h"

using namespace helios;

// -------- GEOMETRIC UTILITY FUNCTIONS --------

helios::vec3 CollisionDetection::linesIntersection(const helios::vec3 &line1_point, const helios::vec3 &line1_direction, const helios::vec3 &line2_point, const helios::vec3 &line2_direction) const {

    helios::vec3 g = line2_point - line1_point;
    helios::vec3 h = cross(line2_direction, g);
    helios::vec3 k = cross(line2_direction, line1_direction);

    float h_mag = sqrt(pow(h.x, 2) + pow(h.y, 2) + pow(h.z, 2));
    float k_mag = sqrt(pow(k.x, 2) + pow(k.y, 2) + pow(k.z, 2));

    // in the same direction
    if (((h.x >= 0 && k.x >= 0) || (h.x < 0 && k.x < 0)) && ((h.y >= 0 && k.y >= 0) || (h.y < 0 && k.y < 0)) && ((h.z >= 0 && k.z >= 0) || (h.z < 0 && k.z < 0))) {
        helios::vec3 rht = (h_mag / k_mag) * line1_direction;
        return line1_point + rht;
    } else { // different direction
        helios::vec3 rht = (h_mag / k_mag) * line1_direction;
        return line1_point - rht;
    }
}

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
    float Duv = sqrtf(powf(uv2.x - uv1.x, 2.0) + powf(uv2.y - uv1.y, 2.0)); // distance between edge vertex uv coordinates
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

    // if the u coordinates of the two vertices are the same
    if (uv2.x == uv1.x) {
        std::vector<float> vec_uv;
        vec_uv.push_back(uv1.y);
        vec_uv.push_back(uv2.y);
        uvs = make_vec2(uv1.x, min(vec_uv) + Duv * (Dxyzs / Dxyz));

    } else {

        // equation for the line between uv coordinates of the two vertices
        float slope = (uv2.y - uv1.y) / (uv2.x - uv1.x);
        float offset = uv1.y - slope * uv1.x;

        // coefficients of the quadratic equation for the u coordinate of the slice point
        float a = powf(slope, 2.0) + 1.0;
        float b = -2.0 * uv1.x + 2.0 * slope * offset - 2.0 * slope * uv1.y;
        float c = (powf(uv1.x, 2.0) + powf(offset, 2.0) - 2.0 * offset * uv1.y + powf(uv1.y, 2.0)) - powf((Dxyzs / Dxyz) * Duv, 2.0);

        // solve the quadratic
        float us_a = (-1.0 * b + sqrtf(powf(b, 2.0) - 4.0 * a * c)) / (2.0 * a);
        float us_b = (-1.0 * b - sqrtf(powf(b, 2.0) - 4.0 * a * c)) / (2.0 * a);
        // get the v coordinate
        float vs_a = slope * us_a + offset;
        float vs_b = slope * us_b + offset;

        // determine which of the roots is the right one
        if (((us_a >= uv1.x && us_a <= uv2.x) || (us_a <= uv1.x && us_a >= uv2.x)) && ((vs_a >= uv1.y && vs_a <= uv2.y) || (vs_a <= uv1.y && vs_a >= uv2.y))) {
            uvs = make_vec2(us_a, vs_a);

        } else if (((us_b >= uv1.x && us_b <= uv2.x) || (us_b <= uv1.x && us_b >= uv2.x)) && ((vs_b >= uv1.y && vs_b <= uv2.y) || (vs_b <= uv1.y && vs_b >= uv2.y))) {
            uvs = make_vec2(us_b, vs_b);

        } else {
            helios_runtime_error("ERROR (CollisionDetection::interpolate_texture_UV_to_slice_point): could not interpolate UV coordinates.");
        }
    }

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

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find the equation of the line where the planes of the patch and voxel face intersect

    // direction of the plane intersection line
    helios::vec3 direction_vector = cross(face_normal, primitive_normal);

    // find a point on the plane intersection line
    // based on https://vicrucann.github.io/tutorials/3d-geometry-algorithms/
    helios::vec3 a = helios::make_vec3(fabs(direction_vector.x), fabs(direction_vector.y), fabs(direction_vector.z));
    uint maxc;
    if (a.x > a.y) {
        if (a.x > a.z) {
            maxc = 1;
        } else {
            maxc = 3;
        }
    } else {
        if (a.y > a.z) {
            maxc = 2;
        } else {
            maxc = 3;
        }
    }

    helios::vec3 d1a = helios::make_vec3(-1 * face_normal.x * voxel_face_vertices.at(0).x, -1 * face_normal.y * voxel_face_vertices.at(0).y, -1 * face_normal.z * voxel_face_vertices.at(0).z);
    helios::vec3 d2a = helios::make_vec3(-1 * primitive_normal.x * primitive_vertices.at(1).x, -1 * primitive_normal.y * primitive_vertices.at(1).y, -1 * primitive_normal.z * primitive_vertices.at(1).z);

    float d1 = d1a.x + d1a.y + d1a.z;
    float d2 = d2a.x + d2a.y + d2a.z;

    float xi;
    float yi;
    float zi;

    if (maxc == 1) {
        xi = 0;
        yi = (d2 * face_normal.z - d1 * primitive_normal.z) / direction_vector.x;
        zi = (d1 * primitive_normal.y - d2 * face_normal.y) / direction_vector.x;
    } else if (maxc == 2) {
        xi = (d1 * primitive_normal.z - d2 * face_normal.z) / direction_vector.y;
        yi = 0;
        zi = (d2 * face_normal.x - d1 * primitive_normal.x) / direction_vector.y;
    } else if (maxc == 3) {
        xi = (d2 * face_normal.y - d1 * primitive_normal.y) / direction_vector.z;
        yi = (d1 * primitive_normal.x - d2 * face_normal.x) / direction_vector.z;
        zi = 0;
    }

    helios::vec3 ipoint = make_vec3(xi, yi, zi);

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // get points of intersection between each edge of the patch and the patch-voxel intersection line

    // vector for points of intersection between edge line and intersection line
    std::vector<helios::vec3> possible_points;
    // vector for points that actually touch the patch
    std::vector<helios::vec3> slice_points;
    std::vector<uint> slice_points_edge_ID;
    uint vertex_index; // index for cases where one slice point is on a vertex (used for patch cases only)

    helios::vec3 vi0;
    helios::vec3 vi1;

    // go through the different edges of the patch and calculate intersection points with line along edge of patch and intersection line
    if (primitive_vertices.size() == 4) {
        possible_points.resize(4);
        possible_points.at(0) = linesIntersection(primitive_vertices.at(1), primitive_vertices.at(1) - primitive_vertices.at(0), ipoint, direction_vector);
        possible_points.at(1) = linesIntersection(primitive_vertices.at(2), primitive_vertices.at(2) - primitive_vertices.at(1), ipoint, direction_vector);
        possible_points.at(2) = linesIntersection(primitive_vertices.at(3), primitive_vertices.at(3) - primitive_vertices.at(2), ipoint, direction_vector);
        possible_points.at(3) = linesIntersection(primitive_vertices.at(0), primitive_vertices.at(0) - primitive_vertices.at(3), ipoint, direction_vector);

        for (uint i = 0; i < 4; i++) {
            if (i == 0) {
                vi1 = primitive_vertices.at(1);
                vi0 = primitive_vertices.at(0);
            } else if (i == 1) {
                vi1 = primitive_vertices.at(2);
                vi0 = primitive_vertices.at(1);
            } else if (i == 2) {
                vi1 = primitive_vertices.at(3);
                vi0 = primitive_vertices.at(2);
            } else if (i == 3) {
                vi1 = primitive_vertices.at(0);
                vi0 = primitive_vertices.at(3);
            }

            bool test_x = ((possible_points.at(i).x >= vi1.x && possible_points.at(i).x <= vi0.x) || (possible_points.at(i).x >= vi0.x && possible_points.at(i).x <= vi1.x));
            bool test_y = ((possible_points.at(i).y >= vi1.y && possible_points.at(i).y <= vi0.y) || (possible_points.at(i).y >= vi0.y && possible_points.at(i).y <= vi1.y));
            bool test_z = ((possible_points.at(i).z >= vi1.z && possible_points.at(i).z <= vi0.z) || (possible_points.at(i).z >= vi0.z && possible_points.at(i).z <= vi1.z));

            if (test_x && test_y && test_z) {
                slice_points.push_back(possible_points.at(i));
                slice_points_edge_ID.push_back(i);
            }
        }

    } else if (primitive_vertices.size() == 3) {

        possible_points.resize(3);
        possible_points.at(0) = linesIntersection(primitive_vertices.at(1), primitive_vertices.at(1) - primitive_vertices.at(0), ipoint, direction_vector);
        possible_points.at(1) = linesIntersection(primitive_vertices.at(2), primitive_vertices.at(2) - primitive_vertices.at(1), ipoint, direction_vector);
        possible_points.at(2) = linesIntersection(primitive_vertices.at(0), primitive_vertices.at(0) - primitive_vertices.at(2), ipoint, direction_vector);

        for (uint i = 0; i < 3; i++) {
            if (i == 0) {
                vi1 = primitive_vertices.at(1);
                vi0 = primitive_vertices.at(0);
            } else if (i == 1) {
                vi1 = primitive_vertices.at(2);
                vi0 = primitive_vertices.at(1);

            } else if (i == 2) {
                vi1 = primitive_vertices.at(0);
                vi0 = primitive_vertices.at(2);
            }

            bool test_x = ((possible_points.at(i).x >= vi1.x && possible_points.at(i).x <= vi0.x) || (possible_points.at(i).x >= vi0.x && possible_points.at(i).x <= vi1.x));
            bool test_y = ((possible_points.at(i).y >= vi1.y && possible_points.at(i).y <= vi0.y) || (possible_points.at(i).y >= vi0.y && possible_points.at(i).y <= vi1.y));
            bool test_z = ((possible_points.at(i).z >= vi1.z && possible_points.at(i).z <= vi0.z) || (possible_points.at(i).z >= vi0.z && possible_points.at(i).z <= vi1.z));

            if (test_x && test_y && test_z) {
                slice_points.push_back(possible_points.at(i));
                slice_points_edge_ID.push_back(i);
            }
        }
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
        uint non_vertex_slice_edge_ID;
        vec3 vertex_slice_point;

        for (uint bb = 0; bb < slice_points.size(); bb++) {
            bool this_point_vert_test = false;
            for (uint cc = 0; cc < primitive_vertices.size(); cc++) {
                bool vert_test = approxSame(slice_points.at(bb), primitive_vertices.at(cc), absTol);
                // std::cout << "-- test = " << vert_test <<" -- slice point " << bb << " = " << slice_points.at(bb) << ", primitive_vertex " << cc << " = " << primitive_vertices.at(cc) << std::endl;
                if (vert_test) {
                    this_point_vert_test = true;
                    vertex_slice_point = primitive_vertices.at(cc);
                    vertex_index = cc;
                }
            }

            if (this_point_vert_test == false) {
                non_vertex_slice_point = slice_points.at(bb);
                non_vertex_slice_edge_ID = slice_points_edge_ID.at(bb);
            }
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
        vec3 non_vertex_slice_point;
        uint non_vertex_slice_edge_ID;
        vec3 vertex_slice_point;
        for (uint bb = 0; bb < slice_points.size(); bb++) {
            bool this_point_vert_test = false;
            for (uint cc = 0; cc < primitive_vertices.size(); cc++) {
                bool vert_test = approxSame(slice_points.at(bb), primitive_vertices.at(cc), absTol);
                // std::cout << "-- test = " << vert_test <<" -- slice point " << bb << " = " << slice_points.at(bb) << ", primitive_vertex " << cc << " = " << primitive_vertices.at(cc) << std::endl;
                if (vert_test) {
                    this_point_vert_test = true;
                    vertex_index = cc;
                }
            }
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
        if (parentID > 0 && this->context->getObjectType(parentID) == helios::OBJECT_TYPE_TILE) {
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

    auto start = std::chrono::high_resolution_clock::now();
    for (uint p = 0; p < UUIDs.size(); p++) {
        bool flag = false;
        for (uint k = 0; k < (grid_divisions.z); k++) {
            for (uint j = 0; j < (grid_divisions.y); j++) {
                for (uint i = 0; i < (grid_divisions.x); i++) {
                    helios::vec3 cell_min = make_vec3(grid_min.x + float(i) * grid_spacing.x, grid_min.y + float(j) * grid_spacing.y, grid_min.z + float(k) * grid_spacing.z);
                    helios::vec3 cell_max = make_vec3(grid_min.x + float(i) * grid_spacing.x + grid_spacing.x, grid_min.y + float(j) * grid_spacing.y + grid_spacing.y, grid_min.z + float(k) * grid_spacing.z + grid_spacing.z);
                    std::vector<helios::vec3> verts = this->context->getPrimitiveVertices(UUIDs.at(p));

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
    // loop through each voxel face
    for (uint i = 0; i < grid_face_vertices.size(); i++) {
        for (uint j = 0; j < UUIDs_to_slice.size(); j++) {
            // slice
            std::vector<uint> resulting_UUIDs;
            resulting_UUIDs = slicePrimitive(UUIDs_to_slice.at(j), grid_face_vertices.at(i), warnings);

            // update the UUIDs_to_slice vector so it doesn't include deleted primitives (the originals that were split)
            bool exists = this->context->doesPrimitiveExist(UUIDs_to_slice.at(j));
            if (!exists) {
                primitives_to_remove.push_back(j);
                primitives_to_add.insert(primitives_to_add.end(), resulting_UUIDs.begin(), resulting_UUIDs.end());
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

        for (uint k = 0; k < (grid_divisions.z); k++) {
            for (uint j = 0; j < (grid_divisions.y); j++) {
                for (uint i = 0; i < (grid_divisions.x); i++) {

                    helios::vec3 cell_min = make_vec3(grid_min.x + i * grid_spacing.x, grid_min.y + j * grid_spacing.y, grid_min.z + k * grid_spacing.z);
                    helios::vec3 cell_max = make_vec3(grid_min.x + i * grid_spacing.x + grid_spacing.x, grid_min.y + j * grid_spacing.y + grid_spacing.y, grid_min.z + k * grid_spacing.z + grid_spacing.z);

                    std::vector<helios::vec3> verts = this->context->getPrimitiveVertices(UUIDs_to_slice.at(p));
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
                std::cerr << "WARNING (CollisionDetection::slicePrimitivesUsingGrid): Primitive " << UUIDs_to_slice.at(p) << " does not fit in any grid cell after slicing" << std::endl;
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

    // Get voxel data (center, size)
    // NOTE: Rotated voxels are NOT supported - voxels must be axis-aligned
    std::vector<vec3> voxel_centers(voxel_uuids.size());
    std::vector<vec3> voxel_sizes(voxel_uuids.size());

    for (size_t i = 0; i < voxel_uuids.size(); i++) {
        voxel_centers[i] = this->context->getVoxelCenter(voxel_uuids[i]);
        voxel_sizes[i] = this->context->getVoxelSize(voxel_uuids[i]);
    }

// Parallel processing with OpenMP (algorithm from insideVolume_vi CUDA kernel)
// WARNING: This implementation assumes axis-aligned voxels (rotation = 0)
#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < static_cast<int>(primitive_uuids.size()); p++) {
        vec3 prim_center = primitive_centers[p];

        // Test against all voxels
        for (size_t v = 0; v < voxel_uuids.size(); v++) {
            vec3 voxel_center = voxel_centers[v];
            vec3 voxel_size = voxel_sizes[v];

            // Note: VoxelIntersection CUDA code sets rotation=0 (not implemented)
            // Skipping rotation transformation for now

            // Ray-box intersection test from origin (0,0,0) to primitive center
            vec3 origin = make_vec3(0, 0, 0);
            vec3 direction = prim_center - origin;
            float distance_to_center = direction.magnitude(); // Save distance BEFORE normalizing
            direction.normalize();

            // AABB intersection (robust slab method with divide-by-zero handling)
            vec3 voxel_min = voxel_center - voxel_size * 0.5f;
            vec3 voxel_max = voxel_center + voxel_size * 0.5f;

            float tx_min, tx_max, ty_min, ty_max, tz_min, tz_max;
            const float epsilon = 1e-8f;

            // X-axis slab
            if (fabs(direction.x) < epsilon) {
                // Ray parallel to YZ plane
                if (origin.x < voxel_min.x || origin.x > voxel_max.x) {
                    continue; // No intersection
                }
                tx_min = -std::numeric_limits<float>::infinity();
                tx_max = std::numeric_limits<float>::infinity();
            } else {
                tx_min = (voxel_min.x - origin.x) / direction.x;
                tx_max = (voxel_max.x - origin.x) / direction.x;
                if (tx_min > tx_max)
                    std::swap(tx_min, tx_max);
            }

            // Y-axis slab
            if (fabs(direction.y) < epsilon) {
                // Ray parallel to XZ plane
                if (origin.y < voxel_min.y || origin.y > voxel_max.y) {
                    continue; // No intersection
                }
                ty_min = -std::numeric_limits<float>::infinity();
                ty_max = std::numeric_limits<float>::infinity();
            } else {
                ty_min = (voxel_min.y - origin.y) / direction.y;
                ty_max = (voxel_max.y - origin.y) / direction.y;
                if (ty_min > ty_max)
                    std::swap(ty_min, ty_max);
            }

            // Z-axis slab
            if (fabs(direction.z) < epsilon) {
                // Ray parallel to XY plane
                if (origin.z < voxel_min.z || origin.z > voxel_max.z) {
                    continue; // No intersection
                }
                tz_min = -std::numeric_limits<float>::infinity();
                tz_max = std::numeric_limits<float>::infinity();
            } else {
                tz_min = (voxel_min.z - origin.z) / direction.z;
                tz_max = (voxel_max.z - origin.z) / direction.z;
                if (tz_min > tz_max)
                    std::swap(tz_min, tz_max);
            }

            float t_enter = std::max({tx_min, ty_min, tz_min});
            float t_exit = std::min({tx_max, ty_max, tz_max});

            // Check if ray intersects box AND primitive center point is inside box
            if (t_enter < t_exit && t_exit > 1e-6f) {
                // Primitive center is inside voxel if its distance falls between entry and exit points
                if (distance_to_center >= t_enter && distance_to_center <= t_exit) {
#pragma omp critical
                    {
                        // Add primitive UUID to voxel's "inside_UUIDs" data (using modern template API)
                        if (!this->context->doesPrimitiveDataExist(voxel_uuids[v], "inside_UUIDs")) {
                            std::vector<uint> inside_list = {primitive_uuids[p]};
                            this->context->setPrimitiveData(voxel_uuids[v], "inside_UUIDs", inside_list);
                        } else {
                            std::vector<uint> inside_list;
                            this->context->getPrimitiveData(voxel_uuids[v], "inside_UUIDs", inside_list);
                            inside_list.push_back(primitive_uuids[p]);
                            this->context->setPrimitiveData(voxel_uuids[v], "inside_UUIDs", inside_list);
                        }
                    }
                    break; // Primitive can only be in one voxel
                }
            }
        }
    }

    if (this->printmessages) {
        std::cout << "done." << std::endl;
    }
}
