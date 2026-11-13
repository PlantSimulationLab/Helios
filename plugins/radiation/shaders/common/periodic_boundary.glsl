/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

/** periodic_boundary.glsl - Periodic boundary condition utilities
 *
 * Implements ray-bbox face intersection and coordinate wrapping for periodic
 * boundary conditions. Bbox faces are thin rectangular walls at domain edges
 * that catch rays exiting the domain and wrap them to the opposite side.
 *
 * Architecture: Bboxes are checked SEPARATELY from the BVH (not integrated).
 * After each BVH traversal, this code tests the 2-4 bbox faces for closer hits.
 * This matches the OptiX backend where bboxes are a separate geometry group.
 *
 * Reference: rayHit.cu:41-65 (wrapping logic), rayGeneration.cu:238-246 (wrap loop)
 */

#ifndef PERIODIC_BOUNDARY_GLSL
#define PERIODIC_BOUNDARY_GLSL

// NOTE: The calling shader must declare bbox_verts_buf and periodic_info_buf before including this file, and include
// bvh_traversal.glsl before it:
//   layout(set = 0, binding = 17) readonly buffer BboxVerticesBuffer {
//       float vertices[]; // 12 floats per face (4 vec3 vertices)
//   } bbox_verts_buf;
//   layout(set = 0, binding = 22) readonly buffer PeriodicInfoBuffer { ... } periodic_info_buf; // see VulkanComputeBackend::uploadPeriodicInfo()

// Periodic copy of the domain the ray being traced is in (0 = central). It is tracked only when periodic_info_buf.tile_tracking is
// set: some primitives are excluded from the periodic boundary and exist only in the central copy, or the number of copies is
// limited. Set with set_periodic_tile(), which also tells the BVH traversal which primitives the ray cannot hit.
ivec2 periodic_tile = ivec2(0);
// Periodic copy of the domain of the primitive the ray was launched from, which the ray does not hit in that copy (see set_periodic_origin_tile())
ivec2 periodic_origin_tile = ivec2(0);

// Whether a periodic copy of the domain lies beyond the limited field of copies, where nothing exists
bool periodic_tile_outside_field(ivec2 tile) {
    return (periodic_info_buf.max_repeats_x >= 0 && abs(tile.x) > periodic_info_buf.max_repeats_x) || (periodic_info_buf.max_repeats_y >= 0 && abs(tile.y) > periodic_info_buf.max_repeats_y);
}

// Make `tile` the periodic copy of the domain the ray is in. Primitives absent from that copy are not hit: a primitive excluded from
// the periodic boundary exists only in the central copy, and nothing exists beyond a limited number of copies. A dual ray that has
// passed an excluded primitive follows the background radiation field, where no copy has them (see traversal_excluded_passed in
// bvh_traversal.glsl). Must match periodicHidden() in the OptiX backends.
void set_periodic_tile(ivec2 tile) {
    periodic_tile = tile;
    traversal_skip_origin = tile == periodic_origin_tile;
    traversal_hidden_flags = 0u;
    traversal_hide_all = false;
    if (periodic_info_buf.tile_tracking == 0u) {
        return;
    }
    if (periodic_tile_outside_field(tile)) {
        traversal_hide_all = true;
    } else if (tile != ivec2(0) || traversal_excluded_passed) {
        traversal_hidden_flags = LEAF_RECORD_PERIODIC_EXCLUDED;
    }
}

// Start a ray launched from a primitive in the periodic copy `tile` of the domain: it does not hit that copy of the primitive, but does hit its other copies. Must match the origin tests of the OptiX backends.
void start_periodic_ray(ivec2 tile) {
    periodic_origin_tile = tile;
    set_periodic_tile(tile);
}

/**
 * Move a coordinate into [lower, upper) by a whole number of periods.
 *
 * A point on the lower face of a ray travelling out through that face is moved to the upper face instead, so the ray starts
 * inside the domain. Must match wrapIntoPeriod() in the OptiX backends.
 */
float wrap_into_period(float coordinate, float direction, float lower, float upper) {
    float period = upper - lower;
    float wrapped = coordinate - period * floor((coordinate - lower) / period);
    if (wrapped < lower || wrapped >= upper) {
        wrapped = lower; // rounding at a whole number of periods
    }
    if (wrapped == lower && direction < 0.0) {
        wrapped = upper;
    }
    return wrapped;
}

/**
 * Ray-rectangle intersection for one bbox face.
 *
 * Plane intersection followed by a point-in-quad test along the face's two edges. A ray is wrapped only by the face it leaves
 * the domain through: faces come in (min, max) pairs, and vertex 0 of a face minus vertex 0 of the other face of its pair points
 * out of the domain. The face a wrapped ray starts on is therefore never hit again, so the ray can be continued with no minimum
 * distance, and a ray leaving the domain is wrapped at whatever distance it is from the face, also one closer than the minimum
 * distance of the BVH traversal.
 * Returns whether the face was hit, and the distance to it in t.
 */
bool intersect_bbox_face(vec3 ray_origin, vec3 ray_dir, uint face_idx, out float t) {
    t = 0.0;
    uint base = face_idx * 12; // 4 vertices * 3 floats each
    vec3 v0 = vec3(bbox_verts_buf.vertices[base + 0], bbox_verts_buf.vertices[base + 1], bbox_verts_buf.vertices[base + 2]);
    vec3 v1 = vec3(bbox_verts_buf.vertices[base + 3], bbox_verts_buf.vertices[base + 4], bbox_verts_buf.vertices[base + 5]);
    vec3 v2 = vec3(bbox_verts_buf.vertices[base + 6], bbox_verts_buf.vertices[base + 7], bbox_verts_buf.vertices[base + 8]);
    vec3 v3 = vec3(bbox_verts_buf.vertices[base + 9], bbox_verts_buf.vertices[base + 10], bbox_verts_buf.vertices[base + 11]);

    vec3 anchor = v0;
    // Test for degeneracy before normalizing: a zero-extent bbox face gives cross() == 0, and
    // the resulting NaN normal would slip past the parallel-ray guard below, since
    // abs(NaN) < 1e-8 evaluates to false.
    vec3 face_normal = cross(v1 - v0, v2 - v0);
    float face_normal_length = length(face_normal);
    if (face_normal_length < 1e-20) return false; // Degenerate face
    vec3 normal = face_normal / face_normal_length;

    float denom = dot(ray_dir, normal);
    if (abs(denom) < 1e-8) return false; // Ray parallel to face

    uint opposite_base = (face_idx ^ 1u) * 12;
    vec3 opposite_v0 = vec3(bbox_verts_buf.vertices[opposite_base + 0], bbox_verts_buf.vertices[opposite_base + 1], bbox_verts_buf.vertices[opposite_base + 2]);
    if (dot(ray_dir, v0 - opposite_v0) <= 0.0) return false; // Ray entering the domain through this face

    t = dot(anchor - ray_origin, normal) / denom;
    if (t > 1e8) return false; // Too far
    // A ray that left the periodic copies continues from beyond the face it left through (see check_periodic_boundary())
    if (periodic_info_buf.tile_tracking != 0u && t < -0.5 * periodic_info_buf.escape_offset) return false;

    // Check if hit point is inside the rectangle
    vec3 p = ray_origin + ray_dir * t;
    vec3 d = p - anchor;
    vec3 a = v1 - v0;
    vec3 b = v3 - v0;

    float ddota = dot(d, a);
    float ddotb = dot(d, b);

    return ddota > 0.0 && ddota < dot(a, a) && ddotb > 0.0 && ddotb < dot(b, b);
}

/**
 * Check all bbox faces for periodic boundary intersection and compute wrapped origin.
 *
 * Tests each bbox face against the ray. If the closest bbox hit is nearer than
 * the closest real geometry hit (bvh_closest_t), computes the wrapped ray origin
 * by translating the hit point across the domain.
 *
 * Wrapping logic matches OptiX rayHit.cu:41-65:
 * - Hit at -x boundary → wrap to +x (add width_x)
 * - Hit at +x boundary → wrap to -x (subtract width_x)
 * - Same for Y boundaries
 *
 * Returns true if a periodic boundary was hit (caller should re-trace from wrapped_origin), and sets boundary_t to the
 * distance along the ray to the boundary.
 */
bool check_periodic_boundary(
    vec3 ray_origin, vec3 ray_dir, float bvh_closest_t,
    uint periodic_flag_x, uint periodic_flag_y, uint bbox_count,
    float domain_xmin, float domain_xmax, float domain_ymin, float domain_ymax,
    out vec3 wrapped_origin, out float boundary_t
) {
    float bbox_t = bvh_closest_t; // Only consider bbox hits closer than real geometry
    int closest_face = -1;

    for (uint b = 0; b < bbox_count; ++b) {
        float t;
        if (intersect_bbox_face(ray_origin, ray_dir, b, t) && t < bbox_t) {
            bbox_t = t;
            closest_face = int(b);
        }
    }

    if (closest_face < 0) return false; // No bbox hit closer than real geometry

    // Compute hit position on bbox face
    boundary_t = bbox_t;
    vec3 hit_pos = ray_origin + ray_dir * bbox_t;

    // Compute wrapped position (matches OptiX rayHit.cu:41-65)
    // Tolerance for identifying the face that was hit. The ray continues from exactly one period away, the same point on the
    // opposite face: moving it any further would shift it off its straight path by that much at every wrap.
    float eps = 1e-5;
    float width_x = domain_xmax - domain_xmin;
    float width_y = domain_ymax - domain_ymin;

    // Leaving through a low face moves the ray into the next periodic copy toward -x (or -y), and through a high face toward +x (or +y)
    ivec2 step = ivec2(0);
    wrapped_origin = hit_pos;
    if (periodic_flag_x == 1 && abs(hit_pos.x - domain_xmin) <= eps) {
        wrapped_origin.x += width_x; // -x boundary → wrap to +x
        step.x = -1;
    } else if (periodic_flag_x == 1 && abs(hit_pos.x - domain_xmax) <= eps) {
        wrapped_origin.x -= width_x; // +x boundary → wrap to -x
        step.x = 1;
    } else if (periodic_flag_y == 1 && abs(hit_pos.y - domain_ymin) <= eps) {
        wrapped_origin.y += width_y; // -y boundary → wrap to +y
        step.y = -1;
    } else if (periodic_flag_y == 1 && abs(hit_pos.y - domain_ymax) <= eps) {
        wrapped_origin.y -= width_y; // +y boundary → wrap to -y
        step.y = 1;
    }

    if (periodic_info_buf.tile_tracking == 0u) {
        set_periodic_tile(periodic_tile + step); // only the copy counts, for the ray to hit the periodic copies of the primitive it was launched from
        return true;
    }

    // The same logic as handlePeriodicBoundaryHit() in the OptiX 8 device code: a ray that leaves the field of copies can hit nothing
    // more; outside the central copy, which it can no longer return to, only repeated geometry can be hit, so a ray above (below) it
    // heading up (down) can hit nothing more, and one heading toward it is moved to it in one step, into the domain by a whole number
    // of periods. A ray that can hit nothing more continues from just beyond the face it left through, where it hits no face again
    // and leaves the scene as a ray that missed everything.
    ivec2 tile = periodic_tile + step;
    bool leaves_field = (periodic_info_buf.max_repeats_x >= 0 && abs(tile.x) > periodic_info_buf.max_repeats_x && float(tile.x) * ray_dir.x > 0.0) ||
                        (periodic_info_buf.max_repeats_y >= 0 && abs(tile.y) > periodic_info_buf.max_repeats_y && float(tile.y) * ray_dir.y > 0.0);
    bool leaves_central_copy = (tile.x != 0 && float(tile.x) * ray_dir.x >= 0.0) || (tile.y != 0 && float(tile.y) * ray_dir.y >= 0.0);
    float zmin = periodic_info_buf.repeated_zmin;
    float zmax = periodic_info_buf.repeated_zmax;
    bool escapes = leaves_field || (leaves_central_copy && ((wrapped_origin.z > zmax && ray_dir.z >= 0.0) || (wrapped_origin.z < zmin && ray_dir.z <= 0.0)));
    if (!escapes && leaves_central_copy && (wrapped_origin.z > zmax || wrapped_origin.z < zmin)) {
        float target_z = wrapped_origin.z > zmax ? zmax : zmin;
        float advance = (target_z - wrapped_origin.z) / ray_dir.z;
        vec3 moved = wrapped_origin + ray_dir * advance;
        moved.z = target_z;
        if (periodic_flag_x == 1u) {
            float unwrapped = moved.x;
            moved.x = wrap_into_period(moved.x, ray_dir.x, domain_xmin, domain_xmax);
            tile.x += int(round((unwrapped - moved.x) / width_x));
        }
        if (periodic_flag_y == 1u) {
            float unwrapped = moved.y;
            moved.y = wrap_into_period(moved.y, ray_dir.y, domain_ymin, domain_ymax);
            tile.y += int(round((unwrapped - moved.y) / width_y));
        }
        escapes = (periodic_info_buf.max_repeats_x >= 0 && abs(tile.x) > periodic_info_buf.max_repeats_x && float(tile.x) * ray_dir.x > 0.0) ||
                  (periodic_info_buf.max_repeats_y >= 0 && abs(tile.y) > periodic_info_buf.max_repeats_y && float(tile.y) * ray_dir.y > 0.0);
        if (!escapes) {
            wrapped_origin = moved;
            boundary_t = bbox_t + advance; // the distance to the point the ray continues from
        }
    }
    if (escapes) {
        wrapped_origin = hit_pos + vec3(float(step.x), float(step.y), 0.0) * periodic_info_buf.escape_offset;
        boundary_t = bbox_t;
        return true;
    }
    set_periodic_tile(tile);
    return true;
}

// As above, for callers that do not need the distance to the boundary.
bool check_periodic_boundary(
    vec3 ray_origin, vec3 ray_dir, float bvh_closest_t,
    uint periodic_flag_x, uint periodic_flag_y, uint bbox_count,
    float domain_xmin, float domain_xmax, float domain_ymin, float domain_ymax,
    out vec3 wrapped_origin
) {
    float boundary_t;
    return check_periodic_boundary(ray_origin, ray_dir, bvh_closest_t, periodic_flag_x, periodic_flag_y, bbox_count, domain_xmin, domain_xmax, domain_ymin, domain_ymax,
                                   wrapped_origin, boundary_t);
}

/**
 * Move a camera ray that starts above or below the periodic boundary faces to the top or bottom of the faces.
 *
 * The faces span only the height of the geometry, so a ray starting above (below) them passes nothing it can hit before it
 * descends (rises) to their top (bottom). It is moved there in one step, then into the domain laterally by a whole number of
 * periods, which is where it would have been wrapped to on the way. A ray that never reaches the faces' height range is left
 * where it is: it hits nothing either way. Must match advanceToPeriodicFaces() in the OptiX backends.
 *
 * Returns the distance the ray was moved along its direction (0 if it was not moved).
 */
float advance_to_periodic_faces(
    inout vec3 ray_origin, vec3 ray_dir,
    uint periodic_flag_x, uint periodic_flag_y,
    float domain_xmin, float domain_xmax, float domain_ymin, float domain_ymax
) {
    // Vertices 0 and 2 of every face are at the bottom and the top of the faces
    float face_zmin = bbox_verts_buf.vertices[2];
    float face_zmax = bbox_verts_buf.vertices[8];

    float face_z;
    if (ray_origin.z > face_zmax && ray_dir.z < 0.0) {
        face_z = face_zmax;
    } else if (ray_origin.z < face_zmin && ray_dir.z > 0.0) {
        face_z = face_zmin;
    } else {
        return 0.0;
    }

    float t = (face_z - ray_origin.z) / ray_dir.z;
    vec3 moved = ray_origin + ray_dir * t;
    moved.z = face_z;
    // The whole periods the ray is moved by laterally are added to the periodic copy of the domain it is in
    ivec2 tile = periodic_tile;
    if (periodic_flag_x == 1u) {
        float unwrapped = moved.x;
        moved.x = wrap_into_period(moved.x, ray_dir.x, domain_xmin, domain_xmax);
        tile.x += int(round((unwrapped - moved.x) / (domain_xmax - domain_xmin)));
    }
    if (periodic_flag_y == 1u) {
        float unwrapped = moved.y;
        moved.y = wrap_into_period(moved.y, ray_dir.y, domain_ymin, domain_ymax);
        tile.y += int(round((unwrapped - moved.y) / (domain_ymax - domain_ymin)));
    }
    set_periodic_tile(tile);
    ray_origin = moved;
    return t;
}

#endif // PERIODIC_BOUNDARY_GLSL
