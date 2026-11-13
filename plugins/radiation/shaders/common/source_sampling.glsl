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

// Sampling a radiation source and tracing a ray toward the point sampled on it, shared by the direct launch (direct_raygen.comp) and by the cameras' per-pixel shadows (camera_raygen.comp), so that
// both draw the same distribution of points on each source and decide alike whether a ray reaches it. GLSL port of sampleSourceDirection() in OptiX8DeviceCode.cu and of the direct ray type's hit
// programs.
//
// The including shader must declare the source buffers (source_types_buf, source_widths_buf, with get_source_position() and get_source_rotation()), normals_buf, and the functions sample_disk() and
// sample_square(), acos_safe() and PI (sky_distribution.glsl provides the last two), periodic_excluded_buf, and include random.glsl, transforms.glsl, bvh_traversal.glsl, periodic_boundary.glsl and glass_cover.glsl first.
#ifndef SOURCE_SAMPLING_GLSL
#define SOURCE_SAMPLING_GLSL

// One sample of radiation source s seen from the point `origin`, drawn as the direct launch draws it. Gives the direction toward the point sampled on the source (the source's direction for a collimated
// source), that point (the source's direction for a collimated source, where no point is sampled), and the factor that, times the source's flux and |n . direction| for a surface of normal n, gives the
// irradiance the sample stands for: the strength of a direct ray before its division by the number of rays. Returns false for a sample on the back of a rectangle or disk source, which delivers
// nothing, for a point that coincides with the sampled point, and for an unknown source type.
bool sample_source_direction(uint s, vec3 origin, inout RNGState rng, out vec3 ray_dir, out vec3 source_target, out float factor) {
    vec3 source_pos = get_source_position(s);
    uint source_type = source_types_buf.types[s];
    ray_dir = vec3(0.0);
    source_target = source_pos;
    factor = 0.0;

    if (source_type == 0u) { // Collimated source (sun): source_pos is its direction
        ray_dir = normalize(source_pos);
        factor = 1.0;
        return true;
    }
    if (source_type == 1u || source_type == 2u) { // Sphere or sun-sphere source
        // Sample a random point on the sphere's surface (matches CUDA rayGeneration.cu)
        float theta_s = acos_safe(1.0 - 2.0 * rng_float(rng));
        float phi_s = rng_float(rng) * 2.0 * PI;
        float diameter = source_widths_buf.widths[s].x; // width stores the diameter
        vec3 sphere_point = 0.5 * diameter * vec3(sin(theta_s) * cos(phi_s), sin(theta_s) * sin(phi_s), cos(theta_s));
        float ray_magnitude;
        source_target = sphere_point + source_pos;
        ray_dir = safe_normalize(source_target - origin, ray_magnitude);
        if (ray_magnitude < 1e-20) {
            return false; // the point coincides with the sampled point on the source
        }
        // Numerical integration over the visible hemisphere of the sphere
        const uint N = 10;
        for (uint j = 0; j < N; j++) {
            for (uint i = 0; i < N; i++) {
                float theta = acos_safe(1.0 - 2.0 * (float(i) + 0.5) / float(N));
                float phi = (float(j) + 0.5) * 2.0 * PI / float(N);
                vec3 light_direction = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
                if (dot(light_direction, ray_dir) < 0.0) {
                    factor += abs(dot(light_direction, ray_dir)) / (ray_magnitude * ray_magnitude) / float(N * N) * diameter * diameter;
                }
            }
        }
        return true;
    }
    if (source_type == 3u) { // Rectangle source
        vec2 source_size = source_widths_buf.widths[s]; // (width, height)
        vec3 square_point = sample_square(rng);
        square_point.x *= source_size.x;
        square_point.y *= source_size.y;
        mat4 light_transform = make_rotation_matrix(get_source_rotation(s));
        vec3 light_point = transform_point(light_transform, square_point);
        vec3 light_direction = normalize(transform_direction(light_transform, vec3(0, 0, 1)));
        source_target = light_point + source_pos;
        ray_dir = source_target - origin;
        if (dot(ray_dir, light_direction) > 0.0) {
            return false; // the source does not emit from its back
        }
        float ray_magnitude;
        ray_dir = safe_normalize(ray_dir, ray_magnitude);
        if (ray_magnitude < 1e-20) {
            return false;
        }
        factor = abs(dot(light_direction, ray_dir)) / (ray_magnitude * ray_magnitude) * source_size.x * source_size.y / PI;
        return true;
    }
    if (source_type == 4u) { // Disk source
        float radius = source_widths_buf.widths[s].x; // x stores the radius
        vec3 disk_point = sample_disk(rng);
        mat4 light_transform = make_rotation_matrix(get_source_rotation(s));
        vec3 light_point = transform_point(light_transform, disk_point);
        vec3 light_direction = normalize(transform_direction(light_transform, vec3(0, 0, 1)));
        source_target = radius * light_point + source_pos; // scaled after the transform, as in the OptiX backends
        ray_dir = source_target - origin;
        if (dot(ray_dir, light_direction) > 0.0) {
            return false;
        }
        float ray_magnitude;
        ray_dir = safe_normalize(ray_dir, ray_magnitude);
        if (ray_magnitude < 1e-20) {
            return false;
        }
        factor = abs(dot(light_direction, ray_dir)) / (ray_magnitude * ray_magnitude) * radius * radius;
        return true;
    }
    return false; // unknown source type
}

// Outcomes of trace_to_source()
#define SOURCE_REACHED 0u
#define SOURCE_BLOCKED 1u
#define SOURCE_WRAP_LIMIT 2u // blocked at the periodic wrap limit

// Trace a ray from `origin` along ray_dir toward a source, distance_to_source away for a finite source (ignored for a collimated one), skipping the primitive origin_prim it leaves. A translucent cover
// (glass in every launched band) passes the ray and multiplies cover_tau by its transmittance in each band; anything else blocks it, and the ray wraps through periodic boundaries. A ray that runs out
// of wraps or segments counts as blocked.
// A dual ray (see traversal_record_excluded in bvh_traversal.glsl) also passes the primitives excluded from the periodic boundary: it reaches the source if it does in the background radiation field,
// and traversal_excluded_passed tells on return whether an excluded primitive blocks it in the scene as it is. Must match __anyhit__direct_dual in OptiX8DeviceCode.cu.
// The ray starts in the periodic copy of the domain the caller's ray is in (see set_periodic_tile()), which is restored on return.
uint trace_to_source_from_tile(vec3 origin, vec3 ray_dir, uint origin_prim, bool finite_source, float distance_to_source, uint periodic_flag_x, uint periodic_flag_y, uint bbox_count, float domain_xmin,
                               float domain_xmax, float domain_ymin, float domain_ymax, uint band_count, uint material_band_count, inout float cover_tau[GLASS_MAX_BANDS]) {
    vec3 trace_origin = origin;
    int boundary_hits = 0;
    // No minimum distance: the primitive the ray was launched from is skipped (origin_prim), and a minimum distance let a ray launched that close to an adjacent surface pass through it where the two meet
    float segment_t_min = 0.0;
    for (int step = 0; step < MAX_RAY_SEGMENTS; ++step) {
        float occlusion_t;
        uint occluder = traverse_cwbvh(trace_origin, ray_dir, segment_t_min, origin_prim, occlusion_t);

        // Periodic boundary wrap (a bbox face is not a real occluder)
        if (bbox_count != 0) {
            vec3 wrapped_origin;
            float boundary_t;
            if (check_periodic_boundary(trace_origin, ray_dir, occlusion_t, periodic_flag_x, periodic_flag_y, bbox_count, domain_xmin, domain_xmax, domain_ymin, domain_ymax, wrapped_origin, boundary_t)) {
                if (finite_source && boundary_t >= distance_to_source) {
                    return SOURCE_REACHED; // the source is reached before the boundary
                }
                if (++boundary_hits >= MAX_PERIODIC_WRAPS) {
                    return SOURCE_WRAP_LIMIT;
                }
                distance_to_source -= boundary_t;
                trace_origin = wrapped_origin;
                segment_t_min = 0.0; // a wrapped ray hits geometry right at the boundary it was wrapped to
                continue;
            }
        }

        if (occluder == 0xFFFFFFFFu) {
            return SOURCE_REACHED; // a miss
        }
        // occlusion_t is measured from trace_origin, which is past any boundaries and translucent covers already crossed
        if (finite_source && occlusion_t >= distance_to_source) {
            return SOURCE_REACHED; // the hit is behind the source
        }
        // A dual ray whose closest hit is on an excluded primitive is blocked in the scene as it is; the segment is traced again without the excluded primitives, for the background field
        if (traversal_record_excluded && !traversal_excluded_passed && periodic_excluded_buf.excluded[occluder] != 0u) {
            traversal_excluded_passed = true;
            set_periodic_tile(periodic_tile);
            continue;
        }
        // A primitive that is not a translucent cover in all launched bands is an opaque blocker
        if (!glass_cover_all_bands(occluder, band_count, material_band_count)) { // origin_prim is hit only in another periodic copy, as any other primitive
            return SOURCE_BLOCKED;
        }

        // Glass pass-through: accumulate the per-band transmittance. What the cover itself absorbs and reflects is booked by the cover's own rays, not here.
        vec3 cover_normal = vec3(normals_buf.normals[occluder * 3], normals_buf.normals[occluder * 3 + 1], normals_buf.normals[occluder * 3 + 2]);
        float cover_cos = abs(dot(ray_dir, normalize(cover_normal)));
        for (uint band = 0; band < band_count && band < uint(GLASS_MAX_BANDS); ++band) {
            cover_tau[band] *= glass_cover_tau_rho_alpha(occluder, band, material_band_count, cover_cos).x;
        }

        // Advance just past the cover hit and continue toward the source
        trace_origin = trace_origin + ray_dir * (occlusion_t + 1e-4);
        distance_to_source -= occlusion_t + 1e-4;
        segment_t_min = 1e-4;
    }
    return SOURCE_BLOCKED; // out of segments
}

uint trace_to_source(vec3 origin, vec3 ray_dir, uint origin_prim, bool finite_source, float distance_to_source, uint periodic_flag_x, uint periodic_flag_y, uint bbox_count, float domain_xmin, float domain_xmax,
                     float domain_ymin, float domain_ymax, uint band_count, uint material_band_count, inout float cover_tau[GLASS_MAX_BANDS]) {
    ivec2 caller_tile = periodic_tile;
    ivec2 caller_origin_tile = periodic_origin_tile;
    traversal_excluded_passed = false;
    start_periodic_ray(caller_tile); // the point the ray leaves is on the copy of origin_prim in the caller's copy of the domain
    uint outcome = trace_to_source_from_tile(origin, ray_dir, origin_prim, finite_source, distance_to_source, periodic_flag_x, periodic_flag_y, bbox_count, domain_xmin, domain_xmax, domain_ymin, domain_ymax,
                                             band_count, material_band_count, cover_tau);
    periodic_origin_tile = caller_origin_tile;
    set_periodic_tile(caller_tile);
    return outcome;
}

#endif // SOURCE_SAMPLING_GLSL
