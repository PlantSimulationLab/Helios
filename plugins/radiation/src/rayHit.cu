/** \file "rayHit.cu" Functions for object intersection.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/


#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "RayTracing.cuh"
#include "BufferIndexing.h"

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable(unsigned int, UUID, attribute UUID, );

// ---------------------------------------------------------------------------
// Translucent cover (glass/plastic) optical model
// ---------------------------------------------------------------------------

//! Angular transmittance/reflectance/absorptance of a single dielectric sheet via Fresnel reflection
//! (two interfaces, both polarizations, internal reflections summed in closed form) plus Bouguer
//! absorption (Duffie & Beckman solar-glazing model). Byte-for-byte port of glass_tau_rho_alpha() in
//! optix8/OptiX8DeviceCode.cu and shaders/common/glass_cover.glsl — KEEP NUMERICALLY IDENTICAL.
//! Inputs:  cos_theta = |cos| of incidence angle, n = refractive index (>1), KL = absorption product K*L.
//! Returns: float3(tau, rho, alpha) with tau + rho + alpha == 1.
static __device__ __inline__ float3 glass_tau_rho_alpha(float cos_theta, float n, float KL) {
    cos_theta = fmaxf(1e-4f, fminf(1.f, cos_theta)); // guard grazing/degenerate
    const float theta = acos_safe(cos_theta);
    const float sin_t = sinf(theta);
    const float sin_tr = sin_t / n; // Snell
    const float cos_tr = sqrtf(fmaxf(0.f, 1.f - sin_tr * sin_tr));
    const float theta_r = asin_safe(sin_tr);

    float r_par, r_per;
    if (theta < 1e-3f) {
        const float r0 = ((n - 1.f) / (n + 1.f)) * ((n - 1.f) / (n + 1.f));
        r_par = r0;
        r_per = r0;
    } else {
        const float s_minus = sinf(theta_r - theta);
        const float s_plus = sinf(theta_r + theta);
        const float t_minus = tanf(theta_r - theta);
        const float t_plus = tanf(theta_r + theta);
        r_per = (s_minus * s_minus) / fmaxf(1e-12f, s_plus * s_plus); // perpendicular
        r_par = (t_minus * t_minus) / fmaxf(1e-12f, t_plus * t_plus); // parallel
    }

    const float tau_a = (KL > 0.f) ? expf(-KL / fmaxf(1e-4f, cos_tr)) : 1.f; // Bouguer absorption

    float tau = 0.f, rho = 0.f;
    for (int pol = 0; pol < 2; pol++) {
        const float r = (pol == 0) ? r_per : r_par;
        const float denom = fmaxf(1e-6f, 1.f - (r * tau_a) * (r * tau_a));
        const float tau_i = tau_a * (1.f - r) * (1.f - r) / denom;
        const float rho_i = r * (1.f + tau_a * tau_i);
        tau += 0.5f * tau_i;
        rho += 0.5f * rho_i;
    }
    tau = fmaxf(0.f, fminf(1.f, tau));
    rho = fmaxf(0.f, fminf(1.f, rho));
    float alpha = 1.f - tau - rho;
    if (alpha < 0.f) {
        alpha = 0.f;
    } // numerical guard
    return make_float3(tau, rho, alpha);
}

//! Reconstruct the surface normal of primitive `hit_position` from its transform matrix and type,
//! returning |cos(incidence angle)| and the hit face (face_top = true when the ray strikes the
//! +normal/top side). Mirrors the normal logic in closest_hit_direct/diffuse and the OptiX 8
//! coverCosTheta().
static __device__ __inline__ float coverNormalFaceCos(uint hit_position, const float3 &ray_dir, bool &face_top) {
    float m[16];
    for (uint i = 0; i < 16; i++) {
        m[i] = transform_matrix[optix::make_uint2(i, hit_position)];
    }
    float3 normal = make_float3(0.f, 0.f, 1.f);
    const uint ptype = primitive_type[hit_position];
    if (ptype == 0 || ptype == 3) { // patch or tile
        float3 s0 = make_float3(0, 0, 0), s1 = make_float3(1, 0, 0), s2 = make_float3(0, 1, 0);
        d_transformPoint(m, s0);
        d_transformPoint(m, s1);
        d_transformPoint(m, s2);
        normal = cross(s1 - s0, s2 - s0);
    } else if (ptype == 1) { // triangle
        float3 v0 = make_float3(0, 0, 0), v1 = make_float3(0, 1, 0), v2 = make_float3(1, 1, 0);
        d_transformPoint(m, v0);
        d_transformPoint(m, v1);
        d_transformPoint(m, v2);
        normal = cross(v1 - v0, v2 - v0);
    } else if (ptype == 2) { // disk
        float3 v0 = make_float3(0, 0, 0), v1 = make_float3(1, 0, 0), v2 = make_float3(0, 1, 0);
        d_transformPoint(m, v0);
        d_transformPoint(m, v1);
        d_transformPoint(m, v2);
        normal = cross(v1 - v0, v2 - v0);
    }
    const float nmag = d_magnitude(normal);
    if (nmag < 1e-12f) {
        face_top = true;
        return 1.f;
    }
    normal = normal / nmag;
    face_top = dot(normal, ray_dir) < 0.f;
    return fminf(1.f, fabsf(dot(ray_dir, normal)));
}

RT_PROGRAM void closest_hit_direct() {

    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if position is invalid
    if (hit_position == UINT_MAX) {
        return;
    }

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }
    }
};

RT_PROGRAM void closest_hit_diffuse() {

    // Convert UUIDs to array positions for buffer indexing
    uint origin_UUID = prd.origin_UUID;
    uint origin_position = primitive_positions[origin_UUID];
    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if positions are invalid
    if (origin_position == UINT_MAX || hit_position == UINT_MAX) {
        return;
    }

    // Create indexers for buffer access
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    MaterialPropertyIndexer mat_indexer(Nsources, Nprimitives, Nbands_global);
    CameraMaterialIndexer cam_mat_indexer(Nsources, Nprimitives, Nbands_global, Ncameras);

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }

    } else {

        // Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being recieved)

        // find out if we hit top or bottom surface
        float3 normal;

        float m[16];
        for (uint i = 0; i < 16; i++) {
            m[i] = transform_matrix[optix::make_uint2(i, hit_position)];
        }

        if (primitive_type[hit_position] == 0 || primitive_type[hit_position] == 3) { // hit patch or tile
            float3 s0 = make_float3(0, 0, 0);
            float3 s1 = make_float3(1, 0, 0);
            float3 s2 = make_float3(0, 1, 0);
            d_transformPoint(m, s0);
            d_transformPoint(m, s1);
            d_transformPoint(m, s2);
            normal = cross(s1 - s0, s2 - s0);
        } else if (primitive_type[hit_position] == 1) { // hit triangle
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(0, 1, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(1, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        } else if (primitive_type[hit_position] == 2) { // hit disk
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(1, 0, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(0, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        } else if (primitive_type[hit_position] == 4) { // hit voxel
            float3 vmin = make_float3(-0.5, -0.5, -0.5);
            d_transformPoint(m, vmin);
            float3 vmax = make_float3(0.5, 0.5, 0.5);
            d_transformPoint(m, vmax);
        }
        normal = normalize(normal);

        bool face = dot(normal, ray.direction) < 0;

        int b = -1;
        for (int b_global = 0; b_global < Nbands_global; b_global++) {

            if (band_launch_flag[b_global] == 0) {
                continue;
            }
            b++;

            // Use BufferIndexer for radiation buffers: [primitive][band]
            // NOTE: b should match the band's index in the original band_labels array
            // This is correct as long as band_launch_flag isn't modified after launch
            size_t ind_origin = rad_indexer(origin_position, b);
            size_t ind_hit = rad_indexer(hit_position, b);

            double strength;
            if (face || primitive_type[hit_position] == 4) {
                strength = radiation_out_top[ind_hit] * prd.strength;
            } else {
                strength = radiation_out_bottom[ind_hit] * prd.strength;
            }

            if (strength == 0) {
                continue;
            }

            // Use BufferIndexer for material properties: [source][primitive][band]
            size_t radprop_ind_global = mat_indexer(prd.source_ID, origin_position, b_global);
            float t_rho = rho[radprop_ind_global];
            float t_tau = tau[radprop_ind_global];

            // Check if ray was launched from voxel (type 4)
            if (primitive_type[origin_position] == 4) { // ray was launched from voxel

                //                float kappa = t_rho; //just a reminder that rho is actually the absorption coefficient
                //                float sigma_s = t_tau; //just a reminder that tau is actually the scattering coefficient
                //                float beta = kappa + sigma_s;
                //
                //                // absorption
                //                atomicAdd(&radiation_in[ind_origin], strength * exp(-beta * 0.5 * prd.area) * kappa / beta);
                //
                //                // scattering
                //                atomicAdd(&scatter_buff_top[ind_origin], strength * exp(-beta * 0.5 * prd.area) * sigma_s / beta);

            } else { // ray was NOT launched from voxel

                // absorption - calculate with defensive check for energy conservation violations
                float absorption_factor = 1.f - t_rho - t_tau;
                float contribution = strength * absorption_factor;


#ifndef NDEBUG
                if (absorption_factor < -1e-5f) {
                    printf("ERROR: Negative absorption! rho=%.6f, tau=%.6f, origin_UUID=%u\n", t_rho, t_tau, origin_UUID);
                    absorption_factor = 0.f;
                }
#endif
                atomicAdd(&radiation_in[ind_origin], contribution);

                if ((t_rho > 0 || t_tau > 0) && strength > 0) {
                    if (prd.face) { // reflection from top, transmission from bottom
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_tau); // transmission
                    } else { // reflection from bottom, transmission from top
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_tau); // transmission
                    }
                }
                if (Ncameras > 0) {
                    // Use BufferIndexer for camera material: [source][primitive][band][camera]
                    size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, camera_ID);
                    float t_rho_cam = rho_cam[indc];
                    float t_tau_cam = tau_cam[indc];
                    if ((t_rho_cam > 0 || t_tau_cam > 0) && strength > 0) {
                        if (prd.face) { // reflection from top, transmission from bottom
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_rho_cam); // reflection
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_tau_cam); // transmission
                        } else { // reflection from bottom, transmission from top
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_rho_cam); // reflection
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_tau_cam); // transmission
                        }
                    }
                    // Note: Don't accumulate scattered radiation to radiation_specular
                    // Specular should only reflect DIRECT source radiation (accumulated in miss_direct)
                }
            }

            // if( primitive_type[UUID] == 4 ){ //if we hit a voxel, reduce strength and launch another ray
            //   optix::Ray ray_transmit = optix::make_Ray(ray.origin+(t_hit+prd.area+1e-5)*ray.direction, ray.direction, ray.ray_type, 1e-4, RT_DEFAULT_MAX);
            //   PerRayData prd_transmit = prd;
            //   float beta = rho[UUID]+tau[UUID];
            //   prd_transmit.strength = prd.strength*(1.f-exp(-beta*0.5*prd.area));
            //   rtTrace( top_object, ray_transmit, prd_transmit);
            // }
        }
    }
}

RT_PROGRAM void closest_hit_camera() {

    // Convert UUID to array position
    uint hit_position = primitive_positions[UUID];

    // Bounds check: skip if position is invalid
    if (hit_position == UINT_MAX) {
        return;
    }

    // For cameras, origin_UUID is actually the pixel index (not a primitive UUID!)
    uint pixel_index = prd.origin_UUID;
    size_t Npixels = camera_resolution_full.x * camera_resolution_full.y;

    // Create indexers
    RadiationBufferIndexer rad_indexer(Npixels, Nbands_launch); // Use Npixels for camera radiation buffer
    SourceFluxIndexer source_flux_indexer(Nsources, Nbands_launch);
    SpecularRadiationIndexer spec_indexer(Nsources, Ncameras, Nprimitives, Nbands_launch);

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }

    } else {

        // Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being received)

        // find out if we hit top or bottom surface per each ray
        float3 normal;

        float m[16];
        for (uint i = 0; i < 16; i++) {
            m[i] = transform_matrix[optix::make_uint2(i, hit_position)];
        }

        if (primitive_type[hit_position] == 0 || primitive_type[hit_position] == 3) { // hit patch or tile
            float3 s0 = make_float3(0, 0, 0);
            float3 s1 = make_float3(1, 0, 0);
            float3 s2 = make_float3(0, 1, 0);
            d_transformPoint(m, s0);
            d_transformPoint(m, s1);
            d_transformPoint(m, s2);
            normal = cross(s1 - s0, s2 - s0);
        } else if (primitive_type[hit_position] == 1) { // hit triangle
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(0, 1, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(1, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        } else if (primitive_type[hit_position] == 2) { // hit disk
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(1, 0, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(0, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        } else if (primitive_type[hit_position] == 4) { // hit voxel
            float3 vmin = make_float3(-0.5, -0.5, -0.5);
            d_transformPoint(m, vmin);
            float3 vmax = make_float3(0.5, 0.5, 0.5);
            d_transformPoint(m, vmax);
        }
        normal = normalize(normal);

        bool face = dot(normal, ray.direction) < 0;


        float3 camera_normal = d_rotatePoint(make_float3(0, 0, 1), -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y);

        double strength;
        for (size_t b = 0; b < Nbands_launch; b++) {

            // Check if any light sources are between camera and hit point
            float source_radiance = 0.0f;
            for (uint s = 0; s < Nsources; s++) {
                float flux = source_fluxes[s * Nbands_launch + b];
                if (flux <= 0.0f)
                    continue;

                uint source_type = source_types[s];

                if (source_type == 1) {
                    // SPHERE
                    float radius = source_widths[s].x * 0.5f;
                    float3 oc = ray.origin - source_positions[s];
                    float b = dot(oc, ray.direction);
                    float c = dot(oc, oc) - radius * radius;
                    float discriminant = b * b - c;

                    if (discriminant >= 0.0f) {
                        float t_sphere = -b - sqrtf(discriminant);
                        if (t_sphere > 0.0f && t_sphere < t_hit) {
                            float area = 4.0f * M_PI * radius * radius;
                            source_radiance += (flux / area) / M_PI;
                        }
                    }
                } else if (source_type == 3) {
                    // RECTANGLE
                    float transform[16];
                    d_makeTransformMatrix(source_rotations[s], transform);
                    float3 normal = make_float3(transform[2], transform[6], transform[10]);

                    float denom = dot(ray.direction, normal);
                    if (denom < -1e-6f) {
                        float3 oc = source_positions[s] - ray.origin;
                        float t_rect = dot(oc, normal) / denom;

                        if (t_rect > 0.0f && t_rect < t_hit) {
                            float3 hit_point = ray.origin + t_rect * ray.direction;
                            float3 local_hit = hit_point - source_positions[s];
                            float inv_transform[16];
                            d_invertMatrix(transform, inv_transform);
                            d_transformPoint(inv_transform, local_hit);

                            if (fabsf(local_hit.x) <= source_widths[s].x * 0.5f && fabsf(local_hit.y) <= source_widths[s].y * 0.5f) {
                                float area = source_widths[s].x * source_widths[s].y;
                                float cos_angle = -denom;
                                source_radiance += (flux / area) * cos_angle / M_PI;
                            }
                        }
                    }
                } else if (source_type == 4) {
                    // DISK
                    float transform[16];
                    d_makeTransformMatrix(source_rotations[s], transform);
                    float3 normal = make_float3(transform[2], transform[6], transform[10]);

                    float denom = dot(ray.direction, normal);
                    if (denom < -1e-6f) {
                        float3 oc = source_positions[s] - ray.origin;
                        float t_disk = dot(oc, normal) / denom;

                        if (t_disk > 0.0f && t_disk < t_hit) {
                            float3 hit_point = ray.origin + t_disk * ray.direction;
                            float3 offset = hit_point - source_positions[s];
                            float dist_sq = dot(offset, offset);

                            float radius = source_widths[s].x;
                            if (dist_sq <= radius * radius) {
                                float area = M_PI * radius * radius;
                                float cos_angle = -denom;
                                source_radiance += (flux / area) * cos_angle / M_PI;
                            }
                        }
                    }
                }
            }

            // Use BufferIndexer: [primitive][band]
            size_t ind_hit = rad_indexer(hit_position, b);

            if (face || primitive_type[hit_position] == 4) {
                strength = radiation_out_top[ind_hit] * prd.strength;
            } else {
                strength = radiation_out_bottom[ind_hit] * prd.strength;
            }

            if (source_radiance > 0.0f) {
                strength += source_radiance * prd.strength;
            }

            // specular reflection
            // Only compute specular on iteration 0 to prevent accumulation across scattering iterations.
            // radiation_specular contains per-source, camera-weighted incident radiation

            double strength_spec = 0;
            if (specular_reflection_enabled > 0 && specular_exponent[hit_position] > 0.f && scattering_iteration == 0) {

                // For each source, compute specular contribution
                for (int rr = 0; rr < Nsources; rr++) {

                    // Get camera-weighted incident radiation from this source
                    // Already has source color and camera response weighting applied
                    // Use BufferIndexer: [source][camera][primitive][band]
                    size_t ind_specular = spec_indexer(rr, camera_ID, hit_position, b);
                    float spec = radiation_specular[ind_specular];

                    // Apply default 0.25 scaling factor (typical Fresnel reflectance for dielectrics is ~4%,
                    // but this accounts for specular lobe concentration and typical surface roughness)
                    spec *= 0.25f;

                    if (spec > 0) {
                        // Determine light direction based on source type
                        float3 light_direction;
                        if (source_types[rr] == 0 || source_types[rr] == 2) {
                            // Collimated or sunsphere: parallel rays from source direction
                            light_direction = normalize(source_positions[rr]);
                        } else {
                            // Sphere, disk, or rectangle: direction from hit point to source center
                            float3 hit_point = ray.origin + t_hit * ray.direction;
                            light_direction = normalize(source_positions[rr] - hit_point);
                        }

                        // Blinn-Phong specular direction (half-vector)
                        float3 specular_direction = normalize(light_direction - ray.direction);

                        float exponent = specular_exponent[hit_position];
                        double scale_coefficient = 1.0;
                        if (specular_reflection_enabled == 2) { // if we are using the scale coefficient
                            scale_coefficient = specular_scale[hit_position];
                        }

                        strength_spec += spec * scale_coefficient * pow(max(0.f, dot(specular_direction, normal)), exponent) * (exponent + 2.f) /
                                         (double(launch_dim.x) * 2.f * M_PI); // launch_dim.x is the number of rays launched per pixel, so we divide by it to get the average flux per ray. (exponent+2)/2pi normalizes reflected distribution to unity.
                    }
                }
            }

            // absorption
            // For cameras, use pixel index directly (no UUID lookup needed)
            // Use BufferIndexer: [pixel][band]
            size_t ind_camera = rad_indexer(pixel_index, b);

            atomicAdd(&radiation_in_camera[ind_camera],
                      (strength + strength_spec) / M_PI); // note: pi factor is to convert from flux to intensity assuming surface is Lambertian. We don't multiply by the solid angle by convention to avoid very small numbers.
        }
    }
}

RT_PROGRAM void closest_hit_pixel_label() {

    uint origin_UUID = prd.origin_UUID;

    uint hit_position = primitive_positions[UUID];

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[hit_position] == 5) { // periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x = xbounds.y - xbounds.x;
        float width_y = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.x) <= eps) { //-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x - xbounds.y) <= eps) { //+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.x) <= eps) { //-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y - ybounds.y) <= eps) { //+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }

    } else {

        // Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being received)
        // Note: We are reserving a value of 0 for the sky, so we will store UUID+1
        camera_pixel_label[origin_UUID] = UUID + 1;

        float depth = prd.strength + t_hit;
        float3 camera_direction3 = d_rotatePoint(make_float3(1, 0, 0), -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y);
        camera_pixel_depth[origin_UUID] = abs(dot(camera_direction3, ray.direction)) * depth;
    }
}

// ---------------------------------------------------------------------------
// Translucent cover (glass/plastic) any-hit programs
// ---------------------------------------------------------------------------
// A glass cover is not an occluder: the ray passes through it (rtIgnoreIntersection) attenuated by the
// Fresnel+Bouguer tau(theta), which accumulates per band in prd.cover_transmittance and is applied to
// the deposited flux in the miss programs. The cover's own reflected/absorbed energy is deposited here.
// Mirrors coverAnyHitBody() in optix8/OptiX8DeviceCode.cu. Direct rays use the per-source flux; diffuse
// rays use the (isotropic) sky flux for the cover's rho/alpha bookkeeping — the transmittance
// accumulation itself is exact and flux-independent.

RT_PROGRAM void any_hit_direct() {

    if (glass_enabled == 0u) {
        return; // glass model not in use this launch — accept the hit (normal occlusion)
    }

    uint hit_position = primitive_positions[UUID];
    if (hit_position == UINT_MAX) {
        return;
    }
    // The intersection program already skips self-hits (origin_UUID == UUID); guard defensively.
    if (UUID == prd.origin_UUID) {
        rtIgnoreIntersection();
        return;
    }

    MaterialPropertyIndexer mat_indexer(Nsources, Nprimitives, Nbands_global);
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    SourceFluxIndexer source_flux_indexer(Nsources, Nbands_launch);

    // Pass-through only if this primitive is glass in EVERY launched band. A single ray carries all
    // launched bands, so a primitive that is glass in some launched bands but opaque in others cannot be
    // both transmitted and blocked — treat it as a normal (opaque) occluder for the whole ray (accept
    // the hit). This matches the OptiX 8 and Vulkan backends.
    {
        int b_check = -1;
        for (int b_global = 0; b_global < Nbands_global; b_global++) {
            if (band_launch_flag[b_global] == 0) {
                continue;
            }
            b_check++;
            size_t mat_ind = mat_indexer(prd.source_ID, hit_position, b_global);
            if (is_glass[mat_ind] == 0.f) {
                return; // opaque in this launched band → block the ray
            }
        }
    }

    bool face_top;
    float cos_theta = coverNormalFaceCos(hit_position, ray.direction, face_top);

    int b = -1;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {
        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;
        if (b >= HELIOS_MAX_RADIATION_BANDS) {
            break; // host guarantees this never trips
        }

        size_t mat_ind = mat_indexer(prd.source_ID, hit_position, b_global);
        float3 tra = glass_tau_rho_alpha(cos_theta, glass_n[mat_ind], glass_KL[mat_ind]); // (tau, rho, alpha)

        size_t flux_idx = source_flux_indexer(prd.source_ID, b);
        double ext_flux = source_fluxes[flux_idx];
        double incoming = prd.strength * ext_flux * (double) prd.cover_transmittance[b];

        size_t cov_ind = rad_indexer(hit_position, b);
        if (incoming > 0.0) {
            atomicFloatAdd(&radiation_in[cov_ind], (float) (incoming * (double) tra.z)); // absorbed
            // Reflection leaves from the hit face; transmission continues out the far face.
            if (face_top) {
                atomicFloatAdd(&scatter_buff_top[cov_ind], (float) (incoming * (double) tra.y));
            } else {
                atomicFloatAdd(&scatter_buff_bottom[cov_ind], (float) (incoming * (double) tra.y));
            }
        }

        prd.cover_transmittance[b] *= tra.x; // attenuate the through-beam for this band
    }

    // All launched bands are glass: pass the (attenuated) ray through toward the source.
    rtIgnoreIntersection();
}

RT_PROGRAM void any_hit_diffuse() {

    if (glass_enabled == 0u) {
        return;
    }

    uint hit_position = primitive_positions[UUID];
    if (hit_position == UINT_MAX) {
        return;
    }
    if (UUID == prd.origin_UUID) {
        rtIgnoreIntersection();
        return;
    }

    MaterialPropertyIndexer mat_indexer(Nsources, Nprimitives, Nbands_global);
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);

    // Pass-through only if glass in EVERY launched band; otherwise treat as opaque (accept → block).
    // Matches the OptiX 8 and Vulkan backends.
    {
        int b_check = -1;
        for (int b_global = 0; b_global < Nbands_global; b_global++) {
            if (band_launch_flag[b_global] == 0) {
                continue;
            }
            b_check++;
            size_t mat_ind = mat_indexer(prd.source_ID, hit_position, b_global);
            if (is_glass[mat_ind] == 0.f) {
                return; // opaque in this launched band → block the ray
            }
        }
    }

    bool face_top;
    float cos_theta = coverNormalFaceCos(hit_position, ray.direction, face_top);

    int b = -1;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {
        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;
        if (b >= HELIOS_MAX_RADIATION_BANDS) {
            break;
        }

        size_t mat_ind = mat_indexer(prd.source_ID, hit_position, b_global);
        float3 tra = glass_tau_rho_alpha(cos_theta, glass_n[mat_ind], glass_KL[mat_ind]);

        double ext_flux = diffuse_flux[b];
        double incoming = prd.strength * ext_flux * (double) prd.cover_transmittance[b];

        size_t cov_ind = rad_indexer(hit_position, b);
        if (incoming > 0.0) {
            atomicFloatAdd(&radiation_in[cov_ind], (float) (incoming * (double) tra.z));
            if (face_top) {
                atomicFloatAdd(&scatter_buff_top[cov_ind], (float) (incoming * (double) tra.y));
            } else {
                atomicFloatAdd(&scatter_buff_bottom[cov_ind], (float) (incoming * (double) tra.y));
            }
        }

        prd.cover_transmittance[b] *= tra.x;
    }

    // All launched bands are glass: pass the (attenuated) ray through toward the sky.
    rtIgnoreIntersection();
}

RT_PROGRAM void miss_direct() {

    // Convert UUID to array position
    uint origin_position = primitive_positions[prd.origin_UUID];

    // Create indexers
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    MaterialPropertyIndexer mat_indexer(Nsources, Nprimitives, Nbands_global);
    SourceFluxIndexer source_flux_indexer(Nsources, Nbands_launch);
    CameraMaterialIndexer cam_mat_indexer(Nsources, Nprimitives, Nbands_global, Ncameras);
    SourceCameraFluxIndexer source_cam_flux_indexer(Nsources, Nbands_launch, Ncameras);
    SpecularRadiationIndexer spec_indexer(Nsources, Ncameras, Nprimitives, Nbands_launch);

    int b = -1;
    for (int b_global = 0; b_global < Nbands_global; b_global++) {

        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;

        // Use BufferIndexer: [primitive][band]
        size_t ind_origin = rad_indexer(origin_position, b);

        // Use BufferIndexer: [source][primitive][band]
        size_t radprop_ind_global = mat_indexer(prd.source_ID, origin_position, b_global);
        float t_rho = rho[radprop_ind_global];
        float t_tau = tau[radprop_ind_global];

        // Use BufferIndexer: [source][band]
        size_t flux_idx = source_flux_indexer(prd.source_ID, b);
        float source_flux = source_fluxes[flux_idx];
        // Attenuation from any translucent covers (glass/plastic) the ray passed through, per band.
        float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;
        double strength = prd.strength * source_flux * cover_tau;
        float absorption = strength * (1.f - t_rho - t_tau);

        // absorption
        atomicAdd(&radiation_in[ind_origin], absorption);

        if (t_rho > 0 || t_tau > 0) {
            if (prd.face) { // reflection from top, transmission from bottom
                atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_rho); // reflection
                atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_tau); // transmission
            } else { // reflection from bottom, transmission from top
                atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_rho); // reflection
                atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_tau); // transmission
            }
        }
        if (Ncameras > 0) {
            // Use BufferIndexer: [source][primitive][band][camera]
            size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, camera_ID);
            float t_rho_cam = rho_cam[indc];
            float t_tau_cam = tau_cam[indc];
            if ((t_rho_cam > 0 || t_tau_cam > 0) && strength > 0) {
                if (prd.face) { // reflection from top, transmission from bottom
                    atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_rho_cam); // reflection
                    atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_tau_cam); // transmission
                } else { // reflection from bottom, transmission from top
                    atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_rho_cam); // reflection
                    atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_tau_cam); // transmission
                }
            }
            // Accumulate incident radiation for specular for ALL cameras (per source, camera-weighted).
            // Direct rays are launched once (not per camera), so we must populate every camera's slot
            // in radiation_specular here so each camera's closest-hit can read its own data. (Mirrors
            // the OptiX 8 path in OptiX8DeviceCode.cu; without the camera loop only camera_ID==0 was
            // populated and additional cameras saw no specular highlight.)
            // Apply camera spectral response weighting: ∫(source × camera) / ∫(source)
            if (strength > 0) {
                for (unsigned int cam = 0; cam < Ncameras; cam++) {
                    // Use BufferIndexer: [source][band][camera]
                    size_t weight_ind = source_cam_flux_indexer(prd.source_ID, b, cam);
                    float camera_weight = source_fluxes_cam[weight_ind];
                    // Use BufferIndexer: [source][camera][primitive][band] (note different order!)
                    size_t ind_specular = spec_indexer(prd.source_ID, cam, origin_position, b);
                    atomicFloatAdd(&radiation_specular[ind_specular], strength * camera_weight);
                }
            }
        }
    }
}

// Unified device function to evaluate diffuse angular distribution
// Supports three modes with automatic priority-based selection:
//   Priority 1: Power-law (Harrison & Coombes) if K > 0
//   Priority 2: Prague sky model if params.w > 0 (valid normalization)
//   Priority 3: Isotropic (uniform) otherwise
__device__ float evaluateDiffuseAngularDistribution(const float3 &ray_dir, const float3 &peak_dir, float power_law_K, float power_law_norm, const float4 &prague_params) {

    // Priority 1: Power-law (if K > 0)
    if (power_law_K > 0.0f) {
        float psi = acos_safe(dot(peak_dir, ray_dir));
        psi = fmaxf(psi, M_PI / 180.0f); // Avoid singularity at 1 degree
        return powf(psi, -power_law_K) * power_law_norm;
    }

    // Priority 2: Prague (if params.w > 0, indicating valid normalization)
    if (prague_params.w > 0.0f) {
        // Angular distance from sun (degrees)
        float gamma = acos_safe(dot(ray_dir, peak_dir)) * 180.0f / float(M_PI);

        // Zenith angle
        float cos_theta = fmaxf(ray_dir.z, 0.0f);

        // Circumsolar + horizon brightening
        // params: (circ_strength, circ_width, horizon_brightness, normalization)
        float pattern = (1.0f + prague_params.x * expf(-gamma / prague_params.y)) * (1.0f + (prague_params.z - 1.0f) * (1.0f - cos_theta));

        // Multiply by π to account for cosine-weighted sampling PDF (cos×sin/π)
        // This ensures correct Monte Carlo integration for Prague angular distribution
        return pattern * prague_params.w * M_PI;
    }

    // Priority 3: Isotropic
    // For isotropic diffuse with cosine-weighted sampling (PDF = cos×sin/π):
    // The π from the PDF denominator must appear in the Monte Carlo weight
    return 1.0f;
}

RT_PROGRAM void miss_diffuse() {

    // Convert UUID to array position
    uint origin_position = primitive_positions[prd.origin_UUID];

    // Create indexers
    RadiationBufferIndexer rad_indexer(Nprimitives, Nbands_launch);
    MaterialPropertyIndexer mat_indexer(Nsources, Nprimitives, Nbands_global);
    CameraMaterialIndexer cam_mat_indexer(Nsources, Nprimitives, Nbands_global, Ncameras);

    int b = -1;
    for (size_t b_global = 0; b_global < Nbands_global; b_global++) {

        if (band_launch_flag[b_global] == 0) {
            continue;
        }
        b++;

        if (diffuse_flux[b] > 0.f) {

            // Use BufferIndexer: [primitive][band]
            size_t ind_origin = rad_indexer(origin_position, b);

            // Use BufferIndexer: [source][primitive][band]
            size_t radprop_ind_global = mat_indexer(prd.source_ID, origin_position, b_global);
            float t_rho = rho[radprop_ind_global];
            float t_tau = tau[radprop_ind_global];

            // Attenuation from any translucent covers (glass/plastic) the sky ray passed through, per band.
            float cover_tau = (b < HELIOS_MAX_RADIATION_BANDS) ? prd.cover_transmittance[b] : 1.f;

            // Check if ray was launched from voxel (type 4)
            if (primitive_type[origin_position] == 4) { // ray was launched from voxel

                float kappa = t_rho; // just a reminder that rho is actually the absorption coefficient
                float sigma_s = t_tau; // just a reminder that tau is actually the scattering coefficient
                float beta = kappa + sigma_s;

                // absorption
                atomicAdd(&radiation_in[ind_origin], diffuse_flux[b] * prd.strength * cover_tau * kappa / beta);

                // scattering
                atomicAdd(&scatter_buff_top[ind_origin], diffuse_flux[b] * prd.strength * cover_tau * sigma_s / beta);

            } else { // ray was NOT launched from voxel

                // Use unified distribution function (supports power-law, Prague, and isotropic modes)
                float fd = evaluateDiffuseAngularDistribution(ray.direction, diffuse_peak_dir[b], diffuse_extinction[b], diffuse_dist_norm[b], sky_radiance_params[b]);

                float strength = fd * diffuse_flux[b] * prd.strength * cover_tau;

                //  absorption
                atomicAdd(&radiation_in[ind_origin], strength * (1.f - t_rho - t_tau));

                if (t_rho > 0 || t_tau > 0) {
                    if (prd.face) { // reflection from top, transmission from bottom
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_tau); // transmission
                    } else { // reflection from bottom, transmission from top
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_rho); // reflection
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_tau); // transmission
                    }
                }
                if (Ncameras > 0) {
                    // Use BufferIndexer: [source][primitive][band][camera]
                    size_t indc = cam_mat_indexer(prd.source_ID, origin_position, b_global, camera_ID);
                    float t_rho_cam = rho_cam[indc];
                    float t_tau_cam = tau_cam[indc];
                    if ((t_rho_cam > 0 || t_tau_cam > 0) && prd.strength > 0) {
                        if (prd.face) { // reflection from top, transmission from bottom
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_rho_cam); // reflection
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_tau_cam); // transmission
                        } else { // reflection from bottom, transmission from top
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_rho_cam); // reflection
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_tau_cam); // transmission
                        }
                    }
                    // Note: Don't accumulate diffuse sky radiation to radiation_specular
                    // Specular should only reflect DIRECT source radiation (accumulated in miss_direct)
                }
            }
        }
    }
}

RT_PROGRAM void miss_camera() {

    // For cameras, origin_UUID is actually the pixel index (not a primitive UUID!)
    uint pixel_index = prd.origin_UUID;
    size_t Npixels = camera_resolution_full.x * camera_resolution_full.y;

    // Create indexer
    RadiationBufferIndexer rad_indexer(Npixels, Nbands_launch); // Use Npixels for camera radiation buffer

    for (size_t b = 0; b < Nbands_launch; b++) {

        float radiance = 0.0f;

        // Check all light sources
        for (uint s = 0; s < Nsources; s++) {
            float flux = source_fluxes[s * Nbands_launch + b];
            if (flux <= 0.0f)
                continue;

            uint source_type = source_types[s];

            if (source_type == 0 || source_type == 2) {
                // COLLIMATED or SUN_SPHERE
                float cos_sun_angle = dot(ray.direction, sun_direction);
                if (cos_sun_angle >= solar_disk_cos_angle && solar_disk_radiance[b] > 0.0f) {
                    radiance += solar_disk_radiance[b];
                }
            } else if (source_type == 1) {
                // SPHERE
                float radius = source_widths[s].x * 0.5f;
                if (d_raySphereIntersect(ray.origin, ray.direction, source_positions[s], radius)) {
                    float area = 4.0f * M_PI * radius * radius;
                    radiance += (flux / area) / M_PI;
                }
            } else if (source_type == 3) {
                // RECTANGLE
                float cos_angle;
                if (d_rayRectangleIntersect(ray.origin, ray.direction, source_positions[s], source_widths[s].x, source_widths[s].y, source_rotations[s], cos_angle)) {
                    float area = source_widths[s].x * source_widths[s].y;
                    radiance += (flux / area) * cos_angle / M_PI;
                }
            } else if (source_type == 4) {
                // DISK
                float cos_angle;
                float radius = source_widths[s].x;
                if (d_rayDiskIntersect(ray.origin, ray.direction, source_positions[s], radius, source_rotations[s], cos_angle)) {
                    float area = M_PI * radius * radius;
                    radiance += (flux / area) * cos_angle / M_PI;
                }
            }
        }

        // Fallback to sky radiance if no sources visible
        if (radiance <= 0.0f && camera_sky_radiance[b] > 0.f) {
            // Evaluate directional sky radiance using unified distribution function
            // camera_sky_radiance[b] contains the base zenith sky radiance (W/m²/sr) from Prague model
            // For camera, power-law is disabled (K=0, norm=1), so Prague params are used
            float angular_weight = evaluateDiffuseAngularDistribution(ray.direction, sun_direction,
                                                                      0.0f, // No power-law for camera
                                                                      1.0f,
                                                                      sky_radiance_params[b]); // Prague params

            radiance = camera_sky_radiance[b] * angular_weight;
        }

        // Isotropic sky emission/longwave: when band has emission enabled, the user-set diffuse_flux
        // represents hemispherical sky thermal/longwave irradiance. Convert to isotropic radiance.
        if (band_emission_flag[b] != 0u && camera_diffuse_flux[b] > 0.f) {
            radiance += camera_diffuse_flux[b] / M_PI;
        }

        if (radiance > 0.0f) {
            // Accumulate radiance directly (same as surface hits accumulate radiation_out)
            // Units: W/m²/sr
            // Monte Carlo averaging: prd.strength = 1/N_rays
            // Use BufferIndexer: [pixel][band]
            size_t ind_camera = rad_indexer(pixel_index, b);
            atomicAdd(&radiation_in_camera[ind_camera], radiance * prd.strength);
        }
    }
}

RT_PROGRAM void miss_pixel_label() {

    camera_pixel_depth[prd.origin_UUID] = -1;
}
