/** \file "rayGeneration.cu" File containing OptiX ray generation programs

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

#include <optix_world.h>
#include "RayTracing.cuh"

using namespace optix;

//! The direct ray generation program for a scene without a background radiation field (see d_dualField()) or with one (dual_scene), whose rays are of the type with that scene's
//! closest-hit and miss programs; inlined with a constant, each is its own code. Must match raygenDirect() in OptiX8DeviceCode.cu.
static __device__ __forceinline__ void raygenDirect(const bool dual_scene) {

    uint Nrays = launch_dim.x * launch_dim.y;
    uint ray_index = launch_dim.x * launch_index.y + launch_index.x;

    PerRayData prd;
    prd.seed = tea<16>(ray_index + Nrays * launch_index.z, random_seed);

    uint objID = launch_offset + launch_index.z;

    uint pID = primitiveID[objID];
    uint ptype = primitive_type[objID];
    int puvID = uvID[objID];

    float3 sp;

    float3 normal;

    // transformation matrix
    float m_trans[16];
    for (uint i = 0; i < 16; i++) {
        m_trans[i] = transform_matrix[optix::make_uint2(i, objID)];
    }

    // looping over sub-patches
    int NX = object_subdivisions[objID].x;
    int NY = object_subdivisions[objID].y;
    for (int jj = 0; jj < NY; jj++) {
        for (int ii = 0; ii < NX; ii++) {

            uint UUID = pID + jj * NX + ii;

            // two random samples [0,1]
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);

            if (ptype == 0 || ptype == 3) { // Patch or Tile

                uint Nx = launch_dim.x;
                uint Ny = launch_dim.y;
                float dx = 1.f / float(NX);
                float dy = 1.f / float(NY);

                // Map sample to rectangle [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + ii * dx + float(launch_index.x) * dx / float(Nx) + Rx * dx / float(Nx);
                sp.y = -0.5f + jj * dy + float(launch_index.y) * dy / float(Ny) + Ry * dy / float(Ny);
                sp.z = 0.f;

                int ID = maskID[objID];
                // FIX: Use objID (position) instead of UUID for primitive_solid_fraction access
                if (ID >= 0 && primitive_solid_fraction[objID] > 0.f && primitive_solid_fraction[objID] < 1.f) { // has texture transparency

                    d_sampleTexture_patch(sp, optix::make_int2(ii, jj), optix::make_float2(dx, dy), prd, ID, puvID);
                }

                // calculate rectangle normal vector (world coordinates)
                float3 v0 = make_float3(0, 0, 0);
                d_transformPoint(m_trans, v0);
                float3 v1 = make_float3(1, 0, 0);
                d_transformPoint(m_trans, v1);
                float3 v2 = make_float3(0, 1, 0);
                d_transformPoint(m_trans, v2);

                normal = normalize(cross(v1 - v0, v2 - v0));

            } else if (ptype == 1) { // Triangle

                // Map sample to triangle with vertices (0,0,0), (0,1,0), (1,1,0)
                if (Rx < Ry) {
                    sp.x = Rx;
                    sp.y = Ry;
                } else {
                    sp.x = Ry;
                    sp.y = Rx;
                }
                sp.z = 0;

                // calculate triangle normal vector (world coordinates)
                float3 v0 = make_float3(0, 0, 0);
                d_transformPoint(m_trans, v0);
                float3 v1 = make_float3(0, 1, 0);
                d_transformPoint(m_trans, v1);
                float3 v2 = make_float3(1, 1, 0);
                d_transformPoint(m_trans, v2);

                normal = normalize(cross(v1 - v0, v2 - v0));

                int ID = maskID[objID];
                // FIX: Use objID (position) instead of UUID for primitive_solid_fraction access
                if (ID >= 0 && primitive_solid_fraction[objID] > 0.f && primitive_solid_fraction[objID] < 1.f) { // has texture transparency

                    d_sampleTexture_triangle(sp, prd, ID, puvID);
                }

            } else if (ptype == 2) { // Disk

                d_sampleDisk(prd.seed, sp);

                // calculate disk normal vector (world coordinates)
                float3 v0 = make_float3(0, 0, 0);
                d_transformPoint(m_trans, v0);
                float3 v1 = make_float3(1, 0, 0);
                d_transformPoint(m_trans, v1);
                float3 v2 = make_float3(0, 1, 0);
                d_transformPoint(m_trans, v2);

                normal = normalize(cross(v1 - v0, v2 - v0));
            }

            // The class of a spectral texture at the launch point, whose reflectivity and transmissivity the rays from it use
            prd.palette_entry = d_spectralPaletteEntry(objID, d_originTextureUV(ptype, sp, puvID));

            // The shading normal of a normal map at the launch point, which only a surface layer's angle-dependent split of the light uses here
            const uint origin_map = d_normalMapOf(objID);
            const bool origin_has_normal = origin_map != helios_nm::NO_MAP && d_hasSurfaceLayer(objID);
            float3 origin_shading_normal = normal;
            if (origin_has_normal) {
                origin_shading_normal = d_normalMapShadingNormalAt(ptype, origin_map, m_trans, puvID, normal, sp, false);
            }

            // translate the ray to the location of the primitive

            float3 ray_origin = sp;
            d_transformPoint(m_trans, ray_origin);

            // Send a ray toward each source
            for (int rr = 0; rr < Nsources; rr++) {

                // set the ray direction
                float3 ray_direction;
                float ray_magnitude;
                double source_factor;
                if (!d_sampleSourceDirection(rr, ray_origin, prd.seed, ray_direction, ray_magnitude, source_factor)) {
                    continue; // the back of a rectangle or disk source, or an unknown source type
                }
                prd.strength = 1. / double(launch_dim.x * launch_dim.y) * fabs(dot(normal, ray_direction)) * source_factor;

                // No minimum distance: the primitive the ray was launched from is skipped (isRayOrigin()), and a minimum distance
                // let a ray launched that close to an adjacent surface pass through it where the two meet
                optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, dual_scene ? direct_dual_ray_type : direct_ray_type, 0.f, ray_magnitude);

                prd.origin_UUID = UUID;
                prd.source_ID = rr;
                prd.hit_periodic_boundary = false;
                prd.tile_x = 0; // launched from the central copy of the domain
                prd.tile_y = 0;
                prd.origin_tile_x = 0;
                prd.origin_tile_y = 0;
                initCoverTransmittance(prd); // translucent-cover attenuation starts at 1 (no covers crossed)
                if (dual_scene) {
                    d_startDualRay(prd, primitive_positions[UUID]);
                }

                if (dot(ray_direction, normal) > 0) {
                    prd.face = 1;
                } else {
                    prd.face = 0;
                }
                // A surface layer sees the light at the angle of the shading normal at the launch point (|m.L|, so that a facet tipped away from the source counts as its mirror image,
                // consistently with the folding of diffuse rays); the beam power the primitive intercepts stays at the geometric cosine
                prd.origin_cos = origin_has_normal ? fabs(dot(ray_direction, origin_shading_normal)) : fabs(dot(ray_direction, normal));

                if ((prd.face == 1 || twosided_flag[objID] == 1) && twosided_flag[objID] != 3) {

                    // A ray that reaches the wrap limit deposits nothing (treated as blocked). After a wrap, a ray toward a finite
                    // source only has the rest of the distance to the point it sampled on the source left to travel.
                    bool wrap_limit_reached = true;
                    for (int wrap = 0; wrap < MAX_PERIODIC_WRAPS; ++wrap) {
                        if (dual_scene) {
                            prd.recorded_t = NO_RECORDED_HIT; // no excluded primitive passed in this segment yet
                        }
                        rtTrace(top_object, ray, prd);

                        if (!prd.hit_periodic_boundary) {
                            wrap_limit_reached = false;
                            break; // real hit or miss → done
                        }

                        ray.origin = prd.periodic_hit;
                        ray.tmin = 0.f; // a wrapped ray hits geometry right at the boundary it was wrapped to
                        if (source_types[rr] != 0) {
                            ray.tmax = fmaxf(ray.tmax - prd.periodic_hit_t, 2e-4f);
                        }
                        prd.hit_periodic_boundary = false;
                    }
                    if (wrap_limit_reached) {
                        atomicAdd(&abandoned_ray_count[0], 1u);
                    }
                }
            }
        }
    }
}

RT_PROGRAM void direct_raygen() {
    raygenDirect(false);
}

//! direct_raygen for a scene with a background radiation field
RT_PROGRAM void direct_raygen_dual() {
    raygenDirect(true);
}

//! The diffuse ray generation program for a scene without a background radiation field (see d_dualField()) or with one (dual_scene), whose rays are of the type with that scene's
//! closest-hit and miss programs; inlined with a constant, each is its own code. Must match raygenDiffuse() in OptiX8DeviceCode.cu.
static __device__ __forceinline__ void raygenDiffuse(const bool dual_scene) {

    uint dimx = launch_dim.x * launch_dim.y;
    uint indx = launch_dim.x * launch_index.y + launch_index.x;

    PerRayData prd;
    prd.seed = tea<16>(indx + dimx * launch_index.z, random_seed);

    uint objID = launch_offset + launch_index.z;

    if (launch_face == 0 && twosided_flag[objID] == 0) { // skip the launch if from the bottom face and twosided_flag = 0
        return;
    }

    uint pID = primitiveID[objID];
    uint ptype = primitive_type[objID];
    int puvID = uvID[objID];

    // transformation matrix
    float m_trans[16];
    for (uint i = 0; i < 16; i++) {
        m_trans[i] = transform_matrix[optix::make_uint2(i, objID)];
    }

    float3 sp, normal;

    // looping over sub-patches
    int NX = object_subdivisions[objID].x;
    int NY = object_subdivisions[objID].y;
    for (int jj = 0; jj < NY; jj++) {
        for (int ii = 0; ii < NX; ii++) {

            uint UUID = pID + jj * NX + ii;

            // two random samples [0,1]
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);

            if (ptype == 0 || ptype == 3) { // Patch or Tile

                // calculate rectangle normal vector (world coordinates)
                float3 s0 = make_float3(0, 0, 0);
                float3 s1 = make_float3(1, 0, 0);
                float3 s2 = make_float3(0, 1, 0);
                d_transformPoint(m_trans, s0);
                d_transformPoint(m_trans, s1);
                d_transformPoint(m_trans, s2);

                normal = normalize(cross(s1 - s0, s2 - s0));

                float dx = 1.f / float(NX);
                float dy = 1.f / float(NY);

                // Map sample to rectangle [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + (ii + Rx) * dx;
                sp.y = -0.5f + (jj + Ry) * dy;
                sp.z = 0.f;

                int ID = maskID[objID];
                if (ID >= 0) { // has texture transparency

                    d_sampleTexture_patch(sp, optix::make_int2(ii, jj), optix::make_float2(dx, dy), prd, ID, puvID);
                }

            } else if (ptype == 1) { // Triangle

                // Map sample to triangle with vertices (0,0,0), (0,1,0), (1,1,0)
                if (Rx < Ry) {
                    sp.x = Rx;
                    sp.y = Ry;
                } else {
                    sp.x = Ry;
                    sp.y = Rx;
                }
                sp.z = 0;

                // calculate triangle normal vector (world coordinates)
                float3 v0 = make_float3(0, 0, 0);
                d_transformPoint(m_trans, v0);
                float3 v1 = make_float3(0, 1, 0);
                d_transformPoint(m_trans, v1);
                float3 v2 = make_float3(1, 1, 0);
                d_transformPoint(m_trans, v2);

                normal = normalize(cross(v1 - v0, v2 - v0));

                int ID = maskID[objID];
                if (ID >= 0) { // has texture transparency

                    d_sampleTexture_triangle(sp, prd, ID, puvID);
                }

            } else if (ptype == 2) { // Disk

                // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

                // first map sample point to rectangle [-1,1] [-1,1]
                sp.x = -1.f + 2.f * Rx;
                sp.y = -1.f + 2.f * Ry;

                float r, p;
                if (sp.x > -sp.y) {
                    if (sp.x > sp.y) {
                        r = sp.x;
                        p = sp.y / sp.x;
                    } else {
                        r = sp.y;
                        p = 2.f - sp.x / sp.y;
                    }
                } else {
                    if (sp.x < sp.y) {
                        r = -sp.x;
                        p = 4.f + sp.y / sp.x;
                    } else {
                        r = -sp.y;
                        if (sp.y != 0.f) { // avoid division by zero at origin
                            p = 6.f - sp.x / sp.y;
                        } else {
                            p = 0.f;
                        }
                    }
                }
                p *= 0.25f * M_PI;

                // find x,y point on unit disk
                sp.x = r * cosf(p);
                sp.y = r * sinf(p);
                sp.z = 0.f;

                // calculate disk normal vector (world coordinates)
                float3 v0 = make_float3(0, 0, 0);
                d_transformPoint(m_trans, v0);
                float3 v1 = make_float3(1, 0, 0);
                d_transformPoint(m_trans, v1);
                float3 v2 = make_float3(0, 1, 0);
                d_transformPoint(m_trans, v2);
                normal = normalize(cross(v1 - v0, v2 - v0));
            }

            // Choose random hemispherical direction - map samples to hemisphere (from Suffern (2007) "Ray tracing fom the ground up" Chap. 6)

            float Rt;
            float Rp;

            Rt = (launch_index.x + rnd(prd.seed)) / float(launch_dim.x);
            Rp = (launch_index.y + rnd(prd.seed)) / float(launch_dim.y);

            float t = asin_safe(sqrtf(Rt));
            float p = 2.f * M_PI * Rp;

            float3 ray_direction;
            ray_direction.x = sin(t) * cos(p);
            ray_direction.y = sin(t) * sin(p);
            ray_direction.z = cos(t);

            float3 ray_origin;
            optix::Ray ray;

            // A normal map leaves the cosine lobe on the facet normal, so that the primitive gathers what its footprint intercepts (as a relief that is a heightfield does)
            ray_direction = d_rotatePoint(ray_direction, acos_safe(normal.z), atan2(normal.y, normal.x));

            prd.strength = 1.f / float(dimx);
            // A surface layer sees each ray at its angle to the shading normal at the launch point
            prd.origin_cos = fabs(dot(ray_direction, normal));
            const uint origin_map = d_normalMapOf(objID);
            if (origin_map != helios_nm::NO_MAP && d_hasSurfaceLayer(objID)) {
                prd.origin_cos = fabs(dot(ray_direction, d_normalMapShadingNormalAt(ptype, origin_map, m_trans, puvID, normal, sp, false)));
            }

            prd.origin_UUID = UUID;
            prd.source_ID = Nsources; // diffuse/scatter material slot, weighted by the band's combined incident spectrum
            // The class of a spectral texture at the launch point, whose reflectivity and transmissivity the rays from it use
            prd.palette_entry = d_spectralPaletteEntry(objID, d_originTextureUV(ptype, sp, puvID));
            prd.hit_periodic_boundary = false;
            initCoverTransmittance(prd); // translucent-cover attenuation starts at 1 (no covers crossed)
            if (dual_scene) {
                d_startDualRay(prd, primitive_positions[UUID]);
            }
            const unsigned int ray_type = dual_scene ? diffuse_dual_ray_type : diffuse_ray_type;

            // ---- "top" surface launch -------
            ray_origin = sp;
            d_transformPoint(m_trans, ray_origin);

            if (launch_face == 1 && twosided_flag[objID] != 3) {

                // No minimum distance: the primitive the ray was launched from is skipped (isRayOrigin()), and a minimum distance
                // let a ray launched that close to an adjacent surface pass through it where the two meet
                ray = optix::make_Ray(ray_origin, ray_direction, ray_type, 0.f, RT_DEFAULT_MAX);

                prd.face = 1;

                // Trace segment by segment: re-trace after wrapping through a periodic boundary and after passing a
                // translucent cover.
                prd.hit_cover = false;
                prd.tile_x = 0; // launched from the central copy of the domain
                prd.tile_y = 0;
                prd.origin_tile_x = 0;
                prd.origin_tile_y = 0;
                int wraps = 0;
                for (int segment = 0; segment < MAX_RAY_SEGMENTS; ++segment) {
                    if (dual_scene) {
                        prd.recorded_t = NO_RECORDED_HIT; // no excluded primitive passed in this segment yet
                    }
                    rtTrace(top_object, ray, prd);

                    if (prd.hit_periodic_boundary) {
                        if (++wraps >= MAX_PERIODIC_WRAPS) {
                            atomicAdd(&abandoned_ray_count[0], 1u);
                            break;
                        }
                        ray.origin = prd.periodic_hit;
                        ray.tmin = 0.f; // a wrapped ray hits geometry right at the boundary it was wrapped to
                        prd.hit_periodic_boundary = false;
                    } else if (prd.hit_cover) {
                        ray.origin = prd.cover_exit;
                        ray.tmin = 1e-5f;
                        prd.hit_cover = false;
                    } else {
                        break; // real hit or miss → done
                    }
                }

                // ---- "bottom" surface launch -------
            } else if (launch_face == 0 && twosided_flag[objID] == 1) {

                ray_direction = -ray_direction;
                ray = optix::make_Ray(ray_origin, ray_direction, ray_type, 0.f, RT_DEFAULT_MAX);

                prd.face = 0;

                // Trace segment by segment: re-trace after wrapping through a periodic boundary and after passing a
                // translucent cover.
                prd.hit_cover = false;
                prd.tile_x = 0; // launched from the central copy of the domain
                prd.tile_y = 0;
                prd.origin_tile_x = 0;
                prd.origin_tile_y = 0;
                int wraps = 0;
                for (int segment = 0; segment < MAX_RAY_SEGMENTS; ++segment) {
                    if (dual_scene) {
                        prd.recorded_t = NO_RECORDED_HIT; // no excluded primitive passed in this segment yet
                    }
                    rtTrace(top_object, ray, prd);

                    if (prd.hit_periodic_boundary) {
                        if (++wraps >= MAX_PERIODIC_WRAPS) {
                            atomicAdd(&abandoned_ray_count[0], 1u);
                            break;
                        }
                        ray.origin = prd.periodic_hit;
                        ray.tmin = 0.f; // a wrapped ray hits geometry right at the boundary it was wrapped to
                        prd.hit_periodic_boundary = false;
                    } else if (prd.hit_cover) {
                        ray.origin = prd.cover_exit;
                        ray.tmin = 1e-5f;
                        prd.hit_cover = false;
                    } else {
                        break; // real hit or miss → done
                    }
                }
            }
            // else: Skip ray trace for bottom face of one-sided primitives
        }
    }
}

RT_PROGRAM void diffuse_raygen() {
    raygenDiffuse(false);
}

//! diffuse_raygen for a scene with a background radiation field
RT_PROGRAM void diffuse_raygen_dual() {
    raygenDiffuse(true);
}

// Move a camera ray that starts above or below the periodic boundary faces to the top or bottom of the faces. The faces span only
// the height of the geometry, so the ray passes nothing it can hit on the way there; it is then moved into the domain laterally
// by a whole number of periods, which is where it would have been wrapped to on the way. A ray that never reaches the faces'
// height range is left where it is. Returns the distance the ray was moved along its direction, and adds the whole periods it was
// moved by laterally to the periodic copy of the domain the ray is in. Must match advance_to_periodic_faces() in the Vulkan shaders.
static __device__ __inline__ float advanceToPeriodicFaces(float3 &ray_origin, const float3 &ray_direction, PerRayData &ray_data) {
    if (periodic_flag.x != 1 && periodic_flag.y != 1) {
        return 0.f;
    }

    // bbox_vertices is indexed (vertex, face). Vertices 0 and 2 of every face are at the bottom and the top of the faces; vertex 0
    // of face 0 and vertex 1 of face 1 are at the low and high corners of the domain in x and y.
    const float face_zmin = bbox_vertices[make_uint2(0, 0)].z;
    const float face_zmax = bbox_vertices[make_uint2(2, 0)].z;

    float face_z;
    if (ray_origin.z > face_zmax && ray_direction.z < 0.f) {
        face_z = face_zmax;
    } else if (ray_origin.z < face_zmin && ray_direction.z > 0.f) {
        face_z = face_zmin;
    } else {
        return 0.f;
    }

    const float t = (face_z - ray_origin.z) / ray_direction.z;
    float3 moved = make_float3(ray_origin.x + t * ray_direction.x, ray_origin.y + t * ray_direction.y, face_z);
    if (periodic_flag.x == 1) {
        const float2 xbounds = periodicXBounds();
        const float unwrapped = moved.x;
        moved.x = wrapIntoPeriod(moved.x, ray_direction.x, xbounds.x, xbounds.y);
        ray_data.tile_x += __float2int_rn((unwrapped - moved.x) / (xbounds.y - xbounds.x));
    }
    if (periodic_flag.y == 1) {
        const float2 ybounds = periodicYBounds();
        const float unwrapped = moved.y;
        moved.y = wrapIntoPeriod(moved.y, ray_direction.y, ybounds.x, ybounds.y);
        ray_data.tile_y += __float2int_rn((unwrapped - moved.y) / (ybounds.y - ybounds.x));
    }
    ray_origin = moved;
    return t;
}

RT_PROGRAM void camera_raygen() {

    uint dimx = launch_dim.x * launch_dim.y; // x number of ray, y width, z length
    uint indx = launch_dim.x * launch_index.y + launch_index.x;

    // Use full camera resolution (not tile size) for correct pixel calculations
    optix::int2 camera_resolution = camera_resolution_full;

    PerRayData prd;
    prd.seed = tea<16>(indx + dimx * launch_index.z, random_seed);

    float3 sp;

    // Calculate global pixel coordinates including tile offsets
    uint ii = camera_pixel_offset_x + launch_index.y; // global x-pixel
    uint jj = camera_pixel_offset_y + launch_index.z; // global y-pixel
    size_t origin_ID = jj * camera_resolution_full.x + ii; // global pixel index


    // distortion
    //    float PPointsRatiox =1.052f;
    //    float PPointsRatioy =0.999f;
    //    float sensorxscale = 1.0054;
    //    float focalxy = 710;
    //    double x =(float(ii)-camera_resolution.x/2 * PPointsRatiox)/focalxy*sensorxscale;// / focalxy;   cam_res.y = 712
    //    double y = (float(jj)-camera_resolution.y/2 * PPointsRatioy )/focalxy; /// focalxy;   cam_res.x = 1072
    //    double r2 = x*x + y*y;
    //    double distCoeffs[4] = {-0.3535674,0.17298, 0, 0};
    //    double ii_d  = x * (1+ distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[2] * x * y + distCoeffs[3] * (r2 + 2 * x * x);
    //    double jj_d = y * (1+ distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[3] * x * y + distCoeffs[2] * (r2 + 2 * y * y);
    //    ii_d = ii_d*focalxy+float(camera_resolution.x)/2 * PPointsRatiox;
    //    jj_d = jj_d*focalxy+float(camera_resolution.y)/2 * PPointsRatioy;

    // *** sample a point on the pixel (view direction coordinate aligned) *** //

    float PPointsRatiox = 1.f;
    float PPointsRatioy = 1.f;
    float Rx = rnd(prd.seed);
    float Ry = rnd(prd.seed);

    // Map sample to pixel
    // Calculate VFOV scaling factor directly from aspect ratio
    // Since FOV_aspect_ratio = HFOV/VFOV, then tan(half_VFOV) = tan(half_HFOV)/FOV_aspect_ratio
    // The multiplier = tan(half_VFOV)/tan(half_HFOV) = 1/FOV_aspect_ratio
    float multiplier = 1.0f / FOV_aspect_ratio;
    sp.y = (-0.5f * PPointsRatioy + (ii + Rx) / float(camera_resolution.x));
    sp.z = (0.5f * PPointsRatiox - (jj + Ry) / float(camera_resolution.y)) * multiplier;
    sp.x = camera_viewplane_length;

    // *** Determine point 'p' on focal plane that passes through the lens center (0,0) and pixel sample (view direction coordinate aligned) *** //
    // Note: camera_focal_length is the focal plane distance (working distance), not the lens optical focal length

    float3 p = make_float3(camera_focal_length, sp.y / camera_viewplane_length * camera_focal_length, sp.z / camera_viewplane_length * camera_focal_length);

    // *** Sample point on lens (view direction coordinate aligned) *** //

    float3 ray_origin = make_float3(0, 0, 0);
    if (camera_lens_diameter > 0) {
        float3 disk_sample;
        d_sampleDisk(prd.seed, disk_sample);
        ray_origin = make_float3(0.f, 0.5f * disk_sample.x * camera_lens_diameter, 0.5f * disk_sample.y * camera_lens_diameter);
    }

    //*** ray direction is line from lens sample to p ***//

    float3 ray_direction = p - ray_origin;

    //*** rotate ray origin and direction into the direction of the camera view *** //

    ray_origin = d_rotatePoint(ray_origin, -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y) + camera_position;

    ray_direction = d_rotatePoint(ray_direction, -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y);
    ray_direction /= d_magnitude(ray_direction);

    optix::Ray ray;

    prd.strength = 1.f / float(launch_dim.x);

    prd.origin_UUID = origin_ID;
    prd.palette_entry = 0xFFFFFFFFu; // camera rays leave no primitive
    prd.face = 1;
    prd.source_ID = 0;
    prd.hit_periodic_boundary = false;
    prd.hit_cover = false;
    initCoverTransmittance(prd); // camera throughput: no translucent covers crossed yet

    // A ray starting above or below a periodic domain is moved to it in one step
    prd.tile_x = camera_tile.x; // the periodic copy of the domain the camera is in
    prd.tile_y = camera_tile.y;
    prd.origin_tile_x = prd.tile_x; // not used: the ray skips no primitive
    prd.origin_tile_y = prd.tile_y;
    advanceToPeriodicFaces(ray_origin, ray_direction, prd);

    ray = optix::make_Ray(ray_origin, ray_direction, camera_ray_type, 1e-5, RT_DEFAULT_MAX);

    // Trace segment by segment: the ray is re-launched after wrapping through a periodic boundary and after
    // passing a translucent cover, whose closest-hit has already added the cover's own radiance.
    int wraps = 0;
    for (int segment = 0; segment < MAX_RAY_SEGMENTS; ++segment) {
        prd.hit_periodic_boundary = false;
        prd.hit_cover = false;
        rtTrace(top_object, ray, prd);

        if (prd.hit_periodic_boundary) {
            if (++wraps >= MAX_PERIODIC_WRAPS) {
                break;
            }
            ray.origin = prd.periodic_hit;
            ray.tmin = 0.f; // a wrapped ray hits geometry right at the boundary it was wrapped to
        } else if (prd.hit_cover) {
            ray.origin = prd.cover_exit;
            ray.tmin = 1e-5f;
        } else {
            break; // real hit or miss → done
        }
    }
}

RT_PROGRAM void pixel_label_raygen() {

    uint indx = launch_dim.y * launch_index.z + launch_index.y;

    // Use full camera resolution (not tile size) for correct pixel calculations
    optix::int2 camera_resolution = camera_resolution_full;

    PerRayData prd;
    prd.seed = tea<16>(indx, random_seed);

    float3 sp;

    // Calculate global pixel coordinates including tile offsets
    uint ii = camera_pixel_offset_x + launch_index.y; // global x-pixel

    uint jj = camera_pixel_offset_y + launch_index.z; // global y-pixel

    size_t origin_ID = jj * camera_resolution_full.x + ii; // global pixel index

    // Map sample to center of pixel
    // Calculate VFOV scaling factor directly from aspect ratio
    // Since FOV_aspect_ratio = HFOV/VFOV, then tan(half_VFOV) = tan(half_HFOV)/FOV_aspect_ratio
    // The multiplier = tan(half_VFOV)/tan(half_HFOV) = 1/FOV_aspect_ratio
    float multiplier = 1.0f / FOV_aspect_ratio;
    sp.y = (-0.5f + (ii + 0.5f) / float(camera_resolution.x));
    sp.z = (0.5f - (jj + 0.5f) / float(camera_resolution.y)) * multiplier;
    sp.x = camera_viewplane_length;


    // *** Determine point 'p' on focal plane that passes through the lens center (0,0) and pixel sample (view direction coordinate aligned) *** //
    // Note: camera_focal_length is the focal plane distance (working distance), not the lens optical focal length

    float3 p = make_float3(camera_focal_length, sp.y / camera_viewplane_length * camera_focal_length, sp.z / camera_viewplane_length * camera_focal_length);

    // *** Ray is launched from center of lens *** //

    float3 ray_origin = make_float3(0, 0, 0);

    //*** ray direction is line from ray origin to p ***//

    float3 ray_direction = p;

    //*** rotate ray origin and direction into the direction of the camera view *** //

    ray_origin = d_rotatePoint(ray_origin, -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y) + camera_position;

    ray_direction = d_rotatePoint(ray_direction, -0.5 * M_PI + camera_direction.x, 0.5f * M_PI - camera_direction.y);
    ray_direction /= d_magnitude(ray_direction);

    optix::Ray ray;

    // A ray starting above or below a periodic domain is moved to it in one step; the distance counts toward the depth
    prd.tile_x = camera_tile.x; // the periodic copy of the domain the camera is in
    prd.tile_y = camera_tile.y;
    prd.origin_tile_x = prd.tile_x; // not used: the ray skips no primitive
    prd.origin_tile_y = prd.tile_y;
    prd.strength = advanceToPeriodicFaces(ray_origin, ray_direction, prd); // distance travelled before any translucent covers passed

    prd.origin_UUID = origin_ID;
    prd.palette_entry = 0xFFFFFFFFu; // camera rays leave no primitive
    prd.face = 1;
    prd.source_ID = 0;
    prd.hit_periodic_boundary = false;
    prd.hit_cover = false;
    initCoverTransmittance(prd);

    ray = optix::make_Ray(ray_origin, ray_direction, pixel_label_ray_type, 1e-5, RT_DEFAULT_MAX);

    // Trace segment by segment, as camera_raygen does, so the label is that of the first surface behind any
    // translucent covers.
    int wraps = 0;
    bool ray_ended = false;
    for (int segment = 0; segment < MAX_RAY_SEGMENTS; ++segment) {
        prd.hit_periodic_boundary = false;
        prd.hit_cover = false;
        rtTrace(top_object, ray, prd);

        if (prd.hit_periodic_boundary) {
            if (++wraps >= MAX_PERIODIC_WRAPS) {
                break;
            }
            ray.origin = prd.periodic_hit;
            ray.tmin = 0.f; // a wrapped ray hits geometry right at the boundary it was wrapped to
        } else if (prd.hit_cover) {
            ray.origin = prd.cover_exit;
            ray.tmin = 1e-5f;
        } else {
            ray_ended = true;
            break; // real hit or miss → done
        }
    }

    // A ray that runs out of wraps or segments is treated as blocked: the pixel shows no primitive, as for a ray that leaves the scene
    if (!ray_ended) {
        camera_pixel_label[origin_ID] = 0;
        camera_pixel_depth[origin_ID] = -1;
    }
}
