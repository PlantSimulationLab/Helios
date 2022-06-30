/** \file "rayGeneration.cu" File containing OptiX ray generation programs
    \author Brian Bailey

    Copyright (C) 2018  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <optix_world.h>

#include "RayTracing.cu.h"

using namespace optix;

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, direct_ray_type, , );
rtDeclareVariable(unsigned int, diffuse_ray_type, , );
rtDeclareVariable(unsigned int, direct_ray_type_MCRT, , );
rtDeclareVariable(unsigned int, diffuse_ray_type_MCRT, , );
rtDeclareVariable(unsigned int, emission_ray_type_MCRT, , );
rtDeclareVariable(unsigned int, random_seed, , );
rtDeclareVariable(unsigned int, launch_offset, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );

// texture-related buffers
rtBuffer<bool, 3> maskdata;
rtBuffer<int2, 1> masksize;
rtBuffer<int, 1> maskID;
rtBuffer<float2, 2> uvdata;
rtBuffer<int, 1> uvID;

// Buffer data for each primitive
rtBuffer<float, 2> transform_matrix;
rtBuffer<unsigned int, 1> primitive_type;
rtBuffer<float, 1> primitive_area;
rtBuffer<char, 1> twosided_flag;
rtBuffer<float, 1> radiation_out_top;
rtBuffer<float, 1> radiation_out_bottom;

// Object ID for a primitive
rtBuffer<uint, 1> objectID;

// ID for first primitive in an object
rtBuffer<uint, 1> primitiveID;

// Radiation sources buffers
rtDeclareVariable(unsigned int, Nsources, , );
rtBuffer<float, 1> source_fluxes;
rtBuffer<float3, 1> source_positions;
rtBuffer<float, 1> source_widths;
rtBuffer<unsigned int, 1> source_types;

rtDeclareVariable(float, diffuseFlux, , );

rtDeclareVariable(float, bound_sphere_radius, , );
rtDeclareVariable(float3, bound_sphere_center, , );

rtBuffer<int2, 1> object_subdivisions;

RT_PROGRAM void direct_raygen() {
    uint dimx = launch_dim.x * launch_dim.y;
    uint indx = launch_dim.x * launch_index.y + launch_index.x;

    PerRayData prd;
    prd.seed = tea<16>(indx + dimx * launch_index.z, random_seed);

    uint objID = launch_offset + launch_index.z;

    float3 sp;
    float r, p;

    float3 normal;

    // transformation matrix
    float m[16];
    for (uint i = 0; i < 16; i++) {
        m[i] = transform_matrix[optix::make_uint2(i, objID)];
    }

    // looping over sub-patches
    for (size_t jj = 0; jj < object_subdivisions[objID].y; jj++) {
        for (size_t ii = 0; ii < object_subdivisions[objID].x; ii++) {
            uint UUID = primitiveID[objID] + jj * object_subdivisions[objID].x + ii;

            // two random samples [0,1]
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);

            if (primitive_type[objID] == 0 || primitive_type[objID] == 3) {  // Patch or Tile

                uint Nx = launch_dim.x;
                uint Ny = launch_dim.y;
                float dx = 1.f / float(object_subdivisions[objID].x);
                float dy = 1.f / float(object_subdivisions[objID].y);

                // Map sample to rectangle [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + ii * dx + float(launch_index.x) * dx / float(Nx) + Rx * dx / float(Nx);
                sp.y = -0.5f + jj * dy + float(launch_index.y) * dy / float(Ny) + Ry * dy / float(Ny);
                sp.z = 0.f;

                int ID = maskID[objID];
                if (ID >= 0 && primitive_area[UUID] > 0) {  // has texture transparency

                    int2 sz = masksize[ID];
                    uint3 ind;

                    bool solid = false;
                    int count = 0;
                    while (!solid) {
                        count++;

                        float2 uv = make_float2(sp.x + 0.5f, 1.f - sp.y - 0.5f);
                        if (uvID[objID] == -1) {  // does not have custom (u,v) coordinates
                            ind = make_uint3(roundf(float(sz.x - 1) * uv.x), roundf(float(sz.y - 1) * uv.y), ID);
                        } else {  // has custom (u,v) coordinates
                            float2 uvmin = uvdata[make_uint2(3, uvID[objID])];
                            float2 duv;
                            duv.x = uvdata[make_uint2(1, uvID[objID])].x - uvdata[make_uint2(0, uvID[objID])].x;
                            duv.y = uvdata[make_uint2(1, uvID[objID])].y - uvdata[make_uint2(2, uvID[objID])].y;
                            ind = make_uint3(roundf(float(sz.x - 1) * (uvmin.x + uv.x * duv.x)),
                                             roundf(float(sz.y - 1) * (uvmin.y + uv.y * duv.y)), ID);
                        }
                        solid = maskdata[ind];
                        if (!solid) {
                            if (count > 10) {
                                break;
                            }
                            sp.x = -0.5f + ii * dx + rnd(prd.seed) * dx;
                            sp.y = -0.5f + jj * dy + rnd(prd.seed) * dy;
                        }
                    }
                }

                // calculate rectangle normal vector (world coordinates)
                float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
                float3 v1 = d_transformPoint(m, make_float3(1, 0, 0));
                float3 v2 = d_transformPoint(m, make_float3(0, 1, 0));

                normal = normalize(cross(v1 - v0, v2 - v0));

            } else if (primitive_type[objID] == 1) {  // Triangle

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
                float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
                float3 v1 = d_transformPoint(m, make_float3(0, 1, 0));
                float3 v2 = d_transformPoint(m, make_float3(1, 1, 0));

                normal = normalize(cross(v1 - v0, v2 - v1));

                int ID = maskID[objID];
                if (ID >= 0 && primitive_area[UUID] > 0) {  // has texture transparency

                    int2 sz = masksize[ID];

                    float a = v0.x - v1.x, b = v0.x - v2.x, d = v0.x;
                    float e = v0.y - v1.y, f = v0.y - v2.y, h = v0.y;
                    float i = v0.z - v1.z, j = v0.z - v2.z, l = v0.z;

                    bool solid = false;
                    int count = 0;
                    while (!solid) {
                        count++;

                        float3 R = d_transformPoint(m, sp);

                        float c = R.x, g = R.y, k = R.z;

                        float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
                        float q = g * i - e * k, s = e * j - f * i;

                        float inv_denom = 1.f / (a * m + b * q + c * s);

                        float e1 = d * m - b * n - c * p;
                        float beta = e1 * inv_denom;

                        float r = r = e * l - h * i;
                        float e2 = a * n + d * q + c * r;
                        float gamma = e2 * inv_denom;

                        float2 uv0 = uvdata[make_uint2(0, uvID[objID])];
                        float2 uv1 = uvdata[make_uint2(1, uvID[objID])];
                        float2 uv2 = uvdata[make_uint2(2, uvID[objID])];

                        float2 uv = uv0 + beta * (uv1 - uv0) + gamma * (uv2 - uv0);

                        uint3 ind =
                            make_uint3(roundf(float(sz.x - 1) * fabs(uv.x)), roundf(float(sz.y - 1) * fabs(uv.y)), ID);

                        solid = maskdata[ind];
                        if (!solid) {
                            if (count > 10) {
                                break;
                            }
                            Rx = rnd(prd.seed);
                            Ry = rnd(prd.seed);
                            if (Rx < Ry) {
                                sp.x = Rx;
                                sp.y = Ry;
                            } else {
                                sp.x = Ry;
                                sp.y = Rx;
                            }
                        }
                    }
                }

            } else if (primitive_type[objID] == 2) {  // Disk

                // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

                // first map sample point to rectangle [-1,1] [-1,1]
                sp.x = -1.f + 2.f * Rx;
                sp.y = -1.f + 2.f * Ry;

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
                        if (sp.y != 0.f) {  // avoid division by zero at origin
                            p = 6.f - sp.x / sp.y;
                        } else {
                            p = 0.f;
                        }
                    }
                }
                p *= 0.25f * M_PI;

                // find x,y point on unit disk
                sp.x = r * cos(p);
                sp.y = r * sin(p);
                sp.z = 0.f;

                // calculate disk normal vector (world coordinates)
                float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
                float3 v1 = d_transformPoint(m, make_float3(1, 0, 0));
                float3 v2 = d_transformPoint(m, make_float3(0, 1, 0));

                normal = normalize(cross(v1 - v0, v2 - v0));

            } else if (primitive_type[objID] == 4) {  // Voxel

                float Rz = rnd(prd.seed);
            }

            // translate the ray to the location of the primitive

            float3 ray_origin = d_transformPoint(m, sp);

            // Send a ray toward each source
            for (int rr = 0; rr < Nsources; rr++) {
                // set the ray direction
                float3 ray_direction;
                float ray_magnitude;
                if (source_types[rr] == 0) {  // collimated source
                    ray_direction = normalize(source_positions[rr]);
                    ray_magnitude = RT_DEFAULT_MAX;
                    prd.strength =
                        source_fluxes[rr] / float(launch_dim.x * launch_dim.y) * fabs(dot(normal, ray_direction));
                } else {  // sphere source

                    // sample point on surface of sphere
                    float theta_s = acos_safe(1.f - 2.f * rnd(prd.seed));
                    float phi_s = rnd(prd.seed) * 2.f * M_PI;
                    float3 sphere_point =
                        0.5 * source_widths[rr] *
                        make_float3(sin(theta_s) * cos(phi_s), sin(theta_s) * sin(phi_s), cos(theta_s));

                    ray_direction = sphere_point + source_positions[rr] - ray_origin;

                    ray_magnitude = d_magnitude(ray_direction);
                    ray_direction = normalize(ray_direction);
                    prd.strength = 0.f;
                    uint N = 10;
                    for (uint j = 0; j < N; j++) {
                        for (uint i = 0; i < N; i++) {
                            float theta = acos_safe(1.f - 2.f * (float(i) + 0.5f) / float(N));
                            float phi = (float(j) + 0.5f) * 2.f * M_PI / float(N);
                            float3 light_direction =
                                make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
                            if (dot(light_direction, ray_direction) < 0) {
                                prd.strength +=
                                    source_fluxes[rr] / float(launch_dim.x * launch_dim.y) *
                                    fabs(dot(normal, ray_direction)) * fabs(dot(light_direction, ray_direction)) /
                                    (ray_magnitude * ray_magnitude) / (N * N) * source_widths[rr] * source_widths[rr];
                            }
                        }
                    }
                }

                optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, direct_ray_type, 1e-4, ray_magnitude);

                prd.origin_UUID = UUID;
                prd.periodic_depth = 0;

                if (dot(ray_direction, normal) > 0) {
                    prd.face = 1;
                } else {
                    prd.face = 0;
                }

                rtTrace(top_object, ray, prd);
            }
        }
    }
}

RT_PROGRAM void diffuse_raygen() {
    uint dimx = launch_dim.x * launch_dim.y;
    uint indx = launch_dim.x * launch_index.y + launch_index.x;

    PerRayData prd;
    prd.seed = tea<16>(indx + dimx * launch_index.z, random_seed);

    uint objID = launch_offset + launch_index.z;

    // transformation matrix
    float m[16];
    for (uint i = 0; i < 16; i++) {
        m[i] = transform_matrix[optix::make_uint2(i, objID)];
    }

    float3 sp, normal;
    float area;

    // looping over sub-patches
    for (size_t jj = 0; jj < object_subdivisions[objID].y; jj++) {
        for (size_t ii = 0; ii < object_subdivisions[objID].x; ii++) {
            uint UUID = primitiveID[objID] + jj * object_subdivisions[objID].x + ii;

            // two random samples [0,1]
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);

            if (primitive_type[objID] > 4) {
                printf("objID = %d\n", objID);
                printf("Invalid primitive type in diffuse ray launch.\n");
            }

            if (primitive_type[objID] == 0 || primitive_type[objID] == 3) {  // Patch or Tile

                // calculate rectangle normal vector (world coordinates)
                float3 s0 = make_float3(0, 0, 0);
                float3 s1 = make_float3(1, 0, 0);
                float3 s2 = make_float3(0, 1, 0);
                s0 = d_transformPoint(m, s0);
                s1 = d_transformPoint(m, s1);
                s2 = d_transformPoint(m, s2);
                area = primitive_area[UUID];

                normal = normalize(cross(s1 - s0, s2 - s0));

                // Map sample to rectangle [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + (ii + Rx) / float(object_subdivisions[objID].x);
                sp.y = -0.5f + (jj + Ry) / float(object_subdivisions[objID].y);
                sp.z = 0.f;

                int ID = maskID[objID];
                if (ID >= 0) {  // has texture transparency

                    int2 sz = masksize[ID];
                    uint3 ind;

                    bool solid = false;
                    int count = 0;
                    while (!solid) {
                        count++;

                        float2 uv = make_float2(sp.x + 0.5f, 1.f - sp.y - 0.5f);
                        if (uvID[objID] == -1) {  // does not have custom (u,v) coordinates
                            ind = make_uint3(roundf(float(sz.x - 1) * uv.x), roundf(float(sz.y - 1) * uv.y), ID);
                        } else {  // has custom (u,v) coordinates
                            float2 uvmin = uvdata[make_uint2(3, uvID[objID])];
                            float2 duv;
                            duv.x = uvdata[make_uint2(1, uvID[objID])].x - uvdata[make_uint2(0, uvID[objID])].x;
                            duv.y = uvdata[make_uint2(1, uvID[objID])].y - uvdata[make_uint2(2, uvID[objID])].y;
                            ind = make_uint3(roundf(float(sz.x - 1) * (uvmin.x + uv.x * duv.x)),
                                             roundf(float(sz.y - 1) * (uvmin.y + uv.y * duv.y)), ID);
                        }
                        solid = maskdata[ind];
                        if (!solid) {
                            if (count > 10) {
                                break;
                            }
                            sp.x = -0.5f + (ii + rnd(prd.seed)) / float(object_subdivisions[objID].x);
                            sp.y = -0.5f + (jj + rnd(prd.seed)) / float(object_subdivisions[objID].y);
                        }
                    }
                }

            } else if (primitive_type[objID] == 1) {  // Triangle

                // Map sample to triangle with vertices (0,0,0), (0,1,0), (1,1,0)
                if (Rx < Ry) {
                    sp.x = Rx;
                    sp.y = Ry;
                } else {
                    sp.x = Ry;
                    sp.y = Rx;
                }
                sp.z = 0;

                // triangle area
                area = primitive_area[UUID];

                // calculate triangle normal vector (world coordinates)
                float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
                float3 v1 = d_transformPoint(m, make_float3(0, 1, 0));
                float3 v2 = d_transformPoint(m, make_float3(1, 1, 0));

                normal = normalize(cross(v1 - v0, v2 - v1));

                int ID = maskID[objID];
                if (ID >= 0) {  // has texture transparency

                    int2 sz = masksize[ID];

                    float a = v0.x - v1.x, b = v0.x - v2.x, d = v0.x;
                    float e = v0.y - v1.y, f = v0.y - v2.y, h = v0.y;
                    float i = v0.z - v1.z, j = v0.z - v2.z, l = v0.z;

                    bool solid = false;
                    int count = 0;
                    while (!solid) {
                        count++;

                        float3 R = d_transformPoint(m, sp);

                        float c = R.x, g = R.y, k = R.z;

                        float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
                        float q = g * i - e * k, s = e * j - f * i;

                        float inv_denom = 1.f / (a * m + b * q + c * s);

                        float e1 = d * m - b * n - c * p;
                        float beta = e1 * inv_denom;

                        float r = r = e * l - h * i;
                        float e2 = a * n + d * q + c * r;
                        float gamma = e2 * inv_denom;

                        float2 uv0 = uvdata[make_uint2(0, uvID[objID])];
                        float2 uv1 = uvdata[make_uint2(1, uvID[objID])];
                        float2 uv2 = uvdata[make_uint2(2, uvID[objID])];

                        float2 uv = uv0 + beta * (uv1 - uv0) + gamma * (uv2 - uv0);

                        uint3 ind =
                            make_uint3(roundf(float(sz.x - 1) * fabs(uv.x)), roundf(float(sz.y - 1) * fabs(uv.y)), ID);

                        solid = maskdata[ind];
                        if (!solid) {
                            if (count > 10) {
                                break;
                            }
                            Rx = rnd(prd.seed);
                            Ry = rnd(prd.seed);
                            if (Rx < Ry) {
                                sp.x = Rx;
                                sp.y = Ry;
                            } else {
                                sp.x = Ry;
                                sp.y = Rx;
                            }
                        }
                    }
                }

            } else if (primitive_type[objID] == 2) {  // Disk

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
                        if (sp.y != 0.f) {  // avoid division by zero at origin
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

                area = primitive_area[UUID];

                // calculate disk normal vector (world coordinates)
                float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
                float3 v1 = d_transformPoint(m, make_float3(1, 0, 0));
                float3 v2 = d_transformPoint(m, make_float3(0, 1, 0));
                normal = normalize(cross(v1 - v0, v2 - v0));

            } else if (primitive_type[objID] == 4) {  // Voxel

                // Map sample to cube [-0.5,0.5] [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + Rx;
                sp.y = -0.5f + Ry;
                sp.z = -0.5f + rnd(prd.seed);
            }

            // Choose random hemispherical direction - map samples to hemisphere (from Suffern (2007) "Ray tracing fom
            // the ground up" Chap. 6)

            float Rt;
            float Rp;

            Rt = (launch_index.x + rnd(prd.seed)) / float(launch_dim.x);
            Rp = (launch_index.y + rnd(prd.seed)) / float(launch_dim.y);

            float t;
            if (primitive_type[objID] == 4) {  // voxel
                t = acos_safe(1.f - Rt);
            } else {  // other
                t = asin_safe(sqrtf(Rt));
            }
            float p = 2.f * M_PI * Rp;

            float3 ray_direction;
            ray_direction.x = sin(t) * cos(p);
            ray_direction.y = sin(t) * sin(p);
            ray_direction.z = cos(t);

            float3 ray_origin;
            optix::Ray ray;

            if (primitive_type[objID] == 4) {  // voxel

                prd.strength = 0.5f / float(dimx);
                prd.area = 0.1;
                prd.origin_UUID = UUID;
                prd.face = 0;
                prd.periodic_depth = 0;

                ray_origin = d_transformPoint(m, sp);

                ray = optix::make_Ray(ray_origin, ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);
                rtTrace(top_object, ray, prd);

                ray = optix::make_Ray(ray_origin, -ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);
                rtTrace(top_object, ray, prd);

                if (UUID == 0) {
                    // rtPrintf("%f,%f,%f;\n",ray_origin.x,ray_origin.y,ray_origin.z);
                    // rtPrintf("%f,%f,%f;\n",ray_direction.x,ray_direction.y,ray_direction.z);
                    // rtPrintf("normal: (%f,%f,%f)\n",normal.x,normal.y,normal.z);
                }

            } else {  // not a voxel

                ray_direction = d_rotatePoint(ray_direction, acos_safe(normal.z), atan2(normal.y, normal.x));

                prd.strength = 1.f / float(dimx);

                prd.area = area;
                prd.origin_UUID = UUID;
                prd.periodic_depth = 0;

                // ---- "top" surface launch -------
                ray_origin = d_transformPoint(m, sp);
                ray = optix::make_Ray(ray_origin, ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);

                prd.face = 1;

                rtTrace(top_object, ray, prd);

                // ---- "bottom" surface launch -------
                if (twosided_flag[objID] == 1) {
                    // ray_origin = d_transformPoint(m,sp);
                    ray_direction = -ray_direction;
                    ray = optix::make_Ray(ray_origin, ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);

                    prd.face = 0;

                    rtTrace(top_object, ray, prd);
                }
            }
        }
    }
}

RT_PROGRAM void direct_raygen_MCRT() {
    uint dimx = launch_dim.x * launch_dim.y;
    uint indx = launch_dim.x * launch_index.y + launch_index.x;

    PerRayData_MCRT prd;
    prd.seed = tea<16>(indx + dimx * launch_index.z, random_seed);

    // Launch rays from each source
    for (int r = 0; r < Nsources; r++) {
        if (source_types[r] == 0) {  // collimated source

            // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

            float3 sp;

            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);

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
                    if (sp.y != 0.f) {  // avoid division by zero at origin
                        p = 6.f - sp.x / sp.y;
                    } else {
                        p = 0.f;
                    }
                }
            }
            p *= 0.25f * M_PI;

            // find x,y point on unit disk
            sp.x = r * cosf(p) * bound_sphere_radius;
            sp.y = r * sinf(p) * bound_sphere_radius;
            sp.z = 0.f;

            float3 ray_origin = bound_sphere_center + sp;

            // set the ray direction
            float3 ray_direction;
            ray_direction = normalize(source_positions[r]);
            prd.strength = source_fluxes[r] / float(launch_dim.x * launch_dim.y) *
                           fabs(dot(make_float3(0, 0, 1), ray_direction)) * M_PI * bound_sphere_radius *
                           bound_sphere_radius;

            // rotate
            ray_origin = d_rotatePoint(ray_origin, acos_safe(ray_direction.z), atan2(ray_direction.y, ray_direction.x));
            // translate
            ray_origin = ray_origin - ray_direction * bound_sphere_radius * 100.f;

            optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, direct_ray_type_MCRT, 1e-4, RT_DEFAULT_MAX);

            prd.scatter_depth = 0;
            prd.origin_UUID = 0;

            rtTrace(top_object, ray, prd);

        } else {  // point source

            uint Nphi = launch_dim.y;
            uint Ntheta = launch_dim.x;

            uint i = launch_index.x;
            uint j = launch_index.y;

            float dt = 1.f / float(Ntheta);
            float dp = 1.f / float(Nphi);

            float Rt = i * dt + rnd(prd.seed) * dt;
            float Rp = j * dp + rnd(prd.seed) * dp;

            float t = acos_safe(1.f - 2.f * Rt);
            float p = 2.f * M_PI * Rp;

            float3 ray_direction;
            ray_direction.x = sin(t) * cos(p);
            ray_direction.y = sin(t) * sin(p);
            ray_direction.z = cos(t);

            float3 ray_origin = source_positions[r] + 1e-3 * ray_direction;

            optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, direct_ray_type_MCRT, 1e-4, RT_DEFAULT_MAX);

            prd.strength = source_fluxes[r] / float(launch_dim.x * launch_dim.y) * 2.f;

            prd.scatter_depth = 0;
            prd.origin_UUID = 0;

            rtTrace(top_object, ray, prd);
        }
    }
}

RT_PROGRAM void diffuse_raygen_MCRT() {
    uint dimx = launch_dim.x * launch_dim.y;
    uint indx = launch_dim.x * launch_index.y + launch_index.x;

    PerRayData_MCRT prd;
    prd.seed = tea<16>(indx + dimx * launch_index.z, random_seed);

    uint Nphi = launch_dim.y;
    uint Ntheta = launch_dim.x;

    // method #2//
    //  float Rx = rnd(prd.seed);
    //  float Ry = rnd(prd.seed);

    // //find x,y point on upper boundary
    // float3 sp;
    // sp.x = bound_box_x.x+Rx*(bound_box_x.y-bound_box_x.x);
    // sp.y = bound_box_y.x+Ry*(bound_box_y.y-bound_box_y.x);
    // sp.z = bound_box_z.y;

    // //Choose random hemispherical direction - map samples to hemisphere (from Suffern (2007) "Ray tracing fom the
    // ground up" Chap. 6)

    // uint i= launch_index.x;
    // uint j = launch_index.y;

    // float dt = 1.f/float(Ntheta);
    // float dp = 1.f/float(Nphi);

    // float Rt = i*dt + rnd(prd.seed)*dt;
    // float Rp = j*dp + rnd(prd.seed)*dp;

    // float theta = sinf(sqrtf(Rt));
    // //float theta = acos_safe(1.f-2.f*Rt);
    // float phi = 2.f*M_PI*Rp;

    // float3 ray_direction;
    // ray_direction.x = -sin(theta)*cos(phi);
    // ray_direction.y = -sin(theta)*sin(phi);
    // ray_direction.z = -cos(theta);

    // float3 ray_origin = sp;
    // method #2//

    uint i = launch_index.x;
    uint j = launch_index.y;

    float dt = 1.f / float(Ntheta);
    float dp = 1.f / float(Nphi);

    float Rt = i * dt + rnd(prd.seed) * dt;
    float Rp = j * dp + rnd(prd.seed) * dp;

    float theta1 = acos_safe(1.f - 2.f * Rt);
    float phi1 = 2.f * M_PI * Rp;

    float theta2 = acos_safe(1.f - 2.f * rnd(prd.seed));
    float phi2 = 2.f * M_PI * rnd(prd.seed);

    float3 ray_origin, ray_end;

    ray_origin.x = bound_sphere_radius * sin(theta1) * cos(phi1) + bound_sphere_center.x;
    ray_origin.y = bound_sphere_radius * sin(theta1) * sin(phi1) + bound_sphere_center.y;
    ray_origin.z = bound_sphere_radius * cos(theta1) + bound_sphere_center.z;

    ray_end.x = bound_sphere_radius * sin(theta2) * cos(phi2) + bound_sphere_center.x;
    ray_end.y = bound_sphere_radius * sin(theta2) * sin(phi2) + bound_sphere_center.y;
    ray_end.z = bound_sphere_radius * cos(theta2) + bound_sphere_center.z;

    float3 ray_direction;
    ray_direction = ray_end - ray_origin;
    ray_direction = normalize(ray_direction);

    // ---- launch -------

    optix::Ray ray;

    ray = optix::make_Ray(ray_origin, ray_direction, diffuse_ray_type_MCRT, 1e-4, RT_DEFAULT_MAX);

    prd.scatter_depth = 0;
    prd.origin_UUID = 0;
    prd.strength = diffuseFlux / float(Ntheta * Nphi) * 4.f * M_PI * bound_sphere_radius * bound_sphere_radius;
    // method #2//
    // prd.strength = diffuseFlux/float(Ntheta*Nphi)*(bound_box_x.y-bound_box_x.x)*(bound_box_y.y-bound_box_y.x);
    // method #2//

    rtTrace(top_object, ray, prd);

    // method #2//
    //  ray_origin.z = bound_box_z.x;
    //  ray = optix::make_Ray(ray_origin, -ray_direction, diffuse_ray_type_MCRT, 1e-4, RT_DEFAULT_MAX);
    //  rtTrace( top_object, ray, prd);
    // method #2//
}

RT_PROGRAM void emission_raygen_MCRT() {
    uint dimx = launch_dim.x * launch_dim.y;
    uint indx = launch_dim.x * launch_index.y + launch_index.x;

    PerRayData_MCRT prd;
    prd.seed = tea<16>(indx + dimx * launch_index.z, random_seed);

    // Primitive UUID for this launch
    uint UUID = launch_offset + launch_index.z;

    // transformation matrix
    float m[16];
    for (uint i = 0; i < 16; i++) {
        m[i] = transform_matrix[optix::make_uint2(i, UUID)];
    }

    float Rx = rnd(prd.seed);
    float Ry = rnd(prd.seed);

    float3 sp, normal;

    if (primitive_type[UUID] == 0 || primitive_type[UUID] == 3) {  // Patch or AlphaMask

        // calculate rectangle normal vector (world coordinates)
        float3 s0 = make_float3(0, 0, 0);
        float3 s1 = make_float3(1, 0, 0);
        float3 s2 = make_float3(0, 1, 0);
        s0 = d_transformPoint(m, s0);
        s1 = d_transformPoint(m, s1);
        s2 = d_transformPoint(m, s2);

        normal = normalize(cross(s1 - s0, s2 - s0));

        // Map sample to rectangle [-0.5,0.5] [-0.5,0.5]
        sp.x = -0.5f + Rx;
        sp.y = -0.5f + Ry;
        sp.z = 0.f;

        int ID = maskID[UUID];
        if (ID >= 0) {  // has texture transparency

            int2 sz = masksize[ID];
            uint3 ind;

            bool solid = false;
            int count = 0;
            while (!solid) {
                count++;

                float2 uv = make_float2(sp.x + 0.5f, 1.f - sp.y - 0.5f);
                if (uvID[UUID] == -1) {  // does not have custom (u,v) coordinates
                    ind = make_uint3(roundf(float(sz.x - 1) * uv.x), roundf(float(sz.y - 1) * uv.y), ID);
                } else {  // has custom (u,v) coordinates
                    float2 uvmin = uvdata[make_uint2(3, uvID[UUID])];
                    float2 duv;
                    duv.x = uvdata[make_uint2(1, uvID[UUID])].x - uvdata[make_uint2(0, uvID[UUID])].x;
                    duv.y = uvdata[make_uint2(1, uvID[UUID])].y - uvdata[make_uint2(2, uvID[UUID])].y;
                    ind = make_uint3(roundf(float(sz.x - 1) * (uvmin.x + uv.x * duv.x)),
                                     roundf(float(sz.y - 1) * (uvmin.y + uv.y * duv.y)), ID);
                }
                solid = maskdata[ind];
                if (!solid) {
                    if (count > 10) {
                        break;
                    }
                    sp.x = -0.5f + rnd(prd.seed);
                    sp.y = -0.5f + rnd(prd.seed);
                }
            }
        }

    } else if (primitive_type[UUID] == 1) {  // Triangle

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
        float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
        float3 v1 = d_transformPoint(m, make_float3(0, 1, 0));
        float3 v2 = d_transformPoint(m, make_float3(1, 1, 0));

        normal = normalize(cross(v1 - v0, v2 - v1));

        int ID = maskID[UUID];
        if (ID >= 0) {  // has texture transparency

            int2 sz = masksize[ID];

            float a = v0.x - v1.x, b = v0.x - v2.x, d = v0.x;
            float e = v0.y - v1.y, f = v0.y - v2.y, h = v0.y;
            float i = v0.z - v1.z, j = v0.z - v2.z, l = v0.z;

            bool solid = false;
            int count = 0;
            while (!solid) {
                count++;

                float3 R = d_transformPoint(m, sp);

                float c = R.x, g = R.y, k = R.z;

                float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
                float q = g * i - e * k, s = e * j - f * i;

                float inv_denom = 1.f / (a * m + b * q + c * s);

                float e1 = d * m - b * n - c * p;
                float beta = e1 * inv_denom;

                float r = r = e * l - h * i;
                float e2 = a * n + d * q + c * r;
                float gamma = e2 * inv_denom;

                float2 uv0 = uvdata[make_uint2(0, uvID[UUID])];
                float2 uv1 = uvdata[make_uint2(1, uvID[UUID])];
                float2 uv2 = uvdata[make_uint2(2, uvID[UUID])];

                float2 uv = uv0 + beta * (uv1 - uv0) + gamma * (uv2 - uv0);

                uint3 ind = make_uint3(roundf(float(sz.x - 1) * fabs(uv.x)), roundf(float(sz.y - 1) * fabs(uv.y)), ID);

                solid = maskdata[ind];
                if (!solid) {
                    if (count > 10) {
                        break;
                    }
                    Rx = rnd(prd.seed);
                    Ry = rnd(prd.seed);
                    if (Rx < Ry) {
                        sp.x = Rx;
                        sp.y = Ry;
                    } else {
                        sp.x = Ry;
                        sp.y = Rx;
                    }
                }
            }
        }

    } else if (primitive_type[UUID] == 2) {  // Disk

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
                if (sp.y != 0.f) {  // avoid division by zero at origin
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
        float3 v0 = d_transformPoint(m, make_float3(0, 0, 0));
        float3 v1 = d_transformPoint(m, make_float3(1, 0, 0));
        float3 v2 = d_transformPoint(m, make_float3(0, 1, 0));
        normal = normalize(cross(v1 - v0, v2 - v0));

    } else if (primitive_type[UUID] == 4) {  // Voxel

        float Rz = rnd(prd.seed);

        normal = make_float3(0, 0, 1);
    }

    float area = primitive_area[UUID];

    // Choose random hemispherical direction - map samples to hemisphere (from Suffern (2007) "Ray tracing fom the
    // ground up" Chap. 6)

    uint Nphi = launch_dim.y;
    uint Ntheta = launch_dim.x;

    uint i = launch_index.x;
    uint j = launch_index.y;

    float dt = 1.f / float(Ntheta);
    float dp = 1.f / float(Nphi);

    float Rt;
    float Rp;

    // Rt = rnd(prd.seed);
    // Rp = rnd(prd.seed);
    Rt = i * dt + rnd(prd.seed) * dt;
    Rp = j * dp + rnd(prd.seed) * dp;

    float t;
    if (primitive_type[UUID] == 4) {  // voxel
        t = acos_safe(1.f - Rt);
    } else {  // other
        t = asin_safe(sqrtf(Rt));
    }
    float p = 2.f * M_PI * Rp;

    float3 ray_direction;
    ray_direction.x = sin(t) * cos(p);
    ray_direction.y = sin(t) * sin(p);
    ray_direction.z = cos(t);

    if (primitive_type[UUID] == 4) {  // voxel
        ray_direction = ray_direction;
    } else {  // other
        ray_direction = d_rotatePoint(ray_direction, acos_safe(normal.z), atan2(normal.y, normal.x));
    }

    float3 ray_origin;
    optix::Ray ray;

    // ---- "top" surface launch -------
    ray_origin = d_transformPoint(m, sp);
    ray = optix::make_Ray(ray_origin, ray_direction, emission_ray_type_MCRT, 1e-4, RT_DEFAULT_MAX);

    prd.scatter_depth = 0;
    prd.origin_UUID = UUID;
    prd.strength = radiation_out_top[UUID] / float(dimx) * primitive_area[UUID];

    rtTrace(top_object, ray, prd);

    // ---- "bottom" surface launch -------
    if (twosided_flag[UUID] == 1) {
        ray_origin = d_transformPoint(m, sp);
        ray = optix::make_Ray(ray_origin, -ray_direction, emission_ray_type_MCRT, 1e-4, RT_DEFAULT_MAX);

        prd.strength = radiation_out_bottom[UUID] / float(dimx) * primitive_area[UUID];

        rtTrace(top_object, ray, prd);
    }
}
